"""BookWise ETL: Books.csv -> relational DB (idempotent upsert) -> vector index.

Usage:
    python etl.py                  # first DEV_LIMIT rows, upsert (never deletes)
    python etl.py --full           # whole CSV
    python etl.py --limit 200      # first 200 rows
    python etl.py --dry-run        # parse + validate only, touch nothing
    python etl.py --reset --yes    # delete ALL books first (requires --yes)
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import BookModel, SessionLocal, init_db
from core.vector_db import vector_store
from shared.logging import get_logger

logger = get_logger("etl-pipeline")

BOOKS_CSV = Path(__file__).resolve().parent / "data" / "raw" / "Books.csv"
DEV_LIMIT = 1000  # default sample size; use --full for the whole dataset
REQUIRED_COLUMNS = ("book_id", "Title")

# Column widths in core.database.BookModel (Postgres enforces these, SQLite does not).
MAX_TITLE, MAX_AUTHOR, MAX_URL = 255, 255, 500


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _clean_str(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def load_books_from_csv(limit: int | None = None) -> list[dict[str, Any]]:
    """Read the CSV and return validated book dicts.

    Rows with an invalid/duplicate id or a blank title are skipped; text is clipped to column widths and
    ratings are clamped to 0..5, so a single bad row can't abort an ingest halfway through.
    """
    df = pd.read_csv(BOOKS_CSV)
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{BOOKS_CSV} is missing required columns: {missing}. Found: {list(df.columns)}")

    if limit is not None:
        df = df.head(limit)

    books: list[dict[str, Any]] = []
    seen: set[int] = set()
    skipped = 0

    for _, row in df.iterrows():
        try:
            book_id = int(row["book_id"])
        except (TypeError, ValueError):
            skipped += 1
            continue

        title = _clean_str(row.get("Title"))
        if not title or book_id in seen:
            skipped += 1
            continue
        seen.add(book_id)

        author = _clean_str(row.get("Author")) or "Unknown"

        rating = pd.to_numeric(row.get("AvgRating"), errors="coerce")
        rating = 0.0 if pd.isna(rating) else max(0.0, min(5.0, float(rating)))

        year = pd.to_numeric(row.get("Publication Year"), errors="coerce")
        year = 0 if pd.isna(year) else int(year)

        cover = _clean_str(row.get("Image-URL")) or None

        # The dataset has no descriptions: build the embedding text only from facts we actually have.
        facts = [f"Title: {title}", f"Author: {author}"]
        if rating:
            facts.append(f"Average reader rating: {rating}")
        if year > 0:
            facts.append(f"First published in {year}")
        description = ". ".join(facts) + "."

        books.append({
            "id": book_id,
            "title": _clip(title, MAX_TITLE),
            "author": _clip(author, MAX_AUTHOR),
            "genres": [],
            "description": description,
            "rating": rating,
            "page_count": 0,  # dataset has no page counts; 0 means "unknown"
            "publication_year": year,
            "cover_image_url": _clip(cover, MAX_URL) if cover else None,
        })

    logger.info(f"Loaded {len(books)} books from CSV ({skipped} rows skipped as invalid/duplicate).")
    return books


def run_etl(limit: int | None = DEV_LIMIT, reset: bool = False, dry_run: bool = False) -> list[dict[str, Any]]:
    """Idempotent ingest: validate CSV -> upsert rows -> index vectors -> verify.

    Existing rows are updated in place; nothing is deleted unless reset=True, and the CSV is fully parsed
    *before* anything is deleted so a bad file can never leave an empty table.
    """
    logger.info("Starting BookWise AI ETL Pipeline...")

    if not Path(BOOKS_CSV).exists():
        raise FileNotFoundError(f"Books CSV not found at {BOOKS_CSV}")

    books = load_books_from_csv(limit=limit)
    if not books:
        raise ValueError("CSV produced zero valid books; refusing to continue.")

    if dry_run:
        logger.info(f"Dry run: {len(books)} books validated. No changes made.")
        return books

    init_db()
    db = SessionLocal()
    try:
        if reset:
            logger.warning("--reset: deleting ALL existing books.")
            db.query(BookModel).delete()
            db.commit()

        logger.info(f"Upserting {len(books)} books into the database...")
        for item in books:
            db.merge(BookModel(
                id=item["id"], title=item["title"], author=item["author"], genres=item["genres"],
                description=item["description"], rating=item["rating"], page_count=item["page_count"],
                publication_year=item["publication_year"], cover_image_url=item["cover_image_url"],
            ))
        db.commit()

        ids = [b["id"] for b in books]
        stored = 0
        for i in range(0, len(ids), 500):  # chunked: keeps the IN() list small
            stored += db.query(BookModel).filter(BookModel.id.in_(ids[i:i + 500])).count()
        if stored != len(books):
            raise RuntimeError(f"Post-run check failed: expected {len(books)} rows in DB, found {stored}.")

        logger.info("Generating embeddings and indexing Vector DB...")
        vector_store.upsert_books(books)
        indexed = vector_store.points_count()
        if indexed < len(books):
            logger.warning(f"Vector index holds {indexed} points but {len(books)} books were loaded.")
        logger.info(f"Vector database indexing completed ({indexed} points).")
    except Exception as e:
        db.rollback()
        logger.exception(f"ETL failed: {e}")
        raise
    finally:
        db.close()

    logger.info("ETL pipeline completed successfully.")
    return books


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load Books.csv into the BookWise database and vector index.")
    parser.add_argument("--limit", type=int, default=DEV_LIMIT, help=f"rows to ingest (default {DEV_LIMIT})")
    parser.add_argument("--full", action="store_true", help="ingest the entire CSV (ignores --limit)")
    parser.add_argument("--dry-run", action="store_true", help="validate the CSV and exit without writing")
    parser.add_argument("--reset", action="store_true", help="delete ALL existing books before loading")
    parser.add_argument("--yes", action="store_true", help="confirm destructive --reset")
    args = parser.parse_args(argv)

    if args.reset and not args.yes:
        parser.error("--reset deletes every book; re-run with --reset --yes to confirm.")

    run_etl(limit=None if args.full else args.limit, reset=args.reset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
