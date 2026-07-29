import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.logging import get_logger
from core.database import init_db, SessionLocal, BookModel
from core.vector_db import vector_store

logger = get_logger("etl-pipeline")

BOOKS_CSV = Path("data/raw/books.csv")
DEV_LIMIT = 1000  # Set to None to ingest the full dataset


def load_books_from_csv(limit: int | None = None) -> List[Dict[str, Any]]:
    """
    Load the dataset from CSV and convert it into the internal Book schema.
    """

    df = pd.read_csv(BOOKS_CSV)

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    logger.info(f"CSV Columns: {list(df.columns)}")

    if limit is not None:
        df = df.head(limit)

    books = []

    for _, row in df.iterrows():

        # ---------- Basic Fields ----------

        title = str(row["Title"]).strip()

        author = row.get("Author")
        if pd.isna(author):
            author = "Unknown"
        author = str(author).strip()

        rating = row.get("AvgRating")
        if pd.isna(rating):
            rating = 0.0
        rating = float(rating)

        publication_year = row.get("Publication Year")
        if pd.isna(publication_year):
            publication_year = 0
        publication_year = int(publication_year)

        # Dataset does not contain page counts
        page_count = 0

        cover = row.get("Image-URL")
        if pd.isna(cover):
            cover = None

        # Goodreads-style datasets often don't contain descriptions.
        # Build a synthetic document for embeddings.
        description = f"""
Title: {title}

Author: {author}

This book was written by {author}.

It has an average reader rating of {rating}.

It was originally published in {publication_year}.
"""

        books.append(
            {
                "id": int(row["book_id"]),
                "title": title,
                "author": author,
                "genres": [],
                "description": description,
                "rating": rating,
                "page_count": page_count,
                "publication_year": publication_year,
                "cover_image_url": cover,
            }
        )

    logger.info(f"Loaded {len(books)} books from CSV.")

    return books


def run_etl():
    """
    Executes the BookWise AI ETL pipeline.

    1. Initialize database
    2. Clear existing books
    3. Insert Goodreads metadata
    4. Generate embeddings
    5. Upload vectors to Qdrant
    """

    logger.info("Starting BookWise AI ETL Pipeline...")

    # Initialize database
    init_db()

    db = SessionLocal()

    books = load_books_from_csv(limit=DEV_LIMIT)

    try:

        # Clear old records
        db.query(BookModel).delete()
        db.commit()

        # Insert books
        logger.info(f"Ingesting {len(books)} books into database...")

        for item in books:

            book = BookModel(
                id=item["id"],
                title=item["title"],
                author=item["author"],
                genres=item["genres"],
                description=item["description"],
                rating=item["rating"],
                page_count=item["page_count"],
                publication_year=item["publication_year"],
                cover_image_url=item["cover_image_url"],
            )

            db.add(book)

        db.commit()

        logger.info(
            f"Successfully inserted {len(books)} books into the database."
        )

        # Generate embeddings and populate vector DB
        logger.info("Generating embeddings and indexing Vector DB...")

        vector_store.upsert_books(books)

        logger.info("Vector database indexing completed.")

    except Exception as e:

        db.rollback()

        logger.exception(f"ETL failed: {e}")

        raise

    finally:

        db.close()

    logger.info("ETL pipeline completed successfully.")


if __name__ == "__main__":
    run_etl()