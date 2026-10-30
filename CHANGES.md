# Changes in this revision

**Root causes found**
1. `postgresql://` URLs silently failed under SQLAlchemy 2.1 (default driver became psycopg v3; only psycopg2 installed) -> every service/ETL quietly fell back to private SQLite. Fixed: URL normalised to `postgresql+psycopg2://`, `ALLOW_SQLITE_FALLBACK=false` in containers, `<2.2` pin.
2. `PROMPT_TEMPLATE.format()` crashed on unescaped JSON braces, so the OpenAI/Ollama path always fell back to the mock. Fixed.
3. Gateway waited ~1 s for the worker; any real LLM call lost the race. Now waits `LLM_JOB_WAIT_SECONDS` (25 s) and worker publishes failures.
4. Compose never passed `OPENAI_API_KEY`/`LLM_PROVIDER` through (worker hard-coded `local`; gateway runs catalog QA but had no key).
5. ETL deleted all books *before* inserting; 6 authors exceed VARCHAR(255) so Postgres would have failed mid-run leaving an empty table. ETL is now an idempotent upsert with clipping, `--reset --yes`, `--dry-run`, pre-flight and post-run verification.
6. Mock explanations and ranker badges fabricated content ("rich narrative depth", "Quick read (0 pages)"). Now metadata-only and honest.
7. In local mode the in-memory vector store was empty on API start, so `/recommend` returned the first 20 rows. Now rebuilt from the DB at startup.

**Added**: `etl` compose service, `.dockerignore`, `.env.example`, Redis response cache, gateway `/catalog/ask` + `/stats`, retrieval `/health` with `points_count`, `scripts/eval_pipeline.py`, `tests/test_rag_pipeline.py` (19 new tests).
