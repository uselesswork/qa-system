"""No-key MySQL, Redis and Qdrant integration smoke for CI and local Compose."""

import os
from uuid import uuid4

from redis import Redis
from sqlalchemy import inspect, text
from qdrant_client import models

from src.db.session import create_database_engine
from src.vectorstores.qdrant_store import QdrantVectorStore


def main() -> None:
    engine = create_database_engine()
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        tables = set(inspect(engine).get_table_names())
        required = {"users", "knowledge_bases", "documents", "chunks", "ingestion_tasks"}
        assert required <= tables, f"Missing MySQL tables: {required - tables}"
    finally:
        engine.dispose()

    redis = Redis.from_url(os.environ["REDIS_URL"])
    try:
        assert redis.ping()
    finally:
        redis.close()

    store = QdrantVectorStore.remote(os.environ["QDRANT_URL"])
    collection = f"qa_smoke_{uuid4().hex}"
    try:
        store.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        store.client.upsert(
            collection_name=collection,
            points=[models.PointStruct(id=str(uuid4()), vector=[1.0, 0.0, 0.0],
                                       payload={"check": "stage15"})],
            wait=True,
        )
        assert store.client.count(collection_name=collection, exact=True).count == 1
    finally:
        if store.client.collection_exists(collection):
            store.client.delete_collection(collection)
        store.close()
    print("MySQL, Redis and Qdrant smoke passed")


if __name__ == "__main__":
    main()
