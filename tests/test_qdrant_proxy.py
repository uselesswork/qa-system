from src.vectorstores import qdrant_store


def test_local_qdrant_bypasses_system_proxy(monkeypatch):
    calls = []
    monkeypatch.setattr(qdrant_store, "QdrantClient", lambda **kwargs: calls.append(kwargs))

    qdrant_store.QdrantVectorStore.remote("http://127.0.0.1:6333")
    qdrant_store.QdrantVectorStore.remote("http://localhost:6333")
    qdrant_store.QdrantVectorStore.remote("http://[::1]:6333")
    qdrant_store.QdrantVectorStore.remote("https://qdrant.example.com")

    assert all(call["trust_env"] is False for call in calls[:3])
    assert "trust_env" not in calls[3]
