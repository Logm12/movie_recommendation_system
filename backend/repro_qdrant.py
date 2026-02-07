from qdrant_client import QdrantClient

try:
    client = QdrantClient(host="vdt_qdrant", port=6333)
    print(f"Client type: {type(client)}")
    print(f"Has search? {hasattr(client, 'search')}")
    print(f"Dir: {dir(client)}")
except Exception as e:
    print(f"Error: {e}")
