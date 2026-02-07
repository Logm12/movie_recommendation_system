import os
import sys
import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from loguru import logger
import tqdm

# Add backend to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
try:
    from config import get_settings
except ImportError:
    # Fallback or manual config
    logger.warning("Could not import config. Using defaults.")
    class Settings:
        postgres_host="localhost"
        postgres_user="postgres"
        postgres_password="password"
        postgres_db="movie_rec"
        postgres_port=5432
    get_settings = lambda: Settings()

def index_content():
    settings = get_settings()
    
    # 1. Connect to DB
    logger.info("Connecting to Database...")
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db,
            port=settings.postgres_port
        )
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return

    cur = conn.cursor()
    cur.execute("SELECT movie_id, title, genres FROM movies")
    rows = cur.fetchall()
    logger.info(f"Fetched {len(rows)} movies.")
    cur.close()
    conn.close()

    # 2. Init Model
    logger.info("Loading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Init Qdrant
    logger.info("Connecting to Qdrant...")
    client = QdrantClient("localhost", port=6333)
    collection_name = "movies_content"
    embedding_dim = 384 # all-MiniLM-L6-v2 dimension
    
    # Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    
    # 4. Embed and Index
    points = []
    logger.info("Embedding and Indexing...")
    
    for row in tqdm.tqdm(rows):
        movie_id, title, genres = row
        # Content String
        text = f"{title}. Genres: {genres.replace('|', ', ')}"
        
        vector = model.encode(text).tolist()
        
        points.append(PointStruct(
            id=movie_id, 
            vector=vector,
            payload={
                "movie_id": movie_id,
                "title": title,
                "genres": genres
            }
        ))
        
        if len(points) >= 100:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []
            
    # Final batch
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    logger.success(f"Successfully indexed {len(rows)} movies to '{collection_name}' collection.")

if __name__ == "__main__":
    index_content()
