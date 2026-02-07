import pandas as pd
import psycopg2
import os

# DB Config
DB_NAME = "movies_db"
DB_USER = "admin"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"

DATA_PATH = "../data/ml-latest-small/movies.csv"

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def create_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            movie_id INT PRIMARY KEY,
            title TEXT,
            genres TEXT,
            poster_url TEXT
        );
    """)

def ingest_data():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}. Run download_data.py first.")
        return

    print("Reading CSV...")
    df = pd.read_csv(DATA_PATH)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    print("Creating table...")
    create_table(cur)
    
    print("Inserting data...")
    # This is a naive insert loop, for 100k it's fine. For larger data, use copy_from.
    for _, row in df.iterrows():
        cur.execute(
            "INSERT INTO movies (movie_id, title, genres) VALUES (%s, %s, %s) ON CONFLICT (movie_id) DO NOTHING",
            (row['movieId'], row['title'], row['genres'])
        )
    
    conn.commit()
    cur.close()
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
