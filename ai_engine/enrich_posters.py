import pandas as pd
import psycopg2
import requests
import os
import time

# Configurations
API_KEY = "09ad8ace66eec34302943272db0e8d2c"
LINKS_PATH = "../data/ml-latest-small/links.csv"
DB_NAME = "movies_db"
DB_USER = "admin"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )

def enrich_posters():
    print("Loading links.csv...")
    try:
        links_df = pd.read_csv(LINKS_PATH)
    except FileNotFoundError:
        print(f"Error: {LINKS_PATH} not found. Please run download_data.py first.")
        return

    # Create a mapping of movieId -> tmdbId
    # Drop rows with NaN tmdbId and convert to int
    links_df = links_df.dropna(subset=['tmdbId'])
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)
    movie_to_tmdb = dict(zip(links_df['movieId'], links_df['tmdbId']))

    print(f"Loaded {len(movie_to_tmdb)} TMDB mappings.")

    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Get Priority IDs from Recommendation Engine
    priority_ids = []
    try:
        print("Fetching recommendations for User 1 to prioritize...")
        rec_resp = requests.get("http://localhost:8000/recommend/1")
        if rec_resp.status_code == 200:
            recs = rec_resp.json().get('recommendations', [])
            priority_ids = [m['id'] for m in recs]
            print(f"Prioritizing {len(priority_ids)} movies: {priority_ids}")
    except Exception as e:
        print(f"Could not fetch priorities: {e}")

    # 2. Get all other movies
    cur.execute("SELECT movie_id FROM movies")
    all_rows = cur.fetchall()
    all_ids = [r[0] for r in all_rows]
    
    # Sort: Priority IDs first
    # Create a set for O(1) lookup
    priority_set = set(priority_ids)
    sorted_ids = priority_ids + [mid for mid in all_ids if mid not in priority_set]
    
    print(f"Total movies to check: {len(sorted_ids)}")
    
    updated_count = 0
    errors = 0
    
    for i, movie_id in enumerate(sorted_ids):
        # Optional: Skip if already has a TMDB URL (check DB value?)
        # For now, just overwrite
        
        tmdb_id = movie_to_tmdb.get(movie_id)
        if not tmdb_id:
            continue
            
        try:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
            resp = requests.get(url, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                poster_path = data.get('poster_path')
                
                if poster_path:
                    full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    
                    cur.execute(
                        "UPDATE movies SET poster_url = %s WHERE movie_id = %s",
                        (full_poster_url, movie_id)
                    )
                    updated_count += 1
                    
                    # Commit frequently for the priority batch
                    if i < 20 or updated_count % 10 == 0:
                        conn.commit()
                        if i < 20:
                             print(f"Updated priority movie {movie_id}")
                        else:
                             print(f"Updated {updated_count} posters...")
            
            elif resp.status_code == 429:
                print("Rate limit. Sleeping 5s...")
                time.sleep(5)

        except Exception as e:
            print(f"Error {movie_id}: {e}")
            errors += 1

        time.sleep(0.05) 
        
        # Stop after 100 for this run to be fast
        if updated_count >= 100: 
             break

    conn.commit()
    cur.close()
    conn.close()
    print(f"Finished. Updated {updated_count} posters.")

if __name__ == "__main__":
    enrich_posters()
