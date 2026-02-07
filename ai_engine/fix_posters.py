"""
Script to check and fix movie posters using TMDB API.
"""
import requests
import psycopg2
import time

# Configuration
TMDB_API_KEY = "a32b6bae6da4ce91bb1e4086c85be06e"  # Free API key
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "movies_db",
    "user": "postgres",
    "password": "postgres"
}

def get_tmdb_poster(title: str, year: str = None) -> str | None:
    """Search TMDB for a movie and return its poster URL."""
    # Clean title (remove year if present)
    clean_title = title.split('(')[0].strip()
    
    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
        "language": "en-US",
        "page": 1
    }
    
    if year:
        params["year"] = year
    
    try:
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        if results and results[0].get("poster_path"):
            return TMDB_IMAGE_BASE + results[0]["poster_path"]
        return None
    except Exception as e:
        print(f"  Error fetching {title}: {e}")
        return None

def check_and_fix_posters():
    """Check all recommended movies and fix missing/broken posters."""
    # First get recommendations to see which movies need fixing
    print("Fetching recommendations from API...")
    
    try:
        resp = requests.get("http://localhost:8000/recommend/1")
        movies = resp.json()["recommendations"]
    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        return
    
    print(f"\nFound {len(movies)} recommended movies:")
    print("-" * 80)
    
    to_fix = []
    for m in movies:
        has_poster = bool(m.get("poster_url") and "tmdb" in m.get("poster_url", ""))
        status = "[OK]" if has_poster else "[MISSING]"
        print(f"{status} {m['id']:5} | {m['title'][:50]}")
        if not has_poster:
            to_fix.append(m)
    
    if not to_fix:
        print("\nAll posters are correct!")
        return
    
    print(f"\n{len(to_fix)} movies need poster fixes. Fetching from TMDB...")
    print("-" * 80)
    
    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    fixed = 0
    for movie in to_fix:
        movie_id = movie["id"]
        title = movie["title"]
        
        # Extract year from title if present
        year = None
        if "(" in title and ")" in title:
            year_str = title.split("(")[-1].split(")")[0]
            if year_str.isdigit() and len(year_str) == 4:
                year = year_str
        
        print(f"  Searching: {title}...", end=" ")
        poster_url = get_tmdb_poster(title, year)
        
        if poster_url:
            cursor.execute(
                "UPDATE movies SET poster_url = %s WHERE id = %s",
                (poster_url, movie_id)
            )
            print(f"FOUND")
            fixed += 1
        else:
            print(f"NOT FOUND")
        
        time.sleep(0.25)  # Rate limiting
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"\nFixed {fixed}/{len(to_fix)} movie posters")

def fix_all_movies_without_posters():
    """Fix ALL movies in database that don't have posters."""
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Get movies without TMDB posters
    cursor.execute("""
        SELECT id, title FROM movies 
        WHERE poster_url IS NULL OR poster_url NOT LIKE '%tmdb%'
        ORDER BY id
        LIMIT 100
    """)
    movies = cursor.fetchall()
    
    print(f"Found {len(movies)} movies without proper posters")
    print("-" * 80)
    
    fixed = 0
    for movie_id, title in movies:
        # Extract year
        year = None
        if "(" in title and ")" in title:
            year_str = title.split("(")[-1].split(")")[0]
            if year_str.isdigit() and len(year_str) == 4:
                year = year_str
        
        print(f"  [{movie_id}] {title[:50]}...", end=" ")
        poster_url = get_tmdb_poster(title, year)
        
        if poster_url:
            cursor.execute(
                "UPDATE movies SET poster_url = %s WHERE id = %s",
                (poster_url, movie_id)
            )
            print(f"OK")
            fixed += 1
        else:
            print(f"SKIP")
        
        time.sleep(0.25)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"\nFixed {fixed}/{len(movies)} movie posters")

if __name__ == "__main__":
    print("=" * 80)
    print("TMDB Poster Fixer")
    print("=" * 80)
    
    # First check and fix recommended movies
    check_and_fix_posters()
    
    print("\n" + "=" * 80)
    print("Fixing additional movies in database...")
    print("=" * 80)
    
    # Then fix more movies
    fix_all_movies_without_posters()
