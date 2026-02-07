
import requests
import json

def verify_logic():
    url = "http://localhost:8000/recommend/cold_start"
    payload = {
        "genres": ["Children"],
        "keywords": [],
        "selected_movie_ids": [],
        "top_k": 10
    }
    
    print(f"Sending request to {url} with payload: {json.dumps(payload)}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print(f"\nStatus Code: {response.status_code}")
        
        recommendations = data.get("recommendations", [])
        print(f"\nReceived {len(recommendations)} recommendations:")
        
        # Analyze results
        pure_matches = 0
        mixed_matches = 0
        
        for i, movie in enumerate(recommendations, 1):
            genres_list = [g.strip() for g in movie['genres'].split('|')]
            is_pure = len(genres_list) == 1 and genres_list[0].lower() == 'children'
            
            status = "[PURE]" if is_pure else "[MIXED]"
            if is_pure:
                pure_matches += 1
            else:
                mixed_matches += 1
            
            print(f"{i}. {movie['title']} (ID: {movie['id']})")
            print(f"   Genres: {movie['genres']} {status}")
        
        print(f"\nSummary:")
        print(f"Pure Matches (Score 1.0): {pure_matches}")
        print(f"Mixed Matches (Score < 1.0): {mixed_matches}")
        
        if pure_matches > 0 and recommendations[0]['genres'].lower() == 'children':
             print("\n[SUCCESS] Ranking Logic Verified: 'Pure' matches appear first.")
        else:
             print("\n[WARNING] Validation Failed: Check if ranking is working correctly.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_logic()
