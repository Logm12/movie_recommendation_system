import os
import requests
import zipfile
import io

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "../data"

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL)
    response.raise_for_status()

    print("Extracting data...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(DATA_DIR)
    
    print("Download complete.")
    
    # Move files to base data dir for easier access if desired, or keep in subfolder
    # The zip creates a 'ml-latest-small' folder.
    print(f"Data saved to {os.path.join(DATA_DIR, 'ml-latest-small')}")

if __name__ == "__main__":
    download_data()
