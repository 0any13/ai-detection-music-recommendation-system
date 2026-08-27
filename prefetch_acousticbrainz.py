import pandas as pd
from pathlib import Path
from improved_acousticbrainz import ImprovedAcousticBrainz
from tqdm import tqdm
import time

def prefetch_features():
    """Fetch AcousticBrainz features for all MBIDs,saves to cache so future loads are instant"""
    

    print("Pre-fetching AcousticBrainz Feature")
    
    #load tracks
    print("\nLoading tracks.csv...")
    df = pd.read_csv("data/tracks.csv", low_memory=False)
    print(f"{len(df):,} tracks")
    
    #get MBIDs
    tracks_with_mbid = df[df['mbid'].notna()]
    print(f"{len(tracks_with_mbid):,} have MBIDs")
    
    #initialize
    ab = ImprovedAcousticBrainz()
    
    print(f"\nFetching features for {len(tracks_with_mbid):,} tracks")
    print("This will take ~30-60 minutes")
    print("(Rate limited to ~1 request/second)")
    
    proceed = input("\nProceed? (y/n): ").lower()
    if proceed != 'y':
        return
    
    #fetch
    found = 0
    not_found = 0
    errors = 0
    
    for idx, row in tqdm(tracks_with_mbid.iterrows(), 
                        total=len(tracks_with_mbid),
                        desc="Fetching features"):
        
        mbid = row['mbid']
        
        try:
            features = ab.get_complete_features(mbid, verbose=False)
            
            if features:
                found += 1
            else:
                not_found += 1
            
            #rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"\n  Error for {mbid}: {e}")
    
    #results
    print("Results")
    print(f"\nProcessed: {len(tracks_with_mbid):,}")
    print(f"Features found: {found:,} ({found/len(tracks_with_mbid)*100:.1f}%)")
    print(f"Not in AcousticBrainz: {not_found:,}")
    print(f"Errors: {errors}")
    
    print("\nCache populated!")


if __name__ == "__main__":
    prefetch_features()