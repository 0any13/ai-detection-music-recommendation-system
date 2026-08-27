import musicbrainzngs as mb
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import json

#configure MusicBrainz
mb.set_useragent(
    "MusicRecommendationSystem",
    "1.0",
    "https://github.com/user/music-rec-sys"
)
mb.set_rate_limit(limit_or_interval=1.0)  #1 request per sec


class SimpleMBIDEnricher:
    """MBID enricher"""
    
    def __init__(self):
        self.cache_file = Path("data/mbid_cache.json")
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load cache"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save cache"""
        self.cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def search_mbid(self, artist, title):
        """Search for MBID"""
        
        #check cache
        cache_key = f"{artist}|||{title}".lower()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            result = mb.search_recordings(
                artist=artist,
                recording=title,
                limit=1,
                strict=False
            )
            
            if result['recording-list']:
                mbid = result['recording-list'][0]['id']
                self.cache[cache_key] = mbid
                return mbid
            
            self.cache[cache_key] = None
            return None
            
        except Exception as e:
            print(f"   Error: {e}")
            return None
    
    def enrich_tracks(self, max_tracks=None):
        """Add MBIDs to tracks.csv"""
        
        print("Adding MBIDS to tracks.csv")
        
        #load tracks
        tracks_file = Path("data/tracks.csv")
        
        if not tracks_file.exists():
            print(f"\nError: {tracks_file} not found!")
            print("Make sure you're in the main project folder")
            return
        
        print(f"\nLoading {tracks_file}...")
        df = pd.read_csv(tracks_file)
        
        print(f"Found {len(df):,} tracks")
        
        #check if mbid column exists
        if 'mbid' in df.columns:
            existing = df['mbid'].notna().sum()
            print(f"Already have {existing:,} MBIDs")
            
            #only process tracks without MBIDs
            df_to_process = df[df['mbid'].isna()].copy()
            print(f"Processing {len(df_to_process):,} remaining tracks")
        else:
            df['mbid'] = None
            df_to_process = df.copy()
        
        #limit if specified
        if max_tracks and len(df_to_process) > max_tracks:
            df_to_process = df_to_process.head(max_tracks)
            print(f"\nLimiting to {max_tracks:,} tracks")
        
        #process tracks
        print(f"\nSearching MusicBrainz...")
        print("(Rate: 1 request/second, be patient!)\n")
        
        found = 0
        not_found = 0
        
        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
            artist = row.get('artist')
            title = row.get('title')
            
            if pd.isna(artist) or pd.isna(title):
                not_found += 1
                continue
            
            mbid = self.search_mbid(artist, title)
            
            if mbid:
                df.loc[idx, 'mbid'] = mbid
                found += 1
            else:
                not_found += 1
            
            #save cache periodically
            if (found + not_found) % 100 == 0:
                self._save_cache()
        
        #final save
        self._save_cache()
        
        #save updated tracks.csv
        print(f"\nSaving updated tracks.csv...")
        df.to_csv(tracks_file, index=False)
        
        print(f"\nDone")
        print(f"   Found: {found:,}")
        print(f"   Not found: {not_found:,}")
        print(f"   Total MBIDs: {df['mbid'].notna().sum():,}")
        
        return df


def main():
    """Main execution"""
    
    print("MUSICBRAINZ MBID ENRICHER")
    
    print("""
This adds MusicBrainz IDs (MBIDs) to tracks.csv
MBIDs are needed to get audio features from AcousticBrainz

How many tracks to process?
1. Test (100 tracks) - ~2 minutes
2. Small (1,000 tracks) - ~17 minutes  
3. Medium (10,000 tracks) - ~3 hours
4. All tracks - ~100+ hours (run overnight/weekends)
    """)
    
    choice = input("Choice (1/2/3/4): ").strip()
    
    max_tracks_map = {
        '1': 100,
        '2': 1000,
        '3': 10000,
        '4': None
    }
    
    max_tracks = max_tracks_map.get(choice, 100)
    
    #initialize enricher
    enricher = SimpleMBIDEnricher()
    
    #process tracks
    try:
        enricher.enrich_tracks(max_tracks=max_tracks)
        
        print("NEXT STEPS")
        print("""
1. Test content-based recommendations:
   python test_content_similarity.py

2. Build hybrid system combining CF + content-based!
        """)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving cache...")
        enricher._save_cache()
        print("Cache saved. Run again to resume.")


if __name__ == "__main__":
    #check if musicbrainzngs is installed
    try:
        import musicbrainzngs
    except ImportError:
        print("Installing musicbrainzngs...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'musicbrainzngs'])
        print("Please run the script again")
        exit(0)
    
    main()