import pandas as pd
from pathlib import Path

def parse_unique_tracks_fixed():
    """Parse unique_tracks.txt with correct format"""

    print("Parsing unique_tracks.txt")
    
    file_path = Path("data/unique_tracks.txt")
    
    if not file_path.exists():
        print(f"{file_path} not found!")
        return None
    
    print(f"\nReading {file_path}...")
    print("   Format: track_id<SEP>song_id<SEP>artist<SEP>title")
    print("   Using column 1 (song_id) as track_id")
    
    tracks = []
    errors = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"   Processed {i:,} lines...")
            
            try:
                #split by <SEP>
                parts = line.strip().split('<SEP>')
                
                if len(parts) >= 4:
                    echo_nest_id = parts[0].strip()  
                    song_id = parts[1].strip()        
                    artist = parts[2].strip()
                    title = parts[3].strip()
                    
                    tracks.append({
                        'track_id': song_id,      #use song_id (SO...)
                        'artist': artist,
                        'title': title,
                        'echo_nest_id': echo_nest_id  
                    })
                else:
                    errors += 1
            
            except Exception as e:
                errors += 1
                continue
    
    print(f"\nParsed {len(tracks):,} tracks")
    if errors > 0:
        print(f"Skipped {errors:,} malformed lines")
    
    #create dataframe
    df = pd.DataFrame(tracks)
    
    #remove duplicates
    df = df.drop_duplicates(subset='track_id', keep='first')
    
    print(f"Unique tracks: {len(df):,}")
    
    #show sample
    print("\nSample tracks (showing SO... IDs):")
    print(df[['track_id', 'artist', 'title']].head(10))
    
    #check coverage
    print("Checking Coverage")
    
    interactions_file = Path("data/user_interactions.csv")
    if interactions_file.exists():
        print(f"\nLoading interactions...")
        interactions_df = pd.read_csv(interactions_file)
        
        your_track_ids = set(interactions_df['track_id'].unique())
        msd_track_ids = set(df['track_id'].unique())
        
        overlap = your_track_ids & msd_track_ids
        
        print(f"\nCoverage Analysis:")
        print(f"   Your unique tracks: {len(your_track_ids):,}")
        print(f"   MSD tracks in file: {len(msd_track_ids):,}")
        print(f"   Overlap: {len(overlap):,}")
        print(f"   Coverage: {len(overlap)/len(your_track_ids)*100:.1f}%")
        
        #check top tracks
        top_100 = interactions_df.groupby('track_id')['play_count'].sum().nlargest(100)
        top_100_ids = set(top_100.index)
        top_100_covered = top_100_ids & msd_track_ids
        
        print(f"\nTop 100 most popular tarcks:")
        print(f"   Covered: {len(top_100_covered)} / 100")
        print(f"   Coverage: {len(top_100_covered)}%")
        
        if len(overlap) > len(your_track_ids) * 0.9:
            print(f"\nExcellent! >90% coverage")
        elif len(overlap) > len(your_track_ids) * 0.5:
            print(f"\nGOOD: >50% coverage")
        else:
            print(f"\nWarning: <50% coverage")
    
    #save
    print("Saving File")
    
    output_file = Path("data/tracks.csv")
    print(f"\nSaving to {output_file}...")
    df[['track_id', 'artist', 'title']].to_csv(output_file, index=False)
    
    file_size = output_file.stat().st_size
    print(f"Saved! ({file_size:,} bytes)")
    
    print("Done!")
    print("""
    Your tracks.csv now has the track IDs (SO... format).
    Next steps: Test the recommender with python collaborative_filtering_recommender.py""")
    
    return df


if __name__ == "__main__":
    try:
        parse_unique_tracks_fixed()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()