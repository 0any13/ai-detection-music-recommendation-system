import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

def find_project_root():
    """Automatically find the project root directory"""
    current = Path.cwd()
    
    #check if we're alr in project root(has 'data' folder)
    if (current / "data").exists():
        return current
    
    #check parent directory
    if (current.parent / "data").exists():
        return current.parent
    
    #if running from 'data' folder
    if current.name == "data":
        return current.parent
    
    #manual input
    print("Could not automatically detect project root.")
    print(f"Current directory: {current}")
    manual_path = input("Enter full path to Music-Recommendation-System folder: ").strip()
    return Path(manual_path)

#get project root
PROJECT_ROOT = find_project_root()
TRAIN_TRIPLETS_FILE = PROJECT_ROOT / "data" / "train_triplets.txt"
OUTPUT_CSV = PROJECT_ROOT / "data" / "user_interactions.csv"

def count_lines(filepath):
    """Count lines in file efficiently"""
    print("Counting lines in dataset (this takes a moment)...")
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count

def parse_user_interactions(max_lines=None):
    """Parse train_triplets.txt from the Echo Nest Taste Profile subset of MSD with format user_id <SEP> song_id <SEP> play_count"""
    
    print("Parsing user interaction data")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Input File: {TRAIN_TRIPLETS_FILE}")
    print(f"Output File: {OUTPUT_CSV}")
    
    #check if file exists
    if not TRAIN_TRIPLETS_FILE.exists():
        print(f"\nError: File not found: {TRAIN_TRIPLETS_FILE}")
        print("\nThe MSD Taste Profile dataset should contain 'train_triplets.txt'")
        print("Download from: http://millionsongdataset.com/tasteprofile/")
        print("\nExpected location: data/train_triplets.txt")
        return None
    
    #get file size
    file_size = TRAIN_TRIPLETS_FILE.stat().st_size
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    #count total lines
    try:
        total_lines = count_lines(TRAIN_TRIPLETS_FILE)
        print(f"Total lines: {total_lines:,}")
    except:
        total_lines = None
        print("Line count: Unknown (file too large to count quickly)")
    
    if max_lines:
        print(f"Processing only first {max_lines:,} lines (test mode)")
    
    print("\nStarting parsing...")
    print("   (This may take several minutes for large datasets)\n")
    
    interactions = []
    line_count = 0
    error_count = 0
    progress_interval = 100000  #update every 100k lines
    
    try:
        with open(TRAIN_TRIPLETS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                
                #progress indicator
                if line_count % progress_interval == 0:
                    percent = (line_count / total_lines * 100) if total_lines else 0
                    print(f"   Progress: {line_count:,} lines | "
                          f"Valid: {len(interactions):,} | "
                          f"Errors: {error_count}" +
                          (f" | {percent:.1f}%" if total_lines else ""))
                
                #stop if max_lines reached
                if max_lines and line_count > max_lines:
                    print(f"\nReached max_lines limit: {max_lines:,}")
                    break
                
                #parse line
                parts = line.strip().split('\t')
                
                if len(parts) != 3:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Malformed line {line_count}: {line[:50]}")
                    continue
                
                user_id, track_id, play_count = parts
                
                try:
                    play_count = int(play_count)
                except ValueError:
                    error_count += 1
                    continue
                
                interactions.append({
                    'user_id': user_id,
                    'track_id': track_id,
                    'play_count': play_count
                })
        
        print(f"\nParsing complete!")
        print(f"   Lines processed: {line_count:,}")
        print(f"   Valid interactions: {len(interactions):,}")
        print(f"   Errors (skipped): {error_count}")
        
        if len(interactions) == 0:
            print("\nNo valid interactions found!")
            print("Please check the file format. Expected format:")
            print("user_id<SEP>track_id<SEP>play_count")
            return None
        
        #convert to dataframe
        print("\nCreating DataFrame...")
        df = pd.DataFrame(interactions)
        
        #calculate stats
        print("Dataset Statistics")
        print(f"Total interactions: {len(df):,}")
        print(f"Unique users: {df['user_id'].nunique():,}")
        print(f"Unique tracks: {df['track_id'].nunique():,}")
        print(f"Average plays per user: {df.groupby('user_id')['play_count'].sum().mean():.1f}")
        print(f"Average plays per track: {df.groupby('track_id')['play_count'].sum().mean():.1f}")
        print(f"Total play count: {df['play_count'].sum():,}")
        
        #play count distribution
        print(f"\nPlay count distribution:")
        print(df['play_count'].describe())
        
        #show most active users
        print(f"\nTop 5 most active users:")
        top_users = df.groupby('user_id')['play_count'].sum().nlargest(5)
        for i, (user_id, plays) in enumerate(top_users.items(), 1):
            print(f"  {i}. {user_id[:20]}... - {plays:,} total plays")
        
        #show most popular tracks
        print(f"\nTop 5 most popular tracks:")
        top_tracks = df.groupby('track_id')['play_count'].sum().nlargest(5)
        for i, (track_id, plays) in enumerate(top_tracks.items(), 1):
            print(f"  {i}. {track_id} - {plays:,} total plays")
        
        #save to CSV
        print(f"\nSaving to: {OUTPUT_CSV}")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        
        saved_size = OUTPUT_CSV.stat().st_size
        print(f"Saved! File size: {saved_size:,} bytes ({saved_size/1024/1024:.1f} MB)")
        
        return df
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_sample(input_csv, output_csv, sample_size=100000):
    """Create a smaller sample for testing"""
    print(f"\nCreating sample of {sample_size:,} interactions...")
    
    df = pd.read_csv(input_csv)
    
    #sample randomly
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"\nSample statistics:")
    print(f"  Interactions: {len(sample_df):,}")
    print(f"  Users: {sample_df['user_id'].nunique():,}")
    print(f"  Tracks: {sample_df['track_id'].nunique():,}")
    
    sample_df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")


def main():
    """Main execution"""
    
    #check if tracks.csv exists (should be created first)
    tracks_csv = PROJECT_ROOT / "data" / "tracks.csv"
    if not tracks_csv.exists() or tracks_csv.stat().st_size < 100:
        print("Warning: tracks.csv not found or is empty!")
        print(f"Expected location: {tracks_csv}")
        
        proceed = input("\nContinue anyway? (y/n): ").lower()
        if proceed != 'y':
            return
    
    #ask test mode
    print("Parsing Mode")
    print("\nOptions:")
    print("1.Full dataset (recommended, but slow)")
    print("2.Test mode - first 100,000 lines (fast)")
    print("3.Test mode - first 1,000,000 lines (medium)")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    max_lines = None
    if choice == '2':
        max_lines = 100000
    elif choice == '3':
        max_lines = 1000000
    
    #parse the data
    df = parse_user_interactions(max_lines=max_lines)
    
    if df is not None:
        print("Success!")
        print(f"\nYou now have user_interactions.csv with {len(df):,} interactions")
        
        #optionally create a smaller sample
        if len(df) > 100000:
            create_sample_choice = input("\nCreate a smaller sample for testing? (y/n): ").lower()
            
            if create_sample_choice == 'y':
                sample_output = PROJECT_ROOT / "data" / "user_interactions_sample.csv"
                create_sample(OUTPUT_CSV, sample_output, sample_size=100000)
        
    else:
        print("Failed")
        print("\nTroubleshooting:")
        print("1.Make sure train_triplets.txt is in data/ folder")
        print("2.Check that the file format matches: user_id<SEP>track_id<SEP>play_count")
        print("3.Verify you have read permissions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()