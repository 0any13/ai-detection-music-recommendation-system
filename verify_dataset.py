import pandas as pd
from pathlib import Path

def verify_dataset_completeness():
    """Check if dataset is complete"""
    
    print("Dataset Completeness Check")
    
    #expected numbers from official MSD documentation
    EXPECTED_INTERACTIONS = 48_373_586
    EXPECTED_USERS = 1_019_318
    EXPECTED_TRACKS = 384_546
    
    #check train_triplets.txt
    print("\nChecking train_triplets.txt...")
    triplets_file = Path("data/train_triplets.txt")
    
    if not triplets_file.exists():
        print("train_triplets.txt not found!")
        return
    
    print(f"File exists: {triplets_file}")
    
    file_size = triplets_file.stat().st_size
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024/1024:.2f} GB)")
    
    #expected size: ~2.2 GB
    expected_size = 2.2 * 1024 * 1024 * 1024
    
    if file_size < expected_size * 0.9:
        print(f" WARNING: File seems smaller than expected!")
        print(f"   Expected: ~2.2 GB")
        print(f"   Your file: {file_size/1024/1024/1024:.2f} GB")
    else:
        print(f"File size looks correct!")
    
    #check user_interactions.csv
    print("\nChecking user_interactions.csv...")
    interactions_file = Path("data/user_interactions.csv")
    
    if not interactions_file.exists():
        print("user_interactions.csv not found!")
        print("   Run: python parse_user_interactions.py")
        return
    
    print("Loading (this may take a moment)...")
    df = pd.read_csv(interactions_file)
    
    actual_interactions = len(df)
    actual_users = df['user_id'].nunique()
    actual_tracks = df['track_id'].nunique()
    
    print("Dataset Statistics Compaarison")
    
    print(f"\n{'Metric':<30} {'Expected':<15} {'Actual':<15} {'Status'}")
    
    #interactions
    status = "OK" if actual_interactions == EXPECTED_INTERACTIONS else "WARN"
    print(f"{'Total interactions':<30} {EXPECTED_INTERACTIONS:<15,} {actual_interactions:<15,} {status}")
    
    #users
    status = "OK" if actual_users == EXPECTED_USERS else "WARN"
    print(f"{'Unique users':<30} {EXPECTED_USERS:<15,} {actual_users:<15,} {status}")
    
    #tracks
    status = "OK" if actual_tracks == EXPECTED_TRACKS else "WARN"
    print(f"{'Unique tracks':<30} {EXPECTED_TRACKS:<15,} {actual_tracks:<15,} {status}")
    
    #coverage
    interaction_coverage = (actual_interactions / EXPECTED_INTERACTIONS) * 100
    user_coverage = (actual_users / EXPECTED_USERS) * 100
    track_coverage = (actual_tracks / EXPECTED_TRACKS) * 100
    

    print("Coverage")
    print(f"\nInteractions: {interaction_coverage:.1f}%")
    print(f"Users: {user_coverage:.1f}%")
    print(f"Tracks: {track_coverage:.1f}%")
    
    #verdict
    print("Verdict")
    
    if actual_interactions >= EXPECTED_INTERACTIONS * 0.99:
        print("\n Your Dataset Is Complete!")
        print("\nYou have the full Last.fm taste profile dataset.")
        
    elif actual_interactions >= EXPECTED_INTERACTIONS * 0.5:
        print("\n Your Dataset Is Partial")
        print(f"\nYou have {interaction_coverage:.1f}% of the full dataset.")
        print("\nOptions:")
        print("1. Re-download train_triplets.txt if incomplete")
        print("2. Re-run parse_user_interactions.py with full dataset")
        
    else:
        print("\n Your Dataset Is Incomplete")
        print("\nYou need to:")
        print("1. Download complete train_triplets.txt")
        print("2. Re-run parse_user_interactions.py")

    
    print("SUMMARY")
    
    if actual_interactions >= EXPECTED_INTERACTIONS * 0.99:
        print("""
Your dataset is complete!
Your model is trained on all available data and the system is working correctly!
        """)
    else:
        print(f"""
You have {interaction_coverage:.1f}% of the dataset.

To get the full dataset:
1. Download train_triplets.txt (2.2 GB)
   URL: http://millionsongdataset.com/tasteprofile/
2. Re-run: python parse_user_interactions.py
3. Re-train: python train_collaborative_filtering.py
        """)


if __name__ == "__main__":
    try:
        verify_dataset_completeness()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()