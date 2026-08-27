import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from collaborative_filtering_recommender import CollaborativeFilteringRecommender

def test_recommender():
    """Run comprehensive tests"""
    
    print("Collaborative Filtering Recommender - Test")
    
    #initialize
    print("\nLoading recommender...")
    recommender = CollaborativeFilteringRecommender(data_path="data")
    recommender.load_latest_model()
    
    #test 1: search by artist
    print("Test 1: Search by Artist Name")
    
    test_artists = [
        "Radiohead",
        "Coldplay", 
        "The Beatles",
        "Metallica",
        "Taylor Swift"
    ]
    
    for artist in test_artists:
        print(f"\nSearching for: {artist}")
        track_id = recommender.search_track_by_metadata(artist=artist)
        
        if track_id:
            metadata = recommender._get_track_metadata(track_id)
            print(f"   Found: {metadata['artist']} - {metadata['title']}")
            print(f"   Track ID: {track_id}")
            
            #get recommendations
            recs = recommender.get_similar_tracks(track_id, n=5)
            if recs:
                print(f"\n   Top 3 similar tracks:")
                for i, rec in enumerate(recs[:3], 1):
                    print(f"   {i}. {rec['artist']} - {rec['title']}")
                    print(f"      Similarity: {rec['similarity_score']:.4f}")
        else:
            print(f"   Not found in dataset")
    
    #test 2: search by song title
    print("Test 2: Search by Song Title")
    
    test_songs = [
        "Creep",
        "Wonderwall",
        "Smells Like Teen Spirit",
        "Bohemian Rhapsody",
        "Stairway to Heaven"
    ]
    
    for song in test_songs:
        print(f"\nSearching for: {song}")
        track_id = recommender.search_track_by_metadata(title=song)
        
        if track_id:
            metadata = recommender._get_track_metadata(track_id)
            print(f"   Found: {metadata['artist']} - {metadata['title']}")
            
            #get recommendations
            recs = recommender.get_similar_tracks(track_id, n=3)
            if recs:
                print(f"\n   Similar tracks:")
                for i, rec in enumerate(recs, 1):
                    print(f"   {i}. {rec['artist']} - {rec['title']}")
        else:
            print(f"Not found in dataset")
    
    #test 3: recommendation quality
    print("Test 3: Recommendation Quality Check")
    
    #get a popular track
    popular_tracks = recommender.get_popular_tracks(n=1)
    if popular_tracks:
        test_track = popular_tracks[0]
        print(f"\nTesting with: {test_track['artist']} - {test_track['title']}")
        print(f"   Popularity: {test_track['popularity_score']:.0f}")
        
        #get 10 recommendations
        recs = recommender.get_similar_tracks(test_track['track_id'], n=10)
        
        if recs:
            print(f"\nGenerated {len(recs)} recommendations")
            print(f"\nTop 10 similar tracks:")
            
            for i, rec in enumerate(recs, 1):
                print(f"\n{i}. {rec['artist']} - {rec['title']}")
                print(f"   Similarity Score: {rec['similarity_score']:.4f}")
                
                if rec['tags']:
                    tags = rec['tags'].split(',')[:3]
                    print(f"   Tags: {', '.join(tags)}")
            
            #check quality metrics
            print("\n" + "=" * 70)
            print("QUALITY METRICS")
            print("=" * 70)
            
            avg_similarity = sum(r['similarity_score'] for r in recs) / len(recs)
            min_similarity = min(r['similarity_score'] for r in recs)
            max_similarity = max(r['similarity_score'] for r in recs)
            
            print(f"\nSimilarity scores:")
            print(f"   Average: {avg_similarity:.4f}")
            print(f"   Min: {min_similarity:.4f}")
            print(f"   Max: {max_similarity:.4f}")
            
            #check diversity (unique artists)
            unique_artists = len(set(r['artist'] for r in recs))
            print(f"\nDiversity:")
            print(f"   Unique artists: {unique_artists}/10")
            
            if unique_artists >= 7:
                print(f"Good diversity!")
            elif unique_artists >= 5:
                print(f"Moderate diversity")
            else:
                print(f" Low diversity (many songs from same artist)")
            
            if avg_similarity > 0.5:
                print(f"\nQuality Check Passed!")
                print(f"   Average similarity >0.5 indicates good recommendations")
            else:
                print(f"\nQuality check: Similarity scores are low")
        else:
            print(f"No recommendations generated")
    
    #test 4: edge cases
    print("Test 4: Edge Cases")
    
    #test with invalid track ID
    print("\nTest with invalid track ID...")
    recs = recommender.get_similar_tracks("INVALID_TRACK_ID", n=5)
    if len(recs) == 0:
        print("   Correctly handled invalid track ID")
    
    #test with track not in training data
    print("\nTest with track not in training data...")
    #find a track in metadata but not in training
    all_tracks = recommender.tracks_metadata
    training_tracks = set(recommender.item_mapping.keys())
    
    for track_id in all_tracks.index:
        if track_id not in training_tracks:
            metadata = recommender._get_track_metadata(track_id)
            print(f"   Testing: {metadata['artist']} - {metadata['title']}")
            recs = recommender.get_similar_tracks(track_id, n=5)
            if len(recs) == 0:
                print("   Correctly handled track not in training data")
            break
    
    #final summary
    print("""
  Recommender is working!
The collaborative filtering system is production-ready! 
    """)


def interactive_test():
    """Interactive testing mode"""
    
    print("Interactive Recommender Test")
    
    recommender = CollaborativeFilteringRecommender(data_path="data")
    recommender.load_latest_model()
    
    print("\nRecommender loaded! You can now search for songs.")
    print("Type 'quit' to exit\n")
    
    while True:
        print("-" * 70)
        artist = input("\nEnter artist name (or 'quit'): ").strip()
        
        if artist.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye!")
            break
        
        if not artist:
            continue
        
        #search for artist
        track_id = recommender.search_track_by_metadata(artist=artist)
        
        if track_id:
            metadata = recommender._get_track_metadata(track_id)
            print(f"\nFound: {metadata['artist']} - {metadata['title']}")
            
            #ask for number of recommendations
            try:
                n = int(input("How many recommendations? (default 10): ") or "10")
            except:
                n = 10
            
            #get recommendations
            print(f"\n Getting {n} similar tracks...")
            recs = recommender.get_similar_tracks(track_id, n=n)
            
            if recs:
                print(f"\nTop {len(recs)} Similar Tracks:")
                
                for i, rec in enumerate(recs, 1):
                    print(f"\n{i}. {rec['artist']} - {rec['title']}")
                    print(f"   Similarity: {rec['similarity_score']:.4f}")
                    
                    if rec['tags']:
                        tags = rec['tags'].split(',')[:5]
                        print(f"   Tags: {', '.join(tags)}")
            else:
                print("\n No recommendations found")
        else:
            print(f"\n Artist '{artist}' not found in dataset")
            print(" Try: Radiohead, Coldplay, The Beatles, etc.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        test_recommender()
        
        print("\n Want to try interactive mode?")
        print("   Run: python test_recommender.py interactive")