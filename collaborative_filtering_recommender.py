import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import re as _re_cf
import unicodedata

def _norm_prefix(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return _re_cf.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

class CollaborativeFilteringRecommender:
    """Load and use trained ALS model for recommendations"""
    
    def __init__(self, model_timestamp: str = None, data_path: str = "data"):
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.tracks_metadata = None
        self.data_path = Path(data_path)
        
        if model_timestamp:
            self.load_model(model_timestamp)
    
    def load_latest_model(self):
        """Load the most recent trained model"""
        models_dir = Path("models")
        metadata_file = models_dir / "latest_model_info.txt"
        
        if not metadata_file.exists():
            raise FileNotFoundError(
                "No trained model found!\n"
                "Run train_collaborative_filtering.py first."
            )
        
        #read timestamp from metadata
        with open(metadata_file, 'r') as f:
            first_line = f.readline()
            timestamp = first_line.split(": ")[1].strip()
        
        print(f"Loading latest model: {timestamp}")
        self.load_model(timestamp)
    
    def load_model(self, timestamp: str):
        """Load trained model and mappings"""
        models_dir = Path("models")
        
        print(f"\nLoading model components...")
        
        model_file = models_dir / f"als_model_{timestamp}.pkl"
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        print(f"   Model loaded!:)")
        
        #load mappings
        mappings_file = models_dir / f"mappings_{timestamp}.pkl"
        with open(mappings_file, 'rb') as f:
            mappings = pickle.load(f)
            self.user_mapping = mappings['user_mapping']
            self.item_mapping = mappings['item_mapping']
            self.reverse_item_mapping = mappings['reverse_item_mapping']
        print(f"   Mappings loaded ({len(self.item_mapping):,} tracks)")
        
        #load matrices
        matrices_file = models_dir / f"matrices_{timestamp}.pkl"
        with open(matrices_file, 'rb') as f:
            matrices = pickle.load(f)
            self.user_item_matrix = matrices['user_item_matrix']
            self.item_user_matrix = matrices['item_user_matrix']
        print(f"    Matrices loaded!:)")
        
        #load track metadata
        self._load_track_metadata()
        
        print(f"\n Model ready for recommendations!")
    
    def _load_track_metadata(self):
        """Load track metadata for enriching recommendations"""
        tracks_file = self.data_path / "tracks.csv"
        
        if tracks_file.exists():
            self.tracks_metadata = pd.read_csv(tracks_file)
            self.tracks_metadata.set_index('track_id', inplace=True)
            print(f"    Track metadata loaded ({len(self.tracks_metadata):,} tracks)")
        else:
            print(f"     Track metadata not found at {tracks_file}")
            self.tracks_metadata = None
    
    def get_similar_tracks(self, track_id: str, n: int = 10) -> List[Dict]:
        """Get tracks similar to given track using collaborative filtering.Returns list of similar tracks with metadata"""
        if not self.model:
            raise ValueError("Model not loaded! Call load_model() or load_latest_model() first.")
        
        if track_id not in self.item_mapping:
            print(f"  Track '{track_id}' not found in training data")
            return []
        
        track_idx = self.item_mapping[track_id]
        
        #get similar items from ALS model
        ids, scores = self.model.similar_items(track_idx, N=n+1)
        
        recommendations = []
        for similar_idx, score in zip(ids, scores):
            #convert numpy int32 to Python int for dictionary lookup
            similar_idx = int(similar_idx)
            
            #check if index exists in mapping
            if similar_idx not in self.reverse_item_mapping:
                print(f"     Warning: Index {similar_idx} not in reverse mapping, skipping")
                continue
            
            similar_track_id = self.reverse_item_mapping[similar_idx]
            
            #skip the same track
            if similar_track_id == track_id:
                continue
            
            #get metadata if available
            metadata = self._get_track_metadata(similar_track_id)
            
            rec = {
                'track_id': similar_track_id,
                'similarity_score': float(score),
                'artist': metadata.get('artist', 'Unknown'),
                'title': metadata.get('title', 'Unknown'),
                'tags': metadata.get('tags', ''),
                'num_tags': metadata.get('num_tags', 0)
            }
            
            recommendations.append(rec)
            
            if len(recommendations) >= n:
                break
        
        return recommendations
    
    def _get_track_metadata(self, track_id: str) -> Dict:
        """Get metadata for a track"""
        if self.tracks_metadata is None:
            return {}
        
        try:
            track_data = self.tracks_metadata.loc[track_id]
            return {
                'artist': track_data.get('artist', 'Unknown'),
                'title': track_data.get('title', 'Unknown'),
                'tags': track_data.get('tags', ''),
                'num_tags': track_data.get('num_tags', 0)
            }
        except KeyError:
            return {}
    
    def search_track_by_metadata(self, artist: str = None, title: str = None) -> Optional[str]:
        """Find a track ID by artist and/or title.Returns track_id if found, None otherwise"""
        if self.tracks_metadata is None:
            return None
        
        #filter by artist
        matches = self.tracks_metadata
        if artist:
            matches = matches[matches['artist'].str.contains(artist, case=False, na=False, regex=False)]
        
        #filter by title
        if title:
            matches = matches[matches['title'].str.contains(title, case=False, na=False, regex=False)]
        
        if len(matches) == 0:
            # MSD truncates long titles at ~30 characters.so we check if the title is a prefix for the query or vice versa
            if artist and title:
                artist_matches = self.tracks_metadata[
                    self.tracks_metadata["artist"].str.contains(
                        artist, case=False, na=False, regex=False
                    )
                ]
                q = _norm_prefix(title)
                for idx, row in artist_matches.iterrows():
                    m = _norm_prefix(str(row.get("title", "")))
                    if len(m) >= 15 and q.startswith(m):
                        return idx   # MSD title is a truncated prefix of query
                    if len(q) >= 15 and m.startswith(q):
                        return idx   # query is a prefix of MSD title
            return None
        
        #return first match's track_id (the index)
        return matches.index[0]
    
    def get_popular_tracks(self, n: int = 10) -> List[Dict]: #n for num of tracks to return
        """Get most popular tracks based on interaction counts.Returns list of popular tracks with metadata"""
        if not self.model:
            raise ValueError("Model not loaded!")
        
        #calculate popularity from interaction matrix
        track_popularity = self.item_user_matrix.sum(axis=1).A1  # Sum of play counts per track
        
        #get top N
        top_indices = track_popularity.argsort()[::-1][:n]
        
        popular_tracks = []
        for idx in top_indices:
            #convert numpy int to Python int
            idx = int(idx)
            
            if idx not in self.reverse_item_mapping:
                continue
                
            track_id = self.reverse_item_mapping[idx]
            metadata = self._get_track_metadata(track_id)
            
            popular_tracks.append({
                'track_id': track_id,
                'popularity_score': float(track_popularity[idx]),
                'artist': metadata.get('artist', 'Unknown'),
                'title': metadata.get('title', 'Unknown'),
                'tags': metadata.get('tags', ''),
                'num_tags': metadata.get('num_tags', 0)
            })
        
        return popular_tracks


def test_recommender():
    """Test the recommender"""
    print("=" * 70)
    print("TESTING COLLABORATIVE FILTERING RECOMMENDER")
    print("=" * 70)
    
    #initialize recommender
    recommender = CollaborativeFilteringRecommender(data_path="data")
    
    try:
        #load latest model
        recommender.load_latest_model()
        
        #Test 1:Get popular tracks
        print("\n" + "=" * 70)
        print("TEST 1: Most Popular Tracks")
        print("=" * 70)
        
        popular = recommender.get_popular_tracks(n=10)
        for i, track in enumerate(popular, 1):
            print(f"\n{i}. {track['title']} - {track['artist']}")
            print(f"   Track ID: {track['track_id']}")
            print(f"   Popularity: {track['popularity_score']:.0f}")
            if track['tags']:
                tags = track['tags'].split(',')[:5]
                print(f"   Tags: {', '.join(tags)}")
        
        #Test 2: Get similar tracks (if we have track IDs)
        if popular:
            test_track_id = popular[0]['track_id']
            
            print("\n" + "=" * 70)
            print(f"TEST 2: Similar to '{popular[0]['title']}'")
            print("=" * 70)
            
            similar = recommender.get_similar_tracks(test_track_id, n=10)
            
            if similar:
                for i, track in enumerate(similar, 1):
                    print(f"\n{i}. {track['title']} - {track['artist']}")
                    print(f"   Similarity: {track['similarity_score']:.4f}")
                    if track['tags']:
                        tags = track['tags'].split(',')[:5]
                        print(f"   Tags: {', '.join(tags)}")
            else:
                print("\n  No similar tracks found")
        
        #Test 3: Search by artist/title
        print("\n" + "=" * 70)
        print("TEST 3: Search Track by Artist/Title")
        print("=" * 70)
        
        search_artist = input("\nEnter artist name to search (or press Enter to skip): ").strip()
        if search_artist:
            found_track_id = recommender.search_track_by_metadata(artist=search_artist)
            if found_track_id:
                print(f"\n Found track: {found_track_id}")
                metadata = recommender._get_track_metadata(found_track_id)
                print(f"   Artist: {metadata['artist']}")
                print(f"   Title: {metadata['title']}")
                
                #get recommendations
                similar = recommender.get_similar_tracks(found_track_id, n=5)
                if similar:
                    print(f"\n   Similar tracks:")
                    for i, track in enumerate(similar, 1):
                        print(f"   {i}. {track['title']} - {track['artist']}")
            else:
                print(f"\n No track found for artist '{search_artist}'")
        
        print("\n" + "=" * 70)
        print("Testing complete!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("\nPlease run train_collaborative_filtering.py first to train a model.")
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        test_recommender()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")