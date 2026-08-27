import requests
import base64
import time
from typing import Dict, List


class SpotifyAPI:
    """Unified Spotify API client with all methods needed across the system"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = 0
        self.base_url = 'https://api.spotify.com/v1'
    
    #Authentification
    
    def get_access_token(self) -> str:
        """Get or refresh Spotify access token"""
        #return cached token if still valid
        if self.access_token and time.time() < self.token_expiry:
            return self.access_token
        
        #request new token
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        headers = {
            'Authorization': f'Basic {auth_base64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {'grant_type': 'client_credentials'}
        
        response = requests.post('https://accounts.spotify.com/api/token',
                                 headers=headers, data=data)
        
        if response.status_code == 200:
            json_result = response.json()
            self.access_token = json_result['access_token']
            #set expiry with 100 second buffer
            self.token_expiry = time.time() + json_result.get('expires_in', 3600) - 100
            return self.access_token
        else:
            raise Exception(f"Failed to get Spotify token: {response.status_code}")
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make authenticated request to Spotify API with automatic token refresh
        Args: API endpoint (e.g., "tracks/TRACK_ID"), Query parameters .Returns JSON response as dict"""
        headers = {'Authorization': f'Bearer {self.get_access_token()}'}
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                #token expired, refresh and retry
                self.access_token = None
                headers = {'Authorization': f'Bearer {self.get_access_token()}'}
                response = requests.get(url, headers=headers, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
            
            #return empty dict for any errors
            return {}
            
        except Exception as e:
            return {}

    
    def extract_track_id(self, url: str) -> str:
        """Extract track ID from Spotify URL"""
        if 'track/' in url:
            return url.split('track/')[1].split('?')[0]
        return url
    
    def get_track_info(self, track_id: str) -> Dict:
        """Get basic track information.Returns Dict with track info (name, artists, album, etc.)"""
        return self._make_request(f"tracks/{track_id}")
    
    #Playlist Operations (used by discovered_on_recommender.py)
    
    def search_playlists(self, query: str, limit: int = 50) -> List[Dict]:
        """Search for playlists by query.Takes as args search query (usually artist name)and max number of results (max 50)Returns a list of playlist objects"""
        params = {
            'q': query,
            'type': 'playlist',
            'limit': limit
        }
        data = self._make_request("search", params=params)
        return data.get('playlists', {}).get('items', [])
    
    def search_track(self, artist: str, title: str) -> Dict:
        """Search for a track by artist + title.Returns the first matching track object (with 'id', 'external_urls','album' -> 'images', etc.) or {} if nothing found."""
        query = f"track:{title} artist:{artist}"
        params = {'q': query, 'type': 'track', 'limit': 1}
        data = self._make_request("search", params=params)
        items = data.get('tracks', {}).get('items', [])
        return items[0] if items else {}
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """Get all tracks from a playlist (handles pagination).Takes as arg Spotify playlist ID and returns list of track objects"""
        all_tracks = []
        offset = 0
        limit = 100
        
        while True:
            params = {'limit': limit, 'offset': offset}
            data = self._make_request(f"playlists/{playlist_id}/tracks", params=params)
            
            if not data:
                break
            
            items = data.get('items', [])
            if not items:
                break
            
            #extract track objects (skip none/invalid items)
            for item in items:
                if not item or not isinstance(item, dict):
                    continue
                
                track = item.get('track')
                if track and isinstance(track, dict):
                    all_tracks.append(track)
            
            #check if there are more tracks
            if len(items) < limit or not data.get('next'):
                break
            
            offset += limit
        
        return all_tracks
    
    #Artist Operations (used by discovered_on_recommender.py)
    
    def get_artist_top_tracks(self, artist_id: str, limit: int = 10) -> List[Dict]:
        """
        Get artist's top tracks (for discography sampling).Takes as args Spotify artist ID and number of tracks to return (max 10 from API)
        Returns a list of track objects."""
        params = {'market': 'US'}
        data = self._make_request(f"artists/{artist_id}/top-tracks", params=params)
        tracks = data.get('tracks', [])
        return tracks[:limit]
    
    def get_artist_albums(self, artist_id: str) -> List[Dict]:
        """Get artist's albums and singles"""
        params = {
            'include_groups': 'album,single',
            'market': 'US',
            'limit': 50
        }
        data = self._make_request(f"artists/{artist_id}/albums", params=params)
        return data.get('items', [])
    
    def get_artist_info(self, artist_id: str) -> Dict:
        """Get artist information.Returns a dict with artist info (name, genres, popularity, etc.)"""
        return self._make_request(f"artists/{artist_id}")
    
    #Recommendations (used by discovered_on_recommender.py)
    
    def get_recommendations(
        self, 
        seed_tracks: List[str] = None,
        seed_artists: List[str] = None,
        seed_genres: List[str] = None,
        limit: int = 20,
        **kwargs
    ) -> List[Dict]:
        """Get Spotify's algorithmic recommendations
        Args:
            seed_tracks: list of track IDs (max 5 total seeds), seed_artists: list of artist IDs
            seed_genres: list of genre strings, limit: number of recommendations
            **kwargs: additional tuning parameters (e.g., target_energy, min_tempo)    
        Returns:
            List of recommended track objects"""
        params = {'limit': limit}
        
        if seed_tracks:
            params['seed_tracks'] = ','.join(seed_tracks[:5])
        if seed_artists:
            params['seed_artists'] = ','.join(seed_artists[:5])
        if seed_genres:
            params['seed_genres'] = ','.join(seed_genres[:5])
        
        #add any additional tuning parameters
        params.update(kwargs)
        
        data = self._make_request("recommendations", params=params)
        return data.get('tracks', [])
    
    #Audio Features (used by discovered_on_recommender.py)
    
    def get_audio_features(self, track_id: str) -> Dict:
        """Get audio features for a track.Returns a dict with audio features (energy, danceability, valence, etc.)"""
        return self._make_request(f"audio-features/{track_id}")
    
    def get_multiple_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Get audio features for multiple tracks (batch req).Takes a list of Spotify track IDs (max 100) and returns list of audio feature dicts"""
        #spotify allows max 100 IDs per request
        track_ids = track_ids[:100]
        params = {'ids': ','.join(track_ids)}
        data = self._make_request("audio-features", params=params)
        return data.get('audio_features', [])
    
    #Utility Methods
    
    def extract_artist_id(self, track_info: Dict) -> str:
        """Extract primary artist ID from track info.Takes the track object from API,returns artist ID string or empty string if not found"""
        try:
            return track_info['artists'][0]['id']
        except (KeyError, IndexError):
            return ""
    
    def extract_artist_name(self, track_info: Dict) -> str:
        """Extract primary artist name from track info.From track object from API,returns artist name string or "Unknown" if not found"""
        try:
            return track_info['artists'][0]['name']
        except (KeyError, IndexError):
            return "Unknown"


#Usage Examples

if __name__ == "__main__":
    """Quick test of the unified SpotifyAPI"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    #initialize
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
        exit(1)
    
    spotify = SpotifyAPI(client_id, client_secret)
    
    #test 1: get track info
    print("TEST 1: Get Track Info")
    
    test_url = input("Enter Spotify track URL: ").strip()
    if test_url:
        track_id = spotify.extract_track_id(test_url)
        track_info = spotify.get_track_info(track_id)
        
        if track_info:
            print(f"\n Track: {track_info['name']}")
            print(f"   Artist: {track_info['artists'][0]['name']}")
            print(f"   Album: {track_info['album']['name']}")
            print(f"   Popularity: {track_info['popularity']}/100")
            
            #test 2: get artist's top tracks
            print("TEST 2: Artist's Top Tracks")
            
            artist_id = spotify.extract_artist_id(track_info)
            top_tracks = spotify.get_artist_top_tracks(artist_id, limit=5)
            
            print(f"\nTop 5 tracks by {track_info['artists'][0]['name']}:")
            for i, track in enumerate(top_tracks, 1):
                print(f"{i}. {track['name']}")
            
            #test 3:search playlists
            print("TEST 3: Search Playlists")
            
            artist_name = track_info['artists'][0]['name']
            playlists = spotify.search_playlists(artist_name, limit=5)
            
            print(f"\nPlaylists featuring {artist_name}:")
            for i, pl in enumerate(playlists, 1):
                owner = pl.get('owner', {}).get('display_name', 'Unknown')
                print(f"{i}. {pl['name']} by {owner}")
        else:
            print(" Could not fetch track info")
    
    print(" Unified SpotifyAPI test complete!")