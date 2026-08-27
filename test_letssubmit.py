import requests
import os
from typing import Optional, Dict, Union
from dotenv import load_dotenv

load_dotenv()


class LetsSubmitAPI:
    """Let's Submit AI Music Checker API Client .This implementation uses 'fileurl' as shown in working examples."""
    
    def __init__(self, api_key: str = None):
        """Initialize the API client"""
        self.api_key = api_key or os.getenv('LETSSUBMIT_API_KEY')
        self.base_url = 'https://api.letssubmit.com'
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Set LETSSUBMIT_API_KEY environment variable "
                "or pass it directly to LetsSubmitAPI(api_key='your_key')"
            )
    
    def analyze_audio_file(self, file_url: str) -> Optional[Dict]:
        """Analyze an audio file from URL.Takes as args file_url:direct URL to MP3 or WAV audio file
        And returns dict: {'ai_probability': 85} or None if error"""
        endpoint = f"{self.base_url}/analyze_song"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        #we use 'fileurl' (no underscore) as per their working examples
        data = {
            'fileurl': file_url
        }
        
        try:
            print(f"Analyzing audio file: {file_url}")
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                ai_prob = result.get('ai_probability')
                
                #handle the case where API returns "-" or other non-numeric values
                if self._is_invalid_probability(ai_prob):
                    print(f" Analysis unavailable (no audio preview or analysis failed)")
                    return None
                
                print(f"Analysis complete: AI Probability = {ai_prob}%")
                return result
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timed out after 60 seconds")
            return None
        except requests.exceptions.RequestException as e:
            print(f" Request failed: {str(e)}")
            return None
    
    def analyze_spotify_track(self, spotify_url: str) -> Optional[Dict]:
        """Analyze a Spotify track (uses 30-second preview).Takes as args spotify_url and eturns dict: {'ai_probability': 85} or None if error"""
        endpoint = f"{self.base_url}/analyze_song"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        #we use 'spotify_url' as per JSON spec
        data = {
            'url': spotify_url
        }
        
        try:
            print(f"Analyzing Spotify track: {spotify_url}")
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                ai_prob = result.get('ai_probability')
                
                #handle the case where API returns "-" or other non-numeric values
                if self._is_invalid_probability(ai_prob):
                    print(f"Analysis unavailable (track may not have a preview)")
                    return None
                
                print(f"Analysis complete: AI Probability = {ai_prob}%")
                return result
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timed out after 60 seconds")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None
    
    def _is_invalid_probability(self, ai_prob) -> bool:
        """Check if the AI probability value is invalid"""
        if ai_prob is None:
            return True
        
        #check if it's the string "-"
        if ai_prob == "-":
            return True
        
        #try to convert to float to validate it's numeric
        try:
            if isinstance(ai_prob, str):
                float(ai_prob)
            return False
        except (ValueError, TypeError):
            return True


#AI Detection Service

class AIDetectionService:
    """Unified AI detection service with corrected Let's Submit integration"""
    
    def __init__(self):
        self.letssubmit_key = os.getenv('LETSSUBMIT_API_KEY')
        
        #initialize Let's Submit API if key is available
        if self.letssubmit_key:
            self.letssubmit = LetsSubmitAPI(self.letssubmit_key)
        else:
            self.letssubmit = None
            print("LETSSUBMIT_API_KEY not set -AI audio detection will use mock data")
    
    def detect_ai_audio(self, audio_url: str = None, spotify_url: str = None) -> Dict:
        """Detect AI-generated audio using Let's Submit API.Args:audio_url,spotify_url.
        Returns:
            {
                'ai_score': float (0.0 to 1.0),
                'confidence': float,
                'details': dict,
                'status': str
            }
        """
        #if no API configured, return mock data
        if not self.letssubmit:
            return self._mock_audio_detection()
        
        try:
            result = None
            
            #try Spotify URL first 
            if spotify_url:
                result = self.letssubmit.analyze_spotify_track(spotify_url)
            #fallback to direct audio URL
            elif audio_url:
                result = self.letssubmit.analyze_audio_file(audio_url)
            
            if result and 'ai_probability' in result:
                ai_prob = result['ai_probability']
                
                #convert to float if it's a string
                if isinstance(ai_prob, str):
                    ai_prob = float(ai_prob)
                
                #convert percentage (0-100) to decimal (0.0-1.0)
                ai_probability = ai_prob / 100.0
                
                return {
                    'ai_score': ai_probability,
                    'confidence': 0.9,  #API doesn't return confidence, using default
                    'details': result,
                    'status': 'success',
                    'raw_response': result
                }
            else:
                print("API returned unexpected format, using mock data")
                return self._mock_audio_detection()
                
        except Exception as e:
            print(f"Audio detection error: {str(e)}")
            return self._mock_audio_detection()
    
    def _mock_audio_detection(self) -> Dict:
        """Mock audio detection for testing when API unavailable"""
        return {
            'ai_score': 0.15,
            'confidence': 0.75,
            'details': {'method': 'mock', 'reason': 'API unavailable or not configured'},
            'status': 'mock'
        }
    


# SIMPLE TEST SCRIPT

def test_api():
    """Quick test of the API"""
    
    #replace with actual API key 
    api_key = os.getenv('LETSSUBMIT_API_KEY', 'YOUR_API_KEY_FROM_EMAIL')
    
    if api_key == 'YOUR_API_KEY_FROM_EMAIL':
        print("Please set your API key!")
        print("Option 1: Set environment variable")
        print("  Windows: set LETSSUBMIT_API_KEY=your_key_here")
        print("  Linux/Mac: export LETSSUBMIT_API_KEY=your_key_here")
        print("\nOption 2: Replace 'YOUR_API_KEY_FROM_EMAIL' in the code")
        return
    
    print("Testing Let'sSubmit API")
    
    #initialize API
    api = LetsSubmitAPI(api_key)
    
    #test with multiple tracks to find one that works
    test_tracks = [
        ("AI song 1", "https://open.spotify.com/track/1peC47aAyOLMMwx5drCc3f?si=ab83a33ab7644635"),
        ("AI song 2", "https://open.spotify.com/track/4LcFN2JzEOZzU0IX7eVkdT?si=8eaa02d0aac74ef8"),
        ("non AI song", "https://open.spotify.com/track/5EQzuYfTZt7B2LqlvTF49l?si=3407ffbb7e824c2e")
    ]
    
    successful_tests = 0
    
    for i, (track_name, spotify_url) in enumerate(test_tracks, 1):
        print(f"\nTest {i}: {track_name}")
        
        result = api.analyze_spotify_track(spotify_url)
        
        if result:
            #if we got a result, it means the AI probability is valid
            ai_prob = result.get('ai_probability')
            ai_prob_num = float(ai_prob) if isinstance(ai_prob, str) else ai_prob
            
            print(f"\n Result: {ai_prob_num}% AI-generated")
            print(f"Verdict: {'LIKELY AI' if ai_prob_num > 50 else 'LIKELY HUMAN'}")
            successful_tests += 1
        else:
            print("Analysis failed or no preview available for this track")
    
    print(f"Test complete! {successful_tests}/{len(test_tracks)} tracks successfully analyzed")
    
    if successful_tests == 0:
        print("\n Note: Spotify tracks without 30-second previews cannot be analyzed.")
        print("Try testing with a direct MP3/WAV URL instead:")


if __name__ == "__main__":
    test_api()