import requests
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import time


class ImprovedAcousticBrainz:
    """AcousticBrainz wrapper using BOTH endpoints:
    - Low-level: MFCCs, BPM, key, spectral features
    - High-level: Mood, genre, danceability, voice, acoustic"""
    
    def __init__(self, cache_file: str = "data/ab_features_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.base_url = "https://acousticbrainz.org/api/v1"
        
        #stats
        self.api_calls = 0
        self.cache_hits = 0
        
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        self.cache_file.parent.mkdir(exist_ok=True, parents=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_complete_features(self, mbid: str, verbose: bool = False) -> Optional[Dict]:
        """
        Get COMPLETE feature set (low-level + high-level).Returns features like mood_party,danceability,etc."""
        
        #check cache
        if mbid in self.cache:
            self.cache_hits += 1
            if verbose:
                print(f"   [Cached]")
            return self.cache[mbid]
        
        #fetch from API
        self.api_calls += 1
        
        features = {}
        
        #1 Get low-level features
        low_level = self._fetch_low_level(mbid, verbose)
        if low_level:
            features.update(low_level)
        else:
            #no features available
            self.cache[mbid] = None
            self._save_cache()
            return None
        
        #2 Get high-level features 
        high_level = self._fetch_high_level(mbid, verbose)
        if high_level:
            features.update(high_level)
        
        #cache complete features
        self.cache[mbid] = features
        self._save_cache()
        
        return features
    
    def _fetch_low_level(self, mbid: str, verbose: bool = False) -> Optional[Dict]:
        """Fetch low-level features (MFCCs, BPM, key, spectral)"""
        
        url = f"{self.base_url}/{mbid}/low-level"
        
        try:
            if verbose:
                print(f"   [API] Getting low-level features...", end='')
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                features = self._parse_low_level(data)
                
                if verbose:
                    print("OK")
                
                return features
            
            elif response.status_code == 404:
                if verbose:
                    print("Not in database")
                return None
            
            else:
                if verbose:
                    print(f" Error {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            if verbose:
                print(" Timeout")
            return None
        
        except Exception as e:
            if verbose:
                print(f" Error: {e}")
            return None
    
    def _fetch_high_level(self, mbid: str, verbose: bool = False) -> Optional[Dict]:
        """Fetch high-level features (mood, genre, danceability, etc.)"""
        
        url = f"{self.base_url}/{mbid}/high-level"
        
        try:
            if verbose:
                print(f"   [API] Getting high-level features...", end='')
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                features = self._parse_high_level(data)
                
                if verbose:
                    print(" OK")
                
                return features
            
            else:
                if verbose:
                    print(" Not available")
                return {}
                
        except Exception as e:
            if verbose:
                print(f" Error")
            return {}
    
    def _parse_low_level(self, data: Dict) -> Dict:
        """Parse low-level features"""
        
        features = {}
        
        try:
            lowlevel = data.get('lowlevel', {})
            rhythm = data.get('rhythm', {})
            tonal = data.get('tonal', {})
            
            # MFCCs (timbre/texture) -13 coefficients
            mfcc = lowlevel.get('mfcc', {})
            mfcc_mean = mfcc.get('mean', [])
            for i, val in enumerate(mfcc_mean[:13]):
                features[f'mfcc_{i+1}'] = float(val)
            
            #spectral features (brightness)
            features['spectral_centroid'] = float(
                lowlevel.get('spectral_centroid', {}).get('mean', 0)
            )
            features['spectral_rolloff'] = float(
                lowlevel.get('spectral_rolloff', {}).get('mean', 0)
            )
            
            #rhythm
            features['bpm'] = float(rhythm.get('bpm', 0))
            
            #tonal
            features['key'] = tonal.get('key_key', '')
            features['scale'] = tonal.get('key_scale', '')
            
            #energy
            features['loudness'] = float(lowlevel.get('average_loudness', 0))
            
        except Exception as e:
            pass
        
        return features
    
    def _parse_high_level(self, data: Dict) -> Dict:
        """Parse high-level features"""
        
        features = {}
        
        try:
            highlevel = data.get('highlevel', {})
            
            #multiple mood classifications
            mood_acoustic = highlevel.get('mood_acoustic', {})
            features['mood'] = mood_acoustic.get('value', '')
            features['mood_probability'] = float(mood_acoustic.get('probability', 0))
            
            #get all mood probabilities if available
            mood_all = mood_acoustic.get('all', {})
            for mood_type, prob in mood_all.items():
                features[f'mood_{mood_type}'] = float(prob)
            
            #primary genre and subgenre
            genre_rosamerica = highlevel.get('genre_rosamerica', {})
            features['genre'] = genre_rosamerica.get('value', '')
            features['genre_probability'] = float(genre_rosamerica.get('probability', 0))
            
            #alternative genre classifications
            genre_electronic = highlevel.get('genre_electronic', {})
            features['genre_electronic'] = genre_electronic.get('value', '')
            
            genre_dortmund = highlevel.get('genre_dortmund', {})
            features['genre_dortmund'] = genre_dortmund.get('value', '')
            
            #danceability
            danceability = highlevel.get('danceability', {})
            features['danceability'] = danceability.get('value', '')
            features['danceability_probability'] = float(danceability.get('probability', 0))
            
            #voice vs instrumental
            voice_instrumental = highlevel.get('voice_instrumental', {})
            features['voice_instrumental'] = voice_instrumental.get('value', '')
            features['voice_instrumental_probability'] = float(
                voice_instrumental.get('probability', 0)
            )
            
            #gender
            gender = highlevel.get('gender', {})
            features['gender'] = gender.get('value', '')
            features['gender_probability'] = float(gender.get('probability', 0))
            
            #acoustic vs electronic
            acoustic_electronic = highlevel.get('mood_electronic', {})
            features['acoustic_electronic'] = acoustic_electronic.get('value', '')
            
            #timbre (bright/dark)
            timbre = highlevel.get('timbre', {})
            features['timbre'] = timbre.get('value', '')
            
            #mood party
            mood_party = highlevel.get('mood_party', {})
            features['mood_party'] = mood_party.get('value', '')
            features['mood_party_probability'] = float(mood_party.get('probability', 0))
            
            #mood aggresive
            mood_aggressive = highlevel.get('mood_aggressive', {})
            features['mood_aggressive'] = mood_aggressive.get('value', '')
            features['mood_aggressive_probability'] = float(
                mood_aggressive.get('probability', 0)
            )
            
            #mood happy
            mood_happy = highlevel.get('mood_happy', {})
            features['mood_happy'] = mood_happy.get('value', '')
            features['mood_happy_probability'] = float(mood_happy.get('probability', 0))
            
            #mood relaxed
            mood_relaxed = highlevel.get('mood_relaxed', {})
            features['mood_relaxed'] = mood_relaxed.get('value', '')
            features['mood_relaxed_probability'] = float(
                mood_relaxed.get('probability', 0)
            )
            
        except Exception as e:
            pass
        
        return features
    
    def get_feature_vector(self, mbid: str, verbose: bool = False) -> Optional[np.ndarray]:
        """Get feature vector for similarity computation.Uses weighted combination of high and low level features."""
        
        features = self.get_complete_features(mbid, verbose)
        
        if not features:
            return None
        
        #build feature vector in order of importance
        vector = []
        
        #high-level features
        
        #mood probabilities 
        vector.append(features.get('mood_happy_probability', 0))
        vector.append(features.get('mood_party_probability', 0))
        vector.append(features.get('mood_aggressive_probability', 0))
        vector.append(features.get('mood_relaxed_probability', 0))
        
        #danceability
        vector.append(features.get('danceability_probability', 0))
        
        # voice/instrumental (0 = instrumental, 1 = voice)
        voice_inst = 1.0 if features.get('voice_instrumental') == 'voice' else 0.0
        vector.append(voice_inst)
        
        #gender (0 =female, 1=male) 
        gender_val = 1.0 if features.get('gender') == 'male' else 0.0
        vector.append(gender_val)
        
        #acoustic/electronic (0 =acoustic, 1 = electronic)
        acoustic_val = 1.0 if features.get('acoustic_electronic') == 'electronic' else 0.0
        vector.append(acoustic_val)
        
        #low-level features
        
        #BPM (normalized to 0-1 range, assuming 60-180 BPM)
        bpm = features.get('bpm', 120)
        bpm_normalized = (bpm - 60) / 120  # 60-180 → 0-1
        vector.append(bpm_normalized)
        
        #key (convert to numeric 0-11)
        key_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                   'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        key_val = key_map.get(features.get('key', 'C'), 0) / 11  # Normalize to 0-1
        vector.append(key_val)
        
        #scale (0 =minor, 1 =major)
        scale_val = 1.0 if features.get('scale') == 'major' else 0.0
        vector.append(scale_val)
        
        #MFCCs (first 5 only, normalized)
        for i in range(1, 6):
            mfcc_val = features.get(f'mfcc_{i}', 0)
            #MFCCs typically range -100 to 100, normalize roughly
            mfcc_normalized = (mfcc_val + 100) / 200
            vector.append(mfcc_normalized)
        
        #spectral features (normalized)
        spectral_cent = features.get('spectral_centroid', 1500)
        spectral_normalized = spectral_cent / 5000  #rough normalization
        vector.append(spectral_normalized)
        
        return np.array(vector)
    
    def print_stats(self):
        """Print API usage statistics"""
        print(f"\nAPI Statistics:")
        print(f"  API calls: {self.api_calls}")
        print(f"  Cache hits: {self.cache_hits}")
        print(f"  Cached tracks: {len([k for k, v in self.cache.items() if v is not None])}")


def test():
    """Test with verbose output"""
    
    print("TESTING ACOUSTICBRAINZ (HIGH-LEVEL FEATURES)")
    
    ab = ImprovedAcousticBrainz()
    
    #test MBID (Beatles - Hey Jude)
    test_mbid = "6b9dfb02-8b3e-4cc5-abc8-c85bfa4a1e3a"
    
    print(f"\nTest track: Beatles - Hey Jude")
    print(f"MBID: {test_mbid}\n")
    
    #get complete features
    features = ab.get_complete_features(test_mbid, verbose=True)
    
    if features:
        print("\n" + "=" * 70)
        print("COMPLETE FEATURE SET")
        print("=" * 70)
        
        print("\nHIGH-LEVEL FEATURES (for recommendations):")
        print(f"  Mood: {features.get('mood', 'N/A')}")
        print(f"  Mood Happy: {features.get('mood_happy', 'N/A')} ({features.get('mood_happy_probability', 0):.2f})")
        print(f"  Mood Party: {features.get('mood_party', 'N/A')} ({features.get('mood_party_probability', 0):.2f})")
        print(f"  Mood Aggressive: {features.get('mood_aggressive', 'N/A')}")
        print(f"  Genre: {features.get('genre', 'N/A')}")
        print(f"  Danceability: {features.get('danceability', 'N/A')} ({features.get('danceability_probability', 0):.2f})")
        print(f"  Voice/Instrumental: {features.get('voice_instrumental', 'N/A')}")
        print(f"  Gender: {features.get('gender', 'N/A')}")
        print(f"  Acoustic/Electronic: {features.get('acoustic_electronic', 'N/A')}")
        print(f"  Timbre: {features.get('timbre', 'N/A')}")
        
        print("\nLOW-LEVEL FEATURES:")
        print(f"  BPM: {features.get('bpm', 0):.1f}")
        print(f"  Key: {features.get('key', 'N/A')} {features.get('scale', '')}")
        print(f"  Loudness: {features.get('loudness', 0):.2f}")
        print(f"  Spectral Centroid: {features.get('spectral_centroid', 0):.2f}")
        
        #get feature vector
        vector = ab.get_feature_vector(test_mbid)
        print(f"\nFeature vector for similarity: {len(vector)} dimensions")
        print(f"  (Mood, danceability, voice, gender, BPM, key, timbre...)")
        
        ab.print_stats()
        
    else:
        print("\nNo features available")
    


if __name__ == "__main__":
    test()