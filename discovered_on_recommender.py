import os
import requests
import csv
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import time

#import unified SpotifyAPI
import sys
sys.path.append(str(Path(__file__).parent / 'apis'))
from apis.spotify_client import SpotifyAPI
from blacklist import AIArtistBlacklist

load_dotenv()

#AI DETECTION SERVICE
class LetsSubmitAPI:
    """Let's Submit AI Music Checker API Client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.letssubmit.com'
    
    def analyze_spotify_track(self, spotify_url: str) -> Optional[Dict]:
        """Analyze a Spotify track for AI generation"""
        endpoint = f"{self.base_url}/analyze_song"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {'spotify_url': spotify_url}
        
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                ai_prob = result.get('ai_probability')
                
                if ai_prob is None or ai_prob == "-":
                    return None
                
                if isinstance(ai_prob, str):
                    ai_prob = float(ai_prob)
                
                return {'ai_probability': ai_prob}
            else:
                return None
                
        except Exception:
            return None


#ENHANCED DISCOVERED ON RECOMMENDER 

class DiscoveredOnRecommender:
    """Find tracks from user-curated playlists where the artist was "discovered" """
    
    def __init__(self, spotify_api: SpotifyAPI, ai_detector: LetsSubmitAPI):
        self.spotify = spotify_api
        self.ai_detector = ai_detector
        self.blacklist = AIArtistBlacklist()
        
        #session cache: normalized artist_name -> (is_ai_artist, ai_percentage)
        self.session_checked_artists = {}
    
    def check_artist_discography(
        self, 
        artist_id: str, 
        artist_name: str,
        ai_threshold: float = 60.0,
        sample_size: int = 10,
        rate_limit_delay: float = 1.0
    ) -> Tuple[bool, float]:
        """Sample artist's discography; if at least ai_threshold% of the sampled
        tracks come back AI, treat the whole artist as AI. Cached per session so the
        same artist isn't sampled twice.Returns (is_ai_artist, ai_percentage)."""
        #normalize artist name for cache lookup
        normalized_name = self.blacklist._normalize_artist_name(artist_name)
        
        #check if we already analyzed artist in current session
        if normalized_name in self.session_checked_artists:
            cached_result = self.session_checked_artists[normalized_name]
            is_ai, percentage = cached_result
            print(f"\n Using cached result for '{artist_name}': "
                  f"{'AI' if is_ai else 'HUMAN'} ({percentage:.1f}%)")
            return is_ai, percentage
        
        print(f"\n Analyzing '{artist_name}' discography...")
        
        #get artist's top tracks (most popular = best sample)
        top_tracks = self.spotify.get_artist_top_tracks(artist_id, limit=sample_size)
        
        if not top_tracks:
            print(f"  Could not get discography")
            #cache negative result
            self.session_checked_artists[normalized_name] = (False, 0.0)
            return False, 0.0
        
        #check each track
        ai_count = 0
        total_checked = 0
        
        print(f"       Checking {len(top_tracks)} tracks from discography...")
        
        for track in top_tracks:
            track_url = track.get('external_urls', {}).get('spotify', '')
            track_name = track.get('name', 'Unknown')
            
            if not track_url:
                continue
            
            #run AI detection
            ai_result = self.ai_detector.analyze_spotify_track(track_url)
            
            if ai_result is None:
                continue  #skip tracks we can't analyze
            
            total_checked += 1
            ai_probability = ai_result['ai_probability']
            
            #consider AI if >40% 
            if ai_probability > 40.0:
                ai_count += 1
                print(f" ✘ {track_name}: {ai_probability:.1f}% AI")
            else:
                print(f" ✔ {track_name}: {ai_probability:.1f}% AI")
            
            time.sleep(rate_limit_delay)
        
        if total_checked == 0:
            print(f" Could not analyze any tracks")
            #cache negative result
            self.session_checked_artists[normalized_name] = (False, 0.0)
            return False, 0.0
        
        #check if sample size is sufficient
        if total_checked < 5:
            print(f" Sample too small ({total_checked} tracks), skipping artist level judgment")
            #cache as human (benefit of the doubt with small sample)
            self.session_checked_artists[normalized_name] = (False, 0.0)
            return False, 0.0
        
        #calculate AI percentage
        ai_percentage = (ai_count / total_checked) * 100
        
        print(f"\n RESULT: {ai_count}/{total_checked} tracks are AI ({ai_percentage:.1f}%)")
        
        #determine if artist is AI
        is_ai_artist = ai_percentage >= ai_threshold
        
        if is_ai_artist:
            print(f" ✘ Artist is AI-GENERATED (>= {ai_threshold}% threshold)")
        else:
            print(f" ✔ Artist appears HUMAN-MADE (< {ai_threshold}% threshold)")
        
        #cache the result for this session
        self.session_checked_artists[normalized_name] = (is_ai_artist, ai_percentage)
        
        return is_ai_artist, ai_percentage
    
    def get_discovered_on_recommendations(
        self,
        track_url: str,
        ai_threshold: float = 35.0,
        original_track_ai_threshold: float = 39.0,
        artist_discography_ai_threshold: float = 60.0,
        max_playlists_to_check: int = 35,
        max_recommendations: int = 5,
        rate_limit_delay: float = 1.0,
        skip_first_n_playlists: int = 1,
        discography_sample_size: int = 10
    ) -> List[Dict]:
        """Find human-made recommendations for a Spotify track by mining the'Discovered On' playlists of its artist.
        Each is filtered per-track (LetsSubmit) and per-artist (discography sampling).
        Returns list of recommendation dicts with track metadata, AI probability,playlist each was discovered in."""
        print("\n" + "=" * 70)
        print("DISCOVERED ON RECOMMENDER ")
        print("=" * 70)
        
        #Step 1: Get original track info
        track_id = self.spotify.extract_track_id(track_url)
        track_info = self.spotify.get_track_info(track_id)
        
        if not track_info:
            print("Could not get track info")
            return []
        
        original_artist_name = self.spotify.extract_artist_name(track_info)
        original_artist_id = self.spotify.extract_artist_id(track_info)
        track_name = track_info['name']
        
        print(f"\n Original Track: {track_name}")
        print(f" Artist: {original_artist_name}")
        
        #Step 1.5: Check if original track is AI-generated
        print("\n Checking if original track is AI-generated...")
        original_ai_result = self.ai_detector.analyze_spotify_track(track_url)
        
        if original_ai_result:
            original_ai_prob = original_ai_result['ai_probability']
            print(f" Original Track AI Probability: {original_ai_prob:.1f}%")
            
            if original_ai_prob < original_track_ai_threshold:
                print(f" Original track appears HUMAN-MADE (< {original_track_ai_threshold}%)")
                print("No need to filter - this is authentic music!")
                return []
            else:
                print(f"Original track is likely AI-GENERATED (>= {original_track_ai_threshold}%)")
                print("Proceeding to find human-made alternatives...")
        else:
            print("Could not analyze original track")
            print("Proceeding anyway...")
                
        #Direction 2: collect playlists from multiple search variants
        MIN_USABLE_PLAYLISTS = 5
        search_queries = [original_artist_name]
        
        print(f"\nSearching for playlists featuring '{original_artist_name}'...")
        seen_playlist_ids = set()
        all_playlists = []
        
        def _add_unique(playlist_list):
            """Merge a search result list into all_playlists, dedup by id."""
            added = 0
            for pl in (playlist_list or []):
                if pl is None:
                    continue
                pid = pl.get('id')
                if not pid or pid in seen_playlist_ids:
                    continue
                seen_playlist_ids.add(pid)
                all_playlists.append(pl)
                added += 1
            return added
        
        #initial search
        initial = self.spotify.search_playlists(original_artist_name, limit=50)
        n0 = _add_unique(initial)
        print(f"  Initial query '{original_artist_name}': {n0} unique playlists")
        
        #apply the filter so far to see how many usable we have
        filtered_so_far = self._filter_playlists(
            all_playlists,
            original_artist_id,
            original_artist_name,
        )
        
        #if too few survive the filter we try broader queries
        if len(filtered_so_far) < MIN_USABLE_PLAYLISTS:
            print(f"  Only {len(filtered_so_far)} usable so far; broadening search...")
            for suffix in ("playlist", "mix", "radio"):
                if len(filtered_so_far) >= MIN_USABLE_PLAYLISTS:
                    break
                q = f"{original_artist_name} {suffix}"
                extra = self.spotify.search_playlists(q, limit=30)
                n = _add_unique(extra)
                print(f"  Broader query '{q}': {n} new unique playlists")
                if n == 0:
                    continue
                #re-filter the cumulative set so we know when to stop
                filtered_so_far = self._filter_playlists(
                    all_playlists,
                    original_artist_id,
                    original_artist_name,
                )
                print(f" Cumulative usable after filter: {len(filtered_so_far)}")
        
        if not all_playlists:
            print("No playlists found through any search variant.")
            return []
        
        print(f"Found {len(all_playlists)} total unique playlists across queries")
        
        #final filter pass on the merged set
        filtered_playlists = self._filter_playlists(
            all_playlists,
            original_artist_id,
            original_artist_name,
        )
        
        if not filtered_playlists:
            print("No playlists survived the filter.")
            #fall through to genre fallback below - don't return [] yet.
            filtered_playlists = []
        
        print(f"{len(filtered_playlists)} playlists to analyse after filtering")
        
        if not filtered_playlists:
            print("No valid playlists found after filtering")
            return []
        
        print(f" {len(filtered_playlists)} playlists to check")
        
        #show blacklist stats
        blacklist_count = self.blacklist.get_blacklist_count()
        print(f"\n Current AI Artist Blacklist: {blacklist_count} artists")
        
        #Step 4: Analyze playlists with discography checking
        recommendations = self._analyze_playlists_with_discography_check(
            filtered_playlists=filtered_playlists,
            original_artist_name=original_artist_name.lower(),
            original_track_id=track_id,
            ai_threshold=ai_threshold,
            artist_discography_ai_threshold=artist_discography_ai_threshold,
            max_playlists=max_playlists_to_check,
            max_recommendations=max_recommendations,
            rate_limit_delay=rate_limit_delay,
            discography_sample_size=discography_sample_size
        )
        
        print("\n" + "=" * 70)
        print(f"FOUND {len(recommendations)} HUMAN-MADE RECOMMENDATIONS !")
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print(f"FOUND {len(recommendations)} HUMAN-MADE RECOMMENDATIONS (Discovered-On)")
        print("=" * 70)
        
        #Direction 3: genre fallback if Discovered-On found nothing
        if not recommendations:
            print("\nDiscovered-On returned nothing. Falling back to genre-based search...")
            genre_recs = self._genre_fallback_recommendations(
                original_artist_id=original_artist_id,
                original_artist_name=original_artist_name,
                original_track_id=track_id,
                ai_threshold=ai_threshold,
                artist_discography_ai_threshold=artist_discography_ai_threshold,
                max_recommendations=max_recommendations,
                rate_limit_delay=rate_limit_delay,
                discography_sample_size=discography_sample_size,
            )
            if genre_recs:
                print(f"Genre fallback recovered {len(genre_recs)} tracks")
                return genre_recs
        
        return recommendations
    
    def _filter_playlists(
        self,
        playlists,
        artist_id,
        artist_name,
        skip_first_n=0,        #kept for backward compatibility, default 0 now
    ):
        """Drop playlists we shouldn't consume: null entries, the query artist's
        own playlists, Spotify-curated 'This Is <Artist>' showcases (single-artist
        and unreadable from a dev-mode app), and playlists whose owner's display
        name matches the artist (artist running their own account)."""
        filtered = []
        normalized_artist = self.blacklist._normalize_artist_name(artist_name)
        artist_name_lower = (artist_name or "").lower().strip()
        this_is_artist = f"this is {artist_name_lower}"
    
        for i, playlist in enumerate(playlists):
            if playlist is None:
                continue
    
            #honor explicit skip_first_n (default 0 = no blunt skip)
            if i < skip_first_n:
                continue
    
            owner = playlist.get('owner')
            if owner is None:
                continue
    
            owner_id = owner.get('id', '') or ''
            owner_name = owner.get('display_name', '') or ''
            pl_name = (playlist.get('name', '') or '').strip()
            pl_name_lower = pl_name.lower()
    
            #1 skip the artist's own playlists (owner = the artist themselves)
            if owner_id and owner_id == artist_id:
                continue
    
            #2 skip user playlists where the owner's display name matches the artist 
            normalized_owner = self.blacklist._normalize_artist_name(owner_name)
            if normalized_owner and normalized_owner == normalized_artist:
                print(f"   [skip] '{pl_name}' - user '{owner_name}' matches artist name")
                continue
    
            # 3. skip "This Is <Artist>" editorial showcases 
            if owner_id.lower() == "spotify" and pl_name_lower == this_is_artist:
                continue
    
            # 4. skip other spotify-owned playlists(return empty in dev mode, so calling them just burns API calls)
            if owner_id.lower() == "spotify":
                #the contents won't load anyway so we skip silently
                print(f"   [skip] '{pl_name}' - Spotify-owned (not readable in dev mode)")
                continue
    
            filtered.append(playlist)
    
        return filtered
    
    def _analyze_playlists_with_discography_check(
        self,
        filtered_playlists: List[Dict],
        original_artist_name: str,
        original_track_id: str,
        ai_threshold: float,
        artist_discography_ai_threshold: float,
        max_playlists: int,
        max_recommendations: int,
        rate_limit_delay: float,
        discography_sample_size: int
    ) -> List[Dict]:
        """Analyze playlists with intelligent artist discography checking"""
        all_recommendations = []
        seen_track_ids = {original_track_id}
        playlists_checked = 0
        playlists_with_artist = 0
        
        print("\n" + "-" * 70)
        print("ANALYZING PLAYLISTS")
        print("-" * 70)
        
        for playlist in filtered_playlists[:max_playlists]:
            if len(all_recommendations) >= max_recommendations:
                break
            
            playlists_checked += 1
            
            playlist_name = playlist['name']
            playlist_id = playlist['id']
            owner_name = playlist.get('owner', {}).get('display_name', 'Unknown')
            
            print(f"\n{playlists_checked}. '{playlist_name}' by {owner_name}")
            
            #get all tracks
            tracks = self.spotify.get_playlist_tracks(playlist_id)
            
            if not tracks:
                print("No tracks found, skipping...")
                continue
            
            #check if original artist appears
            contains_artist = self._playlist_contains_artist(
                tracks,
                original_artist_name
            )
            
            if not contains_artist:
                print(f"{original_artist_name} not found in this playlist")
                continue
            
            print(f" Contains {original_artist_name}!")
            playlists_with_artist += 1
            
            #get tracks NOT by original artist
            other_tracks = self._get_other_artists_tracks(
                tracks,
                original_artist_name,
                seen_track_ids
            )
            
            if not other_tracks:
                print("No other artists' tracks found")
                continue
            
            print(f"Found {len(other_tracks)} tracks by other artists")
            print(f"Checking tracks with discography analysis...")
            
            #process tracks with discography checking
            for track in other_tracks:
                if len(all_recommendations) >= max_recommendations:
                    break
                
                #extract track info
                track_id = track.get('id')
                track_name = track.get('name', 'Unknown')
                track_url = track.get('external_urls', {}).get('spotify', '')
                
                if not track_id or not track_url:
                    continue
                
                #get artist info
                track_artist = self.spotify.extract_artist_name(track)
                track_artist_id = self.spotify.extract_artist_id(track)
                
                if not track_artist_id:
                    continue
                
                #CHECK 1: Is this artist already blacklisted?
                if self.blacklist.is_blacklisted(track_artist):
                    print(f" {track_name} - {track_artist} (BLACKLISTED)")
                    continue
                
                # CHECK 2: New artist ->check their entire discography
                print(f"New artist detected: {track_artist}")
                
                is_ai_artist, ai_percentage = self.check_artist_discography(
                    artist_id=track_artist_id,
                    artist_name=track_artist,
                    ai_threshold=artist_discography_ai_threshold,
                    sample_size=discography_sample_size,
                    rate_limit_delay=rate_limit_delay
                )
                
                if is_ai_artist:
                    #blacklist artist
                    self.blacklist.add_to_blacklist(track_artist)
                    print(f"All future tracks from '{track_artist}' will be skipped")
                else:
                    #now check the specific track
                    ai_result = self.ai_detector.analyze_spotify_track(track_url)
                    
                    if ai_result is not None:
                        ai_probability = ai_result['ai_probability']
                        
                        if ai_probability <= ai_threshold:
                            recommendation = {
                                'name': track_name,
                                'artist': track_artist,
                                'url': track_url,
                                'id': track_id,
                                'ai_probability': ai_probability,
                                'discovered_on': playlist_name,
                                'popularity': track.get('popularity', 0)
                            }
                            
                            all_recommendations.append(recommendation)
                            seen_track_ids.add(track_id)
                            
                            print(f" {track_name} - {track_artist} (AI: {ai_probability:.1f}%)")
                        
                        time.sleep(rate_limit_delay)
            
            if len(all_recommendations) >= max_recommendations:
                print("\n Reached target number of recommendations!")
                break
        
        print("\n" + "-" * 70)
        print(f" SUMMARY:")
        print(f"   Playlists checked: {playlists_checked}")
        print(f"   Playlists with artist: {playlists_with_artist}")
        print(f"   Human-made tracks found: {len(all_recommendations)}")
        print(f"   Artists now blacklisted: {self.blacklist.get_blacklist_count()}")
        print(f"   Artists checked this session: {len(self.session_checked_artists)}")
        print("-" * 70)
        
        return all_recommendations
    
    def get_session_stats(self) -> Dict:
        """Get statistics about the current session"""
        stats = {
            'artists_checked': len(self.session_checked_artists),
            'ai_artists': sum(1 for is_ai, _ in self.session_checked_artists.values() if is_ai),
            'human_artists': sum(1 for is_ai, _ in self.session_checked_artists.values() if not is_ai),
            'blacklist_size': self.blacklist.get_blacklist_count()
        }
        return stats
    
    def print_session_stats(self):
        """Print detailed session statistics"""
        stats = self.get_session_stats()
        
        print("\n" + "=" * 70)
        print("SESSION STATISTICS")
        print("=" * 70)
        print(f"Artists analyzed this session: {stats['artists_checked']}")
        print(f"    Identified as AI: {stats['ai_artists']}")
        print(f"    Identified as Human: {stats['human_artists']}")
        print(f"Total in blacklist: {stats['blacklist_size']}")
        
        if self.session_checked_artists:
            print(f"\n Artists checked this session:")
            for artist_name, (is_ai, percentage) in self.session_checked_artists.items():
                status = "✘ AI" if is_ai else "✔ HUMAN"
                print(f"   {status} - {artist_name} ({percentage:.1f}%)")
        
        print("=" * 70)
    

    def _genre_fallback_recommendations(
        self,
        original_artist_id,
        original_artist_name,
        original_track_id,
        ai_threshold,
        artist_discography_ai_threshold,
        max_recommendations,
        rate_limit_delay,
        discography_sample_size,
    ):
        """Genre-based fallback: search a playlist named after artist's
        primary Spotify genre and walk it until max_recommendations are accepted
        (or MAX_CANDIDATES checked, or LetsSubmit returns 429)."""
        #tunable constants for this fallback path
        HUMAN_THRESHOLD_PCT = 30.0   # < accepted; >= is skipped
        MAX_CANDIDATES      = 30     # never check more than this many
    
        #1 Get artist genres
        try:
            artist_info = self.spotify.get_artist_info(original_artist_id)
        except Exception as e:
            print(f"  Could not fetch artist info for genre fallback: {e}")
            return []
        if not artist_info:
            return []
        genres = artist_info.get('genres') or []
        if not genres:
            print(f"  Artist '{original_artist_name}' has no genre tags on Spotify; "
                f"genre fallback unavailable.")
            return []
    
        primary_genre = genres[0]
        print(f"  Primary genre for '{original_artist_name}': {primary_genre}")
    
        #2 Search for playlists in that genre
        try:
            candidate_playlists = self.spotify.search_playlists(
                primary_genre, limit=10
            )
        except Exception as e:
            print(f"  Genre playlist search failed: {e}")
            return []
    
        filtered = self._filter_playlists(
            candidate_playlists,
            original_artist_id,
            original_artist_name,
        )
        if not filtered:
            print("  No playlists survived the filter for the genre fallback.")
            return []
    
        #load the LetsSubmit cache 
        from letssubmit_cache import LetsSubmitCache
        cache = getattr(self, '_letssubmit_cache', None)
        if cache is None:
            cache = LetsSubmitCache()
            self._letssubmit_cache = cache
    
        #3 Walk playlists in order, per-track LetsSubmit check
        accepted: list = []
        candidates_checked = 0
        quota_tripped_locally = False
        original_artist_lower = (original_artist_name or "").lower().strip()
        seen_track_ids = set()
        seen_artist_names = set()
        playlists_used = 0
    
        def check_track(spotify_track_url):
            """Returns (probability_or_None, hit_quota_429_bool)."""
            try:
                prob, status = cache.check(spotify_track_url)
            except Exception as e:
                print(f"      [letssubmit error] {e}")
                return None, False
            if status == "quota_exhausted":
                print(f"      [quota] LetsSubmit returned 429; halting per-track checks")
                return None, True
            return prob, False
    
        for playlist_rank, pl in enumerate(filtered[:2], start=1):
            if (len(accepted) >= max_recommendations
                    or candidates_checked >= MAX_CANDIDATES
                    or quota_tripped_locally):
                break
    
            pl_name = pl.get('name', '<unknown>')
            pl_id = pl.get('id')
            if not pl_id:
                continue
    
            print(f" Genre fallback: drawing from playlist #{playlist_rank}: '{pl_name}'")
            playlists_used = playlist_rank
    
            try:
                tracks = self.spotify.get_playlist_tracks(pl_id)
            except Exception as e:
                print(f" Could not read tracks from '{pl_name}': {e}")
                continue
            if not tracks:
                print(f" Playlist '{pl_name}' returned no tracks.")
                continue
    
            for track in tracks:
                #stop conditions checked at top of every iteration
                if (len(accepted) >= max_recommendations
                        or candidates_checked >= MAX_CANDIDATES
                        or quota_tripped_locally):
                    break
                if track is None:
                    continue
    
                t_id = track.get('id')
                if not t_id or t_id in seen_track_ids:
                    continue
                seen_track_ids.add(t_id)
    
                t_name = track.get('name') or ''
                artists = track.get('artists') or []
                if not artists:
                    continue
                t_artist = (artists[0] or {}).get('name') or ''
                t_artist_lower = t_artist.lower().strip()
    
                #cheap gates first (no quota cost)
                if t_artist_lower == original_artist_lower:
                    continue
                if t_artist_lower in seen_artist_names:
                    continue
                if self.blacklist.is_blacklisted(t_artist):
                    continue
    
                #per-track letsSubmit check
                spotify_url = f"https://open.spotify.com/track/{t_id}"
                print(f"    [check] {t_artist} - {t_name}")
                prob, hit_quota = check_track(spotify_url)
                candidates_checked += 1
                if hit_quota:
                    quota_tripped_locally = True
                    break
                if prob is None:
                    print(f"      -> unverified (skipped)")
                    continue
                if prob >= HUMAN_THRESHOLD_PCT:
                    print(f"      -> {prob:.1f}% AI (skipped, >= {HUMAN_THRESHOLD_PCT}%)")
                    continue
    
                #accept
                print(f"      -> {prob:.1f}% AI (ACCEPTED)")
                t_album = track.get('album') or {}
                album_images = t_album.get('images') or []
                artwork_url = album_images[0]['url'] if album_images else None
                accepted.append({
                    'track_id':            t_id,
                    'name':                t_name,
                    'title':               t_name,
                    'artist':              t_artist,
                    'album':               t_album.get('name', ''),
                    'spotify_url':         spotify_url,
                    'spotify_id':          t_id,
                    'embed_url':           f"https://open.spotify.com/embed/track/{t_id}",
                    'url':                 spotify_url,
                    'artwork':             artwork_url,
                    'platform':            'spotify',
                    'method':              'genre-fallback',
                    'fallback_genre':      primary_genre,
                    'fallback_playlist':   pl_name,
                    'verification_status': 'track-verified',
                    'ai_probability':      round(prob, 1),
                })
                seen_artist_names.add(t_artist_lower)
    
        #summary log
        if quota_tripped_locally:
            print(f" Genre fallback: halted on quota. Returning {len(accepted)} accepted "
                f"({candidates_checked} candidates checked) from {playlists_used} playlist(s).")
        elif not accepted:
            print(f"Genre fallback: checked {candidates_checked} candidates, "
                f"none scored under {HUMAN_THRESHOLD_PCT}% AI. Returning empty.")
        else:
            print(f"Genre fallback: returning {len(accepted)} accepted "
                f"({candidates_checked} candidates checked) from {playlists_used} playlist(s).")
    
        return accepted

    def _playlist_contains_artist(
        self,
        tracks: List[Dict],
        artist_name: str
    ) -> bool:
        """Check if playlist contains the specified artist"""
        artist_name_lower = artist_name.lower()
        
        for track in tracks:
            if not track or not isinstance(track, dict):
                continue
            
            track_artists = track.get('artists', [])
            if not track_artists:
                continue
            
            for artist in track_artists:
                if not artist or not isinstance(artist, dict):
                    continue
                
                artist_track_name = artist.get('name', '')
                if artist_track_name and artist_track_name.lower() == artist_name_lower:
                    return True
        
        return False
    
    def _get_other_artists_tracks(
        self,
        tracks: List[Dict],
        original_artist_name: str,
        seen_track_ids: set
    ) -> List[Dict]:
        """Get tracks from other artists"""
        other_tracks = []
        original_artist_lower = original_artist_name.lower()
        
        for track in tracks:
            if not track or not isinstance(track, dict):
                continue
            
            track_id = track.get('id')
            if not track_id or track_id in seen_track_ids:
                continue
            
            track_artists = track.get('artists', [])
            if not track_artists:
                continue
            
            #get all artist names
            track_artist_names = []
            for artist in track_artists:
                if artist and isinstance(artist, dict):
                    name = artist.get('name', '')
                    if name:
                        track_artist_names.append(name.lower())
            
            #skip if original artist is involved
            if original_artist_lower in track_artist_names:
                continue
            
            other_tracks.append(track)
        
        return other_tracks


#MAIN APPLICATION

def main():
    """Test the enhanced recommender """
    
    print("=" * 70)
    print("DISCOVERED ON RECOMMENDER -MANUAL TEST")
    print("=" * 70)
    
    #check configuration
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    letssubmit_api_key = os.getenv('LETSSUBMIT_API_KEY')
    
    if not spotify_client_id or not spotify_client_secret:
        print("\nError: Spotify credentials not set in .env file")
        return
    
    if not letssubmit_api_key:
        print("\nError: LETSSUBMIT_API_KEY not set in .env file")
        return
    
    #initialize APIs using unified SpotifyAPI
    spotify = SpotifyAPI(spotify_client_id, spotify_client_secret)
    ai_detector = LetsSubmitAPI(letssubmit_api_key)
    
    #initialize recommender
    recommender = DiscoveredOnRecommender(spotify, ai_detector)
    
    #interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nEnter Spotify track URLs to get recommendations")
    print("Type 'quit' to exit\n")
    
    while True:
        track_url = input(" Enter Spotify track URL (or 'quit'): ").strip()
        
        if track_url.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not track_url or 'spotify.com/track/' not in track_url:
            print("Invalid URL. Please enter a valid Spotify track URL.")
            continue
        
        try:
            recommendations = recommender.get_discovered_on_recommendations(
                track_url=track_url,
                ai_threshold=35.0,
                original_track_ai_threshold=39.0,
                artist_discography_ai_threshold=60.0,
                max_playlists_to_check=35,
                max_recommendations=5,
                rate_limit_delay=1.0,
                skip_first_n_playlists=1,
                discography_sample_size=10
            )
            
            if recommendations:
                print("\n" + "=" * 70)
                print(f" RECOMMENDATIONS FOUND: {len(recommendations)}")
                print("=" * 70)
                
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}.  {rec['name']}")
                    print(f" Artist: {rec['artist']}")
                    print(f" Popularity: {rec['popularity']}/100")
                    print(f" AI Probability: {rec['ai_probability']:.1f}%")
                    print(f" Found in playlist: '{rec['discovered_on']}'")
                    print(f"   {rec['url']}")
                
                #statistics
                avg_ai = sum(r['ai_probability'] for r in recommendations) / len(recommendations)
                avg_popularity = sum(r['popularity'] for r in recommendations) / len(recommendations)
                
                print("\n" + "-" * 70)
                print("STATISTICS:")
                print(f"   Average AI Probability: {avg_ai:.1f}%")
                print(f"   Average Popularity: {avg_popularity:.0f}/100")
                print(f"   Unique playlists: {len(set(r['discovered_on'] for r in recommendations))}")
                print("-" * 70)
        
        except Exception as e:
            print(f"\n Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        continue_prompt = input("\nTest another track? (y/n): ").strip().lower()
        if continue_prompt != 'y':
            print("\n Goodbye!:)")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    except Exception as e:
        print(f"\n Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()