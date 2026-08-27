import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

from blacklist import AIArtistBlacklist
from letssubmit_cache import LetsSubmitCache

#spotify cross-verification (only when spotify is configured)
try:
    from apis.spotify_client import SpotifyAPI
    SPOTIFY_AVAILABLE = True
except Exception:
    SPOTIFY_AVAILABLE = False


load_dotenv()

#Configuration

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

#discography verification thresholds
DISCOGRAPHY_AI_THRESHOLD = 60.0       #>=60% AI ->blacklist
DISCOGRAPHY_SAMPLE_NEWEST = 6
DISCOGRAPHY_SAMPLE_OLDEST = 4
DISCOGRAPHY_MIN_TRACKS = 4            #need at least this many samples

#per-recommendation AI threshold
RECOMMENDATION_AI_THRESHOLD = 40.0    #tracks with prob > this are dropped

#video filtering heuristics (skip non-music uploads on regular channels)
NON_MUSIC_TITLE_HINTS = [
    "vlog", "interview", "behind the scenes", "behind-the-scenes",
    "q&a", "q and a", "reaction", "react", "reacts",
    "live at", "live in", "live from",
    "tour diary", "press conference", "podcast", "episode",
    "tutorial", "lesson", "explained", "documentary",
    "ranked", "tier list", "best to worst", "worst to best",
    "top 10", "top 5", "top 20",
    "full album", "full ep", "album stream", "ep stream",
    "compilation", "mix vol", "playlist mix",
    "review", "analysis", "breakdown", "deep dive",
    "unboxing", "first listen", "discussion",
    " vs ", " vs. ", "side by side",
    "history of", "evolution of",
]
MIN_TRACK_SECONDS = 90      #1:30 minimum (we drop very short clips)
MAX_TRACK_SECONDS = 600     #10 minutes maximum
MAX_REC_TRACK_SECONDS = 600 

#quota-saving caches
ARTIST_PLAYLIST_CACHE = Path("data/youtube_playlist_search_cache.json")
ARTIST_CHANNEL_CACHE = Path("data/youtube_channel_lookup_cache.json")

REQUEST_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.2


#ISO 8601 DURATION PARSER

_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _parse_iso_duration(duration: str) -> int:
    """Convert YouTube's ISO 8601 PT#H#M#S to total seconds. Returns 0 on parse error."""
    if not duration:
        return 0
    match = _DURATION_RE.fullmatch(duration)
    if not match:
        return 0
    h, m, s = (int(g) if g else 0 for g in match.groups())
    return h * 3600 + m * 60 + s


#Youtube Discovered On recommender

class YouTubeDiscoveredOnRecommender:
    """YouTube-side equivalent of DiscoveredOnRecommender (Spotify)."""

    def __init__(self,
                 youtube_api_key: str,
                 letssubmit: LetsSubmitCache,
                 blacklist: AIArtistBlacklist,
                 spotify_api: Optional["SpotifyAPI"] = None):
        if not youtube_api_key:
            raise ValueError("youtube_api_key is required")

        self.youtube_api_key = youtube_api_key
        self.letssubmit = letssubmit
        self.blacklist = blacklist
        self.spotify = spotify_api  #may be None

        #in-memory session caches for artist verification verdicts
        #map: normalized_artist -> (is_ai: bool, ai_percent: float)
        self._session_checked: Dict[str, Tuple[bool, float]] = {}

        #per-run quota state (set fresh at start of every get_recommendations call)
        self.quota_tripped = False
        self._on_quota_trip = None

        #disk caches
        self._playlist_cache = self._load_json_cache(ARTIST_PLAYLIST_CACHE)
        self._channel_cache = self._load_json_cache(ARTIST_CHANNEL_CACHE)

    #quota-guarded letssubmit check

    def _safe_check(self, url: str) -> Optional[float]:
        """LetsSubmit check with quota tracking. After the first 429, all calls return None and on_quota_trip fires."""
        if self.quota_tripped:
            return None
        prob, status = self.letssubmit.check(url)
        if status == "quota_exhausted":
            self.quota_tripped = True
            print("[youtube] LetsSubmit quota tripped(429). Aborting verification.")
            if self._on_quota_trip:
                try:
                    self._on_quota_trip()
                except Exception: 
                    pass
            return None
        return prob

    #disk cache helpers

    @staticmethod
    def _load_json_cache(path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _save_json_cache(path: Path, data: Dict):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[youtube] Cache save failed for {path}: {e}")

    #YouTube API wrappers

    def _api_get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        params = dict(params)
        params["key"] = self.youtube_api_key
        url = f"{YOUTUBE_API_BASE}/{endpoint}"
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 403:
                print(f"[youtube] 403 (quota or auth): {r.text[:200]}")
            else:
                print(f"[youtube] HTTP {r.status_code} on {endpoint}: {r.text[:200]}")
        except Exception as e:
            print(f"[youtube] Request error on {endpoint}: {e}")
        return None

    def _search_playlists_for_artist(self, artist_name: str,
                                     max_results: int = 30) -> List[str]:
        """Returns list of playlist IDs. Cached by lowercase artist name."""
        key = artist_name.lower().strip()
        if key in self._playlist_cache:
            return self._playlist_cache[key]

        data = self._api_get("search", {
            "part": "snippet",
            "q": artist_name,
            "type": "playlist",
            "maxResults": min(max_results, 50),
        })
        if not data:
            return []

        playlist_ids = []
        for item in data.get("items", []):
            pid = item.get("id", {}).get("playlistId")
            if pid:
                playlist_ids.append(pid)

        self._playlist_cache[key] = playlist_ids
        self._save_json_cache(ARTIST_PLAYLIST_CACHE, self._playlist_cache)
        return playlist_ids

    def _fetch_playlist_items(self, playlist_id: str,
                              max_items: int = 50) -> List[Dict]:
        """Fetch up to max_items videos from a playlist with one page call."""
        data = self._api_get("playlistItems", {
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": min(max_items, 50),
        })
        if not data:
            return []
        return data.get("items", [])

    def _fetch_videos_metadata(self, video_ids: List[str]) -> Dict[str, Dict]:
        """Batch fetch video metadata (50 per call). Returns videoId -> item dict."""
        if not video_ids:
            return {}
        out = {}
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i:i + 50]
            data = self._api_get("videos", {
                "part": "snippet,contentDetails",
                "id": ",".join(chunk),
            })
            if data:
                for item in data.get("items", []):
                    out[item["id"]] = item
            time.sleep(RATE_LIMIT_DELAY)
        return out

    #channel resolution n discography sampling

    @staticmethod
    def _uploads_playlist_from_channel(channel_id: str) -> Optional[str]:
        """UC... -> UU..., free derivation, no API call."""
        if channel_id and channel_id.startswith("UC"):
            return "UU" + channel_id[2:]
        return None

    def _sample_artist_uploads(self, channel_id: str) -> List[Dict]:
        """Fetch 6 newest + 4 oldest uploads from a channel, filtered to plausibly-musical items."""
        uploads_pid = self._uploads_playlist_from_channel(channel_id)
        if not uploads_pid:
            return []

        #newest (first page, just take first 6)
        newest_items = self._fetch_playlist_items(uploads_pid, max_items=10)
        newest = newest_items[:DISCOGRAPHY_SAMPLE_NEWEST]

        #oldest: page through to the end. To keep quota reasonable we cap at 200 items (4 pages). 
        #for channels bigger than that, the "oldest" sample becomes "roughly old"(still fine).
        oldest = self._fetch_oldest_uploads(uploads_pid,
                                            n_wanted=DISCOGRAPHY_SAMPLE_OLDEST,
                                            max_pages=4)

        sample_video_ids = [
            item.get("contentDetails", {}).get("videoId")
            for item in (newest + oldest)
            if item.get("contentDetails", {}).get("videoId")
        ]
        #dedupe (small channels can repeat)
        sample_video_ids = list(dict.fromkeys(sample_video_ids))
        if not sample_video_ids:
            return []

        #fetch full metadata to apply duration/title filters
        meta = self._fetch_videos_metadata(sample_video_ids)
        plausible = []
        for vid_id, item in meta.items():
            snippet = item.get("snippet", {}) or {}
            title = (snippet.get("title") or "").lower()
            duration = item.get("contentDetails", {}).get("duration", "")
            seconds = _parse_iso_duration(duration)

            #duration check
            if seconds < MIN_TRACK_SECONDS or seconds > MAX_TRACK_SECONDS:
                continue
            #title keyword check
            if any(hint in title for hint in NON_MUSIC_TITLE_HINTS):
                continue

            plausible.append(item)
        return plausible

    def _fetch_oldest_uploads(self, uploads_playlist_id: str,
                              n_wanted: int, max_pages: int) -> List[Dict]:
        """Page through an uploads playlist and return the last n_wanted items."""
        page_token = None
        last_items: List[Dict] = []
        for _ in range(max_pages):
            params = {
                "part": "snippet,contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": 50,
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._api_get("playlistItems", params)
            if not data:
                break
            items = data.get("items", [])
            if items:
                last_items = items  #keep overwriting; final page is "oldest"
            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(RATE_LIMIT_DELAY)
        return last_items[-n_wanted:]

    #Artist verification: Spotify first, YouTube uploads fallback

    def _verify_artist(self, artist_name: str,
                       fallback_channel_id: Optional[str] = None
                       ) -> Tuple[Optional[bool], float, str]:
        """
        Returns (is_ai_artist, ai_percent, method) where:
          is_ai_artist : True | False | None (None = could not verify)
          ai_percent   : 0-100, 0 if not verified
          method       : 'spotify' | 'youtube' | 'session_cache' | 'unverified'
        """
        #session cache hit
        norm = self.blacklist._normalize_artist_name(artist_name)
        if norm in self._session_checked:
            is_ai, pct = self._session_checked[norm]
            return is_ai, pct, "session_cache"

        #try Spotify discography check first
        if self.spotify is not None:
            verdict = self._verify_via_spotify(artist_name)
            if verdict is not None:
                is_ai, pct = verdict
                self._session_checked[norm] = (is_ai, pct)
                if is_ai:
                    self.blacklist.add_to_blacklist(artist_name)
                return is_ai, pct, "spotify"

        #fall back to youtube uploads sample
        if fallback_channel_id:
            verdict = self._verify_via_youtube_channel(
                artist_name, fallback_channel_id
            )
            if verdict is not None:
                is_ai, pct = verdict
                self._session_checked[norm] = (is_ai, pct)
                if is_ai:
                    self.blacklist.add_to_blacklist(artist_name)
                return is_ai, pct, "youtube"

        return None, 0.0, "unverified"

    def _verify_via_spotify(self, artist_name: str
                            ) -> Optional[Tuple[bool, float]]:
        """Returns (is_ai, pct) or None if Spotify check could not run."""
        if self.spotify is None:
            return None
        try:
            #search for the artist on spotify
            search_result = self.spotify.search(q=artist_name, type_="artist", limit=1)
        except Exception:
            return None

        artists = search_result.get("artists", {}).get("items", [])
        if not artists:
            return None
        artist = artists[0]
        artist_id = artist.get("id")
        if not artist_id:
            return None

        #pull top tracks
        try:
            top = self.spotify.get_artist_top_tracks(artist_id, market="US")
        except Exception:
            return None

        tracks = top.get("tracks", [])[:10]
        if len(tracks) < DISCOGRAPHY_MIN_TRACKS:
            return None

        ai_count = 0
        total = 0
        for tr in tracks:
            ext = tr.get("external_urls", {}) or {}
            sp_url = ext.get("spotify")
            if not sp_url:
                continue
            prob = self._safe_check(sp_url)
            if self.quota_tripped:
                return None  #abort verification
            if prob is None:
                continue
            total += 1
            if prob >= RECOMMENDATION_AI_THRESHOLD:
                ai_count += 1

        if total < DISCOGRAPHY_MIN_TRACKS:
            return None

        pct = (ai_count / total) * 100
        print(f"[verify-spotify] '{artist_name}': {ai_count}/{total} AI = {pct:.1f}%")
        return (pct >= DISCOGRAPHY_AI_THRESHOLD, pct)

    def _verify_via_youtube_channel(self, artist_name: str, channel_id: str
                                    ) -> Optional[Tuple[bool, float]]:
        """Sample uploads from a YouTube channel and run LetsSubmit on each."""
        sample_videos = self._sample_artist_uploads(channel_id)
        if len(sample_videos) < DISCOGRAPHY_MIN_TRACKS:
            return None

        ai_count = 0
        total = 0
        for item in sample_videos:
            vid_id = item.get("id")
            if not vid_id:
                continue
            yt_url = f"https://www.youtube.com/watch?v={vid_id}"
            prob = self._safe_check(yt_url)
            if self.quota_tripped:
                return None
            if prob is None:
                continue
            total += 1
            if prob >= RECOMMENDATION_AI_THRESHOLD:
                ai_count += 1

        if total < DISCOGRAPHY_MIN_TRACKS:
            return None

        pct = (ai_count / total) * 100
        print(f"[verify-youtube] '{artist_name}': {ai_count}/{total} AI = {pct:.1f}%")
        return (pct >= DISCOGRAPHY_AI_THRESHOLD, pct)

    #Main entry point

    def get_recommendations(self, artist: str, title: str,
                            max_playlists: int = 30,
                            max_recommendations: int = 10,
                            strict_mode: bool = False,
                            on_quota_trip=None,
                            exclude_video_ids: Optional[List[str]] = None,
                            ) -> List[Dict]:
        """Return human-made YouTube-track recommendations.
        strict_mode       -- reject any candidate not explicitly verified as human;
                            skips the mix-playlist and music-category fallback passes.
        on_quota_trip     -- callback fired the moment LetsSubmit returns 429;
                            the recommender stops verifying and returns what it has.
        exclude_video_ids -- video IDs to skip (used so the user's input video can't
                            be recommended back to them).
        """
        #reset per-run quota state
        self.quota_tripped = False
        self._on_quota_trip = on_quota_trip

        print(f"[youtube] Discovered-On for: {artist} - {title}"
              f"  (strict={strict_mode})")

        playlist_ids = self._search_playlists_for_artist(artist, max_playlists)
        print(f"[youtube] Found {len(playlist_ids)} candidate playlists")

        recommendations: List[Dict] = []
        #seed seen_video_ids with the caller's exclusions so they're filtered everywhere in the pipeline
        seen_video_ids: Set[str] = set(exclude_video_ids or [])
        if exclude_video_ids:
            print(f"[youtube] Excluding {len(exclude_video_ids)} caller-specified video_id(s)")
        seen_artists_lower: Set[str] = set([artist.lower().strip()])
        original_artist_norm = self.blacklist._normalize_artist_name(artist)

        #collect candidates first, then batch-fetch durations in one videos.list call to filter long-form items before verification
        raw_candidates: List[Dict] = []

        for pid in playlist_ids:
            if len(raw_candidates) >= max_recommendations * 4:
                break

            items = self._fetch_playlist_items(pid, max_items=50)
            if not items:
                continue

            playlist_tracks = []
            for it in items:
                sn = it.get("snippet", {}) or {}
                cd = it.get("contentDetails", {}) or {}
                vid = cd.get("videoId") or sn.get("resourceId", {}).get("videoId")
                if not vid:
                    continue
                playlist_tracks.append({
                    "video_id": vid,
                    "title": sn.get("title", ""),
                    "channel_title": sn.get("videoOwnerChannelTitle") or "",
                    "channel_id": sn.get("videoOwnerChannelId") or "",
                    "playlist_title": pid,
                })

            #check 1: playlist must contain the original artist
            artist_lower = artist.lower()
            contains_artist = any(
                artist_lower in (t["channel_title"] or "").lower()
                or artist_lower in (t["title"] or "").lower()
                for t in playlist_tracks
            )
            if not contains_artist:
                continue

            #per-track filtering
            for t in playlist_tracks:
                if t["video_id"] in seen_video_ids:
                    continue

                #drop obvious non-music titles up front
                title_l = (t["title"] or "").lower()
                if any(hint in title_l for hint in NON_MUSIC_TITLE_HINTS):
                    continue

                #parse artist name from metadata
                candidate_artist = self._artist_from_track(t)
                if not candidate_artist:
                    continue

                #same-artist check (normalized)
                cand_norm = self.blacklist._normalize_artist_name(candidate_artist)
                if cand_norm == original_artist_norm:
                    continue

                cand_lower = candidate_artist.lower().strip()
                if cand_lower in seen_artists_lower:
                    continue

                #check if the title mentions the original artist by name (in case of a reuploader)
                if artist_lower in title_l:
                    continue

                if self.blacklist.is_blacklisted(candidate_artist):
                    continue

                raw_candidates.append({
                    **t,
                    "candidate_artist": candidate_artist,
                    "candidate_norm": cand_norm,
                    "candidate_lower": cand_lower,
                })
                seen_video_ids.add(t["video_id"])

        print(f"[youtube] {len(raw_candidates)} raw candidates after cheap filters")

        #batch fetch durations to filter out long-form content
        candidate_artworks: Dict[str, Optional[str]] = {}
        if raw_candidates:
            video_ids = [c["video_id"] for c in raw_candidates]
            meta = self._fetch_videos_metadata(video_ids)
            filtered_candidates = []
            for c in raw_candidates:
                item = meta.get(c["video_id"])
                if not item:
                    continue
                duration = item.get("contentDetails", {}).get("duration", "")
                seconds = _parse_iso_duration(duration)
                if seconds < MIN_TRACK_SECONDS or seconds > MAX_REC_TRACK_SECONDS:
                    continue
                full_title_l = (item.get("snippet", {}).get("title") or "").lower()
                if any(hint in full_title_l for hint in NON_MUSIC_TITLE_HINTS):
                    continue
                if artist_lower in full_title_l:
                    continue
                #cache artwork URL from the snippet
                thumbs = item.get("snippet", {}).get("thumbnails", {})
                artwork = (thumbs.get("high") or thumbs.get("medium")
                           or thumbs.get("default") or {}).get("url")
                candidate_artworks[c["video_id"]] = artwork
                c["duration_seconds"] = seconds
                filtered_candidates.append(c)
            raw_candidates = filtered_candidates
            print(f"[youtube] {len(raw_candidates)} candidates pass duration/title filters")

        #verify candidate artists 
        for c in raw_candidates:
            if len(recommendations) >= max_recommendations:
                break
            if self.quota_tripped:
                print("[youtube] Quota tripped mid-verify; aborting candidate loop.")
                break

            candidate_artist = c["candidate_artist"]
            cand_lower = c["candidate_lower"]
            if cand_lower in seen_artists_lower:
                continue

            #the verifying
            is_ai, pct, method = self._verify_artist(
                candidate_artist,
                fallback_channel_id=c.get("channel_id"),
            )

            #strict mode only accepts when explicitly verified as human,loose mode: accept verified-human/unverified.
            if is_ai is True:
                print(f"  [skip-ai] {candidate_artist} ({pct:.1f}%)")
                continue
            if strict_mode and is_ai is not False:
                print(f"  [skip-unverified-strict] {candidate_artist} ({method})")
                continue

            seen_artists_lower.add(cand_lower)
            recommendations.append({
                "video_id": c["video_id"],
                "title": c["title"],
                "artist": candidate_artist,
                "channel_id": c.get("channel_id"),
                "url": f"https://www.youtube.com/watch?v={c['video_id']}",
                "artwork": candidate_artworks.get(c["video_id"]),
                "discovered_in_playlist": c.get("playlist_title", ""),
                "verification_method": method,
                "verification_pct": pct,
            })
            print(f"  [accept] {candidate_artist} - {c['title']}  ({method})")

        #if quota tripped, return whatever was verified before the trip(in strict mode that's only candidates explicitly set as human)
        if self.quota_tripped:
            return recommendations[:max_recommendations]

        #in strict mode skip the Mix and music-category fallbacks entirely bc they produce unverified candidates by design.
        if strict_mode:
            return recommendations[:max_recommendations]

        #Mix-playlist signal iff short 
        if len(recommendations) < 5:
            print(f"[youtube] Only {len(recommendations)} found; trying Mix bonus...")
            bonus = self._mix_playlist_bonus(
                f"{artist} {title}",
                seen_video_ids, seen_artists_lower,
                wanted=max_recommendations - len(recommendations),
            )
            recommendations.extend(bonus)

        #last resort: music-category video search 
        if len(recommendations) < 5:
            print(f"[youtube] Still {len(recommendations)}; falling back to music search...")
            fallback = self._music_category_search(
                artist, title,
                seen_video_ids, seen_artists_lower,
                wanted=max_recommendations - len(recommendations),
            )
            recommendations.extend(fallback)

        return recommendations[:max_recommendations]

    #bonus & fallback signals

    def _mix_playlist_bonus(self, query: str,
                            seen_video_ids: Set[str],
                            seen_artists_lower: Set[str],
                            wanted: int) -> List[Dict]:
        """Search '<query> mix' and pull tracks from any RD-prefixed playlist hit.Mix playlists are dynamic so this often yields nothing."""
        if wanted <= 0:
            return []
        data = self._api_get("search", {
            "part": "snippet",
            "q": f"{query} mix",
            "type": "playlist",
            "maxResults": 5,
        })
        if not data:
            return []

        out: List[Dict] = []
        for hit in data.get("items", []):
            pid = hit.get("id", {}).get("playlistId")
            if not pid or not pid.startswith("RD"):
                continue
            items = self._fetch_playlist_items(pid, max_items=20)
            if len(items) <= 1:
                continue  #mix returned only the seed track
            for it in items:
                if len(out) >= wanted:
                    break
                sn = it.get("snippet", {}) or {}
                cd = it.get("contentDetails", {}) or {}
                vid = cd.get("videoId")
                if not vid or vid in seen_video_ids:
                    continue
                title_l = (sn.get("title") or "").lower()
                if any(hint in title_l for hint in NON_MUSIC_TITLE_HINTS):
                    continue
                channel_title = sn.get("videoOwnerChannelTitle") or ""
                candidate_artist = self._derive_artist(channel_title, sn.get("title", ""))
                if not candidate_artist:
                    continue
                cand_lower = candidate_artist.lower().strip()
                if cand_lower in seen_artists_lower:
                    continue
                if self.blacklist.is_blacklisted(candidate_artist):
                    continue
                seen_video_ids.add(vid)
                seen_artists_lower.add(cand_lower)
                thumbs = sn.get("thumbnails", {})
                artwork = (thumbs.get("high") or thumbs.get("medium")
                           or thumbs.get("default") or {}).get("url")
                out.append({
                    "video_id": vid,
                    "title": sn.get("title", ""),
                    "artist": candidate_artist,
                    "channel_id": sn.get("videoOwnerChannelId") or "",
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "artwork": artwork,
                    "discovered_in_playlist": pid,
                    "verification_method": "mix-bonus-unverified",
                    "verification_pct": 0.0,
                })
            if out:
                break
        return out

    def _music_category_search(self, artist: str, title: str,
                               seen_video_ids: Set[str],
                               seen_artists_lower: Set[str],
                               wanted: int) -> List[Dict]:
        """Music-category video search using artist +'similar' as the query."""
        if wanted <= 0:
            return []
        data = self._api_get("search", {
            "part": "snippet",
            "q": f"{artist} similar",
            "type": "video",
            "videoCategoryId": "10",
            "maxResults": min(wanted * 2, 25),
        })
        if not data:
            return []

        out: List[Dict] = []
        for hit in data.get("items", []):
            if len(out) >= wanted:
                break
            vid = hit.get("id", {}).get("videoId")
            sn = hit.get("snippet", {}) or {}
            if not vid or vid in seen_video_ids:
                continue
            title_l = (sn.get("title") or "").lower()
            if any(hint in title_l for hint in NON_MUSIC_TITLE_HINTS):
                continue
            channel_title = sn.get("channelTitle") or ""
            candidate_artist = self._derive_artist(channel_title, sn.get("title", ""))
            if not candidate_artist:
                continue
            cand_lower = candidate_artist.lower().strip()
            if cand_lower in seen_artists_lower:
                continue
            if self.blacklist.is_blacklisted(candidate_artist):
                continue
            seen_video_ids.add(vid)
            seen_artists_lower.add(cand_lower)
            thumbs = sn.get("thumbnails", {})
            artwork = (thumbs.get("high") or thumbs.get("medium")
                       or thumbs.get("default") or {}).get("url")
            out.append({
                "video_id": vid,
                "title": sn.get("title", ""),
                "artist": candidate_artist,
                "channel_id": sn.get("channelId") or "",
                "url": f"https://www.youtube.com/watch?v={vid}",
                "artwork": artwork,
                "discovered_in_playlist": "music-category-search",
                "verification_method": "search-unverified",
                "verification_pct": 0.0,
            })
        return out

    #heuristics for getting the artist name out of youtube metadata

    @staticmethod
    def _artist_from_track(track_dict: Dict) -> Optional[str]:
        ch = (track_dict.get("channel_title") or "").strip()
        title = (track_dict.get("title") or "").strip()
        return YouTubeDiscoveredOnRecommender._derive_artist(ch, title)

    @staticmethod
    def _derive_artist(channel_title: str, video_title: str) -> Optional[str]:
        """Most-likely artist name for a video, in priority order:
        1. '<Artist> - Topic' channel  -> strip the suffix
        2. Title in 'Artist - Track' form -> take the left side
        3. Channel name (last resort; re-uploaders / labels read as the artist here)
        """
        ch = (channel_title or "").strip()
        title = (video_title or "").strip()

        #topic channels are reliable: "<Artist> -Topic"
        if ch.endswith(" - Topic"):
            return ch[: -len(" - Topic")].strip()

        #title convention beats channel name for everything else
        if " - " in title:
            artist_part = title.split(" - ", 1)[0].strip()
            if ":" in artist_part:
                artist_part = artist_part.split(":", 1)[0].strip()
            if artist_part:
                return artist_part

        #last resort
        return ch or None