import os
import sys
import threading
import time
import traceback
import uuid
import secrets
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime, timezone
import re as _re_title

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from apis.spotify_client import SpotifyAPI
from blacklist import AIArtistBlacklist
from letssubmit_cache import LetsSubmitCache
from sightengine_cache import SightengineCache   
from youtube_discovered_on import YouTubeDiscoveredOnRecommender

_VERSION_SUFFIX = _re_title.compile(
    r"\s*[-–(]\s*"
    r"((digitally|lp|album|original|2004|\d{4})\s+)?"   #optional modifier word
    r"(live|acoustic|demo|radio edit|album version|"
    r"remaster(ed)?|single version|studio version|official|version)\b.*$",
    _re_title.IGNORECASE,
)

def _strip_version_suffix(t: str) -> str:
    return _VERSION_SUFFIX.sub("", t).strip()

#Spotify Discovered-On 
try:
    from discovered_on_recommender import DiscoveredOnRecommender
    SPOTIFY_DISCOVERED_AVAILABLE = True
except Exception as _e:
    print(f"[app] discovered_on_recommender unavailable: {_e}")
    SPOTIFY_DISCOVERED_AVAILABLE = False


class _LetsSubmitCacheAdapter:
    """Cache-backed shim that the Spotify recommender calls instead of LetsSubmit directly."""
    def __init__(self, cache: "LetsSubmitCache"):
        self._cache = cache
        self.quota_tripped = False #set on first 429 /subsequent checks short-circuit
        self._on_quota = None  # optional callback when quota first trips

    def analyze_spotify_track(self, spotify_url: str):
        if self.quota_tripped:
            return None
        prob, status = self._cache.check(spotify_url)
        if status == "quota_exhausted":
            self.quota_tripped = True
            if self._on_quota:
                try: 
                    self._on_quota()
                except Exception: 
                    pass
            return None
        if prob is None:
            return None
        return {"ai_probability": prob}


app = Flask(__name__)


# CONFIG + SHARED SERVICES

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
LETSSUBMIT_API_KEY = os.getenv("LETSSUBMIT_API_KEY", "")
SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER", "")       
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET", "") 


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def _persist_job(job_id: str):
    """Called once per job, when the job finishes (success or error)."""
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        #snapshot before releasing lock
        job = dict(JOBS[job_id])

    #strip large fields we don't need to analyse later
    payload = {
        "job_id": job_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - job["created_at"], 2),
        "state": job["state"],
        "track": job.get("track"),
        "ai": job.get("ai"),
        "quota_exhausted": job.get("quota_exhausted", False),
        "error": job.get("error"),
        "progress": job.get("progress", []),  
        "recommendation_count": len(job.get("recommendations", [])),
        "recommendations": [
            #only the fields useful for analysis 
            {
                "track_id": r.get("track_id"),
                "artist": r.get("artist"),
                "title": r.get("title"),
                "tag_score": r.get("tag_score"),
                "cf_score": r.get("cf_score"),
                "audio_score": r.get("audio_score"),
                "final_score": r.get("final_score"),
                "source": r.get("source"),  
            }
            for r in job.get("recommendations", [])
        ],
    }

    #one file per job 
    fname = LOG_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{job_id}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


print("[app] Initializing shared services...")

spotify_client: Optional[SpotifyAPI] = None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        spotify_client = SpotifyAPI(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
        print("[app] Spotify client ready.")
    except Exception as e:
        print(f"[app] Spotify init failed: {e}")

blacklist = AIArtistBlacklist()
letssubmit_cache = LetsSubmitCache(LETSSUBMIT_API_KEY)
sightengine_cache = SightengineCache(                              
    api_user=SIGHTENGINE_API_USER,                                 
    api_secret=SIGHTENGINE_API_SECRET,                             
)                                                                  

youtube_recommender: Optional[YouTubeDiscoveredOnRecommender] = None
if YOUTUBE_API_KEY:
    try:
        youtube_recommender = YouTubeDiscoveredOnRecommender(
            youtube_api_key=YOUTUBE_API_KEY,
            letssubmit=letssubmit_cache,
            blacklist=blacklist,
            spotify_api=spotify_client,
        )
        print("[app] YouTube recommender ready.")
    except Exception as e:
        print(f"[app] YouTube recommender init failed: {e}")

#spotify discovered on recommender (uses our cache via the adapter)
spotify_recommender: Optional["DiscoveredOnRecommender"] = None
if SPOTIFY_DISCOVERED_AVAILABLE and spotify_client is not None:
    try:
        spotify_recommender = DiscoveredOnRecommender(
            spotify_api=spotify_client,
            ai_detector=_LetsSubmitCacheAdapter(letssubmit_cache),
        )
        # replace the recommender's default blacklist with the shared instance
        spotify_recommender.blacklist = blacklist
        print("[app] Spotify Discovered-On recommender ready.")
    except Exception as e:
        print(f"[app] Spotify Discovered-On init failed: {e}")

#hybrid recommender
hybrid_recommender = None
hybrid_load_lock = threading.Lock()


def get_hybrid_recommender():
    """Loads the heavy cascading hybrid recommender on first use."""
    global hybrid_recommender
    if hybrid_recommender is not None and hybrid_recommender is not False:
        return hybrid_recommender
    if hybrid_recommender is False:
        return None
    with hybrid_load_lock:
        if hybrid_recommender is not None:
            return hybrid_recommender if hybrid_recommender is not False else None
        try:
            from cascading_hybrid_recommender import CascadingHybridRecommender
            print("[app] Loading hybrid recommender (this may take a moment)...")
            r = CascadingHybridRecommender(data_path="data")
            r.load_data()
            hybrid_recommender = r
            print("[app] Hybrid recommender ready.")
        except Exception as e:
            print(f"[app] Hybrid recommender failed to load: {e}")
            hybrid_recommender = False  # sentinel: tried, failed
    return hybrid_recommender if hybrid_recommender else None


#JOB STORE

JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 600


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "state": "queued",
            "created_at": time.time(),
            "progress": [],
            "track": None,
            "ai": None,
            "cover_ai": None,          
            "recommendations": [],
            "error": None,
            "quota_exhausted": False,
        }
    #reset the spotify adapter's quota flag -new analyze run, fresh attempt
    if spotify_recommender is not None:
        adapter = getattr(spotify_recommender, "ai_detector", None)
        if adapter is not None and hasattr(adapter, "quota_tripped"):
            adapter.quota_tripped = False
    return job_id


def _update_job(job_id: str, **fields):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _log_job(job_id: str, message: str):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["progress"].append({
                "t": round(time.time() - JOBS[job_id]["created_at"], 2),
                "msg": message,
            })
    print(f"[job {job_id}] {message}")


def _append_recommendation(job_id: str, rec: Dict):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["recommendations"].append(rec)


def _purge_old_jobs():
    now = time.time()
    with JOBS_LOCK:
        stale = [jid for jid, j in JOBS.items()
                 if now - j["created_at"] > JOB_TTL_SECONDS]
        for jid in stale:
            del JOBS[jid]


# INPUT PARSING

def _identify_input(raw: str) -> Dict:
    raw = (raw or "").strip()
    if not raw:
        return {"kind": "empty"}
    lower = raw.lower()
    if "spotify.com/track/" in lower:
        return {"kind": "spotify", "url": raw}
    if "youtube.com/watch" in lower or "youtu.be/" in lower:
        return {"kind": "youtube", "url": raw}
    return {"kind": "search", "query": raw}


def _extract_spotify_id(url: str) -> Optional[str]:
    try:
        return url.split("spotify.com/track/")[1].split("?")[0].split("/")[0]
    except (IndexError, AttributeError):
        return None


def _extract_youtube_id(url: str) -> Optional[str]:
    try:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    except (IndexError, AttributeError):
        pass
    return None


def _clean_youtube_title(raw_title: str, channel: str) -> Dict:
    """Parse 'Artist - Title' (or channel-as-artist) out of a YouTube video title, strip common annotations."""
    import re as _re

    cleaned = raw_title

    #parenthetical suffixes to drop entirely.
    paren_patterns = [
        r"\(Official\s+(?:AI\s+)?(?:Music\s+)?(?:Audio|Video|Lyric Video|Music Video)\)",
        r"\[Official\s+(?:AI\s+)?(?:Music\s+)?(?:Audio|Video|Lyric Video|Music Video)\]",
        r"\((?:Official\s+)?Visualizer\)",
        r"\[(?:Official\s+)?Visualizer\]",
        r"\(Lyric(?:s)?(?:\s+Video)?\)",
        r"\[Lyric(?:s)?(?:\s+Video)?\]",
        r"\((?:Official\s+)?Audio\)",
        r"\[(?:Official\s+)?Audio\]",
        r"\((?:HD|HQ|4K|1080p|720p)\)",
        r"\[(?:HD|HQ|4K|1080p|720p)\]",
        r"\((?:Live|Live\s+Version|Acoustic|Remastered(?:\s+\d{4})?)\)",
        r"\(\d{4}\)",  # year in parens, e.g. (2024)
    ]
    for p in paren_patterns:
        cleaned = _re.sub(p, "", cleaned, flags=_re.IGNORECASE)

    # " | Genre" pipe-suffix annotation (eg "Pale Remains |Gothic Metal")
    cleaned = _re.sub(r"\s*\|\s*[^|]{1,40}$", "", cleaned).strip()

    #double-slash separators: "Title // Genre Year // Official Music video"
    cleaned = _re.sub(r"\s*//.*$", "", cleaned).strip()

    #collapse multiple spaces
    cleaned = _re.sub(r"\s+", " ", cleaned).strip()

    if channel.endswith(" - Topic"):
        return {
            "artist": channel[: -len(" - Topic")].strip(),
            "title": cleaned,
            "cleaned_title": cleaned,
        }
    if " - " in cleaned:
        a, t = cleaned.split(" - ", 1)
        return {
            "artist": a.strip(),
            "title": t.strip(),
            "cleaned_title": cleaned,
        }
    return {"artist": channel, "title": cleaned, "cleaned_title": cleaned}


# WORKER

def _worker(job_id: str, payload: Dict):
    try:
        kind = payload["kind"]
        _update_job(job_id, state="resolving")

        if kind == "spotify":
            track_meta = _resolve_spotify(job_id, payload["url"])
            if track_meta is None:
                _update_job(job_id, state="error", error="Could not fetch Spotify track.")
                return
            _update_job(job_id, track=track_meta, state="ai_checking")
            ai = _run_ai_check_for_track(job_id, track_meta)
            _update_job(job_id, ai=ai)
            cover_ai = _run_cover_check(job_id, track_meta.get("artwork"))   
            _update_job(job_id, cover_ai=cover_ai)
            _update_job(job_id, state="recommending")
            _dispatch_recommendations(job_id, track_meta, ai)

        elif kind == "youtube":
            track_meta = _resolve_youtube(job_id, payload["url"])
            if track_meta is None:
                _update_job(job_id, state="error", error="Could not fetch YouTube video.")
                return
            yt_label = _check_youtube_ai_label(track_meta.get("youtube_id", ""), YOUTUBE_API_KEY)  
            if yt_label["disclosed_as_ai"]:
                _log_job(job_id, f"YouTube discloses this as AI content (via {yt_label['source']}).")
            _update_job(job_id, track=track_meta, state="ai_checking")
            ai = _run_ai_check_for_track(job_id, track_meta)
            _update_job(job_id, ai=ai)
            cover_ai = _run_cover_check(job_id, track_meta.get("artwork"))
            _update_job(job_id, cover_ai=cover_ai)
            _update_job(job_id, state="recommending")
            _dispatch_recommendations(job_id, track_meta, ai)

        elif kind == "resolved":
            # user picked a candidate from the disambiguation UI
            track_meta = payload["track_meta"]
            _log_job(job_id,
                     f"Using user-picked track: {track_meta.get('artist')} - "
                     f"{track_meta.get('title')}")
            _update_job(job_id, track=track_meta, state="ai_checking")
            ai = _run_ai_check_for_track(job_id, track_meta)
            _update_job(job_id, ai=ai)
            cover_ai = _run_cover_check(job_id, track_meta.get("artwork"))  
            _update_job(job_id, cover_ai=cover_ai)   
            _update_job(job_id, state="recommending")
            _dispatch_recommendations(job_id, track_meta, ai)

        else:
            _update_job(job_id, state="error", error="Empty input.")
            _persist_job(job_id)
            return

        _update_job(job_id, state="done")
        _log_job(job_id, "Job complete.")
        _persist_job(job_id)
    except Exception as e:
        traceback.print_exc()
        _update_job(job_id, state="error",
                    error=f"Internal error: {type(e).__name__}: {e}")
        _persist_job(job_id)


# RESOLUTION

import difflib
import unicodedata
import re as _re_search

#spotify similarity at/above which we auto-resolve; below this, show the picker
SPOTIFY_AUTO_THRESHOLD = 0.80
#max candidates to surface in the "did you mean...?" picker
RESOLUTION_CANDIDATES_MAX = 3
# similarity required for a YouTube->Spotify cross-match (so the AI verdict applies to the same recording)
CROSS_MATCH_THRESHOLD = 0.80


def _norm_for_compare(s: str) -> str:
    """Lowercase + strip accents + collapse whitespace + drop punctuation. For similarity scoring only."""
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    no_accents = no_accents.lower()
    no_accents = _re_search.sub(r"[^\w\s]+", " ", no_accents)
    no_accents = _re_search.sub(r"\s+", " ", no_accents).strip()
    return no_accents


def _similarity(query: str, candidate: str) -> float:
    a, b = _norm_for_compare(query), _norm_for_compare(candidate)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _resolve_spotify(job_id: str, url: str) -> Optional[Dict]:
    if spotify_client is None:
        _log_job(job_id, "Spotify client not configured.")
        return None
    track_id = _extract_spotify_id(url)
    if not track_id:
        return None
    _log_job(job_id, f"Spotify track id: {track_id}")
    try:
        info = spotify_client.get_track_info(track_id)
    except Exception as e:
        _log_job(job_id, f"Spotify API error: {e}")
        return None
    if not info or "name" not in info:
        return None

    images = info.get("album", {}).get("images", [])
    return {
        "platform": "spotify",
        "spotify_url": url,
        "spotify_id": track_id,
        "title": info["name"],
        "artist": info["artists"][0]["name"] if info.get("artists") else "Unknown",
        "album": info.get("album", {}).get("name", ""),
        "preview_url": info.get("preview_url"),
        "artwork": images[0]["url"] if images else None,
        "embed_url": f"https://open.spotify.com/embed/track/{track_id}",
    }


def _resolve_youtube(job_id: str, url: str) -> Optional[Dict]:
    video_id = _extract_youtube_id(url)
    if not video_id:
        return None
    _log_job(job_id, f"YouTube video id: {video_id}")
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "snippet,contentDetails", "id": video_id,
                    "key": YOUTUBE_API_KEY},
            timeout=10,
        )
        if r.status_code != 200:
            _log_job(job_id, f"YouTube API HTTP {r.status_code}")
            return None
        items = r.json().get("items", [])
        if not items:
            return None
        snippet = items[0].get("snippet", {})
        parsed = _clean_youtube_title(snippet.get("title", ""),
                                      snippet.get("channelTitle", ""))
        thumbs = snippet.get("thumbnails", {})
        artwork = (thumbs.get("maxres") or thumbs.get("high")
                   or thumbs.get("medium") or thumbs.get("default") or {}).get("url")
        return {
            "platform": "youtube",
            "youtube_url": url,
            "youtube_id": video_id,
            "title": parsed["title"],
            "artist": parsed["artist"],
            "cleaned_title": parsed.get("cleaned_title", parsed["title"]),
            "channel": snippet.get("channelTitle", ""),
            "artwork": artwork,
            "embed_url": f"https://www.youtube.com/embed/{video_id}",
        }
    except Exception as e:
        _log_job(job_id, f"YouTube fetch error: {e}")
        return None


def _search_spotify_candidates(query: str, limit: int = 5) -> List[Dict]:
    """Return a ranked list of Spotify search results for free-text query."""
    if spotify_client is None:
        return []
    try:
        data = spotify_client._make_request(
            "search", params={"q": query, "type": "track", "limit": limit},
        )
    except Exception:
        return []
    items = data.get("tracks", {}).get("items", []) or []
    out = []
    for t in items:
        track_id = t.get("id")
        if not track_id:
            continue
        artist = (t.get("artists", [{}])[0] or {}).get("name", "")
        title = t.get("name", "")
        images = t.get("album", {}).get("images", [])
        candidate = {
            "source": "spotify",
            "spotify_id": track_id,
            "spotify_url": f"https://open.spotify.com/track/{track_id}",
            "embed_url": f"https://open.spotify.com/embed/track/{track_id}",
            "artist": artist,
            "title": title,
            "album": t.get("album", {}).get("name", ""),
            "artwork": images[0]["url"] if images else None,
            "similarity": _similarity(query, f"{artist} {title}"),
        }
        out.append(candidate)
    out.sort(key=lambda c: -c["similarity"])
    return out


def _search_youtube_candidates(query: str, limit: int = 5) -> List[Dict]:
    """Return a ranked list of YouTube music-category results."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "q": query, "type": "video",
                    "videoCategoryId": "10", "maxResults": limit,
                    "key": YOUTUBE_API_KEY},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        items = r.json().get("items", []) or []
    except Exception:
        return []
    out = []
    for it in items:
        vid = it.get("id", {}).get("videoId")
        if not vid:
            continue
        sn = it.get("snippet", {})
        parsed = _clean_youtube_title(sn.get("title", ""), sn.get("channelTitle", ""))
        thumbs = sn.get("thumbnails", {})
        artwork = (thumbs.get("high") or thumbs.get("medium")
                   or thumbs.get("default") or {}).get("url")
        candidate = {
            "source": "youtube",
            "youtube_id": vid,
            "youtube_url": f"https://www.youtube.com/watch?v={vid}",
            "embed_url": f"https://www.youtube.com/embed/{vid}",
            "artist": parsed["artist"],
            "title": parsed["title"],
            "cleaned_title": parsed.get("cleaned_title", parsed["title"]),
            "artwork": artwork,
            "channel": sn.get("channelTitle", ""),
            "similarity": _similarity(query, f"{parsed['artist']} {parsed['title']}"),
        }
        out.append(candidate)
    out.sort(key=lambda c: -c["similarity"])
    return out


def _dedupe_and_rank(spotify_results: List[Dict],
                     youtube_results: List[Dict]) -> List[Dict]:
    """Merge Spotify + YouTube candidates; if a Spotify and YouTube result describe the same song
    (similarity >= 0.85), keep the Spotify one. Rank by similarity desc."""
    merged: List[Dict] = list(spotify_results)
    for yt in youtube_results:
        yt_key = _norm_for_compare(f"{yt['artist']} {yt['title']}")
        is_dup = False
        for sp in spotify_results:
            sp_key = _norm_for_compare(f"{sp['artist']} {sp['title']}")
            if difflib.SequenceMatcher(None, yt_key, sp_key).ratio() >= 0.85:
                is_dup = True
                break
        if not is_dup:
            merged.append(yt)
    merged.sort(key=lambda c: -c["similarity"])
    return merged


def _candidate_to_track_meta(c: Dict) -> Dict:
    """Reshape a search candidate into the same dict shape as _resolve_spotify / _resolve_youtube."""
    base = {
        "platform": "search",
        "title": c.get("title", ""),
        "artist": c.get("artist", ""),
        "album": c.get("album", ""),
        "artwork": c.get("artwork"),
    }
    if c.get("source") == "spotify":
        base["spotify_id"] = c.get("spotify_id")
        base["spotify_url"] = c.get("spotify_url")
        base["embed_url"] = c.get("embed_url")
    else:
        base["youtube_id"] = c.get("youtube_id")
        base["youtube_url"] = c.get("youtube_url")
        base["embed_url"] = c.get("embed_url")
        base["channel"] = c.get("channel", "")
    return base


def _resolve_search_with_picker(query: str) -> Dict:
    """
    For a free-text query, returns one of:
      {"kind": "auto", "track_meta": {...}}          -> confident Spotify match
      {"kind": "candidates", "candidates": [...]}    -> ambiguous, ask the user
      {"kind": "none"}                               -> no result anywhere
    """
    spotify_hits = _search_spotify_candidates(query, limit=5)
    youtube_hits = _search_youtube_candidates(query, limit=5)

    #if spotify's top result is a strong match we use it silently
    if spotify_hits and spotify_hits[0]["similarity"] >= SPOTIFY_AUTO_THRESHOLD:
        return {"kind": "auto",
                "track_meta": _candidate_to_track_meta(spotify_hits[0])}

    merged = _dedupe_and_rank(spotify_hits, youtube_hits)
    if not merged:
        return {"kind": "none"}

    candidates = merged[:RESOLUTION_CANDIDATES_MAX]
    return {"kind": "candidates",
            "candidates": [_candidate_to_track_meta(c) | {"_similarity": c["similarity"]}
                           for c in candidates]}


# AI DETECTION

def _ai_verdict(pct: float) -> str:
    if pct > 70: return "Very likely AI"
    if pct > 40: return "Possibly AI"
    if pct > 20: return "Probably human"
    return "Likely human"


def _run_ai_check_for_track(job_id: str, track_meta: Dict) -> Dict:
    """Pick which URL to AI-check, then run it. Prefers spotify_url. If only youtube_url is available,
    try a Spotify cross-match first; fall back to YouTube otherwise."""
    spotify_url = track_meta.get("spotify_url")
    youtube_url = track_meta.get("youtube_url")

    if spotify_url:
        return _run_ai_check(job_id, spotify_url)

    if youtube_url:
        cross = _cross_match_spotify(job_id, track_meta)
        if cross is not None:
            _log_job(
                job_id,
                f"Cross-matched YouTube track to Spotify "
                f"({cross['artist']} - {cross['title']}, sim={cross['similarity']:.2f}, "
                f"via interp {cross.get('via', '?')}); "
                f"running AI check on Spotify URL instead.")
            #also pass the Spotify URL/id to the recommender as the spotify one is better
            track_meta["spotify_url"] = cross["spotify_url"]
            track_meta["spotify_id"] = cross["spotify_id"]
            # write the verified spotify metadata back so the track card shows the corrected artist/title
            #(artwork stays as the youTube thumbnail -the user's intent was still the YouTube video)
            track_meta["artist"] = cross["artist"]
            track_meta["title"]  = cross["title"]
            ai = _run_ai_check(job_id, cross["spotify_url"])
            if isinstance(ai, dict):
                ai["checked_source"] = "Spotify (via YouTube link)"
                ai["cross_matched"] = True
            return ai
        _log_job(job_id, "No confident Spotify cross-match; AI-checking YouTube URL.")
        return _run_ai_check(job_id, youtube_url)

    return {"probability": None, "verdict": "Unverified",
            "status": "no_url", "cached": False}


def _cross_match_spotify(job_id: str, track_meta: Dict) -> Optional[Dict]:
    """Cross-match a YouTube track to its Spotify equivalent by trying three
    interpretations of the title (Artist-Song, Channel+full-title, Channel+song-before-dash)
    and picking the highest-similarity match above CROSS_MATCH_THRESHOLD."""
    if spotify_client is None:
        return None

    cleaned_title = (track_meta.get("cleaned_title")
                     or track_meta.get("title")
                     or "")
    channel = track_meta.get("channel") or ""
    fallback_artist = track_meta.get("artist") or ""
    fallback_title = track_meta.get("title") or ""

    if not cleaned_title and not fallback_title:
        return None

    #Build the three interpretations
    interpretations = []

    #interpretation A: cleaned_title is "Artist - Song"
    if " - " in cleaned_title:
        a_part, t_part = cleaned_title.split(" - ", 1)
        interpretations.append({
            "label": "A (Artist - Song)",
            "artist": a_part.strip(),
            "title":  t_part.strip(),
        })
    else:
        #no "-" to split on=> we use the parser's choice as Interpretation A
        interpretations.append({
            "label": "A (parser-default)",
            "artist": fallback_artist,
            "title":  fallback_title,
        })

    # strip " -Topic" part
    channel_artist = channel
    if channel_artist.endswith(" - Topic"):
        channel_artist = channel_artist[: -len(" - Topic")].strip()

    if channel_artist:
        #interpretation B: channel-as-artist, full cleaned_title as the title
        # (indie channels where the channel IS the artist & no dashin title)
        candidate_B = {
            "label": "B (channel + full-title)",
            "artist": channel_artist,
            "title":  cleaned_title or fallback_title,
        }
        already_have_B = any(
            i["artist"].lower() == candidate_B["artist"].lower()
            and i["title"].lower() == candidate_B["title"].lower()
            for i in interpretations
        )
        if not already_have_B:
            interpretations.append(candidate_B)

        #interpretation C: channel-as-artist, left side of dash as title
        #(eg "Call of the North - A Timeless Nordic Folk Song" by "Timeless Echoes of Middangeard" -> last is descriptive tag, not song name)
        if " - " in cleaned_title:
            left_of_dash = cleaned_title.split(" - ", 1)[0].strip()
            if left_of_dash and len(left_of_dash) >= 2:
                candidate_C = {
                    "label": "C (channel + song-before-dash)",
                    "artist": channel_artist,
                    "title":  left_of_dash,
                }
                already_have_C = any(
                    i["artist"].lower() == candidate_C["artist"].lower()
                    and i["title"].lower() == candidate_C["title"].lower()
                    for i in interpretations
                )
                if not already_have_C:
                    interpretations.append(candidate_C)

    #run each interpretation against spotify, score each best hit
    best_match = None
    best_score = 0.0

    for interp in interpretations:
        query = f"{interp['artist']} {interp['title']}".strip()
        if not query:
            continue
        candidates = _search_spotify_candidates(query, limit=5)
        if not candidates:
            _log_job(job_id,
                     f"  cross-match interpretation {interp['label']}: "
                     f"no Spotify results for '{query[:80]}'")
            continue

        #find the best candidate that also has a plausible artist match
        for c in candidates:
            combined_sim = _similarity(query, f"{c['artist']} {c['title']}")
            if combined_sim < CROSS_MATCH_THRESHOLD:
                continue
            if not _artist_plausibly_matches(interp["artist"], c["artist"]):
                continue
            if combined_sim > best_score:
                best_score = combined_sim
                best_match = {
                    "spotify_url": c["spotify_url"],
                    "spotify_id":  c["spotify_id"],
                    "artist":      c["artist"],
                    "title":       c["title"],
                    "similarity":  combined_sim,
                    "via":         interp["label"],
                }
            # candidates are sorted by similarity desc, so the first match passing thresholds is the best
            break

        #per-interpretation log line so the user can see which one worked
        if best_match and best_match["via"] == interp["label"]:
            _log_job(job_id,
                     f"  cross-match interpretation {interp['label']}: "
                     f"matched '{best_match['artist']} - {best_match['title']}' "
                     f"at sim={best_match['similarity']:.2f}")
        else:
            _log_job(job_id,
                     f"  cross-match interpretation {interp['label']}: "
                     f"no candidate passed thresholds")

    return best_match


def _artist_plausibly_matches(query_artist: str, candidate_artist: str) -> bool:
    """True if the candidate artist plausibly matches; tolerant of short single-word names."""
    q_norm = _norm_for_compare(query_artist)
    c_norm = _norm_for_compare(candidate_artist)
    if not q_norm or not c_norm:
        return False
    if q_norm == c_norm:
        return True
    if q_norm in c_norm or c_norm in q_norm:
        return True
    return _similarity(query_artist, candidate_artist) >= 0.70


def _run_ai_check(job_id: str, url: Optional[str]) -> Dict:
    if not url:
        _log_job(job_id, "No URL available for AI check.")
        return {"probability": None, "verdict": "Unverified",
                "status": "no_url", "cached": False}

    if "spotify.com" in url:
        source = "Spotify"
    elif "youtube.com" in url or "youtu.be" in url:
        source = "YouTube"
    else:
        source = "URL"

    if letssubmit_cache.has(url):
        prob = letssubmit_cache.get_cached(url)
        msg = (f"AI check on {source} (cache hit): {prob:.1f}%"
               if prob is not None else f"AI check on {source} (cache hit): null")
        _log_job(job_id, msg)
        return {"probability": prob,
                "verdict": _ai_verdict(prob) if prob is not None else "Unverified",
                "status": "cached" if prob is not None else "cache_null",
                "cached": True,
                "checked_url": url, "checked_source": source}

    _log_job(job_id, f"AI check on {source} (calling LetsSubmit)...")
    prob, status = letssubmit_cache.check(url)

    if status == "quota_exhausted":
        _mark_job_quota_exhausted(job_id)
        _log_job(job_id, "LetsSubmit daily quota exhausted. Try again after reset.")
        return {"probability": None, "verdict": "Unverified",
                "status": "quota_exhausted", "cached": False,
                "checked_url": url, "checked_source": source}

    if prob is None:
        _log_job(job_id, f"AI check unavailable ({status}).")
        return {"probability": None, "verdict": "Unverified",
                "status": status, "cached": False,
                "checked_url": url, "checked_source": source}

    _log_job(job_id, f"AI check on {source}: {prob:.1f}%")
    return {"probability": prob, "verdict": _ai_verdict(prob),
            "status": "fresh", "cached": False,
            "checked_url": url, "checked_source": source}

# COVER ART AI DETECTION 

def _cover_ai_verdict(pct: float) -> str:
    if pct > 70: return "Very likely AI-generated"
    if pct > 40: return "Possibly AI-generated"
    if pct > 20: return "Probably human-made"
    return "Likely human-made"


def _run_cover_check(job_id: str, artwork_url: Optional[str]) -> Dict:
    """Sightengine cover-art check. Runs only on the submitted track. Informational only -never affects the audio verdict."""
    if not artwork_url:
        return {"probability": None, "verdict": "No cover art",
                "status": "no_url", "cached": False}

    if sightengine_cache.has(artwork_url):
        prob = sightengine_cache.get_cached(artwork_url)
        _log_job(job_id, f"Cover art check (cache hit): "
                         f"{prob}%" if prob is not None else "Cover art check (cache hit): null")
        return {"probability": prob,
                "verdict": _cover_ai_verdict(prob) if prob is not None else "Unverified",
                "status": "cached", "cached": True}

    if not SIGHTENGINE_API_USER or not SIGHTENGINE_API_SECRET:
        _log_job(job_id, "Cover art check skipped: Sightengine credentials not configured.")
        return {"probability": None, "verdict": "Unverified",
                "status": "no_credentials", "cached": False}

    _log_job(job_id, "Cover art check (calling Sightengine)...")
    prob, status = sightengine_cache.check(artwork_url)

    if status == "quota_exhausted":
        _log_job(job_id, "Sightengine monthly/daily quota exhausted.")
        return {"probability": None, "verdict": "Unverified",
                "status": "quota_exhausted", "cached": False}

    if prob is None:
        _log_job(job_id, f"Cover art check unavailable ({status}).")
        return {"probability": None, "verdict": "Unverified",
                "status": status, "cached": False}

    _log_job(job_id, f"Cover art check: {prob:.1f}% AI probability.")
    return {"probability": prob, "verdict": _cover_ai_verdict(prob),
            "status": "fresh", "cached": False}

def _mark_job_quota_exhausted(job_id: str):
    """Set the quota_exhausted flag on the job and trip the Spotify adapter."""
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["quota_exhausted"] = True
    #trip the spotify recommender's adapter so in-flight calls short-circuit instead of re-hitting the API
    if spotify_recommender is not None:
        adapter = getattr(spotify_recommender, "ai_detector", None)
        if adapter is not None and hasattr(adapter, "quota_tripped"):
            adapter.quota_tripped = True


def _is_job_quota_exhausted(job_id: str) -> bool:
    with JOBS_LOCK:
        return bool(JOBS.get(job_id, {}).get("quota_exhausted", False))


#RECOMMENDATION DISPATCH

def _dispatch_recommendations(job_id: str, track_meta: Dict, ai: Dict):
    """Cascade: hybrid (MSD+tags) -> Spotify Discovered-On -> YouTube top-up. Aborts on quota exhaustion."""
    #if the AI check itself tripped the 429, no point even starting
    if _is_job_quota_exhausted(job_id):
        _log_job(job_id, "Quota exhausted at AI-check step. No recommendations.")
        return

    prob = ai.get("probability")
    force_recs = (prob is None)  

    if prob is not None and prob < 35 and not force_recs:
        _log_job(job_id, "AI check rated human; skipping recommendations.")
        return

    artist = track_meta.get("artist", "")
    title = track_meta.get("title", "")
    spotify_url = track_meta.get("spotify_url")

    #Tier 1: Hybrid 
    hybrid_recs = _try_hybrid(job_id, artist, title)
    if hybrid_recs:
        _log_job(job_id, f"Hybrid produced {len(hybrid_recs)} recs.")
        _emit_hybrid_recs(job_id, hybrid_recs)
        return

    if _is_job_quota_exhausted(job_id):
        _log_job(job_id, "Quota tripped during hybrid. Aborting cascade.")
        return

    #target number of recommendations we always try to reach
    TARGET_RECS = 10

    #Tier 2: Spotify Discovered-On 
    spotify_recs: List[Dict] = []
    if spotify_url and spotify_recommender is not None:
        _log_job(job_id, "Trying Spotify Discovered-On...")
        spotify_recs = _try_spotify_discovered_on(job_id, spotify_url)
        if _is_job_quota_exhausted(job_id):
            if spotify_recs:
                _log_job(job_id, f"Quota tripped mid-run; emitting {len(spotify_recs)} pre-verified rec(s).")
                _emit_spotify_recs(job_id, spotify_recs)
            else:
                _log_job(job_id, "Quota tripped during Spotify Discovered-On. Aborting cascade.")
            return
        if spotify_recs:
            _log_job(job_id, f"Spotify Discovered-On produced {len(spotify_recs)} recs.")
            _emit_spotify_recs(job_id, spotify_recs)
            #if spotify already filled the quota we're done
            if len(spotify_recs) >= TARGET_RECS:
                return
            #otherwise fall through to top up the remainder with YouTube
            _log_job(job_id,
                     f"Only {len(spotify_recs)} Spotify recs (< {TARGET_RECS}); "
                     f"topping up with YouTube.")
        else:
            _log_job(job_id, "Spotify Discovered-On returned nothing.")
    elif not spotify_url:
        _log_job(job_id, "No Spotify URL for query; skipping Spotify Discovered-On.")
    elif spotify_recommender is None:
        #Spotify URL present but the recommender failed to init at boot 
        _log_job(job_id,
                 "Spotify URL present but recommender unavailable "
                 "(init failed at boot); check stdout for init error.")

    #re-check quota before spending more on YouTube verification.
    if _is_job_quota_exhausted(job_id):
        _log_job(job_id, "Quota exhausted before YouTube top-up. Stopping.")
        return

    #Tier 3: YouTube Discovered-On (fill remaining slots)
    remaining = TARGET_RECS - len(spotify_recs)
    if remaining <= 0:
        return

    if spotify_recs:
        _log_job(job_id, f"Filling {remaining} remaining slot(s) via YouTube.")
    else:
        _log_job(job_id, "Falling back to YouTube Discovered-On.")

    #build the set of (artist, title) pairs alr emitted by spotify so youTube doesn't recommend a duplicate 
    already = {
        (str(r.get("artist", "")).lower().strip(),
         str(r.get("name", "")).lower().strip())
        for r in spotify_recs
    }
    _emit_youtube_recs(job_id, track_meta,
                       max_recommendations=remaining,
                       exclude_pairs=already)


def _try_spotify_discovered_on(job_id: str, spotify_url: str) -> List[Dict]:
    """Run the Spotify Discovered-On recommender. After the first 429, the adapter's
    quota_tripped flag short-circuits any further LetsSubmit calls."""
    if spotify_recommender is None:
        return []
    try:
        recs = spotify_recommender.get_discovered_on_recommendations(
            track_url=spotify_url,
            ai_threshold=40.0,
            original_track_ai_threshold=0.0,
            artist_discography_ai_threshold=60.0,
            max_playlists_to_check=20,
            max_recommendations=10,
            rate_limit_delay=0.5,
            skip_first_n_playlists=1,
            discography_sample_size=10,
        )
    except Exception as e:
        _log_job(job_id, f"Spotify Discovered-On error: {e}")
        return []

    #detect if the adapter tripped quota during the call
    adapter = getattr(spotify_recommender, "ai_detector", None)
    if adapter is not None and getattr(adapter, "quota_tripped", False):
        _mark_job_quota_exhausted(job_id)
        _log_job(job_id, "Spotify recommender hit LetsSubmit 429 mid-run.")
        #only candidates still in the queue when quota died are unverified.
        if recs:
            _log_job(job_id, f"Keeping {len(recs)} pre-verified rec(s) from before quota trip.")
        return recs or []

    return recs or []


def _emit_spotify_recs(job_id: str, recs: List[Dict]):
    for r in recs:
        # genre-fallback uses 'track_id'/'spotify_id'; discovered-On uses 'id'
        track_id = r.get("id") or r.get("spotify_id") or r.get("track_id")
        embed_url = r.get("embed_url") or (
            f"https://open.spotify.com/embed/track/{track_id}" if track_id else None
        )
        # use artwork already fetched by the recommender, only re-fetch if missing
        artwork = r.get("artwork")
        if not artwork and track_id and spotify_client is not None:
            try:
                info = spotify_client.get_track_info(track_id)
                imgs = (info or {}).get("album", {}).get("images", [])
                if imgs:
                    artwork = imgs[0]["url"]
            except Exception:
                pass
        _append_recommendation(job_id, {
            "platform":   "spotify",
            "title":      r.get("name") or r.get("title"),
            "artist":     r.get("artist"),
            "url":        r.get("url") or r.get("spotify_url"),
            "spotify_id": track_id,
            "embed_url":  embed_url,
            "artwork":    artwork,
            "verification_method":     r.get("method") or "spotify-discovered-on",
            "verification_pct":        r.get("ai_probability", 0),
            "discovered_in_playlist":  r.get("discovered_on") or r.get("fallback_playlist", ""),
        })
        time.sleep(0.1)


def _try_hybrid(job_id: str, artist: str, title: str) -> List[Dict]:
    hr = get_hybrid_recommender()
    if hr is None:
        _log_job(job_id, "Hybrid recommender not available.")
        return []
    cf = getattr(hr, "cf", None)
    if cf is None:
        _log_job(job_id, "Hybrid CF model missing.")
        return []

    try:
        clean_title = _strip_version_suffix(title)
        track_id = cf.search_track_by_metadata(artist=artist, title=clean_title)
    except Exception as e:
        _log_job(job_id, f"MSD lookup error: {e}")
        return []
    if not track_id:
        _log_job(job_id, f"Not found in MSD: {artist} - {title}")
        return []

    if track_id not in hr.track_tags:
        _log_job(job_id, f"MSD hit but no Last.fm tags for {track_id}; skipping hybrid.")
        return []

    _log_job(job_id, f"MSD hit with tags: {track_id}- running hybrid...")
    try:
        recs = hr.recommend(track_id, n_final=10)
    except Exception as e:
        _log_job(job_id, f"Hybrid recommender error: {e}")
        return []

    artist_lower = artist.lower().strip()
    keep = []
    for r in recs:
        tid = r.get("track_id")
        if not tid:
            continue
        rows = hr.tracks_df[hr.tracks_df["track_id"] == tid]
        if rows.empty:
            continue
        rec_artist = str(rows.iloc[0].get("artist", "")).strip()
        rec_title  = str(rows.iloc[0].get("title",  "")).strip()
        if not rec_artist or not rec_title:
            continue
        if rec_artist.lower() == artist_lower:
            continue
        if blacklist.is_blacklisted(rec_artist):
            continue
        keep.append({
            "track_id":    tid,
            "artist":      rec_artist,
            "title":       rec_title,
            "tag_score":   r.get("tag_score",   0),
            "cf_score":    r.get("cf_score",    0),
            "audio_score": r.get("audio_score", 0),
            "final_score": r.get("final_score", 0),
        })
    return keep


def _emit_hybrid_recs(job_id: str, hybrid_recs: List[Dict]):
    for r in hybrid_recs:
        artist = r["artist"]
        title  = r["title"]

        rec_dict = {
            "title":  title,
            "artist": artist,
            "verification_method": "hybrid-msd",
            "verification_pct":    0.0,
            "scores": {
                "tag":   round(r.get("tag_score", 0), 3),
                "cf":    round(r.get("cf_score", 0), 3),
                "audio": round(r.get("audio_score", 0), 3),
                "final": round(r.get("final_score", 0), 3),
            },
        }

        #try youyube first
        playable = _find_youtube_for_track(artist, title)
        if playable and playable.get("embed_url"):
            rec_dict.update({
                "platform":  "hybrid",
                "url":       playable.get("url"),
                "video_id":  playable.get("video_id"),
                "embed_url": playable.get("embed_url"),
                "artwork":   playable.get("artwork"),
            })
        else:
            #fall back to spotify
            track = None
            if spotify_client is not None:
                try:
                    track = spotify_client.search_track(artist, title)
                except Exception as e:
                    _log_job(job_id, f"Hybrid: Spotify search failed for {artist} - {title}: {e}")

            if track and track.get("id"):
                track_id = track["id"]
                imgs = track.get("album", {}).get("images", [])
                rec_dict.update({
                    "platform":   "spotify",
                    "url":        track.get("external_urls", {}).get("spotify"),
                    "spotify_id": track_id,
                    "embed_url":  f"https://open.spotify.com/embed/track/{track_id}",
                    "artwork":    imgs[0]["url"] if imgs else None,
                })
            else:
                rec_dict.update({
                    "platform":  "hybrid",
                    "url":       None,
                    "video_id":  None,
                    "embed_url": None,
                    "artwork":   None,
                })
                _log_job(job_id, f"Hybrid rec not playable on either platform: {artist} - {title}")

        _append_recommendation(job_id, rec_dict)
        time.sleep(0.1)


def _emit_youtube_recs(job_id: str, track_meta: Dict,
                       max_recommendations: int = 10,
                       exclude_pairs: Optional[set] = None):
    """Emit YouTube Discovered-On recs.
    max_recommendations: how many to request (when topping up Spotify, this is the remaining slots).
    exclude_pairs:       (artist_lower, title_lower) tuples already emitted by an earlier tier;
                        matches are dropped so we never show the same track twice across tiers.
    """
    if youtube_recommender is None:
        _log_job(job_id, "YouTube recommender unavailable.")
        return
    if max_recommendations <= 0:
        return
    exclude_pairs = exclude_pairs or set()

    artist = track_meta.get("artist", "")
    title = track_meta.get("title", "")
    # the input's own video_id, so it doesn't get recommended back to itself
    input_video_id = track_meta.get("youtube_id")

    #callback for a 429 the recommender does so the worker can flag the job and tell all other tiers to stop
    def on_quota_trip():
        _mark_job_quota_exhausted(job_id)
        _log_job(job_id, "YouTube recommender hit LetsSubmit 429 mid-run.")

    try:
        recs = youtube_recommender.get_recommendations(
                    artist=artist, title=title,
                    max_playlists=20,
                    #request extra to absorb dedup losses against exclude_pairs.
                    max_recommendations=max_recommendations + len(exclude_pairs),
                    strict_mode=True,
                    on_quota_trip=on_quota_trip,
                    exclude_video_ids={input_video_id} if input_video_id else None,
                )
        
    except Exception as e:
        _log_job(job_id, f"YouTube recommender error: {e}")
        return

    if _is_job_quota_exhausted(job_id):
        _log_job(job_id, "Quota tripped during YouTube pass; dropping unverified results.")
        return

    emitted = 0
    for r in recs:
        if emitted >= max_recommendations:
            break
        pair = (str(r.get("artist", "")).lower().strip(),
                str(r.get("title", "")).lower().strip())
        if pair in exclude_pairs:
            continue
        _append_recommendation(job_id, {
            "platform": "youtube",
            "title": r.get("title"),
            "artist": r.get("artist"),
            "url": r.get("url"),
            "video_id": r.get("video_id"),
            "embed_url": (f"https://www.youtube.com/embed/{r['video_id']}"
                          if r.get("video_id") else None),
            "artwork": r.get("artwork"),
            "verification_method": r.get("verification_method"),
            "verification_pct": r.get("verification_pct"),
        })
        emitted += 1
        time.sleep(0.15)

    if emitted:
        _log_job(job_id, f"YouTube added {emitted} rec(s).")

def _check_youtube_ai_label(video_id: str, yt_api_key: str) -> Dict:
    """Cheap pre-check for self-disclosed AI content in the YouTube title/tags."""
    try:
        import requests as _req
        resp = _req.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "snippet", "id": video_id, "key": yt_api_key},
            timeout=8,
        )
        items = resp.json().get("items", [])
        if not items:
            return {"disclosed_as_ai": False, "source": "no_video_found"}

        snippet    = items[0].get("snippet", {})
        tags       = [t.lower() for t in (snippet.get("tags") or [])]
        title_lower = (snippet.get("title") or "").lower()

        ai_tagged = any(t in ("ai", "ai music", "ai generated",
                               "artificial intelligence") for t in tags)
        ai_in_title = any(kw in title_lower for kw in
                          ("ai generated", "ai music", "artificial intelligence",
                           "made with ai", "created with ai"))

        return {
            "disclosed_as_ai": ai_tagged or ai_in_title,
            "source": "tags" if ai_tagged else "title" if ai_in_title else "none",
        }
    except Exception as e:
        return {"disclosed_as_ai": False, "source": "error", "error": str(e)}


def _find_youtube_for_track(artist: str, title: str) -> Optional[Dict]:
    if not YOUTUBE_API_KEY:
        return None
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": f"{artist} {title}",
                "type": "video",
                "videoCategoryId": "10",
                "maxResults": 1,
                "key": YOUTUBE_API_KEY,
            },
            timeout=8,
        )
        if r.status_code != 200:
            return None
        items = r.json().get("items", [])
        if not items:
            return None
        vid = items[0].get("id", {}).get("videoId")
        if not vid:
            return None
        thumbs = items[0].get("snippet", {}).get("thumbnails", {})
        artwork = (thumbs.get("high") or thumbs.get("medium")
                   or thumbs.get("default") or {}).get("url")
        return {
            "url": f"https://www.youtube.com/watch?v={vid}",
            "video_id": vid,
            "embed_url": f"https://www.youtube.com/embed/{vid}",
            "artwork": artwork,
        }
    except Exception:
        return None


# ROUTES

@app.route("/")
def index():
    return render_template("index.html")


#pending resolutions between /api/check and /api/resolve, keyed by short token (10-min TTL)
RESOLUTIONS: Dict[str, Dict] = {}
RESOLUTIONS_LOCK = threading.Lock()
RESOLUTION_TTL_SECONDS = 600


def _purge_old_resolutions():
    now = time.time()
    with RESOLUTIONS_LOCK:
        stale = [k for k, v in RESOLUTIONS.items()
                 if now - v.get("created_at", 0) > RESOLUTION_TTL_SECONDS]
        for k in stale:
            RESOLUTIONS.pop(k, None)


def _new_resolution(query: str, candidates: List[Dict]) -> str:
    token = secrets.token_hex(6)
    with RESOLUTIONS_LOCK:
        RESOLUTIONS[token] = {
            "query": query,
            "candidates": candidates,
            "created_at": time.time(),
        }
    return token


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or {}
    raw = (data.get("query") or "").strip()
    detect = _identify_input(raw)
    if detect["kind"] == "empty":
        return jsonify({"error": "Empty input"}), 400

    _purge_old_jobs()
    _purge_old_resolutions()

    #URL inputs: start the job immediately,no picker needed
    if detect["kind"] in ("spotify", "youtube"):
        job_id = _new_job()
        threading.Thread(target=_worker, args=(job_id, detect), daemon=True).start()
        return jsonify({"state": "queued", "job_id": job_id})

    #free-text: search both Spotify and YouTube, decide whether to ask
    if detect["kind"] == "search":
        resolved = _resolve_search_with_picker(detect["query"])

        if resolved["kind"] == "none":
            return jsonify({"error": "No matching track found anywhere."}), 404

        if resolved["kind"] == "auto":
            #confident spotify match -> start the job directly
            job_id = _new_job()
            payload = {"kind": "resolved", "track_meta": resolved["track_meta"]}
            threading.Thread(target=_worker, args=(job_id, payload),
                             daemon=True).start()
            return jsonify({"state": "queued", "job_id": job_id})

        #ambiguous: return candidates for the picker
        token = _new_resolution(detect["query"], resolved["candidates"])
        return jsonify({
            "state": "pending_resolution",
            "resolution_token": token,
            "query": detect["query"],
            "candidates": resolved["candidates"],
        })

    return jsonify({"error": "Unknown input kind"}), 400


@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    """Continue a pending resolution: start a job for the candidate the user picked."""
    data = request.get_json(silent=True) or {}
    token = (data.get("resolution_token") or "").strip()
    index = data.get("candidate_index")
    if not token or index is None:
        return jsonify({"error": "resolution_token and candidate_index required"}), 400

    with RESOLUTIONS_LOCK:
        entry = RESOLUTIONS.pop(token, None)
    if entry is None:
        return jsonify({"error": "Unknown or expired resolution token"}), 404

    try:
        index = int(index)
    except (TypeError, ValueError):
        return jsonify({"error": "candidate_index must be an integer"}), 400
    if index < 0 or index >= len(entry["candidates"]):
        return jsonify({"error": "candidate_index out of range"}), 400

    #strip _similarity before handing to the worker
    track_meta = dict(entry["candidates"][index])
    track_meta.pop("_similarity", None)

    job_id = _new_job()
    payload = {"kind": "resolved", "track_meta": track_meta}
    threading.Thread(target=_worker, args=(job_id, payload), daemon=True).start()
    return jsonify({"state": "queued", "job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        return jsonify(dict(job))


@app.route("/api/job/<job_id>/force_recs", methods=["POST"])
def api_force_recs(job_id):
    """'Show me recommendations anyway' override after a human verdict: re-runs the cascade regardless."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        if job["state"] not in ("done", "error"):
            return jsonify({"error": "Job not yet complete"}), 400
        track_meta = job.get("track")
        ai_dict = dict(job.get("ai") or {})
        if not track_meta:
            return jsonify({"error": "No track metadata on job"}), 400

    #force the recs path even if the track was human-rated
    forced_ai = dict(ai_dict)
    forced_ai["probability"] = None  #makes force_recs trigger in dispatch

    def _continue_worker():
        _update_job(job_id, state="recommending")
        _log_job(job_id, "User requested recommendations anyway; running dispatch.")
        try:
            _dispatch_recommendations(job_id, track_meta, forced_ai)
        except Exception as e:
            traceback.print_exc()
            _update_job(job_id, state="error",
                        error=f"Force-recs error: {type(e).__name__}: {e}")
            _persist_job(job_id)
            return
        _update_job(job_id, state="done")
        _log_job(job_id, "Force-recs run complete.")
        _persist_job(job_id)

    threading.Thread(target=_continue_worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})

#YouTube reverse-search for Spotify-only recs


YT_REVERSE_CACHE: Dict[str, Optional[Dict]] = {}
YT_REVERSE_LOCK = threading.Lock()#minimum similarity needed to accept a YouTube reverse-match
#same threshold as spotify cross-match for consistency
YT_REVERSE_THRESHOLD = 0.80


def _yt_reverse_key(artist: str, title: str) -> str:
    """Stable cache key from normalized artist + title."""
    return f"{_norm_for_compare(artist)}|||{_norm_for_compare(title)}"


@app.route("/api/youtube_lookup", methods=["POST"])
def api_youtube_lookup():
    """Search YouTube for artist+title, return a playable video_id if similarity passes threshold. Cached per (artist, title)."""
    data = request.get_json(silent=True) or {}
    artist = (data.get("artist") or "").strip()
    title = (data.get("title") or "").strip()

    if not artist and not title:
        return jsonify({"error": "artist and/or title required"}), 400

    key = _yt_reverse_key(artist, title)
    with YT_REVERSE_LOCK:
        if key in YT_REVERSE_CACHE:
            cached = YT_REVERSE_CACHE[key]
            if cached is None:
                return jsonify({"video_id": None, "reason": "cached-no-match"})
            return jsonify({**cached, "cached": True})

    #build search query
    query = f"{artist} {title}".strip() if (artist and title) else (artist or title)
    candidates = _search_youtube_candidates(query, limit=5)

    if not candidates:
        with YT_REVERSE_LOCK:
            YT_REVERSE_CACHE[key] = None
        return jsonify({"video_id": None, "reason": "no-results"})

    #best candidate that clears the threshold and has a plausible artist match (same helper the Spotify cross-match uses)
    best = None
    best_score = 0.0
    for c in candidates:
        combined_sim = _similarity(
            f"{artist} {title}",
            f"{c.get('artist','')} {c.get('title','')}",
        )
        if combined_sim < YT_REVERSE_THRESHOLD:
            continue
        if artist and not _artist_plausibly_matches(artist, c.get("artist", "")):
            continue
        if combined_sim > best_score:
            best_score = combined_sim
            best = c

    if best is None:
        with YT_REVERSE_LOCK:
            YT_REVERSE_CACHE[key] = None
        return jsonify({"video_id": None, "reason": "no-confident-match"})

    result = {
        "video_id": best.get("youtube_id"),
        "embed_url": best.get("embed_url"),
        "title": best.get("title"),
        "artist": best.get("artist"),
        "channel": best.get("channel", ""),
        "similarity": round(best_score, 3),
    }
    with YT_REVERSE_LOCK:
        YT_REVERSE_CACHE[key] = result
    return jsonify(result)


if __name__ == "__main__":
    #auto-reload only on files we actively edit - Windows FS events trigger infinite reload loops on untouched scripts
    app.run(
        host="127.0.0.1", port=5000,
        debug=True, threaded=True,
        use_reloader=False,
        exclude_patterns=[
            #standalone scripts not imported by the running web app
            "*/cascading_hybrid_recommender.py",
            "*/parse_*.py",
            "*/prefetch_*.py",
            "*/rebuild_*.py",
            "*/train_*.py",
            "*/analyze_*.py",
            "*/get_mbids.py",
            "*/improved_acousticbrainz.py",
            "*/test_*.py",
            "*/verify_*.py",
            "*/apis/lastfm_client.py",
            "*/apis/youtube_client.py",
            #module trees not used by the running web app
            "*/mbid_recommender/*",
            "*/src/*",
            #build/cache/data directories (defensive)
            "*/data/*",
            "*/models/*",
            "*/__pycache__/*",
            "*/site-packages/*",
            "*/anaconda3/*",
        ],
    )