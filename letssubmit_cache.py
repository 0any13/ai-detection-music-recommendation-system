import json
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


CACHE_FILE = Path("data/letssubmit_cache.json")
LETSSUBMIT_ENDPOINT = "https://api.letssubmit.com/analyze_song"
DEFAULT_TIMEOUT = 60


class LetsSubmitCache:
    def __init__(self, api_key: Optional[str] = None,
                 cache_file: Path = CACHE_FILE):
        self.api_key = api_key or os.getenv("LETSSUBMIT_API_KEY")
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load()


    def _load(self) -> dict:
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[letssubmit_cache] Loaded {len(data)} cached entries.")
            return data
        except Exception as e:
            print(f"[letssubmit_cache] Failed to load cache: {e}")
            return {}

    def _save(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"[letssubmit_cache] Failed to save cache: {e}")


    def has(self, url: str) -> bool:
        return url in self._cache

    def get_cached(self, url: str) -> Optional[float]:
        entry = self._cache.get(url)
        if entry is None:
            return None
        return entry.get("ai_probability")

    def check(self, url: str, force_refresh: bool = False
              ) -> Tuple[Optional[float], str]:
        """
        Returns (probability_or_None, status_string).We should distinguish quota exhaustion from other failures."""
        if not force_refresh and url in self._cache:
            prob = self._cache[url].get("ai_probability")
            return prob, ("cached" if prob is not None else "cached_null")

        if not self.api_key:
            print("[letssubmit_cache] No API key set.")
            return None, "no_api_key"

        prob, status = self._call_live(url)
        if prob is not None:
            self._cache[url] = {
                "ai_probability": prob,
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save()
        return prob, status

    def _call_live(self, url: str) -> Tuple[Optional[float], str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"url": url}
        try:
            response = requests.post(
                LETSSUBMIT_ENDPOINT, headers=headers, json=data,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.exceptions.Timeout:
            print(f"[letssubmit_cache] Timeout for {url}")
            return None, "timeout"
        except Exception as e:
            print(f"[letssubmit_cache] Request error: {e}")
            return None, "request_error"

        if response.status_code == 429:
            print(f"[letssubmit_cache] 429 quota exhausted: {response.text[:120]}")
            return None, "quota_exhausted"
        if response.status_code == 401:
            print("[letssubmit_cache] 401 invalid API key.")
            return None, "auth_error"
        if response.status_code >= 500:
            print(f"[letssubmit_cache] {response.status_code} server error.")
            return None, "server_error"
        if response.status_code != 200:
            print(f"[letssubmit_cache] HTTP {response.status_code}: {response.text[:150]}")
            return None, "server_error"

        try:
            payload = response.json()
        except Exception:
            return None, "no_result"

        ai_prob = payload.get("ai_probability")
        if ai_prob is None or ai_prob == "-":
            return None, "no_result"
        try:
            return float(ai_prob), "ok"
        except (TypeError, ValueError):
            return None, "no_result"

    def stats(self) -> dict:
        total = len(self._cache)
        ai = sum(1 for v in self._cache.values()
                 if (v.get("ai_probability") or 0) > 40)
        return {"total": total, "ai": ai, "human": total - ai}