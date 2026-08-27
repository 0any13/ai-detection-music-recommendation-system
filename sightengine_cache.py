import json
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER", "")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET", "")
SIGHTENGINE_ENDPOINT = "https://api.sightengine.com/1.0/check.json"
CACHE_PATH = Path("data/sightengine_cache.json")
REQUEST_TIMEOUT = 15


class SightengineCache:
    """Persistent cache for results. Probabilities are returned on a 0-100 scale."""

    def __init__(
        self,
        api_user: str = SIGHTENGINE_API_USER,
        api_secret: str = SIGHTENGINE_API_SECRET,
        cache_path: Path = CACHE_PATH,
    ):
        self.api_user = api_user
        self.api_secret = api_secret
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._data = self._load()

    #public interface

    def check(self, image_url: Optional[str]) -> Tuple[Optional[float], str]:
        if not image_url:
            return None, "no_url"

        #cache hit (no API call)
        with self._lock:
            if image_url in self._data:
                return self._data[image_url].get("probability"), "cached"

        #no credentials configured
        if not self.api_user or not self.api_secret:
            return None, "error"

        #live call
        try:
            resp = requests.get(
                SIGHTENGINE_ENDPOINT,
                params={
                    "url": image_url,
                    "models": "genai",
                    "api_user": self.api_user,
                    "api_secret": self.api_secret,
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            return None, "error"

        if resp.status_code == 429:
            return None, "quota_exhausted"
        if resp.status_code != 200:
            return None, "error"

        try:
            body = resp.json()
        except ValueError:
            return None, "error"

        #Sightengine genai response shape:{"status": "success", "type": {"ai_generated": 0.0-1.0}, ...}
        #Guard against the API returning an error payload with HTTP 200.
        if body.get("status") == "failure":
            return None, "error"

        ai_score = body.get("type", {}).get("ai_generated")
        if ai_score is None:
            ai_score = body.get("ai_generated")
        if ai_score is None:
            return None, "error"

        probability = round(float(ai_score) * 100, 1)

        #persist
        with self._lock:
            self._data[image_url] = {
                "probability": probability,
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self._save()

        return probability, "fresh"

    def has(self, image_url: str) -> bool:
        with self._lock:
            return image_url in self._data

    def get_cached(self, image_url: str) -> Optional[float]:
        with self._lock:
            return self._data.get(image_url, {}).get("probability")

    #disk I/O

    def _load(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        return {}

    def _save(self):
        #caller already holds self._lock.
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass