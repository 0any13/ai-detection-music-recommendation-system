import csv
import re
from datetime import datetime
from pathlib import Path
from typing import List, Set


class AIArtistBlacklist:
    """Persistent set of AI-generated artists, read-write CSV-backed."""

    def __init__(self, blacklist_file: str = "ai_artists_blacklist.csv"):
        self.blacklist_file = Path(blacklist_file)
        self.blacklisted_artists: Set[str] = set()
        self._load_blacklist()

    # Normalization

    def _normalize_artist_name(self, artist_name: str) -> str:
        """Lowercase, strip punctuation, sort words alphabetically.
        This means 'Sienna Rose' and 'Rose Sienna' are the same, to catch artists under reordered names."""
        if not artist_name:
            return ""
        name = artist_name.lower().strip()
        name = re.sub(r"[^\w\s]", "", name)
        words = sorted(name.split())
        return " ".join(words)

    # Load /save

    def _load_blacklist(self):
        if not self.blacklist_file.exists():
            print(f"[blacklist] No file at {self.blacklist_file}, starting empty.")
            return

        try:
            with open(self.blacklist_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    artist_name = row.get("artist_name", "").strip()
                    if artist_name:
                        normalized = self._normalize_artist_name(artist_name)
                        self.blacklisted_artists.add(normalized)
            print(f"[blacklist] Loaded {len(self.blacklisted_artists)} artists from "
                  f"{self.blacklist_file}")
        except Exception as e:
            print(f"[blacklist] Error loading file: {e}")

    def _save_blacklist(self):
        """
        Preserves the original date_added for entries already on disk. Only newly added artists get the current timestamp.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        existing_dates = {}
        if self.blacklist_file.exists():
            try:
                with open(self.blacklist_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        name = self._normalize_artist_name(row.get("artist_name", ""))
                        date = row.get("date_added", "").strip()
                        if name and date:
                            existing_dates[name] = date
            except Exception:
                pass  #if read fails, all entries fall back to now_str

        try:
            with open(self.blacklist_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["artist_name", "date_added"])
                for artist in sorted(self.blacklisted_artists):
                    date_str = existing_dates.get(artist, now_str)
                    writer.writerow([artist, date_str])
        except Exception as e:
            print(f"[blacklist] Error saving: {e}")

    #Public API

    def is_blacklisted(self, artist_name: str) -> bool:
        normalized = self._normalize_artist_name(artist_name)
        return normalized in self.blacklisted_artists

    def add_to_blacklist(self, artist_name: str):
        normalized = self._normalize_artist_name(artist_name)
        if normalized and normalized not in self.blacklisted_artists:
            self.blacklisted_artists.add(normalized)
            self._save_blacklist()
            print(f"[blacklist] ADDED: '{artist_name}'")

    def remove_from_blacklist(self, artist_name: str) -> bool:
        normalized = self._normalize_artist_name(artist_name)
        if normalized in self.blacklisted_artists:
            self.blacklisted_artists.remove(normalized)
            self._save_blacklist()
            print(f"[blacklist] REMOVED: '{artist_name}'")
            return True
        print(f"[blacklist] '{artist_name}' was not in blacklist.")
        return False

    def list_blacklisted_artists(self) -> List[str]:
        return sorted(self.blacklisted_artists)

    def get_blacklist_count(self) -> int:
        return len(self.blacklisted_artists)
