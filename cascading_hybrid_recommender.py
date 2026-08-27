

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Set

from collaborative_filtering_recommender import CollaborativeFilteringRecommender


CF_COLD_THRESHOLD = 50   #listeners below this = CF confidence < 100%
MIN_TAG_SCORE = 0.0      #include any tag overlap 


class CascadingHybridRecommender:

    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.tracks_df = None
        self.features: Dict[str, Dict] = {}
        self.track_to_mbid: Dict[str, str] = {}
        self.track_tags: Dict[str, Set[str]] = {}
        self.track_listener_count: Dict[str, int] = {}
        self.cf: Optional[CollaborativeFilteringRecommender] = None

    #Loading

    def load_data(self):
        print("=" * 70)
        print("LOADING HYBRID RECOMMENDER")
        print("=" * 70)

        features_file = self.data_path / "tracks_with_features.json"
        if not features_file.exists():
            raise FileNotFoundError(
                "tracks_with_features.json not found. "
                "Run rebuild_tracks_with_features.py first."
            )
        with open(features_file, "r", encoding="utf-8") as f:
            features_list = json.load(f)
        for entry in features_list:
            tid = entry.get("track_id")
            if tid:
                self.features[tid] = entry
                mbid = entry.get("mbid")
                if mbid:
                    self.track_to_mbid[tid] = mbid
        print(f"\n1. Audio features : {len(self.features):,} tracks")

        tracks_file = self.data_path / "tracks.csv"
        self.tracks_df = pd.read_csv(tracks_file, low_memory=False)
        print(f"2. tracks.csv     : {len(self.tracks_df):,} tracks")

        tags_cache_file = self.data_path / "lastfm_tags_cache.json"
        if tags_cache_file.exists():
            with open(tags_cache_file, "r", encoding="utf-8") as f:
                tags_cache = json.load(f)
            for track_id, tags in tags_cache.items():
                if tags:
                    self.track_tags[track_id] = set(tags)
            print(f"3. Last.fm tags   : {len(self.track_tags):,} tracks with tags")
        else:
            print("3. Last.fm tags   : cache not found, run fetch_lastfm_tags.py")

        interactions_file = self.data_path / "user_interactions.csv"
        if interactions_file.exists():
            interactions = pd.read_csv(
                interactions_file, usecols=["track_id", "user_id"]
            )
            counts = interactions.groupby("track_id")["user_id"].nunique()
            self.track_listener_count = counts.to_dict()
            print(f"4. Listener counts: {len(self.track_listener_count):,} tracks")

        print("\n5. Loading CF model...")
        try:
            self.cf = CollaborativeFilteringRecommender(data_path=str(self.data_path))
            self.cf.load_latest_model()
            print(f"   CF ready: {len(self.cf.item_mapping):,} tracks")
        except Exception as e:
            print(f"   CF not available: {e}")

        print("\nReady.")

    # Main entry point

    def recommend(self, track_id: str, n_final: int = 10) -> List[Dict]:
        print("\n" + "=" * 70)
        print("HYBRID RECOMMENDATION  [Tags -> CF -> Audio]")
        print("=" * 70)

        track_row = self.tracks_df[self.tracks_df["track_id"] == track_id]
        if track_row.empty:
            print(f"Track {track_id} not found in tracks.csv")
            return []

        info = track_row.iloc[0]
        print(f"\nQuery : {info['artist']} - {info['title']}")

        listeners = self.track_listener_count.get(track_id, 0)
        cf_confidence = min(listeners, CF_COLD_THRESHOLD) / CF_COLD_THRESHOLD
        has_tags = track_id in self.track_tags
        has_audio = track_id in self.features
        in_cf = bool(self.cf and track_id in self.cf.item_mapping)

        query_tags = self.track_tags.get(track_id, set())
        query_features = self.features.get(track_id, {})

        print(f"\nData quality:")
        print(f"  Listeners : {listeners}  (CF confidence {cf_confidence:.0%})")
        print(f"  Tags      : {'yes (' + str(len(query_tags)) + ')' if has_tags else 'NO'}")
        if has_tags:
            print(f"  Top tags  : {', '.join(sorted(query_tags)[:8])}")
        print(f"  Audio     : {'yes' if has_audio else 'NO'}")
        print(f"  CF model  : {'yes' if in_cf else 'NO'}")

        if not has_tags and not has_audio and not in_cf:
            print(
                "\nNo signal data for this track. Cannot recommend.\n"
                "This track exists in tracks.csv but has no CF interactions,\n"
                "no AcousticBrainz audio features, and no Last.fm tags."
            )
            return []

        #STAGE 1:Tag candidates (genre signal)
        tag_candidates: Dict[str, Dict] = {}
        if has_tags:
            print(f"\n[Stage 1] Tag candidates from {len(self.track_tags):,}-track pool...")
            for tid, tags in self.track_tags.items():
                if tid == track_id:
                    continue
                sim = self._tag_similarity(query_tags, tags)
                if sim > MIN_TAG_SCORE:
                    tag_candidates[tid] = {
                        "track_id": tid,
                        "tag_score": sim,
                        "cf_score": 0.0,
                        "audio_score": 0.0,
                        "final_score": 0.0,
                    }
            #sort by tag score to show most genre-similar first
            tag_candidates = dict(
                sorted(tag_candidates.items(), key=lambda x: x[1]["tag_score"], reverse=True)
            )
            print(f"  {len(tag_candidates)} tracks with tag overlap")
            if tag_candidates:
                top = list(tag_candidates.values())[0]
                top_row = self.tracks_df[self.tracks_df["track_id"] == top["track_id"]]
                if not top_row.empty:
                    print(f"  Best tag match: {top_row.iloc[0]['artist']} - "
                          f"{top_row.iloc[0]['title']}  (tag_sim={top['tag_score']:.3f})")

        #STAGE 2: Score CF for every tag candidate
        cf_recs_list: List[Dict] = []
        cf_lookup: Dict[str, float] = {}

        if in_cf:
            print(f"\n[Stage 2] CF scoring...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #request enough to cover all tag candidates +fill slots
                cf_recs_list = self.cf.get_similar_tracks(
                    track_id, n=min(len(self.cf.item_mapping), 5000)
                )
            cf_lookup = {r["track_id"]: float(r["similarity_score"]) for r in cf_recs_list}

            matched = 0
            for tid, c in tag_candidates.items():
                if tid in cf_lookup:
                    c["cf_score"] = cf_lookup[tid]
                    matched += 1
            print(f"  CF scores assigned to {matched}/{len(tag_candidates)} tag candidates")
        else:
            print(f"\n[Stage 2] CF: query not in model, skipping")

        #STAGE 3:Audio scoring for tag candidates that have features
        if has_audio and tag_candidates:
            print(f"\n[Stage 3] Audio scoring of tag candidates...")
            audio_scored = 0
            for tid, c in tag_candidates.items():
                feat = self.features.get(tid)
                if feat:
                    sim = self._audio_similarity(query_features, feat)
                    if sim is not None:
                        c["audio_score"] = sim
                        audio_scored += 1
            print(f"  Audio scores assigned to {audio_scored}/{len(tag_candidates)} candidates")

        # STAGE 4:Blend and sort tag candidates
        for c in tag_candidates.values():
            c["final_score"] = self._blend(
                c,
                cf_confidence=cf_confidence,
                has_tags=True,
                has_audio=(c.get("audio_score", 0) > 0),
                has_cf=(c.get("cf_score", 0) > 0),
            )

        tagged_sorted = sorted(
            tag_candidates.values(), key=lambda x: x["final_score"], reverse=True
        )

        #STAGE 5:CF-only fill for any remaining slots
        n_tagged = min(len(tagged_sorted), n_final)
        n_cf_fill = n_final - n_tagged
        cf_fill: List[Dict] = []

        if n_cf_fill > 0 and in_cf:
            print(f"\n[Stage 5] CF-only fill: need {n_cf_fill} more result slots...")
            tagged_ids = set(tag_candidates.keys()) | {track_id}
            for r in cf_recs_list:
                if r["track_id"] not in tagged_ids:
                    cf_score = float(r["similarity_score"])
                    cf_fill.append({
                        "track_id": r["track_id"],
                        "tag_score": 0.0,
                        "cf_score": cf_score,
                        "audio_score": 0.0,
                        "final_score": cf_score * 0.45 * cf_confidence,
                    })
                if len(cf_fill) >= n_cf_fill:
                    break

        #filter unavailable tracks before assembling output
        def _drop_unavailable(pool: List[Dict]) -> List[Dict]:
            kept = []
            for c in pool:
                rows = self.tracks_df[self.tracks_df["track_id"] == c["track_id"]]
                if rows.empty:
                    continue
                title = str(rows.iloc[0].get("title", ""))
                if self._is_likely_unavailable(title):
                    print(f"  Skipping unavailable: {rows.iloc[0]['artist']} - {title}")
                    continue
                kept.append(c)
            return kept

        tagged_sorted = _drop_unavailable(tagged_sorted)
        cf_fill       = _drop_unavailable(cf_fill)

        result = (tagged_sorted + cf_fill)[:n_final]
        tagged_count  = sum(1 for r in result if r.get("tag_score", 0) > 0)
        cf_only_count = len(result) - tagged_count
        print(f"\nOutput mix: {tagged_count} tagged  |  {cf_only_count} CF-only fill")
        return result

    #Blend -Tags 50%, CF 30%, Audio 20% 

    def _blend(self, c: Dict, cf_confidence: float,
               has_tags: bool, has_audio: bool, has_cf: bool) -> float:
        tag = c.get("tag_score", 0.0)
        cf = c.get("cf_score", 0.0)
        audio = c.get("audio_score", 0.0)

        w_tag = 0.50 if has_tags else 0.0
        w_cf = 0.30 * cf_confidence if has_cf else 0.0
        w_audio = 0.20 if has_audio else 0.0

        # Redistribute unused CF weight to tags (70%) and audio (30%)
        unused_cf = (0.30 * cf_confidence) if has_cf else 0.0
        effective_unused = 0.30 - w_cf
        if effective_unused > 0:
            if has_tags and has_audio:
                w_tag += effective_unused * 0.70
                w_audio += effective_unused * 0.30
            elif has_tags:
                w_tag += effective_unused
            elif has_audio:
                w_audio += effective_unused

        return w_tag * tag + w_cf * cf + w_audio * audio

    #similarity helpers

    def _tag_similarity(self, tags1: Set[str], tags2: Set[str]) -> float:
        if not tags1 or not tags2:
            return 0.0
        intersection = len(tags1 & tags2)
        union = len(tags1 | tags2)
        return intersection / union if union > 0 else 0.0

    def _audio_similarity(self, f1: Dict, f2: Dict) -> Optional[float]:
        """Similarity across mood, danceability, BPM, and energy (loudness). Returns [0, 1] or None."""
        score = 0.0
        weight = 0.0

        #mood cosine (35%)
        moods = ["happy", "party", "aggressive", "relaxed"]
        v1 = [f1.get(f"mood_{m}", 0) for m in moods]
        v2 = [f2.get(f"mood_{m}", 0) for m in moods]
        if sum(v1) > 0 and sum(v2) > 0:
            cos = cosine_similarity(
                np.array(v1).reshape(1, -1),
                np.array(v2).reshape(1, -1)
            )[0][0]
            score += cos * 0.35
            weight += 0.35

        #danceability (20%)
        d1 = f1.get("danceability", 0)
        d2 = f2.get("danceability", 0)
        if d1 > 0 and d2 > 0:
            score += (1 - abs(d1 - d2)) * 0.20
            weight += 0.20

        #BPM -penalise if > 40 apart (25%)
        bpm1 = f1.get("bpm", 0)
        bpm2 = f2.get("bpm", 0)
        if bpm1 > 0 and bpm2 > 0:
            diff = abs(bpm1 - bpm2)
            if diff <= 40:
                score += max(0.0, 1 - diff / 40) * 0.25
                weight += 0.25

        #energy proxy: loudness similarity (20%) ;typical range: -20 to 0 dB
        l1 = f1.get("loudness", None)
        l2 = f2.get("loudness", None)
        if l1 is not None and l2 is not None and l1 != 0 and l2 != 0:
            diff = abs(l1 - l2)
            if diff <= 12:
                score += max(0.0, 1 - diff / 12) * 0.20
                weight += 0.20

        return score / weight if weight > 0 else None

    @staticmethod
    def _is_likely_unavailable(title: str) -> bool:
        """Filter out MSD tracks that exist in the dataset but are not on
        streaming platforms (unreleased recordings, demos, bootlegs, etc.)"""
        t = title.lower()
        markers = (
            "unreleased", "demo", "bootleg", "live at ",
            "rehearsal", "rough mix", "work in progress",
            "wip", "unfinished", "alternate take",
        )
        return any(m in t for m in markers)

    # Display

    def display_results(self, results: List[Dict], query_track_id: str = None):
        print("\n" + "=" * 70)
        print("FINAL RECOMMENDATIONS")
        print("=" * 70)

        query_tags = self.track_tags.get(query_track_id, set()) if query_track_id else set()
        separator_shown = False

        for i, rec in enumerate(results, 1):
            tid = rec["track_id"]

            if rec.get("tag_score", 0) == 0 and not separator_shown:
                print(f"\n  --- CF-only fill (no tag data available for these) ---")
                separator_shown = True

            rows = self.tracks_df[self.tracks_df["track_id"] == tid]
            if rows.empty:
                continue
            tr = rows.iloc[0]
            listeners = self.track_listener_count.get(tid, 0)

            print(f"\n{i:2}. {tr['artist']} - {tr['title']}")
            print(
                f"    tag={rec.get('tag_score', 0):.3f} | "
                f"cf={rec.get('cf_score', 0):.3f} | "
                f"audio={rec.get('audio_score', 0):.3f} | "
                f"final={rec.get('final_score', 0):.3f} | "
                f"listeners={listeners}"
            )

            own_tags = self.track_tags.get(tid, set())
            feat = self.features.get(tid, {})

            if own_tags and query_tags:
                shared = own_tags & query_tags
                if shared:
                    print(f"    Shared tags : {', '.join(sorted(shared)[:8])}")
                else:
                    print(f"    Tags        : {', '.join(sorted(own_tags)[:6])}")
            elif own_tags:
                print(f"    Tags        : {', '.join(sorted(own_tags)[:6])}")

            if feat:
                genre = feat.get("genre", "")
                bpm = feat.get("bpm", 0)
                loudness = feat.get("loudness", 0)
                mood_candidates = {
                    "happy": feat.get("mood_happy", 0),
                    "aggressive": feat.get("mood_aggressive", 0),
                    "party": feat.get("mood_party", 0),
                    "relaxed": feat.get("mood_relaxed", 0),
                }
                mood_top = max(mood_candidates, key=mood_candidates.get)
                parts = []
                if genre:
                    parts.append(f"genre={genre}")
                if bpm:
                    parts.append(f"bpm={bpm:.0f}")
                if loudness:
                    parts.append(f"loudness={loudness:.1f}dB")
                parts.append(f"mood={mood_top}({mood_candidates[mood_top]:.2f})")
                print(f"    Audio       : {' | '.join(parts)}")


# Interactive loop

def main():
    recommender = CascadingHybridRecommender()
    recommender.load_data()

    print("\n" + "=" * 70)
    print("INTERACTIVE LOOP  (type 'quit' to exit)")
    print("=" * 70)
    print("Enter one or more track IDs, one per line.")
    print("Use find_test_tracks.py to discover IDs with coverage flags [ATC].")
    print("Type 'quit' to exit.")
    print()

    while True:
        raw = input("Track ID >>> ").strip()

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        row = recommender.tracks_df[recommender.tracks_df["track_id"] == raw]
        if row.empty:
            print(f"  '{raw}' not found in tracks.csv.")
            print("  Use find_test_tracks.py with 'find <artist>' to search.")
            continue

        results = recommender.recommend(raw, n_final=10)
        if results:
            recommender.display_results(results, query_track_id=raw)
        else:
            print("  No recommendations (no signal data for this track).")
        print()


if __name__ == "__main__":
    main()