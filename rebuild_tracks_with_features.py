import json
import pandas as pd
from pathlib import Path


def rebuild(data_path: str = "data"):
    data_path = Path(data_path)

    #load cache 
    cache_file = data_path / "ab_features_cache.json"
    print(f"Loading {cache_file}...")
    with open(cache_file, "r", encoding="utf-8") as f:
        cache = json.load(f)

    non_null = {k: v for k, v in cache.items() if v is not None}
    print(f"  Total entries : {len(cache):,}")
    print(f"  Non-null      : {len(non_null):,}")

    #load tracks.csv to get mbid -> track_id mapping 
    tracks_file = data_path / "tracks.csv"
    print(f"\nLoading {tracks_file}...")
    tracks_df = pd.read_csv(tracks_file, low_memory=False)
    tracks_with_mbid = tracks_df[tracks_df["mbid"].notna()].copy()
    tracks_with_mbid["mbid"] = tracks_with_mbid["mbid"].astype(str).str.strip()
    print(f"  Tracks with MBID: {len(tracks_with_mbid):,}")

    #build mbid -> row lookup
    mbid_to_track = {}
    for _, row in tracks_with_mbid.iterrows():
        mbid_to_track[row["mbid"]] = row

    #merge
    print("\nMerging cache with track metadata...")
    output = []
    matched = 0
    unmatched_mbids = 0

    for mbid, feat in non_null.items():
        track_row = mbid_to_track.get(mbid)
        if track_row is None:
            unmatched_mbids += 1
            continue

        entry = {
            "track_id": track_row["track_id"],
            "mbid": mbid,
            "artist": track_row.get("artist", ""),
            "title": track_row.get("title", ""),
            #core genre fields
            "genre": feat.get("genre", ""),
            "genre_electronic": feat.get("genre_electronic", ""),
            "genre_dortmund": feat.get("genre_dortmund", ""),
            "acoustic_electronic": feat.get("acoustic_electronic", ""),
            #audio features
            "bpm": feat.get("bpm", 0),
            "key": feat.get("key", ""),
            "scale": feat.get("scale", ""),
            "loudness": feat.get("loudness", 0),
            "danceability": feat.get("danceability_probability",
                                     1.0 if feat.get("danceability") == "danceable" else 0.0),
            #mood probabilities
            "mood_happy": feat.get("mood_happy_probability", 0),
            "mood_aggressive": feat.get("mood_aggressive_probability", 0),
            "mood_party": feat.get("mood_party_probability", 0),
            "mood_relaxed": feat.get("mood_relaxed_probability", 0),
            #voice
            "voice_instrumental": feat.get("voice_instrumental", ""),
            "timbre": feat.get("timbre", ""),
            #MFCCs 
            "mfcc_1": feat.get("mfcc_1", 0),
            "mfcc_2": feat.get("mfcc_2", 0),
            "mfcc_3": feat.get("mfcc_3", 0),
            "mfcc_4": feat.get("mfcc_4", 0),
            "mfcc_5": feat.get("mfcc_5", 0),
            #spectral energy bands extracted from AcousticBrainz raw cache.
            "energy_high": (
                feat["spectral_energyband_high"]["mean"]
                if isinstance(feat.get("spectral_energyband_high"), dict)
                else feat.get("spectral_energyband_high", 0)
            ),
            "energy_low": (
                feat["spectral_energyband_low"]["mean"]
                if isinstance(feat.get("spectral_energyband_low"), dict)
                else feat.get("spectral_energyband_low", 0)
            ),
            "spectral_centroid": (
                feat["spectral_centroid"]["mean"]
                if isinstance(feat.get("spectral_centroid"), dict)
                else feat.get("spectral_centroid", 0)
            ),
            "zero_crossing_rate": (
                feat["zerocrossingrate"]["mean"]
                if isinstance(feat.get("zerocrossingrate"), dict)
                else feat.get("zero_crossing_rate", 0)
            ),
        }
        output.append(entry)
        matched += 1

    print(f"  Matched to track_id : {matched:,}")
    print(f"  MBID not in tracks  : {unmatched_mbids:,}")

    #write output
    out_file = data_path / "tracks_with_features.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(output):,} entries to {out_file}")

    #summary of what we have
    genres = {}
    for entry in output:
        g = entry.get("genre", "unknown")
        genres[g] = genres.get(g, 0) + 1

    print("\nGenre breakdown:")
    for genre, count in sorted(genres.items(), key=lambda x: -x[1]):
        print(f"  {genre or 'unknown':8s}: {count:,}")

    print("\nDone. tracks_with_features.json is ready for the recommender.")
    return output


if __name__ == "__main__":
    rebuild()