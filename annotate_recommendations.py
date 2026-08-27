import argparse
import csv
import json
import re
import time
import urllib.parse
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


#defaults overridable via CLI
DEFAULT_LOGS_DIR     = Path("logs")
DEFAULT_ANNOTATIONS  = Path("rec_annotations.json")
DEFAULT_BLACKLIST    = Path("ai_artists_blacklist.csv")

#keyword patterns that strongly suggest an AI artist name
AI_NAME_PATTERNS = [
    re.compile(r"\bai\b",          re.IGNORECASE),
    re.compile(r"artificial",      re.IGNORECASE),
    re.compile(r"generated",       re.IGNORECASE),
    re.compile(r"syntheti[ck]",    re.IGNORECASE),
    re.compile(r"neural",          re.IGNORECASE),
    re.compile(r"deepfake",        re.IGNORECASE),
    re.compile(r"machine.?made",   re.IGNORECASE),
]


#FILE I/O

def load_logs(logs_dir: Path) -> List[Dict]:
    logs = []
    for fp in sorted(logs_dir.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                logs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return logs


def load_annotations(path: Path) -> Dict[str, Dict]:
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_annotations(data: Dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"  Saved {len(data)} annotations -> {path}")


def load_blacklist(path: Path) -> Set[str]:
    """Read artist names from the blacklist CSV. Accepts 'artist', 'name', or 'artist_name' columns."""
    if not path.exists():
        return set()
    artists: Set[str] = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ("artist", "name", "artist_name"):
                    val = (row.get(col) or "").strip().lower()
                    if val:
                        artists.add(val)
                        break
    except (OSError, csv.Error):
        pass
    return artists


#EXTRACTION

def norm_key(artist: str, title: str) -> str:
    return f"{artist.lower().strip()} - {title.lower().strip()}"


def extract_unique_recs(logs: List[Dict]) -> Dict[str, Dict]:
    unique: Dict[str, Dict] = {}
    for l in logs:
        job_id = l.get("job_id", "")
        for rec in l.get("recommendations", []):
            artist = (rec.get("artist") or "").strip()
            title  = (rec.get("title")  or "").strip()
            if not artist and not title:
                continue
            key = norm_key(artist, title)
            if key not in unique:
                unique[key] = {
                    "artist":   artist,
                    "title":    title,
                    "count":    0,
                    "job_ids":  [],
                    "source":   rec.get("source"),
                }
            unique[key]["count"] += 1
            if job_id not in unique[key]["job_ids"]:
                unique[key]["job_ids"].append(job_id)
    return unique


#PASS 1:BLACKLIST

def pass_blacklist(
    unique_recs: Dict[str, Dict],
    annotations: Dict,
    blacklist: Set[str],
) -> int:
    labeled = 0
    for key, rec in unique_recs.items():
        if key in annotations:
            continue
        artist_norm = rec["artist"].lower().strip()
        if artist_norm in blacklist:
            annotations[key] = {
                "status": "ai",
                "method": "blacklist",
                "artist": rec["artist"],
                "title":  rec["title"],
            }
            labeled += 1
    return labeled


#PASS 2: AI KEYWORD HEURISTIC

def pass_keyword_heuristic(
    unique_recs: Dict[str, Dict],
    annotations: Dict,
    flagged_for_review: Dict,
) -> int:
    """Flag artists whose name matches an AI-related keyword.Not auto-labelled."""
    flagged = 0
    for key, rec in unique_recs.items():
        if key in annotations or key in flagged_for_review:
            continue
        artist = rec["artist"]
        if any(pat.search(artist) for pat in AI_NAME_PATTERNS):
            flagged_for_review[key] = {**rec, "flag_reason": "artist name contains AI keyword"}
            flagged += 1
    return flagged



# PASS 3:INTERACTIVE MANUAL REVIEW

def _build_lookup_urls(artist: str, title: str) -> Dict[str, str]:
    """Build Spotify / Last.fm / MusicBrainz lookup URLs for manual verification."""
    artist_enc = urllib.parse.quote(artist)
    query_enc  = urllib.parse.quote(f"{artist} {title}")

    return {
        #spotify search lands directly on the track/artist results
        "Spotify":      f"https://open.spotify.com/search/{query_enc}",
        #last.fm artist page (if the artist is real, this page exists)
        "Last.fm":      f"https://www.last.fm/music/{artist_enc}",
        #musicBrainz disambiguates AI vs human artists
        "MusicBrainz":  f"https://musicbrainz.org/search?query={artist_enc}&type=artist",
    }


def pass_interactive(
    unique_recs: Dict[str, Dict],
    annotations: Dict,
    flagged_for_review: Dict,
    open_browser: bool = True,
) -> int:
    """Interactive prompt for unknown / flagged entries; opens a Spotify search tab if open_browser."""
    #flagged entries (AI keyword match) come first
    to_review: List[Tuple[str, Dict]] = []
    for key in flagged_for_review:
        if key not in annotations:
            to_review.append((key, {**unique_recs.get(key, {}),
                                    "flag_reason": flagged_for_review[key].get("flag_reason")}))
    for key, rec in unique_recs.items():
        if key not in annotations and key not in flagged_for_review:
            to_review.append((key, rec))

    if not to_review:
        print("  Nothing left to review manually.")
        return 0

    total   = len(to_review)
    labeled = 0

    print(f"\n  Manual review: {total} entries.")
    if open_browser:
        print("  A Spotify search tab will open for each entry automatically.")
        print("  Use --no-browser to disable this.")
    else:
        print("  Browser auto-open is disabled. Lookup URLs will be printed.")
    print()
    print("  Commands:  h = human   a = AI   s = skip   q = quit and save")
    print("  FLAGGED entries matched an AI keyword in the artist name.")
    print("-" * 60)

    for i, (key, rec) in enumerate(to_review, 1):
        artist      = rec.get("artist", "?")
        title       = rec.get("title",  "?")
        count       = rec.get("count",  1)
        flag_reason = rec.get("flag_reason", "")

        urls = _build_lookup_urls(artist, title)

        print(f"\n  [{i}/{total}]")
        if flag_reason:
            print(f"  Artist : {artist}  << FLAGGED: {flag_reason}")
        else:
            print(f"  Artist : {artist}")
        print(f"  Title  : {title}")
        print(f"  Seen   : {count} time(s) across logs")
        print()

        #print URLs as fallback in case the browser doesn't open
        for site, url in urls.items():
            print(f"  {site:<14} {url}")

        if open_browser:
            try:
                webbrowser.open(urls["Spotify"])#auto-open spotify; others are for follow-up
            except Exception:
                pass  #non-fatal: some headless environments have no browser

        print()

        while True:
            try:
                choice = input("  Label  [h/a/s/q] > ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n  Interrupted. Progress saved.")
                return labeled

            if choice == "q":
                print("  Quit. Progress saved.")
                return labeled
            if choice == "s":
                break
            if choice in ("h", "a"):
                annotations[key] = {
                    "status": "human" if choice == "h" else "ai",
                    "method": "manual",
                    "artist": artist,
                    "title":  title,
                }
                labeled += 1
                break
            print("  Invalid input. Use h, a, s, or q.")

    return labeled

# STATS DISPLAY
def print_stats(unique_recs: Dict, annotations: Dict):
    n = len(unique_recs)
    human   = sum(1 for v in annotations.values() if v.get("status") == "human")
    ai      = sum(1 for v in annotations.values() if v.get("status") == "ai")
    unknown = sum(1 for k in unique_recs if k not in annotations)

    print("\n  Annotation status:")
    print(f"    Total unique (artist, title) pairs : {n}")
    print(f"    Confirmed human                    : {human}")
    print(f"    Confirmed AI                       : {ai}")
    print(f"    Unknown / unannotated              : {unknown}")
    if human + ai > 0:
        purity = human / (human + ai) * 100
        print(f"    Purity (human / annotated)         : {purity:.1f}%")

    #method breakdown
    method_counts: Dict[str, int] = defaultdict(int)
    for v in annotations.values():
        method_counts[v.get("method", "unspecified")] += 1

    if method_counts:
        print("\n  By annotation method:")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {method:<25} {count}")
    print()


#CSV EXPORT
def export_csv(unique_recs: Dict, annotations: Dict, out_path: Path):
    rows = []
    for key, rec in unique_recs.items():
        ann = annotations.get(key, {})
        rows.append({
            "artist":  rec["artist"],
            "title":   rec["title"],
            "count":   rec["count"],
            "status":  ann.get("status", "unknown"),
            "method":  ann.get("method", ""),
            "prob":    ann.get("prob", ""),
            "source":  rec.get("source", ""),
        })
    rows.sort(key=lambda r: (r["status"], r["artist"]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["artist", "title", "count", "status", "method", "prob", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Exported {len(rows)} rows -> {out_path}")


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Build recommendation ground-truth annotations for the thesis."
    )
    parser.add_argument("--logs-dir",    default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--blacklist",   default=str(DEFAULT_BLACKLIST))
    parser.add_argument("--auto-only",   action="store_true",
                        help="Run passes 1 and 2 only; skip manual review")
    parser.add_argument("--no-browser",  action="store_true",
                        help="Disable automatic browser tab opening during "
                             "manual review (for headless / SSH environments)")
    parser.add_argument("--show-stats",  action="store_true",
                        help="Print current annotation status and exit")
    parser.add_argument("--export",      default=None, metavar="FILE",
                        help="Export annotations to a CSV file and exit")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    ann_path = Path(args.annotations)
    bl_path  = Path(args.blacklist)

    print(f"Loading logs from {logs_dir}...")
    logs = load_logs(logs_dir)
    print(f"  {len(logs)} logs loaded")
    annotations = load_annotations(ann_path)
    print(f"  {len(annotations)} existing annotations")

    blacklist = load_blacklist(bl_path)
    print(f"  {len(blacklist)} artists in blacklist")

    print("\nExtracting unique recommendations...")
    unique_recs = extract_unique_recs(logs)
    print(f"  {len(unique_recs)} unique (artist, title) pairs found")

    if args.show_stats:
        print_stats(unique_recs, annotations)
        return

    if args.export:
        export_csv(unique_recs, annotations, Path(args.export))
        return

    #Pass 1: Blacklist
    print("\n[Pass 1] Blacklist cross-check...")
    n1 = pass_blacklist(unique_recs, annotations, blacklist)
    if n1:
        print(f"  Auto-labeled {n1} as AI (blacklist match)")
        save_annotations(annotations, ann_path)
    else:
        print("  No new matches in blacklist")

    #Pass 2: keyword heuristic
    print("\n[Pass 2] AI keyword heuristic...")
    flagged_for_review: Dict[str, Dict] = {}
    n2 = pass_keyword_heuristic(unique_recs, annotations, flagged_for_review)
    if n2:
        print(f"  Flagged {n2} entries for priority manual review")
    else:
        print("  No new keyword matches")

    still_unknown = sum(1 for k in unique_recs if k not in annotations)
    print(f"\n  After passes 1-2: {still_unknown} entries still unknown")

    if still_unknown == 0:
        print("  All recommendations annotated.")
        print_stats(unique_recs, annotations)
        return

    #Pass 3: Interactive
    if args.auto_only:
        print(f"\n--auto-only set. Skipping manual review.")
        print_stats(unique_recs, annotations)
        return

    still_unknown = sum(1 for k in unique_recs if k not in annotations)
    if still_unknown == 0 and not flagged_for_review:
        print("\nNothing left to review manually.")
        print_stats(unique_recs, annotations)
        return

    print(f"\n[Pass 3] Interactive manual review...")
    n4 = pass_interactive(unique_recs, annotations, flagged_for_review,
                          open_browser=not args.no_browser)
    if n4:
        save_annotations(annotations, ann_path)
        print(f"  Manual review labeled {n4} entries")

    print_stats(unique_recs, annotations)


if __name__ == "__main__":
    main()