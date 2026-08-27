import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_OUT      = Path("ground_truth.json")
DEFAULT_THRESHOLD = 35.0


#Helpers

def load_logs(logs_dir: Path) -> List[Dict]:
    logs = []
    for fp in sorted(logs_dir.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
                data["_filename"] = fp.name
                logs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Warning: skipping {fp.name}: {exc}")
    return logs


def extract_cover_art_probability(log: Dict) -> Optional[float]:
    for entry in log.get("progress", []):
        msg = entry.get("msg", "")
        if "Cover art check:" in msg and "%" in msg:
            try:
                return float(msg.split("Cover art check:")[1].strip().split("%")[0])
            except (IndexError, ValueError):
                pass
    return None


def artist_from_log(log: Dict) -> str:
    track = log.get("track") or {}
    return track.get("artist", "").strip() or "Unknown Artist"

def title_from_log(log: Dict) -> str:
    track = log.get("track") or {}
    return track.get("title", "").strip() or "Unknown Title"

def platform_from_log(log: Dict) -> str:
    track = log.get("track") or {}
    return track.get("platform", "").strip()


# Main builder

def build(logs: List[Dict], threshold: float,
          existing: Dict = None) -> Tuple[Dict, List[Dict]]:
    """Returns:
        ground_truth  -- merged dict: existing entries untouched, new ones added
        review_list   -- only NEW entries sorted by AI probability ascending
    """
    existing = existing or {}
    ground_truth: Dict[str, Dict] = dict(existing)   #start with what we alr have
    review_list: List[Dict] = []

    for log in logs:
        job_id = log.get("job_id")
        if not job_id:
            continue

        #skip job ids already in the file- never overwrite manual edits
        if job_id in existing:
            continue

        artist   = artist_from_log(log)
        title    = title_from_log(log)
        platform = platform_from_log(log)
        ai_block = log.get("ai") or {}
        prob     = ai_block.get("probability")
        verdict  = ai_block.get("verdict", "")
        cover    = extract_cover_art_probability(log)
        state    = log.get("state", "")

        #auto-label
        if prob is None:
            is_ai = None
            auto_label = "null (no probability - set manually)"
        elif prob >= threshold:
            is_ai = True
            auto_label = f"true  (prob {prob:.1f}% >= {threshold}%)"
        else:
            is_ai = False
            auto_label = f"false (prob {prob:.1f}% <  {threshold}%) -- REVIEW"

        entry = {
            "is_ai":    is_ai,
            "genre":    "",
            "artist":   artist,
            "title":    title,
            "platform": platform,
            "_prob":    round(prob, 2) if prob is not None else None,
            "_verdict": verdict,
            "_cover_art_prob": round(cover, 1) if cover is not None else None,
            "_auto_label": auto_label,
            "_state":   state,
        }

        ground_truth[job_id] = entry

        review_list.append({
            "job_id":   job_id,
            "artist":   artist,
            "title":    title,
            "prob":     prob,
            "is_ai":    is_ai,
            "filename": log.get("_filename", ""),
        })

    review_list.sort(key=lambda r: (r["prob"] is None, r["prob"] or 0))
    return ground_truth, review_list


def print_review_report(review_list: List[Dict], threshold: float, top_n: int = 30):
    """Print the logs sorted by AI probability ascending.human controls will be at the top."""
    print()
    print("=" * 72)
    print("  REVIEW CANDIDATES - sorted by AI probability, lowest first")
    print(f"  Entries below {threshold}% were auto-labeled is_ai=false.")
    print("  Verify these are actually human, then check the rest look correct.")
    print("=" * 72)

    human_count = sum(1 for r in review_list if r["is_ai"] is False)
    ai_count    = sum(1 for r in review_list if r["is_ai"] is True)
    null_count  = sum(1 for r in review_list if r["is_ai"] is None)

    print(f"\n  Auto-labeled is_ai=true  : {ai_count}")
    print(f"  Auto-labeled is_ai=false : {human_count}  <- verify these")
    print(f"  Auto-labeled is_ai=null  : {null_count}  <- need manual decision")
    print()

    shown = review_list[:top_n]
    col_w = max((len(r["artist"]) for r in shown), default=20)
    col_w = max(col_w, 20)

    header = (f"  {'Prob':>6}  {'Auto-label':<9}  "
              f"{'Artist':<{col_w}}  Title")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in shown:
        prob_str = f"{r['prob']:.1f}%" if r["prob"] is not None else "  N/A "
        label    = "is_ai=false" if not r["is_ai"] else "is_ai=true "
        if r["is_ai"] is None:
            label = "is_ai=null "
        artist   = r["artist"][:col_w]
        title    = r["title"][:35]
        flag     = "  <-- REVIEW" if r["is_ai"] is False else ""
        print(f"  {prob_str:>6}  {label}  {artist:<{col_w}}  {title}{flag}")

    if len(review_list) > top_n:
        remaining = len(review_list) - top_n
        print(f"\n  ... {remaining} more entries (all auto-labeled is_ai=true, "
              f"prob >= {threshold}%)")

    print()
    print("  Next steps:")
    print("  1. Open ground_truth.json in your editor.")
    print('  2. Find any entries where is_ai should be flipped .')
    print('     Change is_ai to false and set genre to the correct value.')
    print('  3. Fill in "genre" for all entries.')
    print('     Tip: use your editor find-and-replace on artist name to batch-assign.')
    print()


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Add new log entries to ground_truth.json without touching existing ones."
    )
    parser.add_argument("--logs-dir",  default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--out",       default=str(DEFAULT_OUT),
                        help="Path to ground_truth.json (read + updated in place)")
    parser.add_argument("--threshold", default=DEFAULT_THRESHOLD, type=float,
                        help="Probability cutoff for auto-labeling new entries "
                             f"(default: {DEFAULT_THRESHOLD}%%)")
    parser.add_argument("--top",       default=30, type=int,
                        help="How many new entries to show in the review report (default: 30)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print review report only; do not write ground_truth.json")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_path = Path(args.out)

    print(f"Loading logs from {logs_dir}...")
    logs = load_logs(logs_dir)
    print(f"  {len(logs)} logs loaded")

    if not logs:
        print("  No logs found. Check --logs-dir.")
        return

    #load existing ground_truth.json so we never overwrite manual edits
    existing: Dict = {}
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            print(f"  {len(existing)} existing entries loaded from {out_path}")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Warning: could not read {out_path}: {exc}")

    new_count = sum(1 for l in logs
                    if l.get("job_id") and l["job_id"] not in existing)
    print(f"  {new_count} new log(s) not yet in ground_truth.json")

    if new_count == 0:
        print("  Nothing to add. All logs are already in ground_truth.json.")
    else:
        print(f"Building entries for new logs (threshold = {args.threshold}%)...")
        ground_truth, review_list = build(logs, args.threshold, existing=existing)

        print_review_report(review_list, args.threshold, top_n=args.top)

        if args.dry_run:
            print("  --dry-run set. ground_truth.json was NOT written.")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ground_truth, f, indent=2, sort_keys=False)
            print(f"  {new_count} new entries added -> {out_path}  "
                  f"(total: {len(ground_truth)})")

    #always print unique progress regardless of whether new logs were added
    with open(out_path, encoding="utf-8") as f:
        full_gt = json.load(f)

    seen_ai:      set = set()
    seen_human:   set = set()
    seen_unknown: set = set()

    for entry in full_gt.values():
        if isinstance(entry, dict) and "artist" in entry:
            key = (entry.get("artist", "").strip().lower(),
                   entry.get("title",  "").strip().lower())
            is_ai = entry.get("is_ai")
            if is_ai is True:    seen_ai.add(key)
            elif is_ai is False: seen_human.add(key)
            else:                seen_unknown.add(key)

    print()
    print("=" * 72)
    print("  UNIQUE SONG PROGRESS  (duplicates removed)")
    print("=" * 72)
    print(f"    Unique AI songs     : {len(seen_ai):<4}  "
          f"(need {max(0, 100 - len(seen_ai))} more to reach 100)")
    print(f"    Unique human songs  : {len(seen_human):<4}  "
          f"(need {max(0, 100 - len(seen_human))} more to reach 100)")
    if seen_unknown:
        print(f"    Unlabeled           : {len(seen_unknown):<4}  "
              f"(set is_ai in ground_truth.json)")
    print()


if __name__ == "__main__":
    main()