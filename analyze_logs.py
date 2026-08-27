
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re 


DEFAULT_AI_THRESHOLD    = 35.0   


# HELPERS

def load_logs(logs_dir: Path) -> List[Dict]:
    logs = []
    for fp in sorted(logs_dir.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
                data["_filename"] = fp.name
                logs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f" Could not load {fp.name}: {exc}")
    return logs

def load_json_safe(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  Could not load {path}: {exc}")
        return {}

def safe_mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None

def safe_stdev(values: List[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) >= 2 else None

def fmt_opt(value: Optional[float], suffix: str = "", decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"

def norm_key(artist: str, title: str) -> str:
    return f"{artist.lower().strip()} - {title.lower().strip()}"

#deduplication to unique songs 

def _song_key(log: Dict, gt: Dict) -> str:
    # song identity from cleaned artist + title 
    info   = gt.get(log.get("job_id"), {}) or {}
    track  = log.get("track") or {}
    artist = (info.get("artist") or track.get("artist") or "").lower().strip()
    title  = (track.get("title") or info.get("title") or "").lower().strip()
    return f"{artist} ::: {title}"

def _rep_rank(log: Dict) -> Tuple:
    """Rank runs for the same song"""
    ai        = log.get("ai") or {}
    has_prob  = ai.get("probability") is not None
    done      = log.get("state") == "done"
    rec_count = log.get("recommendation_count",
                        len(log.get("recommendations", [])))
    ts        = log.get("timestamp_utc", "")
    return (has_prob, done, rec_count, ts)#( prob >full exec > rec count > recency)

def dedup_to_unique_songs(logs: List[Dict], gt: Dict) -> List[Dict]:
    labelled = [l for l in logs if l.get("job_id") in gt]
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for l in labelled:
        groups[_song_key(l, gt)].append(l)
    return [max(group, key=_rep_rank) for group in groups.values()]





#Confusion matrix and metrics

def confusion_matrix(
    true_labels: List[bool], pred_labels: List[bool]
) -> Tuple[int, int, int, int]:
    """Returns (TP, TN, FP, FN)."""
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t and p)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if not t and not p)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if not t and p)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t and not p)
    return tp, tn, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def roc_auc(true_labels: List[bool], probabilities: List[float]) -> float:
    """Trapezoidal AUC-ROC.NaN if only one class is present."""
    if not true_labels or len(set(true_labels)) < 2:
        return float("nan")

    n_pos = sum(true_labels)
    n_neg = len(true_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    paired = sorted(zip(probabilities, true_labels), reverse=True)
    tp, fp = 0, 0
    prev_tp, prev_fp = 0, 0
    auc = 0.0
    prev_prob: Optional[float] = None

    for prob, label in paired:
        if prev_prob is not None and prob != prev_prob:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp, prev_tp = fp, tp
        if label:
            tp += 1
        else:
            fp += 1
        prev_prob = prob

    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)



# Progress message helpers

def progress_msgs(log: Dict) -> List[str]:
    return [e.get("msg", "") for e in log.get("progress", [])]

def used_hybrid(log: Dict) -> bool:
    return any("Hybrid produced" in m for m in progress_msgs(log))

def used_discovered_on(log: Dict) -> bool:
    return any("Spotify Discovered-On produced" in m for m in progress_msgs(log))

def used_youtube(log: Dict) -> bool:
    return any("YouTube added" in m for m in progress_msgs(log))


YT_ADDED_RE = re.compile(r"YouTube added (\d+) rec")

def reconstruct_rec_sources(log: Dict) -> List[str]:
    """Per-rec source label, in stored order."""
    recs = log.get("recommendations", [])
    msgs = " ".join(e.get("msg", "") for e in log.get("progress", []))
    m = YT_ADDED_RE.search(msgs)
    n_yt = int(m.group(1)) if m else 0

    def has_blend(r):
        return any(r.get(k) is not None for k in ("final_score", "tag_score", "cf_score"))

    non_hybrid_idx = [i for i, r in enumerate(recs) if not has_blend(r)]
    yt_positions = set(non_hybrid_idx[len(non_hybrid_idx) - n_yt:]) if 0 < n_yt <= len(non_hybrid_idx) else set()

    out = []
    for i, r in enumerate(recs):
        if has_blend(r):
            out.append("hybrid")
        elif i in yt_positions:
            out.append("youtube")
        else:
            out.append("spotify_discovered")
    return out


def found_in_msd(log: Dict) -> bool:
    msgs = progress_msgs(log)
    return any(("Found in MSD" in m or "MSD match" in m or "Hybrid produced" in m)
               and "Not found in MSD" not in m for m in msgs)


# REPORT BUILDER

class Report:
    def __init__(self):
        self._lines: List[str] = []
    def h1(self, text: str):
        self._lines += ["", "=" * 72, f"  {text}", "=" * 72]
    def h2(self, text: str):
        self._lines += ["", f"  {text}", "  " + "-" * (len(text) + 2)]
    def row(self, label: str, value, width: int = 42):
        self._lines.append(f"    {label:<{width}} {value}")
    def blank(self):
        self._lines.append("")
    def note(self, text: str):
        self._lines.append(f"    [!] {text}")
    def text(self, text: str):
        self._lines.append(f"    {text}")
    def divider(self, char: str = "-", width: int = 68):
        self._lines.append("  " + char * width)
    def histogram(self, values: List[float], bins: int = 10,
                  lo: float = 0.0, hi: float = 100.0):
        """Render an ASCII histogram for probability values."""
        if not values:
            self._lines.append("    (no data)")
            return
        step = (hi - lo) / bins
        counts = [0] * bins
        for v in values:
            idx = min(int((v - lo) / step), bins - 1)
            counts[idx] += 1
        max_count = max(counts) or 1
        bar_width = 28
        for i, count in enumerate(counts):
            left  = lo + i * step
            right = lo + (i + 1) * step
            bar   = "#" * int(count / max_count * bar_width)
            self._lines.append(
                f"    {left:>5.1f}-{right:<6.1f} | {bar:<{bar_width}} {count}"
            )

    def render(self) -> str:
        return "\n".join(self._lines)
    def print_report(self):
        print(self.render())
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
        print(f"\nReport saved -> {path}")


# SECTION 1: DATASET OVERVIEW

def section_overview(logs: List[Dict], gt: Dict, report: Report):
    report.h1("1. DATASET OVERVIEW")

    labeled    = [l for l in logs if l["job_id"] in gt]
    ai_logs    = [l for l in labeled if gt[l["job_id"]]["is_ai"] is True]
    human_logs = [l for l in labeled if gt[l["job_id"]]["is_ai"] is False]
    null_logs  = [l for l in labeled if gt[l["job_id"]]["is_ai"] is None]
    unlabeled  = [l for l in logs if l["job_id"] not in gt]

    unique_ai    = {(gt[l["job_id"]].get("artist","").lower(), l.get("track",{}).get("title","").lower())
                for l in ai_logs}
    unique_human = {(gt[l["job_id"]].get("artist","").lower(), l.get("track",{}).get("title","").lower())
                    for l in human_logs}

    report.row("  -> unique AI songs (dedup)",    len(unique_ai))
    report.row("  -> unique human songs (dedup)", len(unique_human))

    report.row("Total job logs", len(logs))
    report.row("Labeled (ground truth present)", len(labeled))
    report.row(" -> AI-generated songs", len(ai_logs))
    report.row("  -> Human-made songs (controls)", len(human_logs))
    if null_logs:
        report.row(" -> is_ai=null (excluded, fix in ground_truth.json)", len(null_logs))
    report.row("Unlabeled (no ground truth entry)", len(unlabeled))

    if null_logs:
        report.blank()
        report.note(f"{len(null_logs)} entries have is_ai=null. These are likely")
        report.note("quota-expired runs where probability was never returned.")
        report.note("Set is_ai manually in ground_truth.json to include them.")

    if unlabeled:
        report.blank()
        report.note(
            f"{len(unlabeled)} logs are unlabeled. Add entries to ground_truth.json"
        )
        report.note("to include them in classification metrics.")

    #genre distribution
    genre_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ai": 0, "human": 0})
    for l in labeled:
        info = gt[l["job_id"]]
        if info["is_ai"] is None:
            continue   # skip unlabeled nulls
        genre = info.get("genre", "unknown").lower()
        key   = "ai" if info["is_ai"] else "human"
        genre_counts[genre][key] += 1

    if genre_counts:
        report.h2("Genre Distribution")
        report.row("Genre", "AI    Human", width=24)
        report.divider()
        for genre in sorted(genre_counts):
            counts = genre_counts[genre]
            report.row(genre, f"{counts['ai']:<6}  {counts['human']}", width=24)

    #platforms submitted
    platform_counts: Dict[str, int] = defaultdict(int)
    for l in logs:
        plat = (l.get("track") or {}).get("platform", "unknown")
        platform_counts[plat] += 1

    if platform_counts:
        report.h2("Submission Platform")
        for plat, count in sorted(platform_counts.items(), key=lambda x: -x[1]):
            report.row(plat, count)


# SECTION 2: AI DETECTION EVALUATION

def section_ai_detection(logs: List[Dict], gt: Dict,
                          threshold: float, report: Report):
    report.h1("2. AI DETECTION SYSTEM EVALUATION")
    report.blank()
    report.text(
        "The detection system assigns a probability p that a submitted track is AI-generated."
    )
    report.text(
        f"The binary threshold is p >= {threshold}% = predicted AI, "
        f"p < {threshold}% = predicted human."
    )

    labeled = [
        l for l in logs
        if l["job_id"] in gt
        and l.get("ai")
        and l["ai"].get("probability") is not None
    ]

    if not labeled:
        report.blank()
        report.note("No labeled logs with probability scores. Populate ground_truth.json.")
        return

    true_ai  = [gt[l["job_id"]]["is_ai"] for l in labeled]
    probs    = [l["ai"]["probability"] for l in labeled]
    pred_ai  = [p >= threshold for p in probs]

    tp, tn, fp, fn = confusion_matrix(true_ai, pred_ai)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    accuracy  = (tp + tn) / len(labeled)
    auc       = roc_auc(true_ai, probs)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else None
    tnr       = tn / (tn + fp) if (tn + fp) >0 else None

    report.h2(f"Confusion Matrix  (threshold = {threshold}%)")
    report.row("True  Positives  -- AI correctly flagged as AI", tp)
    report.row("True  Negatives  -- Human correctly cleared",    tn)
    report.row("False Positives  -- Human incorrectly flagged",  fp)
    report.row("False Negatives  -- AI missed by the system",    fn)

    report.h2("Classification Metrics")
    report.row("n (labeled with probability score)", len(labeled))
    report.row("Accuracy",                f"{accuracy:.4f}  ({accuracy*100:.1f}%)")
    report.row("Precision",               f"{precision:.4f}")
    report.row("Recall (Sensitivity / TPR)", f"{recall:.4f}")
    report.row("Specificity (TNR)",       fmt_opt(tnr, decimals=4))
    report.row("False Positive Rate",     fmt_opt(fpr, decimals=4))
    report.row("F1 Score",                f"{f1:.4f}")
    report.row("AUC-ROC (trapezoidal)",
               f"{auc:.4f}" if not math.isnan(auc) else
               "N/A -- need at least one example from each class")

    #probability distributions split by true label
    ai_probs    = [l["ai"]["probability"] for l, t in zip(labeled, true_ai) if t]
    human_probs = [l["ai"]["probability"] for l, t in zip(labeled, true_ai) if not t]

    report.h2("Probability Score Distribution")
    if ai_probs:
        report.text("AI songs (ground truth = AI generated):")
        report.row("  n",        len(ai_probs), width=18)
        report.row("  mean",     fmt_opt(safe_mean(ai_probs),  suffix="%", decimals=2), width=18)
        report.row("  stdev",    fmt_opt(safe_stdev(ai_probs), suffix="%", decimals=2), width=18)
        report.row("  min/max",  f"{min(ai_probs):.2f}% / {max(ai_probs):.2f}%",       width=18)

    if human_probs:
        report.blank()
        report.text("Human songs (ground truth = human made):")
        report.row("  n",        len(human_probs), width=18)
        report.row("  mean",     fmt_opt(safe_mean(human_probs),  suffix="%", decimals=2), width=18)
        report.row("  stdev",    fmt_opt(safe_stdev(human_probs), suffix="%", decimals=2), width=18)
        report.row("  min/max",  f"{min(human_probs):.2f}% / {max(human_probs):.2f}%",   width=18)

    report.h2("AI Probability Histogram (all labeled logs, 0-100%)")
    report.histogram(probs)

    #per-genre breakdown
    genre_data: Dict[str, Dict] = defaultdict(
        lambda: {"true": [], "pred": [], "probs": []}
    )
    for l, is_ai, pred, prob in zip(labeled, true_ai, pred_ai, probs):
        genre = gt[l["job_id"]].get("genre", "unknown").lower()
        genre_data[genre]["true"].append(is_ai)
        genre_data[genre]["pred"].append(pred)
        genre_data[genre]["probs"].append(prob)

    if len(genre_data) > 1:
        report.h2("Per-Genre Detection Metrics")
        report.row("Genre", "n    Acc    Prec   Rec    F1     AvgProb", width=16)
        report.divider()
        for genre in sorted(genre_data):
            d = genre_data[genre]
            n = len(d["true"])
            if n == 0:
                continue
            tp_g, tn_g, fp_g, fn_g = confusion_matrix(d["true"], d["pred"])
            acc_g = (tp_g + tn_g) / n
            prec_g, rec_g, f1_g = precision_recall_f1(tp_g, fp_g, fn_g)
            avg_p  = safe_mean(d["probs"])
            report.row(
                genre,
                (f"{n:<5} {acc_g:.3f}  {prec_g:.3f}  {rec_g:.3f}  "
                 f"{f1_g:.3f}  {avg_p:.1f}%"),
                width=16
            )


    #individual log detail table for thesis appendix
    report.h2("Per-Log Detail (labeled logs)")
    report.row("Artist", "Genre     True   Pred   Prob(%)", width=30)
    report.divider()
    for l, is_ai, pred, prob in zip(labeled, true_ai, pred_ai, probs):
        info   = gt[l["job_id"]]
        artist = info.get("artist", l.get("track", {}).get("artist", "?"))[:28]
        genre  = info.get("genre", "?")[:9]
        t_str  = "AI   " if is_ai  else "Human"
        p_str  = "AI   " if pred   else "Human"
        report.row(artist, f"{genre:<10} {t_str:<7} {p_str:<7} {prob:.1f}", width=30)


# SECTION 3: PIPELINE PERFORMANCE

def section_pipeline(logs: List[Dict], report: Report):
    report.h1("3. RECOMMENDATION PIPELINE PERFORMANCE")

    completed = [l for l in logs if l.get("state") == "done"]
    errored   = [l for l in logs if l.get("state") == "error"]
    in_flight = [l for l in logs if l.get("state") not in ("done", "error")]

    report.row("Total logs",       len(logs))
    report.row("Completed (done)", len(completed))
    report.row("Errored",          len(errored))
    report.row("Incomplete / other", len(in_flight))

    durations = [l["duration_s"] for l in completed if l.get("duration_s") is not None]
    if durations:
        report.h2("Job Duration (seconds)")
        report.row("n",      len(durations))
        report.row("Mean",   fmt_opt(safe_mean(durations),   suffix="s", decimals=1))
        report.row("Median", f"{statistics.median(durations):.1f}s")
        report.row("Stdev",  fmt_opt(safe_stdev(durations),  suffix="s", decimals=1))
        report.row("Min",    f"{min(durations):.1f}s")
        report.row("Max",    f"{max(durations):.1f}s")
        report.h2("Duration Histogram")
        report.histogram(durations, lo=min(durations), hi=max(durations) + 1)

    #source breakdown
    hybrid_n    = sum(1 for l in completed if used_hybrid(l))
    disc_n      = sum(1 for l in completed if used_discovered_on(l))
    yt_n        = sum(1 for l in completed if used_youtube(l))
    msd_n       = sum(1 for l in completed if found_in_msd(l))
    n           = len(completed) or 1

    report.h2("Recommendation Source Breakdown")
    report.row("Jobs using Hybrid recommender (MSD + CF)",
               f"{hybrid_n:<4} ({hybrid_n/n*100:.1f}%)")
    report.row("Jobs using Spotify Discovered-On fallback",
               f"{disc_n:<4} ({disc_n/n*100:.1f}%)")
    report.row("Jobs using YouTube fallback",
               f"{yt_n:<4} ({yt_n/n*100:.1f}%)")
    report.row("Jobs where input track found in MSD",
               f"{msd_n:<4} ({msd_n/n*100:.1f}%)")

    #recommendation counts
    rec_counts = [l.get("recommendation_count", 0) for l in completed]
    if rec_counts:
        report.h2("Recommendations per Job")
        report.row("Mean",                 fmt_opt(safe_mean(rec_counts), decimals=1))
        report.row("Median",               f"{statistics.median(rec_counts):.1f}")
        report.row("Jobs achieving target (>=10 recs)",
                   sum(1 for c in rec_counts if c >= 10))
        report.row("Jobs below target (<10 recs)",
                   sum(1 for c in rec_counts if 0 < c < 10))
        report.row("Jobs with zero recommendations",
                   sum(1 for c in rec_counts if c == 0))

    # AI detection step timing from progress log
    detection_durations: List[float] = []
    for l in completed:
        prog = l.get("progress", [])
        t_start = t_end = None
        for entry in prog:
            msg = entry.get("msg", "")
            if "calling LetsSubmit" in msg:
                t_start = entry.get("t")
            elif "AI check on" in msg and "%" in msg and t_start is not None:
                t_end = entry.get("t")
                break
        if t_start is not None and t_end is not None:
            detection_durations.append(t_end - t_start)

    if detection_durations:
        report.h2("AI Detection Latency (LetsSubmit call, seconds)")
        report.row("n",      len(detection_durations))
        report.row("Mean",   fmt_opt(safe_mean(detection_durations),   suffix="s", decimals=1))
        report.row("Median", f"{statistics.median(detection_durations):.1f}s")
        report.row("Min",    f"{min(detection_durations):.1f}s")
        report.row("Max",    f"{max(detection_durations):.1f}s")

    # time for (dispatch -> "Hybrid produced N recs")
    hybrid_durations = []
    for l in completed:
        t_dispatch = t_done = None
        for e in l.get("progress", []):
            msg = e.get("msg", "")
            if "running hybrid" in msg and t_dispatch is None:
                t_dispatch = e.get("t")
            elif "Hybrid produced" in msg and t_dispatch is not None:
                t_done = e.get("t"); break
        if t_dispatch is not None and t_done is not None:
            hybrid_durations.append(t_done - t_dispatch)
    if hybrid_durations:
        report.h2("Hybrid Recommender Execution Time (seconds)")
        report.row("n",      len(hybrid_durations))
        report.row("Mean",   fmt_opt(safe_mean(hybrid_durations), suffix="s", decimals=1))
        report.row("Median", f"{statistics.median(hybrid_durations):.1f}s")
        report.row("Min",    f"{min(hybrid_durations):.1f}s")
        report.row("Max",    f"{max(hybrid_durations):.1f}s")

    #quota exhaustion
    quota_hit = [l for l in logs if l.get("quota_exhausted")]
    report.h2("API Quota")
    report.row("Jobs where LetsSubmit quota was exhausted",
               f"{len(quota_hit)} / {len(logs)}")

    #cache effectiveness
    cache_hits  = sum(1 for l in completed
                      if l.get("ai", {}).get("status") == "cached")
    cache_miss  = sum(1 for l in completed
                      if l.get("ai", {}).get("status") in ("fresh", None))
    report.h2("LetsSubmit Cache Effectiveness")
    report.row("Cache hits",   cache_hits)
    report.row("Cache misses", cache_miss)
    if cache_hits + cache_miss > 0:
        rate = cache_hits / (cache_hits + cache_miss)
        report.row("Hit rate", f"{rate:.2%}")


# SECTION 4: RECOMMENDATION PURITY

def section_purity(logs: List[Dict], annotations: Dict, report: Report):
    report.h1("4. RECOMMENDATION PURITY")

    report.text(
        "Purity measures what fraction of the system's recommendations are"
    )
    report.text(
        "confirmed human-made music (not AI-generated)."
    )

    if not annotations:
        report.blank()
        report.note("rec_annotations.json not found or empty.")
        report.note("Run annotate_recommendations.py to build annotation labels.")

    #collect all recommendations from all logs
    all_recs: List[Dict] = []
    for l in logs:
        job_id = l.get("job_id", "")
        for rec in l.get("recommendations", []):
            artist = (rec.get("artist") or "").strip()
            title  = (rec.get("title")  or "").strip()
            if not artist and not title:
                continue
            key    = norm_key(artist, title)
            status = annotations.get(key, {}).get("status", "unknown")
            method = annotations.get(key, {}).get("method",  "")
            all_recs.append({
                "job_id": job_id,
                "artist": artist,
                "title":  title,
                "key":    key,
                "status": status,
                "method": method,
                "tag_score":   rec.get("tag_score"),
                "cf_score":    rec.get("cf_score"),
                "final_score": rec.get("final_score"),
                "source":      rec.get("source"),
            })

    report.h2("Volume")
    report.row("Total recommendations across all jobs", len(all_recs))
    unique_artists = {r["artist"] for r in all_recs}
    report.row("Unique recommended artists",            len(unique_artists))

    human_recs   = [r for r in all_recs if r["status"] == "human"]
    ai_recs      = [r for r in all_recs if r["status"] == "ai"]
    unknown_recs = [r for r in all_recs if r["status"] == "unknown"]

    #reconstructed purity by source 
    src_tally = {"hybrid": {"human": 0, "ai": 0, "unknown": 0},
                 "spotify_discovered": {"human": 0, "ai": 0, "unknown": 0},
                 "youtube": {"human": 0, "ai": 0, "unknown": 0}}
    for l in logs:
        recs = l.get("recommendations", [])
        for rec, src in zip(recs, reconstruct_rec_sources(l)):
            artist = (rec.get("artist") or "").strip()
            title  = (rec.get("title")  or "").strip()
            if not artist and not title:
                continue
            status = annotations.get(norm_key(artist, title), {}).get("status", "unknown")
            src_tally[src][status] += 1

    report.h2("Purity by Recommendation Source")
    report.row("Source", "n     Human  AI    AI rate", width=24)
    report.divider()
    for src, label in (("hybrid", "Hybrid (MSD+CF)"),
                       ("spotify_discovered", "Spotify Discovered-On"),
                       ("youtube", "YouTube fallback")):
        c = src_tally[src]
        ann = c["human"] + c["ai"]
        n = ann + c["unknown"]
        rate = f"{c['ai']/ann*100:.1f}%" if ann else "n/a"
        report.row(label, f"{n:<5} {c['human']:<6} {c['ai']:<5} {rate}", width=24)

    report.h2("Annotation Coverage")
    n = len(all_recs) or 1
    report.row("Confirmed human",     f"{len(human_recs):<5} ({len(human_recs)/n*100:.1f}%)")
    report.row("Confirmed AI",        f"{len(ai_recs):<5}  ({len(ai_recs)/n*100:.1f}%)")
    report.row("Not yet annotated",   f"{len(unknown_recs):<5} ({len(unknown_recs)/n*100:.1f}%)")

    annotated = human_recs + ai_recs
    if annotated:
        purity = len(human_recs) / len(annotated)
        report.h2("Purity (annotated recommendations only)")
        report.row("Confirmed human",             len(human_recs))
        report.row("Confirmed AI (filter bypass)", len(ai_recs))
        report.row("Purity",
                   f"{purity:.4f}  ({purity*100:.1f}% of annotated are human)")

    #method breakdown (how the annotation was determined)
    if annotated:
        method_counts: Dict[str, int] = defaultdict(int)
        for r in annotated:
            method_counts[r["method"] or "unspecified"] += 1

        report.h2("Annotation Method Breakdown")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            report.row(method, count)

    #AI recommendations that bypassed the filter 
    if ai_recs:
        report.h2("AI Recommendations That Bypassed the Human Filter")
        report.note("These represent false negatives of the recommendation filter.")
        report.blank()
        seen = set()
        for r in ai_recs:
            k = r["key"]
            if k in seen:
                continue
            seen.add(k)
            report.text(f"  {r['artist']:<35} {r['title']}")
        if len(seen) > 40:
            report.text(f"  ... and {len(ai_recs) - 40} more (see rec_annotations.json)")

    #unknown recommendations for follow-up
    if unknown_recs:
        unique_unknown: Dict[str, Dict] = {}
        for r in unknown_recs:
            if r["key"] not in unique_unknown:
                unique_unknown[r["key"]] = r

        report.h2(f"Unannotated Unique Recommendations ({len(unique_unknown)})")
        report.note("Run annotate_recommendations.py to classify these.")
        for i, (key, r) in enumerate(list(unique_unknown.items())[:40], 1):
            report.text(f"  {i:>3}. {r['artist']:<35} {r['title']}")
        if len(unique_unknown) > 40:
            report.text(f"  ... {len(unique_unknown) - 40} more")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis experiment results from job logs."
    )
    parser.add_argument("--logs-dir",    default="logs",
                        help="Directory containing job JSON logs (default: logs/)")
    parser.add_argument("--gt",          default="ground_truth.json",
                        help="Ground truth labels file (default: ground_truth.json)")
    parser.add_argument("--annotations", default="rec_annotations.json",
                        help="Recommendation annotation file "
                             "(default: rec_annotations.json)")
    parser.add_argument("--threshold",   default=DEFAULT_AI_THRESHOLD, type=float,
                        help=f"AI detection threshold in %% "
                             f"(default: {DEFAULT_AI_THRESHOLD})")
    parser.add_argument("--all-runs", action="store_true",
                        help="Analyse every job log instead of collapsing to "
                             "one representative run per unique labelled song.")
    parser.add_argument("--out",         default=None,
                        help="Save report to this path (optional)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    gt_path  = Path(args.gt)
    ann_path = Path(args.annotations)
    out_path = Path(args.out) if args.out else None

    print(f"Loading logs from: {logs_dir}")
    logs = load_logs(logs_dir)
    print(f"  {len(logs)} logs loaded")

    gt = load_json_safe(gt_path)
    if gt:
        print(f"  {len(gt)} ground truth entries from {gt_path}")
    else:
        print(f"  ground_truth.json not found at {gt_path} -- detection metrics skipped")

    if not args.all_runs and gt:
        n_before = len(logs)
        logs = dedup_to_unique_songs(logs, gt)
        print(f"  Collapsed to {len(logs)} unique labelled songs "
              f"(from {n_before} job logs). Use --all-runs for the full set.")
    elif not args.all_runs and not gt:
        print("  No ground truth -- cannot dedup to unique songs; using all logs.")
    #we drop the is_ai=null songs so every section reports on the same 220.
    if gt:
        logs = [l for l in logs if gt.get(l.get("job_id"), {}).get("is_ai") is not None]


    annotations = load_json_safe(ann_path)
    if annotations:
        print(f"  {len(annotations)} recommendation annotations from {ann_path}")
    else:
        print(f"  rec_annotations.json not found -- purity metrics will be incomplete")

    report = Report()
    report.h1("MUSIC RECOMMENDATION SYSTEM -- EXPERIMENT RESULTS")
    report.text("Bachelor's Thesis Analysis Report")
    report.blank()

    section_overview(logs, gt, report)
    section_ai_detection(logs, gt, args.threshold, report)
    section_pipeline(logs, report)
    section_purity(logs, annotations, report)

    report.blank()
    report.divider("=")
    report.blank()

    report.print_report()
    if out_path:
        report.save(out_path)


if __name__ == "__main__":
    main()