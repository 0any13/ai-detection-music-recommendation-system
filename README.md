# Music Recommendation System — Human-Made Music Recommender

A music recommendation system that takes a song link (Spotify or YouTube), estimates
how likely the track is AI-generated, and - for human-made input - returns similar
tracks that are themselves filtered to be human-made. Similarity is computed by a
three-stage cascade: Last.fm folksonomy tags, collaborative filtering (ALS / matrix
factorization) trained on the Million Song Dataset, and AcousticBrainz audio-feature
re-ranking. An artist blacklist and a per-track AI-audio check (LetsSubmit) keep the
recommendations clean.

This README walks you through building the system from raw data to a running web app.
Nothing in `data/` or `models/` ships with the repository - every artifact is produced
by running the scripts below in order. Budget time: the external-API steps
(MusicBrainz, AcousticBrainz, Last.fm) take hours to days on a first run because they
are politely rate-limited.

---

## 1. Requirements

- Python 3.10 (via conda)
- A few free API keys (see Section 3). Only a Last.fm key is needed to *build* the data;
  the others are needed to *run the app*.
- Disk space: the raw Taste Profile download is ~2.8 GB uncompressed.

---

## 2. Environment setup

The verified setup (Windows + conda):

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n musicrec python=3.10 -y
conda activate musicrec
pip install -r requirements.txt
```

The first two `conda tos accept` lines accept the Anaconda channel terms of service; skip
them if your conda is already configured.

If the `implicit` library fails to build from pip on your machine, install it from
conda-forge instead, then re-run the pip install:

```
conda install -c conda-forge implicit -y
pip install -r requirements.txt
```


---

## 3. API keys (.env)

Create a file named `.env` in the project root. Use your own keys — none are committed.

```
SPOTIFY_CLIENT_ID=your_value
SPOTIFY_CLIENT_SECRET=your_value
LASTFM_API_KEY=your_value
YOUTUBE_API_KEY=your_value
LETSSUBMIT_API_KEY=your_value
SIGHTENGINE_API_USER=your_value
SIGHTENGINE_API_SECRET=your_value
```

Where to get each key:

- Spotify (`SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`): create an app in the Spotify
  Developer Dashboard at https://developer.spotify.com/dashboard — the Client ID and
  Client Secret are on the app's settings page. Note: some Spotify endpoints used by the
  Discovered-On path were restricted to apps created before late 2024; a newer app still
  works for track lookup and search, it just may return less from the recommendation
  endpoints.
- Last.fm (`LASTFM_API_KEY`): create an API account at
  https://www.last.fm/api/account/create - the key is issued instantly.
- YouTube (`YOUTUBE_API_KEY`): in the Google Cloud Console
  (https://console.cloud.google.com/), create a project, enable "YouTube Data API v3",
  then create an API key under Credentials.
- LetsSubmit (`LETSSUBMIT_API_KEY`): https://letssubmit.com/ai-music-checker/api - this is
  the audio AI-detection service and it is a paid API, so expect usage to cost money.
- Sightengine (`SIGHTENGINE_API_USER`, `SIGHTENGINE_API_SECRET`): sign up at
  https://sightengine.com - the dashboard gives you an API user and an API secret (two
  separate strings). The free tier is small (about 400 AI-image checks per month), which is
  why the cover-art check runs only on the submitted song.

Which keys are needed when:

- Building the dataset (Section 5): only `LASTFM_API_KEY`. MusicBrainz and AcousticBrainz
  need no key.
- Running the app (Section 6): all of the above. Spotify and YouTube power the input
  resolution and the Discovered-On fallbacks; LetsSubmit is the audio AI-detection check;
  Sightengine is the (informational) cover-art check.

---

## 4. Data setup — what to download and where to put it

All raw data goes in the `data/` folder. Create it if it does not exist.

### 4.1 Echo Nest Taste Profile triplets (collaborative-filtering training data)

- Download `train_triplets.txt.zip` from: http://millionsongdataset.com/tasteprofile/
- Unzip it and place the file at: `data/train_triplets.txt`
- Format (tab-separated): `user_id <TAB> song_id <TAB> play_count`
- It looks like this:

```
b80344d063b5ccb3212f76538f3d9e43d87dca9e	SOAKIMP12A8C130995	1
b80344d063b5ccb3212f76538f3d9e43d87dca9e	SOBBMDR12A8C13253B	2
```

  Roughly 48 million rows, ~1 million users, ~380k songs. Note the IDs here are
  **song IDs** (`SO...`).

### 4.2 MSD unique-tracks index (the ID bridge + track metadata)

- Download from: http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt
- Place it at: `data/unique_tracks.txt`
- Format (`<SEP>`-separated): `track_id <SEP> song_id <SEP> artist <SEP> title`
- It looks like this:

```
TRMMMYQ128F932D901<SEP>SOQMMHC12AB0180CB8<SEP>Faster Pussy cat<SEP>Silent Night
TRMMMKD128F425225D<SEP>SOVFVAK12A8C1350D9<SEP>Karkkiautomaatti<SEP>Tanssi vaan
```

  This file carries both the MSD track ID (`TR...`) and the song ID (`SO...`) on each
  line. The system keys everything on the **song ID** so that track metadata joins the
  Taste Profile triplets. This `unique_tracks.txt` is what supplies `tracks.csv`; the
  Last.fm per-track tag archive (`lastfm_train`/`lastfm_test`) is **not** used — tags are
  fetched live from the Last.fm API in step 5.6.

After this section, `data/` should contain exactly two files you placed by hand:
`train_triplets.txt` and `unique_tracks.txt`. Everything else in `data/` is generated by
the pipeline.

---

## 5. The pipeline — run these in order

Each step reads what the previous steps produced and writes new files into `data/` (or
`models/`). Run them from the project root with the `musicrec` env active.

### 5.1 Parse the listening triplets

```
python parse_user_interactions.py
```
Reads `data/train_triplets.txt`, writes `data/user_interactions.csv`
(columns: `user_id, track_id, play_count`, where `track_id` is the `SO...` song ID).
Prompts you to choose full vs. a test subset; choose full (option 1) for real results.
Optionally offers to also write `data/user_interactions_sample.csv`.

### 5.2 Build the track metadata table

```
python parse_msd_track_list.py
```
Reads `data/unique_tracks.txt`, writes `data/tracks.csv`
(columns: `track_id, artist, title`, keyed on the `SO...` song ID). If
`user_interactions.csv` already exists it also prints an ID-coverage report so you can
confirm the two files line up.

### 5.3 Add MusicBrainz IDs

```
python get_mbids.py
```
Reads `data/tracks.csv`, looks each track up on MusicBrainz, and adds an `mbid` column
in place; also writes `data/mbid_cache.json` so re-runs resume instantly. Interactive
menu picks how many tracks to process (1=100, 2=1,000, 3=10,000, 4=all). Rate-limited to
~1 request/second, so the full run is long — it is safe to interrupt and resume.

### 5.4 Fetch AcousticBrainz audio features

```
python prefetch_acousticbrainz.py
```
Reads the MBIDs in `data/tracks.csv` and pulls low- and high-level audio features from
AcousticBrainz into `data/ab_features_cache.json`. Uses `improved_acousticbrainz.py`
(the API wrapper — not run directly, but `python improved_acousticbrainz.py` runs a
single-track self-test if you want to verify connectivity). Also rate-limited; resumable.

### 5.5 Assemble the feature table

```
python rebuild_tracks_with_features.py
```
Merges `data/ab_features_cache.json` onto `data/tracks.csv` by `mbid` and writes
`data/tracks_with_features.json` — the flat, song-ID-keyed feature list the recommender
reads at inference time.

### 5.6 Fetch Last.fm tags

```
python fetch_lastfm_tags.py
```
Reads `data/tracks_with_features.json`, queries the Last.fm API (`artist.getTopTags`)
per artist, and writes `data/lastfm_tags_cache.json` (song ID -> list of tags). Requires
`LASTFM_API_KEY` in `.env`.

### 5.7 Train the collaborative-filtering model

```
python train_collaborative_filtering.py
```
Reads `data/user_interactions.csv` and `data/tracks.csv`, trains the ALS model, and
writes a timestamped trio plus a pointer file into `models/`:
`als_model_<timestamp>.pkl`, `mappings_<timestamp>.pkl`, `matrices_<timestamp>.pkl`, and
`latest_model_info.txt`. Answer `n` to the sample prompt to train on the full data.

### 5.8 (Optional) Verify your build matched the reference dataset

```
python verify_dataset.py
```
Compares your `train_triplets.txt` / `user_interactions.csv` against the official MSD
counts and reports coverage. Use this to confirm your download was complete before
trusting the trained model.

---

## 6. Run the app

```
python app.py
```
Starts the Flask server at http://127.0.0.1:5000. Paste a Spotify or YouTube link (or a
free-text song search) and the system resolves the track, runs the AI check, and returns
human-made recommendations. The hybrid recommender is loaded lazily on the first request,
so the first recommendation takes a few extra seconds. Job logs are written to `logs/`.

---

## 7. File reference

### Web app
- `app.py` — Flask server, input resolution, AI check, job orchestration, and the UI
  endpoints. Run with `python app.py`.
- `templates/index.html`, `static/css/style.css`, `static/js/app.js`,
  `static/img/vinyl_player.png` — the single-page front end.

### Data pipeline (Section 5)
- `parse_user_interactions.py` — triplets -> `user_interactions.csv`.
- `parse_msd_track_list.py` — `unique_tracks.txt` -> `tracks.csv`.
- `get_mbids.py` — adds `mbid` to `tracks.csv`, writes `mbid_cache.json`.
- `prefetch_acousticbrainz.py` — populates `ab_features_cache.json`.
- `improved_acousticbrainz.py` — AcousticBrainz API wrapper (library; self-test on run).
- `rebuild_tracks_with_features.py` — builds `tracks_with_features.json`.
- `fetch_lastfm_tags.py` — builds `lastfm_tags_cache.json`.
- `train_collaborative_filtering.py` — trains ALS, writes `models/`.

### Recommender (runtime, imported by the app)
- `cascading_hybrid_recommender.py` — the three-stage cascade (tags / CF / audio).
  Runnable standalone as a smoke test.
- `collaborative_filtering_recommender.py` — loads the trained ALS model and serves
  item-to-item similarity. Running it directly launches an interactive test against the
  latest model.
- `discovered_on_recommender.py` — Spotify "Discovered On" playlist miner (fallback path).
- `youtube_discovered_on.py` — YouTube-based recommender / top-up (fallback path).
- `blacklist.py` — `AIArtistBlacklist`, backed by `ai_artists_blacklist.csv`.
- `letssubmit_cache.py` — cached client for the LetsSubmit audio AI-detection API.
- `sightengine_cache.py` — cached client for the Sightengine cover-art AI check.

### Evaluation (reproduces the results chapter)
These read the per-job logs the app writes to `logs/`. To reproduce the exact thesis
numbers you need the original `logs/`, `ground_truth.json`, and `rec_annotations.json`;
run against fresh `logs/` to evaluate a new set of runs.

- `build_ground_truth.py` — auto-labels each submitted track from the logs into
  `ground_truth.json` (detection ground truth).
  ```
  python build_ground_truth.py
  python build_ground_truth.py --logs-dir logs --out ground_truth.json
  python build_ground_truth.py --threshold 50   # stricter auto-label cutoff
  python build_ground_truth.py --dry-run        # print only, do not write file
  ```
  Important: this only *auto-suggests* labels from the AI probability. After running it you
  must open `ground_truth.json` and, for each entry, set `"is_ai"` to `true` or `false`
  yourself — based on your own check of whether the submitted song is actually AI-generated.
  The auto-label is a starting point; the real ground truth is the value you write in by
  hand for each entry.
- `annotate_recommendations.py` — labels the *recommended* tracks human/AI into
  `rec_annotations.json` (recommendation purity).
  ```
  python annotate_recommendations.py
  python annotate_recommendations.py --auto-only       # blacklist + keyword passes only
  python annotate_recommendations.py --no-browser      # headless manual review
  python annotate_recommendations.py --show-stats      # print current status and exit
  python annotate_recommendations.py --export recs.csv # export annotations to CSV
  ```
  This runs live and interactive: for each recommended track and artist it has not already
  auto-labeled, it prompts you to mark it human or AI. You make that call yourself by
  researching the artist (the script opens/prints Spotify, Last.fm, and MusicBrainz lookup
  links to help), then type your verdict so it is saved into `rec_annotations.json`.
- `analyze_logs.py` — generates the experiment report (confusion matrix, metrics,
  genre/platform breakdowns) from the logs and labels.
  ```
  python analyze_logs.py
  python analyze_logs.py --logs-dir logs --gt ground_truth.json
  python analyze_logs.py --threshold 50          # detection cutoff for metrics
  python analyze_logs.py --out results/report.txt # write report to a file
  ```

### Diagnostics / tests
- `verify_dataset.py` — checks dataset completeness vs. official MSD counts.
- `test_recommender.py` — exercises the recommender.
- `test_letssubmit.py` — exercises the LetsSubmit client.

### Support files (in the repo)
- `requirements.txt` — dependencies.
- `.env` — your API keys (you create it; not committed).
- `apis/__init__.py`, `apis/spotify_client.py` — the Spotify client package.

### Generated, not in the repo
Everything under `data/` (raw downloads, parsed CSVs, and all `*_cache.json` files) and
everything under `models/` is produced by Sections 4 and 5. The cache JSONs are written
automatically as you run the pipeline, so a fresh clone rebuilds them — at the cost of the
external-API time noted above.

---

## 8. Optional shortcut — prebuilt artifacts

If you would rather not spend days rebuilding the dataset and retraining, the author's
prebuilt files can be downloaded and dropped straight into the project to skip the slow
steps. This is optional; the from-scratch build in Sections 4–6 is fully self-sufficient.

Download the prebuilt model files here: [Google Drive folder](https://drive.google.com/drive/folders/1MoNjGOvu5g_iV7vb9DFdiiWl6B4VlAop?usp=sharing)

The download contains:
- the trained model trio plus its pointer file (`als_model_*.pkl`, `mappings_*.pkl`,
  `matrices_*.pkl`, `latest_model_info.txt`) — place them in `models/`.
- `ai_artists_blacklist.csv` — the author's populated blacklist; place it in the project
  root (it overrides the seed list if present).

With the model files in `models/`, you can skip Sections 5.1–5.7 and the long API steps,
and go straight to running the app (Section 6) — provided the feature/cache files the app
reads are also present (either from this download, if included, or by running the relevant
pipeline steps).

---

## 9. Troubleshooting

**Model fails to load / "No trained model found".**
The loader reads `models/latest_model_info.txt`. If you moved the project files, Windows may have saved that pointer file without its `.txt` extension (it
shows as type "Text Document" with no `.txt` in the name). Rename it:
```
Rename-Item "models\latest_model_info" "models\latest_model_info.txt"
```
After that the trained model loads normally.

**`implicit` won't install from pip.**
Install it from conda-forge, then re-run pip (see Section 2).



