

const $ = (sel) => document.querySelector(sel);

const els = {
  form:       $("#check-form"),
  query:      $("#query"),
  btn:        $("#check-btn"),
  hero:       $("#hero"),
  results:    $("#results"),
  status:     $("#meta-status"),

  trackImg:       $("#track-img"),
  trackTitle:     $("#track-title"),
  trackArtist:    $("#track-artist"),
  trackPlatform:  $("#track-platform"),
  trackEmbed:     $("#track-embed"),
  trackArtPlaceholder: $(".track-art-placeholder"),

  gaugeFg:        $("#gauge-fg"),
  gaugePct:       $("#gauge-pct"),
  gaugeUnit:      $("#gauge-unit"),
  gaugeVerdict:   $("#gauge-verdict"),
  gaugeDetail:    $("#gauge-detail"),
  gaugeTicks:     $("#gauge-ticks"),
  coverWrap:      $("#cover-wrap"),       
  coverFg:        $("#cover-fg"),         
  coverPct:       $("#cover-pct"),        
  coverVerdict:   $("#cover-verdict"),    

  logList:        $("#log-list"),
  recsSection:    $("#recs-section"),
  recsList:       $("#recs-list"),
  recsCount:      $("#recs-count"),
  recsEmpty:      $("#recs-empty"),
  recsQuotaError: $("#recs-quota-error"),

  forceRecsPrompt: $("#force-recs-prompt"),
  forceRecsBtn:    $("#force-recs-btn"),

  vinylPlayer:    $("#vinyl-player"),
  vinylStage:     $(".vinyl-stage"),
  vinylSpinner:   $("#vinyl-spinner"),
  vinylArt:       $("#vinyl-art"),
  vinylStatusText: $("#vinyl-status-text"),
  vinylNowTitle:  $("#vinyl-now-title"),
  vinylNowArtist: $("#vinyl-now-artist"),
  vinylStop:      $("#vinyl-stop"),
  vinylExternal:  $("#vinyl-external"),

  audioCorner:      $("#audio-corner"),
  audioCornerLabel: $("#audio-corner-label"),
  audioCornerHost:  $("#audio-corner-host"),
  audioCornerHint:  $("#audio-corner-hint"),
  audioCornerClose: $("#audio-corner-close"),

  picker:        $("#picker"),
  pickerQuery:   $("#picker-query"),
  pickerCards:   $("#picker-cards"),
  pickerCancel:  $("#picker-cancel"),
};

const GAUGE_CIRCUMFERENCE = 628.32; //2*pi*100
const COVER_CIRCUMFERENCE = 314.16; // 2*pi*50
let pollTimer = null;
let currentJobId = null;
let lastJobSnapshot = null;
let renderedLogCount = 0;
let renderedRecsCount = 0;
let forceRecsHandled = false;

/*init*/
(function init() {
  drawGaugeTicks();
  bindForm();
  bindHints();
  bindForceRecs();
  bindVinylPlayer();
  bindPicker();
  setStatus("idle", "idle");
})();

/*ticks around the gauge ring*/
function drawGaugeTicks() {
  const svg = "http://www.w3.org/2000/svg";
  const cx = 120, cy = 120, rOuter = 110, rInner = 116;
  const totalTicks = 60;
  for (let i = 0; i < totalTicks; i++) {
    const angle = (i / totalTicks) * Math.PI * 2 - Math.PI / 2;
    const isMajor = i % 5 === 0;
    const r1 = isMajor ? rOuter - 4 : rOuter;
    const r2 = rInner;
    const line = document.createElementNS(svg, "line");
    line.setAttribute("x1", cx + Math.cos(angle) * r1);
    line.setAttribute("y1", cy + Math.sin(angle) * r1);
    line.setAttribute("x2", cx + Math.cos(angle) * r2);
    line.setAttribute("y2", cy + Math.sin(angle) * r2);
    if (isMajor) line.setAttribute("stroke-opacity", "0.4");
    els.gaugeTicks.appendChild(line);
  }
}

/* form*/
function bindForm() {
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = els.query.value.trim();
    if (!q) return;
    startAnalysis(q);
  });
}

function bindHints() {
  document.querySelectorAll(".hint-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      els.query.value = chip.dataset.fill || chip.textContent;
      els.query.focus();
    });
  });
}

/*start analysis */
async function startAnalysis(query) {
  resetUI();
  hidePicker();
  els.btn.disabled = true;
  setStatus("busy", "analyzing");

  try {
    const resp = await fetch("/api/check", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query}),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();

    if (data.state === "pending_resolution") {
      //backend wants user to disambiguate before running the analysis
      setStatus("idle", "pick a track");
      els.btn.disabled = false;
      showPicker(data.query, data.candidates, data.resolution_token);
      return;
    }

    if (data.state === "queued" && data.job_id) {
      showResults();
      currentJobId = data.job_id;
      startPolling(currentJobId);
      return;
    }

    throw new Error("Unexpected response from /api/check");
  } catch (err) {
    showError(err.message);
    els.btn.disabled = false;
    setStatus("err", "error");
  }
}

/* picker (disambiguation)*/
function showPicker(query, candidates, token) {
  els.pickerQuery.textContent = query || "";
  els.pickerCards.innerHTML = "";
  candidates.forEach((c, i) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "picker-card";
    const platform = c.spotify_url ? "spotify" : (c.youtube_url ? "youtube" : "");
    const sim = (typeof c._similarity === "number")
      ? `sim ${(c._similarity * 100).toFixed(0)}%` : "";
    card.innerHTML = `
      <div class="picker-card-art" ${c.artwork ? `style="background-image:url('${escapeAttr(c.artwork)}')"` : ""}></div>
      <div class="picker-card-body">
        <div class="picker-card-title"></div>
        <div class="picker-card-artist"></div>
        <div class="picker-card-meta">
          ${platform ? `<span class="picker-card-platform ${platform}">${platform}</span>` : ""}
          ${sim ? `<span class="picker-card-sim">${sim}</span>` : ""}
        </div>
      </div>
    `;
    card.querySelector(".picker-card-title").textContent = c.title || "untitled";
    card.querySelector(".picker-card-artist").textContent = c.artist || "unknown";
    card.addEventListener("click", () => pickCandidate(token, i));
    els.pickerCards.appendChild(card);
  });
  els.picker.hidden = false;
  els.picker.scrollIntoView({behavior: "smooth", block: "center"});
}

function hidePicker() {
  els.picker.hidden = true;
  els.pickerCards.innerHTML = "";
  els.pickerQuery.textContent = "";
}

async function pickCandidate(token, index) {
  hidePicker();
  els.btn.disabled = true;
  setStatus("busy", "analyzing");
  showResults();
  try {
    const resp = await fetch("/api/resolve", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({resolution_token: token, candidate_index: index}),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    if (data.state === "queued" && data.job_id) {
      currentJobId = data.job_id;
      startPolling(currentJobId);
      return;
    }
    throw new Error("Unexpected response from /api/resolve");
  } catch (err) {
    showError(err.message);
    els.btn.disabled = false;
    setStatus("err", "error");
  }
}

function bindPicker() {
  els.pickerCancel.addEventListener("click", () => {
    hidePicker();
    setStatus("idle", "idle");
  });
}

function resetUI() {
  els.logList.innerHTML = "";
  els.recsList.innerHTML = "";
  els.recsEmpty.hidden = true;
  els.recsQuotaError.hidden = true;
  els.recsSection.hidden = true;
  els.recsCount.textContent = "0";
  els.forceRecsPrompt.hidden = true;
  delete els.forceRecsPrompt.dataset.quotaExhausted;
  els.forceRecsBtn.disabled = false;
  const btnText = els.forceRecsBtn.querySelector(".btn-text");
  if (btnText) btnText.textContent = "show similar tracks";
  renderedLogCount = 0;
  renderedRecsCount = 0;
  forceRecsHandled = false;
  lastJobSnapshot = null;
  if (els.picker) els.picker.hidden = true;
  resetGauge();
  resetTrack();
}

function resetGauge() {
  lastGaugePct = null;
  els.gaugeFg.classList.remove("unverified");
  els.gaugeFg.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
  els.gaugePct.textContent = "--";
  els.gaugePct.classList.remove("unverified", "scanning");
  els.gaugePct.classList.add("scanning");
  els.gaugeUnit.textContent = "scanning";
  els.gaugeVerdict.textContent = "awaiting signal";
  els.gaugeVerdict.className = "gauge-verdict";
  els.gaugeDetail.innerHTML = "&nbsp;";
  lastGaugePct = null;
  lastCoverPct = null;

  //reset cover sub-gauge
  if (els.coverFg) {
    els.coverFg.classList.remove("unverified", "bad", "warn", "ok");
    els.coverFg.style.strokeDashoffset = COVER_CIRCUMFERENCE;
    els.coverPct.textContent = "--";
    els.coverPct.classList.remove("unverified");
    els.coverVerdict.textContent = "awaiting cover";
    els.coverVerdict.className = "cover-verdict";
  }
}

function resetTrack() {
  lastTrackArt = null; 
  els.trackEmbed.dataset.url = "";
  els.trackImg.hidden = true;
  els.trackImg.src = "";
  els.trackArtPlaceholder.hidden = false;
  els.trackTitle.textContent = "resolving...";
  els.trackArtist.innerHTML = "&nbsp;";
  els.trackPlatform.innerHTML = "&nbsp;";
  els.trackEmbed.innerHTML = "";
}

function showResults() {
  els.results.hidden = false;
  els.hero.classList.add("compact");
}

/* polling loop*/
function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => pollOnce(jobId), 500);
  pollOnce(jobId);
}

async function pollOnce(jobId) {
  try {
    const resp = await fetch(`/api/status/${jobId}`);
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    const job = await resp.json();
    renderJob(job);

    if (job.state === "done" || job.state === "error") {
      clearInterval(pollTimer);
      pollTimer = null;
      els.btn.disabled = false;
      if (job.state === "done") {
        setStatus("ok", "done");
      } else {
        setStatus("err", "error");
        showError(job.error || "Unknown error");
      }
    }
  } catch (err) {
    console.error("Poll error", err);
  }
}

/* render job */
function renderJob(job) {
  lastJobSnapshot = job;
  setStatus(stateColor(job.state), job.state.replace(/_/g, " "));
  renderProgressLog(job.progress);
  if (job.track) renderTrack(job.track);
  if (job.ai)    renderGauge(job.ai);
  if (job.cover_ai) renderCover(job.cover_ai);
  renderRecommendations(job.recommendations || [], job.state, job);
  maybeShowForceRecsPrompt(job);
}

function maybeShowForceRecsPrompt(job) {

  const HUMAN_VERDICT_CEILING = 35;

  const prob = job.ai && job.ai.probability;
  const hasRecs = (job.recommendations || []).length > 0;
  const isHumanVerdict = (typeof prob === "number" && prob < HUMAN_VERDICT_CEILING);
  const isUnverified   = (prob === null || prob === undefined);

  //diagnostic attributes 
  els.forceRecsPrompt.dataset.debugState   = job.state || "";
  els.forceRecsPrompt.dataset.debugProb    = String(prob);
  els.forceRecsPrompt.dataset.debugProbType = typeof prob;
  els.forceRecsPrompt.dataset.debugHasRecs = String(hasRecs);
  els.forceRecsPrompt.dataset.debugQuota   = String(!!job.quota_exhausted);
  els.forceRecsPrompt.dataset.debugHuman   = String(isHumanVerdict);
  els.forceRecsPrompt.dataset.debugUnverif = String(isUnverified);

  if (job.state !== "done") {
    els.forceRecsPrompt.hidden = true;
    els.forceRecsPrompt.dataset.debugReason = "not-done";
    return;
  }
  //without this gate the prompt pops back up on the next poll after the user dismissed it by clicking "yes"
  if (forceRecsHandled || hasRecs) {
    els.forceRecsPrompt.hidden = true;
    els.forceRecsPrompt.dataset.debugReason =
      forceRecsHandled ? "already-handled" : "recs-present";
    return;
  }
  if (job.quota_exhausted) {
    els.forceRecsPrompt.hidden = true;
    els.forceRecsPrompt.dataset.debugReason = "quota-exhausted";
    return;
  }
  if (isHumanVerdict || isUnverified) {
    els.forceRecsPrompt.hidden = false;
    els.forceRecsPrompt.dataset.debugReason = isHumanVerdict ? "human-verdict" : "unverified";
    delete els.forceRecsPrompt.dataset.quotaExhausted;
  } else {

    els.forceRecsPrompt.hidden = true;
    els.forceRecsPrompt.dataset.debugReason = "ai-positive";
  }
}

function stateColor(state) {
  if (state === "done") return "ok";
  if (state === "error") return "err";
  return "busy";
}

/* track render */
let lastTrackArt = null;
function renderTrack(track) {
  els.trackTitle.textContent = track.title || "unknown";
  els.trackArtist.textContent = track.artist || "unknown";
  els.trackPlatform.textContent = `[${(track.platform || "").toUpperCase()}]` +
                                   (track.album ? ` :: ${track.album}` : "");

  if (track.artwork && track.artwork !== lastTrackArt) {
    lastTrackArt = track.artwork;
    els.trackImg.src = track.artwork;
    els.trackImg.hidden = false;
    els.trackArtPlaceholder.hidden = true;
  }

  if (track.embed_url && !els.trackEmbed.dataset.url) {
    els.trackEmbed.dataset.url = track.embed_url;
    const iframe = document.createElement("iframe");
    iframe.src = track.embed_url;
    iframe.allow = "encrypted-media; autoplay";
    iframe.height = (track.embed_url || "").includes("spotify") ? 80 : 160;
    iframe.loading = "lazy";
    els.trackEmbed.appendChild(iframe);
  }
}

/* gauge render*/
let lastGaugePct = null;
function renderGauge(ai) {
  const pct = ai.probability;
  if (pct === null || pct === undefined) {
    if (lastGaugePct !== "unverified") {
      els.gaugeFg.classList.add("unverified");
      els.gaugeFg.style.strokeDashoffset = GAUGE_CIRCUMFERENCE / 2;
      els.gaugePct.classList.remove("scanning");
      els.gaugePct.classList.add("unverified");
      els.gaugePct.textContent = "?";
      els.gaugeUnit.textContent = "unverified";
      els.gaugeVerdict.textContent = "signal unavailable";
      els.gaugeVerdict.className = "gauge-verdict unverified";
      els.gaugeDetail.textContent = detailFromStatus(ai.status);
      lastGaugePct = "unverified";
    }
    return;
  }

  if (lastGaugePct === pct) return;
  lastGaugePct = pct;

  //animate the percentage number from current to target
  const startPct = parseFloat(els.gaugePct.textContent) || 0;
  animateNumber(startPct, pct, 1200, (v) => {
    els.gaugePct.textContent = Math.round(v);
  });

  //animate the ring fill
  els.gaugeFg.classList.remove("unverified");
  els.gaugePct.classList.remove("scanning", "unverified");
  const offset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
  els.gaugeFg.style.strokeDashoffset = offset;

  els.gaugeUnit.textContent = "% ai-generated";
  els.gaugeVerdict.textContent = ai.verdict || "";
  els.gaugeVerdict.className = "gauge-verdict " + verdictClass(pct);
  const source = ai.checked_source ? ` (${ai.checked_source.toLowerCase()})` : "";
  els.gaugeDetail.textContent = (ai.cached ? "from cache" : "live check") + source;
}

let lastCoverPct = null;
function renderCover(cover) {
  if (!els.coverFg) return;
  const pct = cover.probability;

  //no usable result: dashed half-ring, "?" centre.
  if (pct === null || pct === undefined) {
    if (lastCoverPct !== "unverified") {
      els.coverFg.classList.add("unverified");
      els.coverFg.classList.remove("bad", "warn", "ok");
      els.coverFg.style.strokeDashoffset = COVER_CIRCUMFERENCE / 2;
      els.coverPct.textContent = cover.status === "no_url" ? "--" : "?";
      els.coverPct.classList.add("unverified");
      els.coverVerdict.textContent =
        cover.status === "no_url"            ? "no cover art"
        : cover.status === "no_credentials"  ? "detector not configured"
        : cover.status === "quota_exhausted" ? "cover quota reached"
        : "cover unverified";
      els.coverVerdict.className = "cover-verdict unverified";
      lastCoverPct = "unverified";
    }
    return;
  }

  if (lastCoverPct === pct) return;
  lastCoverPct = pct;

  const cls = coverClass(pct);
  const startPct = parseFloat(els.coverPct.textContent) || 0;
  animateNumber(startPct, pct, 1000, (v) => {
    els.coverPct.textContent = Math.round(v);
  });

  els.coverFg.classList.remove("unverified");
  els.coverPct.classList.remove("unverified");
  els.coverFg.classList.remove("bad", "warn", "ok");
  els.coverFg.classList.add(cls);
  els.coverFg.style.strokeDashoffset = COVER_CIRCUMFERENCE * (1 - pct / 100);

  els.coverVerdict.textContent = cover.verdict || "";
  els.coverVerdict.className = "cover-verdict " + cls;
}

function coverClass(pct) {
  if (pct > 70) return "bad";
  if (pct > 40) return "warn";
  return "ok";
}

function verdictClass(pct) {
  if (pct > 70) return "bad";
  if (pct > 40) return "warn";
  if (pct > 20) return "warn";
  return "ok";
}

function detailFromStatus(status) {
  if (status === "service_down") return "letssubmit api unreachable";
  if (status === "no_url")       return "no playable url for detection";
  if (status === "cache_null")   return "cached as unverified";
  return "no result available";
}

function animateNumber(from, to, duration, cb) {
  const start = performance.now();
  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    // easeOutExpo
    const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    cb(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* progress log render*/
function renderProgressLog(entries) {
  if (!entries || entries.length <= renderedLogCount) return;
  for (let i = renderedLogCount; i < entries.length; i++) {
    const item = entries[i];
    const li = document.createElement("li");
    li.innerHTML = `<span class="log-t">+${item.t.toFixed(2)}s</span>` +
                   `<span class="log-msg"></span>`;
    li.querySelector(".log-msg").textContent = item.msg;
    els.logList.appendChild(li);
  }
  renderedLogCount = entries.length;
  els.logList.scrollTop = els.logList.scrollHeight;
}

/* recommendations*/
function renderRecommendations(recs, state, job) {
  if (!recs.length && state !== "done") return;

  const quotaExhausted = !!(job && job.quota_exhausted);

  if (recs.length > 0) {
    els.recsSection.hidden = false;
    els.recsEmpty.hidden = true;
    els.recsQuotaError.hidden = true;
    els.forceRecsPrompt.hidden = true;  //hide prompt once recs arrive
  }
  els.recsCount.textContent = String(recs.length);

  //append only new recs
  for (let i = renderedRecsCount; i < recs.length; i++) {
    const rec = recs[i];
    els.recsList.appendChild(buildDisc(rec, i - renderedRecsCount));
  }
  renderedRecsCount = recs.length;

  //empty-state handling at job completion
  if (state === "done" && recs.length === 0) {
    if (quotaExhausted) {
      //quota error takes precedence over everything else
      els.recsSection.hidden = false;
      els.recsEmpty.hidden = true;
      els.recsQuotaError.hidden = false;
    } else if (!isHumanVerdictAndNoRecs()) {
      els.recsSection.hidden = false;
      els.recsEmpty.hidden = false;
      els.recsQuotaError.hidden = true;
    }
  }
}

function isHumanVerdictAndNoRecs() {
  //the force-recs prompt handles this case 
  return !els.forceRecsPrompt.hidden;
}

function buildDisc(rec, animIndex) {
  const li = document.createElement("li");
  li.className = "disc";
  li.style.animationDelay = `${animIndex * 90}ms`;
  li.dataset.url = rec.url || "";
  li.dataset.embedUrl = rec.embed_url || "";
  li.dataset.platform = rec.platform || "";
  li.dataset.title = rec.title || "untitled";
  li.dataset.artist = rec.artist || "unknown";

  const badge = badgeFor(rec);
  const artwork = rec.artwork || "";
  const artStyle = artwork
    ? `style="background-image:url('${escapeAttr(artwork)}')"`
    : "";

  let scoreChips = "";
  if (rec.scores) {
    const s = rec.scores;
    const chip = (k, v) =>
      `<span class="disc-score-chip" title="${k}">${k}:${v.toFixed(2)}</span>`;
    scoreChips = `
      <span class="disc-scores">
        ${s.tag   > 0 ? chip("tag",   s.tag)   : ""}
        ${s.cf    > 0 ? chip("cf",    s.cf)    : ""}
        ${s.audio > 0 ? chip("audio", s.audio) : ""}
      </span>`;
  }

  li.innerHTML = `
    <div class="disc-face">
      <div class="disc-label" ${artStyle}></div>
      <div class="disc-hole"></div>
      <span class="disc-badge ${badge.cls}">${badge.text}</span>
    </div>
    <div class="disc-meta">
      <span class="disc-title"></span>
      <span class="disc-artist"></span>
      ${scoreChips}
    </div>
  `;
  li.querySelector(".disc-title").textContent = rec.title || "untitled";
  li.querySelector(".disc-artist").textContent = rec.artist || "unknown";

  li.addEventListener("click", () => onDiscClicked(li, rec));
  return li;
}

function escapeAttr(s) {
  return String(s).replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/*disc click -> shrink toward player -> load + play in vinyl player*/

// Active playback state
let currentTrack = null;
let ytPlayer = null;        //youTube IFrame API player instance
let ytApiReady = false;
let ytApiLoading = false;
const YT_API_QUEUE = [];    //callbacks queued before API loads

//YouTube IFrame API loader 
//we load the API once on first use.when it finishes loading,it calls window.onYouTubeIframeAPIReady automatically.
function ensureYouTubeApi(cb) {
  if (ytApiReady) { cb(); return; }
  YT_API_QUEUE.push(cb);
  if (ytApiLoading) return;
  ytApiLoading = true;

  window.onYouTubeIframeAPIReady = () => {
    ytApiReady = true;
    YT_API_QUEUE.splice(0).forEach((fn) => { try { fn(); } catch (e) {} });
  };
  const tag = document.createElement("script");
  tag.src = "https://www.youtube.com/iframe_api";
  document.head.appendChild(tag);
}

function onDiscClicked(discEl, rec) {
  if (!rec.embed_url) {
    setVinylStatus("no preview available", "idle");
    return;
  }

  //compute shrink direction toward the player's spinner center.
  //if the vinyl player isn't onscreen, scroll it into view first.
  els.vinylPlayer.scrollIntoView({ behavior: "smooth", block: "center" });

  //defer the geometry read until after scroll has visually started,so the shrink target lands where the player will be.
  requestAnimationFrame(() => {
    const discRect   = discEl.getBoundingClientRect();
    const targetRect = els.vinylSpinner.getBoundingClientRect();
    const targetX = targetRect.left + targetRect.width / 2;
    const targetY = targetRect.top  + targetRect.height / 2;
    const dx = targetX - (discRect.left + discRect.width / 2);
    const dy = targetY - (discRect.top  + discRect.height / 2);
    discEl.style.setProperty("--shrink-dx", `${dx}px`);
    discEl.style.setProperty("--shrink-dy", `${dy}px`);
    discEl.classList.add("shrinking");
    setTimeout(() => discEl.classList.remove("shrinking"), 700);
  });

  loadIntoVinylPlayer(rec);
}

function loadIntoVinylPlayer(rec) {
  //auto-stop anything currently playing
  teardownAudioCorner();
  currentTrack = rec;

  //update the vinyl meta panel
  els.vinylNowTitle.textContent = rec.title || "untitled";
  els.vinylNowArtist.textContent = rec.artist || "unknown";
  if (rec.artwork) {
    els.vinylArt.style.backgroundImage = `url("${escapeAttr(rec.artwork)}")`;
    els.vinylArt.classList.add("has-art");
  } else {
    els.vinylArt.style.backgroundImage = "";
    els.vinylArt.classList.remove("has-art");
  }
  if (rec.url) {
    els.vinylExternal.href = rec.url;
    els.vinylExternal.hidden = false;
  } else {
    els.vinylExternal.hidden = true;
  }
  els.vinylStop.disabled = false;

  //did the user analyze a YouTube link / YouTube-sourced track? If yes,
  //and this rec only has a Spotify embed, try to find a matching YouTube
  //video so the rec plays on the same platform the user came in on.
  const cameFromYouTube = !!(
    lastJobSnapshot &&
    lastJobSnapshot.track &&
    (lastJobSnapshot.track.youtube_url || lastJobSnapshot.track.youtube_id)
  );
  const recIsSpotifyOnly = !rec.video_id && rec.embed_url;

  if (cameFromYouTube && recIsSpotifyOnly) {
    //show loading state while we ask the backend; player setup happens inside the resolver once we know whether to use YouTube or Spotify.
    setVinylStatus("finding youtube version...", "loading");
    lazyYoutubeLookupAndPlay(rec);
    return;
  }

  //default branching: video_id present -> YouTube; else Spotify embed.
  if (rec.video_id) {
    setupYouTubeCornerWidget(rec);
  } else if (rec.embed_url) {
    setupSpotifyCornerWidget(rec);
  }
}

async function lazyYoutubeLookupAndPlay(rec) {
  // snapshot the rec the user clicked, so if they click another disc before this request returns we don't accidentally play the stale one.
  const clickedRec = rec;
  try {
    const resp = await fetch("/api/youtube_lookup", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        artist: rec.artist || "",
        title:  rec.title  || "",
      }),
    });
    //if the user clicked something else while we were waiting, bail.
    if (currentTrack !== clickedRec) return;

    if (!resp.ok) {
      setupSpotifyCornerWidget(rec);
      return;
    }
    const data = await resp.json();
    if (data && data.video_id) {
      //build an augmented rec with the YouTube video_id and play it.
      const ytWatchUrl = data.url ||
        `https://www.youtube.com/watch?v=${data.video_id}`;
      const augmented = {...rec,
        video_id: data.video_id,
        embed_url: data.embed_url,
        url: ytWatchUrl,
      };
      currentTrack = augmented;
      //update the external-link button to the YouTube URL now.
      if (els.vinylExternal) {
        els.vinylExternal.href = ytWatchUrl;
        els.vinylExternal.hidden = false;
      }
      setupYouTubeCornerWidget(augmented);
    } else {
      //no confident YouTube match. fall back to the Spotify embed.
      setupSpotifyCornerWidget(rec);
    }
  } catch (e) {
    //network or backend error - silently fall back to Spotify.
    if (currentTrack === clickedRec) setupSpotifyCornerWidget(rec);
  }
}

function setupYouTubeCornerWidget(rec) {
  setVinylStatus("loading...", "loading");
  els.audioCornerLabel.textContent = "PLAYING ON YOUTUBE";
  els.audioCornerHint.textContent = "click play below if it doesn't auto-start";
  els.audioCornerHint.hidden = false;
  els.audioCorner.hidden = false;

  //build a target div for the YT API to mount into
  els.audioCornerHost.innerHTML = '<div id="yt-player-target"></div>';

  ensureYouTubeApi(() => {
    //tear down any previous instance
    if (ytPlayer && ytPlayer.destroy) {
      try { ytPlayer.destroy(); } catch (e) {}
      ytPlayer = null;
    }
    ytPlayer = new window.YT.Player("yt-player-target", {
      height: 168,
      width: 296,
      videoId: rec.video_id,
      playerVars: {
        autoplay: 1,
        controls: 1,        //show controls so user can click play if autoplay blocked
        modestbranding: 1,
        playsinline: 1,
        rel: 0,
        fs: 0,              //no fullscreen
      },
      events: {
        onReady: (e) => {
          //try autoplay (may be blocked by browser policy, that's fine)
          try { e.target.playVideo(); } catch (err) {}
        },
        onStateChange: (e) => {
          // 1 = playing, 2 = paused, 0 = ended
          if (e.data === 1) {
            setVinylStatus("now playing", "playing");
            els.audioCornerHint.hidden = true;
          } else if (e.data === 2) {
            setVinylStatus("paused", "idle");
          } else if (e.data === 0) {
            setVinylStatus("ended", "idle");
          }
        },
      },
    });
  });
}

function setupSpotifyCornerWidget(rec) {
  // spotify embeds expose no play-state API to the parent. we show
  // the embed visibly and assume the user will click play;
  setVinylStatus("now playing", "playing");
  els.audioCornerLabel.textContent = "PLAYING ON SPOTIFY";
  els.audioCornerHint.textContent = "click play in the widget above";
  els.audioCornerHint.hidden = false;

  // use spotify's full-width "compact" embed (80px)
  els.audioCornerHost.innerHTML = "";
  const iframe = document.createElement("iframe");
  iframe.src = rec.embed_url;
  iframe.width = "100%";
  iframe.height = 80;
  iframe.allow = "encrypted-media; autoplay; clipboard-write";
  iframe.loading = "eager";
  els.audioCornerHost.appendChild(iframe);

  els.audioCorner.hidden = false;
}

function teardownAudioCorner() {
  // Clean up YouTube API instance if present
  if (ytPlayer && ytPlayer.destroy) {
    try { ytPlayer.destroy(); } catch (e) {}
    ytPlayer = null;
  }
  els.audioCornerHost.innerHTML = "";
}

function stopVinylPlayer() {
  teardownAudioCorner();
  currentTrack = null;
  els.audioCorner.hidden = true;
  els.vinylArt.style.backgroundImage = "";
  els.vinylArt.classList.remove("has-art");
  els.vinylNowTitle.innerHTML = "&mdash;";
  els.vinylNowArtist.innerHTML = "&mdash;";
  els.vinylExternal.hidden = true;
  els.vinylStop.disabled = true;
  setVinylStatus("no disc loaded", "idle");
}

function setVinylStatus(text, state) {
  els.vinylStatusText.textContent = text;
  els.vinylPlayer.dataset.state = state;
}

function bindVinylPlayer() {
  els.vinylStop.addEventListener("click", stopVinylPlayer);
  els.audioCornerClose.addEventListener("click", stopVinylPlayer);
}

function bindForceRecs() {
  els.forceRecsBtn.addEventListener("click", async () => {
    if (!currentJobId) return;

    //silent no-op when LetsSubmit quota is exhausted (per spec).the button remains visible but clicking
    // does nothing (the user can see the persistent quota-error message in the recs area)
    if (lastJobSnapshot && lastJobSnapshot.quota_exhausted) return;

    els.forceRecsBtn.disabled = true;
    els.forceRecsBtn.querySelector(".btn-text").textContent = "running...";
    //record that the user acted on the prompt so the poll loop stops
    //re-showing it once the background recommendation run completes
    forceRecsHandled = true;
    els.forceRecsPrompt.hidden = true;
    try {
      const resp = await fetch(`/api/job/${currentJobId}/force_recs`, {
        method: "POST",
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }
      //continuation runs in the background; resume polling
      startPolling(currentJobId);
      setStatus("busy", "recommending");
    } catch (err) {
      els.forceRecsBtn.disabled = false;
      els.forceRecsBtn.querySelector(".btn-text").textContent = "show similar tracks";
      showError(err.message);
    }
  });
}

function badgeFor(rec) {
  const m = rec.verification_method || "";
  if (m === "hybrid-msd")           return {cls: "hybrid", text: "msd+tags"};
  if (m === "spotify-discovered-on") return {cls: "verified-spotify", text: "spotify"};
  if (m === "spotify")              return {cls: "verified-spotify", text: "spotify"};
  if (m === "youtube")              return {cls: "verified-youtube", text: "youtube"};
  if (m === "session_cache")        return {cls: "verified-youtube", text: "cached"};
  return {cls: "unverified", text: "unverified"};
}

/*status pill*/
function setStatus(state, label) {
  els.status.dataset.state = state;
  els.status.innerHTML = `<span class="meta-dot"></span> ${label}`;
}

function showError(msg) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="log-t">!err</span><span class="log-msg"></span>`;
  li.querySelector(".log-msg").textContent = msg;
  li.style.color = "var(--accent-magenta)";
  els.logList.appendChild(li);
  els.logList.scrollTop = els.logList.scrollHeight;
}