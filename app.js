// Velo — markerless movement analysis. Runs entirely in the browser.
const TV = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const $ = s => document.querySelector(s);
const G = 9.81;

const JOINTS = [
  { id: 'kneeL', name: 'Left knee', a: 23, b: 25, c: 27, color: '#4FA3FF' },
  { id: 'kneeR', name: 'Right knee', a: 24, b: 26, c: 28, color: '#A9D1FF' },
  { id: 'hipL', name: 'Left hip', a: 11, b: 23, c: 25, color: '#FF5A3C' },
  { id: 'hipR', name: 'Right hip', a: 12, b: 24, c: 26, color: '#FFA28F' },
  { id: 'ankleL', name: 'Left ankle', a: 25, b: 27, c: 31, color: '#3DDC97' },
  { id: 'ankleR', name: 'Right ankle', a: 26, b: 28, c: 32, color: '#9EF0C9' },
  { id: 'shoulderL', name: 'Left shoulder', a: 23, b: 11, c: 13, color: '#FFB547' },
  { id: 'shoulderR', name: 'Right shoulder', a: 24, b: 12, c: 14, color: '#FFD796' },
  { id: 'elbowL', name: 'Left elbow', a: 11, b: 13, c: 15, color: '#C792EA' },
  { id: 'elbowR', name: 'Right elbow', a: 12, b: 14, c: 16, color: '#E2C6F5' },
];

const view = $('#view'), ctx = view.getContext('2d');
const vid = $('#vid');
let lmk = null, lmkMode = 'VIDEO', loading = null;
let media = null, mediaType = null, W = 0, H = 0;
let frames = new Map();           // key -> {t, lm, wl}
let A = null;                     // analysis result
let lastDetT = -1, lastTs = 0, dirty = true, lastAnalysis = 0, newFrames = 0;
let mode = 'skeleton';
let shownJoints = new Set();
let injected = null;              // test hook pose

// ---------- utils ----------
const sub = (p, q) => ({ x: p.x - q.x, y: p.y - q.y });
const add = (p, q) => ({ x: p.x + q.x, y: p.y + q.y });
const mul = (p, k) => ({ x: p.x * k, y: p.y * k });
const mid = (p, q) => ({ x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 });
const len = p => Math.hypot(p.x, p.y);
const norm = p => { const l = len(p) || 1; return { x: p.x / l, y: p.y / l }; };
const perp = p => ({ x: -p.y, y: p.x });
const lerp = (p, q, t) => ({ x: p.x + (q.x - p.x) * t, y: p.y + (q.y - p.y) * t });
function angle3(a, b, c) { // interior angle at b, 3D
  const u = [a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0)], v = [c.x - b.x, c.y - b.y, (c.z || 0) - (b.z || 0)];
  const d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2], n = Math.hypot(...u) * Math.hypot(...v) || 1;
  return Math.acos(Math.max(-1, Math.min(1, d / n))) * 180 / Math.PI;
}
function smooth(arr, w) {
  const h = Math.floor(w / 2), out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    let s = 0, n = 0;
    for (let k = i - h; k <= i + h; k++) if (k >= 0 && k < arr.length && Number.isFinite(arr[k])) { s += arr[k]; n++; }
    out[i] = n ? s / n : NaN;
  }
  return out;
}
function deriv(t, v) {
  const out = new Array(v.length).fill(0);
  for (let i = 0; i < v.length; i++) {
    const a = Math.max(0, i - 1), b = Math.min(v.length - 1, i + 1);
    const dt = t[b] - t[a]; out[i] = dt > 0 ? (v[b] - v[a]) / dt : 0;
  }
  return out;
}
const median = a => { const s = a.filter(Number.isFinite).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
const fmt = (v, d = 0) => Number.isFinite(v) ? v.toFixed(d) : '—';
function status(msg, err) { const s = $('#status'); if (!msg) { s.hidden = true; return; } s.hidden = false; s.textContent = msg; s.classList.toggle('err', !!err); }

// ---------- model ----------
async function getLandmarker() {
  if (lmk) return lmk;
  if (loading) return loading;
  loading = (async () => {
    status('Loading the pose model (about 9 MB, first time only)…');
    const vision = await import(`${TV}/vision_bundle.mjs`);
    const fs = await vision.FilesetResolver.forVisionTasks(`${TV}/wasm`);
    const opts = d => ({ baseOptions: { modelAssetPath: MODEL, delegate: d }, runningMode: 'VIDEO', numPoses: 1, minPoseDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    try { lmk = await vision.PoseLandmarker.createFromOptions(fs, opts('GPU')); }
    catch { lmk = await vision.PoseLandmarker.createFromOptions(fs, opts('CPU')); }
    lmkMode = 'VIDEO';
    status('');
    return lmk;
  })().catch(e => { loading = null; status('Could not load the pose model. Check your connection and try again. (' + (e && e.message || e) + ')', true); throw e; });
  return loading;
}
async function setMode(m) { if (lmk && lmkMode !== m) { await lmk.setOptions({ runningMode: m }); lmkMode = m; } }
const copyLm = arr => arr.map(p => ({ x: p.x, y: p.y, z: p.z, v: p.visibility ?? 1 }));

// ---------- media loading ----------
function sizeCanvas(w, h) {
  const k = Math.min(1, 1280 / Math.max(w, h));
  W = Math.round(w * k); H = Math.round(h * k);
  view.width = W; view.height = H;
}
function resetData() { frames = new Map(); A = null; lastDetT = -1; newFrames = 0; injected = null; }
function showUI() { $('#empty').hidden = true; $('#controls').hidden = false; $('#hud').hidden = false; }

async function loadFile(file) {
  if (!file) return;
  resetData();
  const url = URL.createObjectURL(file);
  if (file.type.startsWith('image/')) return loadImage(url);
  mediaType = 'video'; media = vid;
  vid.src = url; vid.loop = true; vid.playbackRate = +$('#rate').value;
  await new Promise((res, rej) => { vid.onloadeddata = res; vid.onerror = () => rej(new Error('This video format is not supported by your browser.')); }).catch(e => { status(e.message, true); throw e; });
  sizeCanvas(vid.videoWidth, vid.videoHeight); showUI(); dirty = true;
  try { await getLandmarker(); await setMode('VIDEO'); } catch { return; }
  status('Analyzing. Let it play through once for the full report; slow speed samples more frames.');
  setTimeout(() => status(''), 5000);
  vid.play().catch(() => {});
}
async function loadImage(url) {
  resetData();
  const img = new Image(); img.src = url;
  await img.decode().catch(() => {});
  mediaType = 'image'; media = img; vid.pause();
  sizeCanvas(img.naturalWidth, img.naturalHeight); showUI(); dirty = true;
  $('#play').disabled = true; $('#scrub').disabled = true;
  try { await getLandmarker(); await setMode('IMAGE'); } catch { render(); return; }
  const r = lmk.detect(img);
  if (r.landmarks && r.landmarks[0]) frames.set(0, { t: 0, lm: copyLm(r.landmarks[0]), wl: copyLm(r.worldLandmarks[0]) });
  else status('No person found in this image. Try a photo where the whole body is visible.', true);
  analyze(); dirty = true;
}

// ---------- detection loop ----------
function tick() {
  requestAnimationFrame(tick);
  if (!media) return;
  if (mediaType === 'video' && vid.readyState >= 2) {
    const t = vid.currentTime;
    if (lmk && lmkMode === 'VIDEO' && t !== lastDetT) {
      let now = performance.now(); if (now <= lastTs) now = lastTs + 1; lastTs = now;
      try {
        const r = lmk.detectForVideo(vid, now);
        const key = Math.round(t * 60);
        if (r.landmarks && r.landmarks[0] && !frames.has(key)) { frames.set(key, { t, lm: copyLm(r.landmarks[0]), wl: copyLm(r.worldLandmarks[0]) }); newFrames++; }
      } catch (e) { /* skip frame */ }
      lastDetT = t; dirty = true;
    }
    if (t !== render.lastT) dirty = true;
    if (!$('#scrub').matches(':active') && vid.duration) $('#scrub').value = Math.round(t / vid.duration * 1000);
  }
  if (newFrames && performance.now() - lastAnalysis > 600) { analyze(); newFrames = 0; lastAnalysis = performance.now(); }
  if (dirty) { render(); dirty = false; }
}

// ---------- analysis ----------
function px(lm, i) { return { x: lm[i].x * W, y: lm[i].y * H, v: lm[i].v }; }
function analyze() {
  const list = [...frames.values()].sort((a, b) => a.t - b.t);
  if (!list.length) { A = null; updatePanel(); return; }
  const t = list.map(f => f.t);
  const ang = {};
  for (const j of JOINTS) ang[j.id] = smooth(list.map(f => angle3(f.wl[j.a], f.wl[j.b], f.wl[j.c])), 5);
  const trunk = smooth(list.map(f => {
    const s = mid(f.wl[11], f.wl[12]), h = mid(f.wl[23], f.wl[24]); const v = sub(s, h);
    return Math.acos(Math.max(-1, Math.min(1, -v.y / (len(v) || 1)))) * 180 / Math.PI;
  }), 5);
  // meters per pixel from leg length (world) vs image
  const scale = median(list.map(f => {
    const wlen = Math.hypot(f.wl[23].x - f.wl[25].x, f.wl[23].y - f.wl[25].y, f.wl[23].z - f.wl[25].z) + Math.hypot(f.wl[25].x - f.wl[27].x, f.wl[25].y - f.wl[27].y, f.wl[25].z - f.wl[27].z);
    const ilen = len(sub(px(f.lm, 23), px(f.lm, 25))) + len(sub(px(f.lm, 25), px(f.lm, 27)));
    return ilen > 20 ? wlen / ilen : NaN;
  }));
  const hipX = smooth(list.map(f => (f.lm[23].x + f.lm[24].x) / 2 * W * scale), 5);
  const hipY = smooth(list.map(f => -(f.lm[23].y + f.lm[24].y) / 2 * H * scale), 5);
  const vx = smooth(deriv(t, hipX), 5), vy = smooth(deriv(t, hipY), 5);
  const speed = vx.map((v, i) => Math.hypot(v, vy[i]));
  const ay = smooth(deriv(t, vy), 7);
  const grf = ay.map(a => Math.max(0, 1 + a / G));
  const vel = {}; for (const j of JOINTS) vel[j.id] = smooth(deriv(t, ang[j.id]), 3);
  // rom & primary joint
  const rom = {}; for (const j of JOINTS) { const a = ang[j.id].filter(Number.isFinite); rom[j.id] = a.length ? Math.max(...a) - Math.min(...a) : 0; }
  const primary = JOINTS.slice().sort((p, q) => rom[q.id] - rom[p.id])[0];
  const reps = countReps(ang[primary.id]);
  // view
  const f0 = list[Math.floor(list.length / 2)].lm;
  const shW = Math.abs(f0[11].x - f0[12].x) * W, torso = len(sub(mid(px(f0, 11), px(f0, 12)), mid(px(f0, 23), px(f0, 24))));
  const frontal = shW / (torso || 1) > 0.45;
  A = { list, t, ang, vel, trunk, speed, grf, rom, primary, reps, frontal, scale };
  if (!shownJoints.size) JOINTS.slice().sort((p, q) => rom[q.id] - rom[p.id]).slice(0, 4).forEach(j => shownJoints.add(j.id));
  updatePanel(); dirty = true;
}
function countReps(a) {
  const v = a.filter(Number.isFinite); if (v.length < 8) return 0;
  const lo = Math.min(...v), hi = Math.max(...v), r = hi - lo; if (r < 25) return 0;
  const tl = lo + 0.3 * r, th = lo + 0.7 * r; let state = v[0] > th ? 'up' : 'mid', n = 0, wentDown = false;
  for (const x of v) {
    if (x < tl) wentDown = true;
    if (x > th) { if (wentDown && state !== 'start') n++; wentDown = false; state = 'up'; }
  }
  return n;
}
function nearest(t) {
  if (injected) return injected;
  if (!A) return null;
  const L = A.list; let lo = 0, hi = L.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (L[m].t < t) lo = m + 1; else hi = m; }
  const i = lo > 0 && Math.abs(L[lo - 1].t - t) < Math.abs(L[lo].t - t) ? lo - 1 : lo;
  return Math.abs(L[i].t - t) < 0.25 || mediaType === 'image' ? { ...L[i], i } : null;
}

// ---------- technique ----------
function techniqueChecks() {
  if (!A) return [];
  const out = [], L = A.list;
  const knee = A.ang.kneeL.map((v, i) => (v + A.ang.kneeR[i]) / 2);
  const kneeRom = (A.rom.kneeL + A.rom.kneeR) / 2;
  const single = L.length < 3;
  let iMin = 0; knee.forEach((v, i) => { if (v < knee[iMin]) iMin = i; });
  const lm = L[iMin].lm;
  const lower = single || kneeRom > 35;
  if (lower) {
    const hipY = (lm[23].y + lm[24].y) / 2 * H, kneeY = (lm[25].y + lm[26].y) / 2 * H;
    const below = hipY >= kneeY - 0.01 * H, kMin = knee[iMin];
    if (kMin < 120) out.push(below
      ? { s: 'ok', h: 'Depth', m: `${fmt(kMin)}°`, p: 'Hip crease reaches knee height or lower. Full-depth rep.' }
      : { s: kMin < 105 ? 'warn' : 'bad', h: 'Depth', m: `${fmt(kMin)}°`, p: `Hips stop above the knees, about ${fmt(Math.max(0, kMin - 90))}° short of parallel. Sit back and down a little further.` });
    if (A.frontal) {
      const side = (h, k, a) => {
        const H0 = px(lm, h), K = px(lm, k), An = px(lm, a); const line = lerp(H0, An, (K.y - H0.y) / ((An.y - H0.y) || 1));
        const midX = (lm[23].x + lm[24].x) / 2 * W; const inward = (Math.abs(line.x - midX) - Math.abs(K.x - midX));
        return inward / (Math.abs(lm[23].x - lm[24].x) * W || 1);
      };
      const v = Math.max(side(23, 25, 27), side(24, 26, 28));
      out.push(v > 0.18 ? { s: 'bad', h: 'Knee tracking', m: `${fmt(v * 100)}% in`, p: 'Knees cave inward at the bottom (valgus). Push knees out over the middle toes.' }
        : v > 0.09 ? { s: 'warn', h: 'Knee tracking', m: `${fmt(v * 100)}% in`, p: 'Slight inward knee drift at the bottom. Cue "spread the floor".' }
          : { s: 'ok', h: 'Knee tracking', m: 'neutral', p: 'Knees stay stacked over the feet.' });
    } else out.push({ s: 'idle', h: 'Knee tracking', m: 'side view', p: 'Film from the front to check whether the knees cave in.' });
  }
  const lean = Math.max(...A.trunk.filter(Number.isFinite));
  if (Number.isFinite(lean)) out.push(lean > 50 ? { s: 'bad', h: 'Trunk angle', m: `${fmt(lean)}°`, p: 'Chest drops well forward. Brace, keep ribs stacked over the hips and drive the chest up.' }
    : lean > 38 ? { s: 'warn', h: 'Trunk angle', m: `${fmt(lean)}°`, p: 'Moderate forward lean. Fine for low-bar or hinge patterns; watch it if the bar drifts forward.' }
      : { s: 'ok', h: 'Trunk angle', m: `${fmt(lean)}°`, p: 'Torso stays upright through the movement.' });
  if (!single) {
    const diff = (a, b) => median(A.ang[a].map((v, i) => Math.abs(v - A.ang[b][i])));
    const dk = diff('kneeL', 'kneeR'), dh = diff('hipL', 'hipR'), d = Math.max(dk, dh);
    out.push(!A.frontal ? { s: 'idle', h: 'Left / right symmetry', m: `${fmt(d)}°`, p: 'One side is partly hidden in a side view. Film front-on for a reliable comparison.' }
      : d > 12 ? { s: 'warn', h: 'Left / right symmetry', m: `${fmt(d)}° gap`, p: `${dk > dh ? 'Knees' : 'Hips'} bend unevenly. One side is doing more of the work.` }
        : { s: 'ok', h: 'Left / right symmetry', m: `${fmt(d)}° gap`, p: 'Both sides move evenly.' });
    const pj = A.primary; const a = A.ang[pj.id].filter(Number.isFinite);
    out.push({ s: 'idle', h: 'Primary motion', m: `${fmt(Math.min(...a))}°–${fmt(Math.max(...a))}°`, p: `${pj.name} drives this movement, peak ${fmt(Math.max(...A.vel[pj.id].map(Math.abs)))}°/s.` });
  }
  return out;
}

// ---------- panel ----------
function updatePanel() {
  const single = A && A.list.length < 3;
  $('#kReps').textContent = A && !single ? String(A.reps) : '—';
  $('#kRepsJ').textContent = A && !single ? A.primary.name.toLowerCase() : 'primary joint';
  $('#kSpeed').textContent = A && !single ? fmt(Math.max(...A.speed.filter(Number.isFinite)), 2) : '—';
  const gmax = A && !single ? Math.max(...A.grf.filter(Number.isFinite)) : NaN;
  $('#kForce').textContent = Number.isFinite(gmax) ? fmt(gmax, 2) + '×' : '—';
  $('#kForceN').textContent = Number.isFinite(gmax) ? `${fmt(gmax * (+$('#bw').value || 70) * G)} N` : '× body weight';
  $('#kRom').textContent = A && !single ? fmt(A.rom[A.primary.id]) + '°' : '—';
  $('#kRomJ').textContent = A && !single ? A.primary.name.toLowerCase() : 'degrees';
  $('#viewNote').textContent = !A ? 'Upload to analyze' : (A.frontal ? 'Front view detected' : 'Side view detected') + ` · ${A.list.length} frame${A.list.length === 1 ? '' : 's'}`;
  $('#hudView').textContent = !A ? '—' : A.frontal ? 'Front view' : 'Side view';
  const checks = techniqueChecks();
  $('#checks').innerHTML = checks.length ? checks.map(c => `<li class="check ${c.s}"><b>${c.h}<em>${c.m}</em></b><p>${c.p}</p></li>`).join('')
    : '<li class="check idle"><b>Waiting for movement</b><p>Checks appear once Velo has seen the body move.</p></li>';
  // chips
  $('#angleChips').innerHTML = JOINTS.filter(j => !A || A.rom[j.id] > 0 || single).map(j => `<button class="chip" data-j="${j.id}" aria-pressed="${shownJoints.has(j.id)}"><i style="background:${j.color}"></i>${j.name}</button>`).join('');
  drawCharts();
}
$('#angleChips').addEventListener('click', e => {
  const b = e.target.closest('.chip'); if (!b) return;
  const id = b.dataset.j; shownJoints.has(id) ? shownJoints.delete(id) : shownJoints.add(id);
  b.setAttribute('aria-pressed', shownJoints.has(id)); drawCharts();
});

function chart(cv, series, opt = {}) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 300, h = +cv.getAttribute('height');
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + 'px';
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  const pl = 36, pr = opt.right ? 36 : 8, pt = 8, pb = 20;
  c.font = '11px "IBM Plex Sans", sans-serif'; c.fillStyle = '#8B97A6';
  if (!A || A.list.length < 3) { c.fillText(mediaType === 'image' ? 'Upload a video to see this over time.' : 'Play the video to collect frames.', pl, h / 2); return; }
  const t = A.t, t0 = t[0], t1 = t[t.length - 1] || 1;
  const X = x => pl + (x - t0) / ((t1 - t0) || 1) * (w - pl - pr);
  const axis = (vals, side) => {
    const v = vals.flat().filter(Number.isFinite); let lo = Math.min(...v), hi = Math.max(...v);
    if (opt.zero && side === 'l') lo = Math.min(lo, 0); if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad; return { lo, hi, Y: y => pt + (1 - (y - lo) / (hi - lo)) * (h - pt - pb) };
  };
  const left = axis(series.filter(s => !s.right).map(s => s.data), 'l');
  const right = opt.right ? axis(series.filter(s => s.right).map(s => s.data), 'r') : null;
  c.strokeStyle = '#242D38'; c.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const y = pt + k / 3 * (h - pt - pb); c.beginPath(); c.moveTo(pl, y); c.lineTo(w - pr, y); c.stroke();
    const val = left.hi - k / 3 * (left.hi - left.lo); c.fillText(fmt(val, Math.abs(left.hi - left.lo) < 6 ? 1 : 0), 2, y + 4);
    if (right) { const rv = right.hi - k / 3 * (right.hi - right.lo); c.fillText(fmt(rv, 1), w - pr + 4, y + 4); }
  }
  c.fillText(`${fmt(t0, 1)} s`, pl, h - 5); c.fillText(`${fmt(t1, 1)} s`, w - pr - 30, h - 5);
  if (opt.zero) { const y = left.Y(0); c.strokeStyle = '#33404F'; c.beginPath(); c.moveTo(pl, y); c.lineTo(w - pr, y); c.stroke(); }
  for (const s of series) {
    const ax = s.right ? right : left; c.strokeStyle = s.color; c.lineWidth = 1.8; c.beginPath(); let pen = false;
    s.data.forEach((v, i) => { if (!Number.isFinite(v)) { pen = false; return; } const x = X(t[i]), y = ax.Y(v); pen ? c.lineTo(x, y) : c.moveTo(x, y); pen = true; });
    c.stroke();
  }
  const now = mediaType === 'video' ? vid.currentTime : 0;
  if (now >= t0 && now <= t1) { c.strokeStyle = '#E7ECF2'; c.globalAlpha = .5; c.beginPath(); c.moveTo(X(now), pt); c.lineTo(X(now), h - pb); c.stroke(); c.globalAlpha = 1; }
}
function drawCharts() {
  const js = JOINTS.filter(j => shownJoints.has(j.id));
  chart($('#cAngles'), A ? js.map(j => ({ data: A.ang[j.id], color: j.color })) : []);
  chart($('#cVel'), A ? js.map(j => ({ data: A.vel[j.id], color: j.color })) : [], { zero: true });
  chart($('#cForce'), A ? [{ data: A.speed, color: '#4FA3FF' }, { data: A.grf, color: '#FF5A3C', right: true }] : [], { right: true });
}

// ---------- overlay drawing ----------
const BONE_FILL = 'rgba(79,163,255,0.20)', BONE_EDGE = 'rgba(168,210,255,0.92)';
function boneGradient(a, b, wmax) {
  const m = mid(a, b), n = perp(norm(sub(b, a)));
  const g = ctx.createLinearGradient(m.x + n.x * wmax, m.y + n.y * wmax, m.x - n.x * wmax, m.y - n.y * wmax);
  g.addColorStop(0, 'rgba(150,200,255,0.55)'); g.addColorStop(0.28, 'rgba(79,163,255,0.16)');
  g.addColorStop(0.72, 'rgba(79,163,255,0.16)'); g.addColorStop(1, 'rgba(150,200,255,0.55)');
  return g;
}
// long bone with anatomical width profile [proximal end, neck, mid-shaft, distal neck, distal end]
function longBone(a, b, prof, k, alpha = 1) {
  const d = sub(b, a), L = len(d); if (L < 2) return;
  const u = norm(d), n = perp(u), ts = [0, 0.1, 0.5, 0.9, 1];
  const left = [], right = [];
  ts.forEach((t, i) => { const c = lerp(a, b, t), w = prof[i] * k / 2; left.push(add(c, mul(n, w))); right.push(add(c, mul(n, -w))); });
  const pts = [...left, ...right.reverse()];
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.beginPath(); ctx.moveTo((pts[0].x + pts[pts.length - 1].x) / 2, (pts[0].y + pts[pts.length - 1].y) / 2);
  for (let i = 0; i < pts.length; i++) { const p = pts[i], q = pts[(i + 1) % pts.length]; ctx.quadraticCurveTo(p.x, p.y, (p.x + q.x) / 2, (p.y + q.y) / 2); }
  ctx.closePath(); ctx.fillStyle = boneGradient(a, b, Math.max(...prof) * k / 2); ctx.fill();
  ctx.strokeStyle = BONE_EDGE; ctx.lineWidth = Math.max(1, k * 0.006); ctx.stroke();
  // epiphyses
  for (const [c, w] of [[lerp(a, b, 0.03), prof[0]], [lerp(a, b, 0.97), prof[4]]]) {
    ctx.beginPath(); ctx.ellipse(c.x, c.y, w * k * 0.5, w * k * 0.36, Math.atan2(u.y, u.x) + Math.PI / 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(120,185,255,0.18)'; ctx.fill(); ctx.stroke();
  }
  // medullary line
  ctx.beginPath(); const m0 = lerp(a, b, 0.18), m1 = lerp(a, b, 0.82); ctx.moveTo(m0.x, m0.y); ctx.lineTo(m1.x, m1.y);
  ctx.strokeStyle = 'rgba(200,228,255,0.35)'; ctx.lineWidth = Math.max(1, k * 0.004); ctx.stroke();
  ctx.restore();
}
function smoothShape(pts, fill, stroke, lw) {
  ctx.beginPath(); const n = pts.length;
  ctx.moveTo((pts[0].x + pts[n - 1].x) / 2, (pts[0].y + pts[n - 1].y) / 2);
  for (let i = 0; i < n; i++) { const p = pts[i], q = pts[(i + 1) % n]; ctx.quadraticCurveTo(p.x, p.y, (p.x + q.x) / 2, (p.y + q.y) / 2); }
  ctx.closePath(); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}
function body(lm) {
  const P = i => px(lm, i);
  const lSh = P(11), rSh = P(12), lHip = P(23), rHip = P(24);
  const midSh = mid(lSh, rSh), midHip = mid(lHip, rHip);
  const S = len(sub(midSh, midHip)) || 100;
  const up = norm(sub(midSh, midHip));
  const earMid = mid(P(7), P(8)), nose = P(0);
  const facing = Math.sign(nose.x - earMid.x) || 1;
  return { P, lSh, rSh, lHip, rHip, midSh, midHip, S, up, earMid, nose, facing };
}
function drawSkeleton(lm, alpha = 1) {
  const B = body(lm), { P, S, up } = B, lw = Math.max(1, S * 0.006);
  ctx.save(); ctx.globalAlpha = alpha;
  // spine
  const sBase = add(B.midHip, mul(up, S * 0.08)), sTop = add(B.midSh, mul(up, S * 0.2));
  for (let i = 0; i < 17; i++) {
    const c = lerp(sBase, sTop, (i + 0.5) / 17), w = S * (0.07 - i * 0.0022), h = S * 0.042;
    ctx.beginPath(); ctx.ellipse(c.x, c.y, w / 2, h / 2, Math.atan2(up.y, up.x) + Math.PI / 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(79,163,255,0.22)'; ctx.fill(); ctx.strokeStyle = BONE_EDGE; ctx.lineWidth = lw; ctx.stroke();
  }
  // pelvis
  const side = norm(sub(B.rHip, B.lHip)), halfW = Math.max(len(sub(B.rHip, B.lHip)) / 2, S * 0.17);
  const o = (base, s, u) => add(add(B.midHip, mul(side, s * halfW)), mul(up, u * S));
  smoothShape([o(0, -1.05, 0.02), o(0, -1.35, 0.2), o(0, -1.0, 0.33), o(0, -0.35, 0.22), o(0, 0.35, 0.22), o(0, 1.0, 0.33), o(0, 1.35, 0.2), o(0, 1.05, 0.02), o(0, 0.45, -0.12), o(0, -0.45, -0.12)], 'rgba(79,163,255,0.17)', BONE_EDGE, lw);
  for (const s of [-1, 1]) { const c = o(0, s * 1.0, 0); ctx.beginPath(); ctx.arc(c.x, c.y, S * 0.065, 0, Math.PI * 2); ctx.strokeStyle = 'rgba(168,210,255,0.6)'; ctx.stroke(); }
  // ribcage
  const rc = lerp(B.midSh, B.midHip, 0.26), rw = Math.max(len(sub(B.rSh, B.lSh)) * 0.42, S * 0.19), rh = S * 0.34;
  const ang = Math.atan2(up.y, up.x) + Math.PI / 2;
  ctx.beginPath(); ctx.ellipse(rc.x, rc.y, rw, rh, ang, 0, Math.PI * 2); ctx.fillStyle = 'rgba(79,163,255,0.08)'; ctx.fill(); ctx.strokeStyle = BONE_EDGE; ctx.lineWidth = lw; ctx.stroke();
  ctx.strokeStyle = 'rgba(168,210,255,0.55)';
  for (let i = 1; i <= 7; i++) {
    const f = -0.75 + i * 0.2, c = add(rc, mul(up, -f * rh)), w = rw * Math.sqrt(Math.max(0.05, 1 - f * f)) * 0.98;
    ctx.beginPath(); ctx.ellipse(c.x, c.y, w, S * 0.035, ang, Math.PI * 0.05, Math.PI * 0.95); ctx.stroke();
  }
  const st0 = add(rc, mul(up, rh * 0.85)), st1 = add(rc, mul(up, -rh * 0.25));
  ctx.lineWidth = S * 0.025; ctx.strokeStyle = 'rgba(120,185,255,0.35)'; ctx.beginPath(); ctx.moveTo(st0.x, st0.y); ctx.lineTo(st1.x, st1.y); ctx.stroke();
  // clavicles
  ctx.lineWidth = lw;
  const neck = add(B.midSh, mul(up, S * 0.04));
  for (const sh of [B.lSh, B.rSh]) longBone(neck, sh, [0.05, 0.035, 0.03, 0.035, 0.05], S);
  // skull
  const fdir = norm(sub(B.nose, B.earMid)); const hr = Math.max(len(sub(P(7), P(8))) * 0.6, S * 0.2);
  const cc = add(add(B.earMid, mul(fdir, -hr * 0.1)), mul(up, hr * 0.25));
  ctx.beginPath(); ctx.ellipse(cc.x, cc.y, hr * 0.92, hr * 1.02, ang, 0, Math.PI * 2); ctx.fillStyle = 'rgba(79,163,255,0.16)'; ctx.fill(); ctx.strokeStyle = BONE_EDGE; ctx.stroke();
  const jaw0 = add(B.earMid, mul(up, -hr * 0.25)), chin = add(add(B.nose, mul(up, -hr * 0.55)), mul(fdir, hr * 0.05));
  ctx.beginPath(); ctx.moveTo(jaw0.x, jaw0.y); ctx.quadraticCurveTo(jaw0.x - up.x * hr * 0.4, jaw0.y - up.y * hr * 0.4, chin.x, chin.y); ctx.stroke();
  const orb = add(add(B.earMid, mul(fdir, Math.min(hr * 0.75, len(sub(B.nose, B.earMid)) * 0.8))), mul(up, hr * 0.12));
  ctx.beginPath(); ctx.ellipse(orb.x, orb.y, hr * 0.16, hr * 0.13, 0, 0, Math.PI * 2); ctx.strokeStyle = 'rgba(168,210,255,0.7)'; ctx.stroke();
  // limbs
  for (const s of [[23, 25, 27, 29, 31, 11, 13, 15, 19], [24, 26, 28, 30, 32, 12, 14, 16, 20]]) {
    const [hip, kn, an, he, ft, sh, el, wr, ix] = s.map(P);
    const hipJ = add(hip, mul(up, S * 0.02));
    longBone(hipJ, kn, [0.14, 0.085, 0.075, 0.1, 0.16], S);           // femur
    const fib = mul(norm(perp(sub(an, kn))), S * 0.045);
    longBone(add(kn, fib), add(an, fib), [0.045, 0.03, 0.025, 0.03, 0.045], S, 0.8); // fibula
    longBone(kn, an, [0.145, 0.09, 0.062, 0.07, 0.095], S);          // tibia
    smoothShape([he, an, lerp(an, ft, 0.6), ft, lerp(he, ft, 0.55)], 'rgba(79,163,255,0.18)', BONE_EDGE, lw); // foot
    longBone(sh, el, [0.11, 0.07, 0.058, 0.065, 0.11], S);           // humerus
    const off = mul(norm(perp(sub(wr, el))), S * 0.025);
    longBone(add(el, off), add(wr, off), [0.06, 0.04, 0.034, 0.04, 0.06], S);  // radius
    longBone(sub(el, off), sub(wr, off), [0.07, 0.04, 0.03, 0.034, 0.045], S, 0.85); // ulna
    longBone(wr, lerp(wr, ix, 1.15), [0.07, 0.05, 0.045, 0.05, 0.04], S, 0.9); // hand
  }
  ctx.restore();
}
function muscle(a, b, t0, t1, off, w, S) {
  const u = norm(sub(b, a)), n = perp(u);
  const p0 = add(lerp(a, b, t0), mul(n, off * S)), p1 = add(lerp(a, b, t1), mul(n, off * S));
  const c = mid(p0, p1), half = len(sub(p1, p0)) / 2, ww = w * S / 2, an = Math.atan2(u.y, u.x);
  ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(an);
  ctx.beginPath(); ctx.moveTo(-half, 0);
  ctx.bezierCurveTo(-half * 0.4, -ww * 1.25, half * 0.4, -ww * 1.25, half, 0);
  ctx.bezierCurveTo(half * 0.4, ww * 1.25, -half * 0.4, ww * 1.25, -half, 0);
  ctx.closePath();
  const g = ctx.createLinearGradient(0, -ww, 0, ww);
  g.addColorStop(0, 'rgba(255,120,90,0.55)'); g.addColorStop(0.5, 'rgba(255,90,60,0.28)'); g.addColorStop(1, 'rgba(255,120,90,0.55)');
  ctx.fillStyle = g; ctx.fill(); ctx.strokeStyle = 'rgba(255,160,135,0.9)'; ctx.lineWidth = Math.max(1, S * 0.005); ctx.stroke();
  ctx.strokeStyle = 'rgba(255,200,185,0.28)'; ctx.lineWidth = 1;
  for (let k = -2; k <= 2; k++) { ctx.beginPath(); ctx.moveTo(-half * 0.85, 0); ctx.quadraticCurveTo(0, k * ww * 0.42, half * 0.85, 0); ctx.stroke(); }
  ctx.restore();
}
function drawMuscles(lm) {
  drawSkeleton(lm, 0.4);
  const B = body(lm), { P, S } = B;
  const frontal = A ? A.frontal : Math.abs(B.lSh.x - B.rSh.x) / S > 0.45;
  for (const s of [[23, 25, 27, 11, 13, 15], [24, 26, 28, 12, 14, 16]]) {
    const [hip, kn, an, sh, el, wr] = s.map(P);
    // anterior sign: perpendicular that points toward facing direction
    const sg = (a, b) => { const n = perp(norm(sub(b, a))); return frontal ? 0 : (Math.sign(n.x) === B.facing ? 1 : -1); };
    const th = sg(hip, kn), sn = sg(kn, an), ua = sg(sh, el);
    muscle(hip, kn, 0.12, 0.92, th * 0.06, 0.15, S);                       // quadriceps
    if (!frontal) muscle(hip, kn, 0.18, 0.9, -th * 0.07, 0.1, S);          // hamstrings
    if (!frontal) { const g = add(add(hip, mul(B.up, S * 0.08)), mul(perp(norm(sub(kn, hip))), -th * S * 0.1)); ctx.save(); ctx.beginPath(); ctx.ellipse(g.x, g.y, S * 0.13, S * 0.1, 0, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,90,60,0.32)'; ctx.fill(); ctx.strokeStyle = 'rgba(255,160,135,0.9)'; ctx.stroke(); ctx.restore(); } // glute
    muscle(kn, an, 0.06, 0.58, -sn * 0.05, frontal ? 0.08 : 0.12, S);     // gastrocnemius
    muscle(kn, an, 0.12, 0.75, sn * 0.035, 0.05, S);                       // tibialis anterior
    muscle(sh, el, -0.08, 0.38, 0, 0.16, S);                               // deltoid
    muscle(sh, el, 0.3, 0.92, ua * 0.04, 0.085, S);                        // biceps
    if (!frontal) muscle(sh, el, 0.25, 0.92, -ua * 0.045, 0.08, S);        // triceps
    muscle(el, wr, 0.05, 0.6, 0, 0.08, S);                                 // forearm flexors
  }
  if (frontal) { // pectorals + abdominals
    for (const sh of [B.lSh, B.rSh]) muscle(lerp(B.midSh, B.midHip, 0.12), sh, 0.1, 0.95, 0, 0.16, S);
    muscle(lerp(B.midSh, B.midHip, 0.3), lerp(B.midSh, B.midHip, 0.95), 0, 1, 0, 0.14, S);
  }
}
let oc = null;
function drawOutline(lm) {
  const B = body(lm), { P, S } = B, k = S;
  if (!oc) oc = document.createElement('canvas');
  oc.width = W; oc.height = H; const c = oc.getContext('2d');
  c.lineCap = 'round'; c.lineJoin = 'round';
  const capsule = (a, b, w) => { c.lineWidth = w * k; c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke(); };
  const segs = [[23, 25, 0.27], [25, 27, 0.19], [24, 26, 0.27], [26, 28, 0.19], [11, 13, 0.16], [13, 15, 0.13], [12, 14, 0.16], [14, 16, 0.13], [27, 31, 0.1], [28, 32, 0.1]];
  const torso = [add(B.lSh, mul(norm(sub(B.lSh, B.rSh)), S * 0.06)), add(B.rSh, mul(norm(sub(B.rSh, B.lSh)), S * 0.06)), B.rHip, B.lHip];
  const hc = add(B.earMid, mul(B.up, S * 0.08));
  for (const [col, extra, cut] of [['rgba(79,163,255,1)', 0.014, false], ['#000', 0, true]]) {
    c.globalCompositeOperation = cut ? 'destination-out' : 'source-over';
    c.strokeStyle = col; c.fillStyle = col;
    for (const [a, b, w] of segs) capsule(P(a), P(b), w + extra * 2);
    c.beginPath(); torso.forEach((p, i) => i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)); c.closePath();
    c.lineWidth = S * (0.12 + extra * 2); c.stroke(); c.fill();
    c.beginPath(); c.arc(hc.x, hc.y, S * (0.24 + extra), 0, Math.PI * 2); c.fill();
    capsule(B.midSh, add(B.midSh, mul(B.up, S * 0.18)), 0.14 + extra * 2);
  }
  ctx.save(); ctx.fillStyle = 'rgba(79,163,255,0.07)';
  ctx.drawImage(oc, 0, 0); ctx.restore();
  ctx.save(); ctx.strokeStyle = 'rgba(79,163,255,0.55)'; ctx.setLineDash([4, 5]); ctx.lineWidth = 1;
  for (const [a, b] of segs) { ctx.beginPath(); ctx.moveTo(P(a).x, P(a).y); ctx.lineTo(P(b).x, P(b).y); ctx.stroke(); }
  for (const j of [23, 24, 25, 26, 27, 28, 11, 12, 13, 14, 15, 16]) { const p = P(j); ctx.setLineDash([]); ctx.fillStyle = '#E7ECF2'; ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(2, S * 0.014), 0, Math.PI * 2); ctx.fill(); }
  ctx.restore();
}
function drawTriads(lm) {
  const { P, S } = body(lm), L = S * 0.16;
  const sets = [[23, 11], [24, 12], [25, 23], [26, 24], [27, 25], [28, 26], [11, 23], [12, 24], [13, 11], [14, 12]];
  ctx.save(); ctx.lineCap = 'round'; ctx.lineWidth = Math.max(1.5, S * 0.012);
  for (const [j, ref] of sets) {
    const c = P(j), u = norm(sub(P(ref), c)), n = perp(u), z = norm(add(mul(u, -0.55), mul(n, -0.8)));
    for (const [d, col, l] of [[n, '#FF3B30', L], [u, '#34C759', L], [z, '#3B82FF', L * 0.75]]) {
      const e = add(c, mul(d, l)); ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(e.x, e.y); ctx.stroke();
    }
    ctx.fillStyle = '#E7ECF2'; ctx.beginPath(); ctx.arc(c.x, c.y, Math.max(2, S * 0.014), 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}
function groundInfo(lm) {
  const { P, S } = body(lm);
  const feet = [27, 28, 29, 30, 31, 32].map(P);
  return { y: Math.max(...feet.map(p => p.y)) + S * 0.02, x: feet.reduce((s, p) => s + p.x, 0) / feet.length, S };
}
function drawGrid(lm) {
  const g = groundInfo(lm), S = g.S, rows = 9, cols = 12, depth = S * 0.5, span = S * 2.4;
  ctx.save(); ctx.strokeStyle = 'rgba(79,163,255,0.28)'; ctx.lineWidth = 1;
  const pt = (u, v) => { const z = v; const w = span * (0.62 + 0.38 * (z + 1) / 2); return { x: g.x + u * w / 2, y: g.y + z * depth * (0.55 + 0.45 * (z + 1) / 2) }; };
  for (let r = 0; r <= rows; r++) { const v = -1 + 2 * r / rows; ctx.globalAlpha = 0.3 + 0.7 * (r / rows); const a = pt(-1, v), b = pt(1, v); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
  for (let c = 0; c <= cols; c++) { const u = -1 + 2 * c / cols; ctx.globalAlpha = 0.6; const a = pt(u, -1), b = pt(u, 1); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
  ctx.restore();
}
function drawForce(lm, bw) {
  const g = groundInfo(lm), S = g.S, Lh = Math.max(0.2, bw) * S * 0.75;
  const a = { x: g.x, y: g.y }, b = { x: g.x, y: g.y - Lh };
  ctx.save(); ctx.strokeStyle = '#FF5A3C'; ctx.fillStyle = '#FF5A3C'; ctx.lineWidth = Math.max(3, S * 0.025); ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y + S * 0.06); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(b.x - S * 0.055, b.y + S * 0.1); ctx.lineTo(b.x + S * 0.055, b.y + S * 0.1); ctx.closePath(); ctx.fill();
  ctx.font = `600 ${Math.max(12, S * 0.075)}px Saira, sans-serif`; ctx.fillStyle = '#FFE1DA';
  const N = bw * (+$('#bw').value || 70) * G;
  ctx.fillText(`GRF ${bw.toFixed(2)}× BW · ${Math.round(N)} N`, b.x + S * 0.08, b.y + S * 0.12);
  ctx.restore();
}
function labelAngles(lm) {
  const { P, S } = body(lm);
  ctx.save(); ctx.font = `500 ${Math.max(11, S * 0.06)}px "JetBrains Mono", monospace`; ctx.textBaseline = 'middle';
  for (const j of JOINTS.filter(j => /knee|hip|elbow/.test(j.id))) {
    const c = P(j.b); if (lm[j.b].v < 0.5) continue;
    const a = angle3(lm[j.a], lm[j.b], lm[j.c]);
    const tx = c.x + S * 0.09, ty = c.y - S * 0.06, txt = `${Math.round(a)}°`, w = ctx.measureText(txt).width;
    ctx.fillStyle = 'rgba(6,8,11,0.72)'; ctx.fillRect(tx - 4, ty - S * 0.045, w + 8, S * 0.09);
    ctx.fillStyle = '#E7ECF2'; ctx.fillText(txt, tx, ty);
  }
  ctx.restore();
}

function render() {
  if (!media) return;
  render.lastT = mediaType === 'video' ? vid.currentTime : 0;
  ctx.clearRect(0, 0, W, H);
  try { ctx.drawImage(media, 0, 0, W, H); } catch { }
  ctx.fillStyle = 'rgba(6,10,16,0.32)'; ctx.fillRect(0, 0, W, H);
  const f = nearest(render.lastT);
  $('#hudT').textContent = render.lastT.toFixed(2);
  $('#hudF').textContent = f ? (f.i ?? 0) + 1 : 0;
  if (!f) return;
  const lm = f.lm;
  if ($('#tGrid').checked) drawGrid(lm);
  if (mode === 'skeleton') drawSkeleton(lm); else if (mode === 'muscle') drawMuscles(lm); else drawOutline(lm);
  if ($('#tTriad').checked && mode !== 'outline') drawTriads(lm);
  if ($('#tForce').checked) drawForce(lm, A && f.i != null && A.grf[f.i] != null && A.list.length > 2 ? A.grf[f.i] : 1);
  labelAngles(lm);
  updateReadout(f);
  if (mediaType === 'video' && !vid.paused && performance.now() - (render.lc || 0) > 250) { drawCharts(); render.lc = performance.now(); }
}
function updateReadout(f) {
  const rows = JOINTS.map(j => [j.name, `${fmt(angle3(f.wl[j.a], f.wl[j.b], f.wl[j.c]))}°`]);
  if (A && f.i != null) {
    rows.push(['Trunk lean', `${fmt(A.trunk[f.i])}°`]);
    if (A.list.length > 2) rows.push(['Hip speed', `${fmt(A.speed[f.i], 2)} m/s`], ['Ground force', `${fmt(A.grf[f.i], 2)}× BW`]);
  }
  $('#readout tBody, #readout tbody').innerHTML = rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
}

// ---------- events ----------
for (const id of ['#file', '#file2']) $(id).addEventListener('change', e => { $('#play').disabled = false; $('#scrub').disabled = false; loadFile(e.target.files[0]); e.target.value = ''; });
$('#sample').addEventListener('click', () => loadImage('img/sample-squat.jpg'));
document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
  mode = b.dataset.mode; document.querySelectorAll('.seg button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); dirty = true;
}));
['#tTriad', '#tGrid', '#tForce'].forEach(id => $(id).addEventListener('change', () => dirty = true));
$('#bw').addEventListener('input', () => { updatePanel(); dirty = true; });
$('#play').addEventListener('click', () => { if (mediaType !== 'video') return; vid.paused ? vid.play() : vid.pause(); });
vid.addEventListener('play', () => $('#playIcon').setAttribute('d', 'M4 2.5h3v11H4zM9 2.5h3v11H9z'));
vid.addEventListener('pause', () => { $('#playIcon').setAttribute('d', 'M4 2.5v11l9-5.5z'); drawCharts(); });
$('#scrub').addEventListener('input', e => { if (mediaType === 'video' && vid.duration) { vid.currentTime = e.target.value / 1000 * vid.duration; } });
$('#rate').addEventListener('change', e => vid.playbackRate = +e.target.value);
window.addEventListener('resize', () => drawCharts());
// drag and drop
const stage = $('#stage');
stage.addEventListener('dragover', e => e.preventDefault());
stage.addEventListener('drop', e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

// test hook (used for offline rendering checks only)
window.__velo = {
  async injectImage(src, lm, wl) {
    const img = new Image(); img.src = src; await img.decode();
    resetData(); mediaType = 'image'; media = img; sizeCanvas(img.naturalWidth, img.naturalHeight); showUI();
    frames.set(0, { t: 0, lm, wl }); analyze(); dirty = true;
  },
  setMode(m) { document.querySelector(`.seg [data-mode="${m}"]`).click(); },
};
updatePanel();
requestAnimationFrame(tick);
