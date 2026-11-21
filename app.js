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
    loader(0);
    const vision = await import(`${TV}/vision_bundle.mjs`); loader(1);
    const fs = await vision.FilesetResolver.forVisionTasks(`${TV}/wasm`); loader(2);
    const opts = d => ({ baseOptions: { modelAssetPath: MODEL, delegate: d }, runningMode: 'VIDEO', numPoses: 1, minPoseDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    try { lmk = await vision.PoseLandmarker.createFromOptions(fs, opts('GPU')); }
    catch { lmk = await vision.PoseLandmarker.createFromOptions(fs, opts('CPU')); }
    lmkMode = 'VIDEO'; loader(3);
    setTimeout(() => loader(-1), 500);
    return lmk;
  })().catch(e => { loading = null; loader(-1); status('Could not load the pose model. Check your connection and try again. (' + (e && e.message || e) + ')', true); throw e; });
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

const HEIC = /\.(heic|heif)$/i, IMG = /\.(jpe?g|png|webp|gif|bmp|avif)$/i;
async function heicToJpeg(file) {
  status('Converting your iPhone photo (HEIC)…');
  if (!window.heic2any) await new Promise((res, rej) => { const sc = document.createElement('script'); sc.src = 'https://cdn.jsdelivr.net/npm/heic2any@0.0.4/dist/heic2any.min.js'; sc.onload = res; sc.onerror = rej; document.head.appendChild(sc); });
  let blob = await window.heic2any({ blob: file, toType: 'image/jpeg', quality: 0.92 }); if (Array.isArray(blob)) blob = blob[0];
  status(''); return blob;
}
async function loadFile(file) {
  if (!file) return;
  resetData();
  const isHeic = /heic|heif/i.test(file.type) || HEIC.test(file.name || '');
  if (isHeic) {
    try { file = await heicToJpeg(file); }
    catch (e) { status('This iPhone photo (HEIC) could not be opened. On your iPhone go to Settings > Camera > Formats > Most Compatible, or send it as a JPG.', true); return; }
  }
  const url = URL.createObjectURL(file);
  if (isHeic || (file.type || '').startsWith('image/') || IMG.test(file.name || '')) return loadImage(url);
  mediaType = 'video'; media = vid;
  vid.src = url; vid.loop = true; vid.playbackRate = +$('#rate').value;
  await new Promise((res, rej) => { vid.onloadeddata = res; vid.onerror = () => rej(new Error('This video format is not supported by your browser.')); }).catch(e => { status(e.message, true); throw e; });
  if (!vid.videoWidth) { status('This video has no picture Velo can read. Try an MP4, or on iPhone set Camera > Formats > Most Compatible.', true); return; }
  sizeCanvas(vid.videoWidth, vid.videoHeight); showUI(); dirty = true; fx.revealAt = 0;
  try { await getLandmarker(); await setMode('VIDEO'); } catch { return; }
  fx.revealAt = performance.now();
  status('Analyzing. Let it play through once for the full report; slow speed samples more frames.');
  setTimeout(() => status(''), 5000);
  vid.play().catch(() => {});
}
async function loadImage(url) {
  resetData();
  const img = new Image(); img.src = url;
  const okImg = await img.decode().then(() => true, () => false);
  if (!okImg || !img.naturalWidth) { status('Velo could not open that file. Try a JPG or PNG photo, or an MP4 or MOV video.', true); return; }
  status(''); mediaType = 'image'; media = img; vid.pause(); fx.revealAt = 0;
  sizeCanvas(img.naturalWidth, img.naturalHeight); showUI(); dirty = true;
  $('#play').disabled = true; $('#scrub').disabled = true;
  try { await getLandmarker(); await setMode('IMAGE'); } catch { render(); return; }
  const r = lmk.detect(img);
  if (r.landmarks && r.landmarks[0]) { frames.set(0, { t: 0, lm: copyLm(r.landmarks[0]), wl: copyLm(r.worldLandmarks[0]) }); fx.revealAt = performance.now(); }
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
  if (fx.revealAt && performance.now() - fx.revealAt < REVEAL + 100) dirty = true;
  if (dirty) { render(); dirty = false; }
  fxFrame();
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
  const frontal = shW / (torso || 1) > 0.62;
  A = { list, t, ang, vel, trunk, speed, grf, rom, primary, reps, frontal, scale };
  A.sport = detectSport();
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
function squatChecks() {
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
      : kMin <= 95 ? { s: 'ok', h: 'Depth', m: `${fmt(kMin)}°`, p: 'At parallel: thighs level with the floor. Go a touch deeper for full depth.' }
      : { s: kMin < 108 ? 'warn' : 'bad', h: 'Depth', m: `${fmt(kMin)}°`, p: `Hips stop above the knees, about ${fmt(kMin - 90)}° short of parallel. Sit back and down a little further.` });
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

// ---------- sport recognition + sport-specific checks ----------
const SPORTS = { golf: 'Golf swing', squat: 'Squat / lift', sprint: 'Sprint / run', jump: 'Jump', throw: 'Throw / pitch', general: 'General movement' };
const sportSel = () => ($('#sport') || {}).value || 'auto';
const handed = () => ($('#hand') || {}).value || 'right';
const clamp01 = x => Math.max(0, Math.min(1, x));
const d2 = (lm, a, b) => Math.hypot((lm[a].x - lm[b].x) * W, (lm[a].y - lm[b].y) * H);
// Reads each frame's pose and scores how much it looks like each sport. Returns {id, conf, scores}.
function detectSport() {
  if (!A) return { id: 'general', conf: 0, scores: {} };
  const L = A.list, n = L.length, sc = { golf: 0, squat: 0, sprint: 0, jump: 0, throw: 0, general: 0.15 };
  const fr = L.map(f => { const lm = f.lm, sw = d2(lm, 11, 12) || 1, shY = (lm[11].y + lm[12].y) / 2, hipY = (lm[23].y + lm[24].y) / 2, legLen = Math.abs((lm[27].y + lm[28].y) / 2 - hipY) || 0.3;
    return { together: d2(lm, 15, 16) / sw, wristUp: Math.max(shY - lm[15].y, shY - lm[16].y) / legLen, oneUp: Math.abs(lm[15].y - lm[16].y) / legLen, stride: Math.abs(lm[27].x - lm[28].x) * W / sw,
      kneeLift: Math.abs(lm[25].y - lm[26].y) / legLen, handsLow: ((lm[15].y + lm[16].y) / 2 - shY) / legLen, hipY, feetY: (lm[27].y + lm[28].y) / 2, legLen }; });
  const med = k => median(fr.map(f => f[k])), mx = k => Math.max(...fr.map(f => f[k]).filter(Number.isFinite));
  const trunk = median(A.trunk), kneeMin = Math.min(...A.ang.kneeL.map((v, i) => (v + A.ang.kneeR[i]) / 2).filter(Number.isFinite));
  // golf: hands joined through the motion, hinged torso, feet planted shoulder width
  sc.golf += 1.4 * clamp01((0.7 - med('together')) / 0.5) + 0.8 * clamp01(1 - Math.abs(trunk - 35) / 25) + 0.4 * clamp01(1 - Math.abs(med('stride') - 1.1) / 0.8);
  if (n > 2) sc.golf += 0.8 * clamp01(mx('wristUp') / 0.6) * clamp01((0.7 - med('together')) / 0.4);
  // squat / lift: deep knee bend, hands at shoulders or on a wide bar, feet level
  sc.squat += 1.2 * clamp01((150 - kneeMin) / 60) + 0.6 * clamp01((0.25 - Math.abs(med('handsLow'))) / 0.25) * clamp01((med('together') - 0.8) / 0.8) + 0.4 * clamp01((0.15 - med('kneeLift')) / 0.15);
  if (n > 2) sc.squat += 0.6 * clamp01(((A.rom.kneeL + A.rom.kneeR) / 2 - 30) / 50);
  // sprint: one knee driven high, long stride, arms apart, forward lean
  sc.sprint += 1.1 * clamp01(mx('kneeLift') / 0.45) + 0.6 * clamp01((mx('stride') - 1.4) / 1.5) + 0.5 * clamp01((med('together') - 0.9) / 1) + 0.3 * clamp01((trunk - 5) / 20);
  // jump: hips travel up and both feet leave together, arms swing overhead
  if (n > 2) { const hy = fr.map(f => f.hipY), fy = fr.map(f => f.feetY), ll = med('legLen'); sc.jump += 1.4 * clamp01(((Math.max(...hy) - Math.min(...hy)) / ll - 0.25) / 0.4) * clamp01(((Math.max(...fy) - Math.min(...fy)) / ll - 0.1) / 0.3) + 0.4 * clamp01(mx('wristUp') / 0.8) * clamp01((0.15 - med('kneeLift')) / 0.15); }
  // throw / pitch: one arm high while the other is low, wide stride, hands apart
  sc.throw += 1.0 * clamp01((mx('oneUp') - 0.5) / 0.6) * clamp01((med('together') - 1) / 1) + 0.5 * clamp01((mx('stride') - 1.5) / 1.5);
  const ent = Object.entries(sc).sort((a, b) => b[1] - a[1]), tot = ent.reduce((a, [, v]) => a + Math.exp(v * 2.2), 0);
  return { id: ent[0][0], conf: Math.exp(ent[0][1] * 2.2) / tot, scores: Object.fromEntries(ent.map(([k, v]) => [k, Math.exp(v * 2.2) / tot])) };
}
function currentSport() { const s = sportSel(); if (s !== 'auto') return { id: s, conf: 1, manual: true }; return A ? (A.sport || (A.sport = detectSport())) : { id: 'general', conf: 0 }; }
function techniqueChecks() {
  if (!A) return [];
  const sp = currentSport().id;
  return sp === 'golf' ? golfChecks() : sp === 'sprint' ? sprintChecks() : sp === 'jump' ? jumpChecks() : sp === 'throw' ? throwChecks() : squatChecks();
}
const chk = (s, h, m, p) => ({ s, h, m, p });
const rate = (v, ok, warn, hi) => hi ? (v >= ok ? 'ok' : v >= warn ? 'warn' : 'bad') : (v <= ok ? 'ok' : v <= warn ? 'warn' : 'bad');
// yaw of a left/right segment in the floor plane, from world landmarks (degrees)
const yaw = (wl, a, b) => Math.atan2(wl[a].z - wl[b].z, wl[a].x - wl[b].x) * 180 / Math.PI;
const angDiff = (a, b) => { let d = Math.abs(a - b) % 360; return d > 180 ? 360 - d : d; };
function golfPhases() {
  const L = A.list, lead = handed() === 'right' ? 15 : 16, wy = L.map(f => f.lm[lead].y), n = L.length;
  const a0 = wy[0]; let start = 0; for (let i = 1; i < n; i++) if (a0 - wy[i] > 0.04) { start = Math.max(0, i - 1); break; }
  let top = start; for (let i = start; i < n; i++) if (wy[i] < wy[top]) top = i;
  let imp = top; for (let i = top; i < n && L[i].t - L[top].t < 1.2; i++) if (wy[i] > wy[imp]) imp = i;
  return { address: 0, start, top, impact: imp, finish: n - 1 };
}
function golfChecks() {
  const out = [], L = A.list, single = L.length < 3, R = handed() === 'right';
  const LS = R ? 11 : 12, LE = R ? 13 : 14, LW = R ? 15 : 16, LH = R ? 23 : 24, TH = R ? 24 : 23, LA = R ? 27 : 28, TK = R ? 26 : 25;
  const f0 = L[0], tilt0 = A.trunk[0], knee0 = (A.ang.kneeL[0] + A.ang.kneeR[0]) / 2;
  out.push(chk(Number.isFinite(tilt0) && tilt0 >= 25 && tilt0 <= 50 ? 'ok' : tilt0 >= 18 && tilt0 <= 58 ? 'warn' : 'bad', 'Address: spine tilt', `${fmt(tilt0)}°`,
    tilt0 < 25 ? 'Standing too tall. Hinge from the hips until the arms hang under the shoulders (about 30–45°).' : tilt0 > 50 ? 'Bent over too far. Stand a little taller so you can turn freely.' : 'Athletic hinge from the hips, a good base to turn around.'));
  out.push(chk(knee0 >= 145 && knee0 <= 168 ? 'ok' : knee0 >= 135 && knee0 <= 175 ? 'warn' : 'bad', 'Address: knee flex', `${fmt(knee0)}°`,
    knee0 > 168 ? 'Knees almost locked. Add a soft flex so the legs can support the turn.' : knee0 < 145 ? 'Too much knee bend, almost sitting. Straighten slightly and hinge more from the hips.' : 'Soft, athletic knee flex.'));
  const sh = mid(px(f0.lm, 11), px(f0.lm, 12)), hands = mid(px(f0.lm, 15), px(f0.lm, 16)), torso = len(sub(sh, mid(px(f0.lm, 23), px(f0.lm, 24)))) || 1;
  if (single) {
    const tog = d2(f0.lm, 15, 16) / (d2(f0.lm, 11, 12) || 1);
    out.push(chk(tog < 0.5 ? 'ok' : 'warn', 'Grip: hands together', tog < 0.5 ? 'joined' : 'apart', tog < 0.5 ? 'Both hands work as one unit on the club.' : 'Hands look separated. Both hands should be together on the grip.'));
    const stance = d2(f0.lm, 27, 28) / (d2(f0.lm, 11, 12) || 1);
    if (!A.frontal) out.push(chk('idle', 'Stance width', 'side view', 'Film face-on to measure stance width.')); else out.push(chk(stance >= 0.8 && stance <= 1.6 ? 'ok' : 'warn', 'Stance width', `${stance.toFixed(2)}× shoulders`, stance < 0.8 ? 'Feet are narrow. Widen to about shoulder width for a stable turn.' : stance > 1.6 ? 'Very wide stance. It can block the hip turn; bring the feet in a little.' : 'Feet about shoulder width, a stable base.'));
    out.push(chk('idle', 'Swing checks', 'need video', 'A photo shows one position. Upload a swing video to check turn, head stability, lead arm, tempo, sway and finish.'));
    return out;
  }
  const P = golfPhases(), ft = L[P.top], fi = L[P.impact], ff = L[P.finish];
  // lead arm at the top
  const arm = angle3(ft.wl[LS], ft.wl[LE], ft.wl[LW]);
  out.push(chk(rate(arm, 160, 145, true), 'Top: lead arm', `${fmt(arm)}°`, arm >= 160 ? 'Lead arm stays long and straight, which keeps the swing wide.' : 'Lead elbow folds at the top. Keep the lead arm extended for width and consistency.'));
  // shoulder and hip turn (world-coordinate estimate)
  const shT = angDiff(yaw(ft.wl, 11, 12), yaw(f0.wl, 11, 12)), hpT = angDiff(yaw(ft.wl, 23, 24), yaw(f0.wl, 23, 24)), xf = shT - hpT;
  out.push(chk(rate(shT, 80, 60, true), 'Top: shoulder turn', `${fmt(shT)}°`, shT >= 80 ? 'Full shoulder turn, close to 90°.' : 'Short shoulder turn. Turn your back to the target for more power. (Estimate from one camera.)'));
  out.push(chk(xf >= 25 ? 'ok' : xf >= 12 ? 'warn' : 'bad', 'Top: X-factor (shoulders vs hips)', `${fmt(xf)}°`, xf >= 25 ? `Shoulders turn ${fmt(xf)}° more than the hips (${fmt(hpT)}°), storing energy for the downswing.` : `Hips turn almost as much as the shoulders (${fmt(hpT)}°). Resist with the lower body to build separation.`));
  // head stability address -> impact
  let hd = 0; for (let i = 0; i <= P.impact; i++) hd = Math.max(hd, len(sub(px(L[i].lm, 0), px(f0.lm, 0))) / torso);
  out.push(chk(rate(hd, 0.12, 0.22), 'Head stability', `${fmt(hd * 100)}% torso`, hd <= 0.12 ? 'Head stays quiet through impact.' : 'Head moves a lot before impact. Keep your eyes on the ball and the head centered.'));
  // hip sway on the backswing
  const hipW = d2(f0.lm, 23, 24) || 1, sway = Math.abs(mid(px(ft.lm, 23), px(ft.lm, 24)).x - mid(px(f0.lm, 23), px(f0.lm, 24)).x) / hipW;
  out.push(chk(A.frontal ? rate(sway, 0.25, 0.45) : 'idle', 'Backswing: hip sway', A.frontal ? `${fmt(sway * 100)}% hip width` : 'side view', !A.frontal ? 'Film face-on to measure sway.' : sway <= 0.25 ? 'Hips rotate in place instead of sliding.' : 'Hips slide away from the target. Turn around the trail hip instead of swaying.'));
  // trail knee holds its flex
  const tk0 = angle3(f0.wl[TH], f0.wl[TK], f0.wl[TK + 2]), tkT = angle3(ft.wl[TH], ft.wl[TK], ft.wl[TK + 2]);
  out.push(chk(rate(tkT - tk0, 8, 15), 'Backswing: trail knee', `${tkT - tk0 >= 0 ? '+' : ''}${fmt(tkT - tk0)}°`, tkT - tk0 <= 8 ? 'Trail knee keeps its flex, giving a stable post to turn against.' : 'Trail leg straightens on the backswing. Keep some flex to stay grounded.'));
  // spine angle kept through impact
  const sd = Math.abs(A.trunk[P.impact] - tilt0);
  out.push(chk(rate(sd, 8, 15), 'Impact: spine angle held', `${fmt(sd)}° change`, sd <= 8 ? 'Posture stays the same from address to impact.' : 'Early extension: the torso stands up before impact. Keep your chest over the ball.'));
  // tempo
  const bs = L[P.top].t - L[P.start].t, ds = L[P.impact].t - L[P.top].t, ratio = ds > 0 ? bs / ds : NaN;
  out.push(chk(Number.isFinite(ratio) ? (ratio >= 2.5 && ratio <= 3.6 ? 'ok' : ratio >= 1.8 && ratio <= 4.5 ? 'warn' : 'bad') : 'idle', 'Tempo (back : down)', Number.isFinite(ratio) ? `${ratio.toFixed(1)} : 1` : '—',
    !Number.isFinite(ratio) ? 'Need the full swing in the clip to time it.' : ratio < 2.5 ? 'Backswing is rushed compared with the downswing. Tour players average about 3 : 1.' : ratio > 3.6 ? 'Very slow backswing for the downswing speed. Aim for about 3 : 1.' : 'Smooth tempo, close to the 3 : 1 tour average.'));
  // balanced finish over the lead side
  const over = Math.abs(mid(px(ff.lm, 23), px(ff.lm, 24)).x - px(ff.lm, LA).x) / hipW, up = A.trunk[P.finish];
  out.push(chk(over < 0.6 && up < 30 ? 'ok' : over < 1 ? 'warn' : 'bad', 'Finish: balance', over < 0.6 ? 'on lead side' : 'hanging back', over < 0.6 && up < 30 ? 'Weight finishes over the lead foot, tall and balanced.' : 'Weight stays on the trail side at the finish. Rotate through and hold a tall finish over the lead foot.'));
  // hand speed (info)
  const hs = Math.max(...L.slice(P.top, P.impact + 1).map((f, k, a) => k ? len(sub(px(f.lm, LW), px(a[k - 1].lm, LW))) * A.scale / Math.max(1e-3, f.t - a[k - 1].t) : 0));
  if (Number.isFinite(hs) && hs > 0) out.push(chk('idle', 'Peak hand speed', `${hs.toFixed(1)} m/s`, `Fastest lead-hand speed on the downswing (≈ ${(hs * 2.237).toFixed(0)} mph).`));
  return out;
}
function sprintChecks() {
  const out = [], L = A.list;
  const lean = median(A.trunk); out.push(chk(lean >= 5 && lean <= 20 ? 'ok' : lean <= 30 ? 'warn' : 'bad', 'Body lean', `${fmt(lean)}°`, lean < 5 ? 'Very upright. A slight forward lean from the ankles helps acceleration.' : lean > 20 ? 'Leaning a lot. Fine in the first steps of a start, but run taller once up to speed.' : 'Good forward lean from the ankles.'));
  const drive = Math.max(...L.map(f => Math.max(180 - angle3(f.wl[11], f.wl[23], f.wl[25]), 180 - angle3(f.wl[12], f.wl[24], f.wl[26]))));
  out.push(chk(rate(drive, 70, 50, true), 'Knee drive', `${fmt(drive)}° hip flexion`, drive >= 70 ? 'High knee drive.' : 'Low knee lift. Drive the knee up toward hip height.'));
  const elbow = median(L.map(f => (angle3(f.wl[11], f.wl[13], f.wl[15]) + angle3(f.wl[12], f.wl[14], f.wl[16])) / 2));
  out.push(chk(elbow >= 70 && elbow <= 110 ? 'ok' : 'warn', 'Arm angle', `${fmt(elbow)}°`, elbow >= 70 && elbow <= 110 ? 'Elbows near 90°, a powerful arm swing.' : 'Arms too straight or too bent. Keep the elbows near 90° and swing from the shoulder.'));
  if (L.length < 3) out.push(chk('idle', 'Stride checks', 'need video', 'Upload a running clip to check stride and ground contact.'));
  return out;
}
function jumpChecks() {
  const out = [], knee = A.ang.kneeL.map((v, i) => (v + A.ang.kneeR[i]) / 2), dip = Math.min(...knee.filter(Number.isFinite));
  out.push(chk(dip >= 90 && dip <= 130 ? 'ok' : 'warn', 'Countermovement depth', `${fmt(dip)}°`, dip < 90 ? 'Very deep dip. A quicker, shallower dip (about 100–120°) is usually more explosive.' : dip > 130 ? 'Shallow dip. Load the hips and knees a bit more.' : 'Good loading depth.'));
  const g = Math.max(...A.grf.filter(Number.isFinite)); if (Number.isFinite(g)) out.push(chk('idle', 'Peak ground force', `${g.toFixed(2)}× BW`, 'Estimated from how fast the hips accelerate.'));
  out.push(...squatChecks().filter(c => /Knee tracking|symmetry/.test(c.h)));
  return out;
}
function throwChecks() {
  const out = [], R = handed() === 'right', S = R ? 12 : 11, E = R ? 14 : 13, Wr = R ? 16 : 15;
  const elbowUp = Math.max(...A.list.map(f => (f.lm[S].y - f.lm[E].y) * H / (d2(f.lm, 11, 12) || 1)));
  out.push(chk(elbowUp > -0.1 ? 'ok' : 'warn', 'Elbow height', elbowUp > -0.1 ? 'at shoulder' : 'low', elbowUp > -0.1 ? 'Throwing elbow reaches shoulder height.' : 'Elbow drops below the shoulder, which stresses the arm. Get it up to shoulder level.'));
  const stride = Math.max(...A.list.map(f => d2(f.lm, 27, 28) / (d2(f.lm, 11, 12) || 1)));
  out.push(chk(rate(stride, 2.2, 1.5, true), 'Stride length', `${stride.toFixed(1)}× shoulders`, stride >= 2.2 ? 'Long stride toward the target.' : 'Short stride. Step further toward the target to use the legs.'));
  if (A.list.length > 2) { const f = A.list[A.list.length >> 1], sep = angDiff(yaw(f.wl, 11, 12), yaw(f.wl, 23, 24)); out.push(chk(rate(sep, 30, 15, true), 'Hip-shoulder separation', `${fmt(sep)}°`, sep >= 30 ? 'Hips open before the shoulders, a strong kinetic chain.' : 'Hips and shoulders turn together. Lead with the hips.')); }
  return out;
}

// ---------- panel ----------
function updatePanel() {
  const single = A && A.list.length < 3;
  countUp('#kReps', A && !single ? String(A.reps) : '—');
  $('#kRepsJ').textContent = A && !single ? A.primary.name.toLowerCase() : 'primary joint';
  countUp('#kSpeed', A && !single ? fmt(Math.max(...A.speed.filter(Number.isFinite)), 2) : '—');
  const gmax = A && !single ? Math.max(...A.grf.filter(Number.isFinite)) : NaN;
  countUp('#kForce', Number.isFinite(gmax) ? fmt(gmax, 2) + '×' : '—');
  $('#kForceN').textContent = Number.isFinite(gmax) ? `${fmt(gmax * (+$('#bw').value || 70) * G)} N` : '× body weight';
  countUp('#kRom', A && !single ? fmt(A.rom[A.primary.id]) + '°' : '—');
  $('#kRomJ').textContent = A && !single ? A.primary.name.toLowerCase() : 'degrees';
  $('#viewNote').textContent = !A ? 'Upload to analyze' : (A.frontal ? 'Front view detected' : 'Side view detected') + ` · ${A.list.length} frame${A.list.length === 1 ? '' : 's'}`;
  const sp = currentSport(); const sb = $('#sportBadge');
  if (sb) { sb.hidden = !A; if (A) { sb.querySelector('b').textContent = SPORTS[sp.id]; sb.querySelector('em').textContent = sp.manual ? 'set by you' : Math.round(sp.conf * 100) + '% sure'; sb.classList.toggle('low', !sp.manual && sp.conf < 0.5); } }
  $('#handWrap').hidden = !['golf', 'throw'].includes(sp.id);
  $('#hudView').textContent = !A ? '—' : A.frontal ? 'Front view' : 'Side view';
  const checks = techniqueChecks();
  $('#checks').innerHTML = checks.length ? checks.map(c => `<li class="check ${c.s}"><b>${c.h}<em>${c.m}</em></b><p>${c.p}</p></li>`).join('')
    : '<li class="check idle"><b>Waiting for movement</b><p>Checks appear once Velo has seen the body move.</p></li>';
  // chips
  if (single) { const a = j => fmt(angle3(A.list[0].wl[j.a], A.list[0].wl[j.b], A.list[0].wl[j.c])) + '°'; countUp('#kRom', a(JOINTS[0])); $('#kRomJ').textContent = 'left knee angle'; }
  $('#angleChips').innerHTML = JOINTS.filter(j => !A || A.rom[j.id] > 0 || single).map(j => `<button class="chip" data-j="${j.id}" aria-pressed="${shownJoints.has(j.id)}"><i style="background:${j.color}"></i>${j.name}</button>`).join('');
  drawCharts();
}
$('#angleChips').addEventListener('click', e => {
  const b = e.target.closest('.chip'); if (!b) return;
  const id = b.dataset.j; shownJoints.has(id) ? shownJoints.delete(id) : shownJoints.add(id);
  b.setAttribute('aria-pressed', shownJoints.has(id)); drawCharts();
});

function chart(cv, series, opt = {}) {
  const dpr = Math.min(2, window.devicePixelRatio || 1), w = Math.max(200, cv.clientWidth || cv.parentElement.clientWidth || 300), h = +(cv.dataset.h || (cv.dataset.h = cv.getAttribute('height')));
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px'; }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  c.fillStyle = 'rgba(4,8,14,.55)'; c.fillRect(0, 0, w, h);
  const pl = 36, pr = opt.right ? 36 : 8, pt = 8, pb = 20;
  c.font = '11px "IBM Plex Sans", sans-serif'; c.fillStyle = '#8B97A6';
  const has = series.some(s => s.data && s.data.some(Number.isFinite));
  if (A && A.list.length >= 3 && !has) { c.fillText(series.length ? 'Not enough clear frames for these joints yet.' : 'Pick a joint above to chart it.', pl, h / 2); return; }
  if (!A || A.list.length < 3) { c.fillText(mediaType === 'image' ? 'Upload a video to see this over time.' : 'Play the video to collect frames.', pl, h / 2); return; }
  const t = A.t, t0 = t[0], t1 = t[t.length - 1] || 1;
  const X = x => pl + (x - t0) / ((t1 - t0) || 1) * (w - pl - pr);
  const axis = (vals, side) => {
    const v = vals.flat().filter(Number.isFinite); let lo = v.length ? Math.min(...v) : 0, hi = v.length ? Math.max(...v) : 1;
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
    const ax = s.right ? right : left; c.save(); c.strokeStyle = s.color; c.lineWidth = 2; c.shadowColor = s.color; c.shadowBlur = 8; c.beginPath(); let pen = false;
    s.data.forEach((v, i) => { if (!Number.isFinite(v)) { pen = false; return; } const x = X(t[i]), y = ax.Y(v); pen ? c.lineTo(x, y) : c.moveTo(x, y); pen = true; });
    c.stroke(); c.restore();
  }
  const now = mediaType === 'video' ? vid.currentTime : 0;
  if (now >= t0 && now <= t1) { c.save(); c.strokeStyle = '#22E6FF'; c.shadowColor = '#22E6FF'; c.shadowBlur = 10; c.lineWidth = 1.5; c.beginPath(); c.moveTo(X(now), pt); c.lineTo(X(now), h - pb); c.stroke(); c.restore();
    for (const s of series) { const ax = s.right ? right : left, i = A.t.findIndex(x => x >= now); const v = i >= 0 ? s.data[i] : NaN; if (!Number.isFinite(v)) continue; c.fillStyle = s.color; c.beginPath(); c.arc(X(now), ax.Y(v), 3.5, 0, Math.PI * 2); c.fill(); } }
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
  const frontal = A ? A.frontal : Math.abs(B.lSh.x - B.rSh.x) / S > 0.62;
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
  fx.pose = f; trackTrails(lm);
  const rv = revealY(); ctx.save(); if (rv < H) { ctx.beginPath(); ctx.rect(0, 0, W, rv); ctx.clip(); }
  if ($('#tGrid').checked) drawGrid(lm);
  if (mode === 'skeleton') drawSkeleton(lm); else if (mode === 'muscle') drawMuscles(lm); else drawOutline(lm);
  if ($('#tTriad').checked && mode !== 'outline') drawTriads(lm);
  if ($('#tForce').checked) drawForce(lm, A && f.i != null && A.grf[f.i] != null && A.list.length > 2 ? A.grf[f.i] : 1);
  labelAngles(lm);
  ctx.restore();
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

// ---------- live analysis FX (overlay, telemetry, hero) ----------
const REVEAL = 1400;
const fx = { pose: null, revealAt: 0, trails: {}, stream: [], lastStream: 0, fpsN: 0, fpsT: performance.now(), fps: 0, g: {} };
const fxc = $('#fx'), fctx = fxc.getContext('2d');
const TRAIL_IDS = [15, 16, 25, 26, 27, 28];
const NEON = '#22E6FF', MAG = '#FF3DA6', LIME = '#B6FF3B';
const ease = k => 1 - Math.pow(1 - Math.min(1, Math.max(0, k)), 3);
function revealY() { if (!fx.revealAt) return H; const k = (performance.now() - fx.revealAt) / REVEAL; return k >= 1 ? H : ease(k) * H; }
function trackTrails(lm) {
  for (const i of TRAIL_IDS) {
    if (lm[i].v < 0.5) continue; const T = fx.trails[i] || (fx.trails[i] = []);
    const p = { x: lm[i].x * W, y: lm[i].y * H }, last = T[T.length - 1];
    if (!last || Math.hypot(p.x - last.x, p.y - last.y) > 0.6) T.push(p); if (T.length > 34) T.shift();
  }
}
function loader(step) {
  const L = $('#loader'); if (step < 0) { L.hidden = true; return; } L.hidden = false;
  [...$('#ldSteps').children].forEach((li, i) => li.className = i < step ? 'done' : i === step ? 'on' : '');
  animatePct([12, 48, 82, 100][step]);
}
function animatePct(to) { const el = $('#ldPct'), from = parseInt(el.textContent) || 0, t0 = performance.now(); const f = () => { const k = Math.min(1, (performance.now() - t0) / 650); el.textContent = Math.round(from + (to - from) * ease(k)) + '%'; if (k < 1) requestAnimationFrame(f); }; f(); }
function countUp(sel, txt) {
  const el = $(sel), m = /^(-?\d+(?:\.\d+)?)(.*)$/.exec(txt);
  if (!m) { el.textContent = txt; el.dataset.v = ''; return; }
  const to = +m[1], dec = (m[1].split('.')[1] || '').length, suf = m[2], prev = parseFloat(el.dataset.v || '0');
  if (el.dataset.v !== '' && prev === to && el.textContent === txt) return;
  el.dataset.v = to; el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop');
  const t0 = performance.now(); const f = () => { const k = Math.min(1, (performance.now() - t0) / 800); el.textContent = (prev + (to - prev) * ease(k)).toFixed(dec) + suf; if (k < 1) requestAnimationFrame(f); }; f();
}
function glowLine(c, pts, color, w, alpha) { if (pts.length < 2) return; c.save(); c.globalAlpha = alpha; c.strokeStyle = color; c.lineWidth = w; c.shadowColor = color; c.shadowBlur = 12; c.lineCap = 'round'; c.lineJoin = 'round'; c.beginPath(); pts.forEach((p, i) => i ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y)); c.stroke(); c.restore(); }

function fxFrame() {
  const now = performance.now();
  fx.fpsN++; if (now - fx.fpsT > 500) { fx.fps = Math.round(fx.fpsN * 1000 / (now - fx.fpsT)); fx.fpsN = 0; fx.fpsT = now; const e = $('#fps'); if (e) e.textContent = media && fx.pose ? fx.fps : 0; }
  const st = $('#stage').getBoundingClientRect(), r = view.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.round(r.width), ch = Math.round(r.height);
  if (fxc.width !== cw * dpr || fxc.height !== ch * dpr) { fxc.width = cw * dpr; fxc.height = ch * dpr; }
  Object.assign(fxc.style, { left: (r.left - st.left) + 'px', top: (r.top - st.top) + 'px', width: cw + 'px', height: ch + 'px' });
  const c = fctx; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, cw, ch);
  if (!media || !W) { drawGauges(null); draw3D(null); drawScore(); return; }
  const k = cw / W, t = now / 1000, f = fx.pose, rv = revealY();
  // corner brackets
  const L = Math.min(cw, ch) * 0.07, m = 10, pulse = 0.55 + 0.45 * Math.sin(t * 3);
  c.save(); c.strokeStyle = NEON; c.globalAlpha = 0.5 + 0.4 * pulse; c.lineWidth = 2; c.shadowColor = NEON; c.shadowBlur = 8;
  for (const [x, y, sx, sy] of [[m, m, 1, 1], [cw - m, m, -1, 1], [m, ch - m, 1, -1], [cw - m, ch - m, -1, -1]]) { c.beginPath(); c.moveTo(x, y + sy * L); c.lineTo(x, y); c.lineTo(x + sx * L, y); c.stroke(); }
  c.restore();
  // continuous scan sweep
  const sy = ((t * 0.38) % 1) * ch, gr = c.createLinearGradient(0, sy - 60, 0, sy);
  gr.addColorStop(0, 'rgba(34,230,255,0)'); gr.addColorStop(1, 'rgba(34,230,255,0.10)'); c.fillStyle = gr; c.fillRect(0, sy - 60, cw, 60);
  c.fillStyle = 'rgba(34,230,255,0.35)'; c.fillRect(0, sy, cw, 1);
  // reveal scan line
  if (rv < H) { const y = rv * k; c.save(); c.shadowColor = LIME; c.shadowBlur = 24; c.fillStyle = LIME; c.fillRect(0, y - 1.5, cw, 3); c.restore();
    c.fillStyle = 'rgba(182,255,59,0.07)'; c.fillRect(0, y, cw, ch - y);
    c.font = '600 12px "JetBrains Mono",monospace'; c.fillStyle = LIME; c.fillText('SCANNING BODY · ' + Math.round(rv / H * 100) + '%', 16, Math.max(18, y - 10)); }
  if (f) {
    const lm = f.lm, P = i => ({ x: lm[i].x * cw, y: lm[i].y * ch }), vis = i => lm[i].v > 0.5 && lm[i].y * H <= rv;
    // motion trails
    for (const i of TRAIL_IDS) { const T = fx.trails[i]; if (!T || T.length < 3 || !vis(i)) continue; const col = i % 2 ? NEON : MAG;
      for (let s = 1; s < T.length; s++) glowLine(c, [{ x: T[s - 1].x * k, y: T[s - 1].y * k }, { x: T[s].x * k, y: T[s].y * k }], col, 1 + 2.5 * s / T.length, 0.12 + 0.75 * s / T.length); }
    // reticles on hips and knees
    c.save(); c.lineWidth = 1.5;
    for (const i of [23, 24, 25, 26]) { if (!vis(i)) continue; const p = P(i), R = 13 + 3 * Math.sin(t * 4 + i); c.strokeStyle = i < 25 ? MAG : NEON; c.shadowColor = c.strokeStyle; c.shadowBlur = 10;
      c.setLineDash([5, 5]); c.lineDashOffset = -t * 30; c.beginPath(); c.arc(p.x, p.y, R, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
      c.beginPath(); c.moveTo(p.x - R - 6, p.y); c.lineTo(p.x - R + 2, p.y); c.moveTo(p.x + R - 2, p.y); c.lineTo(p.x + R + 6, p.y); c.stroke(); }
    c.restore();
    // center of mass
    if (vis(23) && vis(24)) { const h = mid(P(23), P(24)), b = 1 + 0.25 * Math.sin(t * 5); c.save(); c.strokeStyle = LIME; c.fillStyle = LIME; c.shadowColor = LIME; c.shadowBlur = 14;
      c.beginPath(); c.arc(h.x, h.y, 6 * b, 0, Math.PI * 2); c.stroke(); c.beginPath(); c.arc(h.x, h.y, 2.5, 0, Math.PI * 2); c.fill();
      c.globalAlpha = 0.5; c.setLineDash([3, 6]); c.beginPath(); c.moveTo(h.x, h.y); c.lineTo(h.x, ch - 6); c.stroke(); c.restore();
      c.font = '600 11px "JetBrains Mono",monospace'; c.fillStyle = LIME; c.fillText('COM', h.x + 10, h.y - 8); }
    // callouts: knee & hip angle with live velocity
    const callouts = [[25, 'L KNEE', 'kneeL'], [24, 'R HIP', 'hipR']];
    callouts.forEach(([i, lab, id], n) => { if (!vis(i)) return; const p = P(i), j = JOINTS.find(q => q.id === id); const a = angle3(f.wl[j.a], f.wl[j.b], f.wl[j.c]);
      const vel = A && f.i != null && A.vel[id] ? A.vel[id][f.i] : NaN; const bx = n ? Math.max(8, p.x - 160) : Math.min(cw - 150, p.x + 40), by = p.y + (n ? -70 : 30);
      c.save(); c.strokeStyle = 'rgba(34,230,255,.7)'; c.lineWidth = 1; c.beginPath(); c.moveTo(p.x, p.y); c.lineTo(bx + (n ? 140 : 0), by + 18); c.stroke();
      c.fillStyle = 'rgba(6,12,20,.78)'; c.strokeStyle = 'rgba(34,230,255,.55)'; c.beginPath(); c.roundRect(bx, by, 140, 40, 8); c.fill(); c.stroke();
      c.font = '600 10px "JetBrains Mono",monospace'; c.fillStyle = '#8FB9C9'; c.fillText(lab, bx + 10, by + 15);
      c.font = '700 15px "JetBrains Mono",monospace'; c.fillStyle = '#fff'; c.fillText(fmt(a, 1) + '°', bx + 10, by + 32);
      c.font = '500 10px "JetBrains Mono",monospace'; c.fillStyle = NEON; c.fillText(Number.isFinite(vel) ? (vel > 0 ? '+' : '') + fmt(vel) + '°/s' : 'locked', bx + 76, by + 32); c.restore(); });
  }
  // HUD text
  c.font = '600 11px "JetBrains Mono",monospace'; c.textAlign = 'right';
  const conf = f ? f.lm.reduce((a, p) => a + p.v, 0) / f.lm.length : 0;
  c.fillStyle = f ? NEON : '#FFB547'; c.fillText(f ? `POSE LOCK · 33 PTS · CONF ${conf.toFixed(2)}` : 'SEARCHING FOR BODY' + '.'.repeat(1 + Math.floor(t * 3) % 3), cw - 16, 26);
  c.fillStyle = '#7F95AA'; c.fillText(`${mediaType === 'video' ? fx.fps + ' FPS · ' : ''}FRAMES ${frames.size} · ${A && A.frontal ? 'FRONTAL' : 'SAGITTAL'} PLANE`, cw - 16, 42); c.textAlign = 'left';
  drawGauges(f); pushStream(f, conf); draw3D(f); drawScore();
}

function drawGauges(f) {
  const cv = $('#gauges'); if (!cv) return; const dpr = Math.min(2, window.devicePixelRatio || 1), w = cv.clientWidth || 300, h = 128;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + 'px'; }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  const G4 = ['kneeL', 'kneeR', 'hipL', 'hipR'], n = G4.length, cell = w / n, R = Math.min(cell * 0.36, 42), t = performance.now() / 1000;
  G4.forEach((id, i) => { const j = JOINTS.find(q => q.id === id), cx = cell * i + cell / 2, cy = 58;
    const target = f ? angle3(f.wl[j.a], f.wl[j.b], f.wl[j.c]) : 0; const cur = fx.g[id] ?? 0; const v = cur + (target - cur) * 0.18; fx.g[id] = v;
    const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25, av = a0 + (a1 - a0) * Math.min(1, v / 180);
    c.lineCap = 'round'; c.lineWidth = 6; c.strokeStyle = 'rgba(255,255,255,.07)'; c.beginPath(); c.arc(cx, cy, R, a0, a1); c.stroke();
    for (let q = 0; q <= 12; q++) { const aa = a0 + (a1 - a0) * q / 12; c.strokeStyle = 'rgba(255,255,255,.18)'; c.lineWidth = 1; c.beginPath(); c.moveTo(cx + Math.cos(aa) * (R + 6), cy + Math.sin(aa) * (R + 6)); c.lineTo(cx + Math.cos(aa) * (R + 10), cy + Math.sin(aa) * (R + 10)); c.stroke(); }
    const g = c.createLinearGradient(cx - R, cy, cx + R, cy); g.addColorStop(0, j.color); g.addColorStop(1, '#FFFFFF');
    c.save(); c.shadowColor = j.color; c.shadowBlur = 14; c.strokeStyle = g; c.lineWidth = 6; c.beginPath(); c.arc(cx, cy, R, a0, av); c.stroke(); c.restore();
    c.fillStyle = j.color; c.beginPath(); c.arc(cx + Math.cos(av) * R, cy + Math.sin(av) * R, 4 + Math.sin(t * 6) * 1.2, 0, Math.PI * 2); c.fill();
    c.textAlign = 'center'; c.fillStyle = '#fff'; c.font = '700 17px "JetBrains Mono",monospace'; c.fillText(f ? Math.round(v) + '°' : '--', cx, cy + 6);
    c.fillStyle = '#8B97A6'; c.font = '600 10px "IBM Plex Sans",sans-serif'; c.fillText(j.name.toUpperCase(), cx, h - 8); c.textAlign = 'left'; });
}
function pushStream(f, conf) {
  const now = performance.now(); if (!f || now - fx.lastStream < 160) return; fx.lastStream = now;
  const wl = f.wl, ang = id => { const j = JOINTS.find(q => q.id === id); return angle3(wl[j.a], wl[j.b], wl[j.c]); };
  const sp = A && f.i != null && A.speed && A.list.length > 2 ? A.speed[f.i] : NaN;
  const line = `t=${(render.lastT || 0).toFixed(2).padStart(5)}s  KNEE ${fmt(ang('kneeL'), 1).padStart(5)}/${fmt(ang('kneeR'), 1).padStart(5)}°  HIP ${fmt(ang('hipL'), 1).padStart(5)}°  ${Number.isFinite(sp) ? 'v=' + sp.toFixed(2) + 'm/s' : 'c=' + conf.toFixed(3)}`;
  if (fx.stream[0] && fx.stream[0].slice(8) === line.slice(8)) return;
  fx.stream.unshift(line); if (fx.stream.length > 7) fx.stream.pop(); $('#stream').textContent = fx.stream.join('\n');
}

// animated hero: a neon athlete squatting under live tracking
function hero() {
  const cv = $('#hero'), host = $('#empty'); if (!cv) return; const c = cv.getContext('2d'); const trail = { hip: [], knee: [] };
  const frame = () => {
    requestAnimationFrame(frame); if (host.hidden) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1), w = host.clientWidth, h = host.clientHeight; if (!w || !h) return;
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h); const t = performance.now() / 1000;
    // perspective floor grid
    const hz = h * 0.62; c.save(); c.strokeStyle = 'rgba(34,230,255,.16)'; c.lineWidth = 1;
    for (let i = -14; i <= 14; i++) { c.beginPath(); c.moveTo(w * 0.62 + i * 14, hz); c.lineTo(w * 0.62 + i * 120, h); c.stroke(); }
    for (let i = 0; i < 12; i++) { const z = ((i + (t * 0.6) % 1) / 12); const y = hz + (h - hz) * z * z; c.globalAlpha = z; c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }
    c.restore();
    // figure (side view squat)
    const wide = w > 760, S = Math.min(h * 0.78, wide ? w * 0.42 : w * 0.9), ox = wide ? w * 0.7 : w * 0.5, oy = h * 0.93, s = (1 - Math.cos(t * 1.6)) / 2, rad = d => d * Math.PI / 180;
    const pt = (b, ang, L) => ({ x: b.x + Math.sin(ang) * L * S, y: b.y - Math.cos(ang) * L * S });
    const ankle = { x: ox, y: oy }, toe = { x: ox + 0.09 * S, y: oy }, heel = { x: ox - 0.03 * S, y: oy };
    const knee = pt(ankle, rad(6 + 36 * s), 0.25), hip = pt(knee, rad(-8 - 84 * s), 0.25), sh = pt(hip, rad(12 + 34 * s), 0.3);
    const head = pt(sh, rad(14 + 30 * s), 0.1), elb = pt(sh, rad(150), 0.1), wr = pt(elb, rad(20), 0.08);
    const kAng = 180 - (6 + 36 * s + 8 + 84 * s);
    trail.hip.push(hip); trail.knee.push(knee); if (trail.hip.length > 40) { trail.hip.shift(); trail.knee.shift(); }
    c.globalAlpha = wide ? 1 : 0.35;
    for (const [T, col] of [[trail.hip, MAG], [trail.knee, NEON]]) for (let i = 1; i < T.length; i++) glowLine(c, [T[i - 1], T[i]], col, 1 + 3 * i / T.length, 0.1 + 0.7 * i / T.length);
    // body silhouette glow
    const bones = [[ankle, knee], [knee, hip], [hip, sh], [sh, head], [sh, elb], [elb, wr], [heel, toe]];
    for (const [a, b] of bones) { glowLine(c, [a, b], 'rgba(34,230,255,.25)', 16, 1); glowLine(c, [a, b], '#E8FBFF', 3, 1); }
    c.save(); c.fillStyle = 'rgba(34,230,255,.15)'; c.strokeStyle = NEON; c.lineWidth = 2; c.shadowColor = NEON; c.shadowBlur = 16; c.beginPath(); c.arc(head.x, head.y - 0.02 * S, 0.055 * S, 0, Math.PI * 2); c.fill(); c.stroke(); c.restore();
    // barbell
    c.save(); c.strokeStyle = '#9FB3C8'; c.lineWidth = 4; c.beginPath(); c.moveTo(sh.x - 0.14 * S, sh.y + 0.01 * S); c.lineTo(sh.x + 0.14 * S, sh.y + 0.01 * S); c.stroke(); c.fillStyle = '#2B3A4C'; c.strokeStyle = NEON; c.lineWidth = 1.5; c.shadowColor = NEON; c.shadowBlur = 10; c.beginPath(); c.arc(sh.x, sh.y + 0.01 * S, 0.12 * S, 0, Math.PI * 2); c.globalAlpha *= .55; c.fill(); c.globalAlpha = wide ? 1 : .35; c.stroke(); c.restore();
    // joints
    for (const [p, col] of [[ankle, NEON], [knee, NEON], [hip, MAG], [sh, LIME], [elb, NEON], [wr, NEON]]) { c.save(); c.fillStyle = col; c.shadowColor = col; c.shadowBlur = 18; c.beginPath(); c.arc(p.x, p.y, 5 + Math.sin(t * 5) * 1.2, 0, Math.PI * 2); c.fill(); c.restore(); }
    // knee angle arc + label
    const a1 = Math.atan2(ankle.y - knee.y, ankle.x - knee.x), a2 = Math.atan2(hip.y - knee.y, hip.x - knee.x);
    c.save(); c.strokeStyle = LIME; c.lineWidth = 2.5; c.shadowColor = LIME; c.shadowBlur = 12; c.beginPath(); c.arc(knee.x, knee.y, 0.07 * S, Math.min(a1, a2), Math.max(a1, a2)); c.stroke(); c.restore();
    c.font = '700 14px "JetBrains Mono",monospace'; c.fillStyle = LIME; c.fillText(Math.round(kAng) + '°', knee.x + 0.08 * S, knee.y + 4);
    // reticle around hip
    c.save(); c.strokeStyle = MAG; c.lineWidth = 1.5; c.setLineDash([6, 6]); c.lineDashOffset = -t * 40; c.beginPath(); c.arc(hip.x, hip.y, 0.06 * S, 0, Math.PI * 2); c.stroke(); c.restore();
    // rotating HUD rings
    c.save(); c.translate(ox - 0.02 * S, oy - 0.42 * S); c.strokeStyle = 'rgba(34,230,255,.22)'; c.lineWidth = 1;
    for (const [r0, sp, dash] of [[0.5, 0.2, [2, 10]], [0.56, -0.12, [30, 14]], [0.62, 0.05, [1, 5]]]) { c.save(); c.rotate(t * sp); c.setLineDash(dash); c.beginPath(); c.arc(0, 0, r0 * S, 0, Math.PI * 2); c.stroke(); c.restore(); }
    c.restore();
    // floating telemetry
    if (wide) { c.font = '600 11px "JetBrains Mono",monospace'; const rows = [['KNEE FLEX', Math.round(kAng) + '°', NEON], ['HIP DEPTH', (s * 100).toFixed(0) + '%', MAG], ['BAR PATH Δx', (Math.sin(t * 1.6) * 1.8).toFixed(1) + ' cm', LIME], ['GRF', (1 + 0.35 * Math.sin(t * 1.6 + 1)).toFixed(2) + '× BW', '#FFB547']];
      rows.forEach(([a, b, col], i) => { const x = Math.min(ox + 0.24 * S, w - 166), y = oy - 0.98 * S + i * 34; c.fillStyle = 'rgba(6,12,20,.7)'; c.strokeStyle = col; c.globalAlpha = .9; c.beginPath(); c.roundRect(x, y, 150, 26, 6); c.fill(); c.globalAlpha = .45; c.stroke(); c.globalAlpha = 1; c.fillStyle = '#8FA3B6'; c.fillText(a, x + 8, y + 17); c.fillStyle = col; c.textAlign = 'right'; c.fillText(b, x + 142, y + 17); c.textAlign = 'left'; }); }
    // scan line
    const sy = ((t * 0.3) % 1) * h; c.globalAlpha = 1; const gr = c.createLinearGradient(0, sy - 80, 0, sy); gr.addColorStop(0, 'rgba(34,230,255,0)'); gr.addColorStop(1, 'rgba(34,230,255,.09)'); c.fillStyle = gr; c.fillRect(0, sy - 80, w, 80);
  };
  frame();
}

// ---------- 3D body model + form score + snapshot ----------
const BONES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[27,31],[28,30],[30,32],[28,32]];
const v3 = { yaw: 0.6, drag: null, auto: true, last: performance.now() };
(() => { const cv = $('#three'); if (!cv) return;
  cv.addEventListener('pointerdown', e => { v3.drag = { x: e.clientX, yaw: v3.yaw }; v3.auto = false; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (v3.drag) v3.yaw = v3.drag.yaw + (e.clientX - v3.drag.x) * 0.012; });
  const up = () => { v3.drag = null; setTimeout(() => { if (!v3.drag) v3.auto = true; }, 2500); };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up); })();
function sizeCv(cv) { const dpr = Math.min(2, window.devicePixelRatio || 1), w = cv.clientWidth || 260, h = +(cv.dataset.h || (cv.dataset.h = cv.getAttribute('height'))); if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px'; } const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h); return [c, w, h]; }
function draw3D(f) {
  const cv = $('#three'); if (!cv) return; const [c, w, h] = sizeCv(cv), now = performance.now(), t = now / 1000;
  if (v3.auto) v3.yaw += (now - v3.last) / 1000 * 0.5; v3.last = now;
  c.fillStyle = 'rgba(4,8,14,.55)'; c.fillRect(0, 0, w, h);
  const ys = f ? f.wl.map(p => p.y).filter(Number.isFinite) : [-0.8, 0.9], y0 = Math.min(...ys), y1 = Math.max(...ys), span = Math.max(0.6, y1 - y0);
  const cs = Math.cos(v3.yaw), sn = Math.sin(v3.yaw), tilt = 0.32, K = h * 0.74 / span, ym = (y0 + y1) / 2, cx = w / 2, cy = h * 0.47;
  const proj = p => { const x = p.x * cs - p.z * sn, z = p.x * sn + p.z * cs, py = p.y - ym; const y = py * Math.cos(tilt) - z * Math.sin(tilt), d = py * Math.sin(tilt) + z * Math.cos(tilt); return { x: cx + x * K, y: cy + y * K, d }; };
  // floor ring grid
  const foot = f ? Math.max(f.wl[27].y, f.wl[28].y, f.wl[31].y, f.wl[32].y) : 0.9;
  c.save(); c.strokeStyle = 'rgba(34,230,255,.18)'; c.lineWidth = 1;
  for (let r = span * 0.12; r <= span * 0.5; r += span * 0.12) { c.beginPath(); for (let a = 0; a <= 64; a++) { const th = a / 64 * Math.PI * 2, p = proj({ x: Math.cos(th) * r, y: foot, z: Math.sin(th) * r }); a ? c.lineTo(p.x, p.y) : c.moveTo(p.x, p.y); } c.stroke(); }
  for (let a = 0; a < 8; a++) { const th = a / 8 * Math.PI * 2 + t * 0.2, p0 = proj({ x: 0, y: foot, z: 0 }), p1 = proj({ x: Math.cos(th) * span * 0.5, y: foot, z: Math.sin(th) * span * 0.5 }); c.beginPath(); c.moveTo(p0.x, p0.y); c.lineTo(p1.x, p1.y); c.stroke(); }
  c.restore();
  if (!f) { c.font = '600 11px "JetBrains Mono",monospace'; c.fillStyle = '#5E7186'; c.textAlign = 'center'; c.fillText('NO BODY DATA YET', cx, cy); c.textAlign = 'left'; return; }
  const P = f.wl.map(proj);
  // shadow on floor
  c.save(); c.globalAlpha = 0.25; c.fillStyle = '#22E6FF'; for (const i of [27, 28, 23, 24]) { const q = proj({ x: f.wl[i].x, y: foot, z: f.wl[i].z }); c.beginPath(); c.ellipse(q.x, q.y, 8, 3, 0, 0, Math.PI * 2); c.fill(); } c.restore();
  const bones = BONES.map(([a, b]) => ({ a, b, d: (P[a].d + P[b].d) / 2 })).sort((p, q) => q.d - p.d);
  for (const { a, b, d } of bones) { const left = a % 2 === 1 && b % 2 === 1, col = a >= 23 && b >= 23 ? (left ? '#22E6FF' : '#FF3DA6') : left ? '#7FE9FF' : '#FF9AD0', al = 0.55 + 0.45 * Math.max(0, Math.min(1, 0.5 - d));
    glowLine(c, [P[a], P[b]], col, 3, al); }
  const head = P[0]; c.save(); c.strokeStyle = '#B6FF3B'; c.shadowColor = '#B6FF3B'; c.shadowBlur = 10; c.lineWidth = 2; c.beginPath(); c.arc(head.x, head.y, h * 0.045, 0, Math.PI * 2); c.stroke(); c.restore();
  for (const i of [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) { c.fillStyle = '#fff'; c.beginPath(); c.arc(P[i].x, P[i].y, 2.6, 0, Math.PI * 2); c.fill(); }
  const com = proj(mid3(f.wl[23], f.wl[24])); c.save(); c.strokeStyle = '#B6FF3B'; c.setLineDash([3, 4]); c.beginPath(); c.moveTo(com.x, com.y); c.lineTo(proj({ ...mid3(f.wl[23], f.wl[24]), y: foot }).x, proj({ ...mid3(f.wl[23], f.wl[24]), y: foot }).y); c.stroke(); c.restore();
  c.font = '600 10px "JetBrains Mono",monospace'; c.fillStyle = '#5E7186'; c.fillText('YAW ' + (((v3.yaw * 180 / Math.PI) % 360 + 360) % 360).toFixed(0) + '°', 8, h - 8);
}
const mid3 = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });
const SC = { v: 0 };
function drawScore() {
  const cv = $('#score'); if (!cv) return; const [c, w, h] = sizeCv(cv), t = performance.now() / 1000;
  c.fillStyle = 'rgba(4,8,14,.55)'; c.fillRect(0, 0, w, h);
  const ch = A ? techniqueChecks().filter(x => x.s !== 'idle') : [], target = ch.length ? ch.reduce((a, x) => a + (x.s === 'ok' ? 100 : x.s === 'warn' ? 72 : 45), 0) / ch.length : 0;
  SC.v += (target - SC.v) * 0.06; const v = SC.v, cx = w / 2, cy = h * 0.45, R = Math.min(w * 0.36, 78);
  const col = !ch.length ? '#3A4656' : v >= 90 ? '#3DDC97' : v >= 70 ? '#FFB547' : '#FF5A3C';
  c.lineCap = 'round'; c.lineWidth = 10; c.strokeStyle = 'rgba(255,255,255,.06)'; c.beginPath(); c.arc(cx, cy, R, 0, Math.PI * 2); c.stroke();
  c.save(); c.shadowColor = col; c.shadowBlur = 18; c.strokeStyle = col; c.beginPath(); c.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * v / 100); c.stroke(); c.restore();
  c.save(); c.translate(cx, cy); c.rotate(t * 0.8); c.strokeStyle = 'rgba(34,230,255,.35)'; c.lineWidth = 1; c.setLineDash([2, 7]); c.beginPath(); c.arc(0, 0, R + 14, 0, Math.PI * 2); c.stroke(); c.restore();
  c.textAlign = 'center'; c.fillStyle = '#fff'; c.font = '700 34px Saira,sans-serif'; c.fillText(ch.length ? Math.round(v) : '--', cx, cy + 10);
  c.font = '600 10px "JetBrains Mono",monospace'; c.fillStyle = '#8B97A6'; c.fillText('/ 100', cx, cy + 28);
  const grade = !ch.length ? 'WAITING' : v >= 90 ? 'CLEAN REP' : v >= 70 ? 'SOLID, TWEAK IT' : 'NEEDS WORK';
  c.font = '700 12px "JetBrains Mono",monospace'; c.fillStyle = col; c.fillText(grade, cx, h - 34);
  c.font = '500 10px "IBM Plex Sans",sans-serif'; c.fillStyle = '#6C7A8B'; c.fillText(ch.length ? `${ch.filter(x => x.s === 'ok').length} of ${ch.length} checks passed` : 'from the technique checks', cx, h - 16); c.textAlign = 'left';
}
$('#snap').addEventListener('click', () => {
  if (!media) return; const out = document.createElement('canvas'); out.width = view.width; out.height = view.height; const o = out.getContext('2d');
  o.drawImage(view, 0, 0); o.drawImage(fxc, 0, 0, out.width, out.height);
  o.font = `600 ${Math.round(out.width / 45)}px "JetBrains Mono",monospace`; o.fillStyle = '#22E6FF'; o.fillText('VELO · markerless motion lab', 16, out.height - 16);
  const a = document.createElement('a'); a.download = 'velo-analysis.png'; a.href = out.toDataURL('image/png'); a.click();
});

// ---------- events ----------
for (const id of ['#file', '#file2']) $(id).addEventListener('change', e => { $('#play').disabled = false; $('#scrub').disabled = false; loadFile(e.target.files[0]); e.target.value = ''; });
$('#sample').addEventListener('click', () => loadImage('img/sample-squat.jpg'));
document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
  mode = b.dataset.mode; document.querySelectorAll('.seg button').forEach(x => x.setAttribute('aria-pressed', String(x === b))); dirty = true;
}));
['#tTriad', '#tGrid', '#tForce'].forEach(id => $(id).addEventListener('change', () => dirty = true));
['#sport', '#hand'].forEach(id => $(id).addEventListener('change', () => { updatePanel(); dirty = true; }));
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
    frames.set(0, { t: 0, lm, wl }); analyze(); dirty = true; fx.revealAt = performance.now();
  },
  setMode(m) { document.querySelector(`.seg [data-mode="${m}"]`).click(); },
  dump() { return [...frames.values()][0] || null; },
  loadFile,
};
updatePanel();
requestAnimationFrame(tick);
hero();
