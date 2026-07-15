// Trove -- capture-and-collect companion to spatial-capture. Vanilla, self-contained.

// Real captures only. assets/collection.json is the source of truth and the
// pipeline writes it as scenes are captured and fused. There is deliberately no
// sample data: a catalog of scenes that don't exist would misrepresent the work,
// and an empty shelf is a more honest first impression than a staged one.
const EMPTY = { scenes: [], finds: [] };

async function loadData() {
  try {
    const r = await fetch("assets/collection.json", { cache: "no-cache" });
    if (r.ok) {
      const d = await r.json();
      return { scenes: d.scenes ?? [], finds: d.finds ?? [] };
    }
  } catch (_) { /* offline, or opened from file:// -- show the empty state */ }
  return EMPTY;
}

// ---- seeded RNG so each scene's cover art is stable ----
function rng(seed) {
  let s = 0;
  for (const c of String(seed)) s = (s * 31 + c.charCodeAt(0)) >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

function softBlob(ctx, x, y, r, h, s, l, a) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, `hsla(${h},${s}%,${l}%,${a})`);
  g.addColorStop(1, `hsla(${h},${s}%,${l}%,0)`);
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
}

// A scene cover as an out-of-focus point cloud -- the splat's own look.
function drawCover(canvas, hue, seed) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 180, h = canvas.clientHeight || 135;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  ctx.fillStyle = "#07090c"; ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = "lighter";
  const rand = rng(seed);
  for (let i = 0; i < 18; i++) {
    softBlob(ctx, w * (0.12 + 0.76 * rand()), h * (0.12 + 0.76 * rand()),
      h * (0.12 + 0.34 * rand()), hue + (rand() * 40 - 20), 62, 45 + rand() * 25, 0.15 + rand() * 0.14);
  }
  for (let i = 0; i < 6; i++) { // a few bright cores
    softBlob(ctx, w * (0.2 + 0.6 * rand()), h * (0.2 + 0.6 * rand()), h * 0.055, hue, 28, 92, 0.5);
  }
  ctx.globalCompositeOperation = "source-over";
}

// ---- ambient drifting glow field behind the shell ----
function startAmbient() {
  const c = document.getElementById("ambient"), ctx = c.getContext("2d");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  let W, H, blobs;
  const resize = () => { W = innerWidth; H = innerHeight; c.width = W * dpr; c.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
  const init = () => {
    const r = rng("ambient");
    blobs = Array.from({ length: 16 }, () => ({
      x: r() * innerWidth, y: r() * innerHeight, vx: (r() - 0.5) * 0.12, vy: (r() - 0.5) * 0.12,
      rad: 60 + r() * 140, h: r() < 0.6 ? 40 : 190, a: 0.05 + r() * 0.06,
    }));
  };
  const frame = () => {
    ctx.clearRect(0, 0, W, H); ctx.globalCompositeOperation = "lighter";
    for (const b of blobs) {
      b.x += b.vx; b.y += b.vy;
      if (b.x < -220) b.x = W + 220; if (b.x > W + 220) b.x = -220;
      if (b.y < -220) b.y = H + 220; if (b.y > H + 220) b.y = -220;
      softBlob(ctx, b.x, b.y, b.rad, b.h, 60, 55, b.a);
    }
    ctx.globalCompositeOperation = "source-over";
    if (!reduce) requestAnimationFrame(frame);
  };
  resize(); init(); addEventListener("resize", () => { resize(); init(); }); frame();
}

const CAT_HUE = { book: 40, plant: 140, lamp: 44, monitor: 200, keyboard: 210, tool: 190, mug: 20, chair: 275 };
const hueFor = (cat) => CAT_HUE[cat] ?? 210;
const fmtCount = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : "" + n;
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function renderScenes(data) {
  const n = data.scenes.length;
  const grid = document.getElementById("scene-grid");
  document.getElementById("scene-count").textContent = n === 1 ? "1 scene" : n + " scenes";
  grid.innerHTML = "";
  if (!n) {
    grid.innerHTML = `<div class="empty">nothing captured yet — start on the Capture tab</div>`;
    return;
  }
  for (const s of data.scenes) {
    const card = document.createElement("button");
    card.className = "specimen";
    card.innerHTML =
      `<span class="cover"><canvas></canvas>` +
      `<span class="badge ${s.mode}">${s.mode}</span>` +
      `<span class="gcount">${fmtCount(s.stats.gaussians)} gaussians</span></span>` +
      `<span class="body"><span class="title">${esc(s.name)}</span>` +
      `<span class="meta"><span><b>${s.stats.objects}</b> objects</span>` +
      `<span><b>${s.stats.frames}</b> frames</span><span>${esc(s.captured)}</span></span></span>`;
    grid.appendChild(card);
    requestAnimationFrame(() => drawCover(card.querySelector("canvas"), s.hue, s.id));
    card.onclick = () => openScene(s);
  }
}

function openScene(s) {
  if (s.splat) {
    const q = new URLSearchParams({ src: s.splat });
    if (s.scene) q.set("scene", s.scene);
    location.href = "../viewer/?" + q.toString();
  } else {
    toast(`"${s.name}" is still processing`);
  }
}

let activeCat = "all";
function renderFinds(data) {
  const finds = data.finds || [];
  document.getElementById("find-count").textContent = finds.length + " found";
  const fbar = document.getElementById("filters");
  const wrap = document.getElementById("finds");
  fbar.innerHTML = "";
  wrap.innerHTML = "";
  // Nothing found at all is not a filter miss: "all" is a pseudo-category, so the
  // per-category copy would read "no all yet" and the chip row would offer a single
  // filter over an empty set. Say what is actually true instead.
  if (!finds.length) {
    wrap.innerHTML = `<div class="empty">no objects yet — the understanding branch catalogues them when a scene is fused</div>`;
    return;
  }
  const cats = ["all", ...Array.from(new Set(finds.map(f => f.category)))];
  for (const cat of cats) {
    const b = document.createElement("button");
    b.className = "chip" + (cat === activeCat ? " on" : "");
    b.textContent = cat;
    b.onclick = () => { activeCat = cat; renderFinds(data); };
    fbar.appendChild(b);
  }
  const shown = finds.filter(f => activeCat === "all" || f.category === activeCat);
  if (!shown.length) { wrap.innerHTML = `<div class="empty">no ${esc(activeCat)} yet</div>`; return; }
  for (const f of shown) {
    const h = hueFor(f.category);
    const el = document.createElement("div");
    el.className = "find";
    el.innerHTML =
      `<span class="glyph" style="box-shadow:inset 0 0 12px hsla(${h},70%,60%,.5),0 0 10px hsla(${h},70%,55%,.25)"></span>` +
      `<span class="lbl"><span class="n">${esc(f.label)}</span>` +
      `<span class="t">${esc(f.category)} · ${esc(f.scene)} · ${Math.round(f.confidence * 100)}%</span></span>`;
    wrap.appendChild(el);
  }
}

const CHECKS = {
  object: [
    ["Lock exposure, white balance & focus", "consistency beats any one good frame"],
    ["Fast shutter, lots of light", "motion blur is the #1 killer"],
    ["Arc around it — don't rotate it", "move yourself; keep the subject still"],
    ["Three passes: low, eye, high", "cover the top and both sides"],
    ["70–80% overlap between viewpoints", "small steps, ~10–15° apart"],
  ],
  enclosure: [
    ["Lock exposure, white balance & focus", "one look, start to finish"],
    ["Walk — never stand and spin", "parallax comes from your feet"],
    ["Three loops: hip, eye, overhead", "tilt for ceiling & floor seams"],
    ["Anchor on the corners", "they hold the walls in place"],
    ["Close the loop", "finish on your starting view"],
  ],
};
let mode = "object";
function renderChecklist() {
  const list = document.getElementById("checklist");
  list.innerHTML = "";
  CHECKS[mode].forEach(([txt, sub], i) => {
    const row = document.createElement("div");
    row.className = "check";
    row.innerHTML =
      `<span class="idx">${String(i + 1).padStart(2, "0")}</span>` +
      `<span class="txt">${esc(txt)}<small>${esc(sub)}</small></span>` +
      `<span class="tick">✓</span>`;
    row.querySelector(".tick").onclick = () => row.classList.toggle("done");
    list.appendChild(row);
  });
}

// ---- shoot + import ----
// Being straight about the platform instead of faking it: a web page cannot open
// the iOS Camera app or drive its controls, and iOS Safari does not expose the
// MediaStreamTrack constraints (exposureMode / focusMode / whiteBalanceMode) that
// would let it lock a look the way photogrammetry needs. Recording in-page via
// MediaRecorder is possible but strictly worse than the native camera. So Trove
// does not pretend to be a camera: it coaches the shot, you take it in the app
// that actually has the controls, and Trove carries the file to the workstation.
const SHOOT = {
  object:
    "Record in your phone's <b>camera app</b> — it has the exposure and focus lock this page can't reach. " +
    "Frame the object, <b>long-press until AE/AF LOCK appears</b>, then keep that one look for the whole take. " +
    "Orbit it in three passes — low, eye, high. Move your feet; never turn the object.",
  enclosure:
    "Record in your phone's <b>camera app</b>. Point at a mid-bright wall and <b>long-press for AE/AF LOCK</b> " +
    "before you start — an exposure that drifts as you pan is the fastest route to a soft reconstruction. " +
    "Then <b>walk</b> the room in loops; standing and spinning gives the solver no parallax. Finish where you began.",
};

// Stock iOS AE/AF Lock holds exposure and focus but not white balance, and it does not
// stop the meter riding as you pan. Measured on a real take here: brightness swung 92%
// across one capture with AE live. That is the whole ballgame -- worse than noise, worse
// than resolution -- so the manual app is not a nicety.
//
// Frame rate is NOT shutter speed: fps only caps the slowest allowed exposure (1/30 at
// 30fps). Since frames get decimated to ~3fps downstream, 60fps buys no usable frames --
// it only doubles the file. What matters is the shutter, and light is what buys it.
const SHOOT_COMMON =
  "<br><br>Wipe the lens, HDR and filters off, stay on the <b>1× lens</b>, and give the room all the light " +
  "you have — light is what buys you a fast shutter. Stock AE/AF Lock won't hold <b>white balance</b>, and " +
  "won't stop the meter riding as you pan; <b>Blackmagic Camera</b> (free) locks shutter, ISO and WB outright. " +
  "Worth the install for any take you care about. Shoot <b>4K/30 at 1/120s</b> there — 60fps just doubles the " +
  "file, since only ~3 frames a second survive to the solver.";

// The GitHub Pages copy is HTTPS, and a secure page may not POST to a plain-HTTP
// LAN address (mixed content). So import only works when Trove is served by the
// workstation itself -- which is exactly what pipeline/00_import_server.py does.
const canImport = location.protocol === "http:";

function renderShootCopy() {
  document.getElementById("shoot-copy").innerHTML = SHOOT[mode] + SHOOT_COMMON;
}

function initCapture() {
  document.querySelectorAll("#mode-seg button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll("#mode-seg button").forEach(x => x.classList.remove("on"));
      b.classList.add("on"); mode = b.dataset.mode; renderChecklist(); renderShootCopy();
    };
  });
  renderChecklist();
  renderShootCopy();
  initImport();
}

function initImport() {
  const ring = document.getElementById("record");
  const hint = document.getElementById("record-hint");
  const copy = document.getElementById("import-copy");
  const file = document.getElementById("file");

  if (!canImport) {
    copy.innerHTML =
      "Trove is running from the public web here, so it can't reach your machine. To import, run " +
      "<code>python pipeline/00_import_server.py</code> on the workstation and open the " +
      "<code>http://…:8099/app/</code> URL it prints, on the same WiFi.";
    ring.classList.add("off");
    ring.removeAttribute("tabindex");
    ring.setAttribute("aria-disabled", "true");
    hint.textContent = "import needs the workstation's own URL";
    document.getElementById("scene-field").hidden = true;
    return;
  }

  copy.innerHTML =
    "Videos land in <code>data/&lt;scene&gt;/</code>, photos in <code>photos/</code>. " +
    "Pick as many as you like. <b>Shoot to Files, import from Files</b> — the iOS Photos picker " +
    "hands out a re-encoded copy (a 4K60 HEVC take arrives as 4K30 H.264), so anything picked " +
    "from the photo library is not the file you shot. Trove checks what actually arrived and says so.";
  const pick = () => file.click();
  ring.onclick = pick;
  ring.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } };
  file.onchange = () => {
    const picked = Array.from(file.files || []);
    file.value = "";              // so re-picking the same file fires change again
    if (picked.length) enqueue(picked);
  };
}

// ---- upload queue ----
// One file in flight at a time: phone WiFi uplink is the bottleneck, and parallel
// uploads just make every row crawl and every ETA a lie. Sequential means the row
// you're watching is the row that's moving.
const q = [];
let running = false;

function enqueue(files) {
  const wrap = document.getElementById("queue");
  for (const f of files) {
    const row = document.createElement("div");
    row.className = "qrow";
    row.innerHTML =
      `<span class="qname">${esc(f.name)}</span>` +
      `<span class="qmeta">${(f.size / 1048576).toFixed(0)} MB · queued</span>` +
      `<span class="qbar"><span class="qfill"></span></span>`;
    wrap.appendChild(row);
    q.push({ file: f, row });
  }
  document.getElementById("record-hint").textContent =
    `${q.length} queued — keep this screen open and awake`;
  if (!running) pump();
}

function pump() {
  const ring = document.getElementById("record");
  const next = q.shift();
  if (!next) {
    running = false;
    ring.classList.remove("busy");
    document.getElementById("record-hint").textContent = "done — tap to add more";
    return;
  }
  running = true;
  ring.classList.add("busy");
  send(next).finally(pump);
}

function send({ file: f, row }) {
  return new Promise((resolve) => {
    const scene = (document.getElementById("scene").value || "").trim() || "shelf";
    const meta = row.querySelector(".qmeta");
    const fill = row.querySelector(".qfill");
    const mb = f.size / 1048576;
    row.classList.add("on");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?scene=${encodeURIComponent(scene)}&name=${encodeURIComponent(f.name)}`);
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      fill.style.width = (100 * e.loaded / e.total).toFixed(1) + "%";
      meta.textContent = `${(e.loaded / 1048576).toFixed(0)} / ${mb.toFixed(0)} MB`;
    };
    xhr.onload = () => {
      let res = {};
      try { res = JSON.parse(xhr.responseText); } catch (_) { /* non-JSON error body */ }
      row.classList.remove("on");
      if (xhr.status === 200 && res.ok) {
        fill.style.width = "100%";
        row.classList.add("ok");
        const m = res.media || {};
        meta.textContent = m.codec
          ? `${m.codec} ${m.width}x${m.height} ${m.fps}fps ${m.mbps}Mbps · ${m.seconds}s`
          : `${mb.toFixed(0)} MB · in ${res.kind === "photo" ? "photos/" : "data/" + scene}`;
        if (m.warning) {
          row.classList.add("warn");
          const w = document.createElement("span");
          w.className = "qwarn";
          w.textContent = m.warning;
          row.appendChild(w);
        }
      } else {
        row.classList.add("bad");
        meta.textContent = res.error || `failed (${xhr.status})`;
      }
      resolve();
    };
    xhr.onerror = () => {
      row.classList.remove("on");
      row.classList.add("bad");
      meta.textContent = "couldn't reach the workstation — same WiFi?";
      resolve();
    };
    xhr.send(f);
  });
}

function initTabs() {
  document.querySelectorAll(".tabbar button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll(".tabbar button").forEach(x => x.classList.remove("on"));
      document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
      b.classList.add("on");
      document.getElementById("view-" + b.dataset.view).classList.add("active");
      document.querySelector("main").scrollTop = 0;
      window.__view = b.dataset.view;
    };
  });
}

let toastT;
function toast(msg) {
  let t = document.getElementById("toast");
  if (!t) {
    t = document.createElement("div"); t.id = "toast"; document.body.appendChild(t);
    t.style.cssText = "position:fixed;bottom:82px;left:50%;transform:translateX(-50%);z-index:20;" +
      "background:#1a212b;border:1px solid #232c38;color:#e8edf4;font:12px/1.4 ui-monospace,monospace;" +
      "padding:9px 14px;border-radius:10px;opacity:0;transition:opacity .2s";
  }
  t.textContent = msg; t.style.opacity = "1";
  clearTimeout(toastT); toastT = setTimeout(() => { t.style.opacity = "0"; }, 1800);
}

(async () => {
  startAmbient();
  initTabs();
  initCapture();
  const data = await loadData();
  window.__data = data;
  renderScenes(data);
  renderFinds(data);
  window.__ready = true;
  if ("serviceWorker" in navigator) { try { await navigator.serviceWorker.register("sw.js"); } catch (_) { /* ok */ } }
})();
