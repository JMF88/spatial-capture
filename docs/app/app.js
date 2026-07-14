// Trove -- capture-and-collect companion to spatial-capture. Vanilla, self-contained.

// Embedded fallback so the app works even opened from file:// ; the real data
// source is assets/collection.json (populated as you capture + fuse scenes).
const DEMO = {
  scenes: [
    { id: "office", name: "Home Office", mode: "enclosure", captured: "2026-07-12", hue: 40,
      splat: "../viewer/assets/scene.spz", scene: "../viewer/assets/scene.json",
      stats: { gaussians: 1180000, objects: 14, frames: 212 } },
    { id: "bookshelf", name: "Office Bookshelf", mode: "object", captured: "2026-07-12", hue: 150,
      splat: "../viewer/assets/bookshelf.spz", scene: "../viewer/assets/bookshelf.json",
      stats: { gaussians: 640000, objects: 22, frames: 96 } },
    { id: "garage", name: "Garage Bench", mode: "object", captured: "2026-07-10", hue: 205,
      splat: null, scene: null,
      stats: { gaussians: 820000, objects: 9, frames: 140 } },
  ],
  finds: [
    { label: "The Pragmatic Programmer", category: "book", scene: "Office Bookshelf", confidence: 0.91 },
    { label: "Designing Data-Intensive Apps", category: "book", scene: "Office Bookshelf", confidence: 0.88 },
    { label: "Clean Code", category: "book", scene: "Office Bookshelf", confidence: 0.86 },
    { label: "potted plant", category: "plant", scene: "Home Office", confidence: 0.79 },
    { label: "desk lamp", category: "lamp", scene: "Home Office", confidence: 0.84 },
    { label: "monitor", category: "monitor", scene: "Home Office", confidence: 0.93 },
    { label: "mechanical keyboard", category: "keyboard", scene: "Home Office", confidence: 0.77 },
    { label: "coffee mug", category: "mug", scene: "Home Office", confidence: 0.68 },
    { label: "office chair", category: "chair", scene: "Home Office", confidence: 0.90 },
    { label: "cordless drill", category: "tool", scene: "Garage Bench", confidence: 0.81 },
    { label: "socket set", category: "tool", scene: "Garage Bench", confidence: 0.72 },
    { label: "bench vise", category: "tool", scene: "Garage Bench", confidence: 0.74 },
  ],
};

async function loadData() {
  try {
    const r = await fetch("assets/collection.json", { cache: "no-cache" });
    if (r.ok) return await r.json();
  } catch (_) { /* fall through to embedded demo */ }
  return DEMO;
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
  const grid = document.getElementById("scene-grid");
  document.getElementById("scene-count").textContent = data.scenes.length + " scenes";
  grid.innerHTML = "";
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
  const cats = ["all", ...Array.from(new Set(finds.map(f => f.category)))];
  const fbar = document.getElementById("filters");
  fbar.innerHTML = "";
  for (const cat of cats) {
    const b = document.createElement("button");
    b.className = "chip" + (cat === activeCat ? " on" : "");
    b.textContent = cat;
    b.onclick = () => { activeCat = cat; renderFinds(data); };
    fbar.appendChild(b);
  }
  const wrap = document.getElementById("finds");
  const shown = finds.filter(f => activeCat === "all" || f.category === activeCat);
  wrap.innerHTML = "";
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

function initCapture() {
  document.querySelectorAll("#mode-seg button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll("#mode-seg button").forEach(x => x.classList.remove("on"));
      b.classList.add("on"); mode = b.dataset.mode; renderChecklist();
    };
  });
  const start = async () => {
    const hint = document.getElementById("record-hint");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      const v = document.getElementById("cam");
      v.srcObject = stream; v.style.display = "block"; v.play();
      hint.textContent = "recording — walk your capture, then feed the video to the pipeline";
    } catch (_) {
      hint.textContent = "camera unavailable here — open Trove on your phone to capture";
    }
  };
  const rec = document.getElementById("record");
  rec.onclick = start;
  rec.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); start(); } };
  renderChecklist();
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
