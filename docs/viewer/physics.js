// Rigid-body physics for the splat viewer.
//
// The understanding branch already segmented the scene into 129 objects
// (assets/_anim_objects.json: id, category, centroid, aabb, all in the SPLAT's
// own gsplat-normalized frame). This module turns that segmentation into a real
// physics demo: each dynamic object becomes a cannon-es rigid body, and a GPU
// per-splat "objectModifier" (Spark dyno) rigidly transforms every Gaussian by
// its object's live body pose each frame. Two triggers -- Quake (shake the
// contents off the shelf) and Blast (fling them radially) -- plus Reset.
//
// The shelf and the un-segmented background are id 0 = static, so the shelf
// frame stays put while its contents tumble to a floor under gravity.
//
// Frame facts (from the build spec, verified against the asset):
//   UP   = [0.82412, -0.52228, 0.21921]  (the scene is TILTED; up is not axis-aligned)
//   DOWN = -UP  (gravity direction)
// Everything here works in the object/PLY frame -- exactly what objectModifier
// sees -- so object centroids feed straight in with no ?flip applied.

import * as THREE from "three";
import { dyno } from "@sparkjsdev/spark";
import * as CANNON from "./vendor/cannon-es.js";

const { Dyno, dynoBlock, Gsplat, unindent, unindentLines,
        dynoUsampler2D, dynoSampler2D } = dyno;

const UP_RAW = [0.82412, -0.52228, 0.21921];
const GRAVITY = 9.8;                 // scene units ~= metres (scene spans ~2.8u)
const ID_TEX_W = 2048;               // idTex row width (must match the GLSL literal)

// Only these tumble. "shelf" and anything in no object AABB stay static (id 0).
const DYNAMIC_CATEGORIES = new Set([
  "book", "figurine", "lego model", "box", "statue", "owl", "picture frame",
]);

// ---------------------------------------------------------------------------
// Public entry point. Resolves once bodies + GPU modifier are installed.
// ---------------------------------------------------------------------------
export async function installPhysics(splat, objectsUrl = "./assets/_anim_objects.json") {
  const res = await fetch(objectsUrl, { cache: "no-cache" });
  if (!res.ok) throw new Error(`physics: ${res.status} fetching ${objectsUrl}`);
  const doc = await res.json();
  const objects = doc.objects || [];

  // ---- frame basis: up, down, and two horizontal tangents for the quake shake
  const up = new THREE.Vector3().fromArray(UP_RAW).normalize();
  const down = up.clone().multiplyScalar(-1);
  const helper = Math.abs(up.x) < 0.9 ? new THREE.Vector3(1, 0, 0)
                                      : new THREE.Vector3(0, 1, 0);
  const tan1 = new THREE.Vector3().crossVectors(up, helper).normalize();
  const tan2 = new THREE.Vector3().crossVectors(up, tan1).normalize();

  // ---- scene bounds / centroid / floor level (projection onto up) -----------
  const sMin = doc.scene_min, sMax = doc.scene_max;
  const sceneCentroid = new THREE.Vector3(
    (sMin[0] + sMax[0]) / 2, (sMin[1] + sMax[1]) / 2, (sMin[2] + sMax[2]) / 2);
  // Floor: the level tumbled objects settle on. Derive it from the lowest fused
  // OBJECT (its aabb bottom), NOT the raw scene box -- the scene box dips well
  // below the shelf into floaters and void, which dropped the floor beneath the
  // real floor of the room and let things pile into a pit under the shelf. The
  // lowest object bottom is the bottom-shelf resting surface, ~ the room floor.
  let floorProj = Infinity;
  const _fc = new THREE.Vector3();
  for (const o of objects) {
    if (!o.aabb) continue;
    const mn = o.aabb.min, mx = o.aabb.max;
    for (const x of [mn[0], mx[0]])
      for (const y of [mn[1], mx[1]])
        for (const z of [mn[2], mx[2]])
          floorProj = Math.min(floorProj, _fc.set(x, y, z).dot(up));
  }
  if (!isFinite(floorProj)) floorProj = sMin.reduce((a, b, i) => a + b * up.getComponent(i), 0);
  floorProj -= 0.02; // a hair of clearance so things land, not clip

  // ---- assign dynamic ids (1..K); static/background = 0 ---------------------
  // objMeta[i] describes object i from the json: its aabb (for segmentation),
  // its volume (smallest-containing tiebreak), and the id splats get (0 if static).
  const objMeta = objects.map((o) => {
    const mn = o.aabb.min, mx = o.aabb.max;
    const vol = Math.max(mx[0] - mn[0], 1e-4)
              * Math.max(mx[1] - mn[1], 1e-4)
              * Math.max(mx[2] - mn[2], 1e-4);
    return { mn, mx, vol, dynamic: DYNAMIC_CATEGORIES.has(o.category),
             centroid: o.centroid, category: o.category, id: 0 };
  });
  const dynamicMeta = [];
  for (const m of objMeta) {
    if (m.dynamic) { m.id = dynamicMeta.length + 1; dynamicMeta.push(m); }
  }
  const numDynamic = dynamicMeta.length;
  const numCols = numDynamic + 1; // objTex column 0 is the unused/static slot

  // ---- spatial grid over object AABBs, so segmentation is O(N) not O(N*129) --
  const gMin = [Infinity, Infinity, Infinity], gMax = [-Infinity, -Infinity, -Infinity];
  for (const m of objMeta)
    for (let a = 0; a < 3; a++) {
      gMin[a] = Math.min(gMin[a], m.mn[a]); gMax[a] = Math.max(gMax[a], m.mx[a]);
    }
  const CELL = 0.12;
  const dims = [
    Math.max(1, Math.ceil((gMax[0] - gMin[0]) / CELL)),
    Math.max(1, Math.ceil((gMax[1] - gMin[1]) / CELL)),
    Math.max(1, Math.ceil((gMax[2] - gMin[2]) / CELL)),
  ];
  const cellOf = (p, a) =>
    Math.floor((p - gMin[a]) / CELL);
  const grid = new Map(); // key "x,y,z" -> array of objMeta indices
  for (let oi = 0; oi < objMeta.length; oi++) {
    const m = objMeta[oi];
    const c0 = [cellOf(m.mn[0], 0), cellOf(m.mn[1], 1), cellOf(m.mn[2], 2)];
    const c1 = [cellOf(m.mx[0], 0), cellOf(m.mx[1], 1), cellOf(m.mx[2], 2)];
    for (let x = c0[0]; x <= c1[0]; x++)
      for (let y = c0[1]; y <= c1[1]; y++)
        for (let z = c0[2]; z <= c1[2]; z++) {
          const key = x + "," + y + "," + z;
          let arr = grid.get(key);
          if (!arr) grid.set(key, arr = []);
          arr.push(oi);
        }
  }

  // ---- segment every splat: idData[i] = object id of splat i (load order) ----
  const N = splat.numSplats
         ?? splat.packedSplats?.numSplats
         ?? splat.splats?.numSplats
         ?? 0;
  const H = Math.max(1, Math.ceil(N / ID_TEX_W));
  const idData = new Uint32Array(ID_TEX_W * H); // zero = background/static
  let assigned = 0, maxIndex = -1;
  const t0 = performance.now();
  splat.forEachSplat((i, center) => {
    if (i > maxIndex) maxIndex = i;
    const cx = center.x, cy = center.y, cz = center.z;
    const gx = Math.floor((cx - gMin[0]) / CELL);
    const gy = Math.floor((cy - gMin[1]) / CELL);
    const gz = Math.floor((cz - gMin[2]) / CELL);
    if (gx < 0 || gy < 0 || gz < 0 || gx >= dims[0] || gy >= dims[1] || gz >= dims[2]) return;
    const cand = grid.get(gx + "," + gy + "," + gz);
    if (!cand) return;
    let bestVol = Infinity, bestId = 0;
    for (let k = 0; k < cand.length; k++) {
      const m = objMeta[cand[k]];
      if (cx < m.mn[0] || cx > m.mx[0] ||
          cy < m.mn[1] || cy > m.mx[1] ||
          cz < m.mn[2] || cz > m.mx[2]) continue;
      if (m.vol < bestVol) { bestVol = m.vol; bestId = m.id; }
    }
    if (bestId !== 0) { idData[i] = bestId; assigned++; }
  });
  const segMs = Math.round(performance.now() - t0);

  const idTex = new THREE.DataTexture(
    idData, ID_TEX_W, H, THREE.RedIntegerFormat, THREE.UnsignedIntType);
  idTex.internalFormat = "R32UI";
  idTex.minFilter = idTex.magFilter = THREE.NearestFilter;
  idTex.needsUpdate = true;

  // ---- objTex: per object, row0 = initial centroid, row1 = live position,
  //      row2 = live quaternion (x,y,z,w). Column = object id. ----------------
  const objData = new Float32Array(numCols * 3 * 4);
  const setRow = (row, col, x, y, z, w) => {
    const b = (row * numCols + col) * 4;
    objData[b] = x; objData[b + 1] = y; objData[b + 2] = z; objData[b + 3] = w;
  };
  // column 0 (static): identity so any stray id-0 lookup is a pass-through
  setRow(2, 0, 0, 0, 0, 1);

  // ---- cannon world + bodies -----------------------------------------------
  const world = new CANNON.World();
  world.gravity.set(down.x * GRAVITY, down.y * GRAVITY, down.z * GRAVITY);
  world.allowSleep = true;
  world.defaultContactMaterial.friction = 0.4;
  world.defaultContactMaterial.restitution = 0.15;
  world.solver.iterations = 12;

  // floor: infinite plane, world normal = up, at floorProj along up
  const floorBody = new CANNON.Body({ mass: 0 });
  floorBody.addShape(new CANNON.Plane());
  floorBody.quaternion.setFromVectors(new CANNON.Vec3(0, 0, 1),
    new CANNON.Vec3(up.x, up.y, up.z));
  floorBody.position.set(up.x * floorProj, up.y * floorProj, up.z * floorProj);
  world.addBody(floorBody);

  const bodies = []; // parallel to dynamicMeta; body.userData.col = objTex column
  for (const m of dynamicMeta) {
    const he = new CANNON.Vec3(
      Math.max((m.mx[0] - m.mn[0]) / 2, 0.02),
      Math.max((m.mx[1] - m.mn[1]) / 2, 0.02),
      Math.max((m.mx[2] - m.mn[2]) / 2, 0.02));
    const vol = 8 * he.x * he.y * he.z;
    const body = new CANNON.Body({
      mass: Math.max(vol, 0.02),
      position: new CANNON.Vec3(m.centroid[0], m.centroid[1], m.centroid[2]),
      allowSleep: true,
    });
    // centre the collision box on the geometric aabb, keep body origin = centroid
    const aabbCenter = new CANNON.Vec3(
      (m.mn[0] + m.mx[0]) / 2 - m.centroid[0],
      (m.mn[1] + m.mx[1]) / 2 - m.centroid[1],
      (m.mn[2] + m.mx[2]) / 2 - m.centroid[2]);
    body.addShape(new CANNON.Box(he), aabbCenter);
    body.linearDamping = 0.05;
    body.angularDamping = 0.12;
    body.sleepSpeedLimit = 0.04;
    body.sleepTimeLimit = 0.4;
    body.userData = { col: m.id, home: m.centroid.slice() };
    body.sleep(); // frozen until a trigger wakes it
    world.addBody(body);
    bodies.push(body);
    // objTex rest state for this column
    setRow(0, m.id, m.centroid[0], m.centroid[1], m.centroid[2], 0);
    setRow(1, m.id, m.centroid[0], m.centroid[1], m.centroid[2], 0);
    setRow(2, m.id, 0, 0, 0, 1);
  }

  const objTex = new THREE.DataTexture(
    objData, numCols, 3, THREE.RGBAFormat, THREE.FloatType);
  objTex.minFilter = objTex.magFilter = THREE.NearestFilter;
  objTex.needsUpdate = true;

  // ---- GPU objectModifier: rigid-transform each Gaussian by its body pose ----
  const idU = dynoUsampler2D(idTex, "physIdTex");
  const objU = dynoSampler2D(objTex, "physObjTex");

  function makeModifier() {
    return dynoBlock({ gsplat: Gsplat }, { gsplat: Gsplat }, ({ gsplat }) => {
      const d = new Dyno({
        inTypes: { gsplat: Gsplat, idTex: "usampler2D", objTex: "sampler2D" },
        outTypes: { gsplat: Gsplat },
        globals: () => [unindent(`
          vec3 phys_qrot(vec4 q, vec3 v) {
            vec3 u = q.xyz; float s = q.w;
            return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
          }
          vec4 phys_qmul(vec4 a, vec4 b) {
            return vec4(
              a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
              a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
              a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
              a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
            );
          }
        `)],
        statements: ({ inputs, outputs }) => unindentLines(`
          ${outputs.gsplat} = ${inputs.gsplat};
          int phys_idx = ${inputs.gsplat}.index;
          ivec2 phys_it = ivec2(phys_idx % ${ID_TEX_W}, phys_idx / ${ID_TEX_W});
          uint phys_oid = texelFetch(${inputs.idTex}, phys_it, 0).r;
          if (phys_oid != 0u) {
            int phys_c = int(phys_oid);
            vec3 phys_c0 = texelFetch(${inputs.objTex}, ivec2(phys_c, 0), 0).xyz;
            vec3 phys_p  = texelFetch(${inputs.objTex}, ivec2(phys_c, 1), 0).xyz;
            vec4 phys_q  = texelFetch(${inputs.objTex}, ivec2(phys_c, 2), 0);
            vec3 phys_rel = ${inputs.gsplat}.center - phys_c0;
            ${outputs.gsplat}.center = phys_p + phys_qrot(phys_q, phys_rel);
            ${outputs.gsplat}.quaternion = phys_qmul(phys_q, ${inputs.gsplat}.quaternion);
          }
        `),
      });
      return { gsplat: d.apply({ gsplat, idTex: idU, objTex: objU }).gsplat };
    });
  }

  // Bind the modifier into the splat pipeline. Doing this once synchronously in
  // onLoad is too early: Spark finishes constructing the SplatMesh generator
  // *after* the load callback, and a modifier attached before that never binds
  // (the bake keeps running the default pass-through and the objTex sampler is
  // ignored). Re-attaching on the next couple of animation frames -- once the
  // mesh is fully initialised and has rendered -- makes it stick. Verified
  // headless: without the deferred rebind, triggers move the bodies but the
  // splats never budge; with it, the same triggers displace them correctly.
  function bindModifier() {
    splat.objectModifier = makeModifier();
    splat.updateGenerator();
    splat.updateVersion();
  }
  bindModifier();
  requestAnimationFrame(() => { bindModifier(); requestAnimationFrame(bindModifier); });
  // The install-time binds above cover the common case, but the exact frame on
  // which Spark's generator settles is timing-dependent, so also (re)bind on the
  // first trigger -- by then the mesh has rendered for real and the bind is
  // guaranteed to stick. Cheap (one cached recompile) and idempotent.
  let boundAtTrigger = false;
  function ensureBound() { if (!boundAtTrigger) { boundAtTrigger = true; bindModifier(); } }

  // ---- runtime state --------------------------------------------------------
  let active = false;
  let quakeT = 0;            // seconds remaining in the shake window
  let settleTicks = 0;
  let flushFrames = 0;       // reset: no-step re-bakes to flush the rest pose
  const FIXED_DT = 1 / 60;
  const QUAKE_DUR = 1.6;
  const QUAKE_SHAKE = 2.6;   // peak horizontal shake speed (units/s)

  const _t1 = new THREE.Vector3(), _t2 = new THREE.Vector3(), _v = new THREE.Vector3();

  function writePose() {
    for (const b of bodies) {
      const col = b.userData.col;
      const p = b.position, q = b.quaternion;
      setRow(1, col, p.x, p.y, p.z, 0);
      setRow(2, col, q.x, q.y, q.z, q.w);
    }
    objTex.needsUpdate = true;
    splat.updateVersion();
  }

  function wakeAll() {
    for (const b of bodies) b.wakeUp();
  }

  function quake() {
    ensureBound();
    wakeAll();
    for (const b of bodies) {
      // a kick of random tumble; the per-tick shake does the lateral motion
      b.angularVelocity.set(
        (Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6);
    }
    quakeT = QUAKE_DUR;
    settleTicks = 0;
    flushFrames = 0;
    active = true;
    writePose();
  }

  function blast() {
    ensureBound();
    wakeAll();
    for (const b of bodies) {
      const p = b.position;
      _v.set(p.x - sceneCentroid.x, p.y - sceneCentroid.y, p.z - sceneCentroid.z);
      if (_v.lengthSq() < 1e-6) _v.copy(up);
      _v.normalize();
      _v.addScaledVector(up, 0.55).normalize();          // upward/outward arc
      const speed = 2.2 + Math.random() * 1.8;
      b.velocity.set(_v.x * speed, _v.y * speed, _v.z * speed);
      b.angularVelocity.set(
        (Math.random() - 0.5) * 10, (Math.random() - 0.5) * 10, (Math.random() - 0.5) * 10);
    }
    quakeT = 0;
    settleTicks = 0;
    flushFrames = 0;
    active = true;
    writePose();
  }

  function reset() {
    ensureBound();
    for (const b of bodies) {
      const h = b.userData.home;
      b.position.set(h[0], h[1], h[2]);
      b.quaternion.set(0, 0, 0, 1);
      b.velocity.setZero();
      b.angularVelocity.setZero();
      b.sleep();
    }
    quakeT = 0;
    // Flush the rest pose over several frames rather than a single re-bake: a
    // lone updateVersion() can race the objTex upload (the bake may run before
    // three re-uploads the texture), leaving the scattered frame baked. The
    // animating paths bump every frame so they self-correct; a one-shot reset
    // does not, so we hold `active` for a few no-step frames to guarantee the
    // rest state lands.
    flushFrames = 6;
    active = true;
    writePose();
  }

  // Advance the simulation one fixed step and re-bake. Returns whether still active.
  function tick() {
    if (!active) return false;

    // reset flush: re-bake the (static) rest pose a few times, then go idle
    if (flushFrames > 0) {
      writePose();
      if (--flushFrames === 0) active = false;
      return active;
    }

    if (quakeT > 0) {
      const phase = QUAKE_DUR - quakeT;
      const ramp = quakeT / QUAKE_DUR;                    // decays to 0
      const amp = QUAKE_SHAKE * ramp;
      let bi = 0;
      for (const b of bodies) {
        b.wakeUp();
        const ph = bi * 1.7;
        const s1 = Math.sin(phase * 34 + ph) * amp;
        const s2 = Math.sin(phase * 27 + ph * 1.4) * amp;
        // keep the along-up velocity (gravity), overwrite the horizontal part
        const vUp = up.dot(_v.copy(b.velocity));
        _t1.copy(tan1).multiplyScalar(s1);
        _t2.copy(tan2).multiplyScalar(s2);
        b.velocity.set(
          up.x * vUp + _t1.x + _t2.x,
          up.y * vUp + _t1.y + _t2.y,
          up.z * vUp + _t1.z + _t2.z);
        bi++;
      }
      quakeT -= FIXED_DT;
    }

    world.step(FIXED_DT);
    writePose();

    // settle detection: once the shake is done and everything is slow, stop
    if (quakeT <= 0) {
      let maxSpeed = 0;
      for (const b of bodies) maxSpeed = Math.max(maxSpeed, b.velocity.length());
      if (maxSpeed < 0.05) settleTicks++; else settleTicks = 0;
      if (settleTicks > 40) { active = false; }
    }
    return active;
  }

  return {
    quake, blast, reset, tick,
    get active() { return active; },
    stats: {
      numSplats: N, maxIndex, assigned, numDynamic, numCols,
      floorProj, segMs, coveragePct: N ? +(100 * assigned / N).toFixed(2) : 0,
    },
    _internal: { world, bodies, idTex, objTex, up, down },
  };
}
