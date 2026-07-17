// vr.js -- WebXR "Enter VR" for the splat viewer, via Spark's vendored SparkXr.
//
// Why a module: the capture lives in a tilted, unit-scaled frame (gsplat
// normalization). Desktop hides that by transforming the *camera* (the baked
// "view"). VR can't -- the headset owns the camera, so the *world* must be made
// upright, roughly life-sized, and floor-aligned, and the user placed inside it.
// This module does exactly that, plus the OrbitControls<->XR handoff.
//
// True fidelity (comfort, scale, framerate) is only judgeable in a real headset;
// headless can only confirm XR-readiness. So this is built to be correct and
// tested on a Quest, with a readiness path the harness can assert.
//
// Integration (done in index.html, one coordinated pass):
//   import { initVR } from "./vr.js";
//   const vr = initVR({ THREE, SparkXr, renderer, scene, camera, controls,
//                       splat, up: SCENE_UP, buttonEl: document.getElementById("btnVR") });
//   ... in setAnimationLoop, when presenting: vr.onFrame();

export function initVR({ THREE, SparkXr, renderer, scene, camera, controls, splat, up, buttonEl,
                         targetHeightMeters = 1.8, eyeHeightMeters = 1.6 }) {
  // --- 1. Make the world upright + life-sized under an xrGroup ----------------
  // The splat currently sits in `scene` (under its flipGroup). For XR we parent a
  // dedicated group that rotates capture-up -> world +Y, scales to ~life size, and
  // drops the floor to y=0. We only ENGAGE this on session start so desktop is
  // untouched.
  const xrGroup = new THREE.Group();
  xrGroup.visible = false;
  scene.add(xrGroup);

  const capUp = new THREE.Vector3(up[0], up[1], up[2]).normalize();
  const orient = new THREE.Quaternion().setFromUnitVectors(capUp, new THREE.Vector3(0, 1, 0));

  // Scale so the capture's vertical extent ~= targetHeightMeters (uncalibrated
  // guess until the 2-click metric calibration exists; feels roughly life-sized).
  function computeWorldTransform() {
    const box = splat.getBoundingBox ? splat.getBoundingBox(true) : new THREE.Box3().setFromObject(splat);
    // extent along capture-up
    const corners = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
    let lo = Infinity, hi = -Infinity;
    for (const c of corners) { const d = c.dot(capUp); lo = Math.min(lo, d); hi = Math.max(hi, d); }
    const upExtent = Math.max(hi - lo, 1e-4);
    const scale = targetHeightMeters / upExtent;
    return { scale, box, corners };
  }

  // --- 2. Camera rig: XR head pose + thumbstick locomotion move the rig -------
  const rig = new THREE.Group();
  scene.add(rig);
  let cameraHome = null; // desktop camera parent + local state to restore

  const spark = new SparkXr({
    renderer,
    mode: "vr",
    referenceSpaceType: "local-floor", // origin at the real floor, +Y real-up
    frameBufferScaleFactor: 0.5,        // half-res per eye -- Quest fill-rate
    fixedFoveation: 1.0,                // max fixed foveation (Quest 2 help)
    allowMobileXr: true,                // Quest browser reports mobile-ish UA
    element: buttonEl,                  // our HUD button, not Spark's default
    onReady: (supported) => {
      if (buttonEl) {
        buttonEl.style.display = supported ? "" : "none";
        buttonEl.classList.toggle("hidden", !supported);
      }
    },
    onEnterXr: () => onEnter(),
    onExitXr:  () => onExit(),
  });

  function onEnter() {
    // World: reparent the splat's group is invasive; instead we place an upright,
    // scaled COPY-of-transform on xrGroup and move the actual splat under it.
    const { scale } = computeWorldTransform();
    // Reparent splat under xrGroup with the upright transform.
    splat.__xrPrevParent = splat.parent;
    xrGroup.add(splat);
    xrGroup.quaternion.copy(orient);
    xrGroup.scale.setScalar(scale);
    xrGroup.visible = true;
    // Floor-align: after orient+scale, drop so the lowest point is y=0, and push
    // the room out in front of the standing user (-Z), centered laterally.
    xrGroup.position.set(0, 0, 0);
    xrGroup.updateWorldMatrix(true, true);
    const wbox = new THREE.Box3().setFromObject(xrGroup);
    xrGroup.position.y -= wbox.min.y;                          // floor to y=0
    const cx = (wbox.min.x + wbox.max.x) / 2, cz = (wbox.min.z + wbox.max.z) / 2;
    xrGroup.position.x -= cx;
    xrGroup.position.z -= cz - (wbox.max.z - wbox.min.z) * 0.5 - 0.6; // stand ~0.6m back
    // Camera into the rig; disable desktop controls.
    cameraHome = { parent: camera.parent, pos: camera.position.clone(), quat: camera.quaternion.clone(), up: camera.up.clone() };
    rig.add(camera);
    rig.position.set(0, 0, 0);
    if (controls) controls.enabled = false;
  }

  function onExit() {
    // Restore desktop: splat back to its group, camera back, controls on.
    if (splat.__xrPrevParent) { splat.__xrPrevParent.add(splat); splat.__xrPrevParent = null; }
    xrGroup.visible = false;
    xrGroup.position.set(0, 0, 0); xrGroup.quaternion.identity(); xrGroup.scale.setScalar(1);
    if (cameraHome) {
      (cameraHome.parent || scene).add(camera);
      camera.position.copy(cameraHome.pos);
      camera.quaternion.copy(cameraHome.quat);
      camera.up.copy(cameraHome.up);
    }
    if (controls) controls.enabled = true;
  }

  return {
    spark,
    isPresenting: () => renderer.xr.isPresenting,
    // call every frame while presenting (thumbstick locomotion moves camera.parent = rig)
    onFrame: () => { if (renderer.xr.isPresenting) spark.updateControllers(camera); },
  };
}
