// tour.js -- one-button cinematic auto-orbit of the front hemisphere.
//
// For the viewer who won't drag: a smooth, slow sweep across the well-captured
// front arc (never behind a wall-flush shelf), with a gentle push-in and pull-out
// so spines read. Any user input (drag/scroll/key) cancels it. Also doubles as the
// "hero motion" the README wants without shipping a room screenshot.
//
// Integration (index.html): import { initTour } from "./tour.js"; create with the
// viewer's camera+controls and the scene's home view; wire a Tour button; call
// tour.onFrame(dt) in setAnimationLoop; call tour.cancel() from the existing
// user-interaction handlers.

export function initTour({ THREE, camera, controls, homePos, homeTarget, homeUp,
                           arcDeg = 55, periodSec = 22, zoomAmp = 0.18 }) {
  const target = new THREE.Vector3().fromArray(homeTarget);
  const up = new THREE.Vector3().fromArray(homeUp).normalize();
  const home = new THREE.Vector3().fromArray(homePos);
  const off0 = home.clone().sub(target);      // home camera offset from target
  const radius0 = off0.length();
  let active = false, t = 0;

  // Rotate the home offset by `deg` about the up axis, then scale radius by `zoom`.
  const _q = new THREE.Quaternion();
  function poseAt(phase) {
    // phase in [0,1) -> azimuth swings +/- arcDeg as a sine (ease at the ends),
    // radius breathes in/out one cycle behind it so the push-in lands mid-sweep.
    const az = Math.sin(phase * Math.PI * 2) * (arcDeg * Math.PI / 180);
    const zoom = 1 - zoomAmp * (0.5 - 0.5 * Math.cos(phase * Math.PI * 2)); // 1 -> 1-zoomAmp -> 1
    _q.setFromAxisAngle(up, az);
    const off = off0.clone().applyQuaternion(_q).multiplyScalar(zoom);
    return { pos: target.clone().add(off), tgt: target.clone() };
  }

  return {
    isActive: () => active,
    start() { if (active) return; active = true; t = 0; if (controls) controls.enableDamping = true; },
    cancel() {
      if (!active) return;
      active = false;
      // hand control back cleanly from wherever the camera is now (no snap)
      if (controls) controls.update();
    },
    toggle() { active ? this.cancel() : this.start(); },
    // dt seconds; returns true if it drove the camera this frame
    onFrame(dt) {
      if (!active) return false;
      t = (t + dt / periodSec) % 1;
      const { pos, tgt } = poseAt(t);
      camera.up.copy(up);
      camera.position.copy(pos);
      if (controls) controls.target.copy(tgt);
      camera.lookAt(tgt);
      if (controls) controls.update();
      return true;
    },
  };
}
