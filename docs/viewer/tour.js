// tour.js -- one-button cinematic auto-orbit of the front hemisphere.
//
// For the viewer who won't drag: a smooth, slow sweep across the well-captured
// front arc (never behind a wall-flush shelf), with a gentle push-in/out so
// spines read. Captures the CURRENT camera pose on start() (which is already the
// upright baked view), so it needs no scene metadata and no load-timing dance.
// Any user input cancels it (wire cancel() into the interaction handlers).
//
// Integration (index.html):
//   import { initTour } from "./tour.js";
//   const tour = initTour({ THREE, camera, controls });
//   btnTour.onclick = () => tour.toggle();
//   ... in setAnimationLoop: tour.onFrame(dt);
//   ... in each user-interaction handler: tour.cancel();

export function initTour({ THREE, camera, controls, arcDeg = 55, periodSec = 24, zoomAmp = 0.16 }) {
  let active = false, t = 0;
  let target = null, off0 = null, up = null;
  const _q = new THREE.Quaternion();

  return {
    isActive: () => active,
    start() {
      if (active) return;
      // capture the pose we're leaving from as the tour's home
      target = controls.target.clone();
      off0 = camera.position.clone().sub(target);
      up = camera.up.clone().normalize();
      t = 0; active = true;
    },
    cancel() {
      if (!active) return;
      active = false;
      if (controls) controls.update();   // hand back from wherever we are, no snap
    },
    toggle() { active ? this.cancel() : this.start(); },
    // dt seconds; returns true if it drove the camera this frame
    onFrame(dt) {
      if (!active) return false;
      t = (t + dt / periodSec) % 1;
      const az = Math.sin(t * Math.PI * 2) * (arcDeg * Math.PI / 180);       // swing +/-arc
      const zoom = 1 - zoomAmp * (0.5 - 0.5 * Math.cos(t * Math.PI * 2));    // breathe in then out
      _q.setFromAxisAngle(up, az);
      const off = off0.clone().applyQuaternion(_q).multiplyScalar(zoom);
      camera.up.copy(up);
      camera.position.copy(target).add(off);
      if (controls) controls.target.copy(target);
      camera.lookAt(target);
      if (controls) controls.update();
      return true;
    },
  };
}
