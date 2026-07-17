// backdrop.js -- a solid dark-brown plane flush to the back of the shelf.
//
// The capture is a real shelf against a wall, so behind it is unreconstructed
// void: black, with stray floaters. This drops a world-fixed plane at the far
// extent of the content, facing forward, sized to over-cover the shelf -- so the
// gaps between books read as a back panel instead of black, and the animation
// (Quake/Blast) has something to fall against. World-fixed, not a billboard, so
// orbiting reads it as the shelf's real back wall.
//
// Placement comes from the fused object AABBs (clean -- no floaters) and the
// baked view's depth axis, so no bounding-box guessing on the noisy splat.
//
// Integration (index.html, in loadObjects once sceneView + raw are known):
//   import { initBackdrop } from "./backdrop.js";
//   const backdrop = initBackdrop({ THREE, parent: flipGroup, objects: raw, view: sceneView });

export function initBackdrop({ THREE, parent, objects, view,
                              color = 0x2b1f14, pad = 1.2, behind = 0.03 }) {
  if (!parent || !objects || !objects.length || !view ||
      !Array.isArray(view.position) || !Array.isArray(view.target)) {
    return { mesh: null };
  }

  // content box from object AABBs (fall back to positions if an object lacks one)
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (const o of objects) {
    if (o.aabb && o.aabb.min && o.aabb.max) {
      box.expandByPoint(v.fromArray(o.aabb.min));
      box.expandByPoint(v.fromArray(o.aabb.max));
    } else if (Array.isArray(o.position)) {
      box.expandByPoint(v.fromArray(o.position));
    }
  }
  if (box.isEmpty()) return { mesh: null };

  // orthonormal frame from the baked view: fwd = into the scene (front -> back)
  const pos = new THREE.Vector3().fromArray(view.position);
  const tgt = new THREE.Vector3().fromArray(view.target);
  const up = Array.isArray(view.up) ? new THREE.Vector3().fromArray(view.up).normalize()
                                    : new THREE.Vector3(0, 1, 0);
  const fwd = new THREE.Vector3().subVectors(tgt, pos).normalize();
  const right = new THREE.Vector3().crossVectors(fwd, up).normalize();
  const trueUp = new THREE.Vector3().crossVectors(right, fwd).normalize();

  // project the 8 corners onto each axis to get oriented extents
  const corners = [];
  for (const x of [box.min.x, box.max.x])
    for (const y of [box.min.y, box.max.y])
      for (const z of [box.min.z, box.max.z])
        corners.push(new THREE.Vector3(x, y, z));
  const span = (axis) => {
    let lo = Infinity, hi = -Infinity;
    for (const c of corners) { const d = c.dot(axis); lo = Math.min(lo, d); hi = Math.max(hi, d); }
    return { lo, hi, mid: (lo + hi) / 2, ext: hi - lo };
  };
  const sf = span(fwd), sr = span(right), su = span(trueUp);

  // center on the back face, pushed a little further back so it never z-fights
  // with the rear-most splats
  const center = box.getCenter(new THREE.Vector3());
  center.addScaledVector(fwd, (sf.hi - sf.mid) + behind * sf.ext);

  const geo = new THREE.PlaneGeometry(sr.ext * pad, su.ext * pad);
  const mat = new THREE.MeshBasicMaterial({
    color, side: THREE.DoubleSide, depthWrite: false, depthTest: true,
    toneMapped: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(center);
  // plane normal (+Z) faces the camera (-fwd); X->right, Y->trueUp
  mesh.quaternion.setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(right, trueUp, fwd.clone().negate()));
  mesh.renderOrder = -1;          // draw before the splat, so it sits behind
  mesh.name = "shelf-backdrop";
  parent.add(mesh);

  return {
    mesh,
    setVisible: (on) => { mesh.visible = on; },
    setColor: (c) => mesh.material.color.set(c),
  };
}
