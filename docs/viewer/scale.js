// SPDX-License-Identifier: MIT
//
// Metric calibration for an up-to-scale reconstruction.
//
// Structure-from-motion recovers geometry but not size: translate every camera twice
// as far apart and the photos are identical, so the solve is correct up to one unknown
// scalar. A splat therefore measures in "scene units" that mean nothing on their own,
// which is why photoreal capture is routinely called "not measurable".
//
// One known real length fixes the scalar for the whole scene. Photograph something whose
// size you know -- a tape measure laid in the shot is the surveyor's habit, and it costs
// nothing -- measure it in the viewer, type in what it really is, and every subsequent
// measurement is in real units. That is the entire method. It is not clever; it is just
// the step everyone skips.
//
// Kept as a separate module with no three.js import so the arithmetic is testable in
// node without a browser or a GPU. The viewer does the picking; this does the numbers.

// Everything converts through metres. Feet and inches are first-class because the
// intended users think in them; refusing imperial would be a purity that costs a user.
export const UNITS = {
  mm: { label: "mm", perMetre: 1000, decimals: 0 },
  cm: { label: "cm", perMetre: 100, decimals: 1 },
  m: { label: "m", perMetre: 1, decimals: 3 },
  in: { label: "in", perMetre: 39.3700787402, decimals: 1 },
  ft: { label: "ft", perMetre: 3.28083989501, decimals: 2 },
};

export const DEFAULT_UNIT = "in";

/** Metres per scene unit, from one measurement of a known length. */
export function metresPerUnit(sceneDistance, realLength, unit) {
  const u = UNITS[unit];
  if (!u) throw new Error(`unknown unit: ${unit}`);
  if (!(sceneDistance > 0) || !Number.isFinite(sceneDistance)) {
    throw new Error("scene distance must be a positive number");
  }
  if (!(realLength > 0) || !Number.isFinite(realLength)) {
    throw new Error("real length must be a positive number");
  }
  return (realLength / u.perMetre) / sceneDistance;
}

/**
 * Relative error on any measurement made with this calibration.
 *
 * The honest part. Accuracy is set by how precisely the two reference points can be
 * clicked, and that click error is a fixed fraction of the *reference* length: calibrate
 * against a 24-inch tape and a 3mm pick error is 1-in-200; calibrate against the 72-inch
 * shelf the tape is lying against and the same pick error is 1-in-600. Longer baseline,
 * proportionally lower error -- which is why a surveyor reaches for the longest reference
 * in the room. Reported so nobody reads three decimals as three decimals of truth.
 */
export function relativeError(sceneDistanceOfReference, pickErrorScene) {
  if (!(sceneDistanceOfReference > 0)) return Infinity;
  // Two independent picks, each uncertain: errors add in quadrature, not linearly.
  return (Math.SQRT2 * pickErrorScene) / sceneDistanceOfReference;
}

/** Format a scene-unit distance for display. Uncalibrated stays honest about it. */
export function formatDistance(sceneDistance, cal) {
  if (!cal) return `${sceneDistance.toFixed(3)} scene units`;
  const u = UNITS[cal.unit] || UNITS[DEFAULT_UNIT];
  const metres = sceneDistance * cal.metresPerUnit;
  const v = metres * u.perMetre;
  // Feet read badly as decimals to anyone who works in them: 6.42 ft is not a thing
  // you say out loud, 6' 5" is.
  if (cal.unit === "ft") {
    let whole = Math.floor(v);
    let inches = (v - whole) * 12;
    // Round before splitting, not after: 5.99999 ft floors to 5 with 11.99999 inches
    // left over, which renders as 5' 12.0". Carry it.
    if (Number(inches.toFixed(1)) >= 12) { whole += 1; inches = 0; }
    return `${whole}' ${inches.toFixed(1)}"`;
  }
  return `${v.toFixed(u.decimals)} ${u.label}`;
}

/**
 * Round-trip the calibration through the URL so a shared link stays calibrated.
 * Shareability is the point: a measurement nobody else can reproduce is a party trick.
 * Encoded as the raw scalar + the unit to display in, not the reference -- the
 * reference has done its job by then.
 */
export function encodeCalibration(cal) {
  return `${cal.metresPerUnit.toPrecision(10)},${cal.unit}`;
}

export function decodeCalibration(param) {
  if (!param) return null;
  const [mpu, unit] = String(param).split(",");
  const v = Number.parseFloat(mpu);
  if (!Number.isFinite(v) || v <= 0) return null;
  return { metresPerUnit: v, unit: UNITS[unit] ? unit : DEFAULT_UNIT };
}
