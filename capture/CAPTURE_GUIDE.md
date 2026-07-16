# Capture guide: phone -> 3D Gaussian splat

A field guide for shooting phone video that reconstructs cleanly into a 3D
Gaussian splat. Two jobs: an **object** (a bookshelf, shot outside-in) and an
**enclosure** (a whole office, shot inside-out). The enclosure is the harder,
more valuable case — read "why inside-out is harder" before your first attempt.

Everything downstream (frame extraction -> structure-from-motion -> splat
training) is only ever as good as the footage you hand it. A splat cannot invent
detail the camera never saw, and it cannot recover from inconsistent exposure or
missing parallax. **Twenty minutes of disciplined shooting saves hours of
reconstruction cleanup.**

## TL;DR

- **Lock everything the camera changes automatically:** exposure, white balance,
  focus, lens. Consistency across frames beats any single "good" frame.
- **Fast shutter, lots of light.** Motion blur is the number-one silent killer.
- **Walk, don't spin.** The camera must *translate* to create parallax. Standing
  in one spot and rotating gives zero depth and breaks pose estimation.
- **Overlap heavily (70-80%)** and do **2-3 loops at different heights.**
- **Aim for ~150-300 sharp, well-distributed frames** after culling.
- **Hostile surfaces** (mirrors, screens, windows, glossy floors, blank walls)
  each have a specific fix. Handle them before you shoot.

## The two capture modes

| | Object — bookshelf | Enclosure — the office |
|---|---|---|
| Camera position | Outside, looking **in** | Inside, looking **out** |
| Optical axes | **Converge** on the subject | **Diverge** toward the walls |
| Path | Arc / orbit around it | Loops around the perimeter |
| Difficulty | Forgiving | Demanding |
| Main risk | Missing the top/sides | Pose drift, blank walls, no loop closure |

### Why inside-out is harder (divergent vs convergent views)

Structure-from-Motion recovers 3D by triangulating the same feature seen from two
camera positions with a baseline between them. When you **orbit an object**, every
camera points inward at a shared subject — the views **converge**, the same
features appear in many frames from many angles, and triangulation is strongly
conditioned. Small errors wash out.

When you shoot an **enclosure from inside**, the cameras point outward and the
views **diverge** — each wall is seen from a narrow slice of positions, neighboring
frames share less, and the geometry is weakly conditioned. Errors accumulate as
*drift*: by the time you've gone around the room, the far wall may not line up with
where you started. The room also gives fewer long cross-scene sightlines to tie
distant regions together.

The fixes for divergence are baked into the enclosure workflow below: generous
overlap, deliberate **loop closure** (return to where you began and re-shoot it), a
**through-the-middle pass** that links opposite walls, and **anchoring on
feature-rich corners.**

## Phone settings

Shoot in a **manual / "Pro" video mode** if you have one (native Camera on many
Androids; on iPhone use a manual app such as Blackmagic Camera or Halide). The goal
is to stop the phone from silently re-deciding exposure, color, or focus.

| Setting | Target | Why |
|---|---|---|
| **Exposure (AE)** | **Locked** on a mid-tone. | Auto-exposure flicker makes one surface a different brightness frame to frame — confuses pose estimation, produces color-inconsistent, floaty splats. |
| **White balance** | **Locked** to a fixed Kelvin. | A wall must be one color across the whole capture. |
| **Focus** | **Locked** at your working distance. | Kills focus-hunting mid-walk. For enclosures, lock at a mid-room distance. |
| **Shutter** | **1/120 s** (1/60 if light is short) — never 1/250. | The single biggest defense against motion blur, and flicker-safe: LEDs pulse at 120 Hz, so only exposures spanning whole 8.33 ms cycles avoid banding — 1/120 (1 cycle), 1/60 (2). 1/250 is 0.48 cycles and bands. Raise ISO or add light to afford it. |
| **ISO** | As low as the fast shutter permits. | Noise also degrades feature matching; trade toward light, not ISO. |
| **Lens** | **Lock to the main (1x) lens.** Disable auto lens switching. | Phones silently swap to ultra-wide in tight spaces; a mid-capture focal-length change corrupts the camera model. |
| **Resolution / fps** | **4K/24-30.** | 4K for detail. Frame rate is not shutter: fps only caps the slowest allowed exposure, and frames are decimated to ~3 fps downstream, so 60 fps buys no usable frames — it just doubles the file. The proven capture ran 4K/24. |
| **Stabilization** | On is OK; **off is purer** with a gimbal. | Aggressive digital stabilization warps frame geometry slightly. |
| **HDR / cinematic / filters** | **Off.** | They remap tones per frame and destroy photometric consistency. |

**Clean the lens.** One smudge fogs every frame identically and there's no recovery.

## Lighting & scene prep

- **Bright, even, diffuse light.** Turn on every light. Diffuse beats hard
  directional light: hard light casts moving shadows and specular hotspots that
  shift with the camera and confuse reconstruction.
- **Kill moving content.** No people, pets, ceiling fans, or swaying blinds. Anything
  that moves becomes a ghost or floater. The scene must be **static** start to finish.
- **Pre-empt hostile surfaces** (next section) before you press record.
- Don't change the lighting mid-capture.

## The physical path

### Object — bookshelf (outside-in orbit)

1. **Frame it** filling ~60-80% of frame with some surrounding context (floor,
   adjacent wall) — that context gives anchor features and a stable coordinate frame.
2. **Arc around it, don't rotate it.** Move yourself; keep the shelf and background
   perfectly still. (Never use a turntable against a static background — the
   reconstruction can't tell what's moving.)
3. **Sweep a wide arc** — a shelf against a wall won't allow full 360°, so cover
   150-180°, front plus both oblique angles, so shelf depth and spines triangulate.
4. **Three heights, three passes:** low (angled up), eye level (primary), high
   (angled down).
5. **Small steps** — a fresh vantage every ~10-15° of arc, keeping **70-80% overlap.**
6. **Detail pass (optional):** slow close orbit of any intricate region (spine text).

### Enclosure — the office (inside-out walk)

This is a *walk*, not a *pan*.

1. **Perimeter loop, at arm's length from the walls.** Walk following the walls,
   camera pointed outward and slightly ahead, sweeping each wall as you pass. Keep
   translating — **parallax comes from your feet, not your wrists.**
   > **Cardinal rule: never stand still and spin.** Pure rotation gives no baseline
   > and no depth — the fastest way to a failed capture. To reveal a corner, *step*
   > around it.
2. **Three height loops** around the full perimeter: low (hip, angled up), eye level,
   high (overhead, angled down).
3. **Hit the edges deliberately.** On at least one pass, tilt up for the
   **ceiling and ceiling-wall seams**, down for the **floor and floor-wall seams** —
   where reconstructions most often tear, and feature-rich glue between walls.
4. **Anchor on corners.** Slow down and cover each corner from a couple of angles —
   they hold the walls in correct relative position.
5. **Through-the-middle / figure-8 pass.** Cross the room, linking **opposite walls in
   the same frames** to fight divergence drift.
6. **Close the loop.** Finish exactly where you started and **re-shoot the opening
   view** so pose estimation recognizes the return and corrects accumulated drift.
   Skipping this is the most common reason an office splat "banana-bends."
7. **Overlap 70-80%** throughout; move slowly — a new vantage every ~0.3-0.5 m.

## What kills a capture (and how to beat it)

| Killer | Why it breaks reconstruction | Work-around |
|---|---|---|
| **Mirrors** | Reflection is a virtual scene with fake parallax — features moving the wrong way. | Drape/remove, or keep out of frame; mask later if it must stay. |
| **TVs / screens** | Emissive, glossy, often changing. | **Turn them off**; drape large ones (a dark screen is still reflective). |
| **Windows / glass** | Transparency + reflection at incompatible depths; blows out exposure. | Close blinds; if it must show, shoot when interior/exterior brightness balance; mask if needed. |
| **Glossy floors** | Specular highlights slide as you move — read as floating geometry. | More diffuse light; a temporary matte rug; shoot from varied angles. |
| **Blank walls** | No features -> holes and drift. | Never shoot a blank wall alone — keep a textured anchor (corner, outlet, art) in frame; graze the wall; add temporary low-tack markers if allowed. |
| **Motion blur** | Smears the features the matcher needs. | Fast shutter + more light; walk slowly; brace/gimbal. |
| **Moving people/pets/fans** | Violates the static-scene assumption -> ghosts. | Clear the room; pause if something moves through. |
| **Rolling-shutter jello** | Fast phone-CMOS pans skew frame geometry. | Move deliberately; no whip pans. |

## How much to shoot

Target **~150-300 sharp, well-distributed frames** after culling.

- **Object (bookshelf):** ~60-120 s across the three height arcs.
- **Enclosure (office):** ~2-4 min across perimeter loops, the through-the-middle
  pass, and the loop closure.

At 30-60 fps that's thousands of raw frames; `pipeline/01_extract_frames.py`
subsamples for coverage and culls for sharpness down to the 150-300 band.

- **More footage is cheap insurance** — you can cull, but you can't recover an angle
  you never walked.
- **Even coverage beats raw count.** 200 frames across all walls/heights beat 400 on
  one favorite view.
- **Sharpness is non-negotiable** — a frame the filter rejects is a frame you didn't
  shoot.

## Pre-flight checklist

**Before you record**
- [ ] Lens wiped clean
- [ ] Manual/Pro mode; HDR, filters, cinematic **off**
- [ ] Exposure locked on a mid-tone
- [ ] White balance locked (fixed Kelvin)
- [ ] Focus locked at working distance
- [ ] Lens locked to 1x main; auto lens-switching disabled
- [ ] Shutter 1/120 s (or 1/60 — never 1/250); ISO as low as light allows
- [ ] 4K/24-30
- [ ] Lights on, bright and diffuse; no hard shadows/hotspots
- [ ] Screens off, blinds closed, mirrors draped
- [ ] Blank walls have a nearby texture anchor
- [ ] Room is static — no people/pets/fans moving
- [ ] Battery + free storage for 4K

**While recording**
- [ ] Walking, never spinning
- [ ] 70-80% overlap between viewpoints
- [ ] 2-3 loops at low / eye / high
- [ ] Corners and ceiling/floor edges covered
- [ ] Slow, smooth motion — no whip pans
- [ ] (Enclosure) through-the-middle pass done
- [ ] (Enclosure) loop closed — finished on the starting view

**Right after — check on the phone before you leave the room**
- [ ] Scrub the footage: consistent exposure, in focus, no obvious blur
- [ ] No unplanned moving objects
- [ ] Coverage feels complete — reshoot now, while the scene is still set

## Field QC — the human in the loop

The pipeline is automated; the capture is where human judgment earns its keep. Two
minutes reviewing footage **in the room**, while lights/blinds/props are still as you
shot them, is worth an hour of patching a broken splat later. Getting the input right
by hand is what makes the automated half trustworthy.
