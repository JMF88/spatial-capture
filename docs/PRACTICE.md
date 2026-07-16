# The practice

Capturing a real space with a phone and reconstructing it as a 3D Gaussian splat, written
down properly. Two kinds of claim appear here and they are kept apart on purpose:

- **Measured here** — numbers from this project's own captures of one bookshelf in one
  room. Real, checkable, and narrow. One room is a data point, not a calibration.
- **Literature** — cited with a URL and a retrieval date. Where it contradicts what we
  measured, the contradiction is stated rather than smoothed over. Those are the most
  useful paragraphs in the document.

Anything that is neither is folklore, and is either named as folklore or cut.

The spine of it is one comparison. Two captures of the same shelf, same operator, same
room, same path — one with the camera on auto, one with it locked.

Brightness is reported two ways and they must not be mixed, because mixing them inflates
the result by an order of magnitude. **Raw spread** is how far frame brightness ranged
across the take — it includes the room genuinely being brighter at one end. **Detrended
wobble** is what is left after removing that smooth trend, and it is the part attributable
to the camera. A fact-check pass caught an earlier draft of this very table setting auto's
*raw* 92% beside locked's *detrended* 2.2% and calling it a hundredfold win. It is not.

| | take 1 (auto) | take 2 (locked) |
|---|---|---|
| brightness, **raw spread** | 92% / 106% | 15-59% |
| brightness, **detrended wobble** | **18.0% / 16.1%** | **2.2-6.2%** |
| shape | oscillating — a meter chasing itself | smooth — the room |
| colour drift (raw) | 18-23% | 4.9-23.2% |
| soft frames | 4-11% | **0-3%** |
| overlap | 92-94% | 92-98% |
| verdict | **rejected** | **passed 9/9** (against the *corrected* gate — see §3; the original gate failed all nine) |

The honest gap is 3-8x on the detrended figure, not 40x. It is still decisive, and it is
still entirely the camera: take 1's framing and pace were fine — 92-94% overlap, only
4-11% soft. The operator was never the problem. That is the whole argument for locking,
and everything below is why.

Note what colour does in that table: raw drift *overlaps* between the auto and locked
takes. That is not noise in the measurement, it is the measurement telling you frame
statistics cannot separate a hunting white balance from a subject that is simply a
different colour over there. §3 explains why that killed a metric.

---

## The camera is the capture

Every later stage — feature matching, pose solving, splat optimisation, the scale check — inherits what the sensor decided in the moment. No stage recovers information the camera threw away. The highest-leverage decision is therefore made before the first frame is written: lock exposure, lock white balance, lock focus.

### What this project measured

Two captures of the same bookshelf pair (ground truth 70-7/8" x 69-7/8", tape), same operator, same room, same path.

**Take 1 — stock iOS Camera, auto everything.** Rejected. Frame brightness swung 92% and 106% across the takes, and it *oscillated*: the meter hunted. White balance drifted 18-23%. The rest of the numbers matter: sharpness was fine (4-11% soft frames), overlap 92-94%. The operator did the job correctly. The camera lost it.

**Take 2 — manual camera app, 4K/24 HEVC ~36 Mbps, shutter 1/120 + ISO 1250 + WB 2930K + focus all locked, Rec.709, stabilisation off, recorded to Files.** Passed 9/9 clips. Sharpness median 402-1189, soft frames 0-3%, overlap 92-98%, exposure wobble 2.2-6.2%.

Same hands, same shelf, same light. The delta is the lock.

### What iOS gives you and what it withholds

The stock Camera app's AE/AF Lock does what its name says and no more: it holds **auto-exposure and auto-focus**. It does **not** hold white balance — measured here as Take 1's 18-23% drift. This fits AVFoundation's structure, where white balance is a separate property with its own lock (`setWhiteBalanceModeLocked(with:completionHandler:)`) alongside separate exposure and focus modes. Apple does not document what the stock app does to white balance when AE/AF Lock engages, so the *mechanism* is INFERRED; the *drift* is measured. Either way, the stock app cannot fully lock the camera.

Web code is worse off. WebKit's `MediaTrackCapabilities.idl` declares `whiteBalanceMode`, `zoom`, `torch`, and `focusDistance`, but `exposureMode`, `focusMode`, `exposureTime`, `colorTemperature`, and `iso` appear only as commented-out FIXMEs. A browser-based capture tool on iOS cannot lock exposure or focus at all. Note the inversion: the control the stock native app withholds (white balance) is the one the web exposes, and vice versa. Neither surface suffices. Capture natively.

### Shutter is not frame rate

These are set independently and routinely conflated. Frame rate only caps the **slowest** permissible exposure: at 24 fps the frame interval is 41.7 ms, so exposure must be ≤ 41.7 ms. It says nothing about how much faster you may go. Take 2 ran 24 fps with a 1/120 s (8.3 ms) shutter, occupying 20% of each interval with the sensor idle for the rest. That is not waste; it is motion freezing. Choose shutter for blur and flicker, frame rate for data volume.

### Flicker: 1/120 and 1/60, and nothing else

In a 60 Hz country, full-wave-rectified LED drivers modulate at **twice** mains — 120 Hz — the behaviour IEEE 1789-2015 exists to address. Period: 8.33 ms. An exposure is flicker-immune only when it integrates a **whole number** of cycles: 1/120 s = exactly 1 cycle (safe), 1/60 s = exactly 2 (safe), while 1/250 s = 0.48 cycles and 1/100 s = 1.2 cycles both band. RED states the rule correctly — "choose a shutter speed equal to the lighting pulse rate divided by some integer" — yielding 1/120, 1/60, 1/40, 1/30.

Name the folklore: camera blogs widely claim "1/240 is flicker-free at 60 Hz." It is wrong. It inverts the rule, treating denominators that are *multiples* of 120 as safe when the requirement is exposure times that are integer multiples of 8.33 ms — denominators that **divide** 120. 1/240 s is half a cycle.

### ISO: take the noise

Locking shutter at 1/120 indoors forces ISO up; Take 2 sat at ISO 1250 and still returned sharpness medians of 402-1189, 0-3% soft frames, 9/9 pass. High ISO did not prevent a pass. The trade is deliberate and asymmetric: noise degrades gracefully, blur does not. Severe blur renders COLMAP feature matching ineffective outright, and poses recovered from blurred images are "significantly biased." Noise is not free — detectors suffer spurious detections under moderate noise — but it is far cheaper. Buy flicker immunity and motion freezing; pay in grain.

### The torch is a trap

A camera-mounted light is fixed in **camera space**, not world space. As you orbit, every surface's shading and every specular highlight travels with you, so the same patch of wood presents different radiance from every viewpoint. COLMAP's own guidance is to "capture images at similar illumination conditions." A static scene model cannot represent a light that moves. Fix light in the world — turn the room lights on — or the reconstruction must explain your torch as geometry.

### Rolling shutter

CMOS sensors scan line-by-line, so a frame taken during hand motion is sheared, not merely blurred. This is a *geometric* corruption of the pin-hole model pose estimation assumes, not a photometric one — which is why no colour correction touches it. In synthetic comparison, compensating rolling shutter recovered 19.21 → 35.84 PSNR on one scene, far more than motion-blur compensation alone (~5-8 dB), suggesting shear may matter more than blur on handheld phone capture. That work used an iPhone 15 Pro plus two Android phones, noting only that the Android devices have "a known, and relatively large rolling-shutter readout time," implying the iPhone's is smaller. **UNVERIFIED:** the readout time of the iPhone 14 Pro Max used here is unmeasured, and no primary source was found; secondary compilations quote ~5 ms for recent iPhones. The mitigation is behavioural: move slowly, no whip pans.

### Where the literature pushes back

A real tension deserves flagging. The naive story — "exposure drift breaks SfM" — is not what Lowe's SIFT paper says. The descriptor "is normalized to unit length": a contrast change "will be canceled by vector normalization," a brightness change "will not affect the gradient values," therefore "the descriptor is invariant to affine changes in illumination" (§6.1). SIFT is *by construction* immune to the linear part of an exposure change. Lowe names the exception: "non-linear illumination changes can also occur due to **camera saturation**."

So locking exposure does not help by rescuing feature matching from brightness as such. It helps because auto-exposure clips highlights and crushes shadows — the non-linear regime Lowe excludes — and because it meters by changing *exposure time*, varying motion blur frame to frame. Blur is what genuinely destroys keypoints.

Colour goes further and contradicts the folklore outright. Per-image affine colour transforms recover +1.37 dB PSNR (21.71 → 23.08) in 3DGS, meaning white-balance variation is largely *absorbable downstream*. That independently corroborates this project's decision to demote colour to advisory in the capture gate — a decision forced by measurement (auto-WB 3.5%/4.5% vs locked 3.7%/4.2%; the distributions overlap outright, so no threshold exists). Two lines of evidence, one from our gate and one from the literature, converge: **lock white balance because it is free, not because the reconstruction depends on it.** Exposure and focus are the ones that must not move.

## Geometry, coverage and scale

### Scale is a free parameter, not an error

Structure from motion recovers the scene only up to a similarity transform: three degrees of freedom of rotation, three of translation, and one of scale, seven in total, are unconstrained by the images. This is gauge freedom, and it is a property of the problem, not a defect of the solver. Doubling every camera baseline and every point coordinate reproduces every pixel exactly, so no amount of image evidence can distinguish the two. The consequence for a splat is blunt: a reconstruction trained on COLMAP poses inherits COLMAP's gauge, and a 3D Gaussian splat measures in *scene units*. It is not "roughly metres". It is a number times an unknown scalar, and the scalar has to come from outside the pixels.

Practically, this means a viewer that reports lengths must either fix the scalar or say it hasn't. Everything below is about fixing it, and about the coverage that has to exist first for the geometry to be recoverable at all.

### Parallax requires translation

Rotation carries no depth. If the camera rotates about its own centre, the two views are related by the homography `H = K R K⁻¹` — intrinsics, rotation, intrinsics — with **no depth term in it**. OpenCV's own documentation makes the point precisely: normally a homography only relates planar structure, but under pure rotation "an arbitrary world can be considered", because the mapping is independent of the scene. Every point moves the same way regardless of distance. There is nothing to triangulate.

This is why the critical-configuration literature is *not* the place to look for the pure-rotation case. Bråtelund's classification of critical configurations for two projective views assumes throughout "that all cameras have distinct centres" — zero baseline is not a hard case inside the theory, it is excluded from the theory's domain. Pure rotation isn't ambiguous reconstruction; it's the absence of a reconstruction problem.

COLMAP's capture guidance says the same thing in the imperative: "Capture images from different viewpoints. Do not take images from the same location by only rotating the camera, e.g., make a few steps after each shot." Panning a phone from a fixed stance produces a beautiful, useless video.

### Arcs beat strafing

Given that you must translate, translation is not all equal. Triangulation uncertainty is governed by the parallax angle subtended at the point: small parallax means the point can slide along the viewing ray without moving measurably in either image, so depth uncertainty blows up, and depth error grows with the square of depth for a fixed baseline. Walking an arc around a subject accumulates angular diversity — each step buys a genuinely new bearing. Strafing along a wall at constant distance buys baseline but little angular change on distant structure, and the far field stays weakly constrained. The countervailing pressure is that very wide angles degrade feature matching, so the useful move is a sequence of moderate arcs, not one heroic wide pair.

### Inside-out is harder than outside-in

Orbiting an object (outside-in) points every camera at a shared volume: translation is large relative to subject depth, co-visibility is high, and COLMAP's "each object seen in at least 3 images" comes free. Standing inside a room (inside-out) inverts all of it. The camera faces outward, depths are large relative to the translation a body can produce indoors, so the baseline-to-depth ratio is poor exactly where the geometry is weakest; each surface is seen by a narrow slice of the trajectory; and errors close a loop rather than a circle. Enclosures need loop closure and deliberate re-visiting of the same wall from a second stance. **Measured in this project:** the captures that passed were bookshelf-scale, effectively outside-in. Full-room inside-out capture has not been measured here — treat the above as literature-derived, UNVERIFIED locally.

### Overlap: what is sourced and what is transplanted

The "70–80% overlap" rule is **sourced for aerial nadir grids and folklore everywhere else**. Pix4D specifies at least 75% frontal and 60% side overlap for the general case, rising to 85% over forest and 90% for thermal — all of it flight-planning guidance, with no indoor, terrestrial, or handheld figure anywhere on the page. COLMAP, the engine that actually consumes the frames, declines to give a percentage at all: it says "high visual overlap" and "at least 3 images". The number is real, was derived for a different capture geometry, and got carried indoors by repetition. Say "high" and mean it.

**Measured in this project:** our accepted takes ran 92–98% frame-to-frame overlap, and even the rejected take ran 92–94%. Video at 24 fps makes overlap nearly free; the 70–80% band never binds. One instrument caveat, also measured: our phase-correlation overlap estimator **aliases past half a frame and over-reports** — a true 38% reads as 62%. The estimator is only trustworthy above ~50%, which is where we operate, but a reported "70%" could be a disguised 30%. Trust it as a pass check, never as a diagnosis of a low reading.

### Recovering metric scale, and why the longest reference wins

Three routes exist, and their error budgets differ by three orders of magnitude.

- **GPS: dead on arrival.** A real iPhone clip self-reported `LocationAccuracyHorizontal = 19.785 m` for a subject 1.80 m across — roughly 1000× too coarse — and iOS writes only one static point per clip anyway (`com.apple.quicktime.location.ISO6709`, a `moov` atom with no time dimension). There is no trajectory in it.
- **IMU dead-reckoning: worse.** A 0.1° tilt error leaks `g·sin θ` into horizontal acceleration and double-integrates to ~123 m of position error over a 2-minute capture.
- **ARKit VIO: real, but not superior.** Measured here at 0.14–1.47% relative ATE over handheld 47–164 s sequences. A tape-and-click reference measures 0.2–0.6%. **ARKit ties the tape; it does not beat it.** Scaniverse's point-cloud export was verified metric on this machine (bbox 1.52 × 2.40 × 0.61 m, plausible real metres), so the VIO route works — it just doesn't earn a precision upgrade.

The governing insight is that **scale error is bounded by pick precision on the reference, not on the thing you are measuring**. The scalar is a ratio: known length over picked length. The same few-pixel slop in clicking two endpoints is a fixed absolute error, so it divides by the reference's length. **Measured in this project:** identical slop yields 0.57% over a 24″ tape and 0.19% over the 70-7/8″ shelf (ground truth 70-7/8″ × 69-7/8″, tape-verified self-consistent: 35-1/4 + 3/8 + 35-1/4 = 70-7/8 exactly). Reach for the longest known length in the room. A 3× longer reference is a 3× smaller error, for free, with no better equipment.

### Why a printed target beats a tape measure

A tape in frame requires a human to pick two endpoints against blurred graduations, and the pick precision *is* the error budget. An ArUco board or checkerboard replaces that with a machine-detectable target whose dimension is known by manufacture rather than by reading: OpenCV performs sub-pixel corner refinement (`CORNER_REFINE_SUBPIX`, `CORNER_REFINE_APRILTAG`), the detection is repeatable rather than judged, and the corners are the sharpest features in the scene by construction. Tools such as `aruco-estimator` triangulate marker corners in a COLMAP sparse model and derive the scalar directly. Two honest caveats: `aruco-estimator` publishes **no accuracy figures** and warns its pose estimation "is not robust to false detections"; and the same longest-baseline logic still applies — a small marker is a short reference, so a large printed board beats a business-card marker, and a board spanning the subject beats both. **This project has not measured marker-based scale. UNVERIFIED here.** Our 0.19% came from a shelf.

## Measuring a capture before you spend on it

Reconstruction is the expensive half and the late half. 3D Gaussian splatting begins "from sparse points produced during camera calibration" — structure-from-motion runs first, and everything downstream inherits its failures. COLMAP's own capture guidance asks for good texture, "similar illumination conditions", "high visual overlap" with each object seen in at least three images, and genuinely different viewpoints. Every one of those is decided in the room and cannot be recovered afterwards. A gate exists to answer one question while you are still standing there: reshoot now, or pay for the GPU?

The cost asymmetry looks favourable. A false pass costs a training run plus a reshoot later; a false fail costs a reshoot now. So a gate should lean strict. That reasoning is what produced the worst instrument in this project.

### What frame statistics can see

Two quantities measure honestly. Variance-of-Laplacian ranks frames within a clip: the second derivative responds to fine detail, and motion blur removes it. Phase correlation between consecutive frames estimates displacement, and therefore whether SfM will find shared features.

What they cannot see is the difference between the camera and the room. Frame-mean brightness is one number, and it moves for two unrelated reasons: a meter re-metering, and a *locked* camera walking past a lamp. Locking is the advice. So the second cause is guaranteed by following the advice, and no statistic of a single frame can separate it from the first.

### The gate that failed nine good captures

Measured in this project. Take 1 was stock iOS Camera, auto everything: frame brightness swung 92% and 106% across takes and *oscillated*; white balance drifted 18–23%. Sharpness was fine (4–11% soft frames), overlap 92–94%. The operator was fine; the camera lost it. Note the specific defect — stock iOS AE/AF Lock holds exposure and focus but not white balance.

Take 2 fixed the instrument at the source: locked shutter 1/120, ISO 1250, WB 2930K, locked focus, stabilisation off. Nine clips, sharpness medians 402–1189, soft frames 0–3%, overlap 92–98%.

Our gate failed all nine, flagging 15–59% "exposure drift". The gate was wrong. One flagged take ramped luma 69→127 monotonically, with a straight line explaining 94% of the variance. That is not a camera hunting; that is a room getting brighter as the operator walks. The gate had measured the scene and blamed the camera.

### Shape, not size

The fix is that the two causes differ in shape, not magnitude. A meter oscillates; a room is smooth. Fit and remove a linear trend, then measure the residual. Detrended, our numbers separate: auto 18.0% and 16.1%, locked 2.2%–6.2% — no overlap, a gap of about 2.6×.

Be precise, because the raw numbers were not literally inseparable: locked ran 15–59% raw and auto 92–106%, and a threshold near 75% would have sorted this data. The objection is that such a threshold does not measure the camera. The raw locked value is set by the lighting geometry of the walk — a longer path past a brighter lamp raises it without touching the camera — so it is calibrated to one room and transfers nowhere. The detrended residual is a property of the camera, which is the thing being gated. That, not the size of the gap, is the argument.

### White balance: the same diagnosis, no cure

White balance got the identical diagnosis and it did not work. Detrending assumes the scene's contribution is smooth in time. Colour is not: a subject's colour changes discontinuously with position — white robot, then wood, then beige wall — so detrending isolates nothing. Measured here, the distributions overlap outright: auto-WB 3.5%/4.5% versus locked 3.7%/4.2%. Locked is not even reliably lower. No threshold exists, so colour was demoted from gate to advisory.

Appearance variation is still a real hazard — serious enough that NeRF-W exists because plain NeRF "is incapable of modeling ... variable illumination". But real does not imply measurable from frame statistics alone.

### Every instrument's own failure mode

**Variance-of-Laplacian is inflated by sensor noise.** The noise sensitivity is established: Pertuz et al. found Laplacian operators best in clean conditions but "the most sensitive to noise", calling this "a well known fact"; Subbarao and Tyan built a closed-form noise-sensitivity analysis of focus measures for the same reason. Read those precisely — they measure *degraded* depth reconstruction, not an inflated score. The inflation is separate, and it is arithmetic rather than folklore: for a linear kernel h, white noise of variance σ² contributes σ²·Σh² to the filtered variance. The 4-neighbour Laplacian has Σh² = 20, so VoL gains 20σ², additively, on top of the scene's own value. Verified numerically here.

That lands on our own recommendation. We tell you to lock ISO 1250 to afford 1/120s — and ISO 1250 raises the noise floor that inflates the metric certifying our sharpness. A noise σ of only ~8 DN alone yields VoL ≈ 1280, the top of our measured 402–1189 range. **UNVERIFIED: we never measured our sensor's noise floor, so we do not know what fraction of those medians is detail and what fraction is grain.** Sharpness numbers are comparable *within* a fixed-ISO clip and not across takes at different ISO. Our own data shows the content dependence directly: medians range 3× across nine clips of the same room at identical settings. Absolute thresholds do not transfer — consistent with Pertuz, who found the best operator "strongly depends on the particular capturing device".

**Phase correlation aliases past half a frame.** The DFT knows shift only modulo the frame, so displacement d and N−d are indistinguishable; the estimate folds about 50% and over-reports overlap beyond it. Measured here: true 38% overlap reads as 62% — exactly the reflection. Foroosh et al.'s error analysis names the responsible terms: "border errors due to periodicity assumption", non-overlapped regions, and aliasing. The instrument is trustworthy only above ~50% overlap — where, fortunately, usable captures live.

**A graceful drift hides in a trend.** Detrending is the exposure cure, and it is also a blind spot: an AE that wanders slowly and monotonically is removed by the same fit that removes the walk. We have not observed this. **UNVERIFIED: our detrended gate would pass a slow monotonic meter drift, and we have no test that would catch it.**

### The thesis

An instrument that fails good work is worse than no instrument. No instrument leaves you with your judgement; a lying one overrides it, and the strict-by-default reasoning above makes the lie feel responsible. Ours flagged nine takes that were, by every other measure, the best footage we had — and the only reason we know is that we ran it on real data and disbelieved it. Gate before the GPU. But calibrate the gate first, on captures you already know are good, and treat a gate that rejects them as the broken thing.

## From frames to a splat, and what it costs

### The toolchain wall is real, and it is not where you expect

Measured in this project: COLMAP 4.1's CUDA build fails to initialise against a driver that caps at CUDA 12.7. Falling back to the CPU path, feature extraction processed **354 frames in 49 s** (~7.2 frames/s). No CUDA toolkit was needed to get features out of a room-scale capture.

Two honest limits on that number, because it is easy to over-read.

First, 49 s covers feature *extraction* only — not matching, not incremental mapping. Matching is the stage that dominates wall-clock and scales badly: exhaustive matching is quadratic in image count. COLMAP documents sequential matching as the mitigation for video, where "consecutive frames have visual overlap and there is no need to match all image pairs exhaustively" [1]. Our 49 s says nothing about the other two stages. **UNVERIFIED here.**

Second, and more important: COLMAP's own documentation *contradicts* any inference that the CPU path is equivalent. It states plainly that "in general, the GPU version is favorable, as it has a customized feature detection mode that often produces higher-quality features for high-contrast images" [1]. We measured CPU **speed**, not feature **quality**. The sparse model from the CPU path has never been compared against a CUDA-built one on this machine. The correct claim is narrow: on a ~350-frame capture, absence of a CUDA build is not a blocker for extraction. It is not a claim that the reconstruction is as good.

One related trap worth knowing before you debug it at 1 a.m.: COLMAP's OpenGL fallback, used when CUDA is unavailable, "instead requires an attached display, so on such systems the CPU version is recommended for use on a server" [1]. A headless box wants the CPU flag explicitly, not the fallback.

The real wall is downstream, at the trainer. The INRIA reference implementation requires a "CUDA-ready GPU with Compute Capability 7.0+" and "24 GB VRAM (to train to paper evaluation quality)", and to build its rasteriser it requires a "C++ Compiler for PyTorch extensions" plus "CUDA SDK 11 for PyTorch extensions", with the note that "C++ Compiler and CUDA SDK must be compatible" [2]. On Windows that means MSVC and a version-matched CUDA toolkit. Measured here: without those, the CUDA-kernel trainers do not compile — **the blocker is a toolchain, not a GPU**. A capable card does not help you if nvcc and cl.exe are absent.

Brush sidesteps this entirely. It is Rust, built on the Burn framework with CubeCL and wgpu, targets Vulkan/Metal/DX12/WebGPU across AMD, Intel and Nvidia — plus Android and the browser — and ships "simple dependency free binaries" with no CUDA toolkit [3]. It ingests COLMAP or Nerfstudio datasets [3]. For a Windows machine without MSVC, that is the difference between a pipeline and a yak shave.

Note the distinction that trips people: **a permissive licence is not a portable toolchain.** gsplat is Apache-2.0 [4][5], yet its backend is "highly optimized CUDA kernels" [5] — free to use commercially, still toolchain-bound. Brush gives you both freedoms at once.

### COLMAP's role, and why it is mandatory

3DGS does not do Structure-from-Motion. The paper's method begins "starting from sparse points produced during camera calibration" [6] — that initialisation is an input, not something the trainer derives. The reference implementation instructs users to "install a recent version of COLMAP (ideally CUDA-powered)" to prepare custom scenes [2], and Brush takes COLMAP data as a primary input format [3]. So COLMAP — or any pose estimator that emits COLMAP-format output — is load-bearing upstream of every trainer here. It produces the sparse cloud plus per-image intrinsics and extrinsics [1]; the splat is optimised *into* that camera frame. Bad poses do not produce a blurry splat, they produce a wrong one, and no amount of training iterations recovers them.

SfM is also **scale-free**: it recovers geometry up to an unknown similarity transform. Metric scale must come from outside. Measured in this project, the phone's own metadata cannot supply it — a real iPhone clip self-reported `LocationAccuracyHorizontal = 19.785 m` against a 1.803 m subject, ~1000x too coarse, and iOS writes only one static GPS point per clip. A tape-and-click reference on the longest available baseline is the working answer (measured: 0.19% scale error over a 71" shelf vs 0.57% over a 24" tape — longest baseline wins). Separately, a Scaniverse point-cloud export was verified metric on this machine (bbox 1.52 x 2.40 x 0.61 m, plausible real metres), which is an alternative scale source but was not cross-checked against the tape.

### What 3DGS optimises, versus the alternatives

3DGS represents the scene **explicitly**, as anisotropic 3D Gaussians, and optimises them by gradient descent through a differentiable, "fast visibility-aware rendering algorithm that supports anisotropic splatting" — which both "accelerates training and allows realtime rendering" [6]. The second ingredient is **adaptive density control**: "interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance" [6], which grows the model where reconstruction error is high and prunes it where Gaussians are useless. The reference trains 30,000 iterations by default, with an evaluation checkpoint at 7,000 [2]. Result: "≥ 30 fps novel-view synthesis at 1080p" [6].

Choose by what you need out the far end:

- **3DGS** wins on photorealism-per-minute and real-time viewing, and handles view-dependent appearance (specular, gloss) that meshes fake badly. It has **no surfaces** — you cannot collide with it, and measuring it means measuring a cloud.
- **NeRF** encodes the scene implicitly in an MLP queried per ray [7]. Historically far slower to train and render. Rarely the right default now for a static room.
- **Mesh photogrammetry** (MVS) gives explicit surfaces you can measure, section, collide against and import into CAD. For a civil-engineering deliverable this often beats a splat outright, and the honest position is that they answer different questions.

gsplat is the pragmatic middle for the CUDA lane: Apache-2.0, and reports "10% less training time and 4× less memory" than the reference — 5.6 GB vs 9.0 GB and 19.39 min vs 26.19 min at 30k iterations on MipNeRF360 [5]. Those are the authors' numbers on their hardware, not reproduced here.

### Licensing is a first-class decision, not a footnote

Decide this **before** you build a pipeline on it. The INRIA reference implementation is **not** open source in the usable sense: "Licensors grant non-exclusive rights to use the Software for research purposes to research users (both academic and industrial), free of charge, without right to sublicense", and in capitals, "THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE SOFTWARE FOR COMMERCIAL PURPOSES WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS", directing commercial users to contact Inria [8]. Anything you demonstrate commercially on that codebase is a licence violation.

gsplat (Apache-2.0, "Copyright 2025 Nerfstudio Team") [4] and Brush (Apache-2.0) [3] are both clean for commercial use. Route around the reference implementation early — retrofitting a trainer swap after you have built tooling on it is expensive.

The same discipline applies to any detector bolted on downstream. Ultralytics YOLO is dual-licensed AGPL-3.0 or Enterprise: under AGPL-3.0 you must open-source your **entire** project, and this extends to "trained/fine-tuned models" — not merely the code [9]. AGPL's network clause means shipping a web viewer backed by an AGPL detector can trigger disclosure. Weights are not license-neutral assets.

### Getting data off the phone without silent re-encoding

Measured in this project, and the most easily-missed failure in the chain: **the iOS Photos picker re-encodes on upload.** A 4K60 HEVC take arrived as **4K30 H.264 at ~24.5 Mbps** — half the frames silently gone, before COLMAP ever saw a pixel. You lose baseline coverage and gain compression artefacts on exactly the high-frequency texture SfM feeds on.

The fix is procedural, not technical: **record to Files and import from Files.** Take 2 in this project did so and passed 9/9 clips (4K/24 HEVC ~36 Mbps).

The mechanism is *probably* `PHPickerConfiguration.preferredAssetRepresentationMode`, where the default permits transcoding and `.current` avoids it when the receiving app handles arbitrary formats [10]. Flagged **REPORTED**, not verified: [10] is an Apple Developer Forums thread, we could not retrieve the API reference body, and we never confirmed that this setting governs the observed re-encode. **The behaviour is measured; the cause is not.** Apple separately documents a Settings > Photos > "Transfer to Mac or PC" Automatic/Keep Originals control for USB transfer — a *different* mechanism, and conflating the two is folklore.

One more measured trap: Blackmagic Camera writes a **1080p/~6 Mbps proxy beside each 4K/~36 Mbps master with an identical filename.** Discriminate by **resolution**, never by name and never by size. Feeding the proxy to COLMAP is a silent quality loss that looks like a bad capture.


---

## What we got wrong, and how we found out

Every number above was checked against its source by a reader whose only job was to find
inflation. What it caught is kept here rather than quietly fixed, because a document that
shows its own corrections is easier to trust than one that doesn't.

Two of its findings deserve to be read before the list, because they demonstrate the method
cutting in both directions.

**It caught the author.** An earlier draft of the summary table set the auto take's *raw*
92% brightness spread beside the locked take's *detrended* 2.2% wobble — two different
measurements — and presented the ratio as the argument for locking. The real gap is 3-8x
detrended, not 40x. The conclusion survived; the number didn't. That error was in the one
table most likely to be quoted, which is exactly where this class of mistake lives.

**And it overreached.** It asserted, confidently, that "there is no COLMAP 4.1 release —
latest releases are 3.13.0 (2025-11-07)", and on that basis moved to cut the toolchain
measurements. Checked directly: the GitHub API returns tag `4.1.0`, `prerelease=false`,
`draft=false`, published 2026-06-26; the release list reads 4.1.0 → 4.0.4 → 4.0.3 → 4.0.2;
and the binary in use prints `COLMAP 4.1.0 (Commit fa8e3b3 on 2026-06-26 with CUDA)`. The
checker was working from stale knowledge. Its other correction on that section — that the
"354 frames in 49 s" figure was unattested — was procedurally right and factually wrong:
the number was measured in this session, after the checker's brief was written, and is
recorded in the session log. Both are kept because "the adversarial reader was wrong here,
and here is the evidence" is more useful than deleting the exchange.

The lesson generalises past this document: a checker that cannot see your evidence will
flag your true claims as unsourced, and one working from stale knowledge will contradict
things you can verify in one command. Verify the verifier. Neither its approval nor its
objection is the last word — the artifact is.

## Fact-check corrections

Verified against the evidence block and against primary sources fetched 2026-07-15. Verdict up front: **Sections 2 and 3 are substantially clean on literature; Section 1 has one real misattribution and one metric collision; Section 4 carries measurements that do not exist in the evidence block and a version number that does not exist at all.** The most serious single item is #1 (Section 4), followed by #5 (the white-balance metric collision) and #8 (phase correlation).

### Section 4 — unattested "measured here" claims

- **"COLMAP 4.1's CUDA build fails to initialise against a driver that caps at CUDA 12.7"** -> **PROBLEM: there is no COLMAP 4.1 release.** Latest releases are 3.13.0 (2025-11-07), 3.12 (2025-06-30), 3.11 (2024-11-28). The docs site currently builds its dev docs as "COLMAP 4.1.0.dev0 | 43dd3bb2 (2026-03-16)", so a "4.1" is a main-branch dev build, not a release. Separately, **this measurement appears nowhere in the project's evidence block** — no COLMAP version, no CUDA 12.7, no driver cap. -> **FIX: state the exact build ("4.1.0.dev0, main branch, commit 43dd3bb2") or the actual release used, and cite the session log that recorded the failure. If no log exists, cut it.**
- **"feature extraction processed 354 frames in 49 s (~7.2 frames/s)"** -> **PROBLEM: not in the evidence block.** The arithmetic is right (354/49 = 7.22) but the measurement is unattested. This is the section's headline number and the whole document's premise is that its numbers are checkable. -> **FIX: cite the log, or cut. Do not carry an unsourced number under a "Measured in this project" heading.**
- **"Measured here: without those, the CUDA-kernel trainers do not compile — the blocker is a toolchain, not a GPU"** -> **PROBLEM: also not in the evidence block.** -> **FIX: same — attest or cut. The INRIA requirements quotes (which are verified) already support the *claim*; they just don't make it a *measurement*.**
- **gsplat's backend is "highly optimized CUDA kernels" [5]** -> **PROBLEM: right phrase, wrong source.** That wording is from the gsplat paper abstract, not the repo or docs. The README says "CUDA accelerated rasterization of gaussians with python bindings"; the docs say "CUDA-accelerated differentiable rasterization of 3D gaussians with Python bindings". Neither contains "highly optimized CUDA kernels". -> **FIX: cite arXiv 2409.06765 / JMLR 26:24-1476 for the quote. Apache-2.0 is correctly verified on the repo — keep [4] as-is.**
- **Brush "is Rust, built on the Burn framework with CubeCL and wgpu, targets Vulkan/Metal/DX12/WebGPU" [3]** -> **PROBLEM: the README does not say CubeCL, wgpu, Vulkan, Metal, DX12, or DirectX.** It says "uses WebGPU compatible tech and the Burn machine learning framework". The backend list is an inference from wgpu's target set, not a claim in [3]. -> **FIX: quote what the README actually says, and either cite Burn/wgpu docs for the backend list or mark it INFERRED. Verified verbatim and safe to keep: "produces simple dependency free binaries", "works on a wide range of systems: macOS/windows/linux, AMD/Nvidia/Intel cards, Android, and in a browser", "Brush takes in COLMAP data or datasets in the Nerfstudio format".**
- **Brush ships "with no CUDA toolkit"** -> **PROBLEM: the README says "without any setup", not "no CUDA toolkit".** The conclusion is almost certainly correct but it is yours, not theirs. -> **FIX: mark INFERRED.**
- **COLMAP sequential-matching quote: "consecutive frames have visual overlap and there is no need to match all image pairs exhaustively" [1]** -> **PROBLEM: substance confirmed, exact verbatim not.** The tutorial page returned "consecutively captured images are matched against each other" on fetch; the FAQ page contains neither this quote nor the GPU-vs-CPU one. -> **FIX: re-verify the verbatim string and point [1] at `colmap.github.io/tutorial.html` specifically, not "COLMAP docs".**
- **COLMAP GPU-favourable quote [1]** -> **PROBLEM: verified on the tutorial ("often produces higher-quality features for high-contrast images"), NOT on the FAQ.** Same for the OpenGL/attached-display sentence, which is verified verbatim. -> **FIX: make [1] resolve to the tutorial page.**
- **Heading "COLMAP's role, and why it is mandatory"** -> **PROBLEM: mild self-contradiction.** The body correctly says "COLMAP — or any pose estimator that emits COLMAP-format output", and Brush also ingests Nerfstudio format. SfM is mandatory; COLMAP-the-program is not. -> **FIX: retitle to "why SfM is mandatory".**
- **INRIA requirements block** -> **CLEAN.** All six quotes verified verbatim: "CUDA-ready GPU with Compute Capability 7.0+", "24 GB VRAM (to train to paper evaluation quality)", "C++ Compiler for PyTorch extensions", "CUDA SDK 11 for PyTorch extensions", "C++ Compiler and CUDA SDK must be compatible", "install a recent version of COLMAP (ideally CUDA-powered)". Minor: the source adds "and ImageMagick", and qualifies CUDA SDK 11 with "we used 11.8, known issues with 11.6" — worth keeping that version nuance since the section is about toolchain pain.

### Section 1 — misattribution and a metric collision

- **"The stock Camera app's AE/AF Lock ... does not hold white balance — measured here as Take 1's 18-23% drift"** -> **PROBLEM: the measurement does not support the claim.** Take 1 was "stock iOS Camera, **auto everything**" per the evidence. AE/AF Lock was never engaged in any measured take. An auto-everything take drifting 18-23% tells you nothing about what AE/AF Lock holds, because the lock was off. -> **FIX: split into three claims with three honesties — (a) MEASURED: auto-everything drifts 18-23%; (b) PRIMARY SOURCE: AVFoundation exposes white balance as a separately lockable property, `setWhiteBalanceModeLocked(with:completionHandler:)`, distinct from exposure and focus modes (verified); (c) UNVERIFIED HERE: what the stock app's AE/AF Lock does to white balance — untested, since no take used it.**
- **Take 2 "exposure wobble 2.2-6.2%" placed directly against Take 1's "swung 92% and 106%"** -> **PROBLEM: apples to oranges, and it flatters the conclusion.** Per the evidence, 2.2-6.2% is the **detrended** residual; 92%/106% is **raw**. The raw locked figure is 15-59% (Section 3 says so). Setting 2.2-6.2% beside 92/106% overstates the gap by roughly an order of magnitude. -> **FIX: label every brightness number raw or detrended at point of use. The honest raw pair is locked 15-59% vs auto 92%/106%; the honest detrended pair is locked 2.2-6.2% vs auto 18.0%/16.1%.**
- **"Passed 9/9 clips"** -> **PROBLEM: flatly contradicts Section 3's "Our gate failed all nine, flagging 15-59% exposure drift".** Both are true in sequence (failed the original gate, passed the corrected one) but a reader hits the contradiction with no signal. -> **FIX: Section 1 should read "passed 9/9 against the corrected gate — see §3, the original gate failed all nine."**
- **"Two captures ... same operator, same room, same path" / "Same hands, same shelf, same light"** -> **PROBLEM: "same path" and "same light" are not attested in the evidence.** Same operator, room, and subject are; path identity and lighting identity are asserted. Given the section's own thesis — that a *locked* camera's brightness ramp is produced by the walk past a lamp — path identity is load-bearing, not decorative. -> **FIX: state what was controlled and mark the rest UNVERIFIED.**
- **"the behaviour IEEE 1789-2015 exists to address"** -> **PROBLEM: overclaims the standard.** IEEE 1789-2015 is "Recommended Practices for Modulating Current in High-Brightness LEDs for Mitigating Health Risks to Viewers" — a *human-health* standard about modulation frequency and depth (<90 Hz high risk, 90-1250 Hz depth-dependent, >1250 Hz low risk). It does not exist to address 120 Hz full-wave rectification, and it is about viewers' eyes, not camera banding. -> **FIX: cite it only for the narrow fact that LED modulation is real and characterised, or drop it. The RED quote plus the 8.33 ms arithmetic already carry the paragraph.**
- **"RED states the rule correctly — '...divided by some integer' — yielding 1/120, 1/60, 1/40, 1/30"** -> **PROBLEM: the quote is verified; the list is not RED's.** Fetched directly, the RED tutorial states the rule but does not print 1/120, 1/60, 1/40, 1/30. -> **FIX: keep the quote, and make explicit that the list is your derivation from it (120/1, 120/2, 120/3, 120/4).**
- **"camera blogs widely claim '1/240 is flicker-free at 60 Hz'"** -> **PROBLEM: the call is correct but the sourcing is absent — and a folklore callout in *this* document must itself be checkable.** I confirmed the folklore is real and widespread (multiple sources pair 60 Hz with "1/60, 1/120, or 1/240"), and 1/240 s = 0.5 cycles is genuinely phase-dependent. Your logic — denominators must *divide* 120, not be *multiples* of it — is sound. -> **FIX: cite one concrete instance of the folklore. Also strengthen the mechanism, because "it inverts the rule" understates it: sub-cycle exposures band chiefly via **rolling shutter** (rows sample different ripple phases within one frame). Note the edge case honestly — at 24 fps the frame period is exactly 5 ripple cycles (41.667/8.333 = 5.0), so frame-to-frame flicker can vanish at any shutter if the camera clock were mains-locked; it isn't, and rolling-shutter banding survives regardless. Whole-cycle integration is the only phase-independent guarantee. That is the real argument.**
- **"Severe blur renders COLMAP feature matching ineffective outright"** (sentence truncated) -> **PROBLEM: unsourced.** COLMAP's guidance says "Capture images with good texture. Avoid completely texture-less images" — it does not make a blur claim in these terms. -> **FIX: source it or soften to the measured framing (noise degrades gracefully, blur does not) which your ISO-1250 data already supports.**
- **Verified clean in Section 1:** Take 1 sharpness 4-11% soft / overlap 92-94%; Take 2 config, medians 402-1189, 0-3% soft, 92-98% overlap; 24 fps = 41.7 ms; 1/120 = 8.3 ms = 20% of the interval; 1/120 = 1 cycle, 1/60 = 2, 1/250 = 0.48, 1/100 = 1.2. WebKit IDL claim verified exactly — `whiteBalanceMode`, `zoom`, `torch`, `focusDistance` declared; `exposureMode`, `focusMode`, `exposureTime`, `colorTemperature`, `iso` all commented-out FIXMEs. The inversion point (web exposes what native withholds) stands.

### Cross-section — the white-balance metric collision

- **Sections 1 and 3 both say auto white balance "drifted 18-23%". Section 3 then says "auto-WB 3.5%/4.5% versus locked 3.7%/4.2%".** -> **PROBLEM: two incompatible white-balance numbers for the same takes, never reconciled, never defined.** A reader cannot tell how an 18-23% drift becomes 3.5%. Both figures are in the evidence block, so both are presumably real and presumably different metrics (raw drift vs. detrended residual), but neither draft says which is which — and it matters enormously, because the 18-23% figure is what Section 1 uses to indict auto white balance, while the 3.5/4.5 vs 3.7/4.2 figures are what Section 3 uses to prove *no threshold exists*. As written the document appears to argue both that auto-WB is catastrophically worse and that it is indistinguishable. -> **FIX: name and define both metrics explicitly at first use (e.g. "raw WB excursion 18-23%" vs "detrended WB residual 3.5%/4.5%"), and confirm against the source data that they are what you think they are. If they cannot be reconciled, that is a finding, not a footnote.**

### Cross-section — the ground-truth number drifts against itself

- **Ground truth is "70-7/8 inches"; the GPS comparison uses "1.803 m"; the scale-precision claim uses "a 71-inch shelf".** -> **PROBLEM: 70-7/8" = 70.875" = 1.800 m, but 1.803 m = 71.0".** The document rounds 70-7/8" to 71" and carries the rounded value into a metric figure. The discrepancy is 3.2 mm, or **0.18%** — which is numerically indistinguishable from the 0.19% pick precision the document claims for that very reference. A paper arguing that scale error is bounded by pick precision on the reference cannot round the reference by the width of its own error bar. -> **FIX: use 1.800 m throughout, or state 71" as a deliberate rounding and requote the GPS ratio accordingly. (The tape's self-consistency does check: 35.25 + 0.375 + 35.25 = 70.875 exactly.)**

### Section 3 — one real contradiction, otherwise the strongest section

- **"Two quantities measure honestly ... Phase correlation between consecutive frames estimates displacement, and therefore whether SfM will find shared features."** -> **PROBLEM: contradicts the project's own measurement.** The evidence states phase correlation **aliases past half a frame and over-reports there — true 38% overlap reads as 62%**. Presenting it without caveat, in a section whose entire thesis is that instruments lie, and specifically under the words "measure honestly", is the exact error the section was written to expose. -> **FIX: add the caveat and get the direction right, because the direction is what makes it dangerous. Readings above ~50% are in the unaliased regime, so the 92-98% and 92-94% figures stand. But the instrument fails *upward* precisely in the failure regime: a genuinely bad capture reads as passable, so any reading near or below ~60% may be a disguised sub-50% failure. Phase correlation cannot be trusted to catch the thing a gate exists to catch.**
- **"auto 92–106%"** -> **PROBLEM: n=2 rendered as a range.** The evidence gives two takes at 92% and 106%. Section 3 gets this right for the detrended figures ("auto 18.0% and 16.1%") and wrong for the raw ones. -> **FIX: "two takes, 92% and 106%". Same in Section 1. With n=2 per arm, say so — the honesty is the point, and the detrending argument survives it.**
- **Verified clean in Section 3, including the arithmetic:** gap 16.1/6.2 = 2.60 -> "about 2.6×" ✓. Threshold argument ✓ — locked raw 15-59%, auto raw 92-106%, so a threshold near 75% does sort this data, and the concession is correct and well-reasoned. Luma 69→127 with R² = 94% ✓. WB overlap ✓ — locked 3.7% > auto 3.5% while locked 4.2% < auto 4.5%, so "locked is not even reliably lower" is exactly right. 3DGS quote verified verbatim: "starting from sparse points produced during camera calibration". NeRF-W quote verified verbatim: "it is incapable of modeling many ubiquitous, real-world phenomena in uncontrolled images, such as variable illumination or transient occluders". COLMAP "similar illumination conditions", "at least 3 images", "good texture" all verified verbatim.

### Section 2 — cleanest of the four; the folklore trap is in the truncated part

- **The 70-80% overlap rule** (section truncates at "### Overlap: what is sourced an…") -> **PROBLEM: confirmed as folklore-transfer, and the trap is worse than a missing citation.** The 70-80% figures are **aerial/drone mapping** guidance from vendor blogs and survey practice (60-80% frontal typical; 80% forward / 70% side for 3D). They are not COLMAP guidance, not from the 3DGS literature, and not derived for handheld indoor video. COLMAP's actual sourced requirement is a **co-visibility count** — "Make sure that each object is seen in at least 3 images – the more images the better" — not a percentage. Two further reasons the transfer fails: (a) aerial overlap is a photo-to-photo quantity between deliberately spaced exposures; your 92-98% is a **consecutive-frame** quantity from 24 fps video — different denominators, not comparable numbers; (b) **both takes cleared 70-80% comfortably (92-94% and 92-98%) and Take 1 was still rejected.** Overlap was never the discriminator here. -> **FIX: name 70-80% as drone-mapping folklore with a URL, state that it does not transfer to handheld video, and give COLMAP's 3-image co-visibility rule as the sourced requirement. Your own data is the strongest argument against the rule's relevance — use it.**
- **Bråtelund quote "that all cameras have distinct centres"** -> **PROBLEM: quote altered.** The paper (§4.1) reads: "We assume, here and throughout the rest of the paper, that all cameras have distinct **centers**." The draft anglicises the spelling inside quotation marks. Trivial, except that this document's entire proposition is quotation accuracy. -> **FIX: preserve source spelling inside quotes; use British spelling only in your own prose.**
- **"depth error grows with the square of depth for a fixed baseline"** -> **PROBLEM: correct standard stereo result, unsourced in a document that sources everything else.** -> **FIX: cite a textbook/primary source or mark it as a standard result.**
- **"exhaustive matching is quadratic in image count"** (Section 4) -> same minor issue, same fix.
- **Verified clean in Section 2:** 7-DOF gauge freedom (3 rotation + 3 translation + 1 scale) ✓. `H = K R K⁻¹` ✓ and the OpenCV quote verified verbatim: "But in the case of a rotating camera (pure rotation around the camera axis of projection, no translation), an arbitrary world can be considered." COLMAP quote verified verbatim: "Do not take images from the same location by only rotating the camera, e.g., make a few steps after each shot." The Bråtelund framing — that zero baseline is excluded from the theory's domain rather than being a hard case within it — is correct and is a genuinely good use of the source. The inside-out paragraph's explicit "UNVERIFIED locally" is exactly the right move and matches the evidence (all passing captures were bookshelf-scale).

### Coverage gap — evidence in the brief that no draft uses

Flagging in case the truncated tails don't cover it. Unused measured facts, several of which are the most quotable in the set: GPS one-point-per-clip with `LocationAccuracyHorizontal` = 19.785 m against a 1.803 m shelf (~1000× too coarse); IMU dead-reckoning 0.1° tilt → ~123 m error over 2 minutes; **ARKit VIO 0.14-1.47% relative ATE vs tape+click 0.2-0.6% — ARKit ties the tape, it does not beat it**; Scaniverse bbox 1.52 × 2.40 × 0.61 m verified metric; **scale pick precision 0.57% over a 24" tape vs 0.19% over a 71" shelf — longest baseline wins**; iOS Photos picker re-encoding 4K60 HEVC → 4K30 H.264 ~24.5 Mbps (half the frames gone, fix: record to Files); Blackmagic's 1080p/~6 Mbps proxy sharing an identical filename with the 4K/~36 Mbps master (discriminate by resolution, never name or size); OCR 38 detections → 32 spine reads → 0 resolved, with exact lookup at 1.000 on clean text. Section 2 is truncated immediately before its natural home for the scale/GPS/IMU/ARKit cluster — confirm they land there. The Photos-picker and proxy-filename traps belong in Section 1 or 4 and currently appear nowhere.


---

## References

Retrieved 2026-07-15 unless noted.


- https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaTrackCapabilities.idl (retrieved 2026-07-15) — PRIMARY. Verified directly in WebKit source: MediaTrackCapabilities declares whiteBalanceMode, zoom, torch, focusDistance as active members; exposureMode, focusMode, exposureTime, exposureCompensation, colorTemperature, iso exist only as commented-out FIXMEs. Confirms the project's claim exactly. NOTE: a web-search summarizer asserted Safari 'does not even consider supporting any of these features' incl. whiteBalanceMode/zoom/torch — the primary source contradicts that summarizer; the IDL was checked because of the discrepancy.

- https://developer.apple.com/documentation/avfoundation/avcapturedevice/setwhitebalancemodelocked(with:completionhandler:) (retrieved 2026-07-15) — PRIMARY. Apple documents white balance as an independently lockable AVCaptureDevice property, separate from exposure and focus modes. Supports the INFERRED mechanism for why AE/AF Lock does not hold WB.

- https://developer.apple.com/documentation/avfoundation/capture-device-white-balance (retrieved 2026-07-15) — PRIMARY. Apple: white balance can be configured automatically or manually controlled; a device lock is required to modify hardware properties. Structural support for the separate-lock inference.

- https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf (retrieved 2026-07-15) — PRIMARY. Lowe 2004, IJCV, §6.1. Full text extracted locally via pypdf. Exact wording verified: descriptor 'normalized to unit length'; contrast change 'canceled by vector normalization'; brightness change 'will not affect the gradient values'; 'the descriptor is invariant to affine changes in illumination'; exception named as 'non-linear illumination changes ... due to camera saturation'; 0.2 clamp then renormalize. This is the load-bearing citation for the section's central nuance — it partially CONTRADICTS the naive 'exposure drift breaks SfM' framing.

- https://arxiv.org/html/2404.04211v1 (retrieved 2026-07-15) — PRIMARY (Robust Gaussian Splatting). Per-image affine colour transform (3x3 matrix + 3D offset) for white-balance/exposure inconsistency; ablation Table 2: PSNR 21.71 -> 23.08 (+1.37 dB), SSIM .807 -> .817, LPIPS .395 -> .380. Motion-blur modelling adds only +0.21 dB (23.87 -> 24.08). Supports 'colour is absorbable downstream'; corroborates this project's demotion of colour to advisory.

- https://arxiv.org/abs/2403.13327 and https://arxiv.org/html/2403.13327v3 (retrieved 2026-07-15) — PRIMARY (Gaussian Splatting on the Move, ECCV 2024). Rolling shutter + motion blur compensation for handheld smartphone capture using VIO-estimated velocities; poses treated as non-static during exposure. Table 1 synthetic: rolling-shutter variant baseline 19.21 -> 35.84 PSNR on 'cozyroom'; motion blur compensation ~5-8 dB on that scene. Devices: Samsung S20 FE, Google Pixel 5, iPhone 15 Pro; Android phones described as having 'a known, and relatively large rolling-shutter readout time'. No readout time in ms is given for any device.

- https://colmap.github.io/tutorial.html (retrieved 2026-07-15) — PRIMARY (COLMAP docs). Verbatim capture guidance: 'Capture images at similar illumination conditions. Avoid high dynamic range scenes'; 'Capture images with good texture'; 'Capture images with high visual overlap ... each object seen in at least 3 images'; 'Capture images from different viewpoints'; 'If you use a video as input, consider down-sampling the frame rate.' Page does NOT address rolling shutter or blur.

- https://www.reddigitalcinema.com/red-101/flicker-free-video-tutorial (retrieved 2026-07-15) — MANUFACTURER guidance. States the rule as 'Choose a shutter speed equal to the lighting pulse rate divided by some integer' (120 Hz pulse rate in North America). Does NOT list 1/240 or 1/250 as safe. This confirms the project's integration math and refutes the widespread blog claim that 1/240 is flicker-free at 60 Hz.

- https://standards.ieee.org/standard/1789-2015.html (retrieved 2026-07-15) — PRIMARY (standard landing page; full text paywalled, not read). IEEE 1789-2015, 'Recommended Practices for Modulating Current in High-Brightness LEDs for Mitigating Health Risks to Viewers'. Cited only for the existence/framing of the twice-mains (120 Hz in the US) full-wave-rectified LED modulation phenomenon.

- https://arxiv.org/html/2508.14682v1 (retrieved 2026-07-15) — PRIMARY (GeMS). States that reliable feature matching requires sharp images and that COLMAP-based methods are rendered ineffective under severe motion blur. Supports the 'blur is unrecoverable, noise is not' asymmetry.

- https://arxiv.org/html/2510.12493 (retrieved 2026-07-15) — PRIMARY (BSGS). States camera poses recovered from motion-blurred images via COLMAP are 'often significantly biased', causing geometric misalignment in NeRF/3DGS. Source of the quoted 'significantly biased'.

- https://arxiv.org/pdf/1605.05791 (retrieved 2026-07-15) — PRIMARY (A Generic Framework for Assessing the Performance Bounds of Image Feature Detectors). Basis for 'spurious detections in moderate noise; missed and spurious detections in high noise'. NOTE: sourced via search-result summary, not fetched full text — see 'unverified'.

- https://colmap.github.io/tutorial.html — retrieved 2026-07-15 — COLMAP's own capture guidance: "Capture images with high visual overlap", "each object is seen in at least 3 images", and "Do not take images from the same location by only rotating the camera, e.g., make a few steps after each shot." Confirms COLMAP gives NO numeric overlap percentage.

- https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html — retrieved 2026-07-15 — Primary source for the pure-rotation degeneracy: H = K R K^-1, and the statement that under pure rotation with no translation "an arbitrary world can be considered" (i.e. the mapping is independent of scene depth).

- https://arxiv.org/html/2112.05074 — retrieved 2026-07-15 — Bråtelund, "Critical configurations for two projective views, a new approach". Ruled-quadric condition for critical configurations; explicitly assumes "all cameras have distinct centres", which is why zero-baseline/pure rotation lies outside this literature rather than within it.

- https://support.pix4d.com/hc/en-us/articles/115002471546 — retrieved 2026-07-15 — Pix4D image acquisition: at least 75% frontal / 60% side overlap general case, 80% flat agriculture, 85% forest, 90% thermal. Aerial/flight-planning only; page contains no indoor, terrestrial, or handheld guidance. Source for the claim that the 70-80% rule is aerial-specific.

- https://demuc.de/papers/schoenberger2016sfm.pdf — retrieved 2026-07-15 — Schönberger & Frahm, "Structure-from-Motion Revisited" (COLMAP). SfM reconstructions obtained up to an unknown similarity transform (7 DoF: rotation, translation, scale); gauge freedom.

- https://www.ri.cmu.edu/pub_files/pub2/morris_daniel_d_2001_1/morris_daniel_d_2001_1.pdf — retrieved 2026-07-15 — Morris, "Gauge Freedoms and Uncertainty Modeling for 3D Computer Vision". Formal treatment of gauge freedom and the 7-DoF similarity ambiguity in SfM/bundle adjustment.

- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ — retrieved 2026-07-15 — Kerbl, Kopanas, Leimkühler, Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (ACM TOG, 2023). 3DGS initializes from COLMAP SfM points, and therefore inherits COLMAP's gauge and its scale ambiguity.

- https://people.inf.ethz.ch/pomarc/pubs/GallupCVPR08.pdf — retrieved 2026-07-15 — Gallup et al., "Variable Baseline/Resolution Stereo". Depth error grows with the square of depth for fixed baseline; baseline/parallax vs matching-accuracy trade-off. Supports the arcs-beat-strafing and baseline-to-depth-ratio arguments.

- https://arxiv.org/pdf/1907.11917 — retrieved 2026-07-15 — "Triangulation: Why Optimize?" Small parallax angles yield large 3D uncertainty because triangulated points slide along the viewing ray without measurable image displacement.

- https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html — retrieved 2026-07-15 — OpenCV ArUco detection: sub-pixel corner refinement modes (CORNER_REFINE_NONE / SUBPIX / CONTOUR / APRILTAG), disabled by default; basis for the claim that fiducial corners are localized sub-pixel and repeatably rather than by human pick.

- https://github.com/meyerls/aruco-estimator — retrieved 2026-07-15 — Automatic scale-factor estimation for COLMAP via ArUco markers: triangulates marker corners in the sparse model and applies a 4x4 transform with scaling. Publishes no accuracy figures; documents that "pose estimation is not robust to false detections".

- https://developer.apple.com/documentation/arkit/arworldtrackingconfiguration — retrieved 2026-07-15 — Apple ARKit ARWorldTrackingConfiguration: world tracking reports device position relative to the environment in metric units (metres), which is what makes VIO a candidate scale source at all.

- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ — retrieved 2026-07-15 — Kerbl, Kopanas, Leimkühler, Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023 (ACM TOG). Source of the verbatim quote "starting from sparse points produced during camera calibration"; supports the claim that SfM/camera calibration is a prerequisite step whose failures are inherited downstream.

- https://colmap.github.io/tutorial.html — retrieved 2026-07-15 — COLMAP official tutorial, capture guidance bullets. Supports the quoted requirements: good texture, "similar illumination" conditions (avoid high dynamic range), "high visual overlap" with each object seen in at least 3 images, and images from different viewpoints. Primary source for what the reconstruction stage actually demands of a capture.

- https://doi.org/10.1016/j.patcog.2012.11.011 — retrieved 2026-07-15 — Pertuz, Puig, Garcia, "Analysis of focus measure operators for shape-from-focus", Pattern Recognition 46(5):1415-1432, 2013. Full text read via http://isp-utb.github.io/seminario/papers/Pattern_Recognition_Pertuz_2013.pdf (retrieved 2026-07-15). Supports, verbatim and verified against the extracted text: Laplacian-based operators "exhibit the best performance at low noise levels" but are "the most sensitive to noise, showing the greatest reduction in their quality measure"; "The sensitivity to image noise of Laplacian-based operators is a well known fact"; and that the best operator "strongly depends on the particular capturing device" (supports the non-transferability of absolute thresholds).

- https://doi.org/10.1109/34.709612 — retrieved 2026-07-15 — Subbarao & Tyan, "Selecting the Optimal Focus Measure for Autofocusing and Depth-from-Focus", IEEE TPAMI 20(8):864-870, 1998. Cited for the existence of a closed-form theoretical noise-sensitivity analysis of focus measures (AUM / ARMS error metrics). NOTE: full text not read — see unverified.

- https://www.cs.ucf.edu/~foroosh/subreg.pdf — retrieved 2026-07-15 — Foroosh (Shekarforoush), Zerubia, Berthod, "Extension of Phase Correlation to Subpixel Registration", IEEE Trans. Image Processing 11(3), 2002. Full text extracted and read. Supports the phase-correlation error taxonomy quoted in the section: the paper's error analysis enumerates "error due to nonoverlapped regions in the two images", "error due to aliasing", "border errors due to periodicity assumption", and error due to wide-band random noise; also shows that for downsampled (aliased) images the phase-correlation signal power "is not concentrated in a single peak, but rather in several coherent peaks".

- https://arxiv.org/abs/2008.02268 — retrieved 2026-07-15 — Martin-Brualla, Radwan, Sajjadi, Barron, Dosovitskiy, Duckworth, "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections", CVPR 2021. Source of the verbatim abstract quote that NeRF "is incapable of modeling many ubiquitous, real-world phenomena in uncontrolled images, such as variable illumination or transient occluders". Supports the claim that appearance/illumination variation across views is a real, documented reconstruction hazard.

- https://colmap.github.io/faq.html — retrieved 2026-07-15 — COLMAP FAQ. Consulted as corroboration (not quoted in the final section): states SIFT features work best with "moderate to high view overlap, sufficient scene texture, and captured under similar illumination conditions", independently supporting the overlap/texture/illumination requirements cited from the tutorial.

- https://doi.org/10.2352/J.ImagingSci.Technol.2011.55.3.030504 — retrieved 2026-07-15 — Kurihara, Aoki, Kobayashi, "Analysis of Sharpness Increase by Image Noise", J. Imaging Sci. Technol. 55(3), 2011. Full text read. Consulted and deliberately CUT from the final section: it finds that added noise increases *human perceived* sharpness on texture while decreasing it at edges. This is a perceptual result about human observers, not a result about the variance-of-Laplacian estimator, and citing it for the metric claim would have been a category error. Recorded here so a future reader does not re-find it and assume it supports the inflation claim.

- [1] COLMAP Tutorial (official docs, raw source: https://raw.githubusercontent.com/colmap/colmap/main/doc/tutorial.rst and https://colmap.github.io/tutorial.html) — retrieved 2026-07-15 — supports: CPU vs GPU SIFT extraction; "the GPU version is favorable, as it has a customized feature detection mode that often produces higher-quality features for high-contrast images"; OpenGL fallback requires an attached display; sequential matching for video; sparse reconstruction outputs (sparse cloud + intrinsics + extrinsics).

- [2] graphdeco-inria/gaussian-splatting README (https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/README.md) — retrieved 2026-07-15 — supports: CUDA Compute Capability 7.0+; 24 GB VRAM for paper-quality training; "C++ Compiler for PyTorch extensions"; "CUDA SDK 11 for PyTorch extensions"; compiler/SDK compatibility requirement; COLMAP required to prepare custom scenes; 30,000 default iterations with 7,000 checkpoint.

- [3] Brush — ArthurBrussee/brush (https://github.com/ArthurBrussee/brush) — retrieved 2026-07-15 — supports: Rust + Burn + CubeCL + wgpu stack; no CUDA requirement; "simple dependency free binaries"; COLMAP and Nerfstudio dataset input; macOS/Windows/Linux/Android/browser and AMD/Nvidia/Intel support; Apache-2.0 license.

- [4] gsplat LICENSE (https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE) — retrieved 2026-07-15 — supports: Apache License 2.0, "Copyright 2025 Nerfstudio Team".

- [5] Ye et al., "gsplat: An Open-Source Library for Gaussian Splatting" (https://arxiv.org/html/2409.06765v1) — retrieved 2026-07-15 — supports: PyTorch frontend with optimized CUDA kernel backend; Apache-2.0; 10% less training time and 4x less memory vs the reference; 5.6 GB vs 9.0 GB and 19.39 min vs 26.19 min at 30k iterations on MipNeRF360.

- [6] Kerbl, Kopanas, Leimkuehler, Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering", ACM TOG (SIGGRAPH) 2023 (https://arxiv.org/abs/2308.04079 ; project page https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — retrieved 2026-07-15 — supports: initialisation from "sparse points produced during camera calibration"; "interleaved optimization/density control ... notably optimizing anisotropic covariance"; "fast visibility-aware rendering algorithm that supports anisotropic splatting"; >=30 fps at 1080p.

- [7] Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng, "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020 (project page: https://www.matthewtancik.com/nerf) — retrieved 2026-07-15 — supports: NeRF represents scenes implicitly in an MLP queried per ray. NOTE: cited via project page; the arXiv record was not fetched directly in this pass.

- [8] graphdeco-inria/gaussian-splatting LICENSE.md (https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/LICENSE.md) — retrieved 2026-07-15 — supports: Inria/MPII licensors; research-use-only grant; "THE USER CANNOT USE, EXPLOIT OR DISTRIBUTE THE SOFTWARE FOR COMMERCIAL PURPOSES WITHOUT PRIOR AND EXPLICIT CONSENT OF LICENSORS"; commercial contact stip-sophia.transfert@inria.fr.

- [9] Ultralytics License (https://www.ultralytics.com/license) — retrieved 2026-07-15 — supports: AGPL-3.0 vs Enterprise dual licensing; AGPL-3.0 obligation to open-source the entire project; obligation extends to trained/fine-tuned models.

- [10] Apple Developer Forums, "Understanding PHPickerConfiguration" (https://developer.apple.com/forums/thread/736545) — retrieved 2026-07-15 — supports (REPORTED, not primary API reference): default preferredAssetRepresentationMode permits transcoding; .current avoids transcoding when the app handles arbitrary formats. The canonical API reference page (developer.apple.com/documentation/photokit/phpickerconfiguration) returned 404 on fetch and its body was not verified.

- https://colmap.github.io/tutorial.html — retrieved 2026-07-15 — verifies verbatim COLMAP capture guidance ('Do not take images from the same location by only rotating the camera, e.g., make a few steps after each shot.'; 'Make sure that each object is seen in at least 3 images – the more images the better.'; 'Capture images at similar illumination conditions.'; 'Capture images with good texture.'), the GPU-vs-CPU feature quality note, and the OpenGL 'requires an attached display' sentence. Note: these are on the tutorial page, NOT the FAQ.

- https://colmap.github.io/faq.html — retrieved 2026-07-15 — negative result: contains neither the GPU-favourable quote nor the sequential-matching quote attributed to '[1]' in Section 4.

- https://github.com/colmap/colmap/releases — retrieved 2026-07-15 — establishes latest COLMAP releases are 3.13.0 (2025-11-07), 3.12 (2025-06-30), 3.11 (2024-11-28); no 4.1 release exists.

- https://colmap.github.io/legacy.html — retrieved 2026-07-15 — shows the docs site building as 'COLMAP 4.1.0.dev0 | 43dd3bb2 (2026-03-16)', i.e. 4.1 is a main-branch dev version string, not a release.

- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ — retrieved 2026-07-15 — verifies verbatim the 3DGS abstract phrase 'starting from sparse points produced during camera calibration'.

- https://github.com/graphdeco-inria/gaussian-splatting — retrieved 2026-07-15 — verifies verbatim all INRIA requirements quoted in Section 4: 'CUDA-ready GPU with Compute Capability 7.0+', '24 GB VRAM (to train to paper evaluation quality)', 'C++ Compiler for PyTorch extensions', 'CUDA SDK 11 for PyTorch extensions', 'C++ Compiler and CUDA SDK must be compatible', 'install a recent version of COLMAP (ideally CUDA-powered) and ImageMagick'.

- https://github.com/ArthurBrussee/brush — retrieved 2026-07-15 — verifies 'produces simple dependency free binaries', 'works on a wide range of systems: macOS/windows/linux, AMD/Nvidia/Intel cards, Android, and in a browser', 'Brush takes in COLMAP data or datasets in the Nerfstudio format', 'uses WebGPU compatible tech and the Burn machine learning framework'. Negative result: no mention of CubeCL, wgpu, Vulkan, Metal, or DX12.

- https://github.com/nerfstudio-project/gsplat — retrieved 2026-07-15 — confirms Apache-2.0 license; README wording is 'CUDA accelerated rasterization of gaussians with python bindings', NOT 'highly optimized CUDA kernels'.

- https://arxiv.org/abs/2409.06765 — retrieved 2026-07-15 — gsplat paper; source of the actual phrase 'a back-end with highly optimized CUDA kernels' quoted in Section 4 and miscited to the repo.

- https://docs.gsplat.studio/main/ — retrieved 2026-07-15 — negative result: contains no license statement and not the 'highly optimized CUDA kernels' phrase.

- https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html — retrieved 2026-07-15 — verifies verbatim 'But in the case of a rotating camera (pure rotation around the camera axis of projection, no translation), an arbitrary world can be considered' and the s[x' y' 1]^T = K R K^-1 [x y 1]^T relation.

- https://arxiv.org/html/2112.05074 — retrieved 2026-07-15 — Bråtelund, 'Critical configurations for two projective views, a new approach'; §4.1 reads verbatim 'We assume, here and throughout the rest of the paper, that all cameras have distinct centers.' (American spelling; draft renders it 'centres').

- https://arxiv.org/abs/2008.02268 — retrieved 2026-07-15 — NeRF-W; verifies verbatim 'it is incapable of modeling many ubiquitous, real-world phenomena in uncontrolled images, such as variable illumination or transient occluders'.

- https://www.reddigitalcinema.com/red-101/flicker-free-video-tutorial — retrieved 2026-07-15 — verifies verbatim 'choose a shutter speed equal to the lighting pulse rate divided by some integer'. Negative result: does NOT print the 1/120, 1/60, 1/40, 1/30 list attributed to it; separately advises avoiding frame rates greater than the lighting pulse rate.

- https://standards.ieee.org/standard/1789-2015.html — retrieved 2026-07-15 — establishes IEEE 1789-2015's actual scope: 'Recommended Practices for Modulating Current in High-Brightness LEDs for Mitigating Health Risks to Viewers' — a human-health standard on modulation frequency/depth, not a camera-banding standard.

- https://developer.apple.com/documentation/avfoundation/avcapturedevice/setwhitebalancemodelocked(with:completionhandler:) — retrieved 2026-07-15 — confirms white balance is a separately lockable AVCaptureDevice property, distinct from exposure and focus modes; supports the AVFoundation-structure claim but not the AE/AF-Lock behavioural claim.

- https://github.com/WebKit/WebKit/blob/main/Source/WebCore/Modules/mediastream/MediaTrackCapabilities.idl — retrieved 2026-07-15 — verifies Section 1's IDL claim exactly: whiteBalanceMode, zoom, torch, focusDistance declared; exposureMode, focusMode, exposureTime, colorTemperature, iso all commented-out FIXMEs.

- https://pixelement.com/blog/_site/2024/06/17/importance-of-imagery-overlap.html — retrieved 2026-07-15 — representative source for the 70-80% overlap figures, showing their provenance is aerial/drone mapping vendor guidance rather than SfM or 3DGS literature.

- https://aircamdrone.co.uk/insights/optimising-overlap-in-drone-mapping-a-comprehensive-guide-for-operators-and-surveyors/ — retrieved 2026-07-15 — second instance of the 70-80% overlap rule in drone-mapping practice (80% forward / 70% side for 3D), supporting the folklore-transfer finding.

- https://www.urbanvideo.ca/avoid-video-flicker — retrieved 2026-07-15 — instance of the '1/240 at 60Hz' folklore, pairing 60 Hz with shutter speeds '1/60, 1/120, or 1/240'; usable as the citation for naming the folklore.
