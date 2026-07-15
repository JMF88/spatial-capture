# Vendored runtime dependencies

Third-party code, committed on purpose. Nothing here is authored in this repo.

**Why vendored rather than a CDN.** The viewer previously resolved `three` and `spark`
from jsdelivr and sparkjs.dev at runtime. That made a static portfolio page depend on two
other people's hosts being up at the moment someone opens it — and the failure was silent:
with the CDN unreachable the page hung on `Loading...` indefinitely, because an importmap
that cannot resolve means the module script never executes, so no error handler ever runs.
Verified by blocking the hosts and watching it hang. Vendoring also makes the page
auditable: everything it executes is in this repo, at a known hash.

**Why this is safe to vendor as single files.** Spark builds its sort workers from blob
URLs rather than fetching them (which is also why it needs no COOP/COEP headers, and why
it runs on plain GitHub Pages). There is no wasm, no dynamic `import()` of external
modules, and no CDN URL baked into the bundle. The transitive graph is small but not obvious, and closing it took a real render to
find: `three.module.js` re-exports from a sibling `./three.core.js` (three splits the
build in two as of 0.180), and Spark pulls `three/addons/postprocessing/Pass.js`. Miss
either and the page 404s a module and hangs on `Loading...` with **no JS error at all** —
an unresolved import means the module script never executes, so nothing is left alive to
report it. Downloading is not vendoring; rendering is. Everything reachable is below.

## Contents

| File | Package | Version | Licence | SHA-256 | Size |
|---|---|---|---|---|---|
| `three.module.js` | three | 0.180.0 | MIT | `c8211c69345d2e9949dc7a8ac9693804…` | 589 KB |
| `three.core.js` | three | 0.180.0 | MIT | `eb077d2417f61d3e6d9264c317cabc4e…` | 1,371 KB |
| `three-addons/controls/OrbitControls.js` | three (examples/jsm) | 0.180.0 | MIT | `b97879c748170baadeb3fb84cea1ffdf…` | 38 KB |
| `three-addons/postprocessing/Pass.js` | three (examples/jsm) | 0.180.0 | MIT | `444b409c235ead986893c472e720da1b…` | 4 KB |
| `spark.module.js` | @sparkjsdev/spark | 2.1.0 | MIT | `c0355a962f68a6de9b13df69f05b1aba…` | 5,254 KB |

Total: 7.1 MB.


## Provenance

Fetched verbatim, unmodified:

- `three.module.js`
  <https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js>
  `sha256:c8211c69345d2e9949dc7a8ac969380497aa0600a5a8ac6a459c8cd02dd9cb8a`
- `three.core.js`
  <https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.core.js>
  `sha256:eb077d2417f61d3e6d9264c317cabc4ea35769ed6b0ab533067292a550784c20`
- `three-addons/controls/OrbitControls.js`
  <https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js>
  `sha256:b97879c748170baadeb3fb84cea1ffdf4674e283dc06042f34e2acb95a76042c`
- `three-addons/postprocessing/Pass.js`
  <https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/postprocessing/Pass.js>
  `sha256:444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5`
- `spark.module.js`
  <https://sparkjs.dev/releases/spark/2.1.0/spark.module.js>
  `sha256:c0355a962f68a6de9b13df69f05b1aba3614d9aec43a4504975daeb349126a8a`

## Updating

Re-fetch from the URLs above at the new version, update this table's versions and hashes,
and re-run the viewer against a real splat before committing — Spark's loader error strings
and the addon import graph are both version-coupled. Verify offline afterwards: the point
of vendoring is that the page renders with every external host unreachable, so test it that
way rather than assuming.

Licences are upstream's and travel with the files; both are MIT, compatible with this
repo's MIT. This directory is excluded from the repo's own lint and test scope.
