# Bayesian GPR — project context

## Collaboration & scope

Joint work between **Ammar Mian** (LISTIC, USMB) and **Colin Fox** (Otago). Colin's MSc student **Stella** will build the **Bayesian model and the MCMC sampler** — priors, reverse moves, likelihood, posterior analysis are all **their** territory, drawing on their bootcamp template `colin_code/compute4.py`.

**Our deliverable is the GPR-physics half only**: a small Python library that gives them everything they need to evaluate a forward model and compare to data. They take it from there. We do not write the likelihood, the prior, or the sampler.

Concretely, we provide:

1. A **forward simulator** that turns a list of buried objects into a noise-free B-scan.
2. A **scene-building UX** so it's easy and fun to lay out a synthetic ground scene (and pull it back out as the ground truth `(atoms, image)` pair).
3. A **clutter representation** that adds realistic-looking background to a clean B-scan.
4. A `huber(r, delta)` function — just the norm itself, not a likelihood, not an optimizer. Stella decides how to embed it in her MH ratio.
5. One or two example **datasets** (a synthetic scene we render, plus an optional real Geolithe scan), and a **simulation script** (`scripts/make_demo_scene.py` — *not* a Jupyter notebook) that builds a scene end-to-end so they have a runnable starting point.

Anything beyond this — likelihoods, priors, reversible-jump moves, posterior summaries — is Colin & Stella's call. We avoid that to (a) respect their domain, (b) avoid imposing our Bayesian assumptions, and (c) keep our scope shippable.

## Forward-model summary (from DSIPS slides, `sec2_gpr_model.tex`)

B-scan matrix `Y ∈ R^{N_t × N_x}` (N_t time samples × N_x antenna positions):

```
Y = Σ_k  C_k ⊛ H_k   +   L     +   N
    └── target responses ──┘  └clutter┘ └noise┘
```

Each hyperbola atom (Terrasse 2016):

```
H_k(x, t) = r( t  −  (2/v) · √((x − x0)^2 + d^2) )
v = c / √eps_r                          ← wave speed in soil
r(t) = (1 − 2π² f0² t²) exp(−π² f0² t²) ← Ricker wavelet, center freq f0
```

Per-atom parameters: lateral position `x0`, depth `d`, soil permittivity `eps_r`, scatterer radius `R` (radius correction in `dico_phy.py`), amplitude.

### "Convolution" vs. "sum over atoms" — same operation, different API

Colin asked for "sum over objects, not convolution". They are mathematically equivalent — choose by what the caller wants:

```
Convolutional view (good for batch render of a whole scene):

   C_k (sparse coeff. map)            H_k (hyperbola atom template)
   ┌──────────────────┐               ┌──────────┐
   │ . . . . . . . .  │               │   ╱╲     │       one hyperbola per
   │ . . . 2 . . . .  │      ⊛        │  ╱  ╲    │  =    non-zero pixel of C_k,
   │ . . . . . . . .  │               │ ╱    ╲   │       scaled by that value
   └──────────────────┘               └──────────┘

Atom-list view (good for MCMC: change one atom, re-render only that one):

   atoms = [ (x0, d, R, eps_r, amp), ... ]   ──►   Σ render(atom)  =  image
```

Equivalence: each non-zero pixel of `C_k` at position `(x0, d)` with value `amp` ≡ one atom `(x0, d, amp)` of shape `H_k`. We expose **both** APIs from `forward.py`:

- `render_atoms(atoms, grid, wavelet) -> ndarray` — what Stella's MCMC will call. Iterates atoms; supports `render_atoms(..., base=cached_image, delta=[+atom, -atom])` for incremental updates.
- `simulate_bscan_conv(coeff_maps, atoms_dict, ...) -> ndarray` — FFT convolution path, for building synthetic scenes fast.

Both produce identical images (modulo FFT edge handling). Unit-tested for equivalence.

## What we reuse vs. rewrite from `GPR-robust-inversion/`

| Need                          | Already exists                                       | Action                                                       |
|-------------------------------|------------------------------------------------------|--------------------------------------------------------------|
| Hyperbola atom render         | `MIRAG/dictionary/dico_phy.py`                       | Reimplement clean & vectorized; reference the old file       |
| Convolutional forward         | `MIRAG/dictionary/dictionary.py`                     | Reuse the math; expose as `simulate_bscan_conv`              |
| Huber norm                    | `MIRAG/optim/huber_source_separation.py`             | Extract `huber(r, delta)`; drop ADMM machinery               |
| Low-rank clutter              | `source_separation.py` (nuclear-norm SVT)            | Provide as a **generator** (sample `L = U V^T`), not solver  |
| Real data                     | `data/IRADAR__00H.tar.gz` (Geolithe)                 | Ship as optional demo dataset                                |

**Do not import `GPR-robust-inversion/` as a dependency** — its YAML runners, HTCondor wrappers, scikit-learn estimators, and ADMM solvers are all irrelevant here.

## Clutter representation

Two backends, same `add_clutter(image, **params) -> image` signature so the demo script can swap them:

1. **`low_rank`** — generate `L = U V^T` with `U ∈ R^{N_t×k}`, `V ∈ R^{N_x×k}` drawn from a Gaussian, scaled to a target SNR. Cheap, matches the old paper's modeling assumption, easy for Stella to also write a *sampler* for inside her MCMC if she wants to estimate clutter.
2. **`gprmax_peplinski`** *(stretch / v2)* — load a pre-rendered "empty ground" B-scan produced by gprMax with the Peplinski heterogeneous-soil model, and add it. More realistic; lets Stella stress-test her sampler against clutter that does not match her assumed parametric form.

## Scene-building UX (the "fun and expressive" part)

Goal: a researcher should be able to write a 10-line script and see a realistic-looking B-scan come out, **and** get back the ground-truth atom list (so they can score Stella's posterior against truth).

API sketch:

```python
from bayesian_gpr import Scene, presets

s = Scene(x_extent=2.0, t_max=20e-9, dx=0.01, dt=0.05e-9, eps_soil=6.0)

# physical objects → compile to atoms
s.add(presets.Pipe(x=0.4, depth=0.3, radius=0.05, eps_r=1.0))     # air-filled
s.add(presets.Pipe(x=1.2, depth=0.6, radius=0.08, eps_r=80.0))    # water-filled
s.add(presets.RebarGrid(x_range=(0.5,1.5), depth=0.2, spacing=0.15))
s.add(presets.Void(x=1.7, depth=0.5, radius=0.15))

image = s.render(wavelet=presets.ricker(f0=1.5e9))
image = s.add_clutter(image, kind="low_rank", rank=3, snr_db=10)
image = s.add_noise(image, sigma=0.05)

s.save("data/demo_pipes.npz")     # bundles: image, atoms, grid, params
s.plot()                          # matplotlib B-scan with overlaid truth markers
```

Design notes:
- `presets.*` are *physical* object types (Pipe, RebarGrid, Void, BoulderField, LayerInterface). Each compiles to one or more `Atom` records — the user thinks in objects, the library thinks in atoms.
- `Scene` is a plain dataclass; everything is serializable to `.npz` (image, atom list, grid, wavelet params, RNG seed).
- `Scene.plot()` overlays the ground-truth hyperbola curves on the rendered B-scan — instant visual sanity check.
- A small CLI wrapper `bayesian-gpr make-scene scenes/demo.toml` for reproducible scene files (optional, only if it stays trivial).

Stella's loop will then just be: `scene = load_scene(...); D_obs = scene.image; ...her MCMC over atoms... compute = render_atoms(proposal, scene.grid, scene.wavelet); residual = D_obs - compute; cost = huber(residual, delta).sum()`.

## Data sources

1. **Synthetic** (default) — what the simulation script produces. Ground truth is exact, so posterior validation is trivial.
2. **gprMax** (optional, validated) — open-source FDTD EM simulator. Confirmed relevant features: B-scan from buried cylinder (built-in example), **Peplinski heterogeneous-soil** for fractal clutter, layered ground, surface roughness, antenna models (dipole, bowtie GSSI/MALÅ). Scenes are `.in` text files; install via conda. Integration plan: a thin `gprmax_io.py` that emits a `.in` from a `Scene` and loads the resulting `.out` HDF5 back as a NumPy array. **gprMax is not a runtime dep** — we ship 1–2 pre-rendered `.npz` scenes so users without it can still run everything. (Refs: https://github.com/gprMax/gprMax, https://docs.gprmax.com.)
3. **Real** — the Geolithe IRADAR scan already in `GPR-robust-inversion/data/IRADAR__00H.tar.gz`.

## Engineering practices

- **Python package** at repo root: `bayesian_gpr/` with `forward.py`, `scene.py`, `presets.py`, `clutter.py`, `huber.py`, `io.py`, `viz.py`.
- **`pyproject.toml`** (PEP 621) declaring deps (`numpy`, `scipy`, `matplotlib`, `h5py` for gprMax I/O). Optional extras: `[dev]` (pytest, ruff, mypy), `[gprmax]` (documentation pointer only — actually installed via conda).
- **Lockable env**: ship a `requirements.txt` *or* a `uv.lock` / `conda environment.yml`, pick one. Recommendation: `pyproject.toml` + `uv` (modern, fast, lockfile-friendly) and a separate `environment.yml` only for users who need gprMax.
- **Git** — `git init` the directory; `.gitignore` for `__pycache__`, `*.npz` in `data/` except the named demo scenes, `.venv`, etc. Conventional commits optional, single-author so not strict.
- **Tests** (`tests/`) covering: hyperbola apex lands at the correct two-way travel time; convolutional path matches atom-list path; clutter generator hits requested SNR; `huber` matches scipy's reference at edge cases.
- **Docs**: a real `README.md` (what / install / quickstart), docstrings on every public function, and `scripts/make_demo_scene.py` as the executable tutorial. No Jupyter notebooks for v1.
- **Style**: ruff + black defaults; type hints on public API; dataclasses over dicts.

## Open questions (for Ammar / Colin, not for us to decide alone)

1. **Clutter realism**: do we ship the low-rank generator only, or block the v1 release on getting one gprMax-Peplinski scene exported? Recommendation: ship low-rank first, add gprMax scene as a follow-up `data/` asset.
2. **Atom parameter ranges**: what physical ranges should `presets.*` defaults cover? Need sensible numbers for `eps_r`, depth, radius, antenna `f0` that match Colin/Stella's intended scenarios.
3. **Wavelet choice**: Ricker is the default. Do we also expose the differentiated Gaussian / Blackman-Harris used by GSSI antennas, or keep one shape for v1?

## Repo layout (as of 2026-05-24)

```
./colin_code/
  compute4.py         # Stella's reference RJMCMC implementation
  Compute4.pdf        # bootcamp assignment sheet
./GPR-robust-inversion/   # Ammar's old TGRS paper — reference only, do not depend on
  MIRAG/dictionary/{dico_phy.py, dictionary.py}
  MIRAG/optim/huber_source_separation.py
  data/IRADAR__00H.tar.gz
```

The new `bayesian_gpr/` package, `scripts/`, `tests/`, `data/`, `pyproject.toml` and `README.md` do not exist yet — they will be created in the next phase.

## Reference material

- **Ammar's DSIPS2026 slides** (LaTeX sources): `/Users/ammarmian/Documents/Research/Slides/DSIPS2026/sources/`
  - `sec2_gpr_model.tex` — signal model, Ricker wavelet, physical dictionary
  - `sec3_huber_gpr.tex` — Huber norm and robust inversion
- **Colin's bootcamp PDF**: `./colin_code/Compute4.pdf` — the RJMCMC template Stella will adapt.
- **Original physical dictionary paper**: Terrasse 2016 (cited in slides).
- **gprMax**: https://github.com/gprMax/gprMax — docs at https://docs.gprmax.com.

## Working agreement with Claude in this repo

- **Opus is used for planning only** (this conversation). Implementation tasks should be delegated to **Sonnet or Haiku** via subagents or by switching model — Opus is too expensive to burn on coding.
- Plan first, write a short design note, then have Sonnet execute against it.
