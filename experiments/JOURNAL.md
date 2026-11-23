# Experiment journal — excitation -> radiation pattern (32x32 array)

Layout
- `experiments/configs.py` — every experiment, with its hypothesis (id -> config)
- `experiments/run.py` — `python -m experiments.run <ids>`; short training until val
  loss converges (patience), step budget or time cap
- `runs/exp/<id>_<name>/` — config, log, learning curve, best checkpoint, result.json
- `experiments/LEADERBOARD.md` — `python -m experiments.leaderboard`
- `experiments/analysis/` — data diagnostics

Split: identical to the original `runs/main` (seed-0 stratified 600/75/75), so numbers
are comparable to the original hybrid (test: rmse_sph 3.19, top30 0.26, top10 0.17,
beam gain 0.13 dB, val loss 2.259 at best).

## Diagnostics (a01_normalisation.py)

- Sphere-integrated linear gain = -3.5 dB on average with std 0.7-0.8 dB per sample, so
  GainTotal is **not** normalised to radiated power: the normalisation depends on the
  excitation (scan/mismatch loss). The original physics stage normalises to directivity
  plus a constant efficiency, which cannot represent this; that matches its ~0.8 dB
  physics-only beam-gain error.
- 98% of pixels are > 40 dB below peak; the full-sphere RMSE is dominated by sidelobe /
  null structure, while the beam region is already good.
- Original hybrid overfits after ~20 min (train loss 0.15 vs val 2.3); physics-only
  val loss 2.91.

## Round 1 — physics-only ablations (P00-P09)

Hypotheses: (a) incident-power or quadratic-form normalisation, (b) two polarisation
channels (GainTotal = |Eθ|²+|Eφ|²), (c) position-dependent EEPs via rank-R channels with
element maps, (d) bigger coupling kernel, (e) finer element pattern, (f) sqrt amplitude.

Results (val loss; P00 = 2.914 reproduces the original physics stage's 2.911):
- **Incident-power normalisation (P01) is right**: beam-region error collapses
  (top10 0.75 -> 0.11 dB, beam gain 0.66 -> 0.10 dB, pointing 0.23 -> 0.05 deg) with
  4.7k params. Learned quadratic form (P02) is similar but not better.
- sqrt amplitude (P03) and no-floor (P07) both shift the whole pattern ~10 dB to fit the
  floor region -> amplitudes are field amplitudes; floor term is needed at rank 1.
- 2 pol (P04), 15x15 kernel (P06), finer element pattern (P08): no effect at rank 1.
- **Rank-4 coherent channels with per-channel element maps (P09): val 2.03**, beats the
  original 7.9M-param hybrid (2.26) on every test metric, physics only, 2.7 min.
- Residual analysis (a02 on P05): k-scale exactly 1.0 (frequency/pitch right). 76% of
  pixels are > 60 dB below peak; that region carries most of the loss. Model shows a
  flat -40 dBi floor where the simulation has 10-15 dB of structure.

## Round 2 — scaling the coherent model (P10-P18)

- P10 (rank 4, 2x steps) = P09: rank 4 is converged at 2.03.
- **P11 rank 8: val 1.52**, test rmse_sph 2.30 (orig hybrid 3.19). train ≈ val, no
  overfitting: the gain is from capacity of position-dependent EEPs.
- At rank 4: finer element pattern (P15) 2.03, 1 polarisation (P16) 2.02, 15x15 kernel
  (P14) 2.37 (worse, harder optimisation), no floor (P13) unstable (log of deep nulls).
  -> only the number of independent element-map channels matters.
- P17 random map init = edge init (2.02). P18 learned quadratic-form normalisation:
  2.02 loss but 2x worse beam gain -> keep incident-power normalisation.
- P12 rank 16 (6k steps): val 1.29, still improving at the last step.

## Round 3 — speed, rank scaling, rim sources, full-rank EEP

- Engineering: array factor as one complex GEMM against a precomputed [1024, 65160]
  phase matrix instead of the separable einsum: ~6x faster, far less memory
  (rank 32 @ batch 8: 83 ms/step).
- Note: running two training queues at once on this laptop GPU (WDDM time-slicing)
  made each ~12x slower; always queue sequentially.
- **P22 rank 32 (10k steps): val 1.04, test rmse_sph 1.74, top10 0.067, beam gain
  0.054 dB.** Rank scaling: 4 -> 2.03, 8 -> 1.52, 16 -> 1.29, 32 -> 1.04 (~-0.25 per
  doubling), still train ≈ val.
- P19 extended aperture (3 virtual rings): same loss as P09 (2.029), slightly better
  beam metrics. P20 (6 rings, 13x13 kernel): worse (2.39) — large kernels consistently
  hurt (P14 too); likely optimisation noise in many kernel taps.
- **P21 free per-element patterns (5.5M params): overfits hard (train 0.89, val 2.16).**
  The 600 training excitations are smooth steering tapers spanning a small subspace of
  the 1024-dim excitation space; unshared per-element freedom is unconstrained. Shared
  low-rank structure is the right inductive bias.
- **P23 rank 64: val 0.959, test rmse_sph 1.65, top10 0.067, beam gain 0.055 dB.**
  Residuals (a02): MAE in the >60 dB-below-peak floor 3.09 -> 1.03 dB, -50..-40 dB band
  1.52 -> 0.36 dB vs rank 1; the floor ripple is now tracked. Remaining: a few
  back-hemisphere null hot spots (theta 125-140) and faint horizon stripes.
- P24 rank 32 + 4 virtual rings: 1.034 vs 1.043 — rim sources are not the missing piece.

## Round 4 — optimisation / loss / generalisation at rank 16 (10k steps)

- P25 reference (lr 2e-2): 1.188. **P26 lr 1e-2: 1.164** and top30 0.31 -> 0.13.
  P27 lr 5e-2: 1.224. P28 batch 16: 1.156 (1.5x time).
- Bug found: early stopping fired at step 100 for P29/P30 — the near-rank-1 solution at
  step 100 beats the next ~1.2k steps while warmup scrambles the extra channels. Fixed
  with `min_steps_frac=0.3`; both rerun in round 5.
- **H04 U-Net residual on rank-32 physics: val 0.944 by step 1500** (physics alone
  1.043) — the residual net still adds on top of much better physics.

## Round 5 — high rank + better lr (1e-2), holdout reruns

- **P31 rank 64, lr 1e-2, 12k steps: val 0.938**, test rmse_sph 1.61, top30 0.102,
  top10 0.067, beam gain 0.053 dB. Best physics so far. Still improving at the last
  step (train 0.79 / val 0.94) — rank 64 is not done. P49 trains it 2x longer.
- Residual structure of P31 (a03, val): beams are solved (mae 0.06 within 10 dB of
  peak, 0.12 within 30 dB). Error lives in the floor: 76% of pixels are >60 dB below
  peak at mae 1.01 dB; the -50..-40 dB skirt is already 0.33 dB. Worst theta band is
  the horizon (85-95°, mae 1.16), then the near-back (95-120°, 1.10). A global
  angular bias map explains **1.8% of MSE**; pairwise |err|-map correlation is 0.03.
  Leftover error is excitation-specific, not a fixed pattern — so an incoherent
  w-dependent floor (P42) and a conditioned residual (H10-H12) are the right bets,
  and a w-independent dB map would be a waste.
- P34 rank 32 lr 1e-2: val 1.003 (P22 was 1.043). P33 1-pol: 1.004 — second
  polarisation is irrelevant once rank is high.
- P29 10x main-lobe boost: val 1.180 (worse overall), top30 0.104 vs P26 0.125 —
  the default 3x boost is enough.
- P30 holdout (train 1-2 beams): val 1.132, unseen 3-beam top30 0.178 vs in-distribution
  P26 top30 0.125. Linearity mostly holds; some 3-beam-specific structure remains.
- **H04 finished: val 0.909 at step 2400, then overfit** (train 0.25 vs val 0.91 at
  stop). The 8M U-Net on 600 samples is the wrong residual. Beam metrics are excellent
  (top10 0.039, beam gain 0.032) because that region is easy to memorise.

- **P32 rank 128: val 0.932** (P31 rank 64 was 0.938), test top10 0.075 vs P31 0.067.
  Train 0.71 / val 0.93. Doubling rank past 64 buys almost nothing and slightly
  hurts the beams — leftover error is not "need more coherent channels".
- **H05 unconditioned U-Net on P22: val 0.913** (H04 conditioned was 0.909). Same
  overfit curve (train 0.28 vs val 0.94 at the time cap). Dropping the weight encoder
  does not stop the U-Net memorising 600 patterns from the physics map alone.

## Round 6 — kernel size, seed noise, U-Net on P31

- P38 seed-1 of P34: val **1.0025 vs P34 1.0027**. Run-to-run noise is ~0.0002; treat
  0.01 as the smallest difference that means anything.
- Kernel at rank 32: **3x3 = 1.067** (too small), 7x7 = 1.003, 11x11 = 1.011
  (top30 0.28, optimisation noise). 7x7 is the sweet spot; P46 (1x1) is the
  remaining ablation.
- **H08 U-Net on 1-2 beam holdout: val 0.967** (P30 physics 1.132). Unseen 3-beam
  loss 1.400 vs P30 1.408 (no damage); unseen top30 0.178 -> 0.127. The residual
  does not break linearity-based generalisation.
- **H06 U-Net on P31: val 0.832** (physics 0.938, test rmse 1.38, top10 0.041).
  Still improving at the 45 min cap, but train 0.43 vs val 0.83 — the 8M net is
  again memorising. There is ~0.1 dB of real residual on top of the best physics;
  the job of H10-H13 is to take that without the overfit.

## Round 7 — missing mechanisms in the coherent map (queued after round 6)

The exact simulator is G(Ω) = Σ_pol |Σ_mn w_mn EEP_mn(Ω)|² / D(w). Rank-R conv+map
is a structured low-rank EEP. Leftover error after P31 is the floor, theta 125-140
null hot spots, and faint horizon stripes — not the beams. Each of P40-P50 adds one
physical degree of freedom at rank 32 (control P34 = 1.003) and stops at convergence:

| id | idea |
|---|---|
| P40 | finer epat (73×24) — horizon stripes as resolution |
| P41 | split front/back EEPs — back lobe is different physics |
| P42 | 4 incoherent projection channels — excitation-dependent floor |
| P43 | 4 incoherent AF channels — structured scatter |
| P44 | rank-1 global coupling per channel — non-local mutual coupling |
| P45 | ground-plane image factor — back-hemisphere interference |
| P46 | kernel 1×1 — is local conv still doing anything at rank 32? |
| P47 | D2 flip augmentation — 4× data if the sim is equivariant |
| P48 | theta knots concentrated at the horizon |
| P49 | rank 64 × 24k steps (P31 not converged) |
| P50 | P42+P44 together |

Round 7 results so far (control P34 = 1.003, noise 0.0002):
- **P43 incoherent AF × 4: val 0.868**, test rmse 1.42, top10 0.067. Train 0.77 /
  val 0.87, healthy. This is the missing mechanism: leftover floor is structured
  scatter with array-factor lobes that must NOT interfere with the coherent field.
  Beats rank-64 physics (0.938) at rank 32, within 0.04 of the 8M U-Net (H06 0.832).
  Residual vs P31: floor mae 1.01 -> 0.95, horizon 1.16 -> 1.04. Leftover is still
  excitation-specific (pairwise corr 0.03), so more scatter channels may help.
- P42 incoherent projections (no AF): 1.004 — the scatter has grating-lobe structure,
  not a smooth |<w,B>|² |e(Ω)|² map.
- P44 global coupling 1.120, P41 split EEP 1.054, P46 kernel 1×1 1.136: all worse
  (extra freedom / missing local conv). P45 image factor collapsed (1.59, early stop).
- P48 horizon knots 1.007 (no gain). P40 finer epat 1.027 (slightly worse). Angular
  resolution of the EEP was not the bottleneck.
- **P47 D2 flip aug: 2.60, collapsed.** The simulated array is not equivariant under
  x/y flips (feeds / ground plane / coordinate convention). Do not augment.

## Round 8 — cheap residuals on frozen P34 (control 1.003)

- H10 SphereFiLM 0.999, H12 field-mul 0.999: leftover is not a smooth
  excitation-dependent pattern.
- H11 local 1×1 MLP 0.992: a little floor recalibration, cannot move structure.
- **H13 tiny U-Net (8-16-32): val 0.921**, train 0.73 / val 0.92. Spatial mixing is
  required, but 8M params were overkill — this is most of H04's 0.909 without the
  train 0.25 collapse. H21 repeats it on P43.

## Round 9 — scale P43 (queued after P49)

- P49 rank 64 × 24k: val 0.933 vs P31 0.938. Extra steps at rank 64 are not the
  missing piece (same lesson as rank 128).
- Next: P51/P52 more scatter channels, P53 rank 64 + 4 incoh AF, P54 longer P43,
  H18-H21 cheap residuals on frozen P43.
- **P51 8 incoh AF: val 0.830** (194k params) matches the 8M U-Net H06 (0.832).
- **P52 16 incoh AF: val 0.788**, test rmse 1.29, top10 0.068, train 0.67 / val 0.79.
  Scatter-channel scaling: 0 -> 1.003, 4 -> 0.868, 8 -> 0.830, 16 -> 0.788. Beams
  unchanged (~0.067). This is how the floor is generated. Round 10 tries 32 channels,
  longer 16-ch, and rank-64 + 16-ch.
- **P55 32 incoh AF: val 0.708**, test rmse 1.16, train 0.58 / val 0.71, still
  falling at step 10k. Scaling has not saturated: 0/4/8/16/32 = 1.003 / 0.868 /
  0.830 / 0.788 / 0.708. P58 tries 64.
- **P58 rank 32 + 64 incoh AF: val 0.646**, test rmse 1.07, train 0.51 / val 0.65.
  Scatter scaling still not saturated. Beats P57 (rank 64 + 16). Scatter rank is
  the dominant knob. P61 tries 128; P62 is rank 64 + 64.
- **P59 rank 64 + 32 incoh AF: val 0.635**, test rmse 1.05. New best — combining
  both knobs beats more scatter alone (P58) and more steps on fewer scatter
  channels (P60 0.655 vs P57 0.678). Still improving at the last step.
- **P61 rank 32 + 128 scatter: val 0.607**, test rmse 1.02. 64→128 still helps
  (0.646→0.607) but is flattening.
- **P62 rank 64 + 64 scatter: val 0.580**, test rmse 0.97. Current best. Train 0.43 /
  val 0.58, last step still the best. P63/P64 scale this pair.
- **P63 rank 64 + 128 scatter: val 0.528**, test rmse 0.90. New best. 64→128 at
  rank 64 still drops ~0.05 (same as 32→64). Not saturated. Train 0.36 / val 0.53.
- P64 (64+64 × 20k): 0.544 — extra steps help (0.580→0.544) but extra scatter
  helps more. P65/P66 continue that pair.
- **P65 rank 64 + 256 scatter: val 0.609**, test rmse 1.05. Worse than P63 (0.528).
  1.7M params, batch 4, still falling slowly at 12k (train 0.41 / val 0.61) but
  128-ch is the better operating point at this data size. Scatter scaling
  saturated (or the batch/LR did not keep up) between 128 and 256.
- **P66** (P63 × 20k) stopped mid-run at step 17000: train 0.367 / val **0.518**.
  Extra steps on 64+128 still help a little (0.528 → 0.518). No checkpoint saved;
  resume with `python -m experiments.run P66`. Stopped for the day.
- H23 tiny U-Net on P55: 0.705 vs 0.708. Once scatter is in the physics, a spatial
  residual barely moves the loss (it still shaves top10 0.067 -> 0.054).
- P56 (16-ch × 16k): 0.733 — extra steps help 16-ch (0.788 -> 0.733) but 32-ch at
  10k still wins.
- **P57 rank 64 + 16 incoh AF: val 0.678**, test rmse 1.13, train 0.54 / val 0.68,
  best at the last step. Combining coherent rank with scatter rank is the way.
  H22 tiny U-Net on P52: 0.786 (nothing). P59/P60 scale this combination.
- P54 (4-ch × 16k): 0.854 vs P43 0.868 — extra steps help a little; extra scatter
  channels help a lot.
- P53 (rank 64 + 4 incoh): 0.812, worse than rank 32 + 16 incoh. Scatter rank is the
  better knob.
- Residuals on frozen P43: FiLM/MLP/fieldmul ~0.865-0.868 (nothing). Tiny U-Net
  0.855, but top10 0.067 -> 0.041. Once scatter is in the physics, the U-Net has
  little floor left to fix and mostly polishes beams.

## Round 8 — residuals that should not 8M-overfit (frozen physics)

H10 SphereFiLM, H11 local 1×1 MLP, H12 field-multiplicative correction, H13 tiny
U-Net (8-16-32), then the winners on P31 (H14/H15/H17). Frozen physics so the
number is the residual's own contribution.

## Round 11 � residual on P66 / floor DoF (results)

- P66 best at step 17k: val **0.518**. P67 same-arch FT: **0.494**. P68 harm-16: **0.491**.
  P69 front/back scatter: **0.487** (current best physics).
- H24 tiny U-Net on P66: 0.509 (beams only). H25/H26 scatter/chscale: nothing.
- H27/H28 tiny U-Net on P68/P69: val **0.485 / 0.481**, top10 0.064 -> 0.032.
  Spatial residual still ~1% on loss; ~2x on beams.
- H29 med U-Net crashed before first eval (OOM at batch 8) � rerun batch 4 in Round 12.

## Round 12 � overnight push (queued)

Longer FT + scatter resolution/rank + U-Net on new checkpoints. Sequential on the
4070 (do not parallelise). Warm-start loader now pads scatter channels / harmonics /
theta knots without touching Fourier bases.

Queue (~6-8 h): H29 H30 P70 H31 H34 P71 H32 P72 H33 P73 P74 H35
