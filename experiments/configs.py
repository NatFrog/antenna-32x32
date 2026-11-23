# Experiment registry. Each entry states the hypothesis it tests.

#Families
#  P*  physics-only coherent quadratic-form models (CoherentPhysics)
#  H*  physics + residual (U-Net, SphereFiLM, local MLP, field-multiplicative)


CONFIGS = {}


def reg(id, name, desc, model, train=None):
    CONFIGS[id] = {"id": id, "name": name, "desc": desc, "model": {"type": "phys"} | model,
                   "train": train or {}}


BASE = dict(n_pol=1, rank=1, kernel=7, elem_map="shared", norm="directivity", floor=True)

# ---- Round 1: what does the physics stage need? -------------------------------------
reg("P00", "baseline_phys", "re-implementation of the original physics stage", BASE)
reg("P01", "incident_norm", "gain normalised by incident power sum|w|^2 (scan loss "
    "learned through coupling) instead of directivity+const", BASE | dict(norm="incident"))
reg("P02", "quad_norm", "gain normalised by learned quadratic form |c_D*w|^2",
    BASE | dict(norm="quad"))
reg("P03", "sqrt_amp", "drive table amplitudes are powers: field ~ sqrt(|w|)",
    BASE | dict(amp="sqrt"))
reg("P04", "two_pol", "two incoherent coherent channels (E_theta/E_phi) fill nulls",
    BASE | dict(n_pol=2))
reg("P05", "two_pol_incident", "P01 + P04", BASE | dict(n_pol=2, norm="incident"))
reg("P06", "kernel15", "larger coupling kernel 15x15", BASE | dict(kernel=15))
reg("P07", "no_floor", "drop the incoherent floor term", BASE | dict(floor=False))
reg("P08", "finer_epat", "finer element pattern: 61 theta knots, 16 phi harmonics",
    BASE | dict(knots=61, harm=16))
reg("P09", "rank4_edge", "rank-4 channels with per-channel element maps (edge init), "
    "2 pol, incident norm", BASE | dict(n_pol=2, rank=4, elem_map="channel",
                                        map_init="edge", norm="incident"))

# ---- Round 2: scale the coherent low-rank EEP model (P09 was the best so far) -------------
R4 = BASE | dict(n_pol=2, rank=4, elem_map="channel", map_init="edge", norm="incident")
LONG = dict(steps=6000, patience=10)
reg("P10", "rank4_long", "P09 trained 2x longer (is it converged?)", R4, LONG)
reg("P11", "rank8", "rank 8", R4 | dict(rank=8), LONG)
reg("P12", "rank16", "rank 16", R4 | dict(rank=16), LONG | dict(batch=4))
reg("P13", "rank4_nofloor", "rank 4 without incoherent floor: is low-level structure "
    "coherent?", R4 | dict(floor=False), LONG)
reg("P14", "rank4_k15", "rank 4 + 15x15 coupling kernels", R4 | dict(kernel=15), LONG)
reg("P15", "rank4_finepat", "rank 4 + finer element patterns (61 knots, 16 harm)",
    R4 | dict(knots=61, harm=16), LONG)
reg("P16", "rank4_1pol", "rank 4, single polarisation (is the 2nd pol needed?)",
    R4 | dict(n_pol=1), LONG)
reg("P17", "rank4_randinit", "rank 4 with random (not edge) element-map init",
    R4 | dict(map_init="random"), LONG)
reg("P18", "rank4_quad", "rank 4 with learned quadratic-form normalisation",
    R4 | dict(norm="quad"), LONG)

# ---- Round 3: sources beyond the element grid (ground-plane rim diffraction) ---------
reg("P19", "rank4_ext3", "rank 4, aperture extended by 3 virtual element rings fed "
    "through the coupling kernel", R4 | dict(ext=3), LONG)
reg("P20", "rank4_ext6_k13", "rank 4, 6 virtual rings, 13x13 kernel (reach the rim)",
    R4 | dict(ext=6, kernel=13), LONG)
reg("P21", "rank4_eep37x72", "rank 4 + free per-element complex patterns on a 5 deg "
    "knot grid (full-rank EEP correction, 5.5M params), 1 pol",
    R4 | dict(eep=[37, 72], n_pol=1), LONG)

# ---- Round 3b: rank scaling with the GEMM array factor (6x faster) -------------------
XL = dict(steps=10000, patience=12, max_min=45)
reg("P22", "rank32", "rank 32", R4 | dict(rank=32), XL)
reg("P23", "rank64", "rank 64", R4 | dict(rank=64), XL)
reg("P24", "rank32_ext4", "rank 32 + 4 virtual rings", R4 | dict(rank=32, ext=4), XL)
R32 = R4 | dict(rank=32)

# ---- Round 4: optimisation, loss and generalisation at rank 16 (cheap) --------------
R16 = R4 | dict(rank=16)
reg("P25", "r16_xl", "rank 16, 10k steps (reference for this round)", R16, XL)
reg("P26", "r16_lr1e-2", "rank 16, lr 1e-2", R16, XL | dict(lr=1e-2))
reg("P27", "r16_lr5e-2", "rank 16, lr 5e-2", R16, XL | dict(lr=5e-2))
reg("P28", "r16_b16", "rank 16, batch 16", R16, XL | dict(batch=16))
reg("P29", "r16_boost10", "rank 16, lr 1e-2, 10x weight on the top-30 dB region "
    "(compare to P26)", R16, XL | dict(loss="l1db_boost10", lr=1e-2))
reg("P30", "r16_holdout_nb3", "rank 16 trained on 1-2 beam patterns only; 3-beam "
    "patterns are unseen (generalisation by linearity)", R16,
    XL | dict(train_beams=[1, 2], lr=1e-2))

# ---- Round 5: high rank with the better optimiser settings (lr 1e-2) -----------------
XL2 = XL | dict(lr=1e-2)
reg("P31", "r64_lr1e-2_12k", "rank 64, lr 1e-2, 12k steps", R4 | dict(rank=64),
    XL2 | dict(steps=12000, max_min=50))
reg("P32", "r128_lr1e-2", "rank 128, lr 1e-2, 9k steps", R4 | dict(rank=128),
    XL2 | dict(steps=9000, max_min=55))
reg("P33", "r32_1pol_lr1e-2", "rank 32, single polarisation", R32 | dict(n_pol=1), XL2)
reg("P34", "r32_lr1e-2", "rank 32, lr 1e-2 (control for P33)", R32, XL2)

# ---- Round 6: noise floor of comparisons, kernel size at high rank -------------------
reg("P35", "r32_k3", "rank 32, 3x3 coupling kernels", R32 | dict(kernel=3), XL2)
reg("P36", "r32_k11", "rank 32, 11x11 coupling kernels", R32 | dict(kernel=11), XL2)
reg("P38", "r32_lr1e-2_seed1", "P34 with another data-order seed (run-to-run noise)",
    R32, XL2 | dict(seed=1))


def hyb(id, name, desc, phys_id, phys_model, model=None, train=None):
    CONFIGS[id] = {"id": id, "name": name, "desc": desc,
                   "model": {"type": "hybrid", "init_from": phys_id, "unet_ch": [32, 64, 128, 256]}
                   | phys_model | (model or {}),
                   "train": HYB_TRAIN | (train or {})}


HYB_TRAIN = dict(steps=6000, batch=8, lr=4e-4, eval_every=300, patience=5, max_min=45,
                 ema=0.999, phys_lr_scale=0.25)
hyb("H01", "unet_on_P09", "original U-Net residual on top of the rank-4 physics (P09)",
      "P09", R4)
hyb("H02", "unet_nocond_P09", "U-Net residual without excitation conditioning: maps the "
      "physics pattern to a correction only (less overfitting?)", "P09", R4,
      dict(cond=False))
hyb("H03", "unet_small_P09", "4x smaller U-Net (16..128 ch) on P09", "P09", R4,
      dict(unet_ch=[16, 32, 64, 128]))
hyb("H04", "unet_on_P22", "original U-Net residual on the rank-32 physics (P22)", "P22", R32)
hyb("H05", "unet_nocond_P22", "unconditioned U-Net residual on P22", "P22", R32,
    dict(cond=False))
hyb("H06", "unet_on_P31", "U-Net residual on rank-64 physics (P31)", "P31",
    R4 | dict(rank=64))
hyb("H08", "unet_holdout_nb3", "U-Net residual on P30, trained on 1-2 beams only: does "
    "the residual net hurt generalisation to unseen 3-beam patterns?", "P30", R16,
    train=dict(train_beams=[1, 2]))

# ---- Round 7: missing physical mechanisms at rank 32 (control = P34, val 1.003) ------
# Remaining P31 error is floor + back-hemisphere (theta 125-140) + faint horizon stripes.
# Beam region is already ~0.05 dB; these tests target the leftover structure.
reg("P40", "r32_finepat", "rank 32 + finer epat (73 knots, 24 phi harmonics): are "
    "horizon stripes a pattern-resolution limit?", R32 | dict(knots=73, harm=24), XL2)
reg("P41", "r32_split", "rank 32, independent front/back hemisphere element patterns "
    "(back-lobe physics is not a smooth continuation of the forward EEP)",
    R32 | dict(epat_split=True), XL2)
reg("P42", "r32_incoh4", "rank 32 + 4 incoherent projection channels: excitation-"
    "dependent floor that does not interfere (current floor is one shape * |w|^2)",
    R32 | dict(incoh=4), XL2)
reg("P43", "r32_incohaf4", "rank 32 + 4 incoherent AF channels (structured scattered "
    "power with grating-lobe structure)", R32 | dict(incoh=4, incoh_mode="af"), XL2)
reg("P44", "r32_glob", "rank 32 + per-channel rank-1 global coupling: non-local mutual "
    "coupling a 7x7 kernel cannot represent", R32 | dict(glob=True), XL2)
reg("P45", "r32_image", "rank 32 + learnable ground-plane image factor E*(1+Gamma*"
    "exp(2jkh cos theta)) for back-hemisphere nulls", R32 | dict(image=True), XL2)
reg("P46", "r32_k1", "rank 32, 1x1 kernels (no spatial coupling, only element maps): "
    "does local conv still matter at this rank?", R32 | dict(kernel=1), XL2)
reg("P47", "r32_aug", "rank 32 with D2 flip augmentation of the square array (4x "
    "effective excitations if the sim is equivariant)", R32, XL2 | dict(aug=True))
reg("P48", "r32_horizonknots", "rank 32, theta knots concentrated at the horizon",
    R32 | dict(knot_mode="horizon"), XL2)
reg("P49", "r64_24k", "rank 64 trained 24k steps: P31 was still improving at step 12k",
    R4 | dict(rank=64), XL2 | dict(steps=24000, max_min=55))
reg("P50", "r32_incoh_glob", "rank 32 + incoherent projections + global coupling "
    "(the two mechanisms that should not overlap)",
    R32 | dict(incoh=4, glob=True), XL2)

# ---- Round 8: residual heads that should not 8M-overfit (H04 train 0.25 / val 0.91)
CHEAP = dict(steps=4000, batch=8, lr=1e-3, eval_every=200, patience=6, max_min=25,
             ema=0.999, phys_lr_scale=0.1, freeze_phys=True)
hyb("H10", "film_on_P34", "smooth excitation-conditioned sphere residual (FiLM) on "
    "frozen P34: leftover error as a scan-dependent element pattern", "P34", R32,
    dict(res="film"), CHEAP)
hyb("H11", "localmlp_on_P34", "1x1 angle-wise MLP residual on frozen P34: can errors "
    "be fixed locally in angle (no lobe moving)?", "P34", R32, dict(res="localmlp"),
    CHEAP)
hyb("H12", "fieldmul_on_P34", "complex multiplicative field correction on frozen P34 "
    "(residual in E, interference preserved)", "P34", R32, dict(res="fieldmul"), CHEAP)
hyb("H13", "tinyunet_on_P34", "tiny U-Net (8-16-32) on frozen P34: how much of H04 "
    "was capacity vs overfit?", "P34", R32, dict(res="unet", unet_ch=[8, 16, 32]),
    CHEAP)
hyb("H14", "film_on_P31", "sphere FiLM residual on frozen best physics (P31 rank 64)",
    "P31", R4 | dict(rank=64), dict(res="film"), CHEAP)
hyb("H15", "localmlp_on_P31", "local MLP residual on frozen P31", "P31",
    R4 | dict(rank=64), dict(res="localmlp"), CHEAP)
hyb("H17", "fieldmul_on_P31", "field-space residual on frozen P31", "P31",
    R4 | dict(rank=64), dict(res="fieldmul"), CHEAP)

# ---- Round 9: scale the P43 win (incoherent AF scatter) ------------------------------
IAF = R32 | dict(incoh=4, incoh_mode="af")
reg("P51", "r32_incohaf8", "rank 32 + 8 incoherent AF channels (scale the P43 win)",
    R32 | dict(incoh=8, incoh_mode="af"), XL2)
reg("P52", "r32_incohaf16", "rank 32 + 16 incoherent AF channels",
    R32 | dict(incoh=16, incoh_mode="af"), XL2)
reg("P53", "r64_incohaf4", "rank 64 + 4 incoherent AF channels: best coherent rank "
    "plus the missing scatter term",
    R4 | dict(rank=64, incoh=4, incoh_mode="af"), XL2 | dict(steps=12000, max_min=50))
reg("P54", "r32_incohaf4_16k", "P43 trained 16k steps (val still falling at 10k)",
    IAF, XL2 | dict(steps=16000, max_min=30))
hyb("H18", "film_on_P43", "sphere FiLM residual on frozen P43 (best cheap physics)",
    "P43", IAF, dict(res="film"), CHEAP)
hyb("H19", "localmlp_on_P43", "local MLP residual on frozen P43", "P43", IAF,
    dict(res="localmlp"), CHEAP)
hyb("H20", "fieldmul_on_P43", "field-multiplicative residual on frozen P43", "P43", IAF,
    dict(res="fieldmul"), CHEAP)
hyb("H21", "tinyunet_on_P43", "tiny U-Net on frozen P43", "P43", IAF,
    dict(res="unet", unet_ch=[8, 16, 32]), CHEAP)

# ---- Round 10: scatter-rank scaling ( still falling) ---------------
reg("P55", "r32_incohaf32", "rank 32 + 32 incoherent AF channels (continue P52 scaling)",
    R32 | dict(incoh=32, incoh_mode="af"), XL2 | dict(max_min=50))
reg("P56", "r32_incohaf16_16k", "P52 trained 16k steps (still falling at 10k)",
    R32 | dict(incoh=16, incoh_mode="af"), XL2 | dict(steps=16000, max_min=30))
reg("P57", "r64_incohaf16", "rank 64 coherent + 16 incoherent AF (combine the two knobs)",
    R4 | dict(rank=64, incoh=16, incoh_mode="af"), XL2 | dict(steps=12000, max_min=55))
hyb("H22", "tinyunet_on_P52", "tiny U-Net on frozen P52 (best physics so far)",
    "P52", R32 | dict(incoh=16, incoh_mode="af"), dict(res="unet", unet_ch=[8, 16, 32]),
    CHEAP)
reg("P58", "r32_incohaf64", "rank 32 + 64 incoherent AF channels (P55 32-ch still scaling)",
    R32 | dict(incoh=64, incoh_mode="af"), XL2 | dict(max_min=55))
hyb("H23", "tinyunet_on_P55", "tiny U-Net on frozen P55 (32 scatter channels)",
    "P55", R32 | dict(incoh=32, incoh_mode="af"), dict(res="unet", unet_ch=[8, 16, 32]),
    CHEAP)
reg("P59", "r64_incohaf32", "rank 64 coherent + 32 incoherent AF",
    R4 | dict(rank=64, incoh=32, incoh_mode="af"), XL2 | dict(steps=12000, max_min=55))
reg("P60", "r64_incohaf16_20k", "P57 trained 20k steps (best_step was the last one)",
    R4 | dict(rank=64, incoh=16, incoh_mode="af"), XL2 | dict(steps=20000, max_min=55))
reg("P61", "r32_incohaf128", "rank 32 + 128 incoherent AF (P58 64-ch still scaling)",
    R32 | dict(incoh=128, incoh_mode="af"), XL2 | dict(max_min=55))
reg("P62", "r64_incohaf64", "rank 64 coherent + 64 incoherent AF",
    R4 | dict(rank=64, incoh=64, incoh_mode="af"), XL2 | dict(steps=12000, max_min=55))
reg("P63", "r64_incohaf128", "rank 64 coherent + 128 incoherent AF",
    R4 | dict(rank=64, incoh=128, incoh_mode="af"), XL2 | dict(steps=12000, max_min=70))
reg("P64", "r64_incohaf64_20k", "P62 trained 20k steps (still falling at 12k)",
    R4 | dict(rank=64, incoh=64, incoh_mode="af"), XL2 | dict(steps=20000, max_min=70))
reg("P65", "r64_incohaf256", "rank 64 + 256 incoherent AF (P63 128-ch still scaling)",
    R4 | dict(rank=64, incoh=256, incoh_mode="af"),
    XL2 | dict(steps=12000, max_min=90, batch=4))
reg("P66", "r64_incohaf128_20k", "P63 trained 20k steps",
    R4 | dict(rank=64, incoh=128, incoh_mode="af"), XL2 | dict(steps=20000, max_min=90))

# ---- Round 11: residual on the best physics, and floor-specific degrees of freedom ---
# P66 (stopped at step 17k) is the best checkpoint. P63 val error is 0.06 dB
# in the beam and 0.59 dB in the >60 dB floor, which is 87% of the weighted loss, and
# only 3% of that MSE is a shared angular bias. Additive U-Nets on weaker physics
# polished beams and memorised the floor, try more combinations before Monday.
BEST = R4 | dict(rank=64, incoh=128, incoh_mode="af")
FT = dict(steps=4000, batch=8, lr=3e-4, warmup=0.02, eval_every=200, patience=10,
          max_min=40, min_steps_frac=0.4)
CHEAP66 = CHEAP | dict(steps=3000, max_min=40, eval_every=200)
hyb("H24", "tinyunet_on_P66", "tiny U-Net (8-16-32) on frozen P66: does a spatial dB "
    "residual still move the loss once 128 scatter channels are in the physics?",
    "P66", BEST, dict(res="unet", unet_ch=[8, 16, 32]), CHEAP66)
hyb("H25", "scatterunet_on_P66", "low-pass U-Net (no skips, +/-3 dB) multiplies scatter "
    "power on frozen P66: floor correction that cannot repaint the beams or the speckle",
    "P66", BEST, dict(res="scatter", unet_ch=[8, 16, 32]), CHEAP66)
hyb("H26", "chscale_on_P66", "per-excitation +/-6 dB gain per scatter channel on frozen "
    "P66: is the floor the wrong mix of existing scatter patterns?",
    "P66", BEST, dict(res="chscale"), CHEAP66)
reg("P67", "p66_ft", "P66 warm start, same model, lr 3e-4: how much of a later gain is "
    "just more optimisation?", BEST | dict(init_from="P66"), FT)
reg("P68", "p66_scatharm16", "P66 warm start with scatter-pattern phi harmonics 8->16 "
    "(new harmonics init 0): is the floor the angular resolution of the scatter element pattern?",
    BEST | dict(init_from="P66", incoh_harm=16), FT)
reg("P69", "p68_scatsplit", "P68 warm start with independent front/back scatter patterns "
    "(copied so step 0 matches, harmonics kept): a smooth pattern cannot cut the horizon",
    BEST | dict(init_from="P68", incoh_harm=16, incoh_split=True), FT)
hyb("H27", "tinyunet_on_P68", "tiny U-Net on the best physics (P68): H24's beam polish on the "
    "fine-tuned floor", "P68", BEST | dict(incoh_harm=16),
    dict(res="unet", unet_ch=[8, 16, 32]), CHEAP66)
hyb("H28", "tinyunet_on_P69", "tiny U-Net on P69, the best physics (fine-tune, 16 scatter "
    "harmonics, front/back patterns)", "P69",
    BEST | dict(incoh_harm=16, incoh_split=True),
    dict(res="unet", unet_ch=[8, 16, 32]), CHEAP66)
hyb("H29", "medunet_on_P69", "medium U-Net (16-32-64) on frozen P69: H27 did not overfit, "
    "so try more residual capacity", "P69",
    BEST | dict(incoh_harm=16, incoh_split=True),
    dict(res="unet", unet_ch=[16, 32, 64]), CHEAP66 | dict(max_min=40, batch=4))

# ---- Round 12: longer FT, more scatter DoF, U-net on new bests ----
# P69 still improved through step 3k. H28 shaved beams
# but not the floor. Goal: see how far val can go tonight without a new mechanism guess.
BEST69 = BEST | dict(incoh_harm=16, incoh_split=True)
LONG_FT = dict(steps=12000, batch=8, lr=2e-4, warmup=0.02, eval_every=200, patience=12,
               max_min=100, min_steps_frac=0.4)
reg("P70", "p69_ft_12k", "P69 warm start, 12k steps @ 2e-4: give the cosine schedule room "
    "after P69 plateaued under a dying LR", BEST69 | dict(init_from="P69"), LONG_FT)
reg("P71", "p69_incoh192", "P69 warm start with 128->192 scatter channels (new chans init "
    "small): is 128 still short of the floor subspace?",
    BEST69 | dict(init_from="P69", incoh=192), LONG_FT | dict(batch=4, max_min=120))
reg("P72", "p69_scatharm24", "P69 warm start, scatter phi harmonics 16->24: finer azimuth "
    "structure in the floor", BEST69 | dict(init_from="P69", incoh_harm=24), LONG_FT)
reg("P73", "p69_scatknots61", "P69 warm start, scatter theta knots 37->61: finer elevation "
    "near the horizon", BEST69 | dict(init_from="P69", incoh_knots=61), LONG_FT)
hyb("H30", "joint_tinyunet_P69", "tiny U-Net with physics unfrozen (phys_lr 0.05x): can "
    "joint polish beat frozen H28?", "P69", BEST69,
    dict(res="unet", unet_ch=[8, 16, 32]),
    CHEAP66 | dict(freeze_phys=False, phys_lr_scale=0.05, lr=5e-4, max_min=50))
hyb("H31", "tinyunet_on_P70", "tiny U-Net on longer-FT physics", "P70", BEST69,
    dict(res="unet", unet_ch=[8, 16, 32]), CHEAP66)
hyb("H32", "tinyunet_on_P71", "tiny U-Net on 192-scatter physics", "P71",
    BEST69 | dict(incoh=192), dict(res="unet", unet_ch=[8, 16, 32]),
    CHEAP66 | dict(batch=4, max_min=50))
hyb("H33", "tinyunet_on_P72", "tiny U-Net on harm-24 scatter physics", "P72",
    BEST69 | dict(incoh_harm=24), dict(res="unet", unet_ch=[8, 16, 32]), CHEAP66)
hyb("H34", "medunet_on_P70", "medium U-Net on P70 (batch 4; H29 OOM'd at batch 8)", "P70",
    BEST69, dict(res="unet", unet_ch=[16, 32, 64]), CHEAP66 | dict(batch=4, max_min=50))
reg("P74", "p70_ft_seed1", "P70 warm start, another 8k @ 1e-4 seed 1: optimisation noise "
    "vs real gain after the long FT", BEST69 | dict(init_from="P70"),
    dict(steps=8000, batch=8, lr=1e-4, warmup=0.02, eval_every=200, patience=10,
         max_min=80, min_steps_frac=0.4, seed=1))
hyb("H35", "tinyunet_on_P74", "tiny U-Net on the deepest fine-tune", "P74", BEST69,
    dict(res="unet", unet_ch=[8, 16, 32]), CHEAP66)

# ---- Round 13: follow the H30 win (joint unfreeze beat frozen U-Net) -----------------
# H30 val 0.477 vs H28/H29 ~0.480: letting physics move under a tiny residual helps.
# TODO: look at P70 results and check if floor is still improving and when, check what
# could be adjusted to make the model more unique or if there needs to be new goals based
# on what Prof. Fayazi says.
JOINT = CHEAP66 | dict(freeze_phys=False, phys_lr_scale=0.05, lr=5e-4, max_min=60,
                       steps=5000, patience=8)
hyb("H36", "joint_long_P70", "longer joint tiny U-Net on P70: H30 recipe on better physics",
    "P70", BEST69, dict(res="unet", unet_ch=[8, 16, 32]), JOINT)
hyb("H37", "joint_med_P70", "joint medium U-Net on P70 (batch 4)", "P70", BEST69,
    dict(res="unet", unet_ch=[16, 32, 64]), JOINT | dict(batch=4, max_min=80))
hyb("H39", "joint_long_P69", "H30 but 6k steps and gentler phys_lr: was joint under-trained?",
    "P69", BEST69, dict(res="unet", unet_ch=[8, 16, 32]),
    JOINT | dict(steps=6000, phys_lr_scale=0.03, lr=4e-4, max_min=70))
reg("P75", "h30phys_ft", "fine-tune the physics H30 jointly polished (residual stripped)",
    BEST69 | dict(init_from="H30"),
    dict(steps=8000, batch=8, lr=1e-4, warmup=0.02, eval_every=200, patience=10,
         max_min=80, min_steps_frac=0.4))
hyb("H38", "joint_on_P75", "joint tiny U-Net on the H30-polished physics", "P75", BEST69,
    dict(res="unet", unet_ch=[8, 16, 32]), JOINT)
hyb("H40", "joint_on_P71", "joint tiny U-Net on 192-scatter physics if P71 moved the floor",
    "P71", BEST69 | dict(incoh=192), dict(res="unet", unet_ch=[8, 16, 32]),
    JOINT | dict(batch=4, max_min=80))
