# Step 4 (prefix approach) — RESUME GUIDE

Working note (temporary; delete when Step 4 lands). Replaces the risk-tracking
snapshot/restore (build-then-swap) with keys built under a namespacing `prefix`
from the start, per ADR 0004. Decided over the registry/build-then-swap approach
(reverted) because it removes the unregister/restore dance and is re-entrant.

## Status

- **Steps 1–3 (committed):** typed read accessors `get_constraint_scale`,
  `get_objective_scale`, `get_w`, `get_k`, `get_ret`, `get_risk` in
  `08_Base_JuMPOptimisation.jl`; ~260 raw `model[:key]` reads swept. Full suite
  green (4075/4075) at commit `abab096cb`.
- **Phase 1 (committed `d83fbc902`):** `preg!(model, prefix, name, val)` helper +
  `prefix::Symbol = Symbol("")` threaded through the shared infra builders
  (`set_portfolio_returns!` :X, `set_net_portfolio_returns!` :net_X,
  `set_portfolio_returns_plus_one!` :Xap1, `set_portfolio_drawdowns_plus_one!`
  :ddap1, `set_drawdown_constraints!` :dd/:cdd*, `set_sdp_constraints!`
  :W/:M/:M_PSD) and prefixed `get_w`. Default empty = bare keys; behaviour
  identical to baseline.
- **Phase 2 slice 1 (committed `fa6c0f458`):** prefix threaded through the
  drawdown/realisation/range builders — files 10 (OWA), 11 (AverageDrawdown),
  12 (UlcerIndex), 13 (MaximumDrawdown), 14 (BrownianDistanceVariance),
  15 (WorstRealisation), 16 (Range). `check_all()` byte-identical to baseline
  (11/12 + sum 1.0, `trk_var_d`=(false,NaN)). **NEXT: Phase 2 slice 2.**

## Phase 2 slice plan (decided 2026-06-08)

The remaining per-measure files split by difficulty, NOT file order:

- **Easy (per-`i` Category-B keys + infra reads only):** 03 (Moment), 06 (VaR/DaR),
  07 (CVaR/CDaR/DRCVaR), 08 (EVaR/EDaR), 09 (RLVaR/RLDaR), and 05
  (NegativeSkewness, modulo its `:GV` cache). Mechanical: thread `prefix` to the
  infra reads (`set_net_portfolio_returns!`, `set_portfolio_returns_plus_one!`,
  `get_k`). Do as **slice 2**, commit, `check_all()`.
- **Hard (shared/weight-dependent singletons + SDP):** **02 (Variance)**,
  **04 (Kurtosis)**, **20 (VarianceSkewKurtosis)**. Do as **slice 3** (final),
  carefully reviewed in isolation.

**The bare-vs-prefix invariant (decided 2026-06-08):** *prefix a key iff it is
weight-dependent; prior-derived caches stay bare.* Tracking shifts the WEIGHTS
(via the benchmark difference), not the prior moments, so any key that is a pure
function of `pr` is identical in the inner build and is correctly shared bare.
This explains the old swap list post-hoc — and exposes `:L2W` as the one
weight-dependent key it WRONGLY omitted (see slice-3 note below).

## Phase 2 — thread `prefix` through the risk-build spine (NEXT)

Add `prefix::Symbol = Symbol("")` to `set_risk_constraints!` and every per-measure
builder it reaches, and USE it for:

0. **Full Category-A accessor coverage (decided 2026-06-08).** The Step-5 seam-lock
   is **literal `model[:` only** (see Step 5), so prefix-threaded keys become
   computed `model[Symbol(prefix,name)]` and are naturally exempt. Within that:
   - **Cross-file shared infra** (`:X`, `:net_X`, `:Xap1`, `:ddap1`, `:dd`, SDP
     `:W`/`:M`) get **named prefixed accessors** `get_X(model,prefix)` /
     `has_X(model,prefix)` etc. in `08_Base_JuMPOptimisation.jl`, alongside the
     existing `get_w`. Consumers route through these, not raw indexing.
   - **Per-measure singletons** (`:wr_risk`, `:variance_flag`, `:mdd_risk`, `:owa`,
     `:uci`, `:bdvariance_risk`, skew/kurtosis `W*`, …) are read only inside their
     own builder; they stay on the prefix-computed `model[Symbol(prefix,name)]`
     form (lock-exempt, self-documenting in-file). No ceremonial one-caller accessor.
1. **Reads of infra keys** — replace bare reads/`haskey` with the named prefixed
   accessors from (0): `get_X(model,prefix)`/`has_X(model,prefix)`, same for `:dd`,
   `:net_X`, `:Xap1`, `:ddap1`; SDP via `set_sdp_constraints!(model; prefix)`. Use
   `get_w(model, prefix)`.
2. **Risk-specific singleton creations** — register via `preg!(model, prefix, …)`
   and guard via `haskey(model, Symbol(prefix, …))`:
   - `:variance_flag`, `:rc_variance`            → 02_VarianceConstraints
   - `:Au/:Al/:cbucs_variance`, `:E/:WpE/:ceucs_variance` (UCS) → 02
   - `:wr_risk/:cwr`                              → 15_WorstRealisation
   - `:range_risk/:br_risk/:cbr`                 → 16_Range
   - `:mdd_risk/:cmdd_risk`                       → 13_MaximumDrawdown
   - `:uci/:uci_risk/:cuci_soc`                  → 12_UlcerIndex
   - `:owa/:owac`                                 → 10_OWA
   - `:bdvariance_risk/:Dt/:Dx`                  → 14_BrownianDistanceVariance
   - `:W1_vr_sk_kt/:W2_vr_sk_kt/:W3_vr_sk_kt/:L2W1_vr_sk_kt/:M_vr_sk_kt/:M_vr_sk_kt_PSD` → 20_VarianceSkewKurtosis
   - `:W/:M/:M_PSD` (via `set_sdp_constraints!(model; prefix)`, shared with 02) AND
     **`:L2W`** → 04_Kurtosis. `:L2W = L2·vec(W)` is weight-dependent (a function of
     the prefixed SDP `W`), so it MUST be prefixed and added to group 3 — the old
     swap list OMITTED it (correctness gap; see ADR 0005). Prefixing is byte-identical
     on the single-measure golden tests AND closes the latent stale-`:L2W`
     kurtosis-in-kurtosis re-entrancy bug.
3. `:fees` stays BARE everywhere (shared, never recreated by a nested build).
   `:frc_W`/`:frc_M`/`:frc_M_PSD` (FRC) stay BARE — not in the swap groups.
   **Prior-derived caches stay BARE** (weight-independent → identical in inner
   builds, correctly shared): `:G` (chol), `:Gkt` (04), `:GV` (05),
   `:vals_Akt`/`:vecs_Akt` (04). Invariant: prefix iff weight-dependent.

Spine files (per-measure `set_risk_constraints!`/`set_risk!`): 01 (dispatch),
02–20 under `19_RiskMeasureConstraints/`, plus readers in `09_JuMPConstraints/`.
Do it per-subsystem, default-empty (behaviour-preserving), test each.

## Phase 3 — flip risk tracking + delete swap (PAYOFF)

In `18_TrackingRiskMeasureConstraints.jl`: the Independent/Dependent
`set_risk_constraints!` methods currently swap `:w`↔`:oldw` and call
`set_risk_tracking_risk_constraints!`, which hand-rolls the 390-line save→build→
restore (lines ~358–745). Replace with: store the tracking-difference weights at
`Symbol(prefix, :w)` and **compose** the new tracking prefix onto the incoming one —
`Symbol(prefix, :tr_iv_, i, :_)`, NOT a fresh `Symbol(:tr_iv_, i, :_)` — so
tracking-nested-in-tracking is collision-free (decided 2026-06-08; the inner
risk-vector `enumerate` restarts `i` at 1, and `RiskTrackingRiskMeasure`'s inner
`r::AbstractBaseRiskMeasure` permits nesting, so a replace-prefix would alias). This
is what makes ADR 0005's re-entrancy claim true. Call the inner risk build with that
composed `prefix`, and **delete** `set_risk_tracking_risk_constraints!`'s
save/restore body + the `:w`/`:oldw` swap. Same prefix flows to
`set_risk_tr_constraints!` → inner `set_risk_constraints!(; prefix)`.
Also drop the dead commented `set_trdv_risk_constraints!` block (~800–947).
The old swap had 17 snapshot groups; the keys to prefix are those PLUS `:L2W`
(group 3 below — the swap list omitted it, see Phase 2 slice plan):

    (:variance_flag,), (:rc_variance,), (:W,:M,:M_PSD,:L2W),
    (:Au,:Al,:cbucs_variance), (:E,:WpE,:ceucs_variance),
    (:X,), (:net_X,), (:Xap1,), (:ddap1,), (:wr_risk,:cwr),
    (:range_risk,:br_risk,:cbr), (:dd,:cdd_start,:cdd_geq_0,:cdd),
    (:mdd_risk,:cmdd_risk), (:uci,:uci_risk,:cuci_soc), (:owa,:owac),
    (:bdvariance_risk,:Dt,:Dx),
    (:W1_vr_sk_kt,:W2_vr_sk_kt,:W3_vr_sk_kt,:L2W1_vr_sk_kt,:M_vr_sk_kt,:M_vr_sk_kt_PSD)

## How to verify (kaimon)

`check_all()` harness (define in session): MeanRisk var/cvar/cdar, RiskBudgeting,
and RiskTracking {var,sd,cdar,owa}×{Independent,Dependent}. Baseline = 11/12
success+sum 1.0, with `trk_var_d` = (false, NaN) (a pre-existing QuadExpr-under-
dependent-tracking limitation, NOT a regression). Compare with NaN-aware:
`samev(a,b) = a[1]==b[1] && (a[2]===b[2] || isapprox(a[2],b[2]))`.
Gate before each commit milestone: full single-core suite
`Pkg.test("PortfolioOptimisers"; test_args=["--jobs=1"])` (~70 min, expect
4075/4075). Pre-commit Julia formatter reformats new code and aborts the FIRST
commit attempt; just `git add` + re-commit.

## Step 5 (after Step 4)

Seam-lock test: fail CI on raw **literal** `model[:` outside
`08_Base_JuMPOptimisation.jl` (i.e. the regex `model\[:`, NOT all `model[`). Scope
decided 2026-06-08: a literal-key lock catches every Category-A leak once Cat-A has
full accessor coverage (Phase 2 §0), while Category-B's computed
`model[Symbol(:variance_risk_, i)]` is naturally exempt — so we honour ADR 0004 §1
(Cat-B stays raw) without an allowlist. Residual hole (accepted, per ADR 0004 §2):
a sloppy raw `model[Symbol(prefix, :newCatA)]` looks like Cat-B and slips through.

## Note on verification (prefixed path)

The full suite golden-tests every measure in `rs` under BOTH tracking modes
(`mr_block2`=Dependent / `mr_block3`=Independent in `test/test18_setup.jl`, vs
`MeanRiskDT.csv.gz`/`MeanRiskIT.csv.gz`; QuadExpr-under-Dependent expected
`OptimisationFailure`). So the prefixed path is well covered — but only at **Phase 3**
(when tracking flips to prefixes). During Phase 2 the prefix plumbing is dormant
(always default-empty), so `check_all()` should be byte-identical to baseline; a
Phase-2 threading bug only surfaces once Phase 3 activates the path.
