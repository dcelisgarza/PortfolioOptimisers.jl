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
  identical to baseline. **NEXT: Phase 2.**

## Phase 2 — thread `prefix` through the risk-build spine (NEXT)

Add `prefix::Symbol = Symbol("")` to `set_risk_constraints!` and every per-measure
builder it reaches, and USE it for:

1. **Reads of infra keys** — replace bare reads/`haskey` with prefixed:
   `model[:W]`→`model[Symbol(prefix,:W)]` (or `set_sdp_constraints!(model; prefix)`),
   same for `:dd`, `:X`, `:net_X`, `:Xap1`, `:ddap1`. Use `get_w(model, prefix)`.
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
3. `:fees` stays BARE everywhere (shared, never recreated by a nested build).
   `:G` (chol cache), `:frc_W`/`:frc_M`/`:frc_M_PSD` (FRC) stay BARE — not in the
   swap groups.

Spine files (per-measure `set_risk_constraints!`/`set_risk!`): 01 (dispatch),
02–20 under `19_RiskMeasureConstraints/`, plus readers in `09_JuMPConstraints/`.
Do it per-subsystem, default-empty (behaviour-preserving), test each.

## Phase 3 — flip risk tracking + delete swap (PAYOFF)

In `18_TrackingRiskMeasureConstraints.jl`: the Independent/Dependent
`set_risk_constraints!` methods currently swap `:w`↔`:oldw` and call
`set_risk_tracking_risk_constraints!`, which hand-rolls the 390-line save→build→
restore (lines ~358–745). Replace with: store the tracking-difference weights at
`Symbol(prefix, :w)` (prefix e.g. `Symbol(:tr_iv_, i, :_)`), call the inner risk
build with that `prefix`, and **delete** `set_risk_tracking_risk_constraints!`'s
save/restore body + the `:w`/`:oldw` swap. Same prefix flows to
`set_risk_tr_constraints!` → inner `set_risk_constraints!(; prefix)`.
Also drop the dead commented `set_trdv_risk_constraints!` block (~800–947).
The 17 snapshot groups (the keys to prefix) are exactly:

    (:variance_flag,), (:rc_variance,), (:W,:M,:M_PSD),
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

Seam-lock test: fail CI on raw `model[:` outside `08_Base_JuMPOptimisation.jl`.
