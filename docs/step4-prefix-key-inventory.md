# Step 4 (prefix approach) — namespaced model-state key inventory

Working note (temporary; delete when Step 4 lands). These are the **singleton
infrastructure** keys that currently collide between the outer model and a nested
risk build, so each must be constructed as `Symbol(prefix, basename)` with
`prefix = Symbol("")` (bare) by default. Per-`i` scratch keys already namespace via
the compound index the tracking build passes and are NOT listed here.

Groups (first key = existence sentinel / `haskey` guard; rest are companions made
and removed together):

- `(:variance_flag,)`
- `(:rc_variance,)`
- `(:W, :M, :M_PSD)`                         — SDP matrix (set_sdp_constraints!)
- `(:Au, :Al, :cbucs_variance)`              — box UCS variance
- `(:E, :WpE, :ceucs_variance)`              — ellipsoidal UCS variance
- `(:X,)`                                    — portfolio returns
- `(:net_X,)`                                — net portfolio returns (reads :fees)
- `(:Xap1,)`                                 — returns + 1
- `(:ddap1,)`                                — drawdowns + 1
- `(:wr_risk, :cwr)`                         — worst realisation
- `(:range_risk, :br_risk, :cbr)`            — range
- `(:dd, :cdd_start, :cdd_geq_0, :cdd)`      — drawdown variable + constraints
- `(:mdd_risk, :cmdd_risk)`                  — maximum drawdown
- `(:uci, :uci_risk, :cuci_soc)`             — ulcer index
- `(:owa, :owac)`                            — ordered weights array
- `(:bdvariance_risk, :Dt, :Dx)`             — brownian distance variance
- `(:W1_vr_sk_kt, :W2_vr_sk_kt, :W3_vr_sk_kt, :L2W1_vr_sk_kt, :M_vr_sk_kt, :M_vr_sk_kt_PSD)` — variance/skew/kurtosis

Plus `:w` itself: thread via prefix; the tracking build stores the difference
`w - wb*k` at `Symbol(prefix, :w)` and the inner build reads `get_w(model, prefix)`.

Also `:fees` is read by `set_net_portfolio_returns!`; check whether it needs
prefixing for the tracking-difference net returns (likely yes if fees apply to the
tracked difference, else the bare fees are reused — to be confirmed).
