---
status: accepted
---

# The frontier sweep is one seam

## Context

A frontier sweep registers one JuMP parameter and one bound constraint per swept entry,
iterates the product of the sweep points, and collects `(retcode, sol)` per point. Two heads
run it: `MeanRisk` and `NearOptimalCentering`. Each head sweeps two axes — the return frontier
(`:ret_frontier`) and the risk frontier (`:risk_frontier`) — in three combinations, so the
sweep was written across nine `solve_*!` methods in two files.

Only the return side had an interface. `set_ret_frontier_parameters!` was shared by all four
return-side sites. The risk side had none: its parameter loop was written out four times, and
the collect tail seven times.

Two of the four risk-side copies had drifted. `MeanRisk` homogenises the bound with the
head's `k` — `d * sc * (r_expr - ub * k) <= 0`, matching the scalar bound in
`set_risk_upper_bound!` — and both `NearOptimalCentering` copies omitted the factor. The
divergence is latent, not a wrong answer: `NearOptimalCentering` minimises a barrier, so its
head registers `k` as the literal `1` and the factor is a no-op there. It is drift all the
same — inside one `NearOptimalCentering` model the return-frontier bound called the shared
helper and carried `k`, and the risk-frontier bound, written by hand, did not.

The `NearOptimalCentering` both-axes method carried a real defect. Its anchors — `rk_opts` and
`rt_opts`, the risk and return of the `MeanRisk` solution that the barrier centres on — are one
per sweep point, and the method paired them with the sweep by `zip`ping them against the risk
axis *inside* the return loop. `zip` restarts, so every return point re-used the anchors of the
first return point. The single-axis methods paired anchor `i` with point `i` correctly, so the
pairing rule was right everywhere it was written once and wrong where it was written twice.

`test_28_seam_lock.jl` recorded the absence. Rule 3 of ADR 0037 says no file outside the Model
State interface reaches for a key, and `registry_readers` exempted `11_MeanRisk.jl` and
`13_NearOptimalCentering.jl` by name. They were exempt because the sweep had no interface to
reach through: the frontier registries hand out `(bound_var_key, bound_key)` pairs, and the
sweep loops created the parameter and the constraint under exactly those keys.

## Decision

The sweep is one seam, and it lives in the Model State interface,
`20_Optimisation/08_Base_JuMPOptimisation.jl`, beside the two frontier readers that were
already there (`frontier_point_count`, `frontier_sweep_points`).

Register:

- `set_ret_frontier_parameters!(model, ret_frontier)` — moved, unchanged.
- `set_risk_frontier_parameters!(model, risk_frontier)` — new. The polarity `d` and the
    homogenisation `k` are stated once, so the risk bound reads the same in both heads.

Product:

- `frontier_axis(frontier)` — one registry's product, as `(keys, points)`.
- `frontier_sweep_axes(ret_axis, risk_axis)` — the two axes joined into the flat sequence of
    sweep points. Either may be `nothing`. **The risk axis varies fastest**, so the flat order
    is return-outer and risk-inner.

Collect:

- `set_frontier_point!(model, point)` — write one point's bounds.
- `frontier_sweep!(point!, model, opt, T, points)` — solve one model per point and collect.
    `point!` is the per-optimiser hook, called with the **flat** index. `MeanRisk` passes none.
    `NearOptimalCentering` passes `set_noc_anchor!`, which moves `noc_rk` and `noc_rt` onto that
    point's anchor.

The flat index is the contract that fixes the both-axes defect. `near_optimal_centering_setup`
solves the anchors as one `MeanRisk` sweep over the same two frontiers, and that sweep emits
its solutions in the same flat order, so anchor `i` belongs to sweep point `i` in every
combination.

`rebuild_risk_frontier`'s scalar-measure methods returned a one-element `Tuple` and its
vector-measure methods a `Vector`. Both now return a `Vector`, so a resolved registry has one
shape.

`registry_readers` is empty. No file outside the interface reaches for a Model State key.

## Consequences

`NearOptimalCentering`'s risk-frontier constraint now carries `* k`. Its head registers
`k = 1`, so the assembled constraint is unchanged; the factor becomes live only if a
`NearOptimalCentering` formulation ever takes a ratio objective, which is the case the drift
would have made wrong.

The both-axes `NearOptimalCentering` sweep centres each point on its own anchor, so weights on
that path change and one stored asset moves.
`test/assets/NearOptimalCenteringParetoSurfaceRetRk.csv.gz` is that path — a 3-point return
frontier against a 3-point risk frontier, nine sweep points — and it is regenerated. Its nine
anchors are `[a,a,a,a,a,a,b,b,b]`: the return bound binds only at its largest value, so the old
pairing was observably wrong on the last three points alone. Columns 1–7 move by less than
`7e-5`, and columns 8 and 9 by `2.0e-2` and `3.0e-2`. The other Pareto-surface assets sweep two
risk measures, which is a product *within* the risk axis and was already correct, so they do
not move.

A sweep point now writes both axes' parameters rather than writing the outer axis once per
outer iteration. The values are idempotent and a parameter write does not rebuild a
constraint, so this costs a few parameter writes and no solve.

ADR 0037's rule 3 keeps its polarity and loses its only exemption. ADR 0008 (§3: the head and
the tail stay per-optimiser) is untouched — the sweep sits inside the tail. Nothing in ADR
0004 or 0005 is contradicted.
