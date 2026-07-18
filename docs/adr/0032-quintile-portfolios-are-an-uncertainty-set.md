---
status: accepted
---

# Quintile portfolios are an uncertainty set, not an optimiser

## Context

Zhou & Palomar, *Understanding the Quintile Portfolio* (IEEE TSP 68, 2020,
[10.1109/TSP.2020.3006761](https://doi.org/10.1109/TSP.2020.3006761)) reinterprets two
practitioner heuristics — the 1/N portfolio and the quintile portfolio (equally long the top
20% of assets by some characteristic, optionally short the bottom 20%) — as *exact solutions
of a robust optimisation problem*. Their thesis: rank-and-hold heuristics are not
atheoretical folk wisdom; they are what worst-case optimisation over an ℓ1 ball around the
estimated characteristic vector actually returns, with the ball's radius `eps` deciding how
many assets you hold. A tiny radius holds one asset, a moderate one holds a quintile, a large
one holds everything equally.

The paper presents seven models: long-only (its eq. 9), dollar-neutral long-short (14),
inverse-volatility long-only (21) and long-short (25), market-exposure (30), market-neutral
(33), and sector-neutral (35). The obvious reading is "add a `QuintilePortfolio` optimiser".

The obvious reading is wrong. Every one of the seven is

```
maximize   mu' w - eps * ||sd .* w||_inf
subject to <linear constraints>
```

and the library already owned all of it except the first term's shape.
`set_ucs_return_constraints!` already dispatched the worst-case return on uncertainty-set
*shape* — `BoxUncertaintySet` yields `mu'w - d_mu'|w|`, `EllipsoidalUncertaintySet` yields
`mu'w - k*||Gw||_2`. The ℓ1 ball yields `mu'w - eps*||sd .* w||_inf`. That is one missing
method on an existing seam, not a new optimiser.

Two things blocked the recipe, and one model needed machinery that did not exist.

## Decision

**Quintile portfolios are an uncertainty-set shape.** There is deliberately no
`QuintilePortfolio` type. The recipe is:

```julia
MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
         opt = JuMPOptimiser(; slv = slv, bgt = 1.0,
                             ret = ArithmeticReturn(; ucs = CharacteristicUncertaintySet())))
```

Consequences of that choice, in order:

### Two shapes, not one

`L1UncertaintySet` (radius `eps`, optional scaling `sd`) covers both the paper's `S` (eq. 5)
and `A1` (eq. 18) — `S` is `A1` with `sd = 1`, which is algebra, not a judgement call.

`SignedL1UncertaintySet` (radii `ep`, `en`) covers `A2` (eq. 25) and does *not* collapse into
the first: the joint set shares one budget across both signs, giving `max(t+, t-)`; the signed
set spends a budget per sign, giving `ep*t+ + en*t-`. They agree only when `w` is
single-signed. The paper never writes `A2`'s worst case for general `w` because it
immediately *decouples* into its eqs. (27)/(28), and its Remark 12 then admits the
recombination is optimal only when the two legs happen to have complementary support.
Modelling the worst case directly —

```
min_{mu in A2} mu'w = mu'w - ep*[max_i(-sd_i w_i)]_+ - en*[max_i(sd_i w_i)]_+
```

— keeps the problem coupled and dissolves that caveat. Verified: the coupled solve returns a
3-long/1-short portfolio with disjoint support, an asymmetry the decoupled form cannot
produce.

### `NoRisk`

`MeanRisk` requires a risk measure, so `obj = MaximumReturn()` would still build the default
`Variance` term. The objective discards it, but it drags second-order cone constraints into a
model that is a linear program, forcing a conic solver on an LP. `NoRisk` contributes a zero
risk expression. This is not quintile-specific: the global maximum return portfolio (GMRP,
Table I of the paper) had the same defect.

`NoRisk` is only coherent where the objective never consults risk, so it is rejected at
construction under `MinimumRisk` (objective identically zero — *any* feasible portfolio
optimal, silently) and `MaximumRatio` (risk-normalisation vacuous, model unbounded), and in
every optimiser whose formulation *is* its risk measure (`RiskBudgeting`,
`NearOptimalCentering`, `FactorRiskContribution`, `HierarchicalRiskParity`,
`HierarchicalEqualRiskContribution`). ADR 0018's trait cannot express this — it is derived
from family supertypes precisely so it cannot carry per-measure exceptions, and `MeanRisk` and
`RiskBudgeting` are indistinguishable to it — so the check is a dedicated assertion
(`assert_risk_measure_required`), reached through the schedule test-substitution pass for
`TimeDependent` fields.

### `gbgt`: net and gross were pinned only together

`bgt = b, sbgt = s` gives `1'w == b` *and* `norm(w, 1) == b + 2s`. The parameterisation
expresses any **(net, gross) pair**, but only ever *both at once*. Model 6 (eq. 33) needs
`norm(w, 1) == 1` with the net **free** — the one combination unreachable — and it cannot be
approximated away, because without the gross constraint the objective is positively
homogeneous of degree 1 over a cone and the problem is unbounded.

`gbgt` constrains `sum(lw) + sum(sw)` on its own. A number pins it; a `BudgetRange` bounds it,
which incidentally gives the library the leverage cap it lacked. It is rejected when `bgt` and
`sbgt` are both pinned numbers (they already determine the gross exposure) and when the weight
bounds forbid shorts (gross and net coincide; `bgt` already owns it).

### `xbgt`: the budgets bound, they do not pin

`lw` and `sw` are **upper bounds** on the parts of `w` (`lw >= w`, `lw >= 0`), never equal to
them. The existing MIP complementarity (`lw <= ss*ib`) forces the sign pattern — long xor short
— but does *not* close that slack. Verified: identical results with and without the binaries.
So `sbgt = 0.3` means *at most* 30% short. The slack is usually harmless because the objective
pushes against the budget, but it is not the same problem.

Closing it needs a per-asset *sign* bit. `short_mip_threshold_constraints` already has one —
`ilb`/`isb`, the long and short indicators it builds for thresholds and fixed fees — and its
`miprb_flag` block already emits half of what is needed:

```
lw - ss * il <= 0      # complementarity: short => lw == 0
sw - ss * is <= 0      # complementarity: long  => sw == 0
```

So `xbgt` is not a new builder. It rides that one, emitting the complementarity pair
regardless of `miprb_flag` and adding only the two constraints that close the slack:

```
lw - w - ss * (1 - ilb) <= 0   # ilb = 1 => lw <= w,  with lw >= w  => lw == max(w, 0)
sw + w - ss * (1 - isb) <= 0   # isb = 1 => sw <= -w, with sw >= -w => sw == max(-w, 0)
```

with `ilb = isb = 0` giving `lw == sw == 0`, hence `w == 0`. It is keyed on the binaries
rather than on `il`/`is`, which relax to continuous variables when `k` is free. `xbgt` is
opt-in and defaults to `false` because it converts a linear program into a genuine
mixed-integer program — 1m33s for four 20-asset instances against milliseconds for the LP.

Three consequences follow from reusing rather than adding.

**`xbgt` forces the long-short builder.** The long-only `mip_constraints` cannot serve it: its
`ib` marks an asset as *held* and gates both bounds (`lb ⊙ ib <= w <= ub ⊙ ib`), carrying no
sign, and it is what `card` counts. A sign bit and a held bit are different bits; conflating
them would silently redefine `card` to count long positions only and leave shorts uncounted.
So `set_mip_constraints!` routes `xbgt` to `short_mip_threshold_constraints`, where
`i_mip = ilb + isb` remains the held indicator cardinality counts and `ilb` carries the sign.
`xbgt` is ignored outright when the model has no short side, since there is then nothing to
pin.

**A sign bit is total, so `xbgt` alone gets its own lean branch.** The long-short builder
splits each asset three ways — long, short, inactive — and the third state exists only for
consumers of the held indicator: a threshold telling `w = 0` from `w >= lt`, or `card`
counting holdings. Pinning needs neither, because `w = 0` satisfies both `w >= 0` and
`w <= 0`, so either branch of the sign split absorbs it. When no such consumer is present,
`sign_mip_constraints` therefore does the job with one bit per asset instead of two. This is
not a second formulation of the same thing: it is the same four constraints with the
redundant state dropped, taken only when the condition guaranteeing the redundancy holds.
It matters because the models that use `xbgt` most — the paper's market-exposure,
market-neutral and sector-neutral portfolios — carry no cardinality or threshold, and would
otherwise pay `2N` binaries for a state they never read.

**This moved `xbgt` out of `set_weight_constraints!`.** The budgets are still built there, but
the binaries that pin them belong to the MIP builders, which run later
(`set_weight_constraints!` → `assemble_jump_model!` → `set_mip_constraints!`). An earlier
draft kept a bespoke `xbgt_ib` in the budget builder and was wrong twice over: it reached for
`mip_constraints`' `ib` if present (wrong semantics), and it registered its own under the same
`:ib` key, which `mip_constraints` then silently clobbered — leaving the constraints pointing
at orphaned variables and the model carrying `2N` binaries where `N` were live.

Measured on the 20-asset test universe, the result is that `xbgt` costs `N` binaries alone and
**nothing at all** on top of any other MIP feature:

| configuration                  | binaries    |
| :----------------------------- | ----------: |
| `xbgt` alone                   | `N`         |
| `card` alone / `card` + `xbgt` | `N` / `2N`  |
| `st` alone / `st` + `xbgt`     | `2N` / `2N` |

`2N` for `card` + `xbgt` is irreducible — a held bit and a sign bit are independent — and the
`i_mip_ub` cut linking `ilb + isb <= 1` strengthens the formulation the solver sees there.

It matters only past `eps > f(floor(N/2))`, where the paper holds a fixed 50/50 corner at a
**negative** worst-case objective while the relaxation prefers cash. Across the entire radius
range the paper's Fig. 2 illustrates, the LP reproduces the closed forms exactly — objective
and active count, to the digit. The regime is also unreachable through
`ActiveAssetsUncertaintyAlgorithm` (which calibrates to at most `floor(N/2)` pairs by
construction), absent from every long-only model (`bgt = 1` makes `w = 0` infeasible), and
loud when hit: `optimise_JuMP_model!` already rejects an all-zero weight vector as a
structured `OptimisationFailure` (ADR 0023).

### `ActiveAssetsUncertaintyAlgorithm`: `eps` needs a unit conversion

`eps` is dimensionally a sum of characteristic differences, so it has no memorable scale: on
daily S&P 500 returns the *entire* meaningful range is `[0, 0.017]` and the quintile sits at
`0.0021`. Any human-looking round number — `0.05`, `0.1` — is far past the divergence
threshold, and the scale shifts by ~250x on annualised returns.

The paper's corollaries invert cleanly, so the algorithm converts the quantity a caller can
actually reason about ("hold 20% of the universe") into the radius that produces it. It
mirrors `AbstractUncertaintyKAlgorithm`, and following `EllipsoidalUncertaintySetAlgorithm`'s
`method::Num_UcSK` precedent a bare number *is* the explicit case, so no separate "explicit"
algorithm is needed.

It is named as a **radius calibration, not a promise**. The inversion is exact only for the
bare problem the closed forms assume; add weight bounds, cardinality or sector constraints —
which this whole decision exists to allow — and the realised count drifts. Validating
post-solve was rejected (the set is long gone by solve time, and warnings in a fold loop are
noise); `card` remains the honest tool for a hard cardinality constraint.

### Mean-only

The ℓ1 ball bounds a *characteristic* vector and has no covariance analogue, breaking the
family invariant that every uncertainty-set estimator implements `ucs`, `mu_ucs` and
`sigma_ucs`. `CharacteristicUncertaintySet` implements `mu_ucs` only; the other two exist
solely to throw an informative error pointing at `NormalUncertaintySet`. A trait was rejected
as a mechanism with exactly one implementor.

## Consequences

Six of the seven models fall out of constraints that already existed; only model 6 needed new
machinery, and that machinery (`gbgt`) is a general leverage feature rather than a quintile
one. The paper's thesis — that the quintile portfolio *is* robust optimisation — is now true
in the type system rather than restated in prose.

The cost is discoverability: nobody searching "quintile" finds
`ArithmeticReturn(; ucs = CharacteristicUncertaintySet())`. This was accepted knowingly. The
word appears in the docstrings, the example, this ADR, and the bibliography, and a
`QuintilePortfolio` convenience type would have to re-encode the budget, bound and exposure
constraints that `JuMPOptimiser` already owns — reintroducing exactly the duplication this
decision removes.

`gbgt` and `xbgt` are the largest surface change and are not quintile-scoped: they alter
budget semantics library-wide and carry `TimeDependent`, factory and view plumbing. `xbgt`
in particular documents a pre-existing property most callers will not have known —
that a short budget is a bound, not an equality.
