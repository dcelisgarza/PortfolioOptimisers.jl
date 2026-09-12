# 627 — The worst-case measure a distributionally robust measure prices, read off the solved model

Research ticket #627 of wayfinder map #304. Written 2026-09-03.

Sources are the constraint body in
`src/20_Optimisation/20_RiskMeasureConstraints/07_ConditionalXatRiskConstraints.jl`
(`set_dr_conditional_risk_constraints!`, at the tip of `dev`, `e6c7840bc2`), Mohajerin
Esfahani and Kuhn (2018), Mathematical Programming 171, cited as `drcvar` in
`docs/src/References.bib`, the JuMP and MathOptInterface documentation on conic duality, and
a bare Julia process that solved both twins and read the duals back. The script is
`wc_measure.jl`, reproduced at the end of this note. Every number in section 6 comes from
that process.

---

## Summary

- **The worst-case measure is recoverable, and nothing has to be added to the builder.** It
  is *not* read from the primal entries `u`, `v`, `s`, `tau` and `lb`. It is read from the
  **duals of the four constraints** the builder already registers: `cu`, `cv`, `cu_infnorm`
  and `cv_infnorm`. `JuMP.dual` on those entries gives the masses and the transport
  displacements, and the atoms follow by one division.
- **The map is a closed form, one line per atom.** For observation `i` and piece `k`, with
  `mu_ik` the negated dual of the epigraph row and `q_ik` the tail of the dual of the
  infinity-norm cone: mass `mu_ik / sum(mu)`, atom `xhat_i + q_ik / mu_ik`. At most `2T`
  atoms. The derivation is section 3, and it is the finite-dimensional dual of Esfahani and
  Kuhn's reformulation, which is their Theorem 4.4 specialised to the library's support.
- **The ticket's premise was half wrong.** The registered variables are the *dual* side of
  the worst-case problem, so they name which support coordinates bind and which piece is
  active, but not the split of mass between the two pieces and not the displacement. The
  multipliers carry those, and the multipliers live on the constraints, not on the variables.
- **All three tests the ticket asks for pass in a bare process.** The risk recomputed from
  the recovered measure equals the reported risk to `2e-9` in five solves, a radius of
  `1e-9` (the constructor refuses zero) recovers the sample to `2e-8`, and every atom
  satisfies the support `xi >= -1`, including the case where the floor binds and the
  prototype's formula names a return of `-3.8`. Section 6 has the numbers.
- **The drawdown twin recovers mechanically, and its atoms are per-asset drawdown
  scenarios, not return paths.** Its risk identity holds because the programme is an LP dual,
  but the linear term the programme prices is a *linearised* portfolio drawdown. Section 5
  states what the atoms mean, and what they do not.
- **The verb is a post-solve reader over a `JuMPOptimisationResult`**, and it needs a solver
  that returns duals. Section 4 gives a recommendation for the name, the home and the return
  shape. It is a recommendation, not a decision; the decision ticket follows.

---

## 1. The programme the library solves

`set_dr_conditional_risk_constraints!` reads `alpha`, `b1 = r.l`, `radius = r.r`, sets

```
a1 = -1          b1 = l
a2 = -1 - l/α    b2 = l (1 - 1/α)
```

and writes, with `sc` the constraint scale, `series` the per-observation return series, and
`ambiguity = xhat .+ 1`:

```
cu[i]:          b1 τ + a1 series_i + Σ_j u_ij (1 + xhat_ij) - s_i <= 0
cv[i]:          b2 τ + a2 series_i + Σ_j v_ij (1 + xhat_ij) - s_i <= 0
cu_infnorm[i]:  ( tu_i, -u_i - a1 w ) ∈ NormInfinityCone(1 + N)
cv_infnorm[i]:  ( tv_i, -v_i - a2 w ) ∈ NormInfinityCone(1 + N)
cu_lb, cv_lb:   tu_i <= lb,  tv_i <= lb
u, v >= 0
risk = radius * lb + mean(s)        (or the observation-weighted mean)
```

Read against Esfahani and Kuhn's Theorem 4.2 for a piecewise-affine loss `max_k a_k <w, ξ> +
b_k τ` on a polyhedral support `{ξ : C ξ <= d}`, this is their reformulation with

- `C = -I`, `d = 1`, so the support is `ξ >= -1` and `<γ_ik, d - C xhat_i> = <γ_ik, 1 + xhat_i>`,
  which is the `u_cost` and `v_cost` term;
- `γ_i1 = u_i`, `γ_i2 = v_i`, `λ = lb`;
- the dual norm in `||C' γ_ik - a_k w||_* <= λ` is the **infinity norm**, so the ground metric
  of the ball is the **1-norm**;
- the loss `max_k` of the two pieces is `-w'ξ + l [τ + (1/α)(-w'ξ - τ)_+]`, which is the
  paper's section 7.1 mean-risk objective `E[-w'ξ] + l CVaR_α(-w'ξ)` with `ρ = l`.

The library's docstring, rewritten by #313, already states this. What follows is the other
side of the same duality.

## 2. What the registered variables carry, and what they do not

Esfahani and Kuhn's reformulation is the **dual** of the worst-case expectation
`sup_{Q ∈ B_ε(P̂)} E_Q[loss]`. The variables `lb`, `s`, `u`, `v` are dual variables of that
problem, and the worst-case measure is a primal object. Complementary slackness ties them:

- `u_ij > 0` says the atom of piece 1 at observation `i` sits on the support boundary in
  coordinate `j`, at `-1`;
- `s_i` equals the larger of the two rows `cu[i]`, `cv[i]`, and the tight row names the piece
  that carries mass at `i`; when both rows are tight the mass can be split.

Neither statement gives the **mass** of the atom or its **position** when it does not sit on
the boundary. Those are the multipliers of the rows, and JuMP keeps them on the constraint
references the builder stored under `cu`, `cv`, `cu_infnorm` and `cv_infnorm`. So the answer
to the ticket's item 2 is: **recoverable, from entries that are already registered, but from
their duals**. No new entry is needed. The one requirement is a solver that returns duals
(`JuMP.has_duals(model) == true`), which Clarabel does.

## 3. The recovery map

Fix `w` at its optimum and drop `sc`. Write `p_i` for the observation weight (`1/T` unweighted).
The inner problem is

```
min   ε λ + Σ_i p_i s_i
s.t.  b_k τ + a_k <w, xhat_i> + <γ_ik, 1 + xhat_i> <= s_i     multiplier  μ_ik >= 0
      ( λ, -γ_ik - a_k w ) ∈ K_∞                              multiplier  (η_ik, q_ik) ∈ K_1
      γ_ik >= 0                                               multiplier  ν_ik >= 0
```

where `K_∞` is the infinity-norm cone and `K_1`, its dual, is the 1-norm cone
`||q|| _1 <= η`. The Lagrangian is

```
L = ε λ + Σ p_i s_i
    + Σ_ik μ_ik [ b_k τ + a_k <w, xhat_i> + <γ_ik, 1 + xhat_i> - s_i ]
    - Σ_ik [ η_ik λ + <q_ik, -γ_ik - a_k w> ]
    - Σ_ik <ν_ik, γ_ik>
```

Stationarity in each primal variable gives the dual constraints:

| variable | condition | reading |
| :--- | :--- | :--- |
| `λ` | `Σ_ik η_ik = ε` | the transport budget is spent, `Σ_ik ||q_ik||_1 <= ε` |
| `s_i` | `Σ_k μ_ik = p_i` | the two pieces of observation `i` share its weight |
| `τ` | `Σ_ik μ_ik b_k = 0` | `τ` is the Rockafellar-Uryasev minimiser **under the worst case** |
| `γ_ik` | `ν_ik = μ_ik (1 + xhat_i) + q_ik >= 0` | the support constraint, once `q` is rescaled |

Define the atom `ξ_ik = xhat_i + q_ik / μ_ik` for every `(i, k)` with `μ_ik > 0`. Then the
`γ` row reads `μ_ik (1 + ξ_ik) >= 0`, so **every atom satisfies `ξ_ik >= -1`**; the `λ` row
reads `Σ_ik μ_ik ||ξ_ik - xhat_i||_1 <= ε`, which is the **type-1 Wasserstein budget with the
1-norm ground metric**; and the dual objective, after the `τ` term vanishes, is

```
Σ_ik μ_ik a_k <w, ξ_ik>  =  Σ_ik μ_ik [ a_k <w, ξ_ik> + b_k τ ]  =  E_Q[ loss ]
```

for `Q = Σ_ik μ_ik δ_{ξ_ik}`. Strong duality holds (the primal is a feasible conic programme
with a strictly feasible point: any `λ` large enough), so the dual optimum equals the
reported risk. This is Esfahani and Kuhn's Theorem 4.4 with the mass of atom `(i, k)` written
`μ_ik` in place of `α_ik / N`, and the displacement `q_ik / μ_ik` in place of `q_ik / α_ik`.

**The joint optimum over `w` changes nothing.** The library minimises over `w` and the inner
variables at once, and the KKT system of the joint problem contains the KKT system of the
inner problem at `w*`. So the multipliers the solver returns are optimal multipliers of the
inner problem, and the measure they define is a worst case **for the portfolio the model
chose**, which is the one a caller wants.

### Reading the duals off the JuMP model

MathOptInterface's convention for a minimisation is that the dual of `f(x) ∈ S` lies in the
dual cone `S*`, and the Lagrangian subtracts `<y, f(x)>`. For a `<= 0` row `S*` is the
non-positive orthant, so:

| registered entry | `JuMP.dual` returns | recovered quantity |
| :--- | :--- | :--- |
| `cu[i]` | a non-positive scalar | `μ_i1 = -dual(cu[i])` |
| `cv[i]` | a non-positive scalar | `μ_i2 = -dual(cv[i])` |
| `cu_infnorm[i]` | a vector `(η, q)` in the 1-norm cone | `q_i1 = dual(cu_infnorm[i])[2:end]` |
| `cv_infnorm[i]` | the same | `q_i2 = dual(cv_infnorm[i])[2:end]` |

Every raw dual carries the same common factor: `sc` from the constraint scale, `so` from the
objective scale, and, when the risk expression enters the model through a bound or a
scalarisation weight instead of the bare objective, the multiplier of that bound. Call the
product `θ`. The map is scale-free after one normalisation:

```
θ        = Σ_ik μ_ik
mass_ik  = μ_ik / θ                          (sums to one; Σ_k mass_ik = p_i)
atom_ik  = xhat_i + q_ik / μ_ik              (θ cancels in the ratio)
```

so the reader does not need to know `sc`, `so`, or how the measure was scalarised. It needs
`θ > 0`. When `θ = 0` the risk expression carried no weight at the optimum, for example a risk
bound that did not bind, and there is no worst case to read: the model did not price one.

The pseudo-code of the reader, as run in section 6:

```julia
mu    = hcat(-dual.(cu), -dual.(cv))          # T × 2
theta = sum(mu)
for i in 1:T, k in 1:2
    mu[i, k] <= tol * theta && continue
    q  = dual(k == 1 ? cu_infnorm[i] : cv_infnorm[i])[2:end]
    push!(atoms,  xhat[i, :] .+ q ./ mu[i, k])
    push!(masses, mu[i, k] / theta)
end
```

### What the map gives that the prototype's formula cannot

`worst_case_shifted_returns` in `research/prototypes/02_wasserstein_ambiguity.jl:369` moves
each of the `floor(αT)` worst rows by `-(ε/α) g`, with `g` the dual-norm attainer of `w`.
Under the library's ground metric `g` is a one-hot vector on the largest holding, so the
formula pushes one asset down by `ε/α` in every tail row, with no regard for the floor at
`-1`, and it moves no mass between pieces. The recovered measure does three things the
formula does not:

1. it splits an observation into **two atoms** where the solver found that optimal, one per
   piece, with the mass split the multipliers state;
2. it moves an atom **only as far as the support allows**, which is what makes it the worst
   case of the programme the library solved rather than of an unbounded one;
3. it spends the budget where the programme spent it, which need not be `floor(αT)` rows.

## 4. The verb: a recommendation for the decision ticket

The ticket's items 3 and 4 are a design decision, and this is a research ticket, so what
follows is a recommendation with the facts that constrain it.

**Inputs.** The reader needs the solved `JuMP.Model` (`JuMPOptimisationResult.model`, kept
when `save = true`, the default), the matrix the transport cost is measured against (the
Prior's `X` for the returns twin, `absolute_drawdown_arr(X)` for the drawdown twin), the
measure index `i` and the prefix under which the constraints were registered, and the
measure itself for `alpha` and `l` if the caller wants the risk identity checked. It does not
need `w`: the atoms are in asset space, and `w` enters only when a caller reduces them.

**It is not a Calibration Rule**, as the ticket says: a rule reads a Prior before the solve,
and this reads a Result after it. **It is not a risk-measure functor either**: a functor
takes a returns vector. The library has one family that reads a solved model after the fact,
the `expected_*` value-level family of map #295 (ADR 0036's penalty channel is the other
post-solve reader, but it writes). The reader fits that family's shape: a verb over a
`JuMPOptimisationResult` that returns a Result.

**Return shape.** A `T' × N` atom matrix and a `T'` mass vector with `T' <= 2T` are exactly
the `(X, w)` pair a Prior carries, and every functor and constraint builder in the library
already reads that pair through `nothing_scalar_array_selector(r.w, pr.w)` and
`get_observation_weights`. Section 6 uses that route as-is: `ConditionalValueatRisk(; alpha,
w = pweights(masses))(atoms * w)` reproduces the priced CVaR. So the cheapest shape that the
existing machinery understands is **a Prior-shaped pair**, and the decision to make is
whether it is wrapped as a `LowOrderPrior` (so it can be fed back into an optimiser as a
stress scenario set, which is the ticket's second use), or as a small Result of its own
with `atoms`, `masses`, `observation` (the row each atom came from) and `piece` (which of the
two affine pieces carried it). The last two columns are what makes the diagnostic readable
— "observation 17 split, 60 % of it moved to the floor in asset 3" — and a bare Prior loses
them.

**The radius reading.** The ticket's first use, "is the radius sane", falls out of the same
object: the largest 1-norm displacement `max_ik ||atom_ik - xhat_i||_1`, and the count of
atoms that sit on the floor `-1`, are the two numbers a committee reads. Both are one line
over the Result and need no further machinery.

**Constraints the decision must respect.**

- The Range twin builds one programme per tail under `nested_index` and the `gain_` prefix,
  with the gain tail's ambiguity measured against `-X .+ 1`. The reader runs once per tail
  and the gain tail's atoms come back **negated**. Two measures, two worst cases; there is
  no single worst case for a range.
- `save = false` drops the model, and then nothing can be read. The reader's error must say
  so.
- A solver that returns no duals (`JuMP.has_duals` false) gives nothing. Clarabel returns
  them; the reader must refuse, not return zeros.
- A model where the risk is a bound that did not bind has `θ = 0`. Refuse with that reason.

## 5. The drawdown twin

`DistributionallyRobustConditionalDrawdownatRisk` calls the same builder with `series =
-dd[2:T+1]`, the model's **portfolio drawdown variables**, and `ambiguity =
absolute_drawdown_arr(X) .+ 1`, the **per-asset** absolute drawdown panel plus one.

The recovery map applies unchanged, because it is the LP dual of the constraint rows and the
rows have the same shape: masses from `cu_drcdar_`, `cv_drcdar_`, displacements from the two
cone rows, atoms `d_i + q_ik / μ_ik` in **per-asset drawdown space**, support `d >= -1`, and
the dual objective equals the reported risk (section 6 checks it to `1e-8`).

What changes is the reading of an atom. In the returns twin the linear term the programme
prices is `a_k <w, ξ_ik>`, and `<w, ξ>` is the portfolio return of the atom, so an atom is a
return scenario in the full sense. In the drawdown twin the priced term is

```
a_k [ series_i + <w, d_ik - d_i> ]  =  a_k [ -dd_i(w) + <w, d_ik - d_i> ]
```

The portfolio drawdown is **not** `<w, d_i>` — section 6 measures the gap at the optimum —
so the programme is Esfahani and Kuhn's reformulation with the linear term replaced by the
observed portfolio drawdown plus a **first-order** shift `<w, d_ik - d_i>`. That is a
consistent object: the recovered measure is the worst case of the programme the library
solves, and the risk identity holds. But an atom `d_ik` is a per-asset drawdown row, not a
return path, and the portfolio drawdown the programme charges for it is the linearised one,
not the drawdown of any path. So the drawdown twin's atoms are **stress rows in drawdown
space**, and they cannot be fed back into an optimiser as a returns panel, nor scored by the
library's `ConditionalDrawdownatRisk` functor over a rebuilt path and expected to reproduce
the priced number.

**Recommendation for item 5:** one reader serves both twins, parameterised by the matrix the
transport cost was measured against, and the Result names which space its atoms live in. The
decision ticket must decide whether the drawdown twin's Result is a Prior-shaped pair at
all, since the pair's `X` would be drawdowns and no Prior reads that.

## 6. Numerical check

A bare process, `julia -t 1 --project=test`, at `e6c7840bc2`, Clarabel at tolerances `1e-12`.
`T = 40`, `N = 4`, `alpha = 0.1`, `l = 1`, a `MeanRisk` with the measure as its only risk
and `EmpiricalPrior`. Every solve returned `OptimisationSuccess` and `JuMP.has_duals` was
`true` on every model.

### Returns twin, `r = 0.01`, daily-sized returns

| check | value |
| :--- | :--- |
| `max_i abs(sum_k mass_ik - 1/T)` | `1.35e-16` |
| atoms | 40 of a possible 80; no observation split |
| smallest atom coordinate | `-0.0740`, the floor `-1` is inactive |
| `sum_ik mass_ik * norm(atom_ik - xhat_i, 1)` | `0.010000`, ratio to the radius `1.000000` |
| reported risk | `0.03914703` |
| dual objective over the recovered measure | `0.03914703`, difference `6.9e-10` |
| `-E_Q[w'xi] + l * CVaR_alpha^Q(-w'xi)` by the library's weighted functor | `0.03914703`, difference `-6.8e-10` |
| empirical risk at the same `w` | `0.01164703` |
| prototype's closed-form bound `empirical + r * norm(w, Inf) * (1 + l/alpha)` | `0.03914703` |
| `lb` against `norm(w, Inf) * (1 + l/alpha)` | `2.750000` against `2.750000` |

Here the floor is far away, so the closed form is tight and `lb` is the unconstrained
Lipschitz modulus. This is the regime the prototype was checked in, and it is why the
prototype's docstring could call its formula "the honest answer".

### Returns twin, `r = 1e-9`

The constructor refuses `r = 0` (`DomainError: 0 < r must hold`), so the empirical limit is
taken at `1e-9`. The decision ticket's test plan must do the same, or compare against the
plain `ConditionalValueatRisk` programme.

| check | value |
| :--- | :--- |
| `max abs(atom - xhat)` | `1.53e-08` |
| reported risk | `0.00816895` |
| empirical `mean loss + l * CVaR` at the same `w` | `0.00816895` |
| atoms | 44: four observations sit at the Rockafellar-Uryasev kink and split into both pieces |

The split at the kink is the map's "two atoms per observation" case, and it appears even
at a vanishing radius: it is the multiplier splitting a tied observation, not a transport.

### Returns twin, observation weights `pweights(range(1, 3, T))`, `r = 0.01`

| check | value |
| :--- | :--- |
| `max_i abs(sum_k mass_ik - p_i)` | `5.0e-16`, with `p_i` the normalised weight |
| budget ratio | `1.000000` |
| risk identity | difference `2.7e-11` |

So the masses of the two pieces of observation `i` sum to **that observation's weight**, and
the map needs no change for a weighted sample.

### Returns twin, the support floor binds, `r = 0.3`, returns clipped at `-0.95`

| check | value |
| :--- | :--- |
| smallest atom coordinate | `-1.000000` |
| atoms with a coordinate on the floor | 4 of 40; `u > 1e-6` in 16 entries, the same 4 rows in all 4 assets |
| budget ratio | `1.000000` |
| reported risk | `1.30855004`; dual objective difference `1.3e-09`; functor route difference `-1.3e-09` |
| empirical risk at the same `w` | `0.72340179` |
| prototype's closed-form bound | `2.23590179`, which is `0.927352` **above** the priced risk |
| `lb` against `norm(w, Inf) * (1 + l/alpha)` | `0.458333` against `5.041667` |
| prototype's shifted rows | 4 rows moved by `3.000`; smallest shifted return `-3.8035`, below the floor |

This is the ticket's claim in numbers. The prototype's formula names a scenario that loses
380 % of an asset's value, the programme never priced it, and the bound it implies is 71 %
above what the model is robust to. The recovered measure stops at the floor, and its `lb`
is the **constrained** modulus `norm(w, Inf)`, not `norm(w, Inf) * (1 + l/alpha)`: with the
floor active the tail piece cannot be pushed further, so only the mean piece pays.

### Drawdown twin, `r = 0.01`

| check | value |
| :--- | :--- |
| `max_i abs(sum_k mass_ik - 1/T)` | `1.73e-17` |
| smallest atom coordinate | `-0.1802`, the floor is inactive |
| budget ratio | `1.000000` |
| reported risk | `0.05998042`; dual objective difference `3.1e-11` |
| `max_i abs(series_i - (D w)_i)` | `0.0623` |

The last row is section 5's point: the model's series is the portfolio drawdown, which is
not the weighted per-asset drawdown, so the priced linear term is a linearisation. The
identity holds because the dual objective uses `series_i + <w, atom_i - d_i>`, and it would
not hold for a CDaR functor over a path rebuilt from the atoms.

## Reproduction

The script below is what produced section 6. Run it with
`julia -t 1 --project=test research/627-worst-case-measure-recovery.jl` after copying it
out, or paste it into a fresh process. It reads nothing but the public API and
`JuMP.object_dictionary`.

```julia
using PortfolioOptimisers, Clarabel, JuMP, StableRNGs, Statistics, LinearAlgebra, StatsBase
using Printf
BLAS.set_num_threads(1)
const PO = PortfolioOptimisers

rng = StableRNG(20260903)
T, N = 40, 4
X = 0.0005 .+ 0.02 * randn(rng, T, N)
rd = ReturnsResult(; nx = string.(1:N), X = X)
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             check_sol = (; allow_local = true, allow_almost = true),
             settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                             "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                             "max_iter" => 500))

# Read a registered entry whose bare name is `stem` followed by the measure index only.
function entry(model, stem)
    od = JuMP.object_dictionary(model)
    ks = filter(k -> occursin(Regex("^" * stem * "\\d+\$"), string(k)), collect(keys(od)))
    return od[only(ks)]
end

"""
Recover the worst-case measure from the solved model.

Returns `(atoms, masses, theta, budget, mu)` where `atoms` is a `T' x N` matrix, `masses`
a `T'` vector that sums to one, `theta` the common scale of the raw duals, `budget` the
transport cost `sum_i,k mass_ik * ||atom_ik - xhat_i||_1`, and `mu` the `T x 2` raw masses.
"""
function recover(model, xhat, stem)
    T = size(xhat, 1)
    cu = entry(model, "cu_" * stem)
    cv = entry(model, "cv_" * stem)
    cui = entry(model, "cu_" * stem * "infnorm_")
    cvi = entry(model, "cv_" * stem * "infnorm_")
    mu = hcat(-JuMP.dual.(cu), -JuMP.dual.(cv))          # T x 2, raw multipliers
    theta = sum(mu)
    atoms = Vector{Vector{Float64}}()
    masses = Float64[]
    piece = Int[]
    obs = Int[]
    budget = 0.0
    for i in 1:T, (k, con) in enumerate((cui, cvi))
        m = mu[i, k]
        m <= 1e-9 * theta && continue
        y = JuMP.dual(con[i])                             # (eta_i, q_i) in the 1-norm cone
        q = y[2:end]
        xi = xhat[i, :] .+ q ./ m
        push!(atoms, xi)
        push!(masses, m / theta)
        push!(piece, k)
        push!(obs, i)
        budget += (m / theta) * norm(q ./ m, 1)
    end
    return (; atoms = permutedims(reduce(hcat, atoms)), masses, theta, budget, mu, piece, obs)
end

function report(name, model, xhat, stem, alpha, l, radius, series_fn, wobs)
    println("\n=== ", name, " ===")
    println("has_duals = ", JuMP.has_duals(model))
    w = JuMP.value.(model[:w])
    tau = JuMP.value(entry(model, "tau_" * stem))
    lb = JuMP.value(entry(model, "lb_" * stem))
    risk = JuMP.value(entry(model, stem * "risk_"))
    rec = recover(model, xhat, stem)
    T = size(xhat, 1)
    p = isnothing(wobs) ? fill(1 / T, T) : wobs ./ sum(wobs)
    # 1. masses: the two pieces of one observation sum to that observation's weight
    rowsum = vec(sum(rec.mu; dims = 2)) ./ rec.theta
    @printf("masses per observation: max |sum_k mu_ik/theta - p_i| = %.2e\n",
            maximum(abs.(rowsum .- p)))
    @printf("atoms: %d of a possible %d; both pieces active at %d observations\n",
            length(rec.masses), 2T, count(==(2), [count(==(i), rec.obs) for i in 1:T]))
    # 2. support
    @printf("support: min atom = %.6f (must be >= -1); max shift = %.4f\n",
            minimum(rec.atoms), maximum(abs.(rec.atoms .- xhat[rec.obs, :])))
    # 3. budget
    @printf("budget: sum mass*||shift||_1 = %.6f, radius = %.6f, ratio = %.6f\n", rec.budget,
            radius, rec.budget / radius)
    # 4. risk identity via the dual objective
    a = (-1.0, -1.0 - l / alpha)
    b = (l, l * (1 - 1 / alpha))
    ser = series_fn(w)                                    # T: the model's series per observation
    dualobj = sum(rec.masses[j] *
                  (a[rec.piece[j]] * (ser[rec.obs[j]] +
                                      dot(w, rec.atoms[j, :] .- xhat[rec.obs[j], :])) +
                   b[rec.piece[j]] * tau) for j in eachindex(rec.masses))
    @printf("risk reported = %.8f, risk from recovered measure (dual objective) = %.8f, diff = %.2e\n",
            risk, dualobj, risk - dualobj)
    @printf("radius * lb = %.8f, lb = %.6f, ||w||_inf = %.6f\n", radius * lb, lb, norm(w, Inf))
    return (; w, tau, lb, risk, rec, ser)
end

# ---------------------------------------------------------------- returns twin
alpha, l, radius = 0.1, 1.0, 0.01
r = DistributionallyRobustConditionalValueatRisk(; alpha = alpha, l = l, r = radius)
mr = MeanRisk(; r = r, opt = JuMPOptimiser(; slv = slv, pe = EmpiricalPrior()))
res = optimise(mr, rd)
println("retcode: ", res.retcode)
out = report("DR-CVaR, equal weights", res.model, X, "drcvar_", alpha, l, radius, w -> X * w,
             nothing)
# Risk under Q with the library's own functors: E_Q[-w'xi] + l * CVaR_alpha^Q(-w'xi).
retsQ = out.rec.atoms * out.w
mq = pweights(out.rec.masses)
cvarQ = ConditionalValueatRisk(; alpha = alpha, w = mq)(retsQ)
riskQ = -dot(out.rec.masses, retsQ) + l * cvarQ
@printf("risk under Q by the functors: mean loss + l*CVaR = %.8f (reported %.8f, diff %.2e)\n",
        riskQ, out.risk, riskQ - out.risk)
# Empirical risk at the same w, for scale, and the prototype's closed-form upper bound.
rets0 = X * out.w
cvar0 = ConditionalValueatRisk(; alpha = alpha)(rets0)
risk0 = -mean(rets0) + l * cvar0
bound = risk0 + radius * norm(out.w, Inf) * (1 + l / alpha)
@printf("empirical risk at w = %.8f; robust = %.8f; closed-form bound (unbounded support) = %.8f\n",
        risk0, out.risk, bound)
# The prototype's formula applied with the library's ground metric: shift the worst
# floor(alpha*T) rows by -(radius/alpha)*g with g = the inf-norm attainer of w.
k = max(1, floor(Int, alpha * T))
tail = partialsortperm(-rets0, 1:k; rev = true)
g = zeros(N); g[argmax(abs.(out.w))] = sign(out.w[argmax(abs.(out.w))])
Xp = copy(X)
for t in tail
    Xp[t, :] .-= (radius / alpha) .* g
end
@printf("prototype shift: rows moved = %d, min shifted return = %.6f, shift per row = %.6f\n", k,
        minimum(Xp), radius / alpha)

# ------------------------------------------------- radius of zero: the empirical measure
r0 = DistributionallyRobustConditionalValueatRisk(; alpha = alpha, l = l, r = 1e-9)
res0 = optimise(MeanRisk(; r = r0, opt = JuMPOptimiser(; slv = slv, pe = EmpiricalPrior())), rd)
println("retcode (r = 0): ", res0.retcode)
out0 = report("DR-CVaR, radius 1e-9", res0.model, X, "drcvar_", alpha, l, 1e-9, w -> X * w, nothing)
@printf("radius 0: max |atom - xhat| = %.2e\n",
        maximum(abs.(out0.rec.atoms .- X[out0.rec.obs, :])))
rets00 = X * out0.w
@printf("radius 0: reported risk %.8f vs empirical mean loss + l*CVaR %.8f\n", out0.risk,
        -mean(rets00) + l * ConditionalValueatRisk(; alpha = alpha)(rets00))

# ------------------------------------------------------------- observation weights
wobs = pweights(range(; start = 1, stop = 3, length = T))
rw = DistributionallyRobustConditionalValueatRisk(; alpha = alpha, l = l, r = radius, w = wobs)
resw = optimise(MeanRisk(; r = rw, opt = JuMPOptimiser(; slv = slv, pe = EmpiricalPrior())), rd)
println("retcode (weighted): ", resw.retcode)
outw = report("DR-CVaR, observation weights", resw.model, X, "drcvar_", alpha, l, radius,
              w -> X * w, wobs)

# ------------------------------------------------------------------ drawdown twin
D = PO.absolute_drawdown_arr(X)
rd_dd = DistributionallyRobustConditionalDrawdownatRisk(; alpha = alpha, l = l, r = radius)
resd = optimise(MeanRisk(; r = rd_dd, opt = JuMPOptimiser(; slv = slv, pe = EmpiricalPrior())),
                 rd)
println("retcode (drawdown): ", resd.retcode)
ddvar = resd.model[:dd]
outd = report("DR-CDaR", resd.model, D, "drcdar_", alpha, l, radius,
              w -> -JuMP.value.(ddvar)[2:(T + 1)], nothing)
# Is the model's drawdown series the weighted sum of per-asset drawdowns? (It is not.)
@printf("drawdown twin: max |series - D*w| = %.4f (series is the portfolio drawdown, not D*w)\n",
        maximum(abs.(outd.ser .- D * outd.w)))
# A drawdown atom read as a returns path: the library's CDaR of the shifted panel is not
# what the programme priced. Show the size of the gap.

# ---------------------------------------------- the support floor binds
# Returns big enough that an unconstrained shift would cross -1, and a radius that pays
# for it. Here the prototype's closed form is a strict upper bound on the programme.
Xb = clamp.(-0.3 .+ 0.35 * randn(rng, T, N), -0.95, 2.0)
rdb = ReturnsResult(; nx = string.(1:N), X = Xb)
radb = 0.3
rb = DistributionallyRobustConditionalValueatRisk(; alpha = alpha, l = l, r = radb)
resb = optimise(MeanRisk(; r = rb, opt = JuMPOptimiser(; slv = slv, pe = EmpiricalPrior())), rdb)
println("retcode (floor binds): ", resb.retcode)
outb = report("DR-CVaR, support floor binds", resb.model, Xb, "drcvar_", alpha, l, radb,
              w -> Xb * w, nothing)
retsQb = outb.rec.atoms * outb.w
riskQb = -dot(outb.rec.masses, retsQb) +
         l * ConditionalValueatRisk(; alpha = alpha, w = pweights(outb.rec.masses))(retsQb)
rets0b = Xb * outb.w
risk0b = -mean(rets0b) + l * ConditionalValueatRisk(; alpha = alpha)(rets0b)
boundb = risk0b + radb * norm(outb.w, Inf) * (1 + l / alpha)
@printf("floor binds: risk under Q by functors = %.8f (reported %.8f, diff %.2e)\n", riskQb,
        outb.risk, riskQb - outb.risk)
@printf("floor binds: empirical %.8f; robust %.8f; closed-form bound %.8f; bound - robust = %.6f\n",
        risk0b, outb.risk, boundb, boundb - outb.risk)
@printf("floor binds: atoms on the floor (coordinate within 1e-6 of -1) = %d of %d atoms; u > 1e-6 entries = %d\n",
        count(any(outb.rec.atoms .< -1 + 1e-6; dims = 2)), length(outb.rec.masses),
        count(JuMP.value.(entry(resb.model, "u_drcvar_")) .> 1e-6))
kb = max(1, floor(Int, alpha * T))
tailb = partialsortperm(-rets0b, 1:kb; rev = true)
gb = zeros(N); gb[argmax(abs.(outb.w))] = sign(outb.w[argmax(abs.(outb.w))])
Xpb = copy(Xb)
for t in tailb
    Xpb[t, :] .-= (radb / alpha) .* gb
end
@printf("floor binds: prototype shift moves %d rows by %.3f; min shifted return = %.4f (below the floor -1)\n",
        kb, radb / alpha, minimum(Xpb))
println("\nDONE")
```
