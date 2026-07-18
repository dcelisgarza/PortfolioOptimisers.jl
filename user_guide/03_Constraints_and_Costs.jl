#=
# Constraints and costs

Real mandates are not unconstrained. You cap concentration, hold a sector band, limit how much
you trade at each rebalance, and pay transaction costs. In `PortfolioOptimisers.jl` these are
**keywords on the [`JuMPOptimiser`](@ref)** — the optimiser carries the constraints and costs,
the estimator carries the objective. This page shows the common ones with one minimal call
each; for the full treatment see the
[constraints & costs examples](../examples/4_constraints_costs/01_Budget_Constraints.md).

We fix one empirical prior and a minimum-risk objective so each keyword's effect is visible
against the same baseline.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res_base = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
## 1. Weight bounds

[`WeightBounds`](@ref) sets the per-asset lower and upper bound through the `wb` keyword. The
default is `lb = 0, ub = 1` (long-only, fully invested). Capping `ub` forces diversification —
no single name can exceed the bound.
=#

res_cap = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                wb = WeightBounds(; lb = 0.0, ub = 0.10))))

#=
The budget itself is the `bgt` keyword (default `1.0`); `BudgetRange` and a separate short
budget `sbgt` let you build long/short and leveraged mandates — see
[Budget Constraints](../examples/4_constraints_costs/01_Budget_Constraints.md).

## 2. Linear and group constraints

Group and linear constraints are written as plain strings over an [`AssetSets`](@ref) and passed
through `lcse` as a [`LinearConstraintEstimator`](@ref) — the same syntax used for views. Name a
group, then bound it. Here we require the tech group to hold at least 15% (a floor the
unconstrained minimum-risk portfolio would not give it).
=#

sets = AssetSets(; dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"]))
res_grp = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                lcse = LinearConstraintEstimator(;
                                                                                 val = ["tech >= 0.15"]))))

#=
The same `lcse` handles absolute and relative bounds (`"AAPL <= 0.1"`, `"MSFT >= AMD"`). For
constraints built from the asset *hierarchy* — phylogeny and centrality — see
[Phylogeny & Centrality](../examples/4_constraints_costs/04_Phylogeny_Centrality.md).

## 3. Turnover

Costs enter the same way. [`Turnover`](@ref) (`tn`) limits how far the new weights may drift
from a reference portfolio `w` — your current holdings — so a rebalance stays cheap. Here we
anchor at the current minimum-risk portfolio and re-solve under a turnover budget.
=#

res_tn = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               tn = Turnover(; w = res_base.w, val = 0.02))))

#=
A tighter `val` keeps the result closer to the reference holdings; a looser one frees the
optimiser to move toward the unconstrained solution.

## 4. Fees

[`Fees`](@ref) (`fees`) charges proportional (and optionally fixed) transaction costs on long
and short positions, which the objective then trades off against return. The minimal form sets a
per-unit long fee.
=#

res_fee = optimise(MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                fees = Fees(; l = 0.001))))

#=
Soft alternatives to hard turnover/position limits — L1/L2 weight regularisation and a
weight-norm ceiling that doubles as a diversification floor (`l1`, `l2`, `l2c`) — are covered in
[Regularisation](../examples/4_constraints_costs/07_Regularisation.md); benchmark
[`Tracking`](@ref) (the `tr` keyword) in
[Turnover & Tracking](../examples/4_constraints_costs/05_Turnover_and_Tracking.md).

## 5. Custom objectives and constraints

When a mandate needs something no built-in keyword covers, two [`JuMPOptimiser`](@ref) extension
points let you write straight against the JuMP model. Each takes an estimator (or a vector of
them) and dispatches on its type:

  - `cobj` takes a [`CustomJuMPObjective`](@ref) — implement [`add_custom_objective_term!`](@ref)
    to add a term to the objective expression.
  - `ccnt` takes a [`CustomJuMPConstraint`](@ref) — implement [`add_custom_constraint!`](@ref) to
    add a constraint to the model.

To show both, we tilt toward a *momentum score* — each asset's standardised trailing-63-day
return. It is a continuous per-asset number, so no group string could express it.
=#

using JuMP: JuMP

score = let m = vec(sum(rd.X[(end - 62):end, :]; dims = 1))
    (m .- mean(m)) ./ std(m)
end

#=
**A custom objective.** Subtype [`CustomJuMPObjective`](@ref) and mutate the objective
expression. `MinimumRisk` is a *minimisation*, so we *subtract* the score reward (scaled by
`lambda`) to favour high-momentum names; read the weight variables with the [`get_w`](@ref)
accessor rather than reaching into `model[:w]` directly.
=#

struct MomentumTilt{T1, T2} <: PortfolioOptimisers.CustomJuMPObjective
    score::T1
    lambda::T2
end
function PortfolioOptimisers.add_custom_objective_term!(model::JuMP.Model, obj, pret,
                                                        cobj::MomentumTilt, obj_expr, opt,
                                                        pr, args...)
    sign = ifelse(isa(obj, MinimumRisk), -1, 1)
    w = PortfolioOptimisers.get_w(model)
    JuMP.add_to_expression!(obj_expr, sign * cobj.lambda * (cobj.score' * w))
    return nothing
end

res_tilt = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                 cobj = MomentumTilt(score, 1e-4))))

#=
**A custom constraint.** Subtype [`CustomJuMPConstraint`](@ref) and add a JuMP constraint,
following the model idiom: scale the constraint by [`get_constraint_scale`](@ref) and multiply
any constant bound by the homogenisation variable [`get_k`](@ref) (`1` here, but load-bearing
under a ratio objective). This puts a hard *floor* on the same momentum exposure.
=#

struct MomentumFloor{T1, T2} <: PortfolioOptimisers.CustomJuMPConstraint
    score::T1
    floor::T2
end
function PortfolioOptimisers.add_custom_constraint!(model::JuMP.Model, ccnt::MomentumFloor,
                                                    opt, attrs)
    w = PortfolioOptimisers.get_w(model)
    k = PortfolioOptimisers.get_k(model)
    sc = PortfolioOptimisers.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.score' * w - ccnt.floor * k) >= 0)
    return nothing
end

res_floor = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  ccnt = MomentumFloor(score, 1.0))))

#=
Both lift the portfolio's momentum exposure `score'w` above the baseline — the soft tilt as far
as the risk trade-off allows, the hard floor to exactly its bound.
=#

momentum_exposure = (base = score' * res_base.w, tilt = score' * res_tilt.w,
                     floor = score' * res_floor.w)

#=
## 6. Comparing the effect

Same prior, same objective — only the constraint or cost changes the allocation.
=#

results = [res_base, res_cap, res_grp, res_tn, res_fee, res_tilt, res_floor]
labels = ["Base", "Cap 10%", "Tech ≥ 15%", "Turnover", "Fees", "Momentum tilt",
          "Momentum floor"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Weights under each constraint / cost")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Shallow guide page: constraints & costs are JuMPOptimiser keywords (wb/bgt/lcse/tn/fees),
#src   one minimal call each, depth deferred to 4_constraints_costs examples. Verified on kaimon
#src   (session f102cae9).
#src - Verified bindings: wb ub=0.10 caps max weight 37%→10%; group FLOOR "tech >= 0.15" forces
#src   tech 0%→15% (a tech CAP never binds on this slice — min-risk AND max-ratio both put ~0%
#src   in AAPL/AMD/MSFT, so I used a floor for a visible effect). Turnover tightening pulls the
#src   solution toward the reference (directional; `val` is NOT literally sum|Δw|, so prose stays
#src   qualitative). Fees solves on a MaximumRatio objective.
#src - Cross-links to 4_constraints_costs/02 (linear/group), 03 (phylogeny/centrality), 04
#src   (turnover/tracking) point at example pages NOT YET AUTHORED — resolve on Documenter
#src   linkcheck once that group lands. → guide/examples cross-link audit.
#src - §5 Custom objectives/constraints added for ergonomics-review-20260718 card 6 (the
#src   extension point had no worked example). Both verified at Main top level on kaimon
#src   (session 29092601): MomentumTilt cobj lifts score'w 0.204→1.403; MomentumFloor ccnt
#src   pins it to the 1.0 floor. Idiom checks: objective mutates obj_expr via add_to_expression!
#src   + get_w; constraint scales by get_constraint_scale and multiplies the constant by get_k
#src   (k=1 under MinimumRisk but load-bearing under a ratio objective, so kept for correctness).
#src   Scoped to MinimumRisk (Min sense ⇒ subtract the reward); a ratio-objective tilt would
#src   need the k-normalisation caveat spelled out. `dot` is not exported by PO → used score'*w.
