#=
```@meta
Description = "Custom objectives and constraints in PortfolioOptimisers.jl: write your own terms into the JuMP model with CustomJuMPObjective and CustomJuMPConstraint."
```

# Custom objectives and constraints

Each keyword of the [`JuMPOptimiser`](@ref), such as the bounds, the budgets, turnover, fees and
cardinality, adds a ready-made part to the problem. When a mandate needs something that no
keyword covers, you can write a term into the JuMP model yourself, in one of two ways.

  - Subtype [`CustomJuMPObjective`](@ref), pass it to the `cobj` keyword, and write a method of
    [`add_custom_objective_term!`](@ref). The term puts a price on a preference in the
    objective.
  - Subtype [`CustomJuMPConstraint`](@ref), pass it to the `ccnt` keyword, and write a method of
    [`add_custom_constraint!`](@ref). The constraint must hold in the solution.

Each keyword takes one estimator or a vector of them. The library calls the function with your
estimator as an argument, so Julia picks your method by the type of the estimator. A custom term
is a struct that holds its data, and one method. If you subtype one of the two types
and write no method, the optimisation throws an error that names the missing method. This page
builds both kinds of term, shows the two rules that keep a constraint correct, and combines
several terms in one problem. The [constraints and costs
guide](../../user_guide/04_Constraints_and_Costs.md) gives a short summary of the same features.

!!! tip "When to reach for this"
    Reach for a custom term when your preference is a number per asset that no group string can
    express, such as a factor score, a carbon intensity or a liquidity penalty. Reach for it
    also for a relation between weights that is not a plain linear bound. If a linear or group
    constraint (`lcse`) or another keyword can express it, use that instead, because the library
    tests those and they work with every other keyword. Write a custom term only when no keyword
    fits.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes
using JuMP: JuMP

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Data and a momentum score

We fit one empirical prior, set up one solver, and optimise a minimum risk baseline, so you can
compare every custom term with the same portfolio. The preference is a momentum score, the sum
of each asset's daily returns over the last 63 days, standardised across the assets. It is a
number per asset, which is the case that custom terms are for.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

score = let m = vec(sum(rd.X[(end - 62):end, :]; dims = 1))
    (m .- mean(m)) ./ std(m)
end

res_base = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
The momentum exposure of a portfolio is `score' * w`. The minimum risk baseline takes no account
of momentum. We print its exposure. Then we raise the exposure, first with a term in the
objective and then with a constraint.
=#

base_exposure = score' * res_base.w

#=
## 2. A custom objective: a tilt toward momentum

The library calls your method of `add_custom_objective_term!` while it builds the model. The
method adds a term to the objective penalty, a sum of terms that the library adds to the
objective. Its signature is

```julia
add_custom_objective_term!(model, obj, cobj, optimiser, attrs)
```

  - `model` is the [`JuMP`](https://jump.dev/) model that the library builds.
  - `obj` is the [`ObjectiveFunction`](@ref) of the problem, such as `MinimumRisk` or
    `MaximumUtility`. Dispatch on it if your term must change with the objective. You do not
    need it to get the sign right.
  - `cobj` is your estimator, and your method dispatches on its type.
  - `optimiser` is the optimisation estimator, for example the [`MeanRisk`](@ref) itself. Its
    `opt` field is the [`JuMPOptimiser`](@ref).
  - `attrs` is a [`ProcessedJuMPOptimiserAttributes`](@ref), the data of the problem after
    processing. It holds the prior in `attrs.pr`, the returns estimator in `attrs.ret`, the
    weight bounds in `attrs.wb`, and more. Read it if your term computes its numbers from the
    data.

Get the weight variables with [`get_w`](@ref), not with `model[:w]`. If the model has no weight
variables yet, `get_w` throws an error that says so.

Add the term with [`add_to_objective_penalty!`](@ref), not by a change to the objective
expression. Then the term is correct under every objective.

!!! tip "The library gives your term the right sign"
    Some objectives are minimised and some are maximised, and [`MaximumRatio`](@ref) can be
    either, depending on the risk measure. A term that you add to the objective expression
    yourself therefore has no sign that is right in every case. The library adds the objective
    penalty to the objective with a factor of 1 when it minimises and -1 when it maximises. With
    this factor, a positive term always makes the objective worse, and a reward is a negative
    term. Write `-λ * something_good` once, and it is a reward under every objective.

    The objective penalty also becomes quadratic when you add a quadratic term, such as an L2
    tilt or a tracking penalty. A quadratic term therefore works with every objective, including
    the linear ones, where a direct change to the objective expression throws a `MethodError`.
=#

struct MomentumTilt{T1, T2} <: PortfolioOptimisers.CustomJuMPObjective
    score::T1
    lambda::T2
end

function PortfolioOptimisers.add_custom_objective_term!(model::JuMP.Model, obj,
                                                        cobj::MomentumTilt, optimiser,
                                                        attrs)
    w = PortfolioOptimisers.get_w(model)
    ## Negative penalty == reward. No sign dispatch, no objective-type special cases.
    PortfolioOptimisers.add_to_objective_penalty!(model, -cobj.lambda * (cobj.score' * w))
    return nothing
end

#=
`lambda` is the price of momentum against risk. We solve for four values of `lambda`, and print
the momentum exposure and the largest weight of each result. The smallest price above zero,
`1e-4`, already moves the weights far from the baseline. Each larger price raises the exposure
by less than the one before, because the weights move toward the one asset with the highest
score.
=#

lambdas = [0.0, 1e-4, 5e-4, 2e-3]
tilt_res = [optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  cobj = MomentumTilt(score, l))))
            for l in lambdas]

pretty_table(DataFrame("λ (momentum price)" => lambdas,
                       "Momentum exposure" => [score' * r.w for r in tilt_res],
                       "Max weight" => [maximum(r.w) for r in tilt_res]);
             formatters = [resfmt],
             title = "A larger λ buys more momentum exposure, until it saturates")

#=
The term is homogeneous of degree one in `w`. If you multiply the weights by a number, the term
is multiplied by the same number, as the expected return is. The term stays correct
when a ratio objective rescales the weights. A constraint needs more care with that rescaling,
which section 5 shows.

## 3. The same term in a maximisation

`MaximumUtility` is a maximisation, where the problem above is a minimisation. We use the same
`MomentumTilt` with no change, and print the momentum exposure without it and with it. The
library adds the objective penalty with the factor of a maximisation, so the tilt is a reward
here too.
=#

util_base = optimise(MeanRisk(; obj = MaximumUtility(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv)))
util_tilt = optimise(MeanRisk(; obj = MaximumUtility(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  cobj = MomentumTilt(score, 5e-3))))

util_exposures = (base = score' * util_base.w, tilted = score' * util_tilt.w)

#=
!!! note "[`MaximumRatio`](@ref) needs no special case"
    The library solves the maximum-ratio problem with a change of variables. Depending on the
    risk measure, the result is a maximisation, or a minimisation of the risk. Each form adds
    the objective penalty with its own factor, so the tilt is a reward in both. This fixes the
    sign, not the scale. A term that is not homogeneous of degree one in `w` still needs the
    variable `k` of section 5.

## 4. A custom constraint: a floor on momentum

The library calls your method of `add_custom_constraint!` in the same way. Its signature is

```julia
add_custom_constraint!(model, ccnt, optimiser, attrs)
```

The arguments are those of `add_custom_objective_term!` without `obj`. `ccnt` is your estimator,
and your method dispatches on its type.

Follow two rules when you write a constraint.

 1. Multiply the constraint by [`get_constraint_scale`](@ref), which returns `model[:sc]`. The
    library scales every constraint it makes by this number, so yours then has the same scale.
 2. Multiply each constant bound by [`get_k`](@ref), which returns `model[:k]`. For most
    objectives `k` is the constant 1, and the product changes nothing. Under a ratio objective
    `k` is a variable at or above zero, and the solver works with rescaled weights, `w_real = w
    / k`. A constant with no `k` is then compared with the rescaled weights, which is the wrong
    comparison. Section 5 shows the result.
=#

struct MomentumFloor{T1, T2} <: PortfolioOptimisers.CustomJuMPConstraint
    score::T1
    floor::T2
end

function PortfolioOptimisers.add_custom_constraint!(model::JuMP.Model, ccnt::MomentumFloor,
                                                    optimiser, attrs)
    w = PortfolioOptimisers.get_w(model)
    k = PortfolioOptimisers.get_k(model)
    sc = PortfolioOptimisers.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.score' * w - ccnt.floor * k) >= 0)
    return nothing
end

#=
A floor acts differently from the tilt. When the floor binds, the exposure equals the floor, and
the weights minimise the risk under that condition. A floor below the exposure of the baseline
does not bind. We solve for four floors, and print the exposure of each result next to its
floor. The `Binds?` column is `yes` when the exposure is at most `1e-6` above the floor.
=#

floors = [0.0, 0.5, 1.0, 1.35]
floor_res = [optimise(MeanRisk(; obj = MinimumRisk(),
                               opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                   ccnt = MomentumFloor(score, f))))
             for f in floors]

pretty_table(DataFrame("Momentum floor" => floors,
                       "Momentum exposure" => [score' * r.w for r in floor_res],
                       "Binds?" => [score' * r.w > f + 1e-6 ? "no" : "yes"
                                    for (f, r) in zip(floors, floor_res)]);
             formatters = [resfmt], title = "A hard floor clamps the exposure to its bound")

#=
## 5. Why a constant bound needs `k`

We write the same floor without the `* k`. Under `MinimumRisk`, `k` is 1, so the mistake has no
effect there and is easy to miss.
=#

struct MomentumFloorNoK{T1, T2} <: PortfolioOptimisers.CustomJuMPConstraint
    score::T1
    floor::T2
end
function PortfolioOptimisers.add_custom_constraint!(model::JuMP.Model,
                                                    ccnt::MomentumFloorNoK, opt, attrs)
    w = PortfolioOptimisers.get_w(model)
    sc = PortfolioOptimisers.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.score' * w - ccnt.floor) >= 0)  # forgot `* k`
    return nothing
end

#=
Under [`MaximumRatio`](@ref), `k` is a variable of the problem, so the two floors give different
results. The floor with `k` holds the exposure of the final weights at the bound. The floor
without `k` holds the exposure of the rescaled weights at the bound, which is a different
number. We compare the two over three floors. At `floor = 1.38`, the version without `k` gives
an exposure below the floor, and no error tells you.
=#

k_floors = [1.30, 1.35, 1.38]
k_compare = [(f,
              score' * optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                ccnt = MomentumFloor(score, f)))).w,
              score' * optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                ccnt = MomentumFloorNoK(score, f)))).w)
             for f in k_floors]

pretty_table(DataFrame("Requested floor" => first.(k_compare),
                       "With * k (correct)" => getindex.(k_compare, 2),
                       "Without * k (wrong)" => getindex.(k_compare, 3));
             formatters = [resfmt],
             title = "Under a ratio objective, only the k-scaled floor binds where asked")

#=
## 6. Composing several custom pieces

Both keywords take a vector of estimators, and the library applies them in order, with one
method call for each. You can combine custom terms without one struct for all of them.

First, a band from two constraints. A cap is the floor with the inequality reversed, and a floor
and a cap together keep the exposure inside a band. We define the cap, then give both
constraints to one optimiser.
=#

struct MomentumCap{T1, T2} <: PortfolioOptimisers.CustomJuMPConstraint
    score::T1
    cap::T2
end
function PortfolioOptimisers.add_custom_constraint!(model::JuMP.Model, ccnt::MomentumCap,
                                                    optimiser, attrs)
    w = PortfolioOptimisers.get_w(model)
    k = PortfolioOptimisers.get_k(model)
    sc = PortfolioOptimisers.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.cap * k - ccnt.score' * w) >= 0)
    return nothing
end

band = optimise(MeanRisk(; obj = MinimumRisk(),
                         opt = JuMPOptimiser(; pe = pr, slv = slv,
                                             ccnt = [MomentumFloor(score, 0.5),
                                                     MomentumCap(score, 0.8)])))

#=
Second, two objective terms. Each term of a `cobj` vector adds to the same objective penalty, so
two tilts of `1e-4` act as one tilt of `2e-4`. We solve both, and print the three exposures.
=#

two_tilts = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  cobj = [MomentumTilt(score, 1e-4),
                                                          MomentumTilt(score, 1e-4)])))
one_double = optimise(MeanRisk(; obj = MinimumRisk(),
                               opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                   cobj = MomentumTilt(score, 2e-4))))

composition = (band = score' * band.w, two_1e4_tilts = score' * two_tilts.w,
               one_2e4_tilt = score' * one_double.w)

#=
Compare the exposure of the band portfolio with `[0.5, 0.8]`. Then compare the exposure of the
two tilts of `1e-4` with that of the one tilt of `2e-4`.

Last, an objective term and a constraint together. We give a tilt to `cobj` and a floor to
`ccnt` of one optimiser.
=#

res_both = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                 cobj = MomentumTilt(score, 1e-4),
                                                 ccnt = MomentumFloor(score, 1.0))))

#=
## 7. Comparing the effect

We print and plot five portfolios. Only the custom terms differ between them. The tilt raises
the momentum exposure as far as its price against risk allows. The floor and the band hold the
exposure at a bound. The last portfolio has both a tilt and a floor.
=#

results = [res_base, tilt_res[2], floor_res[3], band, res_both]
labels = ["Base", "Tilt λ=1e-4", "Floor 1.0", "Band [0.5,0.8]", "Tilt + floor"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Weights under each custom term")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive (4_constraints_costs/09), lifted and expanded from user_guide §5 (which is
#src   now trimmed to a pointer). Real SP500 slice, verified on kaimon (session e9b04ccf).
#src - Objective hook: MomentumTilt cobj under MinimumRisk, λ sweep 0/1e-4/5e-4/2e-3 →
#src   momentum exposure 0.204/1.295/1.396/1.399 (saturates; maxw 37%→69%→83%→89%).
#src   MaximumUtility base 1.156 → tilt(5e-3) 1.403.
#src - ADR 0036 REWROTE this section. Custom terms now go through the objective PENALTY
#src   (add_to_objective_penalty!) instead of mutating obj_expr, so the library applies the
#src   sense-correct factor and reward_sign is DELETED — one definition (a negative penalty)
#src   rewards under Min and Max alike.
#src   VALUE-PRESERVING, re-verified against the new code path (2026-07-20): the λ sweep still
#src   gives 0.204/1.295/1.396/1.399 (maxw 37.0/69.3/83.0/88.8) and MaximumUtility still gives
#src   1.156 → 1.403. The MaximumUtility pair is the load-bearing check: it is the SENSE-FLIP
#src   case, where the old code ADDED the term (reward_sign = +1) and the new code contributes
#src   a NEGATIVE penalty folded in with factor -1. Same answer ⇒ the two routes agree across
#src   the sense boundary, not just on the Min path where the arithmetic is trivially equal.
#src - MaximumRatio is NO LONGER a special case: both branches of the CC transform (Max form
#src   and risk-Min form) fold the penalty in with their own factor, so a tilt is correct
#src   either way. The old "MethodError by design" carve-out is gone. The k idiom below is
#src   UNAFFECTED — ADR 0036 fixes the sign trap, not the rescaling one.
#src - Constraint hook: MomentumFloor ccnt binds EXACTLY (floor 0.5→0.500, 1.0→1.000, 1.35→1.350;
#src   floor 0.0 non-binding at base 0.204). Idiom: scale by get_constraint_scale, multiply the
#src   constant bound by get_k.
#src - k idiom PAYS OFF under MaximumRatio (k is a free var). MomentumFloorNoK (forgot * k):
#src   requested floor 1.30/1.35/1.38 → correct(with k) 1.300/1.350/1.380, wrong(no k)
#src   1.369/1.369/1.370 — at 1.38 the k-less floor lands BELOW the mandate. Under MinimumRisk
#src   (k=1) both agree, which is why the bug hides. (floor 1.5 infeasible under ratio → NaN.)
#src - VECTOR SUPPORT was BROKEN before this branch: ccnt-vector silently no-op'd (hit the
#src   args... fallback), cobj-vector MethodError'd (set_portfolio_objective_function! typed
#src   cobj::Option{<:CustomJuMPObjective}). Fixed in src (uncommitted, this branch): added
#src   VecJuMPConstr/VecJuMPObj iteration methods + widened the 5 objective-gate signatures to
#src   Option{<:JuMPObj_VecJuMPObj}. Now band [0.5,0.8] binds at 0.500 (was 0.204); two 1e-4
#src   tilts == one 2e-4 tilt (1.3695). test_03b assembly suite still green.
#src - BOTH follow-ups now CLOSED (uncommitted on dev):
#src   1. NearOptimalCentering called add_custom_objective_term! with a DIFFERENT arg order
#src      than the main path, so a custom objective missed dispatch there. Fixed by inserting
#src      MinimumRisk() as `obj` (NOC minimises: @objective(…, Min) + penalty factor 1).
#src      test_20 NOC suite 39/39 green after.
#src   2. New test/test_03c_custom_jump_hooks.jl covers both hooks: single + vector cobj/ccnt,
#src      the k idiom under MaximumRatio, the fallback, and a spy pinning the calling
#src      convention on BOTH the main and NOC paths. Mutation-checked.
#src      Gotcha: `!isempty(spy_log)` is NOT a valid NOC regression detector — NOC solves
#src      MeanRisk sub-problems that reach the hook through the main path and fire the spy 3×
#src      regardless. The NOC builder's own call is the one under test.
#src - ADR 0036 changed what that spy keys on. It USED to key on the NOC call being the only
#src   one passing a JuMPOptimiser (not an optimiser estimator) in the `opt` slot — i.e. on a
#src   type inconsistency that WAS ITSELF THE BUG: a user hook dispatching on `opt`, or
#src   reaching `opt.opt`, worked everywhere except under NOC. NOC now passes `noc` like every
#src   other caller, so the detector keys on `isa(e.optimiser, NearOptimalCentering)` — naming
#src   the estimator under test instead of detecting a type confusion. If you ever reintroduce
#src   a JuMPOptimiser in that slot, this test will NOT catch it; the typed fallback will.
#src - Also new in test_03c: a quadratic term (SpreadPenalty) against MaximumReturn's AFFINE
#src   objective — impossible before ADR 0036 (add_to_expression!(::AffExpr, ::QuadExpr) is a
#src   MethodError), so custom-term validity used to depend on the risk measure and objective.
#src   And an Inert* pair asserting the fallback now RAISES rather than silently no-op'ing.
#src   Also: constrained NOC does NOT solve on this SP500 slice (NaN) with Variance OR
#src   StandardDeviation — test_20 uses a curated dataset + multi-solver list. Hence the NOC
#src   test is deliberately solver-free (asserts on the assembly-time call, not the solution).
#src - plot_stacked_bar_composition lives in the Plots EXTENSION, whose trigger is BOTH
#src   GraphRecipes AND StatsPlots (Project.toml [extensions]) — loading StatsPlots alone leaves
#src   the method undefined (MethodError). Import both, matching the other examples. StatsPlots
#src   also re-exports mean/std, so no separate `using Statistics`.
#src - Cold include verified end-to-end on kaimon: every solve + pretty_table runs; final plot
#src   returns a Plots.Plot. Struct redefinition needs a REPL restart between runs (top-level
#src   structs aren't Revise-reloadable).
