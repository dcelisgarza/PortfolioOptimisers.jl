#=
# Custom objectives and constraints

Every keyword on a [`JuMPOptimiser`](@ref) — bounds, budgets, turnover, fees, cardinality —
is a *pre-built* way to shape the problem. When a mandate needs something none of them covers,
`PortfolioOptimisers.jl` gives you two extension points that write **straight against the JuMP
model**:

  - [`CustomJuMPObjective`](@ref) (the `cobj` keyword) — implement
    [`add_custom_objective_term!`](@ref) to add a reward or penalty term to the objective.
  - [`CustomJuMPConstraint`](@ref) (the `ccnt` keyword) — implement
    [`add_custom_constraint!`](@ref) to add a constraint to the model.

Each keyword takes a single estimator *or a vector of them*, and each hook **dispatches on the
estimator's type**, so a term is just a struct carrying its data plus one method. This page
builds both from scratch, works through the two model idioms that keep them correct
(the constraint scale and the homogenisation variable `k`), and composes several into one
problem. It is the deep dive behind the one-call summary in the
[constraints & costs guide](../../user_guide/03_Constraints_and_Costs.md).

!!! tip "When to reach for this"
    Reach for a custom term when your preference is a **continuous per-asset number** that no
    group string can express — a factor score, a carbon intensity, a liquidity penalty — or a
    relationship between weights that isn't a plain linear bound. If it *can* be written as a
    linear/group constraint (`lcse`) or an existing keyword, prefer that: the built-ins are
    tested and composable. Custom hooks are the escape hatch, not the first tool.
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

We fix one empirical prior, a solver, and a minimum-risk baseline, so every custom term's
effect is visible against the same allocation. The preference we will encode is a **momentum
score**: each asset's trailing-63-day return, standardised across the universe. It is a
continuous per-asset number — exactly the case the extension points exist for.
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
The portfolio's **momentum exposure** is `score' * w` — the score-weighted allocation. The
minimum-risk baseline does not care about momentum, so it lands wherever the risk trade-off
puts it; we will push that exposure up, first softly (an objective), then hard (a constraint).
=#

base_exposure = score' * res_base.w

#=
## 2. A custom objective — a soft tilt

A custom objective is a struct subtyping [`CustomJuMPObjective`](@ref), carrying whatever data
the term needs, plus one method of [`add_custom_objective_term!`](@ref). The method is handed
the model *mid-assembly* and mutates the objective expression in place. Its full signature is

```julia
add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)
```

  - `model` — the [`JuMP`](https://jump.dev/) model under construction.
  - `obj` — the [`ObjectiveFunction`](@ref) (`MinimumRisk`, `MaximumUtility`, …). **Dispatch on
    this** to get the optimisation *sense* right (below).
  - `pret` — the [`JuMPReturnsEstimator`](@ref) (`ArithmeticReturn`/`LogarithmicReturn`).
  - `cobj` — your estimator; the argument you dispatch your method on.
  - `obj_expr` — the objective expression accumulated so far. Add your term into it with
    `JuMP.add_to_expression!`.
  - `opt` — the optimiser estimator (e.g. the [`MeanRisk`](@ref) itself).
  - `pr` — the [`AbstractPriorResult`](@ref); `pr.mu`, `pr.sigma`, … are available if your term
    is data-driven rather than carrying its own numbers.

Read the weight variables with the [`get_w`](@ref) accessor rather than reaching into
`model[:w]` — it asserts the variables have been registered and fails with a clear message if a
hook runs out of order.

!!! warning "The sign follows the optimisation *sense*, not the term"
    `MinimumRisk` **minimises** `obj_expr`, so to *reward* high momentum you **subtract** it.
    `MaximumUtility`/`MaximumReturn` **maximise**, so you **add** it. Get this backwards and you
    will faithfully optimise *against* your own preference. We encode the rule once as
    `reward_sign(obj)` and dispatch it on the objective type.
=#

struct MomentumTilt{T1, T2} <: PortfolioOptimisers.CustomJuMPObjective
    score::T1
    lambda::T2
end

## A reward term improves a minimisation when subtracted, a maximisation when added.
reward_sign(::MinimumRisk) = -1
reward_sign(::Union{MaximumUtility, MaximumReturn}) = 1

function PortfolioOptimisers.add_custom_objective_term!(model::JuMP.Model, obj, pret,
                                                        cobj::MomentumTilt, obj_expr, opt,
                                                        pr, args...)
    w = PortfolioOptimisers.get_w(model)
    JuMP.add_to_expression!(obj_expr, reward_sign(obj) * cobj.lambda * (cobj.score' * w))
    return nothing
end

#=
`lambda` is the price we put on momentum relative to risk. Sweeping it traces the *soft*
trade-off: a small `lambda` barely moves the book, a large one lets momentum dominate — until
the term saturates and the portfolio piles into the single highest-momentum name.
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
Note the term is **homogeneous of degree one in `w`** (it scales with the weights, just like the
return and risk expressions). That is what lets it stay consistent under a ratio objective's
internal rescaling — the constraint side, next, is where that rescaling needs explicit care.

## 3. The sense in action

The same tilt under [`MaximumUtility`](@ref) — a *maximisation* — must **add** the reward, which
`reward_sign` handles. It lifts that objective's momentum exposure the same way.
=#

util_base = optimise(MeanRisk(; obj = MaximumUtility(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv)))
util_tilt = optimise(MeanRisk(; obj = MaximumUtility(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  cobj = MomentumTilt(score, 5e-3))))

util_exposures = (base = score' * util_base.w, tilted = score' * util_tilt.w)

#=
!!! warning "[`MaximumRatio`](@ref) is deliberately left out of `reward_sign`"
    The maximum-ratio problem is solved through a homogenising transform that, depending on the
    risk measure, lands the objective in **either** a maximisation **or** a risk-minimisation
    form. A single fixed sign would be wrong half the time, so `reward_sign` has no
    `MaximumRatio` method: a tilt against a ratio objective raises a clear `MethodError` rather
    than silently optimising the wrong way. If you need it, add the method deliberately for the
    exact configuration you use, and verify the resulting exposure moves in the intended
    direction.

## 4. A custom constraint — a hard floor

A custom constraint is the same shape: a struct subtyping [`CustomJuMPConstraint`](@ref) plus
one method of [`add_custom_constraint!`](@ref), whose signature is

```julia
add_custom_constraint!(model, ccnt, opt, attrs)
```

  - `model`, `ccnt`, `opt` — as before (`ccnt` is what you dispatch on).
  - `attrs` — the [`ProcessedJuMPOptimiserAttributes`](@ref) bundle: `attrs.pr` (prior),
    `attrs.wb` (bounds), and the rest of the processed problem data, if your constraint needs it.

Two model idioms keep a hand-written constraint correct (see ADR 0008, *JuMP model assembly*):

 1. **Scale the constraint** by [`get_constraint_scale`](@ref) (`model[:sc]`), so it sits on the
    same numerical footing as every built-in constraint.
 2. **Multiply any constant bound by [`get_k`](@ref)** (`model[:k]`), the homogenisation
    variable. For most objectives `k == 1` and this is a no-op; under a ratio objective the
    weights are solved in a rescaled space (`w_real = w / k`), and a bare constant would be
    compared against the *rescaled* weights — the wrong thing. §5 shows exactly what breaks.
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

#=
Unlike the soft tilt, a floor binds *exactly*: the optimiser buys just enough momentum to meet
it and no more, spending the rest of its freedom on risk. Sweeping the floor shows it clamping
the exposure to the requested level (a floor below the baseline `0.204` simply never binds).
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
## 5. Why the `k` idiom matters

To see idiom (2) pay off, here is the *same* floor written **without** the `* k` — the mistake
most people make first, because it is invisible under `MinimumRisk` (where `k == 1`).
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
Under [`MaximumRatio`](@ref) the homogenisation variable `k` is a genuine free variable, so the
two versions diverge. The correct floor binds the *recovered* exposure exactly at the bound; the
`k`-less one binds it at some fixed level of the rescaled weights — which, at `floor = 1.38`,
lands **below** the requested floor, silently breaking the mandate.
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

Both keywords accept a **vector** of estimators, applied in order — the hooks iterate and
dispatch each element. That is how you stack custom terms without folding them into one struct.

**A band from two constraints.** Pair the floor with its mirror image — a cap, the same idiom
with the inequality flipped — and the two together bound the exposure into a corridor. Define
the cap first (as always, every custom type before the optimiser that uses it):
=#

struct MomentumCap{T1, T2} <: PortfolioOptimisers.CustomJuMPConstraint
    score::T1
    cap::T2
end
function PortfolioOptimisers.add_custom_constraint!(model::JuMP.Model, ccnt::MomentumCap,
                                                    opt, attrs)
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
**Additive objectives.** A `cobj` vector adds each term into the same expression, so two
`1e-4` tilts compose into one of strength `2e-4` — a quick sanity check that vectors accumulate
rather than replace:
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
The band sits inside `[0.5, 0.8]`, and the two stacked tilts land on exactly the same exposure
as the single double-strength tilt — vectors compose.

**Objective and constraint together.** Nothing stops you mixing them: a soft tilt *and* a hard
floor in the same problem, each through its own keyword.
=#

res_both = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                 cobj = MomentumTilt(score, 1e-4),
                                                 ccnt = MomentumFloor(score, 1.0))))

#=
## 7. Comparing the effect

Same prior, same minimum-risk objective — only the custom term changes the allocation. The soft
tilt leans toward momentum as far as the risk trade-off allows; the hard floor and the band pin
the exposure to a level; combining them does both.
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
#src   momentum exposure 0.204/1.295/1.396/1.399 (saturates; maxw 37%→69%→83%→89%). Sign follows
#src   the optimisation SENSE not the term: MinimumRisk subtracts (Min), MaximumUtility adds
#src   (Max) — reward_sign dispatches. MaximumUtility base 1.156 → tilt(5e-3) 1.403.
#src - MaximumRatio deliberately has NO reward_sign method: the CC transform lands the objective
#src   in a Max OR a risk-Min form depending on the risk measure, so a fixed sign is wrong half
#src   the time. Tilt against a ratio objective → MethodError by design (verified).
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
#src      (model, ret, cobj, obj_expr, opt, pe) than the main path (model, obj, pret, cobj,
#src      obj_expr, opt, pr), so a custom objective missed dispatch there. Fixed by inserting
#src      MinimumRisk() as `obj` (NOC minimises: @objective(…, Min) + penalty factor 1).
#src      test_20 NOC suite 39/39 green after.
#src   2. New test/test_03c_custom_jump_hooks.jl (19 tests) covers both hooks: single + vector
#src      cobj/ccnt, the k idiom under MaximumRatio, the no-op fallback, and a spy that pins
#src      the (model, obj, pret, cobj, obj_expr, opt, pr) calling convention on BOTH the main
#src      and NOC paths. Mutation-checked: reverting the NOC fix makes it fail.
#src      Gotcha: `!isempty(spy_log)` is NOT a valid NOC regression detector — NOC solves
#src      MeanRisk sub-problems that reach the hook through the main path and fire the spy 3×
#src      regardless. The NOC builder's own call is the only one passing a JuMPOptimiser (not
#src      an optimiser estimator) in the `opt` slot, so that's what the test keys on.
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
