#=
# Exotic tail risk measures: beyond CVaR

[`ConditionalValueatRisk`](@ref) (CVaR) is the workhorse coherent tail measure — the expected
loss in the worst ``\alpha`` fraction of outcomes. But it averages over the tail, so two
distributions with the same tail *average* but very different tail *shapes* look identical to
it. The library ships a family of coherent tail measures that weight the extreme tail more
aggressively than CVaR:

  - [`EntropicValueatRisk`](@ref) (EVaR) — the tightest coherent upper bound on Value-at-Risk,
    built from the exponential moment-generating function. It is more conservative than CVaR
    and is solved over the **exponential cone**.
  - [`RelativisticValueatRisk`](@ref) (RLVaR) — a coherent generalisation of EVaR via the
    Tsallis (``\kappa``-deformed) entropy, parametrised by ``\kappa \in (0, 1)``. It
    *interpolates* between EVaR (as ``\kappa \to 0``) and the worst realisation (as
    ``\kappa \to 1``), giving a continuous dial on how hard the extreme tail is penalised.
    Solved over the **power cone**.
  - [`PowerNormValueatRisk`](@ref) (PNVaR) — generalises EVaR by replacing the
    moment-generating function with a power-norm, parametrised by a power ``p \ge 1``. Also
    solved over the **power cone**, and likewise approaches the worst realisation as ``p``
    grows.
  - [`GenericValueatRiskRange`](@ref) — composes *any* two of these measures into a two-sided
    range: one measure on the loss side, another on the gain side.

These measures sit in a conservativeness ladder, ``\mathrm{CVaR} \le \mathrm{EVaR} \le
\mathrm{RLVaR}``, and all share the confidence level ``\alpha`` with CVaR.

!!! tip "When to reach for this"
    Reach for these when CVaR does not punish the *extreme* tail enough — when you care about
    the worst few outcomes more than the average of the worst 5%, but want to stay coherent
    and convex rather than jumping to a raw worst-realisation objective. The ``\kappa`` (RLVaR)
    and ``p`` (PNVaR) parameters are the dial: turn them up to move continuously from
    EVaR-like behaviour toward worst-case behaviour.

!!! note "Conic solver, not a special one"
    EVaR needs the exponential cone; RLVaR and PNVaR need the power cone. **Clarabel supports
    both**, so unlike `VarianceSkewKurtosis` (which needs SCS for PSD cones) these need no
    special solver — the optimiser's Clarabel handles them directly. The measures also accept
    an `slv` field, which is only needed when you evaluate them standalone with
    [`expected_risk`](@ref) (as we do below), not when they are the objective of a `MeanRisk`.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, Statistics

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data and shared setup
=#

using CSV, TimeSeries, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true))]

opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
## 2. Minimising each tail measure

Each measure drops into [`MeanRisk`](@ref) with no extra wiring; the optimiser's Clarabel
solves the exponential/power cones. We use the default confidence level (`alpha = 0.05`),
the default `kappa = 0.3` for RLVaR, and the default `p = 2.0` for PNVaR.
=#

measures = ["CVaR" => ConditionalValueatRisk(), "EVaR" => EntropicValueatRisk(),
            "RLVaR" => RelativisticValueatRisk(), "PNVaR" => PowerNormValueatRisk()]

results = [optimise(MeanRisk(; r = r, opt = opt)) for (_, r) in measures]
names_r = first.(measures)

pretty_table(DataFrame(hcat(rd.nx, [r.w for r in results]...),
                       [:assets; Symbol.(names_r)...]); formatters = [resfmt])

#=
The allocations differ: the more conservative measures (EVaR, RLVaR) push harder into the
names that protect against the *worst* days, not just the worst-5%-on-average days, so they
concentrate differently from CVaR.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition(results, rd)

#=
## 3. How different is each measure from CVaR?

The sharp way to see that these are genuinely different objectives is to **cross-evaluate**:
take each minimum-risk portfolio and measure its realised risk under *every* measure. The
diagonal should be the smallest entry in its column (each portfolio is best at minimising its
own measure), and within any row the values should climb CVaR → EVaR → RLVaR → PNVaR — the
conservativeness ladder. We pass `slv` here because we are evaluating the measures standalone.
=#

evals = ["CVaR" => ConditionalValueatRisk(), "EVaR" => EntropicValueatRisk(; slv = slv),
         "RLVaR" => RelativisticValueatRisk(; slv = slv),
         "PNVaR" => PowerNormValueatRisk(; slv = slv)]

cross = DataFrame(; minimises = names_r)
for (mname, m) in evals
    cross[!, Symbol(mname)] = [expected_risk(m, r.w, rd.X) for r in results]
end
pretty_table(cross; formatters = [resfmt])

#=
Reading the table: each portfolio attains the lowest value of the measure it was built to
minimise (the diagonal), confirming the four measures are not interchangeable. And every row
increases left to right — for the same portfolio, EVaR is larger than CVaR and RLVaR larger
still, because each measure puts more weight on the most extreme losses.

## 4. RLVaR as a dial: from EVaR to the worst realisation

The most useful intuition for [`RelativisticValueatRisk`](@ref) is that ``\kappa`` slides it
continuously between two familiar measures. We hold one portfolio fixed (the CVaR-minimising
one) and evaluate RLVaR across ``\kappa``, alongside EVaR and [`WorstRealisation`](@ref) as
the two limits.
=#

w_fixed = results[1].w   ## CVaR-minimising portfolio
kappas = [0.01, 0.1, 0.3, 0.6, 0.99]
rlvar_curve = [expected_risk(RelativisticValueatRisk(; slv = slv, kappa = k), w_fixed,
                             rd.X) for k in kappas]
evar_ref = expected_risk(EntropicValueatRisk(; slv = slv), w_fixed, rd.X)
wr_ref = expected_risk(WorstRealisation(), w_fixed, rd.X)

pretty_table(DataFrame(; :kappa => kappas, :RLVaR => rlvar_curve); formatters = [resfmt])

#=
As ``\kappa \to 0`` the RLVaR matches EVaR, and as ``\kappa \to 1`` it matches the worst
realisation — the curve climbs monotonically between the two limits. That is the whole point
of RLVaR: a single coherent measure that you can tune from "tight upper bound on VaR" all the
way to "the single worst day", without ever leaving the convex world.
=#

plot(kappas, rlvar_curve; seriestype = :path, marker = (:circle, 5), label = "RLVaR(κ)",
     xlabel = "κ", ylabel = "Realised tail risk", legend = :topleft,
     title = "RLVaR interpolates EVaR (κ→0) and worst realisation (κ→1)")
hline!([evar_ref]; label = "EVaR", linestyle = :dash)
hline!([wr_ref]; label = "Worst realisation", linestyle = :dot)

#=
## 5. PowerNorm Value-at-Risk: the `p` dial

[`PowerNormValueatRisk`](@ref) plays a similar game through its power ``p \ge 1``: larger `p`
pushes the measure toward the worst realisation. We evaluate it on the same fixed portfolio.
=#

ps = [2.0, 4.0, 10.0]
pnvar_curve = [expected_risk(PowerNormValueatRisk(; slv = slv, p = p), w_fixed, rd.X)
               for p in ps]
pretty_table(DataFrame(; :p => ps, :PNVaR => pnvar_curve); formatters = [resfmt])

#=
PNVaR also climbs toward the worst realisation as `p` grows. We start the sweep at `p = 2`
deliberately: although the constructor permits `p = 1`, the power-cone formulation degenerates
at that boundary and the solver stalls (see the findings note in the source).

## 6. Two-sided control with `GenericValueatRiskRange`

[`GenericValueatRiskRange`](@ref) composes any two `XatRisk` measures into a range: a
loss-side measure on the returns plus a gain-side measure on the negated returns. This lets
you treat downside and upside *asymmetrically* — for example, an aggressive EVaR on the loss
side (punish bad tails hard) with a milder CVaR on the gain side. We compare it against the
symmetric [`ConditionalValueatRiskRange`](@ref).
=#

r_asym = GenericValueatRiskRange(; loss = EntropicValueatRisk(),
                                 gain = ConditionalValueatRisk())
res_asym = optimise(MeanRisk(; r = r_asym, opt = opt))
res_sym = optimise(MeanRisk(; r = ConditionalValueatRiskRange(), opt = opt))

pretty_table(DataFrame(; :assets => rd.nx, :EVaR_loss_CVaR_gain => res_asym.w,
                       :CVaR_range => res_sym.w); formatters = [resfmt])

#=
The asymmetric range tilts the portfolio toward names whose *downside* tail is well behaved,
while the symmetric CVaR range treats both sides with the same measure.

A final composition plot of the four single-sided measures from section 2 makes the family's
differences concrete: each coherent tail measure produces a recognisably different allocation.
=#

plot_stacked_bar_composition(results, rd; xticks = ([1, 2, 3, 4], names_r))

#=
## Summary

Beyond CVaR, the library offers a ladder of coherent tail measures that weight the extreme
tail progressively harder:

  - [`EntropicValueatRisk`](@ref) is the tight coherent upper bound on VaR (exponential cone).
  - [`RelativisticValueatRisk`](@ref) generalises EVaR and dials continuously from EVaR
    (``\kappa \to 0``) to the worst realisation (``\kappa \to 1``) via the power cone.
  - [`PowerNormValueatRisk`](@ref) offers the same kind of dial through ``p \ge 1``.
  - [`GenericValueatRiskRange`](@ref) composes any two of them into an asymmetric two-sided
    measure.

All are convex and solved by Clarabel's exponential/power cones — no special solver needed.
Reach for them when the *shape* of the extreme tail matters and CVaR's tail-averaging is too
blunt.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): CVaR/EVaR/RLVaR/PNVaR minimum-risk solves,
#src   the cross-evaluation matrix, the RLVaR kappa-sweep, the PNVaR p-sweep, and the
#src   `GenericValueatRiskRange` (EVaR-loss / CVaR-gain) vs `ConditionalValueatRiskRange` solve
#src   all succeed with Clarabel on the 252-obs / 20-asset slice. No special solver needed —
#src   exponential and power cones are handled directly (contrast with VarianceSkewKurtosis/SCS).
#src - VERIFIED the headline relationships numerically before writing: the cross-eval diagonal is
#src   each column's minimum, every row climbs CVaR<EVaR<RLVaR<PNVaR, and on the fixed CVaR
#src   portfolio RLVaR(kappa=0.01)=0.02062 == EVaR(0.02062) while RLVaR(kappa=0.99)=0.02479 ==
#src   WorstRealisation(0.02479) — a clean EVaR-to-worst-realisation interpolation.
#src - FINDING (record-only → new tail-risk rollup issue): `PowerNormValueatRisk(; p = 1.0)` is
#src   accepted by the constructor (`@argcheck p >= 1`) but `expected_risk` returns `NaN` — all
#src   Clarabel configs hit SLOW_PROGRESS / INSUFFICIENT_PROGRESS at the p=1 boundary (the power
#src   cone degenerates). Either tighten the validation to `p > 1`, or document that p=1 is a
#src   degenerate boundary. The example sweeps p in {2,4,10} to keep the rendered output clean.
#src - The `slv`-only-for-standalone-evaluation contract is a mild ergonomics trap: the same
#src   measure needs `slv` for `expected_risk` but not when it is a `MeanRisk` objective. Noted
#src   explicitly in the opening admonition so a reader who copies a measure between the two
#src   contexts is not surprised by a missing-solver error.
