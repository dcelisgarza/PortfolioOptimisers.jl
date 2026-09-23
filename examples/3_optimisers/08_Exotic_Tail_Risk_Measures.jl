#=
```@meta
Description = "Tail risk measures beyond CVaR in PortfolioOptimisers.jl: EVaR, RLVaR, power-norm VaR and the VaR ranges that weight the extreme tail harder."
```

# Tail risk measures that weight the worst losses more than CVaR

[`ConditionalValueatRisk`](@ref), CVaR, is the most common coherent tail measure. It is the
expected loss over the worst fraction ``\alpha`` of outcomes. Because it averages over the tail,
two distributions with the same tail *average* and very different tail *shapes* have the same
CVaR. The library has a family of coherent tail measures that weight the largest losses more
than CVaR does.

  - [`EntropicValueatRisk`](@ref), EVaR, is the tightest upper bound on VaR and CVaR that the
    Chernoff inequality gives, built from the moment-generating function of the losses. It is
    at least as large as CVaR, and the solver works over the exponential cone.
  - [`RelativisticValueatRisk`](@ref), RLVaR, generalises EVaR with the Kaniadakis entropy,
    which has a deformation parameter ``\kappa \in (0, 1)``. It moves from EVaR as
    ``\kappa \to 0`` to the worst realisation as ``\kappa \to 1``, so ``\kappa`` sets how much
    the largest losses count. The solver works over the power cone.
  - [`PowerNormValueatRisk`](@ref), PNVaR, generalises EVaR by replacing the
    moment-generating function with a power norm of power ``p \ge 1``. The solver also works
    over the power cone, and the measure also moves toward the worst realisation as ``p``
    grows. On a finite sample it reaches the worst realisation at a finite ``p``.
  - [`GenericValueatRiskRange`](@ref) combines *any* two of these measures into a range, one
    measure on the losses and another on the gains.

The measures are ordered, ``\mathrm{CVaR} \le \mathrm{EVaR} \le \mathrm{RLVaR}``, and each
takes the tail fraction ``\alpha`` as CVaR does.

!!! tip "When to reach for this"
    Reach for these measures when CVaR does not penalise the *extreme* tail enough. They suit
    you when the few worst outcomes matter more than the mean of the worst 5 %, and you want a
    measure that still counts more than the single worst day. Raise ``\kappa`` for RLVaR to
    move from EVaR toward the worst case, or ``p`` for PNVaR to move from CVaR toward it.

!!! note "A conic solver is enough"
    EVaR needs the exponential cone, and RLVaR and PNVaR need the power cone. Clarabel
    supports both, so the Clarabel solver in `opt` handles both cones, and no other solver
    is needed. The [previous page](06_Brownian_Distance_Variance_and_VarianceSkewKurtosis.md)
    solves `VarianceSkewKurtosis` with SCS, because it builds a large semidefinite problem.
    These measures also take an `slv` field. You need it only when you evaluate a measure alone with [`expected_risk`](@ref), as we do below, and not when the
    measure is the objective of a `MeanRisk`.
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

We minimise each measure with [`MeanRisk`](@ref). We use the defaults `alpha = 0.05` for every
measure and `kappa = 0.3` for RLVaR. For PNVaR we use `p = 1.5`, not the default `p = 2.0`.
Section 5 explains why, on 252 observations, PNVaR at `p = 2.0` is the worst realisation.
=#

measures = ["CVaR" => ConditionalValueatRisk(), "EVaR" => EntropicValueatRisk(),
            "RLVaR" => RelativisticValueatRisk(),
            "PNVaR" => PowerNormValueatRisk(; p = 1.5)]

results = [optimise(MeanRisk(; r = r, opt = opt)) for (_, r) in measures]
names_r = first.(measures)

pretty_table(DataFrame(hcat(rd.nx, [r.w for r in results]...),
                       [:assets; Symbol.(names_r)...]); formatters = [resfmt])

#=
The portfolios differ. CVaR is the mean loss over the worst 5 % of days. EVaR and RLVaR also
put more weight on the *worst* of those days.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition(results, rd)

#=
## 3. How different is each measure from CVaR?

We evaluate each minimum-risk portfolio under *every* measure. Each row of the table is one
portfolio, and each column one measure. We pass `slv` here, because we evaluate the measures
alone.
=#

evals = ["CVaR" => ConditionalValueatRisk(), "EVaR" => EntropicValueatRisk(; slv = slv),
         "RLVaR" => RelativisticValueatRisk(; slv = slv),
         "PNVaR" => PowerNormValueatRisk(; slv = slv, p = 1.5)]

cross = DataFrame(; minimises = names_r)
for (mname, m) in evals
    cross[!, Symbol(mname)] = [expected_risk(m, r.w, rd.X) for r in results]
end
pretty_table(cross; formatters = [resfmt])

#=
In each column, the smallest value is on the diagonal, where the portfolio minimised that
measure. The portfolio that is best under one measure is not the best under the others. Along
each row CVaR ≤ EVaR ≤ RLVaR, because each puts more weight on the largest losses than the one
before it.

## 4. The parameter ``\kappa`` of RLVaR, from EVaR to the worst realisation

``\kappa`` moves [`RelativisticValueatRisk`](@ref) between two measures you know. We take one
portfolio, the one that minimises CVaR, and evaluate RLVaR on it for five values of ``\kappa``.
We also evaluate EVaR and [`WorstRealisation`](@ref), the two limits.
=#

w_fixed = results[1].w
kappas = [0.01, 0.1, 0.3, 0.6, 0.99]
rlvar_curve = [expected_risk(RelativisticValueatRisk(; slv = slv, kappa = k), w_fixed,
                             rd.X) for k in kappas]
evar_ref = expected_risk(EntropicValueatRisk(; slv = slv), w_fixed, rd.X)
wr_ref = expected_risk(WorstRealisation(), w_fixed, rd.X)

pretty_table(DataFrame(; :kappa => kappas, :RLVaR => rlvar_curve); formatters = [resfmt])

#=
RLVaR rises with ``\kappa``. We plot the curve with EVaR and the worst realisation as
horizontal lines. The plot shows RLVaR near EVaR at ``\kappa = 0.01`` and near the worst
realisation at ``\kappa = 0.99``.
=#

plot(kappas, rlvar_curve; seriestype = :path, marker = (:circle, 5), label = "RLVaR(κ)",
     xlabel = "κ", ylabel = "Realised tail risk", legend = :topleft,
     title = "RLVaR of the minimum-CVaR portfolio for five values of κ")
hline!([evar_ref]; label = "EVaR", linestyle = :dash)
hline!([wr_ref]; label = "Worst realisation", linestyle = :dot)

#=
## 5. The power `p` of the power-norm VaR

[`PowerNormValueatRisk`](@ref) has a power ``p \ge 1`` with a similar effect. A larger `p`
moves the measure toward the worst realisation. We evaluate it on the same portfolio, beside the
worst realisation `wr_ref` of section 4.
=#

ps = [1.25, 1.5, 1.75, 1.8]
pnvar_curve = [expected_risk(PowerNormValueatRisk(; slv = slv, p = p), w_fixed, rd.X)
               for p in ps]
pretty_table(DataFrame(; :p => ps, :PNVaR => pnvar_curve, :worst_realisation => wr_ref);
             formatters = [resfmt])

#=
PNVaR rises with `p`. Unlike RLVaR, it reaches the worst realisation at a finite `p`. With
``T`` equally weighted observations, PNVaR equals the worst realisation when
``\alpha T^{1/p} \le 1``, that is when ``p \ge \log T / \log(1/\alpha)``. For our 252
observations at ``\alpha = 0.05`` this is ``p \ge 1.85``, so the sweep stops at `p = 1.8`. At
the default `p = 2` the measure is the worst realisation. Raise `alpha` or use more
observations to keep a larger `p` below that point.

We start above `p = 1` on purpose. The constructor accepts `p = 1`, but the power cone
degenerates at that value, and the solver stops without a solution.

## 6. Both tails with `GenericValueatRiskRange`

[`GenericValueatRiskRange`](@ref) combines any two of these tail measures into a range. One measure
applies to the returns, for the losses, and the other to the negated returns, for the gains. This
lets you treat the two tails *differently*, for example with EVaR on the losses, which penalises
large losses more, and CVaR on the gains. We compare it with the symmetric
[`ConditionalValueatRiskRange`](@ref).
=#

r_asym = GenericValueatRiskRange(; loss = EntropicValueatRisk(),
                                 gain = ConditionalValueatRisk())
res_asym = optimise(MeanRisk(; r = r_asym, opt = opt))
res_sym = optimise(MeanRisk(; r = ConditionalValueatRiskRange(), opt = opt))

pretty_table(DataFrame(; :assets => rd.nx, :EVaR_loss_CVaR_gain => res_asym.w,
                       :CVaR_range => res_sym.w); formatters = [resfmt])

#=
The asymmetric range weights the largest losses more than the symmetric CVaR range does. Both
ranges measure the gains with CVaR at the same `alpha`, so the two columns differ only by the
measure on the loss side.

We plot the four portfolios of section 2 again, with the name of each measure under its bar.
=#

plot_stacked_bar_composition(results, rd; xticks = ([1, 2, 3, 4], names_r))

#=
## Summary

The library has a family of coherent tail measures that weight the largest losses more than CVaR
does.

  - [`EntropicValueatRisk`](@ref) is the Chernoff upper bound on VaR and CVaR, over the
    exponential cone.
  - [`RelativisticValueatRisk`](@ref) generalises EVaR, and ``\kappa`` moves it from EVaR
    (``\kappa \to 0``) to the worst realisation (``\kappa \to 1``), over the power cone.
  - [`PowerNormValueatRisk`](@ref) moves toward the worst realisation as its power
    ``p \ge 1`` grows, and it equals the worst realisation once ``\alpha T^{1/p} \le 1``.
  - [`GenericValueatRiskRange`](@ref) combines any two of them into one measure of both
    tails, with a different measure on each side.

Every one of them is convex. We minimised all of them with Clarabel on this page.
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
#src   degenerate boundary. The example sweeps p in {1.25,1.5,1.75,1.8} to keep the rendered
#src   output clean.
#src - PNVaR at T = 252, alpha = 0.05 equals WorstRealisation for every p >= log(T)/log(20) = 1.85
#src   (#1282), and the docstring states it. So sections 2 and 3 use p = 1.5, and the
#src   p-sweep stops at 1.8. Measured: PNVaR(w_cvar) = 2.082/2.347/2.472/2.476 % against
#src   WR 2.479 %.
#src - The `slv`-only-for-standalone-evaluation contract is a mild ergonomics trap: the same
#src   measure needs `slv` for `expected_risk` but not when it is a `MeanRisk` objective. Noted
#src   explicitly in the opening admonition so a reader who copies a measure between the two
#src   contexts is not surprised by a missing-solver error.
