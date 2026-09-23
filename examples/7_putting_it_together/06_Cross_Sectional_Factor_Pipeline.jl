#=
```@meta
Description = "The cross-sectional factor model through a Pipeline in PortfolioOptimisers.jl: the same weights as the hand-wired version, with each estimator as a named step."
```

# Cross-sectional factor model through a Pipeline

The [deep dive](05_Cross_Sectional_Factor_Model.md) built a cross-sectional factor model, fitted two
orthogonal uncertainty sets from it, and solved a constrained book, by wiring the estimators into
one [`JuMPOptimiser`](@ref) by hand. This page reaches the same weights through a
[`Pipeline`](@ref), with the same specification.

The panel of point-in-time fields needs no slot of its own in the pipeline. It is part of `rd`, the
returns result, which the pipeline already routes as its `:returns` slot. Three things follow.

  - You declare no new slot. A step that needs the panel takes it from the returns result it
    receives.
  - A step that takes a subset of the assets takes the same subset of the panel, because one view
    of the returns result covers both.
  - A fold takes its subset before the pipeline sees the data, through that same view.

What the pipeline adds is three named steps in place of one call. The middle step, the uncertainty
set, uses the `:prior` slot that the first step wrote. The hand-wired version does the same thing
inside the optimiser, where you cannot see it.
=#

using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, PrettyTables,
      DataFrames, Clarabel

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

#=
## 1. The same panel, and the same specification

We copy the generator and the estimator from the deep dive, with the same seed, so this page solves
the same book. Sections 1 and 2 of [the deep dive](05_Cross_Sectional_Factor_Model.md) say what each
piece is.
=#

function synthetic_panel(; T = 500, N = 80, seed = 661_001)
    rng = StableRNG(seed)
    industries = ["Energy", "Financials", "Health Care", "Technology"]
    Kind = length(industries)
    ind = rand(rng, 1:Kind, N)
    beta = 1.0 .+ 0.35 .* randn(rng, N)
    Ltrue = randn(rng, N, 4)
    alpha = 0.0004 .* randn(rng, N)
    onehot = Float64[ind[i] == k for i in 1:N, k in 1:Kind]
    Btrue = hcat(beta, onehot, Ltrue)
    ftrue = hcat(0.009 .* randn(rng, T), 0.005 .* randn(rng, T, Kind),
                 0.004 .* randn(rng, T, 4))
    ivol = 0.008 .+ 0.012 .* rand(rng, N)
    X = ftrue * transpose(Btrue) + randn(rng, T, N) .* transpose(ivol) .+ transpose(alpha)
    logcap = 20.0 .+ 1.4 .* transpose(view(Ltrue, :, 1)) .+
             0.3 .* cumsum(randn(rng, T, N); dims = 1) ./ sqrt(T)
    mcap = exp.(logcap)
    shares = mcap ./ 50.0
    fields = ["market_cap" => mcap,
              "book_equity" =>
                  mcap .* exp.(-0.7 .+ 0.5 .* transpose(view(Ltrue, :, 2)) .+
                               0.05 .* randn(rng, T, N)),
              "net_income_ttm" =>
                  mcap .* (0.05 .+ 0.02 .* transpose(view(Ltrue, :, 3)) .+
                           0.004 .* randn(rng, T, N)), "adj_shares_outstanding" => shares,
              "adj_volume" =>
                  shares .* exp.(-4.5 .+ 0.6 .* transpose(view(Ltrue, :, 4)) .+
                                 0.15 .* randn(rng, T, N)),
              "signal" => transpose(alpha) .+ 0.0002 .* randn(rng, T, N)]
    listed = [i <= N ÷ 5 ? rand(rng, 20:120) : 1 for i in 1:N]
    amsk = [t >= listed[i] for t in 1:T, i in 1:N]
    for (_, a) in fields
        a[.!amsk] .= NaN
    end
    inputs = [[NumericPanelInput(; name = n, vals = a, alg = ForwardPanelFill(; val = 0.0))
               for (n, a) in fields]
              CategoricalPanelInput(; name = "industry",
                                    vals = repeat(reshape(industries[ind], 1, N), T, 1),
                                    levels = industries)]
    days = filter(d -> Dates.dayofweek(d) <= 5,
                  Date(2015, 1, 1):Day(1):(Date(2015, 1, 1) + Day(2 * T + 10)))[1:T]
    return ReturnsResult(; nx = ["A" * lpad(i, 3, '0') for i in 1:N],
                         X = ifelse.(amsk, X, NaN), ts = days,
                         pnl = asset_panel(inputs; amsk = amsk, emsk = amsk))
end

rd = synthetic_panel()

style(d) = CompositeExposure(; descriptors = [d], family = "style")
factors = ["market" =>
               CompositeExposure(; descriptors = [EWMarketBeta()], outlier = nothing,
                                 scoring = nothing, family = "market"),
           "industry" => OneHotExposure(; field = "industry", family = "industry"),
           "size" => style(LogMarketCap()), "value" => style(BookToPrice()),
           "earnings_yield" => style(EarningsToPrice()),
           "liquidity" => style(EWShareTurnover())]

pe = CrossSectionalFactorPrior(; factors = factors, families = ["industry" => nothing],
                               neutralise = ["style" => "industry"],
                               wa = BlendedInverseVarianceWeights(; lambda = 0.5),
                               rfe = FixedWeightedReturnForecast(;
                                                                 scores = DescriptorScores(;
                                                                                           descriptors = [Passthrough(;
                                                                                                                      field = "signal")],
                                                                                           outlier = nothing,
                                                                                           scoring = nothing),
                                                                 scale = 1.0), lambda = 1.0,
                               c = 1.0)

#=
The panel is part of `rd`, and `rd` is already in the `:returns` slot. No code below names the panel
again.
=#

pretty_table(DataFrame("Carrier" => string(nameof(typeof(rd))),
                       "Panel Fields on rd.pnl" => length(rd.pnl.pf),
                       "Slot it rides" => ":returns"))

#=
## 2. The hand-wired route

This is the book of the deep dive's fifth section, a minimum-risk portfolio under both orthogonal
sets, with a mandate written against a factor name. Every piece is a field of one
[`JuMPOptimiser`](@ref).

[`cross_sectional_factor_sets`](@ref) takes the universe that the mandate is written against from
the estimator, before any fit. It declares the factor axis the fit will produce, one-hot industry
levels included, under the `ncf` key, and one plain group per factor family. It takes the one-hot
levels from the panel, and no code here types out a list of levels that could go out of date.
=#

solver = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                check_sol = (; allow_local = true, allow_almost = true),
                settings = Dict("verbose" => false))
universe = cross_sectional_factor_sets(pe, rd)
mandate = ExposureConstraintEstimator(;
                                      lce = LinearConstraintEstimator(;
                                                                      val = "size >= 0.10"),
                                      space = FactorSpace())
ucs = OrthogonalUncertaintySet(; kappa = 100.0)

direct = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = ucs), obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pe, slv = solver, bgt = 1.0,
                                               wb = WeightBounds(; lb = 0.0, ub = 0.1),
                                               sets = universe, lcse = mandate,
                                               ret = ArithmeticReturn(; ucs = ucs))), rd)

#=
## 3. The Pipeline route

The pipeline has three steps. The first two replace the `pe` and `ucs` arguments of the hand-wired
route, and the third is the optimiser without them.

 1. The prior step is the estimator itself. It writes the `:prior` slot.
 2. The uncertainty step is the same [`OrthogonalUncertaintySet`](@ref), wrapped in a
    [`PipelineStep`](@ref), because a computed uncertainty-set result cannot say on its own which
    parameter it bounds. `target = :both` derives the mean set and the covariance set from one
    [`ucs`](@ref) call. It also uses the `:prior` slot that the first step wrote, so the two sets
    are orthogonal to the factor model of this optimisation and not to some other one.
 3. The optimisation step has no `pe` and no `ucs` of its own. The pipeline passes the two slots to
    it.
=#

pipe = Pipeline(;
                steps = ("prior" => pe,
                         "uncertainty" => PipelineStep(; est = ucs, writes = :uncertainty,
                                                       target = :both),
                         "opt" =>
                             MeanRisk(; r = UncertaintySetVariance(), obj = MinimumRisk(),
                                      opt = JuMPOptimiser(; slv = solver, bgt = 1.0,
                                                          wb = WeightBounds(; lb = 0.0,
                                                                            ub = 0.1),
                                                          sets = universe, lcse = mandate,
                                                          ret = ArithmeticReturn()))))

piped = fit(pipe, rd)

pretty_table(DataFrame("Step" => collect(piped.names),
                       "Wrote" => ["prior", "uncertainty (mu and sigma)", "opt"],
                       "Result" => [string(nameof(typeof(piped.ctx.prior))),
                                    string(nameof(typeof(piped.ctx.uncertainty.mu))) *
                                    " / " *
                                    string(nameof(typeof(piped.ctx.uncertainty.sigma))),
                                    string(nameof(typeof(piped.results[3])))]);
             title = "What each step put in which slot")

#=
## 4. Comparing the two routes

Both routes run the same estimators on the same data in the same order. The table prints the
largest difference between the two weight vectors, the sum of the pipeline's weights, the number
of names it holds, and its exposure to size. A largest difference of zero means the two routes
returned the same weight for every name.
=#

pretty_table(DataFrame("max |w_pipeline - w_direct|" => maximum(abs, piped.w - direct.w),
                       "sum(w)" => sum(piped.w), "Names held" => count(>(1e-6), piped.w),
                       "Size exposure" => (transpose(direct.pa.pr.rr.M) * piped.w)[6]);
             formatters = [numfmt], title = "The Pipeline reaches the deep dive's book")

#=
## 5. Fold by fold

We repeat the comparison under cross-validation. A fold takes its subset of `rd`, panel included,
before either route sees the data. Both routes then refit the prior, refit the two uncertainty
sets against that fold's own factor model, and express the mandate through the loadings that fold
fitted. The first table prints the largest weight difference for each fold, and the second the
largest difference between the returns the two routes predict.
=#

walk = IndexWalkForward(252, 63)
folds_direct = cross_val_predict(MeanRisk(; r = UncertaintySetVariance(; ucs = ucs),
                                          obj = MinimumRisk(),
                                          opt = JuMPOptimiser(; pe = pe, slv = solver,
                                                              bgt = 1.0,
                                                              wb = WeightBounds(; lb = 0.0,
                                                                                ub = 0.1),
                                                              sets = universe,
                                                              lcse = mandate,
                                                              ret = ArithmeticReturn(;
                                                                                     ucs = ucs))),
                                 rd, walk)
folds_pipe = cross_val_predict(pipe, rd, walk)

pretty_table(DataFrame("Fold" => eachindex(folds_pipe.pred),
                       "max |w_pipeline - w_direct|" => [maximum(abs,
                                                                 folds_pipe.pred[i].res.w - folds_direct.pred[i].res.w)
                                                         for i in eachindex(folds_pipe.pred)],
                       "Names held" => [count(>(1e-6), p.res.w) for p in folds_pipe.pred]);
             formatters = [numfmt],
             title = "Fold by fold, the two routes are the same book")

pretty_table(DataFrame("max |predicted returns difference|" =>
                           maximum(abs, folds_pipe.mrd.X - folds_direct.mrd.X));
             formatters = [numfmt], title = "And so are the returns they predict")

#=
## Where to go next

  - [Cross-sectional factor model, end to end](05_Cross_Sectional_Factor_Model.md) is the long
    version of every estimator on this page.
  - [Pipelines](../5_validation_tuning/03_Pipelines.md) covers the slots, the routing and the
    hyper-parameter search this page only touches.
  - [Reading a return forecast before an optimiser sees it](07_Forecast_Evaluation.md) scores the
    return forecast that this page's prior uses, before any optimiser acts on it.
=#
