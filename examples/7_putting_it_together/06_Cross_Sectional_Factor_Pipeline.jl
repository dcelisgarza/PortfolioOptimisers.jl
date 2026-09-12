#=
# Cross-sectional factor model through a Pipeline

The [deep dive](05_Cross_Sectional_Factor_Model.md) built a cross-sectional factor model, fitted
two orthogonal uncertainty sets from it and solved a constrained book, by wiring the estimators
into a [`JuMPOptimiser`](@ref) by hand. This page reaches **the same weights** through a
[`Pipeline`](@ref), and it is short because almost nothing has to change.

The reason it is short is the answer to the question this page exists to settle: **the point-in-time
Asset Panel needs no Pipeline Data Slot of its own.** The panel rides `rd.pnl` on the returns
carrier, and `:returns` is a slot the Pipeline already has. So:

  - No new slot, and no declaration. A step that needs the panel reads the carrier it was already
    handed.
  - Every step that slices the carrier slices the panel with it, because both go through the same
    view contract.
  - A fold slices the panel before the Pipeline sees the data, through that same contract.

What the Pipeline *does* add is that the three pieces become three named steps, and the middle one
— the uncertainty set — reads the `:prior` slot the first step wrote. That is the route the
hand-wired version hides inside the optimiser.
=#

using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, PrettyTables,
      DataFrames, Clarabel

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

#=
## 1. The same panel, and the same specification

The generator and the estimator below are the deep dive's, unchanged and at the same seed, so the
book this page solves is the book that page solved. Read
[§1 and §2 there](05_Cross_Sectional_Factor_Model.md) for what each piece is; here they are just
the input.
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
The panel is already on the carrier, so it is already in the `:returns` slot. Nothing below
mentions it again.
=#

pretty_table(DataFrame("Carrier" => string(nameof(typeof(rd))),
                       "Panel Fields on rd.pnl" => length(rd.pnl.pf),
                       "Slot it rides" => ":returns"))

#=
## 2. The hand-wired route

This is the deep dive's §5 book: a minimum-risk portfolio under both orthogonal sets and a factor
mandate written in a factor name. Everything is a field of one [`JuMPOptimiser`](@ref).
=#

solver = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                check_sol = (; allow_local = true, allow_almost = true),
                settings = Dict("verbose" => false))
universe = UniverseSets(;
                        dict = Dict("nx" => rd.nx,
                                    "ncf" =>
                                        ["market", "industry=Energy", "industry=Financials",
                                         "industry=Health Care", "industry=Technology",
                                         "size", "value", "earnings_yield", "liquidity"]))
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

Three steps, and each one is a piece the hand-wired optimiser held in a field:

 1. The **prior** step is the estimator itself. It writes the `:prior` slot.
 2. The **uncertainty** step is the same [`OrthogonalUncertaintySet`](@ref), wrapped in a
    [`PipelineStep`](@ref) because a computed uncertainty-set result cannot say on its own which
    parameter it bounds. `target = :both` derives the mean set and the covariance set from a single
    [`ucs`](@ref) call — and, crucially, it reads the `:prior` slot the first step wrote, which is
    what makes the sets orthogonal to *this* optimisation's own factor model.
 3. The **optimisation** step carries no `pe` and no `ucs` of its own. The Pipeline injects the two
    slots into it.
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
## 4. The two routes agree

Not approximately — the Pipeline runs the same estimators on the same data in the same order, so
the weights are identical to the last bit.
=#

pretty_table(DataFrame("max |w_pipeline - w_direct|" => maximum(abs, piped.w - direct.w),
                       "sum(w)" => sum(piped.w), "Names held" => count(>(1e-6), piped.w),
                       "Size exposure" => (transpose(direct.pa.pr.rr.M) * piped.w)[6]);
             formatters = [numfmt], title = "The Pipeline reaches the deep dive's book")

#=
## 5. And they agree fold by fold

The same holds under cross-validation, which is the claim that matters: a fold slices the carrier
— and the panel on it — before either route sees the data, so both refit the prior, both refit the
two uncertainty sets against that fold's own factor model, and both re-base the mandate through
the loadings the fold actually fitted.
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
  - [Reading a Return Forecast before an optimiser sees it](07_Forecast_Evaluation.md) scores the
    Return Forecast this page's prior carries, before any optimiser acts on it.
=#
