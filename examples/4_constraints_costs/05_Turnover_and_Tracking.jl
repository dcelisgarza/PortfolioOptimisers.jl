#=
```@meta
Description = "Turnover and tracking in PortfolioOptimisers.jl, each as a constraint on a JuMPOptimiser or as a risk measure to minimise directly."
```

# Turnover and tracking

Turnover is the distance of the weights from a reference weight vector, such as your current
weights when you rebalance. Tracking is the distance of the portfolio returns from the returns of
a benchmark. Each comes in two forms. The constraint form, a keyword of the
[`JuMPOptimiser`](@ref), keeps the distance under a limit. The risk measure form makes the
distance the quantity you minimise.

  - [`Turnover`](@ref), through the `tn` keyword, and [`TurnoverRiskMeasure`](@ref) measure the
    change of the weights from a reference weight vector.
  - [`TrackingError`](@ref), through the `tr` keyword, and [`TrackingRiskMeasure`](@ref) measure
    the distance from a benchmark. The benchmark is either a weight vector,
    [`WeightsTracking`](@ref), or a return series such as an index, [`ReturnsTracking`](@ref).

!!! tip "When to reach for this"
    Reach for turnover when trading is costly and you rebalance often, so the new book must stay
    close to the old one. Reach for tracking when your mandate judges you against a benchmark.
    To replicate an index, minimise the tracking error. To beat it within a limit, maximise the
    return with a tracking-error budget. Use the constraint form for a limit that a mandate
    sets, and the risk measure form when staying near the current book or the benchmark is the
    goal itself.
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

#=
## 1. Data and two benchmarks

The data is one year of S&P 500 prices, with two benchmarks. One is the equal-weight book, which
is a weight vector. The other is the S&P 500 index, which is a return series.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
N = length(rd.nx)

idx = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500_idx.csv.gz")); timestamp = :Date)
index_returns = vec(values(percentchange(idx)))[(end - 251):end]

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

equal_weight = fill(1 / N, N)

#=
## 2. Turnover as a constraint

[`Turnover`](@ref), through `tn`, limits how far each weight may move from a reference weight
vector `w`, which is normally your current holdings. `val` is the largest change it allows for
each asset. We maximise the ratio with the equal-weight book as the reference, for four values
of `val`. For each result we print the sum of the absolute changes from the equal-weight book,
and the largest weight.
=#

turnover_vals = [0.005, 0.02, 0.1, 0.5]
turnover_res = [optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                  opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                      tn = Turnover(; w = equal_weight,
                                                                    val = v))))
                for v in turnover_vals]

drift(w) = sum(abs, w .- equal_weight)
pretty_table(DataFrame("Turnover budget" => turnover_vals,
                       "Drift from start" => drift.(getproperty.(turnover_res, :w)),
                       "Max weight" => maximum.(getproperty.(turnover_res, :w)));
             formatters = [resfmt],
             title = "Turnover budget and drift from the equal-weight book")

#=
## 3. Turnover as a risk measure

[`TurnoverRiskMeasure`](@ref) makes the turnover the quantity to minimise. We minimise it with
the equal-weight book as the reference, and print the drift of the result from that book, the
sum of the absolute changes of section 2. With no other term the turnover is smallest at the
reference, and a drift near zero means that the result is the reference book. The measure is
useful as one term of a problem with more than one risk measure. You can also compute it for a
candidate portfolio, to find how far that portfolio is from the current weights.
=#

res_min_turnover = optimise(MeanRisk(; r = TurnoverRiskMeasure(; w = equal_weight),
                                     obj = MinimumRisk(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv)))
drift(res_min_turnover.w)

#=
## 4. Tracking a benchmark

With [`TrackingRiskMeasure`](@ref) we minimise the tracking error to a benchmark. A
[`WeightsTracking`](@ref) benchmark is a portfolio of the same assets, so the minimum is that
portfolio. A [`ReturnsTracking`](@ref) benchmark, here the return series of the S&P 500 index,
gives the portfolio of our 20 assets whose returns have the smallest tracking error to the
index.
=#

res_replicate_ew = optimise(MeanRisk(;
                                     r = TrackingRiskMeasure(;
                                                             tr = WeightsTracking(;
                                                                                  w = equal_weight)),
                                     obj = MinimumRisk(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv)))
res_replicate_idx = optimise(MeanRisk(;
                                      r = TrackingRiskMeasure(;
                                                              tr = ReturnsTracking(;
                                                                                   w = index_returns)),
                                      obj = MinimumRisk(),
                                      opt = JuMPOptimiser(; pe = pr, slv = slv)))

pretty_table(DataFrame("Asset" => rd.nx, "Replicate EW" => res_replicate_ew.w,
                       "Replicate index" => res_replicate_idx.w); formatters = [resfmt],
             title = "Minimum tracking-error weights for two benchmarks")

#=
## 5. Tracking as a constraint

To beat a benchmark within a limit, you maximise the return with the tracking error at or below
a budget. [`TrackingError`](@ref), through `tr`, keeps the tracking error at or below `err`.
`err` is in the units of the tracking error, which is a norm of the return differences, not a
distance between weights. We maximise the ratio against the equal-weight benchmark with four
values of `err`. A small `err` keeps the portfolio near the benchmark, and a large one gives the
optimiser more room to raise the return.
=#

err_vals = [0.0005, 0.001, 0.005, 0.02]
track_res = [optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                               opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                   tr = TrackingError(;
                                                                      tr = WeightsTracking(;
                                                                                           w = equal_weight),
                                                                      err = e))))
             for e in err_vals]

pretty_table(DataFrame("Tracking-error budget" => err_vals,
                       "Drift from benchmark" => drift.(getproperty.(track_res, :w)),
                       "Max weight" => maximum.(getproperty.(track_res, :w)));
             formatters = [resfmt],
             title = "Drift from the equal-weight benchmark at four tracking-error budgets")

#=
The `alg` field of [`TrackingError`](@ref) sets the norm of the tracking error, which acts on
the series of differences between the portfolio returns and the benchmark returns. The default
is [`L2Norm`](@ref). [`L1Norm`](@ref) uses the sum of the absolute differences, [`LpNorm`](@ref)
a p-norm of them, and [`LInfNorm`](@ref) the largest difference in one period. Each `alg`
except `LInfNorm` divides its norm by a factor of the number of observations, and `err` is in
the units of that result. With `L1Norm` the result is the mean absolute difference. `LInfNorm`
does not divide, so its `err` bounds the largest difference in one period.

## 6. Comparing the approaches

We plot four of the portfolios: the equal-weight replica, two tracking-error budgets and one
turnover limit.
=#

results = [res_replicate_ew, track_res[1], track_res[3], turnover_res[2]]
labels = ["Replicate EW", "Track err 5e-4", "Track err 5e-3", "Turnover 0.02"]

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive (4_constraints_costs). All verified on kaimon (f102cae9), real SP500 slice
#src   + SP500_idx for ReturnsTracking:
#src   - Turnover CONSTRAINT (tn) val sweep monotone: val 0.005/0.02/0.1/0.5 → drift-from-EW
#src     0.1/0.4/1.26/1.7 (val is per-name turnover budget).
#src   - TrackingRiskMeasure(WeightsTracking(ew)) min → reproduces EW exactly (drift 0).
#src     TrackingRiskMeasure(ReturnsTracking(index)) min → 19-name replicator, maxw 18.4%.
#src   - TrackingError CONSTRAINT (tr) err sweep monotone: err 5e-4/1e-3/5e-3/2e-2 → drift-from-EW
#src     0.158/0.312/1.07/1.8 (err is tracking-error units, NOT L1 weight; err=0.10 doesn't bind).
#src   - TurnoverRiskMeasure(w=ref) min → stays at ref (trivial alone; useful in multi-objective).
#src - ReturnsTracking needs a benchmark return series length T (= rows of rd.X); built from
#src   percentchange(SP500_idx) last 252. WeightsTracking needs a length-N weight vector.
#src - Tracking norms L1/Lp/LInf via `alg` — mentioned, not swept (norm choice is qualitative).
