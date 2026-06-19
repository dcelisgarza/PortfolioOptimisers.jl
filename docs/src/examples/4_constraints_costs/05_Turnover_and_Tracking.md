The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/4_constraints_costs/05_Turnover_and_Tracking.jl"
```

# Turnover and tracking

The constraints so far shape *what* the portfolio holds. **Turnover** and **tracking** constrain
how it *moves*: how far it may drift from your current book when you rebalance, and how far it may
stray from a benchmark. Both come in two flavours in `PortfolioOptimisers.jl` — as a **constraint**
on a [`JuMPOptimiser`](@ref), or as a **risk measure** you minimise directly — and the choice
between them is the difference between "respect this limit" and "make this the goal".

- **Turnover** ([`Turnover`](@ref) via `tn`, or [`TurnoverRiskMeasure`](@ref)) penalises trading
    away from a reference weight vector.
- **Tracking** ([`TrackingError`](@ref) via `tr`, or [`TrackingRiskMeasure`](@ref)) penalises
    deviation from a benchmark, specified either as weights ([`WeightsTracking`](@ref)) or as a
    benchmark return series ([`ReturnsTracking`](@ref) — e.g. an index).

!!! tip "When to reach for this"
    Reach for **turnover** when trading is costly and you rebalance often — you want the new book
    close to the old one. Reach for **tracking** when you are benchmarked: index replication
    (minimise tracking error) or enhanced indexing (seek return *subject to* a tracking-error
    budget). Use the *constraint* form when the limit is a hard mandate, the *risk-measure* form
    when staying put / hugging the benchmark is itself the objective.

````@example 05_Turnover_and_Tracking
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. Data and a benchmark

We use the S&P 500 slice, and two benchmarks: an equal-weight book (a weight vector) and the
S&P 500 index itself (a return series).

````@example 05_Turnover_and_Tracking
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
````

## 2. Turnover as a constraint

[`Turnover`](@ref) (`tn`) limits how far the weights may move from a reference book `w` — your
current holdings — so a rebalance stays cheap. We seek the maximum-ratio portfolio but anchor it
at the equal-weight book and tighten the per-name turnover budget `val`. A smaller budget keeps
the result closer to where we started.

````@example 05_Turnover_and_Tracking
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
             title = "Tighter turnover budget keeps the book near the reference")
````

## 3. Turnover as a risk measure

[`TurnoverRiskMeasure`](@ref) makes *minimising* turnover the objective rather than a side
constraint. Minimising turnover from the current book with no other pull simply returns the
current book — useful as one term in a multi-objective problem, or to measure the trading cost of
a target.

````@example 05_Turnover_and_Tracking
res_min_turnover = optimise(MeanRisk(; r = TurnoverRiskMeasure(; w = equal_weight),
                                     obj = MinimumRisk(),
                                     opt = JuMPOptimiser(; pe = pr, slv = slv)))
````

## 4. Tracking a benchmark

[`TrackingRiskMeasure`](@ref) minimises the tracking error to a benchmark. With a
[`WeightsTracking`](@ref) benchmark it reproduces that book exactly; with a
[`ReturnsTracking`](@ref) benchmark — here the S&P 500 index return series — it builds the
*replicating* portfolio from our 20 assets that best tracks the index.

````@example 05_Turnover_and_Tracking
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
             title = "Pure tracking: equal-weight book vs index replication")
````

## 5. Enhanced indexing: tracking as a constraint

The more interesting case is *enhanced indexing* — seek return, but stay within a tracking-error
budget of the benchmark. [`TrackingError`](@ref) (`tr`) bounds the tracking error to `err`. We
maximise the ratio while tightening `err` against the equal-weight benchmark: a small `err` hugs
the benchmark, a large one frees the optimiser to chase return.

````@example 05_Turnover_and_Tracking
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
             title = "Tighter tracking-error budget hugs the benchmark")
````

The tracking error itself can be measured with different norms — [`L1Tracking`](@ref) (absolute,
sparse), [`LpTracking`](@ref) (general p-norm), [`LInfTracking`](@ref) (worst single deviation) —
passed as the `alg`, so you can choose whether to penalise the total drift or the largest single
bet away from the benchmark.

## 6. Comparing the approaches

````@example 05_Turnover_and_Tracking
results = [res_replicate_ew, track_res[1], track_res[3], turnover_res[2]]
labels = ["Replicate EW", "Track err 5e-4", "Track err 5e-3", "Turnover 0.02"]

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
