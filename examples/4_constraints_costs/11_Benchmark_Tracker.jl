#=
```@meta
Description = "A benchmark tracker in PortfolioOptimisers.jl: the portfolio that minimises the tracking error to an index, with a cap on the number of assets that it holds."
```

# [Benchmark tracker](@id example-benchmark-tracker)

A benchmark tracker is a portfolio whose returns follow the returns of an index as closely as
possible. Its objective is the tracking error, the distance between the two series of returns.
A fund that tracks an index often holds only a part of the index, because each position costs
money to hold and to trade, so the tracker also has a cap on the number of assets.

The library states this problem with parts that already exist:

  - [`TrackingRiskMeasure`](@ref) with a [`ReturnsTracking`](@ref) benchmark measures the
    tracking error to a series of returns, such as an index;
  - [`MinimumRisk`](@ref) makes that measure the objective of [`MeanRisk`](@ref);
  - the `card` keyword of [`JuMPOptimiser`](@ref) caps the number of assets that hold a weight.

The [turnover and tracking](@ref example-turnover-and-tracking) example shows the tracking
measures, and the [cardinality and threshold](@ref example-cardinality-and-threshold) example
shows `card`. This page puts them together, and shows how the tracking error grows as the cap
falls.

!!! tip "When to reach for this"
    Reach for a benchmark tracker when your mandate is to follow an index, and you cannot or do
    not want to hold every asset of the index. To beat the index within a limit instead, put the
    tracking error in a constraint, as section 5 of the
    [turnover and tracking](@ref example-turnover-and-tracking) example does.

!!! tip "A mixed-integer solver is required"
    `card` adds binary variables, so the capped trackers need a mixed-integer solver.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, HiGHS, JuMP,
      StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, AbstractFloat) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data, the index and the solvers

The data is one year of S&P 500 prices of 20 assets. The index is the S&P 500 index, and we
take its returns over the same days.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
N = length(rd.nx)

idx = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500_idx.csv.gz")); timestamp = :Date)
index_returns = vec(values(percentchange(idx)))[(end - 251):end]

#=
## 2. The tracking error and the solver

[`TrackingRiskMeasure`](@ref) with a [`ReturnsTracking`](@ref) benchmark measures the distance
between the returns of the portfolio and the returns of the index. Its `alg` field sets the norm
of the daily differences. We use [`L1Norm`](@ref), so the tracking error is the mean absolute
difference between the daily return of the portfolio and the daily return of the index. It is
in the units of a daily return.

With [`L1Norm`](@ref), the tracker is a linear program, and the tracker with a cap is a
mixed-integer linear program. HiGHS solves both.
=#

tr = TrackingRiskMeasure(; tr = ReturnsTracking(; w = index_returns), alg = L1Norm())
te(w) = expected_risk(tr, w, pr)

slv = Solver(; name = :highs,
             solver = optimizer_with_attributes(HiGHS.Optimizer, JuMP.MOI.Silent() => true),
             check_sol = (; allow_local = true, allow_almost = true))

#=
!!! note "The default norm is slower with a cap"
    The default norm, [`L2Norm`](@ref), is the Euclidean norm of the daily differences divided
    by ``\sqrt{T - 1}``, where ``T`` is the number of days. It makes the capped tracker a
    mixed-integer conic problem, which needs a solver such as Pajarito, as in the
    [cardinality and threshold](@ref example-cardinality-and-threshold) example. On this data,
    one solve with `card = 5` took 164 s with Pajarito. The same solve with
    [`L1Norm`](@ref) took 5 s with HiGHS, and the two portfolios had almost the same
    [`L2Norm`](@ref) tracking error.
=#

#=
## 3. The tracker without a cap

We minimise the tracking error with no cap, and compare it with the tracking error of the
equal-weighted portfolio.
=#

res_full = optimise(MeanRisk(; r = tr, obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))
equal_weight = fill(1 / N, N)
(te(res_full.w), te(equal_weight), count(>(1e-6), res_full.w))

#=
## 4. A cap on the number of assets

`card` caps the number of assets that hold a weight. We solve the tracker for four caps.
=#

cards = [3, 5, 8, 12]
res_card = [optimise(MeanRisk(; r = tr, obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv, card = k)))
            for k in cards]

pretty_table(DataFrame(; :cap => [string.(cards); "none"; "equal weighted"],
                       Symbol("assets held") =>
                           [[count(>(1e-6), res.w) for res in res_card];
                            count(>(1e-6), res_full.w); N],
                       Symbol("tracking error") => [te.(getproperty.(res_card, :w));
                                                    te(res_full.w); te(equal_weight)]);
             formatters = [resfmt],
             title = "Tracking error to the S&P 500 index by the cap on the number of assets")

#=
Each tracker holds no more assets than its cap, and each one has a smaller tracking error than
the equal-weighted portfolio of all 20 assets.
=#

all(count(>(1e-6), res.w) <= k && te(res.w) < te(equal_weight)
    for (res, k) in zip(res_card, cards))

#=
A smaller cap gives the optimiser fewer choices, so the tracking error does not get smaller when
the cap gets smaller.
=#

issorted(te.(getproperty.(res_card, :w)); rev = true)

#=
## 5. The tracker against the index

The plot shows the cumulative returns of the index, of the tracker with the cap of five assets,
and of the equal-weighted portfolio.
=#

cumret(x) = cumprod(1 .+ x) .- 1
plot(rd.ts, cumret(index_returns); label = "S&P 500 index", ylabel = "Cumulative return",
     title = "Tracker and index")
plot!(rd.ts, cumret(rd.X * res_card[2].w); label = "tracker, cap of 5")
plot!(rd.ts, cumret(rd.X * equal_weight); label = "equal weighted")

#=
## 6. The compositions

The stacked bars show the weights of the trackers.
=#

plot_stacked_bar_composition([res_card; res_full], rd;
                             xticks = (1:(length(cards) + 1),
                                       [string.("cap ", cards); "no cap"]))
