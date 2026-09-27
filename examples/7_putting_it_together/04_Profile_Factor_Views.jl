#=
```@meta
Description = "An end-to-end profile in PortfolioOptimisers.jl: a desk with factor views propagated through a factor model into a constrained optimisation."
```

# Profile: factor-views desk

The fourth profile states its views on factors rather than on assets. The
[desk monthly profile](@ref example-profile-desk-monthly) stated a thesis about two sectors, healthcare
over energy, and entropy pooling turned it into a prior. This desk states what it expects momentum
and value to earn, and a factor model takes those two numbers to every asset through the
regression of assets on factors. The posterior then goes into an optimiser with caps on it.

The [advanced Black-Litterman](@ref example-advanced-black-litterman-variants) page builds
this prior one variant at a time. This page puts it to work in a whole book, with a
[`FactorBlackLittermanPrior`](@ref), sector caps and an exact allocation.

For this desk, three limits of the
[strategy decision framework](@ref user-guide-choosing-a-strategy) bind.

  - The view is about what a factor earns, so we write it against the factor names and let the
    regression take it to the assets.
  - With only a cap on each asset, the book on this view puts most of its weight into two
    sectors. We give the optimiser a cap on each asset and a cap on each of those two sectors.
  - We take the risk-adjusted book with [`MaximumRatio`](@ref), and turn it into whole shares with
    a mixed-integer solve.

!!! tip "When to reach for this"
    Reach for this profile when your view is about a factor, such as momentum, value, quality,
    size or low volatility, rather than about a name. Write the view against the factors with a
    factor Black-Litterman prior, then cap the sectors in the optimiser.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS,
      StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data, factors, and the factor view

We load the S&P 500 slice together with its factor block, `MTUM`, `QUAL`, `SIZE`, `USMV` and
`VLUE`. Two `UniverseSets` follow. The asset one names the sectors the caps refer to. The factor
one names the assets and the factors in the column order of `rd.F`, which is the order
[`FactorBlackLittermanPrior`](@ref) reads them in. The view itself is two numbers: momentum earns
5 basis points a day, and value earns 3.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
F = TimeArray(CSV.File(joinpath(@__DIR__, "..", "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(price_ingestion(PriceIngestion(), X; F = F))
prices = vec(values(X)[end, :])

asset_sets = UniverseSets(;
                          dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                      "energy" => ["CVX", "XOM", "RRC"],
                                      "healthcare" => ["JNJ", "LLY", "MRK", "PFE", "UNH"]))
factor_sets = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf))
tau = 1 / size(rd.X, 1)

factor_views = LinearConstraintEstimator(; val = ["MTUM == 0.0005", "VLUE == 0.0003"])

#=
## 2. The factor Black-Litterman posterior

[`FactorBlackLittermanPrior`](@ref) regresses the assets on the factors, updates the factor means
and the factor covariance with the two views, and returns a posterior `mu` and `sigma` over the
assets. With `rsd = true` the posterior covariance adds the residual variance of each regression,
so it measures the whole risk of an asset rather than the part the factors explain.
=#

prior_est = FactorBlackLittermanPrior(; pe = EmpiricalPrior(), rsd = true,
                                      sets = factor_sets, tau = tau, views = factor_views)
pr = prior(prior_est, rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

#=
## 3. What the caps are for

A cap on each asset does not spread the book across sectors. We solve the risk-adjusted book once
with a 15% cap on each asset and nothing else, then add up the weight each sector holds.
=#

raw = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                        opt = JuMPOptimiser(; pe = pr, slv = slv,
                                            wb = WeightBounds(; lb = 0.0, ub = 0.15),
                                            sets = asset_sets)))

sector_weight(w, sec) = sum(w[[findfirst(==(t), rd.nx) for t in asset_sets.dict[sec]]])
raw_sectors = DataFrame("sector" => ["tech", "energy", "healthcare"],
                        "weight" => [sector_weight(raw.w, s)
                                     for s in ("tech", "energy", "healthcare")])
pretty_table(raw_sectors; formatters = [resfmt],
             title = "Sector weights with only a per-asset cap")

#=
The healthcare and energy rows take most of the book, and the tech row takes none of it.

## 4. The constrained desk book

The desk caps those two sectors, healthcare at 35% and energy at 25%, on top of the 15% cap on
each asset. The optimiser still chooses which names to buy, and tech has no sector cap.
=#

desk = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                         opt = JuMPOptimiser(; pe = pr, slv = slv,
                                             wb = WeightBounds(; lb = 0.0, ub = 0.15),
                                             lcse = LinearConstraintEstimator(;
                                                                              val = ["healthcare <= 0.35",
                                                                                     "energy <= 0.25"]),
                                             sets = asset_sets)))

constrained_sectors = DataFrame("sector" => ["tech", "energy", "healthcare"],
                                "constrained" => [sector_weight(desk.w, s)
                                                  for s in ("tech", "energy", "healthcare")])
pretty_table(constrained_sectors; formatters = [resfmt],
             title = "Sector weights with healthcare capped at 35% and energy at 25%")

pretty_table(DataFrame("Asset" => rd.nx, "Weight" => desk.w); formatters = [resfmt],
             title = "Factor-views desk tangency weights with the sector caps")

#=
Compare the two sector tables. Healthcare and energy sit at their caps in the second one, and the
weight that left them went to names outside the three sectors. The third table gives the whole
book, name by name.

## 5. Exact finite allocation

The desk invests \$1,000,000. As in the [institutional profile](@ref example-profile-institutional),
[`DiscreteAllocation`](@ref) rounds the target to whole shares with a mixed-integer solve.
=#

mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
alloc = optimise(DiscreteAllocation(; slv = mip_slv),
                 FiniteAllocationInput(; w = desk.w, prices = prices, cash = 1_000_000.0))

invested = sum(alloc.shares .* prices)
pretty_table(DataFrame("Asset" => rd.nx, "Target" => desk.w,
                       "Shares" => round.(Int, alloc.shares), "Realised" => alloc.w);
             formatters = [resfmt],
             title = "\$1,000,000 to invest, \$$(round(Int, invested)) invested, \$$(round(alloc.cash, digits = 2)) left in cash")

#=
## 6. The book
=#

plot_stacked_bar_composition([desk], rd; xticks = (1:1, ["Factor-views desk"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New 4th end-to-end profile (ADR 0014 capstone gap: factor + BL + JuMP-with-constraints,
#src   isolated as a reusable pattern). None of the other three profiles uses a factor-BL prior.
#src - All cells verified end-to-end under Kaimon (docs env, GKSwstype=100) on the last-252-obs
#src   SP500 + Factors slice. FactorBlackLittermanPrior(views MTUM==0.0005, VLUE==0.0003) builds;
#src   MaximumRatio tangency solves (Clarabel); DiscreteAllocation (HiGHS) $1M → invested $999,983,
#src   cash ~$17.
#src - VERIFIED constraints bind: raw per-asset-cap-only book = healthcare 45% / energy 35% / tech 0;
#src   with healthcare<=0.35 + energy<=0.25 caps the book diversifies to healthcare 35% / energy 25%,
#src   8 names, maxw 0.15. tech stays 0 (the factor tilt + risk-adjusted optimum never wants it).
#src - BUG FOUND AND FIXED while authoring: scalar WeightBounds/WeightBoundsEstimator (e.g. ub=0.15)
#src   errored under MaximumRatio ("Subtraction between an array and a JuMP scalar") because
#src   set_weight_constraints! built `w - k*lb` / `w - k*ub` without broadcasting; vector bounds
#src   worked by accident. Fixed to use the project's `⊖` operator (src/.../03_WeightConstraints.jl),
#src   matching the sibling in 01_Base_Optimisation/. Regression test added in
#src   test/test_18k_constraints.jl ("Scalar weight bounds (broadcast ⊖ regression)").
#src - Contrast with the other profiles: retail = cost control, desk monthly = view + frontier,
#src   institutional = constraints + benchmark, factor-views = factor-space view + sector caps.
