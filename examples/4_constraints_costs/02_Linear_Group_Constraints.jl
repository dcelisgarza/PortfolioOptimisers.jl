#=
```@meta
Description = "Linear and group constraints in PortfolioOptimisers.jl: weight bounds, sector bands, relative and sum constraints on a JuMPOptimiser."
```

# [Linear and group constraints](@id example-linear-and-group-constraints)

A mandate can cap the weight of a single name, keep each sector inside a band, keep one group
larger than another, or forbid very small positions. `PortfolioOptimisers.jl` expresses each of
these as a constraint on the [`JuMPOptimiser`](@ref), and you can add them to any objective and
prior. This page covers weight bounds, bounds on each member of a group and on the group's sum,
and relative and sum constraints. Section 6 explains why thresholds and cardinality need a
mixed-integer solver.

You name the assets and the groups once, in a [`UniverseSets`](@ref), and every constraint
refers to those names. The views of the [prior
examples](@ref example-entropy-pooling) use strings of the same `"name op value"`
form.

!!! tip "When to reach for this"
    Reach for these when a mandate limits what the portfolio holds: a 5% cap on a single name, a
    20% ceiling on a sector, "healthcare at least as large as energy", "no position under 2%".
    All of them are convex except the thresholds and cardinality, which need a mixed-integer
    solver. You can use them with any objective, risk measure and prior. For limits that carry a
    cost, such as turnover, fees and tracking, see the other pages of this group.
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
## 1. Data

The data is one year of S&P 500 prices. We fit an empirical prior and set up one Clarabel
solver.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

#=
## 2. Naming assets and groups

A [`UniverseSets`](@ref) maps each name to its members. The `nx` key holds every asset. The
other keys are the groups that the constraints refer to, which here are sectors.
=#

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                "financials" => ["BAC", "JPM"],
                                "energy" => ["CVX", "XOM", "RRC"],
                                "healthcare" => ["JNJ", "LLY", "MRK", "PFE", "UNH"],
                                "staples" => ["KO", "PEP", "PG", "WMT"],
                                "consumer" => ["BBY", "HD"], "industrial" => ["GE"]))

#=
The baseline is a maximum-ratio portfolio with the default constraints only. We print its weight
in each sector. On this year of data it holds two sectors and nothing else, so each constraint
below has an effect you can see.
=#

res_base = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))

sector_weight(w, g) = sum(w[i] for i in eachindex(w) if rd.nx[i] in sets.dict[g])
sectors = ["tech", "financials", "energy", "healthcare", "staples", "consumer",
           "industrial"]
pretty_table(DataFrame("Sector" => sectors,
                       "Baseline" => [sector_weight(res_base.w, g) for g in sectors]);
             formatters = [resfmt], title = "Baseline max-ratio: sector exposure")

#=
## 3. Weight bounds: capping concentration

The simplest constraint is a bound on each weight, through `wb`. A [`WeightBounds`](@ref) with
`ub = 0.15` keeps every weight at or below 15%. The weights sum to one, so the optimiser must
now hold at least seven names.
=#

res_cap = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                wb = WeightBounds(; lb = 0.0, ub = 0.15))))

#=
## 4. A bound on each member against a bound on the group's sum

"A 15% floor on staples" can mean two different constraints. A [`WeightBoundsEstimator`](@ref)
with a group key bounds each member of the group. A [`LinearConstraintEstimator`](@ref) bounds
the sum of the group. Staples has four names here. `WeightBoundsEstimator(lb = ["staples" =>
0.15], ub = nothing)` holds each of the four at 15% or more, which is at least 60% in total.
`LinearConstraintEstimator(val = ["staples >= 0.15"])` holds only their sum at 15% or more.
=#

res_member = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                               opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                   wb = WeightBoundsEstimator(;
                                                                              lb = ["staples" =>
                                                                                        0.15],
                                                                              ub = nothing))))
res_groupsum = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                     lcse = LinearConstraintEstimator(;
                                                                                      val = ["staples >= 0.15"]))))

pretty_table(DataFrame("Interpretation" =>
                           ["per-member (each ≥ 15%)", "group-sum (total ≥ 15%)"],
                       "Staples total" => [sector_weight(res_member.w, "staples"),
                                           sector_weight(res_groupsum.w, "staples")]);
             formatters = [resfmt], title = "Staples total for each form of the 15% floor")

#=
Use the `WeightBoundsEstimator` form when the rule is about each name: "every position in this
list at least x". Use the `LinearConstraintEstimator` form when the rule is about the total of a
sector.

## 5. Linear group constraints: sums and relations

A [`LinearConstraintEstimator`](@ref) string combines group and asset names with `+`, `-`,
numbers that multiply a name, and one of `==`, `<=` and `>=`. With them you can cap a sum,
"these two sectors together at most 60%", or relate two groups, "healthcare at least twice
staples". We cap the two sectors of the baseline at 60% together, and hold tech at or below
financials. At least 40% of the book must then go to sectors that the baseline did not hold.
=#

res_sum = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                lcse = LinearConstraintEstimator(;
                                                                                 val = ["healthcare + energy <= 0.6",
                                                                                        "tech <= financials"]))))

pretty_table(DataFrame("Sector" => sectors,
                       "Baseline" => [sector_weight(res_base.w, g) for g in sectors],
                       "Sum-capped" => [sector_weight(res_sum.w, g) for g in sectors]);
             formatters = [resfmt],
             title = "Sum cap (healthcare + energy ≤ 60%, tech ≤ financials)")

#=
## 6. Thresholds and cardinality need a mixed-integer solver

Two common constraints are not convex, so Clarabel cannot solve them alone.

  - A threshold, [`ThresholdEstimator`](@ref) through the `lt` and `st` keywords, says "if you
    hold a name at all, hold at least x". A weight is then either zero or at least the floor.
  - A cardinality constraint, through the `card` and `gcarde` keywords, says "hold at most k
    names", and the solver must choose which names.

Both need a mixed-integer solver, for example Pajarito with Clarabel as the continuous solver
and HiGHS as the mixed-integer solver. If you give a threshold to a continuous solver, the
result carries a failed `retcode` and `NaN` weights. [Cardinality and
threshold](@ref example-cardinality-and-threshold) sets up such a solver.

Every constraint on this page uses asset names. A mandate in factor names, such as "at most 10%
momentum" or "no net exposure to value", is the same linear form over the factor exposures.
[Factor exposure constraints](@ref example-factor-exposure-constraints) covers it.

## 7. Comparing the constraints

We print the weights of the four portfolios and plot them. The prior and the objective are the
same in each, so only the constraint differs.
=#

results = [res_base, res_cap, res_groupsum, res_sum]
labels = ["Baseline", "Cap 15%", "Staples ≥ 15%", "Sum cap"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Asset weights under each constraint")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive (4_constraints_costs group). All bindings verified on kaimon (f102cae9) on
#src   the real SP500 slice with a MaximumRatio base (concentrates ~66% healthcare / 34% energy,
#src   so caps/floors on the ignored sectors bite cleanly):
#src   - WeightBounds ub=0.15: max weight 37%→15%, holdings 2→8.
#src   - PER-MEMBER vs GROUP-SUM (the teaching highlight): WeightBoundsEstimator(lb=["staples"=>0.15], ub=nothing))
#src     → each of 4 staples ≥15% → 60% total; LinearConstraintEstimator("staples >= 0.15") → 15%
#src     total. Genuinely confusable; documented explicitly in §4.
#src   - Sum cap "healthcare + energy <= 0.6" binds at 60%, pushes 40% into previously-zero sectors.
#src - FINDING (→ this group's issue): ThresholdEstimator(lt) and cardinality are MIP — Clarabel
#src   returns a FAILED retcode (not an error, not a wrong answer). Documented in §6 with the
#src   Pajarito/HiGHS pointer. A clearer "this constraint needs a MIP solver" message would help.
#src - cte (centrality) / phylogeny constraints deferred to 04_Phylogeny_Centrality.
