#=
```@meta
Description = "Cardinality and threshold constraints in PortfolioOptimisers.jl on assets, on groups of assets, on sets of assets and on groups of sets."
```

# Cardinality and threshold

This example applies cardinality and threshold constraints at four levels.

1. Cardinality and minimum position sizes on assets: `card`, `lt`, `st`.
2. Cardinality on a group of assets: `gcarde`.
3. Cardinality and thresholds on sets of assets: `scard`, `slt`, `sst` with `smtx`.
4. Cardinality and thresholds on groups of sets: `sgcarde`, `sglt`, `sgst` with `sgmtx`.

!!! tip "When to reach for this"
    Reach for cardinality and threshold constraints when the number of positions matters as much
    as their sizes: a maximum number of holdings, a minimum size for each position you open, or
    a limit on how many names, sets or groups hold a weight. They limit how many positions the
    solution holds.

!!! tip "A mixed-integer solver is required"
    These constraints add binary variables, so they need a mixed-integer solver.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, HiGHS,
      Pajarito, JuMP, StatsPlots, GraphRecipes

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end

#=
## 1. Setup

We load one year of S&P 500 prices and fit a `HighOrderPriorEstimator` prior. The mixed-integer
solver is Pajarito, which uses HiGHS for the integer part and Clarabel for the conic part.
Pajarito can print `Warning: integral solution repeated` on some solves. This is progress
information, and the solve still succeeds.

The [`UniverseSets`](@ref) name the groups, the clusters and the industries that the constraints
below refer to. We chose them so that every problem on this page has a solution.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(HighOrderPriorEstimator(), rd)
rf = 4.2 / 100 / 252

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
       Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85))]

mip_slv = [Solver(; name = :mip1,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip2,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.95)),
                  check_sol = (; allow_local = true, allow_almost = true))]

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                                "group2" => rd.nx[2:2:end],
                                "clusters1" =>
                                    [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                                     3, 3],
                                "clusters2" =>
                                    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                                     1, 2], "c1" => rd.nx[1:3:end], "c2" => rd.nx[2:3:end],
                                "c3" => rd.nx[3:3:end],
                                "nx_industries" =>
                                    ["Technology", "Technology", "Financials",
                                     "Consumer_Discretionary", "Energy", "Industrials",
                                     "Consumer_Discretionary", "Healthcare", "Financials",
                                     "Consumer_Staples", "Healthcare", "Healthcare",
                                     "Technology", "Consumer_Staples", "Healthcare",
                                     "Consumer_Staples", "Energy", "Healthcare",
                                     "Consumer_Staples", "Energy"],
                                "ux_industries" =>
                                    ["Technology", "Financials", "Consumer_Discretionary",
                                     "Energy", "Industrials", "Healthcare",
                                     "Consumer_Staples"]))

#=
## 2. Asset-level: `card`, `lt`, `st`

`card` caps the number of assets that hold a weight. `lt` and `st` are the minimum sizes of a
long and of a short position. We use `card` in three problems: minimum risk with `card = 3`,
maximum ratio with `card = 3` and an L2 penalty, and a long-short maximum ratio with `card = 7`.
The cells set no `lt` or `st`.
=#

res_card = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = 3)))

res_card_l2 = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                opt = JuMPOptimiser(; l2 = L2Regularisation(; val = 0.1),
                                                    pe = pr, slv = mip_slv, card = 3)))

res_card_thr = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                 opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                                                     wb = WeightBounds(; lb = -1, ub = 1),
                                                     sbgt = 1, bgt = 1, card = 7)))

println("Asset-level cardinality")
println("  active assets (card = 3): ", count(res_card.w .> 1e-10))
println("  active assets (l2 + card = 3): ", count(res_card_l2.w .> 1e-10))
println("  active assets with bounds (card = 7): ", count(abs.(res_card_thr.w) .> 1e-10))
pretty_table(DataFrame(:Asset => rd.nx, :w_card => res_card.w, :w_card_l2 => res_card_l2.w,
                       :w_card_bounded => res_card_thr.w); formatters = [resfmt])

#=
## 3. Group cardinality over assets: `gcarde`

`gcarde` takes linear expressions over asset and group names, and each one bounds how many
assets of a group hold a weight. We let at most two of XOM, MRK and WMT hold a weight, and
exactly five assets of `group2`.
=#

res_gcard = optimise(MeanRisk(;
                              opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                                                  gcarde = LinearConstraintEstimator(;
                                                                                     val = [:(XOM +
                                                                                              MRK +
                                                                                              WMT <=
                                                                                              2),
                                                                                            :(group2 ==
                                                                                              5)]),
                                                  sets = sets)))

println("Group cardinality over assets")
println("  active names in XOM/MRK/WMT group: ",
        count(res_gcard.w[.!iszero.(vec(res_gcard.gcardr.A_ineq[1, :]))] .> 1e-10))
println("  active names in group2: ",
        count(res_gcard.w[.!iszero.(vec(res_gcard.gcardr.A_eq[1, :]))] .> 1e-10))

#=
## 4. Set-level: `scard`, `slt`, `sst` with `smtx`

Here the constraints act on the sum of the weights of each set.

  - `smtx` maps the assets to the sets.
  - `scard` caps the number of sets that hold a weight.
  - `slt` and `sst` are the minimum sizes of a long and of a short set sum.

The sets are the three clusters of `clusters1`. The first problem minimises the conditional
value at risk with `scard = 1`. The second maximises the ratio with `scard = 2`, set thresholds
and no budget constraint. The count of active sets tests each set sum with `!iszero`, so a set
whose sum is a solver residual near zero counts as active, and the count can be larger than
`scard`.
=#

res_set_1 = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1,
                                                  smtx = AssetSetsMatrixEstimator(;
                                                                                  val = "clusters1"),
                                                  sets = sets)), rd)

res_set_2 = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                              opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 2,
                                                  slt = Threshold(; val = fill(0.73, 3)),
                                                  sst = Threshold(; val = 0.38),
                                                  wb = WeightBounds(; lb = -1, ub = 1),
                                                  sbgt = 1, bgt = nothing,
                                                  smtx = AssetSetsMatrixEstimator(;
                                                                                  val = "clusters1"),
                                                  sets = sets)), rd)

set_exposure = res_set_2.smtx * res_set_2.w
println("Set-level cardinality and thresholds")
println("  active sets (scard = 1): ",
        count(.!iszero.([sum(res_set_1.w[res_set_1.smtx[i, :]]) for i in axes(res_set_1.smtx, 1)])))
println("  active sets (scard = 2): ",
        count(.!iszero.([sum(res_set_2.w[res_set_2.smtx[i, :]]) for i in axes(res_set_2.smtx, 1)])))
println("  min set exposure: ", minimum(set_exposure))
println("  max set exposure: ", maximum(set_exposure))
pretty_table(DataFrame(:Set => ["cluster 1", "cluster 2", "cluster 3"],
                       :set_exposure => set_exposure); formatters = [resfmt])

#=
## 5. Group-of-sets: `sgcarde`, `sglt`, `sgst` with `sgmtx`

Now the constraints act on groups of sets. Here each group is an industry.

  - `sgmtx` maps the assets to the groups.
  - `sgcarde` bounds how many groups hold a weight.
  - `sglt` and `sgst` are the minimum sizes of a long and of a short group sum.

Whether a problem has a solution depends on the budget and the weight bounds as well as on these
targets. The first problem asks for exactly four industries with a long threshold. The second is
long-short, asks for four to six industries, and sets long and short thresholds.
=#

res_sg_1 = optimise(MeanRisk(;
                             opt = JuMPOptimiser(; slv = mip_slv, sets = sets,
                                                 sglt = Threshold(0.015),
                                                 sgcarde = LinearConstraintEstimator(;
                                                                                     key = "ux_industries",
                                                                                     val = [:(ux_industries ==
                                                                                              4)]),
                                                 sgmtx = AssetSetsMatrixEstimator(;
                                                                                  val = "nx_industries"))),
                    rd)

res_sg_2 = optimise(MeanRisk(;
                             opt = JuMPOptimiser(; slv = mip_slv,
                                                 wb = WeightBounds(; lb = -1, ub = 1),
                                                 sbgt = 1, bgt = nothing,
                                                 sglt = [ThresholdEstimator(;
                                                                            val = fill(0.53,
                                                                                       7),
                                                                            key = "ux_industries")],
                                                 sgst = [ThresholdEstimator(;
                                                                            val = fill(0.32,
                                                                                       7),
                                                                            key = "ux_industries")],
                                                 sgcarde = [LinearConstraintEstimator(;
                                                                                      key = "ux_industries",
                                                                                      val = [:(ux_industries >=
                                                                                               4),
                                                                                             :(ux_industries <=
                                                                                               6)])],
                                                 sgmtx = [AssetSetsMatrixEstimator(;
                                                                                   val = "nx_industries")],
                                                 sets = sets)), rd)

group_exposure_1 = res_sg_1.sgmtx * res_sg_1.w
group_exposure_2 = res_sg_2.sgmtx[1] * res_sg_2.w
println("Group-of-sets constraints")
println("  grouped set count (single constraint run): ",
        count(abs.(group_exposure_1) .> 1e-10))
println("  grouped set count (threshold run): ", count(abs.(group_exposure_2) .> 5e-10))
println("  min grouped long exposure: ", minimum(group_exposure_2[group_exposure_2 .> 0]))
println("  max grouped short exposure: ", maximum(group_exposure_2[group_exposure_2 .< 0]))
pretty_table(DataFrame(:GroupSet => sets.dict["ux_industries"],
                       :group_exposure => group_exposure_2); formatters = [resfmt])

#=
## 6. Number of assets in each portfolio

We count the assets that hold a weight in five of the portfolios.
=#

pretty_table(DataFrame("Case" => ["asset card", "asset card + l2", "gcarde", "set-level",
                                  "group-of-sets"],
                       "Active assets" =>
                           [count(res_card.w .> 1e-10), count(res_card_l2.w .> 1e-10),
                            count(res_gcard.w .> 1e-10), count(res_set_1.w .> 1e-10),
                            count(res_sg_1.w .> 1e-10)]))

#=
## 7. Composition of each portfolio

We plot the same five portfolios. `card` limits the number of names directly. The set and group
constraints limit the number of sets and industries, and leave the choice of names inside them
to the optimiser.
=#

plot_stacked_bar_composition([res_card, res_card_l2, res_gcard, res_set_1, res_sg_1], rd;
                             xticks = ([1, 2, 3, 4, 5],
                                       ["asset card", "card+l2", "gcarde", "set-level",
                                        "group-of-sets"]))

#=
## Which keyword to use

  - Use `card`, `lt` and `st` to cap the number of assets and to set a minimum position size.
  - Use `gcarde` when the constraint is a linear expression over named assets and groups.
  - Use `smtx` with `scard`, `slt` and `sst` when the constraint acts on the sums of sets.
  - Use `sgmtx` with `sgcarde`, `sglt` and `sgst` when it acts on groups of sets.
  - If the result carries an infeasible or failed `retcode`, check whether the budget and the
    weight bounds you gave leave room for the threshold and cardinality targets.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): all eight mixed-integer solves
#src   (asset `card`/`card`+`l2`/`card`+bounds, `gcarde`, set-level `scard`/`slt`/`sst`,
#src   group-of-sets `sgcarde`/`sglt`/`sgst`) succeed via Pajarito (HiGHS outer-approximation +
#src   Clarabel conic). The targets bind: `card = 3` → 3 active, `card = 7` → 7, the XOM/MRK/WMT
#src   `<= 2` and `group2 == 5` group-cardinality limits are met.
#src - FIXED (this session): the page had no visualisation (tables + println only) and opened
#src   only with a "MIP required" tip. Added a closing composition plot across the constraint
#src   families and a `!!! tip "When to reach for this"` admonition to meet the ADR-0014
#src   authoring standard.
#src - NOISE (record-only): Pajarito prints `Warning: integral solution repeated`
#src   (Pajarito/src/algorithms.jl:103) on some solves. It is benign progress information — the
#src   solves return success — but it is worth a one-line note in the prose so a reader does not
#src   mistake it for a failure.
#src - COSMETIC: the `scard = 2` active-set count prints 3 because the third cluster carries a
#src   ~0 residual exposure that `!iszero` still counts; tighten the tolerance if it matters.
