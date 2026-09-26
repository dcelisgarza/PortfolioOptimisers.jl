#=
```@meta
Description = "Regularisation in PortfolioOptimisers.jl: L1, L2, Lp and L-infinity penalties and ceilings on the weights, as numbers or as calibration rules."
```

# [Regularisation](@id example-regularisation)

This example shows regularisation, a penalty on a norm of the weights that the optimiser adds to
the objective.

Section 2 gives every penalty coefficient as a number. Section 3 replaces the number with a
calibration rule, an estimator that computes the coefficient from the sample. It also adds the
three norm ceilings, `l2c`, `lpc` and `linfc`, which bound a norm of the weights in place of a
penalty on it.
=#
using PortfolioOptimisers, PrettyTables
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
mmtfmt = (v, i, j) -> begin
    if i == j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
summary_row = (data, j) -> begin
    if j == 1
        return "N/A"
    else
        return number_effective_assets(data[:, j])
    end
end

#=
## 1. Data

The data is one year of S&P 500 prices. We fit an empirical prior, and give the optimiser seven
Clarabel settings to try in turn.
=#

using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

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
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.8),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.75),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.70),
              check_sol = (; allow_local = true, allow_almost = true))];

#=
## 2. Regularised portfolios

The coefficient that suits a problem depends on the data and on the norm, and it changes as the
data changes. You can search for it with cross-validation, or compute it from the sample with a
calibration rule, as section 3 shows. This section sets up and solves regularised problems, and
does not search for the best value.

We use the same coefficient, `4e-4`, for every regularisation, so you can compare them.

  - L1 regularisation, also called Lasso, adds a penalty proportional to the sum of the absolute
    values of the weights. It pushes small weights to zero, so the portfolio holds fewer assets.
  - L2 regularisation, [`L2Regularisation`](@ref), adds a penalty proportional to the 2-norm of
    the weights, the square root of the sum of their squares. It penalises large weights and
    spreads the portfolio over more assets. Other values of its `alg` penalise the square of the
    2-norm, which is the Ridge penalty.
  - Lp regularisation, [`LpRegularisation`](@ref), adds a penalty proportional to the p-norm of
    the weights, for a finite `p` greater than 1.
  - L-Inf regularisation adds a penalty proportional to the largest absolute weight. It limits
    the size of the largest position.

### 2.1 Efficient frontier

For each regularisation we compute an efficient frontier of 50 portfolios, with weights between
`-1` and `1`, a short budget of 1 and a budget of 1.
=#

opts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, l1 = 4e-4),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, l2 = L2Regularisation(; val = 4e-4)),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, lp = LpRegularisation(; p = 5, val = 4e-4)),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, linf = 4e-4)]
nocs = [MeanRisk(; opt = opt) for opt in opts]
ress = optimise.(nocs)

#=
For each regularisation we plot the weights along the frontier, and the frontier itself in the
plane of variance and return, coloured by the ratio of return to risk.
=#
using StatsPlots, GraphRecipes

r = Variance()
# Weights along the frontier with no regularisation.
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "No regularisation", legend = :outerright))
# Frontier with no regularisation.
plot_measures(ress[1].w, pr; x = r, y = ExpectedReturn(; rt = ress[1].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "No regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

# Weights with L1 regularisation. Fewer assets hold a weight than with no regularisation.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L1 regularisation", legend = :outerright))
# Frontier with L1 regularisation. It is not smooth, because assets enter and leave the portfolio along it.
plot_measures(ress[2].w, pr; x = r, y = ExpectedReturn(; rt = ress[2].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L1 regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

# Weights with L2 regularisation. The budget fixes the sum of the weights, so the 2-norm is smallest when every weight equals the mean weight. The penalty pulls the weights toward it and spreads the portfolio over more assets.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L2 regularisation", legend = :outerright))
# Frontier with L2 regularisation.
plot_measures(ress[3].w, pr; x = r, y = ExpectedReturn(; rt = ress[3].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L2 regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

# Weights with Lp regularisation, p = 5. As p grows, the p-norm comes closer to the largest absolute weight, which is the norm of the L-Inf penalty.
plot_stacked_area_composition(ress[4].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Lp (p = 5) regularisation",
                                        legend = :outerright))
# Frontier with Lp regularisation, p = 5.
plot_measures(ress[4].w, pr; x = r, y = ExpectedReturn(; rt = ress[4].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "Lp (p = 5) regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

# Weights with L-Inf regularisation.
plot_stacked_area_composition(ress[5].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L-Inf regularisation",
                                        legend = :outerright))
# Frontier with L-Inf regularisation.
plot_measures(ress[5].w, pr; x = r, y = ExpectedReturn(; rt = ress[5].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L-Inf regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nReturn/Risk Ratio",
              right_margin = 6Plots.mm)

#=
### 2.2 Minimum risk portfolios

We now solve only the minimum risk portfolio of each regularisation. The table prints the
weights, and its last row is the number of effective assets of each portfolio.
=#

opts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l1 = 4e-4),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l2 = L2Regularisation(; val = 4e-4)),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, lp = LpRegularisation(; p = 5, val = 4e-4)),
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, linf = 4e-4)]
nocs = [MeanRisk(; opt = opt) for opt in opts]

ress = optimise.(nocs)
pretty_table(DataFrame(:Assets => rd.nx, :No_Reg => ress[1].w, :L1 => ress[2].w,
                       :L2 => ress[3].w, :L5 => ress[4].w, :LInf => ress[5].w);
             formatters = [resfmt], summary_rows = [summary_row],
             summary_row_labels = ["Effective assets"])

#=
Against `No_Reg`, the L1 penalty sets several weights to zero. The L2, Lp and L-Inf penalties
make the weights more even, and the last row shows that each of them gives more effective assets
than L1. How much a penalty changes the weights depends on its size against
the rest of the objective, which here is the risk. The penalty is the coefficient times the
norm of the weights.

The number of effective assets, [`number_effective_assets`](@ref), is `1/(w ⋅ w)`. It measures
how concentrated the weights are, and it is not a count of the weights that are not zero. A
portfolio that holds fewer assets can still have more effective assets, if its weights are more
even. A larger number of effective assets means a less concentrated portfolio.

To combine penalties, give more than one of the regularisation keywords to the
`JuMPOptimiser`. For example, an L1 and an L2 penalty together ask for fewer assets and for more
even weights. Each penalty you add is one more coefficient to choose.
=#

#=
## 3. A rule in place of a penalty value

Each penalty coefficient of section 2 can be a calibration rule in place of a number. A
calibration rule is an estimator that computes the coefficient from the prior of the sample it
gets. `l1` and `linf` take the rule directly, and [`L2Regularisation`](@ref) and
[`LpRegularisation`](@ref) take it in their `val`. The optimiser computes the coefficient when
it builds the model. In a cross-validation, each fold therefore gets a coefficient from its own
training sample, and the rest of the optimiser stays the same. The [calibration
example](@ref example-calibrated-risk-measures-a-rule-in-place-of-a-number) uses a rule in a risk measure.
This section covers the four penalties and the three norm ceilings.

The four penalty coefficients are ambiguity radii. Take a ball of return distributions around
the sample, with distance measured in the dual of the penalised norm. The worst case of the
objective over that ball is the objective plus a penalty on the norm of the weights, and the
coefficient of the penalty is the radius of the ball. The penalties therefore take the rules
that compute a radius.

The three norm ceilings, `l2c`, `lpc` and `linfc`, bound a norm of the weights in place of a
penalty on it. A ceiling is not a radius, so the ceilings take a different family of rules. A
ceiling on a norm of the weights sets a floor on an effective number of assets, as section 3.3
shows.

### 3.1 A rule whose value depends on the penalty

[`DualNormRadius`](@ref) returns the sampling error of the mean returns, measured in the dual of
the norm that the penalty uses. It therefore gives a different number for each penalty, where
the other radius rules of the library give the same number for every penalty. We call the rule
with the key of each penalty and print the four radii.
=#

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

dnr = DualNormRadius(; confidence = 0.95)
lp5 = CalibrationContext(; p = 5)

radius_table = DataFrame(:slot => ["l1", "linf", "l2, val", "lp, val, p = 5"],
                         :penalised_norm => ["1", "Inf", "2", "5"],
                         :ground_metric => ["Inf", "1", "2", "1.25"],
                         :radius => [dnr(:l1, pr, nothing, nothing, CalibrationContext()),
                                     dnr(:linf, pr, nothing, nothing, CalibrationContext()),
                                     dnr(:l2reg_val, pr, nothing, nothing, CalibrationContext()),
                                     dnr(:lpreg_val, pr, nothing, nothing, lp5)])
pretty_table(radius_table; formatters = [numfmt])

#=
Compare the `l1` and `linf` rows. The rule, the sample and the confidence level are the same,
but the two coefficients are about an order of magnitude apart. The L1 penalty uses the ∞-norm
of the error, and the L-Inf penalty uses its 1-norm. The `lp` row needs the `p` of the penalty,
because its dual order `q = p / (p - 1)` depends on it. When the optimiser builds the model, it
gives the rule this `p` in a [`CalibrationContext`](@ref). Here we call the rule ourselves, so
we pass `lp5`, a context with `p = 5`.

Next we give the rule to `l1` of one optimiser, and the number that it computes to `l1` of a
second optimiser. Both use a confidence level of 0.95, so both models get the same coefficient.
The cell prints the largest difference between the two sets of weights.
=#

l1_rule = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, l1 = DualNormRadius())
l1_num = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, l1 = dnr(:l1, pr, nothing, nothing, CalibrationContext()))
res_rule = optimise(MeanRisk(; opt = l1_rule))
res_num = optimise(MeanRisk(; opt = l1_num))
println("largest weight difference = $(maximum(abs, res_rule.w - res_num.w))")

#=
### 3.2 Radii that fall as the sample grows

[`RateRadius`](@ref) returns `c / sqrt(T)`, so the radius falls at the rate of the error of a
sample mean. [`DimensionalRateRadius`](@ref) falls as `T` to the power `-1 / max(N, 2)`, where
`N` is the number of assets. Over a wide universe this is much slower than the power `-1 / 2`.
We compute both rules on three windows of the same price history, of 252, 630 and 1260 days.
=#

X_all = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)

rate_table = DataFrame()
for T in (252, 630, 1260)
    pr_T = prior(EmpiricalPrior(), prices_to_returns(X_all[(end - T):end]))
    push!(rate_table,
          (; T = size(pr_T.X, 1), N = size(pr_T.X, 2),
           rate = RateRadius(; c = 0.02)(:l1, pr_T, nothing, nothing, CalibrationContext()),
           dimensional = DimensionalRateRadius(; confidence = 0.95)(:l1, pr_T, nothing,
                                                                    nothing,
                                                                    CalibrationContext())))
end
pretty_table(rate_table; formatters = [numfmt])

#=
The rate radius falls by the square root of the ratio of the window lengths. The dimensional
radius changes little, and it does not always fall. With 20 assets its power is `-1 / 20`, so a
window five times longer lowers it by less than 8%. The rule also multiplies by a scale that it
computes from each window, the mean standard deviation of the assets, and that scale causes the
rest of the change.

### 3.3 The three norm ceilings

[`EffectiveAssetFloor`](@ref) is the one rule for the ceilings. You give it a fraction of the
universe, and it returns the ceiling on the norm that keeps the effective number of assets at or
above `m = fraction * N`. The effective number of assets of order `p` is `(sum(abs.(w) .^ p))^(1
/ (1 - p))`. The ceiling is `m^(1 / p - 1)`, which is `1 / sqrt(m)` for `p = 2` and `1 / m` for
`p = Inf`.

The order `p` comes from the constraint, not from the rule. `l2c` uses 2, `lpc` uses the `p` of
its [`LpRegularisation`](@ref), and `linfc` uses `Inf`. When we call the rule ourselves, we pass
the order in a `CalibrationContext`, as we did for the `lp` radius. We print the size of the
universe, the floor and the three ceilings.
=#

N = size(pr.X, 2)
println("universe = $N assets, floor = $(0.5 * N) effective assets")

eaf = EffectiveAssetFloor(; fraction = 0.5)

ceiling_table = DataFrame(:slot => ["l2c", "lpc, p = 5", "linfc"],
                          :norm_order => ["2", "5", "Inf"],
                          :ceiling =>
                              [eaf(:l2c, pr, nothing, nothing, CalibrationContext(; p = 2)),
                               eaf(:lpc, pr, nothing, nothing, CalibrationContext(; p = 5)),
                               eaf(:linfc, pr, nothing, nothing,
                                   CalibrationContext(; p = Inf))])
pretty_table(ceiling_table; formatters = [numfmt])

#=
We give the same rule to the three ceilings of three optimisers, and compare them with an
optimiser that has no ceiling. For each portfolio we print the effective number of assets of
order 2, and of the order that its ceiling uses.
=#

ceil_rule = EffectiveAssetFloor(; fraction = 0.5)
copts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1),
         JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, l2c = ceil_rule),
         JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, lpc = LpRegularisation(; p = 5, val = ceil_rule)),
         JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, linfc = ceil_rule)]
cress = [optimise(MeanRisk(; opt = opt)) for opt in copts]

n_eff_p(w, p) = isinf(p) ? inv(maximum(abs, w)) : sum(abs.(w) .^ p)^inv(1 - p)

ceiling_orders = [2, 2, 5, Inf]
effective_table = DataFrame(:ceiling => ["none", "l2c", "lpc, p = 5", "linfc"],
                            :p => ceiling_orders,
                            :n_eff_2 => [number_effective_assets(r.w) for r in cress],
                            :n_eff_p =>
                                [n_eff_p(r.w, p) for (r, p) in zip(cress, ceiling_orders)])
pretty_table(effective_table; formatters = [numfmt])

#=
The `n_eff_p` column is the effective number of assets of the order that each ceiling uses.
Compare it with the floor printed above. The `n_eff_2` column is `number_effective_assets`,
which uses order 2 on every row. The two columns are equal on the first two rows, where the
order is 2, and differ on the other two. A value of `n_eff_2` below the floor on the `lpc` or
`linfc` row does not mean that the ceiling failed, because that ceiling bounds a different
order.

If the context names no order, the rule throws an error, and the message names the three
ceilings that give one.
=#

try
    eaf(:l2c, pr, nothing, nothing, CalibrationContext())
catch e
    println(sprint(showerror, e))
end
