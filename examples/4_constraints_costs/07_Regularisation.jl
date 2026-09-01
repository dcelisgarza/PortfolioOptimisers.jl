#=
# Regularisation

This example shows one of the simplest ways to improve the robustness of portfolios, regularisation penalties.

Section 2 states every coefficient as a number. Section 3 puts a **Calibration Rule** in the same
slots, which computes the number from the sample in front of it, and it adds the three norm
ceilings `l2c`, `lpc` and `linfc` that bound a norm rather than price one.
=#
using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
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
## 1. Setting up

We will use the same data as the previous example.
=#

using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
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

The optimal regularisation penalty value depends on the data, the investor preferences, and type of regularisation. The specific choice of penalty value is so volatile that it can only be estimated via grid search cross-validation or similar techniques, but the "optimal" (to some definition of optimal) value will also change over time as the market conditions change. Therefore, we will simply show how to set up and solve a regularised portfolio optimisation problem, without attempting to find the optimal penalty value.

We will use the same small penalty for all regularisations to illustrate how they differ.

  - L1 regularisation (also known as Lasso regularisation) adds a penalty proportional to the sum of the absolute values of the portfolio weights. This encourages sparsity in the portfolio, leading to fewer assets being selected.
  - L2 regularisation (also known as Ridge regularisation) adds a penalty proportional to the sum of the squares of the portfolio weights. This discourages large weights and promotes diversification.
  - Lp regularisation via [`LpRegularisation`](@ref) adds a penalty proportional to the p-norm of the portfolio weights, where `p > 1` is a positive real number.
  - L-Inf regularisation adds a penalty proportional to the maximum absolute value of the portfolio weights. This limits the influence of any single asset in the portfolio.

### 2.1 Efficient frontier
=#

opts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50)))),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, l1 = 4e-4),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, l2 = L2Regularisation(; val = 4e-4)),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, lp = LpRegularisation(; p = 5, val = 4e-4)),#
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      ret = ArithmeticReturn(;
                                             settings = JuMPReturnsSettings(;
                                                                            lb = Frontier(;
                                                                                          N = 50))),
                      bgt = 1, linf = 4e-4)]
nocs = [MeanRisk(; opt = opt) for opt in opts]
ress = optimise.(nocs)

#=
Let's plot the efficient frontiers.
=#
using StatsPlots, GraphRecipes

r = Variance()
# No regularisation portfolio weights.
plot_stacked_area_composition(ress[1].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "No regularisation", legend = :outerright))
# No regularisation frontier.
plot_measures(ress[1].w, pr; x = r, y = ExpectedReturn(; rt = ress[1].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "No regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

# L1 regularisation portfolio weights. As expected, the portfolio is sparsified, with fewer assets with non-zero weight.
plot_stacked_area_composition(ress[2].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L1 regularisation", legend = :outerright))
# L1 regularisation frontier. The sparsification makes the pareto front non-smooth.
plot_measures(ress[2].w, pr; x = r, y = ExpectedReturn(; rt = ress[2].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L1 regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

# L2 regularisation portfolio weights. Even values of p-norms smooth out the weights, leading to more diversified portfolios. The higher the value, the more highly penalised larger deviations from the mean weight become. This is similar to how moments of even order behave.
plot_stacked_area_composition(ress[3].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L2 regularisation", legend = :outerright))
# L2 regularisation frontier.
plot_measures(ress[3].w, pr; x = r, y = ExpectedReturn(; rt = ress[3].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L2 regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

# Lp regularisation portfolio weights. The higher the value of p, the closer the behaviour is to L-Inf regularisation, where the maximum absolute weight is penalised. This leads to portfolios where all weights are more similar in magnitude, but does not smear the negative weights into positive values like the L2 norm.
plot_stacked_area_composition(ress[4].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "Lp (p = 5) regularisation",
                                        legend = :outerright))
# Lp regularisation frontier.
plot_measures(ress[4].w, pr; x = r, y = ExpectedReturn(; rt = ress[4].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "Lp (p = 5) regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

# L-Inf regularisation portfolio weights.
plot_stacked_area_composition(ress[5].w, rd.nx;
                              kwargs = (; xlabel = "Portfolios", ylabel = "Weight",
                                        title = "L-Inf regularisation",
                                        legend = :outerright))
# L-Inf regularisation frontier.
plot_measures(ress[5].w, pr; x = r, y = ExpectedReturn(; rt = ress[5].ret),
              c = ExpectedReturnRiskRatio(; rt = ress[1].ret, rk = r, rf = 4.2 / 100 / 252),
              title = "L-Inf regularisation", xlabel = "Variance",
              ylabel = "Arithmetic Return", colorbar_title = "\nRisk/Return Ratio",
              right_margin = 6Plots.mm)

#=
### 2.2 Minimum risk portfolios

Lets view only the minimum risk portfolios for each regularisation to get more insight into what regularisation does.
=#

opts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1),# no regularisation
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l1 = 4e-4),# L1 regularisation
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, l2 = L2Regularisation(; val = 4e-4)),# L2 regularisation
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, lp = LpRegularisation(; p = 5, val = 4e-4)),# Lp regularisation with p = 5
        JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                      bgt = 1, linf = 4e-4)]# L-Inf regularisation
nocs = [MeanRisk(; opt = opt) for opt in opts]

ress = optimise.(nocs)
pretty_table(DataFrame(:Assets => rd.nx, :No_Reg => ress[1].w, :L1 => ress[2].w,
                       :L2 => ress[3].w, :L5 => ress[4].w, :LInf => ress[5].w);
             formatters = [resfmt], summary_rows = [summary_row],
             summary_row_labels = ["# Eff. Assets"])

#=
The effect of each regularisation depends on the relative values of the objective function with respect to the value of the relevant norm of the optimised portfolio weights multiplied by the penalty.

Generally, regularised portfolios tend to have more effective assets than unregularised ones. The number of effective assets is different to the sparsity in that it measures the concentration of weights as `1/(w ⋅ w)`, rather than counting the number of non-zero (or near zero) weights. Usually, the larger the number of effective assets, the more diversified the portfolio. Sparsity is a non-smooth measure, while the number of effective assets is smooth, so a portfolio can have higher sparsity and still have a larger number of effective assets.

It is possible to combine multiple regularisation penalties in the same optimisation problem by simultaneously specifying multiple regularisation keywords in the `JuMPOptimiser`. This can be useful to combine the benefits of different regularisations, such as sparsity and diversification, but can make the optimisation more difficult to solve and interpret.
=#

#=
## 3. A rule in place of a penalty value

Every coefficient section 2 states as a number is a **calibration slot**. A slot takes the number
itself, and it takes a **Calibration Rule**, which computes the number from the prior result of
the sample in front of it. The rule resolves where the model is built, so a cross-validation over
folds of unequal length refits the coefficient on every fold and nothing else about the optimiser
moves. The [calibration example](../5_validation_tuning/07_Calibrated_Risk_Measures.md) shows the
slot on a risk measure. This section reads all seven slots this example owns.

The four penalty coefficients are **ambiguity radii**. A norm penalty is the support function of
a ball in the dual of the penalised norm, so its coefficient is the radius of that ball and it
takes an [`AmbiguityRadiusCalibration`](@ref).

The three norm ceilings `l2c`, `lpc` and `linfc` bound a norm rather than price one, so they are
a different quantity and take a role of their own, [`NormCeilingCalibration`](@ref). A ceiling is
a diversification statement: its reciprocal is a floor on the effective number of assets.

### 3.1 One rule reads the slot it stands in

[`DualNormRadius`](@ref) returns the sampling error of the mean vector, measured in the ground
metric of the slot. The ground metric is the dual of the norm the slot penalises, so the rule
reads the slot's own key and answers a different number for every key. Every other radius rule
returns one number for every slot.
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
The `l1` and `linf` rows are the reading to take away. One rule, one sample and one confidence
level give two coefficients an order of magnitude apart, because an L1 penalty is priced in the
∞-norm of the error and an L-Inf penalty in its 1-norm. The `lp` row needs the penalty's own `p`,
because no key can name the conjugate order. The rule holds no order of its own: the penalty
site states it in a `CalibrationContext`, and the `lp5` context above stands in for that site
because the rule runs outside it here.

The slot takes the role, and the coefficient appears when the model is built. The two runs below
differ only in what stands in `l1`, and they hold the same weights.
=#

l1_rule = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1,
                        l1 = AmbiguityRadiusCalibration(; alg = DualNormRadius()))
l1_num = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, l1 = dnr(:l1, pr, nothing, nothing, CalibrationContext()))
res_rule = optimise(MeanRisk(; opt = l1_rule))
res_num = optimise(MeanRisk(; opt = l1_num))
println("largest weight difference = $(maximum(abs, res_rule.w - res_num.w))")

#=
### 3.2 Two rates, and the one the universe sets

[`RateRadius`](@ref) shrinks the ball as `c / sqrt(T)`, which is the rate a sample mean's own
error falls at. [`DimensionalRateRadius`](@ref) shrinks it at the rate the number of assets sets,
which is far slower over a wide universe: the exponent is `1 / max(N, 2)` rather than `1 / 2`.
The table reads both rules over three windows of the same record.
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
The rate radius falls by the square root of the ratio of the lengths. The dimensional radius
barely moves, and it does not fall monotonically either: over this universe the exponent is
`1 / 20`, so a fivefold record buys almost nothing, and what movement the column holds comes from
the scale the rule reads off each window rather than from the length of it.

### 3.3 The three norm ceilings

[`EffectiveAssetFloor`](@ref) is the one rule of the ceiling family. It states a fraction of the
universe to hold effective, and returns the ceiling on the norm that meets it. The order-`p`
effective number of assets is `(sum(abs.(w) .^ p))^(1 / (1 - p))`, and the ceiling is the number
that holds it at or above `fraction * N`.

The order belongs to the constraint rather than to the rule, so each site states it in a
`CalibrationContext`. A rule run outside a site needs a context that names `p`, exactly as the
`lp` radius above did.
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
One rule serves the three slots, and each site reads it against its own norm order.
=#

crole = NormCeilingCalibration(; alg = EffectiveAssetFloor(; fraction = 0.5))
copts = [JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1),# no ceiling
         JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, l2c = crole),# 2-norm ceiling
         JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, lpc = LpRegularisation(; p = 5, val = crole)),# 5-norm ceiling
         JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                       bgt = 1, linfc = crole)]# Inf-norm ceiling
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
Every ceiling meets its own order's floor, which is the `n_eff_p` column. The `n_eff_2` column is
[`number_effective_assets`](@ref), which is the order-2 reading alone, so only the `l2c` row is
read against the order its ceiling was written for. The two columns are one number on that row
and two numbers on the others, and that is a statement about which order the reading uses rather
than about a ceiling that missed.

The rule refuses a resolution whose context names no order, and the message names the three
slots that state one.
=#

try
    eaf(:l2c, pr, nothing, nothing, CalibrationContext())
catch e
    println(sprint(showerror, e))
end
