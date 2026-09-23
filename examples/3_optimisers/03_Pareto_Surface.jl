#=
```@meta
Description = "Sweep a Pareto surface in PortfolioOptimisers.jl: trade off return against two risk measures at once with NearOptimalCentering."
```

# Pareto surface

This page extends the efficient frontier to two risk measures and the return. We compute the
surface with [`NearOptimalCentering`](@ref), which the
[efficient-frontier](02_Efficient_Frontier.md) page introduced.

!!! tip "When to reach for this"
    Reach for a Pareto surface when one frontier of risk and return is not enough, because you
    trade off *more than two* criteria at once, such as two risk measures and the return. The
    efficient frontier is a curve. With three criteria, the portfolios that no other portfolio
    beats on every criterion form a surface, and with more they form a hypersurface. Read the
    efficient-frontier example first, and use it if two criteria are enough.
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

#=
## 1. Data

We use the same data as the previous example.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

rd = prices_to_returns(X)

#=
## 2. Solvers for the Pareto surface

The optimisation on this page is harder and has more constraints, so we pass a vector of seven
solvers. The first uses the default settings, and the next six set
`max_step_fraction` from 0.95 down to 0.70 in steps of 0.05.
=#

using Clarabel
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
## 3. High-order prior statistics

The risk measures on this page are of high order, so the prior needs the high-order moments.
[`HighOrderPriorEstimator`](@ref) computes them, and it takes a second prior estimator for the
low-order moments. One year of data is short, so we denoise the positive definite matrices, the
covariance and the cokurtosis. [`SpectralDenoise`](@ref) sets to zero the eigenvalues that random
noise explains. The coskewness gives a matrix that is not positive definite, and on this data its
denoised form puts a negative number under a square root, so we leave it as it is.
=#

dn = Denoise(; alg = SpectralDenoise(;))
mp = MatrixProcessing(; dn = dn)
pe = HighOrderPriorEstimator(;
                             pe = EmpiricalPrior(;
                                                 ce = PortfolioOptimisersCovariance(;
                                                                                    mp = mp)),
                             kte = Cokurtosis(; mp = mp), ske = Coskewness())

#=
We compute the prior once, because the page uses it several times.
=#

pr = prior(pe, rd)

#=
A Pareto surface needs more dimensions than a frontier. Each risk measure with a range of bounds
adds one, and the product of the ranges gives a mesh of points. The library builds the mesh for
any number of risk measures. More measures give a hypersurface, which is hard to plot, so we use
two, which give a 2D surface that we plot in 3D.

The two measures are the square root of the negative skewness, [`NegativeSkewness`](@ref), and
the square root of the kurtosis, [`Kurtosis`](@ref).
=#

r1 = NegativeSkewness()
r2 = Kurtosis()

#=
## 4. The Pareto surface with near-optimal centring

First we find the bounds of the surface. There are many ways to find them, and the two simplest
are:

  - Minimise the risk under both risk measures at once, with any constraints you choose.
  - Maximise the return, the utility or the ratio, with any constraints you choose.

We maximise the risk-return ratio under each risk measure alone, with no other constraints. This
does not give the whole surface, but it gives a useful range of values.

`NearOptimalCentering` does not return the optimal `MeanRisk` portfolio. It returns the portfolio
at the analytic centre of a region around the optimum. The size of the region comes from splitting
the efficient frontier into bins. We keep the default number of bins, the number of observations
divided by the number of assets, and you can set it with the `bins` field.
=#

rf = 4.2 / 100 / 252
opt = JuMPOptimiser(; pe = pr, slv = slv)
obj = MaximumRatio(; rf = rf)
opt1 = NearOptimalCentering(; r = r1, obj = obj, opt = opt)
opt2 = NearOptimalCentering(; r = r2, obj = obj, opt = opt)

#=
The printout lists the fields of the estimator. Its `alg` field is
`UnconstrainedNearOptimalCentering()`, so the centring step applies only the weight bounds and the
budgets of the optimiser.

We optimise the two portfolios.
=#

res1 = optimise(opt1)
res2 = optimise(opt2)

#=
`expected_risk` needs the moment matrices inside the measure, and `NegativeSkewness` and
`Kurtosis` hold none yet. `factory` returns a copy of a risk measure with those matrices taken
from the prior. For a risk measure that needs a solver, it fills that in too.
=#

r1 = factory(r1, pr)
r2 = factory(r2, pr)

#=
We compute the risk bounds of the surface. Two risk measures on two portfolios give four risks.
From them we take the lower and the upper bound of each risk measure, so the surface spans the
space between the two portfolios.
=#

sk_rk1 = expected_risk(r1, res1.w, pr.X);
kt_rk1 = expected_risk(r2, res1.w, pr.X);
sk_rk2 = expected_risk(r1, res2.w, pr.X);
kt_rk2 = expected_risk(r2, res2.w, pr.X);

#=
We build new risk measures with these bounds, and pass each one through `factory` immediately,
so that each takes its matrices from the prior. Each range has 5 values, and the surface has one
point per pair of values.

We do not know which of `sk_rk1` and `sk_rk2` is the larger, or which of `kt_rk1` and `kt_rk2`,
so we use `min` and `max`.
=#

r1 = factory(NegativeSkewness(;
                              settings = RiskMeasureSettings(;
                                                             ub = range(;
                                                                        start = min(sk_rk1,
                                                                                    sk_rk2),
                                                                        stop = max(sk_rk1,
                                                                                   sk_rk2),
                                                                        length = 5))), pr);
r2 = factory(Kurtosis(;
                      settings = RiskMeasureSettings(;
                                                     ub = range(;
                                                                start = min(kt_rk1, kt_rk2),
                                                                stop = max(kt_rk1, kt_rk2),
                                                                length = 5))), pr);
#=
We maximise the return under both risk measures. The optimisation builds the mesh as the product
of the ranges, in the order in which you give the risk measures. `MeanRisk` builds the mesh the
same way, and `NearOptimalCentering` solves `MeanRisk` problems to find its region.

Our `NearOptimalCentering` is the unconstrained variant, so its portfolios need not stay inside
the risk bounds. To keep them inside, set `alg` to [`ConstrainedNearOptimalCentering`](@ref).
The optimisations are then harder, and some of them can fail.
=#

opt3 = NearOptimalCentering(; r = [r1, r2], obj = MaximumReturn(), opt = opt)

#=
In the printout, `r` is a vector of two risk measures, each with its bounds and its moment
matrices filled in. We optimise the portfolios.
=#

res3 = optimise(opt3)

#=
The result holds `5 × 5 = 25` portfolios, one per pair of bounds.

Its `retcode` shows whether every inner `MeanRisk` problem succeeded.
=#

isa(res3.retcode, OptimisationSuccess)

#=
## 5. Visualising the Pareto surface

The stacked areas show the weights at each of the 25 points of the surface.
=#

using StatsPlots, GraphRecipes
plot_stacked_area_composition(res3.w, rd.nx)

#=
We plot the Pareto surface. The z-axis and the colour both show the ratio of the return, net of
the risk-free rate, to the conditional drawdown at risk, CDaR. The CDaR is not one of the two
optimised measures. We use it to compare the portfolios on a third risk.
=#

plot_measures(res3.w, pr; x = r1, y = r2,
              z = ExpectedReturnRiskRatio(; rk = ConditionalDrawdownatRisk(),
                                          rt = ArithmeticReturn(), rf = rf),
              c = ExpectedReturnRiskRatio(; rk = ConditionalDrawdownatRisk(),
                                          rt = ArithmeticReturn(), rf = rf),
              title = "Pareto surface", xlabel = "Square root of negative skewness",
              ylabel = "Square root of kurtosis", zlabel = "Return/CDaR")

#=
We plot the same surface in 2D, with the ratio as the colour.
=#

plot_measures(res3.w, pr; x = r1, y = r2,
              c = ExpectedReturnRiskRatio(; rk = ConditionalDrawdownatRisk(),
                                          rt = ArithmeticReturn(), rf = rf),
              title = "Pareto front", xlabel = "Square root of negative skewness",
              ylabel = "Square root of kurtosis", colorbar_title = "\n\nReturn/CDaR",
              right_margin = 8Plots.mm)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Sweep clean (ADR 0014 retrofit): the high-order prior (denoised cov + cokurtosis +
#src   coskewness), the two single-measure NOC bound optimisations, and the 5×5 = 25-point
#src   NegativeSkewness×Kurtosis NOC surface all solve, with res3.retcode an
#src   OptimisationSuccess. Both 3D surface and 2D front render. No findings.
#src - NOTE (cross-ref, not a defect): this page already exercises NearOptimalCentering as a
#src   frontier/surface engine; the dedicated 15_Near_Optimal_Centering page should focus on
#src   NOC's *neighbourhood-centering* behaviour vs plain MeanRisk rather than re-deriving a
#src   surface, to avoid overlap. Group rollup: issue #125.
