The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/4_Pareto_Surface.jl"
```

# Example 4: Pareto surface

This example kicks up the complexity a couple of notches. We will introduce a new optimisation estimator, `NearOptimalCentering` optimiser.

````@example 4_Pareto_Surface
using PortfolioOptimisers, PrettyTables
# Format for pretty tables.
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
nothing #hide
````

## 1. ReturnsResult data

We will use the same data as the previous example.

````@example 4_Pareto_Surface
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = tsfmt)

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Preparing solvers for pareto surface

The pareto surface is a generalisation of the efficient frontier, in fact, we can even think of hypersurfaces if we provide more parameters, but that would be difficult to visualise, so we will stick to a 2D surface in 3D space.

We'll provide a vector of solvers because the optimisation type we'll be using is more complex, and will contain various constraints.

````@example 4_Pareto_Surface
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
nothing #hide
````

## 3. High order prior statistics

We will once again precompute the prior statistics because otherwise they'd have to be recomputed a few times.

We will be using high order risk measures, so we need to compute high order moments, we can do this with a `HighOrderPriorEstimator`, which needs a prior estimator that computes low order moments. Since we are only using a year of data, we will denoise our positive definite matrices by eliminating the eigenvalues corresponding to random noise. Denoising the non-positive definite matrix for the data we're using creates a negative square root, so we will not denoise it.

Note how many options this estimator contains.

````@example 4_Pareto_Surface
de = Denoise(; alg = SpectralDenoise(;))
mp = DefaultMatrixProcessing(; denoise = de)
pe = HighOrderPriorEstimator(;
                             # Prior estimator for low order moments
                             pe = EmpiricalPrior(;
                                                 ce = PortfolioOptimisersCovariance(;
                                                                                    mp = mp)),
                             # Estimator for cokurtosis
                             kte = Cokurtosis(; mp = mp),
                             # Estimator for coskewness
                             ske = Coskewness())
````

Let's compute the prior statistics.

````@example 4_Pareto_Surface
pr = prior(pe, rd)
````

In order to generate a pareto surface/hyper-surface, we need more dimensions than we've previously explored. We can do this by adding more risk measure sweeps (and taking their product) to generate a mesh. `PortfolioOptimisers` does this internally and generally, but we will limit ourselves to two risk measures. This will generate a 2D surface which we can visualise in 3D.

We will use the square root `NegativeSkewness` and `Kurtosis`.

````@example 4_Pareto_Surface
r1 = NegativeSkewness()
r2 = Kurtosis()
````

## 4. Near optimal centering pareto surface

First we need to get the bounds of our pareto surface. We can do this in many different ways, the simplest are:

  - Minimise the risk using both risk measures simultaneously subject to optional constraints.
  - Maximise the return, utility or ratio subject to optional constraints.

We will simply maximise the risk-return ratio for both risk measures on their own with no added constraints. This will not give a complete surface, but it will give us a reasonable range of values.

The `NearOptimalCentering` estimator will not return the portfolio which satisfies the traditional `MeanRisk` constraints, but rather a portfolio which is at the centre of an analytical region (neighbourhood) around the optimal solution. The region is parametrised by binning the efficient frontier, we will use the automatic bins here, but it is possible to define them manually.

````@example 4_Pareto_Surface
# Risk-free rate of 4.2/100/252
rf = 4.2 / 100 / 252
opt = JuMPOptimiser(; pe = pr, slv = slv)
obj = MaximumRatio(; rf = rf)
opt1 = NearOptimalCentering(; r = r1, obj = obj, opt = opt)
opt2 = NearOptimalCentering(; r = r2, obj = obj, opt = opt)
````

Note the number of options in the estimator. In particular the `alg` property. Which in this case means the `NearOptimalCentering` alg will not have any external constraints applied to it.

Let's optimise the portfolios.

````@example 4_Pareto_Surface
res1 = optimise(opt1)
res2 = optimise(opt2)
````

In order to allow for multiple risk measures in optimisations, certain measures can take different parameters. In this case, `NegativeSkewness` and `Kurtosis` take the moment matrices, which are used to compute the risk measures. We can use the `factory` function to create a new risk measure with the same parameters as the original, but with the moment matrices from the prior. Other risk measures require a solver, and this function is also used in those cases.

````@example 4_Pareto_Surface
r1 = factory(r1, pr)
r2 = factory(r2, pr)
````

Let's compute the risk bounds for the pareto surface. We need to compute four risks because we have two risk measures and two optimisations. This will let us pick the lower and upper bounds for each risk measure, as we explore the pareto surface from one optimisation to the other.

````@example 4_Pareto_Surface
sk_rk1 = expected_risk(r1, res1.w, pr.X);
kt_rk1 = expected_risk(r2, res1.w, pr.X);
sk_rk2 = expected_risk(r1, res2.w, pr.X);
kt_rk2 = expected_risk(r2, res2.w, pr.X);
nothing #hide
````

We will now create new risk measures bounded by these values. We will also use factories from the get-go. The optimisation procedure prioritises the parameters in the risk measures over the ones in the prior. This lets users provide the same risk measure with different parameters in the same optimisation. We will use two ranges of 5. The total number of points in the pareto surface will be the product of the points of each range.

Since we don't know which `sk_rk1` or `sk_r2`, `kt_rk1` or `kt_rk2` is bigger or smaller, we need to use `min`, `max`.

````@example 4_Pareto_Surface
r1 = factory(NegativeSkewness(;
                              settings = RiskMeasureSettings(;
                                                             # Risk upper bounds go from the minimum to maximum risk given the optimisations.
                                                             ub = range(;
                                                                        start = min(sk_rk1,
                                                                                    sk_rk2),
                                                                        stop = max(sk_rk1,
                                                                                   sk_rk2),
                                                                        length = 5))), pr);
r2 = factory(Kurtosis(;
                                settings = RiskMeasureSettings(;
                                                               ub = range(;
                                                                          start = min(kt_rk1,
                                                                                      kt_rk2),
                                                                          stop = max(kt_rk1,
                                                                                     kt_rk2),
                                                                          length = 5))), pr);
nothing #hide
````

Now we only need to maximise the return given both risk measures. Internally, the optimisation will generate the mesh as a product of the ranges in the order in which the risk measures were provided. This also works with the `MeanRisk` estimator, in fact, `NearOptimalCentering` uses it internally.

Since we are using an unconstrained `NearOptimalCentering`, the risk bound constraints will not be satisfied by the solution. If we wish to satisfy them, we can provide `alg = ConstrainedNearOptimalCentering()`, but would also make the optimisations harder, which may cause them to fail.

````@example 4_Pareto_Surface
opt3 = NearOptimalCentering(; r = [r1, r2], obj = MaximumReturn(), opt = opt)
````

See how `r` is a vector of risk measures with populated properties. We can now optimise the portfolios.

````@example 4_Pareto_Surface
res3 = optimise(opt3)
````

As expected, there are `5 × 5 = 25` solutions. Thankfully there are no warnings about failed optimisations, so there is no need to check the solutions.

The `NearOptimalCentering` estimator contains various return codes because it may need to compute some `MeanRisk` optimisations, it has a `retcode` which summarises whether all other optimisations succeeded. We can check this to make sure it was a success.

````@example 4_Pareto_Surface
isa(res3.retcode, OptimisationSuccess)
````

## 5. Visualising the pareto surface

Let's view how the weights evolve along the pareto surface.

````@example 4_Pareto_Surface
using StatsPlots, GraphRecipes
plot_stacked_area_composition(res3.w, rd.nx)
````

Now we can view the pareto surface. For the z-axis and colourbar, we will use the conditional drawdown at risk to return ratio.

````@example 4_Pareto_Surface
plot_measures(res3.w, pr; x = r1, y = r2,
              z = RatioRiskMeasure(; rk = ConditionalDrawdownatRisk(),
                                   rt = ArithmeticReturn(), rf = rf),
              c = RatioRiskMeasure(; rk = ConditionalDrawdownatRisk(),
                                   rt = ArithmeticReturn(), rf = rf),
              title = "Pareto Surface", xlabel = "Sqrt NSkew", ylabel = "Sqrt Kurt",
              zlabel = "CDaR/Return")
````

We can view it in 2D as well.

````@example 4_Pareto_Surface
gr()
plot_measures(res3.w, pr; x = r1, y = r2,
              c = RatioRiskMeasure(; rk = ConditionalDrawdownatRisk(),
                                   rt = ArithmeticReturn(), rf = rf),
              title = "Pareto Front", xlabel = "Sqrt NSkew", ylabel = "Sqrt Kurt",
              colorbar_title = "\n\nCDaR/Return", right_margin = 8Plots.mm)
````

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
