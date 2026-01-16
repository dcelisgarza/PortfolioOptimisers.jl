The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimiser.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/06_Multiple_Risk_Measures.jl"
```

# Example 6: Multiple risk measures

This example shows how to use multiple risk measures.

````@example 06_Multiple_Risk_Measures
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
mipresfmt = (v, i, j) -> begin
    if j ∈ (1, 2, 3)
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. ReturnsResult data

We will use the same data as the previous example.

````@example 06_Multiple_Risk_Measures
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Preparatory steps

We'll provide a vector of continuous solvers as a failsafe.

````@example 06_Multiple_Risk_Measures
using Clarabel
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.8),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.70),
              check_sol = (; allow_local = true, allow_almost = true))];
nothing #hide
````

## 3. Multiple risk measures

Some risk measures can use precomputed prior statistics which take precedence over the ones in `PriorResult`. We can make use of this to minimise the variance with different covariance matrices simultaneously.

We will also precompute the prior statistics to minimise redundant work. First lets create a vector of Variances onto which we will push the different variances. We'll use 5 variance estimators, and their equally weighted sum.

 1. Denoised covariance using the spectral algorithm.
 2. Gerber 1 covariance.
 3. Smyth Broby 1 covariance.
 4. Mutual Information covariance.
 5. Distance covariance.
 6. Equally weighted sum of all the above covariances.

For the multi risk measure optimisation, we will weigh each risk measure equally. It should give the same result as adding all covariances together, but not the same as averaging the weights of the individual optimisations.

````@example 06_Multiple_Risk_Measures
pr = prior(EmpiricalPrior(), rd.X)

ces = [PortfolioOptimisersCovariance(;
                                     mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                           dn = Denoise(;
                                                                                        alg = SpectralDenoise()))),
       PortfolioOptimisersCovariance(; ce = GerberCovariance()),
       PortfolioOptimisersCovariance(; ce = SmythBrobyCovariance(; alg = SmythBroby1())),
       PortfolioOptimisersCovariance(; ce = MutualInfoCovariance()),
       PortfolioOptimisersCovariance(; ce = DistanceCovariance())]
````

Lets define a vector of variance risk measure using each of the different covariance matrices.

````@example 06_Multiple_Risk_Measures
rs = [Variance(; sigma = cov(ce, rd.X)) for ce in ces]
all_sigmas = zeros(length(rd.nx), length(rd.nx))
for r in rs
    all_sigmas .+= r.sigma
end
push!(rs, Variance(; sigma = all_sigmas))
````

We'll minimise the variance for each individual risk measure and then we'll minimise the equally weighted sum of all risk measures.

````@example 06_Multiple_Risk_Measures
results = [optimise(MeanRisk(; r = r, opt = JuMPOptimiser(; pe = pr, slv = slv)))
           for r in rs]
mean_w = zeros(length(results[1].w))
for res in results[1:5]
    mean_w .+= res.w
end
mean_w ./= 5
res = optimise(MeanRisk(; r = rs, opt = JuMPOptimiser(; pe = pr, slv = slv)))

pretty_table(DataFrame(:assets => rd.nx, :denoise => results[1].w, :gerber1 => results[2].w,
                       :smyth_broby1 => results[3].w, :mutual_info => results[4].w,
                       :distance => results[5].w, :mean_w => mean_w,
                       :sum_covs => results[6].w, :multi_risk => res.w);
             formatters = [resfmt])
````

For extra credit we can do the same but maximising the ratio of return to risk.

````@example 06_Multiple_Risk_Measures
results = [optimise(MeanRisk(; r = r, obj = MaximumRatio(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv))) for r in rs]
mean_w = zeros(length(results[1].w))
for res in results[1:5]
    mean_w .+= res.w
end
mean_w ./= 5
res = optimise(MeanRisk(; r = rs, obj = MaximumRatio(),
                        opt = JuMPOptimiser(; pe = pr, slv = slv)))

pretty_table(DataFrame(:assets => rd.nx, :denoise => results[1].w, :gerber1 => results[2].w,
                       :smyth_broby1 => results[3].w, :mutual_info => results[4].w,
                       :distance => results[5].w, :mean_w => mean_w,
                       :sum_covs => results[6].w, :multi_risk => res.w);
             formatters = [resfmt])
````

* * *

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
