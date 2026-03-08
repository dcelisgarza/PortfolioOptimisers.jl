The source files for all examples can be found in [/examples](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../examples/11_HyperparameterTuning.jl"
```

# Example 11: Hyper parameter tuning

Hyper parameter tuning is a powerful technique to choose parameters based on their performance on test folds. In this example, we will showcase the two implemented approaches implemented in PortfolioOptimisers.jl.

````@example 11_HyperparameterTuning
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

## 1. Setting up

For this example, we will use 5 years of daily data. This is so that we have enough data to perform cross validation on significant amounts of data for both training and testing.

Cross validation cannot have precomputed values like we have done in previous examples. This is because the training and testing sets are generated on the fly, and the performance metrics are computed based on the results of the optimization on these sets.

````@example 11_HyperparameterTuning
using CSV, TimeSeries, DataFrames, Clarabel, Statistics, StableRNGs, Distributions

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 5):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)

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

## 2. Hyper parameter tuning

For this tutorial we will use the [`Stacking`]-(@ref) estimator, but the hyper parameter tuning works for every optimisation estimator. All you need is to use the right propertynames and indexing to access the parameters you want to tune. They use Julia's built-in parsing to create the lenses used by [`Accessors.jl`](https://github.com/JuliaObjects/Accessors.jl) to update the immutable estimators.

The parameter tuning uses a scoring function, a scoring metric (a risk measure), and an appropriate cross validation estimator. Only estimators which have a 1 to 1 ratio of training to test sets can be used, so only [`KFold`](@ref) and [`WalkForwardEstimator`]-(@ref) and their results can be used. This may be expanded in the future, using a similar technique for choosing the best path similarly to how it's done for the [`NestedClusters`]-(@ref) and [`Stacking`]-(@ref) estimators.

````@example 11_HyperparameterTuning
opt = JuMPOptimiser(; slv = slv)
r = MeanReturnRiskRatio(; rk = LowOrderMoment(; alg = SecondMoment()))
st = Stacking(; opti = [MeanRisk(; opt = opt), RiskBudgeting(; opt = opt)],
              opto = MeanRisk(; opt = opt))
````

### 2.1 Grid cross validation search

[`GridSearchCrossValidation`](@ref) performs an exhaustive search over a specified parameter grid via [`search_cross_validation`](@ref). It evaluates the performance of each combination of parameters and selects the best one based on how each point in the grid performs on the test folds.

The parameter grid can be specified as a vector of pairs where the first item is the string representation of the parameter to modify, the second is a vector with the range to try. Alternatively one can use a dictionary where the items are these pairs. The function will compute the product of the grid to create the full parameter grid. When using a dictionary, the order of the parameters is not guaranteed, but this makes no difference to the grid search, it does for the randomised search, so if using the latter use an `OrderedDict` from [`OrderedCollections`](https://github.com/JuliaCollections/OrderedCollections.jl) or a vector of vectors instead.

It is possible to provide a vector of grids, where each grid will be computed independently and then concatenated into a single search space. This allows for the specification of multiple different grids simultaneously.

Due to the typing system, if using a vector of vectors you have to call [`concrete_typed_array`](@ref) to ensure the correct type is inferred, alternatively use a vector of dictionaries.

Here we will search three grids, the final score will reflect the best performing parameter combination among all grids.

````@example 11_HyperparameterTuning
p = concrete_typed_array([["opti[2].opt.l1" => range(; start = 0.0005, stop = 0.0008,
                                                     length = 3),
                           "opti[1].opt.l2" => range(; start = 0.0004, stop = 0.0007,
                                                     length = 3)],
                          ["opti[1].opt.l2" => range(; start = 0.0004, stop = 0.0007,
                                                     length = 3)],
                          ["opti[2].opt.l1" => range(; start = 0.0009, stop = 0.0012,
                                                     length = 3)],
                          ["opti[2]" => [MeanRisk(; opt = opt, obj = MaximumUtility()),
                                         MeanRisk(; opt = opt, obj = MaximumRatio())]]])
gs_cv = GridSearchCrossValidation(p; r = r)
````

Now we can run the grid search cross validation. The result returns the best optimiser, the matrix of test scores, an optional matrix of training scores, the lens and value grids of the searched parameters (in vector form), and the index of the best performing parameters. The number of points in the lens and value grids is equal to the sum of the products of each grid, `sum([3x3, 3x1, 3x1, 2x1]) == 17`.

````@example 11_HyperparameterTuning
gs_res1 = search_cross_validation(st, gs_cv, rd)
````

We can view the best indices and lenses, and that they match the chosen optimiser.

````@example 11_HyperparameterTuning
pretty_table(DataFrame("Lens" => gs_res1.lens_grid[gs_res1.idx],
                       "Value" => collect(gs_res1.val_grid[gs_res1.idx])))

for (lens, val) in zip(gs_res1.lens_grid[gs_res1.idx], gs_res1.val_grid[gs_res1.idx])
    println("$(lpad("lens:", 12)) $lens\n$(lpad("val:", 12)) $val\n$(lpad("Field value:", 12)) $(lens(gs_res1.opt))\n")
end
````

### 2.2 Randomised cross validation search

[`RandomisedSearchCrossValidation`](@ref) performs an randomised search over a sampled parameter grid via [`search_cross_validation`](@ref). It can take the same type of grid as [`GridSearchCrossValidation`](@ref), in which case the parameters are sampled without replacement. It can also take a subtype of [`Distributions.Distribution`](https://juliastats.org/Distributions.jl/latest/types/#Distributions) instead of a vector of values, in which case any parameters given as a vector will be sampled with replacement. The property `n_iter` defines the number of samples to draw from the vectors or distributions. When sampling without replacement, the sampling is performed up until the list of candidates is exhausted, so the maximum number of samples drawn from a list is `min(n_iter, length(list))`. After the parameters have been sampled, a grid search cross validation is performed. It's also possible to provide a vector of grids, which works exactly like [`GridSearchCrossValidation`](@ref).

It is important to note that sampling uses the random state, so the order of the parameters will affect the sampling of distributions and sets of values. To ensure reproducibility, use an ordered dictionary or a vector for each grid, and don't change the order of parameters in each grid, or the order of the grids.

#### 2.2.1 Sampling from a predefined parameter space

First let's sample from the predefined parameter space. This will only sample two parameters from each grid, because all parameters were given as lists, the sampling is without replacement so parameters cannot be sampled twice.

````@example 11_HyperparameterTuning
rs_cv1 = RandomisedSearchCrossValidation(p; rng = StableRNG(42), r = r, n_iter = 2)
````

As you can see the result contains `sum([2x2, 2x1, 2x1, 2x1]) == 10` gridpoints.

````@example 11_HyperparameterTuning
rs_res1 = search_cross_validation(st, rs_cv1, rd)
````

We can view the best indices and lenses, and that they match the chosen optimiser.

````@example 11_HyperparameterTuning
pretty_table(DataFrame("Lens" => rs_res1.lens_grid[rs_res1.idx],
                       "Value" => collect(rs_res1.val_grid[rs_res1.idx])))

for (lens, val) in zip(rs_res1.lens_grid[rs_res1.idx], rs_res1.val_grid[rs_res1.idx])
    println("$(lpad("lens:", 12)) $lens\n$(lpad("val:", 12)) $val\n$(lpad("Field value:", 12)) $(lens(rs_res1.opt))\n")
end
````

#### 2.2.2 Sampling from a distribution

Now let's sample from a combination of the predefined parameter space and a distribution. We will sample 5 parameters from each. We don't need [`concrete_typed_array`](@ref) in [`RandomisedSearchCrossValidation`](@ref) as due to the need to accommodate `Distributions.Distribution` the checks cannot be performed at compile time, so they are performed at runtime.

````@example 11_HyperparameterTuning
p = [["opti[2].opt.l1" => range(; start = 0.0005, stop = 0.0008, length = 3),
      "opti[1].opt.l2" => LogUniform(0.0003, 0.1)],
     ["opti[1].opt.l2" => LogUniform(0.001, 0.1)],
     ["opti[2].opt.l1" => range(; start = 0.0009, stop = 0.0012, length = 3)],
     ["opti[2]" => [MeanRisk(; opt = opt, obj = MaximumUtility()),
                    MeanRisk(; opt = opt, obj = MaximumRatio())]]]

rs_cv2 = RandomisedSearchCrossValidation(p; rng = StableRNG(42), r = r, n_iter = 5)
````

The number of grid points is now `sum([3x5, 5x1, 3x1, 2x1]) = 25`, this is because vectors in a grid with a distribution are sampled with replacement (so `n_iter` samples can be taken from a vector whose length is `< n_iter`), and for vectors which are not in the same grid as a distribution sampling is done without replacement until the set is exhausted, so the number of samples is `min(n_iter, length(list))`.

````@example 11_HyperparameterTuning
rs_res2 = search_cross_validation(st, rs_cv2, rd)
````

We can view the best indices and lenses, and that they match the chosen optimiser.

````@example 11_HyperparameterTuning
pretty_table(DataFrame("Lens" => rs_res2.lens_grid[rs_res2.idx],
                       "Value" => collect(rs_res2.val_grid[rs_res2.idx])))

for (lens, val) in zip(rs_res2.lens_grid[rs_res2.idx], rs_res2.val_grid[rs_res2.idx])
    println("$(lpad("lens:", 12)) $lens\n$(lpad("val:", 12)) $val\n$(lpad("Field value:", 12)) $(lens(rs_res2.opt))\n")
end
````

The hyperparameter tuning can be used on any non finite optimisation estimator. In the future it will also be possible to provide a pipeline which will also allow users to tune pre-selection criteria.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
