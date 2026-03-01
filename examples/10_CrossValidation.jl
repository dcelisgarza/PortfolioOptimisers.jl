#=
# Example 10: Cross validation

Cross validation is a powerful technique to evaluate the performance of a model on unseen data. In this example, we will showcase the different cross validation methods available in PortfolioOptimisers.jl and how to use them to evaluate the performance of our portfolio optimization models.

Cross validation can be used as a standalone method to evaluate the performance of a model, or it can be used in conjunction with other techniques like hyperparameter tuning or model selection. They can also be used in [`NestedClustered`]-(@ref) and [`Stacking`]-(@ref) optimisation estimators to optimise the outer estimator on the out-of-sample performance of the inner estimators.

This example will only focus on showcasing the different cross validation methods, with examples on how to use them and what metrics can be computed. Further analysis like plots or grid searches have not been implemented yet, but are the top priority of future development.
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
#=
## 1. Setting up

For this example, we will use 5 years of daily data. This is so that we have enough data to perform cross validation on significant amounts of data for both training and testing.

Cross validation cannot have precomputed values like we have done in previous examples. This is because the training and testing sets are generated on the fly, and the performance metrics are computed based on the results of the optimization on these sets.
=#

using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 5):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
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

#=
## 2. Cross validation
### 2.1 KFold

The simplest form of cross validation is KFold. This method splits the data into K folds, and then iteratively trains on K-1 folds and tests on the remaining fold. This process is repeated K times, with each fold being used as the test set once.

The [KFold](@ref) indices can be generated independently of the optimisation. Lets say we want to perform 5-fold cross validation, this works out to be roughly one per year.
=#

kfold = KFold(; n = 5)

#=
For demonstration purposes we can generate the splits using the [`split`]-(@ref) method. This is not necessary as the cross validation will generate them internally.
=#

kfold_res = split(kfold, rd)