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

using CSV, TimeSeries, DataFrames, Clarabel, Statistics

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
For this tutorial we will use the basic [`MeanRisk`]-(@ref) estimator, but the cross validation works for all optimisation estimators, even when computing pareto fronts.
=#

mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))

#=
## 2. Cross validation
### 2.1 KFold

The simplest form of cross validation is KFold. This method splits the data into K folds, and then iteratively trains on K-1 folds and tests on the remaining fold. This process is repeated K times, with each fold being used as the test set once.

The [`KFold``](@ref) indices can be generated independently of the optimisation. Let's say we want to perform 5-fold cross validation, this works out to be roughly one per year.
=#

kfold = KFold(; n = 5)

#=
For demonstration purposes we can generate the splits using the [`split`]-(@ref) method. This is not necessary as the cross validation will generate them internally.
=#

kfold_res = split(kfold, rd)

show(kfold_res.train_idx)
show(kfold_res.test_idx)

#=
Let's perform the cross validation.
=#
kfold_pred = cross_val_predict(mr, rd, kfold)

#=
The result is a [`MultiPeriodPredictionResult`]-(@ref) object, which is a wrapper for a vector of [`PredictionResult`]-(@ref) objects, one for each fold. Each [`PredictionResult`]-(@ref) contains the optimisation result based on the training set, and a [`PredictionReturnsResult`]-(@ref) containing the predicted returns result of the optimised portfolio evaluated on its corresponding test set.

We can individually access the result of each fold by indexing into the `pred` field of the [`MultiPeriodPredictionResult`]-(@ref) object, but we can also directly access via the accessing the `mrd` and `mres` properties, which stand for multi-rd and multi-res. `mrd` concatenates the predicted returns into a single [`PredictionReturnsResult`]-(@ref). Since the embargo and purged sizes are zero, the timestamps of the predicted returns should be the same as the timestamps of the original returns result.
=#

println("isequal(kfold_pred.mrd.ts, rd.ts) = $(isequal(kfold_pred.mrd.ts, rd.ts))")

#=
We can also compute performance metrics (risk measures) on the predicted returns. However, we can only use risk measures that use the returns series as an input. This means [`StandardDeviation`]-(@ref), [`NegativeSkewness`]-(@ref), [`TurnoverRiskMeasure`]-(@ref), [`TrackingRiskMeasure`]-(@ref) with [`WeightsTracking`](@ref), [`Variance`]-(@ref), [`UncertaintySetVariance`]-(@ref), [`EqualRiskMeasure`]-(@ref), [`ExpectedReturn`]-(@ref) and [`ExpectedReturnRiskRatio`]-(@ref), as well as any risk measure that uses any of these cannot be used. But there are ways around this, for example:

- For the variance and standard deviation, we can use [`LowOrderMoment`]-(@ref) with the appropriate algorithms.
- For [`NegativeSkewness`]-(@ref) we can use [`HighOrderMoment`]-(@ref), or [`Skewness`]-(@ref).
- For [`ExpectedReturn`]-(@ref) and [`ExpectedReturnRiskRatio`]-(@ref) we can use [`MeanReturn`]-(@ref) and [`MeanReturnRiskRatio`]-(@ref) respectively.

Here we will compute the variance.
=#

println("KFold(5) prediction variance = $(expected_risk(LowOrderMoment(; alg = SecondMoment()), kfold_pred))")

#=
### 2.2 Combinatorial

The [`CombinatorialCrossValidation`](@ref) method generates all possible combinations of the data into training and testing sets. This method is computationally expensive, but provides a more comprehensive evaluation of the model's performance on unseen data.

There is also a way to compute the optimal number of folds and training folds given a user-defined desired training and test set lengths, as well as the relative weight between the training size and number of test paths.
=#

T = size(rd.X, 1)
target_train_size = 200
target_test_size = 70
n_folds, n_test_folds = optimal_number_folds(T, target_train_size, target_test_size)
cfold = CombinatorialCrossValidation(; n_folds = n_folds, n_test_folds = n_test_folds)

#=
Let's see the indices this produces.
=#

cfold_res = split(cfold, rd)

#=
Here we have 78 splits, each testing path split into 11 folds. This means we have 78 * 11 = 858 total folds, which is a significant increase from the 5 folds we had in KFold. This is the trade-off for having a more comprehensive evaluation of the model's performance on unseen data.

But it also means we need a way to find a good representative of the predictions in order to evaluate the out of sample performance. First let's perform the cross validation.

There is some nuance with this approach in that the splits do not represent the same number of paths, in fact there are only 66 unique paths, which can be seen from `cfold_res.path_ids`.
=#

cfold_res.path_ids

#=
We can now perform the cross validation.
=#

cfold_pred = cross_val_predict(mr, rd, cfold)

#=
We can see that there are indeed 66 predictions. Each is a valid representative of the out-of-sample performance of the model. However, for evaluating the performance, we can use a sample or the median of the predictions. The median is a good representative of the performance, as it is not affected by outliers, and it is a good measure of central tendency. We can do this with custom function, or a functor of a subtype of [`PredictionScorer`]-(@ref). We've implemented a simple one called [`NearestQuantilePrediction`]-(@ref) which takes the prediction with the nearest quantile to the desired quantile of the distribution of predictions, it defaults to the median.

We will use the risk return ratio of the variance as our performance metric. The paths are sorted according to their expected risk, return based risk measures sort them based on descending order, while true risk measures sort them in ascending order.
=#

sharpe_scorer = NearestQuantilePrediction(;
                                          r = MeanReturnRiskRatio(;
                                                                  rk = LowOrderMoment(;
                                                                                      alg = SecondMoment())))

#=
Scorer is a functor which takes a population as an input and outputs a tuple of the single prediction and the index in the population which matches the desired quantile of the distribution of predictions. In this case, we are using the mean return risk ratio with the variance as the risk measure, and we are looking for the prediction with the nearest quantile to 0.5, which is the median.
=#

median_pred_max_sharpe = sharpe_scorer(cfold_pred)

#=
The prediction `id` corresponds to the index/path id of the prediction in the population.
=#
median_pred_max_sharpe === cfold_pred.pred[median_pred_max_sharpe.id]

#=
Similarty to the KFold, the timestamps of the predicted returns should be the same as the timestamps of the original returns result, since the embargo and purged sizes are zero.
=#
isequal(median_pred_max_sharpe.mrd.ts, rd.ts)

#=
We can further verify this by computing the risk return ratio of the variance for all predictions and seeing that the prediction with a risk value closest to the median is indeed the same as the one we found with the scorer. Note that the scorer also filters out predictions whose optimisations failed, so in order to be truly rigorous we'd need to skip NaN values in the array of risks, while keeping the indices aligned, but for demonstration purposes this is sufficient.
=#

sharpe_ratios = expected_risk(MeanReturnRiskRatio(;
                                                  rk = LowOrderMoment(;
                                                                      alg = SecondMoment())),
                              cfold_pred)
argmin(abs.(sharpe_ratios .- median(sharpe_ratios))) == median_pred_max_sharpe.id

#=
We can choose any compatible risk measure as outlined above, for demonstration purposes we will now rank them based on the variance.
=#

variance_scorer = NearestQuantilePrediction(; r = LowOrderMoment(; alg = SecondMoment()))
median_pred_min_variance = variance_scorer(cfold_pred)

#=
Again the id matches the prediction with the nearest quantile to the median of the distribution of predictions.
=#
median_pred_min_variance === cfold_pred.pred[median_pred_min_variance.id]

#=
As always, the timestamps match.
=#
isequal(median_pred_min_variance.mrd.ts, rd.ts)

#=
### 2.3 WalkForward

We offer two different walkforward estimators, [`IndexWalkForward`](@ref) and [`DateWalkForward`](@ref). The former splits the data based on the number of observations, while the latter splits the data based on the timestamps, and can be used with Julia's `Dates` module to adjust periods to specific times.

The walkforward method is a more realistic evaluation of the model's performance on unseen data, as it mimics the way the model would be used in practice. It can also dynamically use the previous optimisation weights in constraints and risk measures if so desired.

#### 2.3.1 IndexWalkForward

The simpler estimator is [`IndexWalkForward`](@ref) so we will start with this one. We will use training sets of one full year and test sets of 3 months. Note that a year has roughly 252 trading days. We will again not use any purging, meaning that the test set will immediately follow the training set, and there will be no gap between them. This means that the timestamps of the predicted returns should be the same as the timestamps of the original returns result minus the first 252 entries.
=#

idx_walk_forward = IndexWalkForward(252, round(Int, 252 / 4))
idx_walk_forward_res = split(idx_walk_forward, rd)
show(idx_walk_forward_res.train_idx)
show(idx_walk_forward_res.test_idx)

#=
We can generate the prediction now.
=#
idx_walkforward_pred = cross_val_predict(mr, rd, idx_walk_forward)

#=
Let's check the timestamps.
=#

isequal(idx_walkforward_pred.mrd.ts, rd.ts[253:end])

#=
Now let's see the evolution of the weights across the different splits.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(idx_walkforward_pred.res, :w)),
                            Symbol.(1:16))); formatters = [resfmt])

#=
As we can see, the weights can evolve in a fairly volatile manner. We can avoid this by adding a non-fixed turnover constraint, fee, risk measure, or weight based tracking. For demonstration purposes we will use a turnover constraint with a maximum turnover of 2% per period for all assets from an equal weight starting point, we will provide the [`Turnover`](@ref) directly, which is non-fixed by default, meaning it will be updated every period.
=#
N = size(rd.X, 2)
tn = Turnover(; w = range(; start = 1 / N, stop = 1 / N, length = N), val = 0.02)

#=
We can generate the optimiser with the turnover constraint and then perform the walkforward cross validation again.
=#
mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, tn = tn))
idx_tn_walkforward_pred = cross_val_predict(mr, rd, idx_walk_forward)

#=
Now let's see the evolution of the weights across the different splits. We can see how the weights change at most 2% per period.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(idx_tn_walkforward_pred.res, :w)),
                            Symbol.(1:16))); formatters = [resfmt])

#=
#### 2.3.2 DateWalkForward

The [`DateWalkForward`](@ref) estimator is similar to the [`IndexWalkForward`](@ref) estimator, but it allows us to specify the training and test periods in terms of dates. This can be useful if we want to align our training and test sets with specific calendar periods, such as fiscal years or quarters.

The `Dates` module provides a large amount of functionality to manipulate dates, but we will keep it simple. For this we will define an adjuster function that takes a date range generates a new one made up only of the last day of the month.
=#

function ldm(x)
    val = lastdayofmonth.(x)
    while !isempty(val)
        if val[end] > x[end]
            val = val[1:(end - 1)]
        else
            break
        end
    end
    return val
end;

#=
This estimator can take a few options, the first argument can also be a date period or compound period, but if we leave it as an integer it will take on that many periods. The second argument is always an integer and is the value of that many periods. Combining both gives us a fully determined mixture of training and test set lengths as both can be set to an arbitrarily defined training and testing period.

To keep it simple, we will keep the period unit the same for both. We will again train for a year and test for a quarter, but the dates will now align with the end of calendar months. The date boundaries are determined by searching for the last date in the timestamps less than a value in the date boundary. If the date of the timestamp is not found in the date range, the `previous` flag is used to determine whether to take the last date found (previous = true), or the next available date is used (previous = false). This means that in order to guarantee alignment of the first date of each test set with the last day of the month we need to set `previous` to true.
=#

date_walk_forward = DateWalkForward(12, 3; period = Month(1), adjuster = ldm,
                                    previous = true)

#=
We can see what the splits look like.
=#

date_walk_forward_res = split(date_walk_forward, rd)
show(date_walk_forward_res.train_idx)
show(date_walk_forward_res.test_idx)

#=
We will once more use the turnover constraint, but with this new cross validation method.
=#

date_tn_walkforward_pred = cross_val_predict(mr, rd, date_walk_forward)

#=
We can see the evolution of the weights across the different splits. We can see how the weights change at most 2% per period.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(date_tn_walkforward_pred.res, :w)),
                            Symbol.(1:15))); formatters = [resfmt])

#=
The splits are different to the index walkforward method, so the weights are also different, but we can see there's not too much variation. That's because the training periods are roughly the same. However, the turnover constraint also helps in stabilising the weights.

There is another cross validation method called [`MultipleRandomised`]-(@ref) which uses a walk forward estimator, but also randomly samples the asset universe. Since it is more complex to analyse and understand, we will cover it in a future example.
=#
