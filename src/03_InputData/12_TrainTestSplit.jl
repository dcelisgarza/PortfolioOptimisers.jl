"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve one side of a train/test split into a row count.

A size is an `Integer` count of observations, or a fraction of the observations given as any other `Real` in `(0, 1)`. A count larger than `N` takes all `N` rows. The window checks of [`safe_index`](@ref) then reject a split that leaves either side empty.

# Algorithm

The method that Julia selects is the first step, because a count and a fraction mean different things.

 1. `s` is an `Integer`, so it is a count of rows. Check that `s > 0`, and return `min(Int(s), N)`.
 2. `s` is any other `Real`, so it is a fraction of the rows. Check that `0 < s < 1`.
 3. Round `s * N` to the nearest integer, giving `n`.
 4. If `n / N`, computed in the type of `s`, exceeds `s`, subtract one from `n`. Now `n` is the largest count whose share of the rows does not exceed `s`, which is `floor(s * N)` in exact arithmetic. A direct `floor(Int, s * N)` loses a row when the product lands below a whole number, and `0.29 * 100` is `28.999999999999996`.
 5. Return `clamp(n, 1, N)`. The clamp stops a small fraction of a short window from resolving to zero rows.

# Arguments

  - `s`: One side of the split, as a row count (`Integer`) or a fraction of the observations (any other `Real` in `(0, 1)`).
  - `N`: The number of observations.
  - `name`: The name of the side, which the error messages display.

# Validation

  - `s > 0` when `s` is an `Integer`. Otherwise the method throws a `DomainError`.
  - `0 < s < 1` when `s` is any other `Real`. Otherwise the method throws a `DomainError`.

# Returns

  - `n::Int`: The number of rows that the side takes, in `1:N`.

# Related

  - [`safe_index`](@ref)
  - [`TrainTestSplit`](@ref)
"""
function split_count(s::Integer, N::Integer, name::Symbol)::Int
    @argcheck(s > zero(s),
              DomainError(s, "the $name of a train/test split must be > 0, got $s"))
    return min(Int(s), N)
end
function split_count(s::Real, N::Integer, name::Symbol)::Int
    @argcheck(zero(s) < s < one(s),
              DomainError(s,
                          "the $name of a train/test split must lie in (0, 1) when given as a fraction, got $s"))
    n = round(Int, s * N)
    if n * one(s) / N > s
        n -= 1
    end
    return clamp(n, 1, N)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the `(train, test)` observation ranges of a holdout split over `N` time-ordered rows.

The training window is the first rows of the data and the test window is the last rows, so the test window is always the most recent one. Each size is a row count or a fraction of the observations, and [`split_count`](@ref) resolves it to a count.

  - If neither size is given, the training window takes the fraction `D` of the rows, 75 % by default, and the test window takes the rest.
  - If one size is given, the other window takes the remaining rows. The two windows then partition the data.
  - If both sizes are given, the training window takes the first `lo` rows and the test window takes the last `hi` rows. The rows between the two windows belong to neither window. This gap is an embargo, and only the two sizes set it. The cross-validators [`KFold`](@ref) and [`CombinatorialCrossValidation`](@ref) take a gap around each test fold as `purged_size` and `embargo_size`.

# Algorithm

 1. Resolve the two window lengths `N_l` and `N_h`, through the branch that `lo` and `hi` select.

     1. Neither is given. Resolve `D` with [`split_count`](@ref), giving `n`. Then `N_l = n` and `N_h = N - n`.
     2. Only `lo` is given. Resolve it with [`split_count`](@ref), giving `n`. Then `N_l = n` and `N_h = N - n`.
     3. Only `hi` is given. Resolve it with [`split_count`](@ref), giving `n`. Then `N_l = N - n` and `N_h = n`.
     4. Both are given. Resolve each with [`split_count`](@ref) on its own. Neither length is the complement of the other, so the rows between the two windows belong to neither window.

 2. Check that both windows are non-empty, and that `N_l + N_h <= N`.

 3. Return the two ranges `1:N_l` and `(N - N_h + 1):N`. The embargoed rows lie between them.

# Arguments

  - `lo`: The training rows, as a count (`Integer`) or a fraction (any other `Real` in `(0, 1)`). `nothing` takes the complement of `hi`.
  - `hi`: The test rows, as a count or a fraction. `nothing` takes the complement of `lo`.
  - `N`: The number of observations.
  - `D = 0.75`: The training fraction when neither size is given.

# Validation

  - Each size passes the checks of [`split_count`](@ref), which throws a `DomainError` otherwise.
  - Both windows are non-empty. Otherwise the function throws an `ArgumentError`. A size that takes every row, such as `train_size = N`, leaves no row for the other window.
  - `N_l + N_h <= N`, so that the windows do not overlap. Otherwise the function throws an `ArgumentError`.

# Returns

  - `(train, test)`: The training and test row ranges, as two `UnitRange{Int}`s.

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplit`](@ref)
  - [`split_count`](@ref)
  - [`KFold`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function safe_index(lo::Option{<:Number}, hi::Option{<:Number}, N::Integer, D = 0.75)
    N_l, N_h = if isnothing(lo) && isnothing(hi)
        n = split_count(D, N, :train_size)
        n, N - n
    elseif isnothing(hi)
        n = split_count(lo, N, :train_size)
        n, N - n
    elseif isnothing(lo)
        n = split_count(hi, N, :test_size)
        N - n, n
    else
        split_count(lo, N, :train_size), split_count(hi, N, :test_size)
    end
    @argcheck(N_l > 0 && N_h > 0,
              ArgumentError("a train/test split of $N observations must leave both windows non-empty, got $N_l training and $N_h test observations"))
    @argcheck(N_l + N_h <= N,
              ArgumentError("the training and test windows of a train/test split must not overlap, but $N_l training and $N_h test observations exceed the $N available; rows between the two windows are embargoed, so their sizes may sum to less than $N but never to more"))
    return 1:N_l, (N - N_h + 1):N
end
"""
    train_test_split(rd::ReturnsResult; train_size, test_size) -> (train, test)
    train_test_split(pr::PricesResult; train_size, test_size) -> (train, test)

Cut price or returns data into a training window of the first rows and a test window of the last rows.

This is the function form of [`TrainTestSplit`](@ref). [`port_opt_view`](@ref) makes each window. At the returns level each window holds views of the input arrays, so the split copies no data. At the price level `port_opt_view` indexes a `TimeArray` by date, and that copies the rows. [`safe_index`](@ref) states the sizing rules.

# Algorithm

 1. Read the observation count `N`, as `size(rd.X, 1)` at the returns level and as `size(TimeSeries.values(pr.X), 1)` at the price level.
 2. Resolve the two row ranges `train` and `test` with [`safe_index`](@ref).
 3. Return a [`port_opt_view`](@ref) of each range. The returns-level method gives `:` as the asset index after the row range, because the three-argument form of `port_opt_view` takes the observations first and the assets second.

# Arguments

  - `rd`, `pr`: The data to split.
  - `train_size`: The training rows, as a count (`Integer`) or a fraction (any other `Real` in `(0, 1)`). `nothing` takes the complement of `test_size`.
  - `test_size`: The test rows, as a count or a fraction. `nothing` takes the complement of `train_size`. If neither size is given, the split is 75/25.

# Validation

  - The two sizes pass the checks of [`safe_index`](@ref) for `N` rows.

# Returns

  - `(train, test)`: The two windows. Each is a `ReturnsResult` for a `ReturnsResult` input, and a `PricesResult` for a `PricesResult` input.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\"], X = reshape(collect(0.1:0.1:1.0), 10, 1));

julia> train, test = train_test_split(rd; test_size = 0.2);

julia> size(train.X, 1), size(test.X, 1)
(8, 2)
```

# Related

  - [`TrainTestSplit`](@ref)
  - [`safe_index`](@ref)
"""
function train_test_split(rd::ReturnsResult; train_size::Option{<:Number} = nothing,
                          test_size::Option{<:Number} = nothing)
    N = size(rd.X, 1)
    train, test = safe_index(train_size, test_size, N)
    return port_opt_view(rd, train, :), port_opt_view(rd, test, :)
end
function train_test_split(pr::PricesResult; train_size::Option{<:Number} = nothing,
                          test_size::Option{<:Number} = nothing)
    N = size(TimeSeries.values(pr.X), 1)
    train, test = safe_index(train_size, test_size, N)
    return port_opt_view(pr, train), port_opt_view(pr, test)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator that holds out the last observations as a test window.

It is the estimator form of [`train_test_split`](@ref), and it adds a holdout to a [`Pipeline`](@ref). It must be the first step of the pipeline. It passes the training window to every later step, and it keeps the test window in its fitted [`TrainTestSplitResult`](@ref). `fit_predict(pipe, data)` then predicts on the test window, which no fitted step has seen.

It is the one preprocessing estimator that takes both data levels. It splits the price or returns data that the pipeline input holds, because a holdout selects rows and reads no column.

A fitted split passes new data through unchanged. Its rows index the fitting data alone, so `predict(res, future_data)` predicts on every new observation.

!!! warning

    A pipeline that holds a `TrainTestSplit` cannot also be cross-validated. Cross-validation makes its own train and test windows, so a split inside each fold removes a second test window from every training window. [`cross_val_predict`](@ref) and [`search_cross_validation`](@ref) throw an `ArgumentError` for such a pipeline.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrainTestSplit(;
        train_size::Option{<:Number} = nothing,
        test_size::Option{<:Number} = nothing,
    ) -> TrainTestSplit

Keywords correspond to the struct's fields. A size is a row count (`Integer`) or a fraction of the observations (any other `Real` in `(0, 1)`). One size makes the other window its complement, two sizes embargo the rows between the windows, and no size gives a 75/25 split. The constructor does not check the sizes. [`safe_index`](@ref) checks them when the split runs on data.

# Examples

```jldoctest
julia> pipe = Pipeline(;
                       steps = (TrainTestSplit(; test_size = 0.2), PricesToReturns(),
                                EmpiricalPrior(), EqualWeighted()));

julia> pipe.names
("split", "returns", "prior", "opt")
```

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplitResult`](@ref)
  - [`Pipeline`](@ref)
"""
@concrete struct TrainTestSplit <: AbstractPreprocessingEstimator
    """
    The training observations, as a count (`Integer`) or a fraction (any other `Real` in `(0, 1)`). `nothing` takes the complement of `test_size`.
    """
    train_size
    """
    The test observations, as a count or a fraction. `nothing` takes the complement of `train_size`.
    """
    test_size
    function TrainTestSplit(train_size::Option{<:Number}, test_size::Option{<:Number})
        return new{typeof(train_size), typeof(test_size)}(train_size, test_size)
    end
end
function TrainTestSplit(; train_size::Option{<:Number} = nothing,
                        test_size::Option{<:Number} = nothing)::TrainTestSplit
    return TrainTestSplit(train_size, test_size)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted holdout split that holds the training window and the test window.

It is the result of a [`TrainTestSplit`](@ref). The `test` window is the data that the fitted pipeline has not seen, and `fit_predict(pipe, data)` predicts on it. The result also keeps the `train` window, so a caller can read the training data from the result without a second split.

Both windows are [`port_opt_view`](@ref)s of the input, at the data level of the input.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`TrainTestSplit`](@ref)
  - [`PipelineResult`](@ref)
"""
@concrete struct TrainTestSplitResult <: AbstractResult
    """
    The training window. It is the first rows of the observations, and every later step fits on it.
    """
    train
    """
    The test window. It is the last rows of the observations, and no fitted step has seen it.
    """
    test
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a [`TrainTestSplit`](@ref) by cutting the data into its two windows.

The fitted result does not replay on new data, unlike the fitted results of the other preprocessing estimators. The rows of a holdout belong to the fitting data alone, so [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) returns its data unchanged. `apply_preprocessing` on an unfitted `TrainTestSplit` also returns its data unchanged.

# Algorithm

 1. Call [`train_test_split`](@ref) with `tts` and `data`, which returns the [`TrainTestSplitResult`](@ref) of both windows.

# Validation

  - The sizes of `tts` pass the checks of [`safe_index`](@ref) for the rows of `data`.

# Related

  - [`TrainTestSplit`](@ref)
  - [`train_test_split`](@ref)
"""
function fit_preprocessing(tts::TrainTestSplit, data::Prices_RR)::TrainTestSplitResult
    return train_test_split(tts, data)
end
function apply_preprocessing(::TrainTestSplitResult, data::Prices_RR)
    return data
end
function apply_preprocessing(::TrainTestSplit, data::Prices_RR)
    return data
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Split `data` under a [`TrainTestSplit`](@ref), and return both windows as a [`TrainTestSplitResult`](@ref).

This is the estimator form of the keyword method. `train_test_split(rd; test_size = 0.2)` returns a `(train, test)` tuple, and this method returns the result that the split step of a [`Pipeline`](@ref) makes. So one `TrainTestSplit` gives the same windows inside and outside a pipeline.

# Algorithm

 1. Call the keyword method of [`train_test_split`](@ref) with `tts.train_size` and `tts.test_size`, giving `train` and `test`.
 2. Return `TrainTestSplitResult(train, test)`.

# Validation

  - The sizes of `tts` pass the checks of [`safe_index`](@ref) for the rows of `data`.

# Related

  - [`TrainTestSplit`](@ref)
  - [`TrainTestSplitResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
function train_test_split(tts::TrainTestSplit, data::Prices_RR)::TrainTestSplitResult
    train, test = train_test_split(data; train_size = tts.train_size,
                                   test_size = tts.test_size)
    return TrainTestSplitResult(train, test)
end
export train_test_split, TrainTestSplit, TrainTestSplitResult
