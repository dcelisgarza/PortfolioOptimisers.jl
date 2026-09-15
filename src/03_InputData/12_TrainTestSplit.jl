"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve one side of a train/test split into a row count.

A size is either an `Integer` count of observations, or an `AbstractFloat` fraction of them in `(0, 1)`. Counts saturate at `N` (asking for more rows than exist takes all of them); the [`safe_index`](@ref) window guards then reject a split that leaves either side empty.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and the two mean different things: a count and a fraction.

 1. `s` is an `Integer`, so it is a count of rows: check that `s > 0`, and return `min(Int(s), N)`. A count larger than the data takes every row.
 2. `s` is an `AbstractFloat`, so it is a fraction of the rows: check that `0 < s < 1`, and return `clamp(floor(Int, s * N), 1, N)`. The fraction rounds **down** to whole rows, and the clamp keeps a small fraction of a short window from resolving to zero rows.

# Arguments

  - `s`: One side of the split, as a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`).
  - `N`: Number of observations available.
  - `name`: Symbolic name of the side, displayed in error messages.

# Validation

  - `s > 0` when `s` is an `Integer`.
  - `0 < s < 1` when `s` is an `AbstractFloat`.

# Returns

  - `n::Int`: The number of rows the side takes, in `1:N`.

# Related

  - [`safe_index`](@ref)
  - [`TrainTestSplit`](@ref)
"""
function split_count(s::Integer, N::Integer, name::Symbol)::Int
    @argcheck(s > zero(s),
              DomainError(s, "the $name of a train/test split must be > 0, got $s"))
    return min(Int(s), N)
end
function split_count(s::AbstractFloat, N::Integer, name::Symbol)::Int
    @argcheck(zero(s) < s < one(s),
              DomainError(s,
                          "the $name of a train/test split must lie in (0, 1) when given as a fraction, got $s"))
    return clamp(floor(Int, s * N), 1, N)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the `(train, test)` observation ranges of a holdout split over `N` time-ordered rows.

Training rows come from the head of the data and test rows from the tail, so the test window is always the most recent one. Each size is a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`), resolved by [`split_count`](@ref).

  - **Neither given**: the split falls at `D` (75 % train, 25 % test).
  - **One given**: the other side is its complement, so the two windows partition the data.
  - **Both given**: the head supplies `lo` training rows, the tail supplies `hi` test rows, and any rows between them are **embargoed** — they belong to neither window. This is how a gap between train and test is expressed. The gap is declared by the two sizes and nothing else; a rule that derives one from the label horizon belongs to the purged cross-validators ([`CombinatorialCrossValidation`](@ref)), not here.

# Algorithm

 1. Resolve the two window lengths `N_l` and `N_h`, through the branch that `lo` and `hi` select:

     1. Neither is given: take `n = clamp(floor(Int, D * N), 1, N)`, then `N_l = n` and `N_h = N - n`.
     2. Only `lo` is given: resolve it with [`split_count`](@ref), then `N_l = n` and `N_h = N - n`.
     3. Only `hi` is given: resolve it with [`split_count`](@ref), then `N_l = N - n` and `N_h = n`.
     4. Both are given: resolve each with [`split_count`](@ref) on its own. Neither is the complement of the other, so the rows between the two windows are embargoed.

 2. Check that both windows are non-empty, and that the two do not overlap.

 3. Return the two ranges `1:N_l` and `(N - N_h + 1):N`. The training window is the head of the data, and the test window is the tail, so the embargoed rows sit between them.

# Arguments

  - `lo`: Training rows, as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `hi`.
  - `hi`: Test rows, likewise; `nothing` takes the complement of `lo`.
  - `N`: Number of observations available.
  - `D = 0.75`: Training fraction taken when neither size is given.

# Validation

  - Both windows are non-empty. A split whose sizes saturate the data on one side (`train_size = N`) leaves nothing to test on and throws.
  - The windows do not overlap: `lo + hi <= N`.

# Returns

  - `(train, test)`: The training and test row ranges, as two `UnitRange{Int}`s.

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplit`](@ref)
  - [`split_count`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function safe_index(lo::Option{<:Number}, hi::Option{<:Number}, N::Integer, D = 0.75)
    N_l, N_h = if isnothing(lo) && isnothing(hi)
        n = clamp(floor(Int, D * N), 1, N)
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

Cut price- or returns-level data into a training window (the head) and a held-out test window (the tail).

The free-function form of [`TrainTestSplit`](@ref); the windows are [`port_opt_view`](@ref)s, so no data is copied. See [`safe_index`](@ref) for the sizing rules — complement when one side is given, embargo when both are.

# Algorithm

 1. Read the observation count `N` from the asset data: `size(rd.X, 1)` at returns level, and `size(TimeSeries.values(pr.X), 1)` at price level.
 2. Resolve the two row ranges with [`safe_index`](@ref).
 3. Return a [`port_opt_view`](@ref) of each range. The returns-level method passes a `Colon` asset index after the row range, because that arity indexes observations first and assets second.

# Arguments

  - `rd`/`pr`: The data to split.
  - `train_size`: Training rows as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `test_size`.
  - `test_size`: Test rows, likewise; `nothing` takes the complement of `train_size`. With neither given the split is 75/25.

# Returns

  - `(train, test)`: The two windows, of the same type as the input.

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
function train_test_split(rd::PricesResult; train_size::Option{<:Number} = nothing,
                          test_size::Option{<:Number} = nothing)
    N = size(TimeSeries.values(rd.X), 1)
    train, test = safe_index(train_size, test_size, N)
    return port_opt_view(rd, train), port_opt_view(rd, test)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator reserving the tail of the observations as a held-out test window.

The estimator form of [`train_test_split`](@ref), and the way the holdout protocol enters a [`Pipeline`](@ref): as the **first** step, it hands the training window to every step downstream and stashes the test window in its fitted [`TrainTestSplitResult`](@ref). `fit_predict(pipe, data)` then evaluates the fitted workflow on that held-out window in one line.

It is the one preprocessing estimator that is not pinned to a data level: it splits whichever level the pipeline input provides, price or returns, since a holdout is a statement about *rows*, not about columns or units.

Replaying a fitted split on an unseen window is a **pass-through** — the fitted rows are training-window state, and applying them to new data would be meaningless — so `predict(res, future_data)` keeps working on genuinely new observations.

!!! warning

    A pipeline containing a `TrainTestSplit` may not also be cross-validated: the split and the cross-validator are two evaluation protocols, and cross-validation already defines its own train/test windows. [`search_cross_validation`](@ref) rejects such a pipeline rather than silently shaving a second holdout off every fold.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrainTestSplit(;
        train_size::Option{<:Number} = nothing,
        test_size::Option{<:Number} = nothing,
    ) -> TrainTestSplit

Keywords correspond to the struct's fields. Sizes follow [`safe_index`](@ref): a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`); one side given makes the other its complement; both given embargoes the rows between them; neither given splits 75/25.

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
    Training observations as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `test_size`.
    """
    train_size
    """
    Test observations, likewise; `nothing` takes the complement of `train_size`.
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

Fitted result of a [`TrainTestSplit`](@ref), carrying both windows of the holdout.

The `test` window is the payoff: it is the data the fitted pipeline has never seen, and what `fit_predict(pipe, data)` predicts on. The `train` window is kept alongside it so the raw data the workflow was fitted on is retrievable from the result rather than having to be re-derived.

Both are [`port_opt_view`](@ref)s of the input at whichever level the split ran (price or returns).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`TrainTestSplit`](@ref)
  - [`PipelineResult`](@ref)
"""
@concrete struct TrainTestSplitResult <: AbstractResult
    """
    The training window: the head of the observations, and the data every downstream step is fitted on.
    """
    train
    """
    The held-out test window: the tail of the observations, which no fitted step has seen.
    """
    test
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a [`TrainTestSplit`](@ref) by cutting the data into its two windows.

Unlike the other preprocessing estimators, the fitted result is *not* replayed on unseen data: a holdout's rows are a fact about the fitting window alone, so [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) passes the window through unchanged.

# Algorithm

 1. [`fit_preprocessing`](@ref) calls [`train_test_split`](@ref) on the data, and returns the [`TrainTestSplitResult`](@ref) that holds both windows.
 2. [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) returns its data argument unchanged.
 3. [`apply_preprocessing`](@ref) on a [`TrainTestSplit`](@ref) returns its data argument unchanged, so an unfitted step is a pass-through as well.

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

Split `data` under a [`TrainTestSplit`](@ref), returning both windows as a [`TrainTestSplitResult`](@ref).

The estimator-form counterpart of the keyword form: `train_test_split(rd; test_size = 0.2)` hands back a bare `(train, test)` tuple, while this hands back the same fitted result a pipeline's split step produces, so a holdout configured once can be reused verbatim inside and outside a [`Pipeline`](@ref).

# Algorithm

 1. Call the keyword form of [`train_test_split`](@ref) with `tts.train_size` and `tts.test_size`, giving the two windows.
 2. Wrap the pair in a [`TrainTestSplitResult`](@ref), in the order `(train, test)`.

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
