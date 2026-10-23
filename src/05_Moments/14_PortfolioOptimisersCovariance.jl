"""
$(DocStringExtensions.TYPEDEF)

Runs a covariance estimator, then applies a matrix processing step to its result.

`ce` computes the raw matrix. `mp` repairs or filters it with the positive definite repair, the denoise step and the detone step. The library uses this composite as its default covariance estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PortfolioOptimisersCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> PortfolioOptimisersCovariance

Keywords correspond to the struct's fields.

## The incremental fit

The composite folds by composition. The state in `cache` selects one of two routes.

If `cache` holds `nothing`, [`partial_fit!`](@ref) passes the observation to `ce.ce` and keeps no state of its own. This route works because `mp` needs only a moment and a count. `pdm` and `dt` read only `sigma`, and `dn` reads the effective sample ratio `T / N`. The one-argument [`Statistics.cov`](@ref) reads the folded matrix of the inner estimator. Then it applies `mp` with the count from [`observation_count`](@ref), which is the same `size(X, 1)` that the matrix method reads. An `mp.alg` of the caller is the one step that needs the sample itself. The fold refuses it with an error that names it.

If `cache` holds a [`SampleBufferState`](@ref), which [`Online`](@ref) seeds, the composite takes the buffer route. Then `cov(ce)` with no data runs the batch verb over the observations that the buffer kept. An `mp.alg` needs this route.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ce`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisersCovariance()
PortfolioOptimisersCovariance
  ce ┼ Covariance
     │    me ┼ SimpleExpectedReturns
     │       │   w ┴ nothing
     │    ce ┼ GeneralCovariance
     │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │    w ┴ nothing
     │   alg ┼ FullMoment()
     │     w ┴ nothing
  mp ┼ MatrixProcessing
     │     pdm ┼ Posdef
     │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      dn ┼ nothing
     │      dt ┼ nothing
     │     alg ┼ nothing
     │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
  - [`observation_count`](@ref)
  - [`Online`](@ref)
"""
@propagatable @concrete struct PortfolioOptimisersCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function PortfolioOptimisersCovariance(ce::StatsBase.CovarianceEstimator,
                                           mp::AbstractMatrixProcessingEstimator,
                                           cache::Option{<:AbstractPartialFitState})
        return new{typeof(ce), typeof(mp), typeof(cache)}(ce, mp, cache)
    end
end
function PortfolioOptimisersCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                                       mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                       cache::Option{<:AbstractPartialFitState} = nothing)::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(ce, mp, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`PortfolioOptimisersCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:PortfolioOptimisersCovariance, true)` to render it.

# Arguments

  - `::PortfolioOptimisersCovariance`: The covariance estimator. The method reads only its type.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ce, :mp)`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::PortfolioOptimisersCovariance) = (:ce, :mp)
"""
    Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1,
                   active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)

Compute the covariance matrix with a [`PortfolioOptimisersCovariance`](@ref) estimator, and process the matrix.

The method computes the covariance matrix of `X` with the inner estimator `ce.ce`. Then it applies the matrix processing step `ce.mp`.

The composite does not change a gap. It passes `X` and `active_mask` to `ce.ce` without change, so its [`gap_fill_value`](@ref) is the [`gap_fill_value`](@ref) of `ce.ce`. A composite that wraps a gap-aware estimator does not refuse a sample that is not finite. It repairs the finite block and not the whole frame. The [`AssetPanel`](@ref) method of this file does the same.

# Algorithm

 1. Check `dims`, and orient `X` and `active_mask` to `observations × assets`. Transpose them when `dims == 2`.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref) when [`gap_fill_value`](@ref) on `ce` is finite, because then `ce.ce` is a plain estimator that cannot use a gapped sample.
 3. Compute `sigma` with `Statistics.cov(library_covariance_estimator(ce.ce), X; kwargs...)`. Pass `active_mask` too if it is given.
 4. When `sigma` is immutable, copy it into a `Matrix`, because step 5 writes in place.
 5. Apply [`matrix_processing!`](@ref) with `ce.mp` to `sigma` in place. If `ce.ce` is gap-aware, apply [`matrix_processing_block!`](@ref) instead, so that an asset outside the Coverage Universe keeps its `NaN` row and column.
 6. Return `sigma`.

`ce.ce` runs before `ce.mp`. `ce.mp.order` sets the order of the steps inside the processing. Step
1 orients `X` one time, so the estimator and the processing read the same orientation. Neither of
them takes its own `dims`.

# Arguments

  - `ce`: The composite covariance estimator with a processing step.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The optional boolean matrix with the same size as `X`. The method passes it to `ce.ce` if it is given.
  - `kwargs...`: More keyword arguments for the inner covariance estimator and for the matrix processing step.

# Validation

  - $(val_dict[:dims])

# Returns

  - `sigma::Matrix{<:Number}`: The processed covariance matrix.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    X, amsk = dims_oriented(dims, X, active_mask)
    plain = isfinite(gap_fill_value(ce))
    if plain
        assert_finite_sample(X)
    end
    sigma = if isnothing(amsk)
        Statistics.cov(library_covariance_estimator(ce.ce), X; kwargs...)
    else
        Statistics.cov(library_covariance_estimator(ce.ce), X; active_mask = amsk,
                       kwargs...)
    end
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    if plain
        matrix_processing!(ce.mp, sigma, X; kwargs...)
    else
        matrix_processing_block!(ce.mp, sigma, X; kwargs...)
    end
    return sigma
end
"""
    gap_fill_value(ce::PortfolioOptimisersCovariance) -> Number

Return the gap fill value of `ce.ce`, because the composite passes the sample and the mask to `ce.ce` without change.

The composite adds matrix processing to an inner estimator and reads no cell of the sample, so the inner estimator alone decides what a gap does to the result. A composite that wraps a gap-aware estimator is gap-aware. A composite that wraps a plain estimator is not.

# Arguments

  - `ce`: The composite covariance estimator with a processing step.

# Returns

  - `fv::Number`: [`gap_fill_value`](@ref) of `ce.ce`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`gap_fill_value`](@ref)
"""
function gap_fill_value(ce::PortfolioOptimisersCovariance)
    return gap_fill_value(ce.ce)
end
"""
    Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum,
                   pnl::Option{<:AssetPanel}; dims = 1, kwargs...) -> MatNum
    Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum,
                   pnl::Option{<:AssetPanel}; dims = 1, kwargs...) -> MatNum

Pass the Asset Panel to the inner estimator, then repair the finite block of the frame that it returns.

This method is the override of the composite for the reduce-and-expand root. The inner estimator does the reduction, because only the inner estimator knows if it is plain or mask-aware. This method does the repair. The repair runs on the finite block through [`matrix_processing_block!`](@ref), so an asset outside the Coverage Universe keeps its `NaN` row and column, and no entry outside the finite block changes.

# Algorithm

 1. Check `dims` and orient `X` to `observations × assets`.
 2. Compute the matrix with the inner estimator. Pass `pnl` as its third positional argument.
 3. When the matrix is immutable, copy it into a `Matrix`, because step 4 writes in place.
 4. Repair its finite block with [`matrix_processing_block!`](@ref), with `ce.mp` and the columns of `X` that the block names.
 5. Return the matrix.

# Arguments

  - `ce`: The composite covariance estimator with a processing step.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the inner estimator and for the matrix processing step.

# Validation

  - $(val_dict[:dims])

# Returns

  - `sigma::MatNum`: The processed covariance matrix or correlation matrix, on the full asset universe.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing_block!`](@ref)
  - [`coverage_reduction`](@ref)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims = 1, kwargs...)
    X = dims_oriented(dims, X)
    sigma = Statistics.cov(library_covariance_estimator(ce.ce), X, pnl; dims = 1, kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    matrix_processing_block!(ce.mp, sigma, X; kwargs...)
    return sigma
end
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims = 1, kwargs...)
    X = dims_oriented(dims, X)
    rho = Statistics.cor(library_covariance_estimator(ce.ce), X, pnl; dims = 1, kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    matrix_processing_block!(ce.mp, rho, X; kwargs...)
    return rho
end
"""
    Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1,
                   active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)

Compute the correlation matrix with a [`PortfolioOptimisersCovariance`](@ref) estimator, and process the matrix.

The method computes the correlation matrix of `X` with the inner estimator `ce.ce`. Then it applies the matrix processing step `ce.mp`.

# Algorithm

 1. Check `dims` and orient `X` to `observations × assets`. Transpose it when `dims == 2`.
 2. Compute `rho` with `Statistics.cor(library_covariance_estimator(ce.ce), X; kwargs...)`.
 3. When `rho` is immutable, copy it into a `Matrix`, because step 4 writes in place.
 4. Apply [`matrix_processing!`](@ref) with `ce.mp` to `rho`, in place.
 5. Return `rho`.

`ce.ce` runs before `ce.mp`. `ce.mp.order` sets the order of the steps inside the processing. Step
1 orients `X` one time, so the estimator and the processing read the same orientation. Neither of
them takes its own `dims`.

# Arguments

  - `ce`: The composite covariance estimator with a processing step.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The optional boolean matrix with the same size as `X`. The method passes it to `ce.ce` if it is given.
  - `kwargs...`: More keyword arguments for the inner covariance estimator and for the matrix processing step.

# Validation

  - $(val_dict[:dims])

# Returns

  - `rho::Matrix{<:Number}`: The processed correlation matrix.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cor)
"""
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    X, amsk = dims_oriented(dims, X, active_mask)
    plain = isfinite(gap_fill_value(ce))
    if plain
        assert_finite_sample(X)
    end
    rho = if isnothing(amsk)
        Statistics.cor(library_covariance_estimator(ce.ce), X; kwargs...)
    else
        Statistics.cor(library_covariance_estimator(ce.ce), X; active_mask = amsk,
                       kwargs...)
    end
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    if plain
        matrix_processing!(ce.mp, rho, X; kwargs...)
    else
        matrix_processing_block!(ce.mp, rho, X; kwargs...)
    end
    return rho
end
"""
    find_uncorrelated_indices(X::MatNum;
                              ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                              t::Number = 0.95, absolute::Bool = false,
                              measure::Num_VecToScaM = MeanValue(),
                              scores::Option{<:VecNum} = nothing)

Find the indices of a maximally uncorrelated subset of the assets of a data matrix.

The function finds a subset of the columns of `X` in which no pair of assets has a correlation of `t` or more. If `absolute` is `true`, it compares the absolute value of the correlation. When two assets are too correlated, the function removes the asset with the higher drop score. It returns the indices of the assets that remain.

By default, the drop score of an asset is the summary of its column of the correlation matrix, so the asset that is redundant with the largest part of the universe goes first. The `scores` keyword replaces that criterion, and a higher score removes an asset first. [`RedundancySelector`](@ref) uses `scores` to keep, from each correlated pair, the asset with the better score under a risk measure.

This function is internal. A caller uses [`RedundancySelector`](@ref) with a [`PairwiseCorrelation`](@ref) algorithm.

# Algorithm

 1. Compute the correlation matrix `rho` with `ce`, and take its absolute value when `absolute` is `true`.
 2. When `scores` is `nothing`, collapse each column of `rho` with `measure` into `summary_rho`, the default drop score. Otherwise take `summary_rho` from `scores`.
 3. Read the strict lower triangle of `rho` into `tril_idx`, so that each pair occurs one time.
 4. Keep the pairs whose correlation is at least `t`, and sort them from the most to the least correlated.
 5. Go through that list in order. For a pair whose two assets are both still present, remove the asset with the higher drop score. When the two scores are equal, remove both assets. This is the tie policy of the library: if the function cannot tell two assets apart, it keeps neither of them. Two identical columns leave no survivor.
 6. Return the indices that step 5 did not remove, in ascending order.

Step 5 skips a pair in which one asset or both assets are already removed. Thus the result depends
on the order that step 4 sets.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:ce])
  - `t`: The correlation threshold. Two assets with a correlation of `t` or more are too correlated.
  - `absolute`: If `true`, the function compares the absolute value of the correlation.
  - `measure`: The summary measure that the function applies to each column of the correlation matrix, for example the mean. The result is the default drop score. The function ignores `measure` if `scores` is given.
  - `scores`: The drop score of each asset. From a correlated pair, the function removes the asset with the higher score.

# Validation

  - If `scores` is not `nothing`, `length(scores) == size(X, 2)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `idx::Vector{Int}`: The indices of the assets that form a maximally uncorrelated subset.

# Related

  - [`RedundancySelector`](@ref)
  - [`PairwiseCorrelation`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`Num_VecToScaM`](@ref)
  - [`MeanValue`](@ref)
"""
function find_uncorrelated_indices(X::MatNum;
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   t::Number = 0.95, absolute::Bool = false,
                                   measure::Num_VecToScaM = MeanValue(),
                                   scores::Option{<:VecNum} = nothing)
    N = size(X, 2)
    rho = !absolute ? Statistics.cor(ce, X) : abs.(Statistics.cor(ce, X))
    if !isnothing(scores)
        @argcheck(length(scores) == N,
                  DimensionMismatch("find_uncorrelated_indices got $(length(scores)) scores for $N assets"))
    end
    summary_rho = if isnothing(scores)
        [vec_to_real_measure(measure, x) for x in eachcol(rho)]
    else
        scores
    end
    tril_idx = findall(LinearAlgebra.tril!(trues(size(rho)), -1))
    candidate_idx = findall(x -> x >= t, rho[tril_idx])
    candidate_idx = candidate_idx[sortperm(rho[tril_idx][candidate_idx]; rev = true)]
    to_remove = sizehint!(Set{Int}(), div(length(candidate_idx), 2))
    for idx in candidate_idx
        i, j = tril_idx[idx][1], tril_idx[idx][2]
        if i ∉ to_remove && j ∉ to_remove
            if summary_rho[i] > summary_rho[j]
                push!(to_remove, i)
            elseif summary_rho[i] < summary_rho[j]
                push!(to_remove, j)
            else
                push!(to_remove, i)
                push!(to_remove, j)
            end
        end
    end
    return setdiff(1:N, to_remove)
end

"""
    partial_fit!(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}, X::MatNum;
                 dims::Int = 1, kwargs...)
    partial_fit!(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}, x::VecNum;
                 kwargs...)

Folds observations into a [`PortfolioOptimisersCovariance`](@ref) through `ce.ce`.

The composite is exact by composition and keeps no state of its own. `ce.ce` folds the raw matrix. `ce.mp` is a step of `cov(ce)` with no data, and it runs on the matrix that it gets. The composite has no accumulator, and its `cache` stays `nothing` on this route. The `nothing` in `cache` selects this route. A composite with a [`SampleBufferState`](@ref) got that state from [`Online`](@ref), and it takes the buffer fold.

The step costs what the step of the inner estimator costs, which is `O(N²)`. The composite adds no cost.

`ce.mp` needs only a moment and a count, with one exception. An `mp.alg` of the caller gets the whole sample. [`assert_shape_only_matrix_processing`](@ref) refuses it at the fold. The refusal cannot wait until `cov(ce)` with no data, because the rows are gone at that time.

# Algorithm

 1. Refuse an `mp` that does not run from the shape of the sample, with [`assert_shape_only_matrix_processing`](@ref).
 2. Rebuild `ce` with the estimator that [`partial_fit!`](@ref) returns in `ce.ce`, through [`rebuild_estimator`](@ref).

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - `x`: One observation, with one entry for each asset.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for `ce.ce`.

# Validation

  - `ce.mp` runs from the shape of the sample, as [`assert_shape_only_matrix_processing`](@ref) states. An `ArgumentError` is thrown otherwise.

# Returns

  - `ce`: The composite. Its `ce.ce` is the estimator that holds the state after the last observation.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_shape_only_matrix_processing`](@ref)
  - [`Statistics.cov`](@ref)
  - [`Online`](@ref)
"""
function partial_fit!(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}, X::MatNum;
                      dims::Int = 1, kwargs...)
    assert_shape_only_matrix_processing(ce.mp)
    inner = partial_fit!(ce.ce, X; dims = dims, kwargs...)
    return inner === ce.ce ? ce : rebuild_estimator(ce, (; ce = inner))
end
function partial_fit!(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}, x::VecNum;
                      kwargs...)
    assert_shape_only_matrix_processing(ce.mp)
    inner = partial_fit!(ce.ce, x; kwargs...)
    return inner === ce.ce ? ce : rebuild_estimator(ce, (; ce = inner))
end
"""
    Statistics.cov(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}; kwargs...)
    Statistics.cor(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}; kwargs...)

Reads the covariance or the correlation that a folded [`PortfolioOptimisersCovariance`](@ref) holds.

This method makes the estimate from the fold by composition. It is `cov(ce)` or `cor(ce)` with no data. `ce.ce` returns its own folded matrix. Then the method applies `ce.mp` to that matrix from the shape of the sample. [`observation_count`](@ref) reads the number of folded observations from the inner state. That number is the same `size(X, 1)` that the matrix method of [`matrix_processing!`](@ref) reads, with the `NaN` rows, so the substitution is an identity and not an approximation. This method returns the same result as a batch fit over the same observations.

If the matrix of the inner estimator is immutable, the method copies it, because the processing writes in place. The matrix methods of this file do the same.

# Algorithm

 1. Read the folded matrix of the inner estimator with the one-argument `Statistics.cov` or `Statistics.cor`.
 2. Copy it into a `Matrix` when it is immutable.
 3. Apply `ce.mp` in place through the shape method of [`matrix_processing_block!`](@ref), with `T` from [`observation_count`](@ref) and `N` from the matrix. The method uses the block form, because a fold over a changing universe returns `NaN` for an asset outside the Coverage Universe. The [`AssetPanel`](@ref) methods of this file do the same. A complete matrix costs nothing more.
 4. Return the matrix.

# Arguments

  - $(arg_dict[:ce])
  - `kwargs...`: More keyword arguments for `ce.ce`.

# Validation

  - `ce.ce` holds a partial-fit state. An `ArgumentError` is thrown otherwise.
  - `ce.mp` runs from the shape of the sample, as [`assert_shape_only_matrix_processing`](@ref) states. An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: The processed covariance matrix, or the processed correlation matrix `rho`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`observation_count`](@ref)
  - [`matrix_processing!`](@ref)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}; kwargs...)
    sigma = Statistics.cov(ce.ce; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    matrix_processing_block!(ce.mp, sigma, observation_count(ce.ce), size(sigma, 1);
                             kwargs...)
    return sigma
end
function Statistics.cor(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}; kwargs...)
    rho = Statistics.cor(ce.ce; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    matrix_processing_block!(ce.mp, rho, observation_count(ce.ce), size(rho, 1); kwargs...)
    return rho
end
"""
    Statistics.cov(ce::PortfolioOptimisersCovariance{<:Any, <:Any, <:SampleBufferState})
    Statistics.cor(ce::PortfolioOptimisersCovariance{<:Any, <:Any, <:SampleBufferState})

Reads the covariance or the correlation of a buffered [`PortfolioOptimisersCovariance`](@ref) from a new fit over its buffer.

This method makes the estimate on the buffer route. [`Online`](@ref) seeds this route, and an `mp.alg` of the caller needs it. The method runs the batch verb over the observations that the buffer kept, so it returns the same result as a batch fit over those rows, with `mp.alg`.

# Arguments

  - $(arg_dict[:ce])

# Validation

  - `ce` holds a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: The processed covariance matrix, or the processed correlation matrix `rho`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`Online`](@ref)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance{<:Any, <:Any,
                                                          <:SampleBufferState})
    return Statistics.cov(ce, partial_fit_cache(ce))
end
function Statistics.cor(ce::PortfolioOptimisersCovariance{<:Any, <:Any,
                                                          <:SampleBufferState})
    return Statistics.cor(ce, partial_fit_cache(ce))
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Answers whether [`partial_fit!`](@ref) folds a [`PortfolioOptimisersCovariance`](@ref).

The composite folds by composition, so it folds exactly when the inner estimator folds and `mp` runs from the shape of the sample, as [`supports_shape_processing`](@ref) states. A [`MatrixProcessing`](@ref) passes when it has no `alg` that reads the sample. A matrix processing estimator of the caller never passes. A composite that a wrapper gave a buffer folds in all cases. This is the default buffer route of [`supports_partial_fit`](@ref), and it is the only route that lets an `mp.alg` run online.

# Arguments

  - $(arg_dict[:ce])

# Returns

  - `folds::Bool`: `true` when [`partial_fit!`](@ref) folds this composite.

# Related

  - [`supports_partial_fit`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`assert_shape_only_matrix_processing`](@ref)
"""
function supports_partial_fit(ce::PortfolioOptimisersCovariance)
    return isa(ce.cache, SampleBufferState) ||
           (supports_shape_processing(ce.mp) && supports_partial_fit(ce.ce))
end

export PortfolioOptimisersCovariance
