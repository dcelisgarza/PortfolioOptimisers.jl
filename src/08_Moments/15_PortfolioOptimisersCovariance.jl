"""
$(DocStringExtensions.TYPEDEF)

Runs any covariance estimator, then applies a matrix post-processing step to its result.

`ce` computes the raw matrix and `mp` repairs or filters it — positive-definite repair, denoising, and detoning — so the composite is the estimator the rest of the library takes as its default.

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

The composite folds by **composition**, and the state it carries decides which of the two routes it takes.

With `cache` holding `nothing`, [`partial_fit!`](@ref) forwards the observation to `ce.ce` and keeps nothing of its own, because `mp` reads no observation that a moment and a count cannot stand in for: `pdm` and `dt` read `sigma` alone, and `dn` reads the effective sample ratio `T / N`. The one-argument [`Statistics.cov`](@ref) then reads the inner estimator's folded matrix and applies `mp` from [`observation_count`](@ref), which is exactly the `size(X, 1)` the matrix arm would have read. An `mp.alg` of a caller's own is the one step with no shape substitute, and it is refused by name at the fold.

With `cache` holding a [`SampleBufferState`](@ref) — which [`Online`](@ref) seeds — the composite takes the buffering route instead, and its read-out is the batch verb over the observations the buffer kept. That is the route an `mp.alg` needs.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

## Observation view parameters

When [`obs_weights_view`](@ref) is called on this type:

  - `cache`: Dropped via [`obs_weights_view`](@ref), because no slice of a state exists on the observation axis.

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

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:PortfolioOptimisersCovariance, true)` to render it. ADR 0105 records the decision.

# Arguments

  - `::PortfolioOptimisersCovariance`: Covariance estimator, read for its type alone.

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

Compute the covariance matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`.

The composite is transparent to a gap: it forwards `X` and `active_mask` to `ce.ce` untouched, so [`gap_fill_value`](@ref) on it is [`gap_fill_value`](@ref) on `ce.ce`. A composite that wraps a gap-aware estimator therefore takes no finiteness refusal of its own, and repairs the finite block rather than the whole frame, which is what the [`AssetPanel`](@ref) method beside it does.

# Algorithm

 1. Check `dims`, and orient `X` and `active_mask` to `observations × assets`, transposing them when `dims == 2`.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref) when [`gap_fill_value`](@ref) on `ce` is finite, because `ce.ce` is then a plain estimator that has no answer for one.
 3. Compute `sigma` with `Statistics.cov(ce.ce, X; kwargs...)`, adding `active_mask` when one is given.
 4. When `sigma` is immutable, copy it into a `Matrix`, because step 5 writes in place.
 5. Apply [`matrix_processing!`](@ref) with `ce.mp` to `sigma`, in place, or [`matrix_processing_block!`](@ref) when `ce.ce` is gap-aware, so an asset outside the Coverage Universe keeps its `NaN` row and column.
 6. Return `sigma`.

`ce.ce` runs before `ce.mp`, and `ce.mp.order` fixes the order of the steps inside the
post-processing. Step 1 orients `X` once, so the estimator and the post-processing both read the
same orientation and neither takes a `dims` of its own.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`, forwarded to `ce.ce` when it is given.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

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
        Statistics.cov(ce.ce, X; kwargs...)
    else
        Statistics.cov(ce.ce, X; active_mask = amsk, kwargs...)
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

Answer what `ce.ce` answers, because the composite forwards the sample and the mask untouched.

The composite adds matrix processing to an inner estimator and reads no cell of the sample itself, so a gap is the inner estimator's to keep or to lose. A composite wrapping a gap-aware estimator is therefore gap-aware, and one wrapping a plain estimator is not.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.

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

Forward the Asset Panel to the estimator that this composite wraps, then repair the finite block of the frame it gets back.

This is the composite's override of the reduce-and-expand root. The inner estimator owns the reduction, because it alone knows whether it is plain or mask-aware, and this method owns the repair. The repair runs on the finite block through [`matrix_processing_block!`](@ref), so an asset outside the Coverage Universe keeps its `NaN` row and column and nothing else is touched.

# Algorithm

 1. Check `dims` and orient `X` to `observations × assets`.
 2. Compute the matrix with the inner estimator, forwarding `pnl` as its third positional argument.
 3. When the matrix is immutable, copy it into a `Matrix`, because step 4 writes in place.
 4. Repair its finite block with [`matrix_processing_block!`](@ref), under `ce.mp` and the columns of `X` the block names.
 5. Return the matrix.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator and to the matrix processing step.

# Validation

  - $(val_dict[:dims])

# Returns

  - `sigma::MatNum`: The processed covariance matrix, or correlation matrix, on the full asset universe.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing_block!`](@ref)
  - [`coverage_reduction`](@ref)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims = 1, kwargs...)
    X = dims_oriented(dims, X)
    sigma = Statistics.cov(ce.ce, X, pnl; dims = 1, kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    matrix_processing_block!(ce.mp, sigma, X; kwargs...)
    return sigma
end
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims = 1, kwargs...)
    X = dims_oriented(dims, X)
    rho = Statistics.cor(ce.ce, X, pnl; dims = 1, kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    matrix_processing_block!(ce.mp, rho, X; kwargs...)
    return rho
end
"""
    Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1,
                   active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)

Compute the correlation matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the correlation matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`.

# Algorithm

 1. Check `dims` and orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Compute `rho` with `Statistics.cor(ce.ce, X; kwargs...)`.
 3. When `rho` is immutable, copy it into a `Matrix`, because step 4 writes in place.
 4. Apply [`matrix_processing!`](@ref) with `ce.mp` to `rho`, in place.
 5. Return `rho`.

`ce.ce` runs before `ce.mp`, and `ce.mp.order` fixes the order of the steps inside the
post-processing. Step 1 orients `X` once, so the estimator and the post-processing both read the
same orientation and neither takes a `dims` of its own.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`, forwarded to `ce.ce` when it is given.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

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
        Statistics.cor(ce.ce, X; kwargs...)
    else
        Statistics.cor(ce.ce, X; active_mask = amsk, kwargs...)
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

Find indices of a maximally uncorrelated subset of assets from a data matrix.

This function identifies a subset of asset columns in `X` such that no two assets in the subset have a pairwise (absolute) correlation exceeding the threshold `t`. When two assets are too correlated, the one with the higher *drop score* is removed. The function returns the indices of the remaining uncorrelated assets.

By default the drop score is each asset's summary correlation to every other asset, so the asset that is redundant with the *most* of the universe goes first. Supplying `scores` replaces that criterion — higher means "drop me" — which is how [`RedundancySelector`](@ref) makes the survivor of each correlated pair the better-scoring asset under a risk measure.

Internal machinery — the caller-facing form is [`RedundancySelector`](@ref) with a [`PairwiseCorrelation`](@ref) algorithm.

# Algorithm

 1. Compute the correlation matrix `rho` with `ce`, and take its absolute value when `absolute` is `true`.
 2. When `scores` is `nothing`, collapse each column of `rho` with `measure` into `summary_rho`, the default drop score. Otherwise take `summary_rho` from `scores`.
 3. Read the strict lower triangle of `rho` into `tril_idx`, giving each pair once.
 4. Keep the pairs whose correlation is at least `t`, and sort them from the most to the least correlated.
 5. Walk that list. For a pair whose two assets are both still present, remove the one with the higher drop score. **When the two scores are equal, remove both** — the library's "if we cannot tell them apart, trust neither" tie policy, which is why two identical columns leave no survivor.
 6. Return the indices that step 5 did not remove, in ascending order.

Step 5 skips a pair whose assets are already removed, so the result depends on the order step 4
fixes.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:ce])
  - `t`: Correlation threshold above which two assets are considered too correlated.
  - `absolute`: If `true`, the absolute value of the correlation is used for comparison.
  - `measure`: Summary measure applied to each column of the correlation matrix (e.g., mean) to produce the default drop score. Ignored when `scores` is given.
  - `scores`: Per-asset drop scores; the asset with the *higher* score is removed from a correlated pair.

# Validation

  - If `scores` is not `nothing`, `length(scores) == size(X, 2)`, else a `DimensionMismatch` is raised.

# Returns

  - `idx::Vector{Int}`: Indices of assets that form a maximally uncorrelated subset.

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

Folds observations into a [`PortfolioOptimisersCovariance`](@ref) by forwarding them to `ce.ce`.

The composite is **exact by composition** and keeps no state of its own: `ce.ce` folds the raw matrix, and `ce.mp` is a read-out step that runs on whatever matrix it is given. The composite therefore owns no accumulator, and its `cache` stays `nothing` on this route — which is what selects the route, because a composite carrying a [`SampleBufferState`](@ref) is one [`Online`](@ref) seeded and takes the buffering fold instead.

The step is `O(N²)`, the inner estimator's own, plus nothing.

`ce.mp` reads no observation that a moment and a count cannot stand in for, with one exception: an `mp.alg` of a caller's own is handed the whole sample, and [`assert_shape_only_matrix_processing`](@ref) refuses it here, at the fold, rather than at the read-out where the rows would already be gone.

# Algorithm

 1. Refuse an `mp` carrying a sample-reading `alg`.
 2. Rebind `ce.ce` to the estimator [`partial_fit!`](@ref) gives, with `Accessors.@reset`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - `x`: One observation, whose entries are the assets.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to `ce.ce`.

# Validation

  - `ce.mp.alg` is `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `ce`: The composite, with `ce.ce` rebound to the estimator carrying the state after the last observation.

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
    return Accessors.@reset ce.ce = partial_fit!(ce.ce, X; dims = dims, kwargs...)
end
function partial_fit!(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}, x::VecNum;
                      kwargs...)
    assert_shape_only_matrix_processing(ce.mp)
    return Accessors.@reset ce.ce = partial_fit!(ce.ce, x; kwargs...)
end
"""
    Statistics.cov(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}; kwargs...)
    Statistics.cor(ce::PortfolioOptimisersCovariance{<:Any, <:Any, Nothing}; kwargs...)

Reads the covariance, or the correlation, a folded [`PortfolioOptimisersCovariance`](@ref) has accumulated.

The read-out of the composition fold. `ce.ce` answers its own folded matrix, and `ce.mp` is applied to it from the **shape** of the sample: [`observation_count`](@ref) reads the number of observations folded off the inner state, and it is exactly the `size(X, 1)` the matrix arm of [`matrix_processing!`](@ref) would have read, `NaN` rows included. The substitution is therefore an identity, not an approximation, and this method answers what a batch fit over the same observations answers.

The matrix the inner estimator returns is copied when it is immutable, because the processing writes in place, exactly as the matrix methods beside this one do.

# Algorithm

 1. Read the inner estimator's folded matrix with the one-argument `Statistics.cov` or `Statistics.cor`.
 2. Copy it into a `Matrix` when it is immutable.
 3. Apply `ce.mp` in place through the shape arm of [`matrix_processing_block!`](@ref), with `T` from [`observation_count`](@ref) and `N` from the matrix. It is the block arm rather than the plain one because a fold over a changing universe answers `NaN` for an asset outside the Coverage Universe, exactly as the [`AssetPanel`](@ref) methods beside it do; a complete matrix costs nothing extra.
 4. Return the matrix.

# Arguments

  - $(arg_dict[:ce])
  - `kwargs...`: Additional keyword arguments, forwarded to `ce.ce`.

# Validation

  - `ce.ce` carries a partial-fit state. An `ArgumentError` is thrown otherwise.
  - `ce.mp.alg` is `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: The processed covariance matrix, or `rho`, the processed correlation matrix.

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

Reads the covariance, or the correlation, of a buffered [`PortfolioOptimisersCovariance`](@ref) by refitting over its buffer.

The read-out of the buffering route, which [`Online`](@ref) seeds and which an `mp.alg` of a caller's own needs. It is the batch verb over the observations the buffer kept, so it answers exactly what a batch fit over those rows answers, `mp.alg` included.

# Arguments

  - $(arg_dict[:ce])

# Validation

  - `ce` carries a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: The processed covariance matrix, or `rho`, the processed correlation matrix.

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

[`PortfolioOptimisersCovariance`](@ref) method of [`supports_partial_fit`](@ref).

The composite folds by composition, so it folds exactly when both of its halves allow it: the inner estimator must fold, and `mp` must carry no sample-reading `alg`. A composite a wrapper has given a buffer folds whatever those two say, which is the default and is what makes an `mp.alg` reachable online at all.

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
           (isnothing(ce.mp.alg) && supports_partial_fit(ce.ce))
end

export PortfolioOptimisersCovariance
