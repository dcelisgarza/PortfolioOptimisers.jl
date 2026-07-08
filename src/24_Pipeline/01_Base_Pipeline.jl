"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all pipeline estimator types in `PortfolioOptimisers.jl`.

A pipeline reifies an end-to-end workflow — price preprocessing, prices-to-returns conversion, returns preprocessing, prior estimation, phylogeny, uncertainty sets, constraint generation, and optimisation — as an ordered list of steps executed left-to-right over a [`PipelineContext`](@ref). Pipelines widen the cross-validation and hyperparameter-tuning boundary to the entire workflow, data preparation included.

All concrete pipeline estimators should subtype `AbstractPipelineEstimator`.

See `docs/adr/0028-pipeline-workflow-estimator.md` for the design rationale.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractPipelineResult`](@ref)
  - [`PipelineContext`](@ref)
  - [`PipelineStep`](@ref)
"""
abstract type AbstractPipelineEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all pipeline result types in `PortfolioOptimisers.jl`.

Pipeline results carry the fitted per-step results of executing an [`AbstractPipelineEstimator`](@ref), the final [`PipelineContext`](@ref), and the terminal optimisation result when one exists.

All concrete pipeline results should subtype `AbstractPipelineResult`.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractPipelineEstimator`](@ref)
"""
abstract type AbstractPipelineResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing estimator types in `PortfolioOptimisers.jl`.

Preprocessing estimators transform price or returns data inside a pipeline (prices-to-returns conversion, missing-data filtering, imputation). Fitting one on training data produces a result carrying any fitted state — imputation parameters, thresholds, and the selected asset universe — which is then applied to unseen data so train and test windows are transformed consistently. Stateless preprocessing estimators carry no state and applying them is equivalent to running them.

All concrete preprocessing estimators should subtype one of the two data-level subtypes:

  - [`AbstractPricesPreprocessingEstimator`](@ref): consumes and produces price-level data ([`PricesResult`](@ref)).
  - [`AbstractReturnsPreprocessingEstimator`](@ref): consumes and produces returns-level data ([`ReturnsResult`](@ref)).

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPipelineEstimator`](@ref)
"""
abstract type AbstractPreprocessingEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce price-level data.

Concrete subtypes read and write the `prices` slot of a [`PipelineContext`](@ref), transforming a [`PricesResult`](@ref) into another [`PricesResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce returns-level data.

Concrete subtypes read and write the `returns` slot of a [`PipelineContext`](@ref), transforming a [`ReturnsResult`](@ref) into another [`ReturnsResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing result types in `PortfolioOptimisers.jl`.

Preprocessing results are produced by fitting an [`AbstractPreprocessingEstimator`](@ref) on training data. They carry the fitted state needed to apply the same transformation to unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators produce results that carry only their configuration.

All concrete preprocessing results should subtype `AbstractPreprocessingResult`.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
"""
abstract type AbstractPreprocessingResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all price-level data result types in `PortfolioOptimisers.jl`.

All concrete types representing price-level data should be subtypes of `AbstractPricesResult`.

# Related

  - [`AbstractResult`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

A container for aligned, time-indexed price-level data in `PortfolioOptimisers.jl`.

`PricesResult` is the prices-level mirror of [`ReturnsResult`](@ref): it bundles asset prices with optional factor, benchmark, and implied volatility series, all as `TimeSeries.TimeArray`s. It is the input to price-level preprocessing estimators and prices-to-returns conversion, and the type that defines timestamp-window slicing for pipeline cross-validation via [`prices_view`](@ref).

The asset price series `X` is the master clock: [`prices_view`](@ref) selects observation windows on `X` and aligns the other series to the selected timestamps.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesResult(;
        X::TimeSeries.TimeArray,
        F::Option{<:TimeSeries.TimeArray} = nothing,
        B::Option{<:TimeSeries.TimeArray} = nothing,
        iv::Option{<:TimeSeries.TimeArray} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
    ) -> PricesResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(X)`.
  - If `F` is not `nothing`: `!isempty(F)`.
  - If `B` is not `nothing`: `!isempty(B)`, and `size(values(B), 2) in (1, size(values(X), 2))`.
  - If `iv` is not `nothing`: `!isempty(iv)`, `all(x -> x >= 0, values(iv))`, `all(x -> isfinite(x), values(iv))`, and `size(values(iv), 2) == size(values(X), 2)`.
  - If `ivpa` is not `nothing`: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`; if a vector, `length(ivpa) == size(values(X), 2)`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], ["A", "B"]);

julia> pr = PricesResult(; X = X);

julia> size(values(pr.X))
(3, 2)
```

# Related

  - [`AbstractPricesResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`prices_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`Num_VecNum`](@ref)
"""
@concrete struct PricesResult <: AbstractPricesResult
    """
    Asset price data (observations × assets). The master clock for timestamp-window slicing.
    """
    X
    """
    Optional factor price data (observations × factors).
    """
    F
    """
    Optional benchmark price data (observations × 1) or (observations × assets).
    """
    B
    """
    Optional implied volatility data (observations × assets).
    """
    iv
    """
    Implied volatility risk premium adjustment, if a vector (assets × 1).
    """
    ivpa
    function PricesResult(X::TimeSeries.TimeArray, F::Option{<:TimeSeries.TimeArray},
                          B::Option{<:TimeSeries.TimeArray},
                          iv::Option{<:TimeSeries.TimeArray}, ivpa::Option{<:Num_VecNum})
        @argcheck(!isempty(X), IsEmptyError)
        if !isnothing(F)
            @argcheck(!isempty(F), IsEmptyError)
        end
        if !isnothing(B)
            @argcheck(!isempty(B), IsEmptyError)
            @argcheck(size(values(B), 2) in (1, size(values(X), 2)), DimensionMismatch)
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(values(iv), :iv)
            @argcheck(size(values(iv), 2) == size(values(X), 2), DimensionMismatch)
        end
        if !isnothing(ivpa)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(values(X), 2), DimensionMismatch)
            end
        end
        return new{typeof(X), typeof(F), typeof(B), typeof(iv), typeof(ivpa)}(X, F, B, iv,
                                                                              ivpa)
    end
end
function PricesResult(; X::TimeSeries.TimeArray,
                      F::Option{<:TimeSeries.TimeArray} = nothing,
                      B::Option{<:TimeSeries.TimeArray} = nothing,
                      iv::Option{<:TimeSeries.TimeArray} = nothing,
                      ivpa::Option{<:Num_VecNum} = nothing)::PricesResult
    return PricesResult(X, F, B, iv, ivpa)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `PricesResult` for the observation window `i` of the asset price series `X`.

The asset price series is the master clock: `i` selects rows of `X`, and the factor, benchmark, and implied volatility series are aligned to the selected timestamps (rows whose timestamps are absent from a series are dropped from that series).

# Arguments

  - `pr`: A `PricesResult` object.
  - `i`: Observation window into the rows of `pr.X`. Either integer indices (`AbstractVector{<:Integer}`, `AbstractRange`, or `Colon`) or a vector of timestamps (`AbstractVector{<:Dates.AbstractTime}`).

# Returns

  - `new_pr::PricesResult`: A new `PricesResult` containing only the data for the selected window.

# Details

  - `Colon` returns `pr` unchanged.
  - Integer windows index the rows of `pr.X` directly; the selected timestamps are then used to align `F`, `B`, and `iv`.
  - Timestamp windows are applied to all series directly.
  - `ivpa` is per-asset and passes through unchanged.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], ["A", "B"]);

julia> pr = PricesResult(; X = X);

julia> pv = PortfolioOptimisers.prices_view(pr, 2:3);

julia> first(timestamp(pv.X))
2020-01-02

julia> size(values(pv.X))
(2, 2)
```

# Related

  - [`PricesResult`](@ref)
  - [`returns_result_view`](@ref)
"""
function prices_view(pr::PricesResult, ::Colon)
    return pr
end
function prices_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime})
    X = pr.X[i]
    F = isnothing(pr.F) ? nothing : pr.F[i]
    B = isnothing(pr.B) ? nothing : pr.B[i]
    iv = isnothing(pr.iv) ? nothing : pr.iv[i]
    return PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = pr.ivpa)
end
function prices_view(pr::PricesResult, i::Union{<:VecInt, <:AbstractRange{<:Integer}})
    return prices_view(pr, TimeSeries.timestamp(pr.X)[i])
end
"""
$(DocStringExtensions.TYPEDEF)

The mu/sigma pair held by the `uncertainty` slot of a [`PipelineContext`](@ref).

A computed uncertainty-set result cannot always reveal which parameter it bounds (a `BoxUncertaintySet` may bound either the mean or the covariance), so the slot stores the two targets explicitly. Uncertainty-set steps declare their target through a [`PipelineStep`](@ref) wrapper (`target = :mu` or `target = :sigma`); each step fills its half of the pair, leaving the other untouched.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PipelineUncertaintySets(;
        mu::Option{<:AbstractUncertaintySetResult} = nothing,
        sigma::Option{<:AbstractUncertaintySetResult} = nothing,
    ) -> PipelineUncertaintySets

Keywords correspond to the struct's fields.

# Related

  - [`PipelineContext`](@ref)
  - [`PipelineStep`](@ref)
"""
@concrete struct PipelineUncertaintySets <: AbstractResult
    """
    Uncertainty set bounding the expected returns vector ([`AbstractUncertaintySetResult`](@ref)).
    """
    mu
    """
    Uncertainty set bounding the covariance matrix ([`AbstractUncertaintySetResult`](@ref)).
    """
    sigma
    function PipelineUncertaintySets(mu::Option{<:AbstractUncertaintySetResult},
                                     sigma::Option{<:AbstractUncertaintySetResult})
        return new{typeof(mu), typeof(sigma)}(mu, sigma)
    end
end
function PipelineUncertaintySets(; mu::Option{<:AbstractUncertaintySetResult} = nothing,
                                 sigma::Option{<:AbstractUncertaintySetResult} = nothing)::PipelineUncertaintySets
    return PipelineUncertaintySets(mu, sigma)
end
"""
    const PIPELINE_SLOTS = (:prices, :returns, :prior, :phylogeny, :uncertainty, :constraints, :opt)

The named slots of a [`PipelineContext`](@ref). Each pipeline step reads the slots it needs and writes the slot its estimator family produces.
"""
const PIPELINE_SLOTS = (:prices, :returns, :prior, :phylogeny, :uncertainty, :constraints,
                        :opt)
"""
$(DocStringExtensions.TYPEDEF)

The accumulating blackboard threaded through a pipeline's steps.

A `PipelineContext` holds one typed slot per stage of the workflow. Steps run in user-given order; each reads the slots it needs and writes the slot its estimator family produces. Heterogeneous slots (`uncertainty`, `constraints`) hold one result or a vector of results whose elements are routed to their optimiser targets by result type.

Internal machinery — not part of the user-facing API.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PipelineContext(;
        prices::Option{<:AbstractPricesResult} = nothing,
        returns::Option{<:AbstractReturnsResult} = nothing,
        prior::Option{<:AbstractPriorResult} = nothing,
        phylogeny::Option{<:AbstractPhylogenyResult} = nothing,
        uncertainty::Option{<:PipelineUncertaintySets} = nothing,
        constraints::Option{<:Union{<:AbstractConstraintResult, <:AbstractVector{<:AbstractConstraintResult}}} = nothing,
        opt::Option{<:OptimisationResult} = nothing,
    ) -> PipelineContext

Keywords correspond to the struct's fields.

# Related

  - [`PIPELINE_SLOTS`](@ref)
  - [`AbstractPipelineEstimator`](@ref)
  - [`PipelineStep`](@ref)
"""
@concrete struct PipelineContext <: AbstractResult
    """
    Price-level data ([`AbstractPricesResult`](@ref)).
    """
    prices
    """
    Returns-level data ([`AbstractReturnsResult`](@ref)).
    """
    returns
    """
    Prior statistics ([`AbstractPriorResult`](@ref)).
    """
    prior
    """
    Phylogeny structure ([`AbstractPhylogenyResult`](@ref)).
    """
    phylogeny
    """
    Uncertainty set results as an explicit mu/sigma pair ([`PipelineUncertaintySets`](@ref)).
    """
    uncertainty
    """
    Constraint result(s) ([`AbstractConstraintResult`](@ref) or a vector thereof).
    """
    constraints
    """
    Terminal optimisation result ([`OptimisationResult`](@ref)).
    """
    opt
    function PipelineContext(prices::Option{<:AbstractPricesResult},
                             returns::Option{<:AbstractReturnsResult},
                             prior::Option{<:AbstractPriorResult},
                             phylogeny::Option{<:AbstractPhylogenyResult},
                             uncertainty::Option{<:PipelineUncertaintySets},
                             constraints::Option{<:Union{<:AbstractConstraintResult,
                                                         <:AbstractVector{<:AbstractConstraintResult}}},
                             opt::Option{<:OptimisationResult})
        return new{typeof(prices), typeof(returns), typeof(prior), typeof(phylogeny),
                   typeof(uncertainty), typeof(constraints), typeof(opt)}(prices, returns,
                                                                          prior, phylogeny,
                                                                          uncertainty,
                                                                          constraints, opt)
    end
end
function PipelineContext(; prices::Option{<:AbstractPricesResult} = nothing,
                         returns::Option{<:AbstractReturnsResult} = nothing,
                         prior::Option{<:AbstractPriorResult} = nothing,
                         phylogeny::Option{<:AbstractPhylogenyResult} = nothing,
                         uncertainty::Option{<:PipelineUncertaintySets} = nothing,
                         constraints::Option{<:Union{<:AbstractConstraintResult,
                                                     <:AbstractVector{<:AbstractConstraintResult}}} = nothing,
                         opt::Option{<:OptimisationResult} = nothing)::PipelineContext
    return PipelineContext(prices, returns, prior, phylogeny, uncertainty, constraints, opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`PipelineContext`](@ref) slot a pipeline step writes.

The estimator's family determines the slot via dispatch on the existing abstract-type taxonomy. Estimators whose family cannot be inferred must be wrapped in a [`PipelineStep`](@ref); the fallback method throws an `ArgumentError` saying so.

# Arguments

  - `est`: The step estimator.

# Returns

  - `slot::Symbol`: One of [`PIPELINE_SLOTS`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.pipe_writes(EmpiricalPrior())
:prior
```

# Related

  - [`pipe_reads`](@ref)
  - [`PIPELINE_SLOTS`](@ref)
  - [`PipelineStep`](@ref)
"""
function pipe_writes(est)
    return throw(ArgumentError("cannot infer the pipeline slot written by a $(typeof(est)); wrap it in a PipelineStep to declare its reads/writes explicitly"))
end
pipe_writes(::AbstractPricesPreprocessingEstimator) = :prices
pipe_writes(::AbstractReturnsPreprocessingEstimator) = :returns
pipe_writes(::AbstractPriorEstimator) = :prior
pipe_writes(::AbstractPhylogenyEstimator) = :phylogeny
pipe_writes(::AbstractUncertaintySetEstimator) = :uncertainty
pipe_writes(::AbstractConstraintEstimator) = :constraints
pipe_writes(::OptimisationEstimator) = :opt
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`PipelineContext`](@ref) slots a pipeline step requires to be populated before it runs.

These are the *required* reads used for construction-time dependency validation, not every slot the step may consume. Slots an estimator can compute internally when absent (for example a phylogeny-constraint estimator's own phylogeny) are not listed.

# Arguments

  - `est`: The step estimator.

# Returns

  - `slots::Tuple{Vararg{Symbol}}`: Subset of [`PIPELINE_SLOTS`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.pipe_reads(EmpiricalPrior())
(:returns,)
```

# Related

  - [`pipe_writes`](@ref)
  - [`PIPELINE_SLOTS`](@ref)
  - [`PipelineStep`](@ref)
"""
function pipe_reads(est)
    return throw(ArgumentError("cannot infer the pipeline slots read by a $(typeof(est)); wrap it in a PipelineStep to declare its reads/writes explicitly"))
end
pipe_reads(::AbstractPricesPreprocessingEstimator) = (:prices,)
pipe_reads(::AbstractReturnsPreprocessingEstimator) = (:returns,)
pipe_reads(::AbstractPriorEstimator) = (:returns,)
pipe_reads(::AbstractPhylogenyEstimator) = (:returns,)
pipe_reads(::AbstractUncertaintySetEstimator) = (:returns,)
pipe_reads(::AbstractConstraintEstimator) = (:returns,)
pipe_reads(::OptimisationEstimator) = (:returns,)
"""
$(DocStringExtensions.TYPEDEF)

Explicit pipeline step wrapper — the escape hatch when dispatch cannot infer a step's slots.

Most estimators are used as pipeline steps directly: their family determines which [`PipelineContext`](@ref) slots they read and write via [`pipe_reads`](@ref)/[`pipe_writes`](@ref). `PipelineStep` wraps the cases dispatch cannot infer: custom callables, estimators routed to a nonstandard slot, or steps whose routing is ambiguous (for example an uncertainty-set estimator that must be pinned to the mean or covariance target via `target`).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PipelineStep(;
        est,
        writes::Symbol,
        reads::Tuple{Vararg{Symbol}} = (),
        target::Option{Symbol} = nothing,
    ) -> PipelineStep

Keywords correspond to the struct's fields.

## Validation

  - `writes in PIPELINE_SLOTS`.
  - `all(r -> r in PIPELINE_SLOTS, reads)`.

# Examples

```jldoctest
julia> ps = PipelineStep(; est = NormalUncertaintySet(), reads = (:returns,),
                         writes = :uncertainty, target = :mu);

julia> PortfolioOptimisers.pipe_writes(ps)
:uncertainty

julia> PortfolioOptimisers.pipe_reads(ps)
(:returns,)
```

# Related

  - [`pipe_reads`](@ref)
  - [`pipe_writes`](@ref)
  - [`PIPELINE_SLOTS`](@ref)
  - [`AbstractPipelineEstimator`](@ref)
"""
@concrete struct PipelineStep <: AbstractEstimator
    """
    The wrapped step: an estimator or a callable.
    """
    est
    """
    Slot the step requires to be populated before it runs (subset of [`PIPELINE_SLOTS`](@ref)).
    """
    reads
    """
    Slot the step writes (one of [`PIPELINE_SLOTS`](@ref)).
    """
    writes
    """
    Optional routing annotation for heterogeneous slots (for example `:mu` or `:sigma` for uncertainty sets).
    """
    target
    function PipelineStep(est::Union{<:AbstractEstimator, <:Function},
                          reads::Tuple{Vararg{Symbol}}, writes::Symbol,
                          target::Option{Symbol})
        @argcheck(writes in PIPELINE_SLOTS,
                  ArgumentError("writes must be one of $(PIPELINE_SLOTS), got :$writes"))
        @argcheck(all(r -> r in PIPELINE_SLOTS, reads),
                  ArgumentError("all reads must be members of $(PIPELINE_SLOTS), got $reads"))
        return new{typeof(est), typeof(reads), typeof(writes), typeof(target)}(est, reads,
                                                                               writes,
                                                                               target)
    end
end
function PipelineStep(; est::Union{<:AbstractEstimator, <:Function}, writes::Symbol,
                      reads::Tuple{Vararg{Symbol}} = (),
                      target::Option{Symbol} = nothing)::PipelineStep
    return PipelineStep(est, reads, writes, target)
end
pipe_writes(ps::PipelineStep) = ps.writes
pipe_reads(ps::PipelineStep) = ps.reads

export PricesResult, PipelineStep
