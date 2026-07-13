"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all portfolio optimisation estimators in `PortfolioOptimisers.jl`.

All optimisers and optimisation components should subtype `AbstractOptimisationEstimator` to participate in the optimisation dispatch system.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
"""
abstract type AbstractOptimisationEstimator <: AbstractEstimator end
"""
    const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}

Alias for a vector of portfolio optimisation estimators.

Represents a collection of [`AbstractOptimisationEstimator`](@ref) objects, used for dispatch in routines that process multiple optimisers simultaneously.

# Related

  - [`AbstractOptimisationEstimator`](@ref)
"""
const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base portfolio optimisation estimators.

`BaseOptimisationEstimator` is the parent for all internal optimiser components that configure the optimisation problem but are not directly invokable as top-level optimisers.

# Related

  - [`AbstractOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
"""
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce portfolio weights.

Subtype `OptimisationEstimator` to implement concrete portfolio optimisers. All optimisers that can be invoked with `optimise` should subtype this.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`AbstractOptimisationEstimator`](@ref)
"""
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce continuous (non-integer) portfolio weights.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
"""
abstract type NonFiniteAllocationOptimisationEstimator <: OptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation algorithms used by portfolio optimisers.

# Related

  - [`AbstractAlgorithm`](@ref)
"""
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation result types.

All concrete optimisation result types should subtype `OptimisationResult`.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptimisationReturnCode`](@ref)
"""
abstract type OptimisationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for continuous (non-integer allocation) optimisation results.

# Related

  - [`OptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalResult`](@ref)
  - [`MeanRiskResult`](@ref)
"""
abstract type NonFiniteAllocationOptimisationResult <: OptimisationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-JuMP continuous optimisation results.

Groups the results that do not carry a JuMP model (naive, clustering, and meta-optimiser results). Mirrors the JuMP/non-JuMP split on the result side; the JuMP half is [`RiskJuMPOptimisationResult`](@ref).

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalResult`](@ref)
"""
abstract type NonJuMPOptimisationResult <: NonFiniteAllocationOptimisationResult end
"""
    const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}

Alias for a vector of non-finite allocation optimisation results.

Represents a collection of [`NonFiniteAllocationOptimisationResult`](@ref) objects.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptE_Opt`](@ref)
"""
const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation return codes.

Concrete subtypes indicate whether an optimisation succeeded or failed.

# Related

  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
abstract type OptimisationReturnCode <: AbstractResult end
"""
    const VecOptRetCode = AbstractVector{<:OptimisationReturnCode}

Alias for a vector of optimisation return codes.

# Related

  - [`OptimisationReturnCode`](@ref)
"""
const VecOptRetCode = AbstractVector{<:OptimisationReturnCode}
"""
    const OptRetCode_VecOptRetCode = Union{<:OptimisationReturnCode, <:VecOptRetCode}

Alias for either a single optimisation return code or a vector of return codes.

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`VecOptRetCode`](@ref)
"""
const OptRetCode_VecOptRetCode = Union{<:OptimisationReturnCode, <:VecOptRetCode}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for intermediate optimisation model results.

# Related

  - [`OptimisationResult`](@ref)
"""
abstract type OptimisationModelResult <: AbstractResult end
"""
    const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                           <:NonFiniteAllocationOptimisationResult}

Alias for a non-finite allocation optimisation estimator or result.

Matches either a [`NonFiniteAllocationOptimisationEstimator`](@ref) (specifying an optimiser configuration) or a [`NonFiniteAllocationOptimisationResult`](@ref) (a pre-computed result). Used for dispatch in cross-validation and optimisation workflows that accept either form.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                       <:NonFiniteAllocationOptimisationResult}
"""
    factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt})

Rebuild a continuous optimisation result with an updated fallback optimiser `fb`.

Every optimisation result carries `fb` as its last field, so the generic rebuild copies all fields unchanged except the trailing `fb`. Concrete result types may override this method when rebuilding requires more than swapping `fb`.

# Related

  - [`OptE_Opt`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt})
    flds = ntuple(i -> getfield(res, i), Val(fieldcount(typeof(res))))
    return (typeof(res).name.wrapper)(Base.front(flds)..., fb)
end
"""
    assert_special_nco_requirements(opt)

Assert that the optimiser meets special requirements for Nested Clustered Optimisation (NCO).

The default implementation does nothing. Overridden for estimators (e.g. [`Stacking`](@ref)) that have requirements which must be validated before NCO can proceed.

# Arguments

  - `opt`: Optimisation estimator, result, or vector thereof.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
"""
function assert_special_nco_requirements(::OptE_Opt)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `opt` unchanged.

Default pass-through factory for optimisation estimators and results. Overridden for estimators that carry parameters requiring update at each optimisation step.

# Related

  - [`OptE_Opt`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::OptE_Opt, ::Any)
    return opt
end
"""
    needs_previous_weights(opt)

Return `true` if the optimiser requires the previous period's weights.

The default returns `false`. Overridden for optimisers that contain turnover constraints, tracking error constraints, or other time-dependent components that require the previous optimisation's weights.

# Arguments

  - `opt`: Optimisation estimator, result, risk measure, fee structure, or vector thereof.

# Returns

  - `Bool`: `true` if previous weights are needed.

# Related

  - [`is_time_dependent`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function needs_previous_weights(::OptE_Opt)
    return false
end
#! Start: Overload these for all estimators which can use time-dependent constraints.
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for callable structs used as time-dependent constraint values.

A subtype is a data-carrying alternative to a bare function inside a [`TimeDependent`](@ref): it must implement a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's field value. Because it is a struct, it participates in a trait a bare function cannot: define `needs_previous_weights(::MySubtype) = true` to declare a previous-weights requirement directly (the default is `false`), instead of wrapping in [`PreviousWeightsFunction`](@ref).

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`PreviousWeightsFunction`](@ref)
  - [`needs_previous_weights`](@ref)
"""
abstract type TimeDependentCallable <: AbstractAlgorithm end
function needs_previous_weights(::TimeDependentCallable)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for callable structs whose per-fold value is an *optimiser*.

A subtype implements a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's optimiser (an [`OptE_Opt`](@ref)), so a [`TimeDependent`](@ref) holding it is admissible wherever an optimiser-valued field accepts a schedule (see [`TD_OptE_Opt`](@ref)). Declaring the functor's output kind in the type is what makes the schedule *statically* admissible: a bare `ctx -> optimiser` is admitted as a [`Base.Callable`](@ref) and checked only when the fold loop swaps its value in.

# Related

  - [`TimeDependentCallable`](@ref)
  - [`TimeDependent`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`TimeDependentContext`](@ref)
"""
abstract type TimeDependentOptimiserCallable <: TimeDependentCallable end
"""
$(DocStringExtensions.TYPEDEF)

Wrapper marking a callable time-dependent constraint entry as requiring the previous optimisation's weights.

A bare callable inside a [`TimeDependent`](@ref) cannot be inspected for previous-weight requirements, so it contributes `false` to [`needs_previous_weights`](@ref) and its context's `w_prev` is only populated when something else makes the fold loop sequential. Wrapping the callable in `PreviousWeightsFunction` declares the requirement as data: it contributes `true` to [`needs_previous_weights`](@ref), forcing sequential fold execution and a populated `w_prev` in the [`TimeDependentContext`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PreviousWeightsFunction(; f)

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`needs_previous_weights`](@ref)
"""
struct PreviousWeightsFunction{T} <: AbstractAlgorithm
    """
    Callable evaluated per fold as `f(ctx::TimeDependentContext)`, returning the fold's field value.
    """
    f::T
    function PreviousWeightsFunction(f)
        return new{typeof(f)}(f)
    end
end
function PreviousWeightsFunction(; f)::PreviousWeightsFunction
    return PreviousWeightsFunction(f)
end
function needs_previous_weights(::PreviousWeightsFunction)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDEF)

Marker for "no default here", used in the two places a fold-less value may be missing.

  - As a [`TimeDependent`](@ref)'s `default`: the schedule states no fold-less value of its own, so a fold-less solve falls back to the field's static default (see [`time_dependent_field_defaults`](@ref)).
  - As an entry of a host's [`time_dependent_field_defaults`](@ref): the field is *required* and has no static default (the optimiser-valued fields), so a schedule there must carry its own `default`. A fold-less solve of a host whose required field holds a defaultless schedule throws a [`TimeDependentDefaultError`](@ref).

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentDefaultError`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
"""
struct NoDefault <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Exception thrown when a fold-less solve reaches a [`TimeDependent`](@ref) schedule that has no value to fall back to: the field has no static default and the schedule supplies no `default`.

A schedule is defined *only* over the folds of a cross-validation scheme. Fields with a static default reset to it silently; a required field (the optimiser-valued ones) has nothing to reset to, so the schedule must state the value a fold-less solve should use, via `TimeDependent(val; default = x)`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependentDefaultError(msg)

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`TimeDependent`](@ref)
  - [`NoDefault`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
"""
@concrete struct TimeDependentDefaultError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Time-dependent constraint: an optimiser input whose value changes across the folds of a cross-validation scheme.

A `TimeDependent` is stored *directly in the optimiser field it varies* — e.g. `JuMPOptimiser(; lt = TimeDependent([...]))` — so the field's position names the target and a field holds either a static value or a per-fold schedule, never both. It is recognised at top-level optimiser fields only, never nested inside another input (e.g. inside a [`Fees`](@ref) or a risk measure).

`val` is either a vector of per-fold values — entry `i` is the complete field value for fold `i` of the consuming scheme's `split` enumeration — or a callable evaluated per fold: a bare function `f(ctx::TimeDependentContext)` (optionally wrapped in [`PreviousWeightsFunction`](@ref)) or a [`TimeDependentCallable`](@ref) functor struct.

For a field that itself accepts a *vector of constraints* statically, a per-fold entry is that whole vector, so a schedule of per-fold constraint vectors is a vector of vectors — `TimeDependent([[c₁ᵃ, c₁ᵇ], [c₂ᵃ, c₂ᵇ], …])`, entry `i` being fold `i`'s complete constraint vector. There is no separate "vector of `TimeDependent`" facility and none is needed: `TimeDependent` is recognised only at a top-level field, so to vary individual constraints within a vector, build the fold's vector in a callable — `TimeDependent(ctx -> [dynamic(ctx), static])` — which keeps the shared static parts in one place.

The machinery imposes no ordering of its own: fold `i` is whatever `split(cv, rd)` enumerates `i`-th, which is chronological for walk-forward and (unshuffled) KFold schemes. For schemes whose enumeration is not a timeline (combinatorial splits, randomised paths) it is the user's responsibility to key entries off the fold's indices — a callable sees its own fold's windows via `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` and may derive any ordering from them.

A time-dependent constraint participates only where folds exist and is inert everywhere else — a fold-less `optimise` replaces it with the field's fold-less value (see [`reset_time_dependent_estimator`](@ref)). Vector entries must have length equal to the number of folds of the consuming cross-validation scheme, validated at `split` time. Entries may be `nothing`, giving the field `nothing` for that fold.

The fold-less value is the field's static default, unless `default` overrides it. A field with *no* static default — the required, optimiser-valued fields — has nothing to reset to, so a schedule there **must** supply `default`; a fold-less solve of one that does not throws a [`TimeDependentDefaultError`](@ref).

A vector whose entries are all optimisers or precomputed results ([`OptE_Opt`](@ref)) is stored as a `Vector{OptE_Opt}`, so a *mixed* schedule — fold `i` optimising or predicting depending on what entry `i` is — is admissible in an optimiser-valued field on its element type alone (see [`TD_OptE_Opt`](@ref)) rather than falling out to a `Vector{Any}` the field cannot accept.

Schedules do not nest: neither `val`, nor a vector entry of `val`, nor `default` may be a `TimeDependent`. Entry `i` is fold `i`'s *complete* field value, and the fold-less value is by definition outside every fold loop, so nesting has no meaning. An estimator swapped in by a schedule may itself carry schedules — those resolve against the same fold context after the swap — but they live in *its* fields, not inside this wrapper.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependent(val, bind::Symbol = :outermost; default = NoDefault())
    TimeDependent(; val, bind::Symbol = :outermost, default = NoDefault())

## Validation

  - If `val` is a vector: `!isempty(val)`, and no entry is a `TimeDependent`.
  - `val` is not a `TimeDependent`.
  - `default` is not a `TimeDependent`.
  - `bind in (:outermost, :nearest)`.

# Examples

```jldoctest
julia> TimeDependent([Fees(; l = 0.001), Fees(; l = 0.002)])
TimeDependent
      val ┼ 2-element Vector{Fees}
          │ Fees ⋯
          │ Fees ⋯
     bind ┼ Symbol: :outermost
  default ┴ NoDefault()
```

# Related

  - [`TimeDependentContext`](@ref)
  - [`PreviousWeightsFunction`](@ref)
  - [`NoDefault`](@ref)
  - [`TimeDependentDefaultError`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`is_time_dependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
"""
struct TimeDependent{T1, T2} <: AbstractEstimator
    """
    Vector of per-fold values (in the consuming scheme's `split` enumeration order), or a callable of the fold's [`TimeDependentContext`](@ref): a bare function (optionally wrapped in [`PreviousWeightsFunction`](@ref)) or a [`TimeDependentCallable`](@ref) functor struct.
    """
    val::T1
    """
    Which fold loop consumes the schedule: `:outermost` (default) binds it to the outermost fold loop processing the estimator tree; `:nearest` binds it to the nearest enclosing fold loop — inside a meta-optimiser's inner estimators that is the meta's own cross-validation leg, which then consumes the schedule even when the meta is backtested under an outer fold loop.
    """
    bind::Symbol
    """
    Value the field takes outside every fold loop, overriding the host's static default (see [`time_dependent_field_defaults`](@ref)). [`NoDefault`](@ref) (the default) defers to the host's static default; a field that has none requires this to be set.
    """
    default::T2
    function TimeDependent(val::Union{<:AbstractVector, <:Base.Callable,
                                      <:PreviousWeightsFunction, <:TimeDependentCallable},
                           bind::Symbol = :outermost; default = NoDefault())
        if isa(val, AbstractVector)
            @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
            @argcheck(!any(x -> isa(x, TimeDependent), val),
                      ArgumentError("no entry of val may be a TimeDependent: entry i is fold i's complete field value, so schedules do not nest. To vary parts of a vector-valued field, assemble the fold's vector in a callable: TimeDependent(ctx -> [dynamic(ctx), static])."))
            if !(eltype(val) <: OptE_Opt) && all(x -> isa(x, OptE_Opt), val)
                val = convert(Vector{OptE_Opt}, val)
            end
        end
        @argcheck(!isa(default, TimeDependent),
                  ArgumentError("default cannot be a TimeDependent: it is the field's value outside every fold loop, where a schedule is undefined."))
        @argcheck(bind in (:outermost, :nearest),
                  ArgumentError("bind must be :outermost or :nearest, got :$bind"))
        return new{typeof(val), typeof(default)}(val, bind, default)
    end
end
function TimeDependent(::TimeDependent, args...; kwargs...)
    return throw(ArgumentError("val cannot be a TimeDependent: schedules do not nest. An estimator swapped in by a schedule may carry schedules of its own — they resolve against the same fold context after the swap — but they belong in its fields, not inside this wrapper."))
end
function TimeDependent(;
                       val::Union{<:AbstractVector, <:Base.Callable,
                                  <:PreviousWeightsFunction, <:TimeDependentCallable,
                                  <:TimeDependent}, bind::Symbol = :outermost,
                       default = NoDefault())::TimeDependent
    return TimeDependent(val, bind; default = default)
end
"""
    const TD_Option{X} = Union{Nothing, <:TimeDependent, X}

Alias for an optimiser field that accepts `nothing`, a static value of type `X`, or a per-fold [`TimeDependent`](@ref) schedule.

The set of fields whose constructor signatures use this alias is the single source of truth for which optimiser inputs may vary over folds.

# Related

  - [`TimeDependent`](@ref)
  - [`Option`](@ref)
"""
const TD_Option{X} = Union{Nothing, <:TimeDependent, X}
"""
    const TD{X} = Union{<:TimeDependent, X}

Alias for a *required* optimiser field that accepts a static value of type `X` or a per-fold [`TimeDependent`](@ref) schedule, but not `nothing`.

The problem-definition fields that always carry a value — the prior estimator, the returns model, the scalariser, the clustering estimator, the weight finaliser — are time-dependent through this alias rather than [`TD_Option`](@ref), so `nothing` stays inadmissible where it was never a legal static value. Such a field still has a *static default*, so a schedule in one resets to that default on a fold-less solve, unlike the optimiser-valued fields (see [`TD_OptE_Opt`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`TD_Option`](@ref)
  - [`time_dependent_field_defaults`](@ref)
"""
const TD{X} = Union{<:TimeDependent, X}
"""
    const TD_OptE_Opt = Union{TimeDependent{<:AbstractVector{<:OptE_Opt}},
                              TimeDependent{<:TimeDependentOptimiserCallable},
                              TimeDependent{<:PreviousWeightsFunction},
                              TimeDependent{<:Base.Callable}}

The [`TimeDependent`](@ref) forms admissible in an *optimiser-valued* field — where the scheduled thing is the optimiser itself, not one of its inputs.

Two of the four are statically checked: a vector schedule whose entries are all [`OptE_Opt`](@ref) (an optimiser or a precomputed result — a mixed schedule is allowed, fold `i` optimising or predicting depending on what entry `i` is), and a [`TimeDependentOptimiserCallable`](@ref), which declares its output kind in its type. The other two — a bare `ctx -> optimiser` and a [`PreviousWeightsFunction`](@ref) wrapping one — cannot be checked before they run, so their output is checked when the fold loop swaps it into the field, by the host's own keyword constructor.

Because an optimiser-valued field is *required*, a schedule in one has no static default to reset to on a fold-less solve and must supply `default` (see [`NoDefault`](@ref), [`TimeDependentDefaultError`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentOptimiserCallable`](@ref)
  - [`TDO_Option`](@ref)
  - [`TD_Option`](@ref)
  - [`OptE_Opt`](@ref)
"""
const TD_OptE_Opt = Union{TimeDependent{<:AbstractVector{<:OptE_Opt}},
                          TimeDependent{<:TimeDependentOptimiserCallable},
                          TimeDependent{<:PreviousWeightsFunction},
                          TimeDependent{<:Base.Callable}}
"""
    const TDO_Option{X} = Union{Nothing, <:TD_OptE_Opt, X}

Alias for an *optional* optimiser-valued field (e.g. a fallback) that accepts `nothing`, a static value of type `X`, or a per-fold schedule of optimisers.

A required optimiser-valued field spells its union out — `Union{<:X, <:TD_OptE_Opt}` — since `nothing` is not one of its values.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`TD_Option`](@ref)
  - [`Option`](@ref)
"""
const TDO_Option{X} = Union{Nothing, <:TD_OptE_Opt, X}
"""
    const OptE_TD = Union{<:NonFiniteAllocationOptimisationEstimator, <:TD_OptE_Opt}

Alias for an optimisation estimator, or a [`TimeDependent`](@ref) schedule standing in its place.

This is the entry-point type of the cross-validation fold loops that *fit*: a schedule handed straight to [`cross_val_predict`](@ref) is the optimiser, and fold `i` runs entry `i`. Precomputed results are excluded because a bare result takes the predict-only path, which has no fold loop to resolve a schedule against — but a schedule *whose entries* are results is admissible here, and each such entry takes the predict-only path per fold (see [`OptE_Opt_TD`](@ref)).

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`OptE_Opt_TD`](@ref)
  - [`cross_val_predict`](@ref)
"""
const OptE_TD = Union{<:NonFiniteAllocationOptimisationEstimator, <:TD_OptE_Opt}
"""
    const OptE_Opt_TD = Union{<:OptE_Opt, <:TD_OptE_Opt}

Alias for an optimisation estimator or a precomputed result, or a [`TimeDependent`](@ref) schedule standing in their place.

The entry-point type of the fold loops that accept a precomputed result as well as an estimator. A schedule's entries are [`OptE_Opt`](@ref), so a *mixed* schedule is admissible: fold `i` optimises when entry `i` is an estimator and predicts when it is a result, which the single-fold [`fit_and_predict`](@ref) methods already distinguish by dispatch.

# Related

  - [`OptE_TD`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`OptE_Opt`](@ref)
"""
const OptE_Opt_TD = Union{<:OptE_Opt, <:TD_OptE_Opt}
"""
$(DocStringExtensions.TYPEDEF)

Per-fold context handed to time-dependent constraints when they are resolved.

Carries the fold's position in the consuming scheme's `split` enumeration and the data needed for a callable entry to compute its value. `i` indexes `train_idx`/`test_idx`, so `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` are always the fold's own windows; no ordering beyond the scheme's enumeration is implied. `rd` is the fold's (possibly asset-viewed) returns data, so callables see the current universe and timestamps. `w_prev` is populated only when the fold loop runs sequentially and a previous fold exists; `path_id` only under multi-path schemes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependentContext(;
        i::Integer, n::Integer, rd::ReturnsResult, train_idx, test_idx,
        w_prev::Option{<:VecNum} = nothing, path_id::Option{<:Integer} = nothing
    )

# Related

  - [`TimeDependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
"""
struct TimeDependentContext{T1, T2, T3, T4, T5, T6, T7} <: AbstractResult
    """
    Index of the fold within the scheme's `split` enumeration (1-based); indexes `train_idx`/`test_idx`.
    """
    i::T1
    """
    Number of folds within the path.
    """
    n::T2
    """
    The fold loop's (possibly asset-viewed) returns data.
    """
    rd::T3
    """
    Per-path training index vectors.
    """
    train_idx::T4
    """
    Per-path test index vectors.
    """
    test_idx::T5
    """
    Previous fold's portfolio weights, when threaded; `nothing` otherwise.
    """
    w_prev::T6
    """
    Path identifier under multi-path schemes; `nothing` otherwise.
    """
    path_id::T7
    function TimeDependentContext(i::Integer, n::Integer, rd::ReturnsResult, train_idx,
                                  test_idx, w_prev::Option{<:VecNum},
                                  path_id::Option{<:Integer})
        @argcheck(1 <= i <= n, DomainError(i, "fold index i must be in 1:$n"))
        return new{typeof(i), typeof(n), typeof(rd), typeof(train_idx), typeof(test_idx),
                   typeof(w_prev), typeof(path_id)}(i, n, rd, train_idx, test_idx, w_prev,
                                                    path_id)
    end
end
function TimeDependentContext(; i::Integer, n::Integer, rd::ReturnsResult, train_idx,
                              test_idx, w_prev::Option{<:VecNum} = nothing,
                              path_id::Option{<:Integer} = nothing)::TimeDependentContext
    return TimeDependentContext(i, n, rd, train_idx, test_idx, w_prev, path_id)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a time-dependent constraint to its value for the fold described by `ctx`.

Vector values index entry `ctx.i`; callables are invoked with `ctx`.

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function time_dependent_value(td::TimeDependent, ctx::TimeDependentContext)
    v = td.val
    if isa(v, AbstractVector)
        return v[ctx.i]
    elseif isa(v, PreviousWeightsFunction)
        return v.f(ctx)
    end
    return v(ctx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if a time-dependent constraint requires the previous optimisation's weights.

`true` for a [`PreviousWeightsFunction`](@ref) value; for vector values, delegates to [`needs_previous_weights`](@ref) on entries that support it (turnover, fees, tracking), descending into per-fold vector entries. Bare callables contribute `false` — their output cannot be inspected.

# Related

  - [`TimeDependent`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(td::TimeDependent)::Bool
    v = td.val
    if isa(v, PreviousWeightsFunction)
        return true
    elseif isa(v, TimeDependentCallable)
        return needs_previous_weights(v)
    elseif isa(v, AbstractVector)
        return any(time_dependent_entry_needs_previous_weights, v)
    end
    return false
end
"""
    time_dependent_entry_needs_previous_weights(x)

Return `true` if a per-fold entry value of a [`TimeDependent`](@ref) requires the previous optimisation's weights.

Delegates to [`needs_previous_weights`](@ref) for the value types that support the trait (turnover, fees, tracking); every other value contributes `false`.

# Related

  - [`TimeDependent`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function time_dependent_entry_needs_previous_weights(::Any)::Bool
    return false
end
function time_dependent_entry_needs_previous_weights(x::Union{<:TnE_Tn, <:FeesE_Fees,
                                                              <:Tr_VecTr})::Bool
    return needs_previous_weights(x)
end
function time_dependent_entry_needs_previous_weights(x::AbstractVector)::Bool
    return any(time_dependent_entry_needs_previous_weights, x)
end
function time_dependent_entry_needs_previous_weights(x::OptE_Opt)::Bool
    return needs_previous_weights(x)
end
function port_opt_view(td::TimeDependent, i, args...)
    v = td.val
    if isa(v, AbstractVector)
        v = [port_opt_view(x, i, args...) for x in v]
    end
    return TimeDependent(v, td.bind; default = port_opt_view(td.default, i, args...))
end
"""
    time_dependent_fields(opt, all_binds::Bool = true)

Return the tuple of field names of `opt` whose values are [`TimeDependent`](@ref).

The scan is generic over `fieldnames`, so the widened constructor signatures (see [`TD_Option`](@ref)) remain the single source of truth for which fields may vary over folds — there is no hand-maintained list.

# The `all_binds` argument

`all_binds` encodes something the schedule's own `bind` field cannot: it is a property of the *recursion position*, not of the schedule. A [`TimeDependent`](@ref)'s `bind` (`:outermost` / `:nearest`) says *which* fold loop the schedule wants; `all_binds` says whether the loop currently recursing is *entitled* to consume nearest-bound schedules at this depth. The second fact is not on the schedule.

Why position matters: under `outer CV loop → meta → (meta's inner CV loop) → inner estimator with a :nearest field`, the same `:nearest` field is visited by two loops. The outer loop recurses through the meta (mandatory — that recursion is how an inner estimator's `:outermost` field is resolved against the outer folds) and must *skip* the `:nearest` field, because it is not the nearest enclosing loop. The meta's inner CV loop drives the same estimator directly and must *consume* it, because it is. Same field, same `bind`, opposite actions — the difference is whether a nearer fold-loop boundary was crossed to reach it, which is exactly what `all_binds` carries.

So `all_binds` is `true` at every ordinary (outermost/standalone) fold loop — which is both outermost and nearest, and therefore takes everything remaining, including `:nearest`. It is forced to `false` only where a meta-optimiser recurses into the estimators its own inner CV owns, leaving their `:nearest` schedules for that inner loop. With `all_binds = false`, only fields with `bind === :outermost` are returned.

# Related

  - [`TimeDependent`](@ref)
  - [`TD_Option`](@ref)
  - [`is_time_dependent`](@ref)
"""
function time_dependent_fields(opt, all_binds::Bool = true)
    fns = fieldnames(typeof(opt))
    return filter(f -> begin
                      x = getfield(opt, f)
                      if isa(x, TimeDependent)
                          (all_binds || x.bind === :outermost)
                      else
                          false
                      end
                  end, fns)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Test-substitute every vector entry of the [`TimeDependent`](@ref)-valued fields in `args` through the keyword constructor of `T`.

`args` holds the host constructor's arguments; `defaults` the static defaults of the fields that may be time-dependent (see [`time_dependent_field_defaults`](@ref)). Each per-fold entry, and an explicit `default`, is substituted into its field — with every other time-dependent field standing at a value of its own (see [`time_dependent_stand_in`](@ref)) — and the constructor re-run, surfacing type and cross-field errors at construction time instead of mid-backtest. Substituted calls contain no `TimeDependent` values, so the recursion terminates.

Validation is skipped when a time-dependent field has no stand-in at all — a callable schedule in a required field, whose value only exists once a fold context does.

# Related

  - [`TimeDependent`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`time_dependent_stand_in`](@ref)
"""
function assert_time_dependent_substitution(::Type{T}, args::NamedTuple,
                                            defaults::NamedTuple)::Nothing where {T}
    tdfs = filter(f -> isa(args[f], TimeDependent), keys(args))
    if isempty(tdfs)
        return nothing
    end
    stand_ins = map(f -> time_dependent_stand_in(args[f], defaults, f), tdfs)
    if any(isnothing, stand_ins)
        return nothing
    end
    base = merge(args, NamedTuple{tdfs}(map(something, stand_ins)))
    for f in tdfs
        td = args[f]
        v = td.val
        if isa(v, AbstractVector)
            for x in v
                T(; merge(base, NamedTuple{(f,)}((x,)))...)
            end
        end
        d = td.default
        if !isa(d, NoDefault)
            T(; merge(base, NamedTuple{(f,)}((d,)))...)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a valid static value for a [`TimeDependent`](@ref)-valued field, wrapped in `Some`, or `nothing` if none exists.

Used by [`assert_time_dependent_substitution`](@ref) to stand the *other* time-dependent fields at a valid value while it test-substitutes one of them: the schedule's `default`, else the field's static default, else the schedule's first entry. A callable schedule in a field with no default of either kind has no stand-in — its value exists only inside a fold — so it returns `nothing` and validation is skipped.

Unlike [`time_dependent_reset_value`](@ref) this never throws: a schedule without a fold-less value is legitimate at construction time and only fails if it reaches a fold-less solve.

# Related

  - [`assert_time_dependent_substitution`](@ref)
  - [`time_dependent_reset_value`](@ref)
  - [`NoDefault`](@ref)
"""
function time_dependent_stand_in(td::TimeDependent, defaults::NamedTuple, field::Symbol)
    d = td.default
    if !isa(d, NoDefault)
        return Some(d)
    end
    d = get(defaults, field, nothing)
    if !isa(d, NoDefault)
        return Some(d)
    end
    v = td.val
    return isa(v, AbstractVector) ? Some(v[1]) : nothing
end
"""
    assert_time_dependent_fold_count(opt, n::Integer, all_binds::Bool = true)

Assert that every vector-valued time-dependent constraint in `opt` has exactly `n` entries.

Called by the cross-validation fold loops immediately after `split`, before any fold runs. The default is a no-op; hosts scan their [`time_dependent_fields`](@ref) and wrapper optimisers recurse. When `all_binds` is `false`, `bind === :nearest` schedules are skipped — they are validated by the nearest enclosing fold loop against its own fold count instead (see [`TimeDependent`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`is_time_dependent`](@ref)
"""
function assert_time_dependent_fold_count(::OptE_Opt, ::Integer, ::Bool = true)::Nothing
    return nothing
end
function assert_time_dependent_fold_count(::Nothing, ::Integer, ::Bool = true)::Nothing
    return nothing
end
function assert_time_dependent_fold_count(td::TimeDependent, n::Integer,
                                          field::Symbol)::Nothing
    v = td.val
    if isa(v, AbstractVector)
        @argcheck(length(v) == n,
                  DimensionMismatch("time-dependent entries for $field ($(length(v))) must equal the number of folds ($n)"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert the fold count of every [`TimeDependent`](@ref)-valued field of a host optimiser.

# Related

  - [`assert_time_dependent_fold_count`](@ref)
  - [`time_dependent_fields`](@ref)
"""
function assert_time_dependent_fields_fold_count(opt, n::Integer,
                                                 all_binds::Bool = true)::Nothing
    for f in time_dependent_fields(opt, all_binds)
        assert_time_dependent_fold_count(getfield(opt, f), n, f)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the base optimiser configuration carries time-dependent constraints.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`TimeDependent`](@ref)
  - [`is_time_dependent`](@ref)
"""
function is_time_dependent(opt::BaseOptimisationEstimator)
    return !isempty(time_dependent_fields(opt))
end
function assert_time_dependent_fold_count(opt::BaseOptimisationEstimator, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the time-dependent constraints of a base optimiser configuration for the fold described by `ctx`.

Rebuilds the configuration through its validated keyword constructor with each [`TimeDependent`](@ref)-valued field replaced by its resolved per-fold value, so the result is an ordinary static configuration. When `all_binds` is `false`, `bind === :nearest` fields are left in place for the nearest enclosing fold loop to consume.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`update_time_dependent_fields`](@ref)
"""
function update_time_dependent_estimator(opt::BaseOptimisationEstimator,
                                         ctx::TimeDependentContext, all_binds::Bool = true)
    return update_time_dependent_fields(opt, ctx, all_binds)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace the time-dependent constraints of a base optimiser configuration with their static defaults (see [`time_dependent_field_defaults`](@ref)).

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`reset_time_dependent_fields`](@ref)
"""
function reset_time_dependent_estimator(opt::BaseOptimisationEstimator)
    return reset_time_dependent_fields(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild an estimator through its keyword constructor with the fields in `repl` replaced.

All remaining fields are carried through unchanged. Because the rebuild goes through the validated keyword constructor, every construction invariant re-runs.
"""
function rebuild_estimator(x, repl::NamedTuple)
    fns = fieldnames(typeof(x))
    nt = NamedTuple{fns}(map(f -> getfield(x, f), fns))
    return (typeof(x).name.wrapper)(; merge(nt, repl)...)
end
"""
    is_time_dependent(opt)

Return `true` if the optimiser carries time-dependent constraints.

The default returns `false`. Hosts return `true` when any of their fields holds a [`TimeDependent`](@ref) (see [`time_dependent_fields`](@ref)); wrapper optimisers recurse into their inner optimiser and fallback.

# Arguments

  - `opt`: Optimisation estimator, result, or vector thereof.

# Returns

  - `Bool`: `true` if the estimator is time-dependent.

# Related

  - [`TimeDependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function is_time_dependent(::OptE_Opt)
    return false
end
function is_time_dependent(::Nothing)
    return false
end
"""
    update_time_dependent_estimator(opt, ctx::TimeDependentContext, all_binds::Bool = true)

Resolve the time-dependent constraints of `opt` for the fold described by `ctx`.

The default returns the estimator unchanged. Hosts rebuild themselves through their validated keyword constructor with each [`TimeDependent`](@ref)-valued field replaced by its resolved per-fold value, so the result is an ordinary static estimator; wrapper optimisers recurse.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `ctx::TimeDependentContext`: The fold's context.
  - `all_binds::Bool`: When `false`, `bind === :nearest` schedules are skipped, leaving them for the nearest enclosing fold loop to consume. Meta-optimisers pass `false` when recursing into the estimators their internal fold loop processes; fold loops call with the default `true`.

# Returns

  - Updated estimator.

# Related

  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`is_time_dependent`](@ref)
"""
function update_time_dependent_estimator(opt::OptE_Opt, ::TimeDependentContext,
                                         ::Bool = true)
    return opt
end
function update_time_dependent_estimator(::Nothing, ::TimeDependentContext, ::Bool = true)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a host optimiser with each [`TimeDependent`](@ref)-valued field replaced by its per-fold value for `ctx`.

Shared implementation behind the hosts' [`update_time_dependent_estimator`](@ref) methods. Returns `opt` unchanged when no field is time-dependent.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`time_dependent_value`](@ref)
"""
function update_time_dependent_fields(opt, ctx::TimeDependentContext,
                                      all_binds::Bool = true)
    tdfs = time_dependent_fields(opt, all_binds)
    if isempty(tdfs)
        return opt
    end
    repl = NamedTuple{tdfs}(map(f -> time_dependent_value(getfield(opt, f), ctx), tdfs))
    return rebuild_estimator(opt, repl)
end
"""
    time_dependent_field_defaults(opt)

Return a `NamedTuple` of the static defaults of the optimiser fields that may hold a [`TimeDependent`](@ref), for those whose default is not `nothing`.

Used by [`reset_time_dependent_estimator`](@ref) to replace per-fold schedules with their static defaults on fold-less solves; fields absent from the tuple default to `nothing`. A *required* field — one with no static default at all, i.e. the optimiser-valued fields — is listed with [`NoDefault`](@ref), which is not a value it can take but a declaration that a schedule there must carry its own `default`. The fallback method returns an empty tuple.

# Related

  - [`reset_time_dependent_estimator`](@ref)
  - [`time_dependent_reset_value`](@ref)
  - [`NoDefault`](@ref)
  - [`TD_Option`](@ref)
"""
function time_dependent_field_defaults(::Any)::NamedTuple
    return (;)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the value a [`TimeDependent`](@ref)-valued field takes outside every fold loop.

The schedule's own `default` wins; absent one ([`NoDefault`](@ref)), the host's static default for `field` is used ([`time_dependent_field_defaults`](@ref), `nothing` for fields it omits). Throws a [`TimeDependentDefaultError`](@ref) when neither exists — a schedule in a required field that never said what a fold-less solve should do.

# Related

  - [`reset_time_dependent_fields`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
function time_dependent_reset_value(td::TimeDependent, defaults::NamedTuple, field::Symbol,
                                    opt)
    d = td.default
    if !isa(d, NoDefault)
        return d
    end
    d = get(defaults, field, nothing)
    if isa(d, NoDefault)
        throw(TimeDependentDefaultError("field `$field` of $(nameof(typeof(opt))) holds a TimeDependent schedule, but the field has no static default and the schedule supplies none, so there is no value to use outside a fold loop. A schedule is defined only over the folds of a cross-validation scheme; this solve has none. Give the schedule a fold-less value: TimeDependent(val; default = x)."))
    end
    return d
end
"""
    reset_time_dependent_estimator(opt)

Replace every [`TimeDependent`](@ref)-valued field of `opt` with its static default, recursing through wrapper optimisers.

A time-dependent constraint is defined only over the folds of a cross-validation scheme, so a fold-less solve runs with the affected fields at their static defaults (see [`time_dependent_field_defaults`](@ref)). Called at the top of the `_optimise` methods; per-fold estimators produced by [`update_time_dependent_estimator`](@ref) contain no `TimeDependent` values, so they pass through unchanged. The default returns the estimator unchanged; hosts rebuild themselves, wrapper optimisers recurse.

# Related

  - [`TimeDependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`is_time_dependent`](@ref)
"""
function reset_time_dependent_estimator(opt::OptE_Opt)
    return opt
end
function reset_time_dependent_estimator(::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a host optimiser with each [`TimeDependent`](@ref)-valued field replaced by its fold-less value (see [`time_dependent_reset_value`](@ref)).

Shared implementation behind the hosts' [`reset_time_dependent_estimator`](@ref) methods. Returns `opt` unchanged when no field is time-dependent.

# Related

  - [`reset_time_dependent_estimator`](@ref)
  - [`time_dependent_reset_value`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`time_dependent_fields`](@ref)
"""
function reset_time_dependent_fields(opt)
    tdfs = time_dependent_fields(opt)
    if isempty(tdfs)
        return opt
    end
    defaults = time_dependent_field_defaults(opt)
    repl = NamedTuple{tdfs}(map(f -> time_dependent_reset_value(getfield(opt, f), defaults,
                                                                f, opt), tdfs))
    return rebuild_estimator(opt, repl)
end
#! End: Overload these for all estimators which can use time-dependent constraints.
#! Begin: TimeDependent as an optimiser in its own right.
"""
$(DocStringExtensions.TYPEDSIGNATURES)

A [`TimeDependent`](@ref) schedule is time-dependent by construction.

# Related

  - [`is_time_dependent`](@ref)
"""
function is_time_dependent(::TimeDependent)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a [`TimeDependent`](@ref) schedule in an optimiser position resolved to something that can be optimised or predicted.

The vector and [`TimeDependentOptimiserCallable`](@ref) forms of a schedule declare their output kind in their type and are checked statically (see [`TD_OptE_Opt`](@ref)). The two callable forms — a bare `ctx -> optimiser` and a [`PreviousWeightsFunction`](@ref) wrapping one — cannot be, so their output is checked here, when the fold loop swaps it in.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`update_time_dependent_estimator`](@ref)
"""
function assert_time_dependent_optimiser(::OptE_Opt)::Nothing
    return nothing
end
function assert_time_dependent_optimiser(opt)::Nothing
    return throw(ArgumentError("a TimeDependent schedule in an optimiser position resolved to a $(typeof(opt)), which is neither an optimisation estimator nor a precomputed optimisation result. A callable schedule standing in for an optimiser must return an OptE_Opt for every fold."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a [`TimeDependent`](@ref) schedule standing in for an optimiser to the optimiser of fold `ctx.i`.

Entry `i` may be an estimator or a precomputed result, so a *mixed* schedule optimises on some folds and predicts on others. After the swap the resolved estimator is recursed into with the **same** context, so its own `:outermost` schedules bind to this fold loop rather than going unresolved.

Returns the schedule unchanged when `all_binds` is `false` and it is not `:outermost`-bound — a `:nearest` schedule in an optimiser position is consumed by a fold loop the host itself opens, not by the loop that reached the host.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`assert_time_dependent_optimiser`](@ref)
"""
function update_time_dependent_estimator(td::TD_OptE_Opt, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    if !all_binds && td.bind !== :outermost
        return td
    end
    opt = time_dependent_value(td, ctx)
    assert_time_dependent_optimiser(opt)
    return update_time_dependent_estimator(opt, ctx, all_binds)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a [`TimeDependent`](@ref) schedule standing in for an optimiser has one entry per fold, and that the schedules *within* each entry are sized to the same fold loop.

Entry `i` runs at fold `i` of this loop, so its own `:outermost` schedules bind here too (see [`update_time_dependent_estimator`](@ref)) and are validated against this loop's fold count. The `default` is not — it runs only outside a fold loop, where its schedules reset instead.

Skipped when `all_binds` is `false` and the schedule is not `:outermost`-bound — the fold loop the host opens validates it against its own fold count instead.

# Related

  - [`assert_time_dependent_fold_count`](@ref)
  - [`TD_OptE_Opt`](@ref)
"""
function assert_time_dependent_fold_count(td::TD_OptE_Opt, n::Integer,
                                          all_binds::Bool = true)::Nothing
    if !all_binds && td.bind !== :outermost
        return nothing
    end
    v = td.val
    if isa(v, AbstractVector)
        @argcheck(length(v) == n,
                  DimensionMismatch("a TimeDependent schedule of optimisers has $(length(v)) entries, which must equal the number of folds ($n)"))
        for opt in v
            assert_time_dependent_fold_count(opt, n, all_binds)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the optimiser a [`TimeDependent`](@ref) schedule takes outside every fold loop.

An optimiser position is *required* — there is no static default to fall back to — so the schedule must supply its own `default`, and one that does not throws a [`TimeDependentDefaultError`](@ref). The fold-less optimiser is itself reset, so its own schedules resolve to their defaults too.

# Related

  - [`reset_time_dependent_estimator`](@ref)
  - [`NoDefault`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
function reset_time_dependent_estimator(td::TD_OptE_Opt)
    d = td.default
    if isa(d, NoDefault)
        throw(TimeDependentDefaultError("a TimeDependent schedule stands in for the optimiser itself but supplies no `default`, so there is no optimiser to run outside a fold loop. A schedule is defined only over the folds of a cross-validation scheme; this solve has none. Give the schedule a fold-less optimiser: TimeDependent(val; default = opt)."))
    end
    return reset_time_dependent_estimator(d)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the statically inspectable entries of a [`TimeDependent`](@ref) schedule: the per-fold values of a vector schedule, plus its `default` when it has one. A callable schedule contributes nothing — its per-fold values cannot be inspected before it runs.

# Related

  - [`TimeDependent`](@ref)
"""
function time_dependent_entries(td::TimeDependent)
    v = td.val
    entries = isa(v, AbstractVector) ? collect(v) : Any[]
    if !isa(td.default, NoDefault)
        push!(entries, td.default)
    end
    return entries
end
function assert_internal_optimiser(td::TD_OptE_Opt)::Nothing
    for opt in time_dependent_entries(td)
        assert_internal_optimiser(opt)
    end
    return nothing
end
function assert_external_optimiser(td::TD_OptE_Opt)::Nothing
    for opt in time_dependent_entries(td)
        assert_external_optimiser(opt)
    end
    return nothing
end
function assert_special_nco_requirements(td::TD_OptE_Opt)::Nothing
    for opt in time_dependent_entries(td)
        assert_special_nco_requirements(opt)
    end
    return nothing
end
#! End: TimeDependent as an optimiser in its own right.
"""
    const VecOptE_Opt = AbstractVector{<:OptE_Opt}

Alias for a vector of optimisation estimators or results.

Represents a collection of [`OptE_Opt`](@ref) objects for batch processing.

# Related

  - [`OptE_Opt`](@ref)
"""
const VecOptE_Opt = AbstractVector{<:OptE_Opt}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`factory`](@ref) element-wise to a vector of optimisation estimators or results.

# Related

  - [`VecOptE_Opt`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::VecOptE_Opt, args...)
    return [factory(opti, args...) for opti in opt]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert special NCO requirements for each element of a vector of optimisation estimators or results.

# Related

  - [`assert_special_nco_requirements`](@ref)
  - [`NestedClustered`](@ref)
"""
function assert_special_nco_requirements(opt::VecOptE_Opt)::Nothing
    return assert_special_nco_requirements.(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any element of the vector of optimisation estimators or results requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function needs_previous_weights(opt::VecOptE_Opt)
    return any(needs_previous_weights.(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any element of the vector of optimisation estimators or results is time-dependent.

# Related

  - [`is_time_dependent`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function is_time_dependent(opt::VecOptE_Opt)
    return any(is_time_dependent.(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`update_time_dependent_estimator`](@ref) element-wise to a vector of optimisation estimators or results.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function update_time_dependent_estimator(opt::VecOptE_Opt, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    return [update_time_dependent_estimator(opti, ctx, all_binds) for opti in opt]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`assert_time_dependent_fold_count`](@ref) element-wise to a vector of optimisation estimators or results.

# Related

  - [`assert_time_dependent_fold_count`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function assert_time_dependent_fold_count(opt::VecOptE_Opt, n::Integer,
                                          all_binds::Bool = true)::Nothing
    for opti in opt
        assert_time_dependent_fold_count(opti, n, all_binds)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based weight finaliser formulations.

Defines the interface for norm types used when adjusting portfolio weights to satisfy bounds via a JuMP model.

# Related

  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
abstract type JuMPWeightFinaliserFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L1 norm of relative weight deviations when enforcing weight bounds.

# Constructors

    RelativeErrorWeightFinaliser() -> RelativeErrorWeightFinaliser

# Examples

```jldoctest
julia> RelativeErrorWeightFinaliser()
RelativeErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct RelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds.

# Constructors

    SquaredRelativeErrorWeightFinaliser() -> SquaredRelativeErrorWeightFinaliser

# Examples

```jldoctest
julia> SquaredRelativeErrorWeightFinaliser()
SquaredRelativeErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct SquaredRelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L1 norm of absolute weight deviations when enforcing weight bounds.

# Constructors

    AbsoluteErrorWeightFinaliser() -> AbsoluteErrorWeightFinaliser

# Examples

```jldoctest
julia> AbsoluteErrorWeightFinaliser()
AbsoluteErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct AbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds.

# Constructors

    SquaredAbsoluteErrorWeightFinaliser() -> SquaredAbsoluteErrorWeightFinaliser

# Examples

```jldoctest
julia> SquaredAbsoluteErrorWeightFinaliser()
SquaredAbsoluteErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct SquaredAbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for weight finaliser strategies.

A `WeightFinaliser` enforces weight bounds after the optimisation has produced unconstrained weights.

# Related

  - [`IterativeWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
abstract type WeightFinaliser <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Iteratively projects weights into the feasible region defined by weight bounds.

`IterativeWeightFinaliser` repeatedly clips and redistributes portfolio weights until they satisfy the given lower and upper bounds, or the maximum number of iterations `iter` is reached.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IterativeWeightFinaliser(;
        iter::Integer = 100
    ) -> IterativeWeightFinaliser

Keywords correspond to the struct's fields.

## Validation

  - `iter > 0`.

# Examples

```jldoctest
julia> IterativeWeightFinaliser()
IterativeWeightFinaliser
  iter ┴ Int64: 100
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
@concrete struct IterativeWeightFinaliser <: WeightFinaliser
    """
    $(field_dict[:iter])
    """
    iter
    function IterativeWeightFinaliser(iter::Integer)
        @argcheck(iter > 0, DomainError(iter, "iter must be > 0"))
        return new{typeof(iter)}(iter)
    end
end
function IterativeWeightFinaliser(; iter::Integer = 100)::IterativeWeightFinaliser
    return IterativeWeightFinaliser(iter)
end
"""
$(DocStringExtensions.TYPEDEF)

Uses a JuMP optimisation model to enforce weight bounds.

`JuMPWeightFinaliser` solves a small optimisation problem to find the closest feasible weights (in the sense of the chosen error formulation) that satisfy the given bounds. Falls back to [`IterativeWeightFinaliser`](@ref) if the JuMP model fails.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPWeightFinaliser(;
        slv::Slv_VecSlv,
        sc::Number = 1.0,
        so::Number = 1.0,
        alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser()
    ) -> JuMPWeightFinaliser

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - `sc > 0`, `so > 0`.

# Examples

```jldoctest
julia> JuMPWeightFinaliser(; slv = Solver(; solver = nothing))
JuMPWeightFinaliser
  slv ┼ Solver
      │          name ┼ String: ""
      │        solver ┼ nothing
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
   sc ┼ Float64: 1.0
   so ┼ Float64: 1.0
  alg ┴ RelativeErrorWeightFinaliser()
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliserFormulation`](@ref)
"""
@concrete struct JuMPWeightFinaliser <: WeightFinaliser
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:wfalg])
    """
    alg
    function JuMPWeightFinaliser(slv::Slv_VecSlv, sc::Number, so::Number,
                                 alg::JuMPWeightFinaliserFormulation)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc > zero(sc), DomainError(sc, "sc must be positive"))
        @argcheck(so > zero(so), DomainError(so, "so must be positive"))
        return new{typeof(slv), typeof(sc), typeof(so), typeof(alg)}(slv, sc, so, alg)
    end
end
function JuMPWeightFinaliser(; slv::Slv_VecSlv, sc::Number = 1.0, so::Number = 1.0,
                             alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser())::JuMPWeightFinaliser
    return JuMPWeightFinaliser(slv, sc, so, alg)
end
"""
    set_clustering_weight_finaliser_alg!(model, ...)

Set the clustering weight finalisation algorithm on the JuMP model.

Configures how cluster-level weights are finalised in the hierarchical optimisation model, applying the specified weight finaliser.

# Arguments

  - `model`: JuMP model.
  - Additional clustering and finaliser parameters.

# Returns

  - `nothing`.

# Related

  - [`ClusteringOptimisationEstimator`](@ref)
"""
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::RelativeErrorWeightFinaliser, wi::VecNum)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t;
                      sc * (w ⊘ wi .- one(eltype(wi)))] in
                     JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredRelativeErrorWeightFinaliser,
                                              wi::VecNum)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; sc * (w ⊘ wi .- one(eltype(wi)))] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::AbsoluteErrorWeightFinaliser, wi::VecNum)
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredAbsoluteErrorWeightFinaliser,
                                              wi::VecNum)
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
"""
    opt_weight_bounds(wf, wb, w)

Compute optimised weight bounds from the finaliser, bounds, and current weights.

Adjusts the weight bounds based on the weight finaliser algorithm and the current weight allocation, used in hierarchical weight allocation.

# Arguments

  - `wf`: Weight finaliser algorithm.
  - `wb`: Weight bounds.
  - `w`: Current portfolio weights.

# Returns

  - Updated weight bounds.

# Related

  - [`WeightBounds`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
"""
function opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum)
    lb = wb.lb
    ub = wb.ub
    if !(!isnothing(lb) && any(map((x, y) -> x > y, lb, wi)) ||
         !isnothing(ub) && any(map((x, y) -> x < y, ub, wi)))
        return wi
    end
    model = JuMP.Model()
    JuMP.@expression(model, sc, wf.sc)
    JuMP.@expression(model, so, wf.so)
    JuMP.@variable(model, w[1:length(wi)])
    JuMP.@constraint(model, sc * (sum(w) - sum(wi)) == 0)
    if !isnothing(lb)
        JuMP.@constraint(model, sc * (w ⊖ lb) >= 0)
    end
    if !isnothing(ub)
        JuMP.@constraint(model, sc * (w ⊖ ub) <= 0)
    end
    set_clustering_weight_finaliser_alg!(model, wf.alg, wi)
    return if optimise_JuMP_model!(model, wf.slv).success
        JuMP.value.(get_w(model))
    else
        @warn("Version: $(wf.alg)\nReverting to Heuristic type.")
        opt_weight_bounds(IterativeWeightFinaliser(), wb, wi)
    end
end
function opt_weight_bounds(wf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum)
    lb = wb.lb
    ub = wb.ub
    if isnothing(lb)
        lb = typemin(eltype(w))
    end
    if isnothing(ub)
        ub = typemax(eltype(w))
    end
    if !(any(map((x, y) -> x > y, lb, w)) || any(map((x, y) -> x < y, ub, w)))
        return w
    end
    iter = wf.iter
    s1 = sum(w)
    for _ in 1:iter
        if !(any(map((x, y) -> x > y, lb, w)) || any(map((x, y) -> x < y, ub, w)))
            break
        end
        old_w = copy(w)
        w = max.(min.(w, ub), lb)
        idx = w .< ub .&& w .> lb
        w_add = sum(max.(old_w ⊖ ub, zero(eltype(w))))
        w_sub = sum(min.(old_w ⊖ lb, zero(eltype(w))))
        delta = w_add + w_sub
        if !iszero(delta)
            w[idx] += delta * w[idx] / sum(w[idx])
        end
        w *= s1 / sum(w)
    end
    return w
end
"""
    finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)

Apply weight finalisation to enforce bounds and determine the optimisation return code.

Runs [`opt_weight_bounds`](@ref) with the given finaliser and bounds, then returns a success or failure return code based on whether all weights are finite.

# Arguments

  - `wf::WeightFinaliser`: Weight finaliser algorithm.
  - `wb::WeightBounds`: Weight bounds configuration.
  - `w::VecNum`: Portfolio weights to finalise.

# Returns

  - `(retcode, w)`: Tuple of return code and adjusted weights.

# Related

  - [`WeightFinaliser`](@ref)
  - [`WeightBounds`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)
    w = opt_weight_bounds(wf, wb, w)
    retcode = if !any(!isfinite, w)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "Failure to set bounds\n$wf\n$wb.")
    end
    return retcode, w
end
"""
$(DocStringExtensions.TYPEDEF)

Indicates that a portfolio optimisation completed successfully.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimisationSuccess(; res = nothing) -> OptimisationSuccess

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> OptimisationSuccess()
OptimisationSuccess
  res ┴ nothing
```

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`OptimisationFailure`](@ref)
"""
@concrete struct OptimisationSuccess <: OptimisationReturnCode
    """
    $(field_dict[:res_retcode])
    """
    res
end
function OptimisationSuccess(; res = nothing)
    return OptimisationSuccess(res)
end
"""
$(DocStringExtensions.TYPEDEF)

Indicates that a portfolio optimisation failed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimisationFailure(; res = nothing) -> OptimisationFailure

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> OptimisationFailure()
OptimisationFailure
  res ┴ nothing
```

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`OptimisationSuccess`](@ref)
"""
@concrete struct OptimisationFailure <: OptimisationReturnCode
    """
    $(field_dict[:res_retcode])
    """
    res
end
function OptimisationFailure(; res = nothing)
    return OptimisationFailure(res)
end
"""
    port_opt_view(opt, i, args...)

Return a view or subset of an optimisation estimator for a given cluster index `i`.

Default fallback returns the estimator unchanged. Overridden for composite estimators (e.g. [`JuMPOptimiser`](@ref), [`HierarchicalRiskParity`](@ref)) to slice all sub-estimators for the `i`-th cluster.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `i`: Cluster or asset index.
  - `args...`: Additional arguments (e.g. asset returns matrix).

# Returns

  - Sliced or unchanged optimisation estimator.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`NestedClustered`](@ref)
"""
function port_opt_view(opt::AbstractOptimisationEstimator, ::Any, args...)
    return opt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

A precomputed optimisation result cannot be restricted to an asset subset.

Its weights were solved over the full universe and a sub-portfolio of them has no defined meaning, so an asset-subset view of a result throws. In particular, a [`TimeDependent`](@ref) schedule holding result entries is incompatible with asset-subsampling cross-validation ([`MultipleRandomised`](@ref)), whose fold loops view the optimiser to each fold's asset subset before the swap. The trivial all-assets view (`Colon`) passes the result through unchanged.

# Related

  - [`port_opt_view`](@ref)
  - [`TimeDependent`](@ref)
  - [`MultipleRandomised`](@ref)
"""
function port_opt_view(res::NonFiniteAllocationOptimisationResult, ::Colon, args...)
    return res
end
function port_opt_view(::NonFiniteAllocationOptimisationResult, ::Any, args...)
    return throw(ArgumentError("a precomputed optimisation result cannot be viewed to an asset subset: its weights were solved over the full universe and a sub-portfolio of them has no defined meaning. A TimeDependent schedule holding precomputed results is therefore incompatible with asset-subsampling cross-validation (e.g. MultipleRandomised); use estimator entries there instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`port_opt_view`](@ref) element-wise to a vector of optimisation estimators.

# Related

  - [`port_opt_view`](@ref)
  - [`VecOptE`](@ref)
"""
function port_opt_view(opt::VecOptE, i, args...)
    return [port_opt_view(opti, i, args...) for opti in opt]
end
"""
    optimise(opt::OptimisationEstimator, args...; kwargs...) -> OptimisationResult
    optimise(opt::OptimisationResult, args...; kwargs...) -> OptimisationResult

Run portfolio optimisation using the given estimator `opt` and return an [`OptimisationResult`](@ref).

If `opt` returns an [`OptimisationFailure`](@ref), the fallback estimator is tried automatically until either a successful result is obtained or all fallbacks are exhausted.

Passing an [`OptimisationResult`](@ref) directly returns it unchanged (pass-through method).

# Arguments

  - `opt`: Optimisation estimator (e.g. a [`JuMPOptimisationEstimator`](@ref) subtype).
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

# Returns

  - [`OptimisationResult`](@ref): The optimisation result.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`OptimisationResult`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function optimise(opt::OptimisationResult, args...; kwargs...)
    return opt
end
"""
    _optimise(opt, rd, args...; dims, str_names, save, kwargs...)

Internal dispatch function for portfolio optimisation.

Called by [`optimise`](@ref) to perform the actual optimisation. Each optimisation estimator type implements its own overload. Returns the estimator-specific result type.

# Arguments

  - `opt`: Optimisation estimator (e.g. [`MeanRisk`](@ref), [`RiskBudgeting`](@ref), etc.).
  - `rd::ReturnsResult`: Returns data.
  - `dims::Int`: Observation dimension.
  - `str_names::Bool`: Whether to use string names in the JuMP model.
  - `save::Bool`: Whether to save the JuMP model in the result.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Estimator-specific optimisation result.

# Related

  - [`optimise`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function _optimise end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

High level optimisation function that wraps around estimator-specific optimisation functions. This takes care of fallback methods if the primary optimisation fails. It returns the first successful optimisation result but stores all fallback results in the `fb` field of the result.

# Arguments

  - `opt::OptimisationEstimator`: The optimisation estimator to use.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
"""
function optimise(opt::OptimisationEstimator, args...; kwargs...)
    fb = Tuple{OptimisationEstimator, OptimisationResult}[]
    current_opt = opt
    res = nothing
    while true
        res = _optimise(current_opt, args...; kwargs...)
        if isa(res.retcode, OptimisationSuccess) || isnothing(current_opt.fb)
            break
        else
            push!(fb, (current_opt, res))
            current_opt = current_opt.fb
            @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        end
    end
    return isempty(fb) ? res : factory(res, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Optimise with a [`TimeDependent`](@ref) schedule standing in for the optimiser, outside any fold loop.

There are no folds to index, so the schedule resolves to its `default` and that optimiser runs (see [`reset_time_dependent_estimator`](@ref)); a schedule with no `default` throws a [`TimeDependentDefaultError`](@ref). Inside a fold loop this method is never reached — the loop resolves entry `i` first.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`cross_val_predict`](@ref)
"""
function optimise(td::TD_OptE_Opt, args...; kwargs...)
    return optimise(reset_time_dependent_estimator(td), args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `res` is a valid internal optimisation result.

Default no-op. Overridden for result types that must satisfy internal constraints before use.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `res` is a valid external optimisation result.

Default no-op. Overridden for result types that must satisfy external interface constraints.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function assert_external_optimiser(::NonFiniteAllocationOptimisationResult)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Obtains the fees to use for net return calculations from an optimisation result. If `fees` is provided; if not, it looks for a `fees` property in the result. Returns the fees or `nothing` if not found.

# Arguments

  - `res`: Optimisation result, potentially containing a `fees` property.
  - `fees`: Optional fees to use, which take precedence over `res.fees` if provided.

# Returns

  - `Option{<:Fees}`: The fees to use for net return calculations, or `nothing` if not found.

# Related

  - [`calc_net_returns`](@ref)
  - [`OptimisationResult`](@ref)
  - [`Fees`](@ref)
"""
function extract_fees(res::OptimisationResult, fees::Option{<:Fees} = nothing)
    if isnothing(fees) && hasproperty(res, :fees)
        fees = res.fees
    end
    return fees
end
"""
    calc_net_returns(res::OptimisationResult, X::MatNum, fees = nothing)
    calc_net_returns(res::OptimisationResult, pr::Pr_RR, fees = nothing)

Compute net returns for a [`OptimisationResult`](@ref).

`fees` takes precedence over `res.fees` if both are provided. Delegates to [`calc_net_returns(w, X, fees)`](@ref).

When `pr::Pr_RR` is passed, extracts `X` from `pr.X` and delegates.

# Related

  - [`calc_net_returns`](@ref)
  - [`OptimisationResult`](@ref)
  - [`Pr_RR`](@ref)
"""
function calc_net_returns(res::OptimisationResult, X::MatNum,
                          fees::Option{<:Fees} = nothing)
    fees = extract_fees(res, fees)
    return calc_net_returns(res.w, X, fees)
end
function calc_net_returns(res::OptimisationResult, pr::Pr_RR,
                          fees::Option{<:Fees} = nothing)
    return calc_net_returns(res, pr.X, fees)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extracts the prior result for risk calculation from an optimisation result. Checks for an explicitly provided `pr`, then looks for `res.pr` and `res.pa.pr` before throwing an error if none are found.

# Arguments

  - `res`: Optimisation result, potentially containing a prior result in `res.pr` or `res.pa.pr`.
  - `pr`: Optional prior result to use for risk calculation, which takes precedence over any found in `res`.

# Returns

  - `Option{<:Pr_RR}`: The prior result to use for risk calculation, or throws an error if none is found.
"""
function extract_pr(res::OptimisationResult, pr::Option{<:Pr_RR} = nothing)
    return if !isnothing(pr)
        pr
    elseif hasproperty(res, :pr)
        res.pr
    else
        throw(ArgumentError("`$(nameof(typeof(res)))` has no `.pr` or `.jr.pr`; provide `pr` explicitly"))
    end
end
"""
    expected_risk(r::AbstractBaseRiskMeasure, res::OptimisationResult, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::AbstractBaseRiskMeasure, res::OptimisationResult, pr = nothing, fees = nothing; kwargs...)

Compute the expected risk for an [`OptimisationResult`](@ref).

Extracts `w` from `res` and delegates to the weight-based [`expected_risk`](@ref). `fees` takes precedence over `res.fees` if both are provided.

When `pr::Pr_RR` is `nothing`, tries to extract a prior result from `res.pr` or `res.pa.pr` before delegating.

# Related

  - [`expected_risk`](@ref)
  - [`OptimisationResult`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
"""
function expected_risk(r::AbstractBaseRiskMeasure, res::OptimisationResult, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    fees = extract_fees(res, fees)
    return expected_risk(r, res.w, X, fees; kwargs...)
end
function expected_risk(r::AbstractBaseRiskMeasure, res::OptimisationResult,
                       pr::Option{<:Pr_RR} = nothing, fees::Option{<:Fees} = nothing;
                       kwargs...)
    pr = extract_pr(res, pr)
    return expected_risk(r, res, pr.X, fees; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`.

`nothing` never requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(::Option{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                                 <: AbstractResult}})
    return false
end

export optimise, OptimisationSuccess, OptimisationFailure, IterativeWeightFinaliser,
       RelativeErrorWeightFinaliser, SquaredRelativeErrorWeightFinaliser,
       AbsoluteErrorWeightFinaliser, SquaredAbsoluteErrorWeightFinaliser,
       JuMPWeightFinaliser, TimeDependent, TimeDependentContext, PreviousWeightsFunction,
       TimeDependentCallable, TimeDependentOptimiserCallable, NoDefault,
       TimeDependentDefaultError
