#! Start: Overload these for all estimators which can use time-dependent constraints.
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the callable structs used as time-dependent values.

A subtype is a data-carrying alternative to a bare function inside a [`TimeDependent`](@ref): it must implement a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's field value. Because it is a struct, it participates in a trait a bare function cannot: define `needs_previous_weights(::MySubtype) = true` to declare a previous-weights requirement directly (the default is `false`), instead of wrapping in [`PreviousWeightsFunction`](@ref).

Being a struct also makes it the natural home for **recording what a callable schedule chose**: a bare `ctx -> …` selects nothing by index, so its per-fold decision is not otherwise recoverable (see the provenance note on [`TimeDependent`](@ref)). A functor can carry a mutable field — e.g. a vector it writes at `ctx.i` — and log the fold's resolved value as a side effect of computing it.

The family classifies by what the functor returns, and a subtype declares that kind in its type. Subtype [`TimeDependentConstraintCallable`](@ref) when the per-fold value is a constraint value, and [`TimeDependentOptimiserCallable`](@ref) when it is an optimiser. Only the second is statically admissible in an optimiser-valued field (see [`TD_OptE_Opt`](@ref)), so the classification is what that admissibility is read off. Do not subtype this root directly.

# Interfaces

Subtype one of the two children, not this root, and implement the following:

## The functor

  - `(x::MySubtype)(ctx::TimeDependentContext)`: Returns the field value for the fold that `ctx` describes.

### Arguments

  - `x`: The concrete subtype instance.
  - `ctx`: The fold's context, which carries the fold index, the fold loop's data and, when the loop runs sequentially, the previous fold's weights.

### Returns

  - The complete field value for that fold. Its kind is the one the subtype's supertype declares.

## `needs_previous_weights`

  - `needs_previous_weights(::MySubtype) -> Bool`: Declares whether the functor reads `ctx.w_prev`. The default is `false`. Define it as `true` to force sequential fold execution, which is what [`PreviousWeightsFunction`](@ref) does for a bare function.

# Related

  - [`TimeDependentConstraintCallable`](@ref)
  - [`TimeDependentOptimiserCallable`](@ref)
  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`PreviousWeightsFunction`](@ref)
  - [`needs_previous_weights`](@ref)
"""
abstract type TimeDependentCallable <: AbstractEstimator end
function needs_previous_weights(::TimeDependentCallable)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for callable structs whose per-fold value is a *constraint value*.

A subtype implements a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's value for a constraint-position field — a budget, a set of weight bounds, a fee structure, a turnover limit, anything a [`TimeDependent`](@ref) may carry other than an optimiser. The value is checked when the fold loop swaps it into the field, by the host's own keyword constructor.

This is the kind to subtype for a functor whose output is *not* an optimiser. A functor returning an optimiser declares [`TimeDependentOptimiserCallable`](@ref) instead, which is what makes an optimiser-position schedule statically admissible (see [`TD_OptE_Opt`](@ref)).

# Interfaces

The methods are those of [`TimeDependentCallable`](@ref): the functor `(x::MySubtype)(ctx::TimeDependentContext)`, and the optional `needs_previous_weights`. This child adds no method. It states what the functor returns — a constraint value — and the host's own keyword constructor checks that value when the fold loop swaps it in.

# Related

  - [`TimeDependentCallable`](@ref)
  - [`TimeDependentOptimiserCallable`](@ref)
  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`needs_previous_weights`](@ref)
"""
abstract type TimeDependentConstraintCallable <: TimeDependentCallable end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for callable structs whose per-fold value is an *optimiser*.

A subtype implements a functor `(x::MySubtype)(ctx::TimeDependentContext)` returning the fold's optimiser (an [`OptE_Opt`](@ref)), so a [`TimeDependent`](@ref) holding it is admissible wherever an optimiser-valued field accepts a schedule (see [`TD_OptE_Opt`](@ref)). Declaring the functor's output kind in the type is what makes the schedule *statically* admissible: a bare `ctx -> optimiser` is admitted as a `Base.Callable` and checked only when the fold loop swaps its value in.

# Interfaces

The methods are those of [`TimeDependentCallable`](@ref): the functor `(x::MySubtype)(ctx::TimeDependentContext)`, and the optional `needs_previous_weights`. This child adds no method. It states that the functor returns an [`OptE_Opt`](@ref), and [`assert_time_dependent_optimiser`](@ref) checks that promise when the fold loop swaps the value in.

# Related

  - [`TimeDependentCallable`](@ref)
  - [`TimeDependentConstraintCallable`](@ref)
  - [`TimeDependent`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`TimeDependentContext`](@ref)
"""
abstract type TimeDependentOptimiserCallable <: TimeDependentCallable end
"""
$(DocStringExtensions.TYPEDEF)

Declares that a callable time-dependent entry requires the previous optimisation's weights.

A bare callable inside a [`TimeDependent`](@ref) cannot be inspected for previous-weight requirements, so it contributes `false` to [`needs_previous_weights`](@ref) and its context's `w_prev` is only populated when something else makes the fold loop sequential. Wrapping the callable in `PreviousWeightsFunction` declares the requirement as data: it contributes `true` to [`needs_previous_weights`](@ref), forcing sequential fold execution and a populated `w_prev` in the [`TimeDependentContext`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PreviousWeightsFunction(; f) -> PreviousWeightsFunction

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PreviousWeightsFunction(; f = identity)
PreviousWeightsFunction
  f ┴ typeof(identity): identity
```

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

States that no fold-less value exists. It stands in the two places such a value may be missing.

  - As a [`TimeDependent`](@ref)'s `default`: the schedule states no fold-less value of its own, so a fold-less solve falls back to the field's static default (see [`time_dependent_field_defaults`](@ref)).
  - As an entry of a host's [`time_dependent_field_defaults`](@ref): the field is *required* and has no static default (the optimiser-valued fields), so a schedule there must carry its own `default`. A fold-less solve of a host whose required field holds a defaultless schedule throws a [`TimeDependentDefaultError`](@ref).

# Constructors

    NoDefault() -> NoDefault

# Examples

```jldoctest
julia> NoDefault()
NoDefault()
```

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

Varies one optimiser input across the folds of a cross-validation scheme.

A `TimeDependent` is stored *directly in the optimiser field it varies* — e.g. `JuMPOptimiser(; lt = TimeDependent([...]))` — so the field's position names the target and a field holds either a static value or a per-fold schedule, never both. It is recognised at top-level optimiser fields only, never nested inside another input (e.g. inside a [`Fees`](@ref) or a risk measure).

`val` is either a vector of per-fold values — entry `i` is the complete field value for fold `i` of the consuming scheme's `split` enumeration — or a callable evaluated per fold: a bare function `f(ctx::TimeDependentContext)` (optionally wrapped in [`PreviousWeightsFunction`](@ref)) or a [`TimeDependentCallable`](@ref) functor struct.

For a field that itself accepts a *vector of constraints* statically, a per-fold entry is that whole vector, so a schedule of per-fold constraint vectors is a vector of vectors — `TimeDependent([[c₁ᵃ, c₁ᵇ], [c₂ᵃ, c₂ᵇ], …])`, entry `i` being fold `i`'s complete constraint vector. There is no separate "vector of `TimeDependent`" facility and none is needed: `TimeDependent` is recognised only at a top-level field, so to vary individual constraints within a vector, build the fold's vector in a callable — `TimeDependent(ctx -> [dynamic(ctx), static])` — which keeps the shared static parts in one place.

The machinery imposes no ordering of its own: fold `i` is whatever `split(cv, rd)` enumerates `i`-th, which is chronological for walk-forward and (unshuffled) KFold schemes. For schemes whose enumeration is not a timeline (combinatorial splits, randomised paths) it is the user's responsibility to key entries off the fold's indices — a callable sees its own fold's windows via `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` and may derive any ordering from them.

A time-dependent constraint participates only where folds exist and is inert everywhere else — a fold-less `optimise` replaces it with the field's fold-less value (see [`reset_time_dependent_estimator`](@ref)). Vector entries must have length equal to the number of folds of the consuming cross-validation scheme, validated at `split` time. Entries may be `nothing`, giving the field `nothing` for that fold.

The fold-less value is the field's static default, unless `default` overrides it. A field with *no* static default — the required, optimiser-valued fields — has nothing to reset to, so a schedule there **must** supply `default`; a fold-less solve of one that does not throws a [`TimeDependentDefaultError`](@ref).

A vector whose entries are all optimisers or precomputed results ([`OptE_Opt`](@ref)) is stored as a `Vector{OptE_Opt}`, so a *mixed* schedule — fold `i` optimising or predicting depending on what entry `i` is — is admissible in an optimiser-valued field on its element type alone (see [`TD_OptE_Opt`](@ref)) rather than falling out to a `Vector{Any}` the field cannot accept.

A schedule and an [`Online`](@ref) do not wrap each other, and the reason is when each resolves: a wrapper resolves **once**, at warm-up, because the sample buffer it seeds is threaded from step to step, while a schedule resolves **per fold**, because its value is that fold's. So neither `val`, nor a vector entry of `val`, nor `default` may be an `Online` — a wrapper reached through one of them would be resolved at no fold at all, or re-seeded at every fold, throwing the buffer away. They do compose the other way round: an estimator an `Online` wraps may hold schedules of its own, which resolve per fold after the seeding, and one host may hold a wrapper in one field and a schedule in another.

Schedules do not nest: neither `val`, nor a vector entry of `val`, nor `default` may be a `TimeDependent`. Entry `i` is fold `i`'s *complete* field value, and the fold-less value is by definition outside every fold loop, so nesting has no meaning. An estimator swapped in by a schedule may itself carry schedules — those resolve against the same fold context after the swap — but they live in *its* fields, not inside this wrapper.

**Recovering which entry a fold ran** needs no stored provenance, because a vector schedule is keyed by the fold index and nothing else. Entry `i` runs at fold `i` of the consuming scheme's `split` enumeration ([`time_dependent_value`](@ref) indexes `val[ctx.i]`), so `val[i]` *is* fold `i`'s value — the same index you keyed the schedule by. Under the time-ordered schemes (walk-forward, unshuffled [`KFold`](@ref), [`Pipeline`](@ref)) fold `i` is also the `i`-th entry of the returned [`MultiPeriodPredictionResult`](@ref); under schemes that regroup for reporting ([`MultipleRandomised`](@ref) sorts by test index, combinatorial recombines each split's test groups into paths) the prediction order no longer tracks the fold order, so re-run `split(cv, rd)` and read the fold→path map off its `path_ids` — it is keyed by the very enumeration index the schedule was, so entry `k` still governs enumeration fold `k`. A **callable** schedule computes its value rather than selecting an entry, so there is no index to recover: what it returned is knowable only by re-running it on the fold's [`TimeDependentContext`](@ref), or by having it record its own choice. Recording is a logging concern the caller owns, and the [`TimeDependentCallable`](@ref) struct interface is its natural home — a functor can stash the regime it picked per fold in a field of its own.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependent(val, bind::Symbol = :outermost; default = NoDefault())
    TimeDependent(; val::Union{<:AbstractVector, <:Base.Callable,
                               <:PreviousWeightsFunction, <:TimeDependentCallable,
                               <:TimeDependent}, bind::Symbol = :outermost,
                  default = NoDefault())

## Validation

  - If `val` is a vector: `!isempty(val)`, and no entry is a `TimeDependent` or an [`Online`](@ref).
  - `val` is not a `TimeDependent` or an [`Online`](@ref).
  - `default` is not a `TimeDependent` or an [`Online`](@ref).
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
            @argcheck(!any(x -> isa(x, Online), val),
                      ArgumentError("no entry of val may be an Online: a schedule resolves once per fold, and an `Online` resolves once at warm-up, before the fold loop runs. A wrapper reached through an entry would therefore either be resolved at no fold at all, or re-seeded at every one, throwing away the buffer the step threads. Wrap the estimator that holds the schedule instead, or wrap each entry's own inner estimator."))
            if !(eltype(val) <: OptE_Opt) && all(x -> isa(x, OptE_Opt), val)
                val = convert(Vector{OptE_Opt}, val)
            end
        end
        @argcheck(!isa(default, TimeDependent),
                  ArgumentError("default cannot be a TimeDependent: it is the field's value outside every fold loop, where a schedule is undefined."))
        @argcheck(!isa(default, Online),
                  ArgumentError("default cannot be an Online: it is the field's value outside every fold loop, and an `Online` is resolved at warm-up rather than per fold, so a wrapper there is resolved at no fold at all. Wrap the estimator that holds the schedule instead."))
        @argcheck(bind in (:outermost, :nearest),
                  ArgumentError("bind must be :outermost or :nearest, got :$bind"))
        return new{typeof(val), typeof(default)}(val, bind, default)
    end
end
function TimeDependent(::TimeDependent, args...; kwargs...)
    return throw(ArgumentError("val cannot be a TimeDependent: schedules do not nest. An estimator swapped in by a schedule may carry schedules of its own — they resolve against the same fold context after the swap — but they belong in its fields, not inside this wrapper."))
end
function Online(::TimeDependent, args...; kwargs...)
    return throw(ArgumentError("est cannot be a TimeDependent: a schedule is not one estimator, so there is nothing for a buffer to belong to. The two wrappers resolve at different times and neither wraps the other — an `Online` resolves once at warm-up, because the buffer it seeds is threaded from step to step, and a schedule resolves once per fold, because its value is the fold's. They compose in the other order: an estimator an `Online` wraps may hold schedules of its own, which resolve per fold after the seeding, and one host may hold a wrapper in one field and a schedule in another."))
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
    const TDO_OptE_Opt = Union{<:TD_OptE_Opt,
                               <:TimeDependent{<:AbstractVector{<:Option{<:OptE_Opt}}}}

The [`TimeDependent`](@ref) forms admissible in an *optional* optimiser-valued field (a fallback): every [`TD_OptE_Opt`](@ref) form, plus a vector schedule whose entries may be `nothing`.

`nothing` was always a legal static value of an optional field, and the [`TimeDependent`](@ref) contract says a vector entry may be `nothing`, giving the field `nothing` for that fold — so an optional optimiser field admits `TimeDependent([mr, nothing])`, a fallback switched off on some folds. A *required* optimiser position (the optimiser itself) never admits `nothing`, statically or per fold, so it stays on the strict [`TD_OptE_Opt`](@ref) bound.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`TDO_Option`](@ref)
  - [`Option`](@ref)
"""
const TDO_OptE_Opt = Union{<:TD_OptE_Opt,
                           <:TimeDependent{<:AbstractVector{<:Option{<:OptE_Opt}}}}
"""
    const TDO_Option{X} = Union{Nothing, <:TDO_OptE_Opt, X}

Alias for an *optional* optimiser-valued field (e.g. a fallback) that accepts `nothing`, a static value of type `X`, or a per-fold schedule of optimisers whose entries may be `nothing` (see [`TDO_OptE_Opt`](@ref)).

A required optimiser-valued field spells its union out — `Union{<:X, <:TD_OptE_Opt}` — since `nothing` is not one of its values.

# Related

  - [`TDO_OptE_Opt`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`TD_Option`](@ref)
  - [`Option`](@ref)
"""
const TDO_Option{X} = Union{Nothing, <:TDO_OptE_Opt, X}
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
    const VecOptE_Opt_TD = AbstractVector{<:OptE_Opt_TD}

Alias for a vector of optimisation estimators or results in which individual *elements* may be [`TimeDependent`](@ref) schedules.

This is the element-level admission of schedules, needed where a vector-valued field's elements are themselves optimiser positions consumed by a fold loop one at a time — `Stacking.opti`, whose inner cross-validation is entered per candidate. It is a superset of [`VecOptE_Opt`](@ref), so every method taking it continues to accept plain vectors.

# Related

  - [`OptE_Opt_TD`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
const VecOptE_Opt_TD = AbstractVector{<:OptE_Opt_TD}
"""
    const TD_VecOptE_Opt = Union{TimeDependent{<:AbstractVector{<:VecOptE_Opt_TD}},
                                 TimeDependent{<:TimeDependentOptimiserCallable},
                                 TimeDependent{<:PreviousWeightsFunction},
                                 TimeDependent{<:Base.Callable}}

The [`TimeDependent`](@ref) forms admissible in a *vector-of-optimisers* field (`Stacking.opti`): a vector schedule whose entries are per-fold optimiser vectors, or a callable returning the fold's vector.

Entry `i` is fold `i`'s complete vector of candidates, so a field-level schedule varies the whole candidate set per fold; an entry's own elements may in turn be schedules (a [`VecOptE_Opt_TD`](@ref)), which the consuming host's inner fold loop resolves as usual. Only `bind = :outermost` is admissible at the field level — see the host's constructor for why.

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`VecOptE_Opt_TD`](@ref)
"""
const TD_VecOptE_Opt = Union{TimeDependent{<:AbstractVector{<:VecOptE_Opt_TD}},
                             TimeDependent{<:TimeDependentOptimiserCallable},
                             TimeDependent{<:PreviousWeightsFunction},
                             TimeDependent{<:Base.Callable}}
"""
$(DocStringExtensions.TYPEDEF)

Describes one fold to the time-dependent constraints that resolve against it.

Carries the fold's position in the consuming scheme's `split` enumeration and the data needed for a callable entry to compute its value. `i` indexes `train_idx`/`test_idx`, so `ctx.train_idx[ctx.i]`/`ctx.test_idx[ctx.i]` are always the fold's own windows; no ordering beyond the scheme's enumeration is implied. `rd` is the fold loop's (possibly asset-viewed) input data, so callables see the current universe and timestamps: the returns-level data at the optimiser fold loops, or the raw, pre-preprocessing price- or returns-level input at the [`Pipeline`](@ref) fold loop — a pipeline-level callable sees the fold's data *before* any pipeline step has transformed it. `w_prev` is populated only when the fold loop runs sequentially and a previous fold exists; `path_id` only under multi-path schemes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TimeDependentContext(;
        i::Integer, n::Integer, rd::Prices_RR, train_idx, test_idx,
        w_prev::Option{<:VecNum} = nothing, path_id::Option{<:Integer} = nothing
    ) -> TimeDependentContext

Keywords correspond to the struct's fields.

## Validation

  - `1 <= i <= n`.

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
    function TimeDependentContext(i::Integer, n::Integer, rd::Prices_RR, train_idx,
                                  test_idx, w_prev::Option{<:VecNum},
                                  path_id::Option{<:Integer})
        @argcheck(1 <= i <= n, DomainError(i, "fold index i must be in 1:$n"))
        return new{typeof(i), typeof(n), typeof(rd), typeof(train_idx), typeof(test_idx),
                   typeof(w_prev), typeof(path_id)}(i, n, rd, train_idx, test_idx, w_prev,
                                                    path_id)
    end
end
function TimeDependentContext(; i::Integer, n::Integer, rd::Prices_RR, train_idx, test_idx,
                              w_prev::Option{<:VecNum} = nothing,
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
                                                              <:Tr_VecTr,
                                                              <:AbstractBaseRiskMeasure})::Bool
    return needs_previous_weights(x)
end
function time_dependent_entry_needs_previous_weights(x::AbstractVector)::Bool
    return any(time_dependent_entry_needs_previous_weights, x)
end
function time_dependent_entry_needs_previous_weights(x::OptE_Opt)::Bool
    return needs_previous_weights(x)
end
function time_dependent_entry_needs_previous_weights(x::TimeDependent)::Bool
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
$(DocStringExtensions.TYPEDSIGNATURES)

Slice a [`TimeDependent`](@ref) schedule of scalar-or-array field values (warm starts, initial weights) to asset indices `i`.

Vector schedules slice each per-fold entry, and the `default` when one is set; callable schedules pass through — they see the sliced universe via their fold context's `rd`.

# Related

  - [`TimeDependent`](@ref)
  - [`port_opt_view`](@ref)
"""
function nothing_scalar_array_view(td::TimeDependent, i)
    v = td.val
    if isa(v, AbstractVector)
        v = [nothing_scalar_array_view(x, i) for x in v]
    end
    d = td.default
    if !isa(d, NoDefault)
        d = nothing_scalar_array_view(d, i)
    end
    return TimeDependent(v, td.bind; default = d)
end
"""
    inner_fold_fields(opt)

Field names of `opt` that the host hands across a fold loop it opens *itself*, so the loop that merely reaches the host is never the nearest one for them.

The default is the empty tuple: an ordinary host opens no inner fold loop, so the loop reaching it is both outermost and nearest for every field. A meta-optimiser whose inner cross-validation consumes a field directly declares that field here (e.g. `NestedClustered`'s `opti`, entered per cluster as `cross_val_predict(opti, …; cols = cl)`), and every generic pass — [`time_dependent_fields`](@ref), and through it update, reset and the fold-count assertion — then leaves a `bind = :nearest` schedule in that field for the host's own inner loop. Without the reset leg of this rule, the fold-less reset at the top of `_optimise` would replace a `:nearest` optimiser schedule with its `default` before the inner cross-validation ever saw it.

# Related

  - [`entitled`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`reset_time_dependent_fields`](@ref)
"""
function inner_fold_fields(::Any)::Tuple
    return ()
end
"""
    time_dependent_candidate_fields(opt)

Field names of `opt` whose *type* admits a [`TimeDependent`](@ref) value — the candidate set [`time_dependent_fields`](@ref) narrows by value.

Whether a field can hold a schedule is decidable from `fieldtype` alone: a host built through the widened constructor signatures (see [`TD_Option`](@ref)) records a schedule in the field's type parameter, so a field that holds no schedule cannot have a type intersecting [`TimeDependent`](@ref). The tuple is therefore computed once per host type by a generated function, and a fold-invariant scan over a wide static host such as `JuMPOptimiser`, whose fields number in the dozens, folds to an empty tuple at compile time rather than walking every field dynamically on every `split` and `_optimise`.

This stays derived from the field types — no hand-maintained list — so the constructor signatures remain the single source of truth for which fields may vary over folds.

# Related

  - [`TimeDependent`](@ref)
  - [`TD_Option`](@ref)
  - [`time_dependent_fields`](@ref)
"""
@generated function time_dependent_candidate_fields(::T) where {T}
    fns = Tuple(f
                for f in fieldnames(T)
                if typeintersect(fieldtype(T, f), TimeDependent) !== Union{})
    return :($fns)
end
"""
    entitled(opt, f::Symbol, all_binds::Bool)

Return `true` when the recursion position described by `all_binds` may consume a `bind = :nearest` schedule in field `f` of `opt`.

Entitlement is per-field, not per-host: a loop scanning with `all_binds = true` takes `:nearest` schedules everywhere *except* in the fields the host hands across its own inner fold loop (see [`inner_fold_fields`](@ref)) — for those, the host's inner loop is the nearest one, whatever loop is doing the scanning. A field is taken by a pass iff `entitled(opt, f, all_binds) || bind === :outermost`.

# Related

  - [`inner_fold_fields`](@ref)
  - [`time_dependent_fields`](@ref)
"""
function entitled(opt, f::Symbol, all_binds::Bool)::Bool
    return all_binds && !(f in inner_fold_fields(opt))
end
"""
    time_dependent_fields(opt, all_binds::Bool = true)

Return the tuple of field names of `opt` whose values are [`TimeDependent`](@ref).

The scan is generic over the host's fields, so the widened constructor signatures (see [`TD_Option`](@ref)) remain the single source of truth for which fields may vary over folds — there is no hand-maintained list. Only the fields whose type admits a schedule are visited (see [`time_dependent_candidate_fields`](@ref)); the rest are ruled out at compile time, so a static host returns an empty tuple without touching its fields.

# The `all_binds` argument

`all_binds` encodes something the schedule's own `bind` field cannot: it is a property of the *recursion position*, not of the schedule. A [`TimeDependent`](@ref)'s `bind` (`:outermost` / `:nearest`) says *which* fold loop the schedule wants; `all_binds` says whether the loop currently recursing is *entitled* to consume nearest-bound schedules at this depth. The second fact is not on the schedule.

Why position matters: under `outer CV loop → meta → (meta's inner CV loop) → inner estimator with a :nearest field`, the same `:nearest` field is visited by two loops. The outer loop recurses through the meta (mandatory — that recursion is how an inner estimator's `:outermost` field is resolved against the outer folds) and must *skip* the `:nearest` field, because it is not the nearest enclosing loop. The meta's inner CV loop drives the same estimator directly and must *consume* it, because it is. Same field, same `bind`, opposite actions — the difference is whether a nearer fold-loop boundary was crossed to reach it, which is exactly what `all_binds` carries.

So `all_binds` is `true` at every ordinary (outermost/standalone) fold loop — which is both outermost and nearest, and therefore takes everything remaining, including `:nearest`. It is forced to `false` only where a meta-optimiser recurses into the estimators its own inner CV owns, leaving their `:nearest` schedules for that inner loop. With `all_binds = false`, only fields with `bind === :outermost` are returned.

Entitlement is refined **per field** by [`inner_fold_fields`](@ref): even at `all_binds = true`, a `:nearest` schedule in a field the host hands across its *own* inner fold loop is left alone — the host's inner loop, not the scanning one, is nearest for that field (see [`entitled`](@ref)).

# Related

  - [`TimeDependent`](@ref)
  - [`TD_Option`](@ref)
  - [`is_time_dependent`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`entitled`](@ref)
"""
function time_dependent_fields(opt, all_binds::Bool = true)
    fns = time_dependent_candidate_fields(opt)
    return filter(f -> begin
                      x = getfield(opt, f)
                      if isa(x, TimeDependent)
                          (entitled(opt, f, all_binds) || x.bind === :outermost)
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
        substitute_time_dependent_entries(T, base, f, args[f])
    end
    return nothing
end
"""
    substitute_time_dependent_entries(::Type{T}, base::NamedTuple, f::Symbol, td::TimeDependent) where {T}
    substitute_time_dependent_entries(::Type, ::NamedTuple, ::Symbol, ::Any)

Re-run the keyword constructor of `T` with every vector entry of the schedule `td`, and its explicit `default`, substituted into the field `f` of `base`; a field that holds no schedule substitutes nothing.

The filter of [`assert_time_dependent_substitution`](@ref) keeps the schedules alone, and the dispatch says so to a static analyser at a call whose arguments hold no schedule at all, where an assertion would read as a certain failure.

# Related

  - [`assert_time_dependent_substitution`](@ref)
  - [`TimeDependent`](@ref)
"""
function substitute_time_dependent_entries(::Type{T}, base::NamedTuple, f::Symbol,
                                           td::TimeDependent)::Nothing where {T}
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
    return nothing
end
function substitute_time_dependent_entries(::Type, ::NamedTuple, ::Symbol, ::Any)::Nothing
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

# Related

  - [`update_time_dependent_fields`](@ref)
  - [`reset_time_dependent_fields`](@ref)
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
function assert_time_dependent_fold_count(td::TDO_OptE_Opt, n::Integer,
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
    entries = isa(v, AbstractVector) ? collect(Any, v) : Any[]
    if !isa(td.default, NoDefault)
        push!(entries, td.default)
    end
    return entries
end
function assert_internal_optimiser(td::Union{<:TD_OptE_Opt, <:TD_VecOptE_Opt})::Nothing
    for opt in time_dependent_entries(td)
        assert_internal_optimiser(opt)
    end
    return nothing
end
function assert_external_optimiser(td::Union{<:TD_OptE_Opt, <:TD_VecOptE_Opt})::Nothing
    for opt in time_dependent_entries(td)
        assert_external_optimiser(opt)
    end
    return nothing
end
function assert_special_nco_requirements(td::Union{<:TD_OptE_Opt, <:TD_VecOptE_Opt})::Nothing
    for opt in time_dependent_entries(td)
        assert_special_nco_requirements(opt)
    end
    return nothing
end
"""
    assert_no_nearest_bind_optimiser_schedule(x, field::Symbol, host::Symbol)

Reject a `bind = :nearest` [`TimeDependent`](@ref) schedule in an optimiser-valued position no inner fold loop consumes.

`bind` picks *which fold loop supplies the schedule's index*. `:nearest` therefore says something different from `:outermost` only where the host opens a fold loop of its **own** and hands the field across it — `NestedClustered.opti` (its inner cross-validation is entered per cluster) and `Stacking.opti[k]` (entered per candidate), the positions declared by [`inner_fold_fields`](@ref). Everywhere else the loop that reaches the host *is* the nearest one, and the two binds would name the same loop.

The positions this guards have no such inner loop:

  - **A fallback (`fb`), on every host.** The fallback walk is a retry chain *within a single fold's solve* — it has no fold indices of its own — so `:nearest` there is either redundant with `:outermost` or, behind a meta's inner cross-validation, silently wrong: it would resolve against the inner loop's fold numbers (tuning folds) instead of the backtest's periods, changing meaning with nesting depth. A per-fold fallback is fully expressible with `:outermost`, including `nothing` entries to switch it off on some folds (see [`TDO_OptE_Opt`](@ref)).
  - **The outer optimisers (`opto`).** They consume the *combined* inner output, once per solve.
  - **`SubsetResampling.opt`.** Its internal loop is over randomly drawn asset subsets, not time folds.

So a `:nearest` schedule in any of them has no nearest fold loop to bind to, and is rejected at construction rather than resolving against a loop the caller did not mean. No-op for anything that is not a [`TimeDependent`](@ref).

# Related

  - [`assert_nearest_optimiser_schedule`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`TDO_OptE_Opt`](@ref)
"""
function assert_no_nearest_bind_optimiser_schedule(x, field::Symbol, host::Symbol)::Nothing
    if isa(x, TimeDependent)
        @argcheck(x.bind !== :nearest,
                  ArgumentError("field `$field` of $host holds a `bind = :nearest` TimeDependent schedule, but no inner fold loop of $host consumes `$field`, so there is no nearest fold loop for it to bind to. Use `bind = :outermost`: the fold loop that reaches the $host resolves the schedule."))
    end
    return nothing
end
"""
    assert_nearest_optimiser_schedule(x, field::Symbol, cv, host::Symbol)

Validate a `bind = :nearest` [`TimeDependent`](@ref) schedule in an optimiser-valued position that a host's inner cross-validation *does* consume.

Two construction-time requirements, both consequences of the position's double consumer: the inner cross-validation leg resolves the schedule per fold, while the full-sample leg (the meta's `wi` fit, or the per-cluster optimise) always resolves it fold-lessly to its `default`.

  - An explicit `default` is required — without one, every solve would throw a [`TimeDependentDefaultError`](@ref) when the full-sample leg reaches the schedule, so the error is moved to construction. This deliberately departs from the rule that a defaultless schedule is legal at construction, for this position only.
  - `cv !== nothing` is required — without an inner cross-validation there is no inner fold loop, so the schedule could only ever be its `default`: silently inert.

No-op for anything that is not a `bind = :nearest` [`TimeDependent`](@ref).

# Related

  - [`assert_no_nearest_bind_optimiser_schedule`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
function assert_nearest_optimiser_schedule(x, field::Symbol, cv, host::Symbol)::Nothing
    if isa(x, TimeDependent) && x.bind === :nearest
        @argcheck(!isa(x.default, NoDefault),
                  TimeDependentDefaultError("a `bind = :nearest` schedule in `$field` of $host must supply a `default`: besides the inner cross-validation fold loop, `$field` also has a fold-less full-sample consumer that always resolves the schedule to its `default`, so a defaultless one would throw on every solve. Give it a fold-less optimiser: TimeDependent(val, :nearest; default = opt)."))
        @argcheck(!isnothing(cv),
                  ArgumentError("a `bind = :nearest` schedule in `$field` of $host requires `cv`: without an inner cross-validation there is no inner fold loop, so the schedule could only ever resolve to its `default` — silently inert. Provide `cv`, or use `bind = :outermost` so the fold loop that reaches the $host consumes it."))
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

Apply [`factory`](@ref) through a [`TimeDependent`](@ref) schedule: to each vector entry and to the `default`, rebuilding the schedule.

A schedule can survive a fold loop's resolution pass (a `bind = :nearest` element left for a meta's inner cross-validation), so the factory pass that follows resolution must see through it. Callable forms pass through unchanged — their per-fold values do not exist yet, and a callable receives the fold's context (including `w_prev`) when it runs.

# Related

  - [`factory`](@ref)
  - [`TimeDependent`](@ref)
"""
function factory(td::TimeDependent, args...)
    v = td.val
    if isa(v, AbstractVector)
        v = [factory(x, args...) for x in v]
    end
    d = td.default
    if !isa(d, NoDefault)
        d = factory(d, args...)
    end
    return TimeDependent(v, td.bind; default = d)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert special NCO requirements for each element of a vector of optimisation estimators or results.

# Related

  - [`assert_special_nco_requirements`](@ref)
  - [`NestedClustered`](@ref)
"""
function assert_special_nco_requirements(opt::VecOptE_Opt_TD)::Nothing
    for opti in opt
        assert_special_nco_requirements(opti)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any element of the vector of optimisation estimators or results requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function needs_previous_weights(opt::VecOptE_Opt_TD)
    return any(needs_previous_weights, opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any element of the vector of optimisation estimators or results is time-dependent.

# Related

  - [`is_time_dependent`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function is_time_dependent(opt::VecOptE_Opt_TD)
    return any(is_time_dependent, opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`update_time_dependent_estimator`](@ref) element-wise to a vector of optimisation estimators or results.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function update_time_dependent_estimator(opt::VecOptE_Opt_TD, ctx::TimeDependentContext,
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
function assert_time_dependent_fold_count(opt::VecOptE_Opt_TD, n::Integer,
                                          all_binds::Bool = true)::Nothing
    for opti in opt
        assert_time_dependent_fold_count(opti, n, all_binds)
    end
    return nothing
end

export TimeDependent, TimeDependentContext, PreviousWeightsFunction, NoDefault,
       TimeDependentDefaultError
public time_dependent_field_defaults, TimeDependentCallable,
       TimeDependentConstraintCallable, TimeDependentOptimiserCallable
