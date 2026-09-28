#! Start: Overload these for all estimators which can use time-dependent constraints.
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the callable structs that compute the value of a schedule at each fold.

A subtype takes the place of a bare function in a [`TimeDependent`](@ref), and holds its parameters as data. Because it is a type, it can declare that it reads the previous weights with a method of `needs_previous_weights`. A bare function needs the wrapper [`PreviousWeightsFunction`](@ref) for that.

A subtype can also record what it computed. A bare function `ctx -> …` selects no entry by index, so a later reader cannot recover its value at a fold. A functor can hold a mutable field, for example a vector, and write the value of fold `ctx.i` into it.

The family is divided by what the functor returns, and a subtype states that kind by its supertype. Subtype [`TimeDependentConstraintCallable`](@ref) when the value is a constraint value, and [`TimeDependentOptimiserCallable`](@ref) when the value is an optimiser. Only the second is admitted in a field that holds an optimiser before any fold runs, see [`TD_OptE_Opt`](@ref). Do not subtype this root directly.

# Interfaces

Subtype one of the two children, not this root, and implement the methods below.

## The functor

  - `(x::MySubtype)(ctx::TimeDependentContext)`: Returns the value of the field at the fold that `ctx` describes.

### Arguments

  - `x`: The concrete subtype instance.
  - `ctx`: The context of the fold. It holds the index of the fold, the data of the fold loop, and the weights of the previous fold when the loop runs its folds in sequence.

### Returns

  - The complete value of the field at that fold, of the kind that the supertype of the subtype states.

## `needs_previous_weights`

  - `needs_previous_weights(::MySubtype) -> Bool`: States whether the functor reads `ctx.w_prev`. The fallback returns `false`. Return `true` to make the fold loop run its folds in sequence, as [`PreviousWeightsFunction`](@ref) does for a bare function.

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

Abstract supertype for the callable structs whose value at a fold is a constraint value.

A constraint value is any value of a [`TimeDependent`](@ref) that is not an optimiser, for example a budget, a set of weight bounds, a fee structure or a turnover limit. When the fold loop puts the value into its field, the keyword constructor of the optimiser that holds the field checks it.

A functor that returns an optimiser subtypes [`TimeDependentOptimiserCallable`](@ref) instead.

# Interfaces

The methods are those of [`TimeDependentCallable`](@ref): the functor `(x::MySubtype)(ctx::TimeDependentContext)`, and the optional `needs_previous_weights`. This child adds no method. It states that the functor returns a constraint value.

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

Abstract supertype for the callable structs whose value at a fold is an optimiser.

The functor returns an [`OptE_Opt`](@ref), so a [`TimeDependent`](@ref) that holds it is admitted in every field that holds an optimiser, see [`TD_OptE_Opt`](@ref). The type states the kind of the value before any fold runs. A bare function `ctx -> optimiser` is admitted as a `Base.Callable`, and its value is checked only when the fold loop puts it into the field.

# Interfaces

The methods are those of [`TimeDependentCallable`](@ref): the functor `(x::MySubtype)(ctx::TimeDependentContext)`, and the optional `needs_previous_weights`. This child adds no method. It states that the functor returns an [`OptE_Opt`](@ref), and [`assert_time_dependent_optimiser`](@ref) checks the value when the fold loop puts it into the field.

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

States that a function in a schedule reads the weights of the previous fold.

No code can find out whether a bare function in a [`TimeDependent`](@ref) reads the previous weights. So a bare function gives `false` to [`needs_previous_weights`](@ref), and its context holds `w_prev` only when another input makes the fold loop run in sequence. A function in a `PreviousWeightsFunction` gives `true`. Then the fold loop runs its folds in sequence, and the [`TimeDependentContext`](@ref) of each fold after the first holds `w_prev`.

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
@concrete struct PreviousWeightsFunction <: AbstractAlgorithm
    """
    Function that the fold loop calls at each fold as `f(ctx::TimeDependentContext)`. It returns the value of the field at that fold.
    """
    f
end
function PreviousWeightsFunction(; f)::PreviousWeightsFunction
    return PreviousWeightsFunction(f)
end
function needs_previous_weights(::PreviousWeightsFunction)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDEF)

States that no value exists for a field outside every fold loop.

It has two uses.

  - As the `default` of a [`TimeDependent`](@ref), it states that the schedule gives no value outside every fold loop. A solve with no fold then uses the static default of the field, see [`time_dependent_field_defaults`](@ref).
  - As an entry of the [`time_dependent_field_defaults`](@ref) of an optimiser, it states that the field is required and has no static default. The fields that hold an optimiser are such fields. A schedule in one of them must carry its own `default`, else a solve with no fold throws a [`TimeDependentDefaultError`](@ref).

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

Exception for a solve with no fold that reaches a schedule with no value outside every fold loop.

A [`TimeDependent`](@ref) is defined only over the folds of a cross-validation scheme. A field with a static default takes that default outside every fold loop, and no message is given. A required field, such as a field that holds an optimiser, has no static default. So its schedule must state the value outside every fold loop, as `TimeDependent(val; default = x)`.

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

Changes one optimiser input from fold to fold of a cross-validation scheme.

A `TimeDependent` goes directly into the optimiser field that it changes, for example `JuMPOptimiser(; lt = TimeDependent([...]))`. So the field names the target, and a field holds a static value or a schedule, never both. The fold loop finds a schedule only in a field of an optimiser. It does not find one inside another input, for example inside a [`Fees`](@ref) or a risk measure.

The field `val` is a vector or a function. Entry `i` of a vector is the complete value of the field at fold `i`. A function is called at each fold with the [`TimeDependentContext`](@ref) of the fold. It is a bare function, a bare function in a [`PreviousWeightsFunction`](@ref), or a [`TimeDependentCallable`](@ref).

A field that takes a vector of constraints takes the whole vector at each fold. So a schedule of constraint vectors is a vector of vectors, `TimeDependent([[c₁ᵃ, c₁ᵇ], [c₂ᵃ, c₂ᵇ], …])`. To change one constraint of a vector and keep the others, make the vector in a function, `TimeDependent(ctx -> [dynamic(ctx), static])`.

Fold `i` is fold `i` of the enumeration of `split(cv, rd)`, and the schedule sets no other order. That order is the order of time for a walk-forward scheme and for a `KFold` with no shuffle. For a scheme whose enumeration is not an order of time, such as a combinatorial or a randomised scheme, a function can read the windows of its fold, `ctx.train_idx[ctx.i]` and `ctx.test_idx[ctx.i]`.

A schedule has an effect only at a fold. An `optimise` with no fold replaces it with the value of the field outside every fold loop, see [`reset_time_dependent_estimator`](@ref). That value is the static default of the field, or `default` when the schedule sets one. A required field, such as a field that holds an optimiser, has no static default. So a schedule in such a field must set `default`, else a solve with no fold throws a [`TimeDependentDefaultError`](@ref).

A vector must have one entry for each fold of the scheme, and the fold loop checks the length after `split`. An entry can be `nothing`, which gives the field the value `nothing` at that fold.

The constructor stores a vector whose entries are all optimisers or precomputed results, [`OptE_Opt`](@ref), as a `Vector{OptE_Opt}`. So a mixed schedule, which optimises at some folds and predicts at others, is admitted in a field that holds an optimiser by its element type, see [`TD_OptE_Opt`](@ref). Else the vector is a `Vector{Any}`, which the field does not take.

A schedule and an [`Online`](@ref) do not wrap each other, because they resolve at different times. An `Online` resolves once, at the warm-up, because the step passes its sample buffer on to the next step. A schedule resolves at each fold. So `val`, an entry of `val` and `default` must not be an `Online`. The other order works: an estimator in an `Online` can hold schedules, which resolve at each fold after the warm-up.

Schedules do not nest. So `val`, an entry of `val` and `default` must not be a `TimeDependent`. An estimator that a schedule puts into a field can hold schedules in its own fields. Those resolve against the same fold after the swap.

To find the entry that ran at a fold, read the index of the fold. [`time_dependent_value`](@ref) reads `val[ctx.i]`, so `val[i]` is the value at fold `i`. For a walk-forward scheme, a `KFold` with no shuffle and a [`Pipeline`](@ref), fold `i` is also entry `i` of the [`MultiPeriodPredictionResult`](@ref). [`MultipleRandomised`](@ref) and the combinatorial schemes put their predictions in a different order. For them, run `split(cv, rd)` again, and read the map from fold to path in its `path_ids`.

A function computes its value and selects no entry, so no index records its value. To get the value again, call the function with the context of the fold, or let a [`TimeDependentCallable`](@ref) write its value into one of its fields.

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
  - `val` is not a `TimeDependent`, else an `ArgumentError` is thrown. The constructor has no method for a `val` that is not a vector, a function, a [`PreviousWeightsFunction`](@ref) or a [`TimeDependentCallable`](@ref), such as an `Online`.
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
@concrete struct TimeDependent <: AbstractEstimator
    """
    Vector with one value for each fold, in the order of `split`, or a function of the [`TimeDependentContext`](@ref) of the fold. The function is a bare function, a bare function in a [`PreviousWeightsFunction`](@ref), or a [`TimeDependentCallable`](@ref).
    """
    val
    """
    Fold loop that resolves the schedule. `:outermost`, the default, selects the outermost fold loop over the estimator. `:nearest` selects the nearest fold loop around the field. In an inner estimator of a meta-optimiser, that is the cross-validation of the meta-optimiser, also when an outer fold loop runs the meta-optimiser.
    """
    bind::Symbol
    """
    Value of the field outside every fold loop. It replaces the static default that [`time_dependent_field_defaults`](@ref) gives for the field. [`NoDefault`](@ref), the default, keeps the static default. A field with no static default needs this value.
    """
    default
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
    return throw(ArgumentError("est cannot be a TimeDependent: a schedule is not one estimator, so there is nothing for a buffer to belong to. The two wrappers resolve at different times and neither wraps the other — an `Online` resolves once at warm-up, because the buffer it seeds is threaded from step to step, and a schedule resolves once per fold, because its value is the fold's. They compose in the other order: an estimator an `Online` wraps may hold schedules of its own, which resolve per fold after the seeding, and one estimator may hold a wrapper in one field and a schedule in another."))
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

Alias for an optimiser field that takes `nothing`, a static value of type `X`, or a [`TimeDependent`](@ref) schedule.

The fields whose constructor signatures use this alias, or [`TD`](@ref), are the optimiser inputs that can change from fold to fold. No other list of them exists.

# Related

  - [`TimeDependent`](@ref)
  - [`Option`](@ref)
"""
const TD_Option{X} = Union{Nothing, <:TimeDependent, X}
"""
    const TD{X} = Union{<:TimeDependent, X}

Alias for a required optimiser field that takes a static value of type `X` or a [`TimeDependent`](@ref) schedule, but not `nothing`.

The prior estimator, the returns model, the scalariser, the clustering estimator and the weight finaliser always hold a value. They use this alias and not [`TD_Option`](@ref), so they do not take `nothing`. Such a field has a static default, so a solve with no fold resets a schedule in it to that default. A field that holds an optimiser has no static default, see [`TD_OptE_Opt`](@ref).

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

Alias for the [`TimeDependent`](@ref) forms that a field which holds an optimiser takes, where the schedule changes the optimiser itself.

The type of two forms states their value before any fold runs. The first is a vector whose entries are all [`OptE_Opt`](@ref). An entry is an optimiser or a precomputed result, so a mixed schedule optimises at some folds and predicts at others. The second is a [`TimeDependentOptimiserCallable`](@ref).

The other two forms are a bare function `ctx -> optimiser` and a [`PreviousWeightsFunction`](@ref) that holds one. Their value is not known before they run, so it is checked when the fold loop puts it into place. In a field of an estimator, the keyword constructor of the estimator checks it. When the schedule is the optimiser itself, [`assert_time_dependent_optimiser`](@ref) checks it.

A field that holds an optimiser is required, and it has no static default. So a schedule in it must set `default`, see [`NoDefault`](@ref) and [`TimeDependentDefaultError`](@ref).

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

Alias for the [`TimeDependent`](@ref) forms that an optional field which holds an optimiser takes, such as a fallback.

It holds every form of [`TD_OptE_Opt`](@ref), and a vector whose entries can be `nothing`. An optional field takes `nothing` as a static value, and an entry `nothing` gives the field `nothing` at that fold. So an optional field takes `TimeDependent([mr, nothing])`, a fallback that is off at some folds. A required field that holds an optimiser never takes `nothing`, so it keeps the bound [`TD_OptE_Opt`](@ref).

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`TDO_Option`](@ref)
  - [`Option`](@ref)
"""
const TDO_OptE_Opt = Union{<:TD_OptE_Opt,
                           <:TimeDependent{<:AbstractVector{<:Option{<:OptE_Opt}}}}
"""
    const TDO_Option{X} = Union{Nothing, <:TDO_OptE_Opt, X}

Alias for an optional field that holds an optimiser, such as a fallback.

The field takes `nothing`, a static value of type `X`, or a schedule of optimisers whose entries can be `nothing`, see [`TDO_OptE_Opt`](@ref). A required field that holds an optimiser does not take `nothing`, so its signature writes `Union{<:X, <:TD_OptE_Opt}`.

# Related

  - [`TDO_OptE_Opt`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`TD_Option`](@ref)
  - [`Option`](@ref)
"""
const TDO_Option{X} = Union{Nothing, <:TDO_OptE_Opt, X}
"""
    const OptE_TD = Union{<:NonFiniteAllocationOptimisationEstimator, <:TD_OptE_Opt}

Alias for an optimisation estimator, or a [`TimeDependent`](@ref) schedule in the place of one.

The cross-validation fold loops that fit take this type. A schedule given directly to [`cross_val_predict`](@ref) is the optimiser, and fold `i` runs entry `i`. A bare precomputed result is not a member, because it takes the path that only predicts, which has no fold loop to resolve a schedule against. A schedule whose entries are results is a member, and each such entry predicts at its fold, see [`OptE_Opt_TD`](@ref).

# Related

  - [`TD_OptE_Opt`](@ref)
  - [`OptE_Opt_TD`](@ref)
  - [`cross_val_predict`](@ref)
"""
const OptE_TD = Union{<:NonFiniteAllocationOptimisationEstimator, <:TD_OptE_Opt}
"""
    const OptE_Opt_TD = Union{<:OptE_Opt, <:TD_OptE_Opt}

Alias for an optimisation estimator, a precomputed result, or a [`TimeDependent`](@ref) schedule in the place of one.

The fold loops that take a precomputed result as well as an estimator take this type. The entries of a schedule are [`OptE_Opt`](@ref), so a mixed schedule is a member. Fold `i` optimises when entry `i` is an estimator, and predicts when entry `i` is a result. The methods of [`fit_and_predict`](@ref) select the path by dispatch.

# Related

  - [`OptE_TD`](@ref)
  - [`TD_OptE_Opt`](@ref)
  - [`OptE_Opt`](@ref)
"""
const OptE_Opt_TD = Union{<:OptE_Opt, <:TD_OptE_Opt}
"""
    const VecOptE_Opt_TD = AbstractVector{<:OptE_Opt_TD}

Alias for a vector of optimisation estimators or results in which an entry can be a [`TimeDependent`](@ref) schedule.

A field needs it when a fold loop reads each entry of the vector as an optimiser, one entry at a time. `Stacking.opti` is such a field, because its inner cross-validation runs once for each candidate. The alias contains [`VecOptE_Opt`](@ref), so each method that takes it also takes a vector with no schedule.

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

Alias for the [`TimeDependent`](@ref) forms that a field which holds a vector of optimisers takes, such as `Stacking.opti`.

A vector schedule holds one vector of optimisers for each fold, and a function returns the vector of its fold. Entry `i` is the complete vector of candidates at fold `i`, so the schedule changes the whole set of candidates from fold to fold. An element of an entry can also be a schedule, a [`VecOptE_Opt_TD`](@ref), which the inner fold loop of the optimiser resolves. A schedule of the whole field takes only `bind = :outermost`. The constructor of the optimiser states the reason.

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

Describes one fold to the schedules that resolve against it.

It holds the position of the fold in the enumeration of `split`, and the data that a function in a schedule reads to compute its value. The field `i` indexes `train_idx` and `test_idx`, so `ctx.train_idx[ctx.i]` and `ctx.test_idx[ctx.i]` are the windows of the fold. The context sets no order other than the enumeration of the scheme.

The field `rd` is the input data of the fold loop, which can be a view of a subset of the assets. So a function reads the current universe and timestamps. An optimiser fold loop gives returns. The [`Pipeline`](@ref) fold loop gives the prices or returns that it received, before any step of the pipeline changes them.

The field `w_prev` holds weights only when the fold loop runs in sequence and a previous fold exists. The field `path_id` holds a value only under a scheme with many paths.

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
@concrete struct TimeDependentContext <: AbstractResult
    """
    Index of the fold in the enumeration of `split`, from one. It indexes `train_idx` and `test_idx`.
    """
    i
    """
    Number of folds in the path.
    """
    n
    """
    Input data of the fold loop, which can be a view of a subset of the assets.
    """
    rd
    """
    Training index vectors of the path, one for each fold.
    """
    train_idx
    """
    Test index vectors of the path, one for each fold.
    """
    test_idx
    """
    Portfolio weights of the previous fold when the fold loop runs in sequence, else `nothing`.
    """
    w_prev
    """
    Identifier of the path under a scheme with many paths, else `nothing`.
    """
    path_id
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

Resolves a schedule to its value at the fold that `ctx` describes.

# Algorithm

 1. When `td.val` is a vector, return its entry `ctx.i`.
 2. When `td.val` is a [`PreviousWeightsFunction`](@ref), return `td.val.f(ctx)`.
 3. Else return `td.val(ctx)`.

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

Returns `true` if a schedule reads the weights of the previous fold.

A bare function gives `false`, because no code can find out what it reads.

# Algorithm

 1. When `td.val` is a [`PreviousWeightsFunction`](@ref), return `true`.
 2. When `td.val` is a [`TimeDependentCallable`](@ref), return its own `needs_previous_weights`.
 3. When `td.val` is a vector, return `true` if any entry needs the previous weights, by [`time_dependent_entry_needs_previous_weights`](@ref).
 4. Else return `false`.

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

Returns `true` if an entry of a [`TimeDependent`](@ref) vector reads the weights of the previous fold.

A turnover, a fee, a tracking input, a risk measure, an optimisation estimator or result, and a nested schedule each give their own [`needs_previous_weights`](@ref). A vector entry gives `true` if any of its elements does. Every other value gives `false`.

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

Slices a [`TimeDependent`](@ref) schedule of scalar or array values, such as warm starts or initial weights, to the asset indices `i`.

A function schedule is not sliced, because it reads the sliced universe from the field `rd` of its context.

# Algorithm

 1. When `td.val` is a vector, slice each entry with `nothing_scalar_array_view`.
 2. When `td.default` is not [`NoDefault`](@ref), slice it too.
 3. Return a new schedule with the sliced values and the same `bind`.

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

Returns the names of the fields of `opt` that the optimiser gives to a fold loop of its own.

For such a field, the fold loop that reaches the optimiser is not the nearest fold loop. The fallback returns an empty tuple, because an ordinary optimiser opens no inner fold loop. Then the loop that reaches it is the outermost and the nearest loop for every field.

A meta-optimiser whose inner cross-validation reads a field directly names that field here. For example, `NestedClustered` names `opti`, which it runs once for each cluster as `cross_val_predict(opti, …; cols = cl)`. Then [`time_dependent_fields`](@ref) leaves a schedule with `bind = :nearest` in that field for the inner loop, and so do the update, the reset and the check of the fold count. Without this rule, the reset at the start of `_optimise` replaces such a schedule with its `default` before the inner cross-validation reads it.

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

Returns the names of the fields of `opt` whose type can hold a [`TimeDependent`](@ref).

[`time_dependent_fields`](@ref) reads the values of these fields only. The type of a field decides whether it can hold a schedule. A constructor that takes a schedule, see [`TD_Option`](@ref), records it in a type parameter. So a field that holds no schedule has a type whose intersection with `TimeDependent` is empty.

A generated function computes the tuple once for each type of optimiser. For a `JuMPOptimiser` that holds no schedule, the tuple is empty at compile time, so `split` and `_optimise` read none of its fields. The tuple comes from the field types, and no list of fields is written by hand.

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

Returns `true` when the current fold loop can resolve a schedule with `bind = :nearest` in the field `f` of `opt`.

The answer is for each field, not for each optimiser. A loop with `all_binds = true` resolves such schedules in every field except the fields that the optimiser gives to its own inner fold loop, see [`inner_fold_fields`](@ref). For those fields, the inner loop is the nearest loop. A pass resolves a field when `entitled(opt, f, all_binds) || bind === :outermost` is `true`.

# Related

  - [`inner_fold_fields`](@ref)
  - [`time_dependent_fields`](@ref)
"""
function entitled(opt, f::Symbol, all_binds::Bool)::Bool
    return all_binds && !(f in inner_fold_fields(opt))
end
"""
    time_dependent_fields(opt, all_binds::Bool = true)

Returns the names of the fields of `opt` that hold a [`TimeDependent`](@ref) which the current fold loop resolves.

The function reads only the fields whose type can hold a schedule, see [`time_dependent_candidate_fields`](@ref). So an optimiser that holds no schedule returns an empty tuple, and the function reads none of its fields.

The argument `all_binds` states a fact about the position of the fold loop, and the field `bind` of a schedule cannot state it. Take an outer fold loop that runs a meta-optimiser, whose inner cross-validation runs an estimator with a schedule that has `bind = :nearest`. Both loops reach that field. The outer loop must pass through the meta-optimiser, because that is how it resolves the schedules with `bind = :outermost` of the inner estimator. It must leave the schedule with `:nearest`, because it is not the nearest loop. The inner loop reaches the same estimator directly, and it must resolve that schedule.

So every ordinary fold loop passes `all_binds = true`. Such a loop is the outermost and the nearest loop, so it resolves every schedule that remains. A meta-optimiser passes `false` when it recurses into the estimators of its own inner cross-validation.

[`inner_fold_fields`](@ref) refines the rule for each field. With `all_binds = true`, a schedule with `:nearest` in a field that the optimiser gives to its own inner fold loop stays in place, see [`entitled`](@ref).

# Arguments

  - `opt`: The optimiser.
  - `all_binds::Bool = true`: When `true`, return each field that holds a schedule, except the schedules with `:nearest` in the fields of [`inner_fold_fields`](@ref). When `false`, return only the fields whose schedule has `bind === :outermost`.

# Returns

  - `fns::Tuple`: The names of the fields.

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

Checks each entry of the schedules in `args` with the keyword constructor of `T`.

The constructor of an optimiser calls this function, so a type error or an error between two fields shows at construction and not in the middle of a backtest. The argument `args` holds the arguments of the constructor, and `defaults` holds the static defaults that [`time_dependent_field_defaults`](@ref) gives. A call that the check makes holds no schedule, so the recursion stops.

# Algorithm

 1. Collect the names `tdfs` of the arguments that hold a [`TimeDependent`](@ref). Return when there is none.
 2. Find a static value for each of them with [`time_dependent_stand_in`](@ref). Return with no check when one of them has none. That is a function schedule in a required field, whose value exists only at a fold.
 3. Replace each schedule in `args` with its static value, giving `base`.
 4. For each name in `tdfs`, call the keyword constructor of `T` with `base`, once for each entry of the schedule and once for its explicit `default`, see [`substitute_time_dependent_entries`](@ref).

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

Calls the keyword constructor of `T` with each vector entry of the schedule `td`, and with its explicit `default`, in the field `f` of `base`.

The second method takes a value that is not a schedule, and calls nothing. [`assert_time_dependent_substitution`](@ref) passes only schedules. The second method states this fact to a static analyser, which cannot read the filter at a call whose arguments hold no schedule.

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

Returns a valid static value for a field that holds a [`TimeDependent`](@ref), in a `Some`, or `nothing` when none exists.

[`assert_time_dependent_substitution`](@ref) gives the other scheduled fields these values while it checks one of them. This function never throws, unlike [`time_dependent_reset_value`](@ref). A schedule with no value outside every fold loop is valid at construction, and fails only when it reaches a solve with no fold.

# Algorithm

 1. When `td.default` is not [`NoDefault`](@ref), return it.
 2. Read the static default of `field` from `defaults`, which is `nothing` for a field that `defaults` does not list. Return it when it is not `NoDefault`.
 3. When `td.val` is a vector, return its first entry.
 4. Else return `nothing`. A function schedule in a required field has a value only at a fold.

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

Checks that each vector schedule in `opt` has exactly `n` entries.

The cross-validation fold loops call it directly after `split`, before any fold runs. The fallback checks nothing. An optimiser with scheduled fields checks the fields that [`time_dependent_fields`](@ref) returns, and an optimiser that wraps others recurses. When `all_binds` is `false`, the check leaves out the schedules with `bind === :nearest`. The nearest fold loop checks them against its own fold count.

# Validation

  - Each vector schedule that the check reads has `n` entries, else a `DimensionMismatch` is thrown.

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

Checks the fold count of each field of an optimiser that holds a [`TimeDependent`](@ref) which the current fold loop resolves.

# Validation

  - Each vector schedule has `n` entries, else a `DimensionMismatch` that names the field is thrown.

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

Returns `true` if an optimiser configuration holds a schedule in one of its fields.

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

Resolves the schedules of an optimiser configuration at the fold that `ctx` describes.

The function rebuilds the configuration with its keyword constructor, which checks the values, and puts the value at the fold into each scheduled field. So the answer is an ordinary static configuration. When `all_binds` is `false`, the schedules with `bind === :nearest` stay in place for the nearest fold loop.

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

Replaces the schedules of an optimiser configuration with their values outside every fold loop, see [`time_dependent_reset_value`](@ref).

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

Rebuilds an estimator with its keyword constructor, with the fields in `repl` replaced.

The other fields keep their values. The keyword constructor checks every rule of construction again.

# Algorithm

 1. Read every field of `x` into a `NamedTuple` keyed by the field names.
 2. Merge `repl` into it, so a name in `repl` replaces the value of that field.
 3. Call the keyword constructor of the type of `x` with the merged tuple.

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

Returns `true` if the optimiser holds a schedule.

The fallback returns `false`. An optimiser returns `true` when one of its fields holds a [`TimeDependent`](@ref), see [`time_dependent_fields`](@ref). An optimiser that wraps others recurses into its inner optimiser and its fallback.

# Arguments

  - `opt`: An optimisation estimator or result, or a vector of them.

# Returns

  - `flag::Bool`: `true` if the optimiser holds a schedule.

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

Resolves the schedules of `opt` at the fold that `ctx` describes.

The fallback returns the estimator unchanged. An optimiser with scheduled fields rebuilds itself with its keyword constructor, which checks the values, and puts the value at the fold into each scheduled field. So the answer is an ordinary static estimator. An optimiser that wraps others recurses.

# Arguments

  - `opt`: An optimisation estimator or result.
  - `ctx::TimeDependentContext`: The context of the fold.
  - `all_binds::Bool = true`: When `false`, the schedules with `bind === :nearest` stay in place for the nearest fold loop. A meta-optimiser passes `false` when it recurses into the estimators of its own inner fold loop. A fold loop passes `true`.

# Returns

  - `opt`: The estimator with its schedules resolved.

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

Rebuilds an optimiser with the value at the fold `ctx` in each field that holds a [`TimeDependent`](@ref).

The methods of [`update_time_dependent_estimator`](@ref) for the optimisers call this function.

# Algorithm

 1. Collect the names `tdfs` of the fields that the current fold loop resolves, with [`time_dependent_fields`](@ref). Return `opt` unchanged when there is none.
 2. Resolve each of them with [`time_dependent_value`](@ref), giving `repl`.
 3. Rebuild `opt` with [`rebuild_estimator`](@ref).

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

Returns a `NamedTuple` of the static defaults of the optimiser fields that can hold a [`TimeDependent`](@ref), for the fields whose default is not `nothing`.

[`reset_time_dependent_estimator`](@ref) reads it to replace the schedules in a solve with no fold. A field that the tuple does not list has the default `nothing`. A required field has no static default, and the fields that hold an optimiser are such fields. The tuple lists a required field with [`NoDefault`](@ref). That is not a value of the field. It states that a schedule in the field must carry its own `default`. The fallback method returns an empty tuple.

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

Returns the value of a field that holds a [`TimeDependent`](@ref), outside every fold loop.

# Algorithm

 1. When `td.default` is not [`NoDefault`](@ref), return it.
 2. Read the static default of `field` from `defaults`, see [`time_dependent_field_defaults`](@ref). It is `nothing` for a field that `defaults` does not list.
 3. When the static default is `NoDefault`, throw. Else return it.

# Validation

  - The schedule or the field has a value outside every fold loop, else a [`TimeDependentDefaultError`](@ref) that names the field and the type of `opt` is thrown.

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

Replaces each [`TimeDependent`](@ref) field of `opt` with its value outside every fold loop, and recurses into the optimisers that `opt` wraps.

A schedule is defined only over the folds of a cross-validation scheme. So a solve with no fold runs with each scheduled field at its value outside every fold loop, which [`time_dependent_reset_value`](@ref) gives. The `_optimise` methods call this function first. An estimator that [`update_time_dependent_estimator`](@ref) resolved holds no schedule, so this function returns it unchanged.

This fallback serves every optimisation estimator and every optimisation result, and returns `opt` unchanged. An optimiser that can hold a schedule adds a method that rebuilds it, and an optimiser that wraps others adds a method that recurses.

# Related

  - [`TimeDependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`is_time_dependent`](@ref)
"""
function reset_time_dependent_estimator(opt::Union{<:OptimisationEstimator,
                                                   <:OptimisationResult})
    return opt
end
function reset_time_dependent_estimator(::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuilds an optimiser with the value outside every fold loop in each field that holds a [`TimeDependent`](@ref).

The methods of [`reset_time_dependent_estimator`](@ref) for the optimisers call this function.

# Algorithm

 1. Collect the names `tdfs` of the scheduled fields with [`time_dependent_fields`](@ref), which leaves out the schedules with `:nearest` in the fields of [`inner_fold_fields`](@ref). Return `opt` unchanged when there is none.
 2. Read the static defaults of `opt` with [`time_dependent_field_defaults`](@ref).
 3. Find the value of each field in `tdfs` with [`time_dependent_reset_value`](@ref), giving `repl`.
 4. Rebuild `opt` with [`rebuild_estimator`](@ref).

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

Returns `true`, because a [`TimeDependent`](@ref) is a schedule.

# Related

  - [`is_time_dependent`](@ref)
"""
function is_time_dependent(::TimeDependent)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Checks that a [`TimeDependent`](@ref) in the place of an optimiser resolved to a value that can optimise or predict.

The type of a vector schedule and of a [`TimeDependentOptimiserCallable`](@ref) states the kind of their value before any fold runs, see [`TD_OptE_Opt`](@ref). A bare function `ctx -> optimiser` and a [`PreviousWeightsFunction`](@ref) that holds one do not. So this function checks their value when the fold loop puts it into place.

# Validation

  - `opt` is an [`OptE_Opt`](@ref), else an `ArgumentError` that names the type of `opt` is thrown.

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

Resolves a [`TimeDependent`](@ref) in the place of an optimiser to the optimiser of fold `ctx.i`.

Entry `i` can be an estimator or a precomputed result, so a mixed schedule optimises at some folds and predicts at others. The resolved optimiser can hold schedules of its own. The function resolves them with the same context, so its schedules with `:outermost` resolve against this fold loop.

# Algorithm

 1. When `all_binds` is `false` and `td.bind` is not `:outermost`, return `td` unchanged. The fold loop that the optimiser opens resolves such a schedule, not the loop that reached the optimiser.
 2. Resolve `td` at the fold with [`time_dependent_value`](@ref), giving `opt`.
 3. Check `opt` with [`assert_time_dependent_optimiser`](@ref).
 4. Return `update_time_dependent_estimator(opt, ctx, all_binds)`.

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

Checks that a [`TimeDependent`](@ref) in the place of an optimiser has one entry for each fold, and that the schedules in each entry have the same fold count.

Entry `i` runs at fold `i` of this loop, so its own schedules with `:outermost` resolve against this loop too, see [`update_time_dependent_estimator`](@ref). The check does not read `default`, which runs only outside every fold loop, where its schedules reset.

When `all_binds` is `false` and `td.bind` is not `:outermost`, the function checks nothing. The fold loop that the optimiser opens checks the schedule against its own fold count.

# Validation

  - A vector schedule has `n` entries, else a `DimensionMismatch` is thrown.
  - Each entry passes [`assert_time_dependent_fold_count`](@ref) with the same `n`.

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

Returns the optimiser that a [`TimeDependent`](@ref) in the place of an optimiser gives outside every fold loop.

The place of an optimiser is required and has no static default. So the schedule must carry its own `default`. The function resets that optimiser too, so its own schedules take their values outside every fold loop.

# Validation

  - `td.default` is not [`NoDefault`](@ref), else a [`TimeDependentDefaultError`](@ref) is thrown.

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

Returns the values of a [`TimeDependent`](@ref) that are known before any fold runs.

They are the entries of a vector schedule, and `default` when it is set. A function schedule gives no entry, because its values do not exist before it runs.

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
    assert_no_nearest_bind_optimiser_schedule(x, field::Symbol, opt_name::Symbol)

Refuses a [`TimeDependent`](@ref) with `bind = :nearest` in a field that holds an optimiser and that no inner fold loop reads.

The field `bind` selects the fold loop whose index the schedule reads. So `:nearest` differs from `:outermost` only where the optimiser opens its own fold loop and gives the field to it. [`inner_fold_fields`](@ref) names those fields: `NestedClustered.opti`, which runs once for each cluster, and `Stacking.opti[k]`, which runs once for each candidate. In every other field, the loop that reaches the optimiser is the nearest loop, and the two values of `bind` select the same loop.

This function guards the fields that have no inner fold loop:

  - The fallback `fb` of every optimiser. The fallback chain runs inside the solve of one fold, and it has no fold index of its own. So `:nearest` there selects the same loop as `:outermost`, or it is wrong behind the inner cross-validation of a meta-optimiser. There it reads the index of a tuning fold and not the period of the backtest. A schedule with `:outermost` states every fallback that changes by fold, and an entry `nothing` turns the fallback off at a fold, see [`TDO_OptE_Opt`](@ref).
  - The outer optimiser `opto`, which reads the combined inner output once for each solve.
  - `SubsetResampling.opt`, whose inner loop runs over random subsets of the assets, not over folds of time.

So a schedule with `:nearest` in one of these fields has no nearest fold loop, and the constructor refuses it. The function checks nothing when `x` is not a `TimeDependent`.

# Validation

  - `x.bind !== :nearest` when `x` is a `TimeDependent`, else an `ArgumentError` that names the field and the optimiser is thrown.

# Related

  - [`assert_nearest_optimiser_schedule`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`TDO_OptE_Opt`](@ref)
"""
function assert_no_nearest_bind_optimiser_schedule(x, field::Symbol,
                                                   opt_name::Symbol)::Nothing
    if isa(x, TimeDependent)
        @argcheck(x.bind !== :nearest,
                  ArgumentError("field `$field` of $opt_name holds a `bind = :nearest` TimeDependent schedule, but no inner fold loop of $opt_name consumes `$field`, so there is no nearest fold loop for it to bind to. Use `bind = :outermost`: the fold loop that reaches the $opt_name resolves the schedule."))
    end
    return nothing
end
"""
    assert_nearest_optimiser_schedule(x, field::Symbol, cv, opt_name::Symbol)

Checks a [`TimeDependent`](@ref) with `bind = :nearest` in a field that holds an optimiser and that the inner cross-validation of the optimiser reads.

Two parts of the optimiser read such a field. The inner cross-validation resolves the schedule at each fold. The solve on the full sample, which is the fit of `wi` of a meta-optimiser or the optimisation of each cluster, always resolves it to its `default`. So the constructor needs two conditions, which the validation below states. The first condition applies to this field only. Elsewhere a schedule with no `default` is valid at construction.

The function checks nothing when `x` is not a `TimeDependent` with `bind = :nearest`.

# Validation

  - `x.default` is not [`NoDefault`](@ref), else a [`TimeDependentDefaultError`](@ref) is thrown. With no `default`, every solve throws when the solve on the full sample reads the schedule.
  - `cv !== nothing`, else an `ArgumentError` is thrown. With no inner cross-validation, the schedule always takes its `default`, and it has no effect.

# Related

  - [`assert_no_nearest_bind_optimiser_schedule`](@ref)
  - [`inner_fold_fields`](@ref)
  - [`TimeDependentDefaultError`](@ref)
"""
function assert_nearest_optimiser_schedule(x, field::Symbol, cv, opt_name::Symbol)::Nothing
    if isa(x, TimeDependent) && x.bind === :nearest
        @argcheck(!isa(x.default, NoDefault),
                  TimeDependentDefaultError("a `bind = :nearest` schedule in `$field` of $opt_name must supply a `default`: besides the inner cross-validation fold loop, `$field` also has a fold-less full-sample consumer that always resolves the schedule to its `default`, so a defaultless one would throw on every solve. Give it a fold-less optimiser: TimeDependent(val, :nearest; default = opt)."))
        @argcheck(!isnothing(cv),
                  ArgumentError("a `bind = :nearest` schedule in `$field` of $opt_name requires `cv`: without an inner cross-validation there is no inner fold loop, so the schedule could only ever resolve to its `default` — silently inert. Provide `cv`, or use `bind = :outermost` so the fold loop that reaches the $opt_name consumes it."))
    end
    return nothing
end
#! End: TimeDependent as an optimiser in its own right.
"""
    const VecOptE_Opt = AbstractVector{<:OptE_Opt}

Alias for a vector of continuous optimisation estimators or results.

A method that reads many optimisers, such as the candidates of a meta-optimiser, dispatches on it.

# Related

  - [`OptE_Opt`](@ref)
"""
const VecOptE_Opt = AbstractVector{<:OptE_Opt}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Applies [`factory`](@ref) to each entry of a [`TimeDependent`](@ref) and to its `default`, and rebuilds the schedule.

A schedule can stay in place after the fold loop resolves the others, for example an element with `bind = :nearest` that the inner cross-validation of a meta-optimiser resolves. So the `factory` pass after the resolution must reach into it. A function schedule does not change, because its values do not exist yet. At its fold, it receives the context, which holds `w_prev`.

# Algorithm

 1. When `td.val` is a vector, apply `factory(x, args...)` to each entry `x`.
 2. When `td.default` is not [`NoDefault`](@ref), apply `factory` to it too.
 3. Return a new schedule with the new values and the same `bind`.

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

Checks the requirements of an inner optimiser of [`NestedClustered`](@ref) for each element of a vector of optimisers.

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

Returns `true` if one element of a vector of optimisers needs the weights of the previous fold.

# Related

  - [`needs_previous_weights`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function needs_previous_weights(opt::VecOptE_Opt_TD)
    return any(needs_previous_weights, opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns `true` if one element of a vector of optimisers holds a schedule, or is one.

# Related

  - [`is_time_dependent`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
function is_time_dependent(opt::VecOptE_Opt_TD)
    return any(is_time_dependent, opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Applies [`update_time_dependent_estimator`](@ref) to each element of a vector of optimisers, and returns a new vector.

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

Applies [`assert_time_dependent_fold_count`](@ref) to each element of a vector of optimisers.

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
