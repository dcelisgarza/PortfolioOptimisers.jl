"""
$(DocStringExtensions.TYPEDEF)

Declares that an estimator takes the online step from a buffer of the observations it has seen.

An `Online` goes directly in the estimator field that it wraps, for example `HighOrderPriorEstimator(; pe = Online(EmpiricalPrior()))`, or the `pe` of an optimiser, `JuMPOptimiser(; pe = Online(EmpiricalPrior()), slv = …)`. It is transient, as [`TimeDependent`](@ref) is. [`update_online_estimator`](@ref) walks the fields that hold a wrapper, seeds the `cache` of each wrapped estimator with a [`SampleBufferState`](@ref), and rebuilds the outer estimator through its keyword constructor. The result is an ordinary estimator that carries a state. No `Online` exists after that, so each verb downstream receives a plain estimator.

Only a field whose bound is `Onl` admits a wrapper. These are the `pe` field of a wrapping prior and of an optimiser. A moment field inside a prior, such as `ce`, `me`, `ske` or `kte`, admits none, so the bound of the field refuses `EmpiricalPrior(; ce = Online(…))` before any warm-up. The prior owns the rows. It holds its observations once, at the bottom of its chain, and it refits from those rows each member that does not fold. A buffer under one of its moments would hold the same rows a second time, so wrap the prior. Wrap a moment estimator alone only when it is the estimator under study, as in `update_online_estimator(Online(Coskewness()))`.

It differs from [`TimeDependent`](@ref) in when it resolves. A schedule resolves again at each fold, because its value changes at each fold. An `Online` resolves once, at warm-up. After warm-up, the step passes the state from fold to fold, and a second resolution would discard the buffer.

A wrapper replaces an exact fold, and it does not add to one. Each family that folds exactly narrows the `cache` type parameter of its own [`partial_fit!`](@ref) methods to the state of that fold, so a wrapped estimator never reaches them. It buffers its observations, and it answers each call with no data with the batch verb over the rows of the buffer. The type of the state selects the route, and the caller selects the type when it wraps the estimator or not.

Wrap an estimator in one of these cases:

  - Its estimate has no exact incremental fold.
  - It folds exactly, and it carries its observations for a consumer downstream.
  - Its estimate must come from a window, not from every observation.

An estimator that folds exactly and carries nothing needs no wrapper. Unwrapped, it answers [`partial_fit!`](@ref) and seeds its own state at the first call, and it uses the memory of one state, not the memory of a buffer.

A wrapper and a [`TimeDependent`](@ref) schedule do not wrap each other, because they resolve at different times. Neither `Online(TimeDependent(…))` nor a schedule whose entry or `default` is an `Online` is admissible. A wrapper inside a schedule entry would resolve at no fold, or it would seed again at each fold and discard the buffer that the step passes on. They compose the other way, in two ordinary forms. An estimator that an `Online` wraps can hold schedules of its own, which the seed does not change and which resolve at each fold after it. One outer estimator can hold a wrapper in one field and a schedule in another, and each resolves at its own time. The two field scans are disjoint by construction, because a field that holds one kind is not a candidate of the other scan. So neither resolution reaches the wrapper of the other.

On a scheme, the same word declares an Online Scheme. `Online{<:WalkForwardEstimator}` wraps a walk-forward whose folds the loop fits by the online step, not by a refit. The loop warms one estimator up on the first training window, folds the new observations of each fold into it, and reads it out where a refit would run. Only the function constructor of the scheme builds one, [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref) or [`OnlineHindsightSplit`](@ref). Each takes the keywords of its scheme except the window knob, and it sets that knob `true`, because a fold cannot remove an observation. `Online(cv)` written by hand is refused with an error that names it. On a scheme the wrapper is not transient. Nothing resolves it, the loop reads it at each fold, and each function that takes the scheme decides on its type. It carries no `max_history`, because the estimator declares a window through the wrapper on the prior.

`max_history` caps the buffer, and the cap is the window. An uncapped buffer gives the batch fit over every observation folded so far. A capped buffer gives the batch fit over the last `max_history` observations. This holds for the estimate and for each consumer that reads the observations, such as the scenario risk measures CVaR, EVaR and CDaR. The rule has no special case. A buffer always means the batch verb over its rows. An unwrapped estimator is not affected. It folds exactly, and it stays fitted over every observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Online(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator};
           max_history::Option{<:Integer} = nothing)
    Online(; est::Union{AbstractEstimator, StatsBase.CovarianceEstimator},
           max_history::Option{<:Integer} = nothing)

## Validation

  - `est` is not an `Online`. An `ArgumentError` is thrown otherwise.
  - `est` is not a cross-validation scheme, because the function constructor of a scheme builds its Online Scheme. An `ArgumentError` naming the constructors is thrown otherwise.
  - `est` is not an [`OnlinePortfolioSelection`](@ref) head, because a refit of a head is a batch walk-forward, and the library runs both settings of the wrapper in other forms. An `ArgumentError` naming those forms is thrown otherwise.
  - `est` has a `cache` field. An `ArgumentError` is thrown otherwise. Every covariance estimator of the library has one. A covariance estimator with no such field, such as a `StatsBase.CovarianceEstimator` of another package, takes the buffer of a [`PortfolioOptimisersCovariance`](@ref) that holds it.
  - No field of the tree of `est` holds a windowed estimator under [`SeedWindow`](@ref), because the wrapper refits the tree at each step, and a refit has no first fit to remember. [`seed_window_path`](@ref) finds the field, and an `ArgumentError` that names it is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> Online(Covariance(; alg = SemiMoment()); max_history = 250)
Online
          est ┼ Covariance
              │    me ┼ SimpleExpectedReturns
              │       │   w ┴ nothing
              │    ce ┼ GeneralCovariance
              │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
              │       │    w ┴ nothing
              │   alg ┼ SemiMoment()
              │     w ┴ nothing
  max_history ┴ Int64: 250
```

# Related

  - [`SampleBufferState`](@ref)
  - [`Online_Option`](@ref)
  - [`update_online_estimator`](@ref)
  - [`online_fields`](@ref)
  - [`TimeDependent`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`OnlineDateWalkForward`](@ref)
  - [`OnlineHindsightSplit`](@ref)
"""
@concrete struct Online <: AbstractEstimator
    """
    Estimator that the wrapper seeds a buffer into. The field takes this value after the wrapper resolves.
    """
    est
    """
    $(field_dict[:pf_max_history])
    """
    max_history
    function Online(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator};
                    max_history::Option{<:Integer} = nothing)
        @argcheck(hasfield(typeof(est), :cache),
                  ArgumentError("`$(typeof(est))` has no `cache` field, so it has nowhere to carry a sample buffer and cannot be wrapped in `Online`. A covariance estimator of this kind takes the buffer of a `PortfolioOptimisersCovariance(; ce = …)`, which carries one for any estimator that it holds, and a prior holds the rows for each of its members."))
        path = seed_window_path(est)
        @argcheck(isnothing(path),
                  ArgumentError("`$(typeof(est).name.name)` holds a `SeedWindow` at `$(path)`, and `Online` refits the estimator over its buffer at each step. A refit has no first fit to remember, so it cannot keep the window of the first fit alone. Fold the estimator without `Online` in a prior that folds, such as `EmpiricalPrior`, or set the rule to `RollingWindow()`."))
        if !isnothing(max_history)
            @argcheck(max_history > zero(max_history),
                      DomainError(max_history, "max_history must be positive"))
        end
        return new{typeof(est), typeof(max_history)}(est, max_history)
    end
    # The Online Scheme path. Only the function constructors of the walk-forwards call it,
    # with the window knob already set, so the wrapped scheme is expanding by construction
    # and no value is checked here.
    function Online{T1, T2}(cv::T1,
                            max_history::T2) where {T1 <: CrossValidationEstimator,
                                                    T2 <: Nothing}
        return new{T1, T2}(cv, max_history)
    end
end
function Online(cv::CrossValidationEstimator; kwargs...)
    return throw(ArgumentError("`Online` on a scheme is an Online Scheme, and one is built by its function constructor alone — `OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit` — never by wrapping a `$(nameof(typeof(cv)))` by hand. Each constructor takes its scheme's keywords minus the window knob and sets that knob `true`, because a fold cannot un-fold an observation, so an online run is expanding by construction. A scheme with no constructor of its own has no online form."))
end
function Online(::Online, args...; kwargs...)
    return throw(ArgumentError("est cannot be an Online: wrappers do not nest. One wrapper declares one buffer, and an estimator it wraps may carry wrappers of its own — they resolve at the same warm-up — but they belong in its fields, not inside this one."))
end
function Online(; est::Union{AbstractEstimator, StatsBase.CovarianceEstimator},
                max_history::Option{<:Integer} = nothing)::Online
    return Online(est; max_history = max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an [`Online`](@ref) that reached a fold without being resolved, naming the warm-up that resolves it.

[`update_online_estimator`](@ref) replaces each wrapper that it can reach at warm-up. So a wrapper that reaches a fold is one that the warm-up did not see. A callable [`TimeDependent`](@ref) makes one. The value of a schedule is computed at each fold, after the warm-up, so nothing seeds a wrapper that the callable returns. The constructor of a schedule refuses a wrapper in its vector and `default` forms. It cannot refuse the return value of a callable, because that value does not exist until the fold.

A callable that returns the estimator that the wrapper would make does not solve the problem. A schedule replaces the value of the field at each fold, and the step passes the buffer through that field. So a schedule that returns a captured estimator returns the buffer as it was when the closure was built, and it discards each step since then.

# Arguments

  - `::Online`: The unresolved wrapper.

# Validation

  - Always throws an `ArgumentError`.

# Returns

  - Never returns.

# Related

  - [`Online`](@ref)
  - [`update_online_estimator`](@ref)
  - [`assert_sample_buffer`](@ref)
"""
function assert_sample_buffer(::Online)
    return throw(ArgumentError("an `Online` is a declaration, not an estimator that folds, and this one reached a fold unresolved. `update_online_estimator` resolves every wrapper it can reach at warm-up; a wrapper that survives to a fold came from a callable `TimeDependent`, whose value is computed per fold and so is never seeded. A schedule and a wrapper do not compose in that direction: a schedule replaces the field's value every fold, while the buffer is threaded through it, so a schedule that returns a captured estimator returns the buffer as it stood when the closure was built."))
end
"""
    const Online_Option{X} = Union{Nothing, Online, X}

Alias for a field that accepts `nothing`, a static estimator of type `X`, or an [`Online`](@ref) declaration.

The fields whose constructor signatures use this alias are the fields that can take the online step from a sample buffer, as [`TD_Option`](@ref) marks the fields that can change over folds.

# Related

  - [`Online`](@ref)
  - [`Option`](@ref)
  - [`online_fields`](@ref)
"""
const Online_Option{X} = Union{Nothing, Online, X}
"""
    const Onl{X} = Union{Online, X}

Alias for a required field that accepts a static estimator of type `X` or an [`Online`](@ref) declaration.

This is the form of [`Online_Option`](@ref) for a required field, as [`TD`](@ref) is for [`TD_Option`](@ref). A field that the outer estimator requires takes this alias. A caller can declare the online step there, and the signature refuses `nothing`.

# Related

  - [`Online`](@ref)
  - [`Online_Option`](@ref)
  - [`online_fields`](@ref)
"""
const Onl{X} = Union{Online, X}
"""
    const CVE_Onl = Union{CrossValidationEstimator, Online{<:CrossValidationEstimator}}

Alias for a cross-validation scheme, plain or an Online Scheme.

An Online Scheme is an [`Online`](@ref) around a walk-forward, built by the function constructor of the scheme, [`OnlineIndexWalkForward`](@ref), [`OnlineDateWalkForward`](@ref) or [`OnlineHindsightSplit`](@ref). Its supertype is `AbstractEstimator` and not `CrossValidationEstimator`, because a struct has one supertype. Each function that takes a scheme dispatches on this alias, or on an alias built from it, so the wrapped form reaches the same functions as the plain form. The fold loop then reads [`folds_are_stepped`](@ref) from its type.

# Related

  - [`CrossValidationEstimator`](@ref)
  - [`Online`](@ref)
  - [`CVER`](@ref)
  - [`folds_are_stepped`](@ref)
"""
const CVE_Onl = Union{CrossValidationEstimator, Online{<:CrossValidationEstimator}}
"""
    online_candidate_fields(x)

Field names of `x` whose type admits an [`Online`](@ref), the candidate set that [`online_fields`](@ref) narrows by value.

`fieldtype` alone decides whether a field can hold a wrapper, because a `@concrete` estimator records the type of the value in the type parameter of the field. The type of a field that holds a static estimator does not intersect [`Online`](@ref). So a generated function computes the tuple once for each estimator type, and for an estimator with no wrapper the tuple is empty at compile time. The warm-up then walks no field.

# Related

  - [`Online`](@ref)
  - [`online_fields`](@ref)
"""
@generated function online_candidate_fields(::T) where {T}
    fns = Tuple(f
                for f in fieldnames(T)
                if typeintersect(fieldtype(T, f), Online) !== Union{})
    return :($fns)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the tuple of field names of `est` whose values are [`Online`](@ref).

The scan reads the fields of the outer estimator, so the constructor signatures that admit a wrapper, through [`Online_Option`](@ref) and [`Onl`](@ref), decide which estimators can take the step from a buffer. No list is kept by hand. The scan visits only the candidates of [`online_candidate_fields`](@ref), and the compiler removes the other fields.

# Arguments

  - `est`: Estimator whose fields are scanned.

# Returns

  - `fns::Tuple`: The field names holding an [`Online`](@ref).

# Related

  - [`Online`](@ref)
  - [`online_candidate_fields`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_fields(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator})
    fns = online_candidate_fields(est)
    return filter(f -> isa(getfield(est, f), Online), fns)
end
"""
    update_online_estimator(est)

Resolves the [`Online`](@ref) declarations of an estimator, seeding the sample buffer each one asks for.

The warm-up calls it once, before the first fold. It replaces each wrapper with the estimator that the wrapper holds, rebuilt through its keyword constructor with the state of [`online_state_seed`](@ref) in `cache`. That state is an empty [`SampleBufferState`](@ref) with the cap of the wrapper, unless the family has a seed of its own. The function then rebuilds the outer estimator through its keyword constructor, so each construction check runs again. The result holds no `Online`, and each estimator that was wrapped carries the state that its refit or its carry reads.

The function finds a wrapper in the fields of the estimator that it receives, and in the fields of an estimator inside a wrapper. An outer estimator that passes estimators across a boundary of its own, such as a meta-optimiser or a pipeline, has a method of its own that recurses, as it has for [`update_time_dependent_estimator`](@ref).

The function takes no fold context, unlike [`update_time_dependent_estimator`](@ref). The seed reads nothing from a fold, because the wrapper resolves once and not at each fold.

# Algorithm

 1. For an [`Online`](@ref), rebuild the wrapped estimator with `cache` set to the state of [`online_state_seed`](@ref), and continue with that estimator.
 2. Find the fields of the estimator that hold a wrapper, with [`online_fields`](@ref). Return the estimator when there is none.
 3. Resolve the value of each of these fields with this function.
 4. Rebuild the estimator with the resolved values through [`rebuild_estimator`](@ref).

# Arguments

  - `est`: Estimator, wrapper, or `nothing`.

# Returns

  - Estimator carrying a seeded buffer wherever a wrapper stood, and holding no [`Online`](@ref).

# Related

  - [`Online`](@ref)
  - [`SampleBufferState`](@ref)
  - [`online_fields`](@ref)
  - [`rebuild_estimator`](@ref)
"""
function update_online_estimator(est::Union{AbstractEstimator,
                                            StatsBase.CovarianceEstimator})
    fns = online_fields(est)
    if isempty(fns)
        return est
    end
    repl = NamedTuple{fns}(map(f -> update_online_estimator(getfield(est, f)), fns))
    return rebuild_estimator(est, repl)
end
function update_online_estimator(::Nothing)
    return nothing
end
"""
    estimator_fields(x)

Field names of `x` whose type admits an estimator, the candidate set that [`online_wrapper_path`](@ref) and [`copy_states`](@ref) walk.

`fieldtype` alone decides whether a field can hold an estimator, because a `@concrete` estimator records the type of the value in the type parameter of the field. So a generated function computes the tuple once for each estimator type, as for [`online_candidate_fields`](@ref). A field that holds a vector of estimators or a result is not a candidate. A vector is a batch configuration that the step never folds into, and a result carries no state field. A walk that must reach the estimators a vector holds reads every field instead, as [`online_entry_state`](@ref) and [`seed_window_path`](@ref) do. A field that holds a [`TimeDependent`](@ref) schedule is a candidate, because a schedule is an estimator. The `TimeDependent` methods of the two walks answer `nothing` for it, because the entries of a schedule resolve at each fold.

# Related

  - [`online_entry_state`](@ref)
  - [`online_candidate_fields`](@ref)
"""
@generated function estimator_fields(::T) where {T}
    fns = Tuple(f
                for f in fieldnames(T)
                if typeintersect(fieldtype(T, f),
                                 Union{AbstractEstimator, StatsBase.CovarianceEstimator}) !==
                   Union{})
    return :($fns)
end
"""
    online_entry_state(est)
    online_entry_state(v::AbstractVector)
    online_entry_state(p::Pair)
    online_entry_state(::TimeDependent)

Names the first field of an estimator tree that carries a partial-fit state, or answers `nothing`.

The cold start of the fold loop uses this walk. The online arm of [`fold_loop`](@ref) reads its argument as configuration only, so it refuses by name an estimator that holds a state, and this walk finds the state. The answer is the path from the root, for example `"opt.pe.me.cache"` for the state of a mean estimator under a JuMP head. The walk reads every field, not only the fields of [`estimator_fields`](@ref), because a state can stand in an estimator that a vector holds. A Descriptor that the caller folded before the call, in the factor list of a [`CrossSectionalFactorPrior`](@ref), is such an estimator, and the first step of the carry fold reads its state. So the first [`partial_fit!`](@ref) of that prior refuses a state with this walk too, because a direct call does not pass through the fold loop. A vector answers with the index of its element in brackets, as [`element_path`](@ref) states, and a Pair answers through its value, so the path is `"factors[2].descriptors[1].cache"` for that Descriptor. A [`TimeDependent`](@ref) schedule answers `nothing`, because its entries are batch configuration that resolves at each fold, and the loop passes no state through them. A value that is not an estimator also answers `nothing`.

# Algorithm

 1. Answer `"cache"` when `est` has a `cache` field that is not `nothing`.
 2. Walk every field of `est` in order, and find the path in each field with this function.
 3. Answer the first path found, with the field name before it, as [`tree_path`](@ref) joins them, or `nothing` when no field gives one.

# Arguments

  - `est`: The estimator, or any value a field holds.
  - `v`: A vector that a field holds.
  - `p`: A Pair that a vector holds, such as a member of a factor list.

# Returns

  - `path::Option{<:String}`: The path of the first state found, or `nothing`.

# Related

  - [`estimator_fields`](@ref)
  - [`element_path`](@ref)
  - [`online_wrapper_path`](@ref)
  - [`update_online_estimator`](@ref)
  - [`partial_fit_cache`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function online_entry_state(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator})
    # The index form keeps a type with no `cache` field, such as `HighOrderPriorEstimator`,
    # from a `getfield` by a name that it does not have.
    i = findfirst(==(:cache), fieldnames(typeof(est)))
    if !isnothing(i) && !isnothing(getfield(est, i))
        return "cache"
    end
    for f in fieldnames(typeof(est))
        path = online_entry_state(getfield(est, f))
        if !isnothing(path)
            return tree_path(f, path)
        end
    end
    return nothing
end
function online_entry_state(::Any)
    return nothing
end
"""
    assert_cold_carry_entry(est, ::Nothing, name::AbstractString) -> nothing
    assert_cold_carry_entry(est, ::Any, name::AbstractString) -> nothing

Refuses a partial-fit state in a member of a carry at its first step.

A carry reads every member as configuration at its first step: the carried rows start empty, and each member folds the new observations. A member that holds a state would fold them onto the observations of that state, so its moment would read rows that the carried rows and every refitted member do not read, and the prior would not equal the batch fit over the folded observations. The fold loop refuses such an estimator at its entry, and a direct [`partial_fit!`](@ref) does not pass through the fold loop, so each carry calls this function before its first step. The second argument marks the state of the carry itself: `nothing` at the first step, and any other value after it, when each member holds the state that the carry folded. A carry that keeps its state in a field passes that field. A carry that keeps its rows in a member, as [`HighOrderPriorEstimator`](@ref) keeps them in its embedded prior, passes the [`online_entry_state`](@ref) of that member.

# Algorithm

The method that Julia selects is the algorithm.

 1. `cache` is not `nothing`: return, because the members hold the states that the carry folded.
 2. `cache` is `nothing`: find the first state in `est` with [`online_entry_state`](@ref), and refuse it.

# Arguments

  - `est`: The estimator that starts its carry.
  - `cache`: The state of the carry, or the path of the state of the member that carries the rows, and `nothing` at the first step.
  - `name`: The name of the estimator in the message.

# Validation

  - At the first step, no member of `est` holds a partial-fit state. An `ArgumentError` that names the path of the state is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`online_entry_state`](@ref)
  - [`partial_fit!`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function assert_cold_carry_entry(est, ::Nothing, name::AbstractString)
    path = online_entry_state(est)
    @argcheck(isnothing(path),
              ArgumentError("`$(name)` starts its carry with a partial-fit state at `$(path)`. The carry reads every member as configuration at its first step, so the step would fold the new observations onto the state of that member, and the prior would no longer equal the batch fit over the folded observations. Pass the member with its `cache` at `nothing`."))
    return nothing
end
function assert_cold_carry_entry(::Any, ::Any, ::AbstractString)
    return nothing
end
"""
    online_wrapper_path(est)
    online_wrapper_path(::Online)
    online_wrapper_path(::Online{<:CrossValidationEstimator})
    online_wrapper_path(::TimeDependent)

Names the first field of an estimator tree that holds an [`Online`](@ref) declaration, or answers `nothing`.

[`assert_batch_entry`](@ref) uses this walk. Only the warm-up of the online arm of the fold loop resolves a wrapper, with [`update_online_estimator`](@ref). Without the check, a batch fit that reaches a wrapper gets a `MethodError` at `prior(pe, X)`, not a refusal. The walk reads the fields of [`estimator_fields`](@ref), and the answer is the dotted path from the root. It is `"pe"` for a wrapped prior on a naive optimiser, and `"opt.pe"` for a wrapped prior under a JuMP head. The walk does not enter a field that holds a wrapper, because the warm-up resolves the wrappers below it. A wrapper at the root answers `""`, the empty path. An Online Scheme answers `nothing`. It is a scheme in the `cv` field of a meta-optimiser, such as [`Stacking`](@ref), and not the declaration of a sample buffer, so the fold loop that runs it resolves it. A [`TimeDependent`](@ref) schedule answers `nothing`, because its constructor refuses a wrapper among its entries. A value that is not an estimator also answers `nothing`.

# Algorithm

 1. Answer `""` when `est` is an [`Online`](@ref), and `nothing` when it is an Online Scheme.
 2. Walk the fields of [`estimator_fields`](@ref) in order, and find the path in each field with this function.
 3. Answer the first path found, with the field name before it, as [`tree_path`](@ref) joins them.
 4. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, or any value a field holds.

# Returns

  - `path::Option{<:String}`: The dotted path of the first wrapper found, `""` when `est` is itself one, or `nothing`.

# Related

  - [`assert_batch_entry`](@ref)
  - [`online_entry_state`](@ref)
  - [`estimator_fields`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_wrapper_path(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator})
    for f in estimator_fields(est)
        path = online_wrapper_path(getfield(est, f))
        if !isnothing(path)
            return tree_path(f, path)
        end
    end
    return nothing
end
function online_wrapper_path(::Online)
    return ""
end
function online_wrapper_path(::Online{<:CrossValidationEstimator})
    return nothing
end
function online_wrapper_path(::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the window rule of a windowed moment estimator, the rule that says which observations its window keeps on the online step.

A windowed estimator, such as [`WindowedCovariance`](@ref), fits its inner estimator on a window of observations. In a batch fit every rule keeps the same window, the last `window` observations. The rules differ when the estimator receives its observations one block at a time.

# Interfaces

A rule is a marker for dispatch, and it holds no data. The windowed estimators hold it in their `rule` field, and the methods that fold them dispatch on its type. A new rule needs its own fold methods for the five windowed estimators.

# Related

  - [`RollingWindow`](@ref)
  - [`SeedWindow`](@ref)
  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
"""
abstract type AbstractWindowRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the last observations of the window at every fit, so the window rolls with each new observation. This is the default rule.

The estimator does not fold. On the online step, a prior that carries the observations, such as the carry of [`EmpiricalPrior`](@ref), refits it over the carried rows at each read, and the window of each refit ends at the last row. So the call with no data after any stream of blocks equals the batch fit over every row received.

# Constructors

    RollingWindow() -> RollingWindow

# Examples

```jldoctest
julia> RollingWindow()
RollingWindow()
```

# Related

  - [`AbstractWindowRule`](@ref)
  - [`SeedWindow`](@ref)
"""
struct RollingWindow <: AbstractWindowRule end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the last observations of the window at the first fit alone. After the first fit, the inner estimator folds every new observation, and no observation leaves the estimate.

In a batch fit the rule keeps the same window as [`RollingWindow`](@ref). On the online step the estimator folds. Its first fold cuts the first block to the window, folds the rows that remain into the inner estimator, and returns the estimator with `window = nothing`, so each later fold passes every row. The call with no data after a stream of blocks then gives the fit over the window of the first block, continued by the fold of every later row, and it equals no batch fit.

The rule needs an inner estimator that folds exactly, such as [`ExpWeightedCovariance`](@ref), and a prior that folds, such as the carry of [`EmpiricalPrior`](@ref). [`Online`](@ref) refuses it, because the wrapper refits its estimator at each step, and a refit has no first fit to remember.

# Constructors

    SeedWindow() -> SeedWindow

# Examples

```jldoctest
julia> SeedWindow()
SeedWindow()
```

# Related

  - [`AbstractWindowRule`](@ref)
  - [`RollingWindow`](@ref)
  - [`seed_window_path`](@ref)
  - [`partial_fit!`](@ref)
"""
struct SeedWindow <: AbstractWindowRule end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Choice Rule of an estimator, the rule that says how a choice that a fit makes over its whole sample behaves on the online step.

Some fits make a choice that reads every observation, such as the dropped member of a Factor Family in [`CrossSectionalFactorPrior`](@ref), the factor set of each asset in [`StepwiseRegression`](@ref), or the components of [`DimensionReductionRegression`](@ref). In a batch fit every rule makes the same choice. The rules differ when the estimator receives its observations one block at a time.

# Interfaces

A rule is a marker for dispatch, and it holds no data. An estimator that makes such a choice holds the rule in its `choice` field. The step of its online fold dispatches on the type of the rule. A new rule needs its own method of that step for each estimator that holds it.

# Related

  - [`BatchChoice`](@ref)
  - [`PinnedChoice`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
abstract type AbstractChoiceRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Makes the choice again over every observation at each fit. This is the default rule.

On the online step a refit reads every row of its buffer, so the choice can move from one step to the next, and the call with no data after any stream of blocks equals the batch fit over the same rows. On the carry fold of a [`CrossSectionalFactorPrior`](@ref), a step that moves the dropped member of a Factor Family solves no observation of full rank again. The raw factor returns of such an observation do not depend on the dropped member, so the step selects the columns that the new member keeps, and folds the factor prior again over them.

# Constructors

    BatchChoice() -> BatchChoice

# Examples

```jldoctest
julia> BatchChoice()
BatchChoice()
```

# Related

  - [`AbstractChoiceRule`](@ref)
  - [`PinnedChoice`](@ref)
"""
struct BatchChoice <: AbstractChoiceRule end
"""
$(DocStringExtensions.TYPEDEF)

Keeps the choice of the first fit. After the first fit, every later fit reads the same choice.

In a batch fit the rule makes the same choice as [`BatchChoice`](@ref). An [`EntropyPoolingPrior`](@ref) and a [`MeucciEntropyPoolingPrior`](@ref) build their views on the fit under the prior probabilities, and refit under the posterior weights through [`scenario_factory`](@ref), which holds the regression of that fit, so each refit reads the choice of the fit that the views read under either rule. On the online step the rule holds no state. After the first step whose rows the fit accepts, the step writes the choice into the configuration of the estimator that it returns, for example the name of the dropped member of each Factor Family into the `families` of a [`CrossSectionalFactorPrior`](@ref), the factor set of each asset into `included` of a [`StepwiseRegression`](@ref), or the components into `proj` of a [`DimensionReductionRegression`](@ref). Each later step then reads that choice. The call with no data after a stream of blocks equals the batch fit over the same rows under the choice of the first fit, which differs from a batch fit when the choice of the whole sample moves.

# Constructors

    PinnedChoice() -> PinnedChoice

# Examples

```jldoctest
julia> PinnedChoice()
PinnedChoice()
```

# Related

  - [`AbstractChoiceRule`](@ref)
  - [`BatchChoice`](@ref)
"""
struct PinnedChoice <: AbstractChoiceRule end
"""
    seed_window_path(est)
    seed_window_path(v::AbstractVector)
    seed_window_path(p::Pair)
    seed_window_path(::Any)

Names the first field of an estimator tree that holds a windowed estimator under [`SeedWindow`](@ref), or answers `nothing`.

The constructor of [`Online`](@ref) uses this walk to refuse a seed window, because the wrapper refits its estimator at each step. The answer is the path from the root to the `rule` field, such as `"ce.rule"` for the covariance member of a prior. A windowed estimator under a seed window answers `"rule"` through its own method. A value that is not an estimator answers `nothing`.

The walk reads every field, not only the fields of [`estimator_fields`](@ref). A refit reaches the estimators that a vector holds, such as the factor list of a [`CrossSectionalFactorPrior`](@ref), the Descriptors of a [`DescriptorScores`](@ref) and the priors of an [`OpinionPoolingPrior`](@ref), so the refusal must reach them too. A vector answers with the index of its element in brackets, and a Pair answers through its value, so a Descriptor in a factor list gives a path such as `"factors[2].descriptors[1].ce.rule"`. [`element_path`](@ref) walks the vector.

# Algorithm

 1. Walk every field of `est` in order, and find the path in each field with this function.
 2. Answer the first path found, with the field name before it, as [`tree_path`](@ref) joins them.
 3. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, or any value a field holds.
  - `v`: A vector that a field holds.
  - `p`: A Pair that a vector holds, such as a member of a factor list.

# Returns

  - `path::Option{<:String}`: The path of the first seed window found, or `nothing`.

# Related

  - [`SeedWindow`](@ref)
  - [`Online`](@ref)
  - [`element_path`](@ref)
  - [`tree_path`](@ref)
  - [`estimator_fields`](@ref)
"""
function seed_window_path(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator})
    for f in fieldnames(typeof(est))
        path = seed_window_path(getfield(est, f))
        if !isnothing(path)
            return tree_path(f, path)
        end
    end
    return nothing
end
function seed_window_path(::Any)
    return nothing
end
"""
    assert_batch_entry(est, entry::AbstractString)

Refuses an estimator that holds an [`Online`](@ref) at the entry of a batch fit, with an error that names the path of the wrapper.

A wrapper is a declaration that the online arm of the fold loop resolves at its warm-up. A batch fit runs no warm-up. A batch fit is a plain [`optimise`](@ref), or a fold of a scheme that is not an Online Scheme. So the wrapper would reach `prior(pe, X)` unresolved and get a `MethodError` that names the whole type. The refusal names the dotted path that [`online_wrapper_path`](@ref) finds, and the two exits. One exit is an Online Scheme, such as [`OnlineIndexWalkForward`](@ref). The other exit is the estimator without the wrapper. Each caller passes an optimiser, which is never a wrapper, so the path at this entry is never the empty path.

# Algorithm

 1. Find the path of the first wrapper in `est` with [`online_wrapper_path`](@ref).
 2. Refuse `est` when the path is not `nothing`, and name the path and the entry function.

# Arguments

  - `est`: The estimator handed to the entry function.
  - `entry`: The name of the entry function, as the message reads it.

# Validation

  - No field in the tree of `est` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`online_wrapper_path`](@ref)
  - [`assert_online_entry`](@ref)
  - [`optimise`](@ref)
  - [`fold_loop`](@ref)
"""
function assert_batch_entry(est, entry::AbstractString)
    path = online_wrapper_path(est)
    @argcheck(isnothing(path),
              ArgumentError("`$(typeof(est).name.name)` enters $(entry) holding an `Online` at `$(path)`, and a batch fit cannot resolve it: `Online` declares the sample buffer the fold loop's online arm seeds at its warm-up and folds the wrapped estimator's rows into, and nothing else seeds one, so the wrapper would reach the batch verb unresolved. Run the estimator through an Online Scheme — `OnlineIndexWalkForward`, `OnlineDateWalkForward` or `OnlineHindsightSplit` — or set `$(path)` to the estimator it wraps and let the batch fit refit it from its rows."))
    return nothing
end
"""
    supports_partial_fit(est) -> Bool

Answers whether [`partial_fit!`](@ref) folds this estimator.

An outer estimator that carries the observations asks each of its members. This is the rule for an estimator with mixed members. The outer estimator folds each member that folds, and it runs the batch verb over its own rows for each member that does not. So a caller writes the same estimator as in batch, with no wrapper at the call site, and no member carries a second copy of the sample.

The default is the buffering route. An estimator folds when it carries a [`SampleBufferState`](@ref), which [`Online`](@ref) seeds. A family with an exact fold of its own adds a method that returns `true`, beside that fold, under the same type bound without the `cache` parameter. An estimator of that family folds exactly when its bound matches, and by buffering when a wrapper seeded a buffer. A configuration that a family refuses adds no method, and it takes the default. The `SemiMoment` arms are such a configuration, because their clip moves when the mean moves. For them, the default is `true` only when a wrapper gave the estimator a buffer.

The predicate reads a type and a field, never a method table. A read of the dispatch would count a refusal as a fold, because a refusal is a method too, and the members that refuse are the members that must buffer.

# Arguments

  - `est`: The estimator to ask about.

# Returns

  - `folds::Bool`: `true` when [`partial_fit!`](@ref) folds this estimator.

# Related

  - [`partial_fit!`](@ref)
  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
"""
function supports_partial_fit(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator})
    return hasfield(typeof(est), :cache) && isa(getfield(est, :cache), SampleBufferState)
end
function supports_partial_fit(::Nothing)
    # A member that the outer estimator does not hold folds vacuously: there is nothing to
    # fold and nothing to refit, so the call with no data skips it either way.
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the empty state an [`Online`](@ref) seeds into the estimator it wraps.

One sample buffer works for each estimator that refits, so the seed needs no knowledge of the type. A prior whose batch verb reads a factor matrix beside the returns records the factor rows in the same buffer, and [`needs_factor_returns`](@ref) decides whether its fold receives them. An estimator whose state is not a sample buffer has a method of its own that returns the state that its call with no data reads. A prior-less optimiser head is such an estimator, because its state is a fold context. Nothing else about the wrapper changes.

The function takes the estimator and not its type, so a family that sizes its seed from a field can read that field.

# Arguments

  - `est`: The estimator the wrapper wraps, read for its type.
  - `max_history`: The wrapper's cap, carried into the state.

# Returns

  - `state::AbstractPartialFitState`: The empty state to seed, carrying the cap and no observations.

# Related

  - [`Online`](@ref)
  - [`SampleBufferState`](@ref)
  - [`update_online_estimator`](@ref)
"""
function online_state_seed(::Union{AbstractEstimator, StatsBase.CovarianceEstimator},
                           max_history::Option{<:Integer})
    return SampleBufferState(; max_history = max_history)
end
function update_online_estimator(o::Online)
    est = rebuild_estimator(o.est, (; cache = online_state_seed(o.est, o.max_history)))
    return update_online_estimator(est)
end
function update_online_estimator(::Online{<:CrossValidationEstimator})
    return throw(ArgumentError("`Online` on a scheme is an Online Scheme, and it reached the warm-up in an estimator slot: a scheme seeds no sample buffer, because it is the `cv` argument of `cross_val_predict`, not a field of the estimator that runs through it."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Joins the name of a field, or the bracketed index of an element, to the path that a walk found below it.

A path that starts with a bracket follows the name with no dot, so a walk names a member of a vector field as `"factors[2]"` and not as `"factors.[2]"`. An empty path answers the name alone, so a walk whose leaf answers `""` names the field that holds the leaf, such as `"tr"`. Every other path follows the name after a dot.

# Arguments

  - `head`: The field name, or the bracketed index of an element, such as `"[2]"`.
  - `path`: The path below `head`.

# Returns

  - `path::String`: The joined path.

# Related

  - [`seed_window_path`](@ref)
  - [`online_entry_state`](@ref)
  - [`element_path`](@ref)
"""
function tree_path(head::Union{Symbol, AbstractString}, path::AbstractString)
    return if isempty(path) || startswith(path, '[')
        string(head, path)
    else
        string(head, ".", path)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Walks the elements of a vector in order with `walk`, and answers the path of the first element that gives one, or `nothing`.

The walks of an estimator tree that must see the estimators a vector holds call this function: [`seed_window_path`](@ref), [`online_entry_state`](@ref) and [`returns_tracking_path`](@ref). The answer starts with the index of the element in brackets, as [`tree_path`](@ref) joins it.

# Arguments

  - `walk`: The walk, a function that answers a path or `nothing` for one element.
  - `v`: The vector.

# Returns

  - `path::Option{<:String}`: The path of the first element that gives one, or `nothing`.

# Related

  - [`seed_window_path`](@ref)
  - [`online_entry_state`](@ref)
  - [`tree_path`](@ref)
"""
function element_path(walk, v::AbstractVector)
    i = findfirst(x -> !isnothing(walk(x)), v)
    return isnothing(i) ? nothing : tree_path(string("[", i, "]"), walk(v[i]))
end
function seed_window_path(v::AbstractVector)
    return element_path(seed_window_path, v)
end
function seed_window_path(p::Pair)
    return seed_window_path(last(p))
end
function online_entry_state(v::AbstractVector)
    return element_path(online_entry_state, v)
end
function online_entry_state(p::Pair)
    return online_entry_state(last(p))
end

"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Allocation Set of an online portfolio selection head, the set that contains each allocation of the recursion.

The head is [`OnlinePortfolioSelection`](@ref), and it holds the set in its `set` field. The budget is one on each set, so `⟨w, x⟩` is the wealth factor that the formula of each rule assumes. Cash is an asset with price relative one, and the rule allocates it like any other asset.

# Interfaces

In order to implement a new set kind, subtype `AbstractAllocationSet` with its constraint objects as part of the struct, and implement:

  - `resolve_allocation_set(set::AbstractAllocationSet, N::Integer, strict::Bool, datatype::DataType) -> AbstractAllocationSet`: The set with every estimator resolved to a value over the `N` assets, which the projection then reads.
  - `project(proj::AbstractProjectionGeometry, set::AbstractAllocationSet, q::AbstractVector, w::AbstractVector) -> AbstractVector`: The projection onto the resolved set, for every geometry the set admits.
  - `rows_needed(set::AbstractAllocationSet) -> Union{Nothing, Integer}`: The number of rows the set's constraints read at a step, `0` for a set that reads none.

## Arguments

  - `set`: The set.
  - `N`: The number of assets of the universe the set is resolved over.
  - `strict`: Whether an unknown name in a set is an error.
  - `datatype`: The element type a scalar bound is expanded in.

## Returns

  - `set::AbstractAllocationSet`: The resolved set.

# Related

  - [`BoundedAllocationSet`](@ref)
  - [`project`](@ref)
  - [`AbstractProjectionGeometry`](@ref)
"""
abstract type AbstractAllocationSet <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for an Allocation Set whose projection is a programme, a bare JuMP model that holds the constraints of the set.

The programme lets the set own a risk constraint that the shared risk-measure builders write, as a [`RiskConstraintOwner`](@ref) beside the JuMP optimisers.

# Interfaces

In order to implement a new programme set, subtype `AbstractProgrammeAllocationSet`, implement everything [`AbstractAllocationSet`](@ref) asks for, and implement:

  - `risk_constraint_solver(set::AbstractProgrammeAllocationSet)`: The solver a Deferred Quantity of the set's risk measure is resolved against.

# Related

  - [`AbstractAllocationSet`](@ref)
  - [`RiskConstraintOwner`](@ref)
  - [`risk_constraint_solver`](@ref)
"""
abstract type AbstractProgrammeAllocationSet <: AbstractAllocationSet end

export Online, RollingWindow, SeedWindow, BatchChoice, PinnedChoice
public AbstractWindowRule, AbstractChoiceRule
public AbstractProgrammeAllocationSet, risk_constraint_solver, supports_partial_fit
