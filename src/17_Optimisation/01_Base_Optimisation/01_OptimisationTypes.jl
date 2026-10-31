"""
    const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}

Alias for a vector of portfolio optimisation estimators.

A method that takes many optimisers at one time dispatches on it.

# Related

  - [`AbstractOptimisationEstimator`](@ref)
"""
const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the configurations that an optimiser holds.

[`JuMPOptimiser`](@ref) and [`HierarchicalOptimiser`](@ref) are two of them. A configuration states the optimisation problem. It is not an optimiser, so [`optimise`](@ref) does not take it alone.

# Interfaces

A subtype gets from this supertype the methods that resolve the schedules in its fields: `is_time_dependent`, `update_time_dependent_estimator`, `reset_time_dependent_estimator` and `assert_time_dependent_fold_count`. Each of them reads the fields through `time_dependent_fields`. A subtype can implement one method.

## `time_dependent_field_defaults`

  - `time_dependent_field_defaults(opt::MyConfiguration) -> NamedTuple`: The static default of each field that can hold a [`TimeDependent`](@ref), for the fields whose default is not `nothing`. A required field has the value [`NoDefault`](@ref), which states that a schedule in that field must carry its own `default`.

### Arguments

  - `opt`: The concrete subtype instance.

### Returns

  - `defaults::NamedTuple`: The value of each listed field outside every fold loop. The fallback method returns an empty tuple, which gives each scheduled field the value `nothing` outside every fold loop.

# Related

  - [`AbstractOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
"""
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that select a branch of a portfolio optimiser.

# Interfaces

A subtype is a tag that an optimiser dispatches on, so it declares no method of its own. To add a branch, subtype `OptimisationAlgorithm`, and add the methods of the optimiser that dispatch on the new tag.

# Related

  - [`AbstractAlgorithm`](@ref)
"""
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the results of a portfolio optimisation.

All concrete optimisation results must subtype `OptimisationResult`.

# Interfaces

A subtype declares no method, but [`optimise`](@ref) and [`factory`](@ref) read three of its properties. A subtype holds them as fields, or forwards them from an embedded core, as the JuMP results and the hierarchical results do.

  - `w`: The portfolio weights.
  - `retcode`: An [`OptimisationReturnCode`](@ref). [`optimise`](@ref) runs the fallback chain until it reads an [`OptimisationSuccess`](@ref).
  - `fb`: The record of the fallbacks that ran. It must be the last field of the struct, because [`factory`](@ref) rebuilds the result with a new last field.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptimisationReturnCode`](@ref)
"""
abstract type OptimisationResult <: AbstractResult end
"""
    result_investable_mask(res::OptimisationResult) -> Option{BitVector}

Returns the Investable Mask that an optimisation result was reduced to.

An optimisation reduces its universe to the Investable Mask at its entry, and it expands the solved weights back to the universe of the caller. So `res.w` spans the full universe, and the mask records which assets the optimisation traded. A fold reads the mask to view its test window before it scores the weights.

The families hold the mask in different places. A JuMP result holds it in its processed attributes, a hierarchical result holds it in the core it embeds, and a family that derives no mask has none. So this function dispatches on the type of the result. A family that gets a mask must add its own method here. A result that forwards its properties to a core must also add a method, because a forwarded `res.imsk` does not change the dispatch.

# Arguments

  - `res::OptimisationResult`: The optimisation result.

# Returns

  - `imsk::Option{BitVector}`: The Investable Mask. The fallback method returns `nothing`, which states that the optimisation did not reduce its universe.

# Related

  - [`investable_mask`](@ref)
  - [`investable_reduction`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
"""
function result_investable_mask(::OptimisationResult)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the results of a continuous optimisation, which gives weights and not share counts.

# Interfaces

The family adds no method to [`OptimisationResult`](@ref). It is the bound of the generic method [`factory`](@ref)`(res, fb)`, which rebuilds a result with a new fallback record. So the rule that `fb` is the last field is enforced at this level.

# Related

  - [`OptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`MeanRiskResult`](@ref)
"""
abstract type NonFiniteAllocationOptimisationResult <: OptimisationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the continuous optimisation results that hold no JuMP model.

The naive, hierarchical and meta-optimiser results are its members. On the JuMP side, [`RiskJuMPOptimisationResult`](@ref) holds the results that carry a risk measure, and [`NonRiskJuMPOptimisationResult`](@ref) holds the results that carry none. The hierarchical members have a family of their own one level down, [`HierarchicalOptimisationResult`](@ref).

# Interfaces

`NonJuMPOptimisationResult` adds no method to [`NonFiniteAllocationOptimisationResult`](@ref). It is a classification, so a method signature can state that a result holds no JuMP model.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`NonRiskJuMPOptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
"""
abstract type NonJuMPOptimisationResult <: NonFiniteAllocationOptimisationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the core field block that two hierarchical optimisation results embed.

The type is not in the tree of optimisation results, as [`BaseJuMPOptimisationResult`](@ref) is not on the JuMP side. [`optimise`](@ref) does not return a core, so a core must not match a method whose bound is the family of results, for example [`factory`](@ref)`(res::NonFiniteAllocationOptimisationResult, fb)`.

Its one subtype is [`HierarchicalResult`](@ref), which each of the two results embeds as `hr`.

# Interfaces

A subtype is a block of fields, not a result. It declares no method, and it must not get a method whose bound is the family of optimisation results.

# Related

  - [`HierarchicalResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`BaseJuMPOptimisationResult`](@ref)
"""
abstract type BaseHierarchicalOptimisationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the results of the estimators that hold a hierarchical optimiser in their field `opt`.

The field holds a [`HierarchicalOptimiser`](@ref). The members are exactly the results of [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref). [`NestedClustered`](@ref) holds no `HierarchicalOptimiser`, and its result is not a member.

The family is not called `ClusteringOptimisationResult`. [`ClusteringOptimisationEstimator`](@ref) has four subtypes, and one of them is [`NestedClustered`](@ref), so that name states a set that this family does not hold.

Two of the three members embed [`HierarchicalResult`](@ref) as `hr`. [`SchurComplementHierarchicalRiskParityResult`](@ref) holds its fields directly. So each member that embeds `hr` forwards its own properties, and this family forwards none.

# Interfaces

The family adds no method to [`NonJuMPOptimisationResult`](@ref). A member that embeds [`HierarchicalResult`](@ref) declares its own property forwarding, so that the properties `w` and `retcode` that [`OptimisationResult`](@ref) needs resolve through `hr`.

# Related

  - [`BaseHierarchicalOptimisationResult`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
  - [`SchurComplementHierarchicalRiskParityResult`](@ref)
"""
abstract type HierarchicalOptimisationResult <: NonJuMPOptimisationResult end
"""
    const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}

Alias for a vector of continuous optimisation results.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptE_Opt`](@ref)
"""
const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the return codes that state whether an optimisation succeeded.

# Interfaces

A subtype declares no method. It has one field, `res`, which holds the diagnostic text of a failure, or `nothing`. [`optimise`](@ref) reads the type of the code and nothing else. Only an [`OptimisationSuccess`](@ref) stops the fallback chain, so every other subtype is a failure.

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
    const OptRetCode_VecOptRetCode = Union{OptimisationReturnCode, VecOptRetCode}

Alias for one optimisation return code, or a vector of return codes.

A result that holds a population of weight vectors holds one return code for each member.

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`VecOptRetCode`](@ref)
"""
const OptRetCode_VecOptRetCode = Union{OptimisationReturnCode, VecOptRetCode}

"""
    set_retcode(res::NonFiniteAllocationOptimisationResult, retcode::OptRetCode_VecOptRetCode)

Rebuilds an optimisation result with a different return code, and every other field unchanged.

A cross-validation fold that drifts the weights of a population drops each member whose wealth is not positive. It drops a member when it sets the return code of that member to a failure. A result is immutable, so the drop rebuilds it. Each type has its own method, which names its constructor once, in place of a pass over the field list.

Only a result that can hold a population of weight vectors needs a method, because only such a result holds one return code for each member.

# Arguments

  - `res`: The optimisation result to rebuild.
  - `retcode`: The return code, or one return code for each member of the population.

# Validation

  - The type of `res` has a method of its own. The fallback method throws an `ArgumentError` that names the type.

# Returns

  - `res::NonFiniteAllocationOptimisationResult`: The result, with the new return code.

# Related

  - [`mark_ruined_members`](@ref)
  - [`OptRetCode_VecOptRetCode`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function set_retcode(res::NonFiniteAllocationOptimisationResult, ::OptRetCode_VecOptRetCode)
    return throw(ArgumentError("`set_retcode` has no method for `$(Base.typename(typeof(res)).wrapper)`, so a ruined population member of it cannot be dropped. A result that carries one return code per member needs a method of `set_retcode` that rebuilds it."))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the records of one solver attempt inside an optimisation.

The type is not in the tree of optimisation results, as [`BaseHierarchicalOptimisationResult`](@ref) is not. [`optimise`](@ref) does not return such a record. Its one subtype is [`JuMPOptimisationSolution`](@ref), the record of what a solver returned.

# Interfaces

A subtype declares no method. A result holds it, and an optimiser does not return it.

# Related

  - [`OptimisationResult`](@ref)
"""
abstract type OptimisationModelResult <: AbstractResult end
"""
    const OptE_Opt = Union{NonFiniteAllocationOptimisationEstimator,
                           NonFiniteAllocationOptimisationResult}

Alias for a continuous optimisation estimator, or a continuous optimisation result.

An estimator states an optimiser. A result is a precomputed answer, which a fold predicts with and does not fit. Cross-validation and the fallback field `fb` take either form.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
const OptE_Opt = Union{NonFiniteAllocationOptimisationEstimator,
                       NonFiniteAllocationOptimisationResult}
"""
    const FbChain = AbstractVector{<:Tuple{OptimisationEstimator, OptimisationResult}}

Alias for a fallback chain, the record of the attempts that failed before an optimisation gave its result.

Each entry is the `(estimator, result)` pair of one failed attempt, in the order in which the attempts ran. [`optimise`](@ref) adds one pair each time an attempt fails and its estimator names a fallback. When an attempt succeeds, or no fallback remains, `optimise` gives the vector to `factory(res, fb)`.

A result whose `fb` is a chain was given by a fallback. Then `fb[1][1]` is the estimator that the caller asked first, and `fb[end][2]` is the last failure before the result. A result whose `fb` is `nothing` was given by the estimator that the caller asked.

# Related

  - [`OptE_Opt_FbChain`](@ref)
  - [`FOptE_FOpt_FbChain`](@ref)
  - [`optimise`](@ref)
"""
const FbChain = AbstractVector{<:Tuple{OptimisationEstimator, OptimisationResult}}
"""
    const OptE_Opt_FbChain = Union{OptE_Opt, FbChain}

Alias for the values that the field `fb` of a continuous optimisation result can hold.

The field holds a fallback estimator or a precomputed result ([`OptE_Opt`](@ref)), or the fallback chain that gave the result ([`FbChain`](@ref)).

# Related

  - [`OptE_Opt`](@ref)
  - [`FbChain`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
const OptE_Opt_FbChain = Union{OptE_Opt, FbChain}
"""
    factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt_FbChain})

Rebuilds a continuous optimisation result with a new fallback record `fb`.

A concrete result can add its own method when a rebuild needs more than a new `fb`. [`optimise`](@ref) is the one caller, and it gives the [`FbChain`](@ref) of the attempts that failed.

# Algorithm

 1. Read every field of `res`, in the order of the struct.
 2. Call the constructor of the type of `res` with every field except the last, and with `fb` as the last field.

# Validation

  - `fb` is the last field of the type of `res`. Every concrete optimisation result obeys this rule, so the method checks nothing.

# Related

  - [`OptE_Opt_FbChain`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
function factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt_FbChain})
    flds = ntuple(i -> getfield(res, i), Val(fieldcount(typeof(res))))
    return (typeof(res).name.wrapper)(Base.front(flds)..., fb)
end
"""
    assert_special_nco_requirements(opt)

Checks that an optimiser meets the requirements of an inner optimiser of [`NestedClustered`](@ref).

The fallback method checks nothing. An estimator with a requirement adds a method, for example [`Stacking`](@ref).

# Arguments

  - `opt`: An optimisation estimator or result, a schedule of them, or a vector of them.

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

Returns `opt` unchanged.

This is the fallback [`factory`](@ref) of an optimisation estimator or result. An estimator whose parameters change at each step of an optimisation adds its own method. A precomputed result in a field of an estimator, for example in `fb`, reaches this method when [`factory`](@ref) gives the estimator the weights of the previous fold. A precomputed result does not change.

# Related

  - [`OptE_Opt`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::OptE_Opt, ::Any)
    return opt
end
"""
    needs_previous_weights(opt)

Returns `true` if the optimiser needs the weights of the previous period.

A fold loop that finds such an optimiser runs its folds in sequence, and gives each fold the weights of the fold before it. The fallback returns `false`. An optimiser that holds a turnover constraint, a tracking error constraint or another input that reads the previous weights adds a method that returns `true`.

# Arguments

  - `opt`: An optimisation estimator or result, a risk measure, a fee structure, or a vector of them.

# Returns

  - `flag::Bool`: `true` if the optimiser needs the previous weights.

# Related

  - [`is_time_dependent`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function needs_previous_weights(::OptE_Opt)
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

States that a portfolio optimisation succeeded.

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

States that a portfolio optimisation failed.

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
    field_frames(x, path::AbstractString, refit::Bool) -> ReturnsFrames

Return the frames that the fields of `x` add to the group of a head, for the frame walk.

A head and the configuration that it holds walk every field of their type with [`reader_frames`](@ref), at the path that [`join_path`](@ref) gives. A field that holds no reader adds nothing through the default method of `reader_frames`. An inner optimiser and a fallback have no method either, so the walk stops at them, because each one is a head with its own group. A new slot that holds a reader therefore joins the group with no edit here.

# Arguments

  - $(arg_dict[:frame_x])
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the fields of `x`, joined with [`merge_frames`](@ref).

# Related

  - [`group_frames`](@ref)
  - [`reader_frames`](@ref)
  - [`warn_raw_readers`](@ref)
"""
function field_frames(x, path::AbstractString, refit::Bool)
    return merge_frames(map(f -> reader_frames(getfield(x, f), join_path(path, f),
                                               refit)::ReturnsFrames,
                            fieldnames(typeof(x)))...)
end
"""
    reader_frames(v::AbstractVector{<:Number}, path::AbstractString, refit::Bool) -> ReturnsFrames
    reader_frames(o::Online, path::AbstractString, refit::Bool) -> ReturnsFrames
    refits_regression(o::Online) -> Bool
    returns_source(o::Online) -> Option{<:AbstractReturnsSource}

The frame walk through two values that a slot of a head can hold.

A vector of numbers, such as initial weights, holds no reader, so it adds nothing, and the walk does not visit each number. An [`Online`](@ref) wrapper of a prior answers for the prior that it wraps, because the wrapper only seeds the buffer of the prior. So the walk, the refit rule of a regression reader and the source of the prior see through it before the warm-up resolves it.

# Arguments

  - `v`: A vector of numbers.
  - `o`: An `Online` wrapper.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Related

  - [`field_frames`](@ref)
  - [`reader_frames`](@ref)
  - [`refits_regression`](@ref)
  - [`returns_source`](@ref)
"""
function reader_frames(::AbstractVector{<:Number}, ::AbstractString, ::Bool)
    return ReturnsFrames()
end
function reader_frames(o::Online, path::AbstractString, refit::Bool)
    return reader_frames(o.est, path, refit)
end
function refits_regression(o::Online)
    return refits_regression(o.est)
end
function returns_source(o::Online)
    return returns_source(o.est)
end
"""
    data_benchmark_path(x) -> Option{<:String}

Return the path of the first [`DataBenchmarkTracking`](@ref) in the tree of `x`, or `nothing` when `x` holds none.

The walk has the shape of [`returns_tracking_path`](@ref). It enters every field of an estimator, a result and an algorithm, and every element of a vector. A vector of numbers and the state of a partial fit hold no tracking algorithm, so the walk does not enter them.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`DataBenchmarkTracking`](@ref): return `""`.
 2. An estimator, a result or an algorithm: walk each field in order, giving `paths`, and join the name of the first field that gives a path to that path with [`tree_path`](@ref).
 3. A vector that is not a vector of numbers: walk each element with [`element_path`](@ref).
 4. A vector of numbers, the state of a partial fit, and any other value: return `nothing`.

# Arguments

  - `x`: A tracking algorithm, a tracking error, a risk measure, or a vector of them.

# Returns

  - `path::Option{<:String}`: The path of the first data benchmark below `x`, `""` when `x` is itself one, or `nothing`.

# Related

  - [`assert_data_benchmark_prior`](@ref)
  - [`returns_tracking_path`](@ref)
  - [`tree_path`](@ref)
"""
function data_benchmark_path(est::Union{AbstractEstimator, AbstractResult,
                                        AbstractAlgorithm})
    paths = map(f -> data_benchmark_path(getfield(est, f)), fieldnames(typeof(est)))
    i = findfirst(!isnothing, paths)
    return if isnothing(i)
        nothing
    else
        tree_path(fieldname(typeof(est), i), something(paths[i]))
    end
end
function data_benchmark_path(::DataBenchmarkTracking)
    return ""
end
function data_benchmark_path(::AbstractPartialFitState)
    return nothing
end
function data_benchmark_path(v::AbstractVector)
    return eltype(v) <: Number ? nothing : element_path(data_benchmark_path, v)
end
function data_benchmark_path(::Any)
    return nothing
end
"""
    assert_data_benchmark_prior(x, pe, field::Sym_Str, holder::Sym_Str,
                                pe_field::Sym_Str = :pe) -> Nothing

Refuse a [`DataBenchmarkTracking`](@ref) in `x` when the prior `pe` that its measure reads states `x_src = ExcessSource()`.

Every tracking form compares `pr.X * w` with its benchmark series. Under a prior in the Excess Frame `pr.X` holds `X - B`, so a data benchmark gives the series `(X - B) * w - B`, and the benchmark is subtracted two times. No use of the pair is correct, so the constructor of each holder of the tracking part and the prior refuses it. The configuration alone decides the refusal. A Prior Result in the place of the prior counts by the source that it records, see [`prior_returns_source`](@ref): a Result at `nothing` reads as raw and passes. A graph reader or a regression reader in the Excess Frame does not count, because no tracking form reads its matrix.

# Algorithm

 1. Return when `x` or `pe` is a [`TimeDependent`](@ref) schedule, or when [`reader_frames`](@ref) finds that `pe` does not state [`ExcessSource`](@ref).
 2. Find the path of the first data benchmark in `x` with [`data_benchmark_path`](@ref), giving `path`.
 3. Refuse when `path` is not `nothing`.

# Arguments

  - `x`: The field that can hold a data benchmark: a tracking error or a risk measure, or a vector of them.
  - `pe`: The prior that the measure reads: a prior estimator, a Prior Result, or an [`Online`](@ref) wrapper of a prior estimator. A [`TimeDependent`](@ref) schedule in `x` or in `pe` passes, because the constructor checks each of its entries, and the rebuild of each fold checks the pair of that fold.
  - `field`: The name of the field that holds `x`, for the error message.
  - `holder`: The name of the type that holds `x` and `pe`, for the error message.
  - `pe_field`: The path of `pe` from `holder`, for the error message.

# Validation

  - `pe` does not state `x_src = ExcessSource()`, or `x` holds no [`DataBenchmarkTracking`](@ref). An `ArgumentError` that names the field, the prior and the two remedies is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`DataBenchmarkTracking`](@ref)
  - [`ExcessSource`](@ref)
  - [`data_benchmark_path`](@ref)
  - [`bind_data_benchmark`](@ref)
"""
function assert_data_benchmark_prior(x, pe, field::Sym_Str, holder::Sym_Str,
                                     pe_field::Sym_Str = :pe)::Nothing
    # A schedule pairs its entries fold by fold. The substitution pass of the constructor
    # checks each entry against the static fields, and the rebuild of each fold checks the
    # pair of that fold.
    if isa(x, TimeDependent) ||
       isa(pe, TimeDependent) ||
       !reader_frames(pe, "", false).excess
        return nothing
    end
    path = data_benchmark_path(x)
    @argcheck(isnothing(path),
              ArgumentError("the `$(tree_path(field, something(path, "")))` of `$holder` tracks the benchmark `B` of the returns data with `DataBenchmarkTracking`, and its prior `$pe_field` states `x_src = ExcessSource()`. Every tracking form compares `pr.X * w` with its benchmark series, so the series would be `(X - B) * w - B`, and the benchmark would be subtracted two times. State `x_src = DataSource()` or `nothing` on `$pe_field` to track `X * w - B`, or replace the data benchmark with a plain risk measure on the excess prior. The two models differ: for a vector `B` the plain measure reads `X * w - B * sum(w)`, which equals `X * w - B` only when `sum(w) == 1`."))
    return nothing
end

export OptimisationSuccess, OptimisationFailure
public BaseOptimisationEstimator, OptimisationAlgorithm, OptimisationResult,
       NonFiniteAllocationOptimisationResult, NonJuMPOptimisationResult,
       BaseHierarchicalOptimisationResult, HierarchicalOptimisationResult,
       OptimisationReturnCode, OptimisationModelResult, needs_previous_weights
