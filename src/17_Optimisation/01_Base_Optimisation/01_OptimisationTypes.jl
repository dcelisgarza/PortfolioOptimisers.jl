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

A subtype gets from this supertype the methods that resolve the schedules in its fields: [`is_time_dependent`](@ref), [`update_time_dependent_estimator`](@ref), [`reset_time_dependent_estimator`](@ref) and [`assert_time_dependent_fold_count`](@ref). Each of them reads the fields through [`time_dependent_fields`](@ref). A subtype can implement one method.

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
    const OptRetCode_VecOptRetCode = Union{<:OptimisationReturnCode, <:VecOptRetCode}

Alias for one optimisation return code, or a vector of return codes.

A result that holds a population of weight vectors holds one return code for each member.

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`VecOptRetCode`](@ref)
"""
const OptRetCode_VecOptRetCode = Union{<:OptimisationReturnCode, <:VecOptRetCode}

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
    const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                           <:NonFiniteAllocationOptimisationResult}

Alias for a continuous optimisation estimator, or a continuous optimisation result.

An estimator states an optimiser. A result is a precomputed answer, which a fold predicts with and does not fit. Cross-validation and the fallback field `fb` take either form.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`VecOptE_Opt`](@ref)
"""
const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                       <:NonFiniteAllocationOptimisationResult}
"""
    const FbChain = AbstractVector{<:Tuple{<:OptimisationEstimator, <:OptimisationResult}}

Alias for a fallback chain, the record of the attempts that failed before an optimisation gave its result.

Each entry is the `(estimator, result)` pair of one failed attempt, in the order in which the attempts ran. [`optimise`](@ref) adds one pair each time an attempt fails and its estimator names a fallback. When an attempt succeeds, or no fallback remains, `optimise` gives the vector to `factory(res, fb)`.

A result whose `fb` is a chain was given by a fallback. Then `fb[1][1]` is the estimator that the caller asked first, and `fb[end][2]` is the last failure before the result. A result whose `fb` is `nothing` was given by the estimator that the caller asked.

# Related

  - [`OptE_Opt_FbChain`](@ref)
  - [`FOptE_FOpt_FbChain`](@ref)
  - [`optimise`](@ref)
"""
const FbChain = AbstractVector{<:Tuple{<:OptimisationEstimator, <:OptimisationResult}}
"""
    const OptE_Opt_FbChain = Union{<:OptE_Opt, <:FbChain}

Alias for the values that the field `fb` of a continuous optimisation result can hold.

The field holds a fallback estimator or a precomputed result ([`OptE_Opt`](@ref)), or the fallback chain that gave the result ([`FbChain`](@ref)).

# Related

  - [`OptE_Opt`](@ref)
  - [`FbChain`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
const OptE_Opt_FbChain = Union{<:OptE_Opt, <:FbChain}
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

export OptimisationSuccess, OptimisationFailure
public BaseOptimisationEstimator, OptimisationAlgorithm, OptimisationResult,
       NonFiniteAllocationOptimisationResult, NonJuMPOptimisationResult,
       BaseHierarchicalOptimisationResult, HierarchicalOptimisationResult,
       OptimisationReturnCode, OptimisationModelResult, needs_previous_weights
