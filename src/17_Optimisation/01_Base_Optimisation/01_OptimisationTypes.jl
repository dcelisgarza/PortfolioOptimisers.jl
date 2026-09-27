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

# Interfaces

A subtype gains the time-dependent host methods from this supertype: [`is_time_dependent`](@ref), [`update_time_dependent_estimator`](@ref), [`reset_time_dependent_estimator`](@ref) and [`assert_time_dependent_fold_count`](@ref) all scan its fields generically, through [`time_dependent_fields`](@ref). One method is worth implementing:

## `time_dependent_field_defaults`

  - `time_dependent_field_defaults(opt::MyConfiguration) -> NamedTuple`: The static default of each field that may hold a [`TimeDependent`](@ref), for those whose default is not `nothing`. A *required* field is listed with [`NoDefault`](@ref), which declares that a schedule there must carry its own `default`.

### Arguments

  - `opt`: The concrete subtype instance.

### Returns

  - `defaults::NamedTuple`: The fold-less value of each listed field. The fallback method returns an empty tuple, which gives every scheduled field the fold-less value `nothing`.

# Related

  - [`AbstractOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
"""
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
function reset_time_dependent_estimator(opt::OptimisationEstimator)
    return opt
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation algorithms used by portfolio optimisers.

# Interfaces

A subtype is a tag that an optimiser dispatches on, so it declares no method of its own. To add a behaviour, subtype `OptimisationAlgorithm` and add the methods of the consuming optimiser that are specialised on it.

# Related

  - [`AbstractAlgorithm`](@ref)
"""
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation result types.

All concrete optimisation result types should subtype `OptimisationResult`.

# Interfaces

A subtype declares no method, but [`optimise`](@ref) and [`factory`](@ref) read three properties of it. A subtype exposes them either as its own fields or by forwarding from an embedded core, as the JuMP and hierarchical leaves do:

  - `w`: The portfolio weights.
  - `retcode`: An [`OptimisationReturnCode`](@ref). [`optimise`](@ref) walks the fallback chain until it reads an [`OptimisationSuccess`](@ref).
  - `fb`: The record of the fallbacks that ran. It must be the **last field** of the struct, because [`factory`](@ref) rebuilds the result by replacing its trailing field.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptimisationReturnCode`](@ref)
"""
abstract type OptimisationResult <: AbstractResult end
function reset_time_dependent_estimator(opt::OptimisationResult)
    return opt
end
"""
    result_investable_mask(res::OptimisationResult) -> Option{BitVector}

Read the Investable Mask an optimisation result reduced on.

An optimisation reduces to the Investable Mask at its entry and expands the solved weights back to the caller's universe, so the result's own `w` is on the **full** universe and the mask is the record of which assets the optimisation traded. A fold reads that record to view its test window before it scores the weights.

The mask is read through this verb rather than off a field, because the families carry it in different places: a JuMP result holds it on its processed attribute bundle, a hierarchical leaf holds it on the core it wraps, and a family that derives no mask answers `nothing`. **A family that gains a mask must add its own method here**, and so must a leaf that forwards its properties into a core, because the verb dispatches on the type and a forwarded `res.imsk` never reaches it. `test/test_54_held_gap_filter.jl` censuses the concrete results and fails when one carries a mask this verb cannot read, as a field or as a forwarded property, so the omission cannot be silent.

# Arguments

  - `res::OptimisationResult`: Fitted optimisation result.

# Returns

  - `imsk::Option{BitVector}`: The Investable Mask, or `nothing` when the optimisation reduced on nothing.

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

Abstract supertype for continuous (non-integer allocation) optimisation results.

# Interfaces

The family adds no method to [`OptimisationResult`](@ref), but it is the bound of the generic [`factory`](@ref)`(res, fb)` that rebuilds a result with a new fallback record, which is why the trailing `fb` field is required here rather than one level up.

# Related

  - [`OptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`MeanRiskResult`](@ref)
"""
abstract type NonFiniteAllocationOptimisationResult <: OptimisationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-JuMP continuous optimisation results.

Groups the results that do not carry a JuMP model (naive, clustering, and meta-optimiser results). Mirrors the JuMP/non-JuMP split on the result side. The JuMP side is itself two halves, [`RiskJuMPOptimisationResult`](@ref) for the results that carry a risk measure and [`NonRiskJuMPOptimisationResult`](@ref) for those that carry none.

The hierarchical members are grouped one level further down, under [`HierarchicalOptimisationResult`](@ref).

# Interfaces

`NonJuMPOptimisationResult` adds no method to [`NonFiniteAllocationOptimisationResult`](@ref). It is a classification, and it is what lets a method state "carries no JuMP model" in a signature.

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

Abstract supertype for the **core** field block shared by hierarchical optimisation results.

Sits off the optimisation-result tree on purpose, exactly as [`BaseJuMPOptimisationResult`](@ref) does on the JuMP side. A core is not a thing `optimise` returns, so it must not satisfy methods bounded on the result family — [`factory`](@ref)`(res::NonFiniteAllocationOptimisationResult, fb)` included.

Its one subtype is [`HierarchicalResult`](@ref), embedded as `hr` by each leaf.

# Interfaces

A subtype is a field block, not a result. It declares no method, and it must **not** be given one that is bounded on the optimisation-result family, because the core is never what [`optimise`](@ref) returns.

# Related

  - [`HierarchicalResult`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`BaseJuMPOptimisationResult`](@ref)
"""
abstract type BaseHierarchicalOptimisationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the results of the estimators that embed a hierarchical optimiser.

The optimiser they embed is a [`HierarchicalOptimiser`](@ref), held in their estimator's `opt` field.

The membership rule is exact: [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) each hold an `opt::HierarchicalOptimiser`. [`NestedClustered`](@ref) does not, and its result is **not** in this family.

The family is deliberately **not** called `ClusteringOptimisationResult`: [`ClusteringOptimisationEstimator`](@ref) has four subtypes and the fourth is [`NestedClustered`](@ref), so that name would claim a set this type does not hold.

Two of the three members embed [`HierarchicalResult`](@ref) as `hr`; [`SchurComplementHierarchicalRiskParityResult`](@ref) keeps a flat field block, which is why the property forwarding lives on the leaves rather than here.

# Interfaces

The family adds no method to [`NonJuMPOptimisationResult`](@ref). Because the field block is not shared, a leaf that embeds [`HierarchicalResult`](@ref) declares its own property forwarding, so that the `w` and `retcode` properties [`OptimisationResult`](@ref) requires resolve through `hr`.

# Related

  - [`BaseHierarchicalOptimisationResult`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
  - [`SchurComplementHierarchicalRiskParityResult`](@ref)
"""
abstract type HierarchicalOptimisationResult <: NonJuMPOptimisationResult end
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

# Interfaces

A subtype declares no method. It carries one field, `res`, which holds the diagnostic text of a failure or `nothing`. [`optimise`](@ref) tests the code by type alone: only an [`OptimisationSuccess`](@ref) ends the fallback chain, so any other subtype is read as a failure.

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
    set_retcode(res::NonFiniteAllocationOptimisationResult, retcode::OptRetCode_VecOptRetCode)

Rebuild an optimisation result with a different return code, and every other member unchanged.

A cross-validation fold that drifts a population's weights drops the members whose wealth is not positive, and it drops them by failing their entry of the result's return code. A result is an immutable record, so the drop rebuilds it. The rebuild is a per-type method that writes the constructor name once, rather than a reflection pass over the field list.

Only a result that can carry a population of weight vectors needs a method here, because only such a result can hold one return code per member. A result that reaches the fallback raises, and the message names the type that is missing its method.

# Arguments

  - `res`: Optimisation result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Validation

  - The type of `res` declares a method of its own, else an `ArgumentError` is raised.

# Returns

  - `NonFiniteAllocationOptimisationResult`: The result, with the new return code.

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

Abstract supertype for intermediate optimisation model results.

Sits off the optimisation-result tree, like [`BaseHierarchicalOptimisationResult`](@ref) does: an intermediate record is not a thing [`optimise`](@ref) returns. Its one subtype is [`JuMPOptimisationSolution`](@ref), the record of what a solver returned.

# Interfaces

A subtype is a record of one solver attempt, and it declares no method. It is held by a result rather than returned by an optimiser.

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
    const FbChain = AbstractVector{<:Tuple{<:OptimisationEstimator, <:OptimisationResult}}

Alias for a fallback chain: the `(estimator, result)` pair of every attempt that failed before the result that carries it, in the order the attempts ran.

[`optimise`](@ref) pushes one pair each time an attempt fails and its estimator names a fallback, and hands the vector to `factory(res, fb)` once an attempt succeeds or the chain runs out. A result whose `fb` is a chain was therefore answered by a fallback: `fb[1][1]` is the estimator that was asked first, and `fb[end][2]` is the last failure before the answer. A result whose `fb` is `nothing` was answered by the estimator it was asked of.

# Related

  - [`OptE_Opt_FbChain`](@ref)
  - [`FOptE_FOpt_FbChain`](@ref)
  - [`optimise`](@ref)
"""
const FbChain = AbstractVector{<:Tuple{<:OptimisationEstimator, <:OptimisationResult}}
"""
    const OptE_Opt_FbChain = Union{<:OptE_Opt, <:FbChain}

Alias for what the `fb` field of a continuous optimisation result admits: a fallback estimator or precomputed result ([`OptE_Opt`](@ref)), or the fallback chain that answered the result ([`FbChain`](@ref)).

# Related

  - [`OptE_Opt`](@ref)
  - [`FbChain`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
const OptE_Opt_FbChain = Union{<:OptE_Opt, <:FbChain}
"""
    factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt_FbChain})

Rebuild a continuous optimisation result with an updated fallback record `fb`.

Every optimisation result carries `fb` as its last field, so the generic rebuild copies all fields unchanged except the trailing `fb`. Concrete result types may override this method when rebuilding requires more than swapping `fb`. [`optimise`](@ref) is the one caller, and it hands in the [`FbChain`](@ref) it walked.

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

export OptimisationSuccess, OptimisationFailure
public BaseOptimisationEstimator, OptimisationAlgorithm, OptimisationResult,
       NonFiniteAllocationOptimisationResult, NonJuMPOptimisationResult,
       BaseHierarchicalOptimisationResult, HierarchicalOptimisationResult,
       OptimisationReturnCode, OptimisationModelResult, needs_previous_weights
