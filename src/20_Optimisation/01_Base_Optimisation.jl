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
    is_time_dependent(opt)

Return `true` if the optimiser has time-dependent constraints or objectives.

The default returns `false`. Overridden for estimators that must be updated between periods (e.g. when constraints depend on the current time step).

# Arguments

  - `opt`: Optimisation estimator, result, or vector thereof.

# Returns

  - `Bool`: `true` if the estimator is time-dependent.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function is_time_dependent(::OptE_Opt)
    return false
end
"""
    update_time_dependent_estimator(opt, args...)

Update the estimator for the current time period.

The default returns the estimator unchanged. Overridden for estimators that need to be updated between periods (e.g. sliding window constraints, time-varying parameters).

# Arguments

  - `opt`: Optimisation estimator or result.
  - `args...`: Additional arguments (e.g. current period index, returns data).

# Returns

  - Updated estimator.

# Related

  - [`is_time_dependent`](@ref)
  - [`path_fit_and_predict`](@ref)
"""
function update_time_dependent_estimator(opt::OptE_Opt, args...)
    return opt
end
#! End: Overload these for all estimators which can use time-dependent constraints.
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
function update_time_dependent_estimator(opt::VecOptE_Opt, args...)
    return [update_time_dependent_estimator(opti, args...) for opti in opt]
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
        @argcheck(iter > 0)
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
julia> JuMPWeightFinaliser(; slv = Solver())
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
            @argcheck(!isempty(slv))
        end
        @argcheck(sc > zero(sc))
        @argcheck(so > zero(so))
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
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
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
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; sc * (w ⊘ wi .- one(eltype(wi)))] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::AbsoluteErrorWeightFinaliser, wi::VecNum)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredAbsoluteErrorWeightFinaliser,
                                              wi::VecNum)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
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
        JuMP.value.(model[:w])
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
    opt_view(opt, i, args...)

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
function opt_view(opt::AbstractOptimisationEstimator, args...)
    return opt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply [`opt_view`](@ref) element-wise to a vector of optimisation estimators.

# Related

  - [`opt_view`](@ref)
  - [`VecOptE`](@ref)
"""
function opt_view(opt::VecOptE, args...)
    return [opt_view(opti, args...) for opti in opt]
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
function _extract_fees(res::OptimisationResult, fees::Option{<:Fees} = nothing)
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
    fees = _extract_fees(res, fees)
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
function _extract_pr(res::OptimisationResult, pr::Option{<:Pr_RR} = nothing)
    return if !isnothing(pr)
        pr
    elseif hasproperty(res, :pr)
        res.pr
    elseif hasproperty(res, :pa) && hasproperty(res.pa, :pr)
        res.pa.pr
    else
        throw(ArgumentError("`$(nameof(typeof(res)))` has no `.pr` or `.pa.pr`; provide `pr` explicitly"))
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
    fees = _extract_fees(res, fees)
    return expected_risk(r, res.w, X, fees; kwargs...)
end
function expected_risk(r::AbstractBaseRiskMeasure, res::OptimisationResult,
                       pr::Option{<:Pr_RR} = nothing, fees::Option{<:Fees} = nothing;
                       kwargs...)
    pr = _extract_pr(res, pr)
    return expected_risk(r, res, pr.X, fees; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`.

`nothing` never requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(::Nothing)
    return false
end

export optimise, OptimisationSuccess, OptimisationFailure, IterativeWeightFinaliser,
       RelativeErrorWeightFinaliser, SquaredRelativeErrorWeightFinaliser,
       AbsoluteErrorWeightFinaliser, SquaredAbsoluteErrorWeightFinaliser,
       JuMPWeightFinaliser
