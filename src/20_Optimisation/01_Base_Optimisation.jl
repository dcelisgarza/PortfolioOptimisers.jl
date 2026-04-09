abstract type AbstractOptimisationEstimator <: AbstractEstimator end
const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base portfolio optimisation estimators.

`BaseOptimisationEstimator` is the parent for all internal optimiser components that configure the optimisation problem but are not directly invokable as top-level optimisers.

# Related Types

  - [`AbstractOptimisationEstimator`](@ref)
  - [`OptimisationEstimator`](@ref)
"""
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce portfolio weights.

Subtype `OptimisationEstimator` to implement concrete portfolio optimisers. All optimisers that can be invoked with `optimise` should subtype this.

# Related Types

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`AbstractOptimisationEstimator`](@ref)
"""
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation estimators that produce continuous (non-integer) portfolio weights.

# Related Types

  - [`OptimisationEstimator`](@ref)
  - [`NaiveOptimisationEstimator`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
"""
abstract type NonFiniteAllocationOptimisationEstimator <: OptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation algorithms used by portfolio optimisers.

# Related Types

  - [`AbstractAlgorithm`](@ref)
"""
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio optimisation result types.

All concrete optimisation result types should subtype `OptimisationResult`.

# Related Types

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`OptimisationReturnCode`](@ref)
"""
abstract type OptimisationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for continuous (non-integer allocation) optimisation results.

# Related Types

  - [`OptimisationResult`](@ref)
  - [`NaiveOptimisationResult`](@ref)
  - [`HierarchicalResult`](@ref)
  - [`MeanRiskResult`](@ref)
"""
abstract type NonFiniteAllocationOptimisationResult <: OptimisationResult end
const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation return codes.

Concrete subtypes indicate whether an optimisation succeeded or failed.

# Related Types

  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
abstract type OptimisationReturnCode <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for intermediate optimisation model results.

# Related Types

  - [`OptimisationResult`](@ref)
"""
abstract type OptimisationModelResult <: AbstractResult end
const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                       <:NonFiniteAllocationOptimisationResult}
function assert_special_nco_requirements(::OptE_Opt)
    return nothing
end
function factory(opt::OptE_Opt, ::Any)
    return opt
end
function needs_previous_weights(::OptE_Opt)
    return false
end
#! Start: Overload these for all estimators which can use time-dependent constraints.
function is_time_dependent(::OptE_Opt)
    return false
end
function update_time_dependent_estimator(opt::OptE_Opt, args...)
    return opt
end
#! End: Overload these for all estimators which can use time-dependent constraints.
const VecOptE_Opt = AbstractVector{<:OptE_Opt}
function factory(opt::VecOptE_Opt, args...)
    return [factory(opti, args...) for opti in opt]
end
function assert_special_nco_requirements(opt::VecOptE_Opt)
    return assert_special_nco_requirements.(opt)
end
function needs_previous_weights(opt::VecOptE_Opt)
    return any(needs_previous_weights.(opt))
end
function is_time_dependent(opt::VecOptE_Opt)
    return any(is_time_dependent.(opt))
end
function update_time_dependent_estimator(opt::VecOptE_Opt, args...)
    return [update_time_dependent_estimator(opti, args...) for opti in opt]
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based weight finaliser formulations.

Defines the interface for norm types used when adjusting portfolio weights to satisfy bounds via a JuMP model.

# Related Types

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
"""
struct RelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm (squared) of relative weight deviations when enforcing weight bounds.
"""
struct SquaredRelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L1 norm of absolute weight deviations when enforcing weight bounds.
"""
struct AbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm (squared) of absolute weight deviations when enforcing weight bounds.
"""
struct SquaredAbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for weight finaliser strategies.

A `WeightFinaliser` enforces weight bounds after the optimisation has produced unconstrained weights.

# Related Types

  - [`IterativeWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
abstract type WeightFinaliser <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Iteratively projects weights into the feasible region defined by weight bounds.

`IterativeWeightFinaliser` repeatedly clips and redistributes portfolio weights until they satisfy the given lower and upper bounds, or the maximum number of iterations `iter` is reached.

# Fields

  - `iter`: Maximum number of iterations.

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
    iter
    function IterativeWeightFinaliser(iter::Integer)
        @argcheck(iter > 0)
        return new{typeof(iter)}(iter)
    end
end
function IterativeWeightFinaliser(; iter::Integer = 100)
    return IterativeWeightFinaliser(iter)
end
"""
$(DocStringExtensions.TYPEDEF)

Uses a JuMP optimisation model to enforce weight bounds.

`JuMPWeightFinaliser` solves a small optimisation problem to find the closest feasible weights (in the sense of the chosen error formulation) that satisfy the given bounds. Falls back to [`IterativeWeightFinaliser`](@ref) if the JuMP model fails.

# Fields

  - `slv`: Solver or vector of solvers for the JuMP model.
  - `sc`: Scale factor applied to constraints.
  - `so`: Scale factor applied to the objective.
  - `alg`: Error formulation (L1/L2 relative or absolute).

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

# Related

  - [`WeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliserFormulation`](@ref)
"""
@concrete struct JuMPWeightFinaliser <: WeightFinaliser
    slv
    sc
    so
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
                             alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser())
    return JuMPWeightFinaliser(slv, sc, so, alg)
end
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

  - `res`: Optional result or message from the solver (default: `nothing`).

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`OptimisationFailure`](@ref)
"""
@concrete struct OptimisationSuccess <: OptimisationReturnCode
    res
end
function OptimisationSuccess(; res = nothing)
    return OptimisationSuccess(res)
end
"""
$(DocStringExtensions.TYPEDEF)

Indicates that a portfolio optimisation failed.

# Fields

  - `res`: Optional error message or diagnostic information (default: `nothing`).

# Related

  - [`OptimisationReturnCode`](@ref)
  - [`OptimisationSuccess`](@ref)
"""
@concrete struct OptimisationFailure <: OptimisationReturnCode
    res
end
function OptimisationFailure(; res = nothing)
    return OptimisationFailure(res)
end
function opt_view(opt::AbstractOptimisationEstimator, args...)
    return opt
end
function opt_view(opt::VecOptE, args...)
    return [opt_view(opti, args...) for opti in opt]
end
function optimise(or::OptimisationResult, args...; kwargs...)
    return or
end
function _optimise end
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
function assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)
    return nothing
end
function assert_external_optimiser(::NonFiniteAllocationOptimisationResult)
    return nothing
end
"""
fees takes precedence over res.fees if both are provided
"""
function calc_net_returns(res::NonFiniteAllocationOptimisationResult, X::MatNum,
                          fees::Option{<:Fees} = nothing)
    if isnothing(fees) && hasproperty(res, :fees)
        fees = res.fees
    end
    return calc_net_returns(res.w, X, fees)
end
function calc_net_returns(res::NonFiniteAllocationOptimisationResult, pr::Pr_RR,
                          fees::Option{<:Fees} = nothing)
    return calc_net_returns(res, pr.X, fees)
end
function expected_risk(r::AbstractBaseRiskMeasure, res::OptimisationResult, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    if isnothing(fees) && hasproperty(res, :fees)
        fees = res.fees
    end
    return expected_risk(r, res.w, X, fees; kwargs...)
end
function expected_risk(r::AbstractBaseRiskMeasure, res::OptimisationResult,
                       pr::Option{<:Pr_RR} = nothing, fees::Option{<:Fees} = nothing;
                       kwargs...)
    pr = if !isnothing(pr)
        pr
    elseif isnothing(pr) && hasproperty(res, :pr)
        res.pr
    elseif isnothing(pr) && hasproperty(res, :opt) && hasproperty(res.opt, :pr)
        res.opt.pr
    else
        throw(ArgumentError("`res` is a $(Base.typename(typeof(res)).wrapper), which does not have a valid `res.pr` or `res.opt.pr` field, please provide `pr` or a data matrix as an argument"))
    end
    return expected_risk(r, res, pr.X, fees; kwargs...)
end
function needs_previous_weights(::Nothing)
    return false
end

export optimise, OptimisationSuccess, OptimisationFailure, IterativeWeightFinaliser,
       RelativeErrorWeightFinaliser, SquaredRelativeErrorWeightFinaliser,
       AbsoluteErrorWeightFinaliser, SquaredAbsoluteErrorWeightFinaliser,
       JuMPWeightFinaliser
