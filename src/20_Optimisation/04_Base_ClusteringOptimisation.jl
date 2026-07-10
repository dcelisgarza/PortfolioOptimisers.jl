"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base clustering optimisation estimators.

These are intermediate configuration types used in hierarchical/clustering optimisation pipelines.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`HierarchicalOptimiser`](@ref)
"""
abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for clustering-based portfolio optimisation estimators.

Clustering optimisation estimators use asset clustering to decompose the portfolio optimisation problem. Subtypes include HRP, HERC, and SCHRP.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
abstract type ClusteringOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the inner optimiser or fallback carries time-dependent constraints.
"""
function is_time_dependent(opt::ClusteringOptimisationEstimator)
    return is_time_dependent(opt.opt) || is_time_dependent(opt.fb)
end
function assert_time_dependent_fold_count(opt::ClusteringOptimisationEstimator,
                                          n::Integer)::Nothing
    assert_time_dependent_fold_count(opt.opt, n)
    assert_time_dependent_fold_count(opt.fb, n)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for hierarchical (clustering-based) portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
@concrete struct HierarchicalResult <: NonJuMPOptimisationResult
    """
    $(field_dict[:oe])
    """
    oe
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:clr])
    """
    clr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:fees])
    """
    fees
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:pw])
    """
    w
    """
    $(field_dict[:fb])
    """
    fb
    function HierarchicalResult(oe::Type{<:OptimisationEstimator},
                                pr::Option{<:AbstractPriorResult},
                                clr::Option{<:AbstractClusteringResult},
                                wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                                retcode::OptimisationReturnCode, w::Option{<:VecNum},
                                fb::Option{<:OptE_Opt})
        return new{typeof(oe), typeof(pr), typeof(clr), typeof(wb), typeof(fees),
                   typeof(retcode), typeof(w), typeof(fb)}(oe, pr, clr, wb, fees, retcode,
                                                           w, fb)
    end
end
function HierarchicalResult(; oe::Type{<:OptimisationEstimator},
                            pr::Option{<:AbstractPriorResult},
                            clr::Option{<:AbstractClusteringResult},
                            wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                            retcode::OptimisationReturnCode, w::Option{<:VecNum},
                            fb::Option{<:OptE_Opt})::HierarchicalResult
    return HierarchicalResult(oe, pr, clr, wb, fees, retcode, w, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Base configuration for hierarchical clustering-based portfolio optimisers.

`HierarchicalOptimiser` combines a prior estimator, a clustering estimator, and weight bound/fee specifications to provide a reusable base configuration for hierarchical optimisers (HRP, HERC, SCHRP, etc.).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalOptimiser(;
        pe::PrE_Pr = EmpiricalPrior(),
        cle::HClE_HCl = ClustersEstimator(),
        slv::Option{<:Slv_VecSlv} = nothing,
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::Option{<:AssetSets} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        brt::Bool = false,
        cle_pr::Bool = true,
        strict::Bool = false
    ) -> HierarchicalOptimiser

Keywords correspond to the struct's fields. Fields typed [`TD_Option`](@ref) (`wb`, `fees`) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value; a cross-validation fold loop resolves it per fold, and a fold-less `optimise` runs with the field at its static default.

## Validation

  - If any field holds a [`TimeDependent`](@ref): every vector entry is test-substituted through this constructor so type compatibility errors surface immediately.

# Examples

```jldoctest
julia> HierarchicalOptimiser()
HierarchicalOptimiser
      pe ┼ EmpiricalPrior
         │        ce ┼ PortfolioOptimisersCovariance
         │           │   ce ┼ Covariance
         │           │      │    me ┼ SimpleExpectedReturns
         │           │      │       │   w ┴ nothing
         │           │      │    ce ┼ GeneralCovariance
         │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │           │      │       │    w ┴ nothing
         │           │      │   alg ┴ FullMoment()
         │           │   mp ┼ MatrixProcessing
         │           │      │     pdm ┼ Posdef
         │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │           │      │      dn ┼ nothing
         │           │      │      dt ┼ nothing
         │           │      │     alg ┼ nothing
         │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │        me ┼ SimpleExpectedReturns
         │           │   w ┴ nothing
         │   horizon ┴ nothing
     cle ┼ ClustersEstimator
         │    ce ┼ PortfolioOptimisersCovariance
         │       │   ce ┼ Covariance
         │       │      │    me ┼ SimpleExpectedReturns
         │       │      │       │   w ┴ nothing
         │       │      │    ce ┼ GeneralCovariance
         │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │       │      │       │    w ┴ nothing
         │       │      │   alg ┴ FullMoment()
         │       │   mp ┼ MatrixProcessing
         │       │      │     pdm ┼ Posdef
         │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │       │      │      dn ┼ nothing
         │       │      │      dt ┼ nothing
         │       │      │     alg ┼ nothing
         │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │    de ┼ Distance
         │       │   power ┼ nothing
         │       │     alg ┴ CanonicalDistance()
         │   alg ┼ HClustAlgorithm
         │       │   linkage ┴ Symbol: :ward
         │   onc ┼ OptimalNumberClusters
         │       │   max_k ┼ nothing
         │       │     alg ┼ SecondOrderDifference
         │       │         │   alg ┼ StandardisedValue
         │       │         │       │   mv ┼ MeanValue
         │       │         │       │      │   w ┴ nothing
         │       │         │       │   sv ┼ StdValue
         │       │         │       │      │           w ┼ nothing
         │       │         │       │      │   corrected ┴ Bool: true
     slv ┼ nothing
      wb ┼ WeightBounds
         │   lb ┼ Float64: 0.0
         │   ub ┴ Float64: 1.0
    fees ┼ nothing
    sets ┼ nothing
      wf ┼ IterativeWeightFinaliser
         │   iter ┴ Int64: 100
     brt ┼ Bool: false
  cle_pr ┼ Bool: true
  strict ┴ Bool: false
```

# Related

  - [`BaseClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
@propagatable @concrete struct HierarchicalOptimiser <: BaseClusteringOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    @vprop pe
    """
    $(field_dict[:cle])
    """
    cle
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:fees])
    """
    @fprop @vprop fees
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:cle_pr])
    """
    cle_pr
    """
    $(field_dict[:strict_opt])
    """
    strict
    function HierarchicalOptimiser(pe::PrE_Pr, cle::HClE_HCl, slv::Option{<:Slv_VecSlv},
                                   wb::TD_Option{<:WbE_Wb}, fees::TD_Option{<:FeesE_Fees},
                                   sets::Option{<:AssetSets}, wf::WeightFinaliser,
                                   brt::Bool, cle_pr::Bool, strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        assert_time_dependent_substitution(HierarchicalOptimiser,
                                           (; pe, cle, slv, wb, fees, sets, wf, brt, cle_pr,
                                            strict), (; wb = WeightBounds()))
        return new{typeof(pe), typeof(cle), typeof(slv), typeof(wb), typeof(fees),
                   typeof(sets), typeof(wf), typeof(brt), typeof(cle_pr), typeof(strict)}(pe,
                                                                                          cle,
                                                                                          slv,
                                                                                          wb,
                                                                                          fees,
                                                                                          sets,
                                                                                          wf,
                                                                                          brt,
                                                                                          cle_pr,
                                                                                          strict)
    end
end
function HierarchicalOptimiser(; pe::PrE_Pr = EmpiricalPrior(),
                               cle::HClE_HCl = ClustersEstimator(),
                               slv::Option{<:Slv_VecSlv} = nothing,
                               wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                               fees::TD_Option{<:FeesE_Fees} = nothing,
                               sets::Option{<:AssetSets} = nothing,
                               wf::WeightFinaliser = IterativeWeightFinaliser(),
                               brt::Bool = false, cle_pr::Bool = true,
                               strict::Bool = false)::HierarchicalOptimiser
    return HierarchicalOptimiser(pe, cle, slv, wb, fees, sets, wf, brt, cle_pr, strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`HierarchicalOptimiser`](@ref) requires previous portfolio weights (based on fee structure and time-dependent constraints).

# Related

  - [`needs_previous_weights`](@ref)
  - [`HierarchicalOptimiser`](@ref)
"""
function needs_previous_weights(opt::HierarchicalOptimiser)
    return needs_previous_weights(opt.fees) ||
           any(f -> needs_previous_weights(getfield(opt, f)), time_dependent_fields(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the time-dependent constraints of a [`HierarchicalOptimiser`](@ref) for the fold described by `ctx`.

Rebuilds the optimiser through its validated keyword constructor with each [`TimeDependent`](@ref)-valued field replaced by its resolved per-fold value, so the result is an ordinary static optimiser.

# Related

  - [`HierarchicalOptimiser`](@ref)
  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function update_time_dependent_estimator(opt::HierarchicalOptimiser,
                                         ctx::TimeDependentContext)
    return update_time_dependent_fields(opt, ctx)
end
function time_dependent_field_defaults(::HierarchicalOptimiser)::NamedTuple
    return (; wb = WeightBounds())
end
function reset_time_dependent_estimator(opt::HierarchicalOptimiser)
    return reset_time_dependent_fields(opt)
end
"""
    unitary_expected_risks(r, X, ...)

Compute the expected risk for unitary (equal-weight) portfolios within each cluster.

Returns a vector of risk values, one per cluster, where each risk is computed for the equal-weight portfolio within that cluster.

# Arguments

  - `r`: Risk measure.
  - `X`: Asset return matrix.
  - Additional cluster and weight parameters.

# Returns

  - Vector of expected risk values per cluster.

# Related

  - [`unitary_expected_risks!`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
function unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum,
                                fees::Option{<:Fees} = nothing)
    wk = zeros(eltype(X), size(X, 2))
    rk = Vector{eltype(X)}(undef, size(X, 2))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return rk
end
"""
    unitary_expected_risks!(wk, rk, r, ...)

Compute and store the expected risk for each cluster's unitary portfolio in-place.

Fills `rk` with the expected risk for the equal-weight portfolio within each cluster, and `wk` with the corresponding weights.

# Arguments

  - `wk`: Output weight vector (in-place).
  - `rk`: Output risk vector (in-place).
  - `r`: Risk measure.
  - Additional cluster and weight parameters.

# Returns

  - `nothing` (fills `wk` and `rk` in-place).

# Related

  - [`unitary_expected_risks`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
function unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure,
                                 X::MatNum, fees::Option{<:Fees} = nothing)
    fill!(rk, zero(eltype(X)))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return nothing
end

export HierarchicalResult, HierarchicalOptimiser
