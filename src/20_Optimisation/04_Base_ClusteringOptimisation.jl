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
$(DocStringExtensions.TYPEDEF)

Result type for hierarchical (clustering-based) portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
@concrete struct HierarchicalResult <: NonFiniteAllocationOptimisationResult
    "$(field_dict[:oe])"
    oe
    "$(field_dict[:pr])"
    pr
    "$(field_dict[:clr])"
    clr
    "$(field_dict[:wb])"
    wb
    "$(field_dict[:fees])"
    fees
    "$(field_dict[:retcode])"
    retcode
    "$(field_dict[:pw])"
    w
    "$(field_dict[:fb])"
    fb
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create a [`HierarchicalResult`](@ref) with a new fallback result `fb`.

# Related

  - [`HierarchicalResult`](@ref)
  - [`factory`](@ref)
"""
function factory(res::HierarchicalResult, fb::Option{<:OptE_Opt})
    return HierarchicalResult(res.oe, res.pr, res.clr, res.wb, res.fees, res.retcode, res.w,
                              fb)
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
        wb::Option{<:WbE_Wb} = WeightBounds(),
        fees::Option{<:FeesE_Fees} = nothing,
        sets::Option{<:AssetSets} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        brt::Bool = false,
        cle_pr::Bool = true,
        strict::Bool = false
    ) -> HierarchicalOptimiser

Keywords correspond to the struct's fields.

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
         │           │      │   alg ┴ Full()
         │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
         │           │      │     pdm ┼ Posdef
         │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │           │      │      dn ┼ nothing
         │           │      │      dt ┼ nothing
         │           │      │     alg ┼ nothing
         │           │      │   order ┴ DenoiseDetoneAlg()
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
         │       │      │   alg ┴ Full()
         │       │   mp ┼ DenoiseDetoneAlgMatrixProcessing
         │       │      │     pdm ┼ Posdef
         │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │       │      │      dn ┼ nothing
         │       │      │      dt ┼ nothing
         │       │      │     alg ┼ nothing
         │       │      │   order ┴ DenoiseDetoneAlg()
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
@concrete struct HierarchicalOptimiser <: BaseClusteringOptimisationEstimator
    "$(field_dict[:pe])"
    pe
    "$(field_dict[:cle])"
    cle
    "$(field_dict[:slv])"
    slv
    "$(field_dict[:wb])"
    wb
    "$(field_dict[:fees])"
    fees
    "$(field_dict[:sets])"
    sets
    "$(field_dict[:wf])"
    wf
    "$(field_dict[:brt])"
    brt
    "$(field_dict[:cle_pr])"
    cle_pr
    "$(field_dict[:strict_opt])"
    strict
    function HierarchicalOptimiser(pe::PrE_Pr, cle::HClE_HCl, slv::Option{<:Slv_VecSlv},
                                   wb::Option{<:WbE_Wb}, fees::Option{<:FeesE_Fees},
                                   sets::Option{<:AssetSets}, wf::WeightFinaliser,
                                   brt::Bool, cle_pr::Bool, strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
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
                               wb::Option{<:WbE_Wb} = WeightBounds(),
                               fees::Option{<:FeesE_Fees} = nothing,
                               sets::Option{<:AssetSets} = nothing,
                               wf::WeightFinaliser = IterativeWeightFinaliser(),
                               brt::Bool = false, cle_pr::Bool = true,
                               strict::Bool = false)::HierarchicalOptimiser
    return HierarchicalOptimiser(pe, cle, slv, wb, fees, sets, wf, brt, cle_pr, strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`HierarchicalOptimiser`](@ref) requires previous portfolio weights (based on fee structure).

# Related

  - [`needs_previous_weights`](@ref)
  - [`HierarchicalOptimiser`](@ref)
"""
function needs_previous_weights(opt::HierarchicalOptimiser)
    return needs_previous_weights(opt.fees)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create a [`HierarchicalOptimiser`](@ref) updating the fee structure with weights `w`.

# Related

  - [`HierarchicalOptimiser`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::HierarchicalOptimiser, w::AbstractVector)::HierarchicalOptimiser
    return HierarchicalOptimiser(; pe = opt.pe, cle = opt.cle, slv = opt.slv, wb = opt.wb,
                                 fees = factory(opt.fees, w), sets = opt.sets, wf = opt.wf,
                                 brt = opt.brt, cle_pr = opt.cle_pr, strict = opt.strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`HierarchicalOptimiser`](@ref) `hco` sliced to asset indices `i`.

# Related

  - [`HierarchicalOptimiser`](@ref)
  - [`opt_view`](@ref)
"""
function opt_view(hco::HierarchicalOptimiser, i)::HierarchicalOptimiser
    pe = prior_view(hco.pe, i)
    wb = weight_bounds_view(hco.wb, i)
    fees = fees_view(hco.fees, i)
    sets = asset_sets_view(hco.sets, i)
    return HierarchicalOptimiser(; pe = pe, cle = hco.cle, slv = hco.slv, wb = wb,
                                 fees = fees, wf = hco.wf, sets = sets, brt = hco.brt,
                                 cle_pr = hco.cle_pr, strict = hco.strict)
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
