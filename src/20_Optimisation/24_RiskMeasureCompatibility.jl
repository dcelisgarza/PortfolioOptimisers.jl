"""
    supported_risk_measures(::Type{<:AbstractOptimisationEstimator})
    supported_risk_measures(opt::AbstractOptimisationEstimator)

Return the abstract risk-measure category an optimiser accepts.

The answer is derived entirely from the optimiser's family supertype — the same abstract type
its constraint builders already dispatch on — so it cannot drift from what the optimiser
actually accepts (see ADR 0018):

  - risk-taking JuMP optimisers ([`RiskJuMPOptimisationEstimator`](@ref): `MeanRisk`,
    `NearOptimalCentering`, `FactorRiskContribution`, `RiskBudgeting`) accept
    [`RiskMeasure`](@ref);
  - clustering optimisers ([`ClusteringOptimisationEstimator`](@ref): `HierarchicalRiskParity`,
    `HierarchicalEqualRiskContribution`, `SchurComplementHierarchicalRiskParity`) accept
    [`OptimisationRiskMeasure`](@ref) (i.e. `RiskMeasure` ∪ `HierarchicalRiskMeasure`);
  - every other optimiser (naive, finite allocation) accepts no risk measure and returns
    `Union{}`.

Meta-optimisers (`NestedClustered`, `Stacking`, `SubsetResampling`) have no risk measure of
their own — acceptance is delegated to their inner/outer optimisers — so they **throw** here
rather than return a misleading answer. Query the inner optimiser type directly.

# Related

  - [`supports_risk_measure`](@ref)
"""
function supported_risk_measures end
supported_risk_measures(::Type{<:AbstractOptimisationEstimator}) = Union{}
supported_risk_measures(::Type{<:RiskJuMPOptimisationEstimator}) = RiskMeasure
supported_risk_measures(::Type{<:ClusteringOptimisationEstimator}) = OptimisationRiskMeasure
function supported_risk_measures(opt::AbstractOptimisationEstimator)
    return supported_risk_measures(typeof(opt))
end
function supported_risk_measures(O::SubsetResampling)
    return supported_risk_measures(O.opt)
end
"""
    _delegated_risk_measure_error(O::Type)

Throw an `ArgumentError` explaining that risk-measure acceptance for a meta-optimiser type `O`
is delegated to its inner/outer optimisers and cannot be answered directly. Used by the
meta-optimiser methods of [`supported_risk_measures`](@ref).

# Related

  - [`supported_risk_measures`](@ref)
  - [`supports_risk_measure`](@ref)
"""
function _delegated_risk_measure_error(O::Type)
    return throw(ArgumentError("risk-measure acceptance for `$(nameof(O))` is delegated to its inner/outer optimiser(s); query those directly with `supported_risk_measures`/`supports_risk_measure`. See ADR 0018."))
end
function supported_risk_measures(O::Type{<:NestedClustered})
    return _delegated_risk_measure_error(O)
end
function supported_risk_measures(O::Type{<:BaseStackingOptimisationEstimator})
    return _delegated_risk_measure_error(O)
end
"""
    supports_risk_measure(::Type{<:AbstractOptimisationEstimator},
                          ::Type{<:AbstractBaseRiskMeasure}) -> Bool
    supports_risk_measure(opt::AbstractOptimisationEstimator,
                          r::AbstractBaseRiskMeasure) -> Bool

Whether an optimiser accepts a given risk measure, as a `Bool`.

This is the programmatic form of the measure↔optimiser compatibility table. It is derived from
[`supported_risk_measures`](@ref) — `supports_risk_measure(O, R) === R <: supported_risk_measures(O)`
— so predicate, category, and dispatch share one source of truth and cannot drift.

Meta-optimisers throw (acceptance is delegated to their inner/outer optimisers); see
[`supported_risk_measures`](@ref).

# Examples

```jldoctest
julia> supports_risk_measure(MeanRisk, Variance)
true

julia> supports_risk_measure(HierarchicalRiskParity, EqualRisk)
true

julia> supports_risk_measure(MeanRisk, EqualRisk)
false
```

# Related

  - [`supported_risk_measures`](@ref)
"""
function supports_risk_measure(O::Type{<:AbstractOptimisationEstimator},
                               R::Type{<:AbstractBaseRiskMeasure})
    return R <: supported_risk_measures(O)
end
function supports_risk_measure(opt::AbstractOptimisationEstimator,
                               r::AbstractBaseRiskMeasure)
    return supports_risk_measure(typeof(opt), typeof(r))
end

export supported_risk_measures, supports_risk_measure
