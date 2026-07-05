"""
    supported_risk_measures(::Type{<:AbstractOptimisationEstimator})
    supported_risk_measures(opt::AbstractOptimisationEstimator)

Return the abstract risk-measure category an optimiser accepts.

For a leaf optimiser the answer is derived entirely from its family supertype — the same abstract
type its constraint builders already dispatch on — so it cannot drift from what the optimiser
actually accepts (see ADR 0018):

  - risk-taking JuMP optimisers ([`RiskJuMPOptimisationEstimator`](@ref): `MeanRisk`,
    `NearOptimalCentering`, `FactorRiskContribution`, `RiskBudgeting`) accept
    [`RiskMeasure`](@ref);
  - clustering optimisers ([`ClusteringOptimisationEstimator`](@ref): `HierarchicalRiskParity`,
    `HierarchicalEqualRiskContribution`, `SchurComplementHierarchicalRiskParity`) accept
    [`OptimisationRiskMeasure`](@ref) (i.e. `RiskMeasure` ∪ `HierarchicalRiskMeasure`);
  - every other leaf optimiser (naive, finite allocation) accepts no risk measure and returns
    `Union{}`.

Meta-optimisers (`NestedClustered`, `Stacking`, `SubsetResampling`) have no risk measure of their
own — they **delegate** to their constituent optimisers and accept the *intersection* of what
those optimisers accept, i.e. a measure only when every one of them accepts it:

  - `NestedClustered` — the inner (`opti`) and outer (`opto`) optimisers must both accept it;
  - `Stacking` — the outer (`opto`) and *all* inner (`opti`) optimisers must accept it (the inner
    optimisers may differ, so the category is intersected by broadcasting over the inner vector);
  - `SubsetResampling` — its single inner optimiser (`opt`) must accept it.

Called on an **instance**, the delegation reads the actual child optimisers, so it is exact even
when a `Stacking`'s inner optimisers are heterogeneous. Called on a **type**, it reads the child
field types: exact for single-child metas and inner vectors whose element types are preserved,
and conservatively `Union{}` where a field's concrete children are erased to an abstract type.

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
"""
    _child_category(T::Type)

Return the risk-measure category accepted by a meta-optimiser child of type `T`, or `Union{}`
when `T` is not an optimiser (e.g. an abstract or erased child field type). Used by the type-level
meta-optimiser methods of [`supported_risk_measures`](@ref) to intersect the categories of their
inner/outer optimisers.

# Related

  - [`supported_risk_measures`](@ref)
"""
function _child_category(T::Type)
    return T <: AbstractOptimisationEstimator ? supported_risk_measures(T) : Union{}
end
# Meta-optimisers delegate: a meta accepts a measure only when *every* constituent optimiser does,
# so its accepted category is the intersection of its children's categories. Keyed on the concrete
# meta types. The instance methods read the actual children (exact, and broadcast over Stacking's
# possibly-heterogeneous inner vector); the type methods read the child field types. See ADR 0018.
function supported_risk_measures(opt::NestedClustered)
    return typeintersect(supported_risk_measures(opt.opti),
                         supported_risk_measures(opt.opto))
end
function supported_risk_measures(O::Type{<:NestedClustered})
    return typeintersect(_child_category(fieldtype(O, :opti)),
                         _child_category(fieldtype(O, :opto)))
end
function supported_risk_measures(opt::Stacking)
    return mapreduce(supported_risk_measures, typeintersect, opt.opti;
                     init = supported_risk_measures(opt.opto))
end
function supported_risk_measures(O::Type{<:Stacking})
    return mapreduce(_child_category, typeintersect,
                     Base.uniontypes(eltype(fieldtype(O, :opti)));
                     init = _child_category(fieldtype(O, :opto)))
end
function supported_risk_measures(opt::SubsetResampling)
    return supported_risk_measures(opt.opt)
end
function supported_risk_measures(O::Type{<:SubsetResampling})
    return _child_category(fieldtype(O, :opt))
end
"""
    supports_risk_measure(::Type{<:AbstractOptimisationEstimator},
                          ::Type{<:AbstractBaseRiskMeasure}) -> Bool
    supports_risk_measure(opt::AbstractOptimisationEstimator,
                          r::AbstractBaseRiskMeasure) -> Bool

Whether an optimiser accepts a given risk measure, as a `Bool`.

This is the programmatic form of the measure↔optimiser compatibility table. It is derived from
[`supported_risk_measures`](@ref) — `supports_risk_measure(O, R) === R <: supported_risk_measures(O)`
— so predicate, category, and dispatch share one source of truth and cannot drift. Pass both
arguments as types, or both as instances (an instance query resolves meta-optimiser delegation
against the actual child optimisers); see [`supported_risk_measures`](@ref).

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
    return typeof(r) <: supported_risk_measures(opt)
end

export supported_risk_measures, supports_risk_measure
