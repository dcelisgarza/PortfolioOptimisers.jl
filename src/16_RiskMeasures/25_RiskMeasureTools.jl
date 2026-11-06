function no_bounds_risk_measure(r::RiskMeasure, ::Any = nothing)
    settings = r.settings
    s = RiskMeasureSettings(; rke = settings.rke, scale = settings.scale)
    return s === settings ? r : rebuild_estimator(r, (; settings = s))
end
function no_bounds_no_risk_expr_risk_measure(r::RiskMeasure, ::Any = nothing)
    settings = r.settings
    s = RiskMeasureSettings(; rke = false, scale = one(settings.scale))
    return s === settings ? r : rebuild_estimator(r, (; settings = s))
end
function no_risk_expr_risk_measure(r::RiskMeasure)
    settings = r.settings
    s = RiskMeasureSettings(; rke = false, ub = settings.ub, scale = settings.scale)
    return s === settings ? r : rebuild_estimator(r, (; settings = s))
end
function unit_scale_risk_measure(r::RiskMeasure)
    settings = r.settings
    scale = settings.scale
    if isone(scale)
        return r
    end
    s = RiskMeasureSettings(; scale = one(scale), ub = settings.ub, rke = settings.rke)
    return rebuild_estimator(r, (; settings = s))
end
function bounds_risk_measure(r::RiskMeasure, ub::Number)
    settings = r.settings
    s = RiskMeasureSettings(; ub = ub, rke = settings.rke, scale = settings.scale)
    return s === settings ? r : rebuild_estimator(r, (; settings = s))
end
function no_bounds_risk_measure(r::Union{HierarchicalRiskMeasure,
                                         NonOptimisationRiskMeasure}, ::Any = nothing)
    return r
end
function no_bounds_no_risk_expr_risk_measure(r::Union{HierarchicalRiskMeasure,
                                                      NonOptimisationRiskMeasure},
                                             ::Any = nothing)
    return r
end
function no_risk_expr_risk_measure(r::Union{HierarchicalRiskMeasure,
                                            NonOptimisationRiskMeasure})
    return r
end
function unit_scale_risk_measure(r::Union{HierarchicalRiskMeasure,
                                          NonOptimisationRiskMeasure})
    return r
end
function bounds_risk_measure(r::Union{HierarchicalRiskMeasure, NonOptimisationRiskMeasure},
                             ::Any = nothing)
    return r
end
"""
    no_bounds_risk_measure(r::RiskMeasure, flag = nothing)
    no_bounds_risk_measure(r::VecBaseRM, flag = nothing)

Return a copy of the risk measure `r` with no bound, or a vector of such copies when `r` is a vector.

[`NearOptimalCentering`](@ref) solves its corner portfolios with this copy, so a bound of the measure does not constrain a corner. The copy keeps the risk-expression flag `rke` and the `scale`. A vector applies the method of each element to that element. A field that the table does not name keeps its value.

| Measure                                                                 | Changed                                                                              | Kept                                                                          |
|:----------------------------------------------------------------------- |:------------------------------------------------------------------------------------ |:----------------------------------------------------------------------------- |
| [`RiskMeasure`](@ref)                                                   | `settings.ub = nothing`                                                              | `settings.rke`, `settings.scale`                                              |
| [`Skewness`](@ref)                                                      | `settings.lb = nothing`                                                              | `settings.rke`, `settings.scale`                                              |
| [`VarianceSkewKurtosis`](@ref), [`GenericValueatRiskRange`](@ref)       | `settings.ub = nothing`, and the bound of each child                                 | `settings.rke`, `settings.scale`, and the `rke` and the `scale` of each child |
| [`UncertaintySetVariance`](@ref)                                        | `settings.ub = nothing`. With `flag = Val(false)`, the copy is a [`Variance`](@ref). | `settings.rke`, `settings.scale`                                              |
| [`HierarchicalRiskMeasure`](@ref), [`NonOptimisationRiskMeasure`](@ref) | nothing                                                                              | every field                                                                   |

The method for a [`RiskMeasure`](@ref) returns `r` itself when `r` has no bound.

# Arguments

  - $(arg_dict[:r])
  - $(arg_dict[:flag_nb])

# Returns

  - `r_new`: The measure without its bound, or a vector of such measures in the order of `r`.

# Examples

```jldoctest
julia> rs = [ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.1, scale = 2.0)),
             MaximumDrawdown(; settings = RiskMeasureSettings(; ub = 0.2))];

julia> [r.settings.ub for r in no_bounds_risk_measure(rs)]
2-element Vector{Nothing}:
 nothing
 nothing

julia> [r.settings.scale for r in no_bounds_risk_measure(rs)]
2-element Vector{Float64}:
 2.0
 1.0
```

# Related

  - [`bounds_risk_measure`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function no_bounds_risk_measure(rs::VecBaseRM, flag::Any = nothing)
    return no_bounds_risk_measure.(rs, flag)
end
"""
    no_bounds_no_risk_expr_risk_measure(r::RiskMeasure, flag = nothing)
    no_bounds_no_risk_expr_risk_measure(r::VecBaseRM, flag = nothing)

Return a copy of the risk measure `r` with no bound, with `rke = false` and with a unit `scale`, or a vector of such copies when `r` is a vector.

The constructors of [`RiskTrackingRiskMeasure`](@ref) and [`TrackingRiskMeasure`](@ref) take this copy of the tracked measure, because that measure only measures a distance. The copy bounds nothing and adds nothing to the risk of the objective. The unit `scale` has the type of the old `scale`. A vector applies the method of each element to that element. A field that the table does not name keeps its value.

| Measure                                                                 | Changed                                                                                                                            | Kept                                    |
|:----------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------- |
| [`RiskMeasure`](@ref)                                                   | `settings.ub = nothing`, `settings.rke = false`, `settings.scale = 1`                                                              | nothing in `settings`                   |
| [`Skewness`](@ref)                                                      | `settings.lb = nothing`, `settings.rke = false`, `settings.scale = 1`                                                              | nothing in `settings`                   |
| [`VarianceSkewKurtosis`](@ref), [`GenericValueatRiskRange`](@ref)       | `settings.ub = nothing`, `settings.rke = false`, `settings.scale = 1`, and the bound of each child                                 | the `rke` and the `scale` of each child |
| [`UncertaintySetVariance`](@ref)                                        | `settings.ub = nothing`, `settings.rke = false`, `settings.scale = 1`. With `flag = Val(false)`, the copy is a [`Variance`](@ref). | nothing in `settings`                   |
| [`HierarchicalRiskMeasure`](@ref), [`NonOptimisationRiskMeasure`](@ref) | nothing                                                                                                                            | every field                             |

The method for a [`RiskMeasure`](@ref) returns `r` itself when `r` already has no bound, `rke = false` and a unit `scale`.

# Arguments

  - $(arg_dict[:r])
  - $(arg_dict[:flag_nb])

# Returns

  - `r_new`: The measure without its bound and outside the risk of the objective, or a vector of such measures in the order of `r`.

# Examples

```jldoctest
julia> rs = [ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.1, scale = 2.0)),
             MaximumDrawdown(; settings = RiskMeasureSettings(; ub = 0.2))];

julia> [r.settings.rke for r in no_bounds_no_risk_expr_risk_measure(rs)]
2-element Vector{Bool}:
 0
 0

julia> [r.settings.scale for r in no_bounds_no_risk_expr_risk_measure(rs)]
2-element Vector{Float64}:
 1.0
 1.0
```

# Related

  - [`no_bounds_risk_measure`](@ref)
  - [`no_risk_expr_risk_measure`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
  - [`TrackingRiskMeasure`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure(r::VecBaseRM, flag::Any = nothing)
    return no_bounds_no_risk_expr_risk_measure.(r, flag)
end
"""
    no_risk_expr_risk_measure(r::RiskMeasure)
    no_risk_expr_risk_measure(r::VecBaseRM)

Return a copy of the risk measure `r` with `rke = false`, or a vector of such copies when `r` is a vector.

The copy adds nothing to the risk of the objective, and it keeps its bound. [`VarianceSkewKurtosis`](@ref) and [`GenericValueatRiskRange`](@ref) take this copy of each child, so the term of a child enters the model once, inside the risk expression of its parent. [`set_allocation_risk_ceiling!`](@ref) takes this copy too, because the ceiling of a [`ProgrammeAllocationSet`](@ref) bounds the measure and does not minimise it. A vector applies the method of each element to that element. A field that the table does not name keeps its value.

| Measure                                                                                                                                  | Changed                | Kept                                                           |
|:---------------------------------------------------------------------------------------------------------------------------------------- |:---------------------- |:-------------------------------------------------------------- |
| [`RiskMeasure`](@ref), [`VarianceSkewKurtosis`](@ref), [`GenericValueatRiskRange`](@ref) and [`UncertaintySetVariance`](@ref) among them | `settings.rke = false` | `settings.ub`, `settings.scale`, and every field of each child |
| [`Skewness`](@ref)                                                                                                                       | `settings.rke = false` | `settings.lb`, `settings.scale`                                |
| [`HierarchicalRiskMeasure`](@ref), [`NonOptimisationRiskMeasure`](@ref)                                                                  | nothing                | every field                                                    |

The method for a [`RiskMeasure`](@ref) returns `r` itself when `r` already has `rke = false`.

# Arguments

  - $(arg_dict[:r])

# Returns

  - `r_new`: The measure outside the risk of the objective, or a vector of such measures in the order of `r`.

# Examples

```jldoctest
julia> rs = [ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.1, scale = 2.0)),
             MaximumDrawdown(; settings = RiskMeasureSettings(; ub = 0.2))];

julia> [r.settings.rke for r in no_risk_expr_risk_measure(rs)]
2-element Vector{Bool}:
 0
 0

julia> [r.settings.ub for r in no_risk_expr_risk_measure(rs)]
2-element Vector{Float64}:
 0.1
 0.2
```

# Related

  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`no_bounds_risk_measure`](@ref)
  - [`GenericValueatRiskRange`](@ref)
  - [`VarianceSkewKurtosis`](@ref)
"""
function no_risk_expr_risk_measure(r::VecBaseRM)
    return no_risk_expr_risk_measure.(r)
end
"""
    bounds_risk_measure(r::RiskMeasure, ub::Number)
    bounds_risk_measure(r::VecBaseRM, ubs::VecNum)

Return a copy of the risk measure `r` with its bound set to `ub`, or a vector of such copies when `r` is a vector.

The vector form gives the element `i` of `r` the bound `ubs[i]`. The copy keeps `rke` and `scale`. A field that the table does not name keeps its value.

| Measure                                                                                                                                  | Changed                     | Kept                                                            |
|:---------------------------------------------------------------------------------------------------------------------------------------- |:--------------------------- |:--------------------------------------------------------------- |
| [`RiskMeasure`](@ref), [`VarianceSkewKurtosis`](@ref), [`GenericValueatRiskRange`](@ref) and [`UncertaintySetVariance`](@ref) among them | `settings.ub = ub`          | `settings.rke`, `settings.scale`, and every field of each child |
| [`Skewness`](@ref)                                                                                                                       | `settings.lb = ub`, a floor | `settings.rke`, `settings.scale`                                |
| [`HierarchicalRiskMeasure`](@ref), [`NonOptimisationRiskMeasure`](@ref)                                                                  | nothing                     | every field                                                     |

The method for a [`RiskMeasure`](@ref) returns `r` itself when its bound already is `ub`.

# Arguments

  - $(arg_dict[:r])
  - `ub`, `ubs`: The bound of one measure, or a vector of bounds with one bound for each measure of `r`.

# Validation

  - [`RiskMeasureSettings`](@ref) raises a `DomainError` when a bound is negative or not finite.
  - `length(ubs) == length(r)`, else a `DimensionMismatch`.

# Returns

  - `r_new`: The measure with its new bound, or a vector of such measures in the order of `r`.

# Examples

```jldoctest
julia> rs = [ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.1, scale = 2.0)),
             MaximumDrawdown(; settings = RiskMeasureSettings(; ub = 0.2))];

julia> [r.settings.ub for r in bounds_risk_measure(rs, [0.05, 0.3])]
2-element Vector{Float64}:
 0.05
 0.3
```

# Related

  - [`no_bounds_risk_measure`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
function bounds_risk_measure(r::VecBaseRM, ubs::VecNum)
    @argcheck(length(r) == length(ubs),
              DimensionMismatch("length(r) == length(ubs) must hold: each measure takes one bound. Got\nlength(r) => $(length(r))\nlength(ubs) => $(length(ubs))"))
    return bounds_risk_measure.(r, ubs)
end
"""
    unit_scale_risk_measure(r::AbstractBaseRiskMeasure)

Return a copy of the risk measure `r` with `scale = one(scale)`, and keep its bound and `rke`.

`scale` weighs one measure against the other measures of a vector. One measure has no other measure to weigh against, so [`set_risk_constraints!`](@ref) takes this copy before it builds the risk of one measure. A field that the table does not name keeps its value.

| Measure                                                                                                                                  | Changed                                | Kept                                                         |
|:---------------------------------------------------------------------------------------------------------------------------------------- |:-------------------------------------- |:------------------------------------------------------------ |
| [`RiskMeasure`](@ref), [`VarianceSkewKurtosis`](@ref), [`GenericValueatRiskRange`](@ref) and [`UncertaintySetVariance`](@ref) among them | `settings.scale = one(settings.scale)` | `settings.ub`, `settings.rke`, and every field of each child |
| [`Skewness`](@ref)                                                                                                                       | `settings.scale = one(settings.scale)` | `settings.lb`, `settings.rke`                                |
| [`HierarchicalRiskMeasure`](@ref), [`NonOptimisationRiskMeasure`](@ref)                                                                  | nothing                                | every field                                                  |

[`VarianceSkewKurtosis`](@ref) and [`GenericValueatRiskRange`](@ref) keep the `scale` of each child, because that `scale` weighs the term of the child inside its parent. A measure that already has a unit `scale` returns as `r` itself, with no copy.

# Arguments

  - `r`: The risk measure.

# Returns

  - `r_new`: The measure with a unit `scale`.

# Examples

```jldoctest
julia> r = ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.1, scale = 2.0));

julia> unit_scale_risk_measure(r).settings
RiskMeasureSettings
  scale ┼ Float64: 1.0
     ub ┼ Float64: 0.1
    rke ┴ Bool: true
```

# Related

  - [`no_bounds_risk_measure`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function unit_scale_risk_measure end

"""
    measure_label(r::AbstractBaseRiskMeasure) -> String
    measure_label(rs::VecBaseRM) -> String

Return the name of a risk measure for an axis label, a title or a legend entry.

One measure gives the name of its type. A vector gives the names of its elements, joined by `" + "`. The vector method is the reason for the function. `string(nameof(typeof(rs)))` of a vector gives `"Vector"`, which is a wrong label and raises no error. The measure plots and the frontier plots call this function to name a measure.

# Arguments

  - $(arg_dict[:r])

# Returns

  - `label::String`: The name of the measure, or the joined names of the measures of the vector.

# Examples

```jldoctest
julia> PortfolioOptimisers.measure_label(ConditionalValueatRisk())
"ConditionalValueatRisk"

julia> PortfolioOptimisers.measure_label([Variance(), ConditionalValueatRisk()])
"Variance + ConditionalValueatRisk"
```

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`VecBaseRM`](@ref)
"""
function measure_label(r::AbstractBaseRiskMeasure)
    return string(nameof(typeof(r)))
end
function measure_label(rs::VecBaseRM)
    return join(measure_label.(rs), " + ")
end

"""
    const MomentRiskMeasures{T} = Union{LowOrderMoment{<:Any, T}, HighOrderMoment{<:Any, T}, ThirdCentralMoment{<:Any, T}}

Parameterised union of the three central-moment risk measures that read the returns and whose observation weights have the type `T`.

The members are [`LowOrderMoment`](@ref), [`HighOrderMoment`](@ref) and [`ThirdCentralMoment`](@ref). [`Kurtosis`](@ref) and [`Skewness`](@ref) read a comoment matrix and not the returns, so they are not members. Each member computes its risk in the same three operations. [`resolve_observation_weights`](@ref) resolves the weights against the data, [`calc_deviations_vec`](@ref) gives the deviations from the Moment Target, and [`moment_risk`](@ref) reduces the deviations. So the union holds the two functor methods, one for the asset returns and one for a return series, and each member uses them.

The kernels read only resolved weights. A measure that holds a [`DynamicAbstractWeights`](@ref) rebuilds one time for each call, with the resolved weights, and passes the new measure to each kernel. A measure with stored weights or with no weights does not rebuild, so it has no extra cost.

# Related

  - [`resolve_observation_weights`](@ref)
  - [`moment_risk`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`DynamicAbstractWeights`](@ref)
"""
const MomentRiskMeasures{T} = Union{LowOrderMoment{<:Any, T}, HighOrderMoment{<:Any, T},
                                    ThirdCentralMoment{<:Any, T}}
"""
    resolve_observation_weights(r::AbstractBaseRiskMeasure, X::VecNum_MatNum)

Return the risk measure `r` with its observation weights resolved against the data `X`.

A [`MomentRiskMeasures`](@ref) member that holds a [`DynamicAbstractWeights`](@ref) rebuilds with its keyword constructor. The constructor passes the weights to the variance estimator of the measure. That estimator is `alg.ve` of a [`SecondMoment`](@ref), and `ve` of a [`HighOrderMoment`](@ref) or of a [`ThirdCentralMoment`](@ref). So the measure and its variance estimator read the same resolved weights, as they do after [`factory`](@ref) against a prior.

A member that holds a `StatsBase.AbstractWeights` returns unchanged, after [`checked_observation_weights`](@ref) checks its weights against `X`. Every other measure returns unchanged. A measure with `w = nothing` needs no weights, and a measure outside the union resolves its own weights inside its kernel.

# Algorithm

 1. Resolve `r.w` against `X` with [`checked_observation_weights`](@ref), giving `w`. This also checks that `w` has one entry for each observation.
 2. Rebuild `r` with its keyword constructor, with `w` in place of `r.w`. Every other field keeps its value. Stored weights need no rebuild, so this step applies to a [`DynamicAbstractWeights`](@ref) only.

# Validation

  - [`checked_observation_weights`](@ref) raises a `DimensionMismatch` when the weights do not have one entry for each observation of `X`. It also refuses resolved weights that [`assert_observation_weights`](@ref) refuses.

# Arguments

  - $(arg_dict[:r])
  - `X`: The data that the weights resolve against: the asset returns matrix `observations × assets`, or a return series `observations × 1`.

# Returns

  - `r::AbstractBaseRiskMeasure`: The measure with its weights resolved.

# Related

  - [`MomentRiskMeasures`](@ref)
  - [`checked_observation_weights`](@ref)
  - [`DynamicAbstractWeights`](@ref)
  - [`difference_risk`](@ref)
"""
function resolve_observation_weights(r::AbstractBaseRiskMeasure, ::VecNum_MatNum)
    return r
end
function resolve_observation_weights(r::MomentRiskMeasures{<:StatsBase.AbstractWeights},
                                     X::VecNum_MatNum)
    checked_observation_weights(r.w, X)
    return r
end
function resolve_observation_weights(r::LowOrderMoment{<:Any, <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return LowOrderMoment(; settings = r.settings, w = checked_observation_weights(r.w, X),
                          mu = r.mu, alg = r.alg)
end
function resolve_observation_weights(r::HighOrderMoment{<:Any, <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return HighOrderMoment(; settings = r.settings, w = checked_observation_weights(r.w, X),
                           mu = r.mu, alg = r.alg, ve = r.ve)
end
function resolve_observation_weights(r::ThirdCentralMoment{<:Any, <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return ThirdCentralMoment(; settings = r.settings,
                              w = checked_observation_weights(r.w, X), mu = r.mu, ve = r.ve)
end
function (r::MomentRiskMeasures)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    resolved = resolve_observation_weights(r, X)
    return moment_risk(resolved, calc_deviations_vec(resolved, w, X, fees))
end
function (r::MomentRiskMeasures)(x::VecNum)
    resolved = resolve_observation_weights(r, x)
    return moment_risk(resolved, calc_deviations_vec(resolved, x))
end

export no_bounds_risk_measure, no_bounds_no_risk_expr_risk_measure,
       no_risk_expr_risk_measure, bounds_risk_measure, unit_scale_risk_measure
