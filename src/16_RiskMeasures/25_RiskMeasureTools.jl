function no_bounds_risk_measure(r::RiskMeasure, ::Any = nothing)
    settings = r.settings
    return Accessors.@set r.settings = RiskMeasureSettings(; rke = settings.rke,
                                                           scale = settings.scale)
end
function no_bounds_no_risk_expr_risk_measure(r::RiskMeasure, ::Any = nothing)
    return Accessors.@set r.settings = RiskMeasureSettings(; rke = false, scale = 1)
end
function no_risk_expr_risk_measure(r::RiskMeasure)
    settings = r.settings
    return Accessors.@set r.settings = RiskMeasureSettings(; rke = false, ub = settings.ub,
                                                           scale = settings.scale)
end
function unit_scale_risk_measure(r::RiskMeasure)
    settings = r.settings
    scale = settings.scale
    return if isone(scale)
        r
    else
        Accessors.@set r.settings = RiskMeasureSettings(; scale = one(scale),
                                                        ub = settings.ub,
                                                        rke = settings.rke)
    end
end
function bounds_risk_measure(r::RiskMeasure, ub::Number)
    settings = r.settings
    return Accessors.@set r.settings = RiskMeasureSettings(; ub = ub, rke = settings.rke,
                                                           scale = settings.scale)
end
function no_bounds_risk_measure(r::HierarchicalRiskMeasure, ::Any = nothing)
    return r
end
function no_bounds_no_risk_expr_risk_measure(r::HierarchicalRiskMeasure, ::Any = nothing)
    return r
end
function no_risk_expr_risk_measure(r::HierarchicalRiskMeasure)
    return r
end
function unit_scale_risk_measure(r::HierarchicalRiskMeasure)
    return r
end
function bounds_risk_measure(r::HierarchicalRiskMeasure, ::Any = nothing)
    return r
end
"""
    no_bounds_risk_measure(r, flag = nothing)

Return a copy of risk measure `r` with its upper-bound constraint removed while
preserving all other settings. For hierarchical risk measures, returns the input
unchanged. For vectors of risk measures, applies element-wise.

# Arguments

  - `r`: Risk measure or vector of risk measures.
  - `flag`: Ignored; kept for dispatch compatibility.

# Returns

  - Risk measure without upper bounds.

# Related

  - [`bounds_risk_measure`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_risk_measure(rs::VecBaseRM, flag::Any = nothing)
    return no_bounds_risk_measure.(rs, flag)
end
"""
    no_bounds_no_risk_expr_risk_measure(r, flag = nothing)

Return a copy of risk measure `r` with both its upper-bound constraint and its risk
expression flag disabled. For hierarchical risk measures, returns the input unchanged.
For vectors of risk measures, applies element-wise.

# Arguments

  - `r`: Risk measure or vector of risk measures.
  - `flag`: Ignored; kept for dispatch compatibility.

# Returns

  - Risk measure without bounds or risk expression.

# Related

  - [`no_bounds_risk_measure`](@ref)
  - [`no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure(r::VecBaseRM, flag::Any = nothing)
    return no_bounds_no_risk_expr_risk_measure.(r, flag)
end
"""
    no_risk_expr_risk_measure(r)

Return a copy of risk measure `r` with its risk expression flag disabled while
preserving its upper-bound constraint. For hierarchical risk measures, returns the
input unchanged. For vectors of risk measures, applies element-wise.

# Arguments

  - `r`: Risk measure or vector of risk measures.

# Returns

  - Risk measure without risk expression flag.

# Related

  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`no_bounds_risk_measure`](@ref)
"""
function no_risk_expr_risk_measure(r::VecBaseRM)
    return no_risk_expr_risk_measure.(r)
end
"""
    bounds_risk_measure(r, ubs)

Return a copy of risk measure `r` (or each element of vector `r`) with its upper-bound
set to `ubs` (or the corresponding element of `ubs`). Hierarchical risk measures are
returned unchanged.

# Arguments

  - `r`: Risk measure or vector of risk measures.
  - `ubs`: Upper-bound value or vector of upper-bound values.

# Returns

  - Risk measure or vector of risk measures with updated upper bounds.

# Related

  - [`no_bounds_risk_measure`](@ref)
"""
function bounds_risk_measure(r::VecBaseRM, ubs::VecNum)
    return bounds_risk_measure.(r, ubs)
end
"""
    unit_scale_risk_measure(r)

Return a copy of risk measure `r` with its `scale` set to `one(scale)`, preserving its
upper bound and its `rke` flag. A measure that already carries a unit scale is returned
unchanged, so the common path allocates nothing.

`scale` is a combination weight: it says how much this measure contributes to an aggregate
built from several measures. One measure is not an aggregate, so the weight has nothing to
weigh and the model drops it before it can reach the risk expression. Hierarchical risk
measures are returned unchanged.

# Arguments

  - `r`: Risk measure.

# Returns

  - Risk measure carrying a unit scale.

# Related

  - [`no_bounds_risk_measure`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function unit_scale_risk_measure end

"""
    measure_label(r::AbstractBaseRiskMeasure) -> String
    measure_label(rs::VecBaseRM) -> String

Name a risk measure for an axis label, a title, or a legend entry.

One measure answers its own type name. A vector answers its elements' names joined by `" + "`.

The vector arm is the reason the helper exists. `string(nameof(typeof(rs)))` on a vector evaluates to `"Vector"` — a wrong label, silently, with no error — and seven plot sites spelled that expression inline. Writing the rule once means a future measure-taking plot inherits it.

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
    const MomentRiskMeasures{T} = Union{<:LowOrderMoment{<:Any, T}, <:HighOrderMoment{<:Any, T}, <:Kurtosis{<:Any, T}, <:Skewness{<:Any, <:Any, <:Any, T}, <:ThirdCentralMoment{<:Any, T}}

Parameterised union of the five central-moment risk measures sharing the same observation-weight (`T`) type parameter.

The members are [`LowOrderMoment`](@ref), [`HighOrderMoment`](@ref), [`Kurtosis`](@ref), [`Skewness`](@ref) and [`ThirdCentralMoment`](@ref). Each evaluates the same way: [`resolve_observation_weights`](@ref) against the data, then the deviations from the centring target, then [`moment_risk`](@ref). So the two functor methods are written once here, not once per measure.

The kernels read only resolved weights. A measure that holds a [`DynamicAbstractWeights`](@ref) rebuilds once per call with the resolved weights, and passes the rebuilt measure to each kernel. A measure with resolved weights or no weights is not rebuilt, so it pays nothing.

# Related

  - [`resolve_observation_weights`](@ref)
  - [`moment_risk`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`DynamicAbstractWeights`](@ref)
"""
const MomentRiskMeasures{T} = Union{<:LowOrderMoment{<:Any, T}, <:HighOrderMoment{<:Any, T},
                                    <:Kurtosis{<:Any, T},
                                    <:Skewness{<:Any, <:Any, <:Any, T},
                                    <:ThirdCentralMoment{<:Any, T}}
"""
    resolve_observation_weights(r::AbstractBaseRiskMeasure, X::VecNum_MatNum)

Return the risk measure `r` with its observation weights resolved against the data `X`.

A [`MomentRiskMeasures`](@ref) member that holds a [`DynamicAbstractWeights`](@ref) rebuilds with its keyword constructor. The constructor passes the weights to the variance estimator that the measure holds: `alg.ve` of a [`SecondMoment`](@ref) or a [`StandardisedHighOrderMoment`](@ref), `ve` of a [`Skewness`](@ref). So the measure and its variance estimator read the same resolved weights, as they do after [`factory`](@ref) against a prior. Every other measure returns unchanged: resolved weights and `nothing` need no work, and a measure outside the union resolves its own weights inside its kernel.

# Algorithm

 1. Resolve `r.w` against `X` with [`get_observation_weights`](@ref), giving `w`.
 2. Rebuild `r` with its keyword constructor, with `w` in place of `r.w`. Every other field carries over.

# Arguments

  - $(arg_dict[:r])
  - `X`: The data that the weights resolve against: the asset returns matrix `observations × assets`, or a return series `observations × 1`.

# Returns

  - `r::AbstractBaseRiskMeasure`: The measure with its weights resolved.

# Related

  - [`MomentRiskMeasures`](@ref)
  - [`get_observation_weights`](@ref)
  - [`DynamicAbstractWeights`](@ref)
  - [`difference_risk`](@ref)
"""
function resolve_observation_weights(r::AbstractBaseRiskMeasure, ::VecNum_MatNum)
    return r
end
function resolve_observation_weights(r::LowOrderMoment{<:Any, <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return LowOrderMoment(; settings = r.settings, w = get_observation_weights(r.w, X),
                          mu = r.mu, alg = r.alg)
end
function resolve_observation_weights(r::HighOrderMoment{<:Any, <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return HighOrderMoment(; settings = r.settings, w = get_observation_weights(r.w, X),
                           mu = r.mu, alg = r.alg)
end
function resolve_observation_weights(r::Kurtosis{<:Any, <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return Kurtosis(; settings = r.settings, w = get_observation_weights(r.w, X), mu = r.mu,
                    kt = r.kt, N = r.N, alg1 = r.alg1, alg2 = r.alg2, pe = r.pe)
end
function resolve_observation_weights(r::Skewness{<:Any, <:Any, <:Any,
                                                 <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return Skewness(; settings = r.settings, ve = r.ve, sk = r.sk,
                    w = get_observation_weights(r.w, X), mu = r.mu, pe = r.pe)
end
function resolve_observation_weights(r::ThirdCentralMoment{<:Any, <:DynamicAbstractWeights},
                                     X::VecNum_MatNum)
    return ThirdCentralMoment(; settings = r.settings, w = get_observation_weights(r.w, X),
                              mu = r.mu)
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
