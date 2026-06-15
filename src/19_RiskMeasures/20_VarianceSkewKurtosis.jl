"""
$(DocStringExtensions.TYPEDEF)

Settings type for configuring risk measures that expose a lower bound (maximisation direction).

Encapsulates scaling, lower bounds, and risk evaluation flags for risk measures such as [`Skewness`](@ref) that are maximised in optimisation routines. The `lb` field holds an optional lower bound on the risk expression; when set, the optimiser enforces the risk is at least that value.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaxRiskMeasureSettings(;
        scale::Number = 1.0,
        lb::Option{<:RkRtBounds} = nothing,
        rke::Bool = true,
    ) -> MaxRiskMeasureSettings

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(scale)`.

# Examples

```jldoctest
julia> MaxRiskMeasureSettings()
MaxRiskMeasureSettings
  scale ┼ Float64: 1.0
     lb ┼ nothing
    rke ┴ Bool: true
```

# Related

  - [`JuMPRiskMeasureSettings`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Skewness`](@ref)
  - [`Frontier`](@ref)
"""
@concrete struct MaxRiskMeasureSettings <: JuMPRiskMeasureSettings
    """
    $(field_dict[:scale_rm])
    """
    scale
    """
    $(field_dict[:lb_rms])
    """
    lb
    """
    $(field_dict[:rke])
    """
    rke
    function MaxRiskMeasureSettings(scale::Number, lb::Option{<:RkRtBounds},
                                    rke::Bool)::MaxRiskMeasureSettings
        @argcheck(isfinite(scale))
        return new{typeof(scale), typeof(lb), typeof(rke)}(scale, lb, rke)
    end
end
function MaxRiskMeasureSettings(; scale::Number = 1.0, lb::Option{<:RkRtBounds} = nothing,
                                rke::Bool = true)::MaxRiskMeasureSettings
    return MaxRiskMeasureSettings(scale, lb, rke)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the standardised Skewness risk measure.

`Skewness` computes the third standardised central moment (skewness) of portfolio returns. Positive skewness is preferred (the distribution is skewed towards more positive values), so [`bigger_is_better`](@ref) returns `true` for this measure.

# Mathematical definition

Let ``\\mu`` be the specified centre, ``\\delta_t = x_t - \\mu``, and ``\\sigma`` the standard deviation of returns. The skewness is:

```math
\\begin{align}
\\mathrm{Skew}(\\boldsymbol{x}) &= \\frac{1}{T \\sigma^3} \\sum_{t=1}^{T} \\delta_t^3\\,.
\\end{align}
```

Where:

  - ``\\mathrm{Skew}(\\boldsymbol{x})``: Standardised skewness of portfolio returns.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mu``: Specified centre of the distribution.
  - ``\\delta_t = x_t - \\mu``: Centred deviation at period ``t``.
  - ``\\sigma``: Standard deviation of returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Skewness(;
        settings::MaxRiskMeasureSettings = MaxRiskMeasureSettings(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        sk::Option{<:MatNum} = nothing,
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:Num_VecNum_VecScalar} = nothing
    ) -> Skewness

Keywords correspond to the struct's fields.

## Validation

  - If `sk` is not `nothing`: `!isempty(sk)` and `size(sk, 1)^2 == size(sk, 2)`.
  - If `mu` is a `VecNum`: `!isempty(mu)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::Skewness)(w::VecNum, X::MatNum, fees = nothing)

Computes the skewness of the portfolio returns.

## Arguments

  - $(arg_dict[:pw])
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.

# Examples

```jldoctest
julia> Skewness()
Skewness
  settings ┼ MaxRiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      lb ┼ nothing
           │     rke ┴ Bool: true
        ve ┼ SimpleVariance
           │          me ┼ SimpleExpectedReturns
           │             │   w ┴ nothing
           │           w ┼ nothing
           │   corrected ┴ Bool: true
        sk ┼ nothing
         w ┼ nothing
        mu ┴ nothing
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`bigger_is_better`](@ref)
"""
@concrete struct Skewness <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:ve])
    """
    ve
    """
    $(field_dict[:sk])
    """
    sk
    """
    $(field_dict[:w_rm])
    """
    w
    """
    $(field_dict[:mu_rm])
    """
    mu
    function Skewness(settings::MaxRiskMeasureSettings, ve::AbstractVarianceEstimator,
                      sk::Option{<:MatNum}, w::Option{<:ObsWeights},
                      mu::Option{<:Num_VecNum_VecScalar})
        if !isnothing(sk)
            @argcheck(!isempty(sk))
            @argcheck(size(sk, 1)^2 == size(sk, 2))
        end
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
        end
        return new{typeof(settings), typeof(ve), typeof(sk), typeof(w), typeof(mu)}(settings,
                                                                                    ve, sk,
                                                                                    w, mu)
    end
end
function Skewness(; settings::MaxRiskMeasureSettings = MaxRiskMeasureSettings(),
                  ve::AbstractVarianceEstimator = SimpleVariance(),
                  sk::Option{<:MatNum} = nothing, w::Option{<:ObsWeights} = nothing,
                  mu::Option{<:Num_VecNum_VecScalar} = nothing)::Skewness
    return Skewness(settings, ve, sk, w, mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` because higher skewness is preferred over lower skewness.

# Related

  - [`Skewness`](@ref)
  - [`bigger_is_better`](@ref)
"""
function bigger_is_better(::Skewness)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Skewness`](@ref) by selecting observation weights and expected returns from the risk-measure instance or falling back to the prior result.

# Related

  - [`Skewness`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Skewness, pr::HighOrderPrior, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    sk = nothing_scalar_array_selector(r.sk, pr.sk)
    return Skewness(; ve = factory(r.ve, w), sk = sk, w = w, mu = mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Skewness`](@ref) from a [`LowOrderPrior`](@ref) result, selecting observation weights and expected returns while preserving the coskewness matrix from the risk measure.

# Related

  - [`Skewness`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Skewness, pr::LowOrderPrior, args...; kwargs...)::Skewness
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return Skewness(; ve = factory(r.ve, w), sk = r.sk, w = w, mu = mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`Skewness`](@ref) `r` sliced to asset indices `i`.

Slices the expected returns `mu` for cluster-based optimisation.

# Related

  - [`Skewness`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function port_opt_view(r::Skewness{<:Any, <:Any, <:Nothing}, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return Skewness(; ve = r.ve, sk = r.sk, w = r.w, mu = mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`Skewness`](@ref) `r` sliced to asset indices `i`, also slicing the coskewness matrix `sk`.

# Related

  - [`Skewness`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
  - [`nothing_scalar_array_view_odd_order`](@ref)
  - [`fourth_moment_index_generator`](@ref)
"""
function port_opt_view(r::Skewness{<:Any, <:Any, <:MatNum}, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    idx = fourth_moment_index_generator(size(r.sk, 1), i)
    sk = nothing_scalar_array_view_odd_order(r.sk, i, idx)
    return Skewness(; ve = r.ve, sk = sk, w = r.w, mu = mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with `rke = false`, disabling its contribution to the JuMP objective expression.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`no_risk_expr_risk_measure`](@ref)
"""
function no_risk_expr_risk_measure(r::Skewness)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = false, lb = r.settings.lb,
                                                      scale = r.settings.scale), ve = r.ve,
                    sk = r.sk, w = r.w, mu = r.mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with `rke = false` and `lb = nothing`, removing bounds and disabling its contribution to the JuMP objective expression.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure(r::Skewness, ::Any = nothing)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = false, lb = nothing,
                                                      scale = 1), ve = r.ve, sk = r.sk,
                    w = r.w, mu = r.mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with the lower bound set to `ub`.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`bounds_risk_measure`](@ref)
"""
function bounds_risk_measure(r::Skewness, ub::Number)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = r.settings.rke, lb = ub,
                                                      scale = r.settings.scale), ve = r.ve,
                    sk = r.sk, w = r.w, mu = r.mu)
end
function _moment_risk(r::Skewness{<:Any, <:Any, <:Any,
                                  <:Option{<:StatsBase.AbstractWeights}, <:Any},
                      val::VecNum)
    sigma = Statistics.std(r.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
    return res / sigma^3
end
function (r::Skewness{<:Any, <:Any, <:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any})(w::VecNum,
                                                                                          X::MatNum,
                                                                                          fees::Option{<:Fees} = nothing)
    return _moment_risk(r, calc_deviations_vec(r, w, X, fees))
end
function (r::Skewness{<:Any, <:Any, <:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any})(x::VecNum)
    return _moment_risk(r, calc_deviations_vec(r, x))
end
function (r::Skewness{<:Any, <:Any, <:Any, <:DynamicAbstractWeights, <:Any})(w::VecNum,
                                                                             X::MatNum,
                                                                             fees::Option{<:Fees} = nothing)
    return Skewness(; ve = r.ve, sk = r.sk, w = get_observation_weights(r.w, X), mu = r.mu)(w,
                                                                                            X,
                                                                                            fees)
end
function (r::Skewness{<:Any, <:Any, <:Any, <:DynamicAbstractWeights, <:Any})(x::VecNum)
    return Skewness(; ve = r.ve, sk = r.sk, w = get_observation_weights(r.w, x), mu = r.mu)(x)
end
"""
$(DocStringExtensions.TYPEDEF)

Composite risk measure combining variance, skewness, and kurtosis into a single expression.

`VarianceSkewKurtosis` encodes the joint SDP formulation ``\\sigma^2 - \\mathrm{Skew} + \\kappa`` where each component has its own scale weight. The skewness term is subtracted because higher skewness is preferable.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R}(\\boldsymbol{w}) &= s_{\\sigma^2}\\,\\sigma^2(\\boldsymbol{w})
    - s_{\\mathrm{sk}}\\,\\mathrm{Skew}(\\boldsymbol{w})
    + s_{\\kappa}\\,\\kappa(\\boldsymbol{w})\\,,
\\end{align}
```

Where:

  - ``\\sigma^2(\\boldsymbol{w})``: Portfolio variance (via [`Variance`](@ref)).
  - ``\\mathrm{Skew}(\\boldsymbol{w})``: Standardised portfolio skewness (via [`Skewness`](@ref)).
  - ``\\kappa(\\boldsymbol{w})``: Portfolio kurtosis (via [`Kurtosis`](@ref)).
  - ``s_{\\sigma^2},\\, s_{\\mathrm{sk}},\\, s_{\\kappa}``: Respective scale factors from each sub-measure's settings.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarianceSkewKurtosis(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        vr::Variance = Variance(),
        sk::Skewness = Skewness(),
        kt::Kurtosis = Kurtosis()
    ) -> VarianceSkewKurtosis

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> r = VarianceSkewKurtosis()
VarianceSkewKurtosis
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        vr ┼ Variance
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │      sigma ┼ nothing
           │       chol ┼ nothing
           │         rc ┼ nothing
           │        alg ┴ SquaredSOCRiskExpr()
        sk ┼ Skewness
           │   settings ┼ MaxRiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      lb ┼ nothing
           │            │     rke ┴ Bool: false
           │         ve ┼ SimpleVariance
           │            │          me ┼ SimpleExpectedReturns
           │            │             │   w ┴ nothing
           │            │           w ┼ nothing
           │            │   corrected ┴ Bool: true
           │         sk ┼ nothing
           │          w ┼ nothing
           │         mu ┴ nothing
        kt ┼ Kurtosis
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │          w ┼ nothing
           │         mu ┼ nothing
           │         kt ┼ nothing
           │          N ┼ nothing
           │       alg1 ┼ Full()
           │       alg2 ┴ SOCRiskExpr()
```

# Functor

    (r::VarianceSkewKurtosis)(w::VecNum, X::MatNum, fees = nothing)

Computes the variance skewness kurtosis composite risk measure of the portfolio returns.

## Arguments

  - $(arg_dict[:pw])
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee structure.

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Variance`](@ref)
  - [`Skewness`](@ref)
  - [`Kurtosis`](@ref)
"""
@propagatable @concrete struct VarianceSkewKurtosis <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:vr_rm])
    """
    @fprop vr
    """
    $(field_dict[:sk_rm])
    """
    @fprop sk
    """
    $(field_dict[:kt_rm])
    """
    @fprop kt
    function VarianceSkewKurtosis(settings::RiskMeasureSettings, vr::Variance, sk::Skewness,
                                  kt::Kurtosis)
        vr = no_risk_expr_risk_measure(vr)
        sk = no_risk_expr_risk_measure(sk)
        kt = no_risk_expr_risk_measure(kt)
        return new{typeof(settings), typeof(vr), typeof(sk), typeof(kt)}(settings, vr, sk,
                                                                         kt)
    end
end
function VarianceSkewKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              vr::Variance = Variance(), sk::Skewness = Skewness(),
                              kt::Kurtosis = Kurtosis())
    return VarianceSkewKurtosis(settings, vr, sk, kt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`VarianceSkewKurtosis`](@ref) `r` sliced to asset indices `i` by delegating `port_opt_view` to each sub-measure component.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(r::VarianceSkewKurtosis, i, args...)
    vr = port_opt_view(r.vr, i, args...)
    sk = port_opt_view(r.sk, i, args...)
    kt = port_opt_view(r.kt, i, args...)
    return VarianceSkewKurtosis(; settings = r.settings, vr = vr, sk = sk, kt = kt)
end
function (r::VarianceSkewKurtosis)(w::VecNum, X::MatNum, fees::Option{<:VecNum} = nothing)
    return r.vr(w) * r.vr.settings.scale - r.sk(w, X, fees) * r.sk.settings.scale +
           r.kt(w, X, fees) * r.kt.settings.scale
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Skewness) = WeightsReturnsFeesInput()
risk_input_kind(::VarianceSkewKurtosis) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`Skewness`](@ref) `r` supports precomputed-return evaluation.

Delegates to [`weight_independent_target`](@ref) on `r.mu`: `true` iff the target is
`Nothing`, a `Number`, or a [`MedianCenteringFunction`](@ref); `false` for per-asset targets.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`Skewness`](@ref)
"""
supports_precomputed_returns(r::Skewness) = weight_independent_target(r.mu)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: [`VarianceSkewKurtosis`](@ref) carries a weights-only variance term
`r.vr(w)` with no bare-series form.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`VarianceSkewKurtosis`](@ref)
"""
supports_precomputed_returns(::VarianceSkewKurtosis) = false

export MaxRiskMeasureSettings, Skewness, VarianceSkewKurtosis
