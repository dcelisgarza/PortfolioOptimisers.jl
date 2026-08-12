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
        @argcheck(isfinite(scale), IsNonFiniteError("scale must be finite, got $scale"))
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
        sk::Option{<:SkSlot} = nothing,
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:MuSlot} = nothing,
        pe::Option{<:AbstractPriorEstimator} = nothing
    ) -> Skewness

Keywords correspond to the struct's fields.

## Validation

  - If `sk` is not `nothing`: `!isempty(sk)` and `size(sk, 1)^2 == size(sk, 2)`.
  - If `mu` is a `VecNum`: `!isempty(mu)`.
  - If `w` is not `nothing`: `!isempty(w)`.

!!! warning

    `mu` and `sk` are stated independently, so nothing makes them agree with each other. A caller who wants one consistent set names `pe` alone and lets it fill both from a single fit. A caller who states them by hand must make sure that they agree.

!!! info

    `sk` also admits a [`CoskewnessEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), and `mu` an [`AbstractExpectedReturnsEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref). Either is resolved against the optimisation's own prior — see [`resolve_deferred_quantities`](@ref). A coskewness estimator in `sk` also supplies `mu` from its own `me`, so that the tensor and the centre it was taken about come out of **one** object. A deferred slot wins over `pe`.

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
        mu ┼ nothing
        pe ┴ nothing
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`ThirdCentralMoment`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`bigger_is_better`](@ref)
  - [`resolve_deferred_quantities`](@ref)
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
    """
    $(field_dict[:pe_rm])
    """
    pe
    function Skewness(settings::MaxRiskMeasureSettings, ve::AbstractVarianceEstimator,
                      sk::Option{<:SkSlot}, w::Option{<:ObsWeights}, mu::Option{<:MuSlot},
                      pe::Option{<:AbstractPriorEstimator})
        if isa(sk, MatNum)
            @argcheck(!isempty(sk), IsEmptyError("sk cannot be empty"))
            @argcheck(size(sk, 1)^2 == size(sk, 2),
                      DimensionMismatch("size(sk, 1)^2 ($(size(sk, 1)^2)) must equal size(sk, 2) ($(size(sk, 2)))"))
        end
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        end
        return new{typeof(settings), typeof(ve), typeof(sk), typeof(w), typeof(mu),
                   typeof(pe)}(settings, ve, sk, w, mu, pe)
    end
end
function Skewness(; settings::MaxRiskMeasureSettings = MaxRiskMeasureSettings(),
                  ve::AbstractVarianceEstimator = SimpleVariance(),
                  sk::Option{<:SkSlot} = nothing, w::Option{<:ObsWeights} = nothing,
                  mu::Option{<:MuSlot} = nothing,
                  pe::Option{<:AbstractPriorEstimator} = nothing)::Skewness
    return Skewness(settings, ve, sk, w, mu, pe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`Skewness`](@ref) `r` against prior result `pr`.

Three passes, in order.

 1. A deferred `mu` resolves on its own.
 2. A deferred `sk` resolves next, and it carries the centre with it. `sk` is a moment **about** a centre, so the two are one pair of quantities out of one object: when `mu` is still unstated, [`deferred_centre`](@ref) reads it off the coskewness estimator's own `me`, threads it into the fit as `mean =`, and it becomes the resolved `mu`. A stated `mu` wins and is threaded in its place. An [`AbstractPriorEstimator`](@ref) centres itself, so the centre is read back off the prior result it produced.
 3. `pe` fans out into whatever both passes left `nothing`.

A deferred slot therefore **wins over `pe`**, which is the map's precedence rule one level down. The measure reads no `V`, so only the `sk` half of the coskewness pair is kept — see [`NegativeSkewness`](@ref) for the half that needs both.

# Related

  - [`Skewness`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_centre`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::Skewness, pr::AbstractPriorResult)::Skewness
    if isnothing(r.pe) && !isa(r.mu, DeferredQuantity) && !isa(r.sk, DeferredQuantity)
        return r
    end
    mu = resolve_slot(r.mu, :mu, pr)
    sk = r.sk
    if isa(sk, DeferredQuantity)
        centre = isnothing(mu) ? deferred_centre(sk, pr) : mu
        fitted = fit_deferred_moment(sk, pr, centre)
        sk = deferred_quantity(fitted, :sk)
        mu = nothing_scalar_array_selector(centre, deferred_derived_quantity(fitted, :mu))
    end
    if isnothing(r.pe)
        return Skewness(; settings = r.settings, ve = r.ve, sk = sk, w = r.w, mu = mu,
                        pe = nothing)
    end
    fitted = fit_deferred_quantity(r.pe, pr)
    return Skewness(; settings = r.settings, ve = r.ve, sk = fan_out_slot(fitted, sk, :sk),
                    w = r.w, mu = fan_out_slot(fitted, mu, :mu), pe = nothing)
end
# Deferrable slots — see `deferred_slots`. `ve` holds a variance estimator by design, not a
# Deferred Quantity, so it is not declared here.
deferred_slots(r::Skewness) = (; mu = r.mu, sk = r.sk, pe = r.pe)
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
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    sk = nothing_scalar_array_selector(r.sk, pr.sk)
    return Skewness(; ve = factory(r.ve, w), sk = sk, w = w, mu = mu, pe = nothing)
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
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return Skewness(; ve = factory(r.ve, w), sk = r.sk, w = w, mu = mu, pe = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`Skewness`](@ref) `r` sliced to asset indices `i`.

Slices the expected returns `mu` for cluster-based optimisation. `sk` passes through: it is `nothing`, or it holds a **Deferred Quantity**, which crosses the view unresolved and then fits on the subset.

# Related

  - [`Skewness`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function port_opt_view(r::Skewness, i, args...)
    mu = nothing_scalar_array_view(r.mu, i)
    return Skewness(; settings = r.settings, ve = r.ve, sk = r.sk, w = r.w, mu = mu,
                    pe = r.pe)
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
    return Skewness(; settings = r.settings, ve = r.ve, sk = sk, w = r.w, mu = mu,
                    pe = r.pe)
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
                    sk = r.sk, w = r.w, mu = r.mu, pe = r.pe)
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
                    w = r.w, mu = r.mu, pe = r.pe)
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
                    sk = r.sk, w = r.w, mu = r.mu, pe = r.pe)
end
function moment_risk(r::Skewness{<:Any, <:Any, <:Any, <:Option{<:StatsBase.AbstractWeights},
                                 <:Any}, val::VecNum)
    sigma = Statistics.std(r.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
    return res / sigma^3
end
function (r::Skewness{<:Any, <:Any, <:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any})(w::VecNum,
                                                                                          X::MatNum,
                                                                                          fees::Option{<:Fees} = nothing)
    return moment_risk(r, calc_deviations_vec(r, w, X, fees))
end
function (r::Skewness{<:Any, <:Any, <:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any})(x::VecNum)
    return moment_risk(r, calc_deviations_vec(r, x))
end
function (r::Skewness{<:Any, <:Any, <:Any, <:DynamicAbstractWeights, <:Any})(w::VecNum,
                                                                             X::MatNum,
                                                                             fees::Option{<:Fees} = nothing)
    return Skewness(; ve = r.ve, sk = r.sk, w = get_observation_weights(r.w, X), mu = r.mu,
                    pe = r.pe)(w, X, fees)
end
function (r::Skewness{<:Any, <:Any, <:Any, <:DynamicAbstractWeights, <:Any})(x::VecNum)
    return Skewness(; ve = r.ve, sk = r.sk, w = get_observation_weights(r.w, x), mu = r.mu,
                    pe = r.pe)(x)
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
        kt::Kurtosis = Kurtosis(),
        pe::Option{<:AbstractPriorEstimator} = nothing
    ) -> VarianceSkewKurtosis

Keywords correspond to the struct's fields.

!!! warning

    The three children each state their quantities independently, so nothing makes `sigma`, `sk` and `kt` agree with each other. A caller who wants one consistent set names `pe` on the container alone and lets it fill all three from a single fit. A child that names its own quantity keeps it, and the `pe` fills only what is left.

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
           │         mu ┼ nothing
           │         pe ┴ nothing
        kt ┼ Kurtosis
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │          w ┼ nothing
           │         mu ┼ nothing
           │         kt ┼ nothing
           │          N ┼ nothing
           │       alg1 ┼ FullMoment()
           │       alg2 ┼ SOCRiskExpr()
           │         pe ┴ nothing
        pe ┴ nothing
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
    @fprop @vprop vr
    """
    $(field_dict[:sk_rm])
    """
    @fprop @vprop sk
    """
    $(field_dict[:kt_rm])
    """
    @fprop @vprop kt
    """
    $(field_dict[:pe_rm])
    """
    pe
    function VarianceSkewKurtosis(settings::RiskMeasureSettings, vr::Variance, sk::Skewness,
                                  kt::Kurtosis, pe::Option{<:AbstractPriorEstimator})
        vr = no_risk_expr_risk_measure(vr)
        sk = no_risk_expr_risk_measure(sk)
        kt = no_risk_expr_risk_measure(kt)
        return new{typeof(settings), typeof(vr), typeof(sk), typeof(kt), typeof(pe)}(settings,
                                                                                     vr, sk,
                                                                                     kt, pe)
    end
end
function VarianceSkewKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              vr::Variance = Variance(), sk::Skewness = Skewness(),
                              kt::Kurtosis = Kurtosis(),
                              pe::Option{<:AbstractPriorEstimator} = nothing)
    return VarianceSkewKurtosis(settings, vr, sk, kt, pe)
end
function (r::VarianceSkewKurtosis)(w::VecNum, X::MatNum, fees::Option{<:VecNum} = nothing)
    return r.vr(w) * r.vr.settings.scale - r.sk(w, X, fees) * r.sk.settings.scale +
           r.kt(w, X, fees) * r.kt.settings.scale
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`VarianceSkewKurtosis`](@ref) `r` against prior result `pr`.

Two levels, in order. Each child resolves whatever it holds of its own, then the container's `pe` fans out into every child slot still unstated — `sigma` and its `chol` on `vr`, `sk` and `mu` on `sk`, `kt` and `mu` on `kt` — all from **one** fit.

A child that names its own quantity keeps it. This is the map's precedence rule applied one level down, and it treats a deferred child slot exactly as it already treats a stated one.

The composed measure adds a variance, a skewness and a kurtosis term, so a caller who wants the three to describe one distribution names `pe` on the container and states nothing on the children.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::VarianceSkewKurtosis, pr::AbstractPriorResult)
    vr = resolve_deferred_quantities(r.vr, pr)
    sk = resolve_deferred_quantities(r.sk, pr)
    kt = resolve_deferred_quantities(r.kt, pr)
    if isnothing(r.pe)
        return VarianceSkewKurtosis(; settings = r.settings, vr = vr, sk = sk, kt = kt,
                                    pe = nothing)
    end
    fitted = fit_deferred_quantity(r.pe, pr)
    # `chol` is derived from `sigma`, so it comes from the fan-out only when the fan-out
    # also supplies the `sigma` it factorises.
    sigma_flag = isnothing(vr.sigma)
    vr = Variance(; settings = vr.settings, sigma = fan_out_slot(fitted, vr.sigma, :sigma),
                  chol = sigma_flag ? deferred_derived_quantity(fitted, :chol) : vr.chol,
                  rc = vr.rc, alg = vr.alg)
    sk = Skewness(; settings = sk.settings, ve = sk.ve,
                  sk = fan_out_slot(fitted, sk.sk, :sk), w = sk.w,
                  mu = fan_out_slot(fitted, sk.mu, :mu), pe = nothing)
    kt = Kurtosis(; settings = kt.settings, w = kt.w, mu = fan_out_slot(fitted, kt.mu, :mu),
                  kt = fan_out_slot(fitted, kt.kt, :kt), N = kt.N, alg1 = kt.alg1,
                  alg2 = kt.alg2, pe = nothing)
    return VarianceSkewKurtosis(; settings = r.settings, vr = vr, sk = sk, kt = kt,
                                pe = nothing)
end
# Deferrable slots — see `deferred_slots`. The three children carry their own, so the check
# recurses into them.
deferred_slots(r::VarianceSkewKurtosis) = (; vr = r.vr, sk = r.sk, kt = r.kt, pe = r.pe)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`VarianceSkewKurtosis`](@ref) by fanning `pe` out over its three children, then threading `pr` into each of them.

The container holds no quantity of its own, so [`@fprop`](@ref) alone would reach the children and leave `pe` standing — the measure would say one thing and compute another. This method resolves first, which is the same order the [`JuMP`](https://github.com/jump-dev/JuMP.jl) path already uses in [`set_risk_constraints!`](@ref).

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`factory`](@ref)
"""
function factory(r::VarianceSkewKurtosis, pr::AbstractPriorResult, args...; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    return VarianceSkewKurtosis(; settings = r.settings,
                                vr = factory(r.vr, pr, args...; kwargs...),
                                sk = factory(r.sk, pr, args...; kwargs...),
                                kt = factory(r.kt, pr, args...; kwargs...), pe = nothing)
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
