"""
$(DocStringExtensions.TYPEDEF)

Weights a risk measure inside an aggregate, and bounds its risk expression from **below**.

This is the settings type of a quantity that the optimisation wants **more** of, so its bound is a floor and not a ceiling. It is the skewness floor ``\\bar{\\mu}_3`` of Problem 13 of [sdpmom](@cite). [`Skewness`](@ref) is its one holder, and a stated `lb` reaches the model only through [`VarianceSkewKurtosis`](@ref), which builds the skewness term. `Skewness` is a [`NonOptimisationRiskMeasure`](@ref), so no optimiser takes it alone.

Unlike the `ub` of [`RiskMeasureSettings`](@ref), `lb` is not checked for non-negativity. A third moment can be negative, so a negative floor is valid.

!!! warning

    The floor holds the relaxed third moment of the model, not the third moment of the returned weights. The method of [`set_risk_constraints!`](@ref) for [`VarianceSkewKurtosis`](@ref) states the gap under `## Relaxation`.

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
  - `lb` is validated with [`assert_nonempty_finite_val`](@ref).

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
  - [`VarianceSkewKurtosis`](@ref)
  - [`Frontier`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)

# References

  - $(ref_dict[:sdpmom]) Section 3, Problem 13.
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
        assert_nonempty_finite_val(lb, :lb)
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

Measures the standardised skewness of the portfolio returns, the third central moment divided by the cube of the standard deviation.

A larger skewness is better, so [`bigger_is_better`](@ref) returns `true`. The measure is a [`NonOptimisationRiskMeasure`](@ref) and carries a [`MaxRiskMeasureSettings`](@ref), whose bound is the floor `lb`. So it answers [`bigger_is_better`](@ref), [`bounds_risk_measure`](@ref), [`no_bounds_risk_measure`](@ref), [`no_risk_expr_risk_measure`](@ref) and [`no_bounds_no_risk_expr_risk_measure`](@ref) with methods of its own: the methods for a [`RiskMeasure`](@ref) build a [`RiskMeasureSettings`](@ref), which has a ceiling `ub` and no floor.

No optimiser takes the measure alone. [`VarianceSkewKurtosis`](@ref) reads its `sk` and its `settings`, and it builds the third moment ``m_3(\\boldsymbol{w})`` from `sk`. That term is not standardised, so `settings.lb` and `settings.scale` act on ``m_3`` and not on the value of the functor.

# Mathematical definition

```math
\\begin{align}
\\delta_t &= x_t - \\mu\\,, \\\\
\\mathrm{Skew}(\\boldsymbol{w}) &= \\frac{1}{T \\hat{\\sigma}^3} \\sum_{t=1}^{T} \\delta_t^3\\,, \\\\
\\hat{\\sigma}^2 &= \\frac{1}{T - 1} \\sum_{t=1}^{T} \\delta_t^2\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mu``: Centre of the moment, which follows the rules of [`calc_moment_target`](@ref).
  - ``\\delta_t``: Centred portfolio return at observation ``t``.
  - ``\\hat{\\sigma}``: Standard deviation of the deviations about zero, under the default `ve`. Its `corrected` field sets the divisor ``T - 1``, and `corrected = false` sets ``T``.
  - $(math_dict[:m3_port])
  - $(math_dict[:M3_cosk])
  - $(math_dict[:w_port])

Under the default `ve` the third moment has the divisor ``T`` and the variance has the divisor ``T - 1``. So the value is ``((T - 1)/T)^{3/2}`` times the moment coefficient of skewness, which divides both by ``T``. When `w` holds observation weights, a weighted mean replaces ``\\frac{1}{T} \\sum_{t}``, and `ve` reads the same weights. ``m_3`` agrees with ``\\frac{1}{T} \\sum_{t} \\delta_t^3`` when ``\\mathbf{M}_3`` is the coskewness about the per-asset centres whose image under ``\\boldsymbol{w}`` is ``\\mu``. That is Equation 1 of [sdpmom](@cite).

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

Keywords correspond to the struct's fields. A `w` that is not `nothing` is passed to `ve` with [`factory`](@ref), so the variance estimator reads the same weights as the measure.

## Validation

  - If `sk` is not `nothing`: `!isempty(sk)` and `size(sk, 1)^2 == size(sk, 2)`.
  - If `mu` is a `VecNum`: `!isempty(mu)`.
  - $(val_dict[:oow_rm])

!!! warning

    `mu` and `sk` are stated independently, so nothing makes them agree with each other. A caller who wants one consistent set names `pe` alone and lets it fill both from a single fit. A caller who states them by hand must make sure that they agree.

!!! info

    `sk` also admits a [`CoskewnessEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), and `mu` an [`AbstractExpectedReturnsEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref). [`resolve_deferred_quantities`](@ref) resolves either against the prior of the optimisation. A coskewness estimator in `sk` also supplies `mu` from its own `me`, so that the tensor and its centre come out of **one** object. A deferred slot wins over `pe`.

!!! warning

    A `Number` or a `VecScalar` `mu` beside a [`CoskewnessEstimator`](@ref) in `sk` gives the target of the measure only when the weights sum to one. The estimator takes the tensor about the per-asset centre that [`centring_target`](@ref) makes, and a scalar part ``s`` is spread to every asset. So the tensor centres the portfolio return on ``s \\sum_i w_i``, and the functor centres it on ``s``. Under any other budget, the moment that the tensor gives, which [`VarianceSkewKurtosis`](@ref) reads, and the functor disagree. A budget of 0.5, a leveraged budget and a market-neutral budget are examples. A vector `mu`, or `mu = nothing`, agrees under every budget.

# Functor

    (r::Skewness)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    (r::Skewness)(x::VecNum)

Computes the skewness of the portfolio returns. The second arity takes a precomputed
portfolio return series and skips the weighting step.

## Arguments

  - $(arg_dict[:pw])
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - $(arg_dict[:fees])

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
  - [`VarianceSkewKurtosis`](@ref)
  - [`NegativeSkewness`](@ref): a convex part of the third moment that an optimiser can minimise alone.

# References

  - $(ref_dict[:sdpmom]) Section 2, Equation 1.
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
    $(field_dict[:sk_slot])
    """
    sk
    """
    $(field_dict[:oow])
    """
    w
    """
    $(field_dict[:mu_slot])
    """
    mu
    """
    $(field_dict[:pe_rm])
    """
    pe
    function Skewness(settings::MaxRiskMeasureSettings, ve::AbstractVarianceEstimator,
                      sk::Option{<:SkSlot}, w::Option{<:ObsWeights}, mu::Option{<:MuSlot},
                      pe::Option{<:AbstractPriorEstimator})
        assert_returns_source(returns_source(pe), (nothing,), :Skewness, "pe.x_src")
        if isa(sk, MatNum)
            @argcheck(!isempty(sk), IsEmptyError("sk cannot be empty"))
            @argcheck(size(sk, 1)^2 == size(sk, 2),
                      DimensionMismatch("size(sk, 1)^2 ($(size(sk, 1)^2)) must equal size(sk, 2) ($(size(sk, 2)))"))
        end
        assert_observation_weights(w, :w)
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
    return Skewness(settings, factory(ve, w), sk, w, mu, pe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`Skewness`](@ref) `r` against prior result `pr`.

A deferred slot **wins over `pe`**, because `pe` fills only the slots that the first two steps leave `nothing`. The measure reads no `V`, so it keeps only the `sk` half of the coskewness pair. [`NegativeSkewness`](@ref) reads both halves. A measure that holds no **Deferred Quantity** and no `pe` returns unchanged.

# Algorithm

 1. Resolve a deferred `mu` on its own, giving `mu`.
 2. Resolve a deferred `sk` with its centre, giving `sk` and `mu`. `sk` is a moment **about** a centre, so the two come out of one object. When `mu` is still `nothing`, [`deferred_centre`](@ref) reads the centre off the `me` of the coskewness estimator, the fit takes it as `mean =`, and it becomes the resolved `mu`. A stated `mu` takes its place in the fit. An [`AbstractPriorEstimator`](@ref) centres itself, so the centre is read back off the prior result that it gives.
 3. Fan `pe` out into each slot that steps 1 and 2 left `nothing`, with [`fan_out_pe`](@ref), and return the rebuilt measure.

# Related

  - [`Skewness`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_centre`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::Skewness, pr::AbstractPriorResult,
                                     ::Any = nothing)::Skewness
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
    return fan_out_pe(r, r.pe, pr, (; sk = sk, mu = mu))
end
# Deferrable slots — see `deferred_slots`. `ve` holds a variance estimator by design, not a
# Deferred Quantity, so it is not declared here.
deferred_slots(r::Skewness) = (; mu = r.mu, sk = r.sk, pe = r.pe)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`, because a larger skewness is better.

The generic method for an [`AbstractBaseRiskMeasure`](@ref) returns `false`. The asset selection, the parameter search and the ranking of cross-validation results read this trait to choose the direction of the score.

# Related

  - [`Skewness`](@ref)
  - [`bigger_is_better`](@ref)
"""
function bigger_is_better(::Skewness)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Skewness`](@ref) that holds its weights, its centre and its coskewness matrix, from the measure or from the [`HighOrderPrior`](@ref) result `pr`.

# Algorithm

 1. Resolve each **Deferred Quantity** with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select `w`, `mu` and `sk` field by field with [`nothing_scalar_array_selector`](@ref). A field that the measure states keeps its value, and a field that is `nothing` takes `pr.w`, `pr.mu` or `pr.sk`.
 3. Pass `w` to the variance estimator with [`factory`](@ref), so that `ve` reads the same weights.
 4. Build a new `Skewness` from these fields and the `settings` of `r`, with `pe = nothing`.

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
    return Skewness(; settings = r.settings, ve = factory(r.ve, w), sk = sk, w = w, mu = mu,
                    pe = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Skewness`](@ref) that holds its weights and its centre, from the measure or from the [`LowOrderPrior`](@ref) result `pr`.

A [`LowOrderPrior`](@ref) holds no coskewness matrix, so `sk` keeps the matrix that the measure states, or the one that a **Deferred Quantity** resolves to. The functor reads only the returns, so a measure with `sk = nothing` evaluates.

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
    return Skewness(; settings = r.settings, ve = factory(r.ve, w), sk = r.sk, w = w,
                    mu = mu, pe = nothing)
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

Return a view of [`Skewness`](@ref) `r` sliced to asset indices `i`, with a view of its coskewness matrix `sk`.

The view agrees with the coskewness that a refit on the assets `i` gives.

# Algorithm

 1. Take the view of `mu` at `i` with [`nothing_scalar_array_view`](@ref).
 2. Compute `idx`, the columns of `sk` that belong to the pairs of assets in `i`, with [`fourth_moment_index_generator`](@ref).
 3. Take the view of `sk` at the rows `i` and the columns `idx` with [`nothing_scalar_array_view_odd_order`](@ref).
 4. Build a new `Skewness` from `mu`, `sk` and the other fields of `r`.

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

Return a copy of [`Skewness`](@ref) `r` with `rke = false`, so that the measure adds nothing to the risk of the objective.

The copy keeps `lb` and `scale`. The constructor of [`VarianceSkewKurtosis`](@ref) calls this method on its `sk`, so the composite registers the skewness term once, inside its own risk expression. The method exists because the method for a [`RiskMeasure`](@ref) builds a [`RiskMeasureSettings`](@ref), which has no `lb`.

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

Return a copy of [`Skewness`](@ref) `r` with `rke = false`, `lb = nothing` and a unit `scale`.

This is the copy that a measure takes when it only measures a distance, as the method for a [`RiskMeasure`](@ref) does. The method exists because that method builds a [`RiskMeasureSettings`](@ref), which has no `lb`.

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

Return a copy of [`Skewness`](@ref) `r` with the floor `lb` set to `ub`.

[`bounds_risk_measure`](@ref) passes the bound of every measure as `ub`. The bound of a [`MaxRiskMeasureSettings`](@ref) is a floor, so this method writes it to `lb`, and keeps `rke` and `scale`.

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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`Skewness`](@ref) `r` with `lb = nothing`, and keep `rke` and `scale`.

[`no_bounds_risk_measure`](@ref) of a [`VarianceSkewKurtosis`](@ref) calls this method on its `sk`, so that the floor of the skewness term leaves the model with the other bounds. The method exists because the method for a [`RiskMeasure`](@ref) builds a [`RiskMeasureSettings`](@ref), which has no `lb`.

# Related

  - [`Skewness`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`no_bounds_risk_measure`](@ref)
"""
function no_bounds_risk_measure(r::Skewness, ::Any = nothing)
    return Skewness(;
                    settings = MaxRiskMeasureSettings(; rke = r.settings.rke, lb = nothing,
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
# Evaluation entry points — see `MomentRiskMeasures` in `25_RiskMeasureTools.jl`.
"""
$(DocStringExtensions.TYPEDEF)

Measures a weighted sum of the variance, the third central moment and the fourth central moment of the portfolio returns.

The third moment enters with a negative sign, because a larger skewness is better. Each term takes the `scale` of its child as its weight, and the bound of its child as its bound. The skewness bound is the floor `sk.settings.lb`, and the other two are ceilings. The terms are the raw moments: the third moment is not standardised, and the fourth moment is not square-rooted under any `alg2` of `kt`. The model and the functor read the matrices of the children, so the observation weights `w` of `sk` and of `kt`, and the `alg1` of `kt`, act on the functors of the children alone. To weight a term, name a **Deferred Quantity** that carries the weights, such as `kt = Kurtosis(; kt = Cokurtosis(; w = w))`.

[`set_risk_constraints!`](@ref) states the model. It is the semidefinite relaxation of Problems 12 to 14 of [sdpmom](@cite), and that work states that the relaxation is tight over a range of skewness values only. The `## Relaxation` of the method states the gap of each term.

!!! warning

    A skewness scale ``s_{\\mathrm{sk}}`` that is large against ``s_{\\sigma^2}`` and ``s_{\\kappa}`` loosens the relaxation. The optimum moves to relaxation blocks far from any product of the weights, and the value of the model stops tracking the value of the functor, without bound. A first-order solver can then stop at a point that breaks the weight bounds and the budget, and the solution check accepts that point as solved when `check_sol = (; allow_almost = true)`. Raise ``s_{\\mathrm{sk}}`` in small steps, and compare the value of the model with the value of the functor at the returned weights. The default `check_sol` rejects an approximate solution, so a solver that stops short fails and returns no such point.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R}(\\boldsymbol{w}) &= s_{\\sigma^2}\\, \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}
    - s_{\\mathrm{sk}}\\, m_3(\\boldsymbol{w})
    + s_{\\kappa}\\, m_4(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:m3_port])
  - $(math_dict[:M3_cosk])
  - $(math_dict[:m4_port])
  - $(math_dict[:K_cokurt])
  - ``s_{\\sigma^2}``, ``s_{\\mathrm{sk}}``, ``s_{\\kappa}``: The `settings.scale` of `vr`, `sk` and `kt`.

The three matrices are the moments of the returns, so ``\\mathcal{R}`` is Equation 2 of [sdpmom](@cite) for orders two and four and Equation 1 for order three. With the default prior, the covariance has the divisor ``T - 1`` and the two higher moments have the divisor ``T``.

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

!!! info

    The constructor passes `vr`, `sk` and `kt` through [`no_risk_expr_risk_measure`](@ref), so a child never registers a risk expression of its own. This is why the children show `rke ┴ Bool: false` below.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `vr`: Recursively updated through [`factory`](@ref).
  - `sk`: Recursively updated through [`factory`](@ref).
  - `kt`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `vr`: Recursively viewed through [`port_opt_view`](@ref).
  - `sk`: Recursively viewed through [`port_opt_view`](@ref).
  - `kt`: Recursively viewed through [`port_opt_view`](@ref).

# Functor

    (r::VarianceSkewKurtosis)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)

Computes ``\\mathcal{R}(\\boldsymbol{w})`` from `vr.sigma`, `sk.sk` and `kt.kt`, so a measure needs [`factory`](@ref) with a [`HighOrderPrior`](@ref) result first, unless its children state the three matrices. `X` and `fees` take part in no term. The matrices fix the moments, and a central moment does not move when `fees` shifts every observation by the same constant.

## Arguments

  - $(arg_dict[:pw])
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - $(arg_dict[:fees])

# Examples

```jldoctest
julia> r = VarianceSkewKurtosis()
VarianceSkewKurtosis
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        vr ┼ Variance
           │    settings ┼ RiskMeasureSettings
           │             │   scale ┼ Float64: 1.0
           │             │      ub ┼ nothing
           │             │     rke ┴ Bool: false
           │       sigma ┼ nothing
           │        chol ┼ nothing
           │          rc ┼ nothing
           │         alg ┼ SquaredSOCRiskExpr()
           │    mtx_sqrt ┼ EigenFallbackSquareRoot()
           │   ratio_alg ┼ SquareRootRatio()
           │      corner ┴ nothing
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
           │         pe ┼ nothing
           │   mtx_sqrt ┴ EigenFallbackSquareRoot()
        pe ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Variance`](@ref)
  - [`Skewness`](@ref)
  - [`Kurtosis`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`set_risk_constraints!`](@ref): states the model and its relaxation.
  - [`no_bounds_risk_measure`](@ref): removes the bounds of the container and of the three children.

# References

  - $(ref_dict[:sdpmom]) Section 2, Equations 1 and 2, and Section 3, Problems 12 to 14.
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
        assert_returns_source(returns_source(pe), (nothing,), :VarianceSkewKurtosis,
                              "pe.x_src")
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
# `X` and `fees` are unused, and both facts follow from the model. The three terms are the
# raw moments read off the pinned comoment matrices, exactly as `set_risk_constraints!` reads
# them off the relaxation blocks, so no return series enters. Fees shift every observation by
# the same per-period constant given `w`, and a central moment is invariant to that shift, so
# dropping them changes no value. Delegating to `r.sk` and `r.kt` instead reported the
# STANDARDISED skewness and, under `SOCRiskExpr`, the square root of the fourth moment, so the
# functor did not agree with the model it names.
function (r::VarianceSkewKurtosis)(w::VecNum, ::MatNum, ::Option{<:Fees} = nothing)
    v, s, k = variance_skew_kurtosis_terms(r, w)
    return v - s + k
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the three scaled terms of [`VarianceSkewKurtosis`](@ref) `r` at the weights `w`, as the tuple `(v, s, k)`.

The functor of `r` returns `v - s + k`. The terms have the degrees of homogeneity 2, 3 and 4, so the sum has no single degree. [`adjusted_risk`](@ref) divides each term by its own degree, and so [`risk_contribution`](@ref) decomposes the sum exactly.

# Mathematical definition

```math
\\begin{align}
v &= s_{\\sigma^2}\\, \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,, \\\\
s &= s_{\\mathrm{sk}}\\, m_3(\\boldsymbol{w})\\,, \\\\
k &= s_{\\kappa}\\, m_4(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:m3_port])
  - $(math_dict[:m4_port])
  - ``s_{\\sigma^2}``, ``s_{\\mathrm{sk}}``, ``s_{\\kappa}``: The `settings.scale` of `vr`, `sk` and `kt`.

# Arguments

  - `r::VarianceSkewKurtosis`: Measure whose matrices are stated.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.

# Returns

  - `(v, s, k)::Tuple`: The scaled variance, third moment and fourth moment terms.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`adjusted_risk`](@ref)
  - [`homogeneity_degree`](@ref)
"""
function variance_skew_kurtosis_terms(r::VarianceSkewKurtosis, w::VecNum)
    w2 = LinearAlgebra.kron(w, w)
    return (r.vr(w) * r.vr.settings.scale,
            LinearAlgebra.dot(w, r.sk.sk, w2) * r.sk.settings.scale,
            LinearAlgebra.dot(w2, r.kt.kt, w2) * r.kt.settings.scale)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`VarianceSkewKurtosis`](@ref) `r` against prior result `pr`.

A child that names its own quantity keeps it, whether it states the quantity or defers it. The composite adds a variance, a skewness and a kurtosis term, so a caller who wants the three to describe one distribution names `pe` on the container and states nothing on the children.

# Algorithm

 1. Resolve each child with its own method of [`resolve_deferred_quantities`](@ref), giving `vr`, `sk` and `kt`.
 2. When `pe` is not `nothing`, fit it once against `pr` with [`fan_out_pe`](@ref), and fill each child slot that is still `nothing` from that one fit: `sigma` and its `chol` on `vr`, `sk` and `mu` on `sk`, and `kt` and `mu` on `kt`.
 3. Return the rebuilt measure with `pe = nothing`.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::VarianceSkewKurtosis, pr::AbstractPriorResult,
                                     slv = nothing)
    vr = resolve_deferred_quantities(r.vr, pr, slv)
    sk = resolve_deferred_quantities(r.sk, pr, slv)
    kt = resolve_deferred_quantities(r.kt, pr, slv)
    return fan_out_pe(r, r.pe, pr, (; vr = vr, sk = sk, kt = kt))
end
# The children carry their own slots, and `chol` is derived from `sigma`, so the fan-out of
# this measure is its own method of `fan_out_pe`.
function fan_out_pe(r::VarianceSkewKurtosis, pe::AbstractPriorEstimator,
                    pr::AbstractPriorResult, slots::NamedTuple)
    (; vr, sk, kt) = slots
    fitted = fit_deferred_quantity(pe, pr)
    # `chol` is derived from `sigma`, so it comes from the fan-out only when the fan-out
    # also supplies the `sigma` it factorises.
    chol = isnothing(vr.sigma) ? deferred_derived_quantity(fitted, :chol) : vr.chol
    vr = rebuild_with_slots(vr,
                            (; sigma = fan_out_slot(fitted, vr.sigma, :sigma), chol = chol))
    sk = rebuild_with_slots(sk,
                            (; sk = fan_out_slot(fitted, sk.sk, :sk),
                             mu = fan_out_slot(fitted, sk.mu, :mu), pe = nothing))
    kt = rebuild_with_slots(kt,
                            (; mu = fan_out_slot(fitted, kt.mu, :mu),
                             kt = fan_out_slot(fitted, kt.kt, :kt), pe = nothing))
    return rebuild_with_slots(r, (; vr = vr, sk = sk, kt = kt, pe = nothing))
end
# Deferrable slots — see `deferred_slots`. The three children carry their own, so the check
# recurses into them.
deferred_slots(r::VarianceSkewKurtosis) = (; vr = r.vr, sk = r.sk, kt = r.kt, pe = r.pe)
# The functor reads the co-moment tensors off the children directly, and neither child reads
# its own on a value-level route, so the container names both grandchildren by the path the
# caller wrote. `vr` reads its own `sigma`, so it is reached through `deferred_slots` — see
# `functor_slots`.
functor_slots(r::VarianceSkewKurtosis) = (; var"sk.sk" = r.sk.sk, var"kt.kt" = r.kt.kt)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`VarianceSkewKurtosis`](@ref) by fanning `pe` out over its three children, then threading `pr` into each of them.

The container holds no quantity of its own, so [`@fprop`](@ref) alone would reach the children and leave `pe` in place, and the measure would then hold a `pe` that it never applied. This method resolves first, which is the same order that [`set_risk_constraints!`](@ref) uses.

# Algorithm

 1. Resolve the container with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Thread `pr` into each child with its own method of [`factory`](@ref), giving `vr`, `sk` and `kt`.
 3. Build a new `VarianceSkewKurtosis` from the three children and the `settings` of `r`, with `pe = nothing`.

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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`VarianceSkewKurtosis`](@ref) `r` without its bounds, the bound of the container and the bounds of its three children.

A child holds the bound of its own term, so the method for a [`RiskMeasure`](@ref), which rebuilds only `settings`, would leave `vr.settings.ub`, `sk.settings.lb` and `kt.settings.ub` in the model. [`NearOptimalCentering`](@ref) solves its corner portfolios with this copy, so a child bound would constrain a corner that must be free. Every child and the container keep `rke` and `scale`, because the scale of a child weights its term.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`no_bounds_risk_measure`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function no_bounds_risk_measure(r::VarianceSkewKurtosis, flag::Any = nothing)
    return VarianceSkewKurtosis(;
                                settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                               scale = r.settings.scale),
                                vr = no_bounds_risk_measure(r.vr, flag),
                                sk = no_bounds_risk_measure(r.sk, flag),
                                kt = no_bounds_risk_measure(r.kt, flag), pe = r.pe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of [`VarianceSkewKurtosis`](@ref) `r` without its bounds and outside the risk of the objective.

The container takes `rke = false` and a unit `scale`, as the method for a [`RiskMeasure`](@ref) does. The children lose their bounds through [`no_bounds_risk_measure`](@ref) and keep their scales, because the scale of a child weights its term in the composite.

# Related

  - [`VarianceSkewKurtosis`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`no_bounds_risk_measure`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure(r::VarianceSkewKurtosis, flag::Any = nothing)
    return VarianceSkewKurtosis(;
                                settings = RiskMeasureSettings(; rke = false,
                                                               scale = one(r.settings.scale)),
                                vr = no_bounds_risk_measure(r.vr, flag),
                                sk = no_bounds_risk_measure(r.sk, flag),
                                kt = no_bounds_risk_measure(r.kt, flag), pe = r.pe)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Skewness) = WeightsReturnsFeesInput()
risk_input_kind(::VarianceSkewKurtosis) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`Skewness`](@ref) `r` supports precomputed-return evaluation.

The answer is [`weight_independent_target`](@ref) of `r.mu`. It is `true` when `mu` is `nothing` or a number, because the centre is then a function of the series alone. It is `false` for a vector, a [`VecScalar`](@ref) and a **Deferred Quantity**, whose centre needs the weights.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`Skewness`](@ref)
"""
supports_precomputed_returns(r::Skewness) = weight_independent_target(r.mu)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: the functor of [`VarianceSkewKurtosis`](@ref) reads the weights in all three terms, through the comoment matrices, and has no form that reads a series alone.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`VarianceSkewKurtosis`](@ref)
"""
supports_precomputed_returns(::VarianceSkewKurtosis) = false

export MaxRiskMeasureSettings, Skewness, VarianceSkewKurtosis
