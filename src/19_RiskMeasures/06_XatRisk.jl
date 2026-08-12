"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Value-at-Risk formulation algorithms.

All concrete and/or abstract types representing the formulation for computing Value-at-Risk (e.g., mixed-integer programming, distribution-based) should be subtypes of `ValueatRiskFormulation`.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`ValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
"""
abstract type ValueatRiskFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the Value-at-Risk formulation `alg` unchanged.

Identity pass-through for formulation types that do not depend on prior results.

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::ValueatRiskFormulation, args...; kwargs...)::ValueatRiskFormulation
    return alg
end
"""
    port_opt_view(r, args...)

Get a view or subset of a Value-at-Risk formulation for slicing.

Returns the formulation unchanged (for non-distribution types) or sliced (for distribution-based types). Used internally in hierarchical optimisation.

# Arguments

  - `r`: Value-at-Risk formulation.
  - `args...`: Additional arguments (index, etc.).

# Returns

  - Sliced or unchanged formulation.

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`DistributionValueatRisk`](@ref)
"""
function port_opt_view(r::ValueatRiskFormulation, ::Any, args...)::ValueatRiskFormulation
    return r
end
"""
$(DocStringExtensions.TYPEDEF)

Mixed-integer programming (MIP) formulation for Value-at-Risk.

`MIPValueatRisk` specifies bounds used in the binary variable formulation of Value-at-Risk within a `JuMP` optimisation model.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MIPValueatRisk(;
        b::Option{<:Number} = nothing,
        s::Option{<:Number} = nothing
    ) -> MIPValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `b` is not `nothing`: `b > 0`.
  - If `s` is not `nothing`: `s > 0`.
  - If both are not `nothing`: `b > s`.

# Examples

```jldoctest
julia> MIPValueatRisk()
MIPValueatRisk
  b ┼ nothing
  s ┴ nothing
```

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`ValueatRisk`](@ref)
  - [`Option`](@ref)
"""
@concrete struct MIPValueatRisk <: ValueatRiskFormulation
    """
    $(field_dict[:b_mip])
    """
    b
    """
    $(field_dict[:s_mip])
    """
    s
    function MIPValueatRisk(b::Option{<:Number}, s::Option{<:Number})
        bflag = !isnothing(b)
        sflag = !isnothing(s)
        if bflag
            assert_nonempty_gt0_finite_val(b, :b)
        end
        if sflag
            assert_nonempty_gt0_finite_val(s, :s)
        end
        if bflag && sflag
            @argcheck(b > s, DomainError("b must be greater than s, got b = $b, s = $s"))
        end
        return new{typeof(b), typeof(s)}(b, s)
    end
end
function MIPValueatRisk(; b::Option{<:Number} = nothing,
                        s::Option{<:Number} = nothing)::MIPValueatRisk
    return MIPValueatRisk(b, s)
end
"""
$(DocStringExtensions.TYPEDEF)

Distribution-based formulation for Value-at-Risk.

`DistributionValueatRisk` specifies a parametric distribution for computing Value-at-Risk analytically. The distribution parameters can be overridden by prior results during optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionValueatRisk(;
        mu::Option{<:MuSlot} = nothing,
        sigma::Option{<:SigmaSlot} = nothing,
        chol::Option{<:MatNum} = nothing,
        pe::Option{<:AbstractPriorEstimator} = nothing,
        dist::Distributions.Distribution = Distributions.Normal()
    ) -> DistributionValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is not `nothing`: `!isempty(mu)`.
  - If `sigma` is not `nothing`: `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.
  - If `chol` is not `nothing`: `!isempty(chol)`, and `sigma` is not `nothing`.

!!! warning

    `mu`, `sigma` and `chol` are stated independently, so nothing makes them agree with each other. A caller who wants one consistent set names `pe` alone and lets it fill all three from a single fit. A caller who states them by hand must make sure that they agree.

# Examples

```jldoctest
julia> DistributionValueatRisk()
DistributionValueatRisk
     mu ┼ nothing
  sigma ┼ nothing
   chol ┼ nothing
     pe ┼ nothing
   dist ┴ Distributions.Normal{Float64}: Distributions.Normal{Float64}(μ=0.0, σ=1.0)
```

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`ValueatRisk`](@ref)
  - [`MuSlot`](@ref)
  - [`SigmaSlot`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`Option`](@ref)
"""
@propagatable @concrete struct DistributionValueatRisk <: ValueatRiskFormulation
    """
    $(field_dict[:mu_dvar_slot])
    """
    mu
    """
    $(field_dict[:sigma_slot])
    """
    sigma
    """
    $(field_dict[:chol_slot])
    """
    chol
    """
    $(field_dict[:pe_rm])
    """
    pe
    """
    $(field_dict[:dist])
    """
    dist
    function DistributionValueatRisk(mu::Option{<:MuSlot}, sigma::Option{<:SigmaSlot},
                                     chol::Option{<:MatNum},
                                     pe::Option{<:AbstractPriorEstimator},
                                     dist::Distributions.Distribution)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        end
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
            assert_matrix_issquare(sigma, :sigma)
        end
        if isa(chol, MatNum)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
        end
        assert_derived_slot_has_source(chol, sigma, :chol, :sigma)
        return new{typeof(mu), typeof(sigma), typeof(chol), typeof(pe), typeof(dist)}(mu,
                                                                                      sigma,
                                                                                      chol,
                                                                                      pe,
                                                                                      dist)
    end
end
function DistributionValueatRisk(; mu::Option{<:MuSlot} = nothing,
                                 sigma::Option{<:SigmaSlot} = nothing,
                                 chol::Option{<:MatNum} = nothing,
                                 pe::Option{<:AbstractPriorEstimator} = nothing,
                                 dist::Distributions.Distribution = Distributions.Normal())::DistributionValueatRisk
    return DistributionValueatRisk(mu, sigma, chol, pe, dist)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`DistributionValueatRisk`](@ref) `alg` against prior result `pr`.

The measure carries three prior-derived fields, and `mu` and `sigma` are independent of each other, so it takes a `pe`: one fit fills every slot the caller left unstated. `chol` is derived from `sigma` and travels with it — a `sigma` that names its own estimator supplies the factorisation from that same fit, and never from the `pe`'s.

# Related

  - [`DistributionValueatRisk`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(alg::DistributionValueatRisk,
                                     pr::AbstractPriorResult)::DistributionValueatRisk
    sigma, chol = if isa(alg.sigma, DeferredQuantity)
        fitted = fit_deferred_quantity(alg.sigma, pr)
        deferred_quantity(fitted, :sigma), deferred_derived_quantity(fitted, :chol)
    else
        alg.sigma, alg.chol
    end
    mu = resolve_slot(alg.mu, :mu, pr)
    if isnothing(alg.pe)
        return DistributionValueatRisk(; mu = mu, sigma = sigma, chol = chol, pe = nothing,
                                       dist = alg.dist)
    end
    fitted = fit_deferred_quantity(alg.pe, pr)
    # `chol` is derived from `sigma`, so it comes from the fan-out only when the fan-out
    # also supplies the `sigma` it factorises. Read before `sigma` is filled.
    chol = isnothing(sigma) ? deferred_derived_quantity(fitted, :chol) : chol
    return DistributionValueatRisk(; mu = fan_out_slot(fitted, mu, :mu),
                                   sigma = fan_out_slot(fitted, sigma, :sigma), chol = chol,
                                   pe = nothing, dist = alg.dist)
end
# Deferrable slots — see `deferred_slots`. `chol` is derived and never defers on its own.
function deferred_slots(alg::DistributionValueatRisk)
    return (; mu = alg.mu, sigma = alg.sigma, pe = alg.pe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`DistributionValueatRisk`](@ref) by resolving its **Deferred Quantities**, then falling back to the prior result for whatever is still unstated.

`sigma` and `chol` are selected **as a pair** ([`sigma_chol_selector`](@ref)), not field by field: a stated `sigma` with no factor must not be paired with the prior's, which factorises a different matrix.

# Related

  - [`DistributionValueatRisk`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`sigma_chol_selector`](@ref)
"""
function factory(alg::DistributionValueatRisk, pr::AbstractPriorResult, args...;
                 kwargs...)::DistributionValueatRisk
    alg = resolve_deferred_quantities(alg, pr)
    sigma, chol = sigma_chol_selector(alg.sigma, alg.chol, pr)
    return DistributionValueatRisk(; mu = sel(alg.mu, pr.mu), sigma = sigma, chol = chol,
                                   pe = nothing, dist = alg.dist)
end
function port_opt_view(alg::DistributionValueatRisk, i, args...)::DistributionValueatRisk
    mu = nothing_scalar_array_view(alg.mu, i)
    sigma = nothing_scalar_array_view(alg.sigma, i)
    chol = isnothing(alg.chol) ? nothing : view(alg.chol, :, i)
    return DistributionValueatRisk(; mu = mu, sigma = sigma, chol = chol, pe = alg.pe,
                                   dist = alg.dist)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Value-at-Risk (VaR) risk measure.

`ValueatRisk` quantifies the maximum expected loss at a given confidence level `alpha` over a specified time horizon. It can be computed using empirical quantiles (weighted or unweighted) or via a parametric distribution.

# Mathematical definition

Let ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal`` be the portfolio returns vector and ``x_{(k)}`` the ``k``-th order statistic (``k``-th smallest value). The empirical VaR at significance level ``\\alpha`` is:

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) &= -x_{(\\lceil \\alpha T \\rceil)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Value-at-Risk at significance level ``\\alpha``.
  - ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal``: Portfolio returns vector.
  - ``x_{(k)}``: ``k``-th order statistic (``k``-th smallest value) of ``\\boldsymbol{x}``.
  - ``\\alpha``: Significance level (e.g., ``\\alpha = 0.05`` for 95% VaR).
  - $(math_dict[:T])

For observation-weighted samples with weight vector ``\\boldsymbol{w}`` summing to ``S_w``, VaR is the ``\\alpha S_w``-quantile of the weighted empirical distribution.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing,
        alg::ValueatRiskFormulation = MIPValueatRisk()
    ) -> ValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ValueatRisk)(x::VecNum)

Computes the Value-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ValueatRisk()
ValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┼ nothing
       alg ┼ MIPValueatRisk
           │   b ┼ nothing
           │   s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
"""
@propagatable @concrete struct ValueatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:oow])
    """
    @pprop w
    """
    $(field_dict[:alg])
    """
    @fprop @vprop alg
    function ValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                         w::Option{<:ObsWeights}, alg::ValueatRiskFormulation)
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings, alpha,
                                                                            w, alg)
    end
end
function ValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing,
                     alg::ValueatRiskFormulation = MIPValueatRisk())::ValueatRisk
    return ValueatRisk(settings, alpha, w, alg)
end
function (r::ValueatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    return -partialsort(x, ceil(Int, r.alpha * length(x)))
end
function (r::ValueatRisk{<:Any, <:Any, <:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    return -sorted_x[idx]
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Value-at-Risk Range risk measure.

`ValueatRiskRange` computes the difference between the lower-tail Value-at-Risk (at level `alpha`) and the upper-tail Value-at-Risk (at level `beta`), measuring the spread between downside and upside tail risks.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) - \\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})\\,,\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: Value-at-Risk Range.
  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Lower-tail loss quantile.
  - ``\\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})``: Upper-tail gain quantile.
  - ``\\boldsymbol{x}``: Portfolio returns vector.
  - ``\\alpha``: Lower-tail significance level.
  - ``\\beta``: Upper-tail significance level.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        beta::Number = 0.05,
        w::Option{<:ObsWeights} = nothing,
        alg::ValueatRiskFormulation = MIPValueatRisk()
    ) -> ValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < beta < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ValueatRiskRange)(x::VecNum)

Computes the VaR Range of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ValueatRiskRange()
ValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
      beta ┼ Float64: 0.05
         w ┼ nothing
       alg ┼ MIPValueatRisk
           │   b ┼ nothing
           │   s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ValueatRisk`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
"""
@propagatable @concrete struct ValueatRiskRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:beta])
    """
    beta
    """
    $(field_dict[:oow])
    """
    @pprop w
    """
    $(field_dict[:alg])
    """
    @fprop @vprop alg
    function ValueatRiskRange(settings::RiskMeasureSettings, alpha::Number, beta::Number,
                              w::Option{<:ObsWeights}, alg::ValueatRiskFormulation)
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w), typeof(alg)}(settings,
                                                                                          alpha,
                                                                                          beta,
                                                                                          w,
                                                                                          alg)
    end
end
function ValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          alpha::Number = 0.05, beta::Number = 0.05,
                          w::Option{<:ObsWeights} = nothing,
                          alg::ValueatRiskFormulation = MIPValueatRisk())::ValueatRiskRange
    return ValueatRiskRange(settings, alpha, beta, w, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Recurse into the formulation held by [`ValueatRisk`](@ref) `r`, so that a **Deferred Quantity** on a [`DistributionValueatRisk`](@ref) resolves on the `JuMP` path too.

The `JuMP` model builders never call [`factory`](@ref) on a risk measure, so [`set_risk_constraints!`](@ref) is the second resolution point. It resolves the wrapper, and the quantities live one level down on `alg`. [`MIPValueatRisk`](@ref) defers nothing, so the recursion is the identity for it.

# Related

  - [`ValueatRisk`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function resolve_deferred_quantities(r::ValueatRisk, pr::AbstractPriorResult)::ValueatRisk
    return ValueatRisk(; settings = r.settings, alpha = r.alpha, w = r.w,
                       alg = resolve_deferred_quantities(r.alg, pr))
end
# Deferrable slots — see `deferred_slots`. The formulation carries them, so the check
# recurses into it.
deferred_slots(r::ValueatRisk) = (; alg = r.alg)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Recurse into the formulation held by [`ValueatRiskRange`](@ref) `r`. See [`resolve_deferred_quantities(r::ValueatRisk, pr::AbstractPriorResult)`](@ref).

# Related

  - [`ValueatRiskRange`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
function resolve_deferred_quantities(r::ValueatRiskRange,
                                     pr::AbstractPriorResult)::ValueatRiskRange
    return ValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta,
                            w = r.w, alg = resolve_deferred_quantities(r.alg, pr))
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::ValueatRiskRange) = (; alg = r.alg)
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, Nothing})(x::VecNum)
    x = copy(x)
    loss = -partialsort!(x, ceil(Int, r.alpha * length(x)))
    gain = -partialsort!(x, ceil(Int, r.beta * length(x)); rev = true)
    return loss - gain
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    loss = -sorted_x[idx]

    sorted_x = reverse!(sorted_x)
    sorted_w = reverse!(sorted_w)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.beta)
    idx = ifelse(idx > length(x), idx - 1, idx)
    gain = -sorted_x[idx]
    return loss - gain
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Drawdown-at-Risk (DaR) risk measure.

`DrawdownatRisk` quantifies the maximum drawdown not exceeded at a given confidence level `alpha`. It operates on absolute drawdowns computed from the portfolio returns series.

# Mathematical definition

Define the cumulative wealth process and absolute drawdown at time ``t``:

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} x_s\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])

The Drawdown-at-Risk at level ``\\alpha`` is the ``\\lceil \\alpha T \\rceil``-th smallest (most extreme) drawdown:

```math
\\begin{align}
\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x}) &= -d_{(\\lceil \\alpha T \\rceil)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x})``: Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:dtdd])
  - ``d_{(k)}``: ``k``-th order statistic (sorted ascending) of the drawdown series.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing,
        b::Option{<:Number} = nothing,
        s::Option{<:Number} = nothing
    ) -> DrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.
  - If `b` is not `nothing`: `b > 0`.
  - If `s` is not `nothing`: `s > 0`.
  - If both `b` and `s` are not `nothing`: `b > s`.

# Functor

    (r::DrawdownatRisk)(x::VecNum)

Computes the Drawdown-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DrawdownatRisk()
DrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┼ nothing
         b ┼ nothing
         s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`RelativeDrawdownatRisk`](@ref)
  - [`drawdown_at_risk`](@ref)
"""
@propagatable @concrete struct DrawdownatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:oow])
    """
    @pprop w
    """
    $(field_dict[:b_mip])
    """
    b
    """
    $(field_dict[:s_mip])
    """
    s
    function DrawdownatRisk(settings::RiskMeasureSettings, alpha::Number,
                            w::Option{<:ObsWeights}, b::Option{<:Number},
                            s::Option{<:Number})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        bflag = !isnothing(b)
        sflag = !isnothing(s)
        if bflag
            assert_nonempty_gt0_finite_val(b, :b)
        end
        if sflag
            assert_nonempty_gt0_finite_val(s, :s)
        end
        if bflag && sflag
            @argcheck(b > s, DomainError("b must be greater than s, got b = $b, s = $s"))
        end
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(b), typeof(s)}(settings,
                                                                                     alpha,
                                                                                     w, b,
                                                                                     s)
    end
end
function DrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing,
                        b::Option{<:Number} = nothing,
                        s::Option{<:Number} = nothing)::DrawdownatRisk
    return DrawdownatRisk(settings, alpha, w, b, s)
end
"""
    absolute_drawdown_vec(x::VecNum) -> Vector

Compute the absolute drawdown series for a single-asset return vector.

Each element of the result is the difference between the current cumulative return and its running maximum (always ≤ 0).

The running maximum starts at zero, so the drawdown is measured against the initial portfolio value rather than against the first observation.

`x` is read, never written: the accumulator and the running peak are carried in scalars, so any `AbstractVector` works — a column view, a range, an immutable array.

# Arguments

  - `x::VecNum`: Return series vector.

# Returns

  - `Vector`: Drawdown vector of the same length as `x`.

# Related

  - [`absolute_drawdown_arr`](@ref)
  - [`DrawdownatRisk`](@ref)
"""
function absolute_drawdown_vec(x::VecNum)
    T = eltype(x)
    dd = similar(x, T)
    cs = peak = zero(T)
    for (idx, xi) in pairs(x)
        cs += xi
        peak = ifelse(cs > peak, cs, peak)
        dd[idx] = cs - peak
    end
    return dd
end
"""
    drawdown_at_risk(dd::VecNum, alpha::Real, ::Nothing) -> Number
    drawdown_at_risk(dd::VecNum, alpha::Real, w::VecNum) -> Number

Aggregate a drawdown series into its Drawdown-at-Risk at level `alpha`.

This is the shared aggregation kernel behind [`DrawdownatRisk`](@ref) and [`RelativeDrawdownatRisk`](@ref): the two measures differ only in the drawdown series they feed it ([`absolute_drawdown_vec`](@ref) and [`relative_drawdown_vec`](@ref) respectively), so the tail selection lives here once.

`dd` is **consumed in place** — the unweighted method reorders it via `partialsort!`. Callers pass a freshly computed drawdown vector.

Dispatch on the third argument selects the weighting scheme, so callers resolve observation weights with [`get_observation_weights`](@ref) and let dispatch do the rest.

  - `::Nothing`: unweighted, the `alpha`-quantile of the drawdown series by rank.
  - `w::VecNum`: weighted, the drawdown at which the cumulative observation weight first reaches `alpha`.

# Arguments

  - `dd::VecNum`: Drawdown series, all entries ≤ 0. Consumed in place.
  - `alpha::Real`: Significance level, `0 < alpha < 1`.
  - `w`: Resolved observation weights, or `nothing` for the unweighted aggregation.

# Returns

  - `Number`: Drawdown-at-Risk, returned as a positive loss.

# Related

  - [`DrawdownatRisk`](@ref)
  - [`RelativeDrawdownatRisk`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`conditional_drawdown_at_risk`](@ref)
"""
function drawdown_at_risk(dd::VecNum, alpha::Real, ::Nothing)
    return -partialsort!(dd, ceil(Int, alpha * length(dd)))
end
function drawdown_at_risk(dd::VecNum, alpha::Real, w::VecNum)
    sw = sum(w)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * alpha)
    idx = ifelse(idx > length(dd), idx - 1, idx)
    return -sorted_dd[idx]
end
function (r::DrawdownatRisk)(x::VecNum)
    return drawdown_at_risk(absolute_drawdown_vec(x), r.alpha,
                            get_observation_weights(r.w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Drawdown-at-Risk risk measure for hierarchical optimisation.

`RelativeDrawdownatRisk` quantifies the maximum relative (compounded) drawdown not exceeded at a given confidence level `alpha`. It operates on relative drawdowns computed from the portfolio returns series.

# Mathematical definition

Define the compounded wealth process and relative drawdown at time ``t``:

```math
\\begin{align}
C_t &= \\prod_{s=1}^{t} (1 + x_s)\\,, \\\\
rd_t &= \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1 \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:Ct])
  - $(math_dict[:rdt])

The Relative Drawdown-at-Risk at level ``\\alpha`` is:

```math
\\begin{align}
\\mathrm{RLDaR}_{\\alpha}(\\boldsymbol{x}) &= -rd_{(\\lceil \\alpha T \\rceil)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLDaR}_{\\alpha}(\\boldsymbol{x})``: Relative Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:rdt])
  - ``rd_{(k)}``: ``k``-th order statistic (sorted ascending) of the relative drawdown series.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativeDrawdownatRisk)(x::VecNum)

Computes the Relative Drawdown-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeDrawdownatRisk()
RelativeDrawdownatRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
  - [`drawdown_at_risk`](@ref)
"""
@propagatable @concrete struct RelativeDrawdownatRisk <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativeDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                    alpha::Number, w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Number = 0.05,
                                w::Option{<:ObsWeights} = nothing)::RelativeDrawdownatRisk
    return RelativeDrawdownatRisk(settings, alpha, w)
end
"""
    relative_drawdown_vec(x)

Compute the relative drawdown vector for a vector of portfolio returns.

Returns the relative drawdown at each time step, computed as the current portfolio value relative to its running maximum.

The running maximum starts at one, so the drawdown is measured against the initial portfolio value rather than against the first observation.

`x` is read, never written: the compounding factor and the running peak are carried in scalars, so any `AbstractVector` works — a column view, a range, an immutable array.

# Arguments

  - `x`: Vector of portfolio returns.

# Returns

  - Relative drawdown vector.

# Related

  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_arr`](@ref)
"""
function relative_drawdown_vec(x::VecNum)
    T = eltype(x)
    dd = similar(x, T)
    cp = peak = one(T)
    for (idx, xi) in pairs(x)
        cp *= xi + one(T)
        peak = ifelse(cp > peak, cp, peak)
        dd[idx] = cp / peak - one(peak)
    end
    return dd
end
function (r::RelativeDrawdownatRisk)(x::VecNum)
    return drawdown_at_risk(relative_drawdown_vec(x), r.alpha,
                            get_observation_weights(r.w, x))
end

"""
    const CholRM = Union{<:Variance, <:StandardDeviation, <:DistributionValueatRisk}

Union of risk measures that support Cholesky-factor-based computation.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`DistributionValueatRisk`](@ref)
"""
const CholRM = Union{<:Variance, <:StandardDeviation, <:DistributionValueatRisk}

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::ValueatRisk) = NetReturnsInput()
risk_input_kind(::ValueatRiskRange) = NetReturnsInput()
risk_input_kind(::DrawdownatRisk) = NetReturnsInput()
risk_input_kind(::RelativeDrawdownatRisk) = NetReturnsInput()

export MIPValueatRisk, DistributionValueatRisk, ValueatRisk, ValueatRiskRange,
       DrawdownatRisk, RelativeDrawdownatRisk
