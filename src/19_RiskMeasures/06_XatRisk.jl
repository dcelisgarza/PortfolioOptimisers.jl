"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Value-at-Risk formulation algorithms.

The formulation selects the **estimand**, not only the encoding. [`MIPValueatRisk`](@ref) reports the empirical quantile of the sample, and [`DistributionValueatRisk`](@ref) reports the quantile of a fitted parametric distribution. The two are different numbers on the same data, and a measure that holds one reports that one in its optimisation model and in its functor alike.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the lower-tail z-score for a parametric VaR at significance level `alpha`.

Returns the complementary quantile for Normal and scaled Student-t distributions, and the
closed-form expression for the Laplace distribution.

# Arguments

  - `dist`: Distribution instance (Normal, TDist, or Laplace).
  - `alpha::Number`: Significance level.

# Returns

  - `z::Number`: Lower-tail z-score for the parametric VaR.

# Related

  - [`compute_value_at_risk_cz`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function compute_value_at_risk_z(dist::Distributions.Normal, alpha::Number)
    return Distributions.cquantile(dist, alpha)
end
function compute_value_at_risk_z(dist::Distributions.TDist, alpha::Number)
    d = StatsAPI.dof(dist)
    @argcheck(d > 2, DomainError(d, "degrees of freedom must be greater than 2"))
    return Distributions.cquantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_z(::Distributions.Laplace, alpha::Number)
    return -log(2 * alpha) / sqrt(2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the upper-tail z-score for a parametric VaR at significance level `alpha`.

Used for the high (upper) bound in VaR range constraints. Returns the lower quantile for
Normal and scaled Student-t distributions, and the closed-form expression for Laplace.

# Arguments

  - `dist`: Distribution instance (Normal, TDist, or Laplace).
  - `alpha::Number`: Significance level.

# Returns

  - `z::Number`: Upper-tail z-score for the parametric VaR.

# Related

  - [`compute_value_at_risk_z`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function compute_value_at_risk_cz(dist::Distributions.Normal, alpha::Number)
    return Statistics.quantile(dist, alpha)
end
function compute_value_at_risk_cz(dist::Distributions.TDist, alpha::Number)
    d = StatsAPI.dof(dist)
    @argcheck(d > 2, DomainError(d, "degrees of freedom must be greater than 2"))
    return Statistics.quantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_cz(::Distributions.Laplace, alpha::Number)
    return -log(2 * (one(alpha) - alpha)) / sqrt(2)
end
"""
$(DocStringExtensions.TYPEDEF)

Mixed-integer programming (MIP) formulation for Value-at-Risk.

`MIPValueatRisk` specifies bounds used in the binary variable formulation of Value-at-Risk within a `JuMP` optimisation model. It reports the **empirical** quantile, which is the value the functor of the risk measure computes directly.

# Mathematical definition

One binary indicator per observation flags an exceedance, the cardinality constraint caps how many observations may be flagged, and the big-M constant `b` relaxes the bound on a flagged one. The risk is minimised, so it settles on the smallest value that leaves no more than the permitted number of exceedances.

```math
\\begin{align}
\\underset{r,\\, \\boldsymbol{z}}{\\min} \\quad & r\\\\
\\text{s.t.} \\quad & r \\geq -x_t - b z_t\\,, \\quad t = 1,\\ldots,T\\\\
\\quad & \\sum_{t=1}^{T} z_t \\leq \\left(\\alpha - s\\right) T\\\\
\\quad & z_t \\in \\left\\{0,\\, 1\\right\\}\\,.
\\end{align}
```

Where:

  - ``r``: Value-at-Risk variable.
  - ``x_t``: Net portfolio return at observation ``t``.
  - ``z_t``: Binary exceedance indicator at observation ``t``.
  - ``b``: Big-M constant, the `b` field.
  - ``s``: Cardinality slack, the `s` field.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

Observation weights replace both counts by their weighted sums, so the cardinality constraint reads ``\\boldsymbol{w}_{o}^\\intercal \\boldsymbol{z} \\leq (\\alpha - s) \\sum_{t=1}^{T} w_{o,t}``.

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

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.3.
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

`DistributionValueatRisk` specifies a parametric distribution for computing Value-at-Risk analytically. The distribution parameters can be overridden by prior results during optimisation. This is a different estimand from the empirical quantile that [`MIPValueatRisk`](@ref) reports, so the risk measure that holds it reports the parametric value in the optimisation model and in the functor alike.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{w}) &= -\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} + z_{\\alpha} \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:mu_er])
  - ``\\mathbf{\\Sigma}``: `N × N` covariance matrix.
  - ``z_{\\alpha}``: Lower-tail z-score of the standardised `dist` at ``\\alpha``, from [`compute_value_at_risk_z`](@ref).
  - $(math_dict[:alpha_rm])

The optimisation model states the standard deviation as a second-order cone over ``\\mathbf{G} \\boldsymbol{w}``, where ``\\mathbf{G}`` factorises ``\\mathbf{\\Sigma}``. `mu` and `sigma` fall back to the prior's own, so [`factory`](@ref) fills them before either the model or the functor reads them.

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
  - If `chol` is not `nothing`: `!isempty(chol)`, and `sigma` is a matrix rather than `nothing` or a **Deferred Quantity**.

!!! warning

    `mu`, `sigma` and `chol` are stated independently, so nothing makes them agree with each other. A caller who wants one consistent set names `pe` alone and lets it fill all three from a single fit. A caller who states them by hand must make sure that they agree.

## View parameters

`DistributionValueatRisk` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `mu` is sliced to the selected assets. A **Deferred Quantity** passes through unsliced, and then resolves on the subset.
  - `sigma` is sliced to the selected assets. A stated matrix is sliced on **both** axes. A **Deferred Quantity** passes through unsliced, and then resolves on the subset.
  - `chol` is sliced on its **columns** alone. Its rows index the factorisation, which the asset selection does not address.
  - `pe` and `dist` are carried through unchanged. `dist` describes the standardised loss, so it carries no asset axis.

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
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.3.
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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `alg`: Recursively viewed via [`port_opt_view`](@ref).

# Functor

    (r::ValueatRisk)(x::VecNum)
    (r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk})(w::VecNum, X, fees)

`alg` selects the quantity, and each method reports the one that its own `JuMP` formulation builds. The first computes the empirical quantile of a portfolio returns vector, weighted by `w` when the measure states observation weights. The second computes the parametric quantile from the moments that [`DistributionValueatRisk`](@ref) holds, so it takes the asset weights instead. `X` and `fees` are unused, because the model's terms are the prior's moments and no return series enters them.

## Arguments

  - `x::VecNum`: Portfolio returns vector.
  - `w::VecNum`: Asset weights vector.

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.3.
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
# The empirical order statistic is the `MIPValueatRisk` estimand, so the two functors
# below name that formulation. Leaving `alg` free makes them overlap the parametric
# method further down, which no rule of specificity can order.
function (r::ValueatRisk{<:Any, <:Any, Nothing, <:MIPValueatRisk})(x::VecNum)
    return -partialsort(x, ceil(Int, r.alpha * length(x)))
end
function (r::ValueatRisk{<:Any, <:Any, <:ObsWeights, <:MIPValueatRisk})(x::VecNum)
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
# The parametric formulation is a different estimand from the empirical order statistic
# above, and the `JuMP` model builds the parametric one. The functor must report the same
# number, so it reads `mu` and `sigma` off the formulation, which `factory` fills from the
# prior, exactly as the model does. `X` and `fees` are unused: the model's terms are the
# prior's moments and no return series enters it.
function (r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk})(w::VecNum,
                                                                          ::Any = nothing,
                                                                          ::Any = nothing)
    alg = r.alg
    z = compute_value_at_risk_z(alg.dist, r.alpha)
    return -LinearAlgebra.dot(alg.mu, w) + z * sqrt(LinearAlgebra.dot(w, alg.sigma, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Value-at-Risk Range risk measure.

`ValueatRiskRange` evaluates the Value-at-Risk at level `alpha` on the portfolio returns and the Value-at-Risk at level `beta` on the negated portfolio returns, then sums the two to give the total spread between the downside and the upside tail.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) + \\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: Value-at-Risk Range.
  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Lower-tail loss quantile.
  - ``\\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})``: Upper-tail gain quantile.
  - $(math_dict[:xret])
  - ``\\alpha``: Lower-tail significance level.
  - ``\\beta``: Upper-tail significance level.

$(math_dict[:negated_upper_tail])

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `alg`: Recursively viewed via [`port_opt_view`](@ref).

# Functor

    (r::ValueatRiskRange)(x::VecNum)
    (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk})(w::VecNum, X, fees)

`alg` selects the quantity, as it does for [`ValueatRisk`](@ref). The first method computes the sum of the two empirical tail quantiles of a portfolio returns vector. It holds the upper tail in the negated convention of [`ValueatRisk`](@ref), so it writes the sum as `loss - gain`. The second computes the parametric range, in which the two legs share one ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}`` term that cancels, leaving the spread of the two z-scores over one standard deviation.

## Arguments

  - `x::VecNum`: Portfolio returns vector.
  - `w::VecNum`: Asset weights vector.

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.3.
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
# Deferrable slots — see `deferred_slots`. The formulation carries them, so both the check and
# the derived recursion in `resolve_deferred_quantities` reach them through `alg`.
# `MIPValueatRisk` defers nothing, so the recursion is the identity for it.
deferred_slots(r::ValueatRisk) = (; alg = r.alg)
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::ValueatRiskRange) = (; alg = r.alg)
# Tail decomposition — see `range_tails`. Declared for the MIP formulation only: the
# `DistributionValueatRisk` range shares one `g_var` cone between its two tails, so it fuses
# rather than duplicating, and has no two sub-models to build.
function range_tails(r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:MIPValueatRisk})
    settings = RiskMeasureSettings(; rke = false)
    return (;
            loss = ValueatRisk(; settings = settings, alpha = r.alpha, w = r.w,
                               alg = r.alg),
            gain = ValueatRisk(; settings = settings, alpha = r.beta, w = r.w, alg = r.alg))
end
# The empirical order statistic is the `MIPValueatRisk` estimand, so the two functors
# below name that formulation. Leaving `alg` free makes them overlap the parametric
# method further down, which no rule of specificity can order.
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, Nothing, <:MIPValueatRisk})(x::VecNum)
    x = copy(x)
    loss = -partialsort!(x, ceil(Int, r.alpha * length(x)))
    gain = -partialsort!(x, ceil(Int, r.beta * length(x)); rev = true)
    return loss - gain
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:ObsWeights, <:MIPValueatRisk})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    loss = -sorted_x[idx]

    # Reverse the **permutation**, never the views. `sorted_x` and `sorted_w` are views, so
    # `reverse!` on them writes through into the caller's `x` and into `r.w` —
    # `get_observation_weights` hands back the stored weights object itself, so the measure
    # would permute its own configuration. `order` was just allocated by `sortperm`, so it is
    # ours to mutate, and the element sequence read below is identical.
    reverse!(order)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.beta)
    idx = ifelse(idx > length(x), idx - 1, idx)
    gain = -sorted_x[idx]
    return loss - gain
end
# The parametric twin of the two functors above. The model's two legs share one mean term,
# which cancels in their difference, so the range is the spread of the two z-scores over one
# standard deviation.
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk})(w::VecNum,
                                                                                      ::Any = nothing,
                                                                                      ::Any = nothing)
    alg = r.alg
    z_l = compute_value_at_risk_z(alg.dist, r.alpha)
    z_h = compute_value_at_risk_cz(alg.dist, r.beta)
    return (z_l - z_h) * sqrt(LinearAlgebra.dot(w, alg.sigma, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Drawdown-at-Risk (DaR) risk measure.

`DrawdownatRisk` quantifies the maximum drawdown not exceeded at a given confidence level `alpha`. It operates on absolute drawdowns computed from the portfolio returns series. Its optimisation model is the mixed-integer programme of [`MIPValueatRisk`](@ref) applied to the drawdown series rather than to the return series, so `b` and `s` carry the same meaning here as they do there.

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

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:cajas2025]) Section 7.2.4.3.
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
\\mathrm{RDaR}_{\\alpha}(\\boldsymbol{x}) &= -rd_{(\\lceil \\alpha T \\rceil)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RDaR}_{\\alpha}(\\boldsymbol{x})``: Relative Drawdown-at-Risk at level ``\\alpha``.
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

# References

  - $(ref_dict[:cdar])
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
# The distribution formulation reads the prior's moments against `w`, so it takes the
# weights rather than the net return series.
function risk_input_kind(::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk})
    return WeightsReturnsFeesInput()
end
function risk_input_kind(::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                            <:DistributionValueatRisk})
    return WeightsReturnsFeesInput()
end
# A parametric quantile is read off the weights against a pinned pair of moments, so there
# is nothing a precomputed return series can supply. Same shape as `WeightsTracking`.
function supports_precomputed_returns(::ValueatRisk{<:Any, <:Any, <:Any,
                                                    <:DistributionValueatRisk})
    return false
end
function supports_precomputed_returns(::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                         <:DistributionValueatRisk})
    return false
end
risk_input_kind(::DrawdownatRisk) = NetReturnsInput()
risk_input_kind(::RelativeDrawdownatRisk) = NetReturnsInput()

export MIPValueatRisk, DistributionValueatRisk, ValueatRisk, ValueatRiskRange,
       DrawdownatRisk, RelativeDrawdownatRisk
