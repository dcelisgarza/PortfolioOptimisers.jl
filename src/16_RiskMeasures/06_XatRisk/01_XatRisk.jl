"""
$(DocStringExtensions.TYPEDEF)

Selects whether a Value-at-Risk measure computes its quantile from the sample or from a parametric distribution.

The formulation selects the estimand, not only the encoding. [`MIPValueatRisk`](@ref) reports the empirical quantile of the sample, and [`DistributionValueatRisk`](@ref) reports the quantile of a parametric distribution fitted to the prior's moments. The two are different numbers on the same data. A measure that holds one formulation reports that number in its optimisation model and in its functor alike.

All concrete subtypes should subtype `ValueatRiskFormulation`.

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

A formulation that reads nothing from the prior has nothing to fill. [`DistributionValueatRisk`](@ref) reads the prior's moments, and it has its own method.

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::ValueatRiskFormulation, args...; kwargs...)::ValueatRiskFormulation
    return alg
end
"""
    port_opt_view(r::ValueatRiskFormulation, i, args...) -> ValueatRiskFormulation

Return the Value-at-Risk formulation `r` unchanged for the asset selection `i`.

A formulation that holds no per-asset data has nothing to slice. [`DistributionValueatRisk`](@ref) holds the moments of each asset, and it has its own method, which slices them.

# Arguments

  - `r`: Value-at-Risk formulation.
  - `i`: Asset selection, which this method does not read.
  - `args...`: More arguments, which this method does not read.

# Returns

  - `r`, unchanged.

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`DistributionValueatRisk`](@ref)
"""
function port_opt_view(r::ValueatRiskFormulation, ::Any, args...)::ValueatRiskFormulation
    return r
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the loss-tail z-score of the parametric Value-at-Risk at significance level `alpha`.

The z-score is the ``1 - \\alpha`` quantile of the distribution, the loss that the standardised return exceeds with probability ``\\alpha``.

# Mathematical definition

```math
\\begin{align}
z_{\\alpha} &= F^{-1}(1 - \\alpha)\\,.
\\end{align}
```

Each method reads ``F`` from its own distribution:

  - `Normal`: ``F`` is the cumulative distribution function of `dist` as given, so the default `Normal()` is standardised.
  - `TDist`: ``F`` is the Student-t distribution with ``\\nu`` degrees of freedom, scaled to unit variance, so ``z_{\\alpha} = t_{\\nu}^{-1}(1 - \\alpha) \\sqrt{(\\nu - 2)/\\nu}``.
  - `Laplace`: ``F`` is the Laplace distribution with zero mean and unit variance, whatever the parameters of `dist`. Its closed form is below.

```math
\\begin{align}
z_{\\alpha} &= \\begin{cases} -\\dfrac{\\ln(2 \\alpha)}{\\sqrt{2}}\\,, & \\alpha \\leq \\dfrac{1}{2}\\,,\\\\ \\dfrac{\\ln(2 (1 - \\alpha))}{\\sqrt{2}}\\,, & \\alpha > \\dfrac{1}{2}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``z_{\\alpha}``: Loss-tail z-score.
  - ``F``: Cumulative distribution function of the standardised return.
  - ``t_{\\nu}``: Cumulative distribution function of the Student-t distribution with ``\\nu`` degrees of freedom.
  - ``\\nu``: Degrees of freedom of `dist`.
  - $(math_dict[:alpha_rm])

``z_{\\alpha}`` is positive for ``\\alpha < 1/2`` on each of the three distributions.

# Arguments

  - `dist`: Distribution of the standardised return, a `Normal`, a `TDist` or a `Laplace`.
  - `alpha::Number`: Significance level.

# Validation

  - `TDist`: the degrees of freedom are greater than 2, so that the variance is finite. Otherwise a `DomainError` is thrown.

# Returns

  - `z::Number`: Loss-tail z-score.

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
    return if 2 * alpha <= one(alpha)
        -log(2 * alpha) / sqrt(2)
    else
        log(2 * (one(alpha) - alpha)) / sqrt(2)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the gain-tail z-score of the parametric Value-at-Risk at significance level `alpha`.

The z-score is the ``\\alpha`` quantile of the distribution. The gain tail of [`ValueatRiskRange`](@ref) reads it.

# Mathematical definition

```math
\\begin{align}
z^{c}_{\\alpha} &= F^{-1}(\\alpha)\\,.
\\end{align}
```

Each method reads ``F`` from the same distribution as [`compute_value_at_risk_z`](@ref). The three distributions are symmetric about zero when standardised, so ``z^{c}_{\\alpha} = -z_{\\alpha}``, and the `Laplace` method returns that value.

Where:

  - ``z^{c}_{\\alpha}``: Gain-tail z-score.
  - ``z_{\\alpha}``: Loss-tail z-score, from [`compute_value_at_risk_z`](@ref).
  - ``F``: Cumulative distribution function of the standardised return.
  - $(math_dict[:alpha_rm])

# Arguments

  - `dist`: Distribution of the standardised return, a `Normal`, a `TDist` or a `Laplace`.
  - `alpha::Number`: Significance level.

# Validation

  - `TDist`: the degrees of freedom are greater than 2, so that the variance is finite. Otherwise a `DomainError` is thrown.

# Returns

  - `z::Number`: Gain-tail z-score.

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
function compute_value_at_risk_cz(dist::Distributions.Laplace, alpha::Number)
    return -compute_value_at_risk_z(dist, alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Value-at-Risk as the empirical quantile of the sample, through a mixed-integer programme.

The programme has one binary indicator per observation. The functor of the measure that holds this formulation computes the same order statistic directly, so the model and the functor report one number. The default big-M constant, ``b = 1000``, exceeds any loss of a return series, so the programme is exact with it.

A solver accepts an indicator that is within its integrality tolerance ``\\varepsilon`` of zero as zero. The exceedance row of that observation then holds for a risk as low as ``\\ell_{t} - b \\varepsilon``. So the risk that the model reports can be smaller than the order statistic of the weights that it returns, by up to ``b \\varepsilon``, and those weights need not be optimal. With the default ``b = 1000`` and a tolerance of ``10^{-6}`` the gap can reach ``10^{-3}``, which is large next to the scale of daily returns. A big-M constant that fits the scale of the losses, or a tighter integrality tolerance of the solver, makes the gap smaller.

# Mathematical definition

Each indicator exempts one observation from the bound on the risk, and the cardinality row caps the weight of the exempt observations. The programme minimises ``r``, so ``r`` falls to the largest loss that no indicator exempts:

```math
\\begin{align}
\\underset{r,\\, \\boldsymbol{z}}{\\min} \\quad & r\\\\
\\text{s.t.} \\quad & r \\geq \\ell_{t} - b z_{t}\\,, \\quad t = 1,\\ldots,T\\\\
\\quad & \\sum_{t=1}^{T} w_{t} z_{t} \\leq \\left(\\alpha - s\\right) W_{T}\\\\
\\quad & z_{t} \\in \\left\\{0,\\, 1\\right\\}\\,.
\\end{align}
```

Where:

  - ``r``: Value-at-Risk variable.
  - ``\\ell_{t}``: Loss of observation ``t``. It is ``-x_{t}`` for [`ValueatRisk`](@ref) and ``-d_{t}`` for [`DrawdownatRisk`](@ref).
  - ``z_{t}``: Exceedance indicator of observation ``t``. It is one when the bound on ``r`` does not apply to the observation.
  - ``b``: Big-M constant, the `b` field.
  - $(math_dict[:s_mip_slack])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set.
  - $(math_dict[:W_T_total])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

The minimum is the ``k``-th largest loss, the order statistic that [`ValueatRisk`](@ref) defines. The programme exempts the largest losses first, and it can exempt them while their cumulative weight stays within ``(\\alpha - s) W_{T}``. The slack keeps the exempt weight strictly below ``\\alpha W_{T}``, and it absorbs the rounding error of ``\\alpha W_{T}``.

The programme is exact when ``b`` is at least the largest loss minus the minimum. It needs ``s < \\alpha``, because the cardinality row has no solution for ``s > \\alpha``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MIPValueatRisk(;
        b::Option{<:Number} = nothing,
        s::Option{<:Number} = nothing
    ) -> MIPValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `b` is not `nothing`: `b > 0` and finite.
  - If `s` is not `nothing`: `s > 0` and finite.
  - If both `b` and `s` are not `nothing`: `b > s`.

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
  - [`DrawdownatRisk`](@ref): builds the same programme over the drawdowns.
  - [`Option`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.3, Equation 7.51.
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
            @argcheck(b > s,
                      DomainError((b, s),
                                  "`b` is $b and `s` is $s. The big-M constant `b` relaxes a bound the slack `s` tightens, so `b > s` must hold."))
        end
        return new{typeof(b), typeof(s)}(b, s)
    end
end
function MIPValueatRisk(; b::Option{<:Number} = nothing,
                        s::Option{<:Number} = nothing)::MIPValueatRisk
    return MIPValueatRisk(b, s)
end
"""
    mip_var_bounds(b, s) -> Tuple

Resolve the big-M constant `b` and the cardinality slack `s` of the empirical quantile programme to numbers.

A `nothing` takes the default, `b = 1e3` or `s = 1e-5`. The `JuMP` builder and the functor read the pair from this one function, so both select the same order statistic.

# Arguments

  - `b`: Big-M constant, or `nothing`.
  - `s`: Cardinality slack, or `nothing`.

# Returns

  - `(b, s)`: The two numbers, each the stated value or its default.

# Related

  - [`MIPValueatRisk`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`empirical_value_at_risk`](@ref)
"""
function mip_var_bounds(b::Option{<:Number}, s::Option{<:Number})
    return ifelse(isnothing(b), 1e3, b), ifelse(isnothing(s), 1e-5, s)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Value-at-Risk as a quantile of a parametric distribution, from the mean and the covariance of the asset returns.

This is a different estimand from the empirical quantile that [`MIPValueatRisk`](@ref) reports. The risk measure that holds it reports the parametric value in the optimisation model and in the functor alike. `mu` and `sigma` fall back to the prior's own, so [`factory`](@ref) fills them before either the model or the functor reads them.

The optimisation model bounds the standard deviation from above with a second-order cone, so the model is convex only while the coefficient of the standard deviation is not negative. For the three distributions that is ``\\alpha \\leq 1/2`` for [`ValueatRisk`](@ref), and ``\\alpha + \\beta \\leq 1`` for [`ValueatRiskRange`](@ref). Past that level a minimisation of the risk has no finite solution.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{w}) &= -\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} + z_{\\alpha} \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}\\,,\\\\
\\mathrm{VaRRange}_{\\alpha, \\beta}(\\boldsymbol{w}) &= \\left(z_{\\alpha} - z^{c}_{\\beta}\\right) \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:mu_er])
  - ``\\mathbf{\\Sigma}``: Covariance matrix of the asset returns, ``N \\times N``.
  - ``z_{\\alpha}``: Loss-tail z-score of `dist` at ``\\alpha``, from [`compute_value_at_risk_z`](@ref).
  - ``z^{c}_{\\beta}``: Gain-tail z-score of `dist` at ``\\beta``, from [`compute_value_at_risk_cz`](@ref).
  - $(math_dict[:alpha_rm])
  - ``\\beta``: Significance level of the gain tail, ``\\beta \\in (0, 1)``.

The range is the Value-at-Risk of the returns plus the Value-at-Risk of the negated returns, and the two mean terms cancel.

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
  - If `chol` is not `nothing`: `!isempty(chol)`, and `sigma` is a matrix rather than `nothing` or a Deferred Quantity.

!!! warning

    The caller states `mu`, `sigma` and `chol` independently, and nothing makes them agree with each other. A caller who wants one consistent set gives `pe` alone, and one fit of `pe` fills all three. A caller who states them by hand must make sure that they agree.

## View parameters

`DistributionValueatRisk` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method slices `mu` to the selected assets. A Deferred Quantity passes through unsliced, and resolves on the subset later.
  - The method slices a stated `sigma` on both axes. A Deferred Quantity passes through unsliced, and resolves on the subset later.
  - The method slices `chol` on its columns alone. Its rows index the factorisation, and the asset selection does not address them.
  - `pe` and `dist` pass through unchanged. `dist` describes the standardised loss, so it has no asset axis.

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

  - $(ref_dict[:cajas2025]) Section 7.2.2.3, Equations 7.52 and 7.53, and Section 11.2, Equation 11.5.
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

Resolve every Deferred Quantity held by [`DistributionValueatRisk`](@ref) `alg` against prior result `pr`.

The formulation holds three fields that the prior can fill, and `mu` and `sigma` are independent of each other. So it takes a `pe`, and one fit of `pe` fills every slot that the caller left unstated. `chol` is a factor of `sigma` and goes with it. A `sigma` that names its own estimator gives the factor from that same fit, and never from the fit of `pe`.

# Algorithm

 1. If `sigma` is a Deferred Quantity, fit it against `pr`, and take `sigma` and its factor `chol` from that one fit. Otherwise keep `sigma` and `chol` as stated.
 2. Resolve `mu` with [`resolve_slot`](@ref).
 3. If `pe` is `nothing`, rebuild the formulation with `mu`, `sigma` and `chol`, and return it.
 4. Fit `pe` against `pr`, giving `fitted`.
 5. If `sigma` is still `nothing`, take `chol` from `fitted`, so that the factor and the matrix come from one fit.
 6. Fill `mu` and `sigma` from `fitted` where they are still `nothing`, with [`fan_out_slot`](@ref).
 7. Rebuild the formulation with the four values and `pe = nothing`, and return it.

# Related

  - [`DistributionValueatRisk`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(alg::DistributionValueatRisk, pr::AbstractPriorResult,
                                     ::Any = nothing)::DistributionValueatRisk
    sigma, chol = if isa(alg.sigma, DeferredQuantity)
        fitted = fit_deferred_quantity(alg.sigma, pr)
        deferred_quantity(fitted, :sigma), deferred_derived_quantity(fitted, :chol)
    else
        alg.sigma, alg.chol
    end
    mu = resolve_slot(alg.mu, :mu, pr)
    if isnothing(alg.pe)
        return rebuild_with_slots(alg, (; mu = mu, sigma = sigma, chol = chol))
    end
    fitted = fit_deferred_quantity(alg.pe, pr)
    # `chol` is derived from `sigma`, so it comes from the fan-out only when the fan-out
    # also supplies the `sigma` it factorises. Read before `sigma` is filled.
    chol = isnothing(sigma) ? deferred_derived_quantity(fitted, :chol) : chol
    return rebuild_with_slots(alg,
                              (; mu = fan_out_slot(fitted, mu, :mu),
                               sigma = fan_out_slot(fitted, sigma, :sigma), chol = chol,
                               pe = nothing))
end
# Deferrable slots — see `deferred_slots`. `chol` is derived and never defers on its own.
function deferred_slots(alg::DistributionValueatRisk)
    return (; mu = alg.mu, sigma = alg.sigma, pe = alg.pe)
end
# The parametric functor reads `mu` and `sigma` as they stand, and neither has a sample
# reading here: the model's terms are the prior's moments — see `functor_slots`.
functor_slots(alg::DistributionValueatRisk) = (; mu = alg.mu, sigma = alg.sigma)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`DistributionValueatRisk`](@ref) by resolving its Deferred Quantities, then falling back to the prior result for whatever is still unstated.

[`sigma_chol_selector`](@ref) selects `sigma` and `chol` as a pair, not field by field. A stated `sigma` with no factor must not take the factor of the prior, which is a factor of a different matrix.

# Algorithm

 1. Resolve the Deferred Quantities with [`resolve_deferred_quantities`](@ref).
 2. Select `sigma` and `chol` as a pair with [`sigma_chol_selector`](@ref).
 3. Take `mu` from the formulation, or from `pr` when the formulation states none.
 4. Build the formulation with the three values, `pe = nothing` and the same `dist`.

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

Measures the Value-at-Risk (VaR), the loss that the portfolio exceeds with probability `alpha`.

`alg` selects the estimand. [`MIPValueatRisk`](@ref), the default, gives the empirical quantile of the sample, which the definition below states. [`DistributionValueatRisk`](@ref) gives the quantile of a parametric distribution, which that type states. The measure is not convex, so its empirical model is a mixed-integer programme.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) &= -x_{(k)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Value-at-Risk at level ``\\alpha``.
  - $(math_dict[:xret])
  - $(math_dict[:x_k_sorted])
  - $(math_dict[:k_var_mip])
  - $(math_dict[:W_k_cum])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set.
  - $(math_dict[:s_mip_slack]) It is the `s` field of [`MIPValueatRisk`](@ref).
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

``-x_{(k)}`` is the minimum of the programme of [`MIPValueatRisk`](@ref) over the losses ``-x_{t}``. Without observation weights ``k = \\lfloor (\\alpha - s) T \\rfloor + 1``. This is ``\\lceil \\alpha T \\rceil``, the index of the lower ``\\alpha``-quantile, while ``s T`` is smaller than ``1 - (\\lceil \\alpha T \\rceil - \\alpha T)``. The slack also absorbs the rounding error of ``\\alpha T``. For example, at ``\\alpha = 0.07`` and ``T = 100`` the product rounds to 7.000000000000001, and ``k`` is 7.

The definition ``-\\inf\\{x : F(x) > \\alpha\\}`` of the reference, Equation 7.50, with ``F`` the distribution function of the returns, gives the upper ``\\alpha``-quantile on a sample, the position ``\\lfloor \\alpha T \\rfloor + 1``. It differs from ``k`` by one position when ``\\alpha T`` is an integer. The measure follows the programme of the same reference, Equation 7.51, in its model and in its functor.

# Algorithm

The functor of the empirical estimand:

 1. Read the slack `s` with [`mip_var_bounds`](@ref), which gives the default when `alg.s` is `nothing`.
 2. Read the observation weights `w` with [`get_observation_weights`](@ref).
 3. Return [`empirical_value_at_risk`](@ref) of `x` at `alpha` and `s`. Without weights the kernel reorders its input, so it receives a copy of `x`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing,
        alg::ValueatRiskFormulation = MIPValueatRisk()
    ) -> ValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - $(val_dict[:oow_nonneg])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `alg`: Recursively viewed via [`port_opt_view`](@ref).

# Functor

    (r::ValueatRisk)(x::VecNum)
    (r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk})(w::VecNum, X, fees)

`alg` selects the quantity, and each method reports the value that its own `JuMP` model builds. The first computes the empirical quantile of a portfolio returns vector, weighted by `w` when the measure states observation weights, and leaves `x` unchanged. The second computes the parametric quantile from the moments that [`DistributionValueatRisk`](@ref) holds, so it takes the asset weights instead. It does not read `X` and `fees`, because the terms of the model are the prior's moments and no return series enters them.

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

  - $(ref_dict[:cajas2025]) Section 7.2.2.3, Equations 7.50 and 7.51.
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
    function ValueatRisk(settings::RiskMeasureSettings, alpha::Num_SigCal,
                         w::Option{<:ObsWeights}, alg::ValueatRiskFormulation)
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings, alpha,
                                                                            w, alg)
    end
end
function ValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Num_SigCal = 0.05, w::Option{<:ObsWeights} = nothing,
                     alg::ValueatRiskFormulation = MIPValueatRisk())::ValueatRisk
    return ValueatRisk(settings, alpha, w, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha` of a [`ValueatRisk`](@ref) against prior result `pr`, and resolve the formulation `alg` beside it.

`alpha` can hold a Calibration Rule in place of a number, so it resolves here. The rebuild goes through [`rebuild_with_slots`](@ref), whose call to the inner constructor checks `0 < alpha < 1` again on the calibrated number. So the check that a number of the caller meets also refuses a rule that returns a value outside the slot.

This method is more specific than the derived recursion, so it takes over the `alg` slot that [`deferred_slots`](@ref) declares. It resolves that slot through [`resolve_deferred_child`](@ref), the function that the derived recursion calls.

# Algorithm

 1. Set `ws` to the observation weights of the measure, or to those of `pr` when the measure states none.
 2. Resolve `alpha` with [`resolve_calibration_slot`](@ref), which gives the rule `pr`, `ws` and the solver `slv`. The measure holds no solver of its own.
 3. Resolve `alg` with [`resolve_deferred_child`](@ref).
 4. Rebuild the measure with the resolved `alpha` and `alg`.

# Related

  - [`ValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
  - [`Num_SigCal`](@ref)
"""
function resolve_deferred_quantities(x::ValueatRisk, pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, slv)
    alg = resolve_deferred_child(x.alg, pr, slv)
    return rebuild_with_slots(x, (; alpha = alpha, alg = alg))
end
# Calibration slots — see `calibration_slots`. The significance level is the one quantity of
# this measure that a rule may compute.
calibration_slots(x::ValueatRisk) = (; alpha = x.alpha)
# The empirical order statistic is the `MIPValueatRisk` estimand, so the functor below names
# that formulation. Leaving `alg` free makes it overlap the parametric method further down,
# which no rule of specificity can order.
function (r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk})(x::VecNum)
    _, s = mip_var_bounds(r.alg.b, r.alg.s)
    w = get_observation_weights(r.w, x)
    return empirical_value_at_risk(isnothing(w) ? copy(x) : x, r.alpha, s, w)
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

Measures the Value-at-Risk Range, the spread between the loss quantile at `alpha` and the gain quantile at `beta`.

The range sums the [`ValueatRisk`](@ref) of the portfolio returns and the Value-at-Risk of the negated returns. `alg` selects the estimand of both tails, as it does for [`ValueatRisk`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) + \\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: Value-at-Risk Range.
  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Value-at-Risk of the returns at level ``\\alpha``, the loss quantile that [`ValueatRisk`](@ref) defines.
  - ``\\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})``: Value-at-Risk of the negated returns at level ``\\beta``, which is minus the gain quantile.
  - $(math_dict[:xret])
  - ``\\alpha``: Significance level of the loss tail, ``\\alpha \\in (0, 1)``.
  - ``\\beta``: Significance level of the gain tail, ``\\beta \\in (0, 1)``.

$(math_dict[:negated_upper_tail])

The two tails select their order statistics with the same slack ``s``. The gain tail is the ``k``-th largest return, with ``k`` the index of [`ValueatRisk`](@ref) at ``\\beta``.

# Algorithm

The functor of the empirical estimand:

 1. Read the slack `s` with [`mip_var_bounds`](@ref), which gives the default when `alg.s` is `nothing`.
 2. Read the observation weights `w` with [`get_observation_weights`](@ref).
 3. Compute `loss`, [`empirical_value_at_risk`](@ref) of `x` at `alpha` and `s`. Without weights the kernel reorders its input, so it receives a copy of `x`.
 4. Compute `gain`, [`empirical_value_at_risk`](@ref) of `-x` at `beta` and `s`.
 5. Return `loss + gain`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        beta::Num_SigCal = alpha,
        w::Option{<:ObsWeights} = nothing,
        alg::ValueatRiskFormulation = MIPValueatRisk()
    ) -> ValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `beta` is a number: `0 < beta < 1`.
  - $(val_dict[:oow_nonneg])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `alg`: Recursively viewed via [`port_opt_view`](@ref).

# Functor

    (r::ValueatRiskRange)(x::VecNum)
    (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk})(w::VecNum, X, fees)

`alg` selects the quantity, as it does for [`ValueatRisk`](@ref). The first method computes the sum of the two empirical tail quantiles of a portfolio returns vector, and leaves `x` unchanged. The second computes the parametric range that [`DistributionValueatRisk`](@ref) states. It does not read `X` and `fees`.

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
    function ValueatRiskRange(settings::RiskMeasureSettings, alpha::Num_SigCal,
                              beta::Num_SigCal, w::Option{<:ObsWeights},
                              alg::ValueatRiskFormulation)
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
                          alpha::Num_SigCal = 0.05, beta::Num_SigCal = alpha,
                          w::Option{<:ObsWeights} = nothing,
                          alg::ValueatRiskFormulation = MIPValueatRisk())::ValueatRiskRange
    return ValueatRiskRange(settings, alpha, beta, w, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the two significance levels of a [`ValueatRiskRange`](@ref) against prior result `pr`, and resolve the formulation `alg` beside them.

Each tail has its own slot and its own bound, so a rule for the loss tail and a rule for the gain tail resolve independently. `beta` defaults to `alpha`, so a rule stated for the loss tail alone reaches both tails. The rule states the method, and the slot states the tail. The rebuild goes through [`rebuild_with_slots`](@ref), whose call to the inner constructor checks both levels again on the calibrated numbers.

This method is more specific than the derived recursion, so it takes over the `alg` slot that [`deferred_slots`](@ref) declares, through [`resolve_deferred_child`](@ref).

# Algorithm

 1. Set `ws` to the observation weights of the measure, or to those of `pr` when the measure states none.
 2. Resolve `alpha` with [`resolve_calibration_slot`](@ref), which gives the rule `pr`, `ws` and the solver `slv`.
 3. Resolve `beta` in the same way.
 4. Resolve `alg` with [`resolve_deferred_child`](@ref).
 5. Rebuild the measure with the resolved `alpha`, `beta` and `alg`.

# Related

  - [`ValueatRiskRange`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
  - [`Num_SigCal`](@ref)
"""
function resolve_deferred_quantities(x::ValueatRiskRange, pr::AbstractPriorResult,
                                     slv = nothing)
    ws = sel(x.w, pr.w)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, slv)
    beta = resolve_calibration_slot(x.beta, :beta, pr, ws, slv)
    alg = resolve_deferred_child(x.alg, pr, slv)
    return rebuild_with_slots(x, (; alpha = alpha, beta = beta, alg = alg))
end
# Calibration slots — see `calibration_slots`. One slot per tail, each with its own role.
calibration_slots(x::ValueatRiskRange) = (; alpha = x.alpha, beta = x.beta)
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
# The empirical order statistic is the `MIPValueatRisk` estimand, so the functor below names
# that formulation. Leaving `alg` free makes it overlap the parametric method further down,
# which no rule of specificity can order. The gain tail is the same kernel over `-x`, as the
# model builds it over the negated series.
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:MIPValueatRisk})(x::VecNum)
    _, s = mip_var_bounds(r.alg.b, r.alg.s)
    w = get_observation_weights(r.w, x)
    loss = empirical_value_at_risk(isnothing(w) ? copy(x) : x, r.alpha, s, w)
    gain = empirical_value_at_risk(-x, r.beta, s, w)
    return loss + gain
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

Measures the Drawdown-at-Risk (DaR), the uncompounded drawdown that the portfolio exceeds with probability `alpha`.

It is the [`ValueatRisk`](@ref) of the drawdown series in place of the returns. Its optimisation model is the mixed-integer programme of [`MIPValueatRisk`](@ref) over the drawdowns, so `b` and `s` have the same meaning here as they have there. The measure is not convex.

# Mathematical definition

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} x_s\\,, \\quad c_0 = 0\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s\\,, \\\\
\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x}) &= -d_{(k)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x})``: Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:xret])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])
  - ``d_{(k)}``: ``k``-th smallest drawdown, the ``k``-th deepest, and ``w_{(k)}`` its observation weight.
  - $(math_dict[:k_var_mip])
  - $(math_dict[:W_k_cum])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set.
  - $(math_dict[:s_mip_slack]) It is the `s` field.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

The running peak starts at ``c_0 = 0``, so a series that falls from its first period is in drawdown from the start. ``-d_{(k)}`` is the minimum of the programme of [`MIPValueatRisk`](@ref) over the losses ``-d_{t}``, and ``k`` follows the rules that [`ValueatRisk`](@ref) states.

# Algorithm

 1. Read the slack `s` with [`mip_var_bounds`](@ref), which gives the default when the `s` field is `nothing`.
 2. Compute the drawdown series with [`absolute_drawdown_vec`](@ref).
 3. Read the observation weights with [`get_observation_weights`](@ref).
 4. Return [`empirical_value_at_risk`](@ref) of the drawdown series at `alpha` and `s`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing,
        b::Option{<:Number} = nothing,
        s::Option{<:Number} = nothing
    ) -> DrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - $(val_dict[:oow_nonneg])
  - If `b` is not `nothing`: `b > 0` and finite.
  - If `s` is not `nothing`: `s > 0` and finite.
  - If both `b` and `s` are not `nothing`: `b > s`.

# Functor

    (r::DrawdownatRisk)(x::VecNum)

Computes the Drawdown-at-Risk of a portfolio returns vector `x`, and leaves `x` unchanged.

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
  - [`MIPValueatRisk`](@ref)
  - [`empirical_value_at_risk`](@ref)

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:cajas2025]) Section 7.2.4.3, Equations 7.90 and 7.91.
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
    function DrawdownatRisk(settings::RiskMeasureSettings, alpha::Num_SigCal,
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
            @argcheck(b > s,
                      DomainError((b, s),
                                  "`b` is $b and `s` is $s. The big-M constant `b` relaxes a bound the slack `s` tightens, so `b > s` must hold."))
        end
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(b), typeof(s)}(settings,
                                                                                     alpha,
                                                                                     w, b,
                                                                                     s)
    end
end
function DrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        alpha::Num_SigCal = 0.05, w::Option{<:ObsWeights} = nothing,
                        b::Option{<:Number} = nothing,
                        s::Option{<:Number} = nothing)::DrawdownatRisk
    return DrawdownatRisk(settings, alpha, w, b, s)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::DrawdownatRisk) = (; alpha = x.alpha)
"""
    absolute_drawdown_vec(x::VecNum) -> Vector

Compute the uncompounded drawdown series of one return series.

The function reads `x` and does not write it, so any `AbstractVector` works, for example a column view, a range or an immutable array.

# Mathematical definition

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} x_s\\,, \\quad c_0 = 0\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])

The running peak starts at ``c_0 = 0``, so each drawdown is relative to the initial portfolio value, not to the first observation.

# Algorithm

 1. Allocate `dd`, of the same length and element type as `x`.
 2. Set the cumulative return `cs` and the running peak `peak` to zero.
 3. For each observation, add it to `cs`, raise `peak` to `cs` when `cs` is larger, and store `cs - peak` in `dd`.
 4. Return `dd`.

# Arguments

  - `x::VecNum`: Return series.

# Returns

  - `dd`: Drawdown series of the same length as `x`, with every entry ``\\leq 0``.

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
    empirical_value_at_risk(x::VecNum, alpha::Real, s::Real, ::Nothing) -> Number
    empirical_value_at_risk(x::VecNum, alpha::Real, s::Real, w::VecNum) -> Number

Compute the empirical Value-at-Risk of a series, the negated order statistic that the mixed-integer quantile programme selects.

The four empirical measures of the family call this one kernel: [`ValueatRisk`](@ref) and [`ValueatRiskRange`](@ref) on the returns, [`DrawdownatRisk`](@ref) and [`RelativeDrawdownatRisk`](@ref) on a drawdown series. The method without weights reorders `x` in place, so a caller passes a copy or a vector that it has just computed.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) &= -x_{(k)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_k_sorted])
  - $(math_dict[:k_var_mip])
  - $(math_dict[:W_k_cum])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when `w` is `nothing`.
  - $(math_dict[:s_mip_slack])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

# Algorithm

Without observation weights:

 1. Set `k` to ``\\lfloor (\\alpha - s) T \\rfloor + 1``, clamped to the range from 1 to ``T``.
 2. Partially sort `x`, so that its `k` smallest entries come first, in ascending order.
 3. Return minus the entry at `k`.

With observation weights:

 1. Sort `x` with `sortperm`, giving `order`, and accumulate the sorted weights, giving `cum_w`.
 2. Set `k` to one plus the last position at which `cum_w` does not exceed `(alpha - s)` times the total weight, and cap it at ``T``.
 3. Return minus the `k`-th smallest entry of `x`.

# Arguments

  - `x::VecNum`: Series, for example the portfolio returns or a drawdown series. The method without weights reorders it.
  - `alpha::Real`: Significance level, `0 < alpha < 1`.
  - `s::Real`: Cardinality slack, from [`mip_var_bounds`](@ref).
  - `w`: Observation weights from [`get_observation_weights`](@ref), or `nothing`.

# Returns

  - `Number`: The Value-at-Risk, positive when the order statistic is a loss.

# Related

  - [`ValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`RelativeDrawdownatRisk`](@ref)
  - [`MIPValueatRisk`](@ref): the programme whose minimum this kernel computes.
  - [`conditional_drawdown_at_risk`](@ref)
"""
function empirical_value_at_risk(x::VecNum, alpha::Real, s::Real, ::Nothing)
    T = length(x)
    k = clamp(floor(Int, (alpha - s) * T) + 1, 1, T)
    return -partialsort!(x, k)
end
function empirical_value_at_risk(x::VecNum, alpha::Real, s::Real, w::VecNum)
    order = sortperm(x)
    cum_w = cumsum(view(w, order))
    k = min(searchsortedlast(cum_w, (alpha - s) * cum_w[end]) + 1, length(x))
    return -x[order[k]]
end
function (r::DrawdownatRisk)(x::VecNum)
    _, s = mip_var_bounds(r.b, r.s)
    return empirical_value_at_risk(absolute_drawdown_vec(x), r.alpha, s,
                                   get_observation_weights(r.w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Relative Drawdown-at-Risk, the compounded drawdown that the portfolio exceeds with probability `alpha`.

It is the [`ValueatRisk`](@ref) of the relative drawdown series. The compounded wealth is a product of linear functions of the weights, so the library states no JuMP formulation of the measure, and it is a [`HierarchicalRiskMeasure`](@ref).

# Mathematical definition

```math
\\begin{align}
C_t &= \\prod_{s=1}^{t} (1 + x_s)\\,, \\quad C_0 = 1\\,, \\\\
rd_t &= \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1\\,, \\\\
\\mathrm{RDaR}_{\\alpha}(\\boldsymbol{x}) &= -rd_{(k)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RDaR}_{\\alpha}(\\boldsymbol{x})``: Relative Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:xret])
  - $(math_dict[:Ct])
  - $(math_dict[:rdt])
  - ``rd_{(k)}``: ``k``-th smallest relative drawdown, the ``k``-th deepest, and ``w_{(k)}`` its observation weight.
  - $(math_dict[:k_var_mip])
  - $(math_dict[:W_k_cum])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set.
  - $(math_dict[:s_mip_slack]) The measure has no `s` field, so it takes the default of [`mip_var_bounds`](@ref).
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

The running peak starts at ``C_0 = 1``, so each drawdown is relative to the initial portfolio value. The index ``k`` is the one of [`DrawdownatRisk`](@ref), so the two measures select the same position in their series.

# Algorithm

 1. Read the default slack `s` with [`mip_var_bounds`](@ref).
 2. Compute the relative drawdown series with [`relative_drawdown_vec`](@ref).
 3. Read the observation weights with [`get_observation_weights`](@ref).
 4. Return [`empirical_value_at_risk`](@ref) of the relative drawdown series at `alpha` and `s`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::RelativeDrawdownatRisk)(x::VecNum)

Computes the Relative Drawdown-at-Risk of a portfolio returns vector `x`, and leaves `x` unchanged.

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
  - [`empirical_value_at_risk`](@ref)

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
                                    alpha::Num_SigCal, w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Num_SigCal = 0.05,
                                w::Option{<:ObsWeights} = nothing)::RelativeDrawdownatRisk
    return RelativeDrawdownatRisk(settings, alpha, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::RelativeDrawdownatRisk) = (; alpha = x.alpha)
"""
    relative_drawdown_vec(x::VecNum) -> Vector

Compute the compounded drawdown series of one return series.

The function reads `x` and does not write it, so any `AbstractVector` works, for example a column view, a range or an immutable array.

# Mathematical definition

```math
\\begin{align}
C_t &= \\prod_{s=1}^{t} (1 + x_s)\\,, \\quad C_0 = 1\\,, \\\\
rd_t &= \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:Ct])
  - $(math_dict[:rdt])

The running peak starts at ``C_0 = 1``, so each drawdown is relative to the initial portfolio value, not to the first observation.

# Algorithm

 1. Allocate `dd`, of the same length and element type as `x`.
 2. Set the compounded wealth `cp` and the running peak `peak` to one.
 3. For each observation, multiply `cp` by one plus the return, raise `peak` to `cp` when `cp` is larger, and store `cp / peak - 1` in `dd`.
 4. Return `dd`.

# Arguments

  - `x::VecNum`: Return series.

# Returns

  - `dd`: Relative drawdown series of the same length as `x`, with every entry ``\\leq 0``.

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
    _, s = mip_var_bounds(nothing, nothing)
    return empirical_value_at_risk(relative_drawdown_vec(x), r.alpha, s,
                                   get_observation_weights(r.w, x))
end

"""
    const CholRM = Union{<:Variance, <:StandardDeviation, <:DistributionValueatRisk}

Groups the types that hold an optional covariance matrix `sigma` and its optional factor `chol`.

The group exists so that one method of [`chol_sigma_selector`](@ref) serves all three. The `JuMP` model of each type reads a factor of the covariance, and that method selects it. It takes `chol` when the caller states it, a factor of `sigma` when the caller states only `sigma`, and the factor of the prior otherwise. [`DistributionValueatRisk`](@ref) is a formulation, not a risk measure, but it holds the same two fields.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`chol_sigma_selector`](@ref)
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
