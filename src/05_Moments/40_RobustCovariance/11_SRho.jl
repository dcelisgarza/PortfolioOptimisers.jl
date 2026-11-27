"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rho functions of an S estimator of location and scatter.

An S estimator of [davies1987](@cite) finds the location and the scatter of determinant one whose distances have the smallest M-scale. The rho function gives the M-scale, see [`m_scale`](@ref). It is bounded, so an observation far from the centre adds a fixed amount to the equation of the scale, and it cannot move the estimate much. The weight of an observation in the fit is the derivative of the rho function, see [`srho_weight`](@ref). [`BisquareRho`](@ref) is a subtype.

Each method reads the squared distance ``t = d^2 / s`` of an observation, where ``s`` is the M-scale of the squared distances.

# Interfaces

To implement a new rho function, subtype `AbstractSRho`, keep each parameter in a field, and implement three methods:

  - `PortfolioOptimisers.s_tuning(rho, bdp::Number, n::Integer, p::Integer) -> Tuple{Number, Number}`: Returns the tuning constant `k` of the rho function and the right side `b` of the equation of the M-scale, for the breakdown point `bdp` of `n` observations of `p` assets.
  - `PortfolioOptimisers.srho(rho, t::Number, k::Number) -> Number`: Returns the rho function at the squared distance `t`, a number from zero to one.
  - `PortfolioOptimisers.srho_weight(rho, t::Number, k::Number) -> Number`: Returns the weight of an observation at the squared distance `t`, one at the centre.

# Related

  - [`BisquareRho`](@ref)
  - [`SCovariance`](@ref)
  - [`m_scale`](@ref)

# References

  - $(ref_dict[:davies1987]) Not read. See [`SCovariance`](@ref).
"""
abstract type AbstractSRho <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The bisquare rho function of Tukey, the rho function of the S estimator of [`SCovariance`](@ref).

The function rises from zero at the centre to one at the squared distance ``c^2``, and stays at one beyond it. So an observation beyond ``c^2`` has the weight zero. The constant ``c^2`` makes the M-scale consistent at the normal model for the breakdown point `bdp`, see [`bisquare_constant`](@ref). It is ``3.4529^2`` for three assets at `bdp = 0.5`, the value of the R package `rrcov`, version 1.7-7.

# Mathematical definition

```math
\\begin{align}
\\rho(t) &= \\begin{cases} 1 - \\left(1 - t / c^2\\right)^3 & t \\le c^2\\,, \\\\ 1 & t > c^2\\,, \\end{cases} & w(t) &= \\begin{cases} \\left(1 - t / c^2\\right)^2 & t \\le c^2\\,, \\\\ 0 & t > c^2\\,. \\end{cases}
\\end{align}
```

Where:

  - ``t``: The squared distance of an observation, divided by the M-scale.
  - ``c^2``: The tuning constant.
  - ``w``: The weight of the observation, the derivative of ``\\rho`` divided by its value at the centre.

# Examples

```jldoctest
julia> BisquareRho()
BisquareRho()
```

# Related

  - [`AbstractSRho`](@ref)
  - [`SCovariance`](@ref)
  - [`BisquareReweighting`](@ref)

# References

  - $(ref_dict[:maronna2019]) Not read. The tuning constants agree with the R package `rrcov`, version 1.7-7.
"""
struct BisquareRho <: AbstractSRho end
"""
    s_tuning(rho::BisquareRho, bdp::Number, n::Integer, p::Integer) -> Tuple{Number, Number}

Returns the tuning constant of the rho function and the right side of the equation of the M-scale.

The bisquare rho takes the constant ``c^2`` of [`bisquare_constant`](@ref), and the right side is `bdp`. Neither depends on the number of observations.

# Arguments

  - `rho`: The rho function.
  - `bdp`: The breakdown point.
  - `n`: The number of observations.
  - `p`: The number of assets.

# Returns

  - `(k, b)`: The tuning constant and the right side.

# Related

  - [`AbstractSRho`](@ref)
  - [`m_scale`](@ref)
"""
function s_tuning(::BisquareRho, bdp::Number, ::Integer, p::Integer)
    return bisquare_constant(bdp, p), bdp
end
"""
    srho(rho::BisquareRho, t::Number, c2::Number) -> Number

Returns the bisquare rho function at the squared distance `t`, see [`BisquareRho`](@ref).

# Arguments

  - `rho`: The rho function.
  - `t`: The squared distance, divided by the M-scale.
  - `c2`: The tuning constant ``c^2``.

# Returns

  - `r::Number`: The value, from zero to one.

# Related

  - [`srho_weight`](@ref)
  - [`m_scale`](@ref)
"""
function srho(::BisquareRho, t::Number, c2::Number)
    r = t / c2
    return r >= 1 ? one(r) : 1 - (1 - r)^3
end
"""
    srho_weight(rho::BisquareRho, t::Number, c2::Number) -> Number

Returns the bisquare weight of an observation at the squared distance `t`, see [`BisquareRho`](@ref).

# Arguments

  - `rho`: The rho function.
  - `t`: The squared distance, divided by the M-scale.
  - `c2`: The tuning constant ``c^2``.

# Returns

  - `w::Number`: The weight, one at the centre and zero beyond ``c^2``.

# Related

  - [`srho`](@ref)
  - [`s_isteps`](@ref)
"""
function srho_weight(::BisquareRho, t::Number, c2::Number)
    r = t / c2
    return r >= 1 ? zero(r) : (1 - r)^2
end
"""
    m_scale(rho::AbstractSRho, t::VecNum, k::Number, b::Number, s::Number) -> Number

Returns the M-scale of the squared distances `t`, the value ``s`` at which the mean of ``\\rho(t_i / s)`` is `b`.

The fixed-point step ``s \\leftarrow s \\, \\overline{\\rho(t/s)} / b`` converges to the scale from any positive start, because the rho function is bounded and does not decrease. The step stops when the relative change is at most `4 * eps`, or after [`S_MAX_ITER`](@ref) steps. A start near the scale, such as the scale of the last step of the fit, needs few steps.

# Arguments

  - `rho`: The rho function.
  - `t`: The squared distance of each observation.
  - `k`: The tuning constant of the rho function.
  - `b`: The right side of the equation, see [`s_tuning`](@ref).
  - `s`: The start, a positive number.

# Returns

  - `s::Number`: The M-scale.

# Related

  - [`AbstractSRho`](@ref)
  - [`s_isteps`](@ref)
"""
function m_scale(rho::AbstractSRho, t::VecNum, k::Number, b::Number, s::Number)
    # The step takes the scale as its argument, so no closure captures the loop variable. The
    # first step gives `s` the type of the step, so the loop is type stable.
    mstep = r -> r * Statistics.mean(x -> srho(rho, x / r, k), t) / b
    s = mstep(s)
    for _ in 2:S_MAX_ITER
        sn = mstep(s)
        done = abs(sn - s) <= 4 * eps(typeof(sn)) * sn
        s = sn
        if done
            break
        end
    end
    return s
end
"""
    chisq_truncated_moment(p::Integer, c2::Number, a::AbstractVector{<:Number}) -> Number

Returns the expected value of the polynomial ``\\sum_k a_k t^k`` over ``t \\le c^2``, where ``t`` has the ``\\chi^2_p`` distribution.

The rho functions and the weights of [`BisquareRho`](@ref) are polynomials in ``t`` up to ``c^2``. So their moments at the normal model follow from the moments of the truncated ``\\chi^2_p`` distribution, with no numerical integration.

# Mathematical definition

```math
\\begin{align}
\\mathbb{E}\\left[t^k \\mathbb{1}\\{t \\le c^2\\}\\right] &= F_{\\chi^2_{p + 2k}}(c^2) \\prod_{j=0}^{k-1} (p + 2j)\\,.
\\end{align}
```

# Arguments

  - `p`: The number of assets, the degrees of freedom.
  - `c2`: The bound ``c^2``.
  - `a`: The coefficients of the polynomial, from the power zero up.

# Returns

  - `m::Number`: The expected value.

# Related

  - [`bisquare_constant`](@ref)
  - [`bisquare_efficiency`](@ref)
"""
function chisq_truncated_moment(p::Integer, c2::Number, a::AbstractVector{<:Number})
    m, mk = zero(c2), one(c2)
    for (k, ak) in zip(0:(length(a) - 1), a)
        m += ak * mk * Distributions.cdf(Distributions.Chisq(p + 2k), c2)
        mk *= p + 2k
    end
    return m
end
"""
    bisquare_constant(bdp::Number, p::Integer) -> Number

Returns the tuning constant ``c^2`` of [`BisquareRho`](@ref) at which the expected rho function at the normal model is the breakdown point `bdp`.

With this constant, the M-scale of the squared distances of a normal sample about its true location, in the metric of its true scatter, tends to one. So the S estimate of the scatter is consistent at the normal model with no more factor. The expected value comes from [`chisq_truncated_moment`](@ref), and it falls from one to zero as ``c^2`` rises, so a bisection finds the constant. The interval holds it, because the expected value is at least ``P(t > c^2)`` and at most ``3p/c^2``.

# Mathematical definition

```math
\\begin{align}
\\mathbb{E}\\left[\\rho(t)\\right] &= \\mathbb{E}\\left[\\left(3 \\frac{t}{c^2} - 3 \\frac{t^2}{c^4} + \\frac{t^3}{c^6}\\right) \\mathbb{1}\\{t \\le c^2\\}\\right] + 1 - F_{\\chi^2_p}(c^2) = \\mathrm{bdp}\\,.
\\end{align}
```

# Arguments

  - `bdp`: The breakdown point.
  - `p`: The number of assets.

# Returns

  - `c2::Number`: The tuning constant.

# Related

  - [`BisquareRho`](@ref)
  - [`s_tuning`](@ref)
"""
function bisquare_constant(bdp::Number, p::Integer)
    function excess(c2)
        a = inv(c2)
        rho = chisq_truncated_moment(p, c2, [zero(a), 3 * a, -3 * a^2, a^3])
        return rho + Distributions.ccdf(Distributions.Chisq(p), c2) - bdp
    end
    lo = Distributions.quantile(Distributions.Chisq(p), 1 - bdp) / 2
    return Roots.find_zero(excess, (lo, 6 * p / bdp), Roots.Bisection())
end
"""
    bisquare_efficiency(p::Integer, c2::Number) -> Number

Returns the asymptotic efficiency of the shape of a bisquare M-estimator of scatter at the normal model, relative to the sample covariance.

[salibianbarrera2006](@cite) states the asymptotic variance ``\\sigma_1`` of the shape of an M-estimator. The formula below writes it for the bisquare function in the squared distance ``t``. The efficiency is ``1 / \\sigma_1``. It rises from zero to one as ``c^2`` rises, and the constant of [`BisquareReweighting`](@ref) at 95 % is ``6.0963^2`` for three assets, the value of the R package `rrcov`, version 1.7-7.

# Mathematical definition

```math
\\begin{align}
\\sigma_1 &= \\frac{p (p + 2) \\, \\mathbb{E}\\left[t^2 (1 - t/c^2)^4 \\mathbb{1}\\{t \\le c^2\\}\\right]}{\\mathbb{E}\\left[t (1 - t/c^2) \\left((p + 2) - (p + 6) t / c^2\\right) \\mathbb{1}\\{t \\le c^2\\}\\right]^2}\\,.
\\end{align}
```

Where:

  - ``t``: A variable of the ``\\chi^2_p`` distribution.

# Arguments

  - `p`: The number of assets.
  - `c2`: The tuning constant ``c^2``.

# Returns

  - `eff::Number`: The efficiency, from zero to one.

# Related

  - [`BisquareReweighting`](@ref)
  - [`chisq_truncated_moment`](@ref)

# References

  - $(ref_dict[:salibianbarrera2006]) Not read. The formula agrees with the R package `rrcov`, version 1.7-7.
"""
function bisquare_efficiency(p::Integer, c2::Number)
    a = inv(c2)
    num = p *
          (p + 2) *
          chisq_truncated_moment(p, c2,
                                 [zero(a), zero(a), one(a), -4 * a, 6 * a^2, -4 * a^3, a^4])
    den = chisq_truncated_moment(p, c2,
                                 [zero(a), p + 2 * one(a), -(2p + 8) * a, (p + 6) * a^2])
    return abs2(den) / num
end

public AbstractSRho, s_tuning, srho, srho_weight
export BisquareRho
