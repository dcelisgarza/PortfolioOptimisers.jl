"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rho functions of an S estimator of location and scatter.

An S estimator of [davies1987](@cite) finds the location and the scatter of determinant one whose distances have the smallest M-scale. The rho function gives the M-scale, see [`m_scale`](@ref). It is bounded, so an observation far from the centre adds a fixed amount to the equation of the scale, and it cannot move the estimate much. The weight of an observation in the fit is the derivative of the rho function, see [`srho_weight`](@ref). [`BisquareRho`](@ref) and [`RockeRho`](@ref) are its subtypes.

Each method reads the squared distance ``t = d^2 / s`` of an observation, where ``s`` is the M-scale of the squared distances.

# Interfaces

To implement a new rho function, subtype `AbstractSRho`, keep each parameter in a field, and implement four methods:

  - `PortfolioOptimisers.s_tuning(rho, bdp::Number, n::Integer, p::Integer) -> Tuple{Number, Number}`: Returns the tuning constant `k` of the rho function and the right side `b` of the equation of the M-scale, for the breakdown point `bdp` of `n` observations of `p` assets.
  - `PortfolioOptimisers.srho(rho, t::Number, k::Number) -> Number`: Returns the rho function at the squared distance `t`, a number from zero to one.
  - `PortfolioOptimisers.srho_weight(rho, t::Number, k::Number) -> Number`: Returns the weight of an observation at the squared distance `t`, a number from zero to one.
  - `PortfolioOptimisers.s_consistency(rho, d2::VecNum, p::Integer) -> Number`: Returns the factor that makes the covariance consistent at the normal model, from the squared distances `d2` in the metric of the scale times the shape.

# Related

  - [`BisquareRho`](@ref)
  - [`RockeRho`](@ref)
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
$(DocStringExtensions.TYPEDEF)

The rho function of Rocke, a translated bisquare whose weights fall on both sides of the typical distance.

With many assets, the squared distances of a normal sample are near their mean ``p``, and few observations are near the centre. The bisquare rho of a high breakdown point then gives a low efficiency, or rejects too few outliers. The rho of [rocke1996](@cite) is zero below ``1 - \\gamma`` and one above ``1 + \\gamma``, where ``t = d^2/s`` is centred at one. Its width ``\\gamma`` comes from the asymptotic rejection probability `arp`, the share of the observations of a normal sample that get the weight zero. The right side of the M-scale is ``\\mathrm{bdp} \\, (1 - p/n)``, see [`s_tuning`](@ref). The scale is not consistent at the normal model, so the fit multiplies the covariance by the median of the squared distances over the median of the ``\\chi^2_p`` distribution, see [`s_consistency`](@ref).

The R package `rrcov`, version 1.7-7, with `method = "rocke"`, gives the same estimate where its iterations reach the minimum. On five of six parity samples, the two agree to `1e-3`, and the M-scale of this fit is the smaller one. On the sixth, 40 observations of 6 assets, `rrcov` stops at a local minimum whose M-scale is 1.7 % larger than the one that [`DetS`](@ref) finds.

# Mathematical definition

```math
\\begin{align}
\\rho(t) &= \\begin{cases} 0 & t < 1 - \\gamma\\,, \\\\ \\frac{1}{2} + \\frac{t - 1}{4 \\gamma} \\left(3 - \\left(\\frac{t - 1}{\\gamma}\\right)^2\\right) & \\lvert t - 1 \\rvert \\le \\gamma\\,, \\\\ 1 & t > 1 + \\gamma\\,, \\end{cases} & w(t) &= \\begin{cases} 1 - \\left(\\frac{t - 1}{\\gamma}\\right)^2 & \\lvert t - 1 \\rvert \\le \\gamma\\,, \\\\ 0 & \\text{otherwise}\\,, \\end{cases} \\\\
\\gamma &= \\min\\left(\\frac{\\chi^2_{p, 1 - \\mathrm{arp}}}{p} - 1, 1\\right)\\,.
\\end{align}
```

Where:

  - ``t``: The squared distance of an observation, divided by the M-scale.
  - ``w``: The weight of the observation, the derivative of ``\\rho`` divided by its largest value.
  - ``\\chi^2_{p, 1 - \\mathrm{arp}}``: The quantile ``1 - \\mathrm{arp}`` of the ``\\chi^2_p`` distribution.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RockeRho(;
        arp::Number = 0.1
    ) -> RockeRho

Keywords correspond to the struct's fields.

## Validation

  - `0 < arp < 1`.

# Examples

```jldoctest
julia> RockeRho()
RockeRho
  arp ┴ Float64: 0.1
```

# Related

  - [`AbstractSRho`](@ref)
  - [`BisquareRho`](@ref)
  - [`SCovariance`](@ref)

# References

  - $(ref_dict[:rocke1996]) Not read. The width, the right side and the consistency factor agree with the R package `rrcov`, version 1.7-7.
  - $(ref_dict[:maronna2019]) Not read.
"""
@concrete struct RockeRho <: AbstractSRho
    """
    $(field_dict[:s_arp])
    """
    arp
    function RockeRho(arp::Number)
        assert_open_probability(arp, :arp)
        return new{typeof(arp)}(arp)
    end
end
function RockeRho(; arp::Number = 0.1)::RockeRho
    return RockeRho(arp)
end
"""
    s_tuning(rho::BisquareRho, bdp::Number, n::Integer, p::Integer) -> Tuple{Number, Number}
    s_tuning(rho::RockeRho, bdp::Number, n::Integer, p::Integer) -> Tuple{Number, Number}

Returns the tuning constant of the rho function and the right side of the equation of the M-scale.

The bisquare rho takes the constant ``c^2`` of [`bisquare_constant`](@ref), and the right side is `bdp`. Neither depends on the number of observations. The Rocke rho takes the width ``\\gamma`` of [`RockeRho`](@ref), and the right side ``\\mathrm{bdp} \\, (1 - p/n)``. At `bdp = 1/2` that is the right side of the largest breakdown point that ``n`` observations of ``p`` assets permit.

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
function s_tuning(rho::RockeRho, bdp::Number, n::Integer, p::Integer)
    g = Distributions.quantile(Distributions.Chisq(p), 1 - rho.arp) / p - 1
    return min(g, one(g)), bdp * (1 - p / n)
end
"""
    srho(rho::BisquareRho, t::Number, c2::Number) -> Number
    srho(rho::RockeRho, t::Number, g::Number) -> Number

Returns the rho function at the squared distance `t`, see [`BisquareRho`](@ref) and [`RockeRho`](@ref).

# Arguments

  - `rho`: The rho function.
  - `t`: The squared distance, divided by the M-scale.
  - `c2`, `g`: The tuning constant ``c^2`` of the bisquare rho, or the width ``\\gamma`` of the Rocke rho.

# Returns

  - `r::Number`: The value, from zero to one.

# Related

  - [`srho_weight`](@ref)
  - [`m_scale`](@ref)
"""
function srho(::BisquareRho, t::Number, c2::Number)
    return t >= c2 ? one(t) : 1 - (1 - t / c2)^3
end
function srho(::RockeRho, t::Number, g::Number)
    u = t - 1
    return if u < -g
        zero(t)
    elseif u > g
        one(t)
    else
        u / (4 * g) * (3 - abs2(u / g)) + one(t) / 2
    end
end
"""
    srho_weight(rho::BisquareRho, t::Number, c2::Number) -> Number
    srho_weight(rho::RockeRho, t::Number, g::Number) -> Number

Returns the weight of an observation at the squared distance `t`, see [`BisquareRho`](@ref) and [`RockeRho`](@ref).

# Arguments

  - `rho`: The rho function.
  - `t`: The squared distance, divided by the M-scale.
  - `c2`, `g`: The tuning constant ``c^2`` of the bisquare rho, or the width ``\\gamma`` of the Rocke rho.

# Returns

  - `w::Number`: The weight, from zero to one. The bisquare weight is one at the centre and zero beyond ``c^2``. The Rocke weight is one at ``t = 1`` and zero outside ``1 \\pm \\gamma``.

# Related

  - [`srho`](@ref)
  - [`s_isteps`](@ref)
"""
function srho_weight(::BisquareRho, t::Number, c2::Number)
    return t >= c2 ? zero(t) : (1 - t / c2)^2
end
function srho_weight(::RockeRho, t::Number, g::Number)
    return abs(t - 1) > g ? zero(t) : 1 - abs2((t - 1) / g)
end
"""
    s_consistency(rho::BisquareRho, d2::VecNum, p::Integer) -> Number
    s_consistency(rho::RockeRho, d2::VecNum, p::Integer) -> Number

Returns the factor that makes the covariance of an S fit consistent at the normal model.

The constant of the bisquare rho already makes the M-scale consistent, so its factor is one. The Rocke rho is centred at the typical distance, so its factor is the median of the squared distances over the median of the ``\\chi^2_p`` distribution, as for [`TylerCovariance`](@ref).

# Arguments

  - `rho`: The rho function.
  - `d2`: The squared distance of each observation, in the metric of the scale times the shape.
  - `p`: The number of assets.

# Returns

  - `f::Number`: The factor of the covariance.

# Related

  - [`AbstractSRho`](@ref)
  - [`srho_fit`](@ref)
"""
function s_consistency(::BisquareRho, d2::VecNum, ::Integer)
    return one(eltype(d2))
end
function s_consistency(::RockeRho, d2::VecNum, p::Integer)
    return Statistics.median(d2) / Distributions.quantile(Distributions.Chisq(p), 0.5)
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
    for _ in 1:S_MAX_ITER
        sn = s * Statistics.mean(x -> srho(rho, x / s, k), t) / b
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
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the parts of an M-estimate whose efficiency at the normal model tunes the constant of [`BisquareReweighting`](@ref).

The bisquare M-step has one constant, so it fixes the efficiency of one part. [`ShapeEfficiency`](@ref) and [`LocationEfficiency`](@ref) are its subtypes. At one constant, the efficiency of the location is the higher of the two, so the constant of the shape at 95 % is the larger one.

# Interfaces

To implement a new target, subtype `AbstractEfficiencyTarget`, and implement one method:

  - `PortfolioOptimisers.bisquare_efficiency(target, p::Integer, c2::Number) -> Number`: Returns the asymptotic efficiency of that part of a bisquare M-estimate with the constant `c2`, for `p` assets. It rises with `c2`.

# Related

  - [`ShapeEfficiency`](@ref)
  - [`LocationEfficiency`](@ref)
  - [`BisquareReweighting`](@ref)
"""
abstract type AbstractEfficiencyTarget <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Tunes the bisquare M-step to the efficiency of the shape, the default of [`BisquareReweighting`](@ref) and of `CovMMest` of the R package `rrcov`, version 1.7-7. A covariance is a shape and a scale, and the scale of an MM estimate comes from its S estimate, so this target sets the efficiency of the covariance.

# Examples

```jldoctest
julia> ShapeEfficiency()
ShapeEfficiency()
```

# Related

  - [`AbstractEfficiencyTarget`](@ref)
  - [`LocationEfficiency`](@ref)
"""
struct ShapeEfficiency <: AbstractEfficiencyTarget end
"""
$(DocStringExtensions.TYPEDEF)

Tunes the bisquare M-step to the efficiency of the location, as `CovMMest` of the R package `rrcov`, version 1.7-7, does with `eff.shape = FALSE`. The constant is smaller than the one of [`ShapeEfficiency`](@ref) at the same efficiency, so the step rejects outliers sooner and the shape has a lower efficiency.

# Examples

```jldoctest
julia> LocationEfficiency()
LocationEfficiency()
```

# Related

  - [`AbstractEfficiencyTarget`](@ref)
  - [`ShapeEfficiency`](@ref)
"""
struct LocationEfficiency <: AbstractEfficiencyTarget end
"""
    bisquare_efficiency(target::ShapeEfficiency, p::Integer, c2::Number) -> Number
    bisquare_efficiency(target::LocationEfficiency, p::Integer, c2::Number) -> Number

Returns the asymptotic efficiency of a part of a bisquare M-estimate of location and scatter at the normal model, relative to the sample mean and the sample covariance.

[salibianbarrera2006](@cite) states the asymptotic variance ``\\sigma_1`` of the shape and the asymptotic variance ``\\sigma_\\mu`` of the location of an M-estimator. The formulas below write them for the bisquare function in the squared distance ``t``. The efficiency is ``1 / \\sigma_1`` or ``1 / \\sigma_\\mu``. It rises from zero to one as ``c^2`` rises. At 95 % and three assets, the constant of the shape is ``6.0963^2`` and the one of the location is ``5.4902^2``, the values of the R package `rrcov`, version 1.7-7. For one asset, the constant of the location is ``4.685^2``, the classical constant of the bisquare M-estimator of location.

# Mathematical definition

```math
\\begin{align}
\\sigma_1 &= \\frac{p (p + 2) \\, \\mathbb{E}\\left[t^2 (1 - t/c^2)^4 \\mathbb{1}\\{t \\le c^2\\}\\right]}{\\mathbb{E}\\left[t (1 - t/c^2) \\left((p + 2) - (p + 6) t / c^2\\right) \\mathbb{1}\\{t \\le c^2\\}\\right]^2}\\,, \\\\
\\sigma_\\mu &= \\frac{\\mathbb{E}\\left[t (1 - t/c^2)^4 \\mathbb{1}\\{t \\le c^2\\}\\right] / p}{\\mathbb{E}\\left[\\left(1 - (2 + 4/p) t / c^2 + (1 + 4/p) t^2 / c^4\\right) \\mathbb{1}\\{t \\le c^2\\}\\right]^2}\\,.
\\end{align}
```

Where:

  - ``t``: A variable of the ``\\chi^2_p`` distribution.

# Arguments

  - `target`: The part of the estimate.
  - `p`: The number of assets.
  - `c2`: The tuning constant ``c^2``.

# Returns

  - `eff::Number`: The efficiency, from zero to one.

# Related

  - [`AbstractEfficiencyTarget`](@ref)
  - [`BisquareReweighting`](@ref)
  - [`chisq_truncated_moment`](@ref)

# References

  - $(ref_dict[:salibianbarrera2006]) Not read. The formulas agree with the R package `rrcov`, version 1.7-7.
"""
function bisquare_efficiency(::LocationEfficiency, p::Integer, c2::Number)
    a = inv(c2)
    num = chisq_truncated_moment(p, c2, [one(a), -(2 + 4 / p) * a, (1 + 4 / p) * a^2])
    den = chisq_truncated_moment(p, c2, [zero(a), one(a), -4 * a, 6 * a^2, -4 * a^3, a^4])
    return abs2(num) / (den / p)
end
function bisquare_efficiency(::ShapeEfficiency, p::Integer, c2::Number)
    a = inv(c2)
    num = p *
          (p + 2) *
          chisq_truncated_moment(p, c2,
                                 [zero(a), zero(a), one(a), -4 * a, 6 * a^2, -4 * a^3, a^4])
    den = chisq_truncated_moment(p, c2,
                                 [zero(a), p + 2 * one(a), -(2p + 8) * a, (p + 6) * a^2])
    return abs2(den) / num
end

public AbstractSRho, s_tuning, srho, srho_weight, s_consistency, AbstractEfficiencyTarget,
       bisquare_efficiency
export BisquareRho, RockeRho, ShapeEfficiency, LocationEfficiency
