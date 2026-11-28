"""
    mcd_subset_size(shr::Nothing, alpha::Number, n::Integer, p::Integer) -> Int
    mcd_subset_size(shr::MCDShrinkage, alpha::Number, n::Integer, p::Integer) -> Int

Returns the size `h` of the MCD subset for the subset fraction `alpha`.

At `alpha = 1/2`, ``h = \\lfloor (n + p + 1)/2 \\rfloor``, the size of the largest breakdown point. At `alpha = 1`, ``h = n``. The function interpolates between the two, as the function `h.alpha.n` of the R package `robustbase`, version 0.99-7, does. That function takes the floor of a floating-point product, which can be one too small. This function takes the exact floor, with [`fraction_floor`](@ref).

With a shrinkage, the scatter of a subset exists for every size, so the size does not depend on the number of assets. It is ``\\lceil \\alpha n \\rceil``, as [boudt2020](@cite) has it, with the exact ceiling of [`fraction_ceil`](@ref), and at least the smallest number above ``n/2``, which Theorem 1 of that work needs.

# Mathematical definition

```math
\\begin{align}
h_2 &= \\lfloor (n + p + 1)/2 \\rfloor\\,, \\\\
h &= \\lfloor 2 h_2 - n + 2 (n - h_2) \\alpha \\rfloor\\,.
\\end{align}
```

# Arguments

  - `shr`: The shrinkage, or `nothing`.
  - `alpha`: The subset fraction, `1/2 <= alpha <= 1`.
  - `n`: The number of observations.
  - `p`: The number of assets.

# Validation

  - With no shrinkage, `n > p`, which is the same condition as `h > p`, see [`assert_robust_rows`](@ref). With a shrinkage, `h > 1`. Otherwise, the function raises a `DomainError`.

# Returns

  - `h::Int`: The size of the subset.

# Related

  - [`MCDCovariance`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 4, step 1.
  - $(ref_dict[:boudt2020]) Section 2.1 and Section 3, Theorem 1.
"""
function mcd_subset_size(::Nothing, alpha::Number, n::Integer, p::Integer)
    assert_robust_rows(n, p, "the number of observations")
    h2 = div(n + p + 1, 2)
    return 2 * h2 - n + fraction_floor(alpha, 2 * (n - h2))
end
function mcd_subset_size(::MCDShrinkage, alpha::Number, n::Integer, p::Integer)
    h = max(fraction_ceil(alpha, n), div(n, 2) + 1)
    @argcheck(1 < h <= n,
              DomainError((n, p),
                          "a shrunk MCD needs a subset of two observations or more, got $n observations of $p assets"))
    return h
end
"""
    mcd_consistency(p::Integer, a::Number) -> Number

Returns the factor that makes the covariance of the `a` fraction of the observations nearest to the centre consistent at the normal model.

The covariance of the observations inside the ellipsoid of probability ``a`` of a normal distribution is smaller than the covariance of the distribution, by the factor ``F_{\\chi^2_{p+2}}(\\chi^2_{p,a})/a``. [croux1999](@cite) gives the factor that corrects it. No copy of that work was found, so the formula follows from the moments of the truncated normal distribution, and it agrees with the code of the R package `robustbase`, version 0.99-7. [`MCDCovariance`](@ref) applies it to the raw MCD with ``a = h/n``, and to the reweighted MCD with the probability of the cut-off of the reweighting.

# Mathematical definition

```math
\\begin{align}
c_{p,a} &= \\frac{a}{F_{\\chi^2_{p+2}}\\left(\\chi^2_{p,a}\\right)}\\,.
\\end{align}
```

Where:

  - ``\\chi^2_{p,a}``: The quantile ``a`` of the ``\\chi^2_p`` distribution.
  - ``F_{\\chi^2_{p+2}}``: The distribution function of the ``\\chi^2_{p+2}`` distribution.

# Arguments

  - `p`: The number of assets.
  - `a`: The probability, `0 < a <= 1`.

# Returns

  - `c::Number`: The factor, one at `a = 1`.

# Related

  - [`MCDCovariance`](@ref)
  - [`mcd_correction`](@ref)

# References

  - $(ref_dict[:croux1999])
"""
function mcd_consistency(p::Integer, a::Number)
    qa = Distributions.quantile(Distributions.Chisq(p), a)
    return a / Distributions.cdf(Distributions.Chisq(p + 2), qa)
end
"""
$(DocStringExtensions.TYPEDEF)

Corrects the bias of the MCD covariance in a small sample with the factors of [pison2002](@cite).

The consistency factor of [`mcd_consistency`](@ref) is exact as ``n`` grows. In a small sample the MCD covariance is still too small. [pison2002](@cite) simulates the bias at the subset fractions 1/2 and 7/8 for many ``n`` and ``p``. It fits the factor to a power of ``n`` whose coefficients are a power of ``p``, and interpolates in the subset fraction. [`mcd_correction`](@ref) computes the factor. The raw MCD and the reweighted MCD have separate coefficients.

The coefficients are those of the R package `robustbase`, version 0.99-7, because [pison2002](@cite) prints the coefficients of the least trimmed squares regression alone. For ``n \\leq 20`` and ``p \\leq 9`` at the subset fraction 1/2, that package replaces the fitted factor with a table of simulated values. This type uses the fitted factor at every ``n``, so it differs from that package in such a small sample.

!!! warning

    The figures of [pison2002](@cite) start at 20 observations. For a smaller ``n`` that is near to ``2p``, the fitted factor is too large. At the subset fraction 1/2, ``n = 14`` and ``p = 5``, the factor of the reweighted MCD is 21, and the mean of 1000 simulated normal samples needs 2.3. At ``n = 12`` the factor is negative, and the fit raises a `DomainError`. Use `ssc = nothing` for such a sample.

# Examples

```jldoctest
julia> MCDSmallSampleCorrection()
MCDSmallSampleCorrection()
```

# Related

  - [`MCDCovariance`](@ref)
  - [`mcd_correction`](@ref)
  - [`mcd_consistency`](@ref)

# References

  - $(ref_dict[:pison2002]) Sections 3 and 4.
"""
struct MCDSmallSampleCorrection <: AbstractAlgorithm end
"""
    PISON_COEFFICIENTS

The coefficients of the small-sample factors of the MCD covariance of [`MCDSmallSampleCorrection`](@ref).

The keys are `(:raw, alpha)` and `(:rew, alpha)` for the subset fractions `alpha` of 1/2 and 7/8. Each value holds the pairs `(alfaq, betaq)` at the multipliers `q = 2` and `q = 3` of ``p^2`` for ``p > 2``, and the pairs `(log γ, β)` of the factor ``1 - γ / n^β`` for ``p = 2`` and ``p = 1``. The values are those of the R package `robustbase`, version 0.99-7.

# Related

  - [`mcd_correction`](@ref)
"""
const PISON_COEFFICIENTS = Dict((:raw, 0.5) => (((-1.42764571687802, 1.26263336932151),
                                                 (-1.06141115981725, 1.28907991440387)),
                                                (0.673292623522027, 0.691365864961895),
                                                (0.262024211897096, 0.604756680630497)),
                                (:raw, 0.875) => (((-0.455179464070565, 1.11192541278794),
                                                   (-0.294241208320834, 1.09649329149811)),
                                                  (0.446537815635445, 1.06690782995919),
                                                  (-0.351584646688712, 1.01646567502486)),
                                (:rew, 0.5) => (((-1.02842572724793, 1.67659883081926),
                                                 (-0.26800273450853, 1.35968562893582)),
                                                (3.11101712909049, 1.91401056721863),
                                                (1.11098143415027, 1.5182890270453)),
                                (:rew, 0.875) => (((-0.544482443573914, 1.25994483222292),
                                                   (-0.343791072183285, 1.25159004257133)),
                                                  (0.79473550581058, 1.10081930350091),
                                                  (-0.66046776772861, 0.88939595831888)))
"""
    pison_fraction_factor(kind::Symbol, alpha::Number, p::Integer, n::Integer) -> Number

Returns the fitted ratio ``f`` of the expected MCD covariance to the true covariance, at one of the two simulated subset fractions.

For ``p > 2``, the function solves the two equations ``\\log(-a_q / p^{b_q}) = c_0 - c_1 \\log(q p^2)``, `q = 2, 3`, for ``(c_0, c_1)``. The ratio is then ``1 - e^{c_0} / n^{c_1}``. For ``p \\leq 2`` the coefficients of the ratio are stated directly.

# Arguments

  - `kind`: `:raw` for the raw MCD, or `:rew` for the reweighted MCD.
  - `alpha`: The simulated subset fraction, `0.5` or `0.875`.
  - `p`: The number of assets.
  - `n`: The number of observations.

# Returns

  - `f::Number`: The ratio, below one.

# Related

  - [`mcd_correction`](@ref)
  - [`PISON_COEFFICIENTS`](@ref)

# References

  - $(ref_dict[:pison2002]) Section 3.3, Equations 4 to 7.
"""
function pison_fraction_factor(kind::Symbol, alpha::Number, p::Integer, n::Integer)
    big, c_p2, c_p1 = PISON_COEFFICIENTS[(kind, alpha)]
    if p > 2
        (a2, b2), (a3, b3) = big
        y2, y3 = log(-a2 / p^b2), log(-a3 / p^b3)
        x2, x3 = -log(2 * p^2), -log(3 * p^2)
        c1 = (y2 - y3) / (x2 - x3)
        c0 = y2 - c1 * x2
        return 1 - exp(c0) / n^c1
    end
    lg, b = p == 2 ? c_p2 : c_p1
    return 1 - exp(lg) / n^b
end
"""
    mcd_correction(ssc::Option{<:MCDSmallSampleCorrection}, p::Integer, n::Integer,
                   alpha::Number, kind::Symbol) -> Number

Returns the small-sample factor of the MCD covariance, or one when `ssc` is `nothing`.

The factor is the reciprocal of the ratio of [`pison_fraction_factor`](@ref), interpolated in the subset fraction `alpha`. Between 1/2 and 7/8 the interpolation is linear. Above 7/8 it is linear between the ratio at 7/8 and one at `alpha = 1`.

# Arguments

  - `ssc`: The correction, or `nothing` for no correction.
  - `p`: The number of assets.
  - `n`: The number of observations.
  - `alpha`: The subset fraction.
  - `kind`: `:raw` for the raw MCD, or `:rew` for the reweighted MCD.

# Validation

  - The interpolated ratio is positive. Otherwise, the sample is too small for the fitted factor, and the method raises a `DomainError`.

# Returns

  - `f::Number`: The factor.

# Related

  - [`MCDSmallSampleCorrection`](@ref)
  - [`pison_fraction_factor`](@ref)

# References

  - $(ref_dict[:pison2002]) Section 3.2.
"""
function mcd_correction(::Nothing, p::Integer, n::Integer, alpha::Number, kind::Symbol)
    return 1
end
function mcd_correction(::MCDSmallSampleCorrection, p::Integer, n::Integer, alpha::Number,
                        kind::Symbol)
    f500 = pison_fraction_factor(kind, 0.5, p, n)
    f875 = pison_fraction_factor(kind, 0.875, p, n)
    f = if alpha <= 0.875
        f500 + (f875 - f500) / 0.375 * (alpha - 0.5)
    else
        f875 + (1 - f875) / 0.125 * (alpha - 0.875)
    end
    @argcheck(f > zero(f),
              DomainError((n, p),
                          "the small-sample correction of the MCD has no positive factor for $n observations of $p assets. Use a longer window, or set `ssc = nothing`"))
    return inv(f)
end

export MCDSmallSampleCorrection
