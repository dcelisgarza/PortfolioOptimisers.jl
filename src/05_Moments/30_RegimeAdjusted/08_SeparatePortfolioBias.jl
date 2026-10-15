"""
    portfolio_regime_bias(::Nothing, ce::RegimeAdjustedExpWeightedCovariance, target::PortfolioTarget, C::MatNum, idx::AbstractVector{<:Integer}, K::Integer)
    portfolio_regime_bias(bias::NamedTuple, ce::RegimeAdjustedExpWeightedCovariance, target::PortfolioTarget, C::MatNum, idx::AbstractVector{<:Integer}, K::Integer)

Returns the bias factor of each direction of the portfolio target, at `K` observations.

On one decay every direction reads the table of [`regime_bias!`](@ref) at `decay`, on the law of
the centring of the estimator. On the separate correlation path the estimated variance of a
direction over its true variance, ``Q``, reads the variances at ``\\lambda`` = `decay` and the
correlation at ``\\lambda_{c}`` = `cor_decay`, so each direction mixes the two pre-centred tables
by a weight ``\\theta`` and divides by ``1 + \\mu``, the shift of the mean of ``Q``, from
[`separate_direction_moments`](@ref):

```math
\\begin{align}
f_{p} &= \\frac{f_{\\lambda_{c}} + \\theta_{p} \\left(f_{\\lambda} - f_{\\lambda_{c}}\\right)}{1 + \\mu_{p}}\\,.
\\end{align}
```

To the second order ``Q - 1`` is ``\\sum_{t} w^{c}_{t} (u_{t}^{2} - 1) + \\sum_{t} (w_{t} - w^{c}_{t}) \\xi_{t}``,
with ``u_{t}`` the standardised return of the direction and ``\\xi_{t}`` the weighted mean of the
squared standardised returns of its assets, so the variance of ``Q`` is the variance of one table
at ``\\lambda_{c}`` plus the terms that ``\\theta`` carries. The tables read every moment of their own
law, and the mix holds the second order of each method. The mean of ``Q`` moves at the second order
by the bias of the estimated correlation, by the root of the two estimated variances, and by
their covariance, which ``\\mu`` holds.

On iid Normal returns at 4, 8 and 12 assets, half-lives of 10, 5 and 20 against correlation
half-lives of 20, 40 and 10, ``R = I``, equicorrelations of 0.5 and 0.8 and a three-factor model,
and an equal-weight and an inverse-volatility direction, the true factor of the steady state over
the factor of one table at `cor_decay` reads 0.9716 to 1.1076. Over the mixed factor it reads
0.9940 to 1.0023 for the three methods. Through the estimator, at 8 assets, an equicorrelation of
0.5, half-lives of 10 and 20, either centring and both directions, the squared multiplier of the
separate path is within 0.07 % of the multiplier of one decay on the same data over 3 seeds of
120 000 rows, where the factor at `cor_decay` alone read 1.6 % to 2.6 % above it. In the warm-up
the correlation state weights its first products by ``1 / (1 - \\lambda^{t})``, which neither
factor holds (#1729): at 12 assets, an equicorrelation of 0.5, half-lives of 10 and 20 and 20
observations the ratio is 1.080, from 1.101 at `cor_decay` alone, and it falls to 1.000 by 200
observations. Under HAC the three tables and the weight sums read the Bartlett weights of
`hac_lags`. At two lags and equal weights, on the same data with the first scored row of a zero block left out
(#1730), the separate path reads 0.06 % to 0.89 % below one decay for the root mean square, 0.07 %
to 0.61 % for the first moment and 0.05 % to 0.35 % for the log, where the factor at `cor_decay`
alone read 3.1 % to 5.1 % above it.

# Arguments

  - `bias`: The store of [`regime_bias_store`](@ref), or `nothing` where the estimator takes no
    correction.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `target::PortfolioTarget`: Portfolio regime-adjustment target.
  - `C::MatNum`: Bias-corrected covariance block of the contributing assets.
  - `idx::AbstractVector{<:Integer}`: Index of those assets in the universe.
  - `K::Integer`: Count of observations in the estimate.

# Returns

  - `f::Union{<:Number, <:VecNum}`: One where `bias` is `nothing`, the factor of every direction
    on one decay, and the factor of each direction on the separate path.

# Related

  - [`regime_target_statistic`](@ref)
  - [`separate_direction_moments`](@ref)
  - [`separate_path_sums!`](@ref)
  - [`portfolio_directions`](@ref)
"""
function portfolio_regime_bias(::Nothing, ce::RegimeAdjustedExpWeightedCovariance,
                               ::PortfolioTarget, ::MatNum, ::AbstractVector{<:Integer},
                               ::Integer)
    return one(ce.decay)
end
function portfolio_regime_bias(bias::NamedTuple, ce::RegimeAdjustedExpWeightedCovariance,
                               target::PortfolioTarget, C::MatNum,
                               idx::AbstractVector{<:Integer}, K::Integer)
    if !has_separate_cor_decay(ce)
        return regime_bias!(bias.table, ce.regime_method, ce.decay, K, ce.hac_lags,
                            ce.centring, ce.debias)
    end
    fc = regime_bias!(bias.table, ce.regime_method, ce.cor_decay, K, ce.hac_lags)
    fv = regime_bias!(bias.variance, ce.regime_method, ce.decay, K, ce.hac_lags)
    sums = separate_path_sums!(bias.convolution, ce, K)
    W = portfolio_directions(target, C, idx, ce.min_val)
    return map(eachrow(W)) do p
        theta, mu = separate_direction_moments(C, p, ce.min_val, sums)
        return (fc + theta * (fv - fc)) / (one(mu) + mu)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight of the table at `decay` and the shift of the mean of the estimated variance
of one direction on the separate correlation path, from the block that stands for the true
covariance.

With ``a = D p``, ``D`` the volatilities and ``R`` the correlation of the block, the share of the
noise of the variances in the variance of the direction is ``\\gamma`` alone, and in its square
``\\beta``. Both are one at one asset and at a correlation of one, where the direction reads the
variance alone, and ``1 / n`` at ``R = I`` and equal ``a``, where the noise of ``n`` variances
averages out.

# Mathematical definition

```math
\\begin{align}
b_{i} &= \\frac{a_{i} (R a)_{i}}{a^{\\top} R a}\\,, \\quad
\\gamma = \\sum_{i} b_{i} \\frac{(R a)_{i}^{2}}{a^{\\top} R a}\\,, \\quad
\\beta = \\sum_{i, j} b_{i} b_{j} R_{ij}^{2}\\,, \\\\
\\theta &= \\frac{2 \\gamma A + \\beta B}{2 A + B}\\,, \\quad
\\mu = \\frac{\\sum_{i \\neq j} a_{i} a_{j} R_{ij} (1 - R_{ij}^{2})}{a^{\\top} R a}\\, S\\,.
\\end{align}
```

Where:

  - ``A``, ``B``, ``S``: The sums `mix`, `spread` and `shift` of [`separate_path_sums!`](@ref).

# Arguments

  - `C::MatNum`: Bias-corrected covariance block of the contributing assets.
  - `p::AbstractVector`: Weights of the direction over those assets.
  - `min_val::Number`: Floor applied to each variance.
  - `sums::NamedTuple`: `mix`, `spread` and `shift` of [`separate_path_sums!`](@ref).

# Returns

  - `(theta, mu)::Tuple{<:Number, <:Number}`: The weight ``\\theta`` and the shift ``\\mu``, both
    zero where the variance of the direction is not positive.

# Related

  - [`portfolio_regime_bias`](@ref)
  - [`separate_path_sums!`](@ref)
"""
function separate_direction_moments(C::MatNum, p::AbstractVector, min_val::Number,
                                    sums::NamedTuple)
    Cp = C * p
    v = LinearAlgebra.dot(p, Cp)
    if !(v > zero(v))
        return zero(v), zero(v)
    end
    s2 = max.(LinearAlgebra.diag(C), min_val)
    b = p .* Cp ./ v
    g = sum(b[i] * Cp[i]^2 / s2[i] for i in eachindex(b, Cp, s2)) / v
    be = zero(v)
    m = zero(v)
    for j in axes(C, 2), i in axes(C, 1)
        r2 = C[i, j]^2 / (s2[i] * s2[j])
        be += b[i] * b[j] * r2
        if i != j
            m += p[i] * p[j] * C[i, j] * (one(r2) - r2)
        end
    end
    den = 2 * sums.mix + sums.spread
    theta = iszero(den) ? zero(v) : (2 * g * sums.mix + be * sums.spread) / den
    return theta, m / v * sums.shift
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the three sums of the weights that [`separate_direction_moments`](@ref) reads at `K`
observations, and grows the table of the convolution where the count is past its end.

The variance at ``\\lambda`` = `decay` and the correlation at ``\\lambda_{c}`` = `cor_decay` give the
sums of the products of their weights, ``s_{\\lambda\\lambda}``, ``s_{\\lambda\\lambda_{c}}`` and
``s_{\\lambda_{c}\\lambda_{c}}``, from [`exp_weight_cross_sum`](@ref). The correlation state divides
each product by the running volatility, whose noise reaches the correlation through the
convolution of the two sets of weights, ``Y`` of [`exp_weight_convolution_table`](@ref). The table
grows as the tables of [`regime_bias!`](@ref) do. Under HAC each sum reads the weights ``k_{i}``
of `hac_lags`: a lag ``i`` adds ``2 k_{i}^{2}`` times the part of the sum whose older observation
is in the window, as [`exp_weight_cross_sum`](@ref) does, and the convolution weights depend on
``K`` only through their normalisation.

# Mathematical definition

```math
\\begin{align}
A &= s_{\\lambda\\lambda_{c}} - s_{\\lambda_{c}\\lambda_{c}}\\,, \\quad
B = s_{\\lambda\\lambda} - 2 s_{\\lambda\\lambda_{c}} + s_{\\lambda_{c}\\lambda_{c}}\\,, \\quad
S = s_{\\lambda\\lambda_{c}} - s_{\\lambda\\lambda} - \\frac{s_{\\lambda_{c}\\lambda_{c}} - Y^{h}_{K}}{2}\\,, \\\\
Y^{h}_{K} &= Y_{K} + 2 \\sum_{i=1}^{\\min(L, K - 1)} k_{i}^{2} Y_{K-i} \\left(\\frac{1 - \\lambda_{c}^{K-i}}{1 - \\lambda_{c}^{K}}\\right)^{2}\\,.
\\end{align}
```

Where:

  - ``Y_{K}``: The table of [`exp_weight_convolution_table`](@ref) at ``K`` observations.
  - ``L``: The count of lags, zero without a HAC adjustment.

# Arguments

  - `conv::VecNum`: The table of ``Y`` of the state (mutated where it grows).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration, on the separate
    path.
  - `K::Integer`: Count of observations in the estimate.

# Returns

  - `sums::NamedTuple`: `mix` ``A``, `spread` ``B`` and `shift` ``S``.

# Related

  - [`separate_direction_moments`](@ref)
  - [`exp_weight_convolution_table`](@ref)
  - [`exp_weight_cross_sum`](@ref)
"""
function separate_path_sums!(conv::VecNum, ce::RegimeAdjustedExpWeightedCovariance,
                             K::Integer)
    lv, lc = ce.decay, ce.cor_decay
    Ksat = ceil(Int, log(eps(eltype(conv))) / log(max(lv, lc)))
    if K > length(conv) && length(conv) < Ksat
        len = min(max(2 * K, 64), Ksat)
        resize!(conv, len)
        copyto!(conv, exp_weight_convolution_table(lv, lc, len))
    end
    Kt = min(K, length(conv))
    Y = conv[Kt]
    # A lag reaches the products whose older observation is in the window (#1726).
    k = hac_lag_weights(ce.hac_lags, Y)
    for i in 1:min(length(k), Kt - 1)
        Y += 2 * k[i]^2 * conv[Kt - i] * ((one(lc) - lc^(Kt - i)) / (one(lc) - lc^Kt))^2
    end
    sv = exp_weight_cross_sum(lv, lv, K, ce.hac_lags)
    svc = exp_weight_cross_sum(lv, lc, K, ce.hac_lags)
    sc = exp_weight_cross_sum(lc, lc, K, ce.hac_lags)
    return (; mix = svc - sc, spread = sv - 2 * svc + sc, shift = svc - sv - (sc - Y) / 2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the sum of the squares of the convolution of two sets of exponential weights, for every
count of observations from one to `len`.

The weight of the observation ``m`` steps before the newest in the correlation state, through the
running variance it divides by, is ``c_{m} = \\sum_{l=0}^{m} w^{c}_{l} w_{m-l}``. The recursion of
``h_{m}`` has no difference of the two decays, so it holds where they are close.

# Mathematical definition

```math
\\begin{align}
Y_{K} &= \\sum_{m=0}^{K-1} c_{m}^{2} = \\frac{(1 - \\lambda)^{2} (1 - \\lambda_{c})^{2}}{(1 - \\lambda_{c}^{K})^{2}} \\sum_{m=0}^{K-1} h_{m}^{2}\\,, \\quad
h_{m} = \\lambda h_{m-1} + \\lambda_{c}^{m}\\,, \\quad h_{0} = 1\\,.
\\end{align}
```

Where:

  - ``w_{l} = (1 - \\lambda) \\lambda^{l}``: Weight of the running variance at ``\\lambda`` = `decay`.
  - ``w^{c}_{l} = (1 - \\lambda_{c}) \\lambda_{c}^{l} / (1 - \\lambda_{c}^{K})``: Normalised weight of
    the correlation at ``\\lambda_{c}`` = `cor_decay`.

# Arguments

  - `decay::Number`: Decay of the variance.
  - `cor_decay::Number`: Decay of the correlation.
  - `len::Integer`: Largest count of observations in the table.

# Returns

  - `table::VecNum`: ``Y_{K}`` for each count from one to `len`.

# Related

  - [`separate_path_sums!`](@ref)
"""
function exp_weight_convolution_table(decay::Number, cor_decay::Number, len::Integer)
    c = ((one(decay) - decay) * (one(cor_decay) - cor_decay))^2
    h = zero(c)
    acc = zero(c)
    lc = one(cor_decay)
    table = Vector{typeof(c)}(undef, len)
    for K in 1:len
        h = decay * h + lc
        acc += h^2
        lc *= cor_decay
        table[K] = c * acc / (one(lc) - lc)^2
    end
    return table
end
