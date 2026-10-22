"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber IQ covariance estimators.

Every concrete or abstract type that implements a Gerber IQ covariance estimator must subtype `BaseGerberIQCovariance`. Gerber IQ is the Gerber Information Quality statistic.

The family adds two things to [`BaseGerberCovariance`](@ref). It gives a co-movement the weight of the channel of the return plane that the co-movement falls in, not a count of one. It also discounts a co-movement by its age. [`GerberCovarianceAlgorithm`](@ref) states the Gerber statistic that these estimators reduce to, and this docstring does not repeat it.

# Interfaces

A new subtype must follow the interface of [`AbstractCovarianceEstimator`](@ref).

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.
"""
abstract type BaseGerberIQCovariance <: BaseGerberCovariance end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber IQ squeezing templates.

Every concrete or abstract type that implements a Gerber IQ squeezing template must subtype `GerberIQCovarianceAlgorithm`. A subtype goes in the `kind` field of a [`GerberIQCovariance`](@ref) estimator.

A squeezing template cuts the plane of the returns of two assets into channels. It gives the weight of a co-movement in each channel. The library ships the three templates of the source. In the order of the number of channels, they are:

  - [`BasicGerberIQ`](@ref), with one boundary and one weight.
  - [`PartialGerberIQ`](@ref), with four boundaries and ten weights.
  - [`FullGerberIQ`](@ref), with four boundaries and twenty-one weights over thirty-six channels.

# Interfaces

To add a squeezing template, subtype `GerberIQCovarianceAlgorithm` with its boundaries and its weights as fields. Then implement these methods:

## `gerber_iq_weight`

  - `gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sci::Number, scj::Number, kind::MyGerberIQ) -> Number`: The weight of one co-movement, from the channel of the template that the co-movement falls in.

### Arguments

  - `xi`, `xj`: Returns of assets `i` and `j`.
  - `axi`, `axj`: Absolute returns of assets `i` and `j`.
  - `sci`, `scj`: Scaling factors of assets `i` and `j`.
  - `kind`: The instance of the subtype.

### Returns

  - `res::Number`: Weight of the co-movement.

## `gerber_iq_assert_c_d`

  - `gerber_iq_assert_c_d(c::Number, kind::MyGerberIQ) -> Nothing`: Compares the noise threshold `c` with each boundary of the template. It raises a `DomainError` when `c` is above a boundary.

### Arguments

  - `c`: Noise threshold.
  - `kind`: The instance of the subtype.

### Returns

  - `nothing`. The caller uses the method for its check alone.

## `clamp_gerber_iq_n` (optional)

  - `clamp_gerber_iq_n(kind::MyGerberIQ, alg::GerberCovarianceAlgorithm) -> MyGerberIQ`: The template, with each weight lowered that `alg` needs lower. The fall-through returns `kind` unchanged.

### Arguments

  - `kind`: The instance of the subtype.
  - `alg`: The [`GerberCovarianceAlgorithm`](@ref) marker.

### Returns

  - `kind::MyGerberIQ`: A new template, or `kind` when no weight needs a clamp.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`BasicGerberIQ`](@ref)
  - [`PartialGerberIQ`](@ref)
  - [`FullGerberIQ`](@ref)
  - [`gerber_iq_weight`](@ref)
  - [`gerber_iq_assert_c_d`](@ref)
  - [`clamp_gerber_iq_n`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Sections 3.3.3 to 3.3.5.
"""
abstract type GerberIQCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a Gerber IQ squeezing template unchanged, for each pair of a template and a marker that needs no clamp.

Every pair of a template and a [`GerberCovarianceAlgorithm`](@ref) marker that this file does not clamp comes to this method. Two cases come to it:

  - [`Gerber0`](@ref) and [`Gerber1`](@ref) divide the net weighted count by a sum of the same weights. So the statistic is in ``[-1, 1]`` for every template.
  - Under [`Gerber2`](@ref), [`BasicGerberIQ`](@ref) needs no clamp. Its diagonal weights are `1` and `n`, and its one mixed weight `n^2` is at most their geometric mean ``\\sqrt{n}``.

# Arguments

  - `kind`: The squeezing template.
  - `args...`: The [`GerberCovarianceAlgorithm`](@ref) marker. The method does not read it.

# Returns

  - `kind`: The input `kind` instance.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
"""
function clamp_gerber_iq_n(kind::GerberIQCovarianceAlgorithm, args...)
    return kind
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimators of the delay of the Gerber IQ decay.

Every concrete or abstract type that implements an estimator of the Gerber IQ delay must subtype `GerberIQEpsEstimator`.

A subtype computes the delay of the temporal vector of the source. The delay is the number of periods before the last observation in which a co-movement keeps its full weight. The source names it ``\\varepsilon``, and it leaves the value to expert judgement or to an outer optimisation. The library adds a default formula, which [`gerber_iq_eps`](@ref) states.

# Interfaces

To add a delay estimator, subtype `GerberIQEpsEstimator` and implement this method:

## `gerber_iq_eps`

  - `gerber_iq_eps(e::MyGerberIQEps, X::MatNum) -> Number`: The delay, in periods, computed from `X`. A subtype that implements no method gets the fall-through of [`gerber_iq_eps`](@ref).

### Arguments

  - `e`: The instance of the subtype.
  - `X`: Returns matrix, `observations × assets`.

### Returns

  - `eps::Number`: The delay.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQEps`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.2.
"""
abstract type GerberIQEpsEstimator <: AbstractEstimator end
"""
    const GerberIQEps = Union{<:Number, <:Function, <:GerberIQEpsEstimator}

A type alias for the union of `Number`, `Function` and `GerberIQEpsEstimator`, the types that give the delay of the Gerber IQ decay.

Each arm is one way to give the delay. A `Number` is the delay. A `Function` computes it from the returns matrix. A [`GerberIQEpsEstimator`](@ref) computes it through [`gerber_iq_eps`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQEpsEstimator`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.2.
"""
const GerberIQEps = Union{<:Number, <:Function, <:GerberIQEpsEstimator}
"""
    gerber_iq_eps(e::Number, ::MatNum) -> Number
    gerber_iq_eps(e::Function, X::MatNum) -> Number
    gerber_iq_eps(e::Option{<:GerberIQEpsEstimator}, X::MatNum) -> Number

Returns the delay `e` of the Gerber IQ decay. A rule in `e` computes the delay from `X`.

# Mathematical definition

The fall-through computes the delay from the shape of the returns matrix alone.

```math
\\begin{align}
\\varepsilon &= \\mathrm{round}\\left(T - \\frac{T}{N}\\right)\\,.
\\end{align}
```

Where:

  - ``\\varepsilon``: Delay. A co-movement of age ``T - k`` keeps its full weight when ``T - k \\leq \\varepsilon``.
  - $(math_dict[:T])
  - $(math_dict[:N])

The decay discounts an observation only when its index satisfies ``k < T - \\varepsilon``. So this default discounts about the oldest ``T/N`` observations, and the other observations keep their full weight. The formula is the library's, not the formula of the source. The source states only that the delay is a whole number of periods, no larger than the lookback duration. It leaves the value to expert judgement or to an outer optimisation. Table 2 of [gerber2025squeezing](@cite) sets the delay to zero in the study of the source.

# Algorithm

 1. Return the number unchanged when `e` is a `Number`.
 2. Call `e(X)` when `e` is a `Function`, and return its result.
 3. Otherwise read `T` and `N` from `size(X)`, and return `round(Int, T - T / N)`.

# Arguments

  - `e`: The delay, or the rule that computes it.
      + `::Number`: The method returns the number unchanged.
      + `::Function`: A function that takes the data matrix `X` and returns a `Number`.
      + `::Option{<:GerberIQEpsEstimator}`: The fall-through. It returns `round(Int, T - T / N)`, where `T` is the number of rows of `X` and `N` the number of columns.
  - $(arg_dict[:X])

# Returns

  - `e::Number`: The delay of the decay. The decay does not discount an observation that is no older than `e` periods. The delay is not the window duration ``\\tau`` of the source, which the estimator does not expose. The estimator reads the whole matrix, and the decay discounts its oldest rows.

# Related

  - [`GerberIQEps`](@ref)
  - [`GerberIQEpsEstimator`](@ref)
  - [`gerber_iq_gamma`](@ref)
  - [`ExpGerberIQDecay`](@ref)
  - [`regenerate_decay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.2, Equation 5.
"""
function gerber_iq_eps(e::Number, ::MatNum)
    return e
end
function gerber_iq_eps(e::Function, X::MatNum)
    return e(X)
end
function gerber_iq_eps(::Option{<:GerberIQEpsEstimator}, X::MatNum)
    T, N = size(X)
    return round(Int, T - T / N)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimators of the rate of the Gerber IQ decay.

Every concrete or abstract type that implements an estimator of the Gerber IQ decay rate must subtype `GerberIQGammaEstimator`.

A subtype computes the decay rate of the temporal vector of the source. The decay applies this rate to a co-movement that is older than the delay. The source names the rate ``\\gamma`` and requires it to be positive. It leaves the value to expert judgement or to an outer optimisation. The library adds a default formula, which [`gerber_iq_gamma`](@ref) states.

# Interfaces

To add a decay rate estimator, subtype `GerberIQGammaEstimator` and implement this method:

## `gerber_iq_gamma`

  - `gerber_iq_gamma(y::MyGerberIQGamma, X::MatNum) -> Number`: The decay rate, computed from `X`. A subtype that implements no method gets the fall-through of [`gerber_iq_gamma`](@ref).

### Arguments

  - `y`: The instance of the subtype.
  - `X`: Returns matrix, `observations × assets`.

### Returns

  - `gamma::Number`: The decay rate.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQGamma`](@ref)
  - [`gerber_iq_gamma`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.3.
"""
abstract type GerberIQGammaEstimator <: AbstractEstimator end
"""
    const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}

A type alias for the union of `Number`, `Function` and `GerberIQGammaEstimator`, the types that give the rate of the Gerber IQ decay.

Each arm is one way to give the decay rate. A `Number` is the rate. A `Function` computes it from the returns matrix. A [`GerberIQGammaEstimator`](@ref) computes it through [`gerber_iq_gamma`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQGammaEstimator`](@ref)
  - [`gerber_iq_gamma`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.3.
"""
const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}
"""
    gerber_iq_gamma(y::Number, ::MatNum) -> Number
    gerber_iq_gamma(y::Function, X::MatNum) -> Number
    gerber_iq_gamma(y::Option{<:GerberIQGammaEstimator}, X::MatNum) -> Number

Returns the rate `y` of the Gerber IQ decay. A rule in `y` computes the rate from `X`.

# Mathematical definition

The fall-through computes the decay rate from the number of assets alone.

```math
\\begin{align}
\\gamma &= \\frac{\\ln 2}{N}\\,.
\\end{align}
```

Where:

  - ``\\gamma``: Decay rate. A larger rate gives an old co-movement a smaller weight.
  - $(math_dict[:N])

The default is a half-life of ``N`` periods. After the delay, the weight of a co-movement becomes half for every ``N`` observations, because ``\\exp(-\\gamma N) = 1/2``. The formula is the library's, not the formula of the source. The source requires only ``\\gamma > 0``, and it leaves the value to expert judgement or to an outer optimisation. Table 2 of [gerber2025squeezing](@cite) relates the rate to the half-life by ``T_{1/2} = \\ln 2 / \\gamma``. This default uses that relation with ``T_{1/2} = N``.

# Algorithm

 1. Return the number unchanged when `y` is a `Number`.
 2. Call `y(X)` when `y` is a `Function`, and return its result.
 3. Otherwise return `log(2) / size(X, 2)`.

# Arguments

  - `y`: The decay rate, or the rule that computes it.
      + `::Number`: The method returns the number unchanged.
      + `::Function`: A function that takes the data matrix `X` and returns a `Number`.
      + `::Option{<:GerberIQGammaEstimator}`: The fall-through. It returns `log(2) / size(X, 2)`.
  - $(arg_dict[:X])

# Returns

  - `gamma::Number`: The rate of the decay.

# Related

  - [`GerberIQGamma`](@ref)
  - [`GerberIQGammaEstimator`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`ExpGerberIQDecay`](@ref)
  - [`regenerate_decay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.3, Equation 5, and Table 2.
"""
function gerber_iq_gamma(y::Number, ::MatNum)
    return y
end
function gerber_iq_gamma(y::Function, X::MatNum)
    return y(X)
end
function gerber_iq_gamma(::Option{<:GerberIQGammaEstimator}, X::MatNum)
    return log(2) / size(X, 2)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all threshold scalers of the Gerber IQ statistic.

Every concrete or abstract type that implements a Gerber IQ threshold scaler must subtype `GerberIQScalerEstimator`.

A subtype sets the units of the thresholds of a pair of assets. The source names four choices:

  - The volatility of each asset.
  - The mean of the two volatilities.
  - The larger of the two volatilities.
  - The smaller of the two volatilities.

The library ships the first choice as [`AssetVolatilityGerberIQScaler`](@ref), and the second as the fall-through of [`gerber_iq_scaling`](@ref). The `Function` arm of [`GerberIQScaler`](@ref) gives the other two.

# Interfaces

To add a scaler, subtype `GerberIQScalerEstimator` and implement this method:

## `gerber_iq_scaling`

  - `gerber_iq_scaling(sca::MyGerberIQScaler, sdi::Number, sdj::Number) -> (Number, Number)`: The scales of the thresholds of assets `i` and `j`, computed from their standard deviations. A subtype that implements no method gets the fall-through of [`gerber_iq_scaling`](@ref).

### Arguments

  - `sca`: The instance of the subtype.
  - `sdi`, `sdj`: Standard deviations of assets `i` and `j`.

### Returns

  - `(sci, scj)::Tuple{Number, Number}`: The scales of assets `i` and `j`.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQScaler`](@ref)
  - [`AssetVolatilityGerberIQScaler`](@ref)
  - [`gerber_iq_scaling`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.1, Section 4 and Table 2.
"""
abstract type GerberIQScalerEstimator <: AbstractEstimator end
"""
    const GerberIQScaler = Union{Function, <:GerberIQScalerEstimator}

A type alias for the union of `Function` and `GerberIQScalerEstimator`, the types that scale the thresholds of the Gerber IQ statistic.

The alias has no `Number` arm, because a scaler is a rule on the two standard deviations of the pair, not a value. A `Function` takes `sdi` and `sdj` and returns the two scaled values. A [`GerberIQScalerEstimator`](@ref) does the same through [`gerber_iq_scaling`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQScalerEstimator`](@ref)
  - [`AssetVolatilityGerberIQScaler`](@ref)
  - [`gerber_iq_scaling`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.1, Section 4 and Table 2.
"""
const GerberIQScaler = Union{Function, <:GerberIQScalerEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Scales the thresholds of each asset by the volatility of that asset.

Each asset keeps its own units. So the scaler measures asset ``i`` against ``\\sigma_i`` and asset ``j`` against ``\\sigma_j``. Kendall's tau and the Gerber statistic use the same convention. With this choice, the Gerber IQ statistic reduces to the Gerber statistic, and [`gerber_IQ`](@ref) states that reduction. The fall-through of [`gerber_iq_scaling`](@ref) gives both assets the mean of the two volatilities.

# Mathematical definition

```math
\\begin{align}
(s_i,\\, s_j) &= (\\sigma_i,\\, \\sigma_j)\\,.
\\end{align}
```

Where:

  - ``s_i``, ``s_j``: Threshold scaling factors of the pair.
  - $(math_dict[:sigma_i_asset])

# Constructors

    AssetVolatilityGerberIQScaler() -> AssetVolatilityGerberIQScaler

# Examples

```jldoctest
julia> AssetVolatilityGerberIQScaler()
AssetVolatilityGerberIQScaler()
```

# Related

  - [`GerberIQScalerEstimator`](@ref)
  - [`GerberIQScaler`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`gerber_iq_scaling`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.1 and Section 4.
"""
struct AssetVolatilityGerberIQScaler <: GerberIQScalerEstimator end
"""
    gerber_iq_scaling(sca::AssetVolatilityGerberIQScaler, sdi::Number, sdj::Number) -> (Number, Number)
    gerber_iq_scaling(sca::Function, sdi::Number, sdj::Number) -> (Number, Number)
    gerber_iq_scaling(sca::Option{<:GerberIQScalerEstimator}, sdi::Number, sdj::Number) -> (Number, Number)

Computes the scaling factors of the thresholds of a pair of assets in the Gerber IQ statistic.

The statistic multiplies each threshold of the pair by the factor that this function returns for the axis of the threshold. The thresholds are the noise threshold `c` and each boundary of the squeezing template. So the scaler sets the units in which a return is large.

A scaler is pair-separable when its first factor uses `sdi` alone and its second factor uses `sdj` alone. The thresholds of an asset are then the same for every partner. [`AssetVolatilityGerberIQScaler`](@ref) is pair-separable. The fall-through is not, because the mean of the pair changes with `sdj`. A `Function` can be either. Every scaler is safe under every marker. [`Gerber2`](@ref) computes its denominator in the units of the pair through [`iq_add_diagonal`](@ref). So when a scaler changes the class of an asset, the numerator and the denominator change together.

# Mathematical definition

```math
\\begin{align}
(s_i,\\, s_j) &= \\begin{cases}
(\\sigma_i,\\, \\sigma_j) & \\text{AssetVolatilityGerberIQScaler} \\\\
\\left(\\dfrac{\\sigma_i + \\sigma_j}{2},\\, \\dfrac{\\sigma_i + \\sigma_j}{2}\\right) & \\text{fall-through}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``s_i``, ``s_j``: Threshold scaling factors of the pair.
  - $(math_dict[:sigma_i_asset])

The fall-through gives both assets the same units, so a volatile asset and a quiet asset have the same absolute threshold. [`AssetVolatilityGerberIQScaler`](@ref) gives each asset its own units. The source names both choices.

# Algorithm

 1. Return `(sdi, sdj)` unchanged for [`AssetVolatilityGerberIQScaler`](@ref).
 2. Call `sca(sdi, sdj)` for a `Function`, and return its result.
 3. Otherwise compute `(sdi + sdj) / 2` once, and return it for both axes.

# Arguments

  - `sca`: The threshold scaler.

      + `::AssetVolatilityGerberIQScaler`: The method returns `sdi` and `sdj` unchanged, so each asset has the units of its own volatility.
      + `::Option{<:GerberIQScalerEstimator}`: The fall-through. It returns the mean of `sdi` and `sdj` for both assets. A new [`GerberIQScalerEstimator`](@ref) subtype adds a method to change this.
      + `::Function`: A function that takes `sdi` and `sdj` and returns the two scaling factors.

  - `sdi`: Standard deviation of asset `i`.

  - `sdj`: Standard deviation of asset `j`.

# Returns

  - `scai::Number`: The scaling factor of asset `i`.
  - `scaj::Number`: The scaling factor of asset `j`.

# Related

  - [`GerberIQScaler`](@ref)
  - [`GerberIQScalerEstimator`](@ref)
  - [`AssetVolatilityGerberIQScaler`](@ref)
  - [`GerberIQKernel`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.1, Section 4 and Table 2.
"""
function gerber_iq_scaling(::AssetVolatilityGerberIQScaler, sdi::Number, sdj::Number)
    return sdi, sdj
end
function gerber_iq_scaling(::Option{<:GerberIQScalerEstimator}, sdi::Number, sdj::Number)
    res = (sdi + sdj) / 2
    return res, res
end
function gerber_iq_scaling(sca::Function, sdi::Number, sdj::Number)
    return sca(sdi, sdj)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber IQ estimators that discount an observation by its age.

Every concrete or abstract type that implements a Gerber IQ decay must subtype `GerberIQDecayEstimator`.

A subtype is a non-increasing function of age. The source states that the age penalty can be any non-increasing function, and it uses the exponential form. [`ExpGerberIQDecay`](@ref) implements that form. A subtype is configuration, so it holds its parameters and never a Result. Two methods make a subtype work. [`regenerate_decay`](@ref) gives a value to each parameter that the caller did not set, and the functor returns the weight of one observation.

# Interfaces

To add a Gerber IQ decay, subtype `GerberIQDecayEstimator` with its parameters as fields. Then implement these methods:

## `regenerate_decay`

  - `regenerate_decay(decay::GerberIQDecayEstimator, X::AbstractMatrix) -> GerberIQDecayEstimator`: The decay with a number in each parameter. The method computes from `X` each parameter that the caller did not set.

### Arguments

  - `decay`: The decay estimator to resolve.
  - $(arg_dict[:X])

### Returns

  - `decay::GerberIQDecayEstimator`: An instance of the subtype with a number in each parameter. It can be `decay` itself when each parameter is already a number.

## Functor

  - `(decay::GerberIQDecayEstimator)(T::Number, k::Number) -> Number`: The weight of observation `k` of `T`.

### Arguments

  - `T::Number`: The number of observations.
  - `k::Number`: The index of the observation.

### Returns

  - `d::Number`: The weight of observation `k` of `T`.

# Examples

This example defines a Gaussian decay estimator.

```jldoctest
julia> struct GaussianDecay{T} <: PortfolioOptimisers.GerberIQDecayEstimator
           a::T
           function GaussianDecay(a::Union{Nothing, <:Number})
               if isa(a, Number)
                   PortfolioOptimisers.@argcheck(a >= 0,
                                                 DomainError(a, \"`a` must be non-negative\"))
               end
               return new{typeof(a)}(a)
           end
       end

julia> function GaussianDecay(; a::Union{Nothing, <:Number} = nothing)
           return GaussianDecay(a)
       end
GaussianDecay

julia> function PortfolioOptimisers.regenerate_decay(decay::GaussianDecay{<:Number},
                                                     ::AbstractMatrix)
           return decay
       end

julia> function PortfolioOptimisers.regenerate_decay(decay::GaussianDecay{Nothing},
                                                     X::AbstractMatrix)
           T = size(X, 1)
           return GaussianDecay(; a = inv(log(T)))
       end

julia> function (decay::GaussianDecay)(T::Number, k::Number)
           m = T - k + 1
           return exp(-m^2 / (2 * decay.a^2))
       end

julia> cor(GerberIQCovariance(; decay = GaussianDecay()), [1.0 2.0; 0.3 0.7; 0.5 1.1])
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0

julia> cov(GerberIQCovariance(; decay = GaussianDecay()), [1.0 2.0; 0.3 0.7; 0.5 1.1])
2×2 Matrix{Float64}:
 0.13      0.240069
 0.240069  0.443333
```

# Related

  - [`GerberIQCovariance`](@ref)
  - [`regenerate_decay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.3.
"""
abstract type GerberIQDecayEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Exponential Gerber IQ temporal decay.

This decay is the age penalty of the source, Equation 5 of [gerber2025squeezing](@cite). The age ``m = T - k`` starts at the most recent observation. A co-movement that is no older than `e` periods keeps its full weight. The decay discounts an older co-movement at the rate `y` for each more period. Each field can be `nothing`. [`regenerate_decay`](@ref) then computes it from the returns matrix before the statistic runs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpGerberIQDecay(e::Option{<:GerberIQEps} = nothing,
                     y::Option{<:GerberIQGamma} = nothing)

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `e` when it is a `Number`.
  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `y` when it is a `Number`.

The constructor does not check a field that is not a `Number`, because that field is a rule, not a value. [`regenerate_decay`](@ref) resolves the rule, and this constructor then checks the resolved instance.

# Functors

    (decay::ExpGerberIQDecay)(T::Number, k::Number) -> Number

Computes the exponential decay of the Gerber IQ statistic.

!!! warning

    Call the functor only when `e` and `y` are both numbers. Otherwise, call [`regenerate_decay`](@ref) first. A field that is `nothing` makes the functor raise a `MethodError`.

## Mathematical definition

```math
\\begin{align}
d &= \\exp\\left[-y \\max(0, T - k - e)\\right]\\,.
\\end{align}
```

Where:

  - $(math_dict[:T])
  - ``k``: Index of the observation.
  - ``e``: Delay, the field `e` of the [`ExpGerberIQDecay`](@ref).
  - ``y``: Decay rate, the field `y` of the [`ExpGerberIQDecay`](@ref).
  - ``d``: Decay factor.

## Arguments

  - `T`: Number of observations.
  - `k`: Index of the observation.

## Returns

  - `d`: The decay factor.

# Examples

```jldoctest
julia> ExpGerberIQDecay()
ExpGerberIQDecay
  e ┼ nothing
  y ┴ nothing
```

# Related

  - [`GerberIQDecayEstimator`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`regenerate_decay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.5, Equation 5.
"""
@concrete struct ExpGerberIQDecay <: GerberIQDecayEstimator
    """
    Delay. A co-movement that is no older than `e` periods keeps its full weight. The source names it ``\\varepsilon``.
    """
    e
    """
    Decay rate. The decay discounts a co-movement at this rate for each period of age after `e`. The source names it ``\\gamma``.
    """
    y
    function ExpGerberIQDecay(e::Option{<:GerberIQEps}, y::Option{<:GerberIQGamma})
        if isa(e, Number)
            assert_nonempty_nonneg_finite_val(e, :e)
        end
        if isa(y, Number)
            assert_nonempty_nonneg_finite_val(y, :y)
        end
        return new{typeof(e), typeof(y)}(e, y)
    end
end
function ExpGerberIQDecay(; e::Option{<:GerberIQEps} = nothing,
                          y::Option{<:GerberIQGamma} = nothing)::ExpGerberIQDecay
    return ExpGerberIQDecay(e, y)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the exponential decay weight of one observation, at the age `T - k`.

# Mathematical definition

```math
\\begin{align}
w_k &= \\exp\\!\\left(-y \\cdot \\max(0,\\, T - k - e)\\right)\\,.
\\end{align}
```

Where:

  - ``w_k``: Decay weight of the observation at index ``k``.
  - ``y``: Decay rate, the field `decay.y`.
  - ``e``: Delay, the field `decay.e`. The decay does not discount an observation of the last ``e`` periods.
  - $(math_dict[:T])
  - ``k``: Index of the observation, with ``1 \\leq k \\leq T``.

# Arguments

  - `decay`: An [`ExpGerberIQDecay`](@ref) whose `e` and `y` are numbers.
  - `T`: Number of observations.
  - `k`: Index of the observation.

# Returns

  - `w::Number`: Decay weight of observation `k`.

# Examples

```jldoctest
julia> ExpGerberIQDecay(; e = 5.0, y = 0.1)(10, 5)
1.0

julia> ExpGerberIQDecay(; e = 0.0, y = 0.1)(10, 5)
0.6065306597126334
```

# Related

  - [`ExpGerberIQDecay`](@ref)
  - [`regenerate_decay`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.5, Equation 5.
"""
function (decay::ExpGerberIQDecay)(T::Number, k::Number)
    return exp(-decay.y * max(0, T - k - decay.e))
end
"""
    regenerate_decay(decay::ExpGerberIQDecay, X::AbstractMatrix) -> ExpGerberIQDecay
    regenerate_decay(decay::GerberIQDecayEstimator, X::AbstractMatrix) -> ExpGerberIQDecay

Returns a decay with a number in each parameter. The function computes from `X` each parameter that is not a number.

The function allocates, and it never writes into its argument. A decay estimator is configuration, and the library does not change it. So a resolved parameter goes into a new [`ExpGerberIQDecay`](@ref). The function allocates nothing for an [`ExpGerberIQDecay`](@ref) whose two fields are already numbers. It then returns the same object. The caller must use the returned value, because the argument keeps the parameters that are not numbers.

# Algorithm

 1. Return `decay` unchanged when it is an [`ExpGerberIQDecay`](@ref) whose `e` and `y` are both a `Number`.
 2. Otherwise resolve the delay `e` with [`gerber_iq_eps`](@ref). For an [`ExpGerberIQDecay`](@ref), the function gives it the field `e`. For each other subtype, it gives it `nothing`.
 3. Resolve the decay rate `y` in the same way with [`gerber_iq_gamma`](@ref).
 4. Return a new [`ExpGerberIQDecay`](@ref) with `e` and `y`.

The fall-through does not know the fields of another subtype. So at step 2 it computes both parameters from `X` alone, and it returns an [`ExpGerberIQDecay`](@ref) in place of the subtype. A subtype that must keep its own form implements this function, as the `# Interfaces` section of [`GerberIQDecayEstimator`](@ref) states.

# Arguments

  - `decay`: The decay estimator to resolve.
      + `::ExpGerberIQDecay`: When both parameters are numbers, the function returns `decay`. Otherwise it returns a new [`ExpGerberIQDecay`](@ref) with the resolved parameters.
      + `::GerberIQDecayEstimator`: The fall-through. It computes `e` and `y` from `X` with [`gerber_iq_eps`](@ref) and [`gerber_iq_gamma`](@ref), and gives `nothing` as their first argument. A subtype of [`GerberIQDecayEstimator`](@ref) that implements no method gets this fall-through.
  - $(arg_dict[:X])

# Returns

  - `decay::ExpGerberIQDecay`: The decay, with a number in each parameter.

# Related

  - [`GerberIQDecayEstimator`](@ref)
  - [`ExpGerberIQDecay`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`gerber_iq_gamma`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.4.
"""
function regenerate_decay(decay::ExpGerberIQDecay{<:Number, <:Number}, ::AbstractMatrix)
    return decay
end
function regenerate_decay(decay::ExpGerberIQDecay{<:Any, <:Any}, X::AbstractMatrix)
    e = gerber_iq_eps(decay.e, X)
    y = gerber_iq_gamma(decay.y, X)
    return ExpGerberIQDecay(; e = e, y = y)
end
function regenerate_decay(decay::GerberIQDecayEstimator, X::AbstractMatrix)
    e = gerber_iq_eps(nothing, X)
    y = gerber_iq_gamma(nothing, X)
    return ExpGerberIQDecay(; e = e, y = y)
end
"""
$(DocStringExtensions.TYPEDEF)

Squeezing template with one boundary and one weight for the whole return plane.

This template is the reduced template of the source. The results of the source use it, and it is the default of [`GerberIQCovariance`](@ref).

# Mathematical definition

```math
\\begin{align}
\\eta_{t,\\,i,\\,j} &= \\begin{cases}
1 & d s_i \\leq \\lvert x_{t,\\,i} \\rvert \\; \\text{and} \\; d s_j \\leq \\lvert x_{t,\\,j} \\rvert \\\\
n & \\lvert x_{t,\\,i} \\rvert < d s_i \\; \\text{and} \\; \\lvert x_{t,\\,j} \\rvert < d s_j \\\\
n^2 & \\text{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\eta_{t,\\,i,\\,j}``: Squeezing weight of the co-movement of assets ``i`` and ``j`` at observation ``t``.
  - $(math_dict[:x_ti_ret])
  - ``s_i``, ``s_j``: Threshold scaling factors of the pair.
  - ``d``: Significance threshold.
  - ``n``: Squeezing weight.

The three cases are the tail, the body and the wing of the source. In the tail both returns are large, and in the body both returns are small. In the wing, one return is large and one return is small. Because ``0 \\leq n \\leq 1``, the weights obey ``n^2 \\leq n \\leq 1``. So a co-movement of two similar magnitudes counts no less than a co-movement of two different magnitudes. That order is the judgement of the source, and a template with one free weight cannot give a different order.

Section 3.3.5 of [gerber2025squeezing](@cite) gives the weights `1`, `η` and `η²` to the tail, the body and the wing, with `0 ≤ η ≤ 1`. Table 2 gives 2 to 3 as the typical range of the boundary. The defaults `d = 2` and `n = 0.5` are the choice of the library in those ranges. The figures of the source show no weight on the branches of the noise zone, where one asset is inside the noise threshold. Section 2 states only that a method can use such data or not. The library gives a co-movement there the weight of the channel that the same rule selects, as the diagram shows.

The body case happens only when ``c < d``. The statistic drops a co-movement with both returns inside the noise threshold before the template. So at ``c = d`` the band of `n` is empty, and only ``1`` and ``n^2`` occur.

The diagram shows the regions of `BasicGerberIQ` for `c = 1` and `d = 3`.

  - A dashed line is the noise threshold `c` on one axis.
  - A co-movement in a dashed band has one return inside the noise zone. [`Gerber0`](@ref) ignores it. [`Gerber1`](@ref) adds it to the neutral sum, and [`Gerber2`](@ref) adds the projection of the other asset to its denominator.
  - The statistic always drops a co-movement in the square around `r0`, where both returns are inside the noise zone.
  - A thin line is the zero of an axis.
  - A heavy line is a boundary of the template.
  - The label of a region is the weight of a co-movement in that region.

```
            4 ┬─────┰───────────┬─────┬─────┬───────────┰─────┐
     ┌────    │  1  ┃    n^2    ╎     │     ╎    n^2    ┃  1  │
  d ─┤      3 ┾━━━━━╋━━━━━━━━━━━┿━━━━━┿━━━━━┿━━━━━━━━━━━╋━━━━━┥
     └────    │     ┃           ╎     │     ╎           ┃     │
            2 ┤ n^2 ┃     n     ╎     │     ╎     n     ┃ n^2 │
              │     ┃           ╎     │     ╎           ┃     │
     ┌────  1 ┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┴╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┤
     │        │     ┃           ╎           ╎           ┃     │
 2c ─┤ r_j  0 ┼─────╂───────────┤    r0     ├───────────╂─────┤
     │        │     ┃           ╎           ╎           ┃     │
     └──── -1 ┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┬╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┤
              │     ┃           ╎     │     ╎           ┃     │
           -2 ┤ n^2 ┃     n     ╎     │     ╎     n     ┃ n^2 │
     ┌────    │     ┃           ╎     │     ╎           ┃     │
  d ─┤     -3 ┾━━━━━╋━━━━━━━━━━━┿━━━━━┿━━━━━┿━━━━━━━━━━━╋━━━━━┥
     └────    │  1  ┃    n^2    ┊     │     ╎    n^2    ┃  1  │
           -4 ┼─────╀─────┬─────┼─────┼─────┼─────┬─────╀─────┤
             -4    -3    -2    -1     0     1     2     3     4
                                     r_i
                 │     │        │           │        │     │
                 └──┬──┘        └─────┬─────┘        └──┬──┘
                    d                2c                 d
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BasicGerberIQ(; d::Number = 2.0, n::Number = 0.5)

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `d`.
  - `0 <= n <= 1`.
  - [`gerber_iq_assert_c_d`](@ref) checks `c <= d` when the template goes into a [`GerberIQCovariance`](@ref), not here.

# Examples

```jldoctest
julia> BasicGerberIQ()
BasicGerberIQ
  d ┼ Float64: 2.0
  n ┴ Float64: 0.5
```

# Related

  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.5, Figure 3.
"""
@concrete struct BasicGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Significance threshold. A return that is `d` scaled units or more from zero is large. `d` has the units of the noise threshold `c` of [`GerberIQCovariance`](@ref), and it must be at least `c`.
    """
    d
    """
    Squeezing weight, in `[0, 1]`. A co-movement of two large returns has the full weight of one. A co-movement of two small returns has the weight `n`, and a co-movement of one large and one small return has the weight `n^2`.
    """
    n
    function BasicGerberIQ(d::Number, n::Number)
        assert_nonempty_nonneg_finite_val(d, :d)
        assert_closed_unit_interval(n, :n)
        return new{typeof(d), typeof(n)}(d, n)
    end
end
function BasicGerberIQ(; d::Number = 2.0, n::Number = 0.5)::BasicGerberIQ
    return BasicGerberIQ(d, n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Checks that `c <= kind.d`, where `c` is the noise threshold and `d` is the significance threshold of a [`BasicGerberIQ`](@ref).

The two parameters are on the same axis, in the same scaled units. `c` is the upper limit of the noise zone, and `d` is the lower limit of the significant zone. So the body of the template is the band `c <= |x| < d`. Section 3.3.2 of [gerber2025squeezing](@cite) defines the body of a quadrant by `c < r* ≤ δ`, which is the same band. When `d` is less than `c`, the band is empty. Each return outside the noise zone is then beyond `d`, and the template never selects the weight `n`.

# Arguments

  - `c`: Noise threshold.
  - `kind`: The [`BasicGerberIQ`](@ref) template.

# Validation

  - `c <= kind.d`, else a `DomainError` that names both values.

# Returns

  - `nothing`. The caller uses the function for its check alone.

# Related

  - [`BasicGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.2.
"""
function gerber_iq_assert_c_d(c::Number, kind::BasicGerberIQ)
    @argcheck(c <= kind.d,
              DomainError(c,
                          "`c` is $c, and `kind.d` is $(kind.d). The centre band is inside the boundary, so `c <= kind.d` must hold."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight of a co-movement from the channel of the [`BasicGerberIQ`](@ref) template that the co-movement falls in.

[`BasicGerberIQ`](@ref) states the closed form. The method does not read the signed returns, because the template is symmetric about both axes and uses the magnitudes alone.

# Algorithm

 1. Scale the significance threshold to each axis, `di = d * sci` and `dj = d * scj`.
 2. Return `one(n)` when both absolute returns reach their scaled threshold.
 3. Return `n` when neither absolute return reaches it.
 4. Return `n^2` otherwise. In this case, exactly one absolute return reaches its threshold.

# Arguments

  - `xi`: Return of asset `i`. The method does not read it.
  - `xj`: Return of asset `j`. The method does not read it.
  - `axi`: Absolute return of asset `i`.
  - `axj`: Absolute return of asset `j`.
  - `sci`: Scaling factor of asset `i`.
  - `scj`: Scaling factor of asset `j`.
  - `kind`: The [`BasicGerberIQ`](@ref) template.

# Returns

  - `res::Number`: Weight of the co-movement.

# Related

  - [`BasicGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.5, Figure 3.
"""
function gerber_iq_weight(::Number, ::Number, axi::Number, axj::Number, sci::Number,
                          scj::Number, kind::BasicGerberIQ)
    (; d, n) = kind
    di = d * sci
    dj = d * scj
    return if di <= axi && dj <= axj
        one(n)
    elseif axi < di && axj < dj
        n
    else
        n^2
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Squeezing template with four boundaries, one for each sign of a return in a concordant and in a discordant co-movement.

This template is the sixteen-channel template of the source. Its boundary vector has four independent components, and it has ten distinct weights. [`BasicGerberIQ`](@ref) has one boundary and one weight, and [`FullGerberIQ`](@ref) has thirty-six channels.

# Mathematical definition

Every boundary is scaled onto its own axis before it is compared with a return.

```math
\\begin{align}
\\delta_{i} &= \\delta\\, s_i\\,.
\\end{align}
```

Where:

  - ``\\delta``: One of the four boundaries `dcp`, `dcn`, `ddp` and `ddn`.
  - ``\\delta_{i}``: That boundary on the axis of asset ``i``.
  - ``s_i``: Threshold scaling factor of asset ``i``.

The four boundaries cut the plane into sixteen channels. The plane is symmetric about the line ``x_i = x_j``, so the sixteen channels have ten distinct weights, `n1` to `n10`. The template uses the signed return, not its magnitude, so it separates a pair by sign and by size. It compares a concordant pair of positive returns with `dcp`, and a concordant pair of negative returns with `dcn`. It compares a discordant pair with `ddp` and `ddn`. A co-movement in no named channel has the weight zero, which happens only when a return is exactly zero.

The boundaries `dcp`, `dcn`, `ddp` and `ddn` are the four components of the boundary vector of Figure 1(b) of [gerber2025squeezing](@cite). The weights `n1` to `n10` have the numbers of its ten weights, channel by channel. The source leaves each value to expert judgement or to an outer optimisation. So the defaults are the library's, and so are the geometric-mean defaults of the mixed weights.

The diagram shows the regions of `PartialGerberIQ` for `c = 1`, `dcp = 2`, `ddn = 2`, `ddp = 3` and `dcn = 3`.

  - A dashed line is the noise threshold `c` on one axis.
  - A co-movement in a dashed band has one return inside the noise zone. [`Gerber0`](@ref) ignores it. [`Gerber1`](@ref) adds it to the neutral sum, and [`Gerber2`](@ref) adds the projection of the other asset to its denominator.
  - The statistic always drops a co-movement in the square around `r0`, where both returns are inside the noise zone.
  - A thin line is the zero of an axis.
  - A heavy line is a boundary of the template.
  - The label of a region is the weight of a co-movement in that region.

```
                         ddn                     dcp
                       ┌──┴──┐                 ┌──┴──┐
                       │     │                 │     │
            4 ┬───────────┰─────┬─────┬─────┬─────┰───────────┐
     ┌────    │    n6     ┃ n9  ╎     │     ╎     ┃           │
ddp ─┤      3 ┾━━━━━━━━━━━╋━━━━━┿━━━━━┥     ╎ n7  ┃    n4     │
     └────    │           ┃     ╎     │     ╎     ┃           │ ────┐
            2 ┤    n10    ┃ n3  ╎     ┝━━━━━┿━━━━━╋━━━━━━━━━━━┥     ├─ dcp
              │           ┃     ╎     │     ╎ n1  ┃    n7     │ ────┘
     ┌────  1 ┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┼╌╌╌╌╌┴╌╌╌╌╌┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┤
     │        │           ┃     ╎           ╎     ┃           │
 2c ─┤ r_j  0 ┼─────┰─────┸─────┤    r0     ├─────┸─────┰─────┤
     │        │     ┃           ╎           ╎           ┃     │
     └──── -1 ┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┬╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┤
              │     ┃           ╎     │     ╎    n3     ┃ n9  │ ────┐
           -2 ┤ n8  ┃    n2     ╎     ┝━━━━━┿━━━━━━━━━━━╋━━━━━┥     ├─ ddn
     ┌────    │     ┃           ╎     │     ╎           ┃     │ ────┘
dcn ─┤     -3 ┾━━━━━╋━━━━━━━━━━━┿━━━━━┥     ╎    n10    ┃ n6  │
     └────    │ n5  ┃    n8     ╎     │     ╎           ┃     │
           -4 ┼─────╀─────┬─────┼─────┼─────┼─────┬─────╀─────┤
             -4    -3    -2    -1     0     1     2     3     4
                                     r_i
                 │     │        │           │        │     │
                 └──┬──┘        └─────┬─────┘        └──┬──┘
                   dcn               2c                ddp
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PartialGerberIQ(; dcp::Number = 2.0, dcn::Number = dcp, ddp::Number = dcp,
                      ddn::Number = dcp, n1::Number = 0.5, n2::Number = n1,
                      n3::Number = sqrt(n1 * n2), n4::Number = 1.0, n5::Number = n4,
                      n6::Number = sqrt(n4 * n5), n7::Number = sqrt(n1 * n4),
                      n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n4 * n2),
                      n10::Number = sqrt(n1 * n5))

Keywords correspond to the struct's fields.

Only the four diagonal weights have a free default. `n1`, `n2`, `n4` and `n5` are the weights of the four magnitude classes. The default of each of the other six weights is the geometric mean of the diagonal weights of the two classes that its channel joins. [`clamp_gerber_iq_n`](@ref) enforces that bound under [`Gerber2`](@ref). So the default template meets the bound for all values of the four diagonal weights. The four boundaries are equal by default, so the case split of that method on `ddp` and `dcp` changes no default.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks each of `dcp`, `dcn`, `ddp` and `ddn`.
  - Each weight `n1` to `n10` must be in `[0, 1]`.
  - [`gerber_iq_assert_c_d`](@ref) checks `c <= dcp`, `c <= dcn`, `c <= ddp` and `c <= ddn` when the template goes into a [`GerberIQCovariance`](@ref), not here.

# Examples

```jldoctest
julia> PartialGerberIQ()
PartialGerberIQ
  dcp ┼ Float64: 2.0
  dcn ┼ Float64: 2.0
  ddp ┼ Float64: 2.0
  ddn ┼ Float64: 2.0
   n1 ┼ Float64: 0.5
   n2 ┼ Float64: 0.5
   n3 ┼ Float64: 0.5
   n4 ┼ Float64: 1.0
   n5 ┼ Float64: 1.0
   n6 ┼ Float64: 1.0
   n7 ┼ Float64: 0.7071067811865476
   n8 ┼ Float64: 0.7071067811865476
   n9 ┼ Float64: 0.7071067811865476
  n10 ┴ Float64: 0.7071067811865476
```

# Related

  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 2, Figure 1(b), and Section 3.3.2.
"""
@concrete struct PartialGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Boundary of a concordant co-movement of two positive returns. A positive return at `dcp` or above is large.
    """
    dcp
    """
    Boundary of a concordant co-movement of two negative returns. A negative return at `-dcn` or below is large.
    """
    dcn
    """
    Boundary of the positive return of a discordant co-movement, for either asset. A positive return at `ddp` or above is large.
    """
    ddp
    """
    Boundary of the negative return of a discordant co-movement, for either asset. A negative return at `-ddn` or below is large.
    """
    ddn
    """
    Weight of a concordant co-movement of two small positive returns.
    """
    n1
    """
    Weight of a concordant co-movement of two small negative returns.
    """
    n2
    """
    Weight of a discordant co-movement of a small positive return with a small negative return.
    """
    n3
    """
    Weight of a concordant co-movement of two large positive returns.
    """
    n4
    """
    Weight of a concordant co-movement of two large negative returns.
    """
    n5
    """
    Weight of a discordant co-movement of a large positive return with a large negative return.
    """
    n6
    """
    Weight of a concordant positive co-movement of a small return with a large one.
    """
    n7
    """
    Weight of a concordant negative co-movement of a small return with a large one.
    """
    n8
    """
    Weight of a discordant co-movement whose positive return is large and whose negative return is small.
    """
    n9
    """
    Weight of a discordant co-movement whose positive return is small and whose negative return is large.
    """
    n10
    function PartialGerberIQ(dcp::Number, dcn::Number, ddp::Number, ddn::Number, n1::Number,
                             n2::Number, n3::Number, n4::Number, n5::Number, n6::Number,
                             n7::Number, n8::Number, n9::Number, n10::Number)
        assert_nonempty_nonneg_finite_val(dcp, :dcp)
        assert_nonempty_nonneg_finite_val(dcn, :dcn)
        assert_nonempty_nonneg_finite_val(ddp, :ddp)
        assert_nonempty_nonneg_finite_val(ddn, :ddn)
        assert_closed_unit_interval(n1, :n1)
        assert_closed_unit_interval(n2, :n2)
        assert_closed_unit_interval(n3, :n3)
        assert_closed_unit_interval(n4, :n4)
        assert_closed_unit_interval(n5, :n5)
        assert_closed_unit_interval(n6, :n6)
        assert_closed_unit_interval(n7, :n7)
        assert_closed_unit_interval(n8, :n8)
        assert_closed_unit_interval(n9, :n9)
        assert_closed_unit_interval(n10, :n10)
        return new{typeof(dcp), typeof(dcn), typeof(ddp), typeof(ddn), typeof(n1),
                   typeof(n2), typeof(n3), typeof(n4), typeof(n5), typeof(n6), typeof(n7),
                   typeof(n8), typeof(n9), typeof(n10)}(dcp, dcn, ddp, ddn, n1, n2, n3, n4,
                                                        n5, n6, n7, n8, n9, n10)
    end
end
function PartialGerberIQ(; dcp::Number = 2.0, dcn::Number = dcp, ddp::Number = dcp,
                         ddn::Number = dcp, n1::Number = 0.5, n2::Number = n1,
                         n3::Number = sqrt(n1 * n2), n4::Number = 1.0, n5::Number = n4,
                         n6::Number = sqrt(n4 * n5), n7::Number = sqrt(n1 * n4),
                         n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n4 * n2),
                         n10::Number = sqrt(n1 * n5))::PartialGerberIQ
    return PartialGerberIQ(dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Lowers the mixed-magnitude weights of a [`PartialGerberIQ`](@ref) template, so that the [`Gerber2`](@ref) statistic stays inside `[-1, 1]`. It does not make the matrix positive definite. The `pdm` field of the estimator does that.

Under [`Gerber2`](@ref), the entry of a pair is the net sum `pos - neg`, divided by the geometric mean of the two diagonal projections of the pair. A projection compares an asset with itself. So the projection is always concordant, and it gives a same-magnitude weight. The Internet Appendix of [gerber2025squeezing](@cite) states a condition for a bound of one. Each weight that joins two distinct magnitude classes must be at most the geometric mean of the two same-magnitude weights of those classes. The source proves that the condition is necessary and sufficient for its own alternative statistic, in Equations A1 to A9.

That statistic sums its diagonal projections over the observations on which both assets cross. The [`Gerber2`](@ref) of the library sums them over the observations on which the one asset crosses, which is a larger set. So for the library, the Cauchy-Schwarz inequality makes the condition sufficient. The condition stays necessary for the template as a whole, because a sample in which the two assets always cross together gives the denominator of the source. Six of the ten weights join two classes, and this method lowers each of them to its bound. With `n1 = n4 = 0.2`, an unclamped `n7 = 1.0` gives `5.0`, and the clamp makes it `1.0`.

The four discordant weights need a case split. The template compares a concordant pair with `dcp` and `dcn`, and a discordant pair with `ddp` and `ddn`. So a return beyond a discordant boundary is not always beyond the concordant boundary that sets its diagonal class. When the two boundaries differ, the method uses the smaller of the two candidate weights. That bound holds for either class.

!!! note

    The clamp is a necessary and sufficient condition on the template, and the statistic needs no other condition. `sc` needs no restriction, because [`Gerber2`](@ref) computes its denominator in the units of the pair. The shipped defaults are exactly on the bound, so the clamp does not change them.

# Mathematical definition

Let ``h^{+}`` and ``l^{+}`` be the diagonal weights of the two positive classes that a discordant channel can join. Let ``h^{-}`` and ``l^{-}`` be the diagonal weights of the two negative classes.

```math
\\begin{align}
h^{+} &= \\begin{cases} n_{4} & \\delta_{d}^{+} \\geq \\delta_{c}^{+} \\\\ \\min(n_{1}, n_{4}) & \\text{otherwise} \\end{cases}\\,,
&l^{+} &= \\begin{cases} n_{1} & \\delta_{d}^{+} \\leq \\delta_{c}^{+} \\\\ \\min(n_{1}, n_{4}) & \\text{otherwise} \\end{cases}\\,,
\\end{align}
```

The rule for ``h^{-}`` and ``l^{-}`` is the same, with ``n_{5}``, ``n_{2}``, ``\\delta_{d}^{-}`` and ``\\delta_{c}^{-}``. The six bounds are:

```math
\\begin{align}
n_{7} &\\leftarrow \\min\\left(n_{7},\\, \\sqrt{n_{1} n_{4}}\\right)\\,, &
n_{8} &\\leftarrow \\min\\left(n_{8},\\, \\sqrt{n_{2} n_{5}}\\right)\\,, \\\\
n_{6} &\\leftarrow \\min\\left(n_{6},\\, \\sqrt{h^{+} h^{-}}\\right)\\,, &
n_{9} &\\leftarrow \\min\\left(n_{9},\\, \\sqrt{h^{+} l^{-}}\\right)\\,, \\\\
n_{10} &\\leftarrow \\min\\left(n_{10},\\, \\sqrt{l^{+} h^{-}}\\right)\\,, &
n_{3} &\\leftarrow \\min\\left(n_{3},\\, \\sqrt{l^{+} l^{-}}\\right)\\,.
\\end{align}
```

Where:

  - ``n_{1}``, ``n_{4}``: Small and large positive concordant weights.
  - ``n_{2}``, ``n_{5}``: Small and large negative concordant weights.
  - ``\\delta_{c}^{\\pm}``, ``\\delta_{d}^{\\pm}``: The concordant boundaries `dcp` and `dcn`, and the discordant boundaries `ddp` and `ddn`.

# Algorithm

 1. Resolve the four discordant diagonal weights `hip`, `lop`, `hin` and `lon` from the two pairs of boundaries.
 2. Replace `n7` and `n8`, the weights of the two mixed concordant channels, with their bounds when they are above them. Each bound is the geometric mean of the two weights that the channel joins.
 3. Replace `n3`, `n6`, `n9` and `n10`, the weights of the four discordant channels, in the same way. Each bound uses the resolved weights that the channel joins.
 4. Return a new [`PartialGerberIQ`](@ref) with the six lowered weights and with each other field unchanged.

# Arguments

  - `alg`: The [`PartialGerberIQ`](@ref) template.
  - `::Gerber2`: The [`Gerber2`](@ref) marker.

# Returns

  - `kind::PartialGerberIQ`: A new template. The method allocates and never writes into `alg`.

# Related

  - [`PartialGerberIQ`](@ref)
  - [`Gerber2`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Internet Appendix, Equations A1 to A9.
"""
function clamp_gerber_iq_n(alg::PartialGerberIQ, ::Gerber2)
    (; dcp, dcn, ddp, ddn, n1, n2, n4, n5) = alg
    hip = ddp >= dcp ? n4 : min(n1, n4)
    lop = ddp <= dcp ? n1 : min(n1, n4)
    hin = ddn >= dcn ? n5 : min(n2, n5)
    lon = ddn <= dcn ? n2 : min(n2, n5)
    n3 = min(alg.n3, sqrt(lop * lon))
    n6 = min(alg.n6, sqrt(hip * hin))
    n7 = min(alg.n7, sqrt(n1 * n4))
    n8 = min(alg.n8, sqrt(n2 * n5))
    n9 = min(alg.n9, sqrt(hip * lon))
    n10 = min(alg.n10, sqrt(lop * hin))
    return PartialGerberIQ(; dcp = dcp, dcn = dcn, ddp = ddp, ddn = ddn, n1 = n1, n2 = n2,
                           n3 = n3, n4 = n4, n5 = n5, n6 = n6, n7 = n7, n8 = n8, n9 = n9,
                           n10 = n10)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight of a co-movement from the channel of the [`PartialGerberIQ`](@ref) template that the co-movement falls in.

[`PartialGerberIQ`](@ref) states the channels. The method does not read the absolute returns, because this template uses the sign of each return as well as its size.

# Algorithm

 1. Scale each of the four boundaries to each axis, to get the eight thresholds `dcpi`, `dcni`, `ddpi`, `ddni` and the four thresholds of asset `j`.
 2. Test the positive concordant quadrant from the largest channel. The result is `n4` when both returns reach `dcp`, and `n7` when one return reaches it and the other return is positive. The result is `n1` when both returns are positive and below `dcp`.
 3. Test the negative concordant quadrant in the same way against `dcn`, for `n5`, `n8` and `n2`.
 4. Test the two discordant quadrants against `ddp` and `ddn`. The result is `n6` when both returns are beyond their boundaries, `n9` or `n10` when one return is beyond its boundary, and `n3` when neither return is.
 5. Return `zero(xi)` when no channel matches. This happens only when a return is exactly zero.

The tests go from the largest channel inwards. The first match gives the result, so the method counts no co-movement twice.

# Arguments

  - `xi`: Return of asset `i`.
  - `xj`: Return of asset `j`.
  - `axi`: Absolute return of asset `i`. The method does not read it.
  - `axj`: Absolute return of asset `j`. The method does not read it.
  - `sci`: Scaling factor of asset `i`.
  - `scj`: Scaling factor of asset `j`.
  - `kind`: The [`PartialGerberIQ`](@ref) template.

# Returns

  - `res::Number`: Weight of the co-movement.

# Related

  - [`PartialGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 2, Figure 1(b).
"""
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sci::Number,
                          scj::Number, kind::PartialGerberIQ)
    (; dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10) = kind
    dcpi = dcp * sci
    dcni = dcn * sci
    ddpi = ddp * sci
    ddni = ddn * sci
    dcpj = dcp * scj
    dcnj = dcn * scj
    ddpj = ddp * scj
    ddnj = ddn * scj
    zro = zero(xi)
    return if dcpi <= xi && dcpj <= xj
        n4
    elseif dcpi <= xi && zro < xj < dcpj || dcpj <= xj && zro < xi < dcpi
        n7
    elseif zro < xi < dcpi && zro < xj < dcpj
        n1
    elseif xi <= -dcni && xj <= -dcnj
        n5
    elseif xi <= -dcni && -dcnj < xj < zro || xj <= -dcnj && -dcni < xi < zro
        n8
    elseif -dcni < xi < zro && -dcnj < xj < zro
        n2
    elseif ddpi <= xi && xj <= -ddnj || ddpj <= xj && xi <= -ddni
        n6
    elseif ddpi <= xi && -ddnj < xj < zro || ddpj <= xj && -ddni < xi < zro
        n9
    elseif zro < xi < ddpi && xj <= -ddnj || zro < xj < ddpj && xi <= -ddni
        n10
    elseif zro < xi < ddpi && -ddnj < xj < zro || zro < xj < ddpj && -ddni < xi < zro
        n3
    else
        zro
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Squeezing template with three magnitude classes for each sign of a return.

This template is the full template of the source, and the most general template of the family. Two boundaries on each side of zero put each return in a small, a moderate or a large class. So each axis has six bands, and the plane has thirty-six channels. The plane is symmetric about the line ``x_i = x_j``, so the thirty-six channels have twenty-one distinct weights. The source states this count, and this type has twenty-one weight fields.

# Mathematical definition

Every boundary is scaled onto its own axis before it is compared with a return.

```math
\\begin{align}
\\delta_{i} &= \\delta\\, s_i\\,.
\\end{align}
```

Where:

  - ``\\delta``: One of the four boundaries `dp1`, `dp2`, `dn1` and `dn2`.
  - ``\\delta_{i}``: That boundary on the axis of asset ``i``.
  - ``s_i``: Threshold scaling factor of asset ``i``.

A return of asset ``i`` is large positive at ``dp1_i`` or above, moderate positive in ``[dp2_i, dp1_i)`` and small positive in ``(0, dp2_i)``. The three negative classes are the mirror about zero, with ``dn2_i`` and ``dn1_i``. The channel of a co-movement is the pair of classes of its two returns, and its weight is the field of that pair. A co-movement in no named channel has the weight zero, which happens only when a return is exactly zero.

Figure 2 of [gerber2025squeezing](@cite) shows the same thirty-six channels. The weights `n1` to `n21` have the numbers of its twenty-one weights, channel by channel. The four boundaries of the figure are these fields:

  - ``\\delta_{C^{+}}``, the inner positive boundary, is `dp2`.
  - ``\\delta_{D^{+}}``, the outer positive boundary, is `dp1`.
  - ``\\delta_{D^{-}}``, the inner negative boundary, is `dn2`.
  - ``\\delta_{C^{-}}``, the outer negative boundary, is `dn1`.

The source leaves each value to expert judgement or to an outer optimisation. So the defaults are the library's, and so are the geometric-mean defaults of the mixed weights.

The diagram shows the regions of `FullGerberIQ` for `c = 1`, `dp2 = 2`, `dn2 = 2`, `dp1 = 3` and `dn1 = 3`. The constructor accepts the two boundaries of each sign in either order. It swaps them when necessary, so that `dp1 >= dp2` and `dn1 >= dn2`.

  - A dashed line is the noise threshold `c` on one axis.
  - A co-movement in a dashed band has one return inside the noise zone. [`Gerber0`](@ref) ignores it. [`Gerber1`](@ref) adds it to the neutral sum, and [`Gerber2`](@ref) adds the projection of the other asset to its denominator.
  - The statistic always drops a co-movement in the square around `r0`, where both returns are inside the noise zone.
  - A thin line is the zero of an axis.
  - A heavy line is a boundary of the template.
  - The label of a region is the weight of a co-movement in that region.

```
                         dn2                     dp2
                       ┌──┴──┐                 ┌──┴──┐
                       │     │                 │     │
            4 ┬─────┰─────┰─────┬─────┬─────┬─────┰─────┰─────┐
     ┌────    │ n13 ┃ n19 ┃ n18 ╎     │     ╎ n15 ┃ n14 ┃ n11 │
dp1 ─┤      3 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥
     └────    │ n20 ┃ n6  ┃ n9  ╎     │     ╎ n7  ┃ n4  ┃ n14 │ ────┐
            2 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥     ├─ dp2
              │ n21 ┃ n10 ┃ n3  ╎     │     ╎ n1  ┃ n7  ┃ n15 │ ────┘
     ┌────  1 ┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┼╌╌╌╌╌┴╌╌╌╌╌┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┤
     │        │     ┃     ┃     ╎           ╎     ┃     ┃     │
 2c ─┤ r_j  0 ┼─────╂─────╂─────┤    r0     ├─────╂─────╂─────┤
     │        │     ┃     ┃     ╎           ╎     ┃     ┃     │
     └──── -1 ┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┼╌╌╌╌╌┬╌╌╌╌╌┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┤
              │ n16 ┃ n8  ┃ n2  ╎     │     ╎ n3  ┃ n9  ┃ n18 │ ────┐
           -2 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥     ├─ dn2
     ┌────    │ n17 ┃ n5  ┃ n8  ╎     │     ╎ n10 ┃ n6  ┃ n19 │ ────┘
dn1 ─┤     -3 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥
     └────    │ n12 ┃ n17 ┃ n16 ╎     │     ╎ n21 ┃ n20 ┃ n13 │
           -4 ┼─────╀─────╀─────┼─────┼─────┼─────╀─────╀─────┤
             -4    -3    -2    -1     0     1     2     3     4
                                     r_i
                 │     │        │           │        │     │
                 └──┬──┘        └─────┬─────┘        └──┬──┘
                   dn1               2c                dp1
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FullGerberIQ(; dp1::Number = 2.0, dp2::Number = dp1, dn1::Number = dp1,
                   dn2::Number = dp1, n1::Number = 0.5, n2::Number = n1,
                   n3::Number = sqrt(n1 * n2), n4::Number = 0.75, n5::Number = n4,
                   n6::Number = sqrt(n4 * n5), n7::Number = sqrt(n1 * n4),
                   n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n4 * n2),
                   n10::Number = sqrt(n1 * n5), n11::Number = 1.0, n12::Number = n11,
                   n13::Number = sqrt(n11 * n12), n14::Number = sqrt(n4 * n11),
                   n15::Number = sqrt(n1 * n11), n16::Number = sqrt(n2 * n12),
                   n17::Number = sqrt(n5 * n12), n18::Number = sqrt(n2 * n11),
                   n19::Number = sqrt(n5 * n11), n20::Number = sqrt(n4 * n12),
                   n21::Number = sqrt(n1 * n12))

Keywords correspond to the struct's fields.

Only the six diagonal weights have a free default. `n1`, `n2`, `n4`, `n5`, `n11` and `n12` are the weights of the six magnitude classes. The default of each of the other fifteen weights is the geometric mean of the diagonal weights of the two classes that its channel joins. [`clamp_gerber_iq_n`](@ref) enforces that bound under [`Gerber2`](@ref). So the default template meets the bound for all values of the six diagonal weights, and the clamp never changes a default. A default from two mixed weights can let a sample of two assets give a correlation above `1`.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks each of `dp1`, `dp2`, `dn1` and `dn2`.
  - Each weight `n1` to `n21` must be in `[0, 1]`.
  - [`gerber_iq_assert_c_d`](@ref) checks `c <= dp1`, `c <= dp2`, `c <= dn1` and `c <= dn2` when the template goes into a [`GerberIQCovariance`](@ref), not here.

The swap of the boundaries raises no error. The constructor orders `dp1` and `dp2` with `extrema`, and it orders `dn1` and `dn2` in the same way. So a caller who gives them in the other order gets a valid template.

# Examples

```jldoctest
julia> FullGerberIQ()
FullGerberIQ
  dp1 ┼ Float64: 2.0
  dp2 ┼ Float64: 2.0
  dn1 ┼ Float64: 2.0
  dn2 ┼ Float64: 2.0
   n1 ┼ Float64: 0.5
   n2 ┼ Float64: 0.5
   n3 ┼ Float64: 0.5
   n4 ┼ Float64: 0.75
   n5 ┼ Float64: 0.75
   n6 ┼ Float64: 0.75
   n7 ┼ Float64: 0.6123724356957945
   n8 ┼ Float64: 0.6123724356957945
   n9 ┼ Float64: 0.6123724356957945
  n10 ┼ Float64: 0.6123724356957945
  n11 ┼ Float64: 1.0
  n12 ┼ Float64: 1.0
  n13 ┼ Float64: 1.0
  n14 ┼ Float64: 0.8660254037844386
  n15 ┼ Float64: 0.7071067811865476
  n16 ┼ Float64: 0.7071067811865476
  n17 ┼ Float64: 0.8660254037844386
  n18 ┼ Float64: 0.7071067811865476
  n19 ┼ Float64: 0.8660254037844386
  n20 ┼ Float64: 0.8660254037844386
  n21 ┴ Float64: 0.7071067811865476
```

# Related

  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.4, Figure 2.
"""
@concrete struct FullGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Outer positive boundary. A positive return at `dp1` or above is large. The constructor swaps `dp1` and `dp2` when necessary, so `dp1 >= dp2` always holds.
    """
    dp1
    """
    Inner positive boundary. A positive return in `[dp2, dp1)` is moderate, and a positive return below `dp2` is small.
    """
    dp2
    """
    Outer negative boundary. A negative return at `-dn1` or below is large. The constructor swaps `dn1` and `dn2` when necessary, so `dn1 >= dn2` always holds.
    """
    dn1
    """
    Inner negative boundary. A negative return in `(-dn1, -dn2]` is moderate, and a negative return above `-dn2` is small.
    """
    dn2
    """
    Weight of a concordant co-movement of two small positive returns.
    """
    n1
    """
    Weight of a concordant co-movement of two small negative returns.
    """
    n2
    """
    Weight of a discordant co-movement of a small positive return with a small negative return.
    """
    n3
    """
    Weight of a concordant co-movement of two moderate positive returns.
    """
    n4
    """
    Weight of a concordant co-movement of two moderate negative returns.
    """
    n5
    """
    Weight of a discordant co-movement of a moderate positive return with a moderate negative return.
    """
    n6
    """
    Weight of a concordant positive co-movement of a small return with a moderate one.
    """
    n7
    """
    Weight of a concordant negative co-movement of a small return with a moderate one.
    """
    n8
    """
    Weight of a discordant co-movement whose positive return is moderate and whose negative return is small.
    """
    n9
    """
    Weight of a discordant co-movement whose positive return is small and whose negative return is moderate.
    """
    n10
    """
    Weight of a concordant co-movement of two large positive returns.
    """
    n11
    """
    Weight of a concordant co-movement of two large negative returns.
    """
    n12
    """
    Weight of a discordant co-movement of a large positive return with a large negative return.
    """
    n13
    """
    Weight of a concordant positive co-movement of a moderate return with a large one.
    """
    n14
    """
    Weight of a concordant positive co-movement of a small return with a large one.
    """
    n15
    """
    Weight of a concordant negative co-movement of a small return with a large one.
    """
    n16
    """
    Weight of a concordant negative co-movement of a moderate return with a large one.
    """
    n17
    """
    Weight of a discordant co-movement whose positive return is large and whose negative return is small.
    """
    n18
    """
    Weight of a discordant co-movement whose positive return is large and whose negative return is moderate.
    """
    n19
    """
    Weight of a discordant co-movement whose positive return is moderate and whose negative return is large.
    """
    n20
    """
    Weight of a discordant co-movement whose positive return is small and whose negative return is large.
    """
    n21
    function FullGerberIQ(dp1::Number, dp2::Number, dn1::Number, dn2::Number, n1::Number,
                          n2::Number, n3::Number, n4::Number, n5::Number, n6::Number,
                          n7::Number, n8::Number, n9::Number, n10::Number, n11::Number,
                          n12::Number, n13::Number, n14::Number, n15::Number, n16::Number,
                          n17::Number, n18::Number, n19::Number, n20::Number, n21::Number)
        assert_nonempty_nonneg_finite_val(dp1, :dp1)
        assert_nonempty_nonneg_finite_val(dp2, :dp2)
        assert_nonempty_nonneg_finite_val(dn1, :dn1)
        assert_nonempty_nonneg_finite_val(dn2, :dn2)
        dp2, dp1 = extrema((dp1, dp2))
        dn2, dn1 = extrema((dn1, dn2))
        assert_closed_unit_interval(n1, :n1)
        assert_closed_unit_interval(n2, :n2)
        assert_closed_unit_interval(n3, :n3)
        assert_closed_unit_interval(n4, :n4)
        assert_closed_unit_interval(n5, :n5)
        assert_closed_unit_interval(n6, :n6)
        assert_closed_unit_interval(n7, :n7)
        assert_closed_unit_interval(n8, :n8)
        assert_closed_unit_interval(n9, :n9)
        assert_closed_unit_interval(n10, :n10)
        assert_closed_unit_interval(n11, :n11)
        assert_closed_unit_interval(n12, :n12)
        assert_closed_unit_interval(n13, :n13)
        assert_closed_unit_interval(n14, :n14)
        assert_closed_unit_interval(n15, :n15)
        assert_closed_unit_interval(n16, :n16)
        assert_closed_unit_interval(n17, :n17)
        assert_closed_unit_interval(n18, :n18)
        assert_closed_unit_interval(n19, :n19)
        assert_closed_unit_interval(n20, :n20)
        assert_closed_unit_interval(n21, :n21)
        return new{typeof(dp1), typeof(dp2), typeof(dn1), typeof(dn2), typeof(n1),
                   typeof(n2), typeof(n3), typeof(n4), typeof(n5), typeof(n6), typeof(n7),
                   typeof(n8), typeof(n9), typeof(n10), typeof(n11), typeof(n12),
                   typeof(n13), typeof(n14), typeof(n15), typeof(n16), typeof(n17),
                   typeof(n18), typeof(n19), typeof(n20), typeof(n21)}(dp1, dp2, dn1, dn2,
                                                                       n1, n2, n3, n4, n5,
                                                                       n6, n7, n8, n9, n10,
                                                                       n11, n12, n13, n14,
                                                                       n15, n16, n17, n18,
                                                                       n19, n20, n21)
    end
end
function FullGerberIQ(; dp1::Number = 2.0, dp2::Number = dp1, dn1::Number = dp1,
                      dn2::Number = dp1, n1::Number = 0.5, n2::Number = n1,
                      n3::Number = sqrt(n1 * n2), n4::Number = 0.75, n5::Number = n4,
                      n6::Number = sqrt(n4 * n5), n7::Number = sqrt(n1 * n4),
                      n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n4 * n2),
                      n10::Number = sqrt(n1 * n5), n11::Number = 1.0, n12::Number = n11,
                      n13::Number = sqrt(n11 * n12), n14::Number = sqrt(n4 * n11),
                      n15::Number = sqrt(n1 * n11), n16::Number = sqrt(n2 * n12),
                      n17::Number = sqrt(n5 * n12), n18::Number = sqrt(n2 * n11),
                      n19::Number = sqrt(n5 * n11), n20::Number = sqrt(n4 * n12),
                      n21::Number = sqrt(n1 * n12))
    return FullGerberIQ(dp1, dp2, dn1, dn2, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11,
                        n12, n13, n14, n15, n16, n17, n18, n19, n20, n21)
end
"""
    gerber_iq_assert_c_d(c::Number, kind::Union{<:PartialGerberIQ, <:FullGerberIQ}) -> Nothing

Checks that the noise threshold `c` is at most each boundary of a [`PartialGerberIQ`](@ref) or a [`FullGerberIQ`](@ref).

`c` and each boundary are on the same axis, in the same scaled units. So `c` must be at most the innermost boundary. When a boundary is below `c`, the noise zone covers the band inside that boundary. The function checks each boundary in turn. The error names the boundary that fails, so the caller knows which of the four boundaries is wrong.

# Arguments

  - `c`: Noise threshold.
  - `kind`: The [`PartialGerberIQ`](@ref) or [`FullGerberIQ`](@ref) template.

# Validation

  - `c <= dcp`, `c <= dcn`, `c <= ddp` and `c <= ddn` for a [`PartialGerberIQ`](@ref), else a `DomainError` that names the boundary that fails.
  - `c <= dp1`, `c <= dp2`, `c <= dn1` and `c <= dn2` for a [`FullGerberIQ`](@ref), else a `DomainError` that names the boundary that fails.

# Returns

  - `nothing`. The caller uses the function for its check alone.

# Related

  - [`PartialGerberIQ`](@ref)
  - [`FullGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.2.
"""
function gerber_iq_assert_c_d(c::Number, kind::PartialGerberIQ)
    @argcheck(c <= kind.dcp,
              DomainError(c,
                          "`c` is $c, and `kind.dcp` is $(kind.dcp). The centre band is inside the boundary, so `c <= kind.dcp` must hold."))
    @argcheck(c <= kind.dcn,
              DomainError(c,
                          "`c` is $c, and `kind.dcn` is $(kind.dcn). The centre band is inside the boundary, so `c <= kind.dcn` must hold."))
    @argcheck(c <= kind.ddp,
              DomainError(c,
                          "`c` is $c, and `kind.ddp` is $(kind.ddp). The centre band is inside the boundary, so `c <= kind.ddp` must hold."))
    @argcheck(c <= kind.ddn,
              DomainError(c,
                          "`c` is $c, and `kind.ddn` is $(kind.ddn). The centre band is inside the boundary, so `c <= kind.ddn` must hold."))
    return nothing
end
function gerber_iq_assert_c_d(c::Number, kind::FullGerberIQ)
    @argcheck(c <= kind.dp1,
              DomainError(c,
                          "`c` is $c, and `kind.dp1` is $(kind.dp1). The centre band is inside the boundary, so `c <= kind.dp1` must hold."))
    @argcheck(c <= kind.dp2,
              DomainError(c,
                          "`c` is $c, and `kind.dp2` is $(kind.dp2). The centre band is inside the boundary, so `c <= kind.dp2` must hold."))
    @argcheck(c <= kind.dn1,
              DomainError(c,
                          "`c` is $c, and `kind.dn1` is $(kind.dn1). The centre band is inside the boundary, so `c <= kind.dn1` must hold."))
    @argcheck(c <= kind.dn2,
              DomainError(c,
                          "`c` is $c, and `kind.dn2` is $(kind.dn2). The centre band is inside the boundary, so `c <= kind.dn2` must hold."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Lowers the mixed-magnitude weights of a [`FullGerberIQ`](@ref) template, so that the [`Gerber2`](@ref) statistic stays inside `[-1, 1]`. It does not make the matrix positive definite. The `pdm` field of the estimator does that.

Under [`Gerber2`](@ref), the entry of a pair is the net sum `pos - neg`, divided by the geometric mean of the two diagonal projections of the pair. A projection compares an asset with itself. So both returns fall in the same magnitude class, and the co-movement is always concordant. Thus exactly six of the twenty-one weights are on the diagonal, one for each class. They are `n11`, `n4`, `n1`, `n2`, `n5` and `n12`. Each of the other fifteen weights joins two distinct classes, one weight for each unordered pair of classes, and `C(6, 2) = 15`. The discordant channels are among the fifteen.

The Internet Appendix of [gerber2025squeezing](@cite) states a condition for a bound of one. Each of the fifteen weights must be at most the geometric mean of the diagonal weights of the two classes that it joins. The source proves that the condition is necessary and sufficient for its own alternative statistic, in Equations A1 to A9. The [`Gerber2`](@ref) of the library sums its projections over a larger set of observations. So for the library, the Cauchy-Schwarz inequality makes the condition sufficient, and the condition stays necessary for the template as a whole.

This method lowers all fifteen weights to that bound. A discordant channel follows the same rule, because both classes that it joins have a diagonal weight. A [`FullGerberIQ`](@ref) with `n1 = n11 = 0.1` and `n15 = 1.0` gives `10.0` with no clamp, and `1.0` with the clamp.

!!! note

    The clamp is a necessary and sufficient condition on the template, and the statistic needs no other condition. `sc` needs no restriction, because [`Gerber2`](@ref) computes its denominator in the units of the pair. Each shipped default meets the bound, so the clamp does not change a default.

# Mathematical definition

Let ``n_{k}`` be one of the fifteen mixed weights. Let ``a`` and ``b`` be the two classes that its channel joins, with the diagonal weights ``n_{a}`` and ``n_{b}``. Then:

```math
\\begin{align}
n_{k} &\\leftarrow \\min\\left(n_{k},\\, \\sqrt{n_{a} n_{b}}\\right)\\,.
\\end{align}
```

The fifteen channels, and the pair of classes that each channel joins, are:

```math
\\begin{align}
n_{13} &: (n_{11}, n_{12})\\,, & n_{14} &: (n_{4}, n_{11})\\,, & n_{15} &: (n_{1}, n_{11})\\,, \\\\
n_{16} &: (n_{2}, n_{12})\\,, & n_{17} &: (n_{5}, n_{12})\\,, & n_{18} &: (n_{2}, n_{11})\\,, \\\\
n_{19} &: (n_{5}, n_{11})\\,, & n_{20} &: (n_{4}, n_{12})\\,, & n_{21} &: (n_{1}, n_{12})\\,, \\\\
n_{3} &: (n_{1}, n_{2})\\,, & n_{6} &: (n_{4}, n_{5})\\,, & n_{7} &: (n_{1}, n_{4})\\,, \\\\
n_{8} &: (n_{2}, n_{5})\\,, & n_{9} &: (n_{4}, n_{2})\\,, & n_{10} &: (n_{1}, n_{5})\\,.
\\end{align}
```

Where:

  - ``n_{1}``, ``n_{4}``, ``n_{11}``: Small, moderate and large positive concordant weights.
  - ``n_{2}``, ``n_{5}``, ``n_{12}``: Small, moderate and large negative concordant weights.

# Algorithm

 1. Read the six diagonal weights `n1`, `n2`, `n4`, `n5`, `n11` and `n12` from `alg`.
 2. Replace each of the fifteen mixed weights with its bound when it is above it. The table above gives the two classes of each bound.
 3. Return a new [`FullGerberIQ`](@ref) with the fifteen lowered weights, and with the six diagonal weights and the four boundaries unchanged.

# Arguments

  - `alg`: The [`FullGerberIQ`](@ref) template.
  - `::Gerber2`: The [`Gerber2`](@ref) marker.

# Returns

  - `kind::FullGerberIQ`: A new template. The method allocates and never writes into `alg`.

# Related

  - [`FullGerberIQ`](@ref)
  - [`Gerber2`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Internet Appendix, Equations A1 to A9.
"""
function clamp_gerber_iq_n(alg::FullGerberIQ, ::Gerber2)
    (; n1, n2, n4, n5, n11, n12) = alg
    n3 = min(alg.n3, sqrt(n1 * n2))
    n6 = min(alg.n6, sqrt(n4 * n5))
    n7 = min(alg.n7, sqrt(n1 * n4))
    n8 = min(alg.n8, sqrt(n2 * n5))
    n9 = min(alg.n9, sqrt(n4 * n2))
    n10 = min(alg.n10, sqrt(n1 * n5))
    n13 = min(alg.n13, sqrt(n11 * n12))
    n14 = min(alg.n14, sqrt(n4 * n11))
    n15 = min(alg.n15, sqrt(n1 * n11))
    n16 = min(alg.n16, sqrt(n2 * n12))
    n17 = min(alg.n17, sqrt(n5 * n12))
    n18 = min(alg.n18, sqrt(n2 * n11))
    n19 = min(alg.n19, sqrt(n5 * n11))
    n20 = min(alg.n20, sqrt(n4 * n12))
    n21 = min(alg.n21, sqrt(n1 * n12))
    return FullGerberIQ(; dp1 = alg.dp1, dp2 = alg.dp2, dn1 = alg.dn1, dn2 = alg.dn2,
                        n1 = n1, n2 = n2, n3 = n3, n4 = n4, n5 = n5, n6 = n6, n7 = n7,
                        n8 = n8, n9 = n9, n10 = n10, n11 = n11, n12 = n12, n13 = n13,
                        n14 = n14, n15 = n15, n16 = n16, n17 = n17, n18 = n18, n19 = n19,
                        n20 = n20, n21 = n21)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight of a co-movement from the channel of the [`FullGerberIQ`](@ref) template that the co-movement falls in.

[`FullGerberIQ`](@ref) states the channels. The method does not read the absolute returns, because this template uses the sign of each return as well as its size.

# Algorithm

 1. Scale each of the four boundaries to each axis, to get the eight thresholds `dp1i`, `dp2i`, `dn1i`, `dn2i` and the four thresholds of asset `j`.
 2. Test the six channels in which the return of asset `i` is large positive, for `n11`, `n14`, `n15`, `n18`, `n19` and `n13`. The order of the return of asset `j` is large positive, moderate positive, small positive, small negative, moderate negative and large negative.
 3. Test the five other channels in which the return of asset `i` is moderate positive, for `n4`, `n7`, `n9`, `n6` and `n20`.
 4. Test the four other channels in which the return of asset `i` is small positive, for `n1`, `n3`, `n10` and `n21`.
 5. Test the three other channels in which the return of asset `i` is small negative, for `n2`, `n8` and `n16`.
 6. Test the two other channels in which the return of asset `i` is moderate negative, for `n5` and `n17`.
 7. Return `n12` when both returns are large negative.
 8. Return `zero(xi)` when no channel matches. This happens only when a return is exactly zero.

Each test contains both orders of the pair, so the result is symmetric in the two returns. The tests go from the largest class inwards. The first match gives the result, so the method counts no co-movement twice. The method can return each of the twenty-one weights.

# Arguments

  - `xi`: Return of asset `i`.
  - `xj`: Return of asset `j`.
  - `axi`: Absolute return of asset `i`. The method does not read it.
  - `axj`: Absolute return of asset `j`. The method does not read it.
  - `sci`: Scaling factor of asset `i`.
  - `scj`: Scaling factor of asset `j`.
  - `kind`: The [`FullGerberIQ`](@ref) template.

# Returns

  - `res::Number`: Weight of the co-movement.

# Related

  - [`FullGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.3.4, Figure 2.
"""
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sci::Number,
                          scj::Number, kind::FullGerberIQ)
    (; dp1, dp2, dn1, dn2, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21) = kind
    dp1i = dp1 * sci
    dp2i = dp2 * sci
    dn1i = dn1 * sci
    dn2i = dn2 * sci
    dp1j = dp1 * scj
    dp2j = dp2 * scj
    dn1j = dn1 * scj
    dn2j = dn2 * scj
    zro = zero(xi)
    return if dp1i <= xi && dp1j <= xj
        n11
    elseif dp1i <= xi && dp2j <= xj < dp1j || dp1j <= xj && dp2i <= xi < dp1i
        n14
    elseif dp1i <= xi && zro < xj < dp2j || dp1j <= xj && zro < xi < dp2i
        n15
    elseif dp1i <= xi && -dn2j < xj < zro || dp1j <= xj && -dn2i < xi < zro
        n18
    elseif dp1i <= xi && -dn1j < xj <= -dn2j || dp1j <= xj && -dn1i < xi <= -dn2i
        n19
    elseif dp1i <= xi && xj <= -dn1j || dp1j <= xj && xi <= -dn1i
        n13
    elseif dp2i <= xi < dp1i && dp2j <= xj < dp1j
        n4
    elseif dp2i <= xi < dp1i && zro < xj < dp2j || dp2j <= xj < dp1j && zro < xi < dp2i
        n7
    elseif dp2i <= xi < dp1i && -dn2j < xj < zro || dp2j <= xj < dp1j && -dn2i < xi < zro
        n9
    elseif dp2i <= xi < dp1i && -dn1j < xj <= -dn2j ||
           dp2j <= xj < dp1j && -dn1i < xi <= -dn2i
        n6
    elseif dp2i <= xi < dp1i && xj <= -dn1j || dp2j <= xj < dp1j && xi <= -dn1i
        n20
    elseif zro < xi < dp2i && zro < xj < dp2j
        n1
    elseif zro < xi < dp2i && -dn2j < xj < zro || zro < xj < dp2j && -dn2i < xi < zro
        n3
    elseif zro < xi < dp2i && -dn1j < xj <= -dn2j || zro < xj < dp2j && -dn1i < xi <= -dn2i
        n10
    elseif zro < xi < dp2i && xj <= -dn1j || zro < xj < dp2j && xi <= -dn1i
        n21
    elseif -dn2i < xi < zro && -dn2j < xj < zro
        n2
    elseif -dn2i < xi < zro && -dn1j < xj <= -dn2j ||
           -dn2j < xj < zro && -dn1i < xi <= -dn2i
        n8
    elseif -dn2i < xi < zro && xj <= -dn1j || -dn2j < xj < zro && xi <= -dn1i
        n16
    elseif -dn1i < xi <= -dn2i && -dn1j < xj <= -dn2j
        n5
    elseif -dn1i < xi <= -dn2i && xj <= -dn1j || -dn1j < xj <= -dn2j && xi <= -dn1i
        n17
    elseif xi <= -dn1i && xj <= -dn1j
        n12
    else
        zro
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates a covariance matrix from the weighted and age-discounted co-movements of each pair of assets.

Four fields hold the parameters of the source. `c` is the noise threshold, and `kind` is the squeezing template that gives the weight of a co-movement. `sc` sets the units of the thresholds, and `decay` discounts a co-movement by its age. The source has no parameter for `alg`, because its statistic is the [`Gerber1`](@ref) statistic alone. This estimator also gives [`Gerber0`](@ref) and [`Gerber2`](@ref) from the classic Gerber family. [`gerber_IQ`](@ref) states the three statistics and their reduction to the Gerber family.

The lookback duration ``\\tau`` of the source has no field. The estimator always reads each row of `X`, and `decay` discounts the oldest rows. This is the case ``\\tau = T - 1`` of the source.

[`Gerber1`](@ref) is Equation 4 of [gerber2025squeezing](@cite). Its numerator sums the signed, weighted and discounted contributions over the observations on which both assets leave the noise zone. Its denominator sums the weights over the observations on which at least one asset leaves it. Equation 5 is the discount. A direct evaluation of Equations 4 and 5 for [`BasicGerberIQ`](@ref) agrees with this estimator to machine precision.

The library takes [`Gerber0`](@ref) and [`Gerber2`](@ref) from the classic Gerber family. The Internet Appendix of the source states a third form, which the library does not ship. Equation 1 of the source drops a return with ``\\lvert r^{*} \\rvert \\leq c``, so a return exactly at the threshold is noise there. The library lets that return cross, which is the closed rule of the Gerber statistic. The source centres the returns on a location ``r_0`` that the user chooses. The library centres them on the mean that `me` computes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GerberIQCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                         me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                         pdm::Option{<:AbstractPosdefEstimator} = Posdef(), c::Number = 0.5,
                         decay::GerberIQDecayEstimator = ExpGerberIQDecay(),
                         sc::Option{<:GerberIQScaler} = nothing,
                         kind::GerberIQCovarianceAlgorithm = BasicGerberIQ(),
                         alg::GerberCovarianceAlgorithm = Gerber1(),
                         ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx(),
                         cache::Option{<:AbstractPartialFitState} = nothing)

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks that `c` is finite and non-negative.
  - [`gerber_iq_assert_c_d`](@ref) checks that `c` is at most each boundary of `kind`.

!!! warning

    The constructor can replace `kind`. It gives the template to [`clamp_gerber_iq_n`](@ref), which lowers some weights under [`Gerber2`](@ref). So the stored template is not always the template that the caller gave. Read `ce.kind`, not the argument, when the exact weights matter.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated via [`factory`](@ref).
  - `me`: Recursively updated via [`factory`](@ref).
  - `decay`: Recursively updated via [`factory`](@ref).
  - `alg`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> GerberIQCovariance()
GerberIQCovariance
     ve ┼ SimpleVariance
        │          me ┼ SimpleExpectedReturns
        │             │   w ┴ nothing
        │           w ┼ nothing
        │   corrected ┴ Bool: true
     me ┼ SimpleExpectedReturns
        │   w ┴ nothing
    pdm ┼ Posdef
        │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
      c ┼ Float64: 0.5
  decay ┼ ExpGerberIQDecay
        │   e ┼ nothing
        │   y ┴ nothing
     sc ┼ nothing
   kind ┼ BasicGerberIQ
        │   d ┼ Float64: 2.0
        │   n ┴ Float64: 0.5
    alg ┼ Gerber1()
     ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`BaseGerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQDecayEstimator`](@ref)
  - [`GerberIQScaler`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.5, Equations 1 to 5, and the Internet Appendix.
"""
@propagatable @concrete struct GerberIQCovariance <: BaseGerberIQCovariance
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:pdm])
    """
    pdm
    """
    Noise threshold. A return less than `c` scaled units from zero is noise, and a return of exactly zero is noise for each `c`. The statistic drops a co-movement whose two returns are both noise. `c` must be at most each boundary of `kind`.
    """
    c
    """
    Temporal decay, a [`GerberIQDecayEstimator`](@ref). It discounts a co-movement by its age.
    """
    @fprop decay
    """
    Threshold scaler, a [`GerberIQScaler`](@ref). It sets the units of the thresholds of each pair. `nothing` gives both assets of a pair the mean of their two standard deviations.
    """
    sc
    """
    Squeezing template, a [`GerberIQCovarianceAlgorithm`](@ref). It gives the weight of a co-movement from the channel that the co-movement falls in.
    """
    kind
    """
    $(field_dict[:gerbalg])
    """
    @fprop alg
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function GerberIQCovariance(ve::StatsBase.CovarianceEstimator,
                                me::AbstractExpectedReturnsEstimator,
                                pdm::Option{<:AbstractPosdefEstimator}, c::Number,
                                decay::GerberIQDecayEstimator, sc::Option{<:GerberIQScaler},
                                kind::GerberIQCovarianceAlgorithm,
                                alg::GerberCovarianceAlgorithm,
                                ex::FLoops.Transducers.Executor,
                                cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(c, :c)
        gerber_iq_assert_c_d(c, kind)
        kind = clamp_gerber_iq_n(kind, alg)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(c), typeof(decay),
                   typeof(sc), typeof(kind), typeof(alg), typeof(ex), typeof(cache)}(ve, me,
                                                                                     pdm, c,
                                                                                     decay,
                                                                                     sc,
                                                                                     kind,
                                                                                     alg,
                                                                                     ex,
                                                                                     cache)
    end
end
function GerberIQCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                            me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                            pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                            c::Number = 0.5,
                            decay::GerberIQDecayEstimator = ExpGerberIQDecay(),
                            sc::Option{<:GerberIQScaler} = nothing,
                            kind::GerberIQCovarianceAlgorithm = BasicGerberIQ(),
                            alg::GerberCovarianceAlgorithm = Gerber1(),
                            ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx(),
                            cache::Option{<:AbstractPartialFitState} = nothing)
    return GerberIQCovariance(ve, me, pdm, c, decay, sc, kind, alg, ex, cache)
end

export AssetVolatilityGerberIQScaler, BasicGerberIQ, PartialGerberIQ, FullGerberIQ,
       ExpGerberIQDecay, GerberIQCovariance
public BaseGerberIQCovariance, GerberIQCovarianceAlgorithm, GerberIQEpsEstimator,
       GerberIQGammaEstimator, GerberIQScalerEstimator, GerberIQDecayEstimator,
       regenerate_decay, clamp_gerber_iq_n, gerber_iq_eps, gerber_iq_gamma,
       gerber_iq_scaling, gerber_iq_assert_c_d, gerber_iq_weight
