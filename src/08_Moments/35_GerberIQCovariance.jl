"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber Information Quality covariance estimators in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing Gerber Information Quality covariance estimation algorithms should be subtypes of `BaseGerberIQCovariance`.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
abstract type BaseGerberIQCovariance <: BaseGerberCovariance end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber Information Quality covariance estimation algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing Gerber Information Quality covariance estimation algorithms should be subtypes of `GerberIQCovarianceAlgorithm`.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
abstract type GerberIQCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op for Gerber Information Quality covariance estimation algorithms that do not need their noise suppression parameters clamped.

# Returns

  - `kind`: The input `kind` instance.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
"""
function clamp_gerber_iq_n(kind::GerberIQCovarianceAlgorithm, args...)
    return kind
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all temporal lookback and delay Gerber Information Quality parameter estimators in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing Gerber Information Quality parameter estimators should be subtypes of `GerberIQEpsEstimator`.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
abstract type GerberIQEpsEstimator <: AbstractEstimator end
"""
    const GerberIQEps = Union{<:Number, Function, <:GerberIQEpsEstimator}

A type alias for the union of `Number`, `Function`, and `GerberIQEpsEstimator` used for Gerber Information Quality lookback and delay parameter definitions.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
const GerberIQEps = Union{<:Number, Function, <:GerberIQEpsEstimator}
"""
    gerber_iq_eps(e::Number, ::MatNum) -> Number
    gerber_iq_eps(e::Function, X::MatNum) -> Number
    gerber_iq_eps(e::Option{<:GerberIQEpsEstimator}, X::MatNum) -> Number

Computes or returns the Gerber Information Quality delay parameter `e`, potentially using `X` as an input.

# Arguments

  - `e`: The delay parameter estimator, function or value for use in the decay equation.
      + `::Number`: Use the number as-is.
      + `::Function`: A function which takes the data matrix `X` as an argument and returns a `Number`.
      + `::Option{<:GerberIQEpsEstimator}`: Fallback returning `round(Int, T - T / N)`, where `T` and `N` are the number of rows and columns of `X` respectively.
  - $(arg_dict[:X])

# Returns

  - `tau::Number`: The lookback parameter for use in the decay equation.

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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

Abstract supertype for Gerber IQ estimators for tuning the strength of the lookback decay.

All concrete and/or abstract types implementing Gerber Information Quality parameter estimators should be subtypes of `GerberIQGammaEstimator`.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
abstract type GerberIQGammaEstimator <: AbstractEstimator end
"""
    const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}

A type alias for the union of `Number`, `Function`, and `GerberIQGammaEstimator` used for Gerber Information Quality temporal decay parameter definitions.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}
"""
    gerber_iq_gamma(y::Number, ::MatNum) -> Number
    gerber_iq_gamma(y::Function, X::MatNum) -> Number
    gerber_iq_gamma(y::Option{<:GerberIQGammaEstimator}, X::MatNum) -> Number

Computes or returns the Gerber Information Quality decay strength parameter `y`, potentially using `X` as an input.

# Arguments

  - `y`: The decay strength parameter estimator, function or value for use in the decay equation.
      + `::Number`: Use the number as-is.
      + `::Function`: A function which takes the data matrix `X` as an argument and returns a `Number`.
      + `::Option{<:GerberIQGammaEstimator}`: Fallback returning `log(2) / size(X, 2)`.
  - $(arg_dict[:X])

# Returns

  - `gamma::Number`: The decay strength parameter for use in the decay equation.

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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

Abstract supertype for Gerber IQ estimators for scaling the threshold parameters for defining significant co-movements.

All concrete and/or abstract types implementing threshold scalers for Gerber Information Quality parameter estimators should be subtypes of `GerberIQScalerEstimator`.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
abstract type GerberIQScalerEstimator <: AbstractEstimator end
"""
    const GerberIQScaler = Union{Function, <:GerberIQScalerEstimator}

A type alias for the union of `Function`, and `GerberIQScalerEstimator` used for scaling the threshold parameters for defining significant co-movements in Gerber Information Quality.

# Related

  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
const GerberIQScaler = Union{Function, <:GerberIQScalerEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Scales the threshold parameters using the individual asset volatilities.

# Related

  - [`GerberIQScalerEstimator`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`gerber_iq_scaling`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
struct AssetVolatilityGerberIQScaler <: GerberIQScalerEstimator end
"""
    gerber_iq_scaling(sca::AssetVolatilityGerberIQScaler, sdi::Number, sdj::Number) -> (Number, Number)
    gerber_iq_scaling(sca::Function, sdi::Number, sdj::Number) -> (Number, Number)
    gerber_iq_scaling(sca::Option{<:GerberIQScalerEstimator}, sdi::Number, sdj::Number) -> (Number, Number)

Computes or returns the threshold scaling parameters for defining significant co-movements in Gerber Information Quality.

# Arguments

  - `sca`: The scaling estimator to use.

      + `::AssetVolatilityGerberIQScaler`: Returns the input `sdi` and `sdj` as-is. This lets each asset scale according to its own volatility.
      + `::Option{<:GerberIQScalerEstimator}`: Fallback returning the mean of `sdi` and `sdj` twice so each asset is scaled according to the mean of the two asset volatilities. Overloading this with a custom [`GerberIQScalerEstimator`](@ref) allows for custom scaling behavior.
      + `::Function`: Custom scaling function that takes `sdi` and `sdj` as arguments and returns the scaled values.

# Returns

  - `scai::Number`: The scaled value for `sdi`.
  - `scaj::Number`: The scaled value for `sdj`.

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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

Abstract supertype for Gerber IQ estimators for scaling the threshold parameters for defining significant co-movements.

All concrete and/or abstract types implementing threshold scalers for Gerber Information Quality parameter estimators should be subtypes of `GerberIQDecayEstimator`.

# Interfaces

In order to implement a new Gerber IQ decay estimator which will work seamlessly with the library, subtype `GerberIQDecayEstimator` with all necessary parameters as part of the struct, and implement the following methods:

## Regenerate Decay

  - `PortfolioOptimisers.regenerate_decay(decay::GerberIQDecayEstimator, X::AbstractMatrix) -> GerberIQDecayEstimator`: Fallback for automatically computing the decay parameters based on the input data `X`.

### Arguments

  - `decay`: The decay estimator to regenerate.
  - $(arg_dict[:X])

### Returns

  - `decay::GerberIQDecayEstimator`: A new concrete instance of the subtype of `GerberIQDecayEstimator` with the decay parameters generated from the input data `X`.

## Functor

  - `(decay::GerberIQDecayEstimator)(T::Number, k::Number) -> Number`: Evaluate the decay estimator for observation `k` out of `T`.

### Arguments

  - `T::Number`: The total number of observations.
  - `k::Number`: The current observation index.

### Returns

  - `d::Number`: The decay value for observation `k` out of `T`.

# Examples

We can create a dummy Gerber IQ decay estimator as follows:

```jldoctest
julia> struct GaussianDecay{T} <: PortfolioOptimisers.GerberIQDecayEstimator
           a::T
           function GaussianDecay(a::Union{Nothing, <:Number})
               if isa(a, Number)
                   @assert(a >= 0)
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

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
abstract type GerberIQDecayEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

A concrete type for exponential Gerber IQ temporal decay.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpGerberIQDecay(e::Option{<:GerberIQEps} = nothing,
                     y::Option{<:GerberIQGamma} = nothing)

Keywords correspond to the struct's fields.

# Functors

    (decay::ExpGerberIQDecay)(T::Number, k::Number) -> Number

Implements the exponential decay for Gerber IQ covariance.

!!! Warning

    The functor is not meant to be called directly unless all parameters are numeric. Otherwise, call [`regenerate_decay`](@ref) first.

## Mathematical definition

```math
\\begin{align}
d &= \\exp\\left[-y \\max(0, T - k - e)\\right]\\,.
\\end{align}
```

Where:

  - ``T``: Is the number of observations.
  - ``k``: Is the current index.
  - ``e``: Parameter in the instance of [`ExpGerberIQDecay`](@ref).
  - ``y``: Parameter in the instance of [`ExpGerberIQDecay`](@ref).
  - ``d``: Is the decay factor.

## Arguments

  - `T`: Number of observations.
  - `k`: Current time index.

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

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
@concrete struct ExpGerberIQDecay <: GerberIQDecayEstimator
    """
    Waiting period before the decay starts.
    """
    e
    """
    Decay rate parameter.
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

Compute the exponential decay weight for a single observation at lag `T - k`.

# Mathematical definition

```math
\\begin{align}
w_k &= \\exp\\!\\left(-y \\cdot \\max(0,\\, T - k - e)\\right)\\,.
\\end{align}
```

Where:

  - ``w_k``: Decay weight for observation at index ``k``.
  - ``y``: Decay rate parameter (`decay.y`).
  - ``e``: Lookback delay (`decay.e`); observations within the last ``e`` periods are not discounted.
  - $(math_dict[:T])
  - ``k``: Observation index (``1 \\leq k \\leq T``).

# Arguments

  - `decay`: Fitted [`ExpGerberIQDecay`](@ref) with numeric `e` and `y` fields.
  - `T`: Total number of observations.
  - `k`: Index of the current observation.

# Returns

  - `w::Number`: Decay weight for observation `k`.

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

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function (decay::ExpGerberIQDecay)(T::Number, k::Number)
    return exp(-decay.y * max(0, T - k - decay.e))
end
"""
    regenerate_decay(decay::ExpGerberIQDecay, X::AbstractMatrix) -> ExpGerberIQDecay
    regenerate_decay(decay::GerberIQDecayEstimator, X::AbstractMatrix) -> ExpGerberIQDecay

Automatically sets the decay parameters based on the input data `X`.

# Arguments

  - `decay`: The decay estimator to regenerate.
      + `::ExpGerberIQDecay`: If both parameters are numeric, returns the input, otherwise returns a new [`ExpGerberIQDecay`](@ref) with the regenerated parameters.
      + `::GerberIQDecayEstimator`: Fallback for automatically setting the decay parameters `e`, and `y` based on the input data `X`, using [`gerber_iq_eps`](@ref) and [`gerber_iq_gamma`](@ref) with `nothing` as the first input. Custom subtypes of [`GerberIQDecayEstimator`](@ref) should implement this method, else they default to the fallback.
  - $(arg_dict[:X])

# Returns

  - `decay::ExpGerberIQDecay`: With parameters based on `X`.

# Details

  - Calls [`gerber_iq_eps`](@ref) to set and `e`.
  - Calls [`gerber_iq_gamma`](@ref) to set `y`.
  - Returns a new [`ExpGerberIQDecay`](@ref) with the regenerated parameters.

# Related

  - [`GerberIQDecayEstimator`](@ref)
  - [`ExpGerberIQDecay`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`gerber_iq_gamma`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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

Implements the basic Gerber IQ covariance template. Divides the comovement data into regions and applies the co-movement compression to co-movements falling within each region. Co-movements within the dashed regions may or may not be included depending on the GerberIQ algorithm used. Co-movements within the central region are always ignored.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BasicGerberIQ(; d::Number = 2.0, n::Number = 0.5)

Keywords correspond to the struct's fields.

# Validation

  - `d` is validated via [`assert_nonempty_nonneg_finite_val`](@ref).
  - `0 <= n <= 1`.

# Details

The diagram shows a visual representation of the regions defined by `BasicGerberIQ`. In this case `c = 1` and `d = 3`.

  - The dashed lines indicate the limits of the areas where movements are considered small.
  - Only the [`Gerber1`](@ref) algorithm takes these regions into account as part of the neutral count.
  - The region where co-movements are considered insignificant in both axes (square around r0) are always ignored.
  - Single weight lines indicate zero delimiters.
  - The double weight lines indicate areas where movements are considered significant.
  - Co-movements within each region are weighed according to their labels.

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

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
@concrete struct BasicGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Significance threshold parameter.
    """
    d
    """
    Comovement compression parameter.
    """
    n
    function BasicGerberIQ(d::Number, n::Number)
        assert_nonempty_nonneg_finite_val(d, :d)
        @argcheck(zero(n) <= n <= one(n), DomainError(n, "n must be in [0, 1]"))
        return new{typeof(d), typeof(n)}(d, n)
    end
end
function BasicGerberIQ(; d::Number = 2.0, n::Number = 0.5)::BasicGerberIQ
    return BasicGerberIQ(d, n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Asserts that `c <= kind.d`, where `c` is the small movement threshold and `d` the significance threshold parameter of [`BasicGerberIQ`](@ref).

# Arguments

  - `c`: Small movement threshold.
  - `kind`: [`BasicGerberIQ`](@ref) instance.

# Related

  - [`BasicGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function gerber_iq_assert_c_d(c::Number, kind::BasicGerberIQ)
    @argcheck(c <= kind.d,
              DomainError("c must be <= kind.d, got c = $c, kind.d = $(kind.d)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight for a co-movement according to the region it falls into from the [`BasicGerberIQ`](@ref) template.

# Arguments

  - `xi`: Return of asset `i` (unused).
  - `xj`: Return of asset `j` (unused).
  - `axi`: Absolute return of asset `i`.
  - `axj`: Absolute return of asset `j`.
  - `sci`: Scaling for movement of asset `i`.
  - `scj`: Scaling for movement of asset `j`.
  - `kind`: Instance of [`BasicGerberIQ`](@ref).

# Returns

  - `res::Number`: Co-movement weight.

# Related

  - [`BasicGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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

Implements a partial Gerber Information Quality covariance template with independently configurable asymmetric significance thresholds for concordant and discordant co-movements.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PartialGerberIQ(; dcp::Number = 2.0, dcn::Number = dcp, ddp::Number = dcp,
                      ddn::Number = dcp, n1::Number = 0.5, n2::Number = n1,
                      n3::Number = n1, n4::Number = 1.0, n5::Number = n4,
                      n6::Number = n4, n7::Number = sqrt(n1 * n4),
                      n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n3 * n6),
                      n10::Number = sqrt(n3 * n6))

Keywords correspond to the struct's fields.

# Validation

  - All `d**` parameters are validated via [`assert_nonempty_nonneg_finite_val`](@ref).
  - All `n**` parameters must be `0 <= n** <= 1`.

# Details

The diagram shows a visual representation of the regions defined by `PartialGerberIQ`. In this case `c = 1`, `dcp = 2`, `ddn = 2`, `ddp = 3`, and `dcn = 3`.

  - The dashed lines indicate the limits of the areas where movements are considered small.
  - Only the [`Gerber1`](@ref) algorithm takes these regions into account as part of the neutral count.
  - The region where co-movements are considered insignificant in both axes (square around r0) are always ignored.
  - Single weight lines indicate zero delimiters.
  - The double weight lines indicate areas where movements are considered significant.
  - Co-movements within each region are weighed according to their labels.

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

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
@concrete struct PartialGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Positive concordant movement threshold parameter.
    """
    dcp
    """
    Negative concordant movement threshold parameter.
    """
    dcn
    """
    Discordant movement threshold parameter, positive in `r_i`, negative in `r_j`.
    """
    ddp
    """
    Discordant movement threshold parameter, negative in `r_i`, positive in `r_j`.
    """
    ddn
    """
    Noise suppression parameter for small positive concordant movements.
    """
    n1
    """
    Noise suppression parameter for small negative concordant movements.
    """
    n2
    """
    Noise suppression parameter for small discordant movements.
    """
    n3
    """
    Noise suppression parameter for large positive concordant movements.
    """
    n4
    """
    Noise suppression parameter for large negative concordant movements.
    """
    n5
    """
    Noise suppression parameter for large discordant movements.
    """
    n6
    """
    Noise suppression parameter for positive concordant movements where one axis has a large movement and the other is small.
    """
    n7
    """
    Noise suppression parameter for negative concordant movements where one axis has a large movement and the other is small.
    """
    n8
    """
    Noise suppression parameter for discordant movements where one axis has a large movement and the other is small for the region between `ddn` and zero.
    """
    n9
    """
    Noise suppression parameter for discordant movements where one axis has a large movement and the other is small for the region between `ddp` and zero.
    """
    n10
    function PartialGerberIQ(dcp::Number, dcn::Number, ddp::Number, ddn::Number, n1::Number,
                             n2::Number, n3::Number, n4::Number, n5::Number, n6::Number,
                             n7::Number, n8::Number, n9::Number, n10::Number)
        assert_nonempty_nonneg_finite_val(dcp, :dcp)
        assert_nonempty_nonneg_finite_val(dcn, :dcn)
        assert_nonempty_nonneg_finite_val(ddp, :ddp)
        assert_nonempty_nonneg_finite_val(ddn, :ddn)
        @argcheck(zero(n1) <= n1 <= one(n1), DomainError(n1, "n1 must be in [0, 1]"))
        @argcheck(zero(n2) <= n2 <= one(n2), DomainError(n2, "n2 must be in [0, 1]"))
        @argcheck(zero(n3) <= n3 <= one(n3), DomainError(n3, "n3 must be in [0, 1]"))
        @argcheck(zero(n4) <= n4 <= one(n4), DomainError(n4, "n4 must be in [0, 1]"))
        @argcheck(zero(n5) <= n5 <= one(n5), DomainError(n5, "n5 must be in [0, 1]"))
        @argcheck(zero(n6) <= n6 <= one(n6), DomainError(n6, "n6 must be in [0, 1]"))
        @argcheck(zero(n7) <= n7 <= one(n7), DomainError(n7, "n7 must be in [0, 1]"))
        @argcheck(zero(n8) <= n8 <= one(n8), DomainError(n8, "n8 must be in [0, 1]"))
        @argcheck(zero(n9) <= n9 <= one(n9), DomainError(n9, "n9 must be in [0, 1]"))
        @argcheck(zero(n10) <= n10 <= one(n10), DomainError(n10, "n10 must be in [0, 1]"))
        return new{typeof(dcp), typeof(dcn), typeof(ddp), typeof(ddn), typeof(n1),
                   typeof(n2), typeof(n3), typeof(n4), typeof(n5), typeof(n6), typeof(n7),
                   typeof(n8), typeof(n9), typeof(n10)}(dcp, dcn, ddp, ddn, n1, n2, n3, n4,
                                                        n5, n6, n7, n8, n9, n10)
    end
end
function PartialGerberIQ(; dcp::Number = 2.0, dcn::Number = dcp, ddp::Number = dcp,
                         ddn::Number = dcp, n1::Number = 0.5, n2::Number = n1,
                         n3::Number = n1, n4::Number = 1.0, n5::Number = n4,
                         n6::Number = n4, n7::Number = sqrt(n1 * n4),
                         n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n3 * n6),
                         n10::Number = sqrt(n3 * n6))::PartialGerberIQ
    return PartialGerberIQ(dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Clamps the values of the off-diagonal elements of the covariance matrix for the [`PartialGerberIQ`](@ref) template when using the [`Gerber2`](@ref) algorithm to ensure positive definiteness.

# Arguments

  - `alg`: Instance of [`PartialGerberIQ`](@ref).
  - `::Gerber2`: Instance of [`Gerber2`](@ref).

# Details

  - Clamps off-diagonal elements to be at most equal to the geometric mean of its adjacent diagonal elements.
  - This affects `n7` and `n8`.

# Related

  - [`PartialGerberIQ`](@ref)
  - [`Gerber2`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function clamp_gerber_iq_n(alg::PartialGerberIQ, ::Gerber2)
    (; n1, n2, n4, n5, n7, n8) = alg
    n7 = min(n7, sqrt(n1 * n4))
    n8 = min(n8, sqrt(n2 * n5))
    return PartialGerberIQ(; dcp = alg.dcp, dcn = alg.dcn, ddp = alg.ddp, ddn = alg.ddn,
                           n1 = n1, n2 = n2, n3 = alg.n3, n4 = n4, n5 = n5, n6 = alg.n6,
                           n7 = n7, n8 = n8, n9 = alg.n9, n10 = alg.n10)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight for a co-movement according to the region it falls into from the [`PartialGerberIQ`](@ref) template.

# Arguments

  - `xi`: Return of asset `i`.
  - `xj`: Return of asset `j`.
  - `axi`: Absolute return of asset `i` (unused).
  - `axj`: Absolute return of asset `j` (unused).
  - `sci`: Scaling for movement of asset `i`.
  - `scj`: Scaling for movement of asset `j`.
  - `kind`: Instance of [`PartialGerberIQ`](@ref).

# Returns

  - `res::Number`: Co-movement weight.

# Related

  - [`PartialGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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

Implements a full Gerber Information Quality covariance template with fine-grained asymmetric thresholds, supporting finer region classification between two positive and two negative movement magnitude classes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FullGerberIQ(; dp1::Number = 2.0, dp2::Number = dp1, dn1::Number = dp1,
                   dn2::Number = dp1, n1::Number = 0.5, n2::Number = n1, n3::Number = n1,
                   n4::Number = 0.75, n5::Number = n4, n6::Number = n4,
                   n7::Number = sqrt(n1 * n4), n8::Number = sqrt(n2 * n5),
                   n9::Number = sqrt(n3 * n6), n10::Number = sqrt(n3 * n6),
                   n11::Number = 1.0, n12::Number = n11, n13::Number = n11,
                   n14::Number = sqrt(n4 * n11), n15::Number = sqrt(n7 * n14),
                   n17::Number = sqrt(n5 * n12), n16::Number = sqrt(n8 * n17),
                   n19::Number = sqrt(n6 * n13), n18::Number = sqrt(n9 * n19),
                   n20::Number = sqrt(n6 * n13), n21::Number = sqrt(n10 * n20))

Keywords correspond to the struct's fields.

# Validation

  - All `d**` parameters are validated via [`assert_nonempty_nonneg_finite_val`](@ref).
  - All `n**` parameters must be `0 <= n** <= 1`.

# Details

The diagram shows a visual representation of the regions defined by `PartialGerberIQ`. In this case `c = 1`, `dp2 = 2`, `dn2 = 2`, `dp1 = 3`, and `dn1 = 3`. In this version, the limits are allowed to cross over the zero line. Thus, the constructor ensures `dp1 >= dp2` and `dn1 >= dn2` by swapping values if necessary to ensure consistency.

  - The dashed lines indicate the limits of the areas where movements are considered small.
  - Only the [`Gerber1`](@ref) algorithm takes these regions into account as part of the neutral count.
  - The region where co-movements are considered insignificant in both axes (square around r0) are always ignored.
  - Single weight lines indicate zero delimiters.
  - The double weight lines indicate areas where movements are considered significant.
  - Co-movements within each region are weighed according to their labels.

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
  n15 ┼ Float64: 0.7282376575609851
  n16 ┼ Float64: 0.7282376575609851
  n17 ┼ Float64: 0.8660254037844386
  n18 ┼ Float64: 0.7282376575609851
  n19 ┼ Float64: 0.8660254037844386
  n20 ┼ Float64: 0.8660254037844386
  n21 ┴ Float64: 0.7282376575609851
```

# Related

  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
@concrete struct FullGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Threshold for larger positive co-movements.
    """
    dp1
    """
    Threshold for smaller positive co-movements.
    """
    dp2
    """
    Threshold for larger negative co-movements.
    """
    dn1
    """
    Threshold for smaller negative co-movements.
    """
    dn2
    """
    Threshold for small positive concordant co-movements.
    """
    n1
    """
    Threshold for small negative concordant co-movements.
    """
    n2
    """
    Threshold for small discordant co-movements.
    """
    n3
    """
    Threshold for moderate positive concordant co-movements.
    """
    n4
    """
    Threshold for moderate negative concordant co-movements.
    """
    n5
    """
    Threshold for moderate discordant co-movements.
    """
    n6
    """
    Threshold for small and moderate positive concordant co-movements.
    """
    n7
    """
    Threshold for small and moderate negative concordant co-movements.
    """
    n8
    """
    Threshold for small and moderate discordant co-movements.
    """
    n9
    """
    Threshold for moderate and small discordant co-movements.
    """
    n10
    """
    Threshold for large positive concordant co-movements.
    """
    n11
    """
    Threshold for large negative concordant co-movements.
    """
    n12
    """
    Threshold for large discordant co-movements.
    """
    n13
    """
    Threshold for moderate and large positive concordant co-movements.
    """
    n14
    """
    Threshold for small and large positive concordant co-movements.
    """
    n15
    """
    Threshold for small and large negative concordant co-movements.
    """
    n16
    """
    Threshold for moderate and large positive concordant co-movements.
    """
    n17
    """
    Threshold for small and large discordant co-movements.
    """
    n18
    """
    Threshold for moderate and large discordant co-movements.
    """
    n19
    """
    Threshold for large and moderate discordant co-movements.
    """
    n20
    """
    Threshold for large and small discordant co-movements.
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
        @argcheck(zero(n1) <= n1 <= one(n1), DomainError(n1, "n1 must be in [0, 1]"))
        @argcheck(zero(n2) <= n2 <= one(n2), DomainError(n2, "n2 must be in [0, 1]"))
        @argcheck(zero(n3) <= n3 <= one(n3), DomainError(n3, "n3 must be in [0, 1]"))
        @argcheck(zero(n4) <= n4 <= one(n4), DomainError(n4, "n4 must be in [0, 1]"))
        @argcheck(zero(n5) <= n5 <= one(n5), DomainError(n5, "n5 must be in [0, 1]"))
        @argcheck(zero(n6) <= n6 <= one(n6), DomainError(n6, "n6 must be in [0, 1]"))
        @argcheck(zero(n7) <= n7 <= one(n7), DomainError(n7, "n7 must be in [0, 1]"))
        @argcheck(zero(n8) <= n8 <= one(n8), DomainError(n8, "n8 must be in [0, 1]"))
        @argcheck(zero(n9) <= n9 <= one(n9), DomainError(n9, "n9 must be in [0, 1]"))
        @argcheck(zero(n10) <= n10 <= one(n10), DomainError(n10, "n10 must be in [0, 1]"))
        @argcheck(zero(n11) <= n11 <= one(n11), DomainError(n11, "n11 must be in [0, 1]"))
        @argcheck(zero(n12) <= n12 <= one(n12), DomainError(n12, "n12 must be in [0, 1]"))
        @argcheck(zero(n13) <= n13 <= one(n13), DomainError(n13, "n13 must be in [0, 1]"))
        @argcheck(zero(n14) <= n14 <= one(n14), DomainError(n14, "n14 must be in [0, 1]"))
        @argcheck(zero(n15) <= n15 <= one(n15), DomainError(n15, "n15 must be in [0, 1]"))
        @argcheck(zero(n16) <= n16 <= one(n16), DomainError(n16, "n16 must be in [0, 1]"))
        @argcheck(zero(n17) <= n17 <= one(n17), DomainError(n17, "n17 must be in [0, 1]"))
        @argcheck(zero(n18) <= n18 <= one(n18), DomainError(n18, "n18 must be in [0, 1]"))
        @argcheck(zero(n19) <= n19 <= one(n19), DomainError(n19, "n19 must be in [0, 1]"))
        @argcheck(zero(n20) <= n20 <= one(n20), DomainError(n20, "n20 must be in [0, 1]"))
        @argcheck(zero(n21) <= n21 <= one(n21), DomainError(n21, "n21 must be in [0, 1]"))
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
                      dn2::Number = dp1, n1::Number = 0.5, n2::Number = n1, n3::Number = n1,
                      n4::Number = 0.75, n5::Number = n4, n6::Number = n4,
                      n7::Number = sqrt(n1 * n4), n8::Number = sqrt(n2 * n5),
                      n9::Number = sqrt(n3 * n6), n10::Number = sqrt(n3 * n6),
                      n11::Number = 1.0, n12::Number = n11, n13::Number = n11,
                      n14::Number = sqrt(n4 * n11), n15::Number = sqrt(n7 * n14),
                      n17::Number = sqrt(n5 * n12), n16::Number = sqrt(n8 * n17),
                      n19::Number = sqrt(n6 * n13), n18::Number = sqrt(n9 * n19),
                      n20::Number = sqrt(n6 * n13), n21::Number = sqrt(n10 * n20))
    return FullGerberIQ(dp1, dp2, dn1, dn2, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11,
                        n12, n13, n14, n15, n16, n17, n18, n19, n20, n21)
end
"""
    gerber_iq_assert_c_d(c::Number, kind::Union{<:PartialGerberIQ, <:FullGerberIQ}) -> Nothing

Asserts that all `c <= kind.d**`, where `c` is the small movement threshold and `d**` are the significance threshold parameters of [`PartialGerberIQ`](@ref) or [`FullGerberIQ`](@ref).

# Arguments

  - `c`: Small movement threshold.
  - `kind`: Instance of [`PartialGerberIQ`](@ref) or [`FullGerberIQ`](@ref).

# Related

  - [`PartialGerberIQ`](@ref)
  - [`FullGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function gerber_iq_assert_c_d(c::Number, kind::PartialGerberIQ)
    @argcheck(c <= kind.dcp, DomainError("c ($c) must be <= kind.dcp ($(kind.dcp))"))
    @argcheck(c <= kind.dcn, DomainError("c ($c) must be <= kind.dcn ($(kind.dcn))"))
    @argcheck(c <= kind.ddp, DomainError("c ($c) must be <= kind.ddp ($(kind.ddp))"))
    @argcheck(c <= kind.ddn, DomainError("c ($c) must be <= kind.ddn ($(kind.ddn))"))
    return nothing
end
function gerber_iq_assert_c_d(c::Number, kind::FullGerberIQ)
    @argcheck(c <= kind.dp1, DomainError("c ($c) must be <= kind.dp1 ($(kind.dp1))"))
    @argcheck(c <= kind.dp2, DomainError("c ($c) must be <= kind.dp2 ($(kind.dp2))"))
    @argcheck(c <= kind.dn1, DomainError("c ($c) must be <= kind.dn1 ($(kind.dn1))"))
    @argcheck(c <= kind.dn2, DomainError("c ($c) must be <= kind.dn2 ($(kind.dn2))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Clamps the values of the off-diagonal elements of the covariance matrix for the [`FullGerberIQ`](@ref) template when using the [`Gerber2`](@ref) algorithm to ensure positive definiteness.

# Arguments

  - `alg`: Instance of [`FullGerberIQ`](@ref).
  - `::Gerber2`: Instance of [`Gerber2`](@ref).

# Details

  - Clamps off-diagonal elements to be at most equal to the geometric mean of its adjacent diagonal elements.
  - This affects `n7`, `n8`, `n14`, and `n17`.

# Related

  - [`FullGerberIQ`](@ref)
  - [`Gerber2`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function clamp_gerber_iq_n(alg::FullGerberIQ, ::Gerber2)
    (; n1, n2, n4, n5, n7, n8, n11, n12, n14, n17) = alg
    n7 = min(n7, sqrt(n1 * n4))
    n8 = min(n8, sqrt(n2 * n5))
    n14 = min(n14, sqrt(n4 * n11))
    n17 = min(n17, sqrt(n5 * n12))
    return FullGerberIQ(; dp1 = alg.dp1, dp2 = alg.dp2, dn1 = alg.dn1, dn2 = alg.dn2,
                        n1 = n1, n2 = n2, n3 = alg.n3, n4 = n4, n5 = n5, n6 = alg.n6,
                        n7 = n7, n8 = n8, n9 = alg.n9, n10 = alg.n10, n11 = n11, n12 = n12,
                        n13 = alg.n13, n14 = n14, n15 = alg.n15, n16 = alg.n16, n17 = n17,
                        n18 = alg.n18, n19 = alg.n19, n20 = alg.n20, n21 = alg.n21)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight for a co-movement according to the region it falls into from the [`FullGerberIQ`](@ref) template.

# Arguments

  - `xi`: Return of asset `i`.
  - `xj`: Return of asset `j`.
  - `axi`: Absolute return of asset `i` (unused).
  - `axj`: Absolute return of asset `j` (unused).
  - `sci`: Scaling for movement of asset `i`.
  - `scj`: Scaling for movement of asset `j`.
  - `kind`: Instance of [`FullGerberIQ`](@ref).

# Returns

  - `res::Number`: Co-movement weight.

# Related

  - [`FullGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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
    elseif dp1i <= xi && -dn1j < xj < zro || dp1j <= xj && -dn1i < xi < zro
        n18
    elseif dp1i <= xi && -dn2j < xj <= -dn1j || dp1j <= xj && -dn2i < xi <= -dn1i
        n19
    elseif dp1i <= xi && xj <= -dn2j || dp1j <= xj && xi <= -dn2i
        n13
    elseif dp2i <= xi < dp1i && dp2j <= xj < dp1j
        n4
    elseif dp2i <= xi < dp1i && zro < xj < dp2j || dp2j <= xj < dp1j && zro < xi < dp2i
        n7
    elseif dp2i <= xi < dp1i && -dn1j < xj < zro || dp2j <= xj < dp1j && -dn1i < xi < zro
        n9
    elseif dp2i <= xi < dp1i && -dn2j < xj <= -dn1j ||
           dp2j <= xj < dp1j && -dn2i < xi <= -dn1i
        n6
    elseif dp2i <= xi < dp1i && xj <= -dn2j || dp2j <= xj < dp1j && xi <= -dn2i
        n20
    elseif zro < xi < dp2i && zro < xj < dp2j
        n1
    elseif zro < xi < dp2i && -dn1j < xj < zro || zro < xj < dp2j && -dn1i < xi < zro
        n3
    elseif zro < xi < dp2i && -dn2j < xj <= -dn1j || zro < xj < dp2j && -dn2i < xi <= -dn1i
        n10
    elseif zro < xi < dp2i && xj <= -dn2j || zro < xj < dp2j && xi <= -dn2i
        n21
    elseif -dn1i < xi < zro && -dn1j < xj < zro
        n2
    elseif -dn1i < xi < zro && -dn2j < xj <= -dn1j ||
           -dn1j < xj < zro && -dn2i < xi <= -dn1i
        n8
    elseif -dn1i < xi < zro && xj <= -dn2j || -dn1j < xj < zro && xi <= -dn2i
        n16
    elseif -dn2i < xi <= -dn1i && -dn2j < xj <= -dn1j
        n5
    elseif -dn2i < xi <= -dn1i && xj <= -dn2j || -dn2j < xj <= -dn1j && xi <= -dn2i
        n17
    elseif xi <= -dn2i && xj <= -dn2j
        n12
    else
        zro
    end
end
"""
$(DocStringExtensions.TYPEDEF)

A flexible container type for configuring and applying Gerber Information Quality covariance estimators in `PortfolioOptimisers.jl`.

`GerberIQCovariance` encapsulates all components required for Gerber Information Quality based covariance or correlation estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GerberIQCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                         me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                         pdm::Option{<:Posdef} = Posdef(), c::Number = 0.5,
                         decay::GerberIQDecayEstimator = ExpGerberIQDecay(),
                         sc::Option{<:GerberIQScaler} = nothing,
                         kind::GerberIQCovarianceAlgorithm = BasicGerberIQ(),
                         alg::GerberCovarianceAlgorithm = Gerber1(),
                         ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx())

Keywords correspond to the struct's fields.

## Validation

  - `c >= 0`: `c` must be non-negative.
  - `c <= kind.d` (or equivalent for the chosen `kind`): via [`gerber_iq_assert_c_d`](@ref).

# Related

  - [`BaseGerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQDecayEstimator`](@ref)
  - [`GerberIQScaler`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
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
    Small co-movement threshold.
    """
    c
    """
    Temporal decay rate estimator for past observations [`GerberIQDecayEstimator`](@ref).
    """
    @fprop decay
    """
    Threshold scaling factor estimator for co-movement thresholds [`GerberIQScaler`](@ref).
    """
    sc
    """
    Gerber IQ covariance kind for squeezing co-movement noise [`GerberIQCovarianceAlgorithm`](@ref).
    """
    kind
    """
    $(field_dict[:gerbalg])
    """
    @fprop alg
    """
    $(field_dict[:ex]).
    """
    ex
    function GerberIQCovariance(ve::StatsBase.CovarianceEstimator,
                                me::AbstractExpectedReturnsEstimator, pdm::Option{<:Posdef},
                                c::Number, decay::GerberIQDecayEstimator,
                                sc::Option{<:GerberIQScaler},
                                kind::GerberIQCovarianceAlgorithm,
                                alg::GerberCovarianceAlgorithm,
                                ex::FLoops.Transducers.Executor)
        assert_nonempty_nonneg_finite_val(c, :c)
        gerber_iq_assert_c_d(c, kind)
        kind = clamp_gerber_iq_n(kind, alg)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(c), typeof(decay),
                   typeof(sc), typeof(kind), typeof(alg), typeof(ex)}(ve, me, pdm, c, decay,
                                                                      sc, kind, alg, ex)
    end
end
function GerberIQCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                            me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                            pdm::Option{<:Posdef} = Posdef(), c::Number = 0.5,
                            decay::GerberIQDecayEstimator = ExpGerberIQDecay(),
                            sc::Option{<:GerberIQScaler} = nothing,
                            kind::GerberIQCovarianceAlgorithm = BasicGerberIQ(),
                            alg::GerberCovarianceAlgorithm = Gerber1(),
                            ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx())
    return GerberIQCovariance(ve, me, pdm, c, decay, sc, kind, alg, ex)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the Gerber IQ statistic for a single co-movement.

# Arguments

  - `xi`: Return for asset `i`.
  - `xj`: Return for asset `j`.
  - `axi`: Absolute return for asset `i`.
  - `axj`: Absolute return for asset `j`.
  - `decay`: The decay estimator for the Gerber IQ statistic.
  - `T`: The number of observations.
  - `k`: The current observation.
  - `sci`: Scaling factor for asset `i`.
  - `scj`: Scaling factor for asset `j`.
  - `kind`: The Gerber IQ co-movement template.

# Returns

  - `rho::Number`: The Gerber IQ statistic.

# Details

  - Calls [`gerber_iq_weight`](@ref) to compute the Gerber IQ weight.
  - Calls the functor of `decay` to compute the decay factor.
  - Returns the product of them both.

# Related

  - [`gerber_iq_weight`](@ref)
  - [`GerberIQDecayEstimator`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function gerber_IQ_delta(xi::Number, xj::Number, axi::Number, axj::Number,
                         decay::GerberIQDecayEstimator, T::Integer, k::Number, sci::Number,
                         scj::Number, kind::GerberIQCovarianceAlgorithm)
    w = gerber_iq_weight(xi, xj, axi, axj, sci, scj, kind)
    p = decay(T, k)
    return w * p
end
"""
    gerber_IQ(
        ce::GerberIQCovariance,
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Computes the Gerber IQ statistic matrix using noise compression template in `ce.kind` and numerator/denominator definition according to `ce.alg`.

# Mathematical definition

For each asset pair ``(i,j)`` accumulate weighted concordant and discordant counts:

```math
\\begin{align}
H_{ij}^{+} &= \\sum_{k=1}^{T} w_{ij,k} \\cdot d_k \\cdot \\mathbf{1}[\\text{concordant}]\\,, \\\\
H_{ij}^{-} &= \\sum_{k=1}^{T} w_{ij,k} \\cdot d_k \\cdot \\mathbf{1}[\\text{discordant}]\\,.
\\end{align}
```

Where:

  - ``H_{ij}^{+}``, ``H_{ij}^{-}``: Weighted concordant and discordant co-movement accumulators.
  - $(math_dict[:T])
  - ``w_{ij,k}``: Region weight from the IQ template for observation ``k``.
  - ``d_k = \\exp[-y \\max(0, T-k-e)]``: Temporal decay at observation ``k``.

GerberIQ correlation:

```math
\\begin{align}
\\rho_{ij} &= \\begin{cases}
(H_{ij}^{+} - H_{ij}^{-}) / (H_{ij}^{+} + H_{ij}^{-}) & \\text{Gerber0} \\\\
(H_{ij}^{+} - H_{ij}^{-}) / (H_{ij}^{+} + H_{ij}^{-} + H_{ij}^{0}) & \\text{Gerber1} \\\\
\\rho^{\\mathrm{G0}}_{ij} / \\sqrt{\\rho^{\\mathrm{G0}}_{ii}\\,\\rho^{\\mathrm{G0}}_{jj}} & \\text{Gerber2}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\rho_{ij}``: GerberIQ correlation between assets ``i`` and ``j``.
  - ``H_{ij}^{+}``, ``H_{ij}^{-}``: Weighted concordant and discordant accumulators.
  - ``H_{ij}^{0}``: Weighted neutral (neither concordant nor discordant) accumulator (Gerber1 only).
  - ``\\rho^{\\mathrm{G0}}_{ij}``: Gerber0 correlation (used for Gerber2 normalisation).

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Details

 1. Calls [`regenerate_decay`](@ref).
 2. For each pair of assets `(i, j)`, iterate over all observations.
 3. For every pair at each observation computes the scaling factor for each asset with [`gerber_iq_scaling`](@ref), as well as counters for concordant `pos`, discordant `neg`, and---for [`Gerber1`](@ref)---neutral `nn` counters are initialised to zero.
 4. If the absolute value of the returns of both assets is less than its respective scaled threshold `ce.c`, the observation is skipped.
 5. If the absolute return of both assets is greater than its respective scaled threshold `ce.c`.
    a. If the movement is concordant (both returns have the same sign), the concordant counter is incremented according to [`gerber_IQ_delta`](@ref).
    b. If the movement is discordant (both returns have different signs), the discordant counter is incremented according to [`gerber_IQ_delta`](@ref).
    c. For [`Gerber1`](@ref), if the neither of the previous conditions are met, it means only one of the absolute returns is greater than its respective scaled `ce.c`, so the neutral counter is incremented.
 6. For each [`GerberCovarianceAlgorithm`](@ref), the GerberIQ statistics is computed as follows:
    a. [`Gerber0`](@ref): `(pos - neg) / (pos + neg)`
    b. [`Gerber1`](@ref): `(pos - neg) / (pos + neg + nn)`
    c. [`Gerber2`](@ref): The numerator is computes as when using [`Gerber0`](@ref), but the resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`gerber_IQ_delta`](@ref)
  - [`regenerate_decay`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function gerber_IQ(ce::GerberIQCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:Gerber0}, X::MatNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, decay, sc, kind, ex) = ce
    decay = regenerate_decay(decay, X)
    let decay = decay
        FLoops.@floop ex for j in axes(X, 2)
            sdj = sd[j]
            for i in 1:j
                sdi = sd[i]
                sci, scj = gerber_iq_scaling(sc, sdi, sdj)
                ci, cj = sci * c, scj * c
                neg = zero(eltype(X))
                pos = zero(eltype(X))
                for k in 1:T
                    xi = X[k, i]
                    xj = X[k, j]
                    axi = abs(xi)
                    axj = abs(xj)
                    if axi < ci && axj < cj
                        continue
                    end
                    if axi >= ci && axj >= cj && xi * xj > zero(xi)
                        pos += gerber_IQ_delta(xi, xj, axi, axj, decay, T, k, sci, scj,
                                               kind)
                    elseif axi >= ci && axj >= cj && xi * xj < zero(xi)
                        neg += gerber_IQ_delta(xi, xj, axi, axj, decay, T, k, sci, scj,
                                               kind)
                    end
                end
                den = (pos + neg)
                rho[j, i] = rho[i, j] = if !iszero(den)
                    (pos - neg) / den
                else
                    zero(eltype(X))
                end
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function gerber_IQ(ce::GerberIQCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:Gerber1}, X::MatNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, decay, sc, kind, ex) = ce
    decay = regenerate_decay(decay, X)
    let decay = decay
        FLoops.@floop ex for j in axes(X, 2)
            sdj = sd[j]
            for i in 1:j
                sdi = sd[i]
                sci, scj = gerber_iq_scaling(sc, sdi, sdj)
                ci, cj = sci * c, scj * c
                neg = zero(eltype(X))
                pos = zero(eltype(X))
                nn = zero(eltype(X))
                for k in 1:T
                    xi = X[k, i]
                    xj = X[k, j]
                    axi = abs(xi)
                    axj = abs(xj)
                    if axi < ci && axj < cj
                        continue
                    end
                    if axi >= ci && axj >= cj && xi * xj > zero(xi)
                        pos += gerber_IQ_delta(xi, xj, axi, axj, decay, T, k, sci, scj,
                                               kind)
                    elseif axi >= ci && axj >= cj && xi * xj < zero(xi)
                        neg += gerber_IQ_delta(xi, xj, axi, axj, decay, T, k, sci, scj,
                                               kind)
                    else
                        nn += gerber_IQ_delta(xi, xj, axi, axj, decay, T, k, sci, scj, kind)
                    end
                end
                den = (pos + neg + nn)
                rho[j, i] = rho[i, j] = if !iszero(den)
                    (pos - neg) / den
                else
                    zero(eltype(X))
                end
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function gerber_IQ(ce::GerberIQCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:Gerber2}, X::MatNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, decay, sc, kind, ex) = ce
    decay = regenerate_decay(decay, X)
    let decay = decay
        FLoops.@floop ex for j in axes(X, 2)
            sdj = sd[j]
            for i in 1:j
                sdi = sd[i]
                sci, scj = gerber_iq_scaling(sc, sdi, sdj)
                ci, cj = sci * c, scj * c
                neg = zero(eltype(X))
                pos = zero(eltype(X))
                for k in 1:T
                    xi = X[k, i]
                    xj = X[k, j]
                    axi = abs(xi)
                    axj = abs(xj)
                    if axi < ci && axj < cj
                        continue
                    end
                    if axi >= ci && axj >= cj && xi * xj > zero(xi)
                        pos += gerber_IQ_delta(xi, xj, axi, axj, decay, T, k, sci, scj,
                                               kind)
                    elseif axi >= ci && axj >= cj && xi * xj < zero(xi)
                        neg += gerber_IQ_delta(xi, xj, axi, axj, decay, T, k, sci, scj,
                                               kind)
                    end
                end
                rho[j, i] = rho[i, j] = (pos - neg)
            end
        end
    end
    h = max.(sqrt.(LinearAlgebra.diag(rho)), sqrt(eps(eltype(rho))))
    rho .= LinearAlgebra.Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    Statistics.cor(
        ce::GerberIQCovariance,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the Gerber IQ correlation matrix.

This method computes the Gerber IQ correlation matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Gerber IQ correlation is then computed via [`gerber_IQ`](@ref).

# Arguments

  - `ce`: Gerber IQ covariance estimator.

      + `ce::GerberIQCovariance`: Compute the unstandardised Gerber IQ correlation matrix.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `rho::MatNum`: The Gerber IQ correlation matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function Statistics.cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    return gerber_IQ(ce, X, sd)
end
"""
    Statistics.cov(
        ce::GerberIQCovariance,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the Gerber IQ covariance matrix.

This method computes the Gerber IQ covariance matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Gerber IQ correlation is then computed via [`gerber_IQ`](@ref).

# Arguments

  - `ce`: Gerber IQ covariance estimator.

      + `ce::GerberIQCovariance`: Compute the Gerber IQ covariance matrix.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `sigma::MatNum`: The Gerber IQ covariance matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - [gerber2025squeezing](@cite) Gerber, Sander and Smyth, William and Markowitz, Harry and Miao, Yinsen and Ernst, Philip and Sargen, Paul, *Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation* (December 01, 2025). Available at SSRN: https://ssrn.com/abstract=4986939 or http://dx.doi.org/10.2139/ssrn.4986939
"""
function Statistics.cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    sigma = gerber_IQ(ce, X, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export AssetVolatilityGerberIQScaler, BasicGerberIQ, PartialGerberIQ, FullGerberIQ,
       ExpGerberIQDecay, GerberIQCovariance
