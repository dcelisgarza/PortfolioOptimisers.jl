"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber Information Quality covariance estimators.

All concrete and/or abstract types implementing Gerber Information Quality covariance estimation algorithms should be subtypes of `BaseGerberIQCovariance`.

The family extends [`BaseGerberCovariance`](@ref) in two directions. It weights a co-movement by the region of the return plane it falls in, instead of counting it, and it discounts a co-movement by its age. [`GerberCovarianceAlgorithm`](@ref) states the co-movement statistic these estimators reduce to, and this file does not restate it.

# Interfaces

If moving away from the already established Gerber Information Quality covariance algorithms, you must follow [`AbstractCovarianceEstimator`](@ref) to implement the entire chain.

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
abstract type BaseGerberIQCovariance <: BaseGerberCovariance end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber Information Quality covariance estimation algorithms.

All concrete and/or abstract types implementing Gerber Information Quality covariance estimation algorithms should be subtypes of `GerberIQCovarianceAlgorithm`.

A subtype is a **squeezing template**. It cuts the plane of the two assets' returns into channels, and it names the weight that a co-movement in each channel carries. The library ships the source's three templates, ordered by how many channels they separate: [`BasicGerberIQ`](@ref) with one boundary and one weight, [`PartialGerberIQ`](@ref) with four boundaries and ten weights, and [`FullGerberIQ`](@ref) with four boundaries and twenty-one weights over thirty-six channels.

# Interfaces

A subtype must implement [`gerber_iq_weight`](@ref), which returns the weight of one co-movement, and [`gerber_iq_assert_c_d`](@ref), which checks the noise threshold `c` against the template's boundaries. It may implement [`clamp_gerber_iq_n`](@ref); the fall-through leaves the template unchanged.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`BasicGerberIQ`](@ref)
  - [`PartialGerberIQ`](@ref)
  - [`FullGerberIQ`](@ref)
  - [`gerber_iq_weight`](@ref)
  - [`gerber_iq_assert_c_d`](@ref)
  - [`clamp_gerber_iq_n`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
abstract type GerberIQCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op for Gerber Information Quality covariance estimation algorithms that do not need their noise suppression parameters clamped.

This fall-through catches every pairing of a template with a [`GerberCovarianceAlgorithm`](@ref) that the file does not clamp. Two cases reach it. [`Gerber0`](@ref) and [`Gerber1`](@ref) divide the net weighted vote by a sum of the same weights, so the statistic lies in ``[-1, 1]`` whatever the template holds. [`BasicGerberIQ`](@ref) carries one weight and its square, and neither can exceed the geometric mean of the two diagonal weights that flank it, so it needs no clamp under [`Gerber2`](@ref) either.

# Arguments

  - `kind`: The squeezing template.
  - `args...`: The [`GerberCovarianceAlgorithm`](@ref) marker, ignored here.

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

Abstract supertype for all temporal lookback and delay Gerber Information Quality parameter estimators.

All concrete and/or abstract types implementing Gerber Information Quality parameter estimators should be subtypes of `GerberIQEpsEstimator`.

A subtype computes the **delay** of the source's temporal vector, the number of periods into the past over which a co-movement carries its full weight. The source names it ``\\varepsilon`` and leaves its value to expert judgement or to an outer optimisation. The library adds a default formula; [`gerber_iq_eps`](@ref) states it.

# Interfaces

A subtype must implement `PortfolioOptimisers.gerber_iq_eps(e::MySubtype, X::MatNum) -> Number`. A subtype that implements no method takes the fall-through of [`gerber_iq_eps`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQEps`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
abstract type GerberIQEpsEstimator <: AbstractEstimator end
"""
    const GerberIQEps = Union{<:Number, <:Function, <:GerberIQEpsEstimator}

A type alias for the union of `Number`, `Function`, and `GerberIQEpsEstimator` used for Gerber Information Quality lookback and delay parameter definitions.

The three arms are the three ways to supply the delay. A `Number` is the delay itself. A `Function` computes it from the returns matrix. A [`GerberIQEpsEstimator`](@ref) computes it through [`gerber_iq_eps`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQEpsEstimator`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
const GerberIQEps = Union{<:Number, <:Function, <:GerberIQEpsEstimator}
"""
    gerber_iq_eps(e::Number, ::MatNum) -> Number
    gerber_iq_eps(e::Function, X::MatNum) -> Number
    gerber_iq_eps(e::Option{<:GerberIQEpsEstimator}, X::MatNum) -> Number

Computes or returns the Gerber Information Quality delay parameter `e`, potentially using `X` as an input.

# Mathematical definition

The fall-through sets the delay from the shape of the returns matrix alone.

```math
\\begin{align}
\\varepsilon &= \\mathrm{round}\\left(T - \\frac{T}{N}\\right)\\,.
\\end{align}
```

Where:

  - ``\\varepsilon``: Delay. A co-movement of age ``T - k`` carries its full weight when ``T - k \\leq \\varepsilon``.
  - $(math_dict[:T])
  - $(math_dict[:N])

An observation is therefore discounted only when its index satisfies ``k < T - \\varepsilon``, so this default discounts about the oldest ``T/N`` observations and leaves the rest at full weight. **The formula is the library's, not the source's.** The source states only that the delay is a whole number of periods no larger than the lookback duration, and leaves its value to expert judgement or to an outer optimisation.

# Algorithm

 1. Return the number unchanged when `e` is a `Number`.
 2. Call `e(X)` when `e` is a `Function`, and return its result.
 3. Otherwise read `T` and `N` from `size(X)`, and return `round(Int, T - T / N)`.

# Arguments

  - `e`: The delay parameter estimator, function or value for use in the decay equation.
      + `::Number`: Use the number as-is.
      + `::Function`: A function which takes the data matrix `X` as an argument and returns a `Number`.
      + `::Option{<:GerberIQEpsEstimator}`: Fallback returning `round(Int, T - T / N)`, where `T` and `N` are the number of rows and columns of `X` respectively.
  - $(arg_dict[:X])

# Returns

  - `e::Number`: The delay parameter for use in the decay equation. Observations no older than `e` periods are not discounted. This is not the source's window-duration truncation ``\\tau``, which the estimator does not expose; the estimator always reads the whole matrix and lets the decay discount its oldest rows.

# Related

  - [`GerberIQEps`](@ref)
  - [`GerberIQEpsEstimator`](@ref)
  - [`gerber_iq_gamma`](@ref)
  - [`ExpGerberIQDecay`](@ref)
  - [`regenerate_decay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
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

A subtype computes the **decay rate** of the source's temporal vector, the discount rate applied to a co-movement older than the delay. The source names it ``\\gamma``, requires it to be positive, and leaves its value to expert judgement or to an outer optimisation. The library adds a default formula; [`gerber_iq_gamma`](@ref) states it.

# Interfaces

A subtype must implement `PortfolioOptimisers.gerber_iq_gamma(y::MySubtype, X::MatNum) -> Number`. A subtype that implements no method takes the fall-through of [`gerber_iq_gamma`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQGamma`](@ref)
  - [`gerber_iq_gamma`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
abstract type GerberIQGammaEstimator <: AbstractEstimator end
"""
    const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}

A type alias for the union of `Number`, `Function`, and `GerberIQGammaEstimator` used for Gerber Information Quality temporal decay parameter definitions.

The three arms are the three ways to supply the decay rate. A `Number` is the rate itself. A `Function` computes it from the returns matrix. A [`GerberIQGammaEstimator`](@ref) computes it through [`gerber_iq_gamma`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQGammaEstimator`](@ref)
  - [`gerber_iq_gamma`](@ref)
  - [`ExpGerberIQDecay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}
"""
    gerber_iq_gamma(y::Number, ::MatNum) -> Number
    gerber_iq_gamma(y::Function, X::MatNum) -> Number
    gerber_iq_gamma(y::Option{<:GerberIQGammaEstimator}, X::MatNum) -> Number

Computes or returns the Gerber Information Quality decay strength parameter `y`, potentially using `X` as an input.

# Mathematical definition

The fall-through sets the decay rate from the number of assets alone.

```math
\\begin{align}
\\gamma &= \\frac{\\ln 2}{N}\\,.
\\end{align}
```

Where:

  - ``\\gamma``: Decay rate. A larger value discounts an old co-movement harder.
  - $(math_dict[:N])

This is a half-life of ``N`` periods: beyond the delay, the weight of a co-movement halves every ``N`` observations, because ``\\exp(-\\gamma N) = 1/2``. **The formula is the library's, not the source's.** The source requires only ``\\gamma > 0`` and leaves its value to expert judgement or to an outer optimisation.

# Algorithm

 1. Return the number unchanged when `y` is a `Number`.
 2. Call `y(X)` when `y` is a `Function`, and return its result.
 3. Otherwise return `log(2) / size(X, 2)`.

# Arguments

  - `y`: The decay strength parameter estimator, function or value for use in the decay equation.
      + `::Number`: Use the number as-is.
      + `::Function`: A function which takes the data matrix `X` as an argument and returns a `Number`.
      + `::Option{<:GerberIQGammaEstimator}`: Fallback returning `log(2) / size(X, 2)`.
  - $(arg_dict[:X])

# Returns

  - `gamma::Number`: The decay strength parameter for use in the decay equation.

# Related

  - [`GerberIQGamma`](@ref)
  - [`GerberIQGammaEstimator`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`ExpGerberIQDecay`](@ref)
  - [`regenerate_decay`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
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

A subtype answers one question for a pair of assets: in whose units are the pair's thresholds measured. The source names four answers — each asset's own volatility, the mean of the two, the greater of the two, and the lesser of the two. The library ships the first as [`AssetVolatilityGerberIQScaler`](@ref) and the second as the fall-through of [`gerber_iq_scaling`](@ref). The other two are reached through the `Function` arm of [`GerberIQScaler`](@ref).

# Interfaces

A subtype must implement `PortfolioOptimisers.gerber_iq_scaling(sca::MySubtype, sdi::Number, sdj::Number) -> (Number, Number)`. A subtype that implements no method takes the fall-through of [`gerber_iq_scaling`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQScaler`](@ref)
  - [`AssetVolatilityGerberIQScaler`](@ref)
  - [`gerber_iq_scaling`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
abstract type GerberIQScalerEstimator <: AbstractEstimator end
"""
    const GerberIQScaler = Union{Function, <:GerberIQScalerEstimator}

A type alias for the union of `Function`, and `GerberIQScalerEstimator` used for scaling the threshold parameters for defining significant co-movements in Gerber Information Quality.

The alias has no `Number` arm, because a scaler is a rule over the pair's two standard deviations and not a value. A `Function` takes `sdi` and `sdj` and returns the two scaled values. A [`GerberIQScalerEstimator`](@ref) does the same through [`gerber_iq_scaling`](@ref).

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQScalerEstimator`](@ref)
  - [`AssetVolatilityGerberIQScaler`](@ref)
  - [`gerber_iq_scaling`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
const GerberIQScaler = Union{Function, <:GerberIQScalerEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Scales the threshold parameters using the individual asset volatilities.

Each asset keeps its own units, so asset ``i`` is measured against ``\\sigma_i`` and asset ``j`` against ``\\sigma_j``. This is the convention of Kendall's tau and of the Gerber statistic, and it is the choice under which the Gerber IQ statistic reduces to the Gerber statistic; [`gerber_IQ`](@ref) states that reduction. The fall-through of [`gerber_iq_scaling`](@ref) makes the other choice and gives both assets the mean of the two volatilities.

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

  - $(ref_dict[:gerber2025squeezing])
"""
struct AssetVolatilityGerberIQScaler <: GerberIQScalerEstimator end
"""
    gerber_iq_scaling(sca::AssetVolatilityGerberIQScaler, sdi::Number, sdj::Number) -> (Number, Number)
    gerber_iq_scaling(sca::Function, sdi::Number, sdj::Number) -> (Number, Number)
    gerber_iq_scaling(sca::Option{<:GerberIQScalerEstimator}, sdi::Number, sdj::Number) -> (Number, Number)

Computes or returns the threshold scaling parameters for defining significant co-movements in Gerber Information Quality.

Every threshold of the pair — the noise threshold `c` and each boundary of the squeezing template — is multiplied by the value this function returns for its own axis. So the scaler fixes the units in which a co-movement is judged large.

A scaler is **pair-separable** when its first component reads `sdi` alone, so that an asset's thresholds are the same whatever partner it is measured against. [`AssetVolatilityGerberIQScaler`](@ref) is pair-separable. The fall-through is not, because the pair mean moves with `sdj`, and a `Function` need not be. [`Gerber2`](@ref) is bounded by one only under a pair-separable scaler: it divides by the geometric mean of a diagonal built at the pair `(i, i)`, and a scaler that moves an asset's class off the diagonal breaks that comparison. [#500](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/500) records the defect and a four-row reproduction.

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

The fall-through gives both assets the same units, so a volatile asset and a quiet one are held to the same absolute threshold. [`AssetVolatilityGerberIQScaler`](@ref) gives each asset its own units instead. Both are options the source names.

# Algorithm

 1. Return `(sdi, sdj)` unchanged for [`AssetVolatilityGerberIQScaler`](@ref).
 2. Call `sca(sdi, sdj)` for a `Function`, and return its result.
 3. Otherwise compute `(sdi + sdj) / 2` once, and return it for both axes.

# Arguments

  - `sca`: The scaling estimator to use.

      + `::AssetVolatilityGerberIQScaler`: Returns the input `sdi` and `sdj` as-is. This lets each asset scale according to its own volatility.
      + `::Option{<:GerberIQScalerEstimator}`: Fallback returning the mean of `sdi` and `sdj` twice so each asset is scaled according to the mean of the two asset volatilities. Overloading this with a custom [`GerberIQScalerEstimator`](@ref) allows for custom scaling behavior.
      + `::Function`: Custom scaling function that takes `sdi` and `sdj` as arguments and returns the scaled values.

  - `sdi`: Standard deviation of asset `i`.

  - `sdj`: Standard deviation of asset `j`.

# Returns

  - `scai::Number`: The scaled value for `sdi`.
  - `scaj::Number`: The scaled value for `sdj`.

# Related

  - [`GerberIQScaler`](@ref)
  - [`GerberIQScalerEstimator`](@ref)
  - [`AssetVolatilityGerberIQScaler`](@ref)
  - [`GerberIQKernel`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
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

Abstract supertype for the Gerber IQ estimators that discount an observation by its age.

All concrete and/or abstract types implementing temporal decay for Gerber Information Quality parameter estimators should be subtypes of `GerberIQDecayEstimator`.

A subtype is a **non-increasing function of age**. The source states that the age penalty may be any non-increasing function, and gives the exponential form as its own choice; [`ExpGerberIQDecay`](@ref) implements that form. A subtype is configuration, so it holds its parameters and never a Result. Two methods make it usable: [`regenerate_decay`](@ref) fills any parameter the caller left unset, and the functor returns the weight of one observation.

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

  - $(ref_dict[:gerber2025squeezing])
"""
abstract type GerberIQDecayEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Exponential Gerber IQ temporal decay.

This is the source's own age penalty. A co-movement no older than `e` periods carries its full weight, and one older than that is discounted at the rate `y` for every further period. Either field may be left as `nothing`; [`regenerate_decay`](@ref) then fills it from the returns matrix before the statistic runs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpGerberIQDecay(e::Option{<:GerberIQEps} = nothing,
                     y::Option{<:GerberIQGamma} = nothing)

Keywords correspond to the struct's fields.

## Validation

  - `e` is validated via [`assert_nonempty_nonneg_finite_val`](@ref) when it is a `Number`.
  - `y` is validated via [`assert_nonempty_nonneg_finite_val`](@ref) when it is a `Number`.

A field that is not a `Number` carries no check here, because it is a rule and not a value. [`regenerate_decay`](@ref) resolves it, and the resolved instance is validated by this same constructor.

# Functors

    (decay::ExpGerberIQDecay)(T::Number, k::Number) -> Number

Implements the exponential decay for Gerber IQ covariance.

!!! warning

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

  - $(ref_dict[:gerber2025squeezing])
"""
@concrete struct ExpGerberIQDecay <: GerberIQDecayEstimator
    """
    Delay. A co-movement no older than `e` periods carries its full weight. The source names it ``\\varepsilon``.
    """
    e
    """
    Decay rate. It discounts a co-movement for every period of age beyond `e`. The source names it ``\\gamma``.
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

  - $(ref_dict[:gerber2025squeezing])
"""
function (decay::ExpGerberIQDecay)(T::Number, k::Number)
    return exp(-decay.y * max(0, T - k - decay.e))
end
"""
    regenerate_decay(decay::ExpGerberIQDecay, X::AbstractMatrix) -> ExpGerberIQDecay
    regenerate_decay(decay::GerberIQDecayEstimator, X::AbstractMatrix) -> ExpGerberIQDecay

Automatically sets the decay parameters based on the input data `X`.

**The function allocates; it never writes into its argument.** A decay estimator is configuration and is treated as immutable, so a regenerated parameter arrives in a **new** [`ExpGerberIQDecay`](@ref). The one case that allocates nothing is an [`ExpGerberIQDecay`](@ref) whose two fields are already numbers: there is nothing to resolve, and the same object is returned. The caller must therefore use the returned value; discarding it leaves the unresolved estimator in place.

# Algorithm

 1. Return `decay` unchanged when it is an [`ExpGerberIQDecay`](@ref) whose `e` and `y` are both a `Number`.
 2. Otherwise resolve the delay `e` with [`gerber_iq_eps`](@ref), reading the estimator's own `e` for an [`ExpGerberIQDecay`](@ref) and `nothing` for any other subtype.
 3. Resolve the decay rate `y` the same way with [`gerber_iq_gamma`](@ref).
 4. Return a new [`ExpGerberIQDecay`](@ref) built from `e` and `y`.

Step 2 is why the fall-through discards the subtype: it has no field the two resolvers can read, so it resolves both from `X` alone and returns an [`ExpGerberIQDecay`](@ref). A subtype that must keep its own form implements this function, as the `# Interfaces` section of [`GerberIQDecayEstimator`](@ref) states.

# Arguments

  - `decay`: The decay estimator to regenerate.
      + `::ExpGerberIQDecay`: If both parameters are numeric, returns the input, otherwise returns a new [`ExpGerberIQDecay`](@ref) with the regenerated parameters.
      + `::GerberIQDecayEstimator`: Fallback for automatically setting the decay parameters `e`, and `y` based on the input data `X`, using [`gerber_iq_eps`](@ref) and [`gerber_iq_gamma`](@ref) with `nothing` as the first input. Custom subtypes of [`GerberIQDecayEstimator`](@ref) should implement this method, else they default to the fallback.
  - $(arg_dict[:X])

# Returns

  - `decay::ExpGerberIQDecay`: With parameters based on `X`.

# Related

  - [`GerberIQDecayEstimator`](@ref)
  - [`ExpGerberIQDecay`](@ref)
  - [`GerberIQCovariance`](@ref)
  - [`gerber_iq_eps`](@ref)
  - [`gerber_iq_gamma`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
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

This is the source's own reduced template: one boundary and one weight for the whole plane. It is the template the source's results are built on, and it is this estimator's default.

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
  - ``n``: Compression weight.

The three cases are the source's tail, body and wing. Both returns are large in the tail, both are small in the body, and one of each in the wing. Because ``0 \\leq n \\leq 1`` the weights obey ``n^2 \\leq n \\leq 1``, so a co-movement of two similar magnitudes counts for more than one of two dissimilar magnitudes. That ordering is the source's own judgement, and it is what the single weight `n` buys in exchange for its lost degrees of freedom.

The body case is reachable only when ``c < d``. A co-movement with both returns inside the noise threshold never reaches the template, so at ``c = d`` the band that carries `n` is empty and only ``1`` and ``n^2`` occur.

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

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BasicGerberIQ(; d::Number = 2.0, n::Number = 0.5)

Keywords correspond to the struct's fields.

## Validation

  - `d` is validated via [`assert_nonempty_nonneg_finite_val`](@ref).
  - `0 <= n <= 1`.
  - `c <= d` is checked by [`gerber_iq_assert_c_d`](@ref) when the template reaches a [`GerberIQCovariance`](@ref), not here.

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

  - $(ref_dict[:gerber2025squeezing])
"""
@concrete struct BasicGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Significance threshold. A return at or beyond `d` scaled units from zero is large. It is measured in the same units as the noise threshold `c` of [`GerberIQCovariance`](@ref), and must be at least as large as it.
    """
    d
    """
    Compression weight, in `[0, 1]`. A co-movement of two large returns keeps its full weight of one, a co-movement of two small returns keeps `n`, and a co-movement of one of each keeps `n^2`.
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

The two parameters cut the same axis and are measured in the same scaled units. `c` closes the noise zone from above and `d` opens the significant zone from below, so the body of the template is the band `c <= |x| < d`. A `d` below `c` inverts that band, and the template then has no body at all: every co-movement that survives the noise zone is already beyond `d`, and the weight `n` can never be selected.

# Arguments

  - `c`: Small movement threshold.
  - `kind`: [`BasicGerberIQ`](@ref) instance.

# Validation

  - `c <= kind.d`, else a `DomainError` naming both values.

# Returns

  - `nothing`. The function is called for its raise alone.

# Related

  - [`BasicGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function gerber_iq_assert_c_d(c::Number, kind::BasicGerberIQ)
    @argcheck(c <= kind.d,
              DomainError("c must be <= kind.d, got c = $c, kind.d = $(kind.d)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weight for a co-movement according to the region it falls into from the [`BasicGerberIQ`](@ref) template.

[`BasicGerberIQ`](@ref) states the closed form this method selects from. The signed returns are unused, because the template is symmetric about both axes and reads magnitudes alone.

# Algorithm

 1. Scale the significance threshold onto each axis: `di = d * sci` and `dj = d * scj`.
 2. Return `one(n)` when both absolute returns reach their scaled threshold.
 3. Return `n` when neither does.
 4. Return `n^2` otherwise, which is the case of exactly one absolute return reaching its threshold.

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

  - $(ref_dict[:gerber2025squeezing])
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

Gerber Information Quality template with asymmetric thresholds. Concordant and discordant co-movements take independently configurable significance thresholds.

This is the source's sixteen-channel template, the middle rung of the family. It gives the boundary vector its four independent components and allows ten distinct weights, where [`BasicGerberIQ`](@ref) collapses both to one scalar and [`FullGerberIQ`](@ref) opens the plane to thirty-six channels.

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

The four boundaries cut the plane into sixteen channels, and `n1` to `n10` are the distinct weights those channels take once the plane's symmetry about the line ``x_i = x_j`` is imposed. The template reads the **signed** return, not its magnitude, so it separates a pair by sign as well as by size: a concordant pair is measured against `dcp` when both returns are positive and against `dcn` when both are negative, and a discordant pair is measured against `ddp` and `ddn`. A co-movement in no named channel carries weight zero.

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

The default of every mixed-magnitude weight is the geometric mean of the two same-magnitude weights that flank it, which is the bound [`clamp_gerber_iq_n`](@ref) enforces under [`Gerber2`](@ref).

## Validation

  - All `d**` parameters are validated via [`assert_nonempty_nonneg_finite_val`](@ref).
  - All `n**` parameters must be `0 <= n** <= 1`.
  - `c <= dcp`, `c <= dcn`, `c <= ddp` and `c <= ddn` are checked by [`gerber_iq_assert_c_d`](@ref) when the template reaches a [`GerberIQCovariance`](@ref), not here.

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

  - $(ref_dict[:gerber2025squeezing])
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

Lowers the mixed-magnitude weights of a [`PartialGerberIQ`](@ref) template so that the [`Gerber2`](@ref) statistic stays inside `[-1, 1]`. It does not make the matrix positive definite; that is `pdm`'s work.

Under [`Gerber2`](@ref) the pairwise entry is the raw `pos - neg`, and the matrix is afterwards divided by the geometric mean of its own diagonal. A pair `(i, i)` compares an asset with itself, so its diagonal entry is built from the same-magnitude weights alone. A mixed-magnitude weight above the geometric mean of the two same-magnitude weights that flank it therefore lets the ratio leave `[-1, 1]`. With `n1 = n4 = 0.2` an unclamped `n7 = 1.0` returns `5.0`, and the clamp brings it to `1.0`.

!!! warning

    The clamp covers `n7` and `n8` only, so it is a necessary condition and not a sufficient one. A hand-tuned template can still leave `[-1, 1]` through a weight this method does not touch. [#494](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/494) records the gap and the reproduction. The shipped defaults sit exactly **on** the bound, so they meet it and carry no margin. A pair-dependent `sc` breaks the same bound on its own, whatever the template does; [#500](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/500) records that half.

# Mathematical definition

```math
\\begin{align}
n_{7} &\\leftarrow \\min\\left(n_{7},\\, \\sqrt{n_{1} n_{4}}\\right)\\,, \\\\
n_{8} &\\leftarrow \\min\\left(n_{8},\\, \\sqrt{n_{2} n_{5}}\\right)\\,.
\\end{align}
```

Where:

  - ``n_{1}``, ``n_{4}``: Small and large positive concordant weights, the two that flank ``n_{7}``.
  - ``n_{2}``, ``n_{5}``: Small and large negative concordant weights, the two that flank ``n_{8}``.

# Algorithm

 1. Lower `n7` to `min(n7, sqrt(n1 * n4))`.
 2. Lower `n8` to `min(n8, sqrt(n2 * n5))`.
 3. Return a new [`PartialGerberIQ`](@ref) carrying the two lowered weights and every other field unchanged.

# Arguments

  - `alg`: Instance of [`PartialGerberIQ`](@ref).
  - `::Gerber2`: Instance of [`Gerber2`](@ref).

# Returns

  - `kind::PartialGerberIQ`: A new template. The method allocates and never writes into `alg`.

# Related

  - [`PartialGerberIQ`](@ref)
  - [`Gerber2`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
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

[`PartialGerberIQ`](@ref) states the channel map this method selects from. The absolute returns are unused, because this template reads the sign of each return as well as its size.

# Algorithm

 1. Scale each of the four boundaries onto each axis, giving the eight thresholds `dcpi`, `dcni`, `ddpi`, `ddni` and their `j` counterparts.
 2. Test the positive concordant quadrant in order of size: `n4` when both returns reach `dcp`, `n7` when one does and the other is positive but smaller, and `n1` when both are positive and smaller.
 3. Test the negative concordant quadrant the same way against `dcn`, giving `n5`, `n8` and `n2`.
 4. Test the two discordant quadrants against `ddp` and `ddn`, giving `n6` when both returns are beyond their boundary, `n9` and `n10` when one is and the other is not, and `n3` when neither is.
 5. Return `zero(xi)` when no channel matched, which happens when a return is exactly zero.

The tests are ordered from the largest channel inwards, so the first match wins and no co-movement is counted twice.

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

  - $(ref_dict[:gerber2025squeezing])
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

Gerber Information Quality template with fine-grained asymmetric thresholds. Classifies co-movements into two positive and two negative magnitude classes.

This is the source's full template, the most general form the family takes. Two boundaries on each axis cut every return into a small, a moderate and a large class, giving six bands per axis and thirty-six channels in the plane. Symmetry about the line ``x_i = x_j`` folds those thirty-six channels onto **twenty-one** distinct weights, which is the count the source states and the count this type carries.

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

A return of asset ``i`` is **large positive** at or beyond ``dp1_i``, **moderate positive** in ``[dp2_i, dp1_i)``, **small positive** in ``(0, dp2_i)``, and the three negative classes mirror them about zero against ``dn2_i`` and ``dn1_i``. The channel of a co-movement is the pair of classes its two returns fall in, and its weight is the field named for that pair. A co-movement in no named channel carries weight zero, which happens only when a return is exactly zero.

The diagram shows a visual representation of the regions defined by `FullGerberIQ`. In this case `c = 1`, `dp2 = 2`, `dn2 = 2`, `dp1 = 3`, and `dn1 = 3`. In this version, the limits are allowed to cross over the zero line. Thus, the constructor ensures `dp1 >= dp2` and `dn1 >= dn2` by swapping values if necessary to ensure consistency.

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

Every mixed-class weight defaults to the geometric mean of two other weights. The keyword order is not the field order: `n17` precedes `n16` and `n19` precedes `n18`, because each of those defaults reads the other.

!!! warning

    Four of those defaults do **not** meet the bound [`clamp_gerber_iq_n`](@ref) enforces under [`Gerber2`](@ref). `n15`, `n16`, `n18` and `n21` each default to the geometric mean of two **mixed-class** weights rather than of the two **same-class** weights that flank the channel, and each exceeds its bound by a factor of `(n4^2 / (n1 n11))^(1/4)`. A two-asset sample then returns a correlation entry of `1.0299` under [`Gerber2`](@ref). [#494](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/494) records the gap, the whole table of fifteen bounds and the reproduction.

## Validation

  - All `d**` parameters are validated via [`assert_nonempty_nonneg_finite_val`](@ref).
  - All `n**` parameters must be `0 <= n** <= 1`.
  - `c <= dp1`, `c <= dp2`, `c <= dn1` and `c <= dn2` are checked by [`gerber_iq_assert_c_d`](@ref) when the template reaches a [`GerberIQCovariance`](@ref), not here.

The boundary swap is not a raise. `dp1` and `dp2` are ordered by `extrema` and so are `dn1` and `dn2`, so a caller who names them the other way round gets a working template rather than an error.

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

  - $(ref_dict[:gerber2025squeezing])
"""
@concrete struct FullGerberIQ <: GerberIQCovarianceAlgorithm
    """
    Outer positive boundary. A positive return at or beyond it is large. The constructor swaps `dp1` and `dp2` when needed, so `dp1 >= dp2` always holds.
    """
    dp1
    """
    Inner positive boundary. A positive return between it and `dp1` is moderate, and one below it is small.
    """
    dp2
    """
    Outer negative boundary. A negative return at or beyond `-dn1` is large.
    """
    dn1
    """
    Inner negative boundary. A negative return between `-dn1` and `-dn2` is moderate, and one above `-dn2` is small.
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

`c` and every boundary cut the same axis in the same scaled units, so `c` must sit inside the innermost boundary. A boundary below `c` describes a band that the noise zone has already swallowed, and every weight that names that band becomes unselectable. The check runs once per boundary, and the raise names the boundary that failed, so a caller with four boundaries learns which one is wrong.

# Arguments

  - `c`: Small movement threshold.
  - `kind`: Instance of [`PartialGerberIQ`](@ref) or [`FullGerberIQ`](@ref).

# Validation

  - `c <= dcp`, `c <= dcn`, `c <= ddp` and `c <= ddn` for a [`PartialGerberIQ`](@ref), else a `DomainError` naming the failing boundary.
  - `c <= dp1`, `c <= dp2`, `c <= dn1` and `c <= dn2` for a [`FullGerberIQ`](@ref), else a `DomainError` naming the failing boundary.

# Returns

  - `nothing`. The function is called for its raise alone.

# Related

  - [`PartialGerberIQ`](@ref)
  - [`FullGerberIQ`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
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

Lowers the mixed-magnitude weights of a [`FullGerberIQ`](@ref) template so that the [`Gerber2`](@ref) statistic stays inside `[-1, 1]`. It does not make the matrix positive definite; that is `pdm`'s work.

Under [`Gerber2`](@ref) the pairwise entry is the raw `pos - neg`, and the matrix is afterwards divided by the geometric mean of its own diagonal. A pair `(i, i)` compares an asset with itself, so both returns fall in the same magnitude class and the co-movement is always concordant. Exactly six of the twenty-one weights therefore sit on the diagonal, one per class: `n11`, `n4`, `n1`, `n2`, `n5` and `n12`. The other fifteen each join two distinct classes, one weight per unordered pair, **discordant channels included**, and a weight above the geometric mean of the two same-class weights that flank its channel lets the ratio leave `[-1, 1]`.

!!! warning

    The clamp covers `n7`, `n8`, `n14` and `n17` only. Fifteen of the twenty-one weights join two distinct classes and owe the bound, so the clamp is a necessary condition and not a sufficient one. A [`FullGerberIQ`](@ref) with `n1 = n11 = 0.1` and `n15 = 1.0` returns a correlation entry of `10.0`, and the **shipped defaults** break the bound on `n15`, `n16`, `n18` and `n21`. [#494](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/494) records the gap, the whole table of fifteen bounds and the reproduction. A pair-dependent `sc` breaks the same bound on its own, whatever the template does; [#500](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/500) records that half.

# Mathematical definition

```math
\\begin{align}
n_{7} &\\leftarrow \\min\\left(n_{7},\\, \\sqrt{n_{1} n_{4}}\\right)\\,, \\\\
n_{8} &\\leftarrow \\min\\left(n_{8},\\, \\sqrt{n_{2} n_{5}}\\right)\\,, \\\\
n_{14} &\\leftarrow \\min\\left(n_{14},\\, \\sqrt{n_{4} n_{11}}\\right)\\,, \\\\
n_{17} &\\leftarrow \\min\\left(n_{17},\\, \\sqrt{n_{5} n_{12}}\\right)\\,.
\\end{align}
```

Where:

  - ``n_{1}``, ``n_{4}``, ``n_{11}``: Small, moderate and large positive concordant weights.
  - ``n_{2}``, ``n_{5}``, ``n_{12}``: Small, moderate and large negative concordant weights.

Each clamped weight names the channel between two of those classes, and its bound is the geometric mean of the two.

# Algorithm

 1. Lower `n7` to `min(n7, sqrt(n1 * n4))` and `n14` to `min(n14, sqrt(n4 * n11))`, the two mixed positive concordant channels.
 2. Lower `n8` to `min(n8, sqrt(n2 * n5))` and `n17` to `min(n17, sqrt(n5 * n12))`, their negative counterparts.
 3. Return a new [`FullGerberIQ`](@ref) carrying the four lowered weights and every other field unchanged.

# Arguments

  - `alg`: Instance of [`FullGerberIQ`](@ref).
  - `::Gerber2`: Instance of [`Gerber2`](@ref).

# Returns

  - `kind::FullGerberIQ`: A new template. The method allocates and never writes into `alg`.

# Related

  - [`FullGerberIQ`](@ref)
  - [`Gerber2`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
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

[`FullGerberIQ`](@ref) states the channel map this method selects from. The absolute returns are unused, because this template reads the sign of each return as well as its size.

# Algorithm

 1. Scale each of the four boundaries onto each axis, giving the eight thresholds `dp1i`, `dp2i`, `dn1i`, `dn2i` and their `j` counterparts.
 2. Test the six channels whose asset `i` return is large positive, in the order large positive, moderate positive, small positive, small negative, moderate negative and large negative on asset `j`, giving `n11`, `n14`, `n15`, `n18`, `n19` and `n13`.
 3. Test the five remaining channels whose asset `i` return is moderate positive, giving `n4`, `n7`, `n9`, `n6` and `n20`.
 4. Test the four remaining channels whose asset `i` return is small positive, giving `n1`, `n3`, `n10` and `n21`.
 5. Test the three remaining channels whose asset `i` return is small negative, giving `n2`, `n8` and `n16`.
 6. Test the two remaining channels whose asset `i` return is moderate negative, giving `n5` and `n17`.
 7. Return `n12` when both returns are large negative.
 8. Return `zero(xi)` when no channel matched, which happens when a return is exactly zero.

Every test names both orderings of the pair, so the result is symmetric in its two returns. The tests run from the largest class inwards, so the first match wins and no co-movement is counted twice. All twenty-one weights are reachable.

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

  - $(ref_dict[:gerber2025squeezing])
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

Configures and applies Gerber Information Quality covariance estimators.

`GerberIQCovariance` encapsulates all components required for Gerber Information Quality based covariance or correlation estimation.

Four knobs carry the source's own parameters. `c` is the noise threshold, `kind` is the squeezing template that supplies the weight of a co-movement, `sc` fixes the units the thresholds are measured in, and `decay` discounts a co-movement by its age. `alg` is the one knob the source does not carry: its canonical statistic is the [`Gerber1`](@ref) branch alone, and this estimator also offers [`Gerber0`](@ref) and [`Gerber2`](@ref) from the classic Gerber family. [`gerber_IQ`](@ref) states the three branches and the reduction that ties them to that family.

The source's lookback duration ``\\tau`` has no field. The estimator always reads every row of `X` and lets `decay` discount the oldest ones, which is the ``\\tau = T - 1`` case of the source.

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
                         ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx())

Keywords correspond to the struct's fields.

## Validation

  - `c >= 0`: `c` must be non-negative.
  - `c <= kind.d` (or equivalent for the chosen `kind`): via [`gerber_iq_assert_c_d`](@ref).

!!! warning

    The constructor may **replace** `kind`. It passes the template through [`clamp_gerber_iq_n`](@ref), which lowers some weights under [`Gerber2`](@ref), so the stored template is not always the one that was passed in. Read `ce.kind` rather than the argument when the exact weights matter.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ve`: Recursively updated via [`factory`](@ref).
  - `me`: Recursively updated via [`factory`](@ref).
  - `decay`: Recursively updated via [`factory`](@ref).
  - `alg`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `me`: Recursively viewed via [`port_opt_view`](@ref).

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

  - $(ref_dict[:gerber2025squeezing])
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
    Noise threshold. A return within `c` scaled units of zero is noise, and so is a return of exactly zero at any `c`. A co-movement whose two returns are both noise is dropped. It must be no larger than every boundary of `kind`.
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
                                me::AbstractExpectedReturnsEstimator,
                                pdm::Option{<:AbstractPosdefEstimator}, c::Number,
                                decay::GerberIQDecayEstimator, sc::Option{<:GerberIQScaler},
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
                            pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                            c::Number = 0.5,
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

This is the product of the two halves of the source's squeezing statistic: the spatial weight the template gives the co-movement, and the temporal discount its age earns. The result is the quantity that [`comovement_step`](@ref) adds into one of the pair's three accumulators.

# Mathematical definition

```math
\\begin{align}
\\Delta_{t,\\,i,\\,j} &= \\eta_{t,\\,i,\\,j} \\, v_{t}\\,.
\\end{align}
```

Where:

  - ``\\Delta_{t,\\,i,\\,j}``: Contribution of the co-movement of assets ``i`` and ``j`` at observation ``t``.
  - ``\\eta_{t,\\,i,\\,j}``: Squeezing weight of that co-movement, from the template.
  - ``v_{t}``: Temporal discount of observation ``t``, from the decay estimator.

# Algorithm

 1. Compute the squeezing weight `w` with [`gerber_iq_weight`](@ref), passing the pair's two scaling factors and the template.
 2. Compute the temporal discount `p` by calling the `decay` functor with `T` and `k`.
 3. Return `w * p`.

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

  - `rho::Number`: The weighted, discounted contribution of one co-movement. It is not itself a correlation.

# Related

  - [`gerber_iq_weight`](@ref)
  - [`comovement_step`](@ref)
  - [`gerber_IQ`](@ref)
  - [`GerberIQDecayEstimator`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function gerber_IQ_delta(xi::Number, xj::Number, axi::Number, axj::Number,
                         decay::GerberIQDecayEstimator, T::Integer, k::Number, sci::Number,
                         scj::Number, kind::GerberIQCovarianceAlgorithm)
    w = gerber_iq_weight(xi, xj, axi, axj, sci, scj, kind)
    p = decay(T, k)
    return w * p
end
"""
$(DocStringExtensions.TYPEDEF)

Co-movement policy for [`gerber_comovement!`](@ref) implementing the Gerber IQ family.

Observations are thresholded against the pair's scaled thresholds from [`gerber_iq_scaling`](@ref), classified by the sign of the product of returns, and weighted by the IQ noise-compression template and temporal decay via [`gerber_IQ_delta`](@ref). The `alg` marker selects the denominator policy ([`comovement_ratio`](@ref)).

# Fields

  - `alg`: Gerber algorithm marker selecting the denominator policy.
  - `kind`: Gerber IQ noise-compression template.
  - `decay`: Regenerated temporal decay estimator.
  - `sc`: Threshold scaling factor estimator.
  - `c`: Small co-movement threshold.
  - `sd`: Vector of asset standard deviations.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`gerber_IQ`](@ref)
  - [`gerber_comovement!`](@ref)
"""
struct GerberIQKernel{T1 <: GerberCovarianceAlgorithm, T2 <: GerberIQCovarianceAlgorithm,
                      T3 <: GerberIQDecayEstimator, T4, T5 <: Number, T6 <: ArrNum}
    alg::T1
    kind::T2
    decay::T3
    sc::T4
    c::T5
    sd::T6
end
@inline function comovement_pair_state(pol::GerberIQKernel, i::Integer, j::Integer)
    sci, scj = gerber_iq_scaling(pol.sc, pol.sd[i], pol.sd[j])
    return (sci = sci, scj = scj, ci = sci * pol.c, cj = scj * pol.c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate a neutral (one-sided) observation into the Gerber IQ pair accumulator.

Only [`Gerber1`](@ref) tracks neutral co-movements, adding the [`gerber_IQ_delta`](@ref) weight to the neutral score; the fall-through method returns the accumulator unchanged.

A neutral co-movement is one on which exactly one of the two assets left the noise zone, which [`iq_crossed`](@ref) decides. [`Gerber1`](@ref) is the only branch whose denominator counts it, so the other two branches would carry the sum and never read it.

# Arguments

  - `pol`: The [`GerberIQKernel`](@ref) policy.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.
  - `st`: Pair state from [`comovement_pair_state`](@ref).
  - `xi`, `xj`: Returns of assets `i` and `j` at observation `k`.
  - `axi`, `axj`: Their absolute values.
  - `T`: Number of observations.
  - `k`: Observation index.

# Returns

  - The accumulator, with `nn` raised by the [`gerber_IQ_delta`](@ref) contribution under [`Gerber1`](@ref), and unchanged otherwise.

# Related

  - [`comovement_step`](@ref)
  - [`iq_crossed`](@ref)
  - [`gerber_IQ_delta`](@ref)
  - [`GerberIQKernel`](@ref)
  - [`Gerber1`](@ref)
"""
@inline function iq_add_neutral(pol::GerberIQKernel{<:Gerber1}, acc, st, xi::Number,
                                xj::Number, axi::Number, axj::Number, T::Integer,
                                k::Integer)
    return (; acc...,
            nn = acc.nn +
                 gerber_IQ_delta(xi, xj, axi, axj, pol.decay, T, k, st.sci, st.scj,
                                 pol.kind))
end
@inline function iq_add_neutral(::GerberIQKernel, acc, args...)
    return acc
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Decide whether one asset left the noise zone at one observation.

An asset leaves the noise zone when its return reaches the pair's scaled threshold **and** is not exactly zero. The sign test is redundant for a positive threshold, because `ax >= c > 0` already implies that `x` is not zero. It binds only at `c = 0`, where the closed comparison `ax >= 0` holds for every return, including one that is exactly zero. ADR 0090 settled that a return of exactly zero never crosses, and this is that rule for the Gerber IQ family.

The rule is what keeps the diagonal of the statistic at one. The pair `(i, i)` either crosses on both axes or on neither, so it never reaches the neutral accumulator that [`Gerber1`](@ref) divides by. Without the sign test a zero return crosses on both axes but has no sign, so it fell through to that accumulator and pulled the diagonal below one.

# Arguments

  - `x`: Return of the asset at the observation.
  - `ax`: Its absolute value.
  - `c`: The asset's scaled noise threshold, from [`comovement_pair_state`](@ref).

# Returns

  - `crossed::Bool`: `true` when the asset left the noise zone.

# Related

  - [`comovement_step`](@ref)
  - [`comovement_pair_state`](@ref)
  - [`GerberIQKernel`](@ref)
"""
@inline function iq_crossed(x::Number, ax::Number, c::Number)
    return ax >= c && !iszero(x)
end
@inline function comovement_step(pol::GerberIQKernel, acc, st, xi::Number, xj::Number,
                                 T::Integer, k::Integer)
    axi = abs(xi)
    axj = abs(xj)
    crossi = iq_crossed(xi, axi, st.ci)
    crossj = iq_crossed(xj, axj, st.cj)
    if !crossi && !crossj
        return acc
    end
    return if crossi && crossj && xi * xj > zero(xi)
        (; acc...,
         pos = acc.pos +
               gerber_IQ_delta(xi, xj, axi, axj, pol.decay, T, k, st.sci, st.scj, pol.kind))
    elseif crossi && crossj && xi * xj < zero(xi)
        (; acc...,
         neg = acc.neg +
               gerber_IQ_delta(xi, xj, axi, axj, pol.decay, T, k, st.sci, st.scj, pol.kind))
    else
        iq_add_neutral(pol, acc, st, xi, xj, axi, axj, T, k)
    end
end
@inline function comovement_finalise(pol::GerberIQKernel, acc, ::Type{T}) where {T}
    return comovement_ratio(pol.alg, acc.pos, acc.neg, acc.nn, T)
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
h_{ij} / \\sqrt{h_{ii}\\,h_{jj}} & \\text{Gerber2}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\rho_{ij}``: GerberIQ correlation between assets ``i`` and ``j``.
  - ``H_{ij}^{+}``, ``H_{ij}^{-}``: Weighted concordant and discordant accumulators.
  - ``H_{ij}^{0}``: Weighted neutral (neither concordant nor discordant) accumulator (Gerber1 only).
  - ``h_{ij} = H_{ij}^{+} - H_{ij}^{-}``: The **raw** weighted difference, which is what Gerber2 standardises. Gerber2 does **not** normalise the Gerber0 ratio; the two agree only where ``H^{+} + H^{-}`` is constant across pairs.
  - ``\\sqrt{h_{ii}\\,h_{jj}}``: Geometric mean of the diagonal, with the roots clamped below at ``\\sqrt{\\varepsilon}``.

The Gerber1 branch is the source's own statistic. Its numerator runs over the observations on which both assets left the noise zone, and its denominator over those on which at least one did. The Gerber0 and Gerber2 branches are the classic Gerber family's denominators, applied here to the weighted, discounted accumulators; the source states neither.

The Gerber statistic is the special case of this one that switches the squeezing and the decay off. With every weight set to one, ``\\gamma = 0``, the per-asset volatility scaling of [`AssetVolatilityGerberIQScaler`](@ref), and ``c`` equal to a Gerber threshold, all three branches reproduce [`GerberCovariance`](@ref) to the last bit. The reduction holds at ``c = 0`` as it does at every positive threshold, because [`iq_crossed`](@ref) gives this family the rule ADR 0090 gave that one: a return of exactly zero never leaves the noise zone.

Only the Gerber0 and Gerber1 branches are bounded by construction, because each divides by a sum of the same weights it subtracts. The Gerber2 branch is bounded only when **two** conditions hold together: every mixed-class weight of the template meets the geometric-mean bound [`clamp_gerber_iq_n`](@ref) enforces, and `sc` is **pair-separable**, so that an asset's magnitude class does not move when its partner changes. The clamp is a necessary condition and not a sufficient one, and the shipped [`FullGerberIQ`](@ref) defaults break it; [#494](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/494) records that half. [`AssetVolatilityGerberIQScaler`](@ref) is pair-separable and `nothing` is not, because it reads the pair mean; [#500](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/500) records that half.

# Algorithm

 1. Allocate the `N × N` output matrix `rho`.
 2. Resolve the decay estimator against `X` with [`regenerate_decay`](@ref), so its delay and rate are numbers before the loop starts.
 3. Build the [`GerberIQKernel`](@ref) policy from the resolved decay and the estimator's `alg`, `kind`, `sc`, `c` and the standard deviations `sd`.
 4. Fill `rho` with [`gerber_comovement!`](@ref), which walks every pair and every observation and reduces each pair's accumulators. That loop skeleton is shared with the Smyth-Broby family and lives in one place.
 5. Standardise the matrix with [`standardise_comovement!`](@ref). This is a no-op for [`Gerber0`](@ref) and [`Gerber1`](@ref), and divides by the geometric mean of the diagonal for [`Gerber2`](@ref).
 6. Write one onto a zero diagonal entry with [`comovement_unit_diagonal!`](@ref). An asset that never leaves its noise zone reduces to a zero diagonal entry, and that entry is one by definition.
 7. Repair the matrix with [`posdef!`](@ref), because the statistic is not guaranteed to be positive semi-definite. The source records the same and repairs by the nearest correlation matrix.

Steps 4 and 5 are where the three [`GerberCovarianceAlgorithm`](@ref) branches differ, and [`comovement_ratio`](@ref) owns that difference.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQKernel`](@ref)
  - [`gerber_comovement!`](@ref)
  - [`comovement_unit_diagonal!`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`gerber_IQ_delta`](@ref)
  - [`regenerate_decay`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function gerber_IQ(ce::GerberIQCovariance, X::MatNum, sd::ArrNum)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    decay = regenerate_decay(ce.decay, X)
    pol = GerberIQKernel(ce.alg, ce.kind, decay, ce.sc, ce.c, sd)
    gerber_comovement!(rho, ce.ex, X, pol)
    standardise_comovement!(ce.alg, rho)
    comovement_unit_diagonal!(rho)
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

The standard deviations serve two purposes at once. They scale the thresholds through [`gerber_iq_scaling`](@ref), and in [`cov`](@ref) they rescale the correlation into a covariance.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute the per-asset standard deviations with the estimator's `ve`.
 3. Raise every standard deviation to at least `eps(eltype(sd))`, so a constant asset cannot divide by zero.
 4. Centre the returns with the estimator's `me` through [`demean_returns`](@ref).
 5. Return the matrix that [`gerber_IQ`](@ref) builds from the centred returns and those standard deviations.

# Arguments

  - `ce`: Gerber IQ covariance estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::MatNum`: The Gerber IQ correlation matrix. Its diagonal is one for every asset.

!!! note

    An asset that never leaves its own noise zone gets a **zero row**, because no observation votes for any pair it belongs to. Its diagonal entry is one, which [`comovement_unit_diagonal!`](@ref) writes, so the matrix stays a formal correlation matrix and the asset reads as uncorrelated with every other one. That is what the sample says about it. Lower `c` when a short window meets a quiet asset, and the asset votes again. ADR 0093 records the decision, and [#495](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/495) is the defect that led to it.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function Statistics.cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
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

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\boldsymbol{\\rho} \\odot \\left(\\boldsymbol{\\sigma} \\boldsymbol{\\sigma}^{\\intercal}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``\\boldsymbol{\\rho}``: Gerber IQ correlation matrix.
  - ``\\boldsymbol{\\sigma}``: Vector of asset standard deviations.
  - ``\\odot``: Element-wise multiplication.

The covariance is the correlation of [`cor`](@ref) rescaled by the same standard deviations that scaled its thresholds, so its diagonal is exactly ``\\boldsymbol{\\sigma}^2``.

# Algorithm

 1. Run the five steps of [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref), giving the correlation matrix and the standard deviations.
 2. Rescale that matrix in place with `StatsBase.cor2cov!` and those standard deviations, and return it.

# Arguments

  - `ce`: Gerber IQ covariance estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `sigma::MatNum`: The Gerber IQ covariance matrix. Its diagonal is the variance of each asset, because `cor2cov!` scales a unit correlation diagonal by ``\\boldsymbol{\\sigma}^2``.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function Statistics.cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    sigma = gerber_IQ(ce, X, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export AssetVolatilityGerberIQScaler, BasicGerberIQ, PartialGerberIQ, FullGerberIQ,
       ExpGerberIQDecay, GerberIQCovariance
