"""
    abstract type MomentMeasureAlgorithm <: AbstractAlgorithm end

Abstract supertype for all moment-based risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using statistical moments (e.g., mean, variance, skewness, kurtosis) of the return distribution. All concrete moment risk measure algorithms should subtype `MomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
"""
abstract type MomentMeasureAlgorithm <: AbstractAlgorithm end
"""
    abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end

Abstract supertype for all low-order moment-based risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using low-order statistical moments (e.g., mean, variance, mean absolute deviation) of the return distribution. All concrete low-order moment risk measure algorithms should subtype `LowOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`StandardisedLowOrderMoment`](@ref)
"""
abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
    abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end

Abstract supertype for low-order moment risk measure algorithms that are not standardised by the variance in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using low-order statistical moments without normalising by the variance. All concrete unstandardised low-order moment risk measure algorithms should subtype `UnstandardisedLowOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`FirstLowerMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
"""
abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end
function factory(alg::MomentMeasureAlgorithm, args...; kwargs...)
    return alg
end
"""
    struct FirstLowerMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm end

Represents the first lower moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the first lower moment, which is the negative mean of the deviations of the returns series below a target value.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct FirstLowerMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
    struct MeanAbsoluteDeviation <: UnstandardisedLowOrderMomentMeasureAlgorithm end

Represents the mean absolute deviation risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk as the mean of the absolute deviations of the returns series from a target value.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct MeanAbsoluteDeviation <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
    abstract type UnstandardisedSecondMomentAlgorithm <: UnstandardisedLowOrderMomentMeasureAlgorithm end

Abstract supertype for unstandardised second moment risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using the second moment (such as variance or semi-variance) of the return distribution, without normalising by the variance. All concrete unstandardised second moment risk measure algorithms should subtype `UnstandardisedSecondMomentAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`SecondLowerMoment`](@ref)
  - [`SecondCentralMoment`](@ref)
"""
abstract type UnstandardisedSecondMomentAlgorithm <:
              UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
    struct SecondLowerMoment{T1} <: UnstandardisedSecondMomentAlgorithm
        alg::T1
    end

Represents the second lower moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the second lower moment (semi-variance or semi-standard deviation), which quantifies downside risk by considering only deviations below a target value.

# Fields

  - `alg`: The second moment formulation algorithm used to compute the risk measure.

# Constructors

    SecondLowerMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())

Keyword arguments correspond to the fields above.

# Formulations

Depending on the `alg` field, this can represent either the semi-variance or semi-standard deviation.

## `SOCRiskExpr`

Computes the semi-standard deviation (square root of semi-variance) of the returns below the target.

## `QuadRiskExpr`, `SquaredSOCRiskExpr`, `RSOCRiskExpr`

Computes the semi-variance of the returns below the target.

# Examples

```jldoctest
julia> SecondLowerMoment()
SecondLowerMoment
  alg | SquaredSOCRiskExpr()
```

# Related

  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
  - [`SecondMomentFormulation`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct SecondLowerMoment{T1} <: UnstandardisedSecondMomentAlgorithm
    alg::T1
    function SecondLowerMoment(alg::SecondMomentFormulation)
        return new{typeof(alg)}(alg)
    end
end
function SecondLowerMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondLowerMoment(alg)
end
"""
    struct SecondCentralMoment{T1} <: UnstandardisedSecondMomentAlgorithm
        alg::T1
    end

Represents the second central moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the second central moment (variance or standard deviation), which quantifies risk by considering all deviations from a target value.

# Fields

  - `alg`: The second moment formulation algorithm used to compute the risk measure.

# Constructors

    SecondCentralMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())

Keyword arguments correspond to the fields above.

# Formulations

Depending on the `alg` field, this can represent either the variance or standard deviation.

## `SOCRiskExpr`

Computes the standard deviation (square root of variance) of the returns below the target.

## `QuadRiskExpr`, `SquaredSOCRiskExpr`, `RSOCRiskExpr`

Computes the variance of the returns below the target.

# Examples

```jldoctest
julia> SecondCentralMoment()
SecondCentralMoment
  alg | SquaredSOCRiskExpr()
```

# Related

  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
  - [`SecondMomentFormulation`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct SecondCentralMoment{T1} <: UnstandardisedSecondMomentAlgorithm
    alg::T1
    function SecondCentralMoment(alg::SecondMomentFormulation)
        return new{typeof(alg)}(alg)
    end
end
function SecondCentralMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondCentralMoment(alg)
end
"""
    struct StandardisedLowOrderMoment{T1, T2} <: LowOrderMomentMeasureAlgorithm
        ve::T1
        alg::T2
    end

Represents a standardised low-order moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using a low-order moment algorithm, standardised by a variance estimator. This enables risk measures such as semi-standard deviation or standardised semi-variance, which normalise the risk by the portfolio variance.

# Fields

  - `ve`: Variance estimator used for standardisation.
  - `alg`: Unstandardised second moment algorithm used to compute the risk measure.

# Constructors

    StandardisedLowOrderMoment(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                               alg::UnstandardisedSecondMomentAlgorithm = SecondLowerMoment())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> StandardisedLowOrderMoment()
StandardisedLowOrderMoment
   ve | SimpleVariance
      |          me | nothing
      |           w | nothing
      |   corrected | Bool: true
  alg | SecondLowerMoment
      |   alg | SquaredSOCRiskExpr()
```

# Related

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
  - [`SimpleVariance`](@ref)
  - [`SecondLowerMoment`](@ref)
"""
struct StandardisedLowOrderMoment{T1, T2} <: LowOrderMomentMeasureAlgorithm
    ve::T1
    alg::T2
    function StandardisedLowOrderMoment(ve::AbstractVarianceEstimator,
                                        alg::UnstandardisedSecondMomentAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedLowOrderMoment(;
                                    ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                   me = nothing),
                                    alg::UnstandardisedSecondMomentAlgorithm = SecondLowerMoment())
    return StandardisedLowOrderMoment(ve, alg)
end
"""
    abstract type HighOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end

Abstract supertype for all high-order moment-based risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using high-order statistical moments (e.g., skewness, kurtosis) of the return distribution. All concrete high-order moment risk measure algorithms should subtype `HighOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
"""
abstract type HighOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
    abstract type UnstandardisedHighOrderMomentMeasureAlgorithm <: HighOrderMomentMeasureAlgorithm end

Abstract supertype for high-order moment risk measure algorithms that are not standardised by the variance in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using high-order statistical moments (such as skewness, kurtosis) without normalising by the variance. All concrete unstandardised high-order moment risk measure algorithms should subtype `UnstandardisedHighOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`ThirdLowerMoment`](@ref)
  - [`FourthLowerMoment`](@ref)
  - [`FourthCentralMoment`](@ref)
"""
abstract type UnstandardisedHighOrderMomentMeasureAlgorithm <:
              HighOrderMomentMeasureAlgorithm end
"""
    struct ThirdLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end

Represents the unstandardised semi-skewness risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the third lower moment (unstandardised semi-skewness), which quantifies downside asymmetry by considering only the cubed deviations below a target value. This algorithm is unstandardised and operates directly on the return distribution.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
"""
struct ThirdLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
"""
    struct FourthLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end

Represents the unstandardised semi-kurtosis risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the fourth lower moment (unstandardised semi-kurtosis), which quantifies downside tail risk by considering only the quartic deviations below a target value. This algorithm is unstandardised and operates directly on the return distribution.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
"""
struct FourthLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
"""
    struct FourthCentralMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end

Represents the unstandardised kurtosis risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the fourth central moment (unstandardised kurtosis), which quantifies tail risk by considering all quartic deviations from a target value. This algorithm is unstandardised and operates directly on the return distribution.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
"""
struct FourthCentralMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
"""
    struct StandardisedHighOrderMoment{T1, T2} <: HighOrderMomentMeasureAlgorithm
        ve::T1
        alg::T2
    end

Represents a standardised high-order moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using a high-order moment algorithm (such as semi-skewness or semi-kurtosis), standardised by a variance estimator. This enables risk measures such as standardised semi-skewness or standardised semi-kurtosis, which normalise the risk by the portfolio variance.

# Fields

  - `ve`: Variance estimator used for standardisation.
  - `alg`: Unstandardised high-order moment algorithm used to compute the risk measure.

# Constructors

    StandardisedHighOrderMoment(;
                                ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                                alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> StandardisedHighOrderMoment()
StandardisedHighOrderMoment
   ve | SimpleVariance
      |          me | nothing
      |           w | nothing
      |   corrected | Bool: true
  alg | ThirdLowerMoment()
```

# Related

  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`SimpleVariance`](@ref)
  - [`ThirdLowerMoment`](@ref)
  - [`FourthLowerMoment`](@ref)
  - [`FourthCentralMoment`](@ref)
"""
struct StandardisedHighOrderMoment{T1, T2} <: HighOrderMomentMeasureAlgorithm
    ve::T1
    alg::T2
    function StandardisedHighOrderMoment(ve::AbstractVarianceEstimator,
                                         alg::UnstandardisedHighOrderMomentMeasureAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedHighOrderMoment(;
                                     ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                    me = nothing),
                                     alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return StandardisedHighOrderMoment(ve, alg)
end
for alg in (StandardisedLowOrderMoment, StandardisedHighOrderMoment)
    eval(quote
             function factory(alg::$(alg), w::Union{Nothing, <:AbstractWeights} = nothing)
                 return $(alg)(; ve = factory(alg.ve, w), alg = alg.alg)
             end
         end)
end
"""
"""
struct LowOrderMoment{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function LowOrderMoment(settings::RiskMeasureSettings,
                            w::Union{Nothing, <:AbstractWeights},
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                            alg::LowOrderMomentMeasureAlgorithm)
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu) && all(isfinite, mu))
        elseif isa(mu, Real)
            @argcheck(isfinite(mu))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                        alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment())
    return LowOrderMoment(settings, w, mu, alg)
end
"""
"""
struct HighOrderMoment{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function HighOrderMoment(settings::RiskMeasureSettings,
                             w::Union{Nothing, <:AbstractWeights},
                             mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                             alg::HighOrderMomentMeasureAlgorithm)
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu) && all(isfinite, mu))
        elseif isa(mu, Real)
            @argcheck(isfinite(mu))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Union{Nothing, <:AbstractWeights} = nothing,
                         mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                         alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return HighOrderMoment(settings, w, mu, alg)
end
function calc_moment_target(::Union{<:LowOrderMoment{<:Any, Nothing, Nothing, <:Any},
                                    <:HighOrderMoment{<:Any, Nothing, Nothing, <:Any}},
                            ::Any, x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:AbstractWeights, Nothing,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:AbstractWeights, Nothing,
                                                       <:Any}}, ::Any, x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                       <:Any}}, w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Real, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:Real, <:Any}}, ::Any,
                            ::Any)
    return r.mu
end
function calc_moment_val(r::Union{<:LowOrderMoment, <:HighOrderMoment}, w::AbstractVector,
                         X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment})(w::AbstractVector,
                                                                      X::AbstractMatrix,
                                                                      fees::Union{Nothing,
                                                                                  <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondLowerMoment{<:SOCRiskExpr}}})(w::AbstractVector,
                                                                                               X::AbstractMatrix,
                                                                                               fees::Union{Nothing,
                                                                                                           <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any, <:SecondLowerMoment}})(w::AbstractVector,
                                                                                       X::AbstractMatrix,
                                                                                       fees::Union{Nothing,
                                                                                                   <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondCentralMoment{<:SOCRiskExpr}}})(w::AbstractVector,
                                                                                                 X::AbstractMatrix,
                                                                                                 fees::Union{Nothing,
                                                                                                             <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any, <:SecondCentralMoment}})(w::AbstractVector,
                                                                                         X::AbstractMatrix,
                                                                                         fees::Union{Nothing,
                                                                                                     <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = abs.(calc_moment_val(r, w, X, fees))
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment})(w::AbstractVector,
                                                                       X::AbstractMatrix,
                                                                       fees::Union{Nothing,
                                                                                   <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 3
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthLowerMoment})(w::AbstractVector,
                                                                        X::AbstractMatrix,
                                                                        fees::Union{Nothing,
                                                                                    <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthCentralMoment})(w::AbstractVector,
                                                                          X::AbstractMatrix,
                                                                          fees::Union{Nothing,
                                                                                      <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:ThirdLowerMoment}})(w::AbstractVector,
                                                                                        X::AbstractMatrix,
                                                                                        fees::Union{Nothing,
                                                                                                    <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? -mean(val) : -mean(val, r.w)
    return res / (sigma * sqrt(sigma))
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthLowerMoment}})(w::AbstractVector,
                                                                                         X::AbstractMatrix,
                                                                                         fees::Union{Nothing,
                                                                                                     <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthCentralMoment}})(w::AbstractVector,
                                                                                           X::AbstractMatrix,
                                                                                           fees::Union{Nothing,
                                                                                                       <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
for rt in (LowOrderMoment, HighOrderMoment)
    eval(quote
             function factory(r::$(rt), prior::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_factory(r.w, prior.w)
                 mu = nothing_scalar_array_factory(r.mu, prior.mu)
                 alg = factory(r.alg, w)
                 return $(rt)(; settings = r.settings, alg = alg, w = w, mu = mu)
             end
             function risk_measure_view(r::$(rt), i::AbstractVector, args...)
                 mu = nothing_scalar_array_view(r.mu, i)
                 return $(rt)(; settings = r.settings, alg = r.alg, w = r.w, mu = mu)
             end
         end)
end

export FirstLowerMoment, SecondLowerMoment, SecondCentralMoment, MeanAbsoluteDeviation,
       ThirdLowerMoment, FourthLowerMoment, FourthCentralMoment, StandardisedLowOrderMoment,
       StandardisedHighOrderMoment, LowOrderMoment, HighOrderMoment
