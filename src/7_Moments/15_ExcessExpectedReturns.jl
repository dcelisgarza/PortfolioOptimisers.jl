"""
    struct ExcessExpectedReturns{T1, T2} <: AbstractShrunkExpectedReturnsEstimator
        me::T1
        rf::T2
    end

Container type for excess expected returns estimators.

`ExcessExpectedReturns` encapsulates a mean estimator and a risk-free rate for computing excess expected returns. This enables modular workflows for estimating expected returns above a specified risk-free rate.

# Fields

  - `me`: Mean estimator for expected returns.
  - `rf`: Risk-free rate to subtract from expected returns.

# Constructor

    ExcessExpectedReturns(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                          rf::Real = 0.0)

Construct an `ExcessExpectedReturns` estimator with the specified mean estimator and risk-free rate.

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
struct ExcessExpectedReturns{T1, T2} <: AbstractShrunkExpectedReturnsEstimator
    me::T1
    rf::T2
end
"""
    ExcessExpectedReturns(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                          rf::Real = 0.0)

Construct an [`ExcessExpectedReturns`](@ref) estimator for excess expected returns.

# Arguments

  - `me`: Mean estimator for expected returns.
  - `rf`: Risk-free rate to subtract.

# Returns

  - `ExcessExpectedReturns`: Configured excess expected returns estimator.

# Examples

```jldoctest
julia> ExcessExpectedReturns()
ExcessExpectedReturns
  me | SimpleExpectedReturns
     |   w | nothing
  rf | Float64: 0.0
```

# Related

  - [`ExcessExpectedReturns`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
function ExcessExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               rf::Real = 0.0)
    return ExcessExpectedReturns(me, rf)
end

"""
    mean(me::ExcessExpectedReturns, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute excess expected returns by subtracting the risk-free rate.

This method applies the mean estimator to the data and subtracts the risk-free rate from the resulting expected returns.

# Arguments

  - `me`: Excess expected returns estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the mean.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - `mu::AbstractArray`: Excess expected returns vector.

# Related

  - [`ExcessExpectedReturns`](@ref)
"""
function Statistics.mean(me::ExcessExpectedReturns, X::AbstractMatrix; dims::Int = 1,
                         kwargs...)
    return mean(me.me, X; dims = dims, kwargs...) .- me.rf
end
function factory(me::ExcessExpectedReturns, w::Union{Nothing, <:AbstractWeights} = nothing)
    return ExcessExpectedReturns(; me = factory(me.me, w), rf = me.rf)
end

export ExcessExpectedReturns
