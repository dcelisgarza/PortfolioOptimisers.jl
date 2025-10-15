"""
    abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end

Defines the abstract interface for uncertainty set estimators in portfolio optimisation.
Subtypes of this abstract type are responsible for constructing and estimating uncertainty sets for risk or prior statistics, such as box or ellipse uncertainty sets.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
"""
abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end
"""
    abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end

Defines the abstract interface for algorithms that construct uncertainty sets in portfolio optimisation.
Subtypes implement specific methods for generating uncertainty sets, such as box or ellipse uncertainty sets, which are used to model uncertainty in risk or prior statistics.

# Related

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end
"""
    abstract type AbstractUncertaintySetResult <: AbstractResult end

Abstract type for results produced by uncertainty set algorithms in portfolio optimisation.

Represents the interface for all result types that encode uncertainty sets for risk or prior statistics, such as box or ellipse uncertainty sets. Subtypes store the output of uncertainty set estimation or construction algorithms.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
"""
abstract type AbstractUncertaintySetResult <: AbstractResult end
abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end
function ucs(uc::Union{Nothing,
                       <:Tuple{<:Union{Nothing, <:AbstractUncertaintySetResult},
                               <:Union{Nothing, <:AbstractUncertaintySetResult}}}, args...;
             kwargs...)
    return uc
end
function mu_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)
    return uc
end
function sigma_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)
    return uc
end
function ucs_factory(::Nothing, ::Nothing)
    return nothing
end
function ucs_factory(risk_ucs::Union{<:AbstractUncertaintySetResult,
                                     <:AbstractUncertaintySetEstimator}, ::Any)
    return risk_ucs
end
function ucs_factory(::Nothing,
                     prior_ucs::Union{<:AbstractUncertaintySetResult,
                                      <:AbstractUncertaintySetEstimator})
    return prior_ucs
end
function ucs_view(risk_ucs::Union{Nothing, <:AbstractUncertaintySetEstimator}, ::Any)
    return risk_ucs
end
function ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
function mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return mu_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
function sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return sigma_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
    struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end

Algorithm for constructing box uncertainty sets in portfolio optimisation.
Box uncertainty sets model uncertainty by specifying lower and upper bounds for risk or prior statistics.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
"""
struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
"""
    struct BoxUncertaintySet{T1, T2} <: AbstractUncertaintySetResult
        lb::T1
        ub::T2
    end

Represents a box uncertainty set for risk or prior statistics in portfolio optimisation.
Stores lower and upper bounds for the uncertain quantity, such as expected returns or covariance.

# Fields

  - `lb`: Lower bound array for the uncertainty set.
  - `ub`: Upper bound array for the uncertainty set.

# Constructors

    BoxUncertaintySet(; lb::AbstractArray, ub::AbstractArray)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(lb)`.
  - `!isempty(ub)`.
  - `size(lb) == size(ub)`.

# Examples

```jldoctest
julia> BoxUncertaintySet(; lb = [0.1, 0.2], ub = [0.3, 0.4])
BoxUncertaintySet
  lb | Vector{Float64}: [0.1, 0.2]
  ub | Vector{Float64}: [0.3, 0.4]
```

# Related

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
struct BoxUncertaintySet{T1, T2} <: AbstractUncertaintySetResult
    lb::T1
    ub::T2
    function BoxUncertaintySet(lb::AbstractArray, ub::AbstractArray)
        @argcheck(!isempty(lb) && !isempty(ub))
        @argcheck(size(lb) == size(ub))
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function BoxUncertaintySet(; lb::AbstractArray, ub::AbstractArray)
    return BoxUncertaintySet(lb, ub)
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:AbstractVector, <:AbstractVector},
                  i::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i))
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:AbstractMatrix, <:AbstractMatrix},
                  i::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i, i), ub = view(risk_ucs.ub, i, i))
end
struct NormalKUncertaintyAlgorithm{T1} <: AbstractUncertaintyKAlgorithm
    kwargs::T1
    function NormalKUncertaintyAlgorithm(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function NormalKUncertaintyAlgorithm(; kwargs::NamedTuple = (;))
    return NormalKUncertaintyAlgorithm(kwargs)
end
struct GeneralKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
struct ChiSqKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
function k_ucs(km::NormalKUncertaintyAlgorithm, q::Real, X::AbstractMatrix,
               sigma_X::AbstractMatrix)
    k_mus = diag(X * (sigma_X \ transpose(X)))
    return sqrt(quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_ucs(::GeneralKUncertaintyAlgorithm, q::Real, args...)
    return sqrt((one(q) - q) / q)
end
function k_ucs(::ChiSqKUncertaintyAlgorithm, q::Real, X::AbstractArray, args...)
    return sqrt(cquantile(Chisq(size(X, 1)), q))
end
function k_ucs(type::Real, args...)
    return type
end
"""
    struct EllipseUncertaintySetAlgorithm{T1, T2} <: AbstractUncertaintySetAlgorithm
        method::T1
        diagonal::T2
    end

Algorithm for constructing ellipse uncertainty sets in portfolio optimisation.
Ellipse uncertainty sets model uncertainty by specifying an ellipsoidal region for risk or prior statistics, typically using a covariance matrix and a scaling parameter.

# Fields

  - `method`: Algorithm or value used to determine the scaling parameter for the ellipse.
  - `diagonal`: Indicates whether to use only the diagonal elements of the covariance matrix.

# Constructors

    EllipseUncertaintySetAlgorithm(;
                                   method::Union{<:AbstractUncertaintyKAlgorithm, <:Real} = ChiSqKUncertaintyAlgorithm(),
                                   diagonal::Bool = true)

  - `method`: Sets the scaling algorithm or value for the ellipse.
  - `diagonal`: Sets whether to use only diagonal elements.

# Examples

```jldoctest
julia> EllipseUncertaintySetAlgorithm()
EllipseUncertaintySetAlgorithm
    method | ChiSqKUncertaintyAlgorithm()
  diagonal | Bool: true
```

# Related

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
"""
struct EllipseUncertaintySetAlgorithm{T1, T2} <: AbstractUncertaintySetAlgorithm
    method::T1
    diagonal::T2
    function EllipseUncertaintySetAlgorithm(method::Union{<:AbstractUncertaintyKAlgorithm,
                                                          <:Real}, diagonal::Bool)
        return new{typeof(method), typeof(diagonal)}(method, diagonal)
    end
end
function EllipseUncertaintySetAlgorithm(;
                                        method::Union{<:AbstractUncertaintyKAlgorithm,
                                                      <:Real} = ChiSqKUncertaintyAlgorithm(),
                                        diagonal::Bool = true)
    return EllipseUncertaintySetAlgorithm(method, diagonal)
end
abstract type AbstractEllipseUncertaintySetResultClass <: AbstractUncertaintySetResult end
struct MuEllipseUncertaintySet <: AbstractEllipseUncertaintySetResultClass end
struct SigmaEllipseUncertaintySet <: AbstractEllipseUncertaintySetResultClass end
"""
    struct EllipseUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetResult
        sigma::T1
        k::T2
        class::T3
    end

Represents an ellipse uncertainty set for risk or prior statistics in portfolio optimisation.
Stores a covariance matrix, a scaling parameter, and a class identifier for the uncertain quantity, such as expected returns or covariance.

# Fields

  - `sigma`: Covariance matrix for the uncertainty set.
  - `k`: Scaling parameter for the ellipse.
  - `class`: Identifier for the type of ellipse uncertainty set (e.g., mean or covariance).

# Constructors

    EllipseUncertaintySet(; sigma::AbstractMatrix, k::Real,
                          class::AbstractEllipseUncertaintySetResultClass)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(sigma)`.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `k > 0`.

# Examples

```jldoctest
julia> EllipseUncertaintySet([1.0 0.2; 0.2 1.0], 2.5,
                             PortfolioOptimisers.SigmaEllipseUncertaintySet())
EllipseUncertaintySet
  sigma | 2×2 Matrix{Float64}
      k | Float64: 2.5
  class | PortfolioOptimisers.SigmaEllipseUncertaintySet()
```

# Related

  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`BoxUncertaintySet`](@ref)
"""
struct EllipseUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetResult
    sigma::T1
    k::T2
    class::T3
    function EllipseUncertaintySet(sigma::AbstractMatrix, k::Real,
                                   class::AbstractEllipseUncertaintySetResultClass)
        @argcheck(!isempty(sigma))
        assert_matrix_issquare(sigma)
        @argcheck(k > zero(k))
        return new{typeof(sigma), typeof(k), typeof(class)}(sigma, k, class)
    end
end
function EllipseUncertaintySet(; sigma::AbstractMatrix, k::Real,
                               class::AbstractEllipseUncertaintySetResultClass)
    return EllipseUncertaintySet(sigma, k, class)
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any,
                                                  <:SigmaEllipseUncertaintySet},
                  i::AbstractVector)
    i = fourth_moment_index_factory(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any,
                                                  <:MuEllipseUncertaintySet},
                  i::AbstractVector)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySet,
       NormalKUncertaintyAlgorithm, GeneralKUncertaintyAlgorithm,
       ChiSqKUncertaintyAlgorithm, EllipseUncertaintySetAlgorithm, EllipseUncertaintySet
