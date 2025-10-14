"""
```julia
AbstractUncertaintySetEstimator
```

Abstract supertype for all uncertainty set estimators.

Uncertainty set estimators are used to construct sets that describe the plausible range of portfolio statistics (such as expected returns or covariances) under model or data uncertainty. Subtypes must implement the required interfaces for composability with optimisation routines.

## Related Types

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end
"""
```julia
AbstractUncertaintySetAlgorithm
```

Abstract supertype for all uncertainty set algorithms.

Uncertainty set algorithms define the procedures for constructing uncertainty sets over portfolio statistics (such as expected returns or covariances) given model or data uncertainty. Subtypes implement specific algorithms (e.g., box, ellipse, bootstrap) and must provide the required interfaces for use in optimisation routines.

## Related Types

  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end
"""
```julia
AbstractUncertaintySetResult
```

Abstract supertype for all uncertainty set results.

Uncertainty set results represent concrete, validated sets describing plausible values for expected returns and covariance under model or data uncertainty. Subtypes encode the geometry and data required to describe the set.

## Related Types

  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
abstract type AbstractUncertaintySetResult <: AbstractResult end
"""
```julia
AbstractUncertaintyKAlgorithm
```

Abstract supertype for all uncertainty set radius algorithms.

Uncertainty set radius algorithms define how the scaling parameter k is computed for ellipse-type uncertainty sets, controlling the size of the plausible region for portfolio statistics under model or data uncertainty. Subtypes implement specific methods and must provide the required interface for use in uncertainty set construction.

## Related Types

  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end
"""
```julia
ucs(uc::Union{Nothing,
              <:Tuple{<:Union{Nothing, <:AbstractUncertaintySetResult},
                      <:Union{Nothing, <:AbstractUncertaintySetResult}}}, args...;
    kwargs...)
```

No-op utility for uncertainty set arguments.

Returns the input `uc` unchanged. Used as a generic pass-through for expected returns and covariance uncertainty sets in optimisation routines, allowing for composable interfaces when no uncertainty set is specified.

## Arguments

  - `uc`: Either `nothing` or a tuple of expected returns and covariance uncertainty sets (or `nothing`).
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

## Returns

  - The input `uc`, unchanged.

## Usage Example

```jldoctest
julia> ucs(nothing)

julia> ucs((nothing, nothing))
(nothing, nothing)
```

## Related Methods

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(uc::Union{Nothing,
                       <:Tuple{<:Union{Nothing, <:AbstractUncertaintySetResult},
                               <:Union{Nothing, <:AbstractUncertaintySetResult}}}, args...;
             kwargs...)
    return uc
end
"""
```julia
mu_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)
```

No-op utility for expected returns uncertainty set arguments.

Returns the input `uc` unchanged. Used as a generic pass-through for expected returns uncertainty sets in optimisation routines, enabling composable interfaces when no uncertainty set is specified.

## Arguments

  - `uc`: Either `nothing` or an expected returns uncertainty set result.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

## Returns

  - The input `uc`, unchanged.

## Usage Example

```jldoctest
julia> mu_ucs(nothing)

```

## Related Methods

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)
    return uc
end
"""
```julia
sigma_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)
```

No-op utility for covariance uncertainty set arguments.

Returns the input `uc` unchanged. Used as a generic pass-through for covariance uncertainty sets in optimisation routines, enabling composable interfaces when no uncertainty set is specified.

## Arguments

  - `uc`: Either `nothing` or a covariance uncertainty set result.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

## Returns

  - The input `uc`, unchanged.

## Usage Example

```jldoctest
julia> sigma_ucs(nothing)

```

## Related Methods

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
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
"""
```julia
ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
```

Wrapper for expected returns and covariance uncertainty set estimators to work with [`ReturnsResult`](@ref).

Constructs uncertainty sets for expected returns and/or covariance using the provided estimator and returns data. This method enables composable integration of uncertainty set estimators with optimisation routines by extracting the relevant data from a `ReturnsResult` and forwarding it to the estimator.

## Arguments

  - `uc`: An uncertainty set estimator.
  - `rd`: A [`ReturnsResult`](@ref) containing returns data.
  - `kwargs...`: Additional keyword arguments forwarded to the estimator.

## Returns

  - The result of `ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)`.

## Related Methods

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
    mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)

Wrapper for expected returns uncertainty set estimators to work with [`ReturnsResult`](@ref).

Constructs an expected returns uncertainty set using the provided estimator and returns data. This method enables composable integration of expected returns uncertainty set estimators with optimisation routines by extracting the relevant data from a `ReturnsResult` and forwarding it to the estimator.

## Arguments

  - `uc`: An uncertainty set estimator.
  - `rd`: A [`ReturnsResult`](@ref) containing returns data.
  - `kwargs...`: Additional keyword arguments forwarded to the estimator.

## Returns

  - The result of `mu_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)`, typically an expected returns uncertainty set result.

## Related Methods

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return mu_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
    sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)

Wrapper for covariance uncertainty set estimators to work with [`ReturnsResult`](@ref).

Constructs a covariance uncertainty set using the provided estimator and returns data. This method enables composable integration of covariance uncertainty set estimators with optimisation routines by extracting the relevant data from a `ReturnsResult` and forwarding it to the estimator.

## Arguments

  - `uc`: A covariance uncertainty set estimator subtype of [`AbstractUncertaintySetEstimator`](@ref).
  - `rd`: A [`ReturnsResult`](@ref) containing returns data and metadata.
  - `kwargs...`: Additional keyword arguments forwarded to the estimator.

## Returns

  - The result of `sigma_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)`, typically a covariance uncertainty set result.

## Usage Example

```jldoctest
julia> sigma_ucs(est, rd)
est = SomeCovarianceUncertaintySetEstimator(...)
```

## Related Methods

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return sigma_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
```julia
struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
```

Concrete algorithm type for box uncertainty sets.

`BoxUncertaintySetAlgorithm` encodes the procedure for constructing box-type uncertainty sets on expected returns or covariance. Box uncertainty sets specify lower and upper bounds for each statistic, forming a hyper-rectangle in parameter space. This algorithm is used to generate [`BoxUncertaintySet`](@ref) results for robust optimisation.

## Related Types

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
"""
```julia
struct BoxUncertaintySet{T1, T2} <: AbstractUncertaintySetResult
    lb::T1
    ub::T2
end
```

Concrete result type for box uncertainty sets.

`BoxUncertaintySet` encodes a hyper-rectangular uncertainty set for portfolio statistics (such as expected returns or covariances), defined by lower and upper bounds for each parameter. This type is used to represent the output of box uncertainty set algorithms, supporting robust optimisation by specifying the plausible range for each statistic under model or data uncertainty.

## Fields

  - `lb`: Lower bound array.
  - `ub`: Upper bound array.

## Constructor

```julia
BoxUncertaintySet(; lb::AbstractArray, ub::AbstractArray)
```

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(lb)`.
  - `!isempty(ub)`.
  - `size(lb) == size(ub)`.

## Usage Example

```jldoctest
julia> lb = [0.01, 0.02, 0.03];

julia> ub = [0.05, 0.06, 0.07];

julia> BoxUncertaintySet(lb, ub)
BoxUncertaintySet
  lb | Vector{Float64}: [0.01, 0.02, 0.03]
  ub | Vector{Float64}: [0.05, 0.06, 0.07]
```

## Related Types

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
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
    k_mus = diag(X * (sigma_X \ I) * transpose(X))
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
```julia
struct EllipseUncertaintySetAlgorithm{T1, T2} <: AbstractUncertaintySetAlgorithm
    method::T1
    diagonal::T2
end
```

Concrete algorithm type for ellipse uncertainty sets.

`EllipseUncertaintySetAlgorithm` encodes the procedure for constructing ellipse-type uncertainty sets over portfolio statistics (such as expected returns or covariances). Ellipse uncertainty sets specify a region defined by a covariance matrix and a scaling parameter `k`, forming an ellipsoidal plausible region in parameter space. This algorithm supports configurable methods for computing the scaling parameter (e.g., chi-squared, normal, general) and whether to use the diagonal or full covariance.

## Fields

  - `method`: Algorithm or value for computing the scaling parameter `k` (e.g., [`ChiSqKUncertaintyAlgorithm`](@ref), [`NormalKUncertaintyAlgorithm`](@ref), or a real number).
  - `diagonal`: Boolean indicating whether to use only the diagonal of the covariance matrix.

## Constructor

```julia
EllipseUncertaintySetAlgorithm(; method = ChiSqKUncertaintyAlgorithm(), diagonal = true)
```

Keyword arguments correspond to the fields above.

## Usage Example

```jldoctest
julia> alg = EllipseUncertaintySetAlgorithm(; method = NormalKUncertaintyAlgorithm(),
                                            diagonal = false)
EllipseUncertaintySetAlgorithm
  method   | NormalKUncertaintyAlgorithm
  diagonal | false
```

## Related Types

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
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
struct EllipseUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetResult
    sigma::T1
    k::T2
    class::T3
    function EllipseUncertaintySet(sigma::AbstractMatrix, k::Real,
                                   class::AbstractEllipseUncertaintySetResultClass)
        @argcheck(!isempty(sigma))
        assert_matrix_issquare(sigma)
        @argcheck(zero(k) < k)
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
