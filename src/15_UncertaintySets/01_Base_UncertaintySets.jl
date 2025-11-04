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
"""
    abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end

Defines the abstract interface for algorithms that compute the scaling parameter `k` for ellipse uncertainty sets in portfolio optimisation.

Subtypes implement specific methods for generating the scaling parameter, which controls the size of the ellipsoidal region representing uncertainty in risk or prior statistics.

# Related Types

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
"""
abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end
"""
    ucs(uc::Union{Nothing,
                  <:Tuple{<:Union{Nothing, <:AbstractUncertaintySetResult},
                          <:Union{Nothing, <:AbstractUncertaintySetResult}}}, args...; kwargs...)

Returns the argument(s) unchanged. This is a no-op function used to handle cases where no uncertainty sets, or a tuple of pre-processed sets is provided.

# Arguments

  - `uc`: Tuple of uncertainty sets, or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `uc::Union{Nothing, <:Tuple{<:Union{Nothing, <:AbstractUncertaintySetResult}, <:Union{Nothing, <:AbstractUncertaintySetResult}}}`: The input, unchanged.

# Related

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
function ucs(uc::Union{Nothing,
                       <:Tuple{<:Union{Nothing, <:AbstractUncertaintySetResult},
                               <:Union{Nothing, <:AbstractUncertaintySetResult}}}, args...;
             kwargs...)
    return uc
end
"""
    mu_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)

Returns the argument unchanged. This is a no-op function used to handle cases where no expected returns uncertainty set is provided.

# Arguments

  - `uc`: Expected returns uncertainty set or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `uc::Union{Nothing, <:AbstractUncertaintySetResult}`: The input, unchanged.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
function mu_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)
    return uc
end
"""
    sigma_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)

Returns the argument unchanged. This is a no-op function used to handle cases where no covariance uncertainty set is provided.

# Arguments

  - `uc`: Covariance uncertainty set or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `uc::Union{Nothing, <:AbstractUncertaintySetResult}`: The input, unchanged.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
function sigma_ucs(uc::Union{Nothing, <:AbstractUncertaintySetResult}, args...; kwargs...)
    return uc
end
"""
    ucs_factory(risk_ucs::Nothing, prior_ucs::Nothing)
    ucs_factory(risk_ucs::Union{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetEstimator}, prior_ucs::Any)
    ucs_factory(risk_ucs::Nothing, prior_ucs::Union{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetEstimator})

Factory function for selecting uncertainty sets from risk measure or prior result instances.

# Arguments

  - `risk_ucs`: Risk measure uncertainty set estimator or result, or `nothing`.
  - `prior_ucs`: Prior result uncertainty set estimator or result, or `nothing`.

# Returns

  - `nothing`: If both `risk_ucs` and `prior_ucs` are `nothing`.
  - `risk_ucs::Union{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetEstimator}`: If `risk_ucs` is not `nothing`.
  - `prior_ucs::Union{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetEstimator}`: If `risk_ucs` is `nothing` but `prior_ucs` is not `nothing`.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`factory`](@ref)
"""
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
    ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)

Constructs an uncertainty set from a given estimator and returns data.

# Arguments

  - `uc`: Uncertainty set estimator. Used to construct the uncertainty set.
  - `rd`: ReturnsResult. Contains the returns data and associated metadata.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `uc::Tuple{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetResult}`: Expected returns and covariance uncertainty sets.

# Details

  - Calls the estimator on the returns data and metadata in `rd`.
  - Passes `rd.X`, `rd.F`, and relevant metadata (`iv`, `ivpa`) to the estimator.
  - Additional keyword arguments are forwarded.
  - Used for compatibility with `ReturnsResult` objects.

# Related

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
function ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
    mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)

Constructs an expected returns uncertainty set from a given estimator and returns data.

# Arguments

  - `uc`: Uncertainty set estimator. Used to construct the expected returns uncertainty set.
  - `rd`: ReturnsResult. Contains the returns data and associated metadata.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `uc::AbstractUncertaintySetResult`: Expected returns uncertainty set.

# Details

  - Calls the estimator on the returns data and metadata in `rd`.
  - Passes `rd.X`, `rd.F`, and relevant metadata (`iv`, `ivpa`) to the estimator.
  - Additional keyword arguments are forwarded.
  - Used for compatibility with `ReturnsResult` objects.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
function mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return mu_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
    sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)

Constructs a covariance uncertainty set from a given estimator and returns data.

# Arguments

  - `uc`: Uncertainty set estimator. Used to construct the covariance uncertainty set.
  - `rd`: ReturnsResult. Contains the returns data and associated metadata.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `uc::AbstractUncertaintySetResult`: Covariance uncertainty set.

# Details

  - Calls the estimator on the returns data and metadata in `rd`.
  - Passes `rd.X`, `rd.F`, and relevant metadata (`iv`, `ivpa`) to the estimator.
  - Additional keyword arguments are forwarded.
  - Used for compatibility with `ReturnsResult` objects.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
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

# Constructor

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
  lb ┼ Vector{Float64}: [0.1, 0.2]
  ub ┴ Vector{Float64}: [0.3, 0.4]
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
        @argcheck(!isempty(lb))
        @argcheck(!isempty(ub))
        @argcheck(size(lb) == size(ub))
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function BoxUncertaintySet(; lb::AbstractArray, ub::AbstractArray)
    return BoxUncertaintySet(lb, ub)
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:NumVec, <:NumVec}, i::NumVec)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i))
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:NumMat, <:NumMat}, i::NumVec)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i, i), ub = view(risk_ucs.ub, i, i))
end
"""
    struct NormalKUncertaintyAlgorithm{T1} <: AbstractUncertaintyKAlgorithm
        kwargs::T1
    end

Algorithm for computing the scaling parameter `k` for ellipse uncertainty sets under the assumption of normally distributed returns in portfolio optimisation.

# Fields

  - `kwargs`: Named tuple of keyword arguments for quantile calculation.

# Constructor

    NormalKUncertaintyAlgorithm(; kwargs::NamedTuple = (;))

Keyword arguments correspond to the field above.

## Validation

  - `kwargs` must be a valid `NamedTuple`.

# Examples

```jldoctest
julia> NormalKUncertaintyAlgorithm()
NormalKUncertaintyAlgorithm
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
"""
struct NormalKUncertaintyAlgorithm{T1} <: AbstractUncertaintyKAlgorithm
    kwargs::T1
    function NormalKUncertaintyAlgorithm(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function NormalKUncertaintyAlgorithm(; kwargs::NamedTuple = (;))
    return NormalKUncertaintyAlgorithm(kwargs)
end
"""
    struct GeneralKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end

Algorithm for computing the scaling parameter `k` for ellipse uncertainty sets using a general formula `sqrt((1 - q) / q)`, this ignores the distribution of the underlying data.

# Related Types

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
"""
struct GeneralKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
"""
    struct ChiSqKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end

Algorithm for computing the scaling parameter `k` for ellipse uncertainty sets using the chi-squared distribution in portfolio optimisation.

# Related Types

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
"""
struct ChiSqKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
"""
    k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::NumMat, sigma_X::NumMat)
    k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    k_ucs(::ChiSqKUncertaintyAlgorithm, q::Number, X::AbstractArray, args...)
    k_ucs(type::Number, args...)

Computes the scaling parameter `k` for ellipse uncertainty sets in portfolio optimisation.

# Arguments

  - `km`: Scaling algorithm instance.
  - `q`: Quantile or confidence level.
  - `X`: Data matrix (returns).
  - `sigma_X`: Covariance matrix.
  - `args...`: Additional arguments.
  - `type`: Number value for direct scaling.

# Returns

  - `k::Number`: Scaling parameter.

# Details

  - Uses different algorithms to compute the scaling parameter:

      + Normal: `1 - q`'th quantile of the Mahalanobis distances.
      + General: formula `sqrt((1 - q) / q)`.
      + Chi-squared: `1 - q`'th quantile of the chi-squared distribution.
      + Number: returns the provided value directly.

  - Supports multiple dispatch for extensibility.

# Related

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
"""
function k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::NumMat, sigma_X::NumMat)
    k_mus = diag(X * (sigma_X \ transpose(X)))
    return sqrt(quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    return sqrt((one(q) - q) / q)
end
function k_ucs(::ChiSqKUncertaintyAlgorithm, q::Number, X::AbstractArray, args...)
    return sqrt(cquantile(Chisq(size(X, 1)), q))
end
function k_ucs(type::Number, args...)
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

# Constructor

    EllipseUncertaintySetAlgorithm(;
                                   method::Union{<:AbstractUncertaintyKAlgorithm, <:Number} = ChiSqKUncertaintyAlgorithm(),
                                   diagonal::Bool = true)

  - `method`: Sets the scaling algorithm or value for the ellipse.
  - `diagonal`: Sets whether to use only diagonal elements.

# Examples

```jldoctest
julia> EllipseUncertaintySetAlgorithm()
EllipseUncertaintySetAlgorithm
    method ┼ ChiSqKUncertaintyAlgorithm()
  diagonal ┴ Bool: true
```

# Related

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
"""
struct EllipseUncertaintySetAlgorithm{T1, T2} <: AbstractUncertaintySetAlgorithm
    method::T1
    diagonal::T2
    function EllipseUncertaintySetAlgorithm(method::Union{<:AbstractUncertaintyKAlgorithm,
                                                          <:Number}, diagonal::Bool)
        return new{typeof(method), typeof(diagonal)}(method, diagonal)
    end
end
function EllipseUncertaintySetAlgorithm(;
                                        method::Union{<:AbstractUncertaintyKAlgorithm,
                                                      <:Number} = ChiSqKUncertaintyAlgorithm(),
                                        diagonal::Bool = true)
    return EllipseUncertaintySetAlgorithm(method, diagonal)
end
"""
    abstract type AbstractEllipseUncertaintySetResultClass <: AbstractUncertaintySetResult end

Defines the abstract interface for ellipse uncertainty set result classes in portfolio optimisation.

Subtypes of this abstract type represent the class or category of ellipse uncertainty sets, such as those for mean or covariance statistics. Used to distinguish between different types of ellipse uncertainty set results.

# Related Types

  - [`MuEllipseUncertaintySet`](@ref)
  - [`SigmaEllipseUncertaintySet`](@ref)
"""
abstract type AbstractEllipseUncertaintySetResultClass <: AbstractUncertaintySetResult end
"""
    struct MuEllipseUncertaintySet <: AbstractEllipseUncertaintySetResultClass end

Represents the class identifier for mean ellipse uncertainty sets in portfolio optimisation.

Used to distinguish ellipse uncertainty sets that encode uncertainty for mean statistics, such as expected returns.

# Related Types

  - [`AbstractEllipseUncertaintySetResultClass`](@ref)
  - [`SigmaEllipseUncertaintySet`](@ref)
"""
struct MuEllipseUncertaintySet <: AbstractEllipseUncertaintySetResultClass end
"""
    struct SigmaEllipseUncertaintySet <: AbstractEllipseUncertaintySetResultClass end

Represents the class identifier for covariance ellipse uncertainty sets in portfolio optimisation.

Used to distinguish ellipse uncertainty sets that encode uncertainty for covariance statistics, such as covariance matrices.

# Related Types

  - [`AbstractEllipseUncertaintySetResultClass`](@ref)
  - [`MuEllipseUncertaintySet`](@ref)
"""
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

# Constructor

    EllipseUncertaintySet(; sigma::NumMat, k::Number,
                          class::AbstractEllipseUncertaintySetResultClass)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(sigma)`.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `k > 0`.

# Examples

```jldoctest
julia> EllipseUncertaintySet([1.0 0.2; 0.2 1.0], 2.5, SigmaEllipseUncertaintySet())
EllipseUncertaintySet
  sigma ┼ 2×2 Matrix{Float64}
      k ┼ Float64: 2.5
  class ┴ SigmaEllipseUncertaintySet()
```

# Related

  - [`AbstractEllipseUncertaintySetResultClass`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
"""
struct EllipseUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetResult
    sigma::T1
    k::T2
    class::T3
    function EllipseUncertaintySet(sigma::NumMat, k::Number,
                                   class::AbstractEllipseUncertaintySetResultClass)
        @argcheck(!isempty(sigma))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(k > zero(k))
        return new{typeof(sigma), typeof(k), typeof(class)}(sigma, k, class)
    end
end
function EllipseUncertaintySet(; sigma::NumMat, k::Number,
                               class::AbstractEllipseUncertaintySetResultClass)
    return EllipseUncertaintySet(sigma, k, class)
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:NumMat, <:Any,
                                                  <:SigmaEllipseUncertaintySet}, i::NumVec)
    i = fourth_moment_index_factory(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:NumMat, <:Any,
                                                  <:MuEllipseUncertaintySet}, i::NumVec)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySet,
       NormalKUncertaintyAlgorithm, GeneralKUncertaintyAlgorithm,
       ChiSqKUncertaintyAlgorithm, EllipseUncertaintySetAlgorithm, EllipseUncertaintySet,
       SigmaEllipseUncertaintySet, MuEllipseUncertaintySet
