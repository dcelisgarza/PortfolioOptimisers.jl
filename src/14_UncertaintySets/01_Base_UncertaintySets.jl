"""
$(DocStringExtensions.TYPEDEF)

Defines the abstract interface for uncertainty set estimators in portfolio optimisation.
Subtypes of this abstract type are responsible for constructing and estimating uncertainty sets for risk or prior statistics, such as box or ellipsoidal uncertainty sets.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
"""
abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Defines the abstract interface for algorithms that construct uncertainty sets in portfolio optimisation.
Subtypes implement specific methods for generating uncertainty sets, such as box or ellipsoidal uncertainty sets, which are used to model uncertainty in risk or prior statistics.

# Related

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract type for results produced by uncertainty set algorithms in portfolio optimisation.

Represents the interface for all result types that encode uncertainty sets for risk or prior statistics, such as box or ellipsoidal uncertainty sets. Subtypes store the output of uncertainty set estimation or construction algorithms.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
"""
abstract type AbstractUncertaintySetResult <: AbstractResult end
"""
    const UcSE_UcS = Union{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetEstimator}

Alias for a union of uncertainty set result and estimator types.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
"""
const UcSE_UcS = Union{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Defines the abstract interface for algorithms that compute the scaling parameter `k` for ellipsoidal uncertainty sets in portfolio optimisation.

Subtypes implement specific methods for generating the scaling parameter, which controls the size of the ellipsoidal region representing uncertainty in risk or prior statistics.

# Related

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
"""
abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end
"""
    const Num_UcSK = Union{<:AbstractUncertaintyKAlgorithm, <:Number}

Alias for a union of uncertainty scaling algorithm and numeric types.

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
"""
const Num_UcSK = Union{<:AbstractUncertaintyKAlgorithm, <:Number}
"""
    ucs(uc::Option{<:Tuple{<:Option{<:AbstractUncertaintySetResult},
                           <:Option{<:AbstractUncertaintySetResult}}}, args...; kwargs...)

Returns the argument(s) unchanged. This is a no-op function used to handle cases where uncertainty sets are pre-processed (`nothing` or a tuple of results).

# Arguments

  - `uc`: Tuple of uncertainty sets, or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `uc::Option{<:Tuple{<:Option{<:AbstractUncertaintySetResult}, <:Option{<:AbstractUncertaintySetResult}}}`: The input, unchanged.

# Related

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function ucs(uc::Option{<:Tuple{<:Option{<:AbstractUncertaintySetResult},
                                <:Option{<:AbstractUncertaintySetResult}}}, args...;
             kwargs...)
    return uc
end
"""
    mu_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)

Returns the argument unchanged. This is a no-op function used to handle cases where the expected returns uncertainty set is already a result or is absent (`nothing`).

# Arguments

  - `uc`: Expected returns uncertainty set or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `uc::Option{<:AbstractUncertaintySetResult}`: The input, unchanged.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function mu_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...;
                kwargs...)::Option{<:AbstractUncertaintySetResult}
    return uc
end
"""
    sigma_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)

Returns the argument unchanged. This is a no-op function used to handle cases where the covariance uncertainty set is already a result or is absent (`nothing`).

# Arguments

  - `uc`: Covariance uncertainty set or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `uc::Option{<:AbstractUncertaintySetResult}`: The input, unchanged.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function sigma_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...;
                   kwargs...)::Option{<:AbstractUncertaintySetResult}
    return uc
end
"""
    ucs_selector(risk_ucs::Nothing, prior_ucs::Nothing)
    ucs_selector(risk_ucs::UcSE_UcS, prior_ucs::Any)
    ucs_selector(risk_ucs::Nothing, prior_ucs::UcSE_UcS)

Function for selecting uncertainty sets from risk measure or prior result instances.

# Arguments

  - `risk_ucs`: Risk measure uncertainty set estimator or result, or `nothing`.
  - `prior_ucs`: Prior result uncertainty set estimator or result, or `nothing`.

# Returns

Based on the argument types, returns one of the following:

  - `nothing`: If both `risk_ucs` and `prior_ucs` are `nothing`.
  - `risk_ucs::UcSE_UcS`: If `risk_ucs` is not `nothing`.
  - `prior_ucs::UcSE_UcS`: If `risk_ucs` is `nothing` but `prior_ucs` is not `nothing`.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`factory`](@ref)
"""
function ucs_selector(::Nothing, ::Nothing)::Nothing
    return nothing
end
function ucs_selector(risk_ucs::UcSE_UcS, ::Any)::UcSE_UcS
    return risk_ucs
end
function ucs_selector(::Nothing, prior_ucs::UcSE_UcS)::UcSE_UcS
    return prior_ucs
end
"""
    ucs_view(risk_ucs, i)

Get a view or subset of an uncertainty set for asset cluster index `i`.

Returns the uncertainty set sliced for the given index, or unchanged for estimator types. Used in hierarchical optimisation to apply uncertainty sets per cluster.

# Arguments

  - `risk_ucs`: Uncertainty set result, estimator, or `nothing`.
  - `i`: Cluster or asset index.

# Returns

  - Sliced uncertainty set or unchanged value.

# Related

  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function ucs_view(risk_ucs::Option{<:AbstractUncertaintySetEstimator},
                  ::Any)::Option{<:AbstractUncertaintySetEstimator}
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
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError)
    if isa(uc.pe, AbstractHiLoOrderPriorEstimator_F)
        @argcheck(!isnothing(rd.F), IsNothingError)
    end
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
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError)
    if isa(uc.pe, AbstractHiLoOrderPriorEstimator_F)
        @argcheck(!isnothing(rd.F), IsNothingError)
    end
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
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError)
    if isa(uc.pe, AbstractHiLoOrderPriorEstimator_F)
        @argcheck(!isnothing(rd.F), IsNothingError)
    end
    return sigma_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for constructing box uncertainty sets in portfolio optimisation.
Box uncertainty sets model uncertainty by specifying lower and upper bounds for risk or prior statistics.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
"""
struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Represents a box uncertainty set for risk or prior statistics in portfolio optimisation.
Stores lower and upper bounds for the uncertain quantity, such as expected returns or covariance.

# Mathematical definition

```math
\\begin{align}
U^{\\text{box}}_{\\boldsymbol{\\mu}} &= \\left\\{ \\boldsymbol{\\mu}\\, \\vert\\, \\lvert \\boldsymbol{\\mu} - \\boldsymbol{\\hat{\\mu}} \\rvert \\leq \\delta \\right\\} \\\\
U^{\\text{box}}_{\\mathbf{\\Sigma}} &= \\left\\{ \\mathbf{\\Sigma}\\, \\vert\\, \\mathbf{\\Sigma}_{l} \\leq \\mathbf{\\Sigma} \\leq \\mathbf{\\Sigma}_{u},\\, \\mathbf{\\Sigma} \\succeq 0 \\right\\}\\,.
\\end{align}
```

Where:

  - ``U^{\\text{box}}_{\\boldsymbol{\\mu}}``: Box uncertainty set for expected returns.
  - ``U^{\\text{box}}_{\\mathbf{\\Sigma}}``: Box uncertainty set for the covariance matrix.
  - ``\\boldsymbol{\\mu}``, ``\\mathbf{\\Sigma}``: Uncertain expected returns and covariance.
  - ``\\boldsymbol{\\hat{\\mu}}``: Estimated (reference) mean vector.
  - ``\\delta``: Half-width of the box (element-wise).
  - ``\\mathbf{\\Sigma}_{l}``, ``\\mathbf{\\Sigma}_{u}``: Lower and upper bounds for the covariance matrix.
  - ``\\mathbf{\\Sigma} \\succeq 0``: Positive semi-definiteness constraint.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BoxUncertaintySet(;
        lb::ArrNum,
        ub::ArrNum
    ) -> BoxUncertaintySet

Keywords correspond to the struct's fields.

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
  - [`EllipsoidalUncertaintySet`](@ref)
"""
@concrete struct BoxUncertaintySet <: AbstractUncertaintySetResult
    "$(field_dict[:lb])"
    lb
    "$(field_dict[:ub])"
    ub
    function BoxUncertaintySet(lb::ArrNum, ub::ArrNum)
        @argcheck(!isempty(lb))
        @argcheck(!isempty(ub))
        @argcheck(size(lb) == size(ub))
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function BoxUncertaintySet(; lb::ArrNum, ub::ArrNum)::BoxUncertaintySet
    return BoxUncertaintySet(lb, ub)
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:VecNum, <:VecNum}, i)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i))
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:MatNum, <:MatNum}, i)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i, i), ub = view(risk_ucs.ub, i, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for computing the scaling parameter `k` for ellipsoidal uncertainty sets under the assumption of normally distributed returns in portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormalKUncertaintyAlgorithm(;
        kwargs::NamedTuple = (;)
    )

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
@concrete struct NormalKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm
    "$(field_dict[:kwargs])"
    kwargs
    function NormalKUncertaintyAlgorithm(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function NormalKUncertaintyAlgorithm(;
                                     kwargs::NamedTuple = (;))::NormalKUncertaintyAlgorithm
    return NormalKUncertaintyAlgorithm(kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for computing the scaling parameter `k` for ellipsoidal uncertainty sets using a general formula `sqrt((1 - q) / q)`, this ignores the distribution of the underlying data.

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
"""
struct GeneralKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for computing the scaling parameter `k` for ellipsoidal uncertainty sets using the chi-squared distribution in portfolio optimisation.

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
"""
struct ChiSqKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
"""
    k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::MatNum, sigma_X::MatNum)
    k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    k_ucs(::ChiSqKUncertaintyAlgorithm, q::Number, X::ArrNum, args...)
    k_ucs(type::Number, args...)

Computes the scaling parameter `k` for ellipsoidal uncertainty sets in portfolio optimisation.

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

      + Normal: `1 - q`-th quantile of the Mahalanobis distances.
      + General: formula `sqrt((1 - q) / q)`.
      + Chi-squared: `1 - q`-th quantile of the chi-squared distribution.
      + Number: returns the provided value directly.

  - Supports multiple dispatch for extensibility.

# Related

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
"""
function k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::MatNum, sigma_X::MatNum)
    k_mus = LinearAlgebra.diag(X * (sigma_X \ transpose(X)))
    return sqrt(Statistics.quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    return sqrt((one(q) - q) / q)
end
function k_ucs(::ChiSqKUncertaintyAlgorithm, q::Number, X::ArrNum, args...)
    return sqrt(Distributions.cquantile(Distributions.Chisq(size(X, 1)), q))
end
function k_ucs(type::Number, args...)::Number
    return type
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for constructing ellipsoidal uncertainty sets in portfolio optimisation.
Ellipsoidal uncertainty sets model uncertainty by specifying an ellipsoidal region for risk or prior statistics, typically using a covariance matrix and a scaling parameter.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EllipsoidalUncertaintySetAlgorithm(;
        method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
        diagonal::Bool = true
    ) -> EllipsoidalUncertaintySetAlgorithm

  - `method`: Sets the scaling algorithm or value for the ellipsoidal.
  - `diagonal`: Sets whether to use only diagonal elements.

# Examples

```jldoctest
julia> EllipsoidalUncertaintySetAlgorithm()
EllipsoidalUncertaintySetAlgorithm
    method ┼ ChiSqKUncertaintyAlgorithm()
  diagonal ┴ Bool: true
```

# Related

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
"""
@concrete struct EllipsoidalUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    "$(field_dict[:method_ucs])"
    method
    "$(field_dict[:diagonal])"
    diagonal
    function EllipsoidalUncertaintySetAlgorithm(method::Num_UcSK, diagonal::Bool)
        return new{typeof(method), typeof(diagonal)}(method, diagonal)
    end
end
function EllipsoidalUncertaintySetAlgorithm(;
                                            method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
                                            diagonal::Bool = true)::EllipsoidalUncertaintySetAlgorithm
    return EllipsoidalUncertaintySetAlgorithm(method, diagonal)
end
"""
$(DocStringExtensions.TYPEDEF)

Defines the abstract interface for ellipsoidal uncertainty set result classes in portfolio optimisation.

Subtypes of this abstract type represent the class or category of ellipsoidal uncertainty sets, such as those for mean or covariance statistics. Used to distinguish between different types of ellipsoidal uncertainty set results.

# Related

  - [`MuEllipsoidalUncertaintySet`](@ref)
  - [`SigmaEllipsoidalUncertaintySet`](@ref)
"""
abstract type AbstractEllipsoidalUncertaintySetResultClass <: AbstractUncertaintySetResult end
"""
$(DocStringExtensions.TYPEDEF)

Represents the class identifier for mean ellipsoidal uncertainty sets in portfolio optimisation.

Used to distinguish ellipsoidal uncertainty sets that encode uncertainty for mean statistics, such as expected returns.

# Related

  - [`AbstractEllipsoidalUncertaintySetResultClass`](@ref)
  - [`SigmaEllipsoidalUncertaintySet`](@ref)
"""
struct MuEllipsoidalUncertaintySet <: AbstractEllipsoidalUncertaintySetResultClass end
"""
$(DocStringExtensions.TYPEDEF)

Represents the class identifier for covariance ellipsoidal uncertainty sets in portfolio optimisation.

Used to distinguish ellipsoidal uncertainty sets that encode uncertainty for covariance statistics, such as covariance matrices.

# Related

  - [`AbstractEllipsoidalUncertaintySetResultClass`](@ref)
  - [`MuEllipsoidalUncertaintySet`](@ref)
"""
struct SigmaEllipsoidalUncertaintySet <: AbstractEllipsoidalUncertaintySetResultClass end
"""
$(DocStringExtensions.TYPEDEF)

Represents an ellipsoidal uncertainty set for risk or prior statistics in portfolio optimisation.
Stores a covariance matrix, a scaling parameter, and a class identifier for the uncertain quantity, such as expected returns or covariance.

# Mathematical definition

```math
\\begin{align}
U^{\\text{ellip}}_{\\boldsymbol{\\mu}} &= \\left\\{ \\boldsymbol{\\mu}\\, \\vert\\, \\left( \\boldsymbol{\\mu} - \\boldsymbol{\\hat{\\mu}} \\right)^{\\intercal} \\mathbf{\\Sigma}^{-1}_{\\boldsymbol{\\mu}} \\left( \\boldsymbol{\\mu} - \\boldsymbol{\\hat{\\mu}} \\right) \\leq k^{2}_{\\boldsymbol{\\mu}} \\right\\} \\\\
U^{\\text{ellip}}_{\\mathbf{\\Sigma}} &= \\left\\{ \\mathbf{\\Sigma}\\, \\vert\\, \\left( \\text{vec}\\left(\\mathbf{\\Sigma}\\right) - \\text{vec}\\left(\\mathbf{\\hat{\\Sigma}} \\right) \\right)^{\\intercal} \\mathbf{\\Sigma}^{-1}_{\\mathbf{\\Sigma}} \\left( \\text{vec}\\left(\\mathbf{\\Sigma}\\right) - \\text{vec}\\left(\\mathbf{\\hat{\\Sigma}} \\right) \\right) \\leq k^{2}_{\\mathbf{\\Sigma}},\\, \\mathbf{\\Sigma} \\succeq 0 \\right\\}\\,.
\\end{align}
```

Where:

  - ``U^{\\text{ellip}}_{\\boldsymbol{\\mu}}``: Ellipsoidal uncertainty set for expected returns.
  - ``U^{\\text{ellip}}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for covariance matrix.
  - ``\\boldsymbol{\\mu}``, ``\\mathbf{\\Sigma}``: Uncertain expected returns and covariance.
  - ``\\boldsymbol{\\hat{\\mu}}``, ``\\mathbf{\\hat{\\Sigma}}``: Estimated reference mean and covariance.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Covariance matrix of estimation error in mean.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Covariance matrix of estimation error in covariance (vectorised).
  - ``k^{2}_{\\boldsymbol{\\mu}}``, ``k^{2}_{\\mathbf{\\Sigma}}``: Scaling parameters (squared ellipsoid radii).
  - ``\\text{vec}(\\cdot)``: Vectorisation operator (column-stacking).
  - ``\\mathbf{\\Sigma} \\succeq 0``: Positive semi-definiteness constraint.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EllipsoidalUncertaintySet(;
        sigma::MatNum,
        k::Number,
        class::AbstractEllipsoidalUncertaintySetResultClass
    ) -> EllipsoidalUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(sigma)`.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `k > 0`.

# Examples

```jldoctest
julia> EllipsoidalUncertaintySet([1.0 0.2; 0.2 1.0], 2.5, SigmaEllipsoidalUncertaintySet())
EllipsoidalUncertaintySet
  sigma ┼ 2×2 Matrix{Float64}
      k ┼ Float64: 2.5
  class ┴ SigmaEllipsoidalUncertaintySet()
```

# Related

  - [`AbstractEllipsoidalUncertaintySetResultClass`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
"""
@concrete struct EllipsoidalUncertaintySet <: AbstractUncertaintySetResult
    "$(field_dict[:sigma])"
    sigma
    "$(field_dict[:k_ucs])"
    k
    "$(field_dict[:class_ucs])"
    class
    function EllipsoidalUncertaintySet(sigma::MatNum, k::Number,
                                       class::AbstractEllipsoidalUncertaintySetResultClass)
        @argcheck(!isempty(sigma))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(k > zero(k))
        return new{typeof(sigma), typeof(k), typeof(class)}(sigma, k, class)
    end
end
function EllipsoidalUncertaintySet(; sigma::MatNum, k::Number,
                                   class::AbstractEllipsoidalUncertaintySetResultClass)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(sigma, k, class)
end
function ucs_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                      <:SigmaEllipsoidalUncertaintySet},
                  i)::EllipsoidalUncertaintySet
    i = fourth_moment_index_generator(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipsoidalUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                     class = risk_ucs.class)
end
function ucs_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                      <:MuEllipsoidalUncertaintySet},
                  i)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                     class = risk_ucs.class)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySet,
       NormalKUncertaintyAlgorithm, GeneralKUncertaintyAlgorithm,
       ChiSqKUncertaintyAlgorithm, EllipsoidalUncertaintySetAlgorithm,
       EllipsoidalUncertaintySet, SigmaEllipsoidalUncertaintySet,
       MuEllipsoidalUncertaintySet
