"""
$(DocStringExtensions.TYPEDEF)

Fits an uncertainty set around a prior statistic, so that a downstream model can take the worst case over it.

All concrete subtypes should subtype `AbstractUncertaintySetEstimator`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractUncertaintySetEstimator` and implement the following methods:

## `mu_ucs`

  - `mu_ucs(ue::AbstractUncertaintySetEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...) -> AbstractUncertaintySetResult`: Fits the uncertainty set of the mean.

## `sigma_ucs`

  - `sigma_ucs(ue::AbstractUncertaintySetEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...) -> AbstractUncertaintySetResult`: Fits the uncertainty set of the covariance. An estimator with no covariance analogue throws instead, as [`CharacteristicUncertaintySet`](@ref) does.

## `ucs`

  - `ucs(ue::AbstractUncertaintySetEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...) -> Tuple`: Fits both sets in one pass, so that a shared prior or a shared simulation is computed once.

### Arguments

  - `ue`: The concrete subtype instance.
  - `X`: Matrix of asset returns.
  - `F`: Optional matrix of factor returns, which a factor prior needs.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to the prior estimator.

### Returns

  - `ucs::AbstractUncertaintySetResult`: The fitted set, or a tuple of the mean set and the covariance set for `ucs`.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`DeltaUncertaintySet`](@ref)
  - [`NormalUncertaintySet`](@ref)
"""
abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Selects which shape of uncertainty set an estimator builds, such as a box or an ellipsoid.

All concrete subtypes should subtype `AbstractUncertaintySetAlgorithm`. A subtype carries the parameters of its own shape and nothing else. The estimator does the fitting.

# Interfaces

A subtype is a tag that the estimator dispatches on, so it declares no method of its own. To add a shape, subtype `AbstractUncertaintySetAlgorithm` and add the `ucs`, `mu_ucs`, and `sigma_ucs` methods of [`AbstractUncertaintySetEstimator`](@ref) that are specialised on it, one set for each estimator that is to offer the shape.

# Related

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Carries a fitted uncertainty set, which is the data a worst-case model reads to build its robust expression.

All concrete subtypes should subtype `AbstractUncertaintySetResult`. A subtype also carries the statistic its bounds were calibrated on, so that the consumer bounds that statistic and not an unrelated one. See ADR 0050.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractUncertaintySetResult` and implement the following method:

## `port_opt_view`

  - `port_opt_view(risk_ucs::AbstractUncertaintySetResult, i, args...) -> AbstractUncertaintySetResult`: Returns the set restricted to the asset indices `i`. A hierarchical optimiser calls it once per cluster.

### Arguments

  - `risk_ucs`: The concrete subtype instance.
  - `i`: Asset index of the cluster.
  - `args...`: Additional arguments.

### Returns

  - `risk_ucs::AbstractUncertaintySetResult`: The restricted set.

A model that is to take the worst case over the new shape also needs its own `set_ucs_return_constraints!` method, or its own `set_ucs_variance_risk!` method, or both.

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

Computes the radius `k` of an ellipsoidal uncertainty set, which is how far the true statistic may lie from its estimate.

All concrete subtypes should subtype `AbstractUncertaintyKAlgorithm`. A plain number in place of one is the radius itself.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractUncertaintyKAlgorithm` and implement the following method:

## `k_ucs`

  - `k_ucs(km::AbstractUncertaintyKAlgorithm, q::Number, X, sigma_X::MatNum) -> Number`: Returns the radius.

### Arguments

  - `km`: The concrete subtype instance.
  - `q`: Significance level.
  - `X`: Matrix of sampled estimation errors, one row per sample. An algorithm that runs no simulation absorbs it.
  - `sigma_X`: Shape matrix of the ellipsoid, whose first dimension is the dimension of the ellipsoid.

### Returns

  - `k::Number`: Radius of the ellipsoid.

# Related

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
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
$(DocStringExtensions.TYPEDEF)

Computes the radius `eps` of an ``\\ell_1`` uncertainty set, which controls how far the true characteristic vector may lie from its estimate, and therefore how many assets the portfolio holds.

All concrete subtypes should subtype `AbstractUncertaintyEpsAlgorithm`. A plain number in place of one is the radius itself. It is the counterpart of [`AbstractUncertaintyKAlgorithm`](@ref) for the ``\\ell_1`` family.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractUncertaintyEpsAlgorithm` and implement the following method:

## `l1_resolve_eps`

  - `l1_resolve_eps(method::AbstractUncertaintyEpsAlgorithm, mus::VecNum, sds::Option{<:VecNum}, paired::Bool) -> Number`: Returns the radius.

### Arguments

  - `method`: The concrete subtype instance.
  - `mus`: Characteristic vector, sorted in non-increasing order.
  - `sds`: Per-asset scaling under the same permutation, or `nothing` when the set is unscaled.
  - `paired`: Whether to read the paired ladder of the dollar-neutral problem rather than the long-only one.

### Returns

  - `eps::Number`: Radius of the set.

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintyKAlgorithm`](@ref)
"""
abstract type AbstractUncertaintyEpsAlgorithm <: AbstractAlgorithm end
"""
    const Num_UcSEps = Union{<:AbstractUncertaintyEpsAlgorithm, <:Number}

Alias for a union of ``\\ell_1`` uncertainty radius algorithm and numeric types. A plain number is the radius itself; an algorithm defers its computation to the data.

# Related

  - [`AbstractUncertaintyEpsAlgorithm`](@ref)
"""
const Num_UcSEps = Union{<:AbstractUncertaintyEpsAlgorithm, <:Number}
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
    port_opt_view(risk_ucs, i)

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
function port_opt_view(risk_ucs::Option{<:AbstractUncertaintySetEstimator}, ::Any,
                       args...)::Option{<:AbstractUncertaintySetEstimator}
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
        @argcheck(!isnothing(rd.F),
                  IsNothingError("this is a factor prior; it needs factor returns. ReturnsResult.F is nothing — populate F (e.g. via prices_to_returns on factor prices)."))
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
        @argcheck(!isnothing(rd.F),
                  IsNothingError("this is a factor prior; it needs factor returns. ReturnsResult.F is nothing — populate F (e.g. via prices_to_returns on factor prices)."))
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
        @argcheck(!isnothing(rd.F),
                  IsNothingError("this is a factor prior; it needs factor returns. ReturnsResult.F is nothing — populate F (e.g. via prices_to_returns on factor prices)."))
    end
    return sigma_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects a box uncertainty set, a convex polytope of element-wise bounds, from an estimator that can build either shape.

Its sibling [`EllipsoidalUncertaintySetAlgorithm`](@ref) selects the ellipsoid instead. The box carries no correlation between the entries it bounds, and its worst case is a linear or semidefinite programme rather than a second-order cone one.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 11.3.1.
"""
struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Holds the element-wise lower and upper bounds of a box uncertainty set on a mean vector or on a covariance matrix.

A box is a convex polytope, so it reads as a polyhedral confidence interval on the entries it bounds. Its worst case is Equation 11.19 on the mean axis and Equation 11.20 on the covariance axis of the source.

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
        ub::ArrNum,
        val::Option{<:ArrNum} = nothing
    ) -> BoxUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(lb)`.
  - `!isempty(ub)`.
  - `size(lb) == size(ub)`.
  - If `val` is provided: `size(val) == size(lb)`.

# Details

`val` is the quantity the set is a neighbourhood of. A set produced by [`ucs`](@ref), [`mu_ucs`](@ref) or [`sigma_ucs`](@ref) carries the fit its bounds were calibrated on, so the consumer bounds that quantity rather than an unrelated one. See ADR 0050.

The mean route uses `val` as the centre of the worst-case return. The covariance route does not read it: the worst-case variance over a box is `tr(A_u \\mathbf{\\Sigma}_u) - tr(A_l \\mathbf{\\Sigma}_l)`, which names no centre.

The mean route also reads the bounds only through their half-width ``(\\boldsymbol{u} - \\boldsymbol{\\ell}) / 2``, which is the ``\\delta_{\\mu}`` of Equation 11.14. Two estimators therefore write the same set two ways and agree: [`ARCHUncertaintySet`](@ref) stores the two quantiles of the bootstrap mean, while [`DeltaUncertaintySet`](@ref) stores ``\\boldsymbol{\\ell} = \\boldsymbol{0}`` and ``\\boldsymbol{u} = 2 \\delta_{\\mu}``. Neither `lb` nor `ub` is a bound on the mean on its own.

# Examples

```jldoctest
julia> BoxUncertaintySet(; lb = [0.1, 0.2], ub = [0.3, 0.4])
BoxUncertaintySet
   lb ┼ Vector{Float64}: [0.1, 0.2]
   ub ┼ Vector{Float64}: [0.3, 0.4]
  val ┴ nothing
```

# Related

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.14.
  - $(ref_dict[:sousalobo2000])
"""
@concrete struct BoxUncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:lb])
    """
    lb
    """
    $(field_dict[:ub])
    """
    ub
    """
    $(field_dict[:val_ucs])
    """
    val
    function BoxUncertaintySet(lb::ArrNum, ub::ArrNum, val::Option{<:ArrNum})
        @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
        @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
        @argcheck(size(lb) == size(ub),
                  DimensionMismatch("lb ($(size(lb))) must match ub ($(size(ub)))"))
        if isa(val, ArrNum)
            @argcheck(size(val) == size(lb),
                      DimensionMismatch("val ($(size(val))) must match lb ($(size(lb)))"))
        end
        return new{typeof(lb), typeof(ub), typeof(val)}(lb, ub, val)
    end
end
function BoxUncertaintySet(lb::ArrNum, ub::ArrNum)::BoxUncertaintySet
    return BoxUncertaintySet(lb, ub, nothing)
end
function BoxUncertaintySet(; lb::ArrNum, ub::ArrNum,
                           val::Option{<:ArrNum} = nothing)::BoxUncertaintySet
    return BoxUncertaintySet(lb, ub, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a vector [`BoxUncertaintySet`](@ref) restricted to the asset indices `i`.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::BoxUncertaintySet{<:VecNum, <:VecNum}, i,
                       args...)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i),
                             val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a matrix [`BoxUncertaintySet`](@ref) restricted to the asset indices `i`.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::BoxUncertaintySet{<:MatNum, <:MatNum}, i,
                       args...)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i, i), ub = view(risk_ucs.ub, i, i),
                             val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Fits the ellipsoid radius `k` empirically, as the `1 - q` quantile of the Mahalanobis distances of the sampled estimation errors.

The route makes no distributional assumption: it reads the errors the estimator family sampled, whether they come from a parametric draw or from a bootstrap resample. Its two closed-form siblings are [`ChiSqKUncertaintyAlgorithm`](@ref) and [`GeneralKUncertaintyAlgorithm`](@ref).

The sample must be the estimation **error**, not the estimate. Under normality the centred Mahalanobis distance is a chi-squared variate, so this algorithm and [`ChiSqKUncertaintyAlgorithm`](@ref) then compute the same radius two ways and must agree — on a 252-by-20 sample they give ``5.5989`` and ``5.6045``. Feeding the raw estimates instead makes the distance **non-central**, which inflates the radius by the whole non-centrality: ``7.3876`` on that same sample, at ``T \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}} = 16.17``.

# Mathematical definition

```math
k = \\sqrt{Q_{1-q}\\!\\left(\\left\\{ \\boldsymbol{\\delta}^{(m)\\intercal} \\mathbf{\\Sigma}_{\\boldsymbol{\\delta}}^{-1} \\boldsymbol{\\delta}^{(m)} \\right\\}_{m=1}^{M}\\right)}\\,.
```

Where:

  - ``\\boldsymbol{\\delta}^{(m)}``: The ``m``-th sampled estimation error, a row of the `X` argument of [`k_ucs`](@ref).
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\delta}}``: Shape matrix of the ellipsoid.
  - ``Q_{1-q}``: The `1 - q` quantile function.
  - ``M``: Number of samples.

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

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2.
"""
@concrete struct NormalKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm
    """
    $(field_dict[:kwargs])
    """
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

Computes the ellipsoid radius `k` as `sqrt((1 - q) / q)`, the closed form that holds for any distribution of the estimation errors.

It is the second branch of Equation 11.23 of the source, and it reads neither the data nor the shape matrix. Use [`ChiSqKUncertaintyAlgorithm`](@ref) instead when the errors are normal, because the chi-squared radius is the tighter one there.

# Mathematical definition

```math
k = \\sqrt{\\dfrac{1 - q}{q}}\\,.
```

Where:

  - ``q``: Significance level.

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.23.
  - $(ref_dict[:fabozzi2007])
"""
struct GeneralKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes the ellipsoid radius `k` as the square root of the `1 - q` chi-squared quantile, the closed form that holds when the estimation errors are normal.

The degrees of freedom is the dimension of the ellipsoid, which the shape matrix carries. It is ``N`` on the mean axis and ``N^{2}`` on the covariance axis, so the same algorithm gives a different radius on each. It is the first branch of Equation 11.23 of the source.

# Mathematical definition

```math
k = \\sqrt{\\chi^{2,\\,-1}_{p}(1 - q)}\\,, \\qquad p = \\operatorname{size}(\\mathbf{\\Sigma}_{\\boldsymbol{\\delta}}, 1)\\,.
```

Where:

  - ``\\chi^{2,\\,-1}_{p}``: Inverse cumulative distribution function of the chi-squared distribution with ``p`` degrees of freedom.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\delta}}``: Shape matrix of the ellipsoid.
  - ``q``: Significance level.

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.23.
  - $(ref_dict[:fabozzi2007])
"""
struct ChiSqKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
"""
    k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::MatNum, sigma_X::MatNum)
    k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    k_ucs(::ChiSqKUncertaintyAlgorithm, q::Number, ::Any, sigma_X::MatNum)
    k_ucs(type::Number, args...)

Compute the radius `k` of an ellipsoidal uncertainty set at significance level `q`.

# Arguments

  - `km`: Scaling algorithm instance.
  - `q`: Significance level.
  - `X`: Matrix of estimation errors, one row per sample. Each row is a deviation from the point estimate, not the estimate itself.
  - `sigma_X`: Shape matrix of the ellipsoid. It is ``N \\times N`` on the mean axis and ``N^{2} \\times N^{2}`` on the covariance axis.
  - `args...`: Additional arguments, which the algorithms that need no sample absorb.
  - `type`: Number value for direct scaling.

# Returns

  - `k::Number`: Radius of the ellipsoid.

# Details

Each algorithm reads a different source. The two closed forms are the two branches of Equation 11.23 of [cajas2025](@cite), and the two simulated routes are its empirical counterpart:

  - [`NormalKUncertaintyAlgorithm`](@ref): the square root of the `1 - q` quantile of the Mahalanobis distances `diag(X * inv(sigma_X) * X')` of the sampled estimation errors. It assumes nothing about the law of the errors, so it is the route the bootstrap family uses.
  - [`GeneralKUncertaintyAlgorithm`](@ref): the distribution-free closed form `sqrt((1 - q) / q)`.
  - [`ChiSqKUncertaintyAlgorithm`](@ref): the square root of the `1 - q` quantile of a chi-squared distribution whose degrees of freedom is `size(sigma_X, 1)`, the dimension of the ellipsoid. This route runs no simulation, so it ignores the sample container.
  - `Number`: returns the provided value directly.

# Related

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2.
  - $(ref_dict[:fabozzi2007])
"""
function k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::MatNum, sigma_X::MatNum)
    k_mus = LinearAlgebra.diag(X * (sigma_X \ transpose(X)))
    return sqrt(Statistics.quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    return sqrt((one(q) - q) / q)
end
function k_ucs(::ChiSqKUncertaintyAlgorithm, q::Number, ::Any, sigma_X::MatNum)
    # The degrees of freedom is the dimension of the ellipsoid, which the shape matrix
    # carries: N on the mean axis, N^2 on the covariance axis. The sample container is
    # unused, because this route runs no simulation.
    return sqrt(Distributions.cquantile(Distributions.Chisq(size(sigma_X, 1)), q))
end
function k_ucs(type::Number, args...)::Number
    return type
end
"""
$(DocStringExtensions.TYPEDEF)

Selects an ellipsoidal uncertainty set, and carries the radius algorithm and the diagonal switch it needs.

Its sibling [`BoxUncertaintySetAlgorithm`](@ref) selects the box instead. The ellipsoid reads the correlation between the entries it bounds through its shape matrix, which `diagonal = true` discards to remove the noise in the off-diagonal estimation errors.

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

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2.
"""
@concrete struct EllipsoidalUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    """
    $(field_dict[:method_ucs])
    """
    method
    """
    $(field_dict[:diagonal])
    """
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

Names the axis an [`EllipsoidalUncertaintySet`](@ref) lives on, which fixes the dimension of its shape matrix.

The family has exactly two inhabitants, and both ship. A consumer dispatches on the tag, because a mean ellipsoid and a covariance ellipsoid are the same struct with shape matrices of different size.

# Interfaces

A subtype is a tag that carries no field and declares no method of its own.

# Related

  - [`MuEllipsoidalUncertaintySet`](@ref)
  - [`SigmaEllipsoidalUncertaintySet`](@ref)
"""
abstract type AbstractEllipsoidalUncertaintySetResultClass <: AbstractUncertaintySetResult end
"""
$(DocStringExtensions.TYPEDEF)

Tags an [`EllipsoidalUncertaintySet`](@ref) as living on the mean axis, where the shape matrix is ``N \\times N``.

The tag is what the consumers dispatch on. `port_opt_view` slices such a set with the plain asset index, and the robust-return builder refuses a set that carries the covariance tag instead.

# Related

  - [`AbstractEllipsoidalUncertaintySetResultClass`](@ref)
  - [`SigmaEllipsoidalUncertaintySet`](@ref)
"""
struct MuEllipsoidalUncertaintySet <: AbstractEllipsoidalUncertaintySetResultClass end
"""
$(DocStringExtensions.TYPEDEF)

Tags an [`EllipsoidalUncertaintySet`](@ref) as living on the covariance axis, where the shape matrix is ``N^{2} \\times N^{2}``.

The tag is what the consumers dispatch on. `port_opt_view` maps the asset index through the fourth-moment index generator before it slices the shape matrix, because the ellipsoid bounds a vectorised covariance.

# Related

  - [`AbstractEllipsoidalUncertaintySetResultClass`](@ref)
  - [`MuEllipsoidalUncertaintySet`](@ref)
"""
struct SigmaEllipsoidalUncertaintySet <: AbstractEllipsoidalUncertaintySetResultClass end
"""
$(DocStringExtensions.TYPEDEF)

Holds the shape matrix, the radius, and the axis tag of an ellipsoidal uncertainty set on a mean vector or on a covariance matrix.

An ellipsoid is a Mahalanobis ball, so it reads as a confidence region that carries the correlation between the entries it bounds. Its worst case is Equation 11.25 on the mean axis and Equation 11.26 on the covariance axis of the source, and both are second-order cones.

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
        class::AbstractEllipsoidalUncertaintySetResultClass,
        val::Option{<:ArrNum} = nothing
    ) -> EllipsoidalUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(sigma)`.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `k > 0`.
  - If `val` is provided: `length(val) == size(sigma, 1)`.

# Details

`val` is the quantity the set is a neighbourhood of. A set produced by [`ucs`](@ref), [`mu_ucs`](@ref) or [`sigma_ucs`](@ref) carries the fit its shape matrix was calibrated on, so the consumer bounds that quantity rather than an unrelated one. See ADR 0050.

It is a characteristic vector of length ``N`` on the mean axis, and an ``N \\times N`` covariance matrix on the covariance axis, where the shape matrix is ``N^2 \\times N^2``. Both cases satisfy the length check.

# Examples

```jldoctest
julia> EllipsoidalUncertaintySet([1.0 0.2; 0.2 1.0], 2.5, SigmaEllipsoidalUncertaintySet())
EllipsoidalUncertaintySet
  sigma ┼ 2×2 Matrix{Float64}
      k ┼ Float64: 2.5
  class ┼ SigmaEllipsoidalUncertaintySet()
    val ┴ nothing
```

# Related

  - [`AbstractEllipsoidalUncertaintySetResultClass`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`k_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.22.
  - $(ref_dict[:fengpalomar2016])
"""
@concrete struct EllipsoidalUncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:sigma])
    """
    sigma
    """
    $(field_dict[:k_ucs])
    """
    k
    """
    $(field_dict[:class_ucs])
    """
    class
    """
    $(field_dict[:val_ucs])
    """
    val
    function EllipsoidalUncertaintySet(sigma::MatNum, k::Number,
                                       class::AbstractEllipsoidalUncertaintySetResultClass,
                                       val::Option{<:ArrNum})
        @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(k > zero(k), DomainError(k, "k must be positive"))
        if isa(val, ArrNum)
            @argcheck(length(val) == size(sigma, 1),
                      DimensionMismatch("val ($(length(val))) must match sigma ($(size(sigma, 1)))"))
        end
        return new{typeof(sigma), typeof(k), typeof(class), typeof(val)}(sigma, k, class,
                                                                         val)
    end
end
function EllipsoidalUncertaintySet(sigma::MatNum, k::Number,
                                   class::AbstractEllipsoidalUncertaintySetResultClass)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(sigma, k, class, nothing)
end
function EllipsoidalUncertaintySet(; sigma::MatNum, k::Number,
                                   class::AbstractEllipsoidalUncertaintySetResultClass,
                                   val::Option{<:ArrNum} = nothing)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(sigma, k, class, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a covariance [`EllipsoidalUncertaintySet`](@ref) restricted to assets at index `i`, mapping the sigma index via cokurtosis index generation.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                           <:SigmaEllipsoidalUncertaintySet},
                       i, args...)::EllipsoidalUncertaintySet
    # `val` is the N x N covariance the set is a neighbourhood of, so it takes the asset
    # index, whereas the N^2 x N^2 shape matrix takes the fourth-moment index.
    val = nothing_scalar_array_view(risk_ucs.val, i)
    i = fourth_moment_index_generator(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipsoidalUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                     class = risk_ucs.class, val = val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a mean [`EllipsoidalUncertaintySet`](@ref) restricted to assets at index `i`.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                           <:MuEllipsoidalUncertaintySet},
                       i, args...)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                     class = risk_ucs.class,
                                     val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
    box_quantile_bounds(::Type{TE}, get_ij, N::Integer, q::Number, kwargs) where {TE}

Element-wise lower/upper quantile bounds for a symmetric ``N \\times N`` statistic. `get_ij(i, j)` returns the vector of sampled values for entry ``(i, j)``; `q` is the (already halved)
significance level; `kwargs` is splatted into `Statistics.quantile`. Shared by the box
[`ucs`](@ref)/[`sigma_ucs`](@ref) constructions across estimator families — the accessor
bridges the Wishart (`Vector`-of-matrices) and bootstrap (3-D array) sample containers.
Positive-definite projection, if any, is applied by the caller.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`vec_quantile_bounds`](@ref)
"""
function box_quantile_bounds(::Type{TE}, get_ij, N::Integer, q::Number, kwargs) where {TE}
    lb = Matrix{TE}(undef, N, N)
    ub = Matrix{TE}(undef, N, N)
    for j in 1:N
        for i in j:N
            s_ij = get_ij(i, j)
            lb[j, i] = lb[i, j] = Statistics.quantile(s_ij, q; kwargs...)
            ub[j, i] = ub[i, j] = Statistics.quantile(s_ij, one(q) - q; kwargs...)
        end
    end
    return lb, ub
end
"""
    vec_quantile_bounds(mus::MatNum, q::Number, kwargs)

Element-wise lower/upper quantile bounds for a vector-valued statistic. `mus` is an ``N \\times M`` matrix of `M` samples per component; `q` is the (already halved) significance
level; `kwargs` is splatted into `Statistics.quantile`. Shared by the bootstrap box
[`ucs`](@ref)/[`mu_ucs`](@ref) mean constructions.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`box_quantile_bounds`](@ref)
"""
function vec_quantile_bounds(mus::MatNum, q::Number, kwargs)
    N = size(mus, 1)
    lb = Vector{eltype(mus)}(undef, N)
    ub = Vector{eltype(mus)}(undef, N)
    for j in 1:N
        mu_j = mus[j, :]
        lb[j] = Statistics.quantile(mu_j, q; kwargs...)
        ub[j] = Statistics.quantile(mu_j, one(q) - q; kwargs...)
    end
    return lb, ub
end
"""
    ellipsoidal_set(diagonal::Bool, method, q::Number, samples, cov::MatNum,
                    class::AbstractEllipsoidalUncertaintySetResultClass,
                    val::Option{<:ArrNum} = nothing)

Assemble an [`EllipsoidalUncertaintySet`](@ref) from an already-computed asymptotic
covariance `cov`. Optionally restricts `cov` to its diagonal, fits the scaling `k` via
[`k_ucs`](@ref) (which absorbs unused trailing arguments, so `samples` may be the deviation
matrix, a `1:n_sim` range, or `nothing` depending on `method`), and tags the result with
`class`. Shared by every ellipsoidal [`ucs`](@ref)/[`mu_ucs`](@ref)/[`sigma_ucs`](@ref)
construction across estimator families.

`val` is the quantity the set is a neighbourhood of — the fitted characteristic vector on
the mean axis, the fitted covariance on the covariance axis. Every caller has it in hand,
because every one of them fits a prior before it calls here.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`ucs`](@ref)
"""
function ellipsoidal_set(diagonal::Bool, method, q::Number, samples, cov::MatNum,
                         class::AbstractEllipsoidalUncertaintySetResultClass,
                         val::Option{<:ArrNum} = nothing)
    if diagonal
        cov = LinearAlgebra.Diagonal(cov)
    end
    k = k_ucs(method, q, samples, cov)
    return EllipsoidalUncertaintySet(; sigma = cov, k = k, class = class, val = val)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySet,
       NormalKUncertaintyAlgorithm, GeneralKUncertaintyAlgorithm,
       ChiSqKUncertaintyAlgorithm, EllipsoidalUncertaintySetAlgorithm,
       EllipsoidalUncertaintySet, SigmaEllipsoidalUncertaintySet,
       MuEllipsoidalUncertaintySet, AbstractUncertaintyEpsAlgorithm
