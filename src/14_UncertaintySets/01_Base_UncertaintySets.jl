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

Returns a pair of already-built uncertainty sets unchanged, so that a consumer can call [`ucs`](@ref) without first asking whether its slot holds an estimator or a result.

The method is a passthrough. It runs no procedure and it carries no `# Algorithm` section. Its sibling that takes an [`AbstractUncertaintySetEstimator`](@ref) is the method that fits.

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

Returns an already-built mean uncertainty set unchanged, so that a consumer can call [`mu_ucs`](@ref) without first asking whether its slot holds an estimator or a result.

The method is a passthrough. It runs no procedure and it carries no `# Algorithm` section. Its sibling that takes an [`AbstractUncertaintySetEstimator`](@ref) is the method that fits.

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

Returns an already-built covariance uncertainty set unchanged, so that a consumer can call [`sigma_ucs`](@ref) without first asking whether its slot holds an estimator or a result.

The method is a passthrough. It runs no procedure and it carries no `# Algorithm` section. Its sibling that takes an [`AbstractUncertaintySetEstimator`](@ref) is the method that fits.

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

Chooses between the uncertainty set a risk measure carries and the one a prior carries, so that the risk measure's own set outranks the prior's.

The function is a selector. It states the table below and it carries no `# Algorithm` section, because each of its three methods returns one of its arguments and takes no step. The three methods are exhaustive over the argument pairs the callers form, and the first row is the only one that gives `nothing`.

| `risk_ucs`   | `prior_ucs`  | Result      |
|:------------ |:------------ |:----------- |
| `nothing`    | `nothing`    | `nothing`   |
| a `UcSE_UcS` | anything     | `risk_ucs`  |
| `nothing`    | a `UcSE_UcS` | `prior_ucs` |

# Arguments

  - `risk_ucs`: Risk measure uncertainty set estimator or result, or `nothing`.
  - `prior_ucs`: Prior result uncertainty set estimator or result, or `nothing`.

# Returns

  - `ucs::Option{<:UcSE_UcS}`: The selected set or estimator, by the table above.

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
    port_opt_view(risk_ucs::Option{<:AbstractUncertaintySetEstimator}, i, args...)

Returns an uncertainty set estimator unchanged, because an estimator carries no asset axis to restrict.

The method is a passthrough. It runs no procedure and it carries no `# Algorithm` section. A hierarchical optimiser calls [`port_opt_view`](@ref) once per cluster, and an estimator that reaches this method is fitted later against the cluster's own returns, so the restriction happens in the fit rather than here. The methods that take a built result do index; each states its own steps.

# Arguments

  - `risk_ucs`: Uncertainty set estimator, or `nothing`.
  - `i`: Cluster or asset index (ignored).
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::Option{<:AbstractUncertaintySetEstimator}`: The input, unchanged.

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

Fits both uncertainty sets in one pass from an estimator and a [`ReturnsResult`](@ref).

The method unpacks the container and forwards to the matrix method, so an estimator that shares a prior or a simulation between the two axes computes it once.

# Algorithm

 1. Check that `rd.X` is not `nothing`, and raise otherwise.
 2. When `uc.pe` is an [`AbstractHiLoOrderPriorEstimator_F`](@ref), check that `rd.F` is not `nothing`, and raise otherwise. A factor prior reads the factor returns, and no other prior does.
 3. Forward to `ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)`, giving the pair of fitted sets. The implied volatility fields travel with the returns, because a prior that reads them takes them by keyword.

# Arguments

  - `uc`: Uncertainty set estimator. Used to construct the uncertainty set.
  - `rd`: [`ReturnsResult`](@ref). Contains the returns data and associated metadata.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - `!isnothing(rd.X)`, raising an `IsNothingError`.
  - If `uc.pe` is an [`AbstractHiLoOrderPriorEstimator_F`](@ref): `!isnothing(rd.F)`, raising an `IsNothingError`.

# Returns

  - `uc::Tuple{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetResult}`: Expected returns and covariance uncertainty sets.

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

Fits the mean uncertainty set from an estimator and a [`ReturnsResult`](@ref).

The method unpacks the container and forwards to the matrix method. A caller that needs both axes calls [`ucs`](@ref) instead, which fits them in one pass.

# Algorithm

 1. Check that `rd.X` is not `nothing`, and raise otherwise.
 2. When `uc.pe` is an [`AbstractHiLoOrderPriorEstimator_F`](@ref), check that `rd.F` is not `nothing`, and raise otherwise. A factor prior reads the factor returns, and no other prior does.
 3. Forward to `mu_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)`, giving the fitted mean set. The implied volatility fields travel with the returns, because a prior that reads them takes them by keyword.

# Arguments

  - `uc`: Uncertainty set estimator. Used to construct the expected returns uncertainty set.
  - `rd`: [`ReturnsResult`](@ref). Contains the returns data and associated metadata.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - `!isnothing(rd.X)`, raising an `IsNothingError`.
  - If `uc.pe` is an [`AbstractHiLoOrderPriorEstimator_F`](@ref): `!isnothing(rd.F)`, raising an `IsNothingError`.

# Returns

  - `uc::AbstractUncertaintySetResult`: Expected returns uncertainty set.

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

Fits the covariance uncertainty set from an estimator and a [`ReturnsResult`](@ref).

The method unpacks the container and forwards to the matrix method. An estimator with no covariance analogue raises there, as [`CharacteristicUncertaintySet`](@ref) does.

# Algorithm

 1. Check that `rd.X` is not `nothing`, and raise otherwise.
 2. When `uc.pe` is an [`AbstractHiLoOrderPriorEstimator_F`](@ref), check that `rd.F` is not `nothing`, and raise otherwise. A factor prior reads the factor returns, and no other prior does.
 3. Forward to `sigma_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)`, giving the fitted covariance set. The implied volatility fields travel with the returns, because a prior that reads them takes them by keyword.

# Arguments

  - `uc`: Uncertainty set estimator. Used to construct the covariance uncertainty set.
  - `rd`: [`ReturnsResult`](@ref). Contains the returns data and associated metadata.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Validation

  - `!isnothing(rd.X)`, raising an `IsNothingError`.
  - If `uc.pe` is an [`AbstractHiLoOrderPriorEstimator_F`](@ref): `!isnothing(rd.F)`, raising an `IsNothingError`.

# Returns

  - `uc::AbstractUncertaintySetResult`: Covariance uncertainty set.

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

**The two axes read the bounds differently, so a set fitted for one axis is not a set for the other.** On the mean axis [`set_ucs_return_constraints!`](@ref) reads the bounds only through their half-width ``(\\boldsymbol{u} - \\boldsymbol{\\ell}) / 2``, the ``\\delta_{\\boldsymbol{\\mu}}`` of Equation 11.14, and centres that width on `val`. Neither `lb` nor `ub` is a bound on the mean on its own, which is why two estimators write one set two ways and agree: [`ARCHUncertaintySet`](@ref) stores the two quantiles of the bootstrap mean, while [`DeltaUncertaintySet`](@ref) and the normal box write ``\\boldsymbol{\\ell} = \\boldsymbol{0}`` and put the whole width in ``\\boldsymbol{u}``. On the covariance axis [`set_ucs_variance_risk!`](@ref) reads ``\\operatorname{tr}(\\mathbf{A}_{u} \\mathbf{\\Sigma}_{u}) - \\operatorname{tr}(\\mathbf{A}_{l} \\mathbf{\\Sigma}_{l})`` under ``\\mathbf{A}_{u} - \\mathbf{A}_{l} = \\mathbf{W}``, so both bounds bind on their own and the covariance box is absolute. That route names no centre, so it never reads `val`.

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

The method takes the mean axis, where each bound carries one entry per asset, so the asset index applies to both bounds directly.

# Algorithm

 1. Take `view(risk_ucs.lb, i)` and `view(risk_ucs.ub, i)`, the two bounds restricted to the selected assets.
 2. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the centre restricted to the same assets, which passes a `nothing` through unchanged.
 3. Build a [`BoxUncertaintySet`](@ref) from the three views. The bounds stay a pair, so the half-width the mean route reads is the half-width of the restricted box.

# Arguments

  - `risk_ucs`: Vector-valued box uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::BoxUncertaintySet`: The set restricted to `i`.

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

The method takes the covariance axis, where each bound is an `N × N` matrix, so the asset index applies to both dimensions of each bound.

# Algorithm

 1. Take `view(risk_ucs.lb, i, i)` and `view(risk_ucs.ub, i, i)`, the two bounds restricted to the selected assets on both axes. Both stay symmetric, because the source bounds are symmetric and the same index is applied twice.
 2. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted covariance restricted to the same assets, which passes a `nothing` through unchanged.
 3. Build a [`BoxUncertaintySet`](@ref) from the three views. Both bounds bind on their own on this axis, so each is restricted rather than combined.

# Arguments

  - `risk_ucs`: Matrix-valued box uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::BoxUncertaintySet`: The set restricted to `i`.

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

The sample must be the estimation **error**, not the estimate. Under normality the centred Mahalanobis distance is a chi-squared variate, so this algorithm and [`ChiSqKUncertaintyAlgorithm`](@ref) compute one radius two ways and agree up to sampling noise. On 3000 draws of a 20-asset mean fitted over 252 observations the empirical radius lands within about one percent of ``5.6045``, which is ``\\sqrt{\\chi^{2,\\,-1}_{20}(0.95)}`` and depends on the dimension and the significance level alone. Feeding the raw estimates instead makes the distance **non-central**, and the radius then grows with the non-centrality ``T \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}``: on those same draws it rises to about ``7.3``, inflating by a third a radius that is meant to measure estimation error alone.

The quantile is taken against the shape matrix this algorithm is handed, not against the shape the estimator started from. [`ellipsoidal_set`](@ref) replaces the asymptotic covariance with its diagonal **before** it calls [`k_ucs`](@ref), so under the `diagonal = true` default the radius is a quantile of Mahalanobis distances measured against the diagonal shape. The two radii differ: on a 252-by-5 sample the full shape gives ``3.1673`` and its diagonal gives ``3.1819``.

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

It is the second branch of Equation 11.23 of the source, and it reads neither the data nor the shape matrix. The radius comes from Cantelli's one-sided Chebyshev inequality, so it holds for any law of the estimation errors that has the stated second moment. Use [`ChiSqKUncertaintyAlgorithm`](@ref) instead when the errors are normal, because the chi-squared radius is the tighter one there.

The guarantee is a bound in one direction and not a simultaneous region for the whole vector. It covers the scalar a robust row bounds — the worst-case value of ``\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\mu}`` along the weights the model picks — with probability at least ``1 - q``, whereas the chi-squared radius is a joint confidence region for every entry at once.

# Mathematical definition

```math
k = \\sqrt{\\dfrac{1 - q}{q}}\\,.
```

Where:

  - ``q``: Significance level.

Inverting the form gives ``\\left(1 + k^{2}\\right)^{-1} = q``, which is Cantelli's bound at ``k`` standard deviations. So the radius is the smallest one whose distribution-free tail bound is exactly ``q``, and no assumption on the law tightens it.

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

The degrees of freedom is read from `size(sigma_X, 1)`, the first dimension of the shape matrix. That is ``N`` on the mean axis, where the shape matrix is the asymptotic covariance of the mean, and ``N^{2}`` on the covariance axis, where it is the asymptotic covariance of the vectorised covariance. The same algorithm therefore gives a different radius on each axis.

**The source states this closed form for the mean axis only.** Equation 11.23 defines ``\\kappa^{2}_{\\boldsymbol{\\mu}}`` with ``n`` degrees of freedom, ``n`` being the number of assets, and obtains ``\\kappa^{2}_{\\mathbf{\\Sigma}}`` by simulation rather than in closed form. Applying the same form on the covariance axis is this library's extension of it, and the extension is **conservative**: a symmetric ``N \\times N`` matrix has ``N(N+1)/2`` free entries, and the normal method's shape matrix ``T \\left(\\mathbf{I} + \\mathbf{K}\\right) \\left(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}\\right)`` has exactly that rank, so ``N^{2}`` overstates the dimension of the ellipsoid it calibrates. At ``N = 20`` and ``q = 0.05`` the radius is ``21.157`` where the free-entry count gives ``15.646``. Use [`NormalKUncertaintyAlgorithm`](@ref) on the covariance axis to calibrate the radius on the sampled errors instead.

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

The two closed forms are the two branches of Equation 11.23 of the source, and the simulated route is its empirical counterpart. A plain `Number` in place of an algorithm is the radius itself.

# Algorithm

The first three methods each run one procedure. The fourth, `k_ucs(type::Number, args...)`, returns its own argument and takes no step, so it carries none of the numbered text below.

[`NormalKUncertaintyAlgorithm`](@ref):

 1. Form `k_mus = LinearAlgebra.diag(X * (sigma_X \\ transpose(X)))`, the squared Mahalanobis distance of every row of `X` against the shape matrix. The solve is done once for the whole sample rather than row by row.
 2. Take the `1 - q` quantile of `k_mus` under `km.kwargs`, and return its square root, the radius.

[`GeneralKUncertaintyAlgorithm`](@ref):

 1. Return `sqrt((one(q) - q) / q)`, the radius. The method reads neither `X` nor `sigma_X`, so both are absorbed by `args...`.

[`ChiSqKUncertaintyAlgorithm`](@ref):

 1. Read the degrees of freedom from `size(sigma_X, 1)`, the dimension of the ellipsoid.
 2. Return the square root of the `1 - q` chi-squared quantile at that many degrees of freedom, the radius. The method runs no simulation, so it ignores the sample container.

# Arguments

  - `km`: Scaling algorithm instance.
  - `q`: Significance level.
  - `X`: Matrix of estimation errors, one row per sample. **Every caller passes centred deviations, not levels**: each row is a deviation from the point estimate, and the method cannot check it. An uncentred sample makes the distance non-central and inflates the radius.
  - `sigma_X`: Shape matrix of the ellipsoid, and the shape the distances are measured against. It is ``N \\times N`` on the mean axis and ``N^{2} \\times N^{2}`` on the covariance axis. [`ellipsoidal_set`](@ref) passes the diagonal of the asymptotic covariance under its `diagonal = true` default, so the quantile is taken against that diagonal and not against the full matrix.
  - `args...`: Additional arguments, which the algorithms that need no sample absorb.
  - `type`: Number value for direct scaling.

# Returns

  - `k::Number`: Radius of the ellipsoid.

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

**`class` names the axis, and the axis fixes both the size of `sigma` and the index a view applies.** A [`MuEllipsoidalUncertaintySet`](@ref) carries an ``N \\times N`` shape matrix and takes the plain asset index. A [`SigmaEllipsoidalUncertaintySet`](@ref) carries an ``N^{2} \\times N^{2}`` one, because it bounds a vectorised covariance, so [`port_opt_view`](@ref) recovers ``N`` from the shape matrix and maps the asset index through [`fourth_moment_index_generator`](@ref) before it slices. The two consumers dispatch on the tag too, and the robust-return builder refuses a set that carries the covariance tag.

**A view carries `k` through unchanged, so it is not the set the same estimator would fit on the subset alone.** The restricted shape matrix does equal the one fitted on the subset, entry for entry, whenever the shape is diagonal. The radius does not, because two of the three algorithms calibrate it on the dimension or on the sample: on a four-asset universe restricted to two assets, [`ChiSqKUncertaintyAlgorithm`](@ref) gives ``3.0802`` on the view against ``2.4477`` on the subset fit, and [`NormalKUncertaintyAlgorithm`](@ref) gives ``3.0398`` against ``2.4242``. Only [`GeneralKUncertaintyAlgorithm`](@ref) agrees, because its radius reads neither the data nor the shape. A view is therefore the conservative choice, and a caller who wants the subset's own radius fits the subset.

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
  - If `val` is provided: `length(val) == size(sigma, 1)`. The rule reads a length rather than a size, so it holds on both axes: `val` is a characteristic vector of length ``N`` beside an ``N \\times N`` shape matrix, and an ``N \\times N`` covariance matrix beside an ``N^{2} \\times N^{2}`` one.

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

Return a view of a covariance [`EllipsoidalUncertaintySet`](@ref) restricted to assets at index `i`, mapping the sigma index through the fourth-moment index generator.

The set bounds a vectorised covariance, so its shape matrix lives on the ``N^{2}`` axis while its centre lives on the ``N`` axis. The method therefore applies two different indices, one to each field.

# Algorithm

 1. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted ``N \\times N`` covariance restricted to the selected assets. It takes the plain asset index, and a `nothing` passes through unchanged. The step runs first, because step 2 overwrites `i`.
 2. Recover `N` as `floor(Int, sqrt(size(risk_ucs.sigma, 1)))` from the shape matrix, and expand `i` with `fourth_moment_index_generator(N, i)`, giving the positions the selected assets occupy in the vectorised covariance.
 3. Take `view(risk_ucs.sigma, i, i)` under the expanded index, giving the restricted shape matrix.
 4. Build an [`EllipsoidalUncertaintySet`](@ref) from the two views, carrying `k` and `class` through unchanged. The radius is not recalibrated on the smaller dimension, so the view is more conservative than a fit on the subset under every radius algorithm except [`GeneralKUncertaintyAlgorithm`](@ref).

# Arguments

  - `risk_ucs`: Covariance ellipsoidal uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::EllipsoidalUncertaintySet`: The set restricted to `i`.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`SigmaEllipsoidalUncertaintySet`](@ref)
  - [`fourth_moment_index_generator`](@ref)
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

The set bounds a characteristic vector, so its shape matrix and its centre both live on the ``N`` axis and one index serves both.

# Algorithm

 1. Take `view(risk_ucs.sigma, i, i)`, the ``N \\times N`` shape matrix restricted to the selected assets on both dimensions.
 2. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted characteristic vector restricted to the same assets, which passes a `nothing` through unchanged.
 3. Build an [`EllipsoidalUncertaintySet`](@ref) from the two views, carrying `k` and `class` through unchanged. The radius is not recalibrated on the smaller dimension, so the view is more conservative than a fit on the subset under every radius algorithm except [`GeneralKUncertaintyAlgorithm`](@ref).

# Arguments

  - `risk_ucs`: Mean ellipsoidal uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::EllipsoidalUncertaintySet`: The set restricted to `i`.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`MuEllipsoidalUncertaintySet`](@ref)
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

Element-wise lower and upper quantile bounds for a symmetric ``N \\times N`` statistic.

Shared by the box [`ucs`](@ref) and [`sigma_ucs`](@ref) constructions across estimator families. The `get_ij` accessor is what lets one body serve them all: it bridges the Wishart sample container, a vector of matrices, and the bootstrap one, a three-dimensional array. Positive-definite projection, if any, is applied by the caller.

# Algorithm

 1. Allocate `lb` and `ub`, both `N × N` and of element type `TE`.
 2. For each ordered pair with `j <= i`, read `s_ij = get_ij(i, j)`, the sampled values of that entry.
 3. Write the `q` quantile of `s_ij` into both `lb[i, j]` and `lb[j, i]`, and the `1 - q` quantile into both `ub[i, j]` and `ub[j, i]`. Writing each quantile to both positions makes the two bounds symmetric by construction, and it costs one quantile per pair rather than two.
 4. Return `lb` and `ub`. They satisfy `lb .<= ub` entrywise, because `q` is the smaller quantile level of the same sample, and the sample mean of the statistic lies between them.

# Arguments

  - `TE`: Element type of the two bounds.
  - `get_ij`: Accessor. `get_ij(i, j)` returns the vector of sampled values for entry ``(i, j)``.
  - `N`: Side of the statistic.
  - `q`: Significance level, already halved by the caller.
  - `kwargs`: Splatted into `Statistics.quantile`.

# Returns

  - `lb::Matrix{TE}`: Element-wise lower bound, symmetric.
  - `ub::Matrix{TE}`: Element-wise upper bound, symmetric.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
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

Element-wise lower and upper quantile bounds for a vector-valued statistic.

Shared by the bootstrap box [`ucs`](@ref) and [`mu_ucs`](@ref) mean constructions. **The sample axis is the second one**: the body reads `mus[j, :]`, so `mus` is ``N \\times M``, one row per component and one column per sample. A caller that passes the transpose gets bounds of length ``M``, which the [`BoxUncertaintySet`](@ref) constructor accepts, so the axis is a contract this method cannot check.

# Algorithm

 1. Read `N = size(mus, 1)`, the number of components, and allocate `lb` and `ub` of that length and of `eltype(mus)`.
 2. For each component `j`, read the row `mu_j = mus[j, :]`, the `M` sampled values of that component.
 3. Write the `q` quantile of `mu_j` into `lb[j]` and the `1 - q` quantile into `ub[j]`.
 4. Return `lb` and `ub`. They satisfy `lb .<= ub` entrywise, and they bracket the sample mean of each component.

# Arguments

  - `mus`: Sampled values, ``N \\times M``, one row per component.
  - `q`: Significance level, already halved by the caller.
  - `kwargs`: Splatted into `Statistics.quantile`.

# Returns

  - `lb::Vector`: Element-wise lower bound, length ``N``.
  - `ub::Vector`: Element-wise upper bound, length ``N``.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
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

Assemble an [`EllipsoidalUncertaintySet`](@ref) from an already-computed asymptotic covariance `cov`.

Shared by every ellipsoidal [`ucs`](@ref), [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) construction across estimator families. [`k_ucs`](@ref) absorbs the trailing arguments its own algorithm does not read, so `samples` may be the deviation matrix, a `1:n_sim` range, or `nothing`, whichever the caller has.

**The order of the two steps below is load-bearing.** The diagonal is taken *before* the radius is fitted, so under the `diagonal = true` default an empirical radius is a quantile of Mahalanobis distances measured against the diagonal shape and not against the full one. On a 252-by-5 sample the full shape gives ``3.1673`` and its diagonal gives ``3.1819``. Taking the diagonal afterwards would pair a radius calibrated on one shape with a different shape, and the set would not hold the coverage its significance level names.

# Algorithm

 1. When `diagonal` is `true`, replace `cov` with `LinearAlgebra.Diagonal(cov)`, discarding the estimation-error correlations between entries. The result is stored as a `Diagonal`, not as a dense matrix.
 2. Compute `k = k_ucs(method, q, samples, cov)`, the radius, measured against whichever shape step 1 left.
 3. Build an [`EllipsoidalUncertaintySet`](@ref) from `cov`, `k`, `class` and `val`.

# Arguments

  - `diagonal`: Whether to restrict `cov` to its diagonal before the radius is fitted.
  - `method`: Radius algorithm, or the radius itself as a `Number`.
  - `q`: Significance level.
  - `samples`: Sampled estimation errors, or whatever container `method` reads. An algorithm that runs no simulation absorbs it.
  - `cov`: Asymptotic covariance of the statistic, which becomes the shape matrix.
  - `class`: Axis tag, which fixes the size of the shape matrix and the index a view applies.
  - `val`: Quantity the set is a neighbourhood of — the fitted characteristic vector on the mean axis, the fitted covariance on the covariance axis. Every caller has it in hand, because every one of them fits a prior before it calls here.

# Returns

  - `ucs::EllipsoidalUncertaintySet`: The assembled set.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
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
