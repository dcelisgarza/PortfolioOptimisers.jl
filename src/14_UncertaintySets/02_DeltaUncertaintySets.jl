"""
$(DocStringExtensions.TYPEDEF)

Fits a box uncertainty set by widening the prior statistics by a fixed fraction of their own absolute value.

It is the delta method of Equation 11.15 of the source, the one route in the family that draws no sample: `dmu` and `dsigma` are the two fractions. Its sampling counterparts are [`NormalUncertaintySet`](@ref) and [`ARCHUncertaintySet`](@ref). The two axes do not write the same kind of bound, so read [`mu_delta_box_set`](@ref) and [`sigma_delta_box_set`](@ref) before you read a fitted set entry by entry. A fraction of zero is admitted on either axis and collapses that axis to a degenerate box, which leaves the model with its nominal expression on that axis and no worst case at all.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DeltaUncertaintySet(;
        pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
        dmu::Number = 0.1,
        dsigma::Number = 0.1
    ) -> DeltaUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `dmu >= 0`.
  - `dsigma >= 0`.

# Examples

```jldoctest
julia> DeltaUncertaintySet()
DeltaUncertaintySet
      pe ┼ EmpiricalPrior
         │        ce ┼ PortfolioOptimisersCovariance
         │           │   ce ┼ Covariance
         │           │      │    me ┼ SimpleExpectedReturns
         │           │      │       │   w ┴ nothing
         │           │      │    ce ┼ GeneralCovariance
         │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │           │      │       │    w ┴ nothing
         │           │      │   alg ┼ FullMoment()
         │           │      │     w ┴ nothing
         │           │   mp ┼ MatrixProcessing
         │           │      │     pdm ┼ Posdef
         │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │           │      │      dn ┼ nothing
         │           │      │      dt ┼ nothing
         │           │      │     alg ┼ nothing
         │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │        me ┼ SimpleExpectedReturns
         │           │   w ┴ nothing
         │   horizon ┴ nothing
     dmu ┼ Float64: 0.1
  dsigma ┴ Float64: 0.1
```

# Related

  - [`BoxUncertaintySet`](@ref): the result both axes return, and the owner of the rule that the two axes read their bounds differently.
  - [`mu_delta_box_set`](@ref): the mean-axis builder, which writes a width.
  - [`sigma_delta_box_set`](@ref): the covariance-axis builder, which writes absolute bounds.
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractPriorEstimator`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.15.
"""
@concrete struct DeltaUncertaintySet <: AbstractUncertaintySetEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:dmu])
    """
    dmu
    """
    $(field_dict[:dsigma])
    """
    dsigma
    function DeltaUncertaintySet(pe::AbstractLowOrderPriorEstimator, dmu::Number,
                                 dsigma::Number)
        @argcheck(dmu >= 0.0, DomainError(dmu, "dmu must be >= 0"))
        @argcheck(dsigma >= 0.0, DomainError(dsigma, "dsigma must be >= 0"))
        return new{typeof(pe), typeof(dmu), typeof(dsigma)}(pe, dmu, dsigma)
    end
end
function DeltaUncertaintySet(; pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
                             dmu::Number = 0.1, dsigma::Number = 0.1)::DeltaUncertaintySet
    return DeltaUncertaintySet(pe, dmu, dsigma)
end
"""
    mu_delta_box_set(pr, dmu::Number) -> BoxUncertaintySet

Builds the mean-axis delta box, which writes a **width** on the upper bound and zero on the lower one.

Neither bound is a bound on the mean. [`BoxUncertaintySet`](@ref) owns the rule: on this axis [`set_ucs_return_constraints!`](@ref) reads the pair only through its half-width ``(\\boldsymbol{u} - \\boldsymbol{\\ell}) / 2`` and centres that width on `val`, so a builder is free to put the whole width in ``\\boldsymbol{u}``, and this one does. The set the model sees is therefore ``\\hat{\\boldsymbol{\\mu}} \\pm \\delta_{\\mu} \\lvert \\hat{\\boldsymbol{\\mu}} \\rvert``, the ``\\delta_{\\mu}`` of Equation 11.15. The zero lower bound is not a claim that the mean is non-negative, and `abs` fixes the width alone and never the centre: on ``\\hat{\\mu}_i = -0.6`` with ``\\delta_{\\mu} = 0.1`` the builder writes ``\\ell_i = 0`` and ``u_i = 0.12``, and the model sees ``[-0.66,\\, -0.54]``, centred on the negative value. Its covariance-axis sibling [`sigma_delta_box_set`](@ref) is on the other side of the same rule.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\ell} &= \\boldsymbol{0}\\,, \\\\
\\boldsymbol{u} &= 2 \\delta_{\\mu} \\lvert \\hat{\\boldsymbol{\\mu}} \\rvert\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bounds the builder writes.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - ``\\delta_{\\mu}``: Delta bound for expected returns.
  - ``\\lvert \\cdot \\rvert``: Element-wise absolute value.

# Algorithm

 1. Build `lb`, a range of `length(pr.mu)` zeros of the element type of `pr.mu`. It is a range and not a vector, because every entry holds the same value and the consumer reads it once.
 2. Build `ub`, twice the half-width the model is to see, from `dmu` and the element-wise absolute value of `pr.mu`. The factor of two is what makes the half-width come out at ``\\delta_{\\mu} \\lvert \\hat{\\boldsymbol{\\mu}} \\rvert``.
 3. Build a [`BoxUncertaintySet`](@ref) from `lb`, `ub` and `val = pr.mu`, the characteristic vector the width is centred on.

# Arguments

  - `pr`: Fitted prior. Only `pr.mu` is read.
  - `dmu`: Delta bound for expected returns. A `dmu` of zero writes `lb == ub == 0`, a half-width of zero, so the model's worst case collapses onto the nominal ``\\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w}``.

# Returns

  - `mu_ucs::BoxUncertaintySet`: The mean-axis box, whose `lb` is a range of zeros, whose `ub` holds twice the half-width, and whose `val` is `pr.mu`.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref): the owner of the two-axis convention this builder is one side of.
  - [`sigma_delta_box_set`](@ref): the other side, which writes absolute bounds.
  - [`set_ucs_return_constraints!`](@ref): the consumer that halves the difference.

# References

  - $(ref_dict[:cajas2025]) Equation 11.15.
"""
function mu_delta_box_set(pr, dmu::Number)
    return BoxUncertaintySet(;
                             lb = range(zero(eltype(pr.mu)), zero(eltype(pr.mu));
                                        length = length(pr.mu)), ub = dmu * abs.(pr.mu) * 2,
                             val = pr.mu)
end
"""
    sigma_delta_box_set(pr, dsigma::Number) -> BoxUncertaintySet

Builds the covariance-axis delta box, which writes **absolute bounds**, both of which bind in the model.

It is the other side of the rule [`BoxUncertaintySet`](@ref) owns. On this axis [`set_ucs_variance_risk!`](@ref) reads ``\\operatorname{tr}(\\mathbf{A}_{u} \\mathbf{\\Sigma}_{u}) - \\operatorname{tr}(\\mathbf{A}_{l} \\mathbf{\\Sigma}_{l})`` under ``\\mathbf{A}_{u},\\, \\mathbf{A}_{l} \\geq 0`` and ``\\mathbf{A}_{u} - \\mathbf{A}_{l} = \\mathbf{W}``, so each bound enters on its own and neither is halved against the other. That route names no centre, so it never reads `val`, and its mean-axis sibling is [`mu_delta_box_set`](@ref).

The map is element-wise, and it is not a scaling of the matrix: a positive entry of ``\\hat{\\mathbf{\\Sigma}}`` shrinks to ``1 - \\delta_{\\sigma}`` of itself in ``\\mathbf{\\Sigma}_{l}``, while a negative entry grows to ``1 + \\delta_{\\sigma}``. The order ``\\mathbf{\\Sigma}_{l} \\leq \\hat{\\mathbf{\\Sigma}} \\leq \\mathbf{\\Sigma}_{u}`` therefore holds entry by entry at every ``\\delta_{\\sigma}``, on a negative entry as much as on a positive one: ``-0.2`` with ``\\delta_{\\sigma} = 0.2`` gives ``-0.24`` and ``-0.16``. The **cone** order does not follow. This builder applies no `posdef!`, where its sampling sibling `sigma_normal_box_set` applies one to both bounds, so a large ``\\delta_{\\sigma}`` leaves ``\\mathbf{\\Sigma}_{l}`` indefinite. The consumer does not need it to be definite. It reads the two bounds entry by entry through the two traces above and factorises neither, so an indefinite lower bound builds, solves, and widens the worst case rather than breaking it. The library documents this rather than guarding it, so a ``\\delta_{\\sigma}`` chosen far outside ``(0, 1)`` is the caller's to justify.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{l} &= \\hat{\\mathbf{\\Sigma}} - \\delta_{\\sigma} \\lvert \\hat{\\mathbf{\\Sigma}} \\rvert\\,, \\\\
\\mathbf{\\Sigma}_{u} &= \\hat{\\mathbf{\\Sigma}} + \\delta_{\\sigma} \\lvert \\hat{\\mathbf{\\Sigma}} \\rvert\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{l}``, ``\\mathbf{\\Sigma}_{u}``: Lower and upper bounds for the covariance matrix.
  - $(math_dict[:Sigma_hat])
  - ``\\delta_{\\sigma}``: Delta bound for covariance.
  - ``\\lvert \\cdot \\rvert``: Element-wise absolute value.

# Algorithm

 1. Build `d_sigma`, the element-wise half-width, from `dsigma` and the element-wise absolute value of `pr.sigma`. It is non-negative everywhere, which is what orders the two bounds.
 2. Subtract `d_sigma` from `pr.sigma`, giving the lower bound. It is symmetric, because both operands are.
 3. Add `d_sigma` to `pr.sigma`, giving the upper bound.
 4. Build a [`BoxUncertaintySet`](@ref) from the two bounds and `val = pr.sigma`, the covariance they are calibrated on. The covariance route ignores `val`, which the mean route reads, so the field is carried for the reader and for ADR 0050 rather than for this consumer.

# Arguments

  - `pr`: Fitted prior. Only `pr.sigma` is read.
  - `dsigma`: Delta bound for covariance. A `dsigma` of zero writes `lb == ub == pr.sigma`, so the two traces collapse to ``\\operatorname{tr}(\\mathbf{W} \\hat{\\mathbf{\\Sigma}})`` and the model sees the nominal variance.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: The covariance-axis box, whose `lb` and `ub` are absolute bounds and whose `val` is `pr.sigma`.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref): the owner of the two-axis convention this builder is one side of.
  - [`mu_delta_box_set`](@ref): the other side, which writes a width.
  - [`set_ucs_variance_risk!`](@ref): the consumer that reads both bounds absolutely.

# References

  - $(ref_dict[:cajas2025]) Equation 11.15.
"""
function sigma_delta_box_set(pr, dsigma::Number)
    d_sigma = dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma,
                             val = pr.sigma)
end
"""
    ucs(ue::DeltaUncertaintySet, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for mean and covariance statistics using delta bounds from a prior estimator.

It fits the prior once and hands the one fit to both builders, where [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) fit it once each, so the single-axis pair costs two prior fits for the same two sets. The three verbs agree on their common axes to the last bit, because they call the same two builders on an identically fitted prior.

# Mathematical definition

Given prior mean ``\\hat{\\boldsymbol{\\mu}}`` and covariance ``\\hat{\\mathbf{\\Sigma}}``, the box bounds are:

```math
\\begin{align}
\\boldsymbol{\\mu}_{lb} &= \\boldsymbol{0}\\,, \\\\
\\boldsymbol{\\mu}_{ub} &= 2 \\delta_{\\mu} |\\hat{\\boldsymbol{\\mu}}|\\,.
\\end{align}
```

```math
\\begin{align}
\\mathbf{\\Sigma}_{lb} &= \\hat{\\mathbf{\\Sigma}} - \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,, \\\\
\\mathbf{\\Sigma}_{ub} &= \\hat{\\mathbf{\\Sigma}} + \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{lb}``, ``\\boldsymbol{\\mu}_{ub}``: Lower and upper bounds for expected returns.
  - ``\\mathbf{\\Sigma}_{lb}``, ``\\mathbf{\\Sigma}_{ub}``: Lower and upper bounds for covariance matrix.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - $(math_dict[:Sigma_hat])
  - ``\\delta_{\\mu}``: Delta bound for expected returns.
  - ``\\delta_{\\sigma}``: Delta bound for covariance.
  - ``|\\cdot|``: Element-wise absolute value.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, which carries the mean `pr.mu` and the covariance `pr.sigma` both builders read.
 2. Call [`mu_delta_box_set`](@ref) on `pr` and `ue.dmu`, giving the mean-axis box. It writes a width, so the model halves the difference of its bounds.
 3. Call [`sigma_delta_box_set`](@ref) on `pr` and `ue.dsigma`, giving the covariance-axis box. It writes absolute bounds, both of which bind.
 4. Return the two boxes as a tuple, the mean axis first.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set, whose bounds encode a width.
  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set, whose bounds are absolute.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`mu_delta_box_set`](@ref)
  - [`sigma_delta_box_set`](@ref)
  - [`mu_ucs`](@ref): the mean axis alone, which fits its own prior.
  - [`sigma_ucs`](@ref): the covariance axis alone, which fits its own prior.
"""
function ucs(ue::DeltaUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
             dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    return mu_delta_box_set(pr, ue.dmu), sigma_delta_box_set(pr, ue.dsigma)
end
"""
    mu_ucs(ue::DeltaUncertaintySet, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for expected returns (mean) using delta bounds from a prior estimator.

It fits its own prior, so it reaches the same set as the first element of [`ucs`](@ref) at the cost of a second fit. `ue.dsigma` is not read on this path.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}_{lb} &= \\boldsymbol{0}\\,, \\\\
\\boldsymbol{\\mu}_{ub} &= 2 \\delta_{\\mu} |\\hat{\\boldsymbol{\\mu}}|\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{lb}``, ``\\boldsymbol{\\mu}_{ub}``: Lower and upper bounds for expected returns.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - ``\\delta_{\\mu}``: Delta bound for expected returns.
  - ``|\\cdot|``: Element-wise absolute value.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, whose `pr.mu` is the only quantity this path reads.
 2. Call [`mu_delta_box_set`](@ref) on `pr` and `ue.dmu`, giving the mean-axis box, and return it.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator (default: `nothing`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set, whose bounds encode a width rather than a pair of bounds on the mean.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`mu_delta_box_set`](@ref): the builder this method forwards to, and the owner of the width convention.
  - [`ucs`](@ref): both axes on one prior fit.
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::DeltaUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    return mu_delta_box_set(pr, ue.dmu)
end
"""
    sigma_ucs(ue::DeltaUncertaintySet, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for covariance using delta bounds from a prior estimator.

It fits its own prior, so it reaches the same set as the second element of [`ucs`](@ref) at the cost of a second fit. `ue.dmu` is not read on this path.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{lb} &= \\hat{\\mathbf{\\Sigma}} - \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,, \\\\
\\mathbf{\\Sigma}_{ub} &= \\hat{\\mathbf{\\Sigma}} + \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{lb}``, ``\\mathbf{\\Sigma}_{ub}``: Lower and upper bounds for covariance matrix.
  - $(math_dict[:Sigma_hat])
  - ``\\delta_{\\sigma}``: Delta bound for covariance.
  - ``|\\cdot|``: Element-wise absolute value.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, whose `pr.sigma` is the only quantity this path reads.
 2. Call [`sigma_delta_box_set`](@ref) on `pr` and `ue.dsigma`, giving the covariance-axis box, and return it.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator (default: `nothing`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set, whose two bounds are absolute and bind on their own.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`sigma_delta_box_set`](@ref): the builder this method forwards to, and the owner of the absolute-bound convention and of the positive-semidefiniteness note.
  - [`ucs`](@ref): both axes on one prior fit.
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::DeltaUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                   dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    return sigma_delta_box_set(pr, ue.dsigma)
end

export DeltaUncertaintySet
