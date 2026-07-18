"""
$(DocStringExtensions.TYPEDEF)

Estimator for box uncertainty sets using delta bounds on mean and covariance statistics in portfolio optimisation.

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
         │           │      │   alg ┴ FullMoment()
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

  - [`BoxUncertaintySet`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractPriorEstimator`](@ref)
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
    mu_delta_box_set(pr, dmu::Number)

Box uncertainty set for expected returns from delta bounds: zero lower bound and
`2 * dmu * abs.(pr.mu)` upper bound. Shared by the delta [`ucs`](@ref)/[`mu_ucs`](@ref)
constructions.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`sigma_delta_box_set`](@ref)
"""
function mu_delta_box_set(pr, dmu::Number)
    return BoxUncertaintySet(;
                             lb = range(zero(eltype(pr.mu)), zero(eltype(pr.mu));
                                        length = length(pr.mu)), ub = dmu * abs.(pr.mu) * 2)
end
"""
    sigma_delta_box_set(pr, dsigma::Number)

Box uncertainty set for covariance from delta bounds: `pr.sigma ± dsigma * abs.(pr.sigma)`.
Shared by the delta [`ucs`](@ref)/[`sigma_ucs`](@ref) constructions.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`mu_delta_box_set`](@ref)
"""
function sigma_delta_box_set(pr, dsigma::Number)
    d_sigma = dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma)
end
"""
    ucs(ue::DeltaUncertaintySet, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for mean and covariance statistics using delta bounds from a prior estimator.

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
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - ``\\delta_{\\mu}``: Delta bound for expected returns.
  - ``\\delta_{\\sigma}``: Delta bound for covariance.
  - ``|\\cdot|``: Element-wise absolute value.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Constructs mean uncertainty set with lower bound at zero and upper bound at `2 * dmu * abs.(pr.mu)`.
  - Constructs covariance uncertainty set with bounds at `pr.sigma ± d_sigma`, where `d_sigma = dsigma * abs.(pr.sigma)`.
  - Returns both sets as a tuple.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
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

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator (default: `nothing`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Constructs mean uncertainty set with lower bound at zero and upper bound at `2 * dmu * abs.(pr.mu)`.
  - Ignores additional arguments and keyword arguments except those passed to the prior estimator.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`ucs`](@ref)
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

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{lb} &= \\hat{\\mathbf{\\Sigma}} - \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,, \\\\
\\mathbf{\\Sigma}_{ub} &= \\hat{\\mathbf{\\Sigma}} + \\delta_{\\sigma} |\\hat{\\mathbf{\\Sigma}}|\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{lb}``, ``\\mathbf{\\Sigma}_{ub}``: Lower and upper bounds for covariance matrix.
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - ``\\delta_{\\sigma}``: Delta bound for covariance.
  - ``|\\cdot|``: Element-wise absolute value.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator (default: `nothing`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Constructs covariance uncertainty set with lower bound at `pr.sigma - d_sigma` and upper bound at `pr.sigma + d_sigma`, where `d_sigma = dsigma * abs.(pr.sigma)`.
  - Ignores additional arguments and keyword arguments except those passed to the prior estimator.

# Related

  - [`DeltaUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::DeltaUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                   dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    return sigma_delta_box_set(pr, ue.dsigma)
end

export DeltaUncertaintySet
