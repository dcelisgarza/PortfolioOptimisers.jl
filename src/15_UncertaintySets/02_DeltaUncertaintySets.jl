"""
    struct DeltaUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetEstimator
        pe::T1
        dmu::T2
        dsigma::T3
    end

Estimator for box uncertainty sets using delta bounds on mean and covariance statistics in portfolio optimisation.

# Fields

  - `pe`: Prior estimator used to compute mean and covariance statistics.
  - `dmu`: Delta bound for expected returns (mean).
  - `dsigma`: Delta bound for covariance.

# Constructor

    DeltaUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(), dmu::Real = 0.1,
                        dsigma::Real = 0.1)

Keyword arguments correspond to the fields above.

## Validation

  - `dmu >= 0`.
  - `dsigma >= 0`.

# Examples

```jldoctest
julia> DeltaUncertaintySet()
DeltaUncertaintySet
      pe | EmpiricalPrior
         |        ce | PortfolioOptimisersCovariance
         |           |   ce | Covariance
         |           |      |    me | SimpleExpectedReturns
         |           |      |       |   w | nothing
         |           |      |    ce | GeneralCovariance
         |           |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         |           |      |       |    w | nothing
         |           |      |   alg | Full()
         |           |   mp | DefaultMatrixProcessing
         |           |      |       pdm | Posdef
         |           |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
         |           |      |   denoise | nothing
         |           |      |    detone | nothing
         |           |      |       alg | nothing
         |        me | SimpleExpectedReturns
         |           |   w | nothing
         |   horizon | nothing
     dmu | Float64: 0.1
  dsigma | Float64: 0.1
```

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractPriorEstimator`](@ref)
"""
struct DeltaUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetEstimator
    pe::T1
    dmu::T2
    dsigma::T3
    function DeltaUncertaintySet(pe::AbstractPriorEstimator, dmu::Real, dsigma::Real)
        @argcheck(dmu >= 0.0)
        @argcheck(dsigma >= 0.0)
        return new{typeof(pe), typeof(dmu), typeof(dsigma)}(pe, dmu, dsigma)
    end
end
function DeltaUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                             dmu::Real = 0.1, dsigma::Real = 0.1)
    return DeltaUncertaintySet(pe, dmu, dsigma)
end
"""
    ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
        F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for mean and covariance statistics using delta bounds from a prior estimator.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics (default: 1).
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `(mu_ucs::BoxUncertaintySet, sigma_ucs::BoxUncertaintySet)`: Expected returns and covariance uncertainty sets.

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
function ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    d_sigma = ue.dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(;
                             lb = range(zero(eltype(pr.mu)), zero(eltype(pr.mu));
                                        length = length(pr.mu)),
                             ub = ue.dmu * abs.(pr.mu) * 2),
           BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma)
end
"""
    mu_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
           F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for expected returns (mean) using delta bounds from a prior estimator.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator (default: `nothing`).
  - `dims`: Dimension along which to compute statistics (default: `1`).
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
function mu_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    return BoxUncertaintySet(;
                             lb = range(zero(eltype(pr.mu)), zero(eltype(pr.mu));
                                        length = length(pr.mu)),
                             ub = ue.dmu * abs.(pr.mu) * 2)
end
"""
    sigma_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
              F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for covariance using delta bounds from a prior estimator.

# Arguments

  - `ue`: Delta uncertainty set estimator. Provides delta bounds and prior estimator.
  - `X`: Data matrix (e.g., returns).
  - `F`: Optional factor matrix. Used by the prior estimator (default: `nothing`).
  - `dims`: Dimension along which to compute statistics (default: `1`).
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
function sigma_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    d_sigma = ue.dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma)
end

export DeltaUncertaintySet
