"""
$(DocStringExtensions.TYPEDEF)

Factor-based prior estimator for asset returns.

`FactorPrior` is a low order prior estimator that computes the mean and covariance of asset returns using a factor model. It combines a factor prior estimator, matrix post-processing, regression, and variance estimation to produce posterior moments. Optionally, it can add residual variance to the posterior covariance for robust estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorPrior(;
        pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        re::AbstractRegressionEstimator = StepwiseRegression(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        rsd::Bool = true
    ) -> FactorPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **lifts** a factor-axis prior onto the asset axis, reconstructing `X` as `F * transpose(M) .+ transpose(b)`, so it builds its carrier directly rather than forwarding one along its own axis; the rule of ADR 0046 still governs each field. It is the plain projection of the family — nothing here modifies the factor distribution, so [`FactorBlackLittermanPrior`](@ref) is this estimator with views landing on the factor block on the way through.

  - The factor block `fpr` **is** the wrapped factor prior, forwarded whole and untouched: it needs no reconstruction, because the asset moments are its projection rather than an update of it.
  - `mu` and `sigma` are that block projected through the loadings, so the returned carrier is **internally consistent**: `mu == rr.M * fpr.mu + rr.b` holds by construction. `sigma` optionally gains a residual correction when `rsd` is `true`.
  - `chol` is not forwarded but **rebuilt on the asset axis**, as `M * cholesky(fpr.sigma).L` widened by the residual block when `rsd` is `true`, so it stays in sync with the `sigma` it factorises.
  - `w` is the factor prior's, and is over the right axis: this estimator wraps only a factor prior, and `posterior_X` has exactly `F`'s rows, so it is the only weighting in existence. Its `ens`, `kld` and `ow` travel with it.
  - No `Z` is carried: the only wrapped prior is fit on factors, so its feature matrix would be factors × features and would not describe the asset axis. The drop is a *relocation* rather than a destruction — the factor prior is forwarded whole, so a feature matrix it carried is still reachable at `pr.fpr.Z`, which is where a factor-axis one belongs. For an asset-axis one, wrap this estimator from the *outside*: `FeaturePrior(; pe = FactorPrior(…), ze = RegressionFeatures())` reads the loadings back off the result.

# Examples

```jldoctest
julia> FactorPrior()
FactorPrior
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
   mp ┼ MatrixProcessing
      │     pdm ┼ Posdef
      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      dn ┼ nothing
      │      dt ┼ nothing
      │     alg ┼ nothing
      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
   re ┼ StepwiseRegression
      │   crit ┼ PValue
      │        │   t ┴ Float64: 0.05
      │    alg ┼ ForwardSelection()
      │    tgt ┼ LinearModel
      │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
   ve ┼ SimpleVariance
      │          me ┼ SimpleExpectedReturns
      │             │   w ┴ nothing
      │           w ┼ nothing
      │   corrected ┴ Bool: true
  rsd ┴ Bool: true
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`SimpleVariance`](@ref)
  - [`prior`](@ref)
"""
@propagatable @concrete struct FactorPrior <: AbstractLowOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    @fprop pe
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:re])
    """
    @fprop @vprop re
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:rsd])
    """
    rsd
    function FactorPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                         mp::AbstractMatrixProcessingEstimator,
                         re::AbstractRegressionEstimator, ve::AbstractVarianceEstimator,
                         rsd::Bool)
        return new{typeof(pe), typeof(mp), typeof(re), typeof(ve), typeof(rsd)}(pe, mp, re,
                                                                                ve, rsd)
    end
end
function FactorPrior(; pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                     mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                     re::AbstractRegressionEstimator = StepwiseRegression(),
                     ve::AbstractVarianceEstimator = SimpleVariance(),
                     rsd::Bool = true)::FactorPrior
    return FactorPrior(pe, mp, re, ve, rsd)
end
# Expose `:me` and `:ce` from the embedded asset prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties FactorPrior begin
    forward(pe, me, ce)
end
"""
    prior(pe::FactorPrior, X::MatNum, F::MatNum; dims::Int = 1, kwargs...)

Compute factor-based prior moments for asset returns using a factor model.

`prior` estimates the mean and covariance of asset returns using the specified factor prior estimator, regression, and matrix post-processing. The factor returns matrix `F` is used to compute factor moments, which are then mapped to asset space via regression. Optionally, residual variance is added to the posterior covariance for robust estimation. The result is returned as a [`LowOrderPrior`](@ref) object.

# Mathematical definition

The factor model maps factor moments to asset space via the loadings matrix ``\\mathbf{B}`` (with intercepts ``\\boldsymbol{\\alpha}``):

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{f}} + \\boldsymbol{\\alpha}\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix.
  - ``\\hat{\\boldsymbol{f}}``: ``K \\times 1`` vector of factor expected returns.
  - ``\\boldsymbol{\\alpha}``: ``N \\times 1`` vector of regression intercepts.
  - ``\\mathbf{\\Sigma}_f``: ``K \\times K`` factor covariance matrix.
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal matrix of residual variances (when `rsd = true`).

# Arguments

  - `pe`: Factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to matrix processing and estimators.

# Returns

  - `pr::LowOrderPrior`: Result object containing posterior asset returns, mean vector, covariance matrix, Cholesky factor, regression result, and factor moments.

# Validation

  - `dims in (1, 2)`.

# Related

  - [`FactorPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::FactorPrior, X::MatNum, F::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    f_prior = prior(pe.pe, F)
    f_mu, f_sigma = f_prior.mu, f_prior.sigma
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    posterior_mu = M * f_mu + b
    posterior_sigma = M * f_sigma * transpose(M)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_csigma = M * LinearAlgebra.cholesky(f_sigma).L
    if pe.rsd
        err = X - posterior_X
        err_sigma = LinearAlgebra.diagm(vec(Statistics.var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posdef!(pe.mp.pdm, posterior_sigma)
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    # No `Z` is forwarded: `f_prior` is fit on the factors, so its feature matrix would be
    # factors × features and would not describe the asset axis. To attach features here, wrap
    # this estimator — `FeaturePrior(; pe = FactorPrior(…), ze = RegressionFeatures())` reads
    # the loadings back off the result.
    #
    # The factor block *is* the prior that was fit on the factors: it needs no reconstruction,
    # because nothing here modifies the factor distribution — the asset moments are its
    # projection through `rr`.
    #
    # The asset-side `w` is the factor prior's: this estimator wraps only a factor prior, and
    # `posterior_X = F*M' + b'` has exactly `F`'s rows, so it is the only weighting in
    # existence and it is over the right observation axis. Its `ens`/`kld`/`ow` travel with it
    # — a weighting with no provenance cannot be interrogated (ADR 0046), and `ens` is what
    # sizes every uncertainty set built on this result.
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                  :)), w = f_prior.w, ens = f_prior.ens,
                         kld = f_prior.kld, ow = f_prior.ow, rr = rr, fpr = f_prior)
end

export FactorPrior
