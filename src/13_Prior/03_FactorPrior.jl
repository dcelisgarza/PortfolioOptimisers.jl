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
        re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        rsd::Bool = true
    ) -> FactorPrior

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `re`: Recursively updated via [`factory`](@ref).
  - `ve`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `re`: Recursively viewed via [`port_opt_view`](@ref).
  - `ve`: Recursively viewed via [`port_opt_view`](@ref).

## Composition: what this estimator forwards

This estimator **lifts** a factor-axis prior onto the asset axis, reconstructing `X` as `F * transpose(M) .+ transpose(b)`, so it builds its carrier directly rather than forwarding one along its own axis; the rule of ADR 0046 still governs each field. It is the plain projection of the family — nothing here modifies the factor distribution, so [`FactorBlackLittermanPrior`](@ref) is this estimator with views landing on the factor block on the way through.

  - The factor block `fpr` **is** the wrapped factor prior, forwarded whole and untouched: it needs no reconstruction, because the asset moments are its projection rather than an update of it.
  - `mu` and `sigma` are that block projected through the loadings, so the returned carrier is **internally consistent**: `mu == rr.M * fpr.mu + rr.b` holds by construction. `sigma` optionally gains a residual correction when `rsd` is `true`.
  - `chol` is not forwarded but **rebuilt on the asset axis**, as `M * cholesky(fpr.sigma).L` widened by the residual block when `rsd` is `true`, so it stays in sync with the `sigma` it factorises.
  - `w` is the factor prior's, and is over the right axis: this estimator wraps only a factor prior, and `posterior_X` has exactly `F`'s rows, so it is the only weighting in existence. Its `ens`, `kld` and `ow` travel with it.
  - No `pnl` is carried: the only wrapped prior is fit on factors, so its panel would be over the factors and would not describe the asset axis. The drop is a *relocation* rather than a destruction — the factor prior is forwarded whole, so a panel it carried is still reachable at `pr.fpr.pnl`, which is where a factor-axis one belongs. For an asset-axis one, wrap this estimator from the *outside*: `FeaturePrior(; pe = FactorPrior(…), ze = RegressionFeatures())` reads the loadings back off the result.

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
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`SimpleVariance`](@ref)
  - [`prior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.1, Equations 4.4 and 4.5.
  - $(ref_dict[:fan2008])
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
                         re::AbstractTimeSeriesRegressionEstimator,
                         ve::AbstractVarianceEstimator, rsd::Bool)
        return new{typeof(pe), typeof(mp), typeof(re), typeof(ve), typeof(rsd)}(pe, mp, re,
                                                                                ve, rsd)
    end
end
function FactorPrior(; pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                     mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                     re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
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
    factor_reconstruction(re::AbstractTimeSeriesRegressionEstimator, X::MatNum,
                          F::MatNum) -> Tuple{AbstractLoadingsRegressionResult, MatNum}

Fit the loadings and rebuild the asset returns from the factor returns.

This is the first half of the factor lift, and the only half every factor-axis estimator shares. It fits `re` on `(X, F)` and returns the regression result together with the *posterior returns matrix* `F * transpose(M) .+ transpose(b)` — the reconstruction that [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) each write into `LowOrderPrior.X`.

The second half — projecting the factor moments through the loadings — is [`factor_lift`](@ref). The two are separate because [`FactorBlackLittermanPrior`](@ref) needs the reconstruction before it has the moments to project: its views land on the factor distribution, so the factor moments only exist after the Black-Litterman update.

# Algorithm

 1. Fit `re` on `(X, F)` with [`regression`](@ref), giving `rr`. It carries the ``N \\times K`` loadings `rr.M` and the ``N \\times 1`` intercepts `rr.b`.
 2. Rebuild the asset returns as `F * transpose(rr.M) .+ transpose(rr.b)`, giving `posterior_X`, which has `F`'s rows and `X`'s columns.

# Arguments

  - $(arg_dict[:re])
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Returns

  - `rr::AbstractLoadingsRegressionResult`: Regression result carrying the loadings `M` and intercepts `b`.
  - `posterior_X::MatNum`: Reconstructed asset returns, `observations × assets`.

# Related

  - [`factor_lift`](@ref)
  - [`regression`](@ref)
  - [`FactorPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function factor_reconstruction(re::AbstractTimeSeriesRegressionEstimator, X::MatNum,
                               F::MatNum)
    rr = regression(re, X, F)
    return rr, F * transpose(rr.M) .+ transpose(rr.b)
end
"""
    factor_lift(mp::AbstractMatrixProcessingEstimator, ve::AbstractVarianceEstimator,
                rsd::Bool, rr::AbstractLoadingsRegressionResult, f_mu::VecNum, f_sigma::MatNum,
                X::MatNum, posterior_X::MatNum; kwargs...) -> NamedTuple

Project factor moments onto the asset axis through the regression loadings.

This is the second half of the factor lift, and it owns the algorithm that [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) both apply: map `f_mu` and `f_sigma` through the loadings, process the resulting covariance, and — when `rsd` is `true` — add the diagonal residual block and extend the Cholesky factor with it.

Which factor moments arrive is the caller's decision, and it is the only thing that differs between the two sites: [`FactorPrior`](@ref) passes the wrapped prior's moments unchanged, while [`FactorBlackLittermanPrior`](@ref) passes the Black-Litterman posterior moments.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{f}} + \\boldsymbol{\\alpha}\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` asset expected returns vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` asset covariance matrix.
  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix, `rr.M`.
  - ``\\boldsymbol{\\alpha}``: ``N \\times 1`` vector of regression intercepts, `rr.b`.
  - ``\\hat{\\boldsymbol{f}}``: ``K \\times 1`` factor expected returns vector, `f_mu`.
  - ``\\mathbf{\\Sigma}_f``: ``K \\times K`` factor covariance matrix, `f_sigma`.
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal matrix of residual variances, present only when `rsd` is `true`.

The returned `chol` is the transpose of ``[\\mathbf{B} \\mathbf{L}_f \\quad \\mathbf{\\Sigma}_\\varepsilon^{1/2}]``, where ``\\mathbf{L}_f`` is the lower Cholesky factor of ``\\mathbf{\\Sigma}_f``. It is therefore ``(K + N) \\times N`` when `rsd` is `true`, and ``K \\times N`` when `rsd` is `false`, the residual block being absent from `chol` and from ``\\hat{\\mathbf{\\Sigma}}`` alike.

``\\mathtt{chol}^\\intercal \\mathtt{chol} = \\hat{\\mathbf{\\Sigma}}`` holds **before** matrix processing, and that qualifier is load-bearing. `chol` is built from the `f_sigma` the caller passed, so step 4 of the algorithm rewrites `sigma` without rewriting `chol`. Under an `mp` that leaves the projected covariance where it found it — which the default [`MatrixProcessing`](@ref) does, its `pdm` being a no-op on a matrix that is already positive semi-definite — the two agree and the identity holds on the returned pair. Under an `mp` that denoises or detones, `sigma` moves and `chol` stays behind, so the identity holds against the unprocessed covariance alone. A consumer that needs a factor of the returned `sigma` must refactorise it.

# Algorithm

 1. Read the loadings `M` and the intercepts `b` off `rr`.
 2. Project the factor mean through the loadings, giving `posterior_mu`.
 3. Project the factor covariance through the loadings, giving `posterior_sigma`, the systematic block.
 4. Process `posterior_sigma` in place with [`matrix_processing!`](@ref), under `mp` and `posterior_X`.
 5. Carry the lower Cholesky factor of `f_sigma` through the loadings, giving `posterior_csigma`. This reads the `f_sigma` the caller passed, which step 4 does not touch.
 6. When `rsd` is `true`, take the reconstruction error `err = X - posterior_X`, and read `esigma`, the column variances of `err` under `ve`. Size the residual block as `err_sigma`, the diagonal matrix of those variances. When `rsd` is `false`, `esigma` is `nothing`.
 7. Still under `rsd`, add `err_sigma` to `posterior_sigma` and re-condition the sum with [`posdef!`](@ref), under `mp.pdm`. This is the body's only explicit [`posdef!`](@ref) call. `mp.pdm` also reaches `posterior_sigma` inside step 4, whenever `:pdm` is a member of `mp.order`.
 8. Still under `rsd`, widen `posterior_csigma` with `sqrt.(err_sigma)`, so the block that step 7 added to the covariance enters the factor as well.
 9. Reshape `posterior_csigma` to `length(posterior_mu)` columns, transpose it into `chol`, and return the four quantities.

# Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:ve])
  - $(arg_dict[:rsd])
  - `rr`: Regression result carrying the loadings `M` and intercepts `b`.
  - `f_mu`: Factor expected returns, `factors × 1`.
  - `f_sigma`: Factor covariance matrix, `factors × factors`.
  - $(arg_dict[:X])
  - `posterior_X`: Reconstructed asset returns from [`factor_reconstruction`](@ref).
  - `kwargs...`: Additional keyword arguments passed to matrix processing.

# Returns

  - `(; mu, sigma, chol, esigma)::NamedTuple`: Asset expected returns, asset covariance, the Cholesky-like factor whose trailing block is the residual standard deviations when `rsd` is `true`, and the residual variances themselves. `esigma` is `nothing` when `rsd` is `false`, because no residual block was added. A caller writes it onto the `esigma` field of the loadings result it returns, so that a consumer that needs the idiosyncratic variances reads them off the block instead of recomputing them from the reconstruction error.

# Related

  - [`factor_reconstruction`](@ref)
  - [`factor_residual_config`](@ref)
  - [`Regression`](@ref)
  - [`FactorPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function factor_lift(mp::AbstractMatrixProcessingEstimator, ve::AbstractVarianceEstimator,
                     rsd::Bool, rr::AbstractLoadingsRegressionResult, f_mu::VecNum,
                     f_sigma::MatNum, X::MatNum, posterior_X::MatNum; kwargs...)
    (; b, M) = rr
    posterior_mu = M * f_mu + b
    posterior_sigma = M * f_sigma * transpose(M)
    matrix_processing!(mp, posterior_sigma, posterior_X; kwargs...)
    posterior_csigma = M * LinearAlgebra.cholesky(f_sigma).L
    esigma = nothing
    if rsd
        err = X - posterior_X
        esigma = vec(Statistics.var(ve, err; dims = 1))
        err_sigma = LinearAlgebra.diagm(esigma)
        posterior_sigma .+= err_sigma
        posdef!(mp.pdm, posterior_sigma)
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return (; mu = posterior_mu, sigma = posterior_sigma,
            chol = transpose(reshape(posterior_csigma, length(posterior_mu), :)),
            esigma = esigma)
end
"""
    factor_residual_config(pe::AbstractPriorEstimator) -> Option{<:NamedTuple}

Declare how a prior estimator adds a residual block to the covariance it lifts.

A consumer that needs to *undo* the residual block — [`HighOrderFactorPriorEstimator`](@ref) subtracts it to recover the systematic covariance its residual cokurtosis correction is defined on — cannot read `ve` and `mp.pdm` off the wrapped estimator's fields. Its `pe` slot is bounded [`AbstractLowOrderPriorEstimator_F_AF`](@ref), and only [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) carry those fields; everything else in that bound is a wrapper or a pooling estimator, and a field access reaches past the type bound into a `FieldError`.

The declaration closes that gap. Every estimator answers, and it answers beside its own definition: the two that own a residual block report it, a wrapper forwards the answer of the estimator it wraps, and an estimator that adds none says so with an explicit `nothing` method. A pooling estimator is a wrapper for this purpose. It forwards the one estimator its moments come from, however many priors it pools. A `nothing` answer — and an answer whose `rsd` is `false` — both mean *no residual block was added*, so the consumer leaves the covariance alone.

There is no default. A silent `nothing` fallback cannot separate *this estimator adds no residual block* from *the author of this estimator forgot the method*, and the second reading drops a residual block the covariance really carries. An undeclared type therefore throws and names itself, which is the polarity [`range_tails`](@ref) already uses for a per-type declaration whose absence is a defect rather than an answer.

# Arguments

  - `pe`: Prior estimator.

# Validation

  - Throws an `ArgumentError` when the type of `pe` declares no method.

# Returns

  - `nothing`: The estimator adds no residual block.
  - `(; ve, pdm, rsd)::NamedTuple`: The variance estimator that sizes the residual block, the positive definite matrix estimator that re-conditions a covariance the block was removed from, and whether the block is added at all.

# Related

  - [`factor_lift`](@ref)
  - [`assert_factor_residual_config`](@ref)
  - [`range_tails`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`FactorPrior`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
"""
function factor_residual_config(pe::AbstractPriorEstimator)
    return throw(ArgumentError("`factor_residual_config` is not defined for `$(nameof(typeof(pe)))`. Every concrete `AbstractPriorEstimator` must declare its residual block beside its own definition by adding a method returning `(; ve, pdm, rsd)`, forwarding the estimator it wraps, or returning `nothing` when it adds no residual block."))
end
function factor_residual_config(pe::FactorPrior)
    return (; ve = pe.ve, pdm = pe.mp.pdm, rsd = pe.rsd)
end
"""
    assert_factor_residual_config(pe::AbstractPriorEstimator, cfg) -> Nothing

Check the shape of a [`factor_residual_config`](@ref) answer before a consumer reads it.

A consumer reaches `ve`, `pdm` and `rsd` off the returned value by property access, which has no shape check of its own: a declaration that returns the wrong thing surfaces as a `FieldError` deep inside the correction rather than at the declaration that caused it. This checks the two shapes the contract admits, and names the estimator that answered.

# Arguments

  - `pe`: Prior estimator that answered.
  - `cfg`: The answer of `factor_residual_config(pe)`.

# Validation

  - `cfg` is `nothing`, or a `NamedTuple` carrying `ve`, `pdm` and `rsd`.

# Related

  - [`factor_residual_config`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function assert_factor_residual_config(pe::AbstractPriorEstimator, cfg)::Nothing
    @argcheck(isnothing(cfg) ||
              (isa(cfg, NamedTuple) && all(k -> haskey(cfg, k), (:ve, :pdm, :rsd))),
              ArgumentError("`factor_residual_config(::$(nameof(typeof(pe))))` must return `nothing` or a `NamedTuple` carrying `ve`, `pdm` and `rsd`. Got\ncfg => $(cfg)::$(typeof(cfg))."))
    return nothing
end
"""
    prior(pe::FactorPrior, X::MatNum, F::MatNum; dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute factor-based prior moments for asset returns using a factor model.

`prior` estimates the mean and covariance of asset returns using the specified factor prior estimator, regression, and matrix post-processing. The factor returns matrix `F` is used to compute factor moments, which are then mapped to asset space via regression. Optionally, residual variance is added to the posterior covariance for robust estimation. The result is returned as a [`LowOrderPrior`](@ref) object.

# Mathematical definition

The factor model maps factor moments to asset space via the loadings matrix ``\\mathbf{B}`` (with intercepts ``\\boldsymbol{\\alpha}``):

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{f}} + \\boldsymbol{\\alpha}\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix, `rr.M`.
  - ``\\hat{\\boldsymbol{f}}``: ``K \\times 1`` vector of factor expected returns.
  - ``\\boldsymbol{\\alpha}``: ``N \\times 1`` vector of regression intercepts, `rr.b`.
  - ``\\mathbf{\\Sigma}_f``: ``K \\times K`` factor covariance matrix.
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal matrix of residual variances (when `rsd = true`).

The factor moments ``\\hat{\\boldsymbol{f}}`` and ``\\mathbf{\\Sigma}_f`` come from `pe.pe` fit on `F`, and the loadings from `pe.re` fit on `(X, F)`. The two equations are [`factor_lift`](@ref).

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. Fit the wrapped prior `pe.pe` on `F`, giving `f_prior`, the factor-axis prior result. `strict` reaches it, because `pe.pe` admits [`BlackLittermanPrior`](@ref) and [`EntropyPoolingPrior`](@ref), which resolve view names against a universe.
 3. Fit the loadings and rebuild the asset returns with [`factor_reconstruction`](@ref), giving `rr` and `posterior_X`.
 4. Project `f_prior.mu` and `f_prior.sigma` through `rr` with [`factor_lift`](@ref), giving `mu`, `sigma`, `chol` and `esigma`.
 5. Write `esigma` onto the `esigma` field of `rr`. Under `pe.rsd = true` the field holds the residual variances the lift measured, and under `pe.rsd = false` it holds `nothing`, because the lift added no residual block.
 6. Assemble a [`LowOrderPrior`](@ref) over `posterior_X`, with the oriented `X` under `o_X`, the three lifted moments, the factor prior's `w`, `ens`, `kld` and `ow`, the regression result under `rr`, and `f_prior` itself under `fpr`. No `Z` is carried; the composition note of [`FactorPrior`](@ref) says why.

# Arguments

  - `pe`: Factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - $(arg_dict[:strict])
  - `kwargs...`: Additional keyword arguments passed to matrix processing and estimators.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Result object containing posterior asset returns, mean vector, covariance matrix, Cholesky factor, regression result, and factor moments.

# Related

  - [`FactorPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`factor_reconstruction`](@ref)
  - [`factor_lift`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::FactorPrior, X::MatNum, F::MatNum; dims::Int = 1, strict::Bool = false,
               kwargs...)
    X, F = dims_oriented(dims, X, F)
    # `strict` reaches the wrapped prior: `pe.pe` admits `BlackLittermanPrior` and
    # `EntropyPoolingPrior`, both of which resolve view names against a universe and honour it.
    f_prior = prior(pe.pe, F; strict = strict)
    rr, posterior_X = factor_reconstruction(pe.re, X, F)
    (; mu, sigma, chol, esigma) = factor_lift(pe.mp, pe.ve, pe.rsd, rr, f_prior.mu,
                                              f_prior.sigma, X, posterior_X; kwargs...)
    # The lift already measured the residual variances, so the block carries them instead of
    # making every consumer recompute them from the reconstruction error. Under `rsd = false`
    # the lift added no residual block and `esigma` is `nothing`, which is what the field then
    # holds.
    rr = set_idiosyncratic_covariance(rr, esigma)
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
    return LowOrderPrior(; X = posterior_X, o_X = X, mu = mu, sigma = sigma, chol = chol,
                         w = f_prior.w, ens = f_prior.ens, kld = f_prior.kld,
                         ow = f_prior.ow, rr = rr, fpr = f_prior)
end

export FactorPrior
