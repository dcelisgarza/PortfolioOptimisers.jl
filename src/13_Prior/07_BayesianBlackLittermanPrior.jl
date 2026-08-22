"""
$(DocStringExtensions.TYPEDEF)

Bayesian Black-Litterman prior estimator for asset returns.

`BayesianBlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using a Bayesian Black-Litterman model. It combines a factor prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and a blending parameter `tau`. This estimator supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates Bayesian updating for posterior inference.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BayesianBlackLittermanPrior(;
        pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(;
            pe = EmpiricalPrior(;
                me = EquilibriumExpectedReturns()
            )
        ),
        f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        rf::Number = 0.0,
        tau::Option{<:Number} = nothing
    ) -> BayesianBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

The views are applied to the **factors** and reach the assets through the regression loadings, so this estimator produces a posterior over both blocks. Under ADR 0046 it forwards the wrapped prior whole and spells out its deviations:

  - `mu` and `sigma` are the asset posterior; `chol` is **dropped**, because the posterior covariance supersedes the one it factorises.
  - The factor block `fpr` carries the **posterior** factor moments — `mu_hat` and the inverse of the posterior precision — processed by `f_mp`. Its `chol` is dropped for the same reason; its `w` and that weighting's diagnostics forward untouched, because the views do not touch the observation axis.
  - Everything else forwards: `X` is the wrapped prior's unchanged, so `w`, `ens`, `kld`, `ow` and `Z` all still describe the axis they were computed over, and `rr` is a regression over data the views do not modify.

Because both blocks are posterior, the returned carrier is **internally consistent**: `mu == rr.M * fpr.mu + rr.b` holds exactly. [`FactorBlackLittermanPrior`](@ref) satisfies it too, for the same reason. The other two members do not — see the warnings on [`BlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref).

!!! warning

    The returned `mu` and `sigma` are the Black-Litterman posterior, but `w` is the **wrapped prior's** observation weighting, forwarded unchanged. Black-Litterman produces no observation-level posterior, so there is no Black-Litterman-consistent alternative to forward — and dropping `w` would substitute the unweighted empirical distribution, which is further from the caller's intent than the weights they computed. A caller reading `pr.w`, `pr.ens`, `pr.kld` or `pr.ow` is therefore reading a property of the prior, not of the posterior.

## The views are written on the factor axis

`views` resolves against `sets.dict[sets.fkey]` — the axis [`UniverseSets`](@ref) declares for factors — because the Bayesian update lands on the factor distribution and reaches the assets through the loadings. The asset axis is still required (every `UniverseSets` carries one) and is what [`port_opt_view`](@ref) slices; the factor entries come back untouched, which is why this field is `@vprop` rather than exempted by hand.

`sets.dict[sets.fkey]` must name the columns of `F` **in order**; [`factor_universe`](@ref) checks it, and reports the factor axis rather than the asset one when it is missing or the wrong length.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> BayesianBlackLittermanPrior(;
                                   sets = UniverseSets(;
                                                       dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                                                   \"nf\" => [\"F1\", \"F2\"])),
                                   views = LinearConstraintEstimator(;
                                                                     val = [\"F1 == 0.03\",
                                                                            \"F2 == 0.04\"]))
BayesianBlackLittermanPrior
          pe ┼ FactorPrior
             │    pe ┼ EmpiricalPrior
             │       │        ce ┼ PortfolioOptimisersCovariance
             │       │           │   ce ┼ Covariance
             │       │           │      │    me ┼ SimpleExpectedReturns
             │       │           │      │       │   w ┴ nothing
             │       │           │      │    ce ┼ GeneralCovariance
             │       │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │       │           │      │       │    w ┴ nothing
             │       │           │      │   alg ┴ FullMoment()
             │       │           │   mp ┼ MatrixProcessing
             │       │           │      │     pdm ┼ Posdef
             │       │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │       │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │       │           │      │      dn ┼ nothing
             │       │           │      │      dt ┼ nothing
             │       │           │      │     alg ┼ nothing
             │       │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │       │        me ┼ EquilibriumExpectedReturns
             │       │           │   ce ┼ PortfolioOptimisersCovariance
             │       │           │      │   ce ┼ Covariance
             │       │           │      │      │    me ┼ SimpleExpectedReturns
             │       │           │      │      │       │   w ┴ nothing
             │       │           │      │      │    ce ┼ GeneralCovariance
             │       │           │      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │       │           │      │      │       │    w ┴ nothing
             │       │           │      │      │   alg ┴ FullMoment()
             │       │           │      │   mp ┼ MatrixProcessing
             │       │           │      │      │     pdm ┼ Posdef
             │       │           │      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │       │           │      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │       │           │      │      │      dn ┼ nothing
             │       │           │      │      │      dt ┼ nothing
             │       │           │      │      │     alg ┼ nothing
             │       │           │      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │       │           │    w ┼ nothing
             │       │           │    l ┴ Int64: 1
             │       │   horizon ┴ nothing
             │    mp ┼ MatrixProcessing
             │       │     pdm ┼ Posdef
             │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │       │      dn ┼ nothing
             │       │      dt ┼ nothing
             │       │     alg ┼ nothing
             │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │    re ┼ StepwiseRegression
             │       │   crit ┼ PValue
             │       │        │   t ┴ Float64: 0.05
             │       │    alg ┼ ForwardSelection()
             │       │    tgt ┼ LinearModel
             │       │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │    ve ┼ SimpleVariance
             │       │          me ┼ SimpleExpectedReturns
             │       │             │   w ┴ nothing
             │       │           w ┼ nothing
             │       │   corrected ┴ Bool: true
             │   rsd ┴ Bool: true
        f_mp ┼ MatrixProcessing
             │     pdm ┼ Posdef
             │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      dn ┼ nothing
             │      dt ┼ nothing
             │     alg ┼ nothing
             │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
          mp ┼ MatrixProcessing
             │     pdm ┼ Posdef
             │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      dn ┼ nothing
             │      dt ┼ nothing
             │     alg ┼ nothing
             │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
       views ┼ LinearConstraintEstimator
             │   val ┼ Vector{String}: ["F1 == 0.03", "F2 == 0.04"]
             │   key ┴ nothing
        sets ┼ UniverseSets
             │    xkey ┼ String: "nx"
             │   uxkey ┼ String: "ux"
             │    fkey ┼ String: "nf"
             │   ufkey ┼ String: "uf"
             │    zkey ┼ String: "nz"
             │    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"], "nf" => ["F1", "F2"])
  views_conf ┼ nothing
          rf ┼ Float64: 0.0
         tau ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`FactorPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`UniverseSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:kolmritter2016])
  - $(ref_dict[:cajas2025]) Section 5.3, Equations 5.23, 5.34 and 5.35.
"""
@propagatable @concrete struct BayesianBlackLittermanPrior <:
                               AbstractLowOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:f_mp])
    """
    f_mp
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:views])
    """
    views
    """
    $(field_dict[:sets_f])
    """
    @vprop sets
    """
    $(field_dict[:views_conf])
    """
    views_conf
    """
    $(field_dict[:bl_rf])
    """
    rf
    """
    $(field_dict[:tau])
    """
    tau
    function BayesianBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_F_AF,
                                         f_mp::AbstractMatrixProcessingEstimator,
                                         mp::AbstractMatrixProcessingEstimator,
                                         views::Lc_BLV, sets::Option{<:UniverseSets},
                                         views_conf::Option{<:Num_VecNum}, rf::Number,
                                         tau::Option{<:Number})
        assert_bl(views, sets, views_conf, tau)
        return new{typeof(pe), typeof(f_mp), typeof(mp), typeof(views), typeof(sets),
                   typeof(views_conf), typeof(rf), typeof(tau)}(pe, f_mp, mp, views, sets,
                                                                views_conf, rf, tau)
    end
end
function BayesianBlackLittermanPrior(;
                                     pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(;
                                                                                           pe = EmpiricalPrior(;
                                                                                                               me = EquilibriumExpectedReturns())),
                                     f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                     mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                     views::Lc_BLV, sets::Option{<:UniverseSets} = nothing,
                                     views_conf::Option{<:Num_VecNum} = nothing,
                                     rf::Number = 0.0,
                                     tau::Option{<:Number} = nothing)::BayesianBlackLittermanPrior
    return BayesianBlackLittermanPrior(pe, f_mp, mp, views, sets, views_conf, rf, tau)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties BayesianBlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
    prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute Bayesian Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the Bayesian Black-Litterman model, combining a factor prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and blending parameter `tau`. This method supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates Bayesian updating for posterior inference.

# Mathematical definition

This is **not** the classic Black-Litterman update run on the assets. The views land on the factor parameter ``\\boldsymbol{\\theta}``, and the assets are the posterior *predictive* distribution that the factor model implies. The model is:

```math
\\begin{align}
\\boldsymbol{r} &\\sim \\mathcal{N}(\\mathbf{M}\\boldsymbol{\\theta} + \\boldsymbol{b},\\ \\mathbf{\\Sigma})\\,, \\\\
\\boldsymbol{\\theta} &\\sim \\mathcal{N}(\\boldsymbol{\\Pi}_f,\\ \\mathbf{\\Sigma}_f)\\,, \\\\
\\mathbf{P}\\boldsymbol{\\theta} &\\sim \\mathcal{N}(\\boldsymbol{q},\\ \\mathbf{\\Omega})\\,.
\\end{align}
```

The conditional posterior of ``\\boldsymbol{\\theta}`` given the views is Gaussian, with precision ``\\mathbf{H}``:

```math
\\begin{align}
\\mathbf{H} &= \\mathbf{\\Sigma}_f^{-1} + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\mathbf{P}\\,, \\\\
\\bar{\\mathbf{\\Sigma}}_f &= \\mathbf{H}^{-1}\\,, \\\\
\\bar{\\boldsymbol{\\Pi}}_f &= \\mathbf{H}^{-1}\\left(\\mathbf{\\Sigma}_f^{-1}\\boldsymbol{\\Pi}_f + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\boldsymbol{q}\\right)\\,.
\\end{align}
```

Writing ``\\mathbf{V} = \\left(\\mathbf{H} + \\mathbf{M}^\\intercal \\mathbf{\\Sigma}^{-1} \\mathbf{M}\\right)^{-1}``, the posterior predictive asset moments are:

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{BBL} &= \\left(\\mathbf{\\Sigma}^{-1} - \\mathbf{\\Sigma}^{-1}\\mathbf{M}\\,\\mathbf{V}\\,\\mathbf{M}^\\intercal \\mathbf{\\Sigma}^{-1}\\right)^{-1}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_{BBL} &= \\hat{\\mathbf{\\Sigma}}_{BBL}\\,\\mathbf{\\Sigma}^{-1}\\mathbf{M}\\,\\mathbf{V}\\,\\mathbf{H}\\,\\bar{\\boldsymbol{\\Pi}}_f + \\boldsymbol{b} + r_{f}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{BBL}``: ``N \\times 1`` Bayesian Black-Litterman posterior asset mean.
  - ``\\hat{\\mathbf{\\Sigma}}_{BBL}``: ``N \\times N`` Bayesian Black-Litterman posterior asset covariance.
  - ``\\boldsymbol{\\Pi}_f``, ``\\mathbf{\\Sigma}_f``: ``K \\times 1`` and ``K \\times K`` prior factor moments, from `pe.pe`.
  - ``\\bar{\\boldsymbol{\\Pi}}_f``, ``\\bar{\\mathbf{\\Sigma}}_f``: Posterior factor moments, reported in `pr.fpr`.
  - ``\\mathbf{\\Sigma}``: ``N \\times N`` prior asset covariance matrix, from `pe.pe`.
  - ``\\mathbf{M}``: ``N \\times K`` factor loadings matrix, `pr.rr.M`.
  - ``\\boldsymbol{b}``: ``N \\times 1`` regression intercept vector, `pr.rr.b`.
  - ``\\mathbf{P}``: ``K_v \\times K`` views matrix, over the **factor** axis.
  - ``\\boldsymbol{q}``: ``K_v \\times 1`` views vector.
  - ``\\mathbf{\\Omega}``: ``K_v \\times K_v`` view uncertainty matrix, ``\\mathrm{Diag}(\\mathbf{P}(\\tau\\mathbf{\\Sigma}_f)\\mathbf{P}^\\intercal)`` from [`calc_omega`](@ref) and [`bl_preroll`](@ref).
  - ``\\tau``: Scaling parameter, `1/T` by default, where ``T`` is the number of factor observations.
  - ``r_{f}``: Risk-free rate, added once by [`apply_rf`](@ref).

Two consequences are caller-facing. ``\\mathbf{P}`` is over the factor axis, so it has ``K`` columns and not ``N`` — the classic asset-axis master equation cannot be evaluated with this estimator's own quantities at all. And ``\\hat{\\boldsymbol{\\mu}}_{BBL}`` is ``\\mathbf{M}\\bar{\\boldsymbol{\\Pi}}_f + \\boldsymbol{b}`` by construction, which is the identity the *Composition* section above states.

# Arguments

  - `pe`: Bayesian Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Validation

  - `dims in (1, 2)`.
  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.fkey)` and `length(pe.sets.dict[pe.sets.fkey]) == size(F, 2)`, both via [`factor_universe`](@ref).
  - The prior produced by `pe.pe` must carry a regression result, via [`assert_prior_regression`](@ref).

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, and factor prior details.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - The factor prior is computed using the embedded prior estimator `pe.pe`.
  - Views are extracted using [`black_litterman_views`](@ref) **at `pe.sets.fkey`**, which returns the view matrix `P` and view returns vector `Q`.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of factor observations.
  - The view uncertainty matrix `omega` is computed using [`calc_omega`](@ref) and scaled by `tau` in [`bl_preroll`](@ref).
  - `sigma_hat` is the posterior factor **precision** ``\\mathbf{H}``, not a covariance. `pr.fpr.sigma` is its inverse.
  - The posterior predictive asset moments are computed from ``\\mathbf{H}``, the loadings and the prior asset covariance. This estimator never calls [`vanilla_posteriors`](@ref); its siblings that take asset views do.
  - Matrix processing is applied to the asset posterior covariance using `pe.mp`, and to the factor posterior covariance using `pe.f_mp`.
  - `pe.rf` is added to the asset posterior mean once, by [`apply_rf`](@ref). The factor block never carries it.
  - The result's factor block holds the **posterior** factor moments, so `pr.mu == pr.rr.M * pr.fpr.mu + pr.rr.b + pe.rf` holds exactly. At the default `rf = 0.0` that is the plain identity.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
"""
function prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    # The views update the *factor* distribution — the assets are its projection through the
    # loadings — so they resolve against the declared factor axis, not against `xkey`. Only the
    # views that resolve *names* need a universe: a `BlackLittermanViews` result carries its own
    # `P`, so demanding one for it would reject the legitimate precomputed-views configuration,
    # which `assert_bl` deliberately permits to supply no `sets` at all.
    if isa(pe.views, LinearConstraintEstimator)
        factor_universe(pe.sets, size(F, 2),
                        "BayesianBlackLittermanPrior, whose views are written in factor names",
                        "F")
    end
    prior_result = prior(pe.pe, X, F; strict = strict, kwargs...)
    assert_prior_regression(prior_result, :pe)
    posterior_X, prior_sigma, fpr, rr = prior_result.X, prior_result.sigma,
                                        prior_result.fpr, prior_result.rr
    f_mu, f_sigma = fpr.mu, fpr.sigma
    (; P, Q, omega) = bl_preroll(pe.views, pe.sets, pe.views_conf, f_sigma, pe.tau,
                                 size(F, 1), eltype(posterior_X), strict, :fkey)
    (; b, M) = rr
    sigma_hat = f_sigma \ LinearAlgebra.I + transpose(P) * (omega \ P)
    mu_hat = sigma_hat \ (f_sigma \ f_mu + transpose(P) * (omega \ Q))
    v1 = prior_sigma \ M
    v2 = sigma_hat + transpose(M) * v1
    v3 = prior_sigma \ LinearAlgebra.I
    posterior_sigma = (v3 - v1 * (v2 \ transpose(M)) * v3) \ LinearAlgebra.I
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    # `pe.rf` is applied here and only here (see [`apply_rf`](@ref)): once, on the asset
    # expected returns this estimator returns. The wrapped prior's moments are used as they
    # stand, so a rate that prior applied internally is left alone.
    posterior_mu = apply_rf(pe.rf, posterior_sigma * v1 * (v2 \ sigma_hat) * mu_hat + b)
    # The views land on the *factors*, so `mu_hat` and `sigma_hat` are the posterior factor
    # moments — `sigma_hat` is a precision (`inv(f_sigma) + P'Ω⁻¹P`), so the covariance is its
    # inverse. Reporting them rather than the prior ones is what makes this carrier internally
    # consistent: `mu == rr.M * fpr.mu + rr.b` holds exactly afterwards, where forwarding the
    # prior block left the asset and factor halves describing different distributions.
    # `pe.f_mp` processes the factor block for the same reason `pe.mp` processes the asset one,
    # and is separate for the same reason `FactorBlackLittermanPrior` keeps the two apart.
    f_posterior_sigma = sigma_hat \ LinearAlgebra.I
    matrix_processing!(pe.f_mp, f_posterior_sigma, F; kwargs...)
    # `chol` is the factor block's only drop — `f_posterior_sigma` supersedes the covariance it
    # factorises. The views do not touch the observation axis, so the factor prior's `w` and
    # that weighting's diagnostics forward untouched (ADR 0046).
    posterior_fpr = forward_prior(fpr; mu = mu_hat, sigma = f_posterior_sigma,
                                  chol = nothing)
    # Everything else the wrapped prior carried is forwarded (see [`forward_prior`](@ref));
    # `chol` is the only drop, because `posterior_sigma` supersedes the covariance it
    # factorises. `posterior_X` is `prior_result.X` unchanged, so the wrapped `w` still
    # describes exactly the rows of the returned `X`, its `ens`/`kld`/`ow` still describe that
    # `w`, and the feature matrix is still over this asset axis. `rr` is unchanged — the
    # regression is over data the views do not modify — so the factor block it projects is now
    # the posterior one.
    return forward_prior(prior_result; mu = posterior_mu, sigma = posterior_sigma,
                         chol = nothing, fpr = posterior_fpr)
end

function factor_residual_config(pe::BayesianBlackLittermanPrior)
    return factor_residual_config(pe.pe)
end

export BayesianBlackLittermanPrior
