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

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

# Examples

```jldoctest
julia> BayesianBlackLittermanPrior(;
                                   sets = UniverseSets(; xkey = \"nx\",
                                                       dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"])),
                                   views = LinearConstraintEstimator(;
                                                                     val = [\"A == 0.03\",
                                                                            \"B + C == 0.04\"]))
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
             │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
             │   key ┴ nothing
        sets ┼ UniverseSets
             │    xkey ┼ String: "nx"
             │   uxkey ┼ String: "ux"
             │    fkey ┼ String: "nf"
             │   ufkey ┼ String: "uf"
             │    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
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
    $(field_dict[:sets])
    """
    sets
    """
    $(field_dict[:views_conf])
    """
    views_conf
    """
    $(field_dict[:rf])
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

The Bayesian Black-Litterman model updates the prior ``(\\boldsymbol{\\Pi}, \\mathbf{\\Sigma}/T)`` with views:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{BBL} &= \\boldsymbol{\\Pi} + \\frac{\\mathbf{\\Sigma}}{T} \\mathbf{P}^\\intercal \\left(\\mathbf{P}\\frac{\\mathbf{\\Sigma}}{T}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}\\right)^{-1} (\\boldsymbol{q} - \\mathbf{P}\\boldsymbol{\\Pi})\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{BBL} &= \\mathbf{\\Sigma} + \\frac{\\mathbf{\\Sigma}}{T} - \\frac{\\mathbf{\\Sigma}}{T} \\mathbf{P}^\\intercal \\left(\\mathbf{P}\\frac{\\mathbf{\\Sigma}}{T}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}\\right)^{-1} \\mathbf{P} \\frac{\\mathbf{\\Sigma}}{T}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{BBL}``: Bayesian Black-Litterman posterior mean.
  - ``\\hat{\\mathbf{\\Sigma}}_{BBL}``: Bayesian Black-Litterman posterior covariance.
  - ``\\boldsymbol{\\Pi}``: ``N \\times 1`` prior expected returns.
  - ``\\mathbf{\\Sigma}``: ``N \\times N`` prior covariance matrix.
  - $(math_dict[:T])
  - ``\\mathbf{P}``: ``K \\times N`` views matrix.
  - ``\\boldsymbol{q}``: ``K \\times 1`` views vector.
  - ``\\mathbf{\\Omega}``: ``K \\times K`` view uncertainty matrix.

# Arguments

  - `pe`: Bayesian Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, and factor prior details.

# Validation

  - `dims in (1, 2)`.
  - `length(pe.sets.dict[pe.sets.xkey]) == size(F, 2)`.
  - The prior produced by `pe.pe` must carry a regression result, via [`assert_prior_regression`](@ref).

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - The factor prior is computed using the embedded prior estimator `pe.pe`.
  - Views are extracted using [`black_litterman_views`](@ref), which returns the view matrix `P` and view returns vector `Q`.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of factor observations.
  - The view uncertainty matrix `f_omega` is computed using [`calc_omega`](@ref).
  - Bayesian posterior mean and covariance are computed via the model's update equations.
  - Matrix processing is applied to the asset posterior covariance using `pe.mp`, and to the factor posterior covariance using `pe.f_mp`.
  - The result's factor block holds the **posterior** factor moments, so `pr.mu == pr.rr.M * pr.fpr.mu + pr.rr.b` holds exactly.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
"""
function prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.sets.dict[pe.sets.xkey]) == size(F, 2),
              DimensionMismatch("length(pe.sets.dict[pe.sets.xkey]) ($(length(pe.sets.dict[pe.sets.xkey]))) must match size(F, 2) ($(size(F, 2)))"))
    prior_result = prior(pe.pe, X, F; strict = strict, kwargs...)
    assert_prior_regression(prior_result, :pe)
    posterior_X, prior_sigma, fpr, rr = prior_result.X, prior_result.sigma,
                                        prior_result.fpr, prior_result.rr
    f_mu, f_sigma = fpr.mu, fpr.sigma
    (; P, Q, omega) = bl_preroll(pe.views, pe.sets, pe.views_conf, f_sigma, pe.tau,
                                 size(F, 1), eltype(posterior_X), strict)
    (; b, M) = rr
    sigma_hat = f_sigma \ LinearAlgebra.I + transpose(P) * (omega \ P)
    mu_hat = sigma_hat \ (f_sigma \ f_mu + transpose(P) * (omega \ Q))
    v1 = prior_sigma \ M
    v2 = sigma_hat + transpose(M) * v1
    v3 = prior_sigma \ LinearAlgebra.I
    posterior_sigma = (v3 - v1 * (v2 \ transpose(M)) * v3) \ LinearAlgebra.I
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_mu = (posterior_sigma * v1 * (v2 \ sigma_hat) * mu_hat + b) .+ pe.rf
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

export BayesianBlackLittermanPrior
