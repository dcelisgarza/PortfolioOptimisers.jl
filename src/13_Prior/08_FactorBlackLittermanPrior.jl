"""
$(DocStringExtensions.TYPEDEF)

Factor Black-Litterman prior estimator for asset returns.

`FactorBlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using a factor-based Black-Litterman model. It combines an asset prior estimator, matrix post-processing for factors and assets, regression and variance estimators, user or algorithmic views, asset sets, view confidences, weights, risk-free rate, leverage, blending parameter `tau`, and a residual variance flag. This estimator supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates factor regression and residual adjustment for posterior inference.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorBlackLittermanPrior(;
        pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        re::AbstractRegressionEstimator = StepwiseRegression(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        w::Option{<:ObsWeights} = nothing,
        rf::Number = 0.0,
        l::Option{<:Number} = nothing,
        tau::Option{<:Number} = nothing,
        rsd::Bool = true
    ) -> FactorBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **lifts** a factor-axis prior onto the asset axis, reconstructing `X` as `F * transpose(M) .+ transpose(b)`, so it builds its carrier directly rather than forwarding one along its own axis; the rule of ADR 0046 still governs each field. It is the member of the Black-Litterman family whose factor block is *modified* rather than passed through — the views land on the factor distribution, and the assets are its projection.

  - The factor block `fpr` is the **posterior** factor distribution, processed by `f_mp`, with `chol` dropped because the posterior covariance supersedes the one it factorises. Its `w` and that weighting's diagnostics forward untouched.
  - `mu` and `sigma` are that block projected through the loadings, so the returned carrier is **internally consistent**: `mu == rr.M * fpr.mu + rr.b` holds by construction. `sigma` optionally gains a residual correction when `rsd` is `true`.
  - `w` is the factor prior's, and is over the right axis: this estimator wraps only a factor prior, and `posterior_X` has exactly `F`'s rows, so it is the only weighting in existence.
  - No `Z` is carried: the only wrapped prior is fit on factors, so its feature matrix would be factors × features and would not describe the asset axis. The drop is a *relocation* rather than a destruction — the factor block is forwarded, so a feature matrix the factor prior carried is still reachable at `pr.fpr.Z`. Wrap this estimator from the *outside* with [`FeaturePrior`](@ref) if an asset-axis feature matrix is wanted.

Its siblings differ: [`BayesianBlackLittermanPrior`](@ref) also satisfies the identity exactly, while [`BlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) do not — see their warnings.

## The views are written on the factor axis

`views` resolves against `sets.dict[sets.fkey]` — the axis [`UniverseSets`](@ref) declares for factors — because that is the distribution they update. The asset axis is still required (every `UniverseSets` carries one) and is what [`port_opt_view`](@ref) slices; the factor entries come back untouched, which is why this field is `@vprop` rather than exempted by hand.

`sets.dict[sets.fkey]` must name the columns of `F` **in order**; [`factor_universe`](@ref) checks it, and reports the factor axis rather than the asset one when it is missing or the wrong length.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.
  - If `w` is not `nothing`, `length(w) == size(X, 2)`.

# Examples

```jldoctest
julia> FactorBlackLittermanPrior(;
                                 sets = UniverseSets(;
                                                     dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                                                 \"nf\" => [\"F1\", \"F2\"])),
                                 views = LinearConstraintEstimator(;
                                                                   val = [\"F1 == 0.03\",
                                                                          \"F2 == 0.04\"]))
FactorBlackLittermanPrior
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
           w ┼ nothing
          rf ┼ Float64: 0.0
           l ┼ nothing
         tau ┼ nothing
         rsd ┴ Bool: true
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`UniverseSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
@propagatable @concrete struct FactorBlackLittermanPrior <: AbstractLowOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    @fprop pe
    """
    $(field_dict[:f_mp])
    """
    f_mp
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
    $(field_dict[:eqw])
    """
    w
    """
    $(field_dict[:rf])
    """
    rf
    """
    $(field_dict[:l])
    """
    l
    """
    $(field_dict[:tau])
    """
    tau
    """
    $(field_dict[:rsd])
    """
    rsd
    function FactorBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                                       f_mp::AbstractMatrixProcessingEstimator,
                                       mp::AbstractMatrixProcessingEstimator,
                                       re::AbstractRegressionEstimator,
                                       ve::AbstractVarianceEstimator, views::Lc_BLV,
                                       sets::Option{<:UniverseSets},
                                       views_conf::Option{<:Num_VecNum},
                                       w::Option{<:VecNum}, rf::Number, l::Option{<:Number},
                                       tau::Option{<:Number}, rsd::Bool)
        assert_bl(views, sets, views_conf, tau)
        return new{typeof(pe), typeof(f_mp), typeof(mp), typeof(re), typeof(ve),
                   typeof(views), typeof(sets), typeof(views_conf), typeof(w), typeof(rf),
                   typeof(l), typeof(tau), typeof(rsd)}(pe, f_mp, mp, re, ve, views, sets,
                                                        views_conf, w, rf, l, tau, rsd)
    end
end
function FactorBlackLittermanPrior(;
                                   pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                   f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   re::AbstractRegressionEstimator = StepwiseRegression(),
                                   ve::AbstractVarianceEstimator = SimpleVariance(),
                                   views::Lc_BLV, sets::Option{<:UniverseSets} = nothing,
                                   views_conf::Option{<:Num_VecNum} = nothing,
                                   w::Option{<:VecNum} = nothing, rf::Number = 0.0,
                                   l::Option{<:Number} = nothing,
                                   tau::Option{<:Number} = nothing,
                                   rsd::Bool = true)::FactorBlackLittermanPrior
    return FactorBlackLittermanPrior(pe, f_mp, mp, re, ve, views, sets, views_conf, w, rf,
                                     l, tau, rsd)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties FactorBlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
    prior(pe::FactorBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute factor Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the factor-based Black-Litterman model, combining an asset prior estimator, matrix post-processing for factors and assets, regression and variance estimators, user or algorithmic views, asset sets, view confidences, weights, risk-free rate, leverage, blending parameter `tau`, and a residual variance flag. This method supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates factor regression and residual adjustment for posterior inference.

# Mathematical definition

Black-Litterman views are applied directly to the factor space, updating factor moments ``(\\boldsymbol{\\Pi}_f, \\mathbf{\\Sigma}_f)`` via the standard BL equations, then asset posteriors are reconstructed through the loadings matrix:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{\\mu}}_{f,BL} + \\boldsymbol{\\alpha}\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\hat{\\mathbf{\\Sigma}}_{f,BL} \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` posterior asset mean vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` posterior asset covariance matrix.
  - ``\\hat{\\boldsymbol{\\mu}}_{f,BL}``: ``K \\times 1`` Black-Litterman posterior factor mean.
  - ``\\hat{\\mathbf{\\Sigma}}_{f,BL}``: ``K \\times K`` Black-Litterman posterior factor covariance.
  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix.
  - ``\\boldsymbol{\\alpha}``: ``N \\times 1`` regression intercept vector.
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal residual variance matrix (when `rsd = true`).

# Arguments

  - `pe`: Factor Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, Cholesky factor, weights, regression result, and factor prior details.

# Validation

  - `dims in (1, 2)`.
  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.fkey)` and `length(pe.sets.dict[pe.sets.fkey]) == size(F, 2)`, both via [`factor_universe`](@ref).
  - If `pe.w` is not `nothing`, `length(pe.w) == size(X, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - The factor prior is computed using the embedded asset prior estimator `pe.pe`.
  - Factor regression is performed using the regression estimator `pe.re`.
  - Views are extracted using [`black_litterman_views`](@ref) **at `pe.sets.fkey`**, which returns the view matrix `P` and view returns vector `Q`.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of observations.
  - The view uncertainty matrix `omega` is computed using [`calc_omega`](@ref).
  - If leverage is specified, the prior mean is adjusted accordingly.
  - The Black-Litterman posterior mean and covariance for factors are computed using [`vanilla_posteriors`](@ref).
  - Matrix processing is applied to the factor and asset posterior covariance matrices.
  - The asset posterior mean and covariance are reconstructed using the factor loadings.
  - If `rsd` is `true`, residual variance is added to the posterior covariance and Cholesky factor.
  - The result includes asset returns, posterior mean, posterior covariance, Cholesky factor, weights, regression result, and factor prior details.

# Related

  - [`FactorBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function prior(pe::FactorBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    # The views land on the *factor* distribution, so they resolve against the declared factor
    # axis — not against `xkey`, which names the assets this estimator projects onto. Only the
    # views that resolve *names* need a universe: a `BlackLittermanViews` result carries its own
    # `P`, so demanding one for it would reject the legitimate precomputed-views configuration,
    # which `assert_bl` deliberately permits to supply no `sets` at all.
    if isa(pe.views, LinearConstraintEstimator)
        factor_universe(pe.sets, size(F, 2),
                        "FactorBlackLittermanPrior, whose views are written in factor names",
                        "F")
    end
    # Factor prior.
    f_prior = prior(pe.pe, F; strict = strict)
    prior_mu, prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    # Precomputed views ignore both the sets and the key, and are the one shape that reaches here
    # with `pe.sets === nothing`, so the key is read only when there is a sets to read it from.
    f_key = isnothing(pe.sets) ? nothing : pe.sets.fkey
    (; P, Q, tau, omega) = bl_preroll(pe.views, pe.sets, pe.views_conf, prior_sigma, pe.tau,
                                      size(X, 1), eltype(posterior_X), strict, f_key)
    prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @argcheck(length(pe.w) == size(X, 2),
                      DimensionMismatch("length(pe.w) ($(length(pe.w))) must match size(X, 2) ($(size(X, 2)))"))
            pe.w
        else
            iN = inv(size(X, 2))
            range(iN, iN; length = size(X, 2))
        end
        pe.l * (prior_sigma * transpose(M)) * w
    else
        prior_mu .- pe.rf
    end
    f_posterior_mu, f_posterior_sigma = vanilla_posteriors(tau, pe.rf, prior_mu,
                                                           prior_sigma, omega, P, Q)
    matrix_processing!(pe.f_mp, f_posterior_sigma, F)
    # Reconstruct the posteriors using the black litterman adjusted factor statistics.
    posterior_mu = M * f_posterior_mu + b
    posterior_sigma = M * f_posterior_sigma * transpose(M)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_csigma = M * LinearAlgebra.cholesky(f_posterior_sigma).L
    if pe.rsd
        err = X - posterior_X
        err_sigma = LinearAlgebra.diagm(vec(Statistics.var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posdef!(pe.mp.pdm, posterior_sigma)
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    # No `Z` is forwarded: the only wrapped prior here is `f_prior`, fit on the factors, so
    # its feature matrix would be factors × features and would not describe the asset axis.
    #
    # This is the one site that *modifies* the factor block rather than passing it through:
    # the views land on the factor distribution. `chol` is dropped because `f_posterior_sigma`
    # supersedes the factor prior's covariance (see [`forward_prior`](@ref)); everything else
    # the factor prior carried — its `w` and that weighting's diagnostics — is forwarded.
    fpr = forward_prior(f_prior; mu = f_posterior_mu, sigma = f_posterior_sigma,
                        chol = nothing)
    #
    # The asset-side `w` is the factor prior's: this estimator wraps only a factor prior, and
    # `posterior_X = F*M' + b'` has exactly `F`'s rows, so it is the only weighting in
    # existence and it is over the right observation axis. Its `ens`/`kld`/`ow` travel with it
    # — a weighting with no provenance cannot be interrogated (ADR 0046).
    return LowOrderPrior(; X = posterior_X, o_X = X, mu = posterior_mu,
                         sigma = posterior_sigma,
                         chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                  :)), w = f_prior.w, ens = f_prior.ens,
                         kld = f_prior.kld, ow = f_prior.ow, rr = rr, fpr = fpr)
end

export FactorBlackLittermanPrior
