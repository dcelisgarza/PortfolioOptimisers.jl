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
        sets::Option{<:AssetSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        w::Option{<:ObsWeights} = nothing,
        rf::Number = 0.0,
        l::Option{<:Number} = nothing,
        tau::Option{<:Number} = nothing,
        rsd::Bool = true
    ) -> FactorBlackLittermanPrior

Keywords correspond to the struct's fields.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.
  - If `w` is not `nothing`, `length(w) == size(X, 2)`.

# Examples

```jldoctest
julia> FactorBlackLittermanPrior(;
                                 sets = AssetSets(; key = "nx",
                                                  dict = Dict("nx" => ["A", "B", "C"])),
                                 views = LinearConstraintEstimator(;
                                                                   val = ["A == 0.03",
                                                                          "B + C == 0.04"]))
FactorBlackLittermanPrior
          pe ┼ EmpiricalPrior
             │        ce ┼ PortfolioOptimisersCovariance
             │           │   ce ┼ Covariance
             │           │      │    me ┼ SimpleExpectedReturns
             │           │      │       │   w ┴ nothing
             │           │      │    ce ┼ GeneralCovariance
             │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │           │      │       │    w ┴ nothing
             │           │      │   alg ┴ Full()
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
             │    alg ┼ Forward()
             │    tgt ┼ LinearModel
             │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
          ve ┼ SimpleVariance
             │          me ┼ SimpleExpectedReturns
             │             │   w ┴ nothing
             │           w ┼ nothing
             │   corrected ┴ Bool: true
       views ┼ LinearConstraintEstimator
             │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
             │   key ┴ nothing
        sets ┼ AssetSets
             │    key ┼ String: "nx"
             │   ukey ┼ String: "ux"
             │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
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
  - [`AssetSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
@concrete struct FactorBlackLittermanPrior <: AbstractLowOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    pe
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
    re
    """
    $(field_dict[:ve])
    """
    ve
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
    $(field_dict[:w_rm])
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
                                       sets::Option{<:AssetSets},
                                       views_conf::Option{<:Num_VecNum},
                                       w::Option{<:VecNum}, rf::Number, l::Option{<:Number},
                                       tau::Option{<:Number}, rsd::Bool)
        if isa(views, LinearConstraintEstimator)
            @argcheck(!isnothing(sets))
        end
        assert_bl_views_conf(views_conf, views)
        if !isnothing(tau)
            @argcheck(tau > zero(tau))
        end
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
                                   views::Lc_BLV, sets::Option{<:AssetSets} = nothing,
                                   views_conf::Option{<:Num_VecNum} = nothing,
                                   w::Option{<:VecNum} = nothing, rf::Number = 0.0,
                                   l::Option{<:Number} = nothing,
                                   tau::Option{<:Number} = nothing,
                                   rsd::Bool = true)::FactorBlackLittermanPrior
    return FactorBlackLittermanPrior(pe, f_mp, mp, re, ve, views, sets, views_conf, w, rf,
                                     l, tau, rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`FactorBlackLittermanPrior`](@ref) estimator with observation weights `w` applied to the underlying prior, regression, and variance estimators.

# Related

  - [`FactorBlackLittermanPrior`](@ref)
  - [`factory`](@ref)
"""
function factory(pe::FactorBlackLittermanPrior, w::ObsWeights)::FactorBlackLittermanPrior
    return FactorBlackLittermanPrior(; pe = factory(pe.pe, w), f_mp = pe.f_mp, mp = pe.mp,
                                     re = factory(pe.re, w), ve = factory(pe.ve, w),
                                     views = pe.views, sets = pe.sets,
                                     views_conf = pe.views_conf, w = pe.w, rf = pe.rf,
                                     l = pe.l, tau = pe.tau, rsd = pe.rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`FactorBlackLittermanPrior`](@ref) estimator restricted to the assets at index `i`.

# Related

  - [`FactorBlackLittermanPrior`](@ref)
  - [`prior_view`](@ref)
"""
function prior_view(pe::FactorBlackLittermanPrior, i)
    return FactorPrior(; pe = pe.pe, f_mp = pe.f_mp, mp = pe.mp,
                       re = regression_view(pe.re, i), ve = moment_view(pe.ve, i),
                       views = pe.views, sets = pe.sets, views_conf = pe.views_conf,
                       w = pe.w, rf = pe.rf, l = pe.l, tau = pe.tau, rsd = pe.rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`FactorBlackLittermanPrior`](@ref). Exposes `:me` and `:ce` from the embedded prior estimator `obj.pe` for transparent access.
"""
function Base.getproperty(obj::FactorBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
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
  - `length(pe.sets.dict[pe.sets.key]) == size(F, 2)`.
  - If `pe.w` is not `nothing`, `length(pe.w) == size(X, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - The factor prior is computed using the embedded asset prior estimator `pe.pe`.
  - Factor regression is performed using the regression estimator `pe.re`.
  - Views are extracted using [`black_litterman_views`](@ref), which returns the view matrix `P` and view returns vector `Q`.
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
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.sets.dict[pe.sets.key]) == size(F, 2))
    # Factor prior.
    f_prior = prior(pe.pe, F; strict = strict)
    prior_mu, prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    (; P, Q, excl) = black_litterman_views(pe.views, pe.sets;
                                           datatype = eltype(posterior_X), strict = strict)
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    views_conf = remove_excl_views(pe.views_conf, excl)
    omega = tau * calc_omega(views_conf, P, prior_sigma)
    prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @argcheck(length(pe.w) == size(X, 2))
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
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                  :)), w = f_prior.w, rr = rr,
                         f_mu = f_posterior_mu, f_sigma = f_posterior_sigma,
                         f_w = f_prior.w)
end

export FactorBlackLittermanPrior
