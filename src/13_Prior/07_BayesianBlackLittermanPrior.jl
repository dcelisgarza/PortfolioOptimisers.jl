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
        mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
        views::Lc_BLV,
        sets::Option{<:AssetSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        rf::Number = 0.0,
        tau::Option{<:Number} = nothing
    ) -> BayesianBlackLittermanPrior

Keywords correspond to the struct's fields.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

# Examples

```jldoctest
julia> BayesianBlackLittermanPrior(;
                                   sets = AssetSets(; key = "nx",
                                                    dict = Dict("nx" => ["A", "B", "C"])),
                                   views = LinearConstraintEstimator(;
                                                                     val = ["A == 0.03",
                                                                            "B + C == 0.04"]))
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
             │       │           │      │   alg ┴ Full()
             │       │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
             │       │           │      │     pdm ┼ Posdef
             │       │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │       │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │       │           │      │      dn ┼ nothing
             │       │           │      │      dt ┼ nothing
             │       │           │      │     alg ┼ nothing
             │       │           │      │   order ┴ DenoiseDetoneAlg()
             │       │        me ┼ EquilibriumExpectedReturns
             │       │           │   ce ┼ PortfolioOptimisersCovariance
             │       │           │      │   ce ┼ Covariance
             │       │           │      │      │    me ┼ SimpleExpectedReturns
             │       │           │      │      │       │   w ┴ nothing
             │       │           │      │      │    ce ┼ GeneralCovariance
             │       │           │      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │       │           │      │      │       │    w ┴ nothing
             │       │           │      │      │   alg ┴ Full()
             │       │           │      │   mp ┼ DenoiseDetoneAlgMatrixProcessing
             │       │           │      │      │     pdm ┼ Posdef
             │       │           │      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │       │           │      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │       │           │      │      │      dn ┼ nothing
             │       │           │      │      │      dt ┼ nothing
             │       │           │      │      │     alg ┼ nothing
             │       │           │      │      │   order ┴ DenoiseDetoneAlg()
             │       │           │    w ┼ nothing
             │       │           │    l ┴ Int64: 1
             │       │   horizon ┴ nothing
             │    mp ┼ DenoiseDetoneAlgMatrixProcessing
             │       │     pdm ┼ Posdef
             │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │       │      dn ┼ nothing
             │       │      dt ┼ nothing
             │       │     alg ┼ nothing
             │       │   order ┴ DenoiseDetoneAlg()
             │    re ┼ StepwiseRegression
             │       │   crit ┼ PValue
             │       │        │   t ┴ Float64: 0.05
             │       │    alg ┼ Forward()
             │       │    tgt ┼ LinearModel
             │       │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │    ve ┼ SimpleVariance
             │       │          me ┼ SimpleExpectedReturns
             │       │             │   w ┴ nothing
             │       │           w ┼ nothing
             │       │   corrected ┴ Bool: true
             │   rsd ┴ Bool: true
          mp ┼ DenoiseDetoneAlgMatrixProcessing
             │     pdm ┼ Posdef
             │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      dn ┼ nothing
             │      dt ┼ nothing
             │     alg ┼ nothing
             │   order ┴ DenoiseDetoneAlg()
       views ┼ LinearConstraintEstimator
             │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
             │   key ┴ nothing
        sets ┼ AssetSets
             │    key ┼ String: "nx"
             │   ukey ┼ String: "ux"
             │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
  views_conf ┼ nothing
          rf ┼ Float64: 0.0
         tau ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`FactorPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`AssetSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
@concrete struct BayesianBlackLittermanPrior <: AbstractLowOrderPriorEstimator_F
    "$(field_dict[:pe])"
    pe
    "$(field_dict[:mp])"
    mp
    "$(field_dict[:views])"
    views
    "$(field_dict[:sets])"
    sets
    "$(field_dict[:views_conf])"
    views_conf
    "$(field_dict[:rf])"
    rf
    "$(field_dict[:tau])"
    tau
    function BayesianBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_F_AF,
                                         mp::AbstractMatrixProcessingEstimator,
                                         views::Lc_BLV, sets::Option{<:AssetSets},
                                         views_conf::Option{<:Num_VecNum}, rf::Number,
                                         tau::Option{<:Number})
        if isa(views, LinearConstraintEstimator)
            @argcheck(!isnothing(sets))
        end
        assert_bl_views_conf(views_conf, views)
        if !isnothing(tau)
            @argcheck(tau > zero(tau))
        end
        return new{typeof(pe), typeof(mp), typeof(views), typeof(sets), typeof(views_conf),
                   typeof(rf), typeof(tau)}(pe, mp, views, sets, views_conf, rf, tau)
    end
end
function BayesianBlackLittermanPrior(;
                                     pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(;
                                                                                           pe = EmpiricalPrior(;
                                                                                                               me = EquilibriumExpectedReturns())),
                                     mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
                                     views::Lc_BLV, sets::Option{<:AssetSets} = nothing,
                                     views_conf::Option{<:Num_VecNum} = nothing,
                                     rf::Number = 0.0,
                                     tau::Option{<:Number} = nothing)::BayesianBlackLittermanPrior
    return BayesianBlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`BayesianBlackLittermanPrior`](@ref) estimator with observation weights `w` applied to the underlying prior estimator.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`factory`](@ref)
"""
function factory(pe::BayesianBlackLittermanPrior,
                 w::ObsWeights)::BayesianBlackLittermanPrior
    return BayesianBlackLittermanPrior(; pe = factory(pe.pe, w), mp = pe.mp,
                                       views = pe.views, sets = pe.sets,
                                       views_conf = pe.views_conf, rf = pe.rf, tau = pe.tau)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`BayesianBlackLittermanPrior`](@ref) estimator restricted to the assets at index `i`.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
"""
function prior_view(pr::BayesianBlackLittermanPrior, i)::BayesianBlackLittermanPrior
    return BayesianBlackLittermanPrior(; pe = prior_view(pr.pe, i), mp = pr.mp,
                                       views = pr.views, sets = pr.sets,
                                       views_conf = pr.views_conf, rf = pr.rf, tau = pr.tau)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`BayesianBlackLittermanPrior`](@ref). Exposes `:me` and `:ce` from the embedded prior estimator `obj.pe` for transparent access.
"""
function Base.getproperty(obj::BayesianBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
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
  - `length(pe.sets.dict[pe.sets.key]) == size(F, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - The factor prior is computed using the embedded prior estimator `pe.pe`.
  - Views are extracted using [`black_litterman_views`](@ref), which returns the view matrix `P` and view returns vector `Q`.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of factor observations.
  - The view uncertainty matrix `f_omega` is computed using [`calc_omega`](@ref).
  - Bayesian posterior mean and covariance are computed via the model's update equations.
  - Matrix processing is applied to the posterior covariance and asset returns using the embedded matrix processing estimator `pe.mp`.
  - The result includes factor prior mean and covariance, and regression details.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
"""
function prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.sets.dict[pe.sets.key]) == size(F, 2))
    prior_result = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_sigma, f_mu, f_sigma, rr = prior_result.X, prior_result.sigma,
                                                  prior_result.f_mu, prior_result.f_sigma,
                                                  prior_result.rr
    (; P, Q, excl) = black_litterman_views(pe.views, pe.sets;
                                           datatype = eltype(posterior_X), strict = strict)
    tau = isnothing(pe.tau) ? inv(size(F, 1)) : pe.tau
    views_conf = remove_excl_views(pe.views_conf, excl)
    f_omega = tau * calc_omega(views_conf, P, f_sigma)
    (; b, M) = rr
    sigma_hat = f_sigma \ LinearAlgebra.I + transpose(P) * (f_omega \ P)
    mu_hat = sigma_hat \ (f_sigma \ f_mu + transpose(P) * (f_omega \ Q))
    v1 = prior_sigma \ M
    v2 = sigma_hat + transpose(M) * v1
    v3 = prior_sigma \ LinearAlgebra.I
    posterior_sigma = (v3 - v1 * (v2 \ transpose(M)) * v3) \ LinearAlgebra.I
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_mu = (posterior_sigma * v1 * (v2 \ sigma_hat) * mu_hat + b) .+ pe.rf
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         rr = rr, f_mu = f_mu, f_sigma = f_sigma)
end

export BayesianBlackLittermanPrior
