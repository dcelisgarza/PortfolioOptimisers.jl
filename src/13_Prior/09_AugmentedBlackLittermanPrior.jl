"""
$(DocStringExtensions.TYPEDEF)

Augmented Black-Litterman prior estimator for asset returns.

`AugmentedBlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using an augmented Black-Litterman model. It combines asset and factor prior estimators, matrix post-processing, regression and variance estimators, asset and factor views, asset and factor sets, view confidences, weights, risk-free rate, leverage, and a blending parameter `tau`. This estimator supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates joint asset-factor Bayesian updating for posterior inference.

# Mathematical definition

Factor model linking assets and factors via regression:

```math
\\mathbf{X} \\approx \\mathbf{F}\\mathbf{M}^{\\intercal} + \\mathbf{1}\\boldsymbol{b}^{\\intercal}
```

Augmented prior moments (stacking asset and factor priors):

```math
\\boldsymbol{\\mu}_{aug} = \\begin{pmatrix}\\boldsymbol{\\mu}_a \\\\ \\boldsymbol{\\mu}_f\\end{pmatrix}, \\qquad \\boldsymbol{\\Sigma}_{aug} = \\begin{pmatrix}\\boldsymbol{\\Sigma}_a & \\boldsymbol{\\Sigma}_a\\mathbf{M}^{\\intercal} \\\\ \\mathbf{M}\\boldsymbol{\\Sigma}_a & \\boldsymbol{\\Sigma}_f\\end{pmatrix}
```

Black-Litterman posterior on the augmented space with combined views ``(\\mathbf{P}_{aug}, \\boldsymbol{q}_{aug})``:

```math
\\boldsymbol{\\mu}_{post} = \\boldsymbol{\\mu}_{aug} + \\tau\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal}\\left(\\tau\\mathbf{P}_{aug}\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal} + \\boldsymbol{\\Omega}_{aug}\\right)^{-1}\\!\\left(\\boldsymbol{q}_{aug} - \\mathbf{P}_{aug}\\boldsymbol{\\mu}_{aug}\\right)
```

Asset posterior extracted from the augmented result and adjusted for intercept and risk-free rate.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AugmentedBlackLittermanPrior(;
        a_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        f_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
        re::AbstractRegressionEstimator = StepwiseRegression(),
        a_views::Lc_BLV,
        f_views::Lc_BLV,
        a_sets::Option{<:AssetSets} = nothing,
        f_sets::Option{<:AssetSets} = nothing,
        a_views_conf::Option{<:Num_VecNum} = nothing,
        f_views_conf::Option{<:Num_VecNum} = nothing,
        w::Option{<:VecNum} = nothing,
        rf::Number = 0.0,
        l::Option{<:Number} = nothing,
        tau::Option{<:Number} = nothing
    ) -> AugmentedBlackLittermanPrior

Keywords correspond to the struct's fields.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.
  - If `a_views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(a_sets)`.
  - If `f_views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(f_sets)`.
  - If `a_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `f_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

# Examples

```jldoctest
julia> AugmentedBlackLittermanPrior(;
                                    a_sets = AssetSets(; key = "nx",
                                                       dict = Dict("nx" => ["A", "B", "C"])),
                                    f_sets = AssetSets(; key = "nx",
                                                       dict = Dict("nx" => ["F1", "F2"])),
                                    a_views = LinearConstraintEstimator(;
                                                                        val = ["A == 0.03",
                                                                               "B + C == 0.04"]),
                                    f_views = LinearConstraintEstimator(;
                                                                        val = ["F1 == 0.01",
                                                                               "F2 == 0.02"]))
AugmentedBlackLittermanPrior
          a_pe ┼ EmpiricalPrior
               │        ce ┼ PortfolioOptimisersCovariance
               │           │   ce ┼ Covariance
               │           │      │    me ┼ SimpleExpectedReturns
               │           │      │       │   w ┴ nothing
               │           │      │    ce ┼ GeneralCovariance
               │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
               │           │      │       │    w ┴ nothing
               │           │      │   alg ┴ Full()
               │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
               │           │      │     pdm ┼ Posdef
               │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
               │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
               │           │      │      dn ┼ nothing
               │           │      │      dt ┼ nothing
               │           │      │     alg ┼ nothing
               │           │      │   order ┴ DenoiseDetoneAlg()
               │        me ┼ SimpleExpectedReturns
               │           │   w ┴ nothing
               │   horizon ┴ nothing
          f_pe ┼ EmpiricalPrior
               │        ce ┼ PortfolioOptimisersCovariance
               │           │   ce ┼ Covariance
               │           │      │    me ┼ SimpleExpectedReturns
               │           │      │       │   w ┴ nothing
               │           │      │    ce ┼ GeneralCovariance
               │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
               │           │      │       │    w ┴ nothing
               │           │      │   alg ┴ Full()
               │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
               │           │      │     pdm ┼ Posdef
               │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
               │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
               │           │      │      dn ┼ nothing
               │           │      │      dt ┼ nothing
               │           │      │     alg ┼ nothing
               │           │      │   order ┴ DenoiseDetoneAlg()
               │        me ┼ SimpleExpectedReturns
               │           │   w ┴ nothing
               │   horizon ┴ nothing
            mp ┼ DenoiseDetoneAlgMatrixProcessing
               │     pdm ┼ Posdef
               │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
               │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
               │      dn ┼ nothing
               │      dt ┼ nothing
               │     alg ┼ nothing
               │   order ┴ DenoiseDetoneAlg()
            re ┼ StepwiseRegression
               │   crit ┼ PValue
               │        │   t ┴ Float64: 0.05
               │    alg ┼ Forward()
               │    tgt ┼ LinearModel
               │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
       a_views ┼ LinearConstraintEstimator
               │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
               │   key ┴ nothing
       f_views ┼ LinearConstraintEstimator
               │   val ┼ Vector{String}: ["F1 == 0.01", "F2 == 0.02"]
               │   key ┴ nothing
        a_sets ┼ AssetSets
               │    key ┼ String: "nx"
               │   ukey ┼ String: "ux"
               │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
        f_sets ┼ AssetSets
               │    key ┼ String: "nx"
               │   ukey ┼ String: "ux"
               │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["F1", "F2"])
  a_views_conf ┼ nothing
  f_views_conf ┼ nothing
             w ┼ nothing
            rf ┼ Float64: 0.0
             l ┼ nothing
           tau ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`AssetSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
@concrete struct AugmentedBlackLittermanPrior <: AbstractLowOrderPriorEstimator_F
    "$(field_dict[:a_pe])"
    a_pe
    "$(field_dict[:f_pe])"
    f_pe
    "$(field_dict[:mp])"
    mp
    "$(field_dict[:re])"
    re
    "$(field_dict[:a_views])"
    a_views
    "$(field_dict[:f_views])"
    f_views
    "$(field_dict[:a_sets])"
    a_sets
    "$(field_dict[:f_sets])"
    f_sets
    "$(field_dict[:a_views_conf])"
    a_views_conf
    "$(field_dict[:f_views_conf])"
    f_views_conf
    "$(field_dict[:eqw])"
    w
    "$(field_dict[:rf])"
    rf
    "$(field_dict[:l])"
    l
    "$(field_dict[:tau])"
    tau
    function AugmentedBlackLittermanPrior(a_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          f_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          mp::AbstractMatrixProcessingEstimator,
                                          re::AbstractRegressionEstimator, a_views::Lc_BLV,
                                          f_views::Lc_BLV, a_sets::Option{<:AssetSets},
                                          f_sets::Option{<:AssetSets},
                                          a_views_conf::Option{<:Num_VecNum},
                                          f_views_conf::Option{<:Num_VecNum},
                                          w::Option{<:VecNum}, rf::Number,
                                          l::Option{<:Number}, tau::Option{<:Number})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        if isa(a_views, LinearConstraintEstimator)
            @argcheck(!isnothing(a_sets))
        end
        if isa(f_views, LinearConstraintEstimator)
            @argcheck(!isnothing(f_sets))
        end
        assert_bl_views_conf(a_views_conf, a_views)
        assert_bl_views_conf(f_views_conf, f_views)
        if !isnothing(tau)
            @argcheck(tau > zero(tau))
        end
        return new{typeof(a_pe), typeof(f_pe), typeof(mp), typeof(re), typeof(a_views),
                   typeof(f_views), typeof(a_sets), typeof(f_sets), typeof(a_views_conf),
                   typeof(f_views_conf), typeof(w), typeof(rf), typeof(l), typeof(tau)}(a_pe,
                                                                                        f_pe,
                                                                                        mp,
                                                                                        re,
                                                                                        a_views,
                                                                                        f_views,
                                                                                        a_sets,
                                                                                        f_sets,
                                                                                        a_views_conf,
                                                                                        f_views_conf,
                                                                                        w,
                                                                                        rf,
                                                                                        l,
                                                                                        tau)
    end
end
function AugmentedBlackLittermanPrior(;
                                      a_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                      f_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                      mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
                                      re::AbstractRegressionEstimator = StepwiseRegression(),
                                      a_views::Lc_BLV, f_views::Lc_BLV,
                                      a_sets::Option{<:AssetSets} = nothing,
                                      f_sets::Option{<:AssetSets} = nothing,
                                      a_views_conf::Option{<:Num_VecNum} = nothing,
                                      f_views_conf::Option{<:Num_VecNum} = nothing,
                                      w::Option{<:VecNum} = nothing, rf::Number = 0.0,
                                      l::Option{<:Number} = nothing,
                                      tau::Option{<:Number} = nothing)::AugmentedBlackLittermanPrior
    return AugmentedBlackLittermanPrior(a_pe, f_pe, mp, re, a_views, f_views, a_sets,
                                        f_sets, a_views_conf, f_views_conf, w, rf, l, tau)
end
function factory(pe::AugmentedBlackLittermanPrior,
                 w::ObsWeights)::AugmentedBlackLittermanPrior
    return AugmentedBlackLittermanPrior(; a_pe = factory(pe.a_pe, w),
                                        f_pe = factory(pe.f_pe, w), mp = pe.mp,
                                        re = factory(pe.re, w), a_views = pe.a_views,
                                        f_views = pe.f_views, a_sets = pe.a_sets,
                                        f_sets = pe.f_sets, a_views_conf = pe.a_views_conf,
                                        f_views_conf = pe.f_views_conf, w = pe.w,
                                        rf = pe.rf, l = pe.l, tau = pe.tau)
end
function prior_view(pe::AugmentedBlackLittermanPrior, i)::AugmentedBlackLittermanPrior
    return AugmentedBlackLittermanPrior(; a_pe = prior_view(pe.a_pe, i), f_pe = pe.f_pe,
                                        mp = pe.mp, re = regression_view(pe.re, i),
                                        a_views = pe.a_views, f_views = pe.f_views,
                                        a_sets = asset_sets_view(pe.a_sets, i),
                                        f_sets = pe.f_sets, a_views_conf = pe.a_views_conf,
                                        f_views_conf = pe.f_views_conf,
                                        w = nothing_scalar_array_view(pe.w, i), rf = pe.rf,
                                        l = pe.l, tau = pe.tau)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`AugmentedBlackLittermanPrior`](@ref). Exposes `:me`, `:ce` from the asset prior `obj.a_pe` and `:f_me`, `:f_ce` from the factor prior `obj.f_pe` for transparent access.
"""
function Base.getproperty(obj::AugmentedBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.a_pe.me
    elseif sym == :ce
        obj.a_pe.ce
    elseif sym == :f_me
        obj.f_pe.me
    elseif sym == :f_ce
        obj.f_pe.ce
    else
        getfield(obj, sym)
    end
end
"""
    prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute augmented Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the augmented Black-Litterman model, combining asset and factor prior estimators, matrix post-processing, regression and variance estimators, asset and factor views, asset and factor sets, view confidences, weights, risk-free rate, leverage, and a blending parameter `tau`. This method supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates joint asset-factor Bayesian updating for posterior inference.

# Arguments

  - `pe`: Augmented Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, regression result, and factor prior details.

# Validation

  - `dims in (1, 2)`.
  - `length(pe.a_sets.dict[pe.a_sets.key]) == size(X, 2)`.
  - `length(pe.f_sets.dict[pe.f_sets.key]) == size(F, 2)`.
  - If `pe.w` is not `nothing`, `length(pe.w) == size(X, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - Asset and factor priors are computed using the embedded prior estimators `pe.a_pe` and `pe.f_pe`.
  - Factor regression is performed using the regression estimator `pe.re`.
  - Asset and factor views are extracted using `black_litterman_views`, which returns the view matrices and view returns vectors.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of observations.
  - View uncertainty matrices for assets and factors are computed using `calc_omega`.
  - The augmented prior mean and covariance are constructed by combining asset and factor priors and regression loadings.
  - The augmented Black-Litterman posterior mean and covariance are computed using `vanilla_posteriors`.
  - Matrix processing is applied to the augmented posterior covariance.
  - The final asset posterior mean and covariance are extracted from the augmented results and adjusted for regression intercepts and risk-free rate.
  - The result includes asset returns, posterior mean, posterior covariance, regression result, and factor prior details.

# Related

  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.a_sets.dict[pe.a_sets.key]) == size(X, 2))
    @argcheck(length(pe.f_sets.dict[pe.f_sets.key]) == size(F, 2))
    # Asset prior.
    a_prior = prior(pe.a_pe, X; strict = strict, kwargs...)
    a_prior_mu, a_prior_sigma = a_prior.mu, a_prior.sigma
    # Factor prior.
    f_prior = prior(pe.f_pe, F; strict = strict)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    (; P, Q, excl) = black_litterman_views(pe.a_views, pe.a_sets;
                                           datatype = eltype(posterior_X), strict = strict)
    f_views = black_litterman_views(pe.f_views, pe.f_sets; datatype = eltype(posterior_X),
                                    strict = strict)
    f_P, f_Q, f_excl = f_views.P, f_views.Q, f_views.excl
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    a_views_conf = remove_excl_views(pe.a_views_conf, excl)
    f_views_conf = remove_excl_views(pe.f_views_conf, f_excl)
    a_omega = tau * calc_omega(a_views_conf, P, a_prior_sigma)
    f_omega = tau * calc_omega(f_views_conf, f_P, f_prior_sigma)
    aug_prior_sigma = hcat(vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                           vcat(M * f_prior_sigma, f_prior_sigma))
    aug_P = hcat(vcat(P, zeros(size(f_P, 1), size(P, 2))),
                 vcat(zeros(size(P, 1), size(f_P, 2)), f_P))
    aug_Q = vcat(Q, f_Q)
    aug_omega = hcat(vcat(a_omega, zeros(size(f_omega, 1), size(a_omega, 1))),
                     vcat(zeros(size(a_omega, 1), size(f_omega, 1)), f_omega))
    aug_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @argcheck(length(pe.w) == size(X, 2))
            pe.w
        else
            iN = inv(size(X, 2))
            range(iN, iN; length = size(X, 2))
        end
        pe.l * (vcat(a_prior_sigma, f_prior_sigma * transpose(M))) * w
    else
        vcat(a_prior_mu, f_prior_mu) .- pe.rf
    end
    aug_posterior_mu, aug_posterior_sigma = vanilla_posteriors(tau, pe.rf, aug_prior_mu,
                                                               aug_prior_sigma, aug_omega,
                                                               aug_P, aug_Q)
    matrix_processing!(pe.mp, aug_posterior_sigma, hcat(posterior_X, F))
    posterior_mu = (aug_posterior_mu[1:size(X, 2)] + b) .+ pe.rf
    posterior_sigma = aug_posterior_sigma[1:size(X, 2), 1:size(X, 2)]
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         rr = rr, f_mu = f_prior_mu, f_sigma = f_prior_sigma,
                         f_w = f_prior.w)
end

export AugmentedBlackLittermanPrior
