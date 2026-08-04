"""
$(DocStringExtensions.TYPEDEF)

Augmented Black-Litterman prior estimator for asset returns.

`AugmentedBlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using an augmented Black-Litterman model. It combines asset and factor prior estimators, matrix post-processing, regression and variance estimators, asset and factor views over one dual-axis universe sets, view confidences, weights, risk-free rate, leverage, and a blending parameter `tau`. This estimator supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates joint asset-factor Bayesian updating for posterior inference.

# Mathematical definition

Factor model linking assets and factors via regression:

```math
\\begin{align}
\\mathbf{X} &\\approx \\mathbf{F}\\mathbf{M}^{\\intercal} + \\mathbf{1}\\boldsymbol{b}^{\\intercal}\\,.
\\end{align}
```

Augmented prior moments (stacking asset and factor priors):

```math
\\begin{align}
\\boldsymbol{\\mu}_{aug} &= \\begin{pmatrix}\\boldsymbol{\\mu}_a \\\\ \\boldsymbol{\\mu}_f\\end{pmatrix}\\,, \\\\
\\boldsymbol{\\Sigma}_{aug} &= \\begin{pmatrix}\\boldsymbol{\\Sigma}_a & \\boldsymbol{\\Sigma}_a\\mathbf{M}^{\\intercal} \\\\ \\mathbf{M}\\boldsymbol{\\Sigma}_a & \\boldsymbol{\\Sigma}_f\\end{pmatrix}\\,.
\\end{align}
```

Black-Litterman posterior on the augmented space with combined views ``(\\mathbf{P}_{aug}, \\boldsymbol{q}_{aug})``:

```math
\\begin{align}
\\boldsymbol{\\mu}_{post} &= \\boldsymbol{\\mu}_{aug} + \\tau\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal}\\left(\\tau\\mathbf{P}_{aug}\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal} + \\boldsymbol{\\Omega}_{aug}\\right)^{-1}\\!\\left(\\boldsymbol{q}_{aug} - \\mathbf{P}_{aug}\\boldsymbol{\\mu}_{aug}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathbf{X}``: ``T \\times N`` asset returns matrix.
  - ``\\mathbf{F}``: ``T \\times K`` factor returns matrix.
  - ``\\mathbf{M}``: ``K \\times N`` factor loadings (regression coefficients).
  - ``\\boldsymbol{b}``: ``N \\times 1`` regression intercept vector.
  - ``\\boldsymbol{\\mu}_a``, ``\\boldsymbol{\\Sigma}_a``: Asset prior mean and covariance.
  - ``\\boldsymbol{\\mu}_f``, ``\\boldsymbol{\\Sigma}_f``: Factor prior mean and covariance.
  - ``\\boldsymbol{\\mu}_{aug}``, ``\\boldsymbol{\\Sigma}_{aug}``: Augmented (joint asset-factor) prior moments.
  - ``\\boldsymbol{\\mu}_{post}``: Augmented posterior mean (asset component extracted as final result).
  - ``\\tau``: Scaling parameter for the prior uncertainty.
  - ``\\mathbf{P}_{aug}``: Combined asset and factor views matrix.
  - ``\\boldsymbol{q}_{aug}``: Combined asset and factor views vector.
  - ``\\boldsymbol{\\Omega}_{aug}``: Combined view uncertainty matrix.

Asset posterior extracted from the augmented result and adjusted for intercept and risk-free rate.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AugmentedBlackLittermanPrior(;
        a_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        f_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        re::AbstractRegressionEstimator = StepwiseRegression(),
        a_views::Lc_BLV,
        f_views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        a_views_conf::Option{<:Num_VecNum} = nothing,
        f_views_conf::Option{<:Num_VecNum} = nothing,
        w::Option{<:VecNum} = nothing,
        rf::Number = 0.0,
        l::Option{<:Number} = nothing,
        tau::Option{<:Number} = nothing
    ) -> AugmentedBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **merges two** priors rather than forwarding one along its own axis, so it builds its carrier directly; the rule of ADR 0046 still governs which source each field takes. It solves one augmented Black-Litterman system over `[assets; factors]` and reports both halves:

  - `mu` and `sigma` are the asset half of the augmented posterior; the factor block `fpr` is the **factor half**, so both are posterior. `chol` is dropped on both sides, because the posterior covariances supersede the ones they factorise.
  - `w`, `ens`, `kld` and `ow` come from the **asset** prior, and `fpr`'s own come from the **factor** prior. Two priors disagreeing about observation weights is a legitimate configuration, and the nested block is what keeps the two weightings distinguishable rather than forcing a choice.
  - `Z` comes from the asset prior only: the factor prior's would be factors × features and would not describe this asset axis.

!!! warning

    The returned `mu` and `sigma` are the augmented posterior, but `w` is the **asset prior's** observation weighting, forwarded unchanged (and `fpr.w` is the factor prior's). Black-Litterman produces no observation-level posterior, so there is no Black-Litterman-consistent alternative to forward — and dropping `w` would substitute the unweighted empirical distribution, which is further from the caller's intent than the weights they computed. A caller reading `pr.w`, `pr.ens`, `pr.kld` or `pr.ow` is therefore reading a property of the asset prior, not of the posterior.

!!! warning

    `pr.mu != pr.rr.M * pr.fpr.mu + pr.rr.b`, even though **both** blocks are posterior. The gap is opened by the update, and its cause is **idiosyncratic variance**. (Give `a_pe` and `f_pe` the same observation weighting and the two *priors* do satisfy the identity to machine precision, because least squares with an intercept reproduces the mean. Weighting them differently — two independently pooled priors, say — breaks it before the update as well, so that case carries both causes at once.)

    The augmented covariance stacks the full asset covariance `sigma_a` — factor *and* residual variance — against a cross-covariance `M * sigma_f` that is pure factor. The Black-Litterman update therefore moves the asset half by `tau * sigma_a * P'(…)` and the factor half by `tau * sigma_f * M' * P'(…)`, and for the two to stay related by `M` it would need `sigma_a == M * sigma_f * M'`. That holds only when the factor model is exact. The gap scales with the residual variance and closes to machine precision when there is none; both view sets contribute, so muting either one does not remove it.

    [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) satisfy the identity exactly, because they update the factor distribution alone and *project* it onto the assets rather than updating an asset block alongside it. [`BlackLittermanPrior`](@ref) breaks it for the opposite reason — it takes asset views only and never computes a posterior factor distribution at all.

## One sets, two axes

This is the only estimator whose views land on **both** distributions, and the two axes it needs are the two [`UniverseSets`](@ref) declares: `a_views` resolves against `sets.dict[sets.xkey]`, `f_views` against `sets.dict[sets.fkey]`. Before the axis was declared this took two separate sets objects, and the factor-flavoured one had to be exempted from [`port_opt_view`](@ref) **by hand** — a missing annotation was all that stood between a view and a factor universe sliced by asset indices. With one dual-axis object the exemption is a property of the data: the field is `@vprop`, the slice moves the asset entries and the factor entries come back untouched.

Each axis is required only by the views that resolve names against it. A [`BlackLittermanViews`](@ref) result carries its own `P` and needs no universe at all, so asset-views-only and factor-views-only mandates are both expressible with a single `sets` — and a pair of precomputed view sets needs none.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.
  - If either `a_views` or `f_views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `a_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `f_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

# Examples

```jldoctest
julia> AugmentedBlackLittermanPrior(;
                                    sets = UniverseSets(;
                                                        dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"],
                                                                    \"nf\" => [\"F1\", \"F2\"])),
                                    a_views = LinearConstraintEstimator(;
                                                                        val = [\"A == 0.03\",
                                                                               \"B + C == 0.04\"]),
                                    f_views = LinearConstraintEstimator(;
                                                                        val = [\"F1 == 0.01\",
                                                                               \"F2 == 0.02\"]))
AugmentedBlackLittermanPrior
          a_pe ┼ EmpiricalPrior
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
          f_pe ┼ EmpiricalPrior
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
       a_views ┼ LinearConstraintEstimator
               │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
               │   key ┴ nothing
       f_views ┼ LinearConstraintEstimator
               │   val ┼ Vector{String}: ["F1 == 0.01", "F2 == 0.02"]
               │   key ┴ nothing
          sets ┼ UniverseSets
               │    xkey ┼ String: "nx"
               │   uxkey ┼ String: "ux"
               │    fkey ┼ String: "nf"
               │   ufkey ┼ String: "uf"
               │    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"], "nf" => ["F1", "F2"])
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
  - [`UniverseSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
@propagatable @concrete struct AugmentedBlackLittermanPrior <:
                               AbstractLowOrderPriorEstimator_F
    """
    $(field_dict[:a_pe])
    """
    @fprop @vprop a_pe
    """
    $(field_dict[:f_pe])
    """
    @fprop f_pe
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:re])
    """
    @fprop @vprop re
    """
    $(field_dict[:a_views])
    """
    a_views
    """
    $(field_dict[:f_views])
    """
    f_views
    """
    $(field_dict[:sets_af])
    """
    @vprop sets
    """
    $(field_dict[:a_views_conf])
    """
    a_views_conf
    """
    $(field_dict[:f_views_conf])
    """
    f_views_conf
    """
    $(field_dict[:eqw])
    """
    @vprop w
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
    function AugmentedBlackLittermanPrior(a_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          f_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          mp::AbstractMatrixProcessingEstimator,
                                          re::AbstractRegressionEstimator, a_views::Lc_BLV,
                                          f_views::Lc_BLV, sets::Option{<:UniverseSets},
                                          a_views_conf::Option{<:Num_VecNum},
                                          f_views_conf::Option{<:Num_VecNum},
                                          w::Option{<:VecNum}, rf::Number,
                                          l::Option{<:Number}, tau::Option{<:Number})
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        end
        # One sets now serves both view sets, so the shared helper takes it twice and differs
        # only in which views it is validating — the axis each resolves against is a property
        # of the *read*, checked in `prior` where the matrices are in hand, not of the field.
        assert_bl(a_views, sets, a_views_conf, tau)
        assert_bl(f_views, sets, f_views_conf, tau)
        return new{typeof(a_pe), typeof(f_pe), typeof(mp), typeof(re), typeof(a_views),
                   typeof(f_views), typeof(sets), typeof(a_views_conf),
                   typeof(f_views_conf), typeof(w), typeof(rf), typeof(l), typeof(tau)}(a_pe,
                                                                                        f_pe,
                                                                                        mp,
                                                                                        re,
                                                                                        a_views,
                                                                                        f_views,
                                                                                        sets,
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
                                      mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                      re::AbstractRegressionEstimator = StepwiseRegression(),
                                      a_views::Lc_BLV, f_views::Lc_BLV,
                                      sets::Option{<:UniverseSets} = nothing,
                                      a_views_conf::Option{<:Num_VecNum} = nothing,
                                      f_views_conf::Option{<:Num_VecNum} = nothing,
                                      w::Option{<:VecNum} = nothing, rf::Number = 0.0,
                                      l::Option{<:Number} = nothing,
                                      tau::Option{<:Number} = nothing)::AugmentedBlackLittermanPrior
    return AugmentedBlackLittermanPrior(a_pe, f_pe, mp, re, a_views, f_views, sets,
                                        a_views_conf, f_views_conf, w, rf, l, tau)
end
# Expose `:me`, `:ce` from the asset prior `a_pe` and (renamed) `:f_me`, `:f_ce` from the
# factor prior `f_pe` for transparent access (see [`@forward_properties`](@ref)).
@forward_properties AugmentedBlackLittermanPrior begin
    forward(a_pe, me, ce)
    alias(f_me, f_pe.me)
    alias(f_ce, f_pe.ce)
end
"""
    prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute augmented Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the augmented Black-Litterman model, combining asset and factor prior estimators, matrix post-processing, regression and variance estimators, asset and factor views over one dual-axis universe sets, view confidences, weights, risk-free rate, leverage, and a blending parameter `tau`. This method supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates joint asset-factor Bayesian updating for posterior inference.

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
  - If `pe.a_views` is a [`LinearConstraintEstimator`](@ref), `length(pe.sets.dict[pe.sets.xkey]) == size(X, 2)`.
  - If `pe.f_views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.fkey)` and `length(pe.sets.dict[pe.sets.fkey]) == size(F, 2)`, both via [`factor_universe`](@ref).
  - If `pe.w` is not `nothing`, `length(pe.w) == size(X, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - Asset and factor priors are computed using the embedded prior estimators `pe.a_pe` and `pe.f_pe`.
  - Factor regression is performed using the regression estimator `pe.re`.
  - Asset and factor views are extracted using [`black_litterman_views`](@ref), which returns the view matrices and view returns vectors. The asset views resolve at `pe.sets.xkey` and the factor views **at `pe.sets.fkey`**, out of the one dual-axis sets.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of observations.
  - View uncertainty matrices for assets and factors are computed using `calc_omega`.
  - The augmented prior mean and covariance are constructed by combining asset and factor priors and regression loadings.
  - The augmented Black-Litterman posterior mean and covariance are computed using `vanilla_posteriors`.
  - Matrix processing is applied to the augmented posterior covariance.
  - The final asset posterior mean and covariance are extracted from the augmented results and adjusted for regression intercepts and risk-free rate.
  - The factor block is the **factor half** of the same augmented posterior, so both halves are posterior. It takes no intercept and no risk-free adjustment (the intercept is the regression's, hence asset-only) and no second matrix processing pass, since the augmented covariance was processed as a whole and a principal submatrix of it is already processed.

# Related

  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    # Each axis is checked only by the views that resolve names against it. A
    # `BlackLittermanViews` result carries its own `P` and never touches `sets`, so demanding a
    # universe for it would reject the legitimate precomputed-views configuration — which, with
    # both view sets precomputed, is allowed to supply no `sets` at all.
    if isa(pe.a_views, LinearConstraintEstimator)
        @argcheck(length(pe.sets.dict[pe.sets.xkey]) == size(X, 2),
                  DimensionMismatch("length(pe.sets.dict[pe.sets.xkey]) ($(length(pe.sets.dict[pe.sets.xkey]))) must match size(X, 2) ($(size(X, 2)))"))
    end
    if isa(pe.f_views, LinearConstraintEstimator)
        # The factor views land on the *factor* distribution, so they resolve against the
        # declared factor axis — not against `xkey`, which names the assets `a_views` uses.
        factor_universe(pe.sets, size(F, 2),
                        "AugmentedBlackLittermanPrior, whose `f_views` are written in factor names",
                        "F")
    end
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
    dt = eltype(posterior_X)
    T = size(X, 1)
    # One sets, two keys: the asset views take the default (`xkey`), the factor views `fkey`.
    # Precomputed views ignore both the sets and the key, and are the one shape that reaches
    # here with `pe.sets === nothing`, so the key is read only when there is a sets to read it
    # from.
    f_key = isnothing(pe.sets) ? nothing : pe.sets.fkey
    (; P, Q, tau, omega) = bl_preroll(pe.a_views, pe.sets, pe.a_views_conf, a_prior_sigma,
                                      pe.tau, T, dt, strict)
    a_omega = omega
    f_result = bl_preroll(pe.f_views, pe.sets, pe.f_views_conf, f_prior_sigma, pe.tau, T,
                          dt, strict, f_key)
    f_P, f_Q, f_omega = f_result.P, f_result.Q, f_result.omega
    aug_prior_sigma = hcat(vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                           vcat(M * f_prior_sigma, f_prior_sigma))
    aug_P = hcat(vcat(P, zeros(size(f_P, 1), size(P, 2))),
                 vcat(zeros(size(P, 1), size(f_P, 2)), f_P))
    aug_Q = vcat(Q, f_Q)
    aug_omega = hcat(vcat(a_omega, zeros(size(f_omega, 1), size(a_omega, 1))),
                     vcat(zeros(size(a_omega, 1), size(f_omega, 1)), f_omega))
    aug_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @argcheck(length(pe.w) == size(X, 2),
                      DimensionMismatch("length(pe.w) ($(length(pe.w))) must match size(X, 2) ($(size(X, 2)))"))
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
    # The augmented system is jointly posterior over `[assets; factors]`, so truncating it to
    # the asset half discards a factor half that *is* the posterior factor distribution. The
    # factor block reports that half rather than `f_prior`'s prior moments. No `rf` shift and
    # no `b`: the intercept is the regression's, hence asset-only, and `vanilla_posteriors`
    # already returns the factor half on the same scale `f_prior.mu` was supplied on. No second
    # `matrix_processing!` either — `aug_posterior_sigma` was processed as a whole above, and a
    # principal submatrix of the result is already processed.
    f_idx = (size(X, 2) + 1):length(aug_posterior_mu)
    # `chol` is the factor block's only drop: the posterior covariance supersedes the one
    # `f_prior.chol` factorises. The views do not touch the observation axis, so `f_prior`'s own
    # `w` and its diagnostics forward untouched (ADR 0046).
    fpr = forward_prior(f_prior; mu = aug_posterior_mu[f_idx],
                        sigma = aug_posterior_sigma[f_idx, f_idx], chol = nothing)
    # The feature matrix comes from `a_prior` only: `f_prior` is fit on the factors, so its
    # `Z` would be factors × features and would not describe this asset axis. The augmented
    # system is truncated straight back to the assets, so `a_prior.Z` still binds correctly.
    # `posterior_X` is reconstructed from the factors, so a forwarded `Z` is dimension-correct
    # but was derived from the pre-reconstruction returns (see [`LowOrderPrior`](@ref)).
    #
    # The factor block's `w` and diagnostics are `f_prior`'s, which is how the two priors'
    # weightings stay distinguishable — `w` is `a_prior`'s, `fpr.w` is `f_prior`'s. The asset
    # slot takes `a_prior`'s because a `@pprop w` consumer applies `w` to a return series over
    # the *assets*; two priors disagreeing about observation weights is a legitimate
    # configuration, and there are two slots to hold them. The diagnostics follow their
    # weights (ADR 0046), so `ens`/`kld`/`ow` come from `a_prior` too. `chol` is dropped:
    # `posterior_sigma` supersedes the covariance `a_prior.chol` factorises. This site merges
    # two priors rather than forwarding one along its own axis, so it builds the carrier
    # directly instead of going through [`forward_prior`](@ref).
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         w = a_prior.w, ens = a_prior.ens, kld = a_prior.kld,
                         ow = a_prior.ow, rr = rr, fpr = fpr, Z = a_prior.Z)
end

export AugmentedBlackLittermanPrior
