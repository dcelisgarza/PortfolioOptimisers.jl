"""
$(DocStringExtensions.TYPEDEF)

Augmented Black-Litterman prior estimator for asset returns.

`AugmentedBlackLittermanPrior` is a low order prior estimator. It stacks an asset prior and a factor prior into one joint prior. It applies asset views and factor views to the joint prior with the Black-Litterman master equations. The result is the asset half and the factor half of the posterior.

# Mathematical definition

The regression of the assets on the factors gives the factor model:

```math
\\begin{align}
\\mathbf{X} &\\approx \\mathbf{F}\\mathbf{M}^{\\intercal} + \\mathbf{1}\\boldsymbol{b}^{\\intercal}\\,.
\\end{align}
```

The augmented prior moments stack the asset prior and the factor prior:

```math
\\begin{align}
\\boldsymbol{\\mu}_{aug} &= \\begin{pmatrix}\\boldsymbol{\\mu}_a \\\\ \\boldsymbol{\\mu}_f\\end{pmatrix}
\\quad\\text{or}\\quad
\\lambda\\begin{pmatrix}\\boldsymbol{\\Sigma}_a \\\\ \\boldsymbol{\\Sigma}_f\\mathbf{M}^{\\intercal}\\end{pmatrix}\\boldsymbol{w} + \\begin{pmatrix}\\boldsymbol{b} \\\\ \\mathbf{0}\\end{pmatrix} + r_{f}\\,, \\\\
\\boldsymbol{\\Sigma}_{aug} &= \\begin{pmatrix}\\boldsymbol{\\Sigma}_a & \\mathbf{M}\\boldsymbol{\\Sigma}_f \\\\ \\boldsymbol{\\Sigma}_f\\mathbf{M}^{\\intercal} & \\boldsymbol{\\Sigma}_f\\end{pmatrix}\\,.
\\end{align}
```

The left prior mean stacks the two wrapped means, and the estimator uses it when `pe.l === nothing`. The right prior mean is the equilibrium mean that `pe.l` selects. It adds two levels that the wrapped means contain already: the intercept ``\\boldsymbol{b}`` on the asset half, and the risk-free rate. Both levels must be in ``\\boldsymbol{\\mu}_{aug}`` before the update, because the update blends this mean against ``\\boldsymbol{q}_{aug}`` through the residual ``\\boldsymbol{q}_{aug} - \\mathbf{P}_{aug}\\boldsymbol{\\mu}_{aug}``.

The off-diagonal blocks are the cross-covariance of the factor model, ``\\mathrm{cov}(\\mathbf{X}, \\mathbf{F}) = \\mathbf{M}\\boldsymbol{\\Sigma}_f``. The estimator builds them from the **factor** covariance, not from the asset covariance. So they hold only factor variance, but the first diagonal block holds factor *and* residual variance. This difference opens the gap of the second warning below.

The estimator stacks the views block-diagonally, each set over its own axis:

```math
\\begin{align}
\\mathbf{P}_{aug} &= \\begin{pmatrix}\\mathbf{P} & \\mathbf{0} \\\\ \\mathbf{0} & \\mathbf{P}_f\\end{pmatrix}\\,, \\quad
\\boldsymbol{q}_{aug} = \\begin{pmatrix}\\boldsymbol{q} \\\\ \\boldsymbol{q}_f\\end{pmatrix}\\,, \\quad
\\boldsymbol{\\Omega}_{aug} = \\begin{pmatrix}\\boldsymbol{\\Omega} & \\mathbf{0} \\\\ \\mathbf{0} & \\boldsymbol{\\Omega}_f\\end{pmatrix}\\,.
\\end{align}
```

The master equations give the Black-Litterman posterior on the augmented space:

```math
\\begin{align}
\\boldsymbol{\\mu}_{post} &= \\boldsymbol{\\mu}_{aug} + \\tau\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal}\\left(\\tau\\mathbf{P}_{aug}\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal} + \\boldsymbol{\\Omega}_{aug}\\right)^{-1}\\!\\left(\\boldsymbol{q}_{aug} - \\mathbf{P}_{aug}\\boldsymbol{\\mu}_{aug}\\right)\\,, \\\\
\\boldsymbol{\\Sigma}_{post} &= \\boldsymbol{\\Sigma}_{aug} + \\tau\\boldsymbol{\\Sigma}_{aug} - \\tau\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal}\\left(\\tau\\mathbf{P}_{aug}\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal} + \\boldsymbol{\\Omega}_{aug}\\right)^{-1}\\!\\mathbf{P}_{aug}\\tau\\boldsymbol{\\Sigma}_{aug}\\,.
\\end{align}
```

Truncation then gives the two halves, with no other change. The intercept and the risk-free rate are already in ``\\boldsymbol{\\mu}_{aug}`` above, where the update blends the views against them:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\left(\\boldsymbol{\\mu}_{post}\\right)_{1:N}\\,, \\quad
\\hat{\\mathbf{\\Sigma}} = \\left(\\boldsymbol{\\Sigma}_{post}\\right)_{1:N,\\,1:N}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_f &= \\left(\\boldsymbol{\\mu}_{post}\\right)_{N+1:N+K}\\,, \\quad
\\hat{\\mathbf{\\Sigma}}_f = \\left(\\boldsymbol{\\Sigma}_{post}\\right)_{N+1:N+K,\\,N+1:N+K}\\,.
\\end{align}
```

Where:

  - ``T``, ``N``, ``K``: The number of observations, of assets, and of factors.
  - ``\\mathbf{X}``: ``T \\times N`` asset returns matrix.
  - ``\\mathbf{F}``: ``T \\times K`` factor returns matrix.
  - ``\\mathbf{M}``: ``N \\times K`` factor loadings, the regression coefficients, `pr.rr.M`.
  - ``\\boldsymbol{b}``: ``N \\times 1`` regression intercept vector, `pr.rr.b`.
  - ``\\boldsymbol{\\mu}_a``, ``\\boldsymbol{\\Sigma}_a``: ``N \\times 1`` and ``N \\times N`` asset prior mean and covariance, from `a_pe`.
  - ``\\boldsymbol{\\mu}_f``, ``\\boldsymbol{\\Sigma}_f``: ``K \\times 1`` and ``K \\times K`` factor prior mean and covariance, from `f_pe`.
  - ``\\boldsymbol{\\mu}_{aug}``, ``\\boldsymbol{\\Sigma}_{aug}``: Augmented prior moments over the joint axis of assets and factors, of length and order ``N + K``.
  - ``\\boldsymbol{\\mu}_{post}``, ``\\boldsymbol{\\Sigma}_{post}``: Augmented posterior moments, of the same shape.
  - ``\\tau``: Scaling parameter for the prior uncertainty, `1/T` by default.
  - ``\\mathbf{P}``, ``\\boldsymbol{q}``, ``\\boldsymbol{\\Omega}``: Asset views matrix, returns vector and uncertainty matrix, over the asset axis.
  - ``\\mathbf{P}_f``, ``\\boldsymbol{q}_f``, ``\\boldsymbol{\\Omega}_f``: The same three over the factor axis.
  - ``\\mathbf{P}_{aug}``, ``\\boldsymbol{q}_{aug}``, ``\\boldsymbol{\\Omega}_{aug}``: The same three stacked over the augmented axis, the asset rows above the factor rows.
  - ``\\hat{\\boldsymbol{\\mu}}``, ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times 1`` and ``N \\times N`` posterior asset moments, `pr.mu` and `pr.sigma`.
  - ``\\hat{\\boldsymbol{\\mu}}_f``, ``\\hat{\\mathbf{\\Sigma}}_f``: ``K \\times 1`` and ``K \\times K`` posterior factor moments, `pr.fpr.mu` and `pr.fpr.sigma`.
  - ``\\lambda``, ``\\boldsymbol{w}``: The risk-aversion coefficient `pe.l` and the equilibrium weights `pe.w`. [`equilibrium_mu`](@ref) reads them, and only when `pe.l` is set.
  - ``r_{f}``: Risk-free rate. [`apply_rf`](@ref) adds it once, to the equilibrium mean. It is absent when `pe.l` is `nothing`, because the wrapped means are total returns already.

The stacked inputs and the master equations follow [cajas2025](@citet), who follows [cheung2007](@citet). [cajas2025](@citet) builds the equilibrium mean as ``\\delta\\,(\\boldsymbol{\\Sigma};\\ \\boldsymbol{\\Sigma}_f\\mathbf{M}^{\\intercal})\\,\\boldsymbol{w}`` with no intercept, and adds ``r_f`` to the posterior. The library adds the intercept and ``r_f`` to the equilibrium mean before the update, as the text above states. [cajas2025](@citet) does not state the diagonal blocks of ``\\boldsymbol{\\Omega}_{aug}``. The library sets each block to ``\\tau\\,\\mathrm{Diag}(\\mathbf{P}\\mathbf{\\Sigma}\\mathbf{P}^\\intercal)`` over the covariance of its own axis.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AugmentedBlackLittermanPrior(;
        a_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        f_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
        a_views::Lc_BLV,
        f_views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        a_views_conf::Option{<:Num_VecNum} = nothing,
        f_views_conf::Option{<:Num_VecNum} = nothing,
        w::Option{<:VecNum} = nothing,
        rf::Number = 0.0,
        l::Option{<:Number} = nothing,
        tau::Option{<:Number} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> AugmentedBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **merges two** priors and builds its prior result directly. It does not forward one prior along its own axis. A posterior field still replaces the prior field of the same name, and a structural field still forwards whole. The estimator solves one augmented Black-Litterman system over `[assets; factors]`, and it gives both halves:

  - `mu` and `sigma` are the asset half of the augmented posterior. The factor block `fpr` is the **factor half**, so both blocks are posterior. The estimator drops `chol` on both blocks, because the posterior covariances replace the covariances that `chol` factorises.
  - `w`, `ens`, `kld` and `ow` come from the **asset** prior. The same fields of `fpr` come from the **factor** prior. Two priors with different observation weights are a valid configuration, and the nested block keeps the two weightings apart.
  - `pc`, the cache of an opinion pool, comes from the asset prior only. The cache of the factor prior holds factor returns, and it does not describe the asset axis.

!!! warning

    The returned `mu` and `sigma` are the augmented posterior, but `w` is the observation weighting of the **asset prior**, forwarded with no change. `fpr.w` is the weighting of the factor prior. Black-Litterman gives no posterior over the observations, so no weighting that agrees with the posterior exists. If the estimator drops `w`, the unweighted empirical distribution takes its place, and that is further from the weights that the caller computed. So `pr.w`, `pr.ens`, `pr.kld` and `pr.ow` describe the asset prior, not the posterior. With an entropy pooling prior on each axis, these four fields equal those of the asset prior. `pr.fpr.w` equals the weighting of the factor prior, and it differs from `pr.w`.

!!! warning

    `pr.mu != pr.rr.M * pr.fpr.mu + pr.rr.b`, but **both** blocks are posterior. The update opens the gap in two ways.

    **Idiosyncratic variance, on every branch.** The augmented covariance stacks the full asset covariance `sigma_a` against the cross-covariance `M * sigma_f`. `sigma_a` holds factor *and* residual variance, but `M * sigma_f` holds factor variance alone. So the update moves the asset half by `tau * sigma_a * P'(…)` and the factor half by `tau * sigma_f * M' * P'(…)`. The two halves stay related by `M` only if `sigma_a == M * sigma_f * M'`, and that holds only on an exact factor model. This part grows with the residual variance, and both view sets add to it.

    The intercept is not a third cause. On an exact factor model with a non-zero intercept, the gap is `1.8e-15`. But with a residual, the intercept changes the size of the first part. On a `250 × 5` sample over three factors, the gap is `7.1e-5`. When the intercept is removed from `X`, with no other change, the gap is `1.2e-4`.

    **A non-zero `rf`, on the `l` branch alone.** [`apply_rf`](@ref) adds the rate to the whole equilibrium stack, so the asset half gets `rf` and the factor half gets `rf` too. The identity carries the factor half through the loadings, and this turns that `rf` into `rf * s` for the row sums `s` of `M`. On an exact factor model, the gap is `rf * (1 - s)`, to `1.5e-14` at `rf = 0.03`. With a residual, the rate also goes through the first part, and the gap is not `rf * (1 - s)`. On the same sample at `rf = 0.03`, the part of the gap that the rate adds differs from `rf * (1 - s)` by up to `5.9e-4`. This part is zero at the default `rf = 0.0`. When `pe.l` is `nothing`, no step reads the field.

    Before the update, the two *priors* satisfy the identity only when both means are the plain sample mean. Least squares with an intercept sets the **unweighted** residual mean to zero. One shared *non-uniform* weighting is not enough, because the weighted residual mean is not zero. Two different weightings make the gap larger. On an exact factor model, a shared non-uniform weighting satisfies the identity, because no residual exists.

    [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) satisfy the identity exactly. They update the factor distribution alone, and they *project* it onto the assets. They do not update an asset block next to it. [`BlackLittermanPrior`](@ref) takes asset views only, and it computes no posterior factor distribution. When it wraps a prior with no factor block, `pr.fpr` is `nothing`, and the right side of the identity does not exist. When it wraps a [`FactorPrior`](@ref), `pr.fpr` is the prior factor block, and the gap on the same sample is `1.7e-3`.

## One sets, two axes

This is the only estimator with views on **both** distributions. It needs two axes, and [`UniverseSets`](@ref) declares both. `a_views` resolves against `sets.dict[sets.xkey]`, and `f_views` resolves against `sets.dict[sets.tfkey]`. The field is `@vprop`. [`port_opt_view`](@ref) slices the asset entries, and the factor entries come back with no change. The data, and not a hand-written exemption, keeps the factor universe whole.

The views that resolve names against an axis are the only views that need that axis. A [`BlackLittermanViews`](@ref) result carries its own `P` and needs no universe. With one `sets`, a mandate can have asset views only or factor views only. A pair of precomputed view sets needs no `sets`, and it gives the same result as the same pair with `sets`. [`port_opt_view`](@ref) over a selection of assets keeps the factor axis of `sets` whole, and it cuts the asset axis and `w` to the selection.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.
  - If either `a_views` or `f_views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `a_views_conf` is not `nothing`, [`assert_bl_views_conf`](@ref) checks it.
  - If `f_views_conf` is not `nothing`, [`assert_bl_views_conf`](@ref) checks it.
  - If `tau` is not `nothing`, `tau > 0`.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `a_pe`, `f_pe`, `re` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).

The constructor does not check the **length** of `w`. The length is a property of the returns matrix, and the constructor does not see that matrix. A wrong length raises a `DimensionMismatch` from [`equilibrium_mu`](@ref) in [`prior`](@ref), and only when `l` is set.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `a_pe`: Recursively updated via [`factory`](@ref).
  - `f_pe`: Recursively updated via [`factory`](@ref).
  - `re`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `a_pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `re`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

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
               │           ce ┼ PortfolioOptimisersCovariance
               │              │   ce ┼ Covariance
               │              │      │    me ┼ SimpleExpectedReturns
               │              │      │       │   w ┴ nothing
               │              │      │    ce ┼ GeneralCovariance
               │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
               │              │      │       │    w ┴ nothing
               │              │      │   alg ┼ FullMoment()
               │              │      │     w ┴ nothing
               │              │   mp ┼ MatrixProcessing
               │              │      │     pdm ┼ Posdef
               │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
               │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
               │              │      │      dn ┼ nothing
               │              │      │      dt ┼ nothing
               │              │      │     alg ┼ nothing
               │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
               │           me ┼ SimpleExpectedReturns
               │              │   w ┴ nothing
               │      horizon ┼ nothing
               │   fill_limit ┼ nothing
               │        x_src ┴ nothing
          f_pe ┼ EmpiricalPrior
               │           ce ┼ PortfolioOptimisersCovariance
               │              │   ce ┼ Covariance
               │              │      │    me ┼ SimpleExpectedReturns
               │              │      │       │   w ┴ nothing
               │              │      │    ce ┼ GeneralCovariance
               │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
               │              │      │       │    w ┴ nothing
               │              │      │   alg ┼ FullMoment()
               │              │      │     w ┴ nothing
               │              │   mp ┼ MatrixProcessing
               │              │      │     pdm ┼ Posdef
               │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
               │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
               │              │      │      dn ┼ nothing
               │              │      │      dt ┼ nothing
               │              │      │     alg ┼ nothing
               │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
               │           me ┼ SimpleExpectedReturns
               │              │   w ┴ nothing
               │      horizon ┼ nothing
               │   fill_limit ┼ nothing
               │        x_src ┴ nothing
            mp ┼ MatrixProcessing
               │     pdm ┼ Posdef
               │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
               │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
               │      dn ┼ nothing
               │      dt ┼ nothing
               │     alg ┼ nothing
               │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
            re ┼ StepwiseRegression
               │       crit ┼ PValue
               │            │   t ┴ Float64: 0.05
               │        alg ┼ ForwardSelection()
               │        tgt ┼ LinearModel
               │            │   kwargs ┴ @NamedTuple{}: NamedTuple()
               │     choice ┼ BatchChoice()
               │   included ┼ nothing
               │      x_src ┴ nothing
       a_views ┼ LinearConstraintEstimator
               │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
               │   key ┴ nothing
       f_views ┼ LinearConstraintEstimator
               │   val ┼ Vector{String}: ["F1 == 0.01", "F2 == 0.02"]
               │   key ┴ nothing
          sets ┼ UniverseSets
               │     xkey ┼ String: "nx"
               │    uxkey ┼ String: "ux"
               │    tfkey ┼ String: "nf"
               │   utfkey ┼ String: "uf"
               │    cfkey ┼ String: "ncf"
               │   ucfkey ┼ String: "ucf"
               │    nikey ┼ String: "ni"
               │     dict ┴ Dict{String, Vector{String}}: Dict("nf" => ["F1", "F2"], "nx" => ["A", "B", "C"])
  a_views_conf ┼ nothing
  f_views_conf ┼ nothing
             w ┼ nothing
            rf ┼ Float64: 0.0
             l ┼ nothing
           tau ┼ nothing
         x_src ┴ nothing
```

## The incremental fit

This prior has no exact incremental recursion, and it takes the online step by a **refit from a sample buffer**. [`Online`](@ref) seeds `cache`, and [`partial_fit!`](@ref) adds each observation to it with no change. The [`prior`](@ref) method with one argument then runs the batch verb of this estimator over the rows that the buffer kept. The result is the batch fit over those rows, and a `max_history` on the wrapper sets the window of the whole fit.

`cache` goes through the three propagation channels as every partial-fit state does. [`factory`](@ref) carries it with no change, and [`port_opt_view`](@ref) cuts it to the selected assets. [`obs_weights_view`](@ref) drops it, because a state has no cut on the observation axis. The show method does not print `cache`, because a buffer that changes at each step is not part of the configuration.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`UniverseSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cheung2007])
  - $(ref_dict[:cajas2025]) Section 5.2, Equations 5.17 to 5.19. $(ref_dict[:no_original_source])
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
    $(field_dict[:bl_rf])
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
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function AugmentedBlackLittermanPrior(a_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          f_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          mp::AbstractMatrixProcessingEstimator,
                                          re::AbstractTimeSeriesRegressionEstimator,
                                          a_views::Lc_BLV, f_views::Lc_BLV,
                                          sets::Option{<:UniverseSets},
                                          a_views_conf::Option{<:Num_VecNum},
                                          f_views_conf::Option{<:Num_VecNum},
                                          w::Option{<:VecNum}, rf::Number,
                                          l::Option{<:Number}, tau::Option{<:Number},
                                          cache::Option{<:AbstractPartialFitState},
                                          x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :AugmentedBlackLittermanPrior)
        assert_inner_reader(a_pe, :a_pe, :AugmentedBlackLittermanPrior)
        assert_inner_reader(f_pe, :f_pe, :AugmentedBlackLittermanPrior)
        assert_inner_reader(re, :re, :AugmentedBlackLittermanPrior)
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
                   typeof(f_views_conf), typeof(w), typeof(rf), typeof(l), typeof(tau),
                   typeof(cache), typeof(x_src)}(a_pe, f_pe, mp, re, a_views, f_views, sets,
                                                 a_views_conf, f_views_conf, w, rf, l, tau,
                                                 cache, x_src)
    end
end
function AugmentedBlackLittermanPrior(;
                                      a_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                      f_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                      mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                      re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
                                      a_views::Lc_BLV, f_views::Lc_BLV,
                                      sets::Option{<:UniverseSets} = nothing,
                                      a_views_conf::Option{<:Num_VecNum} = nothing,
                                      f_views_conf::Option{<:Num_VecNum} = nothing,
                                      w::Option{<:VecNum} = nothing, rf::Number = 0.0,
                                      l::Option{<:Number} = nothing,
                                      tau::Option{<:Number} = nothing,
                                      cache::Option{<:AbstractPartialFitState} = nothing,
                                      x_src::Option{<:AbstractReturnsSource} = nothing)::AugmentedBlackLittermanPrior
    return AugmentedBlackLittermanPrior(a_pe, f_pe, mp, re, a_views, f_views, sets,
                                        a_views_conf, f_views_conf, w, rf, l, tau, cache,
                                        x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give every field of an [`AugmentedBlackLittermanPrior`](@ref) to print, except `cache`.

The state in `cache` changes at each step of an incremental fit, and it is not part of the configuration. Each site that prints the estimator also prints this state, unless this method leaves it out. Set `set_show_nothing_fields!(:AugmentedBlackLittermanPrior, true)` to print it.

# Arguments

  - `::AugmentedBlackLittermanPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:a_pe, :f_pe, :mp, :re, :a_views, :f_views, :sets, :a_views_conf, :f_views_conf, :w, :rf, :l, :tau, :x_src)`.

# Related

  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::AugmentedBlackLittermanPrior)
    return (:a_pe, :f_pe, :mp, :re, :a_views, :f_views, :sets, :a_views_conf, :f_views_conf,
            :w, :rf, :l, :tau, :x_src)
end
# Expose `:me`, `:ce` from the asset prior `a_pe` and (renamed) `:f_me`, `:f_ce` from the
# factor prior `f_pe` for transparent access (see [`@forward_properties`](@ref)).
@forward_properties AugmentedBlackLittermanPrior begin
    forward(a_pe, me, ce)
    alias(f_me, f_pe.me)
    alias(f_ce, f_pe.ce)
end
"""
    prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1,
          strict::Bool = false, cache_reg::Bool = false, kwargs...)

Compute augmented Black-Litterman prior moments for asset returns.

`prior` fits the asset prior and the factor prior, and it stacks them into one joint prior. It applies the asset views and the factor views to the joint prior with the Black-Litterman master equations.

When `pe.tau` is `nothing`, the blending parameter is `1/T`, where `T` is the number of observations of the oriented `X`. The update gets a **total** return as its mean, and both halves have that scale. `pe.rf` reaches the update only when `pe.l` is set. There it converts the equilibrium risk premium, and it goes on the whole stack. Let ``\\mathbf{G}`` be the augmented update gain, and let ``\\mathbf{1}`` be the vector of ones. Against the same estimator at `rf = 0`, the result moves by `rf * (I - G P_aug) * 1` on each half. When `pe.l` is `nothing`, no step reads the field, and two fits that differ only in `rf` are equal on both halves.

# Arguments

  - `pe`: Augmented Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:pnl_prior]) It reaches `pe.a_pe`, the prior over the assets. It does not reach `pe.f_pe`, because the estimator fits `pe.f_pe` on the factors and no panel describes the factor axis.
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - $(arg_dict[:cache_reg]) It reaches `pe.a_pe` and `pe.f_pe`.
  - `kwargs...`: More keyword arguments, for `pe.a_pe`. They describe the asset axis, for example an active mask or implied volatilities, so they do not reach `pe.f_pe`.

# Validation

  - `dims in (1, 2)`.
  - If `pe.a_views` is a [`LinearConstraintEstimator`](@ref), `length(pe.sets.dict[pe.sets.xkey]) == size(X, 2)`.
  - If `pe.f_views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.tfkey)` and `length(pe.sets.dict[pe.sets.tfkey]) == size(F, 2)`, both through [`factor_universe`](@ref).

`pe.w` has no named check. When `pe.l` is set, a `pe.w` with a length other than `size(X, 2)` raises a bare `DimensionMismatch` from the product in [`equilibrium_mu`](@ref). When `pe.l` is `nothing`, no step reads `pe.w`.

# Returns

  - `pr::LowOrderPrior`: Result object with the reconstructed asset returns, and with the asset half of the augmented posterior as `mu` and `sigma`. It carries the observation weighting of the asset prior, its diagnostics, the cache of its opinion pool, and the regression result. Its factor block `fpr` holds the **factor half** of the same posterior. `chol` is `nothing` on both blocks.

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. When `pe.a_views` resolves names, check that the declared asset axis has the same length as the width of `X`.
 3. When `pe.f_views` resolves names, check the declared factor axis against the width of `F` with [`factor_universe`](@ref). Only the views that resolve names against an axis check that axis, and a pair of precomputed [`BlackLittermanViews`](@ref) needs no `sets`.
 4. Fit `pe.a_pe` on `X`. The result is `a_prior`.
 5. Find the Investable Mask and the reduced universe of the asset views with [`investable_views`](@ref). Refuse a precomputed asset view matrix over a universe with gaps, with [`assert_bl_precomputed_universe`](@ref). View `a_prior` at the mask with [`investable_prior`](@ref). From this step on, `N` is the reduced count of assets.
 6. Fit `pe.f_pe` on `F`, with `strict` and `cache_reg` alone. The result is `f_prior`.
 7. Reduce `X` at the mask with [`reduce_columns`](@ref). Regress it on `F` with [`factor_reconstruction`](@ref), under `pe.re` viewed to the same mask with [`coverage_regression`](@ref). The results are `rr` and the reconstructed returns `posterior_X`, both on the reduced asset axis.
 8. Make the asset views with [`bl_preroll`](@ref) at the default `:xkey`, over the reduced asset prior covariance. Make the factor views at `:tfkey`, over the factor prior covariance. Either half can lose all its rows. [`bl_view_block`](@ref) then gives that half no row, and the stack stays, because the other half still conditions the joint posterior. A *departure* can empty only the asset half, because a factor axis holds no asset name. When **both** halves are empty, no view is left, and step 12 takes the joint prior.
 9. Build ``\\boldsymbol{\\Sigma}_{aug}``. Its off-diagonal blocks are the cross-covariance ``\\mathbf{M}\\boldsymbol{\\Sigma}_f`` of the factor model and its transpose.
10. Stack ``\\mathbf{P}_{aug}`` block-diagonally, and stack ``\\boldsymbol{q}_{aug}`` and ``\\boldsymbol{\\Omega}_{aug}`` to match. The asset rows are above the factor rows.
11. Put the stacked prior mean on the total-return scale of the views. The result is `aug_prior_mu`. When `pe.l` is `nothing`, it stacks the wrapped means, which are on that scale already. When `pe.l` is set, it is the equilibrium mean of [`equilibrium_mu`](@ref), plus `pe.rf` from [`apply_rf`](@ref), plus `rr.b` on the asset half. The equilibrium mean is a risk premium, and it reads `pe.w` sliced to the reduced axis by [`investable_weights_view`](@ref).
12. Run the master equations with [`vanilla_posteriors`](@ref) over the augmented space. The result is the augmented posterior pair. When **neither** half has a view row left, [`bl_posteriors`](@ref) gives back the joint prior. The master equations with an empty ``\\mathbf{P}_{aug}`` add ``\\tau\\boldsymbol{\\Sigma}_{aug}``, and that makes the joint covariance wider with no view to cause it.
13. Process the augmented posterior covariance in place with [`matrix_processing!`](@ref), under `pe.mp` and `hcat(posterior_X, F)`.
14. Truncate the asset half from `1:N`. Add nothing to it. The intercept and the rate went into the prior mean at step 11, and the update is affine in that mean.
15. Let `n` be the smaller of the row counts of `a_prior.X` and `f_prior.X`. A Scenario Cap in one nested prior keeps only the last rows of its own returns. The two blocks of the result must describe the same observations, so the result keeps the last `n` rows. Cut the nested prior with more rows with [`scenario_tail`](@ref). It refuses to cut a prior that carries weights, and it gives the count of the moments in `ens`. With no cap, `n` is the count of all observations, and the step cuts nothing. The fit of the moments used every observation, so the cut changes no moment.
16. Truncate the factor half from `N+1:N+K`. Forward the factor block with [`forward_prior`](@ref), and drop `chol`. The factor half gets no intercept, because the intercept belongs to the regression and to the asset axis alone. It gets no rate, because the stack had the rate before the update. It gets no second processing pass, because it is a principal submatrix of the matrix that step 13 processed. Do not expand the block, because the reduction did not change the factor axis.
17. Announce the departures once with [`announce_bl_departures`](@ref).
18. Build the prior result directly. It takes `w`, its diagnostics and `pc` from `a_prior`. It takes the last `n` rows of the reconstruction as `X` and of the oriented `X` as `o_X`, both cut by [`scenario_window`](@ref). It writes each block on the asset axis back onto the full universe. [`expand_moment`](@ref) writes the moment pair, [`expand_columns`](@ref) writes the reconstruction, and [`expand_regression`](@ref) writes the regression.

# Related

  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
  - [`apply_rf`](@ref)
  - [`equilibrium_mu`](@ref)
"""
function prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               cache_reg::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
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
        factor_universe(pe.sets, pe.sets.tfkey, size(F, 2),
                        "AugmentedBlackLittermanPrior, whose `f_views` are written in factor names",
                        "F")
    end
    # Asset prior.
    a_prior = prior(pe.a_pe, X, nothing, pnl; strict = strict, cache_reg = cache_reg,
                    kwargs...)
    # The reduction, once, at this estimator's entry. The asset views are a dense linear form
    # over the asset axis and `0 * NaN` is `NaN`, so a departed asset poisons `a_omega`, the
    # whole stacked system, and both truncated posteriors — including under views naming only
    # live assets. See [`investable_views`](@ref). The mask comes from the *asset prior*, not
    # from `X`, because a column can be quoted throughout the window and still be
    # unestimable, and it is the prior's verdict the next layer will re-derive.
    imsk, vsets, ni = investable_views(a_prior, pe.sets)
    assert_bl_precomputed_universe(pe.sets, a_prior)
    vapr = investable_prior(imsk, a_prior)
    a_prior_mu, a_prior_sigma = vapr.mu, vapr.sigma
    # Factor prior.
    # `cache_reg` reaches the factor prior too, so a refit under new scenario weights reads
    # its cache from `fpr` (#1751). The other keywords describe the asset axis, so they do
    # not reach it, as in `FactorPrior`: an active mask of `X` on `F` raised a
    # `DimensionMismatch` (R17 of #1630).
    f_prior = prior(pe.f_pe, F; strict = strict, cache_reg = cache_reg)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # The regression of the reduced assets on the factors. Only the reconstruction is shared
    # with `FactorPrior`:
    # the asset moments here come out of the augmented system, not out of a lift. It runs on
    # the reduced returns, so `M` and `b` land on the same axis as `a_prior_sigma`.
    rr, posterior_X = factor_reconstruction(coverage_regression(pe.re, imsk),
                                            reduce_columns(X, imsk), F)
    (; b, M) = rr
    dt = eltype(posterior_X)
    T = size(X, 1)
    N = size(a_prior_sigma, 1)
    ledger = String[]
    # One sets, two axes: the asset views take the default `:xkey`, the factor views `:tfkey`.
    # Only the asset half can be emptied by a departure — a factor axis carries no asset
    # name, so `counterpart_axis_names` answers empty for `:tfkey` — and when it is, the half
    # contributes no row rather than collapsing the stack: the *joint* posterior is still
    # conditioned by the factor views the departure never touched. See [`bl_view_block`](@ref).
    a_result = bl_preroll(pe.a_views, vsets, pe.a_views_conf, a_prior_sigma, pe.tau, T, dt,
                          strict; ledger = ledger)
    P, Q, a_omega = bl_view_block(a_result, N, dt)
    f_result = bl_preroll(pe.f_views, vsets, pe.f_views_conf, f_prior_sigma, pe.tau, T, dt,
                          strict, :tfkey)
    f_P, f_Q, f_omega = bl_view_block(f_result, size(f_prior_sigma, 1), dt)
    # Neither half kept a row, which is the one case the stack has nothing to condition on.
    # It is read twice below -- once by the master equations and once by the announcement --
    # and stating it once keeps the two readings from drifting apart.
    viewless = isnothing(a_result) && isnothing(f_result)
    # Both halves resolve the same blending parameter from the same `pe.tau` and the same
    # `T`, so reading it off either result gives the same number — but either result can now
    # be `nothing`, and there is no half left to read it from when both are. `bl_preroll`
    # states the rule and this restates it, which is the one place in the family it is
    # written twice.
    tau = isnothing(pe.tau) ? inv(T) : pe.tau
    aug_prior_sigma = hcat(vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                           vcat(M * f_prior_sigma, f_prior_sigma))
    aug_P = hcat(vcat(P, zeros(size(f_P, 1), size(P, 2))),
                 vcat(zeros(size(P, 1), size(f_P, 2)), f_P))
    aug_Q = vcat(Q, f_Q)
    aug_omega = hcat(vcat(a_omega, zeros(size(f_omega, 1), size(a_omega, 1))),
                     vcat(zeros(size(a_omega, 1), size(f_omega, 1)), f_omega))
    # `pe.l` replaces the two priors' own means with one equilibrium mean implied by the asset
    # weights `pe.w`. The expression and its equal-weight fallback belong to
    # [`equilibrium_mu`](@ref).
    #
    # Both branches must leave the stack on the scale this estimator *returns*: a total
    # return, with the regression intercept in the asset half. That is the scale the view
    # returns in `aug_Q` are written on, so it is the scale the view residual
    # `aug_Q - aug_P * mu` must be formed on, and a level missing from the prior mean is a
    # level the views are blended against wrongly (ADR 0063, amended).
    #
    # The two wrapped means are on it already. Least squares with an intercept zeroes the
    # residual mean, so the mean of `X` *is* `M * f_prior_mu + b`: the asset half carries the
    # intercept, and both halves are total returns.
    #
    # The equilibrium mean is not. It is a bare risk premium, so this branch adds the two
    # levels it lacks: `pe.rf` through [`apply_rf`](@ref), and the asset-side intercept `b`.
    # The factor half takes the rate and no intercept, because the intercept is the
    # regression's and hence asset-only.
    #
    # `pe.w` is per-asset configuration written against the caller's full universe, so it is
    # sliced to the axis the stack now sits on. Without that it meets a narrower stack and
    # the product raises a bare `DimensionMismatch` that names no asset.
    aug_prior_mu = if !isnothing(pe.l)
        apply_rf(pe.rf,
                 equilibrium_mu(pe.l, vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                                investable_weights_view(imsk, pe.w))) .+
        vcat(b, zeros(dt, size(F, 2)))
    else
        vcat(a_prior_mu, f_prior_mu)
    end
    # An empty half contributes no row and the stack is still conditioned by the other one,
    # which is why each half takes an empty block rather than collapsing to its prior. Both
    # halves empty is the case that has nothing left: `aug_P` is then `0 × (N + K)`, and
    # running the master equations over it would add the estimation-error term
    # `tau * aug_prior_sigma` and widen the joint covariance on the strength of views that
    # are not there. The joint posterior of a stack with no view is the joint prior.
    aug_posterior_mu, aug_posterior_sigma = if viewless
        bl_posteriors(nothing, aug_prior_mu, aug_prior_sigma)
    else
        vanilla_posteriors(tau, aug_prior_mu, aug_prior_sigma, aug_omega, aug_P, aug_Q)
    end
    matrix_processing!(pe.mp, aug_posterior_sigma, hcat(posterior_X, F))
    # Nothing is added here. `aug_prior_mu` reached the update on the scale this estimator
    # returns, and the update is affine in it, so the asset half comes off that scale with
    # the intercept and the rate already in it. Truncation is the whole of the step.
    posterior_mu = aug_posterior_mu[1:N]
    posterior_sigma = aug_posterior_sigma[1:N, 1:N]
    # The augmented system is jointly posterior over `[assets; factors]`, so truncating it to
    # the asset half discards a factor half that *is* the posterior factor distribution. The
    # factor block reports that half rather than `f_prior`'s prior moments. It is truncated and
    # nothing else: no `b`, because the intercept is the regression's and hence asset-only, and
    # no rate, because the stack reached the update on the total-return scale and comes off it
    # on the same one. The factor half is therefore on the scale `f_prior` supplied. No second
    # `matrix_processing!` either — `aug_posterior_sigma` was processed as a whole above, and a
    # principal submatrix of the result is already processed.
    f_idx = (N + 1):length(aug_posterior_mu)
    # A Scenario Cap in either nested prior keeps the last rows of its own returns alone. The
    # asset and factor blocks describe the same observations, so the result keeps the rows
    # that both keep, and the nested prior that carries more is cut to them. The moments
    # above were fitted over every observation, and the cut changes none of them.
    n = min(size(a_prior.X, 1), size(f_prior.X, 1))
    a_kept = scenario_tail(a_prior, n, :a_pe)
    f_kept = scenario_tail(f_prior, n, :f_pe)
    # `chol` is the factor block's only drop: the posterior covariance supersedes the one
    # `f_prior.chol` factorises. The views do not touch the observation axis, so `f_prior`'s own
    # `w` and its diagnostics forward untouched (ADR 0046).
    fpr = forward_prior(f_kept; mu = aug_posterior_mu[f_idx],
                        sigma = aug_posterior_sigma[f_idx, f_idx], chol = nothing)
    # The cache of the opinion pool comes from `a_prior` only: `f_prior` is fit on the
    # factors, so its cache holds factor returns and does not describe this asset axis.
    #
    # The factor block's `w` and diagnostics are `f_prior`'s, which is how the two priors'
    # weightings stay distinguishable — `w` is `a_prior`'s, `fpr.w` is `f_prior`'s. The asset
    # slot takes `a_prior`'s because a `@pprop w` consumer applies `w` to a return series over
    # the *assets*; two priors disagreeing about observation weights is a legitimate
    # configuration, and there are two slots to hold them. The diagnostics follow their
    # weights (ADR 0046), so `ens`/`kld`/`ow` come from `a_prior` too. `chol` is dropped:
    # `posterior_sigma` supersedes the covariance `a_prior.chol` factorises. This site merges
    # two priors rather than forwarding one along its own axis, so it builds the prior
    # result directly instead of going through [`forward_prior`](@ref).
    announce_bl_departures(ni, ledger, viewless)
    # The expansion, onto the caller's own universe. The moment pair goes back through
    # [`expand_moment`](@ref), the reconstruction through [`expand_columns`](@ref) and the
    # regression through [`expand_regression`](@ref), so a prior result again lives on the
    # FULL asset axis with a `NaN` in `mu` and on the diagonal of `sigma` for an asset that
    # is not investable, and the next layer derives the same mask this one did. `o_X` is the
    # caller's own `X`, cut to the last `n` rows and never reduced. There is no `chol` here to drop: the augmented
    # system produces the asset moments whole and this prior result never held one.
    return LowOrderPrior(; X = expand_columns(scenario_window(n, posterior_X), imsk),
                         o_X = scenario_window(n, X),
                         mu = expand_moment(posterior_mu, imsk, 1),
                         sigma = expand_moment(posterior_sigma, imsk), w = a_kept.w,
                         ens = a_kept.ens, kld = a_kept.kld, ow = a_kept.ow,
                         rr = expand_regression(rr, imsk), fpr = fpr,
                         pc = opinion_pooling_cache(a_kept))
end

function factor_residual_config(::AugmentedBlackLittermanPrior)
    # The augmented system produces the asset moments whole, so this estimator never calls
    # [`factor_lift`](@ref) and adds no residual block. See
    # [`factor_residual_config`](@ref).
    return nothing
end

export AugmentedBlackLittermanPrior
