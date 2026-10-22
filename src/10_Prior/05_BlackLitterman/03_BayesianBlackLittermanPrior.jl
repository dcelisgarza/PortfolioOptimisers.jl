"""
$(DocStringExtensions.TYPEDEF)

Bayesian Black-Litterman prior estimator for asset returns.

`BayesianBlackLittermanPrior` is a low order prior estimator. It applies views to the factors of a factor prior, and it updates the factor distribution by Bayes' rule. The asset moments are the posterior predictive distribution that the factor model gives.

The model and the posterior equations follow [cajas2025](@citet), who follows [kolmritter2016](@citet). [cajas2025](@citet) writes the view error covariance as ``\\tau\\mathbf{\\Sigma}_f``. The library uses ``\\tau\\,\\mathrm{Diag}(\\mathbf{P}\\mathbf{\\Sigma}_f\\mathbf{P}^\\intercal)``, which has one row per view. [cajas2025](@citet) has no regression intercept. The library adds the intercept ``\\boldsymbol{b}`` to the posterior asset mean.

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
        tau::Option{<:Number} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> BayesianBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

The estimator applies the views to the **factors**, and the views reach the assets through the regression loadings. So the estimator gives a posterior for both blocks. It forwards the wrapped prior whole, with these changes:

  - `mu` and `sigma` are the asset posterior. The estimator **drops** `chol`, because the posterior covariance replaces the covariance that `chol` factorises.
  - The factor block `fpr` holds the **posterior** factor moments, `mu_hat` and the inverse of the posterior precision, processed by `f_mp`. The estimator drops its `chol` for the same reason. Its `w` and the diagnostics of that weighting forward with no change, because the views do not change the observation axis.
  - All other fields forward. `X` is the `X` of the wrapped prior with no change, so `w`, `ens`, `kld` and `ow` still describe the axis of their fit. `rr` is a regression over data that the views do not change.

Both blocks are posterior, so the returned prior result is **internally consistent**: `mu == rr.M * fpr.mu + rr.b + rf` holds. At the default `rf = 0.0`, this is the plain identity. On a `250 × 5` sample over three factors at `rf = 0.03`, the largest difference between the two sides is `1.4e-17`. [`FactorBlackLittermanPrior`](@ref) satisfies the identity too, for the same reason. The other two members do not. See the warnings on [`BlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref).

!!! warning

    The returned `mu` and `sigma` are the Black-Litterman posterior, but `w` is the observation weighting of the **wrapped prior**, forwarded with no change. Black-Litterman gives no posterior over the observations, so no weighting that agrees with the posterior exists. If the estimator drops `w`, the unweighted empirical distribution takes its place, and that is further from the weights that the caller computed. So `pr.w`, `pr.ens`, `pr.kld` and `pr.ow` describe the prior, not the posterior.

## What this estimator refuses

The update inverts the factor covariance of the wrapped prior twice, so it needs a factor axis of full rank. A factor model that states a re-based Factor Family through [`has_family_rebasis`](@ref) carries `fpr` on the **raw** axis. The re-basis makes that axis a linear image of a smaller axis, so the covariance on it is singular. The estimator refuses such a prior result with an `ArgumentError` that names the wrapped estimator.

The inversion **raises no error** on such a matrix. It returns entries of the order of `1e18`, and the update continues to a posterior with the same scale as the prior. A caller who reads the result sees no sign that it is meaningless. For this reason, the refusal reads what the result *states* and does not test its rank, and it raises an error, not a warning.

[`HighOrderFactorPriorEstimator`](@ref) accepts the same prior result, because it projects through `rr.M` and does not invert the factor covariance.

## The views are written on the factor axis

`views` resolves against `sets.dict[sets.tfkey]`, the factor axis that [`UniverseSets`](@ref) declares. The Bayesian update changes the factor distribution, and the assets get the update through the loadings. The asset axis is still necessary, because every `UniverseSets` carries one, and [`port_opt_view`](@ref) slices it. The factor entries come back with no change, so this field is `@vprop` and needs no exemption.

`sets.dict[sets.tfkey]` must name the columns of `F` **in order**. [`factor_universe`](@ref) checks it. When the axis is missing or has the wrong length, the error names the factor axis, not the asset axis.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, [`assert_bl_views_conf`](@ref) checks `views_conf`.
  - If `tau` is not `nothing`, `tau > 0`.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pe` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

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
             │         pe ┼ EmpiricalPrior
             │            │           ce ┼ PortfolioOptimisersCovariance
             │            │              │   ce ┼ Covariance
             │            │              │      │    me ┼ SimpleExpectedReturns
             │            │              │      │       │   w ┴ nothing
             │            │              │      │    ce ┼ GeneralCovariance
             │            │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │            │              │      │       │    w ┴ nothing
             │            │              │      │   alg ┼ FullMoment()
             │            │              │      │     w ┴ nothing
             │            │              │   mp ┼ MatrixProcessing
             │            │              │      │     pdm ┼ Posdef
             │            │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │            │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │            │              │      │      dn ┼ nothing
             │            │              │      │      dt ┼ nothing
             │            │              │      │     alg ┼ nothing
             │            │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │            │           me ┼ EquilibriumExpectedReturns
             │            │              │   ce ┼ PortfolioOptimisersCovariance
             │            │              │      │   ce ┼ Covariance
             │            │              │      │      │    me ┼ SimpleExpectedReturns
             │            │              │      │      │       │   w ┴ nothing
             │            │              │      │      │    ce ┼ GeneralCovariance
             │            │              │      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │            │              │      │      │       │    w ┴ nothing
             │            │              │      │      │   alg ┼ FullMoment()
             │            │              │      │      │     w ┴ nothing
             │            │              │      │   mp ┼ MatrixProcessing
             │            │              │      │      │     pdm ┼ Posdef
             │            │              │      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │            │              │      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │            │              │      │      │      dn ┼ nothing
             │            │              │      │      │      dt ┼ nothing
             │            │              │      │      │     alg ┼ nothing
             │            │              │      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │            │              │    w ┼ nothing
             │            │              │    l ┴ Int64: 1
             │            │      horizon ┼ nothing
             │            │   fill_limit ┼ nothing
             │            │        x_src ┴ nothing
             │         mp ┼ MatrixProcessing
             │            │     pdm ┼ Posdef
             │            │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │            │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │            │      dn ┼ nothing
             │            │      dt ┼ nothing
             │            │     alg ┼ nothing
             │            │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │         re ┼ StepwiseRegression
             │            │       crit ┼ PValue
             │            │            │   t ┴ Float64: 0.05
             │            │        alg ┼ ForwardSelection()
             │            │        tgt ┼ LinearModel
             │            │            │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │            │     choice ┼ BatchChoice()
             │            │   included ┼ nothing
             │            │      x_src ┴ nothing
             │         ve ┼ SimpleVariance
             │            │          me ┼ SimpleExpectedReturns
             │            │             │   w ┴ nothing
             │            │           w ┼ nothing
             │            │   corrected ┴ Bool: true
             │        rsd ┼ Bool: true
             │   mtx_sqrt ┼ EigenFallbackSquareRoot()
             │      x_src ┴ nothing
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
             │     xkey ┼ String: "nx"
             │    uxkey ┼ String: "ux"
             │    tfkey ┼ String: "nf"
             │   utfkey ┼ String: "uf"
             │    cfkey ┼ String: "ncf"
             │   ucfkey ┼ String: "ucf"
             │    nikey ┼ String: "ni"
             │     dict ┴ Dict{String, Vector{String}}: Dict("nf" => ["F1", "F2"], "nx" => ["A", "B", "C"])
  views_conf ┼ nothing
          rf ┼ Float64: 0.0
         tau ┼ nothing
       x_src ┴ nothing
```

## The incremental fit

This prior has no exact incremental recursion, and it takes the online step by a **refit from a sample buffer**. [`Online`](@ref) seeds `cache`, and [`partial_fit!`](@ref) adds each observation to it with no change. The [`prior`](@ref) method with one argument then runs the batch verb of this estimator over the rows that the buffer kept. The result is the batch fit over those rows, and a `max_history` on the wrapper sets the window of the whole fit.

`cache` goes through the three propagation channels as every partial-fit state does. [`factory`](@ref) carries it with no change, and [`port_opt_view`](@ref) cuts it to the selected assets. [`obs_weights_view`](@ref) drops it, because a state has no cut on the observation axis. The show method does not print `cache`, because a buffer that changes at each step is not part of the configuration.

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
  - $(ref_dict[:cajas2025]) Section 5.3, Equations 5.23, 5.34 and 5.35. $(ref_dict[:no_original_source])
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
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function BayesianBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_F_AF,
                                         f_mp::AbstractMatrixProcessingEstimator,
                                         mp::AbstractMatrixProcessingEstimator,
                                         views::Lc_BLV, sets::Option{<:UniverseSets},
                                         views_conf::Option{<:Num_VecNum}, rf::Number,
                                         tau::Option{<:Number},
                                         cache::Option{<:AbstractPartialFitState},
                                         x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :BayesianBlackLittermanPrior)
        assert_inner_reader(pe, :pe, :BayesianBlackLittermanPrior)
        assert_bl(views, sets, views_conf, tau)
        return new{typeof(pe), typeof(f_mp), typeof(mp), typeof(views), typeof(sets),
                   typeof(views_conf), typeof(rf), typeof(tau), typeof(cache),
                   typeof(x_src)}(pe, f_mp, mp, views, sets, views_conf, rf, tau, cache,
                                  x_src)
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
                                     rf::Number = 0.0, tau::Option{<:Number} = nothing,
                                     cache::Option{<:AbstractPartialFitState} = nothing,
                                     x_src::Option{<:AbstractReturnsSource} = nothing)::BayesianBlackLittermanPrior
    return BayesianBlackLittermanPrior(pe, f_mp, mp, views, sets, views_conf, rf, tau,
                                       cache, x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give every field of a [`BayesianBlackLittermanPrior`](@ref) to print, except `cache`.

The state in `cache` changes at each step of an incremental fit, and it is not part of the configuration. Each site that prints the estimator also prints this state, unless this method leaves it out. Set `set_show_nothing_fields!(:BayesianBlackLittermanPrior, true)` to print it.

# Arguments

  - `::BayesianBlackLittermanPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pe, :f_mp, :mp, :views, :sets, :views_conf, :rf, :tau, :x_src)`.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::BayesianBlackLittermanPrior)
    return (:pe, :f_mp, :mp, :views, :sets, :views_conf, :rf, :tau, :x_src)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties BayesianBlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
    prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute Bayesian Black-Litterman prior moments for asset returns.

`prior` fits the wrapped factor prior, and it updates the factor distribution with the factor views by Bayes' rule. The asset moments are the posterior predictive distribution of the factor model.

When `pe.tau` is `nothing`, the blending parameter is `1/T`, where `T` is the number of observations of the oriented `F`. `pe.rf` reaches the result once, on the posterior asset expected returns. [`apply_rf`](@ref) owns that rule, and the factor block never carries the rate. This is the one Black-Litterman member that does not call [`vanilla_posteriors`](@ref). Its update is the conditional posterior of the factor parameter. The asset moments are a posterior predictive distribution, not a second run of the master equations.

# Mathematical definition

This is **not** the classic Black-Litterman update on the assets. The views apply to the factor parameter ``\\boldsymbol{\\theta}``, and the asset moments are the posterior *predictive* distribution of the factor model. The model is:

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

With ``\\mathbf{V} = \\left(\\mathbf{H} + \\mathbf{M}^\\intercal \\mathbf{\\Sigma}^{-1} \\mathbf{M}\\right)^{-1}``, the posterior predictive asset moments are:

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{BBL} &= \\left(\\mathbf{\\Sigma}^{-1} - \\mathbf{\\Sigma}^{-1}\\mathbf{M}\\,\\mathbf{V}\\,\\mathbf{M}^\\intercal \\mathbf{\\Sigma}^{-1}\\right)^{-1}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_{BBL} &= \\hat{\\mathbf{\\Sigma}}_{BBL}\\,\\mathbf{\\Sigma}^{-1}\\mathbf{M}\\,\\mathbf{V}\\,\\mathbf{H}\\,\\bar{\\boldsymbol{\\Pi}}_f + \\boldsymbol{b} + r_{f}\\,.
\\end{align}
```

Where:

  - ``N``, ``K``, ``K_v``, ``T``: The number of assets, of factors, of views, and of observations.
  - ``\\boldsymbol{r}``: ``N \\times 1`` asset return vector of the model.
  - ``\\boldsymbol{\\theta}``: ``K \\times 1`` factor parameter that the views apply to.
  - ``\\hat{\\boldsymbol{\\mu}}_{BBL}``: ``N \\times 1`` Bayesian Black-Litterman posterior asset mean, `pr.mu`.
  - ``\\hat{\\mathbf{\\Sigma}}_{BBL}``: ``N \\times N`` Bayesian Black-Litterman posterior asset covariance, `pr.sigma`.
  - ``\\boldsymbol{\\Pi}_f``, ``\\mathbf{\\Sigma}_f``: ``K \\times 1`` and ``K \\times K`` prior factor moments, from `pe.pe`.
  - ``\\mathbf{H}``: ``K \\times K`` posterior factor precision, the sum of the prior precision and the view precision.
  - ``\\bar{\\boldsymbol{\\Pi}}_f``, ``\\bar{\\mathbf{\\Sigma}}_f``: ``K \\times 1`` and ``K \\times K`` posterior factor moments, reported in `pr.fpr`.
  - ``\\mathbf{\\Sigma}``: ``N \\times N`` prior asset covariance matrix, from `pe.pe`.
  - ``\\mathbf{V}``: ``K \\times K`` inverse of the sum of ``\\mathbf{H}`` and the factor precision ``\\mathbf{M}^\\intercal \\mathbf{\\Sigma}^{-1} \\mathbf{M}`` of the asset model.
  - ``\\mathbf{M}``: ``N \\times K`` factor loadings matrix, `pr.rr.M`.
  - ``\\boldsymbol{b}``: ``N \\times 1`` regression intercept vector, `pr.rr.b`.
  - ``\\mathbf{P}``: ``K_v \\times K`` views matrix, over the **factor** axis.
  - ``\\boldsymbol{q}``: ``K_v \\times 1`` views vector.
  - ``\\mathbf{\\Omega}``: ``K_v \\times K_v`` view uncertainty matrix, ``\\mathrm{Diag}(\\mathbf{P}(\\tau\\mathbf{\\Sigma}_f)\\mathbf{P}^\\intercal)`` from [`calc_omega`](@ref) and [`bl_preroll`](@ref).
  - ``\\tau``: Scaling parameter, `1/T` by default.
  - ``r_{f}``: Risk-free rate, added once by [`apply_rf`](@ref).

Two facts change what a caller can do. First, ``\\mathbf{P}`` is over the factor axis, so it has ``K`` columns and not ``N``. The quantities of this estimator cannot give the classic master equation on the asset axis. Second, ``\\hat{\\boldsymbol{\\mu}}_{BBL}`` equals ``\\mathbf{M}\\bar{\\boldsymbol{\\Pi}}_f + \\boldsymbol{b} + r_{f}``. This is the identity of the *Composition* section above.

The estimator checks the width of ``\\mathbf{P}``. A precomputed [`BlackLittermanViews`](@ref) with a `P` of five columns, against a prior of three factors, raises a `DimensionMismatch` from [`bl_preroll`](@ref). The error gives `size(P, 2) => 5` and `size(prior_sigma, 1) => 3`. Views in asset names raise an `IsNothingError`, because no name resolves against the factor universe.

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. When `pe.views` resolves names, check the declared factor axis against the width of `F` with [`factor_universe`](@ref), through [`assert_bbl_views_axis`](@ref). A precomputed [`BlackLittermanViews`](@ref) resolves no name, so step 6 checks its width instead.
 3. Fit the wrapped prior `pe.pe` on `(X, F)`. The result is `prior_result`. [`bbl_posterior`](@ref) does the rest of this step and steps 4 to 16. So the refit of [`scenario_prior`](@ref) under new scenario weights uses the same code to blend the views into the refit `prior_result`. Check that `prior_result` carries a regression with [`assert_prior_regression`](@ref).
 4. Find the Investable Mask with [`investable_mask`](@ref), and view the fitted prior at the mask with [`investable_prior`](@ref). This step uses the mask alone, not [`investable_views`](@ref). The views apply to the factors, so this member needs no Non-Investable Axis and reads no asset universe.
 5. Read `posterior_X`, `prior_sigma`, `fpr` and `rr` from the *reduced* prior. Refuse an `rr` that states a re-based Factor Family through [`has_family_rebasis`](@ref).
 6. Make the views and their uncertainty with [`bl_preroll`](@ref), over the **factor** prior covariance and `size(F, 1)` observations. Read `P`, `Q` and `omega` from the result with [`bl_view_block`](@ref). The axis is `:tfkey`, because these views apply to the factors. A view set with no row left gives a `0 × K` block. The steps below then give the moments of the factor model of this member, and these are *not* the moments of the wrapped prior. See the note under `bl_view_block`.
 7. Build the posterior factor precision ``\\mathbf{H}`` as `sigma_hat`.
 8. Solve `sigma_hat` against the sum of the two precision-weighted means. The result is `mu_hat`, the posterior factor mean ``\\bar{\\boldsymbol{\\Pi}}_f``.
 9. Build the posterior asset covariance `posterior_sigma` from ``\\mathbf{H}``, the loadings and `prior_sigma`.
10. Process `posterior_sigma` in place with [`matrix_processing!`](@ref), under `pe.mp` and `posterior_X`.
11. Build the posterior asset mean from the same quantities, add `rr.b`, and add `pe.rf` with [`apply_rf`](@ref). This is the one site that adds the rate.
12. Invert `sigma_hat` for the posterior factor covariance ``\\bar{\\mathbf{\\Sigma}}_f``, and process it in place under `pe.f_mp` and `F`.
13. Forward the factor block with [`forward_prior`](@ref). Replace `mu` and `sigma` with the posterior factor pair, and drop `chol`. Do not expand the block, because the reduction did not change the factor axis.
14. Announce the departures once with [`announce_bl_departures`](@ref). [`investable_universe_names`](@ref) gives their names.
15. Write both asset posteriors back onto the full asset universe with [`expand_moment`](@ref). A non-investable asset then carries `NaN` in `mu` and on the diagonal of `sigma`.
16. Forward all of `prior_result` with [`forward_prior`](@ref). Replace `mu` and `sigma` with the expanded asset pair, drop `chol`, and replace `fpr` with the block of step 13.

# Arguments

  - `pe`: Bayesian Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:pnl_prior])
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: More keyword arguments, for the wrapped prior and for matrix processing.

# Validation

  - `dims in (1, 2)`.
  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.tfkey)` and `length(pe.sets.dict[pe.sets.tfkey]) == size(F, 2)`, both through [`factor_universe`](@ref).
  - The prior result of `pe.pe` must carry a regression result, through [`assert_prior_regression`](@ref).
  - That regression result must state no re-based Factor Family, through [`has_family_rebasis`](@ref). A re-basis makes `fpr.sigma` singular, and steps 7 and 8 invert it.

# Returns

  - `pr::LowOrderPrior`: Result object with the asset returns, the posterior asset mean vector and the posterior asset covariance matrix. Its factor block `fpr` holds the **posterior** factor moments. Both blocks are posterior, so `pr.mu == pr.rr.M * pr.fpr.mu + pr.rr.b + pe.rf` holds. `chol` is `nothing` on both blocks.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`bl_preroll`](@ref): Assembles `P`, `Q` and `omega` at `pe.sets.tfkey`, and resolves `pe.tau` to `1/T` when the estimator carries none.
  - [`calc_omega`](@ref)
  - [`apply_rf`](@ref)
  - [`forward_prior`](@ref)
  - [`vanilla_posteriors`](@ref): The master equations that this estimator does **not** use. The three other members use them.
"""
function prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
    X, F = dims_oriented(dims, X, F)
    assert_bbl_views_axis(pe, F)
    prior_result = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
    return bbl_posterior(pe, prior_result, F, strict; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a [`BayesianBlackLittermanPrior`](@ref) whose factor universe does not describe the factor returns.

The fit and the refit of [`scenario_prior`](@ref) both run this check before they fit the wrapped prior.

# Arguments

  - `pe`: Bayesian Black-Litterman prior estimator.
  - `F`: Factor returns, `observations × factors`.

# Validation

  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.tfkey)` and `length(pe.sets.dict[pe.sets.tfkey]) == size(F, 2)`, both through [`factor_universe`](@ref).

# Returns

  - `nothing`.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`assert_bl_views_axis`](@ref)
"""
function assert_bbl_views_axis(pe::BayesianBlackLittermanPrior, F::MatNum)
    # The views update the *factor* distribution — the assets are its projection through the
    # loadings — so they resolve against the declared factor axis, not against `xkey`. Only the
    # views that resolve *names* need a universe: a `BlackLittermanViews` result carries its own
    # `P`, so demanding one for it would reject the legitimate precomputed-views configuration,
    # which `assert_bl` deliberately permits to supply no `sets` at all.
    if isa(pe.views, LinearConstraintEstimator)
        factor_universe(pe.sets, pe.sets.tfkey, size(F, 2),
                        "BayesianBlackLittermanPrior, whose views are written in factor names",
                        "F")
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Blends the factor views of a [`BayesianBlackLittermanPrior`](@ref) into the result of its wrapped prior.

This function holds all the work of the estimator after the fit of the wrapped prior. That work is the rest of step 3 and steps 4 to 16 of the returns-matrix method of [`prior`](@ref). The batch fit calls it with the result of a fit over the matrices of the caller. The refit of [`scenario_prior`](@ref) calls it with the result of a refit of the wrapped prior under new scenario weights. The function does not read the asset returns, because the result carries them. It reads the factor returns for the observation count of the views and for the processing of the factor covariance.

# Arguments

  - `pe`: Bayesian Black-Litterman prior estimator.
  - `prior_result`: The result of the wrapped prior `pe.pe`.
  - `F`: Factor returns, `observations × factors`.
  - `strict`: If `true`, enforce strict validation of views and sets.
  - `kwargs...`: More keyword arguments, for matrix processing.

# Validation

  - The rules of the returns-matrix method of [`prior`](@ref), from step 3 on.

# Returns

  - `pr::LowOrderPrior`: The result that the returns-matrix method of [`prior`](@ref) states.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`bl_posterior`](@ref)
  - [`scenario_prior`](@ref)
"""
function bbl_posterior(pe::BayesianBlackLittermanPrior, prior_result::AbstractPriorResult,
                       F::MatNum, strict::Bool; kwargs...)
    assert_prior_regression(prior_result, :pe)
    # The reduction, once, at this estimator's entry. The views land on the *factors*, so the
    # view axis is untouched by a departure and no view row can be dropped for one — which is
    # why this member reduces with the mask alone rather than through [`investable_views`](@ref):
    # it has no Non-Investable Axis to mint, and demanding an asset universe it otherwise
    # never reads would refuse a legitimate factor-only `sets`.
    #
    # What a departure does poison is the asset side, and it poisons all of it: `prior_sigma`
    # is inverted twice below, so one `NaN` column reaches every entry of both posteriors and
    # the fit fails inside LAPACK, naming neither the asset that left nor why it mattered.
    imsk = investable_mask(prior_result)
    vpr = investable_prior(imsk, prior_result)
    posterior_X, prior_sigma, fpr, rr = vpr.X, vpr.sigma, vpr.fpr, vpr.rr
    # The views land on the factors, so the update below inverts `f_sigma` twice. A factor
    # model fitted in a re-based Factor Family states a raw factor axis that is a linear
    # image of a smaller one, so a covariance on that axis is singular by construction.
    #
    # The inversion RAISES NOTHING on such a matrix. Measured on the fixture of
    # `test/test_12i_cross_sectional_factor_rr_slot.jl`: the solve returns entries of order
    # `1e18` and the update carries on to a posterior whose scale looks like the prior's, so
    # a caller reading the result sees no sign that it is meaningless. That is what the
    # refusal is for, and it is why the refusal reads the result's own statement rather than
    # testing a rank -- a tolerance this matrix passes cannot separate the two cases.
    @argcheck(!has_family_rebasis(rr),
              ArgumentError("`pe` returned a prior whose factor model was fitted in a re-based Factor Family, so `pr.fpr.sigma` sits on the raw factor axis of `pr.rr.M` and that axis is a linear image of a smaller one. $(nameof(BayesianBlackLittermanPrior)) applies its views to the factor distribution and inverts that covariance, which is singular by construction, so the update has no answer.\nApply the views on the asset axis with `$(nameof(BlackLittermanPrior))`, which reads no factor covariance, or wrap a prior whose factor model re-bases no family.\nGot\npe => $(nameof(typeof(pe.pe)))\nrr => $(nameof(typeof(rr)))"))
    f_mu, f_sigma = fpr.mu, fpr.sigma
    # `pe.sets` goes through unreduced and unminted: the views resolve against `tfkey`, and
    # the reduction took asset columns, which that universe does not describe. No ledger
    # either — nothing on this path can drop a view row.
    blp = bl_preroll(pe.views, pe.sets, pe.views_conf, f_sigma, pe.tau, size(F, 1),
                     eltype(posterior_X), strict, :tfkey)
    # A view set with no row left takes an empty block rather than the prior pair, and this
    # is the one member where that is the *right* answer rather than a stacking convenience.
    # The update below is a precision sum — `sigma_hat = inv(f_sigma) + P'Ω⁻¹P` — so a
    # `0 × K` `P` adds exactly zero and there is no `tau * sigma` estimation-error term for
    # it to inflate, which is what rules the empty block out for the members that run
    # `vanilla_posteriors`. What comes out is `f_sigma` and `f_mu` unchanged on the factor
    # side, and by Woodbury `sigma_a + M * f_sigma * M'` with `M * f_mu + b` on the asset
    # side: the moments this member's factor model implies. That is NOT its wrapped prior,
    # and it cannot be — this member transforms the prior it wraps whether or not a view is
    # stated, so there is no unadjusted answer for it to fall back to. See
    # [`bl_view_block`](@ref).
    P, Q, omega = bl_view_block(blp, size(f_sigma, 1), eltype(posterior_X))
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
    # inverse. Reporting them rather than the prior ones is what makes this prior result
    # internally consistent: `mu == rr.M * fpr.mu + rr.b` holds exactly afterwards, where
    # forwarding the prior block left the asset and factor halves describing different
    # distributions.
    # `pe.f_mp` processes the factor block for the same reason `pe.mp` processes the asset one,
    # and is separate for the same reason `FactorBlackLittermanPrior` keeps the two apart.
    f_posterior_sigma = sigma_hat \ LinearAlgebra.I
    matrix_processing!(pe.f_mp, f_posterior_sigma, F; kwargs...)
    # `chol` is the factor block's only drop — `f_posterior_sigma` supersedes the covariance it
    # factorises. The views do not touch the observation axis, so the factor prior's `w` and
    # that weighting's diagnostics forward untouched (ADR 0046).
    posterior_fpr = forward_prior(fpr; mu = mu_hat, sigma = f_posterior_sigma,
                                  chol = nothing)
    announce_bl_departures(investable_universe_names(pe.sets, imsk), String[],
                           isnothing(blp))
    # The expansion, onto the caller's own universe: a prior result lives on the FULL asset
    # axis, with a `NaN` in `mu` and on the diagonal of `sigma` for an asset that is not
    # investable, so that the next layer derives the same mask this one did. The factor block
    # is not expanded — the reduction never touched the factor axis.
    posterior_mu = expand_moment(posterior_mu, imsk, 1)
    posterior_sigma = expand_moment(posterior_sigma, imsk)
    # Everything else the wrapped prior carried is forwarded (see [`forward_prior`](@ref));
    # `chol` is the only drop, because `posterior_sigma` supersedes the covariance it
    # factorises. `posterior_X` is `prior_result.X` unchanged, so the wrapped `w` still
    # describes exactly the rows of the returned `X`, and its `ens`/`kld`/`ow` still describe
    # that `w`. `rr` is unchanged — the
    # regression is over data the views do not modify — so the factor block it projects is now
    # the posterior one.
    return forward_prior(prior_result; mu = posterior_mu, sigma = posterior_sigma,
                         chol = nothing, fpr = posterior_fpr)
end

function factor_residual_config(pe::BayesianBlackLittermanPrior)
    return factor_residual_config(pe.pe)
end

export BayesianBlackLittermanPrior
