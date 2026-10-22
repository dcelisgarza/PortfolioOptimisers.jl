"""
$(DocStringExtensions.TYPEDEF)

Factor Black-Litterman prior estimator for asset returns.

`FactorBlackLittermanPrior` is a low order prior estimator. It fits a prior on the factor returns, and it applies the Black-Litterman update to the factor moments. A regression of the assets on the factors then lifts the posterior factor moments onto the assets.

[black1991](@citet) state the update on the asset axis. The library applies the same update to the factor moments. It then lifts the posterior factor moments onto the assets with the factor model of [cajas2025](@citet). [cajas2025](@citet) states the update and the lift as two separate results, and it does not state their composition.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorBlackLittermanPrior(;
        pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        rsd::Bool = true,
        views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        w::Option{<:VecNum} = nothing,
        rf::Number = 0.0,
        l::Option{<:Number} = nothing,
        tau::Option{<:Number} = nothing,
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> FactorBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **lifts** a prior on the factor axis onto the asset axis, and it reconstructs `X` as `F * transpose(M) .+ transpose(b)`. It builds its prior result directly, and it does not forward a prior along its own axis. A posterior field still replaces the prior field of the same name, and a structural field still forwards whole. In this member of the Black-Litterman family, the views *change* the factor block. The views apply to the factor distribution, and the asset moments are its projection.

  - The factor block `fpr` is the **posterior** factor distribution, processed by `f_mp`. The estimator drops its `chol`, because the posterior covariance replaces the covariance that `chol` factorises. Its `w` and the diagnostics of that weighting forward with no change.
  - `mu` and `sigma` are that block projected through the loadings. So the returned prior result is **internally consistent**: `mu == rr.M * fpr.mu + rr.b` holds for all values of `rf`. When the rate reaches the result, it is inside `fpr.mu`. When `rsd` is `true`, `sigma` also gets a residual block.
  - `w` is the `w` of the factor prior, and it is on the correct axis. This estimator wraps only a factor prior, and `posterior_X` has the same rows as `F`. No other weighting exists.

[`BayesianBlackLittermanPrior`](@ref) also satisfies the identity exactly. [`BlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) do not. See their warnings.

## The views are written on the factor axis

`views` resolves against `sets.dict[sets.tfkey]`, the factor axis that [`UniverseSets`](@ref) declares, because the views update the factor distribution. The asset axis is still necessary, because every `UniverseSets` carries one, and [`port_opt_view`](@ref) slices it. The factor entries come back with no change, so this field is `@vprop` and needs no exemption.

`sets.dict[sets.tfkey]` must name the columns of `F` **in order**. [`factor_universe`](@ref) checks it. When the axis is missing or has the wrong length, the error names the factor axis, not the asset axis.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, [`assert_bl_views_conf`](@ref) checks `views_conf`.
  - If `tau` is not `nothing`, `tau > 0`.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pe`, `re` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).
  - When `rsd` is `true`, `mp` states the estimator of its repair, see [`assert_matrix_processing_posdef`](@ref).

The constructor does **not** check `w`. The length of `w` is a property of the returns matrix, and the constructor does not see that matrix. A wrong length raises a `DimensionMismatch` from [`equilibrium_mu`](@ref) in [`prior`](@ref), and only when `l` is set. The constructor also accepts an empty `w`, but the sibling [`AugmentedBlackLittermanPrior`](@ref) refuses one.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `re`: Recursively updated via [`factory`](@ref).
  - `ve`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `re`: Recursively viewed via [`port_opt_view`](@ref).
  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

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
             │       crit ┼ PValue
             │            │   t ┴ Float64: 0.05
             │        alg ┼ ForwardSelection()
             │        tgt ┼ LinearModel
             │            │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │     choice ┼ BatchChoice()
             │   included ┼ nothing
             │      x_src ┴ nothing
          ve ┼ SimpleVariance
             │          me ┼ SimpleExpectedReturns
             │             │   w ┴ nothing
             │           w ┼ nothing
             │   corrected ┴ Bool: true
         rsd ┼ Bool: true
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
           w ┼ nothing
          rf ┼ Float64: 0.0
           l ┼ nothing
         tau ┼ nothing
    mtx_sqrt ┼ EigenFallbackSquareRoot()
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

  - $(ref_dict[:black1991]) Appendix, items 7 and 8. $(ref_dict[:no_original_source])
  - $(ref_dict[:cajas2025]) Section 5.1.5, Equations 5.13 to 5.15, and Section 4.1, Equations 4.4 and 4.5.
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
    $(field_dict[:rsd])
    """
    rsd
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
    $(field_dict[:mtx_sqrt])
    """
    mtx_sqrt
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function FactorBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                                       f_mp::AbstractMatrixProcessingEstimator,
                                       mp::AbstractMatrixProcessingEstimator,
                                       re::AbstractTimeSeriesRegressionEstimator,
                                       ve::AbstractVarianceEstimator, rsd::Bool,
                                       views::Lc_BLV, sets::Option{<:UniverseSets},
                                       views_conf::Option{<:Num_VecNum},
                                       w::Option{<:VecNum}, rf::Number, l::Option{<:Number},
                                       tau::Option{<:Number},
                                       mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm},
                                       cache::Option{<:AbstractPartialFitState},
                                       x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :FactorBlackLittermanPrior)
        assert_inner_reader(pe, :pe, :FactorBlackLittermanPrior)
        assert_inner_reader(re, :re, :FactorBlackLittermanPrior)
        assert_bl(views, sets, views_conf, tau)
        if rsd
            assert_matrix_processing_posdef(mp)
        end
        return new{typeof(pe), typeof(f_mp), typeof(mp), typeof(re), typeof(ve),
                   typeof(rsd), typeof(views), typeof(sets), typeof(views_conf), typeof(w),
                   typeof(rf), typeof(l), typeof(tau), typeof(mtx_sqrt), typeof(cache),
                   typeof(x_src)}(pe, f_mp, mp, re, ve, rsd, views, sets, views_conf, w, rf,
                                  l, tau, mtx_sqrt, cache, x_src)
    end
end
function FactorBlackLittermanPrior(;
                                   pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                   f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
                                   ve::AbstractVarianceEstimator = SimpleVariance(),
                                   rsd::Bool = true, views::Lc_BLV,
                                   sets::Option{<:UniverseSets} = nothing,
                                   views_conf::Option{<:Num_VecNum} = nothing,
                                   w::Option{<:VecNum} = nothing, rf::Number = 0.0,
                                   l::Option{<:Number} = nothing,
                                   tau::Option{<:Number} = nothing,
                                   mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
                                   cache::Option{<:AbstractPartialFitState} = nothing,
                                   x_src::Option{<:AbstractReturnsSource} = nothing)::FactorBlackLittermanPrior
    return FactorBlackLittermanPrior(pe, f_mp, mp, re, ve, rsd, views, sets, views_conf, w,
                                     rf, l, tau, mtx_sqrt, cache, x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give every field of a [`FactorBlackLittermanPrior`](@ref) to print, except `cache`.

The state in `cache` changes at each step of an incremental fit, and it is not part of the configuration. Each site that prints the estimator also prints this state, unless this method leaves it out. Set `set_show_nothing_fields!(:FactorBlackLittermanPrior, true)` to print it.

# Arguments

  - `::FactorBlackLittermanPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pe, :f_mp, :mp, :re, :ve, :rsd, :views, :sets, :views_conf, :w, :rf, :l, :tau, :mtx_sqrt, :x_src)`.

# Related

  - [`FactorBlackLittermanPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::FactorBlackLittermanPrior)
    return (:pe, :f_mp, :mp, :re, :ve, :rsd, :views, :sets, :views_conf, :w, :rf, :l, :tau,
            :mtx_sqrt, :x_src)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties FactorBlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
    prior(pe::FactorBlackLittermanPrior, X::MatNum, F::MatNum,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1,
          strict::Bool = false,
          active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
          cache_reg::Bool = false, kwargs...)

Compute factor Black-Litterman prior moments for asset returns.

`prior` fits the wrapped prior on the factor returns, and it applies the views to the factor moments with the Black-Litterman master equations. It then lifts the posterior factor moments onto the assets.

When `pe.tau` is `nothing`, the blending parameter is `1/T`, where `T` is the number of observations of the oriented `X`. The update gets a **total** return over the factors as its mean. The view returns in `Q` have this scale, and the factor block has it too. `pe.rf` reaches the update only when `pe.l` is set, because only that branch has a prior mean that is a risk premium. The closed form below gives the shift that the rate causes there. When `pe.l` is `nothing`, no step reads the field, and it does not change the result.

# Mathematical definition

The Black-Litterman update applies the views to the factor moments ``(\\boldsymbol{\\Pi}_f, \\mathbf{\\Sigma}_f)`` with the master equations. The loadings matrix then gives the asset posteriors:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{M} \\hat{\\boldsymbol{\\mu}}_{f,BL} + \\boldsymbol{b}\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{M} \\hat{\\mathbf{\\Sigma}}_{f,BL} \\mathbf{M}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``N``, ``K``, ``K_v``, ``T``: The number of assets, of factors, of views, and of observations.
  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` posterior asset mean vector, `pr.mu`.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` posterior asset covariance matrix, `pr.sigma`.
  - ``\\boldsymbol{\\Pi}_f``, ``\\mathbf{\\Sigma}_f``: ``K \\times 1`` and ``K \\times K`` prior factor moments, from `pe.pe` fit on `F`.
  - ``\\hat{\\boldsymbol{\\mu}}_{f,BL}``: ``K \\times 1`` Black-Litterman posterior factor mean, `pr.fpr.mu`.
  - ``\\hat{\\mathbf{\\Sigma}}_{f,BL}``: ``K \\times K`` Black-Litterman posterior factor covariance, `pr.fpr.sigma`.
  - ``\\mathbf{M}``: ``N \\times K`` factor loadings matrix, `pr.rr.M`.
  - ``\\boldsymbol{b}``: ``N \\times 1`` regression intercept vector, `pr.rr.b`.
  - ``\\mathbf{P}``: ``K_v \\times K`` views matrix, over the **factor** axis.
  - ``\\mathbf{\\Omega}``: ``K_v \\times K_v`` view uncertainty matrix, from [`calc_omega`](@ref) and [`bl_preroll`](@ref).
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal residual variance matrix, zero when `rsd = false`.
  - ``\\tau``: Scaling parameter, `1/T` by default.
  - ``r_{f}``: Risk-free rate. [`apply_rf`](@ref) adds it once, to the equilibrium factor mean. So it is inside ``\\hat{\\boldsymbol{\\mu}}_{f,BL}`` when `pe.l` is set, and it is absent when `pe.l` is `nothing`.

The factor moments are the Black-Litterman posterior of [`vanilla_posteriors`](@ref) over the factor axis. ``\\mathbf{P}`` has ``K`` columns, and ``\\mathbf{\\Sigma}_f`` is the covariance of the factor prior. A direct call of [`vanilla_posteriors`](@ref) on the factor prior and on the `P`, `Q` and `omega` of [`bl_preroll`](@ref) gives `pr.fpr.mu` and `pr.fpr.sigma`. The two lifted forms above give `pr.mu` and `pr.sigma` for both values of `rsd`. The returned `chol` is a factor of the returned covariance, so the residual block is in both.

The rate reaches the result through the factors, and only on one branch. When `pe.l` is set, the equilibrium mean is a risk premium. [`apply_rf`](@ref) adds the rate to it before the update, and this gives a total factor return. The blend and the lift then carry all of that factor mean, with the rate in it. Let ``\\mathbf{G} = \\tau\\mathbf{\\Sigma}_f\\mathbf{P}^\\intercal(\\mathbf{P}\\tau\\mathbf{\\Sigma}_f\\mathbf{P}^\\intercal + \\mathbf{\\Omega})^{-1}`` be the update gain, and let ``\\mathbf{1}`` be the vector of ones. Against the same estimator at ``r_f = 0``, the result moves by:

```math
\\begin{align}
\\Delta\\hat{\\boldsymbol{\\mu}} &= r_{f}\\mathbf{M}\\left(\\mathbf{I} - \\mathbf{G}\\mathbf{P}\\right)\\mathbf{1}\\,.
\\end{align}
```

The shift is linear in ``r_f``, and it depends on the views through ``\\mathbf{G}``. On a `250 × 5` sample over three factors, with two factor views and `l = 2.0`, the shift per unit of ``r_f`` is `[0.489, 0.831, 0.868, 0.183, 0.973]`. It agrees with the closed form to `2.2e-16`, and the values at `rf = 0.03` and at `rf = 0.06` agree to `2.8e-16`. When `pe.l` is `nothing`, the prior mean is the mean of the factor prior, which is a total return. No step then reads `pe.rf`, and two fits that differ only in `rf` are equal in each entry.

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. When `pe.views` resolves names, check the declared factor axis against the width of `F` with [`factor_universe`](@ref). A precomputed [`BlackLittermanViews`](@ref) resolves no name, so step 6 checks its width instead.
 3. Reduce `X` to the assets that the fit can use, with [`coverage_reduction`](@ref) under `pnl`. The results are the mask and `Xi`. This member wraps a *factor* prior, so it has no prior result on the asset axis to give an Investable Mask. The reduction reads the gaps from the returns. When the caller gives `active_mask`, the reduction reads it in place of the active mask of `pnl`.
 4. Fit the wrapped prior `pe.pe` on `F` alone. The result is `f_prior`. Read `prior_mu` and `prior_sigma` from it. The type bound of `pe` is a bound for asset priors, but this step gives it the factor matrix. `strict` and `cache_reg` reach it, as in [`FactorPrior`](@ref). So the prior stores the cache that a refit under new scenario weights reads from `fpr`. No other keyword reaches it, because the other keywords describe the asset axis.
 5. Regress `Xi` on `F` with [`factor_reconstruction`](@ref), under `pe.re` viewed to `imsk` with [`coverage_regression`](@ref). The results are the regression result `rr` and the reconstructed returns `posterior_X`.
 6. Make the views and their uncertainty with [`bl_preroll`](@ref), over `prior_sigma` and `size(Xi, 1)` observations. The result is `blp`. The axis is `:tfkey`, so a departed asset drops no view row, and the step keeps no ledger.
 7. Put the prior mean on the total-return scale of the views. The result is `prior_total_mu`. When `pe.l` is set, it is the equilibrium mean of [`equilibrium_mu`](@ref) plus `pe.rf` from [`apply_rf`](@ref). The equilibrium mean is a risk premium, and it reads `pe.w` sliced to the reduced axis by [`investable_weights_view`](@ref). When `pe.l` is `nothing`, it is `prior_mu`, which is on that scale already.
 8. Run the master equations with [`bl_posteriors`](@ref). The result is the posterior factor pair. When no view row is left, it gives back `prior_total_mu` and the factor prior covariance. Step 10 lifts them as it lifts a posterior pair.
 9. Process the posterior factor covariance in place with [`matrix_processing!`](@ref), under `pe.f_mp` and `F`.
10. Lift the posterior factor pair onto the reduced assets with [`factor_lift`](@ref). The results are `mu`, `sigma`, `chol`, `esigma`, `edof` and `ediv`. [`FactorPrior`](@ref) applies the same lift, to different factor moments. The lift adds the residual block when `pe.rsd` is `true`. It takes the square root of the posterior factor covariance for `chol` under `pe.mtx_sqrt`, and it processes `sigma` under `pe.mp`.
11. Write `esigma`, `edof` and `ediv` onto `rr` with [`set_idiosyncratic_covariance`](@ref). Under `pe.rsd = true`, the fields hold the residual variances that the lift measured and their counts. Under `pe.rsd = false`, `esigma` and `ediv` hold `nothing`, because the lift added no residual block.
12. Forward the factor block with [`forward_prior`](@ref). Replace `mu` and `sigma` with the posterior factor pair, and drop `chol`. Do not expand the block, because the reduction did not change the factor axis.
13. Announce the departures once with [`announce_bl_departures`](@ref). [`investable_universe_names`](@ref) gives their names.
14. Write each block on the asset axis back onto the full universe. [`expand_moment`](@ref) writes the moment pair, [`expand_columns`](@ref) writes the reconstruction, and [`expand_regression`](@ref) writes the regression. When step 3 removed an asset, drop `chol` and do not expand it, because a frame with `NaN` entries has no factorisation.
15. Cut the reconstruction and the oriented `X` to the rows that `f_prior.X` carries, with [`scenario_window`](@ref). A Scenario Cap of `pe.pe` keeps the last rows of `F` alone, and the asset side keeps the same rows, as in [`FactorPrior`](@ref). With no cap, the cut keeps every row. The fit of the moments used every observation, so the cut changes no moment.
16. Build the prior result directly. It takes `w` and its diagnostics from `f_prior`, the reconstruction as `X`, and the oriented `X` as `o_X`. Under a cap, `f_prior.ens` gives the count of the observations that the moments read.

# Arguments

  - `pe`: Factor Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:pnl_prior]) The panel reaches the reduction of step 3 alone. The estimator fits its nested prior on the factors, and no panel describes the factor axis.
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - $(arg_dict[:cvg_active_mask])
  - $(arg_dict[:cache_reg])
  - `kwargs...`: More keyword arguments, for the processing of the asset covariance under `pe.mp` in [`factor_lift`](@ref).

# Validation

  - `dims in (1, 2)`.
  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.tfkey)` and `length(pe.sets.dict[pe.sets.tfkey]) == size(F, 2)`, both through [`factor_universe`](@ref).

`pe.w` has no named check. When `pe.l` is set, a `pe.w` with a length other than `size(X, 2)` raises a bare `DimensionMismatch` from the product in [`equilibrium_mu`](@ref). When `pe.l` is `nothing`, no step reads `pe.w`.

# Returns

  - `pr::LowOrderPrior`: Result object with the reconstructed asset returns, the posterior asset mean vector, the posterior asset covariance matrix and its Cholesky factor. It carries the observation weighting of the factor prior, its diagnostics, and the regression result. Its factor block `fpr` holds the **posterior** factor moments, on the total-return scale of the update. `fpr.chol` is `nothing`.

# Related

  - [`FactorBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
  - [`apply_rf`](@ref)
  - [`equilibrium_mu`](@ref)
"""
function prior(pe::FactorBlackLittermanPrior, X::MatNum, F::MatNum,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
               cache_reg::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    # The views land on the *factor* distribution, so they resolve against the declared factor
    # axis — not against `xkey`, which names the assets this estimator projects onto. Only the
    # views that resolve *names* need a universe: a `BlackLittermanViews` result carries its own
    # `P`, so demanding one for it would reject the legitimate precomputed-views configuration,
    # which `assert_bl` deliberately permits to supply no `sets` at all.
    if isa(pe.views, LinearConstraintEstimator)
        factor_universe(pe.sets, pe.sets.tfkey, size(F, 2),
                        "FactorBlackLittermanPrior, whose views are written in factor names",
                        "F")
    end
    # The reduction, once, at this estimator's entry. This member is the odd one of the four:
    # it wraps a *factor* prior, so there is no asset-side prior result to read an Investable
    # Mask off. The gap arrives in `X` itself, and `coverage_mask` is where the library reads
    # a gap out of a returns block — the same verb every mask-aware moment estimator uses.
    # `pnl` is threaded in rather than discarded, because an Asset Panel states listing and
    # delisting that the returns alone need not: a quoted but inactive row is a gap this
    # would otherwise miss, and reading it here is what makes the panel argument mean
    # something on this member.
    #
    # Unreduced, `X`'s `NaN` column reaches the regression, where `StepwiseRegression`
    # selects *zero* factors and raises `BoundsError … at index [1:T, [0]]` — an error that
    # names neither the asset that left nor the reason it mattered.
    imsk, Xi = coverage_reduction(X, pnl; dims = 1,
                                  active_mask = dims_oriented(dims, active_mask))
    # Factor prior. It stores its cache for a refit under new scenario weights, which reads
    # it from `fpr`, as in `FactorPrior`.
    f_prior = prior(pe.pe, F; strict = strict, cache_reg = cache_reg)
    prior_mu, prior_sigma = f_prior.mu, f_prior.sigma
    # The regression of the reduced assets on the factors.
    rr, posterior_X = factor_reconstruction(coverage_regression(pe.re, imsk), Xi, F)
    M = rr.M
    # `pe.sets` goes through unreduced and unminted, and with no ledger, for the reason
    # [`BayesianBlackLittermanPrior`](@ref) gives: the views resolve against `tfkey`, so
    # nothing on this path can drop a view row for a departed asset.
    blp = bl_preroll(pe.views, pe.sets, pe.views_conf, prior_sigma, pe.tau, size(Xi, 1),
                     eltype(posterior_X), strict, :tfkey)
    # `pe.l` replaces the factor prior's own mean with an equilibrium one implied by the asset
    # weights `pe.w`. The expression and its equal-weight fallback belong to
    # [`equilibrium_mu`](@ref).
    #
    # Both branches must leave a *total* return over the factors, because that is the scale
    # the view returns in `Q` are written on, so it is the scale the view residual
    # `Q - P * mu` must be formed on (ADR 0063, amended). The factor prior's own mean is one
    # already. The equilibrium mean is a bare risk premium, so that branch adds the rate
    # through [`apply_rf`](@ref). Both are factor means, so the rate is added on the factor
    # axis in either case, and the Factor Lift below carries the whole factor mean — rate
    # included — to the assets through the loadings.
    #
    # `pe.w` is per-asset configuration written against the caller's full universe, so it is
    # sliced to the same axis `M` now sits on. Without that it meets a narrower `M` and the
    # product raises a bare `DimensionMismatch` that names no asset.
    prior_total_mu = if !isnothing(pe.l)
        apply_rf(pe.rf,
                 equilibrium_mu(pe.l, prior_sigma * transpose(M),
                                investable_weights_view(imsk, pe.w)))
    else
        prior_mu
    end
    # A view set with no row left leaves the factor moments the ones this member was going
    # to update — `prior_total_mu` and the factor prior's covariance — and the Factor Lift
    # below carries them to the assets exactly as it carries a posterior pair. The empty-view
    # algebra is not the answer here: `vanilla_posteriors` would add `tau * prior_sigma` and
    # widen the factor covariance on the strength of views that are not there. See
    # [`bl_posteriors`](@ref).
    f_posterior_mu, f_posterior_sigma = bl_posteriors(blp, prior_total_mu, prior_sigma)
    matrix_processing!(pe.f_mp, f_posterior_sigma, F)
    # Reconstruct the posteriors using the black litterman adjusted factor statistics. The lift
    # is the same one `FactorPrior` applies; only the factor moments handed to it differ.
    (; mu, sigma, chol, esigma, edof, ediv) = factor_lift(pe.mp, pe.ve, pe.rsd, rr,
                                                          f_posterior_mu, f_posterior_sigma,
                                                          Xi, posterior_X;
                                                          mtx_sqrt = pe.mtx_sqrt, kwargs...)
    # The lift already measured the residual variances, so the block carries them instead of
    # making every consumer recompute them from the reconstruction error. Under `rsd = false`
    # the lift added no residual block and `esigma` is `nothing`, which is what the field then
    # holds. The counts of the variances travel beside them, for a consumer that prices their
    # sampling error.
    rr = set_idiosyncratic_covariance(rr, esigma, edof, ediv)
    # Nothing is added to `mu`. `f_posterior_mu` is a total return over the factors, so the
    # lift gives a total return over the assets, and `rr.b` is applied inside the lift once.
    #
    # No `pc` is forwarded: the only wrapped prior here is `f_prior`, fit on the factors, so
    # the cache of its opinion pool holds factor returns and does not describe the asset axis.
    #
    # This is the one site that *modifies* the factor block rather than passing it through:
    # the views land on the factor distribution. `chol` is dropped because `f_posterior_sigma`
    # supersedes the factor prior's covariance (see [`forward_prior`](@ref)); everything else
    # the factor prior carried — its `w` and that weighting's diagnostics — is forwarded.
    fpr = forward_prior(f_prior; mu = f_posterior_mu, sigma = f_posterior_sigma,
                        chol = nothing)
    announce_bl_departures(investable_universe_names(pe.sets, imsk), String[],
                           isnothing(blp))
    # The expansion, onto the caller's own universe. Everything the lift produced is on the
    # asset axis and every one of them goes back: the moment pair through
    # [`expand_moment`](@ref), the reconstruction through [`expand_columns`](@ref), and the
    # regression — its loadings, its intercept and its idiosyncratic block — through
    # [`expand_regression`](@ref). `o_X` is the caller's own `X`, cut to the rows that the
    # factor prior carries and never reduced.
    #
    # `chol` is the one thing that cannot be expanded and is dropped instead. It factorises
    # `sigma`, and a `NaN` frame has no factorisation, so writing one into a frame would
    # hand a consumer a triangular matrix that is not a factor of anything. The all-
    # investable path keeps it, because there is nothing to expand there and nothing to drop.
    mu = expand_moment(mu, imsk, 1)
    sigma = expand_moment(sigma, imsk)
    chol = isnothing(imsk) ? chol : nothing
    rr = expand_regression(rr, imsk)
    # A Scenario Cap of the factor prior keeps the last rows of `F` alone, so the asset side
    # keeps the same rows, as in `FactorPrior`: the two blocks describe the same
    # observations. The moments above were fitted over every observation, and the factor
    # prior's `ens` states that count.
    n = size(f_prior.X, 1)
    posterior_X = expand_columns(scenario_window(n, posterior_X), imsk)
    #
    # The asset-side `w` is the factor prior's: this estimator wraps only a factor prior, and
    # `posterior_X = F*M' + b'` has exactly `F`'s rows, so it is the only weighting in
    # existence and it is over the right observation axis. Its `ens`/`kld`/`ow` travel with it
    # — a weighting with no provenance cannot be interrogated (ADR 0046).
    return LowOrderPrior(; X = posterior_X, o_X = scenario_window(n, X), mu = mu,
                         sigma = sigma, chol = chol, w = f_prior.w, ens = f_prior.ens,
                         kld = f_prior.kld, ow = f_prior.ow, rr = rr, fpr = fpr)
end
function factor_residual_config(pe::FactorBlackLittermanPrior)
    return (; ve = pe.ve, pdm = pe.rsd ? matrix_processing_posdef(pe.mp) : nothing,
            rsd = pe.rsd)
end

export FactorBlackLittermanPrior
