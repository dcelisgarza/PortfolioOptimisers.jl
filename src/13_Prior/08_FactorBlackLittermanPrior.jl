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
        re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        views::Lc_BLV,
        sets::Option{<:UniverseSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        w::Option{<:VecNum} = nothing,
        rf::Number = 0.0,
        l::Option{<:Number} = nothing,
        tau::Option{<:Number} = nothing,
        rsd::Bool = true,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> FactorBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **lifts** a factor-axis prior onto the asset axis, reconstructing `X` as `F * transpose(M) .+ transpose(b)`, so it builds its carrier directly rather than forwarding one along its own axis; the rule of ADR 0046 still governs each field. It is the member of the Black-Litterman family whose factor block is *modified* rather than passed through — the views land on the factor distribution, and the assets are its projection.

  - The factor block `fpr` is the **posterior** factor distribution, processed by `f_mp`, with `chol` dropped because the posterior covariance supersedes the one it factorises. Its `w` and that weighting's diagnostics forward untouched.
  - `mu` and `sigma` are that block projected through the loadings, so the returned carrier is **internally consistent**: `mu == rr.M * fpr.mu + rr.b` holds by construction, whatever `rf` is, because the rate is inside `fpr.mu` where it is present at all. Measured on a `250 × 5` sample over three factors with two factor views, the two sides agree to `0.0` at `rf = 0.0` and at `rf = 0.03`. `sigma` optionally gains a residual correction when `rsd` is `true`.
  - `w` is the factor prior's, and is over the right axis: this estimator wraps only a factor prior, and `posterior_X` has exactly `F`'s rows, so it is the only weighting in existence.

Its siblings differ: [`BayesianBlackLittermanPrior`](@ref) also satisfies the identity exactly, while [`BlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) do not — see their warnings.

## The views are written on the factor axis

`views` resolves against `sets.dict[sets.tfkey]` — the axis [`UniverseSets`](@ref) declares for factors — because that is the distribution they update. The asset axis is still required (every `UniverseSets` carries one) and is what [`port_opt_view`](@ref) slices; the factor entries come back untouched, which is why this field is `@vprop` rather than exempted by hand.

`sets.dict[sets.tfkey]` must name the columns of `F` **in order**; [`factor_universe`](@ref) checks it, and reports the factor axis rather than the asset one when it is missing or the wrong length.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

`w` is **not** validated here. Its length is a property of the returns matrix, which the constructor never sees, so a wrong length surfaces at [`prior`](@ref) as a `DimensionMismatch` out of [`equilibrium_mu`](@ref) and only when `l` is set. The constructor also accepts an empty `w`, where the sibling [`AugmentedBlackLittermanPrior`](@ref) refuses one.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `re`: Recursively updated via [`factory`](@ref).
  - `ve`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `re`: Recursively viewed via [`port_opt_view`](@ref).
  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
             │   fill_limit ┴ nothing
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
         rsd ┴ Bool: true
```

## The incremental fit

This prior has no exact incremental recursion, so it takes the online step by **refitting from a sample buffer**: [`Online`](@ref) seeds `cache`, [`partial_fit!`](@ref) appends each observation to it verbatim, and the one-argument [`prior`](@ref) runs this estimator's own batch verb over the rows the buffer kept. The answer is therefore exactly a batch fit over those rows, and a `max_history` on the wrapper windows the whole fit. ADR 0136 records the decision.

`cache` travels the three propagation channels as every partial-fit state does: [`factory`](@ref) carries it unchanged, [`port_opt_view`](@ref) slices it to the selected assets, and [`obs_weights_view`](@ref) drops it, because no slice of a state exists on the observation axis. It is not rendered, because a running buffer is not the configuration a reader looks the type up for.

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

  - $(ref_dict[:black1992])
  - $(ref_dict[:cajas2025]) Section 5.1, Equations 5.13 to 5.15, over the factor axis, and Section 4.1, Equations 4.2 and 4.3, for the lift onto the assets.
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
    $(field_dict[:rsd])
    """
    rsd
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function FactorBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                                       f_mp::AbstractMatrixProcessingEstimator,
                                       mp::AbstractMatrixProcessingEstimator,
                                       re::AbstractTimeSeriesRegressionEstimator,
                                       ve::AbstractVarianceEstimator, views::Lc_BLV,
                                       sets::Option{<:UniverseSets},
                                       views_conf::Option{<:Num_VecNum},
                                       w::Option{<:VecNum}, rf::Number, l::Option{<:Number},
                                       tau::Option{<:Number}, rsd::Bool,
                                       cache::Option{<:AbstractPartialFitState})
        assert_bl(views, sets, views_conf, tau)
        return new{typeof(pe), typeof(f_mp), typeof(mp), typeof(re), typeof(ve),
                   typeof(views), typeof(sets), typeof(views_conf), typeof(w), typeof(rf),
                   typeof(l), typeof(tau), typeof(rsd), typeof(cache)}(pe, f_mp, mp, re, ve,
                                                                       views, sets,
                                                                       views_conf, w, rf, l,
                                                                       tau, rsd, cache)
    end
end
function FactorBlackLittermanPrior(;
                                   pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                   f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
                                   ve::AbstractVarianceEstimator = SimpleVariance(),
                                   views::Lc_BLV, sets::Option{<:UniverseSets} = nothing,
                                   views_conf::Option{<:Num_VecNum} = nothing,
                                   w::Option{<:VecNum} = nothing, rf::Number = 0.0,
                                   l::Option{<:Number} = nothing,
                                   tau::Option{<:Number} = nothing, rsd::Bool = true,
                                   cache::Option{<:AbstractPartialFitState} = nothing)::FactorBlackLittermanPrior
    return FactorBlackLittermanPrior(pe, f_mp, mp, re, ve, views, sets, views_conf, w, rf,
                                     l, tau, rsd, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`FactorBlackLittermanPrior`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:FactorBlackLittermanPrior, true)` to render it. ADR 0105 records the decision.

# Arguments

  - `::FactorBlackLittermanPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pe, :f_mp, :mp, :re, :ve, :views, :sets, :views_conf, :w, :rf, :l, :tau, :rsd)`.

# Related

  - [`FactorBlackLittermanPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::FactorBlackLittermanPrior)
    return (:pe, :f_mp, :mp, :re, :ve, :views, :sets, :views_conf, :w, :rf, :l, :tau, :rsd)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties FactorBlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
    prior(pe::FactorBlackLittermanPrior, X::MatNum, F::MatNum,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute factor Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the factor-based Black-Litterman model, combining an asset prior estimator, matrix post-processing for factors and assets, regression and variance estimators, user or algorithmic views, asset sets, view confidences, weights, risk-free rate, leverage, blending parameter `tau`, and a residual variance flag. This method supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates factor regression and residual adjustment for posterior inference.

When `pe.tau` is `nothing` the blending parameter is `1/T`, where `T` is the number of observations of the oriented `X`. The mean handed to the update is a **total** return over the factors, which is the scale the view returns in `Q` are written on, and the factor block is reported on that scale. `pe.rf` reaches the update on the `pe.l` branch alone, because that is the only branch whose prior mean is a risk premium rather than a total return. The closed form below states the shift it leaves there. Where `pe.l` is `nothing` nothing reads the field, so it does not reach the answer.

# Mathematical definition

Black-Litterman views are applied directly to the factor space, updating factor moments ``(\\boldsymbol{\\Pi}_f, \\mathbf{\\Sigma}_f)`` via the standard BL equations, then asset posteriors are reconstructed through the loadings matrix:

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
  - ``r_{f}``: Risk-free rate, added once by [`apply_rf`](@ref) to the equilibrium factor mean. It is therefore inside ``\\hat{\\boldsymbol{\\mu}}_{f,BL}`` where `pe.l` is set, and absent where `pe.l` is `nothing`.

The factor moments are the ordinary Black-Litterman posterior of [`vanilla_posteriors`](@ref), computed over the factor axis: ``\\mathbf{P}`` has ``K`` columns, and ``\\mathbf{\\Sigma}_f`` is the factor prior's covariance. That is literal, not an analogy. Running [`vanilla_posteriors`](@ref) by hand on the factor prior and the [`bl_preroll`](@ref) triple reproduces `pr.fpr.mu` and `pr.fpr.sigma` to `0.0`, and the two lifted forms above reproduce `pr.mu` to `0.0` and `pr.sigma` to `4.8e-16`, at `rsd = true` and at `rsd = false` alike, on a ``250 \\times 5`` sample over three factors. The `chol` this lift returns still factorises the covariance it is returned with: ``\\mathbf{R}^\\intercal\\mathbf{R} - \\hat{\\mathbf{\\Sigma}}`` is `4.8e-16` on both `rsd` branches, so the residual block reaches the factor and the covariance together.

The rate reaches the answer through the factors, and only on one branch. Where `pe.l` is set the equilibrium mean is a risk premium, so [`apply_rf`](@ref) converts it to a total factor return before the update. The blend and the lift then carry that whole factor mean, the rate with it. Writing ``\\mathbf{G} = \\tau\\mathbf{\\Sigma}_f\\mathbf{P}^\\intercal(\\mathbf{P}\\tau\\mathbf{\\Sigma}_f\\mathbf{P}^\\intercal + \\mathbf{\\Omega})^{-1}`` for the update gain and ``\\mathbf{1}`` for the vector of ones, the answer moves against the same estimator at ``r_f = 0`` by:

```math
\\begin{align}
\\Delta\\hat{\\boldsymbol{\\mu}} &= r_{f}\\mathbf{M}\\left(\\mathbf{I} - \\mathbf{G}\\mathbf{P}\\right)\\mathbf{1}\\,.
\\end{align}
```

The shift is linear in ``r_f`` and depends on the views through ``\\mathbf{G}``. It is `[0.489, 0.831, 0.868, 0.183, 0.973]` per unit of ``r_f`` on the sample above, matching the closed form to `1e-16` and agreeing between `rf = 0.03` and `rf = 0.06`. Where `pe.l` is `nothing` the prior mean is the factor prior's own, a total return already, so nothing reads `pe.rf` and the same two fits differ by `0.0` in every entry.

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. When `pe.views` resolves names, check the declared factor axis against the width of `F` with [`factor_universe`](@ref). A precomputed [`BlackLittermanViews`](@ref) resolves no name, so step 6 checks its width instead.
 3. Reduce `X` to the assets it can be fitted over with [`coverage_reduction`](@ref), under `pnl`, giving the mask and `Xi`. This member wraps a *factor* prior, so there is no asset-side prior result to read an Investable Mask off and the gap is read out of the returns themselves.
 4. Fit the wrapped prior `pe.pe` on `F` alone, giving `f_prior`, and read `prior_mu` and `prior_sigma` off it. The wrapped estimator is bounded over the asset axis, but the matrix it is handed here is the factor one.
 5. Regress `Xi` on `F` with [`factor_reconstruction`](@ref) under `pe.re`, giving the regression result `rr` and the reconstructed returns `posterior_X`.
 6. Assemble the views and their uncertainty with [`bl_preroll`](@ref), over `prior_sigma` and `size(Xi, 1)` observations, giving `blp`. The axis is `:tfkey`, so no view row is ever dropped for a departed asset and no ledger is kept.
 7. Put the prior mean on the total-return scale the views are written on, giving `prior_total_mu`. When `pe.l` is set this is the equilibrium mean of [`equilibrium_mu`](@ref), a risk premium, plus `pe.rf` by [`apply_rf`](@ref), over `pe.w` sliced to the reduced axis by [`investable_weights_view`](@ref); otherwise it is `prior_mu`, which is on that scale already.
 8. Run the master equations with [`bl_posteriors`](@ref), giving the posterior factor pair. When no view row survived it hands back `prior_total_mu` and the factor prior covariance instead, and step 10 lifts those exactly as it lifts a posterior pair.
 9. Process the posterior factor covariance in place with [`matrix_processing!`](@ref), under `pe.f_mp` and `F`.
10. Lift the posterior factor pair onto the reduced assets with [`factor_lift`](@ref), giving `mu`, `sigma`, `chol` and `esigma`. This is the lift [`FactorPrior`](@ref) applies; only the factor moments handed to it differ. It adds the residual block when `pe.rsd` is `true`, and processes `sigma` under `pe.mp`.
11. Write `esigma` onto the `esigma` field of `rr`. Under `pe.rsd = true` the field holds the residual variances the lift measured, and under `pe.rsd = false` it holds `nothing`, because the lift added no residual block.
12. Forward the factor block with [`forward_prior`](@ref), replacing `mu` and `sigma` by the posterior factor pair and dropping `chol`. It is not expanded: the reduction never touched the factor axis.
13. Announce the departures once with [`announce_bl_departures`](@ref), naming them with [`investable_universe_names`](@ref).
14. Write every asset-axis block back onto the full universe: the moment pair with [`expand_moment`](@ref), the reconstruction with [`expand_columns`](@ref) and the regression with [`expand_regression`](@ref). `chol` is dropped instead of expanded, because a `NaN` frame has no factorisation.
15. Build the carrier directly, taking `w` and its diagnostics from `f_prior` and carrying no `Z`.

# Arguments

  - `pe`: Factor Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:pnl_prior]) The prior this estimator nests is fitted on the factors, whose axis no panel describes, so the panel stops here.
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Validation

  - `dims in (1, 2)`.
  - If `pe.views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.tfkey)` and `length(pe.sets.dict[pe.sets.tfkey]) == size(F, 2)`, both via [`factor_universe`](@ref).

`pe.w` has no named check. When `pe.l` is set, a `pe.w` whose length is not `size(X, 2)` raises a bare `DimensionMismatch` from the multiplication inside [`equilibrium_mu`](@ref). When `pe.l` is `nothing`, `pe.w` is never read.

# Returns

  - `pr::LowOrderPrior`: Result object carrying the reconstructed asset returns, the posterior asset mean vector, the posterior asset covariance matrix, its Cholesky factor, the factor prior's observation weighting and diagnostics, the regression result, and a factor block `fpr` holding the **posterior** factor moments, on the total-return scale the update ran on. `fpr.chol` is `nothing`, and no `Z` is carried.

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
               kwargs...)
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
    imsk, Xi = coverage_reduction(X, pnl; dims = 1)
    # Factor prior.
    f_prior = prior(pe.pe, F; strict = strict)
    prior_mu, prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr, posterior_X = factor_reconstruction(pe.re, Xi, F)
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
    (; mu, sigma, chol, esigma) = factor_lift(pe.mp, pe.ve, pe.rsd, rr, f_posterior_mu,
                                              f_posterior_sigma, Xi, posterior_X; kwargs...)
    # The lift already measured the residual variances, so the block carries them instead of
    # making every consumer recompute them from the reconstruction error. Under `rsd = false`
    # the lift added no residual block and `esigma` is `nothing`, which is what the field then
    # holds.
    rr = set_idiosyncratic_covariance(rr, esigma)
    # Nothing is added to `mu`. `f_posterior_mu` is a total return over the factors, so the
    # lift gives a total return over the assets, and `rr.b` is applied inside the lift once.
    #
    # No `Z` is forwarded: the only wrapped prior here is `f_prior`, fit on the factors, so
    # its feature matrix would be factors × features and would not describe the asset axis.
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
    # [`expand_regression`](@ref). `o_X` is the caller's own `X` and was never reduced.
    #
    # `chol` is the one thing that cannot be expanded and is dropped instead. It factorises
    # `sigma`, and a `NaN` frame has no factorisation, so writing one into a frame would
    # hand a consumer a triangular matrix that is not a factor of anything. The all-
    # investable path keeps it, because there is nothing to expand there and nothing to drop.
    mu = expand_moment(mu, imsk, 1)
    sigma = expand_moment(sigma, imsk)
    chol = isnothing(imsk) ? chol : nothing
    rr = expand_regression(rr, imsk)
    posterior_X = expand_columns(posterior_X, imsk)
    #
    # The asset-side `w` is the factor prior's: this estimator wraps only a factor prior, and
    # `posterior_X = F*M' + b'` has exactly `F`'s rows, so it is the only weighting in
    # existence and it is over the right observation axis. Its `ens`/`kld`/`ow` travel with it
    # — a weighting with no provenance cannot be interrogated (ADR 0046).
    return LowOrderPrior(; X = posterior_X, o_X = X, mu = mu, sigma = sigma, chol = chol,
                         w = f_prior.w, ens = f_prior.ens, kld = f_prior.kld,
                         ow = f_prior.ow, rr = rr, fpr = fpr)
end
function factor_residual_config(pe::FactorBlackLittermanPrior)
    return (; ve = pe.ve, pdm = pe.mp.pdm, rsd = pe.rsd)
end

export FactorBlackLittermanPrior
