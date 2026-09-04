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
        rsd::Bool = true
    ) -> FactorBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **lifts** a factor-axis prior onto the asset axis, reconstructing `X` as `F * transpose(M) .+ transpose(b)`, so it builds its carrier directly rather than forwarding one along its own axis; the rule of ADR 0046 still governs each field. It is the member of the Black-Litterman family whose factor block is *modified* rather than passed through — the views land on the factor distribution, and the assets are its projection.

  - The factor block `fpr` is the **posterior** factor distribution, processed by `f_mp`, with `chol` dropped because the posterior covariance supersedes the one it factorises. Its `w` and that weighting's diagnostics forward untouched.
  - `mu` and `sigma` are that block projected through the loadings, so the returned carrier is **internally consistent**: `mu == rr.M * fpr.mu + rr.b` holds by construction, whatever `rf` is, because the rate is inside `fpr.mu` where it is present at all. Measured on a `250 × 5` sample over three factors with two factor views, the two sides agree to `0.0` at `rf = 0.0` and at `rf = 0.03`. `sigma` optionally gains a residual correction when `rsd` is `true`.
  - `w` is the factor prior's, and is over the right axis: this estimator wraps only a factor prior, and `posterior_X` has exactly `F`'s rows, so it is the only weighting in existence.
  - No `Z` is carried: the only wrapped prior is fit on factors, so its feature matrix would be factors × features and would not describe the asset axis. The drop is a *relocation* rather than a destruction — the factor block is forwarded, so a feature matrix the factor prior carried is still reachable at `pr.fpr.Z`. Wrap this estimator from the *outside* with [`FeaturePrior`](@ref) if an asset-axis feature matrix is wanted.

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
             │        ce ┼ PortfolioOptimisersCovariance
             │           │   ce ┼ Covariance
             │           │      │    me ┼ SimpleExpectedReturns
             │           │      │       │   w ┴ nothing
             │           │      │    ce ┼ GeneralCovariance
             │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │           │      │       │    w ┴ nothing
             │           │      │   alg ┼ FullMoment()
             │           │      │     w ┴ nothing
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
             │     xkey ┼ String: "nx"
             │    uxkey ┼ String: "ux"
             │    tfkey ┼ String: "nf"
             │   utfkey ┼ String: "uf"
             │    cfkey ┼ String: "ncf"
             │   ucfkey ┼ String: "ucf"
             │     zkey ┼ String: "nz"
             │     dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"], "nf" => ["F1", "F2"])
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
    function FactorBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                                       f_mp::AbstractMatrixProcessingEstimator,
                                       mp::AbstractMatrixProcessingEstimator,
                                       re::AbstractTimeSeriesRegressionEstimator,
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
                                   re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
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
 2. When `pe.views` resolves names, check the declared factor axis against the width of `F` with [`factor_universe`](@ref). A precomputed [`BlackLittermanViews`](@ref) resolves no name, so step 4 checks its width instead.
 3. Fit the wrapped prior `pe.pe` on `F` alone, giving `f_prior`, and read `prior_mu` and `prior_sigma` off it. The wrapped estimator is bounded over the asset axis, but the matrix it is handed here is the factor one.
 4. Regress `X` on `F` with [`factor_reconstruction`](@ref) under `pe.re`, giving the regression result `rr` and the reconstructed returns `posterior_X`.
 5. Assemble the views and their uncertainty with [`bl_preroll`](@ref), over `prior_sigma` and `size(X, 1)` observations, giving `P`, `Q`, `tau` and `omega`. The axis is `:tfkey`.
 6. Put the prior mean on the total-return scale the views are written on, giving `prior_total_mu`. When `pe.l` is set this is the equilibrium mean of [`equilibrium_mu`](@ref), a risk premium, plus `pe.rf` by [`apply_rf`](@ref); otherwise it is `prior_mu`, which is on that scale already.
 7. Run the master equations with [`vanilla_posteriors`](@ref), giving the posterior factor pair.
 8. Process the posterior factor covariance in place with [`matrix_processing!`](@ref), under `pe.f_mp` and `F`.
 9. Lift the posterior factor pair onto the assets with [`factor_lift`](@ref), giving `mu`, `sigma` and `chol`. This is the lift [`FactorPrior`](@ref) applies; only the factor moments handed to it differ. It adds the residual block when `pe.rsd` is `true`, and processes `sigma` under `pe.mp`.
10. Forward the factor block with [`forward_prior`](@ref), replacing `mu` and `sigma` by the posterior factor pair and dropping `chol`.
11. Build the carrier directly, taking `w` and its diagnostics from `f_prior` and carrying no `Z`.

# Arguments

  - `pe`: Factor Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
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
function prior(pe::FactorBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
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
    # Factor prior.
    f_prior = prior(pe.pe, F; strict = strict)
    prior_mu, prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr, posterior_X = factor_reconstruction(pe.re, X, F)
    M = rr.M
    (; P, Q, tau, omega) = bl_preroll(pe.views, pe.sets, pe.views_conf, prior_sigma, pe.tau,
                                      size(X, 1), eltype(posterior_X), strict, :tfkey)
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
    prior_total_mu = if !isnothing(pe.l)
        apply_rf(pe.rf, equilibrium_mu(pe.l, prior_sigma * transpose(M), pe.w))
    else
        prior_mu
    end
    f_posterior_mu, f_posterior_sigma = vanilla_posteriors(tau, prior_total_mu, prior_sigma,
                                                           omega, P, Q)
    matrix_processing!(pe.f_mp, f_posterior_sigma, F)
    # Reconstruct the posteriors using the black litterman adjusted factor statistics. The lift
    # is the same one `FactorPrior` applies; only the factor moments handed to it differ.
    (; mu, sigma, chol) = factor_lift(pe.mp, pe.ve, pe.rsd, rr, f_posterior_mu,
                                      f_posterior_sigma, X, posterior_X; kwargs...)
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
