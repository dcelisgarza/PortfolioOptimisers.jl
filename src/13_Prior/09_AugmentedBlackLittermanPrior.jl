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
\\boldsymbol{\\mu}_{aug} &= \\begin{pmatrix}\\boldsymbol{\\mu}_a \\\\ \\boldsymbol{\\mu}_f\\end{pmatrix}
\\quad\\text{or}\\quad
\\lambda\\begin{pmatrix}\\boldsymbol{\\Sigma}_a \\\\ \\boldsymbol{\\Sigma}_f\\mathbf{M}^{\\intercal}\\end{pmatrix}\\boldsymbol{w} + \\begin{pmatrix}\\boldsymbol{b} \\\\ \\mathbf{0}\\end{pmatrix} + r_{f}\\,, \\\\
\\boldsymbol{\\Sigma}_{aug} &= \\begin{pmatrix}\\boldsymbol{\\Sigma}_a & \\mathbf{M}\\boldsymbol{\\Sigma}_f \\\\ \\boldsymbol{\\Sigma}_f\\mathbf{M}^{\\intercal} & \\boldsymbol{\\Sigma}_f\\end{pmatrix}\\,.
\\end{align}
```

The left prior mean is the stacked pair of wrapped means, and it is what `pe.l === nothing` uses. The right one is the equilibrium alternative `pe.l` selects, and it carries two corrections the wrapped pair does not need: the asset-side intercept ``\\boldsymbol{b}``, and the risk-free rate. Both are levels the wrapped means already contain, and both must be in ``\\boldsymbol{\\mu}_{aug}`` rather than added afterwards, because the update blends this mean against ``\\boldsymbol{q}_{aug}`` by forming the residual ``\\boldsymbol{q}_{aug} - \\mathbf{P}_{aug}\\boldsymbol{\\mu}_{aug}``.

The off-diagonal blocks are the cross-covariance the factor model implies, ``\\mathrm{cov}(\\mathbf{X}, \\mathbf{F}) = \\mathbf{M}\\boldsymbol{\\Sigma}_f``. They are built from the **factor** covariance and not from the asset one, so they carry factor variance alone while the leading block carries factor *and* residual variance. That asymmetry is what opens the gap described in the second warning below.

The views are stacked block-diagonally, each set over its own axis:

```math
\\begin{align}
\\mathbf{P}_{aug} &= \\begin{pmatrix}\\mathbf{P} & \\mathbf{0} \\\\ \\mathbf{0} & \\mathbf{P}_f\\end{pmatrix}\\,, \\quad
\\boldsymbol{q}_{aug} = \\begin{pmatrix}\\boldsymbol{q} \\\\ \\boldsymbol{q}_f\\end{pmatrix}\\,, \\quad
\\boldsymbol{\\Omega}_{aug} = \\begin{pmatrix}\\boldsymbol{\\Omega} & \\mathbf{0} \\\\ \\mathbf{0} & \\boldsymbol{\\Omega}_f\\end{pmatrix}\\,.
\\end{align}
```

Black-Litterman posterior on the augmented space, by the ordinary master equations:

```math
\\begin{align}
\\boldsymbol{\\mu}_{post} &= \\boldsymbol{\\mu}_{aug} + \\tau\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal}\\left(\\tau\\mathbf{P}_{aug}\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal} + \\boldsymbol{\\Omega}_{aug}\\right)^{-1}\\!\\left(\\boldsymbol{q}_{aug} - \\mathbf{P}_{aug}\\boldsymbol{\\mu}_{aug}\\right)\\,, \\\\
\\boldsymbol{\\Sigma}_{post} &= \\boldsymbol{\\Sigma}_{aug} + \\tau\\boldsymbol{\\Sigma}_{aug} - \\tau\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal}\\left(\\tau\\mathbf{P}_{aug}\\boldsymbol{\\Sigma}_{aug}\\mathbf{P}_{aug}^{\\intercal} + \\boldsymbol{\\Omega}_{aug}\\right)^{-1}\\!\\mathbf{P}_{aug}\\tau\\boldsymbol{\\Sigma}_{aug}\\,.
\\end{align}
```

The two halves are then read off by truncation, and nothing else. The intercept and the risk-free rate are in ``\\boldsymbol{\\mu}_{aug}`` above, where the views are blended against them:

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
  - ``\\mathbf{M}``: ``N \\times K`` factor loadings (regression coefficients), `pr.rr.M`.
  - ``\\boldsymbol{b}``: ``N \\times 1`` regression intercept vector, `pr.rr.b`.
  - ``\\boldsymbol{\\mu}_a``, ``\\boldsymbol{\\Sigma}_a``: ``N \\times 1`` and ``N \\times N`` asset prior mean and covariance, from `a_pe`.
  - ``\\boldsymbol{\\mu}_f``, ``\\boldsymbol{\\Sigma}_f``: ``K \\times 1`` and ``K \\times K`` factor prior mean and covariance, from `f_pe`.
  - ``\\boldsymbol{\\mu}_{aug}``, ``\\boldsymbol{\\Sigma}_{aug}``: Augmented (joint asset-factor) prior moments, of length and order ``N + K``.
  - ``\\boldsymbol{\\mu}_{post}``, ``\\boldsymbol{\\Sigma}_{post}``: Augmented posterior moments, of the same shape.
  - ``\\tau``: Scaling parameter for the prior uncertainty, `1/T` by default.
  - ``\\mathbf{P}``, ``\\boldsymbol{q}``, ``\\boldsymbol{\\Omega}``: Asset views matrix, returns vector and uncertainty matrix, over the asset axis.
  - ``\\mathbf{P}_f``, ``\\boldsymbol{q}_f``, ``\\boldsymbol{\\Omega}_f``: The same three over the factor axis.
  - ``\\mathbf{P}_{aug}``, ``\\boldsymbol{q}_{aug}``, ``\\boldsymbol{\\Omega}_{aug}``: The same three stacked over the augmented axis, the asset rows above the factor rows.
  - ``\\hat{\\boldsymbol{\\mu}}``, ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times 1`` and ``N \\times N`` posterior asset moments, `pr.mu` and `pr.sigma`.
  - ``\\hat{\\boldsymbol{\\mu}}_f``, ``\\hat{\\mathbf{\\Sigma}}_f``: ``K \\times 1`` and ``K \\times K`` posterior factor moments, `pr.fpr.mu` and `pr.fpr.sigma`.
  - ``\\lambda``, ``\\boldsymbol{w}``: The risk-aversion coefficient `pe.l` and the equilibrium weights `pe.w`, read by [`equilibrium_mu`](@ref) and only where `pe.l` is set.
  - ``r_{f}``: Risk-free rate, added once by [`apply_rf`](@ref) to the equilibrium mean. It is absent where `pe.l` is `nothing`, because the wrapped means are total returns already.

Every block above is measured on a ``250 \\times 5`` sample over three factors with two asset views and two factor views. The off-diagonal blocks of ``\\boldsymbol{\\Sigma}_{aug}`` are ``\\mathbf{M}\\boldsymbol{\\Sigma}_f`` and its transpose to `0.0`, and that product is the empirical ``\\mathrm{cov}(\\mathbf{X}, \\mathbf{F})`` to `7.6e-19` against a scale of `5.3e-4`. The block-diagonal stack and the truncation agree with a hand computation to `0.0` on ``\\hat{\\boldsymbol{\\mu}}``, ``\\hat{\\boldsymbol{\\mu}}_f`` and ``\\hat{\\mathbf{\\Sigma}}_f``.

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
        tau::Option{<:Number} = nothing
    ) -> AugmentedBlackLittermanPrior

Keywords correspond to the struct's fields.

## Composition: what this estimator forwards

This estimator **merges two** priors rather than forwarding one along its own axis, so it builds its carrier directly; the rule of ADR 0046 still governs which source each field takes. It solves one augmented Black-Litterman system over `[assets; factors]` and reports both halves:

  - `mu` and `sigma` are the asset half of the augmented posterior; the factor block `fpr` is the **factor half**, so both are posterior. `chol` is dropped on both sides, because the posterior covariances supersede the ones they factorise.
  - `w`, `ens`, `kld` and `ow` come from the **asset** prior, and `fpr`'s own come from the **factor** prior. Two priors disagreeing about observation weights is a legitimate configuration, and the nested block is what keeps the two weightings distinguishable rather than forcing a choice.
  - `Z` comes from the asset prior only: the factor prior's would be factors × features and would not describe this asset axis.

!!! warning

    The returned `mu` and `sigma` are the augmented posterior, but `w` is the **asset prior's** observation weighting, forwarded unchanged (and `fpr.w` is the factor prior's). Black-Litterman produces no observation-level posterior, so there is no Black-Litterman-consistent alternative to forward — and dropping `w` would substitute the unweighted empirical distribution, which is further from the caller's intent than the weights they computed. A caller reading `pr.w`, `pr.ens`, `pr.kld` or `pr.ow` is therefore reading a property of the asset prior, not of the posterior. Measured: `pr.w`, `pr.ens`, `pr.kld` and `pr.ow` are the identical objects the asset prior carried, and `pr.fpr.w` the identical object the factor prior carried.

!!! warning

    `pr.mu != pr.rr.M * pr.fpr.mu + pr.rr.b`, even though **both** blocks are posterior. Two independent causes open the gap, and both are properties of the update. The intercept is **not** one of them: it enters the prior stack, and the update is affine in that stack, so it reaches both sides of the identity together.

    **Idiosyncratic variance, on every branch.** The augmented covariance stacks the full asset covariance `sigma_a` — factor *and* residual variance — against a cross-covariance `M * sigma_f` that is pure factor. The update therefore moves the asset half by `tau * sigma_a * P'(…)` and the factor half by `tau * sigma_f * M' * P'(…)`, and for the two to stay related by `M` it would need `sigma_a == M * sigma_f * M'`. That holds only when the factor model is exact. This part scales with the residual variance, and both view sets contribute to it.

    **A non-zero `rf`, on the `l` branch alone.** [`apply_rf`](@ref) puts the rate on the whole equilibrium stack, so the asset half gains `rf` and the factor half gains `rf` too. The identity carries the factor half through the loadings, which turns that `rf` into `rf * s` for the row sums `s` of `M`, so the two sides agree only where an asset's loadings sum to one. The gap this opens is `rf * (1 - s)`, exactly. It is zero at the default `rf = 0.0`, and where `pe.l` is `nothing` the field is never read.

    Measured on a `250 × 5` sample over three factors with two asset views and two factor views, at the default `rf = 0.0` unless stated. On returns built as an exact factor model whose intercept is `[0.001, -0.002, 0.0015, 0.0005, 0.0]` the gap is `1.8e-15` at `l = nothing` and `2.3e-15` at `l = 2.0`; on the same returns with a zero intercept it is `3.2e-15` and `3.7e-15`, so the intercept does not reach it. On returns built with a residual and a fitted intercept of `2.4e-4` it is `1.2e-4` and `1.4e-4`. At `l = 2.0` and `rf = 0.03` the exact returns give `1.8e-2`, which is `rf * (1 - s)` to `1e-13`.

    The two *priors* satisfy the identity before the update only when both means are the plain sample mean, because least squares with an intercept zeroes the **unweighted** residual mean — measured at `1.7e-18`. One shared *non-uniform* weighting is not enough: the same weighting on both priors gives `1.1e-4`, because the weighted residual mean is not zero, and two different weightings give `1.3e-3`. On an exact factor model a shared non-uniform weighting does satisfy it, at `4.8e-18`, because there is no residual to weight.

    [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) satisfy the identity exactly, because they update the factor distribution alone and *project* it onto the assets rather than updating an asset block alongside it. [`BlackLittermanPrior`](@ref) breaks it for the opposite reason — it takes asset views only and never computes a posterior factor distribution at all, so `pr.fpr` is `nothing` and the right-hand side cannot be formed.

## One sets, two axes

This is the only estimator whose views land on **both** distributions, and the two axes it needs are the two [`UniverseSets`](@ref) declares: `a_views` resolves against `sets.dict[sets.xkey]`, `f_views` against `sets.dict[sets.tfkey]`. Before the axis was declared this took two separate sets objects, and the factor-flavoured one had to be exempted from [`port_opt_view`](@ref) **by hand** — a missing annotation was all that stood between a view and a factor universe sliced by asset indices. With one dual-axis object the exemption is a property of the data: the field is `@vprop`, the slice moves the asset entries and the factor entries come back untouched.

Each axis is required only by the views that resolve names against it. A [`BlackLittermanViews`](@ref) result carries its own `P` and needs no universe at all, so asset-views-only and factor-views-only mandates are both expressible with a single `sets` — and a pair of precomputed view sets needs none. Measured over four mandates on one fixture: named views on both axes, named asset views against precomputed factor views, precomputed asset views against named factor views, and a precomputed pair. All four run, and the precomputed pair with no `sets` reproduces the same pair supplied with `sets` to `0.0`. [`port_opt_view`](@ref) over the selection `[1, 3, 5]` of five assets leaves `sets.dict[sets.tfkey]` at its full three factors while `sets.dict[sets.xkey]` and `w` both fall to three entries.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.
  - If either `a_views` or `f_views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `a_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `f_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

The **length** of `w` is not validated here. It is a property of the returns matrix, which the constructor never sees, so a wrong length surfaces at [`prior`](@ref) as a `DimensionMismatch` out of [`equilibrium_mu`](@ref) and only when `l` is set.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `a_pe`: Recursively updated via [`factory`](@ref).
  - `f_pe`: Recursively updated via [`factory`](@ref).
  - `re`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `a_pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `re`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
          f_pe ┼ EmpiricalPrior
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
               │     xkey ┼ String: "nx"
               │    uxkey ┼ String: "ux"
               │    tfkey ┼ String: "nf"
               │   utfkey ┼ String: "uf"
               │    cfkey ┼ String: "ncf"
               │   ucfkey ┼ String: "ucf"
               │     zkey ┼ String: "nz"
               │     dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"], "nf" => ["F1", "F2"])
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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cheung2007])
  - $(ref_dict[:cajas2025]) Section 5.2, Equations 5.17 to 5.19.
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
    function AugmentedBlackLittermanPrior(a_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          f_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          mp::AbstractMatrixProcessingEstimator,
                                          re::AbstractTimeSeriesRegressionEstimator,
                                          a_views::Lc_BLV, f_views::Lc_BLV,
                                          sets::Option{<:UniverseSets},
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
                                      re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
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

When `pe.tau` is `nothing` the blending parameter is `1/T`, where `T` is the number of observations of the oriented `X`. The mean handed to the update is a **total** return, and both halves are reported on that scale. `pe.rf` reaches the update on the `pe.l` branch alone, where it converts the equilibrium risk premium, and it goes on the whole stack. Writing ``\\mathbf{G}`` for the augmented update gain and ``\\mathbf{1}`` for the vector of ones, the answer therefore moves against the same estimator at `rf = 0` by `rf * (I - G P_aug) * 1`, read on each half. On a `250 × 5` sample over three factors with two views on each axis the asset half moves by `[0.306, 0.500, 0.289, 0.169, 0.266]` per unit of `rf` and the factor half by `[0.390, 0.453, 0.991]`, both matching that closed form to `1e-16` and linear between `rf = 0.03` and `rf = 0.06`. Where `pe.l` is `nothing` the field is never read, and two fits differing only in `rf` agree to `0.0` on both halves.

# Arguments

  - `pe`: Augmented Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - $(arg_dict[:dims])
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Validation

  - `dims in (1, 2)`.
  - If `pe.a_views` is a [`LinearConstraintEstimator`](@ref), `length(pe.sets.dict[pe.sets.xkey]) == size(X, 2)`.
  - If `pe.f_views` is a [`LinearConstraintEstimator`](@ref), `haskey(pe.sets.dict, pe.sets.tfkey)` and `length(pe.sets.dict[pe.sets.tfkey]) == size(F, 2)`, both via [`factor_universe`](@ref).

`pe.w` has no named check. When `pe.l` is set, a `pe.w` whose length is not `size(X, 2)` raises a bare `DimensionMismatch` from the multiplication inside [`equilibrium_mu`](@ref). When `pe.l` is `nothing`, `pe.w` is never read.

# Returns

  - `pr::LowOrderPrior`: Result object carrying the reconstructed asset returns, the asset half of the augmented posterior as `mu` and `sigma`, the asset prior's observation weighting and diagnostics, its feature matrix, the regression result, and a factor block `fpr` holding the **factor half** of the same posterior. `chol` is `nothing` on both blocks.

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. When `pe.a_views` resolves names, check that the declared asset axis is as long as `X` is wide.
 3. When `pe.f_views` resolves names, check the declared factor axis against the width of `F` with [`factor_universe`](@ref). Each axis is checked only by the views that resolve names against it, so a pair of precomputed [`BlackLittermanViews`](@ref) needs no `sets` at all.
 4. Fit `pe.a_pe` on `X`, giving `a_prior`, and `pe.f_pe` on `F`, giving `f_prior`.
 5. Regress `X` on `F` with [`factor_reconstruction`](@ref) under `pe.re`, giving `rr` and the reconstructed returns `posterior_X`.
 6. Assemble the asset views with [`bl_preroll`](@ref) at the default `:xkey`, over the asset prior covariance, and the factor views at `:tfkey`, over the factor prior covariance.
 7. Build ``\\boldsymbol{\\Sigma}_{aug}``, whose off-diagonal blocks are the model-implied cross-covariance ``\\mathbf{M}\\boldsymbol{\\Sigma}_f`` and its transpose.
 8. Stack ``\\mathbf{P}_{aug}`` block-diagonally, ``\\boldsymbol{q}_{aug}`` and ``\\boldsymbol{\\Omega}_{aug}`` to match, the asset rows above the factor rows.
 9. Put the stacked prior mean on the total-return scale the views are written on, giving `aug_prior_mu`. When `pe.l` is `nothing` this is the stacked wrapped means, which are on that scale already. When `pe.l` is set it is the equilibrium mean of [`equilibrium_mu`](@ref), a bare risk premium, plus `pe.rf` by [`apply_rf`](@ref) and plus `rr.b` on the asset half.
10. Run the master equations with [`vanilla_posteriors`](@ref) over the augmented space, giving the augmented posterior pair.
11. Process the augmented posterior covariance in place with [`matrix_processing!`](@ref), under `pe.mp` and the two return matrices side by side.
12. Truncate the asset half from `1:N`. Nothing is added to it: the intercept and the rate went into the prior mean at step 9, and the update is affine in that mean.
13. Truncate the factor half from `N+1:N+K`, and forward the factor block with [`forward_prior`](@ref), dropping `chol`. The half takes no intercept, because the intercept is the regression's and hence asset-only, no rate, because the stack reached the update carrying the one it needed, and no second processing pass, because a principal submatrix of a processed matrix is already processed.
14. Build the carrier directly, taking `w`, its diagnostics and `Z` from `a_prior`.

# Related

  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
  - [`apply_rf`](@ref)
  - [`equilibrium_mu`](@ref)
"""
function prior(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum; dims::Int = 1,
               strict::Bool = false, kwargs...)
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
    a_prior = prior(pe.a_pe, X; strict = strict, kwargs...)
    a_prior_mu, a_prior_sigma = a_prior.mu, a_prior.sigma
    # Factor prior.
    f_prior = prior(pe.f_pe, F; strict = strict)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors. Only the reconstruction is shared with `FactorPrior`:
    # the asset moments here come out of the augmented system, not out of a lift.
    rr, posterior_X = factor_reconstruction(pe.re, X, F)
    (; b, M) = rr
    dt = eltype(posterior_X)
    T = size(X, 1)
    # One sets, two axes: the asset views take the default `:xkey`, the factor views `:tfkey`.
    (; P, Q, tau, omega) = bl_preroll(pe.a_views, pe.sets, pe.a_views_conf, a_prior_sigma,
                                      pe.tau, T, dt, strict)
    a_omega = omega
    f_result = bl_preroll(pe.f_views, pe.sets, pe.f_views_conf, f_prior_sigma, pe.tau, T,
                          dt, strict, :tfkey)
    f_P, f_Q, f_omega = f_result.P, f_result.Q, f_result.omega
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
    aug_prior_mu = if !isnothing(pe.l)
        apply_rf(pe.rf,
                 equilibrium_mu(pe.l, vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                                pe.w)) .+ vcat(b, zeros(dt, size(F, 2)))
    else
        vcat(a_prior_mu, f_prior_mu)
    end
    aug_posterior_mu, aug_posterior_sigma = vanilla_posteriors(tau, aug_prior_mu,
                                                               aug_prior_sigma, aug_omega,
                                                               aug_P, aug_Q)
    matrix_processing!(pe.mp, aug_posterior_sigma, hcat(posterior_X, F))
    # Nothing is added here. `aug_prior_mu` reached the update on the scale this estimator
    # returns, and the update is affine in it, so the asset half comes off that scale with
    # the intercept and the rate already in it. Truncation is the whole of the step.
    posterior_mu = aug_posterior_mu[1:size(X, 2)]
    posterior_sigma = aug_posterior_sigma[1:size(X, 2), 1:size(X, 2)]
    # The augmented system is jointly posterior over `[assets; factors]`, so truncating it to
    # the asset half discards a factor half that *is* the posterior factor distribution. The
    # factor block reports that half rather than `f_prior`'s prior moments. It is truncated and
    # nothing else: no `b`, because the intercept is the regression's and hence asset-only, and
    # no rate, because the stack reached the update on the total-return scale and comes off it
    # on the same one. The factor half is therefore on the scale `f_prior` supplied. No second
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
    return LowOrderPrior(; X = posterior_X, o_X = X, mu = posterior_mu,
                         sigma = posterior_sigma, w = a_prior.w, ens = a_prior.ens,
                         kld = a_prior.kld, ow = a_prior.ow, rr = rr, fpr = fpr,
                         Z = a_prior.Z)
end

function factor_residual_config(::AugmentedBlackLittermanPrior)
    # The augmented system produces the asset moments whole, so this estimator never calls
    # [`factor_lift`](@ref) and adds no residual block. See
    # [`factor_residual_config`](@ref).
    return nothing
end

export AugmentedBlackLittermanPrior
