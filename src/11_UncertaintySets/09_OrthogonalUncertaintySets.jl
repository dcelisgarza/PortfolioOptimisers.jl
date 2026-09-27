"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the scalings that size each direction inside the Orthogonal Subspace.

A member states the ``r \\times r`` matrix ``\\mathbf{\\Lambda}`` of the mean set, and the square root of that matrix maps the coordinates of the ball onto the basis of the subspace. Every member confines the set to the same subspace, so a scaling changes the shape of the set alone, and the radius reads the same ``r`` degrees of freedom under each member.

# Interfaces

## `orthogonal_scaling`

  - `orthogonal_scaling(scaling::AbstractOrthogonalScaling, G::MatNum, rr::AbstractLoadingsRegressionResult) -> MatNum`: Returns ``\\mathbf{\\Lambda}``, ``r \\times r``, symmetric and positive semi-definite.

# Related

  - [`IdentityScaling`](@ref)
  - [`IdiosyncraticVarianceScaling`](@ref)
  - [`orthogonal_scaling`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
"""
abstract type AbstractOrthogonalScaling <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Gives every direction of the Orthogonal Subspace the same uncertainty, the default.

The geometry map of the mean set is then the basis of the subspace, and the set is a Euclidean ball inside the subspace.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Lambda} &= \\mathbf{I}_{r}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Lambda_orth])
  - $(math_dict[:r_orth])

# Algorithm

The branch of [`orthogonal_scaling`](@ref) that this tag selects runs this step.

 1. Return the identity of side `size(G, 2)`, the dimension of the subspace, as a dense matrix of the element type of `G`. The caller then takes one kind of square root on both branches.

# Examples

```jldoctest
julia> IdentityScaling()
IdentityScaling()
```

# Related

  - [`AbstractOrthogonalScaling`](@ref)
  - [`IdiosyncraticVarianceScaling`](@ref)
  - [`orthogonal_scaling`](@ref)
"""
struct IdentityScaling <: AbstractOrthogonalScaling end
"""
$(DocStringExtensions.TYPEDEF)

Sizes each direction of the Orthogonal Subspace by the idiosyncratic covariance projected onto it.

A direction that the factors leave noisy gets more uncertainty than a quiet one, where [`IdentityScaling`](@ref) gives both the same. The loadings block must carry an idiosyncratic covariance, and a block whose `esigma` is unset refuses.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Lambda} &= \\mathbf{G}^{\\intercal}\\mathbf{D}\\mathbf{G}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Lambda_orth])
  - $(math_dict[:G_orth])
  - $(math_dict[:D_orth])

# Algorithm

The branch of [`orthogonal_scaling`](@ref) that this tag selects runs these steps.

 1. Read `D`, the idiosyncratic covariance of `rr`. A stored matrix is read whole, with its off-diagonal entries. A stored variance vector is read as a `LinearAlgebra.Diagonal`, so the ``N \\times N`` matrix is never materialised. An unset `esigma` refuses through [`idiosyncratic_variances`](@ref).
 2. Form `lambda` by the definition above.
 3. Return the symmetric part of `lambda`. The product carries a round-off asymmetry, and the caller's square root reads a symmetric matrix.

# Examples

```jldoctest
julia> IdiosyncraticVarianceScaling()
IdiosyncraticVarianceScaling()
```

# Related

  - [`AbstractOrthogonalScaling`](@ref)
  - [`IdentityScaling`](@ref)
  - [`orthogonal_scaling`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
struct IdiosyncraticVarianceScaling <: AbstractOrthogonalScaling end
"""
    orthogonality_weights(::IdentityMetric, ::AbstractLoadingsRegressionResult)
    orthogonality_weights(::InverseIdiosyncraticVarianceMetric, rr::AbstractLoadingsRegressionResult)
    orthogonality_weights(::BenchmarkWeightMetric, rr::AbstractLoadingsRegressionResult)
    orthogonality_weights(::RegressionWeightMetric, rr::AbstractLoadingsRegressionResult)
    orthogonality_weights(::BenchmarkWeightMetric, rr::CrossSectionalFactorModel)
    orthogonality_weights(::RegressionWeightMetric, rr::CrossSectionalFactorModel)

Cross-sectional weight vector the [`AbstractOrthogonalityMetric`](@ref) names, read off a fitted loadings block.

# Algorithm

 1. On [`IdentityMetric`](@ref), return `nothing`. The caller reads a `nothing` as a vector of ones, and skips the scaling of the loadings and the division of the projector.
 2. On [`InverseIdiosyncraticVarianceMetric`](@ref), read `d`, the idiosyncratic variances, with [`idiosyncratic_variances`](@ref), refuse an entry that is not finite or not `> 0`, and return the element-wise inverse of `d`.
 3. On [`BenchmarkWeightMetric`](@ref) and [`RegressionWeightMetric`](@ref) over a [`CrossSectionalFactorModel`](@ref), return the last row of `bw` or of `rw`, the weights of the latest observation, through [`latest_orthogonality_weights`](@ref).
 4. On the same two metrics over any other loadings block, throw. A block fitted per asset over the observations carries no cross-sectional weight history, so the metric has nothing to read.

# Arguments

  - `metric`: Orthogonality metric.
  - `rr`: Fitted loadings block.

# Validation

  - On [`InverseIdiosyncraticVarianceMetric`](@ref): every idiosyncratic variance is finite and `> 0`, else a `DomainError`. `idiosyncratic_variances` itself throws when the block carries none.
  - On a weight history: the history is non-empty, and every weight of its last row is finite and `> 0`, else an `IsEmptyError` or a `DomainError`.
  - On a weight metric over a block that carries no history: an `IsNothingError` naming the field.

# Returns

  - `w::Option{<:VecNum}`: Weight vector of length ``N``, or `nothing` on [`IdentityMetric`](@ref).

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`idiosyncratic_variances`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function orthogonality_weights(::IdentityMetric, ::AbstractLoadingsRegressionResult)
    return nothing
end
function orthogonality_weights(::InverseIdiosyncraticVarianceMetric,
                               rr::AbstractLoadingsRegressionResult)
    d = idiosyncratic_variances(rr)
    @argcheck(all(x -> isfinite(x) && x > zero(x), d),
              DomainError(d,
                          "every idiosyncratic variance must be finite and > 0 to be inverted into a cross-sectional weight"))
    return inv.(d)
end
function orthogonality_weights(::BenchmarkWeightMetric,
                               rr::AbstractLoadingsRegressionResult)
    return throw(IsNothingError("`BenchmarkWeightMetric` reads the benchmark weight history `bw` off the loadings block, and a `$(nameof(typeof(rr)))` carries none. A block fitted per asset over the observations states no cross-sectional weight.\nUse `InverseIdiosyncraticVarianceMetric` or `IdentityMetric`, or fit the prior with a cross-sectional factor model, which fills `bw`.\nGot\nrr => $(nameof(typeof(rr)))\nbw => absent"))
end
function orthogonality_weights(::RegressionWeightMetric,
                               rr::AbstractLoadingsRegressionResult)
    return throw(IsNothingError("`RegressionWeightMetric` reads the regression weight history `rw` off the loadings block, and a `$(nameof(typeof(rr)))` carries none. A block fitted per asset over the observations states no cross-sectional weight.\nUse `InverseIdiosyncraticVarianceMetric` or `IdentityMetric`, or fit the prior with a cross-sectional factor model, which fills `rw`.\nGot\nrr => $(nameof(typeof(rr)))\nrw => absent"))
end
function orthogonality_weights(::BenchmarkWeightMetric, rr::CrossSectionalFactorModel)
    return latest_orthogonality_weights(rr.bw, :bw, rr)
end
function orthogonality_weights(::RegressionWeightMetric, rr::CrossSectionalFactorModel)
    return latest_orthogonality_weights(rr.rw, :rw, rr)
end
"""
    latest_orthogonality_weights(::Nothing, name::Symbol, rr::AbstractLoadingsRegressionResult)
    latest_orthogonality_weights(w::MatNum, name::Symbol, ::AbstractLoadingsRegressionResult)

Last row of a cross-sectional weight history, checked as a metric.

# Algorithm

 1. On a `nothing` history, throw. The block declares the field and this fit left it unset, and the message says so.
 2. On a history `w`, refuse an empty one.
 3. Take `wl`, the last row of `w`, as a view. These are the weights of the latest observation. The uncertainty set serves the next decision, so it reads the newest cross-section and not an average of the sample.
 4. Refuse `wl` when an entry is not finite or not `> 0`, and return it otherwise.

# Arguments

  - `w`: Weight history, `observations × assets`, or `nothing`.
  - `name`: Name of the field the history came from, which the refusals quote.
  - `rr`: Fitted loadings block, quoted by the refusals.

# Validation

  - `!isempty(w)`, else an `IsEmptyError`.
  - Every entry of the last row is finite and `> 0`, else a `DomainError`. A weight of zero excluded its asset from the fit, and an excluded asset gives the metric a singular direction. The Investable Mask does not remove this case. An asset can be investable, with a finite return and a stated moment, and still sit outside the estimation universe of the latest cross-section, so its weight is zero while its loadings and variance are finite. The two metrics that read no weight history, [`InverseIdiosyncraticVarianceMetric`](@ref) and [`IdentityMetric`](@ref), accept such an asset.

# Returns

  - `w::VecNum`: Weights of the latest observation, one entry per asset.

# Related

  - [`orthogonality_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function latest_orthogonality_weights(::Nothing, name::Symbol,
                                      rr::AbstractLoadingsRegressionResult)
    return throw(IsNothingError("`$(name)` is unset on this loadings block, so it carries no cross-sectional weight history to read.\nBuild the block with `$(name)` set, or select `InverseIdiosyncraticVarianceMetric` or `IdentityMetric`, which read no weight history.\nGot\nrr => $(nameof(typeof(rr)))\n$(name) => nothing"))
end
function latest_orthogonality_weights(w::MatNum, name::Symbol,
                                      ::AbstractLoadingsRegressionResult)
    @argcheck(!isempty(w), IsEmptyError("$(name) cannot be empty"))
    wl = view(w, size(w, 1), :)
    @argcheck(all(x -> isfinite(x) && x > zero(x), wl),
              DomainError(wl,
                          "every weight of the latest observation of $(name) must be finite and > 0; a weight of 0 excluded its asset from the latest cross-sectional fit and leaves the metric singular. An investable asset can sit outside the estimation universe of that fit; use `InverseIdiosyncraticVarianceMetric` or `IdentityMetric`, which read no weight history"))
    return wl
end
"""
    orthogonal_scaling(::IdentityScaling, G::MatNum, ::AbstractLoadingsRegressionResult)
    orthogonal_scaling(::IdiosyncraticVarianceScaling, G::MatNum, rr::AbstractLoadingsRegressionResult)

Scaling ``\\mathbf{\\Lambda}`` of the mean set inside the Orthogonal Subspace.

# Algorithm

 1. On [`IdentityScaling`](@ref), run the step that its docstring lists.
 2. On [`IdiosyncraticVarianceScaling`](@ref), run the steps that its docstring lists.

# Arguments

  - `scaling`: Orthogonal scaling.
  - `G`: Orthonormal basis of the Orthogonal Subspace, ``N \\times r``.
  - `rr`: Fitted loadings block.

# Validation

  - On [`IdiosyncraticVarianceScaling`](@ref): [`idiosyncratic_variances`](@ref) throws when the block carries no `esigma`.

# Returns

  - `lambda::MatNum`: ``r \\times r`` scaling, symmetric, and positive semi-definite when the idiosyncratic covariance is.

# Related

  - [`AbstractOrthogonalScaling`](@ref)
  - [`IdentityScaling`](@ref)
  - [`IdiosyncraticVarianceScaling`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
function orthogonal_scaling(::IdentityScaling, G::MatNum,
                            ::AbstractLoadingsRegressionResult)
    return Matrix{eltype(G)}(LinearAlgebra.I, size(G, 2), size(G, 2))
end
function orthogonal_scaling(::IdiosyncraticVarianceScaling, G::MatNum,
                            rr::AbstractLoadingsRegressionResult)
    # A stored covariance is read whole, off-diagonal entries included. A variance vector
    # is read as a diagonal, and an unset field refuses through `idiosyncratic_variances`.
    D = if isa(rr.esigma, MatNum)
        rr.esigma
    else
        LinearAlgebra.Diagonal(idiosyncratic_variances(rr))
    end
    lambda = transpose(G) * D * G
    return (lambda + transpose(lambda)) / 2
end
"""
    k_norm_ball(::NormalKUncertaintyAlgorithm, ::Number, ::Nothing, ::MatNum, ::Integer)

Refuses the sampled radius on the Orthogonal Subspace, which simulates no estimation errors for that radius to read.

[`NormalKUncertaintyAlgorithm`](@ref) takes a quantile of a sample of Mahalanobis distances, and [`OrthogonalUncertaintySet`](@ref) draws no sample, because it derives its geometry from the fitted loadings alone. Without this method the absent sample raises a `MethodError`, which names no remedy.

# Arguments

  - The radius algorithm, the significance level, the absent sample, the scaling and the degrees of freedom, all ignored.

# Validation

  - The method always throws an `ArgumentError`. The message names the two algorithms that read no sample, [`ChiSqKUncertaintyAlgorithm`](@ref) and [`GeneralKUncertaintyAlgorithm`](@ref), and a number, which states the radius itself.

# Returns

  - Never returns.

# Related

  - [`k_norm_ball`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
"""
function k_norm_ball(::NormalKUncertaintyAlgorithm, ::Number, ::Nothing, ::MatNum,
                     ::Integer)
    return throw(ArgumentError("`NormalKUncertaintyAlgorithm` sizes the radius from a sample of estimation errors, and `OrthogonalUncertaintySet` simulates none: its geometry comes from the fitted loadings alone, so there is no sample of Mahalanobis distances to take a quantile of.\nUse `ChiSqKUncertaintyAlgorithm`, which reads the dimension of the subspace, `GeneralKUncertaintyAlgorithm`, which reads neither, or state the radius as a number."))
end
"""
$(DocStringExtensions.TYPEDEF)

Fits both uncertainty sets from the factor model of the optimisation's own prior, confined to the directions the factors do not span.

The estimator reads the loadings block `rr` of the Prior Result it receives, and never fits a prior of its own. It is the one member of [`AbstractPriorUncertaintySetEstimator`](@ref). The two JuMP builders pass the reduced prior beside the returns, so the sets and the moments they correct come from one object.

One fit serves both axes. [`ucs`](@ref) takes the weighted factor span once and returns a [`NormBallUncertaintySet`](@ref) on the mean axis, whose geometry map spans the Orthogonal Subspace, and a [`CompactCovarianceUncertaintySet`](@ref) on the covariance axis, whose basis is the weighted factor span. The two sets spare the same portfolios, and `# Mathematical definition` states which.

The point estimates stay as the prior states them. The mean set is centred on `pr.mu` and the covariance set carries `pr.sigma`, so the prior is not shrunk. The correction is a worst case that grows with the exposure of the portfolio to the Orthogonal Subspace, and it counters an optimiser that over-allocates to the directions the factors do not span.

# Mathematical definition

```math
\\begin{align}
\\mathbf{Q}\\mathbf{Q}^{\\intercal} &= \\mathbf{W}^{1/2}\\mathbf{B}\\left(\\mathbf{W}^{1/2}\\mathbf{B}\\right)^{+}\\,, \\\\
\\operatorname{col}(\\mathbf{G}) &= \\left\\{ \\boldsymbol{v} \\in \\mathbb{R}^{N} \\,\\vert\\, \\mathbf{B}^{\\intercal}\\mathbf{W}\\boldsymbol{v} = \\mathbf{0} \\right\\}\\,, \\\\
r &= N - r_{\\mathbf{B}}\\,, \\\\
\\mathbf{L} &= \\mathbf{G}\\mathbf{\\Lambda}^{1/2}\\,, \\\\
\\kappa_{\\boldsymbol{\\mu}} &= \\sqrt{F^{-1}_{\\chi^{2}_{r}}(1 - q)}\\,, \\\\
\\mathbf{C} &= \\mathbf{W}^{-1/2}\\,, \\\\
\\underset{\\boldsymbol{\\mu} \\in U_{\\boldsymbol{\\mu}}}{\\min} \\boldsymbol{w}^{\\intercal}\\boldsymbol{\\mu} &= \\boldsymbol{w}^{\\intercal}\\hat{\\boldsymbol{\\mu}} - \\kappa_{\\boldsymbol{\\mu}} \\lVert \\mathbf{L}^{\\intercal}\\boldsymbol{w} \\rVert_{2}\\,, \\\\
\\underset{\\mathbf{\\Sigma} \\in U_{\\mathbf{\\Sigma}}}{\\max} \\boldsymbol{w}^{\\intercal}\\mathbf{\\Sigma}\\boldsymbol{w} &= \\boldsymbol{w}^{\\intercal}\\hat{\\mathbf{\\Sigma}}\\boldsymbol{w} + \\kappa \\lVert (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal})\\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``: Effective loadings, ``N \\times K``, reduced to a full-rank basis when a Factor Family was re-based.
  - ``\\mathbf{W}``: Diagonal cross-sectional metric that the [`AbstractOrthogonalityMetric`](@ref) names, the identity under [`IdentityMetric`](@ref).
  - ``(\\cdot)^{+}``: Moore-Penrose pseudo-inverse.
  - $(math_dict[:Q_cpt])
  - ``r_{\\mathbf{B}}``: Numerical rank of ``\\mathbf{W}^{1/2}\\mathbf{B}``, the column count of ``\\mathbf{Q}``.
  - $(math_dict[:G_orth])
  - $(math_dict[:r_orth])
  - $(math_dict[:N])
  - $(math_dict[:Lambda_orth])
  - ``\\mathbf{L}``: Geometry map of the mean set, ``N \\times r``.
  - ``\\kappa_{\\boldsymbol{\\mu}}``: Radius of the mean set. The line above is the radius of [`ChiSqKUncertaintyAlgorithm`](@ref), the default. [`GeneralKUncertaintyAlgorithm`](@ref) gives ``\\sqrt{(1 - q)/q}``, and a number is the radius itself.
  - ``F^{-1}_{\\chi^{2}_{r}}``: Quantile function of the chi-squared distribution with ``r`` degrees of freedom.
  - ``q``: Significance level of the mean set.
  - $(math_dict[:C_cpt])
  - ``U_{\\boldsymbol{\\mu}}``, ``U_{\\mathbf{\\Sigma}}``: Mean set and covariance set.
  - $(math_dict[:w_port])
  - ``\\hat{\\boldsymbol{\\mu}}``: Mean vector of the prior result, the centre of the mean set.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:kappa_cpt])

Both penalties are zero on a portfolio in the column space of ``\\mathbf{W}\\mathbf{B}``, because ``\\mathbf{G}^{\\intercal}\\mathbf{W}\\mathbf{B} = \\mathbf{0}`` and ``\\mathbf{C}\\mathbf{W}\\mathbf{B} = \\mathbf{W}^{1/2}\\mathbf{B}``. Every other portfolio pays on both axes when both radii are positive and ``\\mathbf{\\Lambda}`` is positive definite. Under [`IdentityMetric`](@ref) that column space is the span of the loadings. Under any other metric it is a different space, and a portfolio along a column of ``\\mathbf{B}`` pays on both axes.

The rank of the mean set is ``r``, not ``N``, because a flat set is a confidence region of its own subspace. Loadings that span the whole cross-section leave ``r = 0``, a mean set of radius zero and a covariance basis of ``N`` columns, so neither axis carries a penalty.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrthogonalUncertaintySet(;
        q::Number = 0.05,
        method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
        scaling::AbstractOrthogonalScaling = IdentityScaling(),
        kappa::Num_CptRad = 1.0,
        metric::AbstractOrthogonalityMetric = InverseIdiosyncraticVarianceMetric()
    ) -> OrthogonalUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `0 < q < 1`.
  - If `kappa` is a number: `isfinite(kappa)` and `kappa >= 0`. A rule is checked where its number lands, by the constructor of [`CompactCovarianceUncertaintySet`](@ref).

# Examples

```jldoctest
julia> OrthogonalUncertaintySet()
OrthogonalUncertaintySet
        q ┼ Float64: 0.05
   method ┼ ChiSqKUncertaintyAlgorithm()
  scaling ┼ IdentityScaling()
    kappa ┼ Float64: 1.0
   metric ┴ InverseIdiosyncraticVarianceMetric()
```

# Related

  - [`AbstractPriorUncertaintySetEstimator`](@ref)
  - [`AbstractOrthogonalityMetric`](@ref)
  - [`AbstractOrthogonalScaling`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`k_norm_ball`](@ref)

# References

  - $(ref_dict[:palomar2025]) Chapter 14.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
  - $(ref_dict[:bentalnemirovski1998]) Section 3.
"""
@concrete struct OrthogonalUncertaintySet <: AbstractPriorUncertaintySetEstimator
    """
    $(field_dict[:q_bs])
    """
    q
    """
    $(field_dict[:method_ucs])
    """
    method
    """
    Scaling of the mean set inside the Orthogonal Subspace. It changes the shape of the set and not the subspace the set lives in.
    """
    scaling
    """
    Radius ``\\kappa \\geq 0`` of the covariance set, the multiplier of its quadratic penalty. A radius of `0` leaves the nominal variance. The field holds a number that the caller states, or a rule of [`AbstractCompactRadiusAlgorithm`](@ref) that computes one from the sample and the span. It is a plain field, so `"ucs.kappa"` is a lens path that a [`GridSearchCrossValidation`](@ref) or a [`RandomisedSearchCrossValidation`](@ref) grid ranges over, and a grid may hold rules beside numbers.
    """
    kappa
    """
    Cross-sectional weighting under which the factor span is taken. It fixes the geometry of both sets, because both read one span.
    """
    metric
    function OrthogonalUncertaintySet(q::Number, method::Num_UcSK,
                                      scaling::AbstractOrthogonalScaling, kappa::Num_CptRad,
                                      metric::AbstractOrthogonalityMetric)
        @argcheck(zero(q) < q < one(q), DomainError(q, "q must be in (0, 1)"))
        # A rule states no number yet, so its range is checked where the number lands, in
        # `CompactCovarianceUncertaintySet`'s own constructor.
        if isa(kappa, Number)
            @argcheck(isfinite(kappa) && kappa >= zero(kappa),
                      DomainError(kappa, "kappa must be finite and >= 0"))
        end
        return new{typeof(q), typeof(method), typeof(scaling), typeof(kappa),
                   typeof(metric)}(q, method, scaling, kappa, metric)
    end
end
function OrthogonalUncertaintySet(; q::Number = 0.05,
                                  method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
                                  scaling::AbstractOrthogonalScaling = IdentityScaling(),
                                  kappa::Num_CptRad = 1.0,
                                  metric::AbstractOrthogonalityMetric = InverseIdiosyncraticVarianceMetric())::OrthogonalUncertaintySet
    return OrthogonalUncertaintySet(q, method, scaling, kappa, metric)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Weighted factor span of the prior's loadings block, the geometry both sets are built from.

# Algorithm

 1. Refuse when `pr.rr` is `nothing`. The set reads the loadings off the prior result, and a prior that fitted no factor model carries none.
 2. Read `B`, the effective loadings `rr.L`. The field reads back as `rr.M` when no Factor Family was re-based, and a re-based model is already reduced to a full-rank basis.
 3. Count `nnf`, the rows of `B` that hold an entry that is not finite, and refuse when it is not zero. A singular value decomposition of such a row gives a LAPACK error, not a span. The `NaN` rows that a point-in-time Asset Panel writes outside the Investable Mask never reach this step, because the optimiser's builders pass the prior reduced to that mask, and the standalone verbs reduce to it first through [`investable_ucs_reduction`](@ref). A non-finite row here therefore belongs to an asset that the prior calls investable, whose `mu` and variance are finite and whose loadings are not, and the message says so.
 4. Read `w`, the cross-sectional weights, through [`orthogonality_weights`](@ref), and take `w_sqrt`, their element-wise square root, or keep `nothing`.
 5. Scale the rows of `B` by `w_sqrt`, giving `Bw`.
 6. Take `F`, the thin `LinearAlgebra.svd` of `Bw`, with the singular values `s`.
 7. Count `r`, the singular values above `maximum(size(Bw)) * eps * s[1]`. The tolerance reads the larger dimension, so it is wider than the default of `LinearAlgebra.rank`, which reads the smaller one.
 8. Return `rr`, `w_sqrt` and the first `r` left singular vectors of `F`.

Step 7 counts the rank after the re-basis that step 2 reads, because the selected universe can still leave the exposures numerically dependent, and a dependent direction that survives widens the span that the penalty spares.

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the optimisation is solving on.

# Validation

  - `!isnothing(pr.rr)`, else an `IsNothingError` naming the field and the estimator that returned no block.
  - `all(isfinite, rr.L)`, else an [`IsNonFiniteError`](@ref) counting the assets whose loadings are not finite. Every such asset is inside the Investable Mask of `pr`, because the verbs that call this fit reduce to the mask first.

# Returns

  - `rr::AbstractLoadingsRegressionResult`: The loadings block the span came from.
  - `w_sqrt::Option{<:VecNum}`: Element-wise square root of the metric, or `nothing` on [`IdentityMetric`](@ref).
  - `Q::MatNum`: Orthonormal basis of the weighted factor span, ``N \\times r_{\\mathbf{B}}``.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`orthogonality_weights`](@ref)
  - [`ucs`](@ref)
"""
function orthogonal_factor_span(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult)
    rr = pr.rr
    @argcheck(!isnothing(rr),
              IsNothingError("`$(nameof(typeof(ue)))` reads the factor loadings off `pr.rr`, and the prior it was handed carries none, so there is no factor span to take the orthogonal complement of.\nFit the optimisation on a prior that returns a loadings block, such as `FactorPrior` or `CrossSectionalFactorPrior`.\nGot\npr => $(nameof(typeof(pr)))\nrr => nothing"))
    B = rr.L
    nnf = count(i -> !all(isfinite, view(B, i, :)), axes(B, 1))
    @argcheck(iszero(nnf),
              IsNonFiniteError("`$(nameof(typeof(ue)))` takes the span of the factor loadings, and $(nnf) of the $(size(B, 1)) assets of `pr.rr` carry a loading that is not finite, so the span is not defined over them. The fit reduces the prior to its Investable Mask first, so every such asset has a finite `mu` and variance and a loadings row that is not: the prior estimator wrote a moment for an asset it fitted no loadings on.\nCheck the loadings block of the prior, `pr.rr.L`, on those assets.\nGot\npr => $(nameof(typeof(pr)))\nrr => $(nameof(typeof(rr)))\nassets with a non-finite loading => $(nnf)"))
    w = orthogonality_weights(ue.metric, rr)
    w_sqrt = isnothing(w) ? nothing : sqrt.(w)
    Bw = isnothing(w_sqrt) ? B : w_sqrt .* B
    F = LinearAlgebra.svd(Bw)
    s = F.S
    # The tolerance reads `s[1]` inside the predicate rather than above the count, so a
    # block with no factor column needs no branch of its own: `count` over an empty vector
    # never calls the predicate, and answers a rank of zero. The machine epsilon is read off
    # the singular values, which are floating point even when the loadings are integers.
    r = count(x -> x > maximum(size(Bw)) * eps(eltype(s)) * s[1], s)
    return rr, w_sqrt, F.U[:, 1:r]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the mean [`NormBallUncertaintySet`](@ref) on the Orthogonal Subspace, from a span already taken.

# Algorithm

 1. Form `P`, the orthogonal projector `I - Q * Q'`.
 2. Form `A`, the rows of `P` divided by `w_sqrt`, or `P` itself when `w_sqrt` is `nothing`. `A` is the projector read back in the asset coordinates.
 3. Take `E`, the symmetric eigendecomposition of `A' * A`.
 4. Select `keep`, the indices of the trailing `N - size(Q, 2)` eigenvectors of `E`. The count is exact: `Q` is orthonormal, so `P` has rank `N - size(Q, 2)`, the metric scaling is an invertible diagonal, and `LinearAlgebra.eigen` on a `Symmetric` orders the eigenvalues from small to large. An eigenvalue tolerance can sit so close to the eigenvalue it must cut that the cut changes with the reduction order of the machine and states a subspace one dimension too wide. Step 7 of [`orthogonal_factor_span`](@ref) still reads a tolerance, because the rank of the loadings is a property of the data and not of a projector.
 5. Orthonormalise `A` times the selected eigenvectors with a reduced `LinearAlgebra.qr`, giving `G`, and read `r`, the dimension of the Orthogonal Subspace, as its column count. An empty `keep` gives a `G` with no column.
 6. When `r` is `0`, return the set with a radius of zero, a map of one zero column, and `pr.mu` as its centre. The zero radius leaves the nominal mean.
 7. Take `lambda`, the scaling, through [`orthogonal_scaling`](@ref).
 8. Form `L`, the product of `G` and the symmetric square root of `lambda`.
 9. Size the radius with [`k_norm_ball`](@ref) at `r` degrees of freedom, and return the set of order `2` with `pr.mu` as its centre. A set fitted on one prior and passed to another optimisation therefore carries the centre that its geometry was calibrated on.

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the span came from.
  - `rr`: Loadings block the span came from.
  - `w_sqrt`: Element-wise square root of the metric, or `nothing`.
  - `Q`: Orthonormal basis of the weighted factor span.

# Returns

  - `ucs::NormBallUncertaintySet`: Mean set on the Orthogonal Subspace, of order `2`.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`orthogonal_factor_span`](@ref)
  - [`orthogonal_scaling`](@ref)
  - [`k_norm_ball`](@ref)
"""
function orthogonal_mu_set(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult,
                           rr::AbstractLoadingsRegressionResult, w_sqrt::Option{<:VecNum},
                           Q::MatNum)
    N = size(Q, 1)
    P = LinearAlgebra.I - Q * transpose(Q)
    A = isnothing(w_sqrt) ? Matrix(P) : Matrix(P) ./ w_sqrt
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(transpose(A) * A))
    # The rank is counted, not cut at a tolerance. `I - Q * Q'` has rank `N - size(Q, 2)`
    # because `Q` is orthonormal, and the metric scaling is an invertible diagonal, so
    # `A' * A` has exactly that many non-zero eigenvalues. `eigen` on a `Symmetric` returns
    # them in ascending order, so the non-zero block is the trailing one.
    keep = (size(Q, 2) + 1):N
    G = if isempty(keep)
        Matrix{eltype(A)}(undef, N, 0)
    else
        Matrix(LinearAlgebra.qr(A * view(E.vectors, :, keep)).Q)
    end
    r = size(G, 2)
    if iszero(r)
        return NormBallUncertaintySet(; kappa = zero(eltype(A)), L = zeros(eltype(A), N, 1),
                                      p = 2, class = MuUncertaintySetClass(), val = pr.mu)
    end
    lambda = orthogonal_scaling(ue.scaling, G, rr)
    L = G * sqrt(LinearAlgebra.Symmetric(lambda))
    return NormBallUncertaintySet(;
                                  kappa = k_norm_ball(ue.method, ue.q, nothing, lambda, r),
                                  L = L, p = 2, class = MuUncertaintySetClass(),
                                  val = pr.mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the covariance [`CompactCovarianceUncertaintySet`](@ref) on the Orthogonal Subspace, from a span already taken.

# Algorithm

 1. Form `C`, the diagonal metric square root, as the element-wise inverse of `w_sqrt`, or as a vector of ones when `w_sqrt` is `nothing`.
 2. Settle `kappa`, the radius, with [`k_compact`](@ref). A stated number passes through unchanged. A rule of [`AbstractCompactRadiusAlgorithm`](@ref) receives the confidence level, the metric, the prior result, the loadings block, `C`, `Q` and `rd`.
 3. Return the set with `Q` as the basis it spares and `pr.sigma` as the nominal covariance. A span of rank zero gives a basis with no column, which the type admits, and the penalty then reaches every direction.

The set penalises the complement of the span, the same subspace that the mean set lives in, so one decomposition serves both axes and the estimator computes it once.

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the span came from.
  - `rr`: Loadings block the span came from.
  - `w_sqrt`: Element-wise square root of the metric, or `nothing`.
  - `Q`: Orthonormal basis of the weighted factor span.
  - `rd`: Returns data the set was fitted beside, or `nothing`. Only a [`VarianceFraction`](@ref) holding an optimiser reads it.

# Returns

  - `ucs::CompactCovarianceUncertaintySet`: Covariance set that spares the weighted factor span.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`orthogonal_factor_span`](@ref)
  - [`k_compact`](@ref)
"""
function orthogonal_sigma_set(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult,
                              rr::AbstractLoadingsRegressionResult,
                              w_sqrt::Option{<:VecNum}, Q::MatNum, rd = nothing)
    C = isnothing(w_sqrt) ? ones(eltype(Q), size(Q, 1)) : inv.(w_sqrt)
    kappa = k_compact(ue.kappa, ue.q, ue.metric, pr, rr, C, Q, rd)
    return CompactCovarianceUncertaintySet(; kappa = kappa, C = C, Q = Q, val = pr.sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits both uncertainty sets of an [`OrthogonalUncertaintySet`](@ref) from the prior result in one pass.

# Algorithm

 1. Reduce the prior result, and the returns data beside it, to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser the result arrives reduced and the step passes it through. Standalone, on a prior fitted on a point-in-time Asset Panel, the step takes the view that the optimiser takes.
 2. Take the weighted factor span once with [`orthogonal_factor_span`](@ref).
 3. Build the mean set from that span with [`orthogonal_mu_set`](@ref).
 4. Build the covariance set from the same span with [`orthogonal_sigma_set`](@ref).
 5. Write each set back onto the full universe with [`expand_investable_ucs`](@ref). A set fitted standalone then covers the assets of the prior, with a zero row on every asset outside the mask, and a view of it at the mask recovers the reduced fit.

A caller that needs one axis alone calls [`mu_ucs`](@ref) or [`sigma_ucs`](@ref), which take the same span and build one set.

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the optimisation is solving on.
  - `rd`: Returns data the set is being fitted beside, or `nothing`. Only a [`VarianceFraction`](@ref) holding an optimiser reads it, and the three-argument forms of [`ucs`](@ref) fill it in.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Mean set on the Orthogonal Subspace.
  - `sigma_ucs::CompactCovarianceUncertaintySet`: Covariance set that spares the factor span.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`orthogonal_factor_span`](@ref)
"""
function ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; rd = nothing, kwargs...)
    imsk, prr, rdr = investable_ucs_reduction(pr, rd)
    rr, w_sqrt, Q = orthogonal_factor_span(ue, prr)
    return expand_investable_ucs(orthogonal_mu_set(ue, prr, rr, w_sqrt, Q), imsk, pr),
           expand_investable_ucs(orthogonal_sigma_set(ue, prr, rr, w_sqrt, Q, rdr), imsk,
                                 pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits the mean uncertainty set of an [`OrthogonalUncertaintySet`](@ref) from the prior result.

# Algorithm

 1. Reduce the prior result to the Investable Mask with [`investable_ucs_reduction`](@ref), a passthrough on a result that arrived reduced.
 2. Take the weighted factor span with [`orthogonal_factor_span`](@ref).
 3. Build the mean set with [`orthogonal_mu_set`](@ref).
 4. Write it back onto the full universe with [`expand_investable_ucs`](@ref).

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the optimisation is solving on.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `ucs::NormBallUncertaintySet`: Mean set on the Orthogonal Subspace.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`orthogonal_mu_set`](@ref)
"""
function mu_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, nothing)
    rr, w_sqrt, Q = orthogonal_factor_span(ue, prr)
    return expand_investable_ucs(orthogonal_mu_set(ue, prr, rr, w_sqrt, Q), imsk, pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits the covariance uncertainty set of an [`OrthogonalUncertaintySet`](@ref) from the prior result.

# Algorithm

 1. Reduce the prior result, and the returns data beside it, to the Investable Mask with [`investable_ucs_reduction`](@ref), a passthrough on a result that arrived reduced.
 2. Take the weighted factor span with [`orthogonal_factor_span`](@ref).
 3. Build the covariance set with [`orthogonal_sigma_set`](@ref).
 4. Write it back onto the full universe with [`expand_investable_ucs`](@ref).

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the optimisation is solving on.
  - `rd`: Returns data the set is being fitted beside, or `nothing`. Only a [`VarianceFraction`](@ref) holding an optimiser reads it, and the three-argument forms of [`sigma_ucs`](@ref) fill it in.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `ucs::CompactCovarianceUncertaintySet`: Covariance set that spares the factor span.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`orthogonal_sigma_set`](@ref)
"""
function sigma_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; rd = nothing,
                   kwargs...)
    imsk, prr, rdr = investable_ucs_reduction(pr, rd)
    rr, w_sqrt, Q = orthogonal_factor_span(ue, prr)
    return expand_investable_ucs(orthogonal_sigma_set(ue, prr, rr, w_sqrt, Q, rdr), imsk,
                                 pr)
end

"""
    cs_diagnostic_weights(weighting::IdentityMetric, csfm::CrossSectionalFactorModel)
    cs_diagnostic_weights(weighting::BenchmarkWeightMetric, csfm::CrossSectionalFactorModel)
    cs_diagnostic_weights(weighting::RegressionWeightMetric, csfm::CrossSectionalFactorModel)
    cs_diagnostic_weights(weighting::InverseIdiosyncraticVarianceMetric, csfm::CrossSectionalFactorModel)

Return the cross-sectional weight history an [`AbstractOrthogonalityMetric`](@ref) names, over the whole observation axis of a factor model block.

[`orthogonality_weights`](@ref) reads the same metrics off the same block and returns the weights of the latest observation, which an uncertainty set needs. A diagnostic scores every observation, so it reads the whole history through this function instead.

# Algorithm

 1. On [`IdentityMetric`](@ref), return `nothing`.
 2. On [`BenchmarkWeightMetric`](@ref), return the history `csfm.bw` through [`cs_diagnostic_weight_history`](@ref).
 3. On [`RegressionWeightMetric`](@ref), return the history `csfm.rw` through [`cs_diagnostic_weight_history`](@ref).
 4. On [`InverseIdiosyncraticVarianceMetric`](@ref), read `vs`, the idiosyncratic variance history `csfm.vs`, through [`cs_diagnostic_weight_history`](@ref), and return its element-wise inverse.

# Arguments

  - `weighting`: The metric that names the history.
  - `csfm`: A cross-sectional factor model block.

# Validation

  - The field that the metric names is not `nothing`, else an `IsNothingError` that names the field.

# Returns

  - `u::Option{<:MatNum}`: `observations × assets`, and `nothing` under [`IdentityMetric`](@ref), which the diagnostics read as equal weights.

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`orthogonality_weights`](@ref)
  - [`exposure_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cs_diagnostic_weights(::IdentityMetric, ::CrossSectionalFactorModel)
    return nothing
end
function cs_diagnostic_weights(::BenchmarkWeightMetric, csfm::CrossSectionalFactorModel)
    return cs_diagnostic_weight_history(csfm.bw, "bw", "the benchmark weight history")
end
function cs_diagnostic_weights(::RegressionWeightMetric, csfm::CrossSectionalFactorModel)
    return cs_diagnostic_weight_history(csfm.rw, "rw", "the regression weight history")
end
function cs_diagnostic_weights(::InverseIdiosyncraticVarianceMetric,
                               csfm::CrossSectionalFactorModel)
    vs = cs_diagnostic_weight_history(csfm.vs, "vs", "the idiosyncratic variance history")
    return one(real(eltype(vs))) ./ vs
end
"""
    cs_diagnostic_weight_history(A::Nothing, name::AbstractString, what::AbstractString)
    cs_diagnostic_weight_history(A::MatNum, name::AbstractString, what::AbstractString)

Return a weight history of a factor model block, or refuse an absent one by name.

# Arguments

  - `A`: The field of the block the metric names, or `nothing`.
  - `name`: Name of that field, which the refusal states.
  - `what`: What the field holds, which the refusal states.

# Returns

  - `A::MatNum`: The history.

# Related

  - [`cs_diagnostic_weights`](@ref)
"""
function cs_diagnostic_weight_history(::Nothing, name::AbstractString, what::AbstractString)
    return throw(IsNothingError("$name cannot be nothing: an exposure diagnostic reads $what of the block"))
end
function cs_diagnostic_weight_history(A::MatNum, ::AbstractString, ::AbstractString)
    return A
end

export IdentityScaling, IdiosyncraticVarianceScaling, OrthogonalUncertaintySet
public AbstractOrthogonalScaling, orthogonal_scaling
