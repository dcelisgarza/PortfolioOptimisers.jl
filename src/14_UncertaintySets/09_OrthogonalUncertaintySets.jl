"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the cross-sectional weightings that state the metric under which the factor span is taken.

A member of this family **selects** a weight vector the fitted loadings block already carries, and computes none. It is the marker-family form the library uses for a fixed choice among stored quantities, as `class` does on [`NormBallUncertaintySet`](@ref). Its sibling [`AbstractCrossSectionalWeightsAlgorithm`](@ref) is the other side of the pair: that family computes the regression weights inside the prior's own fit, and this one reads the answer back.

The weighting fixes the geometry of the Orthogonal Subspace. The span of the loadings is taken under the inner product ``\\langle \\boldsymbol{x}, \\boldsymbol{y} \\rangle_{\\mathbf{W}} = \\boldsymbol{x}^{\\intercal}\\mathbf{W}\\boldsymbol{y}``, so two weightings give two different orthogonal complements of the same loadings.

# Interface

## `orthogonality_weights`

  - `orthogonality_weights(metric::AbstractOrthogonalityMetric, rr::AbstractLoadingsRegressionResult) -> Option{<:VecNum}`: Returns the weight vector, one entry per asset, or `nothing` when the metric is the unweighted one.

# Related

  - [`BenchmarkWeightMetric`](@ref)
  - [`RegressionWeightMetric`](@ref)
  - [`InverseIdiosyncraticVarianceMetric`](@ref)
  - [`IdentityMetric`](@ref)
  - [`orthogonality_weights`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
"""
abstract type AbstractOrthogonalityMetric <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Weights the cross-section by the benchmark weights of the latest observation.

The span is taken as the benchmark portfolio sees it, so a large benchmark holding pulls the factor directions towards itself and its own direction pays no penalty. The weights come from the `bw` history of the loadings block, and a block that carries none refuses.

# Examples

```jldoctest
julia> BenchmarkWeightMetric()
BenchmarkWeightMetric()
```

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`RegressionWeightMetric`](@ref)
  - [`orthogonality_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
struct BenchmarkWeightMetric <: AbstractOrthogonalityMetric end
"""
$(DocStringExtensions.TYPEDEF)

Weights the cross-section by the regression weights of the latest observation.

The span is taken as the Cross-Sectional Regression itself took it, so the geometry of the uncertainty set matches the geometry the loadings were fitted under. The weights come from the `rw` history of the loadings block, and a block that carries none refuses.

# Examples

```jldoctest
julia> RegressionWeightMetric()
RegressionWeightMetric()
```

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`BenchmarkWeightMetric`](@ref)
  - [`orthogonality_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
struct RegressionWeightMetric <: AbstractOrthogonalityMetric end
"""
$(DocStringExtensions.TYPEDEF)

Weights the cross-section by the inverse of the idiosyncratic variances, the default.

An asset whose returns the factors explain well carries a large weight, so the span leans on the assets the model fits, and the directions the penalty spares are the ones the model has measured. The variances come from `idiosyncratic_variances` on the loadings block, which reads `esigma`, and a non-positive variance refuses.

# Examples

```jldoctest
julia> InverseIdiosyncraticVarianceMetric()
InverseIdiosyncraticVarianceMetric()
```

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`IdentityMetric`](@ref)
  - [`orthogonality_weights`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
struct InverseIdiosyncraticVarianceMetric <: AbstractOrthogonalityMetric end
"""
$(DocStringExtensions.TYPEDEF)

Takes the factor span under the plain Euclidean inner product.

Every asset carries the same weight, so the metric matrix is the identity and the compact set's metric square root is a vector of ones. It is the one member that reads nothing off the loadings block, so it serves a block that carries neither a weight history nor an idiosyncratic variance.

# Examples

```jldoctest
julia> IdentityMetric()
IdentityMetric()
```

# Related

  - [`AbstractOrthogonalityMetric`](@ref)
  - [`InverseIdiosyncraticVarianceMetric`](@ref)
  - [`orthogonality_weights`](@ref)
"""
struct IdentityMetric <: AbstractOrthogonalityMetric end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the scalings that size each direction inside the Orthogonal Subspace.

A member states the ``r \\times r`` matrix ``\\mathbf{\\Lambda}`` of the mean set, whose square root maps the ball's coordinates onto the subspace basis. The scaling changes the shape of the set and not its support: every member confines the set to the same subspace, and the radius reads the same `rank` degrees of freedom.

# Interface

## `orthogonal_scaling`

  - `orthogonal_scaling(scaling::AbstractOrthogonalScaling, G::MatNum, rr::AbstractLoadingsRegressionResult) -> MatNum`: Returns ``\\mathbf{\\Lambda}``, `rank × rank`, symmetric and positive semi-definite.

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

``\\mathbf{\\Lambda} = \\mathbf{I}_{r}``, so the geometry map is the subspace basis itself and the set is a Euclidean ball inside the subspace.

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

``\\mathbf{\\Lambda} = \\mathbf{G}^{\\intercal}\\mathbf{D}\\mathbf{G}``, with ``\\mathbf{D}`` the idiosyncratic covariance the loadings block carries. A direction the factors leave noisy is then given more uncertainty than a quiet one, where [`IdentityScaling`](@ref) gives both the same. The block must carry an idiosyncratic covariance, so an unset `esigma` refuses.

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

 1. On [`IdentityMetric`](@ref), return `nothing`. The caller reads a `nothing` as a vector of ones and skips both the scaling of the loadings and the division of the projector, so the unweighted route costs no arithmetic.
 2. On [`InverseIdiosyncraticVarianceMetric`](@ref), take `idiosyncratic_variances(rr)` and return its element-wise inverse.
 3. On [`BenchmarkWeightMetric`](@ref) and [`RegressionWeightMetric`](@ref) over a [`CrossSectionalFactorModel`](@ref), take the last row of `bw` or of `rw`, the weights of the latest observation.
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

 1. On a `nothing` history, throw. The block declares the field and this fit left it unset, which the message says.
 2. On a history, take its last row, the weights of the latest observation. The uncertainty set is built for the next decision, so it reads the newest cross-section and not an average of the sample.

# Arguments

  - `w`: Weight history, `observations × assets`, or `nothing`.
  - `name`: Name of the field the history came from, which the refusals quote.
  - `rr`: Fitted loadings block, quoted by the refusals.

# Validation

  - `!isempty(w)`, else an `IsEmptyError`.
  - Every entry of the last row is finite and `> 0`, else a `DomainError`. A weight of zero excluded its asset from the fit, and an excluded asset gives the metric a singular direction.

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
                          "every weight of the latest observation of $(name) must be finite and > 0; a weight of 0 excluded its asset from the fit and leaves the metric singular"))
    return wl
end
"""
    orthogonal_scaling(::IdentityScaling, G::MatNum, ::AbstractLoadingsRegressionResult)
    orthogonal_scaling(::IdiosyncraticVarianceScaling, G::MatNum, rr::AbstractLoadingsRegressionResult)

Scaling ``\\\\mathbf{\\\\Lambda}`` of the mean set inside the Orthogonal Subspace.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Lambda}_{\\mathrm{id}} &= \\mathbf{I}_{r}\\,, \\\\
\\mathbf{\\Lambda}_{\\mathrm{idio}} &= \\mathbf{G}^{\\intercal}\\mathbf{D}\\mathbf{G}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{G}``: Orthonormal basis of the Orthogonal Subspace, ``N \\times r``.
  - ``\\mathbf{D}``: Idiosyncratic covariance the loadings block carries, ``N \\times N``, diagonal when the block stores a variance vector.
  - ``r``: Dimension of the subspace.

# Algorithm

 1. On [`IdentityScaling`](@ref), return the ``r \\times r`` identity as a dense matrix, so the caller's square root and the radius read one type on both routes.
 2. On [`IdiosyncraticVarianceScaling`](@ref), read `rr.esigma`, form ``\\mathbf{G}^{\\intercal}\\mathbf{D}\\mathbf{G}`` and symmetrise it. A stored variance vector is used as a diagonal without materialising the ``N \\times N`` matrix.

# Arguments

  - `scaling`: Orthogonal scaling.
  - `G`: Orthonormal basis of the Orthogonal Subspace, ``N \\times r``.
  - `rr`: Fitted loadings block.

# Validation

  - On [`IdiosyncraticVarianceScaling`](@ref): [`idiosyncratic_variances`](@ref) throws when the block carries no `esigma`.

# Returns

  - `lambda::MatNum`: ``r \\times r`` scaling, symmetric and positive semi-definite.

# Related

  - [`AbstractOrthogonalScaling`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
function orthogonal_scaling(::IdentityScaling, G::MatNum,
                            ::AbstractLoadingsRegressionResult)
    return Matrix{eltype(G)}(LinearAlgebra.I, size(G, 2), size(G, 2))
end
function orthogonal_scaling(::IdiosyncraticVarianceScaling, G::MatNum,
                            rr::AbstractLoadingsRegressionResult)
    D = LinearAlgebra.Diagonal(idiosyncratic_variances(rr))
    lambda = transpose(G) * D * G
    return (lambda + transpose(lambda)) / 2
end
"""
    k_norm_ball(::NormalKUncertaintyAlgorithm, ::Number, ::Nothing, ::MatNum, ::Integer)

Always throw. The Orthogonal Subspace fit simulates no estimation errors, so the sampled radius has nothing to read.

The method is a refusal rather than a procedure, so it carries no `# Algorithm` section. [`NormalKUncertaintyAlgorithm`](@ref) reads a sample of Mahalanobis distances and takes their quantile, and [`OrthogonalUncertaintySet`](@ref) draws none: it derives its geometry from the fitted loadings alone. The refusal shadows the `MethodError` the missing sample would otherwise raise.

# Arguments

  - The radius algorithm, the significance level, the absent sample, the scaling and the degrees of freedom, all ignored.

# Validation

  - The method always throws an `ArgumentError`. The message names the two algorithms that read no sample, [`ChiSqKUncertaintyAlgorithm`](@ref) and [`GeneralKUncertaintyAlgorithm`](@ref), and the plain number that states the radius outright.

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

The estimator reads the loadings block `rr` of the Prior Result it is handed, and never fits a prior of its own. It is the one member of [`AbstractPriorUncertaintySetEstimator`](@ref), and the two JuMP builders pass the reduced prior beside the returns so that the sets and the moments they correct are fitted on one object.

**One estimator answers both axes.** The weighted loadings, their thin singular value decomposition and the numerical rank are computed once, and [`ucs`](@ref) returns the pair. The mean axis gets a [`NormBallUncertaintySet`](@ref) whose geometry map spans the Orthogonal Subspace, and the covariance axis gets a [`CompactCovarianceUncertaintySet`](@ref) whose basis is the weighted factor span itself. The two are complementary: a portfolio inside the span pays nothing on either axis, and one outside it pays on both.

**The point estimates are unchanged.** The mean set is centred on `pr.mu` and the covariance set carries `pr.sigma`, so nothing is shrunk in the prior. The correction is a portfolio-dependent worst case that grows with the exposure to the subspace, which is what an optimiser that over-allocates to unspanned directions needs.

# Mathematical definition

```math
\\begin{align}
\\mathbf{B}_{\\mathbf{W}} &= \\mathbf{W}^{1/2}\\mathbf{B}\\,, \\quad \\mathbf{Q} = \\operatorname{svd}_{r_{\\mathbf{B}}}(\\mathbf{B}_{\\mathbf{W}})\\,, \\\\
\\mathbf{A} &= \\mathbf{W}^{-1/2}\\left(\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}\\right)\\,, \\quad \\mathbf{G} = \\operatorname{qr}(\\mathbf{A}\\mathbf{V}_{+})\\,, \\\\
\\mathbf{L} &= \\mathbf{G}\\mathbf{\\Lambda}^{1/2}\\,, \\quad \\kappa_{\\boldsymbol{\\mu}} = \\sqrt{\\chi^{2,\\,-1}_{r}(1 - q)}\\,, \\\\
\\mathbf{C} &= \\mathbf{W}^{-1/2}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``: Effective loadings, ``N \\times K``, reduced to a full-rank basis when a Factor Family was re-based.
  - ``\\mathbf{W} = \\operatorname{diag}(\\boldsymbol{w})``: Cross-sectional metric the [`AbstractOrthogonalityMetric`](@ref) names, the identity on [`IdentityMetric`](@ref).
  - ``\\mathbf{Q}``: Left singular vectors of the weighted loadings kept by the numerical rank ``r_{\\mathbf{B}}``, an orthonormal basis of the weighted factor span.
  - ``\\mathbf{A}``: Orthogonal projector, mapped back through the metric.
  - ``\\mathbf{V}_{+}``: Eigenvectors of ``\\mathbf{A}^{\\intercal}\\mathbf{A}`` whose eigenvalue clears the tolerance.
  - ``\\mathbf{G}``: Orthonormal basis of the Orthogonal Subspace, ``N \\times r``.
  - ``\\mathbf{\\Lambda}``: Scaling the [`AbstractOrthogonalScaling`](@ref) names, ``r \\times r``.
  - ``\\mathbf{L}``: Geometry map of the mean set.
  - ``\\kappa_{\\boldsymbol{\\mu}}``: Radius of the mean set, at ``r`` degrees of freedom.
  - ``\\mathbf{C}``: Diagonal metric square root of the covariance set.

The rank of the mean set is the dimension of the Orthogonal Subspace, not the number of assets, because a flat set is a confidence region of its own subspace and not of the ambient space. A model whose loadings span the whole cross-section leaves ``r = 0``, a radius of zero, and no correction at all.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrthogonalUncertaintySet(;
        q::Number = 0.05,
        method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
        scaling::AbstractOrthogonalScaling = IdentityScaling(),
        kappa::Number = 1.0,
        metric::AbstractOrthogonalityMetric = InverseIdiosyncraticVarianceMetric()
    ) -> OrthogonalUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `0 < q < 1`.
  - `isfinite(kappa)` and `kappa >= 0`.

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
    Radius ``\\kappa \\geq 0`` of the covariance set, the multiplier of its quadratic penalty. It is a size the caller states rather than a quantile, and `0` leaves the nominal variance.
    """
    kappa
    """
    Cross-sectional weighting under which the factor span is taken. It fixes the geometry of both sets, because both read one span.
    """
    metric
    function OrthogonalUncertaintySet(q::Number, method::Num_UcSK,
                                      scaling::AbstractOrthogonalScaling, kappa::Number,
                                      metric::AbstractOrthogonalityMetric)
        @argcheck(zero(q) < q < one(q), DomainError(q, "q must be in (0, 1)"))
        @argcheck(isfinite(kappa) && kappa >= zero(kappa),
                  DomainError(kappa, "kappa must be finite and >= 0"))
        return new{typeof(q), typeof(method), typeof(scaling), typeof(kappa),
                   typeof(metric)}(q, method, scaling, kappa, metric)
    end
end
function OrthogonalUncertaintySet(; q::Number = 0.05,
                                  method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
                                  scaling::AbstractOrthogonalScaling = IdentityScaling(),
                                  kappa::Number = 1.0,
                                  metric::AbstractOrthogonalityMetric = InverseIdiosyncraticVarianceMetric())::OrthogonalUncertaintySet
    return OrthogonalUncertaintySet(q, method, scaling, kappa, metric)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Weighted factor span of the prior's loadings block, the geometry both sets are built from.

# Algorithm

 1. Refuse when `pr.rr` is `nothing`. The set reads the loadings off the prior result, and a prior that fitted no factor model carries none.
 2. Read the effective loadings `rr.L`, which reads back as `rr.M` when no Factor Family was re-based, so a re-based model is already reduced to a full-rank basis here.
 3. Read the cross-sectional weights through [`orthogonality_weights`](@ref) and take their element-wise square root, or leave a `nothing`.
 4. Scale the rows of the loadings by that square root, take a thin `LinearAlgebra.svd`, and count the singular values above `maximum(size) * eps * s[1]`, the tolerance `LinearAlgebra.rank` applies. Keep that many left singular vectors.

Step 4 counts the rank after the family re-basis of step 2, because the selected universe can still leave the exposures numerically dependent, and a dependent direction that survives would widen the span the penalty spares.

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the optimisation is solving on.

# Validation

  - `!isnothing(pr.rr)`, else an `IsNothingError` naming the field and the estimator that returned no block.

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
    w = orthogonality_weights(ue.metric, rr)
    w_sqrt = isnothing(w) ? nothing : sqrt.(w)
    Bw = isnothing(w_sqrt) ? B : w_sqrt .* B
    F = LinearAlgebra.svd(Bw)
    s = F.S
    # The tolerance reads `s[1]` inside the predicate rather than above the count, so a
    # block with no factor column needs no branch of its own: `count` over an empty vector
    # never calls the predicate, and answers a rank of zero.
    r = count(x -> x > maximum(size(Bw)) * eps(float(real(eltype(Bw)))) * s[1], s)
    return rr, w_sqrt, F.U[:, 1:r]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the mean [`NormBallUncertaintySet`](@ref) on the Orthogonal Subspace, from a span already taken.

# Algorithm

 1. Form the orthogonal projector `I - Q * Q'` and divide its rows by the metric square root, giving ``\\mathbf{A}``, the projector read back in the asset coordinates.
 2. Take the symmetric eigendecomposition of `A' * A` and keep the eigenvectors whose eigenvalue clears `max(N * eps, N * maximum(abs, eigenvalues) * eps)`. The two tolerances are the reference implementation's, and the absolute one is what admits a subspace whose eigenvalues are all small.
 3. Orthonormalise `A * V₊` with a reduced `LinearAlgebra.qr`, giving `G`, and read the dimension `r` of the Orthogonal Subspace off its columns.
 4. When `r` is `0`, return the set with a radius of zero and a map of one zero column. The map keeps a column because the type admits a rank-zero map and a consumer that reads a size finds one either way, and the zero radius leaves the nominal mean.
 5. Otherwise take the scaling ``\\mathbf{\\Lambda}`` through [`orthogonal_scaling`](@ref), form `L = G * sqrt(Λ)` with a symmetric square root, and size the radius with [`k_norm_ball`](@ref) at `r` degrees of freedom.
 6. Carry `pr.mu` into `val`, so a set fitted on one prior and handed to another optimisation carries the centre its geometry was calibrated on.

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
    ev = E.values
    tol = max(N * eps(float(real(eltype(A)))),
              N * maximum(abs, ev) * eps(float(real(eltype(A)))))
    keep = findall(x -> x > tol, ev)
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

 1. Take the element-wise inverse of the metric square root as the diagonal metric ``\\mathbf{C}``, or a vector of ones on [`IdentityMetric`](@ref).
 2. Hand the weighted factor span `Q` to the set as the basis it spares. A rank of zero leaves a basis with no column, which the type admits and which leaves the penalty on every direction.
 3. Carry `ue.kappa` as the radius and `pr.sigma` as the nominal covariance.

The set spares the span and penalises its complement, which is the same subspace the mean set lives in. The two axes are therefore built from one decomposition, and the estimator computes it once.

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the span came from.
  - `w_sqrt`: Element-wise square root of the metric, or `nothing`.
  - `Q`: Orthonormal basis of the weighted factor span.

# Returns

  - `ucs::CompactCovarianceUncertaintySet`: Covariance set that spares the weighted factor span.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`orthogonal_factor_span`](@ref)
"""
function orthogonal_sigma_set(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult,
                              w_sqrt::Option{<:VecNum}, Q::MatNum)
    C = isnothing(w_sqrt) ? ones(eltype(Q), size(Q, 1)) : inv.(w_sqrt)
    return CompactCovarianceUncertaintySet(; kappa = ue.kappa, C = C, Q = Q, val = pr.sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits both uncertainty sets of an [`OrthogonalUncertaintySet`](@ref) from the prior result in one pass.

# Algorithm

 1. Take the weighted factor span once with [`orthogonal_factor_span`](@ref).
 2. Build the mean set with [`orthogonal_mu_set`](@ref) and the covariance set with [`orthogonal_sigma_set`](@ref), both from that span.

A caller that needs one axis alone calls [`mu_ucs`](@ref) or [`sigma_ucs`](@ref), which take the same span and build one set.

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the optimisation is solving on.
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
function ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
    rr, w_sqrt, Q = orthogonal_factor_span(ue, pr)
    return orthogonal_mu_set(ue, pr, rr, w_sqrt, Q), orthogonal_sigma_set(ue, pr, w_sqrt,
                                                                          Q)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits the mean uncertainty set of an [`OrthogonalUncertaintySet`](@ref) from the prior result.

# Algorithm

 1. Take the weighted factor span with [`orthogonal_factor_span`](@ref).
 2. Build the mean set with [`orthogonal_mu_set`](@ref).

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
    rr, w_sqrt, Q = orthogonal_factor_span(ue, pr)
    return orthogonal_mu_set(ue, pr, rr, w_sqrt, Q)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits the covariance uncertainty set of an [`OrthogonalUncertaintySet`](@ref) from the prior result.

# Algorithm

 1. Take the weighted factor span with [`orthogonal_factor_span`](@ref).
 2. Build the covariance set with [`orthogonal_sigma_set`](@ref).

# Arguments

  - `ue`: Orthogonal uncertainty set estimator.
  - `pr`: Prior result the optimisation is solving on.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `ucs::CompactCovarianceUncertaintySet`: Covariance set that spares the factor span.

# Related

  - [`OrthogonalUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`orthogonal_sigma_set`](@ref)
"""
function sigma_ucs(ue::OrthogonalUncertaintySet, pr::AbstractPriorResult; kwargs...)
    _, w_sqrt, Q = orthogonal_factor_span(ue, pr)
    return orthogonal_sigma_set(ue, pr, w_sqrt, Q)
end

export BenchmarkWeightMetric, RegressionWeightMetric, InverseIdiosyncraticVarianceMetric,
       IdentityMetric, IdentityScaling, IdiosyncraticVarianceScaling,
       OrthogonalUncertaintySet
