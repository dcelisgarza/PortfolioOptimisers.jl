"""
    compact_radius_dof(rr::Regression, T::Number, N::Integer, K::Integer)
    compact_radius_dof(rr::CrossSectionalFactorModel, T::Number, N::Integer, K::Integer)

Degrees of freedom the fit behind a loadings block left in each idiosyncratic variance.

# Algorithm

 1. On a [`Regression`](@ref), return `T - K - 1`. The block comes from a per-asset time-series fit over `K` factors and an intercept, so each residual series spends `K + 1` of its `T` observations.
 2. On a [`CrossSectionalFactorModel`](@ref), return `T * (N - K) / N`. The block comes from a per-period fit **across** the cross-section, which spends `K` of the `N` assets each period rather than `K` of the `T` observations once, so the count is the fraction of each period that survives, over all `T` periods.

**Neither count is read off the block, because no block records one.** [`FactorPrior`](@ref) writes `esigma` as the column variances of the reconstruction error under whatever variance estimator it was given, and a Cross-Sectional Factor Prior writes the idiosyncratic covariance its own fit measured. So both counts are the count the *fit* spent, stated here, and a caller whose estimator spent a different number states it on the rule instead.

# Arguments

  - `rr`: Fitted loadings block.
  - `T`: Sample size, the effective one when the prior carries observation weights.
  - `N`: Number of assets.
  - `K`: Number of factors the fit spent.

# Returns

  - `dof::Number`: Degrees of freedom, before the rule refuses a non-positive count.

# Related

  - [`ResidualInflation`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function compact_radius_dof(::Regression, T::Number, ::Integer, K::Integer)
    return T - K - one(K)
end
function compact_radius_dof(::CrossSectionalFactorModel, T::Number, N::Integer, K::Integer)
    return T * (N - K) / N
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Effective sample size behind a prior result, Kish's when the result carries observation weights.

A weighted estimate carries the information of `sum(w)^2 / sum(w .^ 2)` equally weighted observations rather than of `size(pr.X, 1)` rows, and a rule that prices estimation error reads the former. This is the reading [`ConcentrationRadius`](@ref) already takes for the same reason, and it is [`effective_sample_size`](@ref) on the result's own weights, so a Scenario Cap's stated count is read here too (ADR 0138).

# Arguments

  - `pr`: Prior result.

# Returns

  - `T::Number`: Kish's effective sample size when `pr.w` is set, the count `pr.ens` states otherwise, and the raw row count when neither is set.

# Related

  - [`ResidualInflation`](@ref)
  - [`effective_sample_size`](@ref)
"""
function compact_radius_sample_size(pr::AbstractPriorResult)
    return effective_sample_size(pr, pr.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Sizes the compact radius as the upper confidence bound on the idiosyncratic variance, so the penalty is a quantile rather than a stated magnitude.

The set's penalty lives exactly where the idiosyncratic variance lives. The directions the factors span pay nothing, and the complement carries ``\\mathbf{D}``, the idiosyncratic covariance the loadings block measured. So the question the radius answers is how far the *estimate* of that covariance can sit from the truth, and the answer is a chi-squared bound on a variance.

**The radius collapses to the relative inflation under the default metric.** ``\\rho`` is dimensionless, and the operator norm below is `1` under [`InverseIdiosyncraticVarianceMetric`](@ref), because ``\\mathbf{W} = \\mathbf{D}^{-1}`` leaves a projector inside the norm. Under [`IdentityMetric`](@ref) the same norm carries the variance units that ``\\kappa`` needs there. One formula therefore serves every [`AbstractOrthogonalityMetric`](@ref), and the metric is read rather than dispatched on.

# Mathematical definition

```math
\\begin{align}
\\rho &= \\dfrac{\\nu}{\\chi^{2,\\,-1}_{\\nu}(q)} - 1\\,, \\\\
\\kappa &= \\rho \\left\\lVert \\mathbf{D}^{1/2}\\mathbf{W}^{1/2}\\left(\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}\\right) \\right\\rVert_{2}^{2}\\,.
\\end{align}
```

Where:

  - ``\\nu``: Degrees of freedom, [`compact_radius_dof`](@ref) when `dof` is `nothing`.
  - ``\\chi^{2,\\,-1}_{\\nu}(q)``: **Lower** ``q`` quantile of the chi-squared distribution, so a smaller `q` raises ``\\rho``.
  - ``\\mathbf{D}``: Idiosyncratic covariance the block carries, read as its diagonal.
  - ``\\mathbf{W}``: Cross-sectional metric, the identity on [`IdentityMetric`](@ref).
  - ``\\mathbf{Q}``: Orthonormal basis of the weighted factor span.

``\\rho`` is the exact upper bound at level ``1 - q``: ``\\nu \\hat{d}_{i} / d_{i}`` is ``\\chi^{2}_{\\nu}``, so ``d_{i} \\leq \\hat{d}_{i}\\nu / \\chi^{2,\\,-1}_{\\nu}(q)`` with that confidence, and ``\\rho`` is the *excess* over the estimate. The operator norm is then the tightest ``\\kappa`` satisfying the set's own bound, because conjugating ``\\mathbf{W}^{-1/2}`` out of ``\\kappa \\mathbf{C}^{\\intercal}(\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal})\\mathbf{C} \\succeq \\rho \\, \\mathbf{\\Pi}\\mathbf{D}\\mathbf{\\Pi}^{\\intercal}`` leaves ``\\kappa \\mathbf{P} \\succeq \\rho \\mathbf{P}\\mathbf{W}^{1/2}\\mathbf{D}\\mathbf{W}^{1/2}\\mathbf{P}``, whose solution on the range of the projector is that norm.

A factor model that spans the whole cross-section leaves ``\\mathbf{P} = \\mathbf{0}`` and a radius of zero, which is the same answer the mean axis gives for the same span.

**The two `q`s are the same kind of number over two different errors.** `ue.q` sizes the mean set, inverting a chi-squared at the dimension of the Orthogonal Subspace and reading no sample length at all. This one inverts a chi-squared at the residual degrees of freedom and shrinks like ``\\sqrt{2/T}``. Both tighten as `q` falls, so `q = nothing` reads `ue.q` and one stated level governs both axes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ResidualInflation(;
        q::Option{<:Number} = nothing,
        dof::Option{<:Number} = nothing
    ) -> ResidualInflation

Keywords correspond to the struct's fields. Both default to `nothing`, so a bare call constructs and reads the confidence level of its owner and the degrees of freedom of the fit behind the block.

## Validation

  - If `q` is not `nothing`: `0 < q < 1`.
  - If `dof` is not `nothing`: `isfinite(dof)` and `dof > 0`.

# Examples

```jldoctest
julia> ResidualInflation()
ResidualInflation
    q ┼ nothing
  dof ┴ nothing
```

```jldoctest
julia> ResidualInflation(; q = 0.01, dof = 240)
ResidualInflation
    q ┼ Float64: 0.01
  dof ┴ Int64: 240
```

# Related

  - [`AbstractCompactRadiusAlgorithm`](@ref)
  - [`VarianceFraction`](@ref)
  - [`k_compact`](@ref)
  - [`compact_radius_dof`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
@concrete struct ResidualInflation <: AbstractCompactRadiusAlgorithm
    """
    Confidence level of the variance bound (`0 < q < 1`), or `nothing` to read the `q` of the owning estimator. A *smaller* `q` gives a *larger* radius.
    """
    q
    """
    Degrees of freedom the fit left in each idiosyncratic variance, or `nothing` to derive them with [`compact_radius_dof`](@ref).
    """
    dof
    function ResidualInflation(q::Option{<:Number}, dof::Option{<:Number})
        if !isnothing(q)
            @argcheck(zero(q) < q < one(q), DomainError(q, "q must be in (0, 1)"))
        end
        if !isnothing(dof)
            @argcheck(isfinite(dof) && dof > zero(dof),
                      DomainError(dof, "dof must be finite and > 0"))
        end
        return new{typeof(q), typeof(dof)}(q, dof)
    end
end
function ResidualInflation(; q::Option{<:Number} = nothing,
                           dof::Option{<:Number} = nothing)::ResidualInflation
    return ResidualInflation(q, dof)
end
"""
$(DocStringExtensions.TYPEDEF)

Sizes the compact radius so the penalty is a stated fraction of the nominal variance at a reference portfolio, giving the caller a unit instead of a bare number.

The radius of a penalty is hard to state because it is a magnitude and not a probability. This rule converts it into one a caller can reason about: `f = 0.1` says *robustify by ten percent of nominal variance*, measured where the reference portfolio sits. It assumes no sampling distribution, so it serves a block whose idiosyncratic variances were measured under an estimator whose degrees of freedom nobody can state.

# Mathematical definition

```math
\\begin{align}
\\kappa &= f \\, \\dfrac{\\boldsymbol{w}_{0}^{\\intercal}\\hat{\\mathbf{\\Sigma}}\\boldsymbol{w}_{0}}{\\left\\lVert \\left(\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}\\right)\\mathbf{C}\\boldsymbol{w}_{0} \\right\\rVert_{2}^{2}}\\,.
\\end{align}
```

Where:

  - ``f``: Fraction of the nominal variance the penalty is to equal at ``\\boldsymbol{w}_{0}``.
  - ``\\boldsymbol{w}_{0}``: Reference portfolio, the equal-weight one when `w0` is `nothing`.
  - ``\\hat{\\mathbf{\\Sigma}}``: Nominal covariance the prior result carries.
  - ``\\mathbf{C}``: Diagonal metric square root of the covariance set.

The denominator is the penalty the set charges ``\\boldsymbol{w}_{0}`` at a unit radius, so the quotient is exactly the radius at which that penalty reaches ``f`` of the nominal variance.

**A span that covers the cross-section returns zero.** The rank test is exact — the basis has as many columns as rows — and it is a statement about the rank rather than a tolerance. The penalty is then identically zero on every portfolio, the set is inert, and zero is the radius the mean axis already returns for the same span.

**A reference portfolio inside the factor span sends the radius to infinity.** ``\\mathbf{C}\\boldsymbol{w}_{0} \\in \\operatorname{col}(\\mathbf{Q})`` leaves a zero denominator with a non-zero projector, which means the penalty vanishes at ``\\boldsymbol{w}_{0}`` while other portfolios still pay it. No finite radius makes a vanishing penalty a fraction of anything, so the quotient diverges. There is no guard against it, and the reason is that the two ways it can arrive are not one case: an exactly vanishing penalty gives a value that is not finite, which [`CompactCovarianceUncertaintySet`](@ref)'s own constructor refuses, while a projector that leaves a rounding residue gives a finite and enormous radius that no threshold separates from a legitimately large one. State a `w0` outside the span, or size the radius with [`ResidualInflation`](@ref), which reads no portfolio.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarianceFraction(;
        f::Number = 0.1,
        w0::Union{Nothing, <:VecNum, <:NonFiniteAllocationOptimisationEstimator} = nothing
    ) -> VarianceFraction

Keywords correspond to the struct's fields. Both default, so a bare call constructs and sizes the penalty at a tenth of the nominal variance of the equal-weight portfolio.

## Validation

  - `isfinite(f)` and `f > 0`.
  - If `w0` is a vector: `!isempty(w0)`.

# Examples

```jldoctest
julia> VarianceFraction()
VarianceFraction
   f ┼ Float64: 0.1
  w0 ┴ nothing
```

```jldoctest
julia> VarianceFraction(; f = 0.25, w0 = [0.5, 0.3, 0.2])
VarianceFraction
   f ┼ Float64: 0.25
  w0 ┴ Vector{Float64}: [0.5, 0.3, 0.2]
```

# Related

  - [`AbstractCompactRadiusAlgorithm`](@ref)
  - [`ResidualInflation`](@ref)
  - [`k_compact`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
"""
@concrete struct VarianceFraction <: AbstractCompactRadiusAlgorithm
    """
    Fraction of the nominal variance the penalty is to equal at the reference portfolio, `> 0`.
    """
    f
    """
    Reference portfolio the fraction is measured at. `nothing` reads the equal-weight portfolio, a vector is the portfolio itself, and an optimiser is run on the returns data the set was fitted beside.
    """
    w0
    function VarianceFraction(f::Number,
                              w0::Union{Nothing, <:VecNum,
                                        <:NonFiniteAllocationOptimisationEstimator})
        @argcheck(isfinite(f) && f > zero(f), DomainError(f, "f must be finite and > 0"))
        if isa(w0, AbstractVector)
            @argcheck(!isempty(w0), IsEmptyError("w0 cannot be empty"))
        end
        return new{typeof(f), typeof(w0)}(f, w0)
    end
end
function VarianceFraction(; f::Number = 0.1,
                          w0::Union{Nothing, <:VecNum,
                                    <:NonFiniteAllocationOptimisationEstimator} = nothing)::VarianceFraction
    return VarianceFraction(f, w0)
end
"""
    compact_reference_weights(::Nothing, N::Integer, ::Any, ::Type{E}) where {E}
    compact_reference_weights(w0::VecNum, N::Integer, ::Any, ::Type)
    compact_reference_weights(w0::NonFiniteAllocationOptimisationEstimator, N::Integer,
                              rd::ReturnsResult, ::Type)

Reference portfolio a [`VarianceFraction`](@ref) sizes its penalty at.

# Algorithm

 1. On `nothing`, return the equal-weight portfolio over the `N` assets the span covers.
 2. On a vector, return it, after checking it carries one entry per asset.
 3. On an optimiser, run [`optimise`](@ref) over `rd` and return the weights it produced. The optimiser carries its own solver, so nothing is threaded into the uncertainty-set fit. A set fitted with no returns data beside it refuses, because an optimiser has nothing to run on.

# Arguments

  - `w0`: The `w0` field of the rule.
  - `N`: Number of assets the span covers.
  - `rd`: Returns data the set was fitted beside, or `nothing`.
  - `E`: Element type of the geometry, which the equal-weight vector is built in.

# Validation

  - A stated vector carries `N` entries, else a `DimensionMismatch`.
  - An optimiser meets a `ReturnsResult`, else an `IsNothingError`.
  - The optimiser's weights carry `N` entries, else a `DimensionMismatch`.

# Returns

  - `w0::VecNum`: Reference portfolio of length `N`.

# Related

  - [`VarianceFraction`](@ref)
  - [`k_compact`](@ref)
  - [`optimise`](@ref)
"""
function compact_reference_weights(::Nothing, N::Integer, ::Any, ::Type{E}) where {E}
    return fill(inv(E(N)), N)
end
function compact_reference_weights(w0::VecNum, N::Integer, ::Any, ::Type)
    @argcheck(length(w0) == N,
              DimensionMismatch("`w0` must carry one entry per asset of the span:\nlength(w0) => $(length(w0))\nN => $(N)"))
    return w0
end
function compact_reference_weights(w0::NonFiniteAllocationOptimisationEstimator, N::Integer,
                                   rd, ::Type)
    @argcheck(isa(rd, ReturnsResult),
              IsNothingError("`VarianceFraction` was given `$(nameof(typeof(w0)))` as its reference portfolio, and an optimiser needs returns data to run on. The uncertainty set was fitted from a prior result alone, so none reached the rule.\nFit the set through an optimiser, which passes the returns data beside the prior, or state `w0` as a weight vector.\nGot\nw0 => $(nameof(typeof(w0)))\nrd => nothing"))
    w = optimise(w0, rd).w
    @argcheck(length(w) == N,
              DimensionMismatch("the reference optimiser returned weights over a different universe than the span:\nlength(w) => $(length(w))\nN => $(N)"))
    return w
end
"""
    k_compact(alg::ResidualInflation, q::Number, metric::AbstractOrthogonalityMetric,
              pr::AbstractPriorResult, rr::AbstractLoadingsRegressionResult, C::VecNum,
              Q::MatNum, rd)
    k_compact(alg::VarianceFraction, q::Number, metric::AbstractOrthogonalityMetric,
              pr::AbstractPriorResult, rr::AbstractLoadingsRegressionResult, C::VecNum,
              Q::MatNum, rd)
    k_compact(kappa::Number, args...)

Radius of a [`CompactCovarianceUncertaintySet`](@ref), computed from the prior result and the geometry the set was built on.

# Algorithm

 1. On a `Number`, return it unchanged. A stated radius is the radius, and this is the method every caller who states one reaches.
 2. On a [`ResidualInflation`](@ref), settle the confidence level as `alg.q` or `q`, settle the degrees of freedom as `alg.dof` or [`compact_radius_dof`](@ref), form the relative inflation `dof / cquantile-complement`, and scale it by the squared operator norm of ``\\mathbf{D}^{1/2}\\mathbf{W}^{1/2}`` against the orthogonal projector. ``\\mathbf{W}^{1/2}`` is the element-wise inverse of `C`, which the set already carries.
 3. On a [`VarianceFraction`](@ref), return zero when the span covers the cross-section, and otherwise divide `f` times the nominal variance at the reference portfolio by the penalty that portfolio pays at a unit radius.

# Arguments

  - `alg`: Rule, or the radius itself.
  - `q`: Confidence level of the owning estimator, read when the rule states none.
  - `metric`: Cross-sectional weighting the span was taken under. It reaches the rules through `C`, which is its inverse square root, so no method dispatches on it.
  - `pr`: Prior result the set is being fitted on.
  - `rr`: Loadings block the span came from.
  - `C`: Diagonal metric square root of the covariance set, ``\\mathbf{W}^{-1/2}``.
  - `Q`: Orthonormal basis of the weighted factor span.
  - `rd`: Returns data the set was fitted beside, or `nothing`.

# Validation

  - On [`ResidualInflation`](@ref): the settled degrees of freedom are finite and `> 0`, else a `DomainError` naming the sample length and the factor count that produced them.
  - [`idiosyncratic_variances`](@ref) refuses a block that carries no `esigma`.

# Returns

  - `kappa::Number`: Radius, which the set's own constructor then refuses if it is not finite and `>= 0`.

# Related

  - [`AbstractCompactRadiusAlgorithm`](@ref)
  - [`ResidualInflation`](@ref)
  - [`VarianceFraction`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`k_norm_ball`](@ref): the mean axis's counterpart, sized through the `method` slot of the same estimator.
"""
function k_compact(kappa::Number, args...)::Number
    return kappa
end
function k_compact(alg::ResidualInflation, q::Number, ::AbstractOrthogonalityMetric,
                   pr::AbstractPriorResult, rr::AbstractLoadingsRegressionResult, C::VecNum,
                   Q::MatNum, ::Any)
    N = size(Q, 1)
    T = compact_radius_sample_size(pr)
    K = size(rr.M, 2)
    dof = isnothing(alg.dof) ? compact_radius_dof(rr, T, N, K) : alg.dof
    @argcheck(isfinite(dof) && dof > zero(dof),
              DomainError(dof,
                          "the fit behind the loadings block left no degrees of freedom in its idiosyncratic variances, so no variance bound is defined.\nFit the prior on a longer sample, or state `dof` on the rule.\nGot\nT => $(T)\nN => $(N)\nK => $(K)\ndof => $(dof)"))
    qe = isnothing(alg.q) ? q : alg.q
    rho = dof / Distributions.quantile(Distributions.Chisq(dof), qe) - one(dof)
    # `C` is the inverse square root of the metric, so its element-wise inverse is `W^{1/2}`
    # and `sqrt.(d) ./ C` is the diagonal of `D^{1/2}W^{1/2}`. The squared operator norm of
    # that matrix against the orthogonal projector is the tightest radius satisfying the
    # set's own bound, and it is exactly `1` when the metric is the inverse idiosyncratic
    # variance, which leaves a bare projector inside the norm.
    d = idiosyncratic_variances(rr)
    P = LinearAlgebra.I - Q * transpose(Q)
    return rho * LinearAlgebra.opnorm((sqrt.(d) ./ C) .* P)^2
end
function k_compact(alg::VarianceFraction, ::Number, ::AbstractOrthogonalityMetric,
                   pr::AbstractPriorResult, ::AbstractLoadingsRegressionResult, C::VecNum,
                   Q::MatNum, rd)
    N = size(Q, 1)
    # The span covers the cross-section, so the penalty is identically zero on every
    # portfolio and no radius changes the set. This is a rank statement and not a
    # tolerance: `Q` is orthonormal, so as many columns as rows is a full basis.
    if size(Q, 2) == N
        return zero(eltype(Q))
    end
    w0 = compact_reference_weights(alg.w0, N, rd, eltype(Q))
    Cw = C .* w0
    return alg.f * LinearAlgebra.dot(w0, pr.sigma, w0) /
           sum(abs2, Cw - Q * (transpose(Q) * Cw))
end
export ResidualInflation, VarianceFraction
