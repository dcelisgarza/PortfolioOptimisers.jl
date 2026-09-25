"""
    compact_radius_dof(rr::Regression, T::Number, N::Integer, K::Integer)
    compact_radius_dof(rr::CrossSectionalFactorModel, T::Number, N::Integer, K::Integer)

Degrees of freedom that the fit behind a loadings block leaves in each idiosyncratic variance.

The block records no count, so this function states the count that each kind of fit spends. [`FactorPrior`](@ref) writes `esigma` as the column variances of the reconstruction error under its own variance estimator, and a Cross-Sectional Factor Prior writes the idiosyncratic covariance of its own fit. A caller whose fit spent a different count states it in the `dof` field of [`ResidualInflation`](@ref).

# Mathematical definition

```math
\\begin{align}
\\nu &= \\begin{cases}
T_{e} - K - 1 & \\textrm{on a time-series fit}\\\\
\\dfrac{T_{e} (N - K)}{N} & \\textrm{on a cross-sectional fit}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\nu``: Degrees of freedom of each idiosyncratic variance.
  - $(math_dict[:cal_T_e])
  - $(math_dict[:N])
  - $(math_dict[:K])

A time-series fit regresses each asset on ``K`` factors and an intercept, so each residual series spends ``K + 1`` of its ``T_{e}`` observations. A cross-sectional fit regresses the ``N`` assets of each period on ``K`` factors. Its ``T_{e} N`` residuals then carry ``T_{e} (N - K)`` degrees of freedom, which is ``T_{e} (N - K) / N`` for each asset.

# Algorithm

 1. On a [`Regression`](@ref), return the time-series count.
 2. On a [`CrossSectionalFactorModel`](@ref), return the cross-sectional count.

# Arguments

  - `rr`: Fitted loadings block.
  - `T`: Sample size, the effective one when the prior carries observation weights.
  - `N`: Number of assets.
  - `K`: Number of factors the fit spent.

# Returns

  - `dof::Number`: Degrees of freedom. The caller refuses a count that is not positive.

# Related

  - [`ResidualInflation`](@ref)
  - [`compact_radius_sample_size`](@ref)
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

Effective sample size of a prior result, Kish's when the result carries observation weights.

A rule that prices estimation error reads the number of equally weighted observations that the estimate is worth, not the row count of `pr.X`. This function is [`effective_sample_size`](@ref) on the result's own weights, so it also reads the count that a Scenario Cap states in `pr.ens`. [`ConcentrationRadius`](@ref) reads the same count.

# Arguments

  - `pr`: Prior result.

# Returns

  - `T::Number`: Kish's effective sample size when `pr.w` is set, the count `pr.ens` states otherwise, and the row count of `pr.X` when neither is set.

# Related

  - [`ResidualInflation`](@ref)
  - [`compact_radius_dof`](@ref)
  - [`effective_sample_size`](@ref)
"""
function compact_radius_sample_size(pr::AbstractPriorResult)
    return effective_sample_size(pr, pr.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Sizes the compact radius from a chi-squared upper confidence bound on each idiosyncratic variance.

The compact set adds no variance on the directions that the factors span. On the other directions it adds variance in proportion to ``\\mathbf{D}``, the idiosyncratic variances of the loadings block, and this rule sizes that addition from the estimation error of ``\\mathbf{D}``. Under the default [`InverseIdiosyncraticVarianceMetric`](@ref) the radius equals ``\\rho`` and has no units. Under [`IdentityMetric`](@ref) it has the units of a variance. One formula serves every [`AbstractOrthogonalityMetric`](@ref), so no method dispatches on the metric.

The level `1 - q` is exact only when each variance of the block is a residual sum of squares divided by ``\\nu``. The default fits of the library do not write that, so the bound holds with a lower probability. The mathematical definition gives that probability. To hold the bound at a stated level, state a smaller `q`, or use [`VarianceFraction`](@ref), which assumes no sampling law.

The `q` of the owning [`OrthogonalUncertaintySet`](@ref) also sizes its mean set. Under the default [`ChiSqKUncertaintyAlgorithm`](@ref) the mean set inverts a chi-squared distribution at the dimension of the Orthogonal Subspace and reads no sample size, while this rule inverts one at ``\\nu`` degrees of freedom. A smaller `q` makes both radii larger, so `q = nothing` reads the owner's `q` and one level sets both.

# Mathematical definition

```math
\\begin{align}
\\rho &= \\dfrac{\\nu}{\\chi^{2,\\,-1}_{\\nu}(q)} - 1\\,, \\\\
\\kappa &= \\rho \\left\\lVert \\mathbf{D}^{1/2}\\mathbf{W}^{1/2}\\mathbf{P} \\right\\rVert_{2}^{2}\\,, \\\\
\\mathbf{P} &= \\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}\\,.
\\end{align}
```

Where:

  - ``\\rho``: Relative inflation, the excess of the variance bound over the estimate, as a fraction of the estimate.
  - $(math_dict[:kappa_cpt])
  - ``\\nu``: Degrees of freedom of each idiosyncratic variance, [`compact_radius_dof`](@ref) when `dof` is `nothing`.
  - ``\\chi^{2,\\,-1}_{\\nu}(q)``: Lower ``q`` quantile of the chi-squared distribution with ``\\nu`` degrees of freedom.
  - ``q``: Confidence level of the bound, which holds with probability ``1 - q``.
  - ``\\mathbf{D}``: Diagonal matrix of the idiosyncratic variances of the loadings block.
  - ``\\mathbf{W}``: Diagonal cross-sectional metric, the identity under [`IdentityMetric`](@ref).
  - ``\\mathbf{P}``: Orthogonal projector onto the complement of the weighted factor span.
  - $(math_dict[:Q_cpt])
  - $(math_dict[:C_cpt])
  - $(math_dict[:w_port])

Let ``\\hat{d}_{i}`` be the estimated idiosyncratic variance of asset ``i`` and ``d_{i}`` its true value. When the residuals are Gaussian and ``\\hat{d}_{i}`` is their sum of squares divided by ``\\nu``, ``\\nu \\hat{d}_{i} / d_{i}`` follows the chi-squared distribution with ``\\nu`` degrees of freedom. Then ``d_{i} \\leq (1 + \\rho)\\hat{d}_{i}`` with probability ``1 - q``, and ``\\rho`` is the smallest inflation with that property.

When ``\\hat{d}_{i}`` divides the sum of squares by ``m`` instead of ``\\nu``, the probability is ``1 - F_{\\nu}\\left(m \\chi^{2,\\,-1}_{\\nu}(q) / \\nu\\right)``, where ``F_{\\nu}`` is the chi-squared distribution function with ``\\nu`` degrees of freedom. It is less than ``1 - q`` when ``m > \\nu``. The default variance estimator of [`FactorPrior`](@ref) divides by ``T - 1``, which is larger than ``\\nu = T - K - 1`` on a time-series fit of ``T`` observations over ``K`` factors. At ``T = 260``, ``K = 3`` and ``q = 0.05`` the probability is about ``0.936``. A Cross-Sectional Factor Prior writes an exponentially weighted variance by default, which does not follow a chi-squared law with ``\\nu`` degrees of freedom, so the level is approximate there too.

For large ``\\nu``, ``\\rho \\approx z_{1-q} \\sqrt{2 / \\nu}``, where ``z_{1-q}`` is the ``1 - q`` quantile of the standard normal distribution. So the radius falls like the inverse square root of the sample size.

For a portfolio ``\\boldsymbol{w}``, let ``\\boldsymbol{v} = \\mathbf{P}\\mathbf{C}\\boldsymbol{w}``. The penalty of the set is ``\\kappa \\lVert \\boldsymbol{v} \\rVert_{2}^{2}``. Because ``\\mathbf{C} = \\mathbf{W}^{-1/2}``, the inflation of the idiosyncratic variance of ``\\mathbf{C}^{-1}\\boldsymbol{v}``, the part of ``\\boldsymbol{w}`` that the span does not cover, is ``\\rho \\lVert \\mathbf{D}^{1/2}\\mathbf{W}^{1/2}\\boldsymbol{v} \\rVert_{2}^{2}``. The ``\\kappa`` of the definition is the smallest radius whose penalty covers that inflation for every ``\\boldsymbol{v}`` in the range of ``\\mathbf{P}``.

Under ``\\mathbf{W} = \\mathbf{D}^{-1}`` the norm is ``\\lVert \\mathbf{P} \\rVert_{2}^{2} = 1`` when ``\\mathbf{P} \\neq \\mathbf{0}``, so ``\\kappa = \\rho``. When the factors span the whole cross-section, ``\\mathbf{P} = \\mathbf{0}`` and ``\\kappa = 0``, which is also the radius of the mean set for the same span.

# Algorithm

The branch of [`k_compact`](@ref) that this rule selects runs these steps.

 1. Read `N`, the number of assets, as the row count of `Q`.
 2. Read `T`, the sample size, with [`compact_radius_sample_size`](@ref).
 3. Read `K`, the number of factors, as the column count of the loadings `rr.M`.
 4. Settle `dof` as `alg.dof`, or as [`compact_radius_dof`](@ref) over `T`, `N` and `K` when the rule states none.
 5. Refuse a `dof` that is not finite and positive, with a `DomainError` that names `T`, `N` and `K`.
 6. Settle `qe` as `alg.q`, or as the owner's `q` when the rule states none.
 7. Form `rho`, the relative inflation ``\\rho`` at `qe` and `dof`.
 8. Read `d`, the idiosyncratic variances of the block, with [`idiosyncratic_variances`](@ref).
 9. Form `P`, the projector ``\\mathbf{P}``.
10. Scale row `i` of `P` by `sqrt(d[i]) / C[i]`, the diagonal entry of ``\\mathbf{D}^{1/2}\\mathbf{W}^{1/2}``.
11. Return `rho` times the squared operator norm of the scaled matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ResidualInflation(;
        q::Option{<:Number} = nothing,
        dof::Option{<:Number} = nothing
    ) -> ResidualInflation

Keywords correspond to the struct's fields. Both default to `nothing`, so a bare call reads the confidence level of the owner and the degrees of freedom of the fit behind the block.

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
    Confidence level of the variance bound (`0 < q < 1`), or `nothing` to read the `q` of the owning estimator. The bound holds with probability `1 - q`, so a smaller `q` gives a larger radius.
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

Sizes the compact radius so that the penalty at a reference portfolio equals a stated fraction of its nominal variance.

The units of a radius change with the metric, so a bare number is hard to choose. With this rule, `f = 0.1` sets a penalty equal to ten percent of the nominal variance of the reference portfolio. The rule assumes no sampling law, so it applies when the degrees of freedom of the idiosyncratic variances are not known.

No guard refuses a reference portfolio inside the factor span. When the penalty of `w0` is exactly zero, the radius is not finite and the constructor of [`CompactCovarianceUncertaintySet`](@ref) refuses it. When round-off leaves a small penalty, the radius is finite and very large, and no threshold separates it from a valid large radius. State a `w0` outside the span, or use [`ResidualInflation`](@ref), which reads no portfolio.

# Mathematical definition

```math
\\begin{align}
\\kappa &= f \\, \\dfrac{\\boldsymbol{w}_{0}^{\\intercal}\\hat{\\mathbf{\\Sigma}}\\boldsymbol{w}_{0}}{\\left\\lVert \\left(\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}\\right)\\mathbf{C}\\boldsymbol{w}_{0} \\right\\rVert_{2}^{2}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:kappa_cpt])
  - ``f``: Fraction of the nominal variance that the penalty equals at ``\\boldsymbol{w}_{0}``.
  - ``\\boldsymbol{w}_{0}``: Reference portfolio, the equal-weight portfolio when `w0` is `nothing`.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:C_cpt])
  - $(math_dict[:Q_cpt])
  - $(math_dict[:N])

The denominator is the penalty of ``\\boldsymbol{w}_{0}`` at ``\\kappa = 1``. So at the radius ``\\kappa`` the penalty of ``\\boldsymbol{w}_{0}`` is ``f`` times its nominal variance.

When ``\\mathbf{C}\\boldsymbol{w}_{0}`` is in the column space of ``\\mathbf{Q}``, the denominator is zero. If ``\\mathbf{Q}`` has fewer than ``N`` columns, other portfolios still pay a penalty, and ``\\kappa`` is infinite. If ``\\mathbf{Q}`` has ``N`` columns, the penalty is zero on every portfolio and no radius changes the set.

# Algorithm

The branch of [`k_compact`](@ref) that this rule selects runs these steps.

 1. Read `N`, the number of assets, as the row count of `Q`.
 2. When `Q` has `N` columns, return zero. The span then covers the cross-section and the quotient is ``0/0``. The test compares the column count with the row count and has no tolerance, because the columns of `Q` are orthonormal.
 3. Read `w0`, the reference portfolio, with [`compact_reference_weights`](@ref).
 4. Form `Cw`, the element-wise product of `C` and `w0`.
 5. Return `f` times the variance of `w0` under `pr.sigma`, divided by the squared norm of `Cw` less its projection onto the columns of `Q`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarianceFraction(;
        f::Number = 0.1,
        w0::Union{Nothing, <:VecNum, <:NonFiniteAllocationOptimisationEstimator} = nothing
    ) -> VarianceFraction

Keywords correspond to the struct's fields. Both have defaults, so a bare call sets the penalty at a tenth of the nominal variance of the equal-weight portfolio.

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
  - [`compact_reference_weights`](@ref)
  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
"""
@concrete struct VarianceFraction <: AbstractCompactRadiusAlgorithm
    """
    Fraction of the nominal variance that the penalty equals at the reference portfolio, `> 0`.
    """
    f
    """
    Reference portfolio at which the fraction is measured. `nothing` reads the equal-weight portfolio, a vector is the portfolio itself, and an optimiser runs on the returns data of the fit.
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
                              rd, ::Type)

Reference portfolio at which a [`VarianceFraction`](@ref) sizes its penalty.

# Algorithm

 1. On `nothing`, return the equal-weight portfolio over the `N` assets of the span, in the element type `E`.
 2. On a vector, check that it has one entry per asset and return it.
 3. On an optimiser, run [`optimise`](@ref) on `rd`, check that the weights have one entry per asset, and return them. The optimiser carries its own solver, so the fit of the set passes it nothing.

# Arguments

  - `w0`: The `w0` field of the rule.
  - `N`: Number of assets of the span.
  - `rd`: Returns data of the fit, or `nothing`.
  - `E`: Element type of the geometry.

# Validation

  - A stated vector has `N` entries, else a `DimensionMismatch`.
  - An optimiser meets a `ReturnsResult`, else an `IsNothingError`. A set fitted from a prior result alone has no returns data for the optimiser to run on.
  - The optimiser's weights have `N` entries, else a `DimensionMismatch`.

# Returns

  - `w0::VecNum`: Reference portfolio of length `N`. A stated vector comes back unchanged.

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

Radius of a [`CompactCovarianceUncertaintySet`](@ref), from the prior result and the geometry of the set.

# Algorithm

 1. On a `Number`, return it unchanged. Every caller that states a radius reaches this method.
 2. On a [`ResidualInflation`](@ref), run the steps that its docstring lists.
 3. On a [`VarianceFraction`](@ref), run the steps that its docstring lists.

# Arguments

  - `alg`: Rule, or the radius itself.
  - `q`: Confidence level of the owning estimator, read when the rule states none.
  - `metric`: Cross-sectional metric of the span. It reaches the rules through `C`, its inverse square root, so no method dispatches on it.
  - `pr`: Prior result of the fit.
  - `rr`: Loadings block of the span.
  - `C`: Diagonal metric square root of the covariance set, ``\\mathbf{W}^{-1/2}``.
  - `Q`: Orthonormal basis of the weighted factor span.
  - `rd`: Returns data of the fit, or `nothing`.

# Validation

  - On a [`ResidualInflation`](@ref): the settled degrees of freedom are finite and `> 0`, else a `DomainError` that names the sample size, the number of assets and the number of factors.
  - On a [`ResidualInflation`](@ref): [`idiosyncratic_variances`](@ref) refuses a block that carries no `esigma`.
  - On a [`VarianceFraction`](@ref): [`compact_reference_weights`](@ref) refuses a reference portfolio of the wrong length, and an optimiser with no returns data.

# Returns

  - `kappa::Number`: Radius. The constructor of [`CompactCovarianceUncertaintySet`](@ref) refuses a radius that is not finite or is negative.

# Related

  - [`AbstractCompactRadiusAlgorithm`](@ref)
  - [`ResidualInflation`](@ref)
  - [`VarianceFraction`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`k_norm_ball`](@ref): the radius of the mean set, which the `method` field of the same estimator sizes.
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
public k_compact
