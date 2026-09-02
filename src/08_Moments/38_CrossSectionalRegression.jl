"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional solve algorithm types.

A member decides what [`CrossSectionalLinearRegression`](@ref) does when the weighted design of an observation is rank deficient, and the members differ on two axes: whether a rank test runs at all, and what happens when it fails. The decision is a real one, because Julia's `\\` answers a deficient design in three different ways. A **square** design goes to an `LU` factorisation that throws `LinearAlgebra.SingularException` on an exactly zero pivot. A **non-square** one goes to a column-pivoted `QR` whose solve completes the orthogonal factorisation, so it returns the minimum-norm solution and agrees with a pseudo-inverse. A design that is only **nearly** dependent passes the rank test of every member and returns a badly conditioned answer that a pseudo-inverse would truncate.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`RankDeficiencyRefusal`](@ref)
  - [`UncheckedSolve`](@ref)
  - [`MinimumNormSolve`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
"""
abstract type AbstractCrossSectionalSolveAlgorithm <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Solves the full-rank design directly and pseudo-inverts a rank-deficient one.

This is the default of [`CrossSectionalLinearRegression`](@ref), and it is the reference implementation's behaviour. It never throws on a dependent factor set: a square design that `\\` would refuse reaches the pseudo-inverse instead, and a non-square one reaches the same minimum-norm answer through either route. The rank test is what it costs, because the design is factorised twice whenever it passes.

# Examples

```jldoctest
julia> PseudoInverseFallback()
PseudoInverseFallback()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`RankDeficiencyRefusal`](@ref)
  - [`MinimumNormSolve`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
"""
struct PseudoInverseFallback <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Solves the full-rank design directly and refuses a rank-deficient one.

The refusal names the observation, the rank it measured, the factor count and the count of eligible assets, so a caller can tell a dependent factor set apart from a cross-section that is too small.

# Examples

```jldoctest
julia> RankDeficiencyRefusal()
RankDeficiencyRefusal()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
"""
struct RankDeficiencyRefusal <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Runs no rank test and takes whatever `\\` returns.

It is the cheapest member, because it factorises the design once instead of twice. A rank-deficient **non-square** design reaches a column-pivoted `QR` whose solve returns the minimum-norm answer, so it agrees with [`MinimumNormSolve`](@ref) there. An exactly singular **square** design reaches an `LU` factorisation instead and throws `LinearAlgebra.SingularException`, which is the one case where taking the answer unchecked costs the fit.

# Examples

```jldoctest
julia> UncheckedSolve()
UncheckedSolve()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
"""
struct UncheckedSolve <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Always pseudo-inverts, so it runs no rank test and takes no threshold.

Every observation takes the minimum-norm least-squares solution, whatever its rank, so the factor returns of two observations are comparable even when one of them lost a factor. It is the most expensive member. It parts from [`UncheckedSolve`](@ref) on a square design, which `\\` sends to an `LU` factorisation, and on a design that is only nearly dependent, where the pseudo-inverse truncates a singular value that `\\` keeps.

# Examples

```jldoctest
julia> MinimumNormSolve()
MinimumNormSolve()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
"""
struct MinimumNormSolve <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Holds the factor returns, the residuals, the eligible asset counts and the optional intercepts of a fitted cross-sectional regression.

The result is a sibling of [`Regression`](@ref) rather than a widening of it, because the two disagree on what an asset index means: a [`Regression`](@ref) holds one row per asset and [`port_opt_view`](@ref) slices its rows, whereas this result holds one row per observation and one column per asset, so the same index slices its columns. It carries no loadings matrix, because the exposures are the regression's input and an Exposure Estimator produces them.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{f}_{t} &= \\underset{\\boldsymbol{f}}{\\arg\\min} \\sum_{i = 1}^{N} w_{t,i} \\left(x_{t,i} - b_{t} - \\boldsymbol{z}_{t,i}^{\\intercal} \\boldsymbol{f}\\right)^{2} \\\\
\\boldsymbol{\\varepsilon}_{t} &= \\boldsymbol{x}_{t} - b_{t} \\boldsymbol{1} - \\mathbf{Z}_{t} \\boldsymbol{f}_{t}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{f}_{t}``: Factor returns of observation ``t``, the ``t``-th row of `f`.
  - $(math_dict[:x_t_obs])
  - ``\\boldsymbol{\\varepsilon}_{t}``: Residuals of observation ``t``, the ``t``-th row of `eps`.
  - ``\\mathbf{Z}_{t}``: Exposure slice of observation ``t``, ``N \\times K``, one row per asset.
  - ``\\boldsymbol{z}_{t,i}``: Exposures of asset ``i`` at observation ``t``, the ``i``-th row of ``\\mathbf{Z}_{t}``.
  - ``w_{t,i} \\geq 0``: Cross-sectional weight of asset ``i`` at observation ``t``. A weight of zero excludes the pair from the fit.
  - ``b_{t}``: Intercept of observation ``t``, the ``t``-th entry of `b`. The term is absent when `b` is unset.
  - $(math_dict[:N])
  - ``K``: Number of factors.

Each observation is one independent problem, so a factor return is a cross-sectional quantity and never a time-series one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalRegression(;
        f::MatNum,
        eps::MatNum,
        n::AbstractVector{<:Integer},
        b::Option{<:VecNum} = nothing
    ) -> CrossSectionalRegression

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(f)`, `!isempty(eps)` and `!isempty(n)`.
  - `size(f, 1) == size(eps, 1) == length(n)`.
  - `all(x -> x >= 0, n)`.
  - If provided, `!isempty(b)`, and `length(b) == size(f, 1)`.

## View parameters

`CrossSectionalRegression` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `eps` is sliced on its **second** axis, which is the asset axis of a cross-sectional result.
  - `f`, `n` and `b` pass through unchanged. Each is indexed by observation and by factor, and neither axis follows an asset selection.

# Examples

```jldoctest
julia> CrossSectionalRegression(; f = [1.0 2.0; 3.0 4.0], eps = [0.1 0.2 0.3; 0.4 0.5 0.6],
                                n = [3, 3])
CrossSectionalRegression
    f ┼ 2×2 Matrix{Float64}
  eps ┼ 2×3 Matrix{Float64}
    n ┼ Vector{Int64}: [3, 3]
    b ┴ nothing
```

# Related

  - [`AbstractCrossSectionalRegressionResult`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct CrossSectionalRegression <: AbstractCrossSectionalRegressionResult
    """
    Factor returns matrix `observations × factors`. Row `t` holds the coefficients of the cross-sectional fit of observation `t`.
    """
    f
    """
    Residual matrix `observations × assets`. It is the part of the returns the exposures do not explain, and an entry of an excluded pair is whatever the arithmetic of that pair produced, so a missing return leaves a missing residual.
    """
    eps
    """
    Count of assets that entered each fit, of length `observations`. An asset enters when its cross-sectional weight is positive.
    """
    n
    """
    $(arg_dict[:b])
    """
    b
    function CrossSectionalRegression(f::MatNum, eps::MatNum, n::AbstractVector{<:Integer},
                                      b::Option{<:VecNum})
        @argcheck(!isempty(f), IsEmptyError("f cannot be empty"))
        @argcheck(!isempty(eps), IsEmptyError("eps cannot be empty"))
        @argcheck(!isempty(n), IsEmptyError("n cannot be empty"))
        @argcheck(size(f, 1) == size(eps, 1) == length(n),
                  DimensionMismatch("f ($(size(f, 1)) rows), eps ($(size(eps, 1)) rows) and n ($(length(n))) must agree on the observation axis"))
        @argcheck(all(x -> x >= zero(x), n),
                  DomainError(n, "all entries of n must be >= 0"))
        if isa(b, VecNum)
            @argcheck(!isempty(b), IsEmptyError("b cannot be empty"))
            @argcheck(length(b) == size(f, 1),
                      DimensionMismatch("b ($(length(b))) must match f ($(size(f, 1)) rows)"))
        end
        return new{typeof(f), typeof(eps), typeof(n), typeof(b)}(f, eps, n, b)
    end
end
function CrossSectionalRegression(; f::MatNum, eps::MatNum, n::AbstractVector{<:Integer},
                                  b::Option{<:VecNum} = nothing)::CrossSectionalRegression
    return CrossSectionalRegression(f, eps, n, b)
end
"""
    port_opt_view(csr::CrossSectionalRegression, i, args...)

Return a view of a [`CrossSectionalRegression`](@ref) result, selecting only the assets indexed by `i`.

# Algorithm

 1. Take a column view of `eps` over `i`, giving the residuals of the selected assets. The asset axis of a cross-sectional result is the **second** one, because a row of `eps` is one observation.
 2. Build a new [`CrossSectionalRegression`](@ref) from that view and the three untouched fields, which re-runs every guard of the constructor.

# Arguments

  - `csr`: A cross-sectional regression result.
  - `i`: Indices of the assets to select.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `csr::CrossSectionalRegression`: A new result whose residuals are restricted to the selected assets.

# Examples

```jldoctest
julia> csr = CrossSectionalRegression(; f = [1.0 2.0], eps = [0.1 0.2 0.3], n = [3])
CrossSectionalRegression
    f ┼ 1×2 Matrix{Float64}
  eps ┼ 1×3 Matrix{Float64}
    n ┼ Vector{Int64}: [3]
    b ┴ nothing

julia> PortfolioOptimisers.port_opt_view(csr, [1, 3])
CrossSectionalRegression
    f ┼ 1×2 Matrix{Float64}
  eps ┼ 1×2 SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Vector{Int64}}, false}
    n ┼ Vector{Int64}: [3]
    b ┴ nothing
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(csr::CrossSectionalRegression, i, args...)::CrossSectionalRegression
    return CrossSectionalRegression(; f = csr.f, eps = view(csr.eps, :, i), n = csr.n,
                                    b = csr.b)
end
"""
$(DocStringExtensions.TYPEDEF)

Fits one weighted least squares per observation across the assets, in closed form.

The solve runs on the weighted design ``\\sqrt{w_{t,i}} \\, \\boldsymbol{z}_{t,i}`` rather than on the normal matrix ``\\mathbf{Z}_{t}^{\\intercal} \\mathbf{W}_{t} \\mathbf{Z}_{t}``, which halves the condition number in the exponent, and `alg` decides what happens when that design is rank deficient.

# Algorithm

 1. Check `Z`, `X` and `W`, and take the eligibility mask, per `# Validation` of [`cross_sectional_regression`](@ref).
 2. For each observation `t`, gather the eligible assets, their weights `w`, their exposures `A` and their returns `y`.
 3. When `intercept` is `true`, subtract the weighted means `ybar` and `xbar` from `y` and from `A`, so the fit runs through the weighted centroid of the cross-section.
 4. Scale `A` and `y` by `sqrt.(w)`, giving the weighted design and the weighted target.
 5. Solve the weighted design through the branch `alg` selects, giving the row `t` of `f`.
 6. When `intercept` is `true`, set the entry `t` of `b` to `ybar - dot(f[t, :], xbar)`.
 7. Subtract the systematic part from `X`, giving `eps`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalLinearRegression(;
        alg::AbstractCrossSectionalSolveAlgorithm = PseudoInverseFallback(),
        intercept::Bool = false
    ) -> CrossSectionalLinearRegression

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CrossSectionalLinearRegression()
CrossSectionalLinearRegression
        alg ┼ PseudoInverseFallback()
  intercept ┴ Bool: false
```

# Related

  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
@concrete struct CrossSectionalLinearRegression <: AbstractCrossSectionalRegressionEstimator
    """
    Solve algorithm, an [`AbstractCrossSectionalSolveAlgorithm`](@ref). It decides what the fit does with a rank-deficient weighted design.
    """
    alg
    """
    $(arg_dict[:csrint])
    """
    intercept
    function CrossSectionalLinearRegression(alg::AbstractCrossSectionalSolveAlgorithm,
                                            intercept::Bool)
        return new{typeof(alg), typeof(intercept)}(alg, intercept)
    end
end
function CrossSectionalLinearRegression(;
                                        alg::AbstractCrossSectionalSolveAlgorithm = PseudoInverseFallback(),
                                        intercept::Bool = false)::CrossSectionalLinearRegression
    return CrossSectionalLinearRegression(alg, intercept)
end
"""
$(DocStringExtensions.TYPEDEF)

Fits one external regression model per observation across the assets.

The cross-sectional weights reach the model as observation weights, through [`factory`](@ref) and the target's own `kwargs`, so any target the library carries — a [`LinearModel`](@ref) or a [`GeneralisedLinearModel`](@ref) — runs here unchanged. Unlike [`CrossSectionalLinearRegression`](@ref), the fit **refuses** an observation with no eligible asset, because an external model has no cross-section to read.

# Algorithm

 1. Check `Z`, `X` and `W`, and take the eligibility mask, per `# Validation` of [`cross_sectional_regression`](@ref).
 2. For each observation `t`, gather the eligible assets, their weights `w`, their exposures `A` and their returns `y`. Refuse when no asset is eligible.
 3. When `intercept` is `true`, subtract the weighted means `ybar` and `xbar` from `y` and from `A`. The target fits no intercept column of its own, so the intercept is recovered from the centroid rather than fitted.
 4. Build the per-observation target with `factory(tgt, StatsBase.aweights(w))`, fit it to `A` and `y`, and read its coefficients into the row `t` of `f`.
 5. When `intercept` is `true`, set the entry `t` of `b` to `ybar - dot(f[t, :], xbar)`.
 6. Subtract the systematic part from `X`, giving `eps`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalTargetRegression(;
        tgt::AbstractRegressionTarget = LinearModel(),
        intercept::Bool = false
    ) -> CrossSectionalTargetRegression

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CrossSectionalTargetRegression()
CrossSectionalTargetRegression
        tgt ┼ LinearModel
            │   kwargs ┴ @NamedTuple{}: NamedTuple()
  intercept ┴ Bool: false
```

# Related

  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`factory`](@ref)
"""
@concrete struct CrossSectionalTargetRegression <: AbstractCrossSectionalRegressionEstimator
    """
    $(field_dict[:retgt])
    """
    tgt
    """
    $(arg_dict[:csrint])
    """
    intercept
    function CrossSectionalTargetRegression(tgt::AbstractRegressionTarget, intercept::Bool)
        return new{typeof(tgt), typeof(intercept)}(tgt, intercept)
    end
end
function CrossSectionalTargetRegression(; tgt::AbstractRegressionTarget = LinearModel(),
                                        intercept::Bool = false)::CrossSectionalTargetRegression
    return CrossSectionalTargetRegression(tgt, intercept)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the numerical rank of `A`, read off the diagonal of a column-pivoted `QR`.

The test is written out rather than delegated to `LinearAlgebra.rank(::QRPivoted)`, which needs Julia 1.12 while this package supports 1.11.

# Algorithm

 1. Return `0` when `A` has no row or no column, because a factorisation of it has no pivot to read.
 2. Take the column-pivoted `LinearAlgebra.qr` of `A`. The magnitudes of the diagonal of its `R` are non-increasing, so they rank the columns by how much each adds to the span.
 3. Count the leading diagonal entries above `min(size(A)...) * eps(float(real(eltype(R)))) * abs(R[1, 1])`, giving the numerical rank. The tolerance is the one `LinearAlgebra.rank` applies to a pivoted `QR`.

# Arguments

  - `A::MatNum`: Weighted design of one observation, `eligible assets × factors`.

# Returns

  - `r::Int`: Numerical rank of `A`.

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`cross_sectional_solve`](@ref)
"""
function cross_sectional_rank(A::MatNum)::Int
    m = minimum(size(A))
    if iszero(m)
        return 0
    end
    R = LinearAlgebra.qr(A, LinearAlgebra.ColumnNorm()).R
    tol = m * eps(float(real(eltype(R)))) * abs(R[1, 1])
    return something(findfirst(i -> abs(R[i, i]) <= tol, 1:m), m + 1) - 1
end
"""
    cross_sectional_solve(alg::AbstractCrossSectionalSolveAlgorithm, A::MatNum, y::VecNum,
                          t::Integer) -> VecNum

Solve the weighted design `A` against the weighted target `y` through the branch `alg` selects.

# Algorithm

 1. [`UncheckedSolve`](@ref) returns `A \\ y`, with no rank test.
 2. [`MinimumNormSolve`](@ref) returns `LinearAlgebra.pinv(A) * y`, with no rank test.
 3. [`PseudoInverseFallback`](@ref) takes [`cross_sectional_rank`](@ref). It returns `A \\ y` when the rank equals the factor count, and `LinearAlgebra.pinv(A) * y` otherwise.
 4. [`RankDeficiencyRefusal`](@ref) takes [`cross_sectional_rank`](@ref), refuses when the rank falls short of the factor count, and returns `A \\ y` otherwise.

# Arguments

  - `alg`: Cross-sectional solve algorithm.
  - `A::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `y::VecNum`: Weighted target of one observation, of length `eligible assets`.
  - `t::Integer`: Index of the observation, named by the refusal of [`RankDeficiencyRefusal`](@ref).

# Validation

  - Under [`RankDeficiencyRefusal`](@ref), the rank of `A` equals `size(A, 2)`. The `ArgumentError` names the observation, the rank, the factor count and the count of eligible assets.

# Returns

  - `f::VecNum`: Factor returns of the observation, of length `size(A, 2)`.

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`cross_sectional_rank`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
function cross_sectional_solve(::UncheckedSolve, A::MatNum, y::VecNum, ::Integer)
    return A \ y
end
function cross_sectional_solve(::MinimumNormSolve, A::MatNum, y::VecNum, ::Integer)
    return LinearAlgebra.pinv(A) * y
end
function cross_sectional_solve(::PseudoInverseFallback, A::MatNum, y::VecNum, ::Integer)
    return cross_sectional_rank(A) == size(A, 2) ? A \ y : LinearAlgebra.pinv(A) * y
end
function cross_sectional_solve(::RankDeficiencyRefusal, A::MatNum, y::VecNum, t::Integer)
    r = cross_sectional_rank(A)
    @argcheck(r == size(A, 2),
              ArgumentError("the weighted design of observation $t has rank $r over $(size(A, 2)) factors and $(size(A, 1)) eligible assets, so its weighted least squares has no unique solution. Use PseudoInverseFallback() or MinimumNormSolve() to take the minimum-norm solution, UncheckedSolve() to take the truncated basic solution, drop the dependent factors, or widen the eligible cross-section"))
    return A \ y
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the eligibility mask of a cross-sectional design, after checking its three arrays.

An `(observation, asset)` pair is eligible when its cross-sectional weight is positive. The weight is the one contract: a pair excluded by a zero weight may carry a missing return or a missing exposure, and a pair with a positive weight may not.

# Arguments

  - `Z::Arr3Num`: Exposure tensor `observations × assets × factors`.
  - `X::MatNum`: Asset returns matrix `observations × assets`.
  - `W::MatNum`: Cross-sectional weights matrix `observations × assets`.

# Validation

  - `!isempty(Z)`, `!isempty(X)` and `!isempty(W)`.
  - `size(Z, 1) == size(X, 1)` and `size(Z, 2) == size(X, 2)`.
  - `size(W) == size(X)`.
  - `all(isfinite, W)` and `all(x -> x >= 0, W)`.
  - Every pair with a positive weight carries a finite return and finite exposures. The `IsNonFiniteError` names the observation, the asset and the weight.

# Returns

  - `act::BitMatrix`: Eligibility mask `observations × assets`, true where the weight is positive.

# Related

  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_r2`](@ref)
"""
function cross_sectional_design_mask(Z::Arr3Num, X::MatNum, W::MatNum)::BitMatrix
    @argcheck(!isempty(Z), IsEmptyError("Z cannot be empty"))
    @argcheck(!isempty(X), IsEmptyError("X cannot be empty"))
    @argcheck(!isempty(W), IsEmptyError("W cannot be empty"))
    @argcheck(size(Z, 1) == size(X, 1) && size(Z, 2) == size(X, 2),
              DimensionMismatch("Z ($(size(Z, 1))×$(size(Z, 2))×$(size(Z, 3))) must match X ($(size(X, 1))×$(size(X, 2))) on the observation and asset axes"))
    @argcheck(size(W) == size(X),
              DimensionMismatch("W ($(size(W, 1))×$(size(W, 2))) must match X ($(size(X, 1))×$(size(X, 2)))"))
    @argcheck(all(isfinite, W), IsNonFiniteError("all entries of W must be finite"))
    @argcheck(all(x -> x >= zero(x), W), DomainError(W, "all entries of W must be >= 0"))
    act = falses(size(X))
    for t in axes(X, 1), i in axes(X, 2)
        if W[t, i] > zero(eltype(W))
            @argcheck(isfinite(X[t, i]) && all(isfinite, view(Z, t, i, :)),
                      IsNonFiniteError("observation $t and asset $i carry the positive weight $(W[t, i]), so their return and all $(size(Z, 3)) of their exposures must be finite. Set the weight to zero to exclude the pair from the fit"))
            act[t, i] = true
        end
    end
    return act
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the factor returns of one observation, through the member's own solve.

[`CrossSectionalLinearRegression`](@ref) scales the design and the target by `sqrt.(w)` and hands them to [`cross_sectional_solve`](@ref). [`CrossSectionalTargetRegression`](@ref) hands the unscaled pair to the target, with `w` as the observation weights, and refuses an empty cross-section.

# Arguments

  - `cre`: Cross-sectional regression estimator.
  - `A::MatNum`: Exposures of the eligible assets, `eligible assets × factors`, already demeaned when an intercept is fitted.
  - `y::VecNum`: Returns of the eligible assets, already demeaned when an intercept is fitted.
  - `w::VecNum`: Cross-sectional weights of the eligible assets.
  - `t::Integer`: Index of the observation.

# Validation

  - Under [`CrossSectionalTargetRegression`](@ref), `!isempty(y)`. An external target has no cross-section to fit when no asset is eligible, and the `ArgumentError` names the observation.

# Returns

  - `f::VecNum`: Factor returns of the observation, of length `size(A, 2)`.

# Related

  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
function cross_sectional_coefficients(cre::CrossSectionalLinearRegression, A::MatNum,
                                      y::VecNum, w::VecNum, t::Integer)
    sq = sqrt.(w)
    return cross_sectional_solve(cre.alg, A .* sq, y .* sq, t)
end
function cross_sectional_coefficients(cre::CrossSectionalTargetRegression, A::MatNum,
                                      y::VecNum, w::VecNum, t::Integer)
    @argcheck(!isempty(y),
              ArgumentError("observation $t has no asset with a positive cross-sectional weight, and $(nameof(typeof(cre.tgt))) has no cross-section to fit. Widen the eligible cross-section, or use CrossSectionalLinearRegression, which answers an empty observation with zero factor returns"))
    return StatsAPI.coef(StatsAPI.fit(factory(cre.tgt, StatsBase.aweights(w)), A, y))
end
"""
    cross_sectional_regression(cre::AbstractCrossSectionalRegressionEstimator, Z::Arr3Num,
                               X::MatNum, W::MatNum) -> CrossSectionalRegression
    cross_sectional_regression(csr::CrossSectionalRegression, args...) -> CrossSectionalRegression

Fit one regression per observation across the assets, or return a fitted result unchanged.

The verb is its own rather than a fourth argument of [`regression`](@ref), because `regression(re::Regression, args...)` is a greedy passthrough that returns its first argument for any trailing arguments, so a time-series result handed to a cross-sectional call would return silently instead of raising.

The weight matrix `W` is an argument rather than a field, because a two-pass weighting scheme calls the estimator **twice** on one design with two different weight matrices, and a policy stored on the estimator would force a second estimator object or a mutation.

# Algorithm

 1. Take the eligibility mask through [`cross_sectional_design_mask`](@ref).
 2. For each observation `t`, gather the eligible assets, their weights `w`, their exposures `A` and their returns `y`, and record their count in `n`.
 3. When `cre.intercept` is `true`, take the weighted means `ybar` and `xbar` of `y` and of `A`, and subtract them. An observation with no eligible asset takes zero for both.
 4. Take the factor returns of the observation through [`cross_sectional_coefficients`](@ref), and write them into the row `t` of `f`. An observation with no eligible asset takes zero factor returns under [`CrossSectionalLinearRegression`](@ref).
 5. When `cre.intercept` is `true`, write `ybar - dot(f[t, :], xbar)` into the entry `t` of `b`.
 6. Subtract the systematic part, through [`cross_sectional_systematic`](@ref), from `X`, giving `eps`.

# Arguments

  - `cre`: Cross-sectional regression estimator.
  - `csr`: A cross-sectional regression result.
  - `Z::Arr3Num`: Exposure tensor `observations × assets × factors`.
  - `X::MatNum`: Asset returns matrix `observations × assets`.
  - `W::MatNum`: Cross-sectional weights matrix `observations × assets`.
  - `args...`: Additional positional arguments (ignored by the passthrough).

# Validation

  - The rules of [`cross_sectional_design_mask`](@ref).

# Returns

  - `csr::CrossSectionalRegression`: The fitted result, or the input result unchanged.

# Examples

```jldoctest
julia> Z = reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 1, 3, 2);

julia> cross_sectional_regression(CrossSectionalLinearRegression(), Z, [1.0 2.0 1.5], ones(1, 3))
CrossSectionalRegression
    f ┼ 1×2 Matrix{Float64}
  eps ┼ 1×3 Matrix{Float64}
    n ┼ Vector{Int64}: [3]
    b ┴ nothing
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
  - [`cross_sectional_design_mask`](@ref)
  - [`cross_sectional_coefficients`](@ref)
  - [`cross_sectional_systematic`](@ref)
  - [`regression`](@ref)
"""
function cross_sectional_regression(cre::AbstractCrossSectionalRegressionEstimator,
                                    Z::Arr3Num, X::MatNum,
                                    W::MatNum)::CrossSectionalRegression
    act = cross_sectional_design_mask(Z, X, W)
    Tf = promote_type(float(real(eltype(Z))), float(real(eltype(X))),
                      float(real(eltype(W))))
    K = size(Z, 3)
    f = zeros(Tf, size(X, 1), K)
    b = cre.intercept ? zeros(Tf, size(X, 1)) : nothing
    n = zeros(Int, size(X, 1))
    xbar = zeros(Tf, K)
    for t in axes(X, 1)
        idx = findall(view(act, t, :))
        n[t] = length(idx)
        w = Tf.(view(W, t, idx))
        A = Tf.(view(Z, t, idx, :))
        y = Tf.(view(X, t, idx))
        ybar = zero(Tf)
        fill!(xbar, zero(Tf))
        if cre.intercept
            sw = sum(w)
            if sw > zero(sw)
                ybar = LinearAlgebra.dot(w, y) / sw
                xbar .= vec(transpose(A) * w) ./ sw
            end
            y = y .- ybar
            A = A .- transpose(xbar)
        end
        fi = cross_sectional_coefficients(cre, A, y, w, t)
        f[t, :] = fi
        if cre.intercept
            b[t] = ybar - LinearAlgebra.dot(fi, xbar)
        end
    end
    return CrossSectionalRegression(; f = f, eps = X - cross_sectional_systematic(f, b, Z),
                                    n = n, b = b)
end
function cross_sectional_regression(csr::CrossSectionalRegression, args...)
    return csr
end
"""
    cross_sectional_systematic(f::MatNum, b::Option{<:VecNum}, Z::Arr3Num) -> MatNum

Return the systematic part of a cross-sectional regression, `observations × assets`.

# Mathematical definition

```math
\\begin{align}
\\hat{x}_{t,i} &= b_{t} + \\boldsymbol{z}_{t,i}^{\\intercal} \\boldsymbol{f}_{t}\\,.
\\end{align}
```

Where:

  - ``\\hat{x}_{t,i}``: Systematic return of asset ``i`` at observation ``t``.
  - ``\\boldsymbol{z}_{t,i}``: Exposures of asset ``i`` at observation ``t``.
  - ``\\boldsymbol{f}_{t}``: Factor returns of observation ``t``.
  - ``b_{t}``: Intercept of observation ``t``. The term is zero when no intercept was fitted.

# Arguments

  - `f::MatNum`: Factor returns matrix `observations × factors`.
  - `b::Option{<:VecNum}`: Intercept vector, or `nothing` when none was fitted.
  - `Z::Arr3Num`: Exposure tensor `observations × assets × factors`. The asset axis may differ from the one the fit saw; the observation and factor axes may not.

# Validation

  - `size(Z, 1) == size(f, 1)` and `size(Z, 3) == size(f, 2)`.

# Returns

  - `Xh::MatNum`: Systematic returns, `observations × assets`.

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
function cross_sectional_systematic(f::MatNum, b::Option{<:VecNum}, Z::Arr3Num)::MatNum
    @argcheck(size(Z, 1) == size(f, 1) && size(Z, 3) == size(f, 2),
              DimensionMismatch("Z ($(size(Z, 1))×$(size(Z, 2))×$(size(Z, 3))) must match f ($(size(f, 1))×$(size(f, 2))) on the observation and factor axes"))
    Xh = similar(f, size(Z, 1), size(Z, 2))
    for t in axes(Z, 1), i in axes(Z, 2)
        Xh[t, i] = LinearAlgebra.dot(view(Z, t, i, :), view(f, t, :))
    end
    return isnothing(b) ? Xh : Xh .+ b
end
"""
    StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num) -> MatNum

Return the systematic part of a fitted cross-sectional regression, `observations × assets`.

The residuals the result already carries are `X - predict(csr, Z)` for the `X` the fit saw, so this method earns its place on an exposure tensor the fit did not see.

# Algorithm

 1. Call [`cross_sectional_systematic`](@ref) with `csr.f`, `csr.b` and `Z`.

# Arguments

  - `csr`: A cross-sectional regression result.
  - `Z::Arr3Num`: Exposure tensor `observations × assets × factors`. The asset axis may differ from the one the fit saw; the observation and factor axes may not.

# Validation

  - The rules of [`cross_sectional_systematic`](@ref).

# Returns

  - `Xh::MatNum`: Systematic returns, `observations × assets`.

# Examples

```jldoctest
julia> Z = reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 1, 3, 2);

julia> csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, [1.0 2.0 1.5],
                                        ones(1, 3));

julia> predict(csr, Z)
1×3 Matrix{Float64}:
 1.0  2.0  1.5
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_systematic`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_r2`](@ref)
"""
function StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)::MatNum
    return cross_sectional_systematic(csr.f, csr.b, Z)
end
"""
    cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                       W::MatNum) -> VecNum

Return the weighted coefficient of determination of every observation.

An observation whose weighted total sum of squares is zero has no defined ratio, and its entry is `NaN`. [`mean_cross_sectional_r2`](@ref) is the scalar summary that skips those entries.

# Mathematical definition

```math
\\begin{align}
R^{2}_{t} &= 1 - \\frac{\\sum_{i} w_{t,i} \\left(x_{t,i} - \\hat{x}_{t,i}\\right)^{2}}{\\sum_{i} w_{t,i} \\left(x_{t,i} - \\bar{x}_{t}\\right)^{2}} \\\\
\\bar{x}_{t} &= \\frac{\\sum_{i} w_{t,i} x_{t,i}}{\\sum_{i} w_{t,i}}\\,.
\\end{align}
```

Where:

  - ``R^{2}_{t}``: Weighted coefficient of determination of observation ``t``.
  - ``x_{t,i}``: Return of asset ``i`` at observation ``t``.
  - ``\\hat{x}_{t,i}``: Systematic return of asset ``i`` at observation ``t``.
  - ``\\bar{x}_{t}``: Weighted mean return of observation ``t``.
  - ``w_{t,i} \\geq 0``: Cross-sectional weight of asset ``i`` at observation ``t``. Both sums run over the eligible assets alone.

# Algorithm

 1. Take the eligibility mask through [`cross_sectional_design_mask`](@ref).
 2. Take the systematic returns through [`StatsAPI.predict(csr::CrossSectionalRegression, Z::Arr3Num)`](@ref).
 3. For each observation, sum the weighted squared residuals and the weighted squared deviations from the weighted mean, over the eligible assets alone.
 4. Return `1 - rss / tss` per observation, and `NaN` where `tss` is not positive.

# Arguments

  - `csr`: A cross-sectional regression result.
  - `Z::Arr3Num`: Exposure tensor `observations × assets × factors`.
  - `X::MatNum`: Asset returns matrix `observations × assets`.
  - `W::MatNum`: Cross-sectional weights matrix `observations × assets`.

# Validation

  - The rules of [`cross_sectional_design_mask`](@ref).

# Returns

  - `r2::VecNum`: Coefficient of determination of every observation, of length `observations`.

# Examples

```jldoctest
julia> Z = reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 1, 3, 2);

julia> csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, [1.0 2.0 1.5],
                                        ones(1, 3));

julia> cross_sectional_r2(csr, Z, [1.0 2.0 1.5], ones(1, 3))
1-element Vector{Float64}:
 1.0
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`mean_cross_sectional_r2`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
function cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                            W::MatNum)::VecNum
    act = cross_sectional_design_mask(Z, X, W)
    Xh = StatsAPI.predict(csr, Z)
    r2 = fill(NaN, size(X, 1))
    for t in axes(X, 1)
        idx = findall(view(act, t, :))
        w = view(W, t, idx)
        y = view(X, t, idx)
        sw = sum(w)
        if sw <= zero(sw)
            continue
        end
        ybar = LinearAlgebra.dot(w, y) / sw
        rss = sum(w[k] * abs2(y[k] - Xh[t, idx[k]]) for k in eachindex(idx))
        tss = sum(w[k] * abs2(y[k] - ybar) for k in eachindex(idx))
        if tss > zero(tss)
            r2[t] = 1 - rss / tss
        end
    end
    return r2
end
"""
    mean_cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                            W::MatNum) -> Number

Return the mean weighted coefficient of determination across the observations.

The mean skips every observation whose ratio is undefined, and it is `NaN` when no observation defines one.

# Algorithm

 1. Take the per-observation vector through [`cross_sectional_r2`](@ref).
 2. Return the mean of its finite entries, and `NaN` when it holds none.

# Arguments

  - `csr`: A cross-sectional regression result.
  - `Z::Arr3Num`: Exposure tensor `observations × assets × factors`.
  - `X::MatNum`: Asset returns matrix `observations × assets`.
  - `W::MatNum`: Cross-sectional weights matrix `observations × assets`.

# Validation

  - The rules of [`cross_sectional_design_mask`](@ref).

# Returns

  - `r2::Number`: Mean coefficient of determination across the observations.

# Examples

```jldoctest
julia> Z = reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 1, 3, 2);

julia> csr = cross_sectional_regression(CrossSectionalLinearRegression(), Z, [1.0 2.0 1.5],
                                        ones(1, 3));

julia> mean_cross_sectional_r2(csr, Z, [1.0 2.0 1.5], ones(1, 3))
1.0
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_r2`](@ref)
"""
function mean_cross_sectional_r2(csr::CrossSectionalRegression, Z::Arr3Num, X::MatNum,
                                 W::MatNum)::Number
    r2 = cross_sectional_r2(csr, Z, X, W)
    keep = filter(isfinite, r2)
    return isempty(keep) ? convert(eltype(r2), NaN) : sum(keep) / length(keep)
end

export CrossSectionalRegression, CrossSectionalLinearRegression,
       CrossSectionalTargetRegression, PseudoInverseFallback, RankDeficiencyRefusal,
       UncheckedSolve, MinimumNormSolve, cross_sectional_regression, cross_sectional_r2,
       mean_cross_sectional_r2
