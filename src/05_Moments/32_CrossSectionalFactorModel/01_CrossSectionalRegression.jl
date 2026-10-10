"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional solve algorithm types.

A member decides what [`CrossSectionalLinearRegression`](@ref) and [`CrossSectionalTargetRegression`](@ref) do when the weighted design of an observation is rank deficient, and the members differ on two axes: whether a rank test runs at all, and what happens when it fails. The decision is a real one, because Julia's `\\` answers a deficient design in three different ways. A **square** design goes to an `LU` factorisation that throws `LinearAlgebra.SingularException` on an exactly zero pivot. A **non-square** one goes to a column-pivoted `QR` whose solve completes the orthogonal factorisation, so it returns the minimum-norm solution and agrees with a pseudo-inverse. A design that is only **nearly** dependent passes the rank test of every member and returns a badly conditioned answer that a pseudo-inverse would truncate.

A member means the same thing under both estimators, so the two give the same factor returns for a least-squares target. [`CrossSectionalTargetRegression`](@ref) refuses [`MinimumNormSolve`](@ref), whose answer on a target is the answer of [`PseudoInverseFallback`](@ref).

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`RankDeficiencyRefusal`](@ref)
  - [`UncheckedSolve`](@ref)
  - [`MinimumNormSolve`](@ref)
  - [`DependentColumnDrop`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
abstract type AbstractCrossSectionalSolveAlgorithm <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Solves the full-rank design directly and pseudo-inverts a rank-deficient one.

This is the default of [`CrossSectionalLinearRegression`](@ref) and of [`CrossSectionalTargetRegression`](@ref). It never throws on a dependent factor set: a square design that `\\` would refuse reaches the pseudo-inverse instead, and a non-square one reaches the same minimum-norm answer through either route. The rank test is what it costs, because the design is factorised twice whenever it passes.

A target has no pseudo-inverse, so on a rank-deficient design it fits the columns that [`DependentColumnDrop`](@ref) keeps, and the answer is projected onto the row space of the weighted design. Every coefficient vector that differs from the fit by a vector of the null space gives the same linear predictor, and the projection is the one of least norm among them. For a least-squares target it is the answer of [`CrossSectionalLinearRegression`](@ref). The projection reads the fit through the linear predictor alone, which is what [`is_basis_invariant`](@ref) states of a target, so the constructor refuses a target that answers `false`.

# Examples

```jldoctest
julia> PseudoInverseFallback()
PseudoInverseFallback()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`RankDeficiencyRefusal`](@ref)
  - [`MinimumNormSolve`](@ref)
  - [`DependentColumnDrop`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
struct PseudoInverseFallback <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Solves the full-rank design directly and refuses a rank-deficient one.

The refusal names the observation, the rank it measured, the factor count and the count of eligible assets, so a caller can tell a dependent factor set apart from a cross-section that is too small. Under [`CrossSectionalTargetRegression`](@ref) the refusal runs before the target fits.

# Examples

```jldoctest
julia> RankDeficiencyRefusal()
RankDeficiencyRefusal()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
struct RankDeficiencyRefusal <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Runs no rank test and takes whatever `\\` returns.

It is the cheapest member, because it factorises the design once instead of twice. A rank-deficient **non-square** design reaches a column-pivoted `QR` whose solve returns the minimum-norm answer, so it agrees with [`MinimumNormSolve`](@ref) there. An exactly singular **square** design reaches an `LU` factorisation instead and throws `LinearAlgebra.SingularException`, which is the one case where taking the answer unchecked costs the fit.

Under [`CrossSectionalTargetRegression`](@ref) the target sees the full design, and its own factorisation decides. `GLM` drops a collinear column by its own pivot and gives it a return of zero, and its pivot need not drop the column that the rank test of [`DependentColumnDrop`](@ref) drops. A factor set that is dependent to rounding alone can pass the pivot of `GLM` and fail its Cholesky factorisation, which throws `LinearAlgebra.PosDefException`. On the carry fold of a [`CrossSectionalFactorPrior`](@ref), a move solves an observation again when [`cross_sectional_rank`](@ref) finds it rank-deficient, so an observation that the pivot of `GLM` drops and the rank test keeps can keep the answer of the old basis.

# Examples

```jldoctest
julia> UncheckedSolve()
UncheckedSolve()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
struct UncheckedSolve <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Always pseudo-inverts, so it runs no rank test and takes no threshold.

Every observation takes the minimum-norm least-squares solution, whatever its rank, so the factor returns of two observations are comparable even when one of them lost a factor. It is the most expensive member. It parts from [`UncheckedSolve`](@ref) on a square design, which `\\` sends to an `LU` factorisation, and on a design that is only nearly dependent, where the pseudo-inverse truncates a singular value that `\\` keeps.

[`CrossSectionalTargetRegression`](@ref) refuses it. A target has no pseudo-inverse, and the projection of [`PseudoInverseFallback`](@ref) changes nothing on a design of full rank, so the member would be a second name for that one.

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

Solves the full-rank design directly, and drops the dependent columns of a rank-deficient one.

The member reads the rank `r` of the weighted design with [`cross_sectional_rank`](@ref). When `r` falls short of the factor count, it keeps the columns that the first `r` pivots of the column-pivoted `QR` of the design name, solves on them, and gives every other factor a return of zero. A design of rank zero gives zero to every factor. The answer is a basic solution, not the answer of least norm: the fitted values and the residuals are those of [`PseudoInverseFallback`](@ref), and the factor returns are not. The pivot prefers the column of larger weighted norm, so which factor takes zero depends on the scale of the exposures and on the basis of a Factor Family.

# Examples

```jldoctest
julia> DependentColumnDrop()
DependentColumnDrop()
```

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`PseudoInverseFallback`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
struct DependentColumnDrop <: AbstractCrossSectionalSolveAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Unseen Member rule of a [`CrossSectionalFactorPrior`](@ref), the rule that says what return an Unseen Member gets at an observation.

An Unseen Member is a member of a constrained Factor Family that no asset of positive regression weight loads on at one observation. The exposures lag the returns, so the zero-sum condition of an observation reads the benchmark weights of an earlier one, which can still weight the member after its last asset delists. The data state nothing about its return there. The rule states the return, and with it the zero-sum condition of the observation.

The fit stores the rule on its [`CrossSectionalFactorModel`](@ref) block. The regression diagnostics and the standard errors of a [`factor_attribution`](@ref) call the verb of the rule on the stored exposures and weights, so they read the design that the fit regressed on.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractUnseenMemberRule` and implement the following method:

## `unseen_member_design`

  - `unseen_member_design(rule::MyUnseenMemberRule, fcb::FactorFamilyBasis, B::Arr3Num, Zl::Arr3Num, W::MatNum) -> NamedTuple`: Returns the design that the regression of the prior solves, and the change of each observation that maps its coefficients back to the reduced factor returns.

### Arguments

  - `rule`: The member of the family.
  - `fcb`: The Factor Family Basis of the lagged exposures, one row per observation of `Zl`.
  - `B`: Lagged exposures on the raw axis, `observations × assets × factors`.
  - `Zl`: Lagged exposures on the reduced axis, which `fcb` gives from `B`.
  - `W`: Cross-sectional weights matrix `observations × assets`. The pairs of positive weight are the sample of each observation.

### Returns

  - `Z::Arr3Num`: The design to regress on.
  - `P`: One `t => P_t` pair per observation that the rule changes, where `P_t` is a `reduced factors × reduced factors` matrix, and the factor returns of `t` are `P_t` times the coefficients of `t`. It is empty when the rule changes no observation.

# Related

  - [`ZeroUnseenMember`](@ref)
  - [`SolvedUnseenMember`](@ref)
  - [`unseen_member_design`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
abstract type AbstractUnseenMemberRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The Unseen Member rule that gives an Unseen Member a return of zero at its observation, and holds the zero-sum condition of the observation over the other members of its family.

The data state nothing about the return of the member, so the rule states the value that an Empty Factor gets, and that the member gets at an observation that gives it no benchmark weight. The condition over the whole family still holds, because the member adds a zero term to it. The observation is identified again, so its factor returns do not depend on the member that the family drops. [`unseen_member_design`](@ref) states the change of the design.

# Constructors

    ZeroUnseenMember() -> ZeroUnseenMember

# Examples

```jldoctest
julia> ZeroUnseenMember()
ZeroUnseenMember()
```

# Related

  - [`AbstractUnseenMemberRule`](@ref)
  - [`SolvedUnseenMember`](@ref)
  - [`unseen_member_design`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
struct ZeroUnseenMember <: AbstractUnseenMemberRule end
"""
$(DocStringExtensions.TYPEDEF)

The Unseen Member rule that keeps an Unseen Member in the zero-sum condition of its observation, and lets the solve algorithm of the regression state its return.

The member keeps the benchmark weight of the lagged exposures in the condition. Its own column of the raw design is zero, so the member fixes the condition by itself, the condition no longer identifies the other members of the family, and the reduced design of the observation is rank-deficient. The solve algorithm of the Cross-Sectional Regression Estimator then gives the answer. Under [`PseudoInverseFallback`](@ref) it is the minimum-norm answer on the reduced axis, which depends on the member that the family drops. The residuals of the pairs of positive weight are the residuals of [`ZeroUnseenMember`](@ref) when the members that the sample sees span the same fitted values, which they do for a one-hot family.

A [`CrossSectionalFactorModel`](@ref) that a caller builds without a rule takes this one, because its design is then the reduced exposures themselves.

# Constructors

    SolvedUnseenMember() -> SolvedUnseenMember

# Examples

```jldoctest
julia> SolvedUnseenMember()
SolvedUnseenMember()
```

# Related

  - [`AbstractUnseenMemberRule`](@ref)
  - [`ZeroUnseenMember`](@ref)
  - [`unseen_member_design`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
struct SolvedUnseenMember <: AbstractUnseenMemberRule end
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
  - $(math_dict[:K])

Each observation is one independent problem, so a factor return is a cross-sectional quantity and never a time-series one.

A pair whose leverage is one has a direction of the design of its own, for example the only member of a level of a one-hot family. The fit reproduces its return whatever the return is, so its residual is zero by construction and tells nothing about its idiosyncratic risk. `h1` marks these pairs, and a least-squares fit writes an exact zero there in place of the round-off of the subtraction.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalRegression(;
        f::MatNum,
        eps::MatNum,
        n::AbstractVector{<:Integer},
        b::Option{<:VecNum} = nothing,
        h1::Option{<:AbstractMatrix{Bool}} = nothing
    ) -> CrossSectionalRegression

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(f)`, `!isempty(eps)` and `!isempty(n)`.
  - `size(f, 1) == size(eps, 1) == length(n)`.
  - `all(x -> x >= 0, n)`.
  - If provided, `!isempty(b)`, and `length(b) == size(f, 1)`.
  - If provided, `size(h1) == size(eps)`.

## View parameters

`CrossSectionalRegression` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `eps` and `h1` are sliced on their **second** axis, which is the asset axis of a cross-sectional result.
  - `f`, `n` and `b` pass through unchanged. Each is indexed by observation and by factor, and neither axis follows an asset selection.

# Examples

```jldoctest
julia> CrossSectionalRegression(; f = [1.0 2.0; 3.0 4.0], eps = [0.1 0.2 0.3; 0.4 0.5 0.6],
                                n = [3, 3])
CrossSectionalRegression
    f ┼ 2×2 Matrix{Float64}
  eps ┼ 2×3 Matrix{Float64}
    n ┼ Vector{Int64}: [3, 3]
    b ┼ nothing
   h1 ┴ nothing
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
    """
    Leverage-one mask `observations × assets`, or `nothing`. An entry is true where the pair entered the fit with a leverage of one, so the fit reproduced its return and its residual is zero by construction. `nothing` states no mark, as a result that a caller builds by hand does.
    """
    h1
    function CrossSectionalRegression(f::MatNum, eps::MatNum, n::AbstractVector{<:Integer},
                                      b::Option{<:VecNum},
                                      h1::Option{<:AbstractMatrix{Bool}})
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
        if !isnothing(h1)
            @argcheck(size(h1) == size(eps),
                      DimensionMismatch("h1 ($(size(h1, 1))×$(size(h1, 2))) must match eps ($(size(eps, 1))×$(size(eps, 2)))"))
        end
        return new{typeof(f), typeof(eps), typeof(n), typeof(b), typeof(h1)}(f, eps, n, b,
                                                                             h1)
    end
end
function CrossSectionalRegression(; f::MatNum, eps::MatNum, n::AbstractVector{<:Integer},
                                  b::Option{<:VecNum} = nothing,
                                  h1::Option{<:AbstractMatrix{Bool}} = nothing)::CrossSectionalRegression
    return CrossSectionalRegression(f, eps, n, b, h1)
end
"""
    port_opt_view(csr::CrossSectionalRegression, i, args...)

Return a view of a [`CrossSectionalRegression`](@ref) result, selecting only the assets indexed by `i`.

# Algorithm

 1. Take a column view of `eps` over `i`, giving the residuals of the selected assets. The asset axis of a cross-sectional result is the **second** one, because a row of `eps` is one observation.
 2. Take the same view of `h1` when it is set, through [`nothing_scalar_array_view_odd_order`](@ref).
 3. Build a new [`CrossSectionalRegression`](@ref) from the two views and the three untouched fields, which re-runs every guard of the constructor.

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
    b ┼ nothing
   h1 ┴ nothing

julia> PortfolioOptimisers.port_opt_view(csr, [1, 3])
CrossSectionalRegression
    f ┼ 1×2 Matrix{Float64}
  eps ┼ 1×2 SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Vector{Int64}}, false}
    n ┼ Vector{Int64}: [3]
    b ┼ nothing
   h1 ┴ nothing
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(csr::CrossSectionalRegression, i, args...)::CrossSectionalRegression
    return CrossSectionalRegression(; f = csr.f, eps = view(csr.eps, :, i), n = csr.n,
                                    b = csr.b,
                                    h1 = nothing_scalar_array_view_odd_order(csr.h1, :, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Fits one weighted least squares per observation across the assets, in closed form.

The solve runs on the weighted design ``\\sqrt{w_{t,i}} \\, \\boldsymbol{z}_{t,i}`` rather than on the normal matrix ``\\mathbf{Z}_{t}^{\\intercal} \\mathbf{W}_{t} \\mathbf{Z}_{t}``, which halves the condition number in the exponent, and `alg` decides what happens when that design is rank deficient.

The square root sets the arithmetic of the solve. An integer and a `Rational` weight have a floating-point square root, so the solve of a `Rational` panel runs in floating point. `f` keeps the `Rational` type of the panel, and holds the floating-point answer written as a `Rational`, not the exact least-squares answer. A caller who needs the exact answer solves the weighted normal equations in `Rational` arithmetic.

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
        intercept::Bool = false,
        ex::FLoops.Transducers.Executor = ThreadedEx()
    ) -> CrossSectionalLinearRegression

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CrossSectionalLinearRegression()
CrossSectionalLinearRegression
        alg ┼ PseudoInverseFallback()
  intercept ┼ Bool: false
         ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
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
    $(field_dict[:csalg])
    """
    alg
    """
    $(arg_dict[:csrint])
    """
    intercept
    """
    $(field_dict[:ex]) It runs the fits of the observations, which are independent problems. Each fit writes its own row, so every executor gives the same result, and a refusal is the one of the first observation that fails.
    """
    ex
    function CrossSectionalLinearRegression(alg::AbstractCrossSectionalSolveAlgorithm,
                                            intercept::Bool,
                                            ex::FLoops.Transducers.Executor)
        return new{typeof(alg), typeof(intercept), typeof(ex)}(alg, intercept, ex)
    end
end
function CrossSectionalLinearRegression(;
                                        alg::AbstractCrossSectionalSolveAlgorithm = PseudoInverseFallback(),
                                        intercept::Bool = false,
                                        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())::CrossSectionalLinearRegression
    return CrossSectionalLinearRegression(alg, intercept, ex)
end
"""
$(DocStringExtensions.TYPEDEF)

Fits one external regression model per observation across the assets.

The cross-sectional weights reach the model as observation weights, through [`factory`](@ref) and the target's own `kwargs`, so any target the library carries — a [`LinearModel`](@ref) or a [`GeneralisedLinearModel`](@ref) — runs here unchanged. A caller's own target runs here too, when it states the methods of the `# Interfaces` section of [`AbstractRegressionTarget`](@ref). A target with no weight method is refused, because its fit would ignore the cross-sectional weights. Unlike [`CrossSectionalLinearRegression`](@ref), the fit **refuses** an observation with no eligible asset, because an external model has no cross-section to read.

`alg` decides what the fit does with a rank-deficient weighted design, as it does for [`CrossSectionalLinearRegression`](@ref), and [`cross_sectional_target_solve`](@ref) states each member. The default [`PseudoInverseFallback`](@ref) fits the columns that the rank test keeps and projects the answer onto the row space of the weighted design, so a least-squares target gives the factor returns of [`CrossSectionalLinearRegression`](@ref). A rank test is needed because a factor set can be dependent to rounding alone, such as a beta that shrinks fully to the mean of its industry, and the factorisation of `GLM` refuses such a design with a `PosDefException`. A rank-deficient design is not a degenerate case: under [`SolvedUnseenMember`](@ref) the design of an observation with an Unseen Member is rank-deficient by construction.

# Algorithm

 1. Check `Z`, `X` and `W`, and take the eligibility mask, per `# Validation` of [`cross_sectional_regression`](@ref).
 2. For each observation `t`, gather the eligible assets, their weights `w`, their exposures `A` and their returns `y`. Refuse when no asset is eligible.
 3. When `intercept` is `true`, subtract the weighted means `ybar` and `xbar` from `y` and from `A`. The target fits no intercept column of its own, so the intercept is recovered from the centroid rather than fitted.
 4. Build the per-observation target with `factory(tgt, StatsBase.aweights(w))`, fit it through the branch `alg` selects, and read its coefficients into the row `t` of `f`.
 5. When `intercept` is `true`, set the entry `t` of `b` to `ybar - dot(f[t, :], xbar)`.
 6. Subtract the systematic part from `X`, giving `eps`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalTargetRegression(;
        tgt::AbstractRegressionTarget = LinearModel(),
        alg::AbstractCrossSectionalSolveAlgorithm = PseudoInverseFallback(),
        intercept::Bool = false,
        ex::FLoops.Transducers.Executor = ThreadedEx()
    ) -> CrossSectionalTargetRegression

Keywords correspond to the struct's fields.

## Validation

  - The rules of [`assert_cross_sectional_target_solve`](@ref): `alg` is not [`MinimumNormSolve`](@ref), and under [`PseudoInverseFallback`](@ref), [`is_basis_invariant`](@ref) answers `true` for `tgt`.

# Examples

```jldoctest
julia> CrossSectionalTargetRegression()
CrossSectionalTargetRegression
        tgt ┼ LinearModel
            │   kwargs ┴ @NamedTuple{}: NamedTuple()
        alg ┼ PseudoInverseFallback()
  intercept ┼ Bool: false
         ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
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
    $(field_dict[:csalg])
    """
    alg
    """
    $(arg_dict[:csrint])
    """
    intercept
    """
    $(field_dict[:ex]) It runs the fits of the observations, which are independent problems. Each fit writes its own row, so every executor gives the same result, and a refusal is the one of the first observation that fails.
    """
    ex
    function CrossSectionalTargetRegression(tgt::AbstractRegressionTarget,
                                            alg::AbstractCrossSectionalSolveAlgorithm,
                                            intercept::Bool,
                                            ex::FLoops.Transducers.Executor)
        assert_cross_sectional_target_solve(alg, tgt)
        return new{typeof(tgt), typeof(alg), typeof(intercept), typeof(ex)}(tgt, alg,
                                                                            intercept, ex)
    end
end
function CrossSectionalTargetRegression(; tgt::AbstractRegressionTarget = LinearModel(),
                                        alg::AbstractCrossSectionalSolveAlgorithm = PseudoInverseFallback(),
                                        intercept::Bool = false,
                                        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())::CrossSectionalTargetRegression
    return CrossSectionalTargetRegression(tgt, alg, intercept, ex)
end
"""
    assert_cross_sectional_target_solve(alg::MinimumNormSolve, tgt::AbstractRegressionTarget)
    assert_cross_sectional_target_solve(alg::PseudoInverseFallback, tgt::AbstractRegressionTarget)
    assert_cross_sectional_target_solve(alg::AbstractCrossSectionalSolveAlgorithm,
                                        tgt::AbstractRegressionTarget)

Check that a [`CrossSectionalTargetRegression`](@ref) can solve a rank-deficient design of `tgt` under `alg`.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`MinimumNormSolve`](@ref) is refused. On a target its answer is the answer of [`PseudoInverseFallback`](@ref).
 2. [`PseudoInverseFallback`](@ref) is refused when [`is_basis_invariant`](@ref) answers `false` for `tgt`. Its projection keeps the linear predictor of the fit, so it is the answer of least norm only for a target that reads the design through the linear predictor alone.
 3. Every other member passes.

# Arguments

  - `alg`: Cross-sectional solve algorithm.
  - `tgt`: Regression target.

# Validation

  - `!isa(alg, MinimumNormSolve)`. Raises an `ArgumentError`.
  - Under [`PseudoInverseFallback`](@ref), `is_basis_invariant(tgt)`. Raises an `ArgumentError` that names the target.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalTargetRegression`](@ref)
  - [`cross_sectional_target_solve`](@ref)
  - [`is_basis_invariant`](@ref)
"""
function assert_cross_sectional_target_solve(::MinimumNormSolve, ::AbstractRegressionTarget)
    return throw(ArgumentError("MinimumNormSolve() has no meaning for a regression target: a target has no pseudo-inverse, and the answer of least norm of a target is the answer of PseudoInverseFallback(). Use PseudoInverseFallback()"))
end
function assert_cross_sectional_target_solve(::PseudoInverseFallback,
                                             tgt::AbstractRegressionTarget)
    @argcheck(is_basis_invariant(tgt),
              ArgumentError("PseudoInverseFallback() projects the fit of $(nameof(typeof(tgt))) onto the row space of a rank-deficient design, which keeps the answer of the target only when the target reads the design through the linear predictor alone, and is_basis_invariant answers false for it. Use UncheckedSolve() to give the target the full design, DependentColumnDrop() to drop the dependent columns, RankDeficiencyRefusal() to refuse a rank-deficient design, or add a method of PortfolioOptimisers.is_basis_invariant that answers true"))
    return nothing
end
function assert_cross_sectional_target_solve(::AbstractCrossSectionalSolveAlgorithm,
                                             ::AbstractRegressionTarget)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the numerical rank of `A`, read off the diagonal of a column-pivoted `QR`.

The test is written out rather than delegated to `LinearAlgebra.rank(::QRPivoted)`, which needs Julia 1.12 while this package supports 1.11.

# Algorithm

 1. Return `0` when `A` has no row or no column, because a factorisation of it has no pivot to read.
 2. Take the column-pivoted `LinearAlgebra.qr` of `A`. The magnitudes of the diagonal of its `R` are non-increasing, so they rank the columns by how much each adds to the span.
 3. Count the leading diagonal entries above `min(size(A)...) * eps(real(eltype(R))) * abs(R[1, 1])`, giving the numerical rank. The tolerance is the one `LinearAlgebra.rank` applies to a pivoted `QR`.

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
    tol = m * eps(real(eltype(R))) * abs(R[1, 1])
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
 4. [`RankDeficiencyRefusal`](@ref) refuses a rank-deficient `A` with [`assert_cross_sectional_rank`](@ref), and returns `A \\ y` otherwise.
 5. [`DependentColumnDrop`](@ref) takes [`cross_sectional_rank`](@ref). It returns `A \\ y` when the rank equals the factor count, and the answer of [`cross_sectional_column_drop`](@ref) otherwise.

# Arguments

  - `alg`: Cross-sectional solve algorithm.
  - `A::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `y::VecNum`: Weighted target of one observation, of length `eligible assets`.
  - `t::Integer`: Index of the observation, named by the refusal of [`RankDeficiencyRefusal`](@ref).

# Validation

  - Under [`RankDeficiencyRefusal`](@ref), the rules of [`assert_cross_sectional_rank`](@ref).

# Returns

  - `f::VecNum`: Factor returns of the observation, of length `size(A, 2)`.

# Related

  - [`AbstractCrossSectionalSolveAlgorithm`](@ref)
  - [`cross_sectional_rank`](@ref)
  - [`cross_sectional_target_solve`](@ref)
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
    assert_cross_sectional_rank(A, t)
    return A \ y
end
function cross_sectional_solve(::DependentColumnDrop, A::MatNum, y::VecNum, ::Integer)
    r = cross_sectional_rank(A)
    return if r == size(A, 2)
        A \ y
    else
        cross_sectional_column_drop(keep -> A[:, keep] \ y, A, y, r)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse the weighted design of one observation when its rank falls short of its factor count.

[`RankDeficiencyRefusal`](@ref) reads it under both cross-sectional regression estimators, so the two refuse the same observation with the same message.

# Arguments

  - `D::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `t::Integer`: Index of the observation, named by the refusal.

# Validation

  - [`cross_sectional_rank`](@ref) of `D` equals `size(D, 2)`. The `ArgumentError` names the observation, the rank, the factor count and the count of eligible assets.

# Returns

  - `nothing`.

# Related

  - [`RankDeficiencyRefusal`](@ref)
  - [`cross_sectional_solve`](@ref)
  - [`cross_sectional_target_solve`](@ref)
"""
function assert_cross_sectional_rank(D::MatNum, t::Integer)::Nothing
    r = cross_sectional_rank(D)
    @argcheck(r == size(D, 2),
              ArgumentError("the weighted design of observation $t has rank $r over $(size(D, 2)) factors and $(size(D, 1)) eligible assets, so its weighted least squares has no unique solution. Use PseudoInverseFallback() to take the minimum-norm solution, DependentColumnDrop() to give the dependent factors a return of zero, UncheckedSolve() to take the answer unchecked, or widen the eligible cross-section"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the factor returns of a rank-deficient observation that fits only the columns its rank test keeps.

# Algorithm

 1. Return zero for every factor when `r` is zero.
 2. Take the column-pivoted `LinearAlgebra.qr` of `D`, and keep the columns that its first `r` pivots name, in their order on the factor axis.
 3. Fit the kept columns with `fit_cols`, and give every other factor a return of zero.

# Arguments

  - `fit_cols`: Function of the kept column indices that returns their coefficients.
  - `D::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `y::VecNum`: Target of the observation. Its type and the type of `D` give the type of the zero answer of rank zero.
  - `r::Integer`: Rank of `D`, read by [`cross_sectional_rank`](@ref), below `size(D, 2)`.

# Returns

  - `f::VecNum`: Factor returns of the observation, of length `size(D, 2)`, zero at each dropped factor.

# Related

  - [`DependentColumnDrop`](@ref)
  - [`cross_sectional_solve`](@ref)
  - [`cross_sectional_target_solve`](@ref)
"""
function cross_sectional_column_drop(fit_cols, D::MatNum, y::VecNum, r::Integer)::VecNum
    if iszero(r)
        return zeros(promote_type(eltype(D), eltype(y)), size(D, 2))
    end
    keep = sort!(LinearAlgebra.qr(D, LinearAlgebra.ColumnNorm()).p[1:r])
    c = fit_cols(keep)
    f = zeros(eltype(c), size(D, 2))
    f[keep] = c
    return f
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the projection of a coefficient vector onto the row space of a rank-deficient weighted design.

Two coefficient vectors whose difference lies in the null space of the design give the same linear predictor. The projection removes the part of `f` in the null space, so it keeps the linear predictor and it is the vector of least norm among those that give it. For a least-squares answer, it is the answer of the pseudo-inverse.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{f}^{\\star} &= \\left(\\mathbf{I} - \\mathbf{N} \\mathbf{N}^{\\intercal}\\right) \\boldsymbol{f}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{f}^{\\star}``: Projected coefficient vector.
  - ``\\boldsymbol{f}``: Coefficient vector of the fit on the kept columns.
  - ``\\mathbf{N}``: Right singular vectors of the weighted design past its rank `r`, an orthonormal basis of its null space.

# Arguments

  - `f::VecNum`: Coefficient vector, of length `size(D, 2)`.
  - `D::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `r::Integer`: Rank of `D`, read by [`cross_sectional_rank`](@ref).

# Returns

  - `f::VecNum`: The projected coefficient vector. `f` itself when `r` is zero, because the fit gave zero to every factor.

# Related

  - [`PseudoInverseFallback`](@ref)
  - [`cross_sectional_target_solve`](@ref)
"""
function cross_sectional_row_space(f::VecNum, D::MatNum, r::Integer)::VecNum
    if iszero(r)
        return f
    end
    N = view(LinearAlgebra.svd(D; full = true).V, :, (r + 1):size(D, 2))
    return f - N * (N' * f)
end
"""
    cross_sectional_target_solve(alg::UncheckedSolve, tgt::AbstractRegressionTarget,
                                 obs::NamedTuple) -> VecNum
    cross_sectional_target_solve(alg::RankDeficiencyRefusal, tgt::AbstractRegressionTarget,
                                 obs::NamedTuple) -> VecNum
    cross_sectional_target_solve(alg::Union{DependentColumnDrop, PseudoInverseFallback},
                                 tgt::AbstractRegressionTarget, obs::NamedTuple) -> VecNum

Fit the target of one observation of a [`CrossSectionalTargetRegression`](@ref) through the branch `alg` selects.

# Algorithm

 1. [`UncheckedSolve`](@ref) fits `tgt` to `A` and `y`, with no rank test.
 2. [`RankDeficiencyRefusal`](@ref) refuses a rank-deficient `D` with [`assert_cross_sectional_rank`](@ref), and fits `tgt` to `A` and `y` otherwise.
 3. [`DependentColumnDrop`](@ref) and [`PseudoInverseFallback`](@ref) take the rank `r` of `D` with [`cross_sectional_rank`](@ref). When `r` equals the factor count, they fit `tgt` to `A` and `y`. Otherwise they fit `tgt` to the columns of `A` that [`cross_sectional_column_drop`](@ref) keeps, and [`PseudoInverseFallback`](@ref) projects the answer onto the row space of `D` with [`cross_sectional_row_space`](@ref).

The constructor of [`CrossSectionalTargetRegression`](@ref) refuses [`MinimumNormSolve`](@ref), so no method takes it.

# Arguments

  - `alg`: Cross-sectional solve algorithm.

  - `tgt`: Regression target of the observation, which carries its weights.

  - `obs::NamedTuple`: The observation, `(; A, y, D, t)`:

      + `A::MatNum`: Exposures of the eligible assets, `eligible assets × factors`, already demeaned when an intercept is fitted.
      + `y::VecNum`: Returns of the eligible assets, already demeaned when an intercept is fitted.
      + `D::MatNum`: Weighted design `sqrt.(w) .* A`, which the rank test reads.
      + `t::Integer`: Index of the observation, named by the refusal of [`RankDeficiencyRefusal`](@ref).

# Validation

  - Under [`RankDeficiencyRefusal`](@ref), the rules of [`assert_cross_sectional_rank`](@ref).

# Returns

  - `f::VecNum`: Factor returns of the observation, of length `size(A, 2)`.

# Related

  - [`CrossSectionalTargetRegression`](@ref)
  - [`cross_sectional_solve`](@ref)
  - [`cross_sectional_coefficients`](@ref)
"""
function cross_sectional_target_solve(::UncheckedSolve, tgt::AbstractRegressionTarget,
                                      obs::NamedTuple)
    return StatsAPI.coef(StatsAPI.fit(tgt, obs.A, obs.y))
end
function cross_sectional_target_solve(::RankDeficiencyRefusal,
                                      tgt::AbstractRegressionTarget, obs::NamedTuple)
    assert_cross_sectional_rank(obs.D, obs.t)
    return StatsAPI.coef(StatsAPI.fit(tgt, obs.A, obs.y))
end
function cross_sectional_target_solve(alg::Union{DependentColumnDrop,
                                                 PseudoInverseFallback},
                                      tgt::AbstractRegressionTarget, obs::NamedTuple)
    (; A, y, D) = obs
    r = cross_sectional_rank(D)
    if r == size(A, 2)
        return StatsAPI.coef(StatsAPI.fit(tgt, A, y))
    end
    f = cross_sectional_column_drop(keep -> StatsAPI.coef(StatsAPI.fit(tgt, A[:, keep], y)),
                                    D, y, r)
    return isa(alg, PseudoInverseFallback) ? cross_sectional_row_space(f, D, r) : f
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

[`CrossSectionalLinearRegression`](@ref) scales the design and the target by `sqrt.(w)` and hands them to [`cross_sectional_solve`](@ref). The square root of a `Rational` weight is a float, so that solve runs in floating point for a `Rational` panel. [`CrossSectionalTargetRegression`](@ref) refuses an empty cross-section, and hands the unscaled pair to [`cross_sectional_target_solve`](@ref), with `w` as the observation weights of the target and the weighted design `sqrt.(w) .* A` for the rank test of its solve algorithm.

# Arguments

  - `cre`: Cross-sectional regression estimator.
  - `A::MatNum`: Exposures of the eligible assets, `eligible assets × factors`, already demeaned when an intercept is fitted.
  - `y::VecNum`: Returns of the eligible assets, already demeaned when an intercept is fitted.
  - `w::VecNum`: Cross-sectional weights of the eligible assets.
  - `t::Integer`: Index of the observation.

# Validation

  - Under [`CrossSectionalTargetRegression`](@ref), `!isempty(y)`. An external target has no cross-section to fit when no asset is eligible, and the `ArgumentError` names the observation.
  - Under [`CrossSectionalTargetRegression`](@ref), the target has a `factory(tgt, w)` method that carries the weights, else [`factory(::AbstractRegressionTarget, ::ObsWeights)`](@ref) throws an `ArgumentError`.

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
    tgt = factory(cre.tgt, StatsBase.aweights(w))
    return cross_sectional_target_solve(cre.alg, tgt,
                                        (; A = A, y = y, D = A .* sqrt.(w), t = t))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the mask of the eligible assets of one observation whose leverage in the weighted design is one.

The leverage of row ``i`` is the diagonal entry ``h_{i}`` of the projection onto the column span of the design. It is one exactly when the design gives the row a direction of its own, for example when the row is the only member of a level of a one-hot family. A least-squares fit then reproduces the target of that row whatever it is, so its residual is zero by construction and its variance under the design is zero too. The mask does not depend on the scale of the target, and it does not depend on a positive weight either, because a positive weight moves no row into or out of the span of the others.

# Mathematical definition

```math
\\begin{align}
h_{i} &= \\sum_{j = 1}^{r} Q_{ij}^{2}\\,, &
\\text{mark}_{i} &= 1 - h_{i} \\le \\sqrt{\\epsilon}\\,.
\\end{align}
```

Where:

  - ``h_{i}``: Leverage of row ``i``.
  - ``\\mathbf{Q}``: Orthonormal factor of the column-pivoted `QR` of the weighted design `sqrt.(w) .* A`, with the column `sqrt.(w)` in front under `intercept`.
  - ``r``: Numerical rank of the design, read as [`cross_sectional_rank`](@ref) reads it.
  - ``\\epsilon``: Machine epsilon of the element type of ``\\mathbf{Q}``.

The tolerance ``\\sqrt{\\epsilon}`` absorbs the round-off of ``h_{i}``, which is of order ``m \\epsilon`` for ``m`` rows. A row whose leverage is below one by more than that keeps a residual variance that the data can estimate.

# Algorithm

 1. Scale the rows of `A` by `sqrt.(w)`, giving the weighted design `D`, and put the column `sqrt.(w)` in front of it when `intercept` is `true`.
 2. Read the numerical rank `r` of `D` through [`cross_sectional_rank`](@ref), which answers zero for a design with no row or no column. Return an all-false mask at rank zero.
 3. Take the column-pivoted `LinearAlgebra.qr` of `D` and the first `r` columns of its `Q`, and sum the squares of each row, giving the leverage `h`.
 4. Mark each row whose `1 - h` is at most `sqrt(eps)`.

# Arguments

  - `A::MatNum`: Exposures of the eligible assets, `eligible assets × factors`, not demeaned.
  - `w::VecNum`: Cross-sectional weights of the eligible assets.
  - `intercept::Bool`: Whether the fit has an intercept.

# Returns

  - `mark::BitVector`: Leverage-one mask, one entry per eligible asset.

# Related

  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_rank`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function cross_sectional_leverage_one(A::MatNum, w::VecNum, intercept::Bool)::BitVector
    sq = sqrt.(w)
    D = intercept ? hcat(sq, A .* sq) : A .* sq
    r = cross_sectional_rank(D)
    if iszero(r)
        return falses(size(D, 1))
    end
    F = LinearAlgebra.qr(D, LinearAlgebra.ColumnNorm())
    h = vec(sum(abs2, F.Q * Matrix{eltype(F.R)}(LinearAlgebra.I, size(D, 1), r); dims = 2))
    return BitVector(one.(h) .- h .<= sqrt(eps(real(eltype(h)))))
end
"""
    leverage_one_nan(h1::Nothing, A::MatNum) -> MatNum
    leverage_one_nan(h1::AbstractMatrix{Bool}, A::MatNum) -> MatNum

Return a history of a cross-sectional fit with `NaN` at every pair whose leverage is one.

A pair that `h1` marks gives no observation of its idiosyncratic return: the fit reproduced its return, so its residual is zero by construction, and so is the variance that reads that residual. A reader that treats such a pair as data reads the design, not the asset. The function writes `NaN` over the pair, so every reader that skips a value that is not finite leaves the pair out. It never changes `A`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: a regression built by hand states no mask, so return `A`.
 2. A mask: check its size, and return `A` when it marks no pair. Otherwise copy `A` into the type of a square root of its elements, which holds a `NaN` and the division the readers take, and write `NaN` at every marked pair.

# Arguments

  - `h1`: The leverage-one mask of [`CrossSectionalRegression`](@ref), `observations × assets`, or `nothing`.
  - `A`: A history on the axes of `h1`, `observations × assets`.

# Validation

  - `size(h1) == size(A)`. Raises a `DimensionMismatch`.

# Returns

  - `A::MatNum`: `A` itself when no pair is marked, otherwise a copy with `NaN` at every marked pair.

# Examples

```jldoctest
julia> PortfolioOptimisers.leverage_one_nan(BitMatrix([0 1; 0 0]), [0.1 0.0; 0.3 0.4])
2×2 Matrix{Float64}:
 0.1  NaN
 0.3    0.4
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_leverage_one`](@ref)
"""
function leverage_one_nan(::Nothing, A::MatNum)::MatNum
    return A
end
function leverage_one_nan(h1::AbstractMatrix{Bool}, A::MatNum)::MatNum
    @argcheck(size(h1) == size(A),
              DimensionMismatch("h1 ($(size(h1, 1))×$(size(h1, 2))) must match the history ($(size(A, 1))×$(size(A, 2)))"))
    if !any(h1)
        return A
    end
    B = typeof(sqrt(one(float_if_integer(real(eltype(A)))))).(A)
    B[h1] .= NaN
    return B
end
"""
    cross_sectional_least_squares(cre::CrossSectionalLinearRegression) -> Bool
    cross_sectional_least_squares(cre::CrossSectionalTargetRegression) -> Bool
    cross_sectional_least_squares(tgt::LinearModel) -> Bool
    cross_sectional_least_squares(tgt::GeneralisedLinearModel) -> Bool
    cross_sectional_least_squares(tgt::AbstractRegressionTarget) -> Bool

Return whether the fit of a cross-sectional regression reproduces the return of a pair whose leverage is one.

[`cross_sectional_regression`](@ref) writes an exact zero residual at such a pair when this answers `true`. A least-squares fit projects the returns onto the span of the design, so the pair's residual is zero in exact arithmetic, and the subtraction leaves only its round-off. A generalised linear model with an identity link solves its score equations with the pair's own parameter, which also fits the pair exactly. Under any other link the fit reproduces the mean of the pair on the scale of the link, so the residual on the scale of the returns is not zero, and the verb answers `false`. So does a target the verb does not know.

# Arguments

  - `cre`: Cross-sectional regression estimator.
  - `tgt`: Regression target of a [`CrossSectionalTargetRegression`](@ref).

# Returns

  - `ls::Bool`: `true` when the residual of a leverage-one pair is zero by construction.

# Related

  - [`cross_sectional_leverage_one`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`LinearModel`](@ref)
  - [`GeneralisedLinearModel`](@ref)
"""
function cross_sectional_least_squares(::CrossSectionalLinearRegression)::Bool
    return true
end
function cross_sectional_least_squares(cre::CrossSectionalTargetRegression)::Bool
    return cross_sectional_least_squares(cre.tgt)
end
function cross_sectional_least_squares(::LinearModel)::Bool
    return true
end
function cross_sectional_least_squares(tgt::GeneralisedLinearModel)::Bool
    link = length(tgt.args) >= 2 ? tgt.args[2] : GLM.canonicallink(tgt.args[1])
    return isa(link, GLM.IdentityLink)
end
function cross_sectional_least_squares(::AbstractRegressionTarget)::Bool
    return false
end
"""
    cross_sectional_foreach(f, ex::FLoops.Transducers.Executor, idx::AbstractVector) -> nothing
    cross_sectional_foreach(f, ex::FLoops.SequentialEx, idx::AbstractVector) -> nothing

Call `f` on every entry of `idx` under the executor `ex`, and raise the error of the first entry that fails.

The calls must be independent, and each must write a slice of its own. A threaded executor wraps an error in a `TaskFailedException`, and the entry that fails first in time depends on the schedule. So each call records its own error, and after the loop the function raises the recorded error of the lowest position, unchanged. Every executor therefore raises the error that a serial loop raises. Under `SequentialEx` the function runs the serial loop itself, so the error keeps the backtrace of the call that raised it.

# Arguments

  - `f`: Function of one entry of `idx`. Its return value is discarded.
  - $(arg_dict[:ex])
  - `idx`: The entries, in the order of the serial loop.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_exposure_history`](@ref)
  - [`descriptor_scores`](@ref)
"""
function cross_sectional_foreach(f, ex::FLoops.Transducers.Executor,
                                 idx::AbstractVector)::Nothing
    err = Vector{Any}(nothing, length(idx))
    FLoops.@floop ex for j in eachindex(err)
        try
            f(idx[begin + j - 1])
        catch e
            err[j] = e
        end
    end
    j = findfirst(!isnothing, err)
    if !isnothing(j)
        throw(err[j])
    end
    return nothing
end
function cross_sectional_foreach(f, ::FLoops.SequentialEx, idx::AbstractVector)::Nothing
    foreach(f, idx)
    return nothing
end
"""
    cross_sectional_regression(cre::Union{CrossSectionalLinearRegression,
                                          CrossSectionalTargetRegression}, Z::Arr3Num,
                               X::MatNum, W::MatNum) -> CrossSectionalRegression
    cross_sectional_regression(csr::CrossSectionalRegression, args...) -> CrossSectionalRegression

Fit one regression per observation across the assets, or return a fitted result unchanged.

The verb is its own rather than a fourth argument of [`regression`](@ref), because `regression(re::Regression, args...)` is a greedy passthrough that returns its first argument for any trailing arguments, so a time-series result handed to a cross-sectional call would return silently instead of raising.

The weight matrix `W` is an argument rather than a field, because a two-pass weighting scheme calls the estimator **twice** on one design with two different weight matrices, and a policy stored on the estimator would force a second estimator object or a mutation.

A refusal names an observation by its row of `X`. The verb fits [`cross_sectional_block_regression`](@ref) with no observation before `X`.

# Algorithm

 1. Take the eligibility mask through [`cross_sectional_design_mask`](@ref).
 2. For each observation `t`, gather the eligible assets, their weights `w`, their exposures `A` and their returns `y`, and record their count in `n`. The observations are independent problems, and `cre.ex` runs them through [`cross_sectional_foreach`](@ref). Each writes its own row, so every executor gives the same result.
 3. Mark the eligible assets of leverage one in the row `t` of `h1`, through [`cross_sectional_leverage_one`](@ref).
 4. When `cre.intercept` is `true`, take the weighted means `ybar` and `xbar` of `y` and of `A`, and subtract them. An observation with no eligible asset takes zero for both.
 5. Take the factor returns of the observation through [`cross_sectional_coefficients`](@ref), and write them into the row `t` of `f`. An observation with no eligible asset takes zero factor returns under [`CrossSectionalLinearRegression`](@ref), except under [`RankDeficiencyRefusal`](@ref), whose rank test reads an empty design as rank zero and refuses it by name.
 6. When `cre.intercept` is `true`, write `ybar - dot(f[t, :], xbar)` into the entry `t` of `b`.
 7. Subtract the systematic part, through [`cross_sectional_systematic`](@ref), from `X`, giving `eps`.
 8. When [`cross_sectional_least_squares`](@ref) answers `true`, write an exact zero into `eps` at every pair that `h1` marks. The fit reproduced the return of that pair, so zero is its residual in exact arithmetic.

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
    b ┼ nothing
   h1 ┴ 1×3 BitMatrix
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
  - [`cross_sectional_design_mask`](@ref)
  - [`cross_sectional_coefficients`](@ref)
  - [`cross_sectional_systematic`](@ref)
  - [`cross_sectional_block_regression`](@ref)
  - [`regression`](@ref)
"""
function cross_sectional_regression(cre::Union{CrossSectionalLinearRegression,
                                               CrossSectionalTargetRegression}, Z::Arr3Num,
                                    X::MatNum, W::MatNum)::CrossSectionalRegression
    return cross_sectional_block_regression(cre, Z, X, W, 0)
end
function cross_sectional_regression(csr::CrossSectionalRegression, args...)
    return csr
end
"""
    cross_sectional_block_regression(cre::Union{CrossSectionalLinearRegression,
                                                CrossSectionalTargetRegression},
                                     Z::Arr3Num, X::MatNum, W::MatNum,
                                     t0::Integer) -> CrossSectionalRegression
    cross_sectional_block_regression(cre::AbstractCrossSectionalRegressionEstimator,
                                     Z::Arr3Num, X::MatNum, W::MatNum,
                                     t0::Integer) -> CrossSectionalRegression

Fit one regression per observation of a block that follows `t0` fitted observations, as [`cross_sectional_regression`](@ref) fits it.

A refusal names the observation at the row `t` of `X` as observation `t0 + t`. So the carry fold of a [`CrossSectionalFactorPrior`](@ref), which fits the new observations of a step alone, names the observation that the batch fit names. The method for an estimator that the library does not know calls [`cross_sectional_regression`](@ref) on the block, and the estimator numbers the observations of the block itself.

# Algorithm

The algorithm of [`cross_sectional_regression`](@ref), with `t0 + t` as the index of the observation that [`cross_sectional_coefficients`](@ref) receives.

# Arguments

  - `cre`: Cross-sectional regression estimator.
  - `Z::Arr3Num`: Exposure tensor `observations × assets × factors`.
  - `X::MatNum`: Asset returns matrix `observations × assets`.
  - `W::MatNum`: Cross-sectional weights matrix `observations × assets`.
  - `t0::Integer`: Number of fitted observations before the block.

# Validation

  - The rules of [`cross_sectional_regression`](@ref).

# Returns

  - `csr::CrossSectionalRegression`: The fit of the block.

# Related

  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_coefficients`](@ref)
  - [`cross_sectional_live_regression`](@ref)
"""
function cross_sectional_block_regression(cre::AbstractCrossSectionalRegressionEstimator,
                                          Z::Arr3Num, X::MatNum, W::MatNum, ::Integer)
    return cross_sectional_regression(cre, Z, X, W)
end
function cross_sectional_block_regression(cre::Union{CrossSectionalLinearRegression,
                                                     CrossSectionalTargetRegression},
                                          Z::Arr3Num, X::MatNum, W::MatNum,
                                          t0::Integer)::CrossSectionalRegression
    act = cross_sectional_design_mask(Z, X, W)
    # The coefficients answer a weighted least squares, so the working type is the inputs'
    # own promotion, widened to a float only when it is an integer: an integer panel
    # regresses in `Float64`, and a `Float32` panel in `Float32`. A `Rational` panel keeps
    # its type in `f`, but it is not fitted exactly, issue #1352. The linear member scales
    # by `sqrt.(w)`, which is a float, and the default target `LinearModel` fits in floating
    # point too.
    Ts = promote_type(real(eltype(Z)), real(eltype(X)), real(eltype(W)))
    Tf = float_if_integer(Ts)
    K = size(Z, 3)
    f = zeros(Tf, size(X, 1), K)
    b = cre.intercept ? zeros(Tf, size(X, 1)) : nothing
    n = zeros(Int, size(X, 1))
    # A `BitMatrix` packs 64 entries into one word, so two threads that write rows of one word
    # would race. Each fit writes its row into a `Matrix{Bool}`, whose entries are bytes.
    h1 = zeros(Bool, size(X))
    cross_sectional_foreach(cre.ex, axes(X, 1)) do t
        idx = findall(view(act, t, :))
        n[t] = length(idx)
        w = Tf.(view(W, t, idx))
        A = Tf.(view(Z, t, idx, :))
        y = Tf.(view(X, t, idx))
        h1[t, idx] = cross_sectional_leverage_one(A, w, cre.intercept)
        ybar = zero(Tf)
        xbar = zeros(Tf, K)
        if cre.intercept
            sw = sum(w)
            if sw > zero(sw)
                ybar = LinearAlgebra.dot(w, y) / sw
                xbar .= vec(transpose(A) * w) ./ sw
            end
            y = y .- ybar
            A = A .- transpose(xbar)
        end
        fi = cross_sectional_coefficients(cre, A, y, w, t0 + t)
        f[t, :] = fi
        if cre.intercept
            b[t] = ybar - LinearAlgebra.dot(fi, xbar)
        end
    end
    eps = X - cross_sectional_systematic(f, b, Z)
    # A fit that is not least squares keeps the residual of a marked pair, and only marks it.
    eps[h1 .& cross_sectional_least_squares(cre)] .= zero(eltype(eps))
    return CrossSectionalRegression(; f = f, eps = eps, n = n, b = b, h1 = BitMatrix(h1))
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

  - `r2::VecNum`: Coefficient of determination of every observation, of length `observations`, in the type a division of the inputs lands in.

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
    # The ratio keeps the type of the inputs, so a `Float32` fit answers a `Float32` ratio
    # rather than one widened by a `Float64` `NaN`.
    Ts = promote_type(real(eltype(Xh)), real(eltype(X)), real(eltype(W)))
    Tf = float_if_integer(Ts)
    r2 = fill(convert(Tf, NaN), size(X, 1))
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
       UncheckedSolve, MinimumNormSolve, DependentColumnDrop, ZeroUnseenMember,
       SolvedUnseenMember, cross_sectional_regression, cross_sectional_r2,
       mean_cross_sectional_r2
public AbstractUnseenMemberRule
