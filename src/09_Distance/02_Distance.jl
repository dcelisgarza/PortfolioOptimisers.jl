"""
$(DocStringExtensions.TYPEDEF)

Pairs a distance algorithm with an optional integer power, and applies it to a correlation matrix or to the data.

This is the estimator every clustering, network and phylogeny routine reaches for a distance matrix. `alg` chooses the transform; `power` raises the quantity that transform is built on to the integer power ``p``, which sharpens the contrast between a strong relationship and a weak one.

!!! note

    `power = 1` reproduces the base distance exactly, for every algorithm. It is the neutral position of the knob, not a way to select the generalised estimator. Only ``p \\geq 2`` changes the result. `power = nothing` and `power = 1` are kept apart by dispatch alone, so that the base case never raises a matrix to a power.

# Mathematical definition

The four correlation-based algorithms (see [`RhoDistanceAlgorithm`](@ref)) raise the *correlation* to ``p`` inside their own base formula.

```math
\\begin{align}
_{g}d_{i,\\,j}^{\\mathrm{S}} &= \\sqrt{\\mathrm{clamp}\\left(s\\left(1 - \\rho_{i,\\,j}^{p}\\right),\\, 0,\\, 1\\right)}\\\\
_{g}d_{i,\\,j}^{\\mathrm{SA}} &= \\sqrt{\\mathrm{clamp}\\left(1 - \\lvert\\rho_{i,\\,j}\\rvert^{p},\\, 0,\\, 1\\right)}\\\\
_{g}d_{i,\\,j}^{\\mathrm{L}} &= \\max\\left(-\\log{\\lvert\\rho_{i,\\,j}\\rvert^{p}},\\, 0\\right)\\\\
_{g}d_{i,\\,j}^{\\mathrm{C}} &= \\sqrt{\\mathrm{clamp}\\left(1 - \\rho_{i,\\,j}^{p},\\, 0,\\, 1\\right)}\\\\
    s &= \\begin{cases}
        1/2 & \\text{if } p \\mod 2 \\neq 0\\\\
        1 & \\text{otherwise}
        \\end{cases}\\,,
\\end{align}
```

[`VariationInfoDistance`](@ref) has no correlation to raise, so it raises the *distance* itself.

```math
\\begin{align}
_{g}d_{i,\\,j}^{\\mathrm{VI}} &= \\left(d_{i,\\,j}^{\\mathrm{VI}}\\right)^{p}\\,,
\\end{align}
```

Where:

  - ``_{g}d_{i,\\,j}``: Generalised distance between assets ``i`` and ``j``, superscripted by the algorithm: [`SimpleDistance`](@ref) (S), [`SimpleAbsoluteDistance`](@ref) (SA), [`LogDistance`](@ref) (L), [`CorrelationDistance`](@ref) (C), [`VariationInfoDistance`](@ref) (VI).
  - ``d_{i,\\,j}``: Base distance computed using the specified distance algorithm.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.
  - ``p``: Integer power.
  - ``s``: Scaling factor of [`SimpleDistance`](@ref) alone (``s = 1/2`` if ``p \\bmod 2 \\neq 0``, else ``s = 1``).

The two cases of ``s`` are not a convention. ``s`` is the normalisation that keeps the radicand inside ``[0,\\,1]`` over the reachable range of ``\\rho_{i,\\,j}^{p}``, so it is ``1 / (1 - m)``, where ``m`` is the smallest value ``\\rho_{i,\\,j}^{p}`` can take. An odd ``p`` keeps the sign, so ``m = -1`` and ``s = 1/2``. An even ``p`` cannot be negative, so ``m = 0`` and ``s = 1``.

The clamp is inert for the first two algorithms at every ``p``: ``s(1 - \\rho_{i,\\,j}^{p})`` and ``1 - \\lvert\\rho_{i,\\,j}\\rvert^{p}`` never leave ``[0,\\,1]``. It **binds** for [`CorrelationDistance`](@ref) at every odd ``p``, where ``\\rho_{i,\\,j}^{p}`` keeps its sign and the radicand runs over ``[0,\\,2]``; that algorithm's own docstring measures the truncation.

[`CanonicalDistance`](@ref) is a redirect and owns no formula. It forwards `power` to the algorithm it selects.

!!! warning "`power` raises two different quantities"

    The field is one field, and the reader must not assume one meaning. For a correlation-based algorithm it raises the **correlation**, inside the transform. For [`VariationInfoDistance`](@ref) there is no correlation to raise, so it raises the **distance** the algorithm returns. So ``_{g}d^{\\mathrm{S}}`` at ``p = 2`` is *not* ``\\left(d^{\\mathrm{S}}\\right)^{2}``, while ``_{g}d^{\\mathrm{VI}}`` at ``p = 2`` *is* ``\\left(d^{\\mathrm{VI}}\\right)^{2}``.

The source carries the four base formulas alone. Section 6.2 of the reference below ends at the tail dissimilarity, so the ``p`` generalisation and the scaling ``s`` are this library's own extension of it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Distance(;
        power::Option{<:Integer} = nothing,
        alg::AbstractDistanceAlgorithm = SimpleDistance()
    ) -> Distance

Keywords correspond to the struct's fields.

!!! note "Why the default `alg` is `SimpleDistance`"

    [`CanonicalDistance`](@ref) picks an algorithm from the covariance estimator, and falls back to [`SimpleDistance`](@ref) when the estimator carries no preference. A bare `Distance()` also serves the matrix entry point `distance(de, rho)`, which holds no estimator to pick from. The two therefore agree except on the estimators [`CanonicalDistance`](@ref) treats specially. Library entry points that always hold a covariance estimator default to `Distance(; alg = CanonicalDistance())` instead, so that the special cases are honoured.

## Validation

  - $(val_dict[:dopower])

# Examples

```jldoctest
julia> Distance()
Distance
  power ┼ nothing
    alg ┴ SimpleDistance()
```

# Related

  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`VariationInfoDistance`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 6.2.
"""
@concrete struct Distance <: AbstractDistanceEstimator
    """
    $(field_dict[:dopower])
    """
    power
    """
    $(field_dict[:dalg])
    """
    alg
    function Distance(power::Option{<:Integer}, alg::AbstractDistanceAlgorithm)
        if !isnothing(power)
            @argcheck(one(power) <= power, DomainError)
        end
        return new{typeof(power), typeof(alg)}(power, alg)
    end
end
function Distance(; power::Option{<:Integer} = nothing,
                  alg::AbstractDistanceAlgorithm = SimpleDistance())::Distance
    return Distance(power, alg)
end
"""
    const RhoDistanceAlgorithm = Union{SimpleDistance, SimpleAbsoluteDistance,
                                       LogDistance, CorrelationDistance}

Union of the correlation-based distance algorithms: those whose distance matrix is a pure function of a correlation matrix via [`_dist_from_cor`](@ref). Excludes [`VariationInfoDistance`](@ref) (information-theoretic, computed from the data matrix) and [`CanonicalDistance`](@ref) (a redirect that selects one of the others from the covariance estimator).

# Related

  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
"""
const RhoDistanceAlgorithm = Union{SimpleDistance, SimpleAbsoluteDistance, LogDistance,
                                   CorrelationDistance}
"""
    _absguard(rho::MatNum) -> MatNum

Supply the magnitude of `rho` to the two algorithms that are defined on it, without allocating when the magnitude is already `rho`.

This is an **allocation guard, not a branch in the mathematics**. `abs.(rho)` equals `rho` entry for entry whenever no entry of `rho` is negative, so both arms return the same numbers for every input, `-0.0` included; the guard only decides whether a second matrix is built. Shared by [`SimpleAbsoluteDistance`](@ref) and [`LogDistance`](@ref).

# Algorithm

 1. Test every entry of `rho` against zero.
 2. When no entry is negative, return `rho` itself, the same object the caller passed.
 3. Otherwise return `abs.(rho)`, a new matrix.

# Arguments

  - $(arg_dict[:rho])

# Returns

  - `rho::MatNum`: The magnitude of the argument. It is the argument itself when the argument holds no negative entry.

# Details

  - The test reads the **whole** matrix, so one negative entry allocates the copy for all of them. That is the intended reading of the two algorithms, which take the magnitude of every entry.
  - `NaN` compares false against zero, so a matrix holding one takes the allocating arm. `abs(NaN)` is `NaN`, so that entry is `NaN` on either arm.

# Related

  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`_dist_from_cor`](@ref)
"""
function _absguard(rho::MatNum)
    return all(x -> zero(x) <= x, rho) ? rho : abs.(rho)
end
"""
    _as_correlation(rho::MatNum, sym::Symbol = :rho) -> MatNum

Coerce a square matrix to a correlation matrix, converting it from a covariance matrix when its diagonal says it is one.

The **value of the diagonal decides, never the type**. A matrix whose diagonal is all ones is already a correlation matrix and is returned as the same object; any other diagonal is read as the variances of a covariance matrix. This is the same test the matrix processing pipeline applies, so the two layers agree on what a correlation matrix is. The square-matrix check runs here, once, for every correlation-based algorithm's matrix entry point.

# Algorithm

 1. Check that `rho` is square, reporting the failure under the name `sym`.
 2. Read the diagonal of `rho` into `s`. `LinearAlgebra.diag` allocates, so `rho` is never written to.
 3. When every entry of `s` is one, return `rho` itself. Steps 4 and 5 do not run.
 4. Otherwise replace `s` with its square roots, giving the standard deviations.
 5. Divide `rho` by the outer product of `s` with `StatsBase.cov2cor`, giving a new correlation matrix.

# Arguments

  - `rho`: Correlation matrix `assets × assets`, or the covariance matrix to convert.
  - `sym`: Name to report the square-matrix failure under.

# Validation

  - `rho` is square.

# Returns

  - $(ret_dict[:rho])

# Details

  - The argument is never mutated on either route. Step 2 copies the diagonal, and step 5 builds a new matrix.
  - The conversion round-trips: the correlation of a covariance matrix built from a correlation matrix and a vector of standard deviations is that correlation matrix again.

# Related

  - [`distance`](@ref)
  - [`_dist_from_cor`](@ref)
"""
function _as_correlation(rho::MatNum, sym::Symbol = :rho)
    assert_matrix_issquare(rho, sym)
    s = LinearAlgebra.diag(rho)
    if any(!isone, s)
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return rho
end
"""
    _dist_from_cor(alg::RhoDistanceAlgorithm, power::Option{<:Integer}, rho::MatNum) -> MatNum

Turn a correlation matrix into a distance matrix, for one of the four correlation-based algorithms.

This is the shared kernel behind the [`distance`](@ref) and [`cor_and_dist`](@ref) entry points: they differ only in how they obtain `rho`, never in the transform they apply to it. Eight methods cover the four algorithms of [`RhoDistanceAlgorithm`](@ref) at each of the two `power` cases.

# Mathematical definition

[`Distance`](@ref) states the eight closed forms and the scaling ``s``. Each algorithm's own docstring states the base case and the range it is defined on.

# Algorithm

`power` selects the method, so the base case never raises `rho` to a power.

 1. [`SimpleAbsoluteDistance`](@ref) and [`LogDistance`](@ref) replace `rho` with its magnitude through [`_absguard`](@ref). [`SimpleDistance`](@ref) and [`CorrelationDistance`](@ref) do not, and read the signed correlation.
 2. When `power` is an `Integer`, raise `rho` to it entry by entry. When `power` is `nothing`, leave `rho` as it is.
 3. [`SimpleDistance`](@ref) scales ``1 - \\rho`` by `1//2` for an odd `power` and by `1//1` for an even one, and by `1//2` in the base case. The other three apply no scaling.
 4. The three square-root algorithms clamp the radicand into ``[0,\\,1]`` with `clamp!` and take its square root. [`LogDistance`](@ref) instead takes ``-\\log`` and floors the result at zero with `max`.

# Arguments

  - $(arg_dict[:dalg])
  - $(arg_dict[:dopower])
  - $(arg_dict[:rho])

# Returns

  - $(ret_dict[:Ddist])

# Details

  - Every method allocates its own result, and `clamp!` writes only into that allocation. `rho` is never mutated.
  - The scale of step 3 is a `Rational`, so the element type of `rho` is carried through: a `Float32` correlation matrix gives a `Float32` distance matrix, as it does under the other three algorithms.
  - [`CanonicalDistance`](@ref) and [`VariationInfoDistance`](@ref) never reach this kernel. The first is a redirect that resolves to one of the four before the call, and the second reads the data matrix and holds no correlation.

# Related

  - [`RhoDistanceAlgorithm`](@ref)
  - [`_absguard`](@ref)
  - [`_as_correlation`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function _dist_from_cor(::SimpleDistance, ::Nothing, rho::MatNum)
    return sqrt.(clamp!((one(eltype(rho)) .- rho) * (1//2), zero(eltype(rho)),
                        one(eltype(rho))))
end
function _dist_from_cor(::SimpleDistance, power::Integer, rho::MatNum)
    scale = isodd(power) ? 1//2 : 1//1
    return sqrt.(clamp!((one(eltype(rho)) .- rho .^ power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
end
function _dist_from_cor(::SimpleAbsoluteDistance, ::Nothing, rho::MatNum)
    rho = _absguard(rho)
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function _dist_from_cor(::SimpleAbsoluteDistance, power::Integer, rho::MatNum)
    rho = _absguard(rho)
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function _dist_from_cor(::LogDistance, ::Nothing, rho::MatNum)
    return max.(-log.(_absguard(rho)), zero(eltype(rho)))
end
function _dist_from_cor(::LogDistance, power::Integer, rho::MatNum)
    return max.(-log.(_absguard(rho) .^ power), zero(eltype(rho)))
end
function _dist_from_cor(::CorrelationDistance, ::Nothing, rho::MatNum)
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function _dist_from_cor(::CorrelationDistance, power::Integer, rho::MatNum)
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ power, zero(eltype(rho)),
                        one(eltype(rho))))
end
"""
    distance(de::Distance{<:Any,
                          <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                                  <:CorrelationDistance, <:CanonicalDistance}},
             ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the correlation matrix with `ce` and `X`, and transform it into a distance matrix with the algorithm `de` names.

This is the data entry point of the correlation-based family. [`cor_and_dist`](@ref) is the same computation with the correlation returned alongside the distance, so a caller that needs both pays for the correlation once.

# Algorithm

 1. Compute the correlation matrix from `ce` and `X` with `Statistics.cor`, along `dims`. `kwargs` are forwarded to it, and it is what refuses a `dims` outside ``(1,\\, 2)``.
 2. Transform that correlation matrix with [`_dist_from_cor`](@ref), under `de.alg` and `de.power`.

A [`CanonicalDistance`](@ref) `de` takes one step first: it rebuilds `de` with [`SimpleDistance`](@ref), carrying `de.power` over, because a `ce` that is a plain `StatsBase.CovarianceEstimator` carries no preference. The four estimators that do carry one have their own methods; see the redirect table on [`CanonicalDistance`](@ref).

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `ce`: Covariance estimator.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:Ddist])

# Details

  - `dims` is enforced by `Statistics.cor`, not by this method. A `dims` outside ``(1,\\, 2)`` raises a `DomainError` from there.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`_dist_from_cor`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::Distance{<:Any, <:RhoDistanceAlgorithm},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return _dist_from_cor(de.alg, de.power, Statistics.cor(ce, X; dims = dims, kwargs...))
end
function distance(de::Distance{<:Any, <:CanonicalDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = SimpleDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    const LTDCov_AllInternalLTDCov = Union{<:LowerTailDependenceCovariance,
                                           <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance}}

Alias for all internal lower tail dependence covariance estimator types.

Matches [`LowerTailDependenceCovariance`](@ref) or any [`PortfolioOptimisersCovariance`](@ref) wrapping it. Used internally for dispatch in distance computation.

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`DistCov_AllInternalDistCov`](@ref)
"""
const LTDCov_AllInternalLTDCov = Union{<:LowerTailDependenceCovariance,
                                       <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance}}
"""
    distance(de::Distance{Nothing, <:VariationInfoDistance}, ::Any, X::MatNum;
             dims::Int = 1, kwargs...)
    distance(de::Distance{<:Integer, <:VariationInfoDistance}, ::Any, X::MatNum;
             dims::Int = 1, kwargs...)

Compute the variation of information distance matrix from the data matrix alone.

This is the one algorithm of the family that reads `X` rather than a correlation matrix, so the covariance estimator is a placeholder that the method ignores. It captures a non-linear relationship that no correlation coefficient sees.

!!! warning "`power` raises the distance here, not the correlation"

    The two methods above are the two `power` cases, and they are the reason the trap exists. For a correlation-based algorithm `power` raises the **correlation** inside the transform. There is no correlation here, so `power` raises the **distance** the algorithm returns: the result is `variation_info(...) .^ de.power`. One field, two quantities.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Read `de.alg.bins` and `de.alg.normalise` off the algorithm, and pass both to [`variation_info`](@ref), which builds the joint histograms and forms the distance. [`VariationInfoDistance`](@ref) states those steps.
 3. When `de.power` is an `Integer`, raise the distance matrix of step 2 to it entry by entry. When `de.power` is `nothing`, return that matrix as it stands.

# Arguments

  - `de`: Distance estimator carrying the [`VariationInfoDistance`](@ref) algorithm.
  - `::Any`: Covariance estimator placeholder for API compatibility. It is ignored.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments. They are ignored.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:Ddist])

# Details

  - `dims` is enforced by [`dims_oriented`](@ref). A `dims` outside ``(1,\\, 2)`` raises a `DomainError` from there.
  - `bins` and `normalise` come from the algorithm, never from a keyword. [`CanonicalDistance`](@ref) is what copies them off a [`MutualInfoCovariance`](@ref).

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`variation_info`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::Distance{Nothing, <:VariationInfoDistance}, ::Any, X::MatNum;
                  dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    return variation_info(X, de.alg.bins, de.alg.normalise)
end
function distance(de::Distance{<:Integer, <:VariationInfoDistance}, ::Any, X::MatNum;
                  dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    return variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
"""
    distance(::Distance{<:Any,
                        <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                                <:CorrelationDistance, <:CanonicalDistance}},
             rho::MatNum, args...; kwargs...)

Compute the distance matrix from a correlation matrix, or from a covariance matrix.

This is the matrix entry point of the correlation-based family, for a caller that already holds the matrix and needs no covariance estimator. The value of the diagonal decides which of the two it was given; see [`_as_correlation`](@ref).

# Algorithm

 1. Coerce `rho` to a correlation matrix with [`_as_correlation`](@ref), which also checks that it is square.
 2. Transform that correlation matrix with [`_dist_from_cor`](@ref), under `de.alg` and `de.power`.

A [`CanonicalDistance`](@ref) `de` takes one step first: it rebuilds `de` with [`SimpleDistance`](@ref), carrying `de.power` over. There is no covariance estimator here to select from, so the redirect table cannot apply and the fallback of the table is taken.

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `rho`: Correlation or covariance matrix.

  - `args...`: Additional arguments (ignored).

  - `kwargs...`: Additional keyword arguments. They are ignored.

# Validation

  - `rho` is square.

# Returns

  - $(ret_dict[:Ddist])

# Details

  - The distance is the one the algorithm defines, and it is not Euclidean under any of the four. [`SimpleDistance`](@ref) and [`SimpleAbsoluteDistance`](@ref) return an angular distance, and [`LogDistance`](@ref) an unbounded dissimilarity.
  - A covariance matrix is converted with `StatsBase.cov2cor`, and the conversion allocates rather than writing into the argument.
  - `args` and `kwargs` exist so that this method and the `ce`-and-`X` method above take the same call. Neither is read.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`_as_correlation`](@ref)
  - [`_dist_from_cor`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::Distance{<:Any, <:RhoDistanceAlgorithm}, rho::MatNum, args...;
                  kwargs...)
    return _dist_from_cor(de.alg, de.power, _as_correlation(rho))
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, rho::MatNum, args...; kwargs...)
    return distance(Distance(; power = de.power, alg = SimpleDistance()), rho; kwargs...)
end
"""
    cor_and_dist(de::Distance{<:Any,
                              <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance,
                                      <:LogDistance, <:CorrelationDistance,
                                      <:VariationInfoDistance, <:CanonicalDistance}},
                 ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the correlation matrix and the distance matrix together, from one pass over the data.

It returns the same `D` that [`distance`](@ref) returns for the same arguments; that agreement is the claim of having two entry points. Take this one when both matrices are wanted, because the correlation-based family then computes the correlation once instead of twice.

# Algorithm

Which of the three routes runs is decided by `de.alg`.

 1. A correlation-based algorithm computes the correlation matrix with `Statistics.cor`, then transforms that same matrix with [`_dist_from_cor`](@ref). This is the route that saves the second correlation.
 2. [`VariationInfoDistance`](@ref) checks `dims` with [`assert_dims`](@ref), computes the correlation matrix for the caller, and calls [`distance`](@ref) for the distance. The distance reads `X`, not the correlation, so nothing is shared between the two.
 3. [`CanonicalDistance`](@ref) rebuilds `de` with the algorithm its redirect table names for `ce`, then re-enters at route 1 or route 2. See [`distance`](@ref) for the table and for the fields the rebuild copies.

# Arguments

  - `de`: Distance estimator.
  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:rho])
  - $(ret_dict[:Ddist])

# Details

  - Route 2 checks `dims` itself with [`assert_dims`](@ref). Route 1 leaves the check to `Statistics.cor`, which raises the same `DomainError`.
  - The correlation returned is the one `ce` computes, untransformed. It is not the magnitude that [`SimpleAbsoluteDistance`](@ref) and [`LogDistance`](@ref) take, nor the power of it that a `de.power` raises.

# Related

  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`_dist_from_cor`](@ref)
"""
function cor_and_dist(de::Distance{<:Any, <:RhoDistanceAlgorithm},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return rho, _dist_from_cor(de.alg, de.power, rho)
end
function cor_and_dist(de::Distance{<:Any, <:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    assert_dims(dims)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return rho, distance(de, ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                      X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power,
                                 alg = VariationInfoDistance(; bins = ce.bins,
                                                             normalise = ce.normalise)), ce,
                        X; dims = dims, kwargs...)
end
"""
    const AllInternalMutualInfoCov = Union{<:PortfolioOptimisersCovariance{<:MutualInfoCovariance}}

Alias for all internal mutual information covariance wrapper types.

Matches any [`PortfolioOptimisersCovariance`](@ref) wrapping a [`MutualInfoCovariance`](@ref). Used internally for dispatch in canonical distance computation.

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`DistCov_AllInternalDistCov`](@ref)
"""
const AllInternalMutualInfoCov = Union{<:PortfolioOptimisersCovariance{<:MutualInfoCovariance}}
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::AllInternalMutualInfoCov, X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power,
                                 alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                             normalise = ce.ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::LTDCov_AllInternalLTDCov, X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power, alg = LogDistance()), ce, X;
                        dims = dims, kwargs...)
end
"""
    const DistCov_AllInternalDistCov = Union{<:DistanceCovariance,
                                             <:PortfolioOptimisersCovariance{<:DistanceCovariance}}

Alias for all internal distance covariance estimator types.

Matches [`DistanceCovariance`](@ref) or any [`PortfolioOptimisersCovariance`](@ref) wrapping it. Used internally for dispatch in canonical distance computation.

# Related

  - [`DistanceCovariance`](@ref)
  - [`LTDCov_AllInternalLTDCov`](@ref)
"""
const DistCov_AllInternalDistCov = Union{<:DistanceCovariance,
                                         <:PortfolioOptimisersCovariance{<:DistanceCovariance}}
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::DistCov_AllInternalDistCov, X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power, alg = CorrelationDistance()), ce, X;
                        dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    return cor_and_dist(Distance(; power = de.power, alg = SimpleDistance()), ce, X;
                        dims = dims, kwargs...)
end
"""
    distance(de::Distance{<:Any, <:CanonicalDistance},
             ce::Union{<:MutualInfoCovariance,
                       <:AllInternalMutualInfoCov,
                       <:LTDCov_AllInternalLTDCov,
                       <:DistCov_AllInternalDistCov},
             X::MatNum; dims::Int = 1, kwargs...)

Rebuild `de` with the distance algorithm that the covariance estimator's own range calls for, and call [`distance`](@ref) again.

The redirect owns no formula. It exists so that a codependence measure reaches the transform its range needs: a signed correlation must be halved, a mutual information has no correlation to transform at all, and a tail dependence coefficient wants an unbounded distance. `de.power` is carried over unchanged onto the algorithm that is selected.

| Covariance estimator                                    | Algorithm selected              | Read from `ce`                  |
|:------------------------------------------------------- |:------------------------------- |:------------------------------- |
| [`MutualInfoCovariance`](@ref)                          | [`VariationInfoDistance`](@ref) | `ce.bins`, `ce.normalise`       |
| [`PortfolioOptimisersCovariance`](@ref) wrapping it     | [`VariationInfoDistance`](@ref) | `ce.ce.bins`, `ce.ce.normalise` |
| [`LowerTailDependenceCovariance`](@ref), wrapped or not | [`LogDistance`](@ref)           | nothing                         |
| [`DistanceCovariance`](@ref), wrapped or not            | [`CorrelationDistance`](@ref)   | nothing                         |
| any other `StatsBase.CovarianceEstimator`               | [`SimpleDistance`](@ref)        | nothing                         |

# Algorithm

 1. Select the row of the table above by the type of `ce`. Dispatch does the selection, so a wrapper reaches the same row as the estimator it wraps.
 2. Build a fresh [`Distance`](@ref) carrying `de.power` and the algorithm of that row.
 3. On the two mutual-information rows, **copy `bins` and `normalise` off `ce` onto the new [`VariationInfoDistance`](@ref)**, one field level deeper for the wrapper. Without the copy the algorithm would take its own defaults, and the distance would be a different number.
 4. Call [`distance`](@ref) with the rebuilt estimator, the same `ce`, and the same `X`, `dims` and `kwargs`.

# Arguments

  - `de`: Distance estimator carrying the [`CanonicalDistance`](@ref) algorithm.
  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the selected algorithm.

# Returns

  - $(ret_dict[:Ddist])

# Details

  - Step 3 is invisible to a reader of the signature, and it is load-bearing. A [`MutualInfoCovariance`](@ref) built with a non-default `bins` gives a different distance matrix from one built with the default, and the redirect is what carries that setting across.
  - The last row is the fallback, and it is also what a bare [`Distance`](@ref) with [`SimpleDistance`](@ref) gives. The two agree on every estimator outside the table.
  - [`cor_and_dist`](@ref) carries the same table, over the same five rows.

# Related

  - [`Distance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`LowerTailDependenceCovariance`](@ref)
  - [`DistanceCovariance`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power,
                             alg = VariationInfoDistance(; bins = ce.bins,
                                                         normalise = ce.normalise)), ce, X;
                    dims = dims, kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::AllInternalMutualInfoCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power,
                             alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                         normalise = ce.ce.normalise)), ce,
                    X; dims = dims, kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::LTDCov_AllInternalLTDCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = LogDistance()), ce, X; dims = dims,
                    kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::DistCov_AllInternalDistCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = CorrelationDistance()), ce, X;
                    dims = dims, kwargs...)
end

export Distance, distance, cor_and_dist
