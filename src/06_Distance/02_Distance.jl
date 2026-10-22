"""
$(DocStringExtensions.TYPEDEF)

Pairs a distance algorithm with an optional integer power, and applies it to a correlation matrix or to the data.

Every clustering, network and phylogeny routine uses this estimator to get a distance matrix. `alg` selects the transform. `power` raises the input of that transform to the integer power ``p``.

!!! note

    `power = 1` gives the base distance exactly, for every algorithm. Only ``p \\geq 2`` changes the result. `power = nothing` and `power = 1` select different methods, so the base case never raises a matrix to a power.

# Mathematical definition

The four correlation-based algorithms of [`RhoDistanceAlgorithm`](@ref) raise the correlation to ``p`` inside their base formula.

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

[`VariationInfoDistance`](@ref) has no correlation to raise, so it raises the distance itself.

```math
\\begin{align}
_{g}d_{i,\\,j}^{\\mathrm{VI}} &= \\left(d_{i,\\,j}^{\\mathrm{VI}}\\right)^{p}\\,,
\\end{align}
```

Where:

  - ``_{g}d_{i,\\,j}``: Generalised distance between assets ``i`` and ``j``. The superscript names the algorithm: S for [`SimpleDistance`](@ref), SA for [`SimpleAbsoluteDistance`](@ref), L for [`LogDistance`](@ref), C for [`CorrelationDistance`](@ref) and VI for [`VariationInfoDistance`](@ref).
  - ``d_{i,\\,j}``: Base distance of the algorithm.
  - $(math_dict[:rho_ij])
  - ``p``: Integer power.
  - ``s``: Scale factor of [`SimpleDistance`](@ref) only. ``s = 1/2`` for an odd ``p``, and ``s = 1`` for an even ``p``.

The two values of ``s`` come from the range of ``\\rho_{i,\\,j}^{p}``. ``s`` keeps the radicand inside ``[0,\\,1]``, so ``s = 1 / (1 - m)``, where ``m`` is the smallest value that ``\\rho_{i,\\,j}^{p}`` can take. An odd ``p`` keeps the sign, so ``m = -1`` and ``s = 1/2``. At an even ``p``, ``\\rho_{i,\\,j}^{p}`` cannot be negative, so ``m = 0`` and ``s = 1``.

The clamp has no effect on the first two algorithms at any ``p``, because ``s(1 - \\rho_{i,\\,j}^{p})`` and ``1 - \\lvert\\rho_{i,\\,j}\\rvert^{p}`` never leave ``[0,\\,1]``. It binds for [`CorrelationDistance`](@ref) at every odd ``p``. There, ``\\rho_{i,\\,j}^{p}`` keeps its sign and the radicand runs over ``[0,\\,2]``. The docstring of [`CorrelationDistance`](@ref) shows the truncation.

[`CanonicalDistance`](@ref) is a redirect and has no formula. It gives `power` to the algorithm that it selects.

!!! warning "`power` raises two different quantities"

    `power` is one field with two meanings. For a correlation-based algorithm, it raises the correlation inside the transform. For [`VariationInfoDistance`](@ref), it raises the distance that the algorithm returns, because there is no correlation. So at ``p = 2``, ``_{g}d^{\\mathrm{S}}`` is not ``\\left(d^{\\mathrm{S}}\\right)^{2}``, but ``_{g}d^{\\mathrm{VI}}`` is ``\\left(d^{\\mathrm{VI}}\\right)^{2}``.

The docstring of each algorithm states its base formula and its original source. The ``p`` generalisation and the scale factor ``s`` are this library's own extension of those formulas.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Distance(;
        power::Option{<:Integer} = nothing,
        alg::AbstractDistanceAlgorithm = SimpleDistance()
    ) -> Distance

Keywords correspond to the struct's fields.

!!! note "Why the default `alg` is `SimpleDistance`"

    [`CanonicalDistance`](@ref) selects an algorithm from the covariance estimator. It uses [`SimpleDistance`](@ref) for every estimator outside its table. A bare `Distance()` also works at the matrix entry point `distance(de, rho)`, which has no estimator to select from. So the two defaults agree, except on the estimators in the table of [`CanonicalDistance`](@ref). A library entry point that always has a covariance estimator uses `Distance(; alg = CanonicalDistance())` as its default, so that each estimator in the table gets its own algorithm.

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
            @argcheck(one(power) <= power,
                      DomainError(power,
                                  "`Distance.power` is $power, but it must be at least `1`. State an integer in `1:typemax(Int)`, or `nothing` for the plain distance."))
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

Union of the correlation-based distance algorithms. [`_dist_from_cor`](@ref) computes the distance matrix of each from a correlation matrix alone. The union does not include [`VariationInfoDistance`](@ref), which reads the data matrix. It does not include [`CanonicalDistance`](@ref) either, which is a redirect that selects one of the other algorithms from the covariance estimator.

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

Supply the magnitude of `rho` to the two algorithms that read it, and allocate no copy when the magnitude is already `rho`.

The guard decides only the allocation, and it does not change the mathematics. `abs.(rho)` equals `rho` entry for entry when no entry of `rho` is negative. So both arms return the same numbers for every input, `-0.0` included. [`SimpleAbsoluteDistance`](@ref) and [`LogDistance`](@ref) use it.

A `NaN` compares false against zero, so a matrix that holds a `NaN` takes the arm that allocates. `abs(NaN)` is `NaN`, so that entry is `NaN` on both arms.

# Algorithm

 1. Compare every entry of `rho` with zero. The test reads the whole matrix, because both algorithms take the magnitude of every entry.
 2. When no entry is negative, return `rho` itself, the same object that the caller gave.
 3. Otherwise return `abs.(rho)`, a new matrix. One negative entry is enough to allocate a copy of the whole matrix.

# Arguments

  - $(arg_dict[:rho])

# Returns

  - `rho::MatNum`: The magnitude of the argument. It is the argument itself when the argument holds no negative entry.

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

Return a square matrix as a correlation matrix, and convert it from a covariance matrix when its diagonal is not all ones.

The values on the diagonal decide, and the type does not. If every diagonal entry is one, the function returns the same matrix. Otherwise it reads the diagonal as the variances of a covariance matrix. The matrix processing pipeline applies the same test, so the two layers agree on what a correlation matrix is. This function does the square-matrix check once, for the matrix entry point of every correlation-based algorithm.

The conversion round-trips. Build a covariance matrix from a correlation matrix and a vector of standard deviations. This function returns that correlation matrix again.

# Algorithm

 1. Check that `rho` is square. A failure names the argument `sym`.
 2. Read the diagonal of `rho` into `s`. `LinearAlgebra.diag` allocates, so the function never writes to `rho`.
 3. When every entry of `s` is one, return `rho` itself. Steps 4 and 5 do not run.
 4. Otherwise replace `s` with its square roots. These are the standard deviations.
 5. Divide `rho` by the outer product of `s` with `StatsBase.cov2cor`. The result is a new correlation matrix.

# Arguments

  - `rho`: Correlation matrix `assets × assets`, or the covariance matrix to convert. The function does not mutate it on either route. Step 2 copies the diagonal, and step 5 builds a new matrix.
  - `sym`: Name of the argument that the square-matrix failure reports.

# Validation

  - `rho` is square.

# Returns

  - $(ret_dict[:rho])

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

[`distance`](@ref) and [`cor_and_dist`](@ref) both call this kernel. The two entry points get `rho` in different ways, and they apply the same transform to it. Eight methods cover the four algorithms of [`RhoDistanceAlgorithm`](@ref) at each of the two `power` cases. Every method allocates its result, and `clamp!` writes only into that result, so `rho` does not change.

# Mathematical definition

[`Distance`](@ref) states the eight closed forms and the scale factor ``s``. The docstring of each algorithm states the base case and the range of the correlation that it reads.

# Algorithm

`power` selects the method, so the base case never raises `rho` to a power.

 1. [`SimpleAbsoluteDistance`](@ref) and [`LogDistance`](@ref) replace `rho` with its magnitude through [`_absguard`](@ref). [`SimpleDistance`](@ref) and [`CorrelationDistance`](@ref) do not, and read the signed correlation.
 2. When `power` is an `Integer`, raise `rho` to it entry by entry. When `power` is `nothing`, leave `rho` as it is.
 3. [`SimpleDistance`](@ref) scales the radicand by `1//2` for an odd `power`, by `1//1` for an even one, and by `1//2` in the base case. The scale is a `Rational`, so the result keeps the element type of `rho`. A `Float32` correlation matrix gives a `Float32` distance matrix, as it does under the other three algorithms. The other three algorithms apply no scale.
 4. The three square-root algorithms clamp the radicand into ``[0,\\,1]`` with `clamp!` and take its square root. [`LogDistance`](@ref) instead takes ``-\\log`` and floors the result at zero with `max`.

# Arguments

  - $(arg_dict[:dalg])
  - $(arg_dict[:dopower])
  - $(arg_dict[:rho])

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`RhoDistanceAlgorithm`](@ref)
  - [`_absguard`](@ref)
  - [`_as_correlation`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`CanonicalDistance`](@ref): never reaches this kernel. It is a redirect that resolves to one of the four before the call.
  - [`VariationInfoDistance`](@ref): never reaches this kernel. It reads the data matrix and holds no correlation.
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
    return max.(.-log.(_absguard(rho)), zero(eltype(rho)))
end
function _dist_from_cor(::LogDistance, power::Integer, rho::MatNum)
    return max.(.-log.(_absguard(rho) .^ power), zero(eltype(rho)))
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

This is the data entry point of the correlation-based family. [`cor_and_dist`](@ref) does the same computation and also returns the correlation, so a caller that needs both matrices computes the correlation once.

# Algorithm

 1. Compute the correlation matrix from `ce` and `X` with `Statistics.cor`, along `dims`. The method gives `kwargs` to `Statistics.cor`, and `Statistics.cor` refuses a `dims` outside ``(1,\\, 2)``.
 2. Transform that correlation matrix with [`_dist_from_cor`](@ref), under `de.alg` and `de.power`.

A [`CanonicalDistance`](@ref) `de` does one step first. It rebuilds `de` with [`SimpleDistance`](@ref) and keeps `de.power`, because the last row of the redirect table gives that algorithm to a plain `StatsBase.CovarianceEstimator`. The estimators of the other rows have their own methods, and the table on [`CanonicalDistance`](@ref) lists them.

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - $(arg_dict[:ce])

  - $(arg_dict[:X])

  - $(arg_dict[:dims])

  - `kwargs...`: Keyword arguments that the method gives to `Statistics.cor`.

# Validation

  - $(val_dict[:dims]) This method does not do the check. `Statistics.cor` raises the `DomainError`.

# Returns

  - $(ret_dict[:Ddist])

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
    return _dist_from_cor(de.alg, de.power,
                          Statistics.cor(library_covariance_estimator(ce), X; dims = dims,
                                         kwargs...))
end
function distance(de::Distance{<:Any, <:CanonicalDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = SimpleDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    const LTDCov_AllInternalLTDCov = Union{<:LowerTailDependenceCovariance,
                                           <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance}}

Alias that matches a [`LowerTailDependenceCovariance`](@ref), bare or in a wrapper.

It matches a [`LowerTailDependenceCovariance`](@ref), or a [`PortfolioOptimisersCovariance`](@ref) that wraps one. The [`CanonicalDistance`](@ref) methods of [`distance`](@ref) and [`cor_and_dist`](@ref) dispatch on it.

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

It is the one algorithm of the family that reads `X` and not a correlation matrix. So the covariance estimator is a placeholder, and the method ignores it.

!!! warning "`power` raises the distance here, not the correlation"

    The two signatures above are the two `power` cases. For a correlation-based algorithm, `power` raises the correlation inside the transform. Here there is no correlation, so `power` raises the distance that the algorithm returns. The result is `variation_info(...) .^ de.power`.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Read `de.alg.bins` and `de.alg.normalise` off the algorithm, and pass both to [`variation_info`](@ref), which builds the joint histograms and forms the distance. [`VariationInfoDistance`](@ref) states those steps.
 3. When `de.power` is an `Integer`, raise the distance matrix of step 2 to it entry by entry. When `de.power` is `nothing`, return that matrix unchanged.

# Arguments

  - `de`: Distance estimator that holds the [`VariationInfoDistance`](@ref) algorithm.
  - `::Any`: Placeholder for the covariance estimator, so that this method takes the same call as the other methods. The method ignores it.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that the method ignores. `bins` and `normalise` come from `de.alg`, never from a keyword.

# Validation

  - $(val_dict[:dims]) This method does not do the check. [`dims_oriented`](@ref) raises the `DomainError`.

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`variation_info`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`CanonicalDistance`](@ref): copies `bins` and `normalise` from a [`MutualInfoCovariance`](@ref) to `de.alg`.
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

This is the matrix entry point of the correlation-based family, for a caller that already has the matrix. The diagonal tells a correlation matrix from a covariance matrix, as [`_as_correlation`](@ref) states. The result is the distance that the algorithm defines. It is not the Euclidean distance between the columns of the data. [`SimpleDistance`](@ref) and [`SimpleAbsoluteDistance`](@ref) return an angular distance, and [`LogDistance`](@ref) returns an unbounded dissimilarity.

# Algorithm

 1. Convert `rho` to a correlation matrix with [`_as_correlation`](@ref), which also checks that it is square.
 2. Transform that correlation matrix with [`_dist_from_cor`](@ref), under `de.alg` and `de.power`.

A [`CanonicalDistance`](@ref) `de` does one step first. It rebuilds `de` with [`SimpleDistance`](@ref) and keeps `de.power`. Here there is no covariance estimator to select from, so the method uses the last row of the redirect table.

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `rho`: Correlation or covariance matrix. [`_as_correlation`](@ref) converts a covariance matrix into a new matrix with `StatsBase.cov2cor`, and does not write into the argument.

  - `args...`: Arguments that the method ignores. They let this method and the `ce`-and-`X` method above take the same call.

  - `kwargs...`: Keyword arguments that the method ignores. They let this method and the `ce`-and-`X` method above take the same call.

# Validation

  - `rho` is square.

# Returns

  - $(ret_dict[:Ddist])

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

Compute the correlation matrix and the distance matrix in one call.

It returns the same `D` that [`distance`](@ref) returns for the same arguments. Use it when the caller needs both matrices, because a correlation-based algorithm then computes the correlation once and not twice. The returned `rho` is the correlation that `ce` computes, with no transform. It is not the magnitude that [`SimpleAbsoluteDistance`](@ref) and [`LogDistance`](@ref) take, and `de.power` does not raise it.

# Algorithm

`de.alg` selects one of three routes.

 1. A correlation-based algorithm computes the correlation matrix with `Statistics.cor`. Then it transforms that same matrix with [`_dist_from_cor`](@ref). On this route the correlation is computed once.
 2. [`VariationInfoDistance`](@ref) checks `dims` with [`assert_dims`](@ref), computes the correlation matrix for the caller, and calls [`distance`](@ref) for the distance. The distance reads `X` and not the correlation, so the two computations share nothing.
 3. [`CanonicalDistance`](@ref) rebuilds `de` with the algorithm that its redirect table names for `ce`. Then it enters route 1 or route 2. The [`CanonicalDistance`](@ref) method of [`distance`](@ref) gives the table and the fields that the rebuild copies.

# Arguments

  - `de`: Distance estimator.
  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that the method gives to `Statistics.cor`.

# Validation

  - $(val_dict[:dims]) Route 2 does the check with [`assert_dims`](@ref). On route 1, `Statistics.cor` does the check and raises the same `DomainError`.

# Returns

  - $(ret_dict[:rho])
  - $(ret_dict[:Ddist])

# Related

  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`_dist_from_cor`](@ref)
"""
function cor_and_dist(de::Distance{<:Any, <:RhoDistanceAlgorithm},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = Statistics.cor(library_covariance_estimator(ce), X; dims = dims, kwargs...)
    return rho, _dist_from_cor(de.alg, de.power, rho)
end
function cor_and_dist(de::Distance{<:Any, <:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    assert_dims(dims)
    rho = Statistics.cor(library_covariance_estimator(ce), X; dims = dims, kwargs...)
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

Alias that matches a [`MutualInfoCovariance`](@ref) in a wrapper.

It matches a [`PortfolioOptimisersCovariance`](@ref) that wraps a [`MutualInfoCovariance`](@ref). The [`CanonicalDistance`](@ref) methods of [`distance`](@ref) and [`cor_and_dist`](@ref) dispatch on it.

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

Alias that matches a [`DistanceCovariance`](@ref), bare or in a wrapper.

It matches a [`DistanceCovariance`](@ref), or a [`PortfolioOptimisersCovariance`](@ref) that wraps one. The [`CanonicalDistance`](@ref) methods of [`distance`](@ref) and [`cor_and_dist`](@ref) dispatch on it.

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

Rebuild `de` with the distance algorithm that fits the range of the covariance estimator, and call [`distance`](@ref) again.

The redirect has no formula. It sends each codependence measure to the transform that fits its range. A signed correlation needs the factor ``1/2``. A mutual information has no correlation to transform. A tail dependence coefficient needs an unbounded distance. The selected algorithm gets `de.power` unchanged.

| Covariance estimator                                    | Algorithm selected              | Read from `ce`                  |
|:------------------------------------------------------- |:------------------------------- |:------------------------------- |
| [`MutualInfoCovariance`](@ref)                          | [`VariationInfoDistance`](@ref) | `ce.bins`, `ce.normalise`       |
| [`PortfolioOptimisersCovariance`](@ref) that wraps it   | [`VariationInfoDistance`](@ref) | `ce.ce.bins`, `ce.ce.normalise` |
| [`LowerTailDependenceCovariance`](@ref), wrapped or not | [`LogDistance`](@ref)           | nothing                         |
| [`DistanceCovariance`](@ref), wrapped or not            | [`CorrelationDistance`](@ref)   | nothing                         |
| any other `StatsBase.CovarianceEstimator`               | [`SimpleDistance`](@ref)        | nothing                         |

The last row is the fallback. On every estimator of that row, the redirect and a bare [`Distance`](@ref) with [`SimpleDistance`](@ref) give the same result.

# Algorithm

 1. Select the row of the table above by the type of `ce`. Dispatch does the selection, so a wrapper reaches the same row as the estimator that it wraps.
 2. Build a new [`Distance`](@ref) with `de.power` and the algorithm of that row.
 3. On the two mutual information rows, copy `bins` and `normalise` from `ce` to the new [`VariationInfoDistance`](@ref). For the wrapper, read them from `ce.ce`. The signature does not show this step, but the result depends on it. A [`MutualInfoCovariance`](@ref) with a non-default `bins` gives a different distance matrix from one with the default `bins`.
 4. Call [`distance`](@ref) with the rebuilt estimator, the same `ce`, and the same `X`, `dims` and `kwargs`.

# Arguments

  - `de`: Distance estimator that holds the [`CanonicalDistance`](@ref) algorithm.
  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that the method gives to the [`distance`](@ref) method of the selected algorithm.

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`Distance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`LowerTailDependenceCovariance`](@ref)
  - [`DistanceCovariance`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`cor_and_dist`](@ref): applies the same table, with the same five rows.
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
