"""
    coskewness_residuals(X::MatNum, me::AbstractExpectedReturnsEstimator)

Build the residual coskewness matrix that a factor lift adds to its projected coskewness.

`X` is the **residual** matrix `X - posterior_X`, not the asset returns. The factor model assumes that the residuals have zero mean and are independent across assets. So each cross term of their coskewness is zero, and only the `N` third moments of the single assets remain.

The function does not remove the mean. The assumption of zero mean belongs to the factor model, and `me` gives the rule of the average, not a centre. When the residuals have a mean ``c_i`` that is not zero, the remaining entries hold the **raw** third moment ``\\mathbb{E}[(\\varepsilon_i + c_i)^3]`` and not the central third moment. The result is a coskewness only when the assumption holds.

# Mathematical definition

```math
(\\hat{\\mathbf{M}}_{3,\\varepsilon})_{i,\\,c} = \\begin{cases}
\\mathbb{E}[\\varepsilon_i^3]\\,, & c = (i - 1)N + i\\,, \\\\
0\\,, & \\text{otherwise.}
\\end{cases}
```

Where:

  - ``\\varepsilon_i``: Residual of asset ``i``, column ``i`` of `X`.
  - $(math_dict[:N])
  - ``\\hat{\\mathbf{M}}_{3,\\varepsilon}``: The ``N \\times N^2`` residual coskewness matrix, `sk_err`.

Column ``(i - 1)N + i`` of a coskewness matrix holds the products ``\\varepsilon_i \\varepsilon_i``. In row ``i``, this column is the one entry that independent residuals with zero mean can make different from zero.

# Algorithm

 1. Take `N = size(X, 2)` and `N2 = N^2`.
 2. Cube each entry of the residuals, and call the result `X3`.
 3. Allocate `sk_err`, an `N × N2` sparse zero matrix of the element type of `X3`.
 4. Take `idx`, the range of linear indices `1:(N2 + N + 1):(N2 * N)`. Julia stores an `N × N2` matrix column by column, so entry `i` of `idx` is the linear index of `(i, (i - 1) * N + i)`.
 5. Average the columns of `X3` under `me`, and write the `N` values at `idx`.

# Arguments

  - `X`: Residual return matrix (observations × assets).
  - `me`: Expected returns estimator. The function averages the cubed residuals with it.

# Returns

  - `sk_err::SparseMatrixCSC`: `N × N²` residual coskewness matrix.

# Related

  - [`cokurtosis_residuals`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function coskewness_residuals(X::MatNum, me::AbstractExpectedReturnsEstimator)
    N = size(X, 2)
    N2 = N^2
    X3 = X .^ 3
    sk_err = SparseArrays.spzeros(eltype(X3), N, N2)
    idx = 1:(N2 + N + 1):(N2 * N)
    sk_err[idx] .= vec(Statistics.mean(me, X3; dims = 1))
    return sk_err
end
"""
    cokurtosis_residuals(sigma::MatNum, X::MatNum, me::AbstractExpectedReturnsEstimator,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())

Build the residual cokurtosis matrix that a factor lift adds to its projected cokurtosis.

`X` is the **residual** matrix `X - posterior_X`, not the asset returns. `sigma` is the **systematic** covariance ``\\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal``, with no residual block. The caller removes the residual block, see [`factor_residual_config`](@ref).

The function does not standardise the residuals. It writes each entry in closed form from `sigma` and from the second and fourth residual moments, `e2 = mean(me, X .^ 2)` and `e4 = mean(me, X .^ 4)`. The closed form uses the assumption of the factor model. The residuals have zero mean, and they are independent of each other and of the factors. Under this assumption, each pattern of indices gives one branch of the loop. Only the pattern with four **distinct** indices gives zero.

Entry `(i - 1) * N + k, (j - 1) * N + l` of the result is the residual contribution to ``\\mathbb{E}[r_i r_k r_j r_l]``. The matrix is symmetric, so the function computes only the upper triangle, and writes each value to both places.

# Mathematical definition

Write ``r_i = s_i + \\varepsilon_i``, with ``s_i`` the systematic return and ``\\varepsilon_i`` the residual. The residual contribution is ``\\mathbb{E}[r_i r_k r_j r_l] - \\mathbb{E}[s_i s_k s_j s_l]``. It depends on the four indices only through which of them are equal:

```math
(\\hat{\\mathbf{\\Sigma}}_{4,\\varepsilon})_{(i-1)N+k,\\;(j-1)N+l} = \\begin{cases}
6 e_{2,a} \\mathbf{\\Sigma}_{aa} + e_{4,a}\\,, & i = k = j = l = a\\,, \\\\
3 e_{2,a} \\mathbf{\\Sigma}_{ab}\\,, & \\text{three indices are } a \\text{ and the fourth is } b\\,, \\\\
e_{2,a} \\mathbf{\\Sigma}_{bb} + e_{2,b} \\mathbf{\\Sigma}_{aa} + e_{2,a} e_{2,b}\\,, & \\text{two pairs, } a \\text{ and } b\\,, \\\\
e_{2,a} \\mathbf{\\Sigma}_{bc}\\,, & \\text{one pair } a \\text{, and singles } b \\text{ and } c\\,, \\\\
0\\,, & \\text{all four distinct.}
\\end{cases}
```

Where:

  - ``e_{2,i} = \\mathbb{E}[\\varepsilon_i^2]``, ``e_{4,i} = \\mathbb{E}[\\varepsilon_i^4]``: The second and fourth residual moments, `e2` and `e4`.
  - ``\\mathbf{\\Sigma}``: The systematic covariance, `sigma`.
  - $(math_dict[:N])
  - ``\\hat{\\mathbf{\\Sigma}}_{4,\\varepsilon}``: The ``N^2 \\times N^2`` residual cokurtosis matrix, `kt_res`.

A single ``\\varepsilon`` term has a mean of zero, and a single ``s`` term is centred, which removes the terms of odd power. With four distinct indices, no ``\\varepsilon`` meets itself, so only this case is zero. A pattern with one pair and two single indices is **not** zero. The pair gives ``e_{2,a}``, and the two single indices give their systematic covariance.

# Algorithm

 1. Take `N = size(X, 2)` and `N2 = N^2`.
 2. Raise each entry of the residuals to the second and to the fourth power, and call the results `X2` and `X4`.
 3. Average the columns of each under `me`, and call the results `e2` and `e4`.
 4. Allocate `kt_res`, of size `(N2, N2)`, in the promotion of the element types of `e4` and `sigma`.
 5. Run `ex` over the `N2` column pairs `(j, l)`. For each column, read the row pairs `(i, k)`, and skip the pair when `row > col`, which keeps the loop to the upper triangle.
 6. Select the value `val` for `(i, k, j, l)` with the chain of branches of the closed form above. The chain tests the patterns with the most equal indices first, so each pattern gets to its own branch before a branch for fewer equal indices.
 7. Write `val` to `kt_res[row, col]` and to `kt_res[col, row]`.

The executor changes only the order of the columns. One iteration of the loop writes each entry, so `FLoops.SequentialEx()` and `FLoops.ThreadedEx()` give bit-identical results.

# Arguments

  - `sigma`: Systematic covariance matrix, `N × N`.
  - `X`: Residual return matrix (observations × assets).
  - `me`: Expected returns estimator. The function averages the squared and the fourth-power residuals with it.
  - `ex`: `FLoops` executor for the loop over the `N²` columns. The default is `FLoops.ThreadedEx()`.

# Returns

  - `kt_res::Matrix`: `N² × N²` residual cokurtosis matrix.

# Related

  - [`coskewness_residuals`](@ref)
  - [`factor_residual_config`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function cokurtosis_residuals(sigma::MatNum, X::MatNum,
                              me::AbstractExpectedReturnsEstimator,
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    N = size(X, 2)
    N2 = N^2
    X2 = X .^ 2
    X4 = X2 .^ 2
    e2 = vec(mean(me, X2; dims = 1))
    e4 = vec(mean(me, X4; dims = 1))
    kt_res = Matrix{promote_type(eltype(e4), eltype(sigma))}(undef, N2, N2)

    @inbounds FLoops.@floop ex for j in 1:N, l in 1:N
        col = (j - 1) * N + l
        for i in 1:N, k in 1:N
            row = (i - 1) * N + k
            if row > col
                continue
            end
            # The patterns with the most equal indices come first, so each pattern meets
            # its own branch before the branch of a pattern with fewer equal indices.
            val = if i == j == k == l
                6 * e2[i] * sigma[i, i] + e4[i]
            elseif i == j == k
                3 * e2[i] * sigma[i, l]
            elseif i == j == l
                3 * e2[i] * sigma[i, k]
            elseif i == k == l
                3 * e2[i] * sigma[i, j]
            elseif j == k == l
                3 * e2[j] * sigma[j, i]
            elseif i == j && k == l
                e2[k] * sigma[i, i] + e2[i] * sigma[k, k] + e2[i] * e2[k]
            elseif (i == k && j == l) || (i == l && j == k)
                e2[j] * sigma[i, i] + e2[i] * sigma[j, j] + e2[i] * e2[j]
            elseif i == j
                e2[i] * sigma[k, l]
            elseif i == k
                e2[i] * sigma[j, l]
            elseif i == l
                e2[i] * sigma[j, k]
            elseif j == k
                e2[j] * sigma[i, l]
            elseif j == l
                e2[j] * sigma[i, k]
            elseif k == l
                e2[k] * sigma[i, j]
            else
                zero(promote_type(eltype(e4), eltype(sigma)))
            end
            kt_res[row, col] = kt_res[col, row] = val
        end
    end
    return kt_res
end
"""
    cokurtosis_processor(kte::CokurtosisEstimator) -> Option{<:AbstractMatrixProcessingEstimator}
    cokurtosis_processor(kte::Cokurtosis) -> AbstractMatrixProcessingEstimator
    cokurtosis_processor(kte::WindowedCokurtosis) -> Option{<:AbstractMatrixProcessingEstimator}

Give the matrix processing estimator that [`HighOrderFactorPriorEstimator`](@ref) applies to the cokurtosis that it lifts onto the asset axis.

The prior processes the lifted cokurtosis with this estimator. Under `rsd = true`, it adds the residual cokurtosis. Then it repairs the sum with the estimator that [`matrix_processing_posdef`](@ref) gives for this one.

# Interfaces

A subtype of [`CokurtosisEstimator`](@ref) that goes into a [`HighOrderFactorPriorEstimator`](@ref) adds a method of this verb. The method returns `nothing` when the subtype needs no processing. The prior then does not process the lifted cokurtosis, and does not repair the sum. The constructor of the prior calls `assert_cokurtosis_processor` on `kte`.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`Cokurtosis`](@ref) gives `kte.mp`.
 2. A [`WindowedCokurtosis`](@ref) gives the processor of the estimator that it wraps.
 3. The fallback refuses every other subtype. A default of `nothing` skips the repair of the sum under `rsd = true` with no message, and the cokurtosis can then be indefinite.

# Arguments

  - `kte`: Cokurtosis estimator.

# Validation

  - The type of `kte` has a method of its own. An `ArgumentError` that names the type is thrown otherwise.

# Returns

  - `mp::Option{<:AbstractMatrixProcessingEstimator}`: The matrix processing estimator, `nothing` when the method of `kte` names none.

# Related

  - [`CokurtosisEstimator`](@ref)
  - [`coskewness_processor`](@ref)
  - [`matrix_processing_posdef`](@ref)
  - [`assert_cokurtosis_processor`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function cokurtosis_processor(kte::CokurtosisEstimator)
    return throw(ArgumentError("`cokurtosis_processor` is not defined for `$(nameof(typeof(kte)))`. A HighOrderFactorPriorEstimator processes the cokurtosis that it lifts under the matrix processing estimator of `kte`, and under rsd = true it repairs the sum with the residual cokurtosis. Add a method `PortfolioOptimisers.cokurtosis_processor(kte::$(nameof(typeof(kte))))` that returns an `AbstractMatrixProcessingEstimator`, or `nothing` to skip the processing and the repair."))
end
function cokurtosis_processor(kte::Cokurtosis)
    return kte.mp
end
function cokurtosis_processor(kte::WindowedCokurtosis)
    return cokurtosis_processor(kte.kte)
end
"""
    cokurtosis_expected_returns(kte::CokurtosisEstimator) -> AbstractExpectedReturnsEstimator
    cokurtosis_expected_returns(kte::Cokurtosis) -> AbstractExpectedReturnsEstimator
    cokurtosis_expected_returns(kte::WindowedCokurtosis) -> AbstractExpectedReturnsEstimator

Give the expected returns estimator with which [`HighOrderFactorPriorEstimator`](@ref) averages the residual moments of its correction of the cokurtosis.

[`cokurtosis_residuals`](@ref) averages the squared and the fourth-power residuals with this estimator. The prior reads it only under `rsd = true`.

# Interfaces

A subtype of [`CokurtosisEstimator`](@ref) that goes into a [`HighOrderFactorPriorEstimator`](@ref) with `rsd = true` adds a method of this verb.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`Cokurtosis`](@ref) gives `kte.me`.
 2. A [`WindowedCokurtosis`](@ref) gives the estimator of the estimator that it wraps.
 3. The fallback refuses every other subtype. An average has no `nothing` form. A default rule that the library selects for the caller hides a missing method.

# Arguments

  - `kte`: Cokurtosis estimator.

# Validation

  - The type of `kte` has a method of its own. An `ArgumentError` that names the type is thrown otherwise.

# Returns

  - `me::AbstractExpectedReturnsEstimator`: The expected returns estimator of the residual moments.

# Related

  - [`coskewness_expected_returns`](@ref)
  - [`cokurtosis_residuals`](@ref)
  - [`assert_factor_comoment_residuals`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function cokurtosis_expected_returns(kte::CokurtosisEstimator)
    return throw(ArgumentError("`cokurtosis_expected_returns` is not defined for `$(nameof(typeof(kte)))`. A HighOrderFactorPriorEstimator with rsd = true averages the residual moments of its cokurtosis under the expected returns estimator of `kte`. Add a method `PortfolioOptimisers.cokurtosis_expected_returns(kte::$(nameof(typeof(kte))))` that returns an `AbstractExpectedReturnsEstimator`, or set rsd = false."))
end
function cokurtosis_expected_returns(kte::Cokurtosis)
    return kte.me
end
function cokurtosis_expected_returns(kte::WindowedCokurtosis)
    return cokurtosis_expected_returns(kte.kte)
end
"""
    coskewness_expected_returns(ske::CoskewnessEstimator) -> AbstractExpectedReturnsEstimator
    coskewness_expected_returns(ske::Coskewness) -> AbstractExpectedReturnsEstimator
    coskewness_expected_returns(ske::WindowedCoskewness) -> AbstractExpectedReturnsEstimator

Give the expected returns estimator with which [`HighOrderFactorPriorEstimator`](@ref) averages the residual moments of its correction of the coskewness.

[`coskewness_residuals`](@ref) averages the cubed residuals with this estimator. The prior reads it only under `rsd = true`.

# Interfaces

A subtype of [`CoskewnessEstimator`](@ref) that goes into a [`HighOrderFactorPriorEstimator`](@ref) with `rsd = true` adds a method of this verb.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`Coskewness`](@ref) gives `ske.me`.
 2. A [`WindowedCoskewness`](@ref) gives the estimator of the estimator that it wraps.
 3. The fallback refuses every other subtype, for the reason that [`cokurtosis_expected_returns`](@ref) states.

# Arguments

  - `ske`: Coskewness estimator.

# Validation

  - The type of `ske` has a method of its own. An `ArgumentError` that names the type is thrown otherwise.

# Returns

  - `me::AbstractExpectedReturnsEstimator`: The expected returns estimator of the residual moments.

# Related

  - [`cokurtosis_expected_returns`](@ref)
  - [`coskewness_residuals`](@ref)
  - [`assert_factor_comoment_residuals`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function coskewness_expected_returns(ske::CoskewnessEstimator)
    return throw(ArgumentError("`coskewness_expected_returns` is not defined for `$(nameof(typeof(ske)))`. A HighOrderFactorPriorEstimator with rsd = true averages the residual moments of its coskewness under the expected returns estimator of `ske`. Add a method `PortfolioOptimisers.coskewness_expected_returns(ske::$(nameof(typeof(ske))))` that returns an `AbstractExpectedReturnsEstimator`, or set rsd = false."))
end
function coskewness_expected_returns(ske::Coskewness)
    return ske.me
end
function coskewness_expected_returns(ske::WindowedCoskewness)
    return coskewness_expected_returns(ske.ske)
end
"""
    assert_cokurtosis_processor(kte::Nothing) -> nothing
    assert_cokurtosis_processor(kte::CokurtosisEstimator) -> nothing

Check at construction that a [`HighOrderFactorPriorEstimator`](@ref) can read the processor of its cokurtosis estimator `kte`.

The prior processes the lifted cokurtosis for each value of `rsd`, so the check runs for each value of `rsd`. With this check, a type with no method of [`cokurtosis_processor`](@ref) fails in the constructor, and not at the fit.

# Algorithm

 1. For `nothing`, return `nothing`. The prior lifts no cokurtosis.
 2. For a [`CokurtosisEstimator`](@ref), get its processor from [`cokurtosis_processor`](@ref). The fallback of that verb refuses a type with no method of its own.
 3. Check that the processor is `nothing` or an [`AbstractMatrixProcessingEstimator`](@ref).

# Arguments

  - `kte`: Cokurtosis estimator, or `nothing`.

# Validation

  - The type of `kte` has a method of [`cokurtosis_processor`](@ref). An `ArgumentError` is thrown otherwise.
  - The method returns `nothing` or an [`AbstractMatrixProcessingEstimator`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`cokurtosis_processor`](@ref)
  - [`assert_factor_comoment_residuals`](@ref)
  - [`assert_matrix_processing_posdef`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function assert_cokurtosis_processor(::Nothing)
    return nothing
end
function assert_cokurtosis_processor(kte::CokurtosisEstimator)
    mp = cokurtosis_processor(kte)
    @argcheck(isa(mp, Option{<:AbstractMatrixProcessingEstimator}),
              ArgumentError("`cokurtosis_processor(::$(nameof(typeof(kte))))` must return `nothing` or an `AbstractMatrixProcessingEstimator`. Got\nmp => $(mp)::$(typeof(mp))."))
    return nothing
end
"""
    assert_factor_comoment_residuals(kte::Option{<:CokurtosisEstimator},
                                     ske::Option{<:CoskewnessEstimator}) -> nothing

Check at construction that a [`HighOrderFactorPriorEstimator`](@ref) with `rsd = true` can read from `kte` and `ske` what its residual corrections need.

With this check, a co-moment estimator with no method fails in the constructor, and not at the fit.

# Algorithm

 1. When `kte` is not `nothing`, get its expected returns estimator from [`cokurtosis_expected_returns`](@ref), and check its type. Get its processor from [`cokurtosis_processor`](@ref), and check the repair of the processor with [`assert_matrix_processing_posdef`](@ref).
 2. When `ske` is not `nothing`, get its expected returns estimator from [`coskewness_expected_returns`](@ref), and check its type.

# Arguments

  - `kte`: Cokurtosis estimator, or `nothing`.
  - `ske`: Coskewness estimator, or `nothing`.

# Validation

  - The fallbacks of [`cokurtosis_expected_returns`](@ref) and [`coskewness_expected_returns`](@ref) refuse a type with no method. An `ArgumentError` is thrown.
  - Each method returns an `AbstractExpectedReturnsEstimator`. An `ArgumentError` is thrown otherwise.
  - The processor of `kte` passes [`assert_matrix_processing_posdef`](@ref).

# Returns

  - `nothing`.

# Related

  - [`cokurtosis_expected_returns`](@ref)
  - [`coskewness_expected_returns`](@ref)
  - [`cokurtosis_processor`](@ref)
  - [`assert_matrix_processing_posdef`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function assert_factor_comoment_residuals(kte::Option{<:CokurtosisEstimator},
                                          ske::Option{<:CoskewnessEstimator})
    if !isnothing(kte)
        me = cokurtosis_expected_returns(kte)
        @argcheck(isa(me, AbstractExpectedReturnsEstimator),
                  ArgumentError("`cokurtosis_expected_returns(::$(nameof(typeof(kte))))` must return an `AbstractExpectedReturnsEstimator`. Got\nme => $(me)::$(typeof(me))."))
        assert_matrix_processing_posdef(cokurtosis_processor(kte))
    end
    if !isnothing(ske)
        me = coskewness_expected_returns(ske)
        @argcheck(isa(me, AbstractExpectedReturnsEstimator),
                  ArgumentError("`coskewness_expected_returns(::$(nameof(typeof(ske))))` must return an `AbstractExpectedReturnsEstimator`. Got\nme => $(me)::$(typeof(me))."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Projects factor coskewness and cokurtosis onto the asset axis through the regression loadings.

`HighOrderFactorPriorEstimator` holds a low order factor prior estimator, a cokurtosis estimator and a coskewness estimator. Its [`prior`](@ref) fits the co-moments of the factor returns, and lifts them onto the asset axis with the loadings of the regression. When `rsd` is `true`, it adds the residual co-moments of the regression.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderFactorPriorEstimator(;
        pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(),
        kte::Option{<:CokurtosisEstimator} = Cokurtosis(; alg = FullMoment()),
        ske::Option{<:CoskewnessEstimator} = Coskewness(; alg = FullMoment()),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        rsd::Bool = true,
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> HighOrderFactorPriorEstimator

Keywords correspond to the struct's fields.

## Validation

  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pe` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).
  - `kte` states its processor, see [`assert_cokurtosis_processor`](@ref).
  - When `rsd` is `true`, `kte` and `ske` state what the residual corrections read, see [`assert_factor_comoment_residuals`](@ref).

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pe`: Recursively updated through [`factory`](@ref).
  - `kte`: Recursively updated through [`factory`](@ref).
  - `ske`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `pe`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> HighOrderFactorPriorEstimator()
HighOrderFactorPriorEstimator
     pe ┼ FactorPrior
        │         pe ┼ EmpiricalPrior
        │            │           ce ┼ PortfolioOptimisersCovariance
        │            │              │   ce ┼ Covariance
        │            │              │      │    me ┼ SimpleExpectedReturns
        │            │              │      │       │   w ┴ nothing
        │            │              │      │    ce ┼ GeneralCovariance
        │            │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │            │              │      │       │    w ┴ nothing
        │            │              │      │   alg ┼ FullMoment()
        │            │              │      │     w ┴ nothing
        │            │              │   mp ┼ MatrixProcessing
        │            │              │      │     pdm ┼ Posdef
        │            │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │            │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │            │              │      │      dn ┼ nothing
        │            │              │      │      dt ┼ nothing
        │            │              │      │     alg ┼ nothing
        │            │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │            │           me ┼ SimpleExpectedReturns
        │            │              │   w ┴ nothing
        │            │      horizon ┼ nothing
        │            │   fill_limit ┼ nothing
        │            │        x_src ┴ nothing
        │         mp ┼ MatrixProcessing
        │            │     pdm ┼ Posdef
        │            │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │            │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │            │      dn ┼ nothing
        │            │      dt ┼ nothing
        │            │     alg ┼ nothing
        │            │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │         re ┼ StepwiseRegression
        │            │       crit ┼ PValue
        │            │            │   t ┴ Float64: 0.05
        │            │        alg ┼ ForwardSelection()
        │            │        tgt ┼ LinearModel
        │            │            │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │            │     choice ┼ BatchChoice()
        │            │   included ┼ nothing
        │            │      x_src ┴ nothing
        │         ve ┼ SimpleVariance
        │            │          me ┼ SimpleExpectedReturns
        │            │             │   w ┴ nothing
        │            │           w ┼ nothing
        │            │   corrected ┴ Bool: true
        │        rsd ┼ Bool: true
        │   mtx_sqrt ┼ EigenFallbackSquareRoot()
        │      x_src ┴ nothing
    kte ┼ Cokurtosis
        │      me ┼ SimpleExpectedReturns
        │         │   w ┴ nothing
        │      mp ┼ MatrixProcessing
        │         │     pdm ┼ Posdef
        │         │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │         │      dn ┼ nothing
        │         │      dt ┼ nothing
        │         │     alg ┼ nothing
        │         │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │     alg ┼ FullMoment()
        │       w ┼ nothing
        │   cache ┴ nothing
    ske ┼ Coskewness
        │      me ┼ SimpleExpectedReturns
        │         │   w ┴ nothing
        │      mp ┼ MatrixProcessing
        │         │     pdm ┼ Posdef
        │         │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │         │      dn ┼ nothing
        │         │      dt ┼ nothing
        │         │     alg ┼ nothing
        │         │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │     alg ┼ FullMoment()
        │       w ┼ nothing
        │   cache ┴ nothing
     ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
    rsd ┼ Bool: true
  x_src ┴ nothing
```

## The incremental fit

This prior has no exact incremental recursion, and it takes the online step by a **refit from a sample buffer**. [`Online`](@ref) seeds `cache`, and [`partial_fit!`](@ref) adds each observation to it with no change. The [`prior`](@ref) method with one argument then runs the batch verb of this estimator over the rows that the buffer kept. The result is the batch fit over those rows, and a `max_history` on the wrapper sets the window of the whole fit.

`cache` goes through the three propagation channels as every partial-fit state does. [`factory`](@ref) carries it with no change, and [`port_opt_view`](@ref) cuts it to the selected assets. [`obs_weights_view`](@ref) drops it, because a state has no cut on the observation axis. The show method does not print `cache`, because a buffer that changes at each step is not part of the configuration.

# Related

  - [`AbstractHighOrderPriorEstimator_F`](@ref)
  - [`FactorPrior`](@ref)
  - [`CokurtosisEstimator`](@ref)
  - [`CoskewnessEstimator`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:boudt2015]) Pages 225–233. $(ref_dict[:no_original_source])
  - $(ref_dict[:martelliniziemann2010])
"""
@propagatable @concrete struct HighOrderFactorPriorEstimator <:
                               AbstractHighOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:kte])
    """
    @fprop kte
    """
    $(field_dict[:ske])
    """
    @fprop ske
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:rsd])
    """
    rsd
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function HighOrderFactorPriorEstimator(pe::AbstractLowOrderPriorEstimator_F_AF,
                                           kte::Option{<:CokurtosisEstimator},
                                           ske::Option{<:CoskewnessEstimator},
                                           ex::FLoops.Transducers.Executor, rsd::Bool,
                                           cache::Option{<:AbstractPartialFitState},
                                           x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :HighOrderFactorPriorEstimator)
        assert_inner_reader(pe, :pe, :HighOrderFactorPriorEstimator)
        assert_cokurtosis_processor(kte)
        if rsd
            assert_factor_comoment_residuals(kte, ske)
        end
        return new{typeof(pe), typeof(kte), typeof(ske), typeof(ex), typeof(rsd),
                   typeof(cache), typeof(x_src)}(pe, kte, ske, ex, rsd, cache, x_src)
    end
end
function HighOrderFactorPriorEstimator(;
                                       pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(),
                                       kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
                                                                                       alg = FullMoment()),
                                       ske::Option{<:CoskewnessEstimator} = Coskewness(;
                                                                                       alg = FullMoment()),
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                       rsd::Bool = true,
                                       cache::Option{<:AbstractPartialFitState} = nothing,
                                       x_src::Option{<:AbstractReturnsSource} = nothing)::HighOrderFactorPriorEstimator
    return HighOrderFactorPriorEstimator(pe, kte, ske, ex, rsd, cache, x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Give every field of a [`HighOrderFactorPriorEstimator`](@ref) to print, except `cache`.

The state in `cache` changes at each step of an incremental fit, and it is not part of the configuration. Each site that prints the estimator also prints this state, unless this method leaves it out. Set `set_show_nothing_fields!(:HighOrderFactorPriorEstimator, true)` to print it.

# Arguments

  - `::HighOrderFactorPriorEstimator`: Prior estimator. The method reads its type alone.

# Returns

  - `fields::Tuple`: The names of the fields to print, `(:pe, :kte, :ske, :ex, :rsd, :x_src)`.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::HighOrderFactorPriorEstimator) = (:pe, :kte, :ske, :ex, :rsd, :x_src)
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties HighOrderFactorPriorEstimator begin
    forward(pe, me, ce)
end
"""
    prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1,
          kwargs...)

Compute the high order prior of the asset returns through a factor model.

`prior` fits the low order moments with the embedded factor prior `pe.pe`. It fits the co-moments of the factor returns, and maps them to the asset axis with the Kronecker product of the loadings. When `pe.rsd` is `true`, it adds the residual co-moments.

!!! note

    A Black-Litterman prior can go under this estimator. Each wrapper estimator passes `rr` and the factor block on, so `HighOrderFactorPriorEstimator(; pe = BlackLittermanPrior(; pe = FactorPrior(…)))` gets to a regression, and it does not throw an `IsNothingError`.

    The co-moments go through `rr.M`, and `mu` and `sigma` hold the views. Black-Litterman gives no third or fourth moments, so the projection of the factor co-moments is the only estimate of them.

!!! warning

    The method computes the co-moments from `F` as the caller gives it. So they describe the factor distribution **before** the views, for each Black-Litterman prior under this estimator. [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) give a *posterior* factor block. Under them, the orders of the nested `fpr` do not agree. `fpr.mu` and `fpr.sigma` hold the views, and `fpr.kt`, `fpr.sk` and `fpr.V` do not. The invariant `fpr.pr === pr.fpr` still holds, because both routes get the same posterior low order block. The views stop at the second order.

    The cause is that Black-Litterman has no update for the third and fourth moments. The method discards no value. [`BlackLittermanPrior`](@ref) behaves in the same way. Its low order factor block is also before the views, so all the orders of its prior result agree.

    An entropy pooling prior and an opinion pooling prior are different. They state their views and opinions as the probabilities `w` of the scenarios. The factor co-moments read `w`, so the views get to every order.

# Mathematical definition

The method maps the factor co-moments to the asset axis through the loadings matrix ``\\mathbf{B}``:

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_4 &= (\\mathbf{B} \\otimes \\mathbf{B}) \\hat{\\mathbf{\\Sigma}}_{4,f} (\\mathbf{B} \\otimes \\mathbf{B})^\\intercal + \\hat{\\mathbf{\\Sigma}}_{4,\\varepsilon}\\,, \\\\
\\hat{\\mathbf{M}}_3 &= \\mathbf{B} \\hat{\\mathbf{M}}_{3,f} (\\mathbf{B} \\otimes \\mathbf{B})^\\intercal + \\hat{\\mathbf{M}}_{3,\\varepsilon}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{\\Sigma}}_4``: ``N^2 \\times N^2`` asset square cokurtosis matrix, `kt`.
  - ``\\hat{\\mathbf{M}}_3``: ``N \\times N^2`` asset coskewness matrix, `sk`.
  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix, `pr.rr.M`.
  - ``\\hat{\\mathbf{\\Sigma}}_{4,f}``: ``K^2 \\times K^2`` factor square cokurtosis matrix, `fpr.kt`.
  - ``\\hat{\\mathbf{M}}_{3,f}``: ``K \\times K^2`` factor coskewness matrix, `fpr.sk`.
  - ``\\hat{\\mathbf{\\Sigma}}_{4,\\varepsilon}``: Residual cokurtosis from [`cokurtosis_residuals`](@ref). It is present only when `rsd` is `true`.
  - ``\\hat{\\mathbf{M}}_{3,\\varepsilon}``: Residual coskewness from [`coskewness_residuals`](@ref). It is present only when `rsd` is `true`.
  - ``\\otimes``: Kronecker product.

`pe.kte` and `pe.ske` fit the factor co-moments on `F`. When the low order block gives the probabilities `w` of its scenarios, `w` replaces the observation weights of both estimators. The fit then reads the last `length(w)` rows of `F`, which `w` weighs. The residual corrections do not read `w`. The residuals belong to the model, and the views change the probabilities of the scenarios, not the model. A different `alg` on either estimator gives a different formula in place of its display above. When an estimator is `nothing`, the asset result and the nested factor block both hold `nothing` for its moment.

# Algorithm

 1. Orient `X` and `F` to `observations × variables` with [`dims_oriented`](@ref).
 2. Compute the low order block `pr` with `pe.pe`, and check that it holds a regression result. Derive the Investable Mask from it with [`investable_mask`](@ref). Cut the whole prior result to the investable universe with [`port_opt_view`](@ref), and call the result `rpr`. `X` is an argument and not a block of the prior result, so cut it in the same way, and call it `Xr`. Take the reconstructed returns `posterior_X = rpr.X` and the loadings `M = rpr.rr.M`. When every asset is investable, the mask is `nothing`, the method cuts nothing, and the steps below run on `pr`. The cut comes **before** the lift, because the lift spreads a non-investable asset much further. One `NaN` row of loadings makes a full `NaN` band of `kron(M, M)`, and the projection spreads it over most of the cokurtosis. [`posdef!`](@ref) then refuses the matrix with a LAPACK error that does not name the asset.
 3. Read the probabilities `w = pr.w` of the scenarios. Cut `F` to the rows `Fw` that `w` weighs, with [`scenario_rows`](@ref). Read the counts `T_sk` and `T_kt` of `factory(pe.ske, w)` and `factory(pe.kte, w)` over `Fw` with [`high_order_observation_count`](@ref). Compute the factor square cokurtosis `f_kt` with `factory(pe.kte, w)` on `Fw`. When `f_kt` is not `nothing`, build `kM = kron(M, M)`, and project `posterior_kt = kM * f_kt * transpose(kM)`. Process `posterior_kt` with the estimator that [`cokurtosis_processor`](@ref)`(pe.kte)` gives, through [`sample_matrix_processing!`](@ref) at the count `T_kt`.
 4. Compute the factor coskewness `f_sk` and its negative spectral form `f_V` with `factory(pe.ske, w)` on `Fw`. When `f_sk` is not `nothing`, build `kM` if step 3 did not, and project `posterior_sk = M * f_sk * transpose(kM)`.
 5. Build the structure matrices with [`dup_elim_sum_matrices`](@ref) two times. Build `D2`, `L2` and `S2` at the **full** asset count, and `f_D2`, `f_L2` and `f_S2` at the factor count. The asset count is the full count, because step 12 expands the co-moments, and the constructor checks the matrices against `length(pr.mu)`. At both dimensions, the method selects the matrices by the rule of `prior(::HighOrderPriorEstimator, …)`.
 6. When `pe.rsd` is `true`, take the reconstruction error `err = Xr - posterior_X`. Under a Scenario Cap, `posterior_X` holds only the last rows. Then the method uses `F * transpose(M) .+ transpose(b)` over every row in place of `posterior_X`, as the residual block of `sigma` does.
 7. Under `pe.rsd`, when step 4 gave a `posterior_sk`, add [`coskewness_residuals`](@ref)`(err, me)` to it. `me` comes from [`coskewness_expected_returns`](@ref)`(pe.ske)`.
 8. Under `pe.rsd`, when step 3 gave a `posterior_kt`, read the declaration of the residual block of the wrapped estimator with [`factor_residual_config`](@ref). Check its shape with [`assert_factor_residual_config`](@ref).
 9. Get the systematic covariance `sigma` from `rpr.sigma`. A `nothing` declaration, or a declaration whose `rsd` is `false`, means that the wrapped estimator added no residual block. Then `sigma` is `rpr.sigma` with no change. Otherwise, read the block with [`idiosyncratic_variances`](@ref) of `rpr.rr`, and call it `err_sigma`. These are the residual variances that the wrapped estimator added, under the weights of its own fit. A variance of `err` under `rsd_cfg.ve` does not read these weights. Subtract the diagonal matrix of `err_sigma` from `rpr.sigma`. When an entry of `err_sigma` is larger than the diagonal entry of `rpr.sigma` that matches it, the subtraction gives a negative variance. Then the step warns, and keeps `rpr.sigma` whole. This happens when the covariance of the wrapped estimator does not hold the block, such as a posterior that shrinks the sum of the lift. In both cases, repair the result with [`posdef!`](@ref) under `rsd_cfg.pdm`.
10. Under `pe.rsd`, add [`cokurtosis_residuals`](@ref)`(sigma, err, me, pe.ex)` to `posterior_kt`. `me` comes from [`cokurtosis_expected_returns`](@ref)`(pe.kte)`. Repair the sum with [`posdef!`](@ref), under the estimator that [`matrix_processing_posdef`](@ref) gives for the processor of step 3.
11. When step 4 gave a `posterior_sk`, compute `posterior_V` from it with [`negative_spectral_coskewness`](@ref), at the count `T_sk`. `V` then describes the corrected coskewness, and not the projected coskewness.
12. Expand `posterior_kt`, `posterior_sk` and `posterior_V` to the full asset universe with [`expand_moment`](@ref). The entries outside the investable block are `NaN`. The method expands `V` and does not compute it again, because `V` is a spectral value of the cut coskewness, and the full frame holds no cut returns. A `nothing` mask needs no branch of its own. For a `nothing` mask, [`expand_moment`](@ref) returns the block with no change, for each count of arguments.
13. Build the nested factor prior result `fpr` over `pr.fpr`, from the factor moments of steps 3 to 5. When `f_kt` and `f_sk` are both `nothing`, `fpr` is `nothing`. Steps 2 and 12 do not change the factor block, because the mask is on the asset axis and these co-moments are on the factor axis.
14. Build the asset [`HighOrderPrior`](@ref) with its keyword constructor, over the **full** `pr` and not the cut `rpr`, because every prior result covers the full asset universe. Both it and `fpr` state the counts `(T_sk, T_kt)` of step 3 in `eoc`.
15. Warn with [`assert_matched_coverage`](@ref) when the co-moments hold fewer assets than the low order block.

The order of steps 9 and 10 is necessary. [`cokurtosis_residuals`](@ref) reads the systematic covariance. So step 9 must remove the residual block that the lift of the wrapped estimator added, before step 10 adds the residual cokurtosis.

# Arguments

  - `pe`: High order factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:pnl_prior])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, passed to the estimators that `pe` holds.

# Validation

  - `dims in (1, 2)`.
  - The prior result of `pe.pe` must hold a regression result, see [`assert_prior_regression`](@ref).
  - The wrapped estimator must declare its residual block with a method of [`factor_residual_config`](@ref). The verb has no default. For an estimator with no method, it throws an `ArgumentError`, and it does not read as *no residual block*. [`assert_factor_residual_config`](@ref) checks the shape of the declaration. It throws an `ArgumentError` when the declaration is not `nothing` and not a `NamedTuple` that holds `ve`, `pdm` and `rsd`. These two errors occur only when `pe.rsd` is `true` and a cokurtosis is present.

# Returns

  - `pr::HighOrderPrior`: Result that holds the asset returns, the mean, the covariance, the coskewness, the square cokurtosis and the factor moments, on the **full** asset universe. An asset that the wrapped prior did not estimate holds `NaN` in `mu`, on the diagonal of `sigma`, and at each index of `kt`, `sk` and `V` that names it. [`HighOrderPriorEstimator`](@ref) makes the same prior result on the same panel with gaps, and [`port_opt_view`](@ref) cuts the `NaN` entries out of it. Its `fpr` is a nested [`HighOrderPrior`](@ref) over the factor block of the wrapped prior, so `fpr.pr === pr.fpr`. Both routes give one distribution for the factor co-moments and the low order factor moments.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
  - [`FactorPrior`](@ref)
  - [`prior`](@ref)
  - [`investable_mask`](@ref): the mask of step 2. Its `nothing` value, for a universe in which every asset is investable, makes the method cut nothing.
  - [`port_opt_view`](@ref): the cut of step 2, which cuts every block that the lift reads.
  - [`expand_moment`](@ref): the expansion of step 12.
  - [`factor_residual_config`](@ref): the declaration of the residual block that step 9 removes. The bound of `pe.pe` is [`AbstractLowOrderPriorEstimator_F_AF`](@ref). Only [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) hold the fields that size the block. So a wrapper over either passes the declaration on, and each other estimator declares `nothing` in a method of its own.
  - [`assert_factor_residual_config`](@ref): the check of the shape of that declaration.
  - [`coskewness_residuals`](@ref)
  - [`cokurtosis_residuals`](@ref)
  - [`scenario_rows`](@ref): the cut of step 3, to the rows of `F` that the probabilities of the scenarios weigh.
"""
function prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
    X, F = dims_oriented(dims, X, F)
    kM = nothing
    D2 = nothing
    L2 = nothing
    S2 = nothing
    f_D2 = nothing
    f_L2 = nothing
    f_S2 = nothing
    posterior_kt = nothing
    posterior_sk = nothing
    posterior_V = nothing
    pr = prior(pe.pe, X, F, pnl; dims = 1, kwargs...)
    assert_prior_regression(pr, :pe)
    # The lift *squares* the reach of a non-investable asset. One `NaN` loading row makes a
    # whole `NaN` band of `kron(M, M)`, and `kM * f_kt * transpose(kM)` carries it across most
    # of the cokurtosis — 369 entries of 625 for one asset in five — which `matrix_processing!`
    # then refuses through `posdef!` with a message that names LAPACK rather than the asset.
    # So the estimator applies the contract's two halves, in the order every optimiser applies
    # them: reduce once here, lift on the investable universe alone, and expand the co-moments
    # back into a `NaN` frame of the full width below. `nothing` is the all-investable
    # sentinel, and it takes the path this estimator always took.
    #
    # `port_opt_view` cuts every block the lift reads — `X`, `mu`, `sigma`, and `rr` on its
    # asset axis — but the caller's `X` is an argument rather than a block of the prior
    # result, so it is cut here alongside, on the same index.
    #
    # A branch, where the expansion below is dispatch. The mask is a *value*, and
    # `investable_mask` is inferred `Union{Nothing, BitVector}`, so a `(::Nothing, …)`/
    # `(::BitVector, …)` pair on it is union-split back into this very branch rather than
    # resolved statically: `report_opt` reports the same 47 either way. Dispatch would buy no
    # inference here and would owe two units in a swept file. `expand_moment` is dispatch
    # because that family already exists, not because the mask is any more static there.
    imsk = investable_mask(pr)
    rpr, Xr = if isnothing(imsk)
        pr, X
    else
        i = findall(imsk)
        port_opt_view(pr, i), X[:, i]
    end
    posterior_X = rpr.X
    M = rpr.rr.M
    # A prior result that answers `w` weighs its scenarios with it, and the rows of `F` pair
    # with the scenarios, so the factor co-moments read the same distribution as the factor
    # moments that `mu` and `sigma` lift (#1650). The residual co-moments below stay with
    # the model, as the residual block of `sigma` does (ADR 0205).
    w = pr.w
    Fw = scenario_rows(w, F, nothing)[1]
    # The asset co-moments are lifted from the factor co-moments, so each is worth the count
    # of the estimator that fits its factor co-moment, and its denoiser reads that count
    # (#1893).
    T_sk = high_order_observation_count(pe.ske, w, Fw)
    T_kt = high_order_observation_count(pe.kte, w, Fw)
    eoc = map(c -> moment_count_entry(c, size(pr.X, 1)), (T_sk, T_kt))
    f_kt = cokurtosis(factory(pe.kte, w), Fw; kwargs...)
    if !isnothing(f_kt)
        kM = kron(M, M)
        posterior_kt = kM * f_kt * transpose(kM)
        sample_matrix_processing!(cokurtosis_processor(pe.kte), posterior_kt, posterior_X,
                                  T_kt; kwargs...)
    end
    f_sk, f_V = coskewness(factory(pe.ske, w), Fw; kwargs...)
    if !isnothing(f_sk)
        if isnothing(kM)
            kM = kron(M, M)
        end
        posterior_sk = M * f_sk * transpose(kM)
    end
    # The same all-or-none branching the asset block gets, at the factor dimension: the
    # nested prior result validates its own `kt`/`L2`/`S2` triple against
    # `length(pr.fpr.mu)`.
    # The structure matrices are sized from the *full* asset count, not from the reduced
    # `posterior_X`: the co-moments are expanded before the prior result is assembled, so
    # the triple the constructor validates against `length(pr.mu)` has to describe that
    # width.
    if !isnothing(f_kt) && !isnothing(f_sk)
        D2, L2, S2 = dup_elim_sum_matrices(length(pr.mu))
        f_D2, f_L2, f_S2 = dup_elim_sum_matrices(size(F, 2))
    elseif !isnothing(f_kt) && isnothing(f_sk)
        L2, S2 = dup_elim_sum_matrices(length(pr.mu))[2:3]
        f_L2, f_S2 = dup_elim_sum_matrices(size(F, 2))[2:3]
    end
    if pe.rsd
        # The residual co-moments stay with the model, as the residual block of `sigma` does,
        # so they read every row of the regression. Under a Scenario Cap `posterior_X` holds
        # the last rows alone, and the model rebuilds the rest (ADR 0206).
        Xm = if size(posterior_X, 1) == size(Xr, 1)
            posterior_X
        else
            F * transpose(M) .+ transpose(rpr.rr.b)
        end
        err = Xr - Xm
        if !isnothing(f_sk)
            posterior_sk .+= coskewness_residuals(err, coskewness_expected_returns(pe.ske))
        end
        if !isnothing(f_kt)
            # `cokurtosis_residuals` is defined on the *systematic* covariance, so a residual
            # block the wrapped estimator added has to come back off. Whether an estimator added
            # one, and under which `pdm`, is a declaration rather than a field read: the `pe`
            # slot is bounded `AbstractLowOrderPriorEstimator_F_AF`, and only `FactorPrior` and
            # `FactorBlackLittermanPrior` carry `rsd` and `mp` — a wrapper over either
            # forwards the declaration, everything else declares `nothing` (see
            # [`factor_residual_config`](@ref)). The shape is checked before the property
            # access, so a wrong declaration names itself here instead of surfacing as a
            # `FieldError` below.
            rsd_cfg = factor_residual_config(pe.pe)
            assert_factor_residual_config(pe.pe, rsd_cfg)
            sigma = if isnothing(rsd_cfg) || !rsd_cfg.rsd
                rpr.sigma
            else
                # The block comes off as the prior added it, from the loadings result. A
                # variance of `err` under `rsd_cfg.ve` read the weights of the configuration,
                # not the weights of the fit: a caller's `w` on an entropy pooling prior, or
                # the pooled weights of an opinion pool, moved it 7.5 % and 1.3 %.
                err_sigma = idiosyncratic_variances(rpr.rr)
                sigma = if any(map((x, y) -> x > y, err_sigma,
                                   LinearAlgebra.diag(rpr.sigma)))
                    @warn("Some residual variances are larger than prior variances; using the prior variances to error correct the posterior kurtosis.",
                          maxlog = log_limit(:warn), _id = @log_id())
                    rpr.sigma
                else
                    rpr.sigma - LinearAlgebra.diagm(err_sigma)
                end
                posdef!(rsd_cfg.pdm, sigma)
                sigma
            end
            err_kt = cokurtosis_residuals(sigma, err, cokurtosis_expected_returns(pe.kte),
                                          pe.ex)
            posterior_kt .+= err_kt
            posdef!(matrix_processing_posdef(cokurtosis_processor(pe.kte)), posterior_kt)
        end
    end
    if !isnothing(f_sk)
        posterior_V = negative_spectral_coskewness(posterior_sk, posterior_X,
                                                   coskewness_processor(pe.ske), T_sk)
    end
    # The expansion, the second half of the contract. Every asset-side co-moment goes back
    # into a `NaN` frame of the full width through the same [`expand_moment`](@ref) the
    # moment estimators use, so this estimator's prior result is the one
    # [`HighOrderPriorEstimator`](@ref) makes on the same gapped panel — the fourth-moment
    # index of a non-investable asset is `NaN` and nothing else is — and `port_opt_view`
    # slices it clean again. `V` is expanded rather than recomputed: it is a spectral
    # quantity of the *reduced* coskewness, and the frame has no reduced returns to rebuild
    # it from. The all-investable sentinel needs no condition of its own: `expand_moment` on a
    # `nothing` mask hands the block straight back, at every one of its arities.
    if !isnothing(posterior_kt)
        posterior_kt = expand_moment(posterior_kt, imsk, Val(:kt))
    end
    if !isnothing(posterior_sk)
        posterior_sk, posterior_V = expand_moment((posterior_sk, posterior_V), imsk)
    end
    # The nested block's `pr` is the wrapped prior's own factor block, which is what the
    # `fpr.pr === pr.fpr` invariant asks for — the factor co-moments and the factor
    # low-order moments describe one distribution, reachable by either route.
    fpr = if isnothing(f_kt) && isnothing(f_sk)
        nothing
    else
        HighOrderPrior(; pr = pr.fpr, kt = f_kt, D2 = f_D2, L2 = f_L2, S2 = f_S2, sk = f_sk,
                       V = f_V,
                       skmp = isnothing(f_sk) ? nothing : coskewness_processor(pe.ske),
                       eoc = eoc)
    end
    hop = HighOrderPrior(; pr = pr, kt = posterior_kt, D2 = D2, L2 = L2, S2 = S2,
                         sk = posterior_sk, V = posterior_V,
                         skmp = isnothing(f_sk) ? nothing : coskewness_processor(pe.ske),
                         eoc = eoc, fpr = fpr)
    # The posterior co-moments run through this estimator's own `ske` and `kte`, so a policy
    # set on the low order alone narrows the Investable Mask here exactly as it does for a
    # plain high order prior. The fit is the one place that says so.
    assert_matched_coverage(hop)
    return hop
end

function factor_residual_config(pe::HighOrderFactorPriorEstimator)
    # `pe.rsd` governs the co-moment corrections, not the covariance: the low-order block of
    # the result is the wrapped estimator's own, so this estimator forwards the wrapped
    # declaration (see [`factor_residual_config`](@ref)).
    return factor_residual_config(pe.pe)
end

export HighOrderFactorPriorEstimator
public cokurtosis_processor, cokurtosis_expected_returns, coskewness_expected_returns
