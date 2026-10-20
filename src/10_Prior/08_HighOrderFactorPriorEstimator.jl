"""
    coskewness_residuals(X::MatNum, me::AbstractExpectedReturnsEstimator)

Build the residual coskewness matrix a factor lift adds to its projected coskewness.

`X` is the **residual** matrix `X - posterior_X`, not the asset returns. The residuals are independent across assets and have zero mean, so every cross term of their coskewness vanishes and only the `N` own-third-moment entries survive.

Nothing is demeaned here. The zero-mean assumption is the factor model's, and `me` supplies the averaging rule rather than a centre. Feed residuals whose mean is not zero and the surviving entries hold the **raw** third moment ``\\mathbb{E}[(\\varepsilon_i + c_i)^3]`` rather than the central one, so the result is a coskewness only as far as the assumption holds.

# Mathematical definition

```math
(\\hat{\\mathbf{M}}_{3,\\varepsilon})_{i,\\,c} = \\begin{cases}
\\mathbb{E}[\\varepsilon_i^3]\\,, & c = (i - 1)N + i\\,, \\\\
0\\,, & \\text{otherwise.}
\\end{cases}
```

Where:

  - ``\\varepsilon_i``: residual of asset ``i``, the ``i``-th column of `X`.
  - $(math_dict[:N])
  - ``\\hat{\\mathbf{M}}_{3,\\varepsilon}``: the ``N \\times N^2`` residual coskewness matrix, `sk_err`.

Column ``(i - 1)N + i`` is the one that holds ``\\varepsilon_i \\varepsilon_i`` in a coskewness matrix, so row ``i`` meets it at the only entry of that row a set of independent zero-mean residuals can fill.

# Algorithm

 1. Take `N = size(X, 2)` and `N2 = N^2`.
 2. Cube the residuals entry by entry, giving `X3`.
 3. Allocate `sk_err`, an `N × N2` sparse zero matrix of the element type of `X3`.
 4. Take `idx`, the linear index range `1:(N2 + N + 1):(N2 * N)`. In an `N × N2` matrix stored column by column, its `i`-th entry addresses `(i, (i - 1) * N + i)`.
 5. Average the columns of `X3` under `me`, and write the `N` values into `idx`.

# Arguments

  - `X`: Residual return matrix (observations × assets).
  - `me`: Expected returns estimator, used to average the cubed residuals.

# Returns

  - `sk_err::SparseMatrixCSC`: `N × N²` residual coskewness matrix.

## The incremental fit

This prior has no exact incremental recursion, so it takes the online step by **refitting from a sample buffer**: [`Online`](@ref) seeds `cache`, [`partial_fit!`](@ref) appends each observation to it verbatim, and the one-argument [`prior`](@ref) runs this estimator's own batch verb over the rows the buffer kept. The answer is therefore exactly a batch fit over those rows, and a `max_history` on the wrapper windows the whole fit.

`cache` travels the three propagation channels as every partial-fit state does: [`factory`](@ref) carries it unchanged, [`port_opt_view`](@ref) slices it to the selected assets, and [`obs_weights_view`](@ref) drops it, because no slice of a state exists on the observation axis. It is not rendered, because a running buffer is not the configuration a reader looks the type up for.

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

Build the residual cokurtosis matrix a factor lift adds to its projected cokurtosis.

`X` is the **residual** matrix `X - posterior_X`, not the asset returns, and `sigma` is the **systematic** covariance ``\\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal``, with any residual block already removed. The caller does that removal; see [`factor_residual_config`](@ref).

Nothing is standardised here. Every entry is written in closed form from the second and fourth residual moments `e2 = mean(me, X .^ 2)` and `e4 = mean(me, X .^ 4)` together with `sigma`, under the factor model's assumption that the residuals have zero mean and are independent both of each other and of the factors. Under those assumptions each index pattern collapses to one of the branches in the loop, and the only pattern that is zero is the one whose four indices are **all distinct**.

Entry `(i - 1) * N + k, (j - 1) * N + l` of the result is the residual contribution to ``\\mathbb{E}[r_i r_k r_j r_l]``. The matrix is symmetric, so only the upper triangle is computed and each value is written to both places.

# Mathematical definition

Write ``r_i = s_i + \\varepsilon_i``, with ``s_i`` the systematic return and ``\\varepsilon_i`` the residual. The residual contribution is ``\\mathbb{E}[r_i r_k r_j r_l] - \\mathbb{E}[s_i s_k s_j s_l]``, and it depends on the four indices only through the pattern of their coincidences:

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

  - ``e_{2,i} = \\mathbb{E}[\\varepsilon_i^2]``, ``e_{4,i} = \\mathbb{E}[\\varepsilon_i^4]``: the second and fourth residual moments, `e2` and `e4`.
  - ``\\mathbf{\\Sigma}``: the systematic covariance, `sigma`.
  - $(math_dict[:N])
  - ``\\hat{\\mathbf{\\Sigma}}_{4,\\varepsilon}``: the ``N^2 \\times N^2`` residual cokurtosis matrix, `kt_res`.

A single ``\\varepsilon`` factor averages to zero and a lone ``s`` factor is centred, which is what removes the odd terms. Four distinct indices leave no ``\\varepsilon`` paired with itself, so that case alone vanishes; a pattern with a pair and two singles does **not**, because the pair contributes ``e_{2,a}`` and the two singles contribute their systematic covariance.

# Algorithm

 1. Take `N = size(X, 2)` and `N2 = N^2`.
 2. Square and fourth-power the residuals entry by entry, giving `X2` and `X4`.
 3. Average the columns of each under `me`, giving `e2` and `e4`.
 4. Allocate `kt_res`, of size `(N2, N2)`, in the promotion of the element types of `e4` and `sigma`.
 5. Run `ex` over the `N2` column pairs `(j, l)`. For each column, walk the row pairs `(i, k)` and skip the pair when `row > col`, so only the upper triangle is visited.
 6. Select the value `val` for `(i, k, j, l)` through the branch chain of the closed form above, ordered so the most common patterns are tested first.
 7. Write `val` to `kt_res[row, col]` and to `kt_res[col, row]`.

The executor changes the order in which the columns are visited and nothing else: each column writes its own entries, so `FLoops.SequentialEx()` and `FLoops.ThreadedEx()` give bit-identical results.

# Arguments

  - `sigma`: Systematic covariance matrix, `N × N`.
  - `X`: Residual return matrix (observations × assets).
  - `me`: Expected returns estimator, used to average the squared and the fourth-power residuals.
  - `ex`: `FLoops` executor for the `N²` loop. Defaults to `FLoops.ThreadedEx()`.

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
            # Conditional logic optimized for most common cases first
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

Gives the matrix processing estimator with which [`HighOrderFactorPriorEstimator`](@ref) processes the cokurtosis that it lifts onto the asset axis.

The prior processes the lifted cokurtosis under this estimator. Under `rsd = true` it adds the residual cokurtosis, and it repairs the sum under the estimator that [`matrix_processing_posdef`](@ref) gives for this one.

# Interfaces

A subtype of [`CokurtosisEstimator`](@ref) that goes into a [`HighOrderFactorPriorEstimator`](@ref) adds a method of this verb. The method returns `nothing` when the subtype wants no processing, and the prior then does not process the lifted cokurtosis or repair the sum. The constructor of the prior calls `assert_cokurtosis_processor` on `kte`.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`Cokurtosis`](@ref) gives `kte.mp`.
 2. A [`WindowedCokurtosis`](@ref) gives the processor of the estimator that it wraps.
 3. Every other subtype is refused. A `nothing` answer would skip the repair of the sum under `rsd = true` with no message, and the cokurtosis could be indefinite (#1779).

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

Gives the expected returns estimator with which [`HighOrderFactorPriorEstimator`](@ref) averages the residual moments of its cokurtosis correction.

[`cokurtosis_residuals`](@ref) averages the squared and the fourth-power residuals under this estimator. The prior reads it only under `rsd = true`.

# Interfaces

A subtype of [`CokurtosisEstimator`](@ref) that goes into a [`HighOrderFactorPriorEstimator`](@ref) with `rsd = true` adds a method of this verb.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`Cokurtosis`](@ref) gives `kte.me`.
 2. A [`WindowedCokurtosis`](@ref) gives the estimator of the estimator that it wraps.
 3. Every other subtype is refused. An average has no `nothing` value, and a default rule chosen for the caller would hide a missing method.

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

Gives the expected returns estimator with which [`HighOrderFactorPriorEstimator`](@ref) averages the residual moments of its coskewness correction.

[`coskewness_residuals`](@ref) averages the cubed residuals under this estimator. The prior reads it only under `rsd = true`.

# Interfaces

A subtype of [`CoskewnessEstimator`](@ref) that goes into a [`HighOrderFactorPriorEstimator`](@ref) with `rsd = true` adds a method of this verb.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`Coskewness`](@ref) gives `ske.me`.
 2. A [`WindowedCoskewness`](@ref) gives the estimator of the estimator that it wraps.
 3. Every other subtype is refused, for the reason that [`cokurtosis_expected_returns`](@ref) states.

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

Checks at construction that a [`HighOrderFactorPriorEstimator`](@ref) can read the processor of its cokurtosis estimator `kte`.

The prior processes the lifted cokurtosis under every `rsd`, so the check runs under every `rsd`. It moves the refusal of a type with no method of [`cokurtosis_processor`](@ref) from the fit to the constructor.

# Algorithm

 1. `nothing` gives `nothing`. The prior lifts no cokurtosis.
 2. A [`CokurtosisEstimator`](@ref) gets its processor from [`cokurtosis_processor`](@ref). Its fallback refuses a type with no method of its own.
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

Checks at construction that a [`HighOrderFactorPriorEstimator`](@ref) with `rsd = true` can read what its residual corrections need from `kte` and `ske`.

The check moves the refusal of a co-moment estimator with no method from the fit to the constructor.

# Algorithm

 1. When `kte` is not `nothing`, get its expected returns estimator from [`cokurtosis_expected_returns`](@ref), and check its type. Check the repair of its processor from [`cokurtosis_processor`](@ref) with [`assert_matrix_processing_posdef`](@ref).
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

`HighOrderFactorPriorEstimator` extends a low-order factor prior with coskewness and cokurtosis moments estimated from a factor model. It supports error correction of higher-order moments using residuals from the factor regression.

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

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `kte`: Recursively updated via [`factory`](@ref).
  - `ske`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).

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

# Related

  - [`AbstractHighOrderPriorEstimator_F`](@ref)
  - [`FactorPrior`](@ref)
  - [`CokurtosisEstimator`](@ref)
  - [`CoskewnessEstimator`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:boudt2015])
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

Renders every field of a [`HighOrderFactorPriorEstimator`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:HighOrderFactorPriorEstimator, true)` to render it.

# Arguments

  - `::HighOrderFactorPriorEstimator`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pe, :kte, :ske, :ex, :rsd)`.

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

Compute high order factor prior moments for asset returns using a factor model.

`prior` estimates the mean, covariance, coskewness, and cokurtosis of asset returns using a factor model with residual error correction. It first computes low order moments via the embedded factor prior, then maps factor higher-order moments to asset space via the Kronecker product of the factor loadings, optionally adding residual corrections.

!!! note

    A Black-Litterman prior underneath this estimator now **returns numbers where it used to throw**. Every wrapping estimator forwards `rr` and the factor block, so `HighOrderFactorPriorEstimator(; pe = BlackLittermanPrior(; pe = FactorPrior(…)))` reaches a regression instead of an `IsNothingError`.

    What comes back is worth understanding. The higher co-moments project through `rr.M` while `mu` and `sigma` carry the views — Black-Litterman makes no claim about third and fourth moments, so the factor projection is the only estimate available.

!!! warning

    The co-moments are computed from `F` as supplied, so they always describe the **pre-view** factor distribution, whichever Black-Litterman member is underneath. Where that member reports a *posterior* factor block — [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) — the nested `fpr` therefore mixes orders: `fpr.mu` and `fpr.sigma` carry the views, `fpr.kt`, `fpr.sk` and `fpr.V` do not. The `fpr.pr === pr.fpr` invariant still holds, because both routes reach the same posterior low order block; what differs is the order at which the views stop.

    This is a consequence of Black-Litterman having no higher-moment update to apply, not of a value being discarded, and it is the same under [`BlackLittermanPrior`](@ref) — where the low order factor block is pre-view too, so the prior result happens to be uniform.

    An entropy pooling prior and an opinion pooling prior differ. They state their views and opinions as the probabilities `w` of the scenarios, and the factor co-moments read `w`, so the views reach every order.

# Mathematical definition

Factor comoments are mapped to asset space through the loadings matrix ``\\mathbf{B}``:

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
  - ``\\hat{\\mathbf{\\Sigma}}_{4,\\varepsilon}``: Residual cokurtosis correction from [`cokurtosis_residuals`](@ref), present only when `rsd` is `true`.
  - ``\\hat{\\mathbf{M}}_{3,\\varepsilon}``: Residual coskewness correction from [`coskewness_residuals`](@ref), present only when `rsd` is `true`.
  - ``\\otimes``: Kronecker product.

The factor comoments come from `pe.kte` and `pe.ske` fit on `F`. When the low order block answers the probabilities `w` of its scenarios, `w` replaces the observation weights of both estimators, and the fit reads the last `length(w)` rows of `F`, which `w` weighs. The residual corrections read no `w`, because the residuals belong to the model, and the views move the probabilities of the scenarios, not the model. A non-default `alg` on either replaces its display above. Either estimator set to `nothing` drops its moment from both the asset result and the nested factor block.

# Algorithm

 1. Orient `X` and `F` to `observations × variables` with [`dims_oriented`](@ref).
 2. Compute the low order block `pr` with `pe.pe`, and check that it carries a regression result. Derive the Investable Mask from it with [`investable_mask`](@ref), and reduce the whole prior result to the investable universe with [`port_opt_view`](@ref), giving `rpr`; `X` is an argument rather than a block of the prior result, so it is cut alongside as `Xr`. Take the reconstructed returns `posterior_X = rpr.X` and the loadings `M = rpr.rr.M`. The `nothing` sentinel of an all-investable universe reduces nothing, and every step below then runs on `pr` itself. The reduction happens **before** the lift because the lift squares the reach of a non-investable asset: one `NaN` loading row makes a whole `NaN` band of `kron(M, M)`, and the projection carries it across most of the cokurtosis, where [`posdef!`](@ref) refuses it in the name of LAPACK rather than of the asset.
 3. Read the probabilities `w = pr.w` of the scenarios, and cut `F` to the rows `Fw` that they weigh with [`scenario_rows`](@ref). Compute the factor square cokurtosis `f_kt` with `factory(pe.kte, w)` on `Fw`. When it exists, build `kM = kron(M, M)`, project `posterior_kt = kM * f_kt * transpose(kM)`, and process it under [`cokurtosis_processor`](@ref)`(pe.kte)`.
 4. Compute the factor coskewness `f_sk` and its negative spectral form `f_V` with `factory(pe.ske, w)` on `Fw`. When `f_sk` exists, build `kM` if step 3 did not, and project `posterior_sk = M * f_sk * transpose(kM)`.
 5. Build the structure matrices with [`dup_elim_sum_matrices`](@ref), twice: at the **full** asset count for `D2`, `L2` and `S2`, and at the factor count for `f_D2`, `f_L2` and `f_S2`. The asset count is the full one because step 12 expands the co-moments, and the constructor validates the triple against `length(pr.mu)`. The all-or-none rule is the one `prior(::HighOrderPriorEstimator, …)` applies, at both dimensions.
 6. When `pe.rsd` is `true`, take the reconstruction error `err = Xr - posterior_X`. Under a Scenario Cap `posterior_X` holds the last rows alone, so the error reads `F * transpose(M) .+ transpose(b)` over every row instead, as the residual block of `sigma` does.
 7. Still under `pe.rsd`, add [`coskewness_residuals`](@ref)`(err, me)` to the `posterior_sk`, with `me` from [`coskewness_expected_returns`](@ref)`(pe.ske)`, of step 4, when there is one.
 8. Still under `pe.rsd`, and when step 3 produced a `posterior_kt`, read the wrapped estimator's residual declaration with [`factor_residual_config`](@ref) and check its shape with [`assert_factor_residual_config`](@ref).
 9. Recover the systematic covariance `sigma` from `rpr.sigma`. A `nothing` declaration, and one whose `rsd` is `false`, both mean that no residual block was added, so `sigma` is `rpr.sigma` unchanged. Otherwise read the block as `err_sigma`, the residual variances that the wrapped estimator added, with [`idiosyncratic_variances`](@ref) of `rpr.rr`. They were measured under the weights of that fit, which a variance of `err` under `rsd_cfg.ve` does not read. Subtract its diagonal matrix from `rpr.sigma`, and re-condition the difference with [`posdef!`](@ref) under `rsd_cfg.pdm`. When any entry of `err_sigma` exceeds the matching diagonal entry of `rpr.sigma` the subtraction would leave a negative variance, so the step warns and keeps `rpr.sigma` whole. That happens when the wrapped estimator reports a covariance the block was never added to — a posterior that shrank it, rather than the lift's own sum.
10. Still under `pe.rsd`, add [`cokurtosis_residuals`](@ref)`(sigma, err, me, pe.ex)` to `posterior_kt`, with `me` from [`cokurtosis_expected_returns`](@ref)`(pe.kte)`. Re-condition the sum with [`posdef!`](@ref) under the estimator that [`matrix_processing_posdef`](@ref) gives for the processor of step 3.
11. When step 4 produced a `posterior_sk`, recompute `posterior_V` from it with [`negative_spectral_coskewness`](@ref), so `V` describes the corrected coskewness rather than the projected one.
12. Expand `posterior_kt`, `posterior_sk` and `posterior_V` back to the full asset universe with [`expand_moment`](@ref), `NaN` outside the investable block. `V` is expanded rather than recomputed, because it is a spectral quantity of the reduced coskewness and the frame carries no reduced returns to rebuild it from. The all-investable sentinel needs no branch of its own: [`expand_moment`](@ref) on a `nothing` mask hands the block straight back, at every one of its arities.
13. Build the nested factor prior result `fpr` over `pr.fpr`, from the factor moments of steps 3 to 5. It is `nothing` when neither `f_kt` nor `f_sk` exists. The factor block is untouched by steps 2 and 12: `i` indexes assets, and these co-moments live on the factor axis.
14. Assemble the asset [`HighOrderPrior`](@ref) through its keyword constructor, over the **full** `pr` rather than the reduced `rpr`, so the prior result lives on the full asset universe as the contract requires.

Steps 9 and 10 are ordered, not independent. [`cokurtosis_residuals`](@ref) is defined on the systematic covariance, so step 9 has to undo the residual block that the wrapped estimator's own lift added before step 10 adds the residual cokurtosis.

# Arguments

  - `pe`: High order factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:pnl_prior])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Validation

  - `dims in (1, 2)`.
  - The prior produced by `pe.pe` must carry a regression result, via [`assert_prior_regression`](@ref).
  - The wrapped estimator must declare its residual block through [`factor_residual_config`](@ref), which has no default: an estimator that declares nothing throws an `ArgumentError` rather than reading as *no residual block*. The declaration's shape is checked with [`assert_factor_residual_config`](@ref), which throws an `ArgumentError` when it is neither `nothing` nor a `NamedTuple` carrying `ve`, `pdm` and `rsd`. Both raises happen only when `pe.rsd` is `true` and there is a cokurtosis to correct.

# Returns

  - `pr::HighOrderPrior`: Result object containing asset returns, mean, covariance, coskewness tensor, cokurtosis tensor, and factor moments, on the **full** asset universe. An asset the wrapped prior could not estimate carries `NaN` in `mu`, on the diagonal of `sigma`, and at every fourth-moment index that names it in `kt`, `sk` and `V` — the prior result that [`HighOrderPriorEstimator`](@ref) makes on the same gapped panel, which [`port_opt_view`](@ref) slices clean. Its `fpr` is a nested [`HighOrderPrior`](@ref) built over the wrapped prior's own factor block, so `fpr.pr === pr.fpr`: the factor co-moments and the low order factor moments describe one distribution, reachable by either route.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
  - [`FactorPrior`](@ref)
  - [`prior`](@ref)
  - [`investable_mask`](@ref): the mask step 2 derives, and the `nothing` sentinel that keeps an all-investable universe on the path it took.
  - [`port_opt_view`](@ref): the reduction of step 2, which cuts every block the lift reads.
  - [`expand_moment`](@ref): the expansion of step 12.
  - [`factor_residual_config`](@ref): the declaration that names the residual block step 9 removes. `pe.pe` is bounded [`AbstractLowOrderPriorEstimator_F_AF`](@ref), and only [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) carry the fields the block is sized from, so a wrapper over either forwards the declaration and everything else declares `nothing` in an explicit method.
  - [`assert_factor_residual_config`](@ref): the shape check that runs on that declaration.
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
    f_kt = cokurtosis(factory(pe.kte, w), Fw; kwargs...)
    if !isnothing(f_kt)
        kM = kron(M, M)
        posterior_kt = kM * f_kt * transpose(kM)
        matrix_processing!(cokurtosis_processor(pe.kte), posterior_kt, posterior_X;
                           kwargs...)
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
                    @warn("Some residual variances are larger than prior variances; using the prior variances to error correct the posterior kurtosis.")
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
                                                   coskewness_processor(pe.ske))
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
                       skmp = isnothing(f_sk) ? nothing : coskewness_processor(pe.ske))
    end
    hop = HighOrderPrior(; pr = pr, kt = posterior_kt, D2 = D2, L2 = L2, S2 = S2,
                         sk = posterior_sk, V = posterior_V,
                         skmp = isnothing(f_sk) ? nothing : coskewness_processor(pe.ske),
                         fpr = fpr)
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
