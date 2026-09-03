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
            elseif i == k && j == l
                e2[j] * sigma[i, i] + e2[i] * sigma[j, j] + e2[i] * e2[j]
            elseif i == l && j == k
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
        rsd::Bool = true
    ) -> HighOrderFactorPriorEstimator

Keywords correspond to the struct's fields.

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
      │    pe ┼ EmpiricalPrior
      │       │        ce ┼ PortfolioOptimisersCovariance
      │       │           │   ce ┼ Covariance
      │       │           │      │    me ┼ SimpleExpectedReturns
      │       │           │      │       │   w ┴ nothing
      │       │           │      │    ce ┼ GeneralCovariance
      │       │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │           │      │       │    w ┴ nothing
      │       │           │      │   alg ┼ FullMoment()
      │       │           │      │     w ┴ nothing
      │       │           │   mp ┼ MatrixProcessing
      │       │           │      │     pdm ┼ Posdef
      │       │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │           │      │      dn ┼ nothing
      │       │           │      │      dt ┼ nothing
      │       │           │      │     alg ┼ nothing
      │       │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │       │        me ┼ SimpleExpectedReturns
      │       │           │   w ┴ nothing
      │       │   horizon ┴ nothing
      │    mp ┼ MatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │    re ┼ StepwiseRegression
      │       │   crit ┼ PValue
      │       │        │   t ┴ Float64: 0.05
      │       │    alg ┼ ForwardSelection()
      │       │    tgt ┼ LinearModel
      │       │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │    ve ┼ SimpleVariance
      │       │          me ┼ SimpleExpectedReturns
      │       │             │   w ┴ nothing
      │       │           w ┼ nothing
      │       │   corrected ┴ Bool: true
      │   rsd ┴ Bool: true
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
  rsd ┴ Bool: true
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
    function HighOrderFactorPriorEstimator(pe::AbstractLowOrderPriorEstimator_F_AF,
                                           kte::Option{<:CokurtosisEstimator},
                                           ske::Option{<:CoskewnessEstimator},
                                           ex::FLoops.Transducers.Executor, rsd::Bool)
        return new{typeof(pe), typeof(kte), typeof(ske), typeof(ex), typeof(rsd)}(pe, kte,
                                                                                  ske, ex,
                                                                                  rsd)
    end
end
function HighOrderFactorPriorEstimator(;
                                       pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(),
                                       kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
                                                                                       alg = FullMoment()),
                                       ske::Option{<:CoskewnessEstimator} = Coskewness(;
                                                                                       alg = FullMoment()),
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                       rsd::Bool = true)::HighOrderFactorPriorEstimator
    return HighOrderFactorPriorEstimator(pe, kte, ske, ex, rsd)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties HighOrderFactorPriorEstimator begin
    forward(pe, me, ce)
end
"""
    prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum; dims::Int = 1,
          kwargs...)

Compute high order factor prior moments for asset returns using a factor model.

`prior` estimates the mean, covariance, coskewness, and cokurtosis of asset returns using a factor model with residual error correction. It first computes low order moments via the embedded factor prior, then maps factor higher-order moments to asset space via the Kronecker product of the factor loadings, optionally adding residual corrections.

!!! note

    A Black-Litterman prior underneath this estimator now **returns numbers where it used to throw**. Every wrapping estimator forwards `rr` and the factor block under ADR 0046, so `HighOrderFactorPriorEstimator(; pe = BlackLittermanPrior(; pe = FactorPrior(…)))` reaches a regression instead of an `IsNothingError`.

    What comes back is worth understanding. The higher co-moments project through `rr.M` while `mu` and `sigma` carry the views — Black-Litterman makes no claim about third and fourth moments, so the factor projection is the only estimate available.

!!! warning

    The co-moments are computed from `F` as supplied, so they always describe the **pre-view** factor distribution, whichever Black-Litterman member is underneath. Where that member reports a *posterior* factor block — [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) — the nested `fpr` therefore mixes orders: `fpr.mu` and `fpr.sigma` carry the views, `fpr.kt`, `fpr.sk` and `fpr.V` do not. The `fpr.pr === pr.fpr` invariant still holds, because both routes reach the same posterior low order block; what differs is the order at which the views stop.

    This is a consequence of Black-Litterman having no higher-moment update to apply, not of a value being discarded, and it is the same under [`BlackLittermanPrior`](@ref) — where the low order factor block is pre-view too, so the carrier happens to be uniform.

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

The factor comoments come from `pe.kte` and `pe.ske` fit on `F`, so a non-default `alg` on either replaces its display above. Either estimator set to `nothing` drops its moment from both the asset result and the nested factor block.

# Algorithm

 1. Orient `X` and `F` to `observations × variables` with [`dims_oriented`](@ref).
 2. Compute the low order block `pr` with `pe.pe`, and check that it carries a regression result. Take the reconstructed returns `posterior_X = pr.X` and the loadings `M = pr.rr.M`.
 3. Compute the factor square cokurtosis `f_kt` with `pe.kte` on `F`. When it exists, build `kM = kron(M, M)`, project `posterior_kt = kM * f_kt * transpose(kM)`, and process it with `pe.kte.mp`.
 4. Compute the factor coskewness `f_sk` and its negative spectral form `f_V` with `pe.ske` on `F`. When `f_sk` exists, build `kM` if step 3 did not, and project `posterior_sk = M * f_sk * transpose(kM)`.
 5. Build the structure matrices with [`dup_elim_sum_matrices`](@ref), twice: at the asset count for `D2`, `L2` and `S2`, and at the factor count for `f_D2`, `f_L2` and `f_S2`. The all-or-none rule is the one `prior(::HighOrderPriorEstimator, …)` applies, at both dimensions.
 6. When `pe.rsd` is `true`, take the reconstruction error `err = X - posterior_X`.
 7. Still under `pe.rsd`, add [`coskewness_residuals`](@ref)`(err, pe.ske.me)` to the `posterior_sk` of step 4, when there is one.
 8. Still under `pe.rsd`, and when step 3 produced a `posterior_kt`, read the wrapped estimator's residual declaration with [`factor_residual_config`](@ref) and check its shape with [`assert_factor_residual_config`](@ref).
 9. Recover the systematic covariance `sigma` from `pr.sigma`. A `nothing` declaration, and one whose `rsd` is `false`, both mean that no residual block was added, so `sigma` is `pr.sigma` unchanged. Otherwise size the block as `err_sigma`, the column variances of `err` under `rsd_cfg.ve`, subtract its diagonal matrix from `pr.sigma`, and re-condition the difference with [`posdef!`](@ref) under `rsd_cfg.pdm`. When any entry of `err_sigma` exceeds the matching diagonal entry of `pr.sigma` the subtraction would leave a negative variance, so the step warns and keeps `pr.sigma` whole. That happens when the wrapped estimator reports a covariance the block was never added to — a posterior that shrank it, rather than the lift's own sum.
10. Still under `pe.rsd`, add [`cokurtosis_residuals`](@ref)`(sigma, err, pe.kte.me, pe.ex)` to `posterior_kt`, and re-condition the sum with [`posdef!`](@ref) under `pe.kte.mp.pdm`.
11. When step 4 produced a `posterior_sk`, recompute `posterior_V` from it with [`negative_spectral_coskewness`](@ref), so `V` describes the corrected coskewness rather than the projected one.
12. Build the nested factor carrier `fpr` over `pr.fpr`, from the factor moments of steps 3 to 5. It is `nothing` when neither `f_kt` nor `f_sk` exists.
13. Assemble the asset [`HighOrderPrior`](@ref) through its keyword constructor.

Steps 9 and 10 are ordered, not independent. [`cokurtosis_residuals`](@ref) is defined on the systematic covariance, so step 9 has to undo the residual block that the wrapped estimator's own lift added before step 10 adds the residual cokurtosis.

# Arguments

  - `pe`: High order factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Validation

  - `dims in (1, 2)`.
  - The prior produced by `pe.pe` must carry a regression result, via [`assert_prior_regression`](@ref).
  - The wrapped estimator must declare its residual block through [`factor_residual_config`](@ref), which has no default: an estimator that declares nothing throws an `ArgumentError` rather than reading as *no residual block*. The declaration's shape is checked with [`assert_factor_residual_config`](@ref), which throws an `ArgumentError` when it is neither `nothing` nor a `NamedTuple` carrying `ve`, `pdm` and `rsd`. Both raises happen only when `pe.rsd` is `true` and there is a cokurtosis to correct.

# Returns

  - `pr::HighOrderPrior`: Result object containing asset returns, mean, covariance, coskewness tensor, cokurtosis tensor, and factor moments. Its `fpr` is a nested [`HighOrderPrior`](@ref) built over the wrapped prior's own factor block, so `fpr.pr === pr.fpr`: the factor co-moments and the low order factor moments describe one distribution, reachable by either route.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`FactorPrior`](@ref)
  - [`prior`](@ref)
  - [`factor_residual_config`](@ref): the declaration that names the residual block step 9 removes. `pe.pe` is bounded [`AbstractLowOrderPriorEstimator_F_AF`](@ref), and only [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) carry the fields the block is sized from, so a wrapper over either forwards the declaration and everything else declares `nothing` in an explicit method.
  - [`assert_factor_residual_config`](@ref): the shape check that runs on that declaration.
  - [`coskewness_residuals`](@ref)
  - [`cokurtosis_residuals`](@ref)
"""
function prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum; dims::Int = 1,
               kwargs...)
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
    pr = prior(pe.pe, X, F; dims = 1, kwargs...)
    assert_prior_regression(pr, :pe)
    posterior_X = pr.X
    M = pr.rr.M
    f_kt = cokurtosis(pe.kte, F; kwargs...)
    if !isnothing(f_kt)
        kM = kron(M, M)
        posterior_kt = kM * f_kt * transpose(kM)
        matrix_processing!(pe.kte.mp, posterior_kt, posterior_X; kwargs...)
    end
    f_sk, f_V = coskewness(pe.ske, F; kwargs...)
    if !isnothing(f_sk)
        if isnothing(kM)
            kM = kron(M, M)
        end
        posterior_sk = M * f_sk * transpose(kM)
    end
    # The same all-or-none branching the asset block gets, at the factor dimension: the
    # nested carrier validates its own `kt`/`L2`/`S2` triple against `length(pr.fpr.mu)`.
    if !isnothing(f_kt) && !isnothing(f_sk)
        D2, L2, S2 = dup_elim_sum_matrices(size(posterior_X, 2))
        f_D2, f_L2, f_S2 = dup_elim_sum_matrices(size(F, 2))
    elseif !isnothing(f_kt) && isnothing(f_sk)
        L2, S2 = dup_elim_sum_matrices(size(posterior_X, 2))[2:3]
        f_L2, f_S2 = dup_elim_sum_matrices(size(F, 2))[2:3]
    end
    if pe.rsd
        err = X - posterior_X
        if !isnothing(f_sk)
            posterior_sk .+= coskewness_residuals(err, pe.ske.me)
        end
        if !isnothing(f_kt)
            # `cokurtosis_residuals` is defined on the *systematic* covariance, so a residual
            # block the wrapped estimator added has to come back off. Which estimator added one,
            # and with what variance estimator, is a declaration rather than a field read: the
            # `pe` slot is bounded `AbstractLowOrderPriorEstimator_F_AF`, and only `FactorPrior`
            # and `FactorBlackLittermanPrior` carry `ve` and `mp.pdm` — a wrapper over either
            # forwards the declaration, everything else declares `nothing` (see
            # [`factor_residual_config`](@ref)). The shape is checked before the property
            # access, so a wrong declaration names itself here instead of surfacing as a
            # `FieldError` below.
            rsd_cfg = factor_residual_config(pe.pe)
            assert_factor_residual_config(pe.pe, rsd_cfg)
            sigma = if isnothing(rsd_cfg) || !rsd_cfg.rsd
                pr.sigma
            else
                err_sigma = vec(Statistics.var(rsd_cfg.ve, err; dims = 1))
                sigma = if any(map((x, y) -> x > y, err_sigma,
                                   LinearAlgebra.diag(pr.sigma)))
                    @warn("Some residual variances are larger than prior variances; using the prior variances to error correct the posterior kurtosis.")
                    pr.sigma
                else
                    pr.sigma - LinearAlgebra.diagm(err_sigma)
                end
                posdef!(rsd_cfg.pdm, sigma)
                sigma
            end
            err_kt = cokurtosis_residuals(sigma, err, pe.kte.me, pe.ex)
            posterior_kt .+= err_kt
            posdef!(pe.kte.mp.pdm, posterior_kt)
        end
    end
    if !isnothing(f_sk)
        posterior_V = negative_spectral_coskewness(posterior_sk, posterior_X, pe.ske.mp)
    end
    # The nested block's `pr` is the wrapped prior's own factor block, which is what the
    # `fpr.pr === pr.fpr` invariant asks for — the factor co-moments and the factor
    # low-order moments describe one distribution, reachable by either route.
    fpr = if isnothing(f_kt) && isnothing(f_sk)
        nothing
    else
        HighOrderPrior(; pr = pr.fpr, kt = f_kt, D2 = f_D2, L2 = f_L2, S2 = f_S2, sk = f_sk,
                       V = f_V, skmp = isnothing(f_sk) ? nothing : pe.ske.mp)
    end
    return HighOrderPrior(; pr = pr, kt = posterior_kt, D2 = D2, L2 = L2, S2 = S2,
                          sk = posterior_sk, V = posterior_V,
                          skmp = isnothing(f_sk) ? nothing : pe.ske.mp, fpr = fpr)
end

function factor_residual_config(pe::HighOrderFactorPriorEstimator)
    # `pe.rsd` governs the co-moment corrections, not the covariance: the low-order block of
    # the result is the wrapped estimator's own, so this estimator forwards the wrapped
    # declaration (see [`factor_residual_config`](@ref)).
    return factor_residual_config(pe.pe)
end

export HighOrderFactorPriorEstimator
