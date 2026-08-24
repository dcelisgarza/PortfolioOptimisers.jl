"""
    coskewness_residuals(X::MatNum, me::AbstractExpectedReturnsEstimator)

Build the residual coskewness matrix a factor lift adds to its projected coskewness.

`X` is the **residual** matrix `X - posterior_X`, not the asset returns. The residuals are independent across assets and have zero mean, so every cross term of their coskewness vanishes and only the `N` own-third-moment entries survive. The result is therefore an `N × N²` sparse matrix carrying `mean(me, X .^ 3)` on the positions `1:(N² + N + 1):(N² * N)` — entry `(i, (i - 1) * N + i)`, which is where ``\\mathbb{E}[\\varepsilon_i^3]`` sits in a coskewness matrix — and zero everywhere else.

Nothing is demeaned here. The zero-mean assumption is the factor model's, and `me` supplies the averaging rule rather than a centre.

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

Nothing is standardised here. Every entry is written in closed form from the second and fourth residual moments `e2 = mean(me, X .^ 2)` and `e4 = mean(me, X .^ 4)` together with `sigma`, under the factor model's assumption that the residuals have zero mean and are independent both of each other and of the factors. Under those assumptions each of the fourteen index patterns collapses to one of the branches in the loop, and every pattern with a lone index — one asset appearing exactly once — is zero.

Entry `(i - 1) * N + k, (j - 1) * N + l` of the result is the residual contribution to ``\\mathbb{E}[r_i r_k r_j r_l]``. The matrix is symmetric, so only the upper triangle is computed and each value is written to both places.

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
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    mp ┼ MatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │   alg ┼ FullMoment()
      │     w ┴ nothing
  ske ┼ Coskewness
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    mp ┼ MatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │   alg ┼ FullMoment()
      │     w ┴ nothing
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

# Arguments

  - `pe`: High order factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Validation

  - `dims in (1, 2)`.
  - The prior produced by `pe.pe` must carry a regression result, via [`assert_prior_regression`](@ref).

# Returns

  - `pr::HighOrderPrior`: Result object containing asset returns, mean, covariance, coskewness tensor, cokurtosis tensor, and factor moments.

# Details

The factor co-moments are computed from `F` directly and nested as a [`HighOrderPrior`](@ref) over the wrapped prior's own factor block, so `fpr.pr === pr.fpr` — the co-moments and the low order factor moments describe one distribution, reachable by either route.

The residual cokurtosis correction is defined on the **systematic** covariance, so a residual block the wrapped estimator added has to come back off first. Which estimator added one, and with what variance estimator, is a declaration the wrapped estimator makes through [`factor_residual_config`](@ref) rather than a field read — `pe` is bounded [`AbstractLowOrderPriorEstimator_F_AF`](@ref), and only [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) carry the fields. A wrapper over either forwards the declaration; everything else declares `nothing` in an explicit method, and a `nothing` answer — like an answer whose `rsd` is `false` — leaves the covariance alone. There is no default, so a type that declares nothing throws rather than reading as *no residual block*.

!!! note

    A Black-Litterman prior underneath this estimator now **returns numbers where it used to throw**. Every wrapping estimator forwards `rr` and the factor block under ADR 0046, so `HighOrderFactorPriorEstimator(; pe = BlackLittermanPrior(; pe = FactorPrior(…)))` reaches a regression instead of an `IsNothingError`.

    What comes back is worth understanding. The higher co-moments project through `rr.M` while `mu` and `sigma` carry the views — Black-Litterman makes no claim about third and fourth moments, so the factor projection is the only estimate available.

!!! warning

    The co-moments are computed from `F` as supplied, so they always describe the **pre-view** factor distribution, whichever Black-Litterman member is underneath. Where that member reports a *posterior* factor block — [`FactorBlackLittermanPrior`](@ref) and [`BayesianBlackLittermanPrior`](@ref) — the nested `fpr` therefore mixes orders: `fpr.mu` and `fpr.sigma` carry the views, `fpr.kt`, `fpr.sk` and `fpr.V` do not. The `fpr.pr === pr.fpr` invariant still holds, because both routes reach the same posterior low order block; what differs is the order at which the views stop.

    This is a consequence of Black-Litterman having no higher-moment update to apply, not of a value being discarded, and it is the same under [`BlackLittermanPrior`](@ref) — where the low order factor block is pre-view too, so the carrier happens to be uniform.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`FactorPrior`](@ref)
  - [`prior`](@ref)
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
