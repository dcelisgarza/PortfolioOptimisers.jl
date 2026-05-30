"""
    coskewness_residuals(X, me)

Compute the coskewness residuals from asset return data.

Internal helper that demeans `X` using the expected returns from `me` and returns the residual return matrix used in coskewness estimation.

# Arguments

  - `X`: Asset return matrix (observations × assets).
  - `me`: Expected returns estimator.

# Returns

  - Residual return matrix.

# Related

  - [`cokurtosis_residuals`](@ref)
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
    cokurtosis_residuals(sigma, X, me)

Compute the cokurtosis residuals from the covariance matrix and return data.

Internal helper that standardises `X` using the covariance matrix `sigma` and expected returns from `me`, returning the standardised residual matrix used in cokurtosis estimation.

# Arguments

  - `sigma`: Covariance matrix.
  - `X`: Asset return matrix (observations × assets).
  - `me`: Expected returns estimator.

# Returns

  - Standardised residual matrix.

# Related

  - [`coskewness_residuals`](@ref)
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

Represents the High Order Factor Prior Estimator.

`HighOrderFactorPriorEstimator` extends a low-order factor prior with coskewness and cokurtosis moments estimated from a factor model. It supports error correction of higher-order moments using residuals from the factor regression.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderFactorPriorEstimator(;
        pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(),
        kte::Option{<:CokurtosisEstimator} = Cokurtosis(; alg = Full()),
        ske::Option{<:CoskewnessEstimator} = Coskewness(; alg = Full()),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        rsd::Bool = true
    ) -> HighOrderFactorPriorEstimator

Keywords correspond to the struct's fields.

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
      │       │           │      │   alg ┴ Full()
      │       │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
      │       │           │      │     pdm ┼ Posdef
      │       │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │           │      │      dn ┼ nothing
      │       │           │      │      dt ┼ nothing
      │       │           │      │     alg ┼ nothing
      │       │           │      │   order ┴ DenoiseDetoneAlg()
      │       │        me ┼ SimpleExpectedReturns
      │       │           │   w ┴ nothing
      │       │   horizon ┴ nothing
      │    mp ┼ DenoiseDetoneAlgMatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ DenoiseDetoneAlg()
      │    re ┼ StepwiseRegression
      │       │   crit ┼ PValue
      │       │        │   t ┴ Float64: 0.05
      │       │    alg ┼ Forward()
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
      │    mp ┼ DenoiseDetoneAlgMatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ DenoiseDetoneAlg()
      │   alg ┼ Full()
      │     w ┴ nothing
  ske ┼ Coskewness
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    mp ┼ DenoiseDetoneAlgMatrixProcessing
      │       │     pdm ┼ Posdef
      │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      dn ┼ nothing
      │       │      dt ┼ nothing
      │       │     alg ┼ nothing
      │       │   order ┴ DenoiseDetoneAlg()
      │   alg ┼ Full()
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
"""
@concrete struct HighOrderFactorPriorEstimator <: AbstractHighOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:kte])
    """
    kte
    """
    $(field_dict[:ske])
    """
    ske
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
                                                                                       alg = Full()),
                                       ske::Option{<:CoskewnessEstimator} = Coskewness(;
                                                                                       alg = Full()),
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                       rsd::Bool = true)::HighOrderFactorPriorEstimator
    return HighOrderFactorPriorEstimator(pe, kte, ske, ex, rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`HighOrderFactorPriorEstimator`](@ref) estimator with observation weights `w` applied to the underlying prior, cokurtosis, and coskewness estimators.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`factory`](@ref)
"""
function factory(pe::HighOrderFactorPriorEstimator,
                 w::ObsWeights)::HighOrderFactorPriorEstimator
    return HighOrderFactorPriorEstimator(; pe = factory(pe.pe, w), kte = factory(pe.kte, w),
                                         ske = factory(pe.ske, w), ex = pe.ex, rsd = pe.rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`HighOrderFactorPriorEstimator`](@ref) estimator restricted to the assets at index `i`.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`prior_view`](@ref)
"""
function prior_view(pe::HighOrderFactorPriorEstimator, i)::HighOrderFactorPriorEstimator
    return HighOrderFactorPriorEstimator(; pe = prior_view(pe.pe, i), kte = pe.kte,
                                         ske = pe.ske, ex = pe.ex, rsd = pe.rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`HighOrderFactorPriorEstimator`](@ref). Exposes `:me` and `:ce` from the embedded prior estimator `obj.pe` for transparent access.
"""
function Base.getproperty(obj::HighOrderFactorPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
"""
    prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum; dims::Int = 1,
          kwargs...)

Compute high order factor prior moments for asset returns using a factor model.

`prior` estimates the mean, covariance, coskewness, and cokurtosis of asset returns using a factor model with residual error correction. It first computes low order moments via the embedded factor prior, then maps factor higher-order moments to asset space via the Kronecker product of the factor loadings, optionally adding residual corrections.

# Mathematical definition

Factor cokurtosis and coskewness are mapped to asset space via the loadings matrix ``\\mathbf{B}`` (with Kronecker product ``\\otimes``):

```math
\\begin{align}
\\hat{\\mathbf{K}} &= (\\mathbf{B} \\otimes \\mathbf{B}) \\hat{\\mathbf{K}}_f (\\mathbf{B} \\otimes \\mathbf{B})^\\intercal + \\hat{\\mathbf{K}}_\\varepsilon\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{S}} &= \\mathbf{B} \\hat{\\mathbf{S}}_f (\\mathbf{B} \\otimes \\mathbf{B})^\\intercal + \\hat{\\mathbf{S}}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{K}}``: ``N^2 \\times N^2`` asset cokurtosis matrix.
  - ``\\hat{\\mathbf{S}}``: ``N \\times N^2`` asset coskewness matrix.
  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix.
  - ``\\hat{\\mathbf{K}}_f``: ``K^2 \\times K^2`` factor cokurtosis matrix.
  - ``\\hat{\\mathbf{S}}_f``: ``K \\times K^2`` factor coskewness matrix.
  - ``\\hat{\\mathbf{K}}_\\varepsilon``: Residual cokurtosis correction (when `rsd = true`).
  - ``\\hat{\\mathbf{S}}_\\varepsilon``: Residual coskewness correction (when `rsd = true`).
  - ``\\otimes``: Kronecker product.

# Arguments

  - `pe`: High order factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Returns

  - `pr::HighOrderPrior`: Result object containing asset returns, mean, covariance, coskewness tensor, cokurtosis tensor, and factor moments.

# Validation

  - `dims in (1, 2)`.

# Related

  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`FactorPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum; dims::Int = 1,
               kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    kM = nothing
    L2 = nothing
    S2 = nothing
    posterior_kt = nothing
    posterior_sk = nothing
    posterior_V = nothing
    pr = prior(pe.pe, X, F; dims = 1, kwargs...)
    posterior_X = pr.X
    M = pr.rr.M
    f_kt = cokurtosis(pe.kte, F; kwargs...)
    if !isnothing(f_kt)
        kM = kron(M, M)
        L2, S2 = dup_elim_sum_matrices(size(posterior_X, 2))[2:3]
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
    if pe.rsd
        err = X - posterior_X
        if !isnothing(f_sk)
            posterior_sk .+= coskewness_residuals(err, pe.ske.me)
        end
        if !isnothing(f_kt)
            if isnothing(pr.chol)
                sigma = pr.sigma
            else
                err_sigma = vec(Statistics.var(pe.pe.ve, err; dims = 1))
                sigma = if any(map((x, y) -> x > y, err_sigma,
                                   LinearAlgebra.diag(pr.sigma)))
                    @warn("Some residual variances are larger than prior variances; using the prior variances to error correct the posterior kurtosis.")
                    pr.sigma
                else
                    pr.sigma - LinearAlgebra.diagm(err_sigma)
                end
                posdef!(pe.pe.mp.pdm, sigma)
            end
            err_kt = cokurtosis_residuals(sigma, err, pe.kte.me, pe.ex)
            posterior_kt .+= err_kt
            posdef!(pe.kte.mp.pdm, posterior_kt)
        end
    end
    if !isnothing(f_sk)
        posterior_V = negative_spectral_coskewness(posterior_sk, posterior_X, pe.ske.mp)
    end
    return HighOrderPrior(; pr = pr, kt = posterior_kt, L2 = L2, S2 = S2, sk = posterior_sk,
                          V = posterior_V, skmp = isnothing(f_sk) ? nothing : pe.ske.mp,
                          f_kt = f_kt, f_sk = f_sk, f_V = f_V)
end

export HighOrderFactorPriorEstimator
