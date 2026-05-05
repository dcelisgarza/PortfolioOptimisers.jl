"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance estimators in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing Smyth-Broby covariance estimation algorithms should be subtypes of `BaseSmythBrobyCovariance`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
"""
abstract type BaseSmythBrobyCovariance <: BaseGerberCovariance end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific Smyth-Broby covariance algorithms should be subtypes of `SmythBrobyCovarianceAlgorithm`.

These types are used to specify the algorithm when constructing a [`SmythBrobyCovariance`](@ref) estimator.

# Related

  - [`BaseSmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovariance`](@ref)
"""
abstract type SmythBrobyCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the original Smyth-Broby covariance algorithm.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
"""
struct SmythBroby0 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the first variant of the Smyth-Broby covariance algorithm.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby2`](@ref)
"""
struct SmythBroby1 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the second variant of the Smyth-Broby covariance algorithm.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
"""
struct SmythBroby2 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the original Smyth-Broby covariance algorithm scaled by vote counts.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
"""
struct SmythBrobyGerber0 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the first variant of the Smyth-Broby covariance algorithm scaled by vote counts.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber2`](@ref)
"""
struct SmythBrobyGerber1 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the second variant of the Smyth-Broby covariance algorithm scaled by vote counts.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
"""
struct SmythBrobyGerber2 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the original Smyth-Broby covariance algorithm using vote counts only.

# Related

    - [`SmythBrobyCovarianceAlgorithm`](@ref)
    - [`SmythBrobyCovariance`](@ref)
    - [`SmythBrobyCount1`](@ref)
    - [`SmythBrobyCount2`](@ref)
"""
struct SmythBrobyCount0 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the first variant of the Smyth-Broby covariance algorithm using vote counts only.

# Related

    - [`SmythBrobyCovarianceAlgorithm`](@ref)
    - [`SmythBrobyCovariance`](@ref)
    - [`SmythBrobyCount0`](@ref)
    - [`SmythBrobyCount2`](@ref)
"""
struct SmythBrobyCount1 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the second variant of the Smyth-Broby covariance algorithm using vote counts only.

# Related

    - [`SmythBrobyCovarianceAlgorithm`](@ref)
    - [`SmythBrobyCovariance`](@ref)
    - [`SmythBrobyCount0`](@ref)
    - [`SmythBrobyCount1`](@ref)
"""
struct SmythBrobyCount2 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

A flexible container type for configuring and applying Smyth-Broby covariance estimators in `PortfolioOptimisers.jl`.

`SmythBrobyCovariance` encapsulates all components required for Smyth-Broby-based covariance or correlation estimation, including the expected returns estimator, variance estimator, positive definite matrix estimator, algorithm parameters, and the specific Smyth-Broby algorithm variant.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SmythBrobyCovariance(;
        ve::StatsBase.CovarianceEstimator = SimpleVariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        pdm::Option{<:Posdef} = Posdef(),
        c1::Number = 0.5,
        c2::Number = 0.5,
        c3::Number = 4,
        n::Number = 2,
        alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1(),
        ex::FLoops.Transducers.Executor = ThreadedEx()
    ) -> SmythBrobyCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:c3c2])

# Examples

```jldoctest
julia> SmythBrobyCovariance()
SmythBrobyCovariance
   ve ┼ SimpleVariance
      │          me ┼ SimpleExpectedReturns
      │             │   w ┴ nothing
      │           w ┼ nothing
      │   corrected ┴ Bool: true
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
   c1 ┼ Float64: 0.5
   c2 ┼ Float64: 0.5
   c3 ┼ Int64: 4
    n ┼ Int64: 2
  alg ┼ SmythBrobyGerber1()
   ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`BaseSmythBrobyCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`SimpleVariance`](@ref)
  - [`Posdef`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
"""
@concrete struct SmythBrobyCovariance <: BaseSmythBrobyCovariance
    "$(field_dict[:ve])"
    ve
    "$(field_dict[:me]) Used for optionally centering the returns."
    me
    "$(field_dict[:pdm])"
    pdm
    "$(field_dict[:c1])"
    c1
    "$(field_dict[:c2])"
    c2
    "$(field_dict[:c3])"
    c3
    "$(field_dict[:sbn])"
    n
    "$(field_dict[:sbalg])"
    alg
    "$(field_dict[:ex])"
    ex
    function SmythBrobyCovariance(ve::StatsBase.CovarianceEstimator,
                                  me::AbstractExpectedReturnsEstimator,
                                  pdm::Option{<:Posdef}, c1::Number, c2::Number, c3::Number,
                                  n::Number, alg::SmythBrobyCovarianceAlgorithm,
                                  ex::FLoops.Transducers.Executor)
        assert_nonempty_nonneg_finite_val(c1, :c1)
        assert_nonempty_nonneg_finite_val(c2, :c2)
        assert_nonempty_nonneg_finite_val(c3, :c3)
        @argcheck(c2 < c3)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(c1), typeof(c2), typeof(c3),
                   typeof(n), typeof(alg), typeof(ex)}(ve, me, pdm, c1, c2, c3, n, alg, ex)
    end
end
function SmythBrobyCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                              me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                              pdm::Option{<:Posdef} = Posdef(), c1::Number = 0.5,
                              c2::Number = 0.5, c3::Number = 4, n::Number = 2,
                              alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1(),
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    return SmythBrobyCovariance(ve, me, pdm, c1, c2, c3, n, alg, ex)
end
"""
    factory(ce::SmythBrobyCovariance, w::ObsWeights) -> SmythBrobyCovariance

Return a new [`SmythBrobyCovariance`](@ref) estimator with observation weights `w` applied to the underlying variance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::SmythBrobyCovariance, w::ObsWeights)
    return SmythBrobyCovariance(; ve = factory(ce.ve, w), me = factory(ce.me, w),
                                pdm = ce.pdm, c1 = ce.c1, c2 = ce.c2, c3 = ce.c3, n = ce.n,
                                alg = factory(ce.alg, w), ex = ce.ex)
end
"""
    sb_delta(ri::Number, rj::Number, n::Number) -> Number

Smyth-Broby kernel function for covariance and correlation computation.

This function computes the kernel value for a pair of asset returns, applying the Smyth-Broby logic for zones of confusion and indecision. It is used to aggregate positive and negative co-movements in Smyth-Broby covariance algorithms. It assumes the returns are centered around zero.

# Arguments

  - `ri`: Absolute standardised return for asset `i`.
  - `rj`: Absolute standardised return for asset `j`.
  - `n`: Exponent parameter for the kernel.

# Returns

  - `score::Number`: The computed score for the pair `(xi, xj)`.

# Details

  - Returns `(sqrt((1 + ri) * (1 + rj)) / (1 + abs(ri - rj)^n), 1)`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`smythbroby`](@ref)
"""
function sb_delta(ri::Number, rj::Number, n::Number)
    kappa = sqrt((one(ri) + ri) * (one(rj) + rj))
    gamma = abs(ri - rj)
    return kappa / (one(gamma) + gamma^n)
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBroby0}, X::MatNum, mu::ArrNum,
               sd::ArrNum)

Implements the original Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the original `SmythBroby0` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBroby0` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions.
 5. The correlation is computed as `(pos - neg) / (pos + neg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBroby0}, X::MatNum, mu::ArrNum,
                    sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += sb_delta(ari, arj, n)
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += sb_delta(ari, arj, n)
                end
            end
            den = pos + neg
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBroby1}, X::MatNum, mu::ArrNum,
               sd::ArrNum)

Implements the first variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the original `SmythBroby1` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBroby1` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive, negative, and neutral co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`), negative (`neg`), and neutral (`nn`) contributions.
 5. The correlation is computed as `(pos - neg) / (pos + neg + nn)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBroby1}, X::MatNum, mu::ArrNum,
                    sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += sb_delta(ari, arj, n)
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += sb_delta(ari, arj, n)
                else
                    nn += sb_delta(ari, arj, n)
                end
            end
            den = pos + neg + nn
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBroby2}, X::MatNum, mu::ArrNum,
               sd::ArrNum)

Implements the second variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBroby2` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements. The resulting matrix is then standardised by the geometric mean of its diagonal elements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBroby2` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, standardised and projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions.
 5. The raw correlation is computed as `pos - neg`.
 6. The resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.
 7. The matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby2`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBroby2}, X::MatNum, mu::ArrNum,
                    sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += sb_delta(ari, arj, n)
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += sb_delta(ari, arj, n)
                end
            end
            rho[j, i] = rho[i, j] = pos - neg
        end
    end
    h = max.(sqrt.(LinearAlgebra.diag(rho)), sqrt(eps(eltype(rho))))
    rho .= LinearAlgebra.Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBrobyGerber0}, X::MatNum,
               mu::ArrNum, sd::ArrNum)

Implements the original Gerber-style variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber0` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber0` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBrobyGerber0}, X::MatNum,
                    mu::ArrNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += sb_delta(ari, arj, n)
                    cpos += 1
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += sb_delta(ari, arj, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = tpos + tneg
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBrobyGerber1}, X::MatNum,
               mu::ArrNum, sd::ArrNum)

Implements the first Gerber-style variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber1` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive, negative, and neutral co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber1` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive, negative, and neutral co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`), negative (`neg`), and neutral (`nn`) contributions, and count the number of positive (`cpos`), negative (`cneg`), and neutral (`cnn`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg + nn * cnn)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBrobyGerber1}, X::MatNum,
                    mu::ArrNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += sb_delta(ari, arj, n)
                    cpos += 1
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += sb_delta(ari, arj, n)
                    cneg += 1
                else
                    nn += sb_delta(ari, arj, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = tpos + tneg + tnn
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBrobyGerber2}, X::MatNum,
               mu::ArrNum, sd::ArrNum)

Implements the second Gerber-style variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber2` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements. The resulting matrix is then standardised by the geometric mean of its diagonal elements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber2` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, standardised and projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The raw correlation is computed as `pos * cpos - neg * cneg`.
 6. The resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.
 7. The matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBrobyGerber2}, X::MatNum,
                    mu::ArrNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        sigmaj = sd[j]
        muj = mu[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += sb_delta(ari, arj, n)
                    cpos += 1
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += sb_delta(ari, arj, n)
                    cneg += 1
                end
            end
            rho[j, i] = rho[i, j] = pos * cpos - neg * cneg
        end
    end
    h = max.(sqrt.(LinearAlgebra.diag(rho)), sqrt(eps(eltype(rho))))
    rho .= LinearAlgebra.Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBrobyCount0}, X::MatNum,
               mu::ArrNum, sd::ArrNum)

Implements the original Gerber-style variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber0` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber0` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBrobyCount0}, X::MatNum,
                    mu::ArrNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = 0
            pos = 0
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += 1
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += 1
                end
            end
            den = pos + neg
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBrobyCount1}, X::MatNum,
                mu::ArrNum, sd::ArrNum)

Implements the first Gerber-style variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber1` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive, negative, and neutral co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber1` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive, negative, and neutral co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`), negative (`neg`), and neutral (`nn`) contributions, and count the number of positive (`cpos`), negative (`cneg`), and neutral (`cnn`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg + nn * cnn)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBrobyCount1}, X::MatNum,
                    mu::ArrNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = 0
            pos = 0
            nn = 0
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += 1
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += 1
                else
                    nn += 1
                end
            end
            den = pos + neg + nn
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Any, <:SmythBrobyCount2}, X::MatNum,
                mu::ArrNum, sd::ArrNum)

Implements the second Gerber-style variant of the Smyth-Broby covariance/correlation algorithm.

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber2` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements. The resulting matrix is then standardised by the geometric mean of its diagonal elements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber2` algorithm.
  - `X`: Data matrix (observations × assets).
  - `sd`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, standardised and projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The raw correlation is computed as `pos * cpos - neg * cneg`.
 6. The resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.
 7. The matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:SmythBrobyCount2}, X::MatNum,
                    mu::ArrNum, sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    FLoops.@floop ce.ex for j in axes(X, 2)
        muj = mu[j]
        sigmaj = sd[j]
        c1j = c1 * sigmaj
        for i in 1:j
            neg = 0
            pos = 0
            mui = mu[i]
            sigmai = sd[i]
            c1i = c1 * sigmai
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                if abs(xi) < c1i && abs(xj) < c1j
                    continue
                end
                ri = (xi - mui) / sigmai
                rj = (xj - muj) / sigmaj
                ari = abs(ri)
                arj = abs(rj)
                if ari > c3 || arj > c3 || ari < c2 && arj < c2
                    continue
                end
                if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
                    pos += 1
                elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
                    neg += 1
                end
            end
            rho[j, i] = rho[i, j] = pos - neg
        end
    end
    h = max.(sqrt.(LinearAlgebra.diag(rho)), sqrt(eps(eltype(rho))))
    rho .= LinearAlgebra.Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    Statistics.cor(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Smyth-Broby correlation matrix.

This method computes the Smyth-Broby correlation matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Smyth-Broby correlation is then computed via [`smythbroby`](@ref).

# Arguments

  - `ce`: Smyth-Broby covariance estimator.

      + `ce::SmythBrobyCovariance`: Compute the unstandardised Smyth-Broby correlation matrix.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`smythbroby`](@ref)
  - [`cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    mu = Statistics.mean(ce.me, X; dims = 1, kwargs...)
    return smythbroby(ce, X, mu, sd)
end
"""
    Statistics.cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Smyth-Broby covariance matrix.

This method computes the Smyth-Broby covariance matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Smyth-Broby covariance is then computed via [`smythbroby`](@ref).

# Arguments

  - `ce`: Smyth-Broby covariance estimator.

      + `ce::SmythBrobyCovariance`: Compute the unstandardised Smyth-Broby covariance matrix.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `sigma::Matrix{<:Number}`: The Smyth-Broby covariance matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`smythbroby`](@ref)
  - [`cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    mu = Statistics.mean(ce.me, X; dims = 1, kwargs...)
    sigma = smythbroby(ce, X, mu, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export SmythBroby0, SmythBroby1, SmythBroby2, SmythBrobyGerber0, SmythBrobyGerber1,
       SmythBrobyGerber2, SmythBrobyCount0, SmythBrobyCount1, SmythBrobyCount2,
       SmythBrobyCovariance
