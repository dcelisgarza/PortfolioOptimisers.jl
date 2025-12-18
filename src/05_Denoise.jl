"""
    abstract type AbstractDenoiseEstimator <: AbstractEstimator end

Abstract supertype for all denoising estimator types in `PortfolioOptimisers.jl`.

All concrete types that implement denoising of covariance or correlation matrices should subtype `AbstractDenoiseEstimator`. This enables a consistent interface for denoising routines throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`Denoise`](@ref)
  - [`denoise!`](@ref)
  - [`denoise`](@ref)
"""
abstract type AbstractDenoiseEstimator <: AbstractEstimator end
"""
    abstract type AbstractDenoiseAlgorithm <: AbstractAlgorithm end

Abstract supertype for all denoising algorithm types in `PortfolioOptimisers.jl`.

All concrete types that implement a specific denoising algorithm should subtype `AbstractDenoiseAlgorithm`. This enables flexible extension and dispatch of denoising routines.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
"""
abstract type AbstractDenoiseAlgorithm <: AbstractAlgorithm end
"""
    struct SpectralDenoise <: AbstractDenoiseAlgorithm end

A denoising algorithm that sets the smallest `num_factors` eigenvalues of a covariance or correlation matrix to zero, effectively removing the principal components relating to random noise according to random matrix theory-based approaches.

# Examples

```jldoctest
julia> SpectralDenoise()
SpectralDenoise()
```

# Related

  - [`AbstractDenoiseAlgorithm`](@ref)
  - [`denoise!`](@ref)
  - [`Denoise`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
struct SpectralDenoise <: AbstractDenoiseAlgorithm end
"""
    struct FixedDenoise <: AbstractDenoiseAlgorithm end

A denoising algorithm that replaces the smallest `num_factors` eigenvalues of a covariance or correlation matrix with their average, effectively averaging the principal components relating to random noise according to random matrix theory-based approaches.

# Examples

```jldoctest
julia> FixedDenoise()
FixedDenoise()
```

# Related

  - [`AbstractDenoiseAlgorithm`](@ref)
  - [`denoise!`](@ref)
  - [`Denoise`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
struct FixedDenoise <: AbstractDenoiseAlgorithm end
"""
    struct ShrunkDenoise{T1} <: AbstractDenoiseAlgorithm
        alpha::T1
    end

A denoising algorithm that shrinks the smallest `num_factors` eigenvalues of a covariance or correlation matrix towards their diagonal, controlled by the shrinkage parameter `alpha`. This approach interpolates between no shrinkage (`alpha = 0`) and full shrinkage (`alpha = 1`), providing a flexible way to regularize noisy eigenvalues.

# Fields

  - `alpha`: The shrinkage parameter controlling the degree of shrinkage applied to the smallest eigenvalues.

# Constructor

    ShrunkDenoise(; alpha::Number = 0.0)

Keyword arguments correspond to the fields above.

## Validation

  - `0 <= alpha <= 1`.

# Examples

```jldoctest
julia> ShrunkDenoise(; alpha = 0.5)
ShrunkDenoise
  alpha ┴ Float64: 0.5
```

# Related

  - [`AbstractDenoiseAlgorithm`](@ref)
  - [`denoise!`](@ref)
  - [`Denoise`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
struct ShrunkDenoise{T1} <: AbstractDenoiseAlgorithm
    alpha::T1
    function ShrunkDenoise(alpha::Number)
        @argcheck(zero(alpha) <= alpha <= one(alpha),
                  DomainError("0 <= alpha <= 1 must hold. Got\nalpha => $alpha"))
        return new{typeof(alpha)}(alpha)
    end
end
function ShrunkDenoise(; alpha::Number = 0.0)
    return ShrunkDenoise(alpha)
end
"""
    struct Denoise{T1, T2, T3, T4, T5, T6, T7} <: AbstractDenoiseEstimator
        alg::T1
        args::T2
        kwargs::T3
        kernel::T4
        m::T5
        n::T6
        pdm::T7
    end

A flexible container type for configuring and applying denoising algorithms to covariance or correlation matrices in `PortfolioOptimisers.jl`.

`Denoise` encapsulates all parameters required for matrix denoising in [`denoise!`](@ref) and [`denoise`](@ref), allowing users to specify the denoising algorithm, optimization parameters, kernel settings for density estimation, and optional positive definite matrix projection.

# Fields

  - `alg`: Denoising algorithm.
  - `args`: Positional arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).
  - `kwargs`: Keyword arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).
  - `kernel`: Kernel function for [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
  - `m`: Number of adjacent histograms to smooth over in [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
  - `n`: Number of points in the range of eigenvalues used in the average shifted histogram density estimation.
  - `pdm`: Optional Positive definite matrix estimator. If provided, ensures the output is positive definite.

# Constructor

    Denoise(; alg::AbstractDenoiseAlgorithm = ShrunkDenoise(), m::Integer = 10,
            n::Integer = 1000, kernel::Any = AverageShiftedHistograms.Kernels.gaussian,
            args::Tuple = (), kwargs::NamedTuple = (;), pdm::Option{<:Posdef} = Posdef())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> Denoise()
Denoise
     alg ┼ ShrunkDenoise
         │   alpha ┴ Float64: 0.0
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
  kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
       m ┼ Int64: 10
       n ┼ Int64: 1000
     pdm ┼ Posdef
         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │   kwargs ┴ @NamedTuple{}: NamedTuple()

julia> Denoise(; alg = SpectralDenoise(), m = 20, n = 500)
Denoise
     alg ┼ SpectralDenoise()
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
  kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
       m ┼ Int64: 20
       n ┼ Int64: 500
     pdm ┼ Posdef
         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractDenoiseEstimator`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
  - [`denoise!`](@ref)
  - [`denoise`](@ref)
  - [`AverageShiftedHistograms.Kernels`](https://joshday.github.io/AverageShiftedHistograms.jl/stable/kernels/)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
struct Denoise{T1, T2, T3, T4, T5, T6, T7} <: AbstractDenoiseEstimator
    alg::T1
    args::T2
    kwargs::T3
    kernel::T4
    m::T5
    n::T6
    pdm::T7
    function Denoise(alg::AbstractDenoiseAlgorithm, args::Tuple, kwargs::NamedTuple, kernel,
                     m::Integer, n::Integer, pdm::Option{<:Posdef} = Posdef())
        @argcheck(1 < m, DomainError)
        @argcheck(1 < n, DomainError)
        return new{typeof(alg), typeof(args), typeof(kwargs), typeof(kernel), typeof(m),
                   typeof(n), typeof(pdm)}(alg, args, kwargs, kernel, m, n, pdm)
    end
end
function Denoise(; alg::AbstractDenoiseAlgorithm = ShrunkDenoise(), args::Tuple = (),
                 kwargs::NamedTuple = (;),
                 kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                 n::Integer = 1000, pdm::Option{<:Posdef} = Posdef())
    return Denoise(alg, args, kwargs, kernel, m, n, pdm)
end
"""
    _denoise!(de::AbstractDenoiseAlgorithm, X::MatNum, vals::VecNum, vecs::MatNum,
              num_factors::Integer)

In-place denoising of a covariance or correlation matrix using a specific denoising algorithm.

These methods are called internally by [`denoise!`](@ref) and [`denoise`](@ref) when a [`Denoise`](@ref) estimator is used, and should not typically be called directly.

# Arguments

  - `alg`: The denoising algorithm to apply.
  - `X`: The matrix to be denoised (modified in-place).
  - `vals`: Eigenvalues of `X`, sorted in ascending order.
  - `vecs`: Corresponding eigenvectors of `X`.
  - `num_factors`: Number of eigenvalues to treat as noise.
  - `pdm`: Positive definite matrix estimator.

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Related

  - [`denoise!`](@ref)
  - [`Denoise`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
  - [`MatNum`](@ref)
  - [`VecNum`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function _denoise!(::SpectralDenoise, X::MatNum, vals::VecNum, vecs::MatNum,
                   num_factors::Integer)
    vals[1:num_factors] .= zero(eltype(X))
    X .= cov2cor(vecs * Diagonal(vals) * transpose(vecs))
    return nothing
end
function _denoise!(::FixedDenoise, X::MatNum, vals::VecNum, vecs::MatNum,
                   num_factors::Integer)
    vals[1:num_factors] .= sum(vals[1:num_factors]) / num_factors
    X .= cov2cor(vecs * Diagonal(vals) * transpose(vecs))
    return nothing
end
function _denoise!(de::ShrunkDenoise, X::MatNum, vals::VecNum, vecs::MatNum,
                   num_factors::Integer)
    # Small
    vals_l = vals[1:num_factors]
    vecs_l = vecs[:, 1:num_factors]

    # Large
    vals_r = vals[(num_factors + 1):end]
    vecs_r = vecs[:, (num_factors + 1):end]

    corr0 = vecs_r * Diagonal(vals_r) * transpose(vecs_r)
    corr1 = vecs_l * Diagonal(vals_l) * transpose(vecs_l)

    X .= corr0 + de.alpha * corr1 + (one(de.alpha) - de.alpha) * Diagonal(corr1)
    return nothing
end
"""
    errPDF(x::Number, vals::VecNum, q::Number,
           kernel::Any = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
           n::Integer = 1000)

Compute the sum of squared errors (SSE) between the theoretical Marčenko–Pastur (MP) eigenvalue density and the empirical eigenvalue density estimated from observed eigenvalues.

This function is used internally to fit the MP distribution to the observed spectrum, as part of the denoising procedure.

# Arguments

  - `x`: Scale parameter for the MP distribution `[0, 1]`.
  - `vals`: Observed eigenvalues.
  - `q`: Effective sample ratio (e.g., `n_obs / n_assets`).
  - `kernel`: Kernel function for [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
  - `m`: Number of adjacent histograms to smooth over.
  - `n`: Number of points in the range of eigenvalues for density estimation.

# Returns

  - `sse::Number`: The sum of squared errors between the empirical and theoretical densities.

# Related

  - [`find_max_eval`](@ref)
  - [`Denoise`](@ref)
  - [`VecNum`](@ref)
  - [`AverageShiftedHistograms.Kernels`](https://joshday.github.io/AverageShiftedHistograms.jl/stable/kernels/)

# References

  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function errPDF(x::Number, vals::VecNum, q::Number,
                kernel::Any = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                n::Integer = 1000)
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max; length = n)
    pdf1 = q ⊘ (2 * pi * x * rg) ⊙
           sqrt.(clamp.((e_max .- rg) ⊙ (rg .- e_min), zero(x), typemax(x)))
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    res = ash(vals; rng = range(e_min, e_max; length = n), kernel = kernel, m = m)
    pdf2 = [AverageShiftedHistograms.pdf(res, i) for i in pdf1]
    pdf2[.!isfinite.(pdf2)] .= zero(q)
    sse = sum((pdf2 - pdf1) .^ 2)
    return sse
end
"""
    find_max_eval(vals::VecNum, q::Number,
                  kernel::Any = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                  n::Integer = 1000, args::Tuple = (), kwargs::NamedTuple = (;))

Estimate the upper edge of the Marčenko–Pastur (MP) distribution for a set of eigenvalues, used to separate signal from noise in random matrix denoising.

This function fits the MP distribution to the observed spectrum by minimizing the sum of squared errors between the empirical and theoretical densities, and returns the estimated maximum eigenvalue for noise.

# Arguments

  - `vals`: Observed eigenvalues (typically sorted in ascending order).
  - `q`: Effective sample ratio (e.g., `n_obs / n_assets`).
  - `kernel`: Kernel function for [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
  - `m`: Number of adjacent histograms to smooth over.
  - `n`: Number of points in the range of eigenvalues for density estimation.
  - `args`: Additional positional arguments for [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).
  - `kwargs`: Additional keyword arguments for [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).

# Returns

  - `e_max::Number`: Estimated upper edge of the noise eigenvalue spectrum.
  - `x::Number`: Fitted scale parameter.

# Related

  - [`errPDF`](@ref)
  - [`Denoise`](@ref)
  - [`VecNum`](@ref)
  - [`AverageShiftedHistograms.Kernels`](https://joshday.github.io/AverageShiftedHistograms.jl/stable/kernels/)

# References

  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function find_max_eval(vals::VecNum, q::Number,
                       kernel::Any = AverageShiftedHistograms.Kernels.gaussian,
                       m::Integer = 10, n::Integer = 1000, args::Tuple = (),
                       kwargs::NamedTuple = (;))
    res = Optim.optimize(x -> errPDF(x, vals, q, kernel, m, n), 0.0, 1.0, args...;
                         kwargs...)
    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0
    e_max = x * (one(q) + sqrt(one(q) / q))^2
    return e_max, x
end
"""
    denoise!(de::Denoise, X::MatNum, q::Number)
    denoise!(::Nothing, args...)

In-place denoising of a covariance or correlation matrix using a [`Denoise`](@ref) estimator.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Arguments

  - `de`: The estimator specifying the denoising algorithm.

      + `de::Denoise`: The specified denoising algorithm is applied to `X` in-place.
      + `de::Nothing`: No-op.

  - `X`: The covariance or correlation matrix to be denoised (modified in-place).
  - `q`: The effective sample ratio (e.g., `n_obs / n_assets`), used for spectral thresholding.

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);

julia> X = X' * X
5×5 Matrix{Float64}:
 3.29494  2.0765   1.73334  2.01524  1.77493
 2.0765   2.46967  1.39953  1.97242  2.07886
 1.73334  1.39953  1.90712  1.17071  1.30459
 2.01524  1.97242  1.17071  2.24818  1.87091
 1.77493  2.07886  1.30459  1.87091  2.44414

julia> denoise!(Denoise(), X, 10 / 5)

julia> X
5×5 Matrix{Float64}:
 3.29494  2.28883  1.70633  2.12343  2.17377
 2.28883  2.46967  1.59575  1.98583  2.0329
 1.70633  1.59575  1.90712  1.48044  1.51553
 2.12343  1.98583  1.48044  2.24818  1.886
 2.17377  2.0329   1.51553  1.886    2.44414
```

# Related

  - [`denoise`](@ref)
  - [`Denoise`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
  - [`posdef!`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function denoise!(::Nothing, args...)
    return nothing
end
function denoise!(de::Denoise, X::MatNum, q::Number)
    assert_matrix_issquare(X, :X)
    s = diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = eigen(X)
    max_val = find_max_eval(vals, q, de.kernel, de.m, de.n, de.args, de.kwargs)[1]
    num_factors = searchsortedlast(vals, max_val)
    _denoise!(de.alg, X, vals, vecs, num_factors)
    posdef!(de.pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
"""
    denoise(de::Denoise, X::MatNum, q::Number)
    denoise(::Nothing, args...)

Out-of-place version of [`denoise!`](@ref).

# Related

  - [`denoise!`](@ref)
  - [`Denoise`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
  - [`posdef`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function denoise(::Nothing, args...)
    return nothing
end
function denoise(de::Denoise, X::MatNum, q::Number)
    X = copy(X)
    denoise!(de, X, q)
    return X
end

export Denoise, SpectralDenoise, FixedDenoise, ShrunkDenoise, denoise, denoise!
