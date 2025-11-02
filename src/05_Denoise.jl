"""
    abstract type AbstractDenoiseEstimator <: AbstractEstimator end

Abstract supertype for all denoising estimator types in PortfolioOptimisers.jl.

All concrete types that implement denoising of covariance or correlation matrices (e.g., via spectral, fixed, or shrinkage methods) should subtype `AbstractDenoiseEstimator`. This enables a consistent interface for denoising routines throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`Denoise`](@ref)
  - [`denoise!`](@ref)
  - [`denoise`](@ref)
"""
abstract type AbstractDenoiseEstimator <: AbstractEstimator end
"""
    abstract type AbstractDenoiseAlgorithm <: AbstractAlgorithm end

Abstract supertype for all denoising algorithm types in PortfolioOptimisers.jl.

All concrete types that implement a specific denoising algorithm (e.g., spectral, fixed, shrinkage) should subtype `AbstractDenoiseAlgorithm`. This enables flexible extension and dispatch of denoising routines.

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

    ShrunkDenoise(; alpha::Real = 0.0)

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
"""
struct ShrunkDenoise{T1} <: AbstractDenoiseAlgorithm
    alpha::T1
    function ShrunkDenoise(alpha::Real)
        @argcheck(zero(alpha) <= alpha <= one(alpha),
                  DomainError(alpha,
                              range_msg("`alpha`", zero(alpha), one(alpha), nothing, true,
                                        true) * "."))
        return new{typeof(alpha)}(alpha)
    end
end
function ShrunkDenoise(; alpha::Real = 0.0)
    return ShrunkDenoise(alpha)
end
"""
    struct Denoise{T1, T2, T3, T4, T5, T6} <: AbstractDenoiseEstimator
        alg::T1
        args::T2
        kwargs::T3
        kernel::T4
        m::T5
        n::T6
    end

A flexible container type for configuring and applying denoising algorithms to covariance or correlation matrices in PortfolioOptimisers.jl.

`Denoise` encapsulates all parameters required for matrix denoising, including the kernel and its arguments for spectral density estimation, the denoising algorithm, and matrix dimensions. It is the standard estimator type for denoising routines and supports a variety of algorithms ([`SpectralDenoise`](@ref), [`FixedDenoise`](@ref), [`ShrunkDenoise`](@ref)).

# Fields

  - `alg`: Denoising algorithm ([`SpectralDenoise`](@ref), [`FixedDenoise`](@ref), [`ShrunkDenoise`](@ref)).
  - `args`: Positional arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).
  - `kwargs`: Keyword arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).
  - `kernel`: Kernel function for [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
  - `m`: Number of adjacent histograms to smooth over in [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
  - `n`: Number of points in the range of eigenvalues used in the average shifted histogram density estimation.

# Constructor

    Denoise(; alg::AbstractDenoiseAlgorithm = ShrunkDenoise(), m::Integer = 10,
            n::Integer = 1000, kernel::Any = AverageShiftedHistograms.Kernels.gaussian,
            args::Tuple = (), kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> Denoise(;)
Denoise
     alg ┼ ShrunkDenoise
         │   alpha ┴ Float64: 0.0
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
  kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
       m ┼ Int64: 10
       n ┴ Int64: 1000

julia> Denoise(; alg = SpectralDenoise(), m = 20, n = 500)
Denoise
     alg ┼ SpectralDenoise()
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
  kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
       m ┼ Int64: 20
       n ┴ Int64: 500
```

# Related

  - [`AbstractDenoiseEstimator`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
  - [`denoise!`](@ref)
  - [`denoise`](@ref)
"""
struct Denoise{T1, T2, T3, T4, T5, T6} <: AbstractDenoiseEstimator
    alg::T1
    args::T2
    kwargs::T3
    kernel::T4
    m::T5
    n::T6
    function Denoise(alg::AbstractDenoiseAlgorithm, args::Tuple, kwargs::NamedTuple, kernel,
                     m::Integer, n::Integer)
        @argcheck(m > 1, DomainError(m, comp_msg("`m`", 1, :gt, m) * "."))
        @argcheck(n > 1, DomainError(n, comp_msg("`n`", 1, :gt, n) * "."))
        return new{typeof(alg), typeof(args), typeof(kwargs), typeof(kernel), typeof(m),
                   typeof(n)}(alg, args, kwargs, kernel, m, n)
    end
end
function Denoise(; alg::AbstractDenoiseAlgorithm = ShrunkDenoise(), args::Tuple = (),
                 kwargs::NamedTuple = (;),
                 kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                 n::Integer = 1000)
    return Denoise(alg, args, kwargs, kernel, m, n)
end
"""
    _denoise!(alg::AbstractDenoiseAlgorithm, X::AbstractMatrix, vals::AbstractVector,
              vecs::AbstractMatrix, num_factors::Integer)

In-place denoising of a covariance or correlation matrix using a specific denoising algorithm.

These methods are called internally by [`denoise!`](@ref) and [`denoise`](@ref) when a [`Denoise`](@ref) estimator is used, and should not typically be called directly.

# Arguments

  - `alg::AbstractDenoiseAlgorithm`: The denoising algorithm to apply.

      + `alg::SpectralDenoise`: Sets the smallest `num_factors` eigenvalues to zero.
      + `alg::FixedDenoise`: Replaces the smallest `num_factors` eigenvalues with their average.
      + `alg::ShrunkDenoise`: Shrinks the smallest `num_factors` eigenvalues towards the diagonal, controlled by `alg.alpha`.

  - `X`: The matrix to be denoised (modified in-place).
  - `vals`: Eigenvalues of `X`, sorted in ascending order.
  - `vecs`: Corresponding eigenvectors of `X`.
  - `num_factors`: Number of eigenvalues to treat as noise.

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Related

  - [`denoise!`](@ref)
  - [`Denoise`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
"""
function _denoise!(::SpectralDenoise, X::AbstractMatrix, vals::AbstractVector,
                   vecs::AbstractMatrix, num_factors::Integer)
    vals[1:num_factors] .= zero(eltype(X))
    X .= cov2cor(vecs * Diagonal(vals) * transpose(vecs))
    return nothing
end
function _denoise!(::FixedDenoise, X::AbstractMatrix, vals::AbstractVector,
                   vecs::AbstractMatrix, num_factors::Integer)
    vals[1:num_factors] .= sum(vals[1:num_factors]) / num_factors
    X .= cov2cor(vecs * Diagonal(vals) * transpose(vecs))
    return nothing
end
function _denoise!(de::ShrunkDenoise, X::AbstractMatrix, vals::AbstractVector,
                   vecs::AbstractMatrix, num_factors::Integer)
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
    errPDF(x::Real, vals::AbstractVector, q::Real;
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

  - `sse::Real`: The sum of squared errors between the empirical and theoretical densities.

# Related

  - [`find_max_eval`](@ref)
  - [`Denoise`](@ref)
"""
function errPDF(x::Real, vals::AbstractVector, q::Real;
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
    find_max_eval(vals::AbstractVector, q::Real;
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

  - `(e_max::Real, x::Real)`: Tuple containing the estimated upper edge of the noise eigenvalue spectrum (`e_max`) and the fitted scale parameter (`x`).

# Related

  - [`errPDF`](@ref)
  - [`Denoise`](@ref)
"""
function find_max_eval(vals::AbstractVector, q::Real;
                       kernel::Any = AverageShiftedHistograms.Kernels.gaussian,
                       m::Integer = 10, n::Integer = 1000, args::Tuple = (),
                       kwargs::NamedTuple = (;))
    res = Optim.optimize(x -> errPDF(x, vals, q; kernel = kernel, m = m, n = n), 0.0, 1.0,
                         args...; kwargs...)
    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0
    e_max = x * (1.0 + sqrt(1.0 / q))^2
    return e_max, x
end
"""
    denoise!(de::Denoise, X::AbstractMatrix, q::Real; pdm::Union{Nothing, <:Posdef} = Posdef())
    denoise!(::Nothing, args...)

In-place denoising of a covariance or correlation matrix using a [`Denoise`](@ref) estimator.

For covariance matrices, the function internally converts to a correlation matrix, applies the algorithm, and then rescales back to covariance.

# Arguments

  - `de`: The estimator specifying the denoising algorithm.

      + `de::Denoise`: The specified denoising algorithm is applied to `X` in-place.
      + `de::Nothing`: No-op.

  - `X`: The covariance or correlation matrix to be denoised (modified in-place).
  - `q`: The effective sample ratio (e.g., `n_obs / n_assets`), used for spectral thresholding.
  - `pdm`: Optional Positive definite matrix estimator. If provided, ensures the output is positive definite.

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
"""
function denoise!(::Nothing, args...)
    return nothing
end
function denoise!(de::Denoise, X::AbstractMatrix, q::Real,
                  pdm::Union{Nothing, <:Posdef} = Posdef())
    assert_matrix_issquare(X)
    s = diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = eigen(X)
    max_val = find_max_eval(vals, q; kernel = de.kernel, m = de.m, n = de.n, args = de.args,
                            kwargs = de.kwargs)[1]
    num_factors = findlast(vals .<= max_val)
    _denoise!(de.alg, X, vals, vecs, num_factors)
    posdef!(pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
"""
    denoise(de::Denoise, X::AbstractMatrix, q::Real; pdm::Union{Nothing, <:Posdef} = Posdef())
    denoise(::Nothing, args...)

Out-of-place version of [`denoise!`](@ref).

# Related

  - [`denoise!`](@ref)
  - [`Denoise`](@ref)
  - [`SpectralDenoise`](@ref)
  - [`FixedDenoise`](@ref)
  - [`ShrunkDenoise`](@ref)
  - [`posdef`](@ref)
"""
function denoise(::Nothing, args...)
    return nothing
end
function denoise(de::Denoise, X::AbstractMatrix, q::Real,
                 pdm::Union{Nothing, <:Posdef} = Posdef())
    X = copy(X)
    denoise!(de, X, q, pdm)
    return X
end

export Denoise, SpectralDenoise, FixedDenoise, ShrunkDenoise, denoise, denoise!
