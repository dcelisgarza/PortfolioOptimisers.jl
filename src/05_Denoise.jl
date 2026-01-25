"""
    abstract type AbstractDenoiseEstimator <: AbstractEstimator end

Abstract supertype for all denoising estimator types in `PortfolioOptimisers.jl`.

All concrete types that implement denoising of covariance-like or correlation-like matrices should subtype `AbstractDenoiseEstimator`.

# Interfaces

In order to implement a new denoising estimator which will work seamlessly with the library, subtype `AbstractDenoiseEstimator` including all necessary parameters as part of the struct, and implement the following methods:

  - `denoise!(de::AbstractDenoiseEstimator, X::MatNum, q::Number)`: In-place denoising.
  - `denoise(de::AbstractDenoiseEstimator, X::MatNum, q::Number)`: Optional out-of-place denoising.

For example, we can create a dummy denoising estimator as follows:

```jldoctest
julia> struct MyDenoiseEstimator <: PortfolioOptimisers.AbstractDenoiseEstimator end

julia> function PortfolioOptimisers.denoise!(de::MyDenoiseEstimator, X::PortfolioOptimisers.MatNum)
           # Implement your in-place denoising estimator here.
           println("Denoising matrix in-place...")
           return X
       end

julia> function PortfolioOptimisers.denoise(de::MyDenoiseEstimator, X::PortfolioOptimisers.MatNum)
           X = copy(X)
           println("Copy X...")
           denoise!(de, X)
           return X
       end

julia> denoise!(MyDenoiseEstimator(), [1.0 2.0; 2.0 1.0])
Denoising matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> denoise(MyDenoiseEstimator(), [1.0 2.0; 2.0 1.0])
Copy X...
Denoising matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

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

# Interfaces

If you wish to implement a new denoising algorithm that works with an existing denoising estimator, subtype `AbstractDenoiseAlgorithm`, including all necessary parameters as part of the struct, and implement the following method:

  - `_denoise!(alg::AbstractDenoiseAlgorithm, X::MatNum, vals::VecNum, vecs::MatNum, num_factors::Integer)`: In-place denoising of a covariance or correlation matrix using the specific algorithm.

For example, we can create a dummy denoising algorithm as follows:

```jldoctest
julia> struct MyDenoiseAlgorithm <: PortfolioOptimisers.AbstractDenoiseAlgorithm end

julia> function PortfolioOptimisers._denoise!(de::MyDenoiseAlgorithm,
                                              X::PortfolioOptimisers.MatNum,
                                              vals::PortfolioOptimisers.VecNum,
                                              vecs::PortfolioOptimisers.MatNum,
                                              num_factors::Integer)
           # Implement your in-place denoising logic here.
           println("Denoising matrix using custom algorithm...")
           return X
       end

julia> denoise!(Denoise(; alg = MyDenoiseAlgorithm()), [2.0 1.0; 1.0 2.0], 1 / 100)
Denoising matrix using custom algorithm...
2×2 Matrix{Float64}:
 2.0  1.0
 1.0  2.0

julia> denoise(Denoise(; alg = MyDenoiseAlgorithm()), [2.0 1.0; 1.0 2.0], 1 / 100)
Denoising matrix using custom algorithm...
2×2 Matrix{Float64}:
 2.0  1.0
 1.0  2.0
```

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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
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
  - $(glossary[:opdm])

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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
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
    _denoise!(alg::AbstractDenoiseAlgorithm, X::MatNum, vals::VecNum, vecs::MatNum,
              num_factors::Integer)

In-place denoising of a covariance or correlation matrix using a specific denoising algorithm.

These methods are called internally by [`denoise!`](@ref) and [`denoise`](@ref) when a [`Denoise`](@ref) estimator is used, and should not typically be called directly.

# Arguments

  - `alg`: Denoising algorithm.
  - $(glossary[:sigrhoX])
  - `vals`: Eigenvalues of `X`, sorted in ascending order.
  - `vecs`: Corresponding eigenvectors of `X`.
  - `num_factors`: Number of eigenvalues to treat as noise.

# Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function _denoise!(::SpectralDenoise, X::MatNum, vals::VecNum, vecs::MatNum,
                   num_factors::Integer)
    vals[1:num_factors] .= zero(eltype(X))
    X .= StatsBase.cov2cor(vecs * LinearAlgebra.Diagonal(vals) * transpose(vecs))
    return X
end
function _denoise!(::FixedDenoise, X::MatNum, vals::VecNum, vecs::MatNum,
                   num_factors::Integer)
    vals[1:num_factors] .= sum(vals[1:num_factors]) / num_factors
    X .= StatsBase.cov2cor(vecs * LinearAlgebra.Diagonal(vals) * transpose(vecs))
    return X
end
function _denoise!(alg::ShrunkDenoise, X::MatNum, vals::VecNum, vecs::MatNum,
                   num_factors::Integer)
    # Small
    vals_l = vals[1:num_factors]
    vecs_l = vecs[:, 1:num_factors]

    # Large
    vals_r = vals[(num_factors + 1):end]
    vecs_r = vecs[:, (num_factors + 1):end]

    corr0 = vecs_r * LinearAlgebra.Diagonal(vals_r) * transpose(vecs_r)
    corr1 = vecs_l * LinearAlgebra.Diagonal(vals_l) * transpose(vecs_l)

    X .= corr0 +
         alg.alpha * corr1 +
         (one(alg.alpha) - alg.alpha) * LinearAlgebra.Diagonal(corr1)
    return X
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

  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function errPDF(x::Number, vals::VecNum, q::Number,
                kernel::Any = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                n::Integer = 1000)
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max; length = n)
    pdf1 = q ⊘ (2 * pi * x * rg) ⊙
           sqrt.(clamp.((e_max .- rg) ⊙ (rg .- e_min), zero(x), typemax(x)))
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    res = AverageShiftedHistograms.ash(vals; rng = range(e_min, e_max; length = n),
                                       kernel = kernel, m = m)
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

  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
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
    denoise!(::Nothing, X::MatNum, args...)

In-place denoising of a covariance or correlation matrix using a [`Denoise`](@ref) estimator.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Arguments

  - $(glossary[:odn])

      + `::Denoise`: The specified denoising algorithm is applied to `X` in-place.
      + `::Nothing`: No-op.

  - $(glossary[:sigrhoX])
  - `q`: The effective sample ratio `observations / assets`, used for spectral thresholding.

# Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function denoise!(::Nothing, X::MatNum, args...)
    return X
end
function denoise!(de::Denoise, X::MatNum, q::Number)
    assert_matrix_issquare(X, :X)
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.StatsBase.cov2cor!(X, s)
    end
    vals, vecs = LinearAlgebra.eigen(X)
    max_val = find_max_eval(vals, q, de.kernel, de.m, de.n, de.args, de.kwargs)[1]
    num_factors = searchsortedlast(vals, max_val)
    _denoise!(de.alg, X, vals, vecs, num_factors)
    posdef!(de.pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return X
end
"""
    denoise(de::Denoise, X::MatNum, q::Number)
    denoise(::Nothing, X::MatNum, args...)

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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function denoise(::Nothing, X::MatNum, args...)
    return X
end
function denoise(de::Denoise, X::MatNum, q::Number)
    X = copy(X)
    denoise!(de, X, q)
    return X
end

export Denoise, SpectralDenoise, FixedDenoise, ShrunkDenoise, denoise, denoise!
