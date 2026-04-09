"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all denoising estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement denoising of covariance-like or correlation-like matrices should be subtypes of `AbstractDenoiseEstimator`.

# Interfaces

In order to implement a new denoising estimator which will work seamlessly with the library, subtype `AbstractDenoiseEstimator` with all necessary parameters as part of the struct, and implement the following methods:

  - `denoise!(dn::AbstractDenoiseEstimator, X::MatNum, q::Number) -> MatNum`: In-place denoising.
  - `denoise(dn::AbstractDenoiseEstimator, X::MatNum, q::Number) -> MatNum`: Optional out-of-place denoising.

## Arguments

  - $(arg_dict[:dn])
  - $(arg_dict[:sigrhoX])
  - `q`: The effective sample ratio `observations / assets`, used for spectral thresholding.

## Returns

  - `X::MatNum`: The denoised input matrix `X`.

# Examples

We can create a dummy denoising estimator as follows:

```jldoctest
julia> struct MyDenoiseEstimator <: PortfolioOptimisers.AbstractDenoiseEstimator end

julia> function PortfolioOptimisers.denoise!(dn::MyDenoiseEstimator, X::PortfolioOptimisers.MatNum, q::Number)
           # Implement your in-place denoising estimator here.
           println("Denoising matrix in-place...")
           return X
       end

julia> function PortfolioOptimisers.denoise(dn::MyDenoiseEstimator, X::PortfolioOptimisers.MatNum, q::Number)
           X = copy(X)
           println("Copy X...")
           denoise!(dn, X, q)
           return X
       end

julia> denoise!(MyDenoiseEstimator(), [1.0 2.0; 2.0 1.0], 2.0)
Denoising matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> denoise(MyDenoiseEstimator(), [1.0 2.0; 2.0 1.0], 2.0)
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
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all denoising algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a specific denoising algorithm should be subtypes of `AbstractDenoiseAlgorithm`.

# Interfaces

If you wish to implement a new denoising algorithm that works with an existing denoising estimator, subtype `AbstractDenoiseAlgorithm`, with all necessary parameters as part of the struct, and implement the following method:

  - `_denoise!(alg::AbstractDenoiseAlgorithm, X::MatNum, vals::VecNum, vecs::MatNum, num_factors::Integer) -> MatNum`: In-place denoising of a covariance or correlation matrix using the specific algorithm.

## Arguments

  - `alg`: Denoising algorithm.
  - $(arg_dict[:sigrhoX])
  - `vals`: Eigenvalues of `X`, sorted in ascending order.
  - `vecs`: Corresponding eigenvectors of `X`.
  - `num_factors`: Number of eigenvalues to treat as noise.

## Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

# Examples

We can create a dummy denoising algorithm as follows:

```jldoctest
julia> struct MyDenoiseAlgorithm <: PortfolioOptimisers.AbstractDenoiseAlgorithm end

julia> function PortfolioOptimisers._denoise!(dn::MyDenoiseAlgorithm,
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
$(DocStringExtensions.TYPEDEF)

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
$(DocStringExtensions.TYPEDEF)

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
$(DocStringExtensions.TYPEDEF)

A denoising algorithm that shrinks the smallest `num_factors` eigenvalues of a covariance or correlation matrix towards their diagonal, controlled by the shrinkage parameter `alpha`. This approach interpolates between no shrinkage (`alpha = 0`) and full shrinkage (`alpha = 1`), providing a flexible way to regularize noisy eigenvalues.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShrunkDenoise(;
        alpha::Number = 0.0,
    ) -> ShrunkDenoise

Keywords correspond to the struct's fields.

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
@concrete struct ShrunkDenoise <: AbstractDenoiseAlgorithm
    "Shrinkage parameter controlling the degree of shrinkage applied to the smallest eigenvalues."
    alpha
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
$(DocStringExtensions.TYPEDEF)

A flexible container type for configuring and applying denoising algorithms to covariance or correlation matrices in `PortfolioOptimisers.jl`.

`Denoise` encapsulates all parameters required for matrix denoising in [`denoise!`](@ref) and [`denoise`](@ref), allowing users to specify the denoising algorithm, optimization parameters, kernel settings for density estimation, and optional positive definite matrix projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Denoise(;
        pdm::Option{<:Posdef} = Posdef(),
        alg::AbstractDenoiseAlgorithm = ShrunkDenoise(),
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        kernel = AverageShiftedHistograms.Kernels.gaussian,
        m::Integer = 10,
        n::Integer = 1000
    ) -> Denoise

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> Denoise()
Denoise
     pdm ┼ Posdef
         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     alg ┼ ShrunkDenoise
         │   alpha ┴ Float64: 0.0
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
  kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
       m ┼ Int64: 10
       n ┴ Int64: 1000

julia> Denoise(; alg = SpectralDenoise(), m = 20, n = 500)
Denoise
     pdm ┼ Posdef
         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │   kwargs ┴ @NamedTuple{}: NamedTuple()
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
  - [`AverageShiftedHistograms.Kernels`](https://joshday.github.io/AverageShiftedHistograms.jl/stable/kernels/)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
@concrete struct Denoise <: AbstractDenoiseEstimator
    "$(field_dict[:opdm])"
    pdm
    "Denoising algorithm."
    alg
    "Positional arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl)."
    args
    "Keyword arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl)."
    kwargs
    "Kernel function for [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl)."
    kernel
    "Number of adjacent histograms to smooth over in [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl)."
    m
    "Number of points in the range of eigenvalues used in the [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl) density estimation."
    n
    function Denoise(pdm::Option{<:Posdef}, alg::AbstractDenoiseAlgorithm, args::Tuple,
                     kwargs::NamedTuple, kernel, m::Integer, n::Integer)
        @argcheck(1 < m, DomainError)
        @argcheck(1 < n, DomainError)
        return new{typeof(pdm), typeof(alg), typeof(args), typeof(kwargs), typeof(kernel),
                   typeof(m), typeof(n)}(pdm, alg, args, kwargs, kernel, m, n)
    end
end
function Denoise(; pdm::Option{<:Posdef} = Posdef(),
                 alg::AbstractDenoiseAlgorithm = ShrunkDenoise(), args::Tuple = (),
                 kwargs::NamedTuple = (;),
                 kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                 n::Integer = 1000)
    return Denoise(pdm, alg, args, kwargs, kernel, m, n)
end
"""
    _denoise!(
        alg::AbstractDenoiseAlgorithm,
        X::MatNum,
        vals::VecNum,
        vecs::MatNum,
        num_factors::Integer
    ) -> MatNum

In-place denoising of a correlation matrix using a specific denoising algorithm.

These methods are called internally by [`denoise!`](@ref) and [`denoise`](@ref) when a [`Denoise`](@ref) estimator is used, and should not typically be called directly.

# Arguments

  - `alg`: Denoising algorithm.
  - $(arg_dict[:sigrhoX])
  - `vals`: Eigenvalues of `X`, sorted in ascending order.
  - `vecs`: Corresponding eigenvectors of `X`.
  - `num_factors`: Number of eigenvalues to treat as noise.

# Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

# Details

  - Applies the algorithm `alg` to `vals` using `num_factors`.
  - Reconstructs the denoised correlation matrix `X` in-place from the modified eigenvalues `vals` and eigenvectors `vecs`.
  - Returns the denoised correlation matrix `X`.

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
    find_max_eval(
        vals::VecNum,
        q::Number,
        kernel::Any = AverageShiftedHistograms.Kernels.gaussian,
        m::Integer = 10,
        n::Integer = 1000,
        args::Tuple = (),
        kwargs::NamedTuple = (;)
    ) -> Number

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

# Details

  - Minimises the sum of squared errors (SSE) between the theoretical Marčenko–Pastur (MP) eigenvalue density and the empirical eigenvalue density estimated from observed eigenvalues.
  - Uses the minimiser and effective sample ratio to compute the maximum feasable noise eigenvalue.
  - Returns the maximum feasable noise eigenvalue.

# Related

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
    pdf = Matrix{eltype(vals)}(undef, n, 2)
    op_sqrt_iq_sq = (one(q) + sqrt(inv(q)))^2
    om_sqrt_iq_sq = (one(q) - sqrt(inv(q)))^2
    # Marčenko-Pastur distribution
    function f(x::Number)
        e_min, e_max = x * om_sqrt_iq_sq, x * op_sqrt_iq_sq
        rg = range(e_min, e_max; length = n)
        pdf[:, 1] .= q ⊘ (2 * pi * x * rg) ⊙
                     sqrt.(clamp.((e_max .- rg) ⊙ (rg .- e_min), zero(x), typemax(x)))
        res = AverageShiftedHistograms.ash(vals; rng = range(e_min, e_max; length = n),
                                           kernel = kernel, m = m)
        for (i, j) in enumerate(view(pdf, :, 1))
            pdf[i, 2] = AverageShiftedHistograms.pdf(res, j)
        end
        pdf[.!isfinite.(view(pdf, :, 2)), 2] .= zero(eltype(x))
        return sum((view(pdf, :, 2) - view(pdf, :, 1)) .^ 2)
    end
    res = Optim.optimize(x -> f(x), zero(eltype(vals)), one(eltype(vals)), args...;
                         kwargs...)
    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0
    return x * op_sqrt_iq_sq
end
"""
    denoise!(dn::Option{<:Denoise}, X::MatNum, q::Number) -> MatNum

In-place denoising of a covariance or correlation matrix using a [`Denoise`](@ref) estimator.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Arguments

  - $(arg_dict[:odn])
      + `::Denoise`: The specified denoising algorithm is applied to `X` in-place.
      + `::Nothing`: No-op.
  - $(arg_dict[:sigrhoX])
  - `q`: The effective sample ratio `observations / assets`, used for spectral thresholding.

# Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

# Details

  - If `dn` is `::Nothing`, the function returns `X` without modification.
  - If `X` is not a correlation matrix, it is converted to one before applying the algorithm.
  - Performs an eigenvector decomposition of `X`.
  - Uses the Marčenko-Pastur distribution to compute the maximum feasable noise eigenvalue.
  - Applies the denoising algorithm to `X` in `dn.alg` via [`_denoise!`](@ref) to the eigenvalues which are below this value.
  - Applies the positive definite projection to `X` in `dn.pdm` via [`denoise!`](@ref).
  - If `X` was not originally a correlation matrix, it is converted back.
  - Returns `X`.

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
function denoise!(dn::Denoise, X::MatNum, q::Number)
    assert_matrix_issquare(X, :X)
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = LinearAlgebra.eigen(X)
    max_val = find_max_eval(vals, q, dn.kernel, dn.m, dn.n, dn.args, dn.kwargs)
    num_factors = searchsortedlast(vals, max_val)
    _denoise!(dn.alg, X, vals, vecs, num_factors)
    posdef!(dn.pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return X
end
"""
    denoise(dn::Option{<:Denoise}, X::MatNum, q::Number) -> MatNum

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
function denoise(dn::Denoise, X::MatNum, q::Number)
    X = copy(X)
    denoise!(dn, X, q)
    return X
end

export Denoise, SpectralDenoise, FixedDenoise, ShrunkDenoise, denoise, denoise!
