"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all denoising estimator types.

All concrete and/or abstract types that implement denoising of covariance-like or correlation-like matrices should be subtypes of `AbstractDenoiseEstimator`.

# Interfaces

In order to implement a new denoising estimator which will work seamlessly with the library, subtype `AbstractDenoiseEstimator` with all necessary parameters as part of the struct, and implement the following methods:

  - `denoise!(dn::AbstractDenoiseEstimator, X::MatNum, q::Number) -> MatNum`: In-place denoising.
  - `denoise(dn::AbstractDenoiseEstimator, X::MatNum, q::Number) -> MatNum`: Optional out-of-place denoising. A fallback method copies `X` and calls `denoise!`, so it is only needed if the copy can be avoided.

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

julia> function PortfolioOptimisers.denoise!(dn::MyDenoiseEstimator, X::PortfolioOptimisers.MatNum,
                                             q::Number)
           # Implement your in-place denoising estimator here.
           println(\"Denoising matrix in-place...\")
           return X
       end

julia> function PortfolioOptimisers.denoise(dn::MyDenoiseEstimator, X::PortfolioOptimisers.MatNum,
                                            q::Number)
           X = copy(X)
           println(\"Copy X...\")
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

Abstract supertype for all denoising algorithm types.

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
           println(\"Denoising matrix using custom algorithm...\")
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

Denoises by setting the noise eigenvalues to zero. This removes the principal components that random matrix theory attributes to noise, then rescales the reconstruction back to unit diagonal.

# Mathematical definition

The noise eigenvalues are set to zero, the matrix is rebuilt from the resulting spectrum, and the result is rescaled to unit diagonal:

```math
\\begin{align}
\\tilde{\\lambda}_i &= \\begin{cases} 0 & \\lambda_i \\leq \\lambda_+ \\\\ \\lambda_i & \\lambda_i > \\lambda_+ \\end{cases}\\,, \\\\
\\mathbf{C}_{\\mathrm{signal}} &= \\mathbf{V} \\, \\mathrm{Diag}(\\tilde{\\boldsymbol{\\lambda}}) \\, \\mathbf{V}^\\intercal\\,, \\\\
\\tilde{X}_{ij} &= \\frac{(C_{\\mathrm{signal}})_{ij}}{\\sqrt{(C_{\\mathrm{signal}})_{ii} \\, (C_{\\mathrm{signal}})_{jj}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_tilde_i])
  - $(math_dict[:lambda_i_eig])
  - $(math_dict[:lambda_plus_mp])
  - $(math_dict[:V_eigvec])
  - $(math_dict[:C_signal])
  - $(math_dict[:X_denoised])

Zeroing the noise eigenvalues takes the diagonal of ``\\mathbf{C}_{\\mathrm{signal}}`` below one, so the rescaling is not cosmetic: it changes every entry.

# Algorithm

The branch of [`_denoise!`](@ref) that this tag selects runs these steps.

 1. Set the `num_factors` smallest entries of `vals` to zero. `vals` is sorted ascending, so those are the noise eigenvalues.
 2. Rebuild the matrix as `vecs * Diagonal(vals) * transpose(vecs)`, which is the signal-only reconstruction ``\\mathbf{C}_{\\mathrm{signal}}``.
 3. Rescale the reconstruction to unit diagonal with `StatsBase.cov2cor`, and write the result into `X`. The rescaling also sheds the round-off of the eigendecomposition, so this branch never pins the diagonal by hand.

# Constructors

    SpectralDenoise() -> SpectralDenoise

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

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
  - $(ref_dict[:cajas2025]) Section 3.5.2.2, Equations 3.54 and 3.55.
"""
struct SpectralDenoise <: AbstractDenoiseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Denoises by replacing the noise eigenvalues with their own mean. This flattens the principal components that random matrix theory attributes to noise, rather than discarding them, then rescales the reconstruction back to unit diagonal.

# Mathematical definition

The noise eigenvalues are replaced by their own mean, the matrix is rebuilt from the flattened spectrum, and the result is rescaled to unit diagonal:

```math
\\begin{align}
\\tilde{\\lambda}_i &= \\begin{cases} \\bar{\\lambda}_\\text{noise} & \\lambda_i \\leq \\lambda_+ \\\\ \\lambda_i & \\lambda_i > \\lambda_+ \\end{cases}\\,, \\\\
\\mathbf{C} &= \\mathbf{V} \\, \\mathrm{Diag}(\\tilde{\\boldsymbol{\\lambda}}) \\, \\mathbf{V}^\\intercal\\,, \\\\
\\tilde{X}_{ij} &= \\frac{C_{ij}}{\\sqrt{C_{ii} C_{jj}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_tilde_i])
  - $(math_dict[:lambda_i_eig])
  - $(math_dict[:lambda_plus_mp])
  - ``\\bar{\\lambda}_\\text{noise}``: Mean of the noise eigenvalues.
  - $(math_dict[:V_eigvec])
  - ``\\mathbf{C}``: Reconstruction from the flattened spectrum.
  - $(math_dict[:X_denoised])

Flattening the noise eigenvalues preserves the trace but not the diagonal, so the rescaling is not cosmetic: it changes every entry.

# Algorithm

The branch of [`_denoise!`](@ref) that this tag selects runs these steps.

 1. Replace the `num_factors` smallest entries of `vals` by their own mean. `vals` is sorted ascending, so those are the noise eigenvalues.
 2. Rebuild the matrix as `vecs * Diagonal(vals) * transpose(vecs)`, which is the reconstruction ``\\mathbf{C}`` from the flattened spectrum.
 3. Rescale the reconstruction to unit diagonal with `StatsBase.cov2cor`, and write the result into `X`. The rescaling also sheds the round-off of the eigendecomposition, so this branch never pins the diagonal by hand.

# Constructors

    FixedDenoise() -> FixedDenoise

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

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
  - $(ref_dict[:cajas2025]) Section 3.5.2.1, Equations 3.52 and 3.53.
"""
struct FixedDenoise <: AbstractDenoiseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Denoises by shrinking the off-diagonal part of the noise block towards zero, keeping its diagonal whole. The polarity of `alpha` is the reverse of the reading its name suggests: `alpha` is the weight *kept* on that off-diagonal part, so `alpha = 0` is total shrinkage and `alpha = 1` returns the input unchanged. The default `alpha = 0.0` is therefore total shrinkage.

# Mathematical definition

The spectrum is split at the Marčenko-Pastur upper bound. The signal block is rebuilt whole, and only the off-diagonal part of the noise block is scaled by ``\\alpha``:

```math
\\begin{align}
\\mathbf{C}_{\\mathrm{signal}} &= \\mathbf{V}_{\\mathrm{signal}} \\, \\mathrm{Diag}(\\boldsymbol{\\lambda}_{\\mathrm{signal}}) \\, \\mathbf{V}_{\\mathrm{signal}}^\\intercal\\,, \\\\
\\mathbf{C}_{\\mathrm{noise}} &= \\mathbf{V}_{\\mathrm{noise}} \\, \\mathrm{Diag}(\\boldsymbol{\\lambda}_{\\mathrm{noise}}) \\, \\mathbf{V}_{\\mathrm{noise}}^\\intercal\\,, \\\\
\\tilde{\\mathbf{X}} &= \\mathbf{C}_{\\mathrm{signal}} + \\alpha \\mathbf{C}_{\\mathrm{noise}} + (1 - \\alpha) \\, \\mathrm{Diag}(\\mathbf{C}_{\\mathrm{noise}})\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_i_eig])
  - $(math_dict[:lambda_plus_mp])
  - $(math_dict[:V_signal])
  - ``\\mathbf{V}_{\\mathrm{noise}}``: Eigenvector block of the noise eigenpairs.
  - $(math_dict[:lambda_vec_signal])
  - ``\\boldsymbol{\\lambda}_{\\mathrm{noise}}``: Noise eigenvalues.
  - $(math_dict[:C_signal])
  - ``\\mathbf{C}_{\\mathrm{noise}}``: Reconstruction from the noise eigenpairs alone.
  - ``\\alpha \\in [0, 1]``: Weight kept on the off-diagonal part of the noise block. ``\\alpha = 0`` keeps only its diagonal, which is total shrinkage. ``\\alpha = 1`` keeps the block whole, so ``\\tilde{\\mathbf{X}} = \\mathbf{X}``.
  - $(math_dict[:X_denoised])

The two ``\\alpha`` weights sum to one on the diagonal, so the reconstruction preserves it in exact arithmetic.

# Algorithm

The branch of [`_denoise!`](@ref) that this tag selects runs these steps.

 1. Split `vals` and `vecs` at `num_factors`. The first `num_factors` entries are the noise block `vals_l` and `vecs_l`, and the rest are the signal block `vals_r` and `vecs_r`.
 2. Build `corr0` from the signal block, which is ``\\mathbf{C}_{\\mathrm{signal}}``.
 3. Build `corr1` from the noise block, which is ``\\mathbf{C}_{\\mathrm{noise}}``.
 4. Write `corr0 + alpha * corr1 + (1 - alpha) * Diagonal(corr1)` into `X`.
 5. Set the diagonal of `X` to one. The reconstruction preserves the diagonal in exact arithmetic, so this step sheds the round-off of the eigendecomposition. This branch reconstructs directly rather than through `StatsBase.cov2cor`, so it is the only branch that must pin its own diagonal.

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

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
  - $(ref_dict[:cajas2025]) Section 3.5.2.3, Equation 3.56.
"""
@concrete struct ShrunkDenoise <: AbstractDenoiseAlgorithm
    """
    Weight kept on the off-diagonal part of the noise block, ``\\alpha \\in [0, 1]``. It is the weight *kept*, not the weight removed: `0` keeps only the diagonal of that block, which is total shrinkage, and `1` keeps the block whole, which returns the input unchanged.
    """
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

Configures and applies denoising algorithms to covariance or correlation matrices.

`Denoise` encapsulates all parameters required for matrix denoising in [`denoise!`](@ref) and [`denoise`](@ref), allowing users to specify the denoising algorithm, optimization parameters, kernel settings for density estimation, and optional positive definite matrix projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Denoise(;
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        alg::AbstractDenoiseAlgorithm = ShrunkDenoise(),
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        kernel = AverageShiftedHistograms.Kernels.gaussian,
        m::Integer = 10,
        n::Integer = 1000
    ) -> Denoise

Keywords correspond to the struct's fields.

## Validation

  - `m > 1`.
  - `n > 1`.

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

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
  - $(ref_dict[:cajas2025]) Section 3.5.2.
"""
@concrete struct Denoise <: AbstractDenoiseEstimator
    """
    $(field_dict[:opdm])
    """
    pdm
    """
    Denoising algorithm.
    """
    alg
    """
    Positional arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).
    """
    args
    """
    Keyword arguments for the univariate [Optim.optimize](https://github.com/JuliaNLSolvers/Optim.jl).
    """
    kwargs
    """
    Kernel function for [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
    """
    kernel
    """
    Number of adjacent histograms to smooth over in [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl).
    """
    m
    """
    Number of points in the range of eigenvalues used in the [AverageShiftedHistograms.ash](https://github.com/joshday/AverageShiftedHistograms.jl) density estimation.
    """
    n
    function Denoise(pdm::Option{<:AbstractPosdefEstimator}, alg::AbstractDenoiseAlgorithm,
                     args::Tuple, kwargs::NamedTuple, kernel, m::Integer,
                     n::Integer)::Denoise
        @argcheck(1 < m, DomainError)
        @argcheck(1 < n, DomainError)
        return new{typeof(pdm), typeof(alg), typeof(args), typeof(kwargs), typeof(kernel),
                   typeof(m), typeof(n)}(pdm, alg, args, kwargs, kernel, m, n)
    end
end
function Denoise(; pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                 alg::AbstractDenoiseAlgorithm = ShrunkDenoise(), args::Tuple = (),
                 kwargs::NamedTuple = (;),
                 kernel = AverageShiftedHistograms.Kernels.gaussian, m::Integer = 10,
                 n::Integer = 1000)::Denoise
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

# Algorithm

The method that Julia selects is the algorithm. `vals` is sorted ascending, so the first `num_factors` entries are the noise eigenvalues and the rest are the signal eigenvalues.

 1. `alg` is a [`SpectralDenoise`](@ref): zero the noise eigenvalues, rebuild from the signal components alone, and rescale to unit diagonal with `StatsBase.cov2cor`.
 2. `alg` is a [`FixedDenoise`](@ref): replace the noise eigenvalues by their own mean, rebuild from the flattened spectrum, and rescale to unit diagonal with `StatsBase.cov2cor`.
 3. `alg` is a [`ShrunkDenoise`](@ref): rebuild the two blocks separately, combine them under `alg.alpha`, and pin the diagonal to one. This branch does not route through `StatsBase.cov2cor`, so it is the only branch that pins its own diagonal.

Every branch writes into `X` and returns it.

# Arguments

  - `alg`: Denoising algorithm.
  - $(arg_dict[:sigrhoX])
  - `vals`: Eigenvalues of `X`, sorted in ascending order.
  - `vecs`: Corresponding eigenvectors of `X`.
  - `num_factors`: Number of eigenvalues to treat as noise.

# Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

# Related

  - [`denoise!`](@ref)
  - [`Denoise`](@ref)
  - [`SpectralDenoise`](@ref): the closed form of the branch of step 1.
  - [`FixedDenoise`](@ref): the closed form of the branch of step 2.
  - [`ShrunkDenoise`](@ref): the closed form of the branch of step 3.
  - [`MatNum`](@ref)
  - [`VecNum`](@ref)

# References

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
  - $(ref_dict[:cajas2025]) Section 3.5.2.
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
    #=
    In exact arithmetic the reconstruction already preserves the diagonal: it is
    `corr0[i, i] + corr1[i, i]`, since the two `alpha` weights sum to one there, and that
    is `X[i, i]` back again. What is left is eigendecomposition round-off, measured at
    `1 ± 1.5e-15`. `SpectralDenoise` and `FixedDenoise` shed it by routing their
    reconstruction through `cov2cor`; this branch reconstructs directly, so it has to pin
    the diagonal itself. `denoise!` has already converted a covariance to a correlation
    before calling here, so one is the definitionally correct value.

    It is worth pinning because the correlation distance kernels take `sqrt(1 - rho[i, i])`,
    which *amplifies*: 1.1e-16 on the correlation diagonal becomes 7.45e-9 on the distance
    diagonal, which is large enough to be a real self-loop weight in a weighted graph and
    to fail `PhylogenyResult`'s zero-diagonal check.
    =#
    X[LinearAlgebra.diagind(X)] .= one(eltype(X))
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

Three traps follow from the shape of that fit. The fitted variance is bounded above by one, so a spectrum whose noise variance exceeds one fits at the boundary and the returned edge is the unit-variance edge; a correlation matrix has unit variance by construction, which is the case this bound is written for. A spectrum whose eigenvalues are all equal carries no fit, because its range is a single point; such a matrix is a multiple of the identity, so it holds no signal to separate. A search that does not converge substitutes a unit variance and warns, so a caller can tell a fitted edge from a fallback edge; only `args` and `kwargs` can make the search fail, and the defaults converge.

# Mathematical definition

For an effective sample ratio ``q = T/N`` and a noise variance ``\\sigma^2``, the Marčenko-Pastur density and its support are

```math
\\begin{align}
f(\\lambda) &= \\begin{cases} \\dfrac{q \\sqrt{(\\lambda_+ - \\lambda)(\\lambda - \\lambda_-)}}{2 \\pi \\lambda \\sigma^2} & \\lambda \\in [\\lambda_-, \\lambda_+] \\\\ 0 & \\text{otherwise} \\end{cases}\\,, \\\\
\\lambda_{\\pm} &= \\sigma^2 \\left(1 \\pm \\sqrt{\\frac{1}{q}}\\right)^2\\,.
\\end{align}
```

The noise variance is the minimiser of the sum of squared errors between that density and an average shifted histogram estimate of the density of the observed eigenvalues,

```math
\\begin{align}
\\hat{\\sigma}^2 &= \\underset{\\sigma^2 \\in [0, 1]}{\\arg\\min} \\sum_{i=1}^{n} \\left(\\hat{f}(\\lambda_i) - f(\\lambda_i)\\right)^2\\,, \\\\
\\hat{\\lambda}_{+} &= \\hat{\\sigma}^2 \\left(1 + \\sqrt{\\frac{1}{q}}\\right)^2\\,.
\\end{align}
```

Where:

  - ``f``: Theoretical Marčenko-Pastur density.
  - ``\\hat{f}``: Average shifted histogram estimate of the density of the observed eigenvalues.
  - ``\\lambda_{\\pm}``: Upper and lower edges of the support of ``f``.
  - ``\\hat{\\lambda}_{+}``: Fitted upper edge, which is the value returned.
  - $(math_dict[:sigma2_noise])
  - $(math_dict[:q_mp])
  - ``n``: Number of grid points, which is the argument `n`.
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. Compute the two edge factors of a unit variance, `op_sqrt_iq_sq` for ``\\lambda_+`` and `om_sqrt_iq_sq` for ``\\lambda_-``.
 2. Estimate the density of `vals` with `AverageShiftedHistograms.ash`, under `kernel` and `m`, over the range of `vals` itself. When that range is a single point, span `[v, v + 1]` instead, because the estimator needs a range it can bin. The estimate is built once, so its support is fixed and it does not depend on the trial variance. An estimate whose support followed the trial variance would renormalise over a shrinking window, which gives the objective a spurious local minimum well below the true variance.
 3. Define the objective on a trial variance `x`. Steps 4 to 7 are its body.
 4. Scale the two edge factors by `x`, giving `e_min` and `e_max`, and place `n` equally spaced points over `[e_min, e_max]`, giving `rg`.
 5. Evaluate the theoretical density on `rg`, giving column 1 of `pdf`. The product under the square root is clamped at zero, so a round-off outside the support gives zero rather than a domain error.
 6. Read the estimate of step 2 on `rg`, the same abscissa as column 1, giving column 2 of `pdf`. Replace a non-finite entry by zero.
 7. Return the sum of the squared differences of the two columns.
 8. Minimise the objective over `x` in `[0, 1]` with `Optim.optimize`, under `args` and `kwargs`.
 9. Take `x` as the minimiser when the search converged. When it did not, warn and substitute `x = 1`, the variance of a correlation matrix, so the returned edge is the unit-variance edge exactly.
10. Return `x * op_sqrt_iq_sq`, the fitted upper edge.

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

# Related

  - [`Denoise`](@ref)
  - [`VecNum`](@ref)
  - [`AverageShiftedHistograms.Kernels`](https://joshday.github.io/AverageShiftedHistograms.jl/stable/kernels/)

# References

  - $(ref_dict[:mpdist])
  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:cajas2025]) Section 3.5.1, Equation 3.51.
"""
function find_max_eval(vals::VecNum, q::Number,
                       kernel::Any = AverageShiftedHistograms.Kernels.gaussian,
                       m::Integer = 10, n::Integer = 1000, args::Tuple = (),
                       kwargs::NamedTuple = (;))
    pdf = Matrix{eltype(vals)}(undef, n, 2)
    op_sqrt_iq_sq = (one(q) + sqrt(inv(q)))^2
    om_sqrt_iq_sq = (one(q) - sqrt(inv(q)))^2
    #=
    The empirical density is estimated once, over the observed spectrum, and it does not
    depend on the trial variance. Only the abscissa at which it is read moves with `x`.

    An estimate whose support moves with `x` is not comparable across `x`: it renormalises
    over a shrinking window, so the objective gains a spurious local minimum well below the
    true variance and the search converges to it. See #475.
    =#
    v_min, v_max = extrema(vals)
    # A spectrum whose eigenvalues are all equal has a range of zero, and `ash` needs a
    # range it can bin. Such a matrix is a multiple of the identity and carries no signal.
    v_hi = v_max > v_min ? v_max : v_min + one(eltype(vals))
    ash_res = AverageShiftedHistograms.ash(vals; rng = range(v_min, v_hi; length = n),
                                           kernel = kernel, m = m)
    # Marčenko-Pastur distribution
    function f(x::Number)
        e_min, e_max = x * om_sqrt_iq_sq, x * op_sqrt_iq_sq
        rg = range(e_min, e_max; length = n)
        pdf[:, 1] .= q ⊘ (2 * pi * x * rg) ⊙
                     sqrt.(clamp.((e_max .- rg) ⊙ (rg .- e_min), zero(x), typemax(x)))
        for (i, j) in enumerate(rg)
            pdf[i, 2] = AverageShiftedHistograms.pdf(ash_res, j)
        end
        pdf[.!isfinite.(view(pdf, :, 2)), 2] .= zero(eltype(x))
        return sum((view(pdf, :, 2) - view(pdf, :, 1)) .^ 2)
    end
    res = Optim.optimize(x -> f(x), zero(eltype(vals)), one(eltype(vals)), args...;
                         kwargs...)
    x = if Optim.converged(res)
        Optim.minimizer(res)
    else
        @warn("Marčenko-Pastur fit did not converge, using a unit noise variance.")
        1.0
    end
    return x * op_sqrt_iq_sq
end
"""
    denoise!(dn::Option{<:AbstractDenoiseEstimator}, X::MatNum, q::Number) -> MatNum

In-place denoising of a covariance or correlation matrix using a [`Denoise`](@ref) estimator.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Mathematical definition

The spectrum of ``\\mathbf{X}`` is split at the Marčenko-Pastur upper bound:

```math
\\begin{align}
\\lambda_{+} &= \\sigma^2 \\left(1 + \\sqrt{\\frac{1}{q}}\\right)^2\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_plus_mp])
  - $(math_dict[:sigma2_noise])
  - $(math_dict[:q_mp])

The split is the whole of the mathematics that this function contributes. What happens to each side of it is the closed form of `dn.alg`.

# Algorithm

 1. Check that `X` is square.
 2. Read the diagonal of `X` into `s`. When any entry of `s` is not one, `X` is a covariance matrix: replace `s` with its square roots and convert `X` to a correlation matrix with `StatsBase.cov2cor!`. The test is `any(!isone, s)`, so it is the value of the diagonal that decides, never the type of `X`.
 3. Eigendecompose `X`, giving the ascending eigenvalues `vals` and the eigenvectors `vecs`.
 4. Fit the Marčenko-Pastur density to `vals` with [`find_max_eval`](@ref), giving `max_val`, the upper edge of the noise band.
 5. Count the eigenvalues that do not exceed `max_val`, giving `num_factors`, the number of noise eigenvalues.
 6. Rebuild `X` from the split spectrum with [`_denoise!`](@ref), through the branch that `dn.alg` selects.
 7. Repair the rebuilt matrix with [`posdef!`](@ref), under `dn.pdm`.
 8. When step 2 converted a covariance matrix, convert `X` back with `StatsBase.cor2cov!`. The standard deviations are the ones read in step 2, so the original diagonal returns exactly.

# Arguments

  - $(arg_dict[:odn])
      + `::Denoise`: The specified denoising algorithm is applied to `X` in-place.
      + `::Nothing`: No-op.
  - $(arg_dict[:sigrhoX])
  - `q`: The effective sample ratio `observations / assets`, used for spectral thresholding.

# Validation

  - `X` is square, checked with [`assert_matrix_issquare`](@ref). The `::Nothing` method returns before the check, so a `dn` of `nothing` accepts any `X`.

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

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
  - $(ref_dict[:cajas2025]) Section 3.5.2.
"""
function denoise!(::Nothing, X::MatNum, args...)::MatNum
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
    denoise(dn::Option{<:AbstractDenoiseEstimator}, X::MatNum, q::Number) -> MatNum

Out-of-place version of [`denoise!`](@ref).

# Algorithm

 1. Copy `X`.
 2. Apply [`denoise!`](@ref) to the copy, and return it. The input is never modified.

# Arguments

  - $(arg_dict[:odn])
      + `::Denoise`: The specified denoising algorithm is applied to a copy of `X`.
      + `::Nothing`: No-op, returns `X` unchanged.
  - $(arg_dict[:sigrhoX])
  - `q`: The effective sample ratio `observations / assets`, used for spectral thresholding.

# Returns

  - `X::MatNum`: A new matrix equal to the denoised version of the input.

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);
       X = X' * X;

julia> Xd = denoise(Denoise(), X, 10 / 5);

julia> size(Xd)
(5, 5)
```

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

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
  - $(ref_dict[:cajas2025]) Section 3.5.2.
"""
function denoise(::Nothing, X::MatNum, args...)::MatNum
    return X
end
function denoise(dn::AbstractDenoiseEstimator, X::MatNum, q::Number)
    X = copy(X)
    denoise!(dn, X, q)
    return X
end

export Denoise, SpectralDenoise, FixedDenoise, ShrunkDenoise, denoise, denoise!
