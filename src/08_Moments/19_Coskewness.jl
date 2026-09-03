"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all coskewness estimators.

All concrete and/or abstract types implementing coskewness estimation algorithms should be subtypes of `CoskewnessEstimator`.

# Interfaces

In order to implement a new coskewness estimator which will work seamlessly with the library, subtype `CoskewnessEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Coskewness

  - `PortfolioOptimisers.coskewness(ske::CoskewnessEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...) -> (MatNum, MatNum)`: Computes the coskewness tensor and processed matrix.

### Arguments

  - $(arg_dict[:ske])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments.

### Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

## Factory

  - `PortfolioOptimisers.factory(ske::CoskewnessEstimator, w::PortfolioOptimisers.ObsWeights) -> CoskewnessEstimator`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:ske])
  - $(arg_dict[:ow])

### Returns

  - `ske::CoskewnessEstimator`: New coskewness estimator of the same type, with the new weights applied.

## View

  - `PortfolioOptimisers.port_opt_view(ske::CoskewnessEstimator, i) -> CoskewnessEstimator`: Returns a view of the estimator for the `i`-th element(s).

### Arguments

  - $(arg_dict[:ske])
  - `i`: Index or indices.

### Returns

  - $(ret_dict[:skev])

# Examples

We can create a dummy coskewness estimator as follows:

```jldoctest
julia> struct MyCoskewnessEstimator{T1} <: PortfolioOptimisers.CoskewnessEstimator
           w::T1
           function MyCoskewnessEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyCoskewnessEstimator(;
                                      w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyCoskewnessEstimator(w)
       end
MyCoskewnessEstimator

julia> function PortfolioOptimisers.factory(::MyCoskewnessEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
           return MyCoskewnessEstimator(; w = w)
       end

julia> function PortfolioOptimisers.port_opt_view(ske::MyCoskewnessEstimator, i)
           return ske
       end

julia> function PortfolioOptimisers.coskewness(ske::MyCoskewnessEstimator,
                                               X::PortfolioOptimisers.MatNum; dims::Int = 1,
                                               mean = nothing, kwargs...)
           N = size(X, 2)
           return zeros(N, N^2), zeros(N, N)
       end

julia> cskew, V = coskewness(MyCoskewnessEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1]);

julia> cskew
2×4 Matrix{Float64}:
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> V
2×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0

julia> PortfolioOptimisers.factory(MyCoskewnessEstimator(), StatsBase.Weights([1, 2, 3]))
MyCoskewnessEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`Coskewness`](@ref)
  - [`AbstractEstimator`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.1.4, Equation 3.6.
  - $(ref_dict[:pkurt])
"""
abstract type CoskewnessEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the coskewness tensor of a returns matrix, together with its negative spectral skewness matrix.

`Coskewness` composes a mean estimator, a matrix processing estimator and a moment algorithm. [`coskewness`](@ref) returns both matrices as a pair: the `assets × assets²` tensor first, and the `assets × assets` matrix that [`negative_spectral_coskewness`](@ref) reduces it to second. The second is not a processed copy of the first.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Coskewness(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        alg::AbstractMomentAlgorithm = FullMoment(),
        w::Option{<:ObsWeights} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> Coskewness

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `me`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> Coskewness()
Coskewness
     me ┼ SimpleExpectedReturns
        │   w ┴ nothing
     mp ┼ MatrixProcessing
        │     pdm ┼ Posdef
        │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │      dn ┼ nothing
        │      dt ┼ nothing
        │     alg ┼ nothing
        │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
    alg ┼ FullMoment()
      w ┼ nothing
  cache ┴ nothing
```

# Related

  - [`CoskewnessEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
  - [`negative_spectral_coskewness`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Sections 3.1.4 and 7.2.5.1, Equations 3.6, 7.104 and 7.105.
  - $(ref_dict[:pkurt])
  - $(ref_dict[:nskew])
"""
@propagatable @concrete struct Coskewness <: CoskewnessEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:malg])
    """
    alg
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:pfcache])
    """
    cache
    function Coskewness(me::AbstractExpectedReturnsEstimator,
                        mp::AbstractMatrixProcessingEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights}, cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(mp), typeof(alg), typeof(w), typeof(cache)}(me, mp,
                                                                                  alg, w,
                                                                                  cache)
    end
end
function Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                    alg::AbstractMomentAlgorithm = FullMoment(),
                    w::Option{<:ObsWeights} = nothing,
                    cache::Option{<:AbstractPartialFitState} = nothing)::Coskewness
    return Coskewness(me, mp, alg, w, cache)
end
"""
    negative_spectral_coskewness(cskew::MatNum, X::MatNum,
                 mp::Option{<:AbstractMatrixProcessingEstimator})

Internal helper that builds the negative spectral skewness matrix.

`negative_spectral_coskewness` splits the coskewness tensor into its `N` symmetric blocks of size `N x N`, keeps the negative part of the spectrum of each block, and sums the negated parts into one `N x N` matrix. The matrix processing estimator runs once, on the summed result, and not on the individual blocks.

# Mathematical definition

Write the coskewness tensor as ``\\mathbf{S} = [\\mathbf{S}_{1} \\vert \\mathbf{S}_{2} \\vert \\ldots \\vert \\mathbf{S}_{N}]``. Each block ``\\mathbf{S}_{i}`` is symmetric, so its eigendecomposition is real. Keep the negative eigenvalues alone and negate the sum:

```math
\\begin{align}
\\mathbf{S}_{i} &= \\mathbf{Q}_{i} \\mathbf{\\Lambda}_{i} \\mathbf{Q}_{i}^{\\intercal}\\,, \\\\
\\mathbf{S}_{i}^{-} &= \\mathbf{Q}_{i} \\mathbf{\\Lambda}_{i}^{-} \\mathbf{Q}_{i}^{\\intercal}\\,, \\\\
\\mathbf{V} &= -\\sum\\limits_{i=1}^{N} \\mathbf{S}_{i}^{-}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{S}_{i}``: ``i``-th ``N \\times N`` block of the coskewness tensor.
  - ``\\mathbf{Q}_{i}``: Matrix of eigenvectors of ``\\mathbf{S}_{i}``.
  - ``\\mathbf{\\Lambda}_{i}^{-}``: Diagonal matrix holding the negative eigenvalues of ``\\mathbf{S}_{i}``, with every non-negative one set to zero.
  - ``\\mathbf{V}``: Negative spectral skewness matrix. It is positive semidefinite, because it is a sum of negated negative semidefinite matrices.
  - $(math_dict[:N])

The entry ``\\mathbf{S}_{i,\\,aj}`` is the third comoment of the deviations of the assets ``a``, ``i`` and ``j``. That comoment does not depend on the order of its three assets, which is why the block is symmetric.

# Algorithm

 1. Read `N` from the row count of `cskew`, and allocate the ``N \\times N`` accumulator `V` of zeros.
 2. For each block index `i`, take `coskew_jk`, the view of the columns `(i - 1) * N + 1` to `i * N` of `cskew`.
 3. Eigendecompose `coskew_jk`, giving the eigenvalues `vals` and the eigenvectors `vecs`.
 4. When `vals` is real, clamp every entry to zero from above, so a non-negative eigenvalue becomes zero and a negative one is kept. Subtract the reconstruction `vecs * Diagonal(vals) * transpose(vecs)` from `V`.
 5. When `vals` is complex, clamp the real part and the imaginary part the same way, and subtract the real part of the reconstruction. `LinearAlgebra.eigen` returns a complex spectrum only when round-off leaves the block asymmetric, so this branch is a fallback and not the definition above.
 6. Run [`matrix_processing!`](@ref) once, on the accumulated `V`. No block is processed on its own.

# Arguments

  - `cskew`: Coskewness tensor, `assets × assets²`, laid out as `N` blocks of `N` columns.
  - `X`: Data matrix (observations × assets). [`matrix_processing!`](@ref) reads it, and the spectral step does not.
  - $(arg_dict[:omp])
      + `::AbstractMatrixProcessingEstimator`: The estimator processes the accumulated `V` in-place.
      + `::Nothing`: No-op. `V` is the raw sum of the negated negative parts.

# Returns

  - $(ret_dict[:cskewV]) It is the ``\\mathbf{V}`` above, so a portfolio's negative quadratic skewness is the quadratic form ``\\boldsymbol{w}^{\\intercal} \\mathbf{V} \\boldsymbol{w}``.

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`matrix_processing!`](@ref)
  - [`coskewness`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.5.1, Equations 7.104 and 7.105.
  - $(ref_dict[:nskew])
"""
function negative_spectral_coskewness(cskew::MatNum, X::MatNum,
                                      mp::Option{<:AbstractMatrixProcessingEstimator})
    N = size(cskew, 1)
    V = zeros(eltype(cskew), N, N)
    for i in 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew, :, j:k)
        vals, vecs = LinearAlgebra.eigen(coskew_jk)
        if eltype(vals) <: Real
            vals .= clamp.(vals, typemin(eltype(cskew)), zero(eltype(cskew)))
            V .-= vecs * LinearAlgebra.Diagonal(vals) * transpose(vecs)
        else
            vals .= clamp.(real.(vals), typemin(eltype(cskew)), zero(eltype(cskew))) +
                    clamp.(imag.(vals), typemin(eltype(cskew)), zero(eltype(cskew)))im
            V .-= real(vecs * LinearAlgebra.Diagonal(vals) * transpose(vecs))
        end
    end
    matrix_processing!(mp, V, X)
    return V
end
"""
    _coskewness(Y::MatNum, X::MatNum, mp::AbstractMatrixProcessingEstimator, w::Option{<:StatsBase.AbstractWeights}) -> (MatNum, MatNum)

Internal helper that builds the coskewness tensor from a deviation matrix.

`_coskewness` returns the tensor together with its negative spectral skewness matrix. The matrix processing estimator runs on the second one alone, inside [`negative_spectral_coskewness`](@ref), and never on the tensor.

# Mathematical definition

The ``N \\times N^{2}`` coskewness tensor is the third comoment matrix of the deviations:

Unweighted:

```math
\\begin{align}
\\hat{\\mathbf{S}} &= \\frac{1}{T} \\mathbf{Y}^\\intercal \\mathbf{Z}\\,, \\\\
\\mathbf{Z}_{t,\\cdot} &= (\\boldsymbol{1}^\\intercal \\otimes \\boldsymbol{y}_t^\\intercal) \\odot (\\boldsymbol{y}_t^\\intercal \\otimes \\boldsymbol{1}^\\intercal)\\,.
\\end{align}
```

Weighted:

```math
\\begin{align}
\\hat{\\mathbf{S}} &= \\frac{1}{\\sum_{t=1}^{T} w_t} (\\boldsymbol{w} \\odot \\mathbf{Y})^\\intercal \\mathbf{Z}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{S}}``: ``N \\times N^{2}`` coskewness tensor. Its entry ``\\hat{\\mathbf{S}}_{a,\\,(i-1)N+j}`` is the third comoment of the deviations of the assets ``a``, ``i`` and ``j``, so each of its ``N`` blocks of ``N`` columns is symmetric.
  - $(math_dict[:Y_dev])
  - $(math_dict[:y_t_dev])
  - $(math_dict[:Z_pairprod])
  - $(math_dict[:w_obs_vec])
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - ``\\boldsymbol{1}``: ``N \\times 1`` vector of ones.
  - ``\\otimes``: Kronecker product.
  - ``\\odot``: Element-wise product. Where the operands differ in shape, it broadcasts along the row axis.

# Algorithm

 1. Build `o`, the ``1 \\times N`` row of ones.
 2. Build `z`, the pairwise expansion `kron(o, Y) ⊙ kron(Y, o)`. Its column `(i - 1) * N + j` is the element-wise product of the columns `i` and `j` of `Y`.
 3. Without weights, form `cskew` as `transpose(Y) * z / size(Y, 1)`.
 4. With weights, form `cskew` as `transpose(w .* Y) * z / sum(w)`. The weights multiply the left factor alone, so each summand carries one weight and not three.
 5. Reduce `cskew` to `V` with [`negative_spectral_coskewness`](@ref), which runs `mp` on the reduced matrix.
 6. Return the pair `(cskew, V)`.

# Arguments

  - `Y`: Deviation matrix (observations × assets), already centred by the caller.
  - `X`: Data matrix (observations × assets). It reaches [`matrix_processing!`](@ref) through [`negative_spectral_coskewness`](@ref).
  - `mp`: Matrix processing estimator.
  - `w`: Optional observation weights. The unweighted method takes `nothing` through its `args...`.

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Related

  - [`Coskewness`](@ref)
  - [`negative_spectral_coskewness`](@ref)
  - [`coskewness`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.1.4, Equation 3.6.
  - $(ref_dict[:pkurt])
"""
function _coskewness(Y::MatNum, X::MatNum, mp::AbstractMatrixProcessingEstimator, args...)
    o = transpose(range(one(eltype(Y)), one(eltype(Y)); length = size(Y, 2)))
    z = kron(o, Y) ⊙ kron(Y, o)
    cskew = transpose(Y) * z / size(Y, 1)
    V = negative_spectral_coskewness(cskew, X, mp)
    return cskew, V
end
function _coskewness(Y::MatNum, X::MatNum, mp::AbstractMatrixProcessingEstimator,
                     w::StatsBase.AbstractWeights)
    o = transpose(range(one(eltype(Y)), one(eltype(Y)); length = size(Y, 2)))
    z = kron(o, Y) ⊙ kron(Y, o)
    cskew = transpose(w .* Y) * z / sum(w)
    V = negative_spectral_coskewness(cskew, X, mp)
    return cskew, V
end
"""
    coskewness(ske::Option{<:Coskewness}, X::MatNum; dims::Int = 1,
               mean = nothing, kwargs...)

Compute the coskewness tensor of a dataset, together with its negative spectral skewness matrix. Observation weights in `ske.w` are applied if set. [`FullMoment`](@ref) takes the centred returns, and [`SemiMoment`](@ref) clips every positive deviation to zero. If the estimator is `nothing`, returns `(nothing, nothing)`.

`ske.w` weights the whole estimate, so it reaches the centre as well as the deviations. When `mean` is `nothing` and `ske.w` is not, the method sends `ske.me` through [`factory`](@ref) with `ske.w`, so `ske.w` wins over the weights that `ske.me` carries. Pass `mean` for a centre that `ske.w` does not describe. ADR 0088 records the decision.

The two returned matrices are different objects. The first is the coskewness tensor itself, and the second is the negative spectral skewness matrix that [`negative_spectral_coskewness`](@ref) reduces it to.

# Algorithm

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`.
 2. Resolve the observation weights `w` from `ske.w` with [`get_observation_weights`](@ref).
 3. Resolve the centre `mu` from `ske.me` and `ske.w` with [`weighted_centre`](@ref), which reads `mean` when the caller gave one.
 4. Form the deviation matrix `Y`. [`FullMoment`](@ref) takes `X .- mu`, and [`SemiMoment`](@ref) takes `min.(X .- mu, 0)`.
 5. Delegate to [`_coskewness`](@ref) with `Y`, `X`, `ske.mp` and `w`, and return the pair it returns.

# Arguments

  - `ske`: Coskewness estimator.

      + `ske::Coskewness{<:Any, <:Any, <:FullMoment}`: Coskewness estimator with [`FullMoment`](@ref) moment algorithm.
      + `ske::Coskewness{<:Any, <:Any, <:SemiMoment}`: Coskewness estimator with [`SemiMoment`](@ref) moment algorithm.
      + `ske::Nothing`: No-op, returns `(nothing, nothing)`.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `mean`: Optional mean vector. If not provided, computed using the estimator's mean estimator.

  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = randn(rng, 10, 3);

julia> cskew, V = coskewness(Coskewness(), X);

julia> cskew
3×9 Matrix{Float64}:
 -0.329646    0.0782455   0.325842  …   0.325842  -0.250881   0.16769
  0.0782455  -0.236104   -0.250881     -0.250881   0.266005   0.144546
  0.325842   -0.250881    0.16769       0.16769    0.144546  -0.605589

julia> V
3×3 Matrix{Float64}:
  0.513743   -0.0452078  -0.290893
 -0.0452078   0.402765   -0.0372996
 -0.290893   -0.0372996   0.837701
```

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`weighted_centre`](@ref)
  - [`negative_spectral_coskewness`](@ref)
"""
function coskewness(ske::Coskewness{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    X = dims_oriented(dims, X)
    w = get_observation_weights(ske.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, ske.me, ske.w; dims = 1, mean = mean, kwargs...)
    Y = X .- mu
    return _coskewness(Y, X, ske.mp, w)
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    X = dims_oriented(dims, X)
    w = get_observation_weights(ske.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, ske.me, ske.w; dims = 1, mean = mean, kwargs...)
    Y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(Y, X, ske.mp, w)
end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end

export Coskewness, coskewness
