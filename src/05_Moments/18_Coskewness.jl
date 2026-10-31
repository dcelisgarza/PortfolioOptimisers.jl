"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all coskewness estimators.

A subtype computes the coskewness tensor of a returns matrix and the negative spectral skewness matrix `V` of that tensor.

The tensor is the ``N \\times N^{2}`` matrix of third comoments that Section 2.1 of [de2004finding](@cite) defines. That section cuts each ``N \\times N`` layer of the third comoment tensor and puts the layers side by side. Section 2.3 of [jondeau2004](@cite) restates it. `V` is the negative spectral skewness matrix that Section 2 of [nskew](@cite) defines.

# Interfaces

To add a coskewness estimator, subtype `CoskewnessEstimator`, keep its parameters and its observation weights in the fields of the struct, and add these methods:

## Coskewness

  - `PortfolioOptimisers.coskewness(ske::CoskewnessEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...) -> (MatNum, MatNum)`: Returns the coskewness tensor and its negative spectral skewness matrix.

### Arguments

  - $(arg_dict[:ske])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments.

### Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

## Factory

  - `PortfolioOptimisers.factory(ske::CoskewnessEstimator, w::PortfolioOptimisers.ObsWeights) -> CoskewnessEstimator`: Returns a copy of the estimator that holds the new observation weights.

### Arguments

  - $(arg_dict[:ske])
  - $(arg_dict[:ow])

### Returns

  - `ske::CoskewnessEstimator`: A new coskewness estimator of the same type, with the new weights.

## View

  - `PortfolioOptimisers.port_opt_view(ske::CoskewnessEstimator, i) -> CoskewnessEstimator`: Returns the estimator for the assets that `i` selects.

### Arguments

  - $(arg_dict[:ske])
  - `i`: The index or the indices of the assets to keep.

### Returns

  - $(ret_dict[:skev])

## Processor and factor prior

The library reads two more verbs of a coskewness estimator:

  - `PortfolioOptimisers.coskewness_processor(ske::CoskewnessEstimator) -> Option{<:AbstractMatrixProcessingEstimator}`: Optional. The matrix processing estimator that builds the negative spectral coskewness `V`. The fallback gives `nothing`. See [`coskewness_processor`](@ref).
  - `PortfolioOptimisers.coskewness_expected_returns(ske::CoskewnessEstimator) -> AbstractExpectedReturnsEstimator`: The expected returns estimator of the residual moments of a [`HighOrderFactorPriorEstimator`](@ref). It is necessary under `rsd = true`. The fallback refuses the type by name, and the prior calls it at construction. See [`coskewness_expected_returns`](@ref).

# Examples

The example below defines a minimal coskewness estimator.

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
  - [`coskewness_processor`](@ref)
  - [`coskewness_expected_returns`](@ref)

# References

  - $(ref_dict[:de2004finding]) Section 2.1, page 1338.
  - $(ref_dict[:jondeau2004]) Section 2.3.
  - $(ref_dict[:nskew]) Section 2.
"""
abstract type CoskewnessEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the coskewness tensor of a returns matrix, together with its negative spectral skewness matrix.

`Coskewness` holds a mean estimator, a matrix processing estimator and a moment algorithm. [`coskewness`](@ref) returns the two matrices as a pair. The first is the `assets × assets²` tensor. The second is the `assets × assets` matrix that [`negative_spectral_coskewness`](@ref) makes from the tensor, and it is not a processed copy of the tensor.

The tensor is the one of Section 2.1 of [de2004finding](@cite) and Section 2.3 of [jondeau2004](@cite). The library computes it with the matrix formula of Appendix A of [pkurt](@cite). The matrix is the negative spectral skewness matrix of Section 2 of [nskew](@cite). The observation weights and the matrix processing of `mp` are additions of this library.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Coskewness(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        alg::AbstractMomentAlgorithm = FullMoment(),
        w::Option{<:ObsWeights} = nothing,
        cvg::Option{<:CoveragePolicy} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> Coskewness

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated through [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `me`: Recursively indexed through [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).
  - `cache`: Dropped through [`obs_weights_view`](@ref), because no slice of a state exists on the observation axis.

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

  - $(ref_dict[:de2004finding]) Section 2.1, page 1338.
  - $(ref_dict[:jondeau2004]) Section 2.3.
  - $(ref_dict[:pkurt]) Appendix A.
  - $(ref_dict[:nskew]) Section 2.
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
    $(field_dict[:cvg])
    """
    cvg
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function Coskewness(me::AbstractExpectedReturnsEstimator,
                        mp::AbstractMatrixProcessingEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights}, cvg::Option{<:CoveragePolicy},
                        cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(mp), typeof(alg), typeof(w), typeof(cvg),
                   typeof(cache)}(me, mp, alg, w, cvg, cache)
    end
end
function Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                    alg::AbstractMomentAlgorithm = FullMoment(),
                    w::Option{<:ObsWeights} = nothing,
                    cvg::Option{<:CoveragePolicy} = nothing,
                    cache::Option{<:AbstractPartialFitState} = nothing)::Coskewness
    return Coskewness(me, mp, alg, w, cvg, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`Coskewness`](@ref), and leaves out `cvg` when it is `nothing`.

A `cvg` of `nothing` selects the reduce-and-expand path, which is the default of every estimator. A line for it gives the reader no data, and it changes the output of every doctest that shows the estimator. A coverage policy that is set is configuration, so it prints.

# Arguments

  - `ske`: The coskewness estimator. The method reads its `cvg` field alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:me, :mp, :alg, :w, :cache)` with no policy and `(:me, :mp, :alg, :w, :cvg, :cache)` with one.

# Related

  - [`Coskewness`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`show_fields`](@ref)
"""
function show_fields(ske::Coskewness)
    return if isnothing(ske.cvg)
        (:me, :mp, :alg, :w, :cache)
    else
        (:me, :mp, :alg, :w, :cvg, :cache)
    end
end
"""
    negative_spectral_part(vals::AbstractVector{<:Real}, vecs::MatNum, Tf::Type) -> MatNum
    negative_spectral_part(vals::AbstractVector{<:Complex}, vecs::MatNum, Tf::Type) -> MatNum

Rebuilds one block of a coskewness tensor from the negative part of its spectrum alone.

This function is the inner step of [`negative_spectral_coskewness`](@ref). It has one method for a real spectrum and one for a complex spectrum. `LinearAlgebra.eigen` of a general real matrix returns a real spectrum or a complex one, and only the values show which. The types of the caller do not set the element type of `vals`, so dispatch on `vals` selects the method, and each method body has a concrete element type.

The method sets every non-negative eigenvalue to zero and keeps every negative one, so the reconstruction is the negative semidefinite part of the block. The caller subtracts it, and the sum of these negated parts is positive semidefinite.

The complex method is a fallback and not a second definition. A block of the tensor is symmetric, so its spectrum is real. `eigen` returns a complex spectrum only when round-off makes the block asymmetric. The fallback clamps the real part and the imaginary part with the same bounds, and it returns the real part of the reconstruction.

# Arguments

  - `vals`: The eigenvalues of the block.
  - `vecs`: The eigenvectors of the block.
  - `Tf`: The element type of the tensor, which sets the bounds of the clamp.

# Returns

  - `S::MatNum`: The negative semidefinite part of the block, of the shape of the block.

# Related

  - [`negative_spectral_coskewness`](@ref)
  - [`Coskewness`](@ref)
"""
function negative_spectral_part(vals::AbstractVector{<:Real}, vecs::MatNum, Tf::Type)
    v = clamp.(vals, typemin(Tf), zero(Tf))
    return vecs * LinearAlgebra.Diagonal(v) * transpose(vecs)
end
function negative_spectral_part(vals::AbstractVector{<:Complex}, vecs::MatNum, Tf::Type)
    v = clamp.(real.(vals), typemin(Tf), zero(Tf)) .+
        clamp.(imag.(vals), typemin(Tf), zero(Tf)) .* im
    return real(vecs * LinearAlgebra.Diagonal(v) * transpose(vecs))
end
"""
    negative_spectral_coskewness(cskew::MatNum, X::MatNum,
                 mp::Option{<:AbstractMatrixProcessingEstimator}, T::Real = size(X, 1))

Builds the negative spectral skewness matrix of a coskewness tensor.

`negative_spectral_coskewness` splits the coskewness tensor into its `N` symmetric blocks of size `N x N`. It keeps the negative part of the spectrum of each block, and it adds the negated parts into one `N x N` matrix. The matrix processing estimator runs once, on that sum, and not on each block.

The function reduces the block of the tensor alone. This is the rule that [`matrix_processing_block!`](@ref) applies to a covariance frame. An available-case fit under a [`CoveragePolicy`](@ref) returns a frame. The tensor carries `NaN` across the row and the pair columns of each asset that the policy refused.

The function finds the block from the per-asset diagonal of the tensor, `cskew[i, (i - 1) * N + i]`. That entry is the third central moment of asset `i`. It is finite if and only if the fit gives a value for that asset. A non-finite cell inside the block is a triple of assets that the fit estimated, but that share no observation. [`assert_finite_block`](@ref) refuses that cell with an error that names it, so the cell never reaches `eigen`. The spectral step and `mp` run on the block alone. The function then writes the result into a `NaN` frame of the full width, so the row and the column of each asset that the policy refused hold `NaN`.

Every site of the library that reduces a coskewness tensor to `V` calls this function, so the block rule has one definition. The callers include [`_coskewness`](@ref), the available-case arm of [`coverage_coskewness`](@ref), [`port_opt_view`](@ref) of a [`HighOrderPrior`](@ref), the high order factor prior and [`NegativeSkewness`](@ref).

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

  - ``\\mathbf{S}_{i}``: The ``i``-th ``N \\times N`` block of the coskewness tensor.
  - ``\\mathbf{Q}_{i}``: The matrix of the eigenvectors of ``\\mathbf{S}_{i}``.
  - ``\\mathbf{\\Lambda}_{i}^{-}``: The diagonal matrix of the negative eigenvalues of ``\\mathbf{S}_{i}``, with each non-negative eigenvalue set to zero.
  - ``\\mathbf{V}``: The negative spectral skewness matrix. It is positive semidefinite, because it is a sum of negated negative semidefinite matrices.
  - $(math_dict[:N])

The entry ``\\mathbf{S}_{i,\\,aj}`` is the third comoment of the deviations of the assets ``a``, ``i`` and ``j``. That comoment does not change when the order of its three assets changes, so the block is symmetric.

Section 2 of [nskew](@cite) splits each block as ``\\mathbf{S}_{i} = \\mathbf{S}_{i}^{+} + \\mathbf{S}_{i}^{-}`` and writes the negative quadratic skewness as ``\\boldsymbol{x}^\\intercal \\mathbf{V}^{-} \\boldsymbol{x}`` with ``\\mathbf{V}^{-} = -\\sum_{i} \\mathbf{S}_{i}^{-}``. That is the ``\\mathbf{V}`` above. The work gets this form when it replaces each weight ``x_i`` of the skewness with ``1/N``. The matrix processing of the last step is an addition of this library.

# Algorithm

 1. Read `N` from the row count of `cskew`. Take the block `blk` as `isfinite.(cskew[i, (i - 1) * N + i])` over `i`.
 2. When `blk` holds no `true`, reduce the whole tensor. The tensor has no block, so no cell is refused here, and `LinearAlgebra.eigen` raises `ArgumentError: matrix contains Infs or NaNs` on the first block.
 3. Refuse a non-finite cell inside the block with [`assert_finite_block`](@ref). When `blk` holds no `false`, the block is the whole tensor, and the refusal covers that case too.
 4. Cut the tensor to the block, at the rows `blk` and the pair columns of [`coverage_pair_index`](@ref). Cut `X` to the same assets. When `blk` holds no `false`, the function does not copy either of them.
 5. Allocate the accumulator `V` of zeros at the width of the block.
 6. For each block index `i`, take `coskew_jk`, the view of the columns `(i - 1) * N + 1` to `i * N` of the cut tensor.
 7. Take the eigenvalues `vals` and the eigenvectors `vecs` of `coskew_jk`. Subtract from `V` the negative semidefinite part that [`negative_spectral_part`](@ref) rebuilds from them.
 8. Run [`sample_matrix_processing!`](@ref) once, on the accumulated `V`, with the cut `X` and `T`. The function does not process a block on its own.
 9. When the block is the whole tensor, return `V`. Otherwise, write `V` into the block of a `NaN` frame of the full width, and return the frame.

# Arguments

  - `cskew`: The coskewness tensor, `assets × assets²`, in `N` blocks of `N` columns. It can be a frame, in which each refused asset carries `NaN` across its row and its pair columns.
  - `X`: The data matrix (observations × assets). [`matrix_processing!`](@ref) reads it, and the spectral step does not.
  - $(arg_dict[:omp])
      + `::AbstractMatrixProcessingEstimator`: The estimator processes the accumulated `V` in place.
      + `::Nothing`: The function does not process `V`, which is the sum of the negated negative parts.
  - `T`: The effective number of observations of the tensor, which the denoising step reads. It is `size(X, 1)` by default, the count of an unweighted tensor. [`_coskewness`](@ref) passes the count of its weights.

# Validation

  - The block of `cskew` is finite. An `IsNonFiniteError` that names the cell is thrown otherwise.

# Returns

  - $(ret_dict[:cskewV]) It is the ``\\mathbf{V}`` above, so a portfolio's negative quadratic skewness is the quadratic form ``\\boldsymbol{w}^{\\intercal} \\mathbf{V} \\boldsymbol{w}``. An asset outside the block carries `NaN` across its row and its column.

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing_block!`](@ref)
  - [`negative_spectral_part`](@ref)
  - [`assert_finite_block`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coskewness`](@ref)

# References

  - $(ref_dict[:nskew]) Section 2.
"""
function negative_spectral_coskewness(cskew::MatNum, X::MatNum,
                                      mp::Option{<:AbstractMatrixProcessingEstimator},
                                      T::Real = size(X, 1))
    N0 = size(cskew, 1)
    blk = BitVector(isfinite(cskew[i, (i - 1) * N0 + i]) for i in 1:N0)
    # A tensor with no finite per-asset diagonal has no block, so it is the plain path's and
    # meets its refusal. Everything else is refused with an error that names it, the
    # complete tensor included: an available-case triple with an empty intersection has a
    # complete diagonal, so a short-circuit past the refusal would send the case it was
    # written for to LAPACK.
    frame = any(blk) && !all(blk)
    Xb = frame ? X[:, blk] : X
    cskew_b = frame ? cskew[blk, coverage_pair_index(blk)] : cskew
    if any(blk)
        assert_finite_block(cskew_b)
    end
    N = size(cskew_b, 1)
    V = zeros(eltype(cskew_b), N, N)
    for i in 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew_b, :, j:k)
        vals, vecs = LinearAlgebra.eigen(coskew_jk)
        V .-= negative_spectral_part(vals, vecs, eltype(cskew_b))
    end
    sample_matrix_processing!(mp, V, Xb, T)
    if !frame
        return V
    end
    Vf = coverage_nan_frame(V, (N0, N0))
    Vf[blk, blk] = V
    return Vf
end
"""
    _coskewness(Y::MatNum, X::MatNum, mp::AbstractMatrixProcessingEstimator, w::Option{<:StatsBase.AbstractWeights}) -> (MatNum, MatNum)

Builds the coskewness tensor of a deviation matrix, and the negative spectral skewness matrix of that tensor.

The matrix processing estimator runs on the negative spectral skewness matrix alone, inside [`negative_spectral_coskewness`](@ref). It never runs on the tensor.

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

  - ``\\hat{\\mathbf{S}}``: The ``N \\times N^{2}`` coskewness tensor. Its entry ``\\hat{\\mathbf{S}}_{a,\\,(i-1)N+j}`` is the third comoment of the deviations of the assets ``a``, ``i`` and ``j``, so each of its ``N`` blocks of ``N`` columns is symmetric.
  - $(math_dict[:Y_dev])
  - $(math_dict[:y_t_dev])
  - $(math_dict[:Z_pairprod])
  - $(math_dict[:w_obs_vec])
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - ``\\boldsymbol{1}``: The ``N \\times 1`` vector of ones.
  - ``\\otimes``: The Kronecker product.
  - ``\\odot``: The element-wise product. When the operands differ in shape, it broadcasts along the row axis.

The unweighted tensor is the matrix formula for ``\\mathbf{M}_3`` in Appendix A of [pkurt](@cite). Its entries are the third comoments of Section 2.1 of [de2004finding](@cite) and Section 2.3 of [jondeau2004](@cite). In this layout, the entry of row ``j`` and column ``(i-1)N+k`` is the third comoment of the assets ``i``, ``j`` and ``k``. The weighted form is an addition of this library.

# Algorithm

 1. Build `o`, the ``1 \\times N`` row of ones.
 2. Build `z`, the pairwise expansion `kron(o, Y) ⊙ kron(Y, o)`. Its column `(i - 1) * N + j` is the element-wise product of the columns `i` and `j` of `Y`.
 3. Without weights, form `cskew` as `transpose(Y) * z / size(Y, 1)`.
 4. With weights, form `cskew` as `transpose(w .* Y) * z / sum(w)`. The weights multiply the left factor alone, so each summand carries one weight and not three.
 5. Reduce `cskew` to `V` with [`negative_spectral_coskewness`](@ref), which runs `mp` on the reduced matrix. With weights, pass the count of the weights from [`observation_weight_count`](@ref), so that the denoising step reads the effective number of observations and not `size(X, 1)`.
 6. Return the pair `(cskew, V)`.

# Arguments

  - `Y`: The deviation matrix (observations × assets). The caller centres it.
  - `X`: The data matrix (observations × assets). It goes to [`sample_matrix_processing!`](@ref) through [`negative_spectral_coskewness`](@ref).
  - `mp`: The matrix processing estimator.
  - `w`: The optional observation weights. The unweighted method takes `nothing` through its `args...`.

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Related

  - [`Coskewness`](@ref)
  - [`negative_spectral_coskewness`](@ref)
  - [`coskewness`](@ref)

# References

  - $(ref_dict[:de2004finding]) Section 2.1, page 1338.
  - $(ref_dict[:jondeau2004]) Section 2.3.
  - $(ref_dict[:pkurt]) Appendix A.
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
    V = negative_spectral_coskewness(cskew, X, mp, observation_weight_count(w, size(X, 1)))
    return cskew, V
end
"""
    coskewness(ske::Option{<:Coskewness}, X::MatNum; dims::Int = 1,
               mean = nothing, kwargs...)

Computes the coskewness tensor of a dataset, and the negative spectral skewness matrix of that tensor.

The method applies the observation weights of `ske.w` when they are set. [`FullMoment`](@ref) takes the centred returns, and [`SemiMoment`](@ref) clips every positive deviation to zero. When the estimator is `nothing`, the method returns `(nothing, nothing)`.

`ske.w` weights the whole estimate, so it applies to the centre and to the deviations. When `mean` is `nothing` and `ske.w` is not, the method sends `ske.me` through [`factory`](@ref) with `ske.w`. So `ske.w` replaces the weights that `ske.me` holds. Pass `mean` for a centre that `ske.w` does not describe.

The two returned matrices are different objects. The first is the coskewness tensor. The second is the negative spectral skewness matrix that [`negative_spectral_coskewness`](@ref) makes from the tensor.

# Algorithm

The method passes its arguments to [`coverage_coskewness`](@ref), which selects an arm by dispatch on `ske.cvg`. When `ske.cvg` is `nothing`, the arm does these steps.

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`.
 2. Resolve the observation weights `w` from `ske.w` with [`get_observation_weights`](@ref).
 3. Resolve the centre `mu` from `ske.me` and `ske.w` with [`weighted_centre`](@ref), which reads `mean` when the caller gave one.
 4. Form the deviation matrix `Y`. [`FullMoment`](@ref) takes `X .- mu`, and [`SemiMoment`](@ref) takes `min.(X .- mu, 0)`.
 5. Delegate to [`_coskewness`](@ref) with `Y`, `X`, `ske.mp` and `w`, and return the pair it returns.

# Arguments

  - `ske`: The coskewness estimator.

      + `ske::Coskewness{<:Any, <:Any, <:FullMoment}`: A coskewness estimator with the [`FullMoment`](@ref) moment algorithm.
      + `ske::Coskewness{<:Any, <:Any, <:SemiMoment}`: A coskewness estimator with the [`SemiMoment`](@ref) moment algorithm.
      + `ske::Nothing`: The method returns `(nothing, nothing)`.

  - `X`: The data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `mean`: The optional mean vector. When it is `nothing`, the method computes the centre with `ske.me` and `ske.w`. The available-case arm refuses a mean vector.

  - `active_mask`: The active mask of the Asset Panel over the window, or `nothing`. [`coverage_coskewness`](@ref) states how each arm reads it.

  - `kwargs...`: More keyword arguments. When `ske.cvg` is `nothing`, the method passes them to [`get_observation_weights`](@ref) and to [`weighted_centre`](@ref).

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
                    mean = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
    return coverage_coskewness(ske, ske.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1,
                    mean = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
    return coverage_coskewness(ske, ske.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
"""
    coverage_coskewness(ske, cvg, X; dims::Int = 1, mean = nothing,
                        active_mask = nothing, kwargs...) -> (MatNum, MatNum)

Computes the coskewness tensor and its negative spectral skewness matrix with the arm that the coverage policy of the estimator selects.

The caller passes the `cvg` field of the estimator as the second argument. So dispatch on the policy selects the arm, and no branch reads the value of the policy. [`coverage_covariance`](@ref) selects the arm of a covariance in the same way. The moment algorithm of `ske` sets the deviations from the mean of each asset. [`FullMoment`](@ref) keeps them whole, and [`SemiMoment`](@ref) sets each positive deviation to zero.

# Arguments

  - `ske`: The coskewness estimator.
  - `cvg`: The coverage policy of the estimator, which selects the arm.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: The optional mean vector. The available-case arm refuses one.
  - `active_mask`: The active mask of the Asset Panel over the window, or `nothing`. The Coverage Universe arm refuses an inactive cell, as it refuses a non-finite return. The available-case arm leaves the cell out of each comoment of its asset.
  - `kwargs...`: More keyword arguments. The Coverage Universe arm passes them to [`get_observation_weights`](@ref) and to [`weighted_centre`](@ref).

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Related

  - [`Coskewness`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_covariance`](@ref)
  - [`coverage_comoment_block`](@ref)
"""
function coverage_coskewness end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the [`FullMoment`](@ref) arm of [`coverage_coskewness`](@ref) for an estimator with no coverage policy.

This is the Coverage Universe arm. It centres the whole window on one vector and delegates to [`_coskewness`](@ref). It refuses an inactive cell of `active_mask` with [`assert_finite_sample`](@ref), because a plain estimator reads that cell as a non-finite return.

# Related

  - [`coverage_coskewness`](@ref)
  - [`_coskewness`](@ref)
"""
function coverage_coskewness(ske::Coskewness{<:Any, <:Any, <:FullMoment}, ::Nothing,
                             X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X, dims_oriented(dims, active_mask))
    w = get_observation_weights(ske.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, ske.me, ske.w; dims = 1, mean = mean, kwargs...)
    Y = X .- mu
    return _coskewness(Y, X, ske.mp, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the [`SemiMoment`](@ref) arm of [`coverage_coskewness`](@ref) for an estimator with no coverage policy.

This is the Coverage Universe arm. It centres the whole window on one vector, sets each positive deviation to zero, and delegates to [`_coskewness`](@ref). It refuses an inactive cell of `active_mask` with [`assert_finite_sample`](@ref), because a plain estimator reads that cell as a non-finite return.

# Related

  - [`coverage_coskewness`](@ref)
  - [`_coskewness`](@ref)
"""
function coverage_coskewness(ske::Coskewness{<:Any, <:Any, <:SemiMoment}, ::Nothing,
                             X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X, dims_oriented(dims, active_mask))
    w = get_observation_weights(ske.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, ske.me, ske.w; dims = 1, mean = mean, kwargs...)
    Y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(Y, X, ske.mp, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the available-case arm of [`coverage_coskewness`](@ref) for an estimator with a [`CoveragePolicy`](@ref).

The method fits each triple on the observations at which its three assets are all finite and active. Each cell has its own denominator.

The centre of each asset is the available-case mean of that asset, over its own finite and active observations. It is not a mean over the observations of the triple, because a centre for each triple needs `assets³` running means. The [`SemiMoment`](@ref) arm of [`coverage_covariance`](@ref) centres each asset on its own mean for the same reason. So the method refuses an external `mean`, as that arm does.

The [`FullMoment`](@ref) and [`SemiMoment`](@ref) cases differ only in the clip, so they share this method through [`coverage_comoment_block`](@ref). Neither case folds. An exact recursion for each cell at this order needs the pairwise second comoments over the observations of each triple. So the online form of an available-case coskewness refits a buffer through [`Online`](@ref).

# Algorithm

 1. Get the valid entries, the deviations from the available-case centre of each asset, and the mask of the admitted assets with [`coverage_comoment_block`](@ref).
 2. Take the numerator as `transpose(Y) * z` and the denominator as `transpose(Mi) * zc`, where `z` and `zc` are the pairwise expansions of the zeroed deviations and of the valid mask.
 3. Divide with [`coverage_divide`](@ref), and frame the refused assets down the asset axis and the pair axis with [`coverage_refuse_comoment!`](@ref).
 4. Reduce the frame with [`negative_spectral_coskewness`](@ref), which keeps the block rule.

# Related

  - [`coverage_coskewness`](@ref)
  - [`coverage_comoment_block`](@ref)
  - [`coverage_divide`](@ref)
  - [`negative_spectral_coskewness`](@ref)
"""
function coverage_coskewness(ske::Coskewness, cvg::CoveragePolicy, X::MatNum; dims::Int = 1,
                             mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    assert_dims(dims)
    assert_partial_fittable(ske.me, ske.w, "Coskewness")
    @argcheck(isnothing(mean),
              ArgumentError("an available-case coskewness centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    Xo, Y, Mi, z, zc, cmsk = coverage_comoment_block(ske.alg, cvg, X, active_mask, dims)
    cskew = coverage_divide(transpose(Y) * z, transpose(Mi) * zc, false, nothing)
    coverage_refuse_comoment!(cskew, cmsk, Val(:sk))
    return cskew, negative_spectral_coskewness(cskew, Xo, ske.mp)
end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end
# A coskewness estimator with no weights of its own reads every row with the same weight. A
# weighted `Coskewness` is worth the count of its weights, the count that its denoiser reads.
# The fit takes `mean` and `active_mask` by name, and the weights read the other keywords.
function effective_observation_count(::CoskewnessEstimator, X::MatNum; kwargs...)
    return size(X, 1)
end
function effective_observation_count(ske::Coskewness, X::MatNum; mean = nothing,
                                     active_mask = nothing, kwargs...)
    return observation_weight_count(get_observation_weights(ske.w, X; dims = 1, kwargs...),
                                    size(X, 1))
end

export Coskewness, coskewness
public CoskewnessEstimator
