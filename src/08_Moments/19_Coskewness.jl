"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all coskewness estimators in `PortfolioOptimisers.jl`.

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

  - `cskew::MatNum`: Coskewness tensor `features × features^2`.
  - `V::MatNum`: Processed coskewness matrix `features × features`.

## Factory

  - `PortfolioOptimisers.factory(ske::CoskewnessEstimator, w::PortfolioOptimisers.ObsWeights) -> CoskewnessEstimator`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:ske])
  - $(arg_dict[:ow])

### Returns

  - `ske::CoskewnessEstimator`: New coskewness estimator of the same type, with the new weights applied.

## View

  - `PortfolioOptimisers.moment_view(ske::CoskewnessEstimator, i) -> CoskewnessEstimator`: Returns a view of the estimator for the `i`-th element(s).

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

julia> function PortfolioOptimisers.moment_view(ske::MyCoskewnessEstimator, i)
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
"""
abstract type CoskewnessEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Container type for coskewness estimators.

`Coskewness` encapsulates the mean estimator, matrix processing estimator, and moment algorithm for coskewness estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Coskewness(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
        alg::AbstractMomentAlgorithm = Full(),
        w::Option{<:ObsWeights} = nothing
    ) -> Coskewness

Keywords correspond to the struct's fields.

# Validation

  - $(val_dict[:oow])

# Examples

```jldoctest
julia> Coskewness()
Coskewness
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   mp ┼ DenoiseDetoneAlgMatrixProcessing
      │     pdm ┼ Posdef
      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      dn ┼ nothing
      │      dt ┼ nothing
      │     alg ┼ nothing
      │   order ┴ DenoiseDetoneAlg()
  alg ┼ Full()
    w ┴ nothing
```

# Related

  - [`CoskewnessEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
@concrete struct Coskewness <: CoskewnessEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:mp])"
    mp
    "$(field_dict[:malg])"
    alg
    "$(field_dict[:oow])"
    w
    function Coskewness(me::AbstractExpectedReturnsEstimator,
                        mp::AbstractMatrixProcessingEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(mp), typeof(alg), typeof(w)}(me, mp, alg, w)
    end
end
function Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full(),
                    w::Option{<:ObsWeights} = nothing)::Coskewness
    return Coskewness(me, mp, alg, w)
end
"""
    factory(ske::Coskewness, w::ObsWeights) -> Coskewness

Return a new [`Coskewness`](@ref) estimator with observation weights `w` applied to the underlying mean estimator.

# Arguments

  - $(arg_dict[:ske])
  - $(arg_dict[:ow])

# Returns

  - `ske::Coskewness`: Updated estimator with weights applied.

# Examples

```jldoctest
julia> ske = Coskewness();

julia> factory(ske, StatsBase.Weights([0.2, 0.3, 0.5]))
Coskewness
   me ┼ SimpleExpectedReturns
      │   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
   mp ┼ DenoiseDetoneAlgMatrixProcessing
      │     pdm ┼ Posdef
      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      dn ┼ nothing
      │      dt ┼ nothing
      │     alg ┼ nothing
      │   order ┴ DenoiseDetoneAlg()
  alg ┼ Full()
    w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
```

# Related

  - [`Coskewness`](@ref)
  - [`factory`](@ref)
"""
function factory(ske::Coskewness, w::ObsWeights)::Coskewness
    return Coskewness(; me = factory(ske.me, w), mp = ske.mp, alg = ske.alg, w = w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the coskewness estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:ske])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:skev])

# Related

  - [`Coskewness`](@ref)
"""
function moment_view(ske::Coskewness, i)::Coskewness
    return Coskewness(; me = moment_view(ske.me, i), mp = ske.mp, alg = ske.alg, w = ske.w)
end
"""
    negative_spectral_coskewness(cskew::MatNum, X::MatNum,
                 mp::AbstractMatrixProcessingEstimator)

Internal helper for coskewness matrix processing.

`negative_spectral_coskewness` processes the coskewness tensor by applying the matrix processing estimator to each block, then projects the result using eigenvalue decomposition and clamps negative values. Used internally for robust coskewness estimation.

# Arguments

  - `cskew`: Coskewness tensor (flattened or block matrix).
  - `X`: Data matrix (observations × assets).
  - `mp`: Matrix processing estimator.

# Returns

  - `V::Matrix{<:Number}`: Processed coskewness matrix.

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`matrix_processing!`](@ref)
  - [`coskewness`](@ref)
"""
function negative_spectral_coskewness(cskew::MatNum, X::MatNum,
                                      mp::AbstractMatrixProcessingEstimator)
    N = size(cskew, 1)
    V = zeros(eltype(cskew), N, N)
    for i in 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew, :, j:k)
        vals, vecs = LinearAlgebra.eigen(coskew_jk)
        if isa(eltype(vals), Number)
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
    _coskewness(Y::MatNum, X::MatNum, mp::AbstractMatrixProcessingEstimator, w::Option{<:StatsBase.AbstractWeights}) -> MatNum

Internal helper for coskewness computation.

`_coskewness` computes the coskewness tensor and applies matrix processing. Used internally by coskewness estimators.

# Mathematical definition

Let ``\\mathbf{Y}`` be the ``T \\times N`` matrix of demeaned returns. Define ``\\boldsymbol{z}_t = \\boldsymbol{y}_t \\otimes \\boldsymbol{y}_t`` (Kronecker-style element-wise product). The ``N \\times N^2`` coskewness tensor is:

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

  - ``\\hat{\\mathbf{S}}``: ``N \\times N^2`` coskewness tensor.
  - $(math_dict[:T])
  - ``\\mathbf{Y}``: ``T \\times N`` matrix of demeaned returns.
  - ``\\mathbf{Z}``: Row-wise outer product expansion matrix.
  - ``\\boldsymbol{y}_t``: Demeaned return vector at time ``t``.
  - ``\\boldsymbol{w}``: Observation weights vector ``T \\times 1``.
  - ``w_t``: Observation weight at time ``t``.
  - ``\\otimes``: Kronecker product.
  - ``\\odot``: Element-wise (row-wise broadcast) multiplication.

# Arguments

  - `Y`: Centered data vector (e.g., `X .- mean`).
  - `X`: Data matrix (observations × assets).
  - `mp`: Matrix processing estimator.
  - `w`: Optional observation weights.

# Returns

  - `cskew::Matrix{<:Number}`: Coskewness tensor.
  - `V::Matrix{<:Number}`: Processed coskewness matrix.

# Related

  - [`Coskewness`](@ref)
  - [`negative_spectral_coskewness`](@ref)
  - [`coskewness`](@ref)
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

Compute the full coskewness tensor and processed matrix for a dataset. Observation weights in `ske.w` are applied if set. For `Full`, it uses all centered data; for `Semi`, it uses only negative deviations. If the estimator is `nothing`, returns `(nothing, nothing)`.

# Arguments

  - `ske`: Coskewness estimator.

      + `ske::Coskewness{<:Any, <:Any, <:Full}`: Coskewness estimator with [`Full`](@ref) moment algorithm.
      + `ske::Coskewness{<:Any, <:Any, <:Semi}`: Coskewness estimator with [`Semi`](@ref) moment algorithm.
      + `ske::Nothing`: No-op, returns `(nothing, nothing)`.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `mean`: Optional mean vector. If not provided, computed using the estimator's mean estimator.

  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `cskew::Matrix{<:Number}`: Coskewness tensor (observations × assets^2).
  - `V::Matrix{<:Number}`: Processed coskewness matrix (assets × assets).

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
  - [`negative_spectral_coskewness`](@ref)
"""
function coskewness(ske::Coskewness{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    w = get_observation_weights(ske.w, X; dims = 1, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    Y = X .- mu
    return _coskewness(Y, X, ske.mp, w)
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:Semi}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    w = get_observation_weights(ske.w, X; dims = 1, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    Y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(Y, X, ske.mp, w)
end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end

export Coskewness, coskewness
