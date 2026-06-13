"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cokurtosis estimators in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing cokurtosis estimation algorithms should be subtypes of `CokurtosisEstimator`.

# Interfaces

In order to implement a new cokurtosis estimator which will work seamlessly with the library, subtype `CokurtosisEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Cokurtosis

  - `PortfolioOptimisers.cokurtosis(kte::CokurtosisEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...) -> MatNum`: Computes the cokurtosis tensor.

### Arguments

  - $(arg_dict[:kte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments.

### Returns

  - `ckurt::MatNum`: Cokurtosis tensor `features^2 × features^2`.

## Factory

  - `PortfolioOptimisers.factory(kte::CokurtosisEstimator, w::PortfolioOptimisers.ObsWeights) -> CokurtosisEstimator`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:kte])
  - $(arg_dict[:ow])

### Returns

  - `kte::CokurtosisEstimator`: New cokurtosis estimator of the same type, with the new weights applied.

## View

  - `PortfolioOptimisers.moment_view(kte::CokurtosisEstimator, i) -> CokurtosisEstimator`: Returns a view of the estimator for the `i`-th element(s).

### Arguments

  - $(arg_dict[:kte])
  - `i`: Index or indices.

### Returns

  - $(ret_dict[:ktev])

# Examples

We can create a dummy cokurtosis estimator as follows:

```jldoctest
julia> struct MyCokurtosisEstimator{T1} <: PortfolioOptimisers.CokurtosisEstimator
           w::T1
           function MyCokurtosisEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyCokurtosisEstimator(;
                                      w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyCokurtosisEstimator(w)
       end
MyCokurtosisEstimator

julia> function PortfolioOptimisers.factory(::MyCokurtosisEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
           return MyCokurtosisEstimator(; w = w)
       end

julia> function PortfolioOptimisers.moment_view(kte::MyCokurtosisEstimator, i)
           return kte
       end

julia> function PortfolioOptimisers.cokurtosis(kte::MyCokurtosisEstimator,
                                               X::PortfolioOptimisers.MatNum; dims::Int = 1,
                                               mean = nothing, kwargs...)
           N = size(X, 2)
           return zeros(N^2, N^2)
       end

julia> cokurtosis(MyCokurtosisEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
4×4 Matrix{Float64}:
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> PortfolioOptimisers.factory(MyCokurtosisEstimator(), StatsBase.Weights([1, 2, 3]))
MyCokurtosisEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`Cokurtosis`](@ref)
  - [`AbstractEstimator`](@ref)
"""
abstract type CokurtosisEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Container type for cokurtosis estimators.

`Cokurtosis` encapsulates the mean estimator, matrix processing estimator, and moment algorithm for cokurtosis estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Cokurtosis(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        alg::AbstractMomentAlgorithm = Full(),
        w::Option{<:ObsWeights} = nothing
    ) -> Cokurtosis

Keywords correspond to the struct's fields.

# Validation

  - $(val_dict[:oow])

# Examples

```jldoctest
julia> Cokurtosis()
Cokurtosis
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
  alg ┼ Full()
    w ┴ nothing
```

# Related

  - [`CokurtosisEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
@concrete struct Cokurtosis <: CokurtosisEstimator
    """
    $(field_dict[:me])
    """
    me
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
    w
    function Cokurtosis(me::AbstractExpectedReturnsEstimator,
                        mp::AbstractMatrixProcessingEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(mp), typeof(alg), typeof(w)}(me, mp, alg, w)
    end
end
function Cokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full(),
                    w::Option{<:ObsWeights} = nothing)::Cokurtosis
    return Cokurtosis(me, mp, alg, w)
end
"""
    factory(kte::Cokurtosis, w::ObsWeights) -> Cokurtosis

Return a new [`Cokurtosis`](@ref) estimator with observation weights `w` applied to the underlying mean estimator.

# Arguments

  - $(arg_dict[:kte])
  - $(arg_dict[:ow])

# Returns

  - `kte::Cokurtosis`: Updated estimator with weights applied.

# Examples

```jldoctest
julia> kte = Cokurtosis();

julia> factory(kte, StatsBase.Weights([0.2, 0.3, 0.5]))
Cokurtosis
   me ┼ SimpleExpectedReturns
      │   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
   mp ┼ MatrixProcessing
      │     pdm ┼ Posdef
      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      dn ┼ nothing
      │      dt ┼ nothing
      │     alg ┼ nothing
      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
  alg ┼ Full()
    w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
```

# Related

  - [`Cokurtosis`](@ref)
  - [`factory`](@ref)
"""
function factory(kte::Cokurtosis, w::ObsWeights)::Cokurtosis
    return Cokurtosis(; me = factory(kte.me, w), mp = kte.mp, alg = kte.alg, w = w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the cokurtosis estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:kte])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:ktev])

# Related

  - [`Cokurtosis`](@ref)
"""
function moment_view(kte::Cokurtosis, i)::Cokurtosis
    return Cokurtosis(; me = moment_view(kte.me, i), mp = kte.mp, alg = kte.alg, w = kte.w)
end
"""
    _cokurtosis(X::MatNum, mp::AbstractMatrixProcessingEstimator, w::Option{<:ObsWeights}) -> MatNum

Internal helper for cokurtosis computation.

`_cokurtosis` computes the cokurtosis tensor for the input data matrix and applies matrix processing using the specified estimator.

# Mathematical definition

Let ``\\mathbf{X}`` be the ``T \\times N`` matrix of demeaned returns. Define the ``T \\times N^2`` matrix ``\\mathbf{Z}`` with rows:

```math
\\begin{align}
\\mathbf{Z}_{t,\\cdot} &= (\\boldsymbol{1}^\\intercal \\otimes \\boldsymbol{x}_t^\\intercal) \\odot (\\boldsymbol{x}_t^\\intercal \\otimes \\boldsymbol{1}^\\intercal)\\,.
\\end{align}
```

Where:

  - ``\\mathbf{Z}_{t,\\cdot}``: ``t``-th row of the auxiliary matrix ``\\mathbf{Z}``.
  - ``\\boldsymbol{x}_t``: ``t``-th row of demeaned returns.
  - ``\\otimes``: Kronecker product.
  - ``\\odot``: Element-wise (Hadamard) product.

The ``N^2 \\times N^2`` cokurtosis tensor is:

Unweighted:

```math
\\begin{align}
\\hat{\\mathbf{K}} &= \\frac{1}{T} \\mathbf{Z}^\\intercal \\mathbf{Z}\\,.
\\end{align}
```

Weighted:

```math
\\begin{align}
\\hat{\\mathbf{K}} &= \\frac{1}{\\sum_{t=1}^{T} w_t} (\\boldsymbol{w} \\odot \\mathbf{Z})^\\intercal \\mathbf{Z}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{K}}``: ``N^2 \\times N^2`` cokurtosis tensor.
  - ``\\mathbf{Z}``: ``T \\times N^2`` auxiliary matrix of pairwise return products.
  - $(math_dict[:T])
  - ``\\boldsymbol{w}``: Observation weights vector ``T \\times 1``.
  - ``w_t``: Observation weight at time ``t``.

# Arguments

  - `X`: Data matrix (observations × assets).
  - `mp`: Matrix processing estimator.
  - `w`: Optional observation weights.

# Returns

  - `ckurt::Matrix{<:Number}`: Cokurtosis tensor after matrix processing.

# Related

  - [`Cokurtosis`](@ref)
  - [`matrix_processing!`](@ref)
  - [`cokurtosis`](@ref)
"""
function _cokurtosis(X::MatNum, mp::AbstractMatrixProcessingEstimator, args...)
    T, N = size(X)
    o = transpose(range(one(eltype(X)), one(eltype(X)); length = N))
    z = kron(o, X) ⊙ kron(X, o)
    ckurt = transpose(z) * z / T
    matrix_processing!(mp, ckurt, X)
    return ckurt
end
function _cokurtosis(X::MatNum, mp::AbstractMatrixProcessingEstimator,
                     w::StatsBase.AbstractWeights)
    o = transpose(range(one(eltype(X)), one(eltype(X)); length = size(X, 2)))
    z = kron(o, X) ⊙ kron(X, o)
    ckurt = transpose(w .* z) * z / sum(w)
    matrix_processing!(mp, ckurt, X)
    return ckurt
end
"""
    cokurtosis(kte::Option{<:Cokurtosis}, X::MatNum; dims::Int = 1,
               mean = nothing, kwargs...)

Compute the cokurtosis tensor for a dataset.

This method computes the cokurtosis tensor using the estimator's mean and matrix processing algorithm. Observation weights in `kte.w` are applied if set. For `Full`, it uses all centered data; for `Semi`, it uses only negative deviations. If the estimator is `nothing`, returns `nothing`.

# Arguments

  - `kte`: Cokurtosis estimator.

      + `kte::Cokurtosis{<:Any, <:Any, <:Full}`: Cokurtosis estimator with [`Full`](@ref) moment algorithm.
      + `kte::Cokurtosis{<:Any, <:Any, <:Semi}`: Cokurtosis estimator with [`Semi`](@ref) moment algorithm.
      + `kte::Nothing`: No-op, returns `nothing`.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `mean`: Optional mean vector. If not provided, computed using the estimator's mean estimator.

  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `ckurt::Matrix{<:Number}`: Cokurtosis tensor (assets^2 × assets^2).

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = randn(rng, 10, 2);

julia> cokurtosis(Cokurtosis(), X)
4×4 Matrix{Float64}:
  1.33947   -0.246726  -0.246726   0.493008
 -0.246726   0.493008   0.493008  -0.201444
 -0.246726   0.493008   0.493008  -0.201444
  0.493008  -0.201444  -0.201444   0.300335
```

# Related

  - [`Cokurtosis`](@ref)
  - [`_cokurtosis`](@ref)
"""
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    w = get_observation_weights(kte.w, X; dims = 1, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(kte.me, X; kwargs...) : mean
    X = X .- mu
    return _cokurtosis(X, kte.mp, w)
end
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:Semi}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    w = get_observation_weights(kte.w, X; dims = 1, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(kte.me, X; kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return _cokurtosis(X, kte.mp, w)
end
function cokurtosis(::Nothing, args...; kwargs...)
    return nothing
end

export cokurtosis, Cokurtosis
