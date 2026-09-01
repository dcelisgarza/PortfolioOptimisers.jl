"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cokurtosis estimators.

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

  - $(ret_dict[:ckurt])

## Factory

  - `PortfolioOptimisers.factory(kte::CokurtosisEstimator, w::PortfolioOptimisers.ObsWeights) -> CokurtosisEstimator`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:kte])
  - $(arg_dict[:ow])

### Returns

  - `kte::CokurtosisEstimator`: New cokurtosis estimator of the same type, with the new weights applied.

## View

  - `PortfolioOptimisers.port_opt_view(kte::CokurtosisEstimator, i) -> CokurtosisEstimator`: Returns a view of the estimator for the `i`-th element(s).

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

julia> function PortfolioOptimisers.port_opt_view(kte::MyCokurtosisEstimator, i)
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

# References

  - $(ref_dict[:cajas2025]) Section 3.1.4, Equation 3.7.
  - $(ref_dict[:pkurt])
"""
abstract type CokurtosisEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the square cokurtosis matrix of a returns matrix.

`Cokurtosis` composes a mean estimator, a matrix processing estimator and a moment algorithm. [`cokurtosis`](@ref) returns one `assets² × assets²` matrix, which is the source's stacked fourth comoment and not the `assets × assets³` tensor of the same name.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Cokurtosis(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        alg::AbstractMomentAlgorithm = FullMoment(),
        w::Option{<:ObsWeights} = nothing
    ) -> Cokurtosis

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
  alg ┼ FullMoment()
    w ┴ nothing
```

# Related

  - [`CokurtosisEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.1.4, Equation 3.7.
  - $(ref_dict[:pkurt])
"""
@propagatable @concrete struct Cokurtosis <: CokurtosisEstimator
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
    function Cokurtosis(me::AbstractExpectedReturnsEstimator,
                        mp::AbstractMatrixProcessingEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(mp), typeof(alg), typeof(w)}(me, mp, alg, w)
    end
end
function Cokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                    alg::AbstractMomentAlgorithm = FullMoment(),
                    w::Option{<:ObsWeights} = nothing)::Cokurtosis
    return Cokurtosis(me, mp, alg, w)
end
"""
    _cokurtosis(X::MatNum, mp::AbstractMatrixProcessingEstimator, w::Option{<:ObsWeights}) -> MatNum

Internal helper that builds the square cokurtosis matrix from a deviation matrix.

`_cokurtosis` returns the ``N^{2} \\times N^{2}`` matrix, which the matrix processing estimator repairs in place before the return.

# Mathematical definition

The pairwise expansion of the deviation matrix has the rows:

```math
\\begin{align}
\\mathbf{Z}_{t,\\cdot} &= (\\boldsymbol{1}^\\intercal \\otimes \\boldsymbol{y}_t^\\intercal) \\odot (\\boldsymbol{y}_t^\\intercal \\otimes \\boldsymbol{1}^\\intercal)\\,.
\\end{align}
```

The ``N^{2} \\times N^{2}`` square cokurtosis matrix is:

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

  - ``\\hat{\\mathbf{K}}``: ``N^{2} \\times N^{2}`` square cokurtosis matrix. Its entry ``\\hat{\\mathbf{K}}_{(i-1)N+j,\\,(k-1)N+l}`` is the fourth comoment of the deviations of the assets ``i``, ``j``, ``k`` and ``l``, so the matrix is symmetric. This is the source's ``\\mathbf{\\Sigma}_{4}``, not its ``\\mathbf{M}_{4}``. The latter is ``N \\times N^{3}`` and the library never builds it.
  - $(math_dict[:Y_dev])
  - $(math_dict[:y_t_dev])
  - $(math_dict[:Z_pairprod])
  - $(math_dict[:w_obs_vec])
  - $(math_dict[:w_t_moment])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - ``\\boldsymbol{1}``: ``N \\times 1`` vector of ones.
  - ``\\otimes``: Kronecker product.
  - ``\\odot``: Element-wise product. Where the operands differ in shape, it broadcasts along the row axis.

# Algorithm

 1. Build `o`, the ``1 \\times N`` row of ones.
 2. Build `z`, the pairwise expansion `kron(o, X) ⊙ kron(X, o)`. Its column `(i - 1) * N + j` is the element-wise product of the columns `i` and `j` of `X`.
 3. Without weights, form `ckurt` as `transpose(z) * z / T`.
 4. With weights, form `ckurt` as `transpose(w .* z) * z / sum(w)`. The weights multiply the left factor alone, so each summand carries one weight and not four.
 5. Run [`matrix_processing!`](@ref) on `ckurt` in place, and return it.

# Arguments

  - `X`: Deviation matrix (observations × assets), already centred by the caller.
  - `mp`: Matrix processing estimator.
  - `w`: Optional observation weights. The unweighted method takes `nothing` through its `args...`.

# Returns

  - $(ret_dict[:ckurt]) It is processed in place by `mp`.

# Related

  - [`Cokurtosis`](@ref)
  - [`matrix_processing!`](@ref)
  - [`cokurtosis`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.1.4, Equation 3.7.
  - $(ref_dict[:pkurt])
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

Compute the square cokurtosis matrix of a dataset.

This method centres the data with the estimator's mean estimator and repairs the result with its matrix processing estimator. Observation weights in `kte.w` are applied if set. [`FullMoment`](@ref) takes the centred returns, and [`SemiMoment`](@ref) clips every positive deviation to zero. If the estimator is `nothing`, returns `nothing`.

`kte.w` weights the whole estimate, so it reaches the centre as well as the deviations. When `mean` is `nothing` and `kte.w` is not, the method sends `kte.me` through [`factory`](@ref) with `kte.w`, so `kte.w` wins over the weights that `kte.me` carries. Pass `mean` for a centre that `kte.w` does not describe. ADR 0088 records the decision.

# Algorithm

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`.
 2. Resolve the observation weights `w` from `kte.w` with [`get_observation_weights`](@ref).
 3. Resolve the centre `mu` from `kte.me` and `kte.w` with [`weighted_centre`](@ref), which reads `mean` when the caller gave one.
 4. Replace `X` with the deviation matrix. [`FullMoment`](@ref) takes `X .- mu`, and [`SemiMoment`](@ref) takes `min.(X .- mu, 0)`.
 5. Delegate to [`_cokurtosis`](@ref) with the deviation matrix, `kte.mp` and `w`, and return the matrix it returns.

# Arguments

  - `kte`: Cokurtosis estimator.

      + `kte::Cokurtosis{<:Any, <:Any, <:FullMoment}`: Cokurtosis estimator with [`FullMoment`](@ref) moment algorithm.
      + `kte::Cokurtosis{<:Any, <:Any, <:SemiMoment}`: Cokurtosis estimator with [`SemiMoment`](@ref) moment algorithm.
      + `kte::Nothing`: No-op, returns `nothing`.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `mean`: Optional mean vector. If not provided, computed using the estimator's mean estimator.

  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - $(ret_dict[:ckurt])

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
  - [`weighted_centre`](@ref)
"""
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    X = dims_oriented(dims, X)
    w = get_observation_weights(kte.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, kte.me, kte.w; dims = 1, mean = mean, kwargs...)
    X = X .- mu
    return _cokurtosis(X, kte.mp, w)
end
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    X = dims_oriented(dims, X)
    w = get_observation_weights(kte.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, kte.me, kte.w; dims = 1, mean = mean, kwargs...)
    X = min.(X .- mu, zero(eltype(X)))
    return _cokurtosis(X, kte.mp, w)
end
function cokurtosis(::Nothing, args...; kwargs...)
    return nothing
end

export cokurtosis, Cokurtosis
