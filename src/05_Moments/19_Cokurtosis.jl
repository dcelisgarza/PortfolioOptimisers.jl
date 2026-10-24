"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cokurtosis estimators.

A subtype computes the square cokurtosis matrix of a returns matrix.

# Interfaces

To add a cokurtosis estimator, subtype `CokurtosisEstimator`, keep its parameters and its observation weights in the fields of the struct, and add these methods:

## Cokurtosis

  - `PortfolioOptimisers.cokurtosis(kte::CokurtosisEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...) -> MatNum`: Returns the square cokurtosis matrix.

### Arguments

  - $(arg_dict[:kte])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments.

### Returns

  - $(ret_dict[:ckurt])

## Factory

  - `PortfolioOptimisers.factory(kte::CokurtosisEstimator, w::PortfolioOptimisers.ObsWeights) -> CokurtosisEstimator`: Returns a copy of the estimator that holds the new observation weights.

### Arguments

  - $(arg_dict[:kte])
  - $(arg_dict[:ow])

### Returns

  - `kte::CokurtosisEstimator`: A new cokurtosis estimator of the same type, with the new weights.

## View

  - `PortfolioOptimisers.port_opt_view(kte::CokurtosisEstimator, i) -> CokurtosisEstimator`: Returns the estimator for the assets that `i` selects.

### Arguments

  - $(arg_dict[:kte])
  - `i`: The index or the indices of the assets to keep.

### Returns

  - $(ret_dict[:ktev])

## Factor prior

[`HighOrderFactorPriorEstimator`](@ref) reads two more verbs of its `kte`:

  - `PortfolioOptimisers.cokurtosis_processor(kte::CokurtosisEstimator) -> Option{<:AbstractMatrixProcessingEstimator}`: The matrix processing estimator of the lifted cokurtosis, or `nothing` to skip the processing and the repair. The fallback refuses the type by name, and the prior calls it at construction under every `rsd`. See [`cokurtosis_processor`](@ref).
  - `PortfolioOptimisers.cokurtosis_expected_returns(kte::CokurtosisEstimator) -> AbstractExpectedReturnsEstimator`: The expected returns estimator of the residual moments. It is necessary under `rsd = true`. The fallback refuses the type by name, and the prior calls it at construction. See [`cokurtosis_expected_returns`](@ref).

# Examples

The example below defines a minimal cokurtosis estimator.

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
  - [`cokurtosis_processor`](@ref)
  - [`cokurtosis_expected_returns`](@ref)

# References

  - $(ref_dict[:pkurt]) Section 2, Equation 4, and Appendix A.
  - $(ref_dict[:jondeau2004]) Section 2.3.
"""
abstract type CokurtosisEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the square cokurtosis matrix of a returns matrix.

`Cokurtosis` holds a mean estimator, a matrix processing estimator and a moment algorithm. [`cokurtosis`](@ref) returns one `assets² × assets²` matrix. That matrix is the stacked fourth comoment ``\\mathbf{\\Sigma}_{4}`` of Section 2 of [pkurt](@cite) and not the `assets × assets³` tensor ``\\mathbf{M}_{4}`` of Section 2.3 of [jondeau2004](@cite).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Cokurtosis(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        alg::AbstractMomentAlgorithm = FullMoment(),
        w::Option{<:ObsWeights} = nothing,
        cvg::Option{<:CoveragePolicy} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> Cokurtosis

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
      w ┼ nothing
  cache ┴ nothing
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

  - $(ref_dict[:pkurt]) Section 2, Equation 4, and Appendix A.
  - $(ref_dict[:jondeau2004]) Section 2.3.
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
    """
    $(field_dict[:cvg])
    """
    cvg
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function Cokurtosis(me::AbstractExpectedReturnsEstimator,
                        mp::AbstractMatrixProcessingEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights}, cvg::Option{<:CoveragePolicy},
                        cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(mp), typeof(alg), typeof(w), typeof(cvg),
                   typeof(cache)}(me, mp, alg, w, cvg, cache)
    end
end
function Cokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                    alg::AbstractMomentAlgorithm = FullMoment(),
                    w::Option{<:ObsWeights} = nothing,
                    cvg::Option{<:CoveragePolicy} = nothing,
                    cache::Option{<:AbstractPartialFitState} = nothing)::Cokurtosis
    return Cokurtosis(me, mp, alg, w, cvg, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`Cokurtosis`](@ref), and leaves out `cvg` when it is `nothing`.

A `cvg` of `nothing` selects the reduce-and-expand path, which is the default of every estimator. A line for it gives the reader no data, and it changes the output of every doctest that shows the estimator. A coverage policy that is set is configuration, so it prints.

# Arguments

  - `kte`: The cokurtosis estimator. The method reads its `cvg` field alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:me, :mp, :alg, :w, :cache)` with no policy and `(:me, :mp, :alg, :w, :cvg, :cache)` with one.

# Related

  - [`Cokurtosis`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`show_fields`](@ref)
"""
function show_fields(kte::Cokurtosis)
    return if isnothing(kte.cvg)
        (:me, :mp, :alg, :w, :cache)
    else
        (:me, :mp, :alg, :w, :cvg, :cache)
    end
end
"""
    _cokurtosis(X::MatNum, mp::AbstractMatrixProcessingEstimator, w::Option{<:ObsWeights}) -> MatNum

Builds the square cokurtosis matrix of a deviation matrix.

`_cokurtosis` returns the ``N^{2} \\times N^{2}`` matrix. The matrix processing estimator repairs it in place before the return.

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

  - ``\\hat{\\mathbf{K}}``: The ``N^{2} \\times N^{2}`` square cokurtosis matrix. Its entry ``\\hat{\\mathbf{K}}_{(i-1)N+j,\\,(k-1)N+l}`` is the fourth comoment of the deviations of the assets ``i``, ``j``, ``k`` and ``l``, so the matrix is symmetric. This is ``\\mathbf{\\Sigma}_{4}`` of Section 2 of [pkurt](@cite), which Appendix A of that work computes with the formula above. It is not ``\\mathbf{M}_{4}`` of Section 2.3 of [jondeau2004](@cite), which is ``N \\times N^{3}``, and the library never builds ``\\mathbf{M}_{4}``. The weighted form is an addition of this library.
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

# Algorithm

 1. Build `o`, the ``1 \\times N`` row of ones.
 2. Build `z`, the pairwise expansion `kron(o, X) ⊙ kron(X, o)`. Its column `(i - 1) * N + j` is the element-wise product of the columns `i` and `j` of `X`.
 3. Without weights, form `ckurt` as `transpose(z) * z / T`.
 4. With weights, form `ckurt` as `transpose(w .* z) * z / sum(w)`. The weights multiply the left factor alone, so each summand carries one weight and not four.
 5. Run [`matrix_processing!`](@ref) on `ckurt` in place, and return it.

# Arguments

  - `X`: The deviation matrix (observations × assets). The caller centres it.
  - `mp`: The matrix processing estimator.
  - `w`: The optional observation weights. The unweighted method takes `nothing` through its `args...`.

# Returns

  - $(ret_dict[:ckurt]) `mp` processes it in place.

# Related

  - [`Cokurtosis`](@ref)
  - [`matrix_processing!`](@ref)
  - [`cokurtosis`](@ref)

# References

  - $(ref_dict[:pkurt]) Section 2, Equation 4, and Appendix A.
  - $(ref_dict[:jondeau2004]) Section 2.3.
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

Computes the square cokurtosis matrix of a dataset.

The method centres the data with `kte.me`, and it repairs the result with `kte.mp`. It applies the observation weights of `kte.w` when they are set. [`FullMoment`](@ref) takes the centred returns, and [`SemiMoment`](@ref) clips every positive deviation to zero. When the estimator is `nothing`, the method returns `nothing`.

`kte.w` weights the whole estimate, so it applies to the centre and to the deviations. When `mean` is `nothing` and `kte.w` is not, the method sends `kte.me` through [`factory`](@ref) with `kte.w`. So `kte.w` replaces the weights that `kte.me` holds. Pass `mean` for a centre that `kte.w` does not describe.

# Algorithm

The method passes its arguments to [`coverage_cokurtosis`](@ref), which selects an arm by dispatch on `kte.cvg`. When `kte.cvg` is `nothing`, the arm does these steps.

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`.
 2. Resolve the observation weights `w` from `kte.w` with [`get_observation_weights`](@ref).
 3. Resolve the centre `mu` from `kte.me` and `kte.w` with [`weighted_centre`](@ref), which reads `mean` when the caller gave one.
 4. Replace `X` with the deviation matrix. [`FullMoment`](@ref) takes `X .- mu`, and [`SemiMoment`](@ref) takes `min.(X .- mu, 0)`.
 5. Delegate to [`_cokurtosis`](@ref) with the deviation matrix, `kte.mp` and `w`, and return the matrix it returns.

# Arguments

  - `kte`: The cokurtosis estimator.

      + `kte::Cokurtosis{<:Any, <:Any, <:FullMoment}`: A cokurtosis estimator with the [`FullMoment`](@ref) moment algorithm.
      + `kte::Cokurtosis{<:Any, <:Any, <:SemiMoment}`: A cokurtosis estimator with the [`SemiMoment`](@ref) moment algorithm.
      + `kte::Nothing`: The method returns `nothing`.

  - `X`: The data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `mean`: The optional mean vector. When it is `nothing`, the method computes the centre with `kte.me` and `kte.w`. The available-case arm refuses a mean vector.

  - `active_mask`: The active mask of the Asset Panel over the window, or `nothing`. [`coverage_cokurtosis`](@ref) states how each arm reads it.

  - `kwargs...`: More keyword arguments. When `kte.cvg` is `nothing`, the method passes them to [`get_observation_weights`](@ref) and to [`weighted_centre`](@ref).

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
                    mean = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
    return coverage_cokurtosis(kte, kte.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:SemiMoment}, X::MatNum; dims::Int = 1,
                    mean = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                    kwargs...)
    return coverage_cokurtosis(kte, kte.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
"""
    coverage_cokurtosis(kte, cvg, X; dims::Int = 1, mean = nothing,
                        active_mask = nothing, kwargs...) -> MatNum

Computes the square cokurtosis matrix with the arm that the coverage policy of the estimator selects.

The caller passes the `cvg` field of the estimator as the second argument. So dispatch on the policy selects the arm, and no branch reads the value of the policy. [`coverage_covariance`](@ref) selects the arm of a covariance in the same way. The moment algorithm of `kte` sets the deviations from the mean of each asset. [`FullMoment`](@ref) keeps them whole, and [`SemiMoment`](@ref) sets each positive deviation to zero.

# Arguments

  - `kte`: The cokurtosis estimator.
  - `cvg`: The coverage policy of the estimator, which selects the arm.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: The optional mean vector. The available-case arm refuses one.
  - `active_mask`: The active mask of the Asset Panel over the window, or `nothing`. The Coverage Universe arm refuses an inactive cell, as it refuses a non-finite return. The available-case arm leaves the cell out of each comoment of its asset.
  - `kwargs...`: More keyword arguments. The Coverage Universe arm passes them to [`get_observation_weights`](@ref) and to [`weighted_centre`](@ref).

# Returns

  - $(ret_dict[:ckurt])

# Related

  - [`Cokurtosis`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_covariance`](@ref)
  - [`coverage_comoment_block`](@ref)
"""
function coverage_cokurtosis end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the [`FullMoment`](@ref) arm of [`coverage_cokurtosis`](@ref) for an estimator with no coverage policy.

This is the Coverage Universe arm. It centres the whole window on one vector and delegates to [`_cokurtosis`](@ref). It refuses an inactive cell of `active_mask` with [`assert_finite_sample`](@ref), because a plain estimator reads that cell as a non-finite return.

# Related

  - [`coverage_cokurtosis`](@ref)
  - [`_cokurtosis`](@ref)
"""
function coverage_cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, ::Nothing,
                             X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X, dims_oriented(dims, active_mask))
    w = get_observation_weights(kte.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, kte.me, kte.w; dims = 1, mean = mean, kwargs...)
    X = X .- mu
    return _cokurtosis(X, kte.mp, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the [`SemiMoment`](@ref) arm of [`coverage_cokurtosis`](@ref) for an estimator with no coverage policy.

This is the Coverage Universe arm. It centres the whole window on one vector, sets each positive deviation to zero, and delegates to [`_cokurtosis`](@ref). It refuses an inactive cell of `active_mask` with [`assert_finite_sample`](@ref), because a plain estimator reads that cell as a non-finite return.

# Related

  - [`coverage_cokurtosis`](@ref)
  - [`_cokurtosis`](@ref)
"""
function coverage_cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:SemiMoment}, ::Nothing,
                             X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X, dims_oriented(dims, active_mask))
    w = get_observation_weights(kte.w, X; dims = 1, kwargs...)
    mu = weighted_centre(X, kte.me, kte.w; dims = 1, mean = mean, kwargs...)
    X = min.(X .- mu, zero(eltype(X)))
    return _cokurtosis(X, kte.mp, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the available-case arm of [`coverage_cokurtosis`](@ref) for an estimator with a [`CoveragePolicy`](@ref).

The method fits each quadruple on the observations at which its four assets are all finite and active. Each cell has its own denominator.

The centre of each asset is the available-case mean of that asset, over its own finite and active observations. It is not a mean over the observations of the quadruple. [`coverage_coskewness`](@ref) states the reason, and the method refuses an external `mean` for the same reason.

The repair reads the block alone. [`matrix_processing_block!`](@ref) finds the block from the diagonal of the matrix. The diagonal entry at the pair `(i, j)` is `E[yᵢ²yⱼ²]`, and it is finite if and only if the pair shares an observation. So the block is the set of pairs that the fit gives a value for, and the repair does not change the cells outside it.

Neither an available-case cokurtosis nor an available-case coskewness folds. An exact recursion for each cell needs the second and the third comoments over the observations of each quadruple. So the online form of an available-case cokurtosis refits a buffer through [`Online`](@ref).

# Algorithm

 1. Get the valid entries, the deviations from the available-case centre of each asset, and the mask of the admitted assets with [`coverage_comoment_block`](@ref).
 2. Take the numerator as `transpose(z) * z` and the denominator as `transpose(zc) * zc`. `z` and `zc` are the pairwise expansions of the zeroed deviations and of the valid mask.
 3. Divide with [`coverage_divide`](@ref), and frame the refused assets down both pair axes with [`coverage_refuse_comoment!`](@ref).
 4. Repair the block with [`matrix_processing_block!`](@ref).

# Related

  - [`coverage_cokurtosis`](@ref)
  - [`coverage_comoment_block`](@ref)
  - [`coverage_divide`](@ref)
  - [`matrix_processing_block!`](@ref)
"""
function coverage_cokurtosis(kte::Cokurtosis, cvg::CoveragePolicy, X::MatNum; dims::Int = 1,
                             mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    assert_dims(dims)
    assert_partial_fittable(kte.me, kte.w, "Cokurtosis")
    @argcheck(isnothing(mean),
              ArgumentError("an available-case cokurtosis centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    Xo, _, _, z, zc, cmsk = coverage_comoment_block(kte.alg, cvg, X, active_mask, dims)
    ckurt = coverage_divide(transpose(z) * z, transpose(zc) * zc, false, nothing)
    coverage_refuse_comoment!(ckurt, cmsk, Val(:kt))
    matrix_processing_block!(kte.mp, ckurt, Xo)
    return ckurt
end
function cokurtosis(::Nothing, args...; kwargs...)
    return nothing
end

export cokurtosis, Cokurtosis
public CokurtosisEstimator
