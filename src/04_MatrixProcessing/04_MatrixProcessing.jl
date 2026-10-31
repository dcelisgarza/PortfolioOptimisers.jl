"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all matrix processing estimator types.

A type that processes a covariance or correlation matrix, for example by denoising or detoning, is a subtype of `AbstractMatrixProcessingEstimator`.

# Interfaces

To add a matrix processing estimator, subtype `AbstractMatrixProcessingEstimator`, keep its parameters in the fields of the struct, and add these methods:

  - `matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...; kwargs...) -> MatNum`: Processes `sigma` in place.
  - `matrix_processing(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...; kwargs...) -> MatNum`: Optional. Returns the processed `sigma` in a new matrix. A fallback method copies `sigma` and calls `matrix_processing!`, so add this method only when it avoids the copy.
  - `matrix_processing_posdef(mp::AbstractMatrixProcessingEstimator) -> Option{<:AbstractPosdefEstimator}`: Returns the positive definite matrix estimator with which a factor prior repairs its covariance after the matrix processing. Add this method when the estimator goes into a factor prior that adds a residual block. The fallback refuses the type by name, and the factor prior calls it at construction. Return `nothing` to skip the repair.

## Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `args...`: The positional arguments for a custom algorithm.
  - `kwargs...`: The keyword arguments for a custom algorithm.

## Returns

  - `sigma::MatNum`: The processed input matrix `sigma`.

# Examples

The example below defines a minimal matrix processing estimator.

```jldoctest
julia> struct MyMatrixProcessingEstimator <: PortfolioOptimisers.AbstractMatrixProcessingEstimator end

julia> function PortfolioOptimisers.matrix_processing!(est::MyMatrixProcessingEstimator,
                                                       sigma::PortfolioOptimisers.MatNum,
                                                       X::PortfolioOptimisers.MatNum)
           # Implement your in-place matrix processing logic here.
           println(\"Processing matrix in-place...\")
           return sigma
       end

julia> function PortfolioOptimisers.matrix_processing(est::MyMatrixProcessingEstimator,
                                                      sigma::PortfolioOptimisers.MatNum,
                                                      X::PortfolioOptimisers.MatNum)
           sigma = copy(sigma)
           println(\"Copy sigma...\")
           matrix_processing!(est, sigma, X)
           return sigma
       end

julia> function PortfolioOptimisers.matrix_processing_posdef(est::MyMatrixProcessingEstimator)
           # The estimator with which a factor prior repairs its covariance.
           return Posdef()
       end

julia> matrix_processing!(MyMatrixProcessingEstimator(), [1.0 2.0; 2.0 1.0], rand(10, 2))
Processing matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> matrix_processing(MyMatrixProcessingEstimator(), [1.0 2.0; 2.0 1.0], rand(10, 2))
Copy sigma...
Processing matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
  - [`matrix_processing_posdef`](@ref)
"""
abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all matrix processing algorithm types.

A type that implements one matrix processing algorithm is a subtype of `AbstractMatrixProcessingAlgorithm`.

# Interfaces

To add a matrix processing algorithm that `MatrixProcessing` can use, subtype `AbstractMatrixProcessingAlgorithm`, keep its parameters in the fields of the struct, and add these methods:

  - `matrix_processing_algorithm!(mpa::AbstractMatrixProcessingAlgorithm, sigma::MatNum, args...; kwargs...) -> MatNum`: Applies the algorithm to `sigma` in place.
  - `matrix_processing_algorithm(mpa::AbstractMatrixProcessingAlgorithm, sigma::MatNum, args...; kwargs...) -> MatNum`: Optional. Applies the algorithm to a copy of `sigma`, and returns the copy.

## Arguments

  - $(arg_dict[:mpa])
  - `args...`: The positional arguments for the algorithm.
  - `kwargs...`: The keyword arguments for the algorithm.

## Returns

  - `sigma::MatNum`: The input matrix `sigma` after the algorithm runs.

# Examples

The example below defines a minimal matrix processing algorithm.

```jldoctest
julia> struct MyMatrixProcessingAlgorithm <: PortfolioOptimisers.AbstractMatrixProcessingAlgorithm end

julia> function PortfolioOptimisers.matrix_processing_algorithm!(alg::MyMatrixProcessingAlgorithm,
                                                                 sigma::PortfolioOptimisers.MatNum,
                                                                 X::PortfolioOptimisers.MatNum;
                                                                 kwargs...)
           # Implement your in-place matrix processing algorithm logic here.
           println(\"Applying custom matrix processing algorithm in-place...\")
           return sigma
       end

julia> function PortfolioOptimisers.matrix_processing_algorithm(alg::MyMatrixProcessingAlgorithm,
                                                                sigma::PortfolioOptimisers.MatNum,
                                                                X::PortfolioOptimisers.MatNum;
                                                                kwargs...)
           sigma = copy(sigma)
           println(\"Copy sigma...\")
           return PortfolioOptimisers.matrix_processing_algorithm!(alg, sigma, X; kwargs...)
       end

julia> matrix_processing!(MatrixProcessing(; alg = MyMatrixProcessingAlgorithm()),
                          [1.0 2.0; 2.0 1.0], rand(10, 2))
Applying custom matrix processing algorithm in-place...
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0

julia> PortfolioOptimisers.matrix_processing_algorithm(MyMatrixProcessingAlgorithm(),
                                                       [1.0 2.0; 2.0 1.0], rand(10, 2))
Copy sigma...
Applying custom matrix processing algorithm in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
  - [`matrix_processing_algorithm`](@ref)
"""
abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end
"""
    matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)

Returns `sigma` unchanged when no matrix processing algorithm is set.

The `:alg` step of [`matrix_processing!`](@ref) calls this method when `mp.alg` is `nothing`. The method does nothing and returns `sigma`, so the pipeline skips the step.

# Arguments

  - `::Nothing`: No matrix processing algorithm is set.
  - `args...`: The positional arguments, which the method ignores.
  - `kwargs...`: The keyword arguments, which the method ignores.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, unchanged.

# Related

  - [`matrix_processing_algorithm`](@ref)
  - [`MatrixProcessing`](@ref)
"""
function matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
"""
    matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)

The out-of-place form of [`matrix_processing_algorithm!`](@ref). The `nothing` method returns `sigma` unchanged.

# Related

  - [`matrix_processing_algorithm!`](@ref)
  - [`MatrixProcessing`](@ref)
"""
function matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
"""
$(DocStringExtensions.TYPEDEF)

Runs a positive definite repair, a denoising, a detoning and a custom algorithm on a matrix, in a set order.

`MatrixProcessing` holds one estimator for each step. The field `order` sets the order in which [`matrix_processing!`](@ref) and [`matrix_processing`](@ref) run the steps. A step whose estimator is `nothing` does nothing.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MatrixProcessing(;
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        dn::Option{<:AbstractDenoiseEstimator} = nothing,
        dt::Option{<:AbstractDetoneEstimator} = nothing,
        alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
        order::Union{Tuple{Vararg{Symbol}},
                     AbstractVector{<:Symbol}} = (:pdm, :dn, :dt, :alg)
    ) -> MatrixProcessing

Keywords correspond to the struct's fields.

## Validation

  - Every symbol in `order` names a field of `MatrixProcessing` other than `order` itself, so it is one of `:pdm`, `:dn`, `:dt` and `:alg`. The constructor raises an `ArgumentError` for any other symbol.

# Examples

```jldoctest
julia> MatrixProcessing()
MatrixProcessing
    pdm ┼ Posdef
        │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
     dn ┼ nothing
     dt ┼ nothing
    alg ┼ nothing
  order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)

julia> MatrixProcessing(; dn = Denoise(), dt = Detone(; n = 2))
MatrixProcessing
    pdm ┼ Posdef
        │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
     dn ┼ Denoise
        │      pdm ┼ Posdef
        │          │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │      alg ┼ ShrunkDenoise
        │          │   alpha ┴ Float64: 0.0
        │     args ┼ Tuple{}: ()
        │   kwargs ┼ @NamedTuple{}: NamedTuple()
        │   kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
        │        m ┼ Int64: 10
        │        n ┴ Int64: 1000
     dt ┼ Detone
        │   pdm ┼ Posdef
        │       │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │     n ┴ Int64: 2
    alg ┼ nothing
  order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
```

# Related

  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)
  - [`Denoise`](@ref)
  - [`Detone`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
"""
@concrete struct MatrixProcessing <: AbstractMatrixProcessingEstimator
    """
    $(field_dict[:opdm])
    """
    pdm
    """
    $(field_dict[:odn])
    """
    dn
    """
    $(field_dict[:odt])
    """
    dt
    """
    An optional custom matrix processing algorithm.
    """
    alg
    """
    A tuple or vector of symbols that names the processing steps in the order that they run. The steps are `:pdm`, `:dn`, `:dt` and `:alg`. The constructor refuses any other symbol.
    """
    order
    function MatrixProcessing(pdm::Option{<:AbstractPosdefEstimator},
                              dn::Option{<:AbstractDenoiseEstimator},
                              dt::Option{<:AbstractDetoneEstimator},
                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                              order::Union{Tuple{Vararg{Symbol}}, AbstractVector{<:Symbol}} = (:pdm,
                                                                                               :dn,
                                                                                               :dt,
                                                                                               :alg))
        keys = setdiff(fieldnames(MatrixProcessing), (:order,))
        inorder = (k in keys for k in order)
        @argcheck(all(inorder), "Unknown field name in order: $(order[.!inorder])")
        return new{typeof(pdm), typeof(dn), typeof(dt), typeof(alg), typeof(order)}(pdm, dn,
                                                                                    dt, alg,
                                                                                    order)
    end
end
function MatrixProcessing(; pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                          dn::Option{<:AbstractDenoiseEstimator} = nothing,
                          dt::Option{<:AbstractDetoneEstimator} = nothing,
                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                          order::Union{Tuple{Vararg{Symbol}}, AbstractVector{<:Symbol}} = (:pdm,
                                                                                           :dn,
                                                                                           :dt,
                                                                                           :alg))
    return MatrixProcessing(pdm, dn, dt, alg, order)
end
"""
    matrix_processing!(
        mp::Option{<:AbstractMatrixProcessingEstimator},
        sigma::MatNum,
        X::MatNum,
        args...;
        kwargs...
    ) -> MatNum

Runs the steps of a matrix processing estimator on the covariance or correlation matrix `sigma` in place.

`mp.order` is a tuple or vector of the symbols `:pdm`, `:dn`, `:dt` and `:alg`, and it sets the steps and their order. [`matrix_processing_step!`](@ref) dispatches each step.

The method reads the number of observations as `T = size(X, 1)`, and gives it to [`sample_matrix_processing!`](@ref). A weighted moment is worth fewer observations, and its caller calls [`sample_matrix_processing!`](@ref) with the count of its weights.

# Algorithm

 1. Take the next symbol of `mp.order`. The default order is `(:pdm, :dn, :dt, :alg)`.
 2. Wrap the symbol in a `Val` and apply [`matrix_processing_step!`](@ref) to `sigma`. Each step reads the estimator that the symbol names: `:pdm` reads `mp.pdm`, `:dn` reads `mp.dn` and the effective sample ratio `T / N` taken from the shape of `X`, `:dt` reads `mp.dt`, and `:alg` reads `mp.alg`. A step whose estimator is `nothing` does nothing, so a `nothing` field skips its step and does not remove it from the order.
 3. Repeat from step 1 until no symbol of `mp.order` is left. Then return `sigma`.

The constructor checks `mp.order`, so no step of this loop names a field that `MatrixProcessing` does not have. A repeated symbol runs its step twice, and that is not an error.

# Arguments

  - $(arg_dict[:omp])
      + `::MatrixProcessing`: Runs the steps of `mp` on `sigma` in place.
      + `::Nothing`: Returns `sigma` unchanged.
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `args...`: The positional arguments for a custom algorithm.
  - `kwargs...`: The keyword arguments for a custom algorithm.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, modified in place.

# Examples

```jldoctest
julia> using StableRNGs, Statistics

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);

julia> sigma = cov(X)
5×5 Matrix{Float64}:
  0.132026     0.0022567   0.0198243    0.00359832  -0.00743829
  0.0022567    0.0514194  -0.0131242    0.004123     0.0312379
  0.0198243   -0.0131242   0.0843837   -0.0325342   -0.00609624
  0.00359832   0.004123   -0.0325342    0.0424332    0.0152574
 -0.00743829   0.0312379  -0.00609624   0.0152574    0.0926441

julia> matrix_processing!(MatrixProcessing(; dn = Denoise()), sigma, X)
5×5 Matrix{Float64}:
 0.132026  0.0        0.0        0.0        0.0
 0.0       0.0514194  0.0        0.0        0.0
 0.0       0.0        0.0843837  0.0        0.0
 0.0       0.0        0.0        0.0424332  0.0
 0.0       0.0        0.0        0.0        0.0926441

julia> sigma = cov(X)
5×5 Matrix{Float64}:
  0.132026     0.0022567   0.0198243    0.00359832  -0.00743829
  0.0022567    0.0514194  -0.0131242    0.004123     0.0312379
  0.0198243   -0.0131242   0.0843837   -0.0325342   -0.00609624
  0.00359832   0.004123   -0.0325342    0.0424332    0.0152574
 -0.00743829   0.0312379  -0.00609624   0.0152574    0.0926441

julia> matrix_processing!(MatrixProcessing(; dt = Detone()), sigma, X)
5×5 Matrix{Float64}:
 0.132026    0.0124802   0.0117303    0.0176194    0.0042142
 0.0124802   0.0514194   0.0273105   -0.0290864    0.0088165
 0.0117303   0.0273105   0.0843837   -0.00279296   0.0619156
 0.0176194  -0.0290864  -0.00279296   0.0424332   -0.0242252
 0.0042142   0.0088165   0.0619156   -0.0242252    0.0926441
```

# Related

  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`matrix_processing_step!`](@ref)
  - [`matrix_processing`](@ref)
  - [`posdef!`](@ref)
  - [`denoise!`](@ref)
  - [`detone!`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
  - [`MatNum`](@ref)
"""
function matrix_processing!(::Nothing, sigma::MatNum, args...; kwargs...)::MatNum
    return sigma
end
function matrix_processing!(mp::MatrixProcessing, sigma::MatNum, X::MatNum, args...;
                            kwargs...)
    return sample_matrix_processing!(mp, sigma, X, size(X, 1); kwargs...)
end
"""
    sample_matrix_processing!(mp::MatrixProcessing, sigma::MatNum, X::MatNum, T::Real;
                              kwargs...) -> MatNum
    sample_matrix_processing!(mp::Option{<:AbstractMatrixProcessingEstimator}, sigma::MatNum,
                              X::MatNum, T::Real; kwargs...) -> MatNum

Runs the steps of a matrix processing estimator on `sigma` in place, from the sample `X` and the effective number of observations `T`.

[`matrix_processing!`](@ref) reads `T = size(X, 1)`, and it calls this function. A weighted moment does not read every row with the same weight, so it is worth fewer observations than `X` has rows. A caller that knows the weights passes their count from [`effective_observation_count`](@ref), and the denoising step then reads the ratio `T / N` of that count. [`PortfolioOptimisersCovariance`](@ref), [`Cokurtosis`](@ref) and [`Coskewness`](@ref) do this. The other steps read `X`, as in [`matrix_processing!`](@ref).

A matrix processing estimator of the caller states only [`matrix_processing!`](@ref) on the sample, so it gets `X` and not `T`.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`MatrixProcessing`](@ref) runs each step of `mp.order` in turn, through the matrix methods of [`matrix_processing_step!`](@ref), with `X` and `T`.
 2. Every other `mp` goes to [`matrix_processing!`](@ref) with `X`, and the function ignores `T`.

# Arguments

  - $(arg_dict[:omp])
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `T`: The effective number of observations of the moment.
  - `kwargs...`: The keyword arguments for a custom algorithm.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, modified in place.

# Related

  - [`matrix_processing!`](@ref)
  - [`effective_observation_count`](@ref)
  - [`matrix_processing_step!`](@ref)
"""
function sample_matrix_processing!(mp::MatrixProcessing, sigma::MatNum, X::MatNum, T::Real;
                                   kwargs...)
    for step in mp.order
        matrix_processing_step!(Val(step), mp, sigma, X, T; kwargs...)
    end
    return sigma
end
function sample_matrix_processing!(mp::Option{<:AbstractMatrixProcessingEstimator},
                                   sigma::MatNum, X::MatNum, ::Real; kwargs...)
    return matrix_processing!(mp, sigma, X; kwargs...)
end
"""
    matrix_processing_step!(::Val{step}, mp::MatrixProcessing, sigma::MatNum, X::MatNum,
                            T::Real; kwargs...) -> MatNum

Applies one named matrix processing step to `sigma` in place. The symbol `step` selects the method.

[`sample_matrix_processing!`](@ref) calls this function once for each symbol of `mp.order`. Each symbol names one estimator field of `mp`. A `nothing` estimator does nothing, because the function that its step calls has a `nothing` method that returns the matrix.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and each method calls the function of its field.

 1. `Val{:pdm}`: apply [`posdef!`](@ref) to `sigma`, under `mp.pdm`. `X` and `T` are not read.
 2. `Val{:dn}`: read `N = size(X, 2)`, then apply [`denoise!`](@ref) to `sigma`, under `mp.dn` and the effective sample ratio `T / N`. This is the one step that reads `T`, and it reads only the number of columns of `X`.
 3. `Val{:dt}`: apply [`detone!`](@ref) to `sigma`, under `mp.dt`. `X` and `T` are not read.
 4. `Val{:alg}`: apply [`matrix_processing_algorithm!`](@ref) to `sigma`, under `mp.alg`, and pass `X` and `kwargs` to it. This is the step that a caller extends.

No method exists for any other symbol, so another symbol raises a `MethodError`. The constructor of [`MatrixProcessing`](@ref) refuses such a symbol first, so only a hand-built `Val` gets to the `MethodError`.

# Arguments

  - `::Val{step}`: The processing step to apply, named by a symbol:

      + `:pdm`: The positive definite repair under `mp.pdm`.
      + `:dn`: The denoising under `mp.dn` and the ratio `T / N`, with `N` the number of columns of `X`.
      + `:dt`: The detoning under `mp.dt`.
      + `:alg`: The optional custom algorithm `mp.alg`.
      + Any other symbol: A `MethodError`.

  - `mp`: The matrix processing estimator that holds the estimator of each step.

  - $(arg_dict[:sigrho])

  - $(arg_dict[:X])

  - `T`: The effective number of observations of the moment. It is `size(X, 1)` for an unweighted moment.

  - `kwargs...`: The keyword arguments for a custom algorithm.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, modified in place.

# Related

  - [`matrix_processing!`](@ref)
  - [`sample_matrix_processing!`](@ref)
  - [`MatrixProcessing`](@ref)
"""
function matrix_processing_step!(::Val{:pdm}, mp::MatrixProcessing, sigma::MatNum, ::MatNum,
                                 ::Real; kwargs...)
    return posdef!(mp.pdm, sigma)
end
function matrix_processing_step!(::Val{:dn}, mp::MatrixProcessing, sigma::MatNum, X::MatNum,
                                 T::Real; kwargs...)
    return denoise!(mp.dn, sigma, T / size(X, 2))
end
function matrix_processing_step!(::Val{:dt}, mp::MatrixProcessing, sigma::MatNum, ::MatNum,
                                 ::Real; kwargs...)
    return detone!(mp.dt, sigma)
end
function matrix_processing_step!(::Val{:alg}, mp::MatrixProcessing, sigma::MatNum,
                                 X::MatNum, ::Real; kwargs...)
    return matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
end
"""
    assert_shape_only_matrix_processing(mp::MatrixProcessing) -> nothing
    assert_shape_only_matrix_processing(mp::AbstractMatrixProcessingEstimator) -> nothing

Refuses a matrix processing estimator whose steps need more than the shape of the sample.

Three of the four steps of [`MatrixProcessing`](@ref) never read the sample. The steps `pdm` and `dt` read `sigma` alone, and `dn` reads only `size(X)`. The fourth step, `alg`, is a step that a caller extends. It gets the whole of `X`, so this function cannot know what it reads. An incremental fit keeps a moment and a count, and not the observations. So it can run the first three steps and cannot run the fourth.

The refusal is an `ArgumentError` that names the type of `alg`. It also names the two ways that keep the observations. [`Online`](@ref) buffers them for the estimator itself, and a prior keeps them for one of its members.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`MatrixProcessing`](@ref) passes when `mp.alg` is `nothing`.
 2. The function refuses every other subtype of [`AbstractMatrixProcessingEstimator`](@ref). Its interface states only [`matrix_processing!`](@ref) on the whole sample, so the shape of the sample cannot replace the sample.

# Arguments

  - `mp`: The matrix processing estimator whose steps the function checks.

# Validation

  - `mp` is a [`MatrixProcessing`](@ref), and `mp.alg` is `nothing`. Otherwise, the function raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`supports_shape_processing`](@ref): the same test as a `Bool`.
  - [`matrix_processing!`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`Online`](@ref)
  - [`partial_fit!`](@ref)
"""
function assert_shape_only_matrix_processing(mp::MatrixProcessing)
    @argcheck(isnothing(mp.alg),
              ArgumentError("`$(typeof(mp.alg))` is a matrix processing algorithm of your own, and it is handed the whole sample, so an incremental fit that keeps a moment and a count cannot run it. Wrap the estimator in `Online`, which buffers the observations the algorithm reads, or put it in a prior that carries them."))
    return nothing
end
function assert_shape_only_matrix_processing(mp::AbstractMatrixProcessingEstimator)
    return throw(ArgumentError("`$(nameof(typeof(mp)))` is a matrix processing estimator of your own, and its `matrix_processing!` is handed the whole sample, so an incremental fit that keeps a moment and a count cannot run it. Wrap the estimator in `Online`, which buffers the observations, or pass a `MatrixProcessing`."))
end
"""
    supports_shape_processing(mp::MatrixProcessing) -> Bool
    supports_shape_processing(mp::AbstractMatrixProcessingEstimator) -> Bool

Tells whether an incremental fit can run the steps of a matrix processing estimator from the shape of the sample alone.

This is the test of [`assert_shape_only_matrix_processing`](@ref) as a `Bool`. A composite estimator reads it to tell whether it folds.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`MatrixProcessing`](@ref) answers `true` when `mp.alg` is `nothing`.
 2. Every other subtype of [`AbstractMatrixProcessingEstimator`](@ref) answers `false`.

# Arguments

  - `mp`: The matrix processing estimator.

# Returns

  - `flag::Bool`: `true` when the steps of `mp` read the shape of the sample and no observation.

# Related

  - [`assert_shape_only_matrix_processing`](@ref)
  - [`supports_partial_fit`](@ref)
  - [`MatrixProcessing`](@ref)
"""
function supports_shape_processing(mp::MatrixProcessing)
    return isnothing(mp.alg)
end
function supports_shape_processing(::AbstractMatrixProcessingEstimator)
    return false
end
"""
    matrix_processing_reads_count(mp::MatrixProcessing) -> Bool
    matrix_processing_reads_count(mp::AbstractMatrixProcessingEstimator) -> Bool

Tells whether the shape route of a matrix processing estimator reads the number of observations `T`.

An estimate from a fold reads `T` from the state with [`observation_count`](@ref). Some states record no such count. So the estimate reads `T` only when a step needs it, and the default `mp` then works over every state. Of the steps of [`MatrixProcessing`](@ref), only the denoising step reads `T`, through the ratio `T / N`. The steps `pdm` and `dt` read the matrix alone.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`MatrixProcessing`](@ref) answers `true` when `mp.dn` is not `nothing`.
 2. Every other subtype of [`AbstractMatrixProcessingEstimator`](@ref) answers `true`, because the library cannot see what it reads.

# Arguments

  - `mp`: The matrix processing estimator.

# Returns

  - `flag::Bool`: `true` when the shape route of `mp` can read `T`.

# Related

  - [`observation_count`](@ref)
  - [`supports_shape_processing`](@ref)
  - [`matrix_processing!`](@ref)
"""
function matrix_processing_reads_count(mp::MatrixProcessing)
    return !isnothing(mp.dn)
end
function matrix_processing_reads_count(::AbstractMatrixProcessingEstimator)
    return true
end
"""
    matrix_processing_posdef(mp::MatrixProcessing) -> Option{<:AbstractPosdefEstimator}
    matrix_processing_posdef(mp::AbstractMatrixProcessingEstimator) -> Option{<:AbstractPosdefEstimator}
    matrix_processing_posdef(mp::Nothing) -> Nothing

Gives the positive definite matrix estimator with which a factor prior repairs a covariance after the matrix processing.

A factor prior processes its systematic block with [`matrix_processing!`](@ref). Then it adds the idiosyncratic block, and it repairs the sum with [`posdef!`](@ref) under the estimator that this verb gives. [`HighOrderFactorPriorEstimator`](@ref) repairs its cokurtosis in the same way.

# Interfaces

A subtype of [`AbstractMatrixProcessingEstimator`](@ref) that goes into a factor prior with a residual block adds a method of this verb. The method returns an [`AbstractPosdefEstimator`](@ref), or `nothing` to skip the repair.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`MatrixProcessing`](@ref) gives `mp.pdm`.
 2. The fallback refuses every other subtype of [`AbstractMatrixProcessingEstimator`](@ref), because it cannot tell whether the type of a caller repairs a matrix. An answer of `nothing` skips the repair, and the covariance can then be indefinite.
 3. `nothing` gives `nothing`. With no matrix processing estimator there is no repair, as [`matrix_processing!`](@ref) of `nothing` does nothing. A cokurtosis estimator whose method of [`cokurtosis_processor`](@ref) gives `nothing` has no processor.

# Arguments

  - $(arg_dict[:omp])

# Validation

  - The type of `mp` has a method of its own. Otherwise, the fallback raises an `ArgumentError` that names the type.

# Returns

  - `pdm::Option{<:AbstractPosdefEstimator}`: The estimator of the repair, or `nothing` to skip the repair.

# Related

  - [`assert_matrix_processing_posdef`](@ref): the check that each factor prior runs at construction.
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`posdef!`](@ref)
  - [`FactorPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function matrix_processing_posdef(mp::MatrixProcessing)
    return mp.pdm
end
function matrix_processing_posdef(::Nothing)
    return nothing
end
function matrix_processing_posdef(mp::AbstractMatrixProcessingEstimator)
    return throw(ArgumentError("`matrix_processing_posdef` is not defined for `$(nameof(typeof(mp)))`. A factor prior repairs its covariance after the matrix processing, under the positive definite matrix estimator of `mp`. Add a method `PortfolioOptimisers.matrix_processing_posdef(mp::$(nameof(typeof(mp))))` that returns an `AbstractPosdefEstimator`, or `nothing` to skip the repair."))
end
"""
    assert_matrix_processing_posdef(mp::Option{<:AbstractMatrixProcessingEstimator}) -> nothing

Checks at construction that a factor prior can read the repair of its matrix processing estimator.

A factor prior calls this check when its fit repairs a covariance under [`matrix_processing_posdef`](@ref). The check moves the refusal of a type with no method from the fit to the constructor.

# Algorithm

 1. Get `pdm` from [`matrix_processing_posdef`](@ref). Its fallback refuses a type with no method of its own.
 2. Check that `pdm` is `nothing` or an [`AbstractPosdefEstimator`](@ref).

# Arguments

  - $(arg_dict[:omp])

# Validation

  - The type of `mp` has a method of [`matrix_processing_posdef`](@ref). Otherwise, the fallback raises an `ArgumentError`.
  - The method returns `nothing` or an [`AbstractPosdefEstimator`](@ref). Otherwise, the function raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`matrix_processing_posdef`](@ref)
  - [`FactorPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`assert_factor_comoment_residuals`](@ref)
"""
function assert_matrix_processing_posdef(mp::Option{<:AbstractMatrixProcessingEstimator})
    pdm = matrix_processing_posdef(mp)
    @argcheck(isa(pdm, Option{<:AbstractPosdefEstimator}),
              ArgumentError("`matrix_processing_posdef(::$(nameof(typeof(mp))))` must return `nothing` or an `AbstractPosdefEstimator`. Got\npdm => $(pdm)::$(typeof(pdm))."))
    return nothing
end
"""
    matrix_processing!(mp::MatrixProcessing, sigma::MatNum, T::Option{<:Real},
                       N::Integer; kwargs...) -> MatNum
    matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::MatNum,
                       T::Option{<:Real}, N::Integer; kwargs...) -> MatNum

Applies matrix processing to `sigma` in place, from the shape of the sample and not from the sample.

An incremental fit calls this method when it makes the estimate from its state. A fold keeps a moment and a count, so the observations that the matrix method reads no longer exist. The matrix method reads one number from them, the effective sample ratio `T / N` of the denoising step, and the state keeps that count. So the substitution is exact, and `T` comes from [`effective_observation_count`](@ref). `T` is a real number, because the effective count of a weighted moment is not a whole number.

A caller passes `T = nothing` when [`matrix_processing_reads_count`](@ref) answers `false`, because then no step reads `T`. The caller then reads no count from the state, and a state that records no count works.

`alg` is the one step that the shape cannot replace. Before any step runs, [`assert_shape_only_matrix_processing`](@ref) refuses it with an `ArgumentError` that names its type. It also refuses a matrix processing estimator that a caller defines, because that interface states no shape method.

# Algorithm

 1. Refuse an `mp` that is not a [`MatrixProcessing`](@ref), and an `mp` whose `alg` is not `nothing`.
 2. Run each step of `mp.order` in turn, through the shape methods of [`matrix_processing_step!`](@ref).

# Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - `T`: The effective number of observations of the fit, with the `NaN` rows, or `nothing` when no step reads it.
  - `N`: The number of assets.
  - `kwargs...`: The keyword arguments for the steps.

# Validation

  - `mp` is a [`MatrixProcessing`](@ref), and `mp.alg` is `nothing`. Otherwise, the method raises an `ArgumentError`.
  - `T` is not `nothing` when `mp.dn` is not `nothing`. Otherwise, the denoising step raises an `ArgumentError`.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, modified in place.

# Related

  - [`matrix_processing!`](@ref)
  - [`assert_shape_only_matrix_processing`](@ref)
  - [`effective_observation_count`](@ref)
  - [`matrix_processing_reads_count`](@ref)
  - [`matrix_processing_step!`](@ref)
"""
function matrix_processing!(mp::MatrixProcessing, sigma::MatNum, T::Option{<:Real},
                            N::Integer; kwargs...)
    assert_shape_only_matrix_processing(mp)
    for step in mp.order
        matrix_processing_step!(Val(step), mp, sigma, T, N; kwargs...)
    end
    return sigma
end
function matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::MatNum,
                            T::Option{<:Real}, N::Integer; kwargs...)
    assert_shape_only_matrix_processing(mp)
    return sigma
end
"""
    matrix_processing_step!(::Val{step}, mp::MatrixProcessing, sigma::MatNum,
                            T::Option{<:Real}, N::Integer; kwargs...) -> MatNum

Applies one named matrix processing step to `sigma` in place, from the shape of the sample.

These methods are the shape forms of the matrix methods of [`matrix_processing_step!`](@ref), one method for each step. Only the shape method of [`matrix_processing!`](@ref) calls them. The steps `pdm` and `dt` read `sigma` alone, so they are the same as the matrix methods. The step `dn` takes the ratio `T / N` from the arguments and not from `size(X)`. The step `alg` does nothing, because the caller refuses an `alg` that is not `nothing` before the loop starts.

A `T` of `nothing` states that the caller read no count, because [`matrix_processing_reads_count`](@ref) answered `false`. The step `dn` then refuses a denoiser that is not `nothing`, because the ratio `T / N` has no value.

# Arguments

  - `::Val{step}`: The processing step to apply, named by a symbol, as in the matrix methods.
  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - `T`: The effective number of observations of the fit, or `nothing` when the caller read no count.
  - `N`: The number of assets.
  - `kwargs...`: The keyword arguments, which these methods ignore.

# Validation

  - The step `dn` with `T = nothing` needs `mp.dn` to be `nothing`. Otherwise, it raises an `ArgumentError`.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, modified in place.

# Related

  - [`matrix_processing!`](@ref)
  - [`matrix_processing_step!`](@ref)
  - [`assert_shape_only_matrix_processing`](@ref)
"""
function matrix_processing_step!(::Val{:pdm}, mp::MatrixProcessing, sigma::MatNum,
                                 ::Option{<:Real}, ::Integer; kwargs...)
    return posdef!(mp.pdm, sigma)
end
function matrix_processing_step!(::Val{:dn}, mp::MatrixProcessing, sigma::MatNum, T::Real,
                                 N::Integer; kwargs...)
    return denoise!(mp.dn, sigma, T / N)
end
function matrix_processing_step!(::Val{:dn}, mp::MatrixProcessing, sigma::MatNum, ::Nothing,
                                 ::Integer; kwargs...)
    @argcheck(isnothing(mp.dn),
              ArgumentError("`$(nameof(typeof(mp.dn)))` reads the ratio `T / N` of the sample, and the call passed no count `T`. Read the count with `effective_observation_count` when `matrix_processing_reads_count(mp)` is `true`."))
    return sigma
end
function matrix_processing_step!(::Val{:dt}, mp::MatrixProcessing, sigma::MatNum,
                                 ::Option{<:Real}, ::Integer; kwargs...)
    return detone!(mp.dt, sigma)
end
function matrix_processing_step!(::Val{:alg}, mp::MatrixProcessing, sigma::MatNum,
                                 ::Option{<:Real}, ::Integer; kwargs...)
    return matrix_processing_algorithm!(mp.alg, sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Applies matrix processing to the finite block of `sigma`, from the shape of the sample and not from the sample.

This is the shape form of [`matrix_processing_block!`](@ref). An incremental fit calls it when it makes the estimate from its state. Its body is the body of the matrix method with one change. The matrix method cuts the columns of `X` to the block, and this method passes the size of the block in their place, because the steps read only the number of those columns.

An estimate from the state calls this method, and not the plain one, for the same reason as the [`AssetPanel`](@ref) methods. An estimator fitted over a changing universe gives `NaN` for an asset outside the Coverage Universe. A positive definite repair of a frame that holds such an entry gets a LAPACK error that does not name the cause. A complete matrix has no frame. For it, the body first refuses a non-finite off-diagonal entry with an error that names it, and then runs the plain method on the whole matrix.

# Algorithm

 1. Read the finite rows from the diagonal of `sigma`. When no asset is finite, run the plain method and return.
 2. Refuse a non-finite entry inside the finite rows with an `IsNonFiniteError`, as the matrix method does.
 3. Set the rows and columns of the zero variances to zero with [`zero_variance_rows!`](@ref), and take the block as the finite rows whose variance is not zero. A zero variance has no correlation, so no step reads it.
 4. When every asset is in the block, run the plain method and return. When no asset is in the block, return `sigma`.
 5. Otherwise, process the block with the number of assets of the block, and write it back.

# Arguments

  - $(arg_dict[:omp])
  - $(arg_dict[:sigrho])
  - `T`: The effective number of observations of the fit, or `nothing` when no step reads it.
  - `N`: The number of assets of the matrix.
  - `kwargs...`: The keyword arguments for the steps.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, modified in place.

# Related

  - [`matrix_processing_block!`](@ref)
  - [`matrix_processing!`](@ref)
  - [`assert_finite_block`](@ref)
  - [`zero_variance_rows!`](@ref)
"""
function matrix_processing_block!(mp::Option{<:AbstractMatrixProcessingEstimator},
                                  sigma::MatNum, T::Option{<:Real}, N::Integer; kwargs...)
    fin = isfinite.(LinearAlgebra.diag(sigma))
    if !any(fin)
        matrix_processing!(mp, sigma, T, N; kwargs...)
        return sigma
    end
    assert_finite_block(view(sigma, fin, fin))
    blk = zero_variance_rows!(sigma, fin)
    if all(blk)
        matrix_processing!(mp, sigma, T, N; kwargs...)
        return sigma
    end
    if !any(blk)
        return sigma
    end
    block = sigma[blk, blk]
    matrix_processing!(mp, block, T, count(blk); kwargs...)
    sigma[blk, blk] = block
    return sigma
end
"""
    matrix_processing(
        mp::Option{<:AbstractMatrixProcessingEstimator},
        sigma::MatNum,
        X::MatNum,
        args...;
        kwargs...
    ) -> MatNum

Out-of-place version of [`matrix_processing!`](@ref).

# Algorithm

 1. Copy `sigma`.
 2. Apply [`matrix_processing!`](@ref) to the copy, and return it. The function never changes the input, and neither form changes `X`.

# Arguments

  - $(arg_dict[:omp])
      + `::AbstractMatrixProcessingEstimator`: Runs the steps of `mp` on a copy of `sigma`.
      + `::Nothing`: Returns `sigma` unchanged.
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `args...`: The positional arguments for a custom algorithm.
  - `kwargs...`: The keyword arguments for a custom algorithm.

# Returns

  - `sigma::MatNum`: A new matrix equal to the processed version of the input.

# Examples

```jldoctest
julia> using StableRNGs, Statistics

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);

julia> sigma = cov(X);

julia> Xs = matrix_processing(MatrixProcessing(; dn = Denoise()), sigma, X);

julia> size(Xs)
(5, 5)
```

# Related

  - [`matrix_processing!`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`posdef!`](@ref)
  - [`denoise!`](@ref)
  - [`detone!`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`MatNum`](@ref)
"""
function matrix_processing(::Nothing, sigma::MatNum, args...; kwargs...)::MatNum
    return sigma
end
function matrix_processing(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum,
                           args...; kwargs...)
    sigma = copy(sigma)
    matrix_processing!(mp, sigma, X, args...; kwargs...)
    return sigma
end

"""
    matrix_processing_block!(mp::Option{<:AbstractMatrixProcessingEstimator},
                             sigma::MatNum, X::MatNum, T::Real = size(X, 1);
                             kwargs...) -> MatNum

Repair the finite block of a covariance-like frame in place, and leave the frame around it alone.

A composite estimator that forwards an Asset Panel to the estimator it wraps can get a frame back. A frame is a matrix whose rows and columns outside the Coverage Universe are `NaN`. The repair has no answer for a `NaN`, so it runs on the finite block alone, and the frame around the block stays unchanged.

The function reads the block from the diagonal, as [`investable_mask`](@ref) reads the Investable Mask from it. The block holds the variances that are finite and not zero. A zero variance belongs to a constant variable. It has no correlation, and a positive semidefinite matrix gives it a zero row and column. So the function sets its row and column to zero and keeps them out of the block, and no step reads them, the denoising and the detoning included. The result is then positive semidefinite and not positive definite.

The function refuses an off-diagonal `NaN` inside the finite block with an `IsNonFiniteError`, and does not peel it at this step. A [`CoveragePolicy`](@ref) peels the assets of such a pair at admission, with the rule in its `peel` field, so its covariance gets to this step with no such cell. Under `peel = nothing` the cell stays, and the caller gets this refusal. A peel at this step removes an asset from the covariance and keeps it in the Coverage Universe. Then a consumer of the covariance reads a universe without that asset, and a consumer of the Coverage Universe reads one with it.

When every diagonal entry is finite, the block is the whole matrix, and the refusal also applies. An available-case pair with an empty intersection has a complete diagonal. So a shortcut past the refusal sends to LAPACK the case that the message is for. A matrix with no finite diagonal is the one case that goes directly to the plain repair, because it has no block to check.

The plain [`matrix_processing!`](@ref) and [`posdef!`](@ref) do not change, so they still refuse a `NaN` that gets to them.

# Algorithm

 1. Take the finite rows `fin` as `isfinite.(diag(sigma))`.
 2. When `fin` holds no `true`, run the plain repair and return. A matrix with no finite diagonal has no block to repair, so the plain repair refuses it.
 3. Refuse a non-finite entry inside the finite rows with an `IsNonFiniteError`.
 4. Set the rows and columns of the zero variances to zero with [`zero_variance_rows!`](@ref), and take the block `blk` as the rows of `fin` whose variance is not zero.
 5. When `blk` holds no `false`, run the plain repair on `sigma` itself and return. A complete matrix has no frame to keep.
 6. When `blk` holds no `true`, return `sigma`. Every finite variance is zero, so nothing is left to repair.
 7. Copy the block out, repair the copy with [`sample_matrix_processing!`](@ref) under `T` and the columns of `X` that the block names, write it back into `sigma`, and return `sigma`. The function cuts `X` only when the axis of `sigma` is the asset axis. A cokurtosis matrix has an axis of asset pairs, so its block names no column of `X`. The function then passes the whole returns matrix, as the plain path does for that matrix.

# Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `T`: The effective number of observations of the moment. It is `size(X, 1)` by default, the count of an unweighted moment. The block keeps it, because the block cuts assets and not observations.
  - `kwargs...`: The keyword arguments for [`matrix_processing!`](@ref).

# Validation

  - Every entry of the block of `sigma` is finite. Otherwise, the function raises an `IsNonFiniteError`.

# Returns

  - `sigma::MatNum`: The input matrix, whose block was repaired in place.

# Related

  - [`matrix_processing!`](@ref)
  - [`sample_matrix_processing!`](@ref)
  - [`investable_mask`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`zero_variance_rows!`](@ref)
"""
function matrix_processing_block!(mp::Option{<:AbstractMatrixProcessingEstimator},
                                  sigma::MatNum, X::MatNum, T::Real = size(X, 1); kwargs...)
    fin = isfinite.(LinearAlgebra.diag(sigma))
    # A matrix with no finite diagonal has no block to repair, so it is the plain repair's and
    # its refusal is the one its caller already met.
    if !any(fin)
        sample_matrix_processing!(mp, sigma, X, T; kwargs...)
        return sigma
    end
    assert_finite_block(view(sigma, fin, fin))
    # A zero variance has a zero row and column and no correlation, so it leaves the block.
    blk = zero_variance_rows!(sigma, fin)
    # A complete matrix has no frame to leave alone, so the repair runs on it whole. The
    # refusal above has already covered its block, which is the matrix itself.
    if all(blk)
        sample_matrix_processing!(mp, sigma, X, T; kwargs...)
        return sigma
    end
    if !any(blk)
        return sigma
    end
    block = sigma[blk, blk]
    # `X` is cut only where the axis of the matrix IS the asset axis. A cokurtosis matrix is
    # indexed by asset pairs, so its block mask is `assets²` long and names no column of `X`;
    # the plain path hands `matrix_processing!` the whole returns matrix at that order too.
    Xb = size(X, 2) == length(blk) ? X[:, blk] : X
    sample_matrix_processing!(mp, block, Xb, T; kwargs...)
    sigma[blk, blk] = block
    return sigma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a co-moment block that has a non-finite entry. The error names the count and the first cell.

[`matrix_processing_block!`](@ref) and [`negative_spectral_coskewness`](@ref) share this refusal of the block rule. A block is the set of cells of the assets that a fit gave an answer for. So a non-finite cell inside it is a pair or a triple of assets that the fit estimated one at a time, with an empty intersection of observations. The repair has no answer for that cell, and the spectral step has none. The LAPACK wrappers of Julia check first, so `eigen` and `nearest_cor!` both throw `ArgumentError: matrix contains Infs or NaNs`. That message does not name the coverage or the cell.

# Arguments

  - `block`: The block of the answer, as a view or an array.

# Validation

  - Every entry of `block` is finite. Otherwise, the function raises an `IsNonFiniteError`.

# Returns

  - `nothing`.

# Related

  - [`matrix_processing_block!`](@ref)
  - [`negative_spectral_coskewness`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
function assert_finite_block(block::AbstractArray)::Nothing
    @argcheck(all(isfinite, block),
              IsNonFiniteError("the finite block of the answer carries $(count(!isfinite, block)) non-finite entries, the first at $(findfirst(!isfinite, block)) of a block of size $(size(block)), indexed over the block's own assets and not the full universe. Every asset of the block was estimated on its own, so this cell names assets that share no observation. An axis of length `n` is the asset axis and an axis of length `n^2` is the pair axis, on which the position `c` is the asset pair `(fld(c - 1, n) + 1, mod1(c, n))`. On a covariance, set the `peel` of the `CoveragePolicy` to a rule that removes the assets of such a pair, for example the default `MinimalPeel()`. Otherwise raise `min_coverage` so that admission refuses an asset of the cell, fit over a window the cells of the block share, or clear `cvg` to fall back on the Coverage Universe."))
    return nothing
end

export MatrixProcessing, matrix_processing, matrix_processing!
public AbstractMatrixProcessingEstimator, AbstractMatrixProcessingAlgorithm,
       matrix_processing_algorithm, matrix_processing_algorithm!, matrix_processing_posdef
