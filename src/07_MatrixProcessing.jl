"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all matrix processing estimator types.

All concrete and/or abstract types that implement matrix processing routines---such as covariance matrix cleaning, denoising, or detoning---should be subtypes of `AbstractMatrixProcessingEstimator`.

# Interfaces

In order to implement a new matrix processing estimator which will work seamlessly with the library, subtype `AbstractMatrixProcessingEstimator` with all necessary parameters as part of the struct, and implement the following methods:

  - `matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...; kwargs...) -> MatNum`: In-place processing of a covariance or correlation matrix.
  - `matrix_processing(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...; kwargs...) -> MatNum`: Optional out-of-place processing of a covariance or correlation matrix. A fallback method copies `sigma` and calls `matrix_processing!`, so it is only needed if the copy can be avoided.

## Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `args...`: Additional positional arguments passed to custom algorithms.
  - `kwargs...`: Additional keyword arguments passed to custom algorithms.

## Returns

  - `sigma::MatNum`: The processed input matrix `sigma`.

# Examples

We can create a dummy matrix processing estimator as follows:

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
"""
abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all matrix processing algorithm types.

All concrete and/or abstract types that implement a specific matrix processing algorithm should be subtypes of `AbstractMatrixProcessingAlgorithm`.

# Interfaces

In order to implement a new matrix processing algorithm that works with the current matrix processing estimator, subtype `AbstractMatrixProcessingAlgorithm`, with all necessary parameters as part of the struct, and implement the following methods:

  - `matrix_processing_algorithm!(mpa::AbstractMatrixProcessingAlgorithm, sigma::MatNum, args...; kwargs...) -> MatNum`: In-place application of a custom matrix processing algorithm.
  - `matrix_processing_algorithm(mpa::AbstractMatrixProcessingAlgorithm, sigma::MatNum, args...; kwargs...) -> MatNum`: Optional out-of-place application of a custom matrix processing algorithm.

## Arguments

  - $(arg_dict[:mpa])
  - `args...`: Additional positional arguments.
  - `kwargs...`: Additional keyword arguments.

## Returns

  - `sigma::MatNum`: The input matrix `sigma` after applying the algorithm.

# Examples

We can create a dummy matrix processing algorithm as follows:

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

No-op fallback for matrix processing algorithm routines.

These methods are called internally when no matrix processing algorithm is specified (i.e., when the algorithm argument is `nothing`). They perform no operation and return `sigma` unchanged, so the matrix processing pipeline can safely skip optional algorithmic steps.

# Arguments

  - `::Nothing`: Indicates that no matrix processing algorithm is specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is returned unchanged.

# Related

  - [`matrix_processing_algorithm`](@ref)
  - [`MatrixProcessing`](@ref)
"""
function matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
"""
    matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)

Same as [`matrix_processing_algorithm!`](@ref), but meant for returning a new matrix instead of modifying it in-place.

# Related

  - [`matrix_processing_algorithm!`](@ref)
  - [`MatrixProcessing`](@ref)
"""
function matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
"""
$(DocStringExtensions.TYPEDEF)

Configures and applies matrix processing routines.

`MatrixProcessing` encapsulates all steps required for processing covariance or correlation matrices, including positive definiteness enforcement, denoising, detoning, and optional custom matrix processing algorithms via [`matrix_processing!`](@ref) and [`matrix_processing`](@ref). This estimator allows users to build complex matrix processing pipelines tailored to their specific needs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MatrixProcessing(;
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        dn::Option{<:AbstractDenoiseEstimator} = nothing,
        dt::Option{<:AbstractDetoneEstimator} = nothing,
        alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
        order::Union{<:NTuple{N, <:Symbol} where {N},
                     <:AbstractVector{<:Symbol}} = (:pdm, :dn, :dt, :alg)
    ) -> MatrixProcessing

Keywords correspond to the struct's fields.

## Validation

  - Every symbol in `order` names a field of `MatrixProcessing` other than `order` itself, so it is one of `:pdm`, `:dn`, `:dt` and `:alg`. An unrecognised symbol throws at construction.

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

# References

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
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
    Optional custom matrix processing algorithm.
    """
    alg
    """
    A tuple or vector of symbols naming the processing steps in the order they are applied. Recognised steps are `:pdm`, `:dn`, `:dt`, and `:alg`; an unrecognised symbol errors at construction.
    """
    order
    function MatrixProcessing(pdm::Option{<:AbstractPosdefEstimator},
                              dn::Option{<:AbstractDenoiseEstimator},
                              dt::Option{<:AbstractDetoneEstimator},
                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                              order::Union{<:NTuple{N, <:Symbol} where {N},
                                           <:AbstractVector{<:Symbol}} = (:pdm, :dn, :dt,
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
                          order::Union{<:NTuple{N, <:Symbol} where {N},
                                       <:AbstractVector{<:Symbol}} = (:pdm, :dn, :dt, :alg))
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

In-place matrix processing pipeline.

This method applies a sequence of matrix processing steps to the input covariance or correlation matrix `sigma`, modifying it in-place. The steps and their order are given by `mp.order`---a tuple or vector of symbols (`:pdm`, `:dn`, `:dt`, `:alg`)---and each step is dispatched through [`matrix_processing_step!`](@ref).

# Algorithm

 1. Take the next symbol of `mp.order`. The default order is `(:pdm, :dn, :dt, :alg)`.
 2. Wrap the symbol in a `Val` and apply [`matrix_processing_step!`](@ref) to `sigma`. Each step reads the estimator that the symbol names: `:pdm` reads `mp.pdm`, `:dn` reads `mp.dn` and the effective sample ratio `T / N` taken from the shape of `X`, `:dt` reads `mp.dt`, and `:alg` reads `mp.alg`. A step whose estimator is `nothing` is a no-op, so a `nothing` field skips its step rather than removing it from the order.
 3. Repeat from step 1 until `mp.order` is exhausted, then return `sigma`.

`mp.order` is validated at construction, so no step of this loop can name a field that `MatrixProcessing` does not carry. A repeated symbol applies its step twice, which is the order's own business and not an error.

# Arguments

  - $(arg_dict[:omp])
      + `::MatrixProcessing`: The specified matrix processing estimator is applied to `X` in-place.
      + `::Nothing`: No-op.
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `args...`: Additional positional arguments passed to custom algorithms.
  - `kwargs...`: Additional keyword arguments passed to custom algorithms.

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is modified in-place.

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

# References

  - $(ref_dict[:mlp1]) Chapter 2.
  - $(ref_dict[:mpdist])
"""
function matrix_processing!(::Nothing, sigma::MatNum, args...; kwargs...)::MatNum
    return sigma
end
function matrix_processing!(mp::MatrixProcessing, sigma::MatNum, X::MatNum, args...;
                            kwargs...)
    for step in mp.order
        matrix_processing_step!(Val(step), mp, sigma, X; kwargs...)
    end
    return sigma
end
"""
    matrix_processing_step!(::Val{step}, mp::MatrixProcessing, sigma::MatNum, X::MatNum; kwargs...) -> MatNum

Apply a single named matrix processing step to `sigma` in-place, dispatching on the step symbol `step`.

This is the per-step worker that [`matrix_processing!`](@ref) calls while iterating over `mp.order`. Each recognised symbol maps to one of the estimator fields of `mp`; the override-or-skip behaviour is inherited from the underlying primitives (a `nothing` estimator is a no-op).

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and each one delegates to the primitive that its field names.

 1. `Val{:pdm}`: apply [`posdef!`](@ref) to `sigma`, under `mp.pdm`. `X` is not read.
 2. `Val{:dn}`: read `T, N = size(X)`, then apply [`denoise!`](@ref) to `sigma`, under `mp.dn` and the effective sample ratio `T / N`. This is the one step that reads `X`, and it reads only its shape.
 3. `Val{:dt}`: apply [`detone!`](@ref) to `sigma`, under `mp.dt`. `X` is not read.
 4. `Val{:alg}`: apply [`matrix_processing_algorithm!`](@ref) to `sigma`, under `mp.alg`, forwarding `X` and `kwargs`. This is the seam a caller extends.

No method is defined for any other symbol, so an unrecognised step raises a `MethodError`. The constructor of [`MatrixProcessing`](@ref) rejects such a symbol first, so the `MethodError` is reachable only through a hand-built `Val`.

# Arguments

  - `::Val{step}`: The processing step to apply, named by a symbol:

      + `:pdm`: positive definiteness enforcement using `mp.pdm`.
      + `:dn`: denoising using `mp.dn` and the ratio `T / N` derived from `X`.
      + `:dt`: detoning using `mp.dt`.
      + `:alg`: optional custom algorithm using `mp.alg`.
      + Any other symbol: MethodError.

  - `mp`: Matrix processing estimator holding the per-step estimators.

  - $(arg_dict[:sigrho])

  - $(arg_dict[:X])

  - `kwargs...`: Additional keyword arguments passed to custom algorithms.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, modified in-place.

# Related

  - [`matrix_processing!`](@ref)
  - [`MatrixProcessing`](@ref)
"""
function matrix_processing_step!(::Val{:pdm}, mp::MatrixProcessing, sigma::MatNum,
                                 X::MatNum; kwargs...)
    return posdef!(mp.pdm, sigma)
end
function matrix_processing_step!(::Val{:dn}, mp::MatrixProcessing, sigma::MatNum, X::MatNum;
                                 kwargs...)
    T, N = size(X)
    return denoise!(mp.dn, sigma, T / N)
end
function matrix_processing_step!(::Val{:dt}, mp::MatrixProcessing, sigma::MatNum, X::MatNum;
                                 kwargs...)
    return detone!(mp.dt, sigma)
end
function matrix_processing_step!(::Val{:alg}, mp::MatrixProcessing, sigma::MatNum,
                                 X::MatNum; kwargs...)
    return matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
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
 2. Apply [`matrix_processing!`](@ref) to the copy, and return it. The input is never modified, and `X` is never modified by either form.

# Arguments

  - $(arg_dict[:omp])
      + `::AbstractMatrixProcessingEstimator`: The specified processing pipeline is applied to a copy of `sigma`.
      + `::Nothing`: No-op, returns `sigma` unchanged.
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `args...`: Additional positional arguments passed to custom algorithms.
  - `kwargs...`: Additional keyword arguments passed to custom algorithms.

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

export MatrixProcessing, matrix_processing, matrix_processing!
