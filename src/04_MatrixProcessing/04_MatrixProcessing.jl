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
 4. `Val{:alg}`: apply [`matrix_processing_algorithm!`](@ref) to `sigma`, under `mp.alg`, forwarding `X` and `kwargs`. This is the step that a caller extends.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a matrix processing estimator whose steps cannot be run from the shape of the sample alone.

Three of the four steps of [`MatrixProcessing`](@ref) never read the sample: `pdm` and `dt` read `sigma` alone, and `dn` reads `size(X)` and nothing else. The fourth, `alg`, is a step that a caller extends, and it is handed `X` whole, so nothing here can say what it reads. An incremental fit keeps a moment and a count rather than the observations, so it can answer the first three and cannot answer the fourth.

The refusal is an `ArgumentError` that names the type of `alg`, and it names the two routes that do carry the observations: [`Online`](@ref), which buffers them for the estimator itself, and a prior that carries them for a member of its own.

# Arguments

  - `mp`: The matrix processing estimator whose steps are checked.

# Validation

  - `mp.alg` is `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

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
"""
    matrix_processing!(mp::MatrixProcessing, sigma::MatNum, T::Integer, N::Integer;
                       kwargs...) -> MatNum

Applies matrix processing to `sigma` in-place, from the **shape** of the sample rather than from the sample.

The arm of the pipeline that an incremental fit reaches when the estimate is made from the state: a fold keeps a moment and a count, so the observations the matrix arm reads no longer exist, while the one number that arm takes off them — the effective sample ratio `T / N` of the denoising step — is exactly the count the state carries. The substitution is therefore not an approximation, and [`observation_count`](@ref) is where `T` comes from.

`alg` is the one step with no shape substitute, and [`assert_shape_only_matrix_processing`](@ref) refuses it with an `ArgumentError` that names its type before any step runs.

# Algorithm

 1. Refuse an `mp` carrying a sample-reading `alg`.
 2. Run each step of `mp.order` in turn, through the shape methods of [`matrix_processing_step!`](@ref).

# Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - `T`: Number of observations the estimate was fitted over, `NaN` rows included.
  - `N`: Number of assets.
  - `kwargs...`: Additional keyword arguments passed to the steps.

# Validation

  - `mp.alg` is `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is modified in-place.

# Related

  - [`matrix_processing!`](@ref)
  - [`assert_shape_only_matrix_processing`](@ref)
  - [`observation_count`](@ref)
  - [`matrix_processing_step!`](@ref)
"""
function matrix_processing!(mp::MatrixProcessing, sigma::MatNum, T::Integer, N::Integer;
                            kwargs...)
    assert_shape_only_matrix_processing(mp)
    for step in mp.order
        matrix_processing_step!(Val(step), mp, sigma, T, N; kwargs...)
    end
    return sigma
end
"""
    matrix_processing_step!(::Val{step}, mp::MatrixProcessing, sigma::MatNum, T::Integer,
                            N::Integer; kwargs...) -> MatNum

Applies a single named matrix processing step to `sigma` in-place, from the shape of the sample.

The shape twin of the matrix methods of [`matrix_processing_step!`](@ref), one method per step, reached only through the shape arm of [`matrix_processing!`](@ref). `pdm` and `dt` read `sigma` alone, so they are the matrix methods verbatim; `dn` takes the ratio `T / N` it would otherwise read off `size(X)`; and `alg` is a no-op, because the arm refuses a non-`nothing` one before the loop starts.

# Arguments

  - `::Val{step}`: The processing step to apply, named by a symbol, as in the matrix methods.
  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - `T`: Number of observations the estimate was fitted over.
  - `N`: Number of assets.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is modified in-place.

# Related

  - [`matrix_processing!`](@ref)
  - [`matrix_processing_step!`](@ref)
  - [`assert_shape_only_matrix_processing`](@ref)
"""
function matrix_processing_step!(::Val{:pdm}, mp::MatrixProcessing, sigma::MatNum,
                                 ::Integer, ::Integer; kwargs...)
    return posdef!(mp.pdm, sigma)
end
function matrix_processing_step!(::Val{:dn}, mp::MatrixProcessing, sigma::MatNum,
                                 T::Integer, N::Integer; kwargs...)
    return denoise!(mp.dn, sigma, T / N)
end
function matrix_processing_step!(::Val{:dt}, mp::MatrixProcessing, sigma::MatNum, ::Integer,
                                 ::Integer; kwargs...)
    return detone!(mp.dt, sigma)
end
function matrix_processing_step!(::Val{:alg}, mp::MatrixProcessing, sigma::MatNum,
                                 ::Integer, ::Integer; kwargs...)
    return matrix_processing_algorithm!(mp.alg, sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Applies matrix processing to the finite block of `sigma`, from the shape of the sample rather than from the sample.

The shape twin of [`matrix_processing_block!`](@ref), and the arm an incremental fit takes when the estimate is made from the state. It is the matrix method's body with one substitution: where that one cuts the columns of `X` to the block, this one cuts the **count** of them, because the only thing the steps read off those columns is how many there are.

An estimate made from the state reaches this arm rather than the plain one for the reason the [`AssetPanel`](@ref) methods do: an estimator fitted over a changing universe answers `NaN` for an asset outside the Coverage Universe, and a positive-definite repair over a frame carrying one meets a LAPACK refusal rather than a named error. A complete matrix has no frame, and the body then runs the plain arm over the whole of it after the refusal, which names a non-finite off-diagonal entry that LAPACK would refuse without a name.

# Algorithm

 1. Read the finite rows off the diagonal of `sigma`, and run the plain arm where no asset is finite.
 2. Refuse a non-finite entry inside the finite rows with an `IsNonFiniteError`, as the matrix method does.
 3. Set the rows and columns of the zero variances to zero with [`zero_variance_rows!`](@ref), and take the block as the finite rows whose variance is not zero. A zero variance has no correlation, so no step reads it.
 4. Run the plain arm where every asset is in the block, and return where no asset is.
 5. Otherwise process the block with the block's own asset count, and write it back.

# Arguments

  - $(arg_dict[:omp])
  - $(arg_dict[:sigrho])
  - `T`: Number of observations the estimate was fitted over.
  - `N`: Number of assets the matrix describes.
  - `kwargs...`: Additional keyword arguments passed to the steps.

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is modified in-place.

# Related

  - [`matrix_processing_block!`](@ref)
  - [`matrix_processing!`](@ref)
  - [`assert_finite_block`](@ref)
  - [`zero_variance_rows!`](@ref)
"""
function matrix_processing_block!(mp::Option{<:AbstractMatrixProcessingEstimator},
                                  sigma::MatNum, T::Integer, N::Integer; kwargs...)
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

"""
    matrix_processing_block!(mp::Option{<:AbstractMatrixProcessingEstimator},
                             sigma::MatNum, X::MatNum, args...; kwargs...) -> MatNum

Repair the finite block of a covariance-like frame in place, and leave the frame around it alone.

A composite estimator that forwards an Asset Panel to the estimator it wraps can get a **frame** back: a matrix whose rows and columns outside the Coverage Universe are `NaN`. The repair has no answer for a `NaN`, so it runs on the finite block alone, and the frame around the block is written back unchanged.

The block is derived from the diagonal, exactly as [`investable_mask`](@ref) derives the Investable Mask from it, and it holds the variances that are finite and not zero. A zero variance belongs to a constant variable: it has no correlation, and a positive semidefinite matrix gives it a zero row and column. So its row and column are set to zero and kept out of the block, and no step, the denoising and the detoning included, reads them. The result is then positive semidefinite and not positive definite. An off-diagonal `NaN` **inside** the finite block is refused with an `IsNonFiniteError`, and is not peeled away here. A [`CoveragePolicy`](@ref) peels the assets of such a pair at admission, with the rule in its `peel` field, so its covariance reaches this step with no such cell. Under [`NoPeel`](@ref) the cell stays, and this refusal is the one the caller meets. A peel here would remove an asset from the covariance and leave it in the Coverage Universe, so two consumers would read two universes.

The block is the whole matrix when every diagonal entry is finite, and **the refusal covers that case too**: a complete diagonal is what an available-case pair with an empty intersection has, so a short-circuit past the refusal would send exactly the case the message was written for to LAPACK. A matrix with no finite diagonal is the one case that still goes straight to the plain repair, because it has no block to refuse anything inside.

The bare [`matrix_processing!`](@ref) and [`posdef!`](@ref) are unchanged, so a `NaN` that reaches a plain path is still refused there.

# Algorithm

 1. Take the finite rows `fin` as `isfinite.(diag(sigma))`.
 2. Run the ordinary repair and return where `fin` holds no `true`. A matrix with no finite diagonal has no block to repair, so it belongs to the plain repair and meets its refusal.
 3. Refuse a non-finite entry inside the finite rows with an `IsNonFiniteError`.
 4. Set the rows and columns of the zero variances to zero with [`zero_variance_rows!`](@ref), and take the block `blk` as the rows of `fin` whose variance is not zero.
 5. Run the ordinary repair on `sigma` itself and return where `blk` holds no `false`. A complete matrix has no frame to leave alone.
 6. Return `sigma` where `blk` holds no `true`. Every finite variance is zero, so nothing is left to repair.
 7. Copy the block out, repair the copy with [`matrix_processing!`](@ref) under the columns of `X` that the block names, write it back into `sigma`, and return `sigma`. `X` is cut only where the axis of `sigma` is the asset axis: a cokurtosis matrix is indexed by asset pairs, so its block names no column of `X` and the whole returns matrix is handed over, which is what the plain path does at that order.

# Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:sigrho])
  - $(arg_dict[:X])
  - `args...`: Additional positional arguments passed to [`matrix_processing!`](@ref).
  - `kwargs...`: Additional keyword arguments passed to [`matrix_processing!`](@ref).

# Validation

  - The block of `sigma` must be finite.

# Returns

  - `sigma::MatNum`: The input matrix, whose block was repaired in place.

# Related

  - [`matrix_processing!`](@ref)
  - [`investable_mask`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`zero_variance_rows!`](@ref)
"""
function matrix_processing_block!(mp::Option{<:AbstractMatrixProcessingEstimator},
                                  sigma::MatNum, X::MatNum, args...; kwargs...)
    fin = isfinite.(LinearAlgebra.diag(sigma))
    # A matrix with no finite diagonal has no block to repair, so it is the plain repair's and
    # its refusal is the one its caller already met.
    if !any(fin)
        matrix_processing!(mp, sigma, X, args...; kwargs...)
        return sigma
    end
    assert_finite_block(view(sigma, fin, fin))
    # A zero variance has a zero row and column and no correlation, so it leaves the block.
    blk = zero_variance_rows!(sigma, fin)
    # A complete matrix has no frame to leave alone, so the repair runs on it whole. The
    # refusal above has already covered its block, which is the matrix itself.
    if all(blk)
        matrix_processing!(mp, sigma, X, args...; kwargs...)
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
    matrix_processing!(mp, block, Xb, args...; kwargs...)
    sigma[blk, blk] = block
    return sigma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a co-moment block that carries a non-finite entry, naming the count and the first cell.

The one refusal of the block rule, shared by [`matrix_processing_block!`](@ref) and [`negative_spectral_coskewness`](@ref). A block is the set of cells among assets a fit answered for, so a non-finite cell inside it is a pair or a triple whose assets were each estimated and whose intersection was empty. The repair has no answer for it, and neither has the spectral step: Julia's LAPACK wrappers check first, so `eigen` and `nearest_cor!` both throw `ArgumentError: matrix contains Infs or NaNs`, which names neither coverage nor the cell that caused it.

# Arguments

  - `block`: The block of the answer, as a view or an array.

# Validation

  - Every entry of `block` is finite. An `IsNonFiniteError` is thrown otherwise.

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
              IsNonFiniteError("the finite block of the answer carries $(count(!isfinite, block)) non-finite entries, the first at $(findfirst(!isfinite, block)) of a block of size $(size(block)), indexed over the block's own assets and not the full universe. Every asset of the block was estimated on its own, so this cell names assets that share no observation. An axis of length `n` is the asset axis and an axis of length `n^2` is the pair axis, on which the position `c` is the asset pair `(fld(c - 1, n) + 1, mod1(c, n))`. Lower `min_coverage`, fit over a window the cells of the block share, or clear `cvg` to fall back on the Coverage Universe."))
    return nothing
end

export MatrixProcessing, matrix_processing, matrix_processing!
public AbstractMatrixProcessingEstimator, AbstractMatrixProcessingAlgorithm,
       matrix_processing_algorithm, matrix_processing_algorithm!
