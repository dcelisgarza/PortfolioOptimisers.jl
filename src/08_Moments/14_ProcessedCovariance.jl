"""
$(DocStringExtensions.TYPEDEF)

A covariance estimator that applies a custom matrix processing algorithm and positive definite projection to the output of another covariance estimator. This type enables flexible post-processing of covariance matrices by first computing a base covariance, then applying a user-specified matrix processing algorithm and positive definiteness correction in sequence.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProcessedCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
        pdm::Option{<:Posdef} = Posdef(),
    ) -> ProcessedCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> ProcessedCovariance()
ProcessedCovariance
   ce ┼ Covariance
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    ce ┼ GeneralCovariance
      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │    w ┴ nothing
      │   alg ┴ Full()
  alg ┼ nothing
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`Posdef`](@ref)
"""
@concrete struct ProcessedCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    ce
    """
    $(field_dict[:mpa])
    """
    alg
    """
    $(field_dict[:pdm])
    """
    pdm
    function ProcessedCovariance(ce::StatsBase.CovarianceEstimator,
                                 alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                 pdm::Option{<:Posdef})
        return new{typeof(ce), typeof(alg), typeof(pdm)}(ce, alg, pdm)
    end
end
function ProcessedCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                             alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                             pdm::Option{<:Posdef} = Posdef())::ProcessedCovariance
    return ProcessedCovariance(ce, alg, pdm)
end
"""
    factory(ce::ProcessedCovariance, w::ObsWeights) -> ProcessedCovariance

Return a new [`ProcessedCovariance`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Examples

```jldoctest
julia> ce = ProcessedCovariance();

julia> ce2 = factory(ce, StatsBase.Weights([0.2, 0.3, 0.5]));

julia> ce2.ce.me.w
3-element Weights{Float64, Float64, Vector{Float64}}:
 0.2
 0.3
 0.5
```

# Related

  - [`ProcessedCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::ProcessedCovariance, w::ObsWeights)::ProcessedCovariance
    return ProcessedCovariance(; ce = factory(ce.ce, w), alg = ce.alg, pdm = ce.pdm)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the covariance estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:ce])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:cev])

# Related

  - [`ProcessedCovariance`](@ref)
"""
function moment_view(ce::ProcessedCovariance, i)::ProcessedCovariance
    return ProcessedCovariance(; ce = moment_view(ce.ce, i), alg = ce.alg, pdm = ce.pdm)
end
"""
    Statistics.cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)

Compute the processed and positive definite projected covariance matrix for the data matrix `X` using the specified `ProcessedCovariance` estimator.

# Arguments

  - `ce`: The `ProcessedCovariance` estimator specifying the base covariance estimator, matrix processing algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and processing algorithm.

# Returns

  - `sigma::MatNum`: processed covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the covariance matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the covariance matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the matrix processing algorithm in `ce.alg` to the covariance matrix.
  - Returns the processed covariance matrix.

# Related

  - [`ProcessedCovariance`](@ref)
  - [`cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Statistics.cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    posdef!(ce.pdm, sigma)
    matrix_processing_algorithm!(ce.alg, sigma, X; kwargs...)
    return sigma
end
"""
    Statistics.cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)

Compute the processed and positive definite projected correlation matrix for the data matrix `X` using the specified `ProcessedCovariance` estimator.

# Arguments

  - `ce`: The `ProcessedCovariance` estimator specifying the base covariance estimator, matrix processing algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator and processing algorithm.

# Returns

  - `rho::MatNum`: processed correlation matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the correlation matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the correlation matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the matrix processing algorithm in `ce.alg` to the correlation matrix.
  - Returns the processed correlation matrix.

# Related

  - [`ProcessedCovariance`](@ref)
  - [`cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = Statistics.cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    posdef!(ce.pdm, rho)
    matrix_processing_algorithm!(ce.alg, rho, X; kwargs...)
    return rho
end

export ProcessedCovariance
