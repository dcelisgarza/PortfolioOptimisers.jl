"""
$(DocStringExtensions.TYPEDEF)

A covariance estimator that applies a detoning algorithm and positive definite projection to the output of another covariance estimator. This type enables robust estimation of covariance matrices by first computing a base covariance, then applying detoning and positive definiteness corrections in sequence.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DetoneCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        dt::Detone = Detone(),
        pdm::Option{<:Posdef} = Posdef(),
    ) -> DetoneCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> DetoneCovariance()
DetoneCovariance
   ce ┼ Covariance
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    ce ┼ GeneralCovariance
      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │    w ┴ nothing
      │   alg ┴ Full()
   dt ┼ Detone
      │   pdm ┼ Posdef
      │       │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │     n ┴ Int64: 1
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Detone`](@ref)
  - [`Posdef`](@ref)
"""
@concrete struct DetoneCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:dt])"
    dt
    "$(field_dict[:pdm])"
    pdm
    function DetoneCovariance(ce::StatsBase.CovarianceEstimator, dt::Detone,
                              pdm::Option{<:Posdef})
        return new{typeof(ce), typeof(dt), typeof(pdm)}(ce, dt, pdm)
    end
end
function DetoneCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                          dt::Detone = Detone(),
                          pdm::Option{<:Posdef} = Posdef())::DetoneCovariance
    return DetoneCovariance(ce, dt, pdm)
end
"""
    factory(ce::DetoneCovariance, w::ObsWeights) -> DetoneCovariance

Return a new [`DetoneCovariance`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Related

  - [`DetoneCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::DetoneCovariance, w::ObsWeights)::DetoneCovariance
    return DetoneCovariance(; ce = factory(ce.ce, w), dt = ce.dt, pdm = ce.pdm)
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

  - [`DetoneCovariance`](@ref)
"""
function moment_view(ce::DetoneCovariance, i)::DetoneCovariance
    return DetoneCovariance(; ce = moment_view(ce.ce, i), dt = ce.dt, pdm = ce.pdm)
end
"""
    Statistics.cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)

Compute the detoned and positive definite projected covariance matrix for the data matrix `X` using the specified `DetoneCovariance` estimator.

# Arguments

  - `ce`: The `DetoneCovariance` estimator specifying the base covariance estimator, detoning algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - `sigma::MatNum`: detoned covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the covariance matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the covariance matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the detoning algorithm in `ce.dt`.
  - Returns the processed covariance matrix.

# Related

  - [`DetoneCovariance`](@ref)
  - [`cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Detone`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Statistics.cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    posdef!(ce.pdm, sigma)
    detone!(ce.dt, sigma)
    return sigma
end
"""
    Statistics.cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)

Compute the detoned and positive definite projected correlation matrix for the data matrix `X` using the specified `DetoneCovariance` estimator.

# Arguments

  - `ce`: The `DetoneCovariance` estimator specifying the base covariance estimator, detoning algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - `rho::MatNum`: detoned correlation matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the correlation matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the correlation matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the detoning algorithm in `ce.dt`.
  - Returns the processed correlation matrix.

# Related

  - [`DetoneCovariance`](@ref)
  - [`cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Detone`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = Statistics.cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    posdef!(ce.pdm, rho)
    detone!(ce.dt, rho)
    return rho
end

export DetoneCovariance
