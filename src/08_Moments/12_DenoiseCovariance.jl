"""
$(DocStringExtensions.TYPEDEF)

A covariance estimator that applies a denoising algorithm and positive definite projection to the output of another covariance estimator. This type enables robust estimation of covariance matrices by first computing a base covariance, then applying denoising and positive definiteness corrections in sequence.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DenoiseCovariance(;
        ce::StatsBase.CovarianceEstimator,
        dn::Denoise = Denoise(),
        pdm::Option{<:Posdef} = Posdef(),
    ) -> DenoiseCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> DenoiseCovariance()
DenoiseCovariance
   ce ┼ Covariance
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    ce ┼ GeneralCovariance
      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │    w ┴ nothing
      │   alg ┴ Full()
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
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Denoise`](@ref)
  - [`Posdef`](@ref)
"""
@concrete struct DenoiseCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    ce
    """
    $(field_dict[:dn])
    """
    dn
    """
    $(field_dict[:pdm])
    """
    pdm
    function DenoiseCovariance(ce::StatsBase.CovarianceEstimator, dn::Denoise,
                               pdm::Option{<:Posdef})
        return new{typeof(ce), typeof(dn), typeof(pdm)}(ce, dn, pdm)
    end
end
function DenoiseCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                           dn::Denoise = Denoise(),
                           pdm::Option{<:Posdef} = Posdef())::DenoiseCovariance
    return DenoiseCovariance(ce, dn, pdm)
end
"""
    factory(ce::DenoiseCovariance, w::ObsWeights) -> DenoiseCovariance

Return a new [`DenoiseCovariance`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Examples

```jldoctest
julia> ce = DenoiseCovariance();

julia> ce2 = factory(ce, StatsBase.Weights([0.2, 0.3, 0.5]));

julia> ce2.ce.me.w
3-element Weights{Float64, Float64, Vector{Float64}}:
 0.2
 0.3
 0.5
```

# Related

  - [`DenoiseCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::DenoiseCovariance, w::ObsWeights)::DenoiseCovariance
    return DenoiseCovariance(; ce = factory(ce.ce, w), dn = ce.dn, pdm = ce.pdm)
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

  - [`DenoiseCovariance`](@ref)
"""
function port_opt_view(ce::DenoiseCovariance, i, args...)::DenoiseCovariance
    return DenoiseCovariance(; ce = port_opt_view(ce.ce, i), dn = ce.dn, pdm = ce.pdm)
end
"""
    Statistics.cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)

Compute the denoised and positive definite projected covariance matrix for the data matrix `X` using the specified `DenoiseCovariance` estimator.

# Arguments

  - `ce`: The `DenoiseCovariance` estimator specifying the base covariance estimator, denoising algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - `sigma::MatNum`: denoised covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the covariance matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the covariance matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the denoising algorithm in `ce.dn` with the aspect ratio `T/N`.
  - Returns the processed covariance matrix.

# Related

  - [`DenoiseCovariance`](@ref)
  - [`cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Denoise`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Statistics.cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    T, N = size(X)
    posdef!(ce.pdm, sigma)
    denoise!(ce.dn, sigma, T / N)
    return sigma
end
"""
    Statistics.cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)

Compute the denoised and positive definite projected correlation matrix for the data matrix `X` using the specified `DenoiseCovariance` estimator.

# Arguments

  - `ce`: The `DenoiseCovariance` estimator specifying the base covariance estimator, denoising algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - `rho::MatNum`: denoised correlation matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the correlation matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the correlation matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the denoising algorithm in `ce.dn` with the aspect ratio `T/N`.
  - Returns the processed correlation matrix.

# Related

  - [`DenoiseCovariance`](@ref)
  - [`cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Denoise`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = Statistics.cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    T, N = size(X)
    posdef!(ce.pdm, rho)
    denoise!(ce.dn, rho, T / N)
    return rho
end

export DenoiseCovariance
