"""
$(DocStringExtensions.TYPEDEF)

A simple wrapper around a [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator), optional [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/), and an optional index. It uses ideas from SCIML to simplify the standard API of [`StatsBase.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.cov).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeneralCovariance(;
        ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
            corrected = true),
        w::Option{<:ObsWeights} = nothing
    ) -> GeneralCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

# Details

  - `ce` can be used to specify any subtype of `StatsBase.CovarianceEstimator`. This allows users to leverage packages such as [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl), which implement custom covariance estimators.

# Examples

```jldoctest
julia> GeneralCovariance()
GeneralCovariance
  ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w ┴ nothing

julia> GeneralCovariance(; w = StatsBase.Weights([0.1, 0.2, 0.7]))
GeneralCovariance
  ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Option`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
@concrete struct GeneralCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:oow])"
    w
    function GeneralCovariance(ce::StatsBase.CovarianceEstimator, w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(ce), typeof(w)}(ce, w)
    end
end
function GeneralCovariance(;
                           ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                          corrected = true),
                           w::Option{<:ObsWeights} = nothing)::GeneralCovariance
    return GeneralCovariance(ce, w)
end
"""
    Statistics.cov(
        ce::GeneralCovariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the covariance matrix using a [`GeneralCovariance`](@ref) estimator.

This method dispatches to the appropriate [`robust_cov`](@ref) depending on `ce.w`, which computes the covariance matrix using `ce.ce`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to [`robust_cov`](@ref).

# Returns

  - $(ret_dict[:sigma])

# Details

  - Calls [`robust_cov`](@ref) with the appropriate covariance estimator.

# Related

  - [`MatNum`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`robust_cov`](@ref)
  - [`cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    return if isnothing(ce.w)
        robust_cov(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cov(ce.ce, X, ce.w; dims = dims, mean = mean, kwargs...)
    end
end
"""
    Statistics.cor(
        ce::GeneralCovariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the correlation matrix using a [`GeneralCovariance`](@ref) estimator.

This method dispatches to the appropriate [`robust_cor`](@ref) depending on `ce.w`, which computes the correlation matrix using `ce.ce`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to [`robust_cor`](@ref).

# Returns

  - $(ret_dict[:rho])

# Details

  - Calls [`robust_cor`](@ref) with the appropriate covariance estimator.

# Related

  - [`MatNum`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`robust_cor`](@ref)
  - [`cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    w = get_observation_weights(ce.w, X; dims = dims, kwargs...)
    if isnothing(w)
        robust_cor(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cor(ce.ce, X, w; dims = dims, mean = mean, kwargs...)
    end
end
"""
    factory(
        ce::GeneralCovariance,
        w::ObsWeights
    ) -> GeneralCovariance

Return a new `GeneralCovariance` estimator with observation weights `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Examples

```jldoctest
julia> ce = GeneralCovariance()
GeneralCovariance
  ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w ┴ nothing

julia> factory(ce, StatsBase.Weights([0.1, 0.2, 0.7]))
GeneralCovariance
  ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
```

# Related

  - [`GeneralCovariance`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`factory`](@ref)
"""
function factory(ce::GeneralCovariance, w::ObsWeights)::GeneralCovariance
    return GeneralCovariance(; ce = factory(ce.ce, w), w = w)
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

  - [`GeneralCovariance`](@ref)
"""
function moment_view(ce::GeneralCovariance, i)::GeneralCovariance
    return GeneralCovariance(; ce = moment_view(ce.ce, i), w = ce.w)
end
"""
$(DocStringExtensions.TYPEDEF)

A flexible container type for covariance estimation in `PortfolioOptimisers.jl`.

`Covariance` encapsulates all components required for estimating the covariance matrix of asset returns, including the expected returns estimator for centering the data, the covariance estimator, and the moment algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Covariance(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
        alg::AbstractMomentAlgorithm = Full()
    ) -> Covariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> Covariance()
Covariance
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ GeneralCovariance
      │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │    w ┴ nothing
  alg ┴ Full()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
"""
@concrete struct Covariance <: AbstractCovarianceEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:malg])"
    alg
    function Covariance(me::AbstractExpectedReturnsEstimator,
                        ce::StatsBase.CovarianceEstimator, alg::AbstractMomentAlgorithm)
        return new{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
    end
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
                    alg::AbstractMomentAlgorithm = Full())::Covariance
    return Covariance(me, ce, alg)
end
"""
    factory(
        ce::Covariance,
        w::ObsWeights
    ) -> Covariance

Return a new `Covariance` estimator with observation weights `w` applied to both the expected returns and covariance estimators.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Details

  - Calls `factory(ce.me, w)` and `factory(ce.ce, w)` to propagate the weights to the mean and covariance estimators.
  - Preserves the moment algorithm `ce.alg` from the original estimator.
  - Enables weighted estimation for both mean and covariance in portfolio workflows.

# Examples

```jldoctest
julia> ce = Covariance()
Covariance
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ GeneralCovariance
      │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │    w ┴ nothing
  alg ┴ Full()

julia> ce_w = factory(ce, StatsBase.Weights([0.2, 0.3, 0.5]))
Covariance
   me ┼ SimpleExpectedReturns
      │   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
   ce ┼ GeneralCovariance
      │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │    w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  alg ┴ Full()
```

# Related

  - [`Covariance`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`factory`](@ref)
"""
function factory(ce::Covariance, w::ObsWeights)::Covariance
    return Covariance(; me = factory(ce.me, w), ce = factory(ce.ce, w), alg = ce.alg)
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

  - [`Covariance`](@ref)
"""
function moment_view(ce::Covariance, i)::Covariance
    return Covariance(; me = moment_view(ce.me, i), ce = moment_view(ce.ce, i),
                      alg = ce.alg)
end
"""
    Statistics.cov(
        ce::Covariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the covariance matrix using a [`Covariance`](@ref) estimator.

# Arguments

  - $(arg_dict[:ce])
      + `ce::Covariance{<:Any, <:Any, <:Full}`: Covariance estimator with [`Full`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:Semi}`: Covariance estimator with [`Semi`](@ref) moment algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
  - [`cor(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return Statistics.cov(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`Semi`](@ref) variant of [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Clips de-meaned returns to zero before computing the covariance matrix, capturing only downside co-movements.
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return Statistics.cov(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end
"""
    Statistics.cor(
        ce::Covariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the correlation matrix using a [`Covariance`](@ref) estimator.

# Arguments

  - $(arg_dict[:ce])

      + `ce::Covariance{<:Any, <:Any, <:Full}`: Covariance estimator with [`Full`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:Semi}`: Covariance estimator with [`Semi`](@ref) moment algorithm.

  - $(arg_dict[:X])

  - $(arg_dict[:dims])

  - $(arg_dict[:omean]) If not provided, computed using `ce.me`.

  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
  - [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Full}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return Statistics.cor(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`Semi`](@ref) variant of [`cor(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Clips de-meaned returns to zero before computing the correlation matrix, capturing only downside co-movements.
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return Statistics.cor(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

export GeneralCovariance, Covariance, cov, cor
