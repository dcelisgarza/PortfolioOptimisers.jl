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
@propagatable @concrete struct GeneralCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:oow])
    """
    @wprop w
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

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cov(GeneralCovariance(), X)
2×2 Matrix{Float64}:
 0.0001  0.0001
 0.0001  0.0001
```
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

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cor(GeneralCovariance(), X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```
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
$(DocStringExtensions.TYPEDEF)

Estimates the covariance matrix of asset returns from a centring estimator, a covariance estimator, and a moment algorithm.

`Covariance` encapsulates all components required for estimating the covariance matrix of asset returns, including the expected returns estimator for centering the data, the covariance estimator, and the moment algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Covariance(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
        alg::AbstractMomentAlgorithm = FullMoment()
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
  alg ┴ FullMoment()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
"""
@propagatable @concrete struct Covariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:malg])
    """
    alg
    function Covariance(me::AbstractExpectedReturnsEstimator,
                        ce::StatsBase.CovarianceEstimator, alg::AbstractMomentAlgorithm)
        return new{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
    end
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
                    alg::AbstractMomentAlgorithm = FullMoment())::Covariance
    return Covariance(me, ce, alg)
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

# Mathematical definition

FullMoment covariance:

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{ij} &= \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{ti} - \\hat{\\mu}_i)(r_{tj} - \\hat{\\mu}_j)\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{\\Sigma}}_{ij}``: Estimated covariance between assets ``i`` and ``j``.
  - ``r_{ti}``: Return of asset ``i`` at time ``t``.
  - ``\\hat{\\mu}_i``: Estimated mean of asset ``i``.
  - $(math_dict[:T])

SemiMoment (downside) covariance — clip de-meaned returns to zero before computing:

```math
\\begin{align}
\\tilde{r}_{tj} &= \\min(r_{tj} - \\hat{\\mu}_j,\\, 0)\\,.
\\end{align}
```

Where:

  - ``\\tilde{r}_{tj}``: Clipped de-meaned return of asset ``j`` at time ``t``.
  - ``r_{tj}``: Return of asset ``j`` at time ``t``.
  - ``\\hat{\\mu}_j``: Estimated mean of asset ``j``.

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}^{\\text{semi}}_{ij} &= \\frac{1}{T-1} \\sum_{t=1}^{T} \\tilde{r}_{ti} \\, \\tilde{r}_{tj}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{\\Sigma}}^{\\text{semi}}_{ij}``: Estimated semi-covariance between assets ``i`` and ``j``.
  - ``\\tilde{r}_{ti}``, ``\\tilde{r}_{tj}``: Clipped de-meaned returns of assets ``i`` and ``j``.
  - $(math_dict[:T])

# Arguments

  - $(arg_dict[:ce])
      + `ce::Covariance{<:Any, <:Any, <:FullMoment}`: Covariance estimator with [`FullMoment`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:SemiMoment}`: Covariance estimator with [`SemiMoment`](@ref) moment algorithm.
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
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`cor(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cov(Covariance(), X)
2×2 Matrix{Float64}:
 0.0001  0.0001
 0.0001  0.0001
```
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return Statistics.cov(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SemiMoment`](@ref) variant of [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Clips de-meaned returns to zero before computing the covariance matrix, capturing only downside co-movements.
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
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

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{P}}_{ij} &= \\frac{\\hat{\\mathbf{\\Sigma}}_{ij}}{\\hat{\\sigma}_i \\hat{\\sigma}_j}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{P}}_{ij}``: Estimated correlation between assets ``i`` and ``j``.
  - ``\\hat{\\mathbf{\\Sigma}}_{ij}``: Estimated covariance between assets ``i`` and ``j``.
  - ``\\hat{\\sigma}_i``: Estimated standard deviation of asset ``i``.

# Arguments

  - $(arg_dict[:ce])

      + `ce::Covariance{<:Any, <:Any, <:FullMoment}`: Covariance estimator with [`FullMoment`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:SemiMoment}`: Covariance estimator with [`SemiMoment`](@ref) moment algorithm.

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
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cor(Covariance(), X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return Statistics.cor(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SemiMoment`](@ref) variant of [`cor(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Clips de-meaned returns to zero before computing the correlation matrix, capturing only downside co-movements.
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return Statistics.cor(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

export GeneralCovariance, Covariance, cov, cor
