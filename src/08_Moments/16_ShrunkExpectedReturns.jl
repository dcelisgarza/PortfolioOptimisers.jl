"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrunk expected returns estimators.

All concrete and/or abstract types implementing shrinkage-based expected returns estimation algorithms should be subtypes of `AbstractShrunkExpectedReturnsEstimator`.

# Related

  - [`ShrunkExpectedReturns`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage algorithms for expected returns estimation.

All concrete and/or abstract types implementing specific shrinkage algorithms (e.g., James-Stein, Bayes-Stein) should be subtypes of `AbstractShrunkExpectedReturnsAlgorithm`.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsAlgorithm <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage targets used in expected returns estimation.

Concrete types implementing specific shrinkage targets (e.g., grand mean, volatility-weighted mean) should subtype `AbstractShrunkExpectedReturnsTarget`.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquaredError`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage target representing the grand mean of expected returns.

`GrandMean` computes the shrinkage target as the mean of all asset expected returns, resulting in a vector where each element is the same grand mean value. This is commonly used in shrinkage estimators to reduce estimation error by pulling individual expected returns toward the overall average.

# Constructors

    GrandMean() -> GrandMean

# Examples

```jldoctest
julia> GrandMean()
GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct GrandMean <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage target representing the volatility-weighted mean of expected returns.

`VolatilityWeighted` computes the shrinkage target as a weighted mean of expected returns, where weights are inversely proportional to asset volatility (from the inverse covariance matrix). This approach accounts for differences in asset risk when estimating the shrinkage target.

# Constructors

    VolatilityWeighted() -> VolatilityWeighted

# Examples

```jldoctest
julia> VolatilityWeighted()
VolatilityWeighted()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage target representing the mean squared error of expected returns.

`MeanSquaredError` computes the shrinkage target as the trace of the covariance matrix divided by the number of observations, resulting in a vector where each element is the same value. This target is useful for certain shrinkage estimators that minimize mean squared error.

# Constructors

    MeanSquaredError() -> MeanSquaredError

# Examples

```jldoctest
julia> MeanSquaredError()
MeanSquaredError()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct MeanSquaredError <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage algorithm implementing the James-Stein estimator for expected returns.

`JamesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean). The estimator reduces estimation error, especially in high-dimensional settings.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JamesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> JamesStein

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> JamesStein()
JamesStein
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
@concrete struct JamesStein <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    function JamesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function JamesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::JamesStein
    return JamesStein(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage algorithm implementing the Bayes-Stein estimator for expected returns.

`BayesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using Bayesian principles. This estimator is useful for reducing estimation error, especially when sample sizes are small.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BayesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> BayesStein

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BayesStein()
BayesStein
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
@concrete struct BayesStein <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    function BayesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BayesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::BayesStein
    return BayesStein(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage algorithm implementing the Bodnar-Okhrin-Parolya estimator for expected returns.

`BodnarOkhrinParolya` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using the Bodnar-Okhrin-Parolya approach. This estimator is designed for robust estimation in high-dimensional settings.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BodnarOkhrinParolya(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> BodnarOkhrinParolya

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BodnarOkhrinParolya()
BodnarOkhrinParolya
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
"""
@concrete struct BodnarOkhrinParolya <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    function BodnarOkhrinParolya(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BodnarOkhrinParolya(;
                             tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::BodnarOkhrinParolya
    return BodnarOkhrinParolya(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Container type for shrinkage-based expected returns estimators.

`ShrunkExpectedReturns` encapsulates all components required for shrinkage estimation of expected returns, including the mean estimator, covariance estimator, and shrinkage algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShrunkExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein()
    ) -> ShrunkExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `alg`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ShrunkExpectedReturns()
ShrunkExpectedReturns
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │   w ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │    w ┴ nothing
      │      │   alg ┴ FullMoment()
      │   mp ┼ MatrixProcessing
      │      │     pdm ┼ Posdef
      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │      dn ┼ nothing
      │      │      dt ┼ nothing
      │      │     alg ┼ nothing
      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
  alg ┼ JamesStein
      │   tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct ShrunkExpectedReturns <:
                               AbstractShrunkExpectedReturnsEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:me_shrink_alg])
    """
    @vprop alg
    function ShrunkExpectedReturns(me::AbstractExpectedReturnsEstimator,
                                   ce::StatsBase.CovarianceEstimator,
                                   alg::AbstractShrunkExpectedReturnsAlgorithm)
        return new{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
    end
end
function ShrunkExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())::ShrunkExpectedReturns
    return ShrunkExpectedReturns(me, ce, alg)
end
"""
    target_mean(::AbstractShrunkExpectedReturnsTarget, mu::ArrNum, sigma::MatNum, args...;
                kwargs...)

Compute the shrinkage target vector for expected returns estimation.

`target_mean` computes the target vector toward which expected returns are shrunk, based on the specified shrinkage target type. This function is used internally by shrinkage estimators such as James-Stein, Bayes-Stein, and Bodnar-Okhrin-Parolya.

# Mathematical definition

**`GrandMean`**: each target element is the grand mean of sample expected returns:

```math
\\begin{align}
b_j &= \\bar{\\mu} = \\frac{1}{N} \\sum_{i=1}^{N} \\hat{\\mu}_i, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - ``b_j``: ``j``-th element of the shrinkage target vector.
  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` vector of sample expected returns.
  - $(math_dict[:N])

**`VolatilityWeighted`**: each target element is the inverse-covariance-weighted mean:

```math
\\begin{align}
b_j &= \\bar{\\mu}_{\\text{vol}} = \\frac{\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}}{\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{1}}, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - ``b_j``: ``j``-th element of the shrinkage target vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` sample covariance matrix.
  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` sample expected returns vector.
  - ``\\boldsymbol{1}``: ``N \\times 1`` vector of ones.

**`MeanSquaredError`**: each target element is the scaled matrix trace:

```math
\\begin{align}
b_j &= \\frac{\\mathrm{tr}(\\hat{\\mathbf{\\Sigma}})}{T}, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - ``b_j``: ``j``-th element of the shrinkage target vector.
  - ``\\mathrm{tr}(\\cdot)``: Matrix trace operator.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` sample covariance matrix.
  - $(math_dict[:T])

# Arguments

  - `tgt`: The shrinkage target type.

      + `tgt::GrandMean`: Returns a vector filled with the mean of `mu`.
      + `tgt::VolatilityWeighted`: Returns a vector filled with the volatility-weighted mean of `mu`, using the inverse covariance matrix.
      + `tgt::MeanSquaredError`: Returns a vector filled with the trace of `sigma` divided by `T`.

  - `mu`: 1D array of expected returns.

  - `sigma`: Covariance matrix of asset returns.

  - `kwargs...`: Additional keyword arguments, such as `T` (number of observations) or `isigma` (inverse covariance matrix).

# Returns

  - `b::ArrNum`: Target vector for shrinkage estimation.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquaredError`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
function target_mean(::GrandMean, mu::ArrNum, sigma::MatNum, args...; kwargs...)
    val = Statistics.mean(mu)
    return range(val, val; length = length(mu))
end
function target_mean(::VolatilityWeighted, mu::ArrNum, sigma::MatNum,
                     isigma::Option{<:MatNum} = nothing; kwargs...)
    if isnothing(isigma)
        isigma = sigma \ LinearAlgebra.I
    end
    if isone(size(mu, 1))
        mu = vec(mu)
    end
    val = sum(isigma * mu) / sum(isigma)
    return range(val, val; length = length(mu))
end
function target_mean(::MeanSquaredError, mu::ArrNum, sigma::MatNum, args...; T::Integer,
                     kwargs...)
    val = LinearAlgebra.tr(sigma) / T
    return range(val, val; length = length(mu))
end
"""
    Statistics.mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute shrunk expected returns using the specified estimator.

This method applies a shrinkage algorithm to the sample expected returns, pulling them toward a specified target to reduce estimation error, especially in high-dimensional settings.

# Mathematical definition

James-Stein shrinkage of sample expected returns toward target ``\\boldsymbol{b}``:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{JS} &= (1 - \\alpha)\\, \\hat{\\boldsymbol{\\mu}} + \\alpha\\, \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{JS}``: James-Stein shrunk expected returns.
  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` sample expected returns.
  - ``\\boldsymbol{b}``: ``N \\times 1`` shrinkage target vector.
  - ``\\alpha``: Shrinkage intensity.

The shrinkage intensity is:

```math
\\begin{align}
\\alpha &= \\frac{N \\bar{\\lambda} - 2 \\lambda_{\\max}}{T \\, \\lVert \\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b} \\rVert_2^2}\\,.
\\end{align}
```

Where:

  - ``\\bar{\\lambda}``: Mean eigenvalue of the covariance matrix.
  - ``\\lambda_{\\max}``: Maximum eigenvalue of the covariance matrix.
  - $(math_dict[:T])
  - $(math_dict[:N])

# Arguments

  - `me`: Shrunk expected returns estimator.

      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}`: Use the James-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}`: Use the Bayes-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}`: Use the Bodnar-Okhrin-Parolya algorithm.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and covariance estimators.

# Returns

  - `mu::ArrNum`: Shrunk expected returns vector.

# Details

  - Computes the sample mean and covariance.

  - Computes the shrinkage target using `target_mean`.

  - Computes the shrinkage intensity `alpha` with:

      + `JamesStein`: The centered mean and eigenvalues of the covariance matrix.
      + `BayesStein`: A Bayesian formula involving the centered mean and inverse covariance.
      + `BodnarOkhrinParolya`: A Bayesian formula involving the target mean, mean and inverse covariance.

  - Returns the shrunk mean vector.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    b = target_mean(me.alg.tgt, mu, sigma; T = T)
    if flag
        b = transpose(b)
    end
    evals = LinearAlgebra.eigvals(sigma)
    mb = mu - b
    alpha = (N * Statistics.mean(evals) - 2 * maximum(evals)) / LinearAlgebra.dot(mb, mb) /
            T
    return (one(alpha) - alpha) * mu + alpha * b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`BayesStein`](@ref) overload of [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Shrinks sample returns toward the target using a Bayesian formula with inverse covariance weighting.

# Mathematical definition

Bayes-Stein shrinkage intensity:

```math
\\begin{align}
\\alpha &= \\frac{N + 2}{(N + 2) + T \\, (\\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b})^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} (\\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b})}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_{BS} &= (1 - \\alpha)\\hat{\\boldsymbol{\\mu}} + \\alpha \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\alpha``: Bayes-Stein shrinkage intensity.
  - ``\\hat{\\boldsymbol{\\mu}}_{BS}``: Bayes-Stein shrunk expected returns.
  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` sample expected returns.
  - ``\\boldsymbol{b}``: ``N \\times 1`` shrinkage target vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` sample covariance matrix.
  - $(math_dict[:T])
  - $(math_dict[:N])
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
    end
    mb = vec(mu - b)
    alpha = (N + 2) / ((N + 2) + T * LinearAlgebra.dot(mb, isigma, mb))
    return (one(alpha) - alpha) * mu + alpha * b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`BodnarOkhrinParolya`](@ref) overload of [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Shrinks sample returns toward the target using the Bodnar-Okhrin-Parolya formula, designed for robust high-dimensional estimation.

# Mathematical definition

Define scalars:

```math
\\begin{align}
u &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}\\,, \\\\
v &= \\boldsymbol{b}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{b}\\,, \\\\
w &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``u``, ``v``, ``w``: Inverse-covariance-weighted quadratic forms.
  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` sample expected returns.
  - ``\\boldsymbol{b}``: ``N \\times 1`` shrinkage target vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` sample covariance matrix.

```math
\\begin{align}
\\alpha &= \\frac{(u - N/(T-N))v - w^2}{uv - w^2}\\,, \\\\
\\beta &= \\frac{(1-\\alpha) w}{u}\\,.
\\end{align}
```

Where:

  - ``\\alpha``, ``\\beta``: Shrinkage coefficients.
  - $(math_dict[:T])
  - $(math_dict[:N])

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{BOP} &= \\alpha \\hat{\\boldsymbol{\\mu}} + \\beta \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{BOP}``: Bodnar-Okhrin-Parolya shrunk expected returns.
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya},
                         X::MatNum; dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
        vm = vec(mu)
        vb = vec(b)
    else
        vm = mu
        vb = b
    end
    u = LinearAlgebra.dot(vm, isigma, vm)
    v = LinearAlgebra.dot(vb, isigma, vb)
    w = LinearAlgebra.dot(vm, isigma, vb)
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= u * v - w^2
    beta = (one(alpha) - alpha) * w / u
    return alpha * mu + beta * b
end

export GrandMean, VolatilityWeighted, MeanSquaredError, JamesStein, BayesStein,
       BodnarOkhrinParolya, ShrunkExpectedReturns
