"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber covariance estimators in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing Gerber covariance estimation algorithms should be subtypes of `BaseGerberCovariance`.

# Interfaces

If moving away from the already established Gerber covariance algorithms, you must follow [`AbstractCovarianceEstimator`](@ref) to implement the entire chain.

# Related

  - [`GerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber covariance algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific Gerber covariance algorithms should be subtypes of `GerberCovarianceAlgorithm`.

These types are used to specify the algorithm when constructing a [`GerberCovariance`](@ref) estimator.

# Interfaces

If moving away from the already established Gerber covariance algorithms, you must follow [`AbstractCovarianceEstimator`](@ref) to implement the entire chain. Else you can follow the instructions and examples in [`GerberCovarianceAlgorithm`](@ref).

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovariance`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
abstract type GerberCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the original Gerber covariance algorithm.

# Related

  - [`GerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
struct Gerber0 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the first variant of the Gerber covariance algorithm.

# Related

  - [`GerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber2`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
struct Gerber1 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements the second variant of the Gerber covariance algorithm.

# Related

  - [`GerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
struct Gerber2 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

A flexible container type for configuring and applying Gerber covariance estimators in `PortfolioOptimisers.jl`.

`GerberCovariance` encapsulates all components required for Gerber-based covariance or correlation estimation, including the variance estimator, positive definite matrix estimator, t parameter, and the specific Gerber algorithm variant.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GerberCovariance(;
        ve::StatsBase.CovarianceEstimator = SimpleVariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        pdm::Option{<:Posdef} = Posdef(),
        t::Number = 0.5,
        alg::GerberCovarianceAlgorithm = Gerber1()
    ) -> GerberCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:gerbt])

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`SimpleVariance`](@ref)
  - [`Posdef`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
@concrete struct GerberCovariance <: BaseGerberCovariance
    "$(field_dict[:ve])"
    ve
    "$(field_dict[:me]) Used for centering the returns."
    me
    "$(field_dict[:pdm])"
    pdm
    "$(field_dict[:t])"
    t
    "$(field_dict[:gerbalg])"
    alg
    function GerberCovariance(ve::StatsBase.CovarianceEstimator,
                              me::AbstractExpectedReturnsEstimator, pdm::Option{<:Posdef},
                              t::Number, alg::GerberCovarianceAlgorithm)
        assert_nonempty_nonneg_finite_val(t, :t)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(t), typeof(alg)}(ve, me, pdm,
                                                                                t, alg)
    end
end
function GerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                          me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                          pdm::Option{<:Posdef} = Posdef(), t::Number = 0.5,
                          alg::GerberCovarianceAlgorithm = Gerber1())::GerberCovariance
    return GerberCovariance(ve, me, pdm, t, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new `GerberCovariance` estimator with the specified observation weights.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:oow])

# Returns

  - $(ret_dict[:ce])

# Details

  - Calls `factory(ce.alg, w)` to update the algorithm (current algorithms do not use weights, this for future proofing).
  - Calls `factory(ce.ve, w)` to update the variance estimator.
  - Calls `factory(ce.me, w)` to update the expected returns estimator.
  - Preserves the other fields of the original estimator.

# Examples

```jldoctest
julia> ce = GerberCovariance()
GerberCovariance
   ve ┼ SimpleVariance
      │          me ┼ SimpleExpectedReturns
      │             │   w ┴ nothing
      │           w ┼ nothing
      │   corrected ┴ Bool: true
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
    t ┼ Float64: 0.5
  alg ┴ Gerber1()

julia> factory(ce, StatsBase.Weights([0.1, 0.2, 0.7]))
GerberCovariance
   ve ┼ SimpleVariance
      │          me ┼ SimpleExpectedReturns
      │             │   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
      │           w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
      │   corrected ┴ Bool: true
   me ┼ SimpleExpectedReturns
      │   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
    t ┼ Float64: 0.5
  alg ┴ Gerber1()
```

# Related

  - [`GerberCovariance`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`factory`](@ref)
"""
function factory(ce::GerberCovariance, w::ObsWeights)::GerberCovariance
    return GerberCovariance(; ve = factory(ce.ve, w), me = factory(ce.me, w), pdm = ce.pdm,
                            t = ce.t, alg = factory(ce.alg, w))
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

  - [`GerberCovariance`](@ref)
"""
function moment_view(ce::GerberCovariance, i)::GerberCovariance
    return GerberCovariance(; ve = moment_view(ce.ve, i), me = moment_view(ce.me, i),
                            pdm = ce.pdm, t = ce.t, alg = ce.alg)
end
"""
    gerber(
        ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0},
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Implements the original Gerber correlation algorithm.

# Mathematical definition

Let ``\\mathbf{U}, \\mathbf{D} \\in \\{0,1\\}^{T \\times N}`` be indicator matrices with:

```math
U_{ti} = \\mathbf{1}[x_{ti} \\geq t \\, \\sigma_i], \\quad D_{ti} = \\mathbf{1}[x_{ti} \\leq -t \\, \\sigma_i]
```

Define ``\\mathbf{H} = \\mathbf{U} - \\mathbf{D}`` and ``\\mathbf{V} = \\mathbf{U} + \\mathbf{D}``. The Gerber0 correlation is:

```math
\\hat{\\boldsymbol{\\rho}} = \\left(\\mathbf{H}^\\intercal \\mathbf{H}\\right) \\oslash \\left(\\mathbf{V}^\\intercal \\mathbf{V}\\right)
```

Where:

  - ``x_{ti}``: Return of asset ``i`` at time ``t``.
  - ``t``: Threshold parameter.
  - ``\\sigma_i``: Standard deviation of asset ``i``.
  - ``T``: Number of observations.
  - ``N``: Number of assets.
  - ``\\oslash``: Element-wise division.

# Arguments

  - $(arg_dict[:gerbce]). Configured with the `Gerber0` algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Details

The algorithm proceeds as follows:

  - For each entry in `X`, compute two Boolean matrices:
      + `U`: Entries where `X .>= ce.t * sd`.
      + `D`: Entries where `X .<= -ce.t * sd`.
  - Compute `UmD = U - D` and `UpD = U + D`.
  - The Gerber correlation is given by `(UmD' * UmD) ⊘ (UpD' * UpD)`.
  - The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`posdef!`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum,
                sd::ArrNum)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    sd = sd * ce.t
    U .= X .>= sd
    D .= X .<= -sd
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    UpD = U + D
    rho = (transpose(UmD) * UmD) ⊘ (transpose(UpD) * UpD)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(
        ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1},
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Implements the first variant of the Gerber correlation algorithm.

# Mathematical definition

Let ``\\mathbf{U}, \\mathbf{D}, \\mathbf{N} \\in \\{0,1\\}^{T \\times N}`` be indicator matrices with:

```math
U_{ti} = \\mathbf{1}[x_{ti} \\geq t \\, \\sigma_i], \\quad D_{ti} = \\mathbf{1}[x_{ti} \\leq -t \\, \\sigma_i], \\quad N_{ti} = \\mathbf{1}[{-t\\sigma_i < x_{ti} < t\\sigma_i}]
```

Define ``\\mathbf{H} = \\mathbf{U} - \\mathbf{D}``. The Gerber1 correlation is:

```math
\\hat{\\boldsymbol{\\rho}} = \\left(\\mathbf{H}^\\intercal \\mathbf{H}\\right) \\oslash \\left(T \\boldsymbol{1}\\boldsymbol{1}^\\intercal - \\mathbf{N}^\\intercal \\mathbf{N}\\right)
```

Where:

  - ``x_{ti}``: Return of asset ``i`` at time ``t``.
  - ``t``: Threshold parameter.
  - ``\\sigma_i``: Standard deviation of asset ``i``.
  - ``T``: Number of observations.
  - ``N``: Number of assets.
  - ``\\oslash``: Element-wise division.
  - ``\\boldsymbol{1}``: Vector of ones.

# Arguments

  - $(arg_dict[:gerbce]). Configured with the `Gerber1` algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Details

The algorithm proceeds as follows:

  - For each entry in `X`, compute three Boolean matrices:
      + `U`: Entries where `X .>= ce.t * sd`.
      + `D`: Entries where `X .<= -ce.t * sd`.
      + `N`: Entries where `X in (-ce.t * sd, ce.t * sd)` (i.e., neither up nor down).
  - Compute `UmD = U - D`.
  - The Gerber1 correlation is given by `(UmD' * UmD) ⊘ (T .- (N' * N))`, where `T` is the number of observations.
  - The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber1`](@ref)
  - [`posdef!`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum,
                sd::ArrNum)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    N = Matrix{Bool}(undef, T, N)
    sd = sd * ce.t
    U .= X .>= sd
    D .= X .<= -sd
    N .= .!U .& .!D
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    rho = transpose(UmD) * (UmD) ⊘ (T .- transpose(N) * N)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(
        ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2},
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Implements the second variant of the Gerber correlation algorithm.

# Mathematical definition

Let ``\\mathbf{U}, \\mathbf{D} \\in \\{0,1\\}^{T \\times N}`` be indicator matrices with:

```math
U_{ti} = \\mathbf{1}[x_{ti} \\geq t \\, \\sigma_i], \\quad D_{ti} = \\mathbf{1}[x_{ti} \\leq -t \\, \\sigma_i]
```

Define ``\\mathbf{H} = (\\mathbf{U} - \\mathbf{D})^\\intercal (\\mathbf{U} - \\mathbf{D})`` and ``\\boldsymbol{h} = \\sqrt{\\mathrm{diag}(\\mathbf{H})}``. The Gerber2 correlation is:

```math
\\hat{\\boldsymbol{\\rho}} = \\mathbf{H} \\oslash (\\boldsymbol{h} \\boldsymbol{h}^\\intercal)
```

Where:

  - ``x_{ti}``: Return of asset ``i`` at time ``t``.
  - ``t``: Threshold parameter.
  - ``\\sigma_i``: Standard deviation of asset ``i``.
  - ``\\mathrm{diag}(\\cdot)``: Diagonal of a matrix.
  - ``\\oslash``: Element-wise division.

# Arguments

  - $(arg_dict[:gerbce]). Configured with the `Gerber2` algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Details

The algorithm proceeds as follows:

  - For each entry in `X`, compute two Boolean matrices:
      + `U`: Entries where `X .>= ce.t * sd`.
      + `D`: Entries where `X .<= -ce.t * sd`.
  - Compute the signed indicator matrix `UmD = U - D`.
  - Compute the raw Gerber2 matrix `H = UmD' * UmD`.
  - Normalise `rho = H ⊘ (h * h')`, where `h = sqrt.(LinearAlgebra.diag(H))`.
  - The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber2`](@ref)
  - [`posdef!`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum,
                sd::ArrNum)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    sd = sd * ce.t
    U .= X .>= sd
    D .= X .<= -sd
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    H = transpose(UmD) * (UmD)
    h = sqrt.(LinearAlgebra.diag(H))
    rho = H ⊘ (h * transpose(h))
    posdef!(ce.pdm, rho)
    return rho
end
"""
    Statistics.cor(
        ce::GerberCovariance,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the Gerber correlation matrix using the algorithm specified in `ce.alg`.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - $(arg_dict[:rho])

# Validation

  - $(val_dict[:dims])

# Details

  - Computes the standard deviation vector for each asset using the estimator's variance estimator.
  - Demeans the returns with `ce.me` and [`demean_returns`](@ref).
  - Computes the Gerber correlation matrix using the Gerber algorithm in `ce.alg`.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum, sd::ArrNum)`](@ref)
  - [`demean_returns`](@ref)
  - [`cov(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function Statistics.cor(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    return gerber(ce, X, sd)
end
"""
    Statistics.cov(
        ce::GerberCovariance,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the Gerber covariance matrix using the algorithm specified in `ce.alg`.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - $(arg_dict[:rho])

# Validation

  - $(val_dict[:dims])

# Details

  - Computes the standard deviation vector for each asset using the estimator's variance estimator.
  - Demeans the returns with `ce.me` and [`demean_returns`](@ref).
  - Computes the Gerber correlation matrix using the Gerber algorithm in `ce.alg`.
  - Rescales the Gerber correlation matrix to a covariance matrix by multiplying with the standard deviation vector outer product.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum, sd::ArrNum)`](@ref)
  - [`demean_returns`](@ref)
  - [`cor(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function Statistics.cov(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    sigma = gerber(ce, X, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export GerberCovariance, Gerber0, Gerber1, Gerber2
