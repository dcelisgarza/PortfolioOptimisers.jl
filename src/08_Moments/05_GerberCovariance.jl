"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber covariance estimators.

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

Abstract supertype for all Gerber covariance algorithm types.

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

# Constructors

    Gerber0() -> Gerber0

# Examples

```jldoctest
julia> Gerber0()
Gerber0()
```

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

# Constructors

    Gerber1() -> Gerber1

# Examples

```jldoctest
julia> Gerber1()
Gerber1()
```

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

# Constructors

    Gerber2() -> Gerber2

# Examples

```jldoctest
julia> Gerber2()
Gerber2()
```

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

Configures and applies Gerber covariance estimators.

`GerberCovariance` encapsulates all components required for Gerber-based covariance or correlation estimation, including the variance estimator, positive definite matrix estimator, t parameter, and the specific Gerber algorithm variant.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GerberCovariance(;
        ve::StatsBase.CovarianceEstimator = SimpleVariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        t::Number = 0.5,
        alg::GerberCovarianceAlgorithm = Gerber1()
    ) -> GerberCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:gerbt])

# Examples

```jldoctest
julia> GerberCovariance()
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
```

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
@propagatable @concrete struct GerberCovariance <: BaseGerberCovariance
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:me]) Used for centering the returns.
    """
    @fprop @vprop me
    """
    $(field_dict[:pdm])
    """
    pdm
    """
    $(field_dict[:t])
    """
    t
    """
    $(field_dict[:gerbalg])
    """
    @fprop alg
    function GerberCovariance(ve::StatsBase.CovarianceEstimator,
                              me::AbstractExpectedReturnsEstimator,
                              pdm::Option{<:AbstractPosdefEstimator}, t::Number,
                              alg::GerberCovarianceAlgorithm)
        assert_nonempty_nonneg_finite_val(t, :t)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(t), typeof(alg)}(ve, me, pdm,
                                                                                t, alg)
    end
end
function GerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                          me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                          pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                          t::Number = 0.5,
                          alg::GerberCovarianceAlgorithm = Gerber1())::GerberCovariance
    return GerberCovariance(ve, me, pdm, t, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the up and down indicator matrices shared by every Gerber correlation variant.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - `(U, D)::Tuple{Matrix{Bool}, Matrix{Bool}}`: `U[t, i]` marks `X[t, i] >= ce.t * sd[i]`, and `D[t, i]` marks `X[t, i] <= -ce.t * sd[i]`.

# Related

  - [`GerberCovariance`](@ref)
  - [`concordance_counts`](@ref)
"""
function gerber_updown(ce::GerberCovariance, X::MatNum, sd::ArrNum)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    ts = sd * ce.t
    U .= X .>= ts
    D .= X .<= -ts
    return U, D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Split the concordant and discordant co-movement counts out of their difference and their sum.

`nconc[i, j]` counts the observations on which assets `i` and `j` both crossed a threshold in the same direction. `ndisc[i, j]` counts the observations on which they crossed in opposite directions. A matrix product delivers the difference and the sum directly, so the two counts are recovered from those instead of by two more matrix products. The split is exact, and the reduction in [`comovement_ratio`](@ref) sees the same numerator and denominator as the matrix formula.

# Arguments

  - `pmn::AbstractMatrix`: The difference `nconc - ndisc`.
  - `ppn::AbstractMatrix`: The sum `nconc + ndisc`.

# Returns

  - `(nconc, ndisc)::Tuple{AbstractMatrix, AbstractMatrix}`: The concordant and the discordant counts.

# Related

  - [`gerber_updown`](@ref)
  - [`comovement_ratio`](@ref)
"""
function concordance_counts(pmn::AbstractMatrix, ppn::AbstractMatrix)
    return (ppn .+ pmn) ./ 2, (ppn .- pmn) ./ 2
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
\\begin{align}
U_{ti} &= \\mathbf{1}[x_{ti} \\geq t \\, \\sigma_i], \\quad D_{ti} = \\mathbf{1}[x_{ti} \\leq -t \\, \\sigma_i]\\,.
\\end{align}
```

Define ``\\mathbf{H} = \\mathbf{U} - \\mathbf{D}`` and ``\\mathbf{V} = \\mathbf{U} + \\mathbf{D}``. The Gerber0 correlation is:

```math
\\begin{align}
\\hat{\\boldsymbol{\\rho}} &= \\left(\\mathbf{H}^\\intercal \\mathbf{H}\\right) \\oslash \\left(\\mathbf{V}^\\intercal \\mathbf{V}\\right)\\,.
\\end{align}
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

  - Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
  - Compute `UmD = U - D` and `UpD = U + D`.
  - Recover the concordant and discordant counts from `UmD' * UmD` and `UpD' * UpD` with [`concordance_counts`](@ref).
  - Reduce each pair to `(nconc - ndisc) / (nconc + ndisc)` with [`comovement_ratio`](@ref), which returns zero when the denominator vanishes.
  - The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`gerber_updown`](@ref)
  - [`concordance_counts`](@ref)
  - [`comovement_ratio`](@ref)
  - [`posdef!`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum,
                sd::ArrNum)
    U, D = gerber_updown(ce, X, sd)
    UmD = U - D
    UpD = U + D
    nconc, ndisc = concordance_counts(transpose(UmD) * UmD, transpose(UpD) * UpD)
    rho = comovement_ratio.(Ref(ce.alg), nconc, ndisc, 0, eltype(X))
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
\\begin{align}
U_{ti} &= \\mathbf{1}[x_{ti} \\geq t \\, \\sigma_i], \\quad D_{ti} = \\mathbf{1}[x_{ti} \\leq -t \\, \\sigma_i], \\quad N_{ti} = \\mathbf{1}[{-t\\sigma_i < x_{ti} < t\\sigma_i}]\\,.
\\end{align}
```

Define ``\\mathbf{H} = \\mathbf{U} - \\mathbf{D}``. The Gerber1 correlation is:

```math
\\begin{align}
\\hat{\\boldsymbol{\\rho}} &= \\left(\\mathbf{H}^\\intercal \\mathbf{H}\\right) \\oslash \\left(T \\boldsymbol{1}\\boldsymbol{1}^\\intercal - \\mathbf{N}^\\intercal \\mathbf{N}\\right)\\,.
\\end{align}
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

  - Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
  - Compute the neutral matrix `Nt`, whose entries mark `X in (-ce.t * sd, ce.t * sd)` (i.e., neither up nor down).
  - Compute `UmD = U - D`.
  - Split the denominator `T .- (Nt' * Nt)` into the observations on which both assets crossed a threshold and the observations on which exactly one of them crossed.
  - Recover the concordant and discordant counts from `UmD' * UmD` and the both-crossed count with [`concordance_counts`](@ref).
  - Reduce each pair to `(nconc - ndisc) / (nconc + ndisc + nneut)` with [`comovement_ratio`](@ref), which returns zero when the denominator vanishes.
  - The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber1`](@ref)
  - [`gerber_updown`](@ref)
  - [`concordance_counts`](@ref)
  - [`comovement_ratio`](@ref)
  - [`posdef!`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum,
                sd::ArrNum)
    T = size(X, 1)
    U, D = gerber_updown(ce, X, sd)
    Nt = Matrix{Bool}(undef, size(X)...)
    Nt .= .!U .& .!D
    NtN = transpose(Nt) * Nt
    nneutral = vec(sum(Nt; dims = 1))
    UmD = U - D
    # A pair is neutral together on NtN observations, so it crosses a threshold together on
    # T - nneutral_i - nneutral_j + NtN of them, and exactly one of the two crosses on
    # nneutral_i + nneutral_j - 2 NtN. The three counts sum to the T .- NtN denominator.
    nconc, ndisc = concordance_counts(transpose(UmD) * UmD,
                                      T .- nneutral .- transpose(nneutral) .+ NtN)
    nneut = nneutral .+ transpose(nneutral) .- 2 .* NtN
    rho = comovement_ratio.(Ref(ce.alg), nconc, ndisc, nneut, eltype(X))
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
\\begin{align}
U_{ti} &= \\mathbf{1}[x_{ti} \\geq t \\, \\sigma_i], \\quad D_{ti} = \\mathbf{1}[x_{ti} \\leq -t \\, \\sigma_i]\\,.
\\end{align}
```

Define ``\\mathbf{H} = (\\mathbf{U} - \\mathbf{D})^\\intercal (\\mathbf{U} - \\mathbf{D})`` and ``\\boldsymbol{h} = \\sqrt{\\mathrm{diag}(\\mathbf{H})}``. The Gerber2 correlation is:

```math
\\begin{align}
\\hat{\\boldsymbol{\\rho}} &= \\mathbf{H} \\oslash (\\boldsymbol{h} \\boldsymbol{h}^\\intercal)\\,.
\\end{align}
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

  - Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
  - Compute the signed indicator matrix `UmD = U - D`.
  - Compute the raw Gerber2 matrix `rho = UmD' * UmD`.
  - Normalise `rho` by the geometric mean of its diagonal with [`standardise_comovement!`](@ref), which clamps the diagonal roots away from zero.
  - The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber2`](@ref)
  - [`gerber_updown`](@ref)
  - [`standardise_comovement!`](@ref)
  - [`posdef!`](@ref)

# References

  - [gerber](@cite) Gerber, Sander and Markowitz, Harry and Ernst, Philip and Miao, Yinsen and Name, No and Sargen, Paul, *The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization* (July 4, 2021). Available at SSRN: https://ssrn.com/abstract=3880054 or http://dx.doi.org/10.2139/ssrn.3880054
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum,
                sd::ArrNum)
    U, D = gerber_updown(ce, X, sd)
    UmD = U - D
    rho = Matrix{eltype(X)}(transpose(UmD) * UmD)
    standardise_comovement!(ce.alg, rho)
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
    X = dims_oriented(dims, X)
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
    X = dims_oriented(dims, X)
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    sigma = gerber(ce, X, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export GerberCovariance, Gerber0, Gerber1, Gerber2
