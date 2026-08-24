"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber covariance estimators.

All concrete and/or abstract types implementing Gerber covariance estimation algorithms should be subtypes of `BaseGerberCovariance`.

# Interfaces

If moving away from the already established Gerber covariance algorithms, you must follow [`AbstractCovarianceEstimator`](@ref) to implement the entire chain.

# Related

  - [`GerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`BaseSmythBrobyCovariance`](@ref): subtype that weights each crossing instead of counting it.
  - [`BaseGerberIQCovariance`](@ref): subtype that scales the threshold per pair and decays each observation in time.

# References

  - $(ref_dict[:gerber])
  - $(ref_dict[:gerber_analysis])
"""
abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber covariance algorithm types.

All concrete and/or abstract types implementing specific Gerber covariance algorithms should be subtypes of `GerberCovarianceAlgorithm`. These types are used to specify the algorithm when constructing a [`GerberCovariance`](@ref) estimator. A subtype selects the denominator that [`comovement_ratio`](@ref) puts under the net co-movement vote, so a new subtype is a new method of that function.

# Interfaces

If moving away from the already established Gerber covariance algorithms, you must follow [`AbstractCovarianceEstimator`](@ref) to implement the entire chain.

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`comovement_ratio`](@ref)

# References

  - $(ref_dict[:gerber])
"""
abstract type GerberCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Normalises the net co-movement vote by the observations on which both assets crossed their threshold. This is the original Gerber statistic.

# Mathematical definition

```math
\\begin{align}
\\rho_{i,\\,j} &= \\frac{n_{c} - n_{d}}{n_{c} + n_{d}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rho_ij])
  - $(math_dict[:nc_gerber])
  - $(math_dict[:nd_gerber])

An observation votes only when both assets cross, so an observation on which exactly one asset crossed leaves the statistic unchanged. The denominator vanishes when no observation moved both assets, and the statistic is zero there.

# Algorithm

The branch of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Add the concordant and discordant counts into the denominator `den`.
 2. Return `zero(T)` when `den` is zero. An asset that never crosses its own threshold gives that case for every pair it belongs to.
 3. Otherwise return `(p - n) / den`.

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
  - [`comovement_ratio`](@ref)

# References

  - $(ref_dict[:gerber])
"""
struct Gerber0 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Normalises the net co-movement vote by every observation on which at least one asset crossed its threshold.

# Mathematical definition

```math
\\begin{align}
\\rho_{i,\\,j} &= \\frac{n_{c} - n_{d}}{n_{c} + n_{d} + n_{n}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rho_ij])
  - $(math_dict[:nc_gerber])
  - $(math_dict[:nd_gerber])
  - $(math_dict[:nn_gerber])

The denominator carries ``n_{n}`` on top of [`Gerber0`](@ref)'s, so it is never smaller and the statistic is never larger in magnitude. The two agree when every crossing is shared.

# Algorithm

The branch of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Add the concordant, discordant and neutral counts into the denominator `den`.
 2. Return `zero(T)` when `den` is zero. An asset that never crosses its own threshold gives that case for every pair it belongs to.
 3. Otherwise return `(p - n) / den`.

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
  - [`comovement_ratio`](@ref)

# References

  - $(ref_dict[:gerber])
"""
struct Gerber1 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Normalises the raw net co-movement vote by the geometric mean of its own diagonal.

# Mathematical definition

```math
\\begin{align}
h_{i,\\,j} &= n_{c} - n_{d}\\,, \\\\
\\rho_{i,\\,j} &= \\frac{h_{i,\\,j}}{\\sqrt{h_{i,\\,i} \\, h_{j,\\,j}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rho_ij])
  - ``h_{i,\\,j}``: Net co-movement vote of the pair, before any normalisation.
  - $(math_dict[:nc_gerber])
  - $(math_dict[:nd_gerber])

The normalisation is a property of the whole matrix and not of one pair, so the diagonal is unit by construction rather than by a per-pair denominator as in [`Gerber0`](@ref) and [`Gerber1`](@ref). An asset crosses concordantly with itself at every crossing, so ``h_{i,\\,i}`` counts the crossings of asset ``i``.

# Algorithm

The branch of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this tag selects runs these steps.

 1. Return the raw difference `p - n` for every pair. This branch applies no denominator of its own.
 2. Divide the assembled matrix by the outer product of the square roots of its own diagonal. The roots are clamped from below by `sqrt(eps(eltype(rho)))`, so an asset that never crosses gives a zero row rather than a division by zero.

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
  - [`comovement_ratio`](@ref)
  - [`standardise_comovement!`](@ref)

# References

  - $(ref_dict[:gerber])
"""
struct Gerber2 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Configures and applies Gerber covariance estimators.

`GerberCovariance` encapsulates all components required for Gerber-based covariance or correlation estimation, including the variance estimator, positive definite matrix estimator, t parameter, and the specific Gerber algorithm variant. A Gerber matrix is a matrix of pairwise votes and is not positive definite in general, so `pdm` projects the result onto the nearest positive definite matrix; `pdm = nothing` returns the raw statistic instead.

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ve`: Recursively updated via [`factory`](@ref).
  - `me`: Recursively updated via [`factory`](@ref).
  - `alg`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `me`: Recursively viewed via [`port_opt_view`](@ref).

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:gerber])
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

# Mathematical definition

```math
\\begin{align}
U_{t,\\,i} &= \\mathbf{1}[x_{t,\\,i} \\geq t \\, \\sigma_i]\\,, \\\\
D_{t,\\,i} &= \\mathbf{1}[x_{t,\\,i} \\leq -t \\, \\sigma_i]\\,.
\\end{align}
```

Where:

  - $(math_dict[:U_gerber])
  - $(math_dict[:D_gerber])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:t_threshold])
  - $(math_dict[:sigma_i_asset])
  - $(math_dict[:T])
  - $(math_dict[:N])

The two bands do not overlap for a positive threshold, so no observation is marked in both matrices and their sum is the crossing indicator.

# Algorithm

 1. Scale the standard deviation vector by the threshold, giving the per-asset band edge `ts = sd * ce.t`.
 2. Mark `U[t, i]` when `X[t, i] >= ts[i]`.
 3. Mark `D[t, i]` when `X[t, i] <= -ts[i]`.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - `(U, D)::Tuple{Matrix{Bool}, Matrix{Bool}}`: The up and the down indicator matrices.

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

A matrix product delivers the difference and the sum directly, so the two counts are recovered from those instead of by two more matrix products. The split is exact, and the reduction in [`comovement_ratio`](@ref) sees the same numerator and denominator as the matrix formula.

# Mathematical definition

```math
\\begin{align}
n_{c} &= \\frac{(n_{c} + n_{d}) + (n_{c} - n_{d})}{2}\\,, \\\\
n_{d} &= \\frac{(n_{c} + n_{d}) - (n_{c} - n_{d})}{2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:nc_gerber])
  - $(math_dict[:nd_gerber])

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

```math
\\begin{align}
\\hat{\\boldsymbol{\\rho}} &= \\left(\\mathbf{H}^\\intercal \\mathbf{H}\\right) \\oslash \\left(\\mathbf{V}^\\intercal \\mathbf{V}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:U_gerber])
  - $(math_dict[:D_gerber])
  - $(math_dict[:H_gerber])
  - $(math_dict[:Vcross_gerber])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:t_threshold])
  - $(math_dict[:sigma_i_asset])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:oslash])

The entry of ``\\mathbf{H}^\\intercal \\mathbf{H}`` is ``n_{c} - n_{d}`` and the entry of ``\\mathbf{V}^\\intercal \\mathbf{V}`` is ``n_{c} + n_{d}``, so this is the pairwise statistic of [`Gerber0`](@ref) written over the whole matrix.

# Algorithm

 1. Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
 2. Form the signed crossing matrix `UmD = U - D` and the crossing matrix `UpD = U + D`.
 3. Recover the concordant count `nconc` and the discordant count `ndisc` from `transpose(UmD) * UmD` and `transpose(UpD) * UpD` with [`concordance_counts`](@ref).
 4. Reduce every pair with [`comovement_ratio`](@ref), giving `rho`.
 5. Repair `rho` with `posdef!`, which is a no-op when `ce.pdm` is `nothing`.

# Arguments

  - $(arg_dict[:gerbce]). Configured with the `Gerber0` algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`gerber_updown`](@ref)
  - [`concordance_counts`](@ref)
  - [`comovement_ratio`](@ref)
  - [`posdef!`](@ref)

# References

  - $(ref_dict[:gerber])
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
        ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1},
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Implements the first variant of the Gerber correlation algorithm.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\rho}} &= \\left(\\mathbf{H}^\\intercal \\mathbf{H}\\right) \\oslash \\left(T \\boldsymbol{1}\\boldsymbol{1}^\\intercal - \\mathbf{N}^\\intercal \\mathbf{N}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:U_gerber])
  - $(math_dict[:D_gerber])
  - $(math_dict[:Nneut_gerber])
  - $(math_dict[:H_gerber])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:t_threshold])
  - $(math_dict[:sigma_i_asset])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

The entry of ``\\mathbf{N}^\\intercal \\mathbf{N}`` counts the observations on which neither asset crossed, so the denominator counts the observations on which at least one of them did. That is ``n_{c} + n_{d} + n_{n}``, the pairwise denominator of [`Gerber1`](@ref).

# Algorithm

 1. Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
 2. Form the neutral matrix `Nt`, which marks the observations on which the asset crossed in neither direction.
 3. Form `NtN = transpose(Nt) * Nt`, the count of observations on which both assets of a pair are neutral, and `nneutral`, the count of neutral observations of each asset on its own.
 4. Form the signed crossing matrix `UmD = U - D`.
 5. Recover the concordant count `nconc` and the discordant count `ndisc` from `transpose(UmD) * UmD` and the both-crossed count `T .- nneutral .- transpose(nneutral) .+ NtN` with [`concordance_counts`](@ref).
 6. Form `nneut = nneutral .+ transpose(nneutral) .- 2 .* NtN`, the count of observations on which exactly one asset of the pair crossed.
 7. Reduce every pair with [`comovement_ratio`](@ref), giving `rho`.
 8. Repair `rho` with `posdef!`, which is a no-op when `ce.pdm` is `nothing`.

# Arguments

  - $(arg_dict[:gerbce]). Configured with the `Gerber1` algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber1`](@ref)
  - [`gerber_updown`](@ref)
  - [`concordance_counts`](@ref)
  - [`comovement_ratio`](@ref)
  - [`posdef!`](@ref)

# References

  - $(ref_dict[:gerber])
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
        ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2},
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Implements the second variant of the Gerber correlation algorithm.

# Mathematical definition

```math
\\begin{align}
\\mathbf{G} &= \\mathbf{H}^\\intercal \\mathbf{H}\\,, \\\\
\\boldsymbol{g} &= \\sqrt{\\mathrm{diag}(\\mathbf{G})}\\,, \\\\
\\hat{\\boldsymbol{\\rho}} &= \\mathbf{G} \\oslash (\\boldsymbol{g} \\boldsymbol{g}^\\intercal)\\,.
\\end{align}
```

Where:

  - $(math_dict[:U_gerber])
  - $(math_dict[:D_gerber])
  - $(math_dict[:H_gerber])
  - ``\\mathbf{G}``: Raw net co-movement matrix, whose entry is ``n_{c} - n_{d}``.
  - ``\\boldsymbol{g}``: Square roots of the diagonal of ``\\mathbf{G}``.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:t_threshold])
  - $(math_dict[:sigma_i_asset])
  - $(math_dict[:oslash])

The diagonal of ``\\mathbf{G}`` counts the crossings of each asset, so the normalisation is a property of the whole matrix and the diagonal of ``\\hat{\\boldsymbol{\\rho}}`` is unit by construction.

# Algorithm

 1. Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
 2. Form the signed crossing matrix `UmD = U - D`.
 3. Form the raw net co-movement matrix `rho = transpose(UmD) * UmD`.
 4. Normalise `rho` in place with [`standardise_comovement!`](@ref).
 5. Repair `rho` with `posdef!`, which is a no-op when `ce.pdm` is `nothing`.

# Arguments

  - $(arg_dict[:gerbce]). Configured with the `Gerber2` algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber2`](@ref)
  - [`gerber_updown`](@ref)
  - [`standardise_comovement!`](@ref)
  - [`posdef!`](@ref)

# References

  - $(ref_dict[:gerber])
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

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute the standard deviation vector `sd` with `ce.ve`, and raise each entry to at least `eps(eltype(sd))`. A constant column leaves a standard deviation and a centring residual of the same round-off order, so the unraised threshold marks every one of its observations as a crossing; the raised threshold marks none of them.
 3. Centre the returns with `ce.me` through [`demean_returns`](@ref).
 4. Return the Gerber correlation matrix from [`gerber`](@ref), through the branch that `ce.alg` selects.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum, sd::ArrNum)`](@ref)
  - [`demean_returns`](@ref)
  - [`cov(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber])
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

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\mathrm{Diag}(\\boldsymbol{\\sigma}) \\, \\hat{\\boldsymbol{\\rho}} \\, \\mathrm{Diag}(\\boldsymbol{\\sigma})\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{\\Sigma}}``: Gerber covariance matrix.
  - ``\\hat{\\boldsymbol{\\rho}}``: Gerber correlation matrix of the same estimator.
  - ``\\boldsymbol{\\sigma}``: Standard deviation vector of the assets.

The Gerber statistic sets the correlations alone, so the variances come from `ce.ve` and the diagonal of ``\\hat{\\mathbf{\\Sigma}}`` is ``\\boldsymbol{\\sigma}^2``.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute the standard deviation vector `sd` with `ce.ve`, and raise each entry to at least `eps(eltype(sd))`. A constant column leaves a standard deviation and a centring residual of the same round-off order, so the unraised threshold marks every one of its observations as a crossing; the raised threshold marks none of them.
 3. Centre the returns with `ce.me` through [`demean_returns`](@ref).
 4. Compute the Gerber correlation matrix `sigma` with [`gerber`](@ref), through the branch that `ce.alg` selects.
 5. Rescale `sigma` in place to a covariance matrix with `StatsBase.cor2cov!` and `sd`.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum, sd::ArrNum)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum, sd::ArrNum)`](@ref)
  - [`demean_returns`](@ref)
  - [`cor(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber])
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
