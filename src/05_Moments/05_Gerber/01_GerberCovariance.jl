"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber covariance estimators.

Every concrete or abstract type that implements a Gerber covariance estimator must subtype `BaseGerberCovariance`.

Section 2 of [gerber](@cite) defines the statistic. Section 2 of [gerber_analysis](@cite) compares its three denominators, and each [`GerberCovarianceAlgorithm`](@ref) subtype selects one of them.

# Interfaces

A new subtype must follow the interface of [`AbstractCovarianceEstimator`](@ref).

# Related

  - [`GerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`BaseSmythBrobyCovariance`](@ref): A subtype that gives each crossing a weight instead of a count of one.
  - [`BaseGerberIQCovariance`](@ref): A subtype that scales the threshold for each pair and gives older observations less weight.

# References

  - $(ref_dict[:gerber]) Section 2.
  - $(ref_dict[:gerber_analysis]) Section 2.
"""
abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Gerber covariance algorithm types.

Every concrete or abstract type that implements a Gerber covariance algorithm must subtype `GerberCovarianceAlgorithm`. A subtype goes in the `alg` field of a [`GerberCovariance`](@ref) estimator. It selects the denominator that [`comovement_ratio`](@ref) puts under the net co-movement count. Each method of [`gerber`](@ref) dispatches on one subtype, so a new subtype needs a new method of `gerber`.

[gerber](@cite) states the three denominators:

  - [`Gerber0`](@ref): Equation 4.
  - [`Gerber1`](@ref): Equation 11.
  - [`Gerber2`](@ref): footnote 3.

[gerber](@cite) is the final version of two earlier preprints, as its footnote 1 states. Section 2 of [gerber_analysis](@cite) compares the three denominators. It attributes the first to the first preprint, the third to the 2019 preprint and the second to the final version.

# Interfaces

The estimator that holds a new subtype must follow the interface of [`AbstractCovarianceEstimator`](@ref).

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`comovement_ratio`](@ref)

# References

  - $(ref_dict[:gerber]) Footnote 1, Section 2.1, Equation 4, Section 2.2, Equation 11, and footnote 3.
  - $(ref_dict[:gerber_analysis]) Section 2.
"""
abstract type GerberCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Counts only the observations on which both assets cross their thresholds, as the original Gerber statistic does. The denominator is the number of these observations.

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

An observation on which exactly one asset crosses does not change the statistic. When the two assets never cross their thresholds on the same observation, the denominator is zero and the statistic is zero.

This is Equation 4 of [gerber](@cite), which identifies it with Kendall's tau when the thresholds are zero. Section 2 of [gerber_analysis](@cite) dates it to the first preprint of the work. [gerber](@cite) is the final version of that preprint, as its footnote 1 states. On a random sample of four assets, Equation 4 gives the entries of the library to the last digit.

# Algorithm

The branch of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Add the concordant and discordant counts into the denominator `den`.
 2. Return `zero(T)` when `den` is zero. An asset that never crosses its own threshold gives a zero `den` for every pair that contains it.
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

  - $(ref_dict[:gerber]) Section 2.1, Equations 3 and 4, and footnote 1.
  - $(ref_dict[:gerber_analysis]) Section 2.
"""
struct Gerber0 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Adds the observations on which exactly one asset crosses its threshold to the denominator of the Gerber statistic.

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

The denominator is that of [`Gerber0`](@ref) plus ``n_{n}``. It is never smaller, so the magnitude of the statistic is never larger. The two statistics are equal when ``n_{n} = 0``.

The sum ``n_{c} + n_{d} + n_{n}`` is the number of observations on which at least one asset crosses. It equals ``T - n^{NN}``, where ``n^{NN}`` is the number of observations on which neither asset crosses. Equation 11 of [gerber](@cite) states the denominator in that form, and Equation 2 of [gerber_analysis](@cite) repeats it. Both identify it as the statistic of the final version of the work.

# Algorithm

The branch of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Add the concordant, discordant and neutral counts into the denominator `den`.
 2. Return `zero(T)` when `den` is zero. An asset that never crosses its own threshold gives a zero `den` for every pair that contains it.
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

  - $(ref_dict[:gerber]) Section 2.2, Equation 11.
  - $(ref_dict[:gerber_analysis]) Section 2, Equation 2.
"""
struct Gerber1 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Divides each net co-movement count by the geometric mean of the crossing counts of the two assets.

# Mathematical definition

```math
\\begin{align}
h_{i,\\,j} &= n_{c} - n_{d}\\,, \\\\
\\rho_{i,\\,j} &= \\frac{h_{i,\\,j}}{\\sqrt{h_{i,\\,i} \\, h_{j,\\,j}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rho_ij])
  - ``h_{i,\\,j}``: Net co-movement count of the pair, before the division.
  - $(math_dict[:nc_gerber])
  - $(math_dict[:nd_gerber])

The denominator comes from the diagonal of the whole matrix, not from a count of the pair alone. [`Gerber0`](@ref) and [`Gerber1`](@ref) divide by a count of the pair. An asset moves in the same direction as itself at each of its crossings, so ``h_{i,\\,i}`` is the number of crossings of asset ``i``. The diagonal of the result is therefore one by construction.

The denominator is the geometric mean of the number of observations on which asset ``i`` crosses and the number on which asset ``j`` crosses. Footnote 3 of [gerber](@cite) states this denominator as the alternative of the 2019 preprint, and states that the covariance is positive semidefinite. The footnote misprints one count of its second factor. Equation 1 of [gerber_analysis](@cite) states the correct count and gives the same attribution. [gerber](@cite) is the final version of that preprint, as its footnote 1 states. On a random sample of four assets, the footnote with the correct count gives the entries of the library to within `2.3e-16`.

# Algorithm

The branch of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this tag selects runs these steps.

 1. Return the difference `p - n` for every pair, with no denominator.
 2. Divide the assembled matrix by the outer product of the square roots of its own diagonal. [`standardise_comovement!`](@ref) clamps each root from below at `sqrt(eps(eltype(rho)))`, so an asset that never crosses gives a zero row and no division by zero.

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

  - $(ref_dict[:gerber]) Section 2.2, footnote 3, and footnote 1.
  - $(ref_dict[:gerber_analysis]) Section 2, Equation 1.
"""
struct Gerber2 <: GerberCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Estimates a covariance matrix from how often the two assets of each pair cross their thresholds together.

Each entry of a Gerber matrix is a separate statistic of one pair, so the matrix is not positive definite in general. Equation 4 of [gerber](@cite) can give a matrix that is not positive semidefinite. The `pdm` field is the repair of the library, not of the source. It projects the result onto the nearest positive definite matrix, and `pdm = nothing` returns the statistic with no repair.

The threshold `t` is the fraction ``c`` of Equation 2 of [gerber](@cite). An asset crosses when its centred return is at least `t` standard deviations from zero. Section 2.1 of [gerber](@cite) calls 1/2 typical and names 7/10 and 9/10 as alternatives, and the default `t = 0.5` follows it. The covariance is the correlation scaled by the standard deviations, as in Equation 9.

**The library compares the centred return with the threshold.** Equation 1 of [gerber](@cite) compares the return itself with the threshold. Footnote 4 of [gerber_analysis](@cite) states that the threshold is centred about zero, not about the mean of each asset. This estimator subtracts the mean that `me` computes before it compares. On a sample of 200 observations of five assets with unit standard deviation and a mean of 0.1, the entries differ from the uncentred statistic by up to 0.06 under [`Gerber0`](@ref), 0.03 under [`Gerber1`](@ref) and 0.04 under [`Gerber2`](@ref). `me = CustomValueExpectedReturns(; val = 0.0)` gives the statistic of the source.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GerberCovariance(;
        ve::StatsBase.CovarianceEstimator = SimpleVariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        t::Number = 0.5,
        alg::GerberCovarianceAlgorithm = Gerber1(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> GerberCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:gerbt])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated via [`factory`](@ref).
  - `me`: Recursively updated via [`factory`](@ref).
  - `alg`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

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

  - $(ref_dict[:gerber]) Section 2, Equations 1, 2 and 9.
  - $(ref_dict[:gerber_analysis]) Footnote 4.
"""
@propagatable @concrete struct GerberCovariance <: BaseGerberCovariance
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:me]) The estimator subtracts its mean from the returns before the threshold test.
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
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function GerberCovariance(ve::StatsBase.CovarianceEstimator,
                              me::AbstractExpectedReturnsEstimator,
                              pdm::Option{<:AbstractPosdefEstimator}, t::Number,
                              alg::GerberCovarianceAlgorithm,
                              cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(t, :t)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(t), typeof(alg),
                   typeof(cache)}(ve, me, pdm, t, alg, cache)
    end
end
function GerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                          me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                          pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                          t::Number = 0.5, alg::GerberCovarianceAlgorithm = Gerber1(),
                          cache::Option{<:AbstractPartialFitState} = nothing)::GerberCovariance
    return GerberCovariance(ve, me, pdm, t, alg, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`GerberCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Without this method, that state prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:GerberCovariance, true)` to render it.

# Arguments

  - `::GerberCovariance`: Gerber covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ve, :me, :pdm, :t, :alg)`.

# Related

  - [`GerberCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::GerberCovariance) = (:ve, :me, :pdm, :t, :alg)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the up and down indicator matrices that every Gerber correlation algorithm uses.

# Mathematical definition

```math
\\begin{align}
U_{t,\\,i} &= \\mathbf{1}[x_{t,\\,i} \\geq t \\, \\sigma_i \\land x_{t,\\,i} > 0]\\,, \\\\
D_{t,\\,i} &= \\mathbf{1}[x_{t,\\,i} \\leq -t \\, \\sigma_i \\land x_{t,\\,i} < 0]\\,.
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

The two bands never overlap. No observation has a mark in both matrices, so their sum is the crossing indicator. At a zero band edge, the sign test keeps the two bands apart. The threshold `ce.t = 0` gives a zero band edge, and a zero entry of `sd` gives one too. The `Statistics.cor` method prevents the second case, because it raises each entry of `sd` to at least `eps`. Without the sign test, both closed comparisons hold on a return of exactly zero, and the sum is two, not one.

For a positive band edge, the sign test changes nothing, because ``x_{t,\\,i} \\geq t \\, \\sigma_i > 0`` already implies ``x_{t,\\,i} > 0``. At a zero band edge, ``\\mathbf{U} - \\mathbf{D}`` is the sign of the return, and the Gerber statistic measures how often the signs of two assets agree.

# Algorithm

 1. Multiply the standard deviation vector by the threshold to get the band edge of each asset, `ts = sd * ce.t`.
 2. Mark `U[t, i]` when `X[t, i] >= ts[i]` and `X[t, i]` is positive.
 3. Mark `D[t, i]` when `X[t, i] <= -ts[i]` and `X[t, i]` is negative.

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
    # The sign test is redundant for a positive band edge, because `X >= ts > 0` already
    # implies `X > 0`. It binds only when the edge is zero, where the two closed
    # comparisons would otherwise both hold on an exactly zero return. ADR 0090.
    zx = zero(eltype(X))
    U .= (X .>= ts) .& (X .> zx)
    D .= (X .<= -ts) .& (X .< zx)
    return U, D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Recover the concordant and the discordant counts from their difference and their sum.

The [`gerber`](@ref) methods get the difference and the sum from matrix products. This function gets the two counts from them with no more matrix products. The arithmetic is exact, so [`comovement_ratio`](@ref) receives the same numerator and denominator as the matrix formula.

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

Compute the Gerber correlation matrix with the denominator of [`Gerber0`](@ref).

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

The entry of ``\\mathbf{H}^\\intercal \\mathbf{H}`` is ``n_{c} - n_{d}`` and the entry of ``\\mathbf{V}^\\intercal \\mathbf{V}`` is ``n_{c} + n_{d}``, so this is the pairwise statistic of [`Gerber0`](@ref) in matrix form. Equations 5 to 8 of [gerber](@cite) give the same matrix form.

# Algorithm

 1. Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
 2. Form the signed crossing matrix `UmD = U - D` and the crossing matrix `UpD = U + D`.
 3. Recover the concordant count `nconc` and the discordant count `ndisc` from `transpose(UmD) * UmD` and `transpose(UpD) * UpD` with [`concordance_counts`](@ref).
 4. Reduce every pair to its statistic with [`comovement_ratio`](@ref). The result is `rho`.
 5. Write one onto a zero diagonal entry of `rho` with [`comovement_unit_diagonal!`](@ref). An asset that never crosses its threshold has a zero diagonal entry, and that entry is one by definition.
 6. Repair `rho` with `posdef!`. It does nothing when `ce.pdm` is `nothing`.

# Arguments

  - $(arg_dict[:gerbce]) Its `alg` field is a `Gerber0`.
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
  - [`comovement_unit_diagonal!`](@ref)
  - [`posdef!`](@ref)

# References

  - $(ref_dict[:gerber]) Section 2.2, Equations 5 to 8.
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum,
                sd::ArrNum)
    U, D = gerber_updown(ce, X, sd)
    UmD = U - D
    UpD = U + D
    nconc, ndisc = concordance_counts(transpose(UmD) * UmD, transpose(UpD) * UpD)
    rho = comovement_ratio.(Ref(ce.alg), nconc, ndisc, 0, eltype(X))
    comovement_unit_diagonal!(rho)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(
        ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber1},
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Compute the Gerber correlation matrix with the denominator of [`Gerber1`](@ref).

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

The entry of ``\\mathbf{N}^\\intercal \\mathbf{N}`` counts the observations on which neither asset crosses. The denominator therefore counts the observations on which at least one of them crosses. That is ``n_{c} + n_{d} + n_{n}``, the pairwise denominator of [`Gerber1`](@ref), and ``T - n^{NN}`` of Equation 11 of [gerber](@cite). The source states no matrix form of this denominator, so the product ``\\mathbf{N}^\\intercal \\mathbf{N}`` is the form of the library.

# Algorithm

 1. Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
 2. Form the neutral matrix `Nt`, which marks the observations on which the asset crosses in neither direction.
 3. Form `NtN = transpose(Nt) * Nt`, the count of observations on which both assets of a pair are neutral. Form `nneutral`, the count of neutral observations of each asset.
 4. Form the signed crossing matrix `UmD = U - D`.
 5. Recover the concordant count `nconc` and the discordant count `ndisc` with [`concordance_counts`](@ref). The difference is `transpose(UmD) * UmD`. The sum is `T .- nneutral .- transpose(nneutral) .+ NtN`, the count of observations on which both assets cross.
 6. Form `nneut = nneutral .+ transpose(nneutral) .- 2 .* NtN`, the count of observations on which exactly one asset of the pair crosses.
 7. Reduce every pair to its statistic with [`comovement_ratio`](@ref). The result is `rho`.
 8. Write one onto a zero diagonal entry of `rho` with [`comovement_unit_diagonal!`](@ref). An asset that never crosses its threshold has a zero diagonal entry, and that entry is one by definition.
 9. Repair `rho` with `posdef!`. It does nothing when `ce.pdm` is `nothing`.

# Arguments

  - $(arg_dict[:gerbce]) Its `alg` field is a `Gerber1`.
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
  - [`comovement_unit_diagonal!`](@ref)
  - [`posdef!`](@ref)

# References

  - $(ref_dict[:gerber]) Section 2.2, Equation 11.
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
    comovement_unit_diagonal!(rho)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(
        ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2},
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Compute the Gerber correlation matrix with the denominator of [`Gerber2`](@ref).

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
  - ``\\mathbf{G}``: Net co-movement matrix before the division. Its entry is ``n_{c} - n_{d}``.
  - ``\\boldsymbol{g}``: Vector of the square roots of the diagonal of ``\\mathbf{G}``.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:t_threshold])
  - $(math_dict[:sigma_i_asset])
  - $(math_dict[:oslash])

The diagonal of ``\\mathbf{G}`` counts the crossings of each asset, so the diagonal of ``\\hat{\\boldsymbol{\\rho}}`` is one by construction. Entry ``(i, j)`` is the net co-movement count divided by the geometric mean of the crossing counts of the two assets. This is Equation 1 of [gerber_analysis](@cite) and footnote 3 of [gerber](@cite). The matrix form is the form of the library.

# Algorithm

 1. Build the indicator matrices `U` and `D` with [`gerber_updown`](@ref).
 2. Form the signed crossing matrix `UmD = U - D`.
 3. Form the net co-movement matrix `rho = transpose(UmD) * UmD`.
 4. Normalise `rho` in place with [`standardise_comovement!`](@ref).
 5. Write one onto a zero diagonal entry of `rho` with [`comovement_unit_diagonal!`](@ref). An asset that never crosses its threshold has a zero diagonal entry of ``\\mathbf{G}``. The clamp of step 4 keeps that entry at zero, and the entry is one by definition.
 6. Repair `rho` with `posdef!`. It does nothing when `ce.pdm` is `nothing`.

# Arguments

  - $(arg_dict[:gerbce]) Its `alg` field is a `Gerber2`.
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber2`](@ref)
  - [`gerber_updown`](@ref)
  - [`standardise_comovement!`](@ref)
  - [`comovement_unit_diagonal!`](@ref)
  - [`posdef!`](@ref)

# References

  - $(ref_dict[:gerber]) Section 2.2, footnote 3.
  - $(ref_dict[:gerber_analysis]) Section 2, Equation 1.
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum,
                sd::ArrNum)
    U, D = gerber_updown(ce, X, sd)
    UmD = U - D
    rho = Matrix{eltype(X)}(transpose(UmD) * UmD)
    standardise_comovement!(ce.alg, rho)
    comovement_unit_diagonal!(rho)
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

Compute the Gerber correlation matrix with the algorithm in `ce.alg`.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute the standard deviation vector `sd` with `ce.ve`, and raise each entry to at least `eps(eltype(sd))`. A constant column has a standard deviation and a demeaned residual of the same round-off order. Without the floor, the threshold marks every observation of that column as a crossing. With the floor, it marks none of them.
 3. Centre the returns with `ce.me` through [`demean_returns`](@ref).
 4. Return the Gerber correlation matrix from [`gerber`](@ref), through the branch that `ce.alg` selects.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for the standard deviation estimator and for [`demean_returns`](@ref).

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

  - $(ref_dict[:gerber]) Section 2.
"""
function Statistics.cor(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
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

Compute the Gerber covariance matrix with the algorithm in `ce.alg`.

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

The Gerber statistic gives only the correlations. The variances come from `ce.ve`, so the diagonal of ``\\hat{\\mathbf{\\Sigma}}`` is ``\\boldsymbol{\\sigma}^2``. This is Equation 9 of [gerber](@cite), with the estimator `ce.ve` in place of the sample standard deviation.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute the standard deviation vector `sd` with `ce.ve`, and raise each entry to at least `eps(eltype(sd))`. A constant column has a standard deviation and a demeaned residual of the same round-off order. Without the floor, the threshold marks every observation of that column as a crossing. With the floor, it marks none of them.
 3. Centre the returns with `ce.me` through [`demean_returns`](@ref).
 4. Compute the Gerber correlation matrix `sigma` with [`gerber`](@ref), through the branch that `ce.alg` selects.
 5. Rescale `sigma` in place to a covariance matrix with `StatsBase.cor2cov!` and `sd`.

# Arguments

  - $(arg_dict[:gerbce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for the standard deviation estimator and for [`demean_returns`](@ref).

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

  - $(ref_dict[:gerber]) Section 2.2, Equation 9.
"""
function Statistics.cov(ce::GerberCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    sigma = gerber(ce, X, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export GerberCovariance, Gerber0, Gerber1, Gerber2
public BaseGerberCovariance, GerberCovarianceAlgorithm
