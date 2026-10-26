"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance estimators.

Every concrete or abstract type that implements a Smyth-Broby covariance estimator must subtype `BaseSmythBrobyCovariance`. It subtypes [`BaseGerberCovariance`](@ref), because a Smyth-Broby statistic is a Gerber statistic with a second zone. Each observation adds a real-valued contribution to it, not a count of one. [`GerberCovariance`](@ref) states the Gerber statistic, and this docstring does not repeat it.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovariance`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2.
"""
abstract type BaseSmythBrobyCovariance <: BaseGerberCovariance end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance algorithm types.

Every concrete or abstract type that implements a Smyth-Broby covariance algorithm must subtype `SmythBrobyCovarianceAlgorithm`. A subtype goes in the `alg` field of a [`SmythBrobyCovariance`](@ref) estimator.

A marker makes two independent choices. Its prefix selects the score triple through [`sb_pair_scores`](@ref). Its trailing digit selects the denominator through [`comovement_ratio`](@ref), and the digit has the same meaning as in the Gerber family.

# Related

  - [`BaseSmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyDeltaAlg`](@ref)
  - [`SmythBrobyGerberAlg`](@ref)
  - [`SmythBrobyCountAlg`](@ref)
  - [`GerberComovementZero`](@ref)
  - [`GerberComovementOne`](@ref)
  - [`GerberComovementTwo`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equation 5.
"""
abstract type SmythBrobyCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Divides the difference of the concordant and discordant Smyth-Broby contribution sums by their sum.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (\\mathrm{pos},\\, \\mathrm{neg},\\, \\mathrm{nn})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{p - q}{p + q}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:possum_sb])
  - $(math_dict[:rho_ij])

The source defines one statistic, [`SmythBroby1`](@ref), and that statistic keeps the neutral sum in the denominator. This marker removes the neutral term, as [`Gerber0`](@ref) does. It is a reduction of the library, not a formulation of the source.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this marker selects runs these steps.

 1. Read the contribution sums `(pos, neg, nn)` from the pair accumulator as the scores `(p, q, u)`. The counts are not read.
 2. Return `(p - q) / (p + q)`, or `zero(T)` when the denominator is zero.

# Constructors

    SmythBroby0() -> SmythBroby0

# Examples

```jldoctest
julia> SmythBroby0()
SmythBroby0()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equation 5.
"""
struct SmythBroby0 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Divides the difference of the concordant and discordant Smyth-Broby contribution sums by their sum plus the neutral sum.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (\\mathrm{pos},\\, \\mathrm{neg},\\, \\mathrm{nn})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{p - q}{p + q + u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:possum_sb])
  - $(math_dict[:rho_ij])

This is Equation 5 of [smyth2022enhanced](@cite), and it is the only statistic that the source defines. The numerator is the concordant sum minus the discordant sum over the lookback window. The denominator adds the concordant, discordant and neutral sums. The neutral zone is the four regions DN, NU, UN and ND of Section 2 of the source. These are the observations on which exactly one asset leaves the indecision zone.

Section 2 of the source states that the neutral sum keeps the covariance matrix positive semidefinite, and its footnote 2 states that this has no proof. The claim is false. On a sample of 30 observations of 20 standard normal returns, the matrix of this marker has a negative eigenvalue. The `pdm` field of [`SmythBrobyCovariance`](@ref) repairs it.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this marker selects runs these steps.

 1. Read the contribution sums `(pos, neg, nn)` from the pair accumulator as the scores `(p, q, u)`. The counts are not read.
 2. Return `(p - q) / (p + q + u)`, or `zero(T)` when the denominator is zero.

# Constructors

    SmythBroby1() -> SmythBroby1

# Examples

```jldoctest
julia> SmythBroby1()
SmythBroby1()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby2`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equation 5.
"""
struct SmythBroby1 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Normalises the net Smyth-Broby contribution of a pair by the geometric mean of its own diagonal.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (\\mathrm{pos},\\, \\mathrm{neg},\\, \\mathrm{nn})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{h_{i,\\,j}}{\\sqrt{h_{i,\\,i} \\, h_{j,\\,j}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:possum_sb])
  - $(math_dict[:h_ij_sb])
  - $(math_dict[:rho_ij])

The denominator comes from the diagonal of the whole matrix, not from the sums of the pair alone. [`SmythBroby0`](@ref) and [`SmythBroby1`](@ref) divide by the sums of the pair. The diagonal of the result is therefore one by construction. The result is not the ratio of [`SmythBroby0`](@ref) after a second division. The two are equal only where ``p + q`` is the same for every pair. The source defines no such denominator, so this marker is a reduction of the library, as [`Gerber2`](@ref) is for the Gerber statistic.

# Algorithm

The branch of [`sb_pair_scores`](@ref), of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this marker selects runs these steps.

 1. Read the contribution sums `(pos, neg, nn)` from the pair accumulator as the scores `(p, q, u)`.
 2. Return the net score `p - q` for every pair, with no denominator.
 3. Divide the assembled matrix by the outer product of the square roots of its own diagonal. [`standardise_comovement!`](@ref) clamps each root from below at `sqrt(eps(eltype(rho)))`, so an asset that admits no observation gives a zero row and no division by zero.

# Constructors

    SmythBroby2() -> SmythBroby2

# Examples

```jldoctest
julia> SmythBroby2()
SmythBroby2()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equations 1 to 4.
"""
struct SmythBroby2 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Weights each Smyth-Broby contribution sum by its own observation count, then divides the difference by the sum.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (\\mathrm{pos} \\, c^{+},\\, \\mathrm{neg} \\, c^{-},\\, \\mathrm{nn} \\, c^{0})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{p - q}{p + q}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:possum_sb])
  - $(math_dict[:poscount_sb])
  - $(math_dict[:rho_ij])

The counts are the Gerber co-movement counts that [`SmythBroby0`](@ref) does not read. A pair that moves together often therefore scores above a pair that moves together rarely but by large amounts. This marker removes the neutral term, as [`SmythBroby0`](@ref) does.

Section 5 of [smyth2022enhanced](@cite) suggests a count beside each sum, and a scale on the total contribution to match the total count. The scale keeps the split between the concordant, discordant and neutral contributions. Section 5 states no formula. One factor on all three sums does not change the ratio of Equation 5. The product of each sum with its own count is therefore a choice of the library, and it does not follow from Section 5.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this marker selects runs these steps.

 1. Multiply each of `pos`, `neg` and `nn` by its own count `cpos`, `cneg` and `cnn`. The products are the scores `(p, q, u)`.
 2. Return `(p - q) / (p + q)`, or `zero(T)` when the denominator is zero.

# Constructors

    SmythBrobyGerber0() -> SmythBrobyGerber0

# Examples

```jldoctest
julia> SmythBrobyGerber0()
SmythBrobyGerber0()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equations 4 and 5, and Section 5.
"""
struct SmythBrobyGerber0 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Weights each Smyth-Broby contribution sum by its own count, then divides the difference by the sum plus the neutral term.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (\\mathrm{pos} \\, c^{+},\\, \\mathrm{neg} \\, c^{-},\\, \\mathrm{nn} \\, c^{0})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{p - q}{p + q + u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:possum_sb])
  - $(math_dict[:poscount_sb])
  - $(math_dict[:rho_ij])

The neutral sum also gets its own count, so each of the three terms of the denominator is a product of a sum and a count. This is the default algorithm of the estimator. It applies the count that Section 5 of the source suggests to the statistic of the source, [`SmythBroby1`](@ref). The source states no formula for that product.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this marker selects runs these steps.

 1. Multiply each of `pos`, `neg` and `nn` by its own count `cpos`, `cneg` and `cnn`. The products are the scores `(p, q, u)`.
 2. Return `(p - q) / (p + q + u)`, or `zero(T)` when the denominator is zero.

# Constructors

    SmythBrobyGerber1() -> SmythBrobyGerber1

# Examples

```jldoctest
julia> SmythBrobyGerber1()
SmythBrobyGerber1()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber2`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equations 4 and 5, and Section 5.
"""
struct SmythBrobyGerber1 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Weights each Smyth-Broby contribution sum by its own count, then normalises the net score by the geometric mean of its own diagonal.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (\\mathrm{pos} \\, c^{+},\\, \\mathrm{neg} \\, c^{-},\\, \\mathrm{nn} \\, c^{0})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{h_{i,\\,j}}{\\sqrt{h_{i,\\,i} \\, h_{j,\\,j}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:possum_sb])
  - $(math_dict[:poscount_sb])
  - $(math_dict[:h_ij_sb])
  - $(math_dict[:rho_ij])

The denominator comes from the diagonal of the whole matrix, as in [`SmythBroby2`](@ref). The source defines no such denominator, so this marker is a reduction of the library, as [`Gerber2`](@ref) is for the Gerber statistic.

# Algorithm

The branch of [`sb_pair_scores`](@ref), of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this marker selects runs these steps.

 1. Multiply each of `pos`, `neg` and `nn` by its own count `cpos`, `cneg` and `cnn`. The products are the scores `(p, q, u)`.
 2. Return the net score `p - q` for every pair, with no denominator.
 3. Divide the assembled matrix by the outer product of the square roots of its own diagonal. [`standardise_comovement!`](@ref) clamps each root from below at `sqrt(eps(eltype(rho)))`, so an asset that admits no observation gives a zero row and no division by zero.

# Constructors

    SmythBrobyGerber2() -> SmythBrobyGerber2

# Examples

```jldoctest
julia> SmythBrobyGerber2()
SmythBrobyGerber2()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equations 4 and 5, and Section 5.
"""
struct SmythBrobyGerber2 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Counts concordant and discordant observations, discards the contribution sums, and divides their difference by their sum.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (c^{+},\\, c^{-},\\, c^{0})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{p - q}{p + q}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:poscount_sb])
  - $(math_dict[:rho_ij])

Without [`sb_delta`](@ref), the statistic is a Gerber statistic on the zones of Smyth-Broby, not on the Gerber threshold. This marker gives the same matrix as [`Gerber0`](@ref) when `c1` is zero, `c2` is the Gerber threshold `t`, `c3` is above every return, and the mean of each asset is zero. Equation 5 of the source adds contributions and keeps no count, because the source argues that a contribution sum holds more information than a count. The count family is therefore a construction of the library, not a formulation of the source.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this marker selects runs these steps.

 1. Read the counts `(cpos, cneg, cnn)` from the pair accumulator as the scores `(p, q, u)`. The contribution sums are not read, and [`sb_delta`](@ref) is not evaluated.
 2. Return `(p - q) / (p + q)`, or `zero(T)` when the denominator is zero.

# Constructors

    SmythBrobyCount0() -> SmythBrobyCount0

# Examples

```jldoctest
julia> SmythBrobyCount0()
SmythBrobyCount0()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCount1`](@ref)
  - [`SmythBrobyCount2`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2.
"""
struct SmythBrobyCount0 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Counts concordant, discordant and neutral observations, discards the contribution sums, and divides the net count by the total.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (c^{+},\\, c^{-},\\, c^{0})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{p - q}{p + q + u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:poscount_sb])
  - $(math_dict[:rho_ij])

Without [`sb_delta`](@ref), the statistic is a Gerber statistic on the zones of Smyth-Broby, not on the Gerber threshold. This marker gives the same matrix as [`Gerber1`](@ref) when `c1` is zero, `c2` is the Gerber threshold `t`, `c3` is above every return, and the mean of each asset is zero. Equation 5 of the source keeps no count, so the count family is a construction of the library, not a formulation of the source.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this marker selects runs these steps.

 1. Read the counts `(cpos, cneg, cnn)` from the pair accumulator as the scores `(p, q, u)`. The contribution sums are not read, and [`sb_delta`](@ref) is not evaluated.
 2. Return `(p - q) / (p + q + u)`, or `zero(T)` when the denominator is zero.

# Constructors

    SmythBrobyCount1() -> SmythBrobyCount1

# Examples

```jldoctest
julia> SmythBrobyCount1()
SmythBrobyCount1()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCount0`](@ref)
  - [`SmythBrobyCount2`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2.
"""
struct SmythBrobyCount1 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Counts concordant and discordant observations, discards the contribution sums, and normalises the net count by the geometric mean of its own diagonal.

# Mathematical definition

```math
\\begin{align}
(p,\\, q,\\, u) &= (c^{+},\\, c^{-},\\, c^{0})\\,, \\\\
\\rho_{i,\\,j} &= \\frac{h_{i,\\,j}}{\\sqrt{h_{i,\\,i} \\, h_{j,\\,j}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pqu_sb])
  - $(math_dict[:poscount_sb])
  - $(math_dict[:h_ij_sb])
  - $(math_dict[:rho_ij])

The denominator comes from the diagonal of the whole matrix, as in [`SmythBroby2`](@ref). This marker gives the same matrix as [`Gerber2`](@ref) when `c1` is zero, `c2` is the Gerber threshold `t`, `c3` is above every return, and the mean of each asset is zero. The source keeps no count and defines no such denominator, so this marker is a construction of the library.

# Algorithm

The branch of [`sb_pair_scores`](@ref), of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this marker selects runs these steps.

 1. Read the counts `(cpos, cneg, cnn)` from the pair accumulator as the scores `(p, q, u)`. The contribution sums are not read, and [`sb_delta`](@ref) is not evaluated.
 2. Return the net score `p - q` for every pair, with no denominator.
 3. Divide the assembled matrix by the outer product of the square roots of its own diagonal. [`standardise_comovement!`](@ref) clamps each root from below at `sqrt(eps(eltype(rho)))`, so an asset that admits no observation gives a zero row and no division by zero.

# Constructors

    SmythBrobyCount2() -> SmythBrobyCount2

# Examples

```jldoctest
julia> SmythBrobyCount2()
SmythBrobyCount2()
```

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCount0`](@ref)
  - [`SmythBrobyCount1`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2.
"""
struct SmythBrobyCount2 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Estimates a covariance matrix from the co-movements of each pair of assets outside a confusion zone and an indecision zone.

Each entry of a Smyth-Broby matrix is a separate ratio of one pair, so the matrix is not positive definite in general. The `pdm` field projects the result onto the nearest positive definite matrix, and `pdm = nothing` returns the statistic with no repair. Of the nine markers, the source defines only [`SmythBroby1`](@ref). The docstring of each other marker states what the library adds to the source. [`smythbroby`](@ref) states the shared statistic once.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SmythBrobyCovariance(;
        ve::StatsBase.CovarianceEstimator = SimpleVariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        c1::Number = 0.5,
        c2::Number = 0.5,
        c3::Number = 4,
        n::Number = 2,
        alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1(),
        ex::FLoops.Transducers.Executor = ThreadedEx(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> SmythBrobyCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:c1])
  - $(val_dict[:c2])
  - $(val_dict[:c3])
  - $(val_dict[:c3c2])
  - $(val_dict[:sbn])
  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `c1`, `c2` and `c3`, so it refuses `Inf` and `NaN`. [`assert_nonneg`](@ref) checks `n`. It refuses a negative `n` and `NaN`, and it accepts `Inf`. At `n = Inf`, [`sb_delta`](@ref) gives zero to each observation whose two magnitudes differ by more than one. A negative `n` reverses the penalty of [`sb_delta`](@ref). An observation with two equal magnitudes then contributes zero, so the diagonal is zero, not one.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated through [`factory`](@ref).
  - `me`: Recursively updated through [`factory`](@ref).
  - `alg`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed through [`port_opt_view`](@ref).
  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> SmythBrobyCovariance()
SmythBrobyCovariance
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
   c1 ┼ Float64: 0.5
   c2 ┼ Float64: 0.5
   c3 ┼ Int64: 4
    n ┼ Int64: 2
  alg ┼ SmythBrobyGerber1()
   ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`BaseSmythBrobyCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`SimpleVariance`](@ref)
  - [`Posdef`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`SmythBrobyCount0`](@ref)
  - [`SmythBrobyCount1`](@ref)
  - [`SmythBrobyCount2`](@ref)
  - [`smythbroby`](@ref)
  - [`GerberCovariance`](@ref): the statistic that this family extends. The confusion zone and the indecision zone replace its threshold.
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equations 1 to 5, and Section 3.
"""
@propagatable @concrete struct SmythBrobyCovariance <: BaseSmythBrobyCovariance
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:me]) Its mean centres the returns before the indecision-zone test and the outer cut-off.
    """
    @fprop @vprop me
    """
    $(field_dict[:pdm])
    """
    pdm
    """
    $(field_dict[:c1])
    """
    c1
    """
    $(field_dict[:c2])
    """
    c2
    """
    $(field_dict[:c3])
    """
    c3
    """
    $(field_dict[:sbn])
    """
    n
    """
    $(field_dict[:sbalg])
    """
    @fprop alg
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function SmythBrobyCovariance(ve::StatsBase.CovarianceEstimator,
                                  me::AbstractExpectedReturnsEstimator,
                                  pdm::Option{<:AbstractPosdefEstimator}, c1::Number,
                                  c2::Number, c3::Number, n::Number,
                                  alg::SmythBrobyCovarianceAlgorithm,
                                  ex::FLoops.Transducers.Executor,
                                  cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(c1, :c1)
        assert_nonempty_nonneg_finite_val(c2, :c2)
        assert_nonempty_nonneg_finite_val(c3, :c3)
        assert_nonneg(n, :n)
        @argcheck(c2 < c3,
                  DomainError((c2, c3),
                              "`SmythBrobyCovariance.c2` is $c2 and `c3` is $c3. `c2` is the inner threshold and `c3` the outer, so the pair must satisfy `c2 < c3`."))
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(c1), typeof(c2), typeof(c3),
                   typeof(n), typeof(alg), typeof(ex), typeof(cache)}(ve, me, pdm, c1, c2,
                                                                      c3, n, alg, ex, cache)
    end
end
function SmythBrobyCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                              me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                              pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                              c1::Number = 0.5, c2::Number = 0.5, c3::Number = 4,
                              n::Number = 2,
                              alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1(),
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                              cache::Option{<:AbstractPartialFitState} = nothing)::SmythBrobyCovariance
    return SmythBrobyCovariance(ve, me, pdm, c1, c2, c3, n, alg, ex, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`SmythBrobyCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Without this method, that state prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:SmythBrobyCovariance, true)` to render it.

# Arguments

  - `::SmythBrobyCovariance`: Smyth-Broby covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ve, :me, :pdm, :c1, :c2, :c3, :n, :alg, :ex)`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::SmythBrobyCovariance) = (:ve, :me, :pdm, :c1, :c2, :c3, :n, :alg, :ex)
"""
    sb_delta(ri::Number, rj::Number, n::Number) -> Number

Contribution of one admitted observation to a Smyth-Broby pair score.

An observation adds this real number to the score of the pair, where the Gerber statistic adds a count of one. A co-movement of four standard deviations therefore adds more than a co-movement of one standard deviation. Both arguments are absolute values, and the caller centres and standardises the returns before it takes them.

# Mathematical definition

```math
\\begin{align}
\\kappa &= \\sqrt{\\left(1 + |\\tilde{r}_{t,\\,i}|\\right) \\left(1 + |\\tilde{r}_{t,\\,j}|\\right)}\\,, \\\\
\\gamma &= \\left\\lVert |\\tilde{r}_{t,\\,i}| - |\\tilde{r}_{t,\\,j}| \\right\\rVert\\,, \\\\
\\delta &= \\frac{\\kappa}{1 + \\gamma^{n}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:r_tilde_sb])
  - $(math_dict[:kappa_sb])
  - $(math_dict[:gamma_sb])
  - $(math_dict[:delta_sb])
  - $(math_dict[:n_sb])

``\\kappa`` increases with the two magnitudes, and ``\\gamma^{n}`` decreases the contribution when the two magnitudes differ. The contribution is therefore largest when both assets move far and by the same amount. ``\\gamma`` is the absolute difference of the two magnitudes, not of the two signed returns. The caller reads the signs first, and the signs put the observation in the concordant or the discordant set.

Equations 1 to 3 of [smyth2022enhanced](@cite) define the contribution for a general severity ``n``. Equation 4 writes it out for ``n = 2``, the value that Section 2 of the source uses. The default `n = 2` of the estimator is that value. The formula above agrees with a direct evaluation of Equation 1 to machine precision.

# Algorithm

 1. Multiply the two gross magnitudes `1 + ri` and `1 + rj`, and take the square root. The result is the amplitude `kappa`.
 2. Take the absolute difference of `ri` and `rj`. The result is the divergence `gamma`.
 3. Return `kappa / (1 + gamma^n)`.

# Arguments

  - `ri`: Absolute centred standardised return of asset `i` at the observation.
  - `rj`: Absolute centred standardised return of asset `j` at the observation.
  - $(arg_dict[:sbn])

# Returns

  - `delta::Number`: The contribution of the observation to the pair's score.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`smythbroby`](@ref)
  - [`sb_add_pos`](@ref)
  - [`sb_add_neg`](@ref)
  - [`sb_add_neutral`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equations 1 to 4.
"""
function sb_delta(ri::Number, rj::Number, n::Number)
    kappa = sqrt((one(ri) + ri) * (one(rj) + rj))
    gamma = abs(ri - rj)
    return kappa / (one(gamma) + gamma^n)
end
"""
$(DocStringExtensions.TYPEDEF)

Union of the Gerber-family markers that divide the net score of a pair by the sum of its concordant and discordant scores.

The statistic is zero when that sum is zero. The trailing `0` has the same meaning in all four families of the Gerber lineage, so one method of [`comovement_ratio`](@ref) serves all four. A new family adds a member to this union, not a new method.

# Related

  - [`Gerber0`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyCount0`](@ref)
  - [`comovement_ratio`](@ref): the method that dispatches on this alias.
  - [`GerberComovementOne`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
const GerberComovementZero = Union{Gerber0, SmythBroby0, SmythBrobyGerber0,
                                   SmythBrobyCount0}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Gerber-family markers that divide the net score of a pair by the sum of its three scores, the neutral score included.

The statistic is zero when that sum is zero. The trailing `1` has the same meaning in all four families of the Gerber lineage, so one method of [`comovement_ratio`](@ref) serves all four. [`Gerber1`](@ref) is the statistic of [gerber](@cite), and [`SmythBroby1`](@ref) is the statistic of [smyth2022enhanced](@cite). Section 2 of [smyth2022enhanced](@cite) states, with no proof, that the neutral term keeps the matrix positive semidefinite. The claim is false for [`SmythBroby1`](@ref) and [`SmythBrobyGerber1`](@ref).

# Related

  - [`Gerber1`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyCount1`](@ref)
  - [`comovement_ratio`](@ref): the method that dispatches on this alias.
  - [`sb_add_neutral`](@ref): the accumulator that fills the neutral score. Only the three Smyth-Broby members of this union reach its acting methods.
  - [`GerberComovementZero`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
const GerberComovementOne = Union{Gerber1, SmythBroby1, SmythBrobyGerber1, SmythBrobyCount1}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Gerber-family markers that keep the net score of each pair and divide the matrix by the geometric mean of its diagonal.

The trailing `2` has the same meaning in all four families of the Gerber lineage. Only this denominator comes from the whole matrix, not from the scores of one pair. These markers are therefore the only ones that reach the acting method of [`standardise_comovement!`](@ref).

# Related

  - [`Gerber2`](@ref)
  - [`SmythBroby2`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`SmythBrobyCount2`](@ref)
  - [`comovement_ratio`](@ref): the method that dispatches on this alias.
  - [`standardise_comovement!`](@ref): the acting method that dispatches on this alias.
  - [`GerberComovementZero`](@ref)
  - [`GerberComovementOne`](@ref)
"""
const GerberComovementTwo = Union{Gerber2, SmythBroby2, SmythBrobyGerber2, SmythBrobyCount2}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reduce the concordant, discordant and neutral scores of a pair to the correlation entry of the pair.

The marker selects the denominator. Here `n` is the discordant score, not the severity exponent of [`sb_delta`](@ref). The two have the same name and no other relation.

# Algorithm

The marker selects one of three branches.

 1. [`GerberComovementZero`](@ref): return `(p - n) / (p + n)`, or `zero(T)` when `p + n` is zero. It does not read `nn`.
 2. [`GerberComovementOne`](@ref): return `(p - n) / (p + n + nn)`, or `zero(T)` when `p + n + nn` is zero. This is the only branch that reads `nn`.
 3. [`GerberComovementTwo`](@ref): return `p - n`, and apply no denominator. [`standardise_comovement!`](@ref) normalises the assembled matrix afterwards.

A zero denominator means that the pair admits no observation. Zero is then the correct value for an off-diagonal entry. It is the wrong value on the diagonal, where a correlation is one by definition. This function does not know whether the pair is `(i, i)`, so it cannot tell the two cases apart. [`comovement_unit_diagonal!`](@ref) writes the diagonal after the matrix is assembled.

# Arguments

  - `alg`: Co-movement algorithm marker.
  - `p`, `n`, `nn`: Accumulated concordant, discordant and neutral scores of one pair.
  - `T`: Element type of the zero that a zero denominator gives.

# Returns

  - The pairwise co-movement statistic.

# Related

  - [`gerber_comovement!`](@ref)
  - [`comovement_finalise`](@ref): the caller that reaches this function once per pair.
  - [`standardise_comovement!`](@ref)
  - [`comovement_unit_diagonal!`](@ref): the function that writes one over a zero on the diagonal.
  - [`GerberComovementZero`](@ref)
  - [`GerberComovementOne`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
function comovement_ratio(::GerberComovementZero, p::Number, n::Number, nn::Number,
                          ::Type{T}) where {T}
    den = p + n
    return !iszero(den) ? (p - n) / den : zero(T)
end
function comovement_ratio(::GerberComovementOne, p::Number, n::Number, nn::Number,
                          ::Type{T}) where {T}
    den = p + n + nn
    return !iszero(den) ? (p - n) / den : zero(T)
end
function comovement_ratio(::GerberComovementTwo, p::Number, n::Number, nn::Number,
                          ::Type{T}) where {T}
    return p - n
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Normalise a net co-movement matrix in place by the geometric mean of its own diagonal.

Only the [`GerberComovementTwo`](@ref) markers reach the acting method. The fall-through method does nothing, so a caller can call this function for every marker. It writes into `rho` and into nothing else. The marker and the estimator that holds the marker do not change.

The Gerber IQ family does not call this function. When `sc` is not pair-separable, the thresholds of an asset change from one pair to the next. The magnitude class of an asset in a pair is then not the class that the diagonal records, and the ratio can leave `[-1, 1]`. [`gerber_IQ`](@ref) divides by the two diagonal projections of the pair instead, which [`iq_add_diagonal`](@ref) accumulates. The other three families compare each asset with a threshold in its own units. The diagonal entry of an asset is then correct for every pair, so this function gives the correct result.

# Algorithm

The acting method runs these steps. The fall-through method runs none of them.

 1. Take the square roots of the diagonal of `rho`, and clamp them from below at `sqrt(eps(eltype(rho)))`. The result is `h`. The clamp prevents a division by zero for an asset that admits no observation.
 2. Divide `rho` element-wise by the outer product `h * transpose(h)`, and copy the upper triangle into the lower triangle.

# Arguments

  - `alg`: Co-movement algorithm marker. It selects the acting method or the method that does nothing.
  - `rho`: `N × N` co-movement matrix, overwritten.

# Returns

  - `nothing`.

# Related

  - [`comovement_ratio`](@ref): the reduction whose [`GerberComovementTwo`](@ref) branch leaves the net score for this function to normalise.
  - [`comovement_unit_diagonal!`](@ref): the repair of the diagonal that runs after this function.
  - [`gerber_comovement!`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
function standardise_comovement!(::Any, ::AbstractMatrix)
    return nothing
end
function standardise_comovement!(::GerberComovementTwo, rho::AbstractMatrix)
    h = max.(sqrt.(LinearAlgebra.diag(rho)), sqrt(eps(eltype(rho))))
    rho .= LinearAlgebra.Symmetric(rho ⊘ (h * transpose(h)), :U)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Write one onto a zero diagonal entry of an assembled co-movement matrix, in place.

The correlation of an asset with itself is one by definition, for every sample. Each reduction of [`comovement_ratio`](@ref) returns one on the diagonal when the asset crosses its threshold at least once. This function corrects the other case. An asset that never crosses its threshold gives a zero denominator for every pair that contains it. Its whole row is then `zero(T)`, and its diagonal entry is `zero(T)` too. The function writes into `rho` and into nothing else, and it changes no other entry.

The function writes only where `iszero` holds, so it does not write over a diagonal entry that is near one. A `2` marker divides the diagonal by its own square root twice. Its diagonal entry is then within a unit in the last place of one, but not always exactly one. [`posdef!`](@ref) tests its diagonal with an exact `isone` to decide whether it holds a correlation matrix or a covariance matrix, and the two branches give different results. An exact one written over such an entry changes the branch, and so changes the result for a sample with no degenerate asset. The `iszero` test limits this function to the defect that it corrects.

# Algorithm

 1. For each index `i` of the diagonal, write `one(eltype(rho))` onto `rho[i, i]` when that entry is zero.

# Arguments

  - `rho`: `N × N` co-movement matrix, whose zero diagonal entries are overwritten.

# Returns

  - `nothing`.

# Related

  - [`comovement_ratio`](@ref): the reduction that gives zero on the diagonal when the denominator is zero.
  - [`standardise_comovement!`](@ref): the division by the diagonal that runs before this function.
  - [`gerber_comovement!`](@ref)
  - [`posdef!`](@ref): the repair that runs after this function. It keeps a zero diagonal entry and its zero row, so it does not write the one.
"""
function comovement_unit_diagonal!(rho::AbstractMatrix)
    o = one(eltype(rho))
    for i in axes(rho, 1)
        if iszero(rho[i, i])
            rho[i, i] = o
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fill the symmetric co-movement matrix `rho` with the pairwise loop that the Gerber-family policies share.

The policy object `pol` is a [`SmythBrobyKernel`](@ref) or a [`GerberIQKernel`](@ref). The policy tests one observation against the thresholds, gives it a class and a weight, and reduces the accumulator of a pair. This function holds the loop over the pairs and the observations. It writes into `rho` and into nothing else, so `X` and `pol` do not change.

# Algorithm

 1. Read the observation count `T` from the first dimension of `X`.
 2. For every asset pair `(i, j)` with `i` at most `j`, run steps 3 to 6. The executor `ex` runs the outer index in parallel.
 3. Build the pair state `st` with [`comovement_pair_state`](@ref).
 4. Open the accumulator `acc` as the named tuple `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. The five scores start at zero of `eltype(X)`, and the three counts start at integer zero. A policy reads the slots that its marker needs, and the other slots stay at zero. `di` and `dj` hold the two diagonal projections that the Gerber IQ `2` marker divides by. Only [`iq_add_diagonal`](@ref) writes them.
 5. Add each observation `k` of the pair into `acc` with [`comovement_step`](@ref).
 6. Reduce `acc` with [`comovement_finalise`](@ref), and write the result into `rho[i, j]` and `rho[j, i]`.

# Arguments

  - `rho::AbstractMatrix`: `N × N` output matrix, overwritten.
  - `ex`: `FLoops` executor that runs the outer asset index in parallel.
  - $(arg_dict[:X])
  - `pol`: Co-movement policy object.

# Returns

  - `nothing`.

# Related

  - [`comovement_pair_state`](@ref)
  - [`comovement_step`](@ref)
  - [`comovement_finalise`](@ref)
  - [`standardise_comovement!`](@ref)
"""
function gerber_comovement!(rho::AbstractMatrix, ex::FLoops.Transducers.Executor, X::MatNum,
                            pol)
    T = size(X, 1)
    FLoops.@floop ex for j in axes(X, 2)
        for i in 1:j
            st = comovement_pair_state(pol, i, j)
            acc = (pos = zero(eltype(X)), neg = zero(eltype(X)), nn = zero(eltype(X)),
                   cpos = 0, cneg = 0, cnn = 0, di = zero(eltype(X)), dj = zero(eltype(X)))
            for k in 1:T
                acc = comovement_step(pol, acc, st, X[k, i], X[k, j], T, k)
            end
            rho[j, i] = rho[i, j] = comovement_finalise(pol, acc, eltype(X))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Co-movement policy of the Smyth-Broby family for the shared pairwise loop.

[`gerber_comovement!`](@ref) takes this policy. The confusion zone compares the raw, uncentred return with `c1 * sigma`. The policy centres and standardises each observation that passes the confusion zone. The indecision zone of `c2`, through [`sb_crossed`](@ref), and the outer cut-off at `c3` then admit or reject the observation. The sign of the product of the standardised returns gives the class of an admitted observation.

The `alg` marker selects the accumulation through [`sb_add_pos`](@ref) and the denominator through [`comovement_ratio`](@ref). This type is configuration. It holds no result, and nothing changes it.

Section 2 of [smyth2022enhanced](@cite) defines the three zones:

  - The confusion zone, of half-width ``c_1`` standard deviations.
  - The indecision zone, of half-width ``c_2``.
  - The outer cut-off at ``c_3``, which the source sets to 4.

Section 3 of the source uses ``c_1 = c_2 = 0.5``, and these values are the defaults. The source keeps a magnitude strictly below the cut-off. The library rejects only a magnitude above `c3`, so it keeps a magnitude equal to `c3`.

# Fields

  - $(arg_dict[:sbalg])
  - `mu`: Vector of asset means, one entry per asset.
  - $(arg_dict[:stdarr])
  - $(arg_dict[:c1])
  - $(arg_dict[:c2])
  - $(arg_dict[:c3])
  - $(arg_dict[:sbn])

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`smythbroby`](@ref): the caller that builds this policy.
  - [`gerber_comovement!`](@ref)
  - [`comovement_pair_state`](@ref)
  - [`comovement_step`](@ref)
  - [`comovement_finalise`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2 and Section 3.
"""
@concrete struct SmythBrobyKernel
    alg
    mu
    sd
    c1
    c2
    c3
    n
    function SmythBrobyKernel(alg::T1, mu::T2, sd::T3, c1::T4, c2::T5, c3::T6,
                              n::T7) where {T1 <: SmythBrobyCovarianceAlgorithm,
                                            T2 <: ArrNum, T3 <: ArrNum, T4 <: Number,
                                            T5 <: Number, T6 <: Number, T7 <: Number}
        return new{T1, T2, T3, T4, T5, T6, T7}(alg, mu, sd, c1, c2, c3, n)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the state of a pair that [`comovement_step`](@ref) reads.

The state holds each quantity that depends on the pair but not on the observation, so the inner loop does not compute it again. The Gerber IQ method returns the threshold scale factors and the scaled thresholds of the pair instead.

# Algorithm

The Smyth-Broby method runs these steps.

 1. Read the two standard deviations `pol.sd[i]` and `pol.sd[j]` as `sigmai` and `sigmaj`.
 2. Multiply each by `pol.c1`. The products are the two confusion-zone thresholds `c1i` and `c1j`.
 3. Return the named tuple `(mui, muj, sigmai, sigmaj, c1i, c1j)`, with the two means read from `pol.mu`.

# Arguments

  - `pol`: Co-movement policy object.
  - `i`, `j`: Indices of the two assets of the pair.

# Returns

  - The per-pair state, as a named tuple.

# Related

  - [`gerber_comovement!`](@ref)
  - [`comovement_step`](@ref): the function that reads this state.
  - [`SmythBrobyKernel`](@ref)
"""
@inline function comovement_pair_state(pol::SmythBrobyKernel, i::Integer, j::Integer)
    sigmai = pol.sd[i]
    sigmaj = pol.sd[j]
    return (mui = pol.mu[i], muj = pol.mu[j], sigmai = sigmai, sigmaj = sigmaj,
            c1i = pol.c1 * sigmai, c1j = pol.c1 * sigmaj)
end
"""
$(DocStringExtensions.TYPEDEF)

Union of the Smyth-Broby markers that accumulate only the [`sb_delta`](@ref) contributions, and not the counts.

The prefix of a marker selects the score triple, and the trailing digit selects the denominator. The two choices are independent, so each prefix has its own union. This prefix is the one that the source defines. It adds contributions and keeps no count.

# Related

  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
  - [`sb_add_pos`](@ref): the accumulator that dispatches on this alias.
  - [`sb_pair_scores`](@ref): the selector that dispatches on this alias.
  - [`SmythBrobyGerberAlg`](@ref)
  - [`SmythBrobyCountAlg`](@ref)
"""
const SmythBrobyDeltaAlg = Union{SmythBroby0, SmythBroby1, SmythBroby2}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Smyth-Broby markers that accumulate both the [`sb_delta`](@ref) contributions and the co-movement counts, and score a pair by their product.

The prefix of a marker selects the score triple, and the trailing digit selects the denominator. The two choices are independent, so each prefix has its own union. Section 5 of the source suggests a count beside each sum and states no formula. The product is a choice of the library.

# Related

  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`sb_add_pos`](@ref): the accumulator that dispatches on this alias.
  - [`sb_pair_scores`](@ref): the selector that dispatches on this alias.
  - [`SmythBrobyDeltaAlg`](@ref)
  - [`SmythBrobyCountAlg`](@ref)
"""
const SmythBrobyGerberAlg = Union{SmythBrobyGerber0, SmythBrobyGerber1, SmythBrobyGerber2}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Smyth-Broby markers that accumulate the co-movement counts only, and never evaluate [`sb_delta`](@ref).

The prefix of a marker selects the score triple, and the trailing digit selects the denominator. The two choices are independent, so each prefix has its own union. This prefix gives a Gerber statistic on the zones of Smyth-Broby. The source keeps no count, so this prefix is a construction of the library.

# Related

  - [`SmythBrobyCount0`](@ref)
  - [`SmythBrobyCount1`](@ref)
  - [`SmythBrobyCount2`](@ref)
  - [`sb_add_pos`](@ref): the accumulator that dispatches on this alias.
  - [`sb_pair_scores`](@ref): the selector that dispatches on this alias.
  - [`GerberCovariance`](@ref): the statistic that this prefix gives when `c1` is zero, `c2` is the Gerber threshold `t`, `c3` is above every return, and the mean of each asset is zero.
  - [`SmythBrobyDeltaAlg`](@ref)
  - [`SmythBrobyGerberAlg`](@ref)
"""
const SmythBrobyCountAlg = Union{SmythBrobyCount0, SmythBrobyCount1, SmythBrobyCount2}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a concordant observation to the pair accumulator, as the prefix of `alg` selects.

The accumulator is a named tuple. Each method returns a new accumulator and changes no argument.

# Algorithm

The marker prefix selects one of three branches.

 1. [`SmythBrobyDeltaAlg`](@ref): add [`sb_delta`](@ref) to `acc.pos`, and do not change `acc.cpos`.
 2. [`SmythBrobyGerberAlg`](@ref): add [`sb_delta`](@ref) to `acc.pos`, and add one to `acc.cpos`.
 3. [`SmythBrobyCountAlg`](@ref): add one to `acc.cpos`, and do not evaluate [`sb_delta`](@ref).

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This family does not read `di` and `dj`.
  - `ari`, `arj`: Absolute centred standardised returns of the two assets at the observation.
  - $(arg_dict[:sbn])

# Returns

  - The updated accumulator.

# Related

  - [`sb_add_neg`](@ref)
  - [`sb_add_neutral`](@ref)
  - [`comovement_step`](@ref)
  - [`sb_delta`](@ref)
"""
@inline function sb_add_pos(::SmythBrobyDeltaAlg, acc, ari::Number, arj::Number, n::Number)
    return (; acc..., pos = acc.pos + sb_delta(ari, arj, n))
end
@inline function sb_add_pos(::SmythBrobyGerberAlg, acc, ari::Number, arj::Number, n::Number)
    return (; acc..., pos = acc.pos + sb_delta(ari, arj, n), cpos = acc.cpos + 1)
end
@inline function sb_add_pos(::SmythBrobyCountAlg, acc, ari::Number, arj::Number, n::Number)
    return (; acc..., cpos = acc.cpos + 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a discordant observation to the pair accumulator, as the prefix of `alg` selects.

It has the same three branches as [`sb_add_pos`](@ref), on the discordant score and count.

# Algorithm

The marker prefix selects one of three branches.

 1. [`SmythBrobyDeltaAlg`](@ref): add [`sb_delta`](@ref) to `acc.neg`, and do not change `acc.cneg`.
 2. [`SmythBrobyGerberAlg`](@ref): add [`sb_delta`](@ref) to `acc.neg`, and add one to `acc.cneg`.
 3. [`SmythBrobyCountAlg`](@ref): add one to `acc.cneg`, and do not evaluate [`sb_delta`](@ref).

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This family does not read `di` and `dj`.
  - `ari`, `arj`: Absolute centred standardised returns of the two assets at the observation.
  - $(arg_dict[:sbn])

# Returns

  - The updated accumulator.

# Related

  - [`sb_add_pos`](@ref)
  - [`sb_add_neutral`](@ref)
  - [`comovement_step`](@ref)
  - [`sb_delta`](@ref)
"""
@inline function sb_add_neg(::SmythBrobyDeltaAlg, acc, ari::Number, arj::Number, n::Number)
    return (; acc..., neg = acc.neg + sb_delta(ari, arj, n))
end
@inline function sb_add_neg(::SmythBrobyGerberAlg, acc, ari::Number, arj::Number, n::Number)
    return (; acc..., neg = acc.neg + sb_delta(ari, arj, n), cneg = acc.cneg + 1)
end
@inline function sb_add_neg(::SmythBrobyCountAlg, acc, ari::Number, arj::Number, n::Number)
    return (; acc..., cneg = acc.cneg + 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a neutral observation to the pair accumulator. An observation is neutral when exactly one of the two assets leaves the indecision zone, and [`sb_crossed`](@ref) decides which assets leave it.

Only the [`GerberComovementOne`](@ref) markers reach an acting method, because only their denominator has a neutral term. The fall-through method returns the accumulator unchanged. The neutral score of every other marker therefore stays at zero, and no reduction reads it.

# Algorithm

The marker selects one of four branches.

 1. [`SmythBroby1`](@ref): add [`sb_delta`](@ref) to `acc.nn`.
 2. [`SmythBrobyGerber1`](@ref): add [`sb_delta`](@ref) to `acc.nn`, and add one to `acc.cnn`.
 3. [`SmythBrobyCount1`](@ref): add one to `acc.cnn`, and do not evaluate [`sb_delta`](@ref).
 4. Any other [`SmythBrobyCovarianceAlgorithm`](@ref): return `acc` unchanged.

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This family does not read `di` and `dj`.
  - `ari`, `arj`: Absolute centred standardised returns of the two assets at the observation. The fall-through method reads neither.
  - $(arg_dict[:sbn])

# Returns

  - The updated accumulator.

# Related

  - [`sb_add_pos`](@ref)
  - [`sb_add_neg`](@ref)
  - [`comovement_step`](@ref)
  - [`GerberComovementOne`](@ref)
  - [`sb_delta`](@ref)
"""
@inline function sb_add_neutral(::SmythBroby1, acc, ari::Number, arj::Number, n::Number)
    return (; acc..., nn = acc.nn + sb_delta(ari, arj, n))
end
@inline function sb_add_neutral(::SmythBrobyGerber1, acc, ari::Number, arj::Number,
                                n::Number)
    return (; acc..., nn = acc.nn + sb_delta(ari, arj, n), cnn = acc.cnn + 1)
end
@inline function sb_add_neutral(::SmythBrobyCount1, acc, ari::Number, arj::Number,
                                n::Number)
    return (; acc..., cnn = acc.cnn + 1)
end
@inline function sb_add_neutral(::SmythBrobyCovarianceAlgorithm, acc, ::Number, ::Number,
                                ::Number)
    return acc
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Decide whether one asset leaves the indecision zone at one observation.

An asset leaves the indecision zone when the magnitude of its centred, standardised return is at least `c2`, and that return is not exactly zero. For a positive threshold, the second test changes nothing, because `ar >= c > 0` already implies that `r` is not zero. The second test matters only at `c2 = 0`. There, `ar >= 0` holds for every return, a return of exactly zero included. With the second test, a return of exactly zero never crosses.

This rule keeps the diagonal of the statistic at one. The pair `(i, i)` crosses on both axes or on neither, so it never reaches the neutral accumulator, which a [`GerberComovementOne`](@ref) marker divides by. Without the second test, at `c2 = 0`, a return of exactly zero crosses on both axes but has no sign. [`comovement_step`](@ref) then adds it to the neutral accumulator, and the diagonal falls below one.

The rule applies to the indecision zone and not to the confusion zone, because the two zones test different quantities. The indecision zone tests the centred return. A centred return of exactly zero is an asset at its own mean, and it has no sign to classify. The confusion zone tests the raw, uncentred return, and the zero of that return is an arbitrary point on the scale of the data. A raw return of zero is a deviation of ``-\\mu`` from the mean, and that deviation has a sign. The confusion zone also only rejects observations and never classifies them, so it cannot give a wrong count.

# Arguments

  - `r`: Centred, standardised return of the asset at the observation.
  - `ar`: Its absolute value.
  - `c`: The indecision-zone threshold `c2`.

# Returns

  - `crossed::Bool`: `true` when the asset left the indecision zone.

# Related

  - [`comovement_step`](@ref)
  - [`SmythBrobyKernel`](@ref)
  - [`sb_add_neutral`](@ref)
"""
@inline function sb_crossed(r::Number, ar::Number, c::Number)
    return ar >= c && !iszero(r)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add one observation of a pair to the co-movement accumulator.

The confusion zone tests the raw, uncentred return, and the indecision zone tests the centred, standardised return. Section 2 of [smyth2022enhanced](@cite) defines the two zones this way, and a confusion zone on the centred return changes the statistic. Because the two zones test different quantities, only the indecision zone has the rule that a return of exactly zero never crosses. [`sb_crossed`](@ref) states that rule. The Gerber IQ method compares absolute returns with the scaled thresholds of the pair through [`iq_crossed`](@ref). It weights each observation by the IQ template and by the temporal decay through [`gerber_IQ_delta`](@ref).

# Algorithm

The Smyth-Broby method runs these steps. It does not read `T` or `k`, because the family applies no temporal decay.

 1. Return `acc` unchanged when `abs(xi)` is below `st.c1i` and `abs(xj)` is below `st.c1j`. This is the confusion zone, on the raw return.
 2. Centre and standardise both returns with the pair state. The results are `ri` and `rj`, and their magnitudes are `ari` and `arj`.
 3. Decide with [`sb_crossed`](@ref) whether each asset leaves the indecision zone of `pol.c2`.
 4. Return `acc` unchanged when either magnitude is above `pol.c3`, or when neither asset leaves the indecision zone. The first case is the outer cut-off, and the second case is the indecision zone.
 5. Accumulate through [`sb_add_pos`](@ref) when both assets cross and the product `ri * rj` is positive.
 6. Accumulate through [`sb_add_neg`](@ref) when both assets cross and the product is negative.
 7. In all other cases, accumulate through [`sb_add_neutral`](@ref). This is the case where exactly one asset crosses. When both assets cross, the product is not zero, so the three branches cover every admitted observation.

# Arguments

  - `pol`: Co-movement policy object.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This family does not read `di` and `dj`.
  - `st`: Pair state from [`comovement_pair_state`](@ref).
  - `xi`, `xj`: Returns of assets `i` and `j` at observation `k`.
  - `T`: Number of observations.
  - `k`: Observation index.

# Returns

  - The updated accumulator.

# Related

  - [`gerber_comovement!`](@ref)
  - [`comovement_finalise`](@ref)
  - [`sb_crossed`](@ref)
"""
@inline function comovement_step(pol::SmythBrobyKernel, acc, st, xi::Number, xj::Number,
                                 ::Integer, ::Integer)
    if abs(xi) < st.c1i && abs(xj) < st.c1j
        return acc
    end
    ri = (xi - st.mui) / st.sigmai
    rj = (xj - st.muj) / st.sigmaj
    ari = abs(ri)
    arj = abs(rj)
    c2 = pol.c2
    crossi = sb_crossed(ri, ari, c2)
    crossj = sb_crossed(rj, arj, c2)
    if ari > pol.c3 || arj > pol.c3 || !crossi && !crossj
        return acc
    end
    return if crossi && crossj && ri * rj > zero(ri)
        sb_add_pos(pol.alg, acc, ari, arj, pol.n)
    elseif crossi && crossj && ri * rj < zero(ri)
        sb_add_neg(pol.alg, acc, ari, arj, pol.n)
    else
        sb_add_neutral(pol.alg, acc, ari, arj, pol.n)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the concordant, discordant and neutral scores of a pair from the accumulator, as the prefix of `alg` selects.

This function dispatches on the prefix of the marker. [`comovement_ratio`](@ref) dispatches on its trailing digit.

# Algorithm

The marker prefix selects one of three branches.

 1. [`SmythBrobyDeltaAlg`](@ref): return the contribution sums `(acc.pos, acc.neg, acc.nn)`.
 2. [`SmythBrobyGerberAlg`](@ref): return each sum times its own count, `(acc.pos * acc.cpos, acc.neg * acc.cneg, acc.nn * acc.cnn)`.
 3. [`SmythBrobyCountAlg`](@ref): return the counts `(acc.cpos, acc.cneg, acc.cnn)`.

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This family does not read `di` and `dj`.

# Returns

  - The score triple `(p, q, u)` of the pair.

# Related

  - [`comovement_finalise`](@ref): the caller that reaches this function once per pair.
  - [`comovement_ratio`](@ref)
  - [`SmythBrobyDeltaAlg`](@ref)
  - [`SmythBrobyGerberAlg`](@ref)
  - [`SmythBrobyCountAlg`](@ref)
"""
@inline function sb_pair_scores(::SmythBrobyDeltaAlg, acc)
    return (acc.pos, acc.neg, acc.nn)
end
@inline function sb_pair_scores(::SmythBrobyGerberAlg, acc)
    return (acc.pos * acc.cpos, acc.neg * acc.cneg, acc.nn * acc.cnn)
end
@inline function sb_pair_scores(::SmythBrobyCountAlg, acc)
    return (acc.cpos, acc.cneg, acc.cnn)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reduce the accumulator of a pair to the co-movement statistic of the pair.

It applies both parts of the marker. The prefix selects the scores, and the trailing digit selects the denominator.

# Algorithm

 1. Select the score triple `(p, n, nn)` from `acc` with [`sb_pair_scores`](@ref), which dispatches on the marker prefix.
 2. Reduce the triple with [`comovement_ratio`](@ref), which dispatches on the trailing digit, and return the result.

# Arguments

  - `pol`: Co-movement policy object.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This family does not read `di` and `dj`.
  - `T`: Element type of the zero that a zero denominator gives.

# Returns

  - The pairwise co-movement statistic.

# Related

  - [`gerber_comovement!`](@ref)
  - [`comovement_step`](@ref)
  - [`sb_pair_scores`](@ref)
  - [`comovement_ratio`](@ref)
"""
@inline function comovement_finalise(pol::SmythBrobyKernel, acc, ::Type{T}) where {T}
    p, n, nn = sb_pair_scores(pol.alg, acc)
    return comovement_ratio(pol.alg, p, n, nn, T)
end
"""
    smythbroby(ce::SmythBrobyCovariance, X::MatNum, mu::ArrNum, sd::ArrNum)

Compute the Smyth-Broby correlation matrix with the marker in `ce.alg`.

All nine markers share the pairwise loop of [`gerber_comovement!`](@ref) through a [`SmythBrobyKernel`](@ref) policy. The marker selects the scores and the denominator.

# Mathematical definition

For each pair ``(i, j)``, an observation ``t`` must pass two tests. The confusion zone compares the raw, uncentred return with ``c_1 \\sigma``. It rejects ``t`` only when both assets are inside it:

```math
\\begin{align}
|x_{ti}| < c_1 \\sigma_i \\quad \\text{and} \\quad |x_{tj}| < c_1 \\sigma_j\\,.
\\end{align}
```

The indecision zone compares the centred, standardised return ``\\tilde{r}_{ti} = (x_{ti} - \\mu_i) / \\sigma_i`` with ``c_2``. Asset ``i`` crosses at ``t`` when

```math
\\begin{align}
|\\tilde{r}_{ti}| \\geq c_2 \\quad \\text{and} \\quad \\tilde{r}_{ti} \\neq 0\\,,
\\end{align}
```

and the indecision zone rejects ``t`` when neither asset crosses. The outer cut-off rejects ``t`` when either asset is above ``c_3``. The second condition matters only at ``c_2 = 0``, because ``|\\tilde{r}| \\geq c_2 > 0`` already excludes a zero return. [`sb_crossed`](@ref) holds this rule. Section 2 of [smyth2022enhanced](@cite) puts the confusion zone on the uncentred return and the indecision zone on the centred return. A confusion zone on the centred return changes the statistic.

An admitted observation is concordant when both assets cross and ``\\tilde{r}_{ti} \\tilde{r}_{tj} > 0``. It is discordant when both assets cross and the product is negative. It is neutral when exactly one asset crosses. The sums and the counts of each class over the admitted observations are:

```math
\\begin{align}
\\text{pos} &= \\sum_t \\delta_t \\, \\mathbf{1}[t \\in C]\\,, \\quad \\text{neg} = \\sum_t \\delta_t \\, \\mathbf{1}[t \\in D]\\,, \\quad \\text{nn} = \\sum_t \\delta_t \\, \\mathbf{1}[t \\in N]\\,, \\\\
c^{+} &= |C|\\,, \\quad c^{-} = |D|\\,, \\quad c^{0} = |N|\\,,
\\end{align}
```

Where:

  - $(math_dict[:r_tilde_sb])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:sigma_i_asset])
  - $(math_dict[:c1_sb])
  - $(math_dict[:c2_sb])
  - $(math_dict[:c3_sb])
  - $(math_dict[:delta_sb])
  - $(math_dict[:CDN_sb])
  - $(math_dict[:possum_sb])
  - $(math_dict[:poscount_sb])
  - $(math_dict[:pqu_sb])

Here ``\\delta_t = \\delta(|\\tilde{r}_{t,\\,i}|, |\\tilde{r}_{t,\\,j}|, n)`` is the [`sb_delta`](@ref) contribution. The marker prefix selects the three scores ``(p, q, u)``:

  - `SmythBroby*`: ``(\\text{pos},\\, \\text{neg},\\, \\text{nn})``.
  - `SmythBrobyGerber*`: ``(\\text{pos} \\, c^{+},\\, \\text{neg} \\, c^{-},\\, \\text{nn} \\, c^{0})``. Each sum has its own count, the neutral sum included.
  - `SmythBrobyCount*`: ``(c^{+},\\, c^{-},\\, c^{0})``.

The trailing digit selects the reduction, where ``h_{ij} = p - q`` is the net score:

```math
\\begin{align}
\\hat{\\rho}_{ij} &= \\begin{cases}
(p - q) / (p + q) & 0 \\\\
(p - q) / (p + q + u) & 1 \\\\
h_{ij} / \\sqrt{h_{ii} \\, h_{jj}} & 2
\\end{cases}\\,.
\\end{align}
```

The digits `0` and `1` give zero when the denominator is zero. The digit `2` divides the matrix of net scores by the geometric mean of its diagonal, and clamps each root from below at ``\\sqrt{\\varepsilon}``. The result is not the ratio of the digit `0` after a second division. The two are equal only where ``p + q`` is the same for every pair.

Of the nine markers, only `SmythBroby1` is a statistic of the source. It is Equation 5 of the source, and the source defines no other statistic. The trailing digits `0` and `2` are reductions of the library, as [`Gerber0`](@ref) and [`Gerber2`](@ref) are for the Gerber statistic. The `SmythBrobyGerber*` prefix is how the library reads Section 5 of the source. Section 5 suggests a count beside each sum and states no formula. The `SmythBrobyCount*` prefix is a construction of the library, because the source keeps no count.

# Algorithm

 1. Read the asset count `N` from the second dimension of `X`, and allocate the `N × N` output `rho`.
 2. Build the policy `pol` as a [`SmythBrobyKernel`](@ref) from `ce.alg`, `mu`, `sd`, `ce.c1`, `ce.c2`, `ce.c3` and `ce.n`.
 3. Fill `rho` with [`gerber_comovement!`](@ref) on the executor `ce.ex`.
 4. Divide `rho` in place by the geometric mean of its diagonal with [`standardise_comovement!`](@ref). Only a `2` marker changes `rho`.
 5. Write one onto a zero diagonal entry of `rho` with [`comovement_unit_diagonal!`](@ref). An asset that admits no observation has a zero diagonal entry, and that entry is one by definition.
 6. Repair `rho` with [`posdef!`](@ref) and the `pdm` field of the estimator, because a Smyth-Broby matrix is not positive definite in general. It does nothing when `ce.pdm` is `nothing`.

# Arguments

  - `ce`: Smyth-Broby covariance estimator.
  - $(arg_dict[:X])
  - `mu`: Vector of asset means, one entry per asset.
  - $(arg_dict[:stdarr])

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, after the repair of the `pdm` field of the estimator.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyKernel`](@ref)
  - [`gerber_comovement!`](@ref)
  - [`standardise_comovement!`](@ref)
  - [`comovement_unit_diagonal!`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
  - [`GerberCovariance`](@ref): the statistic that this family extends. Its docstring states the matrix form, and this docstring does not repeat it.

# References

  - $(ref_dict[:smyth2022enhanced]) Section 2, Equation 5.
"""
function smythbroby(ce::SmythBrobyCovariance, X::MatNum, mu::ArrNum, sd::ArrNum)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    pol = SmythBrobyKernel(ce.alg, mu, sd, ce.c1, ce.c2, ce.c3, ce.n)
    gerber_comovement!(rho, ce.ex, X, pol)
    standardise_comovement!(ce.alg, rho)
    comovement_unit_diagonal!(rho)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    Statistics.cor(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Smyth-Broby correlation matrix.

The `me` and `ve` fields of the estimator compute the mean and the standard deviation. The estimator, not this method, therefore sets the centre and the scale of the zones.

# Algorithm

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref).
 2. Compute the standard deviation `sd` of each column with `ce.ve`, and clamp it from below at `eps(eltype(sd))`. The clamp prevents a division by zero for a constant column.
 3. Compute the mean `mu` of each column with `ce.me`.
 4. Return the matrix that [`smythbroby`](@ref) builds from `X`, `mu` and `sd`.

# Arguments

  - `ce`: Smyth-Broby covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for the mean estimator and for the standard deviation estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`smythbroby`](@ref)
  - [`cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    mu = Statistics.mean(ce.me, X; dims = 1, kwargs...)
    return smythbroby(ce, X, mu, sd)
end
"""
    Statistics.cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Smyth-Broby covariance matrix.

The method scales the correlation matrix by the same `sd` that the zones use. The covariance is therefore exactly the correlation times the outer product of `sd`, and its diagonal is exactly `sd .^ 2`.

# Algorithm

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref).
 2. Compute the standard deviation `sd` of each column with `ce.ve`, and clamp it from below at `eps(eltype(sd))`.
 3. Compute the mean `mu` of each column with `ce.me`.
 4. Build the correlation matrix `sigma` with [`smythbroby`](@ref) from `X`, `mu` and `sd`.
 5. Scale `sigma` in place to a covariance matrix with `StatsBase.cor2cov!` and `sd`, and return it.

# Arguments

  - `ce`: Smyth-Broby covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for the mean estimator and for the standard deviation estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `sigma::Matrix{<:Number}`: The Smyth-Broby covariance matrix.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`smythbroby`](@ref)
  - [`cor(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    mu = Statistics.mean(ce.me, X; dims = 1, kwargs...)
    sigma = smythbroby(ce, X, mu, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export SmythBroby0, SmythBroby1, SmythBroby2, SmythBrobyGerber0, SmythBrobyGerber1,
       SmythBrobyGerber2, SmythBrobyCount0, SmythBrobyCount1, SmythBrobyCount2,
       SmythBrobyCovariance
