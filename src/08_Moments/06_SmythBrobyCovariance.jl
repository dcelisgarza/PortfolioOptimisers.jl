"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance estimators.

All concrete and/or abstract types implementing Smyth-Broby covariance estimation algorithms should be subtypes of `BaseSmythBrobyCovariance`. It is a subtype of [`BaseGerberCovariance`](@ref), because a Smyth-Broby statistic is a Gerber statistic with a second zone and a real-valued contribution in place of the vote. The Gerber statistic itself is stated in `05_GerberCovariance.jl` and is not restated here.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovariance`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced])
"""
abstract type BaseSmythBrobyCovariance <: BaseGerberCovariance end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance algorithm types.

All concrete and/or abstract types implementing specific Smyth-Broby covariance algorithms should be subtypes of `SmythBrobyCovarianceAlgorithm`.

These types are used to specify the algorithm when constructing a [`SmythBrobyCovariance`](@ref) estimator. A marker names two independent choices at once: its **prefix** selects the score triple through [`sb_pair_scores`](@ref), and its **trailing digit** selects the denominator through [`comovement_ratio`](@ref). The digit means the same thing here as it does in the Gerber family.

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

  - $(ref_dict[:smyth2022enhanced])
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

The source defines one statistic, and it keeps the neutral sum in the denominator; [`SmythBroby1`](@ref) is that statistic. This tag drops the neutral term, on the shape [`Gerber0`](@ref) sets, so it is the library's own reduction and not a formulation of the source.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Read `(pos, neg, nn)` from the pair accumulator, giving the scores `(p, q, u)`. The contribution sums are taken and the counts are discarded.
 2. Return `(p - q) / (p + q)`, or `zero(T)` when the denominator vanishes.

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

  - $(ref_dict[:smyth2022enhanced])
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

**This is the statistic the source defines**, and it is the only one the source defines. The neutral sum keeps the matrix positive semidefinite, which is the purpose the Gerber neutral count already served.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Read `(pos, neg, nn)` from the pair accumulator, giving the scores `(p, q, u)`. The contribution sums are taken and the counts are discarded.
 2. Return `(p - q) / (p + q + u)`, or `zero(T)` when the denominator vanishes.

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

  - $(ref_dict[:smyth2022enhanced])
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

The normalisation is a property of the whole matrix and not of one pair, so the diagonal is unit by construction rather than by a per-pair denominator as in [`SmythBroby0`](@ref) and [`SmythBroby1`](@ref). It is not the [`SmythBroby0`](@ref) ratio renormalised: the two agree only where ``p + q`` is constant across pairs. The source defines no such normalisation, so this tag is the library's own reduction, on the shape [`Gerber2`](@ref) sets.

# Algorithm

The branch of [`sb_pair_scores`](@ref), of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this tag selects runs these steps.

 1. Read `(pos, neg, nn)` from the pair accumulator, giving the scores `(p, q, u)`.
 2. Return the net score `p - q` for every pair. This branch applies no denominator of its own.
 3. Divide the assembled matrix by the outer product of the square roots of its own diagonal. The roots are clamped from below by `sqrt(eps(eltype(rho)))`, so an asset that admits no observation gives a zero row rather than a division by zero.

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

  - $(ref_dict[:smyth2022enhanced])
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

The weighting reintroduces the Gerber vote count that [`SmythBroby0`](@ref) discards, so a pair that co-moves often scores above one that co-moves rarely but sharply. The source's conclusion suggests keeping a count beside the sum in one sentence and states no formula for it, so the product above is the library's reading of that sentence. The neutral term is dropped here, as in [`SmythBroby0`](@ref).

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Multiply each of `pos`, `neg` and `nn` by its own count `cpos`, `cneg` and `cnn`, giving the scores `(p, q, u)`.
 2. Return `(p - q) / (p + q)`, or `zero(T)` when the denominator vanishes.

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

  - $(ref_dict[:smyth2022enhanced])
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

The neutral term carries its own count as well, so all three terms of the denominator are scaled alike. This is the estimator's default algorithm. It composes the source's own statistic, [`SmythBroby1`](@ref), with the count the source's conclusion suggests keeping beside the sum; the source states no formula for that product.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Multiply each of `pos`, `neg` and `nn` by its own count `cpos`, `cneg` and `cnn`, giving the scores `(p, q, u)`.
 2. Return `(p - q) / (p + q + u)`, or `zero(T)` when the denominator vanishes.

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

  - $(ref_dict[:smyth2022enhanced])
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

The normalisation is a property of the whole matrix and not of one pair, as in [`SmythBroby2`](@ref). The source defines no such normalisation, so this tag is the library's own reduction, on the shape [`Gerber2`](@ref) sets.

# Algorithm

The branch of [`sb_pair_scores`](@ref), of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this tag selects runs these steps.

 1. Multiply each of `pos`, `neg` and `nn` by its own count `cpos`, `cneg` and `cnn`, giving the scores `(p, q, u)`.
 2. Return the net score `p - q` for every pair. This branch applies no denominator of its own.
 3. Divide the assembled matrix by the outer product of the square roots of its own diagonal. The roots are clamped from below by `sqrt(eps(eltype(rho)))`, so an asset that admits no observation gives a zero row rather than a division by zero.

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

  - $(ref_dict[:smyth2022enhanced])
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

Dropping [`sb_delta`](@ref) recovers a Gerber statistic evaluated on the Smyth-Broby admission rule rather than on the Gerber threshold, so this tag reduces exactly to [`Gerber0`](@ref) when the confusion zone is switched off, the outer cut-off is lifted, and the centre is zero. The source counts no votes — its whole argument is that a contribution sum carries more information than a count — so the count family is the library's own construction and not a formulation of the source.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Read `(cpos, cneg, cnn)` from the pair accumulator, giving the scores `(p, q, u)`. The contribution sums are discarded, and [`sb_delta`](@ref) is never evaluated.
 2. Return `(p - q) / (p + q)`, or `zero(T)` when the denominator vanishes.

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

  - $(ref_dict[:smyth2022enhanced])
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

Dropping [`sb_delta`](@ref) recovers a Gerber statistic evaluated on the Smyth-Broby admission rule rather than on the Gerber threshold, so this tag reduces exactly to [`Gerber1`](@ref) when the confusion zone is switched off, the outer cut-off is lifted, and the centre is zero. The source counts no votes, so the count family is the library's own construction and not a formulation of the source.

# Algorithm

The branch of [`sb_pair_scores`](@ref) and of [`comovement_ratio`](@ref) that this tag selects runs these steps.

 1. Read `(cpos, cneg, cnn)` from the pair accumulator, giving the scores `(p, q, u)`. The contribution sums are discarded, and [`sb_delta`](@ref) is never evaluated.
 2. Return `(p - q) / (p + q + u)`, or `zero(T)` when the denominator vanishes.

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

  - $(ref_dict[:smyth2022enhanced])
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

The normalisation is a property of the whole matrix and not of one pair, as in [`SmythBroby2`](@ref). This tag reduces exactly to [`Gerber2`](@ref) when the confusion zone is switched off, the outer cut-off is lifted, and the centre is zero. The source counts no votes and defines no such normalisation, so this tag is the library's own construction.

# Algorithm

The branch of [`sb_pair_scores`](@ref), of [`comovement_ratio`](@ref) and of [`standardise_comovement!`](@ref) that this tag selects runs these steps.

 1. Read `(cpos, cneg, cnn)` from the pair accumulator, giving the scores `(p, q, u)`. The contribution sums are discarded, and [`sb_delta`](@ref) is never evaluated.
 2. Return the net score `p - q` for every pair. This branch applies no denominator of its own.
 3. Divide the assembled matrix by the outer product of the square roots of its own diagonal. The roots are clamped from below by `sqrt(eps(eltype(rho)))`, so an asset that admits no observation gives a zero row rather than a division by zero.

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

  - $(ref_dict[:smyth2022enhanced])
"""
struct SmythBrobyCount2 <: SmythBrobyCovarianceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Configures and applies Smyth-Broby covariance estimators.

`SmythBrobyCovariance` encapsulates all components required for Smyth-Broby-based covariance or correlation estimation, including the expected returns estimator, variance estimator, positive definite matrix estimator, algorithm parameters, and the specific Smyth-Broby algorithm variant. A Smyth-Broby matrix is a matrix of pairwise contribution ratios and is not positive definite in general, so `pdm` projects the result onto the nearest positive definite matrix; `pdm = nothing` returns the raw statistic instead. **Of the nine algorithm tags the source defines one**, [`SmythBroby1`](@ref); each of the other eight names which part of it is the library's own, and [`smythbroby`](@ref) states the shared statistic once.

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
        ex::FLoops.Transducers.Executor = ThreadedEx()
    ) -> SmythBrobyCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:c1])
  - $(val_dict[:c2])
  - $(val_dict[:c3])
  - $(val_dict[:c3c2])
  - $(val_dict[:sbn])
  - `c1`, `c2` and `c3` are validated with [`assert_nonempty_nonneg_finite_val`](@ref), so `Inf` and `NaN` are rejected. `n` is validated with [`assert_nonneg`](@ref), which rejects a negative `n` and `NaN` and admits `Inf`. The three thresholds are read on the scale of the data, where `Inf` admits no observation at all; `n` is an exponent whose infinite limit is a hard divergence gate, so it is kept. A negative `n` inverts the severity penalty of [`sb_delta`](@ref): a pair whose two magnitudes agree would then contribute nothing, and the diagonal would be zero rather than one.

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
  - [`GerberCovariance`](@ref): the statistic this family extends. Its zoning is the threshold rule the confusion zone and the indecision zone replace.
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced])
"""
@propagatable @concrete struct SmythBrobyCovariance <: BaseSmythBrobyCovariance
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:me]) Used for optionally centering the returns.
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
    function SmythBrobyCovariance(ve::StatsBase.CovarianceEstimator,
                                  me::AbstractExpectedReturnsEstimator,
                                  pdm::Option{<:AbstractPosdefEstimator}, c1::Number,
                                  c2::Number, c3::Number, n::Number,
                                  alg::SmythBrobyCovarianceAlgorithm,
                                  ex::FLoops.Transducers.Executor)
        assert_nonempty_nonneg_finite_val(c1, :c1)
        assert_nonempty_nonneg_finite_val(c2, :c2)
        assert_nonempty_nonneg_finite_val(c3, :c3)
        assert_nonneg(n, :n)
        @argcheck(c2 < c3, DomainError("c2 must be less than c3, got c2 = $c2, c3 = $c3"))
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(c1), typeof(c2), typeof(c3),
                   typeof(n), typeof(alg), typeof(ex)}(ve, me, pdm, c1, c2, c3, n, alg, ex)
    end
end
function SmythBrobyCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                              me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                              pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                              c1::Number = 0.5, c2::Number = 0.5, c3::Number = 4,
                              n::Number = 2,
                              alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1(),
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())::SmythBrobyCovariance
    return SmythBrobyCovariance(ve, me, pdm, c1, c2, c3, n, alg, ex)
end
"""
    sb_delta(ri::Number, rj::Number, n::Number) -> Number

Contribution of one admitted observation to a Smyth-Broby pair score.

This is the quantity that replaces the Gerber vote: an observation contributes a finite real number rather than a count of one, so a co-movement of four standard deviations weighs more than one of a single standard deviation. **Both arguments are already absolute**, and the caller has already centred and standardised them.

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

``\\kappa`` rewards magnitude and ``\\gamma^{n}`` penalises a pair whose two magnitudes disagree, so the contribution is largest when both assets move far and move by the same amount. ``\\gamma`` is the absolute difference of the two **magnitudes**, not of the two signed returns; the sign has already been read by the caller, which is what put the observation in the concordant or the discordant set.

# Algorithm

 1. Multiply the two gross magnitudes `1 + ri` and `1 + rj` and take the square root, giving the amplitude `kappa`.
 2. Take the absolute difference of `ri` and `rj`, giving the divergence `gamma`.
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

  - $(ref_dict[:smyth2022enhanced])
"""
function sb_delta(ri::Number, rj::Number, n::Number)
    kappa = sqrt((one(ri) + ri) * (one(rj) + rj))
    gamma = abs(ri - rj)
    return kappa / (one(gamma) + gamma^n)
end
"""
$(DocStringExtensions.TYPEDEF)

Union of the Gerber-family markers whose pairwise statistic divides the net score by the sum of the concordant and the discordant score, guarded to zero when the denominator vanishes.

The group exists because the trailing `0` means the same thing in all four families of the Gerber lineage, and one method of [`comovement_ratio`](@ref) serves them all. A family adds a member to this union rather than adding a branch to that method.

# Related

  - [`Gerber0`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyCount0`](@ref)
  - [`comovement_ratio`](@ref): the method that dispatches on this alias.
  - [`GerberComovementOne`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
const GerberComovementZero = Union{<:Gerber0, <:SmythBroby0, <:SmythBrobyGerber0,
                                   <:SmythBrobyCount0}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Gerber-family markers whose pairwise statistic divides the net score by the sum of all three scores, the neutral one included, guarded to zero when the denominator vanishes.

The group exists because the trailing `1` means the same thing in all four families of the Gerber lineage, and one method of [`comovement_ratio`](@ref) serves them all. The neutral term is what keeps the statistic positive semidefinite, so this is the canonical member of each family.

# Related

  - [`Gerber1`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyCount1`](@ref)
  - [`comovement_ratio`](@ref): the method that dispatches on this alias.
  - [`sb_add_neutral`](@ref): the accumulator that fills the neutral score, and which only these markers reach.
  - [`GerberComovementZero`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
const GerberComovementOne = Union{<:Gerber1, <:SmythBroby1, <:SmythBrobyGerber1,
                                  <:SmythBrobyCount1}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Gerber-family markers whose pairwise statistic is the net score itself, with the assembled matrix normalised afterwards by the geometric mean of its own diagonal.

The group exists because the trailing `2` means the same thing in all four families of the Gerber lineage. It is the one variant whose normalisation is a property of the whole matrix rather than of one pair, so it is also the only one that reaches the acting method of [`standardise_comovement!`](@ref).

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
const GerberComovementTwo = Union{<:Gerber2, <:SmythBroby2, <:SmythBrobyGerber2,
                                  <:SmythBrobyCount2}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reduce a pair's accumulated positive, negative, and neutral co-movement scores to the pairwise correlation entry.

The variant marker selects the denominator policy. **`n` here is the discordant score, not the severity exponent of [`sb_delta`](@ref)**; the two share a glyph and nothing else.

# Algorithm

The marker selects one of three branches.

 1. [`GerberComovementZero`](@ref): return `(p - n) / (p + n)`, or `zero(T)` when `p + n` is zero. It does not read `nn`.
 2. [`GerberComovementOne`](@ref): return `(p - n) / (p + n + nn)`, or `zero(T)` when `p + n + nn` is zero. This is the only branch that reads `nn`.
 3. [`GerberComovementTwo`](@ref): return `p - n`, and apply no denominator. [`standardise_comovement!`](@ref) normalises the assembled matrix afterwards.

A zero denominator means that the pair qualified no observation, and the guarded zero is the right answer for an **off-diagonal** entry. It is the wrong answer on the **diagonal**, where a correlation is one by definition. This function cannot separate the two cases, because it does not know whether the pair is `(i, i)`. [`comovement_unit_diagonal!`](@ref) writes the diagonal after the matrix is assembled, and ADR 0093 records the decision.

# Arguments

  - `alg`: Co-movement algorithm marker.
  - `p`, `n`, `nn`: Accumulated concordant, discordant and neutral scores of one pair.
  - `T`: Element type used for the guarded zero.

# Returns

  - The pairwise co-movement statistic.

# Related

  - [`gerber_comovement!`](@ref)
  - [`comovement_finalise`](@ref): the caller that reaches this function once per pair.
  - [`standardise_comovement!`](@ref)
  - [`comovement_unit_diagonal!`](@ref): the function that corrects the guarded zero on the diagonal.
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

Only the [`GerberComovementTwo`](@ref) markers reach the acting method; the fall-through method is a no-op, so every caller may call this unconditionally. **It writes into `rho` and into nothing else**, so the marker it is handed and the estimator that owns the marker are unchanged afterwards.

**The Gerber IQ family does not call this function.** Its thresholds move with the pair whenever `sc` is not pair-separable, so an asset's magnitude class off the diagonal is not the class the assembled diagonal records, and the ratio leaves `[-1, 1]`. [`gerber_IQ`](@ref) divides by the pair's own two diagonal projections instead, which [`iq_add_diagonal`](@ref) accumulates. The other three families threshold each asset in its own units, so the assembled diagonal is the same number and this function stands. ADR 0094 records the split.

# Algorithm

The acting method runs these steps. The fall-through method runs none of them.

 1. Take the square roots of the diagonal of `rho`, clamped from below by `sqrt(eps(eltype(rho)))`, giving `h`. The clamp is what keeps an asset that admits no observation from a division by zero.
 2. Divide `rho` element-wise by the outer product `h * transpose(h)`, and write the upper triangle back symmetrically.

# Arguments

  - `alg`: Co-movement algorithm marker. It selects the acting method or the no-op.
  - `rho`: `N × N` co-movement matrix, overwritten.

# Returns

  - `nothing`.

# Related

  - [`comovement_ratio`](@ref): the reduction whose [`GerberComovementTwo`](@ref) branch leaves the net score for this function to normalise.
  - [`comovement_unit_diagonal!`](@ref): the repair of the diagonal that runs immediately after this function.
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

The correlation of an asset with itself is one by definition, whatever the sample holds. Every reduction of [`comovement_ratio`](@ref) already returns one there when the asset crosses its threshold at least once, so only the degenerate case is left. An asset that crosses no threshold gives a zero denominator for every pair it belongs to, takes the guarded `zero(T)` over its whole row, and takes it on its diagonal entry too. **It writes into `rho` and into nothing else**, so every other entry is unchanged afterwards. ADR 0093 records the decision.

**The write is guarded by `iszero`, and does not restate a diagonal that is already one.** A `2` marker divides the diagonal by its own square root twice, so its diagonal entry is one to within a unit in the last place rather than exactly one. [`posdef!`](@ref) reads its diagonal with an exact `isone` test to decide whether it holds a correlation matrix or a covariance matrix, and the two branches answer differently. Writing an exact one over an entry that already reads as one moves that branch, and with it the answer of a sample that carries no degenerate asset. The guard keeps this function to the defect it fixes.

# Algorithm

 1. For each index `i` of the diagonal, write `one(eltype(rho))` onto `rho[i, i]` when that entry is zero.

# Arguments

  - `rho`: `N × N` co-movement matrix, whose zero diagonal entries are overwritten.

# Returns

  - `nothing`.

# Related

  - [`comovement_ratio`](@ref): the reduction whose guarded zero reaches the diagonal.
  - [`standardise_comovement!`](@ref): the normalisation that runs immediately before this function.
  - [`gerber_comovement!`](@ref)
  - [`posdef!`](@ref): the repair that runs immediately after this function, and that a zero diagonal turns into a `NaN`.
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

Fill the symmetric co-movement matrix `rho` by running the shared Gerber-family pairwise kernel.

The policy object `pol` (for example [`SmythBrobyKernel`](@ref) or [`GerberIQKernel`](@ref)) owns the thresholding, the classification and the weighting of a single observation, and the reduction of a pair's accumulator. The loop skeleton lives here once. **It writes into `rho` and into nothing else**, so `X` and `pol` are unchanged afterwards.

# Algorithm

 1. Read the observation count `T` from the first dimension of `X`.
 2. For every asset pair `(i, j)` with `i` at most `j`, run steps 3 to 6. The outer index is parallelised over the executor `ex`.
 3. Build the pair state `st` with [`comovement_pair_state`](@ref).
 4. Open the accumulator `acc` as the named tuple `(pos, neg, nn, cpos, cneg, cnn, di, dj)`, with the five scores at zero of `eltype(X)` and the three counts at integer zero. A policy reads the slots its marker needs and leaves the rest at zero. `di` and `dj` carry the two diagonal projections that the Gerber IQ `2` marker divides by; [`iq_add_diagonal`](@ref) is their only writer.
 5. Fold every observation `k` of the pair through [`comovement_step`](@ref) into `acc`.
 6. Reduce `acc` with [`comovement_finalise`](@ref) and write the result into `rho[i, j]` and `rho[j, i]`.

# Arguments

  - `rho::AbstractMatrix`: `N × N` output matrix, overwritten.
  - `ex`: `FLoops` executor parallelising over the outer asset index.
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

Co-movement policy for [`gerber_comovement!`](@ref) implementing the Smyth-Broby family.

The confusion zone thresholds the raw, uncentred return by `c1 * sigma`. Observations that pass it are centred and standardised per asset, restricted to the significance zone by `c2` through [`sb_crossed`](@ref) and by `c3`, and classified by the sign of the product of the standardised returns. The `alg` marker selects the accumulation family ([`sb_add_pos`](@ref)) and the denominator policy ([`comovement_ratio`](@ref)). This type is configuration handed to [`gerber_comovement!`](@ref); it holds no result and it is never mutated.

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

  - $(ref_dict[:smyth2022enhanced])
"""
struct SmythBrobyKernel{T1 <: SmythBrobyCovarianceAlgorithm, T2 <: ArrNum, T3 <: ArrNum,
                        T4 <: Number, T5 <: Number, T6 <: Number, T7 <: Number}
    alg::T1
    mu::T2
    sd::T3
    c1::T4
    c2::T5
    c3::T6
    n::T7
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the per-pair state consumed by [`comovement_step`](@ref).

The state holds every quantity that depends on the pair but not on the observation, so the inner loop reads it rather than recomputing it. The Gerber IQ method returns the pair's threshold scaling factors and scaled thresholds instead.

# Algorithm

The Smyth-Broby method runs these steps.

 1. Read the two standard deviations `pol.sd[i]` and `pol.sd[j]`, giving `sigmai` and `sigmaj`.
 2. Multiply each by `pol.c1`, giving the two confusion-zone thresholds `c1i` and `c1j`.
 3. Return the named tuple `(mui, muj, sigmai, sigmaj, c1i, c1j)`, with the two means read from `pol.mu`.

# Arguments

  - `pol`: Co-movement policy object.
  - `i`, `j`: Indices of the two assets of the pair.

# Returns

  - The per-pair state, as a named tuple.

# Related

  - [`gerber_comovement!`](@ref)
  - [`comovement_step`](@ref): the consumer of this state.
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

Union of the Smyth-Broby markers that accumulate the [`sb_delta`](@ref) contributions only, and discard the counts.

The group exists because the marker **prefix** selects the score triple while the trailing digit selects the denominator, and the two choices are independent. This is the prefix the source itself defines: it sums contributions and counts no votes.

# Related

  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
  - [`sb_add_pos`](@ref): the accumulator that dispatches on this alias.
  - [`sb_pair_scores`](@ref): the selector that dispatches on this alias.
  - [`SmythBrobyGerberAlg`](@ref)
  - [`SmythBrobyCountAlg`](@ref)
"""
const SmythBrobyDeltaAlg = Union{<:SmythBroby0, <:SmythBroby1, <:SmythBroby2}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Smyth-Broby markers that accumulate both the [`sb_delta`](@ref) contributions and the co-movement counts, and score a pair by their product.

The group exists because the marker **prefix** selects the score triple while the trailing digit selects the denominator, and the two choices are independent. The source's conclusion suggests keeping a count beside the sum and states no formula for it, so the product is the library's reading of that sentence.

# Related

  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`sb_add_pos`](@ref): the accumulator that dispatches on this alias.
  - [`sb_pair_scores`](@ref): the selector that dispatches on this alias.
  - [`SmythBrobyDeltaAlg`](@ref)
  - [`SmythBrobyCountAlg`](@ref)
"""
const SmythBrobyGerberAlg = Union{<:SmythBrobyGerber0, <:SmythBrobyGerber1,
                                  <:SmythBrobyGerber2}
"""
$(DocStringExtensions.TYPEDEF)

Union of the Smyth-Broby markers that accumulate the co-movement counts only, and never evaluate [`sb_delta`](@ref).

The group exists because the marker **prefix** selects the score triple while the trailing digit selects the denominator, and the two choices are independent. This prefix recovers a Gerber statistic evaluated on the Smyth-Broby zoning, so it is the library's own construction: the source counts no votes.

# Related

  - [`SmythBrobyCount0`](@ref)
  - [`SmythBrobyCount1`](@ref)
  - [`SmythBrobyCount2`](@ref)
  - [`sb_add_pos`](@ref): the accumulator that dispatches on this alias.
  - [`sb_pair_scores`](@ref): the selector that dispatches on this alias.
  - [`GerberCovariance`](@ref): the statistic this prefix recovers when the confusion zone is switched off, the outer cut-off is lifted, and the centre is zero.
  - [`SmythBrobyDeltaAlg`](@ref)
  - [`SmythBrobyGerberAlg`](@ref)
"""
const SmythBrobyCountAlg = Union{<:SmythBrobyCount0, <:SmythBrobyCount1, <:SmythBrobyCount2}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate a concordant observation into the pair accumulator, according to the Smyth-Broby family of `alg`.

The accumulator is a named tuple and every method returns a new one, so nothing is mutated.

# Algorithm

The marker prefix selects one of three branches.

 1. [`SmythBrobyDeltaAlg`](@ref): add [`sb_delta`](@ref) to `acc.pos`, and leave `acc.cpos` alone.
 2. [`SmythBrobyGerberAlg`](@ref): add [`sb_delta`](@ref) to `acc.pos`, and add one to `acc.cpos`.
 3. [`SmythBrobyCountAlg`](@ref): add one to `acc.cpos`, and never evaluate [`sb_delta`](@ref).

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.
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

Accumulate a discordant observation into the pair accumulator, according to the Smyth-Broby family of `alg`.

It mirrors [`sb_add_pos`](@ref) on the discordant score and count, branch for branch.

# Algorithm

The marker prefix selects one of three branches.

 1. [`SmythBrobyDeltaAlg`](@ref): add [`sb_delta`](@ref) to `acc.neg`, and leave `acc.cneg` alone.
 2. [`SmythBrobyGerberAlg`](@ref): add [`sb_delta`](@ref) to `acc.neg`, and add one to `acc.cneg`.
 3. [`SmythBrobyCountAlg`](@ref): add one to `acc.cneg`, and never evaluate [`sb_delta`](@ref).

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.
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

Accumulate a neutral observation into the pair accumulator. An observation is neutral when exactly one of the two assets left the indecision zone, which [`sb_crossed`](@ref) decides.

Only the [`GerberComovementOne`](@ref) markers reach an acting method, because they are the only ones whose denominator carries a neutral term. The fall-through method returns the accumulator unchanged, so the neutral score of every other marker stays at zero and is read by no reduction.

# Algorithm

The marker selects one of four branches.

 1. [`SmythBroby1`](@ref): add [`sb_delta`](@ref) to `acc.nn`.
 2. [`SmythBrobyGerber1`](@ref): add [`sb_delta`](@ref) to `acc.nn`, and add one to `acc.cnn`.
 3. [`SmythBrobyCount1`](@ref): add one to `acc.cnn`, and never evaluate [`sb_delta`](@ref).
 4. Any other [`SmythBrobyCovarianceAlgorithm`](@ref): return `acc` unchanged.

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.
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

Decide whether one asset left the indecision zone at one observation.

An asset leaves the indecision zone when the magnitude of its centred, standardised return reaches `c2` **and** that return is not exactly zero. The sign test is redundant for a positive threshold, because `ar >= c > 0` already implies that `r` is not zero. It binds only at `c2 = 0`, where the closed comparison `ar >= 0` holds for every return, including one that is exactly zero. ADR 0090 settled that a return of exactly zero never crosses, and this is that rule for the Smyth-Broby family.

The rule is what keeps the diagonal of the statistic at one. The pair `(i, i)` either crosses on both axes or on neither, so it never reaches the neutral accumulator that a [`GerberComovementOne`](@ref) marker divides by. Without the sign test a zero return crossed on both axes but carried no sign, so it fell through to that accumulator and pulled the diagonal below one.

**The rule binds on this gate and not on the confusion zone.** The two gates read different quantities. Here the quantity is centred, so an exactly zero return is an asset that did not move away from its own mean, and it has no sign to classify. The confusion zone reads the **raw, uncentred** return, whose zero is an arbitrary point of the scale of the data: an asset whose raw return is zero moved by ``-\\mu`` against its mean, which is a deviation with a sign. That gate also only rejects and never classifies, so it produces no wrong count of its own. ADR 0090 records the asymmetry.

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

Fold one observation of a pair into the co-movement accumulator.

**The confusion zone reads the raw, uncentred return and the indecision zone reads the centred, standardised one.** That mix is the source's, not an oversight, and centring the confusion zone as well moves the statistic. The mix also decides which gate carries the rule of ADR 0090: [`sb_crossed`](@ref) keeps a return of exactly zero inside the indecision zone, and the confusion zone takes no such test, because the zero of a raw return is an arbitrary point of the scale of the data. The Gerber IQ method thresholds absolute returns against the pair's scaled thresholds with [`iq_crossed`](@ref), and weights observations by the IQ template and temporal decay via [`gerber_IQ_delta`](@ref).

# Algorithm

The Smyth-Broby method runs these steps. It reads `T` and `k` in neither, because the family applies no temporal decay.

 1. Return `acc` unchanged when `abs(xi)` is below `st.c1i` **and** `abs(xj)` is below `st.c1j`. This is the confusion zone, read on the raw return.
 2. Centre and standardise both returns with the pair state, giving `ri` and `rj`, and take their magnitudes `ari` and `arj`.
 3. Decide with [`sb_crossed`](@ref) whether each asset left the indecision zone of `pol.c2`.
 4. Return `acc` unchanged when either magnitude exceeds `pol.c3`, or when neither asset left the indecision zone. The first is the outer cut-off and the second is the indecision zone itself.
 5. Accumulate through [`sb_add_pos`](@ref) when both assets crossed and the product `ri * rj` is positive.
 6. Accumulate through [`sb_add_neg`](@ref) when both assets crossed and the product is negative.
 7. Accumulate through [`sb_add_neutral`](@ref) otherwise, which is the case where exactly one asset crossed. Two crossings give a product that is not zero, so the three branches are exhaustive.

# Arguments

  - `pol`: Co-movement policy object.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.
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

Select the pair's concordant, discordant and neutral scores from the accumulator, according to the Smyth-Broby family of `alg`.

This is the half of the marker that the **prefix** owns. [`comovement_ratio`](@ref) owns the other half, which the trailing digit selects.

# Algorithm

The marker prefix selects one of three branches.

 1. [`SmythBrobyDeltaAlg`](@ref): return the contribution sums `(acc.pos, acc.neg, acc.nn)`.
 2. [`SmythBrobyGerberAlg`](@ref): return each sum times its own count, `(acc.pos * acc.cpos, acc.neg * acc.cneg, acc.nn * acc.cnn)`.
 3. [`SmythBrobyCountAlg`](@ref): return the counts `(acc.cpos, acc.cneg, acc.cnn)`.

# Arguments

  - `alg`: Smyth-Broby algorithm marker.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.

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

Reduce a pair's accumulator to the pairwise co-movement statistic.

It joins the two halves of the marker: the prefix chooses the scores and the trailing digit chooses the denominator.

# Algorithm

 1. Select the score triple `(p, n, nn)` from `acc` with [`sb_pair_scores`](@ref), which the marker prefix dispatches.
 2. Reduce the triple with [`comovement_ratio`](@ref), which the trailing digit dispatches, and return the result.

# Arguments

  - `pol`: Co-movement policy object.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.
  - `T`: Element type used for the guarded zero.

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

Compute the Smyth-Broby co-movement correlation matrix for the algorithm marker in `ce.alg`.

All nine variants share the pairwise kernel [`gerber_comovement!`](@ref) through a [`SmythBrobyKernel`](@ref) policy: observations are noise-gated by `c1 * sigma`, standardised, restricted to the `[c2, c3]` significance zone by [`sb_crossed`](@ref), and classified as concordant, discordant, or neutral by the sign of the product of standardised returns. The marker selects the accumulation family and denominator policy.

# Mathematical definition

For each pair ``(i, j)`` an observation ``t`` passes two admission tests. The **noise gate** compares the **raw, uncentred** return against ``c_1 \\sigma``, and rejects ``t`` only when both assets fall inside it:

```math
\\begin{align}
|x_{ti}| < c_1 \\sigma_i \\quad \\text{and} \\quad |x_{tj}| < c_1 \\sigma_j\\,.
\\end{align}
```

The **significance zone** compares the **centred, standardised** return ``\\tilde{r}_{ti} = (x_{ti} - \\mu_i) / \\sigma_i``. Asset ``i`` crosses at ``t`` when

```math
\\begin{align}
|\\tilde{r}_{ti}| \\geq c_2 \\quad \\text{and} \\quad \\tilde{r}_{ti} \\neq 0\\,,
\\end{align}
```

and the zone rejects ``t`` when either asset exceeds ``c_3`` or neither asset crosses. The second test of the crossing binds only at ``c_2 = 0``, because ``|\\tilde{r}| \\geq c_2 > 0`` already excludes a zero return. It is the rule of ADR 0090 for this family, and [`sb_crossed`](@ref) is where the code states it. The gate reads the uncentred return and the zone reads the centred one; this mix is the source's, not an oversight. Centering the gate as well moves the statistic, and the rule of ADR 0090 binds on the centred quantity alone.

An admitted observation is concordant when both assets cross and ``\\tilde{r}_{ti} \\tilde{r}_{tj} > 0``, discordant when both cross and the product is negative, and neutral otherwise, which is the case where exactly one asset crosses. Accumulate the kernel and the count of each class over the admitted observations:

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

with ``\\delta_t = \\delta(|\\tilde{r}_{t,\\,i}|, |\\tilde{r}_{t,\\,j}|, n)`` the [`sb_delta`](@ref) contribution. The marker prefix selects the three scores ``(p, q, u)``:

  - `SmythBroby*`: ``(\\text{pos},\\, \\text{neg},\\, \\text{nn})``.
  - `SmythBrobyGerber*`: ``(\\text{pos} \\, c^{+},\\, \\text{neg} \\, c^{-},\\, \\text{nn} \\, c^{0})``. Every term carries its own count, the neutral one included.
  - `SmythBrobyCount*`: ``(c^{+},\\, c^{-},\\, c^{0})``.

The variant number selects the reduction, with ``h_{ij} = p - q`` the raw difference:

```math
\\begin{align}
\\hat{\\rho}_{ij} &= \\begin{cases}
(p - q) / (p + q) & 0 \\\\
(p - q) / (p + q + u) & 1 \\\\
h_{ij} / \\sqrt{h_{ii} \\, h_{jj}} & 2
\\end{cases}\\,.
\\end{align}
```

Variants 0 and 1 return zero when their denominator vanishes. Variant 2 divides the **net** score matrix by the geometric mean of its own diagonal, with the roots clamped below at ``\\sqrt{\\varepsilon}``. It does **not** normalise the variant 0 ratio: the two agree only where ``p + q`` is constant across pairs.

**Only the `SmythBroby1` composition is the source's own.** It is equation (5) of the source, and the source defines no other statistic. The trailing `0` and `2` are the library's reductions, on the shape [`Gerber0`](@ref) and [`Gerber2`](@ref) set. The `SmythBrobyGerber*` prefix reads one sentence of the source's conclusion, which suggests keeping a count beside the sum and states no formula. The `SmythBrobyCount*` prefix is the library's own: the source counts no votes, and that prefix recovers the Gerber statistic on the Smyth-Broby zoning.

# Algorithm

 1. Read the asset count `N` from the second dimension of `X`, and open the `N × N` output `rho`.
 2. Build the policy `pol` as a [`SmythBrobyKernel`](@ref) from `ce.alg`, `mu`, `sd`, `ce.c1`, `ce.c2`, `ce.c3` and `ce.n`.
 3. Fill `rho` with [`gerber_comovement!`](@ref), over the executor `ce.ex`.
 4. Normalise `rho` in place with [`standardise_comovement!`](@ref). Only a `2` marker changes it.
 5. Write one onto a zero diagonal entry of `rho` with [`comovement_unit_diagonal!`](@ref). An asset that qualifies no observation reduces to a zero diagonal entry, and that entry is one by definition.
 6. Repair `rho` with [`posdef!`](@ref) and the estimator's `pdm`. A Smyth-Broby matrix is a matrix of pairwise scores and is not positive definite in general.

# Arguments

  - `ce`: Smyth-Broby covariance estimator.
  - $(arg_dict[:X])
  - `mu`: Vector of asset means, one entry per asset.
  - $(arg_dict[:stdarr])

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyKernel`](@ref)
  - [`gerber_comovement!`](@ref)
  - [`standardise_comovement!`](@ref)
  - [`comovement_unit_diagonal!`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
  - [`GerberCovariance`](@ref): the statistic this family extends. Its matrix form is stated there and is not repeated here.

# References

  - $(ref_dict[:smyth2022enhanced])
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

The mean and the standard deviation are computed by the estimator's own `me` and `ve`, so the centre and the scale that the zoning reads are the estimator's choice and not this method's.

# Algorithm

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref).
 2. Compute the standard deviation of each column with `ce.ve`, giving `sd`, and clamp it from below by `eps(eltype(sd))`. The clamp keeps a constant column from dividing by zero.
 3. Compute the mean of each column with `ce.me`, giving `mu`.
 4. Return the matrix that [`smythbroby`](@ref) builds from `X`, `mu` and `sd`.

# Arguments

  - `ce`: Smyth-Broby covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

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
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    mu = Statistics.mean(ce.me, X; dims = 1, kwargs...)
    return smythbroby(ce, X, mu, sd)
end
"""
    Statistics.cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Smyth-Broby covariance matrix.

The correlation matrix is rescaled by the same `sd` the zoning read, so the covariance is exactly the correlation times the outer product of `sd`, and its diagonal is exactly `sd .^ 2`.

# Algorithm

 1. Orient `X` to observations × assets with [`dims_oriented`](@ref).
 2. Compute the standard deviation of each column with `ce.ve`, giving `sd`, and clamp it from below by `eps(eltype(sd))`.
 3. Compute the mean of each column with `ce.me`, giving `mu`.
 4. Build the correlation matrix `sigma` with [`smythbroby`](@ref) from `X`, `mu` and `sd`.
 5. Rescale `sigma` in place with `StatsBase.cor2cov!` and `sd`, and return it.

# Arguments

  - `ce`: Smyth-Broby covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

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
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    mu = Statistics.mean(ce.me, X; dims = 1, kwargs...)
    sigma = smythbroby(ce, X, mu, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export SmythBroby0, SmythBroby1, SmythBroby2, SmythBrobyGerber0, SmythBrobyGerber1,
       SmythBrobyGerber2, SmythBrobyCount0, SmythBrobyCount1, SmythBrobyCount2,
       SmythBrobyCovariance
