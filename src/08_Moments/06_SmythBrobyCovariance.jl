"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance estimators.

All concrete and/or abstract types implementing Smyth-Broby covariance estimation algorithms should be subtypes of `BaseSmythBrobyCovariance`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced])
"""
abstract type BaseSmythBrobyCovariance <: BaseGerberCovariance end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Smyth-Broby covariance algorithm types.

All concrete and/or abstract types implementing specific Smyth-Broby covariance algorithms should be subtypes of `SmythBrobyCovarianceAlgorithm`.

These types are used to specify the algorithm when constructing a [`SmythBrobyCovariance`](@ref) estimator.

# Related

  - [`BaseSmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovariance`](@ref)

# References

  - $(ref_dict[:smyth2022enhanced])
"""
abstract type SmythBrobyCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Divides the difference of the concordant and discordant Smyth-Broby kernel sums by their sum.

The kernel is [`sb_delta`](@ref), and an observation votes only when both standardised returns reach `c2`. This is the canonical statistic of the source reduced to a two-term denominator.

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

Divides the difference of the concordant and discordant Smyth-Broby kernel sums by their sum plus the neutral sum.

The neutral sum accumulates the kernel over the observations on which exactly one asset reaches `c2`. This is the canonical Smyth-Broby statistic.

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

Normalises the raw difference of the Smyth-Broby kernel sums by the geometric mean of its own diagonal.

The pairwise statistic is ``h_{ij} / \\sqrt{h_{ii} h_{jj}}`` with ``h_{ij}`` the raw difference, so the diagonal is unit by construction rather than by a per-pair denominator.

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

Weights each Smyth-Broby kernel sum by its own observation count, then divides the difference by the sum.

The weighting reintroduces the Gerber vote count that [`SmythBroby0`](@ref) discards, so a pair that co-moves often scores above one that co-moves rarely but sharply.

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

Weights each Smyth-Broby kernel sum by its own count, then divides the difference by the sum plus the neutral term.

The neutral term carries its own count as well, so all three terms of the denominator are scaled alike. This is the estimator's default algorithm.

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

Weights each Smyth-Broby kernel sum by its own count, and standardises the raw difference by its own diagonal.

The pairwise statistic is ``h_{ij} / \\sqrt{h_{ii} h_{jj}}`` with ``h_{ij}`` the count-weighted raw difference.

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

Counts concordant and discordant observations, discards the kernel, and divides their difference by their sum.

Dropping [`sb_delta`](@ref) recovers a Gerber statistic evaluated on the Smyth-Broby admission rule rather than on the Gerber threshold.

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

Counts concordant, discordant and neutral observations, discards the kernel, and divides the net count by the total.

Dropping [`sb_delta`](@ref) recovers a Gerber statistic evaluated on the Smyth-Broby admission rule rather than on the Gerber threshold.

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

Counts concordant and discordant observations, discards the kernel, and standardises the net count by its own diagonal.

The pairwise statistic is ``h_{ij} / \\sqrt{h_{ii} h_{jj}}`` with ``h_{ij}`` the raw difference of the two counts.

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

`SmythBrobyCovariance` encapsulates all components required for Smyth-Broby-based covariance or correlation estimation, including the expected returns estimator, variance estimator, positive definite matrix estimator, algorithm parameters, and the specific Smyth-Broby algorithm variant.

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

  - $(val_dict[:c3c2])

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
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)

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

Smyth-Broby kernel function for covariance and correlation computation.

This function computes the kernel value for a pair of asset returns, applying the Smyth-Broby logic for zones of confusion and indecision. It is used to aggregate positive and negative co-movements in Smyth-Broby covariance algorithms. It assumes the returns are centered around zero.

# Mathematical definition

```math
\\begin{align}
\\kappa(r_i, r_j) &= \\sqrt{(1 + |r_i|)(1 + |r_j|)}\\,, \\\\
\\gamma(r_i, r_j) &= |r_i - r_j|\\,.
\\end{align}
```

Where:

  - ``\\kappa(r_i, r_j)``: Amplitude kernel.
  - ``\\gamma(r_i, r_j)``: Divergence measure between returns.
  - ``r_i, r_j``: Absolute standardised returns for assets ``i`` and ``j``.

```math
\\begin{align}
\\delta(r_i, r_j, n) &= \\frac{\\kappa(r_i, r_j)}{1 + \\gamma(r_i, r_j)^n}\\,.
\\end{align}
```

Where:

  - ``\\delta(r_i, r_j, n)``: Smyth-Broby kernel value.
  - ``n``: Exponent parameter controlling kernel sharpness.

# Arguments

  - `ri`: Absolute standardised return for asset `i`.
  - `rj`: Absolute standardised return for asset `j`.
  - `n`: Exponent parameter for the kernel.

# Returns

  - `score::Number`: The computed score for the pair `(xi, xj)`.

# Details

  - Returns `(sqrt((1 + ri) * (1 + rj)) / (1 + abs(ri - rj)^n), 1)`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`smythbroby`](@ref)
"""
function sb_delta(ri::Number, rj::Number, n::Number)
    kappa = sqrt((one(ri) + ri) * (one(rj) + rj))
    gamma = abs(ri - rj)
    return kappa / (one(gamma) + gamma^n)
end
"""
$(DocStringExtensions.TYPEDEF)

Union of Gerber-family co-movement algorithm markers whose pairwise statistic is `(pos - neg) / (pos + neg)`, guarded to zero when the denominator vanishes.

# Related

  - [`comovement_ratio`](@ref)
  - [`GerberComovementOne`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
const GerberComovementZero = Union{<:Gerber0, <:SmythBroby0, <:SmythBrobyGerber0,
                                   <:SmythBrobyCount0}
"""
$(DocStringExtensions.TYPEDEF)

Union of Gerber-family co-movement algorithm markers whose pairwise statistic is `(pos - neg) / (pos + neg + nn)`, including neutral co-movements in the denominator, guarded to zero when the denominator vanishes.

# Related

  - [`comovement_ratio`](@ref)
  - [`GerberComovementZero`](@ref)
  - [`GerberComovementTwo`](@ref)
"""
const GerberComovementOne = Union{<:Gerber1, <:SmythBroby1, <:SmythBrobyGerber1,
                                  <:SmythBrobyCount1}
"""
$(DocStringExtensions.TYPEDEF)

Union of Gerber-family co-movement algorithm markers whose pairwise statistic is the raw `pos - neg`, with the resulting matrix standardised by the geometric mean of its diagonal via [`standardise_comovement!`](@ref).

# Related

  - [`comovement_ratio`](@ref)
  - [`GerberComovementZero`](@ref)
  - [`GerberComovementOne`](@ref)
"""
const GerberComovementTwo = Union{<:Gerber2, <:SmythBroby2, <:SmythBrobyGerber2,
                                  <:SmythBrobyCount2}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reduce a pair's accumulated positive, negative, and neutral co-movement scores to the pairwise correlation entry.

The variant marker selects the denominator policy:

  - [`GerberComovementZero`](@ref): `(p - n) / (p + n)`, or `zero(T)` when the denominator is zero.
  - [`GerberComovementOne`](@ref): `(p - n) / (p + n + nn)`, or `zero(T)` when the denominator is zero.
  - [`GerberComovementTwo`](@ref): raw `p - n`; the matrix is standardised afterwards by [`standardise_comovement!`](@ref).

# Arguments

  - `alg`: Co-movement algorithm marker.
  - `p`, `n`, `nn`: Accumulated positive, negative, and neutral scores.
  - `T`: Element type used for the guarded zero.

# Returns

  - The pairwise co-movement statistic.

# Related

  - [`gerber_comovement!`](@ref)
  - [`standardise_comovement!`](@ref)
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

Standardise a raw co-movement matrix by the geometric mean of its diagonal.

Only the [`GerberComovementTwo`](@ref) variants standardise; the fall-through method is a no-op. Divides each entry by `sqrt(rho[i, i] * rho[j, j])`, clamping the diagonal roots away from zero.

# Returns

  - `nothing`.

# Related

  - [`comovement_ratio`](@ref)
  - [`gerber_comovement!`](@ref)
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

Fill the symmetric co-movement matrix `rho` by running the shared Gerber-family pairwise kernel.

For every asset pair `(i, j)` the kernel builds the pair state via [`comovement_pair_state`](@ref), folds every observation through [`comovement_step`](@ref) into the accumulator `(pos, neg, nn, cpos, cneg, cnn)` of weighted scores and counts, and stores the reduction [`comovement_finalise`](@ref) symmetrically. The policy object `pol` (e.g. [`SmythBrobyKernel`](@ref), [`GerberIQKernel`](@ref)) owns the thresholding, classification, and weighting of a single observation, and the reduction of a pair's accumulator; the loop skeleton lives here once.

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
                   cpos = 0, cneg = 0, cnn = 0)
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

The noise gate thresholds the raw, uncentred return by `c1 * sigma`. Observations that pass it are centered and standardised per asset, restricted to the `[c2, c3]` significance zone, and classified by the sign of the product of standardised returns. The `alg` marker selects the accumulation family ([`sb_add_pos`](@ref)) and the denominator policy ([`comovement_ratio`](@ref)).

# Fields

  - `alg`: Smyth-Broby algorithm marker.
  - `mu`: Vector of asset means.
  - `sd`: Vector of asset standard deviations.
  - `c1`, `c2`, `c3`: Noise-gate and significance-zone thresholds.
  - `n`: Exponent of the [`sb_delta`](@ref) kernel.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`smythbroby`](@ref)
  - [`gerber_comovement!`](@ref)
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

The Smyth-Broby method returns the pair's means, standard deviations, and noise-gate thresholds `c1 * sigma`. The Gerber IQ method returns the pair's threshold scaling factors and scaled thresholds.

# Related

  - [`gerber_comovement!`](@ref)
"""
@inline function comovement_pair_state(pol::SmythBrobyKernel, i::Integer, j::Integer)
    sigmai = pol.sd[i]
    sigmaj = pol.sd[j]
    return (mui = pol.mu[i], muj = pol.mu[j], sigmai = sigmai, sigmaj = sigmaj,
            c1i = pol.c1 * sigmai, c1j = pol.c1 * sigmaj)
end
"""
$(DocStringExtensions.TYPEDEF)

Union of Smyth-Broby algorithm markers that accumulate the [`sb_delta`](@ref) kernel values only.

# Related

  - [`sb_add_pos`](@ref)
"""
const SmythBrobyDeltaAlg = Union{<:SmythBroby0, <:SmythBroby1, <:SmythBroby2}
"""
$(DocStringExtensions.TYPEDEF)

Union of Smyth-Broby algorithm markers that accumulate both the [`sb_delta`](@ref) kernel values and co-movement counts, scoring pairs by their product.

# Related

  - [`sb_add_pos`](@ref)
"""
const SmythBrobyGerberAlg = Union{<:SmythBrobyGerber0, <:SmythBrobyGerber1,
                                  <:SmythBrobyGerber2}
"""
$(DocStringExtensions.TYPEDEF)

Union of Smyth-Broby algorithm markers that accumulate co-movement counts only.

# Related

  - [`sb_add_pos`](@ref)
"""
const SmythBrobyCountAlg = Union{<:SmythBrobyCount0, <:SmythBrobyCount1, <:SmythBrobyCount2}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate a concordant observation into the pair accumulator, according to the Smyth-Broby family of `alg`.

  - [`SmythBrobyDeltaAlg`](@ref): adds [`sb_delta`](@ref) to the weighted score.
  - [`SmythBrobyGerberAlg`](@ref): adds [`sb_delta`](@ref) to the weighted score and increments the count.
  - [`SmythBrobyCountAlg`](@ref): increments the count only.

# Related

  - [`sb_add_neg`](@ref)
  - [`sb_add_neutral`](@ref)
  - [`comovement_step`](@ref)
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

Mirrors [`sb_add_pos`](@ref) on the negative score and count.

# Related

  - [`sb_add_pos`](@ref)
  - [`sb_add_neutral`](@ref)
  - [`comovement_step`](@ref)
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

Accumulate a neutral (one-sided) observation into the pair accumulator.

Only the [`GerberComovementOne`](@ref) variants track neutral co-movements, mirroring [`sb_add_pos`](@ref) on the neutral score and count; the fall-through method returns the accumulator unchanged.

# Related

  - [`sb_add_pos`](@ref)
  - [`sb_add_neg`](@ref)
  - [`comovement_step`](@ref)
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

Fold one observation of a pair into the co-movement accumulator.

The Smyth-Broby method skips observations inside the noise gate (`|x| < c1 * sigma` for both assets, on the raw uncentred return), then centers and standardises the returns. It skips an observation whose standardised return exceeds `c3` on either asset or falls below `c2` on both, and classifies the rest as concordant, discordant, or neutral by the sign of the product of standardised returns. The Gerber IQ method thresholds absolute returns against the pair's scaled thresholds and weights observations by the IQ template and temporal decay via [`gerber_IQ_delta`](@ref).

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
    if ari > pol.c3 || arj > pol.c3 || ari < c2 && arj < c2
        return acc
    end
    return if ari >= c2 && arj >= c2 && ri * rj > zero(ri)
        sb_add_pos(pol.alg, acc, ari, arj, pol.n)
    elseif ari >= c2 && arj >= c2 && ri * rj < zero(ri)
        sb_add_neg(pol.alg, acc, ari, arj, pol.n)
    else
        sb_add_neutral(pol.alg, acc, ari, arj, pol.n)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Select the pair's positive, negative, and neutral scores from the accumulator, according to the Smyth-Broby family of `alg`.

  - [`SmythBrobyDeltaAlg`](@ref): the weighted scores.
  - [`SmythBrobyGerberAlg`](@ref): the products of weighted scores and counts.
  - [`SmythBrobyCountAlg`](@ref): the counts.

# Related

  - [`comovement_finalise`](@ref)
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

Selects the family scores (for Smyth-Broby, via [`sb_pair_scores`](@ref)) and applies the variant's denominator policy via [`comovement_ratio`](@ref).

# Related

  - [`gerber_comovement!`](@ref)
  - [`comovement_step`](@ref)
"""
@inline function comovement_finalise(pol::SmythBrobyKernel, acc, ::Type{T}) where {T}
    p, n, nn = sb_pair_scores(pol.alg, acc)
    return comovement_ratio(pol.alg, p, n, nn, T)
end
"""
    smythbroby(ce::SmythBrobyCovariance, X::MatNum, mu::ArrNum, sd::ArrNum)

Compute the Smyth-Broby co-movement correlation matrix for the algorithm marker in `ce.alg`.

All nine variants share the pairwise kernel [`gerber_comovement!`](@ref) through a [`SmythBrobyKernel`](@ref) policy: observations are noise-gated by `c1 * sigma`, standardised, restricted to the `[c2, c3]` significance zone, and classified as concordant, discordant, or neutral by the sign of the product of standardised returns. The marker selects the accumulation family and denominator policy.

# Mathematical definition

For each pair ``(i, j)`` an observation ``t`` passes two admission tests. The **noise gate** compares the **raw, uncentred** return against ``c_1 \\sigma``, and rejects ``t`` only when both assets fall inside it:

```math
\\begin{align}
|x_{ti}| < c_1 \\sigma_i \\quad \\text{and} \\quad |x_{tj}| < c_1 \\sigma_j\\,.
\\end{align}
```

The **significance zone** compares the **centred, standardised** return ``\\tilde{r}_{ti} = (x_{ti} - \\mu_i) / \\sigma_i``, and rejects ``t`` when either asset exceeds ``c_3`` or both fall below ``c_2``. The gate reads the uncentred return and the zone reads the centred one; this mix is the source's, not an oversight. Centering the gate as well moves the statistic.

An admitted observation is concordant when both ``|\\tilde{r}|`` reach ``c_2`` and ``\\tilde{r}_{ti} \\tilde{r}_{tj} > 0``, discordant when both reach ``c_2`` and the product is negative, and neutral otherwise. Accumulate the kernel and the count of each class over the admitted observations:

```math
\\begin{align}
\\text{pos} &= \\sum_t \\delta_t \\, \\mathbf{1}[t \\in C]\\,, \\quad \\text{neg} = \\sum_t \\delta_t \\, \\mathbf{1}[t \\in D]\\,, \\quad \\text{nn} = \\sum_t \\delta_t \\, \\mathbf{1}[t \\in N]\\,, \\\\
c^{+} &= |C|\\,, \\quad c^{-} = |D|\\,, \\quad c^{0} = |N|\\,,
\\end{align}
```

with ``\\delta_t = \\delta(|\\tilde{r}_{ti}|, |\\tilde{r}_{tj}|, n)`` the [`sb_delta`](@ref) kernel, and ``C``, ``D``, ``N`` the concordant, discordant and neutral observation sets. The marker prefix selects the three scores ``(p, q, u)``:

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

Variants 0 and 1 return zero when their denominator vanishes. Variant 2 divides the **raw** difference matrix by the geometric mean of its own diagonal, with the roots clamped below at ``\\sqrt{\\varepsilon}``. It does **not** normalise the variant 0 ratio: the two agree only where ``p + q`` is constant across pairs.

# Arguments

  - `ce`: Smyth-Broby covariance estimator.
  - $(arg_dict[:X])
  - `mu`: Vector of asset means.
  - `sd`: Vector of asset standard deviations.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyKernel`](@ref)
  - [`gerber_comovement!`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance, X::MatNum, mu::ArrNum, sd::ArrNum)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    pol = SmythBrobyKernel(ce.alg, mu, sd, ce.c1, ce.c2, ce.c3, ce.n)
    gerber_comovement!(rho, ce.ex, X, pol)
    standardise_comovement!(ce.alg, rho)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    Statistics.cor(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Smyth-Broby correlation matrix.

This method computes the Smyth-Broby correlation matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Smyth-Broby correlation is then computed via [`smythbroby`](@ref).

# Arguments

  - `ce`: Smyth-Broby covariance estimator.

      + `ce::SmythBrobyCovariance`: Compute the unstandardised Smyth-Broby correlation matrix.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `rho::Matrix{<:Number}`: The Smyth-Broby correlation matrix.

# Validation

  - `dims` is either `1` or `2`.

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

This method computes the Smyth-Broby covariance matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Smyth-Broby covariance is then computed via [`smythbroby`](@ref).

# Arguments

  - `ce`: Smyth-Broby covariance estimator.

      + `ce::SmythBrobyCovariance`: Compute the unstandardised Smyth-Broby covariance matrix.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `sigma::Matrix{<:Number}`: The Smyth-Broby covariance matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`smythbroby`](@ref)
  - [`cov(ce::SmythBrobyCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
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
