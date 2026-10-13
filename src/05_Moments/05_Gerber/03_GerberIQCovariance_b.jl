"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`GerberIQCovariance`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:GerberIQCovariance, true)` to render it.

# Arguments

  - `::GerberIQCovariance`: Gerber IQ covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ve, :me, :pdm, :c, :decay, :sc, :kind, :alg, :ex)`.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::GerberIQCovariance) = (:ve, :me, :pdm, :c, :decay, :sc, :kind, :alg, :ex)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the Gerber IQ statistic for a single co-movement.

This is the product of the two halves of the source's squeezing statistic: the spatial weight the template gives the co-movement, and the temporal discount its age earns. The result is the quantity that [`comovement_step`](@ref) adds into one of the pair's three accumulators.

# Mathematical definition

```math
\\begin{align}
\\Delta_{t,\\,i,\\,j} &= \\eta_{t,\\,i,\\,j} \\, v_{t}\\,.
\\end{align}
```

Where:

  - ``\\Delta_{t,\\,i,\\,j}``: Contribution of the co-movement of assets ``i`` and ``j`` at observation ``t``.
  - ``\\eta_{t,\\,i,\\,j}``: Squeezing weight of that co-movement, from the template.
  - ``v_{t}``: Temporal discount of observation ``t``, from the decay estimator.

# Algorithm

 1. Compute the squeezing weight `w` with [`gerber_iq_weight`](@ref), passing the pair's two scaling factors and the template.
 2. Compute the temporal discount `p` by calling the `decay` functor with `T` and `k`.
 3. Return `w * p`.

# Arguments

  - `xi`: Return for asset `i`.
  - `xj`: Return for asset `j`.
  - `axi`: Absolute return for asset `i`.
  - `axj`: Absolute return for asset `j`.
  - `decay`: The decay estimator for the Gerber IQ statistic.
  - `T`: The number of observations.
  - `k`: The current observation.
  - `sci`: Scaling factor for asset `i`.
  - `scj`: Scaling factor for asset `j`.
  - `kind`: The Gerber IQ co-movement template.

# Returns

  - `rho::Number`: The weighted, discounted contribution of one co-movement. It is not itself a correlation.

# Related

  - [`gerber_iq_weight`](@ref)
  - [`comovement_step`](@ref)
  - [`gerber_IQ`](@ref)
  - [`GerberIQDecayEstimator`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function gerber_IQ_delta(xi::Number, xj::Number, axi::Number, axj::Number,
                         decay::GerberIQDecayEstimator, T::Integer, k::Number, sci::Number,
                         scj::Number, kind::GerberIQCovarianceAlgorithm)
    w = gerber_iq_weight(xi, xj, axi, axj, sci, scj, kind)
    p = decay(T, k)
    return w * p
end
"""
$(DocStringExtensions.TYPEDEF)

Co-movement policy for [`gerber_comovement!`](@ref) implementing the Gerber IQ family.

Observations are thresholded against the pair's scaled thresholds from [`gerber_iq_scaling`](@ref), classified by the sign of the product of returns, and weighted by the IQ noise-compression template and temporal decay via [`gerber_IQ_delta`](@ref). The `alg` marker selects the denominator policy ([`comovement_ratio`](@ref)).

# Fields

  - `alg`: Gerber algorithm marker selecting the denominator policy.
  - `kind`: Gerber IQ noise-compression template.
  - `decay`: Regenerated temporal decay estimator.
  - `sc`: Threshold scaling factor estimator.
  - `c`: Small co-movement threshold.
  - `sd`: Vector of asset standard deviations.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`gerber_IQ`](@ref)
  - [`gerber_comovement!`](@ref)
"""
@concrete struct GerberIQKernel
    alg
    kind
    decay
    sc
    c
    sd
    function GerberIQKernel(alg::T1, kind::T2, decay::T3, sc::T4, c::T5,
                            sd::T6) where {T1 <: GerberCovarianceAlgorithm,
                                           T2 <: GerberIQCovarianceAlgorithm,
                                           T3 <: GerberIQDecayEstimator, T4, T5 <: Number,
                                           T6 <: ArrNum}
        return new{T1, T2, T3, T4, T5, T6}(alg, kind, decay, sc, c, sd)
    end
end
@inline function comovement_pair_state(pol::GerberIQKernel, i::Integer, j::Integer)
    sci, scj = gerber_iq_scaling(pol.sc, pol.sd[i], pol.sd[j])
    return (sci = sci, scj = scj, ci = sci * pol.c, cj = scj * pol.c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate a neutral (one-sided) observation into the Gerber IQ pair accumulator.

Only [`Gerber1`](@ref) tracks neutral co-movements, adding the [`gerber_IQ_delta`](@ref) weight to the neutral score; the fall-through method returns the accumulator unchanged.

A neutral co-movement is one on which exactly one of the two assets left the noise zone, which [`iq_crossed`](@ref) decides. [`Gerber1`](@ref) is the only branch whose denominator counts it, so the other two branches would carry the sum and never read it.

# Arguments

  - `pol`: The [`GerberIQKernel`](@ref) policy.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn)`.
  - `st`: Pair state from [`comovement_pair_state`](@ref).
  - `xi`, `xj`: Returns of assets `i` and `j` at observation `k`.
  - `axi`, `axj`: Their absolute values.
  - `T`: Number of observations.
  - `k`: Observation index.

# Returns

  - The accumulator, with `nn` raised by the [`gerber_IQ_delta`](@ref) contribution under [`Gerber1`](@ref), and unchanged otherwise.

# Related

  - [`comovement_step`](@ref)
  - [`iq_crossed`](@ref)
  - [`gerber_IQ_delta`](@ref)
  - [`GerberIQKernel`](@ref)
  - [`Gerber1`](@ref)
"""
@inline function iq_add_neutral(pol::GerberIQKernel{<:Gerber1}, acc, st, xi::Number,
                                xj::Number, axi::Number, axj::Number, T::Integer,
                                k::Integer)
    return (; acc...,
            nn = acc.nn +
                 gerber_IQ_delta(xi, xj, axi, axj, pol.decay, T, k, st.sci, st.scj,
                                 pol.kind))
end
@inline function iq_add_neutral(::GerberIQKernel, acc, args...)
    return acc
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accumulate the two diagonal projections of one observation into the Gerber IQ pair accumulator.

Only [`Gerber2`](@ref) reaches the acting method, because only its denominator reads them. The fall-through method returns the accumulator unchanged, so the other two markers pay for no weight they never divide by.

The projection of an observation in the `i` direction is the co-movement `(x_i, x_i)`, judged in **this pair's** units. Both coordinates then fall in the same magnitude class and the co-movement is concordant, so the projection names the diagonal weight of asset `i` at that observation. The `j` direction is the mirror. An asset contributes only when it left the noise zone, which is the same admission test the numerator applies.

The projection is what keeps the statistic inside `[-1, 1]`. An asset's magnitude class moves with its partner whenever `sc` is not pair-separable, and a denominator read from the assembled diagonal reads the class of the pair `(i, i)` instead. Cauchy-Schwarz then has nothing to stand on. Reading the class in the pair's own units restores it for every scaler.

# Arguments

  - `pol`: The [`GerberIQKernel`](@ref) policy.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`.
  - `st`: Pair state from [`comovement_pair_state`](@ref).
  - `xi`, `xj`: Returns of assets `i` and `j` at observation `k`.
  - `axi`, `axj`: Their absolute values.
  - `crossi`, `crossj`: Whether each asset left the noise zone, from [`iq_crossed`](@ref).
  - `T`: Number of observations.
  - `k`: Observation index.

# Returns

  - The accumulator, with `di` and `dj` raised by the projected [`gerber_IQ_delta`](@ref) contributions under [`Gerber2`](@ref), and unchanged otherwise.

# Related

  - [`comovement_step`](@ref)
  - [`comovement_finalise`](@ref)
  - [`iq_crossed`](@ref)
  - [`gerber_IQ_delta`](@ref)
  - [`GerberIQKernel`](@ref)
  - [`Gerber2`](@ref)
"""
@inline function iq_add_diagonal(pol::GerberIQKernel{<:Gerber2}, acc, st, xi::Number,
                                 xj::Number, axi::Number, axj::Number, crossi::Bool,
                                 crossj::Bool, T::Integer, k::Integer)
    di = if crossi
        acc.di +
        gerber_IQ_delta(xi, xi, axi, axi, pol.decay, T, k, st.sci, st.sci, pol.kind)
    else
        acc.di
    end
    dj = if crossj
        acc.dj +
        gerber_IQ_delta(xj, xj, axj, axj, pol.decay, T, k, st.scj, st.scj, pol.kind)
    else
        acc.dj
    end
    return (; acc..., di = di, dj = dj)
end
@inline function iq_add_diagonal(::GerberIQKernel, acc, args...)
    return acc
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Decide whether one asset left the noise zone at one observation.

An asset leaves the noise zone when its return reaches the pair's scaled threshold **and** is not exactly zero. The sign test is redundant for a positive threshold, because `ax >= c > 0` already implies that `x` is not zero. It binds only at `c = 0`, where the closed comparison `ax >= 0` holds for every return, including one that is exactly zero. A return of exactly zero never crosses, and this is that rule for the Gerber IQ family.

The rule is what keeps the diagonal of the statistic at one. The pair `(i, i)` either crosses on both axes or on neither, so it never reaches the neutral accumulator that [`Gerber1`](@ref) divides by. Without the sign test a zero return crosses on both axes but has no sign, so it fell through to that accumulator and pulled the diagonal below one.

# Arguments

  - `x`: Return of the asset at the observation.
  - `ax`: Its absolute value.
  - `c`: The asset's scaled noise threshold, from [`comovement_pair_state`](@ref).

# Returns

  - `crossed::Bool`: `true` when the asset left the noise zone.

# Related

  - [`comovement_step`](@ref)
  - [`comovement_pair_state`](@ref)
  - [`GerberIQKernel`](@ref)
"""
@inline function iq_crossed(x::Number, ax::Number, c::Number)
    return ax >= c && !iszero(x)
end
@inline function comovement_step(pol::GerberIQKernel, acc, st, xi::Number, xj::Number,
                                 T::Integer, k::Integer)
    axi = abs(xi)
    axj = abs(xj)
    crossi = iq_crossed(xi, axi, st.ci)
    crossj = iq_crossed(xj, axj, st.cj)
    if !crossi && !crossj
        return acc
    end
    acc = iq_add_diagonal(pol, acc, st, xi, xj, axi, axj, crossi, crossj, T, k)
    return if crossi && crossj && xi * xj > zero(xi)
        (; acc...,
         pos = acc.pos +
               gerber_IQ_delta(xi, xj, axi, axj, pol.decay, T, k, st.sci, st.scj, pol.kind))
    elseif crossi && crossj && xi * xj < zero(xi)
        (; acc...,
         neg = acc.neg +
               gerber_IQ_delta(xi, xj, axi, axj, pol.decay, T, k, st.sci, st.scj, pol.kind))
    else
        iq_add_neutral(pol, acc, st, xi, xj, axi, axj, T, k)
    end
end
@inline function comovement_finalise(pol::GerberIQKernel, acc, ::Type{T}) where {T}
    return comovement_ratio(pol.alg, acc.pos, acc.neg, acc.nn, T)
end
@inline function comovement_finalise(::GerberIQKernel{<:Gerber2}, acc, ::Type{T}) where {T}
    den = sqrt(acc.di * acc.dj)
    return !iszero(den) ? (acc.pos - acc.neg) / den : zero(T)
end
"""
    gerber_IQ(
        ce::GerberIQCovariance,
        X::MatNum,
        sd::ArrNum
    ) -> MatNum

Computes the Gerber IQ statistic matrix using noise compression template in `ce.kind` and numerator/denominator definition according to `ce.alg`.

# Mathematical definition

For each asset pair ``(i,j)`` accumulate weighted concordant and discordant counts:

```math
\\begin{align}
H_{ij}^{+} &= \\sum_{k=1}^{T} w_{ij,k} \\cdot d_k \\cdot \\mathbf{1}[\\text{concordant}]\\,, \\\\
H_{ij}^{-} &= \\sum_{k=1}^{T} w_{ij,k} \\cdot d_k \\cdot \\mathbf{1}[\\text{discordant}]\\,.
\\end{align}
```

Where:

  - ``H_{ij}^{+}``, ``H_{ij}^{-}``: Weighted concordant and discordant co-movement accumulators.
  - $(math_dict[:T])
  - ``w_{ij,k}``: Region weight from the IQ template for observation ``k``.
  - ``d_k = \\exp[-y \\max(0, T-k-e)]``: Temporal decay at observation ``k``.

GerberIQ correlation:

```math
\\begin{align}
\\rho_{ij} &= \\begin{cases}
(H_{ij}^{+} - H_{ij}^{-}) / (H_{ij}^{+} + H_{ij}^{-}) & \\text{Gerber0} \\\\
(H_{ij}^{+} - H_{ij}^{-}) / (H_{ij}^{+} + H_{ij}^{-} + H_{ij}^{0}) & \\text{Gerber1} \\\\
(H_{ij}^{+} - H_{ij}^{-}) / \\sqrt{D_{ij}\\,D_{ji}} & \\text{Gerber2}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\rho_{ij}``: GerberIQ correlation between assets ``i`` and ``j``.
  - ``H_{ij}^{+}``, ``H_{ij}^{-}``: Weighted concordant and discordant accumulators.
  - ``H_{ij}^{0}``: Weighted neutral (neither concordant nor discordant) accumulator (Gerber1 only).
  - ``D_{ij} = \\sum_{k} w_{ii,k}\\, d_{k}``, over the observations on which asset ``i`` left the noise zone: the projection of asset ``i`` onto the lead diagonal, **in the units of the pair** ``(i, j)``. The projected co-movement ``(x_{ki}, x_{ki})`` falls in one magnitude class on both axes, so ``w_{ii,k}`` is the diagonal weight of asset ``i`` at that observation. ``D_{ij}`` and ``D_{ji}`` differ, and both move with the pair whenever `sc` is not pair-separable.

The Gerber1 branch is the source's own statistic. Its numerator runs over the observations on which both assets left the noise zone, and its denominator over those on which at least one did. The Gerber0 and Gerber2 branches are the classic Gerber family's denominators, applied here to the weighted, discounted accumulators; the source's main text states neither. The source's internet appendix states a third form, whose denominator is the geometric mean of the two diagonal projections taken over the observations on which **both** assets crossed. This library does not ship that form. `Gerber2` projects the same way but keeps the classic denominator's observation set, which is the one that reduces to [`GerberCovariance`](@ref).

The Gerber statistic is the special case of this one that switches the squeezing and the decay off. With every weight set to one, ``\\gamma = 0``, the per-asset volatility scaling of [`AssetVolatilityGerberIQScaler`](@ref), and ``c`` equal to a Gerber threshold, all three branches reproduce [`GerberCovariance`](@ref) to the last bit. The reduction holds at ``c = 0`` as it does at every positive threshold, because [`iq_crossed`](@ref) gives this family the same rule as that one: a return of exactly zero never leaves the noise zone.

All three branches are bounded by ``|\\rho_{ij}| \\leq 1``. Gerber0 and Gerber1 are bounded by construction, because each divides by a sum of the same weights it subtracts. Gerber2 is bounded by the source's own condition on the template: every weight that joins two distinct magnitude classes is at most the geometric mean of the two diagonal weights of those classes. [`clamp_gerber_iq_n`](@ref) enforces that condition on every such weight, and the source proves it necessary and sufficient. Cauchy-Schwarz then bounds the ratio, because ``D_{ij}`` reads asset ``i``'s class in the same units the numerator reads it in, whatever `sc` does. The diagonal is exactly one, because the pair ``(i, i)`` makes the numerator and both projections the same sum.

# Algorithm

 1. Allocate the `N × N` output matrix `rho`.
 2. Resolve the decay estimator against `X` with [`regenerate_decay`](@ref), so its delay and rate are numbers before the loop starts.
 3. Build the [`GerberIQKernel`](@ref) policy from the resolved decay and the estimator's `alg`, `kind`, `sc`, `c` and the standard deviations `sd`.
 4. Fill `rho` with [`gerber_comovement!`](@ref), which walks every pair and every observation and reduces each pair's accumulators. That loop skeleton is shared with the Smyth-Broby family and lives in one place.
 5. Write one onto a zero diagonal entry with [`comovement_unit_diagonal!`](@ref). An asset that never leaves its noise zone reduces to a zero diagonal entry, and that entry is one by definition.
 6. Repair the matrix with [`posdef!`](@ref), because the statistic is not guaranteed to be positive semi-definite. The source records the same and repairs by the nearest correlation matrix.

Step 4 is where the three [`GerberCovarianceAlgorithm`](@ref) branches differ. [`comovement_ratio`](@ref) owns the [`Gerber0`](@ref) and [`Gerber1`](@ref) denominators, and [`comovement_finalise`](@ref) owns the [`Gerber2`](@ref) one. This family does not call [`standardise_comovement!`](@ref), which normalises after assembly and cannot read the pair's units.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:stdarr])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQKernel`](@ref)
  - [`gerber_comovement!`](@ref)
  - [`comovement_unit_diagonal!`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`gerber_IQ_delta`](@ref)
  - [`regenerate_decay`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function gerber_IQ(ce::GerberIQCovariance, X::MatNum, sd::ArrNum)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    decay = regenerate_decay(ce.decay, X)
    pol = GerberIQKernel(ce.alg, ce.kind, decay, ce.sc, ce.c, sd)
    gerber_comovement!(rho, ce.ex, X, pol)
    comovement_unit_diagonal!(rho)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    Statistics.cor(
        ce::GerberIQCovariance,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the Gerber IQ correlation matrix.

This method computes the Gerber IQ correlation matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Gerber IQ correlation is then computed via [`gerber_IQ`](@ref).

The standard deviations serve two purposes at once. They scale the thresholds through [`gerber_iq_scaling`](@ref), and in [`cov`](@ref) they rescale the correlation into a covariance.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute the per-asset standard deviations with the estimator's `ve`.
 3. Raise every standard deviation to at least `eps(eltype(sd))`, so a constant asset cannot divide by zero.
 4. Centre the returns with the estimator's `me` through [`demean_returns`](@ref).
 5. Return the matrix that [`gerber_IQ`](@ref) builds from the centred returns and those standard deviations.

# Arguments

  - `ce`: Gerber IQ covariance estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::MatNum`: The Gerber IQ correlation matrix. Its diagonal is one for every asset.

!!! note

    An asset that never leaves its own noise zone gets a **zero row**, because no observation votes for any pair it belongs to. Its diagonal entry is one, which [`comovement_unit_diagonal!`](@ref) writes, so the matrix stays a formal correlation matrix and the asset reads as uncorrelated with every other one. That is what the sample says about it. Lower `c` when a short window meets a quiet asset, and the asset votes again.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function Statistics.cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    return gerber_IQ(ce, X, sd)
end
"""
    Statistics.cov(
        ce::GerberIQCovariance,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the Gerber IQ covariance matrix.

This method computes the Gerber IQ covariance matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Gerber IQ correlation is then computed via [`gerber_IQ`](@ref).

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\boldsymbol{\\rho} \\odot \\left(\\boldsymbol{\\sigma} \\boldsymbol{\\sigma}^{\\intercal}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``\\boldsymbol{\\rho}``: Gerber IQ correlation matrix.
  - ``\\boldsymbol{\\sigma}``: Vector of asset standard deviations.
  - ``\\odot``: Element-wise multiplication.

The covariance is the correlation of [`cor`](@ref) rescaled by the same standard deviations that scaled its thresholds, so its diagonal is exactly ``\\boldsymbol{\\sigma}^2``.

# Algorithm

 1. Run the five steps of [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref), giving the correlation matrix and the standard deviations.
 2. Rescale that matrix in place with `StatsBase.cor2cov!` and those standard deviations, and return it.

# Arguments

  - `ce`: Gerber IQ covariance estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `sigma::MatNum`: The Gerber IQ covariance matrix. Its diagonal is the variance of each asset, because `cor2cov!` scales a unit correlation diagonal by ``\\boldsymbol{\\sigma}^2``.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing])
"""
function Statistics.cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sd .= max.(sd, eps(eltype(sd)))
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    sigma = gerber_IQ(ce, X, sd)
    return StatsBase.cor2cov!(sigma, sd)
end
