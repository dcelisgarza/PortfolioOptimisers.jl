"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`GerberIQCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Without this method, that state prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:GerberIQCovariance, true)` to render it.

# Arguments

  - `::GerberIQCovariance`: Gerber IQ covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ve, :me, :pdm, :c, :decay, :oc, :sc, :kind, :alg, :ex)`.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::GerberIQCovariance) = (:ve, :me, :pdm, :c, :decay, :oc, :sc, :kind, :alg, :ex)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the contribution of one co-movement to the Gerber IQ statistic.

The contribution is the product of the two factors of the squeezing statistic of the source. The first factor is the weight that the squeezing template gives the co-movement. The second factor is the discount for the age of the observation. [`comovement_step`](@ref) adds the contribution to `pos`, `neg` or `nn`, the three sums of the pair.

# Mathematical definition

```math
\\begin{align}
\\Delta_{t,\\,i,\\,j} &= \\eta_{t,\\,i,\\,j} \\, v_{t}\\,.
\\end{align}
```

Where:

  - ``\\Delta_{t,\\,i,\\,j}``: Contribution of the co-movement of assets ``i`` and ``j`` at observation ``t``.
  - ``\\eta_{t,\\,i,\\,j}``: Squeezing weight of that co-movement, from the squeezing template.
  - ``v_{t}``: Temporal discount of observation ``t``, from the decay estimator.

# Algorithm

 1. Compute the squeezing weight `w` with [`gerber_iq_weight`](@ref) from the two returns, the two scaling factors of the pair and the squeezing template.
 2. Compute the temporal discount `p` with the `decay` functor at `T` and `k`.
 3. Return `w * p`.

# Arguments

  - `xi`: Return of asset `i`.
  - `xj`: Return of asset `j`.
  - `axi`: Absolute return of asset `i`.
  - `axj`: Absolute return of asset `j`.
  - `decay`: The decay estimator, after [`regenerate_decay`](@ref) resolved it.
  - `T`: The number of observations.
  - `k`: The index of the observation.
  - `sci`: Scaling factor of asset `i`.
  - `scj`: Scaling factor of asset `j`.
  - `kind`: The squeezing template.

# Returns

  - `rho::Number`: The weighted and discounted contribution of one co-movement. It is not a correlation.

# Related

  - [`gerber_iq_weight`](@ref)
  - [`comovement_step`](@ref)
  - [`gerber_IQ`](@ref)
  - [`GerberIQDecayEstimator`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`GerberIQCovariance`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.5, Equations 4 and 5.
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

Co-movement policy of the Gerber IQ family for [`gerber_comovement!`](@ref).

The policy compares each return with the noise threshold of its asset, scaled by [`gerber_iq_scaling`](@ref). It sorts a co-movement by the sign of the product of the two returns. [`gerber_IQ_delta`](@ref) gives the co-movement its squeezing weight and its temporal discount. The `alg` marker selects the denominator. [`comovement_ratio`](@ref) computes it for [`Gerber0`](@ref) and [`Gerber1`](@ref), and [`comovement_finalise`](@ref) computes it for [`Gerber2`](@ref).

# Fields

  - `alg`: Gerber marker that selects the denominator.
  - `kind`: Squeezing template.
  - `decay`: Temporal decay estimator, after [`regenerate_decay`](@ref) resolved it.
  - `sc`: Threshold scaler.
  - `c`: Noise threshold, before the policy scales it.
  - `sd`: Vector of the standard deviations of the assets.

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

Adds a neutral observation to the Gerber IQ pair accumulator.

A neutral observation is one on which exactly one of the two assets leaves the noise zone. [`iq_crossed`](@ref) decides which asset leaves it. Only the denominator of [`Gerber1`](@ref) contains the neutral sum. So the [`Gerber1`](@ref) method adds the [`gerber_IQ_delta`](@ref) contribution to `nn`, and the fall-through method returns the accumulator unchanged.

# Arguments

  - `pol`: The [`GerberIQKernel`](@ref) policy.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This method changes `nn` alone.
  - `st`: Pair state from [`comovement_pair_state`](@ref).
  - `xi`, `xj`: Returns of assets `i` and `j` at observation `k`.
  - `axi`, `axj`: Absolute returns of assets `i` and `j`.
  - `T`: Number of observations.
  - `k`: Index of the observation.

# Returns

  - The accumulator. Under [`Gerber1`](@ref), `nn` increases by the [`gerber_IQ_delta`](@ref) contribution. Under the other two markers, the accumulator does not change.

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

Adds the two diagonal projections of one observation to the Gerber IQ pair accumulator.

Only the denominator of [`Gerber2`](@ref) contains the projections, so only [`Gerber2`](@ref) reaches the method that adds them. The fall-through method returns the accumulator unchanged. The other two markers thus compute no weight that they do not use.

The projection of an observation in the `i` direction is the co-movement `(x_i, x_i)`, in the units of this pair. Both coordinates fall in the same magnitude class, and the co-movement is concordant. So the projection gives the diagonal weight of asset `i` at that observation. The `j` direction is the mirror. An asset contributes only when it leaves the noise zone, which is the test that the numerator applies too.

The projection keeps the statistic inside `[-1, 1]`. When `sc` is not pair-separable, the magnitude class of an asset changes with its partner. A denominator from the assembled diagonal uses the class of the pair `(i, i)` instead, and the Cauchy-Schwarz bound then does not hold. A projection in the units of the pair makes the bound hold for every scaler.

# Arguments

  - `pol`: The [`GerberIQKernel`](@ref) policy.
  - `acc`: Pair accumulator `(pos, neg, nn, cpos, cneg, cnn, di, dj)`. This method changes `di` and `dj` alone.
  - `st`: Pair state from [`comovement_pair_state`](@ref).
  - `xi`, `xj`: Returns of assets `i` and `j` at observation `k`.
  - `axi`, `axj`: Absolute returns of assets `i` and `j`.
  - `crossi`, `crossj`: `true` for an asset that leaves the noise zone, from [`iq_crossed`](@ref).
  - `T`: Number of observations.
  - `k`: Index of the observation.

# Returns

  - The accumulator. Under [`Gerber2`](@ref), `di` and `dj` increase by the projected [`gerber_IQ_delta`](@ref) contributions. Under the other two markers, the accumulator does not change.

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

Decides whether one asset leaves the noise zone at one observation.

An asset leaves the noise zone when its absolute return is at least the scaled threshold of the pair, and the return is not zero. For a positive threshold, the second test changes nothing, because `ax >= c > 0` makes `x` nonzero. The second test matters only at `c = 0`, where `ax >= 0` holds for every return, a zero return too. So a return of exactly zero never crosses. The Gerber statistic has the same rule.

The rule keeps the diagonal of the statistic at one. The pair `(i, i)` crosses on both axes or on neither, so it never adds to the neutral sum that [`Gerber1`](@ref) divides by. Without the second test, a zero return crosses on both axes but has no sign. It then goes to the neutral sum, and the diagonal entry becomes less than one.

# Arguments

  - `x`: Return of the asset at the observation.
  - `ax`: Absolute return of the asset.
  - `c`: The scaled noise threshold of the asset, from [`comovement_pair_state`](@ref).

# Returns

  - `crossed::Bool`: `true` when the asset leaves the noise zone.

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

Computes the Gerber IQ correlation matrix with the squeezing template in `ce.kind` and the denominator that `ce.alg` selects.

# Mathematical definition

For each pair of assets ``(i,j)``, the statistic sums the weighted and discounted co-movements of each sign.

```math
\\begin{align}
H_{ij}^{+} &= \\sum_{k=1}^{T} w_{ij,k} \\cdot d_k \\cdot \\mathbf{1}[\\text{concordant}]\\,, \\\\
H_{ij}^{-} &= \\sum_{k=1}^{T} w_{ij,k} \\cdot d_k \\cdot \\mathbf{1}[\\text{discordant}]\\,.
\\end{align}
```

Where:

  - ``H_{ij}^{+}``, ``H_{ij}^{-}``: Weighted concordant and discordant sums.
  - $(math_dict[:T])
  - ``w_{ij,k}``: Squeezing weight of the co-movement at observation ``k``, from the squeezing template.
  - ``d_k = \\exp[-y \\max(0, T-k-e)]``: Temporal decay at observation ``k``.

The Gerber IQ correlation is:

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

  - ``\\rho_{ij}``: Gerber IQ correlation of assets ``i`` and ``j``.
  - ``H_{ij}^{+}``, ``H_{ij}^{-}``: Weighted concordant and discordant sums.
  - ``H_{ij}^{0}``: Weighted neutral sum, over the observations on which exactly one asset leaves the noise zone. Only the `Gerber1` denominator contains it.
  - ``D_{ij} = \\sum_{k} w_{ii,k}\\, d_{k}``: Projection of asset ``i`` onto the lead diagonal, in the units of the pair ``(i, j)``. The sum runs over the observations on which asset ``i`` leaves the noise zone. The projected co-movement ``(x_{ki}, x_{ki})`` falls in one magnitude class on both axes, so ``w_{ii,k}`` is the diagonal weight of asset ``i`` at that observation. ``D_{ij}`` and ``D_{ji}`` differ. When `sc` is not pair-separable, both change with the pair.

The `Gerber1` statistic is the statistic of the source. Its numerator sums over the observations on which both assets leave the noise zone. Its denominator sums over the observations on which at least one asset leaves it. The `Gerber0` and `Gerber2` statistics apply the denominators of the classic Gerber family to the weighted and discounted sums. The main text of the source states neither of them.

The Internet Appendix of the source states a third form. Its denominator is the geometric mean of the two diagonal projections, summed over the observations on which both assets cross. The library does not ship that form. `Gerber2` makes the same projections, but it keeps the observation set of the classic denominator. With that set, the statistic reduces to [`GerberCovariance`](@ref).

The Gerber statistic is the special case of the Gerber IQ statistic with no squeezing and no decay. Set every weight to one, ``\\gamma = 0``, the scaler [`AssetVolatilityGerberIQScaler`](@ref) and ``c`` equal to a Gerber threshold. All three markers then give the result of [`GerberCovariance`](@ref) to the last bit. The reduction holds at ``c = 0`` too, because [`iq_crossed`](@ref) gives this family the rule of the Gerber statistic. Under that rule, a return of exactly zero never leaves the noise zone.

All three markers keep ``|\\rho_{ij}| \\leq 1``. `Gerber0` and `Gerber1` keep it by construction, because each divides by a sum of the weights that it subtracts. `Gerber2` keeps it under a condition of the source on the squeezing template. Each weight that joins two distinct magnitude classes must be at most the geometric mean of the two diagonal weights of those classes. [`clamp_gerber_iq_n`](@ref) enforces that condition on every such weight.

The Internet Appendix of [gerber2025squeezing](@cite) proves that the condition is necessary and sufficient for its own diagonal projection, in Equations A1 to A9. For the larger observation set of the library, the condition is sufficient. The Cauchy-Schwarz inequality then bounds the ratio, because ``D_{ij}`` measures the class of asset ``i`` in the units of the numerator, for every `sc`. The diagonal is exactly one, because for the pair ``(i, i)`` the numerator and both projections are the same sum.

# Algorithm

 1. Allocate the `N × N` output matrix `rho`.
 2. Resolve the decay estimator against `X` with [`regenerate_decay`](@ref), so that its delay and its rate are numbers before the loop starts.
 3. Make the [`GerberIQKernel`](@ref) policy from the resolved decay, from the `alg`, `kind`, `sc` and `c` of the estimator, and from the standard deviations `sd`.
 4. Fill `rho` with [`gerber_comovement!`](@ref). It loops over every pair and every observation, and it reduces the sums of each pair. The Smyth-Broby family uses the same loop.
 5. Write one on each zero diagonal entry with [`comovement_unit_diagonal!`](@ref). An asset that never leaves its noise zone gets a zero diagonal entry, but the diagonal of a correlation matrix is one by definition.
 6. Repair the matrix with [`posdef!`](@ref), because the statistic is not always positive semidefinite. The source states the same, and it repairs the matrix with the nearest correlation matrix.

The three [`GerberCovarianceAlgorithm`](@ref) markers differ at step 4 alone. [`comovement_ratio`](@ref) computes the [`Gerber0`](@ref) and [`Gerber1`](@ref) denominators, and [`comovement_finalise`](@ref) computes the [`Gerber2`](@ref) denominator. This family does not call [`standardise_comovement!`](@ref). That function normalises the assembled matrix, so it cannot use the units of each pair.

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

  - $(ref_dict[:gerber2025squeezing]) Section 3.5, Equations 1 to 5, and the Internet Appendix, Equation A1.
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

Compute the Gerber IQ correlation matrix with the squeezing template in `ce.kind` and the marker in `ce.alg`.

The method computes the standard deviations with `ce.ve`, and it centres the returns with `ce.me`. [`gerber_IQ`](@ref) then computes the correlation.

The standard deviations have two uses. They scale the thresholds through [`gerber_iq_scaling`](@ref). In [`cov`](@ref), they also scale the correlation into a covariance.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute the standard deviation of each asset with `ce.ve`.
 3. Raise each standard deviation to at least `eps(eltype(sd))`. A constant asset then has a positive scale, and every scaled threshold is positive when `c` is positive.
 4. Centre the returns with `ce.me` through [`demean_returns`](@ref).
 5. Return the matrix that [`gerber_IQ`](@ref) computes from the centred returns and the standard deviations.

# Arguments

  - `ce`: Gerber IQ covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for the standard deviation estimator and for [`demean_returns`](@ref).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::MatNum`: The Gerber IQ correlation matrix. Its diagonal is one for every asset.

!!! note

    An asset that never leaves its noise zone gets a zero row, because no observation adds to a pair that contains it. [`comovement_unit_diagonal!`](@ref) writes one on its diagonal entry. So the matrix stays a correlation matrix, and the asset is uncorrelated with every other asset. The sample gives no other information about the asset. To make the asset count, lower `c`. A short window and a quiet asset make this case more frequent.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.5.
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

Compute the Gerber IQ covariance matrix with the squeezing template in `ce.kind` and the marker in `ce.alg`.

The method computes the standard deviations with `ce.ve`, and it centres the returns with `ce.me`. [`gerber_IQ`](@ref) then computes the correlation, and the method scales it by the standard deviations.

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

The covariance is the correlation of [`cor`](@ref), scaled by the standard deviations that also scale its thresholds. So its diagonal is exactly ``\\boldsymbol{\\sigma}^2``.

# Algorithm

 1. Do the five steps of [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref) to get the correlation matrix and the standard deviations.
 2. Scale that matrix in place with `StatsBase.cor2cov!` and the standard deviations, and return it.

# Arguments

  - `ce`: Gerber IQ covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments for the standard deviation estimator and for [`demean_returns`](@ref).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `sigma::MatNum`: The Gerber IQ covariance matrix. Its diagonal is the variance of each asset, because `cor2cov!` scales the unit diagonal of the correlation by ``\\boldsymbol{\\sigma}^2``.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`GerberIQCovarianceAlgorithm`](@ref)
  - [`demean_returns`](@ref)
  - [`gerber_IQ`](@ref)
  - [`cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:gerber2025squeezing]) Section 3.5.
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
function effective_observation_count(ce::GerberIQCovariance, X::MatNum; kwargs...)
    return gerber_iq_observation_count(ce.oc, ce, X; kwargs...)
end
"""
    gerber_iq_observation_count(oc::Nothing, ce::GerberIQCovariance, X::MatNum;
                                kwargs...) -> Integer
    gerber_iq_observation_count(oc::AbstractObservationCountAlgorithm,
                                ce::GerberIQCovariance, X::MatNum; kwargs...) -> Real

Return the effective number of observations of a [`GerberIQCovariance`](@ref) over `X`, under the rule `oc` of its field.

[`effective_observation_count`](@ref) calls this function, and the denoising step of a [`PortfolioOptimisersCovariance`](@ref) reads the count. The decay gives the observation of index `k` the weight `decay(T, k)`. A rule counts that vector of weights.

# Algorithm

 1. With `oc = nothing`, return `size(X, 1)`. The function then reads no data.
 2. Otherwise, centre `X` with `ce.me` and `kwargs` through [`demean_returns`](@ref), as the batch verbs do. A `mean` of the caller is then the centre of the count and of the fit.
 3. Resolve the decay against the centred returns with [`regenerate_decay`](@ref), as [`gerber_IQ`](@ref) does. A rule in the delay or the rate then reads the same matrix as in the fit.
 4. Return [`weight_observation_count`](@ref) of `oc` over the weights `decay(T, k)`, ``k = 1, \\ldots, T``.

# Arguments

  - `oc`: The rule of the count, from `ce.oc`.
  - `ce`: Gerber IQ covariance estimator.
  - `X`: Returns matrix, `observations × assets`.
  - `kwargs...`: The keywords of the fit, for [`demean_returns`](@ref).

# Returns

  - `T_e::Real`: The effective number of observations of the matrix.

# Related

  - [`GerberIQCovariance`](@ref)
  - [`effective_observation_count`](@ref)
  - [`AbstractObservationCountAlgorithm`](@ref)
  - [`regenerate_decay`](@ref)
"""
function gerber_iq_observation_count(::Nothing, ::GerberIQCovariance, X::MatNum; kwargs...)
    return size(X, 1)
end
function gerber_iq_observation_count(oc::AbstractObservationCountAlgorithm,
                                     ce::GerberIQCovariance, X::MatNum; kwargs...)
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    T = size(X, 1)
    decay = regenerate_decay(ce.decay, X)
    return weight_observation_count(oc, decay.(T, 1:T))
end
