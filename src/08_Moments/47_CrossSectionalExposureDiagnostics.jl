"""
    exposure_weights(B::Arr3Num, w::Nothing)
    exposure_weights(B::Arr3Num, w::MatNum)

Return the cross-sectional weight history an exposure diagnostic reads, checking it against the exposure history.

Every diagnostic of this file weights the assets of one observation, and an absent weight history means equal weights. The absent case is resolved once, into a history of ones, so each kernel reads one matrix and no kernel carries a branch. A weight of zero excludes the asset from the observation, which is the same answer the history of ones gives when no asset is excluded.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.

# Validation

  - `!isempty(B)`.
  - `size(w) == (size(B, 1), size(B, 2))`, when `w` is present.
  - No finite entry of `w` is negative.

# Returns

  - `u::MatNum`: Weight history `observations × assets`. It is a history of ones when `w` is `nothing`, and `w` itself otherwise.

# Related

  - [`exposure_correlation`](@ref)
  - [`exposure_stability`](@ref)
  - [`exposure_dispersion`](@ref)
  - [`exposure_coverage`](@ref)
"""
function exposure_weights(B::Arr3Num, ::Nothing)
    @argcheck(!isempty(B), IsEmptyError("B cannot be empty"))
    return ones(real(eltype(B)), size(B, 1), size(B, 2))
end
function exposure_weights(B::Arr3Num, w::MatNum)
    @argcheck(!isempty(B), IsEmptyError("B cannot be empty"))
    @argcheck(size(w, 1) == size(B, 1) && size(w, 2) == size(B, 2),
              DimensionMismatch("w ($(size(w, 1))×$(size(w, 2))) must match B ($(size(B, 1))×$(size(B, 2)) on its first two axes)"))
    @argcheck(all(x -> !isfinite(x) || x >= zero(x), w),
              DomainError(w, "w cannot carry a negative weight"))
    return w
end
"""
    cs_weighted_correlation(a::AbstractVector, b::AbstractVector, u::AbstractVector;
                            min_count::Integer = 3, eps::Real = 1e-12)

Return the weighted correlation of two cross-sections of one observation.

It is the kernel of every diagnostic of this file that correlates two cross-sections: the information coefficient of an exposure against the forward return, and the stability of an exposure against its own later self. An asset enters when both of its values are finite and its weight is finite and positive, so a missing value costs the pair alone and not the observation.

# Mathematical definition

```math
\\rho = \\frac{\\sum_{i} u_{i} (a_{i} - \\bar{a}) (b_{i} - \\bar{b})}{\\sqrt{\\sum_{i} u_{i} (a_{i} - \\bar{a})^{2}} \\sqrt{\\sum_{i} u_{i} (b_{i} - \\bar{b})^{2}}}
```

Where:

  - ``a_{i}``, ``b_{i}``: The two values of asset ``i``.
  - ``u_{i}``: Weight of asset ``i``.
  - ``\\bar{a} = \\sum_{i} u_{i} a_{i} / \\sum_{i} u_{i}``: Weighted mean of the first cross-section, and likewise ``\\bar{b}``.

# Algorithm

 1. Count the assets that enter, and accumulate the weight sum and the two weighted means over them.
 2. Answer `NaN` when fewer than `min_count` assets enter, or when the weight sum is zero.
 3. Accumulate the weighted central moments over the same assets.
 4. Answer `NaN` when the denominator is at or under `eps`, which is a cross-section that is constant in one of the two values.

# Arguments

  - `a`: First cross-section, one entry per asset.
  - `b`: Second cross-section, one entry per asset.
  - `u`: Weight of each asset. An entry that is not finite, or not positive, excludes the asset.
  - `min_count`: Least number of assets an answer needs.
  - `eps`: Denominator at or under which the answer is `NaN`.

# Returns

  - `rho::Real`: The correlation, or `NaN`.

# Examples

```jldoctest
julia> PortfolioOptimisers.cs_weighted_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], ones(3))
1.0
```

# Related

  - [`cs_spearman_correlation`](@ref)
  - [`exposure_ic`](@ref)
  - [`exposure_stability`](@ref)
"""
function cs_weighted_correlation(a::AbstractVector, b::AbstractVector, u::AbstractVector;
                                 min_count::Integer = 3, eps::Real = 1e-12)
    Tf = promote_type(real(eltype(a)), real(eltype(b)), real(eltype(u)))
    N = length(a)
    n = 0
    ws = zero(Tf)
    sa = zero(Tf)
    sb = zero(Tf)
    for i in 1:N
        if cs_correlation_enters(a, b, u, i)
            n += 1
            wi = Tf(u[i])
            ws += wi
            sa += wi * Tf(a[i])
            sb += wi * Tf(b[i])
        end
    end
    if n < min_count || !(ws > zero(Tf))
        return Tf(NaN)
    end
    ma = sa / ws
    mb = sb / ws
    cab = zero(Tf)
    caa = zero(Tf)
    cbb = zero(Tf)
    for i in 1:N
        if cs_correlation_enters(a, b, u, i)
            wi = Tf(u[i])
            da = Tf(a[i]) - ma
            db = Tf(b[i]) - mb
            cab += wi * da * db
            caa += wi * da * da
            cbb += wi * db * db
        end
    end
    den = sqrt(caa * cbb)
    return den > eps ? cab / den : Tf(NaN)
end
"""
    cs_correlation_enters(a::AbstractVector, b::AbstractVector, u::AbstractVector,
                          i::Integer)

Return whether one asset enters the correlation of two cross-sections.

An asset enters when both of its values are finite and its weight is finite and positive. The two passes of [`cs_weighted_correlation`](@ref) read the same rule, so it is stated once.

# Arguments

  - `a`: First cross-section, one entry per asset.
  - `b`: Second cross-section, one entry per asset.
  - `u`: Weight of each asset.
  - `i`: Position of the asset.

# Returns

  - `enters::Bool`: `true` when the asset enters.

# Related

  - [`cs_weighted_correlation`](@ref)
"""
function cs_correlation_enters(a::AbstractVector, b::AbstractVector, u::AbstractVector,
                               i::Integer)
    return isfinite(a[i]) && isfinite(b[i]) && isfinite(u[i]) && u[i] > 0
end
"""
    cs_ordinal_ranks(key::AbstractVector, valid::AbstractVector{Bool})

Return the ordinal rank of every entry of a cross-section, and `NaN` outside a mask.

The rank is the position the entry takes in the sorted order, so two equal values take two different ranks, in the order of the asset axis. The masked entries are sorted to the end by their key and then written as `NaN`, so they take no rank and shift none.

# Arguments

  - `key`: Sort key of each asset. The key of a masked asset must be `Inf`, so the mask sorts to the end.
  - `valid`: Mask, one entry per asset. A `false` entry gets `NaN`.

# Returns

  - `r::VecNum`: Rank of each asset, or `NaN` outside the mask.

# Related

  - [`cs_spearman_correlation`](@ref)
"""
function cs_ordinal_ranks(key::AbstractVector, valid::AbstractVector{Bool})
    Tf = real(eltype(key))
    N = length(key)
    p = sortperm(key)
    r = Vector{Tf}(undef, N)
    for pos in 1:N
        i = p[pos]
        r[i] = valid[i] ? Tf(pos) : Tf(NaN)
    end
    return r
end
"""
    cs_spearman_correlation(a::AbstractVector, b::AbstractVector;
                            min_count::Integer = 3, eps::Real = 1e-12)

Return the rank correlation of two cross-sections of one observation.

It is the unweighted correlation of the ordinal ranks of the two cross-sections, taken over the assets at which both values are finite. The rank measures the order of the assets and not their level, so one extreme value moves the answer no more than one ordinary value does.

# Algorithm

 1. Mark the assets at which both values are finite.
 2. Rank each cross-section over those assets with [`cs_ordinal_ranks`](@ref).
 3. Correlate the two rank vectors with [`cs_weighted_correlation`](@ref) under equal weights.

# Arguments

  - `a`: First cross-section, one entry per asset.
  - `b`: Second cross-section, one entry per asset.
  - `min_count`: Least number of assets an answer needs.
  - `eps`: Denominator at or under which the answer is `NaN`.

# Returns

  - `rho::Real`: The rank correlation, or `NaN`.

# Related

  - [`cs_ordinal_ranks`](@ref)
  - [`cs_weighted_correlation`](@ref)
  - [`exposure_ic`](@ref)
"""
function cs_spearman_correlation(a::AbstractVector, b::AbstractVector;
                                 min_count::Integer = 3, eps::Real = 1e-12)
    Tf = promote_type(real(eltype(a)), real(eltype(b)))
    N = length(a)
    ka = Vector{Tf}(undef, N)
    kb = Vector{Tf}(undef, N)
    valid = Vector{Bool}(undef, N)
    for i in 1:N
        v = isfinite(a[i]) && isfinite(b[i])
        valid[i] = v
        ka[i] = v ? Tf(a[i]) : Tf(Inf)
        kb[i] = v ? Tf(b[i]) : Tf(Inf)
    end
    ra = cs_ordinal_ranks(ka, valid)
    rb = cs_ordinal_ranks(kb, valid)
    return cs_weighted_correlation(ra, rb, ones(Tf, N); min_count = min_count, eps = eps)
end
"""
    exposure_forward_mean_return(R::MatNum, horizon::Integer)

Return the forward mean asset return of every observation but the last `horizon` of them.

The information coefficient scores an exposure against what the asset earned after the exposure was known, and this verb builds that target. An observation whose forward window carries no finite return gets `NaN`, and a window that is part finite is averaged over the finite part alone.

# Mathematical definition

```math
y_{ti} = \\frac{1}{|\\mathcal{H}_{ti}|} \\sum_{h \\in \\mathcal{H}_{ti}} r_{t+h,i}
```

Where:

  - ``r_{ti}``: Return of asset ``i`` at observation ``t``.
  - ``\\mathcal{H}_{ti}``: The offsets ``1 \\le h \\le H`` at which ``r_{t+h,i}`` is finite.
  - ``H``: Forward window, in observations.

# Arguments

  - `R`: Asset return history `observations × assets`.
  - `horizon`: Forward window, in observations.

# Validation

  - `size(R, 1) > horizon`.

# Returns

  - `y::MatNum`: Forward mean return `(observations - horizon) × assets`. Row `t` averages the returns of observations `t + 1` to `t + horizon`.

# Related

  - [`exposure_ic`](@ref)
"""
function exposure_forward_mean_return(R::MatNum, horizon::Integer)
    T, N = size(R)
    @argcheck(T > horizon,
              DimensionMismatch("R ($T observations) must carry more observations than horizon ($horizon)"))
    Tf = real(eltype(R))
    P = T - horizon
    y = Matrix{Tf}(undef, P, N)
    for i in 1:N, t in 1:P
        s = zero(Tf)
        c = 0
        for h in 1:horizon
            v = R[t + h, i]
            if isfinite(v)
                s += Tf(v)
                c += 1
            end
        end
        y[t, i] = c > 0 ? s / c : Tf(NaN)
    end
    return y
end
"""
    exposure_correlation(B::Arr3Num, w::Option{<:MatNum} = nothing) -> Matrix{<:Real}
    exposure_correlation(csfm::CrossSectionalFactorModel;
                         weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric()) -> Matrix{<:Real}

Return the time-averaged correlation between every pair of factor exposures.

Two exposures that correlate across the assets carry the same reading of the cross-section, so the pair is redundant and the design of the fit is collinear. This answer is the cross-sectional counterpart of the variance inflation factor, which reads the same collinearity off the design of one observation.

Each pair is correlated over the assets at which both exposures are finite, one observation at a time, and the answers are then averaged over the observations at which the pair is defined. A pair one of whose exposures is constant across the assets has no correlation, and it reads `0` by convention. A pair that never shares three finite assets reads `NaN`. The diagonal is `1`.

The level-2 method answers on the **raw** factor axis, because the exposure history is the reading of the panel and not the design of the fit, and it reads the unlagged history.

# Mathematical definition

```math
\\mathbf{C}_{kl} = \\frac{1}{|\\mathcal{T}_{kl}|} \\sum_{t \\in \\mathcal{T}_{kl}} \\rho \\left( \\mathbf{B}_{t \\cdot k}, \\mathbf{B}_{t \\cdot l}, \\boldsymbol{u}_{t} \\right)
```

Where:

  - ``\\mathbf{B}_{t \\cdot k}``: Cross-section of factor ``k`` at observation ``t``.
  - ``\\boldsymbol{u}_{t}``: Cross-sectional weights of observation ``t``.
  - ``\\rho``: The weighted correlation of two cross-sections.
  - ``\\mathcal{T}_{kl}``: The observations at which the pair is defined.

# Algorithm

 1. Resolve the weight history with [`exposure_weights`](@ref).
 2. For each observation and each pair, accumulate the weighted sums over the assets at which both exposures are finite.
 3. Mark the pair as degenerate when either weighted variance falls to the tolerance of its own weighted square sum, and as insufficient when fewer than three assets are shared.
 4. Average the pairs that are neither over the observations. A pair that is degenerate at every observation it covers reads `0`, and one that is never sufficient reads `NaN`.
 5. Write `1` on the diagonal.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `size(w) == (size(B, 1), size(B, 2))`, when `w` is present.
  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.
  - The field `weighting` names is not `nothing`, else an `IsNothingError` naming it is raised.

# Returns

  - `C::Matrix{<:Real}`: `factors × factors`, symmetric, with a diagonal of `1`.

# Related

  - [`exposure_weights`](@ref)
  - [`exposure_stability`](@ref)
  - [`exposure_vif`](@ref)
  - [`cs_diagnostic_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_correlation(B::Arr3Num, w::Option{<:MatNum} = nothing)
    u = exposure_weights(B, w)
    K = size(B, 3)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    Cm = Matrix{Tf}(undef, K, K)
    for l in 1:K, k in 1:l
        v = k == l ? one(Tf) : Tf(exposure_pair_correlation(B, u, k, l))
        Cm[k, l] = v
        Cm[l, k] = v
    end
    return Cm
end
function exposure_correlation(csfm::CrossSectionalFactorModel;
                              weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric())
    return exposure_correlation(cs_diagnostic_exposures(csfm),
                                cs_diagnostic_weights(weighting, csfm))
end
"""
    exposure_pair_correlation(B::Arr3Num, u::MatNum, k::Integer, l::Integer)

Return the time-averaged correlation of one pair of factor exposures.

The pair is correlated at each observation, and the answers are averaged over the observations at which the pair is defined. A pair that is degenerate at every observation it covers reads `0` by convention, and one that never reaches three common assets reads `NaN`.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `u`: Resolved weight history `observations × assets`.
  - `k`, `l`: Positions of the two factors.

# Returns

  - `rho::Real`: The time-averaged correlation, `0` by convention, or `NaN`.

# Related

  - [`exposure_correlation`](@ref)
  - [`exposure_pair_observation`](@ref)
"""
function exposure_pair_correlation(B::Arr3Num, u::MatNum, k::Integer, l::Integer)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    T = size(B, 1)
    acc = zero(Tf)
    cnt = 0
    degen = false
    for t in 1:T
        r, dg = exposure_pair_observation(B, u, t, k, l)
        degen |= dg
        if isfinite(r)
            acc += r
            cnt += 1
        end
    end
    if cnt > 0
        return acc / cnt
    end
    return degen ? zero(Tf) : Tf(NaN)
end
"""
    exposure_pair_observation(B::Arr3Num, u::MatNum, t::Integer, k::Integer, l::Integer)

Return the correlation of one pair of factor exposures at one observation, and whether the pair is degenerate there.

The pair is correlated over the assets at which both exposures are finite. It has no answer at an observation that shares fewer than three such assets, and none at one where either weighted variance falls to the tolerance of its own weighted square sum. The second case is degenerate, which the caller reads as the `0` convention, and the first is not.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `u`: Resolved weight history `observations × assets`.
  - `t`: Position of the observation.
  - `k`, `l`: Positions of the two factors.

# Returns

  - `rho::Real`: The correlation, or `NaN`.
  - `degenerate::Bool`: `true` when the pair shares three assets and one of the two cross-sections is constant over them.

# Related

  - [`exposure_pair_correlation`](@ref)
  - [`exposure_pair_sums`](@ref)
"""
function exposure_pair_observation(B::Arr3Num, u::MatNum, t::Integer, k::Integer,
                                   l::Integer)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    m = exposure_pair_sums(B, u, t, k, l)
    if !(m.W > zero(Tf)) || m.nv < 3
        return Tf(NaN), false
    end
    epsv = Tf(1e-12)
    rtol = Tf(1e-9)
    vk = max(m.Qk - m.Sk * m.Sk / m.W, zero(Tf))
    vl = max(m.Ql - m.Sl * m.Sl / m.W, zero(Tf))
    if vk <= epsv + rtol * m.Qk || vl <= epsv + rtol * m.Ql
        return Tf(NaN), true
    end
    den = sqrt(vk * vl)
    return den > epsv ? (m.C - m.Sk * m.Sl / m.W) / den : Tf(NaN), false
end
"""
    exposure_pair_sums(B::Arr3Num, u::MatNum, t::Integer, k::Integer, l::Integer)

Return the weighted sums of one pair of factor exposures over the assets they share at one observation.

An asset enters when both of its exposures are finite, so the support is the one the pair shares and not the one either factor holds alone. The six sums are what the correlation of the pair is a function of.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `u`: Resolved weight history `observations × assets`.
  - `t`: Position of the observation.
  - `k`, `l`: Positions of the two factors.

# Returns

  - `m::NamedTuple`: `(; nv, W, Sk, Sl, Qk, Ql, C)`, the count of the shared assets, the weight sum, the two weighted sums, the two weighted square sums and the weighted cross sum.

# Related

  - [`exposure_pair_observation`](@ref)
"""
function exposure_pair_sums(B::Arr3Num, u::MatNum, t::Integer, k::Integer, l::Integer)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    N = size(B, 2)
    nv = 0
    W = zero(Tf)
    Sk = zero(Tf)
    Sl = zero(Tf)
    Qk = zero(Tf)
    Ql = zero(Tf)
    C = zero(Tf)
    for i in 1:N
        bk = B[t, i, k]
        bl = B[t, i, l]
        if isfinite(bk) && isfinite(bl)
            nv += 1
            wi = Tf(u[t, i])
            xk = Tf(bk)
            xl = Tf(bl)
            W += wi
            Sk += wi * xk
            Sl += wi * xl
            Qk += wi * xk * xk
            Ql += wi * xl * xl
            C += wi * xk * xl
        end
    end
    return (; nv = nv, W = W, Sk = Sk, Sl = Sl, Qk = Qk, Ql = Ql, C = C)
end
"""
    exposure_ic(B::Arr3Num, R::MatNum, w::Option{<:MatNum} = nothing;
                horizon::Integer = 1, rank::Bool = true) -> Matrix{<:Real}
    exposure_ic(csfm::CrossSectionalFactorModel; horizon::Integer = 1, rank::Bool = true,
                reduced::Bool = false) -> Matrix{<:Real}

Return the information coefficient of every factor exposure, one row per pair of observations.

The information coefficient is the cross-sectional correlation between the exposure known at an observation and the mean asset return over the observations that follow it. It scores the exposure as a **forecast of the return**.

A risk factor with an information coefficient near zero is not a bad risk factor. A risk factor is built to explain the covariance and not to predict the mean, so read [`exposure_stability`](@ref) and the variance the factor contributes before you judge one, and never the information coefficient alone.

The level-2 method reconstructs the asset returns as ``\\mathbf{B}_{t-\\ell} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}``, reads `rw` for the weights of the Pearson form, and answers on the raw factor axis. `reduced` maps the exposures through the family re-basis of the block first, and then the answer is on the reduced axis.

# Mathematical definition

```math
\\mathrm{IC}_{tk} = \\rho \\left( \\mathbf{B}_{t \\cdot k}, \\boldsymbol{y}_{t}, \\boldsymbol{u}_{t} \\right)
\\qquad
y_{ti} = \\frac{1}{H} \\sum_{h=1}^{H} r_{t+h,i}
```

Where:

  - ``\\mathbf{B}_{t \\cdot k}``: Cross-section of factor ``k`` at observation ``t``.
  - ``\\boldsymbol{y}_{t}``: Forward mean asset return of observation ``t``.
  - ``\\rho``: The rank correlation when `rank`, and the weighted correlation otherwise.
  - ``H``: Forward window, in observations.

# Algorithm

 1. Resolve the weight history with [`exposure_weights`](@ref).
 2. Build the forward mean return with [`exposure_forward_mean_return`](@ref).
 3. Correlate each exposure against it, one observation at a time, with [`cs_spearman_correlation`](@ref) when `rank` and [`cs_weighted_correlation`](@ref) otherwise.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `R`: Asset return history `observations × assets`, on the observation axis of `B`.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights. The rank form reads no weights.
  - `horizon`: Forward window, in observations.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise.
  - `csfm`: A cross-sectional factor model block.
  - `reduced`: Map the exposures through the family re-basis of the block before the correlation.

# Validation

  - `!isempty(B)`.
  - `size(R) == (size(B, 1), size(B, 2))`.
  - `horizon >= 1` and `size(B, 1) > horizon`.
  - `csfm.Ms` and `csfm.csr` are not `nothing`, else an `IsNothingError` naming the field is raised.

# Returns

  - `ic::Matrix{<:Real}`: `(observations - horizon) × factors`. Row `t` scores the exposures of observation `t` against the returns that follow it.

# Related

  - [`exposure_ic_summary`](@ref)
  - [`exposure_forward_mean_return`](@ref)
  - [`cs_spearman_correlation`](@ref)
  - [`cs_weighted_correlation`](@ref)
  - [`exposure_stability`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_ic(B::Arr3Num, R::MatNum, w::Option{<:MatNum} = nothing;
                     horizon::Integer = 1, rank::Bool = true)
    u = exposure_weights(B, w)
    T, N, K = size(B)
    @argcheck(size(R, 1) == T && size(R, 2) == N,
              DimensionMismatch("R ($(size(R, 1))×$(size(R, 2))) must match B ($T×$N on its first two axes)"))
    @argcheck(horizon >= 1, DomainError(horizon, "horizon must be >= 1"))
    y = exposure_forward_mean_return(R, horizon)
    Tf = promote_type(real(eltype(B)), real(eltype(y)), real(eltype(u)))
    P = T - horizon
    ic = Matrix{Tf}(undef, P, K)
    for k in 1:K, t in 1:P
        a = view(B, t, :, k)
        b = view(y, t, :)
        ic[t, k] = if rank
            Tf(cs_spearman_correlation(a, b))
        else
            Tf(cs_weighted_correlation(a, b, view(u, t, :)))
        end
    end
    return ic
end
function exposure_ic(csfm::CrossSectionalFactorModel; horizon::Integer = 1,
                     rank::Bool = true, reduced::Bool = false)
    B, R, w = exposure_ic_data(csfm, reduced)
    return exposure_ic(B, R, w; horizon = horizon, rank = rank)
end
"""
    exposure_ic_summary(ic::MatNum)
    exposure_ic_summary(csfm::CrossSectionalFactorModel; horizon::Integer = 1,
                        rank::Bool = true, reduced::Bool = false)

Return the summary of an information coefficient series, one entry per factor.

The mean states the average score, the standard deviation states how much the score moves, their ratio states the score per unit of movement, the t-statistic states whether the mean is far enough from zero to believe over the observations that carried a score, and the hit rate states how often the score was positive. The hit rate counts an observation whose score is `NaN` as a miss, so it is read against every observation while the other four are read against the observations that carried a score.

# Mathematical definition

```math
\\overline{\\mathrm{IC}}_{k} = \\frac{1}{|\\mathcal{T}_{k}|} \\sum_{t \\in \\mathcal{T}_{k}} \\mathrm{IC}_{tk}
\\qquad
\\mathrm{IR}_{k} = \\frac{\\overline{\\mathrm{IC}}_{k}}{s_{k}}
\\qquad
t_{k} = \\mathrm{IR}_{k} \\sqrt{\\left| \\mathcal{T}_{k} \\right|}
\\qquad
\\mathrm{hit}_{k} = \\frac{1}{T} \\sum_{t=1}^{T} \\mathbb{1}\\left[\\mathrm{IC}_{tk} > 0\\right]
```

Where:

  - ``\\mathrm{IC}_{tk}``: Information coefficient of factor ``k`` at observation ``t``.
  - ``\\mathcal{T}_{k}``: The observations at which it is finite.
  - ``s_{k}``: Its standard deviation over ``\\mathcal{T}_{k}``, with one degree of freedom removed.
  - $(math_dict[:T])

# Arguments

  - `ic`: Information coefficient series `pairs × factors`.
  - `csfm`: A cross-sectional factor model block.
  - `horizon`: Forward window, in observations.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise.
  - `reduced`: Map the exposures through the family re-basis of the block before the correlation.

# Validation

  - `!isempty(ic)`.

# Returns

  - `summary::NamedTuple`: `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`, each one entry per factor.

# Related

  - [`exposure_ic`](@ref)
  - [`exposure_ic_factor_summary`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_ic_summary(ic::MatNum)
    @argcheck(!isempty(ic), IsEmptyError("ic cannot be empty"))
    P, K = size(ic)
    Tf = real(eltype(ic))
    mean_ic = Vector{Tf}(undef, K)
    std_ic = Vector{Tf}(undef, K)
    ic_ir = Vector{Tf}(undef, K)
    t_stat = Vector{Tf}(undef, K)
    hit_rate = Vector{Tf}(undef, K)
    for k in 1:K
        m = exposure_ic_factor_summary(ic, k)
        mean_ic[k] = m.mean_ic
        std_ic[k] = m.std_ic
        ic_ir[k] = m.ic_ir
        t_stat[k] = m.t_stat
        hit_rate[k] = m.hit_rate
    end
    return (; mean_ic = mean_ic, std_ic = std_ic, ic_ir = ic_ir, t_stat = t_stat,
            hit_rate = hit_rate)
end
"""
    exposure_ic_factor_summary(ic::MatNum, k::Integer)

Return the four summary numbers of one factor's information coefficient series.

The mean and the standard deviation read the observations at which the coefficient is defined, and the hit rate reads every observation, so an observation with no coefficient counts as a miss. The ratio has no answer where either of its two terms has none, and none where the standard deviation is zero. The t-statistic scales the ratio by the root of the number of observations that carried a coefficient, so it says whether the mean is far enough from zero to believe over the evidence there was; it inherits the ratio's absence.

It is the kernel of every summary of a per-observation correlation series, so [`forecast_ic_summary`](@ref) reads it too, and a coefficient of a factor exposure and a coefficient of a Return Forecast are summarised on the same terms.

# Arguments

  - `ic`: Information coefficient series `pairs × factors`.
  - `k`: Position of the factor.

# Returns

  - `m::NamedTuple`: `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`, five numbers.

# Related

  - [`exposure_ic_summary`](@ref)
  - [`forecast_ic_summary`](@ref)
"""
function exposure_ic_factor_summary(ic::MatNum, k::Integer)
    P = size(ic, 1)
    Tf = real(eltype(ic))
    n = 0
    s = zero(Tf)
    h = 0
    for t in 1:P
        v = ic[t, k]
        if isfinite(v)
            n += 1
            s += Tf(v)
        end
        h += v > 0
    end
    m = n > 0 ? s / n : Tf(NaN)
    q = zero(Tf)
    for t in 1:P
        v = ic[t, k]
        if isfinite(v)
            d = Tf(v) - m
            q += d * d
        end
    end
    sd = n > 1 ? sqrt(q / (n - 1)) : Tf(NaN)
    ir = isfinite(m) && isfinite(sd) && sd > zero(Tf) ? m / sd : Tf(NaN)
    return (; mean_ic = m, std_ic = sd, ic_ir = ir, t_stat = ir * sqrt(Tf(n)),
            hit_rate = Tf(h) / Tf(P))
end
function exposure_ic_summary(csfm::CrossSectionalFactorModel; horizon::Integer = 1,
                             rank::Bool = true, reduced::Bool = false)
    return exposure_ic_summary(exposure_ic(csfm; horizon = horizon, rank = rank,
                                           reduced = reduced))
end
"""
    exposure_stability(B::Arr3Num, w::Option{<:MatNum} = nothing;
                       step::Integer = 21) -> Matrix{<:Real}
    exposure_stability(csfm::CrossSectionalFactorModel; step::Integer = 21,
                       weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric()) -> Matrix{<:Real}

Return the stability of every factor exposure, one row per pair of observations.

The stability is the correlation of the cross-section of an exposure against its own cross-section `step` observations later. An exposure that ranks the assets the same way from one month to the next reads near `1`, and one that re-orders them reads near `0`. A risk factor needs a stable exposure, because a turnover of the exposure is a turnover of the portfolio that holds it.

The level-2 method answers on the raw factor axis, and it reads the unlagged history.

# Mathematical definition

```math
\\mathrm{S}_{tk} = \\rho \\left( \\mathbf{B}_{t \\cdot k}, \\mathbf{B}_{t + s, \\cdot k}, \\boldsymbol{u}_{t} \\right)
```

Where:

  - ``\\mathbf{B}_{t \\cdot k}``: Cross-section of factor ``k`` at observation ``t``.
  - ``\\boldsymbol{u}_{t}``: Cross-sectional weights of observation ``t``, the earlier of the two.
  - ``\\rho``: The weighted correlation of two cross-sections.
  - ``s``: Number of observations between the two cross-sections.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `step`: Number of observations between the two cross-sections.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `step >= 1` and `size(B, 1) > step`.
  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.

# Returns

  - `S::Matrix{<:Real}`: `(observations - step) × factors`.

# Related

  - [`cs_weighted_correlation`](@ref)
  - [`exposure_correlation`](@ref)
  - [`exposure_dispersion`](@ref)
  - [`cs_diagnostic_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_stability(B::Arr3Num, w::Option{<:MatNum} = nothing; step::Integer = 21)
    u = exposure_weights(B, w)
    T, N, K = size(B)
    @argcheck(step >= 1, DomainError(step, "step must be >= 1"))
    @argcheck(T > step,
              DimensionMismatch("B ($T observations) must carry more observations than step ($step)"))
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    P = T - step
    S = Matrix{Tf}(undef, P, K)
    for k in 1:K, t in 1:P
        S[t, k] = Tf(cs_weighted_correlation(view(B, t, :, k), view(B, t + step, :, k),
                                             view(u, t, :)))
    end
    return S
end
function exposure_stability(csfm::CrossSectionalFactorModel; step::Integer = 21,
                            weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric())
    return exposure_stability(cs_diagnostic_exposures(csfm),
                              cs_diagnostic_weights(weighting, csfm); step = step)
end
"""
    exposure_dispersion(B::Arr3Num, w::Option{<:MatNum} = nothing) -> Matrix{<:Real}
    exposure_dispersion(csfm::CrossSectionalFactorModel;
                        weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric()) -> Matrix{<:Real}

Return the weighted cross-sectional standard deviation of every factor exposure, one row per observation.

The level of the answer follows the standardisation the exposures were built under, so read the series and not the level: a collapse is a data feed that stopped, and a jump is one asset that ran away. An observation at which the factor carries no weight reads `NaN`.

The level-2 method answers on the raw factor axis, and it reads the unlagged history.

# Mathematical definition

```math
\\mathrm{D}_{tk} = \\sqrt{\\sum_{i} \\tilde{u}_{tik} \\left( B_{tik} - \\sum_{j} \\tilde{u}_{tjk} B_{tjk} \\right)^{2}}
```

Where:

  - ``B_{tik}``: Exposure of asset ``i`` to factor ``k`` at observation ``t``.
  - ``\\tilde{u}_{tik}``: Weight of asset ``i``, zero where the exposure is not finite, and normalised over the assets to sum to one.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.

# Returns

  - `D::Matrix{<:Real}`: `observations × factors`.

# Related

  - [`exposure_weights`](@ref)
  - [`exposure_stability`](@ref)
  - [`exposure_coverage`](@ref)
  - [`cs_diagnostic_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_dispersion(B::Arr3Num, w::Option{<:MatNum} = nothing)
    u = exposure_weights(B, w)
    T, N, K = size(B)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    D = Matrix{Tf}(undef, T, K)
    for k in 1:K, t in 1:T
        D[t, k] = Tf(exposure_cross_section_std(B, u, t, k))
    end
    return D
end
"""
    exposure_cross_section_std(B::Arr3Num, u::MatNum, t::Integer, k::Integer)

Return the weighted cross-sectional standard deviation of one factor exposure at one observation.

The weights are normalised over the assets at which the exposure is finite, so an absent exposure costs its own asset and no other. An observation at which the factor carries no weight has no answer.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `u`: Resolved weight history `observations × assets`.
  - `t`: Position of the observation.
  - `k`: Position of the factor.

# Returns

  - `sigma::Real`: The standard deviation, or `NaN`.

# Related

  - [`exposure_dispersion`](@ref)
"""
function exposure_cross_section_std(B::Arr3Num, u::MatNum, t::Integer, k::Integer)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    N = size(B, 2)
    ws = zero(Tf)
    for i in 1:N
        ws += isfinite(B[t, i, k]) ? Tf(u[t, i]) : zero(Tf)
    end
    if !(ws > zero(Tf))
        return Tf(NaN)
    end
    m = zero(Tf)
    for i in 1:N
        m += isfinite(B[t, i, k]) ? Tf(u[t, i]) / ws * Tf(B[t, i, k]) : zero(Tf)
    end
    v = zero(Tf)
    for i in 1:N
        d = Tf(B[t, i, k]) - m
        v += isfinite(B[t, i, k]) ? Tf(u[t, i]) / ws * d * d : zero(Tf)
    end
    return sqrt(v)
end
function exposure_dispersion(csfm::CrossSectionalFactorModel;
                             weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric())
    return exposure_dispersion(cs_diagnostic_exposures(csfm),
                               cs_diagnostic_weights(weighting, csfm))
end
"""
    exposure_coverage(B::Arr3Num, w::Option{<:MatNum} = nothing) -> Vector{<:Real}
    exposure_coverage(csfm::CrossSectionalFactorModel;
                      weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric()) -> Vector{<:Real}

Return the coverage of every factor exposure, one entry per factor.

The coverage is the share of the universe at which the exposure is finite, averaged over the observations. The universe of an observation is the assets that carry a positive weight, and it is every asset when no weight history is read. A factor whose coverage falls has lost its panel field on part of the universe, and the fit of that observation then rests on fewer assets than the caller believes.

The level-2 method answers on the raw factor axis, and it reads the unlagged history.

# Mathematical definition

```math
\\mathrm{c}_{k} = \\frac{1}{T} \\sum_{t=1}^{T} \\frac{\\left| \\left\\{ i \\in \\mathcal{U}_{t} : B_{tik} \\text{ is finite} \\right\\} \\right|}{\\left| \\mathcal{U}_{t} \\right|}
```

Where:

  - ``B_{tik}``: Exposure of asset ``i`` to factor ``k`` at observation ``t``.
  - ``\\mathcal{U}_{t}``: Universe of observation ``t``, the assets of positive weight.
  - $(math_dict[:T])

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for every asset.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the universe is read off, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.

# Returns

  - `c::Vector{<:Real}`: One entry per factor, between `0` and `1`. An observation whose universe is empty contributes `0`.

# Related

  - [`exposure_weights`](@ref)
  - [`exposure_dispersion`](@ref)
  - [`cs_diagnostic_weights`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_coverage(B::Arr3Num, w::Option{<:MatNum} = nothing)
    u = exposure_weights(B, w)
    T, N, K = size(B)
    Tf = promote_type(real(eltype(B)), real(eltype(u)))
    c = zeros(Tf, K)
    for t in 1:T
        ne = exposure_universe_size(u, t)
        for k in 1:K
            nc = exposure_covered_count(B, u, t, k)
            c[k] += ne > 0 ? Tf(nc) / Tf(ne) : zero(Tf)
        end
    end
    return c ./ T
end
"""
    exposure_universe_size(u::MatNum, t::Integer)

Return the number of assets in the universe of one observation.

The universe is the assets of positive weight, which is every asset when the caller reads no weight history.

# Arguments

  - `u`: Resolved weight history `observations × assets`.
  - `t`: Position of the observation.

# Returns

  - `n::Int`: Number of assets in the universe.

# Related

  - [`exposure_coverage`](@ref)
  - [`exposure_covered_count`](@ref)
"""
function exposure_universe_size(u::MatNum, t::Integer)
    n = 0
    for i in 1:size(u, 2)
        n += u[t, i] > 0
    end
    return n
end
"""
    exposure_covered_count(B::Arr3Num, u::MatNum, t::Integer, k::Integer)

Return the number of assets of the universe of one observation at which one factor exposure is finite.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `u`: Resolved weight history `observations × assets`.
  - `t`: Position of the observation.
  - `k`: Position of the factor.

# Returns

  - `n::Int`: Number of covered assets.

# Related

  - [`exposure_coverage`](@ref)
  - [`exposure_universe_size`](@ref)
"""
function exposure_covered_count(B::Arr3Num, u::MatNum, t::Integer, k::Integer)
    n = 0
    for i in 1:size(B, 2)
        n += u[t, i] > 0 && isfinite(B[t, i, k])
    end
    return n
end
function exposure_coverage(csfm::CrossSectionalFactorModel;
                           weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric())
    return exposure_coverage(cs_diagnostic_exposures(csfm),
                             cs_diagnostic_weights(weighting, csfm))
end
"""
    cs_diagnostic_exposures(csfm::CrossSectionalFactorModel)
    cs_diagnostic_exposures(Ms::Nothing)
    cs_diagnostic_exposures(Ms::Arr3Num)

Return the unlagged exposure history an exposure diagnostic reads off a factor model block.

The exposure group reads the history as the panel wrote it, so it takes no lag and no family re-basis. The absent case is the dispatch rather than a branch, and its message names the field the caller must populate.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `Ms`: The `Ms` field of the block, or `nothing`.

# Validation

  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.

# Returns

  - `Ms::Arr3Num`: Exposure history `observations × assets × factors`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`exposure_correlation`](@ref)
  - [`cs_diagnostic_weights`](@ref)
"""
function cs_diagnostic_exposures(csfm::CrossSectionalFactorModel)
    return cs_diagnostic_exposures(csfm.Ms)
end
function cs_diagnostic_exposures(::Nothing)
    return throw(IsNothingError("Ms cannot be nothing: an exposure diagnostic reads the exposure history of the block"))
end
function cs_diagnostic_exposures(Ms::Arr3Num)
    return Ms
end
"""
    exposure_ic_data(csfm::CrossSectionalFactorModel, reduced::Bool)

Return the exposure history, the reconstructed asset returns and the regression weights the information coefficient reads off a factor model block.

The block carries no asset returns, so they are reconstructed from the parts it does carry: the systematic return of the fit and the residual. The reconstruction is defined from observation ``\\ell + 1``, so the answer starts at observation ``\\max(1, \\ell)``, whose own return row is written `NaN` and is never read: the information coefficient scores an exposure against the returns that **follow** it.

# Algorithm

 1. Refuse a block that carries no exposure history, or no cross-sectional fit.
 2. Reconstruct the return of each observation as ``\\mathbf{B}_{t-\\ell} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}``, and write `NaN` where the lag leaves it undefined.
 3. Map the exposures through the family re-basis when `reduced`, and trim all three histories to the observations from ``\\max(1, \\ell)``.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `reduced`: Map the exposures through the family re-basis of the block.

# Validation

  - `csfm.Ms` is not `nothing`, else an `IsNothingError` naming `Ms` is raised.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` naming `csr` is raised.
  - `size(csfm.Ms, 1) > csfm.lag`, and `csfm.csr.f` carries the observation axis of `csfm.Ms`.

# Returns

  - `B::Arr3Num`: Exposure history of the trimmed observation axis.
  - `R::MatNum`: Reconstructed asset returns of the same axis.
  - `w::Option{<:MatNum}`: Regression weights of the same axis, or `nothing`.

# Related

  - [`exposure_ic`](@ref)
  - [`cs_regression_lag`](@ref)
  - [`cs_lagged_rows`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_ic_data(csfm::CrossSectionalFactorModel, reduced::Bool)
    return exposure_ic_data(csfm, csfm.Ms, csfm.csr, reduced)
end
function exposure_ic_data(::CrossSectionalFactorModel, ::Nothing,
                          ::Option{<:CrossSectionalRegression}, ::Bool)
    return throw(IsNothingError("Ms cannot be nothing: the exposure information coefficient reads the exposure history of the block"))
end
function exposure_ic_data(::CrossSectionalFactorModel, ::Arr3Num, ::Nothing, ::Bool)
    return throw(IsNothingError("csr cannot be nothing: the exposure information coefficient reads the factor returns and the residuals of the block"))
end
function exposure_ic_data(csfm::CrossSectionalFactorModel, Ms::Arr3Num,
                          csr::CrossSectionalRegression, reduced::Bool)
    lag = cs_regression_lag(csfm.lag)
    T, N, K = size(Ms)
    @argcheck(T > lag,
              DimensionMismatch("Ms ($T observations) must carry more observations than lag ($lag)"))
    @argcheck(size(csr.f, 1) == T && size(csr.f, 2) == K,
              DimensionMismatch("csr.f ($(size(csr.f, 1))×$(size(csr.f, 2))) must match Ms ($T observations, $K factors)"))
    Tf = promote_type(real(eltype(Ms)), real(eltype(csr.f)), real(eltype(csr.eps)))
    start = max(1, lag)
    P = T - start + 1
    R = fill(Tf(NaN), P, N)
    for r in 1:P
        tau = start + r - 1
        j = tau - lag
        if j >= 1
            for i in 1:N
                s = zero(Tf)
                for k in 1:K
                    s += Tf(Ms[j, i, k]) * Tf(csr.f[tau, k])
                end
                R[r, i] = s + Tf(csr.eps[tau, i])
            end
        end
    end
    Bf = exposure_ic_exposures(csfm.fcb, Ms, reduced)
    return Bf[start:T, :, :], R, cs_lagged_rows(csfm.rw, start:T)
end
"""
    exposure_ic_exposures(fcb::Nothing, Ms::Arr3Num, reduced::Bool)
    exposure_ic_exposures(fcb::FactorFamilyBasis, Ms::Arr3Num, reduced::Bool)

Return the exposure history the information coefficient scores, on the raw axis or on the reduced one.

A block that carries no family re-basis has one axis, so that case returns the history unchanged whatever `reduced` states.

# Arguments

  - `fcb`: The `fcb` field of a [`CrossSectionalFactorModel`](@ref), or `nothing`.
  - `Ms`: Unlagged exposure history `observations × assets × factors`.
  - `reduced`: Map the history through the re-basis.

# Returns

  - `B::Arr3Num`: The history, on the reduced axis when a re-basis is set and `reduced` is `true`.

# Related

  - [`FactorFamilyBasis`](@ref)
  - [`reduce_exposures`](@ref)
  - [`exposure_ic_data`](@ref)
"""
function exposure_ic_exposures(::Nothing, Ms::Arr3Num, ::Bool)
    return Ms
end
function exposure_ic_exposures(fcb::FactorFamilyBasis, Ms::Arr3Num, reduced::Bool)
    return reduced ? reduce_exposures(fcb, Ms) : Ms
end

export exposure_correlation, exposure_ic, exposure_ic_summary, exposure_stability,
       exposure_dispersion, exposure_coverage
