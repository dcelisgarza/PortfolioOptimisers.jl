"""
    exposure_weights(B::Arr3Num, w::Nothing)
    exposure_weights(B::Arr3Num, w::MatNum)

Return the cross-sectional weight history an exposure diagnostic reads, checking it against the exposure history.

Every diagnostic of this file weights the assets of one observation, and an absent weight history means equal weights. The function resolves the absent case once, into a history of ones, so each kernel reads one matrix and no kernel carries a branch. An asset enters a diagnostic only where its weight is finite and positive, so a weight of zero, a `NaN` or an infinite weight excludes the asset from that observation.

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

The information coefficient of an exposure against the forward return and the stability of an exposure against its own later self both call this kernel. An asset enters when both of its values are finite and its weight is finite and positive, so a missing value costs the pair alone and not the observation. A cross-section that is constant in one of the two values has no correlation, and the answer there is `NaN`.

# Mathematical definition

```math
\\begin{align}
\\rho &= \\frac{\\sum_{i \\in \\mathcal{E}} u_{i} (a_{i} - \\bar{a}) (b_{i} - \\bar{b})}{\\sqrt{\\sum_{i \\in \\mathcal{E}} u_{i} (a_{i} - \\bar{a})^{2}} \\sqrt{\\sum_{i \\in \\mathcal{E}} u_{i} (b_{i} - \\bar{b})^{2}}}\\,, \\\\
\\bar{a} &= \\frac{\\sum_{i \\in \\mathcal{E}} u_{i} a_{i}}{\\sum_{i \\in \\mathcal{E}} u_{i}}\\,.
\\end{align}
```

Where:

  - ``a_{i}``, ``b_{i}``: The two values of asset ``i``.
  - ``u_{i}``: Weight of asset ``i``.
  - ``\\mathcal{E}``: The assets at which both values are finite and the weight is finite and positive.
  - ``\\bar{a}``: Weighted mean of the first cross-section. ``\\bar{b}`` is the weighted mean of the second.

# Algorithm

 1. Count the assets that enter into `n`. Accumulate the weight sum `ws`, the two weighted sums `sa` and `sb`, and the least and greatest value of each cross-section, `lo` and `hi`, over them.
 2. Answer `NaN` when fewer than `min_count` assets enter, or when the weight sum is zero.
 3. Clamp each weighted mean, `ma` and `mb`, to the range of its cross-section. The mean of equal values rounds, and the clamp makes the mean of a constant cross-section its common value exactly.
 4. Accumulate the weighted central moments `cab`, `caa` and `cbb` over the same assets.
 5. Answer `NaN` when the denominator `den` is at or under `eps`. A cross-section that is constant in one of the two values has zero deviations, so its denominator is zero.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(a)), real(eltype(b)),
                                                       real(eltype(u)))))))
    N = length(a)
    n = 0
    ws = zero(Tf)
    sa = zero(Tf)
    sb = zero(Tf)
    lo = (Tf(Inf), Tf(Inf))
    hi = (Tf(-Inf), Tf(-Inf))
    for i in 1:N
        if cs_correlation_enters(a, b, u, i)
            n += 1
            wi = Tf(u[i])
            ai = Tf(a[i])
            bi = Tf(b[i])
            ws += wi
            sa += wi * ai
            sb += wi * bi
            lo = min.(lo, (ai, bi))
            hi = max.(hi, (ai, bi))
        end
    end
    if n < min_count || !(ws > zero(Tf))
        return Tf(NaN)
    end
    # The mean of equal values rounds, so a constant cross-section would keep deviations of
    # the order of the round-off. The clamp makes its mean the common value exactly.
    ma = clamp(sa / ws, lo[1], hi[1])
    mb = clamp(sb / ws, lo[2], hi[2])
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

An asset enters when both of its values are finite and its weight is finite and positive. The two passes of [`cs_weighted_correlation`](@ref) read the same rule, so the rule has one home.

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
    cs_ranks(key::AbstractVector, valid::AbstractVector{Bool}, ties::Symbol)

Return the rank of every entry of a cross-section, and `NaN` outside a mask.

The rank is the position the entry takes in the sorted order. `ties` sets the rank of equal values. `:average` gives each value of a tie the mean of the positions the tie takes, so the ranks do not read the order of the assets. `:ordinal` gives the values of a tie consecutive positions, in the order of the asset axis, which the stable sort keeps. The masked entries sort to the end by their key, and the function writes them as `NaN`, so they take no rank and shift none.

# Mathematical definition

```math
\\begin{align}
r_{i} &= \\begin{cases}
k_{i} + \\dfrac{m_{i} + 1}{2} & \\texttt{:average}\\,, \\\\
\\pi_{i} & \\texttt{:ordinal}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``r_{i}``: Rank of asset ``i``.
  - ``k_{i}``: Number of assets in the mask whose key is below the key of asset ``i``.
  - ``m_{i}``: Number of assets in the mask whose key equals the key of asset ``i``, asset ``i`` included.
  - ``\\pi_{i}``: Position of asset ``i`` in the stable sort of the keys.

# Algorithm

 1. Sort the positions of `key` with a stable sort, giving `p`.
 2. Walk `p` in runs. Under `:average`, a run holds every entry of one key. Under `:ordinal`, a run holds one entry.
 3. Write the mean position of the run to each entry of the run that is in the mask. Leave `NaN` at each entry outside it.

# Arguments

  - `key`: Sort key of each asset. The key of a masked asset must be `Inf`, so the mask sorts to the end.
  - `valid`: Mask, one entry per asset. A `false` entry gets `NaN`.
  - $(arg_dict[:cs_ties])

# Validation

  - `ties` is `:average` or `:ordinal`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `r::VecNum`: Rank of each asset, or `NaN` outside the mask.

# Examples

```jldoctest
julia> PortfolioOptimisers.cs_ranks([3.0, 1.0, 1.0, 2.0], trues(4), :average)
4-element Vector{Float64}:
 4.0
 1.5
 1.5
 3.0

julia> PortfolioOptimisers.cs_ranks([3.0, 1.0, 1.0, 2.0], trues(4), :ordinal)
4-element Vector{Float64}:
 4.0
 1.0
 2.0
 3.0
```

# Related

  - [`cs_spearman_correlation`](@ref)
  - [`forecast_portfolio_weights`](@ref)
"""
function cs_ranks(key::AbstractVector, valid::AbstractVector{Bool}, ties::Symbol)
    @argcheck(ties in (:average, :ordinal),
              ConflictingArgumentError("ties must be :average or :ordinal, got :$(ties)"))
    Tf = float_if_integer(real(eltype(key)))
    N = length(key)
    p = sortperm(key)
    # `NaN` is written at a masked entry alone, so a key type that holds no `NaN`, such as a
    # `Rational`, ranks a cross-section with no mask.
    r = Vector{Tf}(undef, N)
    lo = 1
    while lo <= N
        hi = lo
        while ties === :average && hi < N && key[p[hi + 1]] == key[p[lo]]
            hi += 1
        end
        rk = Tf(lo + hi) / 2
        for pos in lo:hi
            i = p[pos]
            r[i] = valid[i] ? rk : Tf(NaN)
        end
        lo = hi + 1
    end
    return r
end
"""
    cs_spearman_correlation(a::AbstractVector, b::AbstractVector;
                            min_count::Integer = 3, eps::Real = 1e-12,
                            ties::Symbol = :average)

Return the rank correlation of two cross-sections of one observation.

It is the unweighted correlation of the ranks of the two cross-sections, taken over the assets at which both values are finite. The rank measures the order of the assets and not their level, so one extreme value moves the answer no more than one ordinary value does.

Under the default `ties = :average`, a cross-section that is constant over those assets has constant ranks, so the answer is `NaN`, as it is for [`cs_weighted_correlation`](@ref). Under `ties = :ordinal`, the order of the assets inside a tie sets part of the answer, and a constant cross-section can answer `1` or `-1`.

# Mathematical definition

```math
\\begin{align}
\\rho_{S} &= \\rho \\left( \\boldsymbol{r}(\\boldsymbol{a}), \\boldsymbol{r}(\\boldsymbol{b}), \\boldsymbol{1} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:rho_S_cs])
  - $(math_dict[:rho_w_cs])
  - ``\\boldsymbol{a}``, ``\\boldsymbol{b}``: The two cross-sections.
  - ``\\boldsymbol{r}(\\boldsymbol{x})``: Ranks of the cross-section ``\\boldsymbol{x}`` over the assets at which both cross-sections are finite, under the rule `ties` names.
  - ``\\boldsymbol{1}``: Equal weights.

# Algorithm

 1. Mark the assets at which both values are finite.
 2. Rank each cross-section over those assets with [`cs_ranks`](@ref).
 3. Correlate the two rank vectors with [`cs_weighted_correlation`](@ref) under equal weights.

# Arguments

  - `a`: First cross-section, one entry per asset.
  - `b`: Second cross-section, one entry per asset.
  - `min_count`: Least number of assets an answer needs.
  - `eps`: Denominator at or under which the answer is `NaN`.
  - $(arg_dict[:cs_ties])

# Validation

  - The rules of [`cs_ranks`](@ref).

# Returns

  - `rho::Real`: The rank correlation, or `NaN`.

# Examples

```jldoctest
julia> a = [1.0, 1.0, 2.0, 2.0];

julia> PortfolioOptimisers.cs_spearman_correlation(a, [1.0, 2.0, 3.0, 4.0]) ≈ 4 / sqrt(20)
true

julia> PortfolioOptimisers.cs_spearman_correlation(a, [1.0, 2.0, 3.0, 4.0]; ties = :ordinal)
1.0
```

# Related

  - [`cs_ranks`](@ref)
  - [`cs_weighted_correlation`](@ref)
  - [`exposure_ic`](@ref)
"""
function cs_spearman_correlation(a::AbstractVector, b::AbstractVector;
                                 min_count::Integer = 3, eps::Real = 1e-12,
                                 ties::Symbol = :average)
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(a)), real(eltype(b)))))))
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
    ra = cs_ranks(ka, valid, ties)
    rb = cs_ranks(kb, valid, ties)
    return cs_weighted_correlation(ra, rb, ones(Tf, N); min_count = min_count, eps = eps)
end
"""
    exposure_forward_mean_return(R::MatNum, horizon::Integer)

Return the forward mean asset return of every observation but the last `horizon` of them.

The information coefficient scores an exposure against what the asset earned after the exposure was known, and this verb builds that target. An observation whose forward window carries no finite return gets `NaN`, and the verb averages a window that is part finite over the finite part alone.

# Mathematical definition

```math
\\begin{align}
y_{ti} &= \\frac{1}{|\\mathcal{H}_{ti}|} \\sum_{h \\in \\mathcal{H}_{ti}} r_{t+h,i}\\,.
\\end{align}
```

Where:

  - ``y_{ti}``: Forward mean return of asset ``i`` at observation ``t``, `NaN` when ``\\mathcal{H}_{ti}`` is empty.
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
    Tf = float_if_integer(real(eltype(R)))
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

The verb correlates each pair one observation at a time, over the assets at which both exposures are finite and the weight is finite and positive, and then averages the answers over the observations at which the pair has one. A pair one of whose exposures is constant across the assets has no correlation, and it reads `0` by convention. A pair that never shares three such assets reads `NaN`. The diagonal is `1`.

The block method answers on the raw factor axis, because the exposure history is what the panel recorded and not the design of the fit. It reads the unlagged history.

# Mathematical definition

```math
\\begin{align}
\\mathbf{C}_{kl} &= \\frac{1}{|\\mathcal{T}_{kl}|} \\sum_{t \\in \\mathcal{T}_{kl}} \\rho \\left( \\mathbf{B}_{t \\cdot k}, \\mathbf{B}_{t \\cdot l}, \\boldsymbol{u}_{t} \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathbf{C}_{kl}``: Time-averaged correlation of the exposures to factors ``k`` and ``l``.
  - $(math_dict[:B_tk_cs])
  - $(math_dict[:u_t_cs])
  - $(math_dict[:rho_w_cs])
  - ``\\mathcal{T}_{kl}``: The observations at which the pair shares at least three assets and neither of its two cross-sections is constant over them.

# Algorithm

 1. Resolve the weight history with [`exposure_weights`](@ref).
 2. For each observation and each pair, accumulate the weighted sums over the assets at which both exposures are finite and the weight is finite and positive.
 3. Mark the pair as degenerate when either weighted variance falls to the tolerance of its own weighted square sum, and as insufficient when it shares fewer than three assets.
 4. Average the pairs that are neither over the observations. A pair that is degenerate at every observation it covers reads `0`, and one that is never sufficient reads `NaN`.
 5. Write `1` on the diagonal.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history that the verb reads off the block, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `size(w) == (size(B, 1), size(B, 2))`, when `w` is present.
  - `csfm.Ms` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `Ms`.
  - The field that `weighting` names is not `nothing`. Otherwise the verb raises an `IsNothingError` that names it.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)))))))
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

The verb correlates the pair at each observation, and averages the answers over the observations at which the pair has one. A pair that is degenerate at every observation it covers reads `0` by convention, and one that never reaches three common assets reads `NaN`.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)))))))
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

The correlation reads the assets at which both exposures are finite and the weight is finite and positive. It has no answer at an observation that shares fewer than three such assets, and none at one where either weighted variance falls to the tolerance of its own weighted square sum. The second case is degenerate, which the caller reads as the `0` convention, and the first is not.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)))))))
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

An asset enters when both of its exposures are finite and its weight is finite and positive, so the support is the one the pair shares and not the one either factor holds alone. The correlation of the pair is a function of these six sums alone.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)))))))
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
        ui = u[t, i]
        if isfinite(bk) && isfinite(bl) && 0 < ui < Inf
            nv += 1
            wi = Tf(ui)
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
                horizon::Integer = 1, rank::Bool = true,
                ties::Symbol = :average) -> Matrix{<:Real}
    exposure_ic(csfm::CrossSectionalFactorModel; horizon::Integer = 1, rank::Bool = true,
                reduced::Bool = false, ties::Symbol = :average) -> Matrix{<:Real}

Return the information coefficient of every factor exposure, one row per pair of observations.

The information coefficient is the cross-sectional correlation between the exposure known at an observation and the mean asset return over the observations that follow it. It scores the exposure as a forecast of the return.

A risk factor with an information coefficient near zero is not a bad risk factor. The purpose of a risk factor is to explain the covariance and not to predict the mean, so read [`exposure_stability`](@ref) and the variance the factor contributes before you judge one, and never the information coefficient alone.

The block method reconstructs the asset returns as ``\\mathbf{B}_{t-\\ell} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}``, reads `rw` for the weights of the Pearson form, and answers on the raw factor axis. `reduced` maps the exposures through the family re-basis of the block first, and then the answer is on the reduced axis.

# Mathematical definition

```math
\\begin{align}
\\mathrm{IC}_{tk} &= \\rho \\left( \\mathbf{B}_{t \\cdot k}, \\boldsymbol{y}_{t}, \\boldsymbol{u}_{t} \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{tk}``: Information coefficient of factor ``k`` at observation ``t``.
  - $(math_dict[:B_tk_cs])
  - ``\\boldsymbol{y}_{t}``: Forward mean asset return of observation ``t``, over the finite returns of the ``H`` observations that follow it. [`exposure_forward_mean_return`](@ref) defines it.
  - $(math_dict[:u_t_cs])
  - ``\\rho``: The rank correlation of [`cs_spearman_correlation`](@ref) when `rank`, which reads no weight, and the weighted correlation of [`cs_weighted_correlation`](@ref) otherwise.
  - ``H``: Forward window, in observations.

A one-hot exposure, such as an industry, is a block of `0` and a block of `1`, so every asset of it is in a tie. Under the default `ties = :average`, its rank coefficient compares the forward returns of the two blocks. Under `ties = :ordinal`, the order of the assets inside each block sets most of it.

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
  - $(arg_dict[:cs_ties]) The weighted correlation reads no rank, so it ignores `ties`.

# Validation

  - `!isempty(B)`.
  - `size(R) == (size(B, 1), size(B, 2))`.
  - `horizon >= 1` and `size(B, 1) > horizon`.
  - `csfm.Ms` and `csfm.csr` are not `nothing`. Otherwise the verb raises an `IsNothingError` that names the field.
  - The rules of [`cs_ranks`](@ref), when `rank`.

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
                     horizon::Integer = 1, rank::Bool = true, ties::Symbol = :average)
    u = exposure_weights(B, w)
    T, N, K = size(B)
    @argcheck(size(R, 1) == T && size(R, 2) == N,
              DimensionMismatch("R ($(size(R, 1))×$(size(R, 2))) must match B ($T×$N on its first two axes)"))
    @argcheck(horizon >= 1, DomainError(horizon, "horizon must be >= 1"))
    y = exposure_forward_mean_return(R, horizon)
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(y)),
                                                       real(eltype(u)))))))
    P = T - horizon
    ic = Matrix{Tf}(undef, P, K)
    for k in 1:K, t in 1:P
        a = view(B, t, :, k)
        b = view(y, t, :)
        ic[t, k] = if rank
            Tf(cs_spearman_correlation(a, b; ties = ties))
        else
            Tf(cs_weighted_correlation(a, b, view(u, t, :)))
        end
    end
    return ic
end
function exposure_ic(csfm::CrossSectionalFactorModel; horizon::Integer = 1,
                     rank::Bool = true, reduced::Bool = false, ties::Symbol = :average)
    B, R, w = exposure_ic_data(csfm, reduced)
    return exposure_ic(B, R, w; horizon = horizon, rank = rank, ties = ties)
end
"""
    exposure_ic_summary(ic::MatNum; lags::Integer = 0)
    exposure_ic_summary(csfm::CrossSectionalFactorModel; horizon::Integer = 1,
                        rank::Bool = true, reduced::Bool = false,
                        ties::Symbol = :average)

Return the summary of an information coefficient series, one entry per factor.

The mean states the average score, the standard deviation states how much the score moves, and their ratio states the score per unit of movement. The t-statistic states whether the mean is far enough from zero to believe over the observations that carried a score, and the hit rate states how often the score was positive. All five read the observations that carried a score. An observation whose score is `NaN` is one at which nothing was measured, not a miss, so it is in no denominator here, as it is in none of the library's other summaries. The coverage of the series answers the separate question of how often the series had no score.

The standard error of the t-statistic is the long-run one of [`exposure_ic_factor_summary`](@ref), over `lags` autocovariances. [`exposure_ic`](@ref) scores every observation against a window of `horizon` observations, so each row shares returns with the `horizon - 1` rows on each side of it, and the block method derives `lags` as `horizon - 1`. The bare method takes `lags` from the caller. Its default of `0` is the independent case, which is the case of a series scored at a stride of its window.

# Mathematical definition

```math
\\begin{align}
\\overline{\\mathrm{IC}}_{k} &= \\frac{1}{|\\mathcal{T}_{k}|} \\sum_{t \\in \\mathcal{T}_{k}} \\mathrm{IC}_{tk}\\,, \\\\
s_{k}^{2} &= \\frac{1}{|\\mathcal{T}_{k}| - 1} \\sum_{t \\in \\mathcal{T}_{k}} \\left( \\mathrm{IC}_{tk} - \\overline{\\mathrm{IC}}_{k} \\right)^{2}\\,, \\\\
\\mathrm{IR}_{k} &= \\frac{\\overline{\\mathrm{IC}}_{k}}{s_{k}}\\,, \\\\
t_{k} &= \\frac{\\overline{\\mathrm{IC}}_{k}}{\\sigma_{k}} \\sqrt{\\left| \\mathcal{T}_{k} \\right|}\\,, \\\\
\\mathrm{hit}_{k} &= \\frac{1}{|\\mathcal{T}_{k}|} \\sum_{t \\in \\mathcal{T}_{k}} \\mathbb{1}\\left[\\mathrm{IC}_{tk} > 0\\right]\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{tk}``: Information coefficient of factor ``k`` at observation ``t``.
  - ``\\mathcal{T}_{k}``: The observations at which it is finite.
  - ``\\overline{\\mathrm{IC}}_{k}``: Its mean over ``\\mathcal{T}_{k}``.
  - ``s_{k}``: Its standard deviation over ``\\mathcal{T}_{k}``.
  - ``\\mathrm{IR}_{k}``: Its information ratio.
  - ``\\sigma_{k}``: Its long-run standard deviation over ``\\mathcal{T}_{k}``, which [`exposure_ic_factor_summary`](@ref) defines and which is ``s_{k}`` at `lags = 0`.
  - ``t_{k}``: Its t-statistic.
  - ``\\mathrm{hit}_{k}``: Its hit rate.

# Arguments

  - `ic`: Information coefficient series `pairs × factors`.
  - `lags`: Number of autocovariances the t-statistic's standard error reads. It is one less than the number of rows a forward window spans, `horizon - 1` for a series scored at every observation.
  - `csfm`: A cross-sectional factor model block.
  - `horizon`: Forward window, in observations.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise.
  - `reduced`: Map the exposures through the family re-basis of the block before the correlation.
  - $(arg_dict[:cs_ties]) The weighted correlation reads no rank, so it ignores `ties`.

# Validation

  - `!isempty(ic)`.
  - `lags >= 0`. Raises a `DomainError`.

# Returns

  - `summary::NamedTuple`: `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`, each one entry per factor.

# Related

  - [`exposure_ic`](@ref)
  - [`exposure_ic_factor_summary`](@ref)
  - [`forecast_factor_correlation`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function exposure_ic_summary(ic::MatNum; lags::Integer = 0)
    @argcheck(!isempty(ic), IsEmptyError("ic cannot be empty"))
    @argcheck(lags >= zero(lags), DomainError(lags, "lags must be >= 0"))
    P, K = size(ic)
    Tm = float_if_integer(real(eltype(ic)))
    Ts = typeof(sqrt(one(Tm)))
    mean_ic = Vector{Tm}(undef, K)
    std_ic = Vector{Ts}(undef, K)
    ic_ir = Vector{Ts}(undef, K)
    t_stat = Vector{Ts}(undef, K)
    hit_rate = Vector{Tm}(undef, K)
    for k in 1:K
        m = exposure_ic_factor_summary(ic, k; lags = lags)
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
    exposure_ic_factor_summary(ic::MatNum, k::Integer; lags::Integer = 0)

Return the five summary numbers of one factor's information coefficient series.

The mean, the standard deviation and the hit rate read the observations at which the coefficient is finite, so an observation with no coefficient is in no denominator and the three agree on their sample. A series with no finite coefficient has no hit rate. The ratio has no answer where either of its two terms has none, and none where the standard deviation is zero. The t-statistic is the mean over the standard error of the mean, so it says whether the mean is far enough from zero to believe over the evidence there was.

[`forecast_ic_summary`](@ref) calls this function too, so a coefficient of a factor exposure and a coefficient of a Return Forecast have the same summary.

# The standard error reads the overlap

A coefficient scored against a forward window that is longer than the stride between two scores reads the same returns at consecutive rows. The rows are then not independent, and the standard error of their mean is not ``s / \\sqrt{n}``. The rows that overlap a given row are the ``L`` on each side of it, where ``L`` is one less than the number of strides a window spans. Under no skill, every autocovariance beyond ``L`` is zero.

The autocovariances up to ``L`` are not zero when the scored quantity keeps its ranking of the assets from one row to the next. An exposure is persistent in this sense, so its series is a moving average of order ``L``. A Return Forecast that draws a new ranking at every row gives autocovariances near zero at every lag, and there the plain statistic ``\\mathrm{IR} \\sqrt{n}`` does not overstate the evidence. The long-run variance then adds sampling noise to the standard error and moves it little.

The standard error therefore reads the long-run variance, the variance plus twice the first ``L`` autocovariances, with no taper. The window and the stride fix the order, so the function does not estimate it from the series. A tapered estimate at the same order under-weights autocovariances that the window and the stride imply, and still overstates the statistic. For a persistent ranking under no skill, the autocovariances fall linearly across the window. The Bartlett taper at that order then keeps two thirds of their sum in the limit of a long window, so the statistic is too large by the root of three halves there, and by less at a short window.

The autocovariances read the rows at their positions in `ic`. A row whose coefficient is `NaN` is in no pair, and the lag between two rows is their distance in the series, not in its finite subsequence. Every autocovariance divides by the same ``n - 1`` as the variance, so at `lags = 0` the standard error is exactly ``s / \\sqrt{n}`` and the statistic is ``\\mathrm{IR} \\sqrt{n}``.

The long-run variance is a sum of signed terms, and a short series can sum it to a number that is not positive. The t-statistic is `NaN` there, as it is where the standard deviation is zero. The function does not clamp it, because a clamped value would report a certainty that the series does not carry.

# Mathematical definition

```math
\\begin{align}
\\gamma_{j} &= \\frac{1}{n - 1} \\sum_{t,\\, t + j \\in \\mathcal{T}} \\left( \\mathrm{IC}_{t} - \\overline{\\mathrm{IC}} \\right) \\left( \\mathrm{IC}_{t + j} - \\overline{\\mathrm{IC}} \\right)\\,, \\\\
\\sigma^{2} &= \\gamma_{0} + 2 \\sum_{j = 1}^{L} \\gamma_{j}\\,, \\\\
t &= \\frac{\\overline{\\mathrm{IC}}}{\\sigma} \\sqrt{n}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{t}``: Coefficient of factor ``k`` at row ``t``.
  - ``\\mathcal{T}``: The rows at which it is finite.
  - ``n``: The number of rows in ``\\mathcal{T}``.
  - ``\\overline{\\mathrm{IC}}``: Mean of the coefficient over ``\\mathcal{T}``.
  - ``\\gamma_{j}``: Autocovariance at lag ``j``. ``\\gamma_{0}`` is the variance ``s^{2}``.
  - ``\\sigma``: Long-run standard deviation.
  - ``t``: The t-statistic.
  - ``L``: `lags`.

# Arguments

  - `ic`: Information coefficient series `pairs × factors`.
  - `k`: Position of the factor.
  - `lags`: Number of autocovariances the standard error reads, `0` for independent rows. [`forecast_ic_lags`](@ref) derives it from a window and a stride.

# Validation

  - `lags >= 0`. Raises a `DomainError`.

# Returns

  - `m::NamedTuple`: `(; mean_ic, std_ic, ic_ir, t_stat, hit_rate)`, five numbers.

# Related

  - [`exposure_ic_summary`](@ref)
  - [`forecast_ic_summary`](@ref)
  - [`forecast_ic_lags`](@ref)
"""
function exposure_ic_factor_summary(ic::MatNum, k::Integer; lags::Integer = 0)
    @argcheck(lags >= zero(lags), DomainError(lags, "lags must be >= 0"))
    P = size(ic, 1)
    Tm = float_if_integer(real(eltype(ic)))
    Ts = typeof(sqrt(one(Tm)))
    n = 0
    s = zero(Tm)
    h = 0
    for t in 1:P
        v = ic[t, k]
        if isfinite(v)
            n += 1
            s += Tm(v)
            h += v > 0
        end
    end
    m = n > 0 ? s / n : Tm(NaN)
    q = zero(Tm)
    for t in 1:P
        v = ic[t, k]
        if isfinite(v)
            d = Tm(v) - m
            q += d * d
        end
    end
    sd = n > 1 ? sqrt(q / (n - 1)) : Ts(NaN)
    ir = isfinite(m) && isfinite(sd) && sd > zero(Ts) ? m / sd : Ts(NaN)
    return (; mean_ic = m, std_ic = sd, ic_ir = ir,
            t_stat = exposure_ic_t_stat(view(ic, :, k), m, q, n, lags),
            hit_rate = n > 0 ? Tm(h) / Tm(n) : Tm(NaN))
end
"""
    exposure_ic_t_stat(c::VecNum, m::Real, q::Real, n::Integer, lags::Integer) -> Real

Return the t-statistic of one factor's information coefficient series, over its long-run standard error.

[`exposure_ic_factor_summary`](@ref) gives it the mean, the sum of squared deviations and the count, so the function reads the series once more, for the autocovariances alone. The long-run variance is the sum of squared deviations plus twice the first `lags` autocovariance sums, every one over ``n - 1``. Each pair reads its two positions in the series, so a `NaN` row is in no pair and the lag is a distance in the series. A long-run variance that is not positive, or a count of one, has no statistic.

# Arguments

  - `c`: Information coefficient series of one factor, a column of the `pairs × factors` matrix.
  - `m`: Mean of the finite entries of `c`.
  - `q`: Sum of the squared deviations of the finite entries of `c` from `m`.
  - `n`: Number of finite entries of `c`.
  - `lags`: Number of autocovariances the standard error reads.

# Returns

  - `t::Real`: `m` over the long-run standard error of the mean, or `NaN`.

# Related

  - [`exposure_ic_factor_summary`](@ref)
"""
function exposure_ic_t_stat(c::VecNum, m::Real, q::Real, n::Integer, lags::Integer)
    Tf = typeof(q)
    Ts = typeof(sqrt(one(Tf)))
    P = length(c)
    lrv = q
    for j in 1:lags, t in 1:(P - j)
        v = c[t]
        u = c[t + j]
        if isfinite(v) && isfinite(u)
            lrv += 2 * (Tf(v) - m) * (Tf(u) - m)
        end
    end
    lrsd = n > 1 && lrv > zero(Tf) ? sqrt(lrv / (n - 1)) : Ts(NaN)
    return isfinite(m) && isfinite(lrsd) ? m / lrsd * sqrt(Ts(n)) : Ts(NaN)
end
function exposure_ic_summary(csfm::CrossSectionalFactorModel; horizon::Integer = 1,
                             rank::Bool = true, reduced::Bool = false,
                             ties::Symbol = :average)
    return exposure_ic_summary(exposure_ic(csfm; horizon = horizon, rank = rank,
                                           reduced = reduced, ties = ties);
                               lags = horizon - 1)
end
"""
    exposure_stability(B::Arr3Num, w::Option{<:MatNum} = nothing;
                       step::Integer = 21) -> Matrix{<:Real}
    exposure_stability(csfm::CrossSectionalFactorModel; step::Integer = 21,
                       weighting::AbstractOrthogonalityMetric = BenchmarkWeightMetric()) -> Matrix{<:Real}

Return the stability of every factor exposure, one row per pair of observations.

The stability is the correlation of the cross-section of an exposure against its own cross-section `step` observations later. An exposure that ranks the assets the same way from one month to the next reads near `1`, and one that re-orders them reads near `0`. A risk factor needs a stable exposure, because a turnover of the exposure is a turnover of the portfolio that holds it.

The block method answers on the raw factor axis, and it reads the unlagged history.

# Mathematical definition

```math
\\begin{align}
\\mathrm{S}_{tk} &= \\rho \\left( \\mathbf{B}_{t \\cdot k}, \\mathbf{B}_{t + s, \\cdot k}, \\boldsymbol{u}_{t} \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{S}_{tk}``: Stability of the exposure to factor ``k`` from observation ``t``.
  - $(math_dict[:B_tk_cs])
  - ``\\boldsymbol{u}_{t}``: Cross-sectional weights of observation ``t``, the earlier of the two.
  - $(math_dict[:rho_w_cs])
  - ``s``: Number of observations between the two cross-sections.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `step`: Number of observations between the two cross-sections.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history that the verb reads off the block, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `step >= 1` and `size(B, 1) > step`.
  - `csfm.Ms` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `Ms`.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)))))))
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

The level of the answer follows the standardisation that built the exposures, so read the series and not the level. A collapse is a data feed that stopped, and a jump is one asset with an extreme exposure. A constant exposure reads exactly `0`. An observation at which the factor carries no weight reads `NaN`.

The block method answers on the raw factor axis, and it reads the unlagged history.

# Mathematical definition

```math
\\begin{align}
\\mathrm{D}_{tk} &= \\sqrt{\\sum_{i} \\tilde{u}_{tik} \\left( B_{tik} - \\sum_{j} \\tilde{u}_{tjk} B_{tjk} \\right)^{2}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{D}_{tk}``: Dispersion of the exposure to factor ``k`` at observation ``t``.
  - $(math_dict[:B_tik_cs])
  - ``\\tilde{u}_{tik}``: Weight of asset ``i``, zero where the exposure is not finite or the weight is not finite and positive, and normalised over the assets to sum to one.

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history that the verb reads off the block, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `csfm.Ms` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `Ms`.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)))))))
    D = Matrix{Tf}(undef, T, K)
    for k in 1:K, t in 1:T
        D[t, k] = Tf(exposure_cross_section_std(B, u, t, k))
    end
    return D
end
"""
    exposure_cross_section_std(B::Arr3Num, u::MatNum, t::Integer, k::Integer)

Return the weighted cross-sectional standard deviation of one factor exposure at one observation.

The function normalises the weights over the assets at which the exposure is finite and the weight is finite and positive, so an absent exposure costs its own asset and no other. It clamps the weighted mean to the range of the cross-section, so a constant cross-section has a standard deviation of exactly `0`. An observation at which the factor carries no weight has no answer.

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
    Tf = typeof(sqrt(one(float_if_integer(promote_type(real(eltype(B)), real(eltype(u)))))))
    N = size(B, 2)
    ws = zero(Tf)
    s = zero(Tf)
    lo = Tf(Inf)
    hi = Tf(-Inf)
    for i in 1:N
        x = B[t, i, k]
        wi = u[t, i]
        if isfinite(x) && 0 < wi < Inf
            ws += Tf(wi)
            s += Tf(wi) * Tf(x)
            lo = min(lo, Tf(x))
            hi = max(hi, Tf(x))
        end
    end
    if !(ws > zero(Tf))
        return Tf(NaN)
    end
    # The clamp makes the mean of a constant cross-section its common value exactly, so its
    # deviations are zero and not the round-off of the mean.
    m = clamp(s / ws, lo, hi)
    v = zero(Tf)
    for i in 1:N
        x = B[t, i, k]
        wi = u[t, i]
        if isfinite(x) && 0 < wi < Inf
            d = Tf(x) - m
            v += Tf(wi) * d * d
        end
    end
    return sqrt(v / ws)
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

The coverage is the share of the universe at which the exposure is finite, averaged over the observations. The universe of an observation is the assets that carry a finite positive weight, and it is every asset when the verb reads no weight history. A factor whose coverage falls has lost its panel field on part of the universe, and the fit of that observation then rests on fewer assets than the caller believes.

The block method answers on the raw factor axis, and it reads the unlagged history.

# Mathematical definition

```math
\\begin{align}
\\mathrm{c}_{k} &= \\frac{1}{T} \\sum_{t=1}^{T} \\frac{\\left| \\left\\{ i \\in \\mathcal{U}_{t} : B_{tik} \\text{ is finite} \\right\\} \\right|}{\\left| \\mathcal{U}_{t} \\right|}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{c}_{k}``: Coverage of the exposure to factor ``k``.
  - $(math_dict[:B_tik_cs])
  - ``\\mathcal{U}_{t}``: Universe of observation ``t``, the assets of finite positive weight. An observation whose universe is empty adds zero to the sum.
  - $(math_dict[:T])

# Arguments

  - `B`: Exposure history `observations × assets × factors`, unlagged.
  - `w`: Cross-sectional weight history `observations × assets`, or `nothing` for every asset.
  - `csfm`: A cross-sectional factor model block.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history that sets the universe, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis.

# Validation

  - `!isempty(B)`.
  - `csfm.Ms` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `Ms`.

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
    Tf = float_if_integer(promote_type(real(eltype(B)), real(eltype(u))))
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

The universe is the assets of finite positive weight, which is every asset when the caller reads no weight history.

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
        n += 0 < u[t, i] < Inf
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
        n += 0 < u[t, i] < Inf && isfinite(B[t, i, k])
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

The exposure group reads the history as the panel wrote it, so it takes no lag and no family re-basis. A method for `nothing` handles the absent case, so the function carries no branch, and its message names the field that the caller must populate.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `Ms`: The `Ms` field of the block, or `nothing`.

# Validation

  - `csfm.Ms` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `Ms`.

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

The block carries no asset returns, so the function reconstructs them from the systematic return of the fit and the residual, which the block does carry. The reconstruction exists from observation ``\\ell + 1``, so the answer starts at observation ``\\max(1, \\ell)``. The return row of that first observation is `NaN`, and no score reads it, because the information coefficient scores an exposure against the returns that follow it.

# Algorithm

 1. Refuse a block that carries no exposure history, or no cross-sectional fit.
 2. Reconstruct the return of each observation as ``\\mathbf{B}_{t-\\ell} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}``, and write `NaN` where the lag leaves it undefined.
 3. Map the exposures through the family re-basis when `reduced`, and trim all three histories to the observations from ``\\max(1, \\ell)``.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `reduced`: Map the exposures through the family re-basis of the block.

# Validation

  - `csfm.Ms` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `Ms`.
  - `csfm.csr` is not `nothing`. Otherwise the verb raises an `IsNothingError` that names `csr`.
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
    Tf = float_if_integer(promote_type(real(eltype(Ms)), real(eltype(csr.f)),
                                       real(eltype(csr.eps))))
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
