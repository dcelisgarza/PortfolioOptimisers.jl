"""
$(DocStringExtensions.TYPEDEF)

Estimates the location of each asset, and divides each deviation by the bias that the estimate puts in it. This is the default centring.

The location of an asset is the exponentially weighted mean of its own valid returns, divided by the sum of its weights. The deviation of a valid return is taken from the location of the returns before it, so the first valid return of an asset gives no deviation, and the estimate of a moment starts at the second. The location is independent of the return that it centres, so the deviation has the variance of the return plus the variance of the location. Each squared deviation is divided by that factor, and so is each product of two deviations, so every exponentially weighted mean of these terms is an unbiased estimate from the second valid return.

# Mathematical definition

Let ``w_{i,s}`` be the weight of the valid return ``s`` of asset ``i`` in its location before the observation ``t``, and ``a_{i,s} = w_{i,s} / \\sum_{u} w_{i,u}`` its normalised weight. For returns that are independent in time, with covariance ``\\sigma_{ij}``,

```math
\\begin{align}
m_{i} &= \\sum_{s} a_{i,s}\\, x_{i,s}\\,, \\\\
e_{i} &= x_{i,t} - m_{i}\\,, \\\\
\\mathbb{E}\\left[e_{i}\\, e_{j}\\right] &= \\sigma_{ij} \\left(1 + c_{ij}\\right)\\,, \\quad c_{ij} = \\sum_{s \\in \\mathcal{V}_i \\cap \\mathcal{V}_j} a_{i,s}\\, a_{j,s}\\,.
\\end{align}
```

On the diagonal, ``1 + c_{ii} = 1 + 1 / n^{\\mathrm{eff}}_{i}``, where ``n^{\\mathrm{eff}}_{i}`` is the Kish count of the weights of the location. With equal weights it is ``k / (k - 1)``, the factor of the sample variance. Under a HAC adjustment a lagged product ``e_{i,t}\\, e_{j,r}`` has a non-zero mean too, because the location of ``t`` holds the return of ``r``, and [`centring_lag_factor`](@ref) adds that mean to the factor.

Where:

  - ``x_{i,t}``: The return of asset ``i`` at the observation ``t``.
  - ``m_{i}``: The location of asset ``i`` before the observation ``t``.
  - ``e_{i}``: The deviation of asset ``i`` at the observation ``t``.
  - ``\\mathcal{V}_i``: The valid observations of asset ``i`` before ``t``.
  - ``c_{ij}``: The overlap of the locations of the pair. The two sums of weights it reads follow the recursions ``S_1 \\leftarrow \\lambda S_1 + (1 - \\lambda)`` and ``S_2 \\leftarrow \\lambda^2 S_2 + (1 - \\lambda)^2``.

# Constructors

    EstimatedCentring() -> EstimatedCentring

# Examples

```jldoctest
julia> ExpWeightedVariance().centring
EstimatedCentring()
```

# Related

  - [`AbstractCentring`](@ref)
  - [`PreCentred`](@ref)
  - [`ExpWeightedVariance`](@ref)
  - [`ExpWeightedCovariance`](@ref)
  - [`centring_factor`](@ref)
"""
struct EstimatedCentring <: AbstractCentring end
"""
$(DocStringExtensions.TYPEDEF)

Takes the returns as deviations from a mean of zero, so each deviation is the return itself.

Every valid return is a term of the estimate, and no term is corrected. The estimate is unbiased where the true mean is zero, and it is too large by the square of the mean otherwise. It suits a series whose mean is zero by construction, such as the residuals of a regression.

# Constructors

    PreCentred() -> PreCentred

# Examples

```jldoctest
julia> ExpWeightedVariance(; centring = PreCentred()).centring
PreCentred()
```

# Related

  - [`AbstractCentring`](@ref)
  - [`EstimatedCentring`](@ref)
"""
struct PreCentred <: AbstractCentring end
"""
$(DocStringExtensions.TYPEDEF)

Takes each deviation from a location that starts at zero and is not divided by its weight. It is the uncorrected recursion, kept one keyword away from the default.

The location of an asset takes the step ``m_{k} = \\lambda m_{k - 1} + (1 - \\lambda) x_{k}`` from ``m_{0} = 0``, and the deviation of a return is taken from the location before it, so the first deviation is the return itself. Every valid return gives a term, and no term is corrected. The weights of the location sum to ``1 - \\lambda^{k}``, so it estimates ``(1 - \\lambda^{k}) \\mu`` rather than the mean, and each deviation keeps part of the mean: the variance is too large, by 2.4 % to 36 % in the warm-up and by 3.5 % after it at a half-life of 10. [`EstimatedCentring`](@ref) is the unbiased rule.

# Constructors

    ZeroStartCentring() -> ZeroStartCentring

# Examples

```jldoctest
julia> ExpWeightedVariance(; centring = ZeroStartCentring()).centring
ZeroStartCentring()
```

# Related

  - [`AbstractCentring`](@ref)
  - [`EstimatedCentring`](@ref)
  - [`PreCentred`](@ref)
"""
struct ZeroStartCentring <: AbstractCentring end
"""
    centring_lag(::EstimatedCentring) -> Int
    centring_lag(::PreCentred) -> Int

Count of the leading valid returns of an asset that give no deviation.

The estimated location needs one return before it exists, so the first return of an asset gives no term. A pre-centred return is its own deviation.

# Returns

  - `lag::Int`: One for [`EstimatedCentring`](@ref), zero for [`PreCentred`](@ref).

# Related

  - [`AbstractCentring`](@ref)
  - [`centring_terms`](@ref)
"""
function centring_lag(::EstimatedCentring)
    return 1
end
function centring_lag(::Union{PreCentred, ZeroStartCentring})
    return 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Count of the terms that the valid returns of an asset give to a moment.

# Arguments

  - `c`: The centring.
  - `n`: Count of the valid returns of the asset.

# Returns

  - `k::Integer`: `n` less the lag of `c`, floored at zero.

# Related

  - [`centring_lag`](@ref)
"""
function centring_terms(c::AbstractCentring, n::Integer)
    return max(n - centring_lag(c), zero(n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Whether an asset has enough valid returns for its moment to be reported.

The asset needs `min_obs` valid returns and at least one term, so under [`EstimatedCentring`](@ref) an asset of one valid return is not ready at any `min_obs`.

# Arguments

  - `c`: The centring.
  - `n`: Count of the valid returns of the asset.
  - `min_obs`: The minimum count of valid returns.

# Returns

  - `flag::Bool`: Whether the asset is ready.

# Related

  - [`centring_terms`](@ref)
"""
function centring_ready(c::AbstractCentring, n::Integer, min_obs::Integer)
    return n >= min_obs && centring_terms(c, n) >= 1
end
"""
    centring_deviation_mask(::PreCentred, valid::AbstractVector{<:Bool},
                            obs_count::AbstractVector{<:Integer})
    centring_deviation_mask(::EstimatedCentring, valid::AbstractVector{<:Bool},
                            obs_count::AbstractVector{<:Integer})

Mask of the assets whose deviation at an observation exists.

Under [`PreCentred`](@ref) every valid asset gives a deviation. Under [`EstimatedCentring`](@ref) a valid asset gives one only when it has a location, that is, a valid return before.

# Arguments

  - `valid`: Mask of the valid assets at the observation.
  - `obs_count`: Count of the valid returns of each asset before the observation.

# Returns

  - `dvalid::AbstractVector{<:Bool}`: The mask.

# Related

  - [`centring_location!`](@ref)
"""
function centring_deviation_mask(::Union{PreCentred, ZeroStartCentring},
                                 valid::AbstractVector{<:Bool}, ::AbstractVector{<:Integer})
    return valid
end
function centring_deviation_mask(::EstimatedCentring, valid::AbstractVector{<:Bool},
                                 obs_count::AbstractVector{<:Integer})
    return valid .& (obs_count .> 0)
end
"""
    centring_location!(::PreCentred, location::VecNum, obs_count::AbstractVector{<:Integer},
                       X::VecNum, valid::AbstractVector{<:Bool}, decay::Number)
    centring_location!(::EstimatedCentring, location::VecNum,
                       obs_count::AbstractVector{<:Integer}, X::VecNum,
                       valid::AbstractVector{<:Bool}, decay::Number)

Takes the deviations of one observation, and moves the location of each valid asset.

Under [`PreCentred`](@ref) the deviation is the return, every valid asset gives a term, and the location is not written. Under [`EstimatedCentring`](@ref) the deviation is the return less the location that stands before it, and an asset gives a term only when it has a location, that is, one valid return before. The location of each valid asset then takes one step of the normalised mean.

# Algorithm

 1. Under `PreCentred`, return `(X, valid)`.
 2. Take `dev = X - location`, and `dvalid` as the valid assets whose count is positive.
 3. For each valid asset, take `n = obs_count + 1`, the count after this return, and move its location to `location + (1 - decay) / (1 - decay^n) * (X - location)`. At `n = 1` this is `X`.
 4. Return `(dev, dvalid)`.

# Arguments

  - `location`: The location of each asset before the observation. The method writes it in place. An asset with no valid return holds any value, and the step does not read it.
  - `obs_count`: Count of the valid returns of each asset before the observation.
  - `X`: Returns of the observation.
  - `valid`: Mask of the valid assets.
  - `decay`: Decay of the location.

# Returns

  - `(dev, dvalid)::Tuple`: The deviations, and the mask of the assets that give a term.

# Related

  - [`EstimatedCentring`](@ref)
  - [`centring_factor`](@ref)
"""
function centring_location!(::PreCentred, ::VecNum, ::AbstractVector{<:Integer}, X::VecNum,
                            valid::AbstractVector{<:Bool}, ::Number)
    return X, valid
end
function centring_location!(::ZeroStartCentring, location::VecNum,
                            ::AbstractVector{<:Integer}, X::VecNum,
                            valid::AbstractVector{<:Bool}, decay::Number)
    loc = replace(location, NaN => zero(eltype(location)))
    location[valid] = decay * view(loc, valid) + (one(decay) - decay) * view(X, valid)
    return X - loc, valid
end
function centring_location!(c::EstimatedCentring, location::VecNum,
                            obs_count::AbstractVector{<:Integer}, X::VecNum,
                            valid::AbstractVector{<:Bool}, decay::Number)
    dev = X .- location
    dvalid = centring_deviation_mask(c, valid, obs_count)
    idx = findall(valid)
    step = (one(decay) - decay) ./ (one(decay) .- decay .^ (view(obs_count, idx) .+ 1))
    location[idx] .= ifelse.(iszero.(view(obs_count, idx)), view(X, idx),
                             view(location, idx) .+ step .* view(dev, idx))
    return dev, dvalid
end
"""
    centring_report_location(::PreCentred, location::VecNum)
    centring_report_location(::EstimatedCentring, location::VecNum)

The location that an estimator centres on, as a caller reads it.

A pre-centred estimator centres on zero and never writes its location, so its state keeps the cold value. An estimated location is reported as it stands, and an asset with no valid return is `NaN`.

# Returns

  - `mu::VecNum`: Zero for each asset under [`PreCentred`](@ref), the location under [`EstimatedCentring`](@ref).

# Related

  - [`centring_location!`](@ref)
  - [`forecast_location`](@ref)
"""
function centring_report_location(::PreCentred, location::VecNum)
    return zero(location)
end
function centring_report_location(::Union{EstimatedCentring, ZeroStartCentring},
                                  location::VecNum)
    return location
end
"""
    centring_factor(::PreCentred, decay::Number, n::Integer)
    centring_factor(::EstimatedCentring, decay::Number, n::Integer)

Exact factor of the variance of a deviation, in units of the variance of the return.

Under [`EstimatedCentring`](@ref) the location of ``n`` returns has the variance ``\\sigma^2 / n^{\\mathrm{eff}}``, so the factor is ``1 + 1 / n^{\\mathrm{eff}}``, with ``n^{\\mathrm{eff}}`` the Kish count of [`exp_weighted_variance_count`](@ref). Under [`PreCentred`](@ref) the factor is one.

```math
\\begin{align}
1 + \\frac{1}{n^{\\mathrm{eff}}} &= 1 + \\frac{(1 - \\lambda)(1 + \\lambda^{n})}{(1 + \\lambda)(1 - \\lambda^{n})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_ew])
  - ``n``: Count of the returns that the location holds, at least one.

# Returns

  - `f::Number`: The factor, at least one.

# Related

  - [`EstimatedCentring`](@ref)
  - [`centring_overlap`](@ref)
"""
function centring_factor(::Union{PreCentred, ZeroStartCentring}, decay::Number, ::Integer)
    return one(decay)
end
function centring_factor(::EstimatedCentring, decay::Number, n::Integer)
    ln = decay^n
    return one(decay) +
           (one(decay) - decay) * (one(decay) + ln) /
           ((one(decay) + decay) * (one(decay) - ln))
end
"""
    centring_overlap(::PreCentred, ::Type{T}, N::Integer)
    centring_overlap(::EstimatedCentring, ::Type{T}, N::Integer)

Makes the cold state of the overlap of the locations of each pair of assets.

The overlap ``P_{ij}`` is the sum, over the common valid returns of the pair, of the products of the two unnormalised weights of the locations. The factor of a pair reads it, because the histories of two assets can differ. A pre-centred estimator takes no location, and holds no overlap.

# Returns

  - `P::Option{<:Matrix}`: A zero `N × N` matrix of `T` under [`EstimatedCentring`](@ref), `nothing` under [`PreCentred`](@ref).

# Related

  - [`centring_overlap!`](@ref)
  - [`centring_pair_factor`](@ref)
"""
function centring_overlap(::Union{PreCentred, ZeroStartCentring}, ::Type, ::Integer)
    return nothing
end
function centring_overlap(::EstimatedCentring, ::Type{T}, N::Integer) where {T}
    return zeros(T, N, N)
end
"""
    centring_overlap!(::Nothing, valid::AbstractVector{<:Bool}, decay::Number)
    centring_overlap!(P::MatNum, valid::AbstractVector{<:Bool}, decay::Number)

Folds one observation into the overlap of the locations.

A valid return of an asset ages each weight of its location by `decay`, and a return that both assets of a pair hold adds the product of its two new weights.

```math
\\begin{align}
P_{ij} &\\leftarrow \\lambda^{v_i + v_j} P_{ij} + v_i v_j (1 - \\lambda)^2\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_ew])
  - ``v_i``: One where asset ``i`` is valid at the observation, zero otherwise.

# Returns

  - `P`: The overlap after the observation, written in place, or `nothing`.

# Related

  - [`centring_overlap`](@ref)
"""
function centring_overlap!(::Nothing, ::AbstractVector{<:Bool}, ::Number)
    return nothing
end
function centring_overlap!(P::MatNum, valid::AbstractVector{<:Bool}, decay::Number)
    d = ifelse.(valid, decay, one(decay))
    v = ifelse.(valid, one(decay) - decay, zero(decay))
    P .= (d .* transpose(d)) .* P .+ v .* transpose(v)
    return P
end
"""
    centring_reset!(::Nothing, idx::AbstractVector{<:Bool})
    centring_reset!(P::MatNum, idx::AbstractVector{<:Bool})

Zeroes the rows and the columns of the overlap of the assets that reset.

# Returns

  - `P`: The overlap, written in place, or `nothing`.

# Related

  - [`centring_overlap`](@ref)
"""
function centring_reset!(::Nothing, ::AbstractVector{<:Bool})
    return nothing
end
function centring_reset!(P::MatNum, idx::AbstractVector{<:Bool})
    P[idx, :] .= zero(eltype(P))
    P[:, idx] .= zero(eltype(P))
    return P
end
"""
    centring_pair_factor(::Nothing, obs_count::AbstractVector{<:Integer}, decay::Number,
                         idx::AbstractVector{<:Integer})
    centring_pair_factor(P::MatNum, obs_count::AbstractVector{<:Integer}, decay::Number,
                         idx::AbstractVector{<:Integer})

Exact factor of the mean of the product of the deviations of each pair of a block, in units of the covariance of the pair.

The factor is ``1 + c_{ij}``, with ``c_{ij} = P_{ij} / (S_{1,i}\\, S_{1,j})`` and ``S_{1,i} = 1 - \\lambda^{n_i}``, the sum of the weights of the location of asset ``i``. A pair whose asset holds no return has no factor, and takes one. A pre-centred estimator holds no overlap, and its factor is one.

# Arguments

  - `P`: The overlap of the locations before the observation, or `nothing`.
  - `obs_count`: Count of the valid returns of each asset before the observation.
  - `decay`: Decay of the locations.
  - `idx`: Index of the assets of the block.

# Returns

  - `F::Union{<:Number, <:MatNum}`: The matrix of the factors of the block, or one.

# Related

  - [`centring_factor`](@ref)
  - [`centring_overlap!`](@ref)
"""
function centring_pair_factor(::Nothing, ::AbstractVector{<:Integer}, decay::Number,
                              ::AbstractVector{<:Integer})
    return one(decay)
end
function centring_pair_factor(P::MatNum, obs_count::AbstractVector{<:Integer},
                              decay::Number, idx::AbstractVector{<:Integer})
    s = one(decay) .- decay .^ view(obs_count, idx)
    ss = s .* transpose(s)
    return ifelse.(ss .> zero(eltype(ss)), one(eltype(P)) .+ view(P, idx, idx) ./ ss,
                   one(eltype(P)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Record of one lagged observation, which the factor of a HAC product reads.

The HAC term of an observation adds the products of its deviations with the deviations of the lagged observations. Under [`EstimatedCentring`](@ref) such a product has a non-zero mean, which reads the counts of the lagged observation, the assets valid there, and the overlap of the locations before it.

# Arguments

  - `obs_count`: Count of the valid returns of each asset before the lagged observation.
  - `valid`: Mask of the assets valid at the lagged observation.
  - `P`: The overlap of the locations before the lagged observation, or `nothing` where the estimator reads the diagonal alone.

# Returns

  - `(; obs_count, valid, P)::NamedTuple`: Fresh copies of the three arguments.

# Related

  - [`centring_lag_factor`](@ref)
"""
function centring_lag_record(obs_count::AbstractVector{<:Integer},
                             valid::AbstractVector{<:Bool}, P::Option{<:MatNum})
    return (; obs_count = copy(obs_count), valid = BitVector(valid),
            P = isnothing(P) ? nothing : copy(P))
end
"""
    centring_lag_records(::PreCentred, hac_lags, P)
    centring_lag_records(::EstimatedCentring, hac_lags::Nothing, P)
    centring_lag_records(::EstimatedCentring, hac_lags::Integer, P)

Makes the cold buffer of the lagged records that the factor of a HAC term reads.

Only an estimated location gives a lagged product a non-zero mean, and only a HAC adjustment reads lagged products, so the buffer exists under [`EstimatedCentring`](@ref) with `hac_lags` set, and holds `hac_lags` records.

# Arguments

  - `hac_lags`: Count of the lags of the HAC adjustment, or `nothing`.
  - `P`: The overlap state of the estimator, or `nothing`, which sets the type of a record.

# Returns

  - `records::Option{<:DataStructures.CircularBuffer}`: The empty buffer, or `nothing`.

# Related

  - [`centring_lag_record`](@ref)
  - [`centring_lag_factor`](@ref)
"""
function centring_lag_records(::Union{PreCentred, ZeroStartCentring}, ::Any, ::Any)
    return nothing
end
function centring_lag_records(::EstimatedCentring, ::Nothing, ::Any)
    return nothing
end
function centring_lag_records(::EstimatedCentring, hac_lags::Integer, P::Option{<:MatNum})
    R = typeof(centring_lag_record(Int[], falses(0), P))
    return DataStructures.CircularBuffer{R}(hac_lags)
end
"""
    centring_lag_push!(::Nothing, obs_count::AbstractVector{<:Integer},
                       valid::AbstractVector{<:Bool}, P::Option{<:MatNum})
    centring_lag_push!(records::DataStructures.CircularBuffer,
                       obs_count::AbstractVector{<:Integer}, valid::AbstractVector{<:Bool},
                       P::Option{<:MatNum})

Pushes the record of one observation into the buffer of lagged records.

# Arguments

  - `obs_count`: Count of the valid returns of each asset before the observation.
  - `valid`: Mask of the assets valid at the observation.
  - `P`: The overlap of the locations before the observation, or `nothing`.

# Returns

  - `records`: The buffer, or `nothing`.

# Related

  - [`centring_lag_record`](@ref)
"""
function centring_lag_push!(::Nothing, ::AbstractVector{<:Integer},
                            ::AbstractVector{<:Bool}, ::Option{<:MatNum})
    return nothing
end
function centring_lag_push!(records::DataStructures.CircularBuffer,
                            obs_count::AbstractVector{<:Integer},
                            valid::AbstractVector{<:Bool}, P::Option{<:MatNum})
    return push!(records, centring_lag_record(obs_count, valid, P))
end
"""
    hac_buffer_reset!(::Nothing, idx::AbstractVector{<:Bool})
    hac_buffer_reset!(buffer::DataStructures.CircularBuffer, idx::AbstractVector{<:Bool})

Blanks the lagged deviations of the assets that reset.

A reset discards the history of an asset, so a deviation from before it is no term of a later HAC product. The method sets the entries of each such asset to `NaN` in every lagged observation, which the HAC term reads as zero.

# Returns

  - `buffer`: The buffer, written in place, or `nothing`.

# Related

  - [`centring_lag_reset!`](@ref)
"""
function hac_buffer_reset!(::Nothing, ::AbstractVector{<:Bool})
    return nothing
end
function hac_buffer_reset!(buffer::DataStructures.CircularBuffer,
                           idx::AbstractVector{<:Bool})
    for row in buffer
        row[idx] .= convert(eltype(row), NaN)
    end
    return buffer
end
"""
    centring_lag_reset!(::Nothing, idx::AbstractVector{<:Bool})
    centring_lag_reset!(records, idx::AbstractVector{<:Bool})

Marks the assets that reset as absent from every lagged record.

A reset discards the history of an asset, so its location no longer holds a lagged return, and a deviation from before the reset is no term of a later HAC product. The method sets the count of each such asset to `-1` in every record.

# Returns

  - `records`: The records, written in place, or `nothing`.

# Related

  - [`centring_lag_record`](@ref)
  - [`centring_lag_factor`](@ref)
"""
function centring_lag_reset!(::Nothing, ::AbstractVector{<:Bool})
    return nothing
end
function centring_lag_reset!(records::DataStructures.CircularBuffer,
                             idx::AbstractVector{<:Bool})
    for r in records
        r.obs_count[idx] .= -1
    end
    return records
end
"""
    centring_lag_factor(::PreCentred, decay::Number, n::AbstractVector{<:Integer},
                        dvalid::AbstractVector{<:Bool}, records, w::Function, diagonal::Bool)
    centring_lag_factor(::EstimatedCentring, decay::Number, n::AbstractVector{<:Integer},
                        dvalid::AbstractVector{<:Bool}, records, w::Function, diagonal::Bool)

The part of the exact factor of a HAC term that its lagged products give, in units of the covariance of the pair.

For returns that are independent in time, a pre-centred lagged product has mean zero, so the part is zero under [`PreCentred`](@ref). Under [`EstimatedCentring`](@ref) the location of the observation ``t`` holds the return of the lagged observation ``r``, and both locations hold the earlier returns, so

```math
\\begin{align}
\\frac{\\mathbb{E}\\left[e_{i,t}\\, e_{j,r}\\right]}{\\sigma_{ij}} &= -v_{i,r} \\frac{(1 - \\lambda) \\lambda^{d_i - 1}}{S_{1}(n_{i,t})} + \\frac{\\lambda^{d_i} P_{ij}(r)}{S_{1}(n_{i,t})\\, S_{1}(n_{j,r})}\\,, \\\\
G_{ij} &= \\sum_{l = 1}^{L} \\omega_l \\left(\\frac{\\mathbb{E}\\left[e_{i,t}\\, e_{j,t-l}\\right] + \\mathbb{E}\\left[e_{j,t}\\, e_{i,t-l}\\right]}{\\sigma_{ij}}\\right)\\,.
\\end{align}
```

The exact factor of the HAC term is ``1 + c_{ij} + G_{ij}``. A product whose deviation does not exist is zero in the HAC term, and gives nothing to ``G``.

Where:

  - $(math_dict[:lambda_ew])
  - ``n_{i,t}``, ``n_{j,r}``: Count of the valid returns of the asset before the observation.
  - ``d_i = n_{i,t} - n_{i,r}``: Count of the valid returns of asset ``i`` from ``r`` to ``t - 1``.
  - ``v_{i,r}``: One where asset ``i`` is valid at ``r``.
  - ``S_{1}(n) = 1 - \\lambda^{n}``: The sum of the weights of a location of ``n`` returns.
  - ``P_{ij}(r)``: The overlap of the locations before ``r``. On the diagonal it is ``S_{2}(n) = (1 - \\lambda)^2 (1 - \\lambda^{2n}) / (1 - \\lambda^2)``.
  - ``\\omega_l``: The Bartlett weight of lag ``l``.

# Arguments

  - `decay`: Decay of the locations.
  - `n`: Count of the valid returns of each asset before the observation.
  - `dvalid`: Mask of the assets whose deviation at the observation exists.
  - `records`: The lagged records of [`centring_lag_record`](@ref), oldest first.
  - `w`: The Bartlett weight of a lag, as a function of the lag.
  - `diagonal`: Whether to give the diagonal alone, as a vector. Otherwise the method gives the `N × N` matrix and reads the overlap of each record.

# Returns

  - `G::Union{<:Number, <:VecNum, <:MatNum}`: Zero under `PreCentred`, otherwise the vector or the matrix of ``G``.

# Related

  - [`centring_lag_record`](@ref)
  - [`centring_pair_factor`](@ref)
"""
function centring_lag_factor(::Union{PreCentred, ZeroStartCentring}, decay::Number,
                             ::AbstractVector{<:Integer}, ::AbstractVector{<:Bool}, ::Any,
                             ::Function, ::Bool)
    return zero(decay)
end
function centring_lag_factor(::EstimatedCentring, decay::Number,
                             n::AbstractVector{<:Integer}, dvalid::AbstractVector{<:Bool},
                             records::DataStructures.CircularBuffer, w::Function,
                             diagonal::Bool)
    T = typeof(one(decay) - decay)
    N = length(n)
    G = diagonal ? zeros(T, N) : zeros(T, N, N)
    s1t = one(decay) .- decay .^ n
    for (l, r) in enumerate(Iterators.reverse(records))
        wl = w(l)
        nr = r.obs_count
        # Asset `i` keeps the lagged observation in its location where it has not reset since.
        life = dvalid .& (nr .>= 0)
        d = n .- nr
        own = ifelse.(life .& r.valid, (one(decay) - decay) .* decay .^ (d .- 1) ./ s1t,
                      zero(T))
        hold = ifelse.(life, decay .^ d ./ s1t, zero(T))
        # The deviation of asset `j` at the lagged observation exists.
        jdev = r.valid .& (nr .>= 1)
        s1r = ifelse.(jdev, one(decay) .- decay .^ nr, one(T))
        if diagonal
            s2r = (one(decay) - decay)^2 .* (one(decay) .- decay .^ (2 .* nr)) ./
                  (one(decay) - decay^2)
            g = ifelse.(jdev, hold .* s2r ./ s1r .- own, zero(T))
            G .+= 2 * wl .* g
        else
            g = ifelse.(transpose(jdev), hold .* r.P ./ transpose(s1r) .- own, zero(T))
            G .+= wl .* (g .+ transpose(g))
        end
    end
    return G
end
export EstimatedCentring, PreCentred, ZeroStartCentring
public centring_lag, centring_deviation_mask, centring_location!, centring_report_location,
       centring_factor, centring_overlap, centring_lag_records, centring_lag_factor
