abstract type RegimeAdjustedMethod <: AbstractEstimator end
@concrete struct LogRegimeAdjusted <: RegimeAdjustedMethod
    val
    function LogRegimeAdjusted(val::Number)
        assert_nonempty_nonneg_finite_val(val, :val)
        return new{typeof(val)}(val)
    end
end
function LogRegimeAdjusted(; val::Number = 0.5)
    return LogRegimeAdjusted(val)
end
struct FirstMomentRegimeAdjusted <: RegimeAdjustedMethod end
struct RootMeanSquaredAdjusted <: RegimeAdjustedMethod end
function regime_multiplier(method::LogRegimeAdjusted, regime_state::Number)
    return exp(method.val * regime_state)
end
function regime_multiplier(::FirstMomentRegimeAdjusted, regime_state::Number)
    return regime_state
end
function regime_multiplier(::RootMeanSquaredAdjusted, regime_state::Number)
    return sqrt(max(regime_state, zero(regime_state)))
end
@concrete struct RegimeAdjustedExpWeightedVariance <: AbstractCovarianceEstimator
    decay
    min_obs
    hac_lags
    regime_method
    regime_decay
    regime_min_obs
    regime_lohi_mult
    min_val
    centred
    function RegimeAdjustedExpWeightedVariance(decay::Number, min_obs::Integer,
                                               hac_lags::Option{<:Integer},
                                               regime_method::RegimeAdjustedMethod,
                                               regime_decay::Number,
                                               regime_min_obs::Integer,
                                               regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                                <:Number}},
                                               min_val::Number, centred::Bool)
        assert_nonempty_gt0_finite_val(decay, :decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_nonempty_gt0_finite_val(regime_min_obs, :regime_min_obs)
        if !isnothing(regime_lohi_mult)
            @argcheck(zero(regime_lohi_mult[1]) < regime_lohi_mult[1] < regime_lohi_mult[2],
                      DomainError)
        end
        if !isnothing(hac_lags)
            assert_nonempty_gt0_finite_val(hac_lags, :hac_lags)
        end
        return new{typeof(decay), typeof(min_obs), typeof(hac_lags), typeof(regime_method),
                   typeof(regime_decay), typeof(regime_min_obs), typeof(regime_lohi_mult),
                   typeof(min_val), typeof(centred)}(decay, min_obs, hac_lags,
                                                     regime_method, regime_decay,
                                                     regime_min_obs, regime_lohi_mult,
                                                     min_val, centred)
    end
end
function RegimeAdjustedExpWeightedVariance(; decay::Number = exp2(-inv(40.0)),
                                           min_obs::Integer = round(Int,
                                                                    max(1,
                                                                        inv(log2(inv(decay))))),
                                           hac_lags::Option{<:Integer} = nothing,
                                           regime_method::RegimeAdjustedMethod = FirstMomentRegimeAdjusted(),
                                           regime_decay::Number = exp2(-2 /
                                                                       inv(log2(inv(decay)))),
                                           regime_min_obs::Integer = round(Int,
                                                                           max(1,
                                                                               inv(log2(inv(decay))) /
                                                                               2)),
                                           regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                            <:Number}} = (0.7,
                                                                                          1.6),
                                           min_val::Number = sqrt(eps()),
                                           centred::Bool = false)
    return RegimeAdjustedExpWeightedVariance(decay, min_obs, hac_lags, regime_method,
                                             regime_decay, regime_min_obs, regime_lohi_mult,
                                             min_val, centred)
end
@concrete struct RegimeAdjustedVarianceCache <: AbstractResult
    ret_buffer
    variance
    z2
    location
    obs_count
    old_obs_count
    active
    kappa
    e_abs_z
    regime_state
    n_regime_obs
end
function hac_squared_returns(cache::RegimeAdjustedVarianceCache,
                             ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                             finite_mask::AbstractVector{<:Bool})
    X2 = X .^ 2
    if !isnothing(cache.ret_buffer)
        return X2
    end

    for (i, X_old) in enumerate(reverse(cache.ret_buffer))
    end
    return nothing
end
function process_observation!(cache::RegimeAdjustedVarianceCache,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              estimation_mask::Option{<:AbstractVector{<:Bool}},
                              active_mask::Option{<:AbstractVector{<:Bool}})
    finite_mask = isfinite(X)
    valid = isnothing(active_mask) ? finite_mask : (finite_mask .& active_mask)

    if !isnothing(active_mask)
        newly_inactive = .!active_mask .& cache.active
        if any(newly_inactive)
            cache.variance[newly_inactive] .= zero(eltype(cache.variance))
            cache.obs_count[newly_inactive] .= 0
            if !ce.centred
                cache.location[newly_inactive] .= NaN
            end
        end
        cache.active .= active_mask
    else
        cache.active .= true
    end

    if !any(valid)
        return nothing
    end

    copyto!(cache.old_obs_count, cache.obs_count)

    Xi = if ce.centred
        X
    else
        loc = replace(cache.location, NaN => zero(eltype(cache.location)))
        cache.location[valid] = ce.decay * view(loc, valid) +
                                (one(eltype(cache.location)) - ce.decay) * view(X, valid)
        X - loc
    end

    ready = valid .& (cache.old_obs_count .>= ce.min_obs)
    fill!(cache.z2, NaN)
    var_idx = ready .& (cache.variance .>= ce.min_val)
    if any(var_idx)
        factor = inv.(max.(one(ce.decay) .- ce.decay .^ cache.old_obs_count[var_idx],
                           eps(ce.decay)))
        var_corrected = view(cache.variance, var_idx) * factor
        cache.z2[var_idx] = view(Xi, var_idx) .^ 2 / var_corrected
    end

    X2 = hac_squared_returns(cache, ce, Xi, valid)
    return nothing
end
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                        mean = nothing,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    est_flag = !isnothing(estimation_mask)
    act_flag = !isnothing(active_mask)
    itr, v = ifelse(isone(dims), (eachrow, (x, y) -> view(x, y, :)),
                    (eachcol, (x, y) -> view(x, :, y)))
    if est_flag
        @argcheck(size(X) == size(estimation_mask))
    end
    if act_flag
        @argcheck(size(X) == size(active_mask))
    end
    N = size(X, setdiff((1, 2), (dims,))[1])

    cache = RegimeAdjustedVarianceCache(if isnothing(ce.hac_lags)
                                            nothing
                                        else
                                            DataStructures.Deque{Vector{eltype(X)}}(ce.hac_lags)
                                        end, zeros(eltype(X), N), fill(NaN, N),
                                        ce.centred ? zeros(eltype(X), N) : fill(NaN, N),
                                        zeros(Int, N), zeros(Int, N), trues(N),
                                        SpecialFunctions.digamma(0.5) + log(2),
                                        sqrt(2 * inv(pi)), nothing, 0)
    for (i, Xi) in enumerate(itr(X))
        emi = est_flag ? v(estimation_mask, i) : nothing
        ami = act_flag ? v(active_mask, i) : nothing
        process_observation!(cache, ce, X, emi, ami)
    end

    return itr, estimation_mask, active_mask, cache
end

export LogRegimeAdjusted, FirstMomentRegimeAdjusted, RootMeanSquaredAdjusted,
       RegimeAdjustedExpWeightedVariance
