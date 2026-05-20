abstract type RegimeAdjustedMethod <: AbstractEstimator end
@concrete struct LogRegimeAdjusted <: RegimeAdjustedMethod
    x
    y
    kappa
    function LogRegimeAdjusted(x::Number, y::Number)
        assert_nonempty_nonneg_finite_val(x, :x)
        assert_nonempty_nonneg_finite_val(y, :y)
        kappa = SpecialFunctions.digamma(x) + log(y)
        return new{typeof(x), typeof(y), typeof(kappa)}(x, y, kappa)
    end
end
function LogRegimeAdjusted(; x::Number = 0.5, y::Number = 2.0)::LogRegimeAdjusted
    return LogRegimeAdjusted(x, y)
end
@concrete struct FirstMomentRegimeAdjusted <: RegimeAdjustedMethod
    x
    function FirstMomentRegimeAdjusted(x::Number)
        assert_nonempty_nonneg_finite_val(x, :x)
        return new{typeof(x)}(x)
    end
end
function FirstMomentRegimeAdjusted(;
                                   x::Number = sqrt(2 * inv(pi)))::FirstMomentRegimeAdjusted
    return FirstMomentRegimeAdjusted(x)
end
struct RootMeanSquaredAdjusted <: RegimeAdjustedMethod end
function regime_multiplier(method::LogRegimeAdjusted, regime_state::Number)
    return exp(method.x * regime_state)
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
                                           centred::Bool = false)::RegimeAdjustedExpWeightedVariance
    return RegimeAdjustedExpWeightedVariance(decay, min_obs, hac_lags, regime_method,
                                             regime_decay, regime_min_obs, regime_lohi_mult,
                                             min_val, centred)
end
@concrete struct RegimeAdjustedVarianceCache <: AbstractResult
    ret_buffer
    variance
    X2
    X_old_i
    z2
    location
    obs_count
    old_obs_count
    active
    regime_state
    n_regime_obs
end
function get_regime_state(::RootMeanSquaredAdjusted, z2_valid::VecNum, ::Any)
    return Statistics.mean(z2_valid)
end
function get_regime_state(method::FirstMomentRegimeAdjusted, z2_valid::VecNum, ::Any)
    return Statistics.mean(sqrt.(max.(z2_valid, zero(eltype(z2_valid))))) / method.x
end
function get_regime_state(method::LogRegimeAdjusted, z2_valid::VecNum,
                          min_val::Number = sqrt(eps(eltype(z2_valid))))
    log_z2 = log.(max.(z2_valid, min_val))
    return Statistics.mean(log_z2) - method.kappa
end
function hac_squared_returns!(cache::RegimeAdjustedVarianceCache,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              finite_mask::AbstractVector{<:Bool})
    copyto!(cache.X2, X .^ 2)
    if isnothing(cache.ret_buffer) || isempty(cache.ret_buffer)
        return cache.X2
    end

    for (i, X_old) in enumerate(Iterators.reverse(cache.ret_buffer))
        wi = one(eltype(X)) - i / (ce.hac_lags + 1)
        cache.X_old_i .= replace(X_old, NaN => zero(eltype(X_old)))
        cache.X2 .+= 2 * wi * X .* cache.X_old_i
    end
    cache.X2[finite_mask] .= max.(view(cache.X2, finite_mask), zero(eltype(cache.X2)))

    return cache.X2
end
function process_observation!(cache::RegimeAdjustedVarianceCache,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              estimation_mask::Option{<:AbstractVector{<:Bool}},
                              active_mask::Option{<:AbstractVector{<:Bool}})
    finite_mask = isfinite.(X)
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

    regime_mask = valid .& (cache.old_obs_count .>= ce.min_obs)
    fill!(cache.z2, NaN)
    var_idx = regime_mask .& (cache.variance .>= ce.min_val)
    if any(var_idx)
        factor = inv.(max.(one(ce.decay) .- ce.decay .^ cache.old_obs_count[var_idx],
                           eps(ce.decay)))
        var_corrected = view(cache.variance, var_idx) .* factor
        cache.z2[var_idx] = view(Xi, var_idx) .^ 2 ./ var_corrected
    end

    X2 = hac_squared_returns!(cache, ce, Xi, valid)
    cache.variance[valid] .= ce.decay * view(cache.variance, valid) +
                             (one(ce.decay) - ce.decay) * view(X2, valid)
    cache.obs_count[valid] .+= 1

    if !isnothing(cache.ret_buffer)
        X_new = copy(Xi)
        X_new[.!valid] .= NaN
        push!(cache.ret_buffer, X_new)
    end

    if !isnothing(estimation_mask)
        regime_mask .&= estimation_mask
    end

    if !any(regime_mask)
        return nothing
    end

    z2_valid = filter(!isnan, view(cache.z2, regime_mask))
    if isempty(z2_valid)
        return nothing
    end

    regime_state = get_regime_state(ce.regime_method, z2_valid, ce.min_val)

    Accessors.@reset cache.regime_state = if isnothing(cache.regime_state)
        regime_state
    else
        ce.regime_decay * cache.regime_state +
        (one(eltype(ce.regime_decay)) - ce.regime_decay) * regime_state
    end

    Accessors.@reset cache.n_regime_obs += 1

    return nothing
end
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
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
                                            DataStructures.CircularBuffer{Vector{eltype(X)}}(ce.hac_lags)
                                        end, zeros(eltype(X), N), zeros(eltype(X), N),
                                        zeros(eltype(X), N), fill(NaN, N),
                                        ce.centred ? zeros(eltype(X), N) : fill(NaN, N),
                                        zeros(Int, N), zeros(Int, N), trues(N), nothing,
                                        zero(eltype(X)))
    for (i, Xi) in enumerate(itr(X))
        emi = est_flag ? v(estimation_mask, i) : nothing
        ami = act_flag ? v(active_mask, i) : nothing
        process_observation!(cache, ce, Xi, emi, ami)
    end

    variance = copy(cache.variance)
    correction = ones(eltype(variance), length(variance))
    correction[cache.obs_count .> zero(eltype(cache.obs_count))] .= inv.(max.(one(ce.decay) .-
                                                                              ce.decay .^
                                                                              cache.obs_count,
                                                                              eps(eltype(variance))))
    variance .*= correction
    not_ready = .!cache.active .| (cache.obs_count .< ce.min_obs)

    if any(not_ready)
        variance[not_ready] .= NaN
    end

    if !ce.centred && any(.!cache.active)
        cache.location[.!cache.active] .= NaN
    end

    factor = if cache.n_regime_obs < ce.regime_min_obs
        one(eltype(X))
    else
        regime_multiplier(ce.regime_method, cache.regime_state)
    end

    return variance * factor^2
end

export LogRegimeAdjusted, FirstMomentRegimeAdjusted, RootMeanSquaredAdjusted,
       RegimeAdjustedExpWeightedVariance
