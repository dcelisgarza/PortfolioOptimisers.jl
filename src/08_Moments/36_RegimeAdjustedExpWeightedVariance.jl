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
    function RegimeAdjustedExpWeightedVariance(decay::Number, min_obs::Integer,
                                               hac_lags::Option{<:Integer},
                                               regime_method::RegimeAdjustedMethod,
                                               regime_decay::Number,
                                               regime_min_obs::Integer,
                                               regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                                <:Number}},
                                               min_val::Number)
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
                   typeof(min_val)}(decay, min_obs, hac_lags, regime_method, regime_decay,
                                    regime_min_obs, regime_lohi_mult, min_val)
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
                                           min_val::Number = sqrt(eps()))
    return RegimeAdjustedExpWeightedVariance(decay, min_obs, hac_lags, regime_method,
                                             regime_decay, regime_min_obs, regime_lohi_mult,
                                             min_val)
end
@concrete struct RegimeAdjustedVarianceCache <: AbstractResult
    variance
    active
    obs_count
    kappa
    e_abs_z
    regime_state
    n_regime_obs
end
function process_observation!(cache::RegimeAdjustedVarianceCache,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              estimation_mask::Union{<:Colon, <:AbstractVector{<:Bool}},
                              active_mask::Union{<:Colon, <:AbstractVector{<:Bool}})
    finite_mask = isfinite(X)
    valid = isa(active_mask, Colon) ? Colon() : (finite_mask .& active_mask)

    return nothing
end
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                        mean = nothing,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    est_flag = !isnothing(estimation_mask)
    act_flag = !isnothing(active_mask)
    itr, v = ifelse(isone(dims), (eachrow, (x, y) -> view(x, :, y)),
                    (eachcol, (x, y) -> view(x, y, :)))
    if est_flag
        @argcheck(size(X) == size(estimation_mask))
    else
        estimation_mask = Iterators.repeated(Colon(), size(X, dims))
    end
    if act_flag
        @argcheck(size(X) == size(active_mask))
        active_mask = Iterators.repeated(Colon(), size(X, dims))
    end
    N = size(X, setdiff((1, 2), (dims,))[1])
    cache = RegimeAdjustedVarianceCache(zeros(eltype(X), N), trues(N), zeros(eltype(X), N),
                                        SpecialFunctions.digamma(0.5) + log(2),
                                        sqrt(2 * inv(pi)), nothing, 0)

    return itr, estimation_mask, active_mask, cache
end

export LogRegimeAdjusted, FirstMomentRegimeAdjusted, RootMeanSquaredAdjusted,
       RegimeAdjustedExpWeightedVariance
