@concrete struct ConditionalValueatRisk <: RiskMeasure
    settings
    alpha
    w
    function ConditionalValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                                    w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing)
    return ConditionalValueatRisk(settings, alpha, w)
end
@concrete struct DistributionallyRobustConditionalValueatRisk <: RiskMeasure
    settings
    alpha
    l
    r
    w
    function DistributionallyRobustConditionalValueatRisk(settings::RiskMeasureSettings,
                                                          alpha::Number, l::Number,
                                                          r::Number,
                                                          w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(l > zero(l))
        @argcheck(r > zero(r))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(l), typeof(r), typeof(w)}(settings,
                                                                                     alpha,
                                                                                     l, r,
                                                                                     w)
    end
end
function DistributionallyRobustConditionalValueatRisk(;
                                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                      alpha::Number = 0.05, l::Number = 1.0,
                                                      r::Number = 0.02,
                                                      w::Option{<:ObsWeights} = nothing)
    return DistributionallyRobustConditionalValueatRisk(settings, alpha, l, r, w)
end
const RMCVaR{T} = Union{<:ConditionalValueatRisk{<:Any, <:Any, T},
                        <:DistributionallyRobustConditionalValueatRisk{<:Any, <:Any, <:Any,
                                                                       <:Any, T}}
function (r::RMCVaR{Nothing})(x::VecNum)
    x = copy(x)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end
function (r::RMCVaR{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
@concrete struct ConditionalValueatRiskRange <: RiskMeasure
    settings
    alpha
    beta
    w
    function ConditionalValueatRiskRange(settings::RiskMeasureSettings, alpha::Number,
                                         beta::Number, w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                             alpha, beta, w)
    end
end
function ConditionalValueatRiskRange(;
                                     settings::RiskMeasureSettings = RiskMeasureSettings(),
                                     alpha::Number = 0.05, beta::Number = 0.05,
                                     w::Option{<:ObsWeights} = nothing)
    return ConditionalValueatRiskRange(settings, alpha, beta, w)
end
function factory(r::ConditionalValueatRiskRange, pr::AbstractPriorResult, args...;
                 kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return ConditionalValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                       beta = r.beta, w = w)
end
@concrete struct DistributionallyRobustConditionalValueatRiskRange <: RiskMeasure
    settings
    alpha
    l_a
    r_a
    beta
    l_b
    r_b
    w
    function DistributionallyRobustConditionalValueatRiskRange(settings::RiskMeasureSettings,
                                                               alpha::Number, l_a::Number,
                                                               r_a::Number, beta::Number,
                                                               l_b::Number, r_b::Number,
                                                               w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        @argcheck(l_a > zero(l_a))
        @argcheck(r_a > zero(r_a))
        @argcheck(l_b > zero(l_b))
        @argcheck(r_b > zero(r_b))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(l_a), typeof(r_a), typeof(beta),
                   typeof(l_b), typeof(r_b), typeof(w)}(settings, alpha, l_a, r_a, beta,
                                                        l_b, r_b, w)
    end
end
function DistributionallyRobustConditionalValueatRiskRange(;
                                                           settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                           alpha::Number = 0.05,
                                                           l_a::Number = 1.0,
                                                           r_a::Number = 0.02,
                                                           beta::Number = 0.05,
                                                           l_b::Number = 1.0,
                                                           r_b::Number = 0.02,
                                                           w::Option{<:ObsWeights} = nothing)
    return DistributionallyRobustConditionalValueatRiskRange(settings, alpha, l_a, r_a,
                                                             beta, l_b, r_b, w)
end
function factory(r::DistributionallyRobustConditionalValueatRiskRange,
                 pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return DistributionallyRobustConditionalValueatRiskRange(; settings = r.settings,
                                                             alpha = r.alpha, l_a = r.l_a,
                                                             r_a = r.r_a, beta = r.beta,
                                                             l_b = r.l_b, r_b = r.r_b,
                                                             w = w)
end
const RMCVaRRg{T} = Union{<:ConditionalValueatRiskRange{<:Any, <:Any, <:Any, T},
                          <:DistributionallyRobustConditionalValueatRiskRange{<:Any, <:Any,
                                                                              <:Any, <:Any,
                                                                              <:Any, <:Any,
                                                                              <:Any, T}}
function (r::RMCVaRRg{Nothing})(x::VecNum)
    x = copy(x)
    alpha = r.alpha
    aT = alpha * length(x)
    idx1 = ceil(Int, aT)
    var1 = -partialsort!(x, idx1)
    sum_var1 = zero(eltype(x))
    for i in 1:(idx1 - 1)
        sum_var1 += x[i] + var1
    end
    loss = var1 - sum_var1 / aT

    beta = r.beta
    bT = beta * length(x)
    idx2 = ceil(Int, bT)
    var2 = -partialsort!(x, idx2; rev = true)
    sum_var2 = zero(eltype(x))
    for i in 1:(idx2 - 1)
        sum_var2 += x[i] + var2
    end
    gain = var2 - sum_var2 / bT
    return loss - gain
end
function (r::RMCVaRRg{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    loss = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / (alpha)
    end

    sorted_x = reverse!(sorted_x)
    sorted_w = reverse!(sorted_w)
    cum_w = cumsum(sorted_w)
    beta = sw * r.beta
    idx = searchsortedfirst(cum_w, beta)
    gain = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (beta - cum_w[idx - 1])) / (beta)
    end
    return loss - gain
end
@concrete struct ConditionalDrawdownatRisk <: RiskMeasure
    settings
    alpha
    w
    function ConditionalDrawdownatRisk(settings::RiskMeasureSettings, alpha::Number,
                                       w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing)
    return ConditionalDrawdownatRisk(settings, alpha, w)
end
@concrete struct DistributionallyRobustConditionalDrawdownatRisk <: RiskMeasure
    settings
    alpha
    l
    r
    w
    function DistributionallyRobustConditionalDrawdownatRisk(settings::RiskMeasureSettings,
                                                             alpha::Number, l::Number,
                                                             r::Number,
                                                             w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(l > zero(l))
        @argcheck(r > zero(r))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(l), typeof(r), typeof(w)}(settings,
                                                                                     alpha,
                                                                                     l, r,
                                                                                     w)
    end
end
function DistributionallyRobustConditionalDrawdownatRisk(;
                                                         settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                         alpha::Number = 0.05,
                                                         l::Number = 1.0, r::Number = 0.02,
                                                         w::Option{<:ObsWeights} = nothing)
    return DistributionallyRobustConditionalDrawdownatRisk(settings, alpha, l, r, w)
end
const RMCDaR{T} = Union{<:ConditionalDrawdownatRisk{<:Any, <:Any, <:T},
                        <:DistributionallyRobustConditionalDrawdownatRisk{<:Any, <:Any,
                                                                          <:Any, <:Any,
                                                                          <:T}}
function (r::RMCDaR{Nothing})(x::VecNum)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    dd = absolute_drawdown_vec(x)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end
function (r::RMCDaR{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    dd = absolute_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_dd[1]
    else
        idx = ifelse(idx > length(dd), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_dd[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_dd[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
@concrete struct RelativeConditionalDrawdownatRisk <: HierarchicalRiskMeasure
    settings
    alpha
    w
    function RelativeConditionalDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                               alpha::Number, w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeConditionalDrawdownatRisk(;
                                           settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                           alpha::Number = 0.05,
                                           w::Option{<:ObsWeights} = nothing)
    return RelativeConditionalDrawdownatRisk(settings, alpha, w)
end
function (r::RelativeConditionalDrawdownatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    dd = relative_drawdown_vec(x)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end
function (r::RelativeConditionalDrawdownatRisk{<:Any, <:Any, <:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    dd = relative_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_dd[1]
    else
        idx = ifelse(idx > length(dd), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_dd[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_dd[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
for r in (ConditionalValueatRisk, ConditionalDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 return $(r)(; settings = r.settings, alpha = r.alpha, w = w)
             end
         end)
end
for r in (DistributionallyRobustConditionalValueatRisk,
          DistributionallyRobustConditionalDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 return $(r)(; settings = r.settings, alpha = r.alpha, l = r.l, r = r.r,
                             w = w)
             end
         end)
end

export ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
       ConditionalValueatRiskRange, DistributionallyRobustConditionalValueatRiskRange,
       ConditionalDrawdownatRisk, DistributionallyRobustConditionalDrawdownatRisk,
       RelativeConditionalDrawdownatRisk
