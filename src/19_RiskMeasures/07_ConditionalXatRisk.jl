struct ConditionalValueatRisk{T1, T2, T3} <: RiskMeasure
    settings::T1
    alpha::T2
    w::T3
    function ConditionalValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                                    w::Option{<:AbstractWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Number = 0.05,
                                w::Option{<:AbstractWeights} = nothing)
    return ConditionalValueatRisk(settings, alpha, w)
end
function factory(r::ConditionalValueatRisk, prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return ConditionalValueatRisk(; settings = r.settings, alpha = r.alpha, w = w)
end
struct DistributionallyRobustConditionalValueatRisk{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    alpha::T2
    l::T3
    r::T4
    w::T5
    function DistributionallyRobustConditionalValueatRisk(settings::RiskMeasureSettings,
                                                          alpha::Number, l::Number,
                                                          r::Number,
                                                          w::Option{<:AbstractWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
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
                                                      w::Option{<:AbstractWeights} = nothing)
    return DistributionallyRobustConditionalValueatRisk(settings, alpha, l, r, w)
end
function factory(r::DistributionallyRobustConditionalValueatRisk,
                 prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return DistributionallyRobustConditionalValueatRisk(; settings = r.settings,
                                                        alpha = r.alpha, l = r.l, r = r.r,
                                                        w = w)
end
function (r::Union{<:ConditionalValueatRisk{<:Any, <:Any, Nothing},
                   <:DistributionallyRobustConditionalValueatRisk{<:Any, <:Any, <:Any,
                                                                  <:Any, Nothing}})(x::VecNum)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end
function (r::Union{<:ConditionalValueatRisk{<:Any, <:Any, <:AbstractWeights},
                   <:DistributionallyRobustConditionalValueatRisk{<:Any, <:Any, <:Any,
                                                                  <:Any, <:AbstractWeights}})(x::VecNum)
    sw = sum(r.w)
    order = sortperm(x)
    sorted_x = x[order]
    sorted_w = r.w[order]
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
struct ConditionalValueatRiskRange{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    alpha::T2
    beta::T3
    w::T4
    function ConditionalValueatRiskRange(settings::RiskMeasureSettings, alpha::Number,
                                         beta::Number, w::Option{<:AbstractWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                             alpha, beta, w)
    end
end
function ConditionalValueatRiskRange(;
                                     settings::RiskMeasureSettings = RiskMeasureSettings(),
                                     alpha::Number = 0.05, beta::Number = 0.05,
                                     w::Option{<:AbstractWeights} = nothing)
    return ConditionalValueatRiskRange(settings, alpha, beta, w)
end
function factory(r::ConditionalValueatRiskRange, prior::AbstractPriorResult, args...;
                 kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return ConditionalValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                       beta = r.beta, w = w)
end
struct DistributionallyRobustConditionalValueatRiskRange{T1, T2, T3, T4, T5, T6, T7, T8} <:
       RiskMeasure
    settings::T1
    alpha::T2
    l_a::T3
    r_a::T4
    beta::T5
    l_b::T6
    r_b::T7
    w::T8
    function DistributionallyRobustConditionalValueatRiskRange(settings::RiskMeasureSettings,
                                                               alpha::Number, l_a::Number,
                                                               r_a::Number, beta::Number,
                                                               l_b::Number, r_b::Number,
                                                               w::Option{<:AbstractWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
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
                                                           w::Option{<:AbstractWeights} = nothing)
    return DistributionallyRobustConditionalValueatRiskRange(settings, alpha, l_a, r_a,
                                                             beta, l_b, r_b, w)
end
function factory(r::DistributionallyRobustConditionalValueatRiskRange,
                 prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return DistributionallyRobustConditionalValueatRiskRange(; settings = r.settings,
                                                             alpha = r.alpha, l_a = r.l_a,
                                                             r_a = r.r_a, beta = r.beta,
                                                             l_b = r.l_b, r_b = r.r_b,
                                                             w = w)
end
function (r::Union{<:ConditionalValueatRiskRange{<:Any, <:Any, <:Any, Nothing},
                   <:DistributionallyRobustConditionalValueatRiskRange{<:Any, <:Any, <:Any,
                                                                       <:Any, <:Any, <:Any,
                                                                       <:Any, Nothing}})(x::VecNum)
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
function (r::Union{<:ConditionalValueatRiskRange{<:Any, <:Any, <:Any, <:AbstractWeights},
                   <:DistributionallyRobustConditionalValueatRiskRange{<:Any, <:Any, <:Any,
                                                                       <:Any, <:Any, <:Any,
                                                                       <:Any,
                                                                       <:AbstractWeights}})(x::VecNum)
    sw = sum(r.w)
    order = sortperm(x)
    sorted_x = x[order]
    sorted_w = r.w[order]
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    loss = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / (alpha)
    end

    sorted_x = reverse(sorted_x)
    sorted_w = reverse(sorted_w)
    cum_w = cumsum(sorted_w)
    beta = sw * r.beta
    idx = searchsortedfirst(cum_w, beta)
    gain = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (beta - cum_w[idx - 1])) / (beta)
    end
    return loss - gain
end
struct ConditionalDrawdownatRisk{T1, T2} <: RiskMeasure
    settings::T1
    alpha::T2
    function ConditionalDrawdownatRisk(settings::RiskMeasureSettings, alpha::Number)
        @argcheck(zero(alpha) < alpha < one(alpha))
        return new{typeof(settings), typeof(alpha)}(settings, alpha)
    end
end
function ConditionalDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   alpha::Number = 0.05)
    return ConditionalDrawdownatRisk(settings, alpha)
end
function (r::ConditionalDrawdownatRisk)(x::VecNum)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end
struct RelativeConditionalDrawdownatRisk{T1, T2} <: HierarchicalRiskMeasure
    settings::T1
    alpha::T2
    function RelativeConditionalDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                               alpha::Number)
        @argcheck(zero(alpha) < alpha < one(alpha))
        return new{typeof(settings), typeof(alpha)}(settings, alpha)
    end
end
function RelativeConditionalDrawdownatRisk(;
                                           settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                           alpha::Number = 0.05)
    return RelativeConditionalDrawdownatRisk(settings, alpha)
end
function (r::RelativeConditionalDrawdownatRisk)(x::VecNum)
    aT = r.alpha * length(x)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(x)
    popfirst!(dd)
    idx = ceil(Int, aT)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end

export ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
       ConditionalValueatRiskRange, DistributionallyRobustConditionalValueatRiskRange,
       ConditionalDrawdownatRisk, RelativeConditionalDrawdownatRisk
