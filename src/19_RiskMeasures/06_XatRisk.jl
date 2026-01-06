abstract type ValueatRiskFormulation <: AbstractAlgorithm end
function factory(alg::ValueatRiskFormulation, args...; kwargs...)
    return alg
end
function valueat_risk_formulation_view(r::ValueatRiskFormulation, args...)
    return r
end
struct MIPValueatRisk{T1, T2} <: ValueatRiskFormulation
    b::T1
    s::T2
    function MIPValueatRisk(b::Option{<:Number}, s::Option{<:Number})
        bflag = !isnothing(b)
        sflag = !isnothing(s)
        if bflag
            @argcheck(b > zero(b))
        end
        if sflag
            @argcheck(s > zero(s))
        end
        if bflag && sflag
            @argcheck(b > s)
        end
        return new{typeof(b), typeof(s)}(b, s)
    end
end
function MIPValueatRisk(; b::Option{<:Number} = nothing, s::Option{<:Number} = nothing)
    return MIPValueatRisk(b, s)
end
struct DistributionValueatRisk{T1, T2, T3, T4} <: ValueatRiskFormulation
    mu::T1
    sigma::T2
    chol::T3
    dist::T4
    function DistributionValueatRisk(mu::Option{<:VecNum}, sigma::Option{<:MatNum},
                                     chol::Option{<:MatNum},
                                     dist::Distributions.Distribution)
        if !isnothing(mu)
            @argcheck(!isempty(mu))
        end
        if !isnothing(sigma)
            @argcheck(!isempty(sigma))
            assert_matrix_issquare(sigma, :sigma)
        end
        if !isnothing(chol)
            @argcheck(!isempty(chol))
        end
        return new{typeof(mu), typeof(sigma), typeof(chol), typeof(dist)}(mu, sigma, chol,
                                                                          dist)
    end
end
function DistributionValueatRisk(; mu::Option{<:VecNum} = nothing,
                                 sigma::Option{<:MatNum} = nothing,
                                 chol::Option{<:MatNum} = nothing,
                                 dist::Distributions.Distribution = Distributions.Normal())
    return DistributionValueatRisk(mu, sigma, chol, dist)
end
function factory(alg::DistributionValueatRisk, pr::AbstractPriorResult, args...; kwargs...)
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    sigma = nothing_scalar_array_selector(alg.sigma, pr.sigma)
    chol = nothing_scalar_array_selector(alg.chol, pr.chol)
    return DistributionValueatRisk(; mu = mu, sigma = sigma, chol = chol, dist = alg.dist)
end
function valueat_risk_formulation_view(alg::DistributionValueatRisk, i)
    mu = nothing_scalar_array_view(alg.mu, i)
    sigma = nothing_scalar_array_view(alg.sigma, i)
    chol = isnothing(alg.chol) ? nothing : view(alg.chol, :, i)
    return DistributionValueatRisk(; mu = mu, sigma = sigma, chol = chol, dist = alg.dist)
end
struct ValueatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    alpha::T2
    w::T3
    alg::T4
    function ValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                         w::Option{<:StatsBase.AbstractWeights},
                         alg::ValueatRiskFormulation)
        @argcheck(zero(alpha) < alpha < one(alpha))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings, alpha,
                                                                            w, alg)
    end
end
function ValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Number = 0.05, w::Option{<:StatsBase.AbstractWeights} = nothing,
                     alg::ValueatRiskFormulation = MIPValueatRisk())
    return ValueatRisk(settings, alpha, w, alg)
end
function risk_measure_view(r::ValueatRisk, i)
    alg = valueat_risk_formulation_view(r.alg, i)
    return ValueatRisk(; settings = r.settings, alpha = r.alpha, w = r.w, alg = alg)
end
function (r::ValueatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    return -partialsort(x, ceil(Int, r.alpha * length(x)))
end
function (r::ValueatRisk{<:Any, <:Any, <:StatsBase.AbstractWeights})(x::VecNum)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(r.w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    return -sorted_x[idx]
end
struct ValueatRiskRange{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    alpha::T2
    beta::T3
    w::T4
    alg::T5
    function ValueatRiskRange(settings::RiskMeasureSettings, alpha::Number, beta::Number,
                              w::Option{<:StatsBase.AbstractWeights},
                              alg::ValueatRiskFormulation)
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w), typeof(alg)}(settings,
                                                                                          alpha,
                                                                                          beta,
                                                                                          w,
                                                                                          alg)
    end
end
function ValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          alpha::Number = 0.05, beta::Number = 0.05,
                          w::Option{<:StatsBase.AbstractWeights} = nothing,
                          alg::ValueatRiskFormulation = MIPValueatRisk())
    return ValueatRiskRange(settings, alpha, beta, w, alg)
end
function factory(r::ValueatRiskRange, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    alg = factory(r.alg, pr, args...; kwargs...)
    return ValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta, w = w,
                            alg = alg)
end
function risk_measure_view(r::ValueatRiskRange, i)
    alg = valueat_risk_formulation_view(r.alg, i)
    return ValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta,
                            w = r.w, alg = alg)
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, Nothing})(x::VecNum)
    x = copy(x)
    loss = -partialsort!(x, ceil(Int, r.alpha * length(x)))
    gain = -partialsort!(x, ceil(Int, r.beta * length(x)); rev = true)
    return loss - gain
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:StatsBase.AbstractWeights})(x::VecNum)
    w = r.w
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    loss = -sorted_x[idx]

    sorted_x = reverse!(sorted_x)
    sorted_w = reverse!(sorted_w)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.beta)
    idx = ifelse(idx > length(x), idx - 1, idx)
    gain = -sorted_x[idx]
    return loss - gain
end
struct DrawdownatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    alpha::T2
    w::T3
    alg::T4
    function DrawdownatRisk(settings::RiskMeasureSettings, alpha::Number,
                            w::Option{<:StatsBase.AbstractWeights}, alg::MIPValueatRisk)
        @argcheck(zero(alpha) < alpha < one(alpha))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings, alpha,
                                                                            w, alg)
    end
end
function DrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        alpha::Number = 0.05,
                        w::Option{<:StatsBase.AbstractWeights} = nothing,
                        alg::MIPValueatRisk = MIPValueatRisk())
    return DrawdownatRisk(settings, alpha, w, alg)
end
for r in (ValueatRisk, DrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 alg = factory(r.alg, pr, args...; kwargs...)
                 return $(r)(; settings = r.settings, alpha = r.alpha, w = w, alg = alg)
             end
         end)
end
function absolute_drawdown_vec(x::VecNum)
    pushfirst!(x, zero(eltype(x)))
    cs = cumsum(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        peak = ifelse(i > peak, i, peak)
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    return dd
end
function (r::DrawdownatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end
function (r::DrawdownatRisk{<:Any, <:Any, <:StatsBase.AbstractWeights})(x::VecNum)
    dd = absolute_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(r.w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(dd), idx - 1, idx)
    return -sorted_dd[idx]
end
struct RelativeDrawdownatRisk{T1, T2, T3} <: HierarchicalRiskMeasure
    settings::T1
    alpha::T2
    w::T3
    function RelativeDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                    alpha::Number, w::Option{<:StatsBase.AbstractWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Number = 0.05,
                                w::Option{<:StatsBase.AbstractWeights} = nothing)
    return RelativeDrawdownatRisk(settings, alpha, w)
end
function relative_drawdown_vec(x::VecNum)
    pushfirst!(x, zero(eltype(x)))
    cs = cumprod(x .+ one(eltype(x)))
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        peak = ifelse(i > peak, i, peak)
        dd[idx] = i / peak - one(peak)
    end
    popfirst!(x)
    popfirst!(dd)
    return dd
end
function (r::RelativeDrawdownatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    dd = relative_drawdown_vec(x)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end
function (r::RelativeDrawdownatRisk{<:Any, <:Any, <:StatsBase.AbstractWeights})(x::VecNum)
    dd = relative_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(r.w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(dd), idx - 1, idx)
    return -sorted_dd[idx]
end
function factory(r::RelativeDrawdownatRisk, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return RelativeDrawdownatRisk(; settings = r.settings, alpha = r.alpha, w = w)
end

const CholRM = Union{<:Variance, <:StandardDeviation,
                     <:ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk},
                     <:ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                        <:DistributionValueatRisk}}

export MIPValueatRisk, DistributionValueatRisk, ValueatRisk, ValueatRiskRange,
       DrawdownatRisk, RelativeDrawdownatRisk
