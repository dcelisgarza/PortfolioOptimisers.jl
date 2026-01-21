struct AverageDrawdown{T1, T2} <: RiskMeasure
    settings::T1
    w::T2
    function AverageDrawdown(settings::RiskMeasureSettings,
                             w::Option{<:StatsBase.AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function AverageDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:StatsBase.AbstractWeights} = nothing)
    return AverageDrawdown(settings, w)
end
function (r::AverageDrawdown)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return -(isnothing(r.w) ? Statistics.mean(dd) : Statistics.mean(dd, r.w))
end
struct RelativeAverageDrawdown{T1, T2} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    function RelativeAverageDrawdown(settings::HierarchicalRiskMeasureSettings,
                                     w::Option{<:StatsBase.AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function RelativeAverageDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return RelativeAverageDrawdown(settings, w)
end
function (r::RelativeAverageDrawdown)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return -(isnothing(r.w) ? Statistics.mean(dd) : Statistics.mean(dd, r.w))
end
for r in (AverageDrawdown, RelativeAverageDrawdown)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 return $(r)(; settings = r.settings, w = w)
             end
         end)
end

export AverageDrawdown, RelativeAverageDrawdown
