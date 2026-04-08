@concrete struct AverageDrawdown <: RiskMeasure
    settings
    w
    function AverageDrawdown(settings::RiskMeasureSettings, w::Option{<:ObsWeights})
        validate_observation_weights(w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function AverageDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:ObsWeights} = nothing)
    return AverageDrawdown(settings, w)
end
function (r::AverageDrawdown)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return -(isnothing(r.w) ? Statistics.mean(dd) : Statistics.mean(dd, r.w))
end
@concrete struct RelativeAverageDrawdown <: HierarchicalRiskMeasure
    settings
    w
    function RelativeAverageDrawdown(settings::HierarchicalRiskMeasureSettings,
                                     w::Option{<:ObsWeights})
        validate_observation_weights(w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function RelativeAverageDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Option{<:ObsWeights} = nothing)
    return RelativeAverageDrawdown(settings, w)
end
function (r::RelativeAverageDrawdown)(x::VecNum)
    dd = relative_drawdown_vec(x)
    w = get_observation_weights(r.w, x)
    return -(isnothing(r.w) ? Statistics.mean(dd) : Statistics.mean(dd, w))
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
