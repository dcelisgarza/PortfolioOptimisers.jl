# https://github.com/oxfordcontrol/Clarabel.jl/blob/4915b83e0d900d978681d5e8f3a3a5b8e18086f0/warmstart_test/portfolioOpt/higherorderRiskMeansure.jl#L23
struct PowerValueatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    alpha::T2
    p::T3
    w::T4
    function PowerValueatRisk(settings::RiskMeasureSettings, alpha::Number, p::Number,
                              w::Option{<:AbstractWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p > one(p))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(p), typeof(w)}(settings, alpha,
                                                                          p, w)
    end
end
function PowerValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          alpha::Number = 0.05, p::Number = 2.0,
                          w::Option{<:AbstractWeights} = nothing)
    return PowerValueatRisk(settings, alpha, p, w)
end
function PRMERM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
                w::Option{<:AbstractWeights} = nothing)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
    end
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    T = length(x)
    @variables(model, begin
                   eta
                   y
                   x[1:T]
                   z[1:T]
               end)
    return if optimise_JuMP_model!(model, slv).success
        objective_value(model)
    else
        NaN
    end
end
struct PowerValueatRiskRange <: RiskMeasure end
struct PowerDrawdownatRisk <: RiskMeasure end
struct RelativePowerDrawdownatRisk <: HierarchicalRiskMeasure end
