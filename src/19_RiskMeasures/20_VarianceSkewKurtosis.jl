@concrete struct VarianceSkewKurtosis <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    vr
    sk
    kt
    function VarianceSkewKurtosis(settings::RiskMeasureSettings, vr::Variance,
                                  sk::NegativeSkewness, kt::Kurtosis)
        vr = no_risk_expr_risk_measure(vr)
        sk = no_risk_expr_risk_measure(sk)
        kt = no_risk_expr_risk_measure(kt)
        return new{typeof(settings), typeof(vr), typeof(sk), typeof(kt)}(settings, vr, sk,
                                                                         kt)
    end
end
function VarianceSkewKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              vr::Variance = Variance(),
                              sk::NegativeSkewness = NegativeSkewness(),
                              kt::Kurtosis = Kurtosis())
    return VarianceSkewKurtosis(settings, vr, sk, kt)
end
function factory(r::VarianceSkewKurtosis, pr::AbstractPriorResult, args...;
                 kwargs...)::VarianceSkewKurtosis
    vr = factory(r.vr, pr, args...; kwargs...)
    sk = factory(r.sk, pr, args...; kwargs...)
    kt = factory(r.kt, pr, args...; kwargs...)
    return VarianceSkewKurtosis(; settings = r.settings, vr = vr, sk = sk, kt = kt)
end
function risk_measure_view(r::VarianceSkewKurtosis, i, args...)
    vr = risk_measure_view(r.vr, i, args...)
    sk = risk_measure_view(r.sk, i, args...)
    kt = risk_measure_view(r.kt, i, args...)
    return VarianceSkewKurtosis(; settings = r.settings, vr = vr, sk = sk, kt = kt)
end
function (r::VarianceSkewKurtosis)(w::VecNum, X::MatNum, fees::Option{<:VecNum} = nothing)
    return r.vr(w) * r.vr.settings.scale - r.sk(w) * r.sk.settings.scale +
           r.kt(w, X, fees) * r.kt.settings.scale
end

export VarianceSkewKurtosis
