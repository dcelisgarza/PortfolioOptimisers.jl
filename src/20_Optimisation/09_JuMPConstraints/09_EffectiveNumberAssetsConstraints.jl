function set_number_effective_assets!(args...)
    return nothing
end
function set_number_effective_assets!(model::JuMP.Model, val::Real)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    @variable(model, nea)
    @constraints(model, begin
                     cnea_soc, [sc * nea; sc * w] in SecondOrderCone()
                     cnea, sc * (nea * sqrt(val) - k) <= 0
                 end)
    return nothing
end
