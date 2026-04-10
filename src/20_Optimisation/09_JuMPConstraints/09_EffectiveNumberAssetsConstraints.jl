"""
    set_number_effective_assets!(args...)
    set_number_effective_assets!(model, val)

Add an effective number of assets constraint to the JuMP model.

Enforces ``\\|w\\|_2 \\le 1/\\sqrt{val}`` using a second-order cone constraint,
which is equivalent to requiring ``N_{\\mathrm{eff}} \\ge val``.
The fall-through method does nothing.
"""
function set_number_effective_assets!(args...)
    return nothing
end
function set_number_effective_assets!(model::JuMP.Model, val::Number)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    JuMP.@variable(model, nea)
    JuMP.@constraints(model, begin
                          cnea_soc, [sc * nea; sc * w] in JuMP.SecondOrderCone()
                          cnea, sc * (nea * sqrt(val) - k) <= 0
                      end)
    return nothing
end
