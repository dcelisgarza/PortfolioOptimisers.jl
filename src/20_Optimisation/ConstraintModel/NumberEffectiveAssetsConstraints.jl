function set_number_effective_assets!(model::JuMP.Model, val::Real)
    w, k, sc = get_w_k_sc(model)
    @variable(model, nea)
    @constraints(model, begin
                     c_nea_soc, [sc * nea; sc * w] ∈ SecondOrderCone()
                     c_nea, sc * nea * sqrt(val) <= sc * k
                 end)
    return nothing
end
