function _w_neg_flag(wb::Real)
    return wb < zero(wb)
end
function _w_neg_flag(wb::AbstractVector)
    return any(wb .< zero(eltype(wb)))
end
function set_weight_constraints!(::JuMP.Model, ::Nothing, ::Bool)
    return nothing
end
function set_weight_constraints!(::JuMP.Model, wb::WeightBounds{Nothing, Nothing}, ::Bool)
    return nothing
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 long_only::Bool = false)
    @smart_assert(long_only ⊼ _w_neg_flag(wb.lb),
                  "Long-only strategy cannot have negative weight limits")
    w, k, sc = get_w_k_sc(model)
    if !isnothing(wb.lb)
        @constraint(model, w_lb, sc * w .>= sc * k * wb.lb)
    end
    if !isnothing(wb.ub)
        @constraint(model, w_ub, sc * w .<= sc * k * wb.ub)
    end
    return nothing
end

export WeightBounds
