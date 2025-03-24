abstract type CustomConstraint end
struct NoCustomConstraint <: CustomConstraint end
function set_custom_constraint!(::Any, ::NoCustomConstraint)
    return nothing
end
function get_w_k_sc(model::JuMP.Model)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    return w, k, sc
end
function non_zero_real_or_vec(x::Real)
    return !iszero(x)
end
function non_zero_real_or_vec(x::AbstractVector{<:Real})
    return any(.!iszero.(x))
end
