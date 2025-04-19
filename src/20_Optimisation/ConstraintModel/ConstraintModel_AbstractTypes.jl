abstract type CustomConstraint end
function set_custom_constraint!(::JuMP.Model, ::Nothing, args...)
    return nothing
end
function get_w_k_sc(model::JuMP.Model)
    return model[:w], model[:k], model[:sc]
end
function non_zero_real_or_vec(::Nothing)
    return false
end
function non_zero_real_or_vec(x::Real)
    return !iszero(x)
end
function non_zero_real_or_vec(x::AbstractVector{<:Real})
    return any(!iszero, x)
end
