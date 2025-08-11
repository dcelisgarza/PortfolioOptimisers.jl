function set_risk_budget!(rb::AbstractVector,
                          plca::LinearConstraintSide{<:AbstractVector, <:AbstractVector,
                                                     <:AbstractVector}, sets::DataFrame,
                          strict::Bool = false)
    group_names = names(sets)
    for (group, name, coef) in zip(plca.group, plca.name, plca.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero, idx)
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(plca)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(plca)")
                end
                continue
            end
            rb[idx] .= coef
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(plca)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(plca).")
        end
    end
    return nothing
end
function set_risk_budget!(rb::AbstractVector,
                          plca::LinearConstraintSide{<:Any, <:Any, <:Real}, sets::DataFrame,
                          strict::Bool = false)
    group_names = names(sets)
    (; group, name, coef) = plca
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero, idx)
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(plca)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(plca)")
            end
            return rb
        end
        rb[idx] .= coef
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(plca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(plca)")
    end
    return nothing
end
function risk_budget_constraints(plcas::Union{<:LinearConstraintSide,
                                              <:AbstractVector{<:LinearConstraintSide}},
                                 sets::DataFrame; datatype::DataType = Float64,
                                 strict::Bool = false)
    if isa(plcas, AbstractVector)
        @assert(!isempty(plcas))
    end
    @assert(!isempty(sets))
    rb = Vector{datatype}(undef, nrow(sets))
    fill!(rb, inv(nrow(sets)))
    for plc in plcas
        set_risk_budget!(rb, plc, sets, strict)
    end
    rb ./= sum(rb)
    return rb
end
function risk_budget_constraints(plca::AbstractArray, args...; kwargs...)
    return plca
end

export risk_budget_constraints
