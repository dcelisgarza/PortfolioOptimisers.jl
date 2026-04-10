"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add turnover constraints to the JuMP optimisation model.

The fall-through method does nothing. The concrete method iterates over the collection of [`Turnover`](@ref) objects `tns` and delegates each to [`_set_turnover_constraints!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `tns`: One or more [`Turnover`](@ref) constraint specifications.

# Returns

  - `nothing`.

# Related

  - [`_set_turnover_constraints!`](@ref)
  - [`Turnover`](@ref)
  - [`set_turnover_fees!`](@ref)
"""
function set_turnover_constraints!(args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a single turnover constraint for a [`Turnover`](@ref) object to the JuMP optimisation model.

Introduces auxiliary variable `t_tn_i`, expression `tn_i = w - wb * k`, and enforces `‖w - wb‖₁ ≤ val * k` via NormOneCone constraints.

# Arguments

  - $(arg_dict[:model])
  - `tn::Turnover`: Turnover constraint specification containing benchmark weights `w` and tolerance `val`.
  - `i::Integer = 1`: Index used to generate unique variable and constraint names.

# Returns

  - `nothing`.

# Related

  - [`set_turnover_constraints!`](@ref)
  - [`Turnover`](@ref)
"""
function _set_turnover_constraints!(model::JuMP.Model, tn::Turnover, i::Integer = 1)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    N = length(w)
    wb = tn.w
    val = tn.val
    t_tn = model[Symbol(:t_tn_, i)] = JuMP.@variable(model, [1:N])
    tn = model[Symbol(:tn_, i)] = JuMP.@expression(model, w - wb * k)
    model[Symbol(:ctn_noc_, i)], model[Symbol(:ctn_, i)] = JuMP.@constraints(model,
                                                                             begin
                                                                                 [i = 1:N],
                                                                                 [sc *
                                                                                  t_tn[i]
                                                                                  sc *
                                                                                  tn[i]] in
                                                                                 JuMP.MOI.NormOneCone(2)
                                                                                 sc *
                                                                                 (t_tn ⊖
                                                                                  val * k) <=
                                                                                 0
                                                                             end)
    return nothing
end
function set_turnover_constraints!(model::JuMP.Model, tns::Tn_VecTn)
    for (i, tn) in enumerate(tns)
        _set_turnover_constraints!(model, tn, i)
    end
    return nothing
end
