"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the asset-space cardinality constraint `sum(ib) ≤ card` to the JuMP optimisation model.

`ib` is the *held* indicator returned by the MIP builder that ran, so the sum counts held
assets and the bound caps how many may be non-zero. The fall-through method does nothing when
`card` is `nothing`.

# Arguments

  - $(arg_dict[:model])
  - `ib`: Held indicator returned by the MIP builder that ran.
  - `card::Option{<:Integer}`: Maximum number of non-zero assets, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_gcard_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
"""
function set_card_constraints!(model::JuMP.Model, ib, card::Integer)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, card, sc * (sum(ib) - card) <= 0)
    return nothing
end
function set_card_constraints!(::JuMP.Model, ::Any, ::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the asset-space group-cardinality constraints `A * ib ≤ B` and `A * ib = B` to the model.

Enforces the linear group-cardinality constraint carried by `gcard` on the held indicator `ib`,
capping (or fixing) how many assets in each group may be held. The fall-through method does
nothing when `gcard` is `nothing`.

# Arguments

  - $(arg_dict[:model])
  - `ib`: Held indicator returned by the MIP builder that ran.
  - `gcard::Option{<:LinearConstraint}`: Group-cardinality constraint, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_card_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
"""
function set_gcard_constraints!(model::JuMP.Model, ib, gcard::LinearConstraint)
    sc = get_constraint_scale(model)
    if !isnothing(gcard.ineq)
        A = gcard.ineq.A
        B = gcard.ineq.B
        JuMP.@constraint(model, gcard_ineq, sc * (A * ib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        JuMP.@constraint(model, gcard_eq, sc * (A * ib ⊖ B) == 0)
    end
    return nothing
end
function set_gcard_constraints!(::JuMP.Model, ::Any, ::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all sub-group MIP constraints for a single or multiple selection matrices.

The single-matrix method handles cardinality, group cardinality, long/short threshold, and weight-bound constraints for one sub-group via the shared builders ([`short_mip_threshold_constraints`](@ref), [`mip_constraints`](@ref)) on a [`SubsetMIPSpace`](@ref). The vector method iterates over collections of cardinalities, group constraints, and selection matrices.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: Cardinality bound(s) for the sub-group(s).
  - `gcard`: Group-cardinality constraint(s) for the sub-group(s).
  - `smtx`: Selection matrix (or vector thereof) for the sub-group(s).
  - `lt`: Long-side minimum-holding threshold(s).
  - `st`: Short-side minimum-holding threshold(s).
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: Index for generating unique names (single-matrix method only).

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`set_smip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer},
                                   gcard::Option{<:LinearConstraint},
                                   smtx::Option{<:MatNum}, lt::Option{<:Threshold},
                                   st::Option{<:Threshold}, ss::Option{<:Number},
                                   i::Integer = 1)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = get_constraint_scale(model)
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = if st_flag && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, nothing, nothing, ss,
                                        lt_flag, st_flag, false, false)
    else
        mip_constraints(model, sp, wb, nothing, lt, ss, lt_flag, false, false)
    end
    if card_flag
        model[Symbol(:scard_, i)] = JuMP.@constraint(model, sc * (sum(sib) - card) <= 0)
    end
    if gcard_flag
        if !isnothing(gcard.ineq)
            A = gcard.ineq.A
            B = gcard.ineq.B
            model[Symbol(:sgcard_ineq_, i)] = JuMP.@constraint(model,
                                                               sc * (A * sib ⊖ B) <= 0)
        end
        if !isnothing(gcard.eq)
            A = gcard.eq.A
            B = gcard.eq.B
            model[Symbol(:sgcard_eq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) == 0)
        end
    end
    return nothing
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds, card::VecInt,
                                   gcard::VecLc, smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                   st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (c, g, s)) in enumerate(zip(card, gcard, smtx))
        lti = isa(lt, Option{<:Threshold}) ? lt : lt[i]
        sti = isa(st, Option{<:Threshold}) ? st : st[i]
        set_all_smip_constraints!(model, wb, c, g, s, lti, sti, ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add sub-group cardinality MIP constraints using a selection matrix.

The single-matrix method enforces `sum(sib) ≤ card` for one sub-group. The vector method iterates over collections of cardinalities and selection matrices.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: Cardinality bound(s) for the sub-group(s).
  - `smtx`: Selection matrix (or vector thereof).
  - `lt`: Long-side minimum-holding threshold(s).
  - `st`: Short-side minimum-holding threshold(s).
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: Index for generating unique names (single-matrix method only).

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`set_smip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer}, smtx::Option{<:MatNum},
                                   lt::Option{<:Threshold}, st::Option{<:Threshold},
                                   ss::Option{<:Number}, i::Integer = 1)
    card_flag = !isnothing(card)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || lt_flag || st_flag)
        return nothing
    end
    sc = get_constraint_scale(model)
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = if st_flag && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, nothing, nothing, ss,
                                        lt_flag, st_flag, false, false)
    else
        mip_constraints(model, sp, wb, nothing, lt, ss, lt_flag, false, false)
    end
    model[Symbol(:scard_, i)] = JuMP.@constraint(model, sc * (sum(sib) - card) <= 0)
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds, card::VecInt,
                                   smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                   st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (c, s)) in enumerate(zip(card, smtx))
        lti = isa(lt, Option{<:Threshold}) ? lt : lt[i]
        sti = isa(st, Option{<:Threshold}) ? st : st[i]
        set_scardmip_constraints!(model, wb, c, s, lti, sti, ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add sub-group group-cardinality MIP constraints using a selection matrix.

The single-matrix method enforces linear group cardinality constraints `A * sib ≤ B` and `A * sib = B` on the sub-group binary indicator. The vector method iterates over multiple group constraints and selection matrices.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `gcard`: Group-cardinality constraint(s).
  - `smtx`: Selection matrix (or vector thereof).
  - `lt`: Long-side minimum-holding threshold(s).
  - `st`: Short-side minimum-holding threshold(s).
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: Index for generating unique names (single-matrix method only).

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_smip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::Option{<:LinearConstraint},
                                    smtx::Option{<:MatNum}, lt::Option{<:Threshold},
                                    st::Option{<:Threshold}, ss::Option{<:Number},
                                    i::Integer = 1)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sc = get_constraint_scale(model)
    sp = SubsetMIPSpace(smtx, :sg, i)
    sib = if st_flag && haskey(model, :sw)
        short_mip_threshold_constraints(model, sp, wb, lt, st, nothing, nothing, ss,
                                        lt_flag, st_flag, false, false)
    else
        mip_constraints(model, sp, wb, nothing, lt, ss, lt_flag, false, false)
    end
    if !isnothing(gcard.ineq)
        A = gcard.ineq.A
        B = gcard.ineq.B
        model[Symbol(:sgcard_ineq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        model[Symbol(:sgcard_eq_, i)] = JuMP.@constraint(model, sc * (A * sib ⊖ B) == 0)
    end
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds, gcard::VecLc,
                                    smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                    st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, (gc, s)) in enumerate(zip(gcard, smtx))
        lti = isa(lt, Option{<:Threshold}) ? lt : lt[i]
        sti = isa(st, Option{<:Threshold}) ? st : st[i]
        set_sgcardmip_constraints!(model, wb, gc, s, lti, sti, ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all sub-group MIP constraints (both cardinality and group-cardinality) to the JuMP optimisation model.

Dispatches between combined selection matrices (calling [`set_all_smip_constraints!`](@ref)) and separate cardinality/group-cardinality selection matrices (calling [`set_scardmip_constraints!`](@ref) and [`set_sgcardmip_constraints!`](@ref) independently).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: Cardinality bound(s).
  - `gcard`: Group-cardinality constraint(s).
  - `smtx`: Cardinality selection matrix (or vector thereof).
  - `sgmtx`: Group-cardinality selection matrix (or vector thereof).
  - `lt`: Long-side minimum-holding threshold(s) for cardinality sub-groups.
  - `st`: Short-side minimum-holding threshold(s) for cardinality sub-groups.
  - `glt`: Long-side minimum-holding threshold(s) for group-cardinality sub-groups.
  - `gst`: Short-side minimum-holding threshold(s) for group-cardinality sub-groups.
  - $(arg_dict[:ss_arg])

# Returns

  - `nothing`.

# Related

  - [`set_all_smip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                               card::Option{<:Int_VecInt}, gcard::Option{<:Lc_VecLc},
                               smtx::Option{<:MatNum_VecMatNum},
                               sgmtx::Option{<:MatNum_VecMatNum}, lt::Option{<:Bt_VecOptBt},
                               st::Option{<:Bt_VecOptBt}, glt::Option{<:Bt_VecOptBt},
                               gst::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    if smtx === sgmtx
        set_all_smip_constraints!(model, wb, card, gcard, smtx, lt, st, ss)
    else
        set_scardmip_constraints!(model, wb, card, smtx, lt, st, ss)
        set_sgcardmip_constraints!(model, wb, gcard, sgmtx, glt, gst, ss)
    end
    return nothing
end
