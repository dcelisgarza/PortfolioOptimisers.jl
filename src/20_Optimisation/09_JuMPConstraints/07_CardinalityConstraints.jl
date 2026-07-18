"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the cardinality constraint `sum(ib) ≤ card` to the JuMP optimisation model.

`ind` is the indicator bundle the MIP builder that ran returned; the *held binary*
([`held_bin`](@ref)) is summed, so the sum counts held assets and the bound caps how many may be
non-zero. The constraint key is named for the space `sp` via [`mip_key`](@ref), so the same
emitter serves the asset space and every sub-group. The fall-through method does nothing when
`card` is `nothing` — and so never reads a held indicator the bundle may not carry.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraint acts on.
  - `ind::AbstractMIPIndicators`: Indicator bundle returned by the MIP builder that ran.
  - `card::Option{<:Integer}`: Maximum number of non-zero assets, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_gcard_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`held_bin`](@ref)
  - [`mip_key`](@ref)
"""
function set_card_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                               ind::AbstractMIPIndicators, card::Integer)
    sc = get_constraint_scale(model)
    ib = held_bin(ind)
    model[mip_key(sp, :card)] = JuMP.@constraint(model, sc * (sum(ib) - card) <= 0)
    return nothing
end
function set_card_constraints!(::JuMP.Model, ::AbstractMIPSpace, ::Any, ::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the group-cardinality constraints `A * ib ≤ B` and `A * ib = B` to the model.

Enforces the linear group-cardinality constraint carried by `gcard` on the held binary
([`held_bin`](@ref)) of the bundle `ind`, capping (or fixing) how many assets in each group may
be held. The constraint keys are named for the space `sp` via [`mip_key`](@ref), so the same
emitter serves the asset space and every sub-group. The fall-through method does nothing when
`gcard` is `nothing` — and so never reads a held indicator the bundle may not carry.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraints act on.
  - `ind::AbstractMIPIndicators`: Indicator bundle returned by the MIP builder that ran.
  - `gcard::Option{<:LinearConstraint}`: Group-cardinality constraint, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_card_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`held_bin`](@ref)
  - [`mip_key`](@ref)
"""
function set_gcard_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                ind::AbstractMIPIndicators, gcard::LinearConstraint)
    sc = get_constraint_scale(model)
    ib = held_bin(ind)
    if !isnothing(gcard.ineq)
        A = gcard.ineq.A
        B = gcard.ineq.B
        model[mip_key(sp, :gcard_ineq)] = JuMP.@constraint(model, sc * (A * ib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        model[mip_key(sp, :gcard_eq)] = JuMP.@constraint(model, sc * (A * ib ⊖ B) == 0)
    end
    return nothing
end
function set_gcard_constraints!(::JuMP.Model, ::AbstractMIPSpace, ::Any, ::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add all sub-group MIP constraints for a single or multiple selection matrices.

The single-matrix method runs the shared indicator builder ([`run_mip_builder!`](@ref)) on a [`SubsetMIPSpace`](@ref) — applying long/short thresholds and weight bounds — then emits the sub-group cardinality and group-cardinality constraints through the same space-generic emitters as the asset space ([`set_card_constraints!`](@ref), [`set_gcard_constraints!`](@ref)). The vector method iterates over collections of cardinalities, group constraints, and selection matrices.

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

  - [`run_mip_builder!`](@ref)
  - [`set_card_constraints!`](@ref)
  - [`set_gcard_constraints!`](@ref)
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
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = run_mip_builder!(model, sp, wb, lt, st, ss, lt_flag, st_flag)
    set_card_constraints!(model, sp, sib, card)
    set_gcard_constraints!(model, sp, sib, gcard)
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

The single-matrix method runs the shared indicator builder ([`run_mip_builder!`](@ref)) on the sub-group and enforces `sum(sib) ≤ card` through [`set_card_constraints!`](@ref). The vector method iterates over collections of cardinalities and selection matrices.

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

  - [`run_mip_builder!`](@ref)
  - [`set_card_constraints!`](@ref)
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
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = run_mip_builder!(model, sp, wb, lt, st, ss, lt_flag, st_flag)
    set_card_constraints!(model, sp, sib, card)
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

The single-matrix method runs the shared indicator builder ([`run_mip_builder!`](@ref)) on the sub-group and enforces the linear group cardinality constraints `A * sib ≤ B` and `A * sib = B` through [`set_gcard_constraints!`](@ref). The vector method iterates over multiple group constraints and selection matrices.

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

  - [`run_mip_builder!`](@ref)
  - [`set_gcard_constraints!`](@ref)
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
    sp = SubsetMIPSpace(smtx, :sg, i)
    sib = run_mip_builder!(model, sp, wb, lt, st, ss, lt_flag, st_flag)
    set_gcard_constraints!(model, sp, sib, gcard)
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
