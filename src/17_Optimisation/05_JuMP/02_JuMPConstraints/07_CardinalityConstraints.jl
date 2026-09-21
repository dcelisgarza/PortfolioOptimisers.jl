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
"""
    assert_subgroup_mip_fields(scard, smtx, slt, sst)

Refuses a sub-group MIP field set whose shapes do not agree, at the constructor of every owner that carries the four slots: [`JuMPOptimiser`](@ref) and [`ProgrammeAllocationSet`](@ref).

A `TimeDependent` in any slot is not a term, so the four are left to the per-fold substitution check.

# Validation

  - If `scard` is an `Integer`: `scard > 0` and finite; `smtx` is a matrix or an [`AssetSetsMatrixEstimator`](@ref); `slt` and `sst` are scalar or `nothing`.
  - If `scard` is a vector: every entry `> 0` and finite; `smtx` is a vector of the same length; a vector `slt` or `sst` is non-empty and of the same length.
  - If `scard` is `nothing` and `slt` or `sst` is scalar: `smtx` is a matrix or an estimator.
  - If `scard` is `nothing` and `slt` or `sst` is a vector: `smtx` is a non-empty vector, and each vector threshold has its length.

# Related

  - [`set_smip_constraints!`](@ref)
  - [`assert_subgrouped_mip_fields`](@ref)
"""
function assert_subgroup_mip_fields(scard, smtx, slt, sst)::Nothing
    if any(x -> isa(x, TimeDependent), (scard, smtx, slt, sst))
        return nothing
    end
    if isa(scard, Integer)
        assert_nonempty_gt0_finite_val(scard, :scard)
        @argcheck(isa(smtx, MatNum_ASetMatE),
                  ArgumentError("smtx must be a MatNum_ASetMatE when scard is an Integer, got $(typeof(smtx))"))
        @argcheck(isa(slt, Option{<:BtE_Bt}),
                  ArgumentError("slt must be a scalar BtE_Bt or nothing when scard is an Integer, got $(typeof(slt))"))
        @argcheck(isa(sst, Option{<:BtE_Bt}),
                  ArgumentError("sst must be a scalar BtE_Bt or nothing when scard is an Integer, got $(typeof(sst))"))
    elseif isa(scard, VecInt)
        assert_nonempty_gt0_finite_val(scard, :scard)
        @argcheck(isa(smtx, AbstractVector),
                  ArgumentError("smtx must be an AbstractVector when scard is a VecInt, got $(typeof(smtx))"))
        @argcheck(length(scard) == length(smtx),
                  DimensionMismatch("scard ($(length(scard))) must match smtx ($(length(smtx)))"))
        if isa(slt, AbstractVector)
            @argcheck(!isempty(slt), IsEmptyError("slt cannot be empty"))
            @argcheck(length(scard) == length(slt),
                      DimensionMismatch("scard ($(length(scard))) must match slt ($(length(slt)))"))
        end
        if isa(sst, AbstractVector)
            @argcheck(!isempty(sst), IsEmptyError("sst cannot be empty"))
            @argcheck(length(scard) == length(sst),
                      DimensionMismatch("scard ($(length(scard))) must match sst ($(length(sst)))"))
        end
    elseif isnothing(scard) && (isa(slt, BtE_Bt) || isa(sst, BtE_Bt))
        @argcheck(isa(smtx, MatNum_ASetMatE),
                  ArgumentError("smtx must be a MatNum_ASetMatE when slt or sst is a scalar BtE_Bt, got $(typeof(smtx))"))
    elseif isnothing(scard) && (isa(slt, AbstractVector) || isa(sst, AbstractVector))
        @argcheck(isa(smtx, AbstractVector),
                  ArgumentError("smtx must be an AbstractVector when slt or sst is a vector, got $(typeof(smtx))"))
        @argcheck(!isempty(smtx), IsEmptyError("smtx cannot be empty"))
        if isa(slt, AbstractVector)
            @argcheck(!isempty(slt), IsEmptyError("slt cannot be empty"))
            @argcheck(length(slt) == length(smtx),
                      DimensionMismatch("slt ($(length(slt))) must match smtx ($(length(smtx)))"))
        end
        if isa(sst, AbstractVector)
            @argcheck(!isempty(sst), IsEmptyError("sst cannot be empty"))
            @argcheck(length(sst) == length(smtx),
                      DimensionMismatch("sst ($(length(sst))) must match smtx ($(length(smtx)))"))
        end
    end
    return nothing
end
"""
    assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)

Refuses a sub-grouped cardinality field set whose shapes do not agree, at the constructor of every owner that carries the four slots: [`JuMPOptimiser`](@ref) and [`ProgrammeAllocationSet`](@ref). The builder pairs `sgcarde` with `sgmtx`, so a precomputed [`LinearConstraint`](@ref) is checked against the rows of `sgmtx`.

A `TimeDependent` in any slot is not a term, so the four are left to the per-fold substitution check.

# Validation

  - If `sgcarde` is scalar: `sgmtx` is a matrix or an estimator; `sglt` and `sgst` are scalar or `nothing`; a [`LinearConstraint`](@ref) against a matrix `sgmtx` has as many rows as `sgmtx`.
  - If `sgcarde` is a vector: non-empty; `sgmtx` a non-empty vector of the same length; a vector `sglt` or `sgst` non-empty and of the same length; each precomputed pair agrees in rows.
  - If `sgcarde` is `nothing` and `sglt` or `sgst` is scalar: `sgmtx` is a matrix or an estimator.
  - If `sgcarde` is `nothing` and `sglt` or `sgst` is a vector: `sgmtx` is a non-empty vector, and each vector threshold has its length.

# Related

  - [`set_smip_constraints!`](@ref)
  - [`assert_subgroup_mip_fields`](@ref)
"""
function assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)::Nothing
    if any(x -> isa(x, TimeDependent), (sgcarde, sgmtx, sglt, sgst))
        return nothing
    end
    if isa(sgcarde, LcE_Lc)
        @argcheck(isa(sgmtx, MatNum_ASetMatE),
                  ArgumentError("sgmtx must be a MatNum_ASetMatE when sgcarde is a scalar LcE_Lc, got $(typeof(sgmtx))"))
        @argcheck(isa(sglt, Option{<:BtE_Bt}),
                  ArgumentError("sglt must be a scalar BtE_Bt or nothing when sgcarde is a scalar LcE_Lc, got $(typeof(sglt))"))
        @argcheck(isa(sgst, Option{<:BtE_Bt}),
                  ArgumentError("sgst must be a scalar BtE_Bt or nothing when sgcarde is a scalar LcE_Lc, got $(typeof(sgst))"))
        if isa(sgcarde, LinearConstraint) && isa(sgmtx, MatNum)
            N = size(sgmtx, 1)
            N_ineq = !isnothing(sgcarde.ineq) ? length(sgcarde.B_ineq) : 0
            N_eq = !isnothing(sgcarde.eq) ? length(sgcarde.B_eq) : 0
            @argcheck(N == N_ineq + N_eq,
                      DimensionMismatch("sgmtx rows ($N) must equal N_ineq + N_eq ($(N_ineq + N_eq))"))
        end
    elseif isa(sgcarde, AbstractVector)
        @argcheck(!isempty(sgcarde), IsEmptyError("sgcarde cannot be empty"))
        @argcheck(isa(sgmtx, AbstractVector),
                  ArgumentError("sgmtx must be an AbstractVector when sgcarde is a vector, got $(typeof(sgmtx))"))
        @argcheck(!isempty(sgmtx), IsEmptyError("sgmtx cannot be empty"))
        @argcheck(length(sgcarde) == length(sgmtx),
                  DimensionMismatch("sgcarde ($(length(sgcarde))) must match sgmtx ($(length(sgmtx)))"))
        if isa(sglt, AbstractVector)
            @argcheck(!isempty(sglt), IsEmptyError("sglt cannot be empty"))
            @argcheck(length(sgcarde) == length(sglt),
                      DimensionMismatch("sgcarde ($(length(sgcarde))) must match sglt ($(length(sglt)))"))
        end
        if isa(sgst, AbstractVector)
            @argcheck(!isempty(sgst), IsEmptyError("sgst cannot be empty"))
            @argcheck(length(sgcarde) == length(sgst),
                      DimensionMismatch("sgcarde ($(length(sgcarde))) must match sgst ($(length(sgst)))"))
        end
        for (sgc, smt) in zip(sgcarde, sgmtx)
            if isa(sgc, LinearConstraint) && isa(smt, MatNum)
                N = size(smt, 1)
                N_ineq = !isnothing(sgc.ineq) ? length(sgc.B_ineq) : 0
                N_eq = !isnothing(sgc.eq) ? length(sgc.B_eq) : 0
                @argcheck(N == N_ineq + N_eq,
                          DimensionMismatch("smt rows ($N) must equal N_ineq + N_eq ($(N_ineq + N_eq))"))
            end
        end
    elseif isnothing(sgcarde) && (isa(sglt, BtE_Bt) || isa(sgst, BtE_Bt))
        @argcheck(isa(sgmtx, MatNum_ASetMatE),
                  ArgumentError("sgmtx must be a MatNum_ASetMatE when sglt or sgst is a scalar BtE_Bt, got $(typeof(sgmtx))"))
    elseif isnothing(sgcarde) && (isa(sglt, AbstractVector) || isa(sgst, AbstractVector))
        @argcheck(isa(sgmtx, AbstractVector),
                  ArgumentError("sgmtx must be an AbstractVector when sglt or sgst is a vector, got $(typeof(sgmtx))"))
        @argcheck(!isempty(sgmtx), IsEmptyError("sgmtx cannot be empty"))
        if isa(sglt, AbstractVector)
            @argcheck(!isempty(sglt), IsEmptyError("sglt cannot be empty"))
            @argcheck(length(sglt) == length(sgmtx),
                      DimensionMismatch("sglt ($(length(sglt))) must match sgmtx ($(length(sgmtx)))"))
        end
        if isa(sgst, AbstractVector)
            @argcheck(!isempty(sgst), IsEmptyError("sgst cannot be empty"))
            @argcheck(length(sgst) == length(sgmtx),
                      DimensionMismatch("sgst ($(length(sgst))) must match sgmtx ($(length(sgmtx)))"))
        end
    end
    return nothing
end
