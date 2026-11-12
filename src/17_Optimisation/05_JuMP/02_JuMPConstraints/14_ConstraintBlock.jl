"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the static defaults of the [`ConstraintBlock`](@ref) slots that can hold a [`TimeDependent`](@ref).

The constructor of the block reads them when it checks each entry of a schedule, and [`jump_optimiser_td_defaults`](@ref) adds them to the defaults of the owner. A slot whose static default is `nothing` is not in the tuple.

# Returns

  - `NamedTuple`: The default of `wb`.

# Related

  - [`ConstraintBlock`](@ref)
  - [`jump_optimiser_td_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function constraint_block_td_defaults()::NamedTuple
    return (; wb = WeightBounds())
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the feasible set of the weights of a programme, and the terms that the programme adds to the Objective Penalty.

The block is the Constraint Block of the glossary. It holds the slots that a JuMP optimiser and a programme Allocation Set share: the weight bounds, the leverage caps, the linear, cardinality, threshold and phylogeny constraints, the turnover and tracking constraints, the norm ceilings, the custom constraints and the four regularisation penalties. A slot never writes the objective. The owner of the block holds the net budget, the fees, the return term and the scalariser.

[`JuMPOptimiser`](@ref) and [`ProgrammeAllocationSet`](@ref) each hold a block in their field `cbe`. The keyword constructor of each owner takes each slot of the block as a flat keyword, and `opt.wb` reads the slot of the block. So build a block directly only to give one feasible set to more than one owner.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConstraintBlock(;
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        sbgt::TD_Option{<:Num_BgtRg} = nothing,
        gbgt::TD_Option{<:Num_BgtRg} = nothing,
        lt::TD_Option{<:BtE_Bt} = nothing,
        st::TD_Option{<:BtE_Bt} = nothing,
        lcse::TD_Option{<:Union{EcE_LcE_Lc, AbstractVector}} = nothing,
        cte::TD_Option{<:Lc_CC_VecCC} = nothing,
        gcarde::TD_Option{<:LcE_Lc} = nothing,
        sgcarde::TD_Option{<:Union{LcE_Lc, AbstractVector}} = nothing,
        smtx::TD_Option{<:Union{MatNum_ASetMatE, AbstractVector}} = nothing,
        sgmtx::TD_Option{<:Union{MatNum_ASetMatE, AbstractVector}} = nothing,
        slt::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
        sst::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
        sglt::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
        sgst::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
        tn::TD_Option{<:Union{TnE_Tn, AbstractVector}} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        tr::TD_Option{<:Tr_VecTr} = nothing,
        ple::TD_Option{<:Union{PlCE_PlC, AbstractVector}} = nothing,
        ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr} = nothing,
        cobj::TD_Option{<:JuMPObj_VecJuMPObj} = nothing,
        ss::TD_Option{<:Number} = nothing,
        card::TD_Option{<:Integer} = nothing,
        scard::TD_Option{<:Int_VecInt} = nothing,
        l2c::TD_Option{<:Num_NormCeilCal} = nothing,
        lpc::TD_Option{<:LpReg_VecLpReg} = nothing,
        linfc::TD_Option{<:Num_NormCeilCal} = nothing,
        l1::TD_Option{<:Num_AmbRadCal} = nothing,
        l2::TD_Option{<:L2Reg_VecL2Reg} = nothing,
        lp::TD_Option{<:LpReg_VecLpReg} = nothing,
        linf::TD_Option{<:Num_AmbRadCal} = nothing,
        xbgt::Bool = false,
    ) -> ConstraintBlock

Keywords correspond to the struct's fields. A slot typed [`TD_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value, as the slot of the same name in [`JuMPOptimiser`](@ref) can.

`lcse`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn` and `ple` take a vector that mixes the members of their element type. The keyword constructor narrows it through [`narrow_mixed_vector`](@ref).

## Validation

  - If `sbgt` or `gbgt` is a number: it is finite and `>= 0`.
  - If `lcse`, `cte`, `smtx`, `sgmtx`, `tn`, `tr`, `ple`, `l2`, `lp` or `lpc` is a vector: `!isempty`.
  - If `card`, `l2c`, `linfc`, `l1` or `linf` is a number: it is finite and `> 0`.
  - `lp` holds no norm-ceiling rule, and `lpc` holds no ambiguity-radius rule, see [`assert_penalty_coefficient_role`](@ref) and [`assert_norm_ceiling_role`](@ref).
  - The shapes of `scard`, `smtx`, `slt` and `sst` agree, see [`assert_subgroup_mip_fields`](@ref). The shapes of `sgcarde`, `sgmtx`, `sglt` and `sgst` agree, see [`assert_subgrouped_mip_fields`](@ref).
  - If a slot holds an estimator keyed by name ([`name_keyed`](@ref)): `!isnothing(sets)`.
  - If a slot holds a [`TimeDependent`](@ref): the constructor runs again once for each entry of the schedule, see [`assert_time_dependent_substitution`](@ref).

The checks that read a field of the owner as well, such as [`assert_gross_budget_admissible`](@ref), run in the constructor of the owner.

# Examples

```jldoctest
julia> cbe = ConstraintBlock(; wb = WeightBounds(; lb = 0.0, ub = 0.4), card = 3);

julia> opt = JuMPOptimiser(; slv = Solver(; solver = nothing), cbe = cbe, card = 4);

julia> (opt.wb.ub, opt.card, opt.cbe.card)
(0.4, 4, 4)
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct ConstraintBlock <: AbstractEstimator
    """
    $(field_dict[:wb_jmp])
    """
    wb
    """
    $(field_dict[:sbgt])
    """
    sbgt
    """
    $(field_dict[:gbgt])
    """
    gbgt
    """
    $(field_dict[:lt])
    """
    lt
    """
    $(field_dict[:st])
    """
    st
    """
    $(field_dict[:lcse])
    """
    lcse
    """
    $(field_dict[:cte_jmp])
    """
    cte
    """
    $(field_dict[:gcarde])
    """
    gcarde
    """
    $(field_dict[:sgcarde])
    """
    sgcarde
    """
    $(field_dict[:smtx])
    """
    smtx
    """
    $(field_dict[:sgmtx])
    """
    sgmtx
    """
    $(field_dict[:slt])
    """
    slt
    """
    $(field_dict[:sst])
    """
    sst
    """
    $(field_dict[:sglt])
    """
    sglt
    """
    $(field_dict[:sgst])
    """
    sgst
    """
    $(field_dict[:tn_jmp])
    """
    tn
    """
    $(field_dict[:sets])
    """
    sets
    """
    $(field_dict[:tr_jmp])
    """
    tr
    """
    $(field_dict[:ple_jmp])
    """
    ple
    """
    $(field_dict[:ccnt])
    """
    ccnt
    """
    $(field_dict[:cobj])
    """
    cobj
    """
    $(field_dict[:ss])
    """
    ss
    """
    $(field_dict[:card])
    """
    card
    """
    $(field_dict[:scard])
    """
    scard
    """
    $(field_dict[:l2c])
    """
    l2c
    """
    $(field_dict[:lpc])
    """
    lpc
    """
    $(field_dict[:linfc])
    """
    linfc
    """
    $(field_dict[:l1])
    """
    l1
    """
    $(field_dict[:l2])
    """
    l2
    """
    $(field_dict[:lp])
    """
    lp
    """
    $(field_dict[:linf])
    """
    linf
    """
    $(field_dict[:xbgt])
    """
    xbgt
    function ConstraintBlock(wb::TD_Option{<:WbE_Wb}, sbgt::TD_Option{<:Num_BgtRg},
                             gbgt::TD_Option{<:Num_BgtRg}, lt::TD_Option{<:BtE_Bt},
                             st::TD_Option{<:BtE_Bt},
                             lcse::TD_Option{<:EcE_LcE_Lc_VecEcE_LcE_Lc},
                             cte::TD_Option{<:Lc_CC_VecCC}, gcarde::TD_Option{<:LcE_Lc},
                             sgcarde::TD_Option{<:LcE_Lc_VecLcE_Lc},
                             smtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                             sgmtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                             slt::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                             sst::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                             sglt::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                             sgst::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                             tn::TD_Option{<:TnE_Tn_VecTnE_Tn},
                             sets::TD_Option{<:UniverseSets}, tr::TD_Option{<:Tr_VecTr},
                             ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC},
                             ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr},
                             cobj::TD_Option{<:JuMPObj_VecJuMPObj}, ss::TD_Option{<:Number},
                             card::TD_Option{<:Integer}, scard::TD_Option{<:Int_VecInt},
                             l2c::TD_Option{<:Num_NormCeilCal},
                             lpc::TD_Option{<:LpReg_VecLpReg},
                             linfc::TD_Option{<:Num_NormCeilCal},
                             l1::TD_Option{<:Num_AmbRadCal},
                             l2::TD_Option{<:L2Reg_VecL2Reg},
                             lp::TD_Option{<:LpReg_VecLpReg},
                             linf::TD_Option{<:Num_AmbRadCal}, xbgt::Bool)
        if isa(sbgt, Number)
            assert_nonempty_nonneg_finite_val(sbgt, :sbgt)
        end
        if isa(gbgt, Number)
            assert_nonempty_nonneg_finite_val(gbgt, :gbgt)
        end
        # `map` compiles one call for each slot, so the slots of different types are not boxed.
        map(((lcse, :lcse), (cte, :cte), (smtx, :smtx), (sgmtx, :sgmtx), (tn, :tn),
             (tr, :tr), (ple, :ple), (l2, :l2), (lp, :lp), (lpc, :lpc))) do (x, name)
            if isa(x, AbstractVector)
                @argcheck(!isempty(x), IsEmptyError("$name cannot be empty"))
            end
            return nothing
        end
        # The method for a value other than a number checks nothing, so `nothing`, a schedule
        # and a calibration rule pass these calls.
        assert_nonempty_gt0_finite_val(card, :card)
        assert_nonempty_gt0_finite_val(l2c, :l2c)
        assert_nonempty_gt0_finite_val(linfc, :linfc)
        assert_nonempty_gt0_finite_val(l1, :l1)
        assert_nonempty_gt0_finite_val(linf, :linf)
        # `LpRegularisation.val` is read as a coefficient in `lp` and as a ceiling in
        # `lpc`, and one field cannot carry two bounds. This is the first point at which
        # the reading is known, so it is where the wrong role is refused. A `TimeDependent`
        # needs no guard: it is not a term, so it meets the permissive fallback, and the
        # constructor test-substitutes each of its entries through this same check.
        assert_penalty_coefficient_role(lp)
        assert_norm_ceiling_role(lpc)
        assert_subgroup_mip_fields(scard, smtx, slt, sst)
        assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)
        if any(name_keyed,
               (wb, lt, st, lcse, gcarde, sgcarde, smtx, sgmtx, slt, sst, sglt, sgst, tn))
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when estimator-type fields are provided"))
        end
        assert_time_dependent_substitution(ConstraintBlock,
                                           (; wb, sbgt, gbgt, lt, st, lcse, cte, gcarde,
                                            sgcarde, smtx, sgmtx, slt, sst, sglt, sgst, tn,
                                            sets, tr, ple, ccnt, cobj, ss, card, scard, l2c,
                                            lpc, linfc, l1, l2, lp, linf, xbgt),
                                           constraint_block_td_defaults())
        return new{typeof(wb), typeof(sbgt), typeof(gbgt), typeof(lt), typeof(st),
                   typeof(lcse), typeof(cte), typeof(gcarde), typeof(sgcarde), typeof(smtx),
                   typeof(sgmtx), typeof(slt), typeof(sst), typeof(sglt), typeof(sgst),
                   typeof(tn), typeof(sets), typeof(tr), typeof(ple), typeof(ccnt),
                   typeof(cobj), typeof(ss), typeof(card), typeof(scard), typeof(l2c),
                   typeof(lpc), typeof(linfc), typeof(l1), typeof(l2), typeof(lp),
                   typeof(linf), typeof(xbgt)}(wb, sbgt, gbgt, lt, st, lcse, cte, gcarde,
                                               sgcarde, smtx, sgmtx, slt, sst, sglt, sgst,
                                               tn, sets, tr, ple, ccnt, cobj, ss, card,
                                               scard, l2c, lpc, linfc, l1, l2, lp, linf,
                                               xbgt)
    end
end
function ConstraintBlock(; wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                         sbgt::TD_Option{<:Num_BgtRg} = nothing,
                         gbgt::TD_Option{<:Num_BgtRg} = nothing,
                         lt::TD_Option{<:BtE_Bt} = nothing,
                         st::TD_Option{<:BtE_Bt} = nothing,
                         lcse::TD_Option{<:Union{EcE_LcE_Lc, AbstractVector}} = nothing,
                         cte::TD_Option{<:Lc_CC_VecCC} = nothing,
                         gcarde::TD_Option{<:LcE_Lc} = nothing,
                         sgcarde::TD_Option{<:Union{LcE_Lc, AbstractVector}} = nothing,
                         smtx::TD_Option{<:Union{MatNum_ASetMatE, AbstractVector}} = nothing,
                         sgmtx::TD_Option{<:Union{MatNum_ASetMatE, AbstractVector}} = nothing,
                         slt::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
                         sst::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
                         sglt::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
                         sgst::TD_Option{<:Union{BtE_Bt, AbstractVector}} = nothing,
                         tn::TD_Option{<:Union{TnE_Tn, AbstractVector}} = nothing,
                         sets::TD_Option{<:UniverseSets} = nothing,
                         tr::TD_Option{<:Tr_VecTr} = nothing,
                         ple::TD_Option{<:Union{PlCE_PlC, AbstractVector}} = nothing,
                         ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr} = nothing,
                         cobj::TD_Option{<:JuMPObj_VecJuMPObj} = nothing,
                         ss::TD_Option{<:Number} = nothing,
                         card::TD_Option{<:Integer} = nothing,
                         scard::TD_Option{<:Int_VecInt} = nothing,
                         l2c::TD_Option{<:Num_NormCeilCal} = nothing,
                         lpc::TD_Option{<:LpReg_VecLpReg} = nothing,
                         linfc::TD_Option{<:Num_NormCeilCal} = nothing,
                         l1::TD_Option{<:Num_AmbRadCal} = nothing,
                         l2::TD_Option{<:L2Reg_VecL2Reg} = nothing,
                         lp::TD_Option{<:LpReg_VecLpReg} = nothing,
                         linf::TD_Option{<:Num_AmbRadCal} = nothing,
                         xbgt::Bool = false)::ConstraintBlock
    return ConstraintBlock(wb, sbgt, gbgt, lt, st,
                           narrow_mixed_vector(lcse, EcE_LcE_Lc, :lcse), cte, gcarde,
                           narrow_mixed_vector(sgcarde, LcE_Lc, :sgcarde),
                           narrow_mixed_vector(smtx, MatNum_ASetMatE, :smtx),
                           narrow_mixed_vector(sgmtx, MatNum_ASetMatE, :sgmtx),
                           narrow_mixed_vector(slt, Option{<:BtE_Bt}, :slt),
                           narrow_mixed_vector(sst, Option{<:BtE_Bt}, :sst),
                           narrow_mixed_vector(sglt, Option{<:BtE_Bt}, :sglt),
                           narrow_mixed_vector(sgst, Option{<:BtE_Bt}, :sgst),
                           narrow_mixed_vector(tn, TnE_Tn, :tn), sets, tr,
                           narrow_mixed_vector(ple, PlCE_PlC, :ple), ccnt, cobj, ss, card,
                           scard, l2c, lpc, linfc, l1, l2, lp, linf, xbgt)
end
"""
    constraint_block(cbe::Nothing; slots...) -> ConstraintBlock
    constraint_block(cbe::ConstraintBlock; slots...) -> ConstraintBlock

Returns the [`ConstraintBlock`](@ref) that the keyword constructor of an owner holds, from the block that the caller gives and the slots that the caller names as flat keywords.

The keyword constructors of [`JuMPOptimiser`](@ref) and [`ProgrammeAllocationSet`](@ref) call it. A flat keyword replaces that slot of a given block, and a slot that the caller does not name keeps its value. So the default of a flat keyword never overwrites a slot of a given block. [`rebuild_estimator`](@ref) and `Accessors.set` pass the block and a new slot in one call, and they rely on this rule.

# Algorithm

 1. `cbe` is `nothing`: build the block from `slots` with the keyword constructor of [`ConstraintBlock`](@ref). A slot that `slots` does not name takes its default.
 2. `slots` is empty: return `cbe`.
 3. Otherwise rebuild `cbe` with the slots in `slots` replaced, with [`rebuild_estimator`](@ref).

# Arguments

  - `cbe`: The block that the caller gives, or `nothing` when the caller gives none.
  - `slots...`: The slots that the caller names, as keywords of [`ConstraintBlock`](@ref).

# Validation

  - A name of `slots` that is not a slot of [`ConstraintBlock`](@ref) raises a `MethodError`, as an unknown keyword of a constructor does.

# Returns

  - `cbe::ConstraintBlock`: The block of the owner.

# Related

  - [`ConstraintBlock`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`rebuild_estimator`](@ref)
"""
function constraint_block(::Nothing; slots...)::ConstraintBlock
    return ConstraintBlock(; slots...)
end
function constraint_block(cbe::ConstraintBlock; slots...)::ConstraintBlock
    return isempty(slots) ? cbe : rebuild_estimator(cbe, values(slots))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns every slot of a [`ConstraintBlock`](@ref) as a `NamedTuple`, in the order of the fields.

The constructor of [`JuMPOptimiser`](@ref) merges this tuple with its own fields, and checks the schedules of the owner and of its block as flat keywords, see [`assert_time_dependent_substitution`](@ref). So a check that reads a field of the owner and a slot of the block runs once for each entry of a schedule in either.

# Arguments

  - `cbe::ConstraintBlock`: The block.

# Returns

  - `NamedTuple`: The slots of `cbe`, under their field names.

# Related

  - [`ConstraintBlock`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function constraint_block_slots(cbe::ConstraintBlock)::NamedTuple
    T = typeof(cbe)
    return NamedTuple{fieldnames(T)}(ntuple(i -> getfield(cbe, i), Val(fieldcount(T))))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns `true` if a slot of `cbe` reads the portfolio weights of the previous fold.

The function asks `tn`, `tr`, `ccnt` and `cobj`. It also asks every slot that holds a [`TimeDependent`](@ref) that the fold loop resolves, see [`time_dependent_fields`](@ref).

# Arguments

  - `cbe::ConstraintBlock`: The block.

# Returns

  - `Bool`: `true` if one of these slots reads the previous weights, and `false` otherwise.

# Related

  - [`ConstraintBlock`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(cbe::ConstraintBlock)::Bool
    return (needs_previous_weights(cbe.tn) ||
            needs_previous_weights(cbe.tr) ||
            needs_previous_weights(cbe.ccnt) ||
            needs_previous_weights(cbe.cobj) ||
            any(f -> needs_previous_weights(getfield(cbe, f)), time_dependent_fields(cbe)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `cbe` whose slots that read previous weights hold the weights `w`.

The function calls `factory(x, w)` on `tn`, `tr`, `ccnt` and `cobj`, and it keeps every other slot. A [`TimeDependent`](@ref) in one of these slots applies `factory` to each of its entries. The keyword constructor of [`ConstraintBlock`](@ref) checks the new slots.

# Arguments

  - `cbe::ConstraintBlock`: The block.
  - `w::AbstractVector`: The new portfolio weights.

# Returns

  - `ConstraintBlock`: The block with the four slots rebuilt for `w`.

# Related

  - [`ConstraintBlock`](@ref)
  - [`factory`](@ref)
"""
function factory(cbe::ConstraintBlock, w::AbstractVector)::ConstraintBlock
    return rebuild_estimator(cbe,
                             (; tn = factory(cbe.tn, w), tr = factory(cbe.tr, w),
                              ccnt = factory(cbe.ccnt, w), cobj = factory(cbe.cobj, w)))
end
"""
    obs_weights_view(cbe::ConstraintBlock, i) -> ConstraintBlock

Return a copy of `cbe` whose slots hold their observation weights cut to the observations `i`.

The block is not [`@propagatable`](@ref), so it needs this method of [`obs_weights_view`](@ref). Its phylogeny and centrality estimators and its tracking errors read the observations of the sample, and each can hold fixed observation weights. The method of its owner reaches the block through its field `cbe`, see `obs_weights_view(opt::JuMPOptimiser, i)`.

# Algorithm

 1. Cut every slot of `cbe` with [`obs_weights_view`](@ref), and rebuild the block with its keyword constructor, through [`obs_weights_fields`](@ref).

# Arguments

  - `cbe::ConstraintBlock`: The block.
  - `i`: The observations to keep: a mask of the full sample, or their positions.

# Returns

  - `ConstraintBlock`: The block at the kept observations.

# Related

  - [`ConstraintBlock`](@ref)
  - [`obs_weights_view`](@ref)
"""
function obs_weights_view(cbe::ConstraintBlock, i)::ConstraintBlock
    return obs_weights_fields(cbe, i)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `cbe` restricted to the assets at the indices `i`.

The method of the owner calls it, see `port_opt_view(opt::JuMPOptimiser, i, X)` and `port_opt_view(set::ProgrammeAllocationSet, i, args...)`. A [`JuMPOptimiser`](@ref) passes the returns matrix `X`. A [`ProgrammeAllocationSet`](@ref) passes the arguments that it takes after `i`, and an online head can pass none.

# Algorithm

 1. View at `i` the slots that can hold a value for each asset: `wb`, `lt`, `st`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn`, `lcse`, `sets`, `ccnt` and `cobj`. When `smtx` and `sgmtx` hold the same object, view it once, so that the two views are one object too. The same holds for `slt` and `sglt`, and for `sst` and `sgst`.
 2. View `tr` at `i` with `args`. A tracking measure gives the returns matrix `X` in `args` to the risk measure that it tracks.
 3. Keep every other slot. An estimator in `cte`, `gcarde` or `sgcarde` resolves its names against the viewed `sets`. A precomputed constraint in one of them keeps its width, because no view can narrow its matrix, see `port_opt_view(::LinearConstraint, i)`. `ple` reads the viewed prior of the owner. The other slots hold no value for each asset.
 4. Rebuild the block with its keyword constructor, which checks the viewed slots again.

# Arguments

  - `cbe::ConstraintBlock`: The block.
  - `i`: The indices of the assets that the view keeps.
  - `args...`: The arguments that the tracking errors `tr` read, such as the returns matrix `X` of the full universe, with one column for each asset.

# Returns

  - `ConstraintBlock`: The block restricted to the assets at `i`.

# Related

  - [`ConstraintBlock`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(cbe::ConstraintBlock, i, args...)::ConstraintBlock
    smtx, sgmtx = shared_pair_view(cbe.smtx, cbe.sgmtx, i)
    slt, sglt = shared_pair_view(cbe.slt, cbe.sglt, i)
    sst, sgst = shared_pair_view(cbe.sst, cbe.sgst, i)
    return rebuild_estimator(cbe,
                             (; wb = port_opt_view(cbe.wb, i),
                              lt = port_opt_view(cbe.lt, i), st = port_opt_view(cbe.st, i),
                              lcse = port_opt_view(cbe.lcse, i), smtx, sgmtx, slt, sst,
                              sglt, sgst, tn = port_opt_view(cbe.tn, i),
                              sets = port_opt_view(cbe.sets, i),
                              tr = port_opt_view(cbe.tr, i, args...),
                              ccnt = port_opt_view(cbe.ccnt, i),
                              cobj = port_opt_view(cbe.cobj, i)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the views at `i` of two slots of a [`ConstraintBlock`](@ref) that can hold the same object.

A subgroup slot and its subgrouped twin, such as `smtx` and `sgmtx`, often hold one object. When they do, the function views it once, so that the two views are one object too, and the check of the shapes of the two slots sees one object.

# Arguments

  - `a`, `b`: The values of the two slots.
  - `i`: The indices of the assets that the view keeps.

# Returns

  - `(va, vb)::Tuple`: The views of `a` and `b`.

# Related

  - `port_opt_view(cbe::ConstraintBlock, i, X)`
  - [`port_opt_view`](@ref)
"""
function shared_pair_view(a, b, i)::Tuple
    if a === b
        v = port_opt_view(a, i)
        return v, v
    end
    return port_opt_view(a, i), port_opt_view(b, i)
end
"""
    reader_frames(cbe::ConstraintBlock, path::AbstractString, refit::Bool) -> ReturnsFrames

Walk the readers in the slots of a [`ConstraintBlock`](@ref), for the frame walk of the head that holds it.

The graph readers `cte` and `ple`, the regressions of `lcse` and the readers inside the tracking errors `tr` belong to the group of the head. So the walk of [`field_frames`](@ref) enters the block, at the path of its field `cbe`, as it enters every other field of the owner.

# Arguments

  - `cbe::ConstraintBlock`: The block.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the slots of `cbe`.

# Related

  - [`field_frames`](@ref)
  - [`reader_frames`](@ref)
"""
function reader_frames(cbe::ConstraintBlock, path::AbstractString, refit::Bool)
    return field_frames(cbe, path, refit)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the data that a resolver gives once for each `optimise` call.

A subtype holds the results that a configuration resolves to against the prior and the rows of one fit. A builder of the model reads them, and no builder resolves the configuration again. [`ResolvedConstraintBlock`](@ref) is the Result of one [`ConstraintBlock`](@ref), [`ResolvedJuMPOptimiser`](@ref) is the Result of a JuMP head, and the subtypes of [`AbstractResolvedRiskBudgeting`](@ref) hold the resolved budget of a risk budgeting model.

# Related

  - [`ResolvedConstraintBlock`](@ref)
  - [`ResolvedJuMPOptimiser`](@ref)
  - [`AbstractResolvedRiskBudgeting`](@ref)
"""
abstract type AbstractResolved <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Holds the Resolved Constraint Block of the glossary: the data that a [`ConstraintBlock`](@ref) gives when it resolves against its own prior and rows.

[`resolve_constraint_block`](@ref) makes it at stage 2 of the resolution, and a builder adds it to a programme. The block builder [`set_block_constraints!`](@ref) reads it for both owners: a JuMP head in [`assemble_jump_model!`](@ref), and a [`ProgrammeAllocationSet`](@ref) in [`add_allocation_set_block!`](@ref). So no builder resolves a slot again.

The Result holds three kinds of data:

 1. The prior result `pr` and the rows `rd` of its block. A block that reads no rows holds `nothing` in both.
 2. The resolved slots. Each field holds the result of the slot of the block, and the five slots that change their name take the `r` of a result: `lcse` gives `lcsr`, `cte` gives `ctr`, `gcarde` gives `gcardr`, `sgcarde` gives `sgcardr` and `ple` gives `plr`. A Calibration Rule in `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp` or `linf` is resolved to its value.
 3. The settings that need no resolution: `card`, `scard`, `ss`, `xbgt`, `sbgt`, `gbgt`, `ccnt` and `cobj`.

The tracking errors `tr` and the names `sets` are not fields. The builder of the tracking errors binds them to the rows of the Result, and no builder reads `sets`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ResolvedConstraintBlock(; pr, rd, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx, slt,
                            sst, sglt, sgst, tn, plr, l2c, lpc, linfc, l1, l2, lp, linf, card,
                            scard, ss, xbgt, sbgt, gbgt, ccnt, cobj) -> ResolvedConstraintBlock

Keywords correspond to the struct's fields, and each keyword has no default. Make the Result with [`resolve_constraint_block`](@ref). The constructor checks the type of each field, so a slot that holds an estimator raises a `TypeError`. It checks no value, because the resolution checked the values.

# Related

  - [`ConstraintBlock`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`resolve_constraint_block_names`](@ref)
  - [`ResolvedJuMPOptimiser`](@ref)
"""
@concrete struct ResolvedConstraintBlock <: AbstractResolved
    """
    $(field_dict[:pr]) The prior of the block, or `nothing` when the block reads no rows.
    """
    pr
    """
    $(field_dict[:rd]) The rows of the block, or `nothing` when the block reads no rows.
    """
    rd
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:lt])
    """
    lt
    """
    $(field_dict[:st])
    """
    st
    """
    $(field_dict[:lcsr])
    """
    lcsr
    """
    $(field_dict[:ctr])
    """
    ctr
    """
    $(field_dict[:gcardr])
    """
    gcardr
    """
    $(field_dict[:sgcardr])
    """
    sgcardr
    """
    $(field_dict[:smtx])
    """
    smtx
    """
    $(field_dict[:sgmtx])
    """
    sgmtx
    """
    $(field_dict[:slt])
    """
    slt
    """
    $(field_dict[:sst])
    """
    sst
    """
    $(field_dict[:sglt])
    """
    sglt
    """
    $(field_dict[:sgst])
    """
    sgst
    """
    $(field_dict[:tnr])
    """
    tn
    """
    $(field_dict[:plr])
    """
    plr
    """
    $(field_dict[:l2c])
    """
    l2c
    """
    $(field_dict[:lpc])
    """
    lpc
    """
    $(field_dict[:linfc])
    """
    linfc
    """
    $(field_dict[:l1])
    """
    l1
    """
    $(field_dict[:l2])
    """
    l2
    """
    $(field_dict[:lp])
    """
    lp
    """
    $(field_dict[:linf])
    """
    linf
    """
    $(field_dict[:card])
    """
    card
    """
    $(field_dict[:scard])
    """
    scard
    """
    $(field_dict[:ss])
    """
    ss
    """
    $(field_dict[:xbgt])
    """
    xbgt
    """
    $(field_dict[:sbgt])
    """
    sbgt
    """
    $(field_dict[:gbgt])
    """
    gbgt
    """
    $(field_dict[:ccnt])
    """
    ccnt
    """
    $(field_dict[:cobj])
    """
    cobj
    function ResolvedConstraintBlock(pr::Option{<:AbstractPriorResult},
                                     rd::Option{<:ReturnsResult},
                                     wb::Option{<:WeightBounds}, lt::Option{<:Threshold},
                                     st::Option{<:Threshold}, lcsr::Option{<:Lc_VecLc},
                                     ctr::Option{<:Lc_CC_VecCC},
                                     gcardr::Option{<:LinearConstraint},
                                     sgcardr::Option{<:Lc_VecLc},
                                     smtx::Option{<:MatNum_VecMatNum},
                                     sgmtx::Option{<:MatNum_VecMatNum},
                                     slt::Option{<:Bt_VecOptBt}, sst::Option{<:Bt_VecOptBt},
                                     sglt::Option{<:Bt_VecOptBt},
                                     sgst::Option{<:Bt_VecOptBt}, tn::Option{<:Tn_VecTn},
                                     plr::Option{<:PlC_VecPlC}, l2c::Option{<:Number},
                                     lpc::Option{<:LpReg_VecLpReg}, linfc::Option{<:Number},
                                     l1::Option{<:Number}, l2::Option{<:L2Reg_VecL2Reg},
                                     lp::Option{<:LpReg_VecLpReg}, linf::Option{<:Number},
                                     card::Option{<:Integer}, scard::Option{<:Int_VecInt},
                                     ss::Option{<:Number}, xbgt::Bool,
                                     sbgt::Option{<:Num_BgtRg}, gbgt::Option{<:Num_BgtRg},
                                     ccnt::Option{<:JuMPConstr_VecJuMPConstr},
                                     cobj::Option{<:JuMPObj_VecJuMPObj})
        return new{typeof(pr), typeof(rd), typeof(wb), typeof(lt), typeof(st), typeof(lcsr),
                   typeof(ctr), typeof(gcardr), typeof(sgcardr), typeof(smtx),
                   typeof(sgmtx), typeof(slt), typeof(sst), typeof(sglt), typeof(sgst),
                   typeof(tn), typeof(plr), typeof(l2c), typeof(lpc), typeof(linfc),
                   typeof(l1), typeof(l2), typeof(lp), typeof(linf), typeof(card),
                   typeof(scard), typeof(ss), typeof(xbgt), typeof(sbgt), typeof(gbgt),
                   typeof(ccnt), typeof(cobj)}(pr, rd, wb, lt, st, lcsr, ctr, gcardr,
                                               sgcardr, smtx, sgmtx, slt, sst, sglt, sgst,
                                               tn, plr, l2c, lpc, linfc, l1, l2, lp, linf,
                                               card, scard, ss, xbgt, sbgt, gbgt, ccnt,
                                               cobj)
    end
end
function ResolvedConstraintBlock(; pr::Option{<:AbstractPriorResult},
                                 rd::Option{<:ReturnsResult}, wb::Option{<:WeightBounds},
                                 lt::Option{<:Threshold}, st::Option{<:Threshold},
                                 lcsr::Option{<:Lc_VecLc}, ctr::Option{<:Lc_CC_VecCC},
                                 gcardr::Option{<:LinearConstraint},
                                 sgcardr::Option{<:Lc_VecLc},
                                 smtx::Option{<:MatNum_VecMatNum},
                                 sgmtx::Option{<:MatNum_VecMatNum},
                                 slt::Option{<:Bt_VecOptBt}, sst::Option{<:Bt_VecOptBt},
                                 sglt::Option{<:Bt_VecOptBt}, sgst::Option{<:Bt_VecOptBt},
                                 tn::Option{<:Tn_VecTn}, plr::Option{<:PlC_VecPlC},
                                 l2c::Option{<:Number}, lpc::Option{<:LpReg_VecLpReg},
                                 linfc::Option{<:Number}, l1::Option{<:Number},
                                 l2::Option{<:L2Reg_VecL2Reg}, lp::Option{<:LpReg_VecLpReg},
                                 linf::Option{<:Number}, card::Option{<:Integer},
                                 scard::Option{<:Int_VecInt}, ss::Option{<:Number},
                                 xbgt::Bool, sbgt::Option{<:Num_BgtRg},
                                 gbgt::Option{<:Num_BgtRg},
                                 ccnt::Option{<:JuMPConstr_VecJuMPConstr},
                                 cobj::Option{<:JuMPObj_VecJuMPObj})::ResolvedConstraintBlock
    return ResolvedConstraintBlock(pr, rd, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx,
                                   sgmtx, slt, sst, sglt, sgst, tn, plr, l2c, lpc, linfc,
                                   l1, l2, lp, linf, card, scard, ss, xbgt, sbgt, gbgt,
                                   ccnt, cobj)
end
"""
    const VecRcb = AbstractVector{<:ResolvedConstraintBlock}

Every abstract vector whose elements are [`ResolvedConstraintBlock`](@ref)s. A programme with more than one Constraint Block holds one Result for each block.

# Related

  - [`ResolvedConstraintBlock`](@ref)
  - [`Rcb_VecRcb`](@ref)
"""
const VecRcb = AbstractVector{<:ResolvedConstraintBlock}
"""
    const Rcb_VecRcb = Union{ResolvedConstraintBlock, VecRcb}

One [`ResolvedConstraintBlock`](@ref), or a vector of them. The field `cbr` of [`ResolvedJuMPOptimiser`](@ref) takes this type, as `plr` takes one phylogeny Result or a vector.

# Related

  - [`ResolvedConstraintBlock`](@ref)
  - [`VecRcb`](@ref)
  - [`ResolvedJuMPOptimiser`](@ref)
"""
const Rcb_VecRcb = Union{ResolvedConstraintBlock, VecRcb}
"""
    owner_block(cbr::ResolvedConstraintBlock) -> ResolvedConstraintBlock
    owner_block(cbr::VecRcb) -> ResolvedConstraintBlock

Gives the block Result of the owner of a programme, the first block of the programme.

The field `cbr` of [`ResolvedJuMPOptimiser`](@ref) holds one block Result, or one for each block when the head has more than one block. The block of the owner is the first. The readers of the bundle that mean the block of the head read it through this function: the forward rule of the bundle, `jor.wb`, the builders of [`assemble_jump_model!`](@ref), and [`jump_optimiser_from_attributes`](@ref).

# Arguments

  - `cbr`: One block Result, or a vector of them with the block of the owner first.

# Returns

  - `cbr::ResolvedConstraintBlock`: The block Result of the owner.

# Examples

```jldoctest
julia> pr = prior(EmpiricalPrior(),
                  ReturnsResult(; nx = [\"a\", \"b\"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1]));

julia> cbr = PortfolioOptimisers.resolve_constraint_block(ConstraintBlock(), pr,
                                                          ReturnsResult(; nx = [\"a\", \"b\"],
                                                                        X = pr.X); slv = nothing);

julia> PortfolioOptimisers.owner_block([cbr, cbr]) === PortfolioOptimisers.owner_block(cbr)
true
```

# Related

  - [`Rcb_VecRcb`](@ref)
  - [`ResolvedJuMPOptimiser`](@ref)
  - [`resolve_jump_optimiser`](@ref)
"""
function owner_block(cbr::ResolvedConstraintBlock)::ResolvedConstraintBlock
    return cbr
end
function owner_block(cbr::VecRcb)::ResolvedConstraintBlock
    return first(cbr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves the slots of a [`ConstraintBlock`](@ref) that hold a name or an estimator over `sets`. This is stage 1 of the resolution.

Stage 1 is configuration. It gives the block with values in its slots, and it holds no Result. It runs before the reduction to the Investable Mask, over the universe of the caller, where every name of `sets` is an asset of the caller. The reduction then views each value at the mask, see `port_opt_view(cbe::ConstraintBlock, i, args...)`. A JuMP head runs it once for each `optimise` call, and an online head runs it once for each fold, in [`resolve_allocation_set`](@ref).

A slot is a stage 1 slot when its value follows a view at the mask, so the two orders of the names and the reduction give one value. The other slots resolve at stage 2, in [`resolve_constraint_block`](@ref), after the reduction:

  - `wb` and `lcse`, because their resolvers read the rows `rd`.
  - `gcarde`, because it resolves to a [`LinearConstraint`](@ref) over the assets, and a view does not narrow its matrix. At stage 2 it resolves over the `sets` that carry the Non-Investable Axis.
  - `cte`, `ple` and the slots that can hold a Calibration Rule, because they read the prior.

# Algorithm

 1. Resolve the thresholds `lt` and `st` with [`threshold_constraints`](@ref).
 2. Resolve the sub-grouped cardinality constraint `sgcarde` with [`linear_constraints`](@ref), with integer values. Its columns are the sub-groups and not the assets, so a view does not change it.
 3. Resolve the sub-group matrices `smtx` and `sgmtx` with [`asset_sets_matrix`](@ref), and the sub-group thresholds `slt`, `sst`, `sglt` and `sgst` with [`threshold_constraints`](@ref). When the two slots of a pair hold one object, resolve it once with [`shared_pair_resolution`](@ref).
 4. Resolve the turnover constraints `tn` with [`turnover_constraints`](@ref).
 5. Return `cbe` rebuilt with the resolved slots, with [`rebuild_estimator`](@ref).

# Arguments

  - `cbe`: The block.
  - $(arg_dict[:strict])
  - `datatype`: The element type of the resolved values.

# Returns

  - `cbe'::ConstraintBlock`: The block with its stage 1 slots resolved.

# Related

  - [`resolve_constraint_block`](@ref)
  - [`ConstraintBlock`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`resolve_allocation_set`](@ref)
"""
function resolve_constraint_block_names(cbe::ConstraintBlock, strict::Bool,
                                        datatype::DataType)::ConstraintBlock
    sets = cbe.sets
    thr = x -> threshold_constraints(x, sets; datatype = datatype, strict = strict)
    smtx, sgmtx = shared_pair_resolution(x -> asset_sets_matrix(x, sets), cbe.smtx,
                                         cbe.sgmtx)
    slt, sglt = shared_pair_resolution(thr, cbe.slt, cbe.sglt)
    sst, sgst = shared_pair_resolution(thr, cbe.sst, cbe.sgst)
    return rebuild_estimator(cbe,
                             (; lt = thr(cbe.lt), st = thr(cbe.st),
                              sgcarde = linear_constraints(cbe.sgcarde, sets;
                                                           datatype = Int, strict = strict),
                              smtx, sgmtx, slt, sst, sglt, sgst,
                              tn = turnover_constraints(cbe.tn, sets; datatype = datatype,
                                                        strict = strict)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves two slots of a [`ConstraintBlock`](@ref) that can hold the same object, with one resolver.

When the two slots hold one object, the function resolves it once, so the two results are one object too. The check of the shapes of a sub-group slot and its sub-grouped twin then sees one object, as [`shared_pair_view`](@ref) keeps it for a view.

# Arguments

  - `f`: The resolver of one slot.
  - `a`, `b`: The values of the two slots.

# Returns

  - `(ra, rb)::Tuple`: The results of `a` and `b`.

# Related

  - [`resolve_constraint_block_names`](@ref)
  - [`shared_pair_view`](@ref)
"""
function shared_pair_resolution(f, a, b)::Tuple
    if a === b
        v = f(a)
        return v, v
    end
    return f(a), f(b)
end
"""
    resolve_constraint_block(cbe::ConstraintBlock, pr::AbstractPriorResult, rd::ReturnsResult;
                             slv, strict::Bool = false, kwargs...) -> ResolvedConstraintBlock
    resolve_constraint_block(cbe::ConstraintBlock, pr::Nothing, rd::Option{<:ReturnsResult};
                             strict::Bool = false, kwargs...) -> ResolvedConstraintBlock
    resolve_constraint_block(cbe::ConstraintBlock, pr::AbstractPriorResult, rd::Nothing;
                             kwargs...)

Resolves a [`ConstraintBlock`](@ref) against its own prior and rows, to its [`ResolvedConstraintBlock`](@ref). This is stage 2 of the resolution.

Both owners of a block call it. [`resolve_jump_optimiser`](@ref) calls it once for each `optimise` call, and [`resolve_allocation_set_step`](@ref) calls it at each step of an online head. The block must hold the result of stage 1, [`resolve_constraint_block_names`](@ref), viewed at the Investable Mask of `pr`. So `pr`, `rd` and the block are on the investable assets. A slot that still holds an estimator of stage 1 raises a `TypeError` in the constructor of the Result.

One rule orders the names and the reduction for every block. Stage 1 resolves before the reduction, and a view at the mask follows its values. This function resolves after the reduction, over the `sets` of the block, which carry the Non-Investable Axis. So a name that left the investable universe drops its row and is not refused, and a precomputed constraint keeps its width. [`assert_investable_constraint_width`](@ref) then refuses a precomputed `lcse` or `gcarde` whose width is not the number of assets of the block.

The method for `pr = nothing` is the method of a programme set that reads no rows, see [`rows_needed`](@ref). Its prior-reading slots hold values that read no prior, so it resolves the slots that read names alone, and keeps every other slot. Its Result holds no prior and no rows.

# Algorithm

The method on a prior result runs these steps.

 1. Resolve the weight bounds `wb` with [`weight_bounds_constraints`](@ref), with the rows `rd` and the `N` assets of `pr.X`.
 2. Resolve `lcse` with [`linear_constraints`](@ref), with the factor regression `pr.rr` and the rows, and `gcarde` with integer values. Check the width of both results with [`assert_investable_constraint_width`](@ref).
 3. Fit the centrality rows `ctr` with [`centrality_constraints`](@ref) and the phylogeny result `plr` with [`phylogeny_constraints`](@ref). Each reads the prior, the rows and the implied volatilities `rd.iv` and `rd.ivpa`. A reader that cannot estimate an asset raises a [`DistanceReaderRefusalError`](@ref), which the caller catches.
 4. Resolve each Calibration Rule against the prior and the solver `slv`. The norm ceilings `l2c` and `linfc` state their norm order in a [`CalibrationContext`](@ref), and `lpc` goes through [`norm_ceiling_factory`](@ref). The penalties `l1` and `linf` go through [`resolve_calibration_slot`](@ref), and `l2` and `lp` through their own `factory`.
 5. Check the four slots that hold a number, `l2c`, `linfc`, `l1` and `linf`, with [`assert_nonempty_gt0_finite_val`](@ref). A rule states no number at construction, so this is the first point that knows the number.
 6. Return the Result, with the stage 1 slots and the settings of `cbe`, `pr` and `rd`.

# Arguments

  - `cbe`: The block, after stage 1 and the view at the mask.
  - `pr`: The prior result of the block, or `nothing` when the block reads no rows.
  - `rd`: The rows of the block. The method for `pr = nothing` reads only their element type and their width, and it takes `nothing` outside an Online Update.
  - `slv`: The solver of the owner, which a Calibration Rule resolves against.
  - $(arg_dict[:strict])
  - `kwargs...`: Passed to [`centrality_constraints`](@ref) and [`phylogeny_constraints`](@ref).

# Validation

  - The rules of [`assert_investable_constraint_width`](@ref) for `lcsr` and `gcardr`. The method for `pr = nothing` checks them when it has rows.
  - `l2c`, `linfc`, `l1` and `linf` are `nothing` or finite and `> 0` after the resolution. A `DomainError` or an `IsNonFiniteError` is thrown otherwise.
  - A prior result with no rows. An `IsNothingError` is thrown. The prior of a block is fitted on its rows, so this pair does not occur in the library.

# Returns

  - `cbr::ResolvedConstraintBlock`: The Result of the block.

# Related

  - [`ResolvedConstraintBlock`](@ref)
  - [`resolve_constraint_block_names`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`resolve_allocation_set_step`](@ref)
"""
function resolve_constraint_block(cbe::ConstraintBlock, pr::AbstractPriorResult,
                                  rd::ReturnsResult; slv, strict::Bool = false,
                                  kwargs...)::ResolvedConstraintBlock
    datatype = eltype(pr.X)
    N = size(pr.X, 2)
    sets = cbe.sets
    lcsr = linear_constraints(cbe.lcse, sets; datatype = datatype, strict = strict,
                              rr = pr.rr, rd = rd)
    gcardr = linear_constraints(cbe.gcarde, sets; datatype = Int, strict = strict)
    assert_investable_constraint_width(lcsr, N, "lcse")
    assert_investable_constraint_width(gcardr, N, "gcarde")
    l2c = resolve_calibration_slot(cbe.l2c, :l2c, pr, pr.w, slv,
                                   CalibrationContext(; p = 2))
    linfc = resolve_calibration_slot(cbe.linfc, :linfc, pr, pr.w, slv,
                                     CalibrationContext(; p = Inf))
    l1 = resolve_calibration_slot(cbe.l1, :l1, pr, pr.w, slv)
    linf = resolve_calibration_slot(cbe.linf, :linf, pr, pr.w, slv)
    # A rule states no number at construction, so the four numbers that reach the model raw
    # are checked here, where the rule first gives them.
    map(((l2c, :l2c), (linfc, :linfc), (l1, :l1), (linf, :linf))) do (x, name)
        return assert_nonempty_gt0_finite_val(x, name)
    end
    vals = (;
            wb = weight_bounds_constraints(cbe.wb, sets; rd = rd, N = N, strict = strict,
                                           datatype = datatype), lcsr, gcardr,
            ctr = centrality_constraints(cbe.cte, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd,
                                         strict = strict, kwargs...),
            plr = phylogeny_constraints(cbe.ple, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd,
                                        kwargs...), l2c,
            lpc = norm_ceiling_factory(cbe.lpc, pr, slv), linfc, l1,
            l2 = factory(cbe.l2, pr, slv), lp = factory(cbe.lp, pr, slv), linf)
    return resolved_constraint_block(cbe, pr, rd, vals)
end
function resolve_constraint_block(::ConstraintBlock, ::AbstractPriorResult, ::Nothing;
                                  kwargs...)
    return throw(IsNothingError("the prior result of a block is fitted on its rows, so a block with a prior and no rows cannot resolve the slots that read the rows."))
end
function resolve_constraint_block(cbe::ConstraintBlock, ::Nothing,
                                  rd::Option{<:ReturnsResult}; strict::Bool = false,
                                  kwargs...)::ResolvedConstraintBlock
    sets = cbe.sets
    lcsr = linear_constraints(cbe.lcse, sets; strict = strict, rows_datatype(rd)...)
    gcardr = linear_constraints(cbe.gcarde, sets; datatype = Int, strict = strict)
    assert_rows_constraint_width(lcsr, gcardr, rd)
    vals = (; wb = cbe.wb, lcsr, gcardr, ctr = cbe.cte, plr = cbe.ple, l2c = cbe.l2c,
            lpc = cbe.lpc, linfc = cbe.linfc, l1 = cbe.l1, l2 = cbe.l2, lp = cbe.lp,
            linf = cbe.linf)
    return resolved_constraint_block(cbe, nothing, nothing, vals)
end
"""
    rows_datatype(rd::Nothing)
    rows_datatype(rd::ReturnsResult)

Returns the element type of the rows of a block that reads no prior, as a keyword of a resolver.

A block with no prior and no rows passes no keyword, so the resolver takes its own default.

# Returns

  - `kw::NamedTuple`: `(; datatype = eltype(rd.X))`, or an empty tuple when `rd` is `nothing`.

# Related

  - [`resolve_constraint_block`](@ref)
"""
function rows_datatype(::Nothing)::NamedTuple
    return (;)
end
function rows_datatype(rd::ReturnsResult)::NamedTuple
    return (; datatype = eltype(rd.X))
end
"""
    assert_rows_constraint_width(lcsr, gcardr, rd::Nothing)
    assert_rows_constraint_width(lcsr, gcardr, rd::ReturnsResult)

Checks the width of the linear constraints of a block that reads no prior, against the columns of its rows.

A block with no rows has no width to check against, and the model refuses a wrong width.

# Validation

  - The rules of [`assert_investable_constraint_width`](@ref), with `N = size(rd.X, 2)`.

# Returns

  - `nothing`.

# Related

  - [`resolve_constraint_block`](@ref)
"""
function assert_rows_constraint_width(::Any, ::Any, ::Nothing)::Nothing
    return nothing
end
function assert_rows_constraint_width(lcsr, gcardr, rd::ReturnsResult)::Nothing
    N = size(rd.X, 2)
    assert_investable_constraint_width(lcsr, N, "lcse")
    assert_investable_constraint_width(gcardr, N, "gcarde")
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the [`ResolvedConstraintBlock`](@ref) from the values of stage 2 and the slots of the block that stage 2 does not resolve.

Both methods of [`resolve_constraint_block`](@ref) call it, so the list of the stage 1 slots and of the settings is written once.

# Arguments

  - `cbe`: The block, after stage 1 and the view at the mask.
  - `pr`: The prior result of the block, or `nothing`.
  - `rd`: The rows of the block, or `nothing`.
  - `vals`: The values of stage 2: `wb`, `lcsr`, `gcardr`, `ctr`, `plr`, `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp` and `linf`.

# Returns

  - `cbr::ResolvedConstraintBlock`: The Result of the block.

# Related

  - [`resolve_constraint_block`](@ref)
"""
function resolved_constraint_block(cbe::ConstraintBlock, pr, rd,
                                   vals::NamedTuple)::ResolvedConstraintBlock
    return ResolvedConstraintBlock(; pr = pr, rd = rd, vals..., lt = cbe.lt, st = cbe.st,
                                   sgcardr = cbe.sgcarde, smtx = cbe.smtx,
                                   sgmtx = cbe.sgmtx, slt = cbe.slt, sst = cbe.sst,
                                   sglt = cbe.sglt, sgst = cbe.sgst, tn = cbe.tn,
                                   card = cbe.card, scard = cbe.scard, ss = cbe.ss,
                                   xbgt = cbe.xbgt, sbgt = cbe.sbgt, gbgt = cbe.gbgt,
                                   ccnt = cbe.ccnt, cobj = cbe.cobj)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `cbe` whose slots hold the values of its [`ResolvedConstraintBlock`](@ref).

[`jump_optimiser_from_attributes`](@ref) calls it to build the optimiser of a sub-problem from the Result of a fit. The Result holds no estimator, so the sub-problem resolves no slot again: the resolvers of both stages give a resolved value back.

The function maps each field of the Result onto the slot of the same name, and five fields onto a slot with another name: `lcsr` onto `lcse`, `ctr` onto `cte`, `gcardr` onto `gcarde`, `sgcardr` onto `sgcarde` and `plr` onto `ple`. It derives the list from the fields of the Result, so it drops no field. A field with no slot raises a `MethodError` in the keyword constructor of the block. The prior `pr` and the rows `rd` have no slot, and the function leaves them out. A slot with no field, `tr` or `sets`, keeps its value in `cbe`.

# Arguments

  - `cbe`: The block of the optimiser that the sub-problem copies.
  - `cbr`: The Result of the block.

# Returns

  - `cbe'::ConstraintBlock`: The block with the values of `cbr` in its slots.

# Related

  - [`ResolvedConstraintBlock`](@ref)
  - [`jump_optimiser_from_attributes`](@ref)
"""
function constraint_block_from_resolved(cbe::ConstraintBlock,
                                        cbr::ResolvedConstraintBlock)::ConstraintBlock
    rename = (; lcsr = :lcse, ctr = :cte, gcardr = :gcarde, sgcardr = :sgcarde, plr = :ple)
    rf = filter(f -> f !== :pr && f !== :rd, fieldnames(ResolvedConstraintBlock))
    slots = NamedTuple{map(f -> get(rename, f, f), rf)}(map(f -> getfield(cbr, f), rf))
    return rebuild_estimator(cbe, slots)
end
"""
    set_block_constraints!(model::JuMP.Model, cbr::ResolvedConstraintBlock, owner, bgt, fees;
                           tr = nothing, b1::Option{<:MatNum} = nothing,
                           prefix::Symbol = Symbol(""), kwargs...) -> Nothing
    set_block_post_risk_constraints!(model::JuMP.Model, cbr::ResolvedConstraintBlock, owner,
                                     attrs; prefix::Symbol = Symbol("")) -> Nothing

Add a [`ResolvedConstraintBlock`](@ref) to a model, in its own namespace and in two phases. This is the block builder.

Every programme that holds a block adds it through these two functions: a JuMP head in [`assemble_jump_model!`](@ref), the projection of a [`ProgrammeAllocationSet`](@ref) in [`set_allocation_set_constraints!`](@ref), and the set in the model of a leader through [`add_allocation_set_constraints!`](@ref). So the order of the Model Assembly is written once. The owner of the model runs its own steps between the two phases: the fees, the risk step, the return terms and [`mark_risk_minimised!`](@ref).

Phase 1, `set_block_constraints!`, adds the rows that come before the risk step of the owner. Phase 2, `set_block_post_risk_constraints!`, adds the rows that read the marks of the objective of the owner, `risk_minimised` and `variance_flag`: the semidefinite phylogeny reads them. The integer phylogeny and the custom constraints come after them too, as they did in each owner.

The first block of a model takes the empty prefix and the names of today. Each block after the first takes a Model State namespace. Phase 1 registers the weights of the model under it, with the owner `w_owner = Symbol("")`, so [`weights_prefix`](@ref) keeps the lifted matrix and the marks with the weights. Every builder then registers its entries under the prefix. A kind on two blocks gives two sets of rows and no name collision. The MIP bits, the long and short parts of the weights and the pin of `xbgt` belong to the weights, so the blocks share them. The penalties of each block add to the one Objective Penalty.

The block reads its own prior `cbr.pr` and its own phylogeny `cbr.plr`. The builder takes the values of the owner that the block cannot hold: the net budget `bgt`, for the long-short and gross budgets, the fees, for the MIP kinds and the tracking errors, the tracking errors `tr` that the owner bound to its rows, and the factor loadings `b1` of a factor head. The custom constraints get the owner and the bundle `attrs` that the owner gives them. A block of a set passes the set as the owner, `nothing` as the net budget and the fees, and `nothing` as `attrs`.

**Breaking change.** These two functions replace the middle of `assemble_jump_model!` and the removed `assemble_allocation_set!`. A head no longer adds its weight bounds and its budgets before [`assemble_jump_model!`](@ref). Phase 1 adds them, with the net budget of the head.

# Algorithm

Phase 1:

 1. When `prefix` is not empty, register the weights of the model under it, with the owner `Symbol("")`.
 2. Add the weight bounds, the net budget `bgt`, and the short and gross budgets of the block with [`set_weight_constraints!`](@ref).
 3. Add the linear rows `lcsr`, then the centrality rows `ctr`, with [`set_linear_weight_constraints!`](@ref).
 4. Add the MIP kinds with [`set_mip_constraints!`](@ref), then the sub-group MIP kinds with [`set_smip_constraints!`](@ref).
 5. Add the turnover ceilings with [`set_turnover_constraints!`](@ref).
 6. Add the tracking errors `tr` with [`set_tracking_error_constraints!`](@ref), on the prior of the block.
 7. Add the norm ceilings `l2c`, `lpc` and `linfc`, then the penalties `l1`, `l2`, `lp` and `linf`.

Phase 2:

 1. Add the integer phylogeny rows with [`set_iplg_constraints!`](@ref), then the semidefinite phylogeny rows with [`set_sdp_phylogeny_constraints!`](@ref).
 2. Add the custom constraints `ccnt` with [`add_custom_constraint!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `cbr`: The Result of the block.
  - `owner`: The owner of the block: a JuMP head, or a [`ProgrammeAllocationSet`](@ref). The tracking errors and the custom constraints dispatch on it.
  - `bgt`: The net budget of the owner of the block, or `nothing`.
  - `fees`: The fees of the owner of the block, or `nothing`.
  - `tr`: The tracking errors of the owner, bound to its rows, or `nothing`.
  - $(arg_dict[:b1_opt])
  - `attrs`: The bundle that the custom constraints read, or `nothing`.
  - $(arg_dict[:prefix_kw])
  - `kwargs...`: Passed to [`set_tracking_error_constraints!`](@ref). A JuMP head passes its rows as `rd`.

# Returns

  - `nothing`.

# Related

  - [`ResolvedConstraintBlock`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`set_allocation_risk_return_bounds!`](@ref)
  - [`weights_prefix`](@ref)
"""
function set_block_constraints!(model::JuMP.Model, cbr::ResolvedConstraintBlock, owner,
                                bgt::Option{<:Num_BgtCE}, fees::Option{<:Fees};
                                tr = nothing, b1::Option{<:MatNum} = nothing,
                                prefix::Symbol = Symbol(""), kwargs...)::Nothing
    if prefix != Symbol("")
        state_build!(() -> get_w(model), model, prefix, :w)
        state_build!(() -> Symbol(""), model, prefix, :w_owner)
    end
    (; wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst, plr, ss) = cbr
    set_weight_constraints!(model, wb, bgt, cbr.sbgt; gbgt = cbr.gbgt, prefix = prefix)
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_; prefix = prefix)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_; prefix = prefix)
    set_mip_constraints!(model, wb, cbr.card, gcardr, plr, lt, st, fees, ss, cbr.xbgt;
                         prefix = prefix)
    set_smip_constraints!(model, wb, cbr.scard, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst,
                          ss; prefix = prefix)
    set_turnover_constraints!(model, cbr.tn; prefix = prefix)
    set_tracking_error_constraints!(model, cbr.pr, tr, owner, plr, fees, b1;
                                    prefix = prefix, kwargs...)
    set_weight_norm_2_constraints!(model, cbr.l2c; prefix = prefix)
    set_weight_norm_p_constraints!(model, cbr.lpc; prefix = prefix)
    set_weight_norm_inf_constraints!(model, cbr.linfc; prefix = prefix)
    set_l1_regularisation!(model, cbr.l1; prefix = prefix)
    set_l2_regularisation!(model, cbr.l2; prefix = prefix)
    set_lp_regularisation!(model, cbr.lp; prefix = prefix)
    set_linf_regularisation!(model, cbr.linf; prefix = prefix)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add phase 2 of a [`ResolvedConstraintBlock`](@ref) to a model: the rows that come after the risk step and the marks of the owner.

The owner calls it after [`mark_risk_minimised!`](@ref), because the semidefinite phylogeny reads the marks `risk_minimised` and `variance_flag` of the weights. [`set_block_constraints!`](@ref) states both phases, their order and their arguments.

# Algorithm

 1. Add the integer phylogeny rows with [`set_iplg_constraints!`](@ref), then the semidefinite phylogeny rows with [`set_sdp_phylogeny_constraints!`](@ref), under `prefix`.
 2. Add the custom constraints `ccnt` with [`add_custom_constraint!`](@ref), with `owner` and `attrs`.

# Arguments

  - $(arg_dict[:model])
  - `cbr`: The Result of the block.
  - `owner`: The owner of the block.
  - `attrs`: The bundle that the custom constraints read, or `nothing`.
  - $(arg_dict[:prefix_kw])

# Returns

  - `nothing`.

# Related

  - [`set_block_constraints!`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function set_block_post_risk_constraints!(model::JuMP.Model, cbr::ResolvedConstraintBlock,
                                          owner, attrs;
                                          prefix::Symbol = Symbol(""))::Nothing
    set_iplg_constraints!(model, cbr.plr; prefix = prefix)
    set_sdp_phylogeny_constraints!(model, cbr.plr; prefix = prefix)
    add_custom_constraint!(model, cbr.ccnt, owner, attrs)
    return nothing
end

export ConstraintBlock, ResolvedConstraintBlock
