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

export ConstraintBlock
