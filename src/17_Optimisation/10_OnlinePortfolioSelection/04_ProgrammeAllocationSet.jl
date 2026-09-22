"""
$(DocStringExtensions.TYPEDEF)

The Allocation Set of the full constraint vocabulary: every constraint kind a [`JuMPOptimiser`](@ref) takes, under the optimiser's field names and type bounds, plus the optimiser's direct objective penalties, with a solver required by its field bound, because a projection onto it is a programme.

The projection is a bare JuMP model in the Weight Finaliser's idiom — `w`, `k = 1`, `Σw = 1`, the constraint scale and the objective scale — assembled by the shared builders through [`set_allocation_set_constraints!`](@ref) in the order [`assemble_jump_model!`](@ref) runs them, and given the objective of the rule's Projection Geometry: ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert^2`` for [`EuclideanProjection`](@ref), ``\\sum_i w_i \\log (w_i / q_i)`` for [`EntropicProjection`](@ref), an exponential-cone programme, and ``(\\boldsymbol{w} - \\boldsymbol{q})^\\intercal A (\\boldsymbol{w} - \\boldsymbol{q})`` for [`GramProjection`](@ref), plus the set's penalties through the Objective Penalty. The budget is one and there is no budget field: cash is an asset with price relative one, which the rule allocates like any other.

**The kinds, by the data they read.**

  - The caller's object, or a name resolved over `sets` once per fold: `wb`, `sbgt`, `gbgt`, `xbgt`, `lt`, `st`, `lcse`, `gcarde`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn`, `card`, `scard`, `ss`, `l2c`, `lpc`, `linfc`, `ccnt`, `l1`, `l2`, `lp`, `linf` and `cobj`, each through the builder [`JuMPOptimiser`](@ref) hands it to. The turnover ceiling's reference is replaced by [`factory`](@ref) at every step with the Price-Adjusted Allocation ``\\hat{\\boldsymbol{w}}_t = \\boldsymbol{w}_t \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` of the step — the book the fund trades from, never the distance between two targets — unless the object is `fixed`, in which case the caller's book stays the reference.
  - The head's rows, resolved at every step on the prior result of `pe` fitted on them: `r`, the risk ceilings, each measure's own JuMP builder with the set as the [`RiskConstraintOwner`](@ref) and `settings.ub` as the ceiling, the measure materialised against that prior through [`factory`](@ref) first, so a moment it carries itself is the one it is built on; `tr`, the tracking errors, a `WeightsTracking` that is not `fixed` given the Price-Adjusted Allocation as the turnover is; `cte`, the centrality rows; `ple`, the integer and semidefinite phylogeny kinds; an exposure row in `lcse`, re-based through the loadings its `FactorSpace` pins, because the rows carry no factor returns to refit them from; and `ret`, a return floor on the prior's expected returns. A [`Variance`](@ref) or [`StandardDeviation`](@ref) that holds its matrix reads no rows and no prior: it is the set's own cone ([`set_allocation_risk_ceiling!`](@ref)).

A set that reads the rows — [`rows_needed`](@ref) answers `nothing` — costs one prior fit over the whole prefix at every step on top of its programme; a caller who wants a window states it on `pe`. A covariance of one observation does not exist, so while the head holds fewer than two rows the fit is not attempted and the step is a Held Step, recorded as such ([`allocation_set_ready`](@ref)); a caller who wants a covariance ceiling from row one gives `r` its matrix. A MIP projection is not unique, so the identity between the Causal Pass and the Recursion Read-out is a claim about the code path — the same solves in the same order on the same rows — and holds exactly with a deterministic solver.

The set refuses, by having no field for them: a budget, fees, the return term as an objective, the scalariser, the optimiser's execution knobs and a `TimeDependent` schedule. A frontier or a per-asset `ub` on a measure is refused at construction because a ceiling is one number ([`assert_risk_ceiling`](@ref)). A measure's `rke` and `scale` are not read: the ceiling is a constraint, and the objective is the geometry's divergence plus the penalties. A negative lower bound is admitted under the Euclidean and Gram geometries and refused under the entropic one, at the head's construction when the bound is a value and at the projection when it is resolved from an estimator. The wealth factor of a leveraged allocation can reach zero on an extreme day, where the log wealth and the next gradient are undefined; that is documented here, not guarded.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProgrammeAllocationSet(;
        slv::Slv_VecSlv,
        pe::AbstractPriorEstimator = EmpiricalPrior(),
        r::Option{<:RM_VecRM} = nothing,
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sbgt::Option{<:Num_BgtRg} = nothing,
        gbgt::Option{<:Num_BgtRg} = nothing,
        xbgt::Bool = false,
        lt::Option{<:BtE_Bt} = nothing,
        st::Option{<:BtE_Bt} = nothing,
        lcse::Option{<:EcE_LcE_Lc_VecEcE_LcE_Lc} = nothing,
        cte::Option{<:Lc_CC_VecCC} = nothing,
        gcarde::Option{<:LcE_Lc} = nothing,
        sgcarde::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
        smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
        sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
        slt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sglt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sgst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        tn::Option{<:TnE_Tn_VecTnE_Tn} = nothing,
        sets::Option{<:UniverseSets} = nothing,
        tr::Option{<:Tr_VecTr} = nothing,
        ple::Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
        ret::Option{<:JRE_VecJRE} = nothing,
        ccnt::Option{<:JuMPConstr_VecJuMPConstr} = nothing,
        cobj::Option{<:JuMPObj_VecJuMPObj} = nothing,
        sc::Number = 1,
        so::Number = 1,
        ss::Option{<:Number} = nothing,
        card::Option{<:Integer} = nothing,
        scard::Option{<:Int_VecInt} = nothing,
        l2c::Option{<:Num_NormCeilCal} = nothing,
        lpc::Option{<:LpReg_VecLpReg} = nothing,
        linfc::Option{<:Num_NormCeilCal} = nothing,
        l1::Option{<:Num_AmbRadCal} = nothing,
        l2::Option{<:L2Reg_VecL2Reg} = nothing,
        lp::Option{<:LpReg_VecLpReg} = nothing,
        linf::Option{<:Num_AmbRadCal} = nothing
    ) -> ProgrammeAllocationSet

Keywords correspond to the struct's fields. `slv` has no default: a set that needs a solver cannot be built without one.

## Validation

  - If any slot holds an estimator keyed by name ([`name_keyed`](@ref)): `!isnothing(sets)`. An `IsNothingError` is thrown otherwise.
  - If `slv` is a vector: non-empty. If `sbgt` or `gbgt` is a number: non-negative and finite. `assert_gross_budget_admissible` under the budget of one.
  - If `r` is given: every measure's `settings.ub` is a finite non-negative number, the ceiling ([`assert_risk_ceiling`](@ref)).
  - If `card` is given: `card > 0` and finite. If `cte`, `tn`, `tr`, `l2`, `lp` or `lpc` is a vector: non-empty. If `l2c`, `linfc`, `l1` or `linf` is a number: `> 0` and finite.
  - The sub-group and sub-grouped MIP slots agree in shape ([`assert_subgroup_mip_fields`](@ref), [`assert_subgrouped_mip_fields`](@ref)), and an [`LpRegularisation`](@ref) is a coefficient in `lp` and a ceiling in `lpc`.

## View parameters

When [`port_opt_view`](@ref) is called on this type, every slot with a per-asset axis is viewed recursively as [`JuMPOptimiser`](@ref)'s is, and the rest is carried unchanged. A precomputed [`LinearConstraint`](@ref) is the identity under a view, as it is everywhere.

# Examples

```jldoctest
julia> ProgrammeAllocationSet(; slv = Solver(; solver = nothing),
                              tn = Turnover(; w = fill(0.25, 4), val = 0.1))
ProgrammeAllocationSet
       pe ┼ EmpiricalPrior
          │           ce ┼ PortfolioOptimisersCovariance
          │              │   ce ┼ Covariance
          │              │      │    me ┼ SimpleExpectedReturns
          │              │      │       │   w ┴ nothing
          │              │      │    ce ┼ GeneralCovariance
          │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
          │              │      │       │    w ┴ nothing
          │              │      │   alg ┼ FullMoment()
          │              │      │     w ┴ nothing
          │              │   mp ┼ MatrixProcessing
          │              │      │     pdm ┼ Posdef
          │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
          │              │      │      dn ┼ nothing
          │              │      │      dt ┼ nothing
          │              │      │     alg ┼ nothing
          │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
          │           me ┼ SimpleExpectedReturns
          │              │   w ┴ nothing
          │      horizon ┼ nothing
          │   fill_limit ┴ nothing
      slv ┼ Solver
          │          name ┼ String: \"\"
          │        solver ┼ nothing
          │      settings ┼ nothing
          │     check_sol ┼ @NamedTuple{}: NamedTuple()
          │   add_bridges ┴ Bool: true
        r ┼ nothing
       wb ┼ WeightBounds
          │   lb ┼ Float64: 0.0
          │   ub ┴ Float64: 1.0
     sbgt ┼ nothing
     gbgt ┼ nothing
     xbgt ┼ Bool: false
       lt ┼ nothing
       st ┼ nothing
     lcse ┼ nothing
      cte ┼ nothing
   gcarde ┼ nothing
  sgcarde ┼ nothing
     smtx ┼ nothing
    sgmtx ┼ nothing
      slt ┼ nothing
      sst ┼ nothing
     sglt ┼ nothing
     sgst ┼ nothing
       tn ┼ Turnover
          │       w ┼ Vector{Float64}: [0.25, 0.25, 0.25, 0.25]
          │     val ┼ Float64: 0.1
          │   fixed ┴ Bool: false
     sets ┼ nothing
       tr ┼ nothing
      ple ┼ nothing
      ret ┼ nothing
     ccnt ┼ nothing
     cobj ┼ nothing
       sc ┼ Int64: 1
       so ┼ Int64: 1
       ss ┼ nothing
     card ┼ nothing
    scard ┼ nothing
      l2c ┼ nothing
      lpc ┼ nothing
    linfc ┼ nothing
       l1 ┼ nothing
       l2 ┼ nothing
       lp ┼ nothing
     linf ┴ nothing
```

# Related

  - [`AbstractAllocationSet`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`project`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`HeldStep`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
@concrete struct ProgrammeAllocationSet <: AbstractProgrammeAllocationSet
    """
    The prior estimator the ceilings, the tracking errors, the centrality and phylogeny rows, an exposure row and the return floor are built on, fitted on the head's rows at every step; unread when no slot reads the rows.
    """
    pe
    """
    $(field_dict[:slv])
    """
    slv
    """
    The risk ceilings, a [`RiskMeasure`](@ref) or a vector of them, each with `settings.ub` the ceiling, or `nothing`.
    """
    r
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
    $(field_dict[:xbgt])
    """
    xbgt
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
    The centrality rows: a [`CentralityConstraint`](@ref), a vector of them, or an already-generated [`LinearConstraint`](@ref), resolved on the head's rows at every step.
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
    The turnover ceilings, a [`Turnover`](@ref), a [`TurnoverEstimator`](@ref) resolved over `sets`, or a vector of them, the reference of each replaced at every step by the Price-Adjusted Allocation unless the object is `fixed`.
    """
    tn
    """
    $(field_dict[:sets])
    """
    sets
    """
    The tracking errors, any [`AbstractTracking`](@ref) or a vector of them, over the head's rows: a [`WeightsTracking`](@ref) that is not `fixed` tracks the Price-Adjusted Allocation, and a [`ReturnsTracking`](@ref) series stated over the fold is cut to the prefix the head has folded so far.
    """
    tr
    """
    $(field_dict[:ple_jmp])
    """
    ple
    """
    The return floor: a return term, or a vector of them, whose `settings.lb` bounds the prior's expected return of the projected allocation from below; the term's objective role is not read.
    """
    ret
    """
    $(field_dict[:ccnt])
    """
    ccnt
    """
    $(field_dict[:cobj])
    """
    cobj
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
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
    function ProgrammeAllocationSet(pe::AbstractPriorEstimator, slv::Slv_VecSlv,
                                    r::Option{<:RM_VecRM}, wb::Option{<:WbE_Wb},
                                    sbgt::Option{<:Num_BgtRg}, gbgt::Option{<:Num_BgtRg},
                                    xbgt::Bool, lt::Option{<:BtE_Bt}, st::Option{<:BtE_Bt},
                                    lcse::Option{<:EcE_LcE_Lc_VecEcE_LcE_Lc},
                                    cte::Option{<:Lc_CC_VecCC}, gcarde::Option{<:LcE_Lc},
                                    sgcarde::Option{<:LcE_Lc_VecLcE_Lc},
                                    smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                                    sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                                    slt::Option{<:BtE_Bt_VecOptBtE_Bt},
                                    sst::Option{<:BtE_Bt_VecOptBtE_Bt},
                                    sglt::Option{<:BtE_Bt_VecOptBtE_Bt},
                                    sgst::Option{<:BtE_Bt_VecOptBtE_Bt},
                                    tn::Option{<:TnE_Tn_VecTnE_Tn},
                                    sets::Option{<:UniverseSets}, tr::Option{<:Tr_VecTr},
                                    ple::Option{<:PlCE_PlC_VecPlCE_PlC},
                                    ret::Option{<:JRE_VecJRE},
                                    ccnt::Option{<:JuMPConstr_VecJuMPConstr},
                                    cobj::Option{<:JuMPObj_VecJuMPObj}, sc::Number,
                                    so::Number, ss::Option{<:Number},
                                    card::Option{<:Integer}, scard::Option{<:Int_VecInt},
                                    l2c::Option{<:Num_NormCeilCal},
                                    lpc::Option{<:LpReg_VecLpReg},
                                    linfc::Option{<:Num_NormCeilCal},
                                    l1::Option{<:Num_AmbRadCal},
                                    l2::Option{<:L2Reg_VecL2Reg},
                                    lp::Option{<:LpReg_VecLpReg},
                                    linf::Option{<:Num_AmbRadCal})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        if isa(sbgt, Number)
            assert_nonempty_nonneg_finite_val(sbgt, :sbgt)
        end
        if isa(gbgt, Number)
            assert_nonempty_nonneg_finite_val(gbgt, :gbgt)
        end
        assert_gross_budget_admissible(1, sbgt, gbgt, wb)
        assert_risk_ceiling(r)
        for (x, name) in
            ((cte, :cte), (tn, :tn), (tr, :tr), (l2, :l2), (lp, :lp), (lpc, :lpc),
             (ret, :ret))
            if isa(x, AbstractVector)
                @argcheck(!isempty(x), IsEmptyError("$name cannot be empty"))
            end
        end
        if !isnothing(card)
            assert_nonempty_gt0_finite_val(card, :card)
        end
        for (x, name) in ((l2c, :l2c), (linfc, :linfc), (l1, :l1), (linf, :linf))
            if !isnothing(x)
                assert_nonempty_gt0_finite_val(x, name)
            end
        end
        assert_penalty_coefficient_role(lp)
        assert_norm_ceiling_role(lpc)
        assert_subgroup_mip_fields(scard, smtx, slt, sst)
        assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)
        if any(name_keyed,
               (wb, lt, st, lcse, cte, gcarde, sgcarde, smtx, sgmtx, slt, sst, sglt, sgst,
                tn))
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when a slot holds an estimator keyed by name"))
        end
        return new{typeof(pe), typeof(slv), typeof(r), typeof(wb), typeof(sbgt),
                   typeof(gbgt), typeof(xbgt), typeof(lt), typeof(st), typeof(lcse),
                   typeof(cte), typeof(gcarde), typeof(sgcarde), typeof(smtx),
                   typeof(sgmtx), typeof(slt), typeof(sst), typeof(sglt), typeof(sgst),
                   typeof(tn), typeof(sets), typeof(tr), typeof(ple), typeof(ret),
                   typeof(ccnt), typeof(cobj), typeof(sc), typeof(so), typeof(ss),
                   typeof(card), typeof(scard), typeof(l2c), typeof(lpc), typeof(linfc),
                   typeof(l1), typeof(l2), typeof(lp), typeof(linf)}(pe, slv, r, wb, sbgt,
                                                                     gbgt, xbgt, lt, st,
                                                                     lcse, cte, gcarde,
                                                                     sgcarde, smtx, sgmtx,
                                                                     slt, sst, sglt, sgst,
                                                                     tn, sets, tr, ple, ret,
                                                                     ccnt, cobj, sc, so, ss,
                                                                     card, scard, l2c, lpc,
                                                                     linfc, l1, l2, lp,
                                                                     linf)
    end
end
function ProgrammeAllocationSet(; slv::Slv_VecSlv,
                                pe::AbstractPriorEstimator = EmpiricalPrior(),
                                r::Option{<:RM_VecRM} = nothing,
                                wb::Option{<:WbE_Wb} = WeightBounds(),
                                sbgt::Option{<:Num_BgtRg} = nothing,
                                gbgt::Option{<:Num_BgtRg} = nothing, xbgt::Bool = false,
                                lt::Option{<:BtE_Bt} = nothing,
                                st::Option{<:BtE_Bt} = nothing,
                                lcse::Option{<:EcE_LcE_Lc_VecEcE_LcE_Lc} = nothing,
                                cte::Option{<:Lc_CC_VecCC} = nothing,
                                gcarde::Option{<:LcE_Lc} = nothing,
                                sgcarde::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
                                smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                                sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                                slt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                                sst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                                sglt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                                sgst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                                tn::Option{<:TnE_Tn_VecTnE_Tn} = nothing,
                                sets::Option{<:UniverseSets} = nothing,
                                tr::Option{<:Tr_VecTr} = nothing,
                                ple::Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
                                ret::Option{<:JRE_VecJRE} = nothing,
                                ccnt::Option{<:JuMPConstr_VecJuMPConstr} = nothing,
                                cobj::Option{<:JuMPObj_VecJuMPObj} = nothing,
                                sc::Number = 1, so::Number = 1,
                                ss::Option{<:Number} = nothing,
                                card::Option{<:Integer} = nothing,
                                scard::Option{<:Int_VecInt} = nothing,
                                l2c::Option{<:Num_NormCeilCal} = nothing,
                                lpc::Option{<:LpReg_VecLpReg} = nothing,
                                linfc::Option{<:Num_NormCeilCal} = nothing,
                                l1::Option{<:Num_AmbRadCal} = nothing,
                                l2::Option{<:L2Reg_VecL2Reg} = nothing,
                                lp::Option{<:LpReg_VecLpReg} = nothing,
                                linf::Option{<:Num_AmbRadCal} = nothing)::ProgrammeAllocationSet
    return ProgrammeAllocationSet(pe, slv, r, wb, sbgt, gbgt, xbgt, lt, st, lcse, cte,
                                  gcarde, sgcarde, smtx, sgmtx, slt, sst, sglt, sgst, tn,
                                  sets, tr, ple, ret, ccnt, cobj, sc, so, ss, card, scard,
                                  l2c, lpc, linfc, l1, l2, lp, linf)
end
function port_opt_view(set::ProgrammeAllocationSet, i, args...)
    if set.smtx === set.sgmtx
        smtx = sgmtx = port_opt_view(set.smtx, i)
    else
        smtx = port_opt_view(set.smtx, i)
        sgmtx = port_opt_view(set.sgmtx, i)
    end
    if set.slt === set.sglt
        slt = sglt = port_opt_view(set.slt, i)
    else
        slt = port_opt_view(set.slt, i)
        sglt = port_opt_view(set.sglt, i)
    end
    if set.sst === set.sgst
        sst = sgst = port_opt_view(set.sst, i)
    else
        sst = port_opt_view(set.sst, i)
        sgst = port_opt_view(set.sgst, i)
    end
    return ProgrammeAllocationSet(; pe = set.pe, slv = set.slv,
                                  r = port_opt_view(set.r, i, nothing),
                                  wb = port_opt_view(set.wb, i), sbgt = set.sbgt,
                                  gbgt = set.gbgt, xbgt = set.xbgt,
                                  lt = port_opt_view(set.lt, i),
                                  st = port_opt_view(set.st, i),
                                  lcse = port_opt_view(set.lcse, i), cte = set.cte,
                                  gcarde = set.gcarde, sgcarde = set.sgcarde, smtx = smtx,
                                  sgmtx = sgmtx, slt = slt, sst = sst, sglt = sglt,
                                  sgst = sgst, tn = port_opt_view(set.tn, i),
                                  sets = port_opt_view(set.sets, i),
                                  tr = port_opt_view(set.tr, i, args...), ple = set.ple,
                                  ret = port_opt_view(set.ret, i),
                                  ccnt = port_opt_view(set.ccnt, i),
                                  cobj = port_opt_view(set.cobj, i), sc = set.sc,
                                  so = set.so, ss = set.ss, card = set.card,
                                  scard = set.scard, l2c = set.l2c, lpc = set.lpc,
                                  linfc = set.linfc, l1 = set.l1, l2 = set.l2, lp = set.lp,
                                  linf = set.linf)
end
"""
    name_keyed(x)
    name_keyed(x::AbstractVector)

Whether a constraint slot holds an estimator that resolves its names over `sets`, so its owner must carry them: a [`WeightBoundsEstimator`](@ref), a [`LinearConstraintEstimator`](@ref), an [`ExposureConstraintEstimator`](@ref), a [`ThresholdEstimator`](@ref), an [`AssetSetsMatrixEstimator`](@ref), a [`TurnoverEstimator`](@ref), or a vector holding one.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function name_keyed(::Any)
    return false
end
function name_keyed(::Union{<:WeightBoundsEstimator, <:LinearConstraintEstimator,
                            <:ExposureConstraintEstimator, <:ThresholdEstimator,
                            <:AssetSetsMatrixEstimator, <:TurnoverEstimator})
    return true
end
function name_keyed(x::AbstractVector)
    return any(name_keyed, x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The first stage of a [`ProgrammeAllocationSet`](@ref)'s resolution, once per fold: every constraint keyed by name is resolved to a value over `N` assets through `sets` — the weight bounds, the thresholds, the linear and grouped cardinality constraints, the sub-group selection matrices and thresholds, and the turnover — as [`processed_jump_optimiser_attributes`](@ref) resolves them. A slot that reads the head's rows is carried unchanged to the second stage, [`resolve_allocation_set_rows`](@ref), which runs at every step.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
  - [`weight_bounds_constraints`](@ref)
  - [`linear_constraints`](@ref)
  - [`threshold_constraints`](@ref)
  - [`asset_sets_matrix`](@ref)
  - [`turnover_constraints`](@ref)
"""
function resolve_allocation_set(set::ProgrammeAllocationSet, N::Integer, strict::Bool,
                                datatype::DataType)
    sets = set.sets
    wb = weight_bounds_constraints(set.wb, sets; N = N, strict = strict,
                                   datatype = datatype)
    lt = threshold_constraints(set.lt, sets; datatype = datatype, strict = strict)
    st = threshold_constraints(set.st, sets; datatype = datatype, strict = strict)
    gcarde = linear_constraints(set.gcarde, sets; datatype = Int, strict = strict)
    sgcarde = linear_constraints(set.sgcarde, sets; datatype = Int, strict = strict)
    if set.smtx === set.sgmtx
        smtx = sgmtx = asset_sets_matrix(set.smtx, sets)
    else
        smtx = asset_sets_matrix(set.smtx, sets)
        sgmtx = asset_sets_matrix(set.sgmtx, sets)
    end
    if set.slt === set.sglt
        slt = sglt = threshold_constraints(set.slt, sets; datatype = datatype,
                                           strict = strict)
    else
        slt = threshold_constraints(set.slt, sets; datatype = datatype, strict = strict)
        sglt = threshold_constraints(set.sglt, sets; datatype = datatype, strict = strict)
    end
    if set.sst === set.sgst
        sst = sgst = threshold_constraints(set.sst, sets; datatype = datatype,
                                           strict = strict)
    else
        sst = threshold_constraints(set.sst, sets; datatype = datatype, strict = strict)
        sgst = threshold_constraints(set.sgst, sets; datatype = datatype, strict = strict)
    end
    tn = turnover_constraints(set.tn, sets; datatype = datatype, strict = strict)
    # An exposure row needs the prior's loadings, so `lcse` is resolved on the rows unless
    # it holds no exposure estimator, in which case it resolves here as the optimiser's does.
    lcse = if exposure_keyed(set.lcse)
        set.lcse
    else
        linear_constraints(set.lcse, sets; datatype = datatype, strict = strict)
    end
    return ProgrammeAllocationSet(; pe = set.pe, slv = set.slv, r = set.r, wb = wb,
                                  sbgt = set.sbgt, gbgt = set.gbgt, xbgt = set.xbgt,
                                  lt = lt, st = st, lcse = lcse, cte = set.cte,
                                  gcarde = gcarde, sgcarde = sgcarde, smtx = smtx,
                                  sgmtx = sgmtx, slt = slt, sst = sst, sglt = sglt,
                                  sgst = sgst, tn = tn, sets = sets, tr = set.tr,
                                  ple = set.ple, ret = set.ret, ccnt = set.ccnt,
                                  cobj = set.cobj, sc = set.sc, so = set.so, ss = set.ss,
                                  card = set.card, scard = set.scard, l2c = set.l2c,
                                  lpc = set.lpc, linfc = set.linfc, l1 = set.l1,
                                  l2 = set.l2, lp = set.lp, linf = set.linf)
end
"""
    exposure_keyed(x)
    exposure_keyed(x::AbstractVector)

Whether a linear-constraint slot holds an [`ExposureConstraintEstimator`](@ref), whose rows are written in another basis and re-based through a factor prior's loadings, so it resolves on the head's rows and not once per fold.

# Related

  - [`resolve_allocation_set`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
"""
function exposure_keyed(::Any)
    return false
end
function exposure_keyed(::ExposureConstraintEstimator)
    return true
end
function exposure_keyed(x::AbstractVector)
    return any(exposure_keyed, x)
end
"""
    fitted_on_rows(x)
    fitted_on_rows(x::AbstractVector)

Whether a centrality or phylogeny slot holds an estimator, which is fitted on the head's rows at every step; a precomputed [`LinearConstraint`](@ref), [`IntegerPhylogeny`](@ref) or [`SemiDefinitePhylogeny`](@ref) reads none.

# Related

  - [`rows_needed`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
"""
function fitted_on_rows(::Any)
    return false
end
function fitted_on_rows(::Union{<:AbstractCentralityConstraint,
                                <:AbstractPhylogenyConstraintEstimator})
    return true
end
function fitted_on_rows(x::AbstractVector)
    return any(fitted_on_rows, x)
end
"""
    rows_needed(set::ProgrammeAllocationSet)

The rows a programme set reads at a step: `nothing`, every row folded, when any slot reads the head's rows — a ceiling that reads them ([`risk_reads_rows`](@ref)), a tracking error, a centrality or phylogeny estimator ([`fitted_on_rows`](@ref)), an exposure row ([`exposure_keyed`](@ref)), a return floor, or a Calibration Rule in a norm ceiling or a penalty ([`calibrated`](@ref)); `0` otherwise.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`rows_needed`](@ref)
  - [`risk_reads_rows`](@ref)
"""
function rows_needed(set::ProgrammeAllocationSet)
    reads = risk_reads_rows(set.r) ||
            !isnothing(set.tr) ||
            fitted_on_rows(set.cte) ||
            fitted_on_rows(set.ple) ||
            exposure_keyed(set.lcse) ||
            !isnothing(set.ret) ||
            any(calibrated, (set.l2c, set.lpc, set.linfc, set.l1, set.l2, set.lp, set.linf))
    return reads ? nothing : 0
end
"""
$(DocStringExtensions.TYPEDEF)

The Tsallis Projection Geometry: the raw step is projected onto the Allocation Set in the Bregman divergence of the power potential ``\\Psi_\\alpha(\\boldsymbol{w}) = \\sum_i w_i^\\alpha / (\\alpha (\\alpha - 1))``, ``\\alpha \\in (0, 1)``.

# Mathematical definition

The mirror image of an allocation is ``\\nabla \\Psi_\\alpha(\\boldsymbol{w})_i = w_i^{\\alpha - 1} / (\\alpha - 1)``, so a first-order step of length ``\\eta`` on a gradient ``\\boldsymbol{g}`` and its projection onto the default set are one scalar root,

```math
\\begin{align}
w_{t+1, i} &= \\left( w_{t, i}^{\\alpha - 1} + (1 - \\alpha) (\\eta g_i + \\lambda) \\right)^{1 / (\\alpha - 1)}\\,,
\\end{align}
```

with ``\\lambda`` the budget multiplier, in which the budget is monotone. On a [`BoundedAllocationSet`](@ref) the bounds are clips of the same root, as the entropic arm's are; on a [`ProgrammeAllocationSet`](@ref) the projection is the programme ``\\min_{\\boldsymbol{w}} \\Psi_\\alpha(\\boldsymbol{w}) - \\langle \\nabla \\Psi_\\alpha(\\boldsymbol{q}), \\boldsymbol{w} \\rangle`` through a power cone on the set's solver. The limit ``\\alpha \\to 1`` is the relative entropy of [`EntropicProjection`](@ref) and ``\\alpha \\to 0`` the log barrier of [`LogBarrierProjection`](@ref); the shipped range is the open interval between them. It is the geometry of the Tsallis-entropy mirror-descent and follow-the-regularised-leader steps of Abernethy, Lee and Tewari (2015) and Zimmert and Seldin (2021), whose regret on the simplex is ``O(\\sqrt{T N / (\\alpha (1 - \\alpha))})`` at the tuned rate. Like the entropic map, it cannot zero a positive entry, a zero entry stays zero, and a negative lower bound is refused, because the potential is undefined below zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TsallisProjection(; alpha::Real = 0.5) -> TsallisProjection

Keywords correspond to the struct's fields. The default is the ``\\alpha = 1/2`` of Zimmert and Seldin (2021).

## Validation

  - `0 < alpha < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> TsallisProjection()
TsallisProjection
  alpha ┴ Float64: 0.5
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`LogBarrierProjection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:abernethy2015])
  - $(ref_dict[:zimmertseldin2021])
"""
struct TsallisProjection{T1 <: Real} <: AbstractProjectionGeometry
    """
    The power of the potential, in `(0, 1)`.
    """
    alpha::T1
    function TsallisProjection(alpha::Real)
        @argcheck(zero(alpha) < alpha < one(alpha),
                  DomainError(alpha, "alpha must be in (0, 1)"))
        return new{typeof(alpha)}(alpha)
    end
end
function TsallisProjection(; alpha::Real = 0.5)::TsallisProjection
    return TsallisProjection(alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

The log-barrier Projection Geometry: the raw step is projected onto the Allocation Set in the Bregman divergence of the Burg entropy ``\\Psi(\\boldsymbol{w}) = -\\sum_i \\log w_i``, which is the Itakura–Saito divergence ``\\sum_i \\left( w_i / q_i - 1 - \\log (w_i / q_i) \\right)``.

# Mathematical definition

The mirror image is ``\\nabla \\Psi(\\boldsymbol{w})_i = -1 / w_i``, so a first-order step of length ``\\eta`` on a gradient ``\\boldsymbol{g}`` and its projection onto the default set are one scalar root,

```math
\\begin{align}
w_{t+1, i} &= \\frac{1}{1 / w_{t, i} + \\eta g_i + \\lambda}\\,,
\\end{align}
```

with ``\\lambda`` the budget multiplier, in which the budget is monotone. On a [`BoundedAllocationSet`](@ref) the bounds are clips of the same root; on a [`ProgrammeAllocationSet`](@ref) the projection is the programme ``\\min_{\\boldsymbol{w}} -\\sum_i \\log w_i + \\sum_i w_i / q_i`` through an exponential cone on the set's solver. It is the geometry Orseau, Lattimore and Legg (2017, §7) name as the mirror-descent alternative to their Soft-Bayes step: the one first-order geometry whose portfolio regret, ``O(\\sqrt{N T \\log (T / N)})``, needs no lower bound on the price relatives, because a weight that has shrunk towards zero moves by its own scale. It cannot zero a positive entry, a zero entry stays zero, and a negative lower bound is refused, because the logarithm is undefined below zero.

# Examples

```jldoctest
julia> LogBarrierProjection()
LogBarrierProjection()
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`TsallisProjection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:orseau2017])
"""
struct LogBarrierProjection <: AbstractProjectionGeometry end
"""
    mirror_step(proj::EuclideanProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::EntropicProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::TsallisProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::LogBarrierProjection, u::AbstractVector, s::AbstractVector)

The unconstrained mirror step from the iterate `u` along the scaled gradient `s = η g`, in the geometry's potential: the raw step ``\\nabla \\Psi^*(\\nabla \\Psi(\\boldsymbol{u}) - \\boldsymbol{s})`` that [`project`](@ref) then puts onto the Allocation Set.

The Euclidean arm is `u - s`, the entropic `u ⊙ exp(-s)`, the Tsallis ``(u_i^{\\alpha - 1} + (1 - \\alpha) s_i)^{1 / (\\alpha - 1)}`` and the log-barrier ``1 / (1 / u_i + s_i)``. The last two are defined while every base is positive: under the log barrier that is ``\\eta \\hat{w}_{t, i} < 1`` for every asset, with ``\\hat{\\boldsymbol{w}}_t`` the Price-Adjusted Allocation, so it always holds at a rate below one and fails only where one asset carries more than ``1 / \\eta`` of the period's wealth; the Tsallis condition is ``(1 - \\alpha) \\eta \\hat{w}_{t, i} w_{t, i}^{-\\alpha} < 1``. A base at or below zero is a step to an unbounded allocation, and it is refused, not clipped.

# Validation

  - Under [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): every base is positive. A `DomainError` naming the rate is thrown otherwise.

# Related

  - [`MirrorDescent`](@ref)
  - [`project`](@ref)
"""
function mirror_step(::EuclideanProjection, u::AbstractVector, s::AbstractVector)
    return u .- s
end
function mirror_step(::EntropicProjection, u::AbstractVector, s::AbstractVector)
    return u .* exp.(-s)
end
function mirror_step(proj::TsallisProjection, u::AbstractVector, s::AbstractVector)
    a = proj.alpha
    base = u .^ (a - 1) .+ (1 - a) .* s
    assert_mirror_base(base, s)
    return base .^ inv(a - 1)
end
function mirror_step(::LogBarrierProjection, u::AbstractVector, s::AbstractVector)
    base = inv.(u) .+ s
    assert_mirror_base(base, s)
    return inv.(base)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a barrier mirror step one of whose bases is not positive, naming the scaled gradient that produced it.

# Related

  - [`mirror_step`](@ref)
"""
function assert_mirror_base(base::AbstractVector, s::AbstractVector)::Nothing
    @argcheck(all(x -> x > zero(x), base),
              DomainError(s,
                          "the mirror step leaves the geometry's domain: a base of the barrier potential is not positive, so the unconstrained step is unbounded in some asset. Lower the learning rate; under the log barrier the step exists while `eta * w_i * x_i / <w, x> < 1` for every asset."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The scalar root of a barrier geometry's projection onto a bounded set: the budget multiplier `μ` at which `Σ_i clip(φ(b_i + μ), lb_i, ub_i)` is one, with `b` the mirror image of the raw step and `φ` the inverse mirror map, read as the upper bound where the base is not positive.

`φ` is decreasing on positive bases and unbounded as the base falls to zero, so the clipped budget is non-increasing and continuous in `μ`: every base at or below zero puts its asset at its cap, every base at infinity — a zero raw entry — at its floor. The bracket is `[-max_i b_i, hi]`, where the lower end caps every finite entry and `hi` is widened by doubling until the budget is at most one, and [`bounded_root`](@ref) bisects it. A set whose floors already sum to one is the point `lb`.

# Arguments

  - `b`: The mirror image of the raw step, `Inf` at a zero entry.
  - `phi`: The inverse mirror map of a positive base.
  - `wb`: The resolved bounds.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub`. An `ArgumentError` is thrown otherwise.
  - At least one raw entry is positive, and the caps of the positive entries together with the floors of the zero ones reach the budget. A `DomainError` is thrown otherwise: the zeros stay zero under a barrier, and the rest cannot fill the budget.

# Returns

  - `w'::Vector`: The projected allocation.

# Related

  - [`project`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
  - [`bounded_root`](@ref)
"""
function barrier_projection(b::AbstractVector, phi, wb::WeightBounds)
    assert_feasible_bounds(wb)
    lb, ub = wb.lb, wb.ub
    if sum(lb) >= one(eltype(lb))
        return collect(lb)
    end
    finite = filter(isfinite, b)
    @argcheck(!isempty(finite),
              DomainError(b,
                          "a barrier projection needs a positive raw entry: a raw step of zeros has no projection, because a zero stays zero under the potential"))
    clipped = (m, l, u) -> m > zero(m) ? clamp(phi(m), l, u) : u
    f = mu -> sum(clipped.(b .+ mu, lb, ub))
    lo = -maximum(finite)
    @argcheck(f(lo) >= one(lo),
              DomainError(b,
                          "the zeros of the raw step stay zero under a barrier potential, and the caps of the remaining assets do not reach the budget"))
    mu = bounded_root(f, lo, barrier_upper_bracket(f, lo))
    return clipped.(b .+ mu, lb, ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The upper end of a barrier root's bracket: a multiplier above `lo` and above zero at which the clipped budget is at most one, found by doubling.

# Related

  - [`barrier_projection`](@ref)
"""
function barrier_upper_bracket(f, lo)
    hi = max(lo, zero(lo)) + one(lo)
    while f(hi) > one(hi)
        hi *= 2
    end
    return hi
end
"""
    project(proj::TsallisProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::LogBarrierProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)

The barrier arms of the Constrained Update on the bounded set: the scalar root of [`barrier_projection`](@ref) in the geometry's mirror image, on every bound, the simplex included, because neither potential has a closed form there. A root whose allocation misses the budget in floating point is the Held Step ([`budget_or_held_step`](@ref)): under the log barrier a raw entry of `1e-20` has the base `1e20`, and the multiplier that brings it to a share of the budget cancels against that base.

# Validation

  - `all(>= 0, q)` and every entry finite. A `DomainError` is thrown otherwise.
  - `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise.

# Related

  - [`project`](@ref)
  - [`barrier_projection`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function project(proj::TsallisProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    assert_barrier_raw_step(q, set.wb)
    a = proj.alpha
    return budget_or_held_step(barrier_projection(q .^ (a - 1), m -> m^inv(a - 1), set.wb),
                               w, proj, set)
end
function project(proj::LogBarrierProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    assert_barrier_raw_step(q, set.wb)
    return budget_or_held_step(barrier_projection(inv.(q), inv, set.wb), w, proj, set)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses, for a barrier geometry, a raw step with a negative or non-finite entry and a bound with a negative floor: the potential is undefined below zero.

# Related

  - [`project`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function assert_barrier_raw_step(q::AbstractVector, wb::WeightBounds)::Nothing
    @argcheck(all(x -> isfinite(x) && x >= zero(x), q),
              DomainError(q,
                          "a barrier projection is defined on finite non-negative raw steps alone: the potential is undefined below zero"))
    @argcheck(all(x -> x >= zero(x), wb.lb),
              DomainError(wb.lb,
                          "a barrier projection admits no negative lower bound: the potential is undefined below zero"))
    return nothing
end
"""
    assert_geometry_admits_set(proj::AbstractProjectionGeometry, set::AbstractAllocationSet)

Refuses, where the head's `alg` and `set` meet, a negative lower bound under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref), whose potentials are undefined below zero.

The refusal is at construction when the bound is a value; a bound resolved from an estimator is refused at the projection. Any other geometry admits a negative bound, so a long-short reversion costs nothing.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) or [`LogBarrierProjection`](@ref) with a [`WeightBounds`](@ref) whose `lb` is a number or a vector: `all(>= 0, lb)`. A `DomainError` is thrown otherwise.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`project`](@ref)
"""
function assert_geometry_admits_set(::AbstractProjectionGeometry,
                                    ::AbstractAllocationSet)::Nothing
    return nothing
end
function assert_geometry_admits_set(proj::Union{<:EntropicProjection, <:TsallisProjection,
                                                <:LogBarrierProjection},
                                    set::AbstractAllocationSet)::Nothing
    wb = set.wb
    if isa(wb, WeightBounds) && !isnothing(wb.lb)
        @argcheck(all(x -> x >= zero(x), wb.lb),
                  DomainError(wb.lb,
                              "a `$(nameof(typeof(proj)))` admits no negative lower bound: its potential is undefined below zero. Use a Euclidean rule, or a non-negative bound."))
    end
    return nothing
end
export ProgrammeAllocationSet, TsallisProjection, LogBarrierProjection
public mirror_step
