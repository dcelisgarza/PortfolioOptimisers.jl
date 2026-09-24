"""
$(DocStringExtensions.TYPEDEF)

Constrains an online allocation with every constraint kind of a JuMP optimiser, and projects onto the set with a solver.

The set takes the field names and the type bounds of [`JuMPOptimiser`](@ref), and the direct objective penalties of the optimiser. A projection onto the set is a programme, so `slv` has no default. The budget is one, and the set has no budget field. Cash is an asset whose price relative is one, and the rule allocates it as any other asset.

The projection is a bare JuMP model with the variables and rows of the Weight Finaliser, which are `w`, `k = 1`, `Σw = 1`, the constraint scale and the objective scale. [`set_allocation_set_constraints!`](@ref) adds the constraint kinds through the builders of the optimiser, in the order that [`assemble_jump_model!`](@ref) runs them. [`set_projection_objective!`](@ref) sets the objective, the divergence of the rule's Projection Geometry plus the set's penalties.

The kinds read their data at two times. Once per fold, [`resolve_allocation_set`](@ref) resolves the kinds that hold the caller's object or a name over `sets`. These are `wb`, `sbgt`, `gbgt`, `xbgt`, `lt`, `st`, `lcse`, `gcarde`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn`, `card`, `scard`, `ss`, `l2c`, `lpc`, `linfc`, `ccnt`, `l1`, `l2`, `lp`, `linf` and `cobj`. Each goes to the builder that [`JuMPOptimiser`](@ref) gives it to. At every step, [`resolve_allocation_set_rows`](@ref) resolves the kinds that read the head's rows, on the prior result of `pe` fitted on those rows:

  - `r`, the risk ceilings. Each measure goes through its own JuMP builder, with the set as the [`RiskConstraintOwner`](@ref) and `settings.ub` as the ceiling. [`factory`](@ref) first materialises the measure against the prior, so a moment that the measure carries is the moment its row uses. A [`Variance`](@ref) or [`StandardDeviation`](@ref) that holds its matrix reads no rows and no prior. It goes through the same builder with no prior, so the source of the matrix does not change the formulation ([`set_allocation_risk_ceiling!`](@ref)).
  - `tr`, the tracking errors, and `cte`, the centrality rows.
  - `ple`, the integer and semidefinite phylogeny kinds.
  - An exposure row in `lcse`. The row goes through the loadings that its `FactorSpace` fixes, because the rows carry no factor returns to fit the loadings again.
  - `ret`, a floor on the prior's expected return of the allocation.

At every step [`factory`](@ref) replaces the reference of a turnover ceiling, and of a [`WeightsTracking`](@ref), with the Price-Adjusted Allocation of the step. That is the book the fund trades from. A `fixed` object keeps the caller's reference.

A set that reads the rows, for which [`rows_needed`](@ref) returns `nothing`, fits its prior once over the whole prefix at every step, and then solves its programme. A caller who wants a window states it on `pe`. A covariance of one observation does not exist. So while the head holds fewer than two rows, the set fits no prior, and the step is a Held Step with its reason recorded ([`allocation_set_ready`](@ref)). A caller who wants a covariance ceiling from the first row gives `r` its matrix. A MIP projection is not unique. So the identity between the Causal Pass and the Recursion Read-out is a property of the code path, the same solves in the same order on the same rows, and it holds exactly only with a deterministic solver.

The set has no field for a budget, fees, the return term as an objective, the scalariser, the execution settings of the optimiser or a `TimeDependent` schedule. At construction [`assert_risk_ceiling`](@ref) refuses a frontier or a per-asset `ub` on a measure, because a ceiling is one number. The set does not read the `rke` or the `scale` of a measure. The ceiling is a constraint, and the objective is the divergence plus the penalties. The Euclidean, Gram and diagonal geometries admit a negative lower bound. The entropic, Tsallis and log-barrier geometries refuse it, at the head's construction when the bound is a value ([`assert_geometry_admits_set`](@ref)) and at the projection when an estimator resolves it. The wealth factor of a leveraged allocation can reach zero on an extreme day. At that point the log wealth and the next gradient are undefined, and the set does not guard against it.

# Mathematical definition

```math
\\begin{align}
\\mathcal{W} &= \\left\\{ \\boldsymbol{w} \\in \\mathbb{R}^N : \\boldsymbol{1}^\\intercal \\boldsymbol{w} = 1,\\; \\boldsymbol{w} \\in \\mathcal{C}_k \\text{ for every kind } k \\text{ of the set} \\right\\}\\,, \\\\
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q}) + \\pi(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:W_aset])
  - $(math_dict[:w_port])
  - $(math_dict[:N])
  - ``\\mathcal{C}_k``: Feasible set of the constraint kind ``k``, which the builder of [`JuMPOptimiser`](@ref) for that kind writes. A risk ceiling, a tracking error, a centrality row, a phylogeny kind, an exposure row and a return floor take ``\\mathcal{C}_k`` from the prior of the step, and a turnover ceiling takes it from the Price-Adjusted Allocation.
  - $(math_dict[:w_plus_proj])
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:q_raw])
  - $(math_dict[:pi_obj_pen])

[`set_projection_objective!`](@ref) states the divergence of each geometry.

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

Keywords correspond to the struct's fields. `slv` has no default, because every projection onto the set is a programme.

## Validation

  - If `slv` is a vector: `!isempty(slv)`. An `IsEmptyError` is thrown otherwise.
  - If `sbgt` or `gbgt` is a number: non-negative and finite. A `DomainError` is thrown otherwise.
  - If `gbgt` is given: `sbgt` is not a number, because the budget of one and a number in `sbgt` already fix the gross exposure. If `wb` is a [`WeightBounds`](@ref), one of its bounds admits a short weight. An `ArgumentError` is thrown otherwise.
  - If `r` is given: the `settings.ub` of every measure is a finite non-negative number, the ceiling ([`assert_risk_ceiling`](@ref)). An `ArgumentError` is thrown otherwise, and an `IsEmptyError` for an empty vector.
  - If `cte`, `tn`, `tr`, `l2`, `lp`, `lpc` or `ret` is a vector: `!isempty(x)`. An `IsEmptyError` is thrown otherwise.
  - If `card` is given: `card > 0` and finite. If `l2c`, `linfc`, `l1` or `linf` is a number: `> 0` and finite. A `DomainError` is thrown otherwise.
  - An [`LpRegularisation`](@ref) in `lp` holds no norm ceiling rule, and one in `lpc` holds no ambiguity radius rule. An `ArgumentError` is thrown otherwise.
  - The sub-group and sub-grouped MIP slots agree in shape ([`assert_subgroup_mip_fields`](@ref), [`assert_subgrouped_mip_fields`](@ref)).
  - If a slot holds an estimator keyed by name ([`name_keyed`](@ref)): `!isnothing(sets)`. An `IsNothingError` is thrown otherwise.

## View parameters

`ProgrammeAllocationSet` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method passes the arguments after `i` to the tracking errors `tr` alone. It views the risk ceilings `r` with `nothing` in place of those arguments.
  - The prior estimator `pe`, the per-asset slots `wb`, `lt`, `st`, `lcse`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn`, `sets`, `ret`, `ccnt` and `cobj`, and the ceilings and tracking errors recurse through [`port_opt_view`](@ref), as the fields of [`JuMPOptimiser`](@ref) do.
  - The centrality rows `cte`, the grouped cardinality slots `gcarde` and `sgcarde`, and the phylogeny kinds `ple` pass through unchanged, as they do on the optimiser. A precomputed [`LinearConstraint`](@ref) is the identity under a view.
  - When a sub-group slot and its sub-grouped slot hold one object, such as `smtx` and `sgmtx`, the method views it once, and the two slots stay one object.

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
  - [`BoundedAllocationSet`](@ref): the set of weight bounds alone, with a scalar root for every geometry but the Gram one.
  - [`JuMPOptimiser`](@ref)
  - [`project`](@ref)
  - [`projection_programme`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`set_projection_objective!`](@ref)
  - [`resolve_allocation_set`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
  - [`HeldStep`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct ProgrammeAllocationSet <: AbstractProgrammeAllocationSet
    """
    The prior estimator that the set fits on the head's rows at every step. The risk ceilings, the tracking errors, the centrality and phylogeny rows, an exposure row and the return floor use its result. The set does not read it when no slot reads the rows.
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
    The turnover ceilings: a [`Turnover`](@ref), a [`TurnoverEstimator`](@ref) that resolves over `sets`, or a vector of them. At every step the set replaces the reference of each ceiling with the Price-Adjusted Allocation, unless the ceiling is `fixed`.
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
    return ProgrammeAllocationSet(; pe = port_opt_view(set.pe, i), slv = set.slv,
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

Tells whether a constraint slot holds an estimator that resolves its names over `sets`.

The owner of such a slot must carry `sets`. The estimators are [`WeightBoundsEstimator`](@ref), [`LinearConstraintEstimator`](@ref), [`ExposureConstraintEstimator`](@ref), [`ThresholdEstimator`](@ref), [`AssetSetsMatrixEstimator`](@ref) and [`TurnoverEstimator`](@ref). A vector is keyed by name when one of its entries is.

# Returns

  - `::Bool`: `true` when the slot holds such an estimator.

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

Resolves the slots of a [`ProgrammeAllocationSet`](@ref) that hold a name or an estimator, once per fold.

This is the first stage of the set's resolution. It resolves each slot to a value over the `N` assets through `sets`, as [`processed_jump_optimiser_attributes`](@ref) resolves the same slot of a [`JuMPOptimiser`](@ref). The slots that read the head's rows go unchanged to the second stage, [`resolve_allocation_set_rows`](@ref), which runs at every step.

# Algorithm

 1. Resolve the weight bounds `wb` over the `N` assets with [`weight_bounds_constraints`](@ref).
 2. Resolve the thresholds `lt` and `st` with [`threshold_constraints`](@ref).
 3. Resolve the grouped cardinality constraints `gcarde` and `sgcarde` with [`linear_constraints`](@ref), with integer values.
 4. Resolve the sub-group matrices `smtx` and `sgmtx` with [`asset_sets_matrix`](@ref). When the two slots hold one object, resolve it once and give the result to both slots.
 5. Resolve the sub-group thresholds `slt` and `sglt`, then `sst` and `sgst`, with [`threshold_constraints`](@ref), in the same way.
 6. Resolve the turnover ceilings `tn` with [`turnover_constraints`](@ref).
 7. When `lcse` holds no exposure estimator ([`exposure_keyed`](@ref)), resolve it with [`linear_constraints`](@ref). Otherwise keep it for the second stage, because an exposure row needs the loadings of the prior.
 8. Return a new set that holds the resolved slots and every other slot of `set`.

# Arguments

  - `set`: The programme set.
  - `N`: The number of assets.
  - $(arg_dict[:strict])
  - `datatype`: The element type of the resolved values. The grouped cardinality constraints take `Int`.

# Returns

  - `set'::ProgrammeAllocationSet`: The set with its named slots resolved.

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

Tells whether a linear-constraint slot holds an [`ExposureConstraintEstimator`](@ref).

An exposure estimator writes its rows in the basis of the factors, and the loadings of a factor prior take them to the assets. So such a slot resolves on the head's rows at every step, and not once per fold. A vector is keyed by exposure when one of its entries is.

# Returns

  - `::Bool`: `true` when the slot holds an exposure estimator.

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

Tells whether a centrality or phylogeny slot holds an estimator that the set fits on the head's rows at every step.

A precomputed [`LinearConstraint`](@ref), [`IntegerPhylogeny`](@ref) or [`SemiDefinitePhylogeny`](@ref) reads no rows. A vector reads the rows when one of its entries does.

# Returns

  - `::Bool`: `true` when the slot holds a centrality or phylogeny estimator.

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

Returns the number of rows that a programme set reads at a step.

The set reads every row that the head has folded when one of its slots reads the rows. These slots are a risk ceiling that reads them ([`risk_reads_rows`](@ref)), a tracking error, a centrality or phylogeny estimator ([`fitted_on_rows`](@ref)), an exposure row ([`exposure_keyed`](@ref)), a return floor, and a Calibration Rule in a norm ceiling or in a penalty ([`calibrated`](@ref)).

# Returns

  - `::Union{Nothing, Int}`: `nothing`, every row folded, when a slot reads the rows, and `0` otherwise.

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

Projects the raw step onto the Allocation Set in the Bregman divergence of the Tsallis potential of power `alpha`.

This is the geometry of the Tsallis-entropy steps of Abernethy, Lee and Tewari (2015) and of the Tsallis-INF algorithm of Zimmert and Seldin (2021). The potential of the first paper is ``\\alpha \\Psi_\\alpha`` plus a constant, and the potential of the second is ``\\Psi_\\alpha`` plus an affine term. So each divergence is ``D_{\\Psi_\\alpha}`` times a constant factor, which a learning rate absorbs. Both papers bound the regret of the multi-armed bandit with bounded losses. At the tuned rate the expected regret is at most ``\\sqrt{2 T N / (\\alpha (1 - \\alpha))}`` (Abernethy, Lee and Tewari, 2015, Corollary 3.2). The log-wealth loss of a portfolio has no bounded gradient, so that bound does not apply to it as it stands. The geometry cannot set a positive entry to zero, and a zero entry stays zero. The geometry refuses a negative lower bound, because the potential is undefined below zero.

# Mathematical definition

```math
\\begin{align}
\\Psi_\\alpha(\\boldsymbol{w}) &= \\frac{1}{\\alpha (\\alpha - 1)} \\sum_i w_i^{\\alpha}\\,, \\\\
\\mathrm{Proj}^{\\alpha}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_{\\Psi_\\alpha}(\\boldsymbol{w}, \\boldsymbol{q})\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(\\left(q_i^{\\alpha - 1} + \\theta\\right)^{1 / (\\alpha - 1)},\\, l_i\\right),\\, u_i\\right)\\,, \\\\
w_{t+1, i} &= \\left( w_{t, i}^{\\alpha - 1} + (1 - \\alpha) \\eta_t g_{t, i} + \\theta \\right)^{1 / (\\alpha - 1)}\\,.
\\end{align}
```

Where:

  - ``\\Psi_\\alpha``: Tsallis potential, the potential of this geometry.
  - ``\\alpha``: Power of the potential, in ``(0, 1)``.
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:g_t_loss])

The mirror image is ``\\nabla \\Psi_\\alpha(\\boldsymbol{w})_i = w_i^{\\alpha - 1} / (\\alpha - 1)``, so the projection sets ``\\nabla \\Psi_\\alpha(\\boldsymbol{w}) - \\nabla \\Psi_\\alpha(\\boldsymbol{q})`` to one constant over the free entries. On a [`BoundedAllocationSet`](@ref) the projection is ``\\boldsymbol{w}(\\theta)`` at the root. A base ``q_i^{\\alpha - 1} + \\theta`` at or below zero puts its asset at ``u_i``. The budget ``\\sum_i w_i(\\theta)`` does not increase in ``\\theta``. The last line is a first-order step from ``\\boldsymbol{w}_t`` and its projection onto the simplex together, one scalar root. As ``\\alpha \\to 1``, ``D_{\\Psi_\\alpha}`` tends to the relative entropy of [`EntropicProjection`](@ref). As ``\\alpha \\to 0``, it tends to the Itakura–Saito divergence of [`LogBarrierProjection`](@ref).

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

Projects the raw step onto the Allocation Set in the Itakura–Saito divergence, the Bregman divergence of the Burg entropy.

Orseau, Lattimore and Legg (2017, §7) name the Burg entropy as the natural regulariser of mirror descent on the log loss, the alternative to their Soft-Bayes step. They state, without a proof in the paper, that its regret is ``O(\\sqrt{N T \\log (T / N)})`` and does not depend on the largest gradient. For a portfolio, this means that the bound needs no lower bound on the price relatives. The geometry cannot set a positive entry to zero, and a zero entry stays zero. The geometry refuses a negative lower bound, because the logarithm is undefined below zero.

# Mathematical definition

```math
\\begin{align}
\\Psi(\\boldsymbol{w}) &= -\\sum_i \\log w_i\\,, \\\\
D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q}) &= \\sum_i \\left( \\frac{w_i}{q_i} - 1 - \\log \\frac{w_i}{q_i} \\right)\\,, \\\\
\\mathrm{Proj}^{\\mathrm{IS}}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q})\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(\\frac{1}{1 / q_i + \\theta},\\, l_i\\right),\\, u_i\\right)\\,, \\\\
w_{t+1, i} &= \\frac{1}{1 / w_{t, i} + \\eta_t g_{t, i} + \\theta}\\,.
\\end{align}
```

Where:

  - ``\\Psi``: Burg entropy, the potential of this geometry.
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:g_t_loss])

The mirror image is ``\\nabla \\Psi(\\boldsymbol{w})_i = -1 / w_i``, so the projection sets ``\\nabla \\Psi(\\boldsymbol{w}) - \\nabla \\Psi(\\boldsymbol{q})`` to one constant over the free entries. On a [`BoundedAllocationSet`](@ref) the projection is ``\\boldsymbol{w}(\\theta)`` at the root. A base ``1 / q_i + \\theta`` at or below zero puts its asset at ``u_i``. The budget ``\\sum_i w_i(\\theta)`` does not increase in ``\\theta``. The last line is a first-order step from ``\\boldsymbol{w}_t`` and its projection onto the simplex together, one scalar root. The Hessian of ``\\Psi`` is ``\\mathrm{diag}(1 / w_i^2)``, so the step moves a small weight in proportion to its square.

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

Takes the unconstrained mirror step from the iterate `u` along the scaled gradient `s`, in the potential of the geometry.

The answer is the raw step that [`project`](@ref) then puts onto the Allocation Set. A barrier geometry has a step only while every base is positive. A base at or below zero is a step to an unbounded allocation, and the function refuses it and does not clip it. Under the plain log-wealth gradient every rate below one has a step. A transformed gradient changes that condition. The first step of [`AdaptiveMomentGradient`](@ref) with `gamma1 = 0.9` and `gamma2 = 0.999` scales every entry of the gradient to about `-3.16`, so at the rate `0.9` the log-barrier step fails when one asset holds more than 0.352 of the iterate.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{q} &= \\nabla \\Psi^{*}\\left(\\nabla \\Psi(\\boldsymbol{u}) - \\boldsymbol{s}\\right)\\,, \\\\
q^{\\mathrm{E}}_i &= u_i - s_i\\,, \\\\
q^{\\mathrm{KL}}_i &= u_i \\exp(-s_i)\\,, \\\\
q^{\\alpha}_i &= \\left( u_i^{\\alpha - 1} + (1 - \\alpha) s_i \\right)^{1 / (\\alpha - 1)}\\,, \\\\
q^{\\mathrm{IS}}_i &= \\frac{1}{1 / u_i + s_i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:q_raw])
  - $(math_dict[:Psi_pot])
  - ``\\nabla \\Psi^{*}``: Inverse of the mirror map ``\\nabla \\Psi``.
  - ``\\boldsymbol{u}``: The iterate the step starts from.
  - ``\\boldsymbol{s}``: The scaled gradient, ``\\eta_t \\boldsymbol{g}_t`` for a plain gradient.
  - ``q^{\\mathrm{E}}``, ``q^{\\mathrm{KL}}``, ``q^{\\alpha}``, ``q^{\\mathrm{IS}}``: The raw step under [`EuclideanProjection`](@ref), [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref).
  - ``\\alpha``: Power of the Tsallis potential, in ``(0, 1)``.
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:g_t_loss])
  - $(math_dict[:x_t_rel])

Under the log-wealth loss with a plain gradient, ``\\boldsymbol{s} = -\\eta_t \\boldsymbol{x}_t / \\langle \\boldsymbol{u}, \\boldsymbol{x}_t \\rangle``. The log-barrier base ``1 / u_i + s_i`` is then positive exactly while ``\\eta_t \\hat{u}_i < 1`` for every asset, with ``\\hat{\\boldsymbol{u}} = \\boldsymbol{u} \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{u}, \\boldsymbol{x}_t \\rangle`` the Price-Adjusted Allocation of the iterate. Each ``\\hat{u}_i`` is at most one, so a rate below one always has a step. The Tsallis base is positive exactly while ``(1 - \\alpha) \\eta_t \\hat{u}_i u_i^{-\\alpha} < 1`` for every asset.

# Arguments

  - `proj`: The Projection Geometry.
  - `u`: The iterate the step starts from.
  - `s`: The scaled gradient.

# Validation

  - Under [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): every base is positive. A `DomainError` is thrown otherwise ([`assert_mirror_base`](@ref)).

# Returns

  - `q::AbstractVector`: The raw step.

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

Refuses a barrier mirror step that has a base at or below zero.

The error names the scaled gradient `s` that gave the base.

# Validation

  - `all(> 0, base)`. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`mirror_step`](@ref)
"""
function assert_mirror_base(base::AbstractVector, s::AbstractVector)::Nothing
    @argcheck(all(x -> x > zero(x), base),
              DomainError(s,
                          "the mirror step leaves the geometry's domain: a base of the barrier potential is not positive, so the unconstrained step is unbounded in some asset. Lower the learning rate. Under the log barrier with the plain log-wealth gradient, the step exists while `eta * w_i * x_i / <w, x> < 1` for every asset."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Projects a raw step onto a bounded set in a barrier geometry, through the scalar root of the budget.

The root is the budget multiplier `mu` at which `Σ_i clip(phi(b_i + mu), lb_i, ub_i)` is one. Here `b` is the mirror image of the raw step and `phi` is the inverse mirror map. A base at or below zero puts its asset at its cap. `phi` decreases on positive bases and has no bound as the base falls to zero, so the clipped budget does not increase in `mu` and has no jump. A zero raw entry has the base `Inf`, and it stays at its floor. [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref) state the root as ``\\boldsymbol{w}(\\theta)``.

# Algorithm

 1. Check that `Σ lb ≤ 1 ≤ Σ ub` with [`assert_feasible_bounds`](@ref).
 2. When the floors sum to one, return `lb`.
 3. Collect the finite entries of `b` in `finite`, and refuse an empty `finite`.
 4. Write the clipped budget `f(mu)`. An asset whose base `b_i + mu` is at or below zero takes `ub_i`, and every other asset takes `clamp(phi(b_i + mu), lb_i, ub_i)`.
 5. Set the lower end `lo = -maximum(finite)`, at which every finite entry is at its cap. Refuse `f(lo) < 1`.
 6. Find the upper end `hi` with [`barrier_upper_bracket`](@ref).
 7. Bisect `[lo, hi]` for `mu` with [`bounded_root`](@ref).
 8. Return the clipped allocation at `mu`.

# Arguments

  - `b`: The mirror image of the raw step, `Inf` at a zero entry.
  - `phi`: The inverse mirror map of a positive base.
  - `wb`: The resolved bounds.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub`. An `ArgumentError` is thrown otherwise.
  - At least one raw entry is positive. A `DomainError` is thrown otherwise.
  - The caps of the positive entries and the floors of the zero entries reach the budget. A `DomainError` is thrown otherwise, because a zero entry stays at its floor under a barrier.

# Returns

  - `w'::Vector`: The projected allocation, a new vector.

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

Finds the upper end of the bracket of a barrier root, a positive multiplier above `lo` at which the clipped budget is at most one.

The loop stops, because the clipped budget tends to `Σ lb < 1` as the multiplier grows. [`barrier_projection`](@ref) returns `lb` before it calls this function when `Σ lb = 1`.

# Algorithm

 1. Start from `hi = max(lo, 0) + 1`.
 2. Double `hi` while `f(hi) > 1`.
 3. Return `hi`.

# Arguments

  - `f`: The clipped budget as a function of the multiplier, non-increasing.
  - `lo`: The lower end of the bracket.

# Returns

  - `hi`: The upper end of the bracket.

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

Projects the raw step `q` onto a [`BoundedAllocationSet`](@ref) in the Tsallis or the log-barrier geometry. These are the barrier arms of the Constrained Update, and `w` is the Price-Adjusted Allocation.

Neither potential has a closed-form projection onto the simplex, so the arms use the scalar root of [`barrier_projection`](@ref) on every bound, the simplex included. A root whose allocation misses the budget in floating point gives the Held Step ([`budget_or_held_step`](@ref)). This happens, for example, under the log barrier when a raw entry is `1e-20`. Its base is `1e20`, and the multiplier that brings the entry to a share of the budget cancels against that base.

# Algorithm

 1. Refuse a negative or non-finite entry of `q` and a negative floor, with [`assert_barrier_raw_step`](@ref).
 2. Map `q` to its mirror image `b`, `q .^ (alpha - 1)` under the Tsallis geometry and `inv.(q)` under the log barrier. A zero entry maps to `Inf`.
 3. Find the allocation `wn` with [`barrier_projection`](@ref), on `b`, the inverse mirror map and the bounds of `set`.
 4. Return `wn`, or the Held Step `w` when `wn` misses the budget in floating point, with [`budget_or_held_step`](@ref).

# Arguments

  - `proj`: The barrier geometry.
  - `set`: The bounded set, its bounds resolved.
  - `q`: The raw step.
  - `w`: The Price-Adjusted Allocation the step trades from, the answer of a Held Step.

# Validation

  - `all(>= 0, q)` and every entry finite. A `DomainError` is thrown otherwise.
  - `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise.
  - Everything [`barrier_projection`](@ref) refuses.

# Returns

  - `w'::Vector`: The projected allocation, or `w` copied on a Held Step.

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

Refuses, for a barrier geometry, a raw step with a negative or non-finite entry, and a bound with a negative floor.

The potential of a barrier geometry is undefined below zero.

# Validation

  - Every entry of `q` is finite and non-negative. A `DomainError` is thrown otherwise.
  - `all(>= 0, wb.lb)`. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

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

Refuses a negative lower bound under a geometry whose potential is undefined below zero.

The head calls this function at its construction, with the geometry of its rule and its set. The entropic, Tsallis and log-barrier geometries refuse a negative bound. This function refuses a bound that is a value. The projection refuses a bound that an estimator resolves. The Euclidean, Gram and diagonal geometries admit a negative bound, so a long-short rule needs no other set.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) or [`LogBarrierProjection`](@ref) with a [`WeightBounds`](@ref) whose `lb` is a number or a vector: `all(>= 0, lb)`. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
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
