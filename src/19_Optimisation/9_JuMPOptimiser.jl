struct ProcessedJuMPOptimiserAttributes{
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
    T16,
    T17,
    T18,
    T19,
} <: AbstractResult
    pr::T1
    wb::T2
    lt::T3
    st::T4
    lcs::T5
    cent::T6
    gcard::T7
    sgcard::T8
    smtx::T9
    sgmtx::T10
    slt::T11
    sst::T12
    sglt::T13
    sgst::T14
    nplg::T15
    cplg::T16
    tn::T17
    fees::T18
    ret::T19
end
struct ProcessedFactorRiskBudgettingAttributes{T1,T2,T3} <: AbstractResult
    rkb::T1
    b1::T2
    rr::T3
end
struct ProcessedAssetRiskBudgettingAttributes{T1} <: AbstractResult
    rkb::T1
end
struct JuMPOptimisation{T1,T2,T3,T4,T5} <: OptimisationResult
    oe::T1
    pa::T2
    retcode::T3
    sol::T4
    model::T5
end
struct JuMPOptimisationFactorRiskContribution{T1,T2,T3,T4,T5,T6,T7,T8} <: OptimisationResult
    oe::T1
    pa::T2
    rr::T3
    frc_nplg::T4
    frc_cplg::T5
    retcode::T6
    sol::T7
    model::T8
end
struct JuMPOptimisationRiskBudgetting{T1,T2,T3,T4,T5,T6} <: OptimisationResult
    oe::T1
    pa::T2
    prb::T3
    retcode::T4
    sol::T5
    model::T6
end
function Base.getproperty(r::JuMPOptimisation, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? getfield(r.sol, :w) : getfield.(r.sol, :w)
    elseif sym in (:oe, :pa, :retcode, :sol, :model)
        getfield(r, sym)
    else
        getfield(r.pa, sym)
    end
end
function Base.getproperty(r::JuMPOptimisationFactorRiskContribution, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? getfield(r.sol, :w) : getfield.(r.sol, :w)
    elseif sym in (:oe, :pa, :frc_nplg, :frc_cplg, :retcode, :sol, :model)
        getfield(r, sym)
    else
        getfield(r.pa, sym)
    end
end
function Base.getproperty(r::JuMPOptimisationRiskBudgetting, sym::Symbol)
    return if sym == :w
        r.sol.w
    elseif sym in (:oe, :pa, :prb, :retcode, :sol, :model)
        getfield(r, sym)
    else
        getfield(r.pa, sym)
    end
end
function assert_finite_nonnegative_real_or_vec(val::Real)
    @argcheck(isfinite(val) && val > zero(val))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::AbstractVector{<:Real})
    @argcheck(
        any(isfinite, val) && any(x -> x > zero(x), val) && all(x -> x >= zero(x), val)
    )
    return nothing
end
struct JuMPOptimiser{
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15,
    T16,
    T17,
    T18,
    T19,
    T20,
    T21,
    T22,
    T23,
    T24,
    T25,
    T26,
    T27,
    T28,
    T29,
    T30,
    T31,
    T32,
    T33,
    T34,
    T35,
    T36,
    T37,
} <: BaseJuMPOptimisationEstimator
    pe::T1 # PriorEstimator
    slv::T2
    wb::T3 # WeightBounds
    bgt::T4 # BudgetRange
    sbgt::T5 # LongShortSum
    lt::T6 # l threshold
    st::T7
    lcs::T8
    lcm::T9
    cent::T10
    gcard::T11
    sgcard::T12
    smtx::T13
    sgmtx::T14
    slt::T15
    sst::T16
    sglt::T17
    sgst::T18
    sets::T19
    nplg::T20
    cplg::T21
    tn::T22 # Turnover
    te::T23 # TrackingError
    fees::T24
    ret::T25
    sce::T26
    ccnt::T27
    cobj::T28
    sc::T29
    so::T30
    card::T31
    scard::T32
    nea::T33
    l1::T34
    l2::T35
    ss::T36
    strict::T37
end
function JuMPOptimiser(;
    pe::Union{<:AbstractPriorEstimator,<:AbstractPriorResult} = EmpiricalPrior(),
    slv::Union{<:Solver,<:AbstractVector{<:Solver}},
    wb::Union{Nothing,<:WeightBoundsEstimator,<:WeightBounds} = WeightBounds(),
    bgt::Union{Nothing,<:Real,<:BudgetConstraintEstimator} = 1.0,
    sbgt::Union{Nothing,<:Real,<:BudgetRange} = nothing,
    lt::Union{Nothing,<:BuyInThresholdEstimator,<:BuyInThreshold} = nothing,
    st::Union{Nothing,<:BuyInThresholdEstimator,<:BuyInThreshold} = nothing,
    lcs::Union{Nothing,<:LinearConstraintEstimator,<:LinearConstraint} = nothing,
    lcm::Union{Nothing,<:LinearConstraint} = nothing,
    cent::Union{
        Nothing,
        <:CentralityEstimator,
        <:AbstractVector{<:CentralityEstimator},
        <:LinearConstraint,
    } = nothing,
    gcard::Union{Nothing,<:LinearConstraintEstimator,<:LinearConstraint} = nothing,
    sgcard::Union{
        Nothing,
        <:LinearConstraintEstimator,
        <:LinearConstraint,
        <:AbstractVector{<:Union{<:LinearConstraintEstimator,<:LinearConstraint}},
    } = nothing,
    smtx::Union{
        Nothing,
        <:AssetSetsMatrixEstimator,
        <:AbstractMatrix,
        <:AbstractVector{<:Union{<:AssetSetsMatrixEstimator,<:AbstractMatrix}},
    } = nothing,
    sgmtx::Union{
        Nothing,
        <:AssetSetsMatrixEstimator,
        <:AbstractMatrix,
        <:AbstractVector{<:Union{<:AssetSetsMatrixEstimator,<:AbstractMatrix}},
    } = nothing,
    slt::Union{
        Nothing,
        <:BuyInThreshold,
        <:BuyInThresholdEstimator,
        <:AbstractVector{<:Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}},
    } = nothing,
    sst::Union{
        Nothing,
        <:BuyInThreshold,
        <:BuyInThresholdEstimator,
        <:AbstractVector{<:Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}},
    } = nothing,
    sglt::Union{
        Nothing,
        <:BuyInThreshold,
        <:BuyInThresholdEstimator,
        <:AbstractVector{<:Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}},
    } = nothing,
    sgst::Union{
        Nothing,
        <:BuyInThreshold,
        <:BuyInThresholdEstimator,
        <:AbstractVector{<:Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}},
    } = nothing,
    sets::Union{Nothing,<:AssetSets} = nothing,
    nplg::Union{Nothing,<:PhylogenyEstimator,<:PhylogenyResult} = nothing,
    cplg::Union{Nothing,<:PhylogenyEstimator,<:PhylogenyResult} = nothing,
    tn::Union{
        Nothing,
        <:TurnoverEstimator,
        <:Turnover,
        <:AbstractVector{<:Union{<:TurnoverEstimator,<:Turnover}},
    } = nothing,
    te::Union{Nothing,<:AbstractTracking,<:AbstractVector{<:AbstractTracking}} = nothing,
    fees::Union{Nothing,<:FeesEstimator,<:Fees} = nothing,
    ret::JuMPReturnsEstimator = ArithmeticReturn(),
    sce::Scalariser = SumScalariser(),
    ccnt::Union{Nothing,<:CustomJuMPConstraint} = nothing,
    cobj::Union{Nothing,<:CustomJuMPObjective} = nothing,
    sc::Real = 1,
    so::Real = 1,
    card::Union{Nothing,<:Integer} = nothing,
    scard::Union{Nothing,<:Integer,<:AbstractVector{<:Integer}} = nothing,
    nea::Union{Nothing,<:Real} = nothing,
    l1::Union{Nothing,<:Real} = nothing,
    l2::Union{Nothing,<:Real} = nothing,
    ss::Union{Nothing,<:Real} = nothing,
    strict::Bool = false,
)
    if isa(bgt, Real)
        @argcheck(isfinite(bgt))
    elseif isa(bgt, BudgetCostEstimator)
        @argcheck(isnothing(sbgt))
    end
    if isa(sbgt, Real)
        @argcheck(isfinite(sbgt) && sbgt >= 0)
    end
    if isa(cent, AbstractVector)
        @argcheck(!isempty(cent))
    end
    if !isnothing(card)
        @argcheck(isfinite(card) && card > 0)
    end
    if isa(scard, Integer)
        @argcheck(isfinite(scard) && scard > 0)
        @argcheck(isa(smtx, Union{<:AssetSetsMatrixEstimator,<:AbstractMatrix}))
        @argcheck(isa(slt, Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}))
        @argcheck(isa(sst, Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}))
    elseif isa(scard, AbstractVector)
        @argcheck(!isempty(scard))
        @argcheck(all(isfinite, scard) && all(x -> x > 0, scard))
        @argcheck(isa(smtx, AbstractVector))
        @argcheck(length(scard) == length(smtx))
        if isa(slt, AbstractVector)
            @argcheck(!isempty(slt))
            @argcheck(length(scard) == length(slt))
        end
        if isa(sst, AbstractVector)
            @argcheck(!isempty(sst))
            @argcheck(length(scard) == length(sst))
        end
    elseif isnothing(scard) && (
        isa(slt, Union{<:BuyInThreshold,<:BuyInThresholdEstimator}) ||
        isa(sst, Union{<:BuyInThreshold,<:BuyInThresholdEstimator})
    )
        @argcheck(isa(smtx, Union{<:AssetSetsMatrixEstimator,<:AbstractMatrix}))
    elseif isnothing(scard) && (isa(slt, AbstractVector) || isa(sst, AbstractVector))
        @argcheck(isa(smtx, AbstractVector))
        @argcheck(!isempty(smtx))
        if isa(slt, AbstractVector)
            @argcheck(!isempty(slt))
            @argcheck(length(slt) == length(smtx))
        end
        if isa(sst, AbstractVector)
            @argcheck(!isempty(sst))
            @argcheck(length(sst) == length(smtx))
        end
    end
    if isa(sgcard, Union{<:LinearConstraintEstimator,<:LinearConstraint})
        @argcheck(isa(sgmtx, Union{<:AssetSetsMatrixEstimator,<:AbstractMatrix}))
        @argcheck(isa(sglt, Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}))
        @argcheck(isa(sgst, Union{Nothing,<:BuyInThreshold,<:BuyInThresholdEstimator}))
        if isa(sgcard, LinearConstraint) && isa(smtx, AbstractMatrix)
            N = size(smtx, 1)
            N_ineq = !isnothing(sgcard.ineq) ? length(sgcard.B_ineq) : 0
            N_eq = !isnothing(sgcard.eq) ? length(sgcard.B_eq) : 0
            @argcheck(N == N_ineq + N_eq)
        end
    elseif isa(sgcard, AbstractVector)
        @argcheck(!isempty(sgcard))
        @argcheck(isa(sgmtx, AbstractVector))
        @argcheck(!isempty(sgmtx))
        @argcheck(length(sgcard) == length(sgmtx))
        if isa(sglt, AbstractVector)
            @argcheck(!isempty(sglt))
            @argcheck(length(sgcard) == length(sglt))
        end
        if isa(sgst, AbstractVector)
            @argcheck(length(sgcard) == length(sgst))
        end
        for (sgc, smt) in zip(sgcard, sgmtx)
            if isa(sgc, LinearConstraint) && isa(smt, AbstractMatrix)
                N = size(smt, 1)
                N_ineq = !isnothing(sgc.ineq) ? length(sgc.B_ineq) : 0
                N_eq = !isnothing(sgc.eq) ? length(sgc.B_eq) : 0
                @argcheck(N == N_ineq + N_eq)
            end
        end
    elseif isnothing(sgcard) && (
        isa(sglt, Union{<:BuyInThreshold,<:BuyInThresholdEstimator}) ||
        isa(sgst, Union{<:BuyInThreshold,<:BuyInThresholdEstimator})
    )
        @argcheck(isa(sgmtx, Union{<:AssetSetsMatrixEstimator,<:AbstractMatrix}))
    elseif isnothing(sgcard) && (isa(sglt, AbstractVector) || isa(sgst, AbstractVector))
        @argcheck(isa(sgmtx, AbstractVector))
        @argcheck(!isempty(sgmtx))
        if isa(sglt, AbstractVector)
            @argcheck(!isempty(sglt))
            @argcheck(length(sglt) == length(sgmtx))
        end
        if isa(sgst, AbstractVector)
            @argcheck(!isempty(sgst))
            @argcheck(length(sgst) == length(sgmtx))
        end
    end
    if isa(wb, WeightBoundsEstimator) ||
       isa(lt, BuyInThresholdEstimator) ||
       isa(st, BuyInThresholdEstimator) ||
       isa(lcs, LinearConstraintEstimator) ||
       isa(cent, LinearConstraintEstimator) ||
       isa(gcard, LinearConstraintEstimator) ||
       !isa(
           sgcard,
           Union{Nothing,<:LinearConstraint,<:AbstractVector{<:LinearConstraint}},
       ) ||
       !isnothing(scard) ||
       isa(fees, FeesEstimator)
        @argcheck(!isnothing(sets))
    end
    if isa(tn, AbstractVector)
        @argcheck(!isempty(tn))
    end
    if isa(te, AbstractVector)
        @argcheck(!isempty(te))
    end
    if !isnothing(nea)
        @argcheck(nea > zero(nea))
    end
    if isa(slv, AbstractVector)
        @argcheck(!isempty(slv))
    end
    return JuMPOptimiser(
        pe,
        slv,
        wb,
        bgt,
        sbgt,
        lt,
        st,
        lcs,
        lcm,
        cent,
        gcard,
        sgcard,
        smtx,
        sgmtx,
        slt,
        sst,
        sglt,
        sgst,
        sets,
        nplg,
        cplg,
        tn,
        te,
        fees,
        ret,
        sce,
        ccnt,
        cobj,
        sc,
        so,
        card,
        scard,
        nea,
        l1,
        l2,
        ss,
        strict,
    )
end
function opt_view(opt::JuMPOptimiser, i::AbstractVector, X::AbstractMatrix)
    X = isa(opt.pe, AbstractPriorResult) ? opt.pe.X : X
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    bgt = budget_view(opt.bgt, i)
    lt = threshold_view(opt.lt, i)
    st = threshold_view(opt.st, i)
    if opt.smtx === opt.sgmtx
        smtx = sgmtx = asset_sets_matrix_view(opt.smtx, i)
    else
        smtx = asset_sets_matrix_view(opt.smtx, i)
        sgmtx = asset_sets_matrix_view(opt.sgmtx, i)
    end
    if opt.slt === opt.sglt
        slt = sglt = threshold_view(opt.slt, i)
    else
        slt = threshold_view(opt.slt, i)
        sglt = threshold_view(opt.sglt, i)
    end
    if opt.sst === opt.sgst
        sst = sgst = threshold_view(opt.sst, i)
    else
        sst = threshold_view(opt.sst, i)
        sgst = threshold_view(opt.sgst, i)
    end
    sets = nothing_asset_sets_view(opt.sets, i)
    tn = turnover_view(opt.tn, i)
    te = tracking_view(opt.te, i, X)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(;
        pe = pe,
        slv = opt.slv,
        wb = wb,
        bgt = bgt,
        sbgt = opt.sbgt,
        lt = lt,
        st = st,
        lcs = opt.lcs,
        lcm = opt.lcm,
        cent = opt.cent,
        gcard = opt.gcard,
        sgcard = opt.sgcard,
        smtx = smtx,
        sgmtx = sgmtx,
        slt = slt,
        sst = sst,
        sglt = sglt,
        sgst = sgst,
        sets = sets,
        nplg = opt.nplg,
        cplg = opt.cplg,
        tn = tn,
        te = te,
        fees = fees,
        ret = ret,
        sce = opt.sce,
        ccnt = ccnt,
        cobj = cobj,
        sc = opt.sc,
        so = opt.so,
        card = opt.card,
        scard = opt.scard,
        nea = opt.nea,
        l1 = opt.l1,
        l2 = opt.l2,
        ss = opt.ss,
        strict = opt.strict,
    )
end
function processed_jump_optimiser_attributes(
    opt::JuMPOptimiser,
    rd::ReturnsResult;
    dims::Int = 1,
)
    pr = prior(opt.pe, rd; dims = dims)
    datatype = eltype(pr.X)
    wb = weight_bounds_constraints(
        opt.wb,
        opt.sets;
        N = size(pr.X, 2),
        strict = opt.strict,
        datatype = datatype,
    )
    lt = threshold_constraints(opt.lt, opt.sets; datatype = datatype, strict = opt.strict)
    st = threshold_constraints(opt.st, opt.sets; datatype = datatype, strict = opt.strict)
    lcs = linear_constraints(opt.lcs, opt.sets; datatype = datatype, strict = opt.strict)
    cent = centrality_constraints(opt.cent, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    gcard = linear_constraints(opt.gcard, opt.sets; datatype = Int, strict = opt.strict)
    sgcard = linear_constraints(opt.sgcard, opt.sets; datatype = Int, strict = opt.strict)
    if opt.smtx === opt.sgmtx
        smtx = sgmtx = asset_sets_matrix(opt.smtx, opt.sets)
    else
        smtx = asset_sets_matrix(opt.smtx, opt.sets)
        sgmtx = asset_sets_matrix(opt.sgmtx, opt.sets)
    end
    if opt.slt === opt.sglt
        slt =
            sglt = threshold_constraints(
                opt.slt,
                opt.sets;
                datatype = datatype,
                strict = opt.strict,
            )
    else
        slt = threshold_constraints(
            opt.slt,
            opt.sets;
            datatype = datatype,
            strict = opt.strict,
        )
        sglt = threshold_constraints(
            opt.sglt,
            opt.sets;
            datatype = datatype,
            strict = opt.strict,
        )
    end
    if opt.sst === opt.sgst
        sst =
            sgst = threshold_constraints(
                opt.sst,
                opt.sets;
                datatype = datatype,
                strict = opt.strict,
            )
    else
        sst = threshold_constraints(
            opt.sst,
            opt.sets;
            datatype = datatype,
            strict = opt.strict,
        )
        sgst = threshold_constraints(
            opt.sgst,
            opt.sets;
            datatype = datatype,
            strict = opt.strict,
        )
    end
    nplg = phylogeny_constraints(opt.nplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    cplg = phylogeny_constraints(opt.cplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    tn = turnover_constraints(opt.tn, opt.sets; strict = opt.strict)
    fees = fees_constraints(opt.fees, opt.sets; datatype = datatype, strict = opt.strict)
    ret = jump_returns_factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(
        pr,
        wb,
        lt,
        st,
        lcs,
        cent,
        gcard,
        sgcard,
        smtx,
        sgmtx,
        slt,
        sst,
        sglt,
        sgst,
        nplg,
        cplg,
        tn,
        fees,
        ret,
    )
end
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(propertynames(opt), (:ret,)))
    return JuMPOptimiser(;
        ret = no_bounds_returns_estimator(opt.ret, args...),
        NamedTuple{pnames}(getproperty.(Ref(opt), pnames))...,
    )
end
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1)
    (;
        pr,
        wb,
        lt,
        st,
        lcs,
        cent,
        gcard,
        sgcard,
        smtx,
        sgmtx,
        slt,
        sst,
        sglt,
        sgst,
        nplg,
        cplg,
        tn,
        fees,
        ret,
    ) = processed_jump_optimiser_attributes(opt, rd; dims = dims)
    return JuMPOptimiser(;
        pe = pr,
        slv = opt.slv,
        wb = wb,
        bgt = opt.bgt,
        sbgt = opt.sbgt,
        lt = lt,
        st = st,
        lcs = lcs,
        lcm = opt.lcm,
        cent = cent,
        gcard = gcard,
        sgcard = sgcard,
        smtx = smtx,
        sgmtx = sgmtx,
        slt = slt,
        sst = sst,
        sglt = sglt,
        sgst = sgst,
        sets = opt.sets,
        nplg = nplg,
        cplg = cplg,
        tn = tn,
        te = opt.te,
        fees = fees,
        ret = ret,
        sce = opt.sce,
        ccnt = opt.ccnt,
        cobj = opt.cobj,
        sc = opt.sc,
        so = opt.so,
        card = opt.card,
        nea = opt.nea,
        l1 = opt.l1,
        l2 = opt.l2,
        ss = opt.ss,
        strict = opt.strict,
    )
end

export ProcessedJuMPOptimiserAttributes,
    JuMPOptimisation,
    JuMPOptimisationRiskBudgetting,
    JuMPOptimisationFactorRiskContribution,
    JuMPOptimiser
