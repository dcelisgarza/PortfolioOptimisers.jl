struct ProcessedJuMPOptimiserAttributes{T1 <: AbstractPriorResult,
                                        T2 <: Union{Nothing, <:WeightBounds},
                                        T3 <: Union{Nothing, <:BuyInThreshold},
                                        T4 <: Union{Nothing, <:BuyInThreshold},
                                        T5 <: Union{Nothing, <:LinearConstraint},
                                        T6 <: Union{Nothing, <:LinearConstraint},
                                        T7 <: Union{Nothing, <:LinearConstraint},
                                        T8 <: Union{Nothing, <:LinearConstraint,
                                                    <:AbstractVector{<:LinearConstraint}},
                                        T9 <: Union{Nothing, <:AbstractMatrix,
                                                    <:AbstractVector{<:AbstractMatrix}},
                                        T10 <: Union{Nothing, <:BuyInThreshold,
                                                     <:AbstractVector{<:BuyInThreshold},
                                                     <:AbstractVector{<:Union{Nothing,
                                                                              <:BuyInThreshold}}},
                                        T11 <: Union{Nothing, <:BuyInThreshold,
                                                     <:AbstractVector{<:BuyInThreshold},
                                                     <:AbstractVector{<:Union{Nothing,
                                                                              <:BuyInThreshold}}},
                                        T12 <: Union{Nothing, <:AbstractMatrix,
                                                     <:AbstractVector{<:AbstractMatrix}},
                                        # T13 <: Union{Nothing, <:BuyInThreshold,
                                        #              <:AbstractVector{<:BuyInThreshold},
                                        #              <:AbstractVector{<:Union{Nothing,
                                        #                                       <:BuyInThreshold}}},
                                        # T14 <: Union{Nothing, <:BuyInThreshold,
                                        #              <:AbstractVector{<:BuyInThreshold},
                                        #              <:AbstractVector{<:Union{Nothing,
                                        #                                       <:BuyInThreshold}}},
                                        T15 <: Union{Nothing, <:PhilogenyResult},
                                        T16 <: Union{Nothing, <:PhilogenyResult},
                                        T17 <: Union{Nothing, <:Turnover,
                                                     <:AbstractVector{<:Turnover}},
                                        T18 <: Union{Nothing, <:Fees},
                                        T19 <: JuMPReturnsEstimator} <: AbstractResult
    pr::T1
    wb::T2
    lt::T3
    st::T4
    lcs::T5
    cent::T6
    gcard::T7
    sgcard::T8
    smtx::T9
    slt::T10
    sst::T11
    sgmtx::T12
    # sglt::T13
    # sgst::T14
    nplg::T15
    cplg::T16
    tn::T17
    fees::T18
    ret::T19
end
struct JuMPOptimisation{T1 <: Type, T2 <: ProcessedJuMPOptimiserAttributes,
                        T3 <: Union{<:OptimisationReturnCode,
                                    <:AbstractVector{<:OptimisationReturnCode}},
                        T4 <: Union{<:JuMPOptimisationSolution,
                                    <:AbstractVector{<:JuMPOptimisationSolution}},
                        T5 <: Union{Nothing, JuMP.Model}} <: OptimisationResult
    oe::T1
    pa::T2
    retcode::T3
    sol::T4
    model::T5
end
struct JuMPOptimisationFactorRiskContribution{T1 <: Type,
                                              T2 <: ProcessedJuMPOptimiserAttributes,
                                              T11 <:
                                              Union{Nothing, <:SemiDefinitePhilogeny},
                                              T12 <:
                                              Union{Nothing, <:SemiDefinitePhilogeny},
                                              T13 <: OptimisationReturnCode,
                                              T14 <: JuMPOptimisationSolution,
                                              T15 <: Union{Nothing, JuMP.Model}} <:
       OptimisationResult
    oe::T1
    pa::T2
    frc_nplg::T11
    frc_cplg::T12
    retcode::T13
    sol::T14
    model::T15
end
function Base.getproperty(r::Union{<:JuMPOptimisation,
                                   <:JuMPOptimisationFactorRiskContribution}, sym::Symbol)
    return if sym == :pr
        r.pa.pr
    elseif sym == :wb
        r.pa.wb
    elseif sym == :lt
        r.pa.lt
    elseif sym == :st
        r.pa.st
    elseif sym == :lcs
        r.pa.lcs
    elseif sym == :cent
        r.pa.cent
    elseif sym == :gcard
        r.pa.gcard
    elseif sym == :sgcard
        r.pa.sgcard
    elseif sym == :smtx
        r.pa.smtx
    elseif sym == :slt
        r.pa.slt
    elseif sym == :sst
        r.pa.sst
    elseif sym == :sgmtx
        r.pa.sgmtx
    elseif sym == :sglt
        r.pa.sglt
    elseif sym == :sgst
        r.pa.sgst
    elseif sym == :nplg
        r.pa.nplg
    elseif sym == :cplg
        r.pa.cplg
    elseif sym == :tn
        r.pa.tn
    elseif sym == :fees
        r.pa.fees
    elseif sym == :ret
        r.pa.ret
    elseif sym == :w
        !isa(r.sol, AbstractVector) ? r.sol.w : getproperty.(r.sol, :w)
    else
        getfield(r, sym)
    end
end
struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                     T2 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                     T3 <: Union{Nothing, <:WeightBounds, <:WeightBoundsEstimator},
                     T4 <: Union{Nothing, <:Real, <:BudgetRange, <:BudgetCosts},
                     T5 <: Union{Nothing, <:Real, <:BudgetRange},
                     T6 <: Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                       #! Start: to delete
                                 <:AbstractDict, <:AbstractVector{<:Pair{<:Any, <:Real}}
                       #! End: to delete
                       },
                     T7 <: Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                       #! Start: to delete
                                 <:AbstractDict, <:AbstractVector{<:Pair{<:Any, <:Real}}
                       #! End: to delete
                       },
                     T8 <: Union{Nothing, <:AbstractString, Expr,
                                 <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                 <:AbstractVector{<:Union{<:AbstractString, Expr}},
                       #! Start: to delete
                                 <:LinearConstraintEstimator,
                                 <:AbstractVector{<:LinearConstraintEstimator},
                       #! End: to delete
                                 <:LinearConstraint},
                     T9 <: Union{Nothing, <:LinearConstraint},
                     T10 <: Union{Nothing, <:CentralityEstimator,
                                  <:AbstractVector{<:CentralityEstimator}, <:LinearConstraint},
                     T11 <: Union{Nothing, <:AbstractString, Expr,
                                  <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                  <:AbstractVector{<:Union{<:AbstractString, Expr}},
                       #! Start: to delete
                                  <:LinearConstraintEstimator,
                                  <:AbstractVector{<:LinearConstraintEstimator},
                       #! End: to delete
                                  <:LinearConstraint},
                     T12 <: Union{Nothing, <:AbstractString, Expr,
                                  <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                  <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                  <:AbstractVector{<:AbstractVector{<:AbstractString}},
                                  <:AbstractVector{<:AbstractVector{Expr}},
                                  <:AbstractVector{<:AbstractVector{<:Union{<:AbstractString,
                                                                            Expr}}},
                       #! Start: to delete
                                  <:LinearConstraintEstimator,
                                  <:AbstractVector{<:LinearConstraintEstimator},
                       #! End: to delete
                                  <:LinearConstraint, <:AbstractVector{<:LinearConstraint}},
                     T13 <: Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                  <:AbstractVector{Symbol}, <:AbstractVector{<:AbstractString},
                                  <:AbstractVector{<:AbstractMatrix}},
                     T14 <: Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                  <:AbstractVector{<:Pair{<:Any, <:Real}},
                                  <:AbstractVector{<:AbstractDict},
                                  <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                  <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                           <:AbstractVector{<:Pair{<:Any, <:Real}}}}},
                     T15 <: Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                  <:AbstractVector{<:Pair{<:Any, <:Real}},
                                  <:AbstractVector{<:AbstractDict},
                                  <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                  <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                           <:AbstractVector{<:Pair{<:Any, <:Real}}}}},
                     T16 <: Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                  <:AbstractVector{Symbol}, <:AbstractVector{<:AbstractString},
                                  <:AbstractVector{<:AbstractMatrix}},
                     #  T17 <: Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                     #               <:AbstractVector{<:Pair{<:Any, <:Real}},
                     #               <:AbstractVector{<:AbstractDict},
                     #               <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                     #               <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                     #                                        <:AbstractVector{<:Pair{<:Any, <:Real}}}}},
                     #  T18 <: Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                     #               <:AbstractVector{<:Pair{<:Any, <:Real}},
                     #               <:AbstractVector{<:AbstractDict},
                     #               <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                     #               <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                     #                                        <:AbstractVector{<:Pair{<:Any, <:Real}}}}},
                     T19 <: Union{Nothing, <:AssetSets,
                       #! Start: to delete
                                  <:DataFrame
                       #! End: to delete
                       }, T20 <: Union{Nothing, <:PhilogenyEstimator, <:PhilogenyResult},
                     T21 <: Union{Nothing, <:PhilogenyEstimator, <:PhilogenyResult},
                     T22 <: Union{Nothing, <:TurnoverEstimator,
                                  <:AbstractVector{<:TurnoverEstimator}, <:Turnover,
                                  <:AbstractVector{<:Turnover},
                                  <:AbstractVector{<:Union{<:TurnoverEstimator, <:Turnover}}},
                     T23 <: Union{Nothing, <:AbstractTracking,
                                  <:AbstractVector{<:AbstractTracking}},
                     T24 <: Union{Nothing, <:FeesEstimator, <:Fees},
                     T25 <: JuMPReturnsEstimator, T26 <: Scalariser,
                     T27 <: Union{Nothing, <:CustomConstraint},
                     T28 <: Union{Nothing, <:CustomObjective}, T29 <: Real, T30 <: Real,
                     T31 <: Union{Nothing, <:Integer},
                     T32 <: Union{Nothing, <:Integer, <:AbstractVector{<:Integer}},
                     T33 <: Union{Nothing, <:Real}, T34 <: Union{Nothing, <:Real},
                     T35 <: Union{Nothing, <:Real}, T36 <: Union{Nothing, <:Real},
                     T37 <: Bool} <: BaseJuMPOptimisationEstimator
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
    slt::T14
    sst::T15
    sgmtx::T16
    # sglt::T17
    # sgst::T18
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
function assert_finite_nonnegative_real_or_vec(val::Real)
    @smart_assert(isfinite(val) && val > zero(val))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::AbstractVector{<:Real})
    @smart_assert(any(isfinite, val) &&
                  any(x -> x > zero(x), val) &&
                  all(x -> x >= zero(x), val))
    return nothing
end
function JuMPOptimiser(;
                       pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                       slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                       wb::Union{Nothing, <:WeightBounds, <:WeightBoundsEstimator} = WeightBounds(),
                       bgt::Union{Nothing, <:Real, <:BudgetConstraintEstimator} = 1.0,
                       sbgt::Union{Nothing, <:Real, <:BudgetRange} = nothing,
                       lt::Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                 <:AbstractVector{<:Pair{<:Any, <:Real}}} = nothing,
                       st::Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                 <:AbstractVector{<:Pair{<:Any, <:Real}}} = nothing,
                       lcs::Union{Nothing, <:AbstractString, Expr,
                                  <:AbstractVector{<:AbstractString},
                                  <:AbstractVector{Expr},
                                  <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                  #! Start: to delete
                                  <:LinearConstraintEstimator,
                                  <:AbstractVector{<:LinearConstraintEstimator},
                                  #! End: to delete
                                  <:LinearConstraint} = nothing,
                       lcm::Union{Nothing, <:LinearConstraint} = nothing,
                       cent::Union{Nothing, <:CentralityEstimator,
                                   <:AbstractVector{<:CentralityEstimator},
                                   <:LinearConstraint} = nothing,
                       gcard::Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString},
                                    <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                    #! Start: to delete
                                    <:LinearConstraintEstimator,
                                    <:AbstractVector{<:LinearConstraintEstimator},
                                    #! End: to delete
                                    <:LinearConstraint} = nothing,
                       sgcard::Union{Nothing, <:AbstractString, Expr,
                                     <:AbstractVector{<:AbstractString},
                                     <:AbstractVector{Expr},
                                     <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                     <:AbstractVector{<:AbstractVector{<:AbstractString}},
                                     <:AbstractVector{<:AbstractVector{Expr}},
                                     <:AbstractVector{<:AbstractVector{<:Union{<:AbstractString,
                                                                               Expr}}},
                                     #! Start: to delete
                                     <:LinearConstraintEstimator,
                                     <:AbstractVector{<:LinearConstraintEstimator},
                                     #! End: to delete
                                     <:LinearConstraint,
                                     <:AbstractVector{<:LinearConstraint}} = nothing,
                       smtx::Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                   <:AbstractVector{Symbol},
                                   <:AbstractVector{<:AbstractString},
                                   <:AbstractVector{<:AbstractMatrix}} = nothing,
                       slt::Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                  <:AbstractVector{<:Pair{<:Any, <:Real}},
                                  <:AbstractVector{<:AbstractDict},
                                  <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                  <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                           <:AbstractVector{<:Pair{<:Any,
                                                                                   <:Real}}}}} = nothing,
                       sst::Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                  <:AbstractVector{<:Pair{<:Any, <:Real}},
                                  <:AbstractVector{<:AbstractDict},
                                  <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                  <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                           <:AbstractVector{<:Pair{<:Any,
                                                                                   <:Real}}}}} = nothing,
                       sgmtx::Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                    <:AbstractVector{Symbol},
                                    <:AbstractVector{<:AbstractString},
                                    <:AbstractVector{<:AbstractMatrix}} = nothing,
                       #    sglt::Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                       #                <:AbstractVector{<:Pair{<:Any, <:Real}},
                       #                <:AbstractVector{<:AbstractDict},
                       #                <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                       #                <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                       #                                         <:AbstractVector{<:Pair{<:Any,
                       #                                                                 <:Real}}}}} = nothing,
                       #    sgst::Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                       #                <:AbstractVector{<:Pair{<:Any, <:Real}},
                       #                <:AbstractVector{<:AbstractDict},
                       #                <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                       #                <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                       #                                         <:AbstractVector{<:Pair{<:Any,
                       #                                                                 <:Real}}}}} = nothing,
                       sets::Union{Nothing, <:AssetSets,
                                   #! Start: to delete
                                   <:DataFrame
                                   #! End: to delete
                                   } = nothing,
                       nplg::Union{Nothing, <:PhilogenyEstimator, <:PhilogenyResult} = nothing,
                       cplg::Union{Nothing, <:PhilogenyEstimator, <:PhilogenyResult} = nothing,
                       tn::Union{Nothing, <:TurnoverEstimator,
                                 <:AbstractVector{<:TurnoverEstimator}, <:Turnover,
                                 <:AbstractVector{<:Turnover},
                                 <:AbstractVector{<:Union{<:TurnoverEstimator, <:Turnover}}} = nothing,
                       te::Union{Nothing, <:AbstractTracking,
                                 <:AbstractVector{<:AbstractTracking}} = nothing,
                       fees::Union{Nothing, <:FeesEstimator, <:Fees} = nothing,
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       sce::Scalariser = SumScalariser(),
                       ccnt::Union{Nothing, <:CustomConstraint} = nothing,
                       cobj::Union{Nothing, <:CustomObjective} = nothing, sc::Real = 1,
                       so::Real = 1, card::Union{Nothing, <:Integer} = nothing,
                       scard::Union{Nothing, <:Integer, <:AbstractVector{<:Integer}} = nothing,
                       nea::Union{Nothing, <:Real} = nothing,
                       l1::Union{Nothing, <:Real} = nothing,
                       l2::Union{Nothing, <:Real} = nothing,
                       ss::Union{Nothing, <:Real} = nothing, strict::Bool = false)
    if isa(bgt, Real)
        @smart_assert(isfinite(bgt))
    elseif isa(bgt, BudgetCostEstimator)
        @smart_assert(isnothing(sbgt))
    end
    if isa(sbgt, Real)
        @smart_assert(isfinite(sbgt) && sbgt >= 0)
    elseif isa(sbgt, BudgetRange)
        lb = sbgt.lb
        ub = sbgt.ub
        lb_flag = isnothing(lb)
        ub_flag = isnothing(ub)
        @smart_assert(lb_flag ⊼ ub_flag)
        if !lb_flag
            @smart_assert(lb >= zero(lb))
        end
        if !ub_flag
            @smart_assert(ub >= zero(ub))
        end
        if !lb_flag && !ub_flag
            @smart_assert(lb <= ub)
        end
    end
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    if isa(cent, AbstractVector)
        @smart_assert(!isempty(cent))
    end
    if !isnothing(card)
        @smart_assert(isfinite(card) && card > 0)
    end
    if isa(scard, Integer)
        @smart_assert(isfinite(scard) && scard > 0)
        @smart_assert(isa(smtx, Union{Symbol, <:AbstractString, <:AbstractMatrix}))
        @smart_assert(isa(slt,
                          Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                <:AbstractVector{<:Pair{<:Any, <:Real}}}))
        @smart_assert(isa(sst,
                          Union{Nothing, <:BuyInThreshold, <:AbstractDict,
                                <:AbstractVector{<:Pair{<:Any, <:Real}}}))
    elseif isa(scard, AbstractVector)
        @smart_assert(!isempty(scard))
        @smart_assert(all(isfinite, scard) && all(x -> x > 0, scard))
        @smart_assert(isa(smtx,
                          Union{<:AbstractVector{Symbol},
                                <:AbstractVector{<:AbstractString},
                                <:AbstractVector{<:AbstractMatrix}}))
        @smart_assert(length(scard) == length(smtx))
        if isa(slt,
               Union{<:AbstractVector{<:AbstractDict},
                     <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                     <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                              <:AbstractVector{<:Pair{<:Any, <:Real}}}}})
            @smart_assert(length(scard) == length(slt))
        end
        if isa(sst,
               Union{<:AbstractVector{<:AbstractDict},
                     <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                     <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                              <:AbstractVector{<:Pair{<:Any, <:Real}}}}})
            @smart_assert(length(scard) == length(sst))
        end
    elseif isnothing(scard) && (isa(slt,
                                    Union{<:BuyInThreshold, <:AbstractDict,
                                          <:AbstractVector{<:Pair{<:Any, <:Real}}}) || isa(sst,
                                                                                           Union{<:BuyInThreshold, <:AbstractDict,
                                                                                                 <:AbstractVector{<:Pair{<:Any, <:Real}}}))
        @smart_assert(isa(smtx, Union{Symbol, <:AbstractString, <:AbstractMatrix}))
    elseif isnothing(scard) && (isa(slt,
                                    Union{<:AbstractVector{<:AbstractDict},
                                          <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                          <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                                   <:AbstractVector{<:Pair{<:Any, <:Real}}}}}) ||
                                isa(sst,
                                    Union{<:AbstractVector{<:AbstractDict},
                                          <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
                                          <:AbstractVector{<:Union{Nothing, <:AbstractDict,
                                                                   <:AbstractVector{<:Pair{<:Any, <:Real}}}}}))
        @smart_assert(isa(smtx,
                          Union{<:AbstractVector{Symbol},
                                <:AbstractVector{<:AbstractString},
                                <:AbstractVector{<:AbstractMatrix}}))
        @smart_assert(length(slt) == length(sst) == length(smtx))
    end
    if isa(gcard, AbstractVector)
        @smart_assert(!isempty(gcard))
    end
    if isa(sgcard, AbstractVector)
        @smart_assert(!isempty(sgcard))
    end
    if isa(sgcard,
           Union{<:AbstractString, Expr, <:AbstractVector{<:AbstractString},
                 <:AbstractVector{Expr}, <:AbstractVector{<:Union{<:AbstractString, Expr}},
                 <:LinearConstraint})
        @smart_assert(isa(sgmtx, Union{Symbol, <:AbstractString, <:AbstractMatrix}))
        # @smart_assert(isa(sglt,
        #                   Union{Nothing, <:BuyInThreshold, <:AbstractDict,
        #                         <:AbstractVector{<:Pair{<:Any, <:Real}}}))
        # @smart_assert(isa(sgst,
        #                   Union{Nothing, <:BuyInThreshold, <:AbstractDict,
        #                         <:AbstractVector{<:Pair{<:Any, <:Real}}}))
        if isa(sgcard, LinearConstraint) && isa(smtx, AbstractMatrix)
            N = size(smtx, 1)
            N_ineq = !isnothing(sgcard.ineq) ? length(sgcard.B_ineq) : 0
            N_eq = !isnothing(sgcard.eq) ? length(sgcard.B_eq) : 0
            @smart_assert(N == N_ineq + N_eq)
        end
    elseif isa(sgcard,
               Union{<:AbstractVector{<:AbstractVector},
                     <:AbstractVector{<:LinearConstraint}})
        @smart_assert(isa(sgmtx,
                          Union{<:AbstractVector{Symbol},
                                <:AbstractVector{<:AbstractString},
                                <:AbstractVector{<:AbstractMatrix}}))
        @smart_assert(length(sgcard) == length(sgmtx))
        # if isa(sglt,
        #        Union{<:AbstractVector{<:AbstractDict},
        #              <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
        #              <:AbstractVector{<:Union{Nothing, <:AbstractDict,
        #                                       <:AbstractVector{<:Pair{<:Any, <:Real}}}}})
        #     @smart_assert(length(sgcard) == length(sglt))
        # end
        # if isa(sgst,
        #        Union{<:AbstractVector{<:AbstractDict},
        #              <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
        #              <:AbstractVector{<:Union{Nothing, <:AbstractDict,
        #                                       <:AbstractVector{<:Pair{<:Any, <:Real}}}}})
        #     @smart_assert(length(sgcard) == length(sgst))
        # end
        for (sgc, smt) in zip(sgcard, smtx)
            if isa(sgc, LinearConstraint) && isa(smt, AbstractMatrix)
                N = size(smt, 1)
                N_ineq = !isnothing(sgc.ineq) ? length(sgc.B_ineq) : 0
                N_eq = !isnothing(sgc.eq) ? length(sgc.B_eq) : 0
                @smart_assert(N == N_ineq + N_eq)
            end
        end
        # elseif isnothing(sgcard) && (isa(sglt,
        #                                  Union{<:BuyInThreshold, <:AbstractDict,
        #                                        <:AbstractVector{<:Pair{<:Any, <:Real}}}) || isa(sgst,
        #                                                                                         Union{<:BuyInThreshold, <:AbstractDict,
        #                                                                                               <:AbstractVector{<:Pair{<:Any, <:Real}}}))
        #     @smart_assert(isa(sgmtx, Union{Symbol, <:AbstractString, <:AbstractMatrix}))
        # elseif isnothing(sgcard) && (isa(sglt,
        #                                  Union{<:AbstractVector{<:AbstractDict},
        #                                        <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
        #                                        <:AbstractVector{<:Union{Nothing, <:AbstractDict,
        #                                                                 <:AbstractVector{<:Pair{<:Any, <:Real}}}}}) ||
        #                              isa(sgst,
        #                                  Union{<:AbstractVector{<:AbstractDict},
        #                                        <:AbstractVector{<:AbstractVector{<:Pair{<:Any, <:Real}}},
        #                                        <:AbstractVector{<:Union{Nothing, <:AbstractDict,
        #                                                                 <:AbstractVector{<:Pair{<:Any, <:Real}}}}}))
        #     @smart_assert(isa(sgmtx,
        #                       Union{<:AbstractVector{Symbol},
        #                             <:AbstractVector{<:AbstractString},
        #                             <:AbstractVector{<:AbstractMatrix}}))
        #     @smart_assert(length(sglt) == length(sgst) == length(sgmtx))
    end
    if isa(wb, WeightBoundsEstimator) ||
       !isa(lt, Union{Nothing, <:BuyInThreshold}) ||
       !isa(st, Union{Nothing, <:BuyInThreshold}) ||
       !isa(lcs, Union{Nothing, <:LinearConstraint}) ||
       !isa(cent, Union{Nothing, <:LinearConstraint}) ||
       !isa(gcard, Union{Nothing, <:LinearConstraint}) ||
       !isa(sgcard,
            Union{Nothing, <:LinearConstraint, <:AbstractVector{<:LinearConstraint}}) ||
       !isnothing(scard)
        @smart_assert(!isnothing(sets))
    end
    if isa(tn, AbstractVector)
        @smart_assert(!isempty(tn))
    end
    if isa(te, AbstractVector)
        @smart_assert(!isempty(te))
    end
    if !isnothing(nea)
        @smart_assert(nea > zero(nea))
    end
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return JuMPOptimiser{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                         typeof(lt), typeof(st), typeof(lcs), typeof(lcm), typeof(cent),
                         typeof(gcard), typeof(sgcard), typeof(smtx), typeof(slt),
                         typeof(sst), typeof(sgmtx), #typeof(sglt), typeof(sgst),
                         typeof(sets), typeof(nplg), typeof(cplg), typeof(tn), typeof(te),
                         typeof(fees), typeof(ret), typeof(sce), typeof(ccnt), typeof(cobj),
                         typeof(sc), typeof(so), typeof(card), typeof(scard), typeof(nea),
                         typeof(l1), typeof(l2), typeof(ss), typeof(strict)}(pe, slv, wb,
                                                                             bgt, sbgt, lt,
                                                                             st, lcs, lcm,
                                                                             cent, gcard,
                                                                             sgcard, smtx,
                                                                             slt, sst,
                                                                             sgmtx,
                                                                             #sglt,sgst, 
                                                                             sets, nplg,
                                                                             cplg, tn, te,
                                                                             fees, ret, sce,
                                                                             ccnt, cobj, sc,
                                                                             so, card,
                                                                             scard, nea, l1,
                                                                             l2, ss, strict)
end
function opt_view(opt::JuMPOptimiser, i::AbstractVector, X::AbstractMatrix)
    X = isa(opt.pe, AbstractPriorResult) ? opt.pe.X : X
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    bgt = budget_view(opt.bgt, i)
    lt = threshold_view(opt.lt, i)
    st = threshold_view(opt.st, i)
    # lcs = linear_constraint_view(opt.lcs, i)
    # gcard = linear_constraint_view(opt.gcard, i)
    # sgcard = linear_constraint_view(opt.sgcard, i)
    smtx = asset_sets_matrix_view(opt.smtx, i)
    slt = threshold_view(opt.slt, i)
    sst = threshold_view(opt.sst, i)
    sgmtx = if opt.smtx === opt.sgmtx
        smtx#, slt, sst
    else
        asset_sets_matrix_view(opt.sgmtx, i)
        # , threshold_view(opt.sglt, i),
        # threshold_view(opt.sgst, i)
    end
    sets = nothing_asset_sets_view(opt.sets, i)
    tn = turnover_view(opt.tn, i)
    te = tracking_view(opt.te, i, X)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = opt.lcs, lcm = opt.lcm, cent = opt.cent,
                         gcard = opt.gcard, sgcard = opt.sgcard, smtx = smtx, slt = slt,
                         sst = sst, sgmtx = sgmtx, #sglt = sglt, sgst = sgst, 
                         sets = sets, nplg = opt.nplg, cplg = opt.cplg, tn = tn, te = te,
                         fees = fees, ret = ret, sce = opt.sce, ccnt = ccnt, cobj = cobj,
                         sc = opt.sc, so = opt.so, card = opt.card, scard = opt.scard,
                         nea = opt.nea, l1 = opt.l1, l2 = opt.l2, ss = opt.ss,
                         strict = opt.strict)
end
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1)
    pr = prior(opt.pe, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    datatype = eltype(pr.X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(pr.X, 2), strict = opt.strict,
                                   datatype = datatype)
    lt = threshold_constraints(opt.lt, opt.sets; datatype = datatype, strict = opt.strict)
    st = threshold_constraints(opt.st, opt.sets; datatype = datatype, strict = opt.strict)
    lcs = linear_constraints(opt.lcs, opt.sets; datatype = datatype, strict = opt.strict)
    cent = centrality_constraints(opt.cent, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    gcard = linear_constraints(opt.gcard, opt.sets; datatype = Int, strict = opt.strict)
    sgcard = linear_constraints(opt.sgcard, opt.sets; datatype = Int, strict = opt.strict)
    smtx = asset_sets_matrix(opt.smtx, opt.sets)
    slt = threshold_constraints(opt.slt, opt.sets; datatype = datatype, strict = opt.strict)
    sst = threshold_constraints(opt.sst, opt.sets; datatype = datatype, strict = opt.strict)
    sgmtx = if opt.smtx === opt.sgmtx
        smtx#, slt, sst
    else
        asset_sets_matrix(opt.sgmtx, opt.sets)
        # threshold_constraints(opt.sglt, opt.sets; datatype = datatype, strict = opt.strict),
        # threshold_constraints(opt.sgst, opt.sets; datatype = datatype, strict = opt.strict)
    end
    nplg = philogeny_constraints(opt.nplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    cplg = philogeny_constraints(opt.cplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    tn = turnover_constraints(opt.tn, opt.sets; strict = opt.strict, datatype = datatype)
    fees = fees_constraints(opt.fees, opt.sets; strict = opt.strict, datatype = datatype)
    ret = jump_returns_factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx,
                                            slt, sst, sgmtx, #sglt, sgst, 
                                            nplg, cplg, tn, fees, ret)
end
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(propertynames(opt), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getproperty.(Ref(opt), pnames))...)
end
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, #=sglt, sgst,=# nplg, cplg, tn, fees, ret) = processed_jump_optimiser_attributes(opt,
                                                                                                                                         rd;
                                                                                                                                         dims = dims)
    return JuMPOptimiser(; pe = pr, slv = opt.slv, wb = wb, bgt = opt.bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = lcs, lcm = opt.lcm, cent = cent,
                         gcard = gcard, sgcard = sgcard, smtx = smtx, slt = slt, sst = sst,
                         sgmtx = sgmtx,# sglt = sglt, sgst = sgst, 
                         sets = opt.sets, nplg = nplg, cplg = cplg, tn = tn, te = opt.te,
                         fees = fees, ret = ret, sce = opt.sce, ccnt = opt.ccnt,
                         cobj = opt.cobj, sc = opt.sc, so = opt.so, card = opt.card,
                         nea = opt.nea, l1 = opt.l1, l2 = opt.l2, ss = opt.ss,
                         strict = opt.strict)
end

export JuMPOptimisation, JuMPOptimiser
