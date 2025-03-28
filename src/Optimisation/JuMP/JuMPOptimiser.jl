struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
                     T2 <: Union{Nothing, <:WeightBounds, WeightBoundsConstraints},
                     T3 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T4 <: Union{Nothing, <:LongShortSum}, T5 <: Union{Nothing, <:Real},
                     T6 <: Union{Nothing, BuyInThreshold},
                     T7 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintModel},
                     T8 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintModel},
                     T9 <: Union{Nothing, <:CentralityConstraint,
                                 <:AbstractVector{<:CentralityConstraint},
                                 <:LinearConstraintModel}, T10 <: Union{Nothing, DataFrame},
                     T11 <: Union{Nothing, Fees},
                     T12 <:
                     Union{Nothing, <:AdjacencyConstraint, <:AdjacencyConstraintModel},
                     T13 <:
                     Union{Nothing, <:AdjacencyConstraint, <:AdjacencyConstraintModel},
                     T14 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                     T15 <: Scalariser}
    pe::T1
    wb::T2
    bgt::T3
    lss::T4
    card::T5
    bit::T6
    gcard::T7
    lcs::T8
    cent::T9
    sets::T10
    nadj::T11
    cadj::T12
    fees::T13
    slv::T14
    sce::T15
end
function JuMPOptimiser(; bgt::Union{Nothing, <:Real, <:BudgetConstraint} = 1.0,
                       gcard::Union{Nothing, <:LinearConstraint,
                                    <:AbstractVector{<:LinearConstraint},
                                    <:LinearConstraintModel} = nothing,
                       lcs::Union{Nothing, <:LinearConstraint,
                                  <:AbstractVector{<:LinearConstraint},
                                  <:LinearConstraintModel} = nothing,
                       cent::Union{Nothing, <:CentralityConstraint,
                                   <:AbstractVector{<:CentralityConstraint},
                                   <:LinearConstraintModel} = nothing,
                       sets::Union{Nothing, DataFrame} = nothing)
    if isa(bgt, Real)
        @smart_assert(isfinite(bgt))
    end
    if isa(gcard, LinearConstraint) ||
       isa(gcard, AbstractVector{<:LinearConstraint}) ||
       isa(lcs, LinearConstraint) ||
       isa(lcs, AbstractVector{<:LinearConstraint}) ||
       isa(cent, LinearConstraint) ||
       isa(cent, AbstractVector{<:LinearConstraint})
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
end
