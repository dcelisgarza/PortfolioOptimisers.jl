struct HierarchicalOptimisationResult{T1 <: Type, T2 <: AbstractPriorResult,
                                      T3 <: Union{Nothing, <:WeightBoundsResult},
                                      T4 <: AbstractClusteringResult,
                                      T5 <: OptimisationReturnCode, T6 <: AbstractVector} <:
       OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    clr::T4
    retcode::T5
    w::T6
end
struct HierarchicalOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                             T2 <: Union{<:ClusteringEstimator, <:AbstractClusteringResult},
                             T3 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                             T4 <: Union{Nothing, <:Fees},
                             T5 <:
                             Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                             T6 <: Union{Nothing, <:DataFrame},
                             T7 <: ClusteringWeightFinaliser, T8 <: Scalariser,
                             T9 <: Bool} <: BaseClusteringOptimisationEstimator
    pe::T1
    cle::T2
    slv::T3
    fees::T4
    wb::T5
    sets::T6
    cwf::T7
    sce::T8
    strict::T9
end
function HierarchicalOptimiser(;
                               pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                               cle::Union{<:ClusteringEstimator,
                                          <:AbstractClusteringResult} = ClusteringEstimator(),
                               slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                               fees::Union{Nothing, <:Fees} = nothing,
                               wb::Union{Nothing, <:WeightBoundsResult,
                                         <:WeightBoundsConstraint} = WeightBoundsResult(),
                               sets::Union{Nothing, <:DataFrame} = nothing,
                               cwf::ClusteringWeightFinaliser = HeuristicClusteringWeightFiniliser(),
                               sce::Scalariser = SumScalariser(), strict::Bool = false)
    if isa(wb, WeightBoundsConstraint)
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    return HierarchicalOptimiser{typeof(pe), typeof(cle), typeof(slv), typeof(fees),
                                 typeof(wb), typeof(sets), typeof(cwf), typeof(sce),
                                 typeof(strict)}(pe, cle, slv, fees, wb, sets, cwf, sce,
                                                 strict)
end
function opt_view(hco::HierarchicalOptimiser, i::AbstractVector)
    pe = prior_view(hco.pe, i)
    fees = fees_view(hco.fees, i)
    wb = weight_bounds_view(hco.wb, i)
    sets = nothing_dataframe_view(hco.sets, i)
    return HierarchicalOptimiser(; pe = pe, cle = hco.cle, fees = fees, slv = hco.slv,
                                 sce = hco.sce, wb = wb, cwf = hco.cwf, sets = sets,
                                 strict = hco.strict)
end
function unitary_expected_risks(r::Union{<:OptimisationRiskMeasure,
                                         <:AbstractVector{<:OptimisationRiskMeasure}},
                                X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing)
    wk = zeros(eltype(X), size(X, 2))
    rk = Vector{eltype(X)}(undef, size(X, 2))
    for i ∈ eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return rk
end
function unitary_expected_risks!(wk::AbstractVector, rk::AbstractVector,
                                 r::Union{<:OptimisationRiskMeasure,
                                          <:AbstractVector{<:OptimisationRiskMeasure}},
                                 X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing)
    fill!(rk, zero(eltype(X)))
    for i ∈ eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return nothing
end

export HierarchicalOptimiser
