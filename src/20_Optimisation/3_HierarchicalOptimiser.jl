struct HierarchicalOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                             T2 <: Union{<:ClusteringEstimator, <:AbstractClusteringResult},
                             T3 <: Union{Nothing, <:Fees},
                             T4 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                             T5 <: Scalariser,
                             T6 <:
                             Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                             T7 <: ClusteringWeightFinaliser,
                             T8 <: Union{Nothing, <:DataFrame}, T9 <: Bool} <:
       BaseClusteringOptimisationEstimator
    pe::T1
    cle::T2
    fees::T3
    slv::T4
    sce::T5
    wb::T6
    cwf::T7
    sets::T8
    strict::T9
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
function HierarchicalOptimiser(;
                               pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                               cle::Union{<:ClusteringEstimator,
                                          <:AbstractClusteringResult} = ClusteringEstimator(),
                               fees::Union{Nothing, <:Fees} = nothing,
                               slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                               sce::Scalariser = SumScalariser(),
                               wb::Union{Nothing, <:WeightBoundsResult,
                                         <:WeightBoundsConstraint} = nothing,
                               cwf::ClusteringWeightFinaliser = HeuristicClusteringWeightFiniliser(),
                               sets::Union{Nothing, <:DataFrame} = nothing,
                               strict::Bool = false)
    if isa(wb, WeightBoundsConstraint)
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    return HierarchicalOptimiser{typeof(pe), typeof(cle), typeof(fees), typeof(slv),
                                 typeof(sce), typeof(wb), typeof(cwf), typeof(sets),
                                 typeof(strict)}(pe, cle, fees, slv, sce, wb, cwf, sets,
                                                 strict)
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
