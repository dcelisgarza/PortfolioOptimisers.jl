struct HierarchicalOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
                             T2 <: Union{<:ClusteringEstimator,
                                         <:AbstractPortfolioOptimisersClusteringResult},
                             T3 <: Union{Nothing, Fees},
                             T4 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                             T5 <: Scalariser,
                             T6 <: Union{Nothing, <:WeightBounds, WeightBoundsConstraints},
                             T7 <: ClusteringWeightFinaliser}
    pe::T1
    cle::T2
    fees::T3
    slv::T4
    sce::T5
    wb::T6
    cwf::T7
end
function HierarchicalOptimiser(;
                               pe::Union{<:AbstractPriorEstimator, <:AbstractPriorModel} = EmpiricalPriorEstimator(),
                               cle::Union{<:ClusteringEstimator,
                                          <:AbstractPortfolioOptimisersClusteringResult} = ClusteringEstimator(),
                               fees::Union{Nothing, <:Fees} = nothing,
                               slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                               sce::Scalariser = SumScalariser(),
                               wb::Union{Nothing, <:WeightBounds, WeightBoundsConstraints} = nothing,
                               cwf::ClusteringWeightFinaliser = HeuristicClusteringWeightFiniliser())
    return HierarchicalOptimiser{typeof(pe), typeof(cle), typeof(fees), typeof(slv),
                                 typeof(sce), typeof(wb), typeof(cwf)}(pe, cle, fees, slv,
                                                                       sce, wb, cwf)
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
