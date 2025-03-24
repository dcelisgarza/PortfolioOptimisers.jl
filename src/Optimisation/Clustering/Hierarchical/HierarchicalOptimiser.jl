struct HierarchicalOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
                             T2 <: Union{<:ClusteringEstimator,
                                         <:AbstractPortfolioOptimisersClusteringResult},
                             T3 <: Fees, T4 <: Scalariser, T5 <: WeightLimits,
                             T6 <: ClusteringWeightFinaliser,
                             T7 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}}
    pe::T1
    cle::T2
    fees::T3
    sce::T4
    wl::T5
    cwf::T6
    slv::T7
end
function HierarchicalOptimiser(;
                               pe::Union{<:AbstractPriorEstimator, <:AbstractPriorModel} = EmpiricalPriorEstimator(),
                               cle::Union{<:ClusteringEstimator,
                                          <:AbstractPortfolioOptimisersClusteringResult} = ClusteringEstimator(),
                               fees::Fees = Fees(), sce::Scalariser = SumScalariser(),
                               wl::WeightLimits = WeightLimits(),
                               cwf::ClusteringWeightFinaliser = HeuristicClusteringWeightFiniliser(),
                               slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing)
    return HierarchicalOptimiser{typeof(pe), typeof(cle), typeof(fees), typeof(sce),
                                 typeof(wl), typeof(cwf), typeof(slv)}(pe, cle, fees, sce,
                                                                       wl, cwf, slv)
end
function unitary_expected_risks(r::Union{<:OptimisationRiskMeasure,
                                         <:AbstractVector{<:OptimisationRiskMeasure}},
                                X::AbstractMatrix, fees::Fees = Fees(),
                                sce::Scalariser = SumScalariser())
    w = zeros(eltype(X), size(X, 2))
    rk = zeros(eltype(X), size(X, 2))
    for i ∈ eachindex(w)
        fill!(w, zero(eltype(X)))
        w[i] = one(eltype(X))
        rk[i] = expected_risk(r, w, X, fees, sce)
    end
    return rk
end
export HierarchicalOptimiser
