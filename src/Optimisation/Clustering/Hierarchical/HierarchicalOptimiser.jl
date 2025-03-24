struct HierarchicalOptimiser{T1 <: AbstractPriorEstimator, T2 <: ClusteringEstimator,
                             T3 <: WeightLimits, T4 <: Fees, T5 <: Scalariser,
                             T6 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}}
    pe::T1
    cle::T2
    wl::T3
    fees::T4
    sce::T5
    slv::T6
end
function HierarchicalOptimiser(; pe::AbstractPriorEstimator = EmpiricalPriorEstimator(),
                               cle::ClusteringEstimator = ClusteringEstimator(),
                               wl::WeightLimits = WeightLimits(), fees::Fees = Fees(),
                               sce::Scalariser = SumScalariser(),
                               slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,)
    return HierarchicalOptimiser{typeof(pe), typeof(cle), typeof(wl), typeof(fees),
                                 typeof(sce), typeof(slv)}(pe, cle, wl, fees, sce, slv)
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
