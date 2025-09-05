struct HierarchicalOptimisation{T1, T2, T3, T4, T5, T6, T7} <: OptimisationResult
    oe::T1
    pr::T2
    fees::T3
    wb::T4
    clr::T5
    retcode::T6
    w::T7
end
struct HierarchicalOptimiser{T1, T2, T3, T4, T5, T6, T7, T8} <:
       BaseClusteringOptimisationEstimator
    pe::T1
    cle::T2
    slv::T3
    fees::T4
    wb::T5
    sets::T6
    cwf::T7
    strict::T8
end
function HierarchicalOptimiser(;
                               pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                               cle::Union{<:ClusteringEstimator,
                                          <:AbstractClusteringResult} = ClusteringEstimator(),
                               slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                               fees::Union{Nothing, <:FeesEstimator, <:Fees} = nothing,
                               wb::Union{Nothing, <:WeightBoundsEstimator, <:WeightBounds} = WeightBounds(),
                               sets::Union{Nothing, <:AssetSets} = nothing,
                               cwf::WeightFinaliser = IterativeWeightFinaliser(),
                               strict::Bool = false)
    if isa(wb, WeightBoundsEstimator)
        @argcheck(!isnothing(sets))
    end
    return HierarchicalOptimiser(pe, cle, slv, fees, wb, sets, cwf, strict)
end
function opt_view(hco::HierarchicalOptimiser, i::AbstractVector)
    pe = prior_view(hco.pe, i)
    fees = fees_view(hco.fees, i)
    wb = weight_bounds_view(hco.wb, i)
    sets = nothing_asset_sets_view(hco.sets, i)
    return HierarchicalOptimiser(; pe = pe, cle = hco.cle, fees = fees, slv = hco.slv,
                                 wb = wb, cwf = hco.cwf, sets = sets, strict = hco.strict)
end
function unitary_expected_risks(r::OptimisationRiskMeasure, X::AbstractMatrix,
                                fees::Union{Nothing, <:Fees} = nothing)
    wk = zeros(eltype(X), size(X, 2))
    rk = Vector{eltype(X)}(undef, size(X, 2))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return rk
end
function unitary_expected_risks!(wk::AbstractVector, rk::AbstractVector,
                                 r::OptimisationRiskMeasure, X::AbstractMatrix,
                                 fees::Union{Nothing, <:Fees} = nothing)
    fill!(rk, zero(eltype(X)))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return nothing
end

export HierarchicalOptimisation, HierarchicalOptimiser
