abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
abstract type ClusteringOptimisationEstimator <: OptimisationEstimator end
struct HierarchicalResult{T1, T2, T3, T4, T5, T6, T7, T8} <: OptimisationResult
    oe::T1
    pr::T2
    clr::T3
    wb::T4
    fees::T5
    retcode::T6
    w::T7
    fb::T8
end
function factory(res::HierarchicalResult, fb)
    return HierarchicalResult(res.oe, res.pr, res.clr, res.wb, res.fees, res.retcode, res.w,
                              fb)
end
struct HierarchicalOptimiser{T1, T2, T3, T4, T5, T6, T7, T8} <:
       BaseClusteringOptimisationEstimator
    pr::T1
    clr::T2
    slv::T3
    wb::T4
    fees::T5
    sets::T6
    wf::T7
    strict::T8
    function HierarchicalOptimiser(pr::PrE_Pr, clr::HClE_HCl, slv::Option{<:Slv_VecSlv},
                                   wb::Option{<:WbE_Wb}, fees::Option{<:FeesE_Fees},
                                   sets::Option{<:AssetSets}, wf::WeightFinaliser,
                                   strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pr), typeof(clr), typeof(slv), typeof(wb), typeof(fees),
                   typeof(sets), typeof(wf), typeof(strict)}(pr, clr, slv, wb, fees, sets,
                                                             wf, strict)
    end
end
function HierarchicalOptimiser(; pr::PrE_Pr = EmpiricalPrior(),
                               clr::HClE_HCl = ClustersEstimator(),
                               slv::Option{<:Slv_VecSlv} = nothing,
                               wb::Option{<:WbE_Wb} = WeightBounds(),
                               fees::Option{<:FeesE_Fees} = nothing,
                               sets::Option{<:AssetSets} = nothing,
                               wf::WeightFinaliser = IterativeWeightFinaliser(),
                               strict::Bool = false)
    return HierarchicalOptimiser(pr, clr, slv, wb, fees, sets, wf, strict)
end
function opt_view(hco::HierarchicalOptimiser, i)
    pr = prior_view(hco.pr, i)
    wb = weight_bounds_view(hco.wb, i)
    fees = fees_view(hco.fees, i)
    sets = nothing_asset_sets_view(hco.sets, i)
    return HierarchicalOptimiser(; pr = pr, clr = hco.clr, slv = hco.slv, wb = wb,
                                 fees = fees, wf = hco.wf, sets = sets, strict = hco.strict)
end
function unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum,
                                fees::Option{<:Fees} = nothing)
    wk = zeros(eltype(X), size(X, 2))
    rk = Vector{eltype(X)}(undef, size(X, 2))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return rk
end
function unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure,
                                 X::MatNum, fees::Option{<:Fees} = nothing)
    fill!(rk, zero(eltype(X)))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return nothing
end

export HierarchicalResult, HierarchicalOptimiser
