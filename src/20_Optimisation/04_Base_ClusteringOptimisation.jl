abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
abstract type ClusteringOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
struct HierarchicalResult{T1, T2, T3, T4, T5, T6, T7, T8} <:
       NonFiniteAllocationOptimisationResult
    oe::T1
    pr::T2
    clr::T3
    wb::T4
    fees::T5
    retcode::T6
    w::T7
    fb::T8
end
function factory(res::HierarchicalResult, fb::Option{<:OptE_Opt})
    return HierarchicalResult(res.oe, res.pr, res.clr, res.wb, res.fees, res.retcode, res.w,
                              fb)
end
struct HierarchicalOptimiser{T1, T2, T3, T4, T5, T6, T7, T8, T9} <:
       BaseClusteringOptimisationEstimator
    pe::T1
    cle::T2
    slv::T3
    wb::T4
    fees::T5
    sets::T6
    wf::T7
    brt::T8
    strict::T9
    function HierarchicalOptimiser(pe::PrE_Pr, cle::HClE_HCl, slv::Option{<:Slv_VecSlv},
                                   wb::Option{<:WbE_Wb}, fees::Option{<:FeesE_Fees},
                                   sets::Option{<:AssetSets}, wf::WeightFinaliser,
                                   brt::Bool, strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(cle), typeof(slv), typeof(wb), typeof(fees),
                   typeof(sets), typeof(wf), typeof(brt), typeof(strict)}(pe, cle, slv, wb,
                                                                          fees, sets, wf,
                                                                          brt, strict)
    end
end
function HierarchicalOptimiser(; pe::PrE_Pr = EmpiricalPrior(),
                               cle::HClE_HCl = ClustersEstimator(),
                               slv::Option{<:Slv_VecSlv} = nothing,
                               wb::Option{<:WbE_Wb} = WeightBounds(),
                               fees::Option{<:FeesE_Fees} = nothing,
                               sets::Option{<:AssetSets} = nothing,
                               wf::WeightFinaliser = IterativeWeightFinaliser(),
                               brt::Bool = false, strict::Bool = false)
    return HierarchicalOptimiser(pe, cle, slv, wb, fees, sets, wf, brt, strict)
end
function needs_previous_weights(opt::HierarchicalOptimiser)
    return needs_previous_weights(opt.fees)
end
function factory(opt::HierarchicalOptimiser, w::AbstractVector)
    return HierarchicalOptimiser(; pe = opt.pe, cle = opt.cle, slv = opt.slv, wb = opt.wb,
                                 fees = factory(opt.fees, w), sets = opt.sets, wf = opt.wf,
                                 brt = opt.brt, strict = opt.strict)
end
function opt_view(hco::HierarchicalOptimiser, i)
    pe = prior_view(hco.pe, i)
    wb = weight_bounds_view(hco.wb, i)
    fees = fees_view(hco.fees, i)
    sets = nothing_asset_sets_view(hco.sets, i)
    return HierarchicalOptimiser(; pe = pe, cle = hco.cle, slv = hco.slv, wb = wb,
                                 fees = fees, wf = hco.wf, sets = sets, brt = hco.brt,
                                 strict = hco.strict)
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
