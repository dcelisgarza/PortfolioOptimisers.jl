struct HierarchicalOptimisation{T1, T2, T3, T4, T5, T6, T7, T8} <: OptimisationResult
    oe::T1
    pr::T2
    fees::T3
    wb::T4
    clr::T5
    retcode::T6
    w::T7
    fb::T8
end
function opt_attempt_factory(res::HierarchicalOptimisation, fb)
    return HierarchicalOptimisation(res.oe, res.pr, res.fees, res.wb, res.clr, res.retcode,
                                    res.w, fb)
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
    function HierarchicalOptimiser(pe::PrE_Pr, cle::ClE_Cl, slv::Option{<:Slv_VecSlv},
                                   fees::Option{<:FeesE_Fees}, wb::Option{<:WbE_Wb},
                                   sets::Option{<:AssetSets}, cwf::WeightFinaliser,
                                   strict::Bool)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(cle), typeof(slv), typeof(fees), typeof(wb),
                   typeof(sets), typeof(cwf), typeof(strict)}(pe, cle, slv, fees, wb, sets,
                                                              cwf, strict)
    end
end
function HierarchicalOptimiser(; pe::PrE_Pr = EmpiricalPrior(),
                               cle::ClE_Cl = ClusteringEstimator(),
                               slv::Option{<:Slv_VecSlv} = nothing,
                               fees::Option{<:FeesE_Fees} = nothing,
                               wb::Option{<:WbE_Wb} = WeightBounds(),
                               sets::Option{<:AssetSets} = nothing,
                               cwf::WeightFinaliser = IterativeWeightFinaliser(),
                               strict::Bool = false)
    return HierarchicalOptimiser(pe, cle, slv, fees, wb, sets, cwf, strict)
end
function opt_view(hco::HierarchicalOptimiser, i)
    pe = prior_view(hco.pe, i)
    fees = fees_view(hco.fees, i)
    wb = weight_bounds_view(hco.wb, i)
    sets = nothing_asset_sets_view(hco.sets, i)
    return HierarchicalOptimiser(; pe = pe, cle = hco.cle, fees = fees, slv = hco.slv,
                                 wb = wb, cwf = hco.cwf, sets = sets, strict = hco.strict)
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

export HierarchicalOptimisation, HierarchicalOptimiser
