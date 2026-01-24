abstract type BaseStackingOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
struct StackingResult{T1, T2, T3, T4, T5, T6, T7, T8, T9} <:
       NonFiniteAllocationOptimisationResult
    oe::T1
    pr::T2
    wb::T3
    resi::T4
    reso::T5
    cv::T6
    retcode::T7
    w::T8
    fb::T9
end
function factory(res::StackingResult, fb)
    return StackingResult(res.oe, res.pr, res.wb, res.resi, res.reso, res.cv, res.retcode,
                          res.w, fb)
end
struct Stacking{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       BaseStackingOptimisationEstimator
    pr::T1
    wb::T2
    sets::T3
    opti::T4
    opto::T5
    cv::T6
    wf::T7
    strict::T8
    ex::T9
    fb::T10
    function Stacking(pr::PrE_Pr, wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                      opti::VecOptE_Opt, opto::NonFiniteAllocationOptimisationEstimator,
                      cv::Option{<:CrossValidationEstimator}, wf::WeightFinaliser,
                      strict::Bool, ex::FLoops.Transducers.Executor,
                      fb::Option{<:NonFiniteAllocationOptimisationEstimator})
        assert_external_optimiser(opto)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pr), typeof(wb), typeof(sets), typeof(opti), typeof(opto),
                   typeof(cv), typeof(wf), typeof(strict), typeof(ex), typeof(fb)}(pr, wb,
                                                                                   sets,
                                                                                   opti,
                                                                                   opto, cv,
                                                                                   wf,
                                                                                   strict,
                                                                                   ex, fb)
    end
end
function Stacking(; pr::PrE_Pr = EmpiricalPrior(), wb::Option{<:WbE_Wb} = nothing,
                  sets::Option{<:AssetSets} = nothing, opti::VecOptE_Opt,
                  opto::NonFiniteAllocationOptimisationEstimator,
                  cv::Option{<:CrossValidationEstimator} = nothing,
                  wf::WeightFinaliser = IterativeWeightFinaliser(), strict::Bool = false,
                  ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                  fb::Option{<:NonFiniteAllocationOptimisationEstimator} = nothing)
    return Stacking(pr, wb, sets, opti, opto, cv, wf, strict, ex, fb)
end
function assert_external_optimiser(opt::Stacking)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pr, AbstractPriorResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function assert_internal_optimiser(opt::Stacking)
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    return nothing
end
function opt_view(st::Stacking, i, X::MatNum)
    X = isa(st.pr, AbstractPriorResult) ? st.pr.X : X
    pr = prior_view(st.pr, i)
    wb = weight_bounds_view(st.wb, i)
    opti = opt_view(st.opti, i, X)
    opto = opt_view(st.opto, i, X)
    sets = nothing_asset_sets_view(st.sets, i)
    return Stacking(; pr = pr, wb = wb, opti = opti, opto = opto, cv = st.cv, wf = st.wf,
                    sets = sets, strict = st.strict, ex = st.ex, fb = st.fb)
end
"""
Overload this using st.cv for custom cross-validation prediction
"""
function predict_outer_st_estimator_returns(st::Stacking, rd::ReturnsResult,
                                            pr::AbstractPriorResult, wi::MatNum,
                                            resi::VecOpt)
    iv = isnothing(rd.iv) ? rd.iv : rd.iv * wi
    ivpa = (isnothing(rd.ivpa) || isa(rd.ivpa, Number)) ? rd.ivpa : transpose(wi) * rd.ivpa
    X = zeros(eltype(pr.X), size(pr.X, 1), size(wi, 2))
    for (i, res) in enumerate(resi)
        X[:, i] = predict(res, pr)
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         ts = rd.ts, iv = iv, ivpa = ivpa)
end
function _optimise(st::Stacking, rd::ReturnsResult; dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(st.pr, rd; dims = dims)
    opti = st.opti
    Ni = length(opti)
    wi = zeros(eltype(pr.X), size(pr.X, 2), Ni)
    resi = Vector{NonFiniteAllocationOptimisationResult}(undef, Ni)
    FLoops.@floop st.ex for (i, opt) in pairs(opti)
        res = optimise(opt, rd; dims = dims, branchorder = branchorder,
                       str_names = str_names, save = save, kwargs...)
        #! Support efficient frontier?
        @argcheck(!isa(res.retcode, AbstractVector))
        wi[:, i] = res.w
        resi[i] = res
    end
    rdo = predict_outer_st_estimator_returns(st, rd, pr, wi, resi)
    reso = optimise(st.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(st.wb, st.sets, st.wf, st.strict, resi,
                                                 reso, wi * reso.w; datatype = eltype(pr.X))
    return StackingResult(typeof(st), pr, wb, resi, reso, st.cv, retcode, w, nothing)
end
function optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                               <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(st, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export StackingResult, Stacking
