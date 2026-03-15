abstract type BaseStackingOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
struct StackingResult{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       NonFiniteAllocationOptimisationResult
    oe::T1
    pr::T2
    wb::T3
    fees::T4
    resi::T5
    reso::T6
    cv::T7
    retcode::T8
    w::T9
    fb::T10
end
function factory(res::StackingResult, fb::Option{<:OptE_Opt})
    return StackingResult(res.oe, res.pr, res.wb, res.fees, res.resi, res.reso, res.cv,
                          res.retcode, res.w, fb)
end
struct Stacking{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       BaseStackingOptimisationEstimator
    pe::T1
    wb::T2
    fees::T3
    sets::T4
    scale::T5
    opti::T6
    opto::T7
    cv::T8
    wf::T9
    ex::T10
    fb::T11
    brt::T12
    strict::T13
    function Stacking(pe::PrE_Pr, wb::Option{<:WbE_Wb}, fees::Option{<:FeesE_Fees},
                      sets::Option{<:AssetSets}, scale::Option{<:VecNum}, opti::VecOptE_Opt,
                      opto::NonFiniteAllocationOptimisationEstimator,
                      cv::Option{<:OptimisationCrossValidation}, wf::WeightFinaliser,
                      ex::FLoops.Transducers.Executor, fb::Option{<:OptE_Opt}, brt::Bool,
                      strict::Bool)
        @argcheck(!isempty(opti))
        if !isnothing(scale)
            @argcheck(length(scale) == length(opti))
            @argcheck(all(isfinite, scale))
        end
        assert_external_optimiser(opto)
        if !isnothing(cv)
            assert_external_optimiser(opti)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(wb), typeof(fees), typeof(sets), typeof(scale),
                   typeof(opti), typeof(opto), typeof(cv), typeof(wf), typeof(ex),
                   typeof(fb), typeof(brt), typeof(strict)}(pe, wb, fees, sets, scale, opti,
                                                            opto, cv, wf, ex, fb, brt,
                                                            strict)
    end
end
function Stacking(; pe::PrE_Pr = EmpiricalPrior(), wb::Option{<:WbE_Wb} = nothing,
                  fees::Option{<:FeesE_Fees} = nothing, sets::Option{<:AssetSets} = nothing,
                  scale::Option{<:VecNum} = nothing, opti::VecOptE_Opt,
                  opto::NonFiniteAllocationOptimisationEstimator,
                  cv::Option{<:OptimisationCrossValidation} = nothing,
                  wf::WeightFinaliser = IterativeWeightFinaliser(),
                  ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                  fb::Option{<:OptE_Opt} = nothing, brt::Bool = false, strict::Bool = false)
    return Stacking(pe, wb, fees, sets, scale, opti, opto, cv, wf, ex, fb, brt, strict)
end
function assert_special_nco_requirements(opt::Stacking)
    @argcheck(!any(x -> isa(x, NonFiniteAllocationOptimisationResult), opt.opti))
end
function assert_external_optimiser(opt::Stacking)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_external_optimiser(opt.opto)
    if !isnothing(opt.cv)
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
function needs_previous_weights(opt::Stacking)
    return (needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opti) ||
            needs_previous_weights(opt.opto) ||
            needs_previous_weights(opt.fb))
end
function factory(st::Stacking, w::AbstractVector)
    fees = factory(st.fees, w)
    opti = factory(st.opti, w)
    opto = factory(st.opto, w)
    fb = factory(st.fb, w)
    return Stacking(; pe = st.pe, wb = st.wb, fees = fees, sets = st.sets, scale = st.scale,
                    opti = opti, opto = opto, cv = st.cv, wf = st.wf, ex = st.ex, fb = fb,
                    brt = st.brt, strict = st.strict)
end
function opt_view(st::Stacking, i, X::MatNum)
    X = isa(st.pe, AbstractPriorResult) ? st.pe.X : X
    pe = prior_view(st.pe, i)
    wb = weight_bounds_view(st.wb, i)
    fees = fees_view(st.fees, i)
    opti = opt_view(st.opti, i, X)
    opto = opt_view(st.opto, i, X)
    sets = nothing_asset_sets_view(st.sets, i)
    return Stacking(; pe = pe, wb = wb, fees = fees, scale = st.scale, opti = opti,
                    opto = opto, cv = st.cv, wf = st.wf, sets = sets, ex = st.ex,
                    fb = st.fb, brt = st.brt, strict = st.strict)
end

"""
Overload this using st.cv for custom cross-validation prediction
"""
function predict_outer_st_estimator_returns(st::Option{<:Stacking}, rd::ReturnsResult,
                                            pr::AbstractPriorResult, fees::Option{<:Fees},
                                            wi::MatNum, resi::VecOpt)
    nb, B = if !isa(rd.B, MatNum)
        rd.nb, rd.B
    else
        ["_b$(i)" for i in 1:size(wi, 2)], rd.B * wi
    end
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        wi = abs.(wi)
        if iv_flag
            iv = iv * wi
        end
        if ivpa_flag
            ivpa = transpose(wi) * ivpa
        end
    end
    X = Matrix{eltype(pr.X)}(undef, size(pr.X, 1), size(wi, 2))
    for (i, res) in enumerate(resi)
        X[:, i] = calc_net_returns(res, pr, fees)
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         nb = nb, B = B, ts = rd.ts, iv = iv, ivpa = ivpa)
end
function predict_outer_st_estimator_returns(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, <:Any,
                                                         <:OptimisationCrossValidation{<:NonCombOptCV}},
                                            rd::ReturnsResult, pr::AbstractPriorResult,
                                            fees::Option{<:Fees}, wi::MatNum, resi::VecOpt)
    (; opti, cv, ex) = st
    cv = cv.cv
    predictions = Vector{MultiPeriodPredictionResult}(undef, length(opti))
    let cv = cv
        FLoops.@floop ex for (i, opt) in enumerate(opti)
            cvi = !hasproperty(cv, :rng) ? cv : copy(cv)
            predictions[i] = cross_val_predict(opt, rd, cvi; ex = ex)
        end
    end
    return rebuild_returns_result(rd, predictions)
end
function predict_outer_st_estimator_returns(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, <:Any,
                                                         <:OptimisationCrossValidation{<:CombinatorialCrossValidation}},
                                            rd::ReturnsResult, pr::AbstractPriorResult,
                                            fees::Option{<:Fees}, wi::MatNum, resi::VecOpt)
    (; opti, cv, ex) = st
    (; cv, scorer) = cv
    predictions = Vector{PopulationPredictionResult}(undef, length(opti))
    let cv = cv
        FLoops.@floop ex for (i, opt) in enumerate(opti)
            cvi = !hasproperty(cv, :rng) ? cv : copy(cv)
            predictions[i] = cross_val_predict(opt, rd, cvi; ex = ex)
        end
    end
    if isnothing(scorer)
        scorer = NearestQuantilePrediction()
    end
    best_predictions = [scorer(prediction) for prediction in predictions]
    return rebuild_returns_result(rd, best_predictions)
end
function _optimise(st::Stacking, rd::ReturnsResult; dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    rd = returns_result_picker(rd, st.brt)
    pr = prior(st.pe, rd; dims = dims)
    fees = fees_constraints(st.fees, st.sets; datatype = eltype(pr.X), strict = st.strict)
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
    swi = isnothing(st.scale) ? wi : wi .* transpose(st.scale)
    rdo = predict_outer_st_estimator_returns(st, rd, pr, fees, swi, resi)
    reso = optimise(st.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb, retcode, w = outer_optimisation_finaliser(st.wb, st.sets, st.wf, st.strict, resi,
                                                  reso.retcode, reso.w, wi;
                                                  datatype = eltype(pr.X))
    return StackingResult(typeof(st), pr, wb, fees, resi, reso, st.cv, retcode, w, nothing)
end
function optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                               <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(st, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export StackingResult, Stacking
