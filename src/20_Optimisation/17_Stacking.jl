"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for stacking-based portfolio optimisation estimators.

# Related Types

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`Stacking`](@ref)
"""
abstract type BaseStackingOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Result type for Stacking portfolio optimisation.

# Fields

  - `oe`: Type of the optimisation estimator that produced this result.
  - `pr`: Prior result used in optimisation.
  - `wb`: Weight bounds applied.
  - `fees`: Fee structure applied (or `nothing`).
  - `resi`: Inner optimisation results.
  - `reso`: Outer optimisation result.
  - `cv`: Cross-validation result (or `nothing`).
  - `retcode`: Overall return code.
  - `w`: Final aggregated portfolio weights.
  - `fb`: Fallback result.

# Related

  - [`Stacking`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct StackingResult <: NonFiniteAllocationOptimisationResult
    oe
    pr
    wb
    fees
    resi
    reso
    cv
    retcode
    w
    fb
end
function factory(res::StackingResult, fb::Option{<:OptE_Opt})
    return StackingResult(res.oe, res.pr, res.wb, res.fees, res.resi, res.reso, res.cv,
                          res.retcode, res.w, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Stacking portfolio optimiser.

`Stacking` implements a stacking (model combination) approach to portfolio optimisation. It applies multiple inner optimisers (`opti`) to the data, then combines their outputs with a single outer optimiser (`opto`) to produce a final portfolio. Optionally, cross-validation can be used to weight the inner optimisers' contributions.

# Fields

  - `pe`: Prior estimator or prior result.
  - `wb`: Weight bounds estimator or bounds.
  - `fees`: Fee estimator or fee structure.
  - `sets`: Asset sets.
  - `scale`: Optional scaling vector for inner optimiser weights (length must match `opti`).
  - `opti`: Vector of inner portfolio optimisers.
  - `opto`: Outer portfolio optimiser combining inner results.
  - `cv`: Cross-validation configuration for weighting inner optimisers.
  - `wf`: Weight finaliser for enforcing bounds.
  - `ex`: FLoops executor for parallelism.
  - `fb`: Fallback optimiser.
  - `brt`: If `true`, uses bootstrap returns.
  - `strict`: If `true`, strictly enforces weight bounds.

# Constructors

    Stacking(;
        pe::PrE_Pr = EmpiricalPrior(),
        wb::Option{<:WbE_Wb} = nothing,
        fees::Option{<:FeesE_Fees} = nothing,
        sets::Option{<:AssetSets} = nothing,
        scale::Option{<:VecNum} = nothing,
        opti::VecOptE_Opt,
        opto::NonFiniteAllocationOptimisationEstimator,
        cv::Option{<:OptimisationCrossValidation} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        strict::Bool = false
    ) -> Stacking

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(opti)`.
  - If `scale` is provided: `length(scale) == length(opti)` and all elements are finite.

# Related

  - [`BaseStackingOptimisationEstimator`](@ref)
  - [`NestedClustered`](@ref)
  - [`StackingResult`](@ref)
"""
@concrete struct Stacking <: BaseStackingOptimisationEstimator
    pe
    wb
    fees
    sets
    scale
    opti
    opto
    cv
    wf
    ex
    fb
    brt
    strict
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
                  fb::Option{<:OptE_Opt} = nothing, brt::Bool = false,
                  strict::Bool = false)::Stacking
    return Stacking(pe, wb, fees, sets, scale, opti, opto, cv, wf, ex, fb, brt, strict)
end
function assert_special_nco_requirements(opt::Stacking)::Nothing
    @argcheck(!any(x -> isa(x, NonFiniteAllocationOptimisationResult), opt.opti))
    return nothing
end
function assert_external_optimiser(opt::Stacking)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_external_optimiser(opt.opto)
    if !isnothing(opt.cv)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function assert_internal_optimiser(opt::Stacking)::Nothing
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
function factory(st::Stacking, w::AbstractVector)::Stacking
    fees = factory(st.fees, w)
    opti = factory(st.opti, w)
    opto = factory(st.opto, w)
    fb = factory(st.fb, w)
    return Stacking(; pe = st.pe, wb = st.wb, fees = fees, sets = st.sets, scale = st.scale,
                    opti = opti, opto = opto, cv = st.cv, wf = st.wf, ex = st.ex, fb = fb,
                    brt = st.brt, strict = st.strict)
end
function opt_view(st::Stacking, i, X::MatNum)::Stacking
    X = isa(st.pe, AbstractPriorResult) ? st.pe.X : X
    pe = prior_view(st.pe, i)
    wb = weight_bounds_view(st.wb, i)
    fees = fees_view(st.fees, i)
    sets = asset_sets_view(st.sets, i)
    opti = opt_view(st.opti, i, X)
    opto = opt_view(st.opto, i, X)
    return Stacking(; pe = pe, wb = wb, fees = fees, sets = sets, scale = st.scale,
                    opti = opti, opto = opto, cv = st.cv, wf = st.wf, ex = st.ex,
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
    X = pr.X
    X = Matrix{eltype(X)}(undef, size(X, 1), size(wi, 2))
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
    X = pr.X
    fees = fees_constraints(st.fees, st.sets; datatype = eltype(X), strict = st.strict)
    opti = st.opti
    Ni = length(opti)
    wi = zeros(eltype(X), size(X, 2), Ni)
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
    wb = weight_bounds_constraints(st.wb, st.sets; N = Ni, strict = st.strict,
                                   datatype = eltype(X))
    retcode, w = outer_optimisation_finaliser(wb, st.wf, resi, reso.retcode, reso.w, wi)
    return StackingResult(typeof(st), pr, wb, fees, resi, reso, st.cv, retcode, w, nothing)
end
"""
    optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                     <:Any, <:Any, Nothing
                 }, rd::ReturnsResult;
             dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
             save::Bool = true, kwargs...) -> StackingResult

# Arguments

  - `nco`: The nested clustered optimiser to use.
  - $(arg_dict[:rd])
  - `dims`: The dimension along which observations advance in time.
  - `branchorder`: Passed to the inner and outer optimisers. The branch order to use for the clusterisation.
  - `str_names`: Passed to the inner and outer optimisers. Whether to use string names for the assets in the optimisation.
  - `save`: Passed to the inner and outer optimisers. Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                               <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(st, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export StackingResult, Stacking
