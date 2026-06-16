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

$(DocStringExtensions.FIELDS)

# Related

  - [`Stacking`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct StackingResult <: NonJuMPOptimisationResult
    """
    $(field_dict[:oe])
    """
    oe
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:fees])
    """
    fees
    """
    $(field_dict[:resi])
    """
    resi
    """
    $(field_dict[:reso])
    """
    reso
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    Final aggregated portfolio weights.
    """
    w
    """
    $(field_dict[:fb])
    """
    fb
    function StackingResult(oe::Type{<:OptimisationEstimator},
                            pr::Option{<:AbstractPriorResult}, wb::Option{<:WeightBounds},
                            fees::Option{<:Fees},
                            resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                            reso::OptimisationResult,
                            cv::Option{<:OptimisationCrossValidation},
                            retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                            fb::Option{<:OptE_Opt})
        return new{typeof(oe), typeof(pr), typeof(wb), typeof(fees), typeof(resi),
                   typeof(reso), typeof(cv), typeof(retcode), typeof(w), typeof(fb)}(oe, pr,
                                                                                     wb,
                                                                                     fees,
                                                                                     resi,
                                                                                     reso,
                                                                                     cv,
                                                                                     retcode,
                                                                                     w, fb)
    end
end
function StackingResult(; oe::Type{<:OptimisationEstimator},
                        pr::Option{<:AbstractPriorResult}, wb::Option{<:WeightBounds},
                        fees::Option{<:Fees},
                        resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                        reso::OptimisationResult, cv::Option{<:OptimisationCrossValidation},
                        retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                        fb::Option{<:OptE_Opt})::StackingResult
    return StackingResult(oe, pr, wb, fees, resi, reso, cv, retcode, w, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Stacking portfolio optimiser.

`Stacking` implements a stacking (model combination) approach to portfolio optimisation. It applies multiple inner optimisers (`opti`) to the data, then combines their outputs with a single outer optimiser (`opto`) to produce a final portfolio. Optionally, cross-validation can be used to weight the inner optimisers' contributions.

# Fields

$(DocStringExtensions.FIELDS)

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

# Mathematical definition

Let ``K`` inner optimisers produce weight vectors ``\\boldsymbol{w}_1, \\ldots, \\boldsymbol{w}_K``. Stack them as rows of a returns proxy matrix and pass to outer optimiser ``\\mathrm{opto}``:

```math
\\begin{align}
\\boldsymbol{w}^* &= \\mathrm{opto}\\!\\left(\\sum_{k=1}^{K} s_k \\boldsymbol{W}_k\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}^*``: Final stacked portfolio weights.
  - ``K``: Number of inner optimisers.
  - ``s_k``: Optional scale factor for inner optimiser ``k``.
  - ``\\boldsymbol{W}_k``: Returns proxy matrix weighted by inner-optimiser weights ``\\boldsymbol{w}_k``.
  - ``\\mathrm{opto}``: Outer optimiser applied to the aggregated returns proxy.

# Related

  - [`BaseStackingOptimisationEstimator`](@ref)
  - [`NestedClustered`](@ref)
  - [`StackingResult`](@ref)
"""
@concrete struct Stacking <: BaseStackingOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:wb_jmp])
    """
    wb
    """
    $(field_dict[:feese])
    """
    fees
    """
    $(field_dict[:sets])
    """
    sets
    """
    Optional scaling vector for inner optimiser weights (length must match `opti`).
    """
    scale
    """
    $(field_dict[:opti])
    """
    opti
    """
    $(field_dict[:opto])
    """
    opto
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:fb])
    """
    fb
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:strict_opt])
    """
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (fees, inner optimiser, outer optimiser, or fallback).
"""
function needs_previous_weights(opt::Stacking)
    return (needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opti) ||
            needs_previous_weights(opt.opto) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build an updated [`Stacking`](@ref) with all estimators that track previous weights updated via `factory` using `w`.
"""
function factory(st::Stacking, w::AbstractVector)::Stacking
    fees = factory(st.fees, w)
    opti = factory(st.opti, w)
    opto = factory(st.opto, w)
    fb = factory(st.fb, w)
    return Stacking(; pe = st.pe, wb = st.wb, fees = fees, sets = st.sets, scale = st.scale,
                    opti = opti, opto = opto, cv = st.cv, wf = st.wf, ex = st.ex, fb = fb,
                    brt = st.brt, strict = st.strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`Stacking`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(st::Stacking, i, X::MatNum, args...)::Stacking
    X = isa(st.pe, AbstractPriorResult) ? st.pe.X : X
    pe = port_opt_view(st.pe, i)
    wb = port_opt_view(st.wb, i)
    fees = port_opt_view(st.fees, i)
    sets = port_opt_view(st.sets, i)
    opti = port_opt_view(st.opti, i, X)
    opto = port_opt_view(st.opto, i, X)
    return Stacking(; pe = pe, wb = wb, fees = fees, sets = sets, scale = st.scale,
                    opti = opti, opto = opto, cv = st.cv, wf = st.wf, ex = st.ex,
                    fb = st.fb, brt = st.brt, strict = st.strict)
end
"""
    predict_outer_st_estimator_returns(
        st::Option{<:Stacking},
        rd::ReturnsResult,
        pr::AbstractPriorResult,
        fees::Option{<:Fees},
        wi::MatNum,
        resi::VecOpt
    )

Predict outer portfolio returns for [`Stacking`](@ref) optimisation. Overload this using `st.cv` for custom cross-validation prediction.
"""
function predict_outer_st_estimator_returns(st::Option{<:Stacking}, rd::ReturnsResult,
                                            pr::AbstractPriorResult, fees::Option{<:Fees},
                                            wi::MatNum, resi::VecOpt)
    nb, B, iv, ivpa, X = prepare_outer_rd(rd, wi)
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
            cvi = !hasfield(typeof(cv), :rng) ? cv : copy(cv)
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
            cvi = !hasfield(typeof(cv), :rng) ? cv : copy(cv)
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
    return StackingResult(; oe = typeof(st), pr = pr, wb = wb, fees = fees, resi = resi,
                          reso = reso, cv = st.cv, retcode = retcode, w = w, fb = nothing)
end
"""
    optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                     <:Any, <:Any, Nothing
                 }, rd::ReturnsResult;
             dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
             save::Bool = true, kwargs...) -> StackingResult

Run the Stacking portfolio optimisation.

# Arguments

  - `st`: The stacking optimiser to use.
  - $(arg_dict[:rd])
  - `dims`: The dimension along which observations advance in time.
  - `branchorder`: Passed to the inner and outer optimisers. The branch order to use for the clusterisation.
  - `str_names`: Passed to the inner and outer optimisers. Whether to use string names for the assets in the optimisation.
  - `save`: Passed to the inner and outer optimisers. Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`Stacking`](@ref)
  - [`StackingResult`](@ref)
"""
function optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                               <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(st, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export StackingResult, Stacking
