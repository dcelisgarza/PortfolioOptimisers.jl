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
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`Stacking`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref). The optimiser-valued fields `opti` and `opto` are required and have no static default, so they are marked [`NoDefault`](@ref): a schedule there must carry its own `default` to be usable outside a fold loop. `pe` and `wf` reset to their keyword defaults; fields whose static default is `nothing` (`wb`, `fees`, `sets`, `scale`, `fb`) are omitted.

# Related

  - [`Stacking`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function stacking_td_defaults()::NamedTuple
    return (; pe = EmpiricalPrior(), opti = NoDefault(), opto = NoDefault(),
            wf = IterativeWeightFinaliser())
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Narrow a vector of optimisers so element-level [`TimeDependent`](@ref) schedules type-check.

A literal like `[MeanRisk(), TimeDependent(…)]` infers eltype `AbstractEstimator`, which the [`VecOptE_Opt_TD`](@ref) bound rejects even though every element is admissible. Vectors already matching the bound pass through unchanged; otherwise every element is checked against [`OptE_Opt_TD`](@ref) and the vector is rebuilt with the tightest element-type union. A field-level schedule passes through untouched.

# Related

  - [`VecOptE_Opt_TD`](@ref)
  - [`Stacking`](@ref)
"""
function narrow_optimiser_vector(opti::AbstractVector)
    if isa(opti, VecOptE_Opt_TD)
        return opti
    end
    @argcheck(all(x -> isa(x, OptE_Opt_TD), opti),
              ArgumentError("every element of opti must be an optimisation estimator, a precomputed optimisation result, or a TimeDependent schedule standing in for one"))
    return convert(Vector{Union{unique(typeof.(opti))...}}, opti)
end
function narrow_optimiser_vector(opti::TimeDependent)
    return opti
end
"""
$(DocStringExtensions.TYPEDEF)

Stacking portfolio optimiser.

`Stacking` implements a stacking (model combination) approach to portfolio optimisation. It applies multiple inner optimisers (`opti`) to the data, then combines their outputs with a single outer optimiser (`opto`) to produce a final portfolio. Optionally, cross-validation can be used to weight the inner optimisers' contributions.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Stacking(;
        pe::TD{<:PrE_Pr} = EmpiricalPrior(),
        wb::TD_Option{<:WbE_Wb} = nothing,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:AssetSets} = nothing,
        scale::TD_Option{<:VecNum} = nothing,
        opti::Union{<:AbstractVector, <:TD_VecOptE_Opt},
        opto::OptE_TD,
        cv::Option{<:OptimisationCrossValidation} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        strict::Bool = false
    ) -> Stacking

Keywords correspond to the struct's fields.

## Time-dependent fields

`pe`, `wb`, `fees`, `sets`, `scale`, `wf`, `opto` and `fb` may hold a [`TimeDependent`](@ref) per-fold schedule — no inner fold loop of `Stacking` consumes them, so the fold loop that reaches the `Stacking` resolves them; the optimiser positions `opto` and `fb` are `bind = :outermost` only. `opti` admits schedules at two levels:

  - **Element** (`opti = [static, TimeDependent(…)]`): the element is an optimiser position of the inner cross-validation (entered per candidate), so `bind = :nearest` is legal there — with a mandatory explicit `default` and `cv !== nothing`, because the full-sample `wi` fit always resolves the element fold-lessly to its `default` (see [`assert_nearest_optimiser_schedule`](@ref)).
  - **Field** (`opti = TimeDependent([[…], […]])`, a per-fold vector of candidate vectors, see [`TD_VecOptE_Opt`](@ref)): `bind = :outermost` only. A `:nearest` field-level schedule is rejected — the inner cross-validation is handed the elements, never the field, and a per-fold candidate vector would change the number and identity of the returns-proxy columns `opto` sees.

## Validation

  - If `opti` is a vector: `!isempty(opti)`, every element is an [`OptE_Opt_TD`](@ref), and any `bind = :nearest` element schedule has an explicit `default` and `cv !== nothing`.
  - If `opti` is a [`TimeDependent`](@ref): `bind !== :nearest`.
  - If `scale` is provided and static: all elements are finite, and `length(scale) == length(opti)` when `opti` is a vector.
  - `opto` and `fb` schedules: `bind !== :nearest`.

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `opti`: Recursively updated via [`factory`](@ref).
  - `opto`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Related

  - [`BaseStackingOptimisationEstimator`](@ref)
  - [`NestedClustered`](@ref)
  - [`StackingResult`](@ref)
"""
@propagatable @concrete struct Stacking <: BaseStackingOptimisationEstimator
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
    @fprop fees
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
    @fprop opti
    """
    $(field_dict[:opto])
    """
    @fprop opto
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
    @fprop fb
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:strict_opt])
    """
    strict
    function Stacking(pe::TD{<:PrE_Pr}, wb::TD_Option{<:WbE_Wb},
                      fees::TD_Option{<:FeesE_Fees}, sets::TD_Option{<:AssetSets},
                      scale::TD_Option{<:VecNum},
                      opti::Union{<:VecOptE_Opt_TD, <:TD_VecOptE_Opt}, opto::OptE_TD,
                      cv::Option{<:OptimisationCrossValidation}, wf::TD{<:WeightFinaliser},
                      ex::FLoops.Transducers.Executor, fb::TDO_Option{<:OptE_Opt},
                      brt::Bool, strict::Bool)
        if isa(opti, TimeDependent)
            @argcheck(opti.bind !== :nearest,
                      ArgumentError("opti of Stacking cannot hold a `bind = :nearest` schedule at the field level: Stacking's inner cross-validation is entered per candidate (`cross_val_predict(opti[k], …)`), so the fold loop is handed the elements, never the field — and a per-fold candidate vector would change the number and identity of the returns-proxy columns opto sees. Schedule individual elements instead (`opti = [static, TimeDependent(…, :nearest; default = …)]`), or use `bind = :outermost` to vary the whole vector with the fold loop that reaches the Stacking."))
        else
            @argcheck(!isempty(opti), IsEmptyError("opti cannot be empty"))
            for (i, x) in pairs(opti)
                assert_nearest_optimiser_schedule(x, Symbol("opti[$i]"), cv, :Stacking)
            end
        end
        if !isnothing(scale) && !isa(scale, TimeDependent)
            if isa(opti, AbstractVector)
                @argcheck(length(scale) == length(opti),
                          DimensionMismatch("scale ($(length(scale))) must match opti ($(length(opti)))"))
            end
            @argcheck(all(isfinite, scale),
                      IsNonFiniteError("all elements of scale must be finite"))
        end
        assert_no_nearest_bind_optimiser_schedule(opto, :opto, :Stacking)
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :Stacking)
        assert_external_optimiser(opto)
        if !isnothing(cv)
            assert_external_optimiser(opti)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        assert_time_dependent_substitution(Stacking,
                                           (; pe, wb, fees, sets, scale, opti, opto, cv, wf,
                                            ex, fb, brt, strict), stacking_td_defaults())
        return new{typeof(pe), typeof(wb), typeof(fees), typeof(sets), typeof(scale),
                   typeof(opti), typeof(opto), typeof(cv), typeof(wf), typeof(ex),
                   typeof(fb), typeof(brt), typeof(strict)}(pe, wb, fees, sets, scale, opti,
                                                            opto, cv, wf, ex, fb, brt,
                                                            strict)
    end
end
function Stacking(; pe::TD{<:PrE_Pr} = EmpiricalPrior(), wb::TD_Option{<:WbE_Wb} = nothing,
                  fees::TD_Option{<:FeesE_Fees} = nothing,
                  sets::TD_Option{<:AssetSets} = nothing,
                  scale::TD_Option{<:VecNum} = nothing,
                  opti::Union{<:AbstractVector, <:TD_VecOptE_Opt}, opto::OptE_TD,
                  cv::Option{<:OptimisationCrossValidation} = nothing,
                  wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                  ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                  fb::TDO_Option{<:OptE_Opt} = nothing, brt::Bool = false,
                  strict::Bool = false)::Stacking
    return Stacking(pe, wb, fees, sets, scale, narrow_optimiser_vector(opti), opto, cv, wf,
                    ex, fb, brt, strict)
end
"""
    assert_special_nco_requirements_stacking_opti(opti::AbstractVector)

Ensures there are no Finite optimisation estimators or results in the vector.
"""
function assert_special_nco_requirements_stacking_opti(opti::AbstractVector)::Nothing
    @argcheck(!any(x -> isa(x, NonFiniteAllocationOptimisationResult), opti),
              ArgumentError("opti cannot contain NonFiniteAllocationOptimisationResult elements"))
    return nothing
end
function assert_special_nco_requirements(opt::Stacking)::Nothing
    opti = opt.opti
    if isa(opti, TimeDependent)
        for v in time_dependent_entries(opti)
            assert_special_nco_requirements_stacking_opti(v)
        end
    else
        assert_special_nco_requirements_stacking_opti(opti)
    end
    return nothing
end
function assert_external_optimiser(opt::Stacking)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult),
              ArgumentError("opt.pe cannot be a precomputed AbstractPriorResult; use an estimator instead"))
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
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opti) ||
            needs_previous_weights(opt.opto) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any inner optimiser, the outer optimiser, or the fallback carries time-dependent constraints.
"""
function is_time_dependent(opt::Stacking)
    return (!isempty(time_dependent_fields(opt)) ||
            is_time_dependent(opt.opti) ||
            is_time_dependent(opt.opto) ||
            is_time_dependent(opt.fb))
end
function time_dependent_field_defaults(::Stacking)::NamedTuple
    return stacking_td_defaults()
end
function inner_fold_fields(::Stacking)::Tuple
    return (:opti,)
end
function assert_time_dependent_fold_count(opt::Stacking, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    if isa(opt.opti, AbstractVector)
        assert_time_dependent_fold_count(opt.opti, n, false)
    end
    assert_time_dependent_fold_count(opt.opto, n, all_binds)
    assert_time_dependent_fold_count(opt.fb, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve time-dependent constraints for the fold described by `ctx` by recursing into the inner optimisers, outer optimiser, and fallback.
"""
function update_time_dependent_estimator(opt::Stacking, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    if !is_time_dependent(opt)
        return opt
    end
    opt = update_time_dependent_fields(opt, ctx, all_binds)
    return rebuild_estimator(opt,
                             (;
                              opti = update_time_dependent_estimator(opt.opti, ctx, false),
                              opto = update_time_dependent_estimator(opt.opto, ctx,
                                                                     all_binds),
                              fb = update_time_dependent_estimator(opt.fb, ctx, all_binds)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace this meta-optimiser's own time-dependent fields with their static defaults.

Deliberately does **not** recurse into the wrapped optimisers: a standalone meta solve consumes inner per-fold schedules through its inner cross-validation leg, and its fold-less full-window inner solves reset themselves at their own `_optimise` seam. Only the meta's own fields (applied to the combined weights, resolved by an outer fold loop when one exists) are inert here. A `bind = :nearest` schedule in a field the meta hands across its own inner fold loop (see [`inner_fold_fields`](@ref)) is likewise left in place — resetting it here would replace it with its `default` before the inner cross-validation ever saw it.
"""
function reset_time_dependent_estimator(opt::Stacking)
    return reset_time_dependent_fields(opt)
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
    st = reset_time_dependent_estimator(st)
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
        @argcheck(!isa(res.retcode, AbstractVector),
                  ArgumentError("res.retcode cannot be an AbstractVector; efficient frontier results are not supported here"))
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
