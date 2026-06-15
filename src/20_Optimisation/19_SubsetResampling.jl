"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for subset resampling portfolio optimisation estimators.

# Related Types

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`SubsetResampling`](@ref)
"""
abstract type BaseSubsetResamplingOptimisationEstimator <:
              NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Result type for Subset Resampling portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`SubsetResampling`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct SubsetResamplingResult <: NonJuMPOptimisationResult
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
    Vector of sub-optimisation results for each subset.
    """
    ress
    """
    $(field_dict[:idx])
    """
    idx
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    Aggregated optimal portfolio weights.
    """
    w
    """
    $(field_dict[:fb])
    """
    fb
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a [`SubsetResamplingResult`](@ref) with an updated fallback optimiser `fb`.
"""
function factory(sr::SubsetResamplingResult, fb::Option{<:OptE_Opt})
    return SubsetResamplingResult(sr.oe, sr.pr, sr.wb, sr.fees, sr.ress, sr.idx, sr.retcode,
                                  sr.w, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Subset Resampling portfolio optimiser.

`SubsetResampling` applies a resampling strategy by optimising a base optimiser (`opt`) over randomly drawn subsets of assets, then aggregating the results into a final portfolio weight vector. This improves robustness of portfolio weights to estimation error.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SubsetResampling(;
        pe::PrE_Pr = EmpiricalPrior(),
        wb::Option{<:WbE_Wb} = nothing,
        fees::Option{<:FeesE_Fees} = nothing,
        sets::Option{<:AssetSets} = nothing,
        scale::Option{<:VecNum} = nothing,
        opt::NonFiniteAllocationOptimisationEstimator,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        subset_size::SubsetSizeE = 0.5,
        n_subsets::NumberSubsetsE = 100,
        max_comb::Integer = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        fb::Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        strict::Bool = false
    ) -> SubsetResampling

Keywords correspond to the struct's fields.

## Validation

  - If `scale` is provided: all elements must be non-empty, `> 0`, and finite.
  - `opt` must pass `assert_internal_optimiser`.
  - If `wb` is a `WeightBoundsEstimator`: `!isnothing(sets)`.
  - If `fees` is a `FeesEstimator`: `!isnothing(sets)`.
  - If `subset_size` is an `Integer`: `subset_size >= 1`.
  - If `subset_size` is a `Float`: `0 < subset_size < 1`.
  - If `n_subsets` is an `Integer`: `n_subsets >= 2`.
  - `max_comb > 0` and finite.

# Mathematical definition

Draw ``M`` random subsets ``S_1, \\ldots, S_M`` of assets (size ``\\lfloor s \\cdot N \\rfloor``). Optimise on each:

```math
\\begin{align}
\\boldsymbol{w}^* &= \\frac{1}{M} \\sum_{m=1}^{M} \\boldsymbol{e}_{S_m}(\\boldsymbol{w}_{S_m})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}^*``: Final averaged portfolio weights.
  - ``M``: Number of random subsets.
  - ``S_m``: ``m``-th randomly drawn asset subset (size ``\\lfloor s \\cdot N \\rfloor``).
  - ``\\boldsymbol{w}_{S_m}``: Optimal weights from the optimiser applied to subset ``S_m``.
  - ``\\boldsymbol{e}_{S_m}(\\cdot)``: Embedding operator that places subset weights into full ``N``-asset space (zero-filling excluded assets).
  - ``N``: Total number of assets.

# Related

  - [`BaseSubsetResamplingOptimisationEstimator`](@ref)
  - [`SubsetResamplingResult`](@ref)
  - [`MeanRisk`](@ref)
"""
@concrete struct SubsetResampling <: BaseSubsetResamplingOptimisationEstimator
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
    Optional scaling vector for subset optimiser weights.
    """
    scale
    """
    Base portfolio optimiser applied to each asset subset.
    """
    opt
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:subset_size])
    """
    subset_size
    """
    $(field_dict[:n_subsets])
    """
    n_subsets
    """
    $(field_dict[:max_comb])
    """
    max_comb
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
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
    function SubsetResampling(pe::PrE_Pr, wb::Option{<:WbE_Wb}, fees::Option{<:FeesE_Fees},
                              sets::Option{<:AssetSets}, scale::Option{<:VecNum},
                              opt::NonFiniteAllocationOptimisationEstimator,
                              wf::WeightFinaliser, ex::FLoops.Transducers.Executor,
                              subset_size::SubsetSizeE, n_subsets::NumberSubsetsE,
                              max_comb::Integer, rng::Random.AbstractRNG,
                              seed::Option{<:Integer}, fb::Option{<:OptE_Opt}, brt::Bool,
                              strict::Bool)
        assert_nonempty_gt0_finite_val(scale, :scale)
        assert_internal_optimiser(opt)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets))
        end
        if isa(subset_size, Integer)
            assert_nonempty_nonneg_finite_val(subset_size - 1, "subset_size - 1")
        elseif isa(subset_size, AbstractFloat)
            @argcheck(0 < subset_size < 1)
        end
        if isa(n_subsets, Integer)
            assert_nonempty_nonneg_finite_val(n_subsets - 2, "n_subsets - 2")
        end
        assert_nonempty_gt0_finite_val(max_comb, :max_comb)
        return new{typeof(pe), typeof(wb), typeof(fees), typeof(sets), typeof(scale),
                   typeof(opt), typeof(wf), typeof(ex), typeof(subset_size),
                   typeof(n_subsets), typeof(max_comb), typeof(rng), typeof(seed),
                   typeof(fb), typeof(brt), typeof(strict)}(pe, wb, fees, sets, scale, opt,
                                                            wf, ex, subset_size, n_subsets,
                                                            max_comb, rng, seed, fb, brt,
                                                            strict)
    end
end
function SubsetResampling(; pe::PrE_Pr = EmpiricalPrior(), wb::Option{<:WbE_Wb} = nothing,
                          fees::Option{<:FeesE_Fees} = nothing,
                          sets::Option{<:AssetSets} = nothing,
                          scale::Option{<:VecNum} = nothing,
                          opt::NonFiniteAllocationOptimisationEstimator,
                          wf::WeightFinaliser = IterativeWeightFinaliser(),
                          ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                          subset_size::SubsetSizeE = 0.8, n_subsets::NumberSubsetsE = 2,
                          max_comb::Integer = 1_000_000_000,
                          rng::Random.AbstractRNG = Random.default_rng(),
                          seed::Option{<:Integer} = nothing,
                          fb::Option{<:OptE_Opt} = nothing, brt::Bool = false,
                          strict::Bool = false)::SubsetResampling
    return SubsetResampling(pe, wb, fees, sets, scale, opt, wf, ex, subset_size, n_subsets,
                            max_comb, rng, seed, fb, brt, strict)
end
function assert_external_optimiser(opt::SubsetResampling)::Nothing
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    return assert_external_optimiser(opt.opt)
end
function assert_internal_optimiser(opt::SubsetResampling)::Nothing
    return assert_internal_optimiser(opt.opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (fees, base optimiser, or fallback).
"""
function needs_previous_weights(opt::SubsetResampling)
    return (needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build an updated [`SubsetResampling`](@ref) with all estimators that track previous weights updated via `factory` using `w`.
"""
function factory(sr::SubsetResampling, w::AbstractVector)::SubsetResampling
    fees = factory(sr.fees, w)
    opt = factory(sr.opt, w)
    fb = factory(sr.fb, w)
    return SubsetResampling(; pe = sr.pe, wb = sr.wb, fees = fees, sets = sr.sets,
                            scale = sr.scale, opt = opt, wf = sr.wf, ex = sr.ex,
                            subset_size = sr.subset_size, n_subsets = sr.n_subsets,
                            max_comb = sr.max_comb, rng = sr.rng, seed = sr.seed, fb = fb,
                            brt = sr.brt, strict = sr.strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`SubsetResampling`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(sr::SubsetResampling, i, X::MatNum, args...)::SubsetResampling
    X = isa(sr.pe, AbstractPriorResult) ? sr.pe.X : X
    pe = port_opt_view(sr.pe, i)
    wb = port_opt_view(sr.wb, i)
    fees = port_opt_view(sr.fees, i)
    sets = port_opt_view(sr.sets, i)
    opt = port_opt_view(sr.opt, i, X)
    return SubsetResampling(; pe = pe, wb = wb, fees = fees, sets = sets, scale = sr.scale,
                            opt = opt, wf = sr.wf, ex = sr.ex, subset_size = sr.subset_size,
                            n_subsets = sr.n_subsets, max_comb = sr.max_comb, rng = sr.rng,
                            seed = sr.seed, fb = sr.fb, brt = sr.brt, strict = sr.strict)
end
"""
    subset_resampling_finaliser(N, n_subsets, asset_idx, ...)

Aggregate and finalise portfolio weights from subset resampling.

Combines optimised weights from multiple asset subsets, averaging over subsets to produce the final portfolio weights.

# Arguments

  - `N`: Total number of assets.
  - `n_subsets`: Number of asset subsets used in resampling.
  - `asset_idx`: Matrix of asset indices for each subset.
  - Additional weight and parameter inputs.

# Returns

  - Final aggregated portfolio weight vector.

# Related

  - [`MultipleRandomised`](@ref)
"""
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                     ress::VecOpt, ::VecNum, ::Nothing)
    w = zeros(eltype(ress[1].w), N)
    for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        w[idx] .+= ress[i].w
    end
    w /= n_subsets
    retcode, w = finalise_weight_bounds(wf, wb, w)
    return retcode, w
end
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                     ress::VecOpt, ws::VecVecNum, ::Nothing)
    M = length(ws)
    w = [zeros(eltype(ress[1].w[i]), N) for i in 1:M]
    for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        for j in 1:M
            w[j][idx] .+= ress[i].w[j]
        end
    end
    for j in 1:M
        w[j] /= n_subsets
    end
    retcode_w = [finalise_weight_bounds(wf, wb, wi) for wi in w]
    return map(x -> x[1], retcode_w), map(x -> x[2], retcode_w)
end
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                     ress::VecOpt, ::VecNum, scale::VecNum)
    w = zeros(eltype(ress[1].w), N)
    for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        w[idx] .+= scale[i] * ress[i].w
    end
    w /= sum(scale)
    retcode, w = finalise_weight_bounds(wf, wb, w)
    return retcode, w
end
function subset_resampling_finaliser(N::Integer, n_subsets::Integer, asset_idx::MatNum,
                                     wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                     ress::VecOpt, ws::VecVecNum, scale::VecNum)
    M = length(ws)
    w = [zeros(eltype(ress[1].w[i]), N) for i in 1:M]
    for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        for j in 1:M
            w[j][idx] .+= scale[i] * ress[i].w[j]
        end
    end
    denom = sum(scale)
    for j in 1:M
        w[j] /= denom
    end
    retcode_w = [finalise_weight_bounds(wf, wb, wi) for wi in w]
    return map(x -> x[1], retcode_w), map(x -> x[2], retcode_w)
end
function _optimise(sr::SubsetResampling, rd::ReturnsResult; dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    rd = returns_result_picker(rd, sr.brt)
    pr = prior(sr.pe, rd; dims = dims)
    X = pr.X
    N = size(X, 2)
    (; subset_size, n_subsets, max_comb, rng, seed) = sr
    subset_size = get_subset_size(subset_size, pr)
    n_subsets = get_n_subsets(n_subsets, pr)
    if !isnothing(sr.scale)
        @argcheck(length(sr.scale) == n_subsets)
    end
    n_comb = binomial(N, subset_size)
    @argcheck(n_subsets <= n_comb,
              "n_subsets = $n_subsets must not be greater than `binomial(assets, subset_size) = n_comb => binomial($N, $subset_size) = $n_comb`.")
    asset_idx = sample_unique_assets(N, subset_size, n_subsets; max_comb = max_comb,
                                     rng = rng, seed = seed)
    fees = fees_constraints(sr.fees, sr.sets; datatype = eltype(X), strict = sr.strict)
    opt = sr.opt
    ress = Vector{NonFiniteAllocationOptimisationResult}(undef, n_subsets)
    FLoops.@floop sr.ex for i in 1:n_subsets
        idx = view(asset_idx, :, i)
        opti = port_opt_view(opt, idx, X)
        rdi = returns_result_view(rd, idx)
        ress[i] = optimise(opti, rdi; dims = dims, branchorder = branchorder,
                           str_names = str_names, save = save, kwargs...)
    end
    wb = weight_bounds_constraints(sr.wb, sr.sets; N = N, strict = sr.strict,
                                   datatype = eltype(X))
    retcode, w = subset_resampling_finaliser(N, n_subsets, asset_idx, wb, sr.wf, ress,
                                             ress[1].w, sr.scale)
    return SubsetResamplingResult(typeof(sr), pr, wb, fees, ress, asset_idx, retcode, w,
                                  nothing)
end
"""
    optimise(sr::SubsetResampling{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                     <:Any, <:Any, <:Any, <:Any, <:Any, Nothing
                 }, rd::ReturnsResult;
             dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
             save::Bool = true, kwargs...) -> SubsetResamplingResult

Run the Subset Resampling portfolio optimisation.

# Arguments

  - `sr`: The subset resampling optimiser to use.
  - $(arg_dict[:rd])
  - `dims`: The dimension along which observations advance in time.
  - `branchorder`: Passed to the internal optimiser. The branch order to use for the clusterisation.
  - `str_names`: Passed to the internal optimiser. Whether to use string names for the assets in the optimisation.
  - `save`: Passed to the internal optimiser. Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`SubsetResampling`](@ref)
  - [`SubsetResamplingResult`](@ref)
"""
function optimise(sr::SubsetResampling{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(sr, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export SubsetResampling
