"""
    attribution_net_returns(w::VecNum, X::MatNum, fees::Option{<:Fees}, strict::Bool,
                            amsk::Option{<:AbstractMatrix{Bool}} = nothing)

Return the net portfolio return series over the finite entries of the asset returns.

A point-in-time panel carries a `NaN` at every `(observation, asset)` pair where the asset is inactive: before it lists, after it delists, and at a non-investable asset's whole column. `0 * NaN` is `NaN`, so a zero weight does not remove a `NaN` from the product `X * w`. The series is therefore formed over the finite entries.

A pair with a zero weight contributes nothing whatever it holds, and no message names it. A pair with a non-zero weight and a non-finite return splits by the active mask `amsk`. An active pair is a holiday: the asset is listed and its price does not move, so its return is exactly zero, and the pair fills zero with no message. An inactive pair is a holding with no return to earn. After a delisting its return can be as low as `-100 %`, so a zero there is an assumption the caller must see, and the pair takes the library's strictness policy through [`strict_diagnostic`](@ref). By default a warning names the observations and the assets, and the pair contributes zero. Under `strict`, an `ArgumentError` names them instead. Without a mask no pair is known to be a holiday, so every held pair with a non-finite return takes the policy. Under a walk-forward the held pairs after a delisting carry a zero weight already, so the default is silent there.

# Mathematical definition

```math
\\begin{align}
\\tilde{x}_{ti} &= \\begin{cases} x_{ti} & x_{ti} \\text{ finite}\\,, \\\\ 0 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

The series is the net series [`calc_net_returns`](@ref) forms from ``\\tilde{\\mathbf{X}}`` and ``\\boldsymbol{w}``. A panel with no non-finite entry gives ``\\tilde{\\mathbf{X}} = \\mathbf{X}``.

Where:

  - ``x_{ti}``, ``\\tilde{x}_{ti}``: Return of asset ``i`` at observation ``t``, and its finite part, the entries of ``\\mathbf{X}`` and ``\\tilde{\\mathbf{X}}``.
  - $(math_dict[:w_port])

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns, `observations × assets`.
  - `fees`: Fees the net series is formed against, or `nothing`.
  - `strict`: Whether a non-finite return at an inactive held pair raises rather than warns.
  - `amsk`: The active mask of the asset returns, `observations × assets`, or `nothing` when the returns carry none.

# Validation

  - Every inactive held pair of `X` is finite, else a warning naming the pairs is emitted, or an `ArgumentError` naming them is raised under `strict`. Without a mask the rule reads every held pair.

# Returns

  - `ret::VecNum`: The net portfolio return series, one entry per observation.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_investable_diagnostic`](@ref)
  - [`calc_net_returns`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`held_gap_pairs`](@ref)
"""
function attribution_net_returns(w::VecNum, X::MatNum, fees::Option{<:Fees}, strict::Bool,
                                 amsk::Option{<:AbstractMatrix{Bool}} = nothing)
    if all(isfinite, X)
        return calc_net_returns(w, X, fees)
    end
    held = filter(p -> isnothing(amsk) || !amsk[p[1], p[2]], held_gap_pairs(w, X))
    if !isempty(held)
        assets = unique(last.(held))
        strict_diagnostic("a factor attribution cannot decompose a holding that earns no return. Assets $(assets) carry a non-finite return at $(length(held)) held (observation, asset) pair(s) outside their active span, the first at observation $(first(held)[1]). Those pairs contribute zero to the net series, so the total understates the portfolio by whatever they earned. A pair inside the active span is a holiday and fills zero with no message, which only returns that carry an active mask can show. Pass `strict = true` to refuse instead, zero the weights over the observations the asset is inactive, or pass a weight history.",
                          strict)
    end
    Y = attribution_finite(X)
    return calc_net_returns(w, Y, fees)
end
"""
    attribution_active_mask(pnl::Nothing)
    attribution_active_mask(pnl::AssetPanel)

Return the active mask a realised attribution reads off the Asset Panel of the returns data, or `nothing`.

A returns result without a panel states no mask, so no blank cell of it is known to be a holiday. A static panel puts every asset in its universe at every observation, so every cell of it is active, through [`panel_active_cells`](@ref).

# Arguments

  - `pnl`: The Asset Panel of the returns data, or `nothing`.

# Returns

  - `amsk::Option{<:AbstractMatrix{Bool}}`: The active cells, `observations × assets`, or `nothing`.

# Related

  - [`attribution_net_returns`](@ref)
  - [`panel_active_cells`](@ref)
  - [`AssetPanel`](@ref)
"""
function attribution_active_mask(::Nothing)::Nothing
    return nothing
end
function attribution_active_mask(pnl::AssetPanel)
    return panel_active_cells(pnl)
end
"""
    attribution_returns_entry(w::VecNum, pr::AbstractPriorResult, X::MatNum,
                              amsk::Option{<:AbstractMatrix{Bool}}, fees::Option{<:Fees},
                              args...; strict::Bool = false, kwargs...)

Form the net return series of constant weights over an asset return matrix, then attribute it.

Every realised method of [`factor_attribution`](@ref) that takes constant weights and asset returns calls this function. A bare matrix carries no active mask, and a returns result carries the mask of its Asset Panel, so each reaches [`attribution_net_returns`](@ref) with the rule it can state.

# Algorithm

 1. Refuse weights over another universe through [`assert_attribution_assets`](@ref).
 2. Form the net series with [`attribution_net_returns`](@ref).
 3. Decompose the series with [`attribution_realised_entry`](@ref), or roll it with [`attribution_rolling_entry`](@ref) when `args` holds the window.

# Arguments

  - `w`: Portfolio weights.
  - `pr`: Prior result carrying the factor model block.
  - `X`: Asset returns, `observations × assets`.
  - `amsk`: The active mask of the asset returns, or `nothing`.
  - `fees`: Fees the net series is formed against, or `nothing`.
  - `args`: The positional `window` of a rolling attribution, or nothing.
  - `strict`: Whether a held non-investable asset, or a non-finite return at an inactive held pair, raises rather than warns.
  - `kwargs`: The keywords of the entry function.

# Validation

  - The weights hold one entry for each asset of `pr`, else a `DimensionMismatch` is raised.
  - The rules of [`attribution_net_returns`](@ref), and of the entry function.

# Returns

  - `fa`: The attribution, or one attribution per window.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_net_returns`](@ref)
  - [`attribution_active_mask`](@ref)
"""
function attribution_returns_entry(w::VecNum, pr::AbstractPriorResult, X::MatNum,
                                   amsk::Option{<:AbstractMatrix{Bool}},
                                   fees::Option{<:Fees}, args...; strict::Bool = false,
                                   kwargs...)
    assert_attribution_assets(w, length(pr.mu))
    ret = attribution_net_returns(w, X, fees, strict, amsk)
    if isempty(args)
        return attribution_realised_entry(w, pr, ret; strict = strict, kwargs...)
    end
    return attribution_rolling_entry(w, pr, ret, args...; strict = strict, kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, X::MatNum,
                            fees::Option{<:Fees} = nothing;
                            kwargs...)::FactorAttributionResult
    return attribution_returns_entry(w, pr, X, nothing, fees; kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, X::MatNum,
                            fees::Option{<:Fees}, window::Integer; kwargs...)
    return attribution_returns_entry(w, pr, X, nothing, fees, window; kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, X::MatNum, window::Integer;
                            kwargs...)
    return factor_attribution(w, pr, X, nothing, window; kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, rd::ReturnsResult,
                            fees::Option{<:Fees} = nothing;
                            kwargs...)::FactorAttributionResult
    return attribution_returns_entry(w, pr, rd.X, attribution_active_mask(rd.pnl), fees;
                                     kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, rd::ReturnsResult,
                            fees::Option{<:Fees}, window::Integer; kwargs...)
    return attribution_returns_entry(w, pr, rd.X, attribution_active_mask(rd.pnl), fees,
                                     window; kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, rd::ReturnsResult,
                            window::Integer; kwargs...)
    return factor_attribution(w, pr, rd, nothing, window; kwargs...)
end
function factor_attribution(res::OptimisationResult, pr::Option{<:Pr_RR}, rd::ReturnsResult;
                            kwargs...)::FactorAttributionResult
    # The caller's `rd` is on the universe of `res.w`, as a caller's `pr` is, so it takes
    # the same view the prior and the weights take, its Asset Panel with it.
    imsk, w, pr, fees = result_investable_view(res, pr)
    return factor_attribution(w, pr, investable_returns_view(imsk, rd), fees; kwargs...)
end
function factor_attribution(res::OptimisationResult, pr::Option{<:Pr_RR}, rd::ReturnsResult,
                            window::Integer; kwargs...)
    imsk, w, pr, fees = result_investable_view(res, pr)
    return factor_attribution(w, pr, investable_returns_view(imsk, rd), fees, window;
                              kwargs...)
end
function factor_attribution(W::MatNum, pr::AbstractPriorResult, ret::VecNum;
                            kwargs...)::FactorAttributionResult
    return attribution_realised_entry(W, pr, ret; kwargs...)
end
function factor_attribution(W::MatNum, pr::AbstractPriorResult, ret::VecNum,
                            window::Integer; kwargs...)
    return attribution_rolling_entry(W, pr, ret, window; kwargs...)
end
function factor_attribution(pred::MultiPeriodPredictionResult, pr::AbstractPriorResult;
                            kwargs...)::FactorAttributionResult
    W, ret = attribution_prediction_history(pred)
    return attribution_realised_entry(W, pr, ret; key = attribution_series_key(pred),
                                      kwargs...)
end
function factor_attribution(pred::MultiPeriodPredictionResult, pr::AbstractPriorResult,
                            window::Integer; kwargs...)
    W, ret = attribution_prediction_history(pred)
    return attribution_rolling_entry(W, pr, ret, window; key = attribution_series_key(pred),
                                     kwargs...)
end
"""
    attribution_array_keywords(kwargs) -> NamedTuple

Split the keywords of a realised bare-array method into those of the block and those of the decomposition.

The block keywords describe the arrays, and [`attribution_array_block`](@ref) takes them. Every other keyword goes to [`attribution_realised_entry`](@ref) or [`attribution_rolling_entry`](@ref), whose keyword list is closed, so a misspelt keyword is refused there by name.

# Arguments

  - `kwargs`: The keywords the caller passed.

# Returns

  - `kw::NamedTuple`: The block keywords `blk`, among `lag`, `trim`, `rw`, `vs`, `fcb`, `observed` and `fam`, and the other keywords `entry`.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_array_block`](@ref)
"""
function attribution_array_keywords(kwargs)
    nt = values(kwargs)
    ks = filter(in((:lag, :trim, :rw, :vs, :fcb, :observed, :fam)), keys(nt))
    return (; blk = NamedTuple{ks}(nt), entry = Base.structdiff(nt, NamedTuple{ks}))
end
function factor_attribution(W::VecNum_MatNum, B::MatNum_Arr3Num, f::MatNum, eps::MatNum,
                            ret::VecNum; kwargs...)::FactorAttributionResult
    kw = attribution_array_keywords(kwargs)
    return attribution_realised_entry(W, attribution_array_block(B, f, eps; kw.blk...), ret;
                                      kw.entry...)
end
function factor_attribution(W::VecNum_MatNum, B::MatNum_Arr3Num, f::MatNum, eps::MatNum,
                            ret::VecNum, window::Integer; kwargs...)
    kw = attribution_array_keywords(kwargs)
    return attribution_rolling_entry(W, attribution_array_block(B, f, eps; kw.blk...), ret,
                                     window; kw.entry...)
end
"""
    attribution_prediction_history(pred::MultiPeriodPredictionResult)

Return the weight history and the net return series a cross-validation produced.

Each fold holds its own weights and its own net series, so the history is the folds stacked in order. A fold that recorded a Held Weights result carries its drifted path, and a fold that recorded none held its target weights for the whole fold.

# Arguments

  - `pred`: A multi-period prediction result.

# Returns

  - `W::MatNum`: The weight history, `observations × assets`.
  - `ret::VecNum`: The net portfolio return series.

# Related

  - [`factor_attribution`](@ref)
  - [`weight_path`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function attribution_prediction_history(pred::MultiPeriodPredictionResult)
    W = reduce(vcat, attribution_fold_weights(p) for p in pred.pred)
    ret = reduce(vcat, attribution_fold_returns(p) for p in pred.pred)
    return W, ret
end
"""
    attribution_fold_weights(pred::PredictionResult)

Return the weight history one fold of a cross-validation held.

A fold that recorded a Held Weights result returns its drifted path, and a fold that recorded none held its target weights over every observation of the fold.

# Arguments

  - `pred`: A single-fold prediction result.

# Returns

  - `W::MatNum`: The fold's weight history, `observations × assets`.

# Related

  - [`attribution_prediction_history`](@ref)
  - [`weight_path`](@ref)
"""
function attribution_fold_weights(pred::PredictionResult)
    return attribution_fold_weights(pred, pred.hw)
end
function attribution_fold_weights(pred::PredictionResult, hw::HeldWeightsResult)
    return weight_path(hw, pred.res.w)
end
function attribution_fold_weights(pred::PredictionResult, ::Nothing)
    return repeat(transpose(pred.res.w), length(attribution_fold_returns(pred)))
end
"""
    attribution_fold_returns(pred::PredictionResult)

Return the net return series one fold of a cross-validation produced.

# Arguments

  - `pred`: A single-fold prediction result.

# Returns

  - `ret::VecNum`: The fold's net portfolio return series.

# Related

  - [`attribution_prediction_history`](@ref)
  - [`PredictionResult`](@ref)
"""
function attribution_fold_returns(pred::PredictionResult)
    X = pred.rd.X
    return isa(X, VecVecNum) ? first(X) : X
end
