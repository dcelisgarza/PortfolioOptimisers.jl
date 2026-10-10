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

  - `kw::NamedTuple`: The block keywords `blk`, among `lag`, `trim`, `rw`, `vs`, `fcb`, `unseen`, `observed`, `fam` and `h1`, and the other keywords `entry`.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_array_block`](@ref)
"""
function attribution_array_keywords(kwargs)
    nt = values(kwargs)
    ks = filter(in((:lag, :trim, :rw, :vs, :fcb, :unseen, :observed, :fam, :h1)), keys(nt))
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
"""
    attribution_unknown_kinds(rule::AbstractUnknownEntryRule) -> NamedTuple
    attribution_unknown_kinds(rule::KindwiseUnknown) -> NamedTuple

Return the rule of each kind of unknown entry.

A preset applies to every kind, so it is the rule of each kind. [`KindwiseUnknown`](@ref) holds the rule of each kind in a field.

# Arguments

  - `rule`: The rule for an unknown entry.

# Returns

  - `kinds::NamedTuple`: The rule `unstated` for an entry the model does not state, and the rule `leverage` for the variance split of a Leverage-One Pair.

# Related

  - [`AbstractUnknownEntryRule`](@ref)
  - [`KindwiseUnknown`](@ref)
"""
function attribution_unknown_kinds(rule::AbstractUnknownEntryRule)
    return (; unstated = rule, leverage = rule)
end
function attribution_unknown_kinds(rule::KindwiseUnknown)
    return (; unstated = rule.unstated, leverage = rule.leverage)
end
"""
    attribution_leverage_marks(rr::AbstractLoadingsRegressionResult) -> Nothing
    attribution_leverage_marks(rr::CrossSectionalFactorModel) -> Option{<:AbstractMatrix{Bool}}

Return the leverage-one mask of a factor model block, or `nothing`.

The mask is the `h1` field of the cross-sectional fit, `observations × assets`, on the rows of the idiosyncratic returns. A block that is not cross-sectional fits no design across the assets, so it marks no pair.

# Arguments

  - `rr`: The factor model block.

# Returns

  - `h1::Option{<:AbstractMatrix{Bool}}`: The leverage-one mask, or `nothing` when the block states none.

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`attribution_leverage_split`](@ref)
"""
function attribution_leverage_marks(::AbstractLoadingsRegressionResult)::Nothing
    return nothing
end
function attribution_leverage_marks(rr::CrossSectionalFactorModel)
    return isnothing(rr.csr) ? nothing : rr.csr.h1
end
"""
    attribution_marked_assets(h1::Nothing) -> Nothing
    attribution_marked_assets(h1::AbstractMatrix{Bool}) -> BitVector

Return the assets that the leverage-one mask marks at one row or more.

The predicted side reads one set of loadings and the moments of the whole fit. The factor covariance reads the factor returns of every row, and the factor return of a row where an asset is a Leverage-One Pair holds the own return of that asset. So the predicted split of an asset is not identified when the mask marks it at any row.

# Arguments

  - `h1`: The leverage-one mask, `observations × assets`, or `nothing`.

# Returns

  - `lev::Option{<:BitVector}`: `true` at every asset the mask marks at one row or more, or `nothing`.

# Related

  - [`attribution_leverage_marks`](@ref)
  - [`attribution_leverage_split`](@ref)
"""
function attribution_marked_assets(::Nothing)::Nothing
    return nothing
end
function attribution_marked_assets(h1::AbstractMatrix{Bool})
    return BitVector(vec(any(h1; dims = 1)))
end
"""
    attribution_leverage_split(rule::ZeroUnknown, args...) -> Nothing
    attribution_leverage_split(rule::EntrywiseUnknown, lev::Nothing, args...) -> Nothing
    attribution_leverage_split(rule::EntrywiseUnknown, lev::AbstractVector{Bool}, w::VecNum,
                               M::MatNum) -> Option{<:NamedTuple}
    attribution_leverage_split(rule::EntrywiseUnknown, h1::AbstractMatrix{Bool},
                               W::VecNum_MatNum, B::MatNum_Arr3Num) -> Option{<:NamedTuple}

Return the assets and the factors whose variance split the attribution does not identify, or `nothing`.

A Leverage-One Pair has a direction of the design of its own. At its row the data identify only the sum of the factor return of that direction and the own return of the asset, so every own variance from zero upwards fits the data equally well. [`ZeroUnknown`](@ref) reads the split as fitted, so it marks nothing. [`EntrywiseUnknown`](@ref) marks every asset that the portfolio holds at a row where the mask marks it, and every factor whose only member at that row is the asset.

The predicted method reads an asset mask and one set of loadings, through [`attribution_marked_assets`](@ref). The realised method reads the mask and the weights row by row.

# Arguments

  - `rule`: The rule for the variance split of a Leverage-One Pair.
  - `lev`: Assets the mask marks at one row or more.
  - `h1`: The leverage-one mask, `observations × assets`.
  - `w`: Portfolio weights.
  - `W`: Portfolio weights, or the weight history, `observations × assets`.
  - `M`: Loadings, `assets × factors`.
  - `B`: Loadings, or the exposure history, `observations × assets × factors`, zero at an inactive pair.

# Returns

  - `lv::Option{<:NamedTuple}`: `nothing` when the portfolio holds no marked asset. Otherwise the asset mask `assets` and the factor mask `factors`.

# Related

  - [`KindwiseUnknown`](@ref)
  - [`attribution_sole_factors!`](@ref)
  - [`attribution_leverage_nan`](@ref)
"""
function attribution_leverage_split(::ZeroUnknown, args...)::Nothing
    return nothing
end
function attribution_leverage_split(::EntrywiseUnknown, ::Nothing, args...)::Nothing
    return nothing
end
function attribution_leverage_split(::EntrywiseUnknown, lev::AbstractVector{Bool},
                                    w::VecNum, M::MatNum)
    held = BitVector(lev .& .!iszero.(w))
    if !any(held)
        return nothing
    end
    fac = falses(size(M, 2))
    for i in findall(held)
        attribution_sole_factors!(fac, M, i)
    end
    return (; assets = held, factors = fac)
end
function attribution_leverage_split(::EntrywiseUnknown, h1::AbstractMatrix{Bool},
                                    W::VecNum_MatNum, B::MatNum_Arr3Num)
    held = falses(size(h1, 2))
    fac = falses(size(B, ndims(B)))
    for t in axes(h1, 1)
        wt = attribution_weights(W, t)
        for i in findall(view(h1, t, :) .& .!iszero.(wt))
            held[i] = true
            attribution_sole_factors!(fac, attribution_slice(B, t), i)
        end
    end
    return any(held) ? (; assets = held, factors = fac) : nothing
end
"""
    attribution_sole_factors!(fac::AbstractVector{Bool}, Bt::MatNum, i::Integer)

Mark every factor whose only member is asset `i`.

A member of a factor is an asset with a finite exposure to it that is not zero. A Leverage-One Pair of a one-hot family is the only member of its level, so the factor return of the level holds the own return of the asset.

# Arguments

  - `fac`: The factor mask, changed in place.
  - `Bt`: Exposures of one row, `assets × factors`.
  - `i`: The asset.

# Returns

  - `fac::AbstractVector{Bool}`: The factor mask.

# Related

  - [`attribution_leverage_split`](@ref)
"""
function attribution_sole_factors!(fac::AbstractVector{Bool}, Bt::MatNum, i::Integer)
    member(x) = isfinite(x) && !iszero(x)
    for k in axes(Bt, 2)
        if member(Bt[i, k]) && isone(count(member, view(Bt, :, k)))
            fac[k] = true
        end
    end
    return fac
end
"""
    attribution_leverage_nan(lv::Nothing, x) -> x
    attribution_leverage_nan(lv::NamedTuple, x::Nothing) -> Nothing
    attribution_leverage_nan(lv::NamedTuple, c::AttributionComponent) -> AttributionComponent
    attribution_leverage_nan(lv::NamedTuple, bd::AttributionBreakdown) -> AttributionBreakdown
    attribution_leverage_nan(lv::NamedTuple, abd::AssetAttributionBreakdown)
        -> AssetAttributionBreakdown
    attribution_leverage_nan(lv::NamedTuple, afc::AssetFactorContribution)
        -> AssetFactorContribution

Return a part of an attribution with `NaN` in every variance number that reads the split of a held Leverage-One Pair.

The total return of the pair is identified, and the split between its systematic and its idiosyncratic part is not. A component of the portfolio loses its volatility, its volatility contribution, its variance share and its correlation. A factor row that `lv` marks loses its volatility contribution, its variance share and its correlation, and keeps the standalone volatility of the factor. An asset row that `lv` marks loses its systematic and its idiosyncratic volatility contribution, and keeps its total. Every mean keeps its value, because the own return has an expected value of zero, so the mean split is unbiased. `lv = nothing` changes nothing.

# Arguments

  - `lv`: The marks of [`attribution_leverage_split`](@ref), or `nothing`.
  - `x`, `c`, `bd`, `abd`, `afc`: The part of the attribution, or `nothing` for an axis the caller did not ask for.

# Returns

  - The part, with `NaN` in the numbers that read the split.

# Related

  - [`attribution_leverage_split`](@ref)
  - [`FactorAttributionResult`](@ref)
"""
function attribution_leverage_nan(::Nothing, x)
    return x
end
function attribution_leverage_nan(::NamedTuple, ::Nothing)::Nothing
    return nothing
end
function attribution_leverage_nan(::NamedTuple, c::AttributionComponent)
    nan = oftype(c.vol, NaN)
    return AttributionComponent(nan, nan, nan, c.mu_contrib, nan, c.mu_se)
end
function attribution_leverage_nan(lv::NamedTuple, bd::AttributionBreakdown)
    m = lv.factors
    return AttributionBreakdown(bd.labels, bd.exposure, bd.exposure_std, bd.vol,
                                attribution_mask_nan(bd.corr, m),
                                attribution_mask_nan(bd.vol_contrib, m),
                                attribution_mask_nan(bd.pct_var, m), bd.mu, bd.mu_contrib,
                                bd.mu_se)
end
function attribution_leverage_nan(lv::NamedTuple, abd::AssetAttributionBreakdown)
    m = lv.assets
    return AssetAttributionBreakdown(abd.weight, abd.weight_std,
                                     attribution_mask_nan(abd.sys_vol_contrib, m),
                                     abd.sys_mu_contrib,
                                     attribution_mask_nan(abd.idio_vol_contrib, m),
                                     abd.idio_mu_contrib, abd.vol, abd.corr,
                                     abd.vol_contrib, abd.pct_var, abd.mu, abd.mu_contrib)
end
function attribution_leverage_nan(lv::NamedTuple, afc::AssetFactorContribution)
    return AssetFactorContribution(attribution_mask_nan(afc.vol_contrib,
                                                        lv.assets .& transpose(lv.factors)),
                                   afc.mu_contrib)
end
"""
    attribution_mask_nan(A::AbstractArray, m::AbstractArray{Bool}) -> AbstractArray

Return `A` with `NaN` at every entry that `m` marks.

Each `NaN` takes the type of the entry it replaces.

# Arguments

  - `A`: The numbers.
  - `m`: The mask, the shape of `A`.

# Returns

  - `B::AbstractArray`: A copy of `A` with `NaN` at the marked entries.

# Related

  - [`attribution_leverage_nan`](@ref)
"""
function attribution_mask_nan(A::AbstractArray, m::AbstractArray{Bool})
    return map((x, b) -> b ? oftype(x, NaN) : x, A, m)
end
"""
    attribution_error_pass(s2::AbstractMatrix, g::MatNum, al::NamedTuple, red::NamedTuple,
                           fam::Option{<:VecStr}, s1::Number) -> NamedTuple

Return the standard errors of the mean return contributions for one matrix of idiosyncratic variances.

The sandwich of each observation reads `s2`, and the function sums it over the observations through the portfolio exposure. [`attribution_standard_errors`](@ref) calls it with the variances of the regression pairs. [`attribution_leverage_errors`](@ref) calls it with an indicator in place of the variances, so one reduction gives the answer and the weight that each output puts on a set of pairs. At an observation with an Unseen Member, [`attribution_changed_covariance`](@ref) maps the sandwich of the changed design to the covariance of the factor returns.

# Arguments

  - `s2`: The idiosyncratic variance of each pair, `observations × assets`, zero outside the regression.
  - `g`: The per-observation portfolio exposure on the raw axis, `observations × factors`.
  - `al`: The aligned factor model history, with the regression weights `rw`.
  - `red`: The regression basis and the changes `P` of the observations with an Unseen Member, from [`attribution_reduce_for_errors`](@ref).
  - `fam`: The family label of each raw factor, or `nothing`.
  - `s1`: The factor a mean takes under the annualisation.

# Returns

  - `sys::Real`: The standard error of the systematic mean return contribution.
  - `factor::VecNum`: The standard error of each factor's, `NaN` for an observed factor.
  - `family::Option{<:VecNum}`: The standard error of each family's, or `nothing`.

# Related

  - [`attribution_standard_errors`](@ref)
  - [`attribution_sandwich`](@ref)
  - [`attribution_family_errors`](@ref)
"""
function attribution_error_pass(s2::AbstractMatrix, g::MatNum, al::NamedTuple,
                                red::NamedTuple, fam::Option{<:VecStr}, s1::Number)
    T, K = size(g)
    cur = attribution_observed_indices(al.no, K)
    keep = findall(!, red.observed)
    se(v) = s1 * sqrt(max(zero(v), v)) / T
    V = [attribution_changed_covariance(unseen_member_change_at(red.P, t),
                                        attribution_sandwich(view(red.B, t, :, :),
                                                             view(al.rw, t, :),
                                                             view(s2, t, :), keep), keep)
         for t in 1:T]
    sys = se(sum(LinearAlgebra.dot(view(red.g, t, keep), V[t], view(red.g, t, keep))
                 for t in 1:T))
    Vf = attribution_expand_errors(al.fcb, attribution_scatter(V, keep, red.nr))
    factor = [se(sum(g[t, k]^2 * Vf[t, k, k] for t in 1:T)) for k in 1:K]
    for k in cur
        factor[k] = oftype(factor[k], NaN)
    end
    return (; sys = sys, factor = factor,
            family = attribution_family_errors(fam, g, Vf, s1, T, cur))
end
"""
    attribution_leverage_errors(rule::ZeroUnknown, h1, reg, ses, pass) -> NamedTuple
    attribution_leverage_errors(rule::EntrywiseUnknown, h1::Nothing, reg, ses, pass)
        -> NamedTuple
    attribution_leverage_errors(rule::EntrywiseUnknown, h1::AbstractMatrix{Bool},
                                reg::AbstractMatrix{Bool}, ses::NamedTuple, pass)
        -> NamedTuple

Return the standard errors with `NaN` at every output whose sandwich reads a Leverage-One Pair.

The sandwich reads the idiosyncratic variance of a marked pair, and that variance is not identified: every own variance from zero upwards fits the data equally well. An output reads the pair when its sandwich gives the pair a coefficient that is not zero. The function runs the reduction of `pass` twice more, once with the indicator of the marked pairs and once with the indicator of every regression pair. The first sum is the leverage index of each output, and the second is its scale. An output is `NaN` when its leverage index exceeds ``\\varepsilon`` times its scale.

A coefficient of round-off gives a ratio far below ``\\varepsilon``, of the order of ``10^{-19}`` on a panel of 40 assets, because the quadratic form adds round-off of its own to the square of the coefficient. This is the case of an output that the fit separates from the pair, a style factor or a portfolio that holds no marked pair. A real coefficient, through the zero-sum constraint of a family or through a holding of the pair, gives a ratio far above ``\\varepsilon``, of the order of ``10^{-3}`` or more. The pair stays in the sandwich: without it the Gram matrix is singular in the direction of the level, and the pseudo-inverse gives a minimum-norm number that is not an error.

[`ZeroUnknown`](@ref) reads the plug-in variance of the pair, so it changes nothing. `h1 = nothing` marks no pair.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{c}_{t} &= \\mathbf{Q}_{t} \\mathbf{B}_{t} \\mathbf{G}_{t}^{+} \\boldsymbol{g}_{t}\\,, \\\\
\\ell &= \\sum_{t=1}^{T} \\sum_{i=1}^{N} c_{ti}^{2}\\, h_{ti}\\,, \\\\
S &= \\sum_{t=1}^{T} \\sum_{i=1}^{N} c_{ti}^{2}\\, a_{ti}\\,.
\\end{align}
```

The output is `NaN` when ``\\ell > \\varepsilon\\, S``. The definition is that of the systematic error. A factor or a family takes the same sums with ``\\boldsymbol{g}_{t}`` restricted to its own entries.

Where:

  - ``\\boldsymbol{c}_{t}``: Coefficient of the output on each pair of observation ``t``, one entry ``c_{ti}`` per asset.
  - ``\\ell``, ``S``: Leverage index and scale of the output.
  - ``h_{ti}``: Whether pair ``(t, i)`` is a Leverage-One Pair in the regression.
  - ``a_{ti}``: Whether pair ``(t, i)`` is in the regression: active, with a regression weight that is not zero.
  - ``\\mathbf{G}_{t}^{+}``: Inverse of the Gram matrix of observation ``t``, or its pseudo-inverse, as [`attribution_sandwich`](@ref) states it.
  - $(math_dict[:Q_t_att])
  - $(math_dict[:B_t_att])
  - $(math_dict[:g_t_att])
  - $(math_dict[:eps_machine])
  - $(math_dict[:N])
  - $(math_dict[:T])

# Arguments

  - `rule`: The rule for the variance of a Leverage-One Pair.
  - `h1`: The leverage-one mask, `observations × assets`, or `nothing`.
  - `reg`: Whether each pair is in the regression, `observations × assets`.
  - `ses`: The standard errors, from [`attribution_error_pass`](@ref).
  - `pass`: The reduction, a function of the matrix of variances.

# Returns

  - `ses::NamedTuple`: The standard errors, with `NaN` at every output that reads a marked pair.

# Related

  - [`attribution_standard_errors`](@ref)
  - [`attribution_leverage_split`](@ref)
  - [`KindwiseUnknown`](@ref)
"""
function attribution_leverage_errors(::ZeroUnknown, ::Any, ::Any, ses::NamedTuple, ::Any)
    return ses
end
function attribution_leverage_errors(::EntrywiseUnknown, ::Nothing, ::Any, ses::NamedTuple,
                                     ::Any)
    return ses
end
function attribution_leverage_errors(::EntrywiseUnknown, h1::AbstractMatrix{Bool},
                                     reg::AbstractMatrix{Bool}, ses::NamedTuple, pass)
    mk = reg .& h1
    if !any(mk)
        return ses
    end
    return map(attribution_leverage_error_nan, ses, pass(mk), pass(reg))
end
"""
    attribution_leverage_error_nan(x::Nothing, l::Nothing, s::Nothing) -> Nothing
    attribution_leverage_error_nan(x::Number, l::Number, s::Number) -> Number
    attribution_leverage_error_nan(x::AbstractVector, l::AbstractVector, s::AbstractVector)
        -> AbstractVector

Return a standard error, or `NaN` when its leverage index exceeds ``\\varepsilon`` times its scale.

The two indicator passes of [`attribution_leverage_errors`](@ref) return standard errors, which are the square roots of the two sums, so the test compares their squares. An output that the attribution states as `NaN` already, an observed factor for example, stays `NaN`. A family axis that the attribution does not have stays `nothing`.

# Arguments

  - `x`: The standard error.
  - `l`: The standard error of the indicator of the marked pairs.
  - `s`: The standard error of the indicator of every regression pair.

# Returns

  - `x`: The standard error, or `NaN`.

# Related

  - [`attribution_leverage_errors`](@ref)
"""
function attribution_leverage_error_nan(::Nothing, ::Nothing, ::Nothing)::Nothing
    return nothing
end
function attribution_leverage_error_nan(x::Number, l::Number, s::Number)
    return abs2(l) > eps(real(typeof(x))) * abs2(s) ? oftype(x, NaN) : x
end
function attribution_leverage_error_nan(x::AbstractVector, l::AbstractVector,
                                        s::AbstractVector)
    return attribution_leverage_error_nan.(x, l, s)
end
"""
    attribution_unknown_keep(rule::EntrywiseUnknown, w::VecNum)
    attribution_unknown_keep(rule::ZeroUnknown, w::VecNum)

Return the assets whose unknown entries stay unknown in a predicted attribution.

An entry of an asset with a weight of zero adds nothing to a sum, so its value is irrelevant, and it is replaced by zero under either rule. [`EntrywiseUnknown`](@ref) keeps the unknown entries of every held asset, so a sum that reads one is `NaN`. [`ZeroUnknown`](@ref) keeps none.

# Arguments

  - `rule`: The rule for an unknown entry.
  - `w`: Portfolio weights.

# Returns

  - `keep::BitVector`: `true` at every asset whose unknown entries stay `NaN`.

# Related

  - [`AbstractUnknownEntryRule`](@ref)
  - [`attribution_unknown_rows`](@ref)
  - [`attribution_unknown_block`](@ref)
"""
function attribution_unknown_keep(::EntrywiseUnknown, w::VecNum)
    return .!iszero.(w)
end
function attribution_unknown_keep(::ZeroUnknown, w::VecNum)
    return falses(length(w))
end
"""
    attribution_unknown_rows(A::AbstractArray, keep::AbstractVector{Bool})

Return the loadings, the intercept or the expected returns with every unknown entry outside the kept assets replaced by zero.

A finite entry stays as it is, so a held non-investable asset keeps its finite loadings, and the exposures that read them are exact. A non-finite entry of a kept asset stays too, so every number that reads it is `NaN`.

# Mathematical definition

```math
\\begin{align}
\\tilde{A}_{ik} &= \\begin{cases} A_{ik} & \\text{if } A_{ik} \\text{ is finite or } k_{i} = 1\\,, \\\\ 0 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``A_{ik}``, ``\\tilde{A}_{ik}``: Entry of asset ``i`` in column ``k``, before and after the rule. A vector has one column.
  - ``k_{i}``: Whether asset ``i`` keeps its unknown entries, from [`attribution_unknown_keep`](@ref).

# Arguments

  - `A`: The loadings, `assets × factors`, or a vector with one entry per asset.
  - `keep`: Whether each asset keeps its unknown entries.

# Returns

  - `A::AbstractArray`: The array after the rule.

# Related

  - [`attribution_unknown_keep`](@ref)
  - [`attribution_unknown_block`](@ref)
  - [`predicted_attribution`](@ref)
"""
function attribution_unknown_rows(A::AbstractArray, keep::AbstractVector{Bool})
    return ifelse.(keep .| isfinite.(A), A, zero(eltype(A)))
end
"""
    attribution_unknown_block(E::VecNum_MatNum, keep::AbstractVector{Bool})

Return the idiosyncratic covariance with every unknown entry outside the kept pairs replaced by zero.

The covariance sibling of [`attribution_unknown_rows`](@ref). A diagonal covariance travels as a vector, and takes the rule of the rows. A full covariance keeps an unknown entry only when both of its assets keep their unknown entries, because the quadratic form `w' E w` reads the entry `(i, j)` with the weight `w_i w_j`.

# Mathematical definition

```math
\\begin{align}
\\tilde{E}_{ij} &= \\begin{cases} E_{ij} & \\text{if } E_{ij} \\text{ is finite, or } k_{i} = k_{j} = 1\\,, \\\\ 0 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``E_{ij}``, ``\\tilde{E}_{ij}``: Idiosyncratic covariance of assets ``i`` and ``j``, before and after the rule.
  - ``k_{i}``: Whether asset ``i`` keeps its unknown entries, from [`attribution_unknown_keep`](@ref).

# Arguments

  - `E`: The idiosyncratic variances, one entry per asset, or the idiosyncratic covariance, `assets × assets`.
  - `keep`: Whether each asset keeps its unknown entries.

# Returns

  - `E::VecNum_MatNum`: The variances or the covariance after the rule.

# Related

  - [`attribution_unknown_rows`](@ref)
  - [`attribution_unknown_keep`](@ref)
  - [`attribution_idiosyncratic_matrix`](@ref)
"""
function attribution_unknown_block(e::VecNum, keep::AbstractVector{Bool})
    return attribution_unknown_rows(e, keep)
end
function attribution_unknown_block(E::MatNum, keep::AbstractVector{Bool})
    return ifelse.((keep .& transpose(keep)) .| isfinite.(E), E, zero(eltype(E)))
end
"""
    attribution_standalone(rule::EntrywiseUnknown, A::AbstractArray)
    attribution_standalone(rule::ZeroUnknown, A::AbstractArray)

Return the entries of the block that the standalone moments of the predicted asset axis read.

The standalone volatility and mean of an asset are moments of the asset, not contributions, so they read the block's own entries. [`EntrywiseUnknown`](@ref) reads them as they are, so the model states no volatility for an asset whose idiosyncratic variance is unknown. [`ZeroUnknown`](@ref) reads every unknown entry as zero.

# Arguments

  - `rule`: The rule for an unknown entry.
  - `A`: The loadings, the factor-orthogonal mean or the idiosyncratic block.

# Returns

  - `A::AbstractArray`: The entries the standalone moments read.

# Related

  - [`AbstractUnknownEntryRule`](@ref)
  - [`predicted_attribution_assets`](@ref)
  - [`attribution_finite`](@ref)
"""
function attribution_standalone(::EntrywiseUnknown, A::AbstractArray)
    return A
end
function attribution_standalone(::ZeroUnknown, A::AbstractArray)
    return attribution_finite(A)
end
"""
    attribution_held_entries(X::AbstractArray, w::VecNum)

Return a per-asset contribution with every non-finite entry of an asset the portfolio does not hold replaced by zero.

A contribution of asset `i` is its weight times a number of the model, divided by the portfolio volatility where it is a share. An asset with a weight of zero contributes zero whatever that number is, because the number is finite in the model even where the attribution cannot state it. A held asset whose unknown entry reaches the number keeps its `NaN`. A finite entry stays as it is, so the sign of a zero is kept.

# Mathematical definition

```math
\\begin{align}
\\tilde{x}_{ik} &= \\begin{cases} 0 & \\text{if } w_{i} = 0 \\text{ and } x_{ik} \\text{ is not finite}\\,, \\\\ x_{ik} & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``x_{ik}``, ``\\tilde{x}_{ik}``: Contribution of asset ``i`` in column ``k``, before and after the rule. A vector has one column.
  - $(math_dict[:w_port])

# Arguments

  - `X`: The contributions, one row per asset.
  - `w`: Portfolio weights.

# Returns

  - `X::AbstractArray`: The contributions after the rule.

# Related

  - [`predicted_attribution_assets`](@ref)
  - [`attribution_unknown_keep`](@ref)
"""
function attribution_held_entries(X::AbstractArray, w::VecNum)
    return ifelse.(iszero.(w) .& .!isfinite.(X), zero(eltype(X)), X)
end
"""
    attribution_unknown_note(rule::EntrywiseUnknown)
    attribution_unknown_note(rule::ZeroUnknown)

Return the sentence a predicted attribution adds to the report of a held non-investable asset.

The report names the assets, and this sentence states what the rule did with their unknown entries, so the reader knows which numbers to trust.

# Arguments

  - `rule`: The rule for an unknown entry.

# Returns

  - `note::String`: The sentence.

# Related

  - [`attribution_investable_diagnostic`](@ref)
  - [`AbstractUnknownEntryRule`](@ref)
"""
function attribution_unknown_note(::EntrywiseUnknown)
    return "The entries the prior states for them stay in the decomposition, and every number that reads an entry it does not state is `NaN`: the idiosyncratic part, the total, the remainder and every share of the portfolio volatility. Pass `unknown = ZeroUnknown()` to read those entries as zero"
end
function attribution_unknown_note(::ZeroUnknown)
    return "Every entry the prior does not state for them reads as zero, so the idiosyncratic part and the total understate the variance of the portfolio"
end

public attribution_unknown_keep, attribution_standalone, attribution_standalone_pairs,
       attribution_unknown_note, attribution_leverage_split, attribution_leverage_errors
