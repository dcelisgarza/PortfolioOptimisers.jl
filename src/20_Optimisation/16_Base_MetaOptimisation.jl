"""
    abstract type SubPortfolioUniverse end

Abstract supertype for a meta-optimiser's *sub-portfolio enumeration*: what a sub-portfolio is, and what it sees.

A meta-optimiser solves one inner problem per sub-portfolio, predicts each sub-portfolio's returns, and hands the outer optimiser a synthetic universe with one asset per sub-portfolio. That is one module, and the two shipped meta-optimisers differ in exactly one respect, which this type names:

  - [`NestedClustered`](@ref) enumerates **cluster index sets**. One inner optimiser is viewed onto each cluster, and every full-universe quantity — the Prior Result, the Fees — is viewed onto it too.
  - [`Stacking`](@ref) enumerates **inner optimisers**. Each one sees the whole universe, so nothing is viewed.

[`FullUniverse`](@ref) and [`ClusterUniverse`](@ref) declare the two. The module reads them back through [`sub_portfolio_predict`](@ref), [`sub_portfolio_view`](@ref) and [`fold_weight_matrix`](@ref), and a third meta-optimiser is a third declaration rather than a third copy of the module.

# Related

  - [`FullUniverse`](@ref)
  - [`ClusterUniverse`](@ref)
  - [`sub_portfolio_predict`](@ref)
  - [`sub_portfolio_view`](@ref)
  - [`predict_outer_returns`](@ref)
"""
abstract type SubPortfolioUniverse end
"""
$(DocStringExtensions.TYPEDEF)

Sub-portfolios are the inner optimisers, and each sees the whole universe. [`Stacking`](@ref)'s enumeration.

Nothing is viewed onto a sub-portfolio, and an inner weight vector is already full length — which is why the outer collapse pads nothing (see [`fold_weight_matrix`](@ref)).

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`ClusterUniverse`](@ref)
  - [`Stacking`](@ref)
"""
struct FullUniverse <: SubPortfolioUniverse end
"""
$(DocStringExtensions.TYPEDEF)

Sub-portfolios are cluster index sets, and each sees its own cluster. [`NestedClustered`](@ref)'s enumeration.

One inner optimiser serves every sub-portfolio, viewed onto that sub-portfolio's assets, and an inner weight vector is as long as its cluster — which is why the outer collapse zero-pads it onto the full asset axis (see [`fold_weight_matrix`](@ref)).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`FullUniverse`](@ref)
  - [`NestedClustered`](@ref)
"""
struct ClusterUniverse{T <: VecVecInt} <: SubPortfolioUniverse
    """
    Asset indices of each sub-portfolio. They partition the universe, so a zero-padded column is the sub-portfolio's real weight on the full asset axis.
    """
    cls::T
end
"""
    sub_portfolio_count(u::FullUniverse, opti)
    sub_portfolio_count(u::ClusterUniverse, opti)

Count the sub-portfolios.

A [`FullUniverse`](@ref) enumerates the inner optimisers, so it has as many sub-portfolios as `opti` holds. A [`ClusterUniverse`](@ref) enumerates the clusters, and one inner optimiser serves them all.

# Arguments

  - `u`: Sub-portfolio enumeration.
  - `opti`: The meta-optimiser's inner optimiser field — a vector of optimisers for [`FullUniverse`](@ref), one optimiser for [`ClusterUniverse`](@ref).

# Returns

  - The number of sub-portfolios.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`sub_portfolio_predict`](@ref)
"""
function sub_portfolio_count(::FullUniverse, opti)
    return length(opti)
end
function sub_portfolio_count(u::ClusterUniverse, ::Any)
    return length(u.cls)
end
"""
    sub_portfolio_predict(u::FullUniverse, opti, i, rd, cv, ex)
    sub_portfolio_predict(u::ClusterUniverse, opti, i, rd, cv, ex)

Cross-validate sub-portfolio `i`.

One [`cross_val_predict`](@ref) call, and the enumeration says which optimiser it runs and on which assets. A [`FullUniverse`](@ref) runs `opti[i]` and passes **no** `cols`: the sub-portfolio is the whole universe, and the arity that takes a precomputed [`OptimisationResult`](@ref) has no `cols` keyword at all, so a colon would be a `MethodError` rather than a no-op. A [`ClusterUniverse`](@ref) runs the one inner optimiser on `u.cls[i]`.

# Arguments

  - `u`: Sub-portfolio enumeration.
  - `opti`: The meta-optimiser's inner optimiser field.
  - `i`: Sub-portfolio index.
  - `rd`: Returns data.
  - `cv`: Cross-validation scheme, already copied for this sub-portfolio.
  - `ex`: FLoops executor controlling parallelism.

# Returns

  - Sub-portfolio `i`'s cross-validation prediction result.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`sub_portfolio_count`](@ref)
  - [`sub_portfolio_cv`](@ref)
  - [`cross_val_predict`](@ref)
"""
function sub_portfolio_predict(::FullUniverse, opti, i::Integer, rd::ReturnsResult, cv,
                               ex::FLoops.Transducers.Executor)
    return cross_val_predict(opti[i], rd, cv; ex = ex)
end
function sub_portfolio_predict(u::ClusterUniverse, opti, i::Integer, rd::ReturnsResult, cv,
                               ex::FLoops.Transducers.Executor)
    return cross_val_predict(opti, rd, cv; cols = u.cls[i], ex = ex)
end
"""
    sub_portfolio_view(u::FullUniverse, x, i::Integer)
    sub_portfolio_view(u::ClusterUniverse, x, i::Integer)

View a full-universe quantity onto sub-portfolio `i`.

The quantities are the ones a sub-portfolio's predicted returns are computed from: the Prior Result and the Fees. A [`FullUniverse`](@ref) sub-portfolio holds the whole universe, so `x` is returned unchanged; a [`ClusterUniverse`](@ref) one restricts it through [`port_opt_view`](@ref).

# Arguments

  - `u`: Sub-portfolio enumeration.
  - `x`: Full-universe quantity, or `nothing`.
  - `i`: Sub-portfolio index.

# Returns

  - `x`, viewed onto sub-portfolio `i`.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`port_opt_view`](@ref)
  - [`predict_outer_returns`](@ref)
"""
function sub_portfolio_view(::FullUniverse, x, ::Integer)
    return x
end
function sub_portfolio_view(u::ClusterUniverse, x, i::Integer)
    return port_opt_view(x, u.cls[i])
end
"""
    sub_portfolio_cv(cv)

Give one sub-portfolio its own copy of the cross-validation scheme.

The sub-portfolios are cross-validated in parallel, so a scheme that draws its splits from a random number generator must not be shared: the generator is mutable state, and two sub-portfolios advancing it at once would neither reproduce nor agree on their folds. A scheme with no `rng` field carries no such state and is passed through.

# Arguments

  - `cv`: Cross-validation scheme.

# Returns

  - `cv`, copied when it holds an `rng`.

# Related

  - [`predict_outer_returns`](@ref)
  - [`cross_val_predict`](@ref)
"""
function sub_portfolio_cv(cv)
    return !hasfield(typeof(cv), :rng) ? cv : copy(cv)
end
"""
    outer_optimisation_finaliser(wb, wf, w_inner, w_outer)

Finalise outer optimisation weights for the NCO algorithm.

Combines inner cluster weights `w_inner` with outer portfolio weights `w_outer`, applying weight bounds `wb` and finalisation algorithm `wf`.

# Arguments

  - `wb`: Weight bounds (optional).
  - `wf`: Weight finaliser.
  - `w_inner`: Inner (within-cluster) weights.
  - `w_outer`: Outer (across-cluster) weights.

# Returns

  - `(retcode, w)`: Final combined portfolio weights and return code. On failure of any
    sub-problem, `retcode` is an `OptimisationFailure` whose `res` is a named tuple
    `(; msg, opti, opto, wb)` carrying the failure summary, the inner optimisation return
    codes, the outer return code, and the weight-finalisation return code (including
    their solver trial diagnostics).

# Related

  - [`NestedClustered`](@ref)
  - [`WeightBounds`](@ref)
"""
function outer_optimisation_finaliser(wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                      resi::VecOpt, rco::OptimisationReturnCode, w::VecNum,
                                      wi::MatNum)
    w = wi * w
    retcode, w = finalise_weight_bounds(wf, wb, w)
    wb_flag = isa(retcode, OptimisationFailure)
    opto_flag = isa(rco, OptimisationFailure)
    resi_retcodes = getproperty.(resi, :retcode)
    resi_flag = any(x -> isa(x, OptimisationFailure), resi_retcodes)
    if resi_flag || opto_flag || wb_flag
        msg = ""
        if resi_flag
            msg *= "opti failed.\n"
        end
        if opto_flag
            msg *= "opto failed.\n"
        end
        if wb_flag
            msg *= "weight bounds finalisation failed.\n"
        end
        retcode = OptimisationFailure(;
                                      res = (; msg = msg, opti = resi_retcodes, opto = rco,
                                             wb = retcode))
    end
    return retcode, w
end
function outer_optimisation_finaliser(wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                      resi::VecOpt, rcos::VecOptRetCode, ws::VecVecNum,
                                      wi::MatNum)
    retcode_w = [outer_optimisation_finaliser(wb, wf, resi, rco, w, wi)
                 for (rco, w) in zip(rcos, ws)]
    return map(x -> x[1], retcode_w), map(x -> x[2], retcode_w)
end
"""
    combination_weights(scale::Nothing, w::VecNum_VecVecNum)
    combination_weights(scale::VecNum, w::VecNum)
    combination_weights(scale::VecNum, w::VecVecNum)

Apply a Combination Weight to a meta-optimiser's outer weights.

The outer optimiser decides how much of each sub-portfolio to hold. A Combination Weight is a *fixed* belief about the same quantity, so the two multiply: sub-portfolio `k` carries the coefficient `sₖ·vₖ`, and the coefficients are rescaled to the total the outer optimiser chose. Schur Complement Hierarchical Risk Parity blends its parameter bundles the same way; this is that shape, placed where a meta-optimiser that owns an *outer optimiser* can use it.

## What the rescale buys

Only the ratios between the entries of a Combination Weight carry meaning (ADR 0053), and the rescale is what makes that true here — a common factor cancels, so the weight needs no normalised form of its own. Three cases are then exactly inert:

  - A **uniform** weight gives back `w`, whatever it sums to.
  - A **lone** sub-portfolio gives back `w`. One element is not a combination, which is ADR 0053's rule.
  - An outer optimiser that chose a total other than one keeps it. Rescaling to one instead would silently overrule a `bgt` of `0.9`.

## Why the outer problem never sees the weight

Scaling the synthetic return columns instead would break the weight on two counts:

  - A uniform weight would stop being neutral. A common rescale of every column moves [`MaximumUtility`](@ref)'s trade-off between return and risk, so the neutral setting would not be neutral.
  - Every [`predict_outer_returns`](@ref) overload would have to re-apply it. A custom one would drop it silently, and the two cross-validation methods already did — so a cross-validated run would disagree with a fold-less one on what the weight means. `cv` is execution control and stays that way.

## Degenerate combinations

The rescale factor is `sum(w) / sum(scale .* w)`, and it does not always exist. A zero denominator — a tilt that cancels a long-short outer allocation — makes it infinite or `NaN`; a zero numerator — a dollar-neutral outer allocation — makes it zero, which would collapse the portfolio. One test covers all three, and the tilted coefficients then stand unrescaled: finite, with their ratios intact. No scalar rescale can hold a zero total while tilting, so there is nothing better to return.

A denominator that is merely *near* zero is not degenerate. The large factor is the answer: the tilt genuinely rebalances a combination that nearly cancels. Should it overflow, [`finalise_weight_bounds`](@ref) reports an [`OptimisationFailure`](@ref) rather than a plausible-looking portfolio.

# Arguments

  - `scale`: Combination Weight, one entry per sub-portfolio, or `nothing`.
  - `w`: Outer optimiser weights, one entry per sub-portfolio; a vector of them on an efficient frontier.

# Returns

  - `w` unchanged when `scale` is `nothing`, otherwise the rescaled coefficients.

# Related

  - [`outer_optimisation_finaliser`](@ref)
  - [`Stacking`](@ref)
"""
function combination_weights(::Nothing, w::VecNum_VecVecNum)
    return w
end
function combination_weights(scale::VecNum, w::VecNum)
    c = scale .* w
    f = sum(w) / sum(c)
    # A zero, infinite or NaN factor is every way the rescale can fail to exist: a
    # zero-total combination, a zero-total outer allocation, and the `0/0` of both at
    # once. The tilt stands unrescaled in each, which is finite and keeps its ratios.
    return iszero(f) || !isfinite(f) ? c : c * f
end
function combination_weights(scale::VecNum, w::VecVecNum)
    return [combination_weights(scale, wi) for wi in w]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Prepares the ReturnsResult for outer optimisation, applying the inner cluster weights `wi` to the returns matrix `rd.B`, and adjusting the independent variable matrices `rd.iv` and `rd.ivpa`, and the feature matrix `rd.Z`, accordingly.

!!! warning

    This function returns `nz` and `Z` in addition to the five values it returned before the feature matrix was collapsed onto the synthetic universe, and it returns them **before** the returns buffer `X`. A custom [`predict_outer_returns`](@ref) overload written against the old tuple therefore breaks loudly — it binds `nz` where it expects `X` and fails on the first write — rather than silently continuing to build an outer [`ReturnsResult`](@ref) with no feature matrix and never learning that it should have one. Appending the pair would not have done this: Julia's destructuring discards trailing values without complaint.

# Arguments

  - `rd`: ReturnsResult containing the returns data.
  - `wi`: Inner weights matrix.

# Returns

  - `nb`: New names for the benchmark returns columns after applying inner weights (if `rd.B` is a matrix).
  - `B`: Adjusted benchmarkreturns matrix after applying inner weights (if `rd.B` is a matrix).
  - `iv`: Adjusted independent variable matrix (if present).
  - `ivpa`: Adjusted independent variable per asset matrix (if present).
  - `nz`: Feature names for the collapsed feature matrix (if present). Unchanged when the feature axis is rectangular; the synthetic asset names when it *is* the asset axis, since the collapse is two-sided there.
  - `Z`: Feature matrix collapsed onto the synthetic assets (if present), see [`collapse_feature_matrix`](@ref).
  - `X`: Buffer for the outer returns matrix.

# Related

  - [`ReturnsResult`](@ref)
  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
  - [`collapse_feature_matrix`](@ref)
  - [`features_are_assets`](@ref)
"""
function prepare_outer_rd(rd::ReturnsResult, wi::MatNum)
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
        # `iv` and `ivpa` are intensive, so they collapse as convex combinations.
        wn = synthetic_asset_weights(wi)
        if iv_flag
            iv = iv * wn
        end
        if ivpa_flag
            ivpa = transpose(wn) * ivpa
        end
    end
    # Features are intensive too. When the feature axis *is* the asset axis the collapse is
    # two-sided, so the synthetic universe keeps a square feature matrix whose names are the
    # synthetic asset names — which is what keeps `features_are_assets` true one level up.
    sq = features_are_assets(rd.nz, rd.nx)
    Z = collapse_feature_matrix(rd.Z, sq, wi)
    nz = sq ? ["_$(i)" for i in 1:size(wi, 2)] : rd.nz
    X = Matrix{eltype(rd.X)}(undef, size(rd.X, 1), size(wi, 2))
    return nb, B, iv, ivpa, nz, Z, X
end
"""
    assert_fold_alignment(predictions) -> VecPredRes

Assert that every sub-portfolio's fold `f` covers the same test period, and return the first sub-portfolio's folds.

A meta-optimiser runs the *same* cross-validation scheme over the *same* returns result for every sub-portfolio, so fold `f` covers the same observations whichever sub-portfolio produced it. [`rebuild_returns_result`](@ref) has relied on that silently since long before it was stated — `reshape(X, :, N)` is only meaningful if the `N` stacked return vectors line up row for row — and the per-fold weight matrix it now assembles is only well defined if it holds. This makes the invariant explicit, and it matters most on the combinatorial path, where each sub-portfolio's `scorer` selects a path independently.

Folds are compared on their timestamps where the returns result has a clock, and on their observation counts where it does not — which is the strongest statement available in each case, and exactly the statement `reshape` needs.

# Arguments

  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects, one per sub-portfolio.

# Returns

  - The first sub-portfolio's per-fold [`PredictionResult`](@ref) objects, which every other sub-portfolio now agrees with.

# Related

  - [`rebuild_returns_result`](@ref)
  - [`fold_row_indices`](@ref)
"""
function assert_fold_alignment(predictions::VecMPredRes)
    pred1 = predictions[1].pred
    nf = length(pred1)
    for (i, prediction) in enumerate(predictions)
        predi = prediction.pred
        @argcheck(length(predi) == nf,
                  DimensionMismatch("every sub-portfolio must run the same number of cross-validation folds, but sub-portfolio 1 has $(nf) and sub-portfolio $(i) has $(length(predi))"))
        for f in 1:nf
            ts1 = pred1[f].rd.ts
            aligned = if isnothing(ts1)
                length(predi[f].rd.X) == length(pred1[f].rd.X)
            else
                predi[f].rd.ts == ts1
            end
            @argcheck(aligned,
                      DimensionMismatch("sub-portfolios 1 and $(i) disagree on the observations fold $(f) covers, so their predictions cannot be laid out side by side. Every sub-portfolio of a meta-optimiser must run the same cross-validation over the same returns result."))
        end
    end
    return pred1
end
"""
    fold_row_indices(rd, pred) -> VecVecInt

Recover the rows of the original returns result each cross-validation fold covers.

The folds do not store their row indices, and they do not need to: [`port_opt_view`](@ref) slices `ts` with the very `test_idx` the fold was built from, so a fold's `rd.ts` *is* its slice of the original clock and [`feature_row_indices`](@ref) matches it straight back. Recovering rather than storing is what keeps this correct on the combinatorial path, where [`sort_predictions!`](@ref) assembles a path's folds in split order rather than chronologically: the timestamps carry whatever order actually happened, while a re-derived split would have to reproduce it.

Only a **time-varying** feature matrix needs this — a static one has no observation axis to slice — which is why the clock is required exactly there and nowhere else.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref), whose `ts` is the clock `rd.Z`'s observation axis is parallel to.
  - `pred`: Per-fold [`PredictionResult`](@ref) objects from one sub-portfolio.

# Returns

  - One row-index vector per fold.

# Related

  - [`feature_row_indices`](@ref)
  - [`rebuild_feature_matrix`](@ref)
  - [`assert_fold_alignment`](@ref)
"""
function fold_row_indices(rd::ReturnsResult, pred::VecPredRes)
    @argcheck(!isnothing(rd.ts),
              IsNothingError("a time-varying feature matrix (Z) has its observation axis parallel to the returns result's timestamps, so collapsing it onto a meta-optimiser's synthetic assets fold by fold needs `ts` to say which observation of Z each fold's observations are. Got ts => nothing. Supply timestamps, or pass a static assets × features Z, which has no observation axis to align."))
    return [feature_row_indices(rd.Z, p.rd.ts, rd.ts) for p in pred]
end
"""
    fold_weight_matrix(predictions, u::FullUniverse, f, na)
    fold_weight_matrix(predictions, u::ClusterUniverse, f, na)

Lay fold `f`'s sub-portfolio weights out as the `assets × sub-portfolios` matrix the outer collapse contracts against.

The sub-portfolio enumeration says which. A [`FullUniverse`](@ref)'s inner optimisers see the whole universe, so their weight vectors are already full length. A [`ClusterUniverse`](@ref)'s see one cluster each, so sub-portfolio `i`'s weights are zero-padded onto `u.cls[i]`. The padding invents nothing: the clusters partition the universe, so a padded column *is* the sub-portfolio's real weight on the full asset axis.

# Arguments

  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects, one per sub-portfolio.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).
  - `f`: Fold index.
  - `na`: Number of real assets.

# Returns

  - An `assets × sub-portfolios` weight matrix.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`rebuild_returns_result`](@ref)
  - [`collapse_feature_matrix`](@ref)
"""
function fold_weight_matrix(predictions::VecMPredRes, ::FullUniverse, f::Integer,
                            na::Integer)
    ws = [prediction.pred[f].res.w for prediction in predictions]
    W = Matrix{mapreduce(eltype, promote_type, ws)}(undef, na, length(ws))
    @inbounds for (i, w) in enumerate(ws)
        W[:, i] = w
    end
    return W
end
function fold_weight_matrix(predictions::VecMPredRes, u::ClusterUniverse, f::Integer,
                            na::Integer)
    ws = [prediction.pred[f].res.w for prediction in predictions]
    W = zeros(mapreduce(eltype, promote_type, ws), na, length(ws))
    @inbounds for (i, (w, cl)) in enumerate(zip(ws, u.cls))
        W[cl, i] = w
    end
    return W
end
"""
    fold_feature_matrix(Z::Nothing, sq, wi, anchor)
    fold_feature_matrix(Z::MatNum, sq, wi, nobs::Integer)
    fold_feature_matrix(Z::Arr3Num, sq, wi, rows::VecInt)

Collapse the original feature matrix onto a fold's synthetic universe, with an observation axis.

The collapse itself is [`collapse_feature_matrix`](@ref)'s matrix arity, applied to the *original*, unsliced feature matrix and the fold's weights. What the two shapes need from the fold differs, and dispatch says which:

  - A **static** `Z` has no observation axis, so it needs only the fold's `nobs`. Its single collapsed matrix is repeated across them, because the collapse is a function of *this fold's* weights and is therefore constant within the fold and different in the next one — which is how a static source becomes genuinely time-varying at the outer problem.
  - A **time-varying** `Z` needs the fold's `rows` in the original clock, and comes back with an observation axis already. This is the only place a fold's absolute rows are needed, and [`fold_row_indices`](@ref) recovers them from the fold's timestamps.

# Arguments

  - `Z`: The original feature matrix, unsliced.
  - `sq`: Whether the feature axis is the asset axis, from [`features_are_assets`](@ref).
  - `wi`: The fold's weights, assets × synthetic assets.
  - `nobs`: Number of observations in the fold, for a static `Z`.
  - `rows`: The rows of the original returns result this fold covers, for a time-varying `Z`.

# Returns

  - `nothing`, or an `observations × synthetic assets × features` array.

# Related

  - [`collapse_feature_matrix`](@ref)
  - [`fold_weight_matrix`](@ref)
  - [`fold_row_indices`](@ref)
  - [`rebuild_feature_matrix`](@ref)
"""
function fold_feature_matrix(::Nothing, ::Bool, ::MatNum, ::Any)
    return nothing
end
function fold_feature_matrix(Z::MatNum, sq::Bool, wi::MatNum, nobs::Integer)
    Zc = collapse_feature_matrix(Z, sq, wi)
    Zf = Array{eltype(Zc)}(undef, nobs, size(Zc, 1), size(Zc, 2))
    @inbounds for t in axes(Zf, 1)
        Zf[t, :, :] = Zc
    end
    return Zf
end
function fold_feature_matrix(Z::Arr3Num, sq::Bool, wi::MatNum, rows::VecInt)
    return collapse_feature_matrix(view(Z, rows, :, :), sq, wi)
end
"""
    fold_feature_anchors(rd, pred)

Give each fold whatever [`fold_feature_matrix`](@ref) needs from it: an observation count for a static feature matrix, absolute rows for a time-varying one.

Scoping the row recovery to the shape that needs it is what keeps the clock requirement narrow. A static feature matrix has no observation axis to align, so it runs on fold sizes alone and never asks the returns result for timestamps.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref).
  - `pred`: Per-fold [`PredictionResult`](@ref) objects from one sub-portfolio.

# Returns

  - One anchor per fold: an `Integer` for a static `Z`, a row-index vector for a time-varying one.

# Related

  - [`fold_feature_matrix`](@ref)
  - [`fold_row_indices`](@ref)
"""
function fold_feature_anchors(rd::ReturnsResult, pred::VecPredRes)
    return isa(rd.Z, Arr3Num) ? fold_row_indices(rd, pred) : [length(p.rd.X) for p in pred]
end
"""
    rebuild_feature_matrix(rd, predictions, u, pred1)

Recompute the outer problem's feature matrix at the cross-validation assembly seam.

Per fold, this makes the *same* [`collapse_feature_matrix`](@ref) call [`prepare_outer_rd`](@ref) makes on the non-cross-validated path — same `sq`, same weight-matrix arity, same original `rd.Z` — and stacks the results down the observation axis. That shared call is the whole point: `cv` is execution control, so toggling it must not change what the outer optimiser measures.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref), whose `nz`/`Z` are read unsliced.
  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects, one per sub-portfolio.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).
  - `pred1`: The first sub-portfolio's folds, from [`assert_fold_alignment`](@ref) — every sub-portfolio agrees with them, so they define the fold boundaries.

# Returns

  - `(nz, Z)`: The synthetic asset names when the feature axis *is* the asset axis, `rd.nz` otherwise; and the stacked `observations × synthetic assets × features` matrix. Both `nothing` when `rd` carries no feature matrix.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`rebuild_returns_result`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`fold_feature_matrix`](@ref)
  - [`fold_feature_anchors`](@ref)
"""
function rebuild_feature_matrix(rd::ReturnsResult, predictions::VecMPredRes,
                                u::SubPortfolioUniverse, pred1::VecPredRes)
    if isnothing(rd.Z)
        return nothing, nothing
    end
    N = length(predictions)
    na = size(rd.X, 2)
    # Identical to `prepare_outer_rd`: square indexes both trailing axes precisely because
    # they are the same axis, so there is no square branch here either.
    sq = features_are_assets(rd.nz, rd.nx)
    Zs = [fold_feature_matrix(rd.Z, sq, fold_weight_matrix(predictions, u, f, na), anchor)
          for (f, anchor) in enumerate(fold_feature_anchors(rd, pred1))]
    Z = Array{eltype(Zs[1])}(undef, sum(x -> size(x, 1), Zs), size(Zs[1], 2),
                             size(Zs[1], 3))
    r = 0
    @inbounds for Zf in Zs
        n = size(Zf, 1)
        Z[(r + 1):(r + n), :, :] = Zf
        r += n
    end
    return sq ? ["_$(i)" for i in 1:N] : rd.nz, Z
end
"""
    rebuild_returns_result(rd, predictions, u)

Reconstruct a returns result from cross-validation predictions.

Combines individual fold predictions from `predictions` into a new `ReturnsResult` corresponding to the original data layout. `u` is the sub-portfolio enumeration — a [`ClusterUniverse`](@ref) for [`NestedClustered`](@ref), a [`FullUniverse`](@ref) for [`Stacking`](@ref) — and it is what says whether a fold's weight vectors need padding onto the full asset axis.

!!! warning

    `u` is **positional and required**, not a keyword with a full-universe default. A default would let a stale two-argument call keep working: correct for [`Stacking`](@ref), and for [`NestedClustered`](@ref) silently writing every cluster's weights to the wrong rows — which yields not an error but a plausible-looking feature matrix. The one configuration that most needs the argument is the one a default would mis-serve, so the break is arranged to be loud.

## The feature matrix

The folds carry none. Instead, the collapse onto the synthetic universe is **recomputed here** from the original, unsliced `rd.Z`, using the same [`collapse_feature_matrix`](@ref) call [`prepare_outer_rd`](@ref) makes on the non-cross-validated path — with `sq` from [`features_are_assets`](@ref) flowing through unchanged, and the per-fold `assets × sub-portfolios` weight matrix assembled from `pred[f].res.w` (see [`rebuild_feature_matrix`](@ref)). The fold results stack down the observation axis, giving the `observations × assets × features` shape the time-varying carrier takes, and the outer optimiser's default [`LastObservation`](@ref) reduces them to the most recent fold's collapse.

The inner solves are untouched: each still sees its own cluster-sliced feature matrix. What the recompute buys is that `cv`, which is execution control, no longer changes what the outer problem measures — and it closes the one intersection where the matrix used to be dropped altogether, a square feature matrix under [`NestedClustered`](@ref), whose folds see cluster-sliced returns and so could never agree on a feature axis to stack.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref).
  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects from cross-validation, one per sub-portfolio.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).

# Returns

  - Rebuilt [`ReturnsResult`](@ref).

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`rebuild_feature_matrix`](@ref)
  - [`assert_fold_alignment`](@ref)
  - [`prepare_outer_rd`](@ref)
"""
function rebuild_returns_result(rd::ReturnsResult, predictions::VecMPredRes,
                                u::SubPortfolioUniverse)
    N = length(predictions)
    nb = rd.nb
    B_flag = !isnothing(rd.B)
    iv_flag = !isnothing(rd.iv)
    ivpa_flag = !isnothing(rd.ivpa)
    rd1 = predictions[1].mrd
    # Copies, not the first prediction's own buffers: the loop below grows these, and
    # appending into `predictions[1].mrd` would leave the predictions mutated — a second
    # call on the same vector would then assemble a result of the wrong height. `ivpa` is
    # per-sub-portfolio rather than per-observation, so it is *wrapped*, not copied: each
    # fold already collapsed it to one number and `MultiPeriodPredictionResult` kept the
    # last fold's, so this builds the length-`N` vector from `N` scalars.
    X = copy(rd1.X)
    B = B_flag ? copy(rd1.B) : nothing
    iv = iv_flag ? copy(rd1.iv) : nothing
    ivpa = ivpa_flag ? [rd1.ivpa] : nothing
    pred1 = assert_fold_alignment(predictions)
    @inbounds for i in 2:N
        rdi = predictions[i].mrd
        append!(X, rdi.X)
        if iv_flag
            append!(iv, rdi.iv)
        end
        if ivpa_flag
            push!(ivpa, rdi.ivpa)
        end
        if B_flag
            append!(B, rdi.B)
        end
    end
    X = reshape(X, :, N)
    # The stacked rows are the fold rows, in fold order. `reshape` above has assumed it
    # since before the feature matrix existed; the recompute below depends on it too.
    nobs = sum(p -> length(p.rd.X), pred1)
    @argcheck(nobs == size(X, 1),
              DimensionMismatch("the stacked sub-portfolio returns must have one row per cross-validated observation, but the folds cover $(nobs) observations and the stacked returns have $(size(X, 1))"))
    nz, Z = rebuild_feature_matrix(rd, predictions, u, pred1)
    if B_flag
        B = reshape(B, :, N)
        nb = ["_b$(i)" for i in 1:N]
    end
    iv = iv_flag ? reshape(iv, :, N) : nothing
    return ReturnsResult(; nx = ["_$i" for i in 1:N], X = X, nf = rd1.nf, F = rd1.F,
                         nb = nb, B = B, ts = rd1.ts, iv = iv, ivpa = ivpa, nz = nz, Z = Z)
end
"""
    sub_portfolio_predictions(::Type{T}, opti, u, rd, cv, ex) where {T}

Cross-validate every sub-portfolio, in parallel, over the same returns result.

One [`sub_portfolio_predict`](@ref) call per sub-portfolio, each on its own copy of the scheme (see [`sub_portfolio_cv`](@ref)). Every sub-portfolio therefore runs the *same* cross-validation over the *same* returns result, which is the invariant [`assert_fold_alignment`](@ref) states one level down.

# Arguments

  - `T`: Element type of the prediction vector — a [`MultiPeriodPredictionResult`](@ref) per sub-portfolio on the non-combinatorial path, a [`PopulationPredictionResult`](@ref) on the combinatorial one.
  - `opti`: The meta-optimiser's inner optimiser field.
  - `u`: Sub-portfolio enumeration, a [`SubPortfolioUniverse`](@ref).
  - `rd`: Returns data.
  - `cv`: Cross-validation scheme.
  - `ex`: FLoops executor controlling parallelism.

# Returns

  - One prediction result per sub-portfolio.

# Related

  - [`predict_outer_returns`](@ref)
  - [`sub_portfolio_predict`](@ref)
  - [`sub_portfolio_cv`](@ref)
"""
function sub_portfolio_predictions(::Type{T}, opti, u::SubPortfolioUniverse,
                                   rd::ReturnsResult, cv,
                                   ex::FLoops.Transducers.Executor) where {T}
    predictions = Vector{T}(undef, sub_portfolio_count(u, opti))
    FLoops.@floop ex for i in eachindex(predictions)
        predictions[i] = sub_portfolio_predict(u, opti, i, rd, sub_portfolio_cv(cv), ex)
    end
    return predictions
end
"""
    predict_outer_returns(cv::Option{<:OptimisationCrossValidation}, opt,
                          u::SubPortfolioUniverse, rd::ReturnsResult,
                          pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                          resi::VecOpt)
    predict_outer_returns(cv::OptimisationCrossValidation{<:NonCombOptCV}, opt,
                          u::SubPortfolioUniverse, rd::ReturnsResult,
                          pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                          resi::VecOpt)
    predict_outer_returns(cv::OptimisationCrossValidation{<:CombinatorialCrossValidation},
                          opt, u::SubPortfolioUniverse, rd::ReturnsResult,
                          pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                          resi::VecOpt)

Predict a meta-optimiser's sub-portfolio returns as the outer problem's [`ReturnsResult`](@ref).

One module serves every meta-optimiser that owns an outer optimiser. What varies between them is the sub-portfolio enumeration, which arrives as a [`SubPortfolioUniverse`](@ref) — [`NestedClustered`](@ref) passes a [`ClusterUniverse`](@ref), [`Stacking`](@ref) a [`FullUniverse`](@ref) — and nothing else here reads the meta-optimiser's own type.

**Dispatch is on `cv`**, which is what chooses the prediction, and a custom cross-validation scheme is therefore an overload on that first argument:

  - Fold-less. The sub-portfolios' own solves are already in `resi`, so each column of the synthetic universe is that solve's net returns, on the Prior Result and Fees viewed onto the sub-portfolio ([`sub_portfolio_view`](@ref)).
  - Non-combinatorial cross-validation. Each sub-portfolio is cross-validated and the folds are stacked ([`rebuild_returns_result`](@ref)), so the outer problem is measured out of sample.
  - Combinatorial cross-validation. As above, then each sub-portfolio's `scorer` selects one path from its population. The default is [`NearestQuantilePrediction`](@ref).

`wi` holds the inner optimisers' own weights, and a Combination Weight is **not** applied to them and must not be applied here: it acts at the combination, after the outer solve, so that an overload cannot drop it and a cross-validated run cannot disagree with a fold-less one on what it means (see [`combination_weights`](@ref)).

!!! warning

    A meta-optimiser calls this with its `cv` **first** and its sub-portfolio enumeration **third**. The two verbs this replaces — `predict_outer_nco_estimator_returns` and `predict_outer_st_estimator_returns` — were the same module written twice, and an overload of either is now a method of nothing. Rewrite it as a method of this verb, dispatching on the scheme rather than on the meta-optimiser's type parameters.

# Arguments

  - `cv`: The meta-optimiser's cross-validation scheme.
  - `opt`: The meta-optimiser. Read for its inner optimisers and its executor.
  - `u`: Sub-portfolio enumeration.
  - `rd`: Returns data.
  - `pr`: Prior Result over the whole universe.
  - `fees`: Fees over the whole universe.
  - `wi`: Inner weights, assets × sub-portfolios.
  - `resi`: The sub-portfolios' own optimisation results.

# Returns

  - The outer problem's [`ReturnsResult`](@ref), one synthetic asset per sub-portfolio.

# Related

  - [`SubPortfolioUniverse`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`rebuild_returns_result`](@ref)
  - [`sub_portfolio_predictions`](@ref)
  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
"""
function predict_outer_returns(::Option{<:OptimisationCrossValidation}, ::Any,
                               u::SubPortfolioUniverse, rd::ReturnsResult,
                               pr::AbstractPriorResult, fees::Option{<:Fees}, wi::MatNum,
                               resi::VecOpt)
    nb, B, iv, ivpa, nz, Z, X = prepare_outer_rd(rd, wi)
    for (i, res) in enumerate(resi)
        X[:, i] = calc_net_returns(res, sub_portfolio_view(u, pr, i),
                                   sub_portfolio_view(u, fees, i))
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         nb = nb, B = B, ts = rd.ts, iv = iv, ivpa = ivpa, nz = nz, Z = Z)
end
function predict_outer_returns(cv::OptimisationCrossValidation{<:NonCombOptCV}, opt,
                               u::SubPortfolioUniverse, rd::ReturnsResult,
                               ::AbstractPriorResult, ::Option{<:Fees}, ::MatNum, ::VecOpt)
    predictions = sub_portfolio_predictions(MultiPeriodPredictionResult, opt.opti, u, rd,
                                            cv.cv, opt.ex)
    return rebuild_returns_result(rd, predictions, u)
end
function predict_outer_returns(cv::OptimisationCrossValidation{<:CombinatorialCrossValidation},
                               opt, u::SubPortfolioUniverse, rd::ReturnsResult,
                               ::AbstractPriorResult, ::Option{<:Fees}, ::MatNum, ::VecOpt)
    predictions = sub_portfolio_predictions(PopulationPredictionResult, opt.opti, u, rd,
                                            cv.cv, opt.ex)
    scorer = isnothing(cv.scorer) ? NearestQuantilePrediction() : cv.scorer
    return rebuild_returns_result(rd, [scorer(prediction) for prediction in predictions], u)
end
