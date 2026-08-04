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
$(DocStringExtensions.TYPEDSIGNATURES)

Prepares the ReturnsResult for outer optimisation, applying the inner cluster weights `wi` to the returns matrix `rd.B`, and adjusting the independent variable matrices `rd.iv` and `rd.ivpa`, and the feature matrix `rd.Z`, accordingly.

!!! warning

    This function returns `nz` and `Z` in addition to the five values it returned before the feature matrix was collapsed onto the synthetic universe, and it returns them **before** the returns buffer `X`. A custom [`predict_outer_nco_estimator_returns`](@ref) or [`predict_outer_st_estimator_returns`](@ref) overload written against the old tuple therefore breaks loudly — it binds `nz` where it expects `X` and fails on the first write — rather than silently continuing to build an outer [`ReturnsResult`](@ref) with no feature matrix and never learning that it should have one. Appending the pair would not have done this: Julia's destructuring discards trailing values without complaint.

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
    fold_weight_matrix(predictions, cls::Nothing, f, na)
    fold_weight_matrix(predictions, cls::VecVecInt, f, na)

Lay fold `f`'s sub-portfolio weights out as the `assets × sub-portfolios` matrix the outer collapse contracts against.

[`Stacking`](@ref)'s inner optimisers see the whole universe, so their weight vectors are already full length and `cls` is `nothing`. [`NestedClustered`](@ref)'s see one cluster each, so sub-portfolio `i`'s weights are zero-padded onto `cls[i]`. The padding invents nothing: `cls` partitions the universe, so a padded column *is* the sub-portfolio's real weight on the full asset axis.

# Arguments

  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects, one per sub-portfolio.
  - `cls`: Asset indices per sub-portfolio, or `nothing` when they are full-universe.
  - `f`: Fold index.
  - `na`: Number of real assets.

# Returns

  - An `assets × sub-portfolios` weight matrix.

# Related

  - [`rebuild_returns_result`](@ref)
  - [`collapse_feature_matrix`](@ref)
"""
function fold_weight_matrix(predictions::VecMPredRes, ::Nothing, f::Integer, na::Integer)
    ws = [prediction.pred[f].res.w for prediction in predictions]
    W = Matrix{mapreduce(eltype, promote_type, ws)}(undef, na, length(ws))
    @inbounds for (i, w) in enumerate(ws)
        W[:, i] = w
    end
    return W
end
function fold_weight_matrix(predictions::VecMPredRes, cls::VecVecInt, f::Integer,
                            na::Integer)
    ws = [prediction.pred[f].res.w for prediction in predictions]
    W = zeros(mapreduce(eltype, promote_type, ws), na, length(ws))
    @inbounds for (i, (w, cl)) in enumerate(zip(ws, cls))
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
    rebuild_feature_matrix(rd, predictions, cls, pred1)

Recompute the outer problem's feature matrix at the cross-validation assembly seam.

Per fold, this makes the *same* [`collapse_feature_matrix`](@ref) call [`prepare_outer_rd`](@ref) makes on the non-cross-validated path — same `sq`, same weight-matrix arity, same original `rd.Z` — and stacks the results down the observation axis. That shared call is the whole point: `cv` is execution control, so toggling it must not change what the outer optimiser measures.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref), whose `nz`/`Z` are read unsliced.
  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects, one per sub-portfolio.
  - `cls`: Asset indices per sub-portfolio, or `nothing` when they are full-universe.
  - `pred1`: The first sub-portfolio's folds, from [`assert_fold_alignment`](@ref) — every sub-portfolio agrees with them, so they define the fold boundaries.

# Returns

  - `(nz, Z)`: The synthetic asset names when the feature axis *is* the asset axis, `rd.nz` otherwise; and the stacked `observations × synthetic assets × features` matrix. Both `nothing` when `rd` carries no feature matrix.

# Related

  - [`rebuild_returns_result`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`fold_feature_matrix`](@ref)
  - [`fold_feature_anchors`](@ref)
"""
function rebuild_feature_matrix(rd::ReturnsResult, predictions::VecMPredRes,
                                cls::Option{<:VecVecInt}, pred1::VecPredRes)
    if isnothing(rd.Z)
        return nothing, nothing
    end
    N = length(predictions)
    na = size(rd.X, 2)
    # Identical to `prepare_outer_rd`: square indexes both trailing axes precisely because
    # they are the same axis, so there is no square branch here either.
    sq = features_are_assets(rd.nz, rd.nx)
    Zs = [fold_feature_matrix(rd.Z, sq, fold_weight_matrix(predictions, cls, f, na),
                              anchor)
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
    rebuild_returns_result(rd, predictions, cls)

Reconstruct a returns result from cross-validation predictions.

Combines individual fold predictions from `predictions` into a new `ReturnsResult` corresponding to the original data layout. `cls` is the asset index set of each sub-portfolio — [`NestedClustered`](@ref)'s clusters — or `nothing` when the sub-portfolios are full-universe, as [`Stacking`](@ref)'s are.

!!! warning

    `cls` is **positional**, not a keyword with a full-universe default. A default would let a stale two-argument call keep working: correct for [`Stacking`](@ref), and for [`NestedClustered`](@ref) silently writing every cluster's weights to the wrong rows — which yields not an error but a plausible-looking feature matrix. The one configuration that most needs the argument is the one a default would mis-serve, so the break is arranged to be loud.

## The feature matrix

The folds carry none. Instead, the collapse onto the synthetic universe is **recomputed here** from the original, unsliced `rd.Z`, using the same [`collapse_feature_matrix`](@ref) call [`prepare_outer_rd`](@ref) makes on the non-cross-validated path — with `sq` from [`features_are_assets`](@ref) flowing through unchanged, and the per-fold `assets × sub-portfolios` weight matrix assembled from `pred[f].res.w` (see [`rebuild_feature_matrix`](@ref)). The fold results stack down the observation axis, giving the `observations × assets × features` shape the time-varying carrier takes, and the outer optimiser's default [`LastObservation`](@ref) reduces them to the most recent fold's collapse.

The inner solves are untouched: each still sees its own cluster-sliced feature matrix. What the recompute buys is that `cv`, which is execution control, no longer changes what the outer problem measures — and it closes the one intersection where the matrix used to be dropped altogether, a square feature matrix under [`NestedClustered`](@ref), whose folds see cluster-sliced returns and so could never agree on a feature axis to stack.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref).
  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects from cross-validation, one per sub-portfolio.
  - `cls`: Asset indices per sub-portfolio, or `nothing` when they are full-universe.

# Returns

  - Rebuilt [`ReturnsResult`](@ref).

# Related

  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`rebuild_feature_matrix`](@ref)
  - [`assert_fold_alignment`](@ref)
  - [`prepare_outer_rd`](@ref)
"""
function rebuild_returns_result(rd::ReturnsResult, predictions::VecMPredRes,
                                cls::Option{<:VecVecInt})
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
    nz, Z = rebuild_feature_matrix(rd, predictions, cls, pred1)
    if B_flag
        B = reshape(B, :, N)
        nb = ["_b$(i)" for i in 1:N]
    end
    iv = iv_flag ? reshape(iv, :, N) : nothing
    return ReturnsResult(; nx = ["_$i" for i in 1:N], X = X, nf = rd1.nf, F = rd1.F,
                         nb = nb, B = B, ts = rd1.ts, iv = iv, ivpa = ivpa, nz = nz, Z = Z)
end
