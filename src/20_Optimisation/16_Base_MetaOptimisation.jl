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
    rebuild_returns_result(rd, predictions)

Reconstruct a returns result from cross-validation predictions.

Combines individual fold predictions from `predictions` into a new `ReturnsResult` corresponding to the original data layout.

## The feature matrix

Each prediction has already had its feature matrix collapsed onto its own synthetic asset, fold by fold, and stacked down the observation axis (see [`reconstruct_rd`](@ref) and [`MultiPeriodPredictionResult`](@ref)). All that is left here is to lay the `N` sub-portfolios out along the asset axis, giving the `observations × assets × features` shape the time-varying carrier takes — with the same row count as `X`, since `X` is stacked the same way. The outer optimiser's default [`LastObservation`](@ref) then reduces it to the most recent fold's collapse.

A feature matrix survives only when every sub-portfolio collapsed onto the *same* feature axis. That holds whenever the feature axis is rectangular, since it is never subselected by assets. It does not hold for a square feature matrix under [`NestedClustered`](@ref), whose folds see cluster-sliced returns: each cluster's feature axis is its own asset subset, so there is nothing to stack them against and the feature matrix is dropped. An outer estimator that then asks for features gets the same error it would get had none been supplied, rather than a matrix assembled from mismatched axes.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref).
  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects from cross-validation.

# Returns

  - Rebuilt [`ReturnsResult`](@ref).

# Related

  - [`NestedClustered`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`reconstruct_rd`](@ref)
  - [`collapse_feature_matrix`](@ref)
"""
function rebuild_returns_result(rd::ReturnsResult, predictions::VecMPredRes)
    N = length(predictions)
    nb = rd.nb
    B_flag = !isnothing(rd.B)
    iv_flag = !isnothing(rd.iv)
    ivpa_flag = !isnothing(rd.ivpa)
    rd1 = predictions[1].mrd
    X = rd1.X
    B = B_flag ? rd1.B : nothing
    iv = rd1.iv
    ivpa = ivpa_flag ? [rd1.ivpa] : nothing
    nz = rd1.nz
    Z_flag = !isnothing(rd1.Z) && all(x -> x.mrd.nz == nz, predictions)
    Z = nothing
    if Z_flag
        Z = Array{eltype(rd1.Z)}(undef, size(rd1.Z, 1), N, size(rd1.Z, 2))
        Z[:, 1, :] = rd1.Z
    else
        nz = nothing
    end
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
        if Z_flag
            Z[:, i, :] = rdi.Z
        end
    end
    X = reshape(X, :, N)
    if B_flag
        B = reshape(B, :, N)
        nb = ["_b$(i)" for i in 1:N]
    end
    iv = iv_flag ? reshape(iv, :, N) : nothing
    return ReturnsResult(; nx = ["_$i" for i in 1:N], X = X, nf = rd1.nf, F = rd1.F,
                         nb = nb, B = B, ts = rd1.ts, iv = iv, ivpa = ivpa, nz = nz, Z = Z)
end
