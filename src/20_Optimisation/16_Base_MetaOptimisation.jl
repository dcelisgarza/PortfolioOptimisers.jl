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

  - Final combined portfolio weights.

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
    resi_flag = any(x -> isa(x, OptimisationFailure), getproperty.(resi, :retcode))
    if resi_flag || opto_flag || wb_flag
        msg = ""
        if resi_flag
            msg *= "opti failed.\n"
        end
        if opto_flag
            msg *= "opto failed.\n"
        end
        if wb_flag
            msg *= "Full optimisation failed.\n"
        end
        retcode = OptimisationFailure(; res = msg)
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

Prepares the ReturnsResult for outer optimisation, applying the inner cluster weights `wi` to the returns matrix `rd.B`, and adjusting the independent variable matrices `rd.iv` and `rd.ivpa` accordingly.

# Arguments

  - `rd`: ReturnsResult containing the returns data.
  - `wi`: Inner weights matrix.

# Returns

  - `nb`: New names for the benchmark returns columns after applying inner weights (if `rd.B` is a matrix).
  - `B`: Adjusted benchmarkreturns matrix after applying inner weights (if `rd.B` is a matrix).
  - `iv`: Adjusted independent variable matrix (if present).
  - `ivpa`: Adjusted independent variable per asset matrix (if present).
  - `X`: Buffer for the outer returns matrix.

# Related

  - [`ReturnsResult`](@ref)
  - [`NestedClustered`](@ref)
  - [`Stacking`](@ref)
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
        wi = abs.(wi)
        if iv_flag
            iv = iv * wi
        end
        if ivpa_flag
            ivpa = transpose(wi) * ivpa
        end
    end
    X = Matrix{eltype(rd.X)}(undef, size(rd.X, 1), size(wi, 2))
    return nb, B, iv, ivpa, X
end
"""
    rebuild_returns_result(rd, predictions)

Reconstruct a returns result from cross-validation predictions.

Combines individual fold predictions from `predictions` into a new `ReturnsResult` corresponding to the original data layout.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref).
  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects from cross-validation.

# Returns

  - Rebuilt [`ReturnsResult`](@ref).

# Related

  - [`NestedClustered`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
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
    if B_flag
        B = reshape(B, :, N)
        nb = ["_b$(i)" for i in 1:N]
    end
    iv = iv_flag ? reshape(iv, :, N) : nothing
    return ReturnsResult(; nx = ["_$i" for i in 1:N], X = X, nf = rd1.nf, F = rd1.F,
                         nb = nb, B = B, ts = rd1.ts, iv = iv, ivpa = ivpa)
end
