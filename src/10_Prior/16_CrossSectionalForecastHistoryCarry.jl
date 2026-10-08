"""
    cross_sectional_forecast_refits(rfe::AbstractReturnForecastEstimator)
    cross_sectional_forecast_refits(rfe::Union{FixedWeightedReturnForecast,
                                               ExpWeightedReturnForecast})

Answers whether the history of a Return Forecast Estimator comes from a refit at each observation.

[`FixedWeightedReturnForecast`](@ref) and [`ExpWeightedReturnForecast`](@ref) publish their history in the Result of one fit, so the call with no data of the carry fold reads it there and the state carries no row. Every other member answers `true`. A member that publishes a history and answers `true` costs a fit at each step, and the call with no data still reads the history of its Result.

# Arguments

  - `rfe`: Return Forecast Estimator.

# Returns

  - `refits::Bool`: `true` when the history comes from a refit.

# Related

  - [`cross_sectional_carries_history`](@ref)
  - [`forecast_history`](@ref)
"""
function cross_sectional_forecast_refits(::AbstractReturnForecastEstimator)
    return true
end
function cross_sectional_forecast_refits(::Union{FixedWeightedReturnForecast,
                                                 ExpWeightedReturnForecast})
    return false
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Answers whether the carry fold of a Cross-Sectional Factor Prior carries the Return Forecast history.

It does when a slot of the prior answers `true` to [`reads_forecast_history`](@ref), and the history of its Return Forecast comes from a refit, as [`cross_sectional_forecast_refits`](@ref) answers.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.

# Returns

  - `carries::Bool`: `true` when the state carries the history.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`cross_sectional_carry_history`](@ref)
"""
function cross_sectional_carries_history(pe::CrossSectionalFactorPrior)::Bool
    return (reads_forecast_history(pe.lambda) || reads_forecast_history(pe.c)) &&
           cross_sectional_forecast_refits(pe.rfe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the Return Forecast history that a Cross-Sectional Factor Prior hands a slot that reads it.

The batch fit and the call with no data of the carry fold both take the history here, from the Result `rf` that the prior fitted on the whole sample. A Result that carries a history gives it, so the member is fitted once. Otherwise the batch fit refits the member at each observation with [`forecast_history_refit`](@ref), and the carry fold appends the forecast of the last observation to the rows `H` that it carried.

# Arguments

  - `rf`: The Return Forecast Result of the whole sample.
  - `rfe`: Return Forecast Estimator.
  - $(arg_dict[:rd]) It is the returns data the member was fitted on.
  - `csfm`: The factor-model block the member was fitted on.
  - `H`: The rows that the carry fold carries, at every observation of the block but the last, or `nothing`.

# Validation

  - The rules of [`forecast_history_refit`](@ref) when `H` is `nothing` and `rf` carries no history.

# Returns

  - `hist::MatNum`: Return Forecast history, `observations × assets`, on the rows of the block.

# Related

  - [`forecast_history`](@ref)
  - [`cross_sectional_return_forecast`](@ref)
  - [`cross_sectional_carry_history`](@ref)
"""
function cross_sectional_forecast_history(rf::AbstractReturnForecastResult,
                                          rfe::AbstractReturnForecastEstimator,
                                          rd::ReturnsResult,
                                          csfm::CrossSectionalFactorModel,
                                          H::Option{<:MatNum})::MatNum
    hist = rf.hist
    if !isnothing(hist)
        return hist
    end
    return if isnothing(H)
        forecast_history_refit(rfe, rd, csfm, rf.mu, 1)
    else
        vcat(H, transpose(rf.mu))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Brings the Return Forecast history of the carry fold of a Cross-Sectional Factor Prior up to its fitted observations.

The row of a block observation `t` is the forecast of the member fitted on the returns data through that observation and on the block rows `1:t`, as [`forecast_history_refit`](@ref) fits it. No row reads a later observation, so a step that fits the new observations alone keeps the rows it carried and fits one row for each new observation. The state keeps no row for the last observation: the call with no data fits the member there anyway, and appends that forecast with [`cross_sectional_forecast_history`](@ref).

# Algorithm

 1. Return `st` unchanged when the state carries no history, as [`cross_sectional_carries_history`](@ref) answers. When the member folds, as [`folds_forecast_rows`](@ref) answers, its rows at every fitted observation but the last are the history, so return the state with them. The row of an observation is the forecast that the member publishes there, which is the forecast of a fit through that observation.
 2. Keep the carried rows when `keep` is `true`, and none otherwise.
 3. Build the block of the fitted observations from the state, with [`cross_sectional_carry_append`](@ref) and the returns data of [`cross_sectional_carry_forecast_returns`](@ref). The member reads no idiosyncratic covariance, so the block carries none. Take the member of the Orthogonal Forecast Fit `pe.ofit` with [`orthogonal_forecast_member`](@ref), the member whose history the batch fit reads.
 4. Fit that member at each block row after the kept rows and before the last, on the returns data and the block that [`forecast_history_block`](@ref) cuts to that row.
 5. Append the new rows to the kept rows.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state after the step, after its first fit.
  - `keep`: Whether the step fitted the new observations alone, so that the carried rows stay.

# Validation

  - The rules of [`return_forecast`](@ref).

# Returns

  - `st::CrossSectionalCarryState`: The state with its history brought up to its fitted observations.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
  - [`forecast_history_refit`](@ref)
"""
function cross_sectional_carry_history(pe::CrossSectionalFactorPrior,
                                       st::CrossSectionalCarryState, keep::Bool)
    if !cross_sectional_carries_history(pe)
        return st
    end
    # A member that folds publishes at each fitted observation the forecast that a fit through
    # that observation gives, so its rows are the history.
    if folds_forecast_rows(pe.rfe)
        return cross_sectional_carry_with(st, (; hist = st.fh[1:(end - 1), :]))
    end
    H = keep ? st.hist : nothing
    rd = cross_sectional_carry_forecast_returns(pe, st)
    (; r, ca) = cross_sectional_carry_append(pe, st)
    csfm = CrossSectionalFactorModel(; M = ca.Ms[end, :, :],
                                     b = zeros(real(eltype(st.vs)), size(st.X, 2)),
                                     csr = st.csr, Ms = ca.Ms, vs = st.vs, rw = st.W,
                                     bw = view(st.bw, r, :), nf = ca.nf, fam = ca.fam,
                                     lag = pe.lag, fx = ca.fx)
    rows = return_forecast_rows(rd, csfm)
    # The rows are the history of the member that the batch fit reads under its Orthogonal
    # Forecast Fit, so the carried rows and the appended one come from the same member.
    rfo = orthogonal_forecast_member(pe.ofit, pe.rfe, csfm)
    k = isnothing(H) ? 0 : size(H, 1)
    # The fit of `forecast_history_refit` at block row `tb`. A fit holds two block rows at
    # least, and a step after it brings one new row at least, so a row is always new.
    new = [return_forecast(rfo, port_opt_view(rd, 1:rows[tb], :),
                           forecast_history_block(csfm, tb)).mu
           for tb in (k + 1):(length(rows) - 1)]
    R = permutedims(reduce(hcat, new))
    return cross_sectional_carry_with(st, (; hist = cross_sectional_fold_append(H, R)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Brings the standardised idiosyncratic returns of the carry fold of a Cross-Sectional Factor Prior up to its fitted observations.

The standardised return of an observation reads the residual, the variance and the active mask of that observation alone, and the fill reads the other assets of the same observation. A step that fits the new observations alone changes no past residual and no past variance, so the state keeps its rows and appends the rows of the new observations. A step that fits every observation again can change every row, so it makes every row again. The call with no data then reads the rows as they are, and its cost does not grow with the history.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state after the fit of the step.
  - `stepped`: `true` when the step fitted the new observations alone, `false` when it fitted every observation.

# Returns

  - `st::CrossSectionalCarryState`: The state with `S`, and `Sc` when the threshold `th` of the prior is not zero, at every fitted observation. A state with no fit is returned as it is.

# Related

  - [`cross_sectional_standardised_residuals`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_assemble`](@ref)
"""
function cross_sectional_carry_standardised(pe::CrossSectionalFactorPrior,
                                            st::CrossSectionalCarryState, stepped::Bool)
    if isnothing(st.csr)
        return st
    end
    eps = st.csr.eps
    k = stepped && !isnothing(st.S) ? size(st.S, 1) : 0
    rn = (k + 1):size(eps, 1)
    # The fitted rows sit after the first `lag` rows of the histories.
    E = view(eps, rn, :)
    V = view(st.vs, rn, :)
    A = view(st.amsk, pe.lag .+ rn, :)
    S = cross_sectional_standardised_residuals(E, V, A)
    Sc = if iszero(pe.th)
        nothing
    else
        cross_sectional_standardised_residuals(E, V, A; filled = false)
    end
    # A step that fits every observation again makes every row again.
    S0, Sc0 = iszero(k) ? (nothing, nothing) : (st.S, st.Sc)
    return cross_sectional_carry_with(st,
                                      (; S = cross_sectional_fold_append(S0, S),
                                       Sc = cross_sectional_fold_append(Sc0, Sc)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the returns data that the Return Forecast of the carry fold of a Cross-Sectional Factor Prior scores its Descriptors over.

The batch fit gives the forecast the returns data that the estimated members read: the returns with the benchmark weights on the panel, and under observed factors the returns net of them. The benchmark weights of a row read that row alone, so the function builds them over the rows that it is given.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `win`: Panel rows, with their returns, Asset Panel and Exogenous Series.
  - `der`: The derived series of the rows of `win`, `(; Xn, Xl)`, or `nothing` without an observed factor.

# Returns

  - `rd::ReturnsResult`: The returns data of the forecast over the rows of `win`.

# Related

  - [`cross_sectional_carry_forecast_returns`](@ref)
  - [`cross_sectional_carry_scores`](@ref)
  - [`cross_sectional_benchmark_stage`](@ref)
"""
function cross_sectional_forecast_window(pe::CrossSectionalFactorPrior, win::ReturnsResult,
                                         der::Option{<:NamedTuple})
    (; rdb) = cross_sectional_benchmark_stage(pe, win.X, nothing, win.pnl; ne = win.ne,
                                              E = win.E)
    return if isnothing(der)
        rdb
    else
        ReturnsResult(; nx = rdb.nx, X = der.Xl, ne = rdb.ne, E = rdb.E, pnl = rdb.pnl)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends the Descriptor scores of the new observations of a step to the carry fold of a Cross-Sectional Factor Prior.

A Return Forecast that computes its history one observation at a time, as [`folds_forecast_rows`](@ref) answers, reads its scores at each fitted observation. A score reads the last [`lookback`](@ref) panel rows of its observation and no factor-model block, so a step computes the scores of its new observations alone, from the rows that it carries. A fit of every observation then reads the scores that the state carries, and no panel row before the carried ones.

# Algorithm

 1. Return `st` unchanged when the forecast does not fold, or when the step brings no observation after the warm-up.
 2. Keep the last `lookback(pe.rfe) + m - 1` rows of `win` and of its derived series, and build the returns data of the forecast over them with [`cross_sectional_forecast_window`](@ref).
 3. Score the Descriptors of the forecast over those rows with [`descriptor_panel_scores`](@ref), and keep the last `m` rows.
 4. Append them to the scores that the state carries with [`cross_sectional_fold_append`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, with the rows of the step appended to its histories.
  - `win`: The carried rows followed by the rows of the step.
  - `rows`: The rows of the step, as [`cross_sectional_fold_rows`](@ref) returns them. Its `Ms` holds the `m` new observations after the warm-up, the last rows of `win`, and its `der` the derived series of every row of `win`, or `nothing` without an observed factor.

# Returns

  - `st::CrossSectionalCarryState`: The state with the scores of the new observations appended.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_carry_forecast`](@ref)
"""
function cross_sectional_carry_scores(pe::CrossSectionalFactorPrior,
                                      st::CrossSectionalCarryState, win::ReturnsResult,
                                      rows::NamedTuple)
    m = size(rows.Ms, 1)
    if !folds_forecast_rows(pe.rfe) || iszero(m)
        return st
    end
    der = rows.der
    (; rdi, k) = cross_sectional_exposure_rows(pe.rfe, win, m)
    T = size(win.X, 1)
    dk = isnothing(der) ? nothing : (; Xl = view(der.Xl, (T - k + 1):T, :))
    P = descriptor_panel_scores(pe.rfe.scores, cross_sectional_forecast_window(pe, rdi, dk))
    Pn = map(A -> return_forecast_cut(A, (k - m + 1):k), P)
    return cross_sectional_carry_with(st,
                                      (;
                                       fsc = if isnothing(st.fsc)
                                           Pn
                                       else
                                           # The bound stops JET from pairing an untyped
                                           # `st.fsc` with the `map` method of another package.
                                           map(cross_sectional_fold_append,
                                               st.fsc::NamedTuple, Pn)
                                       end))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Brings the Return Forecast rows of the carry fold of a Cross-Sectional Factor Prior up to its fitted observations.

A Return Forecast that computes its history one observation at a time, as [`folds_forecast_rows`](@ref) answers, computes the rows of its new observations from its fold state and from the scores and the block at those observations and at the [`forecast_target_gap`](@ref) observations before them. A step that fits the new observations alone changes no past row of the block, so the state keeps its rows and its fold state, and computes the rows of the new observations. A step that fits every observation again can change every row of the block, so it computes every row again from an empty fold state, from the scores that the state carries.

# Algorithm

 1. Return `st` unchanged when the forecast does not fold, or before the first fit.
 2. Keep the carried rows and the fold state when `stepped` is `true`, and none otherwise. Take the fitted observations after the kept rows, and the last `forecast_target_gap(pe.rfe)` kept rows before them, or every kept row when there are fewer.
 3. Build the block of those observations from the state, with [`cross_sectional_carry_append`](@ref). Take the member of the Orthogonal Forecast Fit `pe.ofit` with [`orthogonal_forecast_member`](@ref).
 4. Compute the rows of those observations and the new fold state with [`orthogonal_forecast_step`](@ref), from a copy of the scores that the state carries at the rows of [`cross_sectional_forecast_rows`](@ref) and from the kept fold state. A member that trains on the rows before the block reads them too.
 5. Append the rows of the new observations to the kept rows, and keep the new fold state.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state after the fit of the step.
  - `stepped`: `true` when the step fitted the new observations alone, `false` when it fitted every observation.

# Validation

  - The rules of [`return_forecast_step`](@ref).

# Returns

  - `st::CrossSectionalCarryState`: The state with its Return Forecast rows at every fitted observation, and the fold state of the forecast after them.

# Related

  - [`cross_sectional_carry_scores`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
  - [`return_forecast_result`](@ref)
"""
function cross_sectional_carry_forecast(pe::CrossSectionalFactorPrior,
                                        st::CrossSectionalCarryState, stepped::Bool)
    if !folds_forecast_rows(pe.rfe) || isnothing(st.csr)
        return st
    end
    k = stepped && !isnothing(st.fh) ? size(st.fh, 1) : 0
    # The targets of the last `g` kept rows mature at this step, so the step reads them again.
    g = forecast_target_gap(pe.rfe)
    j = (k + 1 - min(k, g)):size(st.vs, 1)
    (; r, ca) = cross_sectional_carry_append(pe, st)
    csr = st.csr
    csfm = CrossSectionalFactorModel(; M = ca.Ms[end, :, :],
                                     b = zeros(real(eltype(st.vs)), size(st.X, 2)),
                                     csr = CrossSectionalRegression(; f = csr.f[j, :],
                                                                    eps = csr.eps[j, :],
                                                                    n = csr.n[j],
                                                                    h1 = nothing_scalar_array_getindex_odd_order(csr.h1,
                                                                                                                 j,
                                                                                                                 :)),
                                     Ms = ca.Ms[j, :, :], vs = st.vs[j, :], rw = st.W[j, :],
                                     bw = st.bw[r[j], :], nf = ca.nf, fam = ca.fam,
                                     lag = pe.lag,
                                     fx = nothing_scalar_array_getindex_odd_order(ca.fx, j,
                                                                                  :))
    rfo = orthogonal_forecast_member(pe.ofit, pe.rfe, csfm)
    # The scores sit on the histories after the warm-up, and the fitted rows after the first
    # `lag` of them. The cut copies, so the step can neutralise the copy in place.
    # A fit of every observation makes every row again, from an empty fold state.
    H0, fs0 = iszero(k) ? (nothing, nothing) : (st.fh, st.fst)
    i = cross_sectional_forecast_rows(rfo, r, k, g)
    (; hist, fs) = orthogonal_forecast_step(pe.ofit, rfo,
                                            map(A -> return_forecast_cut(A, i), st.fsc),
                                            csfm, fs0, pe.cre)
    # The new rows are the last rows of the history of the step.
    H = view(hist, (size(hist, 1) - length(r) + k + 1):size(hist, 1), :)
    return cross_sectional_carry_with(st,
                                      (; fh = cross_sectional_fold_append(H0, H), fst = fs))
end
"""
    cross_sectional_forecast_rows(rfe, r::AbstractUnitRange, k::Integer, g::Integer)
    cross_sectional_forecast_rows(rfe::TargetReturnForecast, r::AbstractUnitRange,
                                  k::Integer, g::Integer)

Returns the rows of the histories after the warm-up that a step of the carry fold of a Cross-Sectional Factor Prior hands its Return Forecast.

The step hands the member its new observations, and before them the observations whose target matures at the step: the last `g` observations that the member read, or all of them when it read fewer.

# Algorithm

The method that Julia selects is the algorithm.

 1. A member that reads the block alone: the fitted observations after the first `k`, and the last `g` of the first `k`, as rows of the histories.
 2. [`TargetReturnForecast`](@ref) under `whole_history`: the member also trains on a row before the block whose forward window reaches into the block, so its observations are the rows of the histories. Return the rows after the first `k` fitted observations, and the `g` rows before them, or every row before them when there are fewer. Without `whole_history`, method 1.

# Arguments

  - `rfe`: The Return Forecast Estimator that the step fits.
  - `r`: The fitted observations, as rows of the histories after the warm-up.
  - `k`: Number of fitted observations that the state keeps.
  - `g`: The observations that a target takes to mature, as [`forecast_target_gap`](@ref) answers.

# Returns

  - `rows::AbstractUnitRange`: Rows of the histories after the warm-up.

# Related

  - [`cross_sectional_carry_forecast`](@ref)
  - [`return_forecast_step`](@ref)
"""
function cross_sectional_forecast_rows(::Any, r::AbstractUnitRange, k::Integer, g::Integer)
    return r[(k + 1 - min(k, g)):end]
end
function cross_sectional_forecast_rows(rfe::TargetReturnForecast, r::AbstractUnitRange,
                                       k::Integer, g::Integer)
    return rfe.whole_history ? (max(1, r[k + 1] - g):last(r)) : r[(k + 1 - min(k, g)):end]
end
"""
    orthogonal_forecast_step(ofit::AbstractOrthogonalForecastFit, rfe, P::NamedTuple,
                             csfm::CrossSectionalFactorModel, fs,
                             cre::AbstractCrossSectionalRegressionEstimator)
    orthogonal_forecast_step(ofit::OrthogonalPartCalibration, rfe::TargetReturnForecast,
                             P::NamedTuple, csfm::CrossSectionalFactorModel, fs,
                             cre::AbstractCrossSectionalRegressionEstimator)

Runs a step of the Return Forecast of the carry fold of a Cross-Sectional Factor Prior under its Orthogonal Forecast Fit, as [`orthogonal_forecast_result`](@ref) fits it in the batch fit.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`OrthogonalPartCalibration`](@ref) with a [`TargetReturnForecast`](@ref): call [`return_forecast_step`](@ref) with `cre`, so the fold state also carries the regression of `κ⊥`.
 2. Any other pair: call [`return_forecast_step`](@ref) with no `cre`.

# Arguments

  - `ofit`: The Orthogonal Forecast Fit of the prior.
  - `rfe`: The Return Forecast Estimator that the step fits.
  - `P`: The scores, the weights and the group labels of the rows of the step.
  - `csfm`: The factor-model block of the rows of the step.
  - `fs`: The fold state that the state carries, or `nothing`.
  - `cre`: Cross-Sectional Regression Estimator of the prior.

# Returns

  - `(; hist, fs)::NamedTuple`: The answer of [`return_forecast_step`](@ref).

# Related

  - [`cross_sectional_carry_forecast`](@ref)
  - [`orthogonal_forecast_result`](@ref)
"""
function orthogonal_forecast_step(::AbstractOrthogonalForecastFit, rfe, P::NamedTuple,
                                  csfm::CrossSectionalFactorModel, fs,
                                  ::AbstractCrossSectionalRegressionEstimator)
    return return_forecast_step(rfe, P, csfm, fs)
end
function orthogonal_forecast_step(::OrthogonalPartCalibration, rfe::TargetReturnForecast,
                                  P::NamedTuple, csfm::CrossSectionalFactorModel, fs,
                                  cre::AbstractCrossSectionalRegressionEstimator)
    return return_forecast_step(rfe, P, csfm, fs, cre)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Brings the outputs of the carry fold of a Cross-Sectional Factor Prior that read the fit up to its fitted observations.

# Algorithm

 1. Bring the Return Forecast rows up with [`cross_sectional_carry_forecast`](@ref).
 2. Bring the Return Forecast history that a slot reads up with [`cross_sectional_carry_history`](@ref). It reads the rows of step 1 when the member folds.
 3. Bring the standardised idiosyncratic returns up with [`cross_sectional_carry_standardised`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state after the fit of the step.
  - `stepped`: `true` when the step fitted the new observations alone, `false` when it fitted every observation.

# Returns

  - `st::CrossSectionalCarryState`: The state with the three outputs at every fitted observation.

# Related

  - [`cross_sectional_carry_fold`](@ref)
"""
function cross_sectional_carry_outputs(pe::CrossSectionalFactorPrior,
                                       st::CrossSectionalCarryState, stepped::Bool)
    st = cross_sectional_carry_forecast(pe, st, stepped)
    st = cross_sectional_carry_history(pe, st, stepped)
    return cross_sectional_carry_standardised(pe, st, stepped)
end
