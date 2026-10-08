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

 1. Return `st` unchanged when the state carries no history, as [`cross_sectional_carries_history`](@ref) answers.
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
