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
 3. Build the block of the fitted observations from the state, with [`cross_sectional_carry_append`](@ref) and the returns data of [`cross_sectional_carry_forecast_returns`](@ref). The member reads no idiosyncratic covariance, so the block carries none.
 4. Fit the member at each block row after the kept rows and before the last, on the returns data and the block that [`forecast_history_block`](@ref) cuts to that row.
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
                                     bw = st.bw[r, :], nf = ca.nf, fam = ca.fam,
                                     lag = pe.lag)
    rows = return_forecast_rows(rd, csfm)
    k = isnothing(H) ? 0 : size(H, 1)
    # The fit of `forecast_history_refit` at block row `tb`. A fit holds two block rows at
    # least, and a step after it brings one new row at least, so a row is always new.
    new = [return_forecast(pe.rfe, port_opt_view(rd, 1:rows[tb], :),
                           forecast_history_block(csfm, tb)).mu
           for tb in (k + 1):(length(rows) - 1)]
    R = permutedims(reduce(hcat, new))
    return cross_sectional_carry_with(st, (; hist = isnothing(H) ? R : vcat(H, R)))
end
