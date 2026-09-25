"""
    forecast_history_block(csfm::CrossSectionalFactorModel,
                           tb::Integer) -> CrossSectionalFactorModel

Return the factor-model block truncated to its first `tb` observations.

[`forecast_history_refit`](@ref) refits a member that computes no history at each row of a grid, and the fit at a row must read nothing after that row. This function gives the block that such a fit reads. It cuts every history of the block to the rows `1:tb`, and it sets the loadings `M` to the last slice of the cut exposure history.

The function keeps the per-asset summaries `b` and `esigma` unchanged. They describe the assets and not the observations, and no Return Forecast Estimator reads either of them. The function drops the family re-basis `L`, its basis `fcb` and the fitted forecast `rf` of the prior. A re-basis of the truncated block does not follow from the block, and an unset `L` reads back as `M`. The history of `rf` covers the rows after `tb`.

# Algorithm

 1. Cut the exposure history `Ms` to the rows `1:tb`, into `Mb`, when the block carries one.
 2. Take the last slice of `Mb` as the loadings `M`, or keep the loadings of the block when it carries no exposure history.
 3. Cut the factor returns `f`, the residuals `eps`, the counts `n` and the intercepts `b` of the cross-sectional fit `csr` to the rows `1:tb`.
 4. Cut the variance history `vs`, the regression weight history `rw` and the benchmark weight history `bw` to the rows `1:tb`.
 5. Build the block from these histories, with `b`, `esigma`, `nf`, `fam` and `lag` unchanged, and with no `L`, `fcb` or `rf`.

# Arguments

  - `csfm`: The fitted factor-model block.
  - `tb`: Number of leading observations of the block to keep.

# Returns

  - `csfm::CrossSectionalFactorModel`: The block on its first `tb` observations.

# Related

  - [`forecast_history`](@ref)
  - [`forecast_history_refit`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function forecast_history_block(csfm::CrossSectionalFactorModel,
                                tb::Integer)::CrossSectionalFactorModel
    csr = csfm.csr
    Ms = csfm.Ms
    vs = csfm.vs
    rw = csfm.rw
    bw = csfm.bw
    Mb = isnothing(Ms) ? nothing : Ms[1:tb, :, :]
    return CrossSectionalFactorModel(; M = isnothing(Mb) ? csfm.M : Mb[end, :, :],
                                     b = csfm.b,
                                     csr = if isnothing(csr)
                                         nothing
                                     else
                                         CrossSectionalRegression(; f = csr.f[1:tb, :],
                                                                  eps = csr.eps[1:tb, :],
                                                                  n = csr.n[1:tb],
                                                                  b = if isnothing(csr.b)
                                                                      nothing
                                                                  else
                                                                      csr.b[1:tb]
                                                                  end)
                                     end, Ms = Mb,
                                     vs = isnothing(vs) ? nothing : vs[1:tb, :],
                                     esigma = csfm.esigma,
                                     rw = isnothing(rw) ? nothing : rw[1:tb, :],
                                     bw = isnothing(bw) ? nothing : bw[1:tb, :],
                                     nf = csfm.nf, fam = csfm.fam, lag = csfm.lag)
end
"""
    forecast_history_refit(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                           csfm::CrossSectionalFactorModel, mu::VecNum,
                           step::Integer) -> MatNum

Build the history of a Return Forecast that computes none, by refitting it along the evaluation grid.

A member that publishes only the latest cross-section gives one row of the history per fit, so the history is the sequence of those rows. The grid starts at the first observation of the block and has a stride of `step`. A row off the grid carries `NaN`.

[`forecast_evaluation_dates`](@ref) scores on the same grid when the evaluation has the same `step`. A row off the grid carries no finite forecast, so the first evaluation date is a row of the grid, and each later date is a stride of `step` after it. So every evaluation date is a row at which the function fitted the member.

The fit at row `t` of the block reads the carrier through the carrier row of that observation, and the block through its row `t`. It reads nothing after the observation that it forecasts. At the last row of the block the fit reads the whole sample, and `mu` is that fit. The function writes `mu` into the last row when the last row is on the grid, and does not fit the member again. When the last row is off the grid, it carries `NaN` as every other row off the grid does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\alpha}_{t} &= \\begin{cases} \\hat{\\boldsymbol{\\mu}}^{(t)} & \\text{if } (t - 1) \\bmod s = 0\\,, \\\\ \\mathrm{NaN} & \\text{otherwise}\\,, \\end{cases} \\\\
r_{t} &= T_{c} - T_{b} + t\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_t_fc])
  - ``\\hat{\\boldsymbol{\\mu}}^{(t)}``: Forecast of the member fitted on the carrier rows ``1`` to ``r_{t}`` and on the block rows ``1`` to ``t``.
  - ``r_{t}``: Carrier row of block row ``t``. The block is a suffix of the carrier, so ``r_{T_{b}} = T_{c}``, and ``\\hat{\\boldsymbol{\\mu}}^{(T_{b})}`` is the fit on the whole sample.
  - ``s``: Number of observations between two refits.
  - ``T_{c}``: Number of observations of the carrier.
  - ``T_{b}``: Number of observations of the block, ``1 \\le t \\le T_{b}``.

# Algorithm

 1. Find the carrier rows of the block with [`return_forecast_rows`](@ref), into `rows`, and their number, into `Tb`.
 2. Fill the history `hist`, `Tb × assets`, with `NaN` in the element type of `mu`.
 3. At each row `tb` of the grid `1:step:Tb`, cut the carrier to the rows `1:rows[tb]` with [`port_opt_view`](@ref).
 4. Cut the block to its rows `1:tb` with [`forecast_history_block`](@ref).
 5. Fit the member on the cut carrier and the cut block with [`return_forecast`](@ref), and write its forecast into row `tb` of `hist`. At `tb == Tb`, write `mu` and fit nothing.

# Arguments

  - `rfe`: Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.
  - `mu`: The member's forecast on the whole sample, which is its fit at the block's last observation.
  - `step`: Number of observations between two refits.

# Returns

  - `hist::MatNum`: Return Forecast history, `observations × assets`, on the rows of the block, in the element type of `mu`. A row off the grid carries `NaN`.

# Related

  - [`forecast_history`](@ref)
  - [`forecast_history_block`](@ref)
  - [`forecast_evaluation_dates`](@ref)
  - [`return_forecast_rows`](@ref)
  - [`return_forecast`](@ref)
"""
function forecast_history_refit(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                                csfm::CrossSectionalFactorModel, mu::VecNum,
                                step::Integer)::MatNum
    rows = return_forecast_rows(rd, csfm)
    Tb = length(rows)
    Tf = real(eltype(mu))
    hist = fill(Tf(NaN), Tb, length(mu))
    for tb in 1:step:Tb
        hist[tb, :] = if tb == Tb
            mu
        else
            return_forecast(rfe, port_opt_view(rd, 1:rows[tb], :),
                            forecast_history_block(csfm, tb)).mu
        end
    end
    return hist
end
"""
    forecast_history(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                     csfm::CrossSectionalFactorModel; step::Integer = 1) -> MatNum
    forecast_history(rfe::CustomValueReturnForecast, rd::ReturnsResult,
                     csfm::CrossSectionalFactorModel; step::Integer = 1)

Return the history of a Return Forecast Estimator, refitting the member if it computes none.

An evaluation scores a forecast at every date it holds, so it needs the forecast at each observation and not only the latest cross-section. Two of the four shipped members compute that history. [`FixedWeightedReturnForecast`](@ref) composes its Descriptor scores observation by observation, and [`ExpWeightedReturnForecast`](@ref) advances a recursion. Both carry `hist`, and this function returns it. [`TargetReturnForecast`](@ref) fits one cross-section over the whole sample and publishes one row, so [`forecast_history_refit`](@ref) builds its history by refitting it along the evaluation grid.

The function fits the member once on the whole sample. When the Result carries a history, the function returns it and fits nothing more. When the Result carries none, the function starts the refit. So a member added later needs no method of this function, and a member that computes a history is fitted once.

# The refit is the expensive path

A member that carries no history costs the first fit and one refit at each row of the grid, except the last row of the block. The grid holds every evaluation date. A `TargetReturnForecast` that carries a [`CrossValidationEstimator`](@ref) in `cv` runs its whole cross-validation at each of those fits. A larger `step` makes fewer fits. Pass the `step` of the evaluation, so that each evaluation date is a row of the grid.

# Algorithm

 1. Check that `step` is at least one.
 2. Fit the member on the whole sample with [`return_forecast`](@ref), into `rf`.
 3. Return `rf.hist` when the Result carries a history.
 4. Otherwise build the history with [`forecast_history_refit`](@ref) from the forecast `rf.mu`, at the stride `step`.

# Arguments

  - `rfe`: Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.
  - `step`: Number of observations between two refits. A member that computes its own history ignores it, because that history gives every observation.

# Validation

  - `step >= 1`. Raises a `DomainError`.
  - [`CustomValueReturnForecast`](@ref) states its forecast and fits nothing, so a refit gives the same values at every observation and no history of it exists. Raises a [`ConflictingArgumentError`](@ref), whatever `step` is.
  - The rules of [`return_forecast`](@ref) for the member.

# Returns

  - `hist::MatNum`: Return Forecast history, `observations × assets`, on the rows of the block, in return units.

# Related

  - [`forecast_history_refit`](@ref)
  - [`forecast_history_block`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`AbstractReturnForecastEstimator`](@ref)
  - [`return_forecast`](@ref)
"""
function forecast_history(rfe::AbstractReturnForecastEstimator, rd::ReturnsResult,
                          csfm::CrossSectionalFactorModel; step::Integer = 1)::MatNum
    @argcheck(step >= one(step), DomainError(step, "step must be >= 1"))
    rf = return_forecast(rfe, rd, csfm)
    hist = rf.hist
    if isnothing(hist)
        return forecast_history_refit(rfe, rd, csfm, rf.mu, step)
    end
    H::MatNum = hist
    return H
end
function forecast_history(::CustomValueReturnForecast, ::ReturnsResult,
                          ::CrossSectionalFactorModel; step::Integer = 1)
    return throw(ConflictingArgumentError("an evaluation scores a Return Forecast at every date it holds, and CustomValueReturnForecast states one cross-section rather than fitting one, so refitting it states the same numbers at every observation and no history of it exists. Score a fitted member, or pair the stated values with a target through the bare method of forecast_evaluation"))
end

export forecast_history
