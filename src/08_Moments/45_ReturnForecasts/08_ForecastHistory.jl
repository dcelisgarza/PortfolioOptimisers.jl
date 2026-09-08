"""
    forecast_history_block(csfm::CrossSectionalFactorModel,
                           tb::Integer) -> CrossSectionalFactorModel

Return the factor-model block truncated to its first `tb` observations.

A member that computes no history is scored by refitting it at each evaluation date, and a refit at an observation may read nothing after it. This verb states what "before an observation" means for a block: every history the Return Forecast family reads is cut to the rows `1:tb`, and the loadings `M` follow the exposure history, whose last slice they are.

The per-asset summaries `b` and `esigma` are carried unchanged, because they describe the assets rather than the observations and no Return Forecast Estimator reads either. The family re-basis `L` and its basis `fcb` are dropped, because a re-basis of the truncated block is not derivable from the block and an unset `L` reads back as `M`, which is the truthful statement that the prefix carries none. The prior's own fitted forecast `rf` is dropped for the same reason: its history would state the untruncated rows.

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

A member that publishes only the latest cross-section states one row of the history per fit, so the path is the sequence of those rows. The grid is anchored at the block's first observation and strides by `step`, which is the grid [`forecast_evaluation_dates`](@ref) then scores on: its own stride is the same `step` and its first date is the first row of this grid that carries a finite pair, so every evaluation date is a row that was fitted. A row off the grid carries `NaN`.

The fit at observation `tb` sees the carrier through the row the block's observation `tb` sits on, and the block through its own row `tb`, so it reads nothing after the observation it answers for. The last observation of the block is the whole sample, and `mu` is already that fit, so the loop reuses it rather than repeating it.

# Arguments

  - `rfe`: Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.
  - `mu`: The member's forecast on the whole sample, which is its fit at the block's last observation.
  - `step`: Number of observations between two refits.

# Returns

  - `hist::MatNum`: Return Forecast history, `observations × assets`, on the block's rows, `NaN` off the grid.

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

An evaluation scores a forecast at every date it holds, so it needs the whole path and not only the latest cross-section. Two of the four shipped members publish one: [`FixedWeightedReturnForecast`](@ref) composes its Descriptor scores observation by observation and [`ExpWeightedReturnForecast`](@ref) advances a recursion, so both carry `hist` and this verb hands it back. [`TargetReturnForecast`](@ref) fits one cross-section over the whole sample and publishes one row, so the path is built by refitting it along the evaluation grid through [`forecast_history_refit`](@ref).

The member is fitted once whatever it is, and its own Result says which of the two happens: a fitted Result that carries a history publishes the path already, and one that carries none is refitted. So a member added later needs no method here, and a member that publishes a history is never refitted behind the caller's back.

# The refit is the expensive path

A member that carries no history costs one fit per evaluation date rather than one fit. A `TargetReturnForecast` that carries a [`CrossValidationEstimator`](@ref) in `cv` runs its whole cross-validation at each of them. `step` is what a caller trades against that cost, and passing it the evaluation's own `step` is what keeps the two grids aligned.

# Arguments

  - `rfe`: Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block.
  - `step`: Number of observations between two refits. A member that publishes its own history ignores it, because the history it publishes states every observation.

# Validation

  - `step >= 1`. Raises a `DomainError`.
  - [`CustomValueReturnForecast`](@ref) states its forecast rather than fitting one, so no refit can give it a path at any observation but the one it states. Raises a [`ConflictingArgumentError`](@ref).
  - The rules of [`return_forecast`](@ref) for the member.

# Returns

  - `hist::MatNum`: Return Forecast history, `observations × assets`, on the block's rows, in return units.

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
