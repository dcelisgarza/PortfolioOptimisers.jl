"""
$(DocStringExtensions.TYPEDEF)

Carries the carry fold of a Cross-Sectional Factor Prior between two online steps.

A [`CrossSectionalFactorPrior`](@ref) with no `cache` seeds this state at its first [`partial_fit!`](@ref), as the carry of [`EmpiricalPrior`](@ref) seeds a [`PriorCarryState`](@ref). The state applies the rule of the carry fold: it folds what folds and refits the rest. A new observation changes no past Factor Exposure, no past regression and no past idiosyncratic variance, so a step computes them for the new observations alone. The idiosyncratic variance and the factor prior fold exactly. So does the idiosyncratic correlation under `th > 0` when its `ce` folds, as [`supports_partial_fit`](@ref) answers. The Return Forecast, and a `ce` that does not fold, refit at the call with no data from the carried rows. A Return Forecast that computes its history one observation at a time, as [`folds_forecast_rows`](@ref) answers, folds too: the state carries its Descriptor scores and its rows, and a step computes the rows of its new observations alone.

Two choices read every fitted observation: the automatic dropped member of a Factor Family under a [`BatchChoice`](@ref), and the mark of the Empty Factors. A step that moves the dropped member solves no observation of full rank again: [`cross_sectional_fold_move`](@ref) selects the columns of the raw factor returns that the new member keeps, and folds the factor prior again over them. That fold reads every fitted factor return, so its cost grows with the stream. The move solves each rank-deficient observation again with [`cross_sectional_move_solve`](@ref): an observation with an Unseen Member under [`SolvedUnseenMember`](@ref), or one with a dependent factor set. Under a [`CrossSectionalTargetRegression`](@ref) whose target answers `false` to [`is_basis_invariant`](@ref), a move fits every carried observation again, as [`cross_sectional_move_basis`](@ref) states. Under a [`GeneralisedLinearModel`](@ref) target the move folds too, because its iterative fit takes the same iterates in every basis. A step where an Empty Factor comes alive solves no fitted observation again: the factor has a return of zero at each of them, and [`cross_sectional_step_factors`](@ref) folds the factor prior again over every fitted factor return. That fold reads every fitted observation, so its cost grows with the stream, and it happens at most once for each factor. Under a regression estimator that changes the answer of an observation whose design gains a zero column, as [`cross_sectional_alive_folds`](@ref) answers, that step fits every carried observation again.

A prior whose tree reads the Exogenous Series, as [`reads_exogenous_series`](@ref) answers, folds an observed factor too. The buffer and the carried rows hold the series. The returns net of the observed factors are derived series: the batch fit derives each row from the observed exposures of the row `lag` observations before it, and the first `lag` rows of the sample from the exposure of the same row. So the state derives each row one time, and carries it.

A slot that reads the Return Forecast history, as [`reads_forecast_history`](@ref) answers, makes the state carry that history, unless the member publishes its own, as [`cross_sectional_carries_history`](@ref) answers. The row of a block observation reads that observation and the observations before it, so a new observation changes no old row. A step that fits the new observations alone therefore fits the Return Forecast once for each new row and appends the row. A step that fits every observation again can change every row, so it makes every row again. The state keeps a row for every observation of the block, because the batch fit gives a rule every row. A Scenario Cap on the factor prior cuts its scenarios and no observation of the block, so it cuts no row of the history.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalCarryState(; buf::SampleBufferState = SampleBufferState(),
                             win::Option{<:ReturnsResult} = nothing, der = nothing,
                             nf = nothing, fam = nothing, Ms = nothing, X = nothing,
                             Xl = nothing, obs = nothing, bw = nothing, mcap = nothing,
                             amsk = nothing, emsk = nothing, sums = nothing,
                             families = nothing, fcb = nothing, Z = nothing, csr = nothing,
                             lv1 = nothing, lv = nothing, W = nothing, vs = nothing,
                             ve = nothing, ve1 = nothing, pe = nothing,
                             seed::Integer = 0, hist = nothing, S = nothing,
                             Sc = nothing, ce = nothing, fsc = nothing, fh = nothing,
                             fst = nothing, xf = nothing, fds = nothing,
                             tip::Base.RefValue{Int} = Ref(0)) -> CrossSectionalCarryState

Keywords correspond to the struct's fields, and every field but `buf` and `tip` defaults to `nothing`, or to `0` for `seed`. `tip` defaults to a new counter at `0`. The default is the empty state that a first step builds.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`partial_fit!`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
  - [`PriorCarryState`](@ref)
  - [`SampleBufferState`](@ref)
"""
@concrete struct CrossSectionalCarryState <: AbstractPartialFitState
    """
    Buffer of every folded observation: the returns, both masks, and the Exogenous Series when the prior reads it. The online step of an optimiser rebuilds its returns data from it with [`returns_buffer`](@ref).
    """
    buf
    """
    The carried rows of the Asset Panel, as a [`ReturnsResult`](@ref) of the returns, the panel, and the Exogenous Series when the prior reads it, or `nothing` before the first step. It holds the last rows that the Factor Exposures of a new observation read, as [`cross_sectional_carry_rows`](@ref) states, or every row.
    """
    win
    """
    The derived series of the carried rows, `(; Xn, Xl, Zo)`, or `nothing` without an observed factor. `Xn` holds the returns net of the observed members that read no returns, which the other observed members read, or `nothing` when no member needs them. `Xl` holds the returns net of every observed factor, which the estimated members read. `Zo` holds the observed exposures of the last `lag` rows, `observations × assets × factors`, which the derivation of the rows of the next step reads.
    """
    der
    """
    Names of the raw factor axis, or `nothing` before the first step.
    """
    nf
    """
    Factor Family label of each raw factor, or `nothing` before the first step.
    """
    fam
    """
    Neutralised exposures of every observation after the Descriptor warm-up, `observations × assets × raw factors`, or `nothing` before the warm-up ends.
    """
    Ms
    """
    Returns of every observation after the warm-up, `observations × assets`.
    """
    X
    """
    Returns net of the observed factors of every observation after the warm-up, which the regression reads, or `nothing` without an observed factor, when the regression reads `X`.
    """
    Xl
    """
    Observed factors of every observation after the warm-up, `(; Z, R, lv, nf, fam)` as [`cross_sectional_observed`](@ref) states them, or `nothing` without an observed factor. The factor prior reads the observed returns `R` beside the factor returns of the regression. The observed exposures `Z` and the exposure history `Ms` are views of one backing, as [`cross_sectional_fold_join`](@ref) appends them, so the call with no data reads the joined history with no copy.
    """
    obs
    """
    Benchmark weights of every observation after the warm-up.
    """
    bw
    """
    Market capitalisation of every observation after the warm-up, or `nothing` when the prior reads none.
    """
    mcap
    """
    Active mask of every observation after the warm-up.
    """
    amsk
    """
    Estimation mask of every observation after the warm-up.
    """
    emsk
    """
    Sum over the observations after the warm-up of the absolute benchmark-weighted exposure of each raw factor, which the automatic choice of a dropped member reads, or `nothing` without a constrained Factor Family.
    """
    sums
    """
    The constrained Factor Families with the dropped member of each one named, as the last fit chose them, or `nothing`.
    """
    families
    """
    Factor Family Basis of every observation after the warm-up, or `nothing`.
    """
    fcb
    """
    Reduced exposures of the last `lag` observations, which the regression of the next observations reads.
    """
    Z
    """
    The regression of every fitted observation, or `nothing` before the first fit.
    """
    csr
    """
    Mark of the factors that are not empty in the first pass of the regression.
    """
    lv1
    """
    Mark of the factors that are not empty in the last pass of the regression.
    """
    lv
    """
    Regression weights of every fitted observation.
    """
    W
    """
    Idiosyncratic variance history of every fitted observation.
    """
    vs
    """
    The variance estimator `ve` of the prior, folded over the residuals of the last pass.
    """
    ve
    """
    The variance estimator `ve` of the prior, folded over the residuals of the first pass, or `nothing` when the weight policy needs no second pass.
    """
    ve1
    """
    The factor prior `pe` of the prior, folded over the factor returns, or the factor prior as it is when it does not fold.
    """
    pe
    """
    Number of fitted observations of the first fit. A fit of every carried observation folds them as one block again, so a [`SeedWindow`](@ref) cuts the rows that it cut at the first fit.
    """
    seed
    """
    The Return Forecast history at every fitted observation but the last, `observations × assets`, or `nothing` when the state carries none. The row of an observation is the forecast of the member fitted on that observation and the observations before it, as [`forecast_history_refit`](@ref) fits it. The call with no data appends the forecast of the last observation.
    """
    hist
    """
    The standardised idiosyncratic returns of every fitted observation, with the fill, as [`cross_sectional_standardised_residuals`](@ref) states them, or `nothing` before the first fit. A row reads its own observation alone, so a step appends the rows of its new observations, and the call with no data reads them as they are.
    """
    S
    """
    The standardised idiosyncratic returns of every fitted observation with no fill, which the idiosyncratic correlation reads, or `nothing` before the first fit, when the threshold `th` of the prior is zero, and when `ce` folds them.
    """
    Sc
    """
    The covariance estimator `ce` of the prior, folded over the standardised idiosyncratic returns with no fill of every fitted observation, or `nothing` before the first fit, when the threshold `th` of the prior is zero, and when `ce` does not fold, as [`supports_partial_fit`](@ref) answers. A step folds the rows of its new observations into it, so the call with no data reads the correlation from its state.
    """
    ce
    """
    The Descriptor scores of the Return Forecast at every observation after the warm-up, `(; S, w, g)` as [`descriptor_panel_scores`](@ref) states them, or `nothing` when the forecast does not compute its history one observation at a time, as [`folds_forecast_rows`](@ref) answers. The scores of the last rows of the warm-up that [`cross_sectional_forecast_lead`](@ref) counts come first. A step appends the scores of its new observations, with [`cross_sectional_carry_scores`](@ref).
    """
    fsc
    """
    The Return Forecast history at every fitted observation, `observations × assets`, or `nothing` when the forecast does not fold. The call with no data reads the Result of the forecast off it, with [`return_forecast_result`](@ref).
    """
    fh
    """
    The fold state of the Return Forecast after the fitted observations whose target is known, as [`return_forecast_step`](@ref) gives it, or `nothing` when the forecast does not fold or carries no state. A step passes it to the forecast, and the call with no data builds the Result of the forecast with it.
    """
    fst
    """
    The factor list, as [`cross_sectional_descriptor_carry`](@ref) and [`cross_sectional_observed`](@ref) folded its members, so that each Descriptor that carries a state holds its state after the folded observations, or `nothing` before the first step. A step folds its new observations into them, and reads the Descriptors of those observations off them.
    """
    xf
    """
    The Descriptor Scores of the Return Forecast, as [`descriptor_carry`](@ref) folded their Descriptors, so that each Descriptor that carries a state holds its state after the folded observations. It is `nothing` before the first step, and when the forecast does not compute its history one observation at a time, as [`folds_forecast_rows`](@ref) answers. A step folds its new observations into them, and scores those observations with the Descriptors that it reads off them, in [`cross_sectional_carry_scores`](@ref).
    """
    fds
    """
    The number of observations that the newest state of this lineage folded. Every state that a step derives from this one shares the counter. A history appends in place only when the state is the newest one, as [`cross_sectional_carry_own`](@ref) checks.
    """
    tip
end
function CrossSectionalCarryState(; buf::SampleBufferState = SampleBufferState(),
                                  win::Option{<:ReturnsResult} = nothing, der = nothing,
                                  nf = nothing, fam = nothing, Ms = nothing, X = nothing,
                                  Xl = nothing, obs = nothing, bw = nothing, mcap = nothing,
                                  amsk = nothing, emsk = nothing, sums = nothing,
                                  families = nothing, fcb = nothing, Z = nothing,
                                  csr = nothing, lv1 = nothing, lv = nothing, W = nothing,
                                  vs = nothing, ve = nothing, ve1 = nothing, pe = nothing,
                                  seed::Integer = 0, hist = nothing, S = nothing,
                                  Sc = nothing, ce = nothing, fsc = nothing, fh = nothing,
                                  fst = nothing, xf = nothing, fds = nothing,
                                  tip::Base.RefValue{Int} = Ref(0))::CrossSectionalCarryState
    return CrossSectionalCarryState(buf, win, der, nf, fam, Ms, X, Xl, obs, bw, mcap, amsk,
                                    emsk, sums, families, fcb, Z, csr, lv1, lv, W, vs, ve,
                                    ve1, pe, seed, hist, S, Sc, ce, fsc, fh, fst, xf, fds,
                                    tip)
end
function returns_buffer(state::CrossSectionalCarryState)
    return state.buf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`CrossSectionalCarryState`](@ref), so that the copy shares no array and no folded state with the original. The copy starts a lineage of its own, so its first step appends to its histories in place.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::CrossSectionalCarryState`: A new state, equal to `x`.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`partial_fit`](@ref)
"""
function Base.copy(x::CrossSectionalCarryState)
    fns = fieldnames(CrossSectionalCarryState)
    # One `deepcopy` of every field keeps two histories that share a backing together.
    c = CrossSectionalCarryState(;
                                 NamedTuple{fns}(deepcopy(map(f -> getfield(x, f), fns)))...)
    return cross_sectional_carry_with(c, (; tip = Ref(c.buf.n)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`CrossSectionalCarryState`](@ref) fitted on disjoint blocks.

The state is not a sufficient statistic of its own block. The first regression of the second block reads the exposures of the last rows of the first block, the Descriptors of its first rows read the panel rows of the first block, and the idiosyncratic variance and the factor prior fold in the order of the observations. So the two blocks fold in sequence.

# Arguments

  - `a`: The state of the first block.
  - `b`: The state of the second block.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(::CrossSectionalCarryState, ::CrossSectionalCarryState)
    return throw(ArgumentError("the carry fold of a Cross-Sectional Factor Prior cannot merge two states fitted on disjoint blocks: the regression of the second block reads the exposures of the last rows of the first, and the idiosyncratic variance and the factor prior fold in the order of the observations. Fold the second block into the state of the first with partial_fit!."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a view of a [`CrossSectionalCarryState`](@ref) on a subset of the assets.

The regression of each observation reads every asset of its cross-section, so the factor returns of a sub-universe are not a slice of the factor returns that the state carries.

# Arguments

  - `x`: The state.
  - `i`: The selected assets.
  - `args...`: Not read.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(::CrossSectionalCarryState, i, args...)
    return throw(ArgumentError("the carry fold of a Cross-Sectional Factor Prior regresses each observation on every asset of its cross-section, so the factor returns of a sub-universe are not a slice of the ones it carries. Fit the sub-universe in batch, or wrap the prior in `Online` to refit it."))
end
"""
    cross_sectional_forecast_reads_panel(rfe::Nothing)
    cross_sectional_forecast_reads_panel(rfe::CustomValueReturnForecast)
    cross_sectional_forecast_reads_panel(rfe::AbstractReturnForecastEstimator)

Answers whether the Return Forecast of a Cross-Sectional Factor Prior reads the Asset Panel at the call with no data of the carry fold.

A Return Forecast that reads the panel refits at each call with no data from the carried rows. Its Result keeps a value at every fitted observation, and [`return_forecast_rows`](@ref) aligns the fitted observations with the last rows of the returns data, so the carry fold carries every panel row under it. A [`CustomValueReturnForecast`](@ref) and an absent forecast read no row. A forecast that computes its history one observation at a time, as [`folds_forecast_rows`](@ref) answers, reads no row at the call with no data either: a step computes its rows, and the state carries them.

# Arguments

  - `rfe`: The Return Forecast Estimator, or `nothing`.

# Returns

  - `reads::Bool`: `true` when the forecast reads the panel.

# Related

  - [`cross_sectional_carry_rows`](@ref)
  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_forecast_reads_panel(::Nothing)
    return false
end
function cross_sectional_forecast_reads_panel(::CustomValueReturnForecast)
    return false
end
function cross_sectional_forecast_reads_panel(rfe::AbstractReturnForecastEstimator)
    return !folds_forecast_rows(rfe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the number of panel rows that the carry fold of a Cross-Sectional Factor Prior carries, or `nothing` for every row.

The Factor Exposures of a new observation read the last rows of the panel that [`carry_lookback`](@ref) of the prior counts, so the carry keeps them. A Descriptor that carries a state reads the new observation alone, as [`descriptor_carry`](@ref) folds it, so it adds no row. Under a Return Forecast that reads the panel at the call with no data, as [`cross_sectional_forecast_reads_panel`](@ref) answers, the carry keeps every row. A forecast that computes its history one observation at a time reads its last [`carry_lookback`](@ref) rows, which [`carry_lookback`](@ref) of the prior counts: a Descriptor of its scores that carries a state counts one row. An unbounded look-back keeps every row too.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.

# Returns

  - `n::Option{<:Integer}`: The number of rows, or `nothing`.

# Related

  - [`carry_lookback`](@ref)
  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_carry_rows(pe::CrossSectionalFactorPrior)::Option{<:Integer}
    return cross_sectional_forecast_reads_panel(pe.rfe) ? nothing : carry_lookback(pe)
end
function carry_lookback(pe::CrossSectionalFactorPrior)::Option{<:Integer}
    return cross_sectional_lookback(pe, carry_lookback(map(last, pe.factors)),
                                    carry_lookback)
end
"""
    cross_sectional_descriptor_carry(est::AbstractVector, rd::ReturnsResult, n::Nothing)
    cross_sectional_descriptor_carry(est::AbstractVector, rd::ReturnsResult, n::Integer)

Folds the last `n` observations of a [`ReturnsResult`](@ref) into the Descriptors of the estimated members of a factor list that carry a state, on the carry fold of a [`CrossSectionalFactorPrior`](@ref).

Each member folds with [`descriptor_carry`](@ref). The batch fit gives `n = nothing`, and gets the members as they are twice.

# Arguments

  - `est`: The estimated members, pairs of `factor name => Exposure Estimator`, as the carry folded them so far.
  - $(arg_dict[:rd]) It holds the panel rows that the carry carries, followed by the new observations.
  - `n`: The number of new observations, or `nothing`.

# Returns

  - `carry::NamedTuple`: `xf`, the members with the state after the new observations, and `xv`, the members whose stateful Descriptors are [`CarriedDescriptor`](@ref) of the new observations.

# Related

  - [`descriptor_carry`](@ref)
  - [`cross_sectional_exposure_series`](@ref)
"""
function cross_sectional_descriptor_carry(est::AbstractVector, ::ReturnsResult, ::Nothing)
    return (; xf = est, xv = est)
end
function cross_sectional_descriptor_carry(est::AbstractVector, rd::ReturnsResult,
                                          n::Integer)
    cs = map(p -> descriptor_carry(last(p), rd, n), est)
    return (; xf = map((p, c) -> first(p) => c.xf, est, cs),
            xv = map((p, c) -> first(p) => c.xv, est, cs))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends the observations of a step to the carried panel rows, and returns the rows that the exposures of the step read.

# Arguments

  - `win`: The carried rows, or `nothing` before the first step.
  - `rd`: The returns data of the step.
  - `xk`: The Exogenous Series of the step, `(; ne, E)` as [`exogenous_step_kwargs`](@ref) answers when the prior reads it, or an empty `NamedTuple`.

# Returns

  - `rd::ReturnsResult`: The carried rows followed by the rows of the step, with the returns, the Asset Panel and the series of `xk` alone.

# Related

  - [`cross_sectional_carry_fold`](@ref)
"""
function cross_sectional_window_append(::Nothing, rd::ReturnsResult, xk::NamedTuple)
    return ReturnsResult(; nx = rd.nx, X = rd.X, pnl = rd.pnl, xk...)
end
function cross_sectional_window_append(win::ReturnsResult, rd::ReturnsResult,
                                       xk::NamedTuple)
    return ReturnsResult(; nx = rd.nx, X = vcat(win.X, rd.X), pnl = vcat(win.pnl, rd.pnl),
                         ne = win.ne,
                         E = cross_sectional_fold_append(win.E, get(xk, :E, nothing)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keeps the last `n` rows of the panel rows of a step, of their derived series, or of the Descriptor scores of a Return Forecast, or every row when `n` is `nothing`.

# Arguments

  - `rd`: The panel rows, an array whose first dimension is the rows, the derived series `(; Xn, Xl, Zo)` of the rows, the scores `(; S, w, g)`, or `nothing`. The carry keeps more rows than `lag`, so the cut keeps every row of `Zo`.
  - `n`: The number of rows to keep, or `nothing`.

# Returns

  - `rd`: The rows that the state carries, of the type of the input.

# Related

  - [`cross_sectional_carry_rows`](@ref)
"""
function cross_sectional_window_trim(rd::ReturnsResult, ::Nothing)
    return rd
end
function cross_sectional_window_trim(rd::ReturnsResult, n::Integer)
    T = size(rd.X, 1)
    return T <= n ? rd : port_opt_view(rd, (T - n + 1):T, :)
end
function cross_sectional_window_trim(::Nothing, ::Any)
    return nothing
end
function cross_sectional_window_trim(A::AbstractArray, ::Nothing)
    return A
end
function cross_sectional_window_trim(A::AbstractArray, n::Integer)
    T = size(A, 1)
    return T <= n ? A : copy(selectdim(A, 1, (T - n + 1):T))
end
function cross_sectional_window_trim(der::NamedTuple, n::Option{<:Integer})
    return map(A -> cross_sectional_window_trim(A, n), der)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds a variance estimator over a block of residuals, and returns its variance after each row.

The first `seed` rows fold as one block: the series of those rows is the batch series, and a [`SeedWindow`](@ref) cuts the block. Each later row folds alone, and the variance after it is read from the folded state, so a row reads no residual after its own. An estimator that does not fold, as [`supports_partial_fit`](@ref) answers, gives the batch series of `E`, and the carry fold then gives it every fitted row at each step.

# Arguments

  - `ve`: The variance estimator, folded so far.
  - `E`: Residuals, `observations × assets`.
  - `em`: Estimation mask of the rows of `E`.
  - `am`: Active mask of the rows of `E`.
  - `seed`: Number of leading rows to fold as one block, `0` when `ve` already folded.

# Returns

  - `V::Matrix`: The variance after each row of `E`.
  - `ve`: The estimator after the last row.

# Related

  - [`variance_series`](@ref)
  - [`partial_fit!`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
"""
function cross_sectional_fold_variance(ve, E::MatNum, em::AbstractMatrix{Bool},
                                       am::AbstractMatrix{Bool}, seed::Integer)
    if !supports_partial_fit(ve)
        return (;
                V = variance_series(ve, E; dims = 1, estimation_mask = em,
                                    active_mask = am), ve = ve)
    end
    rows = Vector{Any}(undef, 0)
    if seed > 0
        b = 1:seed
        push!(rows,
              variance_series(ve, E[b, :]; dims = 1, estimation_mask = em[b, :],
                              active_mask = am[b, :]))
        ve = partial_fit!(ve, E[b, :]; dims = 1, estimation_mask = em[b, :],
                          active_mask = am[b, :])
    end
    for t in (seed + 1):size(E, 1)
        ve = partial_fit!(ve, E[t:t, :]; dims = 1, estimation_mask = em[t:t, :],
                          active_mask = am[t:t, :])
        push!(rows, permutedims(Statistics.var(ve)))
    end
    return (; V = reduce(vcat, rows), ve = ve)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the factor returns of new observations into the factor prior of the carry fold.

A factor prior for which [`carry_folds`](@ref) answers `true` folds the rows with [`partial_fit!`](@ref). Any other factor prior stays as it is, and the call with no data refits it over the factor returns that the state carries.

# Arguments

  - `pe`: The factor prior, folded so far.
  - `f`: The factor returns of the new observations, on the factors that are not empty.

# Returns

  - `pe`: The factor prior after the new observations.

# Related

  - [`carry_folds`](@ref)
  - [`cross_sectional_factor_prior`](@ref)
  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_fold_factors(pe::AbstractPriorEstimator, f::MatNum)
    return carry_folds(pe) ? partial_fit!(pe, f) : pe
end
"""
    carry_folds(pe)
    carry_folds(pe::EmpiricalPrior)

Answers whether the carry fold of a [`CrossSectionalFactorPrior`](@ref) folds the factor prior `pe` one row at a time, at a cost that does not grow with the stream.

This is the verb of a factor prior that folds, and it is `public`. A prior that answers `true` keeps the contract of [`EmpiricalPrior`](@ref) on its carry route. [`partial_fit!`](@ref) folds the factor returns of the new observations, `observations × factors`, into a state that the prior holds, and the first call starts from a prior with no state. A step costs the same however many rows the prior folded before. `prior(pe; strict)` with no data then reads the [`LowOrderPrior`](@ref) of every folded row, equal to the batch call over them. The carry fold folds such a prior with [`cross_sectional_fold_factors`](@ref), and reads it with [`cross_sectional_factor_prior`](@ref). It fits any other prior again over every carried factor return at each call with no data, and [`carry_growing_parts`](@ref) lists that prior.

The verb is not [`supports_partial_fit`](@ref). That verb answers `false` for an `EmpiricalPrior`, because the prior keeps its rows, and an outer estimator that holds the rows already refits it from them. The carry fold gives the factor prior its rows and reads its state, so it folds the prior.

An `EmpiricalPrior` on its carry route answers `true` when `me` and `ce` both answer [`supports_partial_fit`](@ref). Its fold refits a member that does not fold over every carried row, as [`fold_member`](@ref) states. Every other prior answers `false`, an `EmpiricalPrior` that holds a [`SampleBufferState`](@ref) included.

# Arguments

  - `pe`: The factor prior.

# Returns

  - `folds::Bool`: `true` when the carry fold folds the factor prior.

# Related

  - [`cross_sectional_fold_factors`](@ref)
  - [`cross_sectional_factor_prior`](@ref)
  - [`carry_growing_parts`](@ref)
  - [`supports_partial_fit`](@ref)
  - [`carry_lookback`](@ref)
"""
function carry_folds(::AbstractPriorEstimator)::Bool
    return false
end
function carry_folds(pe::EmpiricalPrior{<:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:Option{<:PriorCarryState}})::Bool
    return supports_partial_fit(pe.me) && supports_partial_fit(pe.ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the factor prior of the carry fold out of its state.

A factor prior for which [`carry_folds`](@ref) answers `true` answers from the state that [`cross_sectional_fold_factors`](@ref) folded. Any other factor prior refits over the factor returns `f`.

# Arguments

  - `pe`: The factor prior that the state carries.
  - `f`: The factor returns of every fitted observation, on the factors that are not empty.
  - `strict`: Forwarded to the factor prior.

# Returns

  - `pr::LowOrderPrior`: The factor prior.

# Related

  - [`cross_sectional_fold_factors`](@ref)
  - [`cross_sectional_factor_moments`](@ref)
"""
function cross_sectional_factor_prior(pe::AbstractPriorEstimator, f::MatNum;
                                      strict::Bool = false)
    return carry_folds(pe) ? prior(pe; strict = strict) : prior(pe, f; strict = strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs the regression passes of a Cross-Sectional Factor Prior over a block of observations.

The batch fit runs the same passes over every observation. Each observation is fitted on its own row, so a block gives the rows that the batch fit gives, when the masks of the Empty Factors are the masks of every fitted observation. A first fit or a fit of every carried observation states no mark, and the function takes the mark of the block. A step states the marks of the observations it fitted, and regresses the block over those marks joined to the mark of the block. A factor that the block marks and those marks leave empty comes alive at the block. The function answers `nothing` when the regression estimator does not fold such a factor, as [`cross_sectional_alive_folds`](@ref) answers, because every observation must then be fitted again.

# Algorithm

 1. Take the eligibility mask `msk` with [`cross_sectional_eligible`](@ref), drop every pair whose lagged market capitalisation is not finite, and refuse an observation below [`cross_sectional_minra`](@ref).
 2. Take the first-pass weights with [`cs_weights_initial`](@ref), and run the first pass with [`cross_sectional_fold_pass`](@ref).
 3. When [`needs_second_pass`](@ref) answers `true`, fold `ve1` over the first-pass residuals with [`cross_sectional_fold_variance`](@ref). Invert the variance of the row before each observation with [`cross_sectional_lagged_inverse`](@ref), blend the weights with [`cs_weights_blend`](@ref), and run the second pass with [`cross_sectional_fold_pass`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `blk`: The observations: `Zl`, the lagged reduced exposures; `Xr`, the returns; `em` and `am`, the masks; `mcl`, the lagged market capitalisation or `nothing`; `cb`, the observed factors of the block or `nothing`; `fcb`, the Factor Family Basis of the lagged exposures or `nothing`; and `B`, the lagged exposures on the raw axis.
  - `prev`: The marks `lv1` and `lv`, the folded estimator `ve1` and the number `seed` of rows to fold as one block.

# Validation

  - The rules of every verb the algorithm names.

# Returns

  - `fit::Option{<:NamedTuple}`: `csr`, `W`, `lv1`, `lv` and `ve1`, or `nothing` when a factor comes alive and the regression estimator does not fold it.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`prior`](@ref)
"""
function cross_sectional_fold_regression(pe::CrossSectionalFactorPrior, blk::NamedTuple,
                                         prev::NamedTuple)
    (; Zl, Xr, em, am, mcl, cb) = blk
    msk = cross_sectional_eligible(Xr, Zl, em)
    cross_sectional_cap_finite!(msk, mcl)
    assert_cross_sectional_coverage(msk, cross_sectional_minra(pe, size(Zl, 3)), cb)
    W = cs_weights_initial(pe.wa, mcl, msk)
    p1 = cross_sectional_fold_pass(pe, blk, W, prev.lv1)
    if isnothing(p1) || !needs_second_pass(pe.wa)
        return if isnothing(p1)
            nothing
        else
            (; csr = p1.csr, W = W, lv1 = p1.lv, lv = p1.lv, ve1 = nothing)
        end
    end
    # The variance of the row before the block, read before the fold writes into the state.
    v0 = cross_sectional_previous_variance(prev.ve1)
    fv = cross_sectional_fold_variance(something(prev.ve1, pe.ve), p1.csr.eps, em, am,
                                       prev.seed)
    IV = if isnothing(v0)
        cross_sectional_lagged_inverse(fv.V, msk)
    else
        cross_sectional_lagged_inverse(vcat(permutedims(v0), fv.V),
                                       vcat(falses(1, size(msk, 2)), msk))[2:end, :]
    end
    W = cs_weights_blend(pe.wa, W, IV, msk)
    p2 = cross_sectional_fold_pass(pe, blk, W, prev.lv)
    return if isnothing(p2)
        nothing
    else
        (; csr = p2.csr, W = W, lv1 = p1.lv, lv = p2.lv, ve1 = fv.ve)
    end
end
"""
    cross_sectional_previous_variance(ve::Nothing) -> nothing
    cross_sectional_previous_variance(ve::AbstractCovarianceEstimator) -> VecNum

Reads the variance of the observation before a block out of the folded first-pass variance estimator.

A first fit holds no folded estimator, and its first observation reads no earlier variance, so the method over `nothing` answers `nothing`. A step reads the variance of its folded estimator, and copies it, because the fold of the block writes into the state.

# Arguments

  - `ve`: The folded first-pass variance estimator, or `nothing`.

# Returns

  - `v0::Option{<:VecNum}`: The variance of each asset before the block, or `nothing`.

# Related

  - [`cross_sectional_fold_regression`](@ref)
  - [`cross_sectional_lagged_inverse`](@ref)
"""
function cross_sectional_previous_variance(::Nothing)
    return nothing
end
function cross_sectional_previous_variance(ve::AbstractCovarianceEstimator)
    return copy(Statistics.var(ve))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs one regression pass of the carry fold of a Cross-Sectional Factor Prior over a block of observations.

# Algorithm

 1. Take the design of the regression under the Unseen Member rule `pe.unseen` with [`unseen_member_design`](@ref), giving `Z`.
 2. Join the mark of the factors that are not empty over `Z` to `old` with [`cross_sectional_fold_mark`](@ref). Answer `nothing` when a factor comes alive at the block and the regression estimator does not fold it.
 3. Regress `Z` on the marked factors with [`cross_sectional_live_regression`](@ref), map its coefficients back with [`unseen_member_returns`](@ref), and refuse an intercept with [`assert_cross_sectional_no_intercept`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `blk`: The observations, as [`cross_sectional_fold_regression`](@ref) states them. The pass reads `Zl`, the lagged reduced exposures; `Xr`, the returns; `fcb`, the Factor Family Basis of the lagged exposures or `nothing`; and `B`, the lagged exposures on the raw axis.
  - `W`: Regression weights of the block.
  - `old`: The mark of the fitted observations, or `nothing` for a first fit.

# Returns

  - `pass::Option{<:NamedTuple}`: `csr`, the regression, and `lv`, the mark, or `nothing`.

# Related

  - [`cross_sectional_fold_regression`](@ref)
"""
function cross_sectional_fold_pass(pe::CrossSectionalFactorPrior, blk::NamedTuple,
                                   W::MatNum, old::Option{<:BitVector})
    (; Zl, Xr, fcb, B) = blk
    ud = unseen_member_design(pe.unseen, fcb, B, Zl, W)
    lv = cross_sectional_fold_mark(old, cross_sectional_live_factors(ud.Z, Xr, W), pe.cre)
    return if isnothing(lv)
        nothing
    else
        csr = unseen_member_returns(cross_sectional_live_regression(pe.cre, ud.Z, Xr, W,
                                                                    lv).csr, ud.P)
        (; csr = assert_cross_sectional_no_intercept(csr), lv = lv)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a cross-sectional regression that fitted an intercept, for a Cross-Sectional Factor Prior.

The prior states its moments through the factor returns alone, so an intercept would leave its mean out of `mu` and its variance out of `sigma`. The batch fit and the carry fold both call it after each regression pass.

# Arguments

  - `csr`: The regression.

# Validation

  - `csr.b` is `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `csr::CrossSectionalRegression`: The regression, unchanged.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`prior`](@ref)
"""
function assert_cross_sectional_no_intercept(csr::CrossSectionalRegression)
    @argcheck(isnothing(csr.b),
              ArgumentError("a Cross-Sectional Factor Prior states its moments through the factor returns alone, and its regression estimator fitted an intercept, whose mean and variance the moments would leave out. Give cre an estimator with intercept = false, and state the common return as a factor, for example \"market\" => ConstantExposure()."))
    return csr
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Joins the mark of the factors that are not empty over a block to the mark of the fitted observations.

A factor that the block marks and the fitted observations leave empty comes alive at the block. Its exposure is zero at every pair of positive weight of each fitted observation, so the batch fit regresses each such observation over a design with a zero column. The regression estimator decides if that column changes the answer of the observation, as [`cross_sectional_alive_folds`](@ref) answers. When it does not, the join keeps the fitted observations, and the step folds. When it does, the function answers `nothing`, and every observation must be fitted again.

# Arguments

  - `old`: The mark of the fitted observations, or `nothing` for a first fit.
  - `new`: The mark of the block.
  - `cre`: The cross-sectional regression estimator of the prior.

# Validation

  - Without `old`, at least one factor is not empty. Raises an `ArgumentError`.

# Returns

  - `lv::Option{BitVector}`: `new` for a first fit, `old .| new` when no factor comes alive or `cre` folds it, and `nothing` otherwise.

# Related

  - [`cross_sectional_fold_regression`](@ref)
  - [`cross_sectional_live_factors`](@ref)
  - [`cross_sectional_alive_folds`](@ref)
"""
function cross_sectional_fold_mark(::Nothing, new::BitVector, ::Any)
    @argcheck(any(new),
              ArgumentError("every one of the $(length(new)) factors is empty: no factor has a nonzero exposure at an (observation, asset) pair of positive weight, so the regression has nothing to fit. Widen the eligible cross-section, or give factors that the assets load on."))
    return new
end
function cross_sectional_fold_mark(old::BitVector, new::BitVector,
                                   cre::AbstractCrossSectionalRegressionEstimator)
    lv = old .| new
    return lv == old || cross_sectional_alive_folds(cre) ? lv : nothing
end
"""
    cross_sectional_fold_factor_returns(f::MatNum, lv::AbstractVector{Bool}, cb::Nothing)
    cross_sectional_fold_factor_returns(f::MatNum, lv::AbstractVector{Bool},
                                        cb::NamedTuple)

Returns the factor returns of a block that the factor prior of the carry fold folds: the factor returns of the regression on the factors that are not empty, and the observed returns after them.

The batch fit appends the observed factors after the estimated ones with [`cross_sectional_observed_append`](@ref), and fits the factor prior on the factors that are not empty, so the columns are in the same order.

# Arguments

  - `f`: The factor returns of the regression of the block, on the reduced axis.
  - `lv`: The mark of the factors that are not empty.
  - `cb`: The observed factors of the block from [`cross_sectional_observed_block`](@ref), or `nothing`.

# Returns

  - `f::Matrix`: The factor returns that the factor prior folds.

# Related

  - [`cross_sectional_fold_factors`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
"""
function cross_sectional_fold_factor_returns(f::MatNum, lv::AbstractVector{Bool}, ::Nothing)
    return f[:, lv]
end
function cross_sectional_fold_factor_returns(f::MatNum, lv::AbstractVector{Bool},
                                             cb::NamedTuple)
    return hcat(f[:, lv], cb.R)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Names the dropped member of each constrained Factor Family of a basis.

# Arguments

  - `families`: The constrained Factor Families, or `nothing`.
  - `fcb`: The Factor Family Basis that a fit chose for them, or `nothing`.
  - `nf`: Names of the raw factor axis.

# Returns

  - `families::Option{<:Vector{<:Pair}}`: Pairs of `family label => dropped member`, or `nothing`.

# Related

  - [`cross_sectional_pinned_families`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
"""
function cross_sectional_dropped_names(::Nothing, ::Nothing, ::VecStr)
    return nothing
end
function cross_sectional_dropped_names(families::AbstractVector{<:Pair},
                                       fcb::FactorFamilyBasis, nf::VecStr)
    return map(j -> first(families[j]) => nf[fcb.fi[j][fcb.di[j]]], eachindex(families))
end
"""
    cross_sectional_fold_choice(choice::PinnedChoice, pe::CrossSectionalFactorPrior,
                                st::CrossSectionalCarryState)
    cross_sectional_fold_choice(choice::BatchChoice, pe::CrossSectionalFactorPrior,
                                st::CrossSectionalCarryState)

Applies the Choice Rule of a Cross-Sectional Factor Prior at a step of its carry fold.

A pinned choice keeps the dropped members of the first fit, which the state records. A batch choice chooses each automatic member again over every observation after the warm-up. It reads the sums that the state carries, with the rule of [`factor_family_basis`](@ref): the member with the largest sum of absolute benchmark-weighted exposures, and the first such member on a tie. The state adds the rows to the sums in the order of the batch fit, so the two choose the same member. When the member moves, [`cross_sectional_fold_move`](@ref) folds the move: it solves no observation of full rank again, and folds the factor prior again over every fitted factor return.

# Arguments

  - `choice`: The Choice Rule of `pe`.
  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state after the rows of the step.

# Returns

  - `families::Option{<:Vector{<:Pair}}`: Pairs of `family label => dropped member`, or `nothing`.

# Related

  - [`PinnedChoice`](@ref)
  - [`BatchChoice`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
"""
function cross_sectional_fold_choice(::PinnedChoice, ::CrossSectionalFactorPrior,
                                     st::CrossSectionalCarryState)
    return st.families
end
function cross_sectional_fold_choice(::BatchChoice, pe::CrossSectionalFactorPrior,
                                     st::CrossSectionalCarryState)
    if isnothing(pe.families)
        return nothing
    end
    C = reshape(st.sums, 1, :)
    return map(eachindex(pe.families)) do j
        idx = st.fcb.fi[j]
        d = resolve_dropped_member(last(pe.families[j]), String(first(pe.families[j])), idx,
                                   st.nf, C)
        return first(pe.families[j]) => st.nf[idx[d]]
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the absolute benchmark-weighted exposures of new observations to the sums that the automatic choice of a dropped member reads.

# Arguments

  - `families`: The constrained Factor Families of the prior, or `nothing`.
  - `sums`: The sums so far, or `nothing`.
  - `Ms`: Neutralised exposures of the new observations.
  - `bw`: Benchmark weights of the new observations.

# Returns

  - `sums::Option{<:Vector}`: One sum per raw factor, or `nothing` without a constrained Factor Family.

# Related

  - [`cross_sectional_fold_choice`](@ref)
  - [`factor_family_basis`](@ref)
"""
function cross_sectional_fold_sums(::Nothing, ::Any, ::Arr3Num, ::MatNum)
    return nothing
end
function cross_sectional_fold_sums(::AbstractVector{<:Pair}, sums, Ms::Arr3Num, bw::MatNum)
    Tf = float_if_integer(promote_type(real(eltype(Ms)), real(eltype(bw))))
    c = weighted_family_exposures(Ms, bw, Tf)
    s = isnothing(sums) ? zeros(Tf, size(Ms, 3)) : copy(sums)
    for t in axes(c, 1)
        s .+= abs.(view(c, t, :))
    end
    return s
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the neutralised Factor Exposures of the new observations of a step, and drops the rows of the Descriptor warm-up.

The exposures of an observation read the last [`lookback`](@ref) rows of the panel, so the rows that the state carries give the exposures of the batch fit. The derived series of an observation read the observed exposures of the row `pe.lag` observations before it. A Descriptor that carries a state cannot give that row again, so the state carries the observed exposures of the last `pe.lag` rows in `der.Zo`.

# Algorithm

 1. Run [`cross_sectional_exposure_series`](@ref) over the rows, with the derived series of the carried rows that the state holds. It gives the benchmark weights, the observed factors, the returns net of them and the exposures of the estimated factors. It derives the rows of the step alone, and it computes the exposures of the `n` rows of the step alone, each from the last [`lookback`](@ref) rows that the member reads.
 2. Take the last `n` rows, of the observed exposures with [`cross_sectional_observed_carry`](@ref) too. Before the warm-up ends, drop the rows before the first one with an eligible asset on the estimated and the observed exposures together, as [`cross_sectional_warmup`](@ref) does.
 3. Neutralise the exposures of the rows that remain with [`cross_sectional_neutralise!`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state before the step.
  - `win`: The carried rows followed by the rows of the step.
  - `n`: Number of rows of the step.
  - `g0`: Number of observations before the first row of `win`.

# Returns

  - `rows::NamedTuple`: `nf` and `fam`, the raw factor axis; the rows after the warm-up: `Ms`, `X`, `Xl`, `obs`, `bw`, `mcap`, `amsk` and `emsk`, with `Xl` and `obs` `nothing` without an observed factor; `der`, the derived series of every row of `win` and the observed exposures of its last `pe.lag` rows, or `nothing`; and `xf`, the factor list with the state of each Descriptor after the rows of the step.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_exposure_stage`](@ref)
"""
function cross_sectional_fold_rows(pe::CrossSectionalFactorPrior,
                                   st::CrossSectionalCarryState, win::ReturnsResult,
                                   n::Integer, g0::Integer)
    (; amsk, emsk, mcap, BW, Xu, cc, Xl, Ms, nf, fam, xf) = cross_sectional_exposure_series(pe,
                                                                                            win.X,
                                                                                            nothing,
                                                                                            win.pnl;
                                                                                            ne = win.ne,
                                                                                            E = win.E,
                                                                                            kept = st.der,
                                                                                            g0 = g0,
                                                                                            n = n,
                                                                                            xf = st.xf)
    # `Ms` and `Zn` hold the last `n` rows of `win` alone: row `p[j]` of each is row `q[j]` of
    # `win`.
    T = size(win.X, 1)
    q = (T - n + 1):T
    p = 1:n
    (; Zn, der) = cross_sectional_observed_carry(cc, Xl, n, pe.lag)
    if isnothing(st.Ms)
        Mo = isnothing(cc) ? Ms : cat(Ms, Zn[p, :, :]; dims = 3)
        elig = cross_sectional_eligible(Xu[q, :], Mo, emsk[q, :])
        s = something(findfirst(any, eachrow(elig)), n + 1)
        q = q[s:end]
        p = p[s:end]
    end
    Msn = Ms[p, :, :]
    bwn = BW[q, :]
    if !isempty(q)
        cross_sectional_neutralise!(pe.neutralise, Msn, pe.cre, bwn, nf, fam)
    end
    obs = if isnothing(cc)
        nothing
    else
        (; Z = Zn[p, :, :], R = cc.R[q, :], lv = cc.lv, nf = cc.nf, fam = cc.fam)
    end
    return (; nf = nf, fam = fam, Ms = Msn, X = win.X[q, :],
            Xl = isnothing(cc) ? nothing : Xl[q, :], obs = obs, bw = bwn,
            mcap = cross_sectional_rows(mcap, q), amsk = amsk[q, :], emsk = emsk[q, :],
            der = der, xf = xf)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits every observation that the carry fold of a Cross-Sectional Factor Prior carries after the warm-up.

This is the first fit, and the fit after a step where a factor comes alive and the regression estimator does not fold it, as [`cross_sectional_alive_folds`](@ref) answers, or that moves the automatic choice of a dropped member where [`cross_sectional_fold_move`](@ref) does not fold it. It runs the steps of the batch fit over the carried histories. The idiosyncratic variance and the factor prior fold the first `seed` observations as one block, and every later observation alone, so a [`SeedWindow`](@ref) cuts the rows that it cut at the first fit.

# Algorithm

 1. Build the Factor Family Basis of every observation with [`cross_sectional_family_basis`](@ref), under `families`, or under the families of `pe` at the first fit, and name its dropped members with [`cross_sectional_dropped_names`](@ref).
 2. Take the observed factors of every fitted observation with [`cross_sectional_observed_block`](@ref), which refuses an infinite observed return.
 3. Run the regression passes over every fitted observation with [`cross_sectional_fold_regression`](@ref), on the returns net of the observed factors.
 4. Fold `ve` over the residuals with [`cross_sectional_fold_variance`](@ref), and the factor prior over the factor returns of [`cross_sectional_fold_factor_returns`](@ref) with [`cross_sectional_fold_factors`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, whose histories hold every observation after the warm-up.
  - `families`: The families with their dropped members named, or `nothing` at the first fit.
  - `seed`: Number of fitted observations to fold as one block.

# Returns

  - `st::CrossSectionalCarryState`: The state with every fitted output replaced.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`prior`](@ref)
"""
function cross_sectional_fold_refit(pe::CrossSectionalFactorPrior,
                                    st::CrossSectionalCarryState,
                                    families::Option{<:AbstractVector{<:Pair}},
                                    seed::Integer)
    fb = cross_sectional_family_basis(isnothing(families) ? pe.families : families, st.Ms,
                                      st.bw, st.nf, st.fam)
    Tf = size(st.Ms, 1)
    D = Tf - pe.lag
    d = (pe.lag + 1):Tf
    cb = cross_sectional_observed_block(st.obs, 1:Tf, d, st.buf.n - Tf)
    blk = (; Zl = fb.Ms[1:D, :, :], Xr = something(st.Xl, st.X)[d, :], em = st.emsk[d, :],
           am = st.amsk[d, :], mcl = cross_sectional_rows(st.mcap, 1:D), cb = cb,
           fcb = cross_sectional_basis_now(fb.fcb, 1:D), B = st.Ms[1:D, :, :])
    reg = cross_sectional_fold_regression(pe, blk,
                                          (; lv1 = nothing, lv = nothing, ve1 = nothing,
                                           seed = seed))
    (; V, ve) = cross_sectional_fold_variance(pe.ve, reg.csr.eps, blk.em, blk.am, seed)
    f = cross_sectional_fold_factor_returns(reg.csr.f, reg.lv, cb)
    pf = cross_sectional_refold_factors(pe.pe, f, seed)
    return cross_sectional_carry_with(st,
                                      (;
                                       families = cross_sectional_dropped_names(pe.families,
                                                                                fb.fcb,
                                                                                st.nf),
                                       fcb = fb.fcb, Z = fb.Ms[(D + 1):Tf, :, :],
                                       csr = reg.csr, lv1 = reg.lv1, lv = reg.lv, W = reg.W,
                                       vs = V, ve = ve, ve1 = reg.ve1, pe = pf,
                                       seed = seed))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits the new observations of a step of the carry fold of a Cross-Sectional Factor Prior, and folds them into its state.

Each output of a fitted observation reads no later observation, so the step fits the new observations alone and appends their outputs. A factor that the new observations mark and the fitted observations leave empty comes alive at the step. Its exposure is zero at every pair of positive weight of each fitted observation, so its column adds nothing to the fit of that observation, and the answer of least norm gives it a return of zero there, as [`cross_sectional_live_regression`](@ref) states. So the step keeps every fitted output, and regresses the new observations over the joined marks. It folds the factor prior again over every fitted factor return, with [`cross_sectional_step_factors`](@ref), because the factor prior folds the factors that are not empty. The step answers `nothing` when the regression estimator does not fold a factor that comes alive, as [`cross_sectional_alive_folds`](@ref) answers.

# Algorithm

 1. Reduce the exposures of the new observations with [`cross_sectional_family_basis`](@ref) under `families`, and lag them behind the reduced exposures of the last `lag` observations.
 2. Take the observed factors of the new observations with [`cross_sectional_observed_block`](@ref), which refuses an infinite observed return.
 3. Run the regression passes over the new observations with [`cross_sectional_fold_regression`](@ref), under the marks of the fitted observations, on the returns net of the observed factors.
 4. Fold `ve` over the new residuals with [`cross_sectional_fold_variance`](@ref), and fold the factor prior with [`cross_sectional_step_factors`](@ref).
 5. Append every output to the state with [`cross_sectional_fold_append`](@ref), and record the marks of the step.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, whose histories hold the new observations as their last `m` rows.
  - `families`: The families with their dropped members named, or `nothing`.
  - `m`: Number of new observations.

# Returns

  - `st::Option{CrossSectionalCarryState}`: The state after the step, or `nothing`.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_fold_refit`](@ref)
"""
function cross_sectional_fold_step(pe::CrossSectionalFactorPrior,
                                   st::CrossSectionalCarryState,
                                   families::Option{<:AbstractVector{<:Pair}}, m::Integer)
    Tf = size(st.Ms, 1)
    q = (Tf - m + 1):Tf
    fb = cross_sectional_family_basis(families, st.Ms[q, :, :], st.bw[q, :], st.nf, st.fam)
    fcb = cross_sectional_fold_append(st.fcb, fb.fcb)
    Zc = cat(st.Z, fb.Ms; dims = 1)
    cb = cross_sectional_observed_block(st.obs, 1:Tf, q, st.buf.n - Tf)
    blk = (; Zl = Zc[1:m, :, :], Xr = something(st.Xl, st.X)[q, :], em = st.emsk[q, :],
           am = st.amsk[q, :], mcl = cross_sectional_rows(st.mcap, q .- pe.lag), cb = cb,
           fcb = cross_sectional_basis_now(fcb, q .- pe.lag), B = st.Ms[q .- pe.lag, :, :])
    reg = cross_sectional_fold_regression(pe, blk,
                                          (; lv1 = st.lv1, lv = st.lv, ve1 = st.ve1,
                                           seed = 0))
    if isnothing(reg)
        return nothing
    end
    (; V, ve) = cross_sectional_fold_variance(st.ve, reg.csr.eps, blk.em, blk.am, 0)
    csr = cross_sectional_fold_append(st.csr, reg.csr)
    return cross_sectional_carry_with(st,
                                      (; fcb = fcb, Z = Zc[(m + 1):end, :, :], csr = csr,
                                       W = cross_sectional_fold_append(st.W, reg.W),
                                       vs = cross_sectional_fold_append(st.vs, V), ve = ve,
                                       ve1 = reg.ve1, lv1 = reg.lv1, lv = reg.lv,
                                       pe = cross_sectional_step_factors(pe, st, csr,
                                                                         reg.lv)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends the rows of a step to the histories that the carry fold of a Cross-Sectional Factor Prior carries.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state before the step.
  - `rows`: The rows of the step after the warm-up, from [`cross_sectional_fold_rows`](@ref).

# Returns

  - `st::CrossSectionalCarryState`: The state with the rows appended to its histories and its sums.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_fold_join`](@ref)
"""
function cross_sectional_fold_histories(pe::CrossSectionalFactorPrior,
                                        st::CrossSectionalCarryState, rows::NamedTuple)
    if iszero(size(rows.Ms, 1))
        return cross_sectional_carry_with(st, (; nf = rows.nf, fam = rows.fam))
    end
    (; Ms, obs) = cross_sectional_fold_join(st.Ms, st.obs, rows.Ms, rows.obs)
    return cross_sectional_carry_with(st,
                                      (; nf = rows.nf, fam = rows.fam, Ms = Ms, obs = obs,
                                       X = cross_sectional_fold_append(st.X, rows.X),
                                       Xl = cross_sectional_fold_append(st.Xl, rows.Xl),
                                       bw = cross_sectional_fold_append(st.bw, rows.bw),
                                       mcap = cross_sectional_fold_append(st.mcap,
                                                                          rows.mcap),
                                       amsk = cross_sectional_fold_append(st.amsk,
                                                                          rows.amsk),
                                       emsk = cross_sectional_fold_append(st.emsk,
                                                                          rows.emsk),
                                       sums = cross_sectional_fold_sums(pe.families,
                                                                        st.sums, rows.Ms,
                                                                        rows.bw)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends the observed factors to the fitted factors of the carry fold of a Cross-Sectional Factor Prior.

# Algorithm

 1. Take the observed factors of the fitted observations with [`cross_sectional_observed_block`](@ref).
 2. Append them to the fitted factors with [`cross_sectional_observed_append`](@ref), giving the raw exposures of the fitted observations and the reduced loadings of the last one. The two exposure histories share one backing, so the raw exposures are a view of it, as [`cross_sectional_joined_exposures`](@ref) states.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state of its carry fold, after its first fit.

# Returns

  - `r::UnitRange`: The fitted observations among the observations after the warm-up.
  - `ca::NamedTuple`: The answer of [`cross_sectional_observed_append`](@ref).

# Related

  - [`prior`](@ref)
  - [`cross_sectional_carry_history`](@ref)
"""
function cross_sectional_carry_append(pe::CrossSectionalFactorPrior,
                                      st::CrossSectionalCarryState)
    Tf = size(st.Ms, 1)
    r = (pe.lag + 1):Tf
    cb = cross_sectional_observed_block(st.obs, 1:Tf, r, st.buf.n - Tf)
    ca = cross_sectional_observed_append(cb, st.csr.f, st.Z[end, :, :],
                                         view(st.Ms, r, :, :), st.nf, st.fam, st.fcb)
    return (; r = r, ca = ca)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of a [`ReturnsResult`](@ref) into the carry fold of a Cross-Sectional Factor Prior.

This is the step of a prior that carries a [`CrossSectionalCarryState`](@ref). It applies the rule of the carry fold: the exposures, the observed factors, the returns net of them, the regression and the idiosyncratic variance of a new observation read no later observation, so the step computes them for the new observations alone.

# Algorithm

 1. Refuse returns data that no fold takes with [`assert_prior_fold_returns`](@ref), and a panel that is absent or static. Take the Exogenous Series of the step with [`exogenous_step_kwargs`](@ref), which refuses a step with no series when the prior reads it.
 2. Append the rows of the step and their series to the carried panel rows with [`cross_sectional_window_append`](@ref).
 3. Compute the neutralised exposures, the observed factors and the derived series of the new observations with [`cross_sectional_fold_rows`](@ref), and append the rows after the warm-up to the histories with [`cross_sectional_fold_histories`](@ref), and their Descriptor scores with [`cross_sectional_carry_scores`](@ref). Append the returns, both masks and the series to the buffer `buf`. Keep the panel rows and their derived series that [`cross_sectional_carry_rows`](@ref) names.
 4. Before the first fit, fit every observation with [`cross_sectional_fold_refit`](@ref) once `lag + 2` observations follow the warm-up, as the batch fit needs.
 5. After it, apply the Choice Rule with [`cross_sectional_fold_choice`](@ref). When the choice moves, fold the move into the state with [`cross_sectional_fold_move`](@ref). Then fit the new observations with [`cross_sectional_fold_step`](@ref), which folds a factor that comes alive. Fit every observation again with [`cross_sectional_fold_refit`](@ref) when either one answers `nothing`. A variance estimator `pe.ve` that does not fold, as [`supports_partial_fit`](@ref) answers, carries no state, so every step fits every observation again.
 6. Bring the outputs that read the fit up to the fitted observations with [`cross_sectional_carry_outputs`](@ref): the Return Forecast history that a slot reads, the rows of a Return Forecast that folds, and the standardised idiosyncratic returns. Each keeps the carried rows after a step of [`cross_sectional_fold_step`](@ref), and makes every row again after a fit of every observation.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state before the step.
  - $(arg_dict[:rd]) Its `pnl` is a time-varying Asset Panel.

# Validation

  - The rules of [`assert_prior_fold_returns`](@ref).
  - `rd.pnl` is a time-varying Asset Panel. An `ArgumentError` is thrown otherwise.
  - The rules of [`exogenous_step_kwargs`](@ref) and of the block method of [`partial_fit!`](@ref) for a [`SampleBufferState`](@ref): a prior that reads the Exogenous Series gets it at every step, under the names of the first step.
  - The rules of every verb the algorithm names.

# Returns

  - `st::CrossSectionalCarryState`: The state after the step.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`partial_fit!`](@ref)
  - [`prior`](@ref)
"""
function cross_sectional_carry_fold(pe::CrossSectionalFactorPrior,
                                    st::CrossSectionalCarryState, rd::ReturnsResult)
    assert_prior_fold_returns(rd)
    pnl = rd.pnl
    @argcheck(!isnothing(pnl) && !panel_is_static(pnl),
              ArgumentError("a Cross-Sectional Factor Prior reads its Factor Exposures off a time-varying Asset Panel, and the step carries $(isnothing(pnl) ? "no panel" : "a static panel"). Fold a ReturnsResult whose `pnl` is the one asset_panel returns."))
    xk = exogenous_step_kwargs(pe, rd)
    win = cross_sectional_window_append(st.win, rd, xk)
    rows = cross_sectional_fold_rows(pe, st, win, size(rd.X, 1),
                                     st.buf.n + size(rd.X, 1) - size(win.X, 1))
    m = size(rows.Ms, 1)
    st = cross_sectional_carry_scores(pe, cross_sectional_fold_histories(pe, st, rows), win,
                                      rows, size(rd.X, 1))
    n = cross_sectional_carry_rows(pe)
    st = cross_sectional_carry_with(st,
                                    (;
                                     buf = partial_fit!(st.buf, rd.X;
                                                        active_mask = pnl.amsk,
                                                        estimation_mask = pnl.emsk, xk...),
                                     win = cross_sectional_window_trim(win, n),
                                     der = cross_sectional_window_trim(rows.der, n),
                                     xf = rows.xf))
    Tf = isnothing(st.Ms) ? 0 : size(st.Ms, 1)
    if isnothing(st.csr)
        return if Tf - pe.lag < 2
            st
        else
            cross_sectional_carry_outputs(pe,
                                          cross_sectional_fold_refit(pe, st, nothing,
                                                                     Tf - pe.lag), false)
        end
    end
    # After the first fit every row follows the warm-up, and a `ReturnsResult` holds at least
    # one row, so the step brings `m >= 1` new observations. A variance estimator that does
    # not fold has no state to carry, so every step fits every observation again, over the
    # carried histories.
    families = cross_sectional_fold_choice(pe.choice, pe, st)
    mv = cross_sectional_fold_move(pe, st, families, m)
    stepped = isnothing(mv) ? nothing : cross_sectional_fold_step(pe, mv, families, m)
    st, stepped = if isnothing(stepped)
        cross_sectional_fold_refit(pe, st, families, st.seed), false
    else
        stepped, true
    end
    return cross_sectional_carry_outputs(pe, st, stepped)
end
"""
    partial_fit!(pe::CrossSectionalFactorPrior{<:Any, …, <:Option{<:CrossSectionalCarryState}},
                 rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into a Cross-Sectional Factor Prior on its carry route.

An unwrapped prior follows the rule of the carry fold: it folds what folds and refits the rest. Its first step seeds a [`CrossSectionalCarryState`](@ref), as the carry of [`EmpiricalPrior`](@ref) seeds its own state. Each step computes the Factor Exposures and the observed factors of the new observations from the carried panel rows, runs the regression on the new observations alone, and folds the idiosyncratic variance and the factor prior. A prior whose tree reads the Exogenous Series, as [`reads_exogenous_series`](@ref) answers, records it, so each step must bring it. [`cross_sectional_carry_fold`](@ref) states the step. The call with no data `prior(pe)` equals the batch fit over the folded observations up to rounding, until a [`SeedWindow`](@ref) or a [`PinnedChoice`](@ref) keeps a value of the first fit. A pinned choice is recorded in the state, and `families` stays as the caller wrote it.

[`Online`](@ref) seeds the `cache` with a [`SampleBufferState`](@ref) instead, and the prior refits through the generic method of `partial_fit!` and [`refit_prior_step`](@ref). The method narrows `cache` to the state of the carry fold, so a buffer reaches that route. The matrix form refuses through the generic method, because a matrix carries no Asset Panel, and the prior reads its Factor Exposures off one.

# Algorithm

 1. Seed an empty [`CrossSectionalCarryState`](@ref) when `pe.cache` is `nothing`. Copy a state that is not the newest one of its lineage with [`cross_sectional_carry_own`](@ref), because the step appends to its histories in place.
 2. Fold `rd` with [`cross_sectional_carry_fold`](@ref). The step succeeded, so record the state it returns as the newest one of its lineage in `tip`. A step that throws leaves the counter, and the next step of the same state writes the same spare rows.
 3. Rebuild the prior with the state.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator with no state, or with the state of its carry fold.
  - $(arg_dict[:rd]) Its `pnl` is a time-varying Asset Panel.

# Validation

  - The rules of [`cross_sectional_carry_fold`](@ref).

# Returns

  - `pe::CrossSectionalFactorPrior`: The prior, with its `cache` field set to the state after the last observation.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`CrossSectionalCarryState`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
  - [`refit_prior_step`](@ref)
  - [`Online`](@ref)
  - [`prior`](@ref)
  - [`AbstractChoiceRule`](@ref)
"""
function partial_fit!(pe::CrossSectionalFactorPrior{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any, <:Any,
                                                    <:Option{<:CrossSectionalCarryState}},
                      rd::ReturnsResult)
    st = if isnothing(pe.cache)
        CrossSectionalCarryState()
    else
        cross_sectional_carry_own(pe.cache)
    end
    st = cross_sectional_carry_fold(pe, st, rd)
    st.tip[] = st.buf.n
    return rebuild_estimator(pe, (; cache = st))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads a Cross-Sectional Factor Prior out of its carry fold, with no data.

The call with no data builds the Prior Result with [`cross_sectional_assemble`](@ref), the code of the batch fit, from the histories that the state carries and the factor prior that it folded. The idiosyncratic correlation reads the `ce` that the state folded, or refits there from the carried rows when `ce` does not fold. The Return Forecast refits there too, unless the state carries its rows, as [`folds_forecast_rows`](@ref) answers. Then the call reads its Result off those rows with [`return_forecast_result`](@ref). The standardised idiosyncratic returns are the rows that the state carries, so the call does not standardise the history again.

# Algorithm

 1. Refuse a state that has not made its first fit.
 2. Append the observed factors of the fitted observations to the fitted factors with [`cross_sectional_carry_append`](@ref).
 3. Read the factor prior with [`cross_sectional_factor_prior`](@ref) over the factor returns of the factors that are not empty and the observed returns, and process and expand its moments with [`cross_sectional_factor_moments`](@ref). An Observed Factor is never empty.
 4. Count the residuals of each asset with [`variance_count`](@ref) on the folded `ve`.
 5. Take the returns data of the Return Forecast with [`cross_sectional_carry_forecast_returns`](@ref).
 6. Build the result with [`cross_sectional_assemble`](@ref), with the Return Forecast history and the standardised idiosyncratic returns that the state carries. The block records the position of each fitted row among the folded observations, and no timestamp, because the buffer keeps none.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state of its carry fold.
  - $(arg_dict[:strict]) It is forwarded to the factor prior.
  - `kwargs...`: Additional keyword arguments passed to the matrix processing and the lift.

# Validation

  - `st` holds a fit: `lag + 2` observations follow the Descriptor warm-up. An `ArgumentError` is thrown otherwise.
  - The rules of [`cross_sectional_assemble`](@ref).

# Returns

  - `pr::LowOrderPrior`: The prior that the folded observations state.

# Related

  - [`CrossSectionalCarryState`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_assemble`](@ref)
"""
function prior(pe::CrossSectionalFactorPrior, st::CrossSectionalCarryState;
               strict::Bool = false, kwargs...)
    Tf = isnothing(st.Ms) ? 0 : size(st.Ms, 1)
    @argcheck(!isnothing(st.csr),
              ArgumentError("the carry fold of this Cross-Sectional Factor Prior holds $Tf observation(s) after the Descriptor warm-up, and a fit needs at least lag + 2 = $(pe.lag + 2) of them. Fold more observations before the call with no data."))
    csr = st.csr
    (; r, ca) = cross_sectional_carry_append(pe, st)
    lv = vcat(st.lv, trues(size(ca.f, 2) - length(st.lv)))
    pr = cross_sectional_factor_prior(st.pe, all(lv) ? ca.f : ca.f[:, lv]; strict = strict)
    f_pr = cross_sectional_factor_moments(pr, pe.f_mp, ca.f, lv; kwargs...)
    # The histories hold every observation after the warm-up, the last `Tf` of the `buf.n`
    # the fold read, so the fitted rows sit at these positions of the folded returns.
    fit = (; csr = csr, W = st.W, vs = st.vs, cnt = variance_count(st.ve, csr.eps),
           amr = view(st.amsk, r, :), bwr = view(st.bw, r, :), Xo = view(st.X, r, :), r = r,
           hist = st.hist, S = st.S, Sc = st.Sc, ce = st.ce, idx = (st.buf.n - Tf) .+ r,
           ts = nothing, rf = if folds_forecast_rows(pe.rfe)
               return_forecast_result(pe.rfe, st.fh, st.fst)
           else
               nothing
           end)
    return cross_sectional_assemble(pe, f_pr, ca, fit,
                                    cross_sectional_carry_forecast_returns(pe, st);
                                    kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the returns data that the Return Forecast of the carry fold of a Cross-Sectional Factor Prior reads at the call with no data.

The batch fit gives the forecast the returns data that the estimated members read: the returns with the benchmark weights on the panel, and under observed factors the returns net of them. A forecast that reads the panel at the call with no data makes the state carry every row, so the carried rows and their derived series give that returns data, through [`cross_sectional_forecast_window`](@ref). Any other forecast reads no row, and gets the carried rows.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state of its carry fold.

# Returns

  - `rd::ReturnsResult`: The returns data of the forecast.

# Related

  - [`cross_sectional_forecast_reads_panel`](@ref)
  - [`cross_sectional_benchmark_stage`](@ref)
  - [`cross_sectional_exposure_series`](@ref)
"""
function cross_sectional_carry_forecast_returns(pe::CrossSectionalFactorPrior,
                                                st::CrossSectionalCarryState)
    if !cross_sectional_forecast_reads_panel(pe.rfe)
        return st.win
    end
    return cross_sectional_forecast_window(pe, st.win, st.der)
end
public carry_folds
