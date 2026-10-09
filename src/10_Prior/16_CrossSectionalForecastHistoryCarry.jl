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
    # The observed factor returns fill the reduced factor axis, so the model carries the
    # reduced loadings and the basis, as the batch fit does.
    fnow = cross_sectional_basis_now(ca.fcb, r)
    csfm = CrossSectionalFactorModel(; M = ca.Ms[end, :, :],
                                     L = cross_sectional_reduced_loadings(fnow, ca.L),
                                     b = zeros(real(eltype(st.vs)), size(st.X, 2)),
                                     csr = st.csr, Ms = ca.Ms, vs = st.vs, rw = st.W,
                                     bw = view(st.bw, r, :), nf = ca.nf, fam = ca.fam,
                                     fcb = fnow, unseen = pe.unseen, lag = pe.lag,
                                     fx = ca.fx)
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

Under a threshold `th` above zero, the idiosyncratic correlation reads the rows with no fill, and [`cross_sectional_carry_correlation`](@ref) brings them up. A `ce` that folds, as [`supports_partial_fit`](@ref) answers, folds the rows of the step with [`partial_fit!`](@ref), under the rule of [`cross_sectional_correlation_rows`](@ref) for the gaps. So the state keeps the folded `ce` and no row with no fill, and the call with no data reads the correlation from the state of `ce`. A step that fits every observation again folds every row into the `ce` of the prior again. A `ce` that does not fold makes the state keep the rows with no fill, and the call with no data estimates the correlation over them again.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state after the fit of the step.
  - `stepped`: `true` when the step fitted the new observations alone, `false` when it fitted every observation.

# Returns

  - `st::CrossSectionalCarryState`: The state with `S` at every fitted observation. When the threshold `th` of the prior is not zero, it also holds the folded `ce`, or `Sc` at every fitted observation when `ce` does not fold. A state with no fit is returned as it is.

# Related

  - [`cross_sectional_standardised_residuals`](@ref)
  - [`cross_sectional_carry_correlation`](@ref)
  - [`cross_sectional_correlation_rows`](@ref)
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
    # A step that fits every observation again makes every row again, and folds them into
    # the `ce` of the prior.
    S0, Sc0, ce0 = iszero(k) ? (nothing, nothing, pe.ce) : (st.S, st.Sc, st.ce)
    S = cross_sectional_fold_append(S0, cross_sectional_standardised_residuals(E, V, A))
    return cross_sectional_carry_with(st,
                                      (; S = S,
                                       cross_sectional_carry_correlation(pe, Sc0, ce0, E, V,
                                                                         A)...))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Brings the input of the idiosyncratic correlation of the carry fold of a Cross-Sectional Factor Prior up to its fitted observations.

[`cross_sectional_carry_standardised`](@ref) calls it with the rows of a step. Under a threshold `th` above zero, the correlation reads the standardised idiosyncratic returns with no fill.

# Algorithm

 1. Return no rows and no estimator when `th` is zero, because the correlation reads no residual.
 2. Standardise the rows of the step with no fill with [`cross_sectional_standardised_residuals`](@ref).
 3. When `pe.ce` does not fold, as [`supports_partial_fit`](@ref) answers, append the rows to `Sc0`. The call with no data estimates the correlation over them again.
 4. Otherwise, fold the rows into `ce0` with [`partial_fit!`](@ref), under the rule of [`cross_sectional_correlation_rows`](@ref) for the gaps, and keep no row. The call with no data reads the correlation from the state of `ce`.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `Sc0`: The rows with no fill that the state carries, or `nothing`.
  - `ce0`: The folded `ce` that the state carries, or `pe.ce` at a fit of every observation.
  - `E`: The idiosyncratic returns of the rows of the step.
  - `V`: The idiosyncratic variances of the rows of the step.
  - `A`: The active mask of the rows of the step.

# Returns

  - `(; Sc, ce)::NamedTuple`: The fields `Sc` and `ce` of [`CrossSectionalCarryState`](@ref).

# Related

  - [`cross_sectional_carry_standardised`](@ref)
  - [`cross_sectional_correlation_rows`](@ref)
  - [`cross_sectional_idiosyncratic_covariance`](@ref)
"""
function cross_sectional_carry_correlation(pe::CrossSectionalFactorPrior, Sc0, ce0,
                                           E::MatNum, V::MatNum, A::AbstractMatrix{<:Bool})
    if iszero(pe.th)
        return (; Sc = nothing, ce = nothing)
    end
    Sc = cross_sectional_standardised_residuals(E, V, A; filled = false)
    if !supports_partial_fit(pe.ce)
        return (; Sc = cross_sectional_fold_append(Sc0, Sc), ce = nothing)
    end
    # A `ce` that folds carries the correlation, so the state keeps no row with no fill (#1594).
    (; X, kw) = cross_sectional_correlation_rows(ce0, Sc, A)
    return (; Sc = nothing, ce = partial_fit!(ce0, X; dims = 1, kw...))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the rows and the keyword arguments with which a covariance estimator reads the standardised idiosyncratic returns.

The batch fit passes them to `Statistics.cov`, and the carry fold passes the rows of each step to [`partial_fit!`](@ref). So the two routes read the gaps of the rows by one rule.

# Algorithm

 1. Get `fv` from `ce` with [`gap_fill_value`](@ref). `fv` is the value that `ce` gives a gapped cell of `S`. A cell of `S` is non-finite where the asset is inactive, and where it is active with no finite standardised return: a missing return, or a variance in its warm-up.
 2. If `fv` is finite, write it over every non-finite cell of a copy of `S`, and give no keyword argument. The fallback `fv` is zero, which is the mean of a standardised series.
 3. If `fv` is not finite, give `S` as it stands, with `amsk` as the `active_mask`. A gap-aware `ce` then freezes the block of an inactive asset and does not decay it, and it takes an active non-finite cell as a holiday.

# Arguments

  - `ce`: Covariance estimator of the standardised idiosyncratic returns.
  - `S`: Standardised idiosyncratic returns with no fill, `observations × assets`.
  - `amsk`: The active mask of the rows of `S`, `observations × assets`.

# Returns

  - `(; X, kw)::NamedTuple`: The rows `X` that `ce` reads, and the keyword arguments `kw` that go with them.

# Related

  - [`cross_sectional_idiosyncratic_covariance`](@ref)
  - [`cross_sectional_carry_standardised`](@ref)
  - [`gap_fill_value`](@ref)
"""
function cross_sectional_correlation_rows(ce::StatsBase.CovarianceEstimator, S::MatNum,
                                          amsk::AbstractMatrix{<:Bool})
    fv = gap_fill_value(ce)
    if !isfinite(fv)
        return (; X = S, kw = (; active_mask = amsk))
    end
    Z = Matrix{real(eltype(S))}(S)
    for k in CartesianIndices(Z)
        if !isfinite(Z[k])
            Z[k] = fv
        end
    end
    return (; X = Z, kw = (;))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the idiosyncratic covariance of the latest observation from the covariance of the standardised idiosyncratic returns.

[`cross_sectional_idiosyncratic_covariance`](@ref) states the definition. The batch fit and the carry fold estimate `C` by different routes, and both reach this function.

# Algorithm

 1. Convert `C` to the correlation `R`.
 2. Set to zero every entry of `R` off the diagonal whose magnitude does not exceed `th`, and set the diagonal to one. This step also sets a non-finite correlation to zero.
 3. Rescale `R` by the latest idiosyncratic volatilities, giving `D`. Copy the lower triangle of `D` into the upper one, so `D` is exactly symmetric.
 4. Make the block of `D` over the assets with a finite variance positive definite with [`posdef!`](@ref).

# Arguments

  - `th`: The correlation threshold, above zero.
  - `pdm`: Positive definite matrix estimator, or `nothing`.
  - `C`: The covariance of the standardised idiosyncratic returns, `assets × assets`.
  - `ev`: The latest idiosyncratic variances, one per asset.

# Returns

  - `D::MatNum`: The idiosyncratic covariance.

# Related

  - [`cross_sectional_idiosyncratic_covariance`](@ref)
  - [`posdef!`](@ref)
"""
function cross_sectional_thresholded_covariance(th::Real,
                                                pdm::Option{<:AbstractPosdefEstimator},
                                                C::MatNum, ev::VecNum)
    s = sqrt.(LinearAlgebra.diag(C))
    R = StatsBase.cov2cor(Matrix(C), s)
    for k in CartesianIndices(R)
        if k[1] != k[2] && !(abs(R[k]) > th)
            R[k] = zero(eltype(R))
        end
    end
    for i in axes(R, 1)
        R[i, i] = one(eltype(R))
    end
    se = sqrt.(ev)
    # The two triangles multiply in a different order, so they differ by round-off. The clip
    # accepts such a block, and its square root then refuses it as not Hermitian. Each repair
    # reads the lower triangle, so the copy leaves the input of a repair as it was.
    D = Matrix(LinearAlgebra.Symmetric(R .* se .* transpose(se), :L))
    idx = findall(isfinite, ev)
    if !isempty(idx)
        B = D[idx, idx]
        posdef!(pdm, B)
        D[idx, idx] = B
    end
    return D
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

A Return Forecast that computes its history one observation at a time, as [`folds_forecast_rows`](@ref) answers, reads its scores at each fitted observation. A score reads the last [`lookback`](@ref) panel rows of its observation and no factor-model block, so a step computes the scores of its new observations alone, from the rows that it carries. A Descriptor that carries a state reads the new observations alone, as [`descriptor_carry`](@ref) folds them, so it adds no row: the state carries the Descriptor Scores with the state of each such Descriptor. A fit of every observation then reads the scores that the state carries, and no panel row before the carried ones. A member that trains on rows before the block also reads the last rows of the warm-up that [`cross_sectional_forecast_lead`](@ref) counts, so the state keeps their scores before the scores of the first row of the histories.

# Algorithm

 1. Return `st` unchanged when the forecast does not fold.
 2. Take the Descriptor Scores that the state carries, or the ones of the forecast at the first step. Keep the last `carry_lookback(ds.descriptors) + n - 1` rows of `win` and of its derived series, as [`carry_lookback`](@ref) counts them, and build the returns data of the forecast over them with [`cross_sectional_forecast_window`](@ref).
 3. Fold the `n` rows of the step into the Descriptors that carry a state with [`descriptor_carry`](@ref).
 4. Count the rows of the step to score: the `m` observations after the warm-up and the last `a` rows of the warm-up before them, as [`cross_sectional_forecast_lead`](@ref) counts them, up to `n`. When the histories hold a row before the step, every row of the step follows the warm-up, so the count is `m`. When the count is zero, return the state with the folded Descriptor Scores alone.
 5. Score the Descriptors of the forecast over those rows with [`descriptor_panel_scores`](@ref), with the Descriptors that the fold reads off the states, and keep the counted rows.
 6. Append them to the scores that the state carries with [`cross_sectional_fold_append`](@ref), and keep the last `a` rows before the histories and every row of the histories with [`cross_sectional_window_trim`](@ref). Keep the folded Descriptor Scores.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state, with the rows of the step appended to its histories.
  - `win`: The carried rows followed by the rows of the step.
  - `rows`: The rows of the step, as [`cross_sectional_fold_rows`](@ref) returns them. Its `Ms` holds the `m` new observations after the warm-up, the last rows of `win`, and its `der` the derived series of every row of `win`, or `nothing` without an observed factor.
  - `n`: Number of rows of the step, the last rows of `win`.

# Returns

  - `st::CrossSectionalCarryState`: The state with the scores of the new observations appended, and the Descriptor Scores after the rows of the step in `fds`.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_carry_forecast`](@ref)
  - [`cross_sectional_forecast_lead`](@ref)
"""
function cross_sectional_carry_scores(pe::CrossSectionalFactorPrior,
                                      st::CrossSectionalCarryState, win::ReturnsResult,
                                      rows::NamedTuple, n::Integer)
    if !folds_forecast_rows(pe.rfe)
        return st
    end
    ds = something(st.fds, pe.rfe.scores)
    der = rows.der
    T = size(win.X, 1)
    # A Descriptor that carries a state reads the rows of the step alone, so the window holds
    # the rows that the other Descriptors read before the rows of the step.
    k = min(T, something(carry_lookback(ds.descriptors), T) + n - 1)
    r = (T - k + 1):T
    dk = isnothing(der) ? nothing : (; Xl = view(der.Xl, r, :))
    rdk = cross_sectional_forecast_window(pe, k == T ? win : port_opt_view(win, r, :), dk)
    dc = descriptor_carry(ds, rdk, n)
    m = size(rows.Ms, 1)
    # The state also keeps the scores of the last `a` rows of the warm-up, which a member that
    # trains on rows before the block reads. Once the histories hold a row before the step,
    # every row of the step follows the warm-up, so `m == n` and `c == m`.
    a = cross_sectional_forecast_lead(pe.rfe, pe.lag)
    c = min(n, m + a)
    if iszero(c)
        return cross_sectional_carry_with(st, (; fds = dc.xf))
    end
    P = descriptor_panel_scores(dc.xv, rdk)
    Pn = map(A -> return_forecast_cut(A, (k - c + 1):k), P)
    # The bound stops JET from pairing an untyped `st.fsc` with the `map` method of another
    # package.
    fsc = isnothing(st.fsc) ? Pn : map(cross_sectional_fold_append, st.fsc::NamedTuple, Pn)
    # The scores hold the rows of the histories and at most `a` rows before them, so the cut
    # drops rows of the warm-up alone, until the histories hold a row before the step.
    h = size(something(st.Ms, rows.Ms), 1) + a
    return cross_sectional_carry_with(st,
                                      (; fsc = cross_sectional_window_trim(fsc, h),
                                       fds = dc.xf))
end
function carry_lookback(rfe::Union{FixedWeightedReturnForecast, ExpWeightedReturnForecast,
                                   TargetReturnForecast})::Option{<:Integer}
    return carry_lookback(rfe.scores.descriptors)
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
    fnow = cross_sectional_basis_now(ca.fcb, r[j])
    csfm = CrossSectionalFactorModel(; M = ca.Ms[end, :, :],
                                     L = cross_sectional_reduced_loadings(fnow, ca.L),
                                     b = zeros(real(eltype(st.vs)), size(st.X, 2)),
                                     csr = CrossSectionalRegression(; f = csr.f[j, :],
                                                                    eps = csr.eps[j, :],
                                                                    n = csr.n[j],
                                                                    h1 = nothing_scalar_array_getindex_odd_order(csr.h1,
                                                                                                                 j,
                                                                                                                 :)),
                                     Ms = ca.Ms[j, :, :], vs = st.vs[j, :], rw = st.W[j, :],
                                     bw = st.bw[r[j], :], nf = ca.nf, fam = ca.fam,
                                     fcb = fnow, unseen = pe.unseen, lag = pe.lag,
                                     fx = nothing_scalar_array_getindex_odd_order(ca.fx, j,
                                                                                  :))
    rfo = orthogonal_forecast_member(pe.ofit, pe.rfe, csfm)
    # The scores sit on the histories after the warm-up, and the fitted rows after the first
    # `lag` of them. The cut copies, so the step can neutralise the copy in place.
    # A fit of every observation makes every row again, from an empty fold state.
    H0, fs0 = iszero(k) ? (nothing, nothing) : (st.fh, st.fst)
    # The scores also hold the rows of the warm-up that `cross_sectional_forecast_lead` counts,
    # before the first row of the histories.
    fsc::NamedTuple = st.fsc
    i = cross_sectional_forecast_rows(rfo, r, k, g, size(fsc.S, 1) - size(st.Ms, 1))
    (; hist, fs) = orthogonal_forecast_step(pe.ofit, rfo,
                                            map(A -> return_forecast_cut(A, i), fsc), csfm,
                                            fs0, pe.cre)
    # The new rows are the last rows of the history of the step.
    H = view(hist, (size(hist, 1) - length(r) + k + 1):size(hist, 1), :)
    return cross_sectional_carry_with(st,
                                      (; fh = cross_sectional_fold_append(H0, H), fst = fs))
end
"""
    cross_sectional_forecast_rows(rfe, r::AbstractUnitRange, k::Integer, g::Integer,
                                  a::Integer)
    cross_sectional_forecast_rows(rfe::TargetReturnForecast, r::AbstractUnitRange,
                                  k::Integer, g::Integer, a::Integer)

Returns the rows of the carried Descriptor scores that a step of the carry fold of a Cross-Sectional Factor Prior hands its Return Forecast.

The step hands the member its new observations, and before them the observations whose target matures at the step: the last `g` observations that the member read, or all of them when it read fewer. The scores start `a` rows before the histories after the warm-up, as [`cross_sectional_forecast_lead`](@ref) counts them, so row `t` of the histories is row `a + t` of the scores.

# Algorithm

The method that Julia selects is the algorithm.

 1. A member that reads the block alone: the fitted observations after the first `k`, and the last `g` of the first `k`, as rows of the scores.
 2. [`TargetReturnForecast`](@ref) under `whole_history`: the member also trains on a row before the block whose forward window reaches into the block, so its observations are the rows of the scores. Return the rows after the first `k` fitted observations, and the `g` rows before them, or every row before them when there are fewer. Without `whole_history`, method 1.

# Arguments

  - `rfe`: The Return Forecast Estimator that the step fits.
  - `r`: The fitted observations, as rows of the histories after the warm-up.
  - `k`: Number of fitted observations that the state keeps.
  - `g`: The observations that a target takes to mature, as [`forecast_target_gap`](@ref) answers.
  - `a`: Number of rows of the warm-up that the scores hold before the histories.

# Returns

  - `rows::AbstractUnitRange`: Rows of the carried scores.

# Related

  - [`cross_sectional_forecast_lead`](@ref)
  - [`cross_sectional_carry_forecast`](@ref)
  - [`return_forecast_step`](@ref)
"""
function cross_sectional_forecast_rows(::Any, r::AbstractUnitRange, k::Integer, g::Integer,
                                       a::Integer)
    return (a + r[k + 1 - min(k, g)]):(a + last(r))
end
function cross_sectional_forecast_rows(rfe::TargetReturnForecast, r::AbstractUnitRange,
                                       k::Integer, g::Integer, a::Integer)
    i = rfe.whole_history ? max(1, a + r[k + 1] - g) : a + r[k + 1 - min(k, g)]
    return i:(a + last(r))
end
"""
    cross_sectional_forecast_lead(rfe, lag::Integer)
    cross_sectional_forecast_lead(rfe::TargetReturnForecast, lag::Integer)

Returns the number of rows of the Descriptor warm-up whose scores the carry fold of a Cross-Sectional Factor Prior keeps for its Return Forecast.

The histories of the carry start `lag` rows before the first fitted observation. A member that trains on a row before the block trains on the last [`forecast_target_gap`](@ref) rows before it, because their forward windows reach into the block. When the gap is longer than `lag`, the first of those rows are rows of the warm-up, and the histories hold no such row. So the state keeps their scores, once, before the scores of the first row of the histories.

# Algorithm

The method that Julia selects is the algorithm.

 1. A member that reads the block alone trains on no row before it. Return zero.
 2. [`TargetReturnForecast`](@ref) under `whole_history`: return `forecast_target_gap(rfe) - lag`, or zero when the gap is not longer than `lag`. Without `whole_history`, method 1.

# Arguments

  - `rfe`: The Return Forecast Estimator that the step fits.
  - `lag`: The lag of the exposures of the prior, `pe.lag`.

# Returns

  - `a::Integer`: The number of rows of the warm-up.

# Related

  - [`cross_sectional_carry_scores`](@ref)
  - [`cross_sectional_forecast_rows`](@ref)
"""
function cross_sectional_forecast_lead(::Any, ::Integer)
    return 0
end
function cross_sectional_forecast_lead(rfe::TargetReturnForecast, lag::Integer)
    return rfe.whole_history ? max(forecast_target_gap(rfe) - lag, 0) : 0
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
