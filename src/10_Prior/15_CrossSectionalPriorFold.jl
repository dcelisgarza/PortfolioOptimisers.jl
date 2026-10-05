"""
$(DocStringExtensions.TYPEDEF)

Carries the carry fold of a Cross-Sectional Factor Prior between two online steps.

A [`CrossSectionalFactorPrior`](@ref) with no `cache` seeds this state at its first [`partial_fit!`](@ref), as the carry of [`EmpiricalPrior`](@ref) seeds a [`PriorCarryState`](@ref). The state applies the rule of the carry fold: it folds what folds and refits the rest. A new observation changes no past Factor Exposure, no past regression and no past idiosyncratic variance, so a step computes them for the new observations alone. The idiosyncratic variance and the factor prior fold exactly. The Return Forecast and the idiosyncratic correlation refit at the call with no data from the carried rows.

Two choices read every fitted observation: the automatic dropped member of a Factor Family under a [`BatchChoice`](@ref), and the mark of the Empty Factors. When a step moves either one, the step fits every carried observation again.

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
                             seed::Integer = 0, hist = nothing) -> CrossSectionalCarryState

Keywords correspond to the struct's fields, and every field but `buf` defaults to `nothing`, or to `0` for `seed`. The default is the empty state that a first step builds.

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
    The derived series of the carried rows, `(; Xn, Xl)`, or `nothing` without an observed factor. `Xn` holds the returns net of the observed members that read no returns, which the other observed members read, or `nothing` when no member needs them. `Xl` holds the returns net of every observed factor, which the estimated members read.
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
    Observed factors of every observation after the warm-up, `(; Z, R, lv, nf, fam)` as [`cross_sectional_observed`](@ref) states them, or `nothing` without an observed factor. The factor prior reads the observed returns `R` beside the factor returns of the regression.
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
end
function CrossSectionalCarryState(; buf::SampleBufferState = SampleBufferState(),
                                  win::Option{<:ReturnsResult} = nothing, der = nothing,
                                  nf = nothing, fam = nothing, Ms = nothing, X = nothing,
                                  Xl = nothing, obs = nothing, bw = nothing, mcap = nothing,
                                  amsk = nothing, emsk = nothing, sums = nothing,
                                  families = nothing, fcb = nothing, Z = nothing,
                                  csr = nothing, lv1 = nothing, lv = nothing, W = nothing,
                                  vs = nothing, ve = nothing, ve1 = nothing, pe = nothing,
                                  seed::Integer = 0,
                                  hist = nothing)::CrossSectionalCarryState
    return CrossSectionalCarryState(buf, win, der, nf, fam, Ms, X, Xl, obs, bw, mcap, amsk,
                                    emsk, sums, families, fcb, Z, csr, lv1, lv, W, vs, ve,
                                    ve1, pe, seed, hist)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuilds a [`CrossSectionalCarryState`](@ref) with some of its fields replaced.

# Arguments

  - `st`: The state.
  - `kw`: The fields to replace, by name.

# Returns

  - `st::CrossSectionalCarryState`: The new state.

# Related

  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_carry_with(st::CrossSectionalCarryState, kw::NamedTuple)
    fns = fieldnames(CrossSectionalCarryState)
    return CrossSectionalCarryState(;
                                    merge(NamedTuple{fns}(map(f -> getfield(st, f), fns)),
                                          kw)...)
end
function returns_buffer(state::CrossSectionalCarryState)
    return state.buf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`CrossSectionalCarryState`](@ref), so that the copy shares no array and no folded state with the original.

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
    return CrossSectionalCarryState(;
                                    NamedTuple{fns}(map(f -> deepcopy(getfield(x, f)), fns))...)
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

A Return Forecast that reads the panel refits at each call with no data from the carried rows. Its Result keeps a value at every fitted observation, and [`return_forecast_rows`](@ref) aligns the fitted observations with the last rows of the returns data, so the carry fold carries every panel row under it. A [`CustomValueReturnForecast`](@ref) and an absent forecast read no row.

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
function cross_sectional_forecast_reads_panel(::AbstractReturnForecastEstimator)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the number of panel rows that the carry fold of a Cross-Sectional Factor Prior carries, or `nothing` for every row.

The Factor Exposures of a new observation read the last [`lookback`](@ref) rows of the panel, so the carry keeps them. Under a Return Forecast that reads the panel, as [`cross_sectional_forecast_reads_panel`](@ref) answers, the carry keeps every row. An unbounded look-back keeps every row too.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.

# Returns

  - `n::Option{<:Integer}`: The number of rows, or `nothing`.

# Related

  - [`lookback`](@ref)
  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_carry_rows(pe::CrossSectionalFactorPrior)::Option{<:Integer}
    return cross_sectional_forecast_reads_panel(pe.rfe) ? nothing : lookback(pe)
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

Keeps the last `n` rows of the panel rows of a step, or of their derived series, or every row when `n` is `nothing`.

# Arguments

  - `rd`: The panel rows, a matrix of rows, the derived series `(; Xn, Xl)` of the rows, or `nothing`.
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
function cross_sectional_window_trim(A::AbstractMatrix, ::Nothing)
    return A
end
function cross_sectional_window_trim(A::AbstractMatrix, n::Integer)
    T = size(A, 1)
    return T <= n ? A : A[(T - n + 1):T, :]
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

An [`EmpiricalPrior`](@ref) folds on its carry route. Any other factor prior does not fold, and the call with no data refits it over the factor returns that the state carries.

# Arguments

  - `pe`: The factor prior, folded so far.
  - `f`: The factor returns of the new observations, on the factors that are not empty.

# Returns

  - `pe`: The factor prior after the new observations.

# Related

  - [`cross_sectional_factor_prior`](@ref)
  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_fold_factors(pe::EmpiricalPrior, f::MatNum)
    return partial_fit!(pe, f)
end
function cross_sectional_fold_factors(pe::AbstractPriorEstimator, ::MatNum)
    return pe
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the factor prior of the carry fold out of its state.

A folded [`EmpiricalPrior`](@ref) answers from its carry state. Any other factor prior refits over the factor returns `f`.

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
function cross_sectional_factor_prior(pe::EmpiricalPrior{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:PriorCarryState}, ::MatNum;
                                      strict::Bool = false)
    return prior(pe; strict = strict)
end
function cross_sectional_factor_prior(pe::AbstractPriorEstimator, f::MatNum;
                                      strict::Bool = false)
    return prior(pe, f; strict = strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs the regression passes of a Cross-Sectional Factor Prior over a block of observations.

The batch fit runs the same passes over every observation. Each observation is fitted on its own row, so a block gives the rows that the batch fit gives, when the masks of the Empty Factors are the masks of every fitted observation. A first fit or a fit of every carried observation states no mark, and the function takes the mark of the block. A step states the marks of the observations it fitted. The function answers `nothing` when the block marks a factor that those marks leave empty, because every observation must then be fitted again.

# Algorithm

 1. Take the eligibility mask `msk` with [`cross_sectional_eligible`](@ref), drop every pair whose lagged market capitalisation is not finite, and refuse an observation below [`cross_sectional_minra`](@ref).
 2. Take the first-pass weights with [`cs_weights_initial`](@ref), and run the first pass with [`cross_sectional_fold_pass`](@ref).
 3. When [`needs_second_pass`](@ref) answers `true`, fold `ve1` over the first-pass residuals with [`cross_sectional_fold_variance`](@ref). Invert the variance of the row before each observation with [`cross_sectional_lagged_inverse`](@ref), blend the weights with [`cs_weights_blend`](@ref), and run the second pass with [`cross_sectional_fold_pass`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `blk`: The observations: `Zl`, the lagged reduced exposures; `Xr`, the returns; `em` and `am`, the masks; and `mcl`, the lagged market capitalisation or `nothing`.
  - `prev`: The marks `lv1` and `lv`, the folded estimator `ve1` and the number `seed` of rows to fold as one block.

# Validation

  - The rules of every verb the algorithm names.

# Returns

  - `fit::Option{<:NamedTuple}`: `csr`, `W`, `lv1`, `lv` and `ve1`, or `nothing` when the block marks a new factor.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`prior`](@ref)
"""
function cross_sectional_fold_regression(pe::CrossSectionalFactorPrior, blk::NamedTuple,
                                         prev::NamedTuple)
    (; Zl, Xr, em, am, mcl) = blk
    msk = cross_sectional_eligible(Xr, Zl, em)
    cross_sectional_cap_finite!(msk, mcl)
    assert_cross_sectional_coverage(msk, cross_sectional_minra(pe, size(Zl, 3)))
    W = cs_weights_initial(pe.wa, mcl, msk)
    p1 = cross_sectional_fold_pass(pe, Zl, Xr, W, prev.lv1)
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
    p2 = cross_sectional_fold_pass(pe, Zl, Xr, W, prev.lv)
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

 1. Join the mark of the factors that are not empty over the block to `old` with [`cross_sectional_fold_mark`](@ref). Answer `nothing` when the block marks a new factor.
 2. Regress on the marked factors with [`cross_sectional_live_regression`](@ref), and refuse an intercept with [`assert_cross_sectional_no_intercept`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `Zl`: Lagged reduced exposures of the block.
  - `Xr`: Returns of the block.
  - `W`: Regression weights of the block.
  - `old`: The mark of the fitted observations, or `nothing` for a first fit.

# Returns

  - `pass::Option{<:NamedTuple}`: `csr`, the regression, and `lv`, the mark, or `nothing`.

# Related

  - [`cross_sectional_fold_regression`](@ref)
"""
function cross_sectional_fold_pass(pe::CrossSectionalFactorPrior, Zl::Arr3Num, Xr::MatNum,
                                   W::MatNum, old::Option{<:BitVector})
    lv = cross_sectional_fold_mark(old, cross_sectional_live_factors(Zl, Xr, W))
    return if isnothing(lv)
        nothing
    else
        (;
         csr = assert_cross_sectional_no_intercept(cross_sectional_live_regression(pe.cre,
                                                                                   Zl, Xr,
                                                                                   W, lv).csr),
         lv = lv)
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

# Arguments

  - `old`: The mark of the fitted observations, or `nothing` for a first fit.
  - `new`: The mark of the block.

# Validation

  - Without `old`, at least one factor is not empty. Raises an `ArgumentError`.

# Returns

  - `lv::Option{BitVector}`: `new` for a first fit, `old` when the block marks no other factor, and `nothing` when it does.

# Related

  - [`cross_sectional_fold_regression`](@ref)
  - [`cross_sectional_live_factors`](@ref)
"""
function cross_sectional_fold_mark(::Nothing, new::BitVector)
    @argcheck(any(new),
              ArgumentError("every one of the $(length(new)) factors is empty: no factor has a nonzero exposure at an (observation, asset) pair of positive weight, so the regression has nothing to fit. Widen the eligible cross-section, or give factors that the assets load on."))
    return new
end
function cross_sectional_fold_mark(old::BitVector, new::BitVector)
    return any(i -> new[i] && !old[i], eachindex(new, old)) ? nothing : old
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Appends the rows of a block to a history, or starts the history with them.

# Arguments

  - `a`: The history, or `nothing`.
  - `b`: The rows of the block. The observed factors `(; Z, R, lv, nf, fam)` of a block append their exposures and their returns, and keep the names of `a`.

# Returns

  - `h`: `a` followed by `b` along the observation axis.

# Related

  - [`cross_sectional_carry_fold`](@ref)
"""
function cross_sectional_fold_append(::Nothing, b)
    return b
end
function cross_sectional_fold_append(a::AbstractMatrix, b::AbstractMatrix)
    return vcat(a, b)
end
function cross_sectional_fold_append(a::AbstractArray{<:Any, 3}, b::AbstractArray{<:Any, 3})
    return cat(a, b; dims = 1)
end
function cross_sectional_fold_append(a::FactorFamilyBasis, b::FactorFamilyBasis)
    return FactorFamilyBasis(; fnm = a.fnm, fi = a.fi, di = a.di,
                             ratios = vcat(a.ratios, b.ratios), K = a.K)
end
function cross_sectional_fold_append(a::CrossSectionalRegression,
                                     b::CrossSectionalRegression)
    return CrossSectionalRegression(; f = vcat(a.f, b.f), eps = vcat(a.eps, b.eps),
                                    n = vcat(a.n, b.n), b = nothing, h1 = vcat(a.h1, b.h1))
end
function cross_sectional_fold_append(a::NamedTuple, b::NamedTuple)
    return (; Z = cat(a.Z, b.Z; dims = 1), R = vcat(a.R, b.R), lv = a.lv, nf = a.nf,
            fam = a.fam)
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

A pinned choice keeps the dropped members of the first fit, which the state records. A batch choice chooses each automatic member again over every observation after the warm-up. It reads the sums that the state carries, with the rule of [`factor_family_basis`](@ref): the member with the largest sum of absolute benchmark-weighted exposures, and the first such member on a tie. The state adds the rows to the sums in the order of the batch fit, so the two choose the same member.

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

The exposures of an observation read the last [`lookback`](@ref) rows of the panel, so the rows that the state carries give the exposures of the batch fit. The derived series of an observation read the observed exposures of the row `pe.lag` observations before it, which the carried rows hold too.

# Algorithm

 1. Run [`cross_sectional_exposure_series`](@ref) over the rows, with the derived series of the carried rows that the state holds. It gives the benchmark weights, the observed factors, the returns net of them and the exposures of the estimated factors, and it derives the rows of the step alone.
 2. Take the last `n` rows. Before the warm-up ends, drop the rows before the first one with an eligible asset on the estimated and the observed exposures together, as [`cross_sectional_warmup`](@ref) does.
 3. Neutralise the exposures of the rows that remain with [`cross_sectional_neutralise!`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state before the step.
  - `win`: The carried rows followed by the rows of the step.
  - `n`: Number of rows of the step.
  - `g0`: Number of observations before the first row of `win`.

# Returns

  - `rows::NamedTuple`: `nf` and `fam`, the raw factor axis; the rows after the warm-up: `Ms`, `X`, `Xl`, `obs`, `bw`, `mcap`, `amsk` and `emsk`, with `Xl` and `obs` `nothing` without an observed factor; and `der`, the derived series of every row of `win`, or `nothing`.

# Related

  - [`cross_sectional_carry_fold`](@ref)
  - [`cross_sectional_exposure_stage`](@ref)
"""
function cross_sectional_fold_rows(pe::CrossSectionalFactorPrior,
                                   st::CrossSectionalCarryState, win::ReturnsResult,
                                   n::Integer, g0::Integer)
    (; amsk, emsk, mcap, BW, Xu, cc, Xl, Ms, nf, fam) = cross_sectional_exposure_series(pe,
                                                                                        win.X,
                                                                                        nothing,
                                                                                        win.pnl;
                                                                                        ne = win.ne,
                                                                                        E = win.E,
                                                                                        kept = st.der,
                                                                                        g0 = g0)
    T = size(win.X, 1)
    q = (T - n + 1):T
    if isnothing(st.Ms)
        Mo = isnothing(cc) ? Ms[q, :, :] : cat(Ms[q, :, :], cc.Z[q, :, :]; dims = 3)
        elig = cross_sectional_eligible(Xu[q, :], Mo, emsk[q, :])
        q = q[something(findfirst(any, eachrow(elig)), length(q) + 1):end]
    end
    Msn = Ms[q, :, :]
    bwn = BW[q, :]
    if !isempty(q)
        cross_sectional_neutralise!(pe.neutralise, Msn, pe.cre, bwn, nf, fam)
    end
    obs = if isnothing(cc)
        nothing
    else
        (; Z = cc.Z[q, :, :], R = cc.R[q, :], lv = cc.lv, nf = cc.nf, fam = cc.fam)
    end
    return (; nf = nf, fam = fam, Ms = Msn, X = win.X[q, :],
            Xl = isnothing(cc) ? nothing : Xl[q, :], obs = obs, bw = bwn,
            mcap = cross_sectional_rows(mcap, q), amsk = amsk[q, :], emsk = emsk[q, :],
            der = isnothing(cc) ? nothing : (; Xn = cc.Xn, Xl = Xl))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fits every observation that the carry fold of a Cross-Sectional Factor Prior carries after the warm-up.

This is the first fit, and the fit after a step that moves the automatic choice of a dropped member or the mark of the Empty Factors. It runs the steps of the batch fit over the carried histories. The idiosyncratic variance and the factor prior fold the first `seed` observations as one block, and every later observation alone, so a [`SeedWindow`](@ref) cuts the rows that it cut at the first fit.

# Algorithm

 1. Build the Factor Family Basis of every observation with [`cross_sectional_family_basis`](@ref), under `families`, or under the families of `pe` at the first fit, and name its dropped members with [`cross_sectional_dropped_names`](@ref).
 2. Take the observed factors of every fitted observation with [`cross_sectional_observed_block`](@ref), which refuses a non-finite observed return.
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
           am = st.amsk[d, :], mcl = cross_sectional_rows(st.mcap, 1:D))
    reg = cross_sectional_fold_regression(pe, blk,
                                          (; lv1 = nothing, lv = nothing, ve1 = nothing,
                                           seed = seed))
    (; V, ve) = cross_sectional_fold_variance(pe.ve, reg.csr.eps, blk.em, blk.am, seed)
    f = cross_sectional_fold_factor_returns(reg.csr.f, reg.lv, cb)
    pf = cross_sectional_fold_factors(pe.pe, view(f, 1:seed, :))
    if seed < D
        pf = cross_sectional_fold_factors(pf, view(f, (seed + 1):D, :))
    end
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

Each output of a fitted observation reads no later observation, so the step fits the new observations alone and appends their outputs. It answers `nothing` when the new observations mark a factor that the fitted observations leave empty.

# Algorithm

 1. Reduce the exposures of the new observations with [`cross_sectional_family_basis`](@ref) under `families`, and lag them behind the reduced exposures of the last `lag` observations.
 2. Take the observed factors of the new observations with [`cross_sectional_observed_block`](@ref), which refuses a non-finite observed return.
 3. Run the regression passes over the new observations with [`cross_sectional_fold_regression`](@ref), under the marks of the fitted observations, on the returns net of the observed factors.
 4. Fold `ve` over the new residuals with [`cross_sectional_fold_variance`](@ref), and the factor prior over the new factor returns of [`cross_sectional_fold_factor_returns`](@ref) with [`cross_sectional_fold_factors`](@ref).
 5. Append every output to the state with [`cross_sectional_fold_append`](@ref).

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
    Zc = cat(st.Z, fb.Ms; dims = 1)
    cb = cross_sectional_observed_block(st.obs, 1:Tf, q, st.buf.n - Tf)
    blk = (; Zl = Zc[1:m, :, :], Xr = something(st.Xl, st.X)[q, :], em = st.emsk[q, :],
           am = st.amsk[q, :], mcl = cross_sectional_rows(st.mcap, q .- pe.lag))
    reg = cross_sectional_fold_regression(pe, blk,
                                          (; lv1 = st.lv1, lv = st.lv, ve1 = st.ve1,
                                           seed = 0))
    if isnothing(reg)
        return nothing
    end
    (; V, ve) = cross_sectional_fold_variance(st.ve, reg.csr.eps, blk.em, blk.am, 0)
    pf = cross_sectional_fold_factors(st.pe,
                                      cross_sectional_fold_factor_returns(reg.csr.f, reg.lv,
                                                                          cb))
    return cross_sectional_carry_with(st,
                                      (; fcb = cross_sectional_fold_append(st.fcb, fb.fcb),
                                       Z = Zc[(m + 1):end, :, :],
                                       csr = cross_sectional_fold_append(st.csr, reg.csr),
                                       W = vcat(st.W, reg.W), vs = vcat(st.vs, V), ve = ve,
                                       ve1 = reg.ve1, pe = pf))
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
"""
function cross_sectional_fold_histories(pe::CrossSectionalFactorPrior,
                                        st::CrossSectionalCarryState, rows::NamedTuple)
    if iszero(size(rows.Ms, 1))
        return cross_sectional_carry_with(st, (; nf = rows.nf, fam = rows.fam))
    end
    return cross_sectional_carry_with(st,
                                      (; nf = rows.nf, fam = rows.fam,
                                       Ms = cross_sectional_fold_append(st.Ms, rows.Ms),
                                       X = cross_sectional_fold_append(st.X, rows.X),
                                       Xl = cross_sectional_fold_append(st.Xl, rows.Xl),
                                       obs = cross_sectional_fold_append(st.obs, rows.obs),
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
 2. Append them to the fitted factors with [`cross_sectional_observed_append`](@ref), giving the raw exposures of the fitted observations and the reduced loadings of the last one.

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
    ca = cross_sectional_observed_append(cb, st.csr.f, st.Z[end, :, :], st.Ms[r, :, :],
                                         st.nf, st.fam, st.fcb)
    return (; r = r, ca = ca)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of a [`ReturnsResult`](@ref) into the carry fold of a Cross-Sectional Factor Prior.

This is the step of a prior that carries a [`CrossSectionalCarryState`](@ref). It applies the rule of the carry fold: the exposures, the observed factors, the returns net of them, the regression and the idiosyncratic variance of a new observation read no later observation, so the step computes them for the new observations alone.

# Algorithm

 1. Refuse returns data that no fold takes with [`assert_prior_fold_returns`](@ref), and a panel that is absent or static. Take the Exogenous Series of the step with [`exogenous_step_kwargs`](@ref), which refuses a step with no series when the prior reads it.
 2. Append the rows of the step and their series to the carried panel rows with [`cross_sectional_window_append`](@ref).
 3. Compute the neutralised exposures, the observed factors and the derived series of the new observations with [`cross_sectional_fold_rows`](@ref), and append the rows after the warm-up to the histories with [`cross_sectional_fold_histories`](@ref). Append the returns, both masks and the series to the buffer `buf`. Keep the panel rows and their derived series that [`cross_sectional_carry_rows`](@ref) names.
 4. Before the first fit, fit every observation with [`cross_sectional_fold_refit`](@ref) once `lag + 2` observations follow the warm-up, as the batch fit needs.
 5. After it, apply the Choice Rule with [`cross_sectional_fold_choice`](@ref). When the choice moves, fit every observation again with [`cross_sectional_fold_refit`](@ref). Otherwise fit the new observations with [`cross_sectional_fold_step`](@ref), and fit every observation again when it answers `nothing`. A variance estimator `pe.ve` that does not fold, as [`supports_partial_fit`](@ref) answers, carries no state, so every step fits every observation again.
 6. Bring the Return Forecast history up to the fitted observations with [`cross_sectional_carry_history`](@ref). It keeps the carried rows after a step of [`cross_sectional_fold_step`](@ref), and makes every row again after a fit of every observation.

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
    st = cross_sectional_fold_histories(pe, st, rows)
    n = cross_sectional_carry_rows(pe)
    st = cross_sectional_carry_with(st,
                                    (;
                                     buf = partial_fit!(st.buf, rd.X;
                                                        active_mask = pnl.amsk,
                                                        estimation_mask = pnl.emsk, xk...),
                                     win = cross_sectional_window_trim(win, n),
                                     der = cross_sectional_window_trim(rows.der, n)))
    Tf = isnothing(st.Ms) ? 0 : size(st.Ms, 1)
    if isnothing(st.csr)
        return if Tf - pe.lag < 2
            st
        else
            cross_sectional_carry_history(pe,
                                          cross_sectional_fold_refit(pe, st, nothing,
                                                                     Tf - pe.lag), false)
        end
    end
    # After the first fit every row follows the warm-up, and a `ReturnsResult` holds at least
    # one row, so the step brings `m >= 1` new observations. A variance estimator that does
    # not fold has no state to carry, so every step fits every observation again, over the
    # carried histories.
    families = cross_sectional_fold_choice(pe.choice, pe, st)
    stepped = if families == st.families && supports_partial_fit(pe.ve)
        cross_sectional_fold_step(pe, st, families, m)
    else
        nothing
    end
    return if isnothing(stepped)
        cross_sectional_carry_history(pe,
                                      cross_sectional_fold_refit(pe, st, families, st.seed),
                                      false)
    else
        cross_sectional_carry_history(pe, stepped, true)
    end
end
"""
    partial_fit!(pe::CrossSectionalFactorPrior{<:Any, …, <:Option{<:CrossSectionalCarryState}},
                 rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into a Cross-Sectional Factor Prior on its carry route.

An unwrapped prior follows the rule of the carry fold: it folds what folds and refits the rest. Its first step seeds a [`CrossSectionalCarryState`](@ref), as the carry of [`EmpiricalPrior`](@ref) seeds its own state. Each step computes the Factor Exposures and the observed factors of the new observations from the carried panel rows, runs the regression on the new observations alone, and folds the idiosyncratic variance and the factor prior. A prior whose tree reads the Exogenous Series, as [`reads_exogenous_series`](@ref) answers, records it, so each step must bring it. [`cross_sectional_carry_fold`](@ref) states the step. The call with no data `prior(pe)` equals the batch fit over the folded observations up to rounding, until a [`SeedWindow`](@ref) or a [`PinnedChoice`](@ref) keeps a value of the first fit. A pinned choice is recorded in the state, and `families` stays as the caller wrote it.

[`Online`](@ref) seeds the `cache` with a [`SampleBufferState`](@ref) instead, and the prior refits through the generic method of `partial_fit!` and [`refit_prior_step`](@ref). The method narrows `cache` to the state of the carry fold, so a buffer reaches that route. The matrix form refuses through the generic method, because a matrix carries no Asset Panel, and the prior reads its Factor Exposures off one.

# Algorithm

 1. Seed an empty [`CrossSectionalCarryState`](@ref) when `pe.cache` is `nothing`.
 2. Fold `rd` with [`cross_sectional_carry_fold`](@ref), and rebuild the prior with the state it returns.

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
                                                    <:Any, <:Any, <:Any,
                                                    <:Option{<:CrossSectionalCarryState}},
                      rd::ReturnsResult)
    st = isnothing(pe.cache) ? CrossSectionalCarryState() : pe.cache
    return rebuild_estimator(pe, (; cache = cross_sectional_carry_fold(pe, st, rd)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads a Cross-Sectional Factor Prior out of its carry fold, with no data.

The call with no data builds the Prior Result with [`cross_sectional_assemble`](@ref), the code of the batch fit, from the histories that the state carries and the factor prior that it folded. The Return Forecast and the idiosyncratic correlation refit there from the carried rows.

# Algorithm

 1. Refuse a state that has not made its first fit.
 2. Append the observed factors of the fitted observations to the fitted factors with [`cross_sectional_carry_append`](@ref).
 3. Read the factor prior with [`cross_sectional_factor_prior`](@ref) over the factor returns of the factors that are not empty and the observed returns, and process and expand its moments with [`cross_sectional_factor_moments`](@ref). An Observed Factor is never empty.
 4. Count the residuals of each asset with [`variance_count`](@ref) on the folded `ve`.
 5. Take the returns data of the Return Forecast with [`cross_sectional_carry_forecast_returns`](@ref).
 6. Build the result with [`cross_sectional_assemble`](@ref), with the Return Forecast history that the state carries.

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
    fit = (; csr = csr, W = st.W, vs = st.vs, cnt = variance_count(st.ve, csr.eps),
           amr = st.amsk[r, :], bwr = st.bw[r, :], Xo = st.X[r, :], r = r, hist = st.hist)
    return cross_sectional_assemble(pe, f_pr, ca, fit,
                                    cross_sectional_carry_forecast_returns(pe, st);
                                    kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the returns data that the Return Forecast of the carry fold of a Cross-Sectional Factor Prior reads at the call with no data.

The batch fit gives the forecast the returns data that the estimated members read: the returns with the benchmark weights on the panel, and under observed factors the returns net of them. A forecast that reads the panel makes the state carry every row, so the carried rows and their derived series give that returns data. Any other forecast reads no row, and gets the carried rows.

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
    (; rdb) = cross_sectional_benchmark_stage(pe, st.win.X, nothing, st.win.pnl;
                                              ne = st.win.ne, E = st.win.E)
    return if isnothing(st.der)
        rdb
    else
        ReturnsResult(; nx = rdb.nx, X = st.der.Xl, ne = rdb.ne, E = rdb.E, pnl = rdb.pnl)
    end
end
