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
 6. Build the result with [`cross_sectional_assemble`](@ref), with the Return Forecast history and the standardised idiosyncratic returns that the state carries. The block records the position of each fitted row among the kept observations, and no timestamp, because the buffer keeps none.
 7. Record the Shared Series cut of the buffer with [`shared_series_record`](@ref), which writes the marks of the dropped observations and maps each position to its observation among the folded observations.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `st`: The state of its carry fold.
  - $(arg_dict[:strict]) It is forwarded to the factor prior.
  - $(arg_dict[:cache_reg]) It is forwarded to [`cross_sectional_assemble`](@ref).
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
               strict::Bool = false, cache_reg::Bool = false, kwargs...)
    Tf = isnothing(st.Ms) ? 0 : size(st.Ms, 3)
    @argcheck(!isnothing(st.csr),
              ArgumentError("the carry fold of this Cross-Sectional Factor Prior holds $Tf observation(s) after the Descriptor warm-up, and a fit needs at least lag + 2 = $(pe.lag + 2) of them. Fold more observations before the call with no data."))
    csr = st.csr
    r, ca = cross_sectional_carry_append(pe, st)
    lv = vcat(st.lv, trues(size(ca.f, 2) - length(st.lv)))
    pr = cross_sectional_factor_prior(st.pe, all(lv) ? ca.f : ca.f[:, lv]; strict = strict)
    f_pr = cross_sectional_factor_moments(pr, pe.f_mp, ca.f, lv; kwargs...)
    # The histories hold every observation after the warm-up, the last `Tf` of the
    # observations with no Shared Series gap that the fold read, so the fitted rows sit at
    # these positions of the folded returns that a fit keeps. The buffer keeps every folded
    # observation with its mark, so the record of the cut maps each position to its
    # observation, as it does after a batch fit.
    fit = (; csr = csr, W = st.W, vs = st.vs, cnt = variance_count(st.ve, csr.eps),
           amr = view(st.amsk, r, :), bwr = view(st.bw, r, :), Xo = view(st.X, r, :), r = r,
           hist = st.hist, S = st.S, Sc = st.Sc, ce = st.ce,
           idx = (buffer_fit_count(st.buf) - Tf) .+ r, ts = nothing,
           rf = if folds_forecast_rows(pe.rfe)
               return_forecast_result(pe.rfe, st.fh, st.fst)
           else
               nothing
           end)
    pr = cross_sectional_assemble(pe, f_pr, ca, fit,
                                  cross_sectional_carry_forecast_returns(pe, st);
                                  cache_reg = cache_reg, kwargs...)
    return shared_series_record(pr, st.buf)
end
function fold_prior_step(pe::CrossSectionalFactorPrior, rd::ReturnsResult,
                         gaps::Option{<:AbstractVector{Bool}})
    return cross_sectional_pin_choice(pe.choice, refit_prior_fold(pe, rd, gaps))
end
