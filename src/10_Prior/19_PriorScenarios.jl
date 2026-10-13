"""
$(DocStringExtensions.TYPEDSIGNATURES)

Find the entries a mask-aware prior fills: the non-finite returns of an **investable** asset.

A non-investable asset keeps its whole `NaN` column, because the Investable Mask is derived from that gap and filling it would erase the mask. Only a column whose moments came back finite is filled.

# Arguments

  - `X`: Asset returns, `observations × assets`.
  - `imsk`: The Investable Mask, `true` at every asset whose prior moments were finite.

# Returns

  - `filled::Vector{Tuple{Int, Int}}`: The `(observation, asset)` pairs to fill, asset-major.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_msg`](@ref)
  - [`investable_mask`](@ref)
"""
function scenario_fill_pairs(X::MatNum, imsk::BitVector)
    return [(t, i) for i in axes(X, 2) if imsk[i] for t in axes(X, 1) if !isfinite(X[t, i])]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Write the message a scenario fill raises.

The message names the assets, the count of filled pairs and the first observation, so a caller can find the listing that made them. It then names the **worst investable column**, the share of its own observations the fill invented, and the limit that share was measured against — that share is the one the fill tests, because a matrix-wide denominator scales with the universe and hides the column the notice exists to catch. The matrix-wide share is reported beside it as context, and it trips nothing.

It states the consequence at each of the four consumers that a census of the readers of `pr.X` found, because a caller told only about the tail will not look for the other three, and it states the tail's cost as the identity rather than the adjective: the invented zeros do not enter the tail, they inflate the denominator, so a measure at level `alpha` over an admitted column of coverage `c` reads that column's observed `alpha / c` level.

The limit it names is the estimator's own, after [`resolve_fill_limit`](@ref). A `fill_limit` of `nothing` demands every observation of every investable asset, so the message says so rather than printing a share no caller chose.

# Arguments

  - `filled`: The `(observation, asset)` pairs [`scenario_fill_pairs`](@ref) found.
  - `wi`: The index of the worst investable column, the one with the most filled entries.
  - `worst`: The share of its own observations that column's fill invented.
  - `frac`: The share of the entries of the returns matrix every filled pair is, as context.
  - `fill_limit`: The share of a column the fit allowed in silence, or `nothing` when it allowed none.

# Returns

  - `msg::String`: The message.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_pairs`](@ref)
  - [`resolve_fill_limit`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function scenario_fill_msg(filled::AbstractVector{<:Tuple{Integer, Integer}}, wi::Integer,
                           worst::Real, frac::Real, fill_limit::Option{<:Real})
    assets = unique(last.(filled))
    against = if isnothing(fill_limit)
        "against `fill_limit = nothing`, which asks that every investable asset cover every observation"
    else
        "against a limit of $(fill_limit)"
    end
    return "a mask-aware prior estimated an asset from the observations it had, and the returns matrix still carries the gap. Assets $(assets) carry a non-finite return at $(length(filled)) (observation, asset) pair(s) of an investable column, the first at observation $(first(filled)[1]). The worst column is asset $(wi), $(worst) of whose own observations are invented, measured $(against); the whole matrix is $(frac) invented, which is context and trips nothing. Those entries are filled with zero, and four consumers read them. A scenario-based measure (CVaR, EVaR, CDaR and their kin) understates its risk over the filled rows, and the cost is exact: the zeros do not enter the tail, they inflate the denominator, so a measure at level `alpha` over an admitted column of coverage `c` reads that column's observed `alpha / c` level, and at `c = 0.3` a 5% CVaR is a 16.7% CVaR. A hierarchical optimiser whose graph reader reads the prior, `x_src = :prior`, may branch the asset alone and then overweight it, because `clusterise` recomputes its correlation from this matrix rather than from `sigma`, an invented column reads as idiosyncratic, and an inverse-variance allocation over that branch buys more of it. A graph reader that reads the returns data meets the gap itself, and its own `ce` decides it. An entropy pooling view on the asset is calibrated on the filled column, because the view resolvers read the matrix column by column. And a meta-optimiser carries the fill into the outer problem, whose returns matrix is built from the inner Sub-Portfolios' net returns over this one. `mu` and `sigma` are untouched, because the estimator computed them from the rows it saw. Accept a share of an investable column in silence with the estimator's own field, `EmpiricalPrior(; fill_limit = ...)`, state the share as a `CoveragePolicy`'s `min_coverage` so that admission and the notice are the one number, pass `strict = true` to refuse any fill, or fit over a window every asset covers."
end
"""
    resolve_fill_limit(fill_limit::Nothing, floor::Nothing)
    resolve_fill_limit(fill_limit::Nothing, floor::Real)
    resolve_fill_limit(fill_limit::Real, floor::Nothing)
    resolve_fill_limit(fill_limit::Real, floor::Real)

Resolve the share of an investable column a [`scenario_fill`](@ref) may invent in silence, at the fit.

`fill_limit` and a [`CoveragePolicy`](@ref)'s `min_coverage` are two spellings of one number. [`admits`](@ref) reads an asset's coverage share as its own observation count over the number of observations folded, and the fill counts that column's non-finite entries over the same denominator, so `filled_share == 1 - coverage_share` identically and the admission test **is** the fill test. A limit that did not know this would fire on nearly every fold of an available-case walk-forward, naming the caller for doing precisely what they configured.

So `nothing` **derives**. Where any arm of the fitting estimator states a floor, `nothing` means `1 - floor`, and it never fires: every admitted column satisfies it by construction. Where no arm states one — the exponentially weighted family is mask-aware without a policy, gating on `min_obs`, a *count* that says nothing about the share of a long window — `nothing` keeps its original meaning and names every fill, because a caller who set no floor has weighed no trade.

An explicit `fill_limit` overrides the derivation and must be **tighter** than admission. A value above `1 - floor` is refused, because it is dead by construction: nothing that reaches the fill could trip it, and a knob that cannot fire is worse than no knob. What an explicit value buys is the one configuration the derivation cannot express — admit broadly and be told anyway, `min_coverage = 0.3` with `fill_limit = 0.5`.

The four methods are dispatch rather than a branch, so a call site holding two `Option`s finds a method for every arm of its union split.

# Arguments

  - `fill_limit`: The estimator's own `fill_limit` field, or `nothing` to derive one.
  - `floor`: The binding coverage floor of the estimator's arms, or `nothing` when no arm states one.

# Validation

  - `fill_limit <= 1 - floor` where both are stated, else a `DomainError` naming both is thrown.

# Returns

  - `fill_limit::Option{<:Real}`: The share of an investable column the fill may invent in silence, or `nothing` when it may invent none.

# Related

  - [`scenario_fill`](@ref)
  - [`coverage_floor`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
function resolve_fill_limit(::Nothing, ::Nothing)
    return nothing
end
function resolve_fill_limit(::Nothing, floor::Real)
    return one(floor) - floor
end
function resolve_fill_limit(fill_limit::Real, ::Nothing)
    return fill_limit
end
function resolve_fill_limit(fill_limit::Real, floor::Real)
    @argcheck(fill_limit <= one(floor) - floor,
              DomainError(fill_limit,
                          "fill_limit must be tighter than the coverage floor it is measured against, so it must be at most 1 - min_coverage = $(one(floor) - floor). A looser limit is dead by construction: an asset the fit admits covers at least $(floor) of the observations folded, so its column is at most $(one(floor) - floor) invented, and nothing that reaches the fill could trip $(fill_limit). Pass `nothing`, the default, to derive $(one(floor) - floor) and be told only when a coverage algorithm admits a column thinner than the floor it was given."))
    return fill_limit
end
"""
    scenario_window(::Nothing, X::MatNum)
    scenario_window(max_scenarios::Integer, X::MatNum)

Cut the returns matrix a Prior Result carries down to the last `max_scenarios` observations.

The Scenario Cap, applied. A prior that carries `X` carries it for the scenario risk measures, and a caller who wants a long window of moments and a short window of scenarios says so with one field rather than with two fits. The cap therefore touches `X` alone: `mu` and `sigma` are already computed when this runs, over every observation the fit read, and nothing here can or does move them.

It is deliberately the **same** verb in batch and online. A cap is a property of the result, not of the fold, so `prior(pe, X)` and `prior(pe)` with no data on a folded `pe` cut the same rows off the same tail.

A window at or above the number of observations is the matrix itself, and no copy is taken: the cut is a `view`, so a cap that does nothing costs nothing.

# Arguments

  - `max_scenarios`: The number of observations to carry, or `nothing` to carry every one.
  - `X`: Asset returns, `observations × assets`.

# Returns

  - `X::MatNum`: The last `max_scenarios` rows of `X`, or `X` itself.

# Related

  - [`scenario_fill`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function scenario_window(::Nothing, X::MatNum)
    return X
end
function scenario_window(max_scenarios::Integer, X::MatNum)
    t = size(X, 1)
    return max_scenarios >= t ? X : view(X, (t - max_scenarios + 1):t, :)
end
"""
    scenario_weights(w::Nothing, t::Integer)
    scenario_weights(w::ObsWeights, t::Integer)

Put the probabilities `w` of the scenarios of a capped Prior Result on the `t` observations that its moments read.

[`scenario_window`](@ref) keeps the last rows of the observations as scenarios, and the moment estimators read every observation. A prior that reweights scenarios, such as [`EntropyPoolingPrior`](@ref), states a probability for each scenario and none for the observations that the cap removed. Its posterior gives those observations no probability, so this function gives them a zero weight. A weighted moment over the `t` observations is then the moment of the scenarios under `w`, and the posterior mean is `pr.X' * w`, the mean that a view states.

# Algorithm

 1. A `nothing` `w` returns `nothing`.
 2. Otherwise take `s = length(w)`, and refuse `s > t`.
 3. When `s == t`, the cap removed no row: return `w`.
 4. Otherwise return `t - s` zeros followed by `w`, through [`scenario_probabilities`](@ref): in the kind of `w` when it is a [`ScenarioProbabilities`](@ref), and as `StatsBase.AnalyticWeights` otherwise.

# Arguments

  - `w`: The probabilities of the scenarios, or `nothing`.
  - `t`: The number of observations that the moments read.

# Validation

  - `length(w) <= t`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `w′::Option{<:ObsWeights}`: The weights over the `t` observations, or `nothing`.

# Related

  - [`scenario_window`](@ref)
  - [`scenario_factory`](@ref)
  - [`prior_probability_factory`](@ref)
  - [`scenario_rows`](@ref)
"""
function scenario_weights(::Nothing, ::Integer)
    return nothing
end
function scenario_weights(w::ObsWeights, t::Integer)
    s = length(w)
    @argcheck(s <= t,
              DimensionMismatch("the prior answers $s scenario probabilities, but its moments read $t observations. A Scenario Cap keeps the last rows of the observations, so the probabilities of the scenarios cannot outnumber the observations."))
    if s == t
        return w
    end
    return scenario_probabilities(w, vcat(zeros(eltype(w), t - s), w))
end
"""
    cross_sectional_regression_stage(pe::CrossSectionalFactorPrior, X::MatNum,
                                     F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                     ne = nothing, E = nothing, iv = nothing,
                                     ivpa = nothing, ts = nothing) -> NamedTuple

Fit the cross-sectional regression of a [`CrossSectionalFactorPrior`](@ref), up to the factor returns that its factor prior reads.

The returns-matrix method of [`prior`](@ref) fits `pe.pe` on the factor returns of this stage and assembles the result. [`scenario_factory`](@ref) and [`prior_probability_factory`](@ref) read the same factor returns, because an entropy pooling prior around `pe` states its probabilities on the scenarios of `pe.pe`. The stage reads no weight of `pe.pe`, so a change of `pe.pe` alone leaves its answer the same.

# Algorithm

 1. Run the steps 1 to 9 of the returns-matrix method of [`prior`](@ref): the exposure stage, the Neutralisation, the Factor Family Basis, the lag, the eligibility, the regression and the idiosyncratic variance history.
 2. Append the observed factors with [`cross_sectional_observed_append`](@ref), giving `ca`.
 3. Mark the factors that are not empty in `lv`, on the axis of `ca.f`. An Observed Factor is never empty.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - $(arg_dict[:X])
  - `F`: Optional factor returns matrix, already oriented. The stage does not read it.
  - $(arg_dict[:pnl_prior]) This estimator reads it, and refuses without it.
  - `ne`, `E`, `iv`, `ivpa`, `ts`: As in the returns-matrix method of [`prior`](@ref), with `E` already oriented.

# Validation

  - The rules of the returns-matrix method of [`prior`](@ref), up to the factor prior.

# Returns

  - `ca::NamedTuple`: The output of [`cross_sectional_observed_append`](@ref). `ca.f` holds the factor returns, one row per fitted observation.
  - `lv::BitVector`: `true` at each factor of `ca.f` that is not empty.
  - `fit::NamedTuple`: The regression, the weights, the variance history and the rows that [`cross_sectional_assemble`](@ref) reads.
  - `rde::ReturnsResult`: The returns data that the estimated members read.

# Related

  - [`prior`](@ref)
  - [`cross_sectional_assemble`](@ref)
  - [`cross_sectional_factor_moments`](@ref)
  - [`scenario_factory`](@ref)
"""
function cross_sectional_regression_stage(pe::CrossSectionalFactorPrior, X::MatNum,
                                          F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                          ne::Option{<:VecStr} = nothing,
                                          E::Option{<:MatNum} = nothing,
                                          iv::Option{<:MatNum} = nothing,
                                          ivpa::Option{<:Num_VecNum} = nothing,
                                          ts::Option{<:VecDate} = nothing)
    @argcheck(!isnothing(pnl),
              IsNothingError("a Cross-Sectional Factor Prior reads its Factor Exposures off an Asset Panel, and the panel is nothing. Call prior(pe, rd) with a ReturnsResult whose `pnl` is the one asset_panel returns, or hand the panel to this method as its third positional argument."))
    @argcheck(isnothing(ts) || length(ts) == size(X, 1),
              DimensionMismatch("ts ($(length(something(ts, ())))) must have one entry for each observation of X ($(size(X, 1)))"))
    st = cross_sectional_exposure_stage(pe, X, F, pnl; ne = ne, E = E, iv = iv, ivpa = ivpa)
    (; amsk, emsk, mcap, BW, cc, Xl, rde, Ms, nf, fam, rw) = st
    Msw = Ms[rw, :, :]
    Xw = X[rw, :]
    Xlw = Xl[rw, :]
    bww = BW[rw, :]
    mcw = cross_sectional_rows(mcap, rw)
    cross_sectional_neutralise!(pe.neutralise, Msw, pe.cre, bww, nf, fam)
    fb = cross_sectional_family_basis(pe.families, Msw, bww, nf, fam)
    @argcheck(length(rw) > pe.lag,
              ArgumentError("the exposures lag the returns by lag = $(pe.lag), so a fit needs more than $(pe.lag) observations after the Descriptor warm-up, and $(length(rw)) are left. Give more observations, shorten the warm-up of the Descriptors, or lower lag."))
    @argcheck(length(rw) - pe.lag >= 2,
              ArgumentError("the factor prior states a covariance of the factor returns, and a covariance of one observation is not a number, so a fit needs at least two observations after the Descriptor warm-up and the exposure lag of $(pe.lag), and $(length(rw) - pe.lag) is left. Give more observations, or shorten the warm-up of the Descriptors."))
    r = (pe.lag + 1):length(rw)
    Zl = fb.Ms[r .- pe.lag, :, :]
    Xr = Xlw[r, :]
    cb = cross_sectional_observed_block(cc, rw, r)
    amr = amsk[rw[r], :]
    bwr = bww[r, :]
    msk = cross_sectional_eligible(Xr, Zl, emsk[rw[r], :])
    mcl = cross_sectional_rows(mcw, r .- pe.lag)
    cross_sectional_cap_finite!(msk, mcl)
    assert_cross_sectional_coverage(msk, cross_sectional_minra(pe, size(Zl, 3)), cb)
    W = cs_weights_initial(pe.wa, mcl, msk)
    # The Unseen Member rule states the return of a member of a constrained family that no
    # asset of an observation loads on (#1606).
    Bl = Msw[r .- pe.lag, :, :]
    fbl = cross_sectional_basis_now(fb.fcb, r .- pe.lag)
    ud = unseen_member_design(pe.unseen, fbl, Bl, Zl, W)
    (; csr, lv) = cross_sectional_live_regression(pe.cre, ud.Z, Xr, W)
    csr = unseen_member_returns(csr, ud.P)
    assert_cross_sectional_no_intercept(csr)
    # The idiosyncratic variance is a folding statistic, so it reads both universe masks: it
    # resets an asset the active mask turns off, and a regime-adjusted estimator measures its
    # regime over the estimation universe alone (ADR 0172). An estimator that takes no mask
    # ignores the two keywords.
    emr = emsk[rw[r], :]
    if needs_second_pass(pe.wa)
        W = cs_weights_refine(pe.wa, W, csr.eps, pe.ve, msk; estimation_mask = emr,
                              active_mask = amr)
        ud = unseen_member_design(pe.unseen, fbl, Bl, Zl, W)
        (; csr, lv) = cross_sectional_live_regression(pe.cre, ud.Z, Xr, W)
        csr = unseen_member_returns(csr, ud.P)
    end
    vs = variance_series(pe.ve, csr.eps; dims = 1, estimation_mask = emr, active_mask = amr)
    ca = cross_sectional_observed_append(cb, csr.f, fb.Ms[r[end], :, :], Msw[r, :, :], nf,
                                         fam, fb.fcb)
    fit = (; csr = csr, W = W, vs = vs, cnt = variance_count(pe.ve, csr.eps), amr = amr,
           bwr = bwr, Xo = Xw[r, :], r = r, hist = nothing, S = nothing, Sc = nothing,
           ce = nothing, idx = rw[r], ts = isnothing(ts) ? nothing : ts[rw[r]],
           rf = nothing)
    # An Observed Factor carries the return the caller observed, so it is never empty.
    return (; ca = ca, lv = vcat(lv, trues(size(ca.f, 2) - length(lv))), fit = fit,
            rde = rde)
end
"""
    cross_sectional_factor_scenarios(pe::CrossSectionalFactorPrior, X::MatNum,
                                     F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                     ne = nothing, E = nothing, iv = nothing,
                                     ivpa = nothing, ts = nothing, kwargs...)

Return the factor returns that the factor prior `pe.pe` of a [`CrossSectionalFactorPrior`](@ref) is fitted on, with the regression stage that gives them.

An entropy pooling prior around `pe` states its probabilities on the scenarios of `pe`, and these are the last rows of the factor returns of `pe.pe`. The Descriptor warm-up and the exposure lag drop rows, so the number of factor returns is known only after the regression. [`scenario_factory`](@ref) and [`prior_probability_factory`](@ref) read these rows, to put the probabilities on the observations that `pe.pe` reads. [`scenario_prior`](@ref) also reads the stage, so the refit runs the regression once.

# Algorithm

 1. Fit the regression of `pe` with [`cross_sectional_regression_stage`](@ref), giving the stage `rs`, with the factor returns `rs.ca.f` and the mask `rs.lv` of the factors that are not empty.
 2. Take the columns of `rs.ca.f` that `rs.lv` marks, as the fit of `pe.pe` reads them.
 3. Return them with `rs`, and with the keyword arguments that the stage did not read.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - $(arg_dict[:X])
  - `F`: Optional factor returns matrix, already oriented.
  - $(arg_dict[:pnl_prior])
  - `ne`, `E`, `iv`, `ivpa`, `ts`: As in the returns-matrix method of [`prior`](@ref), with `E` already oriented.
  - `kwargs...`: The other keyword arguments of the fit. The regression reads none of them.

# Validation

  - The rules of [`cross_sectional_regression_stage`](@ref).

# Returns

  - `f::Matrix`: The factor returns of `pe.pe`, one row per fitted observation.
  - `rs::NamedTuple`: The output of [`cross_sectional_regression_stage`](@ref).
  - `kw`: The keyword arguments in `kwargs...`, which the rest of the fit reads.

# Related

  - [`cross_sectional_regression_stage`](@ref)
  - [`scenario_factory`](@ref)
  - [`scenario_prior`](@ref)
  - [`prior_probability_factory`](@ref)
"""
function cross_sectional_factor_scenarios(pe::CrossSectionalFactorPrior, X::MatNum,
                                          F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                          ne::Option{<:VecStr} = nothing,
                                          E::Option{<:MatNum} = nothing,
                                          iv::Option{<:MatNum} = nothing,
                                          ivpa::Option{<:Num_VecNum} = nothing,
                                          ts::Option{<:VecDate} = nothing, kwargs...)
    rs = cross_sectional_regression_stage(pe, X, F, pnl; ne = ne, E = E, iv = iv,
                                          ivpa = ivpa, ts = ts)
    return (; f = all(rs.lv) ? rs.ca.f : rs.ca.f[:, rs.lv], rs = rs, kw = kwargs)
end
# The refit of a Cross-Sectional Factor Prior under the posterior weights holds its model, as
# the refit of a `FactorPrior` does (ADR 0205), and gives the weights to `pe` alone, on the
# factor returns that it reads (#1685). `scenario_factory` documents every method.
function scenario_factory(pe::CrossSectionalFactorPrior, w, X::MatNum, F::Option{<:MatNum},
                          pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    (; f) = cross_sectional_factor_scenarios(pe, X, F, pnl; kwargs...)
    return rebuild_estimator(pe,
                             (;
                              pe = scenario_factory(pe.pe, w, f, nothing, nothing;
                                                    strict = strict)))
end
# The refit of an entropy pooling prior runs the regression stage once, and fits the factor
# prior of `scenario_factory` on its factor returns. `scenario_prior` documents every method.
function scenario_prior(pe::CrossSectionalFactorPrior, w, X::MatNum, F::Option{<:MatNum},
                        pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    (; f, rs, kw) = cross_sectional_factor_scenarios(pe, X, F, pnl; kwargs...)
    fpe = scenario_factory(pe.pe, w, f, nothing, nothing; strict = strict)
    f_pr = cross_sectional_factor_moments(fpe, pe.f_mp, rs.ca.f, rs.lv; strict = strict,
                                          kw...)
    return cross_sectional_assemble(pe, f_pr, rs.ca, rs.fit, rs.rde; kw...)
end
# A caller's prior probabilities weigh the fit, so every field takes them on the rows of the
# factor returns, which each field reads (ADR 0209).
function prior_probability_factory(pe::CrossSectionalFactorPrior, w, X::MatNum,
                                   F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                   kwargs...)
    (; f) = cross_sectional_factor_scenarios(pe, X, F, pnl; kwargs...)
    inner = prior_probability_factory(pe.pe, w, f, nothing, nothing)
    return rebuild_estimator(factory(pe, scenario_weights(w, size(f, 1))), (; pe = inner))
end
function assert_scenario_model(pe::CrossSectionalFactorPrior)
    return assert_scenario_model(pe.pe)
end
"""
    scenario_ens(::Nothing, X::MatNum)
    scenario_ens(max_scenarios::Integer, X::MatNum)

State the number of observations the moments of a capped Prior Result were fitted over, or `nothing` when the cap cuts nothing.

The count a Scenario Cap owes its readers. [`scenario_window`](@ref) cuts the rows the result carries and leaves `mu` and `sigma` fitted over every observation, so a consumer that prices a sample size off `size(pr.X, 1)` — an uncertainty set's `T`, a calibration rule's count — would read `w` rows for moments fitted over `t`, and mis-price every count by `t / w`. The result therefore states `t` in `ens` exactly when the cap cuts, and every count reader takes `ens` before the shape. When the cap does not cut, or there is no cap, the rows carried *are* the observations fitted over, and `ens` stays `nothing`, so a fit without a cap is bit-identical to what it was.

It is the same verb in batch and in `prior(pe)` with no data after a fold, as [`scenario_window`](@ref) is, and the two agree by construction: both read the same `t` off the same matrix.

# Arguments

  - `max_scenarios`: The number of observations carried, or `nothing`.
  - `X`: Asset returns the moments were fitted over, `observations × assets`, **before** the cut.

# Returns

  - `ens::Option{<:Integer}`: `size(X, 1)` when `max_scenarios` cuts it, `nothing` otherwise.

# Related

  - [`scenario_window`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`effective_sample_size`](@ref)
"""
function scenario_ens(::Nothing, ::MatNum)
    return nothing
end
function scenario_ens(max_scenarios::Integer, X::MatNum)
    t = size(X, 1)
    return max_scenarios >= t ? nothing : t
end
"""
    scenario_fill_report(filled, ::Nothing)
    scenario_fill_report(filled, named::AbstractSet{<:Integer})

Narrow the filled pairs a [`scenario_fill`](@ref) reports to the assets it has not named before.

A batch fit calls [`scenario_fill`](@ref) once, so it reports everything it finds and keeps no memory: that is the `nothing` method. A walk-forward reading out at every step calls it once per step over a growing window, and every step would otherwise name the same assets again, so a caller who read the first notice learns to ignore the channel and misses the asset that lists at step 500.

So a carry state holds the set of assets it has already named, and this verb narrows the report to the pairs of assets outside it. The set is the state's memory, and [`scenario_fill_remember!`](@ref) is what writes to it — only for a notice that actually fired, so an asset whose share was under the limit at one step is still named at the step where it goes over.

# Arguments

  - `filled`: The `(observation, asset)` pairs [`scenario_fill_pairs`](@ref) found.
  - `named`: The assets already named, or `nothing` when nothing remembers.

# Returns

  - `report::AbstractVector{<:Tuple{Integer, Integer}}`: The pairs to name, which is `filled` itself when nothing remembers.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_remember!`](@ref)
  - [`PriorCarryState`](@ref)
"""
function scenario_fill_report(filled::AbstractVector{<:Tuple{Integer, Integer}}, ::Nothing)
    return filled
end
function scenario_fill_report(filled::AbstractVector{<:Tuple{Integer, Integer}},
                              named::AbstractSet{<:Integer})
    return filter(p -> last(p) ∉ named, filled)
end
"""
    scenario_fill_remember!(::Nothing, report)
    scenario_fill_remember!(named::AbstractSet{<:Integer}, report)

Record the assets a [`scenario_fill`](@ref) has just named, so a later call of `prior(pe)` with no data does not name them again.

The write side of [`scenario_fill_report`](@ref). It runs after the notice, not before it, so an asset is remembered exactly when a caller was told about it: under `strict` the notice throws and nothing is remembered, and under a limit the notice did not trip nothing is remembered either.

The set is a field of a [`PriorCarryState`](@ref) and is written **in place**, because `prior(pe)` with no data returns a Prior Result rather than the estimator, so there is no other channel by which the memory could survive the call.

# Arguments

  - `named`: The assets already named, or `nothing` when nothing remembers.
  - `report`: The `(observation, asset)` pairs just named.

# Returns

  - `nothing`.

# Related

  - [`scenario_fill`](@ref)
  - [`scenario_fill_report`](@ref)
  - [`PriorCarryState`](@ref)
"""
function scenario_fill_remember!(::Nothing, ::AbstractVector{<:Tuple{Integer, Integer}})
    return nothing
end
function scenario_fill_remember!(named::AbstractSet{<:Integer},
                                 report::AbstractVector{<:Tuple{Integer, Integer}})
    for p in report
        push!(named, last(p))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the returns matrix with the missing rows of every investable asset filled with zero, and say so above the fitting estimator's resolved `fill_limit`.

A mask-aware moment estimator answers a young asset from the observations it has, so the asset is investable and its returns column still carries a `NaN` at every row before it listed. Every consumer of a Prior Result reads that column — the JuMP model, the meta-optimisers and the value-level `expected_*` functions among them — so the fill is paid **once**, here, on the estimator's own pass, rather than at each of them.

A non-investable asset keeps its `NaN` column, so [`investable_mask`](@ref) is unchanged. `mu`, `sigma` and every other block are untouched, because the estimator computed them from the rows it saw.

**The share the fill tests is per asset**: the worst investable column's own count of filled entries over the number of observations. A matrix-wide denominator scales with the universe, so the column the notice exists to catch disappears inside it — one asset of a hundred whose column is seven-tenths invented is seven thousandths of the matrix, under any limit a caller would set. The matrix-wide share is reported in the message as context and trips nothing.

How much of the trade passes in silence is the fitting estimator's own answer, carried in its `fill_limit` field and resolved at the fit by [`resolve_fill_limit`](@ref) against the coverage floor of its arms — the share is a property of one fit, not of the session, so two priors in one program may answer differently. Under `strict = false` the trade is silent while the worst column's share stays at or below the resolved limit, and is named through [`strict_diagnostic`](@ref) above it. Under `strict = true` **any** fill refuses, whatever the limit holds.

Two mask-aware families reach this verb. The exponentially weighted family carries no [`CoveragePolicy`](@ref), so a `fill_limit` of `nothing` names every fill there. The plain family is mask-aware exactly where a policy is set, and a policy derives a limit that never fires, so an available-case fit names nothing while it does what it was configured to do. A plain estimator with no policy never reaches this verb at all: under the whole-window rule, an asset it could not cover leaves the Coverage Universe and is not investable.

# Algorithm

 1. Return `X` itself when every entry of `X` is finite, which is every fit over a complete window.
 2. Derive the Investable Mask from `mu` and the diagonal of `sigma`, as [`investable_mask`](@ref) does, and find the pairs to fill with [`scenario_fill_pairs`](@ref). Return `X` itself when there are none, which is a gap that belongs to a non-investable asset alone.
 3. Narrow the pairs to report with [`scenario_fill_report`](@ref), which drops the assets a carry state has already named and is the identity when nothing remembers. Under `strict` the narrowing is skipped, because `strict` refuses any fill and must not depend on how often the estimator has been read out.
 4. Count the reported pairs per asset, take the worst column's share of its own observations, and report through [`strict_diagnostic`](@ref) when `strict` holds, when `fill_limit` is `nothing`, or when that share exceeds `fill_limit`. Record the assets just named with [`scenario_fill_remember!`](@ref).
 5. Return a copy of `X` with zero written at each of the filled pairs — every one of them, not just the reported ones — and the `NaN` of every non-investable asset left where it is.

# The notice fires once per asset, not once per step

`named` is what separates the batch call from the online one. A batch fit passes `nothing`, reports every fill it finds, and remembers nothing. A call of `prior(pe)` with no data on a folded prior passes the set its [`PriorCarryState`](@ref) carries, so a walk-forward names an asset at the step it lists and stays quiet afterwards instead of emitting the same notice at every one of two thousand steps. The **fill itself is unchanged**: every filled pair is written at every call, whatever the set holds. Under `strict = true` the first fill still throws.

# Arguments

  - `X`: Asset returns, `observations × assets`.
  - `mu`: The expected returns the estimator answered, `NaN` at a non-investable asset.
  - `sigma`: The covariance the estimator answered, `NaN` on the diagonal at a non-investable asset.
  - `strict`: If `true`, any fill raises an `ArgumentError`; if `false`, a fill above the share warns.
  - `fill_limit`: The share of an investable column the fitting estimator accepts in silence, after [`resolve_fill_limit`](@ref), or `nothing` when it accepts none.
  - `named`: The assets already named, written in place when a notice fires, or `nothing` when nothing remembers.

# Validation

  - The worst investable column's filled share is at or below `fill_limit`, else a warning naming the assets is emitted, or an `ArgumentError` naming them is raised under `strict`, which any fill raises. A `fill_limit` of `nothing` is at or below no share, so any fill is named.

# Returns

  - `X::MatNum`: The returns matrix whose investable columns are finite.

# Related

  - [`scenario_fill_pairs`](@ref)
  - [`scenario_fill_msg`](@ref)
  - [`scenario_fill_report`](@ref)
  - [`scenario_fill_remember!`](@ref)
  - [`scenario_window`](@ref)
  - [`resolve_fill_limit`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`investable_mask`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`filter_held_gaps`](@ref)
"""
function scenario_fill(X::MatNum, mu::VecNum, sigma::MatNum, strict::Bool,
                       fill_limit::Option{<:Real},
                       named::Option{<:AbstractSet{<:Integer}} = nothing)
    if all(isfinite, X)
        return X
    end
    imsk = isfinite.(mu) .& isfinite.(LinearAlgebra.diag(sigma))
    filled = scenario_fill_pairs(X, imsk)
    if isempty(filled)
        return X
    end
    # `strict` refuses **any** fill, so it never reads the memory: an asset that a previous
    # call of `prior(pe)` with no data named is still a fill, and silencing it here would
    # make `strict` depend on how many times the estimator had been read out.
    report = strict ? filled : scenario_fill_report(filled, named)
    if !isempty(report)
        counts = zeros(Int, size(X, 2))
        for (_, i) in report
            counts[i] += 1
        end
        wi = argmax(counts)
        worst = counts[wi] / size(X, 1)
        frac = length(filled) / length(X)
        if strict || isnothing(fill_limit) || worst > fill_limit
            strict_diagnostic(scenario_fill_msg(report, wi, worst, frac, fill_limit),
                              strict)
            scenario_fill_remember!(named, report)
        end
    end
    Xf = copy(X)
    for (t, i) in filled
        Xf[t, i] = zero(eltype(Xf))
    end
    return Xf
end
