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
    return "a mask-aware prior estimated an asset from the observations it had, and the returns matrix still carries the gap. Assets $(assets) carry a non-finite return at $(length(filled)) (observation, asset) pair(s) of an investable column, the first at observation $(first(filled)[1]). The worst column is asset $(wi), $(worst) of whose own observations are invented, measured $(against); the whole matrix is $(frac) invented, which is context and trips nothing. Those entries are filled with zero, and four consumers read them. A scenario-based measure (CVaR, EVaR, CDaR and their kin) understates its risk over the filled rows, and the cost is exact: the zeros do not enter the tail, they inflate the denominator, so a measure at level `alpha` over an admitted column of coverage `c` reads that column's observed `alpha / c` level, and at `c = 0.3` a 5% CVaR is a 16.7% CVaR. A hierarchical optimiser whose graph reader reads the prior, `x_src = PriorSource()`, may branch the asset alone and then overweight it, because `clusterise` recomputes its correlation from this matrix rather than from `sigma`, an invented column reads as idiosyncratic, and an inverse-variance allocation over that branch buys more of it. A graph reader that reads the returns data meets the gap itself, and its own `ce` decides it. An entropy pooling view on the asset is calibrated on the filled column, because the view resolvers read the matrix column by column. And a meta-optimiser carries the fill into the outer problem, whose returns matrix is built from the inner Sub-Portfolios' net returns over this one. `mu` and `sigma` are untouched, because the estimator computed them from the rows it saw. Accept a share of an investable column in silence with the estimator's own field, `EmpiricalPrior(; fill_limit = ...)`, state the share as a `CoveragePolicy`'s `min_coverage` so that admission and the notice are the one number, pass `strict = true` to refuse any fill, or fit over a window every asset covers."
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
    scenario_tail(pr::LowOrderPrior, n::Integer, name::Symbol)

Cut a nested Prior Result to the last `n` scenarios that it carries.

[`AugmentedBlackLittermanPrior`](@ref) fits two nested priors, one on the asset returns and one on the factor returns. A Scenario Cap in one of them keeps the last rows of its own returns alone. The two blocks of the result describe the same observations, so the result keeps the rows that both nested priors keep, and this function cuts the nested prior that carries more rows. The moments of `pr` are fitted over every observation, so the cut changes no moment.

An entry of `nothing` in the `eoc` of a Prior Result states `size(pr.X, 1)` as the count of the observations that its moment reads. The cut makes `X` shorter, so it writes that count in each such entry, as [`scenario_moment_counts`](@ref) does for the cap itself.

# Algorithm

 1. Take `t = size(pr.X, 1)`. When `n >= t`, return `pr`.
 2. Refuse a `pr` that carries weights, per `# Validation`.
 3. Forward `pr` with [`forward_prior`](@ref), with `X` and `o_X` cut by [`scenario_window`](@ref), and with each entry of `eoc` set to `t` when it is `nothing`.

# Arguments

  - `pr`: The Prior Result of a nested prior.
  - `n`: The number of scenarios to keep.
  - `name`: The field of the wrapping prior that holds the nested prior, for the error message.

# Validation

  - When the cut removes a row, `pr.w` is `nothing`. A weight vector states a probability for each row that `pr` carries, so it has no meaning on fewer rows. A `DimensionMismatch` that names `name`, both counts and the fix is thrown otherwise.

# Returns

  - `pr::LowOrderPrior`: `pr` with the last `n` rows of `X`, or `pr` itself.

# Related

  - [`scenario_window`](@ref)
  - [`scenario_moment_counts`](@ref)
  - [`AugmentedBlackLittermanPrior`](@ref)
"""
function scenario_tail(pr::LowOrderPrior, n::Integer, name::Symbol)
    t = size(pr.X, 1)
    if n >= t
        return pr
    end
    @argcheck(isnothing(pr.w),
              DimensionMismatch("`$name` answered $t weighted scenarios, and the other nested prior keeps the last $n. The two blocks of the result describe the same observations, so the result keeps $n rows, and the weights of `$name` have no meaning on fewer rows than they weigh. Put the same Scenario Cap inside `$name`, in the prior that it reweights, so that its weights state the same $n scenarios."))
    o_X = isnothing(pr.o_X) ? nothing : scenario_window(n, pr.o_X)
    return forward_prior(pr; X = scenario_window(n, pr.X), o_X = o_X,
                         eoc = map(c -> something(c, t), pr.eoc))
end
"""
    keeps_observations(pe::AbstractPriorEstimator) -> false
    keeps_observations(pe::EmpiricalPrior) -> Bool
    keeps_observations(pe::Union{FactorPrior, FactorBlackLittermanPrior,
                                 BlackLittermanPrior, BayesianBlackLittermanPrior,
                                 EntropyPoolingPrior, MeucciEntropyPoolingPrior,
                                 HighOrderPriorEstimator,
                                 HighOrderFactorPriorEstimator}) -> Bool
    keeps_observations(pe::AugmentedBlackLittermanPrior) -> Bool
    keeps_observations(pe::OpinionPoolingPrior) -> Bool
    keeps_observations(pe::Online{<:AbstractPriorEstimator}) -> false

Return `true` when the result of a prior estimator carries one scenario for each observation that it receives.

A prior that reweights observations, such as [`EntropyPoolingPrior`](@ref), states its prior probabilities on the scenarios of the prior that it wraps. A wrapped prior can carry fewer scenarios than it receives, so the reweighting prior reads the count with [`scenario_count`](@ref) before it fits the wrapped prior under the probabilities. A wrapped prior that keeps the observations carries the count that it receives, so the count needs no fit.

The fallback returns `false`, which makes no promise, and [`scenario_count`](@ref) then fits a prior that has no method of its own to read the count. A wrong `true` makes the reweighting prior check the probabilities against the wrong count. So a prior returns `true` only when no configuration and no data makes its result carry fewer scenarios than the observations that it receives. A [`CrossSectionalFactorPrior`](@ref) keeps the fallback, because it drops the observations that its Descriptors warm up over and the observations that its exposure lag uses. The answer reads the configuration alone. A Scenario Cap at or above the number of observations drops no row, but its prior still returns `false`.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: return `false`.
 2. An [`EmpiricalPrior`](@ref): return `true` when it has no Scenario Cap, that is when `pe.max_scenarios` is `nothing`.
 3. A prior whose result carries the scenarios of the prior in its `pe` field: apply this function to `pe.pe`. A factor prior cuts its asset rows to the rows of its factor prior, and the other priors forward the rows of their wrapped prior.
 4. An [`AugmentedBlackLittermanPrior`](@ref): return `true` when this function returns `true` for both `pe.a_pe` and `pe.f_pe`, because the result keeps the rows that both nested priors keep.
 5. An [`OpinionPoolingPrior`](@ref): return `true` when `pe.pe1` is `nothing`, or when this function returns `true` for `pe.pe1`. The pool works on the rows of `pe.pe1`, and it refuses an opinion or a `pe.pe2` that answers on fewer rows.
 6. An [`Online`](@ref) wrapper of a prior: return `false`. A batch fit does not fit the wrapper, and the online flow replaces it with the prior that it wraps before the fit, so the wrapper makes no promise.

# Arguments

  - $(arg_dict[:pe])

# Returns

  - `flag::Bool`: `true` when the result carries one scenario for each observation, in order, and `false` when the prior makes no such promise.

# Examples

```jldoctest
julia> PortfolioOptimisers.keeps_observations(EmpiricalPrior())
true

julia> PortfolioOptimisers.keeps_observations(EmpiricalPrior(; max_scenarios = 50))
false

julia> PortfolioOptimisers.keeps_observations(EntropyPoolingPrior(; pe = FactorPrior()))
true
```

# Related

  - [`scenario_count`](@ref)
  - [`ep_wrapped_prior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`scenario_window`](@ref)
"""
function keeps_observations(::AbstractPriorEstimator)::Bool
    return false
end
function keeps_observations(pe::EmpiricalPrior)::Bool
    return isnothing(pe.max_scenarios)
end
function keeps_observations(pe::Union{FactorPrior, FactorBlackLittermanPrior,
                                      BlackLittermanPrior, BayesianBlackLittermanPrior,
                                      EntropyPoolingPrior, MeucciEntropyPoolingPrior,
                                      HighOrderPriorEstimator,
                                      HighOrderFactorPriorEstimator})::Bool
    return keeps_observations(pe.pe)
end
function keeps_observations(pe::AugmentedBlackLittermanPrior)::Bool
    return keeps_observations(pe.a_pe) && keeps_observations(pe.f_pe)
end
function keeps_observations(pe::OpinionPoolingPrior)::Bool
    return isnothing(pe.pe1) || keeps_observations(pe.pe1)
end
function keeps_observations(::Online{<:AbstractPriorEstimator})::Bool
    return false
end
"""
    kept_scenarios(pe::AbstractPriorEstimator, T::Integer)
    kept_scenarios(pe::EmpiricalPrior, T::Integer)
    kept_scenarios(pe::Union{FactorPrior, FactorBlackLittermanPrior,
                             BlackLittermanPrior, BayesianBlackLittermanPrior,
                             EntropyPoolingPrior, MeucciEntropyPoolingPrior,
                             HighOrderPriorEstimator, HighOrderFactorPriorEstimator},
                   T::Integer)
    kept_scenarios(pe::AugmentedBlackLittermanPrior, T::Integer)
    kept_scenarios(pe::OpinionPoolingPrior, T::Integer)

Return the number of scenarios that a fit of `pe` on `T` observations carries, when the configuration of `pe` alone states it, and `nothing` when the data decides it.

[`scenario_count`](@ref) reads this function first, so a prior whose count the configuration states needs no data to count its scenarios. A Scenario Cap keeps the last `pe.max_scenarios` rows, and a prior that wraps another carries the rows of the prior that it wraps. A [`CrossSectionalFactorPrior`](@ref) drops the rows that its Descriptors warm up over, which the data decides, so it answers `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: return `T` when [`keeps_observations`](@ref) returns `true` for `pe`, and `nothing` otherwise.
 2. An [`EmpiricalPrior`](@ref): return `T`, or `pe.max_scenarios` when it is set and smaller, because [`scenario_window`](@ref) keeps the last `pe.max_scenarios` rows.
 3. A prior whose result carries the scenarios of the prior in its `pe` field: return this function of `pe.pe`. A factor prior fits `pe.pe` on the factor returns, which have the rows of the asset returns.
 4. An [`AugmentedBlackLittermanPrior`](@ref): return the smaller of this function of `pe.a_pe` and of `pe.f_pe`, because the result keeps the rows that both nested priors keep. Return `nothing` when either one is `nothing`.
 5. An [`OpinionPoolingPrior`](@ref): return `T` when `pe.pe1` is `nothing`, and this function of `pe.pe1` otherwise.

# Arguments

  - $(arg_dict[:pe])
  - `T`: The number of observations that the fit of `pe` receives.

# Returns

  - `n::Option{Int}`: The number of scenarios of the fit, or `nothing` when the data decides it.

# Related

  - [`scenario_count`](@ref)
  - [`keeps_observations`](@ref)
  - [`scenario_window`](@ref)
"""
function kept_scenarios(pe::AbstractPriorEstimator, T::Integer)::Option{Int}
    return keeps_observations(pe) ? T : nothing
end
function kept_scenarios(pe::EmpiricalPrior, T::Integer)::Option{Int}
    return isnothing(pe.max_scenarios) ? T : min(pe.max_scenarios, T)
end
function kept_scenarios(pe::Union{FactorPrior, FactorBlackLittermanPrior,
                                  BlackLittermanPrior, BayesianBlackLittermanPrior,
                                  EntropyPoolingPrior, MeucciEntropyPoolingPrior,
                                  HighOrderPriorEstimator, HighOrderFactorPriorEstimator},
                        T::Integer)::Option{Int}
    return kept_scenarios(pe.pe, T)
end
function kept_scenarios(pe::AugmentedBlackLittermanPrior, T::Integer)::Option{Int}
    a = kept_scenarios(pe.a_pe, T)
    f = kept_scenarios(pe.f_pe, T)
    return isnothing(a) || isnothing(f) ? nothing : min(a, f)
end
function kept_scenarios(pe::OpinionPoolingPrior, T::Integer)::Option{Int}
    return isnothing(pe.pe1) ? T : kept_scenarios(pe.pe1, T)
end
"""
    scenario_count(pe::AbstractPriorEstimator, X::MatNum, F::Option{<:MatNum},
                   pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    scenario_count(pe::Union{FactorPrior, FactorBlackLittermanPrior}, X::MatNum,
                   F::MatNum, pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    scenario_count(pe::Union{BlackLittermanPrior, BayesianBlackLittermanPrior,
                             EntropyPoolingPrior, MeucciEntropyPoolingPrior,
                             HighOrderPriorEstimator, HighOrderFactorPriorEstimator},
                   X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    scenario_count(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum,
                   pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
    scenario_count(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum},
                   pnl::Option{<:AssetPanel}; kwargs...)
    scenario_count(pe::CrossSectionalFactorPrior, X::MatNum, F::Option{<:MatNum},
                   pnl::AssetPanel; strict::Bool = false, ne::Option{<:VecStr} = nothing,
                   E::Option{<:MatNum} = nothing, iv::Option{<:MatNum} = nothing,
                   ivpa::Option{<:Num_VecNum} = nothing, kwargs...)

Return the number of scenarios that the fit of `pe` on `X`, `F` and `pnl` carries, with the least work that states it.

A prior that reweights observations, such as [`EntropyPoolingPrior`](@ref), checks a caller's prior probabilities against the scenarios of the prior that it wraps, before it fits that prior under them. A fit states the count, but the count needs only the steps that drop rows. When the configuration states the count, [`kept_scenarios`](@ref) gives it with no fit. A [`CrossSectionalFactorPrior`](@ref) drops the rows that its Descriptors warm up over, which the data decides, so its count runs its exposure stage, and not its regression, its factor moments or its assembly.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other prior: return [`kept_scenarios`](@ref) of `pe` on `size(X, 1)` observations when it is not `nothing`. Otherwise fit `pe` and return the row count of its scenarios.
 2. A [`FactorPrior`](@ref) or a [`FactorBlackLittermanPrior`](@ref): return this function of `pe.pe` on `F` alone, as the fit of `pe` fits `pe.pe`. The asset rows are cut to the rows of the factor prior.
 3. A prior whose result carries the scenarios of the prior in its `pe` field: return this function of `pe.pe` on `X`, `F` and `pnl`.
 4. An [`AugmentedBlackLittermanPrior`](@ref): return the smaller of this function of `pe.a_pe` on `X` and `pnl`, and of `pe.f_pe` on `F` alone, because the result keeps the rows that both nested priors keep.
 5. An [`OpinionPoolingPrior`](@ref): return `size(X, 1)` when `pe.pe1` is `nothing`, and this function of `pe.pe1` otherwise.
 6. A [`CrossSectionalFactorPrior`](@ref): take the rows `rw` after the Descriptor warm-up with [`cross_sectional_exposure_stage`](@ref), and check them against the exposure lag with [`assert_cross_sectional_lag_rows`](@ref). The regression answers `n = length(rw) - pe.lag` factor returns. Return [`kept_scenarios`](@ref) of `pe.pe` on `n` observations when it is not `nothing`. Otherwise take the factor returns with [`cross_sectional_factor_scenarios`](@ref), and return this function of `pe.pe` on them alone.

# Arguments

  - $(arg_dict[:pe])
  - `X`: Asset returns matrix, already oriented.
  - `F`: Optional factor returns matrix, already oriented.
  - $(arg_dict[:pnl_prior])
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `ne`, `E`, `iv`, `ivpa`: As in the returns-matrix method of [`prior`](@ref), with `E` already oriented.
  - `kwargs...`: Additional keyword arguments of the fit of `pe`.

# Validation

  - The rules of every step of the fit that the method runs.

# Returns

  - `T::Int`: The number of scenarios that the fit of `pe` carries.

# Related

  - [`kept_scenarios`](@ref)
  - [`keeps_observations`](@ref)
  - [`ep_wrapped_prior`](@ref)
  - [`cross_sectional_factor_scenarios`](@ref)
"""
function scenario_count(pe::AbstractPriorEstimator, X::MatNum, F::Option{<:MatNum},
                        pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)::Int
    n = kept_scenarios(pe, size(X, 1))
    if !isnothing(n)
        return n
    end
    return size(prior(pe, X, F, pnl; strict = strict, kwargs...).X, 1)
end
function scenario_count(pe::Union{FactorPrior, FactorBlackLittermanPrior}, ::MatNum,
                        F::MatNum, ::Option{<:AssetPanel}; strict::Bool = false,
                        kwargs...)::Int
    return scenario_count(pe.pe, F, nothing, nothing; strict = strict)
end
function scenario_count(pe::Union{BlackLittermanPrior, BayesianBlackLittermanPrior,
                                  EntropyPoolingPrior, MeucciEntropyPoolingPrior,
                                  HighOrderPriorEstimator, HighOrderFactorPriorEstimator},
                        X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                        kwargs...)::Int
    return scenario_count(pe.pe, X, F, pnl; kwargs...)
end
function scenario_count(pe::AugmentedBlackLittermanPrior, X::MatNum, F::MatNum,
                        pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)::Int
    return min(scenario_count(pe.a_pe, X, nothing, pnl; strict = strict, kwargs...),
               scenario_count(pe.f_pe, F, nothing, nothing; strict = strict))
end
function scenario_count(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum},
                        pnl::Option{<:AssetPanel}; kwargs...)::Int
    return isnothing(pe.pe1) ? size(X, 1) : scenario_count(pe.pe1, X, F, pnl; kwargs...)
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
    assert_cross_sectional_lag_rows(t::Integer, lag::Integer)

Check that the `t` rows after the Descriptor warm-up of a [`CrossSectionalFactorPrior`](@ref) leave at least two factor returns after the exposure lag `lag`.

The regression reads the exposures of the row `lag` observations before each return, so it answers `t - lag` factor returns. The factor prior states a covariance of them, and a covariance of one observation is not a number. [`cross_sectional_regression_stage`](@ref) and [`scenario_count`](@ref) both read this check, so a count refuses the rows that the fit refuses.

# Arguments

  - `t`: The number of rows after the Descriptor warm-up.
  - `lag`: The exposure lag, `pe.lag`.

# Validation

  - `t > lag`. Raises an `ArgumentError`.
  - `t - lag >= 2`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_regression_stage`](@ref)
  - [`scenario_count`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function assert_cross_sectional_lag_rows(t::Integer, lag::Integer)
    @argcheck(t > lag,
              ArgumentError("the exposures lag the returns by lag = $(lag), so a fit needs more than $(lag) observations after the Descriptor warm-up, and $(t) are left. Give more observations, shorten the warm-up of the Descriptors, or lower lag."))
    @argcheck(t - lag >= 2,
              ArgumentError("the factor prior states a covariance of the factor returns, and a covariance of one observation is not a number, so a fit needs at least two observations after the Descriptor warm-up and the exposure lag of $(lag), and $(t - lag) is left. Give more observations, or shorten the warm-up of the Descriptors."))
    return nothing
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
    assert_cross_sectional_lag_rows(length(rw), pe.lag)
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
"""
    cross_sectional_cached_block(pe::CrossSectionalFactorPrior, pr::Nothing, X::MatNum)
    cross_sectional_cached_block(pe::CrossSectionalFactorPrior, pr::AbstractPriorResult,
                                 X::MatNum)
    cross_sectional_cached_block(pe::CrossSectionalFactorPrior, pr::LowOrderPrior, X::MatNum)

Return the block of the fit `pr` of a [`CrossSectionalFactorPrior`](@ref) when it holds the cache that a refit under new scenario weights reads, and `nothing` otherwise.

A fit of `prior` with `cache_reg = true` stores the cache. [`EntropyPoolingPrior`](@ref) and [`MeucciEntropyPoolingPrior`](@ref) fit the prior that they wrap with it, and give the result of that fit to [`scenario_prior`](@ref) and [`scenario_factory`](@ref). The regression, the cross-sectional weights, the residual block and the Return Forecast of a refit are those of the fit, so the refit reads them from the block and runs no regression. A result with no cache gives `nothing`, and the refit then runs the whole fit.

# Algorithm

The method that Julia selects is the algorithm.

 1. No result, or a result of any other type: return `nothing`.
 2. A [`LowOrderPrior`](@ref): return its block `rr` when every condition holds, and `nothing` otherwise:
     1. `rr` is a [`CrossSectionalFactorModel`](@ref) with a regression `csr`, a recorded live-factor mask `csr.lv` and an active-mask history `amr`.
     2. `rr` has one row of `M` for each column of `X`, so the cache describes the asset universe of the refit.
     3. When a Calibration Slot of `pe` reads the Return Forecast history, as [`reads_forecast_history`](@ref) answers, and `pe` states a Return Forecast Estimator, `rr` carries that history in `hist`.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `pr`: The result of the fit that the views read, or `nothing`.
  - $(arg_dict[:X])

# Returns

  - `rr::Option{<:CrossSectionalFactorModel}`: The block that holds the cache, or `nothing`.

# Related

  - [`scenario_prior`](@ref)
  - [`scenario_factory`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`cross_sectional_moment_prior`](@ref)
"""
function cross_sectional_cached_block(::CrossSectionalFactorPrior,
                                      ::Option{<:AbstractPriorResult}, ::MatNum)::Nothing
    return nothing
end
function cross_sectional_cached_block(pe::CrossSectionalFactorPrior, pr::LowOrderPrior,
                                      X::MatNum)::Option{<:CrossSectionalFactorModel}
    rr = pr.rr
    if !isa(rr, CrossSectionalFactorModel) ||
       isnothing(rr.amr) ||
       isnothing(rr.csr) ||
       isnothing(rr.csr.lv) ||
       size(rr.M, 1) != size(X, 2)
        return nothing
    end
    reads = (reads_forecast_history(pe.lambda) || reads_forecast_history(pe.c)) &&
            !isnothing(pe.rfe)
    return reads && isnothing(rr.hist) ? nothing : rr
end
"""
    cross_sectional_cached_scenarios(rr::CrossSectionalFactorModel) -> NamedTuple

Return the factor returns that the factor prior of a [`CrossSectionalFactorPrior`](@ref) is fitted on, from the block of a fit that stores the cache.

The regression stage gives these factor returns to the fit, as [`cross_sectional_factor_scenarios`](@ref) states. The block of a fit with `cache_reg = true` holds the regression and its live-factor mask, so [`scenario_prior`](@ref) and [`scenario_factory`](@ref) read the same factor returns from it and run no regression.

# Algorithm

 1. Take the factor returns of the block with [`cross_sectional_factor_returns`](@ref), the estimated factors and then the observed ones, giving `fa`.
 2. Extend the live-factor mask `rr.csr.lv` of the estimated factors with `true` for each observed factor, giving `lv`. An Observed Factor carries the return the caller observed, so it is never empty.
 3. Take the columns of `fa` that `lv` marks, giving `f`.

# Arguments

  - `rr`: The block of a fit that stores the cache, as [`cross_sectional_cached_block`](@ref) answers.

# Returns

  - `fa::MatNum`: The factor returns on the combined reduced factor axis.
  - `f::MatNum`: The factor returns of the factors that are not empty, which the factor prior reads.
  - `lv::BitVector`: `true` at each factor of `fa` that is not empty.

# Related

  - [`cross_sectional_cached_block`](@ref)
  - [`cross_sectional_factor_scenarios`](@ref)
"""
function cross_sectional_cached_scenarios(rr::CrossSectionalFactorModel)
    fa = cross_sectional_factor_returns(rr)
    lv = vcat(rr.csr.lv, trues(size(fa, 2) - length(rr.csr.lv)))
    return (; fa = fa, f = all(lv) ? fa : fa[:, lv], lv = lv)
end
"""
    cross_sectional_forecast_split(ofit::AbstractOrthogonalForecastFit, rf::Nothing,
                                   csfm::CrossSectionalFactorModel,
                                   cre::AbstractCrossSectionalRegressionEstimator)
    cross_sectional_forecast_split(ofit::AbstractOrthogonalForecastFit,
                                   rf::AbstractReturnForecastResult,
                                   csfm::CrossSectionalFactorModel,
                                   cre::AbstractCrossSectionalRegressionEstimator)

Split a Return Forecast Result against the latest Factor Exposures of the estimated factors of a [`CrossSectionalFactorModel`](@ref), giving its spanned coefficients and its orthogonal part before the Orthogonal Forecast Scale.

[`cross_sectional_return_forecast`](@ref) fits the forecast and splits it here. The refit of [`scenario_prior`](@ref) reads the forecast that the block carries, and splits it here, so it fits no member.

# Algorithm

 1. No Return Forecast Result: return `nothing` for both parts.
 2. Split `rf.mu` against the latest exposures of the estimated factors with [`cross_sectional_alpha_split`](@ref), under the weights of the latest row of `csfm.rw`, giving `g` and `ap`. The observed factors are the trailing columns of `L`, and the split leaves them out, so `g` has one entry per column of `csr.f`.
 3. Scale `ap` with [`orthogonal_forecast_rescale`](@ref), which changes it under [`OrthogonalPartCalibration`](@ref) alone.

# Arguments

  - `ofit`: The Orthogonal Forecast Fit of the prior.
  - `rf`: The Return Forecast Result, or `nothing`.
  - `csfm`: The factor-model block.
  - `cre`: Cross-Sectional Regression Estimator of the split.

# Validation

  - The rules of [`cross_sectional_alpha_split`](@ref).

# Returns

  - `g::Option{<:VecNum}`: The spanned coefficients, or `nothing`.
  - `ap::Option{<:VecNum}`: The orthogonal part before the Orthogonal Forecast Scale, one entry per asset, or `nothing`.

# Related

  - [`cross_sectional_return_forecast`](@ref)
  - [`cross_sectional_alpha_split`](@ref)
  - [`orthogonal_forecast_rescale`](@ref)
"""
function cross_sectional_forecast_split(::AbstractOrthogonalForecastFit, ::Nothing,
                                        ::CrossSectionalFactorModel,
                                        ::AbstractCrossSectionalRegressionEstimator)
    return (; g = nothing, ap = nothing)
end
function cross_sectional_forecast_split(ofit::AbstractOrthogonalForecastFit,
                                        rf::AbstractReturnForecastResult,
                                        csfm::CrossSectionalFactorModel,
                                        cre::AbstractCrossSectionalRegressionEstimator)
    rw = csfm.rw
    #! The observed factors are the trailing columns of `L`, and the fit observes their
    #! returns, so the forecast splits against the loadings the regression estimated.
    L = view(csfm.L, :, 1:size(csfm.csr.f, 2))
    (; g, ap) = cross_sectional_alpha_split(cre, rf.mu, L, @view(rw[size(rw, 1), :]))
    return orthogonal_forecast_rescale(ofit, rf, g, ap)
end
"""
    cross_sectional_moment_prior(pe::CrossSectionalFactorPrior, f_pr::NamedTuple,
                                 csfm::CrossSectionalFactorModel, sp::NamedTuple, L::MatNum,
                                 am::AbstractVector{Bool}, Xs::MatNum, oX::Option{<:MatNum};
                                 kwargs...) -> LowOrderPrior

Build the Prior Result of a [`CrossSectionalFactorPrior`](@ref) from its block and its factor moments, with the steps of [`cross_sectional_assemble`](@ref) that read the factor moments.

The fit and the call with no data of the carry fold reach it through [`cross_sectional_assemble`](@ref). The refit of [`scenario_prior`](@ref) under new scenario weights reaches it with the block of the fit that the views read. The block, the scenarios and the Return Forecast read no scenario weight of the factor prior, so the refit takes them from that fit, and only the steps that this function runs read the new weights.

# Algorithm

 1. Take the ready factors of `f_pr` with [`cross_sectional_ready_factors`](@ref) and the determined assets with [`cross_sectional_determined`](@ref). Refuse a fit that determines no asset with [`assert_cross_sectional_factor_moments`](@ref).
 2. Resolve the Spanned Shrinkage `lambda` and the Orthogonal Forecast Scale `c` with [`cross_sectional_calibration`](@ref), against the factor moments `f_pr` and the block `csfm`.
 3. Write `b = c ap`, the Return Forecast Result and the resolved `lambda` and `c` onto the block with [`cross_sectional_forecast_block`](@ref), giving `rr`. Blend the spanned part into the mean of the estimated factors with [`cross_sectional_forecast_mu`](@ref) under the resolved `lambda`, and keep the mean of the observed factors, giving `f_mu`.
 4. Expand the blended factor moments onto the raw factor axis with [`cross_sectional_expand`](@ref), with the basis of the block.
 5. Take the investable assets `idx` with [`cross_sectional_investable`](@ref), from the active assets `am` of the latest observation, the latest idiosyncratic variances of `csfm.vs`, and the assets that the ready factors of the blended mean determine.
 6. Lift the reduced factor distribution onto the assets with [`cross_sectional_lift`](@ref), and add `b` to the expected return it answers.
 7. Assemble a [`LowOrderPrior`](@ref) over `Xs`, with `oX` under `o_X`, the three lifted moments, the factor prior's `w` divided by its sum, its `ens`, `kld` and `ow`, the block under `rr`, and the expanded factor prior under `fpr`. The factor prior keeps the cache `pc` of a pool that answered it, so the refit of the pool reads it from `fpr`. The factor prior holds the factor returns of the block on the raw axis, on the last rows that the scenarios keep.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `f_pr`: The factor moments that [`cross_sectional_factor_moments`](@ref) states over the combined reduced factor returns.
  - `csfm`: The factor-model block, with the regression, the variance history, the idiosyncratic covariance, the basis and the factor returns of the fit.
  - `sp`: The split of the Return Forecast, `(; rf, g, ap, hist)`.
  - `L`: The loadings of the latest observation on the combined reduced factor axis.
  - `am`: The active mask of the latest observation, one entry per asset.
  - `Xs`: The asset return scenarios.
  - `oX`: The base-currency returns of the scenario rows, or `nothing`.
  - `kwargs...`: Additional keyword arguments passed to [`cross_sectional_lift`](@ref).

# Validation

  - The factor prior determines the factor moments of at least one asset. Raises an [`IsNonFiniteError`](@ref).
  - At least one asset is investable at the latest observation. Raises an [`IsEmptyError`](@ref).
  - The rules of every verb the algorithm names. A resolved `lambda` lies in `[0, 1]`, and a resolved `c` is finite and `>= 0`.

# Returns

  - `pr::LowOrderPrior`: The prior on the full asset universe, as [`prior`](@ref) states it.

# Related

  - [`cross_sectional_assemble`](@ref)
  - [`scenario_prior`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_moment_prior(pe::CrossSectionalFactorPrior, f_pr::NamedTuple,
                                      csfm::CrossSectionalFactorModel, sp::NamedTuple,
                                      L::MatNum, am::AbstractVector{Bool}, Xs::MatNum,
                                      oX::Option{<:MatNum}; kwargs...)
    csr = csfm.csr
    T = size(csr.f, 1)
    # A factor the factor prior states no moment for changes no moment of an asset with a
    # zero loading on it, so the fit refuses only when no asset is determined (#1510). The
    # check runs before the calibration, which reads the factor moments.
    det = cross_sectional_determined(L, cross_sectional_ready_factors(f_pr.mu, f_pr.sigma))
    assert_cross_sectional_factor_moments(f_pr.mu, f_pr.sigma, T, det)
    (; lambda, c) = cross_sectional_calibration(pe, f_pr, csfm, sp)
    rr = cross_sectional_forecast_block(csfm, sp, lambda, c)
    # The forecast spans the estimated factors alone, so the blend reaches their mean and
    # the mean of each observed factor is the one the factor prior states.
    Ke = size(csr.f, 2)
    f_mu = @views vcat(cross_sectional_forecast_mu(lambda, f_pr.mu[1:Ke], sp.g),
                       f_pr.mu[(Ke + 1):end])
    # The block's basis covers its own rows, so the moments expand with the ratios of its
    # last row, the last fitted observation.
    ex = cross_sectional_expand(csfm.fcb, 1:T, f_mu, f_pr.sigma)
    # The blend at `lambda = 0` reads no factor mean, so the investable assets read the
    # ready factors of the blended mean, which hold every factor ready before it.
    rdy = cross_sectional_ready_factors(f_mu, f_pr.sigma)
    idx = cross_sectional_investable(am, @view(csfm.vs[end, :]),
                                     cross_sectional_determined(L, rdy))
    @argcheck(!isempty(idx),
              IsEmptyError("no asset is investable at the latest observation: every asset is either inactive, or carries a non-finite idiosyncratic variance or Factor Exposure, or loads on a factor whose moments the factor prior does not state. Give more observations, or widen the active mask of the Asset Panel."))
    # A factor prior with a Scenario Cap carries its last rows alone, and the scenarios pair
    # the last rows of each history, so the factor returns keep the rows the scenarios keep
    # (#1384).
    rs = (T - size(Xs, 1) + 1):T
    # The systematic block is assembled from the factor covariance, so its denoiser reads the
    # count of that covariance (#1893).
    lift = cross_sectional_lift(pe.srep, pe.mp, L, f_mu, f_pr.sigma, csfm.esigma, idx, Xs,
                                something(f_pr.eoc[2], size(f_pr.X, 1));
                                mtx_sqrt = pe.mtx_sqrt, kwargs...)
    fX = something(csfm.fr, cross_sectional_factor_returns(csfm))
    fpr = LowOrderPrior(; X = fX[rs, :], mu = ex.mu, sigma = ex.sigma, w = f_pr.w,
                        ens = f_pr.ens, eoc = f_pr.eoc, kld = f_pr.kld, ow = f_pr.ow,
                        pc = f_pr.pc)
    return LowOrderPrior(; X = Xs, o_X = oX, mu = lift.mu + rr.b, sigma = lift.sigma,
                         chol = lift.chol, w = f_pr.w, ens = f_pr.ens, eoc = f_pr.eoc,
                         kld = f_pr.kld, ow = f_pr.ow, rr = rr, fpr = fpr)
end
"""
    cross_sectional_stored_cache(csfm::CrossSectionalFactorModel, cache_reg::Bool,
                                 amr::AbstractMatrix{Bool}, hist::Option{<:MatNum})
        -> CrossSectionalFactorModel

Store the cache that the refit of an entropy pooling prior reads on the block of a [`CrossSectionalFactorPrior`](@ref).

[`cross_sectional_assemble`](@ref) calls it after the Return Forecast. A plain fit stores nothing, and a fit with `cache_reg = true` stores the active mask of the fitted rows in `amr` and the Return Forecast history in `hist`, which is `nothing` when no Calibration Slot reads it.

# Arguments

  - `csfm`: The factor-model block.
  - `cache_reg`: If `true`, store the cache.
  - `amr`: The active mask of the fitted rows.
  - `hist`: The Return Forecast history, or `nothing`.

# Returns

  - `csfm::CrossSectionalFactorModel`: The block, with the cache when `cache_reg` is `true`.

# Related

  - [`cross_sectional_assemble`](@ref)
  - [`cross_sectional_cached_block`](@ref)
"""
function cross_sectional_stored_cache(csfm::CrossSectionalFactorModel, cache_reg::Bool,
                                      amr::AbstractMatrix{Bool}, hist::Option{<:MatNum})
    return cache_reg ? rebuild_estimator(csfm, (; amr = amr, hist = hist)) : csfm
end
"""
    cross_sectional_kept_cache(f_pr::NamedTuple, pf::Option{<:AbstractPriorResult}) -> NamedTuple

Keep the cache of a pool in the factor slot of a [`CrossSectionalFactorPrior`](@ref) on the factor moments of its refit.

The refit of [`scenario_prior`](@ref) fits the model of the pool under the new weights, which is `pe2` of the pool and keeps no cache. So the factor prior `fpr` of the refit would carry no cache, and an entropy pooling prior around a nested one would run the opinions of the pool again. The fit that the views read gives the cache in `pf`, the factor prior of its result, and the refit keeps it.

# Arguments

  - `f_pr`: The factor moments of the refit, as [`cross_sectional_factor_moments`](@ref) states them.
  - `pf`: The factor prior of the result of the fit that the views read, or `nothing`.

# Returns

  - `f_pr::NamedTuple`: `f_pr`, with `pc` from `pf` when `f_pr` carries none.

# Related

  - [`opinion_pooling_cache`](@ref)
  - [`scenario_prior`](@ref)
  - [`cross_sectional_moment_prior`](@ref)
"""
function cross_sectional_kept_cache(f_pr::NamedTuple, pf::Option{<:AbstractPriorResult})
    return isnothing(f_pr.pc) ? merge(f_pr, (; pc = opinion_pooling_cache(pf))) : f_pr
end
# The refit of a Cross-Sectional Factor Prior under the posterior weights holds its model, as
# the refit of a `FactorPrior` does (ADR 0205), and gives the weights to `pe` alone, on the
# factor returns that it reads (#1685). A fit that stores the cache gives those factor
# returns with no regression (#1751). `scenario_factory` documents every method.
function scenario_factory(pe::CrossSectionalFactorPrior, w,
                          pr::Option{<:AbstractPriorResult}, X::MatNum, F::Option{<:MatNum},
                          pnl::Option{<:AssetPanel}; strict::Bool = false,
                          cache_reg::Bool = false, kwargs...)
    rr = cross_sectional_cached_block(pe, pr, X)
    f = if isnothing(rr)
        cross_sectional_factor_scenarios(pe, X, F, pnl; kwargs...).f
    else
        cross_sectional_cached_scenarios(rr).f
    end
    return rebuild_estimator(pe,
                             (;
                              pe = scenario_factory(pe.pe, w, scenario_factor_result(pr), f,
                                                    nothing, nothing; strict = strict)))
end
# The refit of an entropy pooling prior fits the factor prior of `scenario_factory` on the
# factor returns of the regression. A fit that stores the cache gives the regression, the
# scenarios and the Return Forecast of the fit that the views read, so the refit runs the
# steps that read the factor moments alone (#1751). Otherwise it runs the regression stage
# once. `scenario_prior` documents every method.
function scenario_prior(pe::CrossSectionalFactorPrior, w, pr::Option{<:AbstractPriorResult},
                        X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                        strict::Bool = false, cache_reg::Bool = false, kwargs...)
    rr = cross_sectional_cached_block(pe, pr, X)
    pf = scenario_factor_result(pr)
    if isnothing(rr)
        (; f, rs, kw) = cross_sectional_factor_scenarios(pe, X, F, pnl; kwargs...)
        fpe = scenario_factory(pe.pe, w, pf, f, nothing, nothing; strict = strict)
        f_pr = cross_sectional_kept_cache(cross_sectional_factor_moments(fpe, pe.f_mp,
                                                                         rs.ca.f, rs.lv;
                                                                         strict = strict,
                                                                         kw...), pf)
        return cross_sectional_assemble(pe, f_pr, rs.ca, rs.fit, rs.rde;
                                        cache_reg = cache_reg, kw...)
    end
    # The refit runs no regression stage, so it drops the keywords that the stage alone reads.
    kw = Base.structdiff(values(kwargs), NamedTuple{(:ne, :E, :iv, :ivpa, :ts)})
    (; fa, f, lv) = cross_sectional_cached_scenarios(rr)
    fpe = scenario_factory(pe.pe, w, pf, f, nothing, nothing; strict = strict)
    f_pr = cross_sectional_kept_cache(cross_sectional_factor_moments(fpe, pe.f_mp, fa, lv;
                                                                     strict = strict,
                                                                     kw...), pf)
    sp = (; rf = rr.rf, cross_sectional_forecast_split(pe.ofit, rr.rf, rr, pe.cre)...,
          hist = rr.hist)
    am = view(rr.amr, size(rr.amr, 1), :)
    return cross_sectional_moment_prior(pe, f_pr, rr, sp, rr.L, am, pr.X, pr.o_X; kw...)
end
# A caller's prior probabilities weigh the fit, so every field takes them on the rows of the
# factor returns, which each field reads (ADR 0209). The regression answers one factor return
# for each row after the Descriptor warm-up and the exposure lag, so the count of those rows
# runs the exposure stage alone (#1751).
function prior_probability_factory(pe::CrossSectionalFactorPrior, w, X::MatNum,
                                   F::Option{<:MatNum}, pnl::AssetPanel;
                                   ne::Option{<:VecStr} = nothing,
                                   E::Option{<:MatNum} = nothing,
                                   iv::Option{<:MatNum} = nothing,
                                   ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
    (; rw) = cross_sectional_exposure_stage(pe, X, F, pnl; ne = ne, E = E, iv = iv,
                                            ivpa = ivpa)
    assert_cross_sectional_lag_rows(length(rw), pe.lag)
    n = length(rw) - pe.lag
    inner = prior_probability_factory(pe.pe, w, n)
    return rebuild_estimator(factory(pe, scenario_weights(w, n)), (; pe = inner))
end
# The regression answers one factor return for each row after the Descriptor warm-up and the
# exposure lag, so the count runs the exposure stage alone. The factor prior then counts its
# scenarios from that number, and only a factor prior that the data decides reads the
# factor returns. `scenario_count` documents every method.
function scenario_count(pe::CrossSectionalFactorPrior, X::MatNum, F::Option{<:MatNum},
                        pnl::AssetPanel; strict::Bool = false,
                        ne::Option{<:VecStr} = nothing, E::Option{<:MatNum} = nothing,
                        iv::Option{<:MatNum} = nothing,
                        ivpa::Option{<:Num_VecNum} = nothing, kwargs...)::Int
    (; rw) = cross_sectional_exposure_stage(pe, X, F, pnl; ne = ne, E = E, iv = iv,
                                            ivpa = ivpa)
    assert_cross_sectional_lag_rows(length(rw), pe.lag)
    n = kept_scenarios(pe.pe, length(rw) - pe.lag)
    if !isnothing(n)
        return n
    end
    (; f) = cross_sectional_factor_scenarios(pe, X, F, pnl; ne = ne, E = E, iv = iv,
                                             ivpa = ivpa, kwargs...)
    return scenario_count(pe.pe, f, nothing, nothing; strict = strict)
end
function assert_scenario_model(pe::CrossSectionalFactorPrior)
    return assert_scenario_model(pe.pe)
end
"""
    scenario_moment_counts(pe::EmpiricalPrior, X::MatNum, t::Integer,
                           T_mu::Option{<:Real} = nothing; kwargs...)
        -> Tuple{Option{<:Real}, Option{<:Real}}

State the effective number of observations of the mean and of the covariance of an empirical Prior Result, in the form of its field `eoc`.

`pe.me` fits `mu` and `pe.ce` fits `sigma` over every row of `X`. Each estimator states the number of observations that its moment is worth with [`effective_observation_count`](@ref). That is `size(X, 1)` for an unweighted estimator, and the count of its weights for a weighted one, such as the Kish count of an exponentially weighted covariance. The two estimators can carry different weights, so each moment has its own count.

`T_mu` is the count of the mean that the fit of `mu` stated, from [`mean_with_count`](@ref). A [`PriorExpectedReturns`](@ref) states it, because its count is known only after the fit of its prior. When `T_mu` is `nothing`, the function reads the count of `pe.me` with [`effective_observation_count`](@ref).

A Scenario Cap cuts the rows that the result carries to the last `t`, and leaves the moments fitted over every row. [`scenario_window`](@ref) makes the cut. A consumer that reads `size(pr.X, 1)` then reads `t` rows for moments fitted over `size(X, 1)`. So an entry of `eoc` states the count whenever it is not `t`, and `nothing` otherwise, through [`moment_count_entry`](@ref). A fit with no cap and no weights states `(nothing, nothing)`.

It is the same function in batch and in `prior(pe)` with no data after a fold, as [`scenario_window`](@ref) is. The fold keeps the rows in its buffer, so both routes read the same counts over the same rows.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: The returns that the moments were fitted over, `observations × assets`, **before** the cut. Under a horizon they are the log-returns.
  - `t`: The number of rows that the result carries, after the cut.
  - `T_mu`: The count of the mean that the fit of `mu` stated, or `nothing`.
  - `kwargs...`: The keywords that the prior gives to the fits of `mu` and `sigma`. The function gives them to [`effective_observation_count`](@ref), so a count reads the same values as its fit.

# Returns

  - `eoc::Tuple`: The entries `(T_mu, T_sigma)` of the field `eoc`.

# Related

  - [`scenario_window`](@ref)
  - [`effective_observation_count`](@ref)
  - [`mean_with_count`](@ref)
  - [`moment_count_entry`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function scenario_moment_counts(pe::EmpiricalPrior, X::MatNum, t::Integer,
                                T_mu::Option{<:Real} = nothing; kwargs...)
    # The count of a mean that a fit states wins, so the second fit of a prior is not paid.
    T_mu = isnothing(T_mu) ? effective_observation_count(pe.me, X; kwargs...) : T_mu
    return (moment_count_entry(T_mu, t),
            moment_count_entry(effective_observation_count(pe.ce, X; kwargs...), t))
end
"""
    mean_with_count(me::AbstractExpectedReturnsEstimator, X::MatNum,
                    pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    mean_with_count(me::AbstractExpectedReturnsEstimator; kwargs...)

Fit the mean of `me` as a vector, and return it with the count of the mean that the fit states.

An [`EmpiricalPrior`](@ref) reads its mean through this function, and gives the count to [`scenario_moment_counts`](@ref). Most mean estimators state their count before the fit, with [`effective_observation_count`](@ref), so the function returns `nothing` as their count. A [`PriorExpectedReturns`](@ref) reads the `mu` of a prior that it fits, and a pooling prior knows its count only after its fit. So its method returns entry 1 of the `eoc` of that prior result, and the prior is fitted once.

The form with no data reads a folded estimator, as `Statistics.mean(me)` does.

# Algorithm

The method that Julia selects is the algorithm.

 1. On a mean estimator with no method of its own, fit `mu` with `Statistics.mean`, and return `vec(mu)` with the count `nothing`.
 2. On [`PriorExpectedReturns`](@ref), fit the prior result `pr` with [`prior`](@ref), with no factor returns. Return `vec(pr.mu)` with the count of the mean of `pr`, from [`moment_observation_count`](@ref). The form with no data reads `prior(me.pe)`.

# Arguments

  - `me`: The expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Forwarded to the fit of the mean.

# Validation

  - Everything the fit of the mean refuses.

# Returns

  - `(mu, T_mu)::Tuple`: The mean vector `mu`, and its count `T_mu`, a `Real` or `nothing`.

# Related

  - [`scenario_moment_counts`](@ref)
  - [`effective_observation_count`](@ref)
  - [`PriorExpectedReturns`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
function mean_with_count(me::AbstractExpectedReturnsEstimator, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    return vec(Statistics.mean(me, X, pnl; dims = dims, kwargs...)), nothing
end
function mean_with_count(me::AbstractExpectedReturnsEstimator; kwargs...)
    return vec(Statistics.mean(me; kwargs...)), nothing
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
                              strict; id = @log_id())
            scenario_fill_remember!(named, report)
        end
    end
    Xf = copy(X)
    for (t, i) in filled
        Xf[t, i] = zero(eltype(Xf))
    end
    return Xf
end
