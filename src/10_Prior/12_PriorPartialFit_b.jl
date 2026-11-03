"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                             <:Option{<:PriorCarryState}}; strict::Bool = false, kwargs...)
    prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                             <:Option{<:PriorCarryState}}; strict::Bool = false, kwargs...)

Reads an [`EmpiricalPrior`](@ref) out of its fold, with no data matrix.

`mu` comes from `pe.me`, `sigma` comes from `pe.ce`, and `X` is the matrix that the carry state already holds, less the observations that it marks with a Shared Series gap. The members never folded those observations, and [`shared_series_buffer`](@ref) drops them from the rows, so the result is the batch fit over the observations that the batch fit keeps. The method copies the rows, and does not refit a member that folded. The horizon method then applies the algebra of the batch method through [`horizon_moments!`](@ref), which holds that algebra once.

`pe.max_scenarios` cuts the rows that the result carries, as it does in batch, and the fill runs over the rows that remain. `eoc` holds the effective number of observations of `mu` and of `sigma` over the buffered rows, through [`scenario_moment_counts`](@ref). When the cap cuts, an unweighted moment counts the number of observations folded, so a consumer that prices a sample size reads ``T`` and not the number of rows carried. The named-asset set of the carry state limits the notice of the fill, so a walk-forward names an asset at the first step whose fill reaches it, and at no later step.

A member that the method refits takes the [`AssetPanel`](@ref) that [`buffer_asset_panel`](@ref) rebuilds from the carried buffer, as the batch fit gives it `rd.pnl`. The buffer records the active mask of each step. So a member that the method refits answers on the Coverage Universe of a batch fit, on a gapped sample and under a time-varying panel. A member that folds read the active mask at each step of its fold, as the keyword form of its verb reads it.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X})\\,, \\\\
\\mathbf{X} &= \\begin{bmatrix} \\boldsymbol{x}_1 & \\cdots & \\boldsymbol{x}_T \\end{bmatrix}^\\intercal\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - $(math_dict[:X_returns])
  - $(math_dict[:x_t_obs])
  - $(math_dict[:T])

The equality holds for `mu`, `sigma`, `X` and `eoc`, with and without a horizon, and under a Scenario Cap. In floating point, the moments of a member that folds differ from the batch moments by rounding, and every other entry is equal.

# Algorithm

 1. Read the carry state with [`partial_fit_cache`](@ref), giving `state`.
 2. Resolve the fill limit against the coverage floor of the members with [`resolve_fill_limit`](@ref), as the batch method does, giving `fill_limit`.
 3. Drop the carried rows of a Shared Series gap with [`shared_series_buffer`](@ref), and read the rows that remain with [`sample_buffer`](@ref), giving `X`. Under the horizon method, take `log1p` of the rows, giving `Xl`.
 4. Read `mu` and `sigma` through [`read_member`](@ref), which refits a member that did not fold over `X`, or over `Xl` under the horizon method. The mean comes from [`mean_with_count`](@ref), with the count `T_mu` that its fit states. A member that it refits also takes the panel of [`buffer_asset_panel`](@ref), and the implied volatilities and `ivpa` that [`implied_volatility_buffer_kwargs`](@ref) reads from the carried buffer, as the batch verb gives `rd.pnl`, `rd.iv` and `rd.ivpa` to both members. A covariance that it refits is the [`library_covariance_estimator`](@ref) of `pe.ce`, as in the batch verb. An [`ImpliedVolatility`](@ref) has no fold, so it is such a member.
 5. Under the horizon method, convert `mu` and `sigma` with [`horizon_moments!`](@ref).
 6. Cut the carried rows to `pe.max_scenarios` with [`scenario_window`](@ref), and copy them into a new matrix `Xs`, because the next fold writes into the storage of the buffer.
 7. Fill `Xs` with [`scenario_fill`](@ref), and build a [`LowOrderPrior`](@ref) whose `eoc` [`scenario_moment_counts`](@ref) reads from `T_mu` and the rows before the cut, the log-returns under a horizon.
 8. Record the observations of the buffer that step 3 dropped with [`shared_series_record`](@ref), and return the prior.

# Arguments

  - `pe`: Empirical prior estimator carrying a [`PriorCarryState`](@ref).
  - `strict`: Whether a zero-filled scenario raises rather than warns. It also reaches the covariance, where a peel of a [`CoveragePolicy`](@ref) refuses under `strict` and warns otherwise.
  - `kwargs...`: Additional keyword arguments, forwarded to the batch verb of a member that did not fold.

# Validation

  - `pe.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.
  - Under `strict`, no investable column needs a fill. An `ArgumentError` is thrown otherwise.
  - The rules of [`shared_series_buffer`](@ref).

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`PriorCarryState`](@ref)
  - [`partial_fit!`](@ref)
  - [`scenario_window`](@ref)
  - [`scenario_fill`](@ref)
  - [`horizon_moments!`](@ref)
  - [`buffer_asset_panel`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any, <:Any,
                                  <:Option{<:PriorCarryState}}; strict::Bool = false,
               kwargs...)
    state = partial_fit_cache(pe)
    fill_limit = resolve_fill_limit(pe.fill_limit, coverage_floor(pe))
    # A member that refits on the buffer loses the weights of the dropped observations (#1960).
    pb, buf = buffer_refit_members(pe, state.buf)
    X = sample_buffer(buf)
    pnl = buffer_asset_panel(buf)
    vols = implied_volatility_buffer_kwargs(buf)
    mu, T_mu = read_member(mean_with_count, pb.me, X, pnl; vols..., kwargs...)
    # A raw `StatsBase` estimator has no Coverage Policy, so it peels nothing and takes no
    # `strict`.
    shared = isa(pb.ce, AbstractCovarianceEstimator) ? (; strict = strict) : (;)
    sigma = read_member(Statistics.cov, pb.ce, X, pnl;
                        batch = library_covariance_estimator(pb.ce), shared = shared,
                        vols..., kwargs...)
    # Materialised, not viewed: the buffer's backing matrix is written and reallocated by the
    # next fold, and a Result holding a view of it would change under its holder.
    Xs = Matrix(scenario_window(pe.max_scenarios, X))
    pr = LowOrderPrior(; X = scenario_fill(Xs, mu, sigma, strict, fill_limit, state.named),
                       mu = mu, sigma = sigma,
                       eoc = scenario_moment_counts(pb, X, size(Xs, 1), T_mu; vols...,
                                                    kwargs...), x_src = pe.x_src)
    return shared_series_record(pr, state.buf)
end
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any, <:Any,
                                  <:Option{<:PriorCarryState}}; strict::Bool = false,
               kwargs...)
    state = partial_fit_cache(pe)
    fill_limit = resolve_fill_limit(pe.fill_limit, coverage_floor(pe))
    # A member that refits on the buffer loses the weights of the dropped observations (#1960).
    pb, buf = buffer_refit_members(pe, state.buf)
    X = sample_buffer(buf)
    # The arms were folded on the log-returns, so a member that did not fold is refitted on
    # them too. The carried rows are the arithmetic ones the caller handed in.
    Xl = log1p.(X)
    pnl = buffer_asset_panel(buf)
    vols = implied_volatility_buffer_kwargs(buf)
    mu, T_mu = read_member(mean_with_count, pb.me, Xl, pnl; vols..., kwargs...)
    # A raw `StatsBase` estimator has no Coverage Policy, so it peels nothing and takes no
    # `strict`.
    shared = isa(pb.ce, AbstractCovarianceEstimator) ? (; strict = strict) : (;)
    sigma = read_member(Statistics.cov, pb.ce, Xl, pnl;
                        batch = library_covariance_estimator(pb.ce), shared = shared,
                        vols..., kwargs...)
    horizon_moments!(mu, sigma, pe.horizon)
    Xs = Matrix(scenario_window(pe.max_scenarios, X))
    pr = LowOrderPrior(; X = scenario_fill(Xs, mu, sigma, strict, fill_limit, state.named),
                       mu = mu, sigma = sigma,
                       eoc = scenario_moment_counts(pb, Xl, size(Xs, 1), T_mu; vols...,
                                                    kwargs...), x_src = pe.x_src)
    return shared_series_record(pr, state.buf)
end
"""
    partial_fit!(pe::BlackLittermanPrior, X, F = nothing; kwargs...)

Folds observations into a [`BlackLittermanPrior`](@ref) by forwarding them to its embedded prior.

The estimator keeps no buffer, for the reason that [`HighOrderPriorEstimator`](@ref) keeps none. It returns [`forward_prior`](@ref) of the result of its embedded prior, so the rows it needs are one level down. The factor observations go down with the rows, as the batch verb passes `F` down. The views are configuration, and they fold nothing.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Observations to fold, a matrix of rows or one observation.
  - `F`: The factor observations beside `X`, or `nothing`. The method forwards them to the embedded prior.
  - `kwargs...`: Additional keyword arguments, forwarded to the embedded prior.

# Validation

  - `pe.pe` folds. The embedded prior refuses the fold otherwise, with an error that names the wrapper that gives it a buffer.

# Returns

  - `pe`: The estimator, with its embedded prior folded.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`prior`](@ref)
  - [`bl_posterior`](@ref)
"""
function partial_fit!(pe::BlackLittermanPrior, X::VecNum_MatNum,
                      F::Option{<:VecNum_MatNum} = nothing; kwargs...)
    return rebuild_estimator(pe, (; pe = partial_fit!(pe.pe, X, F; kwargs...)))
end
"""
    prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)

Reads a [`BlackLittermanPrior`](@ref) out of its fold, with no data matrix.

The embedded prior answers first, and [`bl_posterior`](@ref) is the body of the batch method, unchanged. That body reads every number from the result and none from a returns matrix. These numbers are the number of observations behind `tau`, the asset axis, and the matrix that the processing runs over. So the fold and the batch fit run one body.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P}_T &= \\mathcal{P}(\\mathbf{X})\\,.
\\end{align}
```

Where:

  - $(math_dict[:P_fold_prior])
  - $(math_dict[:P_batch_prior])
  - $(math_dict[:X_returns])

# Algorithm

 1. Read the embedded result `prior_model` out of `pe.pe` with [`prior`](@ref), passing `strict`.
 2. Check the asset axis of the views against `prior_model.X` with [`assert_bl_views_axis`](@ref).
 3. Compute the posterior with [`bl_posterior`](@ref), and return it.

# Arguments

  - `pe`: Black-Litterman prior estimator whose embedded prior carries a state.
  - `strict`: Whether an unresolved view raises rather than warns.
  - `kwargs...`: Additional keyword arguments, forwarded to the embedded prior and the processing.

# Validation

  - Every condition that `prior(pe.pe)` with no data checks.
  - The views of a [`LinearConstraintEstimator`](@ref) name one entry per asset of `prior_model.X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `pr::AbstractPriorResult`: The embedded result with its two moments replaced by the posteriors.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_posterior`](@ref)
  - [`assert_bl_views_axis`](@ref)
  - [`partial_fit!`](@ref)
"""
function prior(pe::BlackLittermanPrior; strict::Bool = false, kwargs...)
    prior_model = prior(pe.pe; strict = strict, kwargs...)
    assert_bl_views_axis(pe, prior_model.X)
    return stamp_returns_source(bl_posterior(pe, prior_model, strict; kwargs...), pe)
end
"""
    update_online_estimator(pe::Union{HighOrderPriorEstimator, BlackLittermanPrior})

Resolves the [`Online`](@ref) declarations under the prior that a wrapping prior embeds, at warm-up.

The generic method scans the fields of the wrapping prior alone, and a wrapping prior holds its embedded prior in its own `pe` field, so the wrapping prior defines the recursion itself. `HighOrderPriorEstimator(; pe = BlackLittermanPrior(; pe = Online(EmpiricalPrior()), ...))` resolves through this method. The generic scan of the own fields of the wrapping prior finds no wrapper there, because `pe` holds a plain prior, so the generic method would leave the wrapper two levels down unseeded. A wrapper in the own `pe` of the wrapping prior, as in `HighOrderPriorEstimator(; pe = Online(EmpiricalPrior()))`, is seeded the same way, because the recursion meets it at the first level. The generic method still scans the co-moments of a [`HighOrderPriorEstimator`](@ref).

# Algorithm

 1. Resolve every field that [`online_fields`](@ref) names with [`update_online_estimator`](@ref), giving `repl`.
 2. Resolve `pe.pe` with [`update_online_estimator`](@ref), which recurses through a wrapping prior.
 3. Rebuild the prior from `repl` and the resolved `pe.pe`, the second taking precedence for `pe`, and return it.

# Arguments

  - `pe`: The wrapping prior.

# Returns

  - `pe`: The prior, with every wrapper under its embedded prior resolved.

# Related

  - [`update_online_estimator`](@ref)
  - [`Online`](@ref)
"""
function update_online_estimator(pe::Union{HighOrderPriorEstimator, BlackLittermanPrior})
    fns = online_fields(pe)
    repl = NamedTuple{fns}(map(f -> update_online_estimator(getfield(pe, f)), fns))
    return rebuild_estimator(pe, merge(repl, (; pe = update_online_estimator(pe.pe))))
end
