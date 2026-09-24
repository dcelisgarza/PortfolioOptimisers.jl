# `arg_dict`: the data, the moments and the matrices estimated from them, the
# regime-adjusted and exponentially weighted estimators, and the partial fit states.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(arg_dict, :arg_dict,
                 # Weight vectors.
                 :pw => "`w`: Portfolio weights vector `assets × 1`.",#
                 :ow => "`w`: Observation weights vector `observations × 1`.",#
                 :oow => "`w`: Optional observation weights vector `observations × 1`, or a concrete subtype of [`DynamicAbstractWeights`](@ref). If `nothing`, the computation is unweighted.",#
                 :eqw => "`w`: Optional equilibrium weights vector `assets × 1`. If `nothing`, equal weights are used.",#
                 # Matrix processing.
                 :pdm => "`pdm`: Positive definite matrix estimator.",
                 :opdm => "`pdm`: Optional positive definite matrix estimator.",
                 :dn => "`dn`: Matrix denoising estimator.",
                 :odn => "`dn`: Optional matrix denoising estimator.",
                 :dt => "`dt`: Matrix detoning estimator.",
                 :odt => "`dt`: Optional matrix detoning estimator.",
                 :mp => "`mp`: Matrix processing estimator.",
                 :omp => "`mp`: Optional matrix processing estimator.",
                 :mpa => "`mpa`: Matrix processing algorithm.",
                 # Moments.
                 :me => "`me`: Expected returns estimator.",
                 :ome => "`me`: Optional expected returns estimator. It is not needed when used on a vector. If `nothing` and used on a matrix, defaults to [`SimpleExpectedReturns`](@ref).",
                 :ce => "`ce`: Covariance estimator.",#
                 :ve => "`ve`: Variance estimator.",#
                 :ske => "`ske`: Coskewness estimator.",
                 :kte => "`kte`: Cokurtosis estimator.",
                 :de => "`de`: Distance matrix estimator.",
                 :malg => "`alg`: Moment algorithm.",
                 :pf_n => "`n`: Number of observations folded into the state.",
                 :pf_mu => "`mu`: Running mean of the observations folded into the state, `assets × 1`.",
                 :pf_M2 => "`M2`: Running second central co-moment accumulator, `assets × assets`. It is the sum over the observations and not the covariance, so a read-out divides it by `n`.",
                 :pf_M3 => "`M3`: Running third central co-moment accumulator, `assets × assets²`. It is the sum over the observations and not the coskewness, so a read-out divides it by `n`.",
                 :pfcache => "`cache`: Optional partial-fit state. It is `nothing` until [`partial_fit!`](@ref) writes one, and the estimator's read-out verb reads it when the caller gives no data matrix. Each propagation channel does one thing with it: [`factory`](@ref) carries it unchanged, because a factory call resolves configuration rather than the sample; [`port_opt_view`](@ref) slices it to the selected assets by index copy, so the viewed estimator answers over those assets alone; and [`obs_weights_view`](@ref) drops it, because no slice of a state exists on the observation axis. A family whose state has no exact asset slice drops it on both axes and names the reason.",
                 :cvg => "`cvg`: Optional [`CoveragePolicy`](@ref). `nothing` is the reduce-and-expand path of the Coverage Universe, in which an asset that is non-finite or inactive at any observation of the window is `NaN` throughout the answer. A policy replaces it by available-case estimation: every cell is fitted on the observations at which the assets of that cell are all finite and active, each cell carries its own denominator, and an asset reaches the answer where [`admits`](@ref) says so.",
                 :pf_cvg => "`cvg`: Optional [`CoverageCounts`](@ref), the per-cell denominators and per-asset bookkeeping of an available-case fold. It is `nothing` when the estimator carries no [`CoveragePolicy`](@ref), so the plain state costs nothing.",
                 :corrected => "`corrected`: Whether to apply Bessel's correction.",#
                 :mutgt => "`tgt`: Shrinkage target.",#
                 :me_shrink_alg => "`alg`: Expected returns shrinkage algorithm.",#
                 :me_cval => "`val`: Custom expected returns value.\n\n  - If a scalar, every asset is assigned this value.\n  - If a vector, each element is one asset's value.\n  - If a callable, it is called as `val(X; dims = dims, kwargs...)` and must return one value per asset.",#
                 :metric => "`metric`: Distance metric used for pairwise computations.",#
                 :metric_args => "`args`: Additional positional arguments for the distance metric.",#
                 :metric_kwargs => "`kwargs`: Additional keyword arguments for the distance metric.",#
                 :t => "`t`: Threshold value.",#
                 :oiv => "`iv`: Optional implied volatility matrix. Used if any internal covariance estimator is an instance of [`ImpliedVolatility`](@ref).",#
                 ## Regression
                 :M => "`M`: Main coefficient (loadings) matrix `assets × factors`.",#
                 :L => "`L`: Reduced dimensionality coefficient (loadings) matrix `assets × reduced_dimensions`.",#
                 :b => "`b`: Regression intercept vector.",#
                 :esigma => "`esigma`: Idiosyncratic covariance. A vector holds the variances alone, and a matrix holds the full covariance.",#
                 :rf_mu => "`mu`: The latest Return Forecast, one entry per asset of the coverage universe, in return units, `NaN` for an asset the member forecasts nothing for.",#
                 :rf_hist => "`hist`: Return Forecast history `observations × assets`, in return units, or `nothing` for a member that computes none.",#
                 :rf_scores => "`scores`: The recipe that turns the Descriptors into cross-sectional scores.",#
                 :rf_unit => "`unit`: The Forecast Unit the intermediate forecast is read in, before the member converts it to return units.",#
                 :rf_scale => "`scale`: Multiplicative scale applied to the Return Forecast after it reaches return units. It sets the strength of the forecast without moving the fitted coefficients.",#
                 :rf_horizon => "`horizon`: Number of forward observations the target of the fit averages over.",#
                 :rf_lag => "`lag`: Number of observations between the scored observation and the first return of its target window.",#
                 :rf_whole_history => "`whole_history`: Whether the fit reads the whole carrier, with the block's histories placed into the rows they were fitted on, rather than the block's rows alone.",#
                 :crit => "`crit`: Factor selection criterion. A [`PValue`](@ref), or a `Val` of one symbol of [`STEPWISE_REGRESSION_CRITERIA`](@ref).",#
                 :r2variant => "`variant`: Name of the pseudo-``R^2`` variant a maximisation criterion reads, or `nothing` to take the default of the criterion.",#
                 :realg => "`alg`: Regression algorithm.",#
                 :retgt => "`tgt`: Regression model target.",#
                 :dretgt => "`retgt`: Regression model target.",#
                 :drtgt => "`drtgt`: Dimension reduction target.",#
                 :csrint => "`intercept`: Whether a per-observation intercept is fitted. When `false`, the regression runs through the origin of the cross-section.",
                 ## Gerber
                 :gerbalg => "`alg`: Gerber covariance algorithm.",#
                 :gerbce => "`ce`: Gerber covariance estimator.",#
                 :stdarr => "`sd`: Standard deviation vector of `X`, shaped to be consistent with `X`.",#
                 :c1 => "`c1`: Zone of confusion threshold, in units of the asset's standard deviation. It is read against the raw, uncentred return, and it rejects an observation only when both assets fall inside it.",#
                 :c2 => "`c2`: Zone of indecision threshold, in units of the asset's standard deviation. It is read against the centred, standardised return, and it rejects an observation when both assets fall inside it. A centred return of exactly zero is inside it at every `c2`.",#
                 :c3 => "`c3`: Outer cut-off, in units of the asset's standard deviation. It is read against the centred, standardised return, and it rejects an observation when either asset exceeds it.",#
                 :sbn => "`n`: Severity exponent of the Smyth-Broby contribution. It sets how hard the divergence of a pair is penalised.",#
                 :sbalg => "`alg`: Smyth-Broby covariance algorithm.",#
                 ## Mutual and var info
                 :bins => "`bins`: Binning algorithm or fixed number of bins.",#
                 :normalise => "`normalise`: Whether to normalise the mutual and/or variation of information calculation.",#
                 :xj => "`xj`: Data vector for variable `j`.",#
                 :xi => "`xi`: Data vector for variable `i`.",#
                 :jidx => "`j`: Index of variable `j`.",#
                 :iidx => "`i`: Index of variable `i`.",#
                 :Tobs => "`T`: Number of observations.",#
                 ## Distance
                 :dopower => "`power`: Optional matrix exponent. `nothing` and `1` both give the base distance, so only `power >= 2` changes the result.",#
                 :dalg => "`alg`: Distance algorithm.",#
                 :dmetric => "`metric`: Distance metric used for the distances of distances computations.",#
                 :dmetric_args => "`args`: Additional positional arguments for the distances of distances metric.",#
                 :dmetric_kwargs => "`kwargs`: Additional keyword arguments for the distances of distances metric.",#
                 :fdmetric => "`metric`: Distance metric applied to the rows of the feature matrix.",#
                 :fcalg => "`alg`: Feature collapse algorithm, used to reduce a window of time-varying features to a single distance matrix. Inert for a 2-D feature matrix.",#
                 :calg => "`alg`: Collapse algorithm, the aggregator applied along the observation axis.",#
                 :fdsim => "`sim`: Similarity matrix algorithm used to derive the similarity counterpart of the feature distance matrix.",#
                 :fdape => "`ape`: Asset Panel producer, or `nothing` to read the panel the data carrier holds. A producer is configuration: it builds a static panel at the point of use, from the prior result and the returns of the subproblem that runs it, so a view passes it through and a fold refits it.",#
                 :fdsel => "`sel`: Feature Selector naming the Panel Fields the Feature Matrix stacks, or `nothing` to stack every field's values. An entry is a field name, a field paired with the levels or labels it keeps, a field paired with one level or label, or a field paired with `:observed`. The vector order is the column order.",#
                 :fdstrict => "`strict`: Whether a `sel` entry naming a field, a level or a label the Asset Panel does not hold throws instead of warning and being dropped.",#
                 :fdrows => "`rows`: The observation rows a time-varying Asset Panel stacks, `Colon()` for every row. A static panel has no observation axis and refuses any other value.",#
                 # Stats.
                 :sigma => "`sigma`: Covariance matrix `assets × assets`.",#
                 :mu => "`mu`: Expected returns vector `assets × 1`.",#
                 :rho => "`rho`: Correlation matrix `assets × assets`.",
                 :sigrho => "`sigma`: Covariance-like or correlation-like matrix `assets × assets`.",
                 :sigrhoX => "`X`: Covariance-like or correlation-like matrix `assets × assets`.",
                 :kt => "`kt`: Cokurtosis matrix `assets^2 × assets^2`.",#
                 :sk => "`sk`: Coskewness matrix `assets × assets^2`.",#
                 :V => "`V`: Sum of the negative spectral slices of the coskewness matrix `assets × assets`.",
                 :X => "`X`: Data matrix `observations × assets` if the `dims` keyword does not exist or `dims = 1`, `assets × observations` when `dims = 2`.",#
                 :o_X => "`o_X`: The returns matrix the caller supplied, kept only when the carrier's own `X` is not it, and `nothing` otherwise. The three estimators that lift a factor-axis prior onto the asset axis overwrite `X` with the reconstruction `F * transpose(M) .+ transpose(b)`; `o_X` is the asset returns they were handed, over the same observations and the same assets. Read it as `original_X`, which is always a matrix, rather than as this field.",#
                 :F => "`F`: Data matrix `observations × factors` if the `dims` keyword does not exist or `dims = 1`, `factors × observations` when `dims = 2`.",#
                 :Xv => "`X`: Data vector `observations × 1`.",#
                 :X_sub => "`X`: Returns matrix of the subproblem, `observations × assets`.",#
                 :X_Xv => "`X`: Data matrix or vector.",#
                 :Z => "`Z`: Feature matrix `assets × features` if `dims = 1`, `features × assets` when `dims = 2`. May also be a 3-D array of time-varying features, in which case the observation axis always leads: `observations × assets × features` if `dims = 1`, `observations × features × assets` when `dims = 2`.",#
                 :plfe => "`pl`: Structure source, always an estimator so that it refits per fold: a network estimator (a graph, whose `sep` measures the separations `alg` grades) or a clustering estimator (a partition, for which `alg` is inert). A precomputed result is not accepted -- an Estimator does not hold a Result.",#
                 :plfalg => "`alg`: Phylogeny feature algorithm: the rule turning the source's separations into feature values. Inert for a partition source, which has no separation to grade.",#
                 :dims => "`dims`: Dimension along which to perform the computation.",#
                 :omean => "`mean`: Optional mean value to use for centering.",
                 :ex => "`ex`: Parallel execution strategy.",#
                 :alpha => "`alpha`: Quantile level for the lower tail. The bound is [`Num_SigCal`](@ref), so the slot takes the level itself, an [`AbstractSignificanceCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :alpha_ltd => "`alpha`: Quantile level for the lower tail.",#
                 :beta => "`beta`: Quantile level for the upper tail. The bound is [`Num_SigCal`](@ref), so the slot takes the level itself, an [`AbstractSignificanceCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 # Risk-free rate.
                 :rf => "`rf`: Risk-free rate.",#
                 # Data carrier fields.
                 :ivpa_iv => "`ivpa`: Implied volatility risk premium adjustment, if a vector (assets × 1).",#
                 # Cross-sectional transforms.
                 :min_group_size => "`min_group_size`: Smallest estimation set a group may carry and still be estimated from. A group below it, and every asset that carries no group, takes the whole observation's statistics instead.",#
                 :atol_cs => "`atol`: Absolute tolerance below which a cross-sectional scale counts as zero. An observation at or below it carries no dispersion, so its finite cells score zero rather than dividing by that scale.",#
                 # Descriptors.
                 :mcap => "`mcap`: Name of the Panel Field that weights the market return, the market capitalisation of each asset.",#
                 :agg_obs => "`agg_obs`: Number of consecutive observations aggregated into one update of the recursion. A value of one updates the recursion at every observation.",#
                 # Regime adjusted estimators.
                 :decay => "`decay`: Exponential decay factor for the exponentially weighted estimator.",#
                 :min_obs => "`min_obs`: Minimum number of observations required before the estimator produces a valid result.",#
                 :hac_lags => "`hac_lags`: Optional number of lags for Heteroskedasticity and Autocorrelation Consistent (HAC) kernel correction of squared returns. If `nothing`, no HAC correction is applied.",#
                 :regime_method => "`regime_method`: Regime adjustment method used to compute the per-observation regime state, or `nothing` to apply no regime adjustment.",#
                 :regime_decay => "`regime_decay`: Exponential decay factor for smoothing the regime state.",#
                 :regime_min_obs => "`regime_min_obs`: Minimum number of regime observations required before the regime multiplier is applied.",#
                 :regime_lohi_mult => "`regime_lohi_mult`: Optional `(lo, hi)` tuple bounding the regime multiplier range. If `nothing`, no clamping is applied.",#
                 :min_val => "`min_val`: Minimum threshold to prevent division by zero or degenerate estimates.",#
                 :centred => "`centred`: Whether to treat the returns as pre-centred (mean zero). If `false`, the location is estimated online.",#
                 :ra_x => "`x`: Shape parameter of the log regime adjustment.",#
                 :ra_y => "`y`: Scale parameter of the log regime adjustment.",#
                 :ra_kappa => "`kappa`: Precomputed normalisation constant `digamma(x) + log(y)` for the log regime adjustment.",#
                 :ra_norm_x => "`x`: First-moment normalisation constant for the regime adjustment.",#
                 :ret_buffer => "`ret_buffer`: Optional circular buffer of recent centred returns for HAC kernel correction.",#
                 :ra_variance => "`variance`: Running per-asset variance vector.",#
                 :ra_X2 => "`X2`: Working array for current (possibly HAC-adjusted) squared returns.",#
                 :ra_X_old_i => "`X_old_i`: Working array for lagged centred returns.",#
                 :ra_z2 => "`z2`: Standardised squared innovations used for regime state computation.",#
                 :ra_location => "`location`: Exponentially smoothed location (mean) vector.",#
                 :obs_count => "`obs_count`: Per-asset count of observations processed.",#
                 :old_obs_count => "`old_obs_count`: Per-asset observation count from the previous step.",#
                 :ra_active => "`active`: Boolean mask indicating which assets are currently active.",#
                 :regime_state => "`regime_state`: Current smoothed regime state value.",#
                 :n_regime_obs => "`n_regime_obs`: Number of observations used to update the regime state.",#
                 :cor_decay => "`cor_decay`: Exponential decay factor for the correlation smoother.",#
                 :regime_target => "`regime_target`: Target structure for the regime-adjusted covariance update.",#
                 :ra_w => "`w`: Optional portfolio weights for the portfolio target, as one vector over the assets or as a matrix whose rows are portfolios. If `nothing`, inverse-volatility weights are used, and they are rebuilt from the running variance at each observation.",#
                 :ra_covariance => "`covariance`: Running exponentially weighted covariance matrix, seeded at zero.",#
                 :ra_cor_state => "`cor_state`: Running exponentially weighted correlation state, or `nothing` where one decay governs both the variance and the correlation.",#
                 :ra_pair_obs_count => "`pair_obs_count`: Pairwise count of co-observations, or `nothing` where one decay governs both the variance and the correlation.",#
                 :ra_XXt => "`XXt`: Working matrix for the current (possibly HAC-adjusted) outer product of the returns.",#
                 :ra_Xi => "`Xi`: Working array for the current centred returns.",#
                 # Plain exponentially weighted estimators.
                 :ew_cache => "`cache`: Running state of an incremental fit, or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. A fit over a matrix ignores it.",#
                 :ew_mu => "`mu`: Running exponentially weighted mean vector, seeded at zero.",#
                 :sq => "`sq`: Whether to use variance instead of volatility in the inverse weighting.",#
                 :wfalg => "`alg`: Weight finaliser error formulation algorithm.",#
                 :res_retcode => "`res`: Optional result or message from the solver.",#
                 :N_msc => "`N`: Number of evenly spaced values of `gamma` the monotonic Schur complement scans for its turning point, both ends included.",#
                 :alpha_dirichlet => "`alpha`: Dirichlet concentration parameter.",#
                 :opt_hier => "`opt`: Base hierarchical optimiser configuration.",#
                 :strict_opt => "`strict`: Whether to strictly enforce weight bounds.",#
                 :strict_conv => "`strict`: Whether to raise an error if convergence is not achieved.",#
                 :schalg => "`alg`: Schur complement algorithm variant.",#
                 # Partial fit states.
                 :pf_M => "`M`: Running second-moment accumulator of the observations folded into the state, about `mu`.",
                 :pf_max_history => "`max_history`: Optional cap on the number of observations the buffer keeps. `nothing` keeps every observation folded so far. A capped buffer drops its oldest observations as new ones arrive and holds the last `max_history` of them. A read-out over the buffer reads those rows only. The cap is the window of the fit: an estimator wrapped in [`Online`](@ref) returns the batch fit over the last `max_history` observations, also when its statistic has an exact update.",#
                 # The masks of a returns buffer.
                 :pf_buffer_A => "`A`: Backing matrix of the active mask, of the shape of `X`, or `nothing` when the buffer records no activity. Rows `off + 1` to `off + n` are the mask of the observations, cell for cell with them. It is fixed by the first append, and it is what lets a read-out tell a delisting from a holiday.",#
                 :pf_buffer_E => "`E`: Backing matrix of the estimation mask, of the shape of `X`, or `nothing` when the buffer records none. It is the second per-observation mask the batch verbs take, it is carried on the same terms as `A`, and only the two regime-adjusted families read it.",#
                 :pf_active_mask => "`active_mask`: The active mask of the block, of the shape of `X`, or `nothing`. A buffer records it for every observation it holds or for none of them.",#
                 :pf_estimation_mask => "`estimation_mask`: The estimation mask of the block, of the shape of `X`, or `nothing`. It is carried on the same terms as `active_mask`.")
