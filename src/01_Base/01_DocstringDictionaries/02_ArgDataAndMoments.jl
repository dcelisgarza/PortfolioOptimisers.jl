# `arg_dict`: the data, the moments and the matrices estimated from them, the
# regime-adjusted and exponentially weighted estimators, and the partial fit states.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(arg_dict, :arg_dict,
                 # Weight vectors.
                 :pw => "`w`: Portfolio weights vector `assets × 1`.",#
                 :ow => "`w`: Observation weights vector `observations × 1`.",#
                 :oow => "`w`: Optional observation weights vector `observations × 1`, or a concrete subtype of [`DynamicAbstractWeights`](@ref). If `nothing`, the computation is unweighted.",#
                 :eqw => "`w`: Optional equilibrium weights vector `assets × 1`. If `nothing`, [`equilibrium_mu`](@ref) uses equal weights.",#
                 :w_pin => "`w`: The observation weights under which the batch fit makes the choice that the method pins, or `nothing` when that fit is unweighted.",#
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
                 :ome => "`me`: Optional expected returns estimator. A vector method does not use it. If it is `nothing`, a matrix method uses a [`SimpleExpectedReturns`](@ref).",
                 :ce => "`ce`: Covariance estimator.",#
                 :ve => "`ve`: Variance estimator.",#
                 :ske => "`ske`: Coskewness estimator.",
                 :kte => "`kte`: Cokurtosis estimator.",
                 :de => "`de`: Distance matrix estimator.",
                 :malg => "`alg`: Moment algorithm.",
                 :pf_n => "`n`: Number of observations folded into the state.",
                 :pf_mu => "`mu`: Running mean of the observations folded into the state, `assets × 1`.",
                 :pf_M2 => "`M2`: Running second central co-moment accumulator, `assets × assets`. It is the sum over the observations and not the covariance, so the call with no data divides it by `n`.",
                 :pf_M3 => "`M3`: Running third central co-moment accumulator, `assets × assets²`. It is the sum over the observations and not the coskewness, so the call with no data divides it by `n`.",
                 :pfcache => "`cache`: Optional partial-fit state. It is `nothing` until [`partial_fit!`](@ref) writes one. When the caller gives no data matrix, the batch verb of the estimator reads it. Each propagation channel does one thing with it. [`factory`](@ref) keeps it unchanged, because a factory call resolves the configuration, not the sample. [`port_opt_view`](@ref) copies the entries of the selected assets into a new state, so the viewed estimator holds the state of those assets alone. [`obs_weights_view`](@ref) removes it, because a state has no slice on the observation axis. Only a type with an observation-weights field has that channel. A type with no such field falls through to the identity method, so it keeps its state. A family whose state has no exact asset slice removes it on both axes and names the reason.",
                 :cvg => "`cvg`: Optional [`CoveragePolicy`](@ref). `nothing` selects the reduce-and-expand path of the Coverage Universe. On that path, an asset that is not finite or not active at an observation of the window is `NaN` in all of the result. A policy replaces that path with available-case estimation. The fit of each cell reads the observations at which all the assets of that cell are finite and active, and each cell has its own denominator. An asset is in the result where [`admits`](@ref) admits it.",
                 :pf_cvg => "`cvg`: Optional [`CoverageCounts`](@ref), the per-cell denominators and the per-asset records of an available-case fold. It is `nothing` when the estimator holds no [`CoveragePolicy`](@ref), so the plain state costs nothing.",
                 :corrected => "`corrected`: If `true`, the estimator applies the Bessel correction. `Statistics` or `StatsBase` applies it, with the divisor that each weight type defines.",#
                 :mutgt => "`tgt`: Shrinkage target.",#
                 :mu_intensity => "`intensity`: The rule that sets the interval of the shrinkage intensity. It is a member of [`AbstractShrunkExpectedReturnsIntensity`](@ref).",#
                 :me_shrink_alg => "`alg`: The shrinkage algorithm. It holds the shrinkage target.",#
                 :me_cval => "`val`: Custom expected returns value.\n\n  - If a scalar, every asset gets this value.\n  - If a vector, each element is the value of one asset.\n  - If a callable, the estimator calls it as `val(X; dims = dims, kwargs...)`, and it must return one value per asset.",#
                 :metric => "`metric`: The metric that `Distances.pairwise` applies to the observations of each series.",#
                 :metric_args => "`args`: More positional arguments, which `Distances.pairwise` takes after the series.",#
                 :metric_kwargs => "`kwargs`: More keyword arguments for `Distances.pairwise`.",#
                 :t => "`t`: Threshold value.",#
                 :oiv => "`iv`: Optional implied volatility matrix. The estimator reads it only when one of its inner covariance estimators is an [`ImpliedVolatility`](@ref).",#
                 ## Regression
                 :M => "`M`: Loadings matrix `assets × factors`.",#
                 :L => "`L`: Loadings matrix `assets × reduced_dimensions` in the reduced basis of a dimension reduction.",#
                 :b => "`b`: Regression intercept vector.",#
                 :esigma => "`esigma`: Idiosyncratic covariance. A vector holds the variances alone, and a matrix holds the full covariance.",#
                 :edof => "`edof`: Degrees of freedom of each residual series, one entry per asset. It is the effective count of the observations that measure the variance of the asset, less the parameters that the fit spent on the asset. A regression estimator writes it over equally weighted observations, and a prior that writes `esigma` restates it in the count of its variance estimator.",#
                 :ediv => "`ediv`: Divisor of each idiosyncratic variance, one entry per asset, in the effective count of `edof`. The variance is the weighted residual sum of squares in that count, divided by this entry. A prior that writes `esigma` writes it.",#
                 :rf_mu => "`mu`: The latest Return Forecast, one entry per asset of the coverage universe, in return units, `NaN` for an asset the member forecasts nothing for.",#
                 :rf_hist => "`hist`: Return Forecast history `observations × assets`, in return units, or `nothing` for a member that computes none.",#
                 :rf_scores => "`scores`: The recipe that turns the Descriptors into cross-sectional scores.",#
                 :rf_unit => "`unit`: The Forecast Unit the intermediate forecast is read in, before the member converts it to return units.",#
                 :rf_scale => "`scale`: Multiplicative scale applied to the Return Forecast after it reaches return units. It sets the strength of the forecast without moving the fitted coefficients.",#
                 :rf_horizon => "`horizon`: Number of forward observations the target of the fit averages over.",#
                 :rf_lag => "`lag`: Number of observations between the scored observation and the first return of its target window.",#
                 :fe_table_ppy => "`ppy`: Periods per year of the annualised columns of the table. It defaults to the `ppy` of the evaluation, so the table and the summary of one evaluation share their units. `1` gives the statistics per period.",#
                 :cs_ties => "`ties`: Rule that ranks the equal values of a cross-section. `:average` gives each value of a tie the mean of the positions the tie takes, so a rank correlation does not read the order of the assets, and a constant cross-section has no rank correlation. `:ordinal` gives the values of a tie consecutive positions, in the order of the asset axis. The two rules give the same ranks to a cross-section with no tie.",#
                 :cs_ic_rd => "`rd`: The returns data that the prior read, or `nothing` to reconstruct the asset returns from the fit of the block.",#
                 :idio_ahead => "`ahead`: Whether each idiosyncratic return is divided by the volatility estimated before it. `true` divides the return of observation ``t`` by the volatility of observation ``t - 1``, whose estimate has not read that return, so a variance forecast that is right gives standardised returns of unit variance, and the first observation has no standardised return. `false` divides the return by the volatility of its own observation, whose estimate has read it, which pulls every standardised return toward zero and caps its size.",#
                 :fd_nf => "`nf`: Name of each factor on the factor axis of the answer, or `nothing` when the block names no factor. The axis is the one that the verb answers on, raw or reduced.",#
                 :fd_fam => "`fam`: Family label of each factor on the factor axis of the answer, or `nothing` when the block labels no family. A view selects the factors of one family with a [`LabelGroup`](@ref).",#
                 :rf_whole_history => "`whole_history`: Whether the fit reads every observation of the returns data, with the block's histories placed into the rows they were fitted on, rather than the block's rows alone.",#
                 :crit => "`crit`: Factor selection criterion. A [`PValue`](@ref), or a `Val` of one symbol of [`STEPWISE_REGRESSION_CRITERIA`](@ref).",#
                 :r2variant => "`variant`: Name of the pseudo-``R^2`` variant a maximisation criterion reads, or `nothing` to take the default of the criterion.",#
                 :realg => "`alg`: Regression algorithm.",#
                 :retgt => "`tgt`: Regression model target.",#
                 :dretgt => "`retgt`: Regression model target.",#
                 :drtgt => "`drtgt`: Dimension reduction target.",#
                 :csrint => "`intercept`: Whether a per-observation intercept is fitted. When `false`, the regression runs through the origin of the cross-section.",
                 :csalg => "`alg`: Solve algorithm, an [`AbstractCrossSectionalSolveAlgorithm`](@ref). It decides what the fit does with a rank-deficient weighted design.",#
                 ## Gerber
                 :gerbalg => "`alg`: Gerber covariance algorithm.",#
                 :gerbce => "`ce`: Gerber covariance estimator.",#
                 :stdarr => "`sd`: Standard deviation vector of `X`, shaped to be consistent with `X`.",#
                 :c1 => "`c1`: Confusion-zone threshold on the raw, uncentred return, in standard deviations of the asset. The confusion zone rejects an observation only when both assets are inside it.",#
                 :c2 => "`c2`: Indecision-zone threshold on the centred, standardised return, in standard deviations of the asset. The indecision zone rejects an observation when both assets are inside it. A centred return of exactly zero is inside the zone at every `c2`.",#
                 :c3 => "`c3`: Outer cut-off on the centred, standardised return, in standard deviations of the asset. The cut-off rejects an observation when either asset is above it.",#
                 :sbn => "`n`: Severity exponent of the Smyth-Broby contribution. The contribution of an observation is `kappa / (1 + gamma^n)`, where `gamma` is the divergence of the pair.",#
                 :sbalg => "`alg`: Smyth-Broby covariance algorithm.",#
                 ## Mutual and var info
                 :bins => "`bins`: Binning algorithm or fixed number of bins.",#
                 :normalise => "`normalise`: If `true`, the result is the normalised mutual information or the normalised variation of information.",#
                 :xj => "`xj`: Data vector for variable `j`.",#
                 :xi => "`xi`: Data vector for variable `i`.",#
                 :jidx => "`j`: Index of variable `j`.",#
                 :iidx => "`i`: Index of variable `i`.",#
                 :Tobs => "`T`: Number of observations.",#
                 ## Distance
                 :dopower => "`power`: Optional integer exponent, which the distance algorithm applies entry by entry. [`Distance`](@ref) states which quantity it raises. `nothing` and `1` both give the base distance, so only `power >= 2` changes the result.",#
                 :dalg => "`alg`: The distance algorithm.",#
                 :dmetric => "`metric`: The metric that `Distances.pairwise` applies to the columns of the base distance matrix.",#
                 :dmetric_args => "`args`: More positional arguments, which `Distances.pairwise` takes after the base distance matrix.",#
                 :dmetric_kwargs => "`kwargs`: More keyword arguments for `Distances.pairwise`. A `dims` here replaces the default `dims = 2`.",#
                 :fdmetric => "`metric`: Distance metric applied to the rows of the feature matrix.",#
                 :fcalg => "`alg`: Feature collapse algorithm, which reduces a window of time-varying features to one distance matrix. It has no effect on a 2-D feature matrix.",#
                 :calg => "`alg`: Collapse algorithm, the aggregator applied along the observation axis.",#
                 :loalg => "`alg`: Rule that names the row of the window that the collapse reads for each asset.",#
                 :fdpair => "`pair`: Rule for a pair of assets with no shared active row in the window.",#
                 :fdsim => "`sim`: Similarity matrix algorithm that computes the similarity matrix from the feature distance matrix.",#
                 :fdape => "`ape`: Asset Panel producer, or `nothing` to read the panel that the returns data holds. A producer is configuration. It builds a static panel at the point of use, from the prior result and the returns of the subproblem that runs it. So a view passes it through, and a fold refits it.",#
                 :fdsel => "`sel`: Feature Selector that names the Panel Fields that the Feature Matrix stacks, or `nothing` to stack the values of every field. An entry is a field name, a field paired with the levels or labels that it keeps, a field paired with one level or label, a tensor field paired with a [`LabelGroup`](@ref), or a field paired with `:observed`. The order of the vector is the order of the columns.",#
                 :fdstrict => "`strict`: Whether a `sel` entry throws when it names a field, a level, a label or a group that the Asset Panel does not hold. When `strict` is `false`, the entry warns, and the estimator drops it.",#
                 :fdrows => "`rows`: The observation rows a time-varying Asset Panel stacks, as positions or as a `Bool` mask, and `Colon()` for every row. A static panel has no observation axis and refuses any other value.",#
                 :fdplc => "`placeholder`: The value that the Feature Matrix holds at a value column's cell that holds the placeholder of a blank that no fill reached, or `nothing` to keep the stored placeholder. A cell that a fill policy wrote holds data, and the Feature Matrix keeps its value. The default `NaN` shows where no data exists, so no reader takes the placeholder as data.",#
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
                 :o_X => "`o_X`: The returns matrix of the caller when the `X` of the prior result is a different matrix, and `nothing` otherwise. The estimators that lift a factor-axis prior onto the asset axis overwrite `X`, with the reconstruction `F * transpose(M) .+ transpose(b)` or with scenarios of the factor model. `o_X` holds the asset returns that they received, over the same observations and the same assets. Read it as `original_X`, which is always a matrix, and not as this field.",#
                 :F => "`F`: Data matrix `observations × factors` if the `dims` keyword does not exist or `dims = 1`, `factors × observations` when `dims = 2`.",#
                 :Xv => "`X`: Data vector `observations × 1`.",#
                 :X_sub => "`X`: Returns matrix of the subproblem, `observations × assets`.",#
                 :X_Xv => "`X`: Data matrix or vector.",#
                 :Z => "`Z`: Feature matrix, `assets × features` when `dims = 1` and `features × assets` when `dims = 2`. It can also be a 3-D array of time-varying features, whose first axis is always the observation axis: `observations × assets × features` when `dims = 1`, and `observations × features × assets` when `dims = 2`.",#
                 :plfe => "`pl`: Structure source, always an estimator so that it refits per fold: a network estimator (a graph, whose `sep` measures the separations `alg` grades) or a clustering estimator (a partition, for which `alg` is inert). A precomputed result is not accepted -- an Estimator does not hold a Result.",#
                 :plfalg => "`alg`: Phylogeny feature algorithm: the rule turning the source's separations into feature values. Inert for a partition source, which has no separation to grade.",#
                 :dims => "`dims`: Dimension along which the computation runs.",#
                 :dims_rd => "`dims`: Accepted and ignored. A `ReturnsResult` holds its observations along the rows, so the call passes `dims = 1`. The method takes no other keyword that it does not list: an unknown keyword raises the `MethodError` of the language, and no estimator swallows it.",#
                 :dims_chk => "`dims`: Dimension selector to check.",#
                 :sym_msg => "`sym`: Symbolic name used in the error message.",#
                 :omean => "`mean`: Optional precomputed centre of the data.",
                 :ex => "`ex`: Parallel execution strategy.",#
                 :alpha => "`alpha`: Quantile level for the lower tail. The bound is [`Num_SigCal`](@ref), so the slot takes the level itself, an [`AbstractSignificanceCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :alpha_ltd => "`alpha`: The quantile level of the lower tail.",#
                 :beta => "`beta`: Quantile level for the upper tail. The bound is [`Num_SigCal`](@ref), so the slot takes the level itself, an [`AbstractSignificanceCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 # Risk-free rate.
                 :rf => "`rf`: Risk-free rate.",#
                 # Fields of the returns data and the price data.
                 :ivpa_iv => "`ivpa`: Implied volatility risk premium adjustment, a positive number that every asset shares, or a vector of positive numbers with one entry per asset.",#
                 # Cross-sectional transforms.
                 :min_group_size => "`min_group_size`: Smallest estimation set a group may carry and still be estimated from. A group below it, and every asset that carries no group, takes the whole observation's statistics instead.",#
                 :atol_cs => "`atol`: Absolute tolerance below which a cross-sectional scale counts as zero. An observation at or below it carries no dispersion, so its finite cells score zero rather than dividing by that scale.",#
                 # Descriptors.
                 :mcap => "`mcap`: Name of the Panel Field that weights the market return, the market capitalisation of each asset.",#
                 :nonneg_pnl => "`nonneg`: Names of Panel Fields that must be non-negative wherever they are observed and active, or `nothing` for no check. A negative value raises a `DomainError`, because a volume, a share count, a price, a short interest or a dividend below zero is a data error and not a signal.",#
                 :gt0_pnl => "`gt0`: Names of Panel Fields that must be strictly positive wherever they are observed and active, or `nothing` for no check. A value at or below zero raises a `DomainError`, because a price, a market capitalisation, a share count or a total of assets at or below zero is a data error and not a state of the firm. A denominator that valid data can make zero or negative carries no such guard, and its ratio is `NaN`.",#
                 :agg_obs => "`agg_obs`: Number of consecutive observations aggregated into one update of the recursion. A value of one updates the recursion at every observation.",#
                 :group_ewb => "`group`: Name of the categorical Panel Field that the shrinkage groups the cross-section by, or `nothing` to leave every beta raw.",#
                 :min_group_size_ewb => "`min_group_size`: Smallest group, counted over the estimation set, that is shrunk toward its own mean. A smaller group is shrunk toward the mean of the whole estimation set, and an asset that carries no group keeps its raw beta.",#
                 :bounds_ewb => "`bounds`: `(lo, hi)` bounds on the weight that a shrunk beta keeps on its raw value. The bounds `(0, 1)` let the data set the weight alone.",#
                 :window_roll => "`window`: Number of observations in the window.",#
                 :skip_roll => "`skip`: Number of the most recent observations that the window excludes, so the window ends at observation `t - skip`.",#
                 :sign_roll => "`sign`: Multiplier of the window sum, `1` or `-1`. A sign of `1` reads the window as momentum, and a sign of `-1` reads it as reversal.",#
                 :roll_cache => "`cache`: The carried state of the cumulative sums, or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. The batch call [`descriptor`](@ref) ignores it. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads the Descriptor of each new observation off it, with [`descriptor_step`](@ref).",#
                 :ew_desc_cache => "`cache`: The carried state of the exponentially weighted recursion, an [`EWMeanState`](@ref), or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. The batch call [`descriptor`](@ref) ignores it. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads the Descriptor of each new observation off it, with [`descriptor_step`](@ref).",#
                 :ew_vol_desc_cache => "`cache`: The carried state of the recursions of the volatility, an [`EWVolatilityState`](@ref), or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. The batch call [`descriptor`](@ref) ignores it. Only a `ce` that is a [`RegimeAdjustedExpWeightedVariance`](@ref) folds, so the state exists for that `ce` alone. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads the Descriptor of each new observation off it, with [`descriptor_step`](@ref).",#
                 :lag_desc_cache => "`cache`: The carried state of the lagged quantity of the last `lag` observations, a [`LagDescriptorState`](@ref), or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. The batch call [`descriptor`](@ref) ignores it. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads the Descriptor of each new observation off it, with [`descriptor_step`](@ref).",#
                 :ew_beta_desc_cache => "`cache`: The carried state of the recursion, an [`EWBlockState`](@ref) for an [`EWBeta`](@ref) or an [`EWMacroSensitivity`](@ref) and an [`EWDownsideBetaState`](@ref) for an [`EWDownsideBeta`](@ref), or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. The batch call [`descriptor`](@ref) ignores it. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads the Descriptor of each new observation off it, with [`descriptor_step`](@ref).",#
                 :exponentiate_roll => "`exponentiate`: Whether the Descriptor is the simple return `exp(sign * S) - 1` rather than the log return `sign * S`, where `S` is the sum of the log returns of the window.",#
                 # Regime adjusted estimators.
                 :decay => "`decay`: Exponential decay factor for the exponentially weighted estimator.",#
                 :min_obs => "`min_obs`: Minimum number of observations required before the estimator produces a valid result.",#
                 :hac_lags => "`hac_lags`: Optional number of lags for Heteroskedasticity and Autocorrelation Consistent (HAC) kernel correction of squared returns. If `nothing`, no HAC correction is applied.",#
                 :hac_floor => "`hac_floor`: Whether each HAC-adjusted square is floored at zero before it enters the variance recursion. On returns with no autocorrelation the floor makes the variance too large, by 16 % at two lags, so [`NoHacFloor`](@ref), the default, keeps the recursion an unbiased quadratic form and floors only the variance that the estimator returns. [`PerTermHacFloor`](@ref) floors each square.",#
                 :hac_vol_before => "`hac_vol_before`: Whether, on the separate correlation path under HAC, each row of the correlation recursion divides by the volatility before the variance step. The diagonal of a HAC row can be negative, and a divisor that holds the row damps a positive row and amplifies a negative one, so the diagonal of the correlation state is skewed down and can fall to zero. [`VolatilityBeforeUpdate`](@ref), the default, divides by the volatility before the step: at two lags the rule after the step has 9 % to 19 % more mean squared error in the correlation on most returns, and at eight lags up to 3 % of its rows fall on a guard, where the rule before the step puts none. [`VolatilityAfterUpdate`](@ref) divides by the volatility after the step, as the plain path does. It has no effect where `hac_lags` or `cor_decay` is `nothing`.",#
                 :cor_vol => "`cor_vol`: The variance that standardises each product of the separate correlation path. The variance state is seeded at zero, so in the warm-up the raw state gives the first products of the correlation more weight than the bias tables of the regime statistic assume. [`BiasCorrectedVolatility`](@ref), the default, divides the state by its weight first. [`RawVolatility`](@ref) reads the raw state, the rule of the oracle. The steady state is the same under both. It has no effect where `cor_decay` is `nothing`.",#
                 :regime_method => "`regime_method`: Regime adjustment method that computes the regime statistic of each observation, or `nothing` for no regime adjustment.",#
                 :regime_decay => "`regime_decay`: Exponential decay factor of the regime state. A value above `2^(-1/138) ≈ 0.99499`, which is a regime half-life above 138 observations, makes the regime state slow to follow the variance that it corrects. After the variance changes level, the multiplier keeps correcting a mismatch that the variance has already absorbed, so it can overshoot and act erratically. The estimator gives no warning, and it computes the same recursion at every value in `(0, 1)`.",#
                 :regime_min_obs => "`regime_min_obs`: Count of regime observations that the estimator needs before it applies the regime multiplier. Before that count the multiplier is one.",#
                 :regime_lohi_mult => "`regime_lohi_mult`: Optional `(lo, hi)` bounds of the regime multiplier. If `nothing`, no clamp runs.",#
                 :min_val => "`min_val`: Minimum threshold to prevent division by zero or degenerate estimates.",#
                 :centring => "`centring`: The location each deviation is taken from. [`EstimatedCentring`](@ref), the default, estimates the location of each asset and divides each term by the bias that the estimate puts in it, so the moment is unbiased from the second valid return whatever the mean. [`PreCentred`](@ref) takes the returns as deviations from a mean of zero. [`ZeroStartCentring`](@ref) takes the location of a recursion from zero that is not divided by its weight.",#
                 :ra_debias => "`debias`: Whether the regime statistic corrects the bias of the estimate it reads. The inverse of an estimated variance is too large on average (Jensen's inequality). Under [`ExactDebias`](@ref), the default, the statistic skips an estimate of `n + 3` terms or fewer, for `n` directions. Where `hac_lags` is set, it also skips an estimate whose effective count of terms is `n + 1` or fewer. It divides the rest by the bias factor that its `regime_method` reads. The diagonal target also divides its sum by the factor of its law, which reads the correlation of the assets and the noise of each estimate, for the first-moment and the log methods. [`LawDebias`](@ref) divides by the law of the estimate alone, and takes the deviation as independent of the estimate. [`RawStatistic`](@ref) scores the raw statistic.",#
                 :ra_lag_records => "`lag_records`: Circular buffer of the records of the recent observations, the counts, the valid assets and, for a covariance, the overlap of the locations before each one, which the exact factor of a HAC product reads. It is `nothing` under [`PreCentred`](@ref) or without HAC.",#
                 :ra_term_method => "`method`: Regime adjustment method whose moments of one term the table holds.",#
                 :ra_x => "`x`: Shape parameter of the log regime adjustment.",#
                 :ra_y => "`y`: Scale parameter of the log regime adjustment.",#
                 :ra_kappa => "`kappa`: Precomputed normalisation constant `digamma(x) + log(y)` for the log regime adjustment.",#
                 :ra_norm_x => "`x`: First-moment normalisation constant for the regime adjustment.",#
                 :ret_buffer => "`ret_buffer`: Optional circular buffer of recent centred returns for HAC kernel correction.",#
                 :ra_variance => "`variance`: Running per-asset variance vector.",#
                 :ra_X2 => "`X2`: Working array for the terms of the current observation, with the HAC products where `hac_lags` is set.",#
                 :ra_X_old_i => "`X_old_i`: Working array for lagged centred returns.",#
                 :ra_z2 => "`z2`: Standardised squared returns of the current observation, which the regime statistic reads.",#
                 :ra_location => "`location`: Exponentially smoothed location (mean) vector.",#
                 :obs_count => "`obs_count`: Per-asset count of observations processed.",#
                 :old_obs_count => "`old_obs_count`: Per-asset observation count from the previous step.",#
                 :ra_active => "`active`: Boolean mask of the assets that are active now.",#
                 :regime_state => "`regime_state`: Current smoothed regime state, or `nothing` before the first scored observation.",#
                 :n_regime_obs => "`n_regime_obs`: Count of the observations that advanced the regime state.",#
                 :ra_bias => "`bias`: Table of the bias factors of the regime statistic, one per count of observations, which grows as the counts grow. It is `nothing` where the estimator has `debias = RawStatistic()` or has no regime method, and a target that reads no such table leaves it empty.",#
                 :ra_cov_bias => "`bias`: Store of the bias factors of the regime statistic. A portfolio target keeps a table with one factor per count of observations, and the diagonal target a table with the moments of one term per count in `moments`, and each table grows as the counts grow. The Mahalanobis target keeps the interpolation nodes of its factor over the count of observations, one set for each count of assets that it meets, in `nodes`, and in `kappa` and `spread` the tables of the factor of the variance noise on the separate correlation path, one entry per count. The diagonal and the Mahalanobis targets keep in `kernel` the HAC kernel of the correlation, which only the separate path under HAC fills. It is `nothing` where the estimator has `debias = RawStatistic()` or has no regime method, and a target that reads no store leaves it empty.",#
                 :cor_decay => "`cor_decay`: Exponential decay factor for the correlation smoother.",#
                 :regime_target => "`regime_target`: Regime-adjustment target, which names the regime statistic of one observation and so the part of the covariance that the multiplier calibrates.",#
                 :ra_w => "`w`: Optional portfolio weights for the portfolio target, as one vector over the assets or as a matrix whose rows are portfolios. If `nothing`, inverse-volatility weights are used, and they are rebuilt from the running variance at each observation.",#
                 :ra_recursion => "`recursion`: The rule that computes the bias factor of the Mahalanobis statistic from the level recursion over its directions. [`MeanLevelRecursion`](@ref), the default, takes the mean log-determinant of each level inside the exponential, which reads 0.2 % to 1.3 % low near the gate. [`GaussianLevelRecursion`](@ref) carries the covariance of the log-determinant of each level too, which reads within the noise of a Monte Carlo of the pre-centred law. Its nodes cost 2 s to 5.4 s for each count of assets that a fit meets, where the default costs 0.3 s to 0.9 s, and it may be removed if its gain on the forecasts of the estimator does not pay that cost.",#
                 :ra_indefinite => "`indefinite`: The rule for a covariance block that is not positive definite. A HAC block in the warm-up, and a block of assets with different histories, need not be one, and a direction with a small positive variance on it scores a large statistic. [`RefuseIndefiniteWarmUp`](@ref), the default, refuses a block with a negative eigenvalue while the block is young, and scores it once the gate that the Mahalanobis target reads under `ExactDebias()` opens for the same assets, under every `debias`. [`ClipIndefiniteBlock`](@ref) clips the eigenvalues of its correlation at zero and keeps the variances. [`ScoreIndefiniteBlock`](@ref) scores the block as it is.",#
                 :ra_covariance => "`covariance`: Running exponentially weighted covariance matrix, seeded at zero.",#
                 :ra_weight => "`weight`: Running total of the weights that each pair of assets holds in `covariance`. It takes the step of `covariance` on a unit outer product, so a pair gains weight only at an observation where both of its assets are valid. It is `nothing` where `cor_decay` carries the correlation in `cor_state`.",#
                 :ra_cor_state => "`cor_state`: Running exponentially weighted correlation state, or `nothing` where one decay governs both the variance and the correlation.",#
                 :ra_cor_weight => "`cor_weight`: Running total of the weights that each pair of assets holds in `cor_state`. It takes the step of `cor_state` on a unit outer product, so a pair gains weight only at an observation where both of its assets are valid. It is `nothing` where one decay governs both the variance and the correlation.",#
                 :ra_XXt => "`XXt`: Working matrix for the current (possibly HAC-adjusted) outer product of the returns.",#
                 :ra_Xi => "`Xi`: Working array for the current centred returns.",#
                 # Plain exponentially weighted estimators.
                 :ew_overlap => "`overlap`: Overlap of the locations of each pair of assets, the sum over their common valid returns of the products of the two unnormalised weights, which the factor of a pair reads. It is `nothing` under [`PreCentred`](@ref), which takes no location.",#
                 :ew_weight => "`weight`: Running total of the weights that each pair of assets holds in `covariance`. It takes the step of `covariance` on a unit outer product, so a pair gains weight only at an observation where both of its assets are valid.",#
                 :ew_cache => "`cache`: Running state of an incremental fit, or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. A fit over a matrix ignores it.",#
                 :ew_mu => "`mu`: Running exponentially weighted mean vector, seeded at zero.",#
                 :sq => "`sq`: Whether to use variance instead of volatility in the inverse weighting.",#
                 :wfalg => "`alg`: Weight finaliser error formulation algorithm.",#
                 :res_retcode => "`res`: Optional result or message from the solver.",#
                 :N_msc => "`N`: Number of evenly spaced values of `gamma` the monotonic Schur complement scans for its turning point, both ends included.",#
                 :alpha_dirichlet => "`alpha`: Dirichlet concentration parameter.",#
                 :opt_hier => "`opt`: Base hierarchical optimiser configuration.",#
                 :strict_opt => "`strict`: Whether a constraint that names an asset or a group that `sets` does not hold raises an error. When it is `false`, the constraint generator warns and skips the entry.",#
                 :strict_conv => "`strict`: Whether to raise an error if convergence is not achieved.",#
                 :schalg => "`alg`: Algorithm that chooses the value of `gamma` that the allocation runs at.",#
                 # Partial fit states.
                 :pf_M => "`M`: Running second-moment accumulator of the observations folded into the state, about `mu`.",
                 :pf_max_history => "`max_history`: Optional cap on the number of observations the buffer keeps. `nothing` keeps every observation folded so far. A capped buffer drops its oldest observations as new ones arrive and holds the last `max_history` of them. The call with no data reads those rows only. The cap is the window of the fit. An estimator wrapped in [`Online`](@ref) returns the batch fit over the last `max_history` observations, also when its statistic has an exact update.",#
                 :pf_nx_pinned => "`nx`: The names of the asset columns, which the first block pins.",#
                 # The masks of a returns buffer.
                 :pf_buffer_A => "`A`: Backing matrix of the active mask, of the shape of `X`, or `nothing` when the buffer records no activity. Rows `off + 1` to `off + n` are the mask of the observations, cell for cell with them. It is fixed by the first append, and it is what lets the call with no data tell a delisting from a holiday.",#
                 :pf_buffer_M => "`M`: Backing matrix of the estimation mask, of the shape of `X`, or `nothing` when the buffer records none. It is the second per-observation mask the batch verbs take, it is carried on the same terms as `A`, and only the two regime-adjusted families read it.",#
                 :pf_active_mask => "`active_mask`: The active mask of the block, of the shape of `X`, or `nothing`. A buffer records it for every observation it holds or for none of them.",#
                 :pf_estimation_mask => "`estimation_mask`: The estimation mask of the block, of the shape of `X`, or `nothing`. It is carried on the same terms as `active_mask`.",#
                 :cvg_active_mask => "`active_mask`: The active mask of `X`, of the shape of `X`, or `nothing`. When it is not `nothing`, it takes the place of the active mask of `pnl`.",#
                 # The price gap fill.
                 :strict_span => "`strict`: If `true`, a window with a gap from a `PricesResult` that states no Listing Span raises an `ArgumentError`. If `false`, it issues a warning.",#
                 # The benchmark weights of a Factor Exposure.
                 :strict_bw => "`strict`: If `true`, a benchmark weight that is not finite on an active cell raises an `ArgumentError`. If `false`, it reads as zero, and a warning names the cells.",#
                 # The source of the returns of a factor regression.
                 :x_src_re => "`x_src`: The source of the loadings that the factor regression gives, `nothing` or an [`AbstractReturnsSource`](@ref). `nothing` takes the loadings of the Prior Result when it holds them, and otherwise fits the regression on the asset returns `rd.X`. [`DataSource`](@ref) always fits on `rd.X`. [`ExcessSource`](@ref) always fits on the returns in excess of the benchmark, `rd.X - rd.B`, the Excess Frame. [`PriorSource`](@ref) takes the loadings of the Prior Result only, and refuses a Prior Result that holds none. A regression inside a prior keeps `nothing`, and the prior states the source.",#
                 # The factor attribution.
                 :unknown_att => "`unknown`: Rule for an entry the factor model does not state, and for the variance split of a Leverage-One Pair, a subtype of [`AbstractUnknownEntryRule`](@ref). [`EntrywiseUnknown`](@ref) gives `NaN` for every number that reads one, [`ZeroUnknown`](@ref) reads it as zero, and [`KindwiseUnknown`](@ref) applies one rule to each kind.",#
                 :trim_att => "`trim`: Whether a static loadings matrix drops the first `lag` observations, as an exposure history with the same lag does. A static matrix reads every observation otherwise.")
