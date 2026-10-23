# `arg_dict`: priors and their results, entropy pooling, Black-Litterman and opinion
# pooling views, higher order priors and uncertainty sets.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(arg_dict, :arg_dict,
                 # Priors.
                 :pe => "`pe`: Prior estimator.",#
                 :pe_ucs => "`pe`: Prior estimator the set fits on the returns it is handed, or `nothing`. With `nothing` the set holds no prior of its own and is calibrated on the prior result it is handed — inside an optimiser, the prior the optimiser is solving on, so its centre is the objective's own — through the prior-result arm of the ucs triple; the returns-data form then refuses by name. The default, `EmpiricalPrior()`, fits an empirical prior on the returns.",#
                 :pr => "`pr`: Prior result.",#
                 :per => "`pr`: Prior estimator or result.",#
                 :pr_rr => "`pr`: Prior result or returns result. Both carry the asset returns matrix `X`, so either can supply it.",#
                 # The risk-free rate of the Black-Litterman update.
                 :bl_rf => "`rf`: Risk-free rate. The Black-Litterman update compares the prior mean with the view returns, so it uses the total-return scale of the views. A mean from a wrapped prior estimator has that scale. An equilibrium mean is only a risk premium, and the rate converts it before the update. A member with no equilibrium branch has no mean to convert, and it adds the rate to the posterior asset expected returns. In both cases the estimator adds the rate once. It does not change the wrapped prior estimators, so a risk-free rate that one of them applied stays in its mean.",#
                 # Prior results.
                 :chol => "`chol`: Cholesky factorisation of the covariance matrix.",#
                 :w_prior => "`w`: Observation weights of the fit of the prior, `observations × 1`, see [`ObsWeights`](@ref), or `nothing` when the fit used no weights. It binds `ens`, `kld` and `ow`, which are diagnostics of it, see [`forward_prior`](@ref).",#
                 :ens => "`ens`: Effective sample size.",#
                 :ens_prior => "`ens`: Effective sample size of the moments, or `nothing`, which every reader takes as `size(X, 1)`. Two estimators write it. An entropy pooling prior writes the effective count of its posterior weights `w`, and `w` binds it, see [`forward_prior`](@ref). [`EmpiricalPrior`](@ref) writes the number of observations of the fit of its moments, when a Scenario Cap `max_scenarios` keeps fewer rows in `X`. So a consumer that needs a sample size reads the count of the moments, and not the number of rows in `X`.",#
                 :kld => "`kld`: Kullback-Leibler divergence of `w` from the weights from which it came. It is a scalar against the prior observation weights for one reweighting, or one entry per opinion when an opinion pool made `w`.",#
                 :fpr => "`fpr`: Prior result over the factor axis, or `nothing`. Its `X` is the factor returns matrix, so its moments describe factors and not assets. It has the same observations as the asset block.",#
                 :op_w => "`ow`: The penalised opinion probabilities of the opinion pool that made the result, one entry per opinion, or `nothing` when no pool made it.",#
                 :reg_rr => "`rr`: Regression result.",#
                 :ss_gaps => "`ss_gaps`: One mark for each observation of the returns data that the prior read, `true` at an observation that the prior dropped for a Shared Series gap, or `nothing` when it dropped none. The rows of `X` are the last `size(X, 1)` observations that the marks keep, so a reader that holds the returns data finds the observation of each row, see [`shared_series_record`](@ref). [`prior`](@ref) writes it when it fits on the returns data, and when it reads out the state of an online fold. An optimiser writes it on the prior that it fits after its own cut. A returns-matrix method of `prior` drops no observation, so it leaves `nothing`.",#
                 :cache_reg => "`cache_reg`: If `true`, a prior stores on its result the cache that its refit under new scenario weights reads: a [`CrossSectionalFactorPrior`](@ref) the active mask and the Return Forecast history on its block, and an [`OpinionPoolingPrior`](@ref) the rows of `pe1`, its pooled weights and the cache of `pe2` in `pc`. The refit of an entropy pooling prior around it then runs no regression and no opinion. Every other prior forwards it to the prior that it wraps. Default is `false`.",#
                 :pc => "`pc`: The cache of the opinion pool that made the result, `(; X, w, pe2)`, or `nothing`. `X` holds the rows of `pe1` that the opinions read, or `nothing` when the pool reads the returns. `w` holds the pooled weights, and `pe2` holds the cache of the result of `pe2`. A pool fitted with `cache_reg = true` stores it. An entropy pooling prior that wraps the pool reads it when it fits again, so no opinion runs again. Every other fit leaves it `nothing`.",#
                 :strict_prior_rd => "`strict`: Passed to the returns-matrix method of the prior, where it decides if a scenario fill or a coverage peel raises in place of a warning.",#
                 :x_src_pr => "`x_src`: The source of the returns on which the prior fitted, `nothing` or an [`AbstractReturnsSource`](@ref). [`prior`](@ref) records the source that the prior estimator states. It does this when it fits on the returns data, and when it reads out the state of an online fold. `nothing` means that the fit stated no source, or that the caller gave a returns matrix, so a reader of the Result reads `X` as raw returns. The outer problem of a meta-optimiser reads it, see [`outer_returns_source`](@ref).",#
                 # Prior estimators.
                 :x_src_pe => "`x_src`: The source of the returns that the prior fits on, `nothing` or an [`AbstractReturnsSource`](@ref). `nothing` and [`DataSource`](@ref) read the asset returns `rd.X`. [`ExcessSource`](@ref) reads the returns in excess of the benchmark, `rd.X - rd.B`, the Excess Frame, and refuses returns data with no benchmark. `nothing` means that the caller stated no source, so an optimiser that holds a reader of the Excess Frame warns about a prior at `nothing`, and not about a prior at `DataSource()`. A prior reads no prior, so it refuses [`PriorSource`](@ref). A prior inside another estimator, such as the `pe` of a [`FactorPrior`](@ref), keeps `nothing`, and the outer estimator states the source.",#
                 :horizon => "`horizon`: Optional investment horizon. The prior fits the log-returns, scales their moments to the horizon, and converts them back to arithmetic moments. If `nothing`, the prior fits the arithmetic returns as they are.",#
                 :fill_limit => "`fill_limit`: The share of the observations of an investable column that [`scenario_fill`](@ref) can fill with no warning. The fill tests the worst column against it. If `nothing`, [`resolve_fill_limit`](@ref) derives it at the fit as `1 - min_coverage` of the binding floor of the arms, which never fires. Where no arm states a floor, no share passes with no warning, and the fit names every fill.",#
                 :max_scenarios => "`max_scenarios`: Optional cap on the number of observations that the Prior Result carries as scenarios. `nothing` carries every observation that the fit read. A cap cuts `X` to its last `max_scenarios` rows, and `mu` and `sigma` stay fitted over every observation. The cap limits the memory that a scenario-based measure reads, and it changes no moment. The cap is a property of the result, so it acts the same in batch and online. When the cap cuts, the result writes into `ens` the number of observations that its moments read. A consumer that prices a sample size, such as an uncertainty set or a calibration rule, reads that count and not the number of carried rows. A bootstrap can resample only the carried rows, so it reads the cap. An entropy pooling prior around the capped prior states probabilities for the scenarios alone. Its posterior moments are those of the scenarios, with a zero weight on every observation that the cap removed, see [`scenario_weights`](@ref).",#
                 :tau => "`tau`: The scale of the uncertainty of the prior mean, `tau > 0`. If `nothing`, the estimator uses `1/T`, where `T` is the number of observations.",#
                 :views => "`views`: The views, a [`LinearConstraintEstimator`](@ref) of view equations or a precomputed [`BlackLittermanViews`](@ref).",#
                 :views_conf => "`views_conf`: The confidence of the views, in `(0, 1)`, or `nothing`. A number is one confidence for all the views, and a vector holds one confidence for each view that the caller wrote. If `nothing`, the view uncertainty has no confidence scale, see [`calc_omega`](@ref).",#
                 :a_pe => "`a_pe`: Asset prior estimator.",#
                 :f_pe => "`f_pe`: Factor prior estimator.",#
                 :a_views => "`a_views`: Asset views estimator or result.",#
                 :f_views => "`f_views`: Factor views estimator or result.",#
                 :sets_af => "`sets`: Universe sets. This estimator reads **two** declared axes. `a_views` resolves against `sets.dict[sets.xkey]`, which must name the columns of `X` in order. `f_views` resolves against the time-series factor axis `sets.dict[sets.tfkey]`, which must name the columns of `F` in order. A [`LinearConstraintEstimator`](@ref) needs only the axis that it resolves names against. Views in a [`BlackLittermanViews`](@ref) result carry their own matrix and need no universe. A view slices the asset axis and keeps the factor entries, so this field is `@vprop`.",#
                 :a_views_conf => "`a_views_conf`: Asset views confidence estimator or result.",#
                 :f_views_conf => "`f_views_conf`: Factor views confidence estimator or result.",#
                 :rsd => "`rsd`: Whether to include residual variance in the posterior covariance.",#
                 :f_mp => "`f_mp`: Factor matrix processing estimator.",#
                 :mtx_sqrt => "`mtx_sqrt`: Square-root algorithm of the factor covariance, and of the idiosyncratic covariance when that block is a full matrix, whose square roots make up `chol`. The default [`EigenFallbackSquareRoot`](@ref) takes the Cholesky factor where it exists, and the eigen square root of a singular positive semidefinite matrix otherwise. [`RidgeCholeskySquareRoot`](@ref) takes the Cholesky factor of the matrix with a small ridge added, and `nothing` takes the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. [`matrix_square_root`](@ref) states each algorithm.",#
                 :srep => "`srep`: Systematic Repair rule of the lift, a member of [`AbstractSystematicRepair`](@ref). [`NoSystematicRepair`](@ref) runs every step of `mp` on the systematic block except the `:pdm` step, and [`SystematicRepair`](@ref) runs every step. Under each rule the lift then adds the idiosyncratic block and repairs the sum under the estimator that [`matrix_processing_posdef`](@ref) gives for `mp`.",#
                 :carry => "`carry`: Carry Rule of the carry fold, a member of [`AbstractCarryRule`](@ref). [`FoldOrRefit`](@ref), the default, folds each part that has a fold and fits every other part again at each step, so the carry fold equals the batch fit at a cost that can grow with the stream. [`FoldOnly`](@ref) makes the constructor refuse a part whose step cost grows, as [`carry_growing_parts`](@ref) lists it. A batch fit and the refit under [`Online`](@ref) ignore the rule.",#
                 :re => "`re`: Regression estimator.",#
                 :pes => "`pes`: Vector of entropy pooling prior estimators, one for each opinion. Each one gives the observation weights of its opinion.",#
                 :pe1 => "`pe1`: Prior estimator whose returns `X` the opinions and `pe2` read, or `nothing` to read the returns as given.",#
                 :pe2 => "`pe2`: Prior estimator that the pool fits under the pooled weights. Every moment of the result comes from it.",#
                 :op_w_pe => "`w`: Opinion probabilities, one for each estimator in `pes`, each in `[0, 1]` and with a sum of at most one, or `nothing` to give every opinion the same probability. A uniform prior over the observations takes the remainder `1 - sum(w)`.",#
                 :p_pool => "`p`: Penalty of robust opinion pooling, above zero, or `nothing` to pool the opinion weights as given. A larger value moves more weight to the opinions nearest the consensus.",#
                 # Entropy pooling.
                 :mu_views => "`mu_views`: Mean views, or `nothing` for no mean view.",#
                 :var_views => "`var_views`: Value at risk views, as one view group or a vector of groups, or `nothing` for no value at risk view.",#
                 :cvar_views => "`cvar_views`: Conditional value at risk views, as one view group or a vector of groups, or `nothing` for no conditional value at risk view.",#
                 :sigma_views => "`sigma_views`: Variance views, or `nothing` for no variance view.",#
                 :sk_views => "`sk_views`: Skewness views, or `nothing` for no skewness view.",#
                 :kt_views => "`kt_views`: Kurtosis views, or `nothing` for no kurtosis view.",#
                 :cov_views => "`cov_views`: Covariance views, or `nothing` for no covariance view.",#
                 :rho_views => "`rho_views`: Correlation views, or `nothing` for no correlation view.",#
                 :ds_opt => "`ds_opt`: Root search for one conditional value at risk view, which holds the arguments and the keyword arguments of `Roots.find_zero`. If `nothing`, the search uses a default [`ConditionalValueatRiskEntropyPooling`](@ref).",#
                 :dm_opt => "`dm_opt`: Optimiser of the search for more than one conditional value at risk view. If `nothing`, the search uses a default [`OptimEntropyPooling`](@ref) over `Optim.Fminbox`.",#
                 :opt_ep => "`opt`: Optimiser of each entropy pooling solve.",#
                 :evar_views => "`evar_views`: Entropic value at risk views, as one view group or a vector of groups, or `nothing` for no entropic value at risk view.",#
                 :sbar => "`sbar`: Number of largest losses that the integer conditional value-at-risk formulation reads. An `Integer` is a count, and a fraction in `(0, 1)` is a fraction of the observations. `nothing` applies the rule `max(2 * s, ceil(Int, 2 * alpha * T))`, capped at `T`. Here `s` is the number of positions, counted from the largest loss, at which the prior probabilities first reach `alpha`. The rule is from the reference, which observes that a view above the prior CVaR needs about `s` positions and that a view below it needs more. `sbar = T` is always exact. A smaller `sbar` admits only the posteriors that put at least `alpha` of their mass on the `sbar` largest losses, so it makes the solve faster and less exact. Increase it when the solve reports infeasibility, or when `entropy_pooling` warns that the window binds.",#
                 :zpct => "`pct`: Half-width of the grid of entropic value-at-risk dual variables, as a fraction of its centre. A lower-bound view uses the value that attains the prior entropic value at risk as the centre. An upper-bound or equality view uses the value that [`ep_evar_anchor`](@ref) finds as the centre, so the width covers only the movement that the other views of the model cause.",#
                 :zK => "`K`: Number of points of the grid of entropic value-at-risk dual variables. It must be odd, so that the centre is a point of the grid. It sets only the resolution of the grid, and the points are evenly spaced from `zc * (1 - pct)` to `zc * (1 + pct)` for a grid with the centre `zc`. Each point is one more binary variable of the mixed-integer program that an upper-bound or equality view builds, so increase `K` together with `pct`, and not on its own.",#
                 :bigM => "`M`: Multiplier of the big-M constant of each row of the grid entropic value-at-risk formulation. The constant of a row is the smallest that releases it, so `M >= 1`, and `M = 1` keeps every row as tight as the data allow.",#
                 :ep_vv_views => "`views`: Estimator of the value at risk view equations.",#
                 :ep_tv_views => "`views`: Estimator of the tail view equations.",#
                 :ep_tv_alpha => "`alpha`: Significance level the views this estimator holds are read under.",#
                 :ep_tv_alg => "`alg`: Formulation used to express each view this estimator holds. A single formulation applies to every view, a vector supplies one per view, and `nothing` lets each view take the cheapest formulation that expresses it exactly.",#
                 :ep_loss => "`x`: Loss series of the asset the view names (`-returns`).",#
                 :ep_ord => "`ord`: Per asset, the indices of the largest losses in ascending order, so the largest loss is last.",#
                 :ep_view_coef => "`coef`: Per asset, the coefficient the view gives its risk measure.",#
                 :ep_view_alpha => "`alpha`: Significance level of the view.",#
                 :ep_view_op => "`op`: Comparison operator of the view, one of `:eq`, `:geq` and `:leq`.",#
                 :ep_view_rhs => "`rhs`: Target value of the view.",#
                 :ep_zgrid => "`z`: Grid of entropic value-at-risk dual variables.",#
                 :rlvar_views => "`rlvar_views`: Relativistic value at risk views, as one view group or a vector of groups, or `nothing` for no relativistic value at risk view.",#
                 :ep_tv_kappa => "`kappa`: Deformation parameter the views this estimator holds are read under.",#
                 :ep_tv_bracket => "`bracket`: The spans of the two scalar searches of this estimator, or `nothing` for the default spans of [`RelativisticValueatRiskViewBracket`](@ref).",#
                 :ep_tv_evar_zlo_frac => "`zlo_frac`: Lower end of the bracket of the dual variable, as a fraction of the upper end, or `nothing` to take the span [`ep_evar`](@ref) states. The upper end is a proof, so it is not a knob and only the lower one is.",#
                 :ep_grid_iters => "`iters`: Largest number of steps of the iteration that finds the centre of the grid. Only the anchor reads it, and a lower-bound view does not run the anchor.",#
                 :ep_grid_tol => "`tol`: Relative distance from the target at which the iteration that finds the centre of the grid stops. Only the anchor reads it, and a lower-bound view does not run the anchor.",#
                 :ep_grid_tilt_iters => "`tilt_iters`: Largest number of bisection steps of the tilt of one row, which [`ep_row_tilt`](@ref) does. The bisection also stops when the midpoint stops moving. For `Float64` this happens near step 64, so the limit applies only to a number type of higher precision.",#
                 :ep_bracket_rlvar_tspan => "`tspan`: Number of loss spans the bracket of the shift is widened by on each side of the loss range.",#
                 :ep_bracket_rlvar_log_zlo => "`log_zlo`: Lower end of the bracket of the logarithm of the dual variable, as an offset from the logarithm of the loss range.",#
                 :ep_bracket_rlvar_log_zhi => "`log_zhi`: Upper end of the bracket of the logarithm of the dual variable, as an offset from the logarithm of the loss range.",#
                 :ep_view_kappa => "`kappa`: Deformation parameter of the view.",#
                 :rlvar_zpct => "`pct`: Half-width of the grid of relativistic value-at-risk dual variables, as a fraction of its centre. The centre is the value that a posterior attains when it meets the view. For a view alone in the model, the centre already holds that value, so the width covers only the movement that the other views of the model cause. A lower-bound view, and a view whose centre the anchor does not find, use the value that attains the prior relativistic value at risk as the centre. The width then decides whether the view reaches its target.",#
                 :rlvar_zK => "`K`: Number of points of the grid of relativistic value-at-risk dual variables. It must be odd, so that the centre is a point of the grid. It sets only the resolution of the grid, and the spacing is `2 * pct * zc / (K - 1)` for a grid with the centre `zc`. Each point is one more binary variable of the mixed-integer program that an upper-bound or equality view builds, so increase `K` together with `pct`, and not on its own.",#
                 :rlvar_bigM => "`M`: Multiplier of the big-M constant of each row of the grid relativistic value-at-risk formulation. The constant of a row is the smallest that releases it, so `M >= 1`, and `M = 1` keeps every row as tight as the data allow.",#
                 :ep_rlvar_zgrid => "`z`: Grid of relativistic value-at-risk dual variables.",#
                 :ep_rlvar_tgrid => "`t`: Shift variable that minimises the objective at each point of `z`, with one entry for each grid point. The minimisation uses the probabilities at the centre of the grid. These are the prior probabilities only when the centre is the pair of the prior.",#
                 :ep_losses => "`x`: Per asset the view names, its loss series (`-returns`).",#
                 :ep_seq_iters => "`iters`: Largest number of re-solves after the first solve. Each re-solve reads the multipliers of the primal representation at the last posterior, which tightens the surrogate row. Zero keeps the first posterior, on which the view holds but the row is slack.",#
                 :ep_seq_tol => "`tol`: Relative gap between the surrogate row and the risk measures it bounds at which the re-solves stop. It is read against the larger of the view's target and the largest loss the view names.",#
                 :ep_seq_xd => "`xd`: Per asset on the dual side of the view, its loss series (`-returns`). Once the view is oriented as a lower bound these are the assets with a positive coefficient, and each takes the exact dual block of its measure.",#
                 :ep_seq_cd => "`cd`: Per asset on the dual side of the view, the coefficient the view gives its risk measure. Positive.",#
                 :ep_seq_xp => "`xp`: Per asset on the primal side of the view, its loss series (`-returns`). Once the view is oriented as a lower bound these are the assets with a negative coefficient, and each takes a linear upper bound read from its primal representation.",#
                 :ep_seq_cp => "`cp`: Per asset on the primal side of the view, the coefficient the view gives its risk measure. Negative.",#
                 :ep_seq_row => "`c`: Coefficients of the surrogate row, one per observation. They are the coefficient-weighted sum of the linear upper bounds of the primal side, read at the last posterior.",#
                 :ep_seq_b => "`b`: Constant of the surrogate row.",#
                 # Black-Litterman views.
                 :P => "`P`: Views matrix, `views × n`, with one column for each name of the axis that the views update.",#
                 :Q => "`Q`: View returns, one for each row of `P`.",#
                 :excl => "`excl`: Indices of the views that `P` and `Q` do not hold, or `nothing`. The indices count over all the views that the caller wrote, so `excl` can be longer than `Q`.",#
                 # High order priors.
                 :skmp => "`skmp`: Coskewness matrix processing estimator.",#
                 :D2 => "`D2`: Duplication matrix.",#
                 :L2 => "`L2`: Elimination matrix.",#
                 :S2 => "`S2`: Summation matrix.",#
                 # Uncertainty sets.
                 :lb => "`lb`: Lower bound.",#
                 :ub => "`ub`: Upper bound.",#
                 :dmu => "`dmu`: Uncertainty bound for expected returns.",#
                 :dsigma => "`dsigma`: Uncertainty bound for covariance.",#
                 :dist => "`dist`: Probability distribution.",#
                 :k_ucs => "`k`: Uncertainty set scaling parameter.",#
                 :class_ucs => "`class`: Uncertainty set class.",#
                 :val_ucs => "`val`: Quantity the set is a neighbourhood of — a characteristic vector on the mean axis, a covariance matrix on the covariance axis. `nothing` defers to the consumer's own quantity. When it is set, it takes precedence over the returns estimator's field and over the prior.",#
                 :method_ucs => "`method`: Ellipsoidal uncertainty set estimation method.",#
                 :diagonal => "`diagonal`: Whether to use only the diagonal of the covariance matrix.",#
                 :dc_ucs => "`dc`: Construction of the diagonal covariance shape. It is read only when `alg` is an [`EllipsoidalUncertaintySetAlgorithm`](@ref) or a [`NormBallUncertaintySetAlgorithm`](@ref) with `diagonal = true`. [`DiagonalOfShape`](@ref) takes the diagonal of the full shape, and [`ShapeOfDiagonal`](@ref) builds the shape from the variances alone.",#
                 :eps_ucs => "`eps`: Radius of the ``\\ell_1`` uncertainty set on the characteristic vector. Larger values admit more estimation error, and therefore activate more assets.",#
                 :ep_ucs => "`ep`: Radius of the positive-error side of the signed ``\\ell_1`` uncertainty set.",#
                 :en_ucs => "`en`: Radius of the negative-error side of the signed ``\\ell_1`` uncertainty set.",#
                 :sd_ucs => "`sd`: Per-asset scaling vector for the ``\\ell_1`` uncertainty set (the estimated standard deviations). `nothing` leaves the set unscaled, so every element of the characteristic vector is assumed to suffer the same estimation error.",#
                 :mu_l1_ucs => "`mu`: Characteristic vector the ``\\ell_1`` set is a neighbourhood of. `nothing` defers to the consumer's own characteristic. When it is set, it takes precedence over the returns estimator's field and over the prior.",#
                 :method_l1_ucs => "`method`: Radius of the ``\\ell_1`` uncertainty set. A number is the radius itself; an [`AbstractUncertaintyEpsAlgorithm`](@ref) computes it from the data.",#
                 :mp_ucs => "`mp`: Radius of the positive-error side. A number is the radius itself; an [`AbstractUncertaintyEpsAlgorithm`](@ref) computes it from the data.",#
                 :mm_ucs => "`mm`: Radius of the negative-error side. A number is the radius itself; an [`AbstractUncertaintyEpsAlgorithm`](@ref) computes it from the data.",#
                 :scaled_ucs => "`scaled`: Whether to scale the uncertainty set by the estimated standard deviations. `false` assumes every characteristic suffers the same estimation error; `true` assumes assets with larger variance suffer larger estimation error, which yields inverse-volatility weights.",#
                 :active_ucs => "`active`: Target number of active assets on the *unconstrained* problem, as a count (integer `>= 1`) or a fraction of the universe (float in `(0, 1)`). This is a radius calibration, not a cardinality constraint: it selects the radius that would activate this many assets subject only to the budget and sign constraints. Any further constraint may change the realised count. Use `card` for a hard cardinality constraint.",#
                 :n_sim => "`n_sim`: Number of simulation samples.",#
                 :block_size => "`block_size`: Block size for bootstrap sampling.",#
                 :q_bs => "`q`: Confidence level that sizes the uncertainty set (`0 < q < 1`). A *smaller* `q` is more demanding and yields a *larger, more conservative* set (wider box intervals / larger ellipsoid radius); a larger `q` gives a tighter set closer to the point estimate.",#
                 :bootstrap => "`bootstrap`: Bootstrap algorithm.",#
                 :ucs => "`ucs`: Uncertainty set.",#
                 :ucsa => "`alg`: Uncertainty set algorithm.",#
                 # Entropy pooling.
                 :sc1 => "`sc1`: Scale of the dual objective.",#
                 :sc2 => "`sc2`: Weight of the norm of the slack of the fixed equality rows in the objective.",#
                 :epalg => "`alg`: Entropy pooling algorithm, which applies the views in one solve or in stages.",#
                 :epoptalg => "`alg`: Algorithm that evaluates the entropy pooling objective.",#
                 :ep_w => "`w`: Prior observation probability weights, on the observations that the wrapped estimator returns. A wrapped estimator that drops rows returns fewer observations than it gets. If `nothing`, the fit uses the `w` of the wrapped result, or the uniform weights when that result has none. The posterior probabilities take the kind of these weights, `StatsBase.AnalyticWeights` or `StatsBase.ProbabilityWeights`, and the kind fixes the bias correction of the posterior covariance. Without a `w` from the caller, the kind is the kind of the `w` of the wrapped result when it is one of the two, and `StatsBase.AnalyticWeights` otherwise. [`ScenarioProbabilities`](@ref) states the rule.",#
                 # Opinion pooling.
                 :opalg => "`alg`: The rule that pools the scenario weights of the opinions, see [`OpinionPoolingAlgorithm`](@ref).")
