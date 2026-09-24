# `arg_dict`: optimisers, solvers and JuMP models, their results, cross-validation,
# prediction, allocation and the online step.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(arg_dict, :arg_dict,
                 # Risk aversion, the ratio problem and the return term it maximises.
                 :l => "`l`: Risk aversion parameter.",#
                 :ohf => "`ohf`: Objective homogenisation factor for the ratio problem, or `nothing` to size it from the resolved characteristic.",#
                 :kmin => "`kmin`: Floor on the ratio problem's homogenisation variable `k`, or `nothing` to size it from the resolved characteristic. It keeps the solver off the degenerate ray `k = 0`, on which every homogeneous constraint holds vacuously.",#
                 :i_ret_term => "`i`: Index of the return term to maximise.",#
                 # Errors
                 :msg => "`msg`: Error message describing the condition that triggered the exception.",#
                 # Solver
                 :name => "`name`: Symbol or string identifier. It is also the **key** under which [`optimise_JuMP_model!`](@ref) files this solver's failure in [`JuMPResult`](@ref)'s `trials`, so two solvers that share a name share one entry and the later failure overwrites the earlier one. Give each solver of a vector its own name, because the default `\"\"` is shared by all of them.",#
                 :solver => "`solver`: The `optimizer_factory` in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).",#
                 :settings => "`settings`: Optional solver-specific settings used in [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute).",#
                 :check_sol => "`check_sol`: Named tuple of keyword arguments splatted into [`assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.assert_is_solved_and_feasible) after each solve. It decides which solver statuses count as a solved model. The default `(;)` accepts JuMP's own defaults, `allow_local = true` and `allow_almost = false`: the termination status must be `OPTIMAL` or `LOCALLY_SOLVED`, and the primal status must be `FEASIBLE_POINT`. The strictness is deliberate — a solution the solver itself flags as approximate is rejected rather than silently accepted, so a solver stage that fails this check falls through to the next solver in the vector. The common relaxation is `check_sol = (; allow_local = true, allow_almost = true)`, which also accepts `ALMOST_OPTIMAL`, `ALMOST_LOCALLY_SOLVED` and `NEARLY_FEASIBLE_POINT`; it is what the examples, the user guide and the test suite pass, because a first-order solver reaching its tolerance on a well-posed portfolio problem is usually good enough. Go the other way with `allow_local = false` to reject `LOCALLY_SOLVED` and demand a certified global optimum, and add `dual = true` to also require a feasible dual point.",#
                 :add_bridges => "`add_bridges`: The `add_bridges` keyword argument in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).",#
                 # RNG
                 :rng => "`rng`: Random number generator.",#
                 :seed => "`seed`: Optional seed. If set, the draws come from a copy of `rng` seeded with it, so `rng` does not advance and every call draws the same values. If `nothing`, the draws come from `rng` itself.",
                 # JuMP Optimisation
                 :model => "`model::JuMP.Model`: The JuMP optimisation model.",
                 :opt_rjumpe => "`opt::RiskConstraintOwner`: The owner of the risk constraint, a risk-based JuMP optimisation estimator or a programme Allocation Set.",
                 :ci => "`i`: Constraint index for unique variable and constraint naming.",
                 :wb_arg => "`wb::WeightBounds`: Weight bound specification containing lower and upper bounds.",
                 :ss_arg => "`ss::Option{<:Number}`: Big-M scaling constant (computed via [`get_mip_ss`](@ref) when `nothing`).",
                 :lt_arg => "`lt::Option{<:Threshold}`: Long-side minimum-holding threshold.",
                 :st_arg => "`st::Option{<:Threshold}`: Short-side minimum-holding threshold.",
                 :lt_flag_arg => "`lt_flag::Bool`: Whether to apply the long-side threshold.",
                 :st_flag_arg => "`st_flag::Bool`: Whether to apply the short-side threshold.",
                 :xbgt_flag_arg => "`xbgt_flag::Bool`: Whether to pin the long/short decomposition, so the budgets built on `lw`/`sw` hold exactly (see [`set_exact_budget_constraints!`](@ref)).",
                 :il_arg => "`il`: Long binary (or continuous relaxation) indicator variable.",
                 :is_arg => "`is`: Short binary (or continuous relaxation) indicator variable.",
                 :smtx_arg => "`smtx::Option{<:MatNum}`: Selection matrix mapping assets to sub-groups.",
                 :r_risk => "`r`: Risk measure instance.",
                 :pr_X => "`pr::AbstractPriorResult`: Prior result containing the returns matrix `X`.",
                 :pr_sigma => "`pr::AbstractPriorResult`: Prior result containing the covariance matrix `sigma`.",
                 :pl_opt => "`pl`: Optional phylogeny constraints.",
                 :fees_opt => "`fees`: Optional fees structure.",
                 :b1_opt => "`b1::Option{<:MatNum} = nothing`: Factor loading matrix for [`FactorRiskContribution`](@ref); `nothing` for all other optimisers.",
                 :optargs => "`args`: Additional positional arguments passed to the optimisation function.",
                 :optkwargs => "`kwargs`: Additional keyword arguments passed to the optimisation function.",
                 :ignargs => "`args`: Additional positional arguments (ignored).",
                 :ignkwargs => "`kwargs`: Additional keyword arguments (ignored).",
                 :rd => "`rd`: The returns result to use.",
                 :pnl_prior => "`pnl`: Optional [`AssetPanel`](@ref), the panel the carrier held. A wrapping prior forwards it unchanged, so that it can compose an estimator that is fitted on a panel. An estimator that reads no panel ignores it.",
                 :pnl_moment => "`pnl`: Optional [`AssetPanel`](@ref), whose active mask the Coverage Universe of the fit is derived from. `nothing` makes the rule finiteness alone.",
                 :window => "`window`: Observation window. An integer selects the last `window` observations, and a vector of indices selects those observations.",
                 # Frontier.
                 :N_fr => "`N`: Number of sweep points on the efficient frontier. The sweep solves the model `N` times, at `N` evenly spaced bound values.",#
                 :factor_fr => "`factor`: Multiplier applied to both ends of the sweep span after `bound` has transformed them. It carries a formulation's own correction factor, such as the `inv(1 / (T - ddof))` of a second-moment bound.",#
                 :bound_fr => "`bound`: [`FrontierBoundEstimator`](@ref) that converts a bound value into the units of the risk expression the bound is applied to. The sweep points are evenly spaced in **those** units, not in the units of the measure.",#
                 # Optimisation results.
                 :pa => "`pa`: Processed optimisation attributes.",#
                 :retcode => "`retcode`: Optimisation return code.",#
                 :sol => "`sol`: Optimisation solution.",#
                 :imsk => "`imsk`: The Investable Mask the optimisation reduced on: `true` at every asset whose prior moments were finite. It is `nothing` when every asset was investable, and that sentinel is what skips both the reduction and the expansion. [`investable_mask`](@ref) derives it once from the full-universe prior result, and the result carries it, because the reduced prior can no longer yield it.",#
                 :fb => "`fb`: Fallback result or estimator.",#
                 :fb_res => "`fb`: The fallback chain that answered this result: the `(estimator, result)` pair of every attempt [`optimise`](@ref) made before this one, in the order they ran, or `nothing` when the estimator it was asked of answered (see [`FbChain`](@ref)).",#
                 # Optimiser fields.
                 :opt_jmp => "`opt`: `JuMP` optimiser configuration.",#
                 :r_opt => "`r`: Risk measure or vector of risk measures.",#
                 :r_res => "`r`: The risk measure the optimisation ran under, or a vector of them, stored **resolved** — a **Deferred Quantity** has already been fitted and an unstated slot has already taken the prior's field. A resolved measure is fitted state, not configuration, so it belongs on the Result. Pass it back as `expected_risk(res.r, res.w, res.pr; sca = res.sca)`.",#
                 :obj => "`obj`: Portfolio objective function.",#
                 :wi => "`wi`: Initial portfolio weights for warm-starting the solver.",#
                 :sca => "`sca`: Scalariser for combining multiple risk measures.",#
                 :sca_res => "`sca`: The scalariser the optimisation ran under, taken from `opt.sca`. Pass it back as `expected_risk(res.r, res.w, res.pr; sca = res.sca)` so the reported figure matches the optimised one.",#
                 :wb_jmp => "`wb`: Weight bounds estimator or weight bounds.",#
                 :bgt => "`bgt`: Net budget, `1ᵀw`. A number pins it, a [`BudgetRange`](@ref) bounds it. By default budgets *bound* the realised exposure rather than pinning it (see `xbgt`). Together with `sbgt` this fixes the net and gross exposures only jointly; to constrain the gross exposure on its own see `gbgt`.",#
                 :sbgt => "`sbgt`: Short-side budget, `sum(sw)`. A number pins it, a [`BudgetRange`](@ref) bounds it; by default it *bounds*, so `sbgt = 0.3` means *at most* 30% short unless `xbgt` pins the long/short decomposition. Together with `bgt` this fixes the net and gross exposures only jointly; to constrain the gross exposure on its own see `gbgt`.",#
                 :gbgt => "`gbgt`: Gross budget (leverage) constraint, `sum(lw) + sum(sw)`. A number pins the gross exposure; a [`BudgetRange`](@ref) bounds it, e.g. `BudgetRange(; lb = nothing, ub = 2.0)` caps leverage at 2x. Unlike `bgt` and `sbgt` — which pin the net and gross exposures only *together* — this constrains the gross exposure on its own, leaving the net free. Requires weight bounds that admit short positions, and is bounded rather than pinned unless `xbgt` is set.",#
                 :xbgt => "`xbgt`: Whether to pin the long/short decomposition exactly. When `false` (the default), `lw` and `sw` are upper bounds on the positive and negative parts of `w`, so `bgt`, `sbgt` and `gbgt` bound the realised exposures rather than pinning them — a short budget of `0.3` means *at most* 30% short. When `true`, the long/short binary indicators force `lw == max(w, 0)` and `sw == max(-w, 0)`, so the budgets hold exactly, at the cost of turning the problem into a mixed-integer program. It reuses the indicators the cardinality, threshold and fee builders already create (see `short_mip_threshold_constraints`) rather than adding its own, and is ignored when the weight bounds admit no shorts.",#
                 :lt => "`lt`: Long-side minimum holding threshold.",#
                 :st => "`st`: Short-side minimum holding threshold.",#
                 :lcse => "`lcse`: Linear constraint set estimator(s). This is the one constraint slot that also admits an `ExposureConstraintEstimator`, so a row may be written in the names of another basis — factor names, say — and re-based through the prior's loadings at generation time. What reaches the model is an ordinary asset-space `LinearConstraint` either way.",#
                 :gcarde => "`gcarde`: Grouped cardinality constraint estimator.",#
                 :sgcarde => "`sgcarde`: Sub-grouped cardinality constraint estimator(s).",#
                 :smtx => "`smtx`: Sub-group selection matrix or estimator.",#
                 :sgmtx => "`sgmtx`: Sub-grouped selection matrix or estimator.",#
                 :slt => "`slt`: Sub-group long threshold.",#
                 :sst => "`sst`: Sub-group short threshold.",#
                 :sglt => "`sglt`: Sub-grouped long threshold.",#
                 :sgst => "`sgst`: Sub-grouped short threshold.",#
                 :tn_jmp => "`tn`: Turnover constraint estimator(s).",#
                 :fees_jmp => "`fees`: Fee estimator or fee structure.",#
                 :tr_jmp => "`tr`: Tracking error constraint(s).",#
                 :ple_jmp => "`ple`: Phylogeny constraint estimator(s).",#
                 :lcsr => "`lcsr`: Processed linear constraint set result.",#
                 :gcardr => "`gcardr`: Processed grouped cardinality constraint result.",#
                 :sgcardr => "`sgcardr`: Processed sub-grouped cardinality constraint result.",#
                 :ret_jmp => "`ret`: Return term, or vector of return terms, for the `JuMP` model. Several terms are weighted-summed into the model's single scalar return expression, in the same way [`MeanRisk`](@ref)'s `r` takes several risk measures.",#
                 :ccnt => "`ccnt`: Custom `JuMP` constraint.",#
                 :cobj => "`cobj`: Custom `JuMP` objective.",#
                 :sc => "`sc`: Constraint scale factor.",#
                 :so => "`so`: Objective scale factor.",#
                 :ss => "`ss`: Optional scalar shrinkage parameter.",#
                 :card => "`card`: Global cardinality constraint.",#
                 :scard => "`scard`: Sub-group cardinality constraint(s).",#
                 :l2c => "`l2c`: 2-norm ceiling on the weights. It bounds `norm(w, 2) <= l2c * k`, where `k` is the homogenisation variable, so the ceiling holds on the weights the result reports and the budget does not scale it. Smaller `l2c` forces a more evenly spread portfolio. Used as a diversification floor via the reciprocal: `l2c = 1 / sqrt(m)` requires at least `m` effective assets (`inv(norm(w, 2)^2) >= m`). Norm-constraint family with `lpc` and `linfc`. The bound is [`Num_NormCeilCal`](@ref) under the time-dependent wrapper, so the slot takes the ceiling itself, an [`AbstractNormCeilingCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :lpc => "`lpc`: p-norm ceiling(s) on the weights at an arbitrary norm order. Each [`LpRegularisation`](@ref) supplies a norm order `p` and a bound `val`, enforcing `norm(w, p) <= val * k`. Smaller `val` forces a more evenly spread portfolio. Used as a diversification floor via the reciprocal: `val = m^(1/p - 1)` requires at least `m` order-`p` effective assets (`sum(abs.(w) .^ p)^inv(1 - p) >= m`), which is [`number_effective_assets`](@ref) taken to an arbitrary order. Norm-constraint family with `l2c` and `linfc`.",#
                 :linfc => "`linfc`: ∞-norm ceiling on the weights — a cap on the largest absolute weight: `norm(w, Inf) <= linfc * k`. So `linfc = 0.2` caps the largest weight at 20% of a fully invested portfolio. Used as a diversification floor via the reciprocal: `linfc = 1 / m` spreads the portfolio across at least `m` assets. Norm-constraint family with `l2c` and `lpc`. The bound is [`Num_NormCeilCal`](@ref) under the time-dependent wrapper, so the slot takes the ceiling itself, an [`AbstractNormCeilingCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :l1 => "`l1`: L1 regularisation coefficient. It is the ambiguity radius of a type-``\\infty`` Wasserstein ground metric, whose dual norm is the 1-norm, so the bound is [`Num_AmbRadCal`](@ref) under the time-dependent wrapper and the slot takes an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :l2 => "`l2`: L2 regularisation term(s).",#
                 :linf => "`linf`: L∞ regularisation coefficient. It is the ambiguity radius of a type-1 Wasserstein ground metric, whose dual norm is the ``\\infty``-norm, so the bound is [`Num_AmbRadCal`](@ref) under the time-dependent wrapper and the slot takes an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :lp => "`lp`: Lp regularisation specification(s).",#
                 :l2reg_val => "`val`: L2 regularisation penalty coefficient. It is the ambiguity radius of a type-2 Wasserstein ground metric, so the bound is [`Num_AmbRadCal`](@ref) and the slot takes an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments. That reading holds only for the un-squared penalty of [`SOCRiskExpr`](@ref), so [`assert_ambiguity_radius_formulation`](@ref) refuses a rule beside a squared formulation.",#
                 :l2reg_alg => "`alg`: Second-moment formulation used to express the L2 penalty.",#
                 :lpreg_p => "`p`: Norm order, `p > 1`.",#
                 :lpreg_val => "`val`: Penalty coefficient when the estimator is used as a regularisation term (the `lp` field of [`JuMPOptimiser`](@ref)), or the upper bound on the p-norm of the weights when it is used as a norm constraint (the `lpc` field). As a regularisation term it is the ambiguity radius of a type-``q`` Wasserstein ground metric with ``1/p + 1/q = 1``. As a norm constraint it is a ceiling, which is a different quantity. One field therefore carries two readings, so the bound is [`Num_AmbRadNormCeilCal`](@ref), which admits both rule families, and each of the two routes refuses the family that has no reading on it. It is the one slot that admits no plain function, because a function names no family and the two routes read the family.",#
                 :brt => "`brt`: Whether to use bootstrap returns.",#
                 :x_src => "`x_src`: Which returns matrix the clustering, phylogeny and centrality estimators read: `:prior` takes the prior result's `X`, `:data` takes the raw returns result's `X`. Ignored when no returns result is available, in which case the prior result's `X` is used.",#
                 :wf => "`wf`: Weight finaliser.",#
                 :rkb => "`rkb`: Risk budget estimator or result.",#
                 :rba => "`rba`: Risk budget algorithm.",#
                 :resi => "`resi`: Inner optimisation results.",#
                 :reso => "`reso`: Outer optimisation results.",#
                 :opti => "`opti`: Inner optimiser.",#
                 :opto => "`opto`: Outer optimiser.",#
                 # Cross-validation.
                 :n_folds => "`n`: Number of folds.",#
                 :n_test_folds => "`n_test_folds`: Number of folds held out for testing in each combination. The remaining `n_folds - n_test_folds` folds train.",#
                 :purged_size => "`purged_size`: Number of observations to purge between train and test sets.",#
                 :embargo_size => "`embargo_size`: Number of observations to embargo after the test set.",#
                 :train_idx => "`train_idx`: Training set indices.",#
                 :test_idx => "`test_idx`: Test set indices.",#
                 :train_size => "`train_size`: Training window size.",#
                 :test_size => "`test_size`: Test window size.",#
                 :period => "`period`: Time period for date-based walk-forward cross-validation.",#
                 :period_offset => "`period_offset`: Offset applied to the walk-forward period.",#
                 :adjuster => "`adjuster`: Function for adjusting walk-forward dates.",#
                 :previous => "`previous`: A date of the period grid that falls between two timestamps maps to the earlier timestamp if `true`, and to the later timestamp if `false`.",#
                 :expand_train => "`expand_train`: Whether to expand the training window over time. An Online Scheme sets it `true`, because a fold cannot un-fold an observation.",#
                 :reduce_test => "`reduce_test`: Whether to allow the last test window to be smaller.",#
                 :subset_size => "`subset_size`: Size of each random subset.",#
                 :n_subsets => "`n_subsets`: Number of random subsets.",#
                 :max_comb => "`max_comb`: Maximum number of unique asset subsets.",#
                 :window_size => "`window_size`: Rolling window size for randomised cross-validation.",#
                 :n_iter => "`n_iter`: Number of random iterations.",#
                 :cv => "`cv`: Cross-validation estimator.",#
                 :scorer => "`scorer`: Scoring function. Given the orientation-normalised score matrix (rows = CV splits, columns = parameter sets), it returns the column index of the best parameter set. The matrix is normalised so that **higher is always better**, whatever the risk measure, so a scorer selects the largest aggregate score (see [`CrossValidationSearchScorer`](@ref)).",#
                 :train_score => "`train_score`: Whether to also compute the training set score.",#
                 :path_ids => "`path_ids`: Path identifiers for cross-validation splits.",#
                 :train_scores => "`train_scores`: Training set scores.",#
                 :test_scores => "`test_scores`: Test set scores.",#
                 :lens_grid => "`lens_grid`: Grid lengths for each parameter.",#
                 :val_grid => "`val_grid`: Grid values for each parameter.",#
                 :opt_cv => "`opt`: Optimal estimator found by cross-validation.",#
                 :idx_cv => "`idx`: Index of the optimal parameter configuration.",#
                 :asset_idx => "`asset_idx`: Asset column indices per fold.",#
                 :q_scorer => "`q`: Quantile level of the population's risks that the selected path is nearest to.",#
                 :r_kwargs => "`r_kwargs`: Keyword arguments forwarded to [`expected_risk`](@ref), such as the scalariser `sca` of a vector of risk measures.",#
                 :q_kwargs => "`q_kwargs`: Keyword arguments forwarded to `Statistics.quantile`, such as its `alpha` and `beta`.",#
                 :p_cv => "`p`: Hyperparameter search grid.",#
                 :wd => "`wd`: Weight drift the fold's return series is read under, or `nothing` to read it at the target weights of the fold.",#
                 :fa_cv => "`fa`: Fee amortisation algorithm the fold's realised series charges the two fixed fee terms on, or `nothing` to inherit the clock the fee itself states. It overrides `Fees.fa` for that series alone, and it reaches the fit not at all.",#
                 :pws => "`pws`: Previous-weights source the fold loop threads into the next fold, or `nothing` to thread the target weights of the previous fold.",#
                 :store_weight_path => "`store_weight_path`: If `true`, the fold stores the weight path it computed; if `false`, a reader rebuilds it on demand.",#
                 :cv_strict => "`strict`: If `true`, a Held Gap raises an `ArgumentError`; if `false`, it warns and the pair contributes zero. A Held Gap is an (observation, asset) pair at which the fold's weight is non-zero and the asset's return is missing, which is what a delisting inside a test window makes.",#
                 :pws_wd => "`wd`: Weight drift the held weights are computed under when the return series carries no drift of its own.",#
                 # Prediction result fields.
                 :pred_nx => "`nx`: Asset name vector.",#
                 :pred_nf => "`nf`: Factor name vector.",#
                 :pred_nb => "`nb`: Benchmark name vector.",#
                 :pred_B => "`B`: Benchmark returns.",#
                 :ts => "`ts`: Timestamp vector.",#
                 :iv_ret => "`iv`: Implied volatilities.",#
                 :ivpa => "`ivpa`: Implied volatility risk premium adjustment.",#
                 :pred_res => "`res`: Optimisation result from the training fold.",#
                 :hw => "`hw`: Held-weights record of the fold, or `nothing` when the fold held its target weights on every observation.",#
                 :hw_X => "`X`: Asset returns of the fold, as the view the fold was scored over.",#
                 :hw_U => "`U`: Weight path, `observations × assets`, or `nothing` when it was not stored and is rebuilt on demand.",#
                 :hw_w => "`w`: Held weights after the last observation of the fold.",#
                 :hw_wd => "`wd`: Weight drift that produced the path and the held weights.",#
                 :pred => "`pred`: Collection of fold predictions.",#
                 :mrd => "`mrd`: Aggregated multi-period returns result.",#
                 :id_pred => "`id`: Path or fold identifier.",#
                 :opt_pred => "`opt`: The estimator the online arm of the fold loop threaded, folded through the last training end, or `nothing` on a batch run. [`Resume`](@ref) re-enters the loop from it.",#
                 # Allocation.
                 :shares => "`shares`: Number of shares allocated per asset.",#
                 :cost_alloc => "`cost`: Cost of the allocation.",#
                 :cash_alloc => "`cash`: Remaining uninvested cash after allocation.",#
                 :fees_alloc => "`fees`: Fee the allocation paid over the whole horizon. It is the sum of the two sides' charges, and it is never signed.",#
                 :unit => "`unit`: Minimum purchase unit (e.g., price per share or lot size).",#
                 # Hierarchical and Schur complement optimisers.
                 :ri => "`ri`: Inner risk measure.",#
                 :ri_res => "`ri`: The intra-cluster risk measure the optimisation ran under, or a vector of them, stored **resolved**.",#
                 :ro_res => "`ro`: The inter-cluster risk measure the optimisation ran under, or a vector of them, stored **resolved**.",#
                 :ro => "`ro`: Outer risk measure.",#
                 :scai => "`scai`: Inner scalariser.",#
                 :scao => "`scao`: Outer scalariser.",#
                 :params => "`params`: Schur complement decomposition parameters.",#
                 :gamma_schur => "`gamma`: Schur complement interpolation parameter, in `[0, 1]`. At `0` no augmentation happens and the allocation is exactly [`HierarchicalRiskParity`](@ref) under `r`. A larger value subtracts more of the cross-cluster block from each sub-cluster covariance, which moves the allocation towards the minimum variance portfolio. Under [`MonotonicSchurComplement`](@ref) it is the **upper end** of the searched range, not the value used.",#
                 :gamma_schur_res => "`gamma`: The Schur complement interpolation parameter the allocation ran at. It parallels `r`: one value for the single-bundle path, one per bundle for the multi-bundle path. Under [`MonotonicSchurComplement`](@ref) this is the value the search chose, which is at most the `gamma` the estimator asked for.",#
                 :flag_schur => "`flag`: Whether to repair an augmented covariance block that is not positive definite. When `true`, `pdm` repairs it, and a failed repair raises. When `false`, no repair happens and the allocation is abandoned instead, which is what the [`MonotonicSchurComplement`](@ref) search needs; a caller that keeps the weights gets an error naming the `gamma` that failed.",#
                 :r_res_schur => "`r`: The risk measure the optimisation ran under, stored **resolved**. It parallels `gamma`: one measure for the single-bundle path, a vector of them for the multi-bundle path. Schur carries **no** scalariser, because it carries no vector of measures to combine — `SchurComplementParams.r` is bounded to a standard deviation or a variance.",#
                 # Iterative solvers.
                 :tol => "`tol`: Convergence tolerance.",#
                 :lambda_sspo => "`lambda`: The weight of the ``L_1`` penalty of the short-term sparse portfolio.",#
                 :gamma_sspo => "`gamma`: The soft-threshold width. The ratio `lambda / gamma` is the quadratic coupling of the paper's iteration.",#
                 :iter => "`iter`: Maximum number of iterations.",#
                 # Near optimal centering.
                 :w_opt_noc => "`w_opt`: Optimal portfolio weights.",#
                 :w_min_noc => "`w_min`: Minimum risk portfolio weights.",#
                 :w_max_noc => "`w_max`: Maximum return portfolio weights.",#
                 :ucs_flag => "`ucs_flag`: Whether to use the uncertainty set.",#
                 # Optimiser config.
                 :kwargs => "`kwargs`: Additional keyword arguments.",#
                 # Near optimal centering result fields.
                 :attrs_noc => "`attrs`: Processed JuMP optimiser attributes for the model-assembly pipeline.",#
                 :w_opt => "`w_opt`: Optimal portfolio weights (vector or vector of vectors).",#
                 :w_max => "`w_max`: Maximum-risk portfolio weights.",#
                 :w_min => "`w_min`: Minimum-risk portfolio weights.",#
                 :w_opt_ini => "`w_opt_ini`: Initial weights for the optimal sub-problem.",#
                 :w_max_ini => "`w_max_ini`: Initial weights for the maximum-risk sub-problem.",#
                 :w_min_ini => "`w_min_ini`: Initial weights for the minimum-risk sub-problem.",#
                 :w_opt_retcode => "`w_opt_retcode`: Return code for the optimal-objective sub-problem.",#
                 :w_max_retcode => "`w_max_retcode`: Return code for the maximum-risk sub-problem.",#
                 :w_min_retcode => "`w_min_retcode`: Return code for the minimum-risk sub-problem.",#
                 :rt_opt => "`rt_opt`: Optimal return target.",#
                 :rt_max => "`rt_max`: Maximum return target.",#
                 :rt_min => "`rt_min`: Minimum return target.",#
                 :rt_ends => "`rt_ends`: Per-term return spans for a return-frontier sweep, as `i => (rt_min_i, rt_max_i)` pairs, or `nothing` when no return term declares a frontier bound. The aggregate `rt_min`/`rt_max` pair above serves the barrier; these serve the sweep, and the two are different quantities because a term's own span must be read off a portfolio that maximised that term alone.",#
                 :rk_opt => "`rk_opt`: Optimal risk target.",#
                 :noc_retcode => "`noc_retcode`: Return code for the near-optimal centering sub-problem.",#
                 # Discrete allocation result fields.
                 :l_model => "`l_model`: `JuMP` model for the long allocation.",#
                 :s_model => "`s_model`: `JuMP` model for the short allocation.",#
                 :l_retcode => "`l_retcode`: Return code for the long allocation sub-problem.",#
                 :s_retcode => "`s_retcode`: Return code for the short allocation sub-problem.",#
                 # Risk budgeting.
                 :prb => "`prb`: Processed risk budgeting configuration.",#
                 :l_wass => "`l`: Weight of the tail term in the Esfahani-Kuhn loss. The mean term is not scaled by it.",#
                 :r_wass => "`r`: Radius of the type-1 Wasserstein ambiguity ball. It multiplies a decision variable, so it is not a constant offset.",#
                 :g_rm => "`g`: Risk aversion parameter.",#
                 :max_phi => "`max_phi`: Maximum allowed value for any OWA weight.",#
                 :w1_owa => "`w1`: Optional first OWA weight vector.",#
                 :w2_owa => "`w2`: Optional second OWA weight vector.",#
                 :rev_owa => "`rev`: Whether `w2` is *already* reversed. It is a done-flag, not an instruction: the constructor reverses `w2` when `rev == false`, and leaves it as-is when `rev == true`. The field is stored as `true` whatever the caller passes, because `w2` is reversed by the time the object exists, so rebuilding an instance from its own fields does not reverse twice. A default-constructed instance therefore prints `rev` as `true`.",#
                 :owa_w => "`w`: Optional OWA weight vector.",#
                 :owa_method => "`method`: OWA weight estimation method.",#
                 :lm_k => "`k`: L-moment order.",#
                 :alpha_i => "`alpha_i`: Lower integration bound for the tail Gini approximation.",#
                 :a_sim => "`a_sim`: Number of integration points for the tail Gini approximation.",#
                 :beta_i => "`beta_i`: Lower integration bound for the upper tail Gini approximation.",#
                 :b_sim => "`b_sim`: Number of integration points for the upper tail Gini approximation.",#
                 # Portfolio summary statistics.
                 :ps_n_periods => "`n_periods`: Number of observations in the return series.",#
                 :ps_ppy => "`periods_per_year`: Annualisation factor. 252 for daily, 52 for weekly, 12 for monthly returns.",#
                 :ps_alpha => "`alpha`: Tail probability used for the CVaR, ``\\alpha \\in (0, 1)``.",#
                 :ps_compound => "`compound`: Whether the wealth path behind the drawdown statistics was compounded.",#
                 :ps_ann_return => "`ann_return`: Annualised arithmetic mean return.",#
                 :ps_ann_volatility => "`ann_volatility`: Annualised sample standard deviation.",#
                 :ps_sharpe => "`sharpe`: Annualised Sharpe ratio at a zero risk-free rate. `NaN` if the volatility is zero.",#
                 :ps_sharpe_stderr => "`sharpe_stderr`: Standard error of `sharpe`, corrected for the skewness and excess kurtosis of the returns, and **not** for their serial dependence. A series scored under a Weight Drift is serially dependent through the weights it held, so this figure understates the true standard error of such a series.",#
                 :ps_sortino => "`sortino`: Annualised Sortino ratio, at a zero minimum acceptable return. `NaN` if the downside deviation is zero.",#
                 :ps_calmar => "`calmar`: Annualised return divided by the absolute maximum drawdown. `NaN` if there is no drawdown.",#
                 :ps_max_drawdown => "`max_drawdown`: Maximum drawdown, in return space, so it is non-positive.",#
                 :ps_cvar => "`cvar`: Conditional Value-at-Risk at `alpha`, in return space, so a tail loss is negative.",#
                 # The fold context of the online step.
                 :cache_opt => "`cache`: Optional [`ReturnsBufferState`](@ref), the fold context of the online step. It is `nothing` until [`partial_fit!`](@ref) writes one, and `optimise(opt)` with no returns reads it. The returns themselves are carried by the prior, which owns the rows once; this holds every other column of the carrier and the context pinned at the first step. [`factory`](@ref) carries it unchanged and [`port_opt_view`](@ref) slices it to the selected assets.",#
                 :cache_rows => "`cache`: Optional [`ReturnsBufferState`](@ref), the fold context of the online step. It is `nothing` until [`partial_fit!`](@ref) writes one, and `optimise(opt)` with no returns reads it. This head holds no prior, so it is the bottom of the chain and its state carries the returns themselves, beside every other column of the carrier and the context pinned at the first step. [`factory`](@ref) carries it unchanged and [`port_opt_view`](@ref) slices it to the selected assets.")
