# `arg_dict`: risk measures and their settings, return terms, tracking, turnover,
# fees and trading costs.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(arg_dict, :arg_dict,
                 # Turnover.
                 :tnr => "`tn`: Turnover result.",
                 # Fees.
                 :feese => "`fees`: Fees estimator.",#
                 :feesr => "`fees`: Fees result.",
                 # Trading costs.
                 :bgt_cost_target => "`bgt`: Budget target or range that the weights and their trading costs must meet together.",#
                 :vp_cost => "`vp`: Cost coefficients for positive weight changes. Non-negative.",#
                 :vn_cost => "`vn`: Cost coefficients for negative weight changes. Non-negative.",#
                 :up_cost => "`up`: Upper limit on positive weight changes. Non-negative.",#
                 :un_cost => "`un`: Upper limit on negative weight changes. Non-negative.",#
                 :beta_mic => "`beta`: Reciprocal of the market impact exponent, `0 < beta < 1`. The realised exponent is `1/beta`.",#
                 # Risk measure settings.
                 :settings_rm => "`settings`: Risk measure settings.",#
                 :scale_rm => "`scale`: Weight of this risk measure in the aggregate risk expression formed from a vector of measures. It is a combination weight, so it is inert on a single measure: an optimiser given one measure drops it before the risk expression is built, and the value-level readers ignore it too. The upper bound in `ub` binds on the measure's own expression, before `scale` is applied.",#
                 :ub_rms => "`ub`: Upper bound(s) on the measure's own risk expression. A scalar bounds one model. A vector and a [`Frontier`](@ref) are sweep axes, one solve per entry, so the optimisation returns one portfolio per bound value.",#
                 :lb_rms => "`lb`: Lower bound(s) on the measure's own risk expression, for a quantity the optimisation maximises. A scalar bounds one model. A vector and a [`Frontier`](@ref) are sweep axes, one solve per entry. A **negative** value is meaningful, because the quantity it bounds may be negative.",#
                 :rke => "`rke`: Whether to include the risk measure value in the `JuMP` risk expression.",#
                 # Return term settings.
                 :settings_rt => "`settings`: Return term settings.",#
                 :scale_rt => "`scale`: Weight of this return term in the weighted sum that forms the `JuMP` return expression. It weighs one term against the others, so it has no effect on a single term. An optimiser given one term drops it before it builds the return expression, and the value-level readers ignore it too. The lower bound in `lb` binds on the term's own expression, before `scale` applies.",#
                 :lb_rts => "`lb`: Lower bound(s) for the return term. Can be a scalar, vector, or [`Frontier`](@ref). The bound binds on the term's own expression, net of the term's own flagged charges and before `scale` is applied, and it binds whether or not `rte` is `true`.",#
                 :rte => "`rte`: Whether to include the return term in the `JuMP` return expression.",#
                 :fee_rts => "`fee`: Whether to subtract the portfolio fees from this return term. Set it to `false` for a term that is not in return units.",#
                 :mic_rts => "`mic`: Whether to subtract the market impact cost from this return term. Set it to `false` for a term that is not in return units, or to leave the cost to the budget constraint alone.",#
                 # Risk measure fields.
                 :rc => "`rc`: Risk contribution constraint.",#
                 :alg => "`alg`: Risk measure optimisation formulation algorithm.",#
                 :ratio_alg => "`ratio_alg`: The ratio that [`MaximumRatio`](@ref) maximises when the model states the measure on a lifted weight matrix. [`SquareRootRatio`](@ref) divides the excess return by the square root of the measure, which is the Sharpe ratio for a plain variance. [`LinearRatio`](@ref) divides it by the measure itself. A formulation that lifts nothing, a bound, and every other objective ignore it.",#
                 :corner_sdp => "`corner`: The constant corner ``c`` of the lift of the objective term, a finite number ``c > 0``. Only [`SquareRootRatio`](@ref) reads it. If `nothing`, the model derives it from the objective and the prior, [`set_sdp_lift_corner!`](@ref). The corner changes the scale of the lifted matrix, and so the conditioning of the solve, but not the weights.",#
                 :cal_n => "`n`: Number of scenarios that the tail must hold. It is a count and not a probability.",#
                 :cal_c => "`c`: Rate coefficient. The significance level is this coefficient divided by the square root of the number of observations.",#
                 :cal_target => "`target`: Target value of the Kaniadakis logarithm. It is the coefficient by which [`RRM`](@ref) multiplies its dual variable. The rule returns the deformation parameter that gives this value.",#
                 :cal_kmin => "`kmin`: Floor under the number of order statistics that the Hill estimate reads. The count is `ceil(alpha * T * N)` over the pool. The rule refuses a count below this floor, because a Hill estimate over too few order statistics changes from fold to fold with no cause in the data, and the deformation parameter changes with it.",#
                 :cal_kmin_rad => "`kmin`: Floor under the number of order statistics that the Hill estimate reads. The count is `ceil(alpha * T)` over the radial series. The rule refuses a count below this floor, because a Hill estimate over too few order statistics changes from fold to fold with no cause in the data, and the deformation parameter changes with it. The radial series holds one entry per observation, so this floor refuses a sample that the same floor over a pool accepts.",#
                 :cal_ctx_alpha => "`alpha`: Significance level of a sibling slot that the owner resolved first, or `nothing` when the site gives none. The resolution method of the owner resolves `alpha` first, and it puts the number in the context of the slot that reads it.",#
                 :cal_ctx_series => "`series`: The series that the slot owner prices: [`ReturnsSeries`](@ref), [`AbsoluteDrawdownSeries`](@ref) or [`RelativeDrawdownSeries`](@ref). [`calibration_series`](@ref) gives it. A drawdown marker replaces each column of `pr.X` with the drawdown series of that column, and each rule reads the replacement in its own way.",#
                 :cal_confidence => "`confidence`: Confidence level of the chi-squared quantile that sets the radius. A higher level gives a larger ball, so the model prices a wider set of probability measures.",#
                 :cal_scale => "`scale`: Scale of the radius, in the units of the series that the slot owner prices, or `nothing` to read the average per-asset dispersion of that series off the sample. The chi-squared factor has no units, so this field gives the radius all of its units.",#
                 :cal_rate_c => "`c`: Rate coefficient. The radius is this coefficient divided by the square root of the number of observations.",#
                 :cal_dim_confidence => "`confidence`: Confidence level of the measure-concentration bound. It enters the radius as `log(1 / (1 - confidence))`, so a higher level gives a larger ball. The exponent `1 / max(N, 2)` applies to this term too, so a change of level moves the radius less as `N` increases.",#
                 :cal_dim_scale => "`scale`: Scale of the radius, in the units of the series that the slot owner prices, or `nothing` to read the average per-asset dispersion of that series off the sample. The rate factor has no units, so this field gives the radius all of its units.",#
                 :cal_fraction => "`fraction`: Fraction of the universe that must stay effective. The rule multiplies this fraction by the asset count of the prior result, so the floor changes with the universe and not with a count that the caller fixes.",#
                 :cal_ctx_p => "`p`: Norm order against which the rule reads the quantity, or `nothing` when the site gives none. The order belongs to the constraint or to the penalty and not to the rule, so each site that has an order gives it here.",#
                 :cal_ctx_cs => "`cs`: The fit of a [`CrossSectionalFactorPrior`](@ref) as `(; g, ap, csfm, hist, cre)`, or `nothing` when the site gives none. `g` holds the spanned coefficients of the Return Forecast, one per estimated factor. `ap` holds its unscaled orthogonal part, one per asset. Both are `nothing` when the prior has no Return Forecast Estimator. `csfm` is the [`CrossSectionalFactorModel`](@ref) block, with a zero `b` and no Return Forecast. It holds the idiosyncratic variance history `vs` and the regression weights `rw`. `hist` is the Return Forecast history from [`forecast_history`](@ref), or `nothing` unless a slot answers `true` to [`reads_forecast_history`](@ref). `cre` is the Cross-Sectional Regression Estimator of the prior. It splits a row of the history in the same way as the prior splits the latest forecast.",#
                 :cal_ctx_n => "`n`: Count of the scenarios that the site prices, or `nothing` when the site prices the observations behind the moments. A distributionally robust measure centres its ball on the empirical measure of its rows, so it gives the count of [`scenario_sample_size`](@ref) here. A radius rule reads [`effective_sample_size`](@ref) when the field is `nothing`.",#
                 :cal_ratio => "`ratio`: Number of mean terms that one tail term is worth. The rule returns the tail weight that prices the tail term at this multiple of the mean term on the sample of the prior result. So `1` is parity, and `2` prices the tail term at two times the mean term.",#
                 :vr_rm => "`vr`: Variance risk measure component.",#
                 :sk_rm => "`sk`: Skewness risk measure component.",#
                 :kt_rm => "`kt`: Kurtosis risk measure component.",#
                 :alg1 => "`alg1`: First algorithm variant.",#
                 :alg2 => "`alg2`: Second algorithm variant.",#
                 :N_kt => "`N`: Optional number of eigenvalues per asset for the approximate cokurtosis formulation.",#
                 :kappa => "`kappa`: Relativistic deformation parameter. The bound is [`Num_DefCal`](@ref), so the slot takes the parameter itself, an [`AbstractDeformationCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :kappa_a => "`kappa_a`: Relativistic deformation parameter for the lower tail. The bound is [`Num_DefCal`](@ref), so the slot takes the parameter itself, an [`AbstractDeformationCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :kappa_b => "`kappa_b`: Relativistic deformation parameter for the upper tail. The bound is [`Num_DefCal`](@ref), so the slot takes the parameter itself, an [`AbstractDeformationCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :l_a => "`l_a`: Weight of the tail term in the Esfahani-Kuhn loss of the lower tail. The mean term is not scaled by it. The bound is [`Num_AmbTwtCal`](@ref), so the slot takes the weight itself, an [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :r_a => "`r_a`: Radius of the type-1 Wasserstein ambiguity ball of the lower tail. It multiplies a decision variable, so it is not a constant offset. The bound is [`Num_AmbRadCal`](@ref), so the slot takes the radius itself, an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :l_b => "`l_b`: Weight of the tail term in the Esfahani-Kuhn loss of the upper tail. The mean term is not scaled by it. The bound is [`Num_AmbTwtCal`](@ref), so the slot takes the weight itself, an [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :r_b => "`r_b`: Radius of the type-1 Wasserstein ambiguity ball of the upper tail. It multiplies a decision variable, so it is not a constant offset. The bound is [`Num_AmbRadCal`](@ref), so the slot takes the radius itself, an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                 :gamma => "`gamma`: Log-sum-exp scalariser smoothing parameter.",#
                 :b_mip => "`b`: Big-M constant of the MIP formulation. It relaxes the bound on an observation that the model flags as an exceedance. If `nothing`, the model derives the smallest exact constant from the data and the weight bounds with [`mip_big_m`](@ref), and uses `1000` where the model admits no bound on the gross exposure.",#
                 :s_mip => "`s`: Cardinality slack of the MIP formulation. It caps the weight of the flagged observations at `(alpha + s)` times the total weight, which is `(alpha + s) * T` without observation weights, so it absorbs the rounding error of `alpha` times the total weight. The functor selects its order statistic with the same slack. If `nothing`, the model and the functor use `1e-9`.",#
                 :slv => "`slv`: Solver or vector of solvers.",#
                 :p_rm => "`p`: Power or order parameter.",#
                 :p_owa => "`p`: Vector of p-norm orders used to approximate the ordered weights array risk.",#
                 :merge_owa => "`merge`: Sorting network that the formulation relaxes. Every sorting network gives the exact risk. The network sets the number of rows and the cost of one solver iteration.",#
                 :ordering_owa => "`ordering`: Order of the network variables in the model. It does not change the risk.",#
                 :pe_rm => "`pe`: Optional prior estimator that fills every prior-derived slot the measure leaves unstated, from a single fit. A stated slot wins. See [`resolve_deferred_quantities`](@ref). It fits on the returns of the Prior Result that the measure receives, so it states no source of its own, and the constructor refuses a `pe` whose `x_src` is not `nothing`, see [`assert_returns_source`](@ref).",#
                 # Deferred Quantity slots. Each admits the value itself or the Estimator that
                 # computes it, resolved against the optimisation's own prior. See
                 # `DeferredQuantity` and ADR 0051.
                 :mu_slot => "`mu`: Optional centre the moment is taken about, a scalar or a vector `assets × 1`. Also admits a **Deferred Quantity** — an expected returns estimator or a prior estimator that computes the centre against the optimisation's own prior, at [`factory`](@ref) time (see [`MuSlot`](@ref) and [`resolve_deferred_quantities`](@ref)). If `nothing`, the prior supplies it.",#
                 :sigma_slot => "`sigma`: Optional covariance matrix `assets × assets`. Also admits a **Deferred Quantity** — a covariance estimator or a prior estimator that computes the matrix against the optimisation's own prior, at [`factory`](@ref) time (see [`SigmaSlot`](@ref) and [`resolve_deferred_quantities`](@ref)). If `nothing`, the prior supplies it.",#
                 :chol_slot => "`chol`: Optional Cholesky factorisation of the covariance matrix. Derived from `sigma`, so it never defers: it arrives as one pair with whatever `sigma` resolves to. Give it with a matrix `sigma` and with neither otherwise — stating it without `sigma`, or while `sigma` holds a Deferred Quantity, is refused at construction (see [`assert_derived_slot_has_source`](@ref)). If `nothing`, the prior supplies the pair, or the kernel derives the factorisation from a stated `sigma`.",#
                 :kt_slot => "`kt`: Optional cokurtosis matrix `assets^2 × assets^2`. Also admits a **Deferred Quantity** — a cokurtosis estimator or a prior estimator that computes the matrix against the optimisation's own prior, at [`factory`](@ref) time (see [`KtSlot`](@ref) and [`resolve_deferred_quantities`](@ref)). A cokurtosis estimator supplies `mu` as well, from its own `me`, so that the tensor and the centre it was taken about come out of one object. If `nothing`, the prior supplies it.",#
                 :sk_slot => "`sk`: Optional coskewness matrix `assets × assets^2`. Also admits a **Deferred Quantity** — a coskewness estimator or a prior estimator that computes the matrix against the optimisation's own prior, at [`factory`](@ref) time (see [`SkSlot`](@ref) and [`resolve_deferred_quantities`](@ref)). A coskewness estimator supplies `mu` as well, from its own `me`, so that the tensor and the centre it was taken about come out of one object. If `nothing`, the prior supplies it.",#
                 :V_slot => "`V`: Optional sum of the negative spectral slices of the coskewness matrix `assets × assets`. Derived from `sk`, so it never defers: it arrives as one pair with whatever `sk` resolves to, and the matrix processing estimator that built it travels with it and replaces `mp`. Give it with a matrix `sk` and with neither otherwise. Stating it while `sk` holds a Deferred Quantity is refused at construction.",#
                 :mu_dvar_slot => "`mu`: Optional expected returns vector `assets × 1`, the location of `dist`. Also admits a **Deferred Quantity** — an expected returns estimator or a prior estimator that computes the vector against the optimisation's own prior, at [`factory`](@ref) time (see [`MuSlot`](@ref) and [`resolve_deferred_quantities`](@ref)). If `nothing`, the prior supplies it.",#
                 :mu_mad_slot => "`mu`: Centre the absolute deviation is taken about. It is a [`MedianCenteringFunction`](@ref) that centres the portfolio series at the point of use, a scalar or a vector `assets × 1`, or a **Deferred Quantity** — an expected returns estimator or a prior estimator that computes the centre against the optimisation's own prior, at [`factory`](@ref) time (see [`MedAbsDevMu`](@ref) and [`resolve_deferred_quantities`](@ref)). There is no `nothing` state; the default is [`MedianCentering`](@ref).",#
                 :mu_ret_slot => "`mu`: Optional expected returns vector `assets × 1`. It also admits a Deferred Quantity, an expected returns estimator or a prior estimator that computes the vector from the optimisation's own prior when [`factory`](@ref) runs. [`ArithRetMu`](@ref) is its bound, and [`resolve_deferred_quantities`](@ref) resolves it. A `ucs` that carries its own centre outranks it, and it outranks the prior's own vector. If `nothing`, the prior supplies it.",#
                 :ddof => "`ddof`: Degrees-of-freedom correction, an integer that the library subtracts from the number of observations itself.",#
                 :flag => "`flag`: Algorithm selection flag.",#
                 # Turnover.
                 :w_tn => "`w`: Reference portfolio weight vector. The deviation of each asset is the difference between the candidate weight and this weight. It is never the candidate weight vector that an optimiser solves for.",#
                 :val_tn => "`val`: Turnover bound. A number is one bound for every asset, and a vector holds one bound per asset. A [`TurnoverEstimator`](@ref) can also name the bounds by asset or by group, and [`turnover_constraints`](@ref) resolves the names to one bound per asset. A [`Fees`](@ref) reads the `val` of its [`Turnover`](@ref) as a fee rate per asset, not as a bound.",#
                 :w_ref => "`w`: Reference portfolio weights vector.",#
                 :w_bm_ret => "`w`: Benchmark return series. It holds `T` returns, one for each observation, not `N` weights. Its length must equal the number of rows of the return matrix of the model.",#
                 :B_data_tr => "`B`: Benchmark returns of the rows that a reader holds, from the `B` field of the [`ReturnsResult`](@ref) of the reader. A vector `T × 1` is the benchmark series. A matrix `T × N` holds one benchmark for each asset, and its series is `B * w`. Every reader that holds the returns data replaces it with the `B` of its own rows through [`bind_data_benchmark`](@ref). It is `nothing` until a reader binds it.",#
                 :fixed => "`fixed`: If `true`, [`factory`](@ref) keeps the reference weights `w` and ignores the new weights. If `false`, `factory` replaces `w` with the new weights.",#
                 # Tracking specification.
                 :tr_spec => "`tr`: Benchmark tracking specification.",#
                 # Power norm parameters. `PowerNormValueatRisk` and its two drawdown
                 # siblings share `p_pnvar`.
                 :p_pnvar => "`p`: Order of the power norm of the tail, `p > 1`. As `p` falls to one, the measure falls to the Conditional Value-at-Risk of the same series at the same level. The value does not fall as `p` rises, and it is the largest loss of the series once `p` is large enough.",#
                 :pa_rm => "`pa`: Order of the power norm of the loss tail, `pa > 1`.",#
                 :pb_rm => "`pb`: Order of the power norm of the gain tail, `pb > 1`.",#
                 # Generic Value-at-Risk range components.
                 :loss_rm => "`loss`: Tail measure of the losses, a member of [`ValueatRiskRMs`](@ref). The range applies it to the portfolio returns.",#
                 :gain_rm => "`gain`: Tail measure of the gains, a member of [`ValueatRiskRMs`](@ref). The range applies it to the negated portfolio returns.",#
                 # Fees.
                 :tn_fees => "`tn`: Turnover estimator or result.",#
                 :l_fees => "`l`: Long proportional fees.",#
                 :s_fees => "`s`: Short proportional fees. A negative rate down to `-l` is a credit on a short position.",#
                 :fl => "`fl`: Long fixed fees.",#
                 :fs => "`fs`: Short fixed fees.",#
                 :lq_fees => "`lq`: Proportional liquidation fees, which the fee charges when a position leaves the Investable Mask. The field `lq` is on the complement of the mask, so its entries are the assets that left and not the assets held. A forced exit trades to zero, so the charge is the rate times the absolute previous weight, and it falls on every period beside `l`, `s` and `tn`.",#
                 :flq_fees => "`flq`: Fixed liquidation fees, which the fee charges when a position leaves the Investable Mask. The field `flq` is on the complement of the mask, as `lq` is. The fee charges the amount once for each entry whose absolute previous weight is not `isapprox` to zero under `kwargs`, and it falls on the clock `fa` names beside `fl` and `fs`.",#
                 :dl => "`dl`: Default long proportional fee.",#
                 :ds => "`ds`: Default short proportional fee. A negative rate is a credit on a short position.",#
                 :dfl => "`dfl`: Default long fixed fee.",#
                 :dfs => "`dfs`: Default short fixed fee.",#
                 :fa_fees => "`fa`: Fee amortisation algorithm, and the clock the fixed fee terms `fl`, `fs` and `flq` fall on. `nothing` and a [`FirstObservationFees`](@ref) charge them one time, on the first observation of a return series. An [`AmortisedFees`](@ref) spreads them evenly over the observation count that the calling site passes in. It affects no other term, because `l`, `s`, `tn` and `lq` are rates per period.",#
                 :kwargs_fee => "`kwargs`: Named tuple of keyword arguments that the fixed fee terms forward to `isapprox`, to decide how near zero a weight must be to count as zero.",#
                 :imsk_fees => "`imsk`: The Investable Mask that [`investable_fees_view`](@ref) reduced this fee on, or `nothing` when the fee is a caller's statement over the full universe. Only `investable_fees_view` writes it, and a caller never does. When it is set, `tn`, `l`, `s`, `fl` and `fs` are on `findall(imsk)`, and `lq` and `flq` are on its complement. `investable_fees_view` returns a marked fee unchanged when it meets it a second time. It drops or slices the two liquidation charges, `lq` and `flq`, of an unmarked fee only.",#
                 # Tracking error.
                 :err => "`err`: Tracking error tolerance.",#
                 :tralg => "`alg`: Tracking formulation algorithm.",#
                 # The risk measures of a ratio, and the scalarisers that combine them.
                 :rt => "`rt`: Return estimator, or a vector of them. The measure sums a vector at the `settings.scale` weights of its terms, and it skips a term whose `settings.rte` is `false`. The return axis has no scalariser.",#
                 :rk => "`rk`: Risk measure for ratio computation, or a vector of them scalarised by `sca`.",#
                 :r1 => "`r1`: First risk measure.",#
                 :r2 => "`r2`: Second risk measure.",#
                 :r1_vec => "`r1`: First risk measure, or a vector of them scalarised by `sca1`.",#
                 :r2_vec => "`r2`: Second risk measure, or a vector of them scalarised by `sca2`.",#
                 :sca_rk => "`sca`: Scalariser combining the risk measures in `rk` into one number. Inert when `rk` holds a single measure. The field beats a `sca` keyword supplied at the call site.",#
                 :sca_r1 => "`sca1`: Scalariser combining the risk measures in `r1` into one number. Inert when `r1` holds a single measure.",#
                 :sca_r2 => "`sca2`: Scalariser combining the risk measures in `r2` into one number. Inert when `r2` holds a single measure.",#
                 # Risk measure.
                 :r => "`r`: Risk measure or vector of risk measures.",#
                 # Tracking.
                 :tr => "`tr`: Tracking algorithm that gives the benchmark return series.",#
                 # Fees.
                 :fees => "`fees`: Fees estimator or result.",#
                 :fees_res => "`fees`: The resolved [`Fees`](@ref) the head was charged with, on the universe it solved on, or `nothing`. A walk-forward fold charges it through [`extract_fees`](@ref).",#
                 :proj => "`proj`: The Projection Geometry the rule projects its raw step onto the Allocation Set in; the slot's type bound names the geometries the rule's theorem covers.",#
                 :price_window => "`window`: The number of price levels the statistic reads, the current one included; `window - 1` returns reconstruct them.",#
                 :forecaster => "`me`: The forecaster whose mean is the Price Relative Forecast, `x̂ = 1 .+ mu`: folded on the Rule State where it has an exact fold, and refit over the rows the head holds otherwise.",#
                 # Non-optimisation risk measures.
                 :rt_mean => "`rt`: Mean return estimator.")
