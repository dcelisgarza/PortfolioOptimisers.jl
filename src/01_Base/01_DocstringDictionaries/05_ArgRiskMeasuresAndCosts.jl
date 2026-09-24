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
                 :scale_rt => "`scale`: Weight of this return term in the weighted sum that forms the `JuMP` return expression. It is a combination weight, so it is inert on a single term: an optimiser given one term drops it before the return expression is built, and the value-level readers ignore it too. The lower bound in `lb` binds on the term's own expression, before `scale` is applied.",#
                 :lb_rts => "`lb`: Lower bound(s) for the return term. Can be a scalar, vector, or [`Frontier`](@ref). The bound binds on the term's own expression, net of the term's own flagged charges and before `scale` is applied, and it binds whether or not `rte` is `true`.",#
                 :rte => "`rte`: Whether to include the return term in the `JuMP` return expression.",#
                 :fee_rts => "`fee`: Whether to subtract the portfolio fees from this return term. Set it to `false` for a term that is not in return units.",#
                 :mic_rts => "`mic`: Whether to subtract the market impact cost from this return term. Set it to `false` for a term that is not in return units, or to leave the cost to the budget constraint alone.",#
                 # Risk measure fields.
                 :rc => "`rc`: Risk contribution constraint.",#
                 :alg => "`alg`: Risk measure optimisation formulation algorithm.",#
                 :cal_n => "`n`: Number of observations the tail is to hold. It is a count, not a probability, and it is the whole content of the rule.",#
                 :cal_c => "`c`: Rate coefficient. The significance level is this coefficient divided by the square root of the number of observations.",#
                 :cal_target => "`target`: Target value of the Kaniadakis logarithm, the coefficient [`RRM`](@ref) multiplies its dual variable by. The rule returns the deformation parameter that meets it.",#
                 :cal_kmin => "`kmin`: Floor under the count of order statistics the Hill estimate reads. The count is `ceil(alpha * T * N)` over the pool, and a count below this floor is refused rather than estimated: a Hill estimate over too few order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it.",#
                 :cal_kmin_rad => "`kmin`: Floor under the count of order statistics the Hill estimate reads. The count is `ceil(alpha * T)` over the radial series, and a count below this floor is refused rather than estimated: a Hill estimate over too few order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. The radial series holds one entry per observation, so the same floor binds harder here than it does over a pool.",#
                 :cal_ctx_alpha => "`alpha`: Significance level of a sibling slot the owner resolved first, or `nothing` when the site names none. The two slots travel together, so the per-type resolution resolves `alpha` first and puts the number in the context of the slot that reads it.",#
                 :cal_ctx_series => "`series`: The series the slot owner prices, one of [`ReturnsSeries`](@ref), [`AbsoluteDrawdownSeries`](@ref) and [`RelativeDrawdownSeries`](@ref), which [`calibration_series`](@ref) states. A drawdown marker puts the per-column drawdown series of `pr.X` in place of its columns, and each rule reads that substitution on its own terms.",#
                 :cal_confidence => "`confidence`: Confidence level of the chi-squared quantile the radius is read off. A higher level buys a larger ball, so the model prices a wider set of measures.",#
                 :cal_scale => "`scale`: Scale of the radius, in the units of the series the slot owner prices, or `nothing` to read the average per-asset dispersion of that series off the sample. The chi-squared factor is dimensionless, so this field carries the whole of the radius' units.",#
                 :cal_rate_c => "`c`: Rate coefficient. The radius is this coefficient divided by the square root of the number of observations.",#
                 :cal_dim_confidence => "`confidence`: Confidence level the measure-concentration bound is read at. It enters the radius as `log(1 / (1 - confidence))`, so a higher level buys a larger ball, and the exponent of the sample size flattens the buying.",#
                 :cal_dim_scale => "`scale`: Scale of the radius, in the units of the series the slot owner prices, or `nothing` to read the average per-asset dispersion of that series off the sample. The rate factor is dimensionless, so this field carries the whole of the radius' units.",#
                 :cal_fraction => "`fraction`: Fraction of the universe that must stay effective. The rule reads the asset count off the prior result and multiplies it by this fraction, so the floor moves with the universe rather than with a count the caller pins.",#
                 :cal_ctx_p => "`p`: Norm order the quantity is read against, or `nothing` when the site names none. The order belongs to the constraint or to the penalty rather than to the rule, so each site that carries one states it here.",#
                 :cal_ratio => "`ratio`: Number of mean terms that one tail term is worth. The rule returns the tail weight that prices the tail term at this multiple of the mean term, on the sample the prior result carries, so `1` is parity and `2` prices the tail term at twice the mean term.",#
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
                 :b_mip => "`b`: Big-M constant of the MIP formulation. It relaxes the bound on an observation that the model flags as an exceedance. If `nothing`, the model uses `1000`.",#
                 :s_mip => "`s`: Cardinality slack of the MIP formulation. It caps the weight of the flagged observations at `(alpha - s)` times the total weight, which is `(alpha - s) * T` without observation weights. The functor selects its order statistic with the same slack. If `nothing`, the model and the functor use `1e-5`.",#
                 :slv => "`slv`: Solver or vector of solvers.",#
                 :p_rm => "`p`: Power or order parameter.",#
                 :p_owa => "`p`: Vector of p-norm orders used to approximate the ordered weights array risk.",#
                 :pe_rm => "`pe`: Optional prior estimator that fills every prior-derived slot the measure leaves unstated, from a single fit. A stated slot wins. See [`resolve_deferred_quantities`](@ref).",#
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
                 :mu_ret_slot => "`mu`: Optional expected returns vector `assets × 1`. Also admits a **Deferred Quantity** — an expected returns estimator or a prior estimator that computes the vector against the optimisation's own prior, at [`factory`](@ref) time (see [`ArithRetMu`](@ref) and [`resolve_deferred_quantities`](@ref)). A `ucs` that carries its own centre outranks it, and it outranks the prior's own vector. If `nothing`, the prior supplies it.",#
                 :ddof => "`ddof`: Degrees-of-freedom correction.",#
                 :flag => "`flag`: Algorithm selection flag.",#
                 # Turnover.
                 :w_tn => "`w`: Reference portfolio weight vector. Deviations are measured against it, and it is never the candidate weight vector an optimiser solves for.",#
                 :w_ref => "`w`: Reference portfolio weights vector.",#
                 :w_bm_ret => "`w`: Benchmark portfolio returns vector. It holds `T` returns, one per observation, and **not** `N` weights, so its length must match the number of rows of the return matrix the model is built on.",#
                 :fixed => "`fixed`: Whether the estimator is fixed and does not update with new weights.",#
                 # Tracking specification.
                 :tr_spec => "`tr`: Benchmark tracking specification.",#
                 # Power norm parameters.
                 :pa_rm => "`pa`: Power norm parameter for the lower tail.",#
                 :pb_rm => "`pb`: Power norm parameter for the upper tail.",#
                 # Generic Value-at-Risk range components.
                 :loss_rm => "`loss`: Loss-side XatRisk risk measure applied to the portfolio returns.",#
                 :gain_rm => "`gain`: Gain-side XatRisk risk measure applied to the negated portfolio returns.",#
                 # Fees.
                 :tn_fees => "`tn`: Turnover estimator or result.",#
                 :l_fees => "`l`: Long proportional fees.",#
                 :s_fees => "`s`: Short proportional fees.",#
                 :fl => "`fl`: Long fixed fees.",#
                 :fs => "`fs`: Short fixed fees.",#
                 :lq_fees => "`lq`: Proportional liquidation fees, charged when a position leaves the Investable Mask. The carrier lives on the **complement** of the mask, so its entries are the assets that left and not the assets held. A forced exit trades to zero, so the charge is the rate times the absolute previous weight, and it falls on every period beside `l`, `s` and `tn`.",#
                 :flq_fees => "`flq`: Fixed liquidation fees, charged when a position leaves the Investable Mask. The carrier lives on the **complement** of the mask, as `lq` does. The amount is charged once for each entry whose absolute previous weight is not `isapprox` to zero under `kwargs`, and it falls on the clock `fa` names beside `fl` and `fs`.",#
                 :dl => "`dl`: Default long proportional fee.",#
                 :ds => "`ds`: Default short proportional fee.",#
                 :dfl => "`dfl`: Default long fixed fee.",#
                 :dfs => "`dfs`: Default short fixed fee.",#
                 :fa_fees => "`fa`: Fee amortisation algorithm, and the clock the two fixed fee terms fall on. `nothing` and a [`FirstObservationFees`](@ref) charge them one time, on the first observation of a return series. An [`AmortisedFees`](@ref) spreads them evenly over the observation count the charging site hands in. It reaches no other term, because `l`, `s` and `tn` are rates per period.",#
                 :kwargs_fee => "`kwargs`: Named tuple of keyword arguments for fee computation.",#
                 :imsk_fees => "`imsk`: The Investable Mask this fee was reduced on, or `nothing` when it is a caller's statement over the full universe. A door writes it, [`investable_fees_view`](@ref), and a caller never does. When it is set, `tn`, `l`, `s`, `fl` and `fs` are on `findall(imsk)` and `lq`, `flq` on its complement; a door that meets the fee a second time then passes it through, where it would drop or re-slice a caller's carriers.",#
                 # Tracking error.
                 :err => "`err`: Tracking error tolerance.",#
                 :tralg => "`alg`: Tracking formulation algorithm.",#
                 # The risk measures of a ratio, and the scalarisers that combine them.
                 :rt => "`rt`: Returns estimator, or a vector of them. A vector is summed at its terms' `settings.scale` weights, skipping any term whose `settings.rte` is `false`. There is no scalariser on the return axis.",#
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
                 :tr => "`tr`: Tracking error constraint estimator.",#
                 # Fees.
                 :fees => "`fees`: Fees estimator or result.",#
                 :fees_res => "`fees`: The resolved [`Fees`](@ref) the head was charged with, on the universe it solved on, or `nothing`. A walk-forward fold charges it through [`extract_fees`](@ref).",#
                 :proj => "`proj`: The Projection Geometry the rule projects its raw step onto the Allocation Set in; the slot's type bound names the geometries the rule's theorem covers.",#
                 :price_window => "`window`: The number of price levels the statistic reads, the current one included; `window - 1` returns reconstruct them.",#
                 :forecaster => "`me`: The forecaster whose mean is the Price Relative Forecast, `x̂ = 1 .+ mu`: folded on the Rule State where it has an exact fold, and refit over the rows the head holds otherwise.",#
                 # Non-optimisation risk measures.
                 :rt_mean => "`rt`: Mean return estimator.")
