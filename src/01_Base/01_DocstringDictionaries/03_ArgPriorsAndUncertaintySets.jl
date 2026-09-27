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
                 :bl_rf => "`rf`: Risk-free rate. The Black-Litterman update blends the prior mean against the view returns, so it runs on the total-return scale those are written on. A mean taken from a wrapped prior estimator is on that scale already; an equilibrium mean is a bare risk premium, and the rate converts it before the update. A member with no equilibrium branch has nothing to convert and adds the rate to the posterior asset expected returns instead. It is added exactly once either way, and the wrapped prior estimators are left alone, so a risk-free rate one of them applied internally stays where it is.",#
                 # Prior results.
                 :chol => "`chol`: Cholesky factorisation of the covariance matrix.",#
                 :w_prior => "`w`: Observation weights the prior was computed under `observations × 1` (see [`ObsWeights`](@ref)), or `nothing` if it was computed unweighted. Binds `ens`, `kld` and `ow`, which are diagnostics of it (see [`forward_prior`](@ref)).",#
                 :ens => "`ens`: Effective sample size.",#
                 :ens_prior => "`ens`: Effective sample size behind the moments, or `nothing`, which every reader takes as `size(X, 1)`. Two producers write it: an entropy-pooling prior writes the effective count of its posterior weights `w`, to which it is bound (see [`forward_prior`](@ref)), and [`EmpiricalPrior`](@ref) writes the number of observations its moments were fitted over when a Scenario Cap `max_scenarios` cuts the rows `X` carries below it, so a consumer that prices a sample size reads the count behind the moments and not the rows carried.",#
                 :kld => "`kld`: Kullback-Leibler divergence of `w` from the weights it was derived from: a scalar against the prior observation weights for a single reweighting, or one entry per opinion when `w` came from pooling several.",#
                 :fpr => "`fpr`: Prior result over the factor axis, or `nothing`. Its `X` is the factor returns matrix, so its `mu`, `sigma` and `w` describe factors rather than assets, over the same observations as the asset block.",#
                 :op_w => "`ow`: Opinion pooling weights.",#
                 :reg_rr => "`rr`: Regression result.",#
                 # Prior estimators.
                 :horizon => "`horizon`: Optional investment horizon for log-normalising returns. If `nothing`, returns are not adjusted.",#
                 :fill_limit => "`fill_limit`: Share of an investable column's own observations a [`scenario_fill`](@ref) may write in silence, tested against the worst column. If `nothing`, [`resolve_fill_limit`](@ref) derives it at the fit as `1 - min_coverage` over the arms that state a coverage floor, which never fires; where no arm states one, no share passes in silence and every fill is named.",#
                 :max_scenarios => "`max_scenarios`: Optional cap on the number of observations the Prior Result carries as scenarios. `nothing` carries every observation the fit read. A cap truncates `X` to the **last** `max_scenarios` rows and leaves `mu` and `sigma` fitted over every observation, so it bounds the memory a scenario-based measure reads and changes no moment. It is the same knob in batch and online, because it is a property of the result rather than of the fold. When the cap cuts, the result states the number of observations its moments were fitted over in `ens`, so a consumer that prices a sample size — an uncertainty set, a calibration rule — reads that count rather than the rows carried; a bootstrap, which can resample nothing but the rows carried, reads the cap.",#
                 :tau => "`tau`: Blending parameter controlling the weight given to the prior relative to the views.",#
                 :views => "`views`: Views estimator or result.",#
                 :views_conf => "`views_conf`: Views confidence estimator or result.",#
                 :a_pe => "`a_pe`: Asset prior estimator.",#
                 :f_pe => "`f_pe`: Factor prior estimator.",#
                 :a_views => "`a_views`: Asset views estimator or result.",#
                 :f_views => "`f_views`: Factor views estimator or result.",#
                 :sets_af => "`sets`: Universe sets. This estimator reads **two** declared axes: `a_views` resolves against `sets.dict[sets.xkey]`, `f_views` against the time-series factor axis `sets.dict[sets.tfkey]`, and each axis must name the columns of `X` and `F` respectively, in order. Only the axis a [`LinearConstraintEstimator`](@ref) actually resolves names against is required — views supplied as a [`BlackLittermanViews`](@ref) result carry their own matrix and need no universe. A view slices the asset axis and leaves the factor entries untouched, which is why this field is `@vprop`.",#
                 :a_views_conf => "`a_views_conf`: Asset views confidence estimator or result.",#
                 :f_views_conf => "`f_views_conf`: Factor views confidence estimator or result.",#
                 :rsd => "`rsd`: Whether to include residual variance in the posterior covariance.",#
                 :f_mp => "`f_mp`: Factor matrix processing estimator.",#
                 :re => "`re`: Regression estimator.",#
                 :pes => "`pes`: Vector of prior estimators.",#
                 :pe1 => "`pe1`: Pre-processing prior estimator.",#
                 :pe2 => "`pe2`: Post-processing prior estimator.",#
                 :p_pool => "`p`: Penalty of robust opinion pooling, above zero, or `nothing` to pool the opinion weights as given. A larger value moves more weight to the opinions nearest the consensus.",#
                 # Entropy pooling.
                 :mu_views => "`mu_views`: Expected returns views estimator or result.",#
                 :var_views => "`var_views`: Value-at-risk views estimator or result.",#
                 :cvar_views => "`cvar_views`: Conditional value-at-risk views estimator or result.",#
                 :sigma_views => "`sigma_views`: Variance views estimator or result.",#
                 :sk_views => "`sk_views`: Skewness views estimator or result.",#
                 :kt_views => "`kt_views`: Kurtosis views estimator or result.",#
                 :cov_views => "`cov_views`: Covariance views estimator or result.",#
                 :rho_views => "`rho_views`: Correlation views estimator or result.",#
                 :ds_opt => "`ds_opt`: Thin wrapper for arguments and keyword arguments used in `Roots.findzero` for use with a single conditional value-at-risk view.",#
                 :dm_opt => "`dm_opt`: Optimiser for multiple conditional value at risk views.",#
                 :opt_ep => "`opt`: Entropy pooling optimisation estimator.",#
                 :evar_views => "`evar_views`: Entropic value-at-risk views estimator or result.",#
                 :sbar => "`sbar`: Number of largest losses considered by the integer conditional value-at-risk formulation. An `Integer` is a count, a fraction in `(0, 1]` is a fraction of the observations, and `nothing` applies the rule of thumb `max(2 * s, ceil(Int, 2 * alpha * T))` capped at `T`, where `s` is the number of positions, counted from the largest loss, at which the prior probabilities first reach `alpha`. The rule comes from the reference, which observes that a view above the prior CVaR needs about `s` positions and a view below it needs more. It trades exactness for solve time: `sbar = T` is always exact, and a smaller `sbar` admits only the posteriors that put at least `alpha` of their mass on the `sbar` largest losses. Raise it when the solve reports infeasibility, or when `entropy_pooling` warns that the window binds.",#
                 :zpct => "`pct`: Fractional half-width of the grid of entropic value-at-risk dual variables, centred on the value that attains the prior entropic value-at-risk. An upper-bound or equality view centres the grid on the value [`ep_evar_anchor`](@ref) finds instead, and the width then covers the movement the other views of the model cause.",#
                 :zK => "`K`: Number of points of the grid of entropic value-at-risk dual variables. Must be odd, so the centre is a point of the grid. The points are equidistant and span `zc * (1 - pct)` to `zc * (1 + pct)` for a grid centred on `zc`, so `K` sets the resolution of the grid alone. Every point is one more binary variable of the mixed-integer program an upper-bound or equality view builds, so raise it when `pct` widens rather than on its own.",#
                 :bigM => "`M`: Multiplier of the big-M constant of each row of the grid entropic value-at-risk formulation. The constant of a row is the smallest that releases it, so `M >= 1`, and `M = 1` keeps every row as tight as the data allow.",#
                 :ep_vv_views => "`views`: Value-at-risk view constraints estimator.",#
                 :ep_tv_views => "`views`: Tail view constraints estimator.",#
                 :ep_tv_alpha => "`alpha`: Significance level the views this estimator holds are read under.",#
                 :ep_tv_alg => "`alg`: Formulation used to express each view this estimator holds. A single formulation applies to every view, a vector supplies one per view, and `nothing` lets each view take the cheapest formulation that expresses it exactly.",#
                 :ep_loss => "`x`: Loss series of the asset the view names (`-returns`).",#
                 :ep_ord => "`ord`: Per asset, the indices of the largest losses in ascending order, so the largest loss is last.",#
                 :ep_view_coef => "`coef`: Per asset, the coefficient the view gives its risk measure.",#
                 :ep_view_alpha => "`alpha`: Significance level of the view.",#
                 :ep_view_op => "`op`: Comparison operator of the view, one of `:eq`, `:geq` and `:leq`.",#
                 :ep_view_rhs => "`rhs`: Target value of the view.",#
                 :ep_zgrid => "`z`: Grid of entropic value-at-risk dual variables.",#
                 :rlvar_views => "`rlvar_views`: Relativistic value-at-risk views estimator or result.",#
                 :ep_tv_kappa => "`kappa`: Deformation parameter the views this estimator holds are read under.",#
                 :ep_tv_bracket => "`bracket`: Spans the two scalar searches of this estimator run over, or `nothing` to take the span each search states.",#
                 :ep_tv_evar_zlo_frac => "`zlo_frac`: Lower end of the bracket of the dual variable, as a fraction of the upper end, or `nothing` to take the span [`ep_evar`](@ref) states. The upper end is a proof, so it is not a knob and only the lower one is.",#
                 :ep_grid_iters => "`iters`: Largest number of steps the iteration that centres the grid takes. It reaches the anchor alone, which a lower-bound view does not run.",#
                 :ep_grid_tol => "`tol`: Relative distance from the target at which the iteration that centres the grid stops. It reaches the anchor alone, which a lower-bound view does not run.",#
                 :ep_grid_tilt_iters => "`tilt_iters`: Largest number of bisection steps the tilt of one row takes (see [`ep_row_tilt`](@ref)). The bisection stops on its own when the midpoint stops moving, which for `Float64` happens near step 64, so this binds only a type of higher precision.",#
                 :ep_bracket_rlvar_tspan => "`tspan`: Number of loss spans the bracket of the shift is widened by on each side of the loss range.",#
                 :ep_bracket_rlvar_log_zlo => "`log_zlo`: Lower end of the bracket of the logarithm of the dual variable, as an offset from the logarithm of the loss range.",#
                 :ep_bracket_rlvar_log_zhi => "`log_zhi`: Upper end of the bracket of the logarithm of the dual variable, as an offset from the logarithm of the loss range.",#
                 :ep_view_kappa => "`kappa`: Deformation parameter of the view.",#
                 :rlvar_zpct => "`pct`: Fractional half-width of the grid of relativistic value-at-risk dual variables, centred on the value a posterior that meets the view attains. The centre already holds that value for a view stated on its own, so the width covers the movement the other views of the model cause. A lower-bound view, and a view whose centre is not found, falls back to the value that attains the prior relativistic value-at-risk, and the width then decides whether the view lands on its target.",#
                 :rlvar_zK => "`K`: Number of points of the grid of relativistic value-at-risk dual variables. Must be odd, so the centre is a point of the grid. It sets the resolution of the grid alone, and the spacing is `2 * pct * zc / (K - 1)` for a grid centred on `zc`. Every point is one more binary variable of the mixed-integer program an upper-bound or equality view builds, so raise it when `pct` widens rather than on its own.",#
                 :rlvar_bigM => "`M`: Multiplier of the big-M constant of each row of the grid relativistic value-at-risk formulation. The constant of a row is the smallest that releases it, so `M >= 1`, and `M = 1` keeps every row as tight as the data allow.",#
                 :ep_rlvar_zgrid => "`z`: Grid of relativistic value-at-risk dual variables.",#
                 :ep_rlvar_tgrid => "`t`: Shift variable that minimises the objective at each point of `z`, one entry per grid point. It is read under the probabilities the grid is centred on, which are the prior's only where the centre is the prior's.",#
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
                 :P => "`P`: Views loading matrix `views × assets`.",#
                 :Q => "`Q`: Views values vector `views × 1`.",#
                 :excl => "`excl`: Indices of views to exclude.",#
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
                 :sc1 => "`sc1`: Scaling parameter for the objective function.",#
                 :sc2 => "`sc2`: Scaling parameter for constraint penalties.",#
                 :epalg => "`alg`: Entropy pooling algorithm.",#
                 :epoptalg => "`alg`: Entropy pooling optimisation algorithm.",#
                 :ep_w => "`w`: Prior observation probability weights, on the observations the wrapped estimator **answers**, which a wrapped estimator that drops rows makes fewer than the observations it is given. If `nothing`, the wrapped result's own `w` is used, and the uniform weights where it carries none.",#
                 # Opinion pooling.
                 :opalg => "`alg`: Opinion pooling algorithm.")
