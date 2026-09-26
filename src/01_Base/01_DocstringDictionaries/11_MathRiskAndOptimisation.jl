# `math_dict`: risk measures, their JuMP formulations, penalties, weight finalisation
# and meta-optimisers.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(math_dict, :math_dict,
                 # Risk measure parameters.
                 :alpha_rm => "``\\alpha``: Significance level (left tail probability), ``\\alpha \\in (0, 1)``.",#
                 :w_port => "``\\boldsymbol{w}``: Portfolio weights vector ``N \\times 1``.",#
                 # The divergence Ambiguity Set reading, in the sense CONTEXT.md
                 # gives the noun. `EntropicValueatRisk` is the Kullback-Leibler
                 # ball and `RelativisticValueatRisk` is its Kaniadakis
                 # counterpart, so the two state one set of symbols.
                 :amb_L_t => "``L_t = -x_t``: Loss at period ``t``.",#
                 :amb_P => "``P``: Sample distribution of the losses, whose ``t``-th probability is ``p_t``. It is uniform over the ``T`` observations, or the normalised observation weights when they are stated.",#
                 :amb_Q => "``Q``: Distribution in the ambiguity ball, whose ``t``-th probability is ``q_t``.",#
                 :amb_EQ_L => "``\\mathbb{E}_{Q}[L] = \\sum_{t=1}^{T} q_t L_t``: Expected loss under ``Q``.",#
                 # The Kaniadakis logarithm. `kappa_log` states it, and the
                 # relativistic risk measures, their JuMP constraint layer and the
                 # entropy pooling views all read the one symbol.
                 :ln_kappa => "``\\ln_{\\kappa}(u) = \\dfrac{u^{\\kappa} - u^{-\\kappa}}{2 \\kappa}``: Kaniadakis logarithm.",#
                 # The primal programme of the relativistic value at risk, whose
                 # per-observation power cones `ep_rlvar_tail` minimises out. The two
                 # entropy pooling view formulations and the three helpers around
                 # them state one programme, so they share these symbols.
                 :kappa_rm => "``\\kappa``: Kaniadakis deformation parameter, ``\\kappa \\in (0, 1)``.",#
                 :rlvar_loss => "``\\boldsymbol{x}``: ``T \\times 1`` loss series of one asset, the negated returns, whose ``j``-th entry is ``x_{j}``.",#
                 :rlvar_probs => "``\\boldsymbol{w}``: ``T \\times 1`` observation probabilities, summing to one. In a view they are the posterior probabilities the model solves for.",#
                 :rlvar_stat => "``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X)``: Relativistic value at risk of the loss series ``\\boldsymbol{x}`` at level ``\\alpha`` and deformation ``\\kappa``.",#
                 :rlvar_t => "``t``: Shift variable of the primal programme.",#
                 :rlvar_z => "``z > 0``: Dual variable of the primal programme.",#
                 :rlvar_u => "``u``: Shifted loss of one observation, ``t - x_{j}``.",#
                 :rlvar_sigma => "``\\sigma``: Positive root of the stationarity condition of ``\\varphi_{\\kappa}``.",#
                 :rlvar_phi => "``\\varphi_{\\kappa}(u, z)``: Smallest sum the pair of power cones of one observation allows.",#
                 :rlvar_target => "``\\bar{\\vartheta}``: Target relativistic value at risk of the view.",#
                 # The primal power-cone programme of `RelativisticValueatRisk`, which
                 # `RRM` builds and solves. The measure states it and `RRM` registers it.
                 :K_pow => "``\\mathcal{K}_{\\mathrm{pow}}(p) = \\{(a,b,c) : a^{p} b^{1-p} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}``: Power cone.",#
                 :rlvar_z_ge => "``z \\geq 0``: Scale variable of the primal power-cone programme.",#
                 :rlvar_aux => "``\\psi_i``, ``\\theta_i``, ``\\epsilon_i``, ``\\omega_i``: Auxiliary variables of observation ``i`` in the primal power-cone programme.",#
                 :p_i_obs => "``p_i``: Probability of observation ``i``. It is ``1/T`` without observation weights, and the ``i``-th observation weight divided by the sum of the weights with them.",#
                 # Absolute drawdown series.
                 :ct => "``c_t``: Cumulative simple portfolio return at period ``t``.",#
                 :dtdd => "``d_t \\leq 0``: Absolute drawdown at period ``t``.",#
                 # Relative drawdown series.
                 :Ct => "``C_t``: Compound wealth process at period ``t``.",#
                 :rdt => "``rd_t \\leq 0``: Relative drawdown at period ``t``.",#
                 # The conditional tail family of `02_ConditionalXatRisk.jl`. The
                 # value-at-risk and drawdown measures share the tail average and its
                 # sorted form, and the two robust measures share the Esfahani-Kuhn loss.
                 :nu_ru => "``\\nu``: Rockafellar-Uryasev threshold, the variable the tail average is minimised over. At a minimiser it is the lower ``\\alpha``-quantile of the series.",#
                 :x_k_sorted => "``x_{(k)}``: ``k``-th smallest entry of the series, and ``w_{(k)}`` its observation weight.",#
                 :W_k_cum => "``W_{k} = \\sum_{j=1}^{k} w_{(j)}``: Cumulative weight of the ``k`` smallest entries, with ``W_{0} = 0``.",#
                 :k_star_tail => "``k^{\\star} = \\min\\{k : W_{k} \\geq \\alpha W_{T}\\}``: Boundary index, the position of the one entry that the tail holds in part.",#
                 :W_T_total => "``W_{T} = \\sum_{t=1}^{T} w_{t}``: Total observation weight.",#
                 # The empirical quantile family of `01_XatRisk.jl`: the mixed-integer
                 # programme and the functors select one order statistic by one rule.
                 :s_mip_slack => "``s``: Cardinality slack of the mixed-integer quantile programme, ``0 < s < \\alpha``.",#
                 :k_var_mip => "``k = \\min\\{k : W_{k} > (\\alpha - s) W_{T}\\}``: Quantile index, the position of the order statistic that the mixed-integer quantile programme selects.",#
                 :l_ek => "``l``: Tail-term weight of the Esfahani-Kuhn loss, ``l > 0``. It does not scale the mean term.",#
                 :tau_ek => "``\\tau``: Esfahani-Kuhn level, the variable the worst-case expected loss is minimised over.",#
                 :pos_part => "``(\\cdot)_{+} = \\max(\\cdot, 0)``: Positive part.",#
                 # JuMP optimisation variables.
                 :k_budget => "``k``: Budget scaling / homogenisation variable.",#
                 :sc_scale => "``s_c``: Constraint scale. It multiplies both sides of a row, so a positive value leaves the feasible set unchanged.",#
                 # The cones of a norm, and the diversification floor a norm ceiling states.
                 :K_q_norm => "``\\mathcal{K}_{q} = \\{(t, \\boldsymbol{x}) : t \\geq \\lVert \\boldsymbol{x} \\rVert_{q}\\}``: Norm cone of order ``q``. ``\\mathcal{K}_{2}`` is the second-order cone.",#
                 :P_alpha_power => "``\\mathcal{P}_{\\alpha} = \\{(x, y, z) : x^{\\alpha} y^{1 - \\alpha} \\geq \\lvert z \\rvert,\\ x \\geq 0,\\ y \\geq 0\\}``: Power cone of exponent ``\\alpha \\in (0, 1)``, in the argument order of `MOI.PowerCone`.",#
                 :ENA_p => "``\\mathrm{ENA}_{p}(\\boldsymbol{w}) = \\lVert \\boldsymbol{w} \\rVert_{p}^{p/(1 - p)}``: Effective number of assets of order ``p``, for weights that sum to one. It is ``m`` on ``m`` equal weights. ``\\mathrm{ENA}_{2}(\\boldsymbol{w}) = 1 / \\lVert \\boldsymbol{w} \\rVert_{2}^{2}`` is [`number_effective_assets`](@ref), and ``\\mathrm{ENA}_{\\infty}(\\boldsymbol{w}) = 1 / \\lVert \\boldsymbol{w} \\rVert_{\\infty}``.",#
                 :m_ena => "``m``: Floor on the effective number of assets.",#
                 :b_held => "``\\boldsymbol{b} \\in \\{0, 1\\}^{M}``: Held binary, one entry for each of the ``M`` assets, sub-groups or factors that a MIP space gates. A position can be non-zero only when its entry is one. Under the long-short bundle it is the sum of the long and the short binary, which are never both one.",#
                 :i_smtx => "``i``: Index of a sub-group selection matrix, its position in the vector of matrices, or `1` for a single matrix. [`mip_key`](@ref) writes it into the name of each sub-group row.",#
                 # The regularisation penalties.
                 :lambda_2_reg => "``\\lambda_2``: L2 penalty coefficient, the `val` field of an [`L2Regularisation`](@ref).",#
                 :q_l2_power => "``q``: Power of the L2 norm that the formulation penalises. It is ``1`` under [`SOCRiskExpr`](@ref), and ``2`` under [`SquaredSOCRiskExpr`](@ref), [`QuadRiskExpr`](@ref) and [`RSOCRiskExpr`](@ref).",#
                 :lambda_p_reg => "``\\lambda_p``: Lp penalty coefficient, the `val` field of an [`LpRegularisation`](@ref) in the `lp` field of [`JuMPOptimiser`](@ref).",#
                 :so_scale => "``s_o``: Objective scale. It multiplies the objective, so a positive value leaves the argument of the optimum unchanged.",#
                 :mu_er => "``\\boldsymbol{\\mu}``: Expected returns vector ``N \\times 1``.",#
                 :R_w => "``R(\\boldsymbol{w})``: Portfolio risk.",#
                 # The value level of a risk measure: a vector of measures, its
                 # scalariser, the homogeneity correction of a risk contribution and
                 # the rolling window.
                 :S_sca_vec => "``S``: Scalariser that combines the scaled risks of a vector of risk measures into one number.",#
                 :s_k_scale => "``s_k``: `scale` of the ``k``-th risk measure of a vector, its weight in the combination.",#
                 :R_k_vec => "``R_k(\\boldsymbol{w})``: Element risk, the risk of the ``k``-th risk measure of a vector.",#
                 :p_k_chain => "``p_k``: Chain-rule weight of the ``k``-th risk measure of a vector, the partial derivative of ``S`` with respect to its ``k``-th argument at ``s_k R_k(\\boldsymbol{w})``.",#
                 :d_homog => "``d``: Degree of homogeneity of the risk measure, the ``d`` for which ``R(c \\boldsymbol{w}) = c^{d} R(\\boldsymbol{w})`` at every ``c > 0``.",#
                 :w_i_asset => "``w_i``: Weight of asset ``i``.",#
                 :n_rm_vec => "``n``: Size of the vector of risk measures.",#
                 :W_roll => "``W``: Rolling window size, in observations.",#
                 :R_t_roll => "``R_t``: Rolling risk, the risk of the window that ends at observation ``t``.",#
                 :R_w_rows => "``R(\\boldsymbol{w};\\, \\mathbf{X}_{a:b})``: Window risk, the risk of the portfolio on the rows ``a`` to ``b`` of the returns matrix.",#
                 # The risk measure layer of a JuMP model.
                 :R_i_riskvec => "``\\mathcal{R}_i``: Entry ``i`` of `risk_vec`, the risk expression of the ``i``-th risk measure times the `scale` of the measure.",#
                 :rhat_t_net => "``\\hat{r}_t``: Net portfolio return at observation ``t``, ``\\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` less the fees charged at ``t``.",#
                 # The two fee expressions of a JuMP model, and the clock of the second.
                 :f_r_fee => "``f_r``: Per period fee, the model's `:fees` expression. Each observation of the net return series pays it in full.",#
                 :f_o_fee => "``f_o``: One-off fee, the model's `:one_time_fees` expression. The holding period pays it one time.",#
                 :c_t_one_off => "``c_t``: Charge fraction of observation ``t``, the part of ``f_o`` that the observation pays. Under a `nothing` or [`FirstObservationFees`](@ref) clock ``c_1 = 1`` and ``c_t = 0`` for ``t > 1``. Under an [`AmortisedFees`](@ref) clock ``c_t = 1 / T``.",#
                 :dd_t_model => "``dd_t``: Drawdown variable of observation ``t``, with ``dd_0 = 0``. The rows hold it at or above the drawdown ``-d_t``.",#
                 :s_t_series => "``s_t``: Entry ``t`` of the series that a conic risk measure reduces. It carries the sign of a return, so a loss is a negative entry.",#
                 # Second-moment formulations.
                 :d_secmom => "``\\boldsymbol{d}``: Deviation vector ``T \\times 1`` that the formulation squares. The risk measure supplies it.",#
                 :c_secmom => "``c``: Correction factor that the risk measure supplies. It is ``1`` when the co-moment matrix already carries it.",#
                 :t_secmom => "``t``: Auxiliary model variable that the cone bounds.",#
                 # Weight finalisation.
                 :w_0_finaliser => "``\\boldsymbol{w}_{0}``: Portfolio weights vector ``N \\times 1`` that the optimisation produced, which the finaliser repairs.",#
                 :lb_ub_finaliser => "``\\boldsymbol{l}``, ``\\boldsymbol{u}``: Lower and upper weight bounds. An absent bound is dropped from the programme rather than set to an infinity.",#
                 # A finite allocation solves one integer programme per side of the book,
                 # so its symbols are those of one side. `DiscreteAllocation` and its
                 # model builders state one programme, and share these symbols.
                 :x_shares => "``\\boldsymbol{x} \\in \\mathbb{Z}_{\\geq 0}^{N}``: Share count vector of one side of a finite allocation, the model variable `x`.",#
                 :p_prices => "``\\boldsymbol{p}``: Asset price vector of one side of a finite allocation.",#
                 :C_side_cash => "``C``: Side cash, the cash that one side of a finite allocation can spend.",#
                 :w_side_target => "``\\boldsymbol{w}``: Side target weights, the weights of one side of a finite allocation normalised to sum to one. The short side is negated first, so every entry is non-negative.",#
                 :u_alloc_err => "``u``: Allocation error bound, the model variable `u` that bounds the deviation of the book from its target.",#
                 :r_cash_left => "``r = C - \\boldsymbol{x}^{\\intercal} \\boldsymbol{p}``: Leftover cash of one side, before the fee.",#
                 :m_money => "``\\boldsymbol{m} = \\boldsymbol{x} \\odot \\boldsymbol{p}``: Position money vector, the money in each position of one side.",#
                 :F_side_fee => "``F(\\boldsymbol{x})``: Side fee, the fee that one side pays over the whole horizon. It is zero when the input states no fee.",#
                 # A finite allocation splits the book into a long and a short side, and
                 # charges each side's fee on money. `setup_alloc_optim`, the collateral
                 # algorithms and the fee helpers of `01_Base_FiniteAllocation.jl` share
                 # these symbols.
                 :C_tot_alloc => "``C_{\\text{tot}}``: Total cash of a finite allocation, the `cash` of [`FiniteAllocationInput`](@ref).",#
                 :b_alloc => "``b = \\sum_{i=1}^{N} w_i``: Total budget of a finite allocation, the sum of the target weights.",#
                 :b_L_alloc => "``b_L``: Long budget, the sum of the non-negative target weights.",#
                 :b_S_alloc => "``b_S``: Short budget, the negated sum of the negative target weights. It is non-negative.",#
                 :C_L_alloc => "``C_L = C_{\\text{tot}}\\, b_L``: Long target, the money of the long side at its target weights.",#
                 :C_S_alloc => "``C_S = C_{\\text{tot}}\\, b_S``: Short target, the money of the short side at its target weights.",#
                 :C_S_prime_alloc => "``C_S^{\\prime}``: Short cash, the cash that the short side can spend.",#
                 :C_L_prime_alloc => "``C_L^{\\prime}``: Long cash, the cash that the long side can spend after the short side trades.",#
                 :m_S_alloc => "``m_S``: Short money, the money of the shares that the short side sold. It is non-negative.",#
                 :F_S_alloc => "``F_S``: Short fee, the fee that the short side paid over the whole horizon.",#
                 :T_alloc => "``T``: Allocation horizon, the number of periods over which a finite allocation charges its rates, the `horizon` of [`FiniteAllocationInput`](@ref).",#
                 :C_prev_alloc => "``C^{\\text{prev}}``: Previous cash, the cash that the portfolio held before the trade, the `prev_cash` of [`FiniteAllocationInput`](@ref).",#
                 :m_prev_alloc => "``\\boldsymbol{m}^{\\text{prev}}``: Previous money vector, the money in each position of one side before the trade.",#
                 :L_liq_alloc => "``L``: Forced-exit charge, the constant fee over the whole horizon for the assets that left the universe.",#
                 # The risk tracking family.
                 :w_b_track => "``\\boldsymbol{w}_b``: Benchmark portfolio weights vector ``N \\times 1``, the `w` of the tracking specification `tr`.",#
                 :rho_track => "``\\rho``: Tracked risk measure, the `r` field. Its value at a weight vector is the risk of that vector as [`expected_risk`](@ref) reports it.",#
                 :X_returns => "``\\mathbf{X}``: Returns matrix ``T \\times N``.",#
                 :F_fee_series => "``F(\\boldsymbol{w})``: Fee series ``T \\times 1`` of the portfolio, zero when `fees` is `nothing`. See [`calc_net_returns`](@ref).",#
                 # The Range convention (ADR 0057).
                 :negated_upper_tail => "The upper tail is the base measure applied to the negated returns ``-\\boldsymbol{x}``, so both tails are reported on the same sign convention and the range is their sum, not their difference.",#
                 # A meta-optimiser's sub-portfolios, the synthetic assets of its outer
                 # problem, and the combination of the two solves.
                 :K_sub => "``K``: Sub-portfolio count of a meta-optimiser, which is also the number of synthetic assets of its outer problem.",#
                 :W_inner => "``\\mathbf{W}``: Inner weight matrix `assets × sub-portfolios`. Column ``k`` holds the weights of sub-portfolio ``k`` over all ``N`` assets, and zero at an asset outside the sub-portfolio.",#
                 :W_tilde_syn => "``\\tilde{\\mathbf{W}}``: Normalised inner weight matrix, ``\\tilde{W}_{ik} = |W_{ik}| / \\sum_{j=1}^{N} |W_{jk}|``. A column of zeros stays a column of zeros.",#
                 :v_outer => "``\\boldsymbol{v}``: Outer weight vector, the weights that the outer optimiser gives the ``K`` synthetic assets.",#
                 :s_k_comb => "``s_k``: Combination Weight of sub-portfolio ``k``.",#
                 :c_k_comb => "``c_k``: Coefficient of sub-portfolio ``k`` in the combination.",#
                 # A search over a parameter grid. The two methods of the grid search score
                 # folds and paths, and both select the winner by one rule.
                 :Theta_grid => "``\\Theta``: Search grid, the sequence of grid points that [`lens_val_grid`](@ref) builds from `p`.",#
                 :theta_i_cand => "``\\theta_i``: Candidate ``i``, the estimator with each tuned parameter set to its value at grid point ``i``. Column ``i`` of every score matrix belongs to it.",#
                 :s_orient_search => "``s \\in \\{1, -1\\}``: Orientation of the score, ``1`` when [`bigger_is_better`](@ref) holds for `r` and ``-1`` otherwise, so a higher score is always better.",#
                 :R_search => "``\\mathcal{R}``: Expected risk under `r`, as [`expected_risk`](@ref) computes it with the keyword arguments `kwargs`.",#
                 :C_finite_cand => "``\\mathcal{C}``: Finite candidates, the candidates whose every score is finite. A failed fold scores `NaN`, so a candidate that failed one is not in ``\\mathcal{C}``.",#
                 :sigma_scorer => "``\\sigma``: Search scorer `scorer`, a map from a score matrix to the position of its best column. The default [`HighestMeanScore`](@ref) takes the column of greatest mean.",#
                 :i_star_cand => "``i^{\\star}``: Selected candidate, the position in the grid of the candidate that the search returns.")
