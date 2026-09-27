# `math_dict`: online portfolio selection.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(math_dict, :math_dict,
                 # The online portfolio selection family. Its rules read one price
                 # relative per period and step an iterate, so these symbols are the
                 # family's notation.
                 :x_t_rel => "``\\boldsymbol{x}_t``: Price relative vector of period ``t``, each asset's price over its price one period earlier.",#
                 :w_t_iter => "``\\boldsymbol{w}_t``: Iterate of the rule at period ``t``, the allocation its update reads.",#
                 :eta_t_lr => "``\\eta_t``: Learning rate of the update at period ``t``.",#
                 :t_period => "``t``: Period index, counted from one.",#
                 :g_t_loss => "``\\boldsymbol{g}_t``: Gradient of the loss of period ``t``, taken at the iterate.",#
                 :ghat_t => "``\\hat{\\boldsymbol{g}}_t``: Transformed gradient of period ``t``, the vector the mirror step reads.",#
                 :alpha_mix => "``\\alpha``: Uniform-mix share, in ``[0, 1)``.",#
                 :v_t_mom => "``\\boldsymbol{v}_t``: Gradient average before period ``t``, zero at ``t = 1``.",#
                 :m_t_mom => "``\\boldsymbol{m}_t``: Squared-gradient average before period ``t``, the square taken per asset, zero at ``t = 1``.",#
                 :gamma1_mom => "``\\gamma_1``: Momentum decay, the weight the gradient average keeps on its past, in ``[0, 1)``.",#
                 :gamma2_mom => "``\\gamma_2``: Squared-gradient decay, the weight the squared-gradient average keeps on its past, in ``[0, 1)``.",#
                 :eps_mom => "``\\epsilon``: Positive offset added to the root of the squared-gradient average before the division.",#
                 :Sigma_t_cw => "``\\Sigma_t``: Diagonal belief covariance over the weights at period ``t``.",#
                 :M_t_cw => "``M_t = \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``: Gross return of the allocation over period ``t``.",#
                 :V_t_cw => "``V_t = \\boldsymbol{x}_t^\\intercal \\Sigma_t \\boldsymbol{x}_t``: Variance of the belief along the price relative.",#
                 :W_t_cw => "``W_t = \\boldsymbol{x}_t^\\intercal \\Sigma_t \\boldsymbol{1}``: Sum of the price relative weighted by the belief covariance.",#
                 :xbar_t_cw => "``\\bar{x}_t = W_t / \\boldsymbol{1}^\\intercal \\Sigma_t \\boldsymbol{1}``: Mean of the price relative weighted by the belief covariance.",#
                 :U_t_cw => "``U_t = \\boldsymbol{x}_t^\\intercal \\Sigma_{t+1} \\boldsymbol{x}_t``: Updated belief variance along the price relative, under the full rank-one update.",#
                 :lambda_cw => "``\\lambda_{t+1}``: Lagrange multiplier of the confidence constraint, which sets the length of the mean step.",#
                 :gamma_cw => "``\\gamma_{t+1}``: Gain, the scale of the squared price relative in the update of the inverse belief covariance.",#
                 :phi_cw => "``\\phi = \\Phi^{-1}(\\theta)``: Confidence quantile, with ``\\Phi`` the standard normal distribution function and ``\\theta`` the confidence level.",#
                 :eps_cw => "``\\epsilon``: Reversion threshold on the gross return of the period.",#
                 :w_1_start => "``\\boldsymbol{w}_1``: Start Allocation, the allocation that the recursion starts from.",#
                 :G_t_cumlog => "``\\boldsymbol{G}_t = \\sum_{s \\leq t} \\log \\boldsymbol{x}_s``: Cumulative log wealth after period ``t``, one entry for each asset, or for each expert on an expert mixture.",#
                 :M_t_hint => "``M_t``: Hint of period ``t``, the forecast of the gradient ``\\boldsymbol{g}_t`` that the Gradient Predictor makes one period earlier.",#
                 :v_t_sec => "``\\boldsymbol{v}_t``: Secondary iterate of the optimistic step at period ``t``, the allocation that the first half-step of period ``t`` starts from.",#
                 :dual_norm_geo => "``\\lVert \\cdot \\rVert_*``: Dual norm of the norm that the potential of the Projection Geometry is strongly convex against.",#
                 :xhat_fc => "``\\hat{\\boldsymbol{x}}_{t+1}``: Price Relative Forecast for period ``t + 1``, made after the row of period ``t``.",#
                 :xtilde_fc => "``\\tilde{\\boldsymbol{x}}_{t+1} = \\hat{\\boldsymbol{x}}_{t+1} - \\bar{x}_{t+1} \\boldsymbol{1}``: Centred Price Relative Forecast, with ``\\bar{x}_{t+1}`` the mean of the entries of ``\\hat{\\boldsymbol{x}}_{t+1}``. Its entries sum to zero.",#
                 :p_t_level => "``\\boldsymbol{p}_t``: Price level vector of period ``t``, one entry per asset.",#
                 :w_levels => "``w``: Price-level window, the number of price levels a statistic reads, the current one included.",#
                 :p_ti_level => "``p_{t,i}``: Price level of asset ``i`` at period ``t``, the entry ``i`` of ``\\boldsymbol{p}_t``.",#
                 :alpha_ema => "``\\alpha``: Smoothing weight on the current level, in ``(0, 1]``. A smaller weight gives a longer memory.",#
                 :s_i_trend => "``s_i``: Trend sign of asset ``i``, ``+1`` for a rising asset, ``0`` for a flat one and ``-1`` for a falling one.",#
                 # The elastic net that the kernel trend pattern fits.
                 :y_enet => "``\\boldsymbol{y}``: Response of the elastic net, one entry per observation.",#
                 :P_enet => "``\\mathbf{P}``: Regressors of the elastic net, one row per observation and one column per regressor. Column ``k`` is ``\\mathbf{P}_k``.",#
                 :z_enet => "``\\boldsymbol{z}``: Coefficients of the elastic net, one per column of ``\\mathbf{P}``. Coefficient ``k`` is ``z_k``.",#
                 :gamma_enet => "``\\gamma``: Regularisation strength of the elastic net.",#
                 :theta_enet => "``\\vartheta``: Share of the ``L_1`` penalty in the elastic net, the `theta` field of [`ElasticNetPath`](@ref).",#
                 :Sigma_hat_pred => "``\\hat{\\mathbf{\\Sigma}}_{t+1}``: Predictive covariance of the price relatives of period ``t + 1``, made after the row of period ``t``.",#
                 :Sigma_hat_RO => "``\\hat{\\mathbf{\\Sigma}}_{\\mathrm{RO}}``: Rank-one covariance estimate of a window, the matrix that [`RankOneCovariance`](@ref) computes.",#
                 # The Constrained Update: the raw step, the set it is projected onto,
                 # and the divergence of the projection.
                 :q_raw => "``\\boldsymbol{q}``: Raw step, the vector the rule's unconstrained step gives before the projection onto the Allocation Set.",#
                 :W_aset => "``\\mathcal{W}``: Allocation Set.",#
                 :Psi_pot => "``\\Psi``: Potential of the Projection Geometry.",#
                 :D_Psi_breg => "``D_\\Psi``: Bregman divergence of ``\\Psi``.",#
                 # The short-term sparse portfolio and the algorithms that find its iterate.
                 :b_sspo => "``\\boldsymbol{b}``: Unscaled target of the short-term sparse portfolio step, the vector whose scaled projection is the next allocation.",#
                 :phi_sspo => "``\\boldsymbol{\\phi}``: Objective vector of the short-term sparse portfolio step, ``-(1.1 \\log \\hat{\\boldsymbol{x}} + \\boldsymbol{1})`` for the Price Relative Forecast ``\\hat{\\boldsymbol{x}}``. Its smallest entry is the largest forecast.",#
                 :lambda_l1 => "``\\lambda``: Penalty weight of the ``L_1`` norm, positive.",#
                 :gamma_st => "``\\gamma``: Soft-threshold width, positive.",#
                 :a_huber => "``a``: Quadratic coupling ``\\lambda / \\gamma`` of the target to the auxiliary vector.",#
                 :g_aux => "``\\boldsymbol{g}``: Auxiliary vector that carries the ``L_1`` penalty in place of the target.",#
                 :nu_budget => "``\\nu``: Multiplier of the budget constraint ``\\boldsymbol{1}^\\intercal \\boldsymbol{b} = 1``.",#
                 # The expert mixtures, their dynamic-regret grids and the switching portfolio.
                 :w_hat_t_padj => "``\\hat{\\boldsymbol{w}}_t = \\boldsymbol{w}_t \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``: Price-Adjusted Allocation, the wealth held in each asset at the end of period ``t``.",#
                 :Proj_W_euclid => "``\\mathrm{Proj}_{\\mathcal{W}}``: Euclidean projection onto the Allocation Set ``\\mathcal{W}``.",#
                 :K_expert => "``K``: Expert count of an expert mixture.",#
                 :h_kt_expert => "``\\boldsymbol{h}_k(t)``: Allocation that expert ``k`` holds during period ``t``.",#
                 :r_t_expert => "``\\boldsymbol{r}_t``: Expert-return vector of period ``t``, with entries ``r_{t,k} = \\langle \\boldsymbol{h}_k(t), \\boldsymbol{x}_t \\rangle``.",#
                 :p_t_expert => "``\\boldsymbol{p}_t``: Weight vector over the experts at period ``t``, with entries ``p_{t,k}`` that sum to one.",#
                 :eta_k_grid => "``\\eta_k``: Grid rate of expert ``k``, the learning rate of its gradient step.",#
                 :eta_min_grid => "``\\eta_{\\min}``: Smallest grid rate, the rate of the first expert.",#
                 :f_t_online => "``f_t``: Online loss of period ``t``, the objective that the experts step on.",#
                 :ell_t_surr => "``\\ell_t(\\boldsymbol{w}) = \\langle \\nabla f_t(\\boldsymbol{w}_t), \\boldsymbol{w} - \\boldsymbol{w}_t \\rangle``: Surrogate loss of period ``t``, the online loss linearised at the played allocation.",#
                 :T_regret => "``T``: Regret horizon, the number of periods that a regret bound covers.",#
                 :D_diam => "``D``: Diameter of the feasible set in the Euclidean norm.",#
                 :G_gradbound => "``G``: Bound on the Euclidean norm of the gradient of every online loss over the feasible set.",#
                 :P_T_path => "``P_T = \\sum_{t=2}^{T} \\lVert \\boldsymbol{u}_t - \\boldsymbol{u}_{t-1} \\rVert_2``: Path length of a comparator sequence ``\\boldsymbol{u}_1, \\ldots, \\boldsymbol{u}_T`` in the feasible set.",#
                 # The follow-the-leader rules: the sample a leader re-solves on, and the
                 # windows a pattern-matching selector compares.
                 :Delta_N_simplex => "``\\Delta_N``: Probability simplex over the ``N`` assets.",#
                 :C_t_sample => "``C_t``: Sample of period ``t``, the past periods whose price relatives the leader re-solves on.",#
                 :w_star_lead => "``\\boldsymbol{w}^\\star_t``: Leader of period ``t``, the allocation that the held optimisation estimator solves for on the sample ``C_t``.",#
                 :w_var_lead => "``\\boldsymbol{w}``: Decision allocation of the leader's programme.",#
                 :w_win => "``w``: Window length, the number of periods in each window that a pattern-matching selector compares.",#
                 :x_win => "``\\boldsymbol{x}_{a}^{b}``: Window of periods ``a`` to ``b``, the price relatives of those periods concatenated into one vector.",#
                 # The scalar roots of the Projection Geometries on a bounded set.
                 :lu_i_aset => "``l_i``, ``u_i``: Lower and upper bounds of asset ``i`` on a [`BoundedAllocationSet`](@ref).",#
                 :theta_aset => "``\\theta``: Budget multiplier, the root of ``\\sum_i w_i(\\theta) = 1``.",#
                 :w_plus_proj => "``\\boldsymbol{w}^{+}``: Projected allocation, the answer of the projection onto the Allocation Set.",#
                 :eps_machine => "``\\varepsilon``: Machine epsilon of the element type, twice its unit roundoff ``u``.",#
                 :pi_obj_pen => "``\\pi``: Objective Penalty of the set, zero when the set has no penalty.",#
                 :h_diag_norm => "``\\boldsymbol{h}``: Positive diagonal of the norm, one entry for each asset.",#
                 # The constant rebalanced portfolio, the passive-aggressive step and the
                 # set that an expert mixture's weighting projects onto.
                 :b_crp => "``\\boldsymbol{b}``: Target of a constant rebalanced portfolio, the allocation that it rebalances to every period.",#
                 :S_t_crp => "``S_t(\\boldsymbol{b}) = \\prod_{s \\leq t} \\langle \\boldsymbol{b}, \\boldsymbol{x}_s \\rangle``: Wealth of the constant rebalanced portfolio ``\\boldsymbol{b}`` after period ``t``, from a wealth of one.",#
                 :tau_t_pa => "``\\tau_t``: Step length of the passive-aggressive update of period ``t``, non-negative.",#
                 :ell_t_pa => "``\\ell_t = \\max(0, \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle - \\epsilon)``: Hinge loss of period ``t``, the part of the gross return above the reversion threshold.",#
                 :xbar_t_pa => "``\\bar{x}_t``: Cross-sectional mean of the price relative of period ``t``, the average of its ``N`` entries.",#
                 :C_pa => "``C``: Aggressiveness of a passive-aggressive slack rule, positive.",#
                 :E_eset => "``\\mathcal{E}``: Expert Set, the Allocation Set over the experts that the weighting of an expert mixture projects onto.")
