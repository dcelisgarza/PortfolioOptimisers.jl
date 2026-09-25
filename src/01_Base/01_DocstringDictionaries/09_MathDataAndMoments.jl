# `math_dict`: data, returns and their moments, distances, feature matrices,
# denoising, norm errors, the Gerber family and the calibration rules.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(math_dict, :math_dict,
                 # Data, target and linear constraints.
                 :Xv => "``\\boldsymbol{X}``: Data vector `observations × 1`.",#
                 :tgt => "``t``: Target value, usually the unweighted (or weighted) expected value ``E[\\boldsymbol{X}]``.",#
                 :A => "``\\mathbf{A}``: Constraint coefficient matrix.",#
                 :B => "``\\boldsymbol{B}``: Constraint response vector.",#
                 :x => "``\\boldsymbol{x}``: Constrained variable.",#
                 :ineq => "``\\text{ineq}``: Subscript for inequality constraints.",#
                 :eq => "``\\text{eq}``: Subscript for equality constraints.",#
                 # Portfolio returns, dimensions and observation weights.
                 :xret => "``\\boldsymbol{x}``: Portfolio returns vector ``T \\times 1``.",#
                 :T => "``T``: Number of observations.",#
                 :x_t_obs => "``\\boldsymbol{x}_t``: Asset returns for observation ``t``, the ``t``-th row of the returns matrix.",#
                 :w_t_obs => "``w_{t}``: Observation weight of observation ``t``.",#
                 :N => "``N``: Number of assets.",#
                 :K => "``K``: Number of factors.",#
                 # Sample moments of the returns matrix.
                 :r_tj => "``r_{tj}``: Return of asset ``j`` at time ``t``.",#
                 :mu_hat_j => "``\\hat{\\mu}_j``: Estimated mean of asset ``j``.",#
                 :sigma2_hat_j => "``\\hat{\\sigma}^2_j``: Estimated variance of asset ``j``.",#
                 :sigma_hat_i => "``\\hat{\\sigma}_i``: Estimated standard deviation of asset ``i``.",#
                 :sigma_rv_hat_i => "``\\hat{\\sigma}^{\\mathrm{rv}}_i``: Predicted realised volatility of asset ``i`` for the period that follows the sample.",#
                 :Sigma_hat => "``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.",#
                 :T_shrink_target => "``\\mathbf{T}``: Covariance shrinkage target, a positive definite matrix.",#
                 :Sigma_hat_ii => "``\\hat{\\mathbf{\\Sigma}}_{ii}``: ``i``-th diagonal entry of ``\\hat{\\mathbf{\\Sigma}}``.",#
                 :Sigma_hat_ij => "``\\hat{\\mathbf{\\Sigma}}_{ij}``: Estimated covariance between assets ``i`` and ``j``.",#
                 :T_ij_shrink_target => "``T_{ij}``: Entry of ``\\mathbf{T}`` in row ``i`` and column ``j``.",#
                 :vbar_avg_var => "``\\bar{v}``: Average variance, the mean of the diagonal of ``\\hat{\\mathbf{\\Sigma}}``.",#
                 :I_identity => "``\\mathbf{I}``: Identity matrix.",#
                 # The covariance forecast evaluation. A forecast is formed at a step
                 # and judged on the returns that follow it.
                 :Sigma_hat_t => "``\\hat{\\mathbf{\\Sigma}}_t``: Covariance forecast formed at step ``t``.",#
                 :S_t_realised => "``\\mathbf{S}_t``: Realised covariance of step ``t``, formed from the centred returns that follow it.",#
                 :h_step => "``h``: Horizon of a step, the number of observations the forecast is judged on.",#
                 :M_steps => "``M``: Steps of the walk-forward, the number of forecasts a run scores.",#
                 :c_weight_bias => "``c``: Bias correction of the weighted denominator. It is fixed by the **type** of the weights, never by the estimator: `corrected = false` gives ``c = 0`` for every type, and `corrected = true` gives ``c = 1`` for `StatsBase.FrequencyWeights`, ``c = \\sum_t w_t^2 / \\sum_t w_t`` for `StatsBase.AnalyticWeights` and ``c = \\sum_t w_t / T`` for `StatsBase.ProbabilityWeights`.",#
                 # Shrinkage of the sample expected returns.
                 :mu_hat_shrink => "``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` vector of sample expected returns, whose ``i``-th entry is ``\\hat{\\mu}_i``.",#
                 :b_shrink_tgt => "``\\boldsymbol{b}``: ``N \\times 1`` shrinkage target vector, every entry of which holds the same value.",#
                 :b_j_shrink_tgt => "``b_j``: ``j``-th entry of the shrinkage target vector.",#
                 :alpha_shrink_mu => "``\\alpha``: Shrinkage intensity, the weight the blend gives the target.",#
                 # Vector reductions and elementwise operands.
                 :v_reduce => "``\\boldsymbol{v}``: The vector to reduce, of length ``n``.",#
                 :v_i_entry => "``v_{i}``: Its ``i``-th entry, ``i = 1,\\ldots,n``.",#
                 :i_linear => "``i``: Linear index, ``i = 1,\\ldots,n``.",#
                 :ab_operands => "``\\boldsymbol{a}``, ``\\boldsymbol{b}``: Array operands, read in linear index order.",#
                 :alpha_beta_scalars => "``\\alpha``, ``\\beta``: Scalar operands.",#
                 :lambda_tilde_i => "``\\tilde{\\lambda}_i``: Denoised ``i``-th eigenvalue.",#
                 :V_eigvec => "``\\mathbf{V}``: Eigenvector matrix of the input.",#
                 # Pairwise distance and correlation.
                 :d_ij_dist => "``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.",#
                 :rho_ij => "``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.",#
                 :D_mat_dist => "``\\mathbf{D}``: Distance matrix.",#
                 :D_ij_dist => "``D_{i,\\,j}``: Distance between assets ``i`` and ``j``.",#
                 :S_ij_sim => "``S_{i,\\,j}``: Similarity between assets ``i`` and ``j``.",#
                 # Feature matrices and the collapse of a feature window.
                 :z_i_feature => "``\\boldsymbol{z}_{i}``: Feature vector of asset ``i``, its row of the feature matrix.",#
                 :z_tik_feature => "``z_{t,\\,i,\\,k}``: Feature window entry: feature ``k`` of asset ``i`` at observation ``t``.",#
                 :zbar_ik_feature => "``\\bar{z}_{i,\\,k}``: Collapsed feature ``k`` of asset ``i``, the aggregate of ``z_{t,\\,i,\\,k}`` over the observation axis.",#
                 # Spectral denoising: the Marcenko-Pastur split of a spectrum.
                 :lambda_i_eig => "``\\lambda_i``: ``i``-th eigenvalue of the input matrix.",#
                 :lambda_plus_mp => "``\\lambda_+``: Marčenko-Pastur upper bound of the noise band. An eigenvalue is noise when ``\\lambda_i \\leq \\lambda_+``, and signal when ``\\lambda_i > \\lambda_+``.",#
                 :V_signal => "``\\mathbf{V}_{\\mathrm{signal}}``: Eigenvector block of the signal eigenpairs.",#
                 :lambda_vec_signal => "``\\boldsymbol{\\lambda}_{\\mathrm{signal}}``: Signal eigenvalues.",#
                 :C_signal => "``\\mathbf{C}_{\\mathrm{signal}}``: Reconstruction from the signal eigenpairs alone.",#
                 :X_denoised => "``\\tilde{\\mathbf{X}}``: Denoised matrix.",#
                 :q_mp => "``q = T/N``: Effective sample ratio, observations to assets.",#
                 :sigma2_noise => "``\\sigma^2``: Variance attributed to noise. A correlation matrix has ``\\sigma^2 = 1``.",#
                 # Norm-based error family.
                 :a_norm_err => "``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.",#
                 :b_norm_err => "``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.",#
                 :d_ddof => "``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.",#
                 :p_norm_order => "``p``: Norm order.",#
                 :tr_l2 => "``\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b})``: L2-norm error.",#
                 :tr_l2sq => "``\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b})``: Squared L2-norm error.",#
                 :tr_l1 => "``\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b})``: L1-norm error.",#
                 :tr_lp => "``\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b})``: Lp-norm error.",#
                 :tr_linf => "``\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b})``: L∞-norm error, the largest absolute deviation.",#
                 # The Gerber family. `01_GerberCovariance.jl` states the statistic,
                 # and `02_SmythBrobyCovariance.jl` and `03_GerberIQCovariance_a.jl`
                 # build on the same symbols.
                 :x_ti_ret => "``x_{t,\\,i}``: Return of asset ``i`` at observation ``t``.",#
                 :t_threshold => "``t``: Threshold parameter, read as a standalone symbol; a subscript ``t`` is the observation index. An asset crosses at an observation when its return is at least ``t`` of its own standard deviations away from zero, and a return of exactly zero never crosses.",#
                 :sigma_i_asset => "``\\sigma_i``: Standard deviation of asset ``i``.",#
                 :mu_hat_i_rank => "``\\hat{\\mu}_i``: ``i``-th entry of the characteristic vector, sorted non-increasing.",#
                 :sigma_i_ucs => "``\\sigma_i``: Per-asset scaling of the ``i``-th entry of the characteristic vector; ``1`` when the set is unscaled.",#
                 :oslash => "``\\oslash``: Element-wise division.",#
                 :U_gerber => "``\\mathbf{U} \\in \\{0,1\\}^{T \\times N}``: Up indicator matrix, ``U_{t,\\,i} = \\mathbf{1}[x_{t,\\,i} \\geq t \\, \\sigma_i \\land x_{t,\\,i} > 0]``.",#
                 :D_gerber => "``\\mathbf{D} \\in \\{0,1\\}^{T \\times N}``: Down indicator matrix, ``D_{t,\\,i} = \\mathbf{1}[x_{t,\\,i} \\leq -t \\, \\sigma_i \\land x_{t,\\,i} < 0]``.",#
                 :Nneut_gerber => "``\\mathbf{N} \\in \\{0,1\\}^{T \\times N}``: Neutral indicator matrix, ``N_{t,\\,i} = \\mathbf{1}[\\lvert x_{t,\\,i} \\rvert < t \\, \\sigma_i \\lor x_{t,\\,i} = 0]``. It is the complement of ``\\mathbf{U} + \\mathbf{D}``.",#
                 :H_gerber => "``\\mathbf{H} = \\mathbf{U} - \\mathbf{D}``: Signed crossing matrix. Its entry is ``1`` when the asset crossed upwards, ``-1`` when it crossed downwards, and ``0`` when it did not cross.",#
                 :Vcross_gerber => "``\\mathbf{V} = \\mathbf{U} + \\mathbf{D}``: Crossing matrix. Its entry is ``1`` when the asset crossed its threshold in either direction, and ``0`` when it did not.",#
                 :nc_gerber => "``n_{c}``: Concordant count of a pair, the observations on which both assets crossed their thresholds in the same direction.",#
                 :nd_gerber => "``n_{d}``: Discordant count of a pair, the observations on which both assets crossed their thresholds in opposite directions.",#
                 :nn_gerber => "``n_{n}``: Neutral count of a pair, the observations on which exactly one of the two assets crossed its threshold.",#
                 # The Smyth-Broby family. `02_SmythBrobyCovariance.jl` states the
                 # statistic, and it shares the Gerber symbols above.
                 :r_tilde_sb => "``\\tilde{r}_{t,\\,i} = (x_{t,\\,i} - \\mu_i) / \\sigma_i``: Centred, standardised return of asset ``i`` at observation ``t``.",#
                 :c1_sb => "``c_1``: Confusion-zone threshold. It is read against the **raw, uncentred** return, and it rejects an observation only when both assets fall inside it.",#
                 :c2_sb => "``c_2``: Indecision-zone threshold. It is read against the **centred, standardised** return, and it rejects an observation when both assets fall inside it. A centred return of exactly zero is inside it at every ``c_2``.",#
                 :c3_sb => "``c_3``: Outer cut-off. It is read against the centred, standardised return, and it rejects an observation when either asset exceeds it.",#
                 :kappa_sb => "``\\kappa``: Amplitude kernel of a pair, the geometric mean of the two gross standardised magnitudes.",#
                 :gamma_sb => "``\\gamma``: Divergence of a pair, the absolute difference of the two standardised magnitudes.",#
                 :n_sb => "``n``: Severity exponent. It sets how hard the divergence of a pair is penalised.",#
                 :delta_sb => "``\\delta``: Smyth-Broby contribution of one admitted observation, in place of the Gerber vote.",#
                 :CDN_sb => "``C``, ``D``, ``N``: Concordant, discordant and neutral observation sets of a pair, over the admitted observations.",#
                 :possum_sb => "``\\mathrm{pos}``, ``\\mathrm{neg}``, ``\\mathrm{nn}``: Contribution sums of a pair over ``C``, ``D`` and ``N``.",#
                 :poscount_sb => "``c^{+}``, ``c^{-}``, ``c^{0}``: Observation counts of a pair over ``C``, ``D`` and ``N``.",#
                 :pqu_sb => "``p``, ``q``, ``u``: Concordant, discordant and neutral scores of a pair, chosen from the sums and the counts by the marker prefix.",#
                 :h_ij_sb => "``h_{i,\\,j} = p - q``: Net score of the pair, before any normalisation.",#
                 # The higher comoments. `18_Coskewness.jl` and `19_Cokurtosis.jl`
                 # build both matrices from one deviation matrix and one pairwise
                 # expansion of it, so the two files share these four symbols.
                 :Y_dev => "``\\mathbf{Y}``: ``T \\times N`` deviation matrix. `FullMoment` takes the centred returns, and `SemiMoment` clips every positive entry of them to zero.",#
                 :y_t_dev => "``\\boldsymbol{y}_t``: ``N \\times 1`` deviation vector of observation ``t``, the ``t``-th row of ``\\mathbf{Y}``. Its ``i``-th entry is ``y_{t,\\,i}``.",#
                 :Z_pairprod => "``\\mathbf{Z}``: ``T \\times N^{2}`` pairwise expansion of ``\\mathbf{Y}``, whose ``t``-th row is ``\\mathbf{Z}_{t,\\cdot}`` and whose entry ``\\mathbf{Z}_{t,\\,(i-1)N+j}`` is the product ``y_{t,\\,i} \\, y_{t,\\,j}``.",#
                 :w_obs_vec => "``\\boldsymbol{w}``: ``T \\times 1`` observation weights vector.",#
                 # The ambiguity radius rules of `06_CalibrationRules.jl`, and the
                 # effective sample size the significance rules share with them. Each
                 # rule returns one radius off one record, so the radius, its scale
                 # and the weighted count of the record are each stated by two or
                 # more Units of that file.
                 :cal_r_radius => "``r``: Ambiguity radius.",#
                 :cal_s_radius => "``s``: Scale of the radius, in the units of the series the slot owner prices.",#
                 :cal_s_i_series => "``\\hat{s}_{i}``: Sample dispersion of the series the slot owner prices, over column ``i``. It is ``\\sqrt{\\hat{\\mathbf{\\Sigma}}_{ii}}`` under a [`ReturnsSeries`](@ref), and the dispersion of column ``i`` of the drawdown sample under a drawdown marker.",#
                 :cal_T_e => "``T_{e}``: Effective sample size, which is Kish's when the observation weights are stated.",#
                 # The idiosyncratic diagnostics of a cross-sectional fit. Each symbol is
                 # stated by two or more Units of that file.
                 :eps_ti_idio => "``\\varepsilon_{ti}``: Idiosyncratic return of asset ``i`` at observation ``t``.",#
                 :v_ti_idio => "``v_{ti}``: Idiosyncratic variance the fit predicted for asset ``i`` at observation ``t``.",#
                 :sigma_ti_idio => "``\\hat{\\sigma}_{ti} = \\sqrt{\\max(v_{ti}, 0)}``: Predicted idiosyncratic volatility of asset ``i`` at observation ``t``. A negative variance counts as zero.",#
                 :z_ti_idio => "``z_{ti}``: Standardised idiosyncratic return of asset ``i`` at observation ``t``.",#
                 :F_t_idio => "``\\mathcal{F}_{t}``: Finite cross-section of observation ``t``, the assets whose standardised return ``z_{ti}`` is finite.",#
                 :n_t_idio => "``n_{t}``: Count of the assets in ``\\mathcal{F}_{t}``.",#
                 :m_pt_idio => "``m_{pt}``: Central moment of order ``p`` of the finite standardised returns of observation ``t``.",#
                 :a_t_series => "``a_{t}``: Diagnostic series value at observation ``t``.",#
                 :A_series_fin => "``\\mathcal{A}``: Finite observations of a diagnostic series, those at which ``a_{t}`` is finite.",#
                 :rho_S_cs => "``\\rho_{S}``: Cross-sectional rank correlation, over the assets at which both cross-sections are finite. It correlates the ordinal ranks, so two equal values take two ranks, in the order of the asset axis.",#
                 # The coverage policy of `01_Base/16_CoveragePolicy.jl`. Each symbol is
                 # stated by two or more Units of that file.
                 :s_i_cvg => "``s_{i}``: Coverage share of asset ``i``, its own observation count over the number of observations folded.",#
                 :c_cvg => "``c``: Coverage floor of the policy, its `min_coverage`.",#
                 :tau_i_cvg => "``\\tau_{i}``: Staleness of asset ``i``, the number of observations folded since the asset was last finite and active.",#
                 :a_i_cvg => "``a_{i}``: Active flag of asset ``i`` at the last observation folded, ``1`` when the asset is active and ``0`` when it is not.",#
                 :nu_k_cvg => "``\\nu_{k}``: Observation count of cell ``k``, the number of observations at which every asset of the cell is finite and active.",#
                 :v_k_cvg => "``v_{k}``: Entry ``k`` of the answer of an available-case fit.",#
                 :A_adm_cvg => "``\\mathcal{A}``: Admitted assets, those that [`admits`](@ref) lets into the answer. A cell is admitted when every asset of the cell is in ``\\mathcal{A}``.",#
                 # The Listing Span of `03_InputData/03_ListingSpan.jl` and its projection
                 # onto the returns clock. Each symbol is stated by two or more Units of
                 # that file.
                 :f_i_span => "``f_{i}``: First observation of the Listing Span of asset ``i``, on the price clock.",#
                 :l_i_span => "``l_{i}``: Last observation of the Listing Span of asset ``i``, on the price clock.",#
                 :a_ti_span => "``a_{t,\\,i}``: Listing indicator of asset ``i`` at observation ``t`` of the price clock, ``1`` when the asset is listed and ``0`` when it is not.",#
                 :a_tilde_ti_act => "``\\tilde{a}_{t,\\,i}``: Active mask entry of asset ``i`` at observation ``t`` of the returns clock, ``1`` when both prices of the return lie inside the listing of the asset.",#
                 :n_span => "``n``: Price clock length, the number of price observations.",#
                 :m_span => "``m``: Returns clock length, the number of return observations.",#
                 :o_span => "``o = n - m``: Clock offset, ``0`` when padding keeps the first observation and ``1`` when it does not.",#
                 # The price gap fill of `03_InputData/08_PriceGapFill.jl`. Each symbol is
                 # stated by two or more Units of that file.
                 :p_ti_price => "``p_{t,\\,i}``: Price of asset ``i`` at observation ``t`` of the price clock, absent at a gap.",#
                 :p_tilde_ti_fill => "``\\tilde{p}_{t,\\,i}``: Filled price of asset ``i`` at observation ``t``, the price after the fill.",#
                 :O_i_fill => "``\\mathcal{O}_{i}``: Observed set of asset ``i``, the observations at which its price is neither `missing` nor `NaN`.",#
                 :v_i_fill => "``v_{i}``: Fitted value of asset ``i``, read off its observed prices over the training window.",#
                 :o_i_fill => "``o_{i}``: Open flag of asset ``i``, ``1`` when no absence from the listing follows its last observed price in the window, so that the price can seed the next window.",#
                 :phi_fill => "``\\phi``: Reduction that a [`Num_VecToScaM`](@ref) states, from the observed prices of an asset to one number. A plain number is the constant reduction.",#
                 :t0_fill => "``t_{0}``: Replay start, the first observation of the window after the end of the training window, or ``n + 1`` when the window holds none.",#
                 :r_ti_run => "``r_{t,\\,i}``: Run start of asset ``i`` at observation ``t``, the first observation of the unbroken listed run that holds ``t``.",#
                 :s_ti_carry => "``s_{t,\\,i}``: Carry source of asset ``i`` at observation ``t``, the last member of ``\\mathcal{O}_{i}`` before ``t`` in the listed run that holds ``t``.",#
                 # The evaluation of a Return Forecast under
                 # `32_CrossSectionalFactorModel/07_ReturnForecasts/`. Each symbol is
                 # stated by two or more Units of those files.
                 :alpha_ti_fc => "``\\alpha_{ti}``: Return Forecast of asset ``i`` at observation ``t``.",#
                 :y_ti_fwd => "``y_{ti}``: Forward target of asset ``i`` at observation ``t``.",#
                 :m_ti_univ => "``m_{ti}``: Universe mask of asset ``i`` at observation ``t``, `true` where the asset is in the estimation universe.",#
                 :u_ti_cs => "``u_{ti}``: Cross-sectional weight of asset ``i`` at observation ``t``.",#
                 :U_t_univ => "``\\mathcal{U}_{t}``: Universe of observation ``t``, the assets in the mask that carry a finite positive weight.",#
                 :t_j_eval => "``t_{j}``: The ``j``-th evaluation date.",#
                 :s_eval_stride => "``s``: The stride between two evaluation dates.",#
                 :c_j_cov => "``c_{j}``: Coverage at evaluation date ``t_{j}``, the share of the universe the evaluation scored there.",#
                 :n_j_scored => "``n_{j}``: Scored count at evaluation date ``t_{j}``, the assets of the universe that carry a finite forecast and a finite target there.",#
                 :a_k_pair => "``a_{k}``: Forecast of the ``k``-th scorable pair.",#
                 :b_k_pair => "``b_{k}``: Forward target of the ``k``-th scorable pair.",#
                 :q_k_pair => "``q_{k}``: Cross-sectional weight of the ``k``-th scorable pair, zero where that weight is not finite.",#
                 :n_pool => "``n``: Number of scorable pairs in the pooling.",#
                 :V_t_cs => "``\\mathcal{V}_{t}``: Cross-section of observation ``t``, the assets whose forecast and target are both finite there.",#
                 # The cross-sectional correlations of `32_CrossSectionalFactorModel/`.
                 # Each symbol is stated by two or more Units of those files.
                 :alpha_t_fc => "``\\boldsymbol{\\alpha}_{t}``: Return Forecast cross-section at observation ``t``.",#
                 :B_tk_cs => "``\\mathbf{B}_{t \\cdot k}``: Exposure cross-section of factor ``k`` at observation ``t``.",#
                 :u_t_cs => "``\\boldsymbol{u}_{t}``: Cross-sectional weights of observation ``t``.",#
                 :rho_rank_cs => "``\\rho^{\\mathrm{S}}``: The correlation of the ordinal ranks of two cross-sections, over the assets at which both values are finite.",#
                 :rho_w_cs => "``\\rho``: The weighted correlation of two cross-sections, over the assets at which both values are finite and the weight is finite and positive.")
