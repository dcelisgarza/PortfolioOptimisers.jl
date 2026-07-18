"""
    arg_dict = Dict(
                 # Weight vectors.
                 :pw => "`w`: Portfolio weights vector.",
                 :ow => "`w`: Observation weights vector.",
                 :oow => "`w`: Optional observation weights vector.",
                 # Matrix processing.
                 :pdm => "`pdm`: Positive definite matrix estimator.",
                 :dn => "`dn`: Matrix denoising estimator.",
                 :dt => "`dt`: Matrix detoning estimator.",
                 :mp => "`mp`: Matrix processing estimator.",
                 # Moments.
                 :me => "`me`: Expected returns estimator.",
                 :ce => "`ce`: Covariance estimator.",
                 :ve => "`ve`: Variance estimator.",
                 :ske => "`ske`: Coskewness estimator.",
                 :kte => "`kte`: Cokurtosis estimator.",
                 :de => "`de`: Distance matrix estimator.",
                 # Priors.
                 :pe => "`pe`: Prior estimator.",
                 :pr => "`pr`: Prior result.",
                 :per => "`pe`: Prior estimator or result.",
                 # Phylogeny.
                 :cle => "`cle`: Clusters estimator.",
                 :clr => "`clr`: Clusters result.",
                 :cler => "`cle`: Clusters estimator or result.",
                 :ple => "`pl`: Phylogeny estimator.",
                 :plr => "`pl`: Phylogeny result.",
                 :pler => "`pl`: Phylogeny estimator or result.",
                 :nte => "`pl`: Network estimator.",
                 :ntr => "`pl`: Network result.",
                 :nter => "`pl`: Network estimator or result.",
                 :cte => "`cte`: Centrality estimator.",
                 :cta => "`ct`: Centrality algorithm.",
                 :ctr => "`ct`: Centrality result.",
                 :cter => "`ct`: Centrality estimator or result.",
                 # Turnover.
                 :tne => "`tn`: Turnover estimator.",
                 :tnr => "`tn`: Turnover result.",
                 :tner => "`tn`: Turnover estimator or result.",
                 :tnes => "`tn`: Turnover estimator(s).",
                 :tnrs => "`tn`: Turnover result(s).",
                 :tners => "`tn`: Turnover estimator(s) or result(s).",
                 # Tracking.
                 :tre => "`tr`: Tracking error estimator.",
                 :trr => "`tr`: Tracking error result.",
                 :trer => "`tr`: Tracking error estimator or result.",
                 :tres => "`tr`: Tracking error estimator(s).",
                 :trrs => "`tr`: Tracking error result(s).",
                 :trers => "`tr`: Tracking error estimator(s) or result(s).",
                 # Weight bounds.
                 :wbe => "`wb`: Weight bounds estimator.",
                 :wbr => "`wb`: Weight bounds result.",
                 :wber => "`wb`: Weight bounds estimator or result.",
                 :wb => "`wb`: Weight bounds.",
                 # Fees.
                 :feese => "`fees`: Fees estimator.",
                 :feesr => "`fees`: Fees result.",
                 :feeser => "`fees`: Fees estimator or result.",
                 :fees => "`fees`: Fees estimator or result.",
                 # Optimiser config.
                 :opt => "`opt`: `JuMP` optimiser configuration.",
                 :kwargs => "`kwargs`: Additional keyword arguments.",
                 # Index.
                 :idx => "`idx`: Index vector.",
                 # Risk measure.
                 :r => "`r`: Risk measure or vector of risk measures.",
                 # Returns estimator.
                 :ret => "`ret`: Returns estimator for `JuMP` models.",
                 # Turnover constraint.
                 :tn => "`tn`: Turnover constraint estimator.",
                 # Tracking constraint.
                 :tr => "`tr`: Tracking error constraint estimator.",
                 # Near optimal centering result fields.
                 :w_opt => "`w_opt`: Optimal portfolio weights.",
                 :w_max => "`w_max`: Maximum-risk portfolio weights.",
                 :w_min => "`w_min`: Minimum-risk portfolio weights.",
                 :w_opt_ini => "`w_opt_ini`: Initial weights for the optimal sub-problem.",
                 :w_max_ini => "`w_max_ini`: Initial weights for the maximum-risk sub-problem.",
                 :w_min_ini => "`w_min_ini`: Initial weights for the minimum-risk sub-problem.",
                 :w_opt_retcode => "`w_opt_retcode`: Return code for the optimal-objective sub-problem.",
                 :w_max_retcode => "`w_max_retcode`: Return code for the maximum-risk sub-problem.",
                 :w_min_retcode => "`w_min_retcode`: Return code for the minimum-risk sub-problem.",
                 :rt_opt => "`rt_opt`: Optimal return target.",
                 :rt_max => "`rt_max`: Maximum return target.",
                 :rt_min => "`rt_min`: Minimum return target.",
                 :rk_opt => "`rk_opt`: Optimal risk target.",
                 :noc_retcode => "`noc_retcode`: Return code for the near-optimal centering sub-problem.",
                 # Discrete allocation result fields.
                 :l_model => "`l_model`: `JuMP` model for the long allocation.",
                 :s_model => "`s_model`: `JuMP` model for the short allocation.",
                 :l_retcode => "`l_retcode`: Return code for the long allocation sub-problem.",
                 :s_retcode => "`s_retcode`: Return code for the short allocation sub-problem.",
                 # Risk budgeting.
                 :prb => "`prb`: Processed risk budgeting configuration.",
                 :sq => "`sq`: Whether to use variance instead of volatility in the inverse weighting.",
                 :wfalg => "`alg`: Weight finaliser error formulation algorithm.",
                 :res_retcode => "`res`: Optional result or message from the solver.",
                 :N_msc => "`N`: Number of bisection steps for the monotonic Schur complement.",
                 :alpha_dirichlet => "`alpha`: Dirichlet concentration parameter.",
                 :opt_hier => "`opt`: Base hierarchical optimiser configuration.",
                 :strict_opt => "`strict`: Whether to strictly enforce weight bounds.",
                 :strict_conv => "`strict`: Whether to raise an error if convergence is not achieved.",
                 :schalg => "`alg`: Schur complement algorithm variant.")

This dictionary contains the arg_dict terms and their corresponding descriptions used in the documentation of `PortfolioOptimisers.jl`.
"""
const arg_dict = Dict(
                      # Weight vectors.
                      :pw => "`w`: Portfolio weights vector `assets × 1`.",#
                      :ow => "`w`: Observation weights vector `observations × 1`.",#
                      :oow => "`w`: Optional observation weights vector `observations × 1`, or a concrete subtype of [`DynamicAbstractWeights`](@ref). If `nothing`, the computation is unweighted.",#
                      :eqw => "`eqw`: Equilibrium weights vector `features × 1`.",#
                      # Matrix processing.
                      :pdm => "`pdm`: Positive definite matrix estimator.",
                      :opdm => "`pdm`: Optional positive definite matrix estimator.",
                      :dn => "`dn`: Matrix denoising estimator.",
                      :odn => "`dn`: Optional matrix denoising estimator.",
                      :dna => "`dna`: Matrix denoising algorithm.",
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
                      :oidx => "`oidx`: Optional indices of the observations to use for estimation `Y × 1` where `Y <= observations`. If `nothing`, all observations are used.",
                      :malg => "`alg`: Moment algorithm.",
                      :corrected => "`corrected`: Whether to apply Bessel's correction.",#
                      :mutgt => "`tgt`: Shrinkage target.",#
                      :me_shrink_alg => "`alg`: Expected returns shrinkage algorithm.",#
                      :metric => "`metric`: Distance metric used for pairwise computations.",#
                      :metric_args => "`args`: Additional positional arguments for the distance metric.",#
                      :metric_kwargs => "`kwargs`: Additional keyword arguments for the distance metric.",#
                      :t => "`t`: Threshold value.",#
                      :iv => "`iv`: Implied volatility matrix.",
                      :oiv => "`iv`: Optional implied volatility matrix. Used if any internal covariance estimator is an instance of [`ImpliedVolatility`](@ref).",#
                      ## Regression
                      :M => "`M`: Main coefficient (loadings) matrix `assets × factors`.",#
                      :L => "`L`: Reduced dimensionsionality coefficient (loadings) matrix `assets × reduced_dimensions`.",#
                      :b => "`b`: Regression intercept vector.",#
                      :crit => "`crit`: Feature selection criterion.",#
                      :realg => "`alg`: Regression algorithm.",#
                      :retgt => "`tgt`: Regression model target.",#
                      :dretgt => "`retgt`: Regression model target.",#
                      :drtgt => "`drtgt`: Dimension reduction target.",
                      ## Gerber
                      :gerbalg => "`alg`: Gerber covariance algorithm.",#
                      :gerbce => "`ce`: Gerber covariance estimator.",#
                      :stdarr => "`sd`: Standard deviation vector of `X`, shaped to be consistent with `X`.",#
                      :c1 => "`c1`: Zone of confusion parameter.",#
                      :c2 => "`c2`: Zone of indecision lower bound.",#
                      :c3 => "`c3`: Zone of indecision upper bound.",#
                      :sbn => "`n`: Exponent parameter for the Smyth-Broby kernel.",#
                      :sbalg => "`alg`: Smyth-Broby covariance algorithm.",#
                      ## Mutual and var info
                      :bins => "`bins`: Binning algorithm or fixed number of bins.",#
                      :normalise => "`normalise`: Whether to normalise the mutual and/or variation of information calculation.",#
                      ## Distance
                      :dopower => "`power`: Optional matrix exponent.",#
                      :dalg => "`alg`: Distance algorithm.",#
                      :dmetric => "`metric`: Distance metric used for the distances of distances computations.",#
                      :dmetric_args => "`args`: Additional positional arguments for the distances of distances metric.",#
                      :dmetric_kwargs => "`kwargs`: Additional keyword arguments for the distances of distances metric.",#
                      # Priors.
                      :pe  => "`pe`: Prior estimator.",#
                      :pr  => "`pr`: Prior result.",#
                      :per => "`pr`: Prior estimator or result.",#
                      # Phylogeny.
                      :cle => "`cle`: Clusters estimator.",#
                      :clr => "`clr`: Clusters result.",#
                      :cler => "`clr`: Clusters estimator or result.",#
                      :ple => "`ple`: Phylogeny estimator.",#
                      :plr => "`plr`: Phylogeny result.",#
                      :pler => "`pl`: Phylogeny estimator or result.",#
                      :nte => "`nte`: Network estimator.",#
                      :ntr => "`pl`: Network result.",#
                      :nter => "`pl`: Network estimator or result.",#
                      :cte => "`cte`: Centrality estimator.",#
                      :cta => "`ct`: Centrality algorithm.",#
                      :ctr => "`ct`: Centrality result.",#
                      :cter => "`ct`: Centrality estimator or result.",#
                      :ctargs => "`args`: Positional arguments for the centrality function.",#
                      :ctkwargs => "`kwargs`: Keyword arguments for the centrality function.",#
                      :treeargs => "`args`: Positional arguments for the centrality function.",#
                      :treekwargs => "`kwargs`: Keyword arguments for the centrality function.",#
                      :ntalg => "`alg`: Tree or similarity matrix algorithm.",#
                      :ntn => "`n`: Number of steps to take in the network for deciding adjacency.",#
                      :clres => "`res`: Clustering result.",#
                      :S => "`S`: Similarity matrix",#
                      :D => "`D`: Distance matrix",#
                      :ck => "`k`: Optimal number of clusters.",#
                      :vsalg => "`alg`: The measure used to evaluate clustering quality.",#
                      :max_k => "`max_k`: Maximum number of clusters to consider. If `nothing`, computed as the `floor(Int, sqrt(features))`.",#
                      :kalg => "`alg`: Algorithm for selecting the optimal number of clusters. If an integer, defines the number of clusters directly.",#
                      :clalg => "`alg`: Clustering algorithm.",#
                      :onc => "`onc`: Optimal number of clusters estimator.",#
                      :phX_Xv => "`X`: Phylogeny matrix or vector.",#
                      :phX => "`X`: Phylogeny matrix.",#
                      :pler => "`pl`: Network estimator, phylogeny result, clustering estimator, or clustering result.",#
                      ## DBHT
                      :dbhtpower => "`power`: Exponent for the the distance matrix when computing the similarity matrix.",#
                      :dbhtcoef => "`coef`: Coefficient for the the distance matrix when computing the similarity matrix.",#
                      :sim => "`sim`: Similarity matrix algorithm.",#
                      :root => "`root`: Root selection method.",#
                      # Estimators
                      :sets => "`sets`: Sets used to map estimator values to features.",#
                      :val => "`val`: Default value to use for the estimator. If `nothing`, the estimator provides the default value.",#
                      :ekey => "`key`: Key to specify the asset universe in `sets.dict`. If `nothing`, the key is taken from `sets.key`.",#
                      :datatype => "`datatype`: Data type to use for the result in case `val` is `nothing`.",#
                      :strict => "`strict`: Whether to throw an error if `sets` does not contain the desired value in `sets.dict[key]`.",#
                      # Constraints
                      :A => "`A`: Linear constraint coefficient matrix.",#
                      :B => "`B`: Linear constraint response vector.",#
                      :eq => "`eq`: Optional equality constraints.",#
                      :ineq => "`ineq`: Optional inequality constraints.",#
                      # Turnover.
                      :tne => "`tn`: Turnover estimator.",#
                      :tnr => "`tn`: Turnover result.",
                      :tner => "`tn`: Turnover estimator or result.",
                      :tnes => "`tn`: Turnover estimator(s).",
                      :tnrs => "`tn`: Turnover result(s).",
                      :tners => "`tn`: Turnover estimator(s) or result(s).",
                      # Tracking.
                      :tre => "`tr`: Tracking error estimator.",
                      :trr => "`tr`: Tracking error result.",
                      :trer => "`tr`: Tracking error estimator or result.",
                      :tres => "`tr`: Tracking error estimator(s).",
                      :trrs => "`tr`: Tracking error result(s).",
                      :trers => "`tr`: Tracking error estimator(s) or result(s).",
                      # Weight bounds.
                      :wbe => "`wb`: Weight bounds estimator.",
                      :wbr => "`wb`: Weight bounds result.",
                      :wber => "`wb`: Weight bounds estimator or result.",
                      # Fees.
                      :feese => "`fees`: Fees estimator.",#
                      :feesr => "`fees`: Fees result.",
                      :feeser => "`fees`: Fees estimator or result.",
                      # Stats.
                      :sigma => "`sigma`: Covariance matrix `features × features`.",#
                      :mu => "`mu`: Expected returns vector `features × 1`.",#
                      :rho => "`rho`: Correlation matrix `features × features`.",
                      :sigrho => "`sigma`: Covariance-like or correlation-like matrix `features × features`.",
                      :sigrhoX => "`X`: Covariance-like or correlation-like matrix `features × features`.",
                      :kt => "`kt`: Cokurtosis matrix `features^2 × features^2`.",#
                      :sk => "`sk`: Coskewness matrix `features × features^2`.",#
                      :V => "`V`: Sum of the negative spectral slices of the coskewness matrix `features × features`.",
                      :X => "`X`: Data matrix `observations × features` if the `dims` keyword does not exist or `dims = 1`, `features × observations` when `dims = 2`.",#
                      :F => "`F`: Data matrix `observations × factors` if the `dims` keyword does not exist or `dims = 1`, `factors × observations` when `dims = 2`.",#
                      :Xv => "`X`: Data vector `observations × 1`.",#
                      :X_Xv => "`X`: Data matrix or vector.",#
                      :dims => "`dims`: Dimension along which to perform the computation.",#
                      :omean => "`mean`: Optional mean value to use for centering.",
                      :stdvec => "`sd`: Vector of standard deviations for each asset.",#
                      :ex => "`ex`: Parallel execution strategy.",#
                      :alpha => "`alpha`: Quantile level for the lower tail.",#
                      :beta => "`beta`: Quantile level for the upper tail.",#
                      :l => "`l`: Risk aversion parameter.",#
                      :rf => "`rf`: Risk-free rate.",#
                      # Errors
                      :msg => "`msg`: Error message describing the condition that triggered the exception.",#
                      # Solver
                      :name => "`name`: Symbol or string identifier for logging purposes.",#
                      :solver => "`solver`: The `optimizer_factory` in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).",#
                      :settings => "`settings`: Optional solver-specific settings used in [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute).",#
                      :check_sol => "`check_sol`: Named tuple of solution for keyword arguments in [`assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.assert_is_solved_and_feasible).",#
                      :add_bridges => "`add_bridges`: The `add_bridges` keyword argument in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).",#
                      # RNG
                      :rng => "`rng`: Random number generator.",#
                      :seed => "`seed`: Seed for the random number generator.",
                      # JuMP Optimisation
                      :model => "`model::JuMP.Model`: The JuMP optimisation model.",
                      :opt_rjumpe => "`opt::RiskJuMPOptimisationEstimator`: Risk-based optimisation estimator.",
                      :opt_jumpe => "`opt::JuMPOptimisationEstimator`: JuMP optimisation estimator.",
                      :ci => "`i`: Constraint index for unique variable and constraint naming.",
                      :key_sym => "`key::Symbol`: Symbol used to name constraints or expressions in the model.",
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
                      :optargs => "`args`: Additional positional arguments passed to the optimisation function.",
                      :optkwargs => "`kwargs`: Additional keyword arguments passed to the optimisation function.",
                      :ignargs => "`args`: Additional positional arguments (ignored).",
                      :ignkwargs => "`kwargs`: Additional keyword arguments (ignored).",
                      :rd => "`rd`: The returns result to use.",
                      :window => "`window`: Observation window.",
                      # Prior results.
                      :chol => "`chol`: Cholesky factorisation of the covariance matrix.",#
                      :w_prior => "`w`: Portfolio weights vector used in prior computation.",#
                      :ens => "`ens`: Effective sample size.",#
                      :kld => "`kld`: Kullback-Leibler divergence.",#
                      :rr => "`rr`: Returns result.",#
                      :f_mu => "`f_mu`: Factor expected returns vector.",#
                      :f_sigma => "`f_sigma`: Factor covariance matrix.",#
                      :f_w => "`f_w`: Factor weights vector.",#
                      :op_w => "`ow`: Opinion pooling weights.",#
                      :reg_rr => "`rr`: Regression result.",#
                      # Prior estimators.
                      :horizon => "`horizon`: Optional investment horizon for log-normalising returns. If `nothing`, returns are not adjusted.",#
                      :tau => "`tau`: Blending parameter controlling the weight given to the prior relative to the views.",#
                      :views => "`views`: Views estimator or result.",#
                      :views_conf => "`views_conf`: Views confidence estimator or result.",#
                      :a_pe => "`a_pe`: Asset prior estimator.",#
                      :f_pe => "`f_pe`: Factor prior estimator.",#
                      :a_views => "`a_views`: Asset views estimator or result.",#
                      :f_views => "`f_views`: Factor views estimator or result.",#
                      :a_sets => "`a_sets`: Asset sets.",#
                      :f_sets => "`f_sets`: Factor sets.",#
                      :a_views_conf => "`a_views_conf`: Asset views confidence estimator or result.",#
                      :f_views_conf => "`f_views_conf`: Factor views confidence estimator or result.",#
                      :rsd => "`rsd`: Whether to include residual variance in the posterior covariance.",#
                      :f_mp => "`f_mp`: Factor matrix processing estimator.",#
                      :re => "`re`: Regression estimator.",#
                      :pes => "`pes`: Vector of prior estimators.",#
                      :pe1 => "`pe1`: Pre-processing prior estimator.",#
                      :pe2 => "`pe2`: Post-processing prior estimator.",#
                      :p_pool => "`p`: Opinion pooling blending parameter.",#
                      # Entropy pooling.
                      :mu_views => "`mu_views`: Expected returns views estimator or result.",#
                      :var_views => "`var_views`: Value-at-risk views estimator or result.",#
                      :cvar_views => "`cvar_views`: Conditional value-at-risk views estimator or result.",#
                      :sigma_views => "`sigma_views`: Variance views estimator or result.",#
                      :sk_views => "`sk_views`: Skewness views estimator or result.",#
                      :kt_views => "`kt_views`: Kurtosis views estimator or result.",#
                      :cov_views => "`cov_views`: Covariance views estimator or result.",#
                      :rho_views => "`rho_views`: Correlation views estimator or result.",#
                      :var_alpha => "`var_alpha`: Quantile level for variance views.",#
                      :cvar_alpha => "`cvar_alpha`: Quantile level for conditional value-at-risk views.",#
                      :ds_opt => "`ds_opt`: Thin wrapper for arguments and keyword arguments used in `Roots.findzero` for use with a single conditional value-at-risk view.",#
                      :dm_opt => "`dm_opt`: Optimiser for multiple conditional value at risk views.",#
                      :opt_ep => "`opt`: Entropy pooling optimisation estimator.",#
                      # Black-Litterman views.
                      :P => "`P`: Views loading matrix `views × assets`.",#
                      :Q => "`Q`: Views values vector `views × 1`.",#
                      :excl => "`excl`: Indices of views to exclude.",#
                      # High order priors.
                      :f_kt => "`f_kt`: Factor cokurtosis matrix.",#
                      :f_sk => "`f_sk`: Factor coskewness matrix.",#
                      :f_V => "`f_V`: Factor sum of negative spectral slices of the coskewness matrix.",#
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
                      :method_ucs => "`method`: Ellipsoidal uncertainty set estimation method.",#
                      :diagonal => "`diagonal`: Whether to use only the diagonal of the covariance matrix.",#
                      :eps_ucs => "`eps`: Radius of the ``\\\\ell_1`` uncertainty set on the characteristic vector. Larger values admit more estimation error, and therefore activate more assets.",#
                      :ep_ucs => "`ep`: Radius of the positive-error side of the signed ``\\\\ell_1`` uncertainty set.",#
                      :em_ucs => "`em`: Radius of the negative-error side of the signed ``\\\\ell_1`` uncertainty set.",#
                      :sd_ucs => "`sd`: Per-asset scaling vector for the ``\\\\ell_1`` uncertainty set (the estimated standard deviations). `nothing` leaves the set unscaled, so every element of the characteristic vector is assumed to suffer the same estimation error.",#
                      :method_l1_ucs => "`method`: Radius of the ``\\\\ell_1`` uncertainty set. A number is the radius itself; an [`AbstractUncertaintyEpsAlgorithm`](@ref) computes it from the data.",#
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
                      # Constraint generation.
                      :dval => "`dval`: Default value for assets not specified in `val`.",#
                      :ukey => "`ukey`: Universe key identifying the full set of assets.",#
                      :dict => "`dict`: Dictionary mapping group identifiers to asset labels.",#
                      :vars => "`vars`: Variable names in the parsed constraint expression.",#
                      :coef_c => "`coef`: Coefficients corresponding to the constraint variables.",#
                      :op => "`op`: Comparison operator (`==`, `<=`, or `>=`).",#
                      :rhs => "`rhs`: Right-hand side value of the constraint.",#
                      :eqn => "`eqn`: Formatted string representation of the constraint equation.",#
                      :ij => "`ij`: Pair of asset indices for correlation-based constraints.",#
                      :comp => "`comp`: Constraint comparison type.",#
                      :scale_c => "`scale`: Scaling factor applied to constraint coefficients.",#
                      # Risk measure settings.
                      :settings_rm => "`settings`: Risk measure settings.",#
                      :scale_rm => "`scale`: Scaling factor applied to the risk measure.",#
                      :ub_rms => "`ub`: Upper bound(s) for the risk measure. Can be a scalar, vector, or [`Frontier`](@ref).",#
                      :lb_rms => "`lb`: Lower bound(s) for the risk measure. Can be a scalar, vector, or [`Frontier`](@ref).",#
                      :rke => "`rke`: Whether to include the risk measure value in the `JuMP` risk expression.",#
                      # Frontier.
                      :N_fr => "`N`: Number of points on the efficient frontier.",#
                      :factor_fr => "`factor`: Scaling factor for the efficient frontier range.",#
                      :bound_fr => "`bound`: What operation needs to be performed on the risk lower bound.",#
                      # Risk measure fields.
                      :rc => "`rc`: Risk contribution constraint.",#
                      :alg => "`alg`: Risk measure optimisation formulation algorithm.",#
                      :vr_rm => "`vr`: Variance risk measure component.",#
                      :sk_rm => "`sk`: Skewness risk measure component.",#
                      :kt_rm => "`kt`: Kurtosis risk measure component.",#
                      :alg1 => "`alg1`: First algorithm variant.",#
                      :alg2 => "`alg2`: Second algorithm variant.",#
                      :N_kt => "`N`: Optional number of eigenvalues per asset for the approximate cokurtosis formulation.",#
                      :kappa => "`kappa`: Relativistic deformation parameter.",#
                      :kappa_a => "`kappa_a`: Relativistic deformation parameter for the lower tail.",#
                      :kappa_b => "`kappa_b`: Relativistic deformation parameter for the upper tail.",#
                      :l_a => "`l_a`: Risk aversion parameter for the lower tail.",#
                      :r_a => "`r_a`: Radius parameter for the lower tail.",#
                      :l_b => "`l_b`: Risk aversion parameter for the upper tail.",#
                      :r_b => "`r_b`: Radius parameter for the upper tail.",#
                      :gamma => "`gamma`: Log-sum-exp scalariser smoothing parameter.",#
                      :b_mip => "`b`: Big-M upper bound for MIP formulations.",#
                      :s_mip => "`s`: Small-M lower bound for MIP formulations.",#
                      :slv => "`slv`: Solver or vector of solvers.",#
                      :p_rm => "`p`: Power or order parameter.",#
                      :mu_rm => "`mu`: Optional mean for centering.",#
                      :w_rm => "`w`: Optional portfolio weights.",#
                      :ddof => "`ddof`: Degrees-of-freedom correction.",#
                      :flag => "`flag`: Algorithm selection flag.",#
                      :pos => "`pos`: Whether to consider only positive deviations.",#
                      # Turnover.
                      :w_tn => "`w`: Current portfolio weights vector.",#
                      :w_ref => "`w`: Reference portfolio weights vector.",#
                      :w_bm_ret => "`w`: Benchmark portfolio returns vector.",#
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
                      :dl => "`dl`: Default long proportional fee.",#
                      :ds => "`ds`: Default short proportional fee.",#
                      :dfl => "`dfl`: Default long fixed fee.",#
                      :dfs => "`dfs`: Default short fixed fee.",#
                      :kwargs_fee => "`kwargs`: Named tuple of keyword arguments for fee computation.",#
                      # Optimisation results.
                      :pa => "`pa`: Processed optimisation attributes.",#
                      :retcode => "`retcode`: Optimisation return code.",#
                      :sol => "`sol`: Optimisation solution.",#
                      :fb => "`fb`: Fallback result or estimator.",#
                      # Optimiser fields.
                      :opt_jmp => "`opt`: `JuMP` optimiser configuration.",#
                      :r_opt => "`r`: Risk measure or vector of risk measures.",#
                      :obj => "`obj`: Portfolio objective function.",#
                      :wi => "`wi`: Initial portfolio weights for warm-starting the solver.",#
                      :sca => "`sca`: Scalariser for combining multiple risk measures.",#
                      :wb_jmp => "`wb`: Weight bounds estimator or weight bounds.",#
                      :bgt => "`bgt`: Net budget, `1ᵀw`. A number pins it, a [`BudgetRange`](@ref) bounds it. By default budgets *bound* the realised exposure rather than pinning it (see `xbgt`). Together with `sbgt` this fixes the net and gross exposures only jointly; to constrain the gross exposure on its own see `gbgt`.",#
                      :sbgt => "`sbgt`: Short-side budget, `sum(sw)`. A number pins it, a [`BudgetRange`](@ref) bounds it; by default it *bounds*, so `sbgt = 0.3` means *at most* 30% short unless `xbgt` pins the long/short decomposition. Together with `bgt` this fixes the net and gross exposures only jointly; to constrain the gross exposure on its own see `gbgt`.",#
                      :gbgt => "`gbgt`: Gross budget (leverage) constraint, `sum(lw) + sum(sw)`. A number pins the gross exposure; a [`BudgetRange`](@ref) bounds it, e.g. `BudgetRange(; lb = nothing, ub = 2.0)` caps leverage at 2x. Unlike `bgt` and `sbgt` — which pin the net and gross exposures only *together* — this constrains the gross exposure on its own, leaving the net free. Requires weight bounds that admit short positions, and is bounded rather than pinned unless `xbgt` is set.",#
                      :xbgt => "`xbgt`: Whether to pin the long/short decomposition exactly. When `false` (the default), `lw` and `sw` are upper bounds on the positive and negative parts of `w`, so `bgt`, `sbgt` and `gbgt` bound the realised exposures rather than pinning them — a short budget of `0.3` means *at most* 30% short. When `true`, the long/short binary indicators force `lw == max(w, 0)` and `sw == max(-w, 0)`, so the budgets hold exactly, at the cost of turning the problem into a mixed-integer program. It reuses the indicators the cardinality, threshold and fee builders already create (see `short_mip_threshold_constraints`) rather than adding its own, and is ignored when the weight bounds admit no shorts.",#
                      :lt => "`lt`: Long-side minimum holding threshold.",#
                      :st => "`st`: Short-side minimum holding threshold.",#
                      :lcse => "`lcse`: Linear constraint set estimator(s).",#
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
                      :ple => "`ple`: Phylogeny constraint estimator(s).",#
                      :lcsr => "`lcsr`: Processed linear constraint set result.",#
                      :gcardr => "`gcardr`: Processed grouped cardinality constraint result.",#
                      :sgcardr => "`sgcardr`: Processed sub-grouped cardinality constraint result.",#
                      :ret_jmp => "`ret`: Returns estimator for the `JuMP` model.",#
                      :ccnt => "`ccnt`: Custom `JuMP` constraint.",#
                      :cobj => "`cobj`: Custom `JuMP` objective.",#
                      :sc => "`sc`: Constraint scale factor.",#
                      :so => "`so`: Objective scale factor.",#
                      :ss => "`ss`: Optional scalar shrinkage parameter.",#
                      :card => "`card`: Global cardinality constraint.",#
                      :scard => "`scard`: Sub-group cardinality constraint(s).",#
                      :l2c => "`l2c`: 2-norm ceiling on the weights — bounds `norm(w, 2) <= l2c * k` (`k` is the budget, `1` for a fully invested portfolio). Smaller `l2c` forces a more evenly spread portfolio. Used as a diversification floor via the reciprocal: `l2c = 1 / sqrt(m)` requires at least `m` effective assets (`inv(norm(w, 2)^2) >= m`). Norm-constraint family with `lpc` and `linfc`.",#
                      :lpc => "`lpc`: p-norm ceiling(s) on the weights at an arbitrary norm order. Each [`LpRegularisation`](@ref) supplies a norm order `p` and a bound `val`, enforcing `norm(w, p) <= val * k`. Smaller `val` forces a more evenly spread portfolio. Used as a diversification floor via the reciprocal: `val = m^(-1/p)` requires at least `m` p-norm effective assets (`inv(norm(w, p)^p) >= m`). Norm-constraint family with `l2c` and `linfc`.",#
                      :linfc => "`linfc`: ∞-norm ceiling on the weights — a cap on the largest absolute weight: `norm(w, Inf) <= linfc * k`. So `linfc = 0.2` caps the largest weight at 20% of a fully invested portfolio. Used as a diversification floor via the reciprocal: `linfc = 1 / m` spreads the portfolio across at least `m` assets. Norm-constraint family with `l2c` and `lpc`.",#
                      :l1 => "`l1`: L1 regularisation coefficient.",#
                      :l2 => "`l2`: L2 regularisation term(s).",#
                      :linf => "`linf`: L∞ regularisation coefficient.",#
                      :lp => "`lp`: Lp regularisation specification(s).",#
                      :l2reg_val => "`val`: L2 regularisation penalty coefficient.",#
                      :l2reg_alg => "`alg`: Second-moment formulation used to express the L2 penalty.",#
                      :lpreg_p => "`p`: Norm order, `p > 1`.",#
                      :lpreg_val => "`val`: Penalty coefficient when the estimator is used as a regularisation term (the `lp` field of [`JuMPOptimiser`](@ref)), or the upper bound on the p-norm of the weights when it is used as a norm constraint (the `lpc` field).",#
                      :brt => "`brt`: Whether to use bootstrap returns.",#
                      :cle_pr => "`cle_pr`: Whether to pass the prior result to the clustering estimator.",#
                      :wf => "`wf`: Weight finaliser.",#
                      :rkb => "`rkb`: Risk budget estimator or result.",#
                      :rba => "`rba`: Risk budget algorithm.",#
                      :resi => "`resi`: Inner optimisation results.",#
                      :reso => "`reso`: Outer optimisation results.",#
                      :opti => "`opti`: Inner optimiser.",#
                      :opto => "`opto`: Outer optimiser.",#
                      # Cross-validation.
                      :n_folds => "`n`: Number of folds.",#
                      :n_test_folds => "`n_test_folds`: Number of test folds.",#
                      :purged_size => "`purged_size`: Number of observations to purge between train and test sets.",#
                      :embargo_size => "`embargo_size`: Number of observations to embargo after the test set.",#
                      :train_idx => "`train_idx`: Training set indices.",#
                      :test_idx => "`test_idx`: Test set indices.",#
                      :train_size => "`train_size`: Training window size.",#
                      :test_size => "`test_size`: Test window size.",#
                      :period => "`period`: Time period for date-based walk-forward cross-validation.",#
                      :period_offset => "`period_offset`: Offset applied to the walk-forward period.",#
                      :adjuster => "`adjuster`: Function for adjusting walk-forward dates.",#
                      :previous => "`previous`: Whether to include the previous period in the training window.",#
                      :expand_train => "`expand_train`: Whether to expand the training window over time.",#
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
                      :q_scorer => "`q`: Target quantile for scoring.",#
                      :r_kwargs => "`r_kwargs`: Keyword arguments passed to the risk measure.",#
                      :q_kwargs => "`q_kwargs`: Keyword arguments passed to `quantile`.",#
                      :p_cv => "`p`: Hyperparameter search grid.",#
                      # Prediction result fields.
                      :pred_nx => "`nx`: Asset name vector.",#
                      :pred_nf => "`nf`: Factor name vector.",#
                      :pred_nb => "`nb`: Benchmark name vector.",#
                      :pred_B => "`B`: Benchmark returns.",#
                      :ts => "`ts`: Timestamp vector.",#
                      :iv_ret => "`iv`: Investment vehicle returns.",#
                      :ivpa => "`ivpa`: Investment vehicle per-asset allocation.",#
                      :pred_res => "`res`: Optimisation result from the training fold.",#
                      :pred => "`pred`: Collection of fold predictions.",#
                      :mrd => "`mrd`: Aggregated multi-period returns result.",#
                      :id_pred => "`id`: Path or fold identifier.",#
                      # Allocation.
                      :shares => "`shares`: Number of shares allocated per asset.",#
                      :cost_alloc => "`cost`: Cost of the allocation.",#
                      :cash_alloc => "`cash`: Remaining uninvested cash after allocation.",#
                      :unit => "`unit`: Minimum purchase unit (e.g., price per share or lot size).",#
                      :sc_alloc => "`sc`: Constraint check named tuple for the allocation solver.",#
                      :so_alloc => "`so`: Objective settings for the allocation solver.",#
                      :wf_alloc => "`wf`: Weight finaliser for the allocation result.",#
                      # Cluster node.
                      :id_node => "`id`: Node identifier.",#
                      :left_node => "`left`: Left child node.",#
                      :right_node => "`right`: Right child node.",#
                      :height_node => "`height`: Height of the node in the dendrogram.",#
                      :level_node => "`level`: Level of the node in the hierarchical structure.",#
                      # Other.
                      :linkage => "`linkage`: Hierarchical clustering linkage method.",#
                      :dlb => "`dlb`: Default lower bound.",#
                      :dub => "`dub`: Default upper bound.",#
                      :err => "`err`: Tracking error tolerance.",#
                      :tralg => "`alg`: Tracking formulation algorithm.",#
                      :rt => "`rt`: Returns estimator.",#
                      :rk => "`rk`: Risk measure for ratio computation.",#
                      :ohf => "`ohf`: Whether to compute the ratio only for long positions.",#
                      :r1 => "`r1`: First risk measure.",#
                      :r2 => "`r2`: Second risk measure.",#
                      :ri => "`ri`: Inner risk measure.",#
                      :ro => "`ro`: Outer risk measure.",#
                      :scai => "`scai`: Inner scalariser.",#
                      :scao => "`scao`: Outer scalariser.",#
                      :params => "`params`: Schur complement decomposition parameters.",#
                      :gamma_schur => "`gamma`: Schur complement decomposition parameter.",#
                      :z => "`z`: Regularisation coefficient for log risk budgeting.",#
                      :tol => "`tol`: Convergence tolerance.",#
                      :iter => "`iter`: Maximum number of iterations.",#
                      :w_opt_noc => "`w_opt`: Optimal portfolio weights.",#
                      :w_min_noc => "`w_min`: Minimum risk portfolio weights.",#
                      :w_max_noc => "`w_max`: Maximum return portfolio weights.",#
                      :ucs_flag => "`ucs_flag`: Whether to use the uncertainty set.",#
                      :slv_alloc => "`slv`: Solver or vector of solvers for the allocation problem.",#
                      # Optimiser config.
                      :opt => "`opt`: `JuMP` optimiser configuration.",#
                      :kwargs => "`kwargs`: Additional keyword arguments.",#
                      # Index.
                      :idx => "`idx`: Index vector.",#
                      # Risk measure.
                      :r => "`r`: Risk measure or vector of risk measures.",#
                      # Returns estimator.
                      :ret => "`ret`: Returns estimator for `JuMP` models.",#
                      # Weight bounds.
                      :wb => "`wb`: Weight bounds.",#
                      # Turnover.
                      :tn => "`tn`: Turnover constraint estimator.",#
                      # Tracking.
                      :tr => "`tr`: Tracking error constraint estimator.",#
                      # Fees.
                      :fees => "`fees`: Fees estimator or result.",#
                      # Near optimal centering result fields.
                      :attrs_noc => "`attrs`: Processed JuMP optimiser attributes for the model-assembly pipeline.",#
                      :w_opt => "`w_opt`: Optimal portfolio weights.",#
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
                      :rk_opt => "`rk_opt`: Optimal risk target.",#
                      :noc_retcode => "`noc_retcode`: Return code for the near-optimal centering sub-problem.",#
                      # Discrete allocation result fields.
                      :l_model => "`l_model`: `JuMP` model for the long allocation.",#
                      :s_model => "`s_model`: `JuMP` model for the short allocation.",#
                      :l_retcode => "`l_retcode`: Return code for the long allocation sub-problem.",#
                      :s_retcode => "`s_retcode`: Return code for the short allocation sub-problem.",#
                      # Risk budgeting.
                      :prb => "`prb`: Processed risk budgeting configuration.",#
                      :l_wass => "`l`: Wasserstein ambiguity scale factor.",#
                      :r_wass => "`r`: Wasserstein radius parameter.",#
                      :g_rm => "`g`: Risk aversion parameter.",#
                      :max_phi => "`max_phi`: Maximum allowed value for any OWA weight.",#
                      :w1_owa => "`w1`: Optional first OWA weight vector.",#
                      :w2_owa => "`w2`: Optional second OWA weight vector.",#
                      :owa_w => "`w`: Optional OWA weight vector.",#
                      :owa_method => "`method`: OWA weight estimation method.",#
                      :lm_k => "`k`: L-moment order.",#
                      :alpha_i => "`alpha_i`: Lower integration bound for the tail Gini approximation.",#
                      :a_sim => "`a_sim`: Number of integration points for the tail Gini approximation.",#
                      :beta_i => "`beta_i`: Lower integration bound for the upper tail Gini approximation.",#
                      :b_sim => "`b_sim`: Number of integration points for the upper tail Gini approximation.",#
                      # Constraint generation.
                      :rkb_val => "`val`: Vector of risk budget allocations.",#
                      :rkbe_val => "`val`: Mapping of names to risk budget values.",#
                      :as_key => "`key`: Key in `dict` identifying the primary asset list.",#
                      :as_ukey => "`ukey`: Key prefix for unique-entry group variants in `dict`.",#
                      :p_phylo => "`p`: Non-negative penalty parameter for the phylogeny constraint.",#
                      :A_phylo => "`A`: Phylogeny constraint matrix.",#
                      :B_phylo => "`B`: Group sizes or allocations vector.",#
                      :scale_phylo => "`scale`: Non-negative big-M scaling factor for the MIP formulation.",#
                      :cc_A => "`A`: Centrality estimator.",#
                      :cc_B => "`B`: Centrality threshold or reduction measure.",#
                      :cc_comp => "`comp`: Comparison operator for the centrality constraint.",#
                      :lce_val => "`val`: Constraint equation(s) to parse.",#
                      :asets_val => "`val`: Group name key for asset set membership matrix extraction.",#
                      :thr_val => "`val`: Asset-specific threshold value(s).",#
                      :thr_res_val => "`val`: Threshold value(s) for portfolio weights.",#
                      # Entropy pooling.
                      :sc1 => "`sc1`: Scaling parameter for the objective function.",#
                      :sc2 => "`sc2`: Scaling parameter for constraint penalties.",#
                      :epalg => "`alg`: Entropy pooling algorithm.",#
                      :epoptalg => "`alg`: Entropy pooling optimisation algorithm.",#
                      :ep_w => "`w`: Prior observation probability weights. If `nothing`, uniform weights are used.",#
                      # Opinion pooling.
                      :opalg => "`alg`: Opinion pooling algorithm.",#
                      # Non-optimisation risk measures.
                      :rt_mean => "`rt`: Mean return estimator.",#
                      # Regime adjusted estimators.
                      :decay => "`decay`: Exponential decay factor for the exponentially weighted estimator.",#
                      :min_obs => "`min_obs`: Minimum number of observations required before the estimator produces a valid result.",#
                      :hac_lags => "`hac_lags`: Optional number of lags for Heteroskedasticity and Autocorrelation Consistent (HAC) kernel correction of squared returns. If `nothing`, no HAC correction is applied.",#
                      :regime_method => "`regime_method`: Regime adjustment method used to compute the per-observation regime state.",#
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
                      :ra_w => "`w`: Optional portfolio weights vector for the portfolio target. If `nothing`, equal weights are used.",#
                      :sq => "`sq`: Whether to use variance instead of volatility in the inverse weighting.",#
                      :wfalg => "`alg`: Weight finaliser error formulation algorithm.",#
                      :res_retcode => "`res`: Optional result or message from the solver.",#
                      :N_msc => "`N`: Number of bisection steps for the monotonic Schur complement.",#
                      :alpha_dirichlet => "`alpha`: Dirichlet concentration parameter.",#
                      :opt_hier => "`opt`: Base hierarchical optimiser configuration.",#
                      :strict_opt => "`strict`: Whether to strictly enforce weight bounds.",#
                      :strict_conv => "`strict`: Whether to raise an error if convergence is not achieved.",#
                      :schalg => "`alg`: Schur complement algorithm variant.")
"""
    field_dict

Derived dictionary mapping argument keys to field description strings, used for `\$(FIELDS)`-style docstring interpolation.

Each entry is derived from [`arg_dict`](@ref) by stripping the leading parameter name prefix (everything up to and including the first `:`).
"""
const field_dict = Dict(key => strip(val[(findfirst(":", val)[1] + 1):end])
                        for (key, val) in arg_dict)
"""
    err_name_dict

Maps high-order-moment argument keys to the domain noun used in error messages, so a
message names what the caller supplied (e.g. `cokurtosis`) rather than the bare field
symbol. The symbol itself is appended at the call site, giving messages like
``cokurtosis (`kt`) cannot be empty``.
"""
const err_name_dict = Dict(:kt => "cokurtosis", :sk => "coskewness",
                           :V => "negative spectral coskewness",
                           :D2 => "duplication matrix", :L2 => "elimination matrix",
                           :S2 => "summation matrix", :f_kt => "factor cokurtosis",
                           :f_sk => "factor coskewness",
                           :f_V => "factor negative spectral coskewness")
"""
    const val_dict = Dict(:oow => "If `w` is not `nothing`, `!isempty(w)`.")

Validation rules for certain arg_dict terms used in the documentation of `PortfolioOptimisers.jl`.
"""
const val_dict = Dict(:oow => "If `w` is not `nothing`, `!isempty(w)`.",
                      :oidx => "If `idx` is not `nothing`, `!isempty(idx)` and all indices are positive integers.",
                      :gerbt => "`0 <= t`.",#
                      :t => "`0 < t < 1`.",#
                      :c1 => "`0 < c1 <= 1`.",#
                      :c2 => "`0 < c2 <= 1`.",#
                      :c3c2 => "`c3 > c2`.",#
                      :dims => "`dims in (1, 2)`.",#
                      :alpha => "`0 < alpha < 1`.",#
                      :beta => "`0 < beta < 1`.",#
                      :bins => "If `bins` is an integer, `bins > 0`.",#
                      :dopower => "If `power` is not `nothing`, `power >= 1`.",#
                      :settings => "If not `nothing`, `!isempty(settings)`.",#
                      :S => "`!isempty(S)`.",#
                      :D => "`!isempty(D)`.",#
                      :ck => "`k >= 1`.",#
                      :lm_k => "`k >= 2`.",#
                      :alpha_i_alpha => "`0 < alpha_i < alpha < 1`.",#
                      :a_sim_pos => "`a_sim > 0`.",#
                      :beta_i_beta => "`0 < beta_i < beta < 1`.",#
                      :b_sim_pos => "`b_sim > 0`.",#
                      :S_D => "size(S) == size(D)`.",#
                      :max_k => "If `max_k` is not `nothing`, `max_k >= 1`.",#
                      :kalg => "If `alg` is not `nothing`, `alg >= 1`.",#
                      :dbhtpower => "`power > 0`.",#
                      :dbhtcoef => "`coef > 0`.", :Xe => "`!isempty(X)`.",#
                      :phX_Xv => "`If `X` is a `MatNum`:\n    + Must be symmetric, `LinearAlgebra.issymmetric(X)`\n    + Must have zero diagonal, `all(iszero, LinearAlgebra.diag(X))`.",#
                      :ntn => "`n >= 1`.",#
                      :A => "`!isempty(A)`.",#
                      :B => "`!isempty(B)`.",#
                      :eqineq => "Both `eq` and `ineq` cannot be `nothing` at the same time, `!(isnothing(ineq) && isnothing(eq))`.",
                      :decay => "`decay > 0`.",#
                      :rf => "`isfinite(rf)`.",#
                      :q_scorer => "`0 <= q <= 1`.",#
                      :unit => "`unit > 0`.",#
                      :katz_alpha => "`alpha > 0`.",#
                      :min_obs => "`min_obs > 0`.",#
                      :hac_lags => "If `hac_lags` is not `nothing`, `hac_lags > 0`.",#
                      :regime_min_obs => "`regime_min_obs > 0`.",#
                      :regime_lohi_mult => "If `regime_lohi_mult` is not `nothing`, `0 < regime_lohi_mult[1] < regime_lohi_mult[2]`.",#
                      :ra_x => "`x` is valid",#
                      :ra_y => "`y` is valid",#
                      :ra_norm_x => "`x` is valid")

"""
Dictionary containing return value descriptions for common parameters used in `PortfolioOptimisers.jl`.
"""
const ret_dict = Dict(:mu => "`mu::ArrNum`: Expected returns vector `features x 1` if the `dims` keyword does not exist or `dims = 2`, `1 x features` if `dims = 1`.",#
                      :sigma => "`sigma::MatNum`: Covariance matrix `features x features`.",#
                      :rho => "`rho::MatNum`: Correlation matrix `features x features`.",#
                      :sigrho => "`sigrho::MatNum`: Covariance/correlation matrix `features x features`.",#
                      :sk => "`sk::MatNum`: Coskewness matrix `features x features`.",#
                      :kte => "`kte::MatNum`: Cokurtosis matrix `features x features`.",#
                      :me => "`me`: New expected returns estimator of the same type as the argument, with the appropriate weights applied.",#
                      :mev => "`mev`: New expected returns estimator of the same type as the argument, for the new view.",#
                      :ce => "`ce`: New covariance estimator of the same type as the argument, with the new weights applied.",#
                      :cev => "`ce`: New covariance estimator of the same type as the argument, for the new view.",#
                      :ve => "`ve`: New variance estimator of the same type as the argument, with the new weights applied.",#
                      :vev => "`ve`: New variance estimator of the same type as the argument, for the new view.",#
                      :skev => "`skev`: New coskewness estimator of the same type as the argument, for the new view.",#
                      :ktev => "`kev`: New cokurtosis estimator of the same type as the argument, for the new view.",#
                      :stdvar => "`res::ArrNum`: Variance or standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",#
                      :stdvarnum => "`res::Number`: Variance or standard deviation `X`",#
                      :stdarr => "`sd::ArrNum`: Standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",
                      :vararr => "`vr::ArrNum`: Variance vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",
                      :stdnum => "`vr::Number`: Standard deviation of `X`",
                      :varnum => "`vr::Number`: Variance of `X`",
                      :algw => "`alg`: New algorithm instance of the same type as the argument, with the new weights applied.",
                      :alg => "`alg`: The original algorithm instance.")
"""
    math_dict

Dictionary of mathematical notation descriptions used for docstring interpolation throughout `PortfolioOptimisers.jl`.

Keys are symbols that identify mathematical variables or subscripts; values are LaTeX-formatted strings suitable for embedding in docstrings.
"""
const math_dict = Dict(:Xv => "``\\boldsymbol{X}``: Data vector `observations × 1`.",#
                       :tgt => "``t``: Target value, usually the unweighted (or weighted) expected value ``E[\\boldsymbol{X}]``.",#
                       :A => "``\\mathbf{A}``: Constraint coefficient matrix.",#
                       :B => "``\\boldsymbol{B}``: Constraint response vector.",#
                       :x => "``\\boldsymbol{x}``: Constrained variable.",#
                       :ineq => "``\\text{ineq}``: Subscript for inequality constraints.",#
                       :eq => "``\\text{eq}``: Subscript for equality constraints.",#
                       # Portfolio returns and dimensions.
                       :xret => "``\\boldsymbol{x}``: Portfolio returns vector ``T \\times 1``.",#
                       :T => "``T``: Number of observations.",#
                       :N => "``N``: Number of assets.",#
                       # Risk measure parameters.
                       :alpha_rm => "``\\alpha``: Significance level (left tail probability), ``\\alpha \\in (0, 1)``.",#
                       :w_port => "``\\boldsymbol{w}``: Portfolio weights vector ``N \\times 1``.",#
                       # Absolute drawdown series.
                       :ct => "``c_t``: Cumulative simple portfolio return at period ``t``.",#
                       :dtdd => "``d_t \\leq 0``: Absolute drawdown at period ``t``.",#
                       # Relative drawdown series.
                       :Ct => "``C_t``: Compound wealth process at period ``t``.",#
                       :rdt => "``rd_t \\leq 0``: Relative drawdown at period ``t``.",#
                       # JuMP optimisation variables.
                       :k_budget => "``k``: Budget scaling / homogenisation variable.",#
                       :mu_er => "``\\boldsymbol{\\mu}``: Expected returns vector ``N \\times 1``.",#
                       :R_w => "``R(\\boldsymbol{w})``: Portfolio risk.")

"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimator types in `PortfolioOptimisers.jl`.

All custom estimators should subtype `AbstractEstimator`.

Estimators consume data to estimate parameters or models. Some estimators may utilise different algorithms. These can range from simple implementation details that don't change the result much but may have different numerical characteristics, to entirely different methodologies or algorithms yielding different results.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all algorithm types in `PortfolioOptimisers.jl`.

All algorithms should subtype `AbstractAlgorithm`.

Algorithms are often used by estimators to perform specific tasks. These can be in the form of simple implementation details to entirely different procedures for estimating a quantity.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all result types in `PortfolioOptimisers.jl`.

All result objects should subtype `AbstractResult`.

Result types encapsulate the outcomes of estimators. This makes dispatch and usage more straightforward, especially when the results encapsulate a wide range of information.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for dynamically computed observation weight estimators.

`DynamicAbstractWeights` subtypes are used when observation weights must be computed from data (rather than supplied directly as a numeric vector). They are passed to estimators that accept an `ObsWeights` argument and evaluated at fit time.

# Interfaces

In order to implement a new dynamic observation weight estimator which will work seamlessly with the library, subtype `DynamicAbstractWeights` with all necessary parameters struct, and implement the following methods:

  - `get_observation_weights(w::DynamicAbstractWeights, X::VecNum; kwargs...) -> StatsBase.AbstractWeights`: Returns observation weights for a 1D vector `X`.
  - `get_observation_weights(w::DynamicAbstractWeights, X::MatNum; dims::Int = 1, kwargs...) -> StatsBase.AbstractWeights`: Returns observation weights for a 2D matrix `X`, with `dims` specifying the dimension along which to compute weights.

## Arguments

  - `w`: Subtype of `DynamicAbstractWeights` with all necessary parameters.
  - $(arg_dict[:X_Xv])
  - `dims`: Dimension along which to compute weights for a 2D matrix `X`.
  - `kwargs...`: Additional keyword arguments passed to the weight computation function.

## Returns

  - `w::StatsBase.AbstractWeights`: Observation weights for the input data `X`.

# Examples

We can create a dummy dynamic observation weight estimator as follows:

```jldoctest
julia> struct MyWeights{T} <: PortfolioOptimisers.DynamicAbstractWeights
           half_life::T
           function MyWeights(half_life::Integer)
               if half_life < one(half_life)
                   throw(DomainError(half_life, "half_life must be an integer greater than zero"))
               end
               return new{typeof(half_life)}(half_life)
           end
       end

julia> function MyWeights(; half_life::Integer = 5)
           return MyWeights(half_life)
       end
MyWeights

julia> function PortfolioOptimisers.get_observation_weights(w::PortfolioOptimisers.DynamicAbstractWeights,
                                                            X::PortfolioOptimisers.VecNum;
                                                            kwargs...)
           lambda = 2^(-inv(w.half_life))
           return eweights(1:length(X), lambda; scale = true)
       end

julia> function PortfolioOptimisers.get_observation_weights(w::PortfolioOptimisers.DynamicAbstractWeights,
                                                            X::PortfolioOptimisers.MatNum;
                                                            dims::Int = 1, kwargs...)
           lambda = 2^(-inv(w.half_life))
           return eweights(1:size(X, dims), lambda; scale = true)
       end

julia> PortfolioOptimisers.get_observation_weights(MyWeights(), 1:10)
10-element Weights{Float64, Float64, Vector{Float64}}:
 1.0207079199119523e-8
 7.88499313633082e-8
 6.091176089370138e-7
 4.705448122809607e-6
 3.63496994859362e-5
 0.00028080229942667527
 0.002169204490777577
 0.016757156662950766
 0.12944943670387588
 1.0

julia> PortfolioOptimisers.get_observation_weights(MyWeights(), ones(3, 10); dims = 2)
10-element Weights{Float64, Float64, Vector{Float64}}:
 1.0207079199119523e-8
 7.88499313633082e-8
 6.091176089370138e-7
 4.705448122809607e-6
 3.63496994859362e-5
 0.00028080229942667527
 0.002169204490777577
 0.016757156662950766
 0.12944943670387588
 1.0
```

# Related

  - [`ObsWeights`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
"""
abstract type DynamicAbstractWeights <: AbstractEstimator end
"""
    define_pretty_show(T, flag::Bool = true)

Macro to define a custom pretty-printing `Base.show` method for types in `PortfolioOptimisers.jl`.

This macro generates a `show` method that displays the type name and all fields in a readable, aligned format. For fields that are themselves custom types or collections, the macro recursively applies pretty-printing for nested structures. Handles compact and multiline IO contexts gracefully.

# Arguments

  - `T`: The type for which to define the pretty-printing method.

# Returns

  - Defines a `Base.show(io::IO, obj::T)` method for the given type.

# Details

  - Prints the type name and all fields with aligned labels.
  - Recursively pretty-prints nested custom types and collections.
  - Handles compact and multiline IO contexts.
  - Displays matrix fields with their size and type.
  - Lists a vector of pretty-printable structs as a `"N-element Vector{Name}"` summary followed by one collapsed line per element (each a wrapper-type name, with a trailing `" ⋯"` when the element has fields). Long listings are truncated head-and-tail with a `"⋮"` line, bounded by [`compact_show_budget`](@ref).
  - Collapses an oversized nested struct field to `Name ⋯` when its rendered height exceeds [`compact_show_budget`](@ref); see [`set_compact_show!`](@ref).
  - Skips fields that are not present or are `nothing`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`Base.show`](https://docs.julialang.org/en/v1/base/io/#Base.show)
"""
macro define_pretty_show(T, flag::Bool = true)
    esc(quote
            if $flag
                has_pretty_show_method(::$T) = true
            end
            function Base.show(io::IO, obj::$T)
                fields = fieldnames(typeof(obj))
                tobj = typeof(obj)
                if isempty(fields)
                    return print(io, string(tobj, "()"), '\n')
                end
                if get(io, :compact, false) || get(io, :multiline, false)
                    return print(io, string(tobj), '\n')
                end
                name = Base.typename(tobj).wrapper
                print(io, name, '\n')
                padding = maximum(map(length, map(string, fields))) + 2
                for (i, field) in enumerate(fields)
                    if hasfield(typeof(obj), field)
                        val = getproperty(obj, field)
                    else
                        continue
                    end
                    flag = has_pretty_show_method(val)
                    sym1 = ifelse(i == length(fields) &&
                                      (!flag || (flag && isempty(fieldnames(typeof(val))))),
                                  '┴', '┼')
                    print(io, lpad(string(field), padding), " ")
                    if isnothing(val)
                        print(io, "$(sym1) nothing", '\n')
                    elseif flag
                        ioalg = IOContext(IOBuffer(), :limit => get(io, :limit, false),
                                          :displaysize => displaysize(io))
                        pc = get(io, :po_compact, :__unset__)
                        if pc !== :__unset__
                            ioalg = IOContext(ioalg, :po_compact => pc)
                        end
                        show(ioalg, val)
                        algstr = String(take!(ioalg.io))
                        alglines = split(algstr, '\n')
                        budget = compact_show_budget(io)
                        if !isnothing(budget) &&
                           count(l -> !(isempty(l) || l == "\n"), alglines) > budget
                            conn = ifelse(i == length(fields), '┴', '┼')
                            print(io, "$(conn) ", Base.typename(typeof(val)).wrapper, " ⋯",
                                  '\n')
                        else
                            print(io, "$(sym1) ", alglines[1], '\n')
                            for l in alglines[2:end]
                                if isempty(l) || l == '\n'
                                    continue
                                end
                                sym2 = '│'
                                print(io, lpad("$sym2 ", padding + 3), l, '\n')
                            end
                        end
                    elseif isa(val, AbstractVector) &&
                           !isempty(val) &&
                           all(has_pretty_show_method, val)
                        print(io, "┼ ", pretty_show_vector_summary(val), '\n')
                        ellines = [pretty_show_vector_element(v) for v in val]
                        for l in pretty_show_vector_body(io, ellines)
                            print(io, lpad("│ ", padding + 3), l, '\n')
                        end
                    elseif isa(val, AbstractMatrix)
                        print(io, "$(sym1) $(size(val,1))×$(size(val,2)) $(typeof(val))",
                              '\n')
                    elseif isa(val, AbstractVector) && length(val) > 6 ||
                           isa(val, AbstractVector{<:AbstractArray})
                        print(io, "$(sym1) $(length(val))-element $(typeof(val))", '\n')
                    elseif isa(val, DataType)
                        tval = typeof(val)
                        valstr = Base.typename(tval).wrapper
                        print(io, "$(sym1) $(tval): ", valstr, '\n')
                    else
                        print(io, "$(sym1) $(typeof(val)): ", repr(val), '\n')
                    end
                end
                return nothing
            end
        end)
end
"""
$(DocStringExtensions.TYPEDEF)

Thread-safe holder for a package-level configuration value, combining a persistent global default with a task-scoped override.

Reads go through `cfg[]`, which returns the innermost active scoped override when inside a `with_*` block, otherwise the global default. The default is an `@atomic` field swapped as a whole — a `set_*!` call is a single atomic store, so concurrent readers (e.g. the `FLoops.@floop` loops inside meta-optimisers) can never observe a torn or partially-updated configuration. The scoped override is a `Base.ScopedValues.ScopedValue`: it is inherited by tasks spawned inside the scope, restored automatically when the scope exits, and invisible to unrelated concurrent tasks.

Configs held this way store *immutable* structs (or bits values); changing any knob builds a new value and swaps it in, never mutates in place.

Used by [`COMPACT_SHOW`](@ref), [`STRING_DISTANCE`](@ref), and [`EQUATION_LIMITS`](@ref); their global defaults are set via the `set_*!` setters, scoped overrides via the `with_*` helpers, and load-time per-project defaults via Preferences.jl (see [`apply_preferences!`](@ref)).

# Related

  - [`set_compact_show!`](@ref) / [`with_compact_show`](@ref)
  - [`set_string_distance!`](@ref) / [`with_string_distance`](@ref)
  - [`set_equation_limits!`](@ref) / [`with_equation_limits`](@ref)
  - [`apply_preferences!`](@ref)
"""
mutable struct ScopedConfig{T}
    @atomic default::T
    const scoped::ScopedValue{Union{Nothing, T}}
    function ScopedConfig{T}(x) where {T}
        return new{T}(convert(T, x), ScopedValue{Union{Nothing, T}}(nothing))
    end
end
ScopedConfig(x::T) where {T} = ScopedConfig{T}(x)
"""
    getindex(cfg::ScopedConfig)

Read the active value of a [`ScopedConfig`](@ref): the innermost task-scoped override when inside a `with_*` block, otherwise the global default (read atomically).
"""
Base.getindex(cfg::ScopedConfig) = @something(cfg.scoped[], @atomic(cfg.default))
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Atomically replace the global default of a [`ScopedConfig`](@ref) with `x` and return it. Does not affect any active scoped override.

# Related

  - [`ScopedConfig`](@ref)
  - [`with_config`](@ref)
"""
function set_default!(cfg::ScopedConfig{T}, x) where {T}
    x = convert(T, x)
    @atomic cfg.default = x
    return x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run `f()` with the [`ScopedConfig`](@ref) `cfg` overridden to `x` for the dynamic extent of the call, restoring the previous value on exit. Thread-safe: the override is task-scoped (inherited by tasks spawned inside `f`, invisible to concurrent tasks outside it).

# Related

  - [`ScopedConfig`](@ref)
  - [`set_default!`](@ref)
"""
function with_config(f, cfg::ScopedConfig{T}, x) where {T}
    return Base.ScopedValues.with(f, cfg.scoped => convert(T, x))
end
"""
Global control for collapsing large nested structs in [`@define_pretty_show`](@ref) output.

Holds one of:

  - `false`: collapsing disabled; nested structs always expand fully.
  - `true`: collapsing enabled with an automatic, terminal-size-derived line budget.
  - `n::Int`: collapsing enabled with a fixed line budget of `n`.

Held in a [`ScopedConfig`](@ref): set the global default via [`set_compact_show!`](@ref), override per scope via [`with_compact_show`](@ref), and read (together with the per-call `:po_compact` IO property) by [`compact_show_budget`](@ref). The default may be seeded per project at load time via the `"compact_show"` preference (see [`apply_preferences!`](@ref)).
"""
const COMPACT_SHOW = ScopedConfig{Union{Bool, Int}}(true)
"""
    set_compact_show!(x::Bool)
    set_compact_show!(n::Integer)

Configure whether [`@define_pretty_show`](@ref) collapses large nested structs.

  - `set_compact_show!(false)`: disable collapsing (always expand fully).
  - `set_compact_show!(true)`: enable collapsing with an automatic, terminal-size-derived budget.
  - `set_compact_show!(n)`: enable collapsing with a fixed line budget `n`.

Collapsing only ever applies to height-limited output (`get(io, :limit, false)`), i.e. the interactive REPL. Non-limited output (`string`, `repr`, file writes) always expands fully. The documentation build disables this so rendered docs keep full detail. Individual calls can override the global setting with the `:po_compact` IO property (`false`, `true`, or an `Int`).

Sets the global default (atomically; see [`ScopedConfig`](@ref)). For a temporary, task-scoped override use [`with_compact_show`](@ref).

# Related

  - [`@define_pretty_show`](@ref)
  - [`compact_show_budget`](@ref)
  - [`with_compact_show`](@ref)
"""
set_compact_show!(x::Bool) = set_default!(COMPACT_SHOW, x)
set_compact_show!(n::Integer) = set_default!(COMPACT_SHOW, Int(n))
"""
    with_compact_show(f, x::Bool)
    with_compact_show(f, n::Integer)

Run `f()` with the [`COMPACT_SHOW`](@ref) collapsing setting overridden to `x`/`n` for the dynamic extent of the call, restoring the previous setting on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched.

# Related

  - [`set_compact_show!`](@ref)
  - [`compact_show_budget`](@ref)
"""
with_compact_show(f, x::Bool) = with_config(f, COMPACT_SHOW, x)
with_compact_show(f, n::Integer) = with_config(f, COMPACT_SHOW, Int(n))
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the line budget that triggers collapsing a nested struct rendered by [`@define_pretty_show`](@ref).

The per-call `:po_compact` IO property takes precedence over the global [`COMPACT_SHOW`](@ref) setting; both accept `false` (disabled), `true` (automatic budget), or an `Int` (fixed budget). The automatic budget is `max(8, displaysize(io)[1] - 4)`, so only subtrees that nearly fill or exceed the terminal collapse.

# Returns

  - `nothing` when collapsing is disabled.
  - `budget::Int` (the maximum number of rendered lines a nested struct may occupy before collapsing) otherwise.

# Related

  - [`set_compact_show!`](@ref)
  - [`@define_pretty_show`](@ref)
"""
function compact_show_budget(io::IO)
    v = get(io, :po_compact, :__unset__)
    if v === :__unset__
        # No per-call override: only collapse height-limited output (the REPL),
        # leaving `string`/`repr`/file writes fully expanded.
        if !(get(io, :limit, false))
            return nothing
        end
        v = COMPACT_SHOW[]
    end
    if v === false
        return nothing
    end
    if v isa Integer && !(v isa Bool)
        return Int(v)
    end
    return max(8, displaysize(io)[1] - 4)
end
"""
$(DocStringExtensions.TYPEDEF)

Global configuration for the fuzzy "did you mean?" suggestions appended to "variable not in asset universe" messages by [`did_you_mean`](@ref).

# Fields

  - `dist`: the `StringDistances.StringDistance` used to score candidate names against the offending one (default `StringDistances.Levenshtein()`).
  - `min_score`: the minimum normalised similarity in `[0, 1]` a candidate must reach before it is suggested (default `0.7`). Raising it toward `1` keeps only near-exact matches; setting it above `1` disables suggestions entirely — useful in meta-optimiser inner loops, where an asset name legitimately absent from a cluster/subset is not a typo and should draw no suggestion.

Immutable; held in the [`STRING_DISTANCE`](@ref) [`ScopedConfig`](@ref). Set the global default via [`set_string_distance!`](@ref), override per scope via [`with_string_distance`](@ref). Read by [`did_you_mean`](@ref).

# Related

  - [`STRING_DISTANCE`](@ref)
  - [`set_string_distance!`](@ref)
  - [`with_string_distance`](@ref)
  - [`did_you_mean`](@ref)
"""
struct StringDistanceConfig
    dist::StringDistances.StringDistance
    min_score::Float64
end
"""
    STRING_DISTANCE = ScopedConfig(StringDistanceConfig(StringDistances.Levenshtein(), 0.7))

Default string distance configuration for fuzzy "did you mean?" suggestions appended to "variable not in asset universe" messages by [`did_you_mean`](@ref). Read as `STRING_DISTANCE[]`; the defaults may be seeded per project at load time via the `"suggestion_distance"` / `"suggestion_min_score"` preferences (see [`apply_preferences!`](@ref)).

# Related

  - [`StringDistanceConfig`](@ref)
  - [`set_string_distance!`](@ref)
  - [`with_string_distance`](@ref)
  - [`did_you_mean`](@ref)
"""
const STRING_DISTANCE = ScopedConfig(StringDistanceConfig(StringDistances.Levenshtein(),
                                                          0.7))
"""
    set_string_distance!(; dist::StringDistances.StringDistance, min_score::Real)

Configure the global default fuzzy-suggestion settings read by [`did_you_mean`](@ref). The store is atomic (see [`ScopedConfig`](@ref)); unspecified keywords keep their current default. For a temporary, task-scoped override use [`with_string_distance`](@ref).

  - `dist`: distance used to rank candidate names (e.g. `StringDistances.Levenshtein()`, `StringDistances.DamerauLevenshtein()`, `StringDistances.JaroWinkler()`).
  - `min_score`: minimum normalised similarity in `[0, 1]` to emit a suggestion; set above `1` to disable suggestions.

Returns the new default [`StringDistanceConfig`](@ref).

# Related

  - [`did_you_mean`](@ref)
  - [`STRING_DISTANCE`](@ref)
  - [`with_string_distance`](@ref)
  - [`set_compact_show!`](@ref)
"""
function set_string_distance!(;
                              dist::StringDistances.StringDistance = (@atomic STRING_DISTANCE.default).dist,
                              min_score::Real = (@atomic STRING_DISTANCE.default).min_score)
    return set_default!(STRING_DISTANCE, StringDistanceConfig(dist, Float64(min_score)))
end
"""
    with_string_distance(f; dist::StringDistances.StringDistance = STRING_DISTANCE[].dist,
                         min_score::Real = STRING_DISTANCE[].min_score)

Run `f()` with the fuzzy-suggestion settings read by [`did_you_mean`](@ref) overridden for the dynamic extent of the call, restoring the previous settings on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched. Unspecified keywords inherit from the currently active value, so nested overrides compose.

Useful around a meta-optimiser run to silence suggestions (`min_score` above `1`) in its inner loops without affecting other concurrent work.

# Related

  - [`set_string_distance!`](@ref)
  - [`STRING_DISTANCE`](@ref)
  - [`did_you_mean`](@ref)
"""
function with_string_distance(f;
                              dist::StringDistances.StringDistance = STRING_DISTANCE[].dist,
                              min_score::Real = STRING_DISTANCE[].min_score)
    return with_config(f, STRING_DISTANCE, StringDistanceConfig(dist, Float64(min_score)))
end
"""
Global resource caps for equation parsing, guarding the string→AST trust boundary against a stack-exhaustion denial of service.

Constraint, Black-Litterman view and entropy-pooling view strings are untrusted input (config files, spreadsheets, UI). They funnel through [`parse_equation`](@ref), which calls `Meta.parse` and then walks the resulting expression tree recursively ([`eval_numeric_functions`](@ref), `collect_terms!`, `has_invalid_plus`). Without a bound, a deeply nested string (e.g. tens of thousands of parentheses) produces an AST deep enough to exhaust the stack and take down the host process. These caps fail closed with a typed `Meta.ParseError` well before that point.

# Fields

  - `max_length`: maximum number of characters in an equation string handed to `Meta.parse` (default `4096`). A legitimate linear constraint is short; the bound sits far above any real constraint and far below the nesting depth that threatens the stack. Because achieving nesting depth `d` from a string needs at least `d` characters, the length cap also bounds the AST depth of the *string* form.
  - `max_depth`: maximum expression-tree depth accepted by the `Expr` form of [`parse_equation`](@ref) (default `256`), which receives a pre-built AST that no length cap covers.

The values are conservative static defaults (portable across build and deployment machines, unlike a value auto-detected during precompilation). Immutable; held in the [`EQUATION_LIMITS`](@ref) [`ScopedConfig`](@ref). Set the global default via [`set_equation_limits!`](@ref), override per scope via [`with_equation_limits`](@ref). Both fields must be positive (enforced by the constructor). See `docs/adr/0027-cap-equation-parser-recursion.md`.
"""
struct EquationLimits
    max_length::Int
    max_depth::Int
    function EquationLimits(max_length::Integer, max_depth::Integer)
        @argcheck(max_length > 0 && max_depth > 0,
                  ArgumentError("max_length and max_depth must be positive."))
        return new(Int(max_length), Int(max_depth))
    end
end
"""
    EQUATION_LIMITS = ScopedConfig(EquationLimits(4096, 256))

Default global resource caps for equation parsing, guarding the string→AST trust boundary against a stack-exhaustion denial of service. Read as `EQUATION_LIMITS[]`; the defaults may be seeded per project at load time via the `"equation_max_length"` / `"equation_max_depth"` preferences (see [`apply_preferences!`](@ref)).

# Related

  - [`EquationLimits`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`with_equation_limits`](@ref)
  - [`parse_equation`](@ref)
"""
const EQUATION_LIMITS = ScopedConfig(EquationLimits(4096, 256))
"""
    set_equation_limits!(; max_length::Integer, max_depth::Integer)

Configure the global default equation-parser resource caps read at the string→AST trust boundary (see [`EQUATION_LIMITS`](@ref)).

  - `max_length`: maximum equation-string length passed to `Meta.parse`.
  - `max_depth`: maximum expression-tree depth accepted by the `Expr` form of [`parse_equation`](@ref).

Raise them for a genuinely large machine-generated constraint set, or lower them to tighten the boundary. Both must be positive; unspecified keywords keep their current default. The store is atomic (see [`ScopedConfig`](@ref)); for a temporary, task-scoped override use [`with_equation_limits`](@ref).

Returns the new default [`EquationLimits`](@ref).

# Related

  - [`EQUATION_LIMITS`](@ref)
  - [`with_equation_limits`](@ref)
  - [`parse_equation`](@ref)
  - [`set_string_distance!`](@ref)
"""
function set_equation_limits!(;
                              max_length::Integer = (@atomic EQUATION_LIMITS.default).max_length,
                              max_depth::Integer = (@atomic EQUATION_LIMITS.default).max_depth)
    return set_default!(EQUATION_LIMITS, EquationLimits(max_length, max_depth))
end
"""
    with_equation_limits(f; max_length::Integer = EQUATION_LIMITS[].max_length,
                         max_depth::Integer = EQUATION_LIMITS[].max_depth)

Run `f()` with the equation-parser resource caps (see [`EQUATION_LIMITS`](@ref)) overridden for the dynamic extent of the call, restoring the previous caps on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched. Unspecified keywords inherit from the currently active value, so nested overrides compose.

Useful to tighten the boundary around one batch of untrusted constraint strings, or to raise it for a single machine-generated constraint set, without affecting other concurrent work.

# Related

  - [`set_equation_limits!`](@ref)
  - [`EQUATION_LIMITS`](@ref)
  - [`parse_equation`](@ref)
"""
function with_equation_limits(f; max_length::Integer = EQUATION_LIMITS[].max_length,
                              max_depth::Integer = EQUATION_LIMITS[].max_depth)
    return with_config(f, EQUATION_LIMITS, EquationLimits(max_length, max_depth))
end
"""
    did_you_mean(name::AbstractString, candidates) -> String

Return a `" (did you mean \`X\`?)"`suffix naming the closest match to`name`among`candidates`, or `""` when no candidate reaches the global [`STRING_DISTANCE`](@ref) `min_score`threshold (or`candidates` is empty).

Used to enrich "variable not in asset universe" messages (see [`unknown_variable_msg`](@ref)) with a typo suggestion. The distance and threshold are read from the active [`STRING_DISTANCE`](@ref) config — global default via [`set_string_distance!`](@ref), task-scoped override via [`with_string_distance`](@ref); the threshold gating means a name legitimately absent from a meta-optimiser cluster/subset (no close neighbour) draws no suggestion.

# Related

  - [`STRING_DISTANCE`](@ref)
  - [`set_string_distance!`](@ref)
  - [`unknown_variable_msg`](@ref)
"""
function did_you_mean(name::AbstractString, candidates)
    if isempty(candidates)
        return ""
    end
    sd = STRING_DISTANCE[]
    match, _ = StringDistances.findnearest(name, candidates, sd.dist;
                                           min_score = sd.min_score)
    return isnothing(match) ? "" : " (did you mean `$(match)`?)"
end
"""
    unknown_variable_msg(v, nx, key; candidates = nx) -> String

Build the warning/error text for a constraint or view variable `v` that is absent from the asset universe `nx` (stored under `key`). Names the variable and the universe *size* only — never the full universe — and appends a [`did_you_mean`](@ref) suggestion when a close match exists.

`candidates` is the pool searched for the typo suggestion (default: the asset universe `nx`). Callers whose valid namespace is broader than the raw asset universe — e.g. [`group_to_val!`](@ref), where a key may name a *group* rather than an asset — pass a wider pool (asset names plus group/set keys) so the suggestion can name a mistyped group. The reported universe *size* is always `length(nx)` regardless of `candidates`.

Shared by [`get_linear_constraints`](@ref), Black-Litterman view generation, entropy-pooling view generation, and [`group_to_val!`](@ref) so the message (and its info-leak-safe shape) lives in exactly one place.

# Related

  - [`did_you_mean`](@ref)
  - [`empty_row_msg`](@ref)
"""
function unknown_variable_msg(v, nx, key; candidates = nx)
    return "variable `$(v)` not in asset universe ($(length(nx)) assets under key `$(key)`); term dropped" *
           did_you_mean(string(v), candidates)
end
"""
    empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint") -> String

Build the warning/error text for a parsed equation `eqn` whose every term missed the asset universe `nx` (stored under `key`), leaving an all-zero row that is dropped. Names the equation and the universe *size* only — never the full universe or the parsed struct. `noun` is `"constraint"` for linear constraints or `"view"` for Black-Litterman views.

Shared by [`get_linear_constraints`](@ref) and Black-Litterman view generation.

# Related

  - [`unknown_variable_msg`](@ref)
"""
function empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint")
    return "$(noun) `$(eqn)` matched no assets in the universe ($(length(nx)) assets under key `$(key)`); row dropped"
end
"""
    missing_group_assets_msg(group, missing_assets, nx, key) -> String

Build the warning/error text for a `group` that resolves in the asset sets but whose members
`missing_assets` are absent from the asset universe `nx` (stored under `key`). Names the group, the
offending member names (which are caller input, not internal state), and the universe *size* only —
never the full universe or the input value dictionary — and appends a [`did_you_mean`](@ref)
suggestion for the first missing member.

Shared by [`group_to_val!`](@ref) so the info-leak-safe message shape lives in exactly one place,
alongside [`unknown_variable_msg`](@ref) and [`empty_row_msg`](@ref).

# Related

  - [`unknown_variable_msg`](@ref)
  - [`empty_row_msg`](@ref)
  - [`did_you_mean`](@ref)
"""
function missing_group_assets_msg(group, missing_assets, nx, key)
    return "group `$(group)`: $(length(missing_assets)) member(s) not in asset universe " *
           "($(length(nx)) assets under key `$(key)`): $(missing_assets); dropped" *
           did_you_mean(string(first(missing_assets)), nx)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render the first line of an error for a log message, truncated to `max_line_length` characters (a trailing `…` marks the cut). Exceptions render via `showerror`, so the line carries the exception type and message; anything else renders via `repr`.

# Related

  - [`failed_solve_msg`](@ref)
"""
function first_error_line(err, max_line_length::Integer)
    s = err isa Exception ? sprint(showerror, err) : repr(err)
    line = String(first(split(s, '\n')))
    return length(line) <= max_line_length ? line : first(line, max_line_length) * "…"
end
"""
    failed_solve_msg(trials::AbstractDict; max_line_length::Integer = 200) -> String

Build the warning text for a JuMP model that no configured solver could solve satisfactorily (see `JuMPResult`). One line per failed stage of each solver trial: the solver name, the stage that failed (`set_optimizer`, `optimize!`, or `assert_is_solved_and_feasible`), and the first line of the error truncated to `max_line_length` characters — so a JuMP termination status stays visible.

Never interpolates the whole trials dictionary, the solver settings, or full exception payloads into the log; the raw data remains available on the returned `JuMPResult.trials`. This is the same info-leak-safe message discipline as [`unknown_variable_msg`](@ref) and its siblings. Solver names and stages are sorted so the message is deterministic.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`empty_row_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
  - [`first_error_line`](@ref)
"""
function failed_solve_msg(trials::AbstractDict; max_line_length::Integer = 200)
    msg = "Model could not be solved satisfactorily ($(length(trials)) solver trial(s))."
    for name in sort!(collect(keys(trials)); by = string)
        trial = trials[name]
        stages = trial isa AbstractDict ? trial : Dict{Symbol, Any}(:trial => trial)
        for stage in sort!(collect(keys(stages)); by = string)
            if stage === :settings
                continue
            end
            msg *= "\n  $(name): $(stage) → $(first_error_line(stages[stage], max_line_length))"
        end
    end
    return msg
end
"""
    PREFERENCE_DISTANCES

Enumerated allowlist mapping the names accepted by the `"suggestion_distance"` preference to their `StringDistances.StringDistance` objects. Membership and dispatch are one `Dict` — the same single-source-of-truth discipline as the equation parser's function allowlist (`docs/adr/0025-enumerated-parser-allowlist.md`): an unknown name fails closed at load with a typed error carrying a [`did_you_mean`](@ref) suggestion.

Supported names: `"levenshtein"`, `"damerau_levenshtein"`, `"jaro"`, `"jaro_winkler"`, `"ratcliff_obershelp"`.

# Related

  - [`apply_preferences!`](@ref)
  - [`set_string_distance!`](@ref)
"""
const PREFERENCE_DISTANCES = Dict{String, StringDistances.StringDistance}("levenshtein" =>
                                                                              StringDistances.Levenshtein(),
                                                                          "damerau_levenshtein" =>
                                                                              StringDistances.DamerauLevenshtein(),
                                                                          "jaro" =>
                                                                              StringDistances.Jaro(),
                                                                          "jaro_winkler" =>
                                                                              StringDistances.JaroWinkler(),
                                                                          "ratcliff_obershelp" =>
                                                                              StringDistances.RatcliffObershelp())
"""
    PREFERENCE_KEYS

The Preferences.jl keys read at package load to seed the global config defaults (see [`apply_preferences!`](@ref)):

  - `"equation_max_length"` / `"equation_max_depth"`: positive integers for [`EQUATION_LIMITS`](@ref).
  - `"suggestion_min_score"`: real number for the [`STRING_DISTANCE`](@ref) threshold.
  - `"suggestion_distance"`: a [`PREFERENCE_DISTANCES`](@ref) name for the [`STRING_DISTANCE`](@ref) metric.
  - `"compact_show"`: boolean or integer for [`COMPACT_SHOW`](@ref).

Preferences.jl offers no way to enumerate the keys a project has set, so a misspelled *key* cannot be detected and is silently ignored (the shipped default applies) — misspelled or invalid *values* under these keys fail closed at load.

# Related

  - [`apply_preferences!`](@ref)
"""
const PREFERENCE_KEYS = ("equation_max_length", "equation_max_depth",
                         "suggestion_min_score", "suggestion_distance", "compact_show")
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply load-time preference values to the global config defaults ([`EQUATION_LIMITS`](@ref), [`STRING_DISTANCE`](@ref), [`COMPACT_SHOW`](@ref)). Called by the package `__init__` with the [`PREFERENCE_KEYS`](@ref) values read via `Preferences.load_preference`; `nothing` values (unset preferences) are skipped and keep the shipped default.

Fails closed: an invalid value throws a typed `ArgumentError` naming the key and value, so the package refuses to load rather than silently running with a weaker cap than the one the project requested. Values are applied through the `set_*!` setters, so they receive the same validation as runtime calls.

To persist a configuration, put the keys in the active project's `LocalPreferences.toml`, e.g.:

```toml
[PortfolioOptimisers]
equation_max_length = 512
equation_max_depth = 64
suggestion_min_score = 0.8
suggestion_distance = "damerau_levenshtein"
compact_show = 4
```

# Related

  - [`PREFERENCE_KEYS`](@ref)
  - [`PREFERENCE_DISTANCES`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`set_string_distance!`](@ref)
  - [`set_compact_show!`](@ref)
"""
function apply_preferences!(prefs::AbstractDict{<:AbstractString, <:Any})
    ml = get(prefs, "equation_max_length", nothing)
    md = get(prefs, "equation_max_depth", nothing)
    if !(isnothing(ml) && isnothing(md))
        for (key, val) in ("equation_max_length" => ml, "equation_max_depth" => md)
            @argcheck(isnothing(val) || val isa Integer && !(val isa Bool) && val > 0,
                      ArgumentError("preference `$(key) = $(repr(val))` must be a positive integer."))
        end
        lim = @atomic EQUATION_LIMITS.default
        set_equation_limits!(; max_length = something(ml, lim.max_length),
                             max_depth = something(md, lim.max_depth))
    end
    ms = get(prefs, "suggestion_min_score", nothing)
    if !isnothing(ms)
        @argcheck(ms isa Real && !(ms isa Bool),
                  ArgumentError("preference `suggestion_min_score = $(repr(ms))` must be a real number."))
        set_string_distance!(; min_score = ms)
    end
    dn = get(prefs, "suggestion_distance", nothing)
    if !isnothing(dn)
        @argcheck(dn isa AbstractString,
                  ArgumentError("preference `suggestion_distance = $(repr(dn))` must be a string."))
        dist = get(PREFERENCE_DISTANCES, dn, nothing)
        if isnothing(dist)
            throw(ArgumentError("preference `suggestion_distance = $(repr(dn))` is not one of the $(length(PREFERENCE_DISTANCES)) supported distance names ($(join(sort!(collect(keys(PREFERENCE_DISTANCES))), ", ")))" *
                                did_you_mean(dn, collect(keys(PREFERENCE_DISTANCES)))))
        end
        set_string_distance!(; dist = dist)
    end
    cs = get(prefs, "compact_show", nothing)
    if !isnothing(cs)
        @argcheck(cs isa Bool || cs isa Integer,
                  ArgumentError("preference `compact_show = $(repr(cs))` must be a boolean or an integer."))
        set_compact_show!(cs)
    end
    return nothing
end
"""
    __init__()

Package load hook: reads the [`PREFERENCE_KEYS`](@ref) preferences of the active project via `Preferences.load_preference` and applies them to the global config defaults through [`apply_preferences!`](@ref). An invalid preference value fails closed — the package refuses to load — rather than silently running with a weaker cap than the one the project requested.
"""
function __init__()
    return apply_preferences!(Dict{String, Any}(key =>
                                                    Preferences.load_preference(@__MODULE__,
                                                                                key,
                                                                                nothing)
                                                for key in PREFERENCE_KEYS))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the single-line summary for a vector field rendered by [`@define_pretty_show`](@ref).

Returns a string of the form `"N-element Vector{Name}"`. A vector is treated as homogeneous when every element shares the same wrapper-type name (so elements that differ only in type parameters are still homogeneous): a homogeneous vector uses that common wrapper name, otherwise the wrapper of the element type, falling back to the raw `eltype` for `Union`s.

# Arguments

  - `val`: Non-empty vector whose elements all have a custom pretty-printing method.

# Returns

  - `summary::String`: Single-line `"N-element Vector{Name}"` summary.

# Related

  - [`@define_pretty_show`](@ref)
  - [`pretty_show_vector_element`](@ref)
  - [`pretty_show_vector_body`](@ref)
"""
function pretty_show_vector_summary(val::AbstractVector)
    names = [string(Base.typename(typeof(v)).wrapper) for v in val]
    et = eltype(val)
    tname = if allequal(names)
        first(names)
    else
        (et isa Union ? string(et) : string(Base.typename(et).wrapper))
    end
    return "$(length(val))-element Vector{$(tname)}"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render a single vector element as a collapsed line for [`@define_pretty_show`](@ref).

Every element of a listed vector is shown as just its wrapper-type name. When the element is a struct with fields, a trailing `" ⋯"` marks it as a collapsed struct (consistent with how an over-budget struct field collapses to `Name ⋯`); fieldless elements are left bare.

# Related

  - [`@define_pretty_show`](@ref)
  - [`pretty_show_vector_summary`](@ref)
  - [`pretty_show_vector_body`](@ref)
"""
function pretty_show_vector_element(@nospecialize(v))
    s = string(Base.typename(typeof(v)).wrapper)
    return isempty(fieldnames(typeof(v))) ? s : s * " ⋯"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply the shared collapse budget to the per-element lines of a vector rendered by [`@define_pretty_show`](@ref).

The budget comes from [`compact_show_budget`](@ref), so vector truncation honours the same `:limit` gate, global [`set_compact_show!`](@ref) setting, and per-call `:po_compact` override as struct collapsing. When the budget is `nothing` (disabled, unlimited output, or override-off) every line is returned. Otherwise, when the listing exceeds the budget it is split head-and-tail, mirroring how `Base` truncates long arrays, with a single `"⋮"` line marking the elision.

# Arguments

  - `io`: Output stream; drives the budget via [`compact_show_budget`](@ref).
  - `lines`: Per-element display strings from [`pretty_show_vector_element`](@ref).

# Returns

  - `body::Vector{String}`: Lines to print, possibly truncated with a `"⋮"` separator.

# Related

  - [`@define_pretty_show`](@ref)
  - [`compact_show_budget`](@ref)
  - [`pretty_show_vector_element`](@ref)
"""
function pretty_show_vector_body(io::IO, lines::AbstractVector{<:AbstractString})
    budget = compact_show_budget(io)
    n = length(lines)
    if isnothing(budget) || n <= budget
        return lines
    end
    nhead = cld(budget, 2)
    ntail = budget - nhead
    return vcat(lines[1:nhead], "⋮", lines[(n - ntail + 1):n])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Default method indicating whether a type has a custom pretty-printing `show` method.

Overloading this method to return `true` indicates that type already has a custom pretty-printing method.

# Arguments

  - `::Any`: Any type.

# Returns

  - `flag::Bool`: `false` by default, indicating no custom pretty-printing method.

# Related

  - [`@define_pretty_show`](@ref)
"""
has_pretty_show_method(::Any)::Bool = false
has_pretty_show_method(::JuMP.Model)::Bool = true
has_pretty_show_method(::Clustering.Hclust)::Bool = true
has_pretty_show_method(::Clustering.KmeansResult)::Bool = true
@define_pretty_show(Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult})
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom exception types in `PortfolioOptimisers.jl`.

All error types specific to `PortfolioOptimisers.jl` should be subtypes of `PortfolioOptimisersError`.

# Related

  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
abstract type PortfolioOptimisersError <: Exception end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IsNothingError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNothingError("Input data must not be nothing"))
ERROR: IsNothingError: Input data must not be nothing
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
@concrete struct IsNothingError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly empty.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IsEmptyError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsEmptyError("Input array must not be empty"))
ERROR: IsEmptyError: Input array must not be empty
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
@concrete struct IsEmptyError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly non-finite (e.g., contains `NaN` or `Inf`).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IsNonFiniteError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNonFiniteError("Input array contains non-finite values"))
ERROR: IsNonFiniteError: Input array contains non-finite values
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
"""
@concrete struct IsNonFiniteError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when a [`@forward_properties`](@ref) nested path cannot be descended because an intermediate node is `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PropertyPathError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(PropertyPathError("cannot descend path `sol.w` on `JuMPOptimisationResult`: intermediate `sol` is `nothing`"))
ERROR: PropertyPathError: cannot descend path `sol.w` on `JuMPOptimisationResult`: intermediate `sol` is `nothing`
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`@forward_properties`](@ref)
"""
@concrete struct PropertyPathError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Print human-readable representation of `PortfolioOptimisersError` subtypes to `io`, stripping parametric type suffixes.
"""
function Base.showerror(io::IO, err::PortfolioOptimisersError)
    name = string(typeof(err))
    name = name[1:(findfirst(x -> (x == '{' || x == '('), name) - 1)]
    return print(io, "$name: $(err.msg)")
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Make estimators, algorithms, and results behave as length-1 iterables, returning the object itself on the first iteration and `nothing` thereafter.
"""
function Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                 <:AbstractResult}, state = 1)
    return state > 1 ? nothing : (obj, state + 1)
end
Base.length(::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}) = 1
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Index into estimators, algorithms, and results as length-1 containers. Only index `1` is valid; any other index throws `BoundsError`.
"""
function Base.getindex(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                  <:AbstractResult}, i::Int)
    return i == 1 ? obj : throw(BoundsError(obj, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of numeric types or JuMP scalar types.

# Related

  - [`VecInt`](@ref)
  - [`MatNum`](@ref)
  - [`JuMP.AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.JuMP.AbstractJuMPScalar)
"""
const VecNum = AbstractVector{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of integer types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const VecInt = AbstractVector{<:Integer}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract matrix of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum = AbstractMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract array of numeric types or JuMP scalar types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const ArrNum = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}}
"""
    const VecNum_MatNum = Union{<:VecNum, <:MatNum}

Alias for a union of a numeric type or an abstract matrix of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const VecNum_MatNum = Union{<:VecNum, <:MatNum}
"""
    const Num_VecNum = Union{<:Number, <:VecNum}

Alias for a union of a numeric type or an abstract vector of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`ArrNum`](@ref)
"""
const Num_VecNum = Union{<:Number, <:VecNum}
"""
    const Func_VecNum = Union{<:Function, <:VecNum}

Alias for a union of a function and a vector of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`Func_Num_VecNum`](@ref)
"""
const Func_VecNum = Union{<:Function, <:VecNum}
"""
    const Func_Num_VecNum = Union{<:Number, <:Func_VecNum}

Alias for a union of a function type or a numeric type or an abstract vector of numeric types.

# Related

  - [`Func_VecNum`](@ref)
"""
const Func_Num_VecNum = Union{<:Number, <:Func_VecNum}
"""
    const Num_ArrNum = Union{<:Number, <:ArrNum}

Alias for a union of a numeric type or an abstract array of numeric types.

# Related

  - [`ArrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Num_ArrNum = Union{<:Number, <:ArrNum}
"""
$(DocStringExtensions.TYPEDEF)

Alias for a pair consisting of an abstract string and a numeric type.

# Related

  - [`DictStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const PairStrNum = Pair{<:AbstractString, <:Number}
"""
$(DocStringExtensions.TYPEDEF)

Alias for a key type used in grid search cross-validation, which can be an abstract string, an expression, a symbol, a composed function, an accessor lens, or an integer (a step position when tuning a `Pipeline`).

# Related

  - [`PairGSCV`](@ref)
  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const GSCVKey = Union{<:AbstractString, Expr, Symbol, <:ComposedFunction,
                      <:Accessors.PropertyLens, <:Accessors.IndexLens, <:Integer}
"""
$(DocStringExtensions.TYPEDEF)

Alias for a value type used in randomised search cross-validation, which can be an abstract vector or a distribution.

# Related

  - [`PairGSCV`](@ref)
  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const RSCVVal = Union{<:AbstractVector, <:Distributions.Distribution}
"""
$(DocStringExtensions.TYPEDEF)

Alias for a pair consisting of an abstract string and an abstract vector.

# Related

  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const PairGSCV = Pair{<:GSCVKey, <:AbstractVector}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract dictionary with string keys and numeric values.

# Related

  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
"""
const DictStrNum = AbstractDict{<:AbstractString, <:Number}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract dictionary with string keys and abstract vector values.

# Related

  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const DictGSCV = AbstractDict{<:GSCVKey, <:AbstractVector}
"""
    const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}

Alias for a union of a dictionary with string keys and numeric values, or a vector of string-number pairs.

# Related

  - [`DictStrNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`EstValType`](@ref)
"""
const MultiEstValType = Union{<:DictStrNum, <:AbstractVector{<:PairStrNum}}
"""
    const MultiGSCVValType = Union{<:DictGSCV, <:AbstractVector{<:PairGSCV}}

Alias for a union of an abstract dictionary with string keys and abstract vector values, or a vector of string-vector pairs.

# Related

  - [`DictGSCV`](@ref)
  - [`PairGSCV`](@ref)
  - [`VecMultiGSCVValType`](@ref)
  - [`MultiGSCVValType_VecMultiGSCVValType`](@ref)
"""
const MultiGSCVValType = Union{<:DictGSCV, <:AbstractVector{<:PairGSCV}}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of `MultiGSCVValType` elements.

# Related

  - [`DictGSCV`](@ref)
  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
  - [`MultiGSCVValType_VecMultiGSCVValType`](@ref)
"""
const VecMultiGSCVValType = AbstractVector{<:MultiGSCVValType}
"""
    const MultiGSCVValType_VecMultiGSCVValType = Union{<:MultiGSCVValType,
                                                       <:VecMultiGSCVValType}

Alias for a union of `MultiGSCVValType` and `VecMultiGSCVValType` elements.

# Related

  - [`DictGSCV`](@ref)
  - [`PairGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
  - [`VecMultiGSCVValType`](@ref)
"""
const MultiGSCVValType_VecMultiGSCVValType = Union{<:MultiGSCVValType,
                                                   <:VecMultiGSCVValType}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimator value algorithm types in `PortfolioOptimisers.jl`.

Subtypes of `AbstractEstimatorValueAlgorithm` implement algorithms for computing constraint result values. These are used to extend or modify the behavior of estimators in a composable and modular fashion.

# Interfaces

In order to implement a new estimator value algorithm which will work seamlessly with the library, subtype `AbstractEstimatorValueAlgorithm` with all necessary parameters struct, and implement the following method:

  - `estimator_to_val(alg::AbstractEstimatorValueAlgorithm, sets::AssetSets, val::Option{<:Number} = nothing, key::Option{<:AbstractString} = nothing; datatype::DataType = Float64, strict::Bool = false) -> Num_VecNum`: Converts an estimator value dictionary to a numeric or vector of numeric value. Usually this should compute some version of:
      + `val = ifelse(isnothing(val), <default value use with datatype element type>, val)`: Computes the default value to use if `val` is `nothing`.
      + `nx = sets.dict[ifelse(isnothing(key), sets.key, key)]`: Gets the universe to use for mapping values to features.

## Arguments

  - `alg`: Concrete subtype of `AbstractEstimatorValueAlgorithm`.
  - $(arg_dict[:sets])
  - $(arg_dict[:val])
  - $(arg_dict[:ekey])
  - $(arg_dict[:datatype])
  - $(arg_dict[:strict])

# Returns

  - `val::Num_VecNum`: The numeric or vector of numeric value.

# Examples

We can create a dummy estimator value algorithm as follows:

```jldoctest
julia> struct MyIncreasingValue <: PortfolioOptimisers.AbstractEstimatorValueAlgorithm end

julia> function PortfolioOptimisers.estimator_to_val(alg::MyIncreasingValue, sets::AssetSets,
                                                     val::PortfolioOptimisers.Option{<:Number} = nothing,
                                                     key::PortfolioOptimisers.Option{<:AbstractString} = nothing;
                                                     datatype::DataType = Float64,
                                                     strict::Bool = false)
           val = ifelse(isnothing(val), zero(datatype), val)
           nx = sets.dict[ifelse(isnothing(key), sets.key, key)]
           arr = ((1 - val):(length(nx) - val))
           return arr
       end

julia> sets = AssetSets(; dict = Dict("nx" => ["sha", "bis", "man"]))
AssetSets
   key ┼ String: "nx"
  ukey ┼ String: "ux"
  dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["sha", "bis", "man"])

julia> estimator_to_val(MyIncreasingValue(), sets)
1.0:1.0:3.0
```

# Related

  - [`EstValType`](@ref)
  - [`estimator_to_val`](@ref)
"""
abstract type AbstractEstimatorValueAlgorithm <: AbstractAlgorithm end
"""
    const EstValType = Union{<:Num_VecNum, <:MatNum, <:PairStrNum, <:MultiEstValType,
                             <:AbstractEstimatorValueAlgorithm}

Alias for a union of numeric, vector of numeric, matrix of numeric, string-number pair, or multi-estimator value types.

# Related

  - [`Num_VecNum`](@ref)
  - [`PairStrNum`](@ref)
  - [`MultiEstValType`](@ref)
  - [`AbstractEstimatorValueAlgorithm`](@ref)
"""
const EstValType = Union{<:Num_VecNum, <:MatNum, <:PairStrNum, <:MultiEstValType,
                         <:AbstractEstimatorValueAlgorithm}
"""
    const Str_Expr = Union{<:AbstractString, Expr}

Alias for a union of abstract string or Julia expression.

# Related

  - [`VecStr_Expr`](@ref)
  - [`EqnType`](@ref)
"""
const Str_Expr = Union{<:AbstractString, Expr}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of strings or Julia expressions.

# Related

  - [`Str_Expr`](@ref)
  - [`EqnType`](@ref)
"""
const VecStr_Expr = AbstractVector{<:Str_Expr}
"""
    const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr,
                          <:AbstractEstimatorValueAlgorithm}

Alias for a union of string, Julia expression, or vector of strings/expressions.

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr,
                      <:AbstractEstimatorValueAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of numeric vectors.

# Related

  - [`VecNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const VecVecNum = AbstractVector{<:VecNum}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of integer vectors.

# Related

  - [`VecInt`](@ref)
"""
const VecVecInt = AbstractVector{<:VecInt}
"""
    const VecInt_VecVecInt = Union{<:VecInt, <:VecVecInt}

Alias for a union of an abstract vector of integers or an abstract vector of integer vectors.

# Related

  - [`VecInt`](@ref)
  - [`VecVecInt`](@ref)
"""
const VecInt_VecVecInt = Union{<:VecInt, <:VecVecInt}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of abstract vector of integer vectors.

# Related

  - [`VecVecInt`](@ref)
"""
const VecVecVecInt = AbstractVector{<:VecVecInt}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of numeric matrices.

# Related

  - [`MatNum`](@ref)
  - [`VecNum`](@ref)
"""
const VecMatNum = AbstractVector{<:MatNum}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of strings.

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const VecStr = AbstractVector{<:AbstractString}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of pairs.

# Related

  - [`PairStrNum`](@ref)
"""
const VecPair = AbstractVector{<:Pair}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of JuMP scalar types.

# Related

  - [`VecNum`](@ref)
"""
const VecJuMPScalar = Union{<:AbstractVector{<:JuMP.AbstractJuMPScalar}}
"""
    const Option{T} = Union{Nothing, T}

Alias for an optional value of type `T`, which may be `nothing`.

# Related

  - [`EstValType`](@ref)
"""
const Option{T} = Union{Nothing, T}
"""
    const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}

Alias for a union of a numeric matrix or a vector of numeric matrices.

# Related

  - [`MatNum`](@ref)
  - [`VecMatNum`](@ref)
"""
const MatNum_VecMatNum = Union{<:MatNum, <:VecMatNum}
"""
    const Int_VecInt = Union{<:Integer, <:VecInt}

Alias for a union of an integer or a vector of integers.

# Related

  - [`VecInt`](@ref)
"""
const Int_VecInt = Union{<:Integer, <:VecInt}
"""
    const VecNum_VecVecNum = Union{<:VecNum, <:VecVecNum}

Alias for a union of a numeric vector or a vector of numeric vectors.

# Related

  - [`VecNum`](@ref)
  - [`VecVecNum`](@ref)
"""
const VecNum_VecVecNum = Union{<:VecNum, <:VecVecNum}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of date or time types.

# Related

  - [`VecNum`](@ref)
  - [`VecStr`](@ref)
"""
const VecDate = AbstractVector{<:Dates.AbstractTime}
"""
    const Dict_Vec = Union{<:AbstractDict, <:AbstractVector}

Alias for a union of an abstract dictionary or an abstract vector.

# Related

  - [`DictStrNum`](@ref)
  - [`VecNum`](@ref)
"""
const Dict_Vec = Union{<:AbstractDict, <:AbstractVector}
"""
    const Sym_Str = Union{Symbol, <:AbstractString}

Alias for a union of a symbol or an abstract string.

# Related

  - [`VecStr`](@ref)
"""
const Sym_Str = Union{Symbol, <:AbstractString}
"""
    const Str_Vec = Union{<:AbstractString, <:AbstractVector}

Alias for a union of an abstract string or an abstract vector.

# Related

  - [`VecStr`](@ref)
  - [`Str_Expr`](@ref)
"""
const Str_Vec = Union{<:AbstractString, <:AbstractVector}
"""
    const ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}

Union type for observation weights accepted by estimators.

Accepts either a [`DynamicAbstractWeights`](@ref) subtype (weights computed from data at fit time) or a `StatsBase.AbstractWeights` instance (pre-computed numeric weights).

# Related

  - [`DynamicAbstractWeights`](@ref)
  - [`get_observation_weights`](@ref)
"""
const ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}
"""
    get_observation_weights(
        w::Option{<:ObsWeights},
        args...;
        kwargs...
    ) -> Option{<:VecNum}

Get the observation weights for statistical estimation.

# Arguments

  - $(arg_dict[:oow])
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

# Returns

  - `w::Option{<:VecNum}`: The observation weights, or `nothing` for when `w` is [`DynamicAbstractWeights`](@ref) or `nothing`.

# Related

  - [`ObsWeights`](@ref)
"""
function get_observation_weights(::Option{<:DynamicAbstractWeights}, args...; kwargs...)
    return nothing
end
function get_observation_weights(w::VecNum, args...; kwargs...)
    return w
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `val` is non-empty.

No-op for `Pair` and `Number` inputs; emptiness does not apply to scalars.

# Arguments

  - `val`: Container to check; one of `AbstractDict`, `VecPair`, or `ArrNum`.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Related

  - [`assert_finite`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_gt0`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty_finite_val`](@ref)
"""
function assert_nonempty(val::Union{<:AbstractDict, <:VecPair, <:ArrNum},
                         sym::Sym_Str = :val)::Nothing
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($sym) must hold. Got\n!isempty($sym) => $(isempty(val))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op overload of [`assert_nonempty`](@ref) for scalar inputs.

Emptiness does not apply to `Pair` or `Number` values.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
"""
function assert_nonempty(::Union{<:Pair, <:Number}, ::Sym_Str = :val)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `val` contains at least one finite element.

Dispatches on the input type:

  - `AbstractDict`: `any(isfinite, values(val))`.
  - `VecPair`: `any(isfinite, getindex.(val, 2))`.
  - `ArrNum`: `any(isfinite, val)`.
  - `Pair`: `isfinite(val[2])`.
  - `Number`: `isfinite(val)`.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_gt0`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty_finite_val`](@ref)
"""
function assert_finite(val::AbstractDict, sym::Sym_Str = :val)::Nothing
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($sym)) must hold. Got\nany(isfinite, values($sym)) => $(any(isfinite, values(val)))"))
    return nothing
end
function assert_finite(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($sym, 2)) must hold. Got\nany(isfinite, getindex.($sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    return nothing
end
function assert_finite(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $sym) must hold. Got\nany(isfinite, $sym) => $(any(isfinite, val))"))
    return nothing
end
function assert_finite(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($sym[2]) must hold. Got\nisfinite($sym[2]) => $(isfinite(val[2]))"))
    return nothing
end
function assert_finite(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(isfinite(val),
              DomainError("isfinite($sym) must hold. Got\nisfinite($sym) => $(isfinite(val))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that all elements of `val` are non-negative (`>= 0`).

Dispatches on the input type:

  - `AbstractDict`: `all(x -> 0 <= x, values(val))`.
  - `VecPair`: `all(x -> 0 <= x[2], val)`.
  - `ArrNum`: `all(x -> 0 <= x, val)`.
  - `Pair`: `0 <= val[2]`.
  - `Number`: `0 <= val`.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_gt0`](@ref)
  - [`assert_nonempty_nonneg_finite_val`](@ref)
"""
function assert_nonneg(val::AbstractDict, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) <= x, values(val)),
              DomainError("all(x -> 0 <= x, values($sym)) must hold. Got\nall(x -> 0 <= x, values($sym)) => $(all(x -> zero(x) <= x, values(val)))"))
    return nothing
end
function assert_nonneg(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x[2]) <= x[2], val),
              DomainError("all(x -> 0 <= x[2], $sym) must hold. Got\nall(x -> 0 <= x[2], $sym) => $(all(x -> zero(x[2]) <= x[2], val))"))
    return nothing
end
function assert_nonneg(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) <= x, val),
              DomainError("all(x -> 0 <= x, $sym) must hold. Got\nall(x -> 0 <= x, $sym) => $(all(x -> zero(x) <= x, val))"))
    return nothing
end
function assert_nonneg(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val[2]) <= val[2],
              DomainError("0 <= $sym[2] must hold. Got\n$sym[2] => $(val[2])"))
    return nothing
end
function assert_nonneg(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) <= val, DomainError("0 <= $sym must hold. Got\n$sym => $(val)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that all elements of `val` are strictly positive (`> 0`).

Dispatches on the input type:

  - `AbstractDict`: `all(x -> 0 < x, values(val))`.
  - `VecPair`: `all(x -> 0 < x[2], val)`.
  - `ArrNum`: `all(x -> 0 < x, val)`.
  - `Pair`: `0 < val[2]`.
  - `Number`: `0 < val`.

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Related

  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
"""
function assert_gt0(val::AbstractDict, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) < x, values(val)),
              DomainError("all(x -> 0 < x, values($sym)) must hold. Got\nall(x -> 0 < x, values($sym)) => $(all(x -> zero(x) < x, values(val)))"))
    return nothing
end
function assert_gt0(val::VecPair, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x[2]) < x[2], val),
              DomainError("all(x -> 0 < x[2], $sym) must hold. Got\nall(x -> 0 < x[2], $sym) => $(all(x -> zero(x[2]) < x[2], val))"))
    return nothing
end
function assert_gt0(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(x -> zero(x) < x, val),
              DomainError("all(x -> 0 < x, $sym) must hold. Got\nall(x -> 0 < x, $sym) => $(all(x -> zero(x) < x, val))"))
    return nothing
end
function assert_gt0(val::Pair, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val[2]) < val[2],
              DomainError("0 < $sym[2] must hold. Got\n$sym[2] => $(val[2])"))
    return nothing
end
function assert_gt0(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) < val, DomainError("0 < $sym must hold. Got\n$sym => $(val)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `val` lies strictly inside the open unit interval (`0 < val < 1`).

# Arguments

  - `val`: Value to check.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Related

  - [`assert_nonneg`](@ref)
  - [`assert_gt0`](@ref)
"""
function assert_unit_interval(val::Number, sym::Sym_Str = :val)::Nothing
    @argcheck(zero(val) < val < one(val),
              DomainError("0 < $sym < 1 must hold. Got\n$sym => $(val)"))
    return nothing
end
"""
    assert_nonempty_nonneg_finite_val(
        val::Union{<:AbstractDict, <:VecPair, <:ArrNum, Pair, Number},
        val_sym::Union{Symbol,<:AbstractString} = :val
    )
    assert_nonempty_nonneg_finite_val(args...)

Validate that the input value is non-empty, non-negative and finite.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Returns

  - `nothing`.

# Details

  - `val`: Input value to validate.

      + `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`, `all(x -> x >= 0, values(val))`.
      + `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`, `all(x -> x[2] >= 0, val)`.
      + `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
      + `::Pair`: `isfinite(val[2])` and `val[2] >= 0`.
      + `::Number`: `isfinite(val)` and `val >= 0`.
      + `args...`: Always passes.

# Related

  - [`assert_nonempty_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_nonneg`](@ref)
"""
function assert_nonempty_nonneg_finite_val(val::Union{<:AbstractDict, <:VecPair, <:ArrNum,
                                                      <:Pair, <:Number},
                                           val_sym::Sym_Str = :val)::Nothing
    assert_nonempty(val, val_sym)
    assert_finite(val, val_sym)
    assert_nonneg(val, val_sym)
    return nothing
end
function assert_nonempty_nonneg_finite_val(args...)::Nothing
    return nothing
end
"""
    assert_nonempty_gt0_finite_val(
        val::Union{<:AbstractDict, <:VecPair, <:ArrNum, Pair, Number},
        val_sym::Union{Symbol,<:AbstractString} = :val
    )
    assert_nonempty_gt0_finite_val(args...)

Validate that the input value is non-empty, greater than zero, and finite.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Returns

  - `nothing`.

# Details

  - `val`: Input value to validate.

      + `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`, `all(x -> x > 0, values(val))`.
      + `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`, `all(x -> x[2] > 0, val)`.
      + `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x > 0, val)`.
      + `::Pair`: `isfinite(val[2])` and `val[2] > 0`.
      + `::Number`: `isfinite(val)` and `val > 0`.
      + `args...`: Always passes.

# Related

  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_finite_val`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
  - [`assert_gt0`](@ref)
"""
function assert_nonempty_gt0_finite_val(val::Union{<:AbstractDict, <:VecPair, <:ArrNum,
                                                   <:Pair, <:Number},
                                        val_sym::Sym_Str = :val)::Nothing
    assert_nonempty(val, val_sym)
    assert_finite(val, val_sym)
    assert_gt0(val, val_sym)
    return nothing
end
function assert_nonempty_gt0_finite_val(args...)::Nothing
    return nothing
end
"""
    assert_nonempty_finite_val(
        val::Union{<:AbstractDict, <:VecPair, <:ArrNum, Pair, Number},
        val_sym::Union{Symbol,<:AbstractString} = :val
    )
    assert_nonempty_finite_val(args...)

Validate that the input value is non-empty and finite.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Returns

  - `nothing`.

# Details

  - `val`: Input value to validate.

      + `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`.
      + `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`.
      + `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`.
      + `::Pair`: `isfinite(val[2])`.
      + `::Number`: `isfinite(val)`.
      + `args...`: Always passes.

# Related

  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_finite`](@ref)
"""
function assert_nonempty_finite_val(val::Union{<:AbstractDict, <:VecPair, <:ArrNum, <:Pair,
                                               <:Number}, val_sym::Sym_Str = :val)::Nothing
    assert_nonempty(val, val_sym)
    assert_finite(val, val_sym)
    return nothing
end
function assert_nonempty_finite_val(args...)::Nothing
    return nothing
end
"""
    assert_matrix_issquare(X::MatNum, X_sym::Symbol = :X)

Assert that the input matrix is square.

# Arguments

  - `X`: Input matrix to validate.
  - `X_sym`: Symbolic name used in error messages.

# Returns

  - `nothing`.

# Validation

  - `size(X, 1) == size(X, 2)`.

# Details

  - Throws `DimensionMismatch` if the check fails.

# Related

  - [`MatNum`](@ref)
"""
function assert_matrix_issquare(X::MatNum, X_sym::Symbol = :X)::Nothing
    @argcheck(size(X, 1) == size(X, 2),
              DimensionMismatch("size($X_sym, 1) == size($X_sym, 2) must hold. Got\nsize($X_sym, 1) => $(size(X, 1))\nsize($X_sym, 2) => $(size(X, 2))."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a composite result containing a vector and a scalar in `PortfolioOptimisers.jl`.

Encapsulates a vector and a scalar value, commonly used for storing results that combine both types of data (e.g., weighted statistics, risk measures).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VecScalar(;
        v::VecNum,
        s::Number
    ) -> VecScalar

Keywords correspond to the struct's fields.

## Validation

  - `v`: `!isempty(v)` and `all(isfinite, v)`.
  - `s`: `isfinite(s)`.

# Examples

```jldoctest
julia> VecScalar([1.0, 2.0, 3.0], 4.2)
VecScalar
  v ┼ Vector{Float64}: [1.0, 2.0, 3.0]
  s ┴ Float64: 4.2
```

# Related

  - [`AbstractResult`](@ref)
  - [`VecNum`](@ref)
"""
@concrete struct VecScalar <: AbstractResult
    """
    Vector component.
    """
    v
    """
    Scalar component.
    """
    s
    function VecScalar(v::VecNum, s::Number)
        assert_nonempty_finite_val(v, :v)
        assert_nonempty_finite_val(s, :s)
        return new{typeof(v), typeof(s)}(v, s)
    end
end
function VecScalar(; v::VecNum, s::Number)
    return VecScalar(v, s)
end
"""
    const Num_VecNum_VecScalar = Union{<:Num_VecNum, <:VecScalar}

Alias for a union of a numeric type, a vector of numeric types, or a `VecScalar` result.

# Related

  - [`Num_VecNum`](@ref)
  - [`VecScalar`](@ref)
"""
const Num_VecNum_VecScalar = Union{<:Num_VecNum, <:VecScalar}
"""
    const Num_ArrNum_VecScalar_DynWeights = Union{<:Num_ArrNum, <:VecScalar, <:DynamicAbstractWeights}

Alias for a union of a numeric type, an array of numeric types, or a `VecScalar` result.

# Related

  - [`Num_ArrNum`](@ref)
  - [`VecScalar`](@ref)
"""
const Num_ArrNum_VecScalar_DynWeights = Union{<:Num_ArrNum, <:VecScalar,
                                              <:DynamicAbstractWeights}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all norm-based error algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing norm-based error algorithms (such as second-order cone or norm-one error) should be subtypes of `NormError`.

# Related

  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`L1Norm`](@ref)
"""
abstract type NormError <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Second-order cone (SOC) norm-based error formulation.

`L2Norm` implements a norm-based error formulation using the Euclidean (L2) norm, scaled by the square root of the number of assets minus the degrees of freedom (`ddof`). This is commonly used for error constraints and objectives in portfolio optimisation.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2}{\\sqrt{T - d}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b})``: L2-norm error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L2Norm(;
        ddof::Integer = 1
    ) -> L2Norm

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> L2Norm()
L2Norm
  ddof ┴ Int64: 1
```

# Related

  - [`NormError`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`L1Norm`](@ref)
  - [`norm_error`](@ref)
"""
@concrete struct L2Norm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    function L2Norm(ddof::Integer)::L2Norm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function L2Norm(; ddof::Integer = 1)::L2Norm
    return L2Norm(ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

Second-order cone (SOC) squared norm-based error formulation.

`SquaredL2Norm` implements a norm-based error formulation using the squared Euclidean (L2) norm, scaled by the number of assets minus the degrees of freedom (`ddof`). This is commonly used for norm error constraints and objectives in portfolio optimisation where squared error is preferred.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2^2}{T - d}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b})``: Squared L2-norm error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SquaredL2Norm(;
        ddof::Integer = 1,
    ) -> SquaredL2Norm

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> SquaredL2Norm()
SquaredL2Norm
  ddof ┴ Int64: 1
```

# Related

  - [`NormError`](@ref)
  - [`L2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`norm_error`](@ref)
"""
@concrete struct SquaredL2Norm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    function SquaredL2Norm(ddof::Integer)::SquaredL2Norm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function SquaredL2Norm(; ddof::Integer = 1)::SquaredL2Norm
    return SquaredL2Norm(ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

Norm-one (NOC) error formulation.

`L1Norm` implements a norm-based error formulation using the L1 (norm-one) distance between portfolio and benchmark weights. This is commonly used for error constraints and objectives in portfolio optimisation where sparsity or absolute deviations are preferred.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_1}{T}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b})``: L1-norm error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T]) When ``T`` is not provided the denominator is 1.

# Constructors

    L1Norm() -> L1Norm

# Examples

```jldoctest
julia> L1Norm()
L1Norm()
```

# Related

  - [`NormError`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`norm_error`](@ref)
"""
struct L1Norm <: NormError end
"""
$(DocStringExtensions.TYPEDEF)

L-p norm error estimator.

Computes the Lp-norm of the difference between portfolio and benchmark returns: ``\\lvert\\mathbf{X} \\boldsymbol{w} - \\boldsymbol{b}\\rvert_p``.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_p}{(T - d)^{1/p}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b})``: Lp-norm error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.
  - ``p``: Norm order.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LpNorm(; p::Number = 3, ddof::Integer = 0) -> LpNorm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> LpNorm()
LpNorm
     p ┼ Int64: 3
  ddof ┴ Int64: 0
```

# Related

  - [`NormError`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`LInfNorm`](@ref)
"""
@concrete struct LpNorm <: NormError
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:ddof])
    """
    ddof
    function LpNorm(p::Number, ddof::Integer)::LpNorm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(p), typeof(ddof)}(p, ddof)
    end
end
function LpNorm(; p::Number = 3, ddof::Integer = 0)::LpNorm
    return LpNorm(p, ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

L-infinity norm (maximum absolute deviation) error estimator.

Computes the L∞-norm (maximum absolute deviation) of the difference between portfolio and benchmark returns.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_\\infty}{T - d}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b})``: L∞-norm error. `pos = true` uses ``+\\infty``, `pos = false` uses ``-\\infty``.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LInfNorm(; ddof::Integer = 0, pos::Bool = true) -> LInfNorm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> LInfNorm()
LInfNorm
  ddof ┼ Int64: 0
   pos ┴ Bool: true
```

# Related

  - [`NormError`](@ref)
  - [`LpNorm`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
"""
@concrete struct LInfNorm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    """
    $(field_dict[:pos])
    """
    pos
    function LInfNorm(ddof::Integer, pos::Bool)::LInfNorm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof), typeof(pos)}(ddof, pos)
    end
end
function LInfNorm(; ddof::Integer = 0, pos::Bool = true)::LInfNorm
    return LInfNorm(ddof, pos)
end
"""
    norm_error(f::L2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::SquaredL2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(::L1Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LpNorm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LInfNorm, a, b, T::Option{<:Number} = nothing)

Compute the norm-based tracking error between portfolio and benchmark weights.

`norm_error` computes the tracking error using either the Euclidean (L2) norm for [`L2Norm`](@ref), squared Euclidean (L2) norm for [`SquaredL2Norm`](@ref), or the L1 (norm-one) distance for [`L1Norm`](@ref). The error is optionally scaled by the number of assets and degrees of freedom for SOC, or by the number of assets for NOC.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2}{\\sqrt{T - d}}\\,, \\\\
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_2^2}{T - d}\\,, \\\\
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_1}{T}\\,, \\\\
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_p}{(T-d)^{1/p}}\\,, \\\\
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_\\infty}{T - d}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`.
  - ``p``: Norm order.

# Arguments

  - `f`: Tracking formulation algorithm.
  - `a`: Portfolio weights.
  - `b`: Benchmark weights.
  - `T`: Optional number of observations.

# Returns

  - `err::Number`: Norm-based tracking error.

# Details

  - For `L2Norm`, computes `LinearAlgebra.norm(a - b, 2) / sqrt(T - f.ddof)` if `T` is not `nothing`, else unscaled.
  - For `SquaredL2Norm`, computes `LinearAlgebra.norm(a - b, 2)^2 / (T - f.ddof)` if `T` is not `nothing`, else unscaled.
  - For `L1Norm`, computes `LinearAlgebra.norm(a - b, 1) / T` if `T` is not `nothing`, else unscaled.

# Examples

```jldoctest
julia> PortfolioOptimisers.norm_error(L2Norm(), [0.5, 0.5], [0.6, 0.4], 2)
0.14142135623730948

julia> PortfolioOptimisers.norm_error(L1Norm(), [0.5, 0.5], [0.6, 0.4], 2)
0.09999999999999998
```

# Related

  - [`L2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`NormError`](@ref)
  - [`Option`](@ref)
"""
function norm_error(f::L2Norm, a, b, T::Option{<:Number} = nothing)
    factor = isnothing(T) ? 1 : sqrt(T - f.ddof)
    return LinearAlgebra.norm(a - b, 2) / factor
end
function norm_error(f::L2Norm, a, T::Option{<:Number} = nothing)
    factor = isnothing(T) ? 1 : sqrt(T - f.ddof)
    return LinearAlgebra.norm(a, 2) / factor
end
function norm_error(::Nothing, a, T::Option{<:Number} = nothing)
    factor = isnothing(T) ? 1 : sqrt(T)
    return LinearAlgebra.norm(a, 2) / factor
end
function norm_error(f::SquaredL2Norm, a, b, T::Option{<:Number} = nothing)
    factor = isnothing(T) ? 1 : (T - f.ddof)
    val = LinearAlgebra.norm(a - b, 2)
    return val^2 / factor
end
function norm_error(f::SquaredL2Norm, a, T::Option{<:Number} = nothing)
    factor = isnothing(T) ? 1 : (T - f.ddof)
    val = LinearAlgebra.norm(a, 2)
    return val^2 / factor
end
function norm_error(::L1Norm, a, b, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T)
    return LinearAlgebra.norm(a - b, 1) / factor
end
function norm_error(::L1Norm, a, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T)
    return LinearAlgebra.norm(a, 1) / factor
end
function norm_error(f::LpNorm, a, b, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T - f.ddof)
    factor = if f.p == 3
        cbrt(factor)
    else
        factor^(inv(f.p))
    end
    return LinearAlgebra.norm(a - b, f.p) / factor
end
function norm_error(f::LpNorm, a, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T - f.ddof)
    factor = if f.p == 3
        cbrt(factor)
    else
        factor^(inv(f.p))
    end
    return LinearAlgebra.norm(a, f.p) / factor
end
function norm_error(f::LInfNorm, a, b, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T - f.ddof)
    ty = promote_type(eltype(a), eltype(b))
    p = ifelse(f.pos, typemax(ty), typemin(ty))
    return LinearAlgebra.norm(a - b, p) / factor
end
function norm_error(f::LInfNorm, a, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T - f.ddof)
    ty = eltype(a)
    p = ifelse(f.pos, typemax(ty), typemin(ty))
    return LinearAlgebra.norm(a, p) / factor
end

export IsEmptyError, IsNothingError, IsNonFiniteError, PropertyPathError, VecScalar, L2Norm,
       SquaredL2Norm, L1Norm, LpNorm, LInfNorm
