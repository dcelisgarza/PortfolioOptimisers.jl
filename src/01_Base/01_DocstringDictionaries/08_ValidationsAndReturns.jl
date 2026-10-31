# `err_name_dict`, `val_dict` and `ret_dict`: the error names, the `# Validation`
# rules and the `# Returns` descriptions.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(err_name_dict, :err_name_dict, :kt => "cokurtosis", :sk => "coskewness",
                 :V => "negative spectral coskewness", :D2 => "duplication matrix",
                 :L2 => "elimination matrix", :S2 => "summation matrix")
unique_key_dict!(val_dict, :val_dict, :oow => "If `w` is not `nothing`, `!isempty(w)`.",
                 :ce_cov => "`ce` is not an [`AbstractVarianceEstimator`](@ref). A variance estimator has no `cov` or `cor`, so [`assert_covariance_estimator`](@ref) raises an `ArgumentError` that names `ce`.",#
                 :sigma_cov => "If `sigma` is an estimator, it is not an [`AbstractVarianceEstimator`](@ref). A variance estimator has no `cov`, so [`assert_covariance_estimator`](@ref) raises an `ArgumentError` that names `sigma`.",#
                 :oow_rm => "If `w` is a `StatsBase.AbstractWeights`, it is not empty, each entry is finite and `>= 0`, and `sum(w) > 0`, through [`assert_observation_weights`](@ref). A [`DynamicAbstractWeights`](@ref) holds no weights until it reads the data, so [`checked_observation_weights`](@ref) checks it at evaluation, and also checks there that the weights have one entry for each observation.",#
                 :gerbt => "`0 <= t`.",#
                 :t => "`0 < t < 1`.",#
                 :c1 => "`0 <= c1`.",#
                 :c2 => "`0 <= c2`.",#
                 :c3 => "`0 <= c3`.",#
                 :c3c2 => "`c3 > c2`.",#
                 :sbn => "`0 <= n`. `Inf` is permitted and `NaN` is not.",#
                 :dims => "`dims in (1, 2)`.",#
                 :nan_frame => "When an asset is outside the Coverage Universe, the element type of the moment holds `NaN`. An `Integer` or a `Rational` element type raises an `ArgumentError`.",#
                 :nan_series_window => "When the window keeps no observation of the first row, the element type of the variance holds `NaN`. An `Integer` or a `Rational` element type raises an `ArgumentError`.",#
                 :fd_panel => "The panel resolves. [`asset_panel`](@ref) raises an [`IsNothingError`](@ref) that names the site when it does not.",#
                 :fd_strict => "Under `de.strict = true`, every entry of `de.sel` names a Panel Field, a level, a label or a group that the panel holds. Raises an `ArgumentError`. Under `de.strict = false`, such an entry warns, and the estimator drops it.",#
                 :alpha => "`0 < alpha < 1`.",#
                 :alpha_tail0 => "If `alpha` is a number: `0 <= alpha < 1`. At `alpha = 0` the measure is the largest loss among the observations with positive weight, [`worst_positive_weight_loss`](@ref), and an observation with zero weight cannot set it.",#
                 :beta => "`0 < beta < 1`.",#
                 :bins => "If `bins` is an integer, `0 < bins <= RESOURCE_LIMITS[].max_bins`. The joint histogram is `bins × bins`, and [`RESOURCE_LIMITS`](@ref) sets the cap.",#
                 :ep_gridK => "`isodd(K)` and `1 <= K <= RESOURCE_LIMITS[].max_ep_grid`. Each grid point is one binary variable of the mixed-integer program that an upper-bound or equality view builds, and [`RESOURCE_LIMITS`](@ref) sets the cap.",#
                 :ep_attainable_row => "The target of each row is strictly inside the band of its left-hand side over the scenarios at a positive prior probability. A target outside the band, or on its edge, raises a `DomainError` that names the view and the band, through [`ep_assert_attainable_rows`](@ref). An inequality row needs only the lower edge of the band.",#
                 :dopower => "If `power` is not `nothing`, `power >= 1`.",#
                 :p_owa => "`!isempty(p)` and `all(x -> x > 1, p)`.",#
                 :settings => "If `settings` is a collection, `!isempty(settings)`, else an [`IsEmptyError`](@ref). If `settings` is not `nothing`, each key is a [`SlvKeys`](@ref), else an `ArgumentError`.",#
                 :S => "`!isempty(S)`.",#
                 :D => "`!isempty(D)`.",#
                 :ck => "`k >= 1`.",#
                 :lm_k => "`k >= 2`.",#
                 :alpha_i_alpha => "`0 < alpha_i < alpha < 1`, checked when `alpha` is a number. When `alpha` holds a Calibration Rule only `0 < alpha_i < 1` is checked here, and the joint bound is checked when the rebuild runs at fold time. A rule that returns a value at or below the stated `alpha_i` is refused there, and this joint bound is the whole of the ordering validation.",#
                 :a_sim_ge2 => "`a_sim >= 2`. The quadrature grid needs its two ends.",#
                 :S_D => "`size(S) == size(D)`.",#
                 :S_P => "If `P` is not `nothing`, `!isempty(P)` and `size(S) == size(P)`.",#
                 :max_k => "If `max_k` is not `nothing`, `max_k >= 1`.",#
                 :kalg => "If `alg` is an `Integer`, `alg >= 1`.",#
                 :dbhtpower => "`power > 0`.",#
                 :dbhtcoef => "`isfinite(coef) && coef > 0`.",#
                 :Xe => "`!isempty(X)`.",#
                 :sdrate => "`rate > 0`.",#
                 :sdpower => "`power > 0`.",#
                 :phX_Xv => "If `X` is a `MatNum`, `LinearAlgebra.issymmetric(X)` and `all(iszero, LinearAlgebra.diag(X))`. The constructor throws an `ArgumentError` otherwise.",#
                 :ntn => "If `n` is an `Integer`, `1 <= n <= RESOURCE_LIMITS[].max_hop_count`. Three readers sum `A^i` over `i in 0:n`, so the compute cost is linear in `n`. See [`RESOURCE_LIMITS`](@ref). [`resolve_separation`](@ref) checks a rule when it resolves the rule, and the constructor does not check it.",#
                 :sepdmax => "If `dmax` is a `Number`, `dmax > 0`. [`resolve_separation`](@ref) checks a rule when it resolves the rule, and the constructor does not check it.",#
                 :sepq => "`0 <= q <= 1`.",#
                 :ctargs_nomat => "No entry of `args` is an `AbstractMatrix`. The weights reach a centrality algorithm through the graph that [`centrality_graph`](@ref) builds for its [`centrality_polarity`](@ref), and never through `args`.",#
                 :treeargs_nochan => "No entry of `args` is an `AbstractMatrix` or an `AbstractVector`, and `kwargs` holds no `minimize` key. Each of those reaches a channel that re-weights or re-orients the graph that the [`NetworkEstimator`](@ref) built.",#
                 :A => "`!isempty(A)`.",#
                 :B => "`!isempty(B)`.",#
                 :A_B => "`size(A, 1) == length(B)`, one row of `A` per entry of `B`.",#
                 :eqineq => "Both `eq` and `ineq` cannot be `nothing` at the same time, `!(isnothing(ineq) && isnothing(eq))`.",
                 :decay => "`0 < decay < 1`.",#
                 :rf => "`isfinite(rf)`.",#
                 :bsr => "`isfinite(bsr)`.",#
                 :q_scorer => "`0 <= q <= 1`.",#
                 :unit => "`unit > 0`.",#
                 :katz_alpha => "`alpha > 0`.",#
                 :min_obs => "`min_obs > 0`.",#
                 :hac_lags => "If `hac_lags` is not `nothing`, `hac_lags > 0`.",#
                 :regime_decay => "`0 < regime_decay < 1`.",#
                 :regime_min_obs => "`regime_min_obs > 0`.",#
                 :regime_lohi_mult => "If `regime_lohi_mult` is not `nothing`, `0 < regime_lohi_mult[1] < regime_lohi_mult[2]`.",#
                 :ra_x => "`x > 0`, and `x` is finite.",#
                 :ra_y => "`y > 0`, and `y` is finite.",#
                 :relax => "The encoding is not exact: the entries below bound the quantity instead of reproducing it, and the bound is tight only under the condition stated here.")
unique_key_dict!(ret_dict, :ret_dict,
                 :mu => "`mu::ArrNum`: Expected returns vector `assets x 1` if the `dims` keyword does not exist or `dims = 2`, `1 x assets` if `dims = 1`.",#
                 :sigma => "`sigma::MatNum`: Covariance matrix `assets x assets`.",#
                 :rho => "`rho::MatNum`: Correlation matrix `assets x assets`.",#
                 :Ddist => "`D::MatNum`: Distance matrix `assets x assets`, in the units the distance algorithm defines.",#
                 :nbins => "`nbins::Integer`: Number of histogram bins for the variable pair.",#
                 :dx => "`dx::Number`: Optimal histogram bin width.",#
                 :sigrho => "`sigrho::MatNum`: Covariance or correlation matrix, `assets × assets`.",#
                 :cskew => "`cskew::MatNum`: Coskewness tensor `assets x assets²`.",#
                 :cskewV => "`V::MatNum`: Negative spectral skewness matrix `assets x assets` of the coskewness tensor, after the matrix processing.",#
                 :ckurt => "`ckurt::MatNum`: Square cokurtosis matrix `assets² x assets²`.",#
                 :me => "`me`: New expected returns estimator of the same type as the argument, with the new weights applied.",#
                 :ce => "`ce`: New covariance estimator of the same type as the argument, with the new weights applied.",#
                 :ve => "`ve`: New variance estimator of the same type as the argument, with the new weights applied.",#
                 :skev => "`skev`: New coskewness estimator of the same type as the argument, for the new view.",#
                 :ktev => "`ktev`: New cokurtosis estimator of the same type as the argument, for the new view.",#
                 :stdvar => "`res::ArrNum`: Variance or standard deviation vector of `X`, with the shape that matches the dimension of the computation.",#
                 :stdvarnum => "`res::Number`: Variance or standard deviation of `X`.",#
                 :stdarr => "`sd::ArrNum`: Standard deviation vector of `X`, with the shape that matches the dimension of the computation.",
                 :vararr => "`vr::ArrNum`: Variance vector of `X`, with the shape that matches the dimension of the computation.",
                 :stdnum => "`sd::Number`: Standard deviation of `X`.",
                 :varnum => "`vr::Number`: Variance of `X`.",
                 :alg => "`alg`: The original algorithm instance.")
