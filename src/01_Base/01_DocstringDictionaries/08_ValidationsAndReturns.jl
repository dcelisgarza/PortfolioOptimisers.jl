# `err_name_dict`, `val_dict` and `ret_dict`: the error names, the `# Validation`
# rules and the `# Returns` descriptions.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(err_name_dict, :err_name_dict, :kt => "cokurtosis", :sk => "coskewness",
                 :V => "negative spectral coskewness", :D2 => "duplication matrix",
                 :L2 => "elimination matrix", :S2 => "summation matrix")
unique_key_dict!(val_dict, :val_dict, :oow => "If `w` is not `nothing`, `!isempty(w)`.",
                 :oow_nonneg => "If `w` is a `StatsBase.AbstractWeights`, it is not empty, and each entry is finite and `>= 0`. A [`DynamicAbstractWeights`](@ref) is not checked here, because it holds no weights until it reads the data.",#
                 :gerbt => "`0 <= t`.",#
                 :t => "`0 < t < 1`.",#
                 :c1 => "`0 <= c1`.",#
                 :c2 => "`0 <= c2`.",#
                 :c3 => "`0 <= c3`.",#
                 :c3c2 => "`c3 > c2`.",#
                 :sbn => "`0 <= n`. `Inf` is permitted and `NaN` is not.",#
                 :dims => "`dims in (1, 2)`.",#
                 :nan_frame => "When an asset is outside the Coverage Universe, the element type of the moment holds `NaN`. An `Integer` or a `Rational` element type raises an `ArgumentError`.",#
                 :fd_panel => "The panel resolves. [`asset_panel`](@ref) raises an [`IsNothingError`](@ref) naming the site when it does not.",#
                 :fd_strict => "Under `de.strict = true`, every entry of `de.sel` names a Panel Field, a level or a label that the panel holds. Raises an `ArgumentError`. Under `de.strict = false`, such an entry warns and is dropped.",#
                 :alpha => "`0 < alpha < 1`.",#
                 :beta => "`0 < beta < 1`.",#
                 :bins => "If `bins` is an integer, `0 < bins <= RESOURCE_LIMITS[].max_bins` (the joint histogram is `bins × bins`; see [`RESOURCE_LIMITS`](@ref)).",#
                 :ep_gridK => "`isodd(K)` and `1 <= K <= RESOURCE_LIMITS[].max_ep_grid` (every grid point is one binary variable of the mixed-integer program an upper-bound or equality view builds; see [`RESOURCE_LIMITS`](@ref)).",#
                 :dopower => "If `power` is not `nothing`, `power >= 1`.",#
                 :p_owa => "`!isempty(p)` and `all(x -> x > 1, p)`.",#
                 :settings => "If not `nothing`, `!isempty(settings)`.",#
                 :S => "`!isempty(S)`.",#
                 :D => "`!isempty(D)`.",#
                 :ck => "`k >= 1`.",#
                 :lm_k => "`k >= 2`.",#
                 :alpha_i_alpha => "`0 < alpha_i < alpha < 1`, checked when `alpha` is a number. When `alpha` holds a Calibration Rule only `0 < alpha_i < 1` is checked here, and the joint bound is checked when the rebuild runs at fold time. A rule that returns a value at or below the stated `alpha_i` is refused there, and this joint bound is the whole of the ordering validation.",#
                 :a_sim_pos => "`a_sim > 0`.",#
                 :beta_i_beta => "`0 < beta_i < beta < 1`, checked when `beta` is a number. When `beta` holds a Calibration Rule only `0 < beta_i < 1` is checked here, and the joint bound is checked when the rebuild runs at fold time, on the terms the lower tail states.",#
                 :b_sim_pos => "`b_sim > 0`.",#
                 :S_D => "`size(S) == size(D)`.",#
                 :S_P => "If `P` is not `nothing`, `!isempty(P)` and `size(S) == size(P)`.",#
                 :max_k => "If `max_k` is not `nothing`, `max_k >= 1`.",#
                 :kalg => "If `alg` is an `Integer`, `alg >= 1`.",#
                 :dbhtpower => "`power > 0`.",#
                 :dbhtcoef => "`isfinite(coef) && coef > 0`.",#
                 :Xe => "`!isempty(X)`.",#
                 :sdrate => "`rate > 0`.",#
                 :sdpower => "`power > 0`.",#
                 :phX_Xv => "`If `X` is a `MatNum`:\n    + Must be symmetric, `LinearAlgebra.issymmetric(X)`\n    + Must have zero diagonal, `all(iszero, LinearAlgebra.diag(X))`.",#
                 :ntn => "If `n` is an `Integer`, `1 <= n <= RESOURCE_LIMITS[].max_hop_count` (three readers sum `A^i` over `i in 0:n`, so the compute cost is linear in `n`; see [`RESOURCE_LIMITS`](@ref)). A rule is checked when it is resolved, not when it is stored.",#
                 :sepdmax => "If `dmax` is a `Number`, `dmax > 0`. A rule is checked when it is resolved, not when it is stored.",#
                 :sepq => "`0 <= q <= 1`.",#
                 :ctargs_nomat => "No entry of `args` is an `AbstractMatrix`. A weight matrix reaches a centrality algorithm through [`centrality_polarity`](@ref), and never through `args`.",#
                 :treeargs_nochan => "No entry of `args` is an `AbstractMatrix` or an `AbstractVector`, and `kwargs` holds no `minimize` key. Each of those reaches a channel that would re-weight or re-orient the graph the [`NetworkEstimator`](@ref) built.",#
                 :A => "`!isempty(A)`.",#
                 :B => "`!isempty(B)`.",#
                 :A_B => "`size(A, 1) == length(B)`, one row of `A` per entry of `B`.",#
                 :eqineq => "Both `eq` and `ineq` cannot be `nothing` at the same time, `!(isnothing(ineq) && isnothing(eq))`.",
                 :decay => "`0 < decay < 1`.",#
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
                 :ra_norm_x => "`x` is valid",#
                 :relax => "The encoding is not exact: the entries below bound the quantity instead of reproducing it, and the bound is tight only under the condition stated here.")
unique_key_dict!(ret_dict, :ret_dict,
                 :mu => "`mu::ArrNum`: Expected returns vector `assets x 1` if the `dims` keyword does not exist or `dims = 2`, `1 x assets` if `dims = 1`.",#
                 :sigma => "`sigma::MatNum`: Covariance matrix `assets x assets`.",#
                 :rho => "`rho::MatNum`: Correlation matrix `assets x assets`.",#
                 :Ddist => "`D::MatNum`: Distance matrix `assets x assets`, in the units the distance algorithm defines.",#
                 :nbins => "`nbins::Integer`: Number of histogram bins for the variable pair.",#
                 :dx => "`dx::Number`: Optimal histogram bin width.",#
                 :sigrho => "`sigrho::MatNum`: Covariance/correlation matrix `assets x assets`.",#
                 :sk => "`sk::MatNum`: Coskewness matrix `assets x assets`.",#
                 :cskew => "`cskew::MatNum`: Coskewness tensor `assets x assets²`.",#
                 :cskewV => "`V::MatNum`: Processed coskewness matrix `assets x assets`.",#
                 :kte => "`kte::MatNum`: Cokurtosis matrix `assets x assets`.",#
                 :ckurt => "`ckurt::MatNum`: Square cokurtosis matrix `assets² x assets²`.",#
                 :me => "`me`: New expected returns estimator of the same type as the argument, with the appropriate weights applied.",#
                 :ce => "`ce`: New covariance estimator of the same type as the argument, with the new weights applied.",#
                 :ve => "`ve`: New variance estimator of the same type as the argument, with the new weights applied.",#
                 :skev => "`skev`: New coskewness estimator of the same type as the argument, for the new view.",#
                 :ktev => "`kev`: New cokurtosis estimator of the same type as the argument, for the new view.",#
                 :stdvar => "`res::ArrNum`: Variance or standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",#
                 :stdvarnum => "`res::Number`: Variance or standard deviation `X`",#
                 :stdarr => "`sd::ArrNum`: Standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",
                 :vararr => "`vr::ArrNum`: Variance vector of `X`, reshaped to be consistent with the dimension along which the value is computed.",
                 :stdnum => "`sd::Number`: Standard deviation of `X`.",
                 :varnum => "`vr::Number`: Variance of `X`.",
                 :alg => "`alg`: The original algorithm instance.")
