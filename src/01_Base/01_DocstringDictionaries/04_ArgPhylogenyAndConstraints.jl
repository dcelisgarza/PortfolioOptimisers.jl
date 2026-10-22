# `arg_dict`: phylogeny and clustering, universe sets, constraints and their
# generation, weight bounds and preselection.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(arg_dict, :arg_dict,
                 # Phylogeny.
                 :cle => "`cle`: Clusters estimator.",#
                 :clr => "`clr`: Clusters result.",#
                 :plr => "`plr`: Phylogeny result.",#
                 :nte => "`nte`: Network estimator.",#
                 :cte => "`cte`: Centrality estimator.",#
                 :cte_jmp => "`cte`: Centrality constraint(s). A `CentralityConstraint`, a vector of them, or an already-generated `LinearConstraint`. Resolved by `centrality_constraints` into the `ctr` slot of [`ProcessedJuMPOptimiserAttributes`](@ref).",#
                 :cta => "`ct`: Centrality algorithm.",#
                 :ctr => "`ctr`: Centrality constraint result. The `LinearConstraint` the centrality constraints resolve to.",#
                 :ctargs => "`args`: Positional arguments for the centrality function.",#
                 :ctkwargs => "`kwargs`: Keyword arguments for the centrality function.",#
                 :ctov => "`ov`: Polarity override. [`TopologyOnly`](@ref) asks for the centrality over the network's topology alone, so [`centrality_polarity`](@ref) answers `nothing` and [`centrality_graph`](@ref) builds the plain graph. `nothing` leaves the algorithm's declared polarity in force.",#
                 :treeargs => "`args`: Positional arguments for the spanning tree function. Every positional slot those functions declare is a weight channel, so [`assert_tree_args`](@ref) refuses a matrix or a vector here: the weights arrive with the graph.",#
                 :treekwargs => "`kwargs`: Keyword arguments for the spanning tree function. [`assert_tree_args`](@ref) refuses `minimize`, which would invert the minimisation the tree branch is defined by.",#
                 :ntalg => "`alg`: Tree or similarity matrix algorithm. A similarity here selects the network by building a PMFG, so the family is the non-negative one and [`AngularSimilarity`](@ref) is refused.",#
                 :ntsep => "`sep`: Separation algorithm, the rule measuring how far apart two assets sit in the network and the budget beyond which they count as unrelated.",#
                 :ntn => "`n`: Separation budget in hops. A pair whose shortest path has more than `n` edges is unrelated. The library uses an `Integer` as it is. A [`HopCountAlgorithm`](@ref) or a `Function` is a rule. [`resolve_separation`](@ref) calls it as `n(nte, X, g; dims = dims, kwargs...)`, where `g` is the structure that the consumer built, and the rule must return an `Integer`.",#
                 :sepdmax => "`dmax`: Separation budget, in the units of the separation. `nothing` means the observed diameter of the structure. A [`PathLengthAlgorithm`](@ref) or a `Function` is a rule. [`resolve_separation`](@ref) calls it as `dmax(nte, X, g; dims = dims, kwargs...)`, where `g` is the structure that the consumer built, and the rule must return a `Number`.",#
                 :sepq => "`q`: Quantile of the observed separations to take as the budget. The reachable off-diagonal pairs are the population, so `q` is the fraction of them the budget relates.",#
                 :clres => "`res`: Clustering result.",#
                 :S => "`S`: Similarity matrix.",#
                 :D => "`D`: Distance matrix.",#
                 :ck => "`k`: Selected number of clusters.",#
                 :vsalg => "`alg`: Vector-to-scalar measure that the score applies at each candidate count.",#
                 :max_k => "`max_k`: Largest number of clusters that the selection considers. The ceiling is `min(floor(Int, sqrt(N)), max_k)`, where `N` is the number of assets. `nothing` sets the ceiling to `floor(Int, sqrt(N))`.",#
                 :kalg => "`alg`: Algorithm that selects the number of clusters, or an `Integer` that states it.",#
                 :clalg => "`alg`: Clustering algorithm.",#
                 :onc => "`onc`: Estimator of the optimal number of clusters.",#
                 :phX_Xv => "`X`: Phylogeny matrix or vector.",#
                 :clP => "`P`: Pseudo-distance matrix that the clustering used, or `nothing` when the clustering used `D`. A [`NetworkClustersEstimator`](@ref) makes it from the structure of the network, see [`clusterise`](@ref).",#
                 :pler => "`pl`: Network estimator, phylogeny result, clustering estimator, or clustering result.",#
                 :plsrc => "`pl`: Network estimator or clustering estimator. The field refuses a precomputed `PhylogenyResult` or `Clusters`. It states how to build the phylogeny for the universe of each call, and a precomputed structure holds one fixed universe. To use a precomputed structure, pass the constraint result instead.",#
                 ## Separation and separation decay
                 :sdecay => "`decay`: Separation decay algorithm, the rule by which the score falls off as two assets get further apart. Distinct from the exponentially weighted moment estimators' `decay`, which is a smoothing constant over observations.",#
                 :sdrate => "`rate`: Rate of the exponential fall-off, `exp(-rate * d)`. A larger rate makes the score fall faster. The retention form `ratio^d` is the same function at `rate = -log(ratio)`.",#
                 :sdpower => "`power`: Exponent of the reciprocal fall-off, `inv((1 + d)^power)`. A larger power makes the score fall faster.",#
                 ## DBHT
                 :dbhtpower => "`power`: The exponent of each distance in the similarity transformation.",#
                 :dbhtcoef => "`coef`: The scale factor of the distance term in the similarity transformation.",#
                 :sim => "`sim`: Similarity matrix algorithm. The PMFG cannot take a negative weight, so the family is the non-negative one and [`AngularSimilarity`](@ref) is refused.",#
                 :root => "`root`: Root selection method.",#
                 :x_src_graph => "`x_src`: The source of the returns that the clustering, phylogeny or centrality estimator reads, `nothing` or an [`AbstractReturnsSource`](@ref). `nothing` and [`DataSource`](@ref) read the asset returns `rd.X`, and `nothing` reads the `X` of the Prior Result when the call gives no returns data. [`ExcessSource`](@ref) reads the returns in excess of the benchmark, `rd.X - rd.B`, the Excess Frame. [`PriorSource`](@ref) reads the `X` of the Prior Result, in the frame that the prior fitted in. Under a factor prior that `X` is a reconstruction of the rank of the factors, and a scenario window cuts its rows. A stated source that the call cannot give, `DataSource()` or `ExcessSource()` with no returns data, or `PriorSource()` on returns data with no Prior Result, gives a warning, and the estimator reads the matrix that the call gives. A call on a bare matrix reads that matrix under every source. An estimator inside another graph estimator keeps `nothing`, and the outer estimator states the source.",#
                 # Universe sets and the values an estimator maps onto assets.
                 :sets => "`sets`: Sets used to map estimator values to assets.",#
                 :val => "`val`: Default value to use for the estimator. If `nothing`, the estimator provides the default value.",#
                 :ekey => "`key`: Key of the universe in `sets.dict` that names resolve against. If `nothing`, the caller reads the key from `sets.xkey`, or from the key of another declared axis when the caller is written against that axis.",#
                 :bl_axis => "`axis`: The field of `sets` that names the declared axis that the views resolve against: `:xkey` for the asset axis, `:tfkey` for the time-series factor axis, `:cfkey` for the cross-sectional factor axis. The function reads the key from `sets`, and only when `sets` is not `nothing`.",#
                 :sets_f => "`sets`: Universe sets. This estimator reads the **time-series factor** axis, `sets.dict[sets.tfkey]`. The views are written in the names of this axis, and the axis must name the columns of `F` in order. [`UniverseSets`](@ref) also needs the asset axis, and a view slices that axis. The factor entries come back from [`port_opt_view`](@ref) with no change.",#
                 :sets_frb => "`sets`: Universe sets. A **factor** axis is what this algorithm reads — [`factor_axis_key`](@ref) picks `sets.tfkey` or `sets.cfkey` off `re`, so the axis follows the loadings family rather than the caller. It is the universe the risk budget is written in, and it must name the columns of `rr.L` in order — the budget is over the factor weights `w1`, one per column of the loadings the risk decomposition uses. It is only read when `rkb` is a [`RiskBudgetEstimator`](@ref); a [`RiskBudget`](@ref) result carries its own vector and resolves no names. The asset axis is required by [`UniverseSets`](@ref) and is what a view slices — the factor entries come back from [`port_opt_view`](@ref) untouched.",#
                 :hedge_frb => "`hedge`: Whether the off-factor weights hedge the factor exposures at minimum variance. When `true` and `flag = true`, the model adds the rows ``\\mathbf{B}_2^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} = \\boldsymbol{0}``, so the off-factor weights take no risk contribution. Ignored when `flag = false`.",#
                 :datatype => "`datatype`: Data type to use for the result in case `val` is `nothing`.",#
                 :strict => "`strict`: Whether to throw an error if `sets` does not contain the desired value in `sets.dict[key]`.",#
                 # Constraints
                 :A => "`A`: Linear constraint coefficient matrix.",#
                 :B => "`B`: Linear constraint response vector.",#
                 :eq => "`eq`: Optional equality constraints.",#
                 :ineq => "`ineq`: Optional inequality constraints.",#
                 # Constraint generation.
                 :dval => "`dval`: Default value of each entry that `val` does not name.",#
                 :dict => "`dict`: Dictionary mapping group identifiers to member labels.",#
                 :vars => "`vars`: Variable names in the parsed constraint expression.",#
                 :coef_c => "`coef`: Coefficients corresponding to the constraint variables.",#
                 :op => "`op`: Comparison operator (`==`, `<=`, or `>=`).",#
                 :rhs => "`rhs`: Right-hand side value of the constraint.",#
                 :rhs_rho => "`rhs`: Right-hand side of the constraint. A view over a single asset pair carries one value. A view over a pair of groups carries one value per spanned pair, in the order of `ij`.",#
                 :eqn => "`eqn`: Formatted string representation of the constraint equation.",#
                 :ij => "`ij`: Pair of asset indices for correlation-based constraints.",#
                 # Cluster node.
                 :id_node => "`id`: The identifier of the node.",#
                 :left_node => "`left`: The left child, or `nothing` for a leaf.",#
                 :right_node => "`right`: The right child, or `nothing` for a leaf.",#
                 :height_node => "`height`: The height of the node in the dendrogram.",#
                 :level_node => "`level`: The number of leaves in the subtree below the node, `1` for a leaf. It is the fourth column of a linkage matrix, and [`pre_order`](@ref) sizes its traversal stack from it.",#
                 # Default weight bounds.
                 :dlb => "`dlb`: Lower bound of an asset that a dictionary or a pair in `lb` does not name. `nothing` resolves to `zero(datatype)`.",#
                 :dub => "`dub`: Upper bound of an asset that a dictionary or a pair in `ub` does not name. `nothing` resolves to `one(datatype)`.",#
                 # Weight bounds.
                 :wb => "`wb`: Weight bounds.",#
                 # Constraint generation.
                 :rkb_val => "`val`: Vector of non-negative risk budgets, one for each entry of the axis that the budget is written against. The estimator method of [`risk_budget_constraints`](@ref) normalises the budget to sum to one. A budget built by hand keeps its own sum. The model reads the budget inside a logarithmic barrier, so only the relative entries change the solution.",#
                 :rkbe_val => "`val`: Map from names to risk budget values. A name can be an asset or a group, and a group gives its value to every asset in the group. A scalar resolves to `RiskBudget(1.0)` for every value of the scalar, so only a one-entry axis can use it. Write the uniform budget as `nothing`.",#
                 :us_xkey => "`xkey`: Key in `dict` identifying the primary asset list. Required, and the axis a view slices.",#
                 :us_uxkey => "`uxkey`: Key prefix for unique-entry asset group variants in `dict`.",#
                 :us_tfkey => "`tfkey`: Key in `dict` of the time-series factor list, the columns of `rd.F`, against which a time-series regression fits one loading vector per asset. Optional. A consumer that needs it and does not find it throws when it reads the axis.",#
                 :us_utfkey => "`utfkey`: Key prefix for unique-entry time-series factor group variants in `dict`. The constructor checks it, and a view never recomputes it.",#
                 :us_cfkey => "`cfkey`: Key in `dict` of the cross-sectional factor list, the exposures against which a cross-sectional regression fits one loading vector per observation. Optional, and the constructor checks it as it checks `tfkey`. It never checks the two axes against each other, so a problem can declare one, both or neither.",#
                 :us_ucfkey => "`ucfkey`: Key prefix for unique-entry cross-sectional factor group variants in `dict`. The constructor checks it, and a view never recomputes it.",#
                 :us_nikey => "`nikey`: Key in `dict` of the Non-Investable Axis, the names that the Investable Mask left out. Optional. The entry of a fit writes it, and a view drops it, so a `UniverseSets` that carries one was reduced by one entry of a fit for one problem. It names assets and has no prefixed partition, because its entries are unique by construction and a plain group already reaches it.",#
                 :p_phylo => "`p`: Non-negative penalty factor on the trace of the semidefinite matrix variable. The model reads it only when the objective does not minimise a variance on the same weights. Such a variance is itself a trace against that variable, so it holds the variable down, and the model adds no second term. The penalty holds the matrix variable down to the outer product of the weights, which the relatedness rows need to bind the weights. The penalty term is also at least ``p \\lVert \\boldsymbol{w} \\rVert_2^2 / k``, so it spreads the weights, and a large `p` can leave both assets of a related pair in the portfolio. The relatedness rows hold on the matrix variable for every `p`.",#
                 :A_phylo => "`A`: Symmetric relatedness matrix with a zero diagonal. A network source gives the range connection matrix, and a clustering source gives the adjacency label matrix. The constructor stores it as given.",#
                 :A_iphylo => "`A`: Row set of the relatedness matrix. The constructor stores `unique(A + I; dims = 1)`, not the matrix passed in. The identity puts each asset in its own row, and `unique` drops the repeated rows. One row survives per distinct neighbourhood or cluster, so the stored matrix usually has fewer rows than columns.",#
                 :B_phylo => "`B`: Right-hand side of `A * z <= B`, where `z` is the held binary. Each entry is the largest number of assets that the portfolio can hold out of one row of `A`. A scalar applies to every row. A vector states one bound per row, so its length must match the row count of the stored `A`, not the number of assets. An estimator has no rows yet. It checks a vector only against the largest number of clusters that a clustering source can return.",#
                 :cc_A => "`A`: Centrality estimator. Its centrality vector gives the coefficients of the generated row.",#
                 :cc_B => "`B`: Right-hand side of the constraint. A number is the threshold itself. A [`VectorToScalarMeasure`](@ref) derives the threshold from the centrality vector that `A` gives, so the threshold moves with the graph. The measure reads that vector, never the row after `comp` flips its sign, so `MinValue()` gives the smallest entry under `<=` and under `>=` alike.",#
                 :cc_comp => "`comp`: Comparison operator for the centrality constraint. `==` builds an equality row, every other operator an inequality row.",#
                 :lce_val => "`val`: Constraint equation(s) to parse.",#
                 :ece_lce => "`lce`: Wrapped linear constraint estimator, precomputed linear constraint, or vector of either, written in the names of the basis of the space. It takes every shape that `lcse` takes except this wrapper, so every such shape can take a re-basis.",#
                 :ece_space => "`space`: Basis in which the wrapped constraint is written. It has no default. A constraint over the assets needs no space, so use a bare `LinearConstraintEstimator` for it.",#
                 :fs_re => "`re`: Source of the loadings through which the rows are re-based, or `nothing` to read the `rr` of the prior. A precomputed `Regression` states the basis. An estimator fits the basis to the returns, so a factor mandate also works on a prior with no factor block. The precedence is the one of `resolve_factor_regression`. At `x_src = nothing`, a precomputed result comes first, then the `rr` of the prior, then a refit.",#
                 :asets_val => "`val`: Group name key for asset set membership matrix extraction.",#
                 :thr_val => "`val`: Minimum-holding thresholds, as a map from asset or group names to values, a scalar, a vector or an algorithmic rule.",#
                 :thr_res_val => "`val`: Minimum-holding thresholds of the portfolio weights, as one scalar for every asset or one value for each asset. A position is zero, or it is at least its threshold. The threshold bounds the held weight and not the trade, so no reference portfolio enters it.",#
                 # Preselection. `20_AssetSelection.jl` pairs two redundancy
                 # algorithms over one correlation matrix, and two selection
                 # rules over one taken set, so each of the five descriptions
                 # below is shared by two fields of that file.
                 :pre_ce_corr => "`ce`: Covariance estimator supplying the correlation matrix.",#
                 :pre_t_corr => "`t`: Correlation at or above which two assets are redundant.",#
                 :pre_absolute => "`absolute`: Whether to compare the absolute value of the correlation.",#
                 :pre_action => "`action`: `:keep` retains the taken assets, `:drop` retains everything else.",#
                 :pre_measure => "`measure`: Reducer producing the fallback drop score from each column of the correlation matrix; ignored when the selector carries a `score`.")
