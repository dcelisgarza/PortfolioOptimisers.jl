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
                 :ntn => "`n`: Number of steps to take in the network for deciding adjacency. An `Integer` is used as it stands. A [`HopCountAlgorithm`](@ref) or a `Function` is a **rule**, called as `n(nte, X, g; dims = dims, kwargs...)` by [`resolve_separation`](@ref) at the point of use, `g` being the structure the consumer already built, and must return an `Integer`.",#
                 :sepdmax => "`dmax`: Separation budget, in the units the separation is measured in. `nothing` means the observed diameter of the structure. A [`PathLengthAlgorithm`](@ref) or a `Function` is a **rule**, called as `dmax(nte, X, g; dims = dims, kwargs...)` by [`resolve_separation`](@ref) at the point of use, `g` being the structure the consumer already built, and must return a `Number`.",#
                 :sepq => "`q`: Quantile of the observed separations to take as the budget. The reachable off-diagonal pairs are the population, so `q` is the fraction of them the budget relates.",#
                 :clres => "`res`: Clustering result.",#
                 :S => "`S`: Similarity matrix.",#
                 :D => "`D`: Distance matrix.",#
                 :ck => "`k`: Optimal number of clusters.",#
                 :vsalg => "`alg`: The measure used to evaluate clustering quality.",#
                 :max_k => "`max_k`: Maximum number of clusters to consider. If `nothing`, computed as the `floor(Int, sqrt(assets))`.",#
                 :kalg => "`alg`: Algorithm for selecting the optimal number of clusters. If an integer, defines the number of clusters directly.",#
                 :clalg => "`alg`: Clustering algorithm.",#
                 :onc => "`onc`: Optimal number of clusters estimator.",#
                 :phX_Xv => "`X`: Phylogeny matrix or vector.",#
                 :clP => "`P`: Pseudo-distance matrix the clustering was run on, `nothing` when the clustering ran on `D` itself. A [`NetworkClustersEstimator`](@ref) builds it by accumulating the network structure out of the distance or similarity matrix; see [`clusterise`](@ref).",#
                 :pler => "`pl`: Network estimator, phylogeny result, clustering estimator, or clustering result.",#
                 :plsrc => "`pl`: Network estimator or clustering estimator. A precomputed `PhylogenyResult` or `Clusters` is **not** accepted: this slot says how to build the phylogeny for whatever universe the estimator is given, and a precomputed one answers for a fixed universe instead. Pass the constraint *result* if you already have the structure.",#
                 ## Separation and separation decay
                 :sdecay => "`decay`: Separation decay algorithm, the rule by which the score falls off as two assets get further apart. Distinct from the exponentially weighted moment estimators' `decay`, which is a smoothing constant over observations.",#
                 :sdrate => "`rate`: Rate of the exponential fall-off, `exp(-rate * d)`. Larger values decay faster. The per-step retention form, `ratio^d`, is `rate = -log(ratio)`.",#
                 :sdpower => "`power`: Exponent of the reciprocal fall-off, `inv((1 + d)^power)`. Larger values decay faster.",#
                 ## DBHT
                 :dbhtpower => "`power`: Exponent for the the distance matrix when computing the similarity matrix.",#
                 :dbhtcoef => "`coef`: Coefficient for the the distance matrix when computing the similarity matrix.",#
                 :sim => "`sim`: Similarity matrix algorithm. The PMFG cannot take a negative weight, so the family is the non-negative one and [`AngularSimilarity`](@ref) is refused.",#
                 :root => "`root`: Root selection method.",#
                 # Universe sets and the values an estimator maps onto assets.
                 :sets => "`sets`: Sets used to map estimator values to assets.",#
                 :val => "`val`: Default value to use for the estimator. If `nothing`, the estimator provides the default value.",#
                 :ekey => "`key`: Key to specify the universe in `sets.dict` that names resolve against. If `nothing`, the key is taken from `sets.xkey` — or, where the caller is written against another declared axis, from that axis' key.",#
                 :bl_axis => "`axis`: Field of `sets` naming the declared axis the views resolve against: `:xkey` for the asset axis, `:tfkey` for the time-series factor axis, `:cfkey` for the cross-sectional one. The key itself is read from `sets` here, and only when `sets` is not `nothing`.",#
                 :sets_f => "`sets`: Universe sets. The **time-series factor** axis, `sets.dict[sets.tfkey]`, is what this estimator reads: it is the universe the views are written in, and it must name the columns of `F` in order. The asset axis is required by [`UniverseSets`](@ref) and is what a view slices — the factor entries come back from [`port_opt_view`](@ref) untouched.",#
                 :sets_frb => "`sets`: Universe sets. A **factor** axis is what this algorithm reads — [`factor_axis_key`](@ref) picks `sets.tfkey` or `sets.cfkey` off `re`, so the axis follows the loadings family rather than the caller. It is the universe the risk budget is written in, and it must name the columns of `rr.L` in order — the budget is over the factor weights `w1`, one per column of the loadings the risk decomposition uses. It is only read when `rkb` is a [`RiskBudgetEstimator`](@ref); a [`RiskBudget`](@ref) result carries its own vector and resolves no names. The asset axis is required by [`UniverseSets`](@ref) and is what a view slices — the factor entries come back from [`port_opt_view`](@ref) untouched.",#
                 :datatype => "`datatype`: Data type to use for the result in case `val` is `nothing`.",#
                 :strict => "`strict`: Whether to throw an error if `sets` does not contain the desired value in `sets.dict[key]`.",#
                 # Constraints
                 :A => "`A`: Linear constraint coefficient matrix.",#
                 :B => "`B`: Linear constraint response vector.",#
                 :eq => "`eq`: Optional equality constraints.",#
                 :ineq => "`ineq`: Optional inequality constraints.",#
                 # Constraint generation.
                 :dval => "`dval`: Default value for assets not specified in `val`.",#
                 :dict => "`dict`: Dictionary mapping group identifiers to member labels.",#
                 :vars => "`vars`: Variable names in the parsed constraint expression.",#
                 :coef_c => "`coef`: Coefficients corresponding to the constraint variables.",#
                 :op => "`op`: Comparison operator (`==`, `<=`, or `>=`).",#
                 :rhs => "`rhs`: Right-hand side value of the constraint.",#
                 :rhs_rho => "`rhs`: Right-hand side of the constraint. A view over a single asset pair carries one value. A view over a pair of groups carries one value per spanned pair, in the order of `ij`.",#
                 :eqn => "`eqn`: Formatted string representation of the constraint equation.",#
                 :ij => "`ij`: Pair of asset indices for correlation-based constraints.",#
                 # Cluster node.
                 :id_node => "`id`: Node identifier.",#
                 :left_node => "`left`: Left child node.",#
                 :right_node => "`right`: Right child node.",#
                 :height_node => "`height`: Height of the node in the dendrogram.",#
                 :level_node => "`level`: Number of leaves in the subtree rooted at the node, `1` for a leaf. It is the fourth column of a linkage matrix, and [`pre_order`](@ref) sizes its traversal stack from it.",#
                 # Default weight bounds.
                 :dlb => "`dlb`: Default lower bound.",#
                 :dub => "`dub`: Default upper bound.",#
                 # Weight bounds.
                 :wb => "`wb`: Weight bounds.",#
                 # Constraint generation.
                 :rkb_val => "`val`: Vector of non-negative risk budgets, one per entry of the axis the budget is written against. [`risk_budget_constraints`](@ref) normalises it to sum to one; a hand-built vector is stored as given, and the model reads it inside a logarithmic barrier, so only its **relative** entries matter.",#
                 :rkbe_val => "`val`: Mapping of names to risk budget values. A name may be an asset or a group, and a group assigns its value to every asset in it. A scalar is accepted and resolves to `RiskBudget(1.0)` whatever the scalar was, so only a one-entry axis can consume it. Write the uniform budget as `nothing`.",#
                 :us_xkey => "`xkey`: Key in `dict` identifying the primary asset list. Required, and the axis a view slices.",#
                 :us_uxkey => "`uxkey`: Key prefix for unique-entry asset group variants in `dict`.",#
                 :us_tfkey => "`tfkey`: Key in `dict` identifying the **time-series** factor list — the columns of `rd.F`, which a time-series regression fits one loading vector per asset against. Optional — a consumer that needs it and does not find it throws at the point of need.",#
                 :us_utfkey => "`utfkey`: Key prefix for unique-entry time-series factor group variants in `dict`. Validated at construction, never recomputed by a view.",#
                 :us_cfkey => "`cfkey`: Key in `dict` identifying the **cross-sectional** factor list — the exposures a cross-sectional regression fits one loading vector per observation against. Optional, and validated exactly as `tfkey` is; the two axes are never validated against each other, so a problem may declare one, both, or neither.",#
                 :us_ucfkey => "`ucfkey`: Key prefix for unique-entry cross-sectional factor group variants in `dict`. Validated at construction, never recomputed by a view.",#
                 :us_nikey => "`nikey`: Key in `dict` identifying the **Non-Investable Axis** — the names the Investable Mask left out. Optional, and minted by a door rather than authored: a view drops it, so a `UniverseSets` that carries one was reduced by exactly one door, for exactly that problem. Bare: it names assets and admits no prefixed partition, because its entries are unique by construction and nothing resolves a group over it that a plain, axis-blind group does not already reach.",#
                 :p_phylo => "`p`: Non-negative penalty factor on the trace of the semidefinite matrix variable. It is read only when the objective does not minimise a variance on the same weights. Such a variance is itself a trace against that variable, so it holds the variable down, and the model adds no second term. The penalty holds the matrix variable down to the outer product of the weights, which the relatedness rows need to bind the weights. It is also at least ``p \\lVert \\boldsymbol{w} \\rVert_2^2 / k``, so it spreads the weights, and a large `p` can leave both assets of a linked pair in the portfolio. The relatedness rows hold on the matrix variable for every `p`.",#
                 :A_phylo => "`A`: Symmetric relatedness matrix with a zero diagonal. A network source gives the range connection matrix, a clustering source the adjacency label matrix. Stored as given.",#
                 :A_iphylo => "`A`: Row set of the relatedness matrix, stored as `unique(A + I; dims = 1)` and **not** as the matrix passed in. The identity puts each asset in its own row, and the deduplication drops rows that repeat. One row per distinct neighbourhood or cluster survives, which is why the stored matrix is usually shorter than it is wide.",#
                 :B_phylo => "`B`: Right-hand side of `A * z <= B`, where `z` is the held indicator: the largest number of assets that may be held out of each row of `A`. A scalar applies to every row. A vector states one bound per row, so its length must match the row count of the stored `A` and not the number of assets. On an estimator the rows do not exist yet, so a vector is only checked against the largest number of clusters the clustering estimator can return.",#
                 :cc_A => "`A`: Centrality estimator. Its centrality vector is the row of the generated linear constraint.",#
                 :cc_B => "`B`: Right-hand side of the constraint. A number is the threshold itself. A [`VectorToScalarMeasure`](@ref) derives the threshold from the centrality vector `A` produces, so the constraint always has a feasible point. The measure reads that vector, never the row after `comp` has flipped its sign, so `MinValue()` gives the smallest entry under `<=` and under `>=` alike.",#
                 :cc_comp => "`comp`: Comparison operator for the centrality constraint. `==` builds an equality row, every other operator an inequality row.",#
                 :lce_val => "`val`: Constraint equation(s) to parse.",#
                 :ece_lce => "`lce`: Wrapped linear constraint estimator(s) or precomputed constraint, written in the names of the space's basis. Exactly what `lcse` itself accepts, so no shape can reach the optimiser un-re-based.",#
                 :ece_space => "`space`: Basis the wrapped constraint is written in. Required — the absence of a re-basis is spelled by using a bare `LinearConstraintEstimator`, not by a space member.",#
                 :fs_re => "`re`: Source of the loadings the rows are re-based through, or `nothing` to read the prior's `rr`. A precomputed `Regression` states the basis outright; an estimator fits one from the returns, which is what makes a factor mandate legal on a prior that carries no factor block. The precedence is `resolve_factor_regression`'s: a precomputed result wins, then the prior's `rr`, then a refit.",#
                 :asets_val => "`val`: Group name key for asset set membership matrix extraction.",#
                 :thr_val => "`val`: Asset-specific minimum-holding threshold value(s).",#
                 :thr_res_val => "`val`: Minimum-holding threshold(s) on the portfolio weights. A held position must reach its threshold; a position below it is driven to zero. The threshold binds the **held** weight, never the trade, so a reference portfolio does not enter it.",#
                 # Preselection. `20_AssetSelection.jl` pairs two redundancy
                 # algorithms over one correlation matrix, and two selection
                 # rules over one taken set, so each of the five descriptions
                 # below is shared by two fields of that file.
                 :pre_ce_corr => "`ce`: Covariance estimator supplying the correlation matrix.",#
                 :pre_t_corr => "`t`: Correlation at or above which two assets are redundant.",#
                 :pre_absolute => "`absolute`: Whether to compare the absolute value of the correlation.",#
                 :pre_action => "`action`: `:keep` retains the taken assets, `:drop` retains everything else.",#
                 :pre_measure => "`measure`: Reducer producing the fallback drop score from each column of the correlation matrix; ignored when the selector carries a `score`.")
