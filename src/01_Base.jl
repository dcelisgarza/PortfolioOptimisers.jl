"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build a documentation dictionary from `pairs`, and throw if a key appears more than once.

A `Dict` literal is last-wins, so a repeated key drops the earlier entry with no warning and
makes its prose unreachable. This constructor is the guard against that: it fails at load
time and names both descriptions, so the duplicate is visible instead of silent. `name` is
the dictionary under construction, and is used in the error message.

# Related

  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
  - [`ret_dict`](@ref)
  - [`err_name_dict`](@ref)
"""
function unique_key_dict(name::Symbol, pairs::Pair{Symbol, <:AbstractString}...)
    dict = Dict{Symbol, String}()
    for (key, val) in pairs
        if haskey(dict, key)
            throw(ArgumentError("`$(name)` has a repeated key, `:$(key)`. Each key must appear exactly once.\n  first: $(dict[key])\n  later: $(val)"))
        end
        dict[key] = val
    end
    return dict
end
"""
    arg_dict

Maps a parameter key to the docstring description of the corresponding argument or
field, so that a single description is written once here and interpolated into every
docstring that mentions that parameter (via `\$(arg_dict[key])` for `# Arguments`
entries, or through the derived [`field_dict`](@ref) for `# Fields` entries).

Each value has the form ``"`name`: description."``, where `name` is the display name
the caller sees and everything after the first `:` is the prose; `field_dict`
strips the ``"`name`: "`` prefix. A few illustrative entries:

    :ce   => "`ce`: Covariance estimator."
    :oow  => "`w`: Optional observation weights vector `observations × 1`, ..."
    :per  => "`pr`: Prior estimator or result."
    :pler => "`pl`: Network estimator, phylogeny result, clustering estimator, or clustering result."
    :plsrc => "`pl`: Network estimator or clustering estimator -- a source that refits, never a precomputed result."

The `const` definition below is the single source of truth; consult it for the full
table of keys and descriptions. A key must appear once: [`unique_key_dict`](@ref) builds
the table and refuses a repeat, because a `Dict` literal drops the earlier entry in silence.

# Related

  - [`unique_key_dict`](@ref)
  - [`field_dict`](@ref)
  - [`val_dict`](@ref)
"""
const arg_dict = unique_key_dict(:arg_dict,
                                 # Weight vectors.
                                 :pw => "`w`: Portfolio weights vector `assets × 1`.",#
                                 :ow => "`w`: Observation weights vector `observations × 1`.",#
                                 :oow => "`w`: Optional observation weights vector `observations × 1`, or a concrete subtype of [`DynamicAbstractWeights`](@ref). If `nothing`, the computation is unweighted.",#
                                 :eqw => "`w`: Optional equilibrium weights vector `assets × 1`. If `nothing`, equal weights are used.",#
                                 # Matrix processing.
                                 :pdm => "`pdm`: Positive definite matrix estimator.",
                                 :opdm => "`pdm`: Optional positive definite matrix estimator.",
                                 :dn => "`dn`: Matrix denoising estimator.",
                                 :odn => "`dn`: Optional matrix denoising estimator.",
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
                                 :malg => "`alg`: Moment algorithm.",
                                 :corrected => "`corrected`: Whether to apply Bessel's correction.",#
                                 :mutgt => "`tgt`: Shrinkage target.",#
                                 :me_shrink_alg => "`alg`: Expected returns shrinkage algorithm.",#
                                 :me_cval => "`val`: Custom expected returns value.\n\n  - If a scalar, every asset is assigned this value.\n  - If a vector, each element is one asset's value.\n  - If a callable, it is called as `val(X; dims = dims, kwargs...)` and must return one value per asset.",#
                                 :metric => "`metric`: Distance metric used for pairwise computations.",#
                                 :metric_args => "`args`: Additional positional arguments for the distance metric.",#
                                 :metric_kwargs => "`kwargs`: Additional keyword arguments for the distance metric.",#
                                 :t => "`t`: Threshold value.",#
                                 :oiv => "`iv`: Optional implied volatility matrix. Used if any internal covariance estimator is an instance of [`ImpliedVolatility`](@ref).",#
                                 ## Regression
                                 :M => "`M`: Main coefficient (loadings) matrix `assets × factors`.",#
                                 :L => "`L`: Reduced dimensionality coefficient (loadings) matrix `assets × reduced_dimensions`.",#
                                 :b => "`b`: Regression intercept vector.",#
                                 :crit => "`crit`: Factor selection criterion. A [`PValue`](@ref), or a `Val` of one symbol of [`STEPWISE_REGRESSION_CRITERIA`](@ref).",#
                                 :r2variant => "`variant`: Name of the pseudo-``R^2`` variant a maximisation criterion reads, or `nothing` to take the default of the criterion.",#
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
                                 :dopower => "`power`: Optional matrix exponent. `nothing` and `1` both give the base distance, so only `power >= 2` changes the result.",#
                                 :dalg => "`alg`: Distance algorithm.",#
                                 :dmetric => "`metric`: Distance metric used for the distances of distances computations.",#
                                 :dmetric_args => "`args`: Additional positional arguments for the distances of distances metric.",#
                                 :dmetric_kwargs => "`kwargs`: Additional keyword arguments for the distances of distances metric.",#
                                 :fdmetric => "`metric`: Distance metric applied to the rows of the feature matrix.",#
                                 :fcalg => "`alg`: Feature collapse algorithm, used to reduce a window of time-varying features to a single distance matrix. Inert for a 2-D feature matrix.",#
                                 :calg => "`alg`: Collapse algorithm, the aggregator applied along the observation axis.",#
                                 :fdsim => "`sim`: Similarity matrix algorithm used to derive the similarity counterpart of the feature distance matrix.",#
                                 # Priors.
                                 :pe => "`pe`: Prior estimator.",#
                                 :pr => "`pr`: Prior result.",#
                                 :per => "`pr`: Prior estimator or result.",#
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
                                 # Estimators
                                 :sets => "`sets`: Sets used to map estimator values to assets.",#
                                 :val => "`val`: Default value to use for the estimator. If `nothing`, the estimator provides the default value.",#
                                 :ekey => "`key`: Key to specify the universe in `sets.dict` that names resolve against. If `nothing`, the key is taken from `sets.xkey` — or, where the caller is written against another declared axis, from that axis' key.",#
                                 :bl_axis => "`axis`: Field of `sets` naming the declared axis the views resolve against: `:xkey` for the asset axis, `:fkey` for the factor axis. The key itself is read from `sets` here, and only when `sets` is not `nothing`.",#
                                 :sets_f => "`sets`: Universe sets. The **factor** axis, `sets.dict[sets.fkey]`, is what this estimator reads: it is the universe the views are written in, and it must name the columns of `F` in order. The asset axis is required by [`UniverseSets`](@ref) and is what a view slices — the factor entries come back from [`port_opt_view`](@ref) untouched.",#
                                 :sets_frb => "`sets`: Universe sets. The **factor** axis, `sets.dict[sets.fkey]`, is what this algorithm reads: it is the universe the risk budget is written in, and it must name the columns of `rr.L` in order — the budget is over the factor weights `w1`, one per column of the loadings the risk decomposition uses. It is only read when `rkb` is a [`RiskBudgetEstimator`](@ref); a [`RiskBudget`](@ref) result carries its own vector and resolves no names. The asset axis is required by [`UniverseSets`](@ref) and is what a view slices — the factor entries come back from [`port_opt_view`](@ref) untouched.",#
                                 :datatype => "`datatype`: Data type to use for the result in case `val` is `nothing`.",#
                                 :strict => "`strict`: Whether to throw an error if `sets` does not contain the desired value in `sets.dict[key]`.",#
                                 # Constraints
                                 :A => "`A`: Linear constraint coefficient matrix.",#
                                 :B => "`B`: Linear constraint response vector.",#
                                 :eq => "`eq`: Optional equality constraints.",#
                                 :ineq => "`ineq`: Optional inequality constraints.",#
                                 # Turnover.
                                 :tnr => "`tn`: Turnover result.",
                                 # Fees.
                                 :feese => "`fees`: Fees estimator.",#
                                 :feesr => "`fees`: Fees result.",
                                 # Stats.
                                 :sigma => "`sigma`: Covariance matrix `assets × assets`.",#
                                 :mu => "`mu`: Expected returns vector `assets × 1`.",#
                                 :rho => "`rho`: Correlation matrix `assets × assets`.",
                                 :sigrho => "`sigma`: Covariance-like or correlation-like matrix `assets × assets`.",
                                 :sigrhoX => "`X`: Covariance-like or correlation-like matrix `assets × assets`.",
                                 :kt => "`kt`: Cokurtosis matrix `assets^2 × assets^2`.",#
                                 :sk => "`sk`: Coskewness matrix `assets × assets^2`.",#
                                 :V => "`V`: Sum of the negative spectral slices of the coskewness matrix `assets × assets`.",
                                 :X => "`X`: Data matrix `observations × assets` if the `dims` keyword does not exist or `dims = 1`, `assets × observations` when `dims = 2`.",#
                                 :o_X => "`o_X`: The returns matrix the caller supplied, kept only when the carrier's own `X` is not it, and `nothing` otherwise. The three estimators that lift a factor-axis prior onto the asset axis overwrite `X` with the reconstruction `F * transpose(M) .+ transpose(b)`; `o_X` is the asset returns they were handed, over the same observations and the same assets. Read it as `original_X`, which is always a matrix, rather than as this field.",#
                                 :F => "`F`: Data matrix `observations × factors` if the `dims` keyword does not exist or `dims = 1`, `factors × observations` when `dims = 2`.",#
                                 :Xv => "`X`: Data vector `observations × 1`.",#
                                 :X_Xv => "`X`: Data matrix or vector.",#
                                 :Z => "`Z`: Feature matrix `assets × features` if `dims = 1`, `features × assets` when `dims = 2`. May also be a 3-D array of time-varying features, in which case the observation axis always leads: `observations × assets × features` if `dims = 1`, `observations × features × assets` when `dims = 2`.",#
                                 :Z_prior => "`Z`: Derived feature matrix, canonically assets-major: `assets × features` when static, `observations × assets × features` when time-varying. Nameless — feature names live on the `ReturnsResult` or come from a `UniverseSets`. Populated only by a producer that declares the matrix to be features; a user's `rd.Z` never reaches a prior result.",#
                                 :ze => "`ze`: Feature matrix estimator: the producer that computes `Z` from the wrapped prior result.",#
                                 :plfe => "`pl`: Structure source, always an estimator so that it refits per fold: a network estimator (a graph, whose `sep` measures the separations `alg` grades) or a clustering estimator (a partition, for which `alg` is inert). A precomputed result is not accepted -- an Estimator does not hold a Result.",#
                                 :plfalg => "`alg`: Phylogeny feature algorithm: the rule turning the source's separations into feature values. Inert for a partition source, which has no separation to grade.",#
                                 :dims => "`dims`: Dimension along which to perform the computation.",#
                                 :omean => "`mean`: Optional mean value to use for centering.",
                                 :ex => "`ex`: Parallel execution strategy.",#
                                 :alpha => "`alpha`: Quantile level for the lower tail.",#
                                 :beta => "`beta`: Quantile level for the upper tail.",#
                                 :l => "`l`: Risk aversion parameter.",#
                                 :ohf => "`ohf`: Objective homogenisation factor for the ratio problem, or `nothing` to size it from the resolved characteristic.",#
                                 :i_ret_term => "`i`: Index of the return term to maximise.",#
                                 :bgt_cost_target => "`bgt`: Budget target or range that the weights and their trading costs must meet together.",#
                                 :vp_cost => "`vp`: Cost coefficients for positive weight changes. Non-negative.",#
                                 :vn_cost => "`vn`: Cost coefficients for negative weight changes. Non-negative.",#
                                 :up_cost => "`up`: Upper limit on positive weight changes. Non-negative.",#
                                 :un_cost => "`un`: Upper limit on negative weight changes. Non-negative.",#
                                 :beta_mic => "`beta`: Reciprocal of the market impact exponent, `0 < beta < 1`. The realised exponent is `1/beta`.",#
                                 :rf => "`rf`: Risk-free rate.",#
                                 :bl_rf => "`rf`: Risk-free rate. The Black-Litterman update runs on excess returns, so a prior mean that arrives as a total return loses the rate first. The rate is added back exactly once, to the posterior asset expected returns. That round trip leaves the wrapped prior estimators alone, so a risk-free rate one of them applied internally stays where it is.",#
                                 # Errors
                                 :msg => "`msg`: Error message describing the condition that triggered the exception.",#
                                 # Solver
                                 :name => "`name`: Symbol or string identifier. It is also the **key** under which [`optimise_JuMP_model!`](@ref) files this solver's failure in [`JuMPResult`](@ref)'s `trials`, so two solvers that share a name share one entry and the later failure overwrites the earlier one. Give each solver of a vector its own name, because the default `\"\"` is shared by all of them.",#
                                 :solver => "`solver`: The `optimizer_factory` in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).",#
                                 :settings => "`settings`: Optional solver-specific settings used in [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute).",#
                                 :check_sol => "`check_sol`: Named tuple of keyword arguments splatted into [`assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.assert_is_solved_and_feasible) after each solve. It decides which solver statuses count as a solved model. The default `(;)` accepts JuMP's own defaults, `allow_local = true` and `allow_almost = false`: the termination status must be `OPTIMAL` or `LOCALLY_SOLVED`, and the primal status must be `FEASIBLE_POINT`. The strictness is deliberate — a solution the solver itself flags as approximate is rejected rather than silently accepted, so a solver stage that fails this check falls through to the next solver in the vector. The common relaxation is `check_sol = (; allow_local = true, allow_almost = true)`, which also accepts `ALMOST_OPTIMAL`, `ALMOST_LOCALLY_SOLVED` and `NEARLY_FEASIBLE_POINT`; it is what the examples, the user guide and the test suite pass, because a first-order solver reaching its tolerance on a well-posed portfolio problem is usually good enough. Go the other way with `allow_local = false` to reject `LOCALLY_SOLVED` and demand a certified global optimum, and add `dual = true` to also require a feasible dual point.",#
                                 :add_bridges => "`add_bridges`: The `add_bridges` keyword argument in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).",#
                                 # RNG
                                 :rng => "`rng`: Random number generator.",#
                                 :seed => "`seed`: Seed for the random number generator.",
                                 # JuMP Optimisation
                                 :model => "`model::JuMP.Model`: The JuMP optimisation model.",
                                 :opt_rjumpe => "`opt::RiskJuMPOptimisationEstimator`: Risk-based optimisation estimator.",
                                 :opt_jumpe => "`opt::JuMPOptimisationEstimator`: JuMP optimisation estimator.",
                                 :ci => "`i`: Constraint index for unique variable and constraint naming.",
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
                                 :b1_opt => "`b1::Option{<:MatNum} = nothing`: Factor loading matrix for [`FactorRiskContribution`](@ref); `nothing` for all other optimisers.",
                                 :optargs => "`args`: Additional positional arguments passed to the optimisation function.",
                                 :optkwargs => "`kwargs`: Additional keyword arguments passed to the optimisation function.",
                                 :ignargs => "`args`: Additional positional arguments (ignored).",
                                 :ignkwargs => "`kwargs`: Additional keyword arguments (ignored).",
                                 :rd => "`rd`: The returns result to use.",
                                 :window => "`window`: Observation window. An integer selects the last `window` observations, and a vector of indices selects those observations.",
                                 # Prior results.
                                 :chol => "`chol`: Cholesky factorisation of the covariance matrix.",#
                                 :w_prior => "`w`: Observation weights the prior was computed under `observations × 1` (see [`ObsWeights`](@ref)), or `nothing` if it was computed unweighted. Binds `ens`, `kld` and `ow`, which are diagnostics of it (see [`forward_prior`](@ref)).",#
                                 :ens => "`ens`: Effective sample size.",#
                                 :kld => "`kld`: Kullback-Leibler divergence of `w` from the weights it was derived from: a scalar against the prior observation weights for a single reweighting, or one entry per opinion when `w` came from pooling several.",#
                                 :fpr => "`fpr`: Prior result over the factor axis, or `nothing`. Its `X` is the factor returns matrix, so its `mu`, `sigma` and `w` describe factors rather than assets, over the same observations as the asset block.",#
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
                                 :sets_af => "`sets`: Universe sets. This estimator reads **both** declared axes: `a_views` resolves against `sets.dict[sets.xkey]`, `f_views` against `sets.dict[sets.fkey]`, and each axis must name the columns of `X` and `F` respectively, in order. Only the axis a [`LinearConstraintEstimator`](@ref) actually resolves names against is required — views supplied as a [`BlackLittermanViews`](@ref) result carry their own matrix and need no universe. A view slices the asset axis and leaves the factor entries untouched, which is why this field is `@vprop`.",#
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
                                 :ds_opt => "`ds_opt`: Thin wrapper for arguments and keyword arguments used in `Roots.findzero` for use with a single conditional value-at-risk view.",#
                                 :dm_opt => "`dm_opt`: Optimiser for multiple conditional value at risk views.",#
                                 :opt_ep => "`opt`: Entropy pooling optimisation estimator.",#
                                 :evar_views => "`evar_views`: Entropic value-at-risk views estimator or result.",#
                                 :sbar => "`sbar`: Number of largest losses considered by the integer conditional value-at-risk formulation. An `Integer` is a count, a fraction in `(0, 1]` is a fraction of the observations, and `nothing` applies the rule of thumb.",#
                                 :zpct => "`pct`: Fractional half-width of the grid of entropic value-at-risk dual variables, centred on the value that attains the prior entropic value-at-risk.",#
                                 :zK => "`K`: Number of points of the grid of entropic value-at-risk dual variables. Must be odd.",#
                                 :bigM => "`M`: Big-M constant of the grid entropic value-at-risk formulation.",#
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
                                 :eps_ucs => "`eps`: Radius of the ``\\\\ell_1`` uncertainty set on the characteristic vector. Larger values admit more estimation error, and therefore activate more assets.",#
                                 :ep_ucs => "`ep`: Radius of the positive-error side of the signed ``\\\\ell_1`` uncertainty set.",#
                                 :en_ucs => "`en`: Radius of the negative-error side of the signed ``\\\\ell_1`` uncertainty set.",#
                                 :sd_ucs => "`sd`: Per-asset scaling vector for the ``\\\\ell_1`` uncertainty set (the estimated standard deviations). `nothing` leaves the set unscaled, so every element of the characteristic vector is assumed to suffer the same estimation error.",#
                                 :mu_l1_ucs => "`mu`: Characteristic vector the ``\\\\ell_1`` set is a neighbourhood of. `nothing` defers to the consumer's own characteristic. When it is set, it takes precedence over the returns estimator's field and over the prior.",#
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
                                 :dict => "`dict`: Dictionary mapping group identifiers to member labels.",#
                                 :vars => "`vars`: Variable names in the parsed constraint expression.",#
                                 :coef_c => "`coef`: Coefficients corresponding to the constraint variables.",#
                                 :op => "`op`: Comparison operator (`==`, `<=`, or `>=`).",#
                                 :rhs => "`rhs`: Right-hand side value of the constraint.",#
                                 :rhs_rho => "`rhs`: Right-hand side of the constraint. A view over a single asset pair carries one value. A view over a pair of groups carries one value per spanned pair, in the order of `ij`.",#
                                 :eqn => "`eqn`: Formatted string representation of the constraint equation.",#
                                 :ij => "`ij`: Pair of asset indices for correlation-based constraints.",#
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
                                 # Frontier.
                                 :N_fr => "`N`: Number of sweep points on the efficient frontier. The sweep solves the model `N` times, at `N` evenly spaced bound values.",#
                                 :factor_fr => "`factor`: Multiplier applied to both ends of the sweep span after `bound` has transformed them. It carries a formulation's own correction factor, such as the `inv(1 / (T - ddof))` of a second-moment bound.",#
                                 :bound_fr => "`bound`: [`FrontierBoundEstimator`](@ref) that converts a bound value into the units of the risk expression the bound is applied to. The sweep points are evenly spaced in **those** units, not in the units of the measure.",#
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
                                 :b_mip => "`b`: Big-M constant of the MIP formulation. It relaxes the bound on an observation that the model flags as an exceedance. If `nothing`, the model uses `1000`.",#
                                 :s_mip => "`s`: Cardinality slack of the MIP formulation. It caps the number of flagged observations at `(alpha - s) * T`. If `nothing`, the model uses `1e-5`.",#
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
                                 :mu_ret_slot => "`mu`: Optional expected returns vector `assets × 1`. Also admits a **Deferred Quantity** — an expected returns estimator or a prior estimator that computes the vector against the optimisation's own prior, at [`factory`](@ref) time (see [`ArithRetMu`](@ref) and [`resolve_deferred_quantities`](@ref)). A `ucs` that carries its own centre outranks it, and it outranks the prior's own vector (ADR 0050). If `nothing`, the prior supplies it.",#
                                 :ddof => "`ddof`: Degrees-of-freedom correction.",#
                                 :flag => "`flag`: Algorithm selection flag.",#
                                 # Turnover.
                                 :w_tn => "`w`: Reference portfolio weight vector. Deviations are measured against it, and it is never the candidate weight vector an optimiser solves for.",#
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
                                 :r_res => "`r`: The risk measure the optimisation ran under, or a vector of them, stored **resolved** — a **Deferred Quantity** has already been fitted and an unstated slot has already taken the prior's field. A resolved measure is fitted state, not configuration, so it belongs on the Result. Pass it back as `expected_risk(res.r, res.w, res.pr; sca = res.sca)`.",#
                                 :obj => "`obj`: Portfolio objective function.",#
                                 :wi => "`wi`: Initial portfolio weights for warm-starting the solver.",#
                                 :sca => "`sca`: Scalariser for combining multiple risk measures.",#
                                 :sca_res => "`sca`: The scalariser the optimisation ran under, taken from `opt.sca`. Pass it back as `expected_risk(res.r, res.w, res.pr; sca = res.sca)` so the reported figure matches the optimised one.",#
                                 :wb_jmp => "`wb`: Weight bounds estimator or weight bounds.",#
                                 :bgt => "`bgt`: Net budget, `1ᵀw`. A number pins it, a [`BudgetRange`](@ref) bounds it. By default budgets *bound* the realised exposure rather than pinning it (see `xbgt`). Together with `sbgt` this fixes the net and gross exposures only jointly; to constrain the gross exposure on its own see `gbgt`.",#
                                 :sbgt => "`sbgt`: Short-side budget, `sum(sw)`. A number pins it, a [`BudgetRange`](@ref) bounds it; by default it *bounds*, so `sbgt = 0.3` means *at most* 30% short unless `xbgt` pins the long/short decomposition. Together with `bgt` this fixes the net and gross exposures only jointly; to constrain the gross exposure on its own see `gbgt`.",#
                                 :gbgt => "`gbgt`: Gross budget (leverage) constraint, `sum(lw) + sum(sw)`. A number pins the gross exposure; a [`BudgetRange`](@ref) bounds it, e.g. `BudgetRange(; lb = nothing, ub = 2.0)` caps leverage at 2x. Unlike `bgt` and `sbgt` — which pin the net and gross exposures only *together* — this constrains the gross exposure on its own, leaving the net free. Requires weight bounds that admit short positions, and is bounded rather than pinned unless `xbgt` is set.",#
                                 :xbgt => "`xbgt`: Whether to pin the long/short decomposition exactly. When `false` (the default), `lw` and `sw` are upper bounds on the positive and negative parts of `w`, so `bgt`, `sbgt` and `gbgt` bound the realised exposures rather than pinning them — a short budget of `0.3` means *at most* 30% short. When `true`, the long/short binary indicators force `lw == max(w, 0)` and `sw == max(-w, 0)`, so the budgets hold exactly, at the cost of turning the problem into a mixed-integer program. It reuses the indicators the cardinality, threshold and fee builders already create (see `short_mip_threshold_constraints`) rather than adding its own, and is ignored when the weight bounds admit no shorts.",#
                                 :lt => "`lt`: Long-side minimum holding threshold.",#
                                 :st => "`st`: Short-side minimum holding threshold.",#
                                 :lcse => "`lcse`: Linear constraint set estimator(s). This is the one constraint slot that also admits an `ExposureConstraintEstimator`, so a row may be written in the names of another basis — factor names, say — and re-based through the prior's loadings at generation time. What reaches the model is an ordinary asset-space `LinearConstraint` either way.",#
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
                                 :ple_jmp => "`ple`: Phylogeny constraint estimator(s).",#
                                 :lcsr => "`lcsr`: Processed linear constraint set result.",#
                                 :gcardr => "`gcardr`: Processed grouped cardinality constraint result.",#
                                 :sgcardr => "`sgcardr`: Processed sub-grouped cardinality constraint result.",#
                                 :ret_jmp => "`ret`: Return term, or vector of return terms, for the `JuMP` model. Several terms are weighted-summed into the model's single scalar return expression, in the same way [`MeanRisk`](@ref)'s `r` takes several risk measures.",#
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
                                 :x_src => "`x_src`: Which returns matrix the clustering, phylogeny and centrality estimators read: `:prior` takes the prior result's `X`, `:data` takes the raw returns result's `X`. Ignored when no returns result is available, in which case the prior result's `X` is used.",#
                                 :z_src => "`z_src`: Which feature matrix a [`FeatureDistance`](@ref) inside the clustering, phylogeny or centrality estimator reads: `:data` takes the raw returns result's `Z`, `:prior` takes the prior result's `Z`. It defaults to `:data` — the opposite of `x_src` — because an explicitly supplied feature matrix outranks a derived one. Ignored when no [`FeatureDistance`](@ref) is present; a [`FeatureDistance`](@ref) with no `Z` on the selected carrier throws. A `Z` that is carried but unused stays silent, matching `iv`, `ivpa`, `F` and `B` — including the one sub-case that *is* a configuration error and is deliberately not detected: setting `z_src` explicitly with no [`FeatureDistance`](@ref) anywhere in the estimator tree. Detecting it would mean walking an arbitrary estimator tree for a type, which the layer resolving `z_src` cannot do.",#
                                 :wf => "`wf`: Weight finaliser.",#
                                 :rkb => "`rkb`: Risk budget estimator or result.",#
                                 :rba => "`rba`: Risk budget algorithm.",#
                                 :resi => "`resi`: Inner optimisation results.",#
                                 :reso => "`reso`: Outer optimisation results.",#
                                 :opti => "`opti`: Inner optimiser.",#
                                 :opto => "`opto`: Outer optimiser.",#
                                 # Cross-validation.
                                 :n_folds => "`n`: Number of folds.",#
                                 :n_test_folds => "`n_test_folds`: Number of folds held out for testing in each combination. The remaining `n_folds - n_test_folds` folds train.",#
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
                                 # Cluster node.
                                 :id_node => "`id`: Node identifier.",#
                                 :left_node => "`left`: Left child node.",#
                                 :right_node => "`right`: Right child node.",#
                                 :height_node => "`height`: Height of the node in the dendrogram.",#
                                 :level_node => "`level`: Number of leaves in the subtree rooted at the node, `1` for a leaf. It is the fourth column of a linkage matrix, and [`pre_order`](@ref) sizes its traversal stack from it.",#
                                 # Other.
                                 :dlb => "`dlb`: Default lower bound.",#
                                 :dub => "`dub`: Default upper bound.",#
                                 :err => "`err`: Tracking error tolerance.",#
                                 :tralg => "`alg`: Tracking formulation algorithm.",#
                                 :rt => "`rt`: Returns estimator, or a vector of them. A vector is summed at its terms' `settings.scale` weights, skipping any term whose `settings.rte` is `false`. There is no scalariser on the return axis.",#
                                 :rk => "`rk`: Risk measure for ratio computation, or a vector of them scalarised by `sca`.",#
                                 :r1 => "`r1`: First risk measure.",#
                                 :r2 => "`r2`: Second risk measure.",#
                                 :r1_vec => "`r1`: First risk measure, or a vector of them scalarised by `sca1`.",#
                                 :r2_vec => "`r2`: Second risk measure, or a vector of them scalarised by `sca2`.",#
                                 :sca_rk => "`sca`: Scalariser combining the risk measures in `rk` into one number. Inert when `rk` holds a single measure. The field beats a `sca` keyword supplied at the call site.",#
                                 :sca_r1 => "`sca1`: Scalariser combining the risk measures in `r1` into one number. Inert when `r1` holds a single measure.",#
                                 :sca_r2 => "`sca2`: Scalariser combining the risk measures in `r2` into one number. Inert when `r2` holds a single measure.",#
                                 :ri => "`ri`: Inner risk measure.",#
                                 :ri_res => "`ri`: The intra-cluster risk measure the optimisation ran under, or a vector of them, stored **resolved**.",#
                                 :ro_res => "`ro`: The inter-cluster risk measure the optimisation ran under, or a vector of them, stored **resolved**.",#
                                 :ro => "`ro`: Outer risk measure.",#
                                 :scai => "`scai`: Inner scalariser.",#
                                 :scao => "`scao`: Outer scalariser.",#
                                 :params => "`params`: Schur complement decomposition parameters.",#
                                 :gamma_schur => "`gamma`: Schur complement interpolation parameter, in `[0, 1]`. At `0` no augmentation happens and the allocation is exactly [`HierarchicalRiskParity`](@ref) under `r`. A larger value subtracts more of the cross-cluster block from each sub-cluster covariance, which moves the allocation towards the minimum variance portfolio. Under [`MonotonicSchurComplement`](@ref) it is the **upper end** of the searched range, not the value used.",#
                                 :gamma_schur_res => "`gamma`: The Schur complement interpolation parameter the allocation ran at. It parallels `r`: one value for the single-bundle path, one per bundle for the multi-bundle path. Under [`MonotonicSchurComplement`](@ref) this is the value the search chose, which is at most the `gamma` the estimator asked for.",#
                                 :flag_schur => "`flag`: Whether to repair an augmented covariance block that is not positive definite. When `true`, `pdm` repairs it, and a failed repair raises. When `false`, no repair happens and the allocation is abandoned instead, which is what the [`MonotonicSchurComplement`](@ref) search needs; a caller that keeps the weights gets an error naming the `gamma` that failed.",#
                                 :r_res_schur => "`r`: The risk measure the optimisation ran under, stored **resolved**. It parallels `gamma`: one measure for the single-bundle path, a vector of them for the multi-bundle path. Schur carries **no** scalariser, because it carries no vector of measures to combine — `SchurComplementParams.r` is bounded to a standard deviation or a variance.",#
                                 :tol => "`tol`: Convergence tolerance.",#
                                 :iter => "`iter`: Maximum number of iterations.",#
                                 :w_opt_noc => "`w_opt`: Optimal portfolio weights.",#
                                 :w_min_noc => "`w_min`: Minimum risk portfolio weights.",#
                                 :w_max_noc => "`w_max`: Maximum return portfolio weights.",#
                                 :ucs_flag => "`ucs_flag`: Whether to use the uncertainty set.",#
                                 # Optimiser config.
                                 :kwargs => "`kwargs`: Additional keyword arguments.",#
                                 # Risk measure.
                                 :r => "`r`: Risk measure or vector of risk measures.",#
                                 # Weight bounds.
                                 :wb => "`wb`: Weight bounds.",#
                                 # Tracking.
                                 :tr => "`tr`: Tracking error constraint estimator.",#
                                 # Fees.
                                 :fees => "`fees`: Fees estimator or result.",#
                                 # Near optimal centering result fields.
                                 :attrs_noc => "`attrs`: Processed JuMP optimiser attributes for the model-assembly pipeline.",#
                                 :w_opt => "`w_opt`: Optimal portfolio weights (vector or vector of vectors).",#
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
                                 :rt_ends => "`rt_ends`: Per-term return spans for a return-frontier sweep, as `i => (rt_min_i, rt_max_i)` pairs, or `nothing` when no return term declares a frontier bound. The aggregate `rt_min`/`rt_max` pair above serves the barrier; these serve the sweep, and the two are different quantities because a term's own span must be read off a portfolio that maximised that term alone.",#
                                 :rk_opt => "`rk_opt`: Optimal risk target.",#
                                 :noc_retcode => "`noc_retcode`: Return code for the near-optimal centering sub-problem.",#
                                 # Discrete allocation result fields.
                                 :l_model => "`l_model`: `JuMP` model for the long allocation.",#
                                 :s_model => "`s_model`: `JuMP` model for the short allocation.",#
                                 :l_retcode => "`l_retcode`: Return code for the long allocation sub-problem.",#
                                 :s_retcode => "`s_retcode`: Return code for the short allocation sub-problem.",#
                                 # Risk budgeting.
                                 :prb => "`prb`: Processed risk budgeting configuration.",#
                                 :l_wass => "`l`: Weight of the tail term in the Esfahani-Kuhn loss. The mean term is not scaled by it.",#
                                 :r_wass => "`r`: Radius of the type-1 Wasserstein ambiguity ball. It multiplies a decision variable, so it is not a constant offset.",#
                                 :g_rm => "`g`: Risk aversion parameter.",#
                                 :max_phi => "`max_phi`: Maximum allowed value for any OWA weight.",#
                                 :w1_owa => "`w1`: Optional first OWA weight vector.",#
                                 :w2_owa => "`w2`: Optional second OWA weight vector.",#
                                 :rev_owa => "`rev`: Whether `w2` is *already* reversed. It is a done-flag, not an instruction: the constructor reverses `w2` when `rev == false`, and leaves it as-is when `rev == true`. The field is stored as `true` whatever the caller passes, because `w2` is reversed by the time the object exists, so rebuilding an instance from its own fields does not reverse twice. A default-constructed instance therefore prints `rev` as `true`.",#
                                 :owa_w => "`w`: Optional OWA weight vector.",#
                                 :owa_method => "`method`: OWA weight estimation method.",#
                                 :lm_k => "`k`: L-moment order.",#
                                 :alpha_i => "`alpha_i`: Lower integration bound for the tail Gini approximation.",#
                                 :a_sim => "`a_sim`: Number of integration points for the tail Gini approximation.",#
                                 :beta_i => "`beta_i`: Lower integration bound for the upper tail Gini approximation.",#
                                 :b_sim => "`b_sim`: Number of integration points for the upper tail Gini approximation.",#
                                 # Constraint generation.
                                 :rkb_val => "`val`: Vector of non-negative risk budgets, one per entry of the axis the budget is written against. [`risk_budget_constraints`](@ref) normalises it to sum to one; a hand-built vector is stored as given, and the model reads it inside a logarithmic barrier, so only its **relative** entries matter.",#
                                 :rkbe_val => "`val`: Mapping of names to risk budget values. A name may be an asset or a group, and a group assigns its value to every asset in it. A scalar is accepted and resolves to `RiskBudget(1.0)` whatever the scalar was, so only a one-entry axis can consume it. Write the uniform budget as `nothing`.",#
                                 :us_xkey => "`xkey`: Key in `dict` identifying the primary asset list. Required, and the axis a view slices.",#
                                 :us_uxkey => "`uxkey`: Key prefix for unique-entry asset group variants in `dict`.",#
                                 :us_fkey => "`fkey`: Key in `dict` identifying the factor list. Optional — a consumer that needs it and does not find it throws at the point of need.",#
                                 :us_ufkey => "`ufkey`: Key prefix for unique-entry factor group variants in `dict`. Validated at construction, never recomputed by a view.",#
                                 :us_zkey => "`zkey`: Key in `dict` identifying the declared feature axis — the node list a graded feature program writes its columns against. Optional, like `fkey`, and it carries no prefix convention: nothing is partitioned over the feature axis, so it has no unique-entry sibling and no length rule beyond `allunique`.",#
                                 :p_phylo => "`p`: Non-negative penalty factor on the trace of the semidefinite matrix variable. It is read **only** when the model does not already minimise a variance: a variance objective is itself a trace against that variable, so it pulls the relaxation down on its own and no second term is added.",#
                                 :A_phylo => "`A`: Symmetric relatedness matrix with a zero diagonal. A network source gives the range connection matrix, a clustering source the adjacency label matrix. Stored as given.",#
                                 :A_iphylo => "`A`: Row set of the relatedness matrix, stored as `unique(A + I; dims = 1)` and **not** as the matrix passed in. The identity puts each asset in its own row, and the deduplication drops rows that repeat. One row per distinct neighbourhood or cluster survives, which is why the stored matrix is usually shorter than it is wide.",#
                                 :B_phylo => "`B`: Right-hand side of `A * z <= B`, where `z` is the held indicator: the largest number of assets that may be held out of each row of `A`. A scalar applies to every row. A vector states one bound per row, so its length must match the row count of the stored `A` and not the number of assets. On an estimator the rows do not exist yet, so a vector is only checked against the largest number of clusters the clustering estimator can return.",#
                                 :cc_A => "`A`: Centrality estimator. Its centrality vector is the row of the generated linear constraint.",#
                                 :cc_B => "`B`: Right-hand side of the constraint. A number is the threshold itself. A [`VectorToScalarMeasure`](@ref) derives the threshold from the centrality vector `A` produces, so the constraint always has a feasible point.",#
                                 :cc_comp => "`comp`: Comparison operator for the centrality constraint. `==` builds an equality row, every other operator an inequality row.",#
                                 :lce_val => "`val`: Constraint equation(s) to parse.",#
                                 :ece_lce => "`lce`: Wrapped linear constraint estimator(s) or precomputed constraint, written in the names of the space's basis. Exactly what `lcse` itself accepts, so no shape can reach the optimiser un-re-based.",#
                                 :ece_space => "`space`: Basis the wrapped constraint is written in. Required — the absence of a re-basis is spelled by using a bare `LinearConstraintEstimator`, not by a space member.",#
                                 :fs_re => "`re`: Source of the loadings the rows are re-based through, or `nothing` to read the prior's `rr`. A precomputed `Regression` states the basis outright; an estimator fits one from the returns, which is what makes a factor mandate legal on a prior that carries no factor block. The precedence is `resolve_factor_regression`'s: a precomputed result wins, then the prior's `rr`, then a refit.",#
                                 :asets_val => "`val`: Group name key for asset set membership matrix extraction.",#
                                 :asets_vals => "`vals`: Either group name keys whose partitions are stacked into the feature axis, at least two (one partition alone is one-hot, which makes the distance two-valued for every metric), or an ordered edge-authoring program of `Pair`s over the declared feature axis `sets.dict[sets.zkey]`. The two are dispatched on element type and are different contracts — see [`asset_sets_features`](@ref).",#
                                 :asets_strict => "`strict`: Whether an unresolvable *name* in a graded `vals` program throws instead of warning. Governs names only: nothing structural is refused, so an all-zero row and a one-column matrix are both legal. Ignored on the group-name-key path, where an absent key is an unconditional `KeyError`.",#
                                 :thr_val => "`val`: Asset-specific minimum-holding threshold value(s).",#
                                 :thr_res_val => "`val`: Minimum-holding threshold(s) on the portfolio weights. A held position must reach its threshold; a position below it is driven to zero. The threshold binds the **held** weight, never the trade, so a reference portfolio does not enter it.",#
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
                                 :schalg => "`alg`: Schur complement algorithm variant.",#
                                 :ps_n_periods => "`n_periods`: Number of observations in the return series.",#
                                 :ps_ppy => "`periods_per_year`: Annualisation factor. 252 for daily, 52 for weekly, 12 for monthly returns.",#
                                 :ps_alpha => "`alpha`: Tail probability used for the CVaR, ``\\alpha \\in (0, 1)``.",#
                                 :ps_compound => "`compound`: Whether the wealth path behind the drawdown statistics was compounded.",#
                                 :ps_ann_return => "`ann_return`: Annualised arithmetic mean return.",#
                                 :ps_ann_volatility => "`ann_volatility`: Annualised sample standard deviation.",#
                                 :ps_sharpe => "`sharpe`: Annualised Sharpe ratio at a zero risk-free rate. `NaN` if the volatility is zero.",#
                                 :ps_sharpe_stderr => "`sharpe_stderr`: Standard error of `sharpe`, corrected for the skewness and excess kurtosis of the returns.",#
                                 :ps_sortino => "`sortino`: Annualised Sortino ratio, at a zero minimum acceptable return. `NaN` if the downside deviation is zero.",#
                                 :ps_calmar => "`calmar`: Annualised return divided by the absolute maximum drawdown. `NaN` if there is no drawdown.",#
                                 :ps_max_drawdown => "`max_drawdown`: Maximum drawdown, in return space, so it is non-positive.",#
                                 :ps_cvar => "`cvar`: Conditional Value-at-Risk at `alpha`, in return space, so a tail loss is negative.")
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
const err_name_dict = unique_key_dict(:err_name_dict, :kt => "cokurtosis",
                                      :sk => "coskewness",
                                      :V => "negative spectral coskewness",
                                      :D2 => "duplication matrix",
                                      :L2 => "elimination matrix",
                                      :S2 => "summation matrix")
"""
    const val_dict = Dict(:oow => "If `w` is not `nothing`, `!isempty(w)`.")

Validation rules for certain arg_dict terms used in the documentation of `PortfolioOptimisers.jl`.

`:relax` is the exception: it is the fixed opening sentence of a `## Relaxation` subsection under `# JuMP formulation`, held here so that the wording cannot drift between docstrings.
"""
const val_dict = unique_key_dict(:val_dict,
                                 :oow => "If `w` is not `nothing`, `!isempty(w)`.",
                                 :oidx => "If `idx` is not `nothing`, `!isempty(idx)` and all indices are positive integers.",
                                 :gerbt => "`0 <= t`.",#
                                 :t => "`0 < t < 1`.",#
                                 :c1 => "`0 < c1 <= 1`.",#
                                 :c2 => "`0 < c2 <= 1`.",#
                                 :c3c2 => "`c3 > c2`.",#
                                 :dims => "`dims in (1, 2)`.",#
                                 :alpha => "`0 < alpha < 1`.",#
                                 :beta => "`0 < beta < 1`.",#
                                 :bins => "If `bins` is an integer, `0 < bins <= RESOURCE_LIMITS[].max_bins` (the joint histogram is `bins × bins`; see [`RESOURCE_LIMITS`](@ref)).",#
                                 :dopower => "If `power` is not `nothing`, `power >= 1`.",#
                                 :p_owa => "`!isempty(p)` and `all(x -> x > 1, p)`.",#
                                 :settings => "If not `nothing`, `!isempty(settings)`.",#
                                 :S => "`!isempty(S)`.",#
                                 :D => "`!isempty(D)`.",#
                                 :ck => "`k >= 1`.",#
                                 :lm_k => "`k >= 2`.",#
                                 :alpha_i_alpha => "`0 < alpha_i < alpha < 1`.",#
                                 :a_sim_pos => "`a_sim > 0`.",#
                                 :beta_i_beta => "`0 < beta_i < beta < 1`.",#
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
                                 :A => "`!isempty(A)`.",#
                                 :B => "`!isempty(B)`.",#
                                 :A_B => "`size(A, 1) == length(B)`, one row of `A` per entry of `B`.",#
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
                                 :ra_norm_x => "`x` is valid",#
                                 :relax => "The encoding is not exact: the rows below bound the quantity instead of reproducing it, and the bound is tight only under the condition stated here.")

"""
Dictionary containing return value descriptions for common parameters used in `PortfolioOptimisers.jl`.
"""
const ret_dict = unique_key_dict(:ret_dict,
                                 :mu => "`mu::ArrNum`: Expected returns vector `assets x 1` if the `dims` keyword does not exist or `dims = 2`, `1 x assets` if `dims = 1`.",#
                                 :sigma => "`sigma::MatNum`: Covariance matrix `assets x assets`.",#
                                 :rho => "`rho::MatNum`: Correlation matrix `assets x assets`.",#
                                 :sigrho => "`sigrho::MatNum`: Covariance/correlation matrix `assets x assets`.",#
                                 :sk => "`sk::MatNum`: Coskewness matrix `assets x assets`.",#
                                 :cskew => "`cskew::MatNum`: Coskewness tensor `assets x assets²`.",#
                                 :cskewV => "`V::MatNum`: Processed coskewness matrix `assets x assets`.",#
                                 :kte => "`kte::MatNum`: Cokurtosis matrix `assets x assets`.",#
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
                       :x_t_obs => "``\\boldsymbol{x}_t``: Asset returns for observation ``t``, the ``t``-th row of the returns matrix.",#
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
                       :sc_scale => "``s_c``: Constraint scale. It multiplies both sides of a row, so a positive value leaves the feasible set unchanged.",#
                       :mu_er => "``\\boldsymbol{\\mu}``: Expected returns vector ``N \\times 1``.",#
                       :R_w => "``R(\\boldsymbol{w})``: Portfolio risk.",#
                       # Second-moment formulations.
                       :d_secmom => "``\\boldsymbol{d}``: Deviation vector ``T \\times 1`` that the formulation squares. The risk measure supplies it.",#
                       :c_secmom => "``c``: Correction factor that the risk measure supplies. It is ``1`` when the co-moment matrix already carries it.",#
                       :t_secmom => "``t``: Auxiliary model variable that the cone bounds.",#
                       # Weight finalisation.
                       :w_0_finaliser => "``\\boldsymbol{w}_{0}``: Portfolio weights vector ``N \\times 1`` that the optimisation produced, which the finaliser repairs.",#
                       :lb_ub_finaliser => "``\\boldsymbol{l}``, ``\\boldsymbol{u}``: Lower and upper weight bounds. An absent bound is dropped from the programme rather than set to an infinity.",#
                       # The Range convention (ADR 0057).
                       :negated_upper_tail => "The upper tail is the base measure applied to the negated returns ``-\\boldsymbol{x}``, so both tails are reported on the same sign convention and the range is their sum, not their difference.")
"""
    ref_dict

Maps a key of `docs/src/References.bib` to the formatted `# References` bullet for that
work, so the reference text is written once here and interpolated wherever a docstring
cites it.

Each value is a complete bullet body: the citation marker for the key, followed by the
reference in the style `DocumenterCitations` renders in the bibliography. A citing docstring
writes the whole bullet as one interpolation of this table and never pastes the reference
prose inline. A pasted copy drifts from the entry in `References.bib` and from the other
copies of itself: before this table existed, `gerber2025squeezing` was pasted 31 times and
`mlp1` 13 times.

The `const` definition below is the single source of truth. A key must appear once:
[`unique_key_dict`](@ref) builds the table and refuses a repeat.

# Related

  - [`unique_key_dict`](@ref)
  - [`math_dict`](@ref)
"""
const ref_dict = unique_key_dict(:ref_dict,
                                 :brinson_attribution => "[brinson_attribution](@cite) G. P. Brinson and N. Fachler. *Measuring non-US. equity portfolio performance*. The Journal of Portfolio Management 11, 73–76 (1985).",#
                                 :bergstra2012 => "[bergstra2012](@cite) J. Bergstra and Y. Bengio. *Random search for hyper-parameter optimization*. Journal of Machine Learning Research 13, 281–305 (2012).",#
                                 :DBHTs => "[DBHTs](@cite) W.-M. Song, T. Di Matteo and T. Aste. *Hierarchical information clustering by means of topologically embedded graphs*. PloS one 7, e31929 (2012).",#
                                 :drcvar => "[drcvar](@cite) P. Mohajerin Esfahani and D. Kuhn. *Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations*. Mathematical Programming 171, 115–166 (2018).",#
                                 :freedman1981 => "[freedman1981](@cite) D. Freedman and P. Diaconis. *On the histogram as a density estimator: L2 theory*. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete 57, 453–476 (1981).",#
                                 :gerber => "[gerber](@cite) S. Gerber, H. Markowitz, P. Ernst, Y. Miao, P. Sargen and others. *The Gerber statistic: A robust co-movement measure for portfolio optimization*. Available at SSRN 3880054 (2021).",#
                                 :gerber_analysis => "[gerber_analysis](@cite) E. Flint and D. Polakow. *Deconstructing the Gerber statistic*. Finance Research Letters 56, 104144 (2023).",#
                                 :gerber2025squeezing => "[gerber2025squeezing](@cite) S. Gerber, W. Smyth, H. Markowitz, Y. Miao, P. Ernst and P. Sargen. *Squeezing financial noise: A novel approach to covariance matrix estimation*. Available at SSRN 4986939 (2025).",#
                                 :J_LoGo => "[J_LoGo](@cite) W. Barfuss, G. P. Massara, T. Di Matteo and T. Aste. *Parsimonious modeling with information filtering networks*. Phys. Rev. E 94, 062306 (2016).",#
                                 :fengpalomar2016 => "[fengpalomar2016](@cite) Y. Feng and D. P. Palomar. *A signal processing perspective of financial engineering*. Foundations and Trends in Signal Processing 9, 1–231 (2016).",#
                                 :fabozzi2007 => "[fabozzi2007](@cite) F. J. Fabozzi, P. N. Kolm, D. A. Pachamanova and S. M. Focardi. *Robust Portfolio Optimization and Management* (John Wiley & Sons, Hoboken, NJ, 2007).",#
                                 :sousalobo2000 => "[sousalobo2000](@cite) M. Sousa Lobo and S. Boyd. *The worst-case risk of a portfolio*. Technical report, Stanford University (2000).",#
                                 :higham2002 => "[higham2002](@cite) N. J. Higham. *Computing the nearest correlation matrix—a problem from finance*. IMA Journal of Numerical Analysis 22, 329–343 (2002).",#
                                 :qisun2006 => "[qisun2006](@cite) H. Qi and D. Sun. *A quadratically convergent Newton method for computing the nearest correlation matrix*. SIAM Journal on Matrix Analysis and Applications 28, 360–385 (2006).",#
                                 :gmd => "[gmd](@cite) S. Yitzhaki. *Stochastic dominance, mean variance, and Gini's mean difference*. The American Economic Review 72, 178–185 (1982).",#
                                 :knuth2019 => "[knuth2019](@cite) K. H. Knuth. *Optimal data-based binning for histograms and histogram-based probability density models*. Digital Signal Processing 95, 102581 (2019).",#
                                 :kunsch1989 => "[kunsch1989](@cite) H. R. Künsch. *The jackknife and the bootstrap for general stationary observations*. The Annals of Statistics 17, 1217–1241 (1989).",#
                                 :markowitz1952 => "[markowitz1952](@cite) H. Markowitz. *Modern portfolio theory*. Journal of Finance 7, 77–91 (1952).",#
                                 :mlp1 => "[mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020).",#
                                 :lopezdeprado2018 => "[lopezdeprado2018](@cite) M. López de Prado. *Advances in Financial Machine Learning* (John Wiley & Sons, Hoboken, NJ, 2018).",#
                                 :mpdist => "[mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).",#
                                 :NHPG => "[NHPG](@cite) W.-M. Song, T. Di Matteo and T. Aste. *Nested hierarchies in planar graphs*. Discrete Applied Mathematics 159, 2135–2146 (2011).",#
                                 :owa1 => "[owa1](@cite) D. Cajas. *OWA portfolio optimization: A disciplined convex programming framework*. Available at SSRN 3988927 (2021).",#
                                 :owaog => "[owaog](@cite) W. Ogryczak and T. Śliwiński. *On solving linear programs with the ordered weighted averaging objective*. European Journal of Operational Research 148, 80–91 (2003).",#
                                 :owa2 => "[owa2](@cite) D. Cajas. *Higher order moment portfolio optimization with L-moments*. Available at SSRN 4393155 (2023).",#
                                 :owa3 => "[owa3](@cite) D. Cajas. *Efficient Gini Mean Difference and Tail Gini Portfolio Optimization based on P-Norms*. Available at SSRN 4711326 (2024).",#
                                 :PMFG => "[PMFG](@cite) G. P. Massara, T. Di Matteo and T. Aste. *Network Filtering for Big Data: Triangulated Maximally Filtered Graph*. Journal of Complex Networks 5, 161–178 (2016).",#
                                 :tgini => "[tgini](@cite) W. Ogryczak and A. Ruszczyński. *Dual stochastic dominance and quantile risk measures*. International Transactions in Operational Research 9, 661–680 (2002).",#
                                 :politis1992circular => "[politis1992circular](@cite) D. N. Politis and J. P. Romano. *A circular block-resampling procedure for stationary data*. In: *Exploring the Limits of Bootstrap* (John Wiley & Sons, 1992); pp. 263–270.",#
                                 :politis1994stationary => "[politis1994stationary](@cite) D. N. Politis and J. P. Romano. *The stationary bootstrap*. Journal of the American Statistical Association 89, 1303–1313 (1994).",#
                                 :quintile => "[quintile](@cite) R. Zhou and D. P. Palomar. *Understanding the Quintile Portfolio*. IEEE Transactions on Signal Processing 68, 4030–4040 (2020).",#
                                 :scott1979 => "[scott1979](@cite) D. W. Scott. *On optimal and data-based histograms*. Biometrika 66, 605–610 (1979).",#
                                 :sharpe_stderr => "[sharpe_stderr](@cite) D. H. Bailey and M. Lopez de Prado. *The Sharpe ratio efficient frontier*. Journal of Risk 15, 3–44 (2012).",#
                                 :smyth2022enhanced => "[smyth2022enhanced](@cite) W. Smyth and D. Broby. *An enhanced Gerber statistic for portfolio optimization*. Finance Research Letters 49, 103229 (2022).",#
                                 :EPTail => "[EPTail](@cite) D. Cajas. *Entropy Pooling with CVaR and EVaR Views*. Available at SSRN 7120258 (2026).",#
                                 :meucci2008 => "[meucci2008](@cite) A. Meucci. *Fully flexible views: theory and practice*. Risk 21, 97–102 (2008).",#
                                 :meucciardiakeel2011 => "[meucciardiakeel2011](@cite) A. Meucci, D. Ardia and S. Keel. *Fully flexible extreme views*. The Journal of Risk 14, 39–49 (2011).",#
                                 :vorobets2021 => "[vorobets2021](@cite) A. Vorobets. *Sequential entropy pooling heuristics*. Available at SSRN 3936392 (2021).",#
                                 :cvar => "[cvar](@cite) R. T. Rockafellar and S. Uryasev. *Optimization of conditional value-at-risk*. Journal of Risk 2, 21–41 (2000).",#
                                 :evar => "[evar](@cite) A. Ahmadi-Javid. *Entropic value-at-risk: A new coherent risk measure*. Journal of Optimization Theory and Applications 155, 1105–1123 (2012).",#
                                 :rlvar => "[rlvar](@cite) D. Cajas. *Portfolio Optimization of Relativistic Value at Risk*. Available at SSRN 4378498 (2023).",#
                                 :cdar => "[cdar](@cite) A. Chekhlov, S. Uryasev and M. Zabarankin. *Drawdown measure in portfolio optimization*. International Journal of Theoretical and Applied Finance 8, 13–58 (2005).",#
                                 :pnvar => "[pnvar](@cite) P. A. Krokhmal. *Higher moment coherent risk measures*. Quantitative Finance 7, 373–387 (2007).",#
                                 :ulcer => "[ulcer](@cite) P. G. Martin and B. B. McCann. *The Investor's Guide to Fidelity Funds* (John Wiley & Sons, 1989).",#
                                 :minimax => "[minimax](@cite) M. R. Young. *A minimax portfolio selection rule with linear programming solution*. Management Science 44, 673–683 (1998).",#
                                 :bdvar => "[bdvar](@cite) D. Cajas. *Portfolio Optimization of Brownian Distance Variance*. Available at SSRN 4561293 (2023).",#
                                 :rousseeuw1993 => "[rousseeuw1993](@cite) P. J. Rousseeuw and C. Croux. *Alternatives to the median absolute deviation*. Journal of the American Statistical Association 88, 1273–1283 (1993).",#
                                 :szekely2007 => "[szekely2007](@cite) G. J. Székely, M. L. Rizzo and N. K. Bakirov. *Measuring and testing dependence by correlation of distances*. The Annals of Statistics 35, 2769–2794 (2007).",#
                                 :pkurt => "[pkurt](@cite) D. Cajas. *Convex Optimization of Portfolio Kurtosis*. Available at SSRN 4202967 (2022).",#
                                 :pkurtapprox => "[pkurtapprox](@cite) D. Cajas. *Approximation of Portfolio Kurtosis through Sum of Squared Quadratic Forms*. Available at SSRN 4472793 (2023).",#
                                 :nskew => "[nskew](@cite) D. Cajas. *On the Spectral Decomposition of Portfolio Skewness and its Application to Portfolio Optimization*. Available at SSRN 4540021 (2023).",#
                                 :robustaa => "[robustaa](@cite) R. H. Tütüncü and M. Koenig. *Robust asset allocation*. Annals of Operations Research 132, 157–187 (2004).",#
                                 :sdpmom => "[sdpmom](@cite) D. Cajas. *Semidefinite Relaxation of Higher Portfolio Moments*. Available at SSRN 5284483 (2025).",#
                                 :emom => "[emom](@cite) D. Cajas. *Portfolio Optimization of Even Moments using Power Cone Programming*. Available at SSRN 6518258 (2026).",#
                                 :mad => "[mad](@cite) H. Konno and H. Yamazaki. *Mean-absolute deviation portfolio optimization model and its applications to Tokyo stock market*. Management Science 37, 519–531 (1991).",#
                                 :lpm => "[lpm](@cite) P. C. Fishburn. *Mean-risk analysis with risk associated with below-target returns*. The American Economic Review 67, 116–126 (1977).",#
                                 :palomar2025 => "[palomar2025](@cite) D. P. Palomar. *Portfolio Optimization: Theory and Application* (Cambridge University Press, 2025).",#
                                 :cajas2025 => "[cajas2025](@cite) D. Cajas. *Advanced Portfolio Optimization: A Cutting-edge Quantitative Approach* (Springer Nature Switzerland, 2025).",#
                                 :meucci2005 => "[meucci2005](@cite) A. Meucci. *Risk and Asset Allocation* (Springer Berlin Heidelberg, 2005).",#
                                 :demiguel2009 => "[demiguel2009](@cite) V. DeMiguel, L. Garlappi, F. J. Nogales and R. Uppal. *A Generalized Approach to Portfolio Optimization: Improving Performance by Constraining Portfolio Norms*. Management Science 55, 798–812 (2009).",#
                                 :fan2008 => "[fan2008](@cite) J. Fan, Y. Fan and J. Lv. *High dimensional covariance matrix estimation using a factor model*. Journal of Econometrics 147, 186–197 (2008).",#
                                 :martelliniziemann2010 => "[martelliniziemann2010](@cite) L. Martellini and V. Ziemann. *Improved estimates of higher-order comoments and implications for portfolio selection*. The Review of Financial Studies 23, 1467–1502 (2010).",#
                                 :boudt2015 => "[boudt2015](@cite) K. Boudt, W. Lu and B. Peeters. *Higher order comoments of multifactor models and asset allocation*. Finance Research Letters 13, 225–233 (2015).",#
                                 :jorion1986 => "[jorion1986](@cite) P. Jorion. *Bayes-Stein estimation for portfolio analysis*. The Journal of Financial and Quantitative Analysis 21, 279–292 (1986).",#
                                 :bodnar2019 => "[bodnar2019](@cite) T. Bodnar, O. Okhrin and N. Parolya. *Optimal shrinkage estimator for high-dimensional mean vector*. Journal of Multivariate Analysis 170, 63–79 (2019).",#
                                 :black1992 => "[black1992](@cite) F. Black and R. Litterman. *Global portfolio optimization*. Financial Analysts Journal 48, 28–43 (1992).",#
                                 :shannon1948 => "[shannon1948](@cite) C. E. Shannon. *A mathematical theory of communication*. The Bell System Technical Journal 27, 379–423 (1948).",#
                                 :sibuya1960 => "[sibuya1960](@cite) M. Sibuya. *Bivariate extreme statistics, I*. Annals of the Institute of Statistical Mathematics 11, 195–210 (1960).",#
                                 :luca2011 => "[luca2011](@cite) G. De Luca and P. Zuccolotto. *A tail dependence-based dissimilarity measure for financial time series clustering*. Advances in Data Analysis and Classification 5, 323–340 (2011).",#
                                 :hacinegharbi2012 => "[hacinegharbi2012](@cite) A. Hacine-Gharbi, P. Ravier, R. Harba and T. Mohamadi. *Low bias histogram-based estimation of mutual information for feature selection*. Pattern Recognition Letters 33, 1302–1308 (2012).",#
                                 :hacinegharbi2018 => "[hacinegharbi2018](@cite) A. Hacine-Gharbi and P. Ravier. *A binning formula of bi-histogram for joint entropy estimation using mean square error minimization*. Pattern Recognition Letters 101, 21–28 (2018).",#
                                 :vandongen2012 => "[vandongen2012](@cite) S. Van Dongen and A. J. Enright. *Metric distances derived from cosine similarity and Pearson and Spearman correlations*. arXiv preprint arXiv:1208.3145 (2012).",#
                                 :rousseeuw1987 => "[rousseeuw1987](@cite) P. J. Rousseeuw. *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics 20, 53–65 (1987).",#
                                 :lopezdeprado2019 => "[lopezdeprado2019](@cite) M. López de Prado and M. J. Lewis. *Detection of false investment strategies using unsupervised learning methods*. Quantitative Finance 19, 1555–1565 (2019).",#
                                 :yue2008 => "[yue2008](@cite) S. Yue, X. Wang and M. Wei. *Application of two-order difference to gap statistic*. Transactions of Tianjin University 14, 217–221 (2008).",#
                                 :tibshirani2001 => "[tibshirani2001](@cite) R. Tibshirani, G. Walther and T. Hastie. *Estimating the number of clusters in a data set via the gap statistic*. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 63, 411–423 (2001).",#
                                 :mullner2011 => "[mullner2011](@cite) D. Müllner. *Modern hierarchical, agglomerative clustering algorithms*. arXiv preprint arXiv:1109.2378 (2011).",#
                                 :virtanen2020 => "[virtanen2020](@cite) P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, İ. Polat, Y. Feng, E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa and P. van Mulbregt. *SciPy 1.0: fundamental algorithms for scientific computing in Python*. Nature Methods 17, 261–272 (2020).",#
                                 :lloyd1982 => "[lloyd1982](@cite) S. P. Lloyd. *Least squares quantization in PCM*. IEEE Transactions on Information Theory 28, 129–137 (1982).",#
                                 :freeman1977 => "[freeman1977](@cite) L. C. Freeman. *A set of measures of centrality based on betweenness*. Sociometry 40, 35–41 (1977).",#
                                 :freeman1979 => "[freeman1979](@cite) L. C. Freeman. *Centrality in social networks conceptual clarification*. Social Networks 1, 215–239 (1979).",#
                                 :brandes2001 => "[brandes2001](@cite) U. Brandes. *A faster algorithm for betweenness centrality*. The Journal of Mathematical Sociology 25, 163–177 (2001).",#
                                 :bonacich1987 => "[bonacich1987](@cite) P. Bonacich. *Power and centrality: a family of measures*. American Journal of Sociology 92, 1170–1182 (1987).",#
                                 :katz1953 => "[katz1953](@cite) L. Katz. *A new status index derived from sociometric analysis*. Psychometrika 18, 39–43 (1953).",#
                                 :brin1998 => "[brin1998](@cite) S. Brin and L. Page. *The anatomy of a large-scale hypertextual Web search engine*. Computer Networks and ISDN Systems 30, 107–117 (1998).",#
                                 :valente1998 => "[valente1998](@cite) T. W. Valente and R. K. Foreman. *Integration and radiality: measuring the extent of an individual's connectedness and reachability in a network*. Social Networks 20, 89–105 (1998).",#
                                 :shimbel1953 => "[shimbel1953](@cite) A. Shimbel. *Structural parameters of communication networks*. The Bulletin of Mathematical Biophysics 15, 501–507 (1953).",#
                                 :estrada2011 => "[estrada2011](@cite) E. Estrada. *The Structure of Complex Networks: Theory and Applications* (Oxford University Press, 2011).",#
                                 :mantegna1999 => "[mantegna1999](@cite) R. N. Mantegna. *Hierarchical structure in financial markets*. The European Physical Journal B 11, 193–197 (1999).",#
                                 :kruskal1956 => "[kruskal1956](@cite) J. B. Kruskal. *On the shortest spanning subtree of a graph and the traveling salesman problem*. Proceedings of the American Mathematical Society 7, 48–50 (1956).",#
                                 :boruvka1926 => "[boruvka1926](@cite) O. Borůvka. *O jistém problému minimálním*. Práce Moravské Přírodovědecké Společnosti 3, 37–58 (1926).",#
                                 :prim1957 => "[prim1957](@cite) R. C. Prim. *Shortest connection networks and some generalizations*. The Bell System Technical Journal 36, 1389–1401 (1957).",#
                                 :tumminello2005 => "[tumminello2005](@cite) M. Tumminello, T. Aste, T. Di Matteo and R. N. Mantegna. *A tool for filtering information in complex systems*. Proceedings of the National Academy of Sciences 102, 10421–10426 (2005).",#
                                 :graphpo1 => "[graphpo1](@cite) D. Cajas. *A Graph Theory Approach to Portfolio Optimization*. Available at SSRN 4602019 (2023).",#
                                 :graphpo2 => "[graphpo2](@cite) D. Cajas. *A Graph Theory Approach to Portfolio Optimization Part II*. Available at SSRN 4667426 (2023).",#
                                 :riccascozzari2024 => "[riccascozzari2024](@cite) F. Ricca and A. Scozzari. *Portfolio optimization through a network approach: network assortative mixing and portfolio diversification*. European Journal of Operational Research 312, 700–717 (2024).",#
                                 :efroymson1960 => "[efroymson1960](@cite) M. A. Efroymson. *Multiple regression analysis*. In: *Mathematical Methods for Digital Computers*, edited by A. Ralston and H. S. Wilf (John Wiley & Sons, 1960); pp. 191–203.",#
                                 :hocking1976 => "[hocking1976](@cite) R. R. Hocking. *The analysis and selection of variables in linear regression*. Biometrics 32, 1–49 (1976).",#
                                 :akaike1974 => "[akaike1974](@cite) H. Akaike. *A new look at the statistical model identification*. IEEE Transactions on Automatic Control 19, 716–723 (1974).",#
                                 :hurvich1989 => "[hurvich1989](@cite) C. M. Hurvich and C.-L. Tsai. *Regression and time series model selection in small samples*. Biometrika 76, 297–307 (1989).",#
                                 :schwarz1978 => "[schwarz1978](@cite) G. Schwarz. *Estimating the dimension of a model*. The Annals of Statistics 6, 461–464 (1978).",#
                                 :theil1961 => "[theil1961](@cite) H. Theil. *Economic Forecasts and Policy*. 2 Edition (North-Holland, 1961).",#
                                 :nelder1972 => "[nelder1972](@cite) J. A. Nelder and R. W. Wedderburn. *Generalized linear models*. Journal of the Royal Statistical Society: Series A (General) 135, 370–384 (1972).",#
                                 :pearson1901 => "[pearson1901](@cite) K. Pearson. *On lines and planes of closest fit to systems of points in space*. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2, 559–572 (1901).",#
                                 :hotelling1933 => "[hotelling1933](@cite) H. Hotelling. *Analysis of a complex of statistical variables into principal components*. Journal of Educational Psychology 24, 417–441 (1933).",#
                                 :tipping1999 => "[tipping1999](@cite) M. E. Tipping and C. M. Bishop. *Probabilistic principal component analysis*. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 61, 611–622 (1999).",#
                                 :fekedulegn2002 => "[fekedulegn2002](@cite) B. D. Fekedulegn, J. J. Colbert, R. R. Hicks, Jr. and M. E. Schuckers. *Coping with multicollinearity: an example on application of principal components regression in dendroecology*. Research Paper NE-RP-721 (U.S. Department of Agriculture, Forest Service, Northeastern Research Station, 2002).",#
                                 :christensenprabhala1998 => "[christensenprabhala1998](@cite) B. J. Christensen and N. R. Prabhala. *The relation between implied and realized volatility*. Journal of Financial Economics 50, 125–150 (1998).",#
                                 :christensenhansen2002 => "[christensenhansen2002](@cite) B. J. Christensen and C. S. Hansen. *New evidence on the implied-realized volatility relation*. The European Journal of Finance 8, 187–205 (2002).",#
                                 :andersen2006 => "[andersen2006](@cite) T. G. Andersen, T. Bollerslev, P. F. Christoffersen and F. X. Diebold. *Volatility and correlation forecasting*. In: *Handbook of Economic Forecasting*, Vol. 1, edited by G. Elliott, C. W. Granger and A. Timmermann (North-Holland, 2006); Chapter 15, pp. 777–878.",#
                                 :egbersswinkels2015 => "[egbersswinkels2015](@cite) T. Egbers and L. Swinkels. *Can implied volatility predict returns on the currency carry trade?*. Journal of Banking & Finance 59, 14–26 (2015).",#
                                 :cheung2007 => "[cheung2007](@cite) W. Cheung. *The augmented Black-Litterman model: a ranking-free approach to factor-based portfolio construction and beyond*. Quantitative Finance 13, 301–316 (2013).",#
                                 :kolmritter2016 => "[kolmritter2016](@cite) P. N. Kolm and G. Ritter. *On the Bayesian interpretation of Black-Litterman*. European Journal of Operational Research 258, 564–572 (2017).",#
                                 :dietrichlist2017 => "[dietrichlist2017](@cite) F. Dietrich and C. List. *Probabilistic opinion pooling generalized. Part one: general agendas*. Social Choice and Welfare 48, 747–786 (2017).",#
                                 :martinisprenger2017 => "[martinisprenger2017](@cite) C. Martini and J. Sprenger. *Opinion Aggregation and Individual Expertise*. In: *Scientific Collaboration and Collective Knowledge* (Oxford University Press, 2017).",#
                                 :good1952 => "[good1952](@cite) I. J. Good. *Rational decisions*. Journal of the Royal Statistical Society: Series B (Methodological) 14, 107–114 (1952).",#
                                 :idzorek2007 => "[idzorek2007](@cite) T. Idzorek. *A step-by-step guide to the Black-Litterman model: incorporating user-specified confidence levels*. In: *Forecasting Expected Returns in the Financial Markets* (Academic Press, 2007); pp. 17–38.",#
                                 :walters2011 => "[walters2011](@cite) J. Walters. *The Black-Litterman model in detail*. SSRN Electronic Journal (2011).",#
                                 :boydvandenberghe2004 => "[boydvandenberghe2004](@cite) S. Boyd and L. Vandenberghe. *Convex Optimization* (Cambridge University Press, Cambridge, UK, 2004).",#
                                 :diamondboyd2016 => "[diamondboyd2016](@cite) S. Diamond and S. Boyd. *CVXPY: A Python-embedded modeling language for convex optimization*. Journal of Machine Learning Research 17, 1–5 (2016).",#
                                 :lopezdeprado2016 => "[lopezdeprado2016](@cite) M. López de Prado. *Building diversified portfolios that outperform out of sample*. The Journal of Portfolio Management 42, 59–69 (2016).",#
                                 :raffinot2017 => "[raffinot2017](@cite) T. Raffinot. *Hierarchical clustering-based asset allocation*. The Journal of Portfolio Management 44, 89–99 (2017).",#
                                 :raffinot2018 => "[raffinot2018](@cite) T. Raffinot. *The hierarchical equal risk contribution portfolio*. SSRN Electronic Journal (2018).",#
                                 :cotton2024 => "[cotton2024](@cite) P. Cotton. *Schur Complementary Allocation: A Unification of Hierarchical Risk Parity and Minimum Variance Portfolios*. arXiv preprint arXiv:2411.05807 (2024).",#
                                 :kelly1956 => "[kelly1956](@cite) J. L. Kelly. *A new interpretation of information rate*. Bell System Technical Journal 35, 917–926 (1956).",#
                                 :thorp2008 => "[thorp2008](@cite) E. O. Thorp. *The Kelly criterion in blackjack, sports betting, and the stock market*. In: *Handbook of Asset and Liability Management*, Vol. 1 (North-Holland, 2008); pp. 385–428.",#
                                 :chares2009 => "[chares2009](@cite) R. Chares. *Cones and interior-point algorithms for structured convex optimization involving powers and exponentials*. Ph.D. Thesis, Université catholique de Louvain, Louvain-la-Neuve, Belgium (2009).",#
                                 :sharpe1964 => "[sharpe1964](@cite) W. F. Sharpe. *Capital asset prices: a theory of market equilibrium under conditions of risk*. The Journal of Finance 19, 425–442 (1964).",#
                                 :schaibleibaraki1983 => "[schaibleibaraki1983](@cite) S. Schaible and T. Ibaraki. *Fractional programming*. European Journal of Operational Research 12, 325–338 (1983).",#
                                 :charnescooper1962 => "[charnescooper1962](@cite) A. Charnes and W. W. Cooper. *Programming with linear fractional functionals*. Naval Research Logistics Quarterly 9, 181–186 (1962).",#
                                 :grinoldkahn1999 => "[grinoldkahn1999](@cite) R. C. Grinold and R. N. Kahn. *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk*. 2 Edition (McGraw-Hill, New York, 1999).",#
                                 :toth2011 => "[toth2011](@cite) B. Tóth, Y. Lempérière, C. Deremble, J. de Lataillade, J. Kockelkoren and J.-P. Bouchaud. *Anomalous price impact and the critical nature of liquidity in financial markets*. Physical Review X 1, 021006 (2011).",#
                                 :lopezdeprado2019robust => "[lopezdeprado2019robust](@cite) M. López de Prado. *A robust estimator of the efficient frontier*. SSRN Electronic Journal (2019).",#
                                 :wolpert1992 => "[wolpert1992](@cite) D. H. Wolpert. *Stacked generalization*. Neural Networks 5, 241–259 (1992).",#
                                 :shen2017 => "[shen2017](@cite) W. Shen and J. Wang. *Portfolio selection via subset resampling*. In: *Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence* (2017); pp. 1517–1523.",#
                                 :martin2021 => "[martin2021](@cite) R. A. Martin. *PyPortfolioOpt: portfolio optimization in Python*. Journal of Open Source Software 6, 3066 (2021).",#
                                 :maillard2008 => "[maillard2008](@cite) S. Maillard, T. Roncalli and J. Teiletche. *On the properties of equally-weighted risk contributions portfolios*. Available at SSRN 1271972 (2008).",#
                                 :bruderroncalli2012 => "[bruderroncalli2012](@cite) B. Bruder and T. Roncalli. *Managing risk exposures using the risk budgeting approach*. Available at SSRN 2009778 (2012).",#
                                 :mausserromanko2014 => "[mausserromanko2014](@cite) H. Mausser and O. Romanko. *Computing equal risk contribution portfolios*. IBM Journal of Research and Development 58, 5:1–5:12 (2014).",#
                                 :roncalliweisang2012 => "[roncalliweisang2012](@cite) T. Roncalli and G. Weisang. *Risk parity portfolios with risk factors*. Available at SSRN 2155159 (2012).",#
                                 :meucci2007 => "[meucci2007](@cite) A. Meucci. *Risk contributions from generic user-defined factors*. Available at SSRN 930034 (2007).",#
                                 :mosek2023c => "[mosek2023c](@cite) MOSEK ApS. *MOSEK Portfolio Optimization Cookbook* (2023).",#
                                 :cajas2019noc => "[cajas2019noc](@cite) D. Cajas. *Robust portfolio selection with near optimal centering*. Available at SSRN 3572435 (2019).",#
                                 :degraaf2016 => "[degraaf2016](@cite) T. de Graaf. *Robust Mean-Variance Optimization*. Master's Thesis, Leiden University (2016).",#
                                 :gambetakwon2020 => "[gambetakwon2020](@cite) V. Gambeta and R. Kwon. *Risk return trade-off in relaxed risk parity portfolio optimization*. Journal of Risk and Financial Management 13, 237 (2020).",#
                                 :richardroncalli2019 => "[richardroncalli2019](@cite) J.-C. Richard and T. Roncalli. *Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles*. arXiv preprint arXiv:1902.05710 (2019).")

"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all estimator types.

All custom estimators should subtype `AbstractEstimator`.

Estimators consume data to estimate parameters or models. Some estimators may utilise different algorithms. These can range from simple implementation details that don't change the result much but may have different numerical characteristics, to entirely different methodologies or algorithms yielding different results.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all algorithm types.

All algorithms should subtype `AbstractAlgorithm`.

Algorithms are often used by estimators to perform specific tasks. These can be in the form of simple implementation details to entirely different procedures for estimating a quantity.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all result types.

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
                   throw(DomainError(half_life, \"half_life must be an integer greater than zero\"))
               end
               return new{typeof(half_life)}(half_life)
           end
       end

julia> function MyWeights(; half_life::Integer = 5)
           return MyWeights(half_life)
       end
MyWeights

julia> function PortfolioOptimisers.get_observation_weights(w::MyWeights,
                                                            X::PortfolioOptimisers.VecNum;
                                                            kwargs...)
           lambda = 2^(-inv(w.half_life))
           return eweights(1:length(X), lambda; scale = true)
       end

julia> function PortfolioOptimisers.get_observation_weights(w::MyWeights,
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

Both methods must be dispatched on the concrete subtype, as above — never on `DynamicAbstractWeights` itself, which would capture every other subtype too.

Implementing only one of the two arities is the mistake to avoid. Rather than silently computing an unweighted result, the unimplemented shape raises [`ObservationWeightsError`](@ref) and names the methods to write:

```jldoctest
julia> struct PartialWeights <: PortfolioOptimisers.DynamicAbstractWeights end

julia> function PortfolioOptimisers.get_observation_weights(w::PartialWeights,
                                                            X::PortfolioOptimisers.VecNum;
                                                            kwargs...)
           return eweights(1:length(X), 0.5; scale = true)
       end

julia> PortfolioOptimisers.get_observation_weights(PartialWeights(), 1:3)
3-element Weights{Float64, Float64, Vector{Float64}}:
 0.25
 0.5
 1.0

julia> PortfolioOptimisers.get_observation_weights(PartialWeights(), ones(3, 10))
ERROR: ObservationWeightsError: PartialWeights is a DynamicAbstractWeights with no `get_observation_weights` method for a 2-dimensional input of size (3, 10). Implement `get_observation_weights(w::PartialWeights, X::VecNum; kwargs...)` and/or `get_observation_weights(w::PartialWeights, X::MatNum; dims::Int = 1, kwargs...)`, or pass a `StatsBase.AbstractWeights` instead (or `nothing` to compute unweighted). See the `DynamicAbstractWeights` docstring for a worked example.
Stacktrace:
[...]
```

# Related

  - [`ObsWeights`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`ObservationWeightsError`](@ref)
  - [`get_observation_weights`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
"""
abstract type DynamicAbstractWeights <: AbstractEstimator end
"""
    define_pretty_show(T, flag::Bool = true)

Macro to define a custom pretty-printing `Base.show` method for types.

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
  - `min_score`: the minimum normalised similarity in `[0, 1]` a candidate must reach before it is suggested (default `0.7`). Raising it toward `1` keeps only near-exact matches; setting it above `1` disables suggestions entirely — useful in meta-optimiser inner loops, where an asset name legitimately absent from a cluster/subset is not a typo and should draw no suggestion. Must be positive (enforced by the constructor). `StringDistances.findnearest` never suggests a candidate scoring exactly `0`, but any threshold at or below `0` admits every candidate with *some* nonzero similarity — so `0` and a negative value behave identically and both defeat the info-leak-safe boundary by naming a real asset for a near-miss probe.

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
    function StringDistanceConfig(dist::StringDistances.StringDistance, min_score::Real)
        @argcheck(min_score > 0,
                  ArgumentError("min_score must be positive; got $(min_score). A value above 1 legitimately disables suggestions, but a zero or negative threshold admits every candidate with any nonzero similarity, making `did_you_mean` echo a real asset name for near-miss probes and defeating the info-leak-safe boundary (ADR 0026)."))
        return new(dist, Float64(min_score))
    end
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
  - `min_score`: minimum normalised similarity in `(0, 1]` to emit a suggestion; set above `1` to disable suggestions. Must be positive: a non-positive threshold admits every candidate with any nonzero similarity.

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
Global resource caps for the sampling- and sweep-based estimators, guarding the config→allocation trust boundary against memory and compute exhaustion.

Draw counts, subset counts, frontier-sweep lengths and histogram bin counts are untrusted configuration integers (config files, tuning grids, UI): each directly multiplies an allocation, and in the subset and frontier cases a whole optimisation. Their own constructors only bound them from *below* (`n_sim > 0`, `n_subsets >= 2`, `N > 0`, `bins > 0`), so an absurd value — a stray extra digit, a mis-scaled sweep — is accepted and the process is killed by the OOM killer rather than told what went wrong. These caps fail closed with a typed `DomainError` at the point the value is resolved.

There is **one cap per sink**, each named to mirror the field it guards. Reuse across sinks is deliberately avoided: a *linear* cap cannot bound a *quadratic* sink, which is why the `bins × bins` histogram gets its own [`max_bins`](@ref ResourceLimits) rather than sharing the linear draw cap.

# Fields

  - `max_n_sim`: maximum Monte-Carlo/bootstrap draws (`n_sim`) accepted by [`NormalUncertaintySet`](@ref) and [`ARCHUncertaintySet`](@ref) (default `1_000_000`). Each draw stores an `N × N` covariance, so the backing array is `N² · n_sim` elements: at 20 assets the default cap already permits a 3.2 GB request, while the shipped `n_sim` is `3_000`. *Memory*-bound.
  - `max_n_subsets`: maximum resampled asset subsets (`n_subsets`) accepted by [`SubsetResampling`](@ref) and [`MultipleRandomised`](@ref) (default `100_000`). This one bounds *compute* far more than memory — every subset runs a full inner optimisation — so the cap sits far above any realistic sweep (the shipped default is `2`) yet well below a value that would wedge a session for days.
  - `max_frontier`: maximum efficient-frontier sweep points accepted by the [`Frontier`](@ref) algorithm of [`MeanRisk`](@ref) and [`NearOptimalCentering`](@ref) (default `100_000`). Like `max_n_subsets` this is *compute*-bound — every point runs a full inner `optimise_JuMP_model!` solve — so it mirrors that ceiling; the shipped `Frontier` default is `N = 20`. Enforced **twice**: [`Frontier`](@ref)'s constructor caps the `N` of one bound, and [`assert_frontier_sweep_cap`](@ref) caps the **product** across every swept return term and every swept risk measure at Model Assembly, since the sweep is an `Iterators.product` and `k` bounds of `N` points cost `N^k` solves.
  - `max_bins`: maximum histogram bins accepted by [`MutualInfoCovariance`](@ref) and [`VariationInfoDistance`](@ref) (default `10_000`). The joint histogram is a `bins × bins` weights matrix built per asset pair, so this bounds a *quadratic* memory allocation: `10_000²` cells is ≈ 800 MB per histogram — below OOM yet far above the ~50-bin range legitimate binning produces.
  - `max_hop_count`: maximum hop count (`n`) accepted by [`HopCount`](@ref) (default `100_000`). Three readers sum `A^i` over `i in 0:n`, so the sink is *linear* in `n` at `N³` flops a power and a large `n` wedges the session on compute rather than memory. Like `max_n_subsets` it is compute-bound and mirrors that ceiling; the shipped default is `n = 1`. The cap is read in `HopCount`'s constructor, which is also where [`resolve_separation`](@ref) sends a rule's answer, so one check covers the stated value and the computed one alike.
  - `max_search_grid`: maximum search-grid candidates accepted by [`GridSearchCrossValidation`](@ref) and [`RandomisedSearchCrossValidation`](@ref) (default `100_000`). Every candidate runs a full cross-validated fit, so this is *compute*-bound like `max_n_subsets`. The grid is an `Iterators.product` materialised by `collect`, so `k` parameters of `N` values cost `N^k` candidates: the cap is asserted on the **product** by [`assert_search_grid_cap`](@ref) where the grid is formed, since a per-parameter check can never see it.

The values are conservative static defaults, deliberately far above legitimate use: they exist to convert an OOM kill into a typed error, not to second-guess a sizing choice. Immutable; held in the [`RESOURCE_LIMITS`](@ref) [`ScopedConfig`](@ref). Set the global default via [`set_resource_limits!`](@ref), override per scope via [`with_resource_limits`](@ref). All fields must be positive (enforced by the constructor). Prefer the keyword constructor `ResourceLimits(; …)` — the six caps are same-typed and four share the value `100_000`, so positional construction is error-prone.

# Related

  - [`RESOURCE_LIMITS`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`with_resource_limits`](@ref)
  - [`assert_resource_cap`](@ref)
  - [`EquationLimits`](@ref)
"""
struct ResourceLimits
    max_n_sim::Int
    max_n_subsets::Int
    max_frontier::Int
    max_bins::Int
    max_hop_count::Int
    max_search_grid::Int
    function ResourceLimits(max_n_sim::Integer, max_n_subsets::Integer,
                            max_frontier::Integer, max_bins::Integer,
                            max_hop_count::Integer, max_search_grid::Integer)
        @argcheck(max_n_sim > 0 &&
                  max_n_subsets > 0 &&
                  max_frontier > 0 &&
                  max_bins > 0 &&
                  max_hop_count > 0 &&
                  max_search_grid > 0,
                  ArgumentError("max_n_sim, max_n_subsets, max_frontier, max_bins, max_hop_count and max_search_grid must be positive."))
        return new(Int(max_n_sim), Int(max_n_subsets), Int(max_frontier), Int(max_bins),
                   Int(max_hop_count), Int(max_search_grid))
    end
end
function ResourceLimits(; max_n_sim::Integer = 1_000_000, max_n_subsets::Integer = 100_000,
                        max_frontier::Integer = 100_000, max_bins::Integer = 10_000,
                        max_hop_count::Integer = 100_000,
                        max_search_grid::Integer = 100_000)
    return ResourceLimits(max_n_sim, max_n_subsets, max_frontier, max_bins, max_hop_count,
                          max_search_grid)
end
"""
    RESOURCE_LIMITS = ScopedConfig(ResourceLimits())

Default global resource caps for the sampling- and sweep-based estimators, guarding the config→allocation trust boundary against memory and compute exhaustion. Read as `RESOURCE_LIMITS[]`; the defaults may be seeded per project at load time via the `"max_n_sim"` / `"max_n_subsets"` / `"max_frontier"` / `"max_bins"` / `"max_hop_count"` / `"max_search_grid"` preferences (see [`apply_preferences!`](@ref)).

# Related

  - [`ResourceLimits`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`with_resource_limits`](@ref)
  - [`assert_resource_cap`](@ref)
"""
const RESOURCE_LIMITS = ScopedConfig(ResourceLimits())
"""
    set_resource_limits!(; max_n_sim::Integer, max_n_subsets::Integer,
                         max_frontier::Integer, max_bins::Integer,
                         max_hop_count::Integer, max_search_grid::Integer)

Configure the global default resource caps read at the config→allocation trust boundary (see [`RESOURCE_LIMITS`](@ref)).

  - `max_n_sim`: maximum `n_sim` accepted by the uncertainty-set estimators.
  - `max_n_subsets`: maximum `n_subsets` accepted by the subset-resampling estimators.
  - `max_frontier`: maximum `N` accepted by one [`Frontier`](@ref), and maximum total sweep points across every swept bound.
  - `max_bins`: maximum `bins` accepted by the mutual-information estimators.
  - `max_hop_count`: maximum `n` accepted by [`HopCount`](@ref), stated or resolved from a rule.
  - `max_search_grid`: maximum total candidates in a search cross-validation grid.

Raise them for a genuinely large machine-authored run on a machine sized for it, or lower them to tighten the boundary. All must be positive; unspecified keywords keep their current default. The store is atomic (see [`ScopedConfig`](@ref)); for a temporary, task-scoped override use [`with_resource_limits`](@ref).

Returns the new default [`ResourceLimits`](@ref).

# Related

  - [`RESOURCE_LIMITS`](@ref)
  - [`with_resource_limits`](@ref)
  - [`set_equation_limits!`](@ref)
"""
function set_resource_limits!(;
                              max_n_sim::Integer = (@atomic RESOURCE_LIMITS.default).max_n_sim,
                              max_n_subsets::Integer = (@atomic RESOURCE_LIMITS.default).max_n_subsets,
                              max_frontier::Integer = (@atomic RESOURCE_LIMITS.default).max_frontier,
                              max_bins::Integer = (@atomic RESOURCE_LIMITS.default).max_bins,
                              max_hop_count::Integer = (@atomic RESOURCE_LIMITS.default).max_hop_count,
                              max_search_grid::Integer = (@atomic RESOURCE_LIMITS.default).max_search_grid)
    return set_default!(RESOURCE_LIMITS,
                        ResourceLimits(; max_n_sim, max_n_subsets, max_frontier, max_bins,
                                       max_hop_count, max_search_grid))
end
"""
    with_resource_limits(f; max_n_sim::Integer = RESOURCE_LIMITS[].max_n_sim,
                         max_n_subsets::Integer = RESOURCE_LIMITS[].max_n_subsets,
                         max_frontier::Integer = RESOURCE_LIMITS[].max_frontier,
                         max_bins::Integer = RESOURCE_LIMITS[].max_bins,
                         max_hop_count::Integer = RESOURCE_LIMITS[].max_hop_count,
                         max_search_grid::Integer = RESOURCE_LIMITS[].max_search_grid)

Run `f()` with the resource caps (see [`RESOURCE_LIMITS`](@ref)) overridden for the dynamic extent of the call, restoring the previous caps on exit. Task-scoped and thread-safe (see [`ScopedConfig`](@ref)); the global default is untouched. Unspecified keywords inherit from the currently active value, so nested overrides compose.

Useful to raise the ceiling for one deliberately large run without loosening the boundary for other concurrent work. Note the cap is read where the value is *resolved*: `n_sim`, `N` and `bins` at estimator construction, `n_subsets` when the optimisation resolves its (possibly [`TimeDependent`](@ref)) schedule — so wrap the constructor call in the former cases and the `optimise` call in the latter.

# Related

  - [`set_resource_limits!`](@ref)
  - [`RESOURCE_LIMITS`](@ref)
"""
function with_resource_limits(f; max_n_sim::Integer = RESOURCE_LIMITS[].max_n_sim,
                              max_n_subsets::Integer = RESOURCE_LIMITS[].max_n_subsets,
                              max_frontier::Integer = RESOURCE_LIMITS[].max_frontier,
                              max_bins::Integer = RESOURCE_LIMITS[].max_bins,
                              max_hop_count::Integer = RESOURCE_LIMITS[].max_hop_count,
                              max_search_grid::Integer = RESOURCE_LIMITS[].max_search_grid)
    return with_config(f, RESOURCE_LIMITS,
                       ResourceLimits(; max_n_sim, max_n_subsets, max_frontier, max_bins,
                                      max_hop_count, max_search_grid))
end
"""
    did_you_mean(name::AbstractString, candidates) -> String

Return a "did you mean" suffix naming the closest match to `name` among `candidates`, or an empty string when no candidate reaches the global [`STRING_DISTANCE`](@ref) `min_score` threshold (or `candidates` is empty). The suffix reads `" (did you mean X?)"`, with the match in place of X.

Do not wrap the suffix in a code span that also carries escaped backticks. `JuliaFormatter` mis-pairs the backticks and deletes the spaces around the neighbouring code spans, which breaks the rendering.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Suggest the nearest `candidates` entry to a mistyped **declaration key**: a macro block key, a dictionary key, a struct field name, or a keyword of a generated constructor.

Wraps [`did_you_mean`](@ref) in a looser scoped configuration than the global default: Damerau-Levenshtein (so a transposed pair costs one edit, not two) at `min_score = 0.5`. The strict global default exists to keep near-miss probes from echoing real *asset names* back to the caller (ADR 0026); that boundary does not apply here, because the candidates are compile-time constants — block keys, dictionary keys and field names — with nothing to leak. At the default `0.7` under plain Levenshtein, short keys never match: `nuon` scores 0.5 against `noun`, so the suggestion would be dead code.

# Related

  - [`did_you_mean`](@ref)
  - [`with_string_distance`](@ref)
  - [`windowed_estimator_suggest`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
function suggest_declared_key(key, candidates)
    return with_string_distance(; dist = StringDistances.DamerauLevenshtein(),
                                min_score = 0.5) do
        return did_you_mean(string(key), string.(collect(candidates)))
    end
end
"""
    unknown_variable_msg(v, nx, key; candidates = nx, axis = "asset") -> String

Build the warning/error text for a constraint or view variable `v` that is absent from the universe `nx` (stored under `key`). Names the variable and the universe *size* only — never the full universe — and appends a [`did_you_mean`](@ref) suggestion when a close match exists.

`candidates` is the pool searched for the typo suggestion (default: the universe `nx`). Callers whose valid namespace is broader than the raw universe — e.g. [`name_to_val!`](@ref), where a key may name a *group* rather than an asset — pass a wider pool (asset names plus group/set keys) so the suggestion can name a mistyped group. The reported universe *size* is always `length(nx)` regardless of `candidates`.

`axis` names the universe the variable was looked up in. It defaults to `"asset"` because that is the axis every constraint resolved against before [`ExposureConstraintEstimator`](@ref); a re-based constraint resolves its names against the *factor* universe and passes `"factor"`, so the message names the axis the user actually wrote in.

Shared by [`get_linear_constraints`](@ref), Black-Litterman view generation, entropy-pooling view generation, and [`name_to_val!`](@ref) so the message (and its info-leak-safe shape) lives in exactly one place.

# Related

  - [`did_you_mean`](@ref)
  - [`empty_row_msg`](@ref)
  - [`empty_projected_row_msg`](@ref)
"""
function unknown_variable_msg(v, nx, key; candidates = nx, axis::AbstractString = "asset")
    return "variable `$(v)` not in $(axis) universe ($(length(nx)) $(axis)s under key `$(key)`); term dropped" *
           did_you_mean(string(v), candidates)
end
"""
    empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint",
                  axis::AbstractString = "asset") -> String

Build the warning/error text for a parsed equation `eqn` whose every term missed the universe `nx` (stored under `key`), leaving an all-zero row that is dropped. Names the equation and the universe *size* only — never the full universe or the parsed struct. `noun` is `"constraint"` for linear constraints or `"view"` for Black-Litterman views; `axis` names the universe, as in [`unknown_variable_msg`](@ref).

Shared by [`get_linear_constraints`](@ref) and Black-Litterman view generation.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`empty_projected_row_msg`](@ref)
"""
function empty_row_msg(eqn, nx, key; noun::AbstractString = "constraint",
                       axis::AbstractString = "asset")
    return "$(noun) `$(eqn)` matched no $(axis)s in the universe ($(length(nx)) $(axis)s under key `$(key)`); row dropped"
end
"""
    empty_projected_row_msg(eqn, nf, key, n; noun::AbstractString = "constraint") -> String

Build the warning/error text for a re-based equation `eqn` whose terms *did* resolve against the factor universe `nf` (stored under `key`), but whose projection through the loadings is an all-zero row over `n` assets.

This diagnosis exists only under a re-basis, and it is a different failure from [`empty_row_msg`](@ref): there the names missed the universe, here they hit it and the basis annihilated them. Reporting the first for the second would send a user hunting for a typo that is not there — the real cause is a factor no asset loads on.

# Related

  - [`empty_row_msg`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
function empty_projected_row_msg(eqn, nf, key, n; noun::AbstractString = "constraint")
    return "$(noun) `$(eqn)` resolved against the factor universe ($(length(nf)) factors under key `$(key)`) but projected to an all-zero row over $(n) assets: every matched factor has zero loadings; row dropped"
end
"""
    gross_budget_bounds_msg(lb, ub) -> String

Build the error text for a gross budget (`gbgt`) whose weight bounds `lb` and `ub` admit no short position. With no negative bound the gross exposure equals the net exposure, so the net budget (`bgt`) already owns the constraint and `gbgt` has nothing left to express.

Names the *size* of the bounds and the failed predicate only — never the bound values — the same info-leak-safe discipline as [`unknown_variable_msg`](@ref) and its siblings. Scalar or absent bounds have no size, so the message names the bounds without a count.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`assert_gross_budget_admissible`](@ref)
  - [`w_neg_flag`](@ref)
"""
function gross_budget_bounds_msg(lb, ub)
    n = max(isa(lb, AbstractVector) ? length(lb) : 0,
            isa(ub, AbstractVector) ? length(ub) : 0)
    scope = iszero(n) ? "weight bounds" : "weight bounds over $(n) assets"
    return "gross budget (gbgt) requires weight bounds that admit short positions: with non-negative bounds no short weights exist, so the gross exposure equals the net exposure and the net budget (bgt) already constrains it. Got $(scope) with no negative element in lb or ub."
end
"""
    strict_diagnostic(msg::AbstractString, strict::Bool) -> Nothing

Report an unresolvable **name**: throw an `ArgumentError` under `strict`, warn otherwise, and in both cases the offending term is dropped.

`strict` governs names only. Nothing structural is refused, and a malformed *entry* throws unconditionally, because there is no reading of it to fall back to. Every name diagnostic in the library routes through here, so the strictness policy is one edit.

# Arguments

  - `msg`: The diagnostic text, built by [`unknown_variable_msg`](@ref) or one of its siblings.
  - `strict`: If `true`, throws an `ArgumentError`; if `false`, issues a warning.

# Returns

  - `nothing`.

# Related

  - [`unknown_variable_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
  - [`empty_row_msg`](@ref)
"""
function strict_diagnostic(msg::AbstractString, strict::Bool)::Nothing
    if strict
        throw(ArgumentError(msg))
    end
    @warn(msg)
    return nothing
end
"""
    missing_group_assets_msg(group, missing_assets, nx, key) -> String

Build the warning/error text for a `group` that resolves in the asset sets but whose members
`missing_assets` are absent from the asset universe `nx` (stored under `key`). Names the group, the
offending member names (which are caller input, not internal state), and the universe *size* only —
never the full universe or the input value dictionary — and appends a [`did_you_mean`](@ref)
suggestion for the first missing member.

Shared by [`name_to_val!`](@ref) so the info-leak-safe message shape lives in exactly one place,
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
    misaligned_axis_msg(declared, names, axis, key, sym) -> String

Build the error text for a universe declared under `key` that disagrees with the axis `sym` of the data it will be used against — `declared` against `names`.

Position is the only link between a name and a column, so a disagreement is not a naming inconvenience: every constraint row, bound and group would be attached to the wrong column and the optimisation would succeed with the wrong answer. The message therefore names what to fix, not just what is wrong.

Two disagreements are reported differently because they have different causes. Different lengths mean the two describe different universes — usually a stale `sets` against freshly sliced data. Equal lengths mean they describe the same universe in a different order, and the first differing position is the whole diagnosis. Names the sizes and the *first* differing pair only — never either universe in full, the same info-leak-safe discipline as [`unknown_variable_msg`](@ref).

# Related

  - [`unknown_variable_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
"""
function misaligned_axis_msg(declared, names, axis, key, sym)
    detail = if length(declared) != length(names)
        "$(length(declared)) $(axis)s are declared, but the data has $(length(names))"
    else
        i = findfirst(declared .!= names)
        "both have $(length(names)) $(axis)s but the order differs, first at position $(i): `$(declared[i])` vs `$(names[i])`"
    end
    return "the $(axis) universe declared under key `$(key)` does not describe the returns data: $(detail). Position is the only link between a name and a column, so this attaches constraints, bounds and groups to the wrong $(axis) rather than failing. Set `sets.dict[\"$(key)\"]` to `rd.$(sym)`, or slice the sets to match the data."
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
  - `"max_n_sim"` / `"max_n_subsets"` / `"max_frontier"` / `"max_bins"` / `"max_hop_count"` / `"max_search_grid"`: positive integers for [`RESOURCE_LIMITS`](@ref).
  - `"suggestion_min_score"`: real number for the [`STRING_DISTANCE`](@ref) threshold.
  - `"suggestion_distance"`: a [`PREFERENCE_DISTANCES`](@ref) name for the [`STRING_DISTANCE`](@ref) metric.
  - `"compact_show"`: boolean or integer for [`COMPACT_SHOW`](@ref).

Preferences.jl offers no way to enumerate the keys a project has set, so a misspelled *key* cannot be detected and is silently ignored (the shipped default applies) — misspelled or invalid *values* under these keys fail closed at load.

A valid value is applied, but a value that *widens* a guard is announced with a warning (see [`relaxed_preferences_msg`](@ref)): a preference file is data, it travels with a cloned project, and it applies before any user code runs.

# Related

  - [`apply_preferences!`](@ref)
  - [`relaxed_preferences_msg`](@ref)
"""
const PREFERENCE_KEYS = ("equation_max_length", "equation_max_depth", "max_n_sim",
                         "max_n_subsets", "max_frontier", "max_bins", "max_hop_count",
                         "max_search_grid", "suggestion_min_score", "suggestion_distance",
                         "compact_show")
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the warning text for the load-time preferences that widened a guard (see [`apply_preferences!`](@ref)). One line per key: the key, the default it replaced, and the value the project asked for.

A preference file is data. It ships with a cloned project or a template, it is often untracked, and [`__init__`](@ref PortfolioOptimisers.__init__) applies it at `using PortfolioOptimisers`, before any user code runs. A value that *tightens* a guard needs no announcement, so the warning names the widened guards alone: the [`RESOURCE_LIMITS`](@ref) and [`EQUATION_LIMITS`](@ref) caps a file raised, and a [`STRING_DISTANCE`](@ref) suggestion threshold it lowered (a lower threshold admits more candidates, which is the info-leak direction of `docs/adr/0026-lenient-constraint-names-with-suggestions.md`).

# Arguments

  - `relaxations`: One `(key, default, value)` triple per widened guard, in [`PREFERENCE_KEYS`](@ref) order.

# Returns

  - `msg::String`: Multi-line warning text, one line per triple.

Never interpolates the whole preference dictionary, so a key the message does not name stays out of the log — the same info-leak-safe message discipline as [`unknown_variable_msg`](@ref).

# Related

  - [`apply_preferences!`](@ref)
  - [`PREFERENCE_KEYS`](@ref)
  - [`unknown_variable_msg`](@ref)
"""
function relaxed_preferences_msg(relaxations::AbstractVector)
    msg = "$(length(relaxations)) load-time preference(s) of the active project widened a PortfolioOptimisers guard. Preferences apply at `using PortfolioOptimisers`, before any user code runs, so these values hold for the whole session:"
    for (key, default, val) in relaxations
        msg *= "\n  $(key): $(repr(default)) → $(repr(val))"
    end
    return msg *
           "\nThe values come from the `[PortfolioOptimisers]` section of the active project's `LocalPreferences.toml`. Delete a key there to restore the default, or widen the guard for one call only with `with_equation_limits`, `with_resource_limits` or `with_string_distance`."
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply load-time preference values to the global config defaults ([`EQUATION_LIMITS`](@ref), [`RESOURCE_LIMITS`](@ref), [`STRING_DISTANCE`](@ref), [`COMPACT_SHOW`](@ref)). Called by the package `__init__` with the [`PREFERENCE_KEYS`](@ref) values read via `Preferences.load_preference`; `nothing` values (unset preferences) are skipped and keep the shipped default.

Fails closed on an *invalid* value: it throws a typed `ArgumentError` naming the key and value, so the package refuses to load rather than silently running with a value the project got wrong. Values are applied through the `set_*!` setters, so they receive the same validation as runtime calls.

A *valid* value is applied whatever its size — the caps exist to turn an OOM kill into a typed error, not to second-guess a sizing choice, and a project on a large machine may legitimately raise one. A value that widens a guard is announced with a `@warn` built by [`relaxed_preferences_msg`](@ref), because the channel needs no code: a `LocalPreferences.toml` is data, it travels with a cloned project, and it applies before any user code runs. Widening means a raised [`RESOURCE_LIMITS`](@ref) or [`EQUATION_LIMITS`](@ref) cap, or a lowered [`STRING_DISTANCE`](@ref) suggestion threshold. A value that tightens a guard, or that equals the default it replaces, is silent. The comparison is against the default *in effect when the preference is applied*, which at load is the shipped default. See the amendment of `docs/adr/0041-one-resource-cap-per-sink.md`.

To persist a configuration, put the keys in the active project's `LocalPreferences.toml`, e.g.:

```toml
[PortfolioOptimisers]
equation_max_length = 512
equation_max_depth = 64
max_n_sim = 50_000
max_n_subsets = 1_000
max_frontier = 1_000
max_bins = 500
max_hop_count = 100
max_search_grid = 10_000
suggestion_min_score = 0.8
suggestion_distance = "damerau_levenshtein"
compact_show = 4
```

# Related

  - [`PREFERENCE_KEYS`](@ref)
  - [`PREFERENCE_DISTANCES`](@ref)
  - [`set_equation_limits!`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`set_string_distance!`](@ref)
  - [`set_compact_show!`](@ref)
  - [`relaxed_preferences_msg`](@ref)
"""
function apply_preferences!(prefs::AbstractDict{<:AbstractString, <:Any})
    relaxations = Vector{Tuple{String, Any, Any}}()
    ml = get(prefs, "equation_max_length", nothing)
    md = get(prefs, "equation_max_depth", nothing)
    if !(isnothing(ml) && isnothing(md))
        for (key, val) in ("equation_max_length" => ml, "equation_max_depth" => md)
            @argcheck(isnothing(val) || val isa Integer && !(val isa Bool) && val > 0,
                      ArgumentError("preference `$(key) = $(repr(val))` must be a positive integer."))
        end
        lim = @atomic EQUATION_LIMITS.default
        for (key, val, default) in (("equation_max_length", ml, lim.max_length),
                                    ("equation_max_depth", md, lim.max_depth))
            if !isnothing(val) && val > default
                push!(relaxations, (key, default, val))
            end
        end
        set_equation_limits!(; max_length = something(ml, lim.max_length),
                             max_depth = something(md, lim.max_depth))
    end
    xs = get(prefs, "max_n_sim", nothing)
    xb = get(prefs, "max_n_subsets", nothing)
    xf = get(prefs, "max_frontier", nothing)
    xn = get(prefs, "max_bins", nothing)
    xh = get(prefs, "max_hop_count", nothing)
    xg = get(prefs, "max_search_grid", nothing)
    if !all(isnothing, (xs, xb, xf, xn, xh, xg))
        for (key, val) in ("max_n_sim" => xs, "max_n_subsets" => xb, "max_frontier" => xf,
                           "max_bins" => xn, "max_hop_count" => xh, "max_search_grid" => xg)
            @argcheck(isnothing(val) || val isa Integer && !(val isa Bool) && val > 0,
                      ArgumentError("preference `$(key) = $(repr(val))` must be a positive integer."))
        end
        rlim = @atomic RESOURCE_LIMITS.default
        for (key, val, default) in
            (("max_n_sim", xs, rlim.max_n_sim), ("max_n_subsets", xb, rlim.max_n_subsets),
             ("max_frontier", xf, rlim.max_frontier), ("max_bins", xn, rlim.max_bins),
             ("max_hop_count", xh, rlim.max_hop_count),
             ("max_search_grid", xg, rlim.max_search_grid))
            if !isnothing(val) && val > default
                push!(relaxations, (key, default, val))
            end
        end
        set_resource_limits!(; max_n_sim = something(xs, rlim.max_n_sim),
                             max_n_subsets = something(xb, rlim.max_n_subsets),
                             max_frontier = something(xf, rlim.max_frontier),
                             max_bins = something(xn, rlim.max_bins),
                             max_hop_count = something(xh, rlim.max_hop_count),
                             max_search_grid = something(xg, rlim.max_search_grid))
    end
    ms = get(prefs, "suggestion_min_score", nothing)
    if !isnothing(ms)
        @argcheck(ms isa Real && !(ms isa Bool),
                  ArgumentError("preference `suggestion_min_score = $(repr(ms))` must be a real number."))
        msd = (@atomic STRING_DISTANCE.default).min_score
        if ms < msd
            push!(relaxations, ("suggestion_min_score", msd, ms))
        end
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
    if !isempty(relaxations)
        @warn relaxed_preferences_msg(relaxations)
    end
    return nothing
end
"""
    __init__()

Package load hook: reads the [`PREFERENCE_KEYS`](@ref) preferences of the active project via `Preferences.load_preference` and applies them to the global config defaults through [`apply_preferences!`](@ref). An invalid preference value fails closed — the package refuses to load — rather than running with a value the project got wrong.

This is the one channel that reaches the guards without running code: a `LocalPreferences.toml` is data, it travels with a cloned project or a template, and it is read here, before any user code. A valid value is therefore applied but not silent — a value that widens a guard is announced with a warning (see [`relaxed_preferences_msg`](@ref)).

# Related

  - [`apply_preferences!`](@ref)
  - [`PREFERENCE_KEYS`](@ref)
  - [`relaxed_preferences_msg`](@ref)
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

Abstract supertype for all custom exception types.

All error types specific to `PortfolioOptimisers.jl` should be subtypes of `PortfolioOptimisersError`.

# Related

  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`ConflictingArgumentError`](@ref)
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
julia> throw(IsNothingError(\"Input data must not be nothing\"))
ERROR: IsNothingError: Input data must not be nothing
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`ConflictingArgumentError`](@ref)
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
julia> throw(IsEmptyError(\"Input array must not be empty\"))
ERROR: IsEmptyError: Input array must not be empty
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`ConflictingArgumentError`](@ref)
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
julia> throw(IsNonFiniteError(\"Input array contains non-finite values\"))
ERROR: IsNonFiniteError: Input array contains non-finite values
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
@concrete struct IsNonFiniteError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is mutually exclusive with another and both were supplied — a "must-be-absent" constraint was violated (e.g. an argument that must be `nothing` because a conflicting one is set).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConflictingArgumentError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(ConflictingArgumentError(\"sbgt must be nothing when bgt is a BudgetCostEstimator\"))
ERROR: ConflictingArgumentError: sbgt must be nothing when bgt is a BudgetCostEstimator
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
@concrete struct ConflictingArgumentError <: PortfolioOptimisersError
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
julia> throw(PropertyPathError(\"cannot descend path `sol.w` on `JuMPOptimisationResult`: intermediate `sol` is `nothing`\"))
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
$(DocStringExtensions.TYPEDEF)

Exception type thrown when a [`DynamicAbstractWeights`](@ref) cannot resolve observation weights for the data it was handed, because no [`get_observation_weights`](@ref) method is implemented for that input's shape.

[`get_observation_weights`](@ref) returns `nothing` to mean *no weights were requested*, never *weights were unavailable*. Every `isnothing` branch downstream reads it the first way and computes an unweighted result, so a `DynamicAbstractWeights` that resolved to `nothing` would silently produce a numerically plausible but unweighted answer with no diagnostic. It raises instead.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ObservationWeightsError(msg)

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(ObservationWeightsError(\"MyWeights has no `get_observation_weights` method for a 2-dimensional input of size (3, 10)\"))
ERROR: ObservationWeightsError: MyWeights has no `get_observation_weights` method for a 2-dimensional input of size (3, 10)
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`DynamicAbstractWeights`](@ref)
  - [`get_observation_weights`](@ref)
"""
@concrete struct ObservationWeightsError <: PortfolioOptimisersError
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
$(DocStringExtensions.TYPEDEF)

Alias for an abstract 3-dimensional array of numeric types or JuMP scalar types.

Rank-restricted counterpart of [`ArrNum`](@ref), for the data that is a stack of matrices rather than a single one — a window of time-varying features, whose observation axis leads. Dispatching on this alias keeps the 2-D and 3-D entry points apart without a runtime `ndims` branch.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
  - [`ArrNum`](@ref)
"""
const Arr3Num = AbstractArray{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}, 3}
"""
    const VecNum_MatNum = Union{<:VecNum, <:MatNum}

Alias for a union of a numeric type or an abstract matrix of numeric types.

# Related

  - [`VecNum`](@ref)
  - [`MatNum`](@ref)
"""
const VecNum_MatNum = Union{<:VecNum, <:MatNum}
"""
    const MatNum_Arr3Num = Union{<:MatNum, <:Arr3Num}

Alias for a union of an abstract matrix and an abstract 3-dimensional array of numeric types.

The two admissible shapes of a feature matrix: a static `assets × features` matrix, and a window of time-varying features whose observation axis leads, `observations × assets × features`. Both shapes are carried by [`PricesResult`](@ref) and [`ReturnsResult`](@ref) and consumed by [`FeatureDistance`](@ref), which distinguishes them by dispatch rather than by an `ndims` branch.

# Related

  - [`MatNum`](@ref)
  - [`Arr3Num`](@ref)
"""
const MatNum_Arr3Num = Union{<:MatNum, <:Arr3Num}
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
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom value algorithms. These are user defined algorithms that return a custom value for an estimator.

The interfaces users must implement depend on the estimator type.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`CVal_Func_Num_VecNum`](@ref)
  - [`CustomValueExpectedReturns`](@ref)
"""
abstract type AbstractCustomValue <: AbstractAlgorithm end
"""
    const CVal_Func_Num_VecNum = Union{<:AbstractCustomValue, <:Func_Num_VecNum}

Alias for the union of `AbstractCustomValue` and `Func_Num_VecNum`. This is used to define the type of the `val` field in [`CustomValueExpectedReturns`](@ref).
"""
const CVal_Func_Num_VecNum = Union{<:AbstractCustomValue, <:Func_Num_VecNum}
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

Abstract supertype for all estimator value algorithm types.

Subtypes of `AbstractEstimatorValueAlgorithm` implement algorithms for computing constraint result values. These are used to extend or modify the behavior of estimators in a composable and modular fashion.

# Interfaces

In order to implement a new estimator value algorithm which will work seamlessly with the library, subtype `AbstractEstimatorValueAlgorithm` with all necessary parameters struct, and implement the following method:

  - `estimator_to_val(alg::AbstractEstimatorValueAlgorithm, sets::UniverseSets, val::Option{<:Number} = nothing, key::Option{<:AbstractString} = nothing; datatype::DataType = Float64, strict::Bool = false) -> Num_VecNum`: Converts an estimator value dictionary to a numeric or vector of numeric value. Usually this should compute some version of:
      + `val = ifelse(isnothing(val), <default value use with datatype element type>, val)`: Computes the default value to use if `val` is `nothing`.
      + `nx = sets.dict[ifelse(isnothing(key), sets.xkey, key)]`: Gets the universe to use for mapping values to assets.

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

julia> function PortfolioOptimisers.estimator_to_val(alg::MyIncreasingValue, sets::UniverseSets,
                                                     val::PortfolioOptimisers.Option{<:Number} = nothing,
                                                     key::PortfolioOptimisers.Option{<:AbstractString} = nothing;
                                                     datatype::DataType = Float64,
                                                     strict::Bool = false)
           val = ifelse(isnothing(val), zero(datatype), val)
           nx = sets.dict[ifelse(isnothing(key), sets.xkey, key)]
           arr = ((1 - val):(length(nx) - val))
           return arr
       end

julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"sha\", \"bis\", \"man\"]))
UniverseSets
   xkey ┼ String: "nx"
  uxkey ┼ String: "ux"
   fkey ┼ String: "nf"
  ufkey ┼ String: "uf"
   zkey ┼ String: "nz"
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

`nothing` is returned only when `w === nothing`, and means *no weights were requested* — every `isnothing` branch downstream reads it that way and computes an unweighted result. It never means *weights were unavailable*: a [`DynamicAbstractWeights`](@ref) with no method for the given input shape throws [`ObservationWeightsError`](@ref) rather than resolving to `nothing`, because returning `nothing` there would silently yield an unweighted answer that looks plausible.

This is why call sites need no strictness check of their own. A `DynamicAbstractWeights` is resolved *before* dispatch (see [`average_drawdown`](@ref) for the pattern), so the estimator downstream only ever sees a concrete weight vector or a deliberate `nothing`.

## The returned vector is borrowed, not owned

For a `StatsBase.AbstractWeights` this returns **the stored object itself**, not a copy — an estimator's `w` field is handed straight back. So the caller may **read** it but must never **mutate** it: writing through it permutes the estimator's own configuration, and every later evaluation of that estimator is then wrong.

This is the same obligation the rest of `src/` already meets: a `reverse!` or a `sort!` is applied only to a vector the surrounding expression has just allocated. Beware the indirect route in particular — `view(w, order)` is a **view**, so `reverse!` on the view writes through into `w` just as surely as `reverse!(w)` would. Reverse the permutation instead, or sort into a fresh vector.

A defensive copy here was considered and rejected: it would cost an allocation on every evaluation of every weighted estimator, and the obligation is cheap to keep.

# Arguments

  - $(arg_dict[:oow])
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

# Returns

  - `w::Option{<:VecNum}`: The observation weights, or `nothing` when `w` is `nothing`.

# Throws

  - [`ObservationWeightsError`](@ref): if `w` is a [`DynamicAbstractWeights`](@ref) with no `get_observation_weights` method for the shape of the given input.

# Related

  - [`ObsWeights`](@ref)
  - [`DynamicAbstractWeights`](@ref)
  - [`ObservationWeightsError`](@ref)
"""
function get_observation_weights(::Nothing, args...; kwargs...)
    return nothing
end
function get_observation_weights(w::DynamicAbstractWeights, args...; kwargs...)
    name = nameof(typeof(w))
    X = isempty(args) ? nothing : first(args)
    shape = if isa(X, AbstractArray)
        "a $(ndims(X))-dimensional input of size $(size(X))"
    else
        "the given input"
    end
    return throw(ObservationWeightsError("$name is a DynamicAbstractWeights with no `get_observation_weights` method for $shape. Implement `get_observation_weights(w::$name, X::VecNum; kwargs...)` and/or `get_observation_weights(w::$name, X::MatNum; dims::Int = 1, kwargs...)`, or pass a `StatsBase.AbstractWeights` instead (or `nothing` to compute unweighted). See the `DynamicAbstractWeights` docstring for a worked example."))
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
    assert_all_finite(val::ArrNum, sym::Sym_Str = :val)

Assert that *every* element of `val` is finite, failing closed with an [`IsNonFiniteError`](@ref) otherwise.

Unlike [`assert_finite`](@ref), which only requires *one* finite element, this demands the whole array be finite. It guards the comparison-based covariance estimators ([`GerberCovariance`](@ref), [`SmythBrobyCovariance`](@ref)): their `X .>= sd` / `X .<= -sd` comparisons silently evaluate a `NaN` entry as `false`, masking it as "no co-movement" and yielding a finite, plausible, *wrong* covariance rather than an error. Clean returns first with an asset selector (e.g. [`CompleteAssetSelector`](@ref)) or [`MissingDataFilter`](@ref) — non-finite entries in a returns matrix are a supported input to *those*, but not to a comparison-based estimator. The message reports the count of offending entries and the first offending index only — never the data values.

# Arguments

  - `val`: Array to check.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Related

  - [`assert_finite`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
function assert_all_finite(val::ArrNum, sym::Sym_Str = :val)::Nothing
    @argcheck(all(isfinite, val),
              IsNonFiniteError("all(isfinite, $sym) must hold. Got $(count(!isfinite, val)) non-finite entries; first at $(findfirst(!isfinite, val))."))
    return nothing
end
"""
    assert_resource_cap(val::Integer, cap::Integer, sym::Sym_Str, knob::Sym_Str)

Assert that an untrusted sizing integer `val` does not exceed the active [`RESOURCE_LIMITS`](@ref) ceiling `cap`, failing closed with a `DomainError` otherwise.

`sym` names the offending field in the message (e.g. `:n_sim`) and `knob` names the [`ResourceLimits`](@ref) field to raise (e.g. `:max_n_sim`), so the error tells the caller both what was rejected and how to allow it deliberately.

# Arguments

  - `val`: The requested size.
  - `cap`: The active ceiling.
  - `sym`: Symbolic name of the offending field.
  - `knob`: Symbolic name of the [`ResourceLimits`](@ref) field that raises the ceiling.

# Returns

  - `nothing`.

# Related

  - [`RESOURCE_LIMITS`](@ref)
  - [`set_resource_limits!`](@ref)
  - [`with_resource_limits`](@ref)
  - [`assert_frontier_sweep_cap`](@ref)
"""
function assert_resource_cap(val::Integer, cap::Integer, sym::Sym_Str,
                             knob::Sym_Str)::Nothing
    @argcheck(val <= cap,
              DomainError(val,
                          "$sym = $val exceeds RESOURCE_LIMITS[].$knob = $cap. Raise it with set_resource_limits!(; $knob) — or with_resource_limits for a single scope — for genuinely large machine-authored runs."))
    return nothing
end
"""
    resolve_rng(rng::Random.AbstractRNG, seed::Option{<:Integer})

Resolve which random number generator to draw from given an optional `seed`.

A supplied `seed` yields a fresh, private generator — a `copy` of `rng` reseeded with `seed`
via `Random.seed!` — so a seeded estimator is reproducible **without** reseeding, and thereby
silently derandomising, the task-global RNG the caller may also own (the default `rng` is
`Random.default_rng()`, a shared object). When `seed` is `nothing`, `rng` is returned unchanged
and used as-is.

Copying `rng` before seeding (rather than constructing a fixed generator type such as
`Random.Xoshiro(seed)`) preserves both the caller's generator *object* — it is never mutated —
and its *type*, so a caller-supplied portable generator (e.g. `StableRNGs.StableRNG`) keeps
producing the same stream `Random.seed!(rng, seed)` did in place. The observable draws are thus
identical to the old in-place seeding; only the side effect on the caller's stream disappears.

# Arguments

  - `rng`: Fallback random number generator, used verbatim when `seed` is `nothing`.
  - `seed`: Optional seed. If set, a private `Random.seed!(copy(rng), seed)` is returned instead of touching `rng`.

# Returns

  - `Random.AbstractRNG`: the generator to draw from.
"""
function resolve_rng(rng::Random.AbstractRNG, seed::Option{<:Integer})
    return isnothing(seed) ? rng : Random.seed!(copy(rng), seed)
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
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a matrix-source selector names one of the two carriers.

Source selectors pick which of the two carriers a matrix is read from: `:prior` reads the prior result, `:data` reads the raw returns result. `x_src` selects the returns matrix `X`.

# Arguments

  - `src`: Selector to check.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`HierarchicalOptimiser`](@ref)
  - [`NestedClustered`](@ref)
"""
function assert_source_selector(src::Symbol, sym::Sym_Str = :x_src)::Nothing
    @argcheck(src in (:prior, :data),
              ArgumentError("$sym must be :prior or :data, got $(repr(src))"))
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

# Validation

  - `size(X, 1) == size(X, 2)`.

# Returns

  - `nothing`.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `dims` selects a valid matrix dimension (`dims in (1, 2)`).

# Arguments

  - `dims`: Dimension selector to check.
  - `sym`: Symbolic name used in the error message.

# Returns

  - `nothing`.

# Details

  - Throws `DomainError` if `dims ∉ (1, 2)`.

# Related

  - [`assert_matrix_issquare`](@ref)
"""
function assert_dims(dims::Integer, sym::Sym_Str = :dims)::Nothing
    @argcheck(dims in (1, 2),
              DomainError(dims, "$sym must be 1 or 2. Got\n$sym => $(dims)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate `dims` and return the matrices with the observations along the rows.

# Arguments

  - `dims`: Dimension along which the observations lie.
  - `A`, `B`, `Cs...`: Matrices to orient. A `nothing` passes through unchanged, so an optional matrix needs no branch of its own.

# Validation

  - `dims in (1, 2)`, by [`assert_dims`](@ref).

# Returns

  - `A`: The oriented matrix, when one matrix is given.
  - `(A, B, Cs...)`: A tuple of the oriented matrices, when more than one is given.

# Details

  - `dims == 1` returns the input untouched. `dims == 2` returns its `transpose`.
  - The guard and the orientation are one call, so a caller cannot orient a matrix without validating `dims`. This is the single decision point: a leaf that spelled the guard and the `transpose` by hand could omit the guard and answer a `dims` of `3` with the raw input.

# Related

  - [`assert_dims`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)
"""
function dims_oriented(dims::Integer, A::Option{<:AbstractMatrix})
    assert_dims(dims)
    return isnothing(A) || isone(dims) ? A : transpose(A)
end
function dims_oriented(dims::Integer, A::Option{<:AbstractMatrix},
                       B::Option{<:AbstractMatrix}, Cs::Option{<:AbstractMatrix}...)
    assert_dims(dims)
    return map(x -> dims_oriented(dims, x), (A, B, Cs...))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a composite result containing a vector and a scalar.

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

Abstract supertype for all norm-based error algorithms.

All concrete and/or abstract types representing norm-based error algorithms (such as second-order cone or norm-one error) should be subtypes of `NormError`.

# Interfaces

In order to implement a new norm-based error algorithm which will work seamlessly with the library, subtype `NormError` with all necessary parameters struct, and implement the following method:

  - `norm_factor(f::NormError, T::Number) -> Number`: Returns the divisor that scales the norm. The `T === nothing` case is already covered by a generic method that returns `1`.

The functor side is [`norm_error`](@ref), and the model side is `set_risk_constraints!` for [`TrackingRiskMeasure`](@ref) and `set_tracking_error_constraints!` for [`TrackingError`](@ref). All three must agree.

# Related

  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
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

The source states the denominator as ``\\sqrt{T}``. The default `ddof = 1` gives the sample denominator ``\\sqrt{T-1}``. Set `ddof = 0` to recover the source.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L2Norm(;
        ddof::Integer = 1
    ) -> L2Norm

Keywords correspond to the struct's fields.

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
  - [`SquaredL2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.16.
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

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Details

  - The value is the square of the [`L2Norm`](@ref) error. A `settings.ub` on a [`TrackingRiskMeasure`](@ref) therefore carries squared units. The JuMP model converts the bound with a square root, so the two encodings accept the same bound.

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
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.16.
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
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.17.
"""
struct L1Norm <: NormError end
"""
$(DocStringExtensions.TYPEDEF)

L-p norm error estimator.

`LpNorm` takes the Lp-norm of the difference between the portfolio and the benchmark returns, and divides it by ``(T - d)^{1/p}``. It generalises [`L1Norm`](@ref) and [`L2Norm`](@ref) to a free norm order.

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

# Details

  - The constructor does not bound `p`. The JuMP model does: both `set_risk_constraints!` and `set_tracking_error_constraints!` need `1 < p` for the power cone, and throw a `DomainError` otherwise. The functor accepts any `p` that `LinearAlgebra.norm` accepts.
  - `norm_factor` computes the divisor with `cbrt` when `p == 3`, the default.

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
  - [`SquaredL2Norm`](@ref)
  - [`LInfNorm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
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

`LInfNorm` takes the largest absolute deviation between the portfolio and the benchmark returns, and divides it by ``T - d``.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\lVert \\boldsymbol{a} - \\boldsymbol{b} \\rVert_\\infty}{T - d}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b})``: L∞-norm error, the largest absolute deviation.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LInfNorm(; ddof::Integer = 0) -> LInfNorm

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> LInfNorm()
LInfNorm
  ddof ┴ Int64: 0
```

# Related

  - [`NormError`](@ref)
  - [`LpNorm`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
"""
@concrete struct LInfNorm <: NormError
    """
    $(field_dict[:ddof])
    """
    ddof
    function LInfNorm(ddof::Integer)::LInfNorm
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function LInfNorm(; ddof::Integer = 0)::LInfNorm
    return LInfNorm(ddof)
end
"""
    norm_error(f::L2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::SquaredL2Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(::L1Norm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LpNorm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::LInfNorm, a, b, T::Option{<:Number} = nothing)
    norm_error(f::Option{<:NormError}, a, T::Option{<:Number} = nothing)

Compute the norm-based tracking error between portfolio and benchmark weights.

`norm_error` takes the norm that `f` selects, and divides it by the [`norm_factor`](@ref) that the same `f` declares. Each [`NormError`](@ref) subtype names one pair. The three-argument form takes the norm of `a - b`. The two-argument form takes the norm of `a` alone, for a caller that already holds the deviation vector; `f = nothing` there means an unweighted L2 norm.

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

  - `f`: Norm-based error algorithm, a [`NormError`](@ref) subtype.
  - `a`: Portfolio weights, or the deviation vector in the two-argument form.
  - `b`: Benchmark weights.
  - `T`: Optional number of observations.

# Returns

  - `err::Number`: Norm-based tracking error.

# Details

  - The norm is divided by [`norm_factor`](@ref), which is `1` when `T` is `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.norm_error(L2Norm(), [0.5, 0.5], [0.6, 0.4], 2)
0.14142135623730948

julia> PortfolioOptimisers.norm_error(L1Norm(), [0.5, 0.5], [0.6, 0.4], 2)
0.09999999999999998
```

# Related

  - [`NormError`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
  - [`Option`](@ref)
  - [`norm_factor`](@ref)
"""
function norm_error end
"""
    norm_factor(f::Union{Nothing, <:NormError}, T::Option{<:Number})

Compute the denominator that scales a norm in [`norm_error`](@ref).

The factor is the single place where the optional observation count `T` is turned into a divisor. Each [`NormError`](@ref) declares its own factor, and the `T === nothing` case is a method, not a branch inside one. A branch is what let `ifelse` evaluate `T - f.ddof` on the `nothing` path.

# Arguments

  - `f`: Norm-based error algorithm, a [`NormError`](@ref) subtype. `nothing` means an unweighted L2 norm.
  - `T`: Optional number of observations.

# Returns

  - `factor::Number`: Divisor for the norm. It is `1` when `T` is `nothing`.

# Details

  - `nothing`: `sqrt(T)`.
  - [`L2Norm`](@ref): `sqrt(T - f.ddof)`.
  - [`SquaredL2Norm`](@ref): `T - f.ddof`.
  - [`L1Norm`](@ref): `T`.
  - [`LpNorm`](@ref): `(T - f.ddof)^(1/f.p)`, computed with `cbrt` when `f.p == 3`.
  - [`LInfNorm`](@ref): `T - f.ddof`.

# Examples

```jldoctest
julia> PortfolioOptimisers.norm_factor(L2Norm(), 4)
1.7320508075688772

julia> PortfolioOptimisers.norm_factor(LInfNorm(), nothing)
1
```

# Related

  - [`norm_error`](@ref)
  - [`NormError`](@ref)
  - [`Option`](@ref)
"""
function norm_factor(::Union{Nothing, <:NormError}, ::Nothing)
    return 1
end
function norm_factor(::Nothing, T::Number)
    return sqrt(T)
end
function norm_factor(f::L2Norm, T::Number)
    return sqrt(T - f.ddof)
end
function norm_factor(f::SquaredL2Norm, T::Number)
    return T - f.ddof
end
function norm_factor(::L1Norm, T::Number)
    return T
end
function norm_factor(f::LpNorm, T::Number)
    factor = T - f.ddof
    return if f.p == 3
        cbrt(factor)
    else
        factor^(inv(f.p))
    end
end
function norm_factor(f::LInfNorm, T::Number)
    return T - f.ddof
end
function norm_error(f::L2Norm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, 2) / norm_factor(f, T)
end
function norm_error(f::L2Norm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, 2) / norm_factor(f, T)
end
function norm_error(f::Nothing, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, 2) / norm_factor(f, T)
end
function norm_error(f::SquaredL2Norm, a, b, T::Option{<:Number} = nothing)
    val = LinearAlgebra.norm(a - b, 2)
    return val^2 / norm_factor(f, T)
end
function norm_error(f::SquaredL2Norm, a, T::Option{<:Number} = nothing)
    val = LinearAlgebra.norm(a, 2)
    return val^2 / norm_factor(f, T)
end
function norm_error(f::L1Norm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, 1) / norm_factor(f, T)
end
function norm_error(f::L1Norm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, 1) / norm_factor(f, T)
end
function norm_error(f::LpNorm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, f.p) / norm_factor(f, T)
end
function norm_error(f::LpNorm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, f.p) / norm_factor(f, T)
end
function norm_error(f::LInfNorm, a, b, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a - b, Inf) / norm_factor(f, T)
end
function norm_error(f::LInfNorm, a, T::Option{<:Number} = nothing)
    return LinearAlgebra.norm(a, Inf) / norm_factor(f, T)
end

export IsEmptyError, IsNothingError, IsNonFiniteError, ConflictingArgumentError,
       PropertyPathError, ObservationWeightsError, VecScalar, L2Norm, SquaredL2Norm, L1Norm,
       LpNorm, LInfNorm
