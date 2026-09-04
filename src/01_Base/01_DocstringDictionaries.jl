"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build a documentation dictionary from `pairs`, and throw if a key appears more than once.

A `Dict` literal is last-wins, so a repeated key drops the earlier entry with no warning and
makes its prose unreachable. This constructor is the guard against that: it fails at load
time and names both descriptions, so the duplicate is visible instead of silent.

# Algorithm

 1. Start `dict` empty.
 2. For each pair, in the order the caller wrote it, raise an `ArgumentError` when `dict` already holds the key. The message names `name`, the key, and both descriptions.
 3. Otherwise store the pair in `dict`.

# Arguments

  - `name::Symbol`: Name of the dictionary under construction, used in the error message.
  - `pairs`: The key-description pairs, in the order they are written.

# Validation

  - Each key appears exactly once. A repeat raises an `ArgumentError` naming `name`, the key, and both descriptions.

# Returns

  - `dict::Dict{Symbol, String}`: The built table.

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
                                 :pf_n => "`n`: Number of observations folded into the state.",
                                 :pf_mu => "`mu`: Running mean of the observations folded into the state, `assets × 1`.",
                                 :pf_M2 => "`M2`: Running second central co-moment accumulator, `assets × assets`. It is the sum over the observations and not the covariance, so a read-out divides it by `n`.",
                                 :pf_M3 => "`M3`: Running third central co-moment accumulator, `assets × assets²`. It is the sum over the observations and not the coskewness, so a read-out divides it by `n`.",
                                 :pfcache => "`cache`: Optional partial-fit state. It is `nothing` until [`partial_fit!`](@ref) writes one, and the estimator's read-out verb reads it when the caller gives no data matrix. Each propagation channel does one thing with it: [`factory`](@ref) carries it unchanged, because a factory call resolves configuration rather than the sample; [`port_opt_view`](@ref) slices it to the selected assets by index copy, so the viewed estimator answers over those assets alone; and [`obs_weights_view`](@ref) drops it, because no slice of a state exists on the observation axis. A family whose state has no exact asset slice drops it on both axes and names the reason.",
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
                                 :drtgt => "`drtgt`: Dimension reduction target.",#
                                 :csrint => "`intercept`: Whether a per-observation intercept is fitted. When `false`, the regression runs through the origin of the cross-section.",
                                 ## Gerber
                                 :gerbalg => "`alg`: Gerber covariance algorithm.",#
                                 :gerbce => "`ce`: Gerber covariance estimator.",#
                                 :stdarr => "`sd`: Standard deviation vector of `X`, shaped to be consistent with `X`.",#
                                 :c1 => "`c1`: Zone of confusion threshold, in units of the asset's standard deviation. It is read against the raw, uncentred return, and it rejects an observation only when both assets fall inside it.",#
                                 :c2 => "`c2`: Zone of indecision threshold, in units of the asset's standard deviation. It is read against the centred, standardised return, and it rejects an observation when both assets fall inside it. A centred return of exactly zero is inside it at every `c2`.",#
                                 :c3 => "`c3`: Outer cut-off, in units of the asset's standard deviation. It is read against the centred, standardised return, and it rejects an observation when either asset exceeds it.",#
                                 :sbn => "`n`: Severity exponent of the Smyth-Broby contribution. It sets how hard the divergence of a pair is penalised.",#
                                 :sbalg => "`alg`: Smyth-Broby covariance algorithm.",#
                                 ## Mutual and var info
                                 :bins => "`bins`: Binning algorithm or fixed number of bins.",#
                                 :normalise => "`normalise`: Whether to normalise the mutual and/or variation of information calculation.",#
                                 :xj => "`xj`: Data vector for variable `j`.",#
                                 :xi => "`xi`: Data vector for variable `i`.",#
                                 :jidx => "`j`: Index of variable `j`.",#
                                 :iidx => "`i`: Index of variable `i`.",#
                                 :Tobs => "`T`: Number of observations.",#
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
                                 :fdsel => "`sel`: Column selector naming the feature slices the distance reads, or `nothing` to read every column. A vector of names resolves against the carrier's `nz`, where an entry that is a key of `sets.dict` expands to that taxonomy's columns. A vector of integers indexes the feature axis directly and needs no names.",#
                                 :fdsets => "`sets`: Universe sets whose taxonomy keys `sel` may name, or `nothing`. The field is checked at construction rather than bounded by its type, because `UniverseSets` is defined in a later file than this one.",#
                                 :fdstrict => "`strict`: Whether a `sel` entry that resolves against no feature column throws instead of warning and being dropped.",#
                                 # Priors.
                                 :pe => "`pe`: Prior estimator.",#
                                 :pr => "`pr`: Prior result.",#
                                 :per => "`pr`: Prior estimator or result.",#
                                 :pr_rr => "`pr`: Prior result or returns result. Both carry the asset returns matrix `X` and the feature matrix `Z`, so either can supply them.",#
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
                                 :alpha => "`alpha`: Quantile level for the lower tail. The bound is [`Num_SigCal`](@ref), so the slot takes the level itself, an [`AbstractSignificanceCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :alpha_ltd => "`alpha`: Quantile level for the lower tail.",#
                                 :beta => "`beta`: Quantile level for the upper tail. The bound is [`Num_SigCal`](@ref), so the slot takes the level itself, an [`AbstractSignificanceCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
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
                                 :bl_rf => "`rf`: Risk-free rate. The Black-Litterman update blends the prior mean against the view returns, so it runs on the total-return scale those are written on. A mean taken from a wrapped prior estimator is on that scale already; an equilibrium mean is a bare risk premium, and the rate converts it before the update. A member with no equilibrium branch has nothing to convert and adds the rate to the posterior asset expected returns instead. It is added exactly once either way, and the wrapped prior estimators are left alone, so a risk-free rate one of them applied internally stays where it is.",#
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
                                 :sets_af => "`sets`: Universe sets. This estimator reads **two** declared axes: `a_views` resolves against `sets.dict[sets.xkey]`, `f_views` against the time-series factor axis `sets.dict[sets.tfkey]`, and each axis must name the columns of `X` and `F` respectively, in order. Only the axis a [`LinearConstraintEstimator`](@ref) actually resolves names against is required — views supplied as a [`BlackLittermanViews`](@ref) result carry their own matrix and need no universe. A view slices the asset axis and leaves the factor entries untouched, which is why this field is `@vprop`.",#
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
                                 :sbar => "`sbar`: Number of largest losses considered by the integer conditional value-at-risk formulation. An `Integer` is a count, a fraction in `(0, 1]` is a fraction of the observations, and `nothing` applies the rule of thumb `max(2 * s, ceil(Int, 2 * alpha * T))` capped at `T`, where `s` is the number of positions, counted from the largest loss, at which the prior probabilities first reach `alpha`. The rule comes from the reference, which observes that a view above the prior CVaR needs about `s` positions and a view below it needs more. It trades exactness for solve time: `sbar = T` is always exact, and a smaller `sbar` is exact whenever the posterior puts at least `alpha` of its mass on the `sbar` largest losses, and infeasible otherwise. Raise it when the solve reports infeasibility.",#
                                 :zpct => "`pct`: Fractional half-width of the grid of entropic value-at-risk dual variables, centred on the value that attains the prior entropic value-at-risk. An upper-bound or equality view centres the grid on the value [`ep_evar_anchor`](@ref) finds instead, and the width then covers the movement the other views of the model cause.",#
                                 :zK => "`K`: Number of points of the grid of entropic value-at-risk dual variables. Must be odd, so the centre is a point of the grid. The points are equidistant and span `zc * (1 - pct)` to `zc * (1 + pct)` for a grid centred on `zc`, so `K` sets the resolution of the grid alone. Every point is one more binary variable of the mixed-integer program an upper-bound or equality view builds, so raise it when `pct` widens rather than on its own.",#
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
                                 :rlvar_bigM => "`M`: Big-M constant of the grid relativistic value-at-risk formulation.",#
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
                                 :cal_n => "`n`: Number of observations the tail is to hold. It is a count, not a probability, and it is the whole content of the rule.",#
                                 :cal_c => "`c`: Rate coefficient. The significance level is this coefficient divided by the square root of the number of observations.",#
                                 :cal_target => "`target`: Target value of the Kaniadakis logarithm, the coefficient [`RRM`](@ref) multiplies its dual variable by. The rule returns the deformation parameter that meets it.",#
                                 :cal_kmin => "`kmin`: Floor under the count of order statistics the Hill estimate reads. The count is `ceil(alpha * T * N)` over the pool, and a count below this floor is refused rather than estimated: a Hill estimate over too few order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it.",#
                                 :cal_kmin_rad => "`kmin`: Floor under the count of order statistics the Hill estimate reads. The count is `ceil(alpha * T)` over the radial series, and a count below this floor is refused rather than estimated: a Hill estimate over too few order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. The radial series holds one entry per observation, so the same floor binds harder here than it does over a pool.",#
                                 :cal_ctx_alpha => "`alpha`: Significance level of a sibling slot the owner resolved first, or `nothing` when the site names none. The two slots travel together, so the per-type resolution resolves `alpha` first and puts the number in the context of the slot that reads it.",#
                                 :cal_ctx_series => "`series`: The series the slot owner prices, one of [`ReturnsSeries`](@ref), [`AbsoluteDrawdownSeries`](@ref) and [`RelativeDrawdownSeries`](@ref), which [`calibration_series`](@ref) states. A drawdown marker puts the per-column drawdown series of `pr.X` in place of its columns, and each rule reads that substitution on its own terms.",#
                                 :cal_confidence => "`confidence`: Confidence level of the chi-squared quantile the radius is read off. A higher level buys a larger ball, so the model prices a wider set of measures.",#
                                 :cal_scale => "`scale`: Scale of the radius, in the units of the series the slot owner prices, or `nothing` to read the average per-asset dispersion of that series off the sample. The chi-squared factor is dimensionless, so this field carries the whole of the radius' units.",#
                                 :cal_rate_c => "`c`: Rate coefficient. The radius is this coefficient divided by the square root of the number of observations.",#
                                 :cal_dim_confidence => "`confidence`: Confidence level the measure-concentration bound is read at. It enters the radius as `log(1 / (1 - confidence))`, so a higher level buys a larger ball, and the exponent of the sample size flattens the buying.",#
                                 :cal_dim_scale => "`scale`: Scale of the radius, in the units of the series the slot owner prices, or `nothing` to read the average per-asset dispersion of that series off the sample. The rate factor is dimensionless, so this field carries the whole of the radius' units.",#
                                 :cal_fraction => "`fraction`: Fraction of the universe that must stay effective. The rule reads the asset count off the prior result and multiplies it by this fraction, so the floor moves with the universe rather than with a count the caller pins.",#
                                 :cal_ctx_p => "`p`: Norm order the quantity is read against, or `nothing` when the site names none. The order belongs to the constraint or to the penalty rather than to the rule, so each site that carries one states it here.",#
                                 :cal_ratio => "`ratio`: Number of mean terms that one tail term is worth. The rule returns the tail weight that prices the tail term at this multiple of the mean term, on the sample the prior result carries, so `1` is parity and `2` prices the tail term at twice the mean term.",#
                                 :vr_rm => "`vr`: Variance risk measure component.",#
                                 :sk_rm => "`sk`: Skewness risk measure component.",#
                                 :kt_rm => "`kt`: Kurtosis risk measure component.",#
                                 :alg1 => "`alg1`: First algorithm variant.",#
                                 :alg2 => "`alg2`: Second algorithm variant.",#
                                 :N_kt => "`N`: Optional number of eigenvalues per asset for the approximate cokurtosis formulation.",#
                                 :kappa => "`kappa`: Relativistic deformation parameter. The bound is [`Num_DefCal`](@ref), so the slot takes the parameter itself, an [`AbstractDeformationCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :kappa_a => "`kappa_a`: Relativistic deformation parameter for the lower tail. The bound is [`Num_DefCal`](@ref), so the slot takes the parameter itself, an [`AbstractDeformationCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :kappa_b => "`kappa_b`: Relativistic deformation parameter for the upper tail. The bound is [`Num_DefCal`](@ref), so the slot takes the parameter itself, an [`AbstractDeformationCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :l_a => "`l_a`: Weight of the tail term in the Esfahani-Kuhn loss of the lower tail. The mean term is not scaled by it. The bound is [`Num_AmbTwtCal`](@ref), so the slot takes the weight itself, an [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :r_a => "`r_a`: Radius of the type-1 Wasserstein ambiguity ball of the lower tail. It multiplies a decision variable, so it is not a constant offset. The bound is [`Num_AmbRadCal`](@ref), so the slot takes the radius itself, an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :l_b => "`l_b`: Weight of the tail term in the Esfahani-Kuhn loss of the upper tail. The mean term is not scaled by it. The bound is [`Num_AmbTwtCal`](@ref), so the slot takes the weight itself, an [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :r_b => "`r_b`: Radius of the type-1 Wasserstein ambiguity ball of the upper tail. It multiplies a decision variable, so it is not a constant offset. The bound is [`Num_AmbRadCal`](@ref), so the slot takes the radius itself, an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
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
                                 :w_bm_ret => "`w`: Benchmark portfolio returns vector. It holds `T` returns, one per observation, and **not** `N` weights, so its length must match the number of rows of the return matrix the model is built on.",#
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
                                 :imsk => "`imsk`: The Investable Mask the optimisation reduced on: `true` at every asset whose prior moments were finite. It is `nothing` when every asset was investable, and that sentinel is what skips both the reduction and the expansion. [`investable_mask`](@ref) derives it once from the full-universe prior result, and the bundle carries it, because the reduced prior can no longer yield it.",#
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
                                 :l2c => "`l2c`: 2-norm ceiling on the weights — bounds `norm(w, 2) <= l2c * k` (`k` is the budget, `1` for a fully invested portfolio). Smaller `l2c` forces a more evenly spread portfolio. Used as a diversification floor via the reciprocal: `l2c = 1 / sqrt(m)` requires at least `m` effective assets (`inv(norm(w, 2)^2) >= m`). Norm-constraint family with `lpc` and `linfc`. The bound is [`Num_NormCeilCal`](@ref) under the time-dependent wrapper, so the slot takes the ceiling itself, an [`AbstractNormCeilingCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :lpc => "`lpc`: p-norm ceiling(s) on the weights at an arbitrary norm order. Each [`LpRegularisation`](@ref) supplies a norm order `p` and a bound `val`, enforcing `norm(w, p) <= val * k`. Smaller `val` forces a more evenly spread portfolio. Used as a diversification floor via the reciprocal: `val = m^(1/p - 1)` requires at least `m` order-`p` effective assets (`sum(abs.(w) .^ p)^inv(1 - p) >= m`), which is [`number_effective_assets`](@ref) taken to an arbitrary order. Norm-constraint family with `l2c` and `linfc`.",#
                                 :linfc => "`linfc`: ∞-norm ceiling on the weights — a cap on the largest absolute weight: `norm(w, Inf) <= linfc * k`. So `linfc = 0.2` caps the largest weight at 20% of a fully invested portfolio. Used as a diversification floor via the reciprocal: `linfc = 1 / m` spreads the portfolio across at least `m` assets. Norm-constraint family with `l2c` and `lpc`. The bound is [`Num_NormCeilCal`](@ref) under the time-dependent wrapper, so the slot takes the ceiling itself, an [`AbstractNormCeilingCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :l1 => "`l1`: L1 regularisation coefficient. It is the ambiguity radius of a type-``\\infty`` Wasserstein ground metric, whose dual norm is the 1-norm, so the bound is [`Num_AmbRadCal`](@ref) under the time-dependent wrapper and the slot takes an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :l2 => "`l2`: L2 regularisation term(s).",#
                                 :linf => "`linf`: L∞ regularisation coefficient. It is the ambiguity radius of a type-1 Wasserstein ground metric, whose dual norm is the ``\\infty``-norm, so the bound is [`Num_AmbRadCal`](@ref) under the time-dependent wrapper and the slot takes an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments.",#
                                 :lp => "`lp`: Lp regularisation specification(s).",#
                                 :l2reg_val => "`val`: L2 regularisation penalty coefficient. It is the ambiguity radius of a type-2 Wasserstein ground metric, so the bound is [`Num_AmbRadCal`](@ref) and the slot takes an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) that computes it from the prior result, or a plain function of the same five arguments. That reading holds only for the un-squared penalty of [`SOCRiskExpr`](@ref), so [`assert_ambiguity_radius_formulation`](@ref) refuses a rule beside a squared formulation.",#
                                 :l2reg_alg => "`alg`: Second-moment formulation used to express the L2 penalty.",#
                                 :lpreg_p => "`p`: Norm order, `p > 1`.",#
                                 :lpreg_val => "`val`: Penalty coefficient when the estimator is used as a regularisation term (the `lp` field of [`JuMPOptimiser`](@ref)), or the upper bound on the p-norm of the weights when it is used as a norm constraint (the `lpc` field). As a regularisation term it is the ambiguity radius of a type-``q`` Wasserstein ground metric with ``1/p + 1/q = 1``. As a norm constraint it is a ceiling, which is a different quantity. One field therefore carries two readings, so the bound is [`Num_AmbRadNormCeilCal`](@ref), which admits both rule families, and each of the two routes refuses the family that has no reading on it. It is the one slot that admits no plain function, because a function names no family and the two routes read the family.",#
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
                                 # Data carrier fields.
                                 :ivpa_iv => "`ivpa`: Implied volatility risk premium adjustment, if a vector (assets × 1).",#
                                 :nz_feat => "`nz`: Names or identifiers of feature columns (features × 1).",#
                                 # Prediction result fields.
                                 :pred_nx => "`nx`: Asset name vector.",#
                                 :pred_nf => "`nf`: Factor name vector.",#
                                 :pred_nb => "`nb`: Benchmark name vector.",#
                                 :pred_B => "`B`: Benchmark returns.",#
                                 :ts => "`ts`: Timestamp vector.",#
                                 :iv_ret => "`iv`: Implied volatilities.",#
                                 :ivpa => "`ivpa`: Implied volatility risk premium adjustment.",#
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
                                 :us_tfkey => "`tfkey`: Key in `dict` identifying the **time-series** factor list — the columns of `rd.F`, which a time-series regression fits one loading vector per asset against. Optional — a consumer that needs it and does not find it throws at the point of need.",#
                                 :us_utfkey => "`utfkey`: Key prefix for unique-entry time-series factor group variants in `dict`. Validated at construction, never recomputed by a view.",#
                                 :us_cfkey => "`cfkey`: Key in `dict` identifying the **cross-sectional** factor list — the exposures a cross-sectional regression fits one loading vector per observation against. Optional, and validated exactly as `tfkey` is; the two axes are never validated against each other, so a problem may declare one, both, or neither.",#
                                 :us_ucfkey => "`ucfkey`: Key prefix for unique-entry cross-sectional factor group variants in `dict`. Validated at construction, never recomputed by a view.",#
                                 :us_zkey => "`zkey`: Key in `dict` identifying the declared feature axis — the node list a graded feature program writes its columns against. Optional, like the two factor keys, and it carries no prefix convention: nothing is partitioned over the feature axis, so it has no unique-entry sibling and no length rule beyond `allunique`.",#
                                 :p_phylo => "`p`: Non-negative penalty factor on the trace of the semidefinite matrix variable. It is read **only** when the model does not already minimise a variance: a variance objective is itself a trace against that variable, so it pulls the relaxation down on its own and no second term is added. What `p` buys is how closely the matrix variable tracks the outer product it stands for, and nothing else. On a 250×6 sample the largest entry of that difference is `2.7e-5` under a variance objective, where the objective itself closes the gap, against `0.0274` under conditional value at risk at the default `p = 0.05` — three orders of magnitude wider. The relatedness rows hold either way, to `2.1e-18` and to `3.1e-21` on that sample.",#
                                 :A_phylo => "`A`: Symmetric relatedness matrix with a zero diagonal. A network source gives the range connection matrix, a clustering source the adjacency label matrix. Stored as given.",#
                                 :A_iphylo => "`A`: Row set of the relatedness matrix, stored as `unique(A + I; dims = 1)` and **not** as the matrix passed in. The identity puts each asset in its own row, and the deduplication drops rows that repeat. One row per distinct neighbourhood or cluster survives, which is why the stored matrix is usually shorter than it is wide.",#
                                 :B_phylo => "`B`: Right-hand side of `A * z <= B`, where `z` is the held indicator: the largest number of assets that may be held out of each row of `A`. A scalar applies to every row. A vector states one bound per row, so its length must match the row count of the stored `A` and not the number of assets. On an estimator the rows do not exist yet, so a vector is only checked against the largest number of clusters the clustering estimator can return.",#
                                 :cc_A => "`A`: Centrality estimator. Its centrality vector is the row of the generated linear constraint.",#
                                 :cc_B => "`B`: Right-hand side of the constraint. A number is the threshold itself. A [`VectorToScalarMeasure`](@ref) derives the threshold from the centrality vector `A` produces, so the constraint always has a feasible point. The measure reads that vector, never the row after `comp` has flipped its sign, so `MinValue()` gives the smallest entry under `<=` and under `>=` alike. On an eight-asset degree vector both give `0.14285714285714285`, and `MaxValue()` gives `0.42857142857142855`.",#
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
                                 # Cross-sectional transforms.
                                 :min_group_size => "`min_group_size`: Smallest estimation set a group may carry and still be estimated from. A group below it, and every asset that carries no group, takes the whole observation's statistics instead.",#
                                 :atol_cs => "`atol`: Absolute tolerance below which a cross-sectional scale counts as zero. An observation at or below it carries no dispersion, so its finite cells score zero rather than dividing by that scale.",#
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
                                 # Preselection. `22_Preselection.jl` pairs two redundancy
                                 # algorithms over one correlation matrix, and two selection
                                 # rules over one taken set, so each of the five descriptions
                                 # below is shared by two fields of that file.
                                 :pre_ce_corr => "`ce`: Covariance estimator supplying the correlation matrix.",#
                                 :pre_t_corr => "`t`: Correlation at or above which two assets are redundant.",#
                                 :pre_absolute => "`absolute`: Whether to compare the absolute value of the correlation.",#
                                 :pre_action => "`action`: `:keep` retains the taken assets, `:drop` retains everything else.",#
                                 :pre_measure => "`measure`: Reducer producing the fallback drop score from each column of the correlation matrix; ignored when the selector carries a `score`.",#
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
                                 :ps_cvar => "`cvar`: Conditional Value-at-Risk at `alpha`, in return space, so a tail loss is negative.",#
                                 :pf_M => "`M`: Running second-moment accumulator of the observations folded into the state, about `mu`.")
"""
    field_dict

Derived dictionary mapping argument keys to field description strings, used for `\$(FIELDS)`-style docstring interpolation.

Each entry is derived from [`arg_dict`](@ref) by stripping the leading parameter name prefix (everything up to and including the first `:`).

# Related

  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
  - [`ret_dict`](@ref)
  - [`math_dict`](@ref)
"""
const field_dict = Dict(key => strip(val[(findfirst(":", val)[1] + 1):end])
                        for (key, val) in arg_dict)
"""
    err_name_dict

Maps high-order-moment argument keys to the domain noun used in error messages, so a
message names what the caller supplied (e.g. `cokurtosis`) rather than the bare field
symbol. The symbol itself is appended at the call site, giving messages like
``cokurtosis (`kt`) cannot be empty``.

# Related

  - [`unique_key_dict`](@ref)
  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
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

# Related

  - [`unique_key_dict`](@ref)
  - [`arg_dict`](@ref)
  - [`field_dict`](@ref)
  - [`ret_dict`](@ref)
"""
const val_dict = unique_key_dict(:val_dict,
                                 :oow => "If `w` is not `nothing`, `!isempty(w)`.",
                                 :gerbt => "`0 <= t`.",#
                                 :t => "`0 < t < 1`.",#
                                 :c1 => "`0 <= c1`.",#
                                 :c2 => "`0 <= c2`.",#
                                 :c3 => "`0 <= c3`.",#
                                 :c3c2 => "`c3 > c2`.",#
                                 :sbn => "`0 <= n`. `Inf` is permitted and `NaN` is not.",#
                                 :dims => "`dims in (1, 2)`.",#
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
                                 :relax => "The encoding is not exact: the entries below bound the quantity instead of reproducing it, and the bound is tight only under the condition stated here.")

"""
    ret_dict

Dictionary containing return value descriptions for common parameters used in `PortfolioOptimisers.jl`.

# Related

  - [`unique_key_dict`](@ref)
  - [`arg_dict`](@ref)
  - [`field_dict`](@ref)
  - [`val_dict`](@ref)
"""
const ret_dict = unique_key_dict(:ret_dict,
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
                                 :stdnum => "`sd::Number`: Standard deviation of `X`.",
                                 :varnum => "`vr::Number`: Variance of `X`.",
                                 :algw => "`alg`: New algorithm instance of the same type as the argument, with the new weights applied.",
                                 :alg => "`alg`: The original algorithm instance.")
"""
    math_dict

Dictionary of mathematical notation descriptions used for docstring interpolation throughout `PortfolioOptimisers.jl`.

Keys are symbols that identify mathematical variables or subscripts; values are LaTeX-formatted strings suitable for embedding in docstrings.

A key owns a definition, not a glyph: one glyph carries different quantities in different families, so a second quantity on the same glyph takes its own key under its own symbol.

# Related

  - [`arg_dict`](@ref)
  - [`val_dict`](@ref)
  - [`ret_dict`](@ref)
  - [`ref_dict`](@ref)
"""
const math_dict = Dict(:Xv => "``\\boldsymbol{X}``: Data vector `observations × 1`.",#
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
                       :Sigma_hat_ii => "``\\hat{\\mathbf{\\Sigma}}_{ii}``: ``i``-th diagonal entry of ``\\hat{\\mathbf{\\Sigma}}``.",#
                       :Sigma_hat_ij => "``\\hat{\\mathbf{\\Sigma}}_{ij}``: Estimated covariance between assets ``i`` and ``j``.",#
                       :c_weight_bias => "``c``: Bias correction of the weighted denominator. It is fixed by the **type** of the weights, never by the estimator: `corrected = false` gives ``c = 0`` for every type, and `corrected = true` gives ``c = 1`` for `StatsBase.FrequencyWeights`, ``c = \\sum_t w_t^2 / \\sum_t w_t`` for `StatsBase.AnalyticWeights` and ``c = \\sum_t w_t / T`` for `StatsBase.ProbabilityWeights`.",#
                       # Shrinkage of the sample expected returns.
                       :mu_hat_shrink => "``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` vector of sample expected returns, whose ``i``-th entry is ``\\hat{\\mu}_i``.",#
                       :b_shrink_tgt => "``\\boldsymbol{b}``: ``N \\times 1`` shrinkage target vector, every entry of which holds the same value.",#
                       :b_j_shrink_tgt => "``b_j``: ``j``-th entry of the shrinkage target vector.",#
                       :alpha_shrink_mu => "``\\alpha``: Shrinkage intensity, the weight the blend gives the target.",#
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
                       # Entropy pooling tail views.
                       :cvar_stat => "``\\mathrm{CVaR}_{\\alpha}(X)``: Conditional value at risk of the loss series ``\\boldsymbol{x}`` at level ``\\alpha``.",#
                       :cvar_target => "``\\bar{c}``: Target conditional value at risk of the view.",#
                       :evar_stat => "``\\mathrm{EVaR}_{\\alpha}(X)``: Entropic value at risk of the loss series ``\\boldsymbol{x}`` at level ``\\alpha``.",#
                       :evar_target => "``\\bar{e}``: Target entropic value at risk of the view.",#
                       :ep_tail_nu => "``\\boldsymbol{\\nu}``: ``T \\times 1`` vector of weights that attains the risk measure, the variable of its dual representation.",#
                       # Entropy pooling.
                       :ep_prior_probs => "``\\boldsymbol{q}``: ``T \\times 1`` prior probabilities of the observations, summing to one.",#
                       :ep_post_probs => "``\\boldsymbol{p}``: ``T \\times 1`` posterior probabilities of the observations, summing to one. They are the unknown of the entropy pooling problem.",#
                       :ep_mu_prior_i => "``\\mu_{i}``: Prior mean of asset ``i``. It is a constant of the view, and a lower moment view or a fixing row holds the posterior mean at it.",#
                       :ep_sigma2_prior_i => "``\\sigma_{i}^{2}``: Prior variance of asset ``i``. It is a constant of the view, and a lower moment view or a fixing row holds the posterior variance at it.",#
                       :ep_sc1 => "``s_{c1}``: Constraint scale of the entropy pooling optimiser. It multiplies both sides of a row, so a positive value leaves the feasible set unchanged.",#
                       :ep_sc2 => "``s_{c2}``: Slack penalty of the fixed equality rows. It weights the norm of the slack in the objective, so a larger value holds those rows tighter.",#
                       :ep_so => "``s_{o}``: Objective scale of the entropy pooling optimiser. It multiplies the objective, so a positive value leaves the argument of the optimum unchanged.",#
                       # Absolute drawdown series.
                       :ct => "``c_t``: Cumulative simple portfolio return at period ``t``.",#
                       :dtdd => "``d_t \\leq 0``: Absolute drawdown at period ``t``.",#
                       # Relative drawdown series.
                       :Ct => "``C_t``: Compound wealth process at period ``t``.",#
                       :rdt => "``rd_t \\leq 0``: Relative drawdown at period ``t``.",#
                       # JuMP optimisation variables.
                       :k_budget => "``k``: Budget scaling / homogenisation variable.",#
                       :sc_scale => "``s_c``: Constraint scale. It multiplies both sides of a row, so a positive value leaves the feasible set unchanged.",#
                       :so_scale => "``s_o``: Objective scale. It multiplies the objective, so a positive value leaves the argument of the optimum unchanged.",#
                       :mu_er => "``\\boldsymbol{\\mu}``: Expected returns vector ``N \\times 1``.",#
                       :R_w => "``R(\\boldsymbol{w})``: Portfolio risk.",#
                       # Second-moment formulations.
                       :d_secmom => "``\\boldsymbol{d}``: Deviation vector ``T \\times 1`` that the formulation squares. The risk measure supplies it.",#
                       :c_secmom => "``c``: Correction factor that the risk measure supplies. It is ``1`` when the co-moment matrix already carries it.",#
                       :t_secmom => "``t``: Auxiliary model variable that the cone bounds.",#
                       # Weight finalisation.
                       :w_0_finaliser => "``\\boldsymbol{w}_{0}``: Portfolio weights vector ``N \\times 1`` that the optimisation produced, which the finaliser repairs.",#
                       :lb_ub_finaliser => "``\\boldsymbol{l}``, ``\\boldsymbol{u}``: Lower and upper weight bounds. An absent bound is dropped from the programme rather than set to an infinity.",#
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
                       :te_l2 => "``\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b})``: L2-norm error.",#
                       :te_l2sq => "``\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b})``: Squared L2-norm error.",#
                       :te_l1 => "``\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b})``: L1-norm error.",#
                       :te_lp => "``\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b})``: Lp-norm error.",#
                       :te_linf => "``\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b})``: L∞-norm error, the largest absolute deviation.",#
                       # The Range convention (ADR 0057).
                       :negated_upper_tail => "The upper tail is the base measure applied to the negated returns ``-\\boldsymbol{x}``, so both tails are reported on the same sign convention and the range is their sum, not their difference.",#
                       # The Gerber family. `05_GerberCovariance.jl` states the statistic,
                       # and `06_SmythBrobyCovariance.jl` and `35_GerberIQCovariance.jl`
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
                       # The Smyth-Broby family. `06_SmythBrobyCovariance.jl` states the
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
                       # The higher comoments. `19_Coskewness.jl` and `20_Cokurtosis.jl`
                       # build both matrices from one deviation matrix and one pairwise
                       # expansion of it, so the two files share these four symbols.
                       :Y_dev => "``\\mathbf{Y}``: ``T \\times N`` deviation matrix. `FullMoment` takes the centred returns, and `SemiMoment` clips every positive entry of them to zero.",#
                       :y_t_dev => "``\\boldsymbol{y}_t``: ``N \\times 1`` deviation vector of observation ``t``, the ``t``-th row of ``\\mathbf{Y}``. Its ``i``-th entry is ``y_{t,\\,i}``.",#
                       :Z_pairprod => "``\\mathbf{Z}``: ``T \\times N^{2}`` pairwise expansion of ``\\mathbf{Y}``, whose ``t``-th row is ``\\mathbf{Z}_{t,\\cdot}`` and whose entry ``\\mathbf{Z}_{t,\\,(i-1)N+j}`` is the product ``y_{t,\\,i} \\, y_{t,\\,j}``.",#
                       :w_obs_vec => "``\\boldsymbol{w}``: ``T \\times 1`` observation weights vector.",#
                       # Separation decay. The four members of
                       # `AbstractSeparationDecayAlgorithm` each state a closed form over the
                       # same separation, so `01_Base_Phylogeny.jl` shares this symbol
                       # between four Units.
                       :d_sep => "``d``: Separation between two assets.",#
                       # Network centrality. The eight members of
                       # `AbstractCentralityAlgorithm` each state a closed form over the
                       # same network, so `14_Centrality.jl` shares these symbols between
                       # eight Units.
                       :A_network => "``\\mathbf{A}``: Adjacency matrix of the network. It is binary on the unweighted route, and carries the edge weights of its own branch where the algorithm declares a polarity.",#
                       :n_network => "``n``: Number of assets, which is the number of vertices of the network.",#
                       :lambda_max_network => "``\\lambda_{\\mathrm{max}}``: Largest eigenvalue of ``\\mathbf{A}``.",#
                       :ell_ij_path => "``\\ell_{i,\\,j}``: Length of a shortest path between assets ``i`` and ``j``. It counts the edges on an unweighted network, and sums the edge weights on a weighted one.",#
                       :sigma_st_paths => "``\\sigma_{s,\\,t}``: Number of shortest paths between assets ``s`` and ``t``.",#
                       :sigma_st_i_paths => "``\\sigma_{s,\\,t}(i)``: Number of the shortest paths between assets ``s`` and ``t`` that pass through asset ``i``.",#
                       # Optimal number of clusters. The two members of
                       # `AbstractOptimalNumberClustersAlgorithm` each maximise a score over
                       # the same candidate counts, so `02_Clusters.jl` shares this symbol
                       # between two Units.
                       :c_star_clusters => "``c^{\\star}``: Selected number of clusters.",#
                       # Preselection. `22_Preselection.jl` states the admitted set of every
                       # selection rule and of every redundancy algorithm, so the four
                       # symbols below are each shared by two or more Units of that file.
                       :s_i_score => "``s_{i}``: Score of asset ``i``, the risk measure evaluated on that asset's own return series.",#
                       :K_keep_set => "``\\mathcal{K}``: Set of the assets a selector keeps.",#
                       :k_tail_count => "``k``: Number of assets taken from one end of the score ordering.",#
                       :t_corr_threshold => "``t``: Correlation at or above which two assets are redundant.",#
                       # The ambiguity radius rules of `06_CalibrationRules.jl`, and the
                       # effective sample size the significance rules share with them. Each
                       # rule returns one radius off one record, so the radius, its scale
                       # and the weighted count of the record are each stated by two or
                       # more Units of that file.
                       :cal_r_radius => "``r``: Ambiguity radius.",#
                       :cal_s_radius => "``s``: Scale of the radius, in the units of the series the slot owner prices.",#
                       :cal_s_i_series => "``\\hat{s}_{i}``: Sample dispersion of the series the slot owner prices, over column ``i``. It is ``\\sqrt{\\hat{\\mathbf{\\Sigma}}_{ii}}`` under a [`ReturnsSeries`](@ref), and the dispersion of column ``i`` of the drawdown sample under a drawdown marker.",#
                       :cal_T_e => "``T_{e}``: Effective sample size, which is Kish's when the observation weights are stated.")
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
                                 :chan1983 => "[chan1983](@cite) T. F. Chan, G. H. Golub and R. J. LeVeque. *Algorithms for computing the sample variance: Analysis and recommendations*. The American Statistician 37, 242–247 (1983).",#
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
                                 :EPRLVaR => "[EPRLVaR](@cite) D. Cajas. *Entropy Pooling with Relativistic Value at Risk Views*. Available at SSRN 7329718 (2026).",#
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
                                 :mcfadden1974 => "[mcfadden1974](@cite) D. McFadden. *Conditional logit analysis of qualitative choice behavior*. In: *Frontiers in Econometrics*, edited by P. Zarembka (Academic Press, 1974); pp. 105–142.",#
                                 :coxsnell1989 => "[coxsnell1989](@cite) D. R. Cox and E. J. Snell. *Analysis of Binary Data*. 2 Edition (Chapman and Hall, 1989).",#
                                 :nagelkerke1991 => "[nagelkerke1991](@cite) N. J. Nagelkerke. *A note on a general definition of the coefficient of determination*. Biometrika 78, 691–692 (1991).",#
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
