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
                 # Fees.
                 :feese => "`fees`: Fees estimator.",
                 :feesr => "`fees`: Fees result.",
                 :feeser => "`fees`: Fees estimator or result.")

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
                      :V => "`V`: Sum of the negative spectral slices of the cokurtosis matrix `features × features`.",
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
                      :miprb_flag_arg => "`miprb_flag::Bool`: Whether to add MIP rebalancing constraints.",
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
                      :window => "`window`: Observation window.")
"""
    field_dict

Derived dictionary mapping argument keys to field description strings, used for `\$(FIELDS)`-style docstring interpolation.

Each entry is derived from [`arg_dict`](@ref) by stripping the leading parameter name prefix (everything up to and including the first `:`).
"""
const field_dict = Dict(key => strip(val[(findfirst(":", val)[1] + 1):end])
                        for (key, val) in arg_dict)
"""
    val_dict = Dict(:oow => "If `w` is not `nothing`, `!isempty(w)`.")

Validation rules for certain arg_dict terms used in the documentation of `PortfolioOptimisers.jl`.
"""
val_dict = Dict(:oow => "If `w` is not `nothing`, `!isempty(w)`.",
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
                :S_D => "size(S) == size(D)`.",#
                :max_k => "If `max_k` is not `nothing`, `max_k >= 1`.",#
                :kalg => "If `alg` is not `nothing`, `alg >= 1`.",#
                :dbhtpower => "`power > 0`.",#
                :dbhtcoef => "`coef > 0`.", :Xe => "`!isempty(X)`.",#
                :phX_Xv => "`If `X` is a `MatNum`:\n    + Must be symmetric, `LinearAlgebra.issymmetric(X)`\n    + Must have zero diagonal, `all(iszero, LinearAlgebra.diag(X))`.",#
                :ntn => "`n >= 1`.",#
                :A => "`!isempty(A)`.",#
                :B => "`!isempty(B)`.",#
                :eqineq => "Both `eq` and `ineq` cannot be `nothing` at the same time, `!(isnothing(ineq) && isnothing(eq))`.")

"""
Dictionary containing return value descriptions for common parameters used in `PortfolioOptimisers.jl`.
"""
ret_dict = Dict(:mu => "`mu::ArrNum`: Expected returns vector `features x 1` if the `dims` keyword does not exist or `dims = 2`, `1 x features` if `dims = 1`.",#
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
math_dict = Dict(:Xv => "``\\boldsymbol{X}``: Data vector `observations × 1`.",#
                 :tgt => "``t``: Target value, usually the unweighted (or weighted) expected value ``E[\\boldsymbol{X}]``.",#
                 :A => "``\\mathbf{A}``: Constraint coefficient matrix.",#
                 :B => "``\\boldsymbol{B}``: Constraint response vector.",#
                 :x => "``\\boldsymbol{x}``: Constrained variable.",
                 :ineq => "``\\text{ineq}``: Subscript for inequality constraints.",#
                 :eq => "``\\text{eq}``: Subscript for equality constraints.")

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
                   throw(ArgumentError("half_life must be a positive integer"))
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
  - Displays matrix/vector fields with their size and type.
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
                fields = propertynames(obj)
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
                    if hasproperty(obj, field)
                        val = getproperty(obj, field)
                    else
                        continue
                    end
                    flag = has_pretty_show_method(val)
                    sym1 = ifelse(i == length(fields) &&
                                  (!flag || (flag && isempty(propertynames(val)))), '┴',
                                  '┼')
                    print(io, lpad(string(field), padding), " ")
                    if isnothing(val)
                        print(io, "$(sym1) nothing", '\n')
                    elseif flag || (isa(val, AbstractVector) &&
                                    length(val) <= 6 &&
                                    all(has_pretty_show_method, val))
                        ioalg = IOBuffer()
                        show(ioalg, val)
                        algstr = String(take!(ioalg))
                        alglines = split(algstr, '\n')
                        print(io, "$(sym1) ", alglines[1], '\n')
                        for l in alglines[2:end]
                            if isempty(l) || l == '\n'
                                continue
                            end
                            sym2 = '│'
                            print(io, lpad("$sym2 ", padding + 3), l, '\n')
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
has_pretty_show_method(::Any) = false
has_pretty_show_method(::JuMP.Model) = true
has_pretty_show_method(::Clustering.Hclust) = true
has_pretty_show_method(::Clustering.KmeansResult) = true
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
    "$(field_dict[:msg])"
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
    "$(field_dict[:msg])"
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
    "$(field_dict[:msg])"
    msg
end
function Base.showerror(io::IO, err::PortfolioOptimisersError)
    name = string(typeof(err))
    name = name[1:(findfirst(x -> (x == '{' || x == '('), name) - 1)]
    return print(io, "$name: $(err.msg)")
end
function Base.iterate(obj::Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                 <:AbstractResult}, state = 1)
    return state > 1 ? nothing : (obj, state + 1)
end
Base.length(::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}) = 1
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
    const Func_Num_VecNum = Union{<:Function, <:Num_VecNum}

Alias for a union of a function type or a numeric type or an abstract vector of numeric types.

# Related

  - [`Num_VecNum`](@ref)
"""
const Func_Num_VecNum = Union{<:Function, <:Num_VecNum}

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

Alias for a key type used in grid search cross-validation, which can be an abstract string, an expression, a symbol, a composed function, or an accessor lens.

# Related

  - [`PairGSCV`](@ref)
  - [`DictGSCV`](@ref)
  - [`MultiGSCVValType`](@ref)
"""
const GSCVKey = Union{<:AbstractString, Expr, Symbol, <:ComposedFunction,
                      <:Accessors.PropertyLens, <:Accessors.IndexLens}
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

```jldoctests
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
    const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr}

Alias for a union of string, Julia expression, or vector of strings/expressions.

# Related

  - [`Str_Expr`](@ref)
  - [`VecStr_Expr`](@ref)
"""
const EqnType = Union{<:AbstractString, Expr, <:VecStr_Expr}
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
"""
function assert_nonempty_nonneg_finite_val(val::AbstractDict, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($val_sym)) must hold. Got\nany(isfinite, values($val_sym)) => $(any(isfinite, values(val)))"))
    @argcheck(all(x -> zero(x) <= x, values(val)),
              DomainError("all(x -> 0 <= x, values($val_sym)) must hold. Got\nall(x -> 0 <= x, values($val_sym)) => $(all(x -> zero(x) <= x, values(val)))"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::VecPair, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($val_sym, 2)) must hold. Got\nany(isfinite, getindex.($val_sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    @argcheck(all(x -> zero(x[2]) <= x[2], val),
              DomainError("all(x -> 0 <= x[2], $val_sym) must hold. Got\nall(x -> 0 <= x[2], $val_sym) => $(all(x -> zero(x[2]) <= x[2], val))"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::ArrNum, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $val_sym) must hold. Got\nany(isfinite, $val_sym) => $(any(isfinite, val))"))
    @argcheck(all(x -> zero(x) <= x, val),
              DomainError("all(x -> 0 <= x, $val_sym) must hold. Got\nall(x -> 0 <= x, $val_sym) => $(all(x -> zero(x) <= x, val))"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::Pair, val_sym::Sym_Str = :val)
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($val_sym[2]) must hold. Got\nisfinite($val_sym[2]) => $(isfinite(val[2]))"))
    @argcheck(zero(val[2]) <= val[2],
              DomainError("0 <= $(val[2]) must hold. Got\n$(val[2]) => $(val[2])"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::Number, val_sym::Sym_Str = :val)
    @argcheck(isfinite(val),
              DomainError("isfinite($val_sym) must hold. Got\nisfinite($val_sym) => $(isfinite(val))"))
    @argcheck(zero(val) <= val, DomainError("0 <= $(val) must hold. Got\n$(val) => $(val)"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(args...)
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
"""
function assert_nonempty_gt0_finite_val(val::AbstractDict, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($val_sym)) must hold. Got\nany(isfinite, values($val_sym)) => $(any(isfinite, values(val)))"))
    @argcheck(all(x -> zero(x) < x, values(val)),
              DomainError("all(x -> 0 < x, values($val_sym)) must hold. Got\nall(x -> 0 < x, values($val_sym)) => $(all(x -> zero(x) < x, values(val)))"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::VecPair, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($val_sym, 2)) must hold. Got\nany(isfinite, getindex.($val_sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    @argcheck(all(x -> zero(x[2]) < x[2], val),
              DomainError("all(x -> 0 < x[2], $val_sym) must hold. Got\nall(x -> 0 < x[2], $val_sym) => $(all(x -> zero(x[2]) < x[2], val))"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::ArrNum, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $val_sym) must hold. Got\nany(isfinite, $val_sym) => $(any(isfinite, val))"))
    @argcheck(all(x -> zero(x) < x, val),
              DomainError("all(x -> 0 < x, $val_sym) must hold. Got\nall(x -> 0 < x, $val_sym) => $(all(x -> zero(x) < x, val))"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::Pair, val_sym::Sym_Str = :val)
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($val_sym[2]) must hold. Got\nisfinite($val_sym[2]) => $(isfinite(val[2]))"))
    @argcheck(zero(val[2]) < val[2],
              DomainError("0 < $(val[2]) must hold. Got\n$(val[2]) => $(val[2])"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::Number, val_sym::Sym_Str = :val)
    @argcheck(isfinite(val),
              DomainError("isfinite($val_sym) must hold. Got\nisfinite($val_sym) => $(isfinite(val))"))
    @argcheck(zero(val) < val, DomainError("0 < $(val) must hold. Got\n$(val) => $(val)"))
    return nothing
end
function assert_nonempty_gt0_finite_val(args...)
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
"""
function assert_nonempty_finite_val(val::AbstractDict, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($val_sym)) must hold. Got\nany(isfinite, values($val_sym)) => $(any(isfinite, values(val)))"))
    return nothing
end
function assert_nonempty_finite_val(val::VecPair, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($val_sym, 2)) must hold. Got\nany(isfinite, getindex.($val_sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    return nothing
end
function assert_nonempty_finite_val(val::ArrNum, val_sym::Sym_Str = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $val_sym) must hold. Got\nany(isfinite, $val_sym) => $(any(isfinite, val))"))
    return nothing
end
function assert_nonempty_finite_val(val::Pair, val_sym::Sym_Str = :val)
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($val_sym[2]) must hold. Got\nisfinite($val_sym[2]) => $(isfinite(val[2]))"))
    return nothing
end
function assert_nonempty_finite_val(val::Number, val_sym::Sym_Str = :val)
    @argcheck(isfinite(val),
              DomainError("isfinite($val_sym) must hold. Got\nisfinite($val_sym) => $(isfinite(val))"))
    return nothing
end
function assert_nonempty_finite_val(args...)
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
"""
function assert_matrix_issquare(X::MatNum, X_sym::Symbol = :X)
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
    "Vector component."
    v
    "Scalar component."
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

Singleton vector type that represents a vector with a single element with value equal to 1. Used for reducing matrix vector products to dropping the matrix's second dimension.

# Constructors

    SingletonVector()
"""
struct SingletonVector{T} <: AbstractVector{T} end
function SingletonVector()
    return SingletonVector{Int}()
end
Base.length(::SingletonVector) = 1
function Base.getindex(A::SingletonVector, i::Int)
    return isone(i) ? 1 : throw(BoundsError(A, i))
end
Base.:*(M::Matrix, ::SingletonVector) = dropdims(M; dims = 2)
Base.size(::SingletonVector) = (1,)

export IsEmptyError, IsNothingError, IsNonFiniteError, VecScalar
