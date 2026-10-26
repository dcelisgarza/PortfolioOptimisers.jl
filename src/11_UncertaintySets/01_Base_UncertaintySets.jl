"""
$(DocStringExtensions.TYPEDEF)

Fits an uncertainty set around a prior statistic, so that a robust model can take the worst case over the set.

Every concrete estimator of an uncertainty set subtypes `AbstractUncertaintySetEstimator`.

# Interfaces

To add an estimator that the library can use, subtype `AbstractUncertaintySetEstimator` and add the methods below.

## `mu_ucs`

  - `mu_ucs(ue::AbstractUncertaintySetEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...) -> AbstractUncertaintySetResult`: Fits the uncertainty set of the mean.

## `sigma_ucs`

  - `sigma_ucs(ue::AbstractUncertaintySetEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...) -> AbstractUncertaintySetResult`: Fits the uncertainty set of the covariance. An estimator with no covariance set throws an error here, as [`CharacteristicUncertaintySet`](@ref) does.

## `ucs`

  - `ucs(ue::AbstractUncertaintySetEstimator, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...) -> Tuple`: Fits the two sets in one call, so that the estimator computes a shared prior or a shared simulation once.

### Arguments

  - `ue`: The instance of the concrete subtype.
  - `X`: Matrix of asset returns.
  - `F`: Optional matrix of factor returns, which a factor prior needs.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method gives to the prior estimator.

### Returns

  - `ucs::AbstractUncertaintySetResult`: The fitted set. `ucs` returns a tuple of the mean set and the covariance set.

## `reads_prior_result`

  - `reads_prior_result(ue::AbstractUncertaintySetEstimator) -> Bool`: States whether the estimator calibrates its set on a Prior Result that the caller gives it, and not on returns data. The root returns `false`, so a subtype that fits from returns data declares no method. A subtype that reads a Prior Result subtypes [`AbstractPriorUncertaintySetEstimator`](@ref) instead. Each of the four returns-data families reads a Prior Result when it holds `pe = nothing`.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`reads_prior_result`](@ref)
  - [`DeltaUncertaintySet`](@ref)
  - [`NormalUncertaintySet`](@ref)
"""
abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Fits an uncertainty set from the Prior Result that the optimisation solves on, and not from returns data.

Every other [`AbstractUncertaintySetEstimator`](@ref) holds its own prior estimator `pe` and fits it on the returns that the caller gives it. So the set that it builds does not know the prior of the optimiser. A subtype of this root reads the fitted Prior Result instead. The inputs of its fit are the factor model, the `mu` and the `sigma` of that result, and it holds no `pe`. Both JuMP builders hold the prior beside the returns, so both give it to the set. The three-argument form of the ucs Triple then gives each estimator the argument that it reads.

The root is not exported. A caller writes its name only to declare a subtype, and every consumer gets to it through the ucs Triple.

# Interfaces

To add an estimator that the library can use, subtype `AbstractPriorUncertaintySetEstimator` and add the methods below.

## `mu_ucs`

  - `mu_ucs(ue::AbstractPriorUncertaintySetEstimator, pr::AbstractPriorResult; kwargs...) -> AbstractUncertaintySetResult`: Fits the uncertainty set of the mean.

## `sigma_ucs`

  - `sigma_ucs(ue::AbstractPriorUncertaintySetEstimator, pr::AbstractPriorResult; kwargs...) -> AbstractUncertaintySetResult`: Fits the uncertainty set of the covariance.

## `ucs`

  - `ucs(ue::AbstractPriorUncertaintySetEstimator, pr::AbstractPriorResult; kwargs...) -> Tuple`: Fits the two sets in one call, so that the estimator computes a shared geometry once.

### Arguments

  - `ue`: The instance of the concrete subtype.
  - `pr`: The Prior Result that the optimisation solves on.
  - `kwargs...`: More keyword arguments.

### Returns

  - `ucs::AbstractUncertaintySetResult`: The fitted set. `ucs` returns a tuple of the mean set and the covariance set.

The root returns `true` from [`reads_prior_result`](@ref). So a subtype gets the three-argument methods of the ucs Triple, which drop the returns data and call the two-argument methods above. A subtype needs no method of the returns-data interface of [`AbstractUncertaintySetEstimator`](@ref), because no consumer calls that interface through this root. Each of the four returns-data families has the same contract with no subtype when its `pe` is `nothing`. [`DeltaUncertaintySet`](@ref), [`NormalUncertaintySet`](@ref), [`ARCHUncertaintySet`](@ref) and [`CharacteristicUncertaintySet`](@ref) each have a method of the ucs Triple over a Prior Result. The optimiser gives such a set the prior that it solves on.

The root has no default fit. It has one method of each of the three verbs, and each method throws an error that names the type of its argument. So a subtype that declares none of the three tells its author which method is missing, and does not fail on the root.

# Related

  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
abstract type AbstractPriorUncertaintySetEstimator <: AbstractUncertaintySetEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Selects the shape of the uncertainty set that an estimator builds, for example a box or an ellipsoid.

Every concrete shape subtypes `AbstractUncertaintySetAlgorithm`. A subtype holds the parameters of its shape and nothing more. The estimator does the fit.

# Interfaces

A subtype is a tag that the estimator dispatches on, so it declares no method. To add a shape, subtype `AbstractUncertaintySetAlgorithm`. Then add the `ucs`, `mu_ucs` and `sigma_ucs` methods of [`AbstractUncertaintySetEstimator`](@ref) for the new shape, one set of methods for each estimator that builds it.

# Related

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
"""
abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Holds a fitted uncertainty set, which a robust model reads to build its worst-case expression.

Every concrete set subtypes `AbstractUncertaintySetResult`. A subtype also holds the statistic that the estimator calibrated its bounds on, so that the consumer bounds that statistic and not a different one.

# Interfaces

To add a set that the library can use, subtype `AbstractUncertaintySetResult` and add the method below.

## `port_opt_view`

  - `port_opt_view(risk_ucs::AbstractUncertaintySetResult, i, args...) -> AbstractUncertaintySetResult`: Returns the set restricted to the asset indices `i`. A hierarchical optimiser calls it once for each cluster.

### Arguments

  - `risk_ucs`: The instance of the concrete subtype.
  - `i`: Asset index of the cluster.
  - `args...`: More arguments.

### Returns

  - `risk_ucs::AbstractUncertaintySetResult`: The restricted set.

A model that takes the worst case over the new shape also needs its own `set_ucs_return_constraints!` method, its own `set_ucs_variance_risk!` method, or both.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
"""
abstract type AbstractUncertaintySetResult <: AbstractResult end
"""
    const UcSE_UcS = Union{AbstractUncertaintySetResult, AbstractUncertaintySetEstimator}

Alias for a union of the result types and the estimator types of an uncertainty set.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
"""
const UcSE_UcS = Union{AbstractUncertaintySetResult, AbstractUncertaintySetEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Computes the radius `k` of an ellipsoidal uncertainty set, which bounds the distance from the estimate to the true statistic.

Every concrete radius algorithm subtypes `AbstractUncertaintyKAlgorithm`. A plain number in place of an algorithm is the radius.

# Interfaces

To add a radius algorithm that the library can use, subtype `AbstractUncertaintyKAlgorithm` and add the methods below. An ellipsoid reads `k_ucs`, and a norm ball reads `k_norm_ball`.

## `k_ucs`

  - `k_ucs(km::AbstractUncertaintyKAlgorithm, q::Number, X, sigma_X::MatNum, df::Integer) -> Number`: Returns the radius.

### Arguments

  - `km`: The instance of the concrete subtype.
  - `q`: Significance level.
  - `X`: Matrix of sampled estimation errors, with one row for each sample. An algorithm that runs no simulation ignores it.
  - `sigma_X`: Shape matrix of the ellipsoid.
  - `df`: Dimension of the set, which `ucs_dimension` gives. On a full covariance shape, it is less than the first dimension of `sigma_X`, because a symmetric matrix has ``N(N+1)/2`` free entries of its ``N^{2}``. An algorithm that reads no dimension ignores it.

### Returns

  - `k::Number`: Radius of the ellipsoid.

## `k_norm_ball`

  - `k_norm_ball(km::AbstractUncertaintyKAlgorithm, q::Number, X, L::MatNum, df::Integer) -> Number`: Returns the radius of a norm ball.

### Arguments

  - `km`, `q`, `X`: The same as for `k_ucs`.
  - `L`: Geometry map of the ball, in place of the shape. The shape is `L * L'`.
  - `df`: Dimension of the ball. For a map that is a factor of a shape, the caller reads `ucs_dimension`. For a map of deviations, the caller reads `LinearAlgebra.rank(L)`. An algorithm that reads no dimension ignores it.

### Returns

  - `k::Number`: Radius of the ball.

# Related

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`k_norm_ball`](@ref)
"""
abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end
"""
    const Num_UcSK = Union{AbstractUncertaintyKAlgorithm, Number}

Alias for a union of the radius algorithms of an ellipsoid and the number types. A number is the radius.

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
"""
const Num_UcSK = Union{AbstractUncertaintyKAlgorithm, Number}
"""
$(DocStringExtensions.TYPEDEF)

Computes the radius ``\\kappa`` of a compact covariance uncertainty set from the Prior Result and the geometry of the set.

The radius of a [`CompactCovarianceUncertaintySet`](@ref) then changes when the sample or the factor span changes. A member goes in the `kappa` field of an [`OrthogonalUncertaintySet`](@ref), whose type bound is [`Num_CptRad`](@ref). A plain number in that field is the radius, and [`k_compact`](@ref) returns it unchanged.

**This family is separate from the Calibration Rules, and the two do not mix.** A **Calibration Rule** reads `(pr, w, slv, ctx)` and nothing more. These four inputs cannot size the compact radius, because its units change with a different field of the same estimator. The penalty is ``\\kappa \\lVert (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal})\\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}`` with ``\\mathbf{C} = \\mathbf{W}^{-1/2}``. Under [`IdentityMetric`](@ref), ``\\kappa`` has the units of a variance. Under [`InverseIdiosyncraticVarianceMetric`](@ref), it has no units.

A rule that reads only the Prior Result must select one of the two units, and it is wrong under the other. So a rule of this family computes the radius **inside the family**, which has the metric, the loadings block and the span. The mean radius of the same estimator follows the same pattern through [`k_norm_ball`](@ref) and the `method` field. A field that holds a radius also accepts the rule that computes it.

The name of a rule is the name of the **method** that it runs, and no other part of the library uses either word. [`ResidualInflation`](@ref) increases the residual variance to a confidence bound. [`VarianceFraction`](@ref) sets the penalty to a fraction of the nominal variance at a reference portfolio.

# Interfaces

## `k_compact`

  - `k_compact(alg::AbstractCompactRadiusAlgorithm, q::Number, metric::AbstractOrthogonalityMetric, pr::AbstractPriorResult, rr::AbstractLoadingsRegressionResult, C::VecNum, Q::MatNum, rd) -> Number`: Returns the radius, which is finite and `>= 0`. `rd` is the returns data of the fit, or `nothing`. The constructor of the estimator that holds the rule states the range, so a rule needs no range check.

# Related

  - [`ResidualInflation`](@ref)
  - [`VarianceFraction`](@ref)
  - [`Num_CptRad`](@ref)
  - [`k_compact`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`AbstractCalibrationAlgorithm`](@ref): the separate family of rules, which computes a quantity whose units do not change with a different field.
"""
abstract type AbstractCompactRadiusAlgorithm <: AbstractAlgorithm end
"""
    const Num_CptRad = Union{AbstractCompactRadiusAlgorithm, Number}

Type bound of the field that holds the compact covariance radius.

The field holds the radius, or a rule of [`AbstractCompactRadiusAlgorithm`](@ref) that computes it. The union names one family only. So the constructor refuses a rule of a different family **at construction**, at the line where the caller writes it. The union does not accept a plain `Function`. A rule of this family reads seven arguments from the call site. Four of them are the metric, the loadings block, the diagonal metric and the span. A closure over the data of the caller must repeat all seven of them before it can run.

# Related

  - [`AbstractCompactRadiusAlgorithm`](@ref)
  - [`k_compact`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
"""
const Num_CptRad = Union{AbstractCompactRadiusAlgorithm, Number}
"""
$(DocStringExtensions.TYPEDEF)

Computes the radius `eps` of an ``\\ell_1`` uncertainty set, which sets how many assets the portfolio holds.

The radius bounds the distance from the estimate to the true characteristic vector. Every concrete radius algorithm of the ``\\ell_1`` family subtypes `AbstractUncertaintyEpsAlgorithm`. A plain number in place of an algorithm is the radius. The family has the same role for the ``\\ell_1`` sets as [`AbstractUncertaintyKAlgorithm`](@ref) has for the ellipsoid.

# Interfaces

To add a radius algorithm that the library can use, subtype `AbstractUncertaintyEpsAlgorithm` and add the method below.

## `l1_resolve_eps`

  - `l1_resolve_eps(method::AbstractUncertaintyEpsAlgorithm, mus::VecNum, sds::Option{<:VecNum}, paired::Bool) -> Number`: Returns the radius.

### Arguments

  - `method`: The instance of the concrete subtype.
  - `mus`: Characteristic vector, sorted in non-increasing order.
  - `sds`: Scale of each asset under the same permutation, or `nothing` when the set has no scale.
  - `paired`: Whether to read the paired ladder of the dollar-neutral problem, and not the long-only ladder.

### Returns

  - `eps::Number`: Radius of the set.

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintyKAlgorithm`](@ref)
"""
abstract type AbstractUncertaintyEpsAlgorithm <: AbstractAlgorithm end
"""
    const Num_UcSEps = Union{AbstractUncertaintyEpsAlgorithm, Number}

Alias for a union of the radius algorithms of an ``\\ell_1`` set and the number types. A number is the radius. An algorithm computes the radius from the data.

# Related

  - [`AbstractUncertaintyEpsAlgorithm`](@ref)
"""
const Num_UcSEps = Union{AbstractUncertaintyEpsAlgorithm, Number}
"""
    ucs(uc::Option{<:Tuple{Option{<:AbstractUncertaintySetResult},
                           Option{<:AbstractUncertaintySetResult}}}, args...; kwargs...)

Returns a pair of built uncertainty sets unchanged, so that a consumer can call [`ucs`](@ref) on an estimator or on a result.

The method is a passthrough. It does no procedure, so it has no `# Algorithm` section. The method that takes an [`AbstractUncertaintySetEstimator`](@ref) does the fit.

# Arguments

  - `uc`: Tuple of uncertainty sets, or `nothing`.
  - `args...`: More positional arguments, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Returns

  - `uc::Option{<:Tuple{Option{<:AbstractUncertaintySetResult}, Option{<:AbstractUncertaintySetResult}}}`: The input, unchanged.

# Related

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function ucs(uc::Option{<:Tuple{Option{<:AbstractUncertaintySetResult},
                                Option{<:AbstractUncertaintySetResult}}}, args...;
             kwargs...)
    return uc
end
"""
    mu_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)

Returns a built uncertainty set of the mean unchanged, so that a consumer can call [`mu_ucs`](@ref) on an estimator or on a result.

The method is a passthrough. It does no procedure, so it has no `# Algorithm` section. The method that takes an [`AbstractUncertaintySetEstimator`](@ref) does the fit.

# Arguments

  - `uc`: Uncertainty set of the mean, or `nothing`.
  - `args...`: More positional arguments, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Returns

  - `uc::Option{<:AbstractUncertaintySetResult}`: The input, unchanged.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function mu_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...;
                kwargs...)::Option{<:AbstractUncertaintySetResult}
    return uc
end
"""
    sigma_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)

Returns a built uncertainty set of the covariance unchanged, so that a consumer can call [`sigma_ucs`](@ref) on an estimator or on a result.

The method is a passthrough. It does no procedure, so it has no `# Algorithm` section. The method that takes an [`AbstractUncertaintySetEstimator`](@ref) does the fit.

# Arguments

  - `uc`: Uncertainty set of the covariance, or `nothing`.
  - `args...`: More positional arguments, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Returns

  - `uc::Option{<:AbstractUncertaintySetResult}`: The input, unchanged.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function sigma_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...;
                   kwargs...)::Option{<:AbstractUncertaintySetResult}
    return uc
end
"""
    ucs_selector(risk_ucs::Nothing, prior_ucs::Nothing)
    ucs_selector(risk_ucs::UcSE_UcS, prior_ucs::Any)
    ucs_selector(risk_ucs::Nothing, prior_ucs::UcSE_UcS)

Chooses between the uncertainty set of a risk measure and the uncertainty set of a prior, and the set of the risk measure has priority.

The function is a selector, so it states the table below and has no `# Algorithm` section. Each of its three methods returns one of its arguments and does no step. The three methods cover every pair of arguments that the callers make. Only the first row gives `nothing`.

| `risk_ucs`   | `prior_ucs`  | Result      |
|:------------ |:------------ |:----------- |
| `nothing`    | `nothing`    | `nothing`   |
| a `UcSE_UcS` | anything     | `risk_ucs`  |
| `nothing`    | a `UcSE_UcS` | `prior_ucs` |

# Arguments

  - `risk_ucs`: Uncertainty set estimator or result of the risk measure, or `nothing`.
  - `prior_ucs`: Uncertainty set estimator or result of the prior, or `nothing`.

# Returns

  - `ucs::Option{<:UcSE_UcS}`: The selected set or estimator, as the table above states.

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`factory`](@ref)
"""
function ucs_selector(::Nothing, ::Nothing)::Nothing
    return nothing
end
function ucs_selector(risk_ucs::UcSE_UcS, ::Any)::UcSE_UcS
    return risk_ucs
end
function ucs_selector(::Nothing, prior_ucs::UcSE_UcS)::UcSE_UcS
    return prior_ucs
end
"""
    port_opt_view(risk_ucs::Option{<:AbstractUncertaintySetEstimator}, i, args...)

Returns an uncertainty set estimator unchanged, because an estimator has no asset axis to restrict.

The method is a passthrough, so it has no `# Algorithm` section. A hierarchical optimiser calls [`port_opt_view`](@ref) once for each cluster. Later, the optimiser fits the estimator on the returns of the cluster. So the fit does the restriction, and this method does not. The methods that take a built result apply the index, and each one states its steps.

# Arguments

  - `risk_ucs`: Uncertainty set estimator, or `nothing`.
  - `i`: Index of the cluster or of the assets, which the method ignores.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `risk_ucs::Option{<:AbstractUncertaintySetEstimator}`: The input, unchanged.

# Related

  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function port_opt_view(risk_ucs::Option{<:AbstractUncertaintySetEstimator}, ::Any,
                       args...)::Option{<:AbstractUncertaintySetEstimator}
    return risk_ucs
end
"""
    investable_ucs_reduction(pr::AbstractPriorResult, rd) -> (imsk, pr, rd)
    investable_ucs_reduction(::Nothing, pr::AbstractPriorResult, rd) -> (nothing, pr, rd)
    investable_ucs_reduction(imsk::BitVector, pr::AbstractPriorResult, rd) -> (imsk, pr_view, rd_view)

Reduces a Prior Result, and the returns data beside it, to the Investable Mask before an estimator fits a set on them outside an optimiser.

A Prior Result covers the full asset universe. An asset outside its Investable Mask has `NaN` in its moments and in every block that a factor fit wrote. So a set fitted on the whole result gets a `NaN` where its arithmetic needs a number. Every optimisation family reduces the result once at its entry, so inside an optimiser the result is already reduced and the mask is `nothing`. Outside an optimiser the result is whole. This function then takes the same view as the optimiser, and the two routes fit the same set.

[`expand_investable_ucs`](@ref) writes the fitted set back onto the full universe. Every Prior Result follows the same pattern of reduction and expansion.

The function has three methods, and dispatch selects the branch, as in `investable_reduction`. The first method finds the mask. The `nothing` method is the path where all assets are investable, and it returns its arguments unchanged. The `BitVector` method takes the two views.

# Algorithm

 1. Find the Investable Mask of the Prior Result with [`investable_mask`](@ref).
 2. If the mask is `nothing`, return `nothing` and the two arguments unchanged.
 3. Else, take a [`port_opt_view`](@ref) of the Prior Result at `findall(imsk)`. Take the same view of the returns data when it is not `nothing`. Return the two views and the mask.

# Arguments

  - $(arg_dict[:pr])
  - `rd`: Returns data beside which the estimator fits the set, or `nothing`.

# Returns

  - `(imsk, pr, rd)`: The Investable Mask and the two arguments reduced to it, or `nothing` and the two arguments unchanged.

# Related

  - [`expand_investable_ucs`](@ref)
  - [`investable_mask`](@ref)
  - [`port_opt_view`](@ref)
"""
function investable_ucs_reduction(pr::AbstractPriorResult, rd)
    return investable_ucs_reduction(investable_mask(pr), pr, rd)
end
function investable_ucs_reduction(::Nothing, pr::AbstractPriorResult, rd)
    return nothing, pr, rd
end
function investable_ucs_reduction(imsk::BitVector, pr::AbstractPriorResult, rd)
    idx = findall(imsk)
    return imsk, port_opt_view(pr, idx), isnothing(rd) ? nothing : port_opt_view(rd, idx)
end
"""
    expand_investable_ucs(set::AbstractUncertaintySetResult, ::Nothing, pr::AbstractPriorResult) -> set

Writes a set fitted on the Investable Mask back onto the full asset universe.

The `nothing` method is the path of [`investable_ucs_reduction`](@ref) where all assets are investable. It returns the set unchanged, so a set that an optimiser fits on a reduced result costs nothing here. Each built set that a route through a Prior Result makes adds its own `BitVector` method beside its [`port_opt_view`](@ref). The expansion is the inverse of that view. It writes the blocks of the set on the asset axis into a frame of the full width. It sets `val` to the full `pr.mu` or `pr.sigma`, with its `NaN` entries.

So `port_opt_view(expand_investable_ucs(set, imsk, pr), findall(imsk))` gives back `set`. This lets a caller fit a set outside an optimiser on a point-in-time prior, and give it to an optimiser on the full universe. The frame is `NaN` for a box and zero for an ellipsoid, and each method states the reason.

The method is a passthrough, so it has no `# Algorithm` section.

# Arguments

  - `set`: Fitted uncertainty set.
  - `imsk`: The Investable Mask, or `nothing`.
  - $(arg_dict[:pr])

# Returns

  - `set::AbstractUncertaintySetResult`: The set on the full asset universe.

# Related

  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::AbstractUncertaintySetResult, ::Nothing,
                               ::AbstractPriorResult)
    return set
end
"""
    ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; dims::Int = 1)

Fits the two uncertainty sets in one call from an estimator and a [`ReturnsResult`](@ref).

The method reads the fields of the container and calls the matrix method. The matrix method fits the two axes together, and an estimator that shares a prior or a simulation between them computes it once. The only keyword is `dims`, and the method ignores it, as [`prior`](@ref) does on a `ReturnsResult`. The container holds every input of the fit, and the prior of the set states the source of the returns. So an unknown keyword causes a `MethodError`, and the method does not ignore it.

# Algorithm

 1. Check that `rd.X` is not `nothing`. If it is, throw an error.
 2. If [`needs_factor_returns`](@ref) returns `true` for `uc.pe`, check that `rd.F` is not `nothing`. If it is, throw an error. A factor leaf at any position in the estimator tree reads the factor returns, and no other prior reads them.
 3. Read the source `x_src` that `uc.pe` states, with [`returns_source`](@ref). The set holds no source of its own. It gives the returns to its prior, and the prior states the source. A set with `pe = nothing` reads `rd.X`.
 4. Remove the observations of a Shared Series gap with [`shared_series_view`](@ref). These are the gaps of a vector benchmark under `x_src = ExcessSource()`, and the gaps of the factor returns when [`reads_factor_returns`](@ref) returns `true` for `uc.pe`.
 5. Read the asset returns `X` with [`benchmark_relative_returns`](@ref) under `x_src`. `X` is `rd.X`, or `rd.X - rd.B` under `ExcessSource()`.
 6. Call `ucs(uc, X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = 1)`, and return the pair of fitted sets. The method gives the implied volatility fields with the returns, because a prior that reads them takes them by keyword. `dims = 1` states that a `ReturnsResult` holds its observations in the rows.

# Arguments

  - `uc`: Estimator of the two uncertainty sets.
  - `rd`: [`ReturnsResult`](@ref) that holds the returns data and its metadata.
  - $(arg_dict[:dims_rd])

# Validation

  - `!isnothing(rd.X)`, else an `IsNothingError`.
  - The rules of [`benchmark_relative_returns`](@ref).
  - If `needs_factor_returns(uc.pe) === true`, then `!isnothing(rd.F)`, else an `IsNothingError`.

# Returns

  - `uc::Tuple{AbstractUncertaintySetResult, AbstractUncertaintySetResult}`: The uncertainty sets of the mean and of the covariance.

# Related

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; dims::Int = 1)
    @argcheck(!isnothing(rd.X), IsNothingError)
    assert_factor_returns(uc.pe, rd.F)
    x_src = returns_source(uc.pe)
    rd = shared_series_view(rd, x_src, reads_factor_returns(uc.pe))
    return ucs(uc, benchmark_relative_returns(rd, x_src), rd.F; iv = rd.iv, ivpa = rd.ivpa,
               dims = 1)
end
"""
    mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; dims::Int = 1)

Fits the uncertainty set of the mean from an estimator and a [`ReturnsResult`](@ref).

The method reads the fields of the container and calls the matrix method. A caller that needs the two axes calls [`ucs`](@ref), which fits them in one call. The only keyword is `dims`, and the method ignores it, as [`ucs`](@ref) states.

# Algorithm

 1. Check that `rd.X` is not `nothing`. If it is, throw an error.
 2. If [`needs_factor_returns`](@ref) returns `true` for `uc.pe`, check that `rd.F` is not `nothing`. If it is, throw an error. A factor leaf at any position in the estimator tree reads the factor returns, and no other prior reads them.
 3. Read the source `x_src` that `uc.pe` states, with [`returns_source`](@ref). The set holds no source of its own. It gives the returns to its prior, and the prior states the source. A set with `pe = nothing` reads `rd.X`.
 4. Remove the observations of a Shared Series gap with [`shared_series_view`](@ref). These are the gaps of a vector benchmark under `x_src = ExcessSource()`, and the gaps of the factor returns when [`reads_factor_returns`](@ref) returns `true` for `uc.pe`.
 5. Read the asset returns `X` with [`benchmark_relative_returns`](@ref) under `x_src`. `X` is `rd.X`, or `rd.X - rd.B` under `ExcessSource()`.
 6. Call `mu_ucs(uc, X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = 1)`, and return the fitted set of the mean. The method gives the implied volatility fields with the returns, because a prior that reads them takes them by keyword. `dims = 1` states that a `ReturnsResult` holds its observations in the rows.

# Arguments

  - `uc`: Estimator of the uncertainty set of the mean.
  - `rd`: [`ReturnsResult`](@ref) that holds the returns data and its metadata.
  - $(arg_dict[:dims_rd])

# Validation

  - `!isnothing(rd.X)`, else an `IsNothingError`.
  - The rules of [`benchmark_relative_returns`](@ref).
  - If `needs_factor_returns(uc.pe) === true`, then `!isnothing(rd.F)`, else an `IsNothingError`.

# Returns

  - `uc::AbstractUncertaintySetResult`: The uncertainty set of the mean.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; dims::Int = 1)
    @argcheck(!isnothing(rd.X), IsNothingError)
    assert_factor_returns(uc.pe, rd.F)
    x_src = returns_source(uc.pe)
    rd = shared_series_view(rd, x_src, reads_factor_returns(uc.pe))
    return mu_ucs(uc, benchmark_relative_returns(rd, x_src), rd.F; iv = rd.iv,
                  ivpa = rd.ivpa, dims = 1)
end
"""
    sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; dims::Int = 1)

Fits the uncertainty set of the covariance from an estimator and a [`ReturnsResult`](@ref).

The method reads the fields of the container and calls the matrix method. An estimator with no covariance set throws an error in the matrix method, as [`CharacteristicUncertaintySet`](@ref) does. The only keyword is `dims`, and the method ignores it, as [`ucs`](@ref) states.

# Algorithm

 1. Check that `rd.X` is not `nothing`. If it is, throw an error.
 2. If [`needs_factor_returns`](@ref) returns `true` for `uc.pe`, check that `rd.F` is not `nothing`. If it is, throw an error. A factor leaf at any position in the estimator tree reads the factor returns, and no other prior reads them.
 3. Read the source `x_src` that `uc.pe` states, with [`returns_source`](@ref). The set holds no source of its own. It gives the returns to its prior, and the prior states the source. A set with `pe = nothing` reads `rd.X`.
 4. Remove the observations of a Shared Series gap with [`shared_series_view`](@ref). These are the gaps of a vector benchmark under `x_src = ExcessSource()`, and the gaps of the factor returns when [`reads_factor_returns`](@ref) returns `true` for `uc.pe`.
 5. Read the asset returns `X` with [`benchmark_relative_returns`](@ref) under `x_src`. `X` is `rd.X`, or `rd.X - rd.B` under `ExcessSource()`.
 6. Call `sigma_ucs(uc, X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = 1)`, and return the fitted set of the covariance. The method gives the implied volatility fields with the returns, because a prior that reads them takes them by keyword. `dims = 1` states that a `ReturnsResult` holds its observations in the rows.

# Arguments

  - `uc`: Estimator of the uncertainty set of the covariance.
  - `rd`: [`ReturnsResult`](@ref) that holds the returns data and its metadata.
  - $(arg_dict[:dims_rd])

# Validation

  - `!isnothing(rd.X)`, else an `IsNothingError`.
  - The rules of [`benchmark_relative_returns`](@ref).
  - If `needs_factor_returns(uc.pe) === true`, then `!isnothing(rd.F)`, else an `IsNothingError`.

# Returns

  - `uc::AbstractUncertaintySetResult`: The uncertainty set of the covariance.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; dims::Int = 1)
    @argcheck(!isnothing(rd.X), IsNothingError)
    assert_factor_returns(uc.pe, rd.F)
    x_src = returns_source(uc.pe)
    rd = shared_series_view(rd, x_src, reads_factor_returns(uc.pe))
    return sigma_ucs(uc, benchmark_relative_returns(rd, x_src), rd.F; iv = rd.iv,
                     ivpa = rd.ivpa, dims = 1)
end
"""
    ucs(ue::AbstractPriorUncertaintySetEstimator, ::AbstractPriorResult; kwargs...)
    mu_ucs(ue::AbstractPriorUncertaintySetEstimator, ::AbstractPriorResult; kwargs...)
    sigma_ucs(ue::AbstractPriorUncertaintySetEstimator, ::AbstractPriorResult; kwargs...)

Throws an error that names an uncertainty set estimator which reads a Prior Result and declares no fit.

[`AbstractPriorUncertaintySetEstimator`](@ref) states three verbs over a Prior Result, and each member must add a method of each verb. These three methods belong to the root, and they throw an error. The three-argument form of the ucs Triple calls them. Without them, a member that declares none of the three fails with a `MethodError` that names the root, not the type that the author wrote.

The root has no default fit. An estimator of this family reads a fitted Prior Result, and no two members read the same part of it. So a default method must invent a set, and it cannot build one. [`factor_residual_config`](@ref) uses the same rule for a declaration of each type, where a missing declaration is a defect and not an answer.

# Arguments

  - `ue`: Uncertainty set estimator, which the error names.
  - `pr`: Prior Result, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Validation

  - Throws an `ArgumentError` that names the type of `ue`.

# Related

  - [`AbstractPriorUncertaintySetEstimator`](@ref)
  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::AbstractPriorUncertaintySetEstimator, ::AbstractPriorResult; kwargs...)
    return throw(ArgumentError("`ucs` is not defined for `$(nameof(typeof(ue)))`. Every concrete `AbstractPriorUncertaintySetEstimator` must fit both sets from the prior result it is handed by adding a method of `ucs`, `mu_ucs` and `sigma_ucs` over an `AbstractPriorResult`."))
end
function mu_ucs(ue::AbstractPriorUncertaintySetEstimator, ::AbstractPriorResult; kwargs...)
    return throw(ArgumentError("`mu_ucs` is not defined for `$(nameof(typeof(ue)))`. Every concrete `AbstractPriorUncertaintySetEstimator` must fit the mean set from the prior result it is handed by adding a method of `mu_ucs` over an `AbstractPriorResult`."))
end
function sigma_ucs(ue::AbstractPriorUncertaintySetEstimator, ::AbstractPriorResult;
                   kwargs...)
    return throw(ArgumentError("`sigma_ucs` is not defined for `$(nameof(typeof(ue)))`. Every concrete `AbstractPriorUncertaintySetEstimator` must fit the covariance set from the prior result it is handed by adding a method of `sigma_ucs` over an `AbstractPriorResult`."))
end
"""
    reads_prior_result(ue::AbstractUncertaintySetEstimator) -> Bool
    reads_prior_result(::AbstractPriorUncertaintySetEstimator) -> Bool
    reads_prior_result(::DeltaUncertaintySet{Nothing}) -> Bool
    reads_prior_result(::NormalUncertaintySet{Nothing}) -> Bool
    reads_prior_result(::ARCHUncertaintySet{Nothing}) -> Bool
    reads_prior_result(::CharacteristicUncertaintySet{Nothing}) -> Bool
    reads_prior_result(::Option{<:AbstractUncertaintySetResult}) -> Bool

States whether an uncertainty set estimator calibrates on a Prior Result from the caller, and not on a prior that it fits.

The predicate reads the type and the `pe` field, and never a method table. Three consumers give an estimator the argument that it reads. These are the three-argument form of the ucs Triple, [`ucs_risk_measure`](@ref) and the uncertainty step of the Pipeline. All three ask this one question, so they agree on the answer.

Two kinds of estimator return `true`. The first kind is every [`AbstractPriorUncertaintySetEstimator`](@ref), which holds no `pe`. The second kind is each of the four returns-data families when its `pe` is `nothing`. Each of these families declares its own method beside its method over a Prior Result.

Every other estimator returns `false`, and a subtype that a caller writes gets this default. A built set, or `nothing`, also returns `false`, because it is already fitted and every verb returns it unchanged. So a caller can ask a field that holds an estimator or a result, with no test of its own.

# Arguments

  - `ue`: Uncertainty set estimator, built set, or `nothing`.

# Returns

  - `reads::Bool`: `true` when the estimator reads a Prior Result. `false` when it fits its own prior, or when it is a built set.

# Related

  - [`AbstractPriorUncertaintySetEstimator`](@ref)
  - [`ucs`](@ref)
  - [`ucs_prior`](@ref)
  - [`ucs_risk_measure`](@ref)
"""
function reads_prior_result(::AbstractUncertaintySetEstimator)::Bool
    return false
end
function reads_prior_result(::AbstractPriorUncertaintySetEstimator)::Bool
    return true
end
function reads_prior_result(::Option{<:AbstractUncertaintySetResult})::Bool
    return false
end
"""
    reader_frames(ue::AbstractUncertaintySetEstimator, path::AbstractString, refit::Bool)
    reader_frames(ue::AbstractPriorUncertaintySetEstimator, path::AbstractString, refit::Bool)

Returns the frames that the prior of an uncertainty set adds to the group of the optimiser that holds the set.

An uncertainty set holds no source of its own. It gives the returns to its `pe`, and that prior states the source. So the method calls [`reader_frames`](@ref) on `pe` at the path `"<path>.pe"`. The warning of [`warn_raw_readers`](@ref) then names the prior of a set as, for example, `"r.ucs.pe"`. A set with `pe = nothing` reads the Prior Result of the optimiser, so it adds no frames. An [`AbstractPriorUncertaintySetEstimator`](@ref) holds no prior, and a built set reads no returns, so they also add no frames.

# Arguments

  - `ue`: The uncertainty set estimator.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the prior of the set.

# Related

  - [`reader_frames`](@ref)
  - [`reads_prior_result`](@ref)
  - [`ucs`](@ref)
"""
function reader_frames(ue::AbstractUncertaintySetEstimator, path::AbstractString,
                       refit::Bool)
    return reader_frames(ue.pe, join_path(path, :pe), refit)
end
function reader_frames(::AbstractPriorUncertaintySetEstimator, ::AbstractString, ::Bool)
    return ReturnsFrames()
end
"""
    ucs_prior(pe::AbstractLowOrderPriorEstimator, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    ucs_prior(::Nothing, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)

Fits the prior that an uncertainty set calibrates on, or throws an error that names `pe` when the set holds no prior estimator.

Every returns-data verb of the four families fits its `pe` through this function. The four families are [`DeltaUncertaintySet`](@ref), [`NormalUncertaintySet`](@ref), [`ARCHUncertaintySet`](@ref) and [`CharacteristicUncertaintySet`](@ref). So the library writes the refusal once, and not once for each verb of each family.

A set whose `pe` is `nothing` calibrates on a Prior Result that the caller gives it, and `nothing` has that meaning only. The fit does not replace `nothing` with an empirical prior over `X`. That replacement makes one estimator calibrate on two different priors, and the call site selects which one. So the returns-data form throws an error, and the error names the form `ucs(ue, pr)` and the field `pe`.

# Arguments

  - `pe`: The prior estimator of the set, or `nothing`.
  - `X`: Data matrix, for example the returns.
  - `F`: Optional factor matrix, which the prior estimator reads.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method gives to the prior estimator.

# Validation

  - `pe` is not `nothing`, else an `ArgumentError` that names the form over a Prior Result.

# Returns

  - `pr::AbstractPriorResult`: The fitted prior.

# Related

  - [`reads_prior_result`](@ref)
  - [`prior`](@ref)
  - [`ucs`](@ref)
"""
function ucs_prior(pe::AbstractLowOrderPriorEstimator, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    return prior(pe, X, F; dims = dims, kwargs...)
end
function ucs_prior(::Nothing, ::MatNum, ::Option{<:MatNum} = nothing; kwargs...)
    return throw(ArgumentError("this uncertainty set holds no prior estimator of its own (`pe = nothing`), so it is calibrated on a prior result it is handed and cannot be fitted from returns data. Call `ucs(ue, pr)`, `mu_ucs(ue, pr)` or `sigma_ucs(ue, pr)` with the fitted prior, or set `pe` to fit one from the returns."))
end
"""
    ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult, pr::AbstractPriorResult)

Fits the two uncertainty sets from an estimator, the returns data **and** the Prior Result that the optimisation solves on.

Both JuMP builders call this form, so one call site serves every estimator. Each estimator reads the one argument that its fit uses, and [`reads_prior_result`](@ref) selects it. The passthrough method takes `args...` and is the third method of this form. So a field that holds a built pair also gets an answer here.

| `uc`                                                                                               | Reads   | Forwards to            |
|:-------------------------------------------------------------------------------------------------- |:------- |:---------------------- |
| an estimator with a `pe` of its own                                                                | `rd`    | `ucs(uc, rd)`          |
| an [`AbstractPriorUncertaintySetEstimator`](@ref), or a returns-data estimator with `pe = nothing` | `pr`    | `ucs(uc, pr; rd = rd)` |
| a built pair, or `nothing`                                                                         | neither | itself, unchanged      |

On the first row, the method drops the prior and does not check it. An estimator that holds its own `pe` fits it on the returns that it gets. The prior of the optimisation is not an input of that fit, and it changes no number. An estimator on the second row calibrates on `pr`. Inside an optimiser, `pr` is the prior of the optimisation. So the centre of the set and the `mu` of the objective are the same number.

# Arguments

  - `uc`: Uncertainty set estimator, built pair, or `nothing`.
  - `rd`: [`ReturnsResult`](@ref). An estimator with its own `pe` reads it under the source that its `pe` states. An estimator that reads a Prior Result drops it, and reads `pr`, which is already in the frame of the fit of the prior.
  - `pr`: [`AbstractPriorResult`](@ref). An estimator that reads a Prior Result reads it, and an estimator with its own `pe` drops it.

# Returns

  - `uc::Tuple{AbstractUncertaintySetResult, AbstractUncertaintySetResult}`: The uncertainty sets of the mean and of the covariance.

# Related

  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`reads_prior_result`](@ref)
  - [`AbstractPriorUncertaintySetEstimator`](@ref)
"""
function ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult,
             pr::AbstractPriorResult)
    return if reads_prior_result(uc)
        ucs(uc, pr; rd = rd)
    else
        ucs(uc, rd)
    end
end
"""
    mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult, pr::AbstractPriorResult)

Fits the uncertainty set of the mean from an estimator, the returns data **and** the Prior Result that the optimisation solves on.

The builder of the robust return calls this form. The table of [`ucs`](@ref) states which argument each estimator reads, and [`reads_prior_result`](@ref) selects it. The passthrough method takes `args...` and is the third method of this form. So a field that holds a built set also gets an answer here.

# Arguments

  - `uc`: Uncertainty set estimator, built set, or `nothing`.
  - `rd`: [`ReturnsResult`](@ref). An estimator with its own `pe` reads it under the source that its `pe` states. An estimator that reads a Prior Result drops it, and reads `pr`, which is already in the frame of the fit of the prior.
  - `pr`: [`AbstractPriorResult`](@ref). An estimator that reads a Prior Result reads it, and an estimator with its own `pe` drops it.

# Returns

  - `uc::AbstractUncertaintySetResult`: The uncertainty set of the mean.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`reads_prior_result`](@ref)
  - [`AbstractPriorUncertaintySetEstimator`](@ref)
"""
function mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult,
                pr::AbstractPriorResult)
    return if reads_prior_result(uc)
        mu_ucs(uc, pr; rd = rd)
    else
        mu_ucs(uc, rd)
    end
end
"""
    sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult, pr::AbstractPriorResult)

Fits the uncertainty set of the covariance from an estimator, the returns data **and** the Prior Result that the optimisation solves on.

The builder of the uncertainty set variance calls this form. The table of [`ucs`](@ref) states which argument each estimator reads, and [`reads_prior_result`](@ref) selects it. The passthrough method takes `args...` and is the third method of this form. So a field that holds a built set also gets an answer here.

# Arguments

  - `uc`: Uncertainty set estimator, built set, or `nothing`.
  - `rd`: [`ReturnsResult`](@ref). An estimator with its own `pe` reads it under the source that its `pe` states. An estimator that reads a Prior Result drops it, and reads `pr`, which is already in the frame of the fit of the prior.
  - `pr`: [`AbstractPriorResult`](@ref). An estimator that reads a Prior Result reads it, and an estimator with its own `pe` drops it.

# Returns

  - `uc::AbstractUncertaintySetResult`: The uncertainty set of the covariance.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`reads_prior_result`](@ref)
  - [`AbstractPriorUncertaintySetEstimator`](@ref)
"""
function sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult,
                   pr::AbstractPriorResult)
    return if reads_prior_result(uc)
        sigma_ucs(uc, pr; rd = rd)
    else
        sigma_ucs(uc, rd)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Selects a box uncertainty set, a convex polytope of element-wise bounds, from an estimator that can build either shape.

[`EllipsoidalUncertaintySetAlgorithm`](@ref) selects the ellipsoid. The box has no correlation between the entries that it bounds. Its worst case is a linear programme or a semidefinite programme, and not a second-order cone programme.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)

# References

  - $(ref_dict[:sousalobo2000]) Sections 3.1 and 4.
"""
struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Holds the element-wise lower and upper bounds of a box uncertainty set on a mean vector or on a covariance matrix.

A box is a convex polytope, so it is a polyhedral confidence interval on the entries that it bounds. Section 3.1 of [sousalobo2000](@cite) states the worst case on the mean axis, ``\\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w} - \\delta^{\\intercal} \\lvert \\boldsymbol{w} \\rvert`` for a box symmetric about ``\\hat{\\boldsymbol{\\mu}}``. Section 5.2.6 of the same work states the worst case on the covariance axis as Equation 7. Equation 7 is the dual of the semidefinite programme of Equation 6.

**The two axes read the bounds differently, so a set fitted for one axis is not a set for the other.** On the mean axis, [`set_ucs_return_constraints!`](@ref) reads only the half-width ``(\\boldsymbol{u} - \\boldsymbol{\\ell}) / 2`` of the bounds, which is the ``\\delta`` of the definition below. It centres that width on `val`. So neither `lb` nor `ub` is a bound on the mean alone. This is why two estimators can write one set in two ways and agree. [`ARCHUncertaintySet`](@ref) stores the two quantiles of the bootstrap mean. [`DeltaUncertaintySet`](@ref) and the normal box write ``\\boldsymbol{\\ell} = \\boldsymbol{0}`` and put the full width in ``\\boldsymbol{u}``.

On the covariance axis, [`set_ucs_variance_risk!`](@ref) reads ``\\operatorname{tr}(\\mathbf{A}_{u} \\mathbf{\\Sigma}_{u}) - \\operatorname{tr}(\\mathbf{A}_{l} \\mathbf{\\Sigma}_{l})`` under ``\\mathbf{A}_{u} - \\mathbf{A}_{l} = \\mathbf{W}``. So each bound binds alone, and the covariance box is absolute. That route has no centre, so it never reads `val`.

# Mathematical definition

```math
\\begin{align}
U^{\\text{box}}_{\\boldsymbol{\\mu}} &= \\left\\{ \\boldsymbol{\\mu}\\, \\vert\\, \\lvert \\boldsymbol{\\mu} - \\boldsymbol{\\hat{\\mu}} \\rvert \\leq \\delta \\right\\} \\\\
U^{\\text{box}}_{\\mathbf{\\Sigma}} &= \\left\\{ \\mathbf{\\Sigma}\\, \\vert\\, \\mathbf{\\Sigma}_{l} \\leq \\mathbf{\\Sigma} \\leq \\mathbf{\\Sigma}_{u},\\, \\mathbf{\\Sigma} \\succeq 0 \\right\\}\\,.
\\end{align}
```

Where:

  - ``U^{\\text{box}}_{\\boldsymbol{\\mu}}``: Box uncertainty set for expected returns.
  - ``U^{\\text{box}}_{\\mathbf{\\Sigma}}``: Box uncertainty set for the covariance matrix.
  - ``\\boldsymbol{\\mu}``, ``\\mathbf{\\Sigma}``: Uncertain expected returns and covariance.
  - ``\\boldsymbol{\\hat{\\mu}}``: Estimated mean vector, the centre of the box.
  - ``\\delta``: Element-wise half-width of the box.
  - ``\\mathbf{\\Sigma}_{l}``, ``\\mathbf{\\Sigma}_{u}``: Lower and upper bounds of the covariance matrix.
  - ``\\mathbf{\\Sigma} \\succeq 0``: Constraint that the matrix is positive semidefinite.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BoxUncertaintySet(;
        lb::ArrNum,
        ub::ArrNum,
        val::Option{<:ArrNum} = nothing
    ) -> BoxUncertaintySet

The keywords are the fields of the struct.

## Validation

  - `!isempty(lb)`.
  - `!isempty(ub)`.
  - `size(lb) == size(ub)`.
  - If `val` is an array, `size(val) == size(lb)`.

# Examples

```jldoctest
julia> BoxUncertaintySet(; lb = [0.1, 0.2], ub = [0.3, 0.4])
BoxUncertaintySet
   lb ┼ Vector{Float64}: [0.1, 0.2]
   ub ┼ Vector{Float64}: [0.3, 0.4]
  val ┴ nothing
```

# Related

  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)

# References

  - $(ref_dict[:sousalobo2000]) Sections 3.1, 4 and 5.2.6, Equations 6 and 7.
"""
@concrete struct BoxUncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:lb])
    """
    lb
    """
    $(field_dict[:ub])
    """
    ub
    """
    $(field_dict[:val_ucs])
    """
    val
    function BoxUncertaintySet(lb::ArrNum, ub::ArrNum, val::Option{<:ArrNum})
        @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
        @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
        @argcheck(size(lb) == size(ub),
                  DimensionMismatch("lb ($(size(lb))) must match ub ($(size(ub)))"))
        if isa(val, ArrNum)
            @argcheck(size(val) == size(lb),
                      DimensionMismatch("val ($(size(val))) must match lb ($(size(lb)))"))
        end
        return new{typeof(lb), typeof(ub), typeof(val)}(lb, ub, val)
    end
end
function BoxUncertaintySet(lb::ArrNum, ub::ArrNum)::BoxUncertaintySet
    return BoxUncertaintySet(lb, ub, nothing)
end
function BoxUncertaintySet(; lb::ArrNum, ub::ArrNum,
                           val::Option{<:ArrNum} = nothing)::BoxUncertaintySet
    return BoxUncertaintySet(lb, ub, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a view of a vector [`BoxUncertaintySet`](@ref) restricted to the asset indices `i`.

The method applies to the mean axis, where each bound has one entry for each asset. So the method applies the asset index to the two bounds.

# Algorithm

 1. Take `view(risk_ucs.lb, i)` and `view(risk_ucs.ub, i)`, the two bounds restricted to the selected assets.
 2. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the centre restricted to the same assets. It returns `nothing` unchanged.
 3. Build a [`BoxUncertaintySet`](@ref) from the three views. The bounds stay a pair, so the half-width that the mean route reads is the half-width of the restricted box.

# Arguments

  - `risk_ucs`: Box uncertainty set with vector bounds.
  - `i`: Index of the cluster or of the assets.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `risk_ucs::BoxUncertaintySet`: The set restricted to `i`.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::BoxUncertaintySet{<:VecNum, <:VecNum}, i,
                       args...)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i),
                             val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Writes a vector [`BoxUncertaintySet`](@ref) fitted on the Investable Mask back onto the full asset universe.

The expansion is the inverse of [`port_opt_view`](@ref). The view slices the two bounds and the centre by the asset index, and reads nothing more. So the view removes all of the frame outside the mask, and it gives back the fitted set. The frame is `NaN`, as in the moment. A bound on `mu` has the units and the positions of `mu`, and outside the mask the prior states `NaN` for `mu`. The centre is the full `pr.mu`.

# Algorithm

 1. Write `set.lb` and `set.ub` at the rows that the mask keeps into two `NaN` frames of `length(imsk)` entries, with [`expand_vector`](@ref).
 2. Build a [`BoxUncertaintySet`](@ref) from the two frames, with `pr.mu` as `val`.

# Arguments

  - `set`: Box uncertainty set with vector bounds, fitted on the reduced prior.
  - `imsk`: The Investable Mask of the full prior.
  - $(arg_dict[:pr])

# Returns

  - `set::BoxUncertaintySet`: The set on the full asset universe.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`expand_vector`](@ref)
  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::BoxUncertaintySet{<:VecNum, <:VecNum}, imsk::BitVector,
                               pr::AbstractPriorResult)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = expand_vector(set.lb, imsk),
                             ub = expand_vector(set.ub, imsk), val = pr.mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a view of a matrix [`BoxUncertaintySet`](@ref) restricted to the asset indices `i`.

The method applies to the covariance axis, where each bound is an `N × N` matrix. So the method applies the asset index to the two dimensions of each bound.

# Algorithm

 1. Take `view(risk_ucs.lb, i, i)` and `view(risk_ucs.ub, i, i)`, the two bounds restricted to the selected assets on the two axes. The two bounds stay symmetric, because the source bounds are symmetric and the method applies the same index two times.
 2. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted covariance restricted to the same assets. It returns `nothing` unchanged.
 3. Build a [`BoxUncertaintySet`](@ref) from the three views. On this axis each bound binds alone, so the method restricts each bound and does not combine them.

# Arguments

  - `risk_ucs`: Box uncertainty set with matrix bounds.
  - `i`: Index of the cluster or of the assets.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `risk_ucs::BoxUncertaintySet`: The set restricted to `i`.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::BoxUncertaintySet{<:MatNum, <:MatNum}, i,
                       args...)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i, i), ub = view(risk_ucs.ub, i, i),
                             val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Writes a matrix [`BoxUncertaintySet`](@ref) fitted on the Investable Mask back onto the full asset universe.

The expansion is the inverse of [`port_opt_view`](@ref). The view slices the two bounds on the two axes, and the centre, by the asset index. So the view removes all of the frame outside the mask, and it gives back the fitted set. The frame is `NaN`, as in the moment. A bound on `sigma` has the positions of `sigma`, and outside the mask the prior states `NaN` for its row and its column. The centre is the full `pr.sigma`.

# Algorithm

 1. Write `set.lb` and `set.ub` at `(imsk, imsk)` into two `NaN` frames of `length(imsk)` rows and columns, with the covariance method of [`expand_moment`](@ref).
 2. Build a [`BoxUncertaintySet`](@ref) from the two frames, with `pr.sigma` as `val`.

# Arguments

  - `set`: Box uncertainty set with matrix bounds, fitted on the reduced prior.
  - `imsk`: The Investable Mask of the full prior.
  - $(arg_dict[:pr])

# Returns

  - `set::BoxUncertaintySet`: The set on the full asset universe.

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`expand_moment`](@ref)
  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::BoxUncertaintySet{<:MatNum, <:MatNum}, imsk::BitVector,
                               pr::AbstractPriorResult)::BoxUncertaintySet
    return BoxUncertaintySet(; lb = expand_moment(set.lb, imsk),
                             ub = expand_moment(set.ub, imsk), val = pr.sigma)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the radius `k` of the ellipsoid from a sample, as the `1 - q` quantile of the Mahalanobis distances of the sampled estimation errors.

The algorithm makes no assumption about the distribution. It reads the errors that the estimator sampled, from a parametric draw or from a bootstrap resample. [`ChiSqKUncertaintyAlgorithm`](@ref) and [`GeneralKUncertaintyAlgorithm`](@ref) give the radius in closed form.

The sample must be the estimation **error**, not the estimate. When the errors are normal, the squared Mahalanobis distance of a centred error has a chi-squared distribution. So this algorithm and [`ChiSqKUncertaintyAlgorithm`](@ref) compute one radius in two ways, and they agree to within the noise of the sample. That radius depends only on the dimension and the significance level. A sample of the raw estimates makes the distance **non-central**. The radius then increases with the non-centrality ``T \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}``, although it must measure only the estimation error.

The algorithm measures the distances against the shape matrix that it gets, not against the first shape of the estimator. [`ellipsoidal_set`](@ref) replaces the asymptotic covariance with its diagonal **before** it calls [`k_ucs`](@ref). So under the default `diagonal = true`, the radius is a quantile of the distances against the diagonal shape. The two radii are different, and neither shape always gives the larger radius.

# Mathematical definition

```math
k = \\sqrt{Q_{1-q}\\!\\left(\\left\\{ \\boldsymbol{\\delta}^{(m)\\intercal} \\mathbf{\\Sigma}_{\\boldsymbol{\\delta}}^{-1} \\boldsymbol{\\delta}^{(m)} \\right\\}_{m=1}^{M}\\right)}\\,.
```

Where:

  - ``\\boldsymbol{\\delta}^{(m)}``: The ``m``-th sampled estimation error, a row of the `X` argument of [`k_ucs`](@ref).
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\delta}}``: Shape matrix of the ellipsoid.
  - ``Q_{1-q}``: The `1 - q` quantile function.
  - ``M``: Number of samples.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormalKUncertaintyAlgorithm(;
        kwargs::NamedTuple = (;)
    )

The keyword is the field above.

## Validation

  - `kwargs` is a `NamedTuple`.

# Examples

```jldoctest
julia> NormalKUncertaintyAlgorithm()
NormalKUncertaintyAlgorithm
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2, Listings 11.10 and 11.11. $(ref_dict[:no_original_source])
"""
@concrete struct NormalKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function NormalKUncertaintyAlgorithm(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function NormalKUncertaintyAlgorithm(;
                                     kwargs::NamedTuple = (;))::NormalKUncertaintyAlgorithm
    return NormalKUncertaintyAlgorithm(kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the radius `k` of the ellipsoid as `sqrt((1 - q) / q)`, a closed form that holds for every distribution of the estimation errors.

The radius is the factor ``\\kappa(\\epsilon)`` of Theorem 1 of [elghaoui2003](@cite), Equation 8, at ``\\epsilon = q``. It reads neither the data nor the shape matrix. The theorem bounds the worst-case value at risk over every distribution with a given mean and covariance. The radius comes from the one-sided Chebyshev inequality of Cantelli. So it holds for every distribution of the estimation errors that has the stated second moment. When the errors are normal, use [`ChiSqKUncertaintyAlgorithm`](@ref), because its radius is smaller there.

The guarantee is a bound in one direction, not a joint region for the full vector. It covers the scalar that a robust row bounds, which is the worst-case value of ``\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\mu}`` at the weights that the model selects. It holds with a probability of at least ``1 - q``. The chi-squared radius gives a joint confidence region for all entries together.

# Mathematical definition

```math
k = \\sqrt{\\dfrac{1 - q}{q}}\\,.
```

Where:

  - ``q``: Significance level.

The inverse of the form is ``\\left(1 + k^{2}\\right)^{-1} = q``, which is the bound of Cantelli at ``k`` standard deviations. So the radius is the smallest radius whose tail bound is ``q`` for every distribution with the stated moments.

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)

# References

  - $(ref_dict[:elghaoui2003]) Theorem 1, Equations 7 and 8.
"""
struct GeneralKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes the radius `k` of the ellipsoid as the square root of a chi-squared quantile, a closed form for normal estimation errors.

The degrees of freedom are the dimension of the set, which is the number of free coordinates of the estimation error. On the mean axis, the dimension is ``N``. On the covariance axis, the error is a vectorised **symmetric** ``N \\times N`` matrix. The entries ``(i, j)`` and ``(j, i)`` are one number, so the error has ``N(N+1)/2`` free coordinates of the ``N^{2}``. The shape of the normal method, ``\\left(\\mathbf{I} + \\mathbf{K}\\right) \\left(\\mathbf{\\Sigma} \\otimes \\mathbf{\\Sigma}\\right) / T``, has that rank. Theorem 3.1 (vi) of [magnus1979](@cite) gives ``\\mathbf{K}`` the eigenvalue ``+1`` with multiplicity ``N(N+1)/2``.

The squared Mahalanobis distance of a normal error has a chi-squared distribution at the rank of its shape. So a full covariance shape reads ``N(N+1)/2``. A diagonal shape reads ``N^{2}``, because its statistic is a sum of ``N^{2}`` terms with a mean of one. [`ucs_dimension`](@ref) states the rule, and the set builders give its value to [`k_ucs`](@ref) and [`k_norm_ball`](@ref).

**`ambient = true` reads the dimension of the ambient space instead.** That is the first dimension of the shape matrix or of the geometry map, which is ``N^{2}`` on every covariance shape. At ``N = 5`` and ``q = 0.05``, that radius is ``6.136``, and the dimension of the set gives ``5.000``. So the set covers ``0.999`` of the errors, not ``0.95``, and the radius is too large for its level. As ``N`` increases, the ratio of the two radii decreases to ``1 / \\sqrt{2}``. The field keeps that rule available for a comparison with a caller that counts every entry of the vectorised matrix.

**The sources state this closed form for the mean axis only.** Equation 50 of [goldfarbiyengar2003](@cite) states that the squared Mahalanobis distance of a normal estimation error has a chi-squared distribution at the dimension of the estimate. That work sizes its sets with an F quantile, because it estimates the variance. Equation 11.23 of [cajas2025](@cite) sets ``\\kappa^{2}_{\\boldsymbol{\\mu}}`` to the chi-squared quantile at ``n`` degrees of freedom, where ``n`` is the number of assets. It gets ``\\kappa^{2}_{\\mathbf{\\Sigma}}`` from a simulation, not in closed form. The form on the covariance axis is an extension that this library makes.

Section 4.1 of [sousalobo2000](@cite) bounds the ``N(N+1)/2`` distinct entries of the covariance with its ellipsoid, and the dimension of the set reads that count. [`NormalKUncertaintyAlgorithm`](@ref) on the same errors gives the radius from the sample, and the two radii agree to within the noise of the simulation.

# Mathematical definition

```math
\\begin{align}
k &= \\sqrt{\\chi^{2,\\,-1}_{p}(1 - q)}\\,.
\\end{align}
```

Where:

  - ``\\chi^{2,\\,-1}_{p}``: Inverse cumulative distribution function of the chi-squared distribution with ``p`` degrees of freedom.
  - ``p``: Degrees of freedom, the dimension of the set, or the first dimension of the shape matrix under `ambient = true`.
  - ``q``: Significance level.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ChiSqKUncertaintyAlgorithm(;
        ambient::Bool = false
    )

The keyword is the field above.

# Examples

```jldoctest
julia> ChiSqKUncertaintyAlgorithm()
ChiSqKUncertaintyAlgorithm
  ambient ┴ Bool: false
```

# Related

  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`k_norm_ball`](@ref)
  - [`ucs_dimension`](@ref)

# References

  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equations 50 and 54.
  - $(ref_dict[:sousalobo2000]) Section 4.1.
  - $(ref_dict[:magnus1979]) Theorem 3.1 (vi).
  - $(ref_dict[:cajas2025]) Equation 11.23. $(ref_dict[:no_original_source])
"""
@concrete struct ChiSqKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm
    """
    Whether the degrees of freedom are the first dimension of the shape matrix, and not the dimension of the set. The first dimension is ``N^{2}`` on every covariance shape. The default is `false`.
    """
    ambient
    function ChiSqKUncertaintyAlgorithm(ambient::Bool)
        return new{typeof(ambient)}(ambient)
    end
end
function ChiSqKUncertaintyAlgorithm(; ambient::Bool = false)::ChiSqKUncertaintyAlgorithm
    return ChiSqKUncertaintyAlgorithm(ambient)
end
"""
    k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::MatNum, sigma_X::MatNum,
          ::Integer = size(sigma_X, 1))
    k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    k_ucs(km::ChiSqKUncertaintyAlgorithm, q::Number, ::Any, sigma_X::MatNum,
          df::Integer = size(sigma_X, 1))
    k_ucs(type::Number, args...)

Computes the radius `k` of an ellipsoidal uncertainty set at the significance level `q`.

The two closed forms are the two branches of Equation 11.23 of [cajas2025](@cite). The branch for every distribution is the factor of Theorem 1 of [elghaoui2003](@cite). The chi-squared branch comes from Equation 50 of [goldfarbiyengar2003](@cite). The radius from a simulated sample is the method that Section 11.3.2 of [cajas2025](@cite) describes. A plain `Number` in place of an algorithm is the radius.

# Algorithm

Each of the first three methods does one procedure. The fourth method, `k_ucs(type::Number, args...)`, returns its argument and does no step, so it has no numbered steps below.

[`NormalKUncertaintyAlgorithm`](@ref):

 1. Compute `k_mus = LinearAlgebra.diag(X * (sigma_X \\ transpose(X)))`, the squared Mahalanobis distance of each row of `X` against the shape matrix. The method does the solve one time for the full sample, not one row at a time.
 2. Take the `1 - q` quantile of `k_mus` with the keywords `km.kwargs`, and return its square root as the radius.

[`GeneralKUncertaintyAlgorithm`](@ref):

 1. Return `sqrt((one(q) - q) / q)` as the radius. The method reads neither `X` nor `sigma_X`, so `args...` takes the two of them.

[`ChiSqKUncertaintyAlgorithm`](@ref):

 1. Read the degrees of freedom from `df`, the dimension of the set. If `km.ambient` is `true`, read them from `size(sigma_X, 1)`.
 2. Return the square root of the `1 - q` chi-squared quantile at those degrees of freedom as the radius. The method runs no simulation, so it ignores the sample argument.

# Arguments

  - `km`: Radius algorithm.
  - `q`: Significance level.
  - `X`: Matrix of estimation errors, with one row for each sample. **Every caller gives centred deviations, not levels.** Each row is a deviation from the point estimate, and the method cannot check it. A sample that is not centred makes the distance non-central, and makes the radius too large.
  - `sigma_X`: Shape matrix of the ellipsoid, against which the method measures the distances. It is ``N \\times N`` on the mean axis and ``N^{2} \\times N^{2}`` on the covariance axis. Under its default `diagonal = true`, [`ellipsoidal_set`](@ref) gives the diagonal of the asymptotic covariance. So the method takes the quantile against that diagonal, not against the full matrix.
  - `df`: Dimension of the set, which [`ucs_dimension`](@ref) gives. The default `size(sigma_X, 1)` is correct on a shape of full rank. It is too large on a full covariance shape, whose rank is ``N(N+1)/2``, so [`ellipsoidal_set`](@ref) always gives `df`. [`NormalKUncertaintyAlgorithm`](@ref) ignores it, because its sampled distances contain the dimension.
  - `args...`: More arguments, which the algorithms that need no sample ignore.
  - `type`: The radius as a number.

# Returns

  - `k::Number`: Radius of the ellipsoid.

# Related

  - [`NormalKUncertaintyAlgorithm`](@ref)
  - [`GeneralKUncertaintyAlgorithm`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`ucs_dimension`](@ref)

# References

  - $(ref_dict[:elghaoui2003]) Theorem 1, Equations 7 and 8.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equation 50.
  - $(ref_dict[:cajas2025]) Section 11.3.2, Equation 11.23 and Listings 11.10 and 11.11. $(ref_dict[:no_original_source])
"""
function k_ucs(km::NormalKUncertaintyAlgorithm, q::Number, X::MatNum, sigma_X::MatNum,
               ::Integer = size(sigma_X, 1))
    A = sigma_X \ transpose(X)
    k_mus = [transpose(view(X, i, :)) * view(A, :, i) for i in axes(X, 1)]
    return sqrt(Statistics.quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_ucs(::GeneralKUncertaintyAlgorithm, q::Number, args...)
    return sqrt((one(q) - q) / q)
end
function k_ucs(km::ChiSqKUncertaintyAlgorithm, q::Number, ::Any, sigma_X::MatNum,
               df::Integer = size(sigma_X, 1))
    # The sample container is unused, because this route runs no simulation.
    p = km.ambient ? size(sigma_X, 1) : df
    return sqrt(Distributions.cquantile(Distributions.Chisq(p), q))
end
function k_ucs(type::Number, args...)::Number
    return type
end
"""
$(DocStringExtensions.TYPEDEF)

Selects an ellipsoidal uncertainty set, and holds the radius algorithm and the diagonal switch of the set.

[`BoxUncertaintySetAlgorithm`](@ref) selects the box. The shape matrix of the ellipsoid holds the correlation between the entries that it bounds. `diagonal = true` removes that correlation, to remove the noise in the off-diagonal estimation errors. Section 11.3.2 of [cajas2025](@cite) states that practice. Sections 3.2 and 4.1 of [sousalobo2000](@cite) state the two ellipsoids.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EllipsoidalUncertaintySetAlgorithm(;
        method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
        diagonal::Bool = true
    ) -> EllipsoidalUncertaintySetAlgorithm

  - `method`: The radius algorithm, or the radius as a number.
  - `diagonal`: Whether the shape matrix keeps only its diagonal.

# Examples

```jldoctest
julia> EllipsoidalUncertaintySetAlgorithm()
EllipsoidalUncertaintySetAlgorithm
    method ┼ ChiSqKUncertaintyAlgorithm
           │   ambient ┴ Bool: false
  diagonal ┴ Bool: true
```

# Related

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)

# References

  - $(ref_dict[:sousalobo2000]) Sections 3.2 and 4.1.
  - $(ref_dict[:cajas2025]) Section 11.3.2, Equation 11.24. $(ref_dict[:no_original_source])
"""
@concrete struct EllipsoidalUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    """
    $(field_dict[:method_ucs])
    """
    method
    """
    $(field_dict[:diagonal])
    """
    diagonal
    function EllipsoidalUncertaintySetAlgorithm(method::Num_UcSK, diagonal::Bool)
        return new{typeof(method), typeof(diagonal)}(method, diagonal)
    end
end
function EllipsoidalUncertaintySetAlgorithm(;
                                            method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
                                            diagonal::Bool = true)::EllipsoidalUncertaintySetAlgorithm
    return EllipsoidalUncertaintySetAlgorithm(method, diagonal)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects a norm-ball uncertainty set, and holds the radius algorithm, the diagonal switch and the norm order of the set.

[`BoxUncertaintySetAlgorithm`](@ref) and [`EllipsoidalUncertaintySetAlgorithm`](@ref) select the two older shapes. This algorithm selects a [`NormBallUncertaintySet`](@ref), which holds a geometry map and not a shape matrix. So an estimator that makes it factorises the shape one time during the fit, or stores its own deviations and makes no shape. `diagonal = true` removes the correlation between the estimation errors of different entries, as it does on the ellipsoid.

Theorem 2 of [bertsimas2011](@cite) states the worst case over a norm ball, which reads the dual norm. That work gives the result to [bertsimas2004](@cite). Equation 14 of [bentalnemirovski1998](@cite) defines the Euclidean case, which is an ellipsoid that can be flat.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormBallUncertaintySetAlgorithm(;
        method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
        diagonal::Bool = true,
        p::Number = 2,
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot()
    ) -> NormBallUncertaintySetAlgorithm

The keywords are the fields of the struct.

## Validation

  - `!isnan(p)` and `p >= 1`. `Inf` is valid.

# Examples

```jldoctest
julia> NormBallUncertaintySetAlgorithm()
NormBallUncertaintySetAlgorithm
    method ┼ ChiSqKUncertaintyAlgorithm
           │   ambient ┴ Bool: false
  diagonal ┼ Bool: true
         p ┼ Int64: 2
  mtx_sqrt ┴ EigenFallbackSquareRoot()
```

# Related

  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintyKAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`norm_ball_set`](@ref)
  - [`norm_ball_deviation_set`](@ref)

# References

  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
"""
@concrete struct NormBallUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    """
    Radius algorithm of the ball, or the radius as a `Number`. The algorithm reads the geometry map and not a shape matrix, so [`k_norm_ball`](@ref) computes the radius, and [`k_ucs`](@ref) does not.
    """
    method
    """
    $(field_dict[:diagonal])
    """
    diagonal
    """
    Norm order ``p \\geq 1`` of the ball. `Inf` is valid. The set gets it unchanged, and the consumer adds the cone of the dual order.
    """
    p
    """
    Square-root algorithm of the full asymptotic covariance, which gives the geometry map. `nothing` selects the plain Cholesky factor, which throws a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. [`matrix_square_root`](@ref) states each algorithm. The covariance of the covariance is always singular, and only the repair of `pdm` makes it positive definite. When the repair is off, the default takes the square root from the eigendecomposition. A [`NormalKUncertaintyAlgorithm`](@ref) radius applies the pseudo-inverse of the map. So it reads the eigen square root of a singular matrix, in the subspace that the cut of [`norm_ball_coordinates`](@ref) keeps. A ridge gives a map of full rank and a different set, and a diagonal shape reads no square root.
    """
    mtx_sqrt
    function NormBallUncertaintySetAlgorithm(method::Num_UcSK, diagonal::Bool, p::Number,
                                             mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm})
        @argcheck(!isnan(p) && p >= one(p), DomainError(p, "p must be >= 1"))
        return new{typeof(method), typeof(diagonal), typeof(p), typeof(mtx_sqrt)}(method,
                                                                                  diagonal,
                                                                                  p,
                                                                                  mtx_sqrt)
    end
end
function NormBallUncertaintySetAlgorithm(; method::Num_UcSK = ChiSqKUncertaintyAlgorithm(),
                                         diagonal::Bool = true, p::Number = 2,
                                         mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())::NormBallUncertaintySetAlgorithm
    return NormBallUncertaintySetAlgorithm(method, diagonal, p, mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDEF)

Names the axis of an uncertainty set, which sets the row count of its shape matrix or of its geometry map.

The family has two members, and the library exports both. A consumer dispatches on the tag. A mean set and a covariance set are the same struct with a different size of shape matrix or of geometry map. [`EllipsoidalUncertaintySet`](@ref) and [`NormBallUncertaintySet`](@ref) both hold a tag. The tag names only the axis, and not the geometry. So a norm ball of order one, which is not an ellipsoid, has the same tag as an ellipsoid on the same axis.

# Interfaces

A subtype is a tag. It has no field and declares no method.

# Related

  - [`MuUncertaintySetClass`](@ref)
  - [`SigmaUncertaintySetClass`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`NormBallUncertaintySet`](@ref)
"""
abstract type AbstractUncertaintySetClass <: AbstractUncertaintySetResult end
"""
$(DocStringExtensions.TYPEDEF)

Marks an uncertainty set as a set on the mean axis, where the shape matrix is ``N \\times N`` and the geometry map has ``N`` rows.

The tag applies to [`EllipsoidalUncertaintySet`](@ref) and [`NormBallUncertaintySet`](@ref), and the consumers dispatch on it. [`port_opt_view`](@ref) slices such a set with the plain asset index. The builder of the robust return refuses a set that has the covariance tag.

# Related

  - [`AbstractUncertaintySetClass`](@ref)
  - [`SigmaUncertaintySetClass`](@ref)
"""
struct MuUncertaintySetClass <: AbstractUncertaintySetClass end
"""
$(DocStringExtensions.TYPEDEF)

Marks an uncertainty set as a set on the covariance axis, where the shape matrix is ``N^{2} \\times N^{2}`` and the geometry map has ``N^{2}`` rows.

The tag applies to [`EllipsoidalUncertaintySet`](@ref) and [`NormBallUncertaintySet`](@ref), and the consumers dispatch on it. The set bounds a vectorised covariance. So [`port_opt_view`](@ref) maps the asset index through [`fourth_moment_index_generator`](@ref) before it slices the shape matrix or the geometry map.

# Related

  - [`AbstractUncertaintySetClass`](@ref)
  - [`MuUncertaintySetClass`](@ref)
"""
struct SigmaUncertaintySetClass <: AbstractUncertaintySetClass end
"""
$(DocStringExtensions.TYPEDEF)

Holds the shape matrix, the radius, and the axis tag of an ellipsoidal uncertainty set on a mean vector or on a covariance matrix.

An ellipsoid is a Mahalanobis ball, so it is a confidence region that holds the correlation between the entries that it bounds. Section 3.2 of [sousalobo2000](@cite) states the set and its worst case on the mean axis. Section 4.1 of the same work bounds the distinct entries of the covariance with an ellipsoid. Equation 6.12 of [fengpalomar2016](@cite) states that set again on ``\\text{vec}(\\mathbf{\\Sigma})``, and Equations 6.13 to 6.15 state its worst case. The two worst cases are second-order cones.

**`class` names the axis, and the axis sets the size of `sigma` and the index that a view applies.** A set with [`MuUncertaintySetClass`](@ref) has an ``N \\times N`` shape matrix and takes the plain asset index. A set with [`SigmaUncertaintySetClass`](@ref) has an ``N^{2} \\times N^{2}`` shape matrix, because it bounds a vectorised covariance. So [`port_opt_view`](@ref) finds ``N`` from the shape matrix, and maps the asset index through [`fourth_moment_index_generator`](@ref) before it slices. The two consumers also dispatch on the tag, and the builder of the robust return refuses a set with the covariance tag.

**A view keeps `k` unchanged, so it is not the set that the same estimator fits on the subset alone.** When the shape is diagonal, the restricted shape matrix is equal to the shape matrix fitted on the subset, entry for entry. The radius is not equal, because two of the three algorithms calibrate it on the dimension or on the sample. Only [`GeneralKUncertaintyAlgorithm`](@ref) gives the same radius, because it reads neither the data nor the shape. So a view is the conservative choice. A caller who needs the radius of the subset fits the subset.

# Mathematical definition

```math
\\begin{align}
U^{\\text{ellip}}_{\\boldsymbol{\\mu}} &= \\left\\{ \\boldsymbol{\\mu}\\, \\vert\\, \\left( \\boldsymbol{\\mu} - \\boldsymbol{\\hat{\\mu}} \\right)^{\\intercal} \\mathbf{\\Sigma}^{-1}_{\\boldsymbol{\\mu}} \\left( \\boldsymbol{\\mu} - \\boldsymbol{\\hat{\\mu}} \\right) \\leq k^{2}_{\\boldsymbol{\\mu}} \\right\\} \\\\
U^{\\text{ellip}}_{\\mathbf{\\Sigma}} &= \\left\\{ \\mathbf{\\Sigma}\\, \\vert\\, \\left( \\text{vec}\\left(\\mathbf{\\Sigma}\\right) - \\text{vec}\\left(\\mathbf{\\hat{\\Sigma}} \\right) \\right)^{\\intercal} \\mathbf{\\Sigma}^{-1}_{\\mathbf{\\Sigma}} \\left( \\text{vec}\\left(\\mathbf{\\Sigma}\\right) - \\text{vec}\\left(\\mathbf{\\hat{\\Sigma}} \\right) \\right) \\leq k^{2}_{\\mathbf{\\Sigma}},\\, \\mathbf{\\Sigma} \\succeq 0 \\right\\}\\,.
\\end{align}
```

Where:

  - ``U^{\\text{ellip}}_{\\boldsymbol{\\mu}}``: Ellipsoidal uncertainty set for expected returns.
  - ``U^{\\text{ellip}}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for the covariance matrix.
  - ``\\boldsymbol{\\mu}``, ``\\mathbf{\\Sigma}``: Uncertain expected returns and covariance.
  - ``\\boldsymbol{\\hat{\\mu}}``, ``\\mathbf{\\hat{\\Sigma}}``: Estimated mean and covariance, the centres of the two sets.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Covariance matrix of the estimation error of the mean.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Covariance matrix of the estimation error of the vectorised covariance.
  - ``k^{2}_{\\boldsymbol{\\mu}}``, ``k^{2}_{\\mathbf{\\Sigma}}``: Squared radii of the two ellipsoids.
  - ``\\text{vec}(\\cdot)``: Vectorisation operator, which stacks the columns.
  - ``\\mathbf{\\Sigma} \\succeq 0``: Constraint that the matrix is positive semidefinite.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EllipsoidalUncertaintySet(;
        sigma::MatNum,
        k::Number,
        class::AbstractUncertaintySetClass,
        val::Option{<:ArrNum} = nothing
    ) -> EllipsoidalUncertaintySet

The keywords are the fields of the struct.

## Validation

  - `!isempty(sigma)`.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `!(k isa Bool)`, else an `ArgumentError`. The converter to [`NormBallUncertaintySet`](@ref) puts `k` in its radius `kappa`, which also refuses a `Bool`.
  - `k > 0`.
  - If `val` is an array, `length(val) == size(sigma, 1)`. The rule reads a length and not a size, so it holds on the two axes. On the mean axis, `val` is a vector of length ``N`` beside an ``N \\times N`` shape matrix. On the covariance axis, it is an ``N \\times N`` covariance matrix beside an ``N^{2} \\times N^{2}`` shape matrix.

# Examples

```jldoctest
julia> EllipsoidalUncertaintySet([1.0 0.2; 0.2 1.0], 2.5, SigmaUncertaintySetClass())
EllipsoidalUncertaintySet
  sigma ┼ 2×2 Matrix{Float64}
      k ┼ Float64: 2.5
  class ┼ SigmaUncertaintySetClass()
    val ┴ nothing
```

# Related

  - [`AbstractUncertaintySetClass`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`k_ucs`](@ref)

# References

  - $(ref_dict[:sousalobo2000]) Sections 3.2 and 4.1.
  - $(ref_dict[:fengpalomar2016]) Section 6.1, Equations 6.6, 6.7 and 6.12 to 6.15.
"""
@concrete struct EllipsoidalUncertaintySet <: AbstractUncertaintySetResult
    """
    Shape matrix of the ellipsoid, which the estimator derives from the covariance of the estimation error. It is ``N \\times N`` on the mean axis and ``N^{2} \\times N^{2}`` on the covariance axis.
    """
    sigma
    """
    $(field_dict[:k_ucs])
    """
    k
    """
    $(field_dict[:class_ucs])
    """
    class
    """
    $(field_dict[:val_ucs])
    """
    val
    function EllipsoidalUncertaintySet(sigma::MatNum, k::Number,
                                       class::AbstractUncertaintySetClass,
                                       val::Option{<:ArrNum})
        @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(!isa(k, Bool),
                  ArgumentError("k is a radius, so it must be a number and not a Bool. Got\nk => $k."))
        @argcheck(k > zero(k), DomainError(k, "k must be positive"))
        if isa(val, ArrNum)
            @argcheck(length(val) == size(sigma, 1),
                      DimensionMismatch("val ($(length(val))) must match sigma ($(size(sigma, 1)))"))
        end
        return new{typeof(sigma), typeof(k), typeof(class), typeof(val)}(sigma, k, class,
                                                                         val)
    end
end
function EllipsoidalUncertaintySet(sigma::MatNum, k::Number,
                                   class::AbstractUncertaintySetClass)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(sigma, k, class, nothing)
end
function EllipsoidalUncertaintySet(; sigma::MatNum, k::Number,
                                   class::AbstractUncertaintySetClass,
                                   val::Option{<:ArrNum} = nothing)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(sigma, k, class, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a view of a covariance [`EllipsoidalUncertaintySet`](@ref) restricted to the assets at the index `i`.

The set bounds a vectorised covariance. So its shape matrix is on the ``N^{2}`` axis, and its centre is on the ``N`` axis. The method applies a different index to each of the two fields.

# Algorithm

 1. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted ``N \\times N`` covariance restricted to the selected assets. It takes the plain asset index, and it returns `nothing` unchanged. This step is first, because step 2 writes a new value to `i`.
 2. Find `N` as `floor(Int, sqrt(size(risk_ucs.sigma, 1)))` from the shape matrix. Expand `i` with `fourth_moment_index_generator(N, i)`, which gives the positions of the selected assets in the vectorised covariance.
 3. Take `view(risk_ucs.sigma, i, i)` with the expanded index, which gives the restricted shape matrix.
 4. Build an [`EllipsoidalUncertaintySet`](@ref) from the two views, with `k` and `class` unchanged. The method does not calibrate the radius again on the smaller dimension. So under every radius algorithm except [`GeneralKUncertaintyAlgorithm`](@ref), the view is more conservative than a fit on the subset.

# Arguments

  - `risk_ucs`: Ellipsoidal uncertainty set on the covariance axis.
  - `i`: Index of the cluster or of the assets.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `risk_ucs::EllipsoidalUncertaintySet`: The set restricted to `i`.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`SigmaUncertaintySetClass`](@ref)
  - [`fourth_moment_index_generator`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                           <:SigmaUncertaintySetClass}, i,
                       args...)::EllipsoidalUncertaintySet
    # `val` is the N x N covariance the set is a neighbourhood of, so it takes the asset
    # index, whereas the N^2 x N^2 shape matrix takes the fourth-moment index.
    val = nothing_scalar_array_view(risk_ucs.val, i)
    i = fourth_moment_index_generator(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipsoidalUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                     class = risk_ucs.class, val = val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Writes a covariance [`EllipsoidalUncertaintySet`](@ref) fitted on the Investable Mask back onto the full asset universe.

The expansion is the inverse of [`port_opt_view`](@ref). The view slices the shape matrix at the fourth-moment index of the selected assets, and the centre at the plain index. So the view removes all of the zero block outside the mask, and it gives back the fitted set. The centre is the full `pr.sigma`.

The frame is zero and not `NaN`, because the shape matrix is geometry. The same set converts into a [`NormBallUncertaintySet`](@ref), whose map must be finite. A zero row of that map has no effect on an asset that the prior did not estimate.

# Algorithm

 1. Find the positions of the investable pairs in the vectorised covariance with [`coverage_pair_index`](@ref), in the order of [`fourth_moment_index_generator`](@ref).
 2. Make a zero frame of `length(imsk)^2` rows and columns. Write `set.sigma` at those positions on the two axes.
 3. Build an [`EllipsoidalUncertaintySet`](@ref) from the frame, with `k` and `class` unchanged and `pr.sigma` as `val`.

# Arguments

  - `set`: Ellipsoidal uncertainty set on the covariance axis, fitted on the reduced prior.
  - `imsk`: The Investable Mask of the full prior.
  - $(arg_dict[:pr])

# Returns

  - `set::EllipsoidalUncertaintySet`: The set on the full asset universe.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`SigmaUncertaintySetClass`](@ref)
  - [`coverage_pair_index`](@ref)
  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                              <:SigmaUncertaintySetClass},
                               imsk::BitVector,
                               pr::AbstractPriorResult)::EllipsoidalUncertaintySet
    idx = coverage_pair_index(imsk)
    N2 = length(imsk)^2
    sigma = zeros(eltype(set.sigma), N2, N2)
    sigma[idx, idx] = set.sigma
    return EllipsoidalUncertaintySet(; sigma = sigma, k = set.k, class = set.class,
                                     val = pr.sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a view of a mean [`EllipsoidalUncertaintySet`](@ref) restricted to the assets at the index `i`.

The set bounds a mean vector, so its shape matrix and its centre are both on the ``N`` axis. One index serves the two fields.

# Algorithm

 1. Take `view(risk_ucs.sigma, i, i)`, the ``N \\times N`` shape matrix restricted to the selected assets on the two dimensions.
 2. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the fitted mean vector restricted to the same assets. It returns `nothing` unchanged.
 3. Build an [`EllipsoidalUncertaintySet`](@ref) from the two views, with `k` and `class` unchanged. The method does not calibrate the radius again on the smaller dimension. So under every radius algorithm except [`GeneralKUncertaintyAlgorithm`](@ref), the view is more conservative than a fit on the subset.

# Arguments

  - `risk_ucs`: Ellipsoidal uncertainty set on the mean axis.
  - `i`: Index of the cluster or of the assets.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `risk_ucs::EllipsoidalUncertaintySet`: The set restricted to `i`.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`MuUncertaintySetClass`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                           <:MuUncertaintySetClass}, i,
                       args...)::EllipsoidalUncertaintySet
    return EllipsoidalUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                     class = risk_ucs.class,
                                     val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Writes a mean [`EllipsoidalUncertaintySet`](@ref) fitted on the Investable Mask back onto the full asset universe.

The expansion is the inverse of [`port_opt_view`](@ref). The view slices the shape matrix on the two axes, and the centre, by the asset index. So the view removes all of the zero rows and columns outside the mask, and it gives back the fitted set. The centre is the full `pr.mu`.

The frame is zero and not `NaN`, because the shape matrix is geometry. The same set converts into a [`NormBallUncertaintySet`](@ref), whose map must be finite. A zero row of that map has no effect on an asset that the prior did not estimate.

# Algorithm

 1. Make a zero frame of `length(imsk)` rows and columns. Write `set.sigma` at `(imsk, imsk)`.
 2. Build an [`EllipsoidalUncertaintySet`](@ref) from the frame, with `k` and `class` unchanged and `pr.mu` as `val`.

# Arguments

  - `set`: Ellipsoidal uncertainty set on the mean axis, fitted on the reduced prior.
  - `imsk`: The Investable Mask of the full prior.
  - $(arg_dict[:pr])

# Returns

  - `set::EllipsoidalUncertaintySet`: The set on the full asset universe.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`MuUncertaintySetClass`](@ref)
  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::EllipsoidalUncertaintySet{<:MatNum, <:Any,
                                                              <:MuUncertaintySetClass},
                               imsk::BitVector,
                               pr::AbstractPriorResult)::EllipsoidalUncertaintySet
    N = length(imsk)
    sigma = zeros(eltype(set.sigma), N, N)
    sigma[imsk, imsk] = set.sigma
    return EllipsoidalUncertaintySet(; sigma = sigma, k = set.k, class = set.class,
                                     val = pr.mu)
end
"""
    box_quantile_bounds(::Type{TE}, get_ij, N::Integer, q::Number, kwargs) where {TE}

Computes the element-wise lower and upper quantile bounds of a symmetric ``N \\times N`` statistic.

The box methods of [`ucs`](@ref) and [`sigma_ucs`](@ref) of all estimator families use this function. The accessor `get_ij` lets one body serve all of them. It reads the Wishart sample, which is a vector of matrices, and the bootstrap sample, which is a three-dimensional array. The caller does the projection to a positive definite matrix, if one is necessary.

The two bounds do not always contain the sample mean of the statistic. A sample with one large outlier can put its mean outside the two quantiles.

# Algorithm

 1. Make `lb` and `ub`, both `N × N` with the element type `TE`.
 2. For each pair with `j <= i`, read `s_ij = get_ij(i, j)`, the sampled values of that entry.
 3. Write the `q` quantile of `s_ij` into `lb[i, j]` and `lb[j, i]`. Write the `1 - q` quantile into `ub[i, j]` and `ub[j, i]`. Each quantile goes to two positions, so the two bounds are symmetric, and each pair costs one quantile and not two.
 4. Return `lb` and `ub`. The caller gives `q < 1/2`, so `q` is the smaller quantile level of the same sample and `lb .<= ub` entrywise.

# Arguments

  - `TE`: Element type of the two bounds.
  - `get_ij`: Accessor function. `get_ij(i, j)` returns the vector of sampled values of the entry ``(i, j)``.
  - `N`: Number of rows and columns of the statistic.
  - `q`: Significance level, which the caller halved.
  - `kwargs`: Keyword arguments, which the function splats into `Statistics.quantile`.

# Returns

  - `lb::Matrix{TE}`: Element-wise lower bound, symmetric.
  - `ub::Matrix{TE}`: Element-wise upper bound, symmetric.

# Related

  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`vec_quantile_bounds`](@ref)
"""
function box_quantile_bounds(::Type{TE}, get_ij, N::Integer, q::Number, kwargs) where {TE}
    lb = Matrix{TE}(undef, N, N)
    ub = Matrix{TE}(undef, N, N)
    for j in 1:N
        for i in j:N
            s_ij = get_ij(i, j)
            lb[j, i] = lb[i, j] = Statistics.quantile(s_ij, q; kwargs...)
            ub[j, i] = ub[i, j] = Statistics.quantile(s_ij, one(q) - q; kwargs...)
        end
    end
    return lb, ub
end
"""
    vec_quantile_bounds(mus::MatNum, q::Number, kwargs)

Computes the element-wise lower and upper quantile bounds of a vector statistic.

The bootstrap box methods of [`ucs`](@ref) and [`mu_ucs`](@ref) use this function for the mean. **The samples are on the second axis.** The body reads `mus[j, :]`, so `mus` is ``N \\times M``, with one row for each component and one column for each sample. A caller that gives the transpose gets bounds of length ``M``, and the [`BoxUncertaintySet`](@ref) constructor accepts them. So the axis is a contract that this method cannot check.

The two bounds do not always contain the sample mean of a component. A sample with one large outlier can put its mean outside the two quantiles.

# Algorithm

 1. Read `N = size(mus, 1)`, the number of components. Make `lb` and `ub` of that length, with the element type `eltype(mus)`.
 2. For each component `j`, read the row `mu_j = mus[j, :]`, the `M` sampled values of that component.
 3. Write the `q` quantile of `mu_j` into `lb[j]`, and the `1 - q` quantile into `ub[j]`.
 4. Return `lb` and `ub`. The caller gives `q < 1/2`, so `lb .<= ub` entrywise.

# Arguments

  - `mus`: Sampled values, ``N \\times M``, with one row for each component.
  - `q`: Significance level, which the caller halved.
  - `kwargs`: Keyword arguments, which the function splats into `Statistics.quantile`.

# Returns

  - `lb::Vector`: Element-wise lower bound, length ``N``.
  - `ub::Vector`: Element-wise upper bound, length ``N``.

# Related

  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`box_quantile_bounds`](@ref)
"""
function vec_quantile_bounds(mus::MatNum, q::Number, kwargs)
    N = size(mus, 1)
    lb = Vector{eltype(mus)}(undef, N)
    ub = Vector{eltype(mus)}(undef, N)
    for j in 1:N
        mu_j = mus[j, :]
        lb[j] = Statistics.quantile(mu_j, q; kwargs...)
        ub[j] = Statistics.quantile(mu_j, one(q) - q; kwargs...)
    end
    return lb, ub
end
"""
    ucs_dimension(::MuUncertaintySetClass, ::Bool, m::Integer)
    ucs_dimension(::SigmaUncertaintySetClass, diagonal::Bool, m::Integer)
    ucs_dimension(class::AbstractUncertaintySetClass, diagonal::Bool, m::Integer,
                  M::Integer)

Returns the dimension of an uncertainty set on a shape of side `m`, which is the degrees of freedom of its chi-squared radius.

The dimension is the number of free coordinates of the estimation error. The squared Mahalanobis distance of a normal error has a chi-squared distribution at that number. On the mean axis, the dimension is `m`, the number of assets. On the covariance axis, the error is a vectorised symmetric matrix, so the entries ``(i, j)`` and ``(j, i)`` are one number. A full shape then has rank ``N(N+1)/2`` of its ``N^{2}``. The positive definite repair gives it a nominal full rank, and adds ``N(N-1)/2`` flat directions that no error fills, the antisymmetric matrices.

A diagonal shape keeps ``N^{2}``. Its statistic is a sum of ``N^{2}`` terms with a mean of one, so its expected value is ``N^{2}``.

The four-argument method is for a full shape that is the sample covariance of ``M`` sampled errors. The sample is centred, so that shape has a rank of ``M - 1`` or less, and the method limits the dimension to that value. The limit applies when ``M - 1`` is less than ``N`` on the mean axis, or less than ``N(N+1)/2`` on the covariance axis. A diagonal shape of sample variances has full rank, so it has no limit. The limit assumes a plain sample covariance. A shrinkage estimator gives a shape of full rank, and the limit then makes its dimension too small.

# Mathematical definition

```math
\\begin{align}
d &= \\begin{cases}
m & \\text{mean axis, or a diagonal shape}\\,, \\\\
\\dfrac{N(N+1)}{2}\\,, \\quad N = \\sqrt{m} & \\text{full covariance shape}\\,,
\\end{cases} \\\\
d_{M} &= \\begin{cases}
d & \\text{diagonal shape}\\,, \\\\
\\min(d, M - 1) & \\text{full shape}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``d``: Dimension of the set.
  - ``d_{M}``: Dimension of a set whose full shape is the sample covariance of ``M`` errors.
  - ``M``: Number of sampled errors.
  - ``m``: Side of the shape matrix, or row count of the geometry map.
  - $(math_dict[:N])

# Arguments

  - `class`: Axis tag of the set.
  - `diagonal`: Whether the shape is the diagonal of the asymptotic covariance.
  - `m`: Side of the shape matrix, ``N`` on the mean axis and ``N^{2}`` on the covariance axis.
  - `M`: Number of sampled errors, of which the shape is the sample covariance.

# Returns

  - `d::Integer`: Dimension of the set.

# Related

  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`k_norm_ball`](@ref)
  - [`ellipsoidal_set`](@ref)
  - [`norm_ball_set`](@ref)
"""
function ucs_dimension(::MuUncertaintySetClass, ::Bool, m::Integer)
    return m
end
function ucs_dimension(::SigmaUncertaintySetClass, diagonal::Bool, m::Integer)
    if diagonal
        return m
    end
    N = isqrt(m)
    return div(N * (N + 1), 2)
end
function ucs_dimension(class::AbstractUncertaintySetClass, diagonal::Bool, m::Integer,
                       M::Integer)
    d = ucs_dimension(class, diagonal, m)
    return diagonal ? d : min(d, M - 1)
end
"""
    ellipsoidal_set(alg::EllipsoidalUncertaintySetAlgorithm, q::Number, samples,
                    cov::MatNum, class::AbstractUncertaintySetClass,
                    val::Option{<:ArrNum} = nothing;
                    df::Integer = ucs_dimension(class, alg.diagonal, size(cov, 1)))

Builds an [`EllipsoidalUncertaintySet`](@ref) from an asymptotic covariance `cov` that the caller computed.

Every ellipsoidal method of [`ucs`](@ref), [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) in all estimator families uses this function. It is the ellipsoidal counterpart of [`norm_ball_set`](@ref), and it takes the set algorithm in the same position. [`k_ucs`](@ref) ignores the last arguments that its algorithm does not read. So `samples` can be the deviation matrix, a `1:n_sim` range or `nothing`, as the caller has it.

**The order of the first two steps below is necessary.** The function takes the diagonal *before* it computes the radius. So under the default `diagonal = true`, a radius from a sample is a quantile of Mahalanobis distances against the diagonal shape, not against the full shape. Neither shape always gives the larger radius. If the function takes the diagonal after the radius, the radius comes from one shape and the set uses a different shape. The set then does not have the coverage that its significance level states.

# Algorithm

 1. If `alg.diagonal` is `true`, replace `cov` with `LinearAlgebra.Diagonal(cov)`. This removes the correlations of the estimation errors between entries. The result is a `Diagonal`, not a dense matrix.
 2. Compute the radius `k = k_ucs(alg.method, q, samples, cov, df)` against the shape that step 1 gives. `df` is the dimension of the set, which is ``N(N+1)/2`` on a full covariance shape, and not the side of the shape.
 3. Build an [`EllipsoidalUncertaintySet`](@ref) from `cov`, `k`, `class` and `val`.

# Arguments

  - `alg`: Ellipsoidal uncertainty set algorithm, which holds the radius algorithm `alg.method` and the diagonal switch `alg.diagonal`.
  - `q`: Significance level.
  - `samples`: Sampled estimation errors, or the container that `alg.method` reads. An algorithm that runs no simulation ignores it.
  - `cov`: Asymptotic covariance of the statistic, which becomes the shape matrix.
  - `class`: Axis tag, which sets the size of the shape matrix and the index that a view applies.
  - `val`: Quantity that the set is a neighbourhood of. It is the fitted mean vector on the mean axis, and the fitted covariance on the covariance axis. Every caller has it, because every caller fits a prior before it calls this function.
  - `df`: Dimension of the set, which is the degrees of freedom of a chi-squared radius. The default is the dimension of the shape of the statistic. A caller whose shape is the sample covariance of ``M`` errors gives the dimension limited to ``M - 1``, the rank of that shape.

# Returns

  - `ucs::EllipsoidalUncertaintySet`: The assembled set.

# Related

  - [`EllipsoidalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`ucs_dimension`](@ref)
  - [`norm_ball_set`](@ref)
  - [`ucs`](@ref)
"""
function ellipsoidal_set(alg::EllipsoidalUncertaintySetAlgorithm, q::Number, samples,
                         cov::MatNum, class::AbstractUncertaintySetClass,
                         val::Option{<:ArrNum} = nothing;
                         df::Integer = ucs_dimension(class, alg.diagonal, size(cov, 1)))
    if alg.diagonal
        cov = LinearAlgebra.Diagonal(cov)
    end
    k = k_ucs(alg.method, q, samples, cov, df)
    return EllipsoidalUncertaintySet(; sigma = cov, k = k, class = class, val = val)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySet,
       NormalKUncertaintyAlgorithm, GeneralKUncertaintyAlgorithm,
       ChiSqKUncertaintyAlgorithm, EllipsoidalUncertaintySetAlgorithm,
       NormBallUncertaintySetAlgorithm, EllipsoidalUncertaintySet, SigmaUncertaintySetClass,
       MuUncertaintySetClass, AbstractUncertaintyEpsAlgorithm
public AbstractUncertaintySetEstimator, AbstractPriorUncertaintySetEstimator,
       AbstractUncertaintySetAlgorithm, AbstractUncertaintySetResult,
       AbstractUncertaintyKAlgorithm, AbstractCompactRadiusAlgorithm,
       AbstractUncertaintySetClass, reads_prior_result, k_ucs
