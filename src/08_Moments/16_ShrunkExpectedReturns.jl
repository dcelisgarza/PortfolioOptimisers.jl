"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrunk expected returns estimators.

All concrete and/or abstract types implementing shrinkage-based expected returns estimation algorithms should be subtypes of `AbstractShrunkExpectedReturnsEstimator`.

# Related

  - [`ShrunkExpectedReturns`](@ref)
  - [`EquilibriumExpectedReturns`](@ref)
  - [`ExcessExpectedReturns`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage algorithms for expected returns estimation.

All concrete and/or abstract types implementing specific shrinkage algorithms (e.g., James-Stein, Bayes-Stein) should be subtypes of `AbstractShrunkExpectedReturnsAlgorithm`.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.
"""
abstract type AbstractShrunkExpectedReturnsAlgorithm <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage targets used in expected returns estimation.

Concrete types implementing specific shrinkage targets (e.g., grand mean, volatility-weighted mean) should subtype `AbstractShrunkExpectedReturnsTarget`.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquaredError`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 3.43.
  - $(ref_dict[:meucci2005])
  - $(ref_dict[:fengpalomar2016])
"""
abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Fills the shrinkage target with the grand mean of the sample expected returns.

Every element of the target holds the same value, so a shrinkage estimator pulls each asset toward the average of the whole universe. The three targets are each a multiple of the vector of ones, and only the multiplier separates them.

# Mathematical definition

```math
\\begin{align}
b_j &= \\bar{\\mu} = \\frac{1}{N} \\sum_{i=1}^{N} \\hat{\\mu}_i\\,, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_j_shrink_tgt])
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:N])

The sample covariance matrix does not enter the form, so this target is the only one of the three that a singular covariance matrix leaves untouched.

# Algorithm

The branch of [`target_mean`](@ref) that this tag selects runs these steps.

 1. Take the unweighted mean of `mu`, giving `val`.
 2. Return the constant range that repeats `val` `length(mu)` times.

# Constructors

    GrandMean() -> GrandMean

# Examples

```jldoctest
julia> GrandMean()
GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 3.43.
  - $(ref_dict[:meucci2005])
  - $(ref_dict[:fengpalomar2016])
"""
struct GrandMean <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Fills the shrinkage target with the inverse-covariance-weighted mean of the sample expected returns.

The inverse covariance matrix supplies the weights. Under a diagonal covariance matrix each weight is the reciprocal of the asset's variance, so a riskier asset counts for less. The name says volatility, and the form reads the whole inverse covariance matrix, so an off-diagonal entry moves the target too.

# Mathematical definition

```math
\\begin{align}
b_j &= \\bar{\\mu}_{\\mathrm{vol}} = \\frac{\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}}{\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{1}}\\,, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_j_shrink_tgt])
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:Sigma_hat])
  - ``\\boldsymbol{1}``: ``N \\times 1`` vector of ones.
  - $(math_dict[:N])

# Algorithm

The branch of [`target_mean`](@ref) that this tag selects runs these steps.

 1. When `isigma` is `nothing`, solve `sigma \\ LinearAlgebra.I`, giving `isigma`. A caller that already holds the inverse passes it, so the solve runs once per estimate at most.
 2. When `mu` has one row, flatten it with `vec`, so that the product `isigma * mu` is defined.
 3. Divide the sum of `isigma * mu` by the sum of `isigma`, giving `val`. Summing a matrix-vector product is the numerator ``\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}``, and summing the matrix is the denominator ``\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{1}``.
 4. Return the constant range that repeats `val` `length(mu)` times.

# Constructors

    VolatilityWeighted() -> VolatilityWeighted

# Examples

```jldoctest
julia> VolatilityWeighted()
VolatilityWeighted()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 3.43.
  - $(ref_dict[:meucci2005])
  - $(ref_dict[:fengpalomar2016])
"""
struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Fills the shrinkage target with the trace of the covariance matrix divided by the number of observations.

Every element of the target holds the same value. The target reads a scale off the covariance matrix alone, so the sample expected returns do not enter it. It is the only one of the three targets that a shift of every asset's mean leaves where it was.

# Mathematical definition

```math
\\begin{align}
b_j &= \\frac{\\mathrm{tr}(\\hat{\\mathbf{\\Sigma}})}{T}\\,, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_j_shrink_tgt])
  - ``\\mathrm{tr}(\\cdot)``: Matrix trace operator.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

The branch of [`target_mean`](@ref) that this tag selects runs these steps.

 1. Divide the trace of `sigma` by `T`, giving `val`. `T` is a required keyword of this branch alone.
 2. Return the constant range that repeats `val` `length(mu)` times.

# Constructors

    MeanSquaredError() -> MeanSquaredError

# Examples

```jldoctest
julia> MeanSquaredError()
MeanSquaredError()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 3.43.
  - $(ref_dict[:meucci2005])
  - $(ref_dict[:fengpalomar2016])
"""
struct MeanSquaredError <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Blends the sample expected returns with the target under an intensity read off the covariance eigenvalues.

The intensity grows with the number of assets and falls with the distance between the sample mean and the target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JamesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> JamesStein

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> JamesStein()
JamesStein
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.1.
  - $(ref_dict[:meucci2005])
"""
@concrete struct JamesStein <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    function JamesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function JamesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::JamesStein
    return JamesStein(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Blends the sample expected returns with the target under an empirical Bayes intensity.

The intensity falls with the inverse-covariance-weighted distance between the sample mean and the target, so a short sample shrinks harder.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BayesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> BayesStein

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BayesStein()
BayesStein
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.2.
  - $(ref_dict[:jorion1986])
"""
@concrete struct BayesStein <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    function BayesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BayesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::BayesStein
    return BayesStein(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Combines the sample expected returns and the target under two coefficients from random matrix theory.

The two coefficients are set separately and neither is a convex weight, so the result is not a blend between the sample mean and the target. It suits a universe whose asset count is a large fraction of its observation count, and it needs more observations than assets.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BodnarOkhrinParolya(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> BodnarOkhrinParolya

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BodnarOkhrinParolya()
BodnarOkhrinParolya
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.3.
  - $(ref_dict[:bodnar2019])
"""
@concrete struct BodnarOkhrinParolya <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    function BodnarOkhrinParolya(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BodnarOkhrinParolya(;
                             tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::BodnarOkhrinParolya
    return BodnarOkhrinParolya(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the sample expected returns toward a target chosen by the shrinkage algorithm.

It holds the three parts the shrinkage needs: a mean estimator, a covariance estimator and a shrinkage algorithm, which carries the target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShrunkExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein()
    ) -> ShrunkExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `alg`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ShrunkExpectedReturns()
ShrunkExpectedReturns
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │   w ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │    w ┴ nothing
      │      │   alg ┼ FullMoment()
      │      │     w ┴ nothing
      │   mp ┼ MatrixProcessing
      │      │     pdm ┼ Posdef
      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │      dn ┼ nothing
      │      │      dt ┼ nothing
      │      │     alg ┼ nothing
      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
  alg ┼ JamesStein
      │   tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`target_mean`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.
"""
@propagatable @concrete struct ShrunkExpectedReturns <:
                               AbstractShrunkExpectedReturnsEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:me_shrink_alg])
    """
    @vprop alg
    function ShrunkExpectedReturns(me::AbstractExpectedReturnsEstimator,
                                   ce::StatsBase.CovarianceEstimator,
                                   alg::AbstractShrunkExpectedReturnsAlgorithm)
        return new{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
    end
end
function ShrunkExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())::ShrunkExpectedReturns
    return ShrunkExpectedReturns(me, ce, alg)
end
"""
    target_mean(
        tgt::GrandMean,
        mu::ArrNum,
        sigma::MatNum,
        args...;
        kwargs...
    ) -> StepRangeLen

    target_mean(
        tgt::VolatilityWeighted,
        mu::ArrNum,
        sigma::MatNum,
        isigma::Option{<:MatNum} = nothing;
        kwargs...
    ) -> StepRangeLen

    target_mean(
        tgt::MeanSquaredError,
        mu::ArrNum,
        sigma::MatNum,
        args...;
        T::Integer,
        kwargs...
    ) -> StepRangeLen

Compute the shrinkage target vector for expected returns estimation.

`target_mean` is the single owner of the three shrinkage targets. [`JamesStein`](@ref), [`BayesStein`](@ref) and [`BodnarOkhrinParolya`](@ref) all reach it, so each target is written once.

Every element of the returned vector holds the same value, so the function returns a `StepRangeLen` rather than a dense vector.

# Algorithm

The method that Julia selects is the algorithm, and the closed form of each branch lives on the tag that selects it. Every branch ends the same way: it computes one scalar `val` and returns `range(val, val; length = length(mu))`, a constant range rather than a dense vector.

 1. `tgt` is a [`GrandMean`](@ref): take the unweighted mean of `mu`. It reads neither `sigma` nor `isigma` nor `T`.
 2. `tgt` is a [`VolatilityWeighted`](@ref): solve for `isigma` when the caller passed none, flatten `mu` when it has one row, then divide the sum of `isigma * mu` by the sum of `isigma`.
 3. `tgt` is a [`MeanSquaredError`](@ref): divide the trace of `sigma` by the keyword `T`. It reads neither `mu` nor `isigma`, so only the length of `mu` reaches the result.

Each branch takes the arguments the other two do not need through `args...` and `kwargs...`, so one call site serves all three.

# Arguments

  - `tgt`: The shrinkage target type.

      + `tgt::GrandMean`: Fills the vector with the mean of `mu`.
      + `tgt::VolatilityWeighted`: Fills the vector with the inverse-covariance-weighted mean of `mu`.
      + `tgt::MeanSquaredError`: Fills the vector with the trace of `sigma` divided by `T`.

  - `mu`: 1D array of expected returns.

  - $(arg_dict[:sigma])

  - `isigma`: Inverse covariance matrix, taken **positionally** by the [`VolatilityWeighted`](@ref) method. If `nothing`, the method computes `sigma \\ LinearAlgebra.I` itself. The other two methods swallow it in `args...`.

  - `T`: Number of observations. It is a **required** keyword of the [`MeanSquaredError`](@ref) method. The other two methods swallow it in `kwargs...`.

  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `b::StepRangeLen`: Target vector for shrinkage estimation, of length `length(mu)`.

# Related

  - [`GrandMean`](@ref): the closed form of the branch of step 1.
  - [`VolatilityWeighted`](@ref): the closed form of the branch of step 2.
  - [`MeanSquaredError`](@ref): the closed form of the branch of step 3.
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`ArrNum`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 3.43.
  - $(ref_dict[:meucci2005])
  - $(ref_dict[:fengpalomar2016])
"""
function target_mean(::GrandMean, mu::ArrNum, sigma::MatNum, args...; kwargs...)
    val = Statistics.mean(mu)
    return range(val, val; length = length(mu))
end
function target_mean(::VolatilityWeighted, mu::ArrNum, sigma::MatNum,
                     isigma::Option{<:MatNum} = nothing; kwargs...)
    if isnothing(isigma)
        isigma = sigma \ LinearAlgebra.I
    end
    if isone(size(mu, 1))
        mu = vec(mu)
    end
    val = sum(isigma * mu) / sum(isigma)
    return range(val, val; length = length(mu))
end
function target_mean(::MeanSquaredError, mu::ArrNum, sigma::MatNum, args...; T::Integer,
                     kwargs...)
    val = LinearAlgebra.tr(sigma) / T
    return range(val, val; length = length(mu))
end
"""
    Statistics.mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute shrunk expected returns using the specified estimator.

This method applies a shrinkage algorithm to the sample expected returns, pulling them toward a target to reduce estimation error, especially in high-dimensional settings. **No method of this family clamps its coefficients.** [`JamesStein`](@ref) and [`BayesStein`](@ref) write `(1 - alpha) * mu + alpha * b`, and nothing holds `alpha` inside ``[0, 1]``, so the result can sit outside the segment that joins the sample mean and the target. [`BodnarOkhrinParolya`](@ref) sets its two coefficients separately and they do not sum to one.

# Mathematical definition

James-Stein shrinkage of the sample expected returns toward the target:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{JS} &= (1 - \\alpha)\\, \\hat{\\boldsymbol{\\mu}} + \\alpha\\, \\boldsymbol{b}\\,, \\\\
\\alpha &= \\frac{N \\bar{\\lambda} - 2 \\lambda_{\\max}}{T \\, \\lVert \\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b} \\rVert_2^2}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{JS}``: James-Stein shrunk expected returns.
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:b_shrink_tgt])
  - $(math_dict[:alpha_shrink_mu])
  - ``\\bar{\\lambda}``: Mean eigenvalue of the covariance matrix.
  - ``\\lambda_{\\max}``: Maximum eigenvalue of the covariance matrix.
  - $(math_dict[:T])
  - $(math_dict[:N])

Two consequences of the form bound where it is usable.

  - ``N \\bar{\\lambda}`` is the trace of the covariance matrix, so ``N \\bar{\\lambda} \\leq 2 \\lambda_{\\max}`` whenever ``N \\leq 2``. The intensity is then negative and the blend extrapolates away from the target rather than toward it.
  - The denominator is zero when the target equals the sample mean. [`GrandMean`](@ref) and [`VolatilityWeighted`](@ref) both reduce to the sample mean at ``N = 1``, so a one-asset sample returns `NaN` under either of them. [`MeanSquaredError`](@ref) does not read the sample mean, so it stays finite there.

# Algorithm

 1. Compute the sample expected returns with `me.me`, giving `mu`.
 2. Compute the covariance matrix with `me.ce`, giving `sigma`.
 3. Read `T` and `N` off `size(X)`, and swap them when `dims` is `2`.
 4. Compute the shrinkage target with [`target_mean`](@ref), giving `b`, and transpose it into a row when `dims` is `1`.
 5. Eigendecompose `sigma`, giving `evals`.
 6. Subtract `b` from `mu`, giving `mb`, and form the intensity `alpha` from `evals`, `mb`, `N` and `T`.
 7. Return the blend `(1 - alpha) * mu + alpha * b`.

# Arguments

  - `me`: Shrunk expected returns estimator.

      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}`: Use the James-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}`: Use the Bayes-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}`: Use the Bodnar-Okhrin-Parolya algorithm.

  - $(arg_dict[:X])

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the mean and covariance estimators.

# Returns

  - $(ret_dict[:mu])

# Related

  - [`JamesStein`](@ref): the tag that selects this method.
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)
  - [`ArrNum`](@ref)
  - [`MatNum`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.1.
  - $(ref_dict[:meucci2005])
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    b = target_mean(me.alg.tgt, mu, sigma; T = T)
    if flag
        b = transpose(b)
    end
    evals = LinearAlgebra.eigvals(sigma)
    mb = mu - b
    alpha = (N * Statistics.mean(evals) - 2 * maximum(evals)) / LinearAlgebra.dot(mb, mb) /
            T
    return (one(alpha) - alpha) * mu + alpha * b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`BayesStein`](@ref) overload of [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Shrinks sample returns toward the target using a Bayesian formula with inverse covariance weighting.

# Mathematical definition

Bayes-Stein shrinkage of the sample expected returns toward the target:

```math
\\begin{align}
\\alpha &= \\frac{N + 2}{(N + 2) + T \\, (\\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b})^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} (\\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b})}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_{BS} &= (1 - \\alpha)\\hat{\\boldsymbol{\\mu}} + \\alpha \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_shrink_mu])
  - ``\\hat{\\boldsymbol{\\mu}}_{BS}``: Bayes-Stein shrunk expected returns.
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:b_shrink_tgt])
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:N])

Two consequences of the form separate this intensity from the James-Stein one.

  - The quadratic form is non-negative whenever ``\\hat{\\mathbf{\\Sigma}}`` is positive semidefinite, so ``\\alpha`` then lies in ``(0, 1]``. This is the only one of the three algorithms whose coefficient is a convex weight without a clamp, and it returns the target exactly when the quadratic form is zero. A covariance estimator that returns an indefinite matrix breaks the bound.
  - The quadratic form uses the inverse of the covariance matrix that `me.ce` returns. Equation 3.44 of [cajas2025](@cite) states the same intensity over the bias-corrected matrix ``\\bar{\\mathbf{\\Sigma}} = \\frac{T-1}{T-N-1} \\hat{\\mathbf{\\Sigma}}``, and its own reference implementation uses ``\\hat{\\mathbf{\\Sigma}}``, as this method does. The correction raises ``\\alpha``.

# Algorithm

 1. Compute the sample expected returns with `me.me`, giving `mu`.
 2. Compute the covariance matrix with `me.ce`, giving `sigma`.
 3. Read `T` and `N` off `size(X)`, and swap them when `dims` is `2`.
 4. Solve `sigma \\ LinearAlgebra.I`, giving `isigma`, and pass it to [`target_mean`](@ref) so that the [`VolatilityWeighted`](@ref) branch does not solve a second time.
 5. Compute the shrinkage target, giving `b`, and transpose it into a row when `dims` is `1`.
 6. Flatten `mu - b` with `vec`, giving `mb`, and form the intensity `alpha` from `mb`, `isigma`, `N` and `T`.
 7. Return the blend `(1 - alpha) * mu + alpha * b`.

# Related

  - [`BayesStein`](@ref): the tag that selects this method.
  - [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref): the arguments, the return value and the trap the three overloads share.
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.2, Equation 3.44.
  - $(ref_dict[:jorion1986])
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
    end
    mb = vec(mu - b)
    alpha = (N + 2) / ((N + 2) + T * LinearAlgebra.dot(mb, isigma, mb))
    return (one(alpha) - alpha) * mu + alpha * b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`BodnarOkhrinParolya`](@ref) overload of [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Shrinks sample returns toward the target using the Bodnar-Okhrin-Parolya formula, designed for robust high-dimensional estimation. It needs ``T > N``: the term ``N/(T-N)`` is undefined at ``T = N`` and changes sign below it, so a square or wide returns matrix returns `Inf` or a coefficient of the wrong sign.

# Mathematical definition

Three inverse-covariance-weighted quadratic forms carry the sample mean and the target:

```math
\\begin{align}
u &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}\\,, \\\\
v &= \\boldsymbol{b}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{b}\\,, \\\\
w &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{b}\\,.
\\end{align}
```

The two coefficients and the combination follow from them:

```math
\\begin{align}
\\alpha &= \\frac{(u - N/(T-N))v - w^2}{uv - w^2}\\,, \\\\
\\beta &= \\frac{(1-\\alpha) w}{u}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_{BOP} &= \\alpha \\hat{\\boldsymbol{\\mu}} + \\beta \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``u``, ``v``, ``w``: Inverse-covariance-weighted quadratic forms.
  - ``\\alpha``, ``\\beta``: Shrinkage coefficients.
  - ``\\hat{\\boldsymbol{\\mu}}_{BOP}``: Bodnar-Okhrin-Parolya shrunk expected returns.
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:b_shrink_tgt])
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:N])

Three consequences of the form separate this algorithm from the other two.

  - ``\\alpha`` and ``\\beta`` are set separately and do not sum to one, so the result is not a point on the segment that joins ``\\hat{\\boldsymbol{\\mu}}`` and ``\\boldsymbol{b}``. Cancelling the ``w^2`` term rewrites the coefficient as ``\\alpha = 1 - \\frac{N}{T-N} \\frac{v}{uv - w^2}``, so ``\\alpha < 1`` always, and ``\\alpha < 0`` exactly when ``\\frac{N}{T-N} v > uv - w^2``. The combination then extrapolates away from the sample mean.
  - ``uv - w^2`` is a Cauchy-Schwarz gap in the inner product ``\\langle \\boldsymbol{x}, \\boldsymbol{y} \\rangle = \\boldsymbol{x}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{y}``, so it vanishes exactly when the target is a multiple of the sample mean. At ``N = 1`` every vector is such a multiple, so a one-asset sample divides zero by zero and returns `NaN` under all three targets.
  - Every target of this file is a multiple of the vector of ones, so writing ``\\boldsymbol{b} = c \\boldsymbol{1}`` makes ``v`` and ``w`` scale with ``c^2`` and ``c``. The factor cancels in ``\\alpha``, which is therefore the same for the three targets on one sample, and survives in ``\\beta \\boldsymbol{b}``, which is not.

# Algorithm

 1. Compute the sample expected returns with `me.me`, giving `mu`.
 2. Compute the covariance matrix with `me.ce`, giving `sigma`.
 3. Read `T` and `N` off `size(X)`, and swap them when `dims` is `2`.
 4. Solve `sigma \\ LinearAlgebra.I`, giving `isigma`, and pass it to [`target_mean`](@ref) so that the [`VolatilityWeighted`](@ref) branch does not solve a second time.
 5. Compute the shrinkage target, giving `b`, and transpose it into a row when `dims` is `1`.
 6. Flatten `mu` and `b` into the vectors `vm` and `vb`, which the quadratic forms need whichever way `dims` orients the data.
 7. Form the three quadratic forms `u`, `v` and `w` from `vm`, `vb` and `isigma`.
 8. Form `alpha` from `u`, `v`, `w`, `N` and `T`, then `beta` from `alpha`, `w` and `u`.
 9. Return the combination `alpha * mu + beta * b`.

# Related

  - [`BodnarOkhrinParolya`](@ref): the tag that selects this method.
  - [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref): the arguments, the return value and the trap the three overloads share.
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.4.1.3.
  - $(ref_dict[:bodnar2019])
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya},
                         X::MatNum; dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
        vm = vec(mu)
        vb = vec(b)
    else
        vm = mu
        vb = b
    end
    u = LinearAlgebra.dot(vm, isigma, vm)
    v = LinearAlgebra.dot(vb, isigma, vb)
    w = LinearAlgebra.dot(vm, isigma, vb)
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= u * v - w^2
    beta = (one(alpha) - alpha) * w / u
    return alpha * mu + beta * b
end

export GrandMean, VolatilityWeighted, MeanSquaredError, JamesStein, BayesStein,
       BodnarOkhrinParolya, ShrunkExpectedReturns
