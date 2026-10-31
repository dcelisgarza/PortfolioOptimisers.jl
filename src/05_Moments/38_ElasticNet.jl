"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that set the regularisation strength of an [`ElasticNet`](@ref).

The strength scales the penalty against the weighted squared residuals. [`MiddleOfPath`](@ref) reads it off the data, as [lai2018ktpt](@citet) do. [`FixedStrength`](@ref) takes it from the caller.

# Interfaces

To add a strength rule, subtype `AbstractPenaltyStrength`, keep its parameters in the fields of the struct, and add the methods below. A target with `sdiv = nothing` does not read `penalty_strength_gradient`, so a rule that runs under that target alone needs only `penalty_strength`.

## `penalty_strength`

  - `penalty_strength(s::MyPenaltyStrength, g::VecNum, theta::Real) -> Real`: Return the regularisation strength ``\\gamma`` of the fit.

[`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref) calls it once, before the coordinate sweeps. When every entry of `g` is zero, the fit returns its start point at every strength, and it does not read the strength. It records the strength in the fit.

### Arguments

  - `s`: The new strength rule.
  - `g`: The product of each penalised column with the residual of the start point, ``\\mathbf{P}_k^\\intercal \\mathbf{V} \\boldsymbol{r}_0`` for ``k \\in \\mathcal{K}``. It is empty when the design has no penalised column.
  - `theta`: The share of the ``L_1`` penalty, the `theta` field of the target.

### Returns

  - `gamma::Real`: The regularisation strength, non-negative.

## `penalty_strength_gradient`

  - `penalty_strength_gradient(s::MyPenaltyStrength, g::VecNum, theta::Real) -> VecNum`: Return the gradient of the strength that `penalty_strength` returns, with respect to `g`.

[`StrengthDivergence`](@ref) reads it, because a strength that reads the response adds a term to the divergence of the fit. A rule that does not read `g` returns zeros.

### Arguments

  - `s`: The new strength rule.
  - `g`: The vector that `penalty_strength` reads.
  - `theta`: The share of the ``L_1`` penalty.

### Returns

  - `dg::VecNum`: The partial derivative of ``\\gamma`` with respect to each entry of `g`, one entry per entry of `g`.

# Related

  - [`ElasticNet`](@ref)
  - [`MiddleOfPath`](@ref)
  - [`FixedStrength`](@ref)
  - [`penalty_strength`](@ref)
  - [`penalty_strength_gradient`](@ref)
  - [`StrengthDivergence`](@ref)
"""
abstract type AbstractPenaltyStrength <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Sets the regularisation strength of an [`ElasticNet`](@ref) at the geometric middle of its regularisation path.

[lai2018ktpt](@citet) take "the middle one in the regularization path" and state no grid. Section 2.5 of [friedman2010](@cite) runs the path over 100 strengths on a log scale, from the smallest strength at which every penalised coefficient is zero down to `ratio` times that strength. Such a grid has no single middle point. The rule takes the geometric middle of the path, which lies between the 50th and the 51st point of that grid at the default `ratio = 1e-3`.

# Mathematical definition

```math
\\begin{align}
\\gamma_{\\max} &= \\max_{k \\in \\mathcal{K}} \\frac{\\lvert \\mathbf{P}_k^\\intercal \\mathbf{V} \\boldsymbol{r}_0 \\rvert}{\\vartheta}\\,,\\\\
\\gamma &= \\gamma_{\\max} \\sqrt{\\rho}\\,.
\\end{align}
```

Where:

  - ``\\gamma_{\\max}``: Smallest strength at which every penalised coefficient is zero.
  - $(math_dict[:K_enet])
  - $(math_dict[:P_enet])
  - $(math_dict[:V_enet])
  - ``\\boldsymbol{r}_0``: Residual of the start point, the response less the fit of the intercept column alone, or the response when the design has no intercept column.
  - $(math_dict[:theta_enet])
  - $(math_dict[:gamma_enet])
  - ``\\rho``: Floor of the path as a fraction of ``\\gamma_{\\max}``, the `ratio` field.

Every penalised coefficient is zero at and above ``\\gamma_{\\max}``. The path runs from ``\\gamma_{\\max}`` down to ``\\rho\\, \\gamma_{\\max}`` on a log scale, so its geometric middle is ``\\gamma_{\\max} \\sqrt{\\rho}``. A ridge alone, ``\\vartheta = 0``, has no ``\\gamma_{\\max}``, so the constructor of [`ElasticNet`](@ref) refuses `theta = 0` with this rule.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MiddleOfPath(; ratio::Real = 1e-3) -> MiddleOfPath

Keywords correspond to the struct's fields.

## Validation

  - `0 < ratio < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> MiddleOfPath()
MiddleOfPath
  ratio ┴ Float64: 0.001
```

# Related

  - [`AbstractPenaltyStrength`](@ref)
  - [`FixedStrength`](@ref)
  - [`ElasticNet`](@ref)
  - [`penalty_strength`](@ref)

# References

  - $(ref_dict[:friedman2010]) Section 2.5.
  - $(ref_dict[:lai2018ktpt]) Section 3.1.2.
"""
@concrete struct MiddleOfPath <: AbstractPenaltyStrength
    """
    The floor of the path as a fraction of the strength above which every penalised coefficient is zero.
    """
    ratio
    function MiddleOfPath(ratio::Real)
        @argcheck(zero(ratio) < ratio < one(ratio),
                  DomainError(ratio, "ratio must be in (0, 1)"))
        return new{typeof(ratio)}(ratio)
    end
end
function MiddleOfPath(; ratio::Real = 1e-3)::MiddleOfPath
    return MiddleOfPath(ratio)
end
"""
$(DocStringExtensions.TYPEDEF)

Sets the regularisation strength of an [`ElasticNet`](@ref) to a value that the caller states.

`gamma` is the ``\\gamma`` of the objective of [`ElasticNet`](@ref). It weighs the penalty against the sum of the weighted squared residuals, and it does not divide that sum by the number of observations. So the same `gamma` penalises a fit over more observations less, in relation to its residuals. [friedman2010](@citet) divide the objective by twice the sum of the weights, so their strength ``\\lambda`` of Equation (1) is ``\\gamma / T_{e}``, with ``T_{e}`` the sum of the scaled weights, which is the number of observations of an unweighted fit. With a variance estimator in the `ve` field of the target, `gamma` acts on the standardised columns. `theta = 0` with this rule is a ridge regression.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FixedStrength(; gamma::Real) -> FixedStrength

Keywords correspond to the struct's fields.

## Validation

  - `gamma > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> FixedStrength(; gamma = 0.5)
FixedStrength
  gamma ┴ Float64: 0.5
```

# Related

  - [`AbstractPenaltyStrength`](@ref)
  - [`MiddleOfPath`](@ref)
  - [`ElasticNet`](@ref)
  - [`penalty_strength`](@ref)

# References

  - $(ref_dict[:friedman2010]) Equation (1).
"""
@concrete struct FixedStrength <: AbstractPenaltyStrength
    """
    The regularisation strength ``\\gamma`` of the objective.
    """
    gamma
    function FixedStrength(gamma::Real)
        @argcheck(gamma > zero(gamma), DomainError(gamma, "gamma must be positive"))
        return new{typeof(gamma)}(gamma)
    end
end
function FixedStrength(; gamma::Real)::FixedStrength
    return FixedStrength(gamma)
end
"""
    penalty_strength(s::MiddleOfPath, g::AbstractVector, theta::Real) -> Real
    penalty_strength(s::FixedStrength, g::AbstractVector, theta::Real) -> Real

Return the regularisation strength of an [`ElasticNet`](@ref) under its strength rule.

[`MiddleOfPath`](@ref) states the middle of the path, and [`FixedStrength`](@ref) returns its own `gamma`. The `# Interfaces` section of [`AbstractPenaltyStrength`](@ref) states the method that a rule of the caller adds.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`MiddleOfPath`](@ref): divide the largest absolute entry of `g` by `theta`, which gives `lmax`, and return `lmax * sqrt(s.ratio)`. An empty `g` gives zero.
 2. [`FixedStrength`](@ref): return `s.gamma`, and read neither `g` nor `theta`.

# Arguments

  - `s`: The strength rule.
  - `g`: The product of each penalised column with the residual of the start point.
  - `theta`: The share of the ``L_1`` penalty.

# Returns

  - `gamma::Real`: The regularisation strength.

# Related

  - [`AbstractPenaltyStrength`](@ref)
  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)
"""
function penalty_strength(s::MiddleOfPath, g::AbstractVector, theta::Real)
    lmax = maximum(abs, g; init = zero(eltype(g))) / theta
    return lmax * sqrt(s.ratio)
end
function penalty_strength(s::FixedStrength, ::AbstractVector, ::Real)
    return s.gamma
end
"""
    penalty_strength_gradient(s::MiddleOfPath, g::AbstractVector, theta::Real) -> VecNum
    penalty_strength_gradient(s::FixedStrength, g::AbstractVector, theta::Real) -> VecNum

Return the gradient of the regularisation strength of an [`ElasticNet`](@ref) with respect to the vector that [`penalty_strength`](@ref) reads.

[`StrengthDivergence`](@ref) chains it to the gradient with respect to the response. The `# Interfaces` section of [`AbstractPenaltyStrength`](@ref) states the method that a rule of the caller adds.

# Mathematical definition

```math
\\begin{align}
\\frac{\\partial \\gamma}{\\partial g_{j}} &= \\frac{\\sqrt{\\rho}}{\\vartheta} \\operatorname{sign}(g_{k^{*}})\\, \\mathbb{1}[j = k^{*}]\\,,\\quad k^{*} = \\underset{k}{\\arg\\max} \\lvert g_{k} \\rvert\\,.
\\end{align}
```

Where:

  - $(math_dict[:gamma_enet])
  - ``g_{k}``: Entry ``k`` of `g`, the product of penalised column ``k`` with the residual of the start point.
  - ``\\rho``: Floor of the path as a fraction of ``\\gamma_{\\max}``, the `ratio` field of [`MiddleOfPath`](@ref).
  - $(math_dict[:theta_enet])

The form is the derivative of ``\\gamma = \\sqrt{\\rho} \\max_{k} \\lvert g_{k} \\rvert / \\vartheta``, at a point where one entry holds the largest absolute value.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`MiddleOfPath`](@ref): return the empty `g` when `g` is empty. Otherwise find the index `k` of the largest absolute entry of `g`, and return a vector of zeros with `sign(g[k]) * sqrt(s.ratio) / theta` at `k`.
 2. [`FixedStrength`](@ref): return zeros, because the strength does not read `g`.

# Arguments

  - `s`: The strength rule.
  - `g`: The product of each penalised column with the residual of the start point.
  - `theta`: The share of the ``L_1`` penalty.

# Returns

  - `dg::VecNum`: The gradient, one entry per entry of `g`.

# Related

  - [`AbstractPenaltyStrength`](@ref)
  - [`penalty_strength`](@ref)
  - [`StrengthDivergence`](@ref)
"""
function penalty_strength_gradient(s::MiddleOfPath, g::AbstractVector, theta::Real)
    if isempty(g)
        return g
    end
    k = argmax(abs.(g))
    val = sign(g[k]) * sqrt(s.ratio) / theta
    dg = zeros(typeof(val), length(g))
    dg[k] = val
    return dg
end
function penalty_strength_gradient(::FixedStrength, g::AbstractVector, ::Real)
    return zero(g)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that add the term of a strength that reads the response to the effective degrees of freedom of an [`ElasticNet`](@ref).

The trace of the smoother on the active set is the effective degrees of freedom of a fit at a fixed strength, Remark (The elastic net) after Equation (28) of [tibshirani2012](@cite). [`MiddleOfPath`](@ref) reads the strength off the response, so the divergence of its fit has one more term. [`StrengthDivergence`](@ref) adds it. The `sdiv` field of [`ElasticNet`](@ref) holds the rule, and `nothing`, the default, leaves the term out.

# Interfaces

To add a rule, subtype `AbstractStrengthDivergence`, and add the method below.

## `strength_divergence`

  - `strength_divergence(sdiv::MyStrengthDivergence, tgt::ElasticNet, sys::NamedTuple) -> Real`: Return the term that the rule adds to the trace of the smoother.

[`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref) calls it once, after the fit, through `elastic_net_counts`. The term changes `df` of [`ElasticNetFit`](@ref) alone, not the residual count `rdof`.

### Arguments

  - `sdiv`: The new rule.

  - `tgt`: The regression target.

  - `sys::NamedTuple`: The fit, `(; P, Q, y, z, pen, ridge, g, lam, nrm2, w, A, Mi)`:

      + `P`, `Q`: The standardised design and the weighted design, as `elastic_net_design` returns them.
      + `y`: The response.
      + `z`: The coefficients of the fit on the columns of `P`.
      + `pen`: Whether each column is penalised.
      + `ridge`: The ridge term of each column.
      + `g`, `lam`: The vector that [`penalty_strength`](@ref) read, and the strength that it returned.
      + `nrm2`: The weighted squared norm of each column.
      + `w`: The frequency weights of the rows, or `nothing`.
      + `A`: The indices of the active set.
      + `Mi`: The pseudo-inverse of the matrix of the equations of the active set.

### Returns

  - `t::Real`: The term.

# Related

  - [`StrengthDivergence`](@ref)
  - [`ElasticNet`](@ref)
  - [`penalty_strength_gradient`](@ref)
  - [`strength_divergence`](@ref)

# References

  - $(ref_dict[:tibshirani2012]) Remark (The elastic net) after Equation (28).
"""
abstract type AbstractStrengthDivergence <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Adds the term of a strength that reads the response to the effective degrees of freedom of an [`ElasticNet`](@ref).

The degrees of freedom of a fit ``\\hat{\\boldsymbol{\\mu}}`` is the expected divergence ``\\sum_{i} \\partial \\hat{\\mu}_{i} / \\partial y_{i}``, Equations (10) and (11) of [tibshirani2012](@cite). On a fixed active set, the fit at a fixed strength is linear in the response, and its divergence is the trace of the smoother. A strength that reads the response adds the chain term below. Under [`FixedStrength`](@ref) the gradient of the strength is zero, so the rule adds zero.

# Mathematical definition

```math
\\begin{align}
\\operatorname{div} \\hat{\\boldsymbol{\\mu}} &= \\operatorname{tr}(\\mathbf{H}) + \\nabla_{\\boldsymbol{y}} \\gamma^{\\intercal} \\frac{\\partial \\hat{\\boldsymbol{\\mu}}}{\\partial \\gamma}\\,,\\\\
\\frac{\\partial \\hat{\\boldsymbol{\\mu}}}{\\partial \\gamma} &= -\\mathbf{P}_{\\mathcal{A}} \\mathbf{M}^{-1} \\left((1 - \\vartheta) \\mathbf{D}_{\\mathcal{A}} \\boldsymbol{z}_{\\mathcal{A}} + \\vartheta \\boldsymbol{s}\\right)\\,,\\\\
\\nabla_{\\boldsymbol{y}} \\gamma &= (\\mathbf{I} - \\mathbf{H}_{0})^{\\intercal} \\mathbf{V} \\mathbf{P}_{\\mathcal{K}} \\frac{\\partial \\gamma}{\\partial \\boldsymbol{g}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: Fitted response, ``\\mathbf{P} \\hat{\\boldsymbol{z}}``.
  - ``\\mathbf{H}``: Smoother of the fit on its active set, which [`ElasticNetFit`](@ref) states.
  - ``\\mathbf{M}``: Matrix of the equations of the active set, ``\\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\mathbf{P}_{\\mathcal{A}} + \\gamma (1 - \\vartheta) \\mathbf{D}_{\\mathcal{A}}``.
  - ``\\mathbf{P}_{\\mathcal{A}}``, ``\\boldsymbol{z}_{\\mathcal{A}}``: Columns of the regressors and coefficients of the active set.
  - ``\\mathbf{D}_{\\mathcal{A}}``: Diagonal matrix with one for each penalised column of the active set and zero for the intercept column.
  - ``\\boldsymbol{s}``: Signs of the coefficients of the active set, zero for the intercept column.
  - ``\\mathbf{H}_{0}``: Weighted projection onto the intercept column, the zero matrix when the design has none. The start point of [`penalty_strength`](@ref) is the residual ``(\\mathbf{I} - \\mathbf{H}_{0}) \\boldsymbol{y}``.
  - ``\\mathbf{P}_{\\mathcal{K}}``: Penalised columns of the design.
  - ``\\partial \\gamma / \\partial \\boldsymbol{g}``: Gradient of the strength rule, [`penalty_strength_gradient`](@ref).
  - $(math_dict[:P_enet])
  - $(math_dict[:V_enet])
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - $(math_dict[:K_enet])

The second line differentiates the equations of [`elastic_net_polish`](@ref) in ``\\gamma`` on the fixed active set. On the design of [`ElasticNet`](@ref) that states the gap, the count with the term matches the mean finite-difference divergence of the fit.

# Examples

```jldoctest
julia> StrengthDivergence()
StrengthDivergence()
```

# Related

  - [`AbstractStrengthDivergence`](@ref)
  - [`ElasticNet`](@ref)
  - [`penalty_strength_gradient`](@ref)
  - [`strength_divergence`](@ref)

# References

  - $(ref_dict[:tibshirani2012]) Equations (10) and (11), and Remark (The elastic net) after Equation (28).
"""
struct StrengthDivergence <: AbstractStrengthDivergence end
"""
$(DocStringExtensions.TYPEDEF)

Fits each response by the elastic net, a least-squares fit with a penalty on the ``L_1`` norm and on the squared ``L_2`` norm of the coefficients.

The objective is Equation (15) of [lai2018ktpt](@cite), the elastic net of [zouhastie2005](@citet), under observation weights. It is Equation (3) of [zouhastie2005](@cite) with ``\\lambda_1 = 2 \\vartheta \\gamma`` and ``\\lambda_2 = (1 - \\vartheta) \\gamma``, so ``\\vartheta`` is the mixing weight of Equation (2) of [friedman2010](@cite), not the ratio ``\\lambda_2 / (\\lambda_1 + \\lambda_2)`` of [zouhastie2005](@cite). Equation (10) of [friedman2010](@cite) is the coordinate update. `strength` sets ``\\gamma``: [`MiddleOfPath`](@ref), the default, reads it off the data as [lai2018ktpt](@citet) do, and [`FixedStrength`](@ref) takes it from the caller. `theta = 0.99` is the value of [lai2018ktpt](@cite). It is almost the lasso, and it keeps the objective strictly convex.

The target adds no intercept and does not standardise the columns by default, as in the model of [lai2018ktpt](@cite). The glmnet package does both by default. An estimator that adds a column of ones names it with [`InterceptColumn`](@ref) through [`factory`](@ref), and the fit then leaves that coefficient out of the penalty. A variance estimator in `ve` divides each penalised column by its standard deviation before the fit, and the fit scales the coefficients back. A column whose standard deviation is zero has no scale, so it keeps its units. `ve` reads the observation weights of the target, so the scale and the fit weigh the rows alike.

The observation weights enter as the frequency weights of [`glm_weights`](@ref), the weights that [`LinearModel`](@ref) passes to `GLM`. Weights that are not `StatsBase.FrequencyWeights` then sum to their Kish effective sample size. Uniform weights give the unweighted fit at the same strength. The scale of the weights moves the fit under [`FixedStrength`](@ref) alone, because [`MiddleOfPath`](@ref) scales with them.

When the columns are almost collinear, for example the levels of consecutive periods in [`KernelTrendPattern`](@ref), the coordinate sweeps converge slowly. Each sweep therefore ends with an exact solve on the sign pattern that it leaves, see [`elastic_net_polish`](@ref). The sweeps must still find the sign pattern of the optimum. On the 4 × 5 window of the test fixture of [`KernelTrendPattern`](@ref) they find it after between 2000 and 5000 sweeps, and a cap of 1000 stops more than 0.5 from the optimum in one coefficient. When the sweeps stop at `tol` or at `iters` before that, the result is the last sweep, which is not the optimum.

The penalty depends on the units of each column, so the fit is not invariant to the basis of the design, and [`is_basis_invariant`](@ref) returns `false`. [`StepwiseRegression`](@ref) accepts the target under `:aic`, `:aicc` and `:bic` alone, because these criteria read the effective degrees of freedom that [`ElasticNetFit`](@ref) states.

The effective degrees of freedom is the trace of the smoother on the active set of the fit. That trace is exact at a fixed strength. [`MiddleOfPath`](@ref) reads the strength off the response, so the divergence of its fit has one more term, and `sdiv` gates it. With `sdiv = nothing`, the default, the count leaves the term out. The gap was measured on 2000 draws of a design of 60 observations, an intercept and 5 factors, with the coefficients `[0.5, 1, -0.5, 0, 0, 0.25]` and a unit noise. There the trace was above the mean finite-difference divergence of the fit by 0.06 at `theta = 0.5` and by 0.03 at `theta = 0.99`, with a standard error of 0.014. One factor costs about one degree of freedom, so the gap did not change a comparison of two models. [`StrengthDivergence`](@ref) adds the term. Under [`FixedStrength`](@ref) the term is zero, so `StrengthDivergence()` adds zero.

# Mathematical definition

```math
\\begin{align}
\\mathbf{P} &= \\mathbf{X} \\mathbf{S}^{-1}\\,,\\\\
\\hat{\\boldsymbol{z}} &= \\underset{\\boldsymbol{z}}{\\arg\\min} \\left(\\boldsymbol{y} - \\mathbf{P} \\boldsymbol{z}\\right)^\\intercal \\mathbf{V} \\left(\\boldsymbol{y} - \\mathbf{P} \\boldsymbol{z}\\right) + \\gamma \\sum_{k \\in \\mathcal{K}} \\left( 2 \\vartheta \\lvert z_k \\rvert + (1 - \\vartheta) z_k^2 \\right)\\,,\\\\
\\hat{\\boldsymbol{b}} &= \\mathbf{S}^{-1} \\hat{\\boldsymbol{z}}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{X}``: Design matrix of the caller, one row per observation and one column per regressor.
  - ``\\mathbf{S}``: Diagonal matrix of the standard deviations of the columns of ``\\mathbf{X}`` under `ve`, with one in place of the intercept column and of a column of zero standard deviation. It is the identity when `ve` is `nothing`.
  - $(math_dict[:P_enet])
  - $(math_dict[:y_enet])
  - $(math_dict[:z_enet])
  - $(math_dict[:V_enet]) Its diagonal holds the frequency weights of [`glm_weights`](@ref).
  - $(math_dict[:K_enet])
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - ``\\hat{\\boldsymbol{b}}``: Coefficients of the fit on the columns of ``\\mathbf{X}``, which `StatsAPI.coef` returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ElasticNet(;
        theta::Real = 0.99,
        strength::AbstractPenaltyStrength = MiddleOfPath(),
        sdiv::Option{<:AbstractStrengthDivergence} = nothing,
        ve::Option{<:AbstractVarianceEstimator} = nothing,
        iters::Integer = 10_000,
        tol::Real = 1e-10,
        w::Option{<:ObsWeights} = nothing,
        ic::Option{<:Integer} = nothing
    ) -> ElasticNet

Keywords correspond to the struct's fields. The constructor passes `w` to `ve` through [`factory`](@ref).

## Validation

  - `0 <= theta <= 1`. A `DomainError` is thrown otherwise.
  - `theta > 0` when `strength` is a [`MiddleOfPath`](@ref). A `DomainError` is thrown otherwise, because a ridge alone has no strength above which every coefficient is zero.
  - `iters >= 1`. A `DomainError` is thrown otherwise.
  - `tol > 0`. A `DomainError` is thrown otherwise.
  - If `w` is a vector, `!isempty(w)`. An `IsEmptyError` is thrown otherwise.
  - If `ic` is not `nothing`, `ic >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ElasticNet()
ElasticNet
     theta ┼ Float64: 0.99
  strength ┼ MiddleOfPath
           │   ratio ┴ Float64: 0.001
      sdiv ┼ nothing
        ve ┼ nothing
     iters ┼ Int64: 10000
       tol ┼ Float64: 1.0e-10
         w ┼ nothing
        ic ┴ nothing

julia> X = [1.0 0.5; 1.0 -0.2; 1.0 0.9; 1.0 0.1];

julia> y = [1.2, 0.4, 1.9, 0.6];

julia> tgt = factory(ElasticNet(; strength = FixedStrength(; gamma = 1e6)), InterceptColumn(1));

julia> m = PortfolioOptimisers.StatsAPI.fit(tgt, X, y);

julia> PortfolioOptimisers.StatsAPI.coef(m) ≈ [sum(y) / 4, 0.0]
true
```

# Related

  - [`AbstractPenalisedRegressionTarget`](@ref)
  - [`AbstractPenaltyStrength`](@ref)
  - [`MiddleOfPath`](@ref)
  - [`FixedStrength`](@ref)
  - [`InterceptColumn`](@ref)
  - [`LinearModel`](@ref)
  - [`KernelTrendPattern`](@ref)
  - [`StrengthDivergence`](@ref)
  - [`ElasticNetFit`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)

# References

  - $(ref_dict[:friedman2010]) Section 2.5, Equations (1), (2) and (10).
  - $(ref_dict[:zouhastie2005]) Equation (3).
  - $(ref_dict[:lai2018ktpt]) Equation (15).
  - $(ref_dict[:tibshirani2012]) Remark (The elastic net) after Equation (28).
"""
@concrete struct ElasticNet <: AbstractPenalisedRegressionTarget
    """
    The share of the ``L_1`` penalty, `1` for the lasso and `0` for the ridge.
    """
    theta
    """
    The rule that sets the regularisation strength.
    """
    strength
    """
    The rule that adds the term of a strength that reads the response to the effective degrees of freedom of the fit, or `nothing` to leave the term out.
    """
    sdiv
    """
    $(field_dict[:ve]) It standardises each penalised column, or `nothing` to fit the columns as they are.
    """
    ve
    """
    Maximum number of coordinate-descent sweeps. Each sweep ends with the exact solve on its sign pattern.
    """
    iters
    """
    Tolerance on the largest change of a coefficient over one sweep. The sweeps stop below it when no exact solve has given the optimum.
    """
    tol
    """
    $(field_dict[:ow])
    """
    w
    """
    The index of the column of ones that the estimator names with [`InterceptColumn`](@ref), or `nothing`. The fit does not penalise or standardise that column.
    """
    ic
    function ElasticNet(theta::Real, strength::AbstractPenaltyStrength,
                        sdiv::Option{<:AbstractStrengthDivergence},
                        ve::Option{<:AbstractVarianceEstimator}, iters::Integer, tol::Real,
                        w::Option{<:ObsWeights}, ic::Option{<:Integer})
        @argcheck(zero(theta) <= theta <= one(theta),
                  DomainError(theta, "theta must be in [0, 1]"))
        @argcheck(!(iszero(theta) && isa(strength, MiddleOfPath)),
                  DomainError(theta,
                              "theta = 0 is a ridge alone, which has no strength above which every coefficient is zero, so MiddleOfPath has no path to read. State theta > 0, or a FixedStrength"))
        @argcheck(iters >= 1, DomainError(iters, "iters must be at least 1"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be positive"))
        if isa(w, AbstractVector)
            @argcheck(!isempty(w), IsEmptyError)
        end
        if !isnothing(ic)
            @argcheck(ic >= 1, DomainError(ic, "ic must be at least 1"))
        end
        fs = (theta, strength, sdiv, ve, iters, tol, w, ic)
        return new{map(typeof, fs)...}(fs...)
    end
end
function ElasticNet(; theta::Real = 0.99,
                    strength::AbstractPenaltyStrength = MiddleOfPath(),
                    sdiv::Option{<:AbstractStrengthDivergence} = nothing,
                    ve::Option{<:AbstractVarianceEstimator} = nothing,
                    iters::Integer = 10_000, tol::Real = 1e-10,
                    w::Option{<:ObsWeights} = nothing,
                    ic::Option{<:Integer} = nothing)::ElasticNet
    return ElasticNet(theta, strength, sdiv, factory(ve, w), iters, tol, w, ic)
end
"""
    factory(tgt::ElasticNet, w::ObsWeights) -> ElasticNet

Return a new [`ElasticNet`](@ref) whose fit weighs the observations by `w`.

# Algorithm

 1. Build a new [`ElasticNet`](@ref) with `w` in place of `tgt.w` and every other field unchanged. Its keyword constructor passes `w` to `tgt.ve` through [`factory`](@ref), so the standardisation reads the same weights.

# Arguments

  - `tgt`: The regression target.
  - $(arg_dict[:ow])

# Returns

  - `tgt::ElasticNet`: A new regression target that holds the weights.

# Related

  - [`ElasticNet`](@ref)
  - [`factory`](@ref)
  - [`regression_target_weights(::ElasticNet)`](@ref)
"""
function factory(tgt::ElasticNet, w::ObsWeights)::ElasticNet
    return ElasticNet(; theta = tgt.theta, strength = tgt.strength, sdiv = tgt.sdiv,
                      ve = tgt.ve, iters = tgt.iters, tol = tgt.tol, w = w, ic = tgt.ic)
end
"""
    factory(tgt::ElasticNet, ic::InterceptColumn) -> ElasticNet

Return a new [`ElasticNet`](@ref) whose fit leaves the column of ones that an estimator names out of the penalty.

# Algorithm

 1. Build a new [`ElasticNet`](@ref) with `ic.k` in place of `tgt.ic` and every other field unchanged.

# Arguments

  - `tgt`: The regression target.
  - `ic`: The intercept column.

# Returns

  - `tgt::ElasticNet`: A new regression target that does not penalise or standardise column `ic.k`.

# Related

  - [`ElasticNet`](@ref)
  - [`InterceptColumn`](@ref)
  - [`factory(::AbstractRegressionTarget, ::InterceptColumn)`](@ref)
"""
function factory(tgt::ElasticNet, ic::InterceptColumn)::ElasticNet
    return ElasticNet(; theta = tgt.theta, strength = tgt.strength, sdiv = tgt.sdiv,
                      ve = tgt.ve, iters = tgt.iters, tol = tgt.tol, w = tgt.w, ic = ic.k)
end
"""
    regression_target_weights(tgt::ElasticNet)

Return the observation weights that an [`ElasticNet`](@ref) carries into its fit, or `nothing`.

The method returns `tgt.w` as it is, so it does not resolve a [`DynamicAbstractWeights`](@ref) against a design.

# Arguments

  - `tgt`: The regression target.

# Returns

  - `w::Option{ObsWeights}`: `tgt.w`.

# Related

  - [`ElasticNet`](@ref)
  - [`regression_target_weights`](@ref)
  - [`factory(::ElasticNet, ::ObsWeights)`](@ref)
"""
function regression_target_weights(tgt::ElasticNet)
    return tgt.w
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the coefficients of a fitted [`ElasticNet`](@ref), the strength at which the fit ran, and the counts that its scores read.

[`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref) returns it. `StatsAPI.coef` returns `b`, and `StatsAPI.predict` multiplies a design by `b`. `StatsAPI.dof`, `StatsAPI.nobs`, `StatsAPI.deviance`, `StatsAPI.loglikelihood`, `StatsAPI.aic`, `StatsAPI.aicc` and `StatsAPI.bic` read the counts, in the forms that `GLM` uses for a fitted [`LinearModel`](@ref). [`residual_count`](@ref) returns `rdof`.

# Mathematical definition

```math
\\begin{align}
\\mathbf{H} &= \\mathbf{P}_{\\mathcal{A}} \\left(\\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\mathbf{P}_{\\mathcal{A}} + \\gamma (1 - \\vartheta) \\mathbf{D}_{\\mathcal{A}}\\right)^{-1} \\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V}\\,,\\\\
\\widehat{\\mathrm{df}} &= \\operatorname{tr}(\\mathbf{H}) + \\delta\\,,\\\\
\\nu &= T - \\operatorname{tr}\\left(2 \\mathbf{S} - \\mathbf{S}^\\intercal \\mathbf{S}\\right)\\,,\\quad \\mathbf{S} = \\mathbf{V}^{1/2} \\mathbf{H} \\mathbf{V}^{-1/2}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{H}``: Smoother of the fit on its active set. The fitted response is ``\\mathbf{H} \\boldsymbol{y}`` while the active set and the strength do not move.
  - ``\\mathbf{P}_{\\mathcal{A}}``: Columns of the active set ``\\mathcal{A}``, the columns whose coefficient is nonzero and the intercept column.
  - ``\\mathbf{D}_{\\mathcal{A}}``: Diagonal matrix with one for each penalised column of the active set and zero for the intercept column.
  - $(math_dict[:V_enet])
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - ``\\widehat{\\mathrm{df}}``: Effective degrees of freedom of the fit, `df`.
  - ``\\delta``: Term of a strength that reads the response, which the `sdiv` field of the target adds through [`strength_divergence`](@ref). It is zero under `sdiv = nothing`.
  - ``\\nu``: Residual count, `rdof`.
  - ``\\mathbf{S}``: Smoother of the weighted rows. It is ``\\mathbf{H}`` itself without weights, and it is symmetric.
  - $(math_dict[:T])

Remark (The elastic net) after Equation (28) of [tibshirani2012](@cite) states the trace of ``\\mathbf{H}`` as an unbiased estimate of the degrees of freedom at a fixed strength. Remark (The lasso with intercept) after it extends the result to an unpenalised intercept.

The residual count follows from the mean of the weighted residual sum of squares. With ``\\boldsymbol{y} = \\boldsymbol{\\mu} + \\boldsymbol{\\varepsilon}`` and ``\\mathbf{V}^{1/2} \\boldsymbol{\\varepsilon}`` of covariance ``\\sigma^{2} \\mathbf{I}``,

```math
\\begin{align}
\\mathbb{E}\\left[\\mathrm{RSS}\\right] &= \\sigma^{2} \\operatorname{tr}\\left((\\mathbf{I} - \\mathbf{S})^\\intercal (\\mathbf{I} - \\mathbf{S})\\right) + \\lVert \\mathbf{V}^{1/2} (\\mathbf{I} - \\mathbf{H}) \\boldsymbol{\\mu} \\rVert^{2}\\,.
\\end{align}
```

So when the bias term is small, ``\\mathrm{RSS} / \\sigma^{2}`` has the mean ``\\nu``, and ``\\nu`` is the count of the chi-squared law that [`ResidualInflation`](@ref) reads. For a projection, ``\\mathbf{S}^\\intercal \\mathbf{S} = \\mathbf{S}``, and ``\\nu = T - p``, the count of a fitted [`LinearModel`](@ref) with ``p`` columns. A ridge part shrinks ``\\mathbf{S}``, so ``\\nu`` is below ``T - \\operatorname{tr}(\\mathbf{H})``, and the count ``T - \\operatorname{tr}(\\mathbf{H})`` overstates the residual count. The term ``\\delta`` does not enter ``\\nu``, because no source states the residual law of a fit whose strength reads the response.

The scores take the residual variance at its maximum-likelihood value ``\\mathrm{RSS} / n``, with ``n`` the sum of the frequency weights of [`glm_weights`](@ref). `StatsAPI.dof` adds one to ``\\widehat{\\mathrm{df}}`` for that variance, as `GLM` adds one to the rank of a fitted [`LinearModel`](@ref). Equation (2.5) of [hurvich1998](@cite) states AICc of a linear smoother with ``\\operatorname{tr}(\\mathbf{H})`` in place of the number of columns, and `StatsAPI.aicc` with this `dof` is ``n`` times that form plus a constant. Equation (2.16) of [zou2007](@cite) proposes BIC with the effective degrees of freedom for the lasso, and does not derive it.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ElasticNet`](@ref)
  - [`StatsAPI.coef(::ElasticNetFit)`](@ref)
  - [`StatsAPI.predict(::ElasticNetFit, ::MatNum)`](@ref)
  - [`StatsAPI.dof(::ElasticNetFit)`](@ref)
  - [`residual_count`](@ref)
  - [`elastic_net_counts`](@ref)
  - [`StepwiseRegression`](@ref)

# References

  - $(ref_dict[:tibshirani2012]) Remark (The elastic net) after Equation (28), and Remark (The lasso with intercept).
  - $(ref_dict[:hurvich1998]) Equation (2.5). Page 273 cites Section 3.5 of [hastie1990](@cite) for the effective degrees of freedom of a linear smoother, and the residual count above follows the derivation of this docstring. $(ref_dict[:no_original_source])
  - $(ref_dict[:zou2007]) Equation (2.16).
  - $(ref_dict[:hastie2009]) Section 7.6, Equation (7.32).
"""
@concrete struct ElasticNetFit <: AbstractResult
    """
    The coefficients on the columns of the design of the caller, one per column.
    """
    b
    """
    The regularisation strength ``\\gamma`` that the strength rule gave, on the standardised columns when the target has a `ve`.
    """
    gamma
    """
    The effective degrees of freedom of the fit, the trace of the smoother on the active set plus the term of `sdiv`.
    """
    df
    """
    The weighted residual sum of squares of the fit, under the frequency weights of [`glm_weights`](@ref).
    """
    rss
    """
    The number of observations, the sum of the frequency weights, or the number of rows without weights.
    """
    nobs
    """
    The residual count, the number of rows less ``\\operatorname{tr}(2 \\mathbf{S} - \\mathbf{S}^\\intercal \\mathbf{S})``.
    """
    rdof
end
"""
    StatsAPI.dof(m::ElasticNetFit) -> Real
    StatsAPI.nobs(m::ElasticNetFit) -> Real
    StatsAPI.deviance(m::ElasticNetFit) -> Real
    StatsAPI.loglikelihood(m::ElasticNetFit) -> Real
    StatsAPI.aic(m::ElasticNetFit) -> Real
    StatsAPI.aicc(m::ElasticNetFit) -> Real
    StatsAPI.bic(m::ElasticNetFit) -> Real

Return the counts and the scores of a fitted [`ElasticNet`](@ref).

[`ElasticNetFit`](@ref) is not a `StatsAPI.StatisticalModel`, so each score has a method of its own, in the form of `StatsAPI`. The forms match `GLM` for a fitted [`LinearModel`](@ref) under the same weights, with the effective degrees of freedom in place of the rank.

# Mathematical definition

```math
\\begin{align}
k &= \\widehat{\\mathrm{df}} + 1\\,,\\\\
\\ln \\hat{L} &= -\\frac{n}{2} \\left(\\ln \\frac{2 \\pi\\, \\mathrm{RSS}}{n} + 1\\right)\\,,\\\\
\\mathrm{AIC} &= 2k - 2\\ln\\hat{L}\\,,\\\\
\\mathrm{AICC} &= \\mathrm{AIC} + \\frac{2k(k+1)}{n - k - 1}\\,,\\\\
\\mathrm{BIC} &= k\\ln n - 2\\ln\\hat{L}\\,.
\\end{align}
```

Where:

  - ``k``: `StatsAPI.dof`, the effective degrees of freedom plus one for the residual variance.
  - ``\\widehat{\\mathrm{df}}``: `m.df`.
  - ``\\hat{L}``: Maximum of the Gaussian likelihood, at the residual variance ``\\mathrm{RSS} / n``.
  - ``\\mathrm{RSS}``: `StatsAPI.deviance`, the weighted residual sum of squares `m.rss`.
  - ``n``: `StatsAPI.nobs`, `m.nobs`.

Equation (2.5) of [hurvich1998](@cite) states AICc with the residual variance estimated, and `StatsAPI.aicc` is ``n`` times that form plus ``n \\ln 2\\pi``. Equation (2.16) of [zou2007](@cite) writes BIC with a known error variance ``\\sigma^{2}``, as ``\\mathrm{RSS} / (n \\sigma^{2}) + \\ln(n)\\, \\widehat{\\mathrm{df}} / n``. The form here puts the maximum-likelihood estimate ``\\mathrm{RSS} / n`` in place of ``\\sigma^{2}``, as `GLM` does for a fitted [`LinearModel`](@ref).

# Arguments

  - `m`: The fitted model.

# Returns

  - `x::Real`: The count or the score.

# Related

  - [`ElasticNetFit`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`regression_criterion_func`](@ref)

# References

  - $(ref_dict[:hurvich1998]) Equation (2.5).
  - $(ref_dict[:zou2007]) Equation (2.16).
"""
function StatsAPI.dof(m::ElasticNetFit)
    return m.df + 1
end
function StatsAPI.nobs(m::ElasticNetFit)
    return m.nobs
end
function StatsAPI.deviance(m::ElasticNetFit)
    return m.rss
end
function StatsAPI.loglikelihood(m::ElasticNetFit)
    n = m.nobs
    return -n / 2 * (log(2 * pi * m.rss / n) + 1)
end
function StatsAPI.aic(m::ElasticNetFit)
    return -2 * StatsAPI.loglikelihood(m) + 2 * StatsAPI.dof(m)
end
function StatsAPI.aicc(m::ElasticNetFit)
    k = StatsAPI.dof(m)
    n = m.nobs
    return StatsAPI.aic(m) + 2 * k * (k + 1) / (n - k - 1)
end
function StatsAPI.bic(m::ElasticNetFit)
    return -2 * StatsAPI.loglikelihood(m) + StatsAPI.dof(m) * log(m.nobs)
end
"""
    residual_count(m::ElasticNetFit, X::MatNum) -> Real

Return the residual count of a fitted [`ElasticNet`](@ref), `m.rdof`.

[`ElasticNetFit`](@ref) states the count and its derivation. The design does not change it, because the fit already holds the count of its own design.

# Arguments

  - `m`: The fitted model.
  - `X`: The design of the fit, which the method does not read.

# Returns

  - `nu::Real`: `m.rdof`.

# Related

  - [`residual_count`](@ref)
  - [`ElasticNetFit`](@ref)
"""
function residual_count(m::ElasticNetFit, ::MatNum)
    return m.rdof
end
"""
    StatsAPI.coef(m::ElasticNetFit) -> VecNum

Return the coefficients of a fitted [`ElasticNet`](@ref), one per column of the design.

# Arguments

  - `m`: The fitted model.

# Returns

  - `b::VecNum`: `m.b`.

# Related

  - [`ElasticNetFit`](@ref)
  - [`StatsAPI.predict(::ElasticNetFit, ::MatNum)`](@ref)
"""
function StatsAPI.coef(m::ElasticNetFit)
    return m.b
end
"""
    StatsAPI.predict(m::ElasticNetFit, X::MatNum) -> VecNum

Return the fitted response of a fitted [`ElasticNet`](@ref) on a design.

# Arguments

  - `m`: The fitted model.
  - `X`: The design matrix, `observations × regressors`, with the columns of the design of the fit.

# Returns

  - `yhat::VecNum`: `X * m.b`, one entry per row of `X`.

# Related

  - [`ElasticNetFit`](@ref)
  - [`StatsAPI.coef(::ElasticNetFit)`](@ref)
"""
function StatsAPI.predict(m::ElasticNetFit, X::MatNum)
    return X * m.b
end
"""
    elastic_net_design(tgt::ElasticNet, X::MatNum) -> Tuple

Return the standardised design of an [`ElasticNet`](@ref), the design with each row weighted, the scale of the columns, and the weights of the rows.

# Algorithm

 1. When `tgt.ve` is `nothing`, set the scale `s` to `nothing`. Otherwise take the standard deviation of each column of `X` under `tgt.ve`, and put one in place of a zero and in place of the intercept column `tgt.ic`. That is `s`.
 2. Divide each column of `X` by its entry of `s`, which gives `P`. When `s` is `nothing`, `P` is `X`.
 3. Resolve `tgt.w` against `X` with [`get_observation_weights`](@ref), and scale it with [`glm_weights`](@ref), which gives `v`. When `v` is `nothing`, `Q` is `P`. Otherwise multiply each row of `P` by its entry of `v`, which gives `Q`.

# Arguments

  - `tgt`: The regression target.
  - `X`: The design matrix `observations × regressors`.

# Returns

  - `(P, Q, s, v)::Tuple`: The standardised design ``\\mathbf{P}``, the weighted design ``\\mathbf{V} \\mathbf{P}``, the scale of the columns or `nothing`, and the frequency weights of the rows or `nothing`.

# Related

  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)
  - [`glm_weights`](@ref)
"""
function elastic_net_design(tgt::ElasticNet, X::MatNum)
    s = if isnothing(tgt.ve)
        nothing
    else
        sd = map(v -> iszero(v) ? one(v) : v, vec(Statistics.std(tgt.ve, X; dims = 1)))
        if !isnothing(tgt.ic)
            sd[tgt.ic] = one(eltype(sd))
        end
        sd
    end
    P = isnothing(s) ? X : X ./ transpose(s)
    v = glm_weights(get_observation_weights(tgt.w, X))
    Q = isnothing(v) ? P : v .* P
    return P, Q, s, v
end
"""
    StatsAPI.fit(tgt::ElasticNet, X::MatNum, y::VecNum) -> ElasticNetFit

Fit an [`ElasticNet`](@ref) to the design `X` and the response `y` by cyclic coordinate descent.

[`ElasticNet`](@ref) states the objective. The fit solves it at one strength, which the strength rule gives. A design with few columns does not need the rest of the path as warm starts.

# Algorithm

 1. Build the standardised design `P`, the weighted design `Q`, the scale `s` and the weights `v` with [`elastic_net_design`](@ref).
 2. Mark the penalised columns in `pen`: every column but `tgt.ic`. Compute `nrm2`, the weighted squared norm of each column.
 3. Start from the coefficients `z` at zero and the residual `r` equal to `y`. With an intercept column, fit that column alone first, and take its residual as `r`.
 4. Compute `g`, the product of each penalised column of `Q` with `r`, the strength `lam` with [`penalty_strength`](@ref), and the ridge term `ridge` of each column, zero at the intercept column.
 5. When every entry of `g` is zero, the start point is the optimum. Otherwise compute the threshold `a` of each column, zero at the intercept column, and run [`elastic_net_sweeps`](@ref) from the start point.
 6. Compute the counts of the fit with [`elastic_net_counts`](@ref).
 7. Divide the coefficients by `s`, and return them with `lam` and the counts in an [`ElasticNetFit`](@ref).

# Arguments

  - `tgt`: The regression target.
  - `X`: The design matrix `observations × regressors`.
  - `y`: The response vector.

# Returns

  - `m::ElasticNetFit`: The fitted model.

# Related

  - [`ElasticNet`](@ref)
  - [`ElasticNetFit`](@ref)
  - [`elastic_net_design`](@ref)
  - [`elastic_net_sweeps`](@ref)
  - [`elastic_net_counts`](@ref)
"""
function StatsAPI.fit(tgt::ElasticNet, X::MatNum, y::VecNum)
    P, Q, s, v = elastic_net_design(tgt, X)
    n = size(P, 2)
    pen = trues(n)
    if !isnothing(tgt.ic)
        pen[tgt.ic] = false
    end
    nrm2 = vec(sum(P .* Q; dims = 1))
    z = zeros(float_if_integer(promote_type(eltype(Q), eltype(y))), n)
    r = copyto!(similar(y, eltype(z)), y)
    if !isnothing(tgt.ic)
        z[tgt.ic] = LinearAlgebra.dot(view(Q, :, tgt.ic), y) / nrm2[tgt.ic]
        r .-= z[tgt.ic] .* view(P, :, tgt.ic)
    end
    g = [LinearAlgebra.dot(view(Q, :, k), r) for k in findall(pen)]
    lam = penalty_strength(tgt.strength, g, tgt.theta)
    ridge = (lam * (one(tgt.theta) - tgt.theta)) .* pen
    if !all(iszero, g)
        a = (lam * tgt.theta) .* pen
        z = elastic_net_sweeps(tgt, P, Q, y, z, r, nrm2, a, ridge, pen)
    end
    cnt = elastic_net_counts(tgt,
                             (; P = P, Q = Q, y = y, z = z, pen = pen, ridge = ridge, g = g,
                              lam = lam, nrm2 = nrm2, w = v))
    return ElasticNetFit(isnothing(s) ? z : z ./ s, lam, cnt.df, cnt.rss, cnt.nobs,
                         cnt.rdof)
end
"""
    elastic_net_counts(tgt::ElasticNet, sys::NamedTuple) -> NamedTuple

Return the effective degrees of freedom, the weighted residual sum of squares, the number of observations and the residual count of a fitted [`ElasticNet`](@ref).

[`ElasticNetFit`](@ref) states the counts and their derivation. The method reads them off the equations of the active set that [`elastic_net_polish`](@ref) solves, so it needs the pseudo-inverse of one matrix with one row per active column.

# Mathematical definition

```math
\\begin{align}
\\mathbf{B} &= \\mathbf{M}^{-1} \\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\mathbf{P}_{\\mathcal{A}}\\,,\\quad \\mathbf{M} = \\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\mathbf{P}_{\\mathcal{A}} + \\gamma (1 - \\vartheta) \\mathbf{D}_{\\mathcal{A}}\\,,\\\\
\\operatorname{tr}(\\mathbf{H}) &= \\operatorname{tr}(\\mathbf{B})\\,,\\quad \\operatorname{tr}(\\mathbf{S}^\\intercal \\mathbf{S}) = \\operatorname{tr}(\\mathbf{B}^{2})\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``: Matrix with one row and one column per active column. It has the nonzero eigenvalues of ``\\mathbf{H}``.
  - ``\\mathbf{M}``: Matrix of the equations of the active set.
  - ``\\mathbf{H}``, ``\\mathbf{S}``: Smoother of the fit and smoother of the weighted rows, as [`ElasticNetFit`](@ref) states them.
  - ``\\mathbf{P}_{\\mathcal{A}}``: Columns of the active set.
  - ``\\mathbf{D}_{\\mathcal{A}}``: Diagonal matrix with one for each penalised column of the active set and zero for the intercept column.
  - $(math_dict[:V_enet])
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])

The trace is invariant under a cyclic permutation, so ``\\operatorname{tr}(\\mathbf{H}) = \\operatorname{tr}(\\mathbf{B})``. ``\\mathbf{S}`` is symmetric, and ``\\mathbf{S}^{2}`` has the trace of ``\\mathbf{B}^{2}`` for the same reason.

# Algorithm

 1. Find `A`, the indices of the nonzero coefficients of `sys.z` and of the column that `sys.pen` does not penalise.
 2. Build the matrix of the equations of the active set, and take its pseudo-inverse `Mi`. For `theta < 1` the matrix is positive definite, and `Mi` is its inverse. At `theta = 1` the matrix is singular when the active set holds more columns than there are observations, and `B` is then the projection onto the row space of the active columns, whose trace is their rank. An LU solve of such a matrix does not always fail, and it returns rounding noise in place of the rank.
 3. Multiply `Mi` by the weighted Gram matrix of the active set, which gives `B`. Take `tH`, the trace of `B`.
 4. Add the term of `tgt.sdiv` to `tH` with [`strength_divergence`](@ref), which gives `df`.
 5. Compute the weighted residual sum of squares `rss` and the number of observations `nobs` under `sys.w`.
 6. Take the residual count `rdof`, the number of rows less `2 tH - tr(B^2)`.

# Arguments

  - `tgt`: The regression target.
  - `sys`: The fit, as the `# Interfaces` section of [`AbstractStrengthDivergence`](@ref) states it, without `A` and `Mi`.

# Returns

  - `(; df, rss, nobs, rdof)::NamedTuple`: The counts that [`ElasticNetFit`](@ref) holds.

# Related

  - [`ElasticNetFit`](@ref)
  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)
  - [`strength_divergence`](@ref)
  - [`elastic_net_polish`](@ref)
"""
function elastic_net_counts(tgt::ElasticNet, sys::NamedTuple)
    (; P, Q, y, z, pen, ridge, w) = sys
    A = findall(.!iszero.(z) .| .!pen)
    G = transpose(view(Q, :, A)) * view(P, :, A)
    Mi = LinearAlgebra.pinv(G + LinearAlgebra.Diagonal(ridge[A]))
    B = Mi * G
    tH = LinearAlgebra.tr(B)
    df = tH + strength_divergence(tgt.sdiv, tgt, (; sys..., A = A, Mi = Mi))
    res = y - P * z
    rss = isnothing(w) ? sum(abs2, res) : LinearAlgebra.dot(w, abs2.(res))
    nobs = isnothing(w) ? size(P, 1) : sum(w)
    rdof = size(P, 1) - (2 * tH - LinearAlgebra.dot(B, transpose(B)))
    return (; df = df, rss = rss, nobs = nobs, rdof = rdof)
end
"""
    strength_divergence(sdiv::Nothing, tgt::ElasticNet, sys::NamedTuple) -> Real
    strength_divergence(sdiv::StrengthDivergence, tgt::ElasticNet, sys::NamedTuple) -> Real

Return the term that a strength rule which reads the response adds to the effective degrees of freedom of an [`ElasticNet`](@ref).

[`StrengthDivergence`](@ref) states the term. The `# Interfaces` section of [`AbstractStrengthDivergence`](@ref) states `sys` and the method that a rule of the caller adds.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: return zero, so the count is the trace of the smoother.
 2. [`StrengthDivergence`](@ref): take `dg`, the gradient of the strength with [`penalty_strength_gradient`](@ref). Multiply the pseudo-inverse `sys.Mi` by `(1 - theta) z + theta sign(z)` on the penalised columns of the active set, zero at the intercept column, and negate it, which gives the derivative of the active coefficients in the strength. Multiply the active columns of `P` by it, which gives `v`, the derivative of the fitted response. With an intercept column, remove from `v` its weighted projection onto that column. Return the product of `dg` with the penalised columns of `Q` and `v`.

# Arguments

  - `sdiv`: The rule, or `nothing`.
  - `tgt`: The regression target.
  - `sys`: The fit.

# Returns

  - `t::Real`: The term, ``\\nabla_{\\boldsymbol{y}} \\gamma^{\\intercal} \\partial \\hat{\\boldsymbol{\\mu}} / \\partial \\gamma``.

# Related

  - [`AbstractStrengthDivergence`](@ref)
  - [`StrengthDivergence`](@ref)
  - [`elastic_net_counts`](@ref)
  - [`penalty_strength_gradient`](@ref)
"""
function strength_divergence(::Nothing, ::ElasticNet, sys::NamedTuple)
    return zero(eltype(sys.z))
end
function strength_divergence(::StrengthDivergence, tgt::ElasticNet, sys::NamedTuple)
    (; P, Q, z, pen, g, nrm2, A, Mi) = sys
    dg = penalty_strength_gradient(tgt.strength, g, tgt.theta)
    zA = z[A]
    pA = pen[A]
    dz = -(Mi * (((one(tgt.theta) - tgt.theta) .* zA .+ tgt.theta .* sign.(zA)) .* pA))
    v = view(P, :, A) * dz
    if !isnothing(tgt.ic)
        v .-= (LinearAlgebra.dot(view(Q, :, tgt.ic), v) / nrm2[tgt.ic]) .*
              view(P, :, tgt.ic)
    end
    return LinearAlgebra.dot(dg, transpose(view(Q, :, pen)) * v)
end
"""
    elastic_net_sweeps(tgt::ElasticNet, P::MatNum, Q::MatNum, y::VecNum, z::VecNum,
                       r::VecNum, nrm2::VecNum, a::VecNum, ridge::VecNum,
                       pen::AbstractVector{Bool}) -> VecNum

Return the coefficients of an [`ElasticNet`](@ref) by cyclic coordinate descent from `z`, with an exact solve at the end of each sweep.

# Mathematical definition

The minimiser of the objective of [`ElasticNet`](@ref) over one coefficient, with the others fixed, is

```math
\\begin{align}
z_k &= \\frac{S\\left(\\mathbf{P}_k^\\intercal \\mathbf{V} \\boldsymbol{r}_k,\\ a_k\\right)}{\\mathbf{P}_k^\\intercal \\mathbf{V} \\mathbf{P}_k + d_k}\\,,\\quad S(v, a) = \\operatorname{sign}(v) \\max(\\lvert v \\rvert - a, 0)\\,,\\\\
a_k &= \\gamma \\vartheta\\, \\mathbb{1}[k \\in \\mathcal{K}]\\,,\\quad d_k = \\gamma (1 - \\vartheta)\\, \\mathbb{1}[k \\in \\mathcal{K}]\\,.
\\end{align}
```

Where:

  - $(math_dict[:z_enet])
  - $(math_dict[:P_enet])
  - $(math_dict[:V_enet])
  - ``\\boldsymbol{r}_k``: Residual of the response ``\\boldsymbol{y}`` without the term of column ``k``.
  - ``S``: Soft threshold.
  - ``a_k``, ``d_k``: Threshold and ridge term of column ``k``, both zero at the intercept column.
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - $(math_dict[:K_enet])

At the intercept column the update is the least-squares coefficient of that column on the residual.

# Algorithm

 1. Take `den`, the denominator of each coordinate update.
 2. Sweep the columns in order. For each column `k`, compute `rho`, the product of the column of `Q` with the residual without its own term, and apply the coordinate update, which gives `znew`. Update `r`, and record `delta`, the largest change of a coefficient in the sweep.
 3. Try the exact solve of [`elastic_net_polish`](@ref) on the sign pattern of `z`. When it gives the optimum, return it.
 4. Repeat steps 2 and 3 until `delta` is below `tgt.tol` or `tgt.iters` sweeps have run, and then return `z`.

# Arguments

  - `tgt`: The regression target, which gives `iters` and `tol`.
  - `P`: The standardised design, `observations × columns`.
  - `Q`: The weighted design, ``\\mathbf{V} \\mathbf{P}``.
  - `y`: The response.
  - `z`: The coefficients of the start point. The function changes them.
  - `r`: The residual of the start point. The function changes it.
  - `nrm2`: The weighted squared norm of each column.
  - `a`: The threshold of each column.
  - `ridge`: The ridge term of each column.
  - `pen`: Whether each column is penalised.

# Returns

  - `z::VecNum`: The coefficients on the columns of `P`.

# Related

  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)
  - [`elastic_net_polish`](@ref)
"""
function elastic_net_sweeps(tgt::ElasticNet, P::MatNum, Q::MatNum, y::VecNum, z::VecNum,
                            r::VecNum, nrm2::VecNum, a::VecNum, ridge::VecNum,
                            pen::AbstractVector{Bool})
    den = nrm2 .+ ridge
    for _ in 1:(tgt.iters)
        delta = zero(eltype(z))
        for k in eachindex(z)
            rho = LinearAlgebra.dot(view(Q, :, k), r) + nrm2[k] * z[k]
            znew = sign(rho) * max(abs(rho) - a[k], zero(a[k])) / den[k]
            if znew != z[k]
                r .-= (znew - z[k]) .* view(P, :, k)
                delta = max(delta, abs(znew - z[k]))
                z[k] = znew
            end
        end
        zp = elastic_net_polish(P, Q, y, z, a, ridge, pen)
        if !isnothing(zp)
            return zp
        end
        delta < tgt.tol && break
    end
    return z
end
"""
    elastic_net_polish(P::MatNum, Q::MatNum, y::VecNum, z::VecNum, a::VecNum,
                       ridge::VecNum, pen::AbstractVector{Bool}) -> Option{VecNum}

Returns the exact elastic-net optimum on the sign pattern of `z` when that pattern is the pattern of the optimum, and `nothing` otherwise.

A point that meets the optimality conditions below minimises the objective of [`ElasticNet`](@ref), because the objective is convex. For `theta < 1` the objective is strictly convex in the penalised coefficients, and the point is its only minimiser. At `theta = 1` the objective is the lasso, and it can have more than one minimiser. Then the equations of the active columns are singular when the pattern holds more columns than there are observations, and the function returns `nothing` for such a pattern. The intercept column is always active, and its sign is free.

# Mathematical definition

With ``\\mathcal{A}`` the active columns, the optimum on the sign pattern ``\\boldsymbol{s}`` solves

```math
\\begin{align}
\\left(\\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\mathbf{P}_{\\mathcal{A}} + \\gamma (1 - \\vartheta) \\mathbf{D}_{\\mathcal{A}}\\right) \\boldsymbol{z}_{\\mathcal{A}} &= \\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\boldsymbol{y} - \\gamma \\vartheta\\, \\boldsymbol{s}\\,,
\\end{align}
```

and it is the optimum of the objective when

```math
\\begin{align}
\\operatorname{sign}(z_k) &= s_k\\,,\\quad k \\in \\mathcal{A} \\cap \\mathcal{K}\\,,\\\\
\\lvert \\mathbf{P}_k^\\intercal \\mathbf{V} (\\boldsymbol{y} - \\mathbf{P}_{\\mathcal{A}} \\boldsymbol{z}_{\\mathcal{A}}) \\rvert &\\leq \\gamma \\vartheta\\,,\\quad k \\notin \\mathcal{A}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{A}``: Active set, the columns whose coefficient in ``\\boldsymbol{z}`` is nonzero, and the intercept column.
  - ``\\mathbf{P}_{\\mathcal{A}}``, ``\\boldsymbol{z}_{\\mathcal{A}}``: Columns of the regressors and coefficients of the active set.
  - ``\\mathbf{D}_{\\mathcal{A}}``: Diagonal matrix with one for each penalised column of the active set and zero for the intercept column.
  - ``\\boldsymbol{s}``: Signs of the coefficients of the active set in ``\\boldsymbol{z}``, zero for the intercept column.
  - $(math_dict[:z_enet])
  - $(math_dict[:P_enet])
  - $(math_dict[:V_enet])
  - $(math_dict[:y_enet])
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - $(math_dict[:K_enet])

# Algorithm

 1. Find `active`, the nonzero coefficients of `z` and the column that `pen` does not penalise, and their indices `A`. When `A` is empty, return `nothing`.
 2. Take `s`, the signs of the coefficients of `A`, zero at the intercept column. Factorise the matrix of the equations of the active columns, which gives `F`. When the matrix is singular, return `nothing`.
 3. Solve the equations with `F`, which gives `zA`. When a penalised coefficient of `zA` does not keep its sign in `s`, return `nothing`.
 4. Write `zA` into `zp`, zero elsewhere, and compute `corr`, the product of each column of `Q` with the residual.
 5. When an inactive column has `abs(corr[k]) > a[k]`, return `nothing`. Otherwise return `zp`.

# Arguments

  - `P`: The standardised design, `observations × columns`.
  - `Q`: The weighted design, ``\\mathbf{V} \\mathbf{P}``.
  - `y`: The response.
  - `z`: The coefficients whose sign pattern the function tries.
  - `a`: The threshold of each column, ``\\gamma \\vartheta`` or zero at the intercept column.
  - `ridge`: The ridge term of each column, ``\\gamma (1 - \\vartheta)`` or zero at the intercept column.
  - `pen`: Whether each column is penalised.

# Returns

  - `z'::Option{VecNum}`: The optimum, or `nothing`.

# Related

  - [`elastic_net_sweeps`](@ref)
  - [`ElasticNet`](@ref)
"""
function elastic_net_polish(P::MatNum, Q::MatNum, y::VecNum, z::VecNum, a::VecNum,
                            ridge::VecNum, pen::AbstractVector{Bool})
    active = .!iszero.(z) .| .!pen
    A = findall(active)
    if isempty(A)
        return nothing
    end
    PA = view(P, :, A)
    QA = view(Q, :, A)
    s = sign.(view(z, A)) .* view(pen, A)
    # At `theta = 1` the matrix is singular when the pattern holds more columns than
    # observations, and the pattern is then refused rather than solved.
    F = LinearAlgebra.lu(QA' * PA + LinearAlgebra.Diagonal(view(ridge, A)); check = false)
    if !LinearAlgebra.issuccess(F)
        return nothing
    end
    zA = F \ (QA' * y .- view(a, A) .* s)
    if any(j -> pen[A[j]] && zA[j] * s[j] <= zero(zA[j]), eachindex(zA))
        return nothing
    end
    zp = zero(z)
    zp[A] .= zA
    corr = Q' * (y .- P * zp)
    for k in eachindex(zp)
        if !active[k] && abs(corr[k]) > a[k]
            return nothing
        end
    end
    return zp
end
export ElasticNet, MiddleOfPath, FixedStrength, StrengthDivergence
public AbstractPenaltyStrength, penalty_strength, penalty_strength_gradient
public AbstractStrengthDivergence, strength_divergence
