"""
    abstract type SecondMomentFormulation <: AbstractAlgorithm end

Abstract supertype for optimisation formulations of second moment risk measures in PortfolioOptimisers.jl.

# Related Types

  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
"""
abstract type SecondMomentFormulation <: AbstractAlgorithm end
"""
    abstract type VarianceFormulation <: SecondMomentFormulation end

Abstract supertype for optimisation formulations of variance-based risk measures in PortfolioOptimisers.jl.

# Related Types

  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
abstract type VarianceFormulation <: SecondMomentFormulation end
"""
    struct QuadRiskExpr <: VarianceFormulation end

Direct quadratic risk expression optimisation formulation for variance-like risk measures. The risk measure is implemented using an explicitly quadratic form. This can be in two ways.

# Summary statistics

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} &\\qquad \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N×N` co-moment matrix.

# Scenario-based

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} &\\qquad \\boldsymbol{d} \\cdot \\boldsymbol{d}.\\\\
\\textrm{s.t.} &\\qquad \\boldsymbol{d} \\in \\mathcal{S}_{w}.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` deviations vector.
  - ``\\mathcal{S}_{w}``: Scenario set for portfolio `x`.

# Related Types

  - [`VarianceFormulation`](@ref)
  - [`Variance`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
struct QuadRiskExpr <: VarianceFormulation end
"""
    struct SquaredSOCRiskExpr <: VarianceFormulation end

Squared second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure is implemented using the square of a variable constrained by a second order cone.

# Related

  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`Variance`](@ref)
"""
struct SquaredSOCRiskExpr <: VarianceFormulation end
"""
    struct RSOCRiskExpr <: SecondMomentFormulation end

Rotated second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure using a variable constrained to be in a rotated second order cone representing the sum of squares.

# Related Types

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
struct RSOCRiskExpr <: SecondMomentFormulation end
"""
    struct SOCRiskExpr <: SecondMomentFormulation end

Second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure is implemented using a variable constrained by a second order cone.

# Related

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
"""
struct SOCRiskExpr <: SecondMomentFormulation end
const QuadSecondMomentFormulations = Union{QuadRiskExpr, SquaredSOCRiskExpr, RSOCRiskExpr}
"""
    struct Variance{T1, T2, T3, T4} <: RiskMeasure
        settings::T1
        sigma::T2
        rc::T3
        alg::T4
    end

Represents the portfolio variance using a covariance matrix.

# Fields

  - `settings`: Risk measure configuration.
  - `sigma`: Optional covariance matrix that overrides the prior covariance when provided. Also used to compute the risk represented by a vector.
  - `rc`: Optional specification of risk contribution constraints.
  - `alg`: The optimisation formulation used to represent the variance risk expression.

# Constructors

    Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             sigma::Option{<:MatNum} = nothing,
             rc::Option{<:LcULcE} = nothing,
             alg::VarianceFormulation = SquaredSOCRiskExpr())

Keyword arguments correspond to the fields above.

## Validation

  - If `sigma` is provided, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.

# `JuMP` Formulations

!!! info

    Regardless of the formulation used, an auxiliary variable representing the standard deviation is needed in order to constrain the risk or maximise the risk-adjusted return ratio. This is because quadratic constraints are not strictly convex, and the transformation needed to maximise the risk-adjusted return ratio requires affine variables in the numerator and denominator.

Depending on the `alg` field, the variance risk measure is formulated using `JuMP` as follows:

## `QuadRiskExpr`

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} &\\qquad \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N×N` covariance matrix.

## `SquaredSOCRiskExpr`

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} &\\qquad \\sigma^2\\nonumber\\\\
\\textrm{s.t.} &\\qquad \\left\\lVert \\mathbf{G} \\boldsymbol{w} \\right\\rVert_{2} \\leq \\sigma\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\sigma``: Variable representing the optimised portfolio's standard deviation.
  - ``\\mathbf{G}``: Suitable factorisation of the `N×N` covariance matrix, such as the square root matrix, or the Cholesky factorisation.
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm, which is modelled as a [SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Functor

    (r::Variance)(w::VecNum)

Computes the variance risk of a portfolio with weights `w` using the covariance matrix `r.sigma`.

```math
\\begin{align}
\\mathrm{Variance}(\\boldsymbol{w},\\, \\mathbf{\\Sigma}) &= \\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N×N` covariance matrix.

## Arguments

  - `w::VecNum`: Asset weights.

# Examples

```jldoctest
julia> w = [0.3803452066954233, 0.5900852659955864, 0.029569527308990307];

julia> r = Variance(;
                    sigma = [0.97780 -0.06400 0.84818;
                             -0.06400 3.28564 1.84588;
                             0.84818 1.84588 2.16317])
Variance
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     sigma ┼ 3×3 Matrix{Float64}
        rc ┼ nothing
       alg ┴ SquaredSOCRiskExpr()

julia> r(w)
1.3421705804186579
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`factory(r::Variance, prior::AbstractPriorResult, args...; kwargs...)`](@ref)
  - [`expected_risk`](@ref)
"""
struct Variance{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    sigma::T2
    rc::T3
    alg::T4
    function Variance(settings::RiskMeasureSettings, sigma::Option{<:MatNum},
                      rc::Option{<:LcULcE}, alg::VarianceFormulation)
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma))
            assert_matrix_issquare(sigma, :sigma)
        end
        return new{typeof(settings), typeof(sigma), typeof(rc), typeof(alg)}(settings,
                                                                             sigma, rc, alg)
    end
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  sigma::Option{<:MatNum} = nothing, rc::Option{<:LcULcE} = nothing,
                  alg::VarianceFormulation = SquaredSOCRiskExpr())
    return Variance(settings, sigma, rc, alg)
end
function (r::Variance)(w::VecNum)
    return dot(w, r.sigma, w)
end
"""
    factory(r::Variance, prior::AbstractPriorResult, args...; kwargs...)

Create an instance of [`Variance`](@ref) by selecting the covariance matrix from the risk-measure instance or falling back to the prior result (see [`nothing_scalar_array_factory`](@ref)).

# Arguments

  - `r`: Prototype risk measure whose `settings`, `rc` and `alg` fields are reused for the new instance.
  - `prior`: Prior result providing `prior.sigma` to use when `r.sigma === nothing`.
  - `args...`: Extra positional arguments are accepted for API compatibility but are ignored by this constructor.
  - `kwargs...` : Keyword arguments are accepted for API compatibility but are ignored by this constructor.

# Returns

  - `r_new::Variance`: A new `Variance` instance.

# Details

  - Selects `sigma` using [`nothing_scalar_array_factory`](@ref).
  - Other fields are taken from `r`.

# Related

  - [`Variance`](@ref)
  - [`nothing_scalar_array_factory`](@ref)
"""
function factory(r::Variance, prior::AbstractPriorResult, args...; kwargs...)
    sigma = nothing_scalar_array_factory(r.sigma, prior.sigma)
    return Variance(; settings = r.settings, sigma = sigma, rc = r.rc, alg = r.alg)
end
function risk_measure_view(r::Variance, i, args...)
    sigma = nothing_scalar_array_view(r.sigma, i)
    @argcheck(!isa(r.rc, LinearConstraint),
              "`rc` cannot be a `LinearConstraint` because there is no way to only consider items from a specific group and because this would break factor risk contribution")
    return Variance(; settings = r.settings, sigma = sigma, rc = r.rc, alg = r.alg)
end
"""
    struct StandardDeviation{T1, T2} <: RiskMeasure
        settings::T1
        sigma::T2
    end

Represents the portfolio standard deviation using a covariance matrix. It is the square root of the variance.

# Fields

  - `settings`: Risk measure configuration.
  - `sigma`: Optional covariance matrix that overrides the prior covariance when provided. Also used to compute the risk represented by a vector.

# Constructors

    StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                       sigma::Option{<:MatNum} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - If `sigma` is provided, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.

## `JuMP` Formulation

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} &\\qquad \\sigma\\nonumber\\\\
\\textrm{s.t.} &\\qquad \\left\\lVert \\mathbf{G} \\boldsymbol{w} \\right\\rVert_{2} \\leq \\sigma\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\sigma``: Variable representing the optimised portfolio's standard deviation.
  - ``\\mathbf{G}``: Suitable factorisation of the `N×N` covariance matrix, such as the square root matrix, or the Cholesky factorisation.
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm, which is modelled as a [SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Functor

    (r::StandardDeviation)(w::VecNum)

Computes the standard deviation risk of a portfolio with weights `w` using the covariance matrix `r.sigma`.

```math
\\begin{align}
\\mathrm{StandardDeviation}(\\boldsymbol{w},\\, \\mathbf{\\Sigma}) &= \\sqrt{\\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N×N` covariance matrix.

## Arguments

  - `w::VecNum`: Asset weights.

# Examples

```jldoctest
julia> w = [0.3803452066954233, 0.5900852659955864, 0.029569527308990307];

julia> r = StandardDeviation(;
                             sigma = [0.97780 -0.06400 0.84818;
                                      -0.06400 3.28564 1.84588;
                                      0.84818 1.84588 2.16317])
StandardDeviation
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     sigma ┴ 3×3 Matrix{Float64}

julia> r(w)
1.1585208588621345
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`factory(r::StandardDeviation, prior::AbstractPriorResult, args...; kwargs...)`](@ref)
  - [`expected_risk`](@ref)
"""
struct StandardDeviation{T1, T2} <: RiskMeasure
    settings::T1
    sigma::T2
    function StandardDeviation(settings::RiskMeasureSettings, sigma::Option{<:MatNum})
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma))
            assert_matrix_issquare(sigma, :sigma)
        end
        return new{typeof(settings), typeof(sigma)}(settings, sigma)
    end
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Option{<:MatNum} = nothing)
    return StandardDeviation(settings, sigma)
end
function (r::StandardDeviation)(w::VecNum)
    return sqrt(dot(w, r.sigma, w))
end
"""
    factory(r::StandardDeviation, prior::AbstractPriorResult, args...; kwargs...)

Create an instance of [`StandardDeviation`](@ref) by selecting the covariance matrix from the risk-measure instance or falling back to the prior result (see [`nothing_scalar_array_factory`](@ref)).

# Arguments

  - `r`: Prototype risk measure whose `settings`, `rc` and `alg` fields are reused for the new instance.
  - `prior`: Prior result providing `prior.sigma` to use when `r.sigma === nothing`.
  - `args...`: Extra positional arguments are accepted for API compatibility but are ignored by this constructor.
  - `kwargs...` : Keyword arguments are accepted for API compatibility but are ignored by this constructor.

# Returns

  - `r_new::StandardDeviation`: A new `StandardDeviation` instance.

# Details

  - Selects `sigma` using [`nothing_scalar_array_factory`](@ref).
  - Other fields are taken from `r`.

# Related

  - [`StandardDeviation`](@ref)
  - [`nothing_scalar_array_factory`](@ref)
"""
function factory(r::StandardDeviation, prior::AbstractPriorResult, args...; kwargs...)
    sigma = nothing_scalar_array_factory(r.sigma, prior.sigma)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end
function risk_measure_view(r::StandardDeviation, i, args...)
    sigma = nothing_scalar_array_view(r.sigma, i)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end
"""
    struct UncertaintySetVariance{T1, T2, T3} <: RiskMeasure

Represents the variance risk measure under uncertainty sets. Works the same way as the [`Variance`](@ref) risk measure but allows specifying uncertainty set estimators or results. These are only used in `JuMP`-based optimisations because they dictate how the variance is formulated as an optimisation problem. By encapsulating the uncertainty set estimator or result, enables the use of multiple uncertainty set variances in the same optimisation model.

# Fields

  - `settings`: Risk measure configuration.
  - `ucs`: Uncertainty set estimator or result that defines the uncertainty model for the variance calculation.
  - `sigma`: Optional covariance matrix that overrides the prior covariance when provided. Also used to compute the risk represented by a vector.

# Constructors

    UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           ucs::Option{<:UcUUcE} = NormalUncertaintySet(),
                           sigma::Option{<:MatNum} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - If `sigma` is provided, `!isempty(sigma)`.

# `JuMP` Formulations

When using an uncertainty set on the variance, the optimisation problem becomes:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} &\\qquad \\underset{\\mathbf{\\Sigma} \\in U_{\\mathbf{\\Sigma}}}{\\max} \\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N×N` covariance matrix.
  - ``U_{\\mathbf{\\Sigma}}``: Uncertainty set for the covariance matrix.

This problem can be reformulated depending on the type of uncertainty set used.

## Box uncertainty set

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} & \\quad \\mathrm{Tr}\\left(\\mathbf{A}_u \\mathbf{\\Sigma}_u\\right) - \\mathrm{Tr}\\left(\\mathbf{A}_l \\mathbf{\\Sigma}_l\\right)\\\\
\\textrm{s.t.} & \\quad \\mathbf{A}_u \\geq 0\\\\
               & \\quad \\mathbf{A}_l \\geq 0\\\\
               & \\quad \\begin{bmatrix}
                            \\mathbf{W} & \\boldsymbol{w}\\\\
                            \\boldsymbol{w}^\\intercal & k
                        \\end{bmatrix} \\succeq 0 \\\\
               & \\quad \\mathbf{A}_u - \\mathbf{A}_l = \\mathbf{W}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.

  - ``\\mathbf{A}_u``, ``\\mathbf{A}_l``, ``\\mathbf{W}``: `N×N` auxiliary symmetric matrices.
  - ``\\mathbf{\\Sigma}_l``: `N×N` lower bound of the covariance matrix.
  - ``\\mathbf{\\Sigma}_u``: `N×N` upper bound of the covariance matrix.
  - ``k``: Scalar variable/constant.

      + If the objective risk-adjusted return, it is a non-negative variable.
      + Else it is equal to 1.
  - ``\\mathrm{Tr}(\\cdot)``: Trace operator.

## Ellipse uncertainty set

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} & \\quad \\mathrm{Tr}\\left( \\mathbf{\\Sigma} \\left( \\mathbf{W} + \\mathbf{E} \\right) \\right) + k_{\\mathbf{\\Sigma}} \\sigma \\\\
\\textrm{s.t.} & \\quad \\begin{bmatrix}
                            \\mathbf{W} & \\boldsymbol{w}\\\\
                            \\boldsymbol{w}^\\intercal & k
                        \\end{bmatrix} \\succeq 0 \\\\
               & \\quad \\mathbf{E} \\succeq 0 \\\\
               & \\quad \\lVert \\mathbf{G} \\mathrm{vec}\\left( \\mathbf{W} + \\mathbf{E} \\right) \\rVert_{2} \\leq \\sigma \\\\
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.

  - ``\\mathbf{\\Sigma}``: `N×N` covariance matrix.
  - ``\\mathbf{W}``, ``\\mathbf{E}``: `N×N` auxiliary symmetric matrices.
  - ``k_{\\mathbf{\\Sigma}}``: Scalar constant defining the size of the uncertainty set.
  - ``\\sigma``: Variable representing the portfolio's variance of the variance.
  - ``\\mathbf{G}``: Suitable factorisation of the `N^2×N^2` covariance of the covariance matrix of the uncertainty set, such as the square root matrix, or the Cholesky factorisation.
  - ``k``: Scalar variable/constant.

      + If the objective risk-adjusted return, it is a non-negative variable.
      + Else it is equal to 1.
  - ``\\mathrm{Tr}(\\cdot)``: Trace operator.
  - ``\\mathrm{vec}(\\cdot)``: Vectorisation operator, which unrolls a matrix as a column vector in column-major order.
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm, which is modelled as a [SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Functor

    (r::UncertaintySetVariance)(w::VecNum)

Computes the variance risk of a portfolio with weights `w` using the covariance matrix `r.sigma`.

```math
\\begin{align}
\\mathrm{UncertaintySetVariance}(\\boldsymbol{w},\\, \\mathbf{\\Sigma}) &= \\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N×N` covariance matrix.

## Arguments

  - `w::VecNum`: Asset weights.

# Examples

```jldoctest
julia> w = [0.3803452066954233, 0.5900852659955864, 0.029569527308990307];

julia> r = UncertaintySetVariance(;
                                  sigma = [0.97780 -0.06400 0.84818;
                                           -0.06400 3.28564 1.84588;
                                           0.84818 1.84588 2.16317])
UncertaintySetVariance
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       ucs ┼ NormalUncertaintySet
           │      pe ┼ EmpiricalPrior
           │         │        ce ┼ PortfolioOptimisersCovariance
           │         │           │   ce ┼ Covariance
           │         │           │      │    me ┼ SimpleExpectedReturns
           │         │           │      │       │   w ┴ nothing
           │         │           │      │    ce ┼ GeneralCovariance
           │         │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
           │         │           │      │       │    w ┴ nothing
           │         │           │      │   alg ┴ Full()
           │         │           │   mp ┼ DefaultMatrixProcessing
           │         │           │      │       pdm ┼ Posdef
           │         │           │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
           │         │           │      │   denoise ┼ nothing
           │         │           │      │    detone ┼ nothing
           │         │           │      │       alg ┴ nothing
           │         │        me ┼ SimpleExpectedReturns
           │         │           │   w ┴ nothing
           │         │   horizon ┴ nothing
           │     alg ┼ BoxUncertaintySetAlgorithm()
           │   n_sim ┼ Int64: 3000
           │       q ┼ Float64: 0.05
           │     rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
           │    seed ┴ nothing
     sigma ┴ 3×3 Matrix{Float64}

julia> r(w)
1.3421705804186579
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`Variance`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`factory(r::UncertaintySetVariance, prior::AbstractPriorResult, args...; kwargs...)`](@ref)
  - [`expected_risk`](@ref)
"""
struct UncertaintySetVariance{T1, T2, T3} <: RiskMeasure
    settings::T1
    ucs::T2
    sigma::T3
    function UncertaintySetVariance(settings::RiskMeasureSettings, ucs::Option{<:UcUUcE},
                                    sigma::Option{<:MatNum})
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma))
        end
        return new{typeof(settings), typeof(ucs), typeof(sigma)}(settings, ucs, sigma)
    end
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                ucs::Option{<:UcUUcE} = NormalUncertaintySet(),
                                sigma::Option{<:MatNum} = nothing)
    return UncertaintySetVariance(settings, ucs, sigma)
end
function (r::UncertaintySetVariance)(w::VecNum)
    return dot(w, r.sigma, w)
end
function no_bounds_risk_measure(r::UncertaintySetVariance, flag::Bool = true)
    return if flag
        UncertaintySetVariance(;
                               settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                              scale = r.settings.scale),
                               r.ucs, sigma = r.sigma)
    else
        Variance(;
                 settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                scale = r.settings.scale), nothing,
                 sigma = r.sigma)
    end
end
"""
    factory(r::UncertaintySetVariance, prior::AbstractPriorResult, ::Any,
            ucs::Option{<:UcUUcE} = nothing, args...;
            kwargs...)

Create an instance of [`UncertaintySetVariance`](@ref) by selecting the uncertainty set and covariance matrix from the risk-measure instance or falling back to the prior result.

# Arguments

  - `r`: Prototype risk measure whose `settings` and `sigma` fields are reused for the new instance.
  - `prior`: Prior result providing `prior.sigma` to use when `r.sigma === nothing`.
  - `::Any`: Placeholder positional argument for API compatibility.
  - `ucs`: Optional uncertainty set estimator or result to override `r.ucs`.
  - `args...`: Extra positional arguments are accepted for API compatibility but are ignored by this constructor.
  - `kwargs...`: Keyword arguments are accepted for API compatibility but are ignored by this constructor.

# Returns

  - `r_new::UncertaintySetVariance`: A new `UncertaintySetVariance` instance.

# Details

  - Selects `ucs` using [`ucs_factory`](@ref).
  - Selects `sigma` using [`nothing_scalar_array_factory`](@ref).
  - Other fields are taken from `r`.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`ucs_factory`](@ref)
  - [`nothing_scalar_array_factory`](@ref)
"""
function factory(r::UncertaintySetVariance, prior::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcUUcE} = nothing, args...; kwargs...)
    ucs = ucs_factory(r.ucs, ucs)
    sigma = nothing_scalar_array_factory(r.sigma, prior.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
function risk_measure_view(r::UncertaintySetVariance, i, args...)
    ucs = ucs_view(r.ucs, i)
    sigma = nothing_scalar_array_view(r.sigma, i)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end

export SOCRiskExpr, QuadRiskExpr, SquaredSOCRiskExpr, RSOCRiskExpr, Variance,
       StandardDeviation, UncertaintySetVariance
