"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation formulations of second moment risk measures.

# Related Types

  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
"""
abstract type SecondMomentFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for optimisation formulations of variance-based risk measures.

# Related Types

  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
abstract type VarianceFormulation <: SecondMomentFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Direct quadratic risk expression optimisation formulation for variance-like risk measures. The risk measure is implemented using an explicitly quadratic form. This can be in two ways.

# Summary statistics

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} \\quad & \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N × N` co-moment matrix.

# Scenario-based

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} \\quad & \\boldsymbol{d} \\cdot \\boldsymbol{d}.\\\\
\\text{s.t.} \\quad & \\boldsymbol{d} \\in \\mathcal{S}_{w}.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` deviations vector.
  - ``\\mathcal{S}_{w}``: Scenario set for portfolio `x`.

# Related Types

  - [`VarianceFormulation`](@ref)
  - [`Variance`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
struct QuadRiskExpr <: VarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Squared second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure is implemented using the square of a variable constrained by a second order cone.

# Related

  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`Variance`](@ref)
"""
struct SquaredSOCRiskExpr <: VarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Rotated second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure using a variable constrained to be in a rotated second order cone representing the sum of squares.

# Related Types

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
struct RSOCRiskExpr <: SecondMomentFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure is implemented using a variable constrained by a second order cone.

# Related

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
"""
struct SOCRiskExpr <: SecondMomentFormulation end
"""
    const NSkeQuadFormulations

Union type of quadratic OWA risk expression formulations for the Negative Skewness risk measure.

Specifically: `Union{<:QuadRiskExpr, <:SquaredSOCRiskExpr}`.

# Related

  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`NegativeSkewness`](@ref)
"""
const NSkeQuadFormulations = Union{<:QuadRiskExpr, <:SquaredSOCRiskExpr}
"""
    const QuadSecondMomentFormulations = Union{<:NSkeQuadFormulations, <:RSOCRiskExpr}

Union of quadratic and RSOC formulations for second-moment (variance-based) risk expressions.

# Related

  - [`NSkeQuadFormulations`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`Variance`](@ref)
"""
const QuadSecondMomentFormulations = Union{<:NSkeQuadFormulations, <:RSOCRiskExpr}
"""
$(DocStringExtensions.TYPEDEF)

Represents the portfolio variance using a covariance matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Variance(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        sigma::Option{<:MatNum} = nothing,
        rc::Option{<:LcE_Lc} = nothing,
        alg::VarianceFormulation = SquaredSOCRiskExpr(),
    ) -> Variance

Keywords correspond to the struct's fields.

## Validation

  - If `sigma` is not `nothing`, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.

# `JuMP` Formulations

!!! info

    Regardless of the formulation used, an auxiliary variable representing the standard deviation is needed in order to constrain the risk or maximise the risk-adjusted return ratio. This is because quadratic constraints are not strictly convex, and the transformation needed to maximise the risk-adjusted return ratio requires affine variables in the numerator and denominator.

Depending on the `alg` field, the variance risk measure is formulated using `JuMP` as follows:

## `QuadRiskExpr`

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} \\quad & \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N × N` covariance matrix.

## `SquaredSOCRiskExpr`

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} \\quad & \\sigma^2\\nonumber\\\\
\\text{s.t.} \\quad & \\left\\lVert \\mathbf{G} \\boldsymbol{w} \\right\\rVert_{2} \\leq \\sigma\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\sigma``: Variable representing the optimised portfolio's standard deviation.
  - ``\\mathbf{G}``: Suitable factorisation of the `N × N` covariance matrix, such as the square root matrix, or the Cholesky factorisation.
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm, which is modelled as a [JuMP.SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Functor

    (r::Variance)(w::VecNum)

Computes the variance risk of a portfolio with weights `w` using the covariance matrix `r.sigma`.

```math
\\begin{align}
\\mathrm{Variance}(\\boldsymbol{w},\\, \\mathbf{\\Sigma}) &= \\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N × N` covariance matrix.

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
      chol ┼ nothing
        rc ┼ nothing
       alg ┴ SquaredSOCRiskExpr()

julia> r(w)
1.3421705804186579
```

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`scalarise_risk_expression!`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`factory`](@ref)
  - [`expected_risk`](@ref)
"""
@propagatable @concrete struct Variance <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:sigma])
    """
    @pprop sigma
    """
    $(field_dict[:chol])
    """
    @pprop chol
    """
    $(field_dict[:rc])
    """
    rc
    """
    $(field_dict[:alg])
    """
    alg
    function Variance(settings::RiskMeasureSettings, sigma::Option{<:MatNum},
                      chol::Option{<:MatNum}, rc::Option{<:LcE_Lc},
                      alg::VarianceFormulation)::Variance
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
            assert_matrix_issquare(sigma, :sigma)
        end
        if isa(chol, MatNum)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
        end
        return new{typeof(settings), typeof(sigma), typeof(chol), typeof(rc), typeof(alg)}(settings,
                                                                                           sigma,
                                                                                           chol,
                                                                                           rc,
                                                                                           alg)
    end
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  sigma::Option{<:MatNum} = nothing, chol::Option{<:MatNum} = nothing,
                  rc::Option{<:LcE_Lc} = nothing,
                  alg::VarianceFormulation = SquaredSOCRiskExpr())::Variance
    return Variance(settings, sigma, chol, rc, alg)
end
function (r::Variance)(w::VecNum)
    return LinearAlgebra.dot(w, r.sigma, w)
end
function port_opt_view(r::Variance, i, args...)
    sigma = nothing_scalar_array_view(r.sigma, i)
    chol = isnothing(r.chol) ? nothing : view(r.chol, :, i)
    @argcheck(!isa(r.rc, LinearConstraint),
              "`rc` cannot be a `LinearConstraint` because there is no way to only consider items from a specific group and because this would break factor risk contribution")
    return Variance(; settings = r.settings, sigma = sigma, chol = chol, rc = r.rc,
                    alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the portfolio standard deviation using a covariance matrix. It is the square root of the variance.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardDeviation(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        sigma::Option{<:MatNum} = nothing,
    ) -> StandardDeviation

Keywords correspond to the struct's fields.

## Validation

  - If `sigma` is not `nothing`, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.

## `JuMP` Formulation

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} \\quad & \\sigma\\nonumber\\\\
\\text{s.t.} \\quad & \\left\\lVert \\mathbf{G} \\boldsymbol{w} \\right\\rVert_{2} \\leq \\sigma\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\sigma``: Variable representing the optimised portfolio's standard deviation.
  - ``\\mathbf{G}``: Suitable factorisation of the `N × N` covariance matrix, such as the square root matrix, or the Cholesky factorisation.
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm, which is modelled as a [JuMP.SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Functor

    (r::StandardDeviation)(w::VecNum)

Computes the standard deviation risk of a portfolio with weights `w` using the covariance matrix `r.sigma`.

```math
\\begin{align}
\\mathrm{StandardDeviation}(\\boldsymbol{w},\\, \\mathbf{\\Sigma}) &= \\sqrt{\\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N × N` covariance matrix.

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
     sigma ┼ 3×3 Matrix{Float64}
      chol ┴ nothing

julia> r(w)
1.1585208588621345
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`factory`](@ref)
  - [`expected_risk`](@ref)
"""
@propagatable @concrete struct StandardDeviation <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:sigma])
    """
    @pprop sigma
    """
    $(field_dict[:chol])
    """
    @pprop chol
    function StandardDeviation(settings::RiskMeasureSettings, sigma::Option{<:MatNum},
                               chol::Option{<:MatNum})::StandardDeviation
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
            assert_matrix_issquare(sigma, :sigma)
        end
        if isa(chol, MatNum)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
        end
        return new{typeof(settings), typeof(sigma), typeof(chol)}(settings, sigma, chol)
    end
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Option{<:MatNum} = nothing,
                           chol::Option{<:MatNum} = nothing)::StandardDeviation
    return StandardDeviation(settings, sigma, chol)
end
function (r::StandardDeviation)(w::VecNum)
    return sqrt(LinearAlgebra.dot(w, r.sigma, w))
end
function port_opt_view(r::StandardDeviation, i, args...)
    sigma = nothing_scalar_array_view(r.sigma, i)
    chol = isnothing(r.chol) ? nothing : view(r.chol, :, i)
    return StandardDeviation(; settings = r.settings, sigma = sigma, chol = chol)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the variance risk measure under uncertainty sets. Works the same way as the [`Variance`](@ref) risk measure but allows specifying uncertainty set estimators or results. These are only used in `JuMP`-based optimisations because they dictate how the variance is formulated as an optimisation problem. By encapsulating the uncertainty set estimator or result, enables the use of multiple uncertainty set variances in the same optimisation model.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    UncertaintySetVariance(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        ucs::Option{<:UcSE_UcS} = NormalUncertaintySet(),
        sigma::Option{<:MatNum} = nothing,
    ) -> UncertaintySetVariance

Keywords correspond to the struct's fields.

## Validation

  - If `sigma` is not `nothing`, `!isempty(sigma)`.

# `JuMP` Formulations

When using an uncertainty set on the variance, the optimisation problem becomes:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} \\quad & \\underset{\\mathbf{\\Sigma} \\in U_{\\mathbf{\\Sigma}}}{\\max} \\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N × N` covariance matrix.
  - ``U_{\\mathbf{\\Sigma}}``: Uncertainty set for the covariance matrix.

This problem can be reformulated depending on the type of uncertainty set used.

## Box uncertainty set

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} & \\quad \\mathrm{Tr}\\left(\\mathbf{A}_u \\mathbf{\\Sigma}_u\\right) - \\mathrm{Tr}\\left(\\mathbf{A}_l \\mathbf{\\Sigma}_l\\right)\\\\
\\text{s.t.} & \\quad \\mathbf{A}_u \\geq 0\\\\
               & \\quad \\mathbf{A}_l \\geq 0\\\\
               & \\quad \\begin{bmatrix}
                            \\mathbf{W} & \\boldsymbol{w}\\\\
                            \\boldsymbol{w}^\\intercal & k
                        \\end{bmatrix} \\succeq 0 \\\\
               & \\quad \\mathbf{A}_u - \\mathbf{A}_l = \\mathbf{W}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.

  - ``\\mathbf{A}_u``, ``\\mathbf{A}_l``, ``\\mathbf{W}``: `N × N` auxiliary symmetric matrices.

  - ``\\mathbf{\\Sigma}_l``: `N × N` lower bound of the covariance matrix.

  - ``\\mathbf{\\Sigma}_u``: `N × N` upper bound of the covariance matrix.

  - ``k``: Scalar variable/constant.

      + If the objective risk-adjusted return, it is a non-negative variable.
      + Else it is equal to 1.

  - ``\\mathrm{Tr}(\\cdot)``: Trace operator.

## Ellipsoidal uncertainty set

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}} & \\quad \\mathrm{Tr}\\left( \\mathbf{\\Sigma} \\left( \\mathbf{W} + \\mathbf{E} \\right) \\right) + k_{\\mathbf{\\Sigma}} \\sigma \\\\
\\text{s.t.} & \\quad \\begin{bmatrix}
                            \\mathbf{W} & \\boldsymbol{w}\\\\
                            \\boldsymbol{w}^\\intercal & k
                        \\end{bmatrix} \\succeq 0 \\\\
               & \\quad \\mathbf{E} \\succeq 0 \\\\
               & \\quad \\lVert \\mathbf{G} \\mathrm{vec}\\left( \\mathbf{W} + \\mathbf{E} \\right) \\rVert_{2} \\leq \\sigma \\\\
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.

  - ``\\mathbf{\\Sigma}``: `N × N` covariance matrix.

  - ``\\mathbf{W}``, ``\\mathbf{E}``: `N × N` auxiliary symmetric matrices.

  - ``k_{\\mathbf{\\Sigma}}``: Scalar constant defining the size of the uncertainty set.

  - ``\\sigma``: Variable representing the portfolio's variance of the variance.

  - ``\\mathbf{G}``: Suitable factorisation of the `N^2 × N^2` covariance of the covariance matrix of the uncertainty set, such as the square root matrix, or the Cholesky factorisation.

  - ``k``: Scalar variable/constant.

      + If the objective risk-adjusted return, it is a non-negative variable.
      + Else it is equal to 1.

  - ``\\mathrm{Tr}(\\cdot)``: Trace operator.

  - ``\\mathrm{vec}(\\cdot)``: Vectorisation operator, which unrolls a matrix as a column vector in column-major order.

  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm, which is modelled as a [JuMP.SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Functor

    (r::UncertaintySetVariance)(w::VecNum)

Computes the variance risk of a portfolio with weights `w` using the covariance matrix `r.sigma`.

```math
\\begin{align}
\\mathrm{UncertaintySetVariance}(\\boldsymbol{w},\\, \\mathbf{\\Sigma}) &= \\boldsymbol{w}^\\intercal \\, \\mathbf{\\Sigma}\\, \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\mathbf{\\Sigma}``: `N × N` covariance matrix.

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
           │       pe ┼ EmpiricalPrior
           │          │        ce ┼ PortfolioOptimisersCovariance
           │          │           │   ce ┼ Covariance
           │          │           │      │    me ┼ SimpleExpectedReturns
           │          │           │      │       │   w ┴ nothing
           │          │           │      │    ce ┼ GeneralCovariance
           │          │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
           │          │           │      │       │    w ┴ nothing
           │          │           │      │   alg ┴ FullMoment()
           │          │           │   mp ┼ MatrixProcessing
           │          │           │      │     pdm ┼ Posdef
           │          │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
           │          │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │          │           │      │      dn ┼ nothing
           │          │           │      │      dt ┼ nothing
           │          │           │      │     alg ┼ nothing
           │          │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
           │          │        me ┼ SimpleExpectedReturns
           │          │           │   w ┴ nothing
           │          │   horizon ┴ nothing
           │      alg ┼ BoxUncertaintySetAlgorithm()
           │    n_sim ┼ Int64: 3000
           │        q ┼ Float64: 0.05
           │      rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
           │     seed ┼ nothing
           │      ens ┼ nothing
           │      pdm ┼ Posdef
           │          │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
           │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │   kwargs ┴ @NamedTuple{}: NamedTuple()
     sigma ┴ 3×3 Matrix{Float64}

julia> r(w)
1.3421705804186579
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`Variance`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`factory(r::UncertaintySetVariance, pr::AbstractPriorResult, args...; kwargs...)`](@ref)
  - [`expected_risk`](@ref)
"""
@concrete struct UncertaintySetVariance <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:ucs])
    """
    ucs
    """
    $(field_dict[:sigma])
    """
    sigma
    function UncertaintySetVariance(settings::RiskMeasureSettings, ucs::Option{<:UcSE_UcS},
                                    sigma::Option{<:MatNum})
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
        end
        return new{typeof(settings), typeof(ucs), typeof(sigma)}(settings, ucs, sigma)
    end
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                ucs::Option{<:UcSE_UcS} = NormalUncertaintySet(),
                                sigma::Option{<:MatNum} = nothing)
    return UncertaintySetVariance(settings, ucs, sigma)
end
"""
    (r::UncertaintySetVariance)(w::VecNum)

Compute the risk of weights `w` under an [`UncertaintySetVariance`](@ref) measure.

When `r.ucs` is a fitted [`AbstractUncertaintySetResult`](@ref), returns the worst-case
variance over the uncertainty set via [`ucs_variance`](@ref) — consistent with the risk
expression built by [`set_ucs_variance_risk!`](@ref). With an unfitted estimator (or
`nothing`), falls back to the nominal variance `w' * sigma * w`.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`ucs_variance`](@ref)
"""
function (r::UncertaintySetVariance{<:Any, <:AbstractUncertaintySetResult, <:Any})(w::VecNum)
    return ucs_variance(r.ucs, r.sigma, w)
end
function (r::UncertaintySetVariance)(w::VecNum)
    return LinearAlgebra.dot(w, r.sigma, w)
end
"""
    ucs_variance(ucs::AbstractUncertaintySetResult, sigma::MatNum, w::VecNum)

Compute the worst-case portfolio variance of weights `w` over a fitted uncertainty set.

This is the scalar twin of the JuMP expression built by [`set_ucs_variance_risk!`](@ref):
for a [`BoxUncertaintySet`](@ref) it evaluates `tr(Au * ub) - tr(Al * lb)` at the optimal
`Au = max.(W, 0)`, `Al = max.(-W, 0)` with `W = w * w'`; for an
[`EllipsoidalUncertaintySet`](@ref) it evaluates `tr(sigma * W) + k * norm(G * vec(W))`
with `G` the upper Cholesky factor of the set's shape matrix (the `E = 0` evaluation of
the model expression, an upper bound on its optimum).

The [`UncertaintySetVariance`](@ref) functor dispatches here when its `ucs` field is a
fitted result, keeping scalar risk evaluation consistent with the risk expression the
optimiser sees.

# Arguments

  - `ucs`: Fitted uncertainty set result.
  - `sigma::MatNum`: Nominal covariance matrix.
  - `w::VecNum`: Vector of portfolio weights.

# Returns

  - `risk::Number`: Worst-case portfolio variance.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function ucs_variance(ucs::BoxUncertaintySet, ::Any, w::VecNum)
    W = w * transpose(w)
    z = zero(eltype(W))
    return sum(ucs.ub .* max.(W, z)) - sum(ucs.lb .* max.(-W, z))
end
function ucs_variance(ucs::EllipsoidalUncertaintySet, sigma::MatNum, w::VecNum)
    W = w * transpose(w)
    G = LinearAlgebra.cholesky(ucs.sigma).U
    return LinearAlgebra.tr(sigma * W) + ucs.k * LinearAlgebra.norm(G * vec(W))
end
"""
    _no_bounds_risk_measure(r, flag)

Return a version of the risk measure stripped of bounds for unbounded optimisation sub-problems.

Internal helper used in frontier construction sub-problems where bounds are temporarily removed.

# Arguments

  - `r`: Risk measure.
  - `flag`: Flag controlling which bounds to remove.

# Returns

  - Risk measure without bounds.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`_no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function _no_bounds_risk_measure(r::UncertaintySetVariance, ::Union{Val{true}, Nothing})
    return UncertaintySetVariance(;
                                  settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                                 scale = r.settings.scale),
                                  r.ucs, sigma = r.sigma)
end
function _no_bounds_risk_measure(r::UncertaintySetVariance, ::Val{false})
    return Variance(;
                    settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                   scale = r.settings.scale),
                    sigma = r.sigma)
end
function no_bounds_risk_measure(r::UncertaintySetVariance,
                                flag::Union{Val{false}, Val{true}, Nothing} = nothing)
    return _no_bounds_risk_measure(r, flag)
end
"""
    _no_bounds_no_risk_expr_risk_measure(r, flag)

Return a version of the risk measure with neither bounds nor risk expressions for unbounded sub-problems.

Internal helper used in frontier sub-problems that require removing all risk expression constraints.

# Arguments

  - `r`: Risk measure.
  - `flag`: Flag controlling configuration.

# Returns

  - Simplified risk measure.

# Related

  - [`_no_bounds_risk_measure`](@ref)
  - [`UncertaintySetVariance`](@ref)
"""
function _no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance,
                                              ::Union{Val{true}, Nothing})
    return UncertaintySetVariance(;
                                  settings = RiskMeasureSettings(; rke = false,
                                                                 scale = r.settings.scale),
                                  r.ucs, sigma = r.sigma)
end
function _no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance, ::Val{false})
    return Variance(;
                    settings = RiskMeasureSettings(; rke = false, scale = r.settings.scale),
                    rc = nothing, sigma = r.sigma)
end
function no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance,
                                             flag::Union{Val{false}, Val{true}, Nothing} = nothing)
    return _no_bounds_no_risk_expr_risk_measure(r, flag)
end
"""
    factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any,
            ucs::Option{<:UcSE_UcS} = nothing, args...;
            kwargs...)

Create an instance of [`UncertaintySetVariance`](@ref) by selecting the uncertainty set and covariance matrix from the risk-measure instance or falling back to the prior result.

# Arguments

  - `r`: Prototype risk measure whose `settings` and `sigma` fields are reused for the new instance.
  - `prior`: Prior result providing `pr.sigma` to use when `r.sigma === nothing`.
  - `::Any`: Placeholder positional argument for API compatibility.
  - `ucs`: Optional uncertainty set estimator or result to override `r.ucs`.
  - `args...`: Extra positional arguments are accepted for API compatibility but are ignored by this constructor.
  - `kwargs...`: Keyword arguments are accepted for API compatibility but are ignored by this constructor.

# Returns

  - `r_new::UncertaintySetVariance`: A new `UncertaintySetVariance` instance.

# Details

  - Selects `ucs` using [`ucs_selector`](@ref).
  - Selects `sigma` using [`nothing_scalar_array_selector`](@ref).
  - Other fields are taken from `r`.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`ucs_selector`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcSE_UcS} = nothing, args...; kwargs...)
    ucs = ucs_selector(r.ucs, ucs)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`UncertaintySetVariance`](@ref) without a placeholder positional argument (see [`factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any, ucs, args...; kwargs...)`](@ref)).

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`factory`](@ref)
  - [`ucs_selector`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::UncertaintySetVariance, pr::AbstractPriorResult,
                 ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
    ucs = ucs_selector(r.ucs, ucs)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`UncertaintySetVariance`](@ref) with the uncertainty set as the first override argument.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`factory`](@ref)
  - [`ucs_selector`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::UncertaintySetVariance, ucs::UcSE_UcS,
                 pr::Option{<:AbstractPriorResult} = nothing; kwargs...)
    ucs = ucs_selector(r.ucs, ucs)
    sigma = if isnothing(pr)
        r.sigma
    else
        nothing_scalar_array_selector(r.sigma, pr.sigma)
    end
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
"""
    ucs_risk_measure(r, rd::ReturnsResult)

Resolve the uncertainty set of an [`UncertaintySetVariance`](@ref) risk measure to a
fitted [`AbstractUncertaintySetResult`](@ref) using the returns data. Other risk measures
are returned unchanged; vectors of risk measures are resolved element-wise.

Used by [`near_optimal_centering_setup`](@ref) so that the barrier risk targets, the
sub-problem solves, and the NOC model all share the same fitted uncertainty set (fitted
results pass through [`sigma_ucs`](@ref) unchanged). With a fitted set the
[`UncertaintySetVariance`](@ref) functor evaluates the worst-case variance via
[`ucs_variance`](@ref), keeping the barrier targets consistent with the model risk
expression.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`sigma_ucs`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
function ucs_risk_measure(r::UncertaintySetVariance, rd::ReturnsResult)
    return Accessors.@set r.ucs = sigma_ucs(r.ucs, rd)
end
function ucs_risk_measure(r::Any, ::ReturnsResult)
    return r
end
function ucs_risk_measure(rs::VecBaseRM, rd::ReturnsResult)
    return ucs_risk_measure.(rs, Ref(rd))
end
function port_opt_view(r::UncertaintySetVariance, i, args...)
    ucs = port_opt_view(r.ucs, i)
    sigma = nothing_scalar_array_view(r.sigma, i)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Variance) = WeightsInput()
risk_input_kind(::StandardDeviation) = WeightsInput()
risk_input_kind(::UncertaintySetVariance) = WeightsInput()

export SOCRiskExpr, QuadRiskExpr, SquaredSOCRiskExpr, RSOCRiskExpr, Variance,
       StandardDeviation, UncertaintySetVariance
