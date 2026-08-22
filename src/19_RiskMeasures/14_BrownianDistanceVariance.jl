"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Brownian Distance Variance formulation algorithms.

All concrete types implementing specific formulations for the Brownian Distance Variance optimisation constraint should subtype `BrownianDistanceVarianceFormulation`.

# Related

  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
abstract type BrownianDistanceVarianceFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Norm-one cone formulation for the Brownian Distance Variance optimisation constraint.

Uses a norm-one cone constraint to encode the ``L^1`` structure of the Brownian distance matrix in the optimisation model.

# Related

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
struct NormOneConeBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Inequality formulation for the Brownian Distance Variance optimisation constraint.

Uses explicit linear inequality constraints to encode the absolute value structure of the Brownian distance matrix in the optimisation model.

# Related

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
struct IneqBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
"""
    const BDVarRkFormulations = Union{<:RSOCRiskExpr, <:QuadRiskExpr}

Union of valid optimisation formulations for the [`BrownianDistanceVariance`](@ref) risk measure.

# Related

  - [`RSOCRiskExpr`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
const BDVarRkFormulations = Union{<:RSOCRiskExpr, <:QuadRiskExpr}
"""
$(DocStringExtensions.TYPEDEF)

Represents the Brownian Distance Variance (BDVar) risk measure.

`BrownianDistanceVariance` measures the dispersion of the portfolio return series through its distance variance, the Brownian distance covariance of the series with itself. It is built from the pairwise distances between observations rather than from their deviations about the mean, so it reads non-linear structure that the variance cannot see. The functor and the optimisation model both return a disciplined-convex upper bound on that statistic, not the statistic itself.

# Mathematical definition

Given a portfolio returns vector ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal``, define the pairwise absolute distance matrix, the centring matrix, and the doubly centred distance matrix:

```math
\\begin{align}
D_{ij} &= |x_i - x_j|\\,, \\\\
\\mathbf{C}_T &= \\mathbf{I}_T - \\frac{1}{T} \\boldsymbol{1}_T \\boldsymbol{1}_T^\\intercal\\,, \\\\
\\mathbf{A} &= \\mathbf{C}_T \\mathbf{D} \\mathbf{C}_T\\,.
\\end{align}
```

Where:

  - ``D_{ij}``: Pairwise absolute distance between returns at periods ``i`` and ``j``.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\mathbf{D}``: ``T \\times T`` pairwise distance matrix.
  - ``\\mathbf{C}_T``: ``T \\times T`` centring matrix.
  - ``\\mathbf{A}``: ``T \\times T`` doubly centred distance matrix.
  - ``\\boldsymbol{1}_T``: Column vector of ones, ``T \\times 1``.

The distance variance is the mean square of the centred matrix:

```math
\\begin{align}
\\mathrm{dVar}(\\boldsymbol{x}) &= \\frac{1}{T^2} \\lVert \\mathbf{A} \\rVert_F^2 = \\frac{1}{T^2} \\left( \\mathrm{tr}\\left( \\mathbf{D}^\\intercal \\mathbf{D} \\left( \\mathbf{I}_T - \\frac{2}{T} \\boldsymbol{1}_T \\boldsymbol{1}_T^\\intercal \\right) \\right) + \\frac{1}{T^2} \\left( \\boldsymbol{1}_T^\\intercal \\mathbf{D} \\boldsymbol{1}_T \\right)^2 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{dVar}(\\boldsymbol{x})``: Distance variance.
  - ``\\lVert \\cdot \\rVert_F``: Frobenius norm.

``\\mathbf{I}_T - \\frac{2}{T} \\boldsymbol{1}_T \\boldsymbol{1}_T^\\intercal`` is indefinite, with eigenvalues ``1`` and ``-1``, so that expression is not disciplined-convex and no solver takes it directly. The trace inequality ``\\mathrm{tr}(\\mathbf{A}\\mathbf{B}) \\leq \\lambda_{\\max}(\\mathbf{A}) \\mathrm{tr}(\\mathbf{B})`` holds for a symmetric ``\\mathbf{A}`` and a positive semi-definite ``\\mathbf{B}``, and here ``\\lambda_{\\max} = 1``. This gives the disciplined-convex upper bound that this type computes:

```math
\\begin{align}
\\mathrm{BDVar}(\\boldsymbol{x}) &= \\frac{1}{T^2} \\left( \\lVert \\mathbf{D} \\rVert_F^2 + \\frac{1}{T^2} \\left( \\sum_{i,j} D_{ij} \\right)^2 \\right) \\geq \\mathrm{dVar}(\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{BDVar}(\\boldsymbol{x})``: The upper bound on the distance variance, and the value this type reports.

The functor and [`set_risk_constraints!`](@ref) compute the same bound, so a reported figure and the objective the optimiser minimised are the same quantity.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BrownianDistanceVariance(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alg1::BDVarRkFormulations = QuadRiskExpr(),
        alg2::BrownianDistanceVarianceFormulation = NormOneConeBrownianDistanceVariance()
    ) -> BrownianDistanceVariance

Keywords correspond to the struct's fields.

# Functor

    (r::BrownianDistanceVariance)(x::VecNum)

Computes the Brownian Distance Variance of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> BrownianDistanceVariance()
BrownianDistanceVariance
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
      alg1 ┼ QuadRiskExpr()
      alg2 ┴ NormOneConeBrownianDistanceVariance()
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)

# References

  - $(ref_dict[:szekely2007])
  - $(ref_dict[:bdvar])
"""
@concrete struct BrownianDistanceVariance <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alg1])
    """
    alg1
    """
    $(field_dict[:alg2])
    """
    alg2
    function BrownianDistanceVariance(settings::RiskMeasureSettings,
                                      alg1::BDVarRkFormulations,
                                      alg2::BrownianDistanceVarianceFormulation)
        return new{typeof(settings), typeof(alg1), typeof(alg2)}(settings, alg1, alg2)
    end
end
function BrownianDistanceVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  alg1::BDVarRkFormulations = QuadRiskExpr(),
                                  alg2::BrownianDistanceVarianceFormulation = NormOneConeBrownianDistanceVariance())::BrownianDistanceVariance
    return BrownianDistanceVariance(settings, alg1, alg2)
end
function (::BrownianDistanceVariance)(x::VecNum)
    T = length(x)
    iT2 = inv(T^2)
    D = Matrix{eltype(x)}(undef, T, T)
    D .= x
    D .-= transpose(x)
    D .= abs.(D)
    val = iT2 * (LinearAlgebra.dot(D, D) + iT2 * sum(D)^2)
    return val
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::BrownianDistanceVariance) = NetReturnsInput()

export NormOneConeBrownianDistanceVariance, IneqBrownianDistanceVariance,
       BrownianDistanceVariance
