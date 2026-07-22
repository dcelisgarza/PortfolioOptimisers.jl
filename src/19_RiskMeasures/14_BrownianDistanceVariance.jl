"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Brownian Distance Variance formulation algorithms.

All concrete types implementing specific formulations for the Brownian Distance Variance optimisation constraint should subtype `BrownianDistanceVarianceFormulation`.

# Related Types

  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
abstract type BrownianDistanceVarianceFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Norm-one cone formulation for the Brownian Distance Variance optimisation constraint.

Uses a norm-one cone constraint to encode the ``L^1`` structure of the Brownian distance matrix in the optimisation model.

# Related Types

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
struct NormOneConeBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Inequality formulation for the Brownian Distance Variance optimisation constraint.

Uses explicit linear inequality constraints to encode the absolute value structure of the Brownian distance matrix in the optimisation model.

# Related Types

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

`BrownianDistanceVariance` measures dependence between portfolio returns and a reference using the Brownian (distance) covariance framework. It captures non-linear dependence and is zero if and only if the returns are independent of the reference.

# Mathematical definition

Given a portfolio returns vector ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal``, define the pairwise absolute distance matrix:

```math
\\begin{align}
D_{ij} &= |x_i - x_j|\\,.
\\end{align}
```

Where:

  - ``D_{ij}``: Pairwise absolute distance between returns at periods ``i`` and ``j``.
  - $(math_dict[:xret])

The Brownian Distance Variance is:

```math
\\begin{align}
\\mathrm{BDVar}(\\boldsymbol{x}) &= \\frac{1}{T^2} \\left( \\lVert \\mathbf{D} \\rVert_F^2 + \\frac{1}{T^2} \\left( \\sum_{i,j} D_{ij} \\right)^2 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{BDVar}(\\boldsymbol{x})``: Brownian distance variance.
  - $(math_dict[:T])
  - ``\\mathbf{D}``: ``T \\times T`` pairwise distance matrix.
  - ``\\lVert \\cdot \\rVert_F``: Frobenius norm.

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
