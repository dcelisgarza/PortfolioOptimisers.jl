"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rows that tie the distance matrix of a [`BrownianDistanceVariance`](@ref) model to the portfolio returns.

The model holds the distances ``|x_i - x_j|`` in a symmetric matrix variable ``\\mathbf{D}``, and a member of this family selects the rows that bound each entry from below, ``D_{ij} \\geq |x_i - x_j|``. Both members state the same feasible set, so both give the same optimum and the same risk. They differ in the type and the number of rows that the solver reads. The risk expression increases in each entry of ``\\mathbf{D}``, so the rows are tight when the objective minimises the risk. [`set_brownian_distance_variance_constraints!`](@ref) states the rows and the relaxation.

All concrete types implementing a formulation of these rows should subtype `BrownianDistanceVarianceFormulation`.

# Related

  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`set_brownian_distance_variance_constraints!`](@ref)
"""
abstract type BrownianDistanceVarianceFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Bounds each distance of a [`BrownianDistanceVariance`](@ref) model with a norm-one cone of dimension two.

The model adds one cone for each pair of observations on or below the diagonal, ``T (T + 1) / 2`` cones in all. This is the default. Model 10 of [bdvar](@cite) states the same bound as two linear inequalities, which [`IneqBrownianDistanceVariance`](@ref) selects.

# Mathematical definition

```math
\\begin{align}
(D_{ij},\\, x_i - x_j) &\\in \\mathcal{K}_{1}\\,, \\quad \\forall\\, 1 \\leq j \\leq i \\leq T\\,.
\\end{align}
```

Where:

  - ``D_{ij}``: Entry of the symmetric matrix variable of the distances between the portfolio returns.
  - ``x_i``, ``x_j``: Portfolio returns at the periods ``i`` and ``j``.
  - $(math_dict[:K_q_norm])
  - $(math_dict[:T])

The cone states ``D_{ij} \\geq |x_i - x_j|``.

# Related

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`set_brownian_distance_variance_constraints!`](@ref)
"""
struct NormOneConeBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Bounds each distance of a [`BrownianDistanceVariance`](@ref) model with two linear inequalities.

The model adds two matrix rows of ``T^2`` inequalities each. These are the rows of model 10 of [bdvar](@cite). The difference matrix is antisymmetric and the distance matrix is symmetric, so the entry ``(j, i)`` of each matrix row repeats the entry ``(i, j)`` of the other.

# Mathematical definition

```math
\\begin{align}
\\mathbf{D} - \\left( \\boldsymbol{x} \\boldsymbol{1}^\\intercal - \\boldsymbol{1} \\boldsymbol{x}^\\intercal \\right) &\\geq 0\\,, \\\\
\\mathbf{D} + \\left( \\boldsymbol{x} \\boldsymbol{1}^\\intercal - \\boldsymbol{1} \\boldsymbol{x}^\\intercal \\right) &\\geq 0\\,.
\\end{align}
```

Where:

  - ``\\mathbf{D}``: Symmetric ``T \\times T`` matrix variable of the distances between the portfolio returns.
  - $(math_dict[:xret])
  - ``\\boldsymbol{1}``: Column vector of ones, ``T \\times 1``.

Each inequality holds entry by entry, and together they state ``D_{ij} \\geq |x_i - x_j|``.

# Related

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`set_brownian_distance_variance_constraints!`](@ref)
"""
struct IneqBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
"""
    const BDVarRkFormulations = Union{<:RSOCRiskExpr, <:QuadRiskExpr}

Groups the two encodings that [`BrownianDistanceVariance`](@ref) accepts for its risk expression.

Both report the bound itself, not a root. [`QuadRiskExpr`](@ref) states the risk as a quadratic form in the distance matrix. [`RSOCRiskExpr`](@ref) holds each of the two squares of the bound with a rotated second-order cone, so the risk expression is affine and a bound on the risk is a conic row. A bound on the quadratic form needs a bridge to a cone. The type bound of the constructor refuses [`SOCRiskExpr`](@ref) and [`SquaredSOCRiskExpr`](@ref). [`set_brownian_distance_risk_constraint!`](@ref) states the expressions.

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

The distance variance is the mean square of the centred matrix, ``\\mathcal{V}_n^2`` of equation 2.9 of [szekely2007](@cite):

```math
\\begin{align}
\\mathrm{dVar}(\\boldsymbol{x}) &= \\frac{1}{T^2} \\lVert \\mathbf{A} \\rVert_F^2 = \\frac{1}{T^2} \\left( \\mathrm{tr}\\left( \\mathbf{D}^\\intercal \\mathbf{D} \\left( \\mathbf{I}_T - \\frac{2}{T} \\boldsymbol{1}_T \\boldsymbol{1}_T^\\intercal \\right) \\right) + \\frac{1}{T^2} \\left( \\boldsymbol{1}_T^\\intercal \\mathbf{D} \\boldsymbol{1}_T \\right)^2 \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{dVar}(\\boldsymbol{x})``: Distance variance.
  - ``\\lVert \\cdot \\rVert_F``: Frobenius norm.

``\\mathbf{I}_T - \\frac{2}{T} \\boldsymbol{1}_T \\boldsymbol{1}_T^\\intercal`` is indefinite, with eigenvalues ``1`` and ``-1``, so that expression is not disciplined-convex and no solver takes it directly. The trace inequality ``\\mathrm{tr}(\\mathbf{A}\\mathbf{B}) \\leq \\lambda_{\\max}(\\mathbf{A}) \\mathrm{tr}(\\mathbf{B})`` holds for a symmetric ``\\mathbf{A}`` and a positive semi-definite ``\\mathbf{B}``, and here ``\\lambda_{\\max} = 1``. This gives the disciplined-convex upper bound that this type computes, equation 9 of [bdvar](@cite):

```math
\\begin{align}
\\mathrm{BDVar}(\\boldsymbol{x}) &= \\frac{1}{T^2} \\left( \\lVert \\mathbf{D} \\rVert_F^2 + \\frac{1}{T^2} \\left( \\sum_{i,j} D_{ij} \\right)^2 \\right) \\geq \\mathrm{dVar}(\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{BDVar}(\\boldsymbol{x})``: The upper bound on the distance variance, and the value this type reports.

The bound and its gap to the distance variance have closed forms:

```math
\\begin{align}
\\mathrm{BDVar}(\\boldsymbol{x}) &= 2 \\sigma^2(\\boldsymbol{x}) + \\bar{D}^2\\,, \\\\
\\mathrm{BDVar}(\\boldsymbol{x}) - \\mathrm{dVar}(\\boldsymbol{x}) &= \\frac{2}{T^3} \\lVert \\mathbf{D} \\boldsymbol{1}_T \\rVert_2^2\\,.
\\end{align}
```

Where:

  - ``\\sigma^2(\\boldsymbol{x}) = \\frac{1}{T} \\sum_{i} (x_i - \\bar{x})^2``: Variance of the portfolio returns, with the divisor ``T``, and ``\\bar{x}`` their mean.
  - ``\\bar{D} = \\frac{1}{T^2} \\sum_{i,j} D_{ij}``: Mean pairwise distance.

The gap is zero only when ``\\mathbf{D}`` is zero, that is, for a constant series, where both values are zero. Otherwise the bound is several times the statistic: on Gaussian returns at ``T = 50`` it is 6.9 times the distance variance.

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

Computes ``\\mathrm{BDVar}(\\boldsymbol{x})``, the upper bound on the distance variance of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector. [`expected_risk`](@ref) passes the returns net of fees.

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
