"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the pair formulations of the mean distance of a [`BrownianDistanceVariance`](@ref) model.

A pair formulation holds one distance variable ``d_{p}`` for each pair ``p = (i, j)`` of observations with ``i > j``, and a member of this family selects the rows that bound each one from below, ``d_{p} \\geq |x_i - x_j|``. The mean distance of the model is then ``\\frac{2}{T^{2}} \\sum_{p} d_{p}``. Both members state the same feasible set, so both give the same optimum and the same risk. They differ in the type and the number of rows that the solver reads. The risk expression increases in each ``d_{p}``, so the rows are tight when the objective minimises the risk. The model costs ``T (T - 1) / 2`` variables, so [`SortingNetworkOrderedWeightsArray`](@ref) is the default formulation of the mean distance. [`set_brownian_distance_mean!`](@ref) states the rows and the relaxation.

All concrete types implementing a pair formulation should subtype `BrownianDistanceVarianceFormulation`.

# Related

  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BDVarMeanDistanceFormulations`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`set_brownian_distance_mean!`](@ref)
"""
abstract type BrownianDistanceVarianceFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Bounds each pair distance of a [`BrownianDistanceVariance`](@ref) model with a norm-one cone of dimension two.

The model adds one cone for each pair of observations below the diagonal, ``T (T - 1) / 2`` cones in all. Model 10 of [bdvar](@cite) states the same bound as two linear inequalities, which [`IneqBrownianDistanceVariance`](@ref) selects. A solver that has no norm-one cone, such as Clarabel, reads the cone through a bridge to linear rows and one more variable for each pair, so [`IneqBrownianDistanceVariance`](@ref) is the smaller model there.

# Mathematical definition

```math
\\begin{align}
(D_{ij},\\, x_i - x_j) &\\in \\mathcal{K}_{1}\\,, \\quad \\forall\\, 1 \\leq j < i \\leq T\\,.
\\end{align}
```

Where:

  - ``D_{ij}``: The distance variable of the pair ``(i, j)``.
  - ``x_i``, ``x_j``: Portfolio returns at the periods ``i`` and ``j``.
  - $(math_dict[:K_q_norm])
  - $(math_dict[:T])

The cone states ``D_{ij} \\geq |x_i - x_j|``.

# Related

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`IneqBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`set_brownian_distance_mean!`](@ref)
"""
struct NormOneConeBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Bounds each pair distance of a [`BrownianDistanceVariance`](@ref) model with two linear inequalities.

The model adds two blocks of ``T (T - 1) / 2`` inequalities, one row of each block for each pair of observations below the diagonal. These are the rows of model 10 of [bdvar](@cite), with the diagonal and the upper triangle left out. The diagonal distance is zero, and an entry of the upper triangle repeats its mirror.

# Mathematical definition

```math
\\begin{align}
D_{ij} - (x_i - x_j) &\\geq 0\\,, \\quad \\forall\\, 1 \\leq j < i \\leq T\\,, \\\\
D_{ij} + (x_i - x_j) &\\geq 0\\,, \\quad \\forall\\, 1 \\leq j < i \\leq T\\,.
\\end{align}
```

Where:

  - ``D_{ij}``: The distance variable of the pair ``(i, j)``.
  - ``x_i``, ``x_j``: Portfolio returns at the periods ``i`` and ``j``.
  - $(math_dict[:T])

Together the two inequalities state ``D_{ij} \\geq |x_i - x_j|``.

# Related

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`NormOneConeBrownianDistanceVariance`](@ref)
  - [`BrownianDistanceVariance`](@ref)
  - [`set_brownian_distance_mean!`](@ref)
"""
struct IneqBrownianDistanceVariance <: BrownianDistanceVarianceFormulation end
"""
    const BDVarRkFormulations = Union{RSOCRiskExpr, QuadRiskExpr, SquaredSOCRiskExpr}

Groups the three encodings that [`BrownianDistanceVariance`](@ref) accepts for its risk expression.

All three report the bound itself, not a root, so the model reports the value of the functor. [`QuadRiskExpr`](@ref) states the risk as a quadratic form. [`RSOCRiskExpr`](@ref) holds each of the two squares of the bound with a rotated second-order cone, so the risk expression is affine. [`SquaredSOCRiskExpr`](@ref) squares the deviation of one second-order cone. The type bound of the constructor refuses [`SOCRiskExpr`](@ref), because it reports the root. [`set_brownian_distance_risk_constraint!`](@ref) states the expressions.

# Related

  - [`RSOCRiskExpr`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
const BDVarRkFormulations = Union{RSOCRiskExpr, QuadRiskExpr, SquaredSOCRiskExpr}
"""
    const BDVarMeanDistanceFormulations = Union{BrownianDistanceVarianceFormulation,
                                                ExactOrderedWeightsArray,
                                                SortingNetworkOrderedWeightsArray}

Groups the formulations that [`BrownianDistanceVariance`](@ref) accepts for the rows of its mean distance.

The mean pairwise distance of the returns is an ordered weighted average with nondecreasing weights, see [`brownian_distance_owa_weights`](@ref). So the two exact OWA formulations state it as they state any convex OWA, beside the two pair formulations of [`BrownianDistanceVarianceFormulation`](@ref). [`ApproxOrderedWeightsArray`](@ref) is not a member, because it approximates the OWA. [`set_brownian_distance_mean!`](@ref) states the rows of each member.

# Related

  - [`BrownianDistanceVarianceFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`BrownianDistanceVariance`](@ref)
"""
const BDVarMeanDistanceFormulations = Union{BrownianDistanceVarianceFormulation,
                                            ExactOrderedWeightsArray,
                                            SortingNetworkOrderedWeightsArray}
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

The functor and [`set_risk_constraints!`](@ref) compute the same bound, so a reported figure and the objective the optimiser minimised are the same quantity. The model states the closed form. The variance term reads the returns directly, and only the mean distance needs rows, which `alg2` selects.

!!! info

    `alg2` sets the size of the model, and `alg1` sets how the solver reads the two squares. Measured with Clarabel on 5 assets and ``T`` from 50 to 300, under [`MeanRisk`](@ref) with [`MinimumRisk`](@ref), [`MaximumRatio`](@ref) and [`MaximumReturn`](@ref) with a bound, and under [`RiskBudgeting`](@ref), and on 20 assets at ``T = 300`` and ``T = 600``:

      - The default, [`QuadRiskExpr`](@ref) with [`SortingNetworkOrderedWeightsArray`](@ref), solved to `OPTIMAL` in every case on 5 assets at the default tolerances of Clarabel. Its model grows as ``T \\log^{2} T``, so it is the choice for a long sample, and when you do not know the problem. The solve time grows faster than the model: on 20 assets a solve under [`MinimumRisk`](@ref) took 4 s at ``T = 300`` and 60 s at ``T = 600``.
      - [`RSOCRiskExpr`](@ref) gave the same optimum, and at `tol_gap_abs = tol_gap_rel = tol_feas = 1e-11` it was the only expression with `OPTIMAL` in every case on 5 assets with no bound under [`RiskBudgeting`](@ref), where [`QuadRiskExpr`](@ref) stopped at `ALMOST_OPTIMAL` in 3 cases of [`RiskBudgeting`](@ref). At the default tolerances its weights were less accurate: up to `3.7e-4` from the optimum with the sorting network, against `1.0e-4` for [`QuadRiskExpr`](@ref). Choose it with tight tolerances.
      - [`SquaredSOCRiskExpr`](@ref) gave the same optimum, but under [`RiskBudgeting`](@ref) without a bound it stopped at `SLOW_PROGRESS` in some cases.
      - The pair formulations, [`IneqBrownianDistanceVariance`](@ref) and [`NormOneConeBrownianDistanceVariance`](@ref), add ``T (T - 1) / 2`` variables, and each of their rows reads every asset. They gave the most accurate weights at the default tolerances, within `8e-5` of the optimum under [`MinimumRisk`](@ref) and within `1.2e-5` under [`MaximumReturn`](@ref) with a bound, with [`IneqBrownianDistanceVariance`](@ref). On 5 assets they solved as fast as the sorting network or faster. On 20 assets they were slower at ``T = 300``, 10 s against 4 s with [`QuadRiskExpr`](@ref), and they used twice the memory at ``T = 600``, 2.8 GB against 1.3 GB. Choose them for a short sample with few assets. Clarabel has no norm-one cone, so [`IneqBrownianDistanceVariance`](@ref) is the smaller of the two there.
      - [`ExactOrderedWeightsArray`](@ref) adds ``T^{2}`` rows. At `1e-11` it stopped at `ALMOST_OPTIMAL` more often than the sorting network, and with [`RSOCRiskExpr`](@ref) under [`RiskBudgeting`](@ref) it stopped at `SLOW_PROGRESS`.
      - [`QuadRiskExpr`](@ref) with a linear formulation in `alg2` and no bound gives a quadratic programme with linear rows, so a solver with no cones can take it.

    Under [`RiskBudgeting`](@ref) with a bound in `settings.ub`, Clarabel stopped at `SLOW_PROGRESS` in 30 of 36 cases, after the first few steps, and every formulation stopped in some of them. The row of the logarithm of that optimiser adds exponential cones, and Clarabel changes its scaling of those cones when a step is shorter than `min_switch_step_length`, `0.1` by default. With `"min_switch_step_length" => 1e-3` in the settings of the solver, all 36 cases solved, and the risk was within `1.1e-6` of the reference.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BrownianDistanceVariance(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alg1::BDVarRkFormulations = QuadRiskExpr(),
        alg2::BDVarMeanDistanceFormulations = SortingNetworkOrderedWeightsArray()
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
      alg2 ┼ SortingNetworkOrderedWeightsArray
           │      merge ┼ BitonicMerge()
           │   ordering ┴ ComparatorOrdering()
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`BDVarRkFormulations`](@ref)
  - [`BDVarMeanDistanceFormulations`](@ref)
  - [`set_risk_constraints!`](@ref)

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
                                      alg2::BDVarMeanDistanceFormulations)
        return new{typeof(settings), typeof(alg1), typeof(alg2)}(settings, alg1, alg2)
    end
end
function BrownianDistanceVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  alg1::BDVarRkFormulations = QuadRiskExpr(),
                                  alg2::BDVarMeanDistanceFormulations = SortingNetworkOrderedWeightsArray())::BrownianDistanceVariance
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
