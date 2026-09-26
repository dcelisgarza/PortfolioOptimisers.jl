"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the `JuMP` encodings of a second moment.

A second-moment risk measure gives the encoding a deviation vector and a correction factor, and the encoding selects the quadratic form or the cone that holds the sum of squares. [`SOCRiskExpr`](@ref) reports the square root of the second moment, and the other three encodings report the second moment itself. The functor of the measure reports the units of its model, and the measure reads a bound in `settings.ub` in the same units.

The encoding sets the risk expression alone. Every consumer also adds a second-order cone variable that bounds the square root of the second moment. The bound in `settings.ub` and the [`MaximumRatio`](@ref) objective act on that variable, because each needs an expression of degree one in the weights. The cone encodings bound the sum of squares from above, so they are tight when the objective minimises the risk or when a bound holds it.

All concrete types implementing a second-moment `JuMP` encoding should subtype `SecondMomentFormulation`.

# Related

  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
"""
abstract type SecondMomentFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the second-moment encodings that state the risk as an explicit square.

[`Variance`](@ref) accepts these two and no others. Both report the variance itself. [`QuadRiskExpr`](@ref) states it as a quadratic form in the weights, and [`SquaredSOCRiskExpr`](@ref) states it as the square of a second-order cone variable.

# Related

  - [`SecondMomentFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`Variance`](@ref)
"""
abstract type VarianceFormulation <: SecondMomentFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Encodes the second moment as an explicit quadratic form in the risk expression.

The risk expression reads no auxiliary variable. It takes two shapes. A risk measure that holds a co-moment matrix uses the first, and a risk measure that builds a deviation vector uses the second.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,,\\\\
R(\\boldsymbol{w}) &= c \\, \\boldsymbol{d}^\\intercal \\boldsymbol{d}\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:w_port])
  - ``\\mathbf{\\Sigma}``: `N × N` co-moment matrix.
  - $(math_dict[:d_secmom])
  - $(math_dict[:c_secmom])

# Related

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`Variance`](@ref)
"""
struct QuadRiskExpr <: VarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Encodes the second moment as the square of a second-order cone variable.

The cone bounds the norm of the deviation vector, and the risk expression squares that variable, so the encoding reports the second moment itself.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= c \\, t^{2}\\,,\\\\
\\text{s.t.} \\quad & \\left\\lVert \\boldsymbol{d} \\right\\rVert_{2} \\leq t\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:d_secmom])
  - $(math_dict[:c_secmom])
  - $(math_dict[:t_secmom])
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm. The model states it with a [JuMP.SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Related

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`Variance`](@ref)
"""
struct SquaredSOCRiskExpr <: VarianceFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Encodes the second moment as a variable that a rotated second-order cone bounds.

The cone holds the square, so the risk expression is linear in the auxiliary variable. The row is `[t; 1/2; d] in JuMP.RotatedSecondOrderCone()`. That cone states ``2 t u \\geq \\lVert \\boldsymbol{d} \\rVert_{2}^{2}``, and the second entry fixes ``u = 1/2``, so the row states ``t \\geq \\lVert \\boldsymbol{d} \\rVert_{2}^{2}``. The encoding reports the second moment itself.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= c \\, t\\,,\\\\
\\text{s.t.} \\quad & \\left\\lVert \\boldsymbol{d} \\right\\rVert_{2}^{2} \\leq t\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:d_secmom])
  - $(math_dict[:c_secmom])
  - $(math_dict[:t_secmom])
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm. The model states its square with a [JuMP.RotatedSecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Rotated-Second-Order-Cone).

# Related

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
"""
struct RSOCRiskExpr <: SecondMomentFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Encodes the square root of the second moment as a second-order cone variable.

This is the only one of the four encodings that reports a root. A risk measure that takes it reports a standard deviation where the other three report a variance, in the model and in the functor alike. The measure reads a bound in `settings.ub` in the same units.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\sqrt{c} \\, t\\,,\\\\
\\text{s.t.} \\quad & \\left\\lVert \\boldsymbol{d} \\right\\rVert_{2} \\leq t\\,.
\\end{align}
```

Where:

  - $(math_dict[:R_w])
  - $(math_dict[:d_secmom])
  - $(math_dict[:c_secmom])
  - $(math_dict[:t_secmom])
  - ``\\lVert \\cdot \\rVert_{2}``: L2 norm. The model states it with a [JuMP.SecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Second-Order-Cone).

# Related

  - [`SecondMomentFormulation`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
"""
struct SOCRiskExpr <: SecondMomentFormulation end
"""
    const NSkeQuadFormulations = Union{<:QuadRiskExpr, <:SquaredSOCRiskExpr}

Groups the two encodings under which [`NegativeSkewness`](@ref) reports the quadratic form of its skewness matrix rather than its square root.

The functor of [`NegativeSkewness`](@ref) dispatches on the group. It returns ``\\boldsymbol{w}^\\intercal \\mathbf{V} \\boldsymbol{w}``, with ``\\mathbf{V}`` the field `V` of the measure, under these two encodings, and the square root under [`SOCRiskExpr`](@ref), so the functor reports the units of the model.

# Related

  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`NegativeSkewness`](@ref)
"""
const NSkeQuadFormulations = Union{<:QuadRiskExpr, <:SquaredSOCRiskExpr}
"""
    const QuadSecondMomentFormulations = Union{<:NSkeQuadFormulations, <:RSOCRiskExpr}

Groups the three encodings that report the second moment itself rather than its square root.

The value level of [`SecondMoment`](@ref) dispatches on the group. It returns the variance of the deviations under these three encodings, and the standard deviation under [`SOCRiskExpr`](@ref), so the functor reports the units of the model.

# Related

  - [`NSkeQuadFormulations`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`SecondMoment`](@ref)
"""
const QuadSecondMomentFormulations = Union{<:NSkeQuadFormulations, <:RSOCRiskExpr}
"""
$(DocStringExtensions.TYPEDEF)

Measures the portfolio variance, the quadratic form of the weights in a covariance matrix.

`alg` selects the risk expression that the model builds, and each [`VarianceFormulation`](@ref) reports the variance itself. [`QuadRiskExpr`](@ref) states the quadratic form ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}``, and [`SquaredSOCRiskExpr`](@ref) states ``\\sigma^{2}``, the square of a second-order cone variable with ``\\sigma \\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_{2}`` and ``\\mathbf{G}^\\intercal \\mathbf{G} = \\mathbf{\\Sigma}``. The model adds ``\\sigma`` under both encodings, because the bound in `settings.ub` and the [`MaximumRatio`](@ref) objective need an expression of degree one in the weights. The bound acts on ``\\sigma`` at the square root of `settings.ub`, so a caller states it in the units of a variance. [`set_risk_constraints!`](@ref) states the model.

When `rc` holds rows, the model uses the semidefinite formulation of [sdprp](@cite) whatever `alg` says. A [`SemiDefinitePhylogeny`](@ref) in the constraints also moves the variance to this formulation. The formulation lifts the weights into a symmetric matrix ``\\mathbf{W}``, and the variance becomes the trace ``\\mathrm{Tr}(\\mathbf{\\Sigma} \\mathbf{W})``, which has degree one in ``(\\boldsymbol{w}, k)``. So under [`MaximumRatio`](@ref) this formulation maximises the excess return per unit of variance, not the Sharpe ratio. The `## The degree of the risk` subsection of [`MaximumRatio`](@ref) states the rule, and [`rc_variance_constraints!`](@ref) states the rows.

!!! warning

    The semidefinite formulation is a relaxation. The model states ``\\mathbf{W} \\succeq \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k`` and not equality, so the rows bind the returned weights only when the solution has ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``. No term of the model forces a matrix of rank one. A solver can add a positive semidefinite part to ``\\mathbf{W}`` that moves the shares the rows constrain, and the solve still reports success. The rows then hold on ``\\mathbf{W}`` while the shares of the returned weights miss them, under a [`MinimumRisk`](@ref) objective as well as a [`MaximumUtility`](@ref) one. To check a result, compute [`risk_contribution`](@ref) of the returned weights, or [`factor_risk_contribution`](@ref) under [`FactorRiskContribution`](@ref), divide it by its sum, and compare the shares with the rows.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Variance}(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} = \\sum_{i=1}^{N} \\mathrm{RC}_{i}(\\boldsymbol{w})\\,,\\\\
\\mathrm{RC}_{i}(\\boldsymbol{w}) &= w_{i} (\\mathbf{\\Sigma} \\boldsymbol{w})_{i}\\,,\\\\
\\mathbf{A} \\, \\mathbf{RC}(\\boldsymbol{w}) &\\leq \\boldsymbol{b} \\, \\mathrm{Variance}(\\boldsymbol{w})\\,,\\\\
\\mathbf{C} \\, \\mathbf{RC}(\\boldsymbol{w}) &= \\boldsymbol{d} \\, \\mathrm{Variance}(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:Sigma_rm])
  - ``\\mathrm{RC}_{i}(\\boldsymbol{w})``: Risk contribution of asset ``i``. The contributions sum to the variance.
  - ``\\mathbf{RC}(\\boldsymbol{w})``: Vector of the ``N`` risk contributions.
  - ``\\mathbf{A}``, ``\\boldsymbol{b}``: The inequality rows of `rc` and their bounds.
  - ``\\mathbf{C}``, ``\\boldsymbol{d}``: The equality rows of `rc` and their targets.

The last two lines state the rows of `rc`. Each row bounds a share of the variance, and the lines hold only when `rc` holds rows. Under [`FactorRiskContribution`](@ref) the rows read the contribution of each factor in place of ``\\mathrm{RC}_{i}``, the value that [`factor_risk_contribution`](@ref) reports, as a share of the whole variance.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Variance(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        sigma::Option{<:SigmaSlot} = nothing,
        chol::Option{<:MatNum} = nothing,
        rc::Option{<:LcE_Lc} = nothing,
        alg::VarianceFormulation = SquaredSOCRiskExpr(),
    ) -> Variance

Keywords correspond to the struct's fields.

## Validation

  - If `sigma` is a matrix, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.
  - If `chol` is a matrix, `!isempty(chol)`.
  - `chol` is `nothing` when `sigma` is `nothing` or a **Deferred Quantity**.

!!! warning

    `sigma` and `chol` are a pair, and a stated `chol` factorises the `sigma` beside it. A caller who wants one consistent pair names `sigma` alone. A matrix leaves the factorisation to the model, and a **Deferred Quantity** fits both from one prior. A caller who states both by hand must make sure that they agree. A stated matrix is also fixed. It crosses a Cross-Validation fold or a subset view as the answer for the whole universe, while a **Deferred Quantity** crosses unresolved and refits on the subset.

## View parameters

`Variance` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method slices a stated `sigma` to the selected assets on both axes. A **Deferred Quantity** passes through unsliced, and then resolves on the subset.
  - The method slices `chol` on its columns alone. Its rows index the factorisation, which the asset selection does not address.
  - The method refuses an `rc` that is a [`LinearConstraint`](@ref). A group constraint cannot be restricted to a part of its own group, and the restriction would break factor risk contribution.
  - `settings`, `rc` and `alg` pass through unchanged.

# Functor

    (r::Variance)(w::VecNum)

Computes ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}`` with the matrix `r.sigma`, the value of the risk expression under both encodings. The functor reads `sigma` as it stands, so a measure whose `sigma` is `nothing` or a **Deferred Quantity** needs [`factory`](@ref) with a prior result first.

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

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`StandardDeviation`](@ref)
  - [`UncertaintySetVariance`](@ref)
  - [`VarianceFormulation`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`scalarise_risk_expression!`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`expected_risk`](@ref)
  - [`risk_contribution`](@ref)
  - [`rc_variance_constraints!`](@ref): registers the rows of `rc` on the lifted matrix.

# References

  - $(ref_dict[:markowitz1952])
  - $(ref_dict[:sdprp]) Formulations 9 and 10.
"""
@propagatable @concrete struct Variance <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:sigma_slot])
    """
    sigma
    """
    $(field_dict[:chol_slot])
    """
    chol
    """
    $(field_dict[:rc])
    """
    rc
    """
    $(field_dict[:alg])
    """
    alg
    function Variance(settings::RiskMeasureSettings, sigma::Option{<:SigmaSlot},
                      chol::Option{<:MatNum}, rc::Option{<:LcE_Lc},
                      alg::VarianceFormulation)::Variance
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
            assert_matrix_issquare(sigma, :sigma)
        end
        if isa(chol, MatNum)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
        end
        assert_derived_slot_has_source(chol, sigma, :chol, :sigma)
        return new{typeof(settings), typeof(sigma), typeof(chol), typeof(rc), typeof(alg)}(settings,
                                                                                           sigma,
                                                                                           chol,
                                                                                           rc,
                                                                                           alg)
    end
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  sigma::Option{<:SigmaSlot} = nothing, chol::Option{<:MatNum} = nothing,
                  rc::Option{<:LcE_Lc} = nothing,
                  alg::VarianceFormulation = SquaredSOCRiskExpr())::Variance
    return Variance(settings, sigma, chol, rc, alg)
end
function (r::Variance)(w::VecNum)
    return LinearAlgebra.dot(w, r.sigma, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `sigma` slot of a [`Variance`](@ref) against prior result `pr`.

`sigma` and `chol` are one pair, so both come from the same fit. The method never meets a stated `chol`, because [`assert_derived_slot_has_source`](@ref) refuses one beside a deferred `sigma` at construction. A covariance estimator gives no factorisation, so `chol` becomes `nothing` and the model derives it from the resolved `sigma`. A prior estimator gives both, so the sparse factorisation of a factor prior reaches the slot unchanged. A measure whose `sigma` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`Variance`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::Variance, pr::AbstractPriorResult, ::Any = nothing)
    if !isa(r.sigma, DeferredQuantity)
        return r
    end
    fitted = fit_deferred_quantity(r.sigma, pr)
    return rebuild_with_slots(r,
                              (; sigma = deferred_quantity(fitted, :sigma),
                               chol = deferred_derived_quantity(fitted, :chol)))
end
# Deferrable slots — see `deferred_slots`. `chol` is derived and never defers on its own.
deferred_slots(r::Variance) = (; sigma = r.sigma)
# The functor is `dot(w, r.sigma, w)`, so an empty `sigma` is refused on a value-level route
# — see `functor_slots`.
functor_slots(r::Variance) = (; sigma = r.sigma)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Variance`](@ref) that holds a covariance matrix, taken from the measure or from the prior result `pr`.

The method selects `sigma` and `chol` as one pair, not field by field. A stated `sigma` with no factorisation keeps `chol = nothing`, because the factorisation of the prior belongs to a different matrix.

# Algorithm

 1. Resolve a **Deferred Quantity** in `sigma` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select the pair `sigma`, `chol` with [`sigma_chol_selector`](@ref). A measure that states neither takes `pr.sigma` and `pr.chol`, and every other measure keeps its own pair.
 3. Build a new `Variance` from the pair and from the `settings`, `rc` and `alg` of `r`.

# Related

  - [`Variance`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`sigma_chol_selector`](@ref)
"""
function factory(r::Variance, pr::AbstractPriorResult, args...; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    sigma, chol = sigma_chol_selector(r.sigma, r.chol, pr)
    return Variance(; settings = r.settings, sigma = sigma, chol = chol, rc = r.rc,
                    alg = r.alg)
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

Measures the portfolio standard deviation, the square root of the variance.

The model states it as a second-order cone variable ``\\sigma \\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_{2}``, which is tight when the objective minimises the risk or when a bound holds it. The variable has degree one in the weights, so the bound in `settings.ub` acts on it directly and a caller states the bound in the units of a standard deviation. [`set_risk_constraints!`](@ref) states the model. Under [`MaximumRatio`](@ref) the objective is the Sharpe ratio.

# Mathematical definition

```math
\\begin{align}
\\mathrm{StandardDeviation}(\\boldsymbol{w}) &= \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:Sigma_rm])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardDeviation(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        sigma::Option{<:SigmaSlot} = nothing,
        chol::Option{<:MatNum} = nothing,
    ) -> StandardDeviation

Keywords correspond to the struct's fields.

## Validation

  - If `sigma` is a matrix, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.
  - If `chol` is a matrix, `!isempty(chol)`.
  - `chol` is `nothing` when `sigma` is `nothing` or a **Deferred Quantity**.

!!! warning

    `sigma` and `chol` are a pair, as in [`Variance`](@ref). A caller who wants one consistent pair names `sigma` alone, and a caller who states both by hand must make sure that they agree. A stated matrix is fixed across a Cross-Validation fold or a subset view, while a **Deferred Quantity** refits on the subset.

## View parameters

`StandardDeviation` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method slices a stated `sigma` to the selected assets on both axes. A **Deferred Quantity** passes through unsliced, and then resolves on the subset.
  - The method slices `chol` on its columns alone. Its rows index the factorisation, which the asset selection does not address.
  - `settings` passes through unchanged.

# Functor

    (r::StandardDeviation)(w::VecNum)

Computes ``\\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}`` with the matrix `r.sigma`. The functor reads `sigma` as it stands, so a measure whose `sigma` is `nothing` or a **Deferred Quantity** needs [`factory`](@ref) with a prior result first.

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

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Variance`](@ref)
  - [`UncertaintySetVariance`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`expected_risk`](@ref)

# References

  - $(ref_dict[:markowitz1952])
"""
@propagatable @concrete struct StandardDeviation <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:sigma_slot])
    """
    sigma
    """
    $(field_dict[:chol_slot])
    """
    chol
    function StandardDeviation(settings::RiskMeasureSettings, sigma::Option{<:SigmaSlot},
                               chol::Option{<:MatNum})::StandardDeviation
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
            assert_matrix_issquare(sigma, :sigma)
        end
        if isa(chol, MatNum)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
        end
        assert_derived_slot_has_source(chol, sigma, :chol, :sigma)
        return new{typeof(settings), typeof(sigma), typeof(chol)}(settings, sigma, chol)
    end
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Option{<:SigmaSlot} = nothing,
                           chol::Option{<:MatNum} = nothing)::StandardDeviation
    return StandardDeviation(settings, sigma, chol)
end
function (r::StandardDeviation)(w::VecNum)
    return sqrt(LinearAlgebra.dot(w, r.sigma, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `sigma` slot of a [`StandardDeviation`](@ref) against prior result `pr`.

`sigma` and `chol` come from the same fit, by the rule that [`resolve_deferred_quantities(r::Variance, pr::AbstractPriorResult)`](@ref) states. A measure whose `sigma` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`StandardDeviation`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::StandardDeviation, pr::AbstractPriorResult,
                                     ::Any = nothing)
    if !isa(r.sigma, DeferredQuantity)
        return r
    end
    fitted = fit_deferred_quantity(r.sigma, pr)
    return rebuild_with_slots(r,
                              (; sigma = deferred_quantity(fitted, :sigma),
                               chol = deferred_derived_quantity(fitted, :chol)))
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::StandardDeviation) = (; sigma = r.sigma)
# See `functor_slots`.
functor_slots(r::StandardDeviation) = (; sigma = r.sigma)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`StandardDeviation`](@ref) that holds a covariance matrix, taken from the measure or from the prior result `pr`.

The method selects `sigma` and `chol` as one pair, as [`factory(r::Variance, pr::AbstractPriorResult, args...; kwargs...)`](@ref) does.

# Algorithm

 1. Resolve a **Deferred Quantity** in `sigma` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select the pair `sigma`, `chol` with [`sigma_chol_selector`](@ref). A measure that states neither takes `pr.sigma` and `pr.chol`, and every other measure keeps its own pair.
 3. Build a new `StandardDeviation` from the pair and from the `settings` of `r`.

# Related

  - [`StandardDeviation`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`sigma_chol_selector`](@ref)
"""
function factory(r::StandardDeviation, pr::AbstractPriorResult, args...; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    sigma, chol = sigma_chol_selector(r.sigma, r.chol, pr)
    return StandardDeviation(; settings = r.settings, sigma = sigma, chol = chol)
end
function port_opt_view(r::StandardDeviation, i, args...)
    sigma = nothing_scalar_array_view(r.sigma, i)
    chol = isnothing(r.chol) ? nothing : view(r.chol, :, i)
    return StandardDeviation(; settings = r.settings, sigma = sigma, chol = chol)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the worst-case portfolio variance over an uncertainty set of covariance matrices.

`ucs` holds the set, as a fitted [`AbstractUncertaintySetResult`](@ref) or as an estimator that the optimisation fits from its returns data or its prior. Each measure holds its own set, so one model can hold more than one worst-case variance. The model states the worst case of the box, the ellipsoid and the norm ball as a dual problem over the lifted weight matrix ``\\mathbf{W}``, and the worst case of the compact set as a second-order cone problem in the weights. [`set_ucs_variance_risk!`](@ref) states the rows and the relaxation. A bound in `settings.ub` is in the units of a variance.

# Mathematical definition

```math
\\begin{align}
\\mathrm{UncertaintySetVariance}(\\boldsymbol{w}) &= \\underset{\\mathbf{\\Sigma} \\in U_{\\mathbf{\\Sigma}}}{\\max} \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}\\,,\\\\
R_{\\mathrm{box}}(\\boldsymbol{w}) &= \\langle \\mathbf{\\Sigma}_u, (\\boldsymbol{w} \\boldsymbol{w}^\\intercal)_{+} \\rangle - \\langle \\mathbf{\\Sigma}_l, (-\\boldsymbol{w} \\boldsymbol{w}^\\intercal)_{+} \\rangle\\,,\\\\
R_{\\mathrm{ell}}(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + k_{e} \\lVert \\mathbf{G}_{\\Omega} \\, \\mathrm{vec}(\\boldsymbol{w} \\boldsymbol{w}^\\intercal) \\rVert_{2}\\,,\\\\
R_{\\mathrm{nb}}(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + \\kappa_{b} \\lVert \\mathbf{L}^\\intercal \\mathrm{vec}(\\boldsymbol{w} \\boldsymbol{w}^\\intercal) \\rVert_{p^{*}}\\,,\\\\
R_{\\mathrm{cpt}}(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + \\kappa \\lVert (\\mathbf{I} - \\mathbf{Q} \\mathbf{Q}^{+}) \\mathbf{C} \\boldsymbol{w} \\rVert_{2}^{2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``U_{\\mathbf{\\Sigma}}``: Uncertainty set of the covariance matrix.
  - ``R_{\\mathrm{box}}``, ``R_{\\mathrm{ell}}``, ``R_{\\mathrm{nb}}``, ``R_{\\mathrm{cpt}}``: Value of the first line for the box, for the ellipsoid and the norm ball without the condition ``\\mathbf{\\Sigma} \\succeq 0``, and for the compact set.
  - ``\\mathbf{\\Sigma}_l``, ``\\mathbf{\\Sigma}_u``: Lower and upper bounds of the box, `lb` and `ub`.
  - ``(\\cdot)_{+}``: Positive part, entry by entry.
  - ``\\hat{\\mathbf{\\Sigma}}``: Centre of the set, the `val` of the set when it states one and the `sigma` of the measure otherwise.
  - ``k_{e}``, ``\\mathbf{G}_{\\Omega}``: Radius of the ellipsoid, `k`, and the upper Cholesky factor of its matrix ``\\mathbf{\\Omega}``, `sigma`.
  - ``\\kappa_{b}``, ``\\mathbf{L}``, ``p^{*}``: Radius of the norm ball, its map, and the dual order of its norm.
  - ``\\mathbf{C}``, ``\\mathbf{Q}``: Diagonal metric of the compact set and its basis. ``\\mathbf{Q}^{+}`` is the pseudo-inverse, so ``\\mathbf{Q} \\mathbf{Q}^{+}`` projects onto the span of ``\\mathbf{Q}``.
  - $(math_dict[:kappa_cpt])
  - ``\\langle \\mathbf{X}, \\mathbf{Y} \\rangle = \\mathrm{Tr}(\\mathbf{X}^\\intercal \\mathbf{Y})``: Inner product of two matrices.

The box value and the compact value are the worst case of the first line. The ellipsoid and the norm ball also require ``\\mathbf{\\Sigma} \\succeq 0``, and their values above omit that condition, so each lies at or above the worst case of its set. The model keeps the condition through the dual matrix ``\\mathbf{E} \\succeq 0``, so its optimum lies at or below these values.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    UncertaintySetVariance(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        ucs::Option{<:UcSE_UcS} = NormalUncertaintySet(),
        sigma::Option{<:SigmaSlot} = nothing,
    ) -> UncertaintySetVariance

Keywords correspond to the struct's fields.

## Validation

  - If `sigma` is a matrix, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.

!!! warning

    A stated `sigma` is fixed. It crosses a Cross-Validation fold or a subset view as the answer for the whole universe, so it does not follow the refit that the optimisation runs, and nothing makes it agree with the set beside it. A caller who wants it to follow the fit names a **Deferred Quantity** in `sigma`, or leaves the slot `nothing` so that the prior supplies it.

## View parameters

`UncertaintySetVariance` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `ucs` recurses through [`port_opt_view`](@ref) with the asset index alone. A set on the covariance axis maps the index to the entries of the vectorised matrix before it slices.
  - The method slices a stated `sigma` to the selected assets on both axes. A **Deferred Quantity** passes through unsliced, and then resolves on the subset.
  - `settings` passes through unchanged.

# Functor

    (r::UncertaintySetVariance)(w::VecNum)

Computes the variance of the weights `w`. The value depends on what `ucs` holds, because an estimator that is not fitted defines no set.

  - `ucs` holds an [`AbstractUncertaintySetResult`](@ref): the value of the definition above for that set, computed by [`ucs_variance`](@ref).
  - `ucs` holds an estimator or `nothing`: the nominal variance ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}``, with ``\\mathbf{\\Sigma}`` the `sigma` of the measure.

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
           │          │           ce ┼ PortfolioOptimisersCovariance
           │          │              │   ce ┼ Covariance
           │          │              │      │    me ┼ SimpleExpectedReturns
           │          │              │      │       │   w ┴ nothing
           │          │              │      │    ce ┼ GeneralCovariance
           │          │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
           │          │              │      │       │    w ┴ nothing
           │          │              │      │   alg ┼ FullMoment()
           │          │              │      │     w ┴ nothing
           │          │              │   mp ┼ MatrixProcessing
           │          │              │      │     pdm ┼ Posdef
           │          │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
           │          │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │          │              │      │      dn ┼ nothing
           │          │              │      │      dt ┼ nothing
           │          │              │      │     alg ┼ nothing
           │          │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
           │          │           me ┼ SimpleExpectedReturns
           │          │              │   w ┴ nothing
           │          │      horizon ┼ nothing
           │          │   fill_limit ┴ nothing
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
  - [`ucs_variance`](@ref)
  - [`set_ucs_variance_risk!`](@ref): The model of the worst case, with its rows and its relaxation.
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`expected_risk`](@ref)

# References

  - $(ref_dict[:robustaa])
  - $(ref_dict[:fengpalomar2016])
  - $(ref_dict[:cajas2025]) Section 11.3.
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
    $(field_dict[:sigma_slot])
    """
    sigma
    function UncertaintySetVariance(settings::RiskMeasureSettings, ucs::Option{<:UcSE_UcS},
                                    sigma::Option{<:SigmaSlot})
        if isa(sigma, MatNum)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
            assert_matrix_issquare(sigma, :sigma)
        end
        return new{typeof(settings), typeof(ucs), typeof(sigma)}(settings, ucs, sigma)
    end
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                ucs::Option{<:UcSE_UcS} = NormalUncertaintySet(),
                                sigma::Option{<:SigmaSlot} = nothing)
    return UncertaintySetVariance(settings, ucs, sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in the `sigma` slot of an [`UncertaintySetVariance`](@ref) against prior result `pr`.

The measure has one slot that the prior supplies, and no derived slot, so [`resolve_slot`](@ref) fills `sigma` alone. A measure whose `sigma` is not a **Deferred Quantity** returns unchanged.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(r::UncertaintySetVariance, pr::AbstractPriorResult,
                                     ::Any = nothing)
    if !isa(r.sigma, DeferredQuantity)
        return r
    end
    return rebuild_with_slots(r, (; sigma = resolve_slot(r.sigma, :sigma, pr)))
end
# Deferrable slots — see `deferred_slots`. `ucs` holds an Estimator by design, not a
# Deferred Quantity, so it is not declared here.
deferred_slots(r::UncertaintySetVariance) = (; sigma = r.sigma)
# See `functor_slots`.
functor_slots(r::UncertaintySetVariance) = (; sigma = r.sigma)
"""
    (r::UncertaintySetVariance)(w::VecNum)

Compute the variance of the weights `w` under an [`UncertaintySetVariance`](@ref).

A fitted [`AbstractUncertaintySetResult`](@ref) in `r.ucs` gives [`ucs_variance`](@ref), the value that the `# Mathematical definition` of [`UncertaintySetVariance`](@ref) states. An estimator or `nothing` in `r.ucs` gives the nominal variance ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}``, with ``\\mathbf{\\Sigma}`` the field `sigma`.

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

Compute the worst-case portfolio variance of the weights `w` over a fitted uncertainty set.

It is the value level of the risk expression that [`set_ucs_variance_risk!`](@ref) builds, at ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal``. The functor of [`UncertaintySetVariance`](@ref) calls it when `ucs` holds a fitted set. The box and the compact values equal the optimum of the model expression. The ellipsoid and the norm-ball values take the dual matrix ``\\mathbf{E} = 0``, so each lies at or above the optimum of its model expression.

# Mathematical definition

```math
\\begin{align}
R_{\\mathrm{box}}(\\boldsymbol{w}) &= \\langle \\mathbf{\\Sigma}_u, (\\boldsymbol{w} \\boldsymbol{w}^\\intercal)_{+} \\rangle - \\langle \\mathbf{\\Sigma}_l, (-\\boldsymbol{w} \\boldsymbol{w}^\\intercal)_{+} \\rangle\\,,\\\\
R_{\\mathrm{ell}}(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + k_{e} \\lVert \\mathbf{G}_{\\Omega} \\, \\mathrm{vec}(\\boldsymbol{w} \\boldsymbol{w}^\\intercal) \\rVert_{2}\\,,\\\\
R_{\\mathrm{nb}}(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + \\kappa_{b} \\lVert \\mathbf{L}^\\intercal \\mathrm{vec}(\\boldsymbol{w} \\boldsymbol{w}^\\intercal) \\rVert_{p^{*}}\\,,\\\\
R_{\\mathrm{cpt}}(\\boldsymbol{w}) &= \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + \\kappa \\underset{\\boldsymbol{z}}{\\min} \\lVert \\mathbf{C} \\boldsymbol{w} - \\mathbf{Q} \\boldsymbol{z} \\rVert_{2}^{2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``R_{\\mathrm{box}}``, ``R_{\\mathrm{ell}}``, ``R_{\\mathrm{nb}}``, ``R_{\\mathrm{cpt}}``: Value for a [`BoxUncertaintySet`](@ref), an [`EllipsoidalUncertaintySet`](@ref), a covariance [`NormBallUncertaintySet`](@ref) and a [`CompactCovarianceUncertaintySet`](@ref).
  - ``\\mathbf{\\Sigma}_l``, ``\\mathbf{\\Sigma}_u``: Lower and upper bounds of the box, `lb` and `ub`.
  - ``(\\cdot)_{+}``: Positive part, entry by entry.
  - ``\\hat{\\mathbf{\\Sigma}}``: Centre of the set, the `val` of the set when it states one and `sigma` otherwise.
  - ``k_{e}``, ``\\mathbf{G}_{\\Omega}``: Radius of the ellipsoid, `k`, and the upper Cholesky factor of its matrix ``\\mathbf{\\Omega}``, `sigma`.
  - ``\\kappa_{b}``, ``\\mathbf{L}``, ``p^{*}``: Radius of the norm ball, its map, and the dual order of its norm. A map with no column adds nothing.
  - ``\\mathbf{C}``, ``\\mathbf{Q}``, ``\\boldsymbol{z}``: Diagonal metric of the compact set, its basis, and the free coefficients of the basis. A basis with no column leaves ``\\lVert \\mathbf{C} \\boldsymbol{w} \\rVert_{2}^{2}``.
  - $(math_dict[:kappa_cpt])
  - ``\\langle \\mathbf{X}, \\mathbf{Y} \\rangle = \\mathrm{Tr}(\\mathbf{X}^\\intercal \\mathbf{Y})``: Inner product of two matrices.

The least-squares problem of the compact line projects onto the span of ``\\mathbf{Q}`` whether or not the columns of ``\\mathbf{Q}`` are orthonormal.

# Arguments

  - `ucs`: Fitted uncertainty set.
  - `sigma::MatNum`: Fallback centre of the set. The `val` of the set wins over it, and the box reads neither.
  - `w::VecNum`: Asset weights.

# Returns

  - `risk::Number`: Worst-case portfolio variance.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`set_ucs_variance_risk!`](@ref): The model expression, whose optimum is at or below this value.
"""
function ucs_variance(ucs::BoxUncertaintySet, ::Any, w::VecNum)
    W = w * transpose(w)
    z = zero(eltype(W))
    return sum(ucs.ub[i] * max(W[i], z) for i in eachindex(ucs.ub, W)) -
           sum(ucs.lb[i] * max(-W[i], z) for i in eachindex(ucs.lb, W))
end
function ucs_variance(ucs::EllipsoidalUncertaintySet, sigma::MatNum, w::VecNum)
    W = w * transpose(w)
    # The set names its own centre; `sigma` is the fallback (ADR 0050).
    sigma = something(ucs.val, sigma)
    G = LinearAlgebra.cholesky(ucs.sigma).U
    return LinearAlgebra.dot(w, sigma, w) + ucs.k * LinearAlgebra.norm(G * vec(W))
end
function ucs_variance(ucs::CompactCovarianceUncertaintySet, sigma::MatNum, w::VecNum)
    # The set names its own centre; `sigma` is the fallback (ADR 0050).
    sigma = something(ucs.val, sigma)
    Cw = ucs.C .* w
    Q = ucs.Q
    # The left division is the least-squares solve the model's `z_cucs` performs, so it
    # projects onto the span of `Q` whether or not the columns of `Q` are orthonormal.
    res = size(Q, 2) > zero(Int) ? Cw - Q * (Q \ Cw) : Cw
    return LinearAlgebra.dot(w, sigma, w) + ucs.kappa * sum(abs2, res)
end
function ucs_variance(ucs::NormBallUncertaintySet{<:Any, <:Any, <:Any,
                                                  <:SigmaUncertaintySetClass},
                      sigma::MatNum, w::VecNum)
    W = w * transpose(w)
    # The set names its own centre; `sigma` is the fallback (ADR 0050).
    sigma = something(ucs.val, sigma)
    # `norm` of an empty vector is zero under every order, so a map with no column pays
    # nothing without a branch.
    penalty = LinearAlgebra.norm(transpose(ucs.L) * vec(W), dual_norm_order(ucs.p))
    return LinearAlgebra.dot(w, sigma, w) + ucs.kappa * penalty
end
"""
    _no_bounds_risk_measure(r, flag)

Return a copy of an [`UncertaintySetVariance`](@ref) without the bound in `settings.ub`, for the sub-problems of [`NearOptimalCentering`](@ref) that solve without bounds.

`flag` is the `ucs_flag` of [`NearOptimalCentering`](@ref), and it selects whether the copy keeps the uncertainty set. Both copies keep `settings.rke` and `settings.scale`.

# Arguments

  - `r`: The measure.
  - `flag`:
      + `::Val{true}` or `nothing`: Keep the uncertainty set, and return an [`UncertaintySetVariance`](@ref).
      + `::Val{false}`: Drop the uncertainty set, and return a [`Variance`](@ref) of the nominal matrix `sigma`.

# Returns

  - `r_new`: The measure without its bound.

# Related

  - [`no_bounds_risk_measure`](@ref)
  - [`_no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`UncertaintySetVariance`](@ref)
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

Return a copy of an [`UncertaintySetVariance`](@ref) without the bound in `settings.ub` and outside the risk of the objective.

A measure that only measures a distance, such as the tracked measure of a [`RiskTrackingRiskMeasure`](@ref), takes this copy. The copy sets `rke = false` and a unit `scale`, as the method for every other measure does. `flag` selects whether the copy keeps the uncertainty set.

# Arguments

  - `r`: The measure.
  - `flag`:
      + `::Val{true}` or `nothing`: Keep the uncertainty set, and return an [`UncertaintySetVariance`](@ref).
      + `::Val{false}`: Drop the uncertainty set, and return a [`Variance`](@ref) of the nominal matrix `sigma`.

# Returns

  - `r_new`: The measure without its bound and outside the risk of the objective.

# Related

  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
  - [`_no_bounds_risk_measure`](@ref)
  - [`UncertaintySetVariance`](@ref)
"""
function _no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance,
                                              ::Union{Val{true}, Nothing})
    return UncertaintySetVariance(;
                                  settings = RiskMeasureSettings(; rke = false,
                                                                 scale = one(r.settings.scale)),
                                  r.ucs, sigma = r.sigma)
end
function _no_bounds_no_risk_expr_risk_measure(r::UncertaintySetVariance, ::Val{false})
    return Variance(;
                    settings = RiskMeasureSettings(; rke = false,
                                                   scale = one(r.settings.scale)),
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

Create an instance of [`UncertaintySetVariance`](@ref) whose empty slots take the uncertainty set `ucs` and the covariance matrix of the prior result `pr`.

A slot that the measure states keeps its value. So `ucs` fills `r.ucs` only when `r.ucs` is `nothing`, and it does not replace a set that the measure holds.

# Algorithm

 1. Resolve a **Deferred Quantity** in `sigma` with [`resolve_deferred_quantities`](@ref), giving `r`.
 2. Select the set with [`ucs_selector`](@ref), giving `r.ucs` when it is not `nothing` and `ucs` otherwise.
 3. Select the matrix with [`nothing_scalar_array_selector`](@ref), giving `r.sigma` when it is not `nothing` and `pr.sigma` otherwise.
 4. Build a new `UncertaintySetVariance` from the set, the matrix and the `settings` of `r`.

# Arguments

  - `r::UncertaintySetVariance`: The measure.
  - `pr::AbstractPriorResult`: The prior result, which supplies `pr.sigma`.
  - `::Any`: A positional argument that the method ignores, such as a solver.
  - `ucs`: The uncertainty set for a measure whose `ucs` is `nothing`.
  - `args...`, `kwargs...`: Ignored.

# Returns

  - `r_new::UncertaintySetVariance`: The new measure.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`ucs_selector`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcSE_UcS} = nothing, args...; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    ucs = ucs_selector(r.ucs, ucs)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`UncertaintySetVariance`](@ref) from the prior result `pr` and the uncertainty set `ucs`, with no positional argument between them.

It takes the steps of [`factory(r::UncertaintySetVariance, pr::AbstractPriorResult, ::Any, ucs, args...; kwargs...)`](@ref). So `ucs` fills `r.ucs` only when `r.ucs` is `nothing`.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`factory`](@ref)
  - [`ucs_selector`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::UncertaintySetVariance, pr::AbstractPriorResult,
                 ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
    r = resolve_deferred_quantities(r, pr)
    ucs = ucs_selector(r.ucs, ucs)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`UncertaintySetVariance`](@ref) from the uncertainty set `ucs` and an optional prior result `pr`.

`ucs` fills `r.ucs` only when `r.ucs` is `nothing`. Without `pr` the method keeps `r.sigma` as it stands, and with `pr` it resolves a **Deferred Quantity** in `sigma` and fills an empty `sigma` with `pr.sigma`.

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
        nothing_scalar_array_selector(resolve_deferred_quantities(r, pr).sigma, pr.sigma)
    end
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
"""
    ucs_risk_measure(r, rd::ReturnsResult)

Fit the uncertainty set of an [`UncertaintySetVariance`](@ref) on the returns data `rd`, so that the set becomes an [`AbstractUncertaintySetResult`](@ref).

[`near_optimal_centering_setup`](@ref) calls it once, so that the risk targets of the barrier, the sub-problem solves and the model of [`NearOptimalCentering`](@ref) read one fitted set. The functor of a measure with a fitted set gives [`ucs_variance`](@ref), so the targets agree with the risk expression of the model.

Three inputs return unchanged. A measure of another type returns unchanged. A set that is already a result passes through [`sigma_ucs`](@ref) unchanged. An estimator that reads a prior result, one for which [`reads_prior_result`](@ref) is `true`, returns unchanged too. Such an estimator is an [`AbstractPriorUncertaintySetEstimator`](@ref), or an estimator of returns data with `pe = nothing`. It is fitted on the prior of the optimisation, and no prior exists when this method runs, so each solve fits it later from its own prior. A vector of measures is fitted element by element.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`sigma_ucs`](@ref)
  - [`reads_prior_result`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
function ucs_risk_measure(r::UncertaintySetVariance, rd::ReturnsResult)
    # The pre-fit runs before any prior exists. An estimator that is calibrated on the
    # optimisation's own prior result has nothing to fit here, so it passes through unchanged
    # and each corner solve fits it inside its own builder, from the prior that solve was
    # handed. The predicate reads the type, so the branch folds.
    return if reads_prior_result(r.ucs)
        r
    else
        UncertaintySetVariance(; settings = r.settings, ucs = sigma_ucs(r.ucs, rd),
                               sigma = r.sigma)
    end
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
