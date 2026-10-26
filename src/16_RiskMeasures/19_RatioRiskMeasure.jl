"""
$(DocStringExtensions.TYPEDEF)

Measures the risk of `r1` divided by the risk of `r2`, for a hierarchical optimiser.

`r1` is the numerator and `r2` is the denominator. Both sides are optimisation risk measures, so a hierarchical optimiser can use the ratio as its risk. The ratio follows IEEE division: a denominator of zero gives `Inf` or `-Inf`, and `0 / 0` gives `NaN`. [`bigger_is_better`](@ref) of the ratio is that of `r1`. The ratio is homogeneous of the degree of `r1` minus the degree of `r2`. When the two degrees are equal, the contributions of [`risk_contribution`](@ref) sum to zero and not to the ratio.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RiskRatio}(\\boldsymbol{x}) &= \\frac{r_1(\\boldsymbol{x})}{r_2(\\boldsymbol{x})}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RiskRatio}(\\boldsymbol{x})``: Risk ratio of the portfolio.
  - $(math_dict[:xret])
  - ``r_1``: Risk measure of the numerator.
  - ``r_2``: Risk measure of the denominator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskRatio(;
        r1::OptimisationRiskMeasure = Variance(),
        r2::OptimisationRiskMeasure = ConditionalValueatRisk()
    ) -> RiskRatio

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `r1`: Recursively updated through [`factory`](@ref).
  - `r2`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> RiskRatio()
RiskRatio
  r1 ┼ Variance
     │    settings ┼ RiskMeasureSettings
     │             │   scale ┼ Float64: 1.0
     │             │      ub ┼ nothing
     │             │     rke ┴ Bool: true
     │       sigma ┼ nothing
     │        chol ┼ nothing
     │          rc ┼ nothing
     │         alg ┼ SquaredSOCRiskExpr()
     │    mtx_sqrt ┼ EigenFallbackSquareRoot()
     │   ratio_alg ┼ SquareRootRatio()
     │      corner ┴ nothing
  r2 ┼ ConditionalValueatRisk
     │   settings ┼ RiskMeasureSettings
     │            │   scale ┼ Float64: 1.0
     │            │      ub ┼ nothing
     │            │     rke ┴ Bool: true
     │      alpha ┼ Float64: 0.05
     │          w ┴ nothing
```

On a returns path whose Range is 0.625 and whose worst realisation is 0.25, the ratio is 2.5.

```jldoctest
julia> RiskRatio(; r1 = Range(), r2 = WorstRealisation())([0.125, -0.25, 0.0625, -0.125, 0.375])
2.5
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`OptimisationRiskMeasure`](@ref)
  - [`NonOptimisationRiskRatio`](@ref)
  - [`homogeneity_degree`](@ref)
"""
@propagatable @concrete struct RiskRatio <: HierarchicalRiskMeasure
    """
    $(field_dict[:r1])
    """
    @fprop r1
    """
    $(field_dict[:r2])
    """
    @fprop r2
    function RiskRatio(r1::OptimisationRiskMeasure, r2::OptimisationRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function RiskRatio(; r1::OptimisationRiskMeasure = Variance(),
                   r2::OptimisationRiskMeasure = ConditionalValueatRisk())::RiskRatio
    return RiskRatio(r1, r2)
end
# Deferrable slots — see `deferred_slots`. The two measures carry their own, so both the check
# and the derived recursion in `resolve_deferred_quantities` reach them through the children.
deferred_slots(r::RiskRatio) = (; r1 = r.r1, r2 = r.r2)
"""
$(DocStringExtensions.TYPEDEF)

Measures the risk of `r1` divided by the risk of `r2`, for analysis and for scoring.

No optimiser takes the ratio as an objective or as a constraint, so each side can be any base risk measure, such as [`MeanReturn`](@ref), or a vector of them. `r1` is the numerator and `r2` is the denominator. The ratio follows IEEE division: a denominator of zero gives `Inf` or `-Inf`, and `0 / 0` gives `NaN`.

[`bigger_is_better`](@ref) of the ratio is that of `r1`, so a return over a risk is better when it is larger, and a risk over a return is better when it is smaller. A vector `r1` must agree on its polarity. The ratio is homogeneous of the degree of `r1` minus the degree of `r2` when each side has one degree. When the two degrees are equal, the contributions of [`risk_contribution`](@ref) sum to zero and not to the ratio.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RiskRatio}(\\boldsymbol{x}) &= \\frac{S_1\\left(r_1(\\boldsymbol{x})\\right)}{S_2\\left(r_2(\\boldsymbol{x})\\right)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RiskRatio}(\\boldsymbol{x})``: Risk ratio of the portfolio.
  - $(math_dict[:xret])
  - ``r_1``: Risk measure of the numerator, or a vector of them.
  - ``r_2``: Risk measure of the denominator, or a vector of them.
  - ``S_1``, ``S_2``: The scalarisers `sca1` and `sca2`, which are the identity on a single measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NonOptimisationRiskRatio(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        r1::BaseRM_VecBaseRM = Variance(),
        sca1::Scalariser = SumScalariser(),
        r2::BaseRM_VecBaseRM = ConditionalValueatRisk(),
        sca2::Scalariser = SumScalariser()
    ) -> NonOptimisationRiskRatio

Keywords correspond to the struct's fields.

## Multiplicity

Each side takes one risk measure or a vector of them, and each side has its own scalariser. `sca1` combines `r1`, and `sca2` combines `r2`. Each scalariser field comes directly after the field that it combines, and the two scalarisers can differ.

The two fields take precedence over a `sca` keyword of the caller. The ratio does not pass that keyword to its sides, so its value always uses the scalarisers that the type holds.

## Validation

  - If `r1` is a vector: `!isempty(r1)`.
  - If `r2` is a vector: `!isempty(r2)`.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `r1`: Recursively updated through [`factory`](@ref).
  - `r2`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> NonOptimisationRiskRatio()
NonOptimisationRiskRatio
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
        r1 ┼ Variance
           │    settings ┼ RiskMeasureSettings
           │             │   scale ┼ Float64: 1.0
           │             │      ub ┼ nothing
           │             │     rke ┴ Bool: true
           │       sigma ┼ nothing
           │        chol ┼ nothing
           │          rc ┼ nothing
           │         alg ┼ SquaredSOCRiskExpr()
           │    mtx_sqrt ┼ EigenFallbackSquareRoot()
           │   ratio_alg ┼ SquareRootRatio()
           │      corner ┴ nothing
      sca1 ┼ SumScalariser()
        r2 ┼ ConditionalValueatRisk
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: true
           │      alpha ┼ Float64: 0.05
           │          w ┴ nothing
      sca2 ┴ SumScalariser()
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskRatio`](@ref)
"""
@propagatable @concrete struct NonOptimisationRiskRatio <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:r1_vec])
    """
    @fprop r1
    """
    $(field_dict[:sca_r1])
    """
    sca1
    """
    $(field_dict[:r2_vec])
    """
    @fprop r2
    """
    $(field_dict[:sca_r2])
    """
    sca2
    function NonOptimisationRiskRatio(settings::HierarchicalRiskMeasureSettings,
                                      r1::BaseRM_VecBaseRM, sca1::Scalariser,
                                      r2::BaseRM_VecBaseRM, sca2::Scalariser)
        if isa(r1, VecBaseRM)
            @argcheck(!isempty(r1), IsEmptyError("r1 cannot be empty"))
        end
        if isa(r2, VecBaseRM)
            @argcheck(!isempty(r2), IsEmptyError("r2 cannot be empty"))
        end
        return new{typeof(settings), typeof(r1), typeof(sca1), typeof(r2), typeof(sca2)}(settings,
                                                                                         r1,
                                                                                         sca1,
                                                                                         r2,
                                                                                         sca2)
    end
end
function NonOptimisationRiskRatio(;
                                  settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                  r1::BaseRM_VecBaseRM = Variance(),
                                  sca1::Scalariser = SumScalariser(),
                                  r2::BaseRM_VecBaseRM = ConditionalValueatRisk(),
                                  sca2::Scalariser = SumScalariser())::NonOptimisationRiskRatio
    return NonOptimisationRiskRatio(settings, r1, sca1, r2, sca2)
end
# Deferrable slots — see `deferred_slots`. Each side is one measure or a vector of them, and
# the derived recursion resolves a vector element by element.
deferred_slots(r::NonOptimisationRiskRatio) = (; r1 = r.r1, r2 = r.r2)
"""
    bigger_is_better(r::Union{RiskRatio, NonOptimisationRiskRatio}) -> Bool

Return whether a larger value of the ratio `r` is better, which is the answer of its numerator `r1`.

The ratio grows with its numerator and falls with its denominator. So a return over a risk is better when it is larger, and a risk over a risk, or a risk over a return, is better when it is smaller. [`MeanReturnRiskRatio`](@ref) follows the same rule, and returns `true`. A vector `r1` answers through the method for a vector, which throws an `ArgumentError` when its elements disagree.

# Arguments

  - `r`: The ratio measure.

# Validation

  - When `r1` is a vector, its elements agree on `bigger_is_better`.

# Returns

  - `Bool`: `bigger_is_better(r.r1)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.bigger_is_better(NonOptimisationRiskRatio(; r1 = MeanReturn(),
                                                                     r2 = Variance()))
true

julia> PortfolioOptimisers.bigger_is_better(NonOptimisationRiskRatio(; r1 = Variance(),
                                                                     r2 = MeanReturn()))
false

julia> PortfolioOptimisers.bigger_is_better(RiskRatio())
false
```

# Related

  - [`bigger_is_better`](@ref)
  - [`RiskRatio`](@ref)
  - [`NonOptimisationRiskRatio`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
"""
function bigger_is_better(r::Union{RiskRatio, NonOptimisationRiskRatio})::Bool
    return bigger_is_better(r.r1)
end

export RiskRatio, NonOptimisationRiskRatio
