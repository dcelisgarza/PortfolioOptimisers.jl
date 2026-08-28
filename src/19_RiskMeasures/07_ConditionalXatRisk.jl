"""
$(DocStringExtensions.TYPEDEF)

Represents the Conditional Value-at-Risk (CVaR) risk measure, also known as Expected Shortfall (ES).

`ConditionalValueatRisk` computes the expected loss given that the loss exceeds the Value-at-Risk at level `alpha`. It provides a coherent risk measure for tail risk quantification.

# Mathematical definition

Let ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal`` be the portfolio returns vector. The CVaR (also known as Expected Shortfall) at level ``\\alpha`` is the expected loss in the worst ``\\alpha`` fraction of scenarios:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{\\nu}{\\min} \\left\\{ -\\nu + \\frac{1}{\\alpha T} \\sum_{t=1}^{T} \\max(\\nu - x_t,\\, 0) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x})``: Conditional Value-at-Risk (Expected Shortfall) at level ``\\alpha``.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``\\nu``: Auxiliary variable (Value-at-Risk threshold).

Equivalently, it is the expected loss conditional on exceeding the VaR:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) &= -\\mathbb{E}\\!\\left[x \\mid x \\leq -\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})\\right]\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x})``: CVaR as the expected loss conditional on exceeding the VaR threshold.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Value-at-Risk at level ``\\alpha``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ConditionalValueatRisk)(x::VecNum)

Computes the CVaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ConditionalValueatRisk()
ConditionalValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)

# References

  - $(ref_dict[:cvar])
  - $(ref_dict[:cajas2025]) Section 7.2.2.4.
"""
@propagatable @concrete struct ConditionalValueatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:oow])
    """
    @pprop w
    function ConditionalValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                                    w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Number = 0.05,
                                w::Option{<:ObsWeights} = nothing)::ConditionalValueatRisk
    return ConditionalValueatRisk(settings, alpha, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Distributionally Robust Conditional Value-at-Risk (DR-CVaR) risk measure.

`DistributionallyRobustConditionalValueatRisk` is a robust variant of CVaR that accounts for distributional uncertainty using Wasserstein ambiguity sets. It provides robustness against model misspecification in the tails of the return distribution. It is the Esfahani-Kuhn [drcvar](@cite) reformulation of a mean-CVaR loss over a Wasserstein ball, and it is a measure of a *portfolio*: it is defined on the weight vector ``\\boldsymbol{w}`` and the scenario matrix, not on a realised return series.

# Mathematical definition

Let ``\\hat{\\mathbb{P}}`` be the empirical distribution of the ``T`` scenarios ``\\boldsymbol{\\xi}_{t}``, and let ``\\mathcal{B}_{r}(\\hat{\\mathbb{P}})`` be the type-1 Wasserstein ball of radius ``r`` around it, restricted to the support ``\\boldsymbol{\\xi} \\geq -\\boldsymbol{1}``. The measure is the worst-case mean of the mean-CVaR loss ``\\ell_{\\tau}`` over that ball, minimised over the Value-at-Risk level ``\\tau``:

```math
\\begin{align}
\\mathrm{DR\\text{-}CVaR}_{\\alpha, l, r}(\\boldsymbol{w}) &= \\min_{\\tau \\in \\mathbb{R}} \\; \\sup_{\\mathbb{Q} \\in \\mathcal{B}_{r}(\\hat{\\mathbb{P}})} \\; \\mathbb{E}_{\\mathbb{Q}}\\left[\\ell_{\\tau}(\\boldsymbol{\\xi})\\right]\\\\
\\ell_{\\tau}(\\boldsymbol{\\xi}) &= -\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi} + l \\left(\\tau + \\dfrac{1}{\\alpha} \\left(-\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi} - \\tau\\right)_{+}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DR\\text{-}CVaR}_{\\alpha, l, r}(\\boldsymbol{w})``: Distributionally Robust CVaR.
  - ``\\boldsymbol{w}``: Portfolio weights vector ``N \\times 1``.
  - ``\\boldsymbol{\\xi}``: Asset returns scenario vector ``N \\times 1``.
  - $(math_dict[:alpha_rm])
  - ``l``: Weight of the CVaR term in the loss, ``l > 0``.
  - ``r``: Wasserstein ball radius, ``r > 0``.
  - ``(\\cdot)_{+} = \\max(\\cdot, 0)``.

The loss ``\\ell_{\\tau}`` is piecewise linear in ``\\boldsymbol{\\xi}`` with the two pieces ``b_{i} \\tau + a_{i} \\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}``, where ``a_{1} = -1``, ``b_{1} = l``, ``a_{2} = -1 - l/\\alpha`` and ``b_{2} = l (1 - 1/\\alpha)``. Taking the dual of the inner supremum gives the conic program that is actually solved:

```math
\\begin{align}
\\min_{\\tau,\\, \\lambda,\\, \\boldsymbol{s},\\, \\boldsymbol{u},\\, \\boldsymbol{v}} \\quad & r \\lambda + \\dfrac{1}{T} \\sum_{t=1}^{T} s_{t}\\\\
\\textrm{s.t.} \\quad & b_{i} \\tau + a_{i} \\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}_{t} + \\boldsymbol{u}_{t}^{(i)\\intercal} \\left(\\boldsymbol{\\xi}_{t} + \\boldsymbol{1}\\right) \\leq s_{t}\\\\
& \\left\\lVert -\\boldsymbol{u}_{t}^{(i)} - a_{i} \\boldsymbol{w} \\right\\rVert_{\\infty} \\leq \\lambda\\\\
& \\boldsymbol{u}_{t}^{(i)} \\geq \\boldsymbol{0} \\quad \\forall\\, t \\in 1 \\ldots T,\\; i \\in \\{1, 2\\}\\,.
\\end{align}
```

The dual variables ``\\boldsymbol{u}_{t}^{(i)}`` price the support constraint ``\\boldsymbol{\\xi} \\geq -\\boldsymbol{1}``, and ``\\lambda`` bounds the dual-norm Lipschitz modulus of the loss. Two consequences follow, and both contradict a reading of the measure as CVaR plus a constant:

  - **The robustness premium ``r \\lambda`` is not a constant.** ``\\lambda`` is a decision variable that depends on ``\\boldsymbol{w}``, so ``r`` cannot be factored out of the optimisation.
  - **The loss carries a mean term that ``l`` does not scale.** At ``r \\to 0`` the ball collapses to ``\\hat{\\mathbb{P}}`` and the measure reduces to ``-\\mathbb{E}[\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}] + l \\, \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi})``, not to ``\\mathrm{CVaR}_{\\alpha}`` alone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionallyRobustConditionalValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        l::Num_AmbTwtCal = 1.0,
        r::Num_AmbRadCal = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `l` is a number: `l > 0` and finite.
  - If `r` is a number: `r > 0` and finite.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::DistributionallyRobustConditionalValueatRisk)(x::VecNum)

Computes the **plain** CVaR of a portfolio returns vector `x`, at level `alpha`. The functor shares its method with [`ConditionalValueatRisk`](@ref) through the `RMCVaR` union, and it ignores `l` and `r`.

This is not an omission. The robust term ``r \\lambda`` is a function of the weight vector and the scenario matrix, and a realised return series carries neither, so it cannot be evaluated here. Use the measure inside a JuMP optimisation to get the robust value.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DistributionallyRobustConditionalValueatRisk()
DistributionallyRobustConditionalValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         l ┼ Float64: 1.0
         r ┼ Float64: 0.02
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)

# References

  - $(ref_dict[:cvar])
  - $(ref_dict[:drcvar])
"""
@propagatable @concrete struct DistributionallyRobustConditionalValueatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:l_wass])
    """
    l
    """
    $(field_dict[:r_wass])
    """
    r
    """
    $(field_dict[:oow])
    """
    @pprop w
    function DistributionallyRobustConditionalValueatRisk(settings::RiskMeasureSettings,
                                                          alpha::Number, l::Num_AmbTwtCal,
                                                          r::Num_AmbRadCal,
                                                          w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_gt0_finite_val(l, :l)
        assert_nonempty_gt0_finite_val(r, :r)
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        end
        return new{typeof(settings), typeof(alpha), typeof(l), typeof(r), typeof(w)}(settings,
                                                                                     alpha,
                                                                                     l, r,
                                                                                     w)
    end
end
function DistributionallyRobustConditionalValueatRisk(;
                                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                      alpha::Number = 0.05,
                                                      l::Num_AmbTwtCal = 1.0,
                                                      r::Num_AmbRadCal = 0.02,
                                                      w::Option{<:ObsWeights} = nothing)::DistributionallyRobustConditionalValueatRisk
    return DistributionallyRobustConditionalValueatRisk(settings, alpha, l, r, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the ambiguity radius `r` and the tail weight `l` of a [`DistributionallyRobustConditionalValueatRisk`](@ref) against prior result `pr`.

Both slots take a **Calibration Rule** in place of the number, so both resolve here. The struct is rebuilt through its ordinary keyword constructor, and that call is what re-runs the positivity check on the calibrated number: a rule that returns a value the slot does not admit is refused at fold time, by the same constructor a caller's own number meets.

The effective observation weights are computed locally as `sel(r.w, pr.w)` and threaded to the rule, so a rule that reads a weighted sample size sees the weights the optimisation settled on. The measure carries no solver, so the rule receives none. That holds on both routes: the third argument carries the effective solver for a measure that has a slot for one, and this measure has none.

A measure whose two slots both hold numbers is returned unchanged, so the common case allocates nothing.

# Related

  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
"""
function resolve_deferred_quantities(x::DistributionallyRobustConditionalValueatRisk,
                                     pr::AbstractPriorResult, ::Any = nothing)
    ws = sel(x.w, pr.w)
    l = resolve_calibration_slot(x.l, :l, pr, ws)
    r = resolve_calibration_slot(x.r, :r, pr, ws)
    return if l === x.l && r === x.r
        x
    else
        DistributionallyRobustConditionalValueatRisk(; settings = x.settings,
                                                     alpha = x.alpha, l = l, r = r, w = x.w)
    end
end
# Calibration slots — see `calibration_slots`. The radius and the tail weight are the two
# quantities of the Esfahani-Kuhn loss that a rule may compute.
function calibration_slots(x::DistributionallyRobustConditionalValueatRisk)
    return (; l = x.l, r = x.r)
end
"""
    const RMCVaR{T} = Union{...}

Parameterised union of [`ConditionalValueatRisk`](@ref) and [`DistributionallyRobustConditionalValueatRisk`](@ref) sharing the same observation-weight type parameter `T`.

Used for unified dispatch on CVaR computation methods.

# Related

  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
"""
const RMCVaR{T} = Union{<:ConditionalValueatRisk{<:Any, <:Any, T},
                        <:DistributionallyRobustConditionalValueatRisk{<:Any, <:Any, <:Any,
                                                                       <:Any, T}}
function (r::RMCVaR{Nothing})(x::VecNum)
    x = copy(x)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end
function (r::RMCVaR{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Conditional Value-at-Risk Range (CVaR Range) risk measure.

`ConditionalValueatRiskRange` evaluates the CVaR at level `alpha` on the portfolio returns and the CVaR at level `beta` on the negated portfolio returns, then sums the two to give the total spread between the downside and the upside expected tail.

# Mathematical definition

```math
\\begin{align}
\\mathrm{CVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) + \\mathrm{CVaR}_{\\beta}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: CVaR range (tail spread).
  - $(math_dict[:xret])
  - ``\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x})``: Lower-tail expected shortfall at level ``\\alpha``.
  - ``\\mathrm{CVaR}_{\\beta}(-\\boldsymbol{x})``: Upper-tail expected surplus at level ``\\beta``.

$(math_dict[:negated_upper_tail])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        beta::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < beta < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ConditionalValueatRiskRange)(x::VecNum)

Computes the CVaR Range of a portfolio returns vector `x`, as the sum of the two tail averages. The functor shares its method with [`DistributionallyRobustConditionalValueatRiskRange`](@ref) through the `RMCVaRRg` union. That method holds the upper tail in the negated convention of [`ConditionalValueatRisk`](@ref), so it writes the sum as `loss - gain`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ConditionalValueatRiskRange()
ConditionalValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
      beta ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)

# References

  - $(ref_dict[:cvar])
  - $(ref_dict[:cajas2025]) Section 7.2.3.
"""
@propagatable @concrete struct ConditionalValueatRiskRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:beta])
    """
    beta
    """
    $(field_dict[:oow])
    """
    @pprop w
    function ConditionalValueatRiskRange(settings::RiskMeasureSettings, alpha::Number,
                                         beta::Number, w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                             alpha, beta, w)
    end
end
function ConditionalValueatRiskRange(;
                                     settings::RiskMeasureSettings = RiskMeasureSettings(),
                                     alpha::Number = 0.05, beta::Number = 0.05,
                                     w::Option{<:ObsWeights} = nothing)::ConditionalValueatRiskRange
    return ConditionalValueatRiskRange(settings, alpha, beta, w)
end
# Tail decomposition — see `range_tails`.
function range_tails(r::ConditionalValueatRiskRange)
    settings = RiskMeasureSettings(; rke = false)
    return (;
            loss = ConditionalValueatRisk(; settings = settings, alpha = r.alpha, w = r.w),
            gain = ConditionalValueatRisk(; settings = settings, alpha = r.beta, w = r.w))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Distributionally Robust Conditional Value-at-Risk Range (DR-CVaR Range) risk measure.

`DistributionallyRobustConditionalValueatRiskRange` evaluates the DR-CVaR at level `alpha` on the portfolio returns and the DR-CVaR at level `beta` on the negated portfolio returns, then sums the two to give the total spread between the downside and the upside robust tail. Each tail carries its own Wasserstein ambiguity parameters.

# Mathematical definition

```math
\\begin{align}
\\mathrm{DR\\text{-}CVaRRange}(\\boldsymbol{x}) &= \\mathrm{DR\\text{-}CVaR}_{\\alpha, l_a, r_a}(\\boldsymbol{x}) + \\mathrm{DR\\text{-}CVaR}_{\\beta, l_b, r_b}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DR\\text{-}CVaRRange}(\\boldsymbol{x})``: DR-CVaR range.
  - $(math_dict[:xret])
  - ``\\mathrm{DR\\text{-}CVaR}_{\\alpha, l_a, r_a}(\\boldsymbol{x})``: Lower-tail DR-CVaR with Wasserstein parameters ``(l_a, r_a)``.
  - ``\\mathrm{DR\\text{-}CVaR}_{\\beta, l_b, r_b}(-\\boldsymbol{x})``: Upper-tail DR-CVaR with Wasserstein parameters ``(l_b, r_b)``.

$(math_dict[:negated_upper_tail])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionallyRobustConditionalValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        l_a::Num_AmbTwtCal = 1.0,
        r_a::Num_AmbRadCal = 0.02,
        beta::Number = 0.05,
        l_b::Num_AmbTwtCal = 1.0,
        r_b::Num_AmbRadCal = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < beta < 1`.
  - Each of `l_a`, `r_a`, `l_b` and `r_b` that is a number: `> 0` and finite.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::DistributionallyRobustConditionalValueatRiskRange)(x::VecNum)

Computes the **plain** CVaR Range of a portfolio returns vector `x`, at levels `alpha` and `beta`. The functor shares its method with [`ConditionalValueatRiskRange`](@ref) through the `RMCVaRRg` union, and it ignores `l_a`, `r_a`, `l_b` and `r_b`.

This is not an omission. Each robust term ``r \\lambda`` is a function of the weight vector and the scenario matrix, and a realised return series carries neither, so it cannot be evaluated here. Use the measure inside a JuMP optimisation to get the robust value.

The method returns the sum of the two tail averages. It holds the upper tail in the negated convention of [`ConditionalValueatRisk`](@ref), so it writes the sum as `loss - gain`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DistributionallyRobustConditionalValueatRiskRange()
DistributionallyRobustConditionalValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
       l_a ┼ Float64: 1.0
       r_a ┼ Float64: 0.02
      beta ┼ Float64: 0.05
       l_b ┼ Float64: 1.0
       r_b ┼ Float64: 0.02
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)

# References

  - $(ref_dict[:cvar])
  - $(ref_dict[:drcvar])
"""
@propagatable @concrete struct DistributionallyRobustConditionalValueatRiskRange <:
                               RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:l_a])
    """
    l_a
    """
    $(field_dict[:r_a])
    """
    r_a
    """
    $(field_dict[:beta])
    """
    beta
    """
    $(field_dict[:l_b])
    """
    l_b
    """
    $(field_dict[:r_b])
    """
    r_b
    """
    $(field_dict[:oow])
    """
    @pprop w
    function DistributionallyRobustConditionalValueatRiskRange(settings::RiskMeasureSettings,
                                                               alpha::Number,
                                                               l_a::Num_AmbTwtCal,
                                                               r_a::Num_AmbRadCal,
                                                               beta::Number,
                                                               l_b::Num_AmbTwtCal,
                                                               r_b::Num_AmbRadCal,
                                                               w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        assert_nonempty_gt0_finite_val(l_a, :l_a)
        assert_nonempty_gt0_finite_val(r_a, :r_a)
        assert_nonempty_gt0_finite_val(l_b, :l_b)
        assert_nonempty_gt0_finite_val(r_b, :r_b)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(l_a), typeof(r_a), typeof(beta),
                   typeof(l_b), typeof(r_b), typeof(w)}(settings, alpha, l_a, r_a, beta,
                                                        l_b, r_b, w)
    end
end
function DistributionallyRobustConditionalValueatRiskRange(;
                                                           settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                           alpha::Number = 0.05,
                                                           l_a::Num_AmbTwtCal = 1.0,
                                                           r_a::Num_AmbRadCal = 0.02,
                                                           beta::Number = 0.05,
                                                           l_b::Num_AmbTwtCal = 1.0,
                                                           r_b::Num_AmbRadCal = 0.02,
                                                           w::Option{<:ObsWeights} = nothing)::DistributionallyRobustConditionalValueatRiskRange
    return DistributionallyRobustConditionalValueatRiskRange(settings, alpha, l_a, r_a,
                                                             beta, l_b, r_b, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the two ambiguity radii and the two tail weights of a [`DistributionallyRobustConditionalValueatRiskRange`](@ref) against prior result `pr`.

Each tail keeps its own pair, so four slots resolve here. It carries the reading of the scalar measure's method unchanged: the rebuild re-runs every positivity check, the effective observation weights are computed locally, and a measure whose four slots all hold numbers is returned unchanged.

The two tails take one role and not two. A radius names no end of the distribution, so a rule placed in the loss-side pair and the same rule placed in the gain-side pair resolve independently, and [`mirror_role`](@ref) has nothing to carry across.

# Related

  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::DistributionallyRobustConditionalValueatRiskRange,
                                     pr::AbstractPriorResult, ::Any = nothing)
    ws = sel(x.w, pr.w)
    slots = (x.l_a, x.r_a, x.l_b, x.r_b)
    l_a, r_a, l_b, r_b = map((slot, key) -> resolve_calibration_slot(slot, key, pr, ws),
                             slots, (:l_a, :r_a, :l_b, :r_b))
    # The comparison is the one `resolve_deferred_quantities` makes over its derived slots:
    # a measure whose four slots all resolved to themselves is returned unchanged.
    return if all(map(===, (l_a, r_a, l_b, r_b), slots))
        x
    else
        DistributionallyRobustConditionalValueatRiskRange(; settings = x.settings,
                                                          alpha = x.alpha, l_a = l_a,
                                                          r_a = r_a, beta = x.beta,
                                                          l_b = l_b, r_b = r_b, w = x.w)
    end
end
# Calibration slots — see `calibration_slots`. Each tail carries its own ambiguity pair.
function calibration_slots(x::DistributionallyRobustConditionalValueatRiskRange)
    return (; l_a = x.l_a, r_a = x.r_a, l_b = x.l_b, r_b = x.r_b)
end
# Tail decomposition — see `range_tails`. Each tail keeps its own ambiguity parameters:
# `l_a`/`r_a` describe the loss-side Wasserstein ball, `l_b`/`r_b` the gain-side one.
function range_tails(r::DistributionallyRobustConditionalValueatRiskRange)
    settings = RiskMeasureSettings(; rke = false)
    return (;
            loss = DistributionallyRobustConditionalValueatRisk(; settings = settings,
                                                                alpha = r.alpha, l = r.l_a,
                                                                r = r.r_a, w = r.w),
            gain = DistributionallyRobustConditionalValueatRisk(; settings = settings,
                                                                alpha = r.beta, l = r.l_b,
                                                                r = r.r_b, w = r.w))
end
"""
    const RMCVaRRg{T} = Union{...}

Parameterised union of [`ConditionalValueatRiskRange`](@ref) and [`DistributionallyRobustConditionalValueatRiskRange`](@ref) sharing the same observation-weight type parameter `T`.

Used for unified dispatch on CVaR-range computation methods.

# Related

  - [`ConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
"""
const RMCVaRRg{T} = Union{<:ConditionalValueatRiskRange{<:Any, <:Any, <:Any, T},
                          <:DistributionallyRobustConditionalValueatRiskRange{<:Any, <:Any,
                                                                              <:Any, <:Any,
                                                                              <:Any, <:Any,
                                                                              <:Any, T}}
function (r::RMCVaRRg{Nothing})(x::VecNum)
    x = copy(x)
    alpha = r.alpha
    aT = alpha * length(x)
    idx1 = ceil(Int, aT)
    var1 = -partialsort!(x, idx1)
    sum_var1 = zero(eltype(x))
    for i in 1:(idx1 - 1)
        sum_var1 += x[i] + var1
    end
    loss = var1 - sum_var1 / aT

    beta = r.beta
    bT = beta * length(x)
    idx2 = ceil(Int, bT)
    var2 = -partialsort!(x, idx2; rev = true)
    sum_var2 = zero(eltype(x))
    for i in 1:(idx2 - 1)
        sum_var2 += x[i] + var2
    end
    gain = var2 - sum_var2 / bT
    return loss - gain
end
function (r::RMCVaRRg{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    loss = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / (alpha)
    end

    # Reverse the **permutation**, never the views. `sorted_x` and `sorted_w` are views, so
    # `reverse!` on them writes through into the caller's `x` and into `r.w` —
    # `get_observation_weights` hands back the stored weights object itself, so the measure
    # would permute its own configuration. `order` was just allocated by `sortperm`, so it is
    # ours to mutate, and the element sequence read below is identical.
    reverse!(order)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    beta = sw * r.beta
    idx = searchsortedfirst(cum_w, beta)
    gain = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (beta - cum_w[idx - 1])) / (beta)
    end
    return loss - gain
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Conditional Drawdown-at-Risk (CDaR) risk measure, also known as Expected Maximum Drawdown.

`ConditionalDrawdownatRisk` computes the expected drawdown given that the drawdown exceeds the Drawdown-at-Risk at level `alpha`. It provides a coherent risk measure for drawdown tail risk.

# Mathematical definition

Define the absolute drawdown series:

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} x_s\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])

The CDaR is the CVaR of the drawdown series ``\\boldsymbol{d} = (d_1, \\ldots, d_T)^\\intercal``:

```math
\\begin{align}
\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{\\nu}{\\min} \\left\\{ -\\nu + \\frac{1}{\\alpha T} \\sum_{t=1}^{T} \\max(\\nu - d_t,\\, 0) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x})``: Conditional Drawdown-at-Risk (Expected Maximum Drawdown).
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:dtdd])
  - ``\\nu``: Auxiliary variable (DaR threshold).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ConditionalDrawdownatRisk)(x::VecNum)

Computes the CDaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ConditionalDrawdownatRisk()
ConditionalDrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
  - [`conditional_drawdown_at_risk`](@ref)

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:cvar])
"""
@propagatable @concrete struct ConditionalDrawdownatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:oow])
    """
    @pprop w
    function ConditionalDrawdownatRisk(settings::RiskMeasureSettings, alpha::Number,
                                       w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   alpha::Number = 0.05,
                                   w::Option{<:ObsWeights} = nothing)::ConditionalDrawdownatRisk
    return ConditionalDrawdownatRisk(settings, alpha, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Distributionally Robust Conditional Drawdown-at-Risk (DR-CDaR) risk measure.

`DistributionallyRobustConditionalDrawdownatRisk` is a robust variant of CDaR that accounts for distributional uncertainty using Wasserstein ambiguity sets, applied to drawdown sequences. It is the drawdown twin of [`DistributionallyRobustConditionalValueatRisk`](@ref): the same Esfahani-Kuhn [drcvar](@cite) reformulation, with the uncompounded portfolio drawdown path in place of the portfolio return.

# Mathematical definition

Let ``d_{t}(\\boldsymbol{w})`` be the uncompounded portfolio drawdown at period ``t``, a non-negative number. The measure is the worst-case mean of the mean-CDaR loss ``\\ell_{\\tau}`` over a type-1 Wasserstein ball of radius ``r`` around the empirical distribution of the drawdown path, minimised over the Drawdown-at-Risk level ``\\tau``:

```math
\\begin{align}
\\mathrm{DR\\text{-}CDaR}_{\\alpha, l, r}(\\boldsymbol{w}) &= \\min_{\\tau \\in \\mathbb{R}} \\; \\sup_{\\mathbb{Q} \\in \\mathcal{B}_{r}(\\hat{\\mathbb{P}})} \\; \\mathbb{E}_{\\mathbb{Q}}\\left[\\ell_{\\tau}(d)\\right]\\\\
\\ell_{\\tau}(d) &= d + l \\left(\\tau + \\dfrac{1}{\\alpha} \\left(d - \\tau\\right)_{+}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DR\\text{-}CDaR}_{\\alpha, l, r}(\\boldsymbol{w})``: Distributionally Robust CDaR.
  - ``\\boldsymbol{w}``: Portfolio weights vector ``N \\times 1``.
  - ``d``: Uncompounded portfolio drawdown, ``d \\geq 0``.
  - $(math_dict[:alpha_rm])
  - ``l``: Weight of the CDaR term in the loss, ``l > 0``.
  - ``r``: Wasserstein ball radius, ``r > 0``.
  - ``(\\cdot)_{+} = \\max(\\cdot, 0)``.

The two pieces of ``\\ell_{\\tau}`` and the dual conic program are those of [`DistributionallyRobustConditionalValueatRisk`](@ref), with ``-d_{t}`` substituted for ``\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}_{t}`` and the drawdown support constraint ``d \\geq -1`` priced by the same non-negative dual variables. The same two consequences hold: ``r`` multiplies a decision variable and cannot be factored out, and the mean drawdown term is not scaled by ``l``. At ``r \\to 0`` the measure reduces to ``\\mathbb{E}[d] + l \\, \\mathrm{CDaR}_{\\alpha}(d)``, not to ``\\mathrm{CDaR}_{\\alpha}`` alone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionallyRobustConditionalDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        l::Num_AmbTwtCal = 1.0,
        r::Num_AmbRadCal = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `l` is a number: `l > 0` and finite.
  - If `r` is a number: `r > 0` and finite.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::DistributionallyRobustConditionalDrawdownatRisk)(x::VecNum)

Computes the **plain** CDaR of a portfolio returns vector `x`, at level `alpha`, and ignores `l` and `r`. The robust term ``r \\lambda`` is a function of the weight vector and the scenario matrix, and a realised return series carries neither, so it cannot be evaluated here. Use the measure inside a JuMP optimisation to get the robust value.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DistributionallyRobustConditionalDrawdownatRisk()
DistributionallyRobustConditionalDrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         l ┼ Float64: 1.0
         r ┼ Float64: 0.02
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`conditional_drawdown_at_risk`](@ref)

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:cvar])
  - $(ref_dict[:drcvar])
"""
@propagatable @concrete struct DistributionallyRobustConditionalDrawdownatRisk <:
                               RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:l_wass])
    """
    l
    """
    $(field_dict[:r_wass])
    """
    r
    """
    $(field_dict[:oow])
    """
    @pprop w
    function DistributionallyRobustConditionalDrawdownatRisk(settings::RiskMeasureSettings,
                                                             alpha::Number,
                                                             l::Num_AmbTwtCal,
                                                             r::Num_AmbRadCal,
                                                             w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_gt0_finite_val(l, :l)
        assert_nonempty_gt0_finite_val(r, :r)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(l), typeof(r), typeof(w)}(settings,
                                                                                     alpha,
                                                                                     l, r,
                                                                                     w)
    end
end
function DistributionallyRobustConditionalDrawdownatRisk(;
                                                         settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                         alpha::Number = 0.05,
                                                         l::Num_AmbTwtCal = 1.0,
                                                         r::Num_AmbRadCal = 0.02,
                                                         w::Option{<:ObsWeights} = nothing)::DistributionallyRobustConditionalDrawdownatRisk
    return DistributionallyRobustConditionalDrawdownatRisk(settings, alpha, l, r, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the ambiguity radius `r` and the tail weight `l` of a [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) against prior result `pr`.

It carries the reading of [`resolve_deferred_quantities`](@ref) on the value-at-risk measure unchanged. The ball is a ball over the returns of the sample, and the drawdown series is built from those returns, so a rule reads the same sample size here as it does there.

# Related

  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::DistributionallyRobustConditionalDrawdownatRisk,
                                     pr::AbstractPriorResult, ::Any = nothing)
    ws = sel(x.w, pr.w)
    l = resolve_calibration_slot(x.l, :l, pr, ws)
    r = resolve_calibration_slot(x.r, :r, pr, ws)
    return if l === x.l && r === x.r
        x
    else
        DistributionallyRobustConditionalDrawdownatRisk(; settings = x.settings,
                                                        alpha = x.alpha, l = l, r = r,
                                                        w = x.w)
    end
end
# Calibration slots — see `calibration_slots`.
function calibration_slots(x::DistributionallyRobustConditionalDrawdownatRisk)
    return (; l = x.l, r = x.r)
end
"""
    const RMCDaR{T} = Union{...}

Parameterised union of [`ConditionalDrawdownatRisk`](@ref) and [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) sharing the same observation-weight type parameter `T`.

Used for unified dispatch on CDaR computation methods.

# Related

  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
"""
const RMCDaR{T} = Union{<:ConditionalDrawdownatRisk{<:Any, <:Any, <:T},
                        <:DistributionallyRobustConditionalDrawdownatRisk{<:Any, <:Any,
                                                                          <:Any, <:Any,
                                                                          <:T}}
"""
    conditional_drawdown_at_risk(dd::VecNum, alpha::Real, ::Nothing) -> Number
    conditional_drawdown_at_risk(dd::VecNum, alpha::Real, w::VecNum) -> Number

Aggregate a drawdown series into its Conditional Drawdown-at-Risk at level `alpha`.

This is the shared aggregation kernel behind [`ConditionalDrawdownatRisk`](@ref), [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) and [`RelativeConditionalDrawdownatRisk`](@ref): the measures differ only in the drawdown series they feed it ([`absolute_drawdown_vec`](@ref) and [`relative_drawdown_vec`](@ref) respectively), so the tail averaging lives here once.

`dd` is **consumed in place** — the unweighted method reorders it via `partialsort!`. Callers pass a freshly computed drawdown vector.

Dispatch on the third argument selects the weighting scheme, so callers resolve observation weights with [`get_observation_weights`](@ref) and let dispatch do the rest.

  - `::Nothing`: unweighted, the mean of the drawdowns beyond the `alpha`-quantile by rank.
  - `w::VecNum`: weighted, the weighted mean of the tail, with the boundary observation contributing only its partial weight.

# Arguments

  - `dd::VecNum`: Drawdown series, all entries ≤ 0. Consumed in place.
  - `alpha::Real`: Significance level, `0 < alpha < 1`.
  - `w`: Resolved observation weights, or `nothing` for the unweighted aggregation.

# Returns

  - `Number`: Conditional Drawdown-at-Risk, returned as a positive loss.

# Related

  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`drawdown_at_risk`](@ref)
"""
function conditional_drawdown_at_risk(dd::VecNum, alpha::Real, ::Nothing)
    aT = alpha * length(dd)
    idx = ceil(Int, aT)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(dd))
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end
function conditional_drawdown_at_risk(dd::VecNum, alpha::Real, w::VecNum)
    sw = sum(w)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_dd[1]
    else
        idx = ifelse(idx > length(dd), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_dd[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_dd[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
function (r::RMCDaR)(x::VecNum)
    return conditional_drawdown_at_risk(absolute_drawdown_vec(x), r.alpha,
                                        get_observation_weights(r.w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Conditional Drawdown-at-Risk risk measure for hierarchical optimisation.

`RelativeConditionalDrawdownatRisk` computes the expected relative (compounded) drawdown given that the drawdown exceeds the Relative Drawdown-at-Risk at level `alpha`.

# Mathematical definition

Define the compounded wealth process and relative drawdown series:

```math
\\begin{align}
C_t &= \\prod_{s=1}^{t} (1 + x_s)\\,, \\\\
rd_t &= \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1 \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:Ct])
  - $(math_dict[:rdt])

The Relative CDaR is the CVaR of the relative drawdown series ``\\boldsymbol{rd}``:

```math
\\begin{align}
\\mathrm{RCDaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{\\nu}{\\min} \\left\\{ -\\nu + \\frac{1}{\\alpha T} \\sum_{t=1}^{T} \\max(\\nu - rd_t,\\, 0) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RCDaR}_{\\alpha}(\\boldsymbol{x})``: Relative Conditional Drawdown-at-Risk.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:rdt])
  - ``\\nu``: Auxiliary variable (RDaR threshold).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeConditionalDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativeConditionalDrawdownatRisk)(x::VecNum)

Computes the Relative CDaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeConditionalDrawdownatRisk()
RelativeConditionalDrawdownatRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`RelativeDrawdownatRisk`](@ref)
  - [`conditional_drawdown_at_risk`](@ref)

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:cvar])
"""
@propagatable @concrete struct RelativeConditionalDrawdownatRisk <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativeConditionalDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                               alpha::Number, w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeConditionalDrawdownatRisk(;
                                           settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                           alpha::Number = 0.05,
                                           w::Option{<:ObsWeights} = nothing)::RelativeConditionalDrawdownatRisk
    return RelativeConditionalDrawdownatRisk(settings, alpha, w)
end
function (r::RelativeConditionalDrawdownatRisk)(x::VecNum)
    return conditional_drawdown_at_risk(relative_drawdown_vec(x), r.alpha,
                                        get_observation_weights(r.w, x))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::ConditionalValueatRisk) = NetReturnsInput()
risk_input_kind(::DistributionallyRobustConditionalValueatRisk) = NetReturnsInput()
risk_input_kind(::ConditionalValueatRiskRange) = NetReturnsInput()
risk_input_kind(::DistributionallyRobustConditionalValueatRiskRange) = NetReturnsInput()
risk_input_kind(::ConditionalDrawdownatRisk) = NetReturnsInput()
risk_input_kind(::DistributionallyRobustConditionalDrawdownatRisk) = NetReturnsInput()
risk_input_kind(::RelativeConditionalDrawdownatRisk) = NetReturnsInput()

export ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
       ConditionalValueatRiskRange, DistributionallyRobustConditionalValueatRiskRange,
       ConditionalDrawdownatRisk, DistributionallyRobustConditionalDrawdownatRisk,
       RelativeConditionalDrawdownatRisk
