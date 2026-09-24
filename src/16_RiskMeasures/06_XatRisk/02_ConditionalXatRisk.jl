"""
$(DocStringExtensions.TYPEDEF)

Measures the Conditional Value-at-Risk (CVaR), the average loss over the worst `alpha` fraction of the portfolio returns.

The measure is also called Expected Shortfall. It is a coherent risk measure, and it is never below the [`ValueatRisk`](@ref) at the same level.

# Mathematical definition

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{\\nu}{\\min} \\left\\{ -\\nu + \\frac{1}{\\alpha W_{T}} \\sum_{t=1}^{T} w_{t} \\max(\\nu - x_t,\\, 0) \\right\\}\\,.
\\end{align}
```

The minimum has a closed form on the sorted returns. The tail holds the ``k^{\\star} - 1`` smallest returns in full and the boundary return in part, so that its weight is ``\\alpha W_{T}`` exactly:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) &= -\\frac{1}{\\alpha W_{T}} \\left( \\sum_{k=1}^{k^{\\star} - 1} w_{(k)} x_{(k)} + \\left(\\alpha W_{T} - W_{k^{\\star} - 1}\\right) x_{(k^{\\star})} \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x})``: Conditional Value-at-Risk at level ``\\alpha``.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set, and then ``W_{T} = T`` and ``k^{\\star} = \\lceil \\alpha T \\rceil``.
  - $(math_dict[:nu_ru])
  - $(math_dict[:x_k_sorted])
  - $(math_dict[:W_k_cum])
  - $(math_dict[:k_star_tail])

For a continuous distribution the measure equals ``-\\mathbb{E}[x \\mid x \\leq -\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})]``. On a sample that conditional mean can hold more or less than ``\\alpha W_{T}`` of the weight, and the part taken of the boundary return is what corrects it.

# Algorithm

The functor has one method for each kind of `w`.

Without observation weights:

 1. Copy `x`, so that the caller's vector keeps its order.
 2. Compute `aT`, which is ``\\alpha T``, and `idx`, which is ``k^{\\star} = \\lceil \\alpha T \\rceil``.
 3. Partially sort the copy, so that its `idx` smallest entries come first, in ascending order.
 4. Set `var` to minus the entry at `idx`, the Value-at-Risk.
 5. Sum `x[i] + var` over the `idx - 1` entries before it, giving `sum_var`.
 6. Return `var - sum_var / aT`.

With observation weights:

 1. Read the weights `w` with [`get_observation_weights`](@ref), and their sum `sw`, which is ``W_{T}``.
 2. Sort `x` with `sortperm`, giving `order`, and accumulate the sorted weights, giving `cum_w`.
 3. Set `alpha` to `sw * r.alpha`, and find `idx`, the first position at which `cum_w` reaches `alpha`. A rounding error in `cum_w` can put `idx` one past the end, and then `idx` moves back to the last position.
 4. If `idx` is one, return minus the smallest return.
 5. Otherwise, return minus the weighted sum of the `idx - 1` smallest returns plus the boundary return times `alpha - cum_w[idx - 1]`, divided by `alpha`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::ConditionalValueatRisk)(x::VecNum)

Computes the CVaR of a portfolio returns vector `x`, and leaves `x` unchanged.

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
    function ConditionalValueatRisk(settings::RiskMeasureSettings, alpha::Num_SigCal,
                                    w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Num_SigCal = 0.05,
                                w::Option{<:ObsWeights} = nothing)::ConditionalValueatRisk
    return ConditionalValueatRisk(settings, alpha, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::ConditionalValueatRisk) = (; alpha = x.alpha)
"""
$(DocStringExtensions.TYPEDEF)

Measures the worst-case mean-CVaR loss of a portfolio over a Wasserstein ball of distributions around the sample.

It is the distributionally robust mean-CVaR problem of Mohajerin Esfahani and Kuhn [drcvar](@cite). It is a measure of a portfolio, defined on the weights and the scenario matrix, so a realised return series does not carry enough to evaluate it.

# Mathematical definition

Let ``\\hat{\\mathbb{P}}`` be the distribution that puts the weight ``w_{t} / W_{T}`` on scenario ``\\boldsymbol{\\xi}_{t}``. Let ``\\mathcal{B}_{r}(\\hat{\\mathbb{P}})`` be the ball of radius ``r`` around it in the type-1 Wasserstein metric, with the 1-norm as the transport cost, over the distributions supported on ``\\boldsymbol{\\xi} \\geq -\\boldsymbol{1}``. That support says that no asset loses more than all of its value. The measure is the worst-case expected mean-CVaR loss over the ball:

```math
\\begin{align}
\\mathrm{DR\\text{-}CVaR}_{\\alpha, l, r}(\\boldsymbol{w}) &= \\underset{\\tau}{\\min} \\; \\sup_{\\mathbb{Q} \\in \\mathcal{B}_{r}(\\hat{\\mathbb{P}})} \\; \\mathbb{E}_{\\mathbb{Q}}\\left[\\ell_{\\tau}(\\boldsymbol{\\xi})\\right]\\,, \\\\
\\ell_{\\tau}(\\boldsymbol{\\xi}) &= -\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi} + l \\left(\\tau + \\dfrac{1}{\\alpha} \\left(-\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi} - \\tau\\right)_{+}\\right)\\,.
\\end{align}
```

The loss is the larger of two affine pieces ``b_{i} \\tau + a_{i} \\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}``, with ``a_{1} = -1``, ``b_{1} = l``, ``a_{2} = -1 - l/\\alpha`` and ``b_{2} = l (1 - 1/\\alpha)``. The dual of the supremum is a finite convex program:

```math
\\begin{align}
\\mathrm{DR\\text{-}CVaR}_{\\alpha, l, r}(\\boldsymbol{w}) = \\underset{\\tau,\\, \\lambda,\\, \\boldsymbol{s},\\, \\boldsymbol{u}}{\\min} \\quad & r \\lambda + \\dfrac{1}{W_{T}} \\sum_{t=1}^{T} w_{t} s_{t}\\\\
\\textrm{s.t.} \\quad & b_{i} \\tau + a_{i} \\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}_{t} + \\boldsymbol{u}_{t}^{(i)\\intercal} \\left(\\boldsymbol{\\xi}_{t} + \\boldsymbol{1}\\right) \\leq s_{t}\\,,\\\\
& \\left\\lVert -\\boldsymbol{u}_{t}^{(i)} - a_{i} \\boldsymbol{w} \\right\\rVert_{\\infty} \\leq \\lambda\\,,\\\\
& \\boldsymbol{u}_{t}^{(i)} \\geq \\boldsymbol{0} \\quad \\forall\\, t \\in 1 \\ldots T,\\; i \\in \\{1, 2\\}\\,.
\\end{align}
```

This is Equation 27 of the paper, which Corollary 5.1 gives for the support ``C \\boldsymbol{\\xi} \\leq \\boldsymbol{d}`` with ``C = -I`` and ``\\boldsymbol{d} = \\boldsymbol{1}``. The infinity norm is the dual of the 1-norm of the transport cost.

Where:

  - ``\\mathrm{DR\\text{-}CVaR}_{\\alpha, l, r}(\\boldsymbol{w})``: Distributionally Robust CVaR.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\xi}_{t}``: Asset returns of scenario ``t``, ``N \\times 1``.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:l_ek])
  - $(math_dict[:cal_r_radius]) It is the radius of the Wasserstein ball, ``r > 0``.
  - $(math_dict[:tau_ek])
  - $(math_dict[:pos_part])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set.
  - $(math_dict[:W_T_total])
  - ``\\lambda``: Bound on the Lipschitz modulus of the loss in the dual norm.
  - ``s_{t}``: Epigraph variable of the worst-case loss at scenario ``t``.
  - ``\\boldsymbol{u}_{t}^{(i)}``: Dual variables of the support constraint, for scenario ``t`` and piece ``i``, ``N \\times 1``.

Three consequences follow. The first two contradict a reading of the measure as CVaR plus a constant.

  - The robustness premium ``r \\lambda`` is not a constant. ``\\lambda`` depends on ``\\boldsymbol{w}``, so ``r`` cannot be factored out of the optimisation.
  - The loss carries a mean term that ``l`` does not scale. As ``r \\to 0`` the ball collapses to ``\\hat{\\mathbb{P}}``, and the measure goes to ``-\\mathbb{E}_{\\hat{\\mathbb{P}}}[\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}] + l \\, \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi})``, not to ``\\mathrm{CVaR}_{\\alpha}`` alone.
  - As ``r`` grows, the long-only portfolio of minimum measure goes to the equally weighted portfolio. This is Proposition 7.2 of the paper, for this support.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionallyRobustConditionalValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        l::Num_AmbTwtCal = 1.0,
        r::Num_AmbRadCal = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `l` is a number: `l > 0` and finite.
  - If `r` is a number: `r > 0` and finite.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::DistributionallyRobustConditionalValueatRisk)(x::VecNum)

Computes the plain CVaR of a portfolio returns vector `x` at level `alpha`, and ignores `l` and `r`. The functor shares its method with [`ConditionalValueatRisk`](@ref) through the `RMCVaR` union, so the `# Algorithm` of that type is its algorithm.

The robust value needs the weights and the scenario matrix, and a return series carries neither. A JuMP optimisation builds the program of the definition, and the risk it reports is the robust value.

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
  - $(ref_dict[:drcvar]) Section 7.1, Equation 27, Corollary 5.1 and Proposition 7.2.
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
                                                          alpha::Num_SigCal,
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
function DistributionallyRobustConditionalValueatRisk(;
                                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                      alpha::Num_SigCal = 0.05,
                                                      l::Num_AmbTwtCal = 1.0,
                                                      r::Num_AmbRadCal = 0.02,
                                                      w::Option{<:ObsWeights} = nothing)::DistributionallyRobustConditionalValueatRisk
    return DistributionallyRobustConditionalValueatRisk(settings, alpha, l, r, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha`, the tail weight `l` and the ambiguity radius `r` of a [`DistributionallyRobustConditionalValueatRisk`](@ref) against prior result `pr`.

Each of the three slots takes a Calibration Rule in place of a number. The rebuild calls the inner constructor, so a rule that returns a value the slot does not admit meets the same check as a number the caller writes, at fold time.

`alpha` resolves before `l`, because [`TailTermParity`](@ref) prices the tail term at the measure's own significance level. The two form a travelling pair, and the resolved `alpha` reaches the `l` slot in its [`CalibrationContext`](@ref). A number, a plain function and a rule that reads no sibling all ignore that field. The radius reads neither slot.

Both contexts also carry the series that the measure prices. For this measure it is the returns, which is the default of [`calibration_series`](@ref). The method passes it all the same, because the marker belongs to the measure, and a rule carries no marker of its own.

The measure holds no solver, so the rule receives the `slv` that the caller passes, which is `nothing` by default.

# Algorithm

 1. Compute the effective observation weights `ws`, the measure's own `w` or else the weights of `pr`.
 2. Read `s`, the series marker, with [`calibration_series`](@ref).
 3. Resolve `alpha` with [`resolve_calibration_slot`](@ref).
 4. Resolve `l`, with a context that carries the resolved `alpha` and `s`.
 5. Resolve `r`, with a context that carries `s` alone.
 6. Rebuild the measure with [`rebuild_with_slots`](@ref). It returns `x` itself when all three slots already held numbers, so the common case allocates nothing.

# Related

  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series`](@ref)
  - [`TailTermParity`](@ref)
"""
function resolve_deferred_quantities(x::DistributionallyRobustConditionalValueatRisk,
                                     pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, slv)
    l = resolve_calibration_slot(x.l, :l, pr, ws, slv,
                                 CalibrationContext(; alpha = alpha, series = s))
    r = resolve_calibration_slot(x.r, :r, pr, ws, slv, CalibrationContext(; series = s))
    return rebuild_with_slots(x, (; alpha = alpha, l = l, r = r))
end
# Calibration slots — see `calibration_slots`. The significance level, the radius and the
# tail weight are the three quantities of the Esfahani-Kuhn loss that a rule may compute.
function calibration_slots(x::DistributionallyRobustConditionalValueatRisk)
    return (; alpha = x.alpha, l = x.l, r = x.r)
end
"""
    const RMCVaR{T} = Union{...}

Groups the two measures whose functor computes the plain CVaR of a return series.

The robust measure cannot evaluate its robust term on a return series, so it shares the functor of the plain measure, and this union is the type that functor dispatches on. `T` is the type of the observation weights, so `RMCVaR{Nothing}` selects the unweighted method.

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
    partialsort!(x, 1:idx)
    var = -x[idx]
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
        k = idx - 1
        ord = view(order, 1:k)
        -(LinearAlgebra.dot(view(x, ord), view(w, ord)) +
          sorted_x[idx] * (alpha - cum_w[k])) / alpha
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the spread between the two tails of the portfolio returns, the CVaR of the losses plus the CVaR of the gains.

The lower tail is the [`ConditionalValueatRisk`](@ref) at level `alpha` of the returns, and the upper tail is the same measure at level `beta` of the negated returns.

# Mathematical definition

```math
\\begin{align}
\\mathrm{CVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) + \\mathrm{CVaR}_{\\beta}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: CVaR range, the spread between the two tails.
  - $(math_dict[:xret])
  - ``\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x})``: Lower-tail expected shortfall at level ``\\alpha``.
  - ``\\mathrm{CVaR}_{\\beta}(-\\boldsymbol{x})``: Upper-tail expected surplus at level ``\\beta``.

$(math_dict[:negated_upper_tail]) Both tails read the same observation weights.

# Algorithm

The functor has one method for each kind of `w`. Each tail follows the steps of the matching method of [`ConditionalValueatRisk`](@ref).

Without observation weights:

 1. Copy `x`, so that the caller's vector keeps its order.
 2. Partially sort the copy ascending, and compute `loss`, the CVaR at level `alpha` of the returns.
 3. Partially sort the copy descending, and compute `gain`, minus the CVaR at level `beta` of the negated returns.
 4. Return `loss - gain`.

With observation weights:

 1. Sort `x` with `sortperm`, giving `order`, and compute `loss` from the ascending returns.
 2. Reverse `order`, and compute `gain` from the descending returns with the same weights. The method reverses the permutation and never a view, because a view writes through into `x` and into the stored weights.
 3. Return `loss - gain`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        beta::Num_SigCal = alpha,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `beta` is a number: `0 < beta < 1`.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::ConditionalValueatRiskRange)(x::VecNum)

Computes the CVaR Range of a portfolio returns vector `x`, as the sum of the two tail averages, and leaves `x` unchanged. The functor shares its method with [`DistributionallyRobustConditionalValueatRiskRange`](@ref) through the `RMCVaRRg` union. That method holds the upper tail in the negated convention of [`ConditionalValueatRisk`](@ref), so it writes the sum as `loss - gain`.

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
    function ConditionalValueatRiskRange(settings::RiskMeasureSettings, alpha::Num_SigCal,
                                         beta::Num_SigCal, w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                             alpha, beta, w)
    end
end
function ConditionalValueatRiskRange(;
                                     settings::RiskMeasureSettings = RiskMeasureSettings(),
                                     alpha::Num_SigCal = 0.05, beta::Num_SigCal = alpha,
                                     w::Option{<:ObsWeights} = nothing)::ConditionalValueatRiskRange
    return ConditionalValueatRiskRange(settings, alpha, beta, w)
end
# Calibration slots — see `calibration_slots`. One slot per tail, each with its own role.
calibration_slots(x::ConditionalValueatRiskRange) = (; alpha = x.alpha, beta = x.beta)
# Tail decomposition — see `range_tails`.
function range_tails(r::ConditionalValueatRiskRange)
    settings = RiskMeasureSettings(; rke = false)
    return (;
            loss = ConditionalValueatRisk(; settings = settings, alpha = r.alpha, w = r.w),
            gain = ConditionalValueatRisk(; settings = settings, alpha = r.beta, w = r.w))
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the spread between the two robust tails of a portfolio, a worst-case mean-CVaR loss on each side.

Each tail is the program of [`DistributionallyRobustConditionalValueatRisk`](@ref) with its own significance level, tail weight and radius. The paper that states that program treats one tail only, so this measure is the library's generalisation of it to two tails.

# Mathematical definition

```math
\\begin{align}
\\mathrm{DR\\text{-}CVaRRange}(\\boldsymbol{w}) &= \\mathrm{DR\\text{-}CVaR}_{\\alpha, l_a, r_a}(\\boldsymbol{w};\\, \\boldsymbol{\\xi}) + \\mathrm{DR\\text{-}CVaR}_{\\beta, l_b, r_b}(\\boldsymbol{w};\\, -\\boldsymbol{\\xi})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DR\\text{-}CVaRRange}(\\boldsymbol{w})``: DR-CVaR range.
  - $(math_dict[:w_port])
  - ``\\mathrm{DR\\text{-}CVaR}_{\\alpha, l_a, r_a}(\\boldsymbol{w};\\, \\boldsymbol{\\xi})``: Lower-tail DR-CVaR on the scenarios ``\\boldsymbol{\\xi}_{t}``, with the support ``\\boldsymbol{\\xi} \\geq -\\boldsymbol{1}``.
  - ``\\mathrm{DR\\text{-}CVaR}_{\\beta, l_b, r_b}(\\boldsymbol{w};\\, -\\boldsymbol{\\xi})``: Upper-tail DR-CVaR on the negated scenarios ``-\\boldsymbol{\\xi}_{t}``. Its support ``-\\boldsymbol{\\xi} \\geq -\\boldsymbol{1}`` says that no asset gains more than all of its value.

$(math_dict[:negated_upper_tail]) The two tails have separate Wasserstein balls, and both read the same observation weights.

As both radii go to zero, the two mean terms cancel, and the measure goes to ``l_a \\, \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) + l_b \\, \\mathrm{CVaR}_{\\beta}(-\\boldsymbol{x})``, with ``\\boldsymbol{x}`` the portfolio returns of the scenarios.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionallyRobustConditionalValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        l_a::Num_AmbTwtCal = 1.0,
        r_a::Num_AmbRadCal = 0.02,
        beta::Num_SigCal = alpha,
        l_b::Num_AmbTwtCal = 1.0,
        r_b::Num_AmbRadCal = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `beta` is a number: `0 < beta < 1`.
  - Each of `l_a`, `r_a`, `l_b` and `r_b` that is a number: `> 0` and finite.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::DistributionallyRobustConditionalValueatRiskRange)(x::VecNum)

Computes the plain CVaR Range of a portfolio returns vector `x` at levels `alpha` and `beta`, and ignores `l_a`, `r_a`, `l_b` and `r_b`. The functor shares its method with [`ConditionalValueatRiskRange`](@ref) through the `RMCVaRRg` union, so the `# Algorithm` of that type is its algorithm.

The robust value needs the weights and the scenario matrix, and a return series carries neither. A JuMP optimisation builds the two programs of the definition, and the risk it reports is the robust value.

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
  - $(ref_dict[:drcvar]) Section 7.1, Equation 27.
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
                                                               alpha::Num_SigCal,
                                                               l_a::Num_AmbTwtCal,
                                                               r_a::Num_AmbRadCal,
                                                               beta::Num_SigCal,
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
                                                           alpha::Num_SigCal = 0.05,
                                                           l_a::Num_AmbTwtCal = 1.0,
                                                           r_a::Num_AmbRadCal = 0.02,
                                                           beta::Num_SigCal = alpha,
                                                           l_b::Num_AmbTwtCal = 1.0,
                                                           r_b::Num_AmbRadCal = 0.02,
                                                           w::Option{<:ObsWeights} = nothing)::DistributionallyRobustConditionalValueatRiskRange
    return DistributionallyRobustConditionalValueatRiskRange(settings, alpha, l_a, r_a,
                                                             beta, l_b, r_b, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the two significance levels, the two tail weights and the two ambiguity radii of a [`DistributionallyRobustConditionalValueatRiskRange`](@ref) against prior result `pr`.

Each tail keeps its own level, weight and radius, so six slots resolve here. The method keeps the rules of the scalar measure. The rebuild re-runs every check of the constructor, and the method returns a measure whose six slots all hold numbers unchanged.

Each tail weight reads the level of its own tail. `l_a` reads `alpha` and `l_b` reads `beta`, so on a skewed sample [`TailTermParity`](@ref) gives the two tails two different weights, which is what the rule is for on a range measure. A radius reads neither level, and each radius resolves independently of the other.

Both tails price one series, the returns, so the same marker stands in all three contexts. The series belongs to the measure, and the level belongs to a tail.

# Algorithm

 1. Compute the effective observation weights `ws`, the measure's own `w` or else the weights of `pr`.
 2. Read `se`, the series marker, with [`calibration_series`](@ref).
 3. Resolve `alpha` and `beta` with [`resolve_calibration_slot`](@ref).
 4. Build three contexts: `cs` carries `se` alone, `ca` carries the resolved `alpha` and `se`, and `cb` carries the resolved `beta` and `se`.
 5. Resolve `l_a` in `ca`, `r_a` in `cs`, `l_b` in `cb` and `r_b` in `cs`.
 6. Rebuild the measure with [`rebuild_with_slots`](@ref), which returns `x` itself when every slot already held a number.

# Related

  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series`](@ref)
  - [`TailTermParity`](@ref)
"""
function resolve_deferred_quantities(x::DistributionallyRobustConditionalValueatRiskRange,
                                     pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    se = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, slv)
    beta = resolve_calibration_slot(x.beta, :beta, pr, ws, slv)
    # Each tail weight reads the probability of its OWN end, and the two radii read
    # neither, so the four slots take three contexts. Both ends price one series, so the
    # same marker stands in all three. The occupant itself is passed through untouched, so
    # `rebuild_with_slots` compares each resolved value against the field of `x`.
    cs = CalibrationContext(; series = se)
    ca = CalibrationContext(; alpha = alpha, series = se)
    cb = CalibrationContext(; alpha = beta, series = se)
    l_a, r_a, l_b, r_b = map((slot, key, ctx) -> resolve_calibration_slot(slot, key, pr, ws,
                                                                          slv, ctx),
                             (x.l_a, x.r_a, x.l_b, x.r_b), (:l_a, :r_a, :l_b, :r_b),
                             (ca, cs, cb, cs))
    return rebuild_with_slots(x,
                              (; alpha = alpha, l_a = l_a, r_a = r_a, beta = beta,
                               l_b = l_b, r_b = r_b))
end
# Calibration slots — see `calibration_slots`. Each tail carries its own significance level
# and its own ambiguity pair.
function calibration_slots(x::DistributionallyRobustConditionalValueatRiskRange)
    return (; alpha = x.alpha, l_a = x.l_a, r_a = x.r_a, beta = x.beta, l_b = x.l_b,
            r_b = x.r_b)
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

Groups the two measures whose functor computes the plain CVaR Range of a return series.

The robust range cannot evaluate its robust terms on a return series, so it shares the functor of the plain range, and this union is the type that functor dispatches on. `T` is the type of the observation weights, so `RMCVaRRg{Nothing}` selects the unweighted method.

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
    partialsort!(x, 1:idx1)
    var1 = -x[idx1]
    sum_var1 = zero(eltype(x))
    for i in 1:(idx1 - 1)
        sum_var1 += x[i] + var1
    end
    loss = var1 - sum_var1 / aT

    beta = r.beta
    bT = beta * length(x)
    idx2 = ceil(Int, bT)
    partialsort!(x, 1:idx2; rev = true)
    var2 = -x[idx2]
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
        k = idx - 1
        ord = view(order, 1:k)
        -(LinearAlgebra.dot(view(x, ord), view(w, ord)) +
          sorted_x[idx] * (alpha - cum_w[k])) / (alpha)
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
        k = idx - 1
        ord = view(order, 1:k)
        -(LinearAlgebra.dot(view(x, ord), view(w, ord)) + sorted_x[idx] * (beta - cum_w[k])) /
        (beta)
    end
    return loss - gain
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Conditional Drawdown-at-Risk (CDaR), the average of the worst `alpha` fraction of the uncompounded drawdowns.

It is the [`ConditionalValueatRisk`](@ref) of the drawdown series in place of the returns. The size of each drawdown is a convex and positively homogeneous function of the returns, so the measure is too.

# Mathematical definition

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} x_s\\,, \\quad c_0 = 0\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s\\,, \\\\
\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{\\nu}{\\min} \\left\\{ -\\nu + \\frac{1}{\\alpha W_{T}} \\sum_{t=1}^{T} w_{t} \\max(\\nu - d_t,\\, 0) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x})``: Conditional Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:xret])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set.
  - $(math_dict[:W_T_total])
  - $(math_dict[:nu_ru])

The running peak starts at ``c_0 = 0``, so a series that falls from its first period is in drawdown from the start. The minimum has the sorted closed form that [`conditional_drawdown_at_risk`](@ref) states.

# Algorithm

 1. Compute the drawdown series with [`absolute_drawdown_vec`](@ref).
 2. Read the observation weights with [`get_observation_weights`](@ref).
 3. Reduce the drawdown series to its tail average with [`conditional_drawdown_at_risk`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::ConditionalDrawdownatRisk)(x::VecNum)

Computes the CDaR of a portfolio returns vector `x`, and leaves `x` unchanged.

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
    function ConditionalDrawdownatRisk(settings::RiskMeasureSettings, alpha::Num_SigCal,
                                       w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   alpha::Num_SigCal = 0.05,
                                   w::Option{<:ObsWeights} = nothing)::ConditionalDrawdownatRisk
    return ConditionalDrawdownatRisk(settings, alpha, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::ConditionalDrawdownatRisk) = (; alpha = x.alpha)
"""
$(DocStringExtensions.TYPEDEF)

Measures the worst-case mean-CDaR loss of a portfolio over a Wasserstein ball of drawdown scenarios around the sample.

It is the library's generalisation of the program of [`DistributionallyRobustConditionalValueatRisk`](@ref) to drawdowns. The paper that states that program treats returns only.

# Mathematical definition

Let ``d_{t}(\\boldsymbol{w}) \\geq 0`` be the uncompounded drawdown of the portfolio at period ``t``, and let ``\\mathbf{D}_{t}`` be the vector of the uncompounded drawdowns of the assets at period ``t``, each ``\\leq 0``. The loss is the mean-CDaR loss of the portfolio drawdown:

```math
\\begin{align}
\\ell_{\\tau}(d) &= d + l \\left(\\tau + \\dfrac{1}{\\alpha} \\left(d - \\tau\\right)_{+}\\right)\\,.
\\end{align}
```

The measure ``\\mathrm{DR\\text{-}CDaR}_{\\alpha, l, r}(\\boldsymbol{w})`` is the program of [`DistributionallyRobustConditionalValueatRisk`](@ref) with two substitutions.

  - The rows read the portfolio drawdown, so ``-d_{t}(\\boldsymbol{w})`` takes the place of ``\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}_{t}``.
  - The transport cost reads the asset drawdowns, so ``\\mathbf{D}_{t}`` takes the place of ``\\boldsymbol{\\xi}_{t}``, so the support constraint is ``\\mathbf{D} \\geq -\\boldsymbol{1}``.

Where:

  - ``\\mathrm{DR\\text{-}CDaR}_{\\alpha, l, r}(\\boldsymbol{w})``: Distributionally Robust CDaR.
  - $(math_dict[:w_port])
  - ``d_{t}(\\boldsymbol{w})``: Uncompounded drawdown of the portfolio at period ``t``, the running peak of the cumulative return minus its current value, with the peak starting at zero.
  - ``\\mathbf{D}_{t}``: Uncompounded drawdowns of the assets at period ``t``, ``N \\times 1``.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:l_ek])
  - $(math_dict[:cal_r_radius]) It is the radius of the Wasserstein ball, ``r > 0``.
  - $(math_dict[:tau_ek])
  - $(math_dict[:pos_part])

Three consequences follow.

  - The radius multiplies a decision variable, so it cannot be factored out, and ``l`` does not scale the mean drawdown term. As ``r \\to 0`` the measure goes to ``\\mathbb{E}[d] + l \\, \\mathrm{CDaR}_{\\alpha}(d)``, not to ``\\mathrm{CDaR}_{\\alpha}`` alone.
  - The rows read ``d_{t}(\\boldsymbol{w})``, and the robust premium reads the linear function ``-\\boldsymbol{w}^{\\intercal} \\mathbf{D}_{t}``. For long-only weights ``d_{t}(\\boldsymbol{w}) \\leq -\\boldsymbol{w}^{\\intercal} \\mathbf{D}_{t}``, with equality when every asset with a positive weight has its peak in the same period as the portfolio.
  - The sample lies inside the support when no asset's cumulative return falls more than one below its running peak.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionallyRobustConditionalDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        l::Num_AmbTwtCal = 1.0,
        r::Num_AmbRadCal = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `l` is a number: `l > 0` and finite.
  - If `r` is a number: `r > 0` and finite.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::DistributionallyRobustConditionalDrawdownatRisk)(x::VecNum)

Computes the plain CDaR of a portfolio returns vector `x` at level `alpha`, and ignores `l` and `r`. The functor shares its method with [`ConditionalDrawdownatRisk`](@ref) through the `RMCDaR` union, so the `# Algorithm` of that type is its algorithm.

The robust value needs the weights and the asset drawdowns, and a return series carries neither. A JuMP optimisation builds the program of the definition, and the risk it reports is the robust value.

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
  - $(ref_dict[:drcvar]) Section 7.1, Equation 27.
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
                                                             alpha::Num_SigCal,
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
                                                         alpha::Num_SigCal = 0.05,
                                                         l::Num_AmbTwtCal = 1.0,
                                                         r::Num_AmbRadCal = 0.02,
                                                         w::Option{<:ObsWeights} = nothing)::DistributionallyRobustConditionalDrawdownatRisk
    return DistributionallyRobustConditionalDrawdownatRisk(settings, alpha, l, r, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha`, the tail weight `l` and the ambiguity radius `r` of a [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) against prior result `pr`.

The steps are those of the method for [`DistributionallyRobustConditionalValueatRisk`](@ref), except for the series. This measure prices drawdowns, so [`calibration_series`](@ref) gives [`AbsoluteDrawdownSeries`](@ref), and the marker reaches both the `l` slot and the `r` slot. [`TailTermParity`](@ref) then prices the mean drawdown of each column against the ``\\mathrm{CDaR}_{\\alpha}`` of that column, and a radius rule reads its scale off the drawdown sample. The keys `:l` and `:r` are the same on the two measures, so the marker is the only thing that tells a rule which quantity it prices.

A drawdown series holds one entry per observation, so a rule reads the same sample size here as on the returns.

# Algorithm

 1. Compute the effective observation weights `ws`, the measure's own `w` or else the weights of `pr`.
 2. Read `s`, the series marker, which is [`AbsoluteDrawdownSeries`](@ref).
 3. Resolve `alpha` with [`resolve_calibration_slot`](@ref).
 4. Resolve `l`, with a context that carries the resolved `alpha` and `s`.
 5. Resolve `r`, with a context that carries `s` alone.
 6. Rebuild the measure with [`rebuild_with_slots`](@ref), which returns `x` itself when all three slots already held numbers.

# Related

  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`calibration_slots`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`TailTermParity`](@ref)
"""
function resolve_deferred_quantities(x::DistributionallyRobustConditionalDrawdownatRisk,
                                     pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, slv)
    l = resolve_calibration_slot(x.l, :l, pr, ws, slv,
                                 CalibrationContext(; alpha = alpha, series = s))
    r = resolve_calibration_slot(x.r, :r, pr, ws, slv, CalibrationContext(; series = s))
    return rebuild_with_slots(x, (; alpha = alpha, l = l, r = r))
end
# Calibration slots — see `calibration_slots`.
function calibration_slots(x::DistributionallyRobustConditionalDrawdownatRisk)
    return (; alpha = x.alpha, l = x.l, r = x.r)
end
# Calibration series — see `calibration_series`. Both the loss this measure prices and the
# matrix its own transport cost is measured against are the per-asset absolute drawdowns of
# the sample, so a rule that carries units reads that quantity and not the returns.
function calibration_series(::DistributionallyRobustConditionalDrawdownatRisk)
    return AbsoluteDrawdownSeries()
end
"""
    const RMCDaR{T} = Union{...}

Groups the two measures whose functor computes the plain CDaR of a return series.

The robust measure cannot evaluate its robust term on a return series, so it shares the functor of the plain measure, and this union is the type that functor dispatches on. `T` bounds the type of the observation weights. The functor dispatches on the whole union and lets [`conditional_drawdown_at_risk`](@ref) select the weighted or the unweighted method.

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

Reduces a drawdown series to its Conditional Drawdown-at-Risk at level `alpha`.

[`ConditionalDrawdownatRisk`](@ref), [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) and [`RelativeConditionalDrawdownatRisk`](@ref) all call it. They differ only in the drawdown series they pass, from [`absolute_drawdown_vec`](@ref) or [`relative_drawdown_vec`](@ref). The unweighted method reorders `dd` in place, so a caller passes a vector it computed for the call.

# Mathematical definition

```math
\\begin{align}
\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{d}) &= -\\frac{1}{\\alpha W_{T}} \\left( \\sum_{k=1}^{k^{\\star} - 1} w_{(k)} d_{(k)} + \\left(\\alpha W_{T} - W_{k^{\\star} - 1}\\right) d_{(k^{\\star})} \\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{d})``: Conditional Drawdown-at-Risk of the drawdown series, a non-negative loss.
  - ``\\boldsymbol{d}``: Drawdown series ``T \\times 1``, each entry ``\\leq 0``.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when `w` is `nothing`, and then ``W_{T} = T`` and ``k^{\\star} = \\lceil \\alpha T \\rceil``.
  - $(math_dict[:x_k_sorted]) Here the series is ``\\boldsymbol{d}``, so the entry is ``d_{(k)}``.
  - $(math_dict[:W_k_cum])
  - $(math_dict[:k_star_tail])

It is the closed form of the Rockafellar-Uryasev minimum in the definition of [`ConditionalValueatRisk`](@ref), with the drawdowns in place of the returns.

# Algorithm

The third argument selects the method.

`nothing`:

 1. Compute `aT`, which is ``\\alpha T``, and `idx`, which is ``\\lceil \\alpha T \\rceil``.
 2. Partially sort `dd` in place, so that its `idx` smallest entries come first, in ascending order.
 3. Set `var` to minus the entry at `idx`.
 4. Sum `dd[i] + var` over the `idx - 1` entries before it, giving `sum_var`.
 5. Return `var - sum_var / aT`.

A weights vector `w`:

 1. Sort `dd` with `sortperm`, giving `order`, and accumulate the sorted weights, giving `cum_w`.
 2. Set `alpha` to `sum(w) * alpha`, and find `idx`, the first position at which `cum_w` reaches it. A rounding error in `cum_w` can put `idx` one past the end, and then `idx` moves back to the last position.
 3. If `idx` is one, return minus the smallest drawdown.
 4. Otherwise, return minus the weighted sum of the `idx - 1` smallest drawdowns plus the boundary drawdown times `alpha - cum_w[idx - 1]`, divided by `alpha`.

# Arguments

  - `dd::VecNum`: Drawdown series, each entry ``\\leq 0``. The unweighted method reorders it.
  - `alpha::Real`: Significance level, `0 < alpha < 1`.
  - `w`: Observation weights from [`get_observation_weights`](@ref), or `nothing` for the unweighted method.

# Returns

  - `Number`: Conditional Drawdown-at-Risk, returned as a positive loss.

# Related

  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`empirical_value_at_risk`](@ref)
"""
function conditional_drawdown_at_risk(dd::VecNum, alpha::Real, ::Nothing)
    aT = alpha * length(dd)
    idx = ceil(Int, aT)
    partialsort!(dd, 1:idx)
    var = -dd[idx]
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
        k = idx - 1
        ord = view(order, 1:k)
        -(LinearAlgebra.dot(view(dd, ord), view(w, ord)) +
          sorted_dd[idx] * (alpha - cum_w[k])) / alpha
    end
end
function (r::RMCDaR)(x::VecNum)
    return conditional_drawdown_at_risk(absolute_drawdown_vec(x), r.alpha,
                                        get_observation_weights(r.w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Relative Conditional Drawdown-at-Risk, the average of the worst `alpha` fraction of the compounded drawdowns.

It is the [`ConditionalValueatRisk`](@ref) of the relative drawdown series. The compounded wealth is a product of linear functions of the weights, so the library states no JuMP formulation of the measure, and it is a [`HierarchicalRiskMeasure`](@ref).

# Mathematical definition

```math
\\begin{align}
C_t &= \\prod_{s=1}^{t} (1 + x_s)\\,, \\quad C_0 = 1\\,, \\\\
rd_t &= \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1\\,, \\\\
\\mathrm{RCDaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{\\nu}{\\min} \\left\\{ -\\nu + \\frac{1}{\\alpha W_{T}} \\sum_{t=1}^{T} w_{t} \\max(\\nu - rd_t,\\, 0) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RCDaR}_{\\alpha}(\\boldsymbol{x})``: Relative Conditional Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:xret])
  - $(math_dict[:Ct])
  - $(math_dict[:rdt])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is one when no observation weights are set.
  - $(math_dict[:W_T_total])
  - $(math_dict[:nu_ru])

The running peak starts at ``C_0 = 1``, the initial wealth. The minimum has the sorted closed form that [`conditional_drawdown_at_risk`](@ref) states.

# Algorithm

 1. Compute the relative drawdown series with [`relative_drawdown_vec`](@ref).
 2. Read the observation weights with [`get_observation_weights`](@ref).
 3. Reduce the series to its tail average with [`conditional_drawdown_at_risk`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeConditionalDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - $(val_dict[:oow_nonneg])

# Functor

    (r::RelativeConditionalDrawdownatRisk)(x::VecNum)

Computes the Relative CDaR of a portfolio returns vector `x`, and leaves `x` unchanged.

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
                                               alpha::Num_SigCal, w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeConditionalDrawdownatRisk(;
                                           settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                           alpha::Num_SigCal = 0.05,
                                           w::Option{<:ObsWeights} = nothing)::RelativeConditionalDrawdownatRisk
    return RelativeConditionalDrawdownatRisk(settings, alpha, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::RelativeConditionalDrawdownatRisk) = (; alpha = x.alpha)
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
