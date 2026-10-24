# Clarabel.jl 4915b83e0d, `warmstart_test/portfolioOpt/higherorderRiskMeansure.jl`, L23
"""
    PRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, p::Number = 2.0,
        w::Option{<:ObsWeights} = nothing) -> Number

Compute the Power Norm Value-at-Risk of the returns `x` with a power cone programme.

[`PowerNormValueatRisk`](@ref) states the measure, its programme and the condition under which it is the largest loss. Under that condition the function returns the largest loss and builds no model. Otherwise every call builds a new `JuMP` model. The model holds the rows of the observations of positive weight only. An observation of weight zero does not change the optimum, but the rows of a large loss can stop the solver away from the optimum.

# Algorithm

 1. Check `w` against `x` with [`checked_observation_weights`](@ref).
 2. Check that `slv` is not empty when it is a vector.
 3. Find `rows`, the indices of the observations of positive weight, and `N`, their number. Without `w`, `rows` holds every observation.
 4. Find `l`, the largest loss among the observations of positive weight, with [`worst_positive_weight_loss`](@ref).
 5. Find `pl`, the probability of the observations at the loss `l`. It is their number divided by the number of observations without `w`, and their share of the sum of `w` with it.
 6. When `alpha` is at most `pl` to the power `1/p`, return `l`. This step builds no model.
 7. Build the programme below in a new model. Its objective reads `iaT`, which is the inverse of `alpha` times the number of observations to the power `1/p` without `w`, and the inverse of `alpha` times the sum of `w` to the power `1/p` with it. The rows read `xr`, the entries of `x` in `rows`.
 8. Solve the model with the solvers of `slv` in turn.
 9. When a solver solves the model, return its objective value. Otherwise, return `NaN`.

# JuMP formulation

## Variables

  - `pvar_eta`: free scalar ``\\eta``.
  - `pvar_t`: free scalar ``t``.
  - `pvar_w`: ``N \\times 1`` vector ``\\boldsymbol{w}``, bounded below by zero, with one entry for each observation in ``\\mathcal{I}``.
  - `pvar_v`: free ``N \\times 1`` vector ``\\boldsymbol{v}``, with one entry for each observation in ``\\mathcal{I}``.

## Constraints

No row carries a name. The model registers these rows:

  - ``\\sum_{i \\in \\mathcal{I}} v_i - t \\leq 0`` without observation weights, and ``\\sum_{i \\in \\mathcal{I}} q_i v_i - t \\leq 0`` with them.
  - ``x_i + w_i + \\eta \\geq 0``, one row for each ``i \\in \\mathcal{I}``.
  - ``(v_i,\\, t,\\, w_i) \\in \\mathcal{K}_{\\mathrm{pow}}(1/p)``, one row for each ``i \\in \\mathcal{I}``.

The cone rows give ``v_i \\geq w_i^{p} / t^{p-1}``, so the first row gives ``t \\geq \\left(\\sum_{i \\in \\mathcal{I}} w_i^{p}\\right)^{1/p}`` without observation weights. At the optimum ``w_i = \\max(L_i - \\eta, 0)``, so the objective is the minimum of [`PowerNormValueatRisk`](@ref) over ``\\eta``. An observation of weight zero has ``p_i = 0``, so it adds no term to that minimum.

## Objective

  - The model minimises ``\\eta + t / (\\alpha T^{1/p})`` without observation weights, and ``\\eta + t / (\\alpha W_{T}^{1/p})`` with them.

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:p_pnvar_order])
  - ``L_i = -x_i``: Loss at observation ``i``.
  - $(math_dict[:I_pos_obs])
  - ``q_i``: Observation weight of observation ``i``.
  - ``W_{T} = \\sum_{t=1}^{T} q_t``: Total observation weight.
  - $(math_dict[:p_i_obs])
  - $(math_dict[:pnvar_eta])
  - $(math_dict[:pnvar_t])
  - $(math_dict[:pnvar_w])
  - $(math_dict[:pnvar_v])
  - $(math_dict[:K_pow])

# Arguments

  - `x`: Vector of portfolio returns.
  - $(arg_dict[:slv])
  - `alpha`: Significance level, ``\\alpha \\in (0, 1)``.
  - `p`: Order of the power norm, ``p > 1``.
  - `w`: Observation weights, or `nothing` for equal weights.

# Validation

  - `w` passes [`checked_observation_weights`](@ref) against `x`. Otherwise that function raises its error.
  - If `slv` is a `VecSlv`: `!isempty(slv)`, else `IsEmptyError`.

The function does not check `alpha` or `p`. The constructors of the measures check them.

# Returns

  - The Power Norm Value-at-Risk of `x`, or `NaN` when step 6 does not return and no solver of `slv` solves the programme.

# Related

  - [`PowerNormValueatRisk`](@ref)
  - [`ERM`](@ref)
  - [`RRM`](@ref)
  - [`worst_positive_weight_loss`](@ref)
  - [`positive_weight_rows`](@ref)
  - [`checked_observation_weights`](@ref)
  - [`optimise_JuMP_model!`](@ref)
  - [`Slv_VecSlv`](@ref)

# References

  - $(ref_dict[:pnvar]) Equation 32.
  - $(ref_dict[:chares2009]) Section 4.1, Equations 4.4 and 4.5.
"""
function PRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, p::Number = 2.0,
             w::Option{<:ObsWeights} = nothing)
    w = checked_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
    end
    T = length(x)
    rows = positive_weight_rows(w, T)
    N = length(rows)
    xr = view(x, rows)
    ip = inv(p)
    l = worst_positive_weight_loss(x, w)
    pl = if isnothing(w)
        count(v -> -v == l, x) / T
    else
        sum(w[t] for t in rows if -x[t] == l) / sum(w)
    end
    if alpha <= pl^ip
        return l
    end
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    JuMP.@variables(model, begin
                        pvar_eta
                        pvar_t
                        pvar_w[1:N] >= 0
                        pvar_v[1:N]
                    end)
    iaT = if isnothing(w)
        JuMP.@constraint(model, sum(pvar_v) - pvar_t <= 0)
        inv(alpha * T^ip)
    else
        JuMP.@constraint(model, LinearAlgebra.dot(view(w, rows), pvar_v) - pvar_t <= 0)
        inv(alpha * sum(w)^ip)
    end
    JuMP.@constraints(model,
                      begin
                          (xr + pvar_w) .+ pvar_eta >= 0
                          [k = 1:N],
                          [pvar_v[k], pvar_t, pvar_w[k]] in JuMP.MOI.PowerCone(ip)
                      end)
    JuMP.@objective(model, Min, pvar_eta + iaT * pvar_t)
    return if optimise_JuMP_model!(model, slv).success
        JuMP.objective_value(model)
    else
        NaN
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Power Norm Value-at-Risk (PNVaR), a threshold plus the power norm of the losses above it, divided by the significance level.

[pnvar](@cite) defines the measure as the higher moment coherent risk measure. As ``p`` falls to one it falls to the Conditional Value-at-Risk, and it rises with ``p`` to the largest loss. [`PRM`](@ref) computes it with a power cone programme. The constructor refuses ``p = 1``, because the power cone of that programme is then degenerate and a solver does not solve it. [`ConditionalValueatRisk`](@ref) measures that case. On a short sample it equals [`WorstRealisation`](@ref). At the defaults `alpha = 0.05` and `p = 2`, this holds on every series of at most 400 equally weighted observations.

# Mathematical definition

The PNVaR at level ``\\alpha`` with power ``p`` is:

```math
\\begin{align}
\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x}) &= \\underset{\\eta}{\\min} \\left\\{ \\eta + \\frac{1}{\\alpha} \\left( \\sum_{i=1}^{T} p_i \\max(L_i - \\eta,\\, 0)^{p} \\right)^{1/p} \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x})``: Power Norm Value-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:p_pnvar_order])
  - ``L_i = -x_i``: Loss at observation ``i``.
  - $(math_dict[:p_i_obs])
  - $(math_dict[:pnvar_eta])

Equation 4 of [pnvar](@cite) states this definition with the confidence level ``1 - \\alpha`` in place of ``\\alpha``, for ``p \\geq 1``. For a sample, the minimum is that of a power cone programme:

```math
\\begin{align}
\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x}) &= \\underset{\\eta,\\, t,\\, \\boldsymbol{w},\\, \\boldsymbol{v}}{\\min} \\left\\{ \\eta + \\frac{t}{\\alpha T^{1/p}} \\;:\\; \\boldsymbol{w} \\geq \\boldsymbol{0},\\; \\sum_{i=1}^{T} v_i \\leq t,\\; (x_i + w_i) + \\eta \\geq 0,\\; (v_i, t, w_i) \\in \\mathcal{K}_{\\mathrm{pow}}(1/p)\\; \\forall i \\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:pnvar_t])
  - $(math_dict[:pnvar_w])
  - $(math_dict[:pnvar_v])
  - $(math_dict[:K_pow])

Equation 32 of [pnvar](@cite) states this programme for equally weighted observations. The cone rows give ``v_i \\geq w_i^{p} / t^{p-1}``, so the budget row gives ``t \\geq \\|\\boldsymbol{w}\\|_{p}``. Equations 4.4 and 4.5 of [chares2009](@cite) state that encoding of the power norm.

The conic variable ``\\boldsymbol{w}`` is unrelated to the `w` field, which carries the observation weights. Write those weights ``\\boldsymbol{q}``. When they are present, the normalisation ``T^{1/p}`` becomes ``\\left(\\sum_{t=1}^{T} q_t\\right)^{1/p}`` and the budget constraint becomes ``\\sum_{i=1}^{T} q_i v_i \\leq t``. That weighted programme is the library's extension.

The definition has these consequences:

```math
\\begin{align}
\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) = \\lim_{p \\to 1} \\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x}) \\leq \\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x}) &\\leq \\mathrm{PNVaR}_{\\alpha,p'}(\\boldsymbol{x}) \\leq \\max_{i} L_i\\,, \\quad 1 < p \\leq p'\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CVaR}_{\\alpha}``: Conditional Value-at-Risk at the same level ``\\alpha``.

Section 3 of [pnvar](@cite) states that ``p = 1`` gives the Conditional Value-at-Risk, and its Equation 5 states the order in ``p``. The objective is continuous in ``p``, so the measure tends to the Conditional Value-at-Risk as ``p`` tends to one. No order holds between the PNVaR and [`EntropicValueatRisk`](@ref). Either of the two can be the larger on a sample.

Let ``\\ell = \\max_i L_i`` be the largest loss, and let ``\\pi_\\ell`` be the probability of the observations at ``\\ell``. For ``m`` equally weighted observations at ``\\ell``, ``\\pi_\\ell = m / T``. With observation weights, ``\\pi_\\ell`` is the share of ``\\sum_{t=1}^{T} q_t`` that those observations carry. The objective is convex in ``\\eta``, and its slope just below ``\\eta = \\ell`` is ``1 - \\pi_\\ell^{1/p} / \\alpha``. So the minimum is at ``\\eta = \\ell`` exactly when that slope is not positive:

```math
\\begin{align}
\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x}) = \\ell &\\iff \\alpha\\, \\pi_\\ell^{-1/p} \\leq 1 \\iff p \\geq \\frac{\\ln \\pi_\\ell}{\\ln \\alpha}\\,.
\\end{align}
```

So the measure is the largest loss at every ``p`` above a finite value. With equal weights and no tie, the condition is ``\\alpha T^{1/p} \\leq 1``, that is ``T \\leq \\alpha^{-p}``. At ``\\alpha = 0.05`` and ``p = 2``, this is ``T \\leq 400``. A portfolio that minimises the measure can tie ``m`` of its losses at ``\\ell``. This raises ``\\pi_\\ell`` to ``m / T``, so at such a portfolio the measure can equal the largest loss for ``T`` up to ``m\\, \\alpha^{-p}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerNormValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        p::Number = 2.0,
        w::Option{<:ObsWeights} = nothing
    ) -> PowerNormValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - `p > 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::PowerNormValueatRisk)(x::VecNum)

Computes the PNVaR of a portfolio returns vector `x` with [`PRM`](@ref). The value equals the largest loss of `x` under the condition in the mathematical definition, and [`PRM`](@ref) then returns it without a solver. Otherwise it returns `NaN` when no solver of `slv` solves the programme. The functor needs a solver: with `slv = nothing` the call raises a `MethodError`. [`factory`](@ref) fills `slv` from the solver of the optimiser.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> PowerNormValueatRisk()
PowerNormValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
         p ┼ Float64: 2.0
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalValueatRisk`](@ref): the value of the measure at ``p = 1``.
  - [`EntropicValueatRisk`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`PowerNormValueatRiskRange`](@ref)
  - [`PowerNormDrawdownatRisk`](@ref)
  - [`WorstRealisation`](@ref): the value of the measure when the largest loss carries a probability of at least ``\\alpha^{p}``.
  - [`PRM`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:pnvar]) Section 3, Equations 4, 5 and 32.
  - $(ref_dict[:chares2009]) Section 4.1, Equations 4.4 and 4.5.
"""
@propagatable @concrete struct PowerNormValueatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    @cprop slv
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:p_pnvar])
    """
    p
    """
    $(field_dict[:oow])
    """
    @pprop w
    function PowerNormValueatRisk(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                                  alpha::Num_SigCal, p::Number,
                                  w::Option{<:StatsBase.AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        @argcheck(p > one(p), DomainError(p, "p must be > 1"))
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function PowerNormValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              slv::Option{<:Slv_VecSlv} = nothing, alpha::Num_SigCal = 0.05,
                              p::Number = 2.0,
                              w::Option{<:ObsWeights} = nothing)::PowerNormValueatRisk
    return PowerNormValueatRisk(settings, slv, alpha, p, w)
end
# Calibration slots — see `calibration_slots`. `p` is the order of the norm, not a quantity
# of the sample, so it is not one of them.
calibration_slots(x::PowerNormValueatRisk) = (; alpha = x.alpha)
function (r::PowerNormValueatRisk)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.p, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the spread of the returns as the sum of the Power Norm Value-at-Risk of the losses and that of the gains.

The loss tail has the level `alpha` and the order `pa`, and the gain tail has the level `beta` and the order `pb`. Each tail equals its largest loss under the condition that [`PowerNormValueatRisk`](@ref) states.

# Mathematical definition

```math
\\begin{align}
\\mathrm{PNVaRRange}_{\\alpha,p_a,\\beta,p_b}(\\boldsymbol{x}) &= \\mathrm{PNVaR}_{\\alpha,p_a}(\\boldsymbol{x}) + \\mathrm{PNVaR}_{\\beta,p_b}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PNVaRRange}_{\\alpha,p_a,\\beta,p_b}(\\boldsymbol{x})``: Power Norm VaR range.
  - $(math_dict[:xret])
  - ``\\mathrm{PNVaR}_{\\alpha,p_a}(\\boldsymbol{x})``: Power Norm Value-at-Risk of the losses at level ``\\alpha`` and order ``p_a``.
  - ``\\mathrm{PNVaR}_{\\beta,p_b}(-\\boldsymbol{x})``: Power Norm Value-at-Risk of the gains at level ``\\beta`` and order ``p_b``.

$(math_dict[:negated_upper_tail])

[pnvar](@cite) defines one tail. The range of two tails is the library's extension.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerNormValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        beta::Num_SigCal = alpha,
        pa::Number = 2.0,
        pb::Number = pa,
        w::Option{<:ObsWeights} = nothing
    ) -> PowerNormValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`. If `beta` is a number: `0 < beta < 1`.
  - `pa > 1`, `pb > 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::PowerNormValueatRiskRange)(x::VecNum)

Computes the PNVaR Range of a portfolio returns vector `x` with two calls of [`PRM`](@ref). The functor needs a solver, as the functor of [`PowerNormValueatRisk`](@ref) does.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> PowerNormValueatRiskRange()
PowerNormValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
      beta ┼ Float64: 0.05
        pa ┼ Float64: 2.0
        pb ┼ Float64: 2.0
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`PowerNormValueatRisk`](@ref)
  - [`EntropicValueatRiskRange`](@ref)
  - [`GenericValueatRiskRange`](@ref): the same value, with two [`PowerNormValueatRisk`](@ref) tails.
  - [`range_tails`](@ref)
  - [`PRM`](@ref)

# References

  - $(ref_dict[:pnvar]) Equation 4.
"""
@propagatable @concrete struct PowerNormValueatRiskRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    @cprop slv
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:beta])
    """
    beta
    """
    $(field_dict[:pa_rm])
    """
    pa
    """
    $(field_dict[:pb_rm])
    """
    pb
    """
    $(field_dict[:oow])
    """
    @pprop w
    function PowerNormValueatRiskRange(settings::RiskMeasureSettings,
                                       slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                       beta::Num_SigCal, pa::Number, pb::Number,
                                       w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        @argcheck(pa > one(pa), DomainError(pa, "pa must be > 1"))
        @argcheck(pb > one(pb), DomainError(pb, "pb must be > 1"))
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(beta), typeof(pa),
                   typeof(pb), typeof(w)}(settings, slv, alpha, beta, pa, pb, w)
    end
end
function PowerNormValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   slv::Option{<:Slv_VecSlv} = nothing,
                                   alpha::Num_SigCal = 0.05, beta::Num_SigCal = alpha,
                                   pa::Number = 2.0, pb::Number = pa,
                                   w::Option{<:ObsWeights} = nothing)::PowerNormValueatRiskRange
    return PowerNormValueatRiskRange(settings, slv, alpha, beta, pa, pb, w)
end
# Calibration slots — see `calibration_slots`. One slot per tail, each with its own role.
calibration_slots(x::PowerNormValueatRiskRange) = (; alpha = x.alpha, beta = x.beta)
# Tail decomposition — see `range_tails`. Each tail carries its own norm order: `pa` shapes
# the loss side, `pb` the gain side. The functor below is the value-level twin, and it is
# what pins that pairing.
function range_tails(r::PowerNormValueatRiskRange)
    settings = RiskMeasureSettings(; rke = false)
    return (;
            loss = PowerNormValueatRisk(; settings = settings, slv = r.slv, alpha = r.alpha,
                                        p = r.pa, w = r.w),
            gain = PowerNormValueatRisk(; settings = settings, slv = r.slv, alpha = r.beta,
                                        p = r.pb, w = r.w))
end
function (r::PowerNormValueatRiskRange)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.pa, r.w) + PRM(-x, r.slv, r.beta, r.pb, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Power Norm Drawdown-at-Risk (PNDaR), the Power Norm Value-at-Risk of the absolute drawdown series of the portfolio.

It equals the largest drawdown under the condition that [`PowerNormValueatRisk`](@ref) states, with the drawdowns in place of the losses.

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

The Power Norm Drawdown-at-Risk is the PNVaR of the drawdown series:

```math
\\begin{align}
\\mathrm{PNDaR}_{\\alpha,p}(\\boldsymbol{x}) &= \\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{d}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PNDaR}_{\\alpha,p}(\\boldsymbol{x})``: Power Norm Drawdown-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:p_pnvar_order])
  - ``\\boldsymbol{d}(\\boldsymbol{x})``: Absolute drawdown series.

The order of [`PowerNormValueatRisk`](@ref) holds for the drawdowns too:

```math
\\begin{align}
\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{PNDaR}_{\\alpha,p}(\\boldsymbol{x}) \\leq \\mathrm{PNDaR}_{\\alpha,p'}(\\boldsymbol{x}) \\leq \\mathrm{MDD}(\\boldsymbol{x})\\,, \\quad 1 < p \\leq p'\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CDaR}_{\\alpha}``: Conditional Drawdown-at-Risk at the same level ``\\alpha``.
  - ``\\mathrm{MDD}(\\boldsymbol{x}) = -\\min_{t} d_t``: Maximum drawdown.

Equation 3.2 of [cdar](@cite) defines the drawdown, with the opposite sign. Neither that work nor [pnvar](@cite) defines a power norm measure of the drawdowns, so this measure is the library's extension of them.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerNormDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        p::Number = 2.0,
        w::Option{<:ObsWeights} = nothing
    ) -> PowerNormDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - `p > 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::PowerNormDrawdownatRisk)(x::VecNum)

Computes the PNDaR of a portfolio returns vector `x` with [`PRM`](@ref) on its drawdown series. The functor needs a solver, as the functor of [`PowerNormValueatRisk`](@ref) does.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> PowerNormDrawdownatRisk()
PowerNormDrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
         p ┼ Float64: 2.0
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`PowerNormValueatRisk`](@ref)
  - [`RelativisticDrawdownatRisk`](@ref)
  - [`EntropicDrawdownatRisk`](@ref)
  - [`RelativePowerNormDrawdownatRisk`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`PRM`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3, Equation 3.2.
  - $(ref_dict[:pnvar]) Equations 4 and 5.
"""
@propagatable @concrete struct PowerNormDrawdownatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    @cprop slv
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:p_pnvar])
    """
    p
    """
    $(field_dict[:oow])
    """
    @pprop w
    function PowerNormDrawdownatRisk(settings::RiskMeasureSettings,
                                     slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                     p::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        @argcheck(p > one(p), DomainError(p, "p must be > 1"))
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function PowerNormDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 slv::Option{<:Slv_VecSlv} = nothing,
                                 alpha::Num_SigCal = 0.05, p::Number = 2.0,
                                 w::Option{<:ObsWeights} = nothing)::PowerNormDrawdownatRisk
    return PowerNormDrawdownatRisk(settings, slv, alpha, p, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::PowerNormDrawdownatRisk) = (; alpha = x.alpha)
function (r::PowerNormDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return PRM(dd, r.slv, r.alpha, r.p, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Relative Power Norm Drawdown-at-Risk, the Power Norm Value-at-Risk of the compounded drawdown series, for hierarchical optimisation.

It equals the largest relative drawdown under the condition that [`PowerNormValueatRisk`](@ref) states, with the drawdowns in place of the losses.

# Mathematical definition

Define the relative drawdown series:

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

The Relative Power Norm Drawdown-at-Risk is the PNVaR of the relative drawdown series:

```math
\\begin{align}
\\mathrm{RPNDaR}_{\\alpha,p}(\\boldsymbol{x}) &= \\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{rd}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RPNDaR}_{\\alpha,p}(\\boldsymbol{x})``: Relative Power Norm Drawdown-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:p_pnvar_order])
  - ``\\boldsymbol{rd}(\\boldsymbol{x})``: Relative drawdown series.

Section 3 of [cdar](@cite) defines the drawdown of the uncompounded cumulative return only, and [pnvar](@cite) defines the measure on a loss. Neither work defines a measure on the relative drawdown, so this measure is the library's extension of them.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativePowerNormDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        p::Number = 2.0,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativePowerNormDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - `p > 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::RelativePowerNormDrawdownatRisk)(x::VecNum)

Computes the Relative PNDaR of a portfolio returns vector `x` with [`PRM`](@ref) on its relative drawdown series. The functor needs a solver, as the functor of [`PowerNormValueatRisk`](@ref) does.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativePowerNormDrawdownatRisk()
RelativePowerNormDrawdownatRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
       slv ┼ nothing
     alpha ┼ Float64: 0.05
         p ┼ Float64: 2.0
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`PowerNormDrawdownatRisk`](@ref)
  - [`RelativeRelativisticDrawdownatRisk`](@ref)
  - [`RelativeEntropicDrawdownatRisk`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`PRM`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3, Equation 3.2.
  - $(ref_dict[:pnvar]) Equation 4.
"""
@propagatable @concrete struct RelativePowerNormDrawdownatRisk <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    @cprop slv
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:p_pnvar])
    """
    p
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativePowerNormDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                             slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                             p::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        @argcheck(p > one(p), DomainError(p, "p must be > 1"))
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function RelativePowerNormDrawdownatRisk(;
                                         settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                         slv::Option{<:Slv_VecSlv} = nothing,
                                         alpha::Num_SigCal = 0.05, p::Number = 2.0,
                                         w::Option{<:ObsWeights} = nothing)::RelativePowerNormDrawdownatRisk
    return RelativePowerNormDrawdownatRisk(settings, slv, alpha, p, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::RelativePowerNormDrawdownatRisk) = (; alpha = x.alpha)
function (r::RelativePowerNormDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return PRM(dd, r.slv, r.alpha, r.p, r.w)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::PowerNormValueatRisk) = NetReturnsInput()
risk_input_kind(::PowerNormValueatRiskRange) = NetReturnsInput()
risk_input_kind(::PowerNormDrawdownatRisk) = NetReturnsInput()
risk_input_kind(::RelativePowerNormDrawdownatRisk) = NetReturnsInput()

export PowerNormValueatRisk, PowerNormValueatRiskRange, PowerNormDrawdownatRisk,
       RelativePowerNormDrawdownatRisk
