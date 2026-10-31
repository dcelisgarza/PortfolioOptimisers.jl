"""
    ERM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing) -> Number

Compute the Entropic Value-at-Risk of the returns `x` with an exponential cone programme.

[`EntropicValueatRisk`](@ref) states the measure, its Kullback-Leibler ball and its programme. Every call builds a new `JuMP` model. The model holds the cone rows of the observations of positive weight only. An observation of weight zero does not change the optimum, but the cone row of a large loss can stop the solver far from the optimum.

# Algorithm

 1. Check `w` against `x` with [`checked_observation_weights`](@ref).
 2. Check that `slv` is not empty when it is a vector.
 3. When `alpha` is zero, return the largest loss among the observations of positive weight, with [`worst_positive_weight_loss`](@ref). This step builds no model.
 4. Build and solve the exponential cone programme that [`ERM_model`](@ref) states.
 5. When a solver solves the model, return its objective value. Otherwise, return `NaN`.

# Arguments

  - `x`: Vector of portfolio returns.
  - $(arg_dict[:slv])
  - `alpha`: Significance level, ``\\alpha \\in [0, 1)``.
  - `w`: Observation weights, or `nothing` for equal weights.

# Validation

  - `w` passes [`checked_observation_weights`](@ref) against `x`. Otherwise that function raises its error.
  - If `slv` is a `VecSlv`: `!isempty(slv)`, else `IsEmptyError`.

The function does not check `alpha`. The constructors of the measures check it.

# Returns

  - The Entropic Value-at-Risk of `x`, or `NaN` when no solver of `slv` solves the programme.

# Related

  - [`EntropicValueatRisk`](@ref)
  - [`ERM_model`](@ref)
  - [`ERM_worst_case`](@ref)
  - [`RRM`](@ref)
  - [`worst_positive_weight_loss`](@ref)
  - [`checked_observation_weights`](@ref)
  - [`optimise_JuMP_model!`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
function ERM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
             w::Option{<:ObsWeights} = nothing)
    w = checked_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
    end
    if iszero(alpha)
        return worst_positive_weight_loss(x, w)
    end
    model = ERM_model(x, slv, alpha, w)
    return if isnothing(model)
        NaN
    else
        JuMP.objective_value(model)
    end
end
"""
    ERM_worst_case(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
                   w::Option{<:ObsWeights} = nothing) -> Option{<:VecNum}

Compute the Worst-Case Distribution of the Entropic Value-at-Risk of the returns `x`, from the dual of the programme of [`ERM_model`](@ref).

The Entropic Value-at-Risk is the largest expected loss over a Kullback-Leibler ball, and the Worst-Case Distribution ``\\boldsymbol{q}`` is the distribution in the ball that reaches it. So the expected loss under ``\\boldsymbol{q}`` equals the value of [`ERM`](@ref), to the tolerance of the solver, and ``-\\boldsymbol{q}`` is the gradient of that value with respect to `x`. [`risk_contribution`](@ref) and [`risk_gradient`](@ref) read it through [`worst_case_distribution`](@ref), so that their finite difference does not difference two solves. [`ERM`](@ref) does not compute it, so a call of [`expected_risk`](@ref) does not pay for it.

# Algorithm

 1. Check `w` against `x` with [`checked_observation_weights`](@ref).
 2. Check that `slv` is not empty when it is a vector.
 3. When `alpha` is zero, return the distribution of the largest loss, with [`worst_loss_distribution`](@ref). This step builds no model.
 4. Build and solve the programme with [`ERM_model`](@ref).
 5. When no solver solves the model, or the solver gives no dual, return `nothing`.
 6. Otherwise, negate the first entry of the dual of each row of `erm_cone`, and expand it to `T` entries with [`expand_positive_weight_rows`](@ref).

# Arguments

  - `x`: Vector of portfolio returns.
  - $(arg_dict[:slv])
  - `alpha`: Significance level, ``\\alpha \\in [0, 1)``.
  - `w`: Observation weights, or `nothing` for equal weights.

# Validation

  - `w` passes [`checked_observation_weights`](@ref) against `x`. Otherwise that function raises its error.
  - If `slv` is a `VecSlv`: `!isempty(slv)`, else `IsEmptyError`.

# Returns

  - `q::VecNum`: The Worst-Case Distribution, one entry for each observation. An observation of weight zero has the entry zero.
  - `nothing`: When no solver of `slv` solves the programme, or the solver gives no dual.

# Related

  - [`ERM`](@ref)
  - [`ERM_model`](@ref)
  - [`RRM_worst_case`](@ref)
  - [`worst_case_distribution`](@ref)
"""
function ERM_worst_case(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
                        w::Option{<:ObsWeights} = nothing)
    w = checked_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
    end
    if iszero(alpha)
        return worst_loss_distribution(x, w)
    end
    model = ERM_model(x, slv, alpha, w)
    if isnothing(model) || !JuMP.has_duals(model)
        return nothing
    end
    qr = .-first.(JuMP.dual.(model[:erm_cone]))
    return expand_positive_weight_rows(qr, positive_weight_rows(w, length(x)), length(x))
end
"""
    ERM_model(x::VecNum, slv::Slv_VecSlv, alpha::Number, w::Option{<:VecNum})
        -> Option{JuMP.Model}

Build the exponential cone programme of [`ERM`](@ref) in a new model, and solve it.

[`ERM`](@ref) reads the objective value of the solved model, and [`ERM_worst_case`](@ref) reads the dual of its cone rows. The callers check `w` and `slv` first, and they answer `alpha = 0` without a model.

# Algorithm

 1. Find `rows`, the indices of the observations of positive weight, and `N`, their number. Without `w`, `rows` holds every observation.
 2. Build the programme below in a new model. Its radius term reads `aT`, which is `alpha` times the number of observations without `w`, and `alpha` times the sum of `w` with it. The cone rows read `xr` and `ur`, the entries of `x` and of `u` in `rows`.
 3. Solve the model with the solvers of `slv` in turn.
 4. When a solver solves the model, return the model. Otherwise, return `nothing`.

# JuMP formulation

## Variables

  - `t`: free scalar ``t``.
  - `z`: scalar ``z``, bounded below by zero.
  - `u`: free ``T \\times 1`` vector ``\\boldsymbol{u}``. The entry of an observation of weight zero has a zero coefficient in the budget row, and no cone row reads it.

## Expressions

  - `risk`: ``t - z \\ln(\\alpha T)`` without observation weights, and ``t - z \\ln(\\alpha W_{T})`` with them.

## Constraints

No row carries a string name. The model registers these rows:

  - ``\\sum_{i=1}^{T} u_i - z \\leq 0`` without observation weights, and ``\\sum_{i=1}^{T} w_i u_i - z \\leq 0`` with them.
  - `erm_cone`: ``(-x_i - t,\\, z,\\, u_i) \\in \\mathcal{K}_{\\exp}``, one row for each ``i \\in \\mathcal{I}``.

The cone rows give ``u_i \\geq z\\, e^{(L_i - t)/z}``. So at the optimum, `risk` is the minimum over ``z`` of ``z \\ln\\!\\left(\\tfrac{1}{\\alpha} \\sum_{i \\in \\mathcal{I}} p_i e^{L_i / z}\\right)``, the Chernoff bound of [`EntropicValueatRisk`](@ref). An observation of weight zero has ``p_i = 0``, so it adds no term to that sum.

The first entry of the dual of the cone row of observation ``i`` is ``-q_i``, where ``\\boldsymbol{q}`` is the Worst-Case Distribution of the Kullback-Leibler ball. [`ERM_worst_case`](@ref) reads it.

## Objective

  - The model minimises `risk`.

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - $(math_dict[:alpha_rm])
  - ``L_i = -x_i``: Loss at observation ``i``.
  - $(math_dict[:I_pos_obs])
  - ``w_i``: Observation weight of observation ``i``.
  - $(math_dict[:W_T_total])
  - $(math_dict[:p_i_obs])
  - ``t``, ``z``: Scalar variables of the exponential cone programme.
  - ``u_i``: Variable of observation ``i`` in the exponential cone programme.
  - $(math_dict[:K_exp])

# Arguments

  - `x`: Vector of portfolio returns.
  - $(arg_dict[:slv])
  - `alpha`: Significance level, ``\\alpha \\in (0, 1)``.
  - `w`: Checked observation weights, or `nothing` for equal weights.

# Returns

  - `model::JuMP.Model`: The solved model.
  - `nothing`: When no solver of `slv` solves the programme.

# Related

  - [`ERM`](@ref)
  - [`ERM_worst_case`](@ref)
  - [`positive_weight_rows`](@ref)
  - [`optimise_JuMP_model!`](@ref)
"""
function ERM_model(x::VecNum, slv::Slv_VecSlv, alpha::Number, w::Option{<:VecNum})
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    T = length(x)
    rows = positive_weight_rows(w, T)
    N = length(rows)
    xr = view(x, rows)
    JuMP.@variables(model, begin
                        t
                        z >= 0
                        u[1:T]
                    end)
    aT = if isnothing(w)
        JuMP.@constraint(model, sum(u) - z <= 0)
        alpha * T
    else
        JuMP.@constraint(model, LinearAlgebra.dot(w, u) - z <= 0)
        alpha * sum(w)
    end
    ur = view(u, rows)
    JuMP.@constraint(model, erm_cone[k = 1:N],
                     [-xr[k] - t, z, ur[k]] in JuMP.MOI.ExponentialCone())
    JuMP.@expression(model, risk, t - z * log(aT))
    JuMP.@objective(model, Min, risk)
    return if optimise_JuMP_model!(model, slv).success
        model
    else
        nothing
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Entropic Value-at-Risk (EVaR), the tightest upper bound on the Value-at-Risk that the Chernoff inequality gives.

The EVaR is a coherent risk measure, and it is an upper bound on the Conditional Value-at-Risk. [`ERM`](@ref) computes it with an exponential cone programme. It is also the divergence Ambiguity Set of the library, read as a risk measure: the worst expected loss over a Kullback-Leibler ball about the sample distribution, at radius ``-\\ln(\\alpha)``.

# Mathematical definition

The EVaR is the tightest exponential upper bound on the Value-at-Risk:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) &= \\inf_{z > 0} \\left\\{ z \\ln\\!\\left( \\frac{M_{L}(1/z)}{\\alpha} \\right) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x})``: Entropic Value-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:amb_L_t])
  - ``M_L(u) = \\mathbb{E}[e^{uL}]``: Moment-generating function of the loss.
  - ``z``: Exponential tilt parameter.

For a sample, the infimum is the minimum of an exponential cone programme:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{t,\\, z,\\, \\boldsymbol{u}}{\\min} \\left\\{ t - z \\ln(\\alpha T) \\;:\\; z \\geq 0,\\; \\sum_{i=1}^{T} u_i \\leq z,\\; (-x_i - t,\\, z,\\, u_i) \\in \\mathcal{K}_{\\exp}\\; \\forall i \\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:T])
  - ``t``, ``z``: Scalar variables of the exponential cone programme.
  - ``u_i``: Variable of observation ``i`` in the exponential cone programme.
  - $(math_dict[:K_exp])

With observation weights ``\\boldsymbol{w}``, the radius term reads ``\\alpha W_{T}`` in place of ``\\alpha T``, and the budget row reads ``\\boldsymbol{w}^\\intercal \\boldsymbol{u} \\leq z``. The natural logarithm takes the total weight out of the sum, so the measure is that of the probabilities ``w_t / W_{T}``.

Where:

  - $(math_dict[:W_T_total])

The dual of that programme is the worst expected loss over a Kullback-Leibler ball about the sample distribution:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{Q \\in \\mathcal{Q}_{\\mathrm{KL}}(\\alpha)}{\\sup} \\mathbb{E}_{Q}[L]\\,, \\\\
\\mathcal{Q}_{\\mathrm{KL}}(\\alpha) &= \\left\\{ Q : D_{\\mathrm{KL}}(Q \\,\\|\\, P) \\leq -\\ln(\\alpha) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{Q}_{\\mathrm{KL}}(\\alpha)``: Kullback-Leibler ambiguity ball of radius ``-\\ln(\\alpha)``.
  - ``D_{\\mathrm{KL}}(Q \\,\\|\\, P) = \\sum_{t=1}^{T} q_t \\ln\\!\\left(\\frac{q_t}{p_t}\\right)``: Kullback-Leibler divergence.
  - $(math_dict[:amb_Q])
  - $(math_dict[:amb_P])
  - $(math_dict[:amb_EQ_L])

So the significance level is the Ambiguity Radius, through ``-\\ln(\\alpha)``, and a smaller ``\\alpha`` widens the ball. The ball is a reading of this measure and not an object, so no estimator constructs one.

The definition has these consequences:

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) &\\leq \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\max_{t} L_t\\,, \\\\
\\lim_{\\alpha \\to 1} \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) &= \\mathbb{E}_{P}[L]\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}_{\\alpha}``, ``\\mathrm{CVaR}_{\\alpha}``: Value-at-Risk and Conditional Value-at-Risk at the same level ``\\alpha``.
  - ``\\mathbb{E}_{P}[L]``: Expected loss under the sample distribution.

The EVaR falls as ``\\alpha`` rises. It does not reach the expected loss, because the constructor refuses ``\\alpha = 1``. At ``\\alpha = 0`` the EVaR is the largest loss among the observations of positive weight. With equal observation weights and ``\\alpha T \\leq 1``, the ball holds every distribution on the sample, so the EVaR is the largest loss too.

[evar](@cite) defines the EVaR. Section 7.2.2.6 of [cajas2025](@cite) states the definition, Equation 7.61, and the order above, Equation 7.62. It also writes the sample form as an exponential cone programme, Equations 7.63 and 7.64, and names an earlier work of its author as the source of that programme. No copy of that work was found. The weighted programme is the library's extension.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropicValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> EntropicValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha_tail0])
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::EntropicValueatRisk)(x::VecNum)

Computes the EVaR of a portfolio returns vector `x` with [`ERM`](@ref), and returns `NaN` when no solver of `slv` solves the programme. The functor needs a solver: with `slv = nothing` the call raises a `MethodError`. [`factory`](@ref) fills `slv` from the solver of the optimiser.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> EntropicValueatRisk()
EntropicValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`EntropicValueatRiskRange`](@ref)
  - [`EntropicDrawdownatRisk`](@ref)
  - [`ERM`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:evar])
  - $(ref_dict[:cajas2025]) Section 7.2.2.6, Equations 7.61 to 7.64. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct EntropicValueatRisk <: RiskMeasure
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
    $(field_dict[:oow])
    """
    @pprop w
    function EntropicValueatRisk(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                                 alpha::Num_SigCal, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_half_open_unit_interval(alpha, :alpha)
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(w)}(settings, slv,
                                                                            alpha, w)
    end
end
function EntropicValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             slv::Option{<:Slv_VecSlv} = nothing, alpha::Num_SigCal = 0.05,
                             w::Option{<:ObsWeights} = nothing)::EntropicValueatRisk
    return EntropicValueatRisk(settings, slv, alpha, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::EntropicValueatRisk) = (; alpha = x.alpha)
function (r::EntropicValueatRisk)(x::VecNum)
    return ERM(x, r.slv, r.alpha, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the spread of the returns as the sum of the Entropic Value-at-Risk of the losses and that of the gains.

The loss tail has the level `alpha`, and the gain tail has the level `beta`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{EVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) + \\mathrm{EVaR}_{\\beta}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{EVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: EVaR range.
  - $(math_dict[:xret])
  - ``\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x})``: Entropic Value-at-Risk of the losses at level ``\\alpha``.
  - ``\\mathrm{EVaR}_{\\beta}(-\\boldsymbol{x})``: Entropic Value-at-Risk of the gains at level ``\\beta``.

$(math_dict[:negated_upper_tail])

Each term is the worst expected loss over its own Kullback-Leibler ball about the sample distribution, at radius ``-\\ln(\\alpha)`` on the lower tail and ``-\\ln(\\beta)`` on the upper tail. [`EntropicValueatRisk`](@ref) states the ball.

Section 7.2.3.4 of [cajas2025](@cite) defines this range, and its Section 7.2.3 names [owa1](@cite) as the source of the range measures. No copy of that work was found. Equation 7.81 of [cajas2025](@cite) prints a difference of the two tails, but its bound, Equation 7.82, and its programme, Equation 7.83, state the sum above.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropicValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        beta::Num_SigCal = alpha,
        w::Option{<:ObsWeights} = nothing
    ) -> EntropicValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`. If `beta` is a number: `0 < beta < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::EntropicValueatRiskRange)(x::VecNum)

Computes the EVaR Range of a portfolio returns vector `x` with two calls of [`ERM`](@ref). The functor needs a solver, as the functor of [`EntropicValueatRisk`](@ref) does.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> EntropicValueatRiskRange(; beta = 0.1)
EntropicValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
      beta ┼ Float64: 0.1
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`RelativisticValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`ERM`](@ref)

# References

  - $(ref_dict[:evar])
  - $(ref_dict[:cajas2025]) Section 7.2.3.4, Equations 7.81 to 7.83. $(ref_dict[:no_original_source])
  - $(ref_dict[:owa1])
"""
@propagatable @concrete struct EntropicValueatRiskRange <: RiskMeasure
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
    $(field_dict[:oow])
    """
    @pprop w
    function EntropicValueatRiskRange(settings::RiskMeasureSettings,
                                      slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                      beta::Num_SigCal, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                                          slv,
                                                                                          alpha,
                                                                                          beta,
                                                                                          w)
    end
end
function EntropicValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  slv::Option{<:Slv_VecSlv} = nothing,
                                  alpha::Num_SigCal = 0.05, beta::Num_SigCal = alpha,
                                  w::Option{<:ObsWeights} = nothing)::EntropicValueatRiskRange
    return EntropicValueatRiskRange(settings, slv, alpha, beta, w)
end
# Calibration slots — see `calibration_slots`. One slot per tail, each with its own role.
calibration_slots(x::EntropicValueatRiskRange) = (; alpha = x.alpha, beta = x.beta)
# Tail decomposition — see `range_tails`. The functor below is the value-level twin: it is
# the same two tails, evaluated instead of built.
function range_tails(r::EntropicValueatRiskRange)
    settings = RiskMeasureSettings(; rke = false)
    return (;
            loss = EntropicValueatRisk(; settings = settings, slv = r.slv, alpha = r.alpha,
                                       w = r.w),
            gain = EntropicValueatRisk(; settings = settings, slv = r.slv, alpha = r.beta,
                                       w = r.w))
end
function (r::EntropicValueatRiskRange)(x::VecNum)
    return ERM(x, r.slv, r.alpha, r.w) + ERM(-x, r.slv, r.beta, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Entropic Drawdown-at-Risk (EDaR), the Entropic Value-at-Risk of the absolute drawdown series of the portfolio.

The EDaR is an upper bound on the Drawdown-at-Risk and on the Conditional Drawdown-at-Risk.

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

The EDaR is the EVaR of the drawdown series:

```math
\\begin{align}
\\mathrm{EDaR}_{\\alpha}(\\boldsymbol{x}) &= \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{d}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{EDaR}_{\\alpha}(\\boldsymbol{x})``: Entropic Drawdown-at-Risk.
  - $(math_dict[:alpha_rm])
  - ``\\boldsymbol{d}(\\boldsymbol{x})``: Absolute drawdown series vector ``T \\times 1``.

So the EDaR is the worst expected drawdown over a Kullback-Leibler ball about the sample distribution of ``\\boldsymbol{d}(\\boldsymbol{x})``, at radius ``-\\ln(\\alpha)``. [`EntropicValueatRisk`](@ref) states the ball. The order of that measure holds for the drawdowns too:

```math
\\begin{align}
\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{EDaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{MDD}(\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DaR}_{\\alpha}``, ``\\mathrm{CDaR}_{\\alpha}``: Drawdown-at-Risk and Conditional Drawdown-at-Risk at the same level ``\\alpha``.
  - ``\\mathrm{MDD}(\\boldsymbol{x}) = -\\min_{t} d_t``: Maximum drawdown.

Equation 3.2 of [cdar](@cite) defines the drawdown, with the opposite sign. Section 7.2.4.5 of [cajas2025](@cite) defines the EDaR, Equation 7.95, and states the order above, Equation 7.96. It names an earlier work of its author as the source of the EDaR. No copy of that work was found.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropicDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> EntropicDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha_tail0])
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::EntropicDrawdownatRisk)(x::VecNum)

Computes the EDaR of a portfolio returns vector `x` with [`ERM`](@ref) on its drawdown series. The functor needs a solver, as the functor of [`EntropicValueatRisk`](@ref) does.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> EntropicDrawdownatRisk()
EntropicDrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`RelativisticDrawdownatRisk`](@ref)
  - [`RelativeEntropicDrawdownatRisk`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`ERM`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3, Equation 3.2.
  - $(ref_dict[:evar])
  - $(ref_dict[:cajas2025]) Section 7.2.4.5, Equations 7.95 and 7.96. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct EntropicDrawdownatRisk <: RiskMeasure
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
    $(field_dict[:oow])
    """
    @pprop w
    function EntropicDrawdownatRisk(settings::RiskMeasureSettings,
                                    slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                    w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_half_open_unit_interval(alpha, :alpha)
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(w)}(settings, slv,
                                                                            alpha, w)
    end
end
function EntropicDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                slv::Option{<:Slv_VecSlv} = nothing,
                                alpha::Num_SigCal = 0.05,
                                w::Option{<:ObsWeights} = nothing)::EntropicDrawdownatRisk
    return EntropicDrawdownatRisk(settings, slv, alpha, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::EntropicDrawdownatRisk) = (; alpha = x.alpha)
function (r::EntropicDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return ERM(dd, r.slv, r.alpha, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Relative Entropic Drawdown-at-Risk, the Entropic Value-at-Risk of the compounded drawdown series, for hierarchical optimisation.

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

The Relative EDaR is the EVaR of the relative drawdown series:

```math
\\begin{align}
\\mathrm{REDaR}_{\\alpha}(\\boldsymbol{x}) &= \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{rd}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{REDaR}_{\\alpha}(\\boldsymbol{x})``: Relative Entropic Drawdown-at-Risk.
  - $(math_dict[:alpha_rm])
  - ``\\boldsymbol{rd}(\\boldsymbol{x})``: Relative drawdown series vector ``T \\times 1``.

So the Relative EDaR is the worst expected relative drawdown over a Kullback-Leibler ball about the sample distribution of ``\\boldsymbol{rd}(\\boldsymbol{x})``, at radius ``-\\ln(\\alpha)``. [`EntropicValueatRisk`](@ref) states the ball.

Section 3 of [cdar](@cite) defines the drawdown of the uncompounded cumulative return only, and Equation 7.95 of [cajas2025](@cite) defines the EDaR on that series. Neither work defines a measure on the relative drawdown, so this measure is the library's extension of them.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeEntropicDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeEntropicDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha_tail0])
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - $(val_dict[:oow_rm])

# Functor

    (r::RelativeEntropicDrawdownatRisk)(x::VecNum)

Computes the Relative EDaR of a portfolio returns vector `x` with [`ERM`](@ref) on its relative drawdown series. The functor needs a solver, as the functor of [`EntropicValueatRisk`](@ref) does.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeEntropicDrawdownatRisk()
RelativeEntropicDrawdownatRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
       slv ┼ nothing
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`EntropicDrawdownatRisk`](@ref)
  - [`RelativeRelativisticDrawdownatRisk`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`ERM`](@ref)

# References

  - $(ref_dict[:cdar]) Section 3, Equation 3.2.
  - $(ref_dict[:evar])
  - $(ref_dict[:cajas2025]) Section 7.2.4.5, Equation 7.95. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct RelativeEntropicDrawdownatRisk <: HierarchicalRiskMeasure
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
    $(field_dict[:oow])
    """
    @pprop w
    function RelativeEntropicDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                            slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                            w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_half_open_unit_interval(alpha, :alpha)
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(w)}(settings, slv,
                                                                            alpha, w)
    end
end
function RelativeEntropicDrawdownatRisk(;
                                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                        slv::Option{<:Slv_VecSlv} = nothing,
                                        alpha::Num_SigCal = 0.05,
                                        w::Option{<:ObsWeights} = nothing)::RelativeEntropicDrawdownatRisk
    return RelativeEntropicDrawdownatRisk(settings, slv, alpha, w)
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::RelativeEntropicDrawdownatRisk) = (; alpha = x.alpha)
function (r::RelativeEntropicDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return ERM(dd, r.slv, r.alpha, r.w)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::EntropicValueatRisk) = NetReturnsInput()
risk_input_kind(::EntropicValueatRiskRange) = NetReturnsInput()
risk_input_kind(::EntropicDrawdownatRisk) = NetReturnsInput()
risk_input_kind(::RelativeEntropicDrawdownatRisk) = NetReturnsInput()
# Worst-Case Distribution — see `worst_case_distribution` and `worst_case_risk`.
function worst_case_distribution(r::EntropicValueatRisk, x::VecNum)
    return ERM_worst_case(x, r.slv, r.alpha, r.w)
end
function worst_case_distribution(r::EntropicValueatRiskRange, x::VecNum)
    return worst_case_pair(ERM_worst_case(x, r.slv, r.alpha, r.w),
                           ERM_worst_case(-x, r.slv, r.beta, r.w))
end
function worst_case_distribution(r::EntropicDrawdownatRisk, x::VecNum)
    return ERM_worst_case(absolute_drawdown_vec(x), r.slv, r.alpha, r.w)
end
function worst_case_distribution(r::RelativeEntropicDrawdownatRisk, x::VecNum)
    return ERM_worst_case(relative_drawdown_vec(x), r.slv, r.alpha, r.w)
end
function worst_case_risk(::EntropicDrawdownatRisk, x::VecNum, q::VecNum)
    return -LinearAlgebra.dot(q, absolute_drawdown_vec(x))
end
function worst_case_risk(::RelativeEntropicDrawdownatRisk, x::VecNum, q::VecNum)
    return -LinearAlgebra.dot(q, relative_drawdown_vec(x))
end

export EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
       RelativeEntropicDrawdownatRisk
