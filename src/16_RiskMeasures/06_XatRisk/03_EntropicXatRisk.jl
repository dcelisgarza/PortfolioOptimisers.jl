"""
    ERM(x, slv, alpha = 0.05, ...; kwargs...)

Compute the Entropic Risk Measure (ERM) for a vector of portfolio returns.

Solves a convex optimisation problem to compute the ERM at confidence level `alpha`, using the specified solver(s). The ERM is a coherent risk measure based on the exponential moment of the loss distribution.

# Arguments

  - `x`: Vector of portfolio returns.
  - `slv`: Solver or vector of solvers.
  - `alpha`: Confidence level (default `0.05`).
  - Additional parameters depending on the specific ERM formulation.
  - `kwargs...`: Additional keyword arguments passed to the solver.

# Returns

  - ERM value (scalar), or `NaN` if no solver in `slv` succeeds.

# Related

  - [`EntropicValueatRisk`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
function ERM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
             w::Option{<:ObsWeights} = nothing)
    w = get_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
    end
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    T = length(x)
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
    JuMP.@constraint(model, [i = 1:T], [-x[i] - t, z, u[i]] in JuMP.MOI.ExponentialCone())
    JuMP.@expression(model, risk, t - z * log(aT))
    JuMP.@objective(model, Min, risk)
    return if optimise_JuMP_model!(model, slv).success
        JuMP.objective_value(model)
    else
        NaN
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Entropic Value-at-Risk (EVaR) risk measure.

`EntropicValueatRisk` is a coherent risk measure based on the Chernoff bound. It is an upper bound for both CVaR and VaR and is computed by solving a conic optimisation problem via an external solver. It is also the divergence Ambiguity Set of the library, read as a risk measure: the worst expected loss over a Kullback-Leibler ball about the sample distribution, at radius ``-\\ln(\\alpha)``.

# Mathematical definition

The EVaR is defined via the Chernoff bound as the tightest exponential upper bound on VaR and CVaR:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) &= \\inf_{z > 0} \\left\\{ z \\ln\\!\\left( \\frac{M_{L}(1/z)}{\\alpha} \\right) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x})``: Entropic Value-at-Risk (tightest exponential upper bound on VaR and CVaR).
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:amb_L_t])
  - ``M_L(u) = \\mathbb{E}[e^{uL}]``: Moment-generating function of the loss.
  - ``z``: Exponential tilt parameter.

Computationally, it is solved via the conic programme:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) &= \\underset{t,\\, z,\\, \\boldsymbol{u}}{\\min} \\left\\{ t - z \\ln(\\alpha T) \\;:\\; z \\geq 0,\\; \\sum_{i=1}^{T} u_i \\leq z,\\; (-x_i - t,\\, z,\\, u_i) \\in K_{\\exp}\\; \\forall i \\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:T])
  - ``t``, ``z``, ``\\boldsymbol{u}``: Conic optimisation variables.
  - ``K_{\\exp} = \\{(a, b, c) : b\\, e^{a/b} \\leq c,\\, b > 0\\}``: Exponential cone.

For observation-weighted samples with weight vector ``\\boldsymbol{w}``, the normalisation ``\\alpha T`` becomes ``\\alpha \\sum_{t=1}^{T} w_t`` and the budget constraint becomes ``\\boldsymbol{w}^\\intercal \\boldsymbol{u} \\leq z``.

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

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::EntropicValueatRisk)(x::VecNum)

Computes the EVaR of a portfolio returns vector `x`.

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

# References

  - $(ref_dict[:evar])
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
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
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

Represents the Entropic Value-at-Risk Range (EVaR Range) risk measure.

`EntropicValueatRiskRange` computes the sum of the lower-tail EVaR (at level `alpha`) and the upper-tail EVaR (at level `beta`).

# Mathematical definition

```math
\\begin{align}
\\mathrm{EVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) + \\mathrm{EVaR}_{\\beta}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{EVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: EVaR range (entropic tail spread).
  - $(math_dict[:xret])
  - ``\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x})``: Lower-tail entropic risk at level ``\\alpha``.
  - ``\\mathrm{EVaR}_{\\beta}(-\\boldsymbol{x})``: Upper-tail entropic risk at level ``\\beta``.

$(math_dict[:negated_upper_tail])

Each term is the worst expected loss over its own Kullback-Leibler ball about the sample distribution, at radius ``-\\ln(\\alpha)`` on the lower tail and ``-\\ln(\\beta)`` on the upper tail. [`EntropicValueatRisk`](@ref) states the ball.

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
  - If `w` is not `nothing`: `!isempty(w)`.

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`EntropicValueatRisk`](@ref)

# References

  - $(ref_dict[:evar])
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
        assert_nonempty_nonneg_finite_val(w, :w)
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

Represents the Entropic Drawdown-at-Risk (EDaR) risk measure.

`EntropicDrawdownatRisk` applies the Entropic Value-at-Risk framework to the absolute drawdown series of portfolio returns. It is a coherent risk measure providing an upper bound on both the Drawdown-at-Risk and Conditional Drawdown-at-Risk.

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

So the EDaR is the worst expected drawdown over a Kullback-Leibler ball about the sample distribution of ``\\boldsymbol{d}(\\boldsymbol{x})``, at radius ``-\\ln(\\alpha)``. [`EntropicValueatRisk`](@ref) states the ball.

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

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::EntropicDrawdownatRisk)(x::VecNum)

Computes the EDaR of a portfolio returns vector `x`.

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

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:evar])
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
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
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

Represents the Relative Entropic Drawdown-at-Risk (Relative EDaR) risk measure for hierarchical optimisation.

`RelativeEntropicDrawdownatRisk` applies the Entropic Value-at-Risk framework to the relative (compounded) drawdown series of portfolio returns.

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

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativeEntropicDrawdownatRisk)(x::VecNum)

Computes the Relative EDaR of a portfolio returns vector `x`.

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

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:evar])
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
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
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

export EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
       RelativeEntropicDrawdownatRisk
