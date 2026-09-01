"""
    RRM(x, slv, alpha = 0.05, kappa = 0.3, ...; kwargs...)

Compute the Relativistic Risk Measure (RRM) for a vector of portfolio returns.

Solves a convex optimisation problem to compute the RRM at confidence level `alpha` with relativistic parameter `kappa`, using the specified solver(s).

The primal power-cone programme is tried first. If no solver in `slv` succeeds on it, the equivalent dual programme is tried, which is numerically better conditioned for some solvers. If neither succeeds, the result is `NaN`.

# Arguments

  - `x`: Vector of portfolio returns.
  - `slv`: Solver or vector of solvers.
  - `alpha`: Confidence level (default `0.05`).
  - `kappa`: Relativistic parameter (default `0.3`).
  - Additional parameters depending on the specific RRM formulation.
  - `kwargs...`: Additional keyword arguments passed to the solver.

# Returns

  - RRM value (scalar), or `NaN` if neither the primal nor the dual programme is solved.

# Related

  - [`RelativisticValueatRisk`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
function RRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, kappa::Number = 0.3,
             w::Option{<:ObsWeights} = nothing)
    w = get_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
    end
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    ik2 = inv(2 * kappa)
    T = length(x)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    JuMP.@variables(model, begin
                        t
                        z >= 0
                        omega[1:T]
                        psi[1:T]
                        theta[1:T]
                        epsilon[1:T]
                    end)
    ln_k = kappa_log(inv(alpha * T), kappa)
    if isnothing(w)
        JuMP.@expression(model, risk, t + ln_k * z + sum(psi + theta))
    else
        wi = w / sum(w)
        JuMP.@expression(model, risk, t + ln_k * z + T * LinearAlgebra.dot(wi, psi + theta))
    end
    JuMP.@constraints(model,
                      begin
                          [i = 1:T],
                          [z * opk * ik2, psi[i] * opk * ik, epsilon[i]] in
                          JuMP.MOI.PowerCone(iopk)
                          [i = 1:T],
                          [omega[i] * iomk, theta[i] * ik, -z * ik2] in
                          JuMP.MOI.PowerCone(omk)
                          (epsilon + omega - x) .- t <= 0
                      end)
    JuMP.@objective(model, Min, risk)
    return if optimise_JuMP_model!(model, slv).success
        JuMP.objective_value(model)
    else
        model = JuMP.Model()
        JuMP.set_string_names_on_creation(model, false)
        JuMP.@variables(model, begin
                            z[1:T]
                            nu[1:T]
                            tau[1:T]
                        end)
        JuMP.@constraints(model, begin
                              sum(z) - 1 == 0
                              sum(nu - tau) * ik2 - ln_k <= 0
                          end)
        JuMP.@expression(model, risk, -LinearAlgebra.dot(z, x))
        if isnothing(w)
            JuMP.@constraints(model,
                              begin
                                  [i = 1:T], [nu[i], 1, z[i]] in JuMP.MOI.PowerCone(iopk)
                                  [i = 1:T], [z[i], 1, tau[i]] in JuMP.MOI.PowerCone(omk)
                              end)
        else
            wi = w / sum(w)
            JuMP.@constraints(model,
                              begin
                                  [i = 1:T],
                                  [nu[i], wi[i] * T, z[i]] in JuMP.MOI.PowerCone(iopk)
                                  [i = 1:T],
                                  [z[i], wi[i] * T, tau[i]] in JuMP.MOI.PowerCone(omk)
                              end)
        end
        JuMP.@objective(model, Max, risk)
        if optimise_JuMP_model!(model, slv).success
            JuMP.objective_value(model)
        else
            NaN
        end
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relativistic Value-at-Risk (RLVaR) risk measure.

`RelativisticValueatRisk` is a coherent risk measure generalising EVaR via the Kaniadakis (``\\kappa``-deformed) entropy. It is parametrised by a deformation parameter ``\\kappa \\in (0, 1)`` and reduces to EVaR in the limit ``\\kappa \\to 0``. It is solved via a conic programme. It is the Kaniadakis counterpart of the Kullback-Leibler ambiguity ball that [`EntropicValueatRisk`](@ref) reads as a risk measure.

# Mathematical definition

The RLVaR is:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\underset{t,\\, z}{\\min} \\Bigl\\{ t + \\ln_{\\kappa}\\!\\left(\\tfrac{1}{\\alpha T}\\right) z + \\sum_{i=1}^{T} (\\psi_i + \\theta_i) \\;:\\; z \\geq 0 \\Bigr\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x})``: Relativistic Value-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:ln_kappa])
  - ``t``, ``z``, ``\\psi_i``, ``\\theta_i``, ``\\epsilon_i``, ``\\omega_i``: Conic optimisation variables.

subject to the power-cone constraints:

```math
\\begin{align}
& \\left(\\tfrac{z(1+\\kappa)}{2\\kappa},\\, \\tfrac{\\psi_i(1+\\kappa)}{\\kappa},\\, \\epsilon_i\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{1+\\kappa}\\right) \\quad \\forall i\\,,\\\\
& \\left(\\tfrac{\\omega_i}{1-\\kappa},\\, \\tfrac{\\theta_i}{\\kappa},\\, -\\tfrac{z}{2\\kappa}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa) \\quad \\forall i\\,,\\\\
& \\epsilon_i + \\omega_i \\leq x_i + t \\quad \\forall i\\,.
\\end{align}
```

Where:

  - ``\\mathcal{K}_{\\mathrm{pow}}(p) = \\{(a,b,c) : a^p b^{1-p} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}``: Power cone.

For observation-weighted samples the weight vector is normalised to ``\\boldsymbol{w}`` with ``\\sum_{t=1}^{T} w_t = 1``. The Kaniadakis logarithm keeps the argument ``\\frac{1}{\\alpha T}``, and the sum ``\\sum_{i=1}^{T} (\\psi_i + \\theta_i)`` becomes ``T \\sum_{i=1}^{T} w_i (\\psi_i + \\theta_i)``. The Kaniadakis logarithm has no multiplication-to-addition property, so the normalisation ``\\alpha T`` cannot absorb the weights the way it does for [`EntropicValueatRisk`](@ref).

The dual of that programme is the worst expected loss over a Kaniadakis ball about the sample distribution:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\underset{Q \\in \\mathcal{Q}_{\\kappa}(\\alpha)}{\\sup} \\mathbb{E}_{Q}[L]\\,, \\\\
\\mathcal{Q}_{\\kappa}(\\alpha) &= \\left\\{ Q : \\sum_{t=1}^{T} q_t \\ln_{\\kappa}\\!\\left(\\frac{q_t}{p_t T}\\right) \\leq \\ln_{\\kappa}\\!\\left(\\frac{1}{\\alpha T}\\right) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{Q}_{\\kappa}(\\alpha)``: Kaniadakis ambiguity ball of radius ``\\ln_{\\kappa}\\!\\left(\\frac{1}{\\alpha T}\\right)``.
  - $(math_dict[:amb_Q])
  - $(math_dict[:amb_P])
  - $(math_dict[:amb_EQ_L])
  - $(math_dict[:amb_L_t])

The left side takes the place the Kullback-Leibler divergence holds for [`EntropicValueatRisk`](@ref). With equal observation weights ``p_t = 1/T`` it is the negated Kaniadakis entropy of ``Q``. Because ``\\ln_{\\kappa}`` has no multiplication-to-addition property, the sample size ``T`` stays inside both sides, and neither side separates into a term in ``T`` and a term in ``\\alpha``. The Kullback-Leibler ball at radius ``-\\ln(\\alpha)`` is recovered in the limit ``\\kappa \\to 0``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigTailCal = 0.05,
        kappa::Num_DefTailCal = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativisticValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `kappa` is a number: `0 < kappa < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativisticValueatRisk)(x::VecNum)

Computes the RLVaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativisticValueatRisk()
RelativisticValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
     kappa ┼ Float64: 0.3
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`RelativisticValueatRiskRange`](@ref)
  - [`RelativisticDrawdownatRisk`](@ref)
  - [`kappa_log`](@ref)

# References

  - $(ref_dict[:rlvar])
"""
@propagatable @concrete struct RelativisticValueatRisk <: RiskMeasure
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
    $(field_dict[:kappa])
    """
    kappa
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativisticValueatRisk(settings::RiskMeasureSettings,
                                     slv::Option{<:Slv_VecSlv}, alpha::Num_SigTailCal,
                                     kappa::Num_DefTailCal, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(kappa, :kappa)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa), typeof(w)}(settings,
                                                                                           slv,
                                                                                           alpha,
                                                                                           kappa,
                                                                                           w)
    end
end
function RelativisticValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 slv::Option{<:Slv_VecSlv} = nothing,
                                 alpha::Num_SigTailCal = 0.05, kappa::Num_DefTailCal = 0.3,
                                 w::Option{<:ObsWeights} = nothing)::RelativisticValueatRisk
    return RelativisticValueatRisk(settings, slv, alpha, kappa, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha` and the deformation parameter `kappa` of a [`RelativisticValueatRisk`](@ref) against prior result `pr`.

`alpha` and `kappa` are a **travelling pair**: [`EntropyBudget`](@ref) reads the significance level of its sibling slot. So `alpha` resolves first, and the number it produced is handed to the `kappa` slot through [`bind_alpha`](@ref) before that slot is resolved. A stated number, a plain function and a rule that reads no sibling all pass through `bind_alpha` untouched, so the order costs nothing where no rule reads a sibling.

The series this measure prices travels the same way, through [`bind_series`](@ref). It is the returns, which is the default [`calibration_series`](@ref) states, so the call binds what a rule already carries and is written for the reason every site writes it: the marker belongs to the measure, and a rule that carries a drawdown marker into this slot is corrected rather than obeyed.

The solver is settled once, as `sel(x.slv, slv)`, and handed to both rules, so a rule may call [`RRM`](@ref) itself. The rebuild goes through [`rebuild_with_slots`](@ref), whose positional call runs the inner constructor and re-runs both range checks on the calibrated numbers.

# Related

  - [`RelativisticValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series`](@ref)
  - [`EntropyBudget`](@ref)
"""
function resolve_deferred_quantities(x::RelativisticValueatRisk, pr::AbstractPriorResult,
                                     slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa = resolve_calibration_slot(bind_series(bind_alpha(x.kappa, alpha), s), :kappa, pr,
                                     ws, sv)
    return rebuild_with_slots(x, (; alpha = alpha, kappa = kappa))
end
# Calibration slots — see `calibration_slots`. The two travel together, and the resolution
# above is what orders them.
calibration_slots(x::RelativisticValueatRisk) = (; alpha = x.alpha, kappa = x.kappa)
function (r::RelativisticValueatRisk)(x::VecNum)
    return RRM(x, r.slv, r.alpha, r.kappa, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relativistic Value-at-Risk Range (RLVaR Range) risk measure.

`RelativisticValueatRiskRange` computes the sum of the lower-tail RLVaR (at level `alpha` with deformation `kappa_a`) and the upper-tail RLVaR (at level `beta` with deformation `kappa_b`).

# Mathematical definition

```math
\\begin{align}
\\mathrm{RVaRRange}_{\\alpha,\\kappa_a,\\beta,\\kappa_b}(\\boldsymbol{x}) &= \\mathrm{RLVaR}_{\\alpha,\\kappa_a}(\\boldsymbol{x}) + \\mathrm{RLVaR}_{\\beta,\\kappa_b}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RVaRRange}_{\\alpha,\\kappa_a,\\beta,\\kappa_b}(\\boldsymbol{x})``: Relativistic VaR range.
  - $(math_dict[:xret])
  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa_a}(\\boldsymbol{x})``: Lower-tail RLVaR with parameters ``(\\alpha, \\kappa_a)``.
  - ``\\mathrm{RLVaR}_{\\beta,\\kappa_b}(-\\boldsymbol{x})``: Upper-tail RLVaR with parameters ``(\\beta, \\kappa_b)``.

$(math_dict[:negated_upper_tail])

Each term is the worst expected loss over its own Kaniadakis ball about the sample distribution, one deformed by ``\\kappa_a`` at level ``\\alpha`` and one deformed by ``\\kappa_b`` at level ``\\beta``. [`RelativisticValueatRisk`](@ref) states the ball.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigTailCal = 0.05,
        kappa_a::Num_DefTailCal = 0.3,
        beta::Num_SigHeadCal = 0.05,
        kappa_b::Num_DefHeadCal = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativisticValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - Each of `alpha` and `kappa_a` that is a number: `0 < val < 1`.
  - Each of `beta` and `kappa_b` that is a number: `0 < val < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativisticValueatRiskRange)(x::VecNum)

Computes the RLVaR Range of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativisticValueatRiskRange()
RelativisticValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
   kappa_a ┼ Float64: 0.3
      beta ┼ Float64: 0.05
   kappa_b ┼ Float64: 0.3
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`EntropicValueatRiskRange`](@ref)

# References

  - $(ref_dict[:rlvar])
"""
@propagatable @concrete struct RelativisticValueatRiskRange <: RiskMeasure
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
    $(field_dict[:kappa_a])
    """
    kappa_a
    """
    $(field_dict[:beta])
    """
    beta
    """
    $(field_dict[:kappa_b])
    """
    kappa_b
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativisticValueatRiskRange(settings::RiskMeasureSettings,
                                          slv::Option{<:Slv_VecSlv}, alpha::Num_SigTailCal,
                                          kappa_a::Num_DefTailCal, beta::Num_SigHeadCal,
                                          kappa_b::Num_DefHeadCal, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(kappa_a, :kappa_a)
        assert_unit_interval(beta, :beta)
        assert_unit_interval(kappa_b, :kappa_b)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa_a),
                   typeof(beta), typeof(kappa_b), typeof(w)}(settings, slv, alpha, kappa_a,
                                                             beta, kappa_b, w)
    end
end
function RelativisticValueatRiskRange(;
                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                      slv::Option{<:Slv_VecSlv} = nothing,
                                      alpha::Num_SigTailCal = 0.05,
                                      kappa_a::Num_DefTailCal = 0.3,
                                      beta::Num_SigHeadCal = 0.05,
                                      kappa_b::Num_DefHeadCal = 0.3,
                                      w::Option{<:ObsWeights} = nothing)::RelativisticValueatRiskRange
    return RelativisticValueatRiskRange(settings, slv, alpha, kappa_a, beta, kappa_b, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the two significance levels and the two deformation parameters of a [`RelativisticValueatRiskRange`](@ref) against prior result `pr`.

Each end carries a **travelling pair** of its own: `kappa_a` reads `alpha` and `kappa_b` reads `beta`. So the resolution runs the pair of the loss side and then the pair of the gain side, and neither side reads the other's number. That is the pairing [`range_tails`](@ref) builds and the functor evaluates.

The four slots carry four different bounds, so a rule of the wrong end or the wrong family is refused at construction. The solver is settled once and handed to all four rules.

Both ends price one series, which is the returns, so [`bind_series`](@ref) carries the same marker to both `kappa` slots. The series is a property of the measure and not of an end, where the significance level is a property of the end.

# Related

  - [`RelativisticValueatRiskRange`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series`](@ref)
  - [`EntropyBudget`](@ref)
"""
function resolve_deferred_quantities(x::RelativisticValueatRiskRange,
                                     pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa_a = resolve_calibration_slot(bind_series(bind_alpha(x.kappa_a, alpha), s),
                                       :kappa_a, pr, ws, sv)
    beta = resolve_calibration_slot(x.beta, :beta, pr, ws, sv)
    kappa_b = resolve_calibration_slot(bind_series(bind_alpha(x.kappa_b, beta), s),
                                       :kappa_b, pr, ws, sv)
    return rebuild_with_slots(x,
                              (; alpha = alpha, kappa_a = kappa_a, beta = beta,
                               kappa_b = kappa_b))
end
# Calibration slots — see `calibration_slots`. One travelling pair per tail.
function calibration_slots(x::RelativisticValueatRiskRange)
    return (; alpha = x.alpha, kappa_a = x.kappa_a, beta = x.beta, kappa_b = x.kappa_b)
end
# Tail decomposition — see `range_tails`. Each tail carries its own deformation parameter:
# `kappa_a` shapes the loss side, `kappa_b` the gain side. The functor below is the
# value-level twin, and it is what pins that pairing.
function range_tails(r::RelativisticValueatRiskRange)
    settings = RiskMeasureSettings(; rke = false)
    return (;
            loss = RelativisticValueatRisk(; settings = settings, slv = r.slv,
                                           alpha = r.alpha, kappa = r.kappa_a, w = r.w),
            gain = RelativisticValueatRisk(; settings = settings, slv = r.slv,
                                           alpha = r.beta, kappa = r.kappa_b, w = r.w))
end
function (r::RelativisticValueatRiskRange)(x::VecNum)
    return RRM(x, r.slv, r.alpha, r.kappa_a, r.w) + RRM(-x, r.slv, r.beta, r.kappa_b, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relativistic Drawdown-at-Risk (RLDaR) risk measure.

`RelativisticDrawdownatRisk` applies the Relativistic Value-at-Risk framework to the absolute drawdown series of portfolio returns.

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

The Relativistic Drawdown-at-Risk is the RLVaR of the drawdown series:

```math
\\begin{align}
\\mathrm{RLDaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{d}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLDaR}_{\\alpha,\\kappa}(\\boldsymbol{x})``: Relativistic Drawdown-at-Risk.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - ``\\boldsymbol{d}(\\boldsymbol{x})``: Absolute drawdown series vector ``T \\times 1``.

So the RLDaR is the worst expected drawdown over a Kaniadakis ball about the sample distribution of ``\\boldsymbol{d}(\\boldsymbol{x})``. [`RelativisticValueatRisk`](@ref) states the ball.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigTailCal = 0.05,
        kappa::Num_DefTailCal = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativisticDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `kappa` is a number: `0 < kappa < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativisticDrawdownatRisk)(x::VecNum)

Computes the Relativistic Drawdown-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativisticDrawdownatRisk()
RelativisticDrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
       slv ┼ nothing
     alpha ┼ Float64: 0.05
     kappa ┼ Float64: 0.3
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`EntropicDrawdownatRisk`](@ref)
  - [`RelativeRelativisticDrawdownatRisk`](@ref)

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:rlvar])
"""
@propagatable @concrete struct RelativisticDrawdownatRisk <: RiskMeasure
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
    $(field_dict[:kappa])
    """
    kappa
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativisticDrawdownatRisk(settings::RiskMeasureSettings,
                                        slv::Option{<:Slv_VecSlv}, alpha::Num_SigTailCal,
                                        kappa::Num_DefTailCal, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(kappa, :kappa)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa), typeof(w)}(settings,
                                                                                           slv,
                                                                                           alpha,
                                                                                           kappa,
                                                                                           w)
    end
end
function RelativisticDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                    slv::Option{<:Slv_VecSlv} = nothing,
                                    alpha::Num_SigTailCal = 0.05,
                                    kappa::Num_DefTailCal = 0.3,
                                    w::Option{<:ObsWeights} = nothing)::RelativisticDrawdownatRisk
    return RelativisticDrawdownatRisk(settings, slv, alpha, kappa, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha` and the deformation parameter `kappa` of a [`RelativisticDrawdownatRisk`](@ref) against prior result `pr`.

It carries the reading of [`resolve_deferred_quantities`](@ref) on the value-at-risk twin unchanged: `alpha` resolves first and travels to the `kappa` slot through [`bind_alpha`](@ref). The drawdown series has one entry per row of the sample, so a rule reads the same sample size here as it does there.

**The series does not carry over, and [`bind_series`](@ref) is what says so.** This measure prices the absolute drawdown series of the portfolio, so [`calibration_series`](@ref) states [`AbsoluteDrawdownSeries`](@ref) and the marker travels beside `alpha`. A rule that reads the shape of a series then reads the drawdown series of each column of the sample, in place of the columns themselves, and the `alpha` it reads is the level of that same drawdown series. The key `:kappa` names this slot and the twin's slot alike, so nothing else could have told the rule which quantity it stands in front of.

# Related

  - [`RelativisticDrawdownatRisk`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::RelativisticDrawdownatRisk, pr::AbstractPriorResult,
                                     slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa = resolve_calibration_slot(bind_series(bind_alpha(x.kappa, alpha), s), :kappa, pr,
                                     ws, sv)
    return rebuild_with_slots(x, (; alpha = alpha, kappa = kappa))
end
# Calibration slots — see `calibration_slots`.
calibration_slots(x::RelativisticDrawdownatRisk) = (; alpha = x.alpha, kappa = x.kappa)
# Calibration series — see `calibration_series`. The measure prices the drawdown series of
# the portfolio, so a rule reads the drawdown series of each column and not the columns.
calibration_series(::RelativisticDrawdownatRisk) = AbsoluteDrawdownSeries()
function (r::RelativisticDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return RRM(dd, r.slv, r.alpha, r.kappa, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Relativistic Drawdown-at-Risk (Relative RLDaR) risk measure for hierarchical optimisation.

`RelativeRelativisticDrawdownatRisk` applies the Relativistic Value-at-Risk framework to the relative (compounded) drawdown series of portfolio returns.

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

The Relative Relativistic Drawdown-at-Risk is the RLVaR of the relative drawdown series:

```math
\\begin{align}
\\mathrm{RRDDaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{rd}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRDDaR}_{\\alpha,\\kappa}(\\boldsymbol{x})``: Relative Relativistic Drawdown-at-Risk.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - ``\\boldsymbol{rd}(\\boldsymbol{x})``: Relative drawdown series vector ``T \\times 1``.

So the Relative RLDaR is the worst expected relative drawdown over a Kaniadakis ball about the sample distribution of ``\\boldsymbol{rd}(\\boldsymbol{x})``. [`RelativisticValueatRisk`](@ref) states the ball.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeRelativisticDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigTailCal = 0.05,
        kappa::Num_DefTailCal = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeRelativisticDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `alpha` is a number: `0 < alpha < 1`.
  - If `kappa` is a number: `0 < kappa < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativeRelativisticDrawdownatRisk)(x::VecNum)

Computes the Relative Relativistic Drawdown-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeRelativisticDrawdownatRisk()
RelativeRelativisticDrawdownatRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
       slv ┼ nothing
     alpha ┼ Float64: 0.05
     kappa ┼ Float64: 0.3
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`RelativisticDrawdownatRisk`](@ref)
  - [`RelativeEntropicDrawdownatRisk`](@ref)

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:rlvar])
"""
@propagatable @concrete struct RelativeRelativisticDrawdownatRisk <: HierarchicalRiskMeasure
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
    $(field_dict[:kappa])
    """
    kappa
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativeRelativisticDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                                slv::Option{<:Slv_VecSlv},
                                                alpha::Num_SigTailCal,
                                                kappa::Num_DefTailCal,
                                                w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(kappa, :kappa)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa), typeof(w)}(settings,
                                                                                           slv,
                                                                                           alpha,
                                                                                           kappa,
                                                                                           w)
    end
end
function RelativeRelativisticDrawdownatRisk(;
                                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Num_SigTailCal = 0.05,
                                            kappa::Num_DefTailCal = 0.3,
                                            w::Option{<:ObsWeights} = nothing)::RelativeRelativisticDrawdownatRisk
    return RelativeRelativisticDrawdownatRisk(settings, slv, alpha, kappa, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha` and the deformation parameter `kappa` of a [`RelativeRelativisticDrawdownatRisk`](@ref) against prior result `pr`.

The measure is a hierarchical one, so it reaches no `JuMP` model and the [`factory`](@ref) route is its only resolution. The travelling pair is resolved in the order the absolute twin states.

The series is the twin's reading in its own units: this measure compounds the path, so [`calibration_series`](@ref) states [`RelativeDrawdownSeries`](@ref) and [`bind_series`](@ref) carries it. The two markers name two different series of the same column, and a rule that reads the shape of a series answers differently on each.

# Related

  - [`RelativeRelativisticDrawdownatRisk`](@ref)
  - [`RelativisticDrawdownatRisk`](@ref)
  - [`RelativeDrawdownSeries`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::RelativeRelativisticDrawdownatRisk,
                                     pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa = resolve_calibration_slot(bind_series(bind_alpha(x.kappa, alpha), s), :kappa, pr,
                                     ws, sv)
    return rebuild_with_slots(x, (; alpha = alpha, kappa = kappa))
end
# Calibration slots — see `calibration_slots`.
function calibration_slots(x::RelativeRelativisticDrawdownatRisk)
    return (; alpha = x.alpha, kappa = x.kappa)
end
# Calibration series — see `calibration_series`. The path compounds here, where the absolute
# twin sums it, so the two measures name two different series of one column.
calibration_series(::RelativeRelativisticDrawdownatRisk) = RelativeDrawdownSeries()
function (r::RelativeRelativisticDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return RRM(dd, r.slv, r.alpha, r.kappa, r.w)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::RelativisticValueatRisk) = NetReturnsInput()
risk_input_kind(::RelativisticValueatRiskRange) = NetReturnsInput()
risk_input_kind(::RelativisticDrawdownatRisk) = NetReturnsInput()
risk_input_kind(::RelativeRelativisticDrawdownatRisk) = NetReturnsInput()

export RelativisticValueatRisk, RelativisticValueatRiskRange, RelativisticDrawdownatRisk,
       RelativeRelativisticDrawdownatRisk
