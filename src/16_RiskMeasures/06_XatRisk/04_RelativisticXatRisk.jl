"""
    RRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, kappa::Number = 0.3,
        w::Option{<:ObsWeights} = nothing) -> Number

Compute the Relativistic Value-at-Risk of the returns `x` with a power-cone programme.

[`RelativisticValueatRisk`](@ref) states the measure, its Kaniadakis ball and its primal programme. Every call builds a new `JuMP` model. The primal and the dual programmes have the same optimal value, but a solver can stall on one of them and solve the other, so the dual programme is the fallback of the primal one.

# Algorithm

 1. Check that `slv` is not empty when it is a vector.
 2. When `w` is stated, divide it by its sum, giving `wi`.
 3. Compute `ln_k`, the Kaniadakis logarithm of ``1/(\\alpha T)``.
 4. Build the primal programme in a new model, and solve it with the solvers of `slv` in turn.
 5. When a solver solves the primal programme, return its objective value.
 6. Otherwise, build the dual programme in a new model, and solve it with the solvers of `slv` in turn.
 7. When a solver solves the dual programme, return its objective value. Otherwise, return `NaN`.

# JuMP formulation

## Variables

The primal model creates these variables:

  - `t`: free scalar ``t``.
  - `z`: scalar ``z``, bounded below by zero.
  - `omega`, `psi`, `theta`, `epsilon`: free ``T \\times 1`` vectors ``\\boldsymbol{\\omega}``, ``\\boldsymbol{\\psi}``, ``\\boldsymbol{\\theta}`` and ``\\boldsymbol{\\epsilon}``.

The dual model creates these variables:

  - `z`, `nu`, `tau`: free ``T \\times 1`` vectors ``\\boldsymbol{q}``, ``\\boldsymbol{\\nu}`` and ``\\boldsymbol{\\tau}``.

## Expressions

  - `risk`, in the primal model: ``t + \\ln_{\\kappa}\\!\\left(\\tfrac{1}{\\alpha T}\\right) z + T \\sum_{i=1}^{T} p_i (\\psi_i + \\theta_i)``.
  - `risk`, in the dual model: ``-\\boldsymbol{q}^\\intercal \\boldsymbol{x}``.

## Constraints

No row carries a name. The primal model registers these rows:

  - ``\\left(\\tfrac{z(1+\\kappa)}{2\\kappa},\\, \\tfrac{\\psi_i(1+\\kappa)}{\\kappa},\\, \\epsilon_i\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{1+\\kappa}\\right)``, one row for each ``i``.
  - ``\\left(\\tfrac{\\omega_i}{1-\\kappa},\\, \\tfrac{\\theta_i}{\\kappa},\\, -\\tfrac{z}{2\\kappa}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)``, one row for each ``i``.
  - ``\\epsilon_i + \\omega_i - x_i - t \\leq 0``, one row for each ``i``.

The dual model registers these rows:

  - ``\\sum_{i=1}^{T} q_i - 1 = 0``.
  - ``\\tfrac{1}{2\\kappa} \\sum_{i=1}^{T} (\\nu_i - \\tau_i) - \\ln_{\\kappa}\\!\\left(\\tfrac{1}{\\alpha T}\\right) \\leq 0``.
  - ``(\\nu_i,\\, T p_i,\\, q_i) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{1+\\kappa}\\right)``, one row for each ``i``.
  - ``(q_i,\\, T p_i,\\, \\tau_i) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)``, one row for each ``i``.

The two power-cone rows of the dual model give ``\\nu_i \\geq q_i^{1+\\kappa} (T p_i)^{-\\kappa}`` and ``\\tau_i \\leq q_i^{1-\\kappa} (T p_i)^{\\kappa}``. So the second row is the Kaniadakis ball of [`RelativisticValueatRisk`](@ref), ``\\sum_{i} q_i \\ln_{\\kappa}\\!\\left(\\tfrac{q_i}{p_i T}\\right) \\leq \\ln_{\\kappa}\\!\\left(\\tfrac{1}{\\alpha T}\\right)``.

## Objective

  - The primal model minimises `risk`, and the dual model maximises `risk`. The two optimal values are equal.

Where:

  - $(math_dict[:xret])
  - $(math_dict[:T])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:p_i_obs])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z_ge])
  - $(math_dict[:rlvar_aux])
  - ``\\boldsymbol{q}``: ``T \\times 1`` probability vector of the dual programme, whose ``i``-th entry is ``q_i``.
  - ``\\nu_i``, ``\\tau_i``: Auxiliary variables of observation ``i`` in the dual programme.
  - $(math_dict[:K_pow])

# Arguments

  - `x`: Vector of portfolio returns.
  - $(arg_dict[:slv])
  - `alpha`: Significance level, ``\\alpha \\in (0, 1)``.
  - `kappa`: Kaniadakis deformation parameter, ``\\kappa \\in (0, 1)``.
  - `w`: Observation weights, or `nothing` for equal weights.

# Validation

  - If `slv` is a `VecSlv`: `!isempty(slv)`, else `IsEmptyError`.

# Returns

  - The Relativistic Value-at-Risk of `x`, or `NaN` when no solver of `slv` solves either programme.

# Related

  - [`RelativisticValueatRisk`](@ref)
  - [`kappa_log`](@ref)
  - [`optimise_JuMP_model!`](@ref)
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

Measures the Relativistic Value-at-Risk (RLVaR), the worst expected loss over a Kaniadakis entropy ball about the sample distribution.

The RLVaR is a coherent risk measure. [`EntropicValueatRisk`](@ref) measures its ball with the Kullback-Leibler divergence, and the RLVaR tends to it as the deformation parameter ``\\kappa`` tends to zero.

# Mathematical definition

The RLVaR is the worst expected loss over a Kaniadakis ball about the sample distribution:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\underset{Q \\in \\mathcal{Q}_{\\kappa}(\\alpha)}{\\sup} \\mathbb{E}_{Q}[L]\\,, \\\\
\\mathcal{Q}_{\\kappa}(\\alpha) &= \\left\\{ Q : \\sum_{t=1}^{T} q_t \\ln_{\\kappa}\\!\\left(\\frac{q_t}{p_t T}\\right) \\leq \\ln_{\\kappa}\\!\\left(\\frac{1}{\\alpha T}\\right) \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x})``: Relativistic Value-at-Risk.
  - ``\\mathcal{Q}_{\\kappa}(\\alpha)``: Kaniadakis ambiguity ball of radius ``\\ln_{\\kappa}\\!\\left(\\frac{1}{\\alpha T}\\right)``.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:amb_Q])
  - $(math_dict[:amb_P])
  - $(math_dict[:amb_EQ_L])
  - $(math_dict[:amb_L_t])

The left side of the ball takes the place that the Kullback-Leibler divergence holds for [`EntropicValueatRisk`](@ref). With equal observation weights ``p_t = 1/T``, it is the negated Kaniadakis entropy of ``Q``. The Kaniadakis logarithm does not change a product into a sum, so the sample size ``T`` stays inside both sides, and neither side separates into a term in ``T`` and a term in ``\\alpha``.

Conic duality gives the equivalent primal programme:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) = \\underset{t,\\, z,\\, \\boldsymbol{\\psi},\\, \\boldsymbol{\\theta},\\, \\boldsymbol{\\epsilon},\\, \\boldsymbol{\\omega}}{\\min} \\quad & t + \\ln_{\\kappa}\\!\\left(\\tfrac{1}{\\alpha T}\\right) z + T \\sum_{i=1}^{T} p_i (\\psi_i + \\theta_i) \\\\
\\mathrm{s.t.} \\quad & \\left(\\tfrac{z(1+\\kappa)}{2\\kappa},\\, \\tfrac{\\psi_i(1+\\kappa)}{\\kappa},\\, \\epsilon_i\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{1+\\kappa}\\right) \\quad \\forall i\\,, \\\\
& \\left(\\tfrac{\\omega_i}{1-\\kappa},\\, \\tfrac{\\theta_i}{\\kappa},\\, -\\tfrac{z}{2\\kappa}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa) \\quad \\forall i\\,, \\\\
& \\epsilon_i + \\omega_i \\leq x_i + t \\quad \\forall i\\,, \\\\
& z \\geq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:p_i_obs])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z_ge])
  - $(math_dict[:rlvar_aux])
  - $(math_dict[:K_pow])

With equal observation weights, the sum in the objective is ``\\sum_{i=1}^{T} (\\psi_i + \\theta_i)``. The radius keeps the argument ``\\frac{1}{\\alpha T}`` when the weights are stated, because the Kaniadakis logarithm cannot absorb the weights into ``\\alpha T`` the way the natural logarithm does for [`EntropicValueatRisk`](@ref).

The definition has these consequences:

```math
\\begin{align}
\\lim_{\\kappa \\to 0} \\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x})\\,, \\\\
\\lim_{\\kappa \\to 1} \\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\max_{t} L_t\\,, \\\\
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) &\\leq \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) \\leq \\max_{t} L_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}_{\\alpha}``, ``\\mathrm{CVaR}_{\\alpha}``, ``\\mathrm{EVaR}_{\\alpha}``: Value-at-Risk, Conditional Value-at-Risk and Entropic Value-at-Risk at the same level ``\\alpha``.

With equal observation weights and ``\\alpha T \\leq 1``, the radius is not negative, so the ball holds every distribution on the sample, and the RLVaR is the largest loss ``\\max_{t} L_t``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        kappa::Num_DefCal = 0.3,
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

Computes the RLVaR of a portfolio returns vector `x` with [`RRM`](@ref), and returns `NaN` when no solver of `slv` solves the programme. The functor needs a solver: with `slv = nothing` the call raises a `MethodError`. [`factory`](@ref) fills `slv` from the solver of the optimiser.

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
  - [`RRM`](@ref)
  - [`kappa_log`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:rlvar])
  - $(ref_dict[:cajas2025]) Section 7.2.2.7, Equations 7.65 to 7.69.
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
                                     slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                     kappa::Num_DefCal, w::Option{<:ObsWeights})
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
                                 alpha::Num_SigCal = 0.05, kappa::Num_DefCal = 0.3,
                                 w::Option{<:ObsWeights} = nothing)::RelativisticValueatRisk
    return RelativisticValueatRisk(settings, slv, alpha, kappa, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha` and the deformation parameter `kappa` of a [`RelativisticValueatRisk`](@ref) against prior result `pr`.

`alpha` and `kappa` are a Travelling Pair. A rule in the `kappa` slot, such as [`EntropyBudget`](@ref), can read the significance level of the `alpha` slot, so `alpha` resolves first. A stated number, a plain function and a rule that reads no sibling ignore the context, so the order changes nothing for them.

# Algorithm

 1. Select the observation weights `ws` with `sel(x.w, pr.w)`.
 2. Select the solver `sv` with `sel(x.slv, slv)`. Both rules receive it, so a rule can call [`RRM`](@ref).
 3. Read the series marker `s` with [`calibration_series`](@ref). The measure prices the returns, which is also the default of the context.
 4. Resolve the `alpha` slot, giving `alpha`.
 5. Resolve the `kappa` slot with a [`CalibrationContext`](@ref) that holds `alpha` and `s`, giving `kappa`.
 6. Rebuild the measure with [`rebuild_with_slots`](@ref). Its positional call runs the inner constructor, which checks the range of both calibrated numbers again.

# Related

  - [`RelativisticValueatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series`](@ref)
  - [`EntropyBudget`](@ref)
"""
function resolve_deferred_quantities(x::RelativisticValueatRisk, pr::AbstractPriorResult,
                                     slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa = resolve_calibration_slot(x.kappa, :kappa, pr, ws, sv,
                                     CalibrationContext(; alpha = alpha, series = s))
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

Measures the spread of the returns as the sum of the Relativistic Value-at-Risk of the losses and that of the gains.

The loss tail has the level `alpha` and the deformation `kappa_a`. The gain tail has the level `beta` and the deformation `kappa_b`.

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

Each term is the worst expected loss over its own Kaniadakis ball about the sample distribution. The first ball has the deformation ``\\kappa_a`` and the level ``\\alpha``, and the second ball has the deformation ``\\kappa_b`` and the level ``\\beta``. [`RelativisticValueatRisk`](@ref) states the ball.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        kappa_a::Num_DefCal = 0.3,
        beta::Num_SigCal = alpha,
        kappa_b::Num_DefCal = kappa_a,
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

Computes the RLVaR Range of a portfolio returns vector `x` with two calls of [`RRM`](@ref). The functor needs a solver, as the functor of [`RelativisticValueatRisk`](@ref) does.

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
  - [`range_tails`](@ref)

# References

  - $(ref_dict[:rlvar])
  - $(ref_dict[:cajas2025]) Section 7.2.3.5, Equation 7.84.
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
                                          slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                          kappa_a::Num_DefCal, beta::Num_SigCal,
                                          kappa_b::Num_DefCal, w::Option{<:ObsWeights})
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
                                      alpha::Num_SigCal = 0.05, kappa_a::Num_DefCal = 0.3,
                                      beta::Num_SigCal = alpha,
                                      kappa_b::Num_DefCal = kappa_a,
                                      w::Option{<:ObsWeights} = nothing)::RelativisticValueatRiskRange
    return RelativisticValueatRiskRange(settings, slv, alpha, kappa_a, beta, kappa_b, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the two significance levels and the two deformation parameters of a [`RelativisticValueatRiskRange`](@ref) against prior result `pr`.

Each end has its own Travelling Pair: `kappa_a` reads `alpha`, and `kappa_b` reads `beta`. The gain-side pair defaults to the loss-side pair, `beta` to `alpha` and `kappa_b` to `kappa_a`, so a pair stated on the loss side alone reaches both ends. Neither end reads the number of the other end. [`range_tails`](@ref) and the functor use the same pairing.

The four slots have four different bounds, so the constructor refuses a rule of the wrong end or of the wrong family. Both ends price the returns, so one series marker goes into the context of both `kappa` slots, while each end has its own significance level.

# Algorithm

 1. Select the observation weights `ws` with `sel(x.w, pr.w)`.
 2. Select the solver `sv` with `sel(x.slv, slv)`. All four rules receive it.
 3. Read the series marker `s` with [`calibration_series`](@ref).
 4. Resolve the `alpha` slot, giving `alpha`.
 5. Resolve the `kappa_a` slot with a [`CalibrationContext`](@ref) that holds `alpha` and `s`, giving `kappa_a`.
 6. Resolve the `beta` slot, giving `beta`.
 7. Resolve the `kappa_b` slot with a [`CalibrationContext`](@ref) that holds `beta` as its significance level and `s`, giving `kappa_b`.
 8. Rebuild the measure with [`rebuild_with_slots`](@ref), which checks the range of all four calibrated numbers again.

# Related

  - [`RelativisticValueatRiskRange`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series`](@ref)
  - [`EntropyBudget`](@ref)
"""
function resolve_deferred_quantities(x::RelativisticValueatRiskRange,
                                     pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa_a = resolve_calibration_slot(x.kappa_a, :kappa_a, pr, ws, sv,
                                       CalibrationContext(; alpha = alpha, series = s))
    beta = resolve_calibration_slot(x.beta, :beta, pr, ws, sv)
    kappa_b = resolve_calibration_slot(x.kappa_b, :kappa_b, pr, ws, sv,
                                       CalibrationContext(; alpha = beta, series = s))
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

Measures the Relativistic Drawdown-at-Risk (RLDaR), the Relativistic Value-at-Risk of the absolute drawdown series of the portfolio.

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

So the RLDaR is the worst expected drawdown over a Kaniadakis ball about the sample distribution of ``\\boldsymbol{d}(\\boldsymbol{x})``. [`RelativisticValueatRisk`](@ref) states the ball. The order of that measure holds for the drawdowns too:

```math
\\begin{align}
\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{EDaR}_{\\alpha}(\\boldsymbol{x}) \\leq \\mathrm{RLDaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) \\leq \\mathrm{MDD}(\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DaR}_{\\alpha}``, ``\\mathrm{CDaR}_{\\alpha}``, ``\\mathrm{EDaR}_{\\alpha}``: Drawdown-at-Risk, Conditional Drawdown-at-Risk and Entropic Drawdown-at-Risk at the same level ``\\alpha``.
  - ``\\mathrm{MDD}(\\boldsymbol{x}) = -\\min_{t} d_t``: Maximum drawdown.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Num_SigCal = 0.05,
        kappa::Num_DefCal = 0.3,
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

Computes the Relativistic Drawdown-at-Risk of a portfolio returns vector `x` with [`RRM`](@ref) on its drawdown series. The functor needs a solver, as the functor of [`RelativisticValueatRisk`](@ref) does.

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
  - [`absolute_drawdown_vec`](@ref)

# References

  - $(ref_dict[:cdar])
  - $(ref_dict[:rlvar])
  - $(ref_dict[:cajas2025]) Section 7.2.4.6, Equations 7.99 and 7.100.
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
                                        slv::Option{<:Slv_VecSlv}, alpha::Num_SigCal,
                                        kappa::Num_DefCal, w::Option{<:ObsWeights})
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
                                    alpha::Num_SigCal = 0.05, kappa::Num_DefCal = 0.3,
                                    w::Option{<:ObsWeights} = nothing)::RelativisticDrawdownatRisk
    return RelativisticDrawdownatRisk(settings, slv, alpha, kappa, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha` and the deformation parameter `kappa` of a [`RelativisticDrawdownatRisk`](@ref) against prior result `pr`.

The order is the one that [`RelativisticValueatRisk`](@ref) uses: `alpha` resolves first, and it reaches the `kappa` slot in its [`CalibrationContext`](@ref). The drawdown series has one entry per row of the sample, so a rule reads the same sample size as it does for the value-at-risk measure.

The series differs from that of [`RelativisticValueatRisk`](@ref). This measure prices the absolute drawdown series of the portfolio, so [`calibration_series`](@ref) states [`AbsoluteDrawdownSeries`](@ref), and the context carries that marker beside `alpha`. A rule that reads the shape of a series then reads the drawdown series of each column of the sample, not the columns, and `alpha` is the level of that drawdown series. The key `:kappa` names this slot and the slot of [`RelativisticValueatRisk`](@ref) alike, so the marker is the only thing that tells the rule which series it reads.

# Algorithm

 1. Select the observation weights `ws` with `sel(x.w, pr.w)`.
 2. Select the solver `sv` with `sel(x.slv, slv)`.
 3. Read the series marker `s`, which is [`AbsoluteDrawdownSeries`](@ref).
 4. Resolve the `alpha` slot, giving `alpha`.
 5. Resolve the `kappa` slot with a [`CalibrationContext`](@ref) that holds `alpha` and `s`, giving `kappa`.
 6. Rebuild the measure with [`rebuild_with_slots`](@ref), which checks the range of both calibrated numbers again.

# Related

  - [`RelativisticDrawdownatRisk`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::RelativisticDrawdownatRisk, pr::AbstractPriorResult,
                                     slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa = resolve_calibration_slot(x.kappa, :kappa, pr, ws, sv,
                                     CalibrationContext(; alpha = alpha, series = s))
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

Measures the Relative Relativistic Drawdown-at-Risk, the Relativistic Value-at-Risk of the compounded drawdown series, for hierarchical optimisation.

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
        alpha::Num_SigCal = 0.05,
        kappa::Num_DefCal = 0.3,
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

Computes the Relative Relativistic Drawdown-at-Risk of a portfolio returns vector `x` with [`RRM`](@ref) on its relative drawdown series. The functor needs a solver, as the functor of [`RelativisticValueatRisk`](@ref) does.

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
                                                alpha::Num_SigCal, kappa::Num_DefCal,
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
                                            alpha::Num_SigCal = 0.05,
                                            kappa::Num_DefCal = 0.3,
                                            w::Option{<:ObsWeights} = nothing)::RelativeRelativisticDrawdownatRisk
    return RelativeRelativisticDrawdownatRisk(settings, slv, alpha, kappa, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the significance level `alpha` and the deformation parameter `kappa` of a [`RelativeRelativisticDrawdownatRisk`](@ref) against prior result `pr`.

The measure is hierarchical, so it reaches no `JuMP` model, and [`factory`](@ref) is the only route that resolves it. The Travelling Pair resolves in the order that [`RelativisticDrawdownatRisk`](@ref) uses.

This measure compounds the path, so [`calibration_series`](@ref) states [`RelativeDrawdownSeries`](@ref), and the context carries that marker. The two drawdown measures name two different series of the same column, and a rule that reads the shape of a series gives a different answer on each.

# Algorithm

 1. Select the observation weights `ws` with `sel(x.w, pr.w)`.
 2. Select the solver `sv` with `sel(x.slv, slv)`.
 3. Read the series marker `s`, which is [`RelativeDrawdownSeries`](@ref).
 4. Resolve the `alpha` slot, giving `alpha`.
 5. Resolve the `kappa` slot with a [`CalibrationContext`](@ref) that holds `alpha` and `s`, giving `kappa`.
 6. Rebuild the measure with [`rebuild_with_slots`](@ref), which checks the range of both calibrated numbers again.

# Related

  - [`RelativeRelativisticDrawdownatRisk`](@ref)
  - [`RelativisticDrawdownatRisk`](@ref)
  - [`RelativeDrawdownSeries`](@ref)
  - [`CalibrationContext`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::RelativeRelativisticDrawdownatRisk,
                                     pr::AbstractPriorResult, slv = nothing)
    ws = sel(x.w, pr.w)
    sv = sel(x.slv, slv)
    s = calibration_series(x)
    alpha = resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv)
    kappa = resolve_calibration_slot(x.kappa, :kappa, pr, ws, sv,
                                     CalibrationContext(; alpha = alpha, series = s))
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
