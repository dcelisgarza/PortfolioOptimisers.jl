"""
    RRM(x, slv, alpha = 0.05, kappa = 0.3, ...; kwargs...)

Compute the Relativistic Risk Measure (RRM) for a vector of portfolio returns.

Solves a convex optimisation problem to compute the RRM at confidence level `alpha` with relativistic parameter `kappa`, using the specified solver(s).

# Arguments

  - `x`: Vector of portfolio returns.
  - `slv`: Solver or vector of solvers.
  - `alpha`: Confidence level (default `0.05`).
  - `kappa`: Relativistic parameter (default `0.3`).
  - Additional parameters depending on the specific RRM formulation.
  - `kwargs...`: Additional keyword arguments passed to the solver.

# Returns

  - RRM value (scalar).

# Related

  - [`RelativisticValueatRisk`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
function RRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, kappa::Number = 0.3,
             w::Option{<:ObsWeights} = nothing)
    w = get_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
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
    if isnothing(w)
        invat = inv(alpha * T)
        ln_k = (invat^kappa - invat^(-kappa)) * ik2
        JuMP.@expression(model, risk, t + ln_k * z + sum(psi + theta))
    else
        sw = sum(w)
        invat = inv(alpha * sw)
        ln_k = (invat^kappa - invat^(-kappa)) * ik2
        JuMP.@expression(model, risk, t + ln_k * z + LinearAlgebra.dot(w, psi + theta))
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
        if isnothing(w)
            JuMP.@constraints(model, begin
                                  sum(z) - 1 == 0
                                  sum(nu - tau) * ik2 - ln_k <= 0
                              end)
            JuMP.@expression(model, risk, -LinearAlgebra.dot(z, x))
        else
            JuMP.@constraints(model, begin
                                  LinearAlgebra.dot(w, z) - 1 == 0
                                  LinearAlgebra.dot(w, nu - tau) * ik2 - ln_k <= 0
                              end)
            JuMP.@expression(model, risk, -LinearAlgebra.dot(w .* z, x))
        end
        JuMP.@constraints(model,
                          begin
                              [i = 1:T], [nu[i], 1, z[i]] in JuMP.MOI.PowerCone(iopk)
                              [i = 1:T], [z[i], 1, tau[i]] in JuMP.MOI.PowerCone(omk)
                          end)
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

Represents the Relativistic Value-at-Risk (RVaR) risk measure.

`RelativisticValueatRisk` is a coherent risk measure generalising EVaR via the Tsallis (``\\kappa``-deformed) entropy. It is parametrised by a deformation parameter ``\\kappa \\in (0, 1)`` and reduces to EVaR in the limit ``\\kappa \\to 0``. It is solved via a conic programme.

# Mathematical definition

Define the ``\\kappa``-logarithm ``\\ell_\\kappa(u) = \\frac{u^\\kappa - u^{-\\kappa}}{2\\kappa}``. The RVaR is:

```math
\\begin{align}
\\mathrm{RVaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\underset{t,\\, z}{\\min} \\Bigl\\{ t + \\ell_\\kappa(\\alpha T)\\, z + \\sum_{i=1}^{T} (\\psi_i + \\theta_i) \\;:\\; z \\geq 0 \\Bigr\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RVaR}_{\\alpha,\\kappa}(\\boldsymbol{x})``: Relativistic Value-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``\\kappa \\in (0,1)``: Tsallis deformation parameter.
  - ``\\ell_\\kappa(u) = \\frac{u^\\kappa - u^{-\\kappa}}{2\\kappa}``: ``\\kappa``-logarithm.
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

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativisticValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativisticValueatRisk)(x::VecNum)

Computes the RVaR of a portfolio returns vector `x`.

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
"""
@concrete struct RelativisticValueatRisk <: RiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    "$(field_dict[:slv])"
    slv
    "$(field_dict[:alpha])"
    alpha
    "$(field_dict[:kappa])"
    kappa
    "$(field_dict[:oow])"
    w
    function RelativisticValueatRisk(settings::RiskMeasureSettings,
                                     slv::Option{<:Slv_VecSlv}, alpha::Number,
                                     kappa::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa) < kappa < one(kappa))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa), typeof(w)}(settings,
                                                                                           slv,
                                                                                           alpha,
                                                                                           kappa,
                                                                                           w)
    end
end
function RelativisticValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                 kappa::Number = 0.3,
                                 w::Option{<:ObsWeights} = nothing)::RelativisticValueatRisk
    return RelativisticValueatRisk(settings, slv, alpha, kappa, w)
end
function (r::RelativisticValueatRisk)(x::VecNum)
    return RRM(x, r.slv, r.alpha, r.kappa, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relativistic Value-at-Risk Range (RVaR Range) risk measure.

`RelativisticValueatRiskRange` computes the sum of the lower-tail RVaR (at level `alpha` with deformation `kappa_a`) and the upper-tail RVaR (at level `beta` with deformation `kappa_b`).

# Mathematical definition

```math
\\begin{align}
\\mathrm{RVaRRange}_{\\alpha,\\kappa_a,\\beta,\\kappa_b}(\\boldsymbol{x}) &= \\mathrm{RVaR}_{\\alpha,\\kappa_a}(\\boldsymbol{x}) + \\mathrm{RVaR}_{\\beta,\\kappa_b}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RVaRRange}_{\\alpha,\\kappa_a,\\beta,\\kappa_b}(\\boldsymbol{x})``: Relativistic VaR range.
  - $(math_dict[:xret])
  - ``\\mathrm{RVaR}_{\\alpha,\\kappa_a}(\\boldsymbol{x})``: Lower-tail RVaR with parameters ``(\\alpha, \\kappa_a)``.
  - ``\\mathrm{RVaR}_{\\beta,\\kappa_b}(-\\boldsymbol{x})``: Upper-tail RVaR with parameters ``(\\beta, \\kappa_b)``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        kappa_a::Number = 0.3,
        beta::Number = 0.05,
        kappa_b::Number = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativisticValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`, `0 < kappa_a < 1`.
  - `0 < beta < 1`, `0 < kappa_b < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativisticValueatRiskRange)(x::VecNum)

Computes the RVaR Range of a portfolio returns vector `x`.

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
"""
@concrete struct RelativisticValueatRiskRange <: RiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    "$(field_dict[:slv])"
    slv
    "$(field_dict[:alpha])"
    alpha
    "$(field_dict[:kappa_a])"
    kappa_a
    "$(field_dict[:beta])"
    beta
    "$(field_dict[:kappa_b])"
    kappa_b
    "$(field_dict[:oow])"
    w
    function RelativisticValueatRiskRange(settings::RiskMeasureSettings,
                                          slv::Option{<:Slv_VecSlv}, alpha::Number,
                                          kappa_a::Number, beta::Number, kappa_b::Number,
                                          w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa_a) < kappa_a < one(kappa_a))
        @argcheck(zero(beta) < beta < one(beta))
        @argcheck(zero(kappa_b) < kappa_b < one(kappa_b))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa_a),
                   typeof(beta), typeof(kappa_b), typeof(w)}(settings, slv, alpha, kappa_a,
                                                             beta, kappa_b, w)
    end
end
function RelativisticValueatRiskRange(;
                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                      slv::Option{<:Slv_VecSlv} = nothing,
                                      alpha::Number = 0.05, kappa_a::Number = 0.3,
                                      beta::Number = 0.05, kappa_b::Number = 0.3,
                                      w::Option{<:ObsWeights} = nothing)::RelativisticValueatRiskRange
    return RelativisticValueatRiskRange(settings, slv, alpha, kappa_a, beta, kappa_b, w)
end
function (r::RelativisticValueatRiskRange)(x::VecNum)
    return RRM(x, r.slv, r.alpha, r.kappa_a, r.w) + RRM(-x, r.slv, r.beta, r.kappa_b, r.w)
end
function factory(r::RelativisticValueatRiskRange, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...;
                 kwargs...)::RelativisticValueatRiskRange
    slv = solver_selector(r.slv, slv)
    return RelativisticValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                        kappa_a = r.kappa_a, beta = r.beta,
                                        kappa_b = r.kappa_b, slv = slv)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relativistic Drawdown-at-Risk (RDDaR) risk measure.

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

The Relativistic Drawdown-at-Risk is the RVaR of the drawdown series:

```math
\\begin{align}
\\mathrm{RDDaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\mathrm{RVaR}_{\\alpha,\\kappa}(\\boldsymbol{d}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RDDaR}_{\\alpha,\\kappa}(\\boldsymbol{x})``: Relativistic Drawdown-at-Risk.
  - $(math_dict[:alpha_rm])
  - ``\\kappa \\in (0,1)``: Tsallis deformation parameter.
  - ``\\boldsymbol{d}(\\boldsymbol{x})``: Absolute drawdown series vector ``T \\times 1``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativisticDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.
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
"""
@concrete struct RelativisticDrawdownatRisk <: RiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    "$(field_dict[:slv])"
    slv
    "$(field_dict[:alpha])"
    alpha
    "$(field_dict[:kappa])"
    kappa
    "$(field_dict[:oow])"
    w
    function RelativisticDrawdownatRisk(settings, slv::Option{<:Slv_VecSlv}, alpha::Number,
                                        kappa::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa) < kappa < one(kappa))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa), typeof(w)}(settings,
                                                                                           slv,
                                                                                           alpha,
                                                                                           kappa,
                                                                                           w)
    end
end
function RelativisticDrawdownatRisk(; settings = RiskMeasureSettings(),
                                    slv::Option{<:Slv_VecSlv} = nothing,
                                    alpha::Number = 0.05, kappa::Number = 0.3,
                                    w::Option{<:ObsWeights} = nothing)::RelativisticDrawdownatRisk
    return RelativisticDrawdownatRisk(settings, slv, alpha, kappa, w)
end
function (r::RelativisticDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return RRM(dd, r.slv, r.alpha, r.kappa, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Relativistic Drawdown-at-Risk (Relative RDDaR) risk measure for hierarchical optimisation.

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

The Relative Relativistic Drawdown-at-Risk is the RVaR of the relative drawdown series:

```math
\\begin{align}
\\mathrm{RRDDaR}_{\\alpha,\\kappa}(\\boldsymbol{x}) &= \\mathrm{RVaR}_{\\alpha,\\kappa}(\\boldsymbol{rd}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RRDDaR}_{\\alpha,\\kappa}(\\boldsymbol{x})``: Relative Relativistic Drawdown-at-Risk.
  - $(math_dict[:alpha_rm])
  - ``\\kappa \\in (0,1)``: Tsallis deformation parameter.
  - ``\\boldsymbol{rd}(\\boldsymbol{x})``: Relative drawdown series vector ``T \\times 1``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeRelativisticDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeRelativisticDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.
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
"""
@concrete struct RelativeRelativisticDrawdownatRisk <: HierarchicalRiskMeasure
    "$(field_dict[:settings_rm])"
    settings
    "$(field_dict[:slv])"
    slv
    "$(field_dict[:alpha])"
    alpha
    "$(field_dict[:kappa])"
    kappa
    "$(field_dict[:oow])"
    w
    function RelativeRelativisticDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                                slv::Option{<:Slv_VecSlv}, alpha::Number,
                                                kappa::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa) < kappa < one(kappa))
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
                                            alpha::Number = 0.05, kappa::Number = 0.3,
                                            w::Option{<:ObsWeights} = nothing)::RelativeRelativisticDrawdownatRisk
    return RelativeRelativisticDrawdownatRisk(settings, slv, alpha, kappa, w)
end
function (r::RelativeRelativisticDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return RRM(dd, r.slv, r.alpha, r.kappa, r.w)
end
for r in (RelativisticValueatRisk, RelativisticDrawdownatRisk,
          RelativeRelativisticDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv},
                              args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, slv = slv, alpha = r.alpha,
                             kappa = r.kappa, w = w)
             end
             function factory(r::$(r), slv::Slv_VecSlv,
                              pr::Option{<:AbstractPriorResult} = nothing, args...;
                              kwargs...)
                 w = isnothing(pr) ? r.w : nothing_scalar_array_selector(r.w, pr.w)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, kappa = r.kappa,
                             slv = slv, w = w)
             end
         end)
end

export RelativisticValueatRisk, RelativisticValueatRiskRange, RelativisticDrawdownatRisk,
       RelativeRelativisticDrawdownatRisk
