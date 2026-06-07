# https://github.com/oxfordcontrol/Clarabel.jl/blob/4915b83e0d900d978681d5e8f3a3a5b8e18086f0/warmstart_test/portfolioOpt/higherorderRiskMeansure.jl#L23
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the Power-Norm Risk Measure (PRM) for a vector of portfolio returns.

Solves a convex optimisation problem to compute the PRM at confidence level `alpha` with Lp-norm parameter `p`, using the specified solver(s).

# Arguments

  - `x`: Vector of portfolio returns.
  - `slv`: Solver or vector of solvers.
  - `alpha`: Confidence level (default `0.05`).
  - `p`: Lp-norm parameter (default `2.0`).
  - Additional parameters depending on the specific PRM formulation.
  - `kwargs...`: Additional keyword arguments passed to the solver.

# Returns

  - PRM value (scalar).

# Related

  - [`PowerNormValueatRisk`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
function PRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, p::Number = 2.0,
             w::Option{<:ObsWeights} = nothing)
    w = get_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
    end
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    T = length(x)
    ip = inv(p)
    JuMP.@variables(model, begin
                        pvar_eta
                        pvar_t
                        pvar_w[1:T] >= 0
                        pvar_v[1:T]
                    end)
    iaT = if isnothing(w)
        JuMP.@constraint(model, sum(pvar_v) - pvar_t <= 0)
        inv(alpha * T^ip)
    else
        JuMP.@constraint(model, LinearAlgebra.dot(w, pvar_v) - pvar_t <= 0)
        inv(alpha * sum(w)^ip)
    end
    JuMP.@constraints(model,
                      begin
                          (x + pvar_w) .+ pvar_eta >= 0
                          [i = 1:T],
                          [pvar_v[i], pvar_t, pvar_w[i]] in JuMP.MOI.PowerCone(ip)
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

Represents the Power Norm Value-at-Risk (PNVaR) risk measure.

`PowerNormValueatRisk` is a coherent risk measure that generalises EVaR by replacing the exponential moment-generating function with a power-norm. It is parametrised by a power ``p \\geq 1`` and a significance level ``\\alpha``, and is solved via a conic programme.

# Mathematical definition

The PNVaR at level ``\\alpha`` with power ``p`` is:

```math
\\begin{align}
\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x}) &= \\underset{\\eta,\\, t,\\, \\boldsymbol{w},\\, \\boldsymbol{v}}{\\min} \\left\\{ \\eta + \\frac{t}{\\alpha T^{1/p}} \\;:\\; \\boldsymbol{w} \\geq \\boldsymbol{0},\\; \\sum_{i=1}^{T} v_i \\leq t,\\; (x_i + w_i) + \\eta \\geq 0,\\; (v_i, t, w_i) \\in \\mathcal{K}_{\\mathrm{pow}}(1/p)\\; \\forall i \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{x})``: Power Norm Value-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``p \\geq 1``: Power parameter.
  - ``\\eta``, ``t``, ``\\boldsymbol{w}``, ``\\boldsymbol{v}``: Conic optimisation variables.
  - ``\\mathcal{K}_{\\mathrm{pow}}(p') = \\{(a,b,c) : a^{p'} b^{1-p'} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}``: Power cone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerNormValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        p::Number = 2.0,
        w::Option{<:ObsWeights} = nothing
    ) -> PowerNormValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `p >= 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::PowerNormValueatRisk)(x::VecNum)

Computes the PNVaR of a portfolio returns vector `x`.

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
  - [`EntropicValueatRisk`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`PowerNormValueatRiskRange`](@ref)
  - [`PowerNormDrawdownatRisk`](@ref)
"""
@concrete struct PowerNormValueatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:w_rm])
    """
    w
    function PowerNormValueatRisk(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                                  alpha::Number, p::Number,
                                  w::Option{<:StatsBase.AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p >= one(p))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function PowerNormValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                              p::Number = 2.0,
                              w::Option{<:ObsWeights} = nothing)::PowerNormValueatRisk
    return PowerNormValueatRisk(settings, slv, alpha, p, w)
end
function (r::PowerNormValueatRisk)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.p, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Power Norm Value-at-Risk Range (PNVaRRange) risk measure.

`PowerNormValueatRiskRange` computes the sum of the lower-tail PNVaR (at level `alpha` with power `pa`) and the upper-tail PNVaR (at level `beta` with power `pb`).

# Mathematical definition

```math
\\begin{align}
\\mathrm{PNVaRRange}_{\\alpha,p_a,\\beta,p_b}(\\boldsymbol{x}) &= \\mathrm{PNVaR}_{\\alpha,p_a}(\\boldsymbol{x}) + \\mathrm{PNVaR}_{\\beta,p_b}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PNVaRRange}_{\\alpha,p_a,\\beta,p_b}(\\boldsymbol{x})``: Power Norm VaR range.
  - $(math_dict[:xret])
  - ``\\mathrm{PNVaR}_{\\alpha,p_a}(\\boldsymbol{x})``: Lower-tail PNVaR with parameters ``(\\alpha, p_a)``.
  - ``\\mathrm{PNVaR}_{\\beta,p_b}(-\\boldsymbol{x})``: Upper-tail PNVaR with parameters ``(\\beta, p_b)``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerNormValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        beta::Number = 0.05,
        pa::Number = 2.0,
        pb::Number = 2.0,
        w::Option{<:ObsWeights} = nothing
    ) -> PowerNormValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`, `0 < beta < 1`.
  - `pa > 1`, `pb > 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::PowerNormValueatRiskRange)(x::VecNum)

Computes the PNVaR Range of a portfolio returns vector `x`.

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
"""
@concrete struct PowerNormValueatRiskRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    slv
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
    $(field_dict[:w_rm])
    """
    w
    function PowerNormValueatRiskRange(settings::RiskMeasureSettings,
                                       slv::Option{<:Slv_VecSlv}, alpha::Number,
                                       beta::Number, pa::Number, pb::Number,
                                       w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        @argcheck(pa > one(pa))
        @argcheck(pb > one(pb))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(beta), typeof(pa),
                   typeof(pb), typeof(w)}(settings, slv, alpha, beta, pa, pb, w)
    end
end
function PowerNormValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   slv::Option{<:Slv_VecSlv} = nothing,
                                   alpha::Number = 0.05, beta::Number = 0.05,
                                   pa::Number = 2.0, pb::Number = 2.0,
                                   w::Option{<:ObsWeights} = nothing)::PowerNormValueatRiskRange
    return PowerNormValueatRiskRange(settings, slv, alpha, beta, pa, pb, w)
end
function (r::PowerNormValueatRiskRange)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.pa, r.w) + PRM(-x, r.slv, r.beta, r.pb, r.w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`PowerNormValueatRiskRange`](@ref) by selecting observation weights and solver from the risk-measure instance or falling back to the prior result.

# Related

  - [`PowerNormValueatRiskRange`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
"""
function factory(r::PowerNormValueatRiskRange, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...; kwargs...)::PowerNormValueatRiskRange
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerNormValueatRiskRange(; settings = r.settings, slv = slv, alpha = r.alpha,
                                     beta = r.beta, pa = r.pa, pb = r.pb, w = w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Power Norm Drawdown-at-Risk (PNDDaR) risk measure.

`PowerNormDrawdownatRisk` applies the Power Norm Value-at-Risk framework to the absolute drawdown series of portfolio returns.

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
\\mathrm{PNDDaR}_{\\alpha,p}(\\boldsymbol{x}) &= \\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{d}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PNDDaR}_{\\alpha,p}(\\boldsymbol{x})``: Power Norm Drawdown-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - ``p \\geq 1``: Power parameter.
  - ``\\boldsymbol{d}(\\boldsymbol{x})``: Absolute drawdown series.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerNormDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        p::Number = 2.0,
        w::Option{<:ObsWeights} = nothing
    ) -> PowerNormDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `p >= 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::PowerNormDrawdownatRisk)(x::VecNum)

Computes the PNDDaR of a portfolio returns vector `x`.

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
"""
@concrete struct PowerNormDrawdownatRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:w_rm])
    """
    w
    function PowerNormDrawdownatRisk(settings::RiskMeasureSettings,
                                     slv::Option{<:Slv_VecSlv}, alpha::Number, p::Number,
                                     w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p >= one(p))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function PowerNormDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                 p::Number = 2.0,
                                 w::Option{<:ObsWeights} = nothing)::PowerNormDrawdownatRisk
    return PowerNormDrawdownatRisk(settings, slv, alpha, p, w)
end
function (r::PowerNormDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return PRM(dd, r.slv, r.alpha, r.p, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Power Norm Drawdown-at-Risk (Relative PNDDaR) risk measure for hierarchical optimisation.

`RelativePowerNormDrawdownatRisk` applies the Power Norm Value-at-Risk framework to the relative (compounded) drawdown series of portfolio returns.

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
\\mathrm{RPNDDaR}_{\\alpha,p}(\\boldsymbol{x}) &= \\mathrm{PNVaR}_{\\alpha,p}(\\boldsymbol{rd}(\\boldsymbol{x}))\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RPNDDaR}_{\\alpha,p}(\\boldsymbol{x})``: Relative Power Norm Drawdown-at-Risk.
  - $(math_dict[:xret])
  - $(math_dict[:alpha_rm])
  - ``p \\geq 1``: Power parameter.
  - ``\\boldsymbol{rd}(\\boldsymbol{x})``: Relative drawdown series.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativePowerNormDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        p::Number = 2.0,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativePowerNormDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `p >= 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativePowerNormDrawdownatRisk)(x::VecNum)

Computes the Relative PNDDaR of a portfolio returns vector `x`.

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
"""
@concrete struct RelativePowerNormDrawdownatRisk <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:alpha])
    """
    alpha
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:w_rm])
    """
    w
    function RelativePowerNormDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                             slv::Option{<:Slv_VecSlv}, alpha::Number,
                                             p::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p >= one(p))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function RelativePowerNormDrawdownatRisk(;
                                         settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                         slv::Option{<:Slv_VecSlv} = nothing,
                                         alpha::Number = 0.05, p::Number = 2.0,
                                         w::Option{<:ObsWeights} = nothing)::RelativePowerNormDrawdownatRisk
    return RelativePowerNormDrawdownatRisk(settings, slv, alpha, p, w)
end
function (r::RelativePowerNormDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return PRM(dd, r.slv, r.alpha, r.p, r.w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`PowerNormValueatRisk`](@ref) by selecting observation weights and solver from the risk-measure instance or falling back to the prior result.

# Related

  - [`PowerNormValueatRisk`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
"""
function factory(r::PowerNormValueatRisk, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv} = nothing, args...;
                 kwargs...)::PowerNormValueatRisk
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerNormValueatRisk(; settings = r.settings, slv = slv, alpha = r.alpha,
                                p = r.p, w = w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`PowerNormValueatRisk`](@ref) by overriding the solver and optionally selecting observation weights from the prior result.

# Related

  - [`PowerNormValueatRisk`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
"""
function factory(r::PowerNormValueatRisk, slv::Slv_VecSlv,
                 pr::Option{<:AbstractPriorResult} = nothing;
                 kwargs...)::PowerNormValueatRisk
    w = isnothing(pr) ? r.w : nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerNormValueatRisk(; settings = r.settings, alpha = r.alpha, p = r.p,
                                slv = slv, w = w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`PowerNormDrawdownatRisk`](@ref) by selecting observation weights and solver from the risk-measure instance or falling back to the prior result.

# Related

  - [`PowerNormDrawdownatRisk`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
"""
function factory(r::PowerNormDrawdownatRisk, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv} = nothing, args...;
                 kwargs...)::PowerNormDrawdownatRisk
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerNormDrawdownatRisk(; settings = r.settings, slv = slv, alpha = r.alpha,
                                   p = r.p, w = w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`PowerNormDrawdownatRisk`](@ref) by overriding the solver and optionally selecting observation weights from the prior result.

# Related

  - [`PowerNormDrawdownatRisk`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
"""
function factory(r::PowerNormDrawdownatRisk, slv::Slv_VecSlv,
                 pr::Option{<:AbstractPriorResult} = nothing;
                 kwargs...)::PowerNormDrawdownatRisk
    w = isnothing(pr) ? r.w : nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerNormDrawdownatRisk(; settings = r.settings, alpha = r.alpha, p = r.p,
                                   slv = slv, w = w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RelativePowerNormDrawdownatRisk`](@ref) by selecting observation weights and solver from the risk-measure instance or falling back to the prior result.

# Related

  - [`RelativePowerNormDrawdownatRisk`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
"""
function factory(r::RelativePowerNormDrawdownatRisk, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv} = nothing, args...;
                 kwargs...)::RelativePowerNormDrawdownatRisk
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return RelativePowerNormDrawdownatRisk(; settings = r.settings, slv = slv,
                                           alpha = r.alpha, p = r.p, w = w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RelativePowerNormDrawdownatRisk`](@ref) by overriding the solver and optionally selecting observation weights from the prior result.

# Related

  - [`RelativePowerNormDrawdownatRisk`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
"""
function factory(r::RelativePowerNormDrawdownatRisk, slv::Slv_VecSlv,
                 pr::Option{<:AbstractPriorResult} = nothing;
                 kwargs...)::RelativePowerNormDrawdownatRisk
    w = isnothing(pr) ? r.w : nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return RelativePowerNormDrawdownatRisk(; settings = r.settings, alpha = r.alpha,
                                           p = r.p, slv = slv, w = w)
end

export PowerNormValueatRisk, PowerNormValueatRiskRange, PowerNormDrawdownatRisk,
       RelativePowerNormDrawdownatRisk
