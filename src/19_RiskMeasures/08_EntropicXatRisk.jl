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

  - ERM value (scalar).

# Related

  - [`EntropicValueatRisk`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
function ERM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
             w::Option{<:ObsWeights} = nothing)
    w = get_observation_weights(w, x)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
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

`EntropicValueatRisk` is a coherent risk measure based on the Chernoff bound. It is an upper bound for both CVaR and VaR and is computed by solving a conic optimisation problem via an external solver.

# Mathematical Definition

The EVaR is defined via the Chernoff bound as the tightest exponential upper bound on VaR and CVaR:

```math
\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) = \\inf_{z > 0} \\left\\{ z \\ln\\!\\left( \\frac{M_{L}(1/z)}{\\alpha} \\right) \\right\\}\\,,
```

where ``L_t = -x_t`` is the loss and ``M_L(u) = \\mathbb{E}[e^{uL}]`` is the moment-generating function. Computationally, it is solved via the conic programme:

```math
\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) = \\min_{t,\\, z \\geq 0,\\, \\boldsymbol{u}} \\left\\{ t - z \\ln(\\alpha T) \\;:\\; \\sum_{i=1}^{T} u_i \\leq z,\\; (-x_i - t,\\, z,\\, u_i) \\in K_{\\exp}\\; \\forall i \\right\\}\\,,
```

where ``K_{\\exp} = \\{(a, b, c) : b\\, e^{a/b} \\leq c,\\, b > 0\\}`` is the exponential cone.

# Fields

  - `settings`: Risk measure configuration.
  - `slv`: Solver or vector of solvers for the conic optimisation.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.

# Constructors

    EntropicValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> EntropicValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
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
"""
@concrete struct EntropicValueatRisk <: RiskMeasure
    settings
    slv
    alpha
    w
    function EntropicValueatRisk(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                                 alpha::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(w)}(settings, slv,
                                                                            alpha, w)
    end
end
function EntropicValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                             w::Option{<:ObsWeights} = nothing)::EntropicValueatRisk
    return EntropicValueatRisk(settings, slv, alpha, w)
end
function (r::EntropicValueatRisk)(x::VecNum)
    return ERM(x, r.slv, r.alpha, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Entropic Value-at-Risk Range (EVaR Range) risk measure.

`EntropicValueatRiskRange` computes the difference between the lower-tail EVaR (at level `alpha`) and the upper-tail EVaR (at level `beta`).

# Mathematical Definition

```math
\\mathrm{EVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) = \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x}) + \\mathrm{EVaR}_{\\beta}(-\\boldsymbol{x})\\,,
```

where ``\\mathrm{EVaR}_{\\alpha}(\\boldsymbol{x})`` captures the lower-tail entropic risk and ``\\mathrm{EVaR}_{\\beta}(-\\boldsymbol{x})`` captures the upper-tail entropic risk (gain).

# Fields

  - `settings`: Risk measure configuration.
  - `slv`: Solver or vector of solvers for the conic optimisation.
  - `alpha`: Significance level for the lower tail.
  - `beta`: Significance level for the upper tail.
  - `w`: Optional observation weights.

# Constructors

    EntropicValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        beta::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> EntropicValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`, `0 < beta < 1`.
  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`EntropicValueatRisk`](@ref)
"""
@concrete struct EntropicValueatRiskRange <: RiskMeasure
    settings
    slv
    alpha
    beta
    w
    function EntropicValueatRiskRange(settings::RiskMeasureSettings,
                                      slv::Option{<:Slv_VecSlv}, alpha::Number,
                                      beta::Number, w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                                          slv,
                                                                                          alpha,
                                                                                          beta,
                                                                                          w)
    end
end
function EntropicValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                  beta::Number = 0.05,
                                  w::Option{<:ObsWeights} = nothing)::EntropicValueatRiskRange
    return EntropicValueatRiskRange(settings, slv, alpha, beta, w)
end
function (r::EntropicValueatRiskRange)(x::VecNum)
    return ERM(x, r.slv, r.alpha, r.w) + ERM(-x, r.slv, r.beta, r.w)
end
function factory(r::EntropicValueatRiskRange, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...; kwargs...)::EntropicValueatRiskRange
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return EntropicValueatRiskRange(; settings = r.settings, slv = slv, alpha = r.alpha,
                                    beta = r.beta, w = w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Entropic Drawdown-at-Risk (EDaR) risk measure.

`EntropicDrawdownatRisk` applies the Entropic Value-at-Risk framework to the absolute drawdown series of portfolio returns. It is a coherent risk measure providing an upper bound on both the Drawdown-at-Risk and Conditional Drawdown-at-Risk.

# Mathematical Definition

Define the absolute drawdown series:

```math
c_t = \\sum_{s=1}^{t} x_s\\,, \\qquad d_t = c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
```

The EDaR is the EVaR of the drawdown series:

```math
\\mathrm{EDaR}_{\\alpha}(\\boldsymbol{x}) = \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{d}(\\boldsymbol{x}))\\,.
```

# Fields

  - `settings`: Risk measure configuration.
  - `slv`: Solver or vector of solvers for the conic optimisation.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.

# Constructors

    EntropicDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> EntropicDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
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
"""
@concrete struct EntropicDrawdownatRisk <: RiskMeasure
    settings
    slv
    alpha
    w
    function EntropicDrawdownatRisk(settings::RiskMeasureSettings,
                                    slv::Option{<:Slv_VecSlv}, alpha::Number,
                                    w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(w)}(settings, slv,
                                                                            alpha, w)
    end
end
function EntropicDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                w::Option{<:ObsWeights} = nothing)::EntropicDrawdownatRisk
    return EntropicDrawdownatRisk(settings, slv, alpha, w)
end
function (r::EntropicDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return ERM(dd, r.slv, r.alpha, r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Entropic Drawdown-at-Risk (Relative EDaR) risk measure for hierarchical optimisation.

`RelativeEntropicDrawdownatRisk` applies the Entropic Value-at-Risk framework to the relative (compounded) drawdown series of portfolio returns.

# Mathematical Definition

Define the compounded wealth process and relative drawdown series:

```math
C_t = \\prod_{s=1}^{t} (1 + x_s)\\,, \\qquad rd_t = \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1 \\leq 0\\,.
```

The Relative EDaR is the EVaR of the relative drawdown series:

```math
\\mathrm{REDaR}_{\\alpha}(\\boldsymbol{x}) = \\mathrm{EVaR}_{\\alpha}(\\boldsymbol{rd}(\\boldsymbol{x}))\\,.
```

# Fields

  - `settings`: Hierarchical risk measure configuration.
  - `slv`: Solver or vector of solvers for the conic optimisation.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.

# Constructors

    RelativeEntropicDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeEntropicDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
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
"""
@concrete struct RelativeEntropicDrawdownatRisk <: HierarchicalRiskMeasure
    settings
    slv
    alpha
    w
    function RelativeEntropicDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                            slv::Option{<:Slv_VecSlv}, alpha::Number,
                                            w::Option{<:ObsWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(w)}(settings, slv,
                                                                            alpha, w)
    end
end
function RelativeEntropicDrawdownatRisk(;
                                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                        slv::Option{<:Slv_VecSlv} = nothing,
                                        alpha::Number = 0.05,
                                        w::Option{<:ObsWeights} = nothing)::RelativeEntropicDrawdownatRisk
    return RelativeEntropicDrawdownatRisk(settings, slv, alpha, w)
end
function (r::RelativeEntropicDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return ERM(dd, r.slv, r.alpha, r.w)
end
for r in (EntropicValueatRisk, EntropicDrawdownatRisk, RelativeEntropicDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult,
                              slv::Option{<:Slv_VecSlv} = nothing, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, slv = slv, alpha = r.alpha, w = w)
             end
             function factory(r::$(r), slv::Slv_VecSlv,
                              pr::Option{<:AbstractPriorResult} = nothing, args...;
                              kwargs...)
                 w = isnothing(pr) ? r.w : nothing_scalar_array_selector(r.w, pr.w)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, slv = slv, alpha = r.alpha, w = w)
             end
         end)
end

export EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
       RelativeEntropicDrawdownatRisk
