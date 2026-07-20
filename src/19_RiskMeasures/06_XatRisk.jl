"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Value-at-Risk formulation algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing the formulation for computing Value-at-Risk (e.g., mixed-integer programming, distribution-based) should be subtypes of `ValueatRiskFormulation`.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`ValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
"""
abstract type ValueatRiskFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the Value-at-Risk formulation `alg` unchanged.

Identity pass-through for formulation types that do not depend on prior results.

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::ValueatRiskFormulation, args...; kwargs...)::ValueatRiskFormulation
    return alg
end
"""
    port_opt_view(r, args...)

Get a view or subset of a Value-at-Risk formulation for slicing.

Returns the formulation unchanged (for non-distribution types) or sliced (for distribution-based types). Used internally in hierarchical optimisation.

# Arguments

  - `r`: Value-at-Risk formulation.
  - `args...`: Additional arguments (index, etc.).

# Returns

  - Sliced or unchanged formulation.

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`DistributionValueatRisk`](@ref)
"""
function port_opt_view(r::ValueatRiskFormulation, ::Any, args...)::ValueatRiskFormulation
    return r
end
"""
$(DocStringExtensions.TYPEDEF)

Mixed-integer programming (MIP) formulation for Value-at-Risk.

`MIPValueatRisk` specifies bounds used in the binary variable formulation of Value-at-Risk within a `JuMP` optimisation model.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MIPValueatRisk(;
        b::Option{<:Number} = nothing,
        s::Option{<:Number} = nothing
    ) -> MIPValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `b` is not `nothing`: `b > 0`.
  - If `s` is not `nothing`: `s > 0`.
  - If both are not `nothing`: `b > s`.

# Examples

```jldoctest
julia> MIPValueatRisk()
MIPValueatRisk
  b ┼ nothing
  s ┴ nothing
```

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`ValueatRisk`](@ref)
  - [`Option`](@ref)
"""
@concrete struct MIPValueatRisk <: ValueatRiskFormulation
    """
    $(field_dict[:b_mip])
    """
    b
    """
    $(field_dict[:s_mip])
    """
    s
    function MIPValueatRisk(b::Option{<:Number}, s::Option{<:Number})
        bflag = !isnothing(b)
        sflag = !isnothing(s)
        if bflag
            assert_nonempty_gt0_finite_val(b, :b)
        end
        if sflag
            assert_nonempty_gt0_finite_val(s, :s)
        end
        if bflag && sflag
            @argcheck(b > s, DomainError("b must be greater than s, got b = $b, s = $s"))
        end
        return new{typeof(b), typeof(s)}(b, s)
    end
end
function MIPValueatRisk(; b::Option{<:Number} = nothing,
                        s::Option{<:Number} = nothing)::MIPValueatRisk
    return MIPValueatRisk(b, s)
end
"""
$(DocStringExtensions.TYPEDEF)

Distribution-based formulation for Value-at-Risk.

`DistributionValueatRisk` specifies a parametric distribution for computing Value-at-Risk analytically. The distribution parameters can be overridden by prior results during optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistributionValueatRisk(;
        mu::Option{<:VecNum} = nothing,
        sigma::Option{<:MatNum} = nothing,
        chol::Option{<:MatNum} = nothing,
        dist::Distributions.Distribution = Distributions.Normal()
    ) -> DistributionValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is not `nothing`: `!isempty(mu)`.
  - If `sigma` is not `nothing`: `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.
  - If `chol` is not `nothing`: `!isempty(chol)`.

# Examples

```jldoctest
julia> DistributionValueatRisk()
DistributionValueatRisk
     mu ┼ nothing
  sigma ┼ nothing
   chol ┼ nothing
   dist ┴ Distributions.Normal{Float64}: Distributions.Normal{Float64}(μ=0.0, σ=1.0)
```

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`ValueatRisk`](@ref)
  - [`Option`](@ref)
"""
@propagatable @concrete struct DistributionValueatRisk <: ValueatRiskFormulation
    """
    $(field_dict[:mu_rm])
    """
    @pprop mu
    """
    $(field_dict[:sigma])
    """
    @pprop sigma
    """
    $(field_dict[:chol])
    """
    @pprop chol
    """
    $(field_dict[:dist])
    """
    dist
    function DistributionValueatRisk(mu::Option{<:VecNum}, sigma::Option{<:MatNum},
                                     chol::Option{<:MatNum},
                                     dist::Distributions.Distribution)
        if !isnothing(mu)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        end
        if !isnothing(sigma)
            @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
            assert_matrix_issquare(sigma, :sigma)
        end
        if !isnothing(chol)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
        end
        return new{typeof(mu), typeof(sigma), typeof(chol), typeof(dist)}(mu, sigma, chol,
                                                                          dist)
    end
end
function DistributionValueatRisk(; mu::Option{<:VecNum} = nothing,
                                 sigma::Option{<:MatNum} = nothing,
                                 chol::Option{<:MatNum} = nothing,
                                 dist::Distributions.Distribution = Distributions.Normal())::DistributionValueatRisk
    return DistributionValueatRisk(mu, sigma, chol, dist)
end
function port_opt_view(alg::DistributionValueatRisk, i, args...)::DistributionValueatRisk
    mu = nothing_scalar_array_view(alg.mu, i)
    sigma = nothing_scalar_array_view(alg.sigma, i)
    chol = isnothing(alg.chol) ? nothing : view(alg.chol, :, i)
    return DistributionValueatRisk(; mu = mu, sigma = sigma, chol = chol, dist = alg.dist)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Value-at-Risk (VaR) risk measure.

`ValueatRisk` quantifies the maximum expected loss at a given confidence level `alpha` over a specified time horizon. It can be computed using empirical quantiles (weighted or unweighted) or via a parametric distribution.

# Mathematical definition

Let ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal`` be the portfolio returns vector and ``x_{(k)}`` the ``k``-th order statistic (``k``-th smallest value). The empirical VaR at significance level ``\\alpha`` is:

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) &= -x_{(\\lceil \\alpha T \\rceil)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Value-at-Risk at significance level ``\\alpha``.
  - ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal``: Portfolio returns vector.
  - ``x_{(k)}``: ``k``-th order statistic (``k``-th smallest value) of ``\\boldsymbol{x}``.
  - ``\\alpha``: Significance level (e.g., ``\\alpha = 0.05`` for 95% VaR).
  - $(math_dict[:T])

For observation-weighted samples with weight vector ``\\boldsymbol{w}`` summing to ``S_w``, VaR is the ``\\alpha S_w``-quantile of the weighted empirical distribution.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing,
        alg::ValueatRiskFormulation = MIPValueatRisk()
    ) -> ValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ValueatRisk)(x::VecNum)

Computes the Value-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ValueatRisk()
ValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┼ nothing
       alg ┼ MIPValueatRisk
           │   b ┼ nothing
           │   s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
"""
@propagatable @concrete struct ValueatRisk <: RiskMeasure
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
    """
    $(field_dict[:alg])
    """
    @fprop @vprop alg
    function ValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                         w::Option{<:ObsWeights}, alg::ValueatRiskFormulation)
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings, alpha,
                                                                            w, alg)
    end
end
function ValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing,
                     alg::ValueatRiskFormulation = MIPValueatRisk())::ValueatRisk
    return ValueatRisk(settings, alpha, w, alg)
end
function (r::ValueatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    return -partialsort(x, ceil(Int, r.alpha * length(x)))
end
function (r::ValueatRisk{<:Any, <:Any, <:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    return -sorted_x[idx]
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Value-at-Risk Range risk measure.

`ValueatRiskRange` computes the difference between the lower-tail Value-at-Risk (at level `alpha`) and the upper-tail Value-at-Risk (at level `beta`), measuring the spread between downside and upside tail risks.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) &= \\mathrm{VaR}_{\\alpha}(\\boldsymbol{x}) - \\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})\\,,\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaRRange}_{\\alpha,\\beta}(\\boldsymbol{x})``: Value-at-Risk Range.
  - ``\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})``: Lower-tail loss quantile.
  - ``\\mathrm{VaR}_{\\beta}(-\\boldsymbol{x})``: Upper-tail gain quantile.
  - ``\\boldsymbol{x}``: Portfolio returns vector.
  - ``\\alpha``: Lower-tail significance level.
  - ``\\beta``: Upper-tail significance level.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        beta::Number = 0.05,
        w::Option{<:ObsWeights} = nothing,
        alg::ValueatRiskFormulation = MIPValueatRisk()
    ) -> ValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < beta < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ValueatRiskRange)(x::VecNum)

Computes the VaR Range of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ValueatRiskRange()
ValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
      beta ┼ Float64: 0.05
         w ┼ nothing
       alg ┼ MIPValueatRisk
           │   b ┼ nothing
           │   s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ValueatRisk`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
"""
@propagatable @concrete struct ValueatRiskRange <: RiskMeasure
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
    """
    $(field_dict[:alg])
    """
    @fprop @vprop alg
    function ValueatRiskRange(settings::RiskMeasureSettings, alpha::Number, beta::Number,
                              w::Option{<:ObsWeights}, alg::ValueatRiskFormulation)
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(beta, :beta)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w), typeof(alg)}(settings,
                                                                                          alpha,
                                                                                          beta,
                                                                                          w,
                                                                                          alg)
    end
end
function ValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          alpha::Number = 0.05, beta::Number = 0.05,
                          w::Option{<:ObsWeights} = nothing,
                          alg::ValueatRiskFormulation = MIPValueatRisk())::ValueatRiskRange
    return ValueatRiskRange(settings, alpha, beta, w, alg)
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, Nothing})(x::VecNum)
    x = copy(x)
    loss = -partialsort!(x, ceil(Int, r.alpha * length(x)))
    gain = -partialsort!(x, ceil(Int, r.beta * length(x)); rev = true)
    return loss - gain
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    loss = -sorted_x[idx]

    sorted_x = reverse!(sorted_x)
    sorted_w = reverse!(sorted_w)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, sw * r.beta)
    idx = ifelse(idx > length(x), idx - 1, idx)
    gain = -sorted_x[idx]
    return loss - gain
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Drawdown-at-Risk (DaR) risk measure.

`DrawdownatRisk` quantifies the maximum drawdown not exceeded at a given confidence level `alpha`. It operates on absolute drawdowns computed from the portfolio returns series.

# Mathematical definition

Define the cumulative wealth process and absolute drawdown at time ``t``:

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

The Drawdown-at-Risk at level ``\\alpha`` is the ``\\lceil \\alpha T \\rceil``-th smallest (most extreme) drawdown:

```math
\\begin{align}
\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x}) &= -d_{(\\lceil \\alpha T \\rceil)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DaR}_{\\alpha}(\\boldsymbol{x})``: Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:dtdd])
  - ``d_{(k)}``: ``k``-th order statistic (sorted ascending) of the drawdown series.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing,
        b::Option{<:Number} = nothing,
        s::Option{<:Number} = nothing
    ) -> DrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.
  - If `b` is not `nothing`: `b > 0`.
  - If `s` is not `nothing`: `s > 0`.
  - If both `b` and `s` are not `nothing`: `b > s`.

# Functor

    (r::DrawdownatRisk)(x::VecNum)

Computes the Drawdown-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DrawdownatRisk()
DrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┼ nothing
         b ┼ nothing
         s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`RelativeDrawdownatRisk`](@ref)
"""
@propagatable @concrete struct DrawdownatRisk <: RiskMeasure
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
    """
    $(field_dict[:b_mip])
    """
    b
    """
    $(field_dict[:s_mip])
    """
    s
    function DrawdownatRisk(settings::RiskMeasureSettings, alpha::Number,
                            w::Option{<:ObsWeights}, b::Option{<:Number},
                            s::Option{<:Number})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        bflag = !isnothing(b)
        sflag = !isnothing(s)
        if bflag
            assert_nonempty_gt0_finite_val(b, :b)
        end
        if sflag
            assert_nonempty_gt0_finite_val(s, :s)
        end
        if bflag && sflag
            @argcheck(b > s, DomainError("b must be greater than s, got b = $b, s = $s"))
        end
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(b), typeof(s)}(settings,
                                                                                     alpha,
                                                                                     w, b,
                                                                                     s)
    end
end
function DrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing,
                        b::Option{<:Number} = nothing,
                        s::Option{<:Number} = nothing)::DrawdownatRisk
    return DrawdownatRisk(settings, alpha, w, b, s)
end
"""
    absolute_drawdown_vec(x::VecNum) -> Vector

Compute the absolute drawdown series for a single-asset return vector.

Each element of the result is the difference between the current cumulative return and its running maximum (always ≤ 0).

The running maximum starts at zero, so the drawdown is measured against the initial portfolio value rather than against the first observation.

`x` is read, never written: the accumulator and the running peak are carried in scalars, so any `AbstractVector` works — a column view, a range, an immutable array.

# Arguments

  - `x::VecNum`: Return series vector.

# Returns

  - `Vector`: Drawdown vector of the same length as `x`.

# Related

  - [`absolute_drawdown_arr`](@ref)
  - [`DrawdownatRisk`](@ref)
"""
function absolute_drawdown_vec(x::VecNum)
    T = eltype(x)
    dd = similar(x, T)
    cs = peak = zero(T)
    for (idx, xi) in pairs(x)
        cs += xi
        peak = ifelse(cs > peak, cs, peak)
        dd[idx] = cs - peak
    end
    return dd
end
function (r::DrawdownatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end
function (r::DrawdownatRisk{<:Any, <:Any, <:ObsWeights})(x::VecNum)
    dd = absolute_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    w = get_observation_weights(r.w, x)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(dd), idx - 1, idx)
    return -sorted_dd[idx]
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Drawdown-at-Risk risk measure for hierarchical optimisation.

`RelativeDrawdownatRisk` quantifies the maximum relative (compounded) drawdown not exceeded at a given confidence level `alpha`. It operates on relative drawdowns computed from the portfolio returns series.

# Mathematical definition

Define the compounded wealth process and relative drawdown at time ``t``:

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

The Relative Drawdown-at-Risk at level ``\\alpha`` is:

```math
\\begin{align}
\\mathrm{RLDaR}_{\\alpha}(\\boldsymbol{x}) &= -rd_{(\\lceil \\alpha T \\rceil)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLDaR}_{\\alpha}(\\boldsymbol{x})``: Relative Drawdown-at-Risk at level ``\\alpha``.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:rdt])
  - ``rd_{(k)}``: ``k``-th order statistic (sorted ascending) of the relative drawdown series.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativeDrawdownatRisk)(x::VecNum)

Computes the Relative Drawdown-at-Risk of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeDrawdownatRisk()
RelativeDrawdownatRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
"""
@propagatable @concrete struct RelativeDrawdownatRisk <: HierarchicalRiskMeasure
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
    function RelativeDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                    alpha::Number, w::Option{<:ObsWeights})
        assert_unit_interval(alpha, :alpha)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Number = 0.05,
                                w::Option{<:ObsWeights} = nothing)::RelativeDrawdownatRisk
    return RelativeDrawdownatRisk(settings, alpha, w)
end
"""
    relative_drawdown_vec(x)

Compute the relative drawdown vector for a vector of portfolio returns.

Returns the relative drawdown at each time step, computed as the current portfolio value relative to its running maximum.

The running maximum starts at one, so the drawdown is measured against the initial portfolio value rather than against the first observation.

`x` is read, never written: the compounding factor and the running peak are carried in scalars, so any `AbstractVector` works — a column view, a range, an immutable array.

# Arguments

  - `x`: Vector of portfolio returns.

# Returns

  - Relative drawdown vector.

# Related

  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_arr`](@ref)
"""
function relative_drawdown_vec(x::VecNum)
    T = eltype(x)
    dd = similar(x, T)
    cp = peak = one(T)
    for (idx, xi) in pairs(x)
        cp *= xi + one(T)
        peak = ifelse(cp > peak, cp, peak)
        dd[idx] = cp / peak - one(peak)
    end
    return dd
end
function (r::RelativeDrawdownatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    dd = relative_drawdown_vec(x)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end
function (r::RelativeDrawdownatRisk{<:Any, <:Any, <:ObsWeights})(x::VecNum)
    dd = relative_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    w = get_observation_weights(r.w, x)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(dd), idx - 1, idx)
    return -sorted_dd[idx]
end

"""
    const CholRM = Union{<:Variance, <:StandardDeviation, <:DistributionValueatRisk}

Union of risk measures that support Cholesky-factor-based computation.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`DistributionValueatRisk`](@ref)
"""
const CholRM = Union{<:Variance, <:StandardDeviation, <:DistributionValueatRisk}

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::ValueatRisk) = NetReturnsInput()
risk_input_kind(::ValueatRiskRange) = NetReturnsInput()
risk_input_kind(::DrawdownatRisk) = NetReturnsInput()
risk_input_kind(::RelativeDrawdownatRisk) = NetReturnsInput()

export MIPValueatRisk, DistributionValueatRisk, ValueatRisk, ValueatRiskRange,
       DrawdownatRisk, RelativeDrawdownatRisk
