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
function factory(alg::ValueatRiskFormulation, args...; kwargs...)
    return alg
end
function valueat_risk_formulation_view(r::ValueatRiskFormulation, args...)
    return r
end
"""
$(DocStringExtensions.TYPEDEF)

Mixed-integer programming (MIP) formulation for Value-at-Risk.

`MIPValueatRisk` specifies bounds used in the binary variable formulation of Value-at-Risk within a `JuMP` optimisation model.

# Fields

  - `b`: Optional big-M upper bound for the binary variable formulation. Must be positive and strictly greater than `s` if both are provided.
  - `s`: Optional small-M lower bound for the binary variable formulation. Must be positive and strictly less than `b` if both are provided.

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
    b
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
            @argcheck(b > s)
        end
        return new{typeof(b), typeof(s)}(b, s)
    end
end
function MIPValueatRisk(; b::Option{<:Number} = nothing, s::Option{<:Number} = nothing)
    return MIPValueatRisk(b, s)
end
"""
$(DocStringExtensions.TYPEDEF)

Distribution-based formulation for Value-at-Risk.

`DistributionValueatRisk` specifies a parametric distribution for computing Value-at-Risk analytically. The distribution parameters can be overridden by prior results during optimisation.

# Fields

  - `mu`: Optional expected returns vector. If `nothing`, uses prior result.
  - `sigma`: Optional covariance matrix. If `nothing`, uses prior result.
  - `chol`: Optional Cholesky factorisation of the covariance matrix. If `nothing`, uses prior result.
  - `dist`: Probability distribution to use for Value-at-Risk computation.

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
  dist ┴ Normal{Float64}(μ=0.0, σ=1.0)
```

# Related

  - [`ValueatRiskFormulation`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`ValueatRisk`](@ref)
  - [`Option`](@ref)
"""
@concrete struct DistributionValueatRisk <: ValueatRiskFormulation
    mu
    sigma
    chol
    dist
    function DistributionValueatRisk(mu::Option{<:VecNum}, sigma::Option{<:MatNum},
                                     chol::Option{<:MatNum},
                                     dist::Distributions.Distribution)
        if !isnothing(mu)
            @argcheck(!isempty(mu))
        end
        if !isnothing(sigma)
            @argcheck(!isempty(sigma))
            assert_matrix_issquare(sigma, :sigma)
        end
        if !isnothing(chol)
            @argcheck(!isempty(chol))
        end
        return new{typeof(mu), typeof(sigma), typeof(chol), typeof(dist)}(mu, sigma, chol,
                                                                          dist)
    end
end
function DistributionValueatRisk(; mu::Option{<:VecNum} = nothing,
                                 sigma::Option{<:MatNum} = nothing,
                                 chol::Option{<:MatNum} = nothing,
                                 dist::Distributions.Distribution = Distributions.Normal())
    return DistributionValueatRisk(mu, sigma, chol, dist)
end
function factory(alg::DistributionValueatRisk, pr::AbstractPriorResult, args...; kwargs...)
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    sigma = nothing_scalar_array_selector(alg.sigma, pr.sigma)
    chol = nothing_scalar_array_selector(alg.chol, pr.chol)
    return DistributionValueatRisk(; mu = mu, sigma = sigma, chol = chol, dist = alg.dist)
end
function valueat_risk_formulation_view(alg::DistributionValueatRisk, i)
    mu = nothing_scalar_array_view(alg.mu, i)
    sigma = nothing_scalar_array_view(alg.sigma, i)
    chol = isnothing(alg.chol) ? nothing : view(alg.chol, :, i)
    return DistributionValueatRisk(; mu = mu, sigma = sigma, chol = chol, dist = alg.dist)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Value-at-Risk (VaR) risk measure.

`ValueatRisk` quantifies the maximum expected loss at a given confidence level `alpha` over a specified time horizon. It can be computed using empirical quantiles (weighted or unweighted) or via a parametric distribution.

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail (e.g., `0.05` for 95% VaR).
  - `w`: Optional observation weights.
  - `alg`: Formulation algorithm for computing VaR.

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
       alg ┴ MIPValueatRisk
            b ┼ nothing
            s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
"""
@concrete struct ValueatRisk <: RiskMeasure
    settings
    alpha
    w
    alg
    function ValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                         w::Option{<:ObsWeights}, alg::ValueatRiskFormulation)
        @argcheck(zero(alpha) < alpha < one(alpha))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings, alpha,
                                                                            w, alg)
    end
end
function ValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing,
                     alg::ValueatRiskFormulation = MIPValueatRisk())
    return ValueatRisk(settings, alpha, w, alg)
end
function factory(r::ValueatRisk, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    alg = factory(r.alg, pr, args...; kwargs...)
    return ValueatRisk(; settings = r.settings, alpha = r.alpha, w = w, alg = alg)
end
function risk_measure_view(r::ValueatRisk, i, args...)
    alg = valueat_risk_formulation_view(r.alg, i)
    return ValueatRisk(; settings = r.settings, alpha = r.alpha, w = r.w, alg = alg)
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

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `beta`: Significance level for the upper tail.
  - `w`: Optional observation weights.
  - `alg`: Formulation algorithm for computing VaR.

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
       alg ┴ MIPValueatRisk
            b ┼ nothing
            s ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ValueatRisk`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
"""
@concrete struct ValueatRiskRange <: RiskMeasure
    settings
    alpha
    beta
    w
    alg
    function ValueatRiskRange(settings::RiskMeasureSettings, alpha::Number, beta::Number,
                              w::Option{<:ObsWeights}, alg::ValueatRiskFormulation)
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        validate_observation_weights(w)
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
                          alg::ValueatRiskFormulation = MIPValueatRisk())
    return ValueatRiskRange(settings, alpha, beta, w, alg)
end
function factory(r::ValueatRiskRange, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    alg = factory(r.alg, pr, args...; kwargs...)
    return ValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta, w = w,
                            alg = alg)
end
function risk_measure_view(r::ValueatRiskRange, i, args...)
    alg = valueat_risk_formulation_view(r.alg, i)
    return ValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta,
                            w = r.w, alg = alg)
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

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.
  - `b`: Optional big-M upper bound for the binary variable formulation in `JuMP` models.
  - `s`: Optional small-M lower bound for the binary variable formulation in `JuMP` models.

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
@concrete struct DrawdownatRisk <: RiskMeasure
    settings
    alpha
    w
    b
    s
    function DrawdownatRisk(settings::RiskMeasureSettings, alpha::Number,
                            w::Option{<:ObsWeights}, b::Option{<:Number},
                            s::Option{<:Number})
        @argcheck(zero(alpha) < alpha < one(alpha))
        validate_observation_weights(w)
        bflag = !isnothing(b)
        sflag = !isnothing(s)
        if bflag
            assert_nonempty_gt0_finite_val(b, :b)
        end
        if sflag
            assert_nonempty_gt0_finite_val(s, :s)
        end
        if bflag && sflag
            @argcheck(b > s)
        end
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(b), typeof(s)}(settings,
                                                                                     alpha,
                                                                                     w, b,
                                                                                     s)
    end
end
function DrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing,
                        b::Option{<:Number} = nothing, s::Option{<:Number} = nothing)
    return DrawdownatRisk(settings, alpha, w, b, s)
end
function factory(r::DrawdownatRisk, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return DrawdownatRisk(; settings = r.settings, alpha = r.alpha, w = w, b = r.b, s = r.s)
end
function absolute_drawdown_vec(x::VecNum)
    pushfirst!(x, zero(eltype(x)))
    cs = cumsum(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        peak = ifelse(i > peak, i, peak)
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
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

# Fields

  - `settings`: Hierarchical risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.

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
@concrete struct RelativeDrawdownatRisk <: HierarchicalRiskMeasure
    settings
    alpha
    w
    function RelativeDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                    alpha::Number, w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        validate_observation_weights(w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing)
    return RelativeDrawdownatRisk(settings, alpha, w)
end
function factory(r::RelativeDrawdownatRisk, pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return RelativeDrawdownatRisk(; settings = r.settings, alpha = r.alpha, w = w)
end
function relative_drawdown_vec(x::VecNum)
    pushfirst!(x, zero(eltype(x)))
    cs = cumprod(x .+ one(eltype(x)))
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        peak = ifelse(i > peak, i, peak)
        dd[idx] = i / peak - one(peak)
    end
    popfirst!(x)
    popfirst!(dd)
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

const CholRM = Union{<:Variance, <:StandardDeviation, <:DistributionValueatRisk}

export MIPValueatRisk, DistributionValueatRisk, ValueatRisk, ValueatRiskRange,
       DrawdownatRisk, RelativeDrawdownatRisk
