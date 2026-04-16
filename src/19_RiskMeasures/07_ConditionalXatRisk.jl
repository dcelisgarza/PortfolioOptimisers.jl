"""
$(DocStringExtensions.TYPEDEF)

Represents the Conditional Value-at-Risk (CVaR) risk measure, also known as Expected Shortfall (ES).

`ConditionalValueatRisk` computes the expected loss given that the loss exceeds the Value-at-Risk at level `alpha`. It provides a coherent risk measure for tail risk quantification.

# Mathematical Definition

Let ``\\boldsymbol{x} = (x_1, \\ldots, x_T)^\\intercal`` be the portfolio returns vector. The CVaR (also known as Expected Shortfall) at level ``\\alpha`` is the expected loss in the worst ``\\alpha`` fraction of scenarios:

```math
\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) = \\min_{\\nu \\in \\mathbb{R}} \\left\\{ -\\nu + \\frac{1}{\\alpha T} \\sum_{t=1}^{T} \\max(-x_t - \\nu,\\, 0) \\right\\}\\,.
```

Equivalently, it is the expected loss conditional on exceeding the VaR:

```math
\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) = -\\mathbb{E}\\!\\left[x \\mid x \\leq -\\mathrm{VaR}_{\\alpha}(\\boldsymbol{x})\\right]\\,.
```

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.

# Constructors

    ConditionalValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ConditionalValueatRisk)(x::VecNum)

Computes the CVaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ConditionalValueatRisk()
ConditionalValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
"""
@concrete struct ConditionalValueatRisk <: RiskMeasure
    settings
    alpha
    w
    function ConditionalValueatRisk(settings::RiskMeasureSettings, alpha::Number,
                                    w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing)
    return ConditionalValueatRisk(settings, alpha, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Distributionally Robust Conditional Value-at-Risk (DR-CVaR) risk measure.

`DistributionallyRobustConditionalValueatRisk` is a robust variant of CVaR that accounts for distributional uncertainty using Wasserstein ambiguity sets. It provides robustness against model misspecification in the tails of the return distribution.

# Mathematical Definition

The DR-CVaR with Wasserstein ambiguity parameter ``l`` and radius ``r`` is a robust upper bound on CVaR under distributional uncertainty within a Wasserstein ball of radius ``r``:

```math
\\mathrm{DR\\text{-}CVaR}_{\\alpha, l, r}(\\boldsymbol{x}) = \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) + l \\cdot r\\,.
```

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `l`: Wasserstein ambiguity parameter (radius scale factor).
  - `r`: Wasserstein radius parameter.
  - `w`: Optional observation weights.

# Constructors

    DistributionallyRobustConditionalValueatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        l::Number = 1.0,
        r::Number = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalValueatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `l > 0`.
  - `r > 0`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::DistributionallyRobustConditionalValueatRisk)(x::VecNum)

Computes the DR-CVaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DistributionallyRobustConditionalValueatRisk()
DistributionallyRobustConditionalValueatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         l ┼ Float64: 1.0
         r ┼ Float64: 0.02
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
"""
@concrete struct DistributionallyRobustConditionalValueatRisk <: RiskMeasure
    settings
    alpha
    l
    r
    w
    function DistributionallyRobustConditionalValueatRisk(settings::RiskMeasureSettings,
                                                          alpha::Number, l::Number,
                                                          r::Number,
                                                          w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(l > zero(l))
        @argcheck(r > zero(r))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(l), typeof(r), typeof(w)}(settings,
                                                                                     alpha,
                                                                                     l, r,
                                                                                     w)
    end
end
function DistributionallyRobustConditionalValueatRisk(;
                                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                      alpha::Number = 0.05, l::Number = 1.0,
                                                      r::Number = 0.02,
                                                      w::Option{<:ObsWeights} = nothing)
    return DistributionallyRobustConditionalValueatRisk(settings, alpha, l, r, w)
end
"""
    const RMCVaR{T} = Union{...}

Parameterised union of [`ConditionalValueatRisk`](@ref) and [`DistributionallyRobustConditionalValueatRisk`](@ref) sharing the same observation-weight type parameter `T`.

Used for unified dispatch on CVaR computation methods.

# Related

  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
"""
const RMCVaR{T} = Union{<:ConditionalValueatRisk{<:Any, <:Any, T},
                        <:DistributionallyRobustConditionalValueatRisk{<:Any, <:Any, <:Any,
                                                                       <:Any, T}}
function (r::RMCVaR{Nothing})(x::VecNum)
    x = copy(x)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end
function (r::RMCVaR{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Conditional Value-at-Risk Range (CVaR Range) risk measure.

`ConditionalValueatRiskRange` computes the difference between the lower-tail CVaR (at level `alpha`) and the upper-tail CVaR (at level `beta`), measuring the spread between downside and upside expected tail risks.

# Mathematical Definition

```math
\\mathrm{CVaRRange}_{\\alpha,\\beta}(\\boldsymbol{x}) = \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x}) - \\mathrm{CVaR}_{\\beta}(-\\boldsymbol{x})\\,,
```

where ``\\mathrm{CVaR}_{\\alpha}(\\boldsymbol{x})`` captures the lower-tail expected shortfall and ``\\mathrm{CVaR}_{\\beta}(-\\boldsymbol{x})`` captures the upper-tail expected surplus (gain).

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `beta`: Significance level for the upper tail.
  - `w`: Optional observation weights.

# Constructors

    ConditionalValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        beta::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < beta < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ConditionalValueatRiskRange)(x::VecNum)

Computes the CVaR Range of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ConditionalValueatRiskRange()
ConditionalValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
      beta ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
"""
@concrete struct ConditionalValueatRiskRange <: RiskMeasure
    settings
    alpha
    beta
    w
    function ConditionalValueatRiskRange(settings::RiskMeasureSettings, alpha::Number,
                                         beta::Number, w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                             alpha, beta, w)
    end
end
function ConditionalValueatRiskRange(;
                                     settings::RiskMeasureSettings = RiskMeasureSettings(),
                                     alpha::Number = 0.05, beta::Number = 0.05,
                                     w::Option{<:ObsWeights} = nothing)
    return ConditionalValueatRiskRange(settings, alpha, beta, w)
end
function factory(r::ConditionalValueatRiskRange, pr::AbstractPriorResult, args...;
                 kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return ConditionalValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                       beta = r.beta, w = w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Distributionally Robust Conditional Value-at-Risk Range (DR-CVaR Range) risk measure.

`DistributionallyRobustConditionalValueatRiskRange` computes the difference between the lower-tail DR-CVaR (at level `alpha`) and the upper-tail DR-CVaR (at level `beta`), with separate Wasserstein ambiguity parameters for each tail.

# Mathematical Definition

```math
\\mathrm{DR\\text{-}CVaRRange}(\\boldsymbol{x}) = \\mathrm{DR\\text{-}CVaR}_{\\alpha, l_a, r_a}(\\boldsymbol{x}) - \\mathrm{DR\\text{-}CVaR}_{\\beta, l_b, r_b}(-\\boldsymbol{x})\\,,
```

where each DR-CVaR uses its own Wasserstein ambiguity parameters ``(l_a, r_a)`` and ``(l_b, r_b)`` respectively.

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `l_a`: Wasserstein ambiguity scale factor for the lower tail.
  - `r_a`: Wasserstein radius for the lower tail.
  - `beta`: Significance level for the upper tail.
  - `l_b`: Wasserstein ambiguity scale factor for the upper tail.
  - `r_b`: Wasserstein radius for the upper tail.
  - `w`: Optional observation weights.

# Constructors

    DistributionallyRobustConditionalValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        l_a::Number = 1.0,
        r_a::Number = 0.02,
        beta::Number = 0.05,
        l_b::Number = 1.0,
        r_b::Number = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalValueatRiskRange

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < beta < 1`.
  - `l_a > 0`, `r_a > 0`, `l_b > 0`, `r_b > 0`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::DistributionallyRobustConditionalValueatRiskRange)(x::VecNum)

Computes the DR-CVaR Range of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DistributionallyRobustConditionalValueatRiskRange()
DistributionallyRobustConditionalValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
       l_a ┼ Float64: 1.0
       r_a ┼ Float64: 0.02
      beta ┼ Float64: 0.05
       l_b ┼ Float64: 1.0
       r_b ┼ Float64: 0.02
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
"""
@concrete struct DistributionallyRobustConditionalValueatRiskRange <: RiskMeasure
    settings
    alpha
    l_a
    r_a
    beta
    l_b
    r_b
    w
    function DistributionallyRobustConditionalValueatRiskRange(settings::RiskMeasureSettings,
                                                               alpha::Number, l_a::Number,
                                                               r_a::Number, beta::Number,
                                                               l_b::Number, r_b::Number,
                                                               w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        @argcheck(l_a > zero(l_a))
        @argcheck(r_a > zero(r_a))
        @argcheck(l_b > zero(l_b))
        @argcheck(r_b > zero(r_b))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(l_a), typeof(r_a), typeof(beta),
                   typeof(l_b), typeof(r_b), typeof(w)}(settings, alpha, l_a, r_a, beta,
                                                        l_b, r_b, w)
    end
end
function DistributionallyRobustConditionalValueatRiskRange(;
                                                           settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                           alpha::Number = 0.05,
                                                           l_a::Number = 1.0,
                                                           r_a::Number = 0.02,
                                                           beta::Number = 0.05,
                                                           l_b::Number = 1.0,
                                                           r_b::Number = 0.02,
                                                           w::Option{<:ObsWeights} = nothing)
    return DistributionallyRobustConditionalValueatRiskRange(settings, alpha, l_a, r_a,
                                                             beta, l_b, r_b, w)
end
function factory(r::DistributionallyRobustConditionalValueatRiskRange,
                 pr::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    return DistributionallyRobustConditionalValueatRiskRange(; settings = r.settings,
                                                             alpha = r.alpha, l_a = r.l_a,
                                                             r_a = r.r_a, beta = r.beta,
                                                             l_b = r.l_b, r_b = r.r_b,
                                                             w = w)
end
"""
    const RMCVaRRg{T} = Union{...}

Parameterised union of [`ConditionalValueatRiskRange`](@ref) and [`DistributionallyRobustConditionalValueatRiskRange`](@ref) sharing the same observation-weight type parameter `T`.

Used for unified dispatch on CVaR-range computation methods.

# Related

  - [`ConditionalValueatRiskRange`](@ref)
  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
"""
const RMCVaRRg{T} = Union{<:ConditionalValueatRiskRange{<:Any, <:Any, <:Any, T},
                          <:DistributionallyRobustConditionalValueatRiskRange{<:Any, <:Any,
                                                                              <:Any, <:Any,
                                                                              <:Any, <:Any,
                                                                              <:Any, T}}
function (r::RMCVaRRg{Nothing})(x::VecNum)
    x = copy(x)
    alpha = r.alpha
    aT = alpha * length(x)
    idx1 = ceil(Int, aT)
    var1 = -partialsort!(x, idx1)
    sum_var1 = zero(eltype(x))
    for i in 1:(idx1 - 1)
        sum_var1 += x[i] + var1
    end
    loss = var1 - sum_var1 / aT

    beta = r.beta
    bT = beta * length(x)
    idx2 = ceil(Int, bT)
    var2 = -partialsort!(x, idx2; rev = true)
    sum_var2 = zero(eltype(x))
    for i in 1:(idx2 - 1)
        sum_var2 += x[i] + var2
    end
    gain = var2 - sum_var2 / bT
    return loss - gain
end
function (r::RMCVaRRg{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    loss = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (alpha - cum_w[idx - 1])) / (alpha)
    end

    sorted_x = reverse!(sorted_x)
    sorted_w = reverse!(sorted_w)
    cum_w = cumsum(sorted_w)
    beta = sw * r.beta
    idx = searchsortedfirst(cum_w, beta)
    gain = if idx == 1
        -sorted_x[1]
    else
        idx = ifelse(idx > length(x), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_x[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_x[idx] * (beta - cum_w[idx - 1])) / (beta)
    end
    return loss - gain
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Conditional Drawdown-at-Risk (CDaR) risk measure, also known as Expected Maximum Drawdown.

`ConditionalDrawdownatRisk` computes the expected drawdown given that the drawdown exceeds the Drawdown-at-Risk at level `alpha`. It provides a coherent risk measure for drawdown tail risk.

# Mathematical Definition

Define the absolute drawdown series:

```math
c_t = \\sum_{s=1}^{t} x_s\\,, \\qquad d_t = c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
```

The CDaR is the CVaR of the drawdown series ``\\boldsymbol{d} = (d_1, \\ldots, d_T)^\\intercal``:

```math
\\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x}) = \\min_{\\nu \\in \\mathbb{R}} \\left\\{ -\\nu + \\frac{1}{\\alpha T} \\sum_{t=1}^{T} \\max(-d_t - \\nu,\\, 0) \\right\\}\\,.
```

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.

# Constructors

    ConditionalDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> ConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::ConditionalDrawdownatRisk)(x::VecNum)

Computes the CDaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> ConditionalDrawdownatRisk()
ConditionalDrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`RelativeConditionalDrawdownatRisk`](@ref)
"""
@concrete struct ConditionalDrawdownatRisk <: RiskMeasure
    settings
    alpha
    w
    function ConditionalDrawdownatRisk(settings::RiskMeasureSettings, alpha::Number,
                                       w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function ConditionalDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   alpha::Number = 0.05, w::Option{<:ObsWeights} = nothing)
    return ConditionalDrawdownatRisk(settings, alpha, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Distributionally Robust Conditional Drawdown-at-Risk (DR-CDaR) risk measure.

`DistributionallyRobustConditionalDrawdownatRisk` is a robust variant of CDaR that accounts for distributional uncertainty using Wasserstein ambiguity sets, applied to drawdown sequences.

# Mathematical Definition

```math
\\mathrm{DR\\text{-}CDaR}_{\\alpha, l, r}(\\boldsymbol{x}) = \\mathrm{CDaR}_{\\alpha}(\\boldsymbol{x}) + l \\cdot r\\,.
```

# Fields

  - `settings`: Risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `l`: Wasserstein ambiguity scale factor.
  - `r`: Wasserstein radius parameter.
  - `w`: Optional observation weights.

# Constructors

    DistributionallyRobustConditionalDrawdownatRisk(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        alpha::Number = 0.05,
        l::Number = 1.0,
        r::Number = 0.02,
        w::Option{<:ObsWeights} = nothing
    ) -> DistributionallyRobustConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `l > 0`.
  - `r > 0`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::DistributionallyRobustConditionalDrawdownatRisk)(x::VecNum)

Computes the DR-CDaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> DistributionallyRobustConditionalDrawdownatRisk()
DistributionallyRobustConditionalDrawdownatRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
     alpha ┼ Float64: 0.05
         l ┼ Float64: 1.0
         r ┼ Float64: 0.02
         w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
"""
@concrete struct DistributionallyRobustConditionalDrawdownatRisk <: RiskMeasure
    settings
    alpha
    l
    r
    w
    function DistributionallyRobustConditionalDrawdownatRisk(settings::RiskMeasureSettings,
                                                             alpha::Number, l::Number,
                                                             r::Number,
                                                             w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(l > zero(l))
        @argcheck(r > zero(r))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(l), typeof(r), typeof(w)}(settings,
                                                                                     alpha,
                                                                                     l, r,
                                                                                     w)
    end
end
function DistributionallyRobustConditionalDrawdownatRisk(;
                                                         settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                         alpha::Number = 0.05,
                                                         l::Number = 1.0, r::Number = 0.02,
                                                         w::Option{<:ObsWeights} = nothing)
    return DistributionallyRobustConditionalDrawdownatRisk(settings, alpha, l, r, w)
end
"""
    const RMCDaR{T} = Union{...}

Parameterised union of [`ConditionalDrawdownatRisk`](@ref) and [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) sharing the same observation-weight type parameter `T`.

Used for unified dispatch on CDaR computation methods.

# Related

  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
"""
const RMCDaR{T} = Union{<:ConditionalDrawdownatRisk{<:Any, <:Any, <:T},
                        <:DistributionallyRobustConditionalDrawdownatRisk{<:Any, <:Any,
                                                                          <:Any, <:Any,
                                                                          <:T}}
function (r::RMCDaR{Nothing})(x::VecNum)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    dd = absolute_drawdown_vec(x)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end
function (r::RMCDaR{<:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    dd = absolute_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_dd[1]
    else
        idx = ifelse(idx > length(dd), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_dd[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_dd[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the Relative Conditional Drawdown-at-Risk risk measure for hierarchical optimisation.

`RelativeConditionalDrawdownatRisk` computes the expected relative (compounded) drawdown given that the drawdown exceeds the Relative Drawdown-at-Risk at level `alpha`.

# Mathematical Definition

Define the compounded wealth process and relative drawdown series:

```math
C_t = \\prod_{s=1}^{t} (1 + x_s)\\,, \\qquad rd_t = \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1 \\leq 0\\,.
```

The Relative CDaR is the CVaR of the relative drawdown series ``\\boldsymbol{rd}``:

```math
\\mathrm{RCDaR}_{\\alpha}(\\boldsymbol{x}) = \\min_{\\nu \\in \\mathbb{R}} \\left\\{ -\\nu + \\frac{1}{\\alpha T} \\sum_{t=1}^{T} \\max(-rd_t - \\nu,\\, 0) \\right\\}\\,.
```

# Fields

  - `settings`: Hierarchical risk measure configuration.
  - `alpha`: Significance level for the lower tail.
  - `w`: Optional observation weights.

# Constructors

    RelativeConditionalDrawdownatRisk(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        alpha::Number = 0.05,
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeConditionalDrawdownatRisk

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `w` is not `nothing`: `!isempty(w)`.

# Functor

    (r::RelativeConditionalDrawdownatRisk)(x::VecNum)

Computes the Relative CDaR of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeConditionalDrawdownatRisk()
RelativeConditionalDrawdownatRisk
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
     alpha ┼ Float64: 0.05
         w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`RelativeDrawdownatRisk`](@ref)
"""
@concrete struct RelativeConditionalDrawdownatRisk <: HierarchicalRiskMeasure
    settings
    alpha
    w
    function RelativeConditionalDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                               alpha::Number, w::Option{<:ObsWeights})
        @argcheck(zero(alpha) < alpha < one(alpha))
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
    end
end
function RelativeConditionalDrawdownatRisk(;
                                           settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                           alpha::Number = 0.05,
                                           w::Option{<:ObsWeights} = nothing)
    return RelativeConditionalDrawdownatRisk(settings, alpha, w)
end
function (r::RelativeConditionalDrawdownatRisk{<:Any, <:Any, Nothing})(x::VecNum)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    dd = relative_drawdown_vec(x)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i in 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end
function (r::RelativeConditionalDrawdownatRisk{<:Any, <:Any, <:ObsWeights})(x::VecNum)
    w = get_observation_weights(r.w, x)
    sw = sum(w)
    dd = relative_drawdown_vec(x)
    order = sortperm(dd)
    sorted_dd = view(dd, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    alpha = sw * r.alpha
    idx = searchsortedfirst(cum_w, alpha)
    return if idx == 1
        -sorted_dd[1]
    else
        idx = ifelse(idx > length(dd), idx - 1, idx)
        -(LinearAlgebra.dot(sorted_dd[1:(idx - 1)], sorted_w[1:(idx - 1)]) +
          sorted_dd[idx] * (alpha - cum_w[idx - 1])) / alpha
    end
end
for r in (ConditionalValueatRisk, ConditionalDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 return $(r)(; settings = r.settings, alpha = r.alpha, w = w)
             end
         end)
end
for r in (DistributionallyRobustConditionalValueatRisk,
          DistributionallyRobustConditionalDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 return $(r)(; settings = r.settings, alpha = r.alpha, l = r.l, r = r.r,
                             w = w)
             end
         end)
end

export ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
       ConditionalValueatRiskRange, DistributionallyRobustConditionalValueatRiskRange,
       ConditionalDrawdownatRisk, DistributionallyRobustConditionalDrawdownatRisk,
       RelativeConditionalDrawdownatRisk
