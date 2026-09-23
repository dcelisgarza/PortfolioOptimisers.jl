"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the two confidence formulations of the confidence weighted mean reversion rule.

The two formulations differ in how the constraint on the return of the period reads the spread of the belief. [`VarianceUpdate`](@ref) reads its variance, and [`StandardDeviationUpdate`](@ref) reads its standard deviation. [`ConfidenceWeightedMeanReversion`](@ref) holds one of them in its `formulation` field.

# Interfaces

To implement a new formulation, subtype `AbstractConfidenceUpdate` and implement both methods below.

## `confidence_step`

  - `confidence_step(formulation::AbstractConfidenceUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real, eps::Real) -> Real`: The Lagrange multiplier ``\\lambda_{t+1}`` of the period. It is `0` when the constraint holds at the current belief.

### Arguments

  - `formulation`: The formulation.
  - `V`: ``V_t``, the variance of the belief along the price relative.
  - `phi`: ``\\phi``, the confidence quantile.
  - `xbar`: ``\\bar{x}_t``, the mean of the price relative weighted by the belief covariance.
  - `W`: ``W_t``, the sum of the price relative weighted by the belief covariance.
  - `M`: ``M_t``, the gross return of the allocation over the period.
  - `eps`: ``\\epsilon``, the reversion threshold.

### Returns

  - `lambda::Real`: The multiplier, non-negative.

## `confidence_gain`

  - `confidence_gain(formulation::AbstractConfidenceUpdate, lambda::Real, phi::Real, V::Real) -> Real`: The scale ``\\gamma_{t+1}`` of the squared price relative that the update adds to the inverse belief covariance.

### Arguments

  - `formulation`: The formulation.
  - `lambda`: ``\\lambda_{t+1}``, the multiplier that `confidence_step` returned.
  - `phi`: ``\\phi``, the confidence quantile.
  - `V`: ``V_t``, the variance of the belief along the price relative.

### Returns

  - `gain::Real`: The scale, non-negative.

# Examples

A formulation that never moves the belief keeps the start allocation.

```jldoctest
julia> struct PassiveUpdate <: PortfolioOptimisers.AbstractConfidenceUpdate end

julia> function PortfolioOptimisers.confidence_step(::PassiveUpdate, V::Real, phi::Real,
                                                    xbar::Real, W::Real, M::Real, eps::Real)
           return zero(V)
       end

julia> function PortfolioOptimisers.confidence_gain(::PassiveUpdate, lambda::Real, phi::Real,
                                                    V::Real)
           return zero(V)
       end

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 -0.1; -0.05 0.05]);

julia> alg = ConfidenceWeightedMeanReversion(; formulation = PassiveUpdate());

julia> optimise(OnlinePortfolioSelection(; alg = alg), rd).w
2-element Vector{Float64}:
 0.5
 0.5
```

# Related

  - [`VarianceUpdate`](@ref)
  - [`StandardDeviationUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
abstract type AbstractConfidenceUpdate <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects the confidence constraint that reads the variance of the belief along the price relative.

This is CWMR-Var, the first of the two formulations of the confidence weighted mean reversion rule.

# Mathematical definition

The constraint on the new belief ``(\\boldsymbol{w}, \\Sigma)`` is linear in ``\\Sigma``:

```math
\\begin{align}
\\epsilon &\\geq \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle + \\phi\\, \\boldsymbol{x}_t^\\intercal \\Sigma \\boldsymbol{x}_t\\,.
\\end{align}
```

The multiplier ``\\lambda_{t+1}`` is the largest non-negative root of ``a \\lambda^2 + b \\lambda + c = 0``, and ``0`` when no root is real and non-negative. The gain ``\\gamma_{t+1}`` scales the squared price relative in the update of the inverse covariance:

```math
\\begin{align}
a &= 2 \\phi V_t \\left( V_t - \\bar{x}_t W_t \\right)\\,,\\\\
b &= 2 \\phi V_t \\left( \\epsilon - M_t \\right) + V_t - \\bar{x}_t W_t\\,,\\\\
c &= \\epsilon - M_t - \\phi V_t\\,,\\\\
\\gamma_{t+1} &= 2 \\lambda_{t+1} \\phi\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:Sigma_t_cw])
  - $(math_dict[:M_t_cw])
  - $(math_dict[:V_t_cw])
  - $(math_dict[:W_t_cw])
  - $(math_dict[:xbar_t_cw])
  - $(math_dict[:lambda_cw])
  - $(math_dict[:gamma_cw])
  - $(math_dict[:phi_cw])
  - $(math_dict[:eps_cw])

The root makes the constraint hold with equality at the mean step and at the full rank-one update ``\\Sigma_{t+1}^{-1} = \\Sigma_t^{-1} + \\gamma_{t+1} \\boldsymbol{x}_t \\boldsymbol{x}_t^\\intercal``. The rule then keeps the diagonal of that update. By the Cauchy-Schwarz inequality ``V_t \\geq \\bar{x}_t W_t``, so ``a \\geq 0``. When the constraint holds at the current belief, ``c \\geq 0`` and ``b \\geq 0``, so no root is positive and ``\\lambda_{t+1} = 0``.

# Examples

```jldoctest
julia> VarianceUpdate()
VarianceUpdate()
```

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`StandardDeviationUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
  - [`confidence_step`](@ref)
  - [`confidence_gain`](@ref)

# References

  - $(ref_dict[:li2013cwmr]) Proposition 4.1 and Appendix A, Equation 11.
"""
struct VarianceUpdate <: AbstractConfidenceUpdate end
"""
$(DocStringExtensions.TYPEDEF)

Selects the confidence constraint that reads the standard deviation of the belief along the price relative.

This is CWMR-Stdev, the second of the two formulations of the confidence weighted mean reversion rule.

# Mathematical definition

The constraint on the new belief ``(\\boldsymbol{w}, \\Sigma)`` reads the standard deviation:

```math
\\begin{align}
\\epsilon &\\geq \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle + \\phi \\sqrt{\\boldsymbol{x}_t^\\intercal \\Sigma \\boldsymbol{x}_t}\\,.
\\end{align}
```

The multiplier ``\\lambda_{t+1}`` is the largest non-negative root of ``a \\lambda^2 + b \\lambda + c = 0``, and ``0`` when no root is real and non-negative. The gain ``\\gamma_{t+1}`` scales the squared price relative in the update of the inverse covariance:

```math
\\begin{align}
h &= V_t - \\bar{x}_t W_t + \\frac{\\phi^2 V_t}{2}\\,,\\\\
a &= h^2 - \\frac{\\phi^4 V_t^2}{4}\\,,\\\\
b &= 2 \\left( \\epsilon - M_t \\right) h\\,,\\\\
c &= \\left( \\epsilon - M_t \\right)^2 - \\phi^2 V_t\\,,\\\\
\\sqrt{U_t} &= \\frac{-\\lambda_{t+1} \\phi V_t + \\sqrt{\\lambda_{t+1}^2 \\phi^2 V_t^2 + 4 V_t}}{2}\\,,\\\\
\\gamma_{t+1} &= \\frac{\\lambda_{t+1} \\phi}{\\sqrt{U_t}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:Sigma_t_cw])
  - $(math_dict[:M_t_cw])
  - $(math_dict[:V_t_cw])
  - $(math_dict[:W_t_cw])
  - $(math_dict[:xbar_t_cw])
  - $(math_dict[:U_t_cw])
  - $(math_dict[:lambda_cw])
  - $(math_dict[:gamma_cw])
  - $(math_dict[:phi_cw])
  - $(math_dict[:eps_cw])
  - ``h``: Shared term of the coefficients.

The root makes the constraint hold with equality at the mean step and at the full rank-one update ``\\Sigma_{t+1}^{-1} = \\Sigma_t^{-1} + \\gamma_{t+1} \\boldsymbol{x}_t \\boldsymbol{x}_t^\\intercal``, and ``\\sqrt{U_t}`` is the standard deviation of that update along ``\\boldsymbol{x}_t``. The rule then keeps the diagonal of that update. The coefficient ``a = (V_t - \\bar{x}_t W_t)(V_t - \\bar{x}_t W_t + \\phi^2 V_t)`` is non-negative. When the constraint holds at the current belief, ``\\epsilon - M_t \\geq \\phi \\sqrt{V_t}``, so ``c \\geq 0`` and ``b \\geq 0``, no root is positive and ``\\lambda_{t+1} = 0``.

# Examples

```jldoctest
julia> StandardDeviationUpdate()
StandardDeviationUpdate()
```

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`VarianceUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
  - [`confidence_step`](@ref)
  - [`confidence_gain`](@ref)

# References

  - $(ref_dict[:li2013cwmr]) Proposition 4.2 and Appendix B, Equations 14 and 15.
"""
struct StandardDeviationUpdate <: AbstractConfidenceUpdate end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the largest non-negative root of ``a \\lambda^2 + b \\lambda + c = 0``, or `0` when no root is real and non-negative.

# Algorithm

 1. Set `z`, a zero of the type of `a / b`.
 2. When `a` is zero, the equation is the line ``b \\lambda + c = 0``. Return `z` when `b` is zero too, and otherwise the larger of `z` and `-c / b`.
 3. Compute the discriminant `disc = b^2 - 4 * a * c`. Return `z` when it is negative, because no root is real.
 4. Return the largest of `z` and the two roots `(-b ± sqrt(disc)) / (2 * a)`.

# Arguments

  - `a`: Coefficient of ``\\lambda^2``.
  - `b`: Coefficient of ``\\lambda``.
  - `c`: Constant term.

# Returns

  - `lambda::Real`: The root, non-negative.

# Related

  - [`confidence_step`](@ref)
"""
function nonneg_quadratic_root(a::Real, b::Real, c::Real)
    z = zero(a / b)
    if iszero(a)
        return iszero(b) ? z : max(z, -c / b)
    end
    disc = b^2 - 4 * a * c
    if disc < zero(disc)
        return z
    end
    s = sqrt(disc)
    return max(z, (-b + s) / (2 * a), (-b - s) / (2 * a))
end
"""
    confidence_step(::VarianceUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real, eps::Real)
    confidence_step(::StandardDeviationUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real, eps::Real)

Returns the Lagrange multiplier ``\\lambda_{t+1}`` of a confidence weighted step under the formulation.

Each method builds the coefficients of its formulation's quadratic in ``\\lambda`` and solves it with [`nonneg_quadratic_root`](@ref). [`VarianceUpdate`](@ref) and [`StandardDeviationUpdate`](@ref) state the coefficients.

# Arguments

  - `formulation`: The formulation, [`VarianceUpdate`](@ref) or [`StandardDeviationUpdate`](@ref).
  - `V`: ``V_t``, the variance of the belief along the price relative.
  - `phi`: ``\\phi``, the confidence quantile.
  - `xbar`: ``\\bar{x}_t``, the mean of the price relative weighted by the belief covariance.
  - `W`: ``W_t``, the sum of the price relative weighted by the belief covariance.
  - `M`: ``M_t``, the gross return of the allocation over the period.
  - `eps`: ``\\epsilon``, the reversion threshold.

# Returns

  - `lambda::Real`: The multiplier, non-negative. It is `0` when the constraint holds at the current belief.

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`nonneg_quadratic_root`](@ref)
  - [`confidence_gain`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
function confidence_step(::VarianceUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real,
                         eps::Real)
    a = 2 * phi * V^2 - 2 * phi * xbar * V * W
    b = 2 * phi * eps * V - 2 * phi * V * M + V - xbar * W
    c = eps - M - phi * V
    return nonneg_quadratic_root(a, b, c)
end
function confidence_step(::StandardDeviationUpdate, V::Real, phi::Real, xbar::Real, W::Real,
                         M::Real, eps::Real)
    h = V - xbar * W + phi^2 * V / 2
    a = h^2 - phi^4 * V^2 / 4
    b = 2 * (eps - M) * h
    c = (eps - M)^2 - phi^2 * V
    return nonneg_quadratic_root(a, b, c)
end
"""
    confidence_gain(::VarianceUpdate, lambda::Real, phi::Real, V::Real)
    confidence_gain(::StandardDeviationUpdate, lambda::Real, phi::Real, V::Real)

Returns the gain ``\\gamma_{t+1}``, the scale of the squared price relative in the update of the inverse belief covariance.

The gain is ``2 \\lambda_{t+1} \\phi`` under [`VarianceUpdate`](@ref), and ``\\lambda_{t+1} \\phi / \\sqrt{U_t}`` under [`StandardDeviationUpdate`](@ref), which states ``\\sqrt{U_t}``.

# Arguments

  - `formulation`: The formulation, [`VarianceUpdate`](@ref) or [`StandardDeviationUpdate`](@ref).
  - `lambda`: ``\\lambda_{t+1}``, the multiplier that [`confidence_step`](@ref) returned.
  - `phi`: ``\\phi``, the confidence quantile.
  - `V`: ``V_t``, the variance of the belief along the price relative. The variance formulation does not read it.

# Returns

  - `gain::Real`: The gain, non-negative.

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`confidence_step`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
function confidence_gain(::VarianceUpdate, lambda::Real, phi::Real, ::Real)
    return 2 * lambda * phi
end
function confidence_gain(::StandardDeviationUpdate, lambda::Real, phi::Real, V::Real)
    u = (-lambda * phi * V + sqrt(lambda^2 * phi^2 * V^2 + 4 * V)) / 2
    return lambda * phi / u
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the diagonal of the belief covariance over the weights, for the confidence weighted mean reversion rule.

[`ConfidenceWeightedMeanReversion`](@ref) seeds it and updates it once per period. The mean of the belief is the allocation held, so the carrier holds no mean.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ConfidenceWeightedMeanReversion`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct ConfidenceWeightedMeanReversionState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The diagonal of ``\\Sigma_t``, the belief covariance over the weights, with one entry per asset. The update writes it in place. The seed is ``1 / N^2`` per asset, and every step rescales it to the trace ``1 / N`` of the seed.
    """
    sigma
end
function merge_states(::ConfidenceWeightedMeanReversionState,
                      ::ConfidenceWeightedMeanReversionState)
    return throw(ArgumentError("a `ConfidenceWeightedMeanReversionState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::ConfidenceWeightedMeanReversionState)
    return ConfidenceWeightedMeanReversionState(x.n, copy(x.sigma))
end
function port_opt_view(x::ConfidenceWeightedMeanReversionState, i, args...)
    return ConfidenceWeightedMeanReversionState(x.n, x.sigma[i])
end
"""
$(DocStringExtensions.TYPEDEF)

Moves a Gaussian belief over the weights by the least relative entropy that puts the last return below `eps` with confidence.

This is the confidence weighted mean reversion (CWMR) of Li, Hoi, Zhao and Gopalkrishnan. The mean of the belief is the allocation held, and its covariance is the confidence in each weight. The covariance is the rule's own. It is a belief over weights, not a moment of returns, so it never comes from a Prior. The rule sells what rose in the last period, a bet on mean reversion over one period, like [`PassiveAggressiveMeanReversion`](@ref).

The Allocation Set constrains the mean alone. The paper projects the mean onto the simplex in the Euclidean distance, so the `proj` field is bound to [`EuclideanProjection`](@ref). The covariance is rescaled and never projected.

# Mathematical definition

The new belief solves

```math
\\begin{align}
\\left( \\boldsymbol{w}_{t+1}, \\Sigma_{t+1} \\right) &= \\underset{\\boldsymbol{w} \\in \\Delta_N,\\, \\Sigma}{\\arg\\min}\\; D_{\\mathrm{KL}}\\left( \\mathcal{N}(\\boldsymbol{w}, \\Sigma) \\,\\|\\, \\mathcal{N}(\\boldsymbol{w}_t, \\Sigma_t) \\right) \\quad \\mathrm{s.t.} \\quad \\Pr\\left[ \\langle \\boldsymbol{b}, \\boldsymbol{x}_t \\rangle \\leq \\epsilon \\right] \\geq \\theta\\,,\\quad \\boldsymbol{b} \\sim \\mathcal{N}(\\boldsymbol{w}, \\Sigma)\\,.
\\end{align}
```

The formulation turns the probability constraint into a constraint on the variance or on the standard deviation of the belief, and the solution keeps the diagonal of the covariance:

```math
\\begin{align}
\\tilde{\\boldsymbol{w}}_{t+1} &= \\boldsymbol{w}_t - \\lambda_{t+1} \\Sigma_t \\left( \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\right)\\,,\\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\Delta_N}{\\arg\\min}\\; \\left\\lVert \\boldsymbol{w} - \\tilde{\\boldsymbol{w}}_{t+1} \\right\\rVert_2^2\\,,\\\\
\\tilde{\\Sigma}_{t+1}^{-1} &= \\Sigma_t^{-1} + \\gamma_{t+1}\\, \\mathrm{diag}(\\boldsymbol{x}_t)^2\\,,\\\\
\\Sigma_{t+1} &= \\frac{\\tilde{\\Sigma}_{t+1}}{N \\mathrm{tr}(\\tilde{\\Sigma}_{t+1})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:Sigma_t_cw])
  - $(math_dict[:xbar_t_cw])
  - $(math_dict[:lambda_cw])
  - $(math_dict[:gamma_cw])
  - $(math_dict[:eps_cw])
  - $(math_dict[:N])
  - ``\\theta``: Confidence level of the constraint, with ``\\phi = \\Phi^{-1}(\\theta)``.
  - ``\\Delta_N``: Probability simplex over the ``N`` assets.
  - ``D_{\\mathrm{KL}}``: Relative entropy between two Gaussian distributions.
  - ``\\boldsymbol{b}``: Allocation drawn from the new belief.
  - ``\\tilde{\\boldsymbol{w}}_{t+1}``, ``\\tilde{\\Sigma}_{t+1}``: Mean and covariance before the projection and the rescale.

[`VarianceUpdate`](@ref) and [`StandardDeviationUpdate`](@ref) state ``\\lambda_{t+1}`` and ``\\gamma_{t+1}``. The multiplier is ``0`` when the constraint holds at the current belief, so the mean does not move on such a period. The mean moves each weight in proportion to its variance in the belief. The rescale keeps the trace of the covariance at ``1 / N``, the trace of the seed ``I / N^2``.

The 2013 text states the constraint on the return with ``\\epsilon = 0.5``, and this rule follows it. The 2011 text states it on the logarithm of the return with ``\\epsilon = -0.5`` and linearises the logarithm at the current mean. It also rescales the covariance to the trace ``1 / N^2``.

# Algorithm

 1. Read `sigma`, the diagonal of ``\\Sigma_t``, from the carrier `st`.
 2. Compute `M = dot(w, x)`, `V = dot(sigma, x .^ 2)`, `W = dot(sigma, x)` and `xbar = W / sum(sigma)`.
 3. Compute the multiplier `lam` with [`confidence_step`](@ref) under `alg.formulation`.
 4. Move the mean to `q = w .- lam .* sigma .* (x .- xbar)`.
 5. Compute the gain `gain` with [`confidence_gain`](@ref).
 6. Write `inv.(inv.(sigma) .+ gain .* x .^ 2)` into `sigma`.
 7. Divide `sigma` by `length(sigma) * sum(sigma)`, so its sum is ``1 / N``.
 8. Project `q` onto the Allocation Set with `alg.proj`, from the Price-Adjusted Allocation. Return the carrier with its period count raised by one, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConfidenceWeightedMeanReversion(;
        eps::Real = 0.5,
        phi::Real = 2,
        formulation::AbstractConfidenceUpdate = VarianceUpdate(),
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ConfidenceWeightedMeanReversion

Keywords correspond to the struct's fields. The defaults `eps = 0.5` and `phi = 2` are the values the 2013 paper sets without tuning. The paper reports that `phi` has little effect on the result.

## Validation

  - `eps >= 0`. A `DomainError` is thrown otherwise.
  - `phi >= 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ConfidenceWeightedMeanReversion()
ConfidenceWeightedMeanReversion
          eps ┼ Float64: 0.5
          phi ┼ Int64: 2
  formulation ┼ VarianceUpdate()
         proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`AbstractConfidenceUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversionState`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref): the other rule that bets on mean reversion over one period.
  - [`EuclideanProjection`](@ref)

# References

  - $(ref_dict[:li2011cwmr]) Algorithm 1.
  - $(ref_dict[:li2013cwmr]) Section 4, Algorithm 2.
"""
struct ConfidenceWeightedMeanReversion{T1 <: Real, T2 <: Real,
                                       T3 <: AbstractConfidenceUpdate,
                                       T4 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Reversion threshold ``\\epsilon``. The rule moves the belief only when the return of the last period, plus the confidence margin, exceeds it.
    """
    eps::T1
    """
    Confidence quantile ``\\phi = \\Phi^{-1}(\\theta)``. It scales the margin that the spread of the belief adds to the constraint.
    """
    phi::T2
    """
    Formulation of the constraint, [`VarianceUpdate`](@ref) for CWMR-Var or [`StandardDeviationUpdate`](@ref) for CWMR-Stdev.
    """
    formulation::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function ConfidenceWeightedMeanReversion(eps::Real, phi::Real,
                                             formulation::AbstractConfidenceUpdate,
                                             proj::EuclideanProjection)
        @argcheck(eps >= zero(eps), DomainError(eps, "eps must be non-negative"))
        @argcheck(phi >= zero(phi), DomainError(phi, "phi must be non-negative"))
        return new{typeof(eps), typeof(phi), typeof(formulation), typeof(proj)}(eps, phi,
                                                                                formulation,
                                                                                proj)
    end
end
function ConfidenceWeightedMeanReversion(; eps::Real = 0.5, phi::Real = 2,
                                         formulation::AbstractConfidenceUpdate = VarianceUpdate(),
                                         proj::EuclideanProjection = EuclideanProjection())::ConfidenceWeightedMeanReversion
    return ConfidenceWeightedMeanReversion(eps, phi, formulation, proj)
end
function rule_state_seed(::ConfidenceWeightedMeanReversion, w::AbstractVector)
    N = length(w)
    return ConfidenceWeightedMeanReversionState(0, fill(one(eltype(w)) / N^2, N))
end
function online_update!(alg::ConfidenceWeightedMeanReversion,
                        st::ConfidenceWeightedMeanReversionState, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    sigma = st.sigma
    M = LinearAlgebra.dot(w, x)
    V = LinearAlgebra.dot(sigma, x .^ 2)
    W = LinearAlgebra.dot(sigma, x)
    xbar = W / sum(sigma)
    lam = confidence_step(alg.formulation, V, alg.phi, xbar, W, M, alg.eps)
    q = w .- lam .* sigma .* (x .- xbar)
    gain = confidence_gain(alg.formulation, lam, alg.phi, V)
    sigma .= inv.(inv.(sigma) .+ gain .* x .^ 2)
    sigma ./= length(sigma) * sum(sigma)
    wn = project(alg.proj, set, q, price_adjusted_allocation(w, x))
    return ConfidenceWeightedMeanReversionState(st.n + 1, sigma), wn
end
"""
$(DocStringExtensions.TYPEDEF)

Moves wealth from an asset that grew more to one whose log returns follow its own one window later.

This is the anti-correlation rule (Anticor) of Borodin, El-Yaniv and Gogan. The asset that receives the wealth is expected to repeat the earlier growth of the asset that gives it. The paper's headline algorithm, ANTI₁, is the uniform buy-and-hold mixture of this rule over the windows 2 to 30, which is an [`ExpertMixture`](@ref) over those experts under its default weighting.

The rule recomputes the correlation from the rows the head holds at every period and carries nothing, so it reads the last ``2 \\ell`` rows. Until the head holds ``2 \\ell`` rows, the rule returns the wealth held, which is buy-and-hold. A gap in a row reads as a flat period, the leg held in cash, as [`price_relative`](@ref) reads it. The rule takes the logarithm of a price relative as `log1p` of the return. The correlation removes the mean of each window, and `1 + r` loses the precision of a small return. The raw step is in the simplex, so on the default Allocation Set the Euclidean projection returns it unchanged. On a stated set, the projection repairs it.

# Mathematical definition

For ``t \\geq 2 \\ell``, the log price relatives of the two consecutive windows that end at period ``t`` give the lagged correlation of asset ``i`` over the first window with asset ``j`` over the second:

```math
\\begin{align}
L^{(1)}_{k, i} &= \\log x_{t - 2\\ell + k, i}\\,,\\quad L^{(2)}_{k, i} = \\log x_{t - \\ell + k, i}\\,,\\quad k = 1, \\ldots, \\ell\\,,\\\\
M_{\\mathrm{cor}}(i, j) &= \\frac{1}{\\sigma_{1, i}\\, \\sigma_{2, j}} \\frac{1}{\\ell - 1} \\sum_{k = 1}^{\\ell} \\left( L^{(1)}_{k, i} - \\mu_{1, i} \\right) \\left( L^{(2)}_{k, j} - \\mu_{2, j} \\right)\\,.
\\end{align}
```

``M_{\\mathrm{cor}}(i, j)`` is zero when ``\\sigma_{1, i}`` or ``\\sigma_{2, j}`` is zero. Asset ``i`` claims a transfer to asset ``j \\neq i`` when ``\\mu_{2, i} \\geq \\mu_{2, j}`` and ``M_{\\mathrm{cor}}(i, j) > 0``. Every other claim is zero:

```math
\\begin{align}
\\mathrm{claim}_{i \\to j} &= M_{\\mathrm{cor}}(i, j) + \\max\\left( 0, -M_{\\mathrm{cor}}(i, i) \\right) + \\max\\left( 0, -M_{\\mathrm{cor}}(j, j) \\right)\\,,\\\\
\\mathrm{transfer}_{i \\to j} &= \\hat{w}_{t, i} \\frac{\\mathrm{claim}_{i \\to j}}{\\sum_{k} \\mathrm{claim}_{i \\to k}}\\,,\\\\
w_{t+1, i} &= \\hat{w}_{t, i} - \\sum_{j \\neq i} \\mathrm{transfer}_{i \\to j} + \\sum_{j \\neq i} \\mathrm{transfer}_{j \\to i}\\,.
\\end{align}
```

For ``t < 2 \\ell``, ``\\boldsymbol{w}_{t+1} = \\hat{\\boldsymbol{w}}_t``.

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:t_period])
  - ``\\hat{\\boldsymbol{w}}_t = \\boldsymbol{w}_t \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``: Price-Adjusted Allocation, the wealth held in each asset at the end of period ``t``.
  - ``\\ell``: Window length.
  - ``L^{(1)}``, ``L^{(2)}``: Log price relatives of the first and the second window, ``\\ell`` rows each.
  - ``\\boldsymbol{\\mu}_1``, ``\\boldsymbol{\\mu}_2``: Column means of ``L^{(1)}`` and ``L^{(2)}``.
  - ``\\boldsymbol{\\sigma}_1``, ``\\boldsymbol{\\sigma}_2``: Column sample standard deviations of ``L^{(1)}`` and ``L^{(2)}``.
  - ``M_{\\mathrm{cor}}(i, j)``: Lagged correlation of asset ``i`` over the first window with asset ``j`` over the second.
  - ``\\mathrm{claim}_{i \\to j}``, ``\\mathrm{transfer}_{i \\to j}``: Claim of asset ``i`` on a transfer to asset ``j``, and the wealth that moves.

The negative autocorrelation of an asset raises every claim that the asset takes part in. The transfers out of asset ``i`` sum to ``\\hat{w}_{t, i}`` or to zero, so no asset gives more than it holds, the budget stays one, and ``\\boldsymbol{w}_{t+1}`` is in the simplex.

The paper states the transfers twice. Its prose computes them from ``\\boldsymbol{w}_t``, the allocation at the start of the period. Its algorithm box takes ``\\hat{\\boldsymbol{w}}_t`` as its only allocation input, returns it while ``t < 2 \\ell``, and starts the new allocation from it. The box's step 6(a) writes the transfer from ``\\boldsymbol{w}_t``, which is not an input of the box. This rule computes every transfer from ``\\hat{\\boldsymbol{w}}_t``. The box's step 5 also lets ``i = j`` claim a transfer to itself, which keeps a part of its wealth in place. The prose asks for ``\\mu_{2, i} > \\mu_{2, j}``, which excludes ``i = j``. This rule takes the box's ``\\geq`` and excludes ``i = j``.

# Algorithm

 1. Set `T`, the number of rows the head holds, or `0` when `rows` is `nothing`.
 2. Compute `wh`, the Price-Adjusted Allocation of `w` over `x`.
 3. When `T < 2 * wn`, set `q = wh` and go to step 7.
 4. Take `Rw`, the last `2 * wn` rows of returns. Replace each return that is not finite with zero, and take `L`, the `log1p` of the result.
 5. Compute `cor` and `mu2` with [`lagged_window_correlation`](@ref) over the first `wn` rows and the last `wn` rows of `L`.
 6. Compute the claims with [`anticorrelation_claims`](@ref), and `q` with [`wealth_transfer`](@ref) from `wh`.
 7. Project `q` onto the Allocation Set with `alg.proj`, from `wh`. Return the carrier `st` unchanged, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AntiCorrelation(; window::Integer = 5, proj::EuclideanProjection = EuclideanProjection()) -> AntiCorrelation

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`, because a window of one row has no sample standard deviation. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> AntiCorrelation()
AntiCorrelation
  window ┼ Int64: 5
    proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`ExpertMixture`](@ref): the mixture over the windows is the paper's headline algorithm.
  - [`lagged_window_correlation`](@ref)
  - [`anticorrelation_claims`](@ref)
  - [`wealth_transfer`](@ref)
  - [`price_adjusted_allocation`](@ref)

# References

  - $(ref_dict[:borodin2004]) Section 3, Equations 2 and 3, and Figure 1.
"""
struct AntiCorrelation{T1 <: Integer, T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Window length ``\\ell``. The rule reads the last ``2 \\ell`` rows as two consecutive windows of ``\\ell`` rows.
    """
    window::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function AntiCorrelation(window::Integer, proj::EuclideanProjection)
        @argcheck(window >= 2, DomainError(window, "window must be at least 2"))
        return new{typeof(window), typeof(proj)}(window, proj)
    end
end
function AntiCorrelation(; window::Integer = 5,
                         proj::EuclideanProjection = EuclideanProjection())::AntiCorrelation
    return AntiCorrelation(window, proj)
end
function rows_needed(alg::AntiCorrelation)
    return 2 * alg.window
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the lagged correlation of two consecutive windows of log price relatives, and the column means of the second window.

Entry `(i, j)` is the correlation of column `i` of `L1` with column `j` of `L2`. The sample moments divide by the number of rows less one. An entry is zero when either column is constant. [`AntiCorrelation`](@ref) states the formula as ``M_{\\mathrm{cor}}``, and ranks the assets by the column means of the second window.

# Arguments

  - `L1`: Log price relatives of the first window, `rows × assets`.
  - `L2`: Log price relatives of the second window, with the same size as `L1`.

# Returns

  - `(cor::AbstractMatrix, mu2::AbstractVector)`: The `assets × assets` lagged correlation, and the column means of `L2`.

# Related

  - [`AntiCorrelation`](@ref)
  - [`anticorrelation_claims`](@ref)
"""
function lagged_window_correlation(L1::AbstractMatrix, L2::AbstractMatrix)
    mu1 = vec(Statistics.mean(L1; dims = 1))
    mu2 = vec(Statistics.mean(L2; dims = 1))
    s1 = vec(Statistics.std(L1; dims = 1))
    s2 = vec(Statistics.std(L2; dims = 1))
    cov = transpose(L1 .- transpose(mu1)) * (L2 .- transpose(mu2)) ./ (size(L1, 1) - 1)
    scale = s1 * transpose(s2)
    cor = cov ./ scale
    for k in eachindex(cor, scale)
        if iszero(scale[k])
            cor[k] = zero(eltype(cor))
        end
    end
    return cor, mu2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the claim matrix of an anti-correlation step.

Entry `(i, j)` is the claim of asset `i` on a transfer of its wealth to asset `j`. It is positive only when `i` differs from `j`, `mu2[i] >= mu2[j]`, and `cor[i, j]` is positive. The negative diagonal entries of `cor` add to every claim of their asset. [`AntiCorrelation`](@ref) states the formula.

# Arguments

  - `cor`: Lagged correlation from [`lagged_window_correlation`](@ref).
  - `mu2`: Column means of the second window.

# Returns

  - `claim::Matrix`: The `assets × assets` claims, zero on the diagonal.

# Related

  - [`AntiCorrelation`](@ref)
  - [`lagged_window_correlation`](@ref)
  - [`wealth_transfer`](@ref)
"""
function anticorrelation_claims(cor::AbstractMatrix, mu2::AbstractVector)
    N = length(mu2)
    claim = zeros(eltype(cor), N, N)
    for i in 1:N, j in 1:N
        if i != j && mu2[i] >= mu2[j] && cor[i, j] > zero(eltype(cor))
            claim[i, j] = cor[i, j] +
                          max(zero(eltype(cor)), -cor[i, i]) +
                          max(zero(eltype(cor)), -cor[j, j])
        end
    end
    return claim
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the allocation after the transfers that a claim matrix names, from the wealth `w` held in each asset.

Each asset with a positive claim moves all of its wealth out, in proportion to its claims, and receives what the other assets move to it. The sum of the allocation stays the sum of `w`, and no asset gives more than it holds.

# Arguments

  - `w`: Wealth held in each asset, the Price-Adjusted Allocation.
  - `claim`: Claims from [`anticorrelation_claims`](@ref).

# Returns

  - `q::AbstractVector`: The allocation after the transfers.

# Related

  - [`AntiCorrelation`](@ref)
  - [`anticorrelation_claims`](@ref)
"""
function wealth_transfer(w::AbstractVector, claim::AbstractMatrix)
    q = copy(w)
    for i in eachindex(w)
        total = sum(view(claim, i, :))
        if total > zero(total)
            out = w[i] .* view(claim, i, :) ./ total
            q .+= out
            q[i] -= sum(out)
        end
    end
    return q
end
function online_update!(alg::AntiCorrelation, st, w::AbstractVector, x::AbstractVector,
                        rows::Option{<:ReturnsResult}, set::AbstractAllocationSet)
    wn = alg.window
    T = isnothing(rows) ? 0 : size(rows.X, 1)
    # The transfers move the wealth held at the end of the period, the paper's `b̂_t`.
    wh = price_adjusted_allocation(w, x)
    q = if T < 2 * wn
        wh
    else
        # The windows are a kernel over price relatives, so a gap reads as one: the leg
        # sat in cash, as the step reads it (see [`price_relative`](@ref)). The logarithm
        # of that relative is `log1p` of the return, which keeps the precision of a small
        # return that the correlation then cancels.
        Rw = view(rows.X, (T - 2 * wn + 1):T, :)
        L = log1p.(ifelse.(isfinite.(Rw), Rw, zero(eltype(Rw))))
        cor, mu2 = lagged_window_correlation(view(L, 1:wn, :),
                                             view(L, (wn + 1):(2 * wn), :))
        wealth_transfer(wh, anticorrelation_claims(cor, mu2))
    end
    return st, project(alg.proj, set, q, wh)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the Start Allocation and the statistic of the Learning-Rate Schedule, for the expectation-maximisation rule.

The online form of [`ExpectationMaximisation`](@ref) pulls towards the Start Allocation when its rate falls.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ExpectationMaximisation`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct ExpectationMaximisationState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    Start Allocation ``\\boldsymbol{w}_1``, which the online form pulls towards.
    """
    w1
    """
    Statistic of the Learning-Rate Schedule, or `nothing` for a fixed rate.
    """
    s
end
function merge_states(::ExpectationMaximisationState, ::ExpectationMaximisationState)
    return throw(ArgumentError("an `ExpectationMaximisationState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::ExpectationMaximisationState)
    return ExpectationMaximisationState(x.n, copy(x.w1), copy_column(x.s))
end
function port_opt_view(x::ExpectationMaximisationState, i, args...)
    return ExpectationMaximisationState(x.n, renormalised_view(x.w1, i),
                                        schedule_state_view(x.s, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Moves the allocation a share `eta` of the way to the wealth held at the end of the period.

This is the expectation-maximisation update (EM) of Helmbold, Schapire, Singer and Warmuth. Orseau, Lattimore and Legg study the same update as Soft-Bayes. The wealth held at the end of the period is the posterior of the Bayesian mixture over the single assets, with the allocation as its prior. At `eta = 1` the rule is buy-and-hold, and below one it is that mixture slowed down. As the weighting of an [`ExpertMixture`](@ref), the rule is Soft-Bayes over the experts, the setting of the 2017 paper.

The raw step is positive wherever the allocation is, and it sums to one, so on the default Allocation Set the Euclidean projection returns it unchanged. The `proj` field is bound to [`EuclideanProjection`](@ref), which repairs the step on a stated set.

# Mathematical definition

The plain step at a fixed rate ``\\eta`` is

```math
\\begin{align}
w_{t+1, i} &= w_{t, i} \\left( 1 - \\eta + \\eta\\, g_{t, i} \\right) = (1 - \\eta)\\, w_{t, i} + \\eta\\, \\frac{w_{t, i}\\, x_{t, i}}{\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle}\\,,\\quad g_{t, i} = \\frac{x_{t, i}}{\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle}\\,.
\\end{align}
```

The rule writes the step in the online form of the 2017 paper, which pulls towards the Start Allocation when the rate falls:

```math
\\begin{align}
w_{t+1, i} &= \\rho_t\\, w_{t, i} \\left( 1 - \\eta_t + \\eta_t\\, g_{t, i} \\right) + \\left( 1 - \\rho_t \\right) w_{1, i}\\,,\\\\
\\rho_t &= \\min\\left( \\frac{\\eta_{t+1}}{\\eta_t},\\, c_t \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_1_start])
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:t_period])
  - $(math_dict[:N])
  - ``\\eta``: Fixed learning rate, in ``(0, 1)``.
  - ``\\boldsymbol{g}_t``: Price relative over the gross return of the allocation, so ``\\langle \\boldsymbol{w}_t, \\boldsymbol{g}_t \\rangle = 1``.
  - ``\\rho_t``: Ratio of the online form.
  - ``c_t``: Ceiling on the ratio, which [`correction_ratio_cap`](@ref) returns.

The step sums to one without a normalisation, because ``\\langle \\boldsymbol{w}_t, \\boldsymbol{g}_t \\rangle = 1``. A weight rises by at most ``\\eta``, ``w_{t+1, i} \\leq (1 - \\eta)\\, w_{t, i} + \\eta``, and falls to no less than ``(1 - \\eta)\\, w_{t, i}``. Helmbold and co-authors derive the step with the chi-squared distance to ``\\boldsymbol{w}_t`` in place of the relative entropy, and it is the first-order approximation of [`ExponentiatedGradient`](@ref). At a constant rate, ``\\rho_t = 1`` and the online form is the plain step. When the rate falls, the online form moves a share ``1 - \\rho_t`` of the allocation back to the Start Allocation, so no weight falls below ``(1 - \\rho_t)\\, w_{1, i}``. The form needs a rate that does not rise.

The 2017 paper bounds the regret against every constant rebalanced portfolio, for every sequence of non-negative price relatives, with no lower bound on the price relatives. With ``\\bar{\\eta} = \\eta / (1 - \\eta)`` tuned to ``\\sqrt{\\log N / (T m)}``, the regret is at most ``2 \\sqrt{T m \\log N} + m \\log(N / m) + \\log N``, and so at most ``2 \\sqrt{T N \\log N} + \\log N``. Here ``m \\leq N`` is the number of assets that are the best of their period at least once, and ``T`` is the number of periods. With ``C_1 = \\sum_t \\max_i (g_{t, i} - 1)``, the regret is at most ``\\min(C_1,\\, \\eta^{-1} \\log N + \\eta C_1 / 2 + \\eta^2 T)``, and at ``\\eta = \\sqrt{2 \\log N / C_1}`` at most ``\\min(C_1,\\, \\sqrt{2 C_1 \\log N} + 2 T \\log N / C_1)``. [`SelfConfidentRate`](@ref) reads that rate from the running ``C_1``, and the paper states no bound for this online rate. Under that rate, the paper advises the ceiling ``c_t = \\sqrt{t / (t + 1)}``. Its rate stays still while the mixture predicts well, and without the ceiling a weight can decay exponentially. With the ceiling, no weight falls below ``O(1 / t)`` of its start.

# Algorithm

 1. Set the period `t = st.n + 1`.
 2. When `restart(alg.eta, t)` holds, return the carrier with the period count `t`, the Start Allocation `st.w1` and the statistic of the schedule at its seed, and a copy of `st.w1`.
 3. Read the rate `eta_t` of the period with [`learning_rate`](@ref).
 4. Compute `g = x ./ dot(w, x)` and the plain step `q = w .* (1 - eta_t .+ eta_t .* g)`.
 5. Build the new carrier `stn`, with the period count `t` and the statistic that `schedule_update!` returns for the row.
 6. Compute `ratio`, the smaller of the rate of the next period over `eta_t` and [`correction_ratio_cap`](@ref). [`learning_rate`](@ref) reads the rate of the next period from `stn`.
 7. When `ratio` is not one, set `q = ratio .* q .+ (1 - ratio) .* st.w1`.
 8. Project `q` onto the Allocation Set with `alg.proj`, from the Price-Adjusted Allocation. Return `stn` and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpectationMaximisation(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, proj::EuclideanProjection = EuclideanProjection()) -> ExpectationMaximisation

Keywords correspond to the struct's fields.

## Validation

  - When `eta` is a number, `0 < eta < 1`. The regret bounds hold on that interval, and above one a weight of the raw step can turn negative. A `DomainError` is thrown otherwise. The rule reads a schedule at every period and does not check the rate it returns.
  - `eta` is not a schedule that reads the row of its own period. The online form also reads the rate of the next period, and that row does not exist yet. An `ArgumentError` is thrown otherwise.

# Examples

```jldoctest
julia> ExpectationMaximisation()
ExpectationMaximisation
   eta ┼ Float64: 0.05
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`ExpectationMaximisationState`](@ref)
  - [`ExponentiatedGradient`](@ref)
  - [`AbstractLearningRateSchedule`](@ref)
  - [`learning_rate`](@ref)
  - [`correction_ratio_cap`](@ref)
  - [`SelfConfidentRate`](@ref)

# References

  - $(ref_dict[:helmbold1997]) Section 3, Equation 7.
  - $(ref_dict[:orseau2017]) Equations 7 and 14, Theorems 3 and 6, and Remark 8.
"""
struct ExpectationMaximisation{T1 <: Union{<:Real, <:AbstractLearningRateSchedule},
                               T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Learning rate, a number in `(0, 1)` or a Learning-Rate Schedule. It is the share of the way that the allocation moves to the wealth held at the end of the period.
    """
    eta::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function ExpectationMaximisation(eta::Union{<:Real, <:AbstractLearningRateSchedule},
                                     proj::EuclideanProjection)
        if isa(eta, Real)
            @argcheck(zero(eta) < eta < one(eta), DomainError(eta, "eta must be in (0, 1)"))
        end
        @argcheck(!reads_period_row(eta),
                  ArgumentError("`ExpectationMaximisation` refuses a `$(typeof(eta).name.name)` on `eta`: its online form reads the rate of the next period as well, and a schedule that chooses each period's rate from that period's row has no next row to read it from."))
        return new{typeof(eta), typeof(proj)}(eta, proj)
    end
end
function ExpectationMaximisation(;
                                 eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
                                 proj::EuclideanProjection = EuclideanProjection())::ExpectationMaximisation
    return ExpectationMaximisation(eta, proj)
end
function rule_state_seed(alg::ExpectationMaximisation, w::AbstractVector)
    return ExpectationMaximisationState(0, copy(w), schedule_state_seed(alg.eta, w))
end
"""
    correction_ratio_cap(eta::Real, t::Integer)
    correction_ratio_cap(sched::AbstractLearningRateSchedule, t::Integer)

Returns the ceiling on the ratio ``\\eta_{t+1} / \\eta_t`` in the online form of [`ExpectationMaximisation`](@ref) at period `t`.

The ceiling is one for a number and for every Learning-Rate Schedule that names no other, so the pull towards the Start Allocation comes from the fall of the rate alone. Under [`SelfConfidentRate`](@ref) it is ``\\sqrt{t / (t + 1)}``. That rate stays still while the mixture predicts well, and without the ceiling the weight of a bad predictor can decay exponentially. With the ceiling, no weight falls below ``O(1 / t)`` of its start, so the mixture can still find an expert that becomes good late.

# Arguments

  - `eta`: The `eta` field of the rule, a number or a Learning-Rate Schedule.
  - `t`: The period.

# Returns

  - `cap::Real`: The ceiling, in ``(0, 1]``.

# Related

  - [`ExpectationMaximisation`](@ref)
  - [`SelfConfidentRate`](@ref)
  - [`learning_rate`](@ref)
"""
function correction_ratio_cap(::Union{<:Real, <:AbstractLearningRateSchedule}, ::Integer)
    return 1
end
function online_update!(alg::ExpectationMaximisation, st::ExpectationMaximisationState,
                        w::AbstractVector, x::AbstractVector, ::Any,
                        set::AbstractAllocationSet)
    t = st.n + 1
    if restart(alg.eta, t)
        # The period count survives the restart: the schedule's stages are cumulative.
        return ExpectationMaximisationState(t, st.w1, schedule_state_seed(alg.eta, st.w1)),
               copy(st.w1)
    end
    eta_t = learning_rate(alg.eta, t, st)
    g = x ./ LinearAlgebra.dot(w, x)
    q = w .* (one(eta_t) - eta_t .+ eta_t .* g)
    stn = ExpectationMaximisationState(t, st.w1, schedule_update!(alg.eta, st.s, w, x))
    ratio = min(learning_rate(alg.eta, t + 1, stn) / eta_t,
                correction_ratio_cap(alg.eta, t))
    if !isone(ratio)
        q .= ratio .* q .+ (one(ratio) - ratio) .* st.w1
    end
    return stn, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Multiplies each weight by its price relative raised to the rate `eta`, then normalises the result.

This is the aggregating algorithm (AA) of Vovk and Watkins over a finite set. At `eta = 1` it is the wealth weighting of [`BuyAndHold`](@ref). Over experts it is the wealth-weighted mixture, and over sampled constant rebalanced portfolios it is Cover's universal portfolio. Below one it discounts the evidence of every period alike. The rate is fixed, and [`WeakAggregatingAlgorithm`](@ref) is the weighting whose rate shrinks with the period. The raw step multiplies a non-negative vector, so its geometry is the entropic one, and the `proj` field is bound to [`EntropicProjection`](@ref).

# Mathematical definition

```math
\\begin{align}
w_{t+1, i} &= \\frac{w_{t, i}\\, x_{t, i}^{\\eta}}{\\sum_{j = 1}^{N} w_{t, j}\\, x_{t, j}^{\\eta}} = \\frac{w_{1, i} \\exp\\left( \\eta\\, G_{t, i} \\right)}{\\sum_{j = 1}^{N} w_{1, j} \\exp\\left( \\eta\\, G_{t, j} \\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_1_start])
  - $(math_dict[:G_t_cumlog])
  - $(math_dict[:t_period])
  - $(math_dict[:N])
  - ``\\eta > 0``: Rate that the price relative is raised to.

The weight of an asset, or of an expert, is its start weight times its wealth raised to ``\\eta``. As the weighting of an [`ExpertMixture`](@ref), the blend is the average of the experts' allocations under these weights, which is the prediction of Vovk and Watkins' algorithm. As ``\\eta`` grows without bound, the weight gathers on the wealthiest asset, which is [`TopK`](@ref) at ``k = 1``. As ``\\eta`` falls to zero, the weights stay at the Start Allocation.

# Algorithm

 1. Compute `q = w .* x .^ alg.eta`.
 2. Project `q` onto the Allocation Set with `alg.proj`, from the Price-Adjusted Allocation. On the default set the entropic projection divides `q` by its sum. Return the carrier `st` unchanged, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AggregatingAlgorithm(; eta::Real = 1, proj::EntropicProjection = EntropicProjection()) -> AggregatingAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - `eta > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> AggregatingAlgorithm()
AggregatingAlgorithm
   eta ┼ Int64: 1
  proj ┴ EntropicProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ExpertMixture`](@ref)
  - [`BuyAndHold`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)
  - [`TopK`](@ref)

# References

  - $(ref_dict[:vovkwatkins1998]) Algorithm 1.
"""
struct AggregatingAlgorithm{T1 <: Real, T2 <: EntropicProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Rate ``\\eta`` that the price relative is raised to. At `1` the rule is the wealth weighting.
    """
    eta::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function AggregatingAlgorithm(eta::Real, proj::EntropicProjection)
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        return new{typeof(eta), typeof(proj)}(eta, proj)
    end
end
function AggregatingAlgorithm(; eta::Real = 1,
                              proj::EntropicProjection = EntropicProjection())::AggregatingAlgorithm
    return AggregatingAlgorithm(eta, proj)
end
function online_update!(alg::AggregatingAlgorithm, st, w::AbstractVector, x::AbstractVector,
                        ::Any, set::AbstractAllocationSet)
    q = w .* x .^ alg.eta
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the cumulative log wealth of each asset and the Start Allocation, for the top-``k`` and weak aggregating weightings.

[`TopK`](@ref) and [`WeakAggregatingAlgorithm`](@ref) share it. On an [`ExpertMixture`](@ref), each entry belongs to an expert and not to an asset.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`TopK`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct CumulativeWealthState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    Cumulative log wealth ``\\boldsymbol{G}_t = \\sum_{s \\leq t} \\log \\boldsymbol{x}_s`` of each asset. The update writes it in place.
    """
    G
    """
    Start Allocation, the prior of the weak aggregating weights. [`TopK`](@ref) does not read it.
    """
    p0
end
function merge_states(::CumulativeWealthState, ::CumulativeWealthState)
    return throw(ArgumentError("a `CumulativeWealthState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::CumulativeWealthState)
    return CumulativeWealthState(x.n, copy(x.G), copy(x.p0))
end
function port_opt_view(x::CumulativeWealthState, i, args...)
    return CumulativeWealthState(x.n, x.G[i], renormalised_view(x.p0, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the ``k`` wealthiest assets, or experts, each weighted by its wealth.

This is the top-``k`` combination of Li, Hoi and Gopalkrishnan (CORN-K). The paper combines its correlation-driven experts this way, with ``k = 5``. On an [`ExpertMixture`](@ref), the rule weights the experts. At ``k = 1`` it follows the single wealthiest asset. At ``k = N`` it is buy-and-hold from the uniform allocation. The raw step is in the simplex, so on the default Allocation Set the Euclidean projection returns it unchanged. On a stated set, the projection repairs it, and the `proj` field is bound to [`EuclideanProjection`](@ref).

# Mathematical definition

```math
\\begin{align}
w_{t+1, i} &= \\begin{cases} \\dfrac{\\exp\\left( G_{t, i} \\right)}{\\sum_{j \\in \\mathcal{K}_t} \\exp\\left( G_{t, j} \\right)} & i \\in \\mathcal{K}_t\\,,\\\\ 0 & i \\notin \\mathcal{K}_t\\,. \\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:G_t_cumlog])
  - $(math_dict[:t_period])
  - $(math_dict[:N])
  - ``\\mathcal{K}_t``: The ``k`` assets of largest ``G_{t, i}``. Equal wealth goes to the lower index.

``\\exp(G_{t, i})`` is the wealth of asset ``i`` after period ``t`` from a unit start, so the weights are the paper's uniform distribution over the top ``k`` times their wealth.

# Algorithm

 1. Add `log.(x)` to `st.G`.
 2. Sort the assets by `st.G`, largest first, with a stable sort so that equal wealth keeps the index order. Take the first `alg.k` of them, `top`.
 3. Set `q` to `exp.(st.G[top] .- st.G[first(order)])` on `top` and to zero elsewhere, and divide `q` by its sum. The shift by the largest log wealth keeps the exponential finite.
 4. Project `q` onto the Allocation Set with `alg.proj`, from the Price-Adjusted Allocation. Return the carrier with its period count raised by one, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TopK(; k::Integer = 5, proj::EuclideanProjection = EuclideanProjection()) -> TopK

Keywords correspond to the struct's fields. The default `k = 5` is the value of the paper.

## Validation

  - `k >= 1`. A `DomainError` is thrown otherwise.
  - `k` is at most the number of assets or experts. The seed checks it, because the count is first known there. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> TopK()
TopK
     k ┼ Int64: 5
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ExpertMixture`](@ref)
  - [`CumulativeWealthState`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)
  - [`AggregatingAlgorithm`](@ref): its limit as the rate grows without bound is this rule at ``k = 1``.

# References

  - $(ref_dict[:li2011corn]) Equation 8 and Algorithm 3.
"""
struct TopK{T1 <: Integer, T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Number of assets, or experts, held.
    """
    k::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function TopK(k::Integer, proj::EuclideanProjection)
        @argcheck(k >= 1, DomainError(k, "k must be at least 1"))
        return new{typeof(k), typeof(proj)}(k, proj)
    end
end
function TopK(; k::Integer = 5, proj::EuclideanProjection = EuclideanProjection())::TopK
    return TopK(k, proj)
end
function rule_state_seed(alg::TopK, w::AbstractVector)
    @argcheck(alg.k <= length(w),
              DomainError(alg.k,
                          "k must be at most the number of assets or experts, $(length(w))"))
    return CumulativeWealthState(0, zeros(eltype(w), length(w)), copy(w))
end
function online_update!(alg::TopK, st::CumulativeWealthState, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    st.G .+= log.(x)
    order = sortperm(st.G; rev = true, alg = Base.Sort.DEFAULT_STABLE)
    top = view(order, 1:(alg.k))
    # The paper weights the top k by their wealth. The shift by the largest log wealth, which
    # the normalisation undoes, keeps a long run from overflowing the exponential.
    q = zeros(eltype(st.G), length(w))
    q[top] .= exp.(st.G[top] .- st.G[first(order)])
    q ./= sum(q)
    wn = project(alg.proj, set, q, price_adjusted_allocation(w, x))
    return CumulativeWealthState(st.n + 1, st.G, st.p0), wn
end
"""
$(DocStringExtensions.TYPEDEF)

Weights each asset, or expert, by its prior times its wealth raised to a power that shrinks with the period.

This is the weak aggregating algorithm (WAA) of Kalnishkan and Vyugin, in the form that Yang, He, Lin and Zhang (2020) and Yang, He and Zhang (2022) apply to portfolios. It differs from [`AggregatingAlgorithm`](@ref) by the shrinking rate. The weights read the cumulative wealth and not the previous weights, so the carrier holds the prior and the wealth. The weights are normalised, which is the entropic projection onto the simplex, and the `proj` field is bound to [`EntropicProjection`](@ref).

# Mathematical definition

```math
\\begin{align}
w_{t+1, i} &= \\frac{w_{1, i} \\exp\\left( G_{t, i} / \\sqrt{t + 1} \\right)}{\\sum_{j = 1}^{N} w_{1, j} \\exp\\left( G_{t, j} / \\sqrt{t + 1} \\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_1_start])
  - $(math_dict[:G_t_cumlog])
  - $(math_dict[:t_period])
  - $(math_dict[:N])

Both applied papers take the rate ``1 / \\sqrt{t + 1}`` with no constant. At a fixed rate ``\\eta`` in place of ``1 / \\sqrt{t + 1}``, the weights are those of [`AggregatingAlgorithm`](@ref). Over experts with log returns in ``[-L, 0]``, Lemma 1 of the 2020 paper puts the cumulative log wealth of the mixture within ``\\sqrt{T} \\left( \\log K + L^2 \\right)`` of the best of ``K`` experts under a uniform prior. A common scale of the price relatives of a period moves every log return by the same amount, so the bound holds for every sequence whose worst price relative in a period is at least ``e^{-L}`` times its best. The average log growth rate of the mixture then tends to that of the best expert.

# Algorithm

 1. Add `log.(x)` to `st.G`, and set the period `n = st.n + 1`.
 2. Compute `q = st.p0 .* exp.((st.G .- maximum(st.G)) ./ sqrt(n + 1))`. The shift by the largest log wealth, which the normalisation removes, keeps the exponential finite.
 3. Project `q` onto the Allocation Set with `alg.proj`, from the Price-Adjusted Allocation. On the default set the entropic projection divides `q` by its sum. Return the carrier with the period count `n`, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WeakAggregatingAlgorithm(; proj::EntropicProjection = EntropicProjection()) -> WeakAggregatingAlgorithm

Keywords correspond to the struct's fields. The prior is the Start Allocation. On an [`ExpertMixture`](@ref) it is the mixture's `p0`, which is uniform by default.

# Examples

```jldoctest
julia> WeakAggregatingAlgorithm()
WeakAggregatingAlgorithm
  proj ┴ EntropicProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ExpertMixture`](@ref)
  - [`AggregatingExponentialGradient`](@ref)
  - [`CumulativeWealthState`](@ref)
  - [`AggregatingAlgorithm`](@ref)

# References

  - $(ref_dict[:kalnishkanvyugin2008])
  - $(ref_dict[:yang2020waeg]) Section 3.3, Algorithm 2 and Lemma 1, and Equations 9 and 11.
  - $(ref_dict[:yang2022caeg]) Equation 10 and Algorithm 1.
"""
struct WeakAggregatingAlgorithm{T1 <: EntropicProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:proj])
    """
    proj::T1
    function WeakAggregatingAlgorithm(proj::EntropicProjection)
        return new{typeof(proj)}(proj)
    end
end
function WeakAggregatingAlgorithm(;
                                  proj::EntropicProjection = EntropicProjection())::WeakAggregatingAlgorithm
    return WeakAggregatingAlgorithm(proj)
end
function rule_state_seed(::WeakAggregatingAlgorithm, w::AbstractVector)
    return CumulativeWealthState(0, zeros(eltype(w), length(w)), copy(w))
end
function online_update!(alg::WeakAggregatingAlgorithm, st::CumulativeWealthState,
                        w::AbstractVector, x::AbstractVector, ::Any,
                        set::AbstractAllocationSet)
    st.G .+= log.(x)
    n = st.n + 1
    # The exponent is shifted by its maximum, which the normalisation undoes, so a long run
    # never overflows the exponential.
    q = st.p0 .* exp.((st.G .- maximum(st.G)) ./ sqrt((n + 1) * one(eltype(st.G))))
    wn = project(alg.proj, set, q, price_adjusted_allocation(w, x))
    return CumulativeWealthState(n, st.G, st.p0), wn
end
"""
    AggregatingExponentialGradient(;
        etas::AbstractVector{<:Real} = 0.01:0.01:0.2,
        eset::Option{<:BoundedAllocationSet} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ExpertMixture

Builds the mixture of exponentiated-gradient experts under the weak aggregating algorithm, with a uniform prior.

This is the weak aggregating exponential gradient (WAEG) of Yang, He, Lin and Zhang (2020). The result is an [`ExpertMixture`](@ref) under [`WeakAggregatingAlgorithm`](@ref), with one [`ExponentiatedGradient`](@ref) expert for each rate in `etas`. Both papers prove the mixture universal under their assumptions on the price relatives: its average log growth rate tends to that of the best constant rebalanced portfolio.

Yang, He and Zhang (2022) index the experts by a continuum of rates under a measure (CAEG). In their experiments they discretise the continuum to the grid `0.01:0.01:0.2`, and the weight of an expert is then the weight of this constructor. So on that grid the two forms give the same allocations. The default of `etas` is that grid, which both papers use. The 2020 paper reports that the final wealth hardly changes with the number of experts on this range, and falls slowly as the upper end of the range rises from `0.1` to `0.8`.

# Arguments

  - `etas`: Rates of the experts, one exponentiated-gradient expert for each.
  - `eset`: Expert Set that the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: Geometry in which the mixture projects its blend onto the Allocation Set of the head.

# Validation

  - `etas` is not empty. An `IsEmptyError` is thrown otherwise.
  - Each rate in `etas` is positive and finite. A `DomainError` is thrown otherwise.

# Returns

  - `mix::ExpertMixture`: The mixture.

# Examples

```jldoctest
julia> mix = AggregatingExponentialGradient(; etas = [0.05, 0.1]);

julia> length(mix.experts), mix.alg
(2, WeakAggregatingAlgorithm
  proj ┴ EntropicProjection()
)
```

# Related

  - [`ExpertMixture`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)
  - [`ExponentiatedGradient`](@ref)

# References

  - $(ref_dict[:yang2020waeg]) Section 4, Algorithm 3 and Section 5.4.
  - $(ref_dict[:yang2022caeg]) Section 6.2.1.
"""
function AggregatingExponentialGradient(; etas::AbstractVector{<:Real} = 0.01:0.01:0.2,
                                        eset::Option{<:BoundedAllocationSet} = nothing,
                                        proj::EuclideanProjection = EuclideanProjection())::ExpertMixture
    assert_nonempty_gt0_finite_val(etas, :etas)
    experts = map(eta -> ExponentiatedGradient(; eta = eta), etas)
    return ExpertMixture(; experts = experts, alg = WeakAggregatingAlgorithm(), eset = eset,
                         proj = proj)
end
export ConfidenceWeightedMeanReversion, VarianceUpdate, StandardDeviationUpdate,
       AntiCorrelation, ExpectationMaximisation, AggregatingAlgorithm, TopK,
       WeakAggregatingAlgorithm, AggregatingExponentialGradient
public AbstractConfidenceUpdate, confidence_step, confidence_gain
