"""
$(DocStringExtensions.TYPEDEF)

The anytime rate ``\\eta_t = c / \\sqrt{t}``: the schedule of Zinkevich (2003, Theorem 1) for online gradient descent, and at ``c = \\sqrt{\\log N / (2 N)}`` the anytime Soft-Bayes rate of Orseau, Lattimore and Legg (2017, Theorem 10).

It reads the period count alone. Under the Euclidean map it carries Zinkevich's ``O(\\sqrt{T})`` regret with no horizon; under the entropic map the fixed-horizon constant of Helmbold, Schapire, Singer and Warmuth (1998, Theorem 4.1), ``\\eta = r \\sqrt{2 \\log N / T}``, needs the horizon and the lower bound ``r`` on the price relatives, which no online rule knows, so it is a documented formula and not a schedule.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    InverseSquareRootRate(; c::Real = 1) -> InverseSquareRootRate

Keywords correspond to the struct's fields.

## Validation

  - `c > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> InverseSquareRootRate()
InverseSquareRootRate
  c ┴ Int64: 1
```

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`MirrorDescent`](@ref)

# References

  - $(ref_dict[:zinkevich2003])
  - $(ref_dict[:orseau2017])
"""
struct InverseSquareRootRate{T1 <: Real} <: AbstractLearningRateSchedule
    """
    The constant the inverse square root of the period count is scaled by.
    """
    c::T1
    function InverseSquareRootRate(c::Real)
        @argcheck(c > zero(c), DomainError(c, "c must be positive"))
        return new{typeof(c)}(c)
    end
end
function InverseSquareRootRate(; c::Real = 1)::InverseSquareRootRate
    return InverseSquareRootRate(c)
end
function learning_rate(sched::InverseSquareRootRate, t::Integer, ::Any)
    return sched.c / sqrt(t)
end
"""
$(DocStringExtensions.TYPEDEF)

The doubling trick of Helmbold, Schapire, Singer and Warmuth (1998, Corollary 4.3): the exponentiated gradient with the uniform mix is run in stages, restarted at uniform at every stage boundary, with the share and the rate of each stage set from its length as Theorem 4.2 sets them from the horizon.

# Mathematical definition

Stage ``0`` lasts ``\\lceil 2 N^2 \\log N \\rceil`` periods and stage ``i > 0`` lasts ``\\lceil 2^i N^2 \\log N \\rceil``. Within a stage of length ``T_i``,

```math
\\begin{align}
\\alpha_i &= \\left( \\frac{N^2 \\log N}{8 T_i} \\right)^{1/4}\\,,\\quad
\\eta_i = \\sqrt{\\frac{8 \\alpha_i^2 \\log N}{N^2 T_i}}\\,,
\\end{align}
```

and the update of the last period of a stage answers the Start Allocation for the first period of the next, with the rule's carrier back at its seed. The staged rule is universal: its regret against every constant rebalanced portfolio is ``6 N^2 \\log N (1 + (T / (2 N^2 \\log N))^{3/4})`` for every sequence of non-negative price relatives, with no lower bound on them. The schedule answers both the rate and the share, so the `alpha` of a [`MirrorDescent`](@ref) that holds it is not read: the stage's share takes precedence. The theorem is the entropic map's; under any other map the schedule is a restart rule with a rate.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DoublingTrickRate(; N::Integer) -> DoublingTrickRate

Keywords correspond to the struct's fields. `N` is the number of assets the rule moves over — the number of experts when the rule is a mixture's weighting — and has no default, because the stage lengths are its function.

## Validation

  - `N >= 2`. A `DomainError` is thrown otherwise: a single asset has no stage.

# Examples

```jldoctest
julia> DoublingTrickRate(; N = 4)
DoublingTrickRate
  N ┴ Int64: 4
```

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`MirrorDescent`](@ref)
  - [`ExponentiatedGradient`](@ref)

# References

  - $(ref_dict[:helmbold1998])
"""
struct DoublingTrickRate{T1 <: Integer} <: AbstractLearningRateSchedule
    """
    The number of assets, from which the stage lengths, the shares and the rates follow.
    """
    N::T1
    function DoublingTrickRate(N::Integer)
        @argcheck(N >= 2, DomainError(N, "N must be at least 2"))
        return new{typeof(N)}(N)
    end
end
function DoublingTrickRate(; N::Integer)::DoublingTrickRate
    return DoublingTrickRate(N)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The stage of a [`DoublingTrickRate`](@ref) that period `t` falls in: its length and the period that closes it.

# Related

  - [`DoublingTrickRate`](@ref)
"""
function doubling_stage(sched::DoublingTrickRate, t::Integer)
    base = sched.N^2 * log(sched.N)
    len = ceil(Int, 2 * base)
    fin = len
    i = 0
    while t > fin
        i += 1
        len = ceil(Int, 2^i * base)
        fin += len
    end
    return len, fin
end
function learning_rate(sched::DoublingTrickRate, t::Integer, ::Any)
    len, _ = doubling_stage(sched, t)
    N = sched.N
    alpha = mixing_share(sched, t, zero(len))
    return sqrt(8 * alpha^2 * log(N) / (N^2 * len))
end
function mixing_share(sched::DoublingTrickRate, t::Integer, ::Real)
    len, _ = doubling_stage(sched, t)
    N = sched.N
    return (N^2 * log(N) / (8 * len))^(1 / 4)
end
function restart(sched::DoublingTrickRate, t::Integer)
    _, fin = doubling_stage(sched, t)
    return t == fin
end
"""
$(DocStringExtensions.TYPEDEF)

The self-confident rate of Orseau, Lattimore and Legg (2017, Theorems 5 and 6): ``\\eta_t = \\sqrt{2 \\log N / C_{1, t-1}}``, read from the running first-order excess ``C_{1, t} = \\sum_{s \\leq t} \\max_i \\left( x_{s, i} / \\langle \\boldsymbol{w}_s, \\boldsymbol{x}_s \\rangle - 1 \\right)``, which the rule's carrier keeps for it beside ``\\log N``.

The statistic is non-decreasing, so the rate is non-increasing and may be updated online, as the paper notes; with it Theorem 6's regret against every constant rebalanced portfolio is ``\\min(C_1, 2 \\sqrt{C_1 \\log N} + \\sqrt{2 T \\log N / C_1})``, small when one asset is the best predictor for long stretches and never worse than ``O(\\sqrt{T \\log N})``. The rate is capped at `eta_max`, because the statistic is zero before the first row and the theorem's rate lies in ``(0, 1)``; the paper states no cap, and the default is the interval's end. The excess is measured at the allocation the rule played, so the mix of a [`MirrorDescent`](@ref) with a positive `alpha` is what it reads.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SelfConfidentRate(; eta_max::Real = 1) -> SelfConfidentRate

Keywords correspond to the struct's fields.

## Validation

  - `eta_max > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> SelfConfidentRate()
SelfConfidentRate
  eta_max ┴ Int64: 1
```

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`MirrorDescent`](@ref)

# References

  - $(ref_dict[:orseau2017])
"""
struct SelfConfidentRate{T1 <: Real} <: AbstractLearningRateSchedule
    """
    The cap on the rate, answered while the running excess is zero and whenever the formula exceeds it.
    """
    eta_max::T1
    function SelfConfidentRate(eta_max::Real)
        @argcheck(eta_max > zero(eta_max), DomainError(eta_max, "eta_max must be positive"))
        return new{typeof(eta_max)}(eta_max)
    end
end
function SelfConfidentRate(; eta_max::Real = 1)::SelfConfidentRate
    return SelfConfidentRate(eta_max)
end
function schedule_state_seed(::SelfConfidentRate, w::AbstractVector)
    # The statistic is the pair `[C₁, log N]`, so the rate reads the carrier's `s` alone.
    s = zeros(eltype(w), 2)
    s[2] = log(length(w))
    return s
end
function schedule_update!(::SelfConfidentRate, s::AbstractVector, w::AbstractVector,
                          x::AbstractVector)
    s[1] += maximum(x ./ LinearAlgebra.dot(w, x)) - one(eltype(s))
    return s
end
function learning_rate(sched::SelfConfidentRate, ::Integer, st)
    c = st.s[1]
    if !(c > zero(c))
        return sched.eta_max
    end
    return min(sqrt(2 * st.s[2] / c), sched.eta_max)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Gradient Transforms a [`MirrorDescent`](@ref) rule may apply to its gradient before the mirror step: the momentum carriers of Li, Zheng, Chen, Wang and Xu (2022), which rescale the exponent of the multiplicative update and leave the update itself untouched.

# Interfaces

In order to implement a new transform, subtype `AbstractGradientTransform` and implement:

  - `gradient_state_seed(grad::AbstractGradientTransform, w::AbstractVector)`: The carrier the transform keeps on the Rule State before the first row, or `nothing`, the default; `w` is the Start Allocation, whose length and element type the carrier takes.
  - `transform_gradient!(grad::AbstractGradientTransform, gs, g::AbstractVector) -> AbstractVector`: The transformed gradient of the period from the raw one, writing the carrier `gs` in place.

A carrier that is `nothing`, a vector or a pair of vectors is sliced and copied with the head's state already; a carrier of another shape needs a slice and a copy of its own, through the two private helpers [`MirrorDescentState`](@ref) names.

# Related

  - [`PlainGradient`](@ref)
  - [`GradientMomentum`](@ref)
  - [`RootMeanSquareGradient`](@ref)
  - [`AdaptiveMomentGradient`](@ref)
  - [`MirrorDescent`](@ref)
"""
abstract type AbstractGradientTransform <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The identity Gradient Transform: the mirror step reads the period's gradient as it is. The default of [`MirrorDescent`](@ref).

# Examples

```jldoctest
julia> PlainGradient()
PlainGradient()
```

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`MirrorDescent`](@ref)
"""
struct PlainGradient <: AbstractGradientTransform end
"""
$(DocStringExtensions.TYPEDEF)

The exponential moving average of the gradient (Li, Zheng, Chen, Wang and Xu 2022, Eq. 7): ``\\boldsymbol{v}_{t+1} = \\gamma_1 \\boldsymbol{v}_t + (1 - \\gamma_1) \\boldsymbol{g}_t`` from ``\\boldsymbol{v}_1 = \\boldsymbol{0}``, and the step reads ``\\boldsymbol{v}_{t+1}``.

With the entropic map it is the paper's EGE; the regret is the exponentiated gradient's ``O(\\sqrt{T \\log N})`` (their Theorem 1). The first step is scaled by ``1 - \\gamma_1``, because the average starts at zero; the paper applies no bias correction and neither does this.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GradientMomentum(; gamma1::Real = 0.99) -> GradientMomentum

Keywords correspond to the struct's fields. The default is the paper's.

## Validation

  - `0 <= gamma1 < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> GradientMomentum()
GradientMomentum
  gamma1 ┴ Float64: 0.99
```

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`AdaptiveMomentGradient`](@ref)
  - [`EGE`](@ref)

# References

  - $(ref_dict[:li2022egm])
"""
struct GradientMomentum{T1 <: Real} <: AbstractGradientTransform
    """
    The decay of the moving average of the gradient, in `[0, 1)`.
    """
    gamma1::T1
    function GradientMomentum(gamma1::Real)
        @argcheck(zero(gamma1) <= gamma1 < one(gamma1),
                  DomainError(gamma1, "gamma1 must be in [0, 1)"))
        return new{typeof(gamma1)}(gamma1)
    end
end
function GradientMomentum(; gamma1::Real = 0.99)::GradientMomentum
    return GradientMomentum(gamma1)
end
"""
$(DocStringExtensions.TYPEDEF)

The root-mean-square rescaling of the gradient (Li, Zheng, Chen, Wang and Xu 2022, Eq. 8): ``\\boldsymbol{m}_{t+1} = \\gamma_2 \\boldsymbol{m}_t + (1 - \\gamma_2) \\boldsymbol{g}_t^2`` from ``\\boldsymbol{m}_1 = \\boldsymbol{0}``, and the step reads ``\\boldsymbol{g}_t / (\\sqrt{\\boldsymbol{m}_{t+1}} + \\epsilon)``.

With the entropic map it is the paper's EGR. **At the paper's recommended ``\\gamma_2 = 0``, the default, ``\\boldsymbol{m}_{t+1} = \\boldsymbol{g}_t^2`` and the transformed gradient is ``\\boldsymbol{g}_t / (\\lvert \\boldsymbol{g}_t \\rvert + \\epsilon)``, the sign of the gradient up to ``\\epsilon``: every asset's exponent is the same, so the multiplicative update moves nothing and the rule holds the Price-Adjusted Allocation's normalisation of its own iterate** — the paper's own choice makes the rule nearly buy-and-hold on the current mix. A positive ``\\gamma_2`` is what gives the rescaling a memory.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RootMeanSquareGradient(; gamma2::Real = 0, eps::Real = 1e-8) -> RootMeanSquareGradient

Keywords correspond to the struct's fields. The defaults are the paper's.

## Validation

  - `0 <= gamma2 < 1`, `eps > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> RootMeanSquareGradient()
RootMeanSquareGradient
  gamma2 ┼ Int64: 0
     eps ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`AdaptiveMomentGradient`](@ref)
  - [`EGR`](@ref)

# References

  - $(ref_dict[:li2022egm])
"""
struct RootMeanSquareGradient{T1 <: Real, T2 <: Real} <: AbstractGradientTransform
    """
    The decay of the moving average of the squared gradient, in `[0, 1)`.
    """
    gamma2::T1
    """
    The constant added to the root of the average before the division.
    """
    eps::T2
    function RootMeanSquareGradient(gamma2::Real, eps::Real)
        @argcheck(zero(gamma2) <= gamma2 < one(gamma2),
                  DomainError(gamma2, "gamma2 must be in [0, 1)"))
        @argcheck(eps > zero(eps), DomainError(eps, "eps must be positive"))
        return new{typeof(gamma2), typeof(eps)}(gamma2, eps)
    end
end
function RootMeanSquareGradient(; gamma2::Real = 0,
                                eps::Real = 1e-8)::RootMeanSquareGradient
    return RootMeanSquareGradient(gamma2, eps)
end
"""
$(DocStringExtensions.TYPEDEF)

The adaptive-moment rescaling of the gradient (Li, Zheng, Chen, Wang and Xu 2022, Eq. 9): both averages of [`GradientMomentum`](@ref) and [`RootMeanSquareGradient`](@ref), and the step reads ``\\boldsymbol{v}_{t+1} / (\\sqrt{\\boldsymbol{m}_{t+1}} + \\epsilon)`` with no bias correction.

With the entropic map it is the paper's EGA. At the paper's ``\\gamma_2 = 0``, the default, the divisor is ``\\lvert \\boldsymbol{g}_t \\rvert + \\epsilon``, so the step reads the momentum in units of the current gradient's size.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AdaptiveMomentGradient(; gamma1::Real = 0.99, gamma2::Real = 0, eps::Real = 1e-8) -> AdaptiveMomentGradient

Keywords correspond to the struct's fields. The defaults are the paper's.

## Validation

  - `0 <= gamma1 < 1`, `0 <= gamma2 < 1`, `eps > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> AdaptiveMomentGradient()
AdaptiveMomentGradient
  gamma1 ┼ Float64: 0.99
  gamma2 ┼ Int64: 0
     eps ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`GradientMomentum`](@ref)
  - [`RootMeanSquareGradient`](@ref)
  - [`EGA`](@ref)

# References

  - $(ref_dict[:li2022egm])
"""
struct AdaptiveMomentGradient{T1 <: Real, T2 <: Real, T3 <: Real} <:
       AbstractGradientTransform
    """
    The decay of the moving average of the gradient, in `[0, 1)`.
    """
    gamma1::T1
    """
    The decay of the moving average of the squared gradient, in `[0, 1)`.
    """
    gamma2::T2
    """
    The constant added to the root of the second average before the division.
    """
    eps::T3
    function AdaptiveMomentGradient(gamma1::Real, gamma2::Real, eps::Real)
        @argcheck(zero(gamma1) <= gamma1 < one(gamma1),
                  DomainError(gamma1, "gamma1 must be in [0, 1)"))
        @argcheck(zero(gamma2) <= gamma2 < one(gamma2),
                  DomainError(gamma2, "gamma2 must be in [0, 1)"))
        @argcheck(eps > zero(eps), DomainError(eps, "eps must be positive"))
        return new{typeof(gamma1), typeof(gamma2), typeof(eps)}(gamma1, gamma2, eps)
    end
end
function AdaptiveMomentGradient(; gamma1::Real = 0.99, gamma2::Real = 0,
                                eps::Real = 1e-8)::AdaptiveMomentGradient
    return AdaptiveMomentGradient(gamma1, gamma2, eps)
end
"""
    gradient_state_seed(grad::PlainGradient, w::AbstractVector)
    gradient_state_seed(grad::GradientMomentum, w::AbstractVector)
    gradient_state_seed(grad::RootMeanSquareGradient, w::AbstractVector)
    gradient_state_seed(grad::AdaptiveMomentGradient, w::AbstractVector)

The carrier a Gradient Transform keeps on the Rule State before the first row: `nothing`, the zero average, the zero squared average, or the pair.

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`transform_gradient!`](@ref)
"""
function gradient_state_seed(::PlainGradient, ::AbstractVector)
    return nothing
end
function gradient_state_seed(::GradientMomentum, w::AbstractVector)
    return zeros(eltype(w), length(w))
end
function gradient_state_seed(::RootMeanSquareGradient, w::AbstractVector)
    return zeros(eltype(w), length(w))
end
function gradient_state_seed(::AdaptiveMomentGradient, w::AbstractVector)
    return (zeros(eltype(w), length(w)), zeros(eltype(w), length(w)))
end
"""
    transform_gradient!(grad::PlainGradient, gs, g::AbstractVector)
    transform_gradient!(grad::GradientMomentum, v::AbstractVector, g::AbstractVector)
    transform_gradient!(grad::RootMeanSquareGradient, m::AbstractVector, g::AbstractVector)
    transform_gradient!(grad::AdaptiveMomentGradient, gs::Tuple, g::AbstractVector)

The transformed gradient of the period, the carrier written in place: `g` itself, the updated average `v`, `g` over the root of the updated squared average `m` plus `eps`, or the updated `v` over that root.

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`gradient_state_seed`](@ref)
  - [`MirrorDescent`](@ref)
"""
function transform_gradient!(::PlainGradient, ::Nothing, g::AbstractVector)
    return g
end
function transform_gradient!(grad::GradientMomentum, v::AbstractVector, g::AbstractVector)
    v .= grad.gamma1 .* v .+ (1 - grad.gamma1) .* g
    return copy(v)
end
function transform_gradient!(grad::RootMeanSquareGradient, m::AbstractVector,
                             g::AbstractVector)
    m .= grad.gamma2 .* m .+ (1 - grad.gamma2) .* g .^ 2
    return g ./ (sqrt.(m) .+ grad.eps)
end
function transform_gradient!(grad::AdaptiveMomentGradient, gs::Tuple, g::AbstractVector)
    v, m = gs
    v .= grad.gamma1 .* v .+ (1 - grad.gamma1) .* g
    m .= grad.gamma2 .* m .+ (1 - grad.gamma2) .* g .^ 2
    return v ./ (sqrt.(m) .+ grad.eps)
end
"""
    gradient_state_view(gs::Nothing, i)
    gradient_state_view(gs::AbstractVector, i)
    gradient_state_view(gs::Tuple, i)

A Gradient Transform's carrier sliced to the assets `i`, as a copy.

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`MirrorDescentState`](@ref)
"""
function gradient_state_view(::Nothing, ::Any)
    return nothing
end
function gradient_state_view(gs::AbstractVector, i)
    return gs[i]
end
function gradient_state_view(gs::Tuple, i)
    return (gs[1][i], gs[2][i])
end
"""
    copy_gradient_state(gs::Nothing)
    copy_gradient_state(gs::AbstractVector)
    copy_gradient_state(gs::Tuple)

A copy of a Gradient Transform's carrier sharing no array with it.

# Related

  - [`AbstractGradientTransform`](@ref)
  - [`MirrorDescentState`](@ref)
"""
function copy_gradient_state(::Nothing)
    return nothing
end
function copy_gradient_state(gs::AbstractVector)
    return copy(gs)
end
function copy_gradient_state(gs::Tuple)
    return (copy(gs[1]), copy(gs[2]))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the objectives a first-order Online Selection Rule steps on: the loss whose gradient at the iterate the mirror step reads.

# Interfaces

In order to implement a new objective, subtype `AbstractOnlineObjective` and implement:

  - `loss_gradient(obj::AbstractOnlineObjective, u::AbstractVector, x::AbstractVector, rows) -> AbstractVector`: The gradient of the period's loss at the iterate `u`, from the period's price relative `x` and the rows the head holds, `nothing` when the objective reads none.
  - `rows_needed(obj::AbstractOnlineObjective) -> Union{Nothing, Integer}`: The number of rows the objective reads at a step, `0` for one that reads none, which the rule that holds it answers as its own.

# Related

  - [`LogWealth`](@ref)
  - [`RiskLoss`](@ref)
  - [`MirrorDescent`](@ref)
"""
abstract type AbstractOnlineObjective <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The log-wealth objective of a first-order Online Selection Rule: the loss ``-\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle`` of the period, whose gradient is ``-\\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``, one row. The default of [`MirrorDescent`](@ref), and every rule of the literature.

# Examples

```jldoctest
julia> LogWealth()
LogWealth()
```

# Related

  - [`AbstractOnlineObjective`](@ref)
  - [`RiskLoss`](@ref)
  - [`MirrorDescent`](@ref)
"""
struct LogWealth <: AbstractOnlineObjective end
"""
$(DocStringExtensions.TYPEDEF)

The Risk Loss: a risk measure, or a scalarised vector of them, over the last `window` rows the head holds, as the objective of a first-order Online Selection Rule in place of log wealth.

# Mathematical definition

At period ``t`` the loss is ``\\rho_r(\\boldsymbol{w}; \\boldsymbol{X}_t)``, the measure evaluated at the iterate over the window ``\\boldsymbol{X}_t`` of the head's returns — returns, not price relatives, because a risk measure reads returns — and the step reads its gradient ``\\nabla_{\\boldsymbol{w}} \\rho_r(\\boldsymbol{w}_t; \\boldsymbol{X}_t)`` through [`risk_gradient`](@ref): the closed form where the library states one, the finite difference otherwise. The measure is resolved against the prior `pe` fits on the window at every step, so a [`Variance`](@ref) reads the window's covariance and a measure that holds its own matrix keeps it; the rule takes one step per period toward the measure's minimiser, re-estimated each period. A vector of measures is scalarised by `sca` at each element's `settings.scale`, so `[MeanReturn(), Variance()]` with the mean's scale negative is one step on the mean–variance utility.

A prior fitted on one row has no covariance, so the loss is read once the head holds two rows and the step before that is the identity on the iterate. The rows reach the loss through the head's rows buffer, capped at `window`, so the first `window - 1` steps read a shorter window than stated.

The regret theorems of the first-order rules are stated for a convex loss, and hold here for the measures that are convex in the weights: [`Variance`](@ref), [`StandardDeviation`](@ref), [`ConditionalValueatRisk`](@ref), [`EntropicValueatRisk`](@ref), [`WorstRealisation`](@ref), [`Range`](@ref), [`MaximumDrawdown`](@ref), [`AverageDrawdown`](@ref), [`ConditionalDrawdownatRisk`](@ref), [`EntropicDrawdownatRisk`](@ref), the low-order moment measures, and [`MeanReturn`](@ref), which is linear. A quantile measure — [`ValueatRisk`](@ref), [`DrawdownatRisk`](@ref) — a kurtosis, a skewness, or a ratio is not, and on it the step is a heuristic with no bound. A finite-difference gradient at a kink of a convex measure is the chord across it, a subgradient's neighbour, as [`risk_gradient`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskLoss(;
        r::BaseRM_VecBaseRM = Variance(),
        window::Integer = 20,
        sca::Scalariser = SumScalariser(),
        pe::AbstractPriorEstimator = EmpiricalPrior()
    ) -> RiskLoss

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise: a prior fitted on one row has no covariance, so a window of one is never read.

# Examples

```jldoctest
julia> RiskLoss()
RiskLoss
       r ┼ Variance
         │   settings ┼ RiskMeasureSettings
         │            │   scale ┼ Float64: 1.0
         │            │      ub ┼ nothing
         │            │     rke ┴ Bool: true
         │      sigma ┼ nothing
         │       chol ┼ nothing
         │         rc ┼ nothing
         │        alg ┴ SquaredSOCRiskExpr()
  window ┼ Int64: 20
     sca ┼ SumScalariser()
      pe ┼ EmpiricalPrior
         │           ce ┼ PortfolioOptimisersCovariance
         │              │   ce ┼ Covariance
         │              │      │    me ┼ SimpleExpectedReturns
         │              │      │       │   w ┴ nothing
         │              │      │    ce ┼ GeneralCovariance
         │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │              │      │       │    w ┴ nothing
         │              │      │   alg ┼ FullMoment()
         │              │      │     w ┴ nothing
         │              │   mp ┼ MatrixProcessing
         │              │      │     pdm ┼ Posdef
         │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │              │      │      dn ┼ nothing
         │              │      │      dt ┼ nothing
         │              │      │     alg ┼ nothing
         │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │           me ┼ SimpleExpectedReturns
         │              │   w ┴ nothing
         │      horizon ┼ nothing
         │   fill_limit ┴ nothing
```

# Related

  - [`AbstractOnlineObjective`](@ref)
  - [`LogWealth`](@ref)
  - [`MirrorDescent`](@ref)
  - [`risk_gradient`](@ref)
  - [`expected_risk`](@ref)
"""
struct RiskLoss{T1 <: BaseRM_VecBaseRM, T2 <: Integer, T3 <: Scalariser,
                T4 <: AbstractPriorEstimator} <: AbstractOnlineObjective
    """
    $(field_dict[:r])
    """
    r::T1
    """
    The number of rows of the head's returns buffer the loss is evaluated over.
    """
    window::T2
    """
    $(field_dict[:sca])
    """
    sca::T3
    """
    $(field_dict[:pe])
    """
    pe::T4
    function RiskLoss(r::T1, window::T2, sca::T3,
                      pe::T4) where {T1 <: BaseRM_VecBaseRM, T2 <: Integer,
                                     T3 <: Scalariser, T4 <: AbstractPriorEstimator}
        @argcheck(window >= 2, DomainError(window, "window must be at least 2"))
        return new{T1, T2, T3, T4}(r, window, sca, pe)
    end
end
function RiskLoss(; r::BaseRM_VecBaseRM = Variance(), window::Integer = 20,
                  sca::Scalariser = SumScalariser(),
                  pe::AbstractPriorEstimator = EmpiricalPrior())::RiskLoss
    return RiskLoss(r, window, sca, pe)
end
"""
    rows_needed(obj::LogWealth)
    rows_needed(obj::RiskLoss)
    rows_needed(alg::MirrorDescent)

The rows a first-order rule's objective reads at a step, which the rule answers as its own: none for log wealth, the window of a Risk Loss.

# Related

  - [`AbstractOnlineObjective`](@ref)
  - [`MirrorDescent`](@ref)
"""
function rows_needed(::LogWealth)
    return 0
end
function rows_needed(obj::RiskLoss)
    return obj.window
end
"""
    loss_gradient(obj::LogWealth, u::AbstractVector, x::AbstractVector, rows)
    loss_gradient(obj::RiskLoss, u::AbstractVector, x::AbstractVector, rows)

The gradient of the period's loss at the iterate `u`: ``-\\boldsymbol{x} / \\langle \\boldsymbol{u}, \\boldsymbol{x} \\rangle`` for log wealth, from the price relative alone; and [`risk_gradient`](@ref) of the Risk Loss's measure at `u`, resolved against its prior fitted on `rows`, or the zero vector while the head holds fewer than two rows.

# Related

  - [`AbstractOnlineObjective`](@ref)
  - [`MirrorDescent`](@ref)
  - [`risk_gradient`](@ref)
"""
function loss_gradient(::LogWealth, u::AbstractVector, x::AbstractVector, ::Any)
    return -x ./ LinearAlgebra.dot(u, x)
end
function loss_gradient(obj::RiskLoss, u::AbstractVector, ::AbstractVector, rows)
    if isnothing(rows) || size(rows, 1) < 2
        return zeros(eltype(u), length(u))
    end
    pr = prior(obj.pe, rows, nothing, nothing)
    return risk_gradient(obj.r, u, pr; sca = obj.sca)
end
"""
$(DocStringExtensions.TYPEDEF)

The carrier of [`MirrorDescent`](@ref): the unmixed iterate, the Start Allocation the rule restarts at, the schedule's statistic and the Gradient Transform's averages.

The iterate is what the mirror step reads and writes; the allocation the head plays is its uniform mix when `alpha` is positive, and the iterate itself otherwise.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`MirrorDescent`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct MirrorDescentState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The unmixed iterate ``\\boldsymbol{w}_t``, over the full pinned universe, summing to one.
    """
    u
    """
    The Start Allocation projected onto the Allocation Set, which a restart returns to.
    """
    w0
    """
    The schedule's statistic, or `nothing`.
    """
    s
    """
    The Gradient Transform's carrier, or `nothing`.
    """
    gs
end
function merge_states(::MirrorDescentState, ::MirrorDescentState)
    return throw(ArgumentError("a `MirrorDescentState` is not merged on its own: it holds an iterate that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::MirrorDescentState)
    return MirrorDescentState(x.n, copy(x.u), copy(x.w0), copy_column(x.s),
                              copy_gradient_state(x.gs))
end
function port_opt_view(x::MirrorDescentState, i, args...)
    return MirrorDescentState(x.n, renormalised_view(x.u, i), renormalised_view(x.w0, i),
                              copy_column(x.s), gradient_state_view(x.gs, i))
end
"""
$(DocStringExtensions.TYPEDEF)

The family's first-order Online Selection Rule: one mirror-descent step on the period's gradient, in the divergence its Projection Geometry holds, and the projection back onto the Allocation Set in the same geometry.

# Mathematical definition

With ``\\boldsymbol{g}_t = -\\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` the gradient of the period's log-wealth loss at the iterate, ``\\hat{\\boldsymbol{g}}_t`` its Gradient Transform and ``\\Psi`` the geometry's potential,

```math
\\begin{align}
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta_t \\langle \\hat{\\boldsymbol{g}}_t, \\boldsymbol{w} \\rangle + D_\\Psi(\\boldsymbol{w}, \\boldsymbol{w}_t) = \\mathrm{Proj}_\\Psi\\left( \\nabla \\Psi^* \\left( \\nabla \\Psi(\\boldsymbol{w}_t) - \\eta_t \\hat{\\boldsymbol{g}}_t \\right) \\right)\\,.
\\end{align}
```

Under [`EntropicProjection`](@ref) the step is the multiplicative update ``w_{t+1, i} \\propto w_{t, i} \\exp(\\eta x_{t, i} / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle)`` of Helmbold, Schapire, Singer and Warmuth (1998), the exponentiated gradient, with regret ``O(\\sqrt{T \\log N})`` at ``\\eta = r \\sqrt{2 \\log N / T}`` when every price relative is at least ``r``; under [`EuclideanProjection`](@ref) it is the additive step ``\\boldsymbol{w}_t + \\eta \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` followed by the projection onto the simplex, Zinkevich's (2003) online gradient descent and, up to the projection the 1997 paper assumes rather than enforces, the gradient projection of Helmbold and co-authors (1997), with regret ``O(\\sqrt{T N})``; under [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref) it is the barrier step of Abernethy, Lee and Tewari (2015), Zimmert and Seldin (2021) and Orseau, Lattimore and Legg (2017, §7), each a scalar root on the default set. Every geometry admits every Allocation Set the head admits; the fixed-horizon constants of the theorems need the horizon, which no online rule knows, and are documented formulas, and the anytime rates are [`InverseSquareRootRate`](@ref), [`SelfConfidentRate`](@ref) and [`DoublingTrickRate`](@ref) on `eta`.

`alpha` is the uniform mix of Helmbold and co-authors (1998, Theorem 4.2): the update reads the mixed price relatives ``\\tilde{\\boldsymbol{x}}_t = (1 - \\alpha / N) \\boldsymbol{x}_t + (\\alpha / N) \\boldsymbol{1}`` at the unmixed iterate, and the head plays ``\\tilde{\\boldsymbol{w}}_{t+1} = (1 - \\alpha) \\boldsymbol{w}_{t+1} + (\\alpha / N) \\boldsymbol{1}``; the carrier holds the unmixed iterate, so the mix is never inverted, and at `alpha = 0` the carrier is the played allocation and the rule is the plain step. The mix is a convex shift, not a projection, so it is admitted under every geometry; the theorem — ``O(T^{3/4})`` with no lower bound on the price relatives, at ``\\alpha = (N^2 \\log N / (8 T))^{1/4}`` — is the entropic map's, and the doubling trick of Corollary 4.3 sets ``\\alpha`` and ``\\eta`` per stage and takes precedence over this slot. The Start Allocation is played unmixed for the first period, as every rule plays it.

A schedule that names a restart makes the update of that period answer the Start Allocation projected onto the set, held on the carrier, and puts the carrier back at its seed with the period count kept, because the stages are cumulative; the Gradient Transform's averages restart with it. As the weighting of an [`ExpertMixture`](@ref) the rule moves the weight over the experts on their period returns. As an expert of a mixture under [`BlendPoint`](@ref) the rule reads its gradient at the mixture's played blend — the seven-argument [`online_update!`](@ref) hands it the Gradient Point — while stepping from its own iterate, the shared gradient of Zhang, Lu and Zhou (2018) and Zhao, Zhang, Zhang and Zhou (2020); on the head, and under [`OwnPoint`](@ref), the point is the iterate itself.

`obj` is the loss the gradient is taken of: [`LogWealth`](@ref), the period's ``-\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle`` of every rule above, or a [`RiskLoss`](@ref), a risk measure over the last `window` rows the head holds, whose gradient [`risk_gradient`](@ref) answers at the iterate; the rule's `rows_needed` is the objective's. The uniform mix is read by the log-wealth gradient alone, because a risk measure reads the returns and not the period's relative, and the mix of the played allocation applies under either.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MirrorDescent(;
        eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
        proj::Union{<:EuclideanProjection, <:EntropicProjection, <:TsallisProjection, <:LogBarrierProjection} = EntropicProjection(),
        alpha::Real = 0,
        obj::AbstractOnlineObjective = LogWealth(),
        grad::AbstractGradientTransform = PlainGradient()
    ) -> MirrorDescent

Keywords correspond to the struct's fields. The `proj` slot is bound to the geometries that are a scalar root on the default set; [`ExponentiatedGradient`](@ref) and [`GradientProjection`](@ref) fill it with the entropic and the Euclidean map.

## Validation

  - If `eta` is a number: `eta > 0`. A `DomainError` is thrown otherwise.
  - `0 <= alpha < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> MirrorDescent()
MirrorDescent
    eta ┼ Float64: 0.05
   proj ┼ EntropicProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┴ PlainGradient()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`ExponentiatedGradient`](@ref)
  - [`GradientProjection`](@ref)
  - [`MirrorDescentState`](@ref)
  - [`AbstractLearningRateSchedule`](@ref)
  - [`AbstractGradientTransform`](@ref)

# References

  - $(ref_dict[:helmbold1998])
  - $(ref_dict[:helmbold1997])
  - $(ref_dict[:zinkevich2003])
  - $(ref_dict[:abernethy2015])
  - $(ref_dict[:zimmertseldin2021])
  - $(ref_dict[:orseau2017])
"""
struct MirrorDescent{T1 <: Union{<:Real, <:AbstractLearningRateSchedule},
                     T2 <:
                     Union{<:EuclideanProjection, <:EntropicProjection, <:TsallisProjection,
                           <:LogBarrierProjection}, T3 <: Real,
                     T4 <: AbstractOnlineObjective, T5 <: AbstractGradientTransform} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Learning rate, a number or a Learning-Rate Schedule. Larger reacts faster and is less stable.
    """
    eta::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    """
    The uniform-mix share, in `[0, 1)`; `0` is the plain step.
    """
    alpha::T3
    """
    The objective the gradient is taken of: log wealth, or a Risk Loss over the head's rows.
    """
    obj::T4
    """
    The Gradient Transform applied before the mirror step.
    """
    grad::T5
    function MirrorDescent(eta::T1, proj::T2, alpha::T3, obj::T4,
                           grad::T5) where {T1 <:
                                            Union{<:Real, <:AbstractLearningRateSchedule},
                                            T2 <: Union{<:EuclideanProjection,
                                                        <:EntropicProjection, <:TsallisProjection,
                                                        <:LogBarrierProjection}, T3 <: Real,
                                            T4 <: AbstractOnlineObjective,
                                            T5 <: AbstractGradientTransform}
        if isa(eta, Real)
            @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        end
        @argcheck(zero(alpha) <= alpha < one(alpha),
                  DomainError(alpha, "alpha must be in [0, 1)"))
        return new{T1, T2, T3, T4, T5}(eta, proj, alpha, obj, grad)
    end
end
function MirrorDescent(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
                       proj::Union{<:EuclideanProjection, <:EntropicProjection,
                                   <:TsallisProjection, <:LogBarrierProjection} = EntropicProjection(),
                       alpha::Real = 0, obj::AbstractOnlineObjective = LogWealth(),
                       grad::AbstractGradientTransform = PlainGradient())::MirrorDescent
    return MirrorDescent(eta, proj, alpha, obj, grad)
end
function rule_state_seed(alg::MirrorDescent, w::AbstractVector)
    return MirrorDescentState(0, copy(w), copy(w), schedule_state_seed(alg.eta, w),
                              gradient_state_seed(alg.grad, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The played allocation of a [`MirrorDescent`](@ref) rule from its iterate: the iterate copied at `alpha = 0`, and its uniform mix ``(1 - \\alpha) \\boldsymbol{w} + (\\alpha / N) \\boldsymbol{1}`` otherwise.

# Related

  - [`MirrorDescent`](@ref)
"""
function played_allocation(u::AbstractVector, alpha::Real)
    if iszero(alpha)
        return copy(u)
    end
    return (1 - alpha) .* u .+ alpha / length(u)
end
function rows_needed(alg::MirrorDescent)
    return rows_needed(alg.obj)
end
function online_update!(alg::MirrorDescent, st::MirrorDescentState, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet)
    return online_update!(alg, st, w, x, rows, set, st.u)
end
# The seven-argument form is the primitive: the gradient is read at `point`, the rule's own
# unmixed iterate on the head and the mixture's played blend under `BlendPoint`.
function online_update!(alg::MirrorDescent, st::MirrorDescentState, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet,
                        point::AbstractVector)
    t = st.n + 1
    if restart(alg.eta, t)
        # The period count survives the restart: the schedule's stages are cumulative.
        return MirrorDescentState(t, copy(st.w0), st.w0,
                                  schedule_state_seed(alg.eta, st.w0),
                                  gradient_state_seed(alg.grad, st.w0)), copy(st.w0)
    end
    eta = learning_rate(alg.eta, t, st)
    alpha = mixing_share(alg.eta, t, alg.alpha)
    xm = iszero(alpha) ? x : (1 - alpha / length(x)) .* x .+ alpha / length(x)
    g = loss_gradient(alg.obj, point, xm, rows)
    q = mirror_step(alg.proj, st.u, eta .* transform_gradient!(alg.grad, st.gs, g))
    u = project(alg.proj, set, q, price_adjusted_allocation(w, x))
    s = schedule_update!(alg.eta, st.s, w, x)
    return MirrorDescentState(t, u, st.w0, s, st.gs), played_allocation(u, alpha)
end
"""
    ExponentiatedGradient(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, alpha::Real = 0, obj::AbstractOnlineObjective = LogWealth(), grad::AbstractGradientTransform = PlainGradient())

The exponentiated gradient of Helmbold, Schapire, Singer and Warmuth (1998): a [`MirrorDescent`](@ref) rule under [`EntropicProjection`](@ref) (EG).

The multiplicative update ``w_{t+1, i} \\propto w_{t, i} \\exp(\\eta x_{t, i} / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle)`` normalised to sum to one, which is the entropic projection onto the simplex; a momentum rule that increases the weight of whatever just did well, with regret ``O(\\sqrt{T \\log N})`` at a rate tuned to the horizon, and universal under the uniform mix `alpha` with the doubling trick. The default rate is the paper's ``0.05`` (§5.2: rates from ``0.01`` to ``0.15`` all do well, and a rate above one loses money on their two-stock case). As the weighting of an [`ExpertMixture`](@ref) it is the online gradient update over the expert-return vector.

# Examples

```jldoctest
julia> ExponentiatedGradient()
MirrorDescent
    eta ┼ Float64: 0.05
   proj ┼ EntropicProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┴ PlainGradient()
```

# Related

  - [`MirrorDescent`](@ref)
  - [`GradientProjection`](@ref)
  - [`EntropicProjection`](@ref)

# References

  - $(ref_dict[:helmbold1998])
"""
function ExponentiatedGradient(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
                               alpha::Real = 0, obj::AbstractOnlineObjective = LogWealth(),
                               grad::AbstractGradientTransform = PlainGradient())::MirrorDescent
    return MirrorDescent(; eta = eta, proj = EntropicProjection(), alpha = alpha, obj = obj,
                         grad = grad)
end
"""
    GradientProjection(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, alpha::Real = 0, obj::AbstractOnlineObjective = LogWealth(), grad::AbstractGradientTransform = PlainGradient())

The gradient projection of Helmbold, Schapire, Singer and Warmuth (1997), which is Zinkevich's (2003) online gradient descent on the simplex: a [`MirrorDescent`](@ref) rule under [`EuclideanProjection`](@ref) (GP, OGD).

The additive step ``\\boldsymbol{w}_t + \\eta \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` followed by the Euclidean projection onto the Allocation Set, which zeroes the entries the step pushes below its threshold. The 1997 paper subtracts the mean of the gradient to keep the budget and assumes non-negativity rather than enforcing it; the projection does both, and on the simplex is Zinkevich's fixed-rate step exactly, with regret ``O(\\sqrt{T})`` at ``\\eta \\propto 1 / \\sqrt{T}`` and the anytime rate [`InverseSquareRootRate`](@ref). Neither paper states a rate for the portfolio problem; the default is the entropic constructor's.

# Examples

```jldoctest
julia> GradientProjection()
MirrorDescent
    eta ┼ Float64: 0.05
   proj ┼ EuclideanProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┴ PlainGradient()
```

# Related

  - [`MirrorDescent`](@ref)
  - [`ExponentiatedGradient`](@ref)
  - [`EuclideanProjection`](@ref)

# References

  - $(ref_dict[:helmbold1997])
  - $(ref_dict[:zinkevich2003])
"""
function GradientProjection(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
                            alpha::Real = 0, obj::AbstractOnlineObjective = LogWealth(),
                            grad::AbstractGradientTransform = PlainGradient())::MirrorDescent
    return MirrorDescent(; eta = eta, proj = EuclideanProjection(), alpha = alpha,
                         obj = obj, grad = grad)
end
"""
    EGE(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, gamma1::Real = 0.99, alpha::Real = 0)

The exponentiated gradient with exponential-moving-average momentum of Li, Zheng, Chen, Wang and Xu (2022): [`ExponentiatedGradient`](@ref) under [`GradientMomentum`](@ref).

The paper's rate is Helmbold's ``2 r \\sqrt{2 \\log N / T}`` with the doubling trick when the horizon is unknown, which is [`DoublingTrickRate`](@ref) on `eta`; the default here is the entropic constructor's.

# Examples

```jldoctest
julia> EGE()
MirrorDescent
    eta ┼ Float64: 0.05
   proj ┼ EntropicProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┼ GradientMomentum
        │   gamma1 ┴ Float64: 0.99
```

# Related

  - [`MirrorDescent`](@ref)
  - [`GradientMomentum`](@ref)
  - [`EGR`](@ref)
  - [`EGA`](@ref)

# References

  - $(ref_dict[:li2022egm])
"""
function EGE(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
             gamma1::Real = 0.99, alpha::Real = 0)::MirrorDescent
    return ExponentiatedGradient(; eta = eta, alpha = alpha,
                                 grad = GradientMomentum(; gamma1 = gamma1))
end
"""
    EGR(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, gamma2::Real = 0, eps::Real = 1e-8, alpha::Real = 0)

The exponentiated gradient with root-mean-square rescaling of Li, Zheng, Chen, Wang and Xu (2022): [`ExponentiatedGradient`](@ref) under [`RootMeanSquareGradient`](@ref).

At the paper's recommended `gamma2 = 0`, the default, the transformed gradient is the sign of the gradient up to `eps`, the same at every asset, so the update moves nothing: the rule holds its iterate's normalisation. [`RootMeanSquareGradient`](@ref) states why.

# Examples

```jldoctest
julia> EGR()
MirrorDescent
    eta ┼ Float64: 0.05
   proj ┼ EntropicProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┼ RootMeanSquareGradient
        │   gamma2 ┼ Int64: 0
        │      eps ┴ Float64: 1.0e-8
```

# Related

  - [`MirrorDescent`](@ref)
  - [`RootMeanSquareGradient`](@ref)
  - [`EGE`](@ref)
  - [`EGA`](@ref)

# References

  - $(ref_dict[:li2022egm])
"""
function EGR(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, gamma2::Real = 0,
             eps::Real = 1e-8, alpha::Real = 0)::MirrorDescent
    return ExponentiatedGradient(; eta = eta, alpha = alpha,
                                 grad = RootMeanSquareGradient(; gamma2 = gamma2,
                                                               eps = eps))
end
"""
    EGA(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, gamma1::Real = 0.99, gamma2::Real = 0, eps::Real = 1e-8, alpha::Real = 0)

The exponentiated gradient with adaptive-moment rescaling of Li, Zheng, Chen, Wang and Xu (2022): [`ExponentiatedGradient`](@ref) under [`AdaptiveMomentGradient`](@ref).

# Examples

```jldoctest
julia> EGA()
MirrorDescent
    eta ┼ Float64: 0.05
   proj ┼ EntropicProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┼ AdaptiveMomentGradient
        │   gamma1 ┼ Float64: 0.99
        │   gamma2 ┼ Int64: 0
        │      eps ┴ Float64: 1.0e-8
```

# Related

  - [`MirrorDescent`](@ref)
  - [`AdaptiveMomentGradient`](@ref)
  - [`EGE`](@ref)
  - [`EGR`](@ref)

# References

  - $(ref_dict[:li2022egm])
"""
function EGA(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
             gamma1::Real = 0.99, gamma2::Real = 0, eps::Real = 1e-8,
             alpha::Real = 0)::MirrorDescent
    return ExponentiatedGradient(; eta = eta, alpha = alpha,
                                 grad = AdaptiveMomentGradient(; gamma1 = gamma1,
                                                               gamma2 = gamma2, eps = eps))
end
export MirrorDescent, ExponentiatedGradient, GradientProjection, EGE, EGR, EGA, LogWealth,
       RiskLoss, InverseSquareRootRate, DoublingTrickRate, SelfConfidentRate, PlainGradient,
       GradientMomentum, RootMeanSquareGradient, AdaptiveMomentGradient
public AbstractGradientTransform, gradient_state_seed, transform_gradient!,
       gradient_state_view, AbstractOnlineObjective, loss_gradient
