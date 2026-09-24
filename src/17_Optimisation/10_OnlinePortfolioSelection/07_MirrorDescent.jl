"""
$(DocStringExtensions.TYPEDEF)

Shrinks the learning rate with the square root of the period count, and needs no horizon.

The schedule reads the period count and nothing else. It is the schedule of Zinkevich (2003, Theorem 1) for online gradient descent, which gives an ``O(\\sqrt{T})`` regret under the Euclidean map. At ``c = \\sqrt{\\log N / (2 N)}`` it is the anytime Soft-Bayes rate of Orseau, Lattimore and Legg (2017, Theorem 10).

Under the entropic map, Helmbold, Schapire, Singer and Warmuth (1998, Theorem 4.1) tune a fixed rate, ``\\eta = 2 r \\sqrt{2 \\log N / T}``. That rate needs the horizon ``T`` and a lower bound ``r`` on each price relative as a fraction of the largest one of its period. An online rule knows neither, so the library states that rate here and does not implement it as a schedule.

# Mathematical definition

```math
\\begin{align}
\\eta_t &= \\frac{c}{\\sqrt{t}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_t_lr])
  - ``c > 0``: Scale of the rate.
  - $(math_dict[:t_period])

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
    The scale of the rate, the numerator of ``c / \\sqrt{t}``.
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

Runs the rule in stages of growing length, and restarts it at the Start Allocation after the last period of each stage.

This is the doubling trick of Helmbold, Schapire, Singer and Warmuth (1998, Corollary 4.3) for the exponentiated gradient with the uniform mix. Each stage sets its uniform-mix share and its rate from its own length, as their Theorem 4.2 sets them from the horizon. The schedule sets both, so a [`MirrorDescent`](@ref) that holds it does not read its own `alpha`.

From the uniform Start Allocation, the staged rule is universal. For every sequence of non-negative price relatives, with no lower bound on them, its regret against every constant rebalanced portfolio over ``T`` periods is at most ``6 N^2 \\log N (1 + (T / (2 N^2 \\log N))^{3/4})``. The bound holds under the entropic map. Under another map the schedule restarts the rule and sets its rate, and no bound is known.

# Mathematical definition

```math
\\begin{align}
T_0 &= \\lceil 2 N^2 \\log N \\rceil\\,, \\\\
T_i &= \\lceil 2^i N^2 \\log N \\rceil\\,, \\quad i > 0\\,, \\\\
\\alpha_i &= \\left( \\frac{N^2 \\log N}{8 T_i} \\right)^{1/4}\\,, \\\\
\\eta_i &= \\sqrt{\\frac{8 \\alpha_i^2 \\log N}{N^2 T_i}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:N])
  - ``i``: Stage index, counted from zero.
  - ``T_i``: Length of stage ``i``, in periods.
  - ``\\alpha_i``: Uniform-mix share of every period of stage ``i``.
  - ``\\eta_i``: Learning rate of every period of stage ``i``.

Stages ``0`` and ``1`` have the same length. At ``T_i \\geq 2 N^2 \\log N`` the share is at most ``1/2``, the range the theorem needs.

# Algorithm

At period `t` the schedule runs these steps.

 1. Find the stage that holds `t` with [`doubling_stage`](@ref), giving its length `len` and its last period `fin`.
 2. Compute the stage's share ``\\alpha_i`` from `len`. The rule reads it in place of its own `alpha`.
 3. Compute the stage's rate ``\\eta_i`` from the share and `len`.
 4. When `t == fin`, name a restart. The rule's update of period `t` then returns the Start Allocation, and puts its carrier back at its seed. The period count stays, because the stages are cumulative.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DoublingTrickRate(; N::Integer) -> DoublingTrickRate

Keywords correspond to the struct's fields. `N` is the number of assets the rule moves over, or the number of experts when the rule weights a mixture. It has no default, because the stage lengths depend on it.

## Validation

  - `N >= 2`. A `DomainError` is thrown otherwise, because ``\\log 1 = 0`` gives a stage of no periods.

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
    The number of assets. The stage lengths, the shares and the rates depend on it.
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

Finds the stage of a [`DoublingTrickRate`](@ref) that holds period `t`.

# Algorithm

 1. Compute `base = N^2 * log(N)`.
 2. Set the length `len` of stage 0 to `ceil(2 * base)`, and its last period `fin` to `len`.
 3. While `t > fin`, go to the next stage `i`. Set `len` to `ceil(2^i * base)`, and add `len` to `fin`.

# Returns

  - `(len, fin)::Tuple{Int, Int}`: The length of the stage that holds `t`, and its last period.

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

Reads the learning rate from the running excess of each period's best asset over the played allocation.

This is the self-confident rate of Orseau, Lattimore and Legg (2017, Theorem 6). The rule's carrier keeps the excess for the schedule, beside ``\\log N``. The excess is measured at the allocation the rule played, so under a [`MirrorDescent`](@ref) with a positive `alpha` it reads the mix.

Theorem 6 bounds the regret of the Soft-Bayes step, [`ExpectationMaximisation`](@ref), against every constant rebalanced portfolio. At the fixed rate ``\\sqrt{2 \\log N / C_1}``, with ``C_1`` the excess of the whole run, the bound is ``\\min\\{C_1, \\sqrt{2 C_1 \\log N} + 2 T \\log N / C_1\\}``. It is small when one asset is the best predictor for long stretches. The paper notes that the excess never falls, so the rate can be read online from the running excess, but it states no bound for that online rate. Under a [`MirrorDescent`](@ref) the rate is read online in the same way, and no bound is known either.

Under the online form of the Soft-Bayes step, this rate caps the pull towards the Start Allocation at ``\\sqrt{t / (t + 1)}``, as the paper advises ([`correction_ratio_cap`](@ref)). Without the cap, the rate stays almost constant while the mixture predicts well, and a weight can then decay exponentially.

The rate is capped at `eta_max`, because the excess is zero before the first row. The theorem needs a rate in ``(0, 1)``, and the paper states no cap, so the default cap is the end of that interval.

# Mathematical definition

```math
\\begin{align}
C_{1, t} &= \\sum_{s = 1}^{t} \\left( \\max_i \\frac{x_{s, i}}{\\langle \\boldsymbol{w}_s, \\boldsymbol{x}_s \\rangle} - 1 \\right)\\,, \\\\
\\eta_t &= \\min\\left( \\sqrt{\\frac{2 \\log N}{C_{1, t-1}}}, \\eta_{\\max} \\right)\\,.
\\end{align}
```

Where:

  - ``C_{1, t}``: First-order excess through period ``t``, zero at ``t = 0``.
  - ``x_{s, i}``: Price relative of asset ``i`` in period ``s``.
  - ``\\boldsymbol{x}_s``: Price relative vector of period ``s``.
  - ``\\boldsymbol{w}_s``: Allocation the rule played in period ``s``.
  - $(math_dict[:eta_t_lr])
  - ``\\eta_{\\max} > 0``: Cap on the rate.
  - $(math_dict[:N])

Each term of ``C_{1, t}`` is non-negative, because the largest price relative is at least their mean under ``\\boldsymbol{w}_s``. So the excess never falls, and the rate never rises. While the excess is zero, the rate is ``\\eta_{\\max}``.

# Algorithm

 1. Before the first row, seed the statistic `s = [0, log(N)]`.
 2. Read the rate of period `t` from `s`: `eta_max` while `s[1]` is not positive, and ``\\eta_t`` otherwise.
 3. After the rule's step of period `t`, add `maximum(x ./ dot(w, x)) - 1` to `s[1]`, with `w` the allocation the rule played in period `t` and `x` its price relative.

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
  - [`ExpectationMaximisation`](@ref)
  - [`correction_ratio_cap`](@ref)

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
function correction_ratio_cap(::SelfConfidentRate, t::Integer)
    return sqrt(t / (t + 1))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the state of a [`WindowedBestRate`](@ref) on the rule's carrier.

The state holds one expert allocation per rate, each an exponentiated-gradient run, and a ring of the experts' log period returns. The schedule sums the ring over the window.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`WindowedBestRate`](@ref)
  - [`MirrorDescentState`](@ref)
"""
@concrete struct WindowedBestRateState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The experts' allocations, `assets × rates`. Column `k` is the exponentiated-gradient run at the `k`-th rate, from the Start Allocation.
    """
    B
    """
    The experts' log period returns, `window × rates`. The row of period `t` overwrites row `mod1(t, window)` of this ring. Under the whole history the matrix is `1 × rates` and holds a running sum.
    """
    L
end
function Base.copy(x::WindowedBestRateState)
    return WindowedBestRateState(x.n, copy(x.B), copy(x.L))
end
function schedule_state_view(s::WindowedBestRateState, i)
    B = s.B[i, :]
    for k in axes(B, 2)
        B[:, k] = renormalised_view(view(s.B, :, k), i)
    end
    return WindowedBestRateState(s.n, B, copy(s.L))
end
"""
$(DocStringExtensions.TYPEDEF)

Chooses the learning rate of each period from a set of rates, by the recent wealth of one exponentiated-gradient run per rate.

This is the moving-window adaptive rate of Zhang, Lin, Zheng and Yang (2022). Each rate in `etas` names an expert, the exponentiated gradient run at that rate from the Start Allocation. The rate of the period is the rate of the expert with the largest wealth over the last `window` periods. When `window` is `nothing`, the schedule compares the wealth over the whole history. The paper calls the first rule MAEG and the second AEG.

The experts are a statistic of the run, and the rule never plays them. Each expert takes the paper's plain entropic step over the simplex, from the rule's Start Allocation, whatever the rule's own geometry, Allocation Set, uniform mix, Gradient Transform and objective. The rule applies the chosen rate to its own step. So on another geometry the schedule chooses the rate by a replay of the exponentiated gradient.

The paper takes the rate set `0.001:0.001:0.2` from the practice of Helmbold, Schapire, Singer and Warmuth (1998). It chose the window `30` over `7` on its own data. Over windows from `2` to `50`, it reports that the final wealth stays within a small range, and that the window must stay within a reasonable range. A period costs one entropic step per rate and one sum over the ring, which is `O((N + w) K)` at `K` rates. The online form of [`ExpectationMaximisation`](@ref) refuses this schedule by name, because that form also reads the rate of the next period.

# Mathematical definition

```math
\\begin{align}
t' &= \\max(1, t - w + 1)\\,, \\\\
S_t(\\eta) &= \\prod_{\\tau = t'}^{t} \\langle \\boldsymbol{b}_\\tau(\\eta), \\boldsymbol{x}_\\tau \\rangle\\,, \\\\
\\eta_t &= \\underset{\\eta \\in H}{\\arg\\max} \\; S_t(\\eta)\\,.
\\end{align}
```

Where:

  - ``H``: Set of rates, the field `etas`.
  - ``w``: Window, in periods. Under the whole history ``t' = 1``.
  - ``\\boldsymbol{b}_\\tau(\\eta)``: Allocation of the expert at rate ``\\eta`` in period ``\\tau``.
  - $(math_dict[:x_t_rel])
  - ``S_t(\\eta)``: Wealth of the expert at rate ``\\eta`` over the window that ends at period ``t``.
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:t_period])

The rate of the update at period ``t`` reads ``\\boldsymbol{x}_t``. It is the paper's ``\\eta_{t+1}``, which forms ``\\boldsymbol{b}_{t+1}`` from ``\\boldsymbol{b}_t`` and ``\\boldsymbol{x}_t``. At a tie the first rate of `etas` wins.

# Algorithm

 1. Before the first row, seed the state: every column of `B` is the Start Allocation, and `L` is zero, with `window` rows, or one row under the whole history.
 2. The schedule answers `true` to [`reads_period_row`](@ref), so the rule writes the state before it reads the rate. At period `n`, find the row `row` of `L` with [`ring_row`](@ref).
 3. For the expert `b` at each rate `eta`, compute its period return `r = dot(b, x)`. Write `log(r)` into `L[row, k]`: overwrite it under a window, and add it to the running sum under the whole history.
 4. Step `b` to `b .* exp.(eta .* x ./ r)`, and divide it by its sum.
 5. Sum each column of `L`, and take the rate of the column with the largest sum. The sum is in logs, so the comparison never overflows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedBestRate(; etas::AbstractVector{<:Real} = 0.001:0.001:0.2, window::Option{<:Integer} = 30) -> WindowedBestRate

Keywords correspond to the struct's fields.

## Validation

  - `etas`: non-empty, and every element is positive and finite.
  - `window >= 1` when it is a number. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> WindowedBestRate()
WindowedBestRate
    etas ┼ 200-element StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
  window ┴ Int64: 30
```

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`WindowedBestRateState`](@ref)
  - [`MirrorDescent`](@ref)
  - [`MAEG`](@ref)
  - [`AEG`](@ref)

# References

  - $(ref_dict[:zhang2022maeg])
  - $(ref_dict[:helmbold1998])
"""
struct WindowedBestRate{T1 <: AbstractVector{<:Real}, T2 <: Option{<:Integer}} <:
       AbstractLearningRateSchedule
    """
    The rates the experts run at, one exponentiated-gradient run each.
    """
    etas::T1
    """
    The number of periods over which the schedule compares the experts' wealth. `nothing` compares the whole history.
    """
    window::T2
    function WindowedBestRate(etas::AbstractVector{<:Real}, window::Option{<:Integer})
        assert_nonempty_gt0_finite_val(etas, :etas)
        if !isnothing(window)
            @argcheck(window >= 1, DomainError(window, "window must be at least 1"))
        end
        return new{typeof(etas), typeof(window)}(etas, window)
    end
end
function WindowedBestRate(; etas::AbstractVector{<:Real} = 0.001:0.001:0.2,
                          window::Option{<:Integer} = 30)::WindowedBestRate
    return WindowedBestRate(etas, window)
end
function reads_period_row(::WindowedBestRate)
    return true
end
function schedule_state_seed(sched::WindowedBestRate, w::AbstractVector)
    K = length(sched.etas)
    rows = isnothing(sched.window) ? 1 : sched.window
    return WindowedBestRateState(0, repeat(w, 1, K), zeros(eltype(w), rows, K))
end
"""
    ring_row(window::Nothing, n::Integer)
    ring_row(window::Integer, n::Integer)

Finds the row of a [`WindowedBestRateState`](@ref)'s ring that period `n` writes.

Under a window the row is `mod1(n, window)`. Under the whole history it is row `1`, the one row of the running sum.

# Related

  - [`WindowedBestRate`](@ref)
  - [`WindowedBestRateState`](@ref)
"""
function ring_row(::Nothing, ::Integer)
    return 1
end
function ring_row(window::Integer, n::Integer)
    return mod1(n, window)
end
function schedule_update!(sched::WindowedBestRate, s::WindowedBestRateState,
                          ::AbstractVector, x::AbstractVector)
    n = s.n + 1
    row = ring_row(sched.window, n)
    # A ring row is overwritten; the one row of the whole history accumulates.
    keep = isnothing(sched.window)
    for (k, eta) in enumerate(sched.etas)
        b = view(s.B, :, k)
        r = LinearAlgebra.dot(b, x)
        s.L[row, k] = ifelse(keep, s.L[row, k], zero(r)) + log(r)
        b .*= exp.(eta .* x ./ r)
        b ./= sum(b)
    end
    return WindowedBestRateState(n, s.B, s.L)
end
function learning_rate(sched::WindowedBestRate, ::Integer, st)
    return sched.etas[argmax(vec(sum(st.s.L; dims = 1)))]
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Gradient Transforms, which change the gradient of a first-order rule before its mirror step.

A [`MirrorDescent`](@ref) rule applies its transform to the gradient of every period. The momentum transforms of Li, Zheng, Chen, Wang and Xu (2022) change the exponent of the multiplicative update, and do not change the form of the update.

# Interfaces

In order to implement a new transform, subtype `AbstractGradientTransform` and implement:

  - `gradient_state_seed(grad::AbstractGradientTransform, w::AbstractVector)`: The carrier the transform keeps on the Rule State before the first row, or `nothing`, the default. `w` is the Start Allocation, and the carrier takes its length and its element type.
  - `transform_gradient!(grad::AbstractGradientTransform, gs, g::AbstractVector) -> AbstractVector`: The transformed gradient of the period, computed from the raw gradient `g`. It writes the carrier `gs` in place.

The head's state already slices and copies a carrier that is `nothing`, a vector or a pair of vectors. A carrier of another shape needs its own methods of the two private helpers [`gradient_state_view`](@ref) and [`copy_gradient_state`](@ref).

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

Leaves the gradient unchanged, so the mirror step reads the period's gradient as it is.

It is the default Gradient Transform of [`MirrorDescent`](@ref).

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

Replaces the gradient with its exponential moving average, which gives the step a momentum.

This is Eq. 7 of Li, Zheng, Chen, Wang and Xu (2022). Under the entropic map it is the paper's EGE, and its Theorem 1 gives the regret ``O(\\sqrt{T \\log N})`` of the exponentiated gradient.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{v}_{t+1} &= \\gamma_1 \\boldsymbol{v}_t + (1 - \\gamma_1) \\boldsymbol{g}_t\\,, \\\\
\\hat{\\boldsymbol{g}}_t &= \\boldsymbol{v}_{t+1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_t_mom])
  - $(math_dict[:gamma1_mom])
  - $(math_dict[:g_t_loss])
  - $(math_dict[:ghat_t])

The average starts at zero, so the first transformed gradient is ``(1 - \\gamma_1) \\boldsymbol{g}_1``. The paper applies no bias correction, and neither does this transform.

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

Divides the gradient by the root of a moving average of its square, per asset.

This is Eq. 8 of Li, Zheng, Chen, Wang and Xu (2022). Under the entropic map it is the paper's EGR.

**At the default ``\\gamma_2 = 0``, which the paper recommends, the rule under [`LogWealth`](@ref) stays at its Start Allocation, up to the offset ``\\epsilon``.** At ``\\gamma_2 = 0`` the transformed gradient is the sign of the gradient, up to ``\\epsilon``. Every entry of the log-wealth gradient is negative, so the step moves every asset by the same amount, and the normalisation undoes it. The rule is then the constant rebalanced portfolio at its Start Allocation, not buy-and-hold. A positive ``\\gamma_2`` gives the average a memory, and the rule then moves.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{m}_{t+1} &= \\gamma_2 \\boldsymbol{m}_t + (1 - \\gamma_2) \\boldsymbol{g}_t^2\\,, \\\\
\\hat{\\boldsymbol{g}}_t &= \\frac{\\boldsymbol{g}_t}{\\sqrt{\\boldsymbol{m}_{t+1}} + \\epsilon}\\,.
\\end{align}
```

Where:

  - $(math_dict[:m_t_mom])
  - $(math_dict[:gamma2_mom])
  - $(math_dict[:g_t_loss])
  - $(math_dict[:ghat_t])
  - $(math_dict[:eps_mom])

The square, the root and the division are per asset. At ``\\gamma_2 = 0``, ``\\hat{\\boldsymbol{g}}_t = \\boldsymbol{g}_t / (\\lvert \\boldsymbol{g}_t \\rvert + \\epsilon)``.

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

Divides the moving average of the gradient by the root of the moving average of its square, per asset.

This is Eq. 9 of Li, Zheng, Chen, Wang and Xu (2022), with both averages of [`GradientMomentum`](@ref) and [`RootMeanSquareGradient`](@ref) and no bias correction. Under the entropic map it is the paper's EGA.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{v}_{t+1} &= \\gamma_1 \\boldsymbol{v}_t + (1 - \\gamma_1) \\boldsymbol{g}_t\\,, \\\\
\\boldsymbol{m}_{t+1} &= \\gamma_2 \\boldsymbol{m}_t + (1 - \\gamma_2) \\boldsymbol{g}_t^2\\,, \\\\
\\hat{\\boldsymbol{g}}_t &= \\frac{\\boldsymbol{v}_{t+1}}{\\sqrt{\\boldsymbol{m}_{t+1}} + \\epsilon}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_t_mom])
  - $(math_dict[:m_t_mom])
  - $(math_dict[:gamma1_mom])
  - $(math_dict[:gamma2_mom])
  - $(math_dict[:g_t_loss])
  - $(math_dict[:ghat_t])
  - $(math_dict[:eps_mom])

At the default ``\\gamma_2 = 0``, which the paper recommends, the divisor is ``\\lvert \\boldsymbol{g}_t \\rvert + \\epsilon``. The step then reads the momentum in units of the size of the current gradient.

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

Seeds the carrier that a Gradient Transform keeps on the Rule State before the first row.

The seed is `nothing` for [`PlainGradient`](@ref), the zero average for [`GradientMomentum`](@ref), the zero squared average for [`RootMeanSquareGradient`](@ref), and the pair of both for [`AdaptiveMomentGradient`](@ref).

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

Computes the transformed gradient of the period, and writes the updated carrier in place.

[`PlainGradient`](@ref) returns `g` itself. [`GradientMomentum`](@ref) returns a copy of the updated average `v`. [`RootMeanSquareGradient`](@ref) returns `g` over the root of the updated squared average `m`, plus `eps`. [`AdaptiveMomentGradient`](@ref) returns the updated `v` over that root plus `eps`. The type docstrings state the formulas.

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

Slices a Gradient Transform's carrier to the assets `i`, as a copy.

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

Copies a Gradient Transform's carrier, so that the copy shares no array with it.

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

Abstract supertype for the objectives of a first-order Online Selection Rule.

The objective is the loss of each period. The mirror step reads the gradient of that loss at the iterate.

# Interfaces

In order to implement a new objective, subtype `AbstractOnlineObjective` and implement:

  - `loss_gradient(obj::AbstractOnlineObjective, u::AbstractVector, x::AbstractVector, rows) -> AbstractVector`: The gradient of the period's loss at the iterate `u`. It reads the period's price relative `x` and the rows the head holds, which are `nothing` when the objective reads no rows.
  - `rows_needed(obj::AbstractOnlineObjective) -> Union{Nothing, Integer}`: The number of rows the objective reads at a step, `0` when it reads none. The rule that holds the objective answers this number as its own.

# Related

  - [`LogWealth`](@ref)
  - [`RiskLoss`](@ref)
  - [`MirrorDescent`](@ref)
"""
abstract type AbstractOnlineObjective <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Takes the negative log of the period's wealth as the loss of a first-order Online Selection Rule.

It reads the period's price relative and no rows. It is the default objective of [`MirrorDescent`](@ref), and the objective of every rule that the docstring of [`MirrorDescent`](@ref) cites.

# Mathematical definition

```math
\\begin{align}
\\ell_t(\\boldsymbol{w}) &= -\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle\\,, \\\\
\\boldsymbol{g}_t &= \\nabla \\ell_t(\\boldsymbol{w}_t) = -\\frac{\\boldsymbol{x}_t}{\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle}\\,.
\\end{align}
```

Where:

  - ``\\ell_t``: Loss of period ``t``.
  - ``\\boldsymbol{w}``: Allocation at which the loss is evaluated.
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:g_t_loss])

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

Takes a risk measure over the last `window` rows the head holds as the loss of a first-order Online Selection Rule.

The Risk Loss replaces log wealth as the objective. `r` is one risk measure or a vector of them. The Risk Loss reads returns and not price relatives, because a risk measure reads returns. At every step the prior `pe` is fitted on the window, and the measure is resolved against that prior. So a [`Variance`](@ref) reads the covariance of the window, and a measure that holds its own matrix keeps it. The rule takes one step per period towards the minimiser of the measure, which it estimates again each period.

`sca` scalarises a vector of measures at the `settings.scale` of each measure. So `[MeanReturn(), Variance()]`, with a negative scale on the mean, gives one step on the mean-variance utility.

A prior fitted on one row has no covariance. So the loss starts when the head holds two rows, and the step before that leaves the iterate unchanged. The head's rows buffer holds at most `window` rows, so the first `window - 1` steps read a shorter window.

**A risk gradient is small, so a rule on a Risk Loss needs a much larger rate than a log-wealth rule.** A variance gradient ``2 \\boldsymbol{\\Sigma} \\boldsymbol{w}`` has the size of one period's variance. Take 1000 periods of returns `randn(StableRNG(7), 1000, 4) .* [0.01 0.02 0.03 0.04]`, a window of 20 and the uniform allocation. Over every window, the log-wealth gradient is then between 1100 and 6400 times larger than the variance gradient. The entropic rule on [`Variance`](@ref) at the log-wealth default `eta = 0.05` ends 0.005 from its uniform start, and the minimum-variance allocation is 0.45 away. At `eta = 100` the rule ends within 0.06 of that allocation.

The prior is fitted on the head's rows carrier as a batch prior is fitted on any carrier. The carrier holds `NaN` where there was no return, and the active mask beside it. So under a time-varying panel the prior reduces to the Coverage Universe of the window, and answers `NaN` at an asset that the window does not cover. The loss reads its gradient on the Investable Mask of that result, and writes zero at every other leg. The window holds no data about such a leg, so the loss does not move it, and the projection alone places it. A plain `pe` therefore never meets a constant column. A leg unlisted for a part of the window is outside the Coverage Universe of a plain prior until the window clears that part. A mask-aware prior takes the leg in after its own warm-up.

The regret theorems of the first-order rules need a convex loss. So they hold here for the measures that are convex in the weights: [`Variance`](@ref), [`StandardDeviation`](@ref), [`ConditionalValueatRisk`](@ref), [`EntropicValueatRisk`](@ref), [`WorstRealisation`](@ref), [`Range`](@ref), [`MaximumDrawdown`](@ref), [`AverageDrawdown`](@ref), [`ConditionalDrawdownatRisk`](@ref), [`EntropicDrawdownatRisk`](@ref), the low-order moment measures, and [`MeanReturn`](@ref), which is linear. A quantile measure such as [`ValueatRisk`](@ref) or [`DrawdownatRisk`](@ref), a kurtosis, a skewness and a ratio are not convex, and on them the step is a heuristic with no bound. At a kink of a convex measure, a finite-difference gradient is the chord across the kink, near a subgradient, as [`risk_gradient`](@ref) states.

# Mathematical definition

```math
\\begin{align}
\\ell_t(\\boldsymbol{w}) &= \\rho(\\boldsymbol{w}; \\mathbf{X}_t)\\,, \\\\
\\boldsymbol{g}_t &= \\nabla_{\\boldsymbol{w}} \\rho(\\boldsymbol{w}_t; \\mathbf{X}_t)\\,.
\\end{align}
```

Where:

  - ``\\ell_t``: Loss of period ``t``.
  - ``\\rho``: Risk measure, resolved against the prior fitted on ``\\mathbf{X}_t``. A vector of measures is their scalarisation.
  - ``\\mathbf{X}_t``: Returns of the last ``\\min(t, W)`` periods through period ``t``, with ``W`` the field `window`.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:g_t_loss])

[`risk_gradient`](@ref) computes ``\\boldsymbol{g}_t``, in closed form where the library states one and by a finite difference otherwise. [`loss_gradient`](@ref) states the steps.

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

  - `window >= 2`. A `DomainError` is thrown otherwise, because a prior fitted on one row has no covariance, and the loss never reads a window of one row.

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
    The number of rows of the head's returns buffer that the loss reads.
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

Answers the number of rows that a first-order rule's objective reads at a step.

The rule answers the number of its objective as its own. Log wealth reads no rows, so its number is `0`. A Risk Loss reads its `window`.

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

Computes the gradient of the period's loss at the iterate `u`.

For [`LogWealth`](@ref) the gradient reads the price relative `x` alone. For a [`RiskLoss`](@ref) it reads the rows carrier `rows`. The type docstrings state the two losses and their gradients.

# Algorithm

The [`LogWealth`](@ref) method returns `-x ./ dot(u, x)`. The [`RiskLoss`](@ref) method runs these steps.

 1. When `rows` is `nothing`, or holds fewer than two rows, return the zero vector.
 2. Fit the prior `pr` of the loss on `rows`, as a batch prior is fitted, on the rows of the carrier and its Asset Panel. The prior answers `NaN` at an asset outside its Coverage Universe.
 3. Read the Investable Mask of `pr`.
 4. Compute the gradient of the measure on that mask with [`investable_risk_gradient`](@ref). It is zero at every other leg.

# Related

  - [`AbstractOnlineObjective`](@ref)
  - [`MirrorDescent`](@ref)
  - [`risk_gradient`](@ref)
"""
function loss_gradient(::LogWealth, u::AbstractVector, x::AbstractVector, ::Any)
    return -x ./ LinearAlgebra.dot(u, x)
end
function loss_gradient(obj::RiskLoss, u::AbstractVector, ::AbstractVector,
                       rows::Option{<:ReturnsResult})
    if isnothing(rows) || size(rows.X, 1) < 2
        return zeros(eltype(u), length(u))
    end
    pr = prior(obj.pe, rows)
    return investable_risk_gradient(obj, u, pr, investable_mask(pr))
end
"""
    investable_risk_gradient(obj::RiskLoss, u::AbstractVector, pr::AbstractPriorResult, imsk::Nothing)
    investable_risk_gradient(obj::RiskLoss, u::AbstractVector, pr::AbstractPriorResult, imsk::BitVector)

Computes the gradient of a Risk Loss on the Investable Mask of its prior result.

A leg that the window does not cover gets a zero gradient, so the loss does not move it.

# Algorithm

When the mask is `nothing`, every asset is priced, and the method returns [`risk_gradient`](@ref) of the measure at `u`. Otherwise it runs these steps.

 1. Find the indices `idx` of the assets on the mask.
 2. Make the zero vector `g` of the length of `u`.
 3. Write [`risk_gradient`](@ref) of the measure at `u[idx]`, against the prior viewed at `idx`, into `g[idx]`. The slice of `u` is not renormalised.

# Related

  - [`loss_gradient`](@ref)
  - [`risk_gradient`](@ref)
  - [`investable_mask`](@ref)
"""
function investable_risk_gradient(obj::RiskLoss, u::AbstractVector, pr::AbstractPriorResult,
                                  ::Nothing)
    return risk_gradient(obj.r, u, pr; sca = obj.sca)
end
function investable_risk_gradient(obj::RiskLoss, u::AbstractVector, pr::AbstractPriorResult,
                                  imsk::BitVector)
    idx = findall(imsk)
    g = zeros(eltype(u), length(u))
    g[idx] .= risk_gradient(obj.r, u[idx], port_opt_view(pr, idx); sca = obj.sca)
    return g
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the state of a [`MirrorDescent`](@ref) rule between its updates.

The state holds the unmixed iterate, the Start Allocation that a restart returns to, the statistic of the schedule and the averages of the Gradient Transform. The mirror step reads and writes the iterate. The head plays the uniform mix of the iterate when `alpha` is positive, and the iterate itself otherwise.

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
    The unmixed iterate ``\\boldsymbol{w}_t``, over the full pinned universe. It sums to one.
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
                              schedule_state_view(x.s, i), gradient_state_view(x.gs, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Takes one mirror-descent step on the period's gradient, then projects the result back onto the Allocation Set.

This is the first-order Online Selection Rule of the family. The step and the projection both use the divergence of the rule's Projection Geometry.

Each geometry gives a published rule.

  - Under [`EntropicProjection`](@ref) the step is the multiplicative update of Helmbold, Schapire, Singer and Warmuth (1998), the exponentiated gradient. Its regret is ``O(\\sqrt{T \\log N})`` at ``\\eta = 2 r \\sqrt{2 \\log N / T}``, when every price relative is at least ``r`` times the largest one of its period.
  - Under [`EuclideanProjection`](@ref) the step is additive, and the projection onto the simplex follows it. This is the online gradient descent of Zinkevich (2003), with regret ``O(\\sqrt{T N})``. It is also the gradient projection of Helmbold and co-authors (1997), except that the 1997 paper assumes the projection and does not enforce it.
  - Under [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref) the step is the barrier step of Abernethy, Lee and Tewari (2015), Zimmert and Seldin (2021) and Orseau, Lattimore and Legg (2017, §7). On the default set each projection is a scalar root.

The Euclidean map admits every Allocation Set the head admits. The other three maps are defined on the positive orthant, so they refuse a set with a negative lower bound. The tuned rates of the theorems need the horizon, which an online rule does not know, so the library states them as formulas only. The anytime rates are [`InverseSquareRootRate`](@ref), [`SelfConfidentRate`](@ref) and [`DoublingTrickRate`](@ref), on `eta`.

`alpha` is the uniform mix of Helmbold and co-authors (1998, Theorem 4.2). The update reads the mixed price relatives at the unmixed iterate, and the head plays the mix of the new iterate. [`mixed_relatives`](@ref) states how the mix scales to the paper's relatives, which have a period maximum of one. The carrier holds the unmixed iterate, so the rule never inverts the mix. At `alpha = 0` the played allocation is the iterate, and the rule is the plain step. The mix is a convex shift and not a projection, so every geometry admits it. The theorem, ``O(T^{3/4})`` regret at ``\\alpha = (N^2 \\log N / (8 T))^{1/4}`` with no lower bound on the price relatives, holds under the entropic map. The doubling trick of Corollary 4.3 sets ``\\alpha`` and ``\\eta`` per stage, and its share takes precedence over `alpha`. The rule plays the Start Allocation unmixed in the first period, as every rule does.

The mix lies in the Allocation Set when the uniform allocation does. Otherwise [`reprojection`](@ref) projects it onto the set again, in the rule's geometry. A bounded set that excludes the uniform allocation, a turnover ceiling and a MIP kind are such sets. On the default set the reprojection costs nothing.

A schedule can name a restart. The update of that period then returns the Start Allocation, re-entered onto the set from the book the fund holds by [`reprojection`](@ref). The update puts the carrier back at its seed, and the averages of the Gradient Transform restart with it. The period count stays, because the stages are cumulative.

The rule can also serve inside a mixture. As the weighting of an [`ExpertMixture`](@ref), it moves the weight over the experts on their period returns. As an expert of a mixture under [`BlendPoint`](@ref), the rule reads its gradient at the blend that the mixture played, and steps from its own iterate. The seven-argument [`online_update!`](@ref) hands it that Gradient Point. This is the shared gradient of Zhang, Lu and Zhou (2018) and of Zhao, Zhang, Zhang and Zhou (2020). On the head, and under [`OwnPoint`](@ref), the point is the iterate.

`obj` is the loss. [`LogWealth`](@ref) is the loss of every rule above. A [`RiskLoss`](@ref) is a risk measure over the last `window` rows the head holds. The rule reads as many rows as its objective. Only the log-wealth gradient reads the mixed price relatives, because a risk measure reads returns and not price relatives. The mix of the played allocation applies under both.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{x}}_t &= \\left(1 - \\frac{\\alpha}{N}\\right) \\boldsymbol{x}_t + \\frac{\\alpha}{N} \\max_i x_{t, i} \\boldsymbol{1}\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta_t \\langle \\hat{\\boldsymbol{g}}_t, \\boldsymbol{w} \\rangle + D_\\Psi(\\boldsymbol{w}, \\boldsymbol{w}_t) = \\mathrm{Proj}_\\Psi\\left( \\nabla \\Psi^* \\left( \\nabla \\Psi(\\boldsymbol{w}_t) - \\eta_t \\hat{\\boldsymbol{g}}_t \\right) \\right)\\,, \\\\
\\tilde{\\boldsymbol{w}}_{t+1} &= (1 - \\alpha) \\boldsymbol{w}_{t+1} + \\frac{\\alpha}{N} \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_t_rel])
  - ``x_{t, i}``: Price relative of asset ``i`` in period ``t``.
  - ``\\tilde{\\boldsymbol{x}}_t``: Mixed price relative vector of period ``t``. The log-wealth gradient reads it in place of ``\\boldsymbol{x}_t``.
  - $(math_dict[:alpha_mix])
  - $(math_dict[:N])
  - ``\\boldsymbol{1}``: Vector of ones.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:ghat_t]) It is the Gradient Transform of ``\\boldsymbol{g}_t``, the gradient of the loss.
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:W_aset])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:D_Psi_breg])
  - ``\\Psi^*``: Convex conjugate of ``\\Psi``.
  - ``\\mathrm{Proj}_\\Psi``: Projection onto ``\\mathcal{W}`` in the divergence ``D_\\Psi``.
  - ``\\tilde{\\boldsymbol{w}}_{t+1}``: Allocation the head plays in period ``t + 1``.

Under the entropic map the step is ``w_{t+1, i} \\propto w_{t, i} \\exp(-\\eta_t \\hat{g}_{t, i})``. Under the Euclidean map it is ``\\mathrm{Proj}_{\\mathcal{W}}(\\boldsymbol{w}_t - \\eta_t \\hat{\\boldsymbol{g}}_t)``. At ``\\alpha = 0``, ``\\tilde{\\boldsymbol{x}}_t = \\boldsymbol{x}_t`` and ``\\tilde{\\boldsymbol{w}}_{t+1} = \\boldsymbol{w}_{t+1}``.

# Algorithm

The seven-argument [`online_update!`](@ref) runs these steps at the period's row `x`. The six-argument form calls it with the iterate as the Gradient Point `point`.

 1. Set the period `t = st.n + 1`, and the Price-Adjusted Allocation `wh` of the book `w` after the row.
 2. When the schedule names a restart at `t`, re-enter the Start Allocation `st.w0` onto the set from `wh` with [`reprojection`](@ref). Put the statistic and the carrier of the Gradient Transform back at their seeds, keep `t`, and return that allocation. Stop.
 3. When the schedule reads the period's row, write its statistic from `w` and `x`.
 4. Read the rate `eta` and the share `alpha` of period `t`. A schedule's share takes precedence over the field `alpha`.
 5. Mix the price relatives into `xm` with [`mixed_relatives`](@ref).
 6. Take the gradient `g` of the loss at `point`, from `xm` and the head's `rows`.
 7. Transform `g` with the Gradient Transform, which updates its carrier.
 8. Take the mirror step `q` from the iterate `st.u`, on `eta` times the transformed gradient.
 9. Project `q` onto the set from `wh`, which gives the new iterate `u`.
10. When the schedule reads the past alone, write its statistic from `w` and `x`.
11. Form the played allocation from `u` with [`played_allocation`](@ref). When `alpha` is positive, project it with [`reprojection`](@ref).
12. Return the new state and the played allocation.

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

Keywords correspond to the struct's fields. The `proj` slot takes only the geometries whose projection onto the default set is a scalar root. [`ExponentiatedGradient`](@ref) fills it with the entropic map, and [`GradientProjection`](@ref) with the Euclidean map.

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
    The learning rate, a number or a Learning-Rate Schedule. A larger rate reacts faster and is less stable.
    """
    eta::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    """
    The uniform-mix share, in `[0, 1)`. At `0` the rule takes the plain step.
    """
    alpha::T3
    """
    The objective whose gradient the rule reads, log wealth or a Risk Loss over the head's rows.
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

Forms the allocation that a [`MirrorDescent`](@ref) rule plays from its iterate.

At `alpha = 0` the function returns a copy of the iterate `u`.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{w}} &= (1 - \\alpha) \\boldsymbol{w} + \\frac{\\alpha}{N} \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``\\tilde{\\boldsymbol{w}}``: Played allocation.
  - ``\\boldsymbol{w}``: Iterate of the rule.
  - $(math_dict[:alpha_mix])
  - $(math_dict[:N])
  - ``\\boldsymbol{1}``: Vector of ones.

# Related

  - [`MirrorDescent`](@ref)
"""
function played_allocation(u::AbstractVector, alpha::Real)
    if iszero(alpha)
        return copy(u)
    end
    return (1 - alpha) .* u .+ alpha / length(u)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Mixes the price relatives that a [`MirrorDescent`](@ref) rule's update reads under the uniform mix `alpha`.

At `alpha = 0` the function returns `x` itself.

Helmbold, Schapire, Singer and Warmuth (1998, Theorem 4.2) state the mix for price relatives with a period maximum of one, ``(1 - \\alpha / N) \\boldsymbol{x} / \\max_i x_i + (\\alpha / N) \\boldsymbol{1}``. Every reader of the mixed vector, the gradient of the log-wealth loss and the hint residual, gives the same result at every scale of the vector. So the function scales the floor to the period's maximum, and does not scale the price relatives down to one.

The mix of the raw price relatives with an unscaled floor is a different step, outside the proof of the theorem. Take the exponentiated gradient at the default rate and ``\\alpha = 0.2``, over the price relatives `1 .+ 0.02 .* randn(StableRNG(7), 60, 4)`. The two steps then give played allocations that differ by at most ``2.6 \\times 10^{-6}``. From the uniform allocation, over one period in which the first of four assets rises by half and the others stay flat, they differ by ``7.0 \\times 10^{-5}``.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{x}} &= \\left(1 - \\frac{\\alpha}{N}\\right) \\boldsymbol{x} + \\frac{\\alpha}{N} \\max_i x_i \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``\\tilde{\\boldsymbol{x}}``: Mixed price relative vector.
  - ``\\boldsymbol{x}``: Price relative vector of the period.
  - ``x_i``: Price relative of asset ``i``.
  - $(math_dict[:alpha_mix])
  - $(math_dict[:N])
  - ``\\boldsymbol{1}``: Vector of ones.

The result is ``\\max_i x_i`` times the paper's mix.

# Related

  - [`MirrorDescent`](@ref)
  - [`played_allocation`](@ref)
"""
function mixed_relatives(x::AbstractVector, alpha::Real)
    if iszero(alpha)
        return x
    end
    N = length(x)
    return (1 - alpha / N) .* x .+ (alpha / N) * maximum(x)
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
    wh = price_adjusted_allocation(w, x)
    if restart(alg.eta, t)
        # The period count survives the restart: the schedule's stages are cumulative. The
        # start re-enters the set from the book the fund holds, as it entered at the seed.
        u0 = reprojection(alg.proj, set, st.w0, wh)
        return MirrorDescentState(t, u0, st.w0, schedule_state_seed(alg.eta, st.w0),
                                  gradient_state_seed(alg.grad, st.w0)), copy(u0)
    end
    # A schedule that chooses the rate from the period's row writes its statistic here, and
    # every other one after the step; the carrier the rate is read from carries the answer.
    st = MirrorDescentState(st.n, st.u, st.w0, statistic_before_rate(alg.eta, st.s, w, x),
                            st.gs)
    eta = learning_rate(alg.eta, t, st)
    alpha = mixing_share(alg.eta, t, alg.alpha)
    xm = mixed_relatives(x, alpha)
    g = loss_gradient(alg.obj, point, xm, rows)
    q = mirror_step(alg.proj, st.u, eta .* transform_gradient!(alg.grad, st.gs, g))
    u = project(alg.proj, set, q, wh)
    s = statistic_after_step(alg.eta, st.s, w, x)
    played = played_allocation(u, alpha)
    return MirrorDescentState(t, u, st.w0, s, st.gs),
           iszero(alpha) ? played : reprojection(alg.proj, set, played, wh)
end
"""
    reprojection(proj::AbstractProjectionGeometry, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    reprojection(proj::AbstractProjectionGeometry, set::AbstractAllocationSet, q::AbstractVector, w::AbstractVector)

Projects onto the head's Allocation Set an allocation that a [`MirrorDescent`](@ref) rule plays without a step to it.

There are two such allocations. One is the uniform mix of the iterate under `alpha`. The other is the Start Allocation that a restart of a Learning-Rate Schedule returns to. Either can lie outside the set. The mix does when the set excludes the uniform allocation. The start does when a turnover ceiling or a MIP kind excludes it from the book the fund now holds. So the head projects them again, in the rule's own geometry, as an [`ExpertMixture`](@ref) projects its blend with [`blend_projection`](@ref).

On a [`BoundedAllocationSet`](@ref), the function returns an allocation inside the bounds unchanged, so the default set costs nothing. On every other set the projection is a programme, and a rule with a positive `alpha` solves two programmes per period.

# Arguments

  - `proj`: The rule's geometry.
  - `set`: The set, resolved.
  - `q`: The allocation to play.
  - `w`: The Price-Adjusted Allocation the step trades from.

# Returns

  - `w'::AbstractVector`: `q` itself on a bounded set it lies in, its projection otherwise.

# Related

  - [`project`](@ref)
  - [`blend_projection`](@ref)
  - [`played_allocation`](@ref)
"""
function reprojection(proj::AbstractProjectionGeometry, set::BoundedAllocationSet,
                      q::AbstractVector, w::AbstractVector)
    wb = set.wb
    return all(wb.lb .<= q .<= wb.ub) ? q : project(proj, set, q, w)
end
function reprojection(proj::AbstractProjectionGeometry, set::AbstractAllocationSet,
                      q::AbstractVector, w::AbstractVector)
    return project(proj, set, q, w)
end
"""
    ExponentiatedGradient(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, alpha::Real = 0, obj::AbstractOnlineObjective = LogWealth(), grad::AbstractGradientTransform = PlainGradient())

Builds the exponentiated gradient of Helmbold, Schapire, Singer and Warmuth (1998), a [`MirrorDescent`](@ref) rule under [`EntropicProjection`](@ref) (EG).

The rule moves weight towards the assets that just did well. Its regret is ``O(\\sqrt{T \\log N})`` at a rate tuned to the horizon, and under the uniform mix `alpha` with the doubling trick it is universal. The default rate is the paper's ``0.05``. In its §5.2, rates from ``0.01`` to ``0.15`` all do well, and a rate above one loses money on the two-stock example. As the weighting of an [`ExpertMixture`](@ref), the rule takes the online gradient step over the vector of expert returns.

# Mathematical definition

Under [`LogWealth`](@ref), at `alpha = 0` and with no Gradient Transform,

```math
\\begin{align}
w_{t+1, i} &= \\frac{w_{t, i} \\exp(\\eta_t x_{t, i} / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle)}{\\sum_j w_{t, j} \\exp(\\eta_t x_{t, j} / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle)}\\,.
\\end{align}
```

Where:

  - ``w_{t, i}``: Weight of asset ``i`` in the iterate of period ``t``.
  - ``x_{t, i}``: Price relative of asset ``i`` in period ``t``.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:eta_t_lr])

The normalisation is the entropic projection onto the simplex.

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

Builds the gradient projection of Helmbold, Schapire, Singer and Warmuth (1997), a [`MirrorDescent`](@ref) rule under [`EuclideanProjection`](@ref) (GP, OGD).

On the simplex this is the online gradient descent of Zinkevich (2003). The 1997 paper subtracts the mean of the gradient to keep the budget, and assumes that the weights stay non-negative. The projection keeps the budget and enforces the sign, and on the simplex it is Zinkevich's fixed-rate step. The regret is ``O(\\sqrt{T})`` at ``\\eta \\propto 1 / \\sqrt{T}``, and [`InverseSquareRootRate`](@ref) is the anytime rate. Neither paper states a rate for the portfolio problem, so the default is the rate of [`ExponentiatedGradient`](@ref).

# Mathematical definition

Under [`LogWealth`](@ref), at `alpha = 0` and with no Gradient Transform,

```math
\\begin{align}
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{w}_t + \\eta_t \\frac{\\boldsymbol{x}_t}{\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:Proj_W_euclid])

On the simplex, the projection sets to zero every entry that the step leaves below the projection's threshold.

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

Builds the exponentiated gradient with the momentum of an exponential moving average, [`ExponentiatedGradient`](@ref) under [`GradientMomentum`](@ref) (EGE).

This is the EGE of Li, Zheng, Chen, Wang and Xu (2022). The paper's rate is the rate ``2 r \\sqrt{2 \\log N / T}`` of Helmbold and co-authors, with the doubling trick when the horizon is unknown. [`DoublingTrickRate`](@ref) on `eta` gives that trick. The default rate is the rate of [`ExponentiatedGradient`](@ref).

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

Builds the exponentiated gradient with a root-mean-square scale on the gradient, [`ExponentiatedGradient`](@ref) under [`RootMeanSquareGradient`](@ref) (EGR).

This is the EGR of Li, Zheng, Chen, Wang and Xu (2022). At the default `gamma2 = 0`, which the paper recommends, the rule stays at its Start Allocation, up to `eps`. The transformed gradient is then the sign of the gradient, the same at every asset, so the update moves nothing, and the rule is the constant rebalanced portfolio at its Start Allocation. [`RootMeanSquareGradient`](@ref) states why.

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

Builds the exponentiated gradient with an adaptive-moment scale on the gradient, [`ExponentiatedGradient`](@ref) under [`AdaptiveMomentGradient`](@ref) (EGA).

This is the EGA of Li, Zheng, Chen, Wang and Xu (2022).

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
"""
    MAEG(; etas::AbstractVector{<:Real} = 0.001:0.001:0.2, window::Integer = 30, alpha::Real = 0)

Builds the moving-window adaptive exponential gradient, [`ExponentiatedGradient`](@ref) under a [`WindowedBestRate`](@ref) over `window` periods (MAEG).

This is the MAEG of Zhang, Lin, Zheng and Yang (2022). The defaults are the paper's, the rate set of Helmbold, Schapire, Singer and Warmuth (1998) and the window `30`, which the paper chose over `7`.

# Examples

```jldoctest
julia> MAEG(; etas = [0.05, 0.1], window = 5)
MirrorDescent
    eta ┼ WindowedBestRate
        │     etas ┼ Vector{Float64}: [0.05, 0.1]
        │   window ┴ Int64: 5
   proj ┼ EntropicProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┴ PlainGradient()
```

# Related

  - [`MirrorDescent`](@ref)
  - [`WindowedBestRate`](@ref)
  - [`AEG`](@ref)

# References

  - $(ref_dict[:zhang2022maeg])
"""
function MAEG(; etas::AbstractVector{<:Real} = 0.001:0.001:0.2, window::Integer = 30,
              alpha::Real = 0)::MirrorDescent
    return ExponentiatedGradient(; eta = WindowedBestRate(; etas = etas, window = window),
                                 alpha = alpha)
end
"""
    AEG(; etas::AbstractVector{<:Real} = 0.001:0.001:0.2, alpha::Real = 0)

Builds the adaptive exponential gradient, [`ExponentiatedGradient`](@ref) under a [`WindowedBestRate`](@ref) over the whole history (AEG).

This is the AEG of Zhang, Lin, Zheng and Yang (2022), the special case of [`MAEG`](@ref) whose window is the horizon.

# Examples

```jldoctest
julia> AEG(; etas = [0.05, 0.1])
MirrorDescent
    eta ┼ WindowedBestRate
        │     etas ┼ Vector{Float64}: [0.05, 0.1]
        │   window ┴ nothing
   proj ┼ EntropicProjection()
  alpha ┼ Int64: 0
    obj ┼ LogWealth()
   grad ┴ PlainGradient()
```

# Related

  - [`MirrorDescent`](@ref)
  - [`WindowedBestRate`](@ref)
  - [`MAEG`](@ref)

# References

  - $(ref_dict[:zhang2022maeg])
"""
function AEG(; etas::AbstractVector{<:Real} = 0.001:0.001:0.2,
             alpha::Real = 0)::MirrorDescent
    return ExponentiatedGradient(; eta = WindowedBestRate(; etas = etas, window = nothing),
                                 alpha = alpha)
end
export MirrorDescent, ExponentiatedGradient, GradientProjection, EGE, EGR, EGA, LogWealth,
       RiskLoss, InverseSquareRootRate, DoublingTrickRate, SelfConfidentRate,
       WindowedBestRate, MAEG, AEG, PlainGradient, GradientMomentum, RootMeanSquareGradient,
       AdaptiveMomentGradient
public AbstractGradientTransform, gradient_state_seed, transform_gradient!,
       gradient_state_view, AbstractOnlineObjective, loss_gradient
