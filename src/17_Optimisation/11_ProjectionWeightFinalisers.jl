"""
$(DocStringExtensions.TYPEDEF)

Moves weights into their bounds by the exact Euclidean projection, which needs no solver.

It solves the programme of a [`JuMPWeightFinaliser`](@ref) with a [`SquaredAbsoluteErrorWeightFinaliser`](@ref) in closed form. It keeps any budget that the input weights carry, a zero or a negative budget too, and it accepts long-short bounds. When the budget lies outside the sums of the bounds, no projection exists, and [`finalise_weight_bounds`](@ref) then reports an [`OptimisationFailure`](@ref).

The projection moves every free weight by the same amount, so it does not keep the ratios of the free weights. [`EntropicWeightFinaliser`](@ref) and [`IterativeWeightFinaliser`](@ref) scale them instead. Under `ub = 0.5`, the weights `[0.9, 0.09, 0.01]` become `[0.5, 0.29, 0.21]` here and `[0.5, 0.45, 0.05]` there.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\right\\rVert_{2}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

The minimiser clips the input at one shift:

```math
w_{i} = \\mathrm{clip}\\left(w_{0,i} - \\theta,\\, l_{i},\\, u_{i}\\right)\\,, \\qquad \\sum_{i} \\mathrm{clip}\\left(w_{0,i} - \\theta,\\, l_{i},\\, u_{i}\\right) = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,.
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - ``\\theta``: The shift that restores the budget.
  - ``\\mathrm{clip}(x, l, u) = \\min(\\max(x, l), u)``: The clip of ``x`` to the interval ``[l, u]``.

The sum of the clipped weights is continuous and does not increase in ``\\theta``. It falls from ``\\boldsymbol{1}^\\intercal \\boldsymbol{u}`` to ``\\boldsymbol{1}^\\intercal \\boldsymbol{l}``, so a shift exists if and only if ``\\boldsymbol{1}^\\intercal \\boldsymbol{l} \\leq \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0} \\leq \\boldsymbol{1}^\\intercal \\boldsymbol{u}``.

# Algorithm

The method of [`opt_weight_bounds`](@ref) for this tag runs these steps.

 1. When [`weights_break_bounds`](@ref) finds no weight of `w` outside the bounds, return `w` itself.
 2. Otherwise, return the projection of `w` from [`euclidean_weight_projection`](@ref).

# Constructors

    EuclideanWeightFinaliser() -> EuclideanWeightFinaliser

# Examples

```jldoctest
julia> EuclideanWeightFinaliser()
EuclideanWeightFinaliser()

julia> PortfolioOptimisers.opt_weight_bounds(EuclideanWeightFinaliser(),
                                             WeightBounds(; lb = 0.0, ub = 0.5), [0.9, 0.09, 0.01])
3-element Vector{Float64}:
 0.5
 0.29000000000000004
 0.21000000000000008
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`EntropicWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct EuclideanWeightFinaliser <: WeightFinaliser end
"""
    weights_break_bounds(wb::WeightBounds, w::VecNum) -> Bool

Tests whether a weight in `w` lies outside the bounds `wb`.

An absent bound breaks nothing. A `NaN` weight lies outside no bound, so this test passes it, and [`weights_meet_bounds`](@ref) rejects it later.

# Mathematical definition

```math
\\exists\\, i : w_{i} < l_{i} \\quad \\textrm{or} \\quad w_{i} > u_{i}\\,.
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:lb_ub_finaliser])

# Arguments

  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `w`: The weights to test.

# Returns

  - `flag::Bool`: `true` if a weight lies below its lower bound or above its upper bound.

# Related

  - [`opt_weight_bounds`](@ref)
  - [`weights_meet_bounds`](@ref)
"""
function weights_break_bounds(wb::WeightBounds, w::VecNum)::Bool
    return !isnothing(wb.lb) &&
           any(Broadcast.instantiate(Broadcast.broadcasted(<, w, wb.lb))) ||
           !isnothing(wb.ub) &&
           any(Broadcast.instantiate(Broadcast.broadcasted(>, w, wb.ub)))
end
"""
    euclidean_weight_projection(w::VecNum, wb::WeightBounds) -> VecNum

Projects `w` in the Euclidean norm onto the weights that lie in the bounds `wb` and sum to `sum(w)`.

[`EuclideanWeightFinaliser`](@ref) states the programme and its minimiser. This function finds the shift of that minimiser exactly from the kinks of the budget, with no solver and no tolerance.

# Algorithm

 1. Read the budget `s` as `sum(w)`.
 2. Read the bounds `lb` and `ub`. An absent bound is the infinity of the type that a division of two weights returns, so it adds no kink.
 3. Collect the kinks `w .- ub` and `w .- lb`. At a kink, one weight reaches one of its bounds. Between two neighbouring kinks, the sum of `clamp.(w .- theta, lb, ub)` is linear in `theta` and does not increase.
 4. Find the shift `theta` at which that sum equals `s`, with [`breakpoint_root`](@ref) over the kinks.
 5. Return `clamp.(w .- theta, lb, ub)`.

# Arguments

  - `w`: The weights to project.
  - `wb`: The weight bounds. Either bound can be `nothing`.

# Returns

  - `w::VecNum`: The projected weights, a new vector. If `Σ lb > sum(w)` or `Σ ub < sum(w)`, no projection exists. [`breakpoint_root`](@ref) then finds the root on the flat tail past the last kink, and the returned weights lie on their bounds with a different sum.

# Related

  - [`EuclideanWeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`breakpoint_root`](@ref)
"""
function euclidean_weight_projection(w::VecNum, wb::WeightBounds)
    s = sum(w)
    # An absent bound is an infinity of the type the root lands in, so it adds no kink. An
    # integer's own `typemin` is finite, and `w .- typemin` overflows.
    T = typeof(one(eltype(w)) / one(eltype(w)))
    lb = isnothing(wb.lb) ? typemin(T) : wb.lb
    ub = isnothing(wb.ub) ? typemax(T) : wb.ub
    # `Σ clip(w − θ, lb, ub)` is piecewise linear and non-increasing in `θ`, with its kinks
    # at `w − ub` and `w − lb`. `breakpoint_root` finds where a function crosses one, so the
    # budget `s` is moved to one.
    clip = t -> Broadcast.instantiate(Broadcast.broadcasted((x, l, u) -> clamp(x - t, l, u),
                                                            w, lb, ub))
    theta = breakpoint_root(t -> sum(clip(t)) - s + one(s), [w .- ub; w .- lb])
    return clamp.(w .- theta, lb, ub)
end
"""
$(DocStringExtensions.TYPEDEF)

Moves long-only weights into their bounds by the exact entropic projection, which needs no solver and keeps the ratios of the free weights.

The projection scales every free weight by the same factor, so their ratios hold. For positive weights under zero lower bounds, the loop of [`IterativeWeightFinaliser`](@ref) converges to the same point. Under two-sided bounds the loop can stop at a different point. Under `lb = 0.3` and `ub = 0.45`, the weights `[0.5, 0.4, 0.1]` become `[0.4, 0.3, 0.3]` there and `[0.389, 0.311, 0.3]` here. The second point is nearer to the input in relative entropy.

The projection is defined only when the weights and the lower bounds are non-negative, the budget is positive, and a scaled vector can meet the bounds. Outside this domain, the finaliser returns the Euclidean projection of [`EuclideanWeightFinaliser`](@ref) instead.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\sum_{i} w_{i} \\log \\frac{w_{i}}{w_{0,i}}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

The minimiser clips the scaled input:

```math
w_{i} = \\mathrm{clip}\\left(\\frac{w_{0,i}}{Z},\\, l_{i},\\, u_{i}\\right)\\,, \\qquad \\sum_{i} \\mathrm{clip}\\left(\\frac{w_{0,i}}{Z},\\, l_{i},\\, u_{i}\\right) = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,.
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - ``Z``: The positive normaliser that restores the budget.
  - ``\\mathrm{clip}(x, l, u) = \\min(\\max(x, l), u)``: The clip of ``x`` to the interval ``[l, u]``.

A zero ``w_{0,i}`` with a positive ``l_{i}`` gives the programme no finite objective. The clip sets such a weight to ``l_{i}`` for every ``Z``, so a zero weight cannot rise above its lower bound. The sum of the clipped weights rises with ``1 / Z`` from ``\\boldsymbol{1}^\\intercal \\boldsymbol{l}`` to the sum of ``u_{i}`` over the positive weights and ``l_{i}`` over the zero weights. A normaliser therefore exists when the budget is above the first sum and at most the second. At the first sum, the only feasible vector is ``\\boldsymbol{l}``.

# Algorithm

The method of [`opt_weight_bounds`](@ref) for this tag runs these steps.

 1. When [`weights_break_bounds`](@ref) finds no weight of `w` outside the bounds, return `w` itself.
 2. Otherwise, return the projection of `w` from [`entropic_weight_projection`](@ref).

# Constructors

    EntropicWeightFinaliser() -> EntropicWeightFinaliser

# Examples

```jldoctest
julia> EntropicWeightFinaliser()
EntropicWeightFinaliser()

julia> PortfolioOptimisers.opt_weight_bounds(EntropicWeightFinaliser(),
                                             WeightBounds(; lb = 0.0, ub = 0.5), [0.9, 0.09, 0.01])
3-element Vector{Float64}:
 0.5
 0.44999999999999996
 0.05
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`EuclideanWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct EntropicWeightFinaliser <: WeightFinaliser end
"""
    entropic_weight_projection(w::VecNum, wb::WeightBounds) -> VecNum

Projects `w` in relative entropy onto the weights that lie in the bounds `wb` and sum to `sum(w)`, or returns the Euclidean projection where the entropic one is not defined.

[`EntropicWeightFinaliser`](@ref) states the programme, its minimiser and its domain. This function finds the scale of that minimiser exactly from the kinks of the budget, with no solver and no tolerance.

# Algorithm

 1. Read the budget `s` as `sum(w)`.
 2. Spread the bounds over the assets as `lb` and `ub`. An absent lower bound is zero, which changes no clip of a non-negative weight. An absent upper bound is the infinity of the type that a division of two weights returns.
 3. When the entropic projection is not defined, return the projection from [`euclidean_weight_projection`](@ref). The projection is not defined when a weight or a lower bound is negative, or when `s` is not positive. It is also not defined when `s` is at most `sum(lb)`, or above the sum of `ub` over the positive weights and `lb` over the zero weights. At `s = sum(lb)`, the only feasible vector is `lb`, and the Euclidean projection returns it.
 4. Collect the kinks `lb ./ w` and `ub ./ w`. At a kink, one weight reaches one of its bounds. Between two neighbouring kinks, the sum of `clamp.(t .* w, lb, ub)` is linear in the scale `t` and does not decrease.
 5. Find the scale `t` at which that sum equals `s`, with [`breakpoint_root`](@ref) over the negated kinks. [`breakpoint_root`](@ref) needs a sum that does not increase, so it searches in `-t`.
 6. Return `clamp.(t .* w, lb, ub)`.

# Arguments

  - `w`: The weights to project.
  - `wb`: The weight bounds. Either bound can be `nothing`.

# Returns

  - `w::VecNum`: The projected weights, a new vector.

# Related

  - [`EntropicWeightFinaliser`](@ref)
  - [`euclidean_weight_projection`](@ref)
  - [`breakpoint_root`](@ref)
"""
function entropic_weight_projection(w::VecNum, wb::WeightBounds)
    s = sum(w)
    # `zero(w) .+ b` spreads a scalar bound over the assets.
    T = typeof(one(eltype(w)) / one(eltype(w)))
    lb = isnothing(wb.lb) ? zero(w) : zero(w) .+ wb.lb
    ub = isnothing(wb.ub) ? zero(w) .+ typemax(T) : zero(w) .+ wb.ub
    if !(s > zero(s) &&
         all(x -> x >= zero(x), w) &&
         all(x -> x >= zero(x), lb) &&
         sum(lb) < s <= sum(i -> ifelse(iszero(w[i]), lb[i], ub[i]), eachindex(w, lb, ub)))
        return euclidean_weight_projection(w, wb)
    end
    # `Σ clip(t w, lb, ub)` is piecewise linear and non-decreasing in the scale `t`, with its
    # kinks at `lb / w` and `ub / w`; the root is taken in `-t`, where it is non-increasing.
    t = -breakpoint_root(u -> sum(i -> clamp(-u * w[i], lb[i], ub[i]),
                                  eachindex(w, lb, ub)) - s + one(s), [-lb ./ w; -ub ./ w])
    return clamp.(t .* w, lb, ub)
end
"""
    weights_meet_bounds(wb::WeightBounds, w::VecNum, s::Number) -> Bool

Tests whether the finalised weights `w` are finite, lie in the bounds `wb`, and sum to the budget `s`. The last two conditions hold to a tolerance.

[`finalise_weight_bounds`](@ref) reads it to choose its return code.

# Mathematical definition

Every ``w_{i}`` is finite, and:

```math
\\begin{align}
\\boldsymbol{l} - \\tau &\\leq \\boldsymbol{w} \\leq \\boldsymbol{u} + \\tau\\,, \\\\
\\left\\lvert \\boldsymbol{1}^\\intercal \\boldsymbol{w} - s \\right\\rvert &\\leq \\tau \\max\\left(1, \\lvert s \\rvert\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:lb_ub_finaliser])
  - ``s``: The budget of the weights before the finaliser moved them.
  - ``\\tau = \\sqrt{\\epsilon}``: The tolerance, with ``\\epsilon`` the machine epsilon of the type that a division of two weights returns.

The tolerance on the bounds is absolute. The tolerance on the budget is relative when ``\\lvert s \\rvert > 1``.

# Arguments

  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `w`: The finalised weights.
  - `s`: The budget of the weights before the finaliser moved them.

# Returns

  - `flag::Bool`: `true` if the weights meet all three conditions.

# Related

  - [`finalise_weight_bounds`](@ref)
  - [`weights_break_bounds`](@ref)
"""
function weights_meet_bounds(wb::WeightBounds, w::VecNum, s::Number)::Bool
    if !all(isfinite, w)
        return false
    end
    # The type of a division, so an integer vector that no finaliser moved has an `eps`.
    tol = sqrt(eps(typeof(one(eltype(w)) / one(eltype(w)))))
    lb_ok = isnothing(wb.lb) ||
            all(Broadcast.instantiate(Broadcast.broadcasted((x, l) -> x >= l - tol, w,
                                                            wb.lb)))
    ub_ok = isnothing(wb.ub) ||
            all(Broadcast.instantiate(Broadcast.broadcasted((x, u) -> x <= u + tol, w,
                                                            wb.ub)))
    return lb_ok && ub_ok && abs(sum(w) - s) <= tol * max(one(s), abs(s))
end
function opt_weight_bounds(::EuclideanWeightFinaliser, wb::WeightBounds, w::VecNum)
    return weights_break_bounds(wb, w) ? euclidean_weight_projection(w, wb) : w
end
function opt_weight_bounds(::EntropicWeightFinaliser, wb::WeightBounds, w::VecNum)
    return weights_break_bounds(wb, w) ? entropic_weight_projection(w, wb) : w
end
function opt_weight_bounds(wf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum)
    if !weights_break_bounds(wb, w)
        return w
    end
    T = typeof(one(eltype(w)) / one(eltype(w)))
    lb = isnothing(wb.lb) ? typemin(T) : wb.lb
    ub = isnothing(wb.ub) ? typemax(T) : wb.ub
    w0 = w
    s1 = sum(w)
    for _ in 1:(wf.iter)
        old_w = w
        w = max.(min.(w, ub), lb)
        idx = w .< ub .&& w .> lb
        w_add = sum(x -> max(x, zero(eltype(w))), old_w ⊖ ub)
        w_sub = sum(x -> min(x, zero(eltype(w))), old_w ⊖ lb)
        delta = w_add + w_sub
        if !iszero(delta)
            # A new vector, which the division widens when the clip kept integers.
            w = ifelse.(idx, w .+ delta .* w ./ sum(w[idx]), w)
        end
        w *= s1 / sum(w)
        if !weights_break_bounds(wb, w)
            break
        end
    end
    # The loop stalls when no weight lies strictly inside its bounds, and it can diverge on
    # long-short bounds. Its last vector then breaks the bounds, or is not finite, so the
    # exact projection of the input replaces it.
    return if all(isfinite, w) && !weights_break_bounds(wb, w)
        w
    else
        euclidean_weight_projection(w0, wb)
    end
end

export EuclideanWeightFinaliser, EntropicWeightFinaliser
