"""
$(DocStringExtensions.TYPEDEF)

Moves weights into their bounds by the exact Euclidean projection, which needs no solver.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\right\\rVert_{2}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])

The minimiser is ``w_i = \\mathrm{clip}(w_{0,i} - \\theta, l_i, u_i)``, and the scalar ``\\theta`` restores the budget. The budget is monotone in ``\\theta``, so it is found exactly from the kinks where a weight reaches a bound (see [`euclidean_weight_projection`](@ref)).

It is the programme of a [`JuMPWeightFinaliser`](@ref) with a [`SquaredAbsoluteErrorWeightFinaliser`](@ref), solved in closed form. It holds any budget the input weights carry, a zero or a negative one included, and it holds long-short bounds. A bound set that cannot hold the budget has no projection, and [`finalise_weight_bounds`](@ref) then reports an [`OptimisationFailure`](@ref).

The projection moves every free weight by the same amount, so it does not keep the ratios of the free weights. [`EntropicWeightFinaliser`](@ref) and [`IterativeWeightFinaliser`](@ref) scale them instead: under `ub = 0.5`, the weights `[0.9, 0.09, 0.01]` become `[0.5, 0.29, 0.21]` here and `[0.5, 0.45, 0.05]` there.

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

Whether a weight in `w` lies outside the bounds `wb`. An absent bound breaks nothing.

# Related

  - [`opt_weight_bounds`](@ref)
"""
function weights_break_bounds(wb::WeightBounds, w::VecNum)::Bool
    return !isnothing(wb.lb) && any(w .< wb.lb) || !isnothing(wb.ub) && any(w .> wb.ub)
end
"""
    euclidean_weight_projection(w::VecNum, wb::WeightBounds) -> VecNum

The Euclidean projection of `w` onto the weights that lie in the bounds `wb` and sum to `sum(w)`.

The projection is ``\\mathrm{clip}(w - \\theta, l, u)``. Its sum is piecewise linear in ``\\theta`` and falls from `Σ ub` to `Σ lb` between the kinks `w - ub` and `w - lb`, so [`breakpoint_root`](@ref) finds the ``\\theta`` of the budget exactly from the kinks. An absent bound adds no kink.

# Returns

  - `w::VecNum`: The projected weights, a new vector. If `Σ lb > sum(w)` or `Σ ub < sum(w)`, no projection exists: the root is found on the flat tail past the last kink, and the returned weights sit on their bounds with a different sum.

# Related

  - [`EuclideanWeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`breakpoint_root`](@ref)
"""
function euclidean_weight_projection(w::VecNum, wb::WeightBounds)
    s = sum(w)
    lb = isnothing(wb.lb) ? typemin(eltype(w)) : wb.lb
    ub = isnothing(wb.ub) ? typemax(eltype(w)) : wb.ub
    # `Σ clip(w − θ, lb, ub)` is piecewise linear and non-increasing in `θ`, with its kinks
    # at `w − ub` and `w − lb`. `breakpoint_root` finds where a function crosses one, so the
    # budget `s` is moved to one.
    theta = breakpoint_root(t -> sum(clamp.(w .- t, lb, ub)) - s + one(s),
                            [w .- ub; w .- lb])
    return clamp.(w .- theta, lb, ub)
end
"""
$(DocStringExtensions.TYPEDEF)

Moves long-only weights into their bounds by the exact entropic projection, which needs no solver and keeps the ratios of the free weights.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\sum_{i} w_{i} \\log \\frac{w_{i}}{w_{0,i}}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])

The minimiser is ``w_i = \\mathrm{clip}(w_{0,i} / Z, l_i, u_i)``, and the scalar ``Z`` restores the budget. Every free weight is scaled by the same factor, so their ratios hold. This is the point the redistribution of [`IterativeWeightFinaliser`](@ref) aims at, and the finaliser reaches it exactly, with no passes. The loop can stall where a zero weight must take mass, and there the entropic projection is not defined either.

The projection is defined only when the weights and the lower bounds are non-negative, the budget is positive, and a scaled vector can meet the bounds. A zero weight stays at its lower bound for every ``Z``, so it cannot take mass. Outside this domain the finaliser returns the Euclidean projection of [`EuclideanWeightFinaliser`](@ref) instead (see [`entropic_weight_projection`](@ref)).

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

The entropic projection of `w` onto the weights that lie in the bounds `wb` and sum to `sum(w)`, or the Euclidean projection where the entropic one is not defined.

The projection is ``\\mathrm{clip}(t w, l, u)`` with ``t = 1 / Z``. Its sum is piecewise linear in ``t`` and rises from `Σ lb` to `Σ ub` over the positive weights plus `Σ lb` over the zero ones, between the kinks `lb / w` and `ub / w`, so [`breakpoint_root`](@ref) finds the ``t`` of the budget exactly from the kinks. An absent lower bound is read as zero, which changes no clip of a non-negative weight.

The function returns [`euclidean_weight_projection`](@ref) if a weight or a lower bound is negative, if the budget is not positive, or if the budget lies outside the open range of sums above. At `Σ lb` itself the only feasible vector is `lb`, which both projections return.

# Related

  - [`EntropicWeightFinaliser`](@ref)
  - [`euclidean_weight_projection`](@ref)
  - [`breakpoint_root`](@ref)
"""
function entropic_weight_projection(w::VecNum, wb::WeightBounds)
    s = sum(w)
    # `zero(w) .+ b` spreads a scalar bound over the assets.
    lb = isnothing(wb.lb) ? zero(w) : zero(w) .+ wb.lb
    ub = isnothing(wb.ub) ? zero(w) .+ typemax(eltype(w)) : zero(w) .+ wb.ub
    if !(s > zero(s) &&
         all(x -> x >= zero(x), w) &&
         all(x -> x >= zero(x), lb) &&
         sum(lb) < s <= sum(ifelse.(iszero.(w), lb, ub)))
        return euclidean_weight_projection(w, wb)
    end
    # `Σ clip(t w, lb, ub)` is piecewise linear and non-decreasing in the scale `t`, with its
    # kinks at `lb / w` and `ub / w`; the root is taken in `-t`, where it is non-increasing.
    t = -breakpoint_root(u -> sum(clamp.(-u .* w, lb, ub)) - s + one(s),
                         [-lb ./ w; -ub ./ w])
    return clamp.(t .* w, lb, ub)
end
"""
    weights_meet_bounds(wb::WeightBounds, w::VecNum, s::Number) -> Bool

Whether the finalised weights `w` are finite, lie in the bounds `wb` and sum to the budget `s`, each to the tolerance `sqrt(eps(eltype(w)))`. The budget tolerance is relative when `abs(s) > 1`.

[`finalise_weight_bounds`](@ref) reads it to choose its return code.

# Related

  - [`finalise_weight_bounds`](@ref)
"""
function weights_meet_bounds(wb::WeightBounds, w::VecNum, s::Number)::Bool
    if !all(isfinite, w)
        return false
    end
    tol = sqrt(eps(eltype(w)))
    lb_ok = isnothing(wb.lb) || all(w .>= wb.lb .- tol)
    ub_ok = isnothing(wb.ub) || all(w .<= wb.ub .+ tol)
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
    lb = isnothing(wb.lb) ? typemin(eltype(w)) : wb.lb
    ub = isnothing(wb.ub) ? typemax(eltype(w)) : wb.ub
    w0 = w
    s1 = sum(w)
    for _ in 1:(wf.iter)
        old_w = copy(w)
        w = max.(min.(w, ub), lb)
        idx = w .< ub .&& w .> lb
        w_add = sum(max.(old_w ⊖ ub, zero(eltype(w))))
        w_sub = sum(min.(old_w ⊖ lb, zero(eltype(w))))
        delta = w_add + w_sub
        if !iszero(delta)
            w[idx] += delta * w[idx] / sum(w[idx])
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
