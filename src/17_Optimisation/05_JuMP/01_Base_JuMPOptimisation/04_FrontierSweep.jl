"""
    frontier_point_count(front::Frontier)
    frontier_point_count(front::VecNum)

Number of sweep points one frontier bound asks for.

A [`Frontier`](@ref) states its count in `N`; a stated vector of bounds states it in its
length. Both shapes are admissible in `:ret_frontier` and `:risk_frontier` (see
[`Front_NumVec`](@ref)), and at Model Assembly time a `Frontier` has not yet been resolved
into its range, so the count is read from the shape rather than from a materialised vector.

# Related

  - [`frontier_sweep_points`](@ref)
  - [`Frontier`](@ref)
"""
function frontier_point_count(front::Frontier)
    return Int(front.N)
end
function frontier_point_count(front::VecNum)
    return length(front)
end
"""
    frontier_sweep_points(model::JuMP.Model)

Count the solves the model's frontier sweep runs, and the factors that make up the count.

The sweep is a **product**: every swept return term and every swept risk measure joins the
same `Iterators.product`, so `k` bounds of `N` points each cost `N^k` full solves rather
than `k * N`. This reads both frontier registries — `:ret_frontier` and `:risk_frontier` —
and multiplies their per-entry counts together.

The product is accumulated as a `BigInt`, so it is exact and cannot overflow into a value
that would pass a cap it should fail.

# Returns

  - `(total, factors)`: the total number of sweep points, and a `bound_key => count` pair
    per swept entry, in registration order (return terms first).

# Related

  - [`assert_frontier_sweep_cap`](@ref)
  - [`frontier_point_count`](@ref)
  - [`SHARED_STATE`](@ref)
"""
function frontier_sweep_points(model::JuMP.Model)
    factors = Pair{Symbol, Int}[]
    for name in (:ret_frontier, :risk_frontier)
        if !shared_has(model, name)
            continue
        end
        for entry in shared_get(model, name)
            push!(factors, entry.first[2] => frontier_point_count(entry.second[2]))
        end
    end
    return prod(big(n) for (_, n) in factors; init = big(1)), factors
end
"""
    assert_frontier_sweep_cap(model::JuMP.Model)

Assert the **total** frontier sweep does not exceed the active `max_frontier` ceiling.

[`Frontier`](@ref)'s constructor caps the `N` of one bound; nothing there sees the product,
so `k` bounds at the ceiling cost `max_frontier^k` solves and no guard fires. This is the
guard, and it runs at Model Assembly — the point at which both frontier registries are
complete and no sweep solve has started yet.

Every sweep point runs a full inner `optimise_JuMP_model!` solve, so the product is the
compute-exhaustion sink `max_frontier` exists to bound (see [`RESOURCE_LIMITS`](@ref)). The
cap applies to the risk side and the return side alike.

# Returns

  - `nothing`.

# Throws

  - `DomainError` if the product exceeds `RESOURCE_LIMITS[].max_frontier`. The message names
    the product, the factors that made it, and the knob that raises the ceiling.

# Related

  - [`frontier_sweep_points`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`RESOURCE_LIMITS`](@ref)
  - [`assert_resource_cap`](@ref)
"""
function assert_frontier_sweep_cap(model::JuMP.Model)
    total, factors = frontier_sweep_points(model)
    if isempty(factors)
        return nothing
    end
    cap = RESOURCE_LIMITS[].max_frontier
    @argcheck(total <= cap,
              DomainError(total,
                          "the frontier sweep is $total points — the product $(join(("$key = $n" for (key, n) in factors), " × ")) across every swept return term and every swept risk measure — and exceeds RESOURCE_LIMITS[].max_frontier = $cap. Every point runs a full solve, so the ceiling is on the product, not on any single Frontier's N. Sweep fewer bounds, or lower an N. Raise the ceiling with set_resource_limits!(; max_frontier) — or with_resource_limits for a single scope, or the \"max_frontier\" preference for a whole project — for genuinely large machine-authored runs."))
    return nothing
end
"""
    frontier_axis(frontier::VecPair)

Turn one resolved frontier registry into the sweep axis it stands for.

Both registries — `:ret_frontier` and `:risk_frontier` — hold
`(bound_var_key, bound_key) => (expr, points, …)` entries, and both are swept as a **product**
across their own entries: two swept risk measures of `N` points each cost `N^2` solves on the
risk axis alone. This is that product, in two halves — the keys of the bound parameters, and
the values to write into them — so [`set_frontier_point!`](@ref) can zip one against the
other.

# Arguments

  - `frontier::VecPair`: A resolved frontier registry. Every entry's bound is already a vector
    of sweep points.

# Returns

  - `(keys, points)`: Two product iterators of equal length.

# Related

  - [`set_ret_frontier_parameters!`](@ref)
  - [`set_risk_frontier_parameters!`](@ref)
  - [`frontier_sweep_axes`](@ref)
"""
function frontier_axis(frontier::VecPair)
    itrs = [(Iterators.repeated(entry.first[1], length(entry.second[2])), entry.second[2])
            for entry in frontier]
    return Iterators.product.(itrs...)
end
"""
    set_ret_frontier_parameters!(model::JuMP.Model, ret_frontier::VecPair)

Register one parameter and one lower-bound constraint per swept return term.

Each term's bound binds on that term's **own** expression, so the return side is a product
across terms rather than a single ladder. The bound is homogenised by `k`, exactly as the
scalar bound in [`set_return_bounds!`](@ref) is.

# Arguments

  - $(arg_dict[:model])
  - `ret_frontier::VecPair`: The resolved `:ret_frontier` registry.

# Returns

  - The return sweep axis, as [`frontier_axis`](@ref) builds it.

# Related

  - [`set_risk_frontier_parameters!`](@ref)
  - [`frontier_sweep!`](@ref)
  - [`set_return_bounds!`](@ref)
"""
function set_ret_frontier_parameters!(model::JuMP.Model, ret_frontier::VecPair)
    sc = get_constraint_scale(model)
    k = get_k(model)
    for (keys, vals) in ret_frontier
        lb = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        model[keys[2]] = JuMP.@constraint(model, sc * (vals[1] - lb * k) >= 0)
    end
    return frontier_axis(ret_frontier)
end
"""
    set_risk_frontier_parameters!(model::JuMP.Model, risk_frontier::VecPair)

Register one parameter and one bound constraint per swept risk measure.

The twin of [`set_ret_frontier_parameters!`](@ref), and the one place the risk side's two
extra pieces are stated: the polarity `d`, which flips the inequality for a measure whose
bigger value is better, and the homogenisation `k`, which the scalar bound in
[`set_risk_upper_bound!`](@ref) also applies. `k` is the literal `1` under every head whose
objective is fixed — [`NearOptimalCentering`](@ref) minimises a barrier, so its head registers
`k = 1` and the factor is a no-op there — and the ratio variable under [`MaximumRatio`](@ref).
Reading it here rather than at each call site is what keeps the two heads from drifting apart.

# Arguments

  - $(arg_dict[:model])
  - `risk_frontier::VecPair`: The resolved `:risk_frontier` registry.

# Returns

  - The risk sweep axis, as [`frontier_axis`](@ref) builds it.

# Related

  - [`set_ret_frontier_parameters!`](@ref)
  - [`frontier_sweep!`](@ref)
  - [`set_risk_upper_bound!`](@ref)
  - [`get_k`](@ref)
"""
function set_risk_frontier_parameters!(model::JuMP.Model, risk_frontier::VecPair)
    sc = get_constraint_scale(model)
    k = get_k(model)
    for (keys, vals) in risk_frontier
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        d = ifelse(vals[3], 1, -1)
        model[keys[2]] = JuMP.@constraint(model, d * sc * (vals[1] - ub * k) <= 0)
    end
    return frontier_axis(risk_frontier)
end
"""
    frontier_sweep_axes(ret_axis, risk_axis)

Join the two sweep axes into the flat sequence of sweep points.

The risk axis varies **fastest**, so the flat order is return-outer and risk-inner. That order
is load-bearing rather than cosmetic: [`NearOptimalCentering`](@ref) solves its anchor
portfolios as one [`MeanRisk`](@ref) sweep over the same two frontiers, and pairs anchor `i`
with sweep point `i`. Stating the order once here is what keeps the two sweeps aligned. Either
axis may be `nothing`, which means that side is not swept. If both axes are `nothing`, the
sweep is one point that writes nothing.

# Arguments

  - `ret_axis`: The return axis from [`set_ret_frontier_parameters!`](@ref), or `nothing`.
  - `risk_axis`: The risk axis from [`set_risk_frontier_parameters!`](@ref), or `nothing`.

# Returns

  - An iterator of sweep points. Each point is a tuple of `(keys, bounds)` pairs, one per
    swept axis, and its `length` is the number of solves the sweep runs.

# Related

  - [`frontier_axis`](@ref)
  - [`frontier_sweep!`](@ref)
  - [`frontier_sweep_points`](@ref)
"""
function frontier_sweep_axes(ret_axis, risk_axis)
    return Iterators.product(zip(risk_axis...), zip(ret_axis...))
end
function frontier_sweep_axes(::Nothing, risk_axis)
    return Iterators.product(zip(risk_axis...))
end
function frontier_sweep_axes(ret_axis, ::Nothing)
    return Iterators.product(zip(ret_axis...))
end
function frontier_sweep_axes(::Nothing, ::Nothing)
    return Iterators.product()
end
"""
    set_frontier_point!(model::JuMP.Model, point::Tuple)

Write one sweep point's bounds into the frontier parameters.

# Arguments

  - $(arg_dict[:model])
  - `point::Tuple`: One element of a [`frontier_sweep_axes`](@ref) iterator. An empty tuple
    writes nothing, which is the sweep that has no frontier at all.

# Returns

  - `nothing`.

# Related

  - [`frontier_sweep!`](@ref)
  - [`frontier_sweep_axes`](@ref)
"""
function set_frontier_point!(model::JuMP.Model, point::Tuple)
    for (keys, bounds) in point
        for (key, bound) in zip(keys, bounds)
            JuMP.set_parameter_value(model[key], bound)
        end
    end
    return nothing
end
"""
    frontier_sweep!(point!, model, opt, ::Type{T}, points)
    frontier_sweep!(point!, model, opt, ::Type{T}, n::Integer)
    frontier_sweep!(model, opt, ::Type{T}, points)

Solve one model per sweep point and collect the outcomes.

The collect tail every frontier sweep shares. The model is assembled once and its objective is
set once; a sweep point changes only parameter values, so no constraint is rebuilt between
solves. `point!` is the per-optimiser hook, called with the **flat 1-based index** of the point
after its bounds are written — [`NearOptimalCentering`](@ref) uses it to move `noc_rk` and
`noc_rt` onto that point's anchor, and [`MeanRisk`](@ref) needs no hook at all.

The `n::Integer` method sweeps `n` points with no frontier bound to write, which is the
unconstrained [`NearOptimalCentering`](@ref) sweep over a vector of anchors.

# Arguments

  - `point!`: Hook of one argument, the flat index of the sweep point. Defaults to a no-op.
  - $(arg_dict[:model])
  - `opt::JuMPOptimisationEstimator`: The optimiser, for [`optimise_JuMP_model!`](@ref).
  - `::Type{T}`: Element type of the returns matrix.
  - `points`: A [`frontier_sweep_axes`](@ref) iterator, or the point count `n`.

# Returns

  - `(retcodes, sols)`: One entry per sweep point, in flat sweep order.

# Related

  - [`frontier_sweep_axes`](@ref)
  - [`set_frontier_point!`](@ref)
  - [`solve_mean_risk!`](@ref)
  - [`solve_noc!`](@ref)
"""
function frontier_sweep!(point!, model::JuMP.Model, opt::JuMPOptimisationEstimator,
                         ::Type{T}, points) where {T}
    n = length(points)
    retcodes = sizehint!(OptimisationReturnCode[], n)
    sols = sizehint!(JuMPOptimisationSolution[], n)
    for (i, point) in enumerate(points)
        set_frontier_point!(model, point)
        point!(i)
        retcode, sol = optimise_JuMP_model!(model, opt, T)
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function frontier_sweep!(point!, model::JuMP.Model, opt::JuMPOptimisationEstimator,
                         ::Type{T}, n::Integer) where {T}
    return frontier_sweep!(point!, model, opt, T, Iterators.repeated((), n))
end
function frontier_sweep!(model::JuMP.Model, opt::JuMPOptimisationEstimator, ::Type{T},
                         points) where {T}
    return frontier_sweep!(Returns(nothing), model, opt, T, points)
end
