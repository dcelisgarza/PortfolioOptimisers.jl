"""
    frontier_point_count(front::Frontier)
    frontier_point_count(front::VecNum)

Return the number of sweep points that one frontier bound asks for.

A [`Frontier`](@ref) states the count in its field `N`, and a vector of bounds states it in its length. The registries `:ret_frontier` and `:risk_frontier` accept both forms (see [`Front_NumVec`](@ref)). During Model Assembly a `Frontier` is not yet resolved into its points, so the count comes from the form of the bound and not from a vector of points.

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

Count the solves of the frontier sweep of the model, and return the factors of the count.

Every swept return term and every swept risk measure joins one product, so `k` bounds of `N` points each cost ``N^k`` solves and not ``k N``.

# Algorithm

 1. For each registry, `:ret_frontier` first and `:risk_frontier` second, that the model holds, push one `bound_key => count` pair per entry onto `factors`. The count is [`frontier_point_count`](@ref) of the bound of the entry.
 2. Multiply the counts as `BigInt` values, starting from one, giving `total`. The product is exact, so an overflow cannot give a value that passes a cap that it must fail.

# Returns

  - `(total, factors)`: The total number of sweep points, a `BigInt`, and the vector of `bound_key => count` pairs in the order of step 1. A model with no frontier gives `(1, [])`.

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

Check that the total number of points of the frontier sweep does not exceed the `max_frontier` ceiling.

The constructor of [`Frontier`](@ref) caps the `N` of one bound and does not see the product, so `k` bounds at the ceiling cost ``\\mathrm{max\\_frontier}^k`` solves. This check caps the product. It runs during Model Assembly, when both registries are complete and no solve of the sweep has started.

Each point runs a full solve with [`optimise_JuMP_model!`](@ref), so the product is the cost that `max_frontier` bounds (see [`RESOURCE_LIMITS`](@ref)). The cap applies to the return side and the risk side alike.

# Algorithm

 1. Count the points with [`frontier_sweep_points`](@ref), giving `total` and `factors`.
 2. Return when `factors` is empty, because the model sweeps no frontier.
 3. Read the ceiling `cap` from `RESOURCE_LIMITS[].max_frontier`.
 4. Check that `total <= cap`.

# Validation

  - `total <= RESOURCE_LIMITS[].max_frontier`. Otherwise a `DomainError` names the product, the factors that make it, and how to raise the ceiling.

# Returns

  - `nothing`.

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

Turn one resolved frontier registry into the sweep axis that it defines.

Each registry, `:ret_frontier` or `:risk_frontier`, holds entries of the form `(bound_var_key, bound_key) => (expr, points, …)`. The entries of one registry form a product, so two swept risk measures of `N` points each cost ``N^2`` solves on the risk axis alone. The axis is that product in two parts: the keys of the bound parameters, and the values to write into them.

# Algorithm

 1. For each entry, pair the key `bound_var_key`, repeated once per point, with the vector of points.
 2. Take the product of the repeated keys over the entries, giving `keys`, and the product of the points over the entries, giving `points`.

# Arguments

  - `frontier::VecPair`: A resolved frontier registry. The bound of each entry is a vector of sweep points.

# Returns

  - `(keys, points)`: Two product iterators of equal length. [`set_frontier_point!`](@ref) reads them in pairs.

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
    parameter_scale(vals::VecNum)

Return the fixed scale of one swept parameter: the largest absolute value of its points, or one when every point is zero.

A solver that does not take a parameter directly receives it through a bridge, as a variable that an equality row fixes. A parameter of the size of a fourth moment, near `1e-7`, can then break the row that reads it. Clarabel, for example, judges a residual against a norm of order one, so a residual that passes can be a large share of the value. The equilibration of the solver cannot repair it. The coefficient `-1` of the parameter stops the scale-up of a row whose other coefficients are small, and a constant row of the same small coefficients gets that scale-up.

So a builder writes the parameter in units of this scale. The scale is a fixed coefficient of the row, and the parameter takes the values `vals ./ parameter_scale(vals)`, which lie in ``[-1, 1]``. The row means the same, and its coefficients keep the size of the expression that it bounds. For a ceiling on the kurtosis term of [`VarianceSkewKurtosis`](@ref) under [`ConstrainedNearOptimalCentering`](@ref), a parameter in the units of the point broke the ceiling by `0.22 %`. A parameter in the units of this scale held it to `1e-7`, the same as a constant ceiling.

# Arguments

  - `vals::VecNum`: The values that the parameter takes during the sweep.

# Returns

  - The scale, of the type of `abs` of the element type of `vals`.

# Related

  - [`unit_frontier`](@ref)
  - [`set_ret_frontier_parameters!`](@ref)
  - [`set_risk_frontier_parameters!`](@ref)
  - [`set_noc_anchor_parameters!`](@ref)
"""
function parameter_scale(vals::VecNum)
    s = maximum(abs, vals; init = abs(zero(eltype(vals))))
    return ifelse(iszero(s), one(s), s)
end
"""
    unit_frontier(frontier::VecPair)

Return a copy of a resolved frontier registry with the points of each entry in the units of its [`parameter_scale`](@ref).

The parameters of the sweep take these values. The builders pass the copy to [`frontier_axis`](@ref), so [`set_frontier_point!`](@ref) writes each point in the units of the coefficient of its row.

# Arguments

  - `frontier::VecPair`: A resolved frontier registry. The bound of each entry is a vector of sweep points.

# Returns

  - A registry with the same keys and the same entries, where the second value of each entry is divided by its scale.

# Related

  - [`parameter_scale`](@ref)
  - [`frontier_axis`](@ref)
"""
function unit_frontier(frontier::VecPair)
    return [entry.first => Base.setindex(entry.second,
                                         entry.second[2] ./ parameter_scale(entry.second[2]), 2)
            for entry in frontier]
end
"""
    set_ret_frontier_parameters!(model::JuMP.Model, ret_frontier::VecPair)

Register one parameter and one floor row for each swept return term.

The bound of each term acts on the expression of that term, so the return side is a product over the terms and not one ladder. `k` multiplies the bound, as it multiplies the scalar bound of [`set_return_bounds!`](@ref).

# JuMP formulation

## Variables

  - `bound_var_key` of each entry: a new parameter ``b_j``, with the value zero until [`set_frontier_point!`](@ref) writes a point into it.
  - `k`: read from the model.

## Constraints

  - `bound_key` of each entry: ``s_c \\left(\\mathrm{ret}_j - m_j b_j k\\right) \\geq 0``.

Where:

  - ``\\mathrm{ret}_j``: The return expression of the ``j``-th swept return term, the first value of the entry.
  - ``b_j``: The return floor of the ``j``-th swept return term in units of ``m_j``, a parameter. A point ``p`` of the sweep writes ``p / m_j``.
  - ``m_j``: The [`parameter_scale`](@ref) of the points of the ``j``-th entry, a constant.
  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])

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
        m = parameter_scale(vals[2])
        lb = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        model[keys[2]] = JuMP.@constraint(model, sc * (vals[1] - m * lb * k) >= 0)
    end
    return frontier_axis(unit_frontier(ret_frontier))
end
"""
    set_risk_frontier_parameters!(model::JuMP.Model, risk_frontier::VecPair)

Register one parameter and one bound row for each swept risk measure.

It is the twin of [`set_ret_frontier_parameters!`](@ref). The risk side adds the direction ``d`` of the bound, which turns the ceiling into a floor for a measure whose flag is `false`. `k` multiplies the bound, as it multiplies the scalar bound of [`set_risk_upper_bound!`](@ref). Two heads call this builder. Under [`MeanRisk`](@ref), `k` is the ratio variable for a [`MaximumRatio`](@ref) objective and the number `1` for any other objective. [`NearOptimalCentering`](@ref) minimises a barrier and registers `k = 1`, so the factor has no effect there. One builder for both heads keeps the two from drifting apart.

# JuMP formulation

## Variables

  - `bound_var_key` of each entry: a new parameter ``u_j``, with the value zero until [`set_frontier_point!`](@ref) writes a point into it.
  - `k`: read from the model.

## Constraints

  - `bound_key` of each entry: ``d_j s_c \\left(R_j - m_j u_j k\\right) \\leq 0``. With ``d_j = 1`` the row is the ceiling ``R_j \\leq m_j u_j k``, and with ``d_j = -1`` it is the floor ``R_j \\geq m_j u_j k``.

Where:

  - ``R_j``: The risk expression of the ``j``-th swept risk measure, the first value of the entry.
  - ``u_j``: The bound of the ``j``-th swept risk measure in units of ``m_j``, a parameter. A point ``p`` of the sweep writes ``p / m_j``.
  - ``m_j``: The [`parameter_scale`](@ref) of the points of the ``j``-th entry, a constant. Without it, a bound of the size of a fourth moment breaks its row by a large share of the bound.
  - ``d_j``: ``1`` when the flag of the entry, its third value, is `true`, and ``-1`` otherwise.
  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])

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
        m = parameter_scale(vals[2])
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        d = ifelse(vals[3], 1, -1)
        model[keys[2]] = JuMP.@constraint(model, d * sc * (vals[1] - m * ub * k) <= 0)
    end
    return frontier_axis(unit_frontier(risk_frontier))
end
"""
    frontier_sweep_axes(ret_axis, risk_axis)

Join the two sweep axes into the flat sequence of sweep points.

The risk axis changes fastest, so the flat order runs over the risk points inside each return point. [`NearOptimalCentering`](@ref) solves its anchor portfolios as one [`MeanRisk`](@ref) sweep over the same two frontiers and pairs anchor `i` with sweep point `i`, so this one statement of the order keeps the two sweeps aligned. An axis that is `nothing` is not swept. When both are `nothing`, the sweep is one point that writes nothing.

# Arguments

  - `ret_axis`: The return axis from [`set_ret_frontier_parameters!`](@ref), or `nothing`.
  - `risk_axis`: The risk axis from [`set_risk_frontier_parameters!`](@ref), or `nothing`.

# Returns

  - An iterator of sweep points. Each point is a tuple of `(keys, bounds)` pairs, one for each swept axis, and its `length` is the number of solves that the sweep runs.

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

Write the bounds of one sweep point into the frontier parameters.

The bounds are in the units of the [`parameter_scale`](@ref) of each entry, because the builders build the axis from [`unit_frontier`](@ref).

# Algorithm

 1. For each `(keys, bounds)` pair of `point`, and for each `key` and `bound` of the pair, set the value of the parameter `model[key]` to `bound`.

# Arguments

  - $(arg_dict[:model])
  - `point::Tuple`: One element of a [`frontier_sweep_axes`](@ref) iterator. The empty tuple, the one point of a sweep with no frontier, writes nothing.

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

Solve the model once for each sweep point and collect the outcomes.

Every frontier sweep ends in this loop. The head builds the model once and sets its objective once. A sweep point changes only the values of the parameters, so the loop builds no constraint between two solves. `point!` is the hook of the optimiser. The loop calls it with the flat index of the point, from one, after it writes the bounds of the point. [`NearOptimalCentering`](@ref) uses it to move `noc_rk` and `noc_rt` to the anchor of that point, and [`MeanRisk`](@ref) passes no hook.

The method with `n::Integer` sweeps `n` points that write no bound. It is the [`NearOptimalCentering`](@ref) sweep over a vector of anchors with no frontier.

# Algorithm

 1. For each sweep point, with its flat index `i`:
     1. Write the bounds of the point with [`set_frontier_point!`](@ref).
     2. Call `point!(i)`.
     3. Solve the model with [`optimise_JuMP_model!`](@ref), giving `retcode` and `sol`.
     4. Push `retcode` onto `retcodes` and `sol` onto `sols`.

# Arguments

  - `point!`: Hook with one argument, the flat index of the sweep point. The method without it passes a hook that does nothing.
  - $(arg_dict[:model])
  - `opt::JuMPOptimisationEstimator`: The optimiser, for [`optimise_JuMP_model!`](@ref).
  - `::Type{T}`: Element type of the returns matrix.
  - `points`: A [`frontier_sweep_axes`](@ref) iterator, or the point count `n`.

# Returns

  - `(retcodes, sols)`: One entry for each sweep point, in the flat order of the sweep.

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
