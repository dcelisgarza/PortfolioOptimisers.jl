"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based weight finaliser formulations.

Defines the interface for norm types used when adjusting portfolio weights to satisfy bounds via a JuMP model.

# Interfaces

In order to implement a new formulation that works seamlessly with the library, subtype `JuMPWeightFinaliserFormulation` and implement the following method:

## `set_clustering_weight_finaliser_alg!`

  - `set_clustering_weight_finaliser_alg!(model::JuMP.Model, alg::MyFormulation, wi::VecNum) -> Nothing`: Adds the deviation objective to a model that already carries the decision vector `w`, the budget equality and the weight bounds.

### Arguments

  - `model`: The JuMP model, built by [`opt_weight_bounds`](@ref).
  - `alg`: The concrete subtype instance.
  - `wi`: The weights the optimisation produced, which the model repairs.

### Returns

  - `nothing`. The method works by adding variables, constraints and the objective to `model`.

# Related

  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
abstract type JuMPWeightFinaliserFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L1 norm of relative weight deviations when enforcing weight bounds.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} \\oslash \\boldsymbol{w}_{0} - \\boldsymbol{1} \\right\\rVert_{1}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - ``\\oslash``: Elementwise division. A zero entry of ``\\boldsymbol{w}_{0}`` is replaced by `eps` before the division, so the ratio stays finite.

# Constructors

    RelativeErrorWeightFinaliser() -> RelativeErrorWeightFinaliser

# Examples

```jldoctest
julia> RelativeErrorWeightFinaliser()
RelativeErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct RelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm of relative weight deviations when enforcing weight bounds.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} \\oslash \\boldsymbol{w}_{0} - \\boldsymbol{1} \\right\\rVert_{2}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - ``\\oslash``: Elementwise division. A zero entry of ``\\boldsymbol{w}_{0}`` is replaced by `eps` before the division, so the ratio stays finite.

The second-order cone bounds the norm itself, so the objective value is the L2 norm and not its square. The name records the squared-error criterion, whose minimiser is the same because the square is monotonic on a non-negative norm. [`RelativeErrorWeightFinaliser`](@ref) differs in the norm, not in the power.

# Constructors

    SquaredRelativeErrorWeightFinaliser() -> SquaredRelativeErrorWeightFinaliser

# Examples

```jldoctest
julia> SquaredRelativeErrorWeightFinaliser()
SquaredRelativeErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct SquaredRelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L1 norm of absolute weight deviations when enforcing weight bounds.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\right\\rVert_{1}\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])

# Constructors

    AbsoluteErrorWeightFinaliser() -> AbsoluteErrorWeightFinaliser

# Examples

```jldoctest
julia> AbsoluteErrorWeightFinaliser()
AbsoluteErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct AbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Minimises the L2 norm of absolute weight deviations when enforcing weight bounds.

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

The second-order cone bounds the norm itself, so the objective value is the L2 norm and not its square. The name records the squared-error criterion, whose minimiser is the same because the square is monotonic on a non-negative norm. [`AbsoluteErrorWeightFinaliser`](@ref) differs in the norm, not in the power.

# Constructors

    SquaredAbsoluteErrorWeightFinaliser() -> SquaredAbsoluteErrorWeightFinaliser

# Examples

```jldoctest
julia> SquaredAbsoluteErrorWeightFinaliser()
SquaredAbsoluteErrorWeightFinaliser()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`RelativeErrorWeightFinaliser`](@ref)
  - [`SquaredRelativeErrorWeightFinaliser`](@ref)
  - [`AbsoluteErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
struct SquaredAbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for weight finaliser strategies.

A `WeightFinaliser` enforces weight bounds after the optimisation has produced unconstrained weights.

# Interfaces

In order to implement a new strategy that works seamlessly with the library, subtype `WeightFinaliser` and implement the following method:

## `opt_weight_bounds`

  - `opt_weight_bounds(wf::MyFinaliser, wb::WeightBounds, w::VecNum) -> VecNum`: Moves `w` into the bounds `wb`, keeping the budget it already carries.

### Arguments

  - `wf`: The concrete subtype instance.
  - `wb`: The weight bounds. Either bound may be `nothing`.
  - `w`: The weights the optimisation produced.

### Returns

  - `w::VecNum`: The repaired weight vector. [`finalise_weight_bounds`](@ref) then pairs it with a return code.

# Related

  - [`IterativeWeightFinaliser`](@ref)
  - [`EuclideanWeightFinaliser`](@ref)
  - [`EntropicWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
abstract type WeightFinaliser <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Iteratively projects weights into the feasible region defined by weight bounds.

Each pass clips the weights to the bounds, then redistributes the clipped mass over the entries that lie strictly inside the bounds, in proportion to their own weights. The pass ends by rescaling the vector to the budget it started with, so the sum is preserved. Passes run until the bounds hold or until `iter` passes are done. An absent bound is read as `typemin` or `typemax` of the weight element type.

The redistribution keeps the ratios of the free weights, so the answer differs from the Euclidean projection of [`EuclideanWeightFinaliser`](@ref). The loop can fail to reach the bounds. It stalls when no weight lies strictly inside its bounds, for example `[0.6, 0.4, 0.0]` under `ub = 0.4`, and it can diverge on long-short bounds. If the last pass still breaks a bound, or is not finite, the finaliser returns the Euclidean projection of the input instead, which is `[0.4, 0.4, 0.2]` in that example. A loop that reaches the bounds keeps its own answer.

A bound set that cannot hold the budget has no feasible vector: four assets summing to `1` under `lb = 0.3` return their lower bounds, and [`finalise_weight_bounds`](@ref) reports an [`OptimisationFailure`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IterativeWeightFinaliser(;
        iter::Integer = 100
    ) -> IterativeWeightFinaliser

Keywords correspond to the struct's fields.

## Validation

  - `iter > 0`.

# Examples

```jldoctest
julia> IterativeWeightFinaliser()
IterativeWeightFinaliser
  iter ┴ Int64: 100
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`EuclideanWeightFinaliser`](@ref)
  - [`EntropicWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
@concrete struct IterativeWeightFinaliser <: WeightFinaliser
    """
    $(field_dict[:iter])
    """
    iter
    function IterativeWeightFinaliser(iter::Integer)
        @argcheck(iter > 0, DomainError(iter, "iter must be > 0"))
        return new{typeof(iter)}(iter)
    end
end
function IterativeWeightFinaliser(; iter::Integer = 100)::IterativeWeightFinaliser
    return IterativeWeightFinaliser(iter)
end
"""
$(DocStringExtensions.TYPEDEF)

Uses a JuMP optimisation model to enforce weight bounds.

The programme keeps the budget of the input weights and holds every weight between the bounds, and `alg` states which deviation it minimises over that set. An absent bound adds no constraint. A failed solve raises a warning and falls back to a default [`IterativeWeightFinaliser`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPWeightFinaliser(;
        slv::Slv_VecSlv,
        sc::Number = 1.0,
        so::Number = 1.0,
        alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser()
    ) -> JuMPWeightFinaliser

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a `VecSlv`: `!isempty(slv)`.
  - `sc > 0`, `so > 0`.

# Examples

```jldoctest
julia> JuMPWeightFinaliser(; slv = Solver(; solver = nothing))
JuMPWeightFinaliser
  slv ┼ Solver
      │          name ┼ String: ""
      │        solver ┼ nothing
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
   sc ┼ Float64: 1.0
   so ┼ Float64: 1.0
  alg ┴ RelativeErrorWeightFinaliser()
```

# Related

  - [`WeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliserFormulation`](@ref)
"""
@concrete struct JuMPWeightFinaliser <: WeightFinaliser
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:wfalg])
    """
    alg
    function JuMPWeightFinaliser(slv::Slv_VecSlv, sc::Number, so::Number,
                                 alg::JuMPWeightFinaliserFormulation)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc > zero(sc), DomainError(sc, "sc must be positive"))
        @argcheck(so > zero(so), DomainError(so, "so must be positive"))
        return new{typeof(slv), typeof(sc), typeof(so), typeof(alg)}(slv, sc, so, alg)
    end
end
function JuMPWeightFinaliser(; slv::Slv_VecSlv, sc::Number = 1.0, so::Number = 1.0,
                             alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser())::JuMPWeightFinaliser
    return JuMPWeightFinaliser(slv, sc, so, alg)
end
"""
    set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                         alg::JuMPWeightFinaliserFormulation,
                                         wi::VecNum)

Add the deviation objective of `alg` to the weight finalisation model.

[`opt_weight_bounds`](@ref) has already added the decision vector `w`, the budget equality and the weight bounds. This method adds the epigraph variable `t`, the cone that bounds the deviation of `w` from `wi`, and the objective `Min so * t`. The cone is a `NormOneCone` for the two L1 formulations and a `SecondOrderCone` for the two L2 formulations.

# Arguments

  - `model`: JuMP model, which must already carry `w` and the two scale expressions.
  - `alg`: The deviation formulation, one of the four [`JuMPWeightFinaliserFormulation`](@ref) subtypes.
  - `wi`: The weights the optimisation produced, which the model repairs.

# Returns

  - `nothing`.

# Details

  - The two relative formulations divide by `wi`, so they first replace each zero entry of `wi` **in place** with `eps(eltype(wi))`. The caller's vector carries that substitution afterwards.

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`opt_weight_bounds`](@ref)
"""
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::RelativeErrorWeightFinaliser, wi::VecNum)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t;
                      sc * (w ⊘ wi .- one(eltype(wi)))] in
                     JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredRelativeErrorWeightFinaliser,
                                              wi::VecNum)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; sc * (w ⊘ wi .- one(eltype(wi)))] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::AbsoluteErrorWeightFinaliser, wi::VecNum)
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredAbsoluteErrorWeightFinaliser,
                                              wi::VecNum)
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
"""
    opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum) -> VecNum
    opt_weight_bounds(wf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum) -> VecNum
    opt_weight_bounds(wf::EuclideanWeightFinaliser, wb::WeightBounds, w::VecNum) -> VecNum
    opt_weight_bounds(wf::EntropicWeightFinaliser, wb::WeightBounds, w::VecNum) -> VecNum

Move a weight vector into the bounds `wb`, keeping the budget it already carries.

The bounds themselves are not changed. Weights that already satisfy the bounds are returned unchanged, without a solve.

The [`JuMPWeightFinaliser`](@ref) method builds the programme of its `alg` (see [`set_clustering_weight_finaliser_alg!`](@ref)) and solves it. A failed solve warns and falls back to a default [`IterativeWeightFinaliser`](@ref). The [`IterativeWeightFinaliser`](@ref) method clips and redistributes instead, and returns the Euclidean projection if its passes end with a bound still broken. The [`EuclideanWeightFinaliser`](@ref) method returns the Euclidean projection ([`euclidean_weight_projection`](@ref)). The [`EntropicWeightFinaliser`](@ref) method returns the entropic projection ([`entropic_weight_projection`](@ref)), or the Euclidean one where the entropic one is not defined.

# Arguments

  - `wf`: Weight finaliser algorithm.
  - `wb`: Weight bounds.
  - `wi`, `w`: The weights the optimisation produced.

# Returns

  - `w::VecNum`: The repaired weight vector.

# Related

  - [`WeightBounds`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`IterativeWeightFinaliser`](@ref)
  - [`EuclideanWeightFinaliser`](@ref)
  - [`EntropicWeightFinaliser`](@ref)
  - [`finalise_weight_bounds`](@ref)
"""
function opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum)
    if !weights_break_bounds(wb, wi)
        return wi
    end
    lb = wb.lb
    ub = wb.ub
    model = JuMP.Model()
    JuMP.@expression(model, sc, wf.sc)
    JuMP.@expression(model, so, wf.so)
    JuMP.@variable(model, w[1:length(wi)])
    JuMP.@constraint(model, sc * (sum(w) - sum(wi)) == 0)
    if !isnothing(lb)
        JuMP.@constraint(model, sc * (w ⊖ lb) >= 0)
    end
    if !isnothing(ub)
        JuMP.@constraint(model, sc * (w ⊖ ub) <= 0)
    end
    set_clustering_weight_finaliser_alg!(model, wf.alg, wi)
    return if optimise_JuMP_model!(model, wf.slv).success
        JuMP.value.(get_w(model))
    else
        @warn("Version: $(wf.alg)\nReverting to Heuristic type.")
        opt_weight_bounds(IterativeWeightFinaliser(), wb, wi)
    end
end
"""
    finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)

Apply weight finalisation to enforce bounds and determine the optimisation return code.

Runs [`opt_weight_bounds`](@ref) with the given finaliser and bounds. The return code is an [`OptimisationSuccess`](@ref) if the weights are finite, lie in the bounds and keep the budget of the input, each to a tolerance ([`weights_meet_bounds`](@ref)), and an [`OptimisationFailure`](@ref) otherwise. A failure lets the fallback chain run.

A bound set that cannot hold the budget, `Σ lb > sum(w)` or `Σ ub < sum(w)`, always fails.

# Arguments

  - `wf::WeightFinaliser`: Weight finaliser algorithm.
  - `wb::WeightBounds`: Weight bounds configuration.
  - `w::VecNum`: Portfolio weights to finalise.

# Returns

  - `(retcode, w)`: Tuple of return code and adjusted weights.

# Related

  - [`WeightFinaliser`](@ref)
  - [`WeightBounds`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)
    s = sum(w)
    w = opt_weight_bounds(wf, wb, w)
    retcode = if weights_meet_bounds(wb, w, s)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "Failure to set bounds\n$wf\n$wb.")
    end
    return retcode, w
end

export IterativeWeightFinaliser, RelativeErrorWeightFinaliser,
       SquaredRelativeErrorWeightFinaliser, AbsoluteErrorWeightFinaliser,
       SquaredAbsoluteErrorWeightFinaliser, JuMPWeightFinaliser
public JuMPWeightFinaliserFormulation, set_clustering_weight_finaliser_alg!,
       WeightFinaliser, opt_weight_bounds
