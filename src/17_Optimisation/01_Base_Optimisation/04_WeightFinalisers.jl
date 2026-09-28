"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the deviations that a JuMP weight finaliser can minimise.

A subtype is the field `alg` of a [`JuMPWeightFinaliser`](@ref). It selects the norm of the deviation, and whether the deviation is absolute or relative to the weights that the optimisation produced.

# Interfaces

To add a deviation, subtype `JuMPWeightFinaliserFormulation` and implement the method below.

## `set_clustering_weight_finaliser_alg!`

  - `set_clustering_weight_finaliser_alg!(model::JuMP.Model, alg::MyFormulation, wi::VecNum) -> Nothing`: Adds the deviation objective to a model that already holds the weights `w`, the budget row and the bound rows.

### Arguments

  - `model`: The JuMP model that [`opt_weight_bounds`](@ref) builds.
  - `alg`: The concrete subtype instance.
  - `wi`: The weights that the optimisation produced. The method must not change them.

### Returns

  - `nothing`. The method adds variables, rows and the objective to `model`.

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

Selects the L1 norm of the relative weight deviation as the objective of a JuMP weight finaliser.

A relative deviation weighs a change to a small weight more than the same change to a large weight. So the finaliser moves mass to the largest free weights first.

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
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

The relative deviation is defined only where every entry of ``\\boldsymbol{w}_{0}`` is not zero.

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

Selects the L2 norm of the relative weight deviation as the objective of a JuMP weight finaliser.

The name comes from the squared error. The programme minimises the norm and not its square, and the two have the same minimiser because the square increases on non-negative values. So the value of the objective is the L2 norm.

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
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

The relative deviation is defined only where every entry of ``\\boldsymbol{w}_{0}`` is not zero. Where the bounds bind none of the free weights, the minimiser is ``w_{i} = w_{0,i} + c\\, w_{0,i}^{2}``, with one ``c`` that restores the budget.

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

Selects the L1 norm of the absolute weight deviation as the objective of a JuMP weight finaliser.

Every transfer of mass between two weights costs the same, so the minimiser is often not unique. The value of the objective is twice the mass that the finaliser moves.

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
  - ``\\boldsymbol{1}``: Vector of ones.

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

Selects the L2 norm of the absolute weight deviation as the objective of a JuMP weight finaliser.

The name comes from the squared error. The programme minimises the norm and not its square, and the two have the same minimiser because the square increases on non-negative values. So the value of the objective is the L2 norm.

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
  - ``\\boldsymbol{1}``: Vector of ones.

The minimiser is the Euclidean projection of ``\\boldsymbol{w}_{0}`` onto the feasible set, which [`EuclideanWeightFinaliser`](@ref) finds with no solver. Where the bounds bind none of the free weights, the minimiser is ``w_{i} = w_{0,i} + c``, with one ``c`` that restores the budget.

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

Abstract supertype for the methods that move the weights of an optimisation into their weight bounds.

A subtype runs after the optimisation. It keeps the sum of the weights, and it changes weights that already lie in the bounds not at all.

# Interfaces

To add a method, subtype `WeightFinaliser` and implement the method below.

## `opt_weight_bounds`

  - `opt_weight_bounds(wf::MyFinaliser, wb::WeightBounds, w::VecNum) -> VecNum`: Moves `w` into the bounds `wb`, and keeps the sum of `w`.

### Arguments

  - `wf`: The concrete subtype instance.
  - `wb`: The weight bounds. Either bound can be `nothing`.
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

Moves the weights into their bounds by a repeated clip and a proportional redistribution of the clipped mass.

The redistribution keeps the ratios of the free weights, so the answer differs from the Euclidean projection of [`EuclideanWeightFinaliser`](@ref). The loop stops at no answer when no weight lies strictly inside its bounds, for example `[0.6, 0.4, 0.0]` under `ub = 0.4`. It can also diverge on long-short bounds. In both cases step 9 returns the Euclidean projection, which is `[0.4, 0.4, 0.2]` in the example.

A set of bounds that cannot hold the budget has no feasible vector. Four weights that sum to `1` under `lb = 0.3` become their lower bounds, and [`finalise_weight_bounds`](@ref) reports an [`OptimisationFailure`](@ref).

# Algorithm

The steps are those of [`opt_weight_bounds`](@ref) for this type.

 1. Return `w` unchanged when it breaks no bound.
 2. Read the budget `s1` as `sum(w)`, and keep the input as `w0`. Read an absent bound as `typemin` or `typemax` of the element type of `w`, which becomes a float type when it is an integer type.
 3. Clip `w` to the bounds.
 4. Mark the free entries `idx`, which lie strictly inside their bounds after the clip.
 5. Subtract the mass that the clip added below `lb` from the mass that it removed above `ub`, giving `delta`.
 6. When `delta` is not zero, add `delta` to the free entries, in proportion to their weights.
 7. Multiply `w` by `s1 / sum(w)`, which restores the budget.
 8. Stop when `w` breaks no bound. Else repeat from step 3, for at most `iter` passes.
 9. Return `w` when it is finite and breaks no bound. Else return the Euclidean projection of `w0`, [`euclidean_weight_projection`](@ref).

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

Moves the weights into their bounds with the solution of a JuMP programme.

The programme keeps the sum of the input weights and holds every weight between its bounds. The field `alg` selects the deviation from the input that the programme minimises. An absent bound adds no row. When the solve fails, the finaliser logs a warning and uses a default [`IterativeWeightFinaliser`](@ref).

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

Adds the deviation objective of `alg` to the weight finalisation model.

[`opt_weight_bounds`](@ref) adds the weights `w`, the budget row and the bound rows before it calls this function. The two relative formulations divide by `wi`, so they replace each zero entry of a copy of `wi` with `eps(eltype(wi))`. They do not change `wi`.

# JuMP formulation

## Variables

  - `w`: The weights, read from the model.
  - `t`: The bound on the deviation, created with no key.

## Constraints

One unnamed row, which `alg` selects:

  - [`RelativeErrorWeightFinaliser`](@ref): ``(s_c t,\\, s_c (\\boldsymbol{w} \\oslash \\tilde{\\boldsymbol{w}}_{0} - \\boldsymbol{1})) \\in \\mathcal{K}_{1}``.
  - [`SquaredRelativeErrorWeightFinaliser`](@ref): ``(s_c t,\\, s_c (\\boldsymbol{w} \\oslash \\tilde{\\boldsymbol{w}}_{0} - \\boldsymbol{1})) \\in \\mathcal{K}_{2}``.
  - [`AbsoluteErrorWeightFinaliser`](@ref): ``(s_c t,\\, s_c (\\boldsymbol{w} - \\boldsymbol{w}_{0})) \\in \\mathcal{K}_{1}``.
  - [`SquaredAbsoluteErrorWeightFinaliser`](@ref): ``(s_c t,\\, s_c (\\boldsymbol{w} - \\boldsymbol{w}_{0})) \\in \\mathcal{K}_{2}``.

## Objective

  - `Min`: ``s_o t``. The objective pulls ``t`` down, so ``t`` equals the norm at the optimum.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - ``\\tilde{\\boldsymbol{w}}_{0}``: ``\\boldsymbol{w}_{0}`` with each zero entry replaced by `eps(eltype(wi))`, so that the division is defined.
  - ``t``: Bound on the norm of the deviation.
  - $(math_dict[:sc_scale])
  - $(math_dict[:so_scale])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

# Arguments

  - `model`: The JuMP model. It must already hold `w` and the two scales `sc` and `so`.
  - `alg`: The deviation, one of the four [`JuMPWeightFinaliserFormulation`](@ref) subtypes.
  - `wi`: The weights that the optimisation produced.

# Returns

  - `nothing`.

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`opt_weight_bounds`](@ref)
"""
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::RelativeErrorWeightFinaliser, wi::VecNum)
    mask = iszero.(wi)
    if any(mask)
        wi = copy(wi)
        wi[mask] .= eps(eltype(wi))
    end
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
    mask = iszero.(wi)
    if any(mask)
        wi = copy(wi)
        wi[mask] .= eps(eltype(wi))
    end
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

Moves a weight vector into the bounds `wb`, and keeps the sum of the vector.

Every method returns weights that break no bound unchanged, with no solve. The bounds do not change.

Each method runs the procedure of its finaliser type:

  - [`JuMPWeightFinaliser`](@ref): the programme of its `alg`, below. When the solve fails, the method logs a warning and runs the method of a default [`IterativeWeightFinaliser`](@ref).
  - [`IterativeWeightFinaliser`](@ref): the clip and redistribution that its docstring states.
  - [`EuclideanWeightFinaliser`](@ref): the Euclidean projection, [`euclidean_weight_projection`](@ref).
  - [`EntropicWeightFinaliser`](@ref): the entropic projection, [`entropic_weight_projection`](@ref), or the Euclidean projection where the entropic one is not defined.

# Algorithm

The steps are those of the [`JuMPWeightFinaliser`](@ref) method.

 1. Return `wi` unchanged when it breaks no bound.
 2. Build an empty model, and register the scales `sc` and `so` of `wf`.
 3. Add the weights `w`, the budget row, and a bound row for each bound that is not `nothing`.
 4. Add the objective of `wf.alg` with [`set_clustering_weight_finaliser_alg!`](@ref).
 5. Solve with the solvers of `wf.slv`. Return the values of `w` when the solve succeeds.
 6. Else log a warning, and return the weights of a default [`IterativeWeightFinaliser`](@ref) for `wi`.

# JuMP formulation

## Variables

  - `w`: The weights, created.

## Expressions

  - `sc`: ``s_c``.
  - `so`: ``s_o``.

## Constraints

  - An unnamed budget row: ``s_c (\\boldsymbol{1}^\\intercal \\boldsymbol{w} - \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}) = 0``.
  - An unnamed lower bound row, when `wb.lb` is not `nothing`: ``s_c (\\boldsymbol{w} - \\boldsymbol{l}) \\geq \\boldsymbol{0}``.
  - An unnamed upper bound row, when `wb.ub` is not `nothing`: ``s_c (\\boldsymbol{w} - \\boldsymbol{u}) \\leq \\boldsymbol{0}``.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - $(math_dict[:sc_scale])
  - $(math_dict[:so_scale])
  - ``\\boldsymbol{1}``: Vector of ones.

# Arguments

  - `wf`: The weight finaliser.
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `wi`, `w`: The weights that the optimisation produced.

# Returns

  - `w::VecNum`: The weights in the bounds, or the input when it breaks no bound.

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
        @warn("The weight finalisation model of $(wf.alg) could not be solved, so the weights are finalised with IterativeWeightFinaliser() instead.")
        opt_weight_bounds(IterativeWeightFinaliser(), wb, wi)
    end
end
"""
    finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)

Moves the weights into their bounds, and gives the return code that states whether the move succeeded.

A failure lets the fallback chain of [`optimise`](@ref) run. A set of bounds that cannot hold the budget, `sum(lb) > sum(w)` or `sum(ub) < sum(w)`, always fails.

# Algorithm

 1. Read the budget `s` as `sum(w)`.
 2. Move `w` into the bounds with [`opt_weight_bounds`](@ref).
 3. Check the moved weights with [`weights_meet_bounds`](@ref): they are finite, they lie in the bounds, and they sum to `s`, each to a tolerance.
 4. Return an [`OptimisationSuccess`](@ref) when the check passes, else an [`OptimisationFailure`](@ref) that names `wf` and `wb`, together with the moved weights.

# Arguments

  - `wf::WeightFinaliser`: The weight finaliser.
  - `wb::WeightBounds`: The weight bounds.
  - `w::VecNum`: The weights that the optimisation produced.

# Returns

  - `(retcode, w)`: The return code and the moved weights.

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
        OptimisationFailure(; res = "The weights do not meet the bounds.\n$wf\n$wb.")
    end
    return retcode, w
end

export IterativeWeightFinaliser, RelativeErrorWeightFinaliser,
       SquaredRelativeErrorWeightFinaliser, AbsoluteErrorWeightFinaliser,
       SquaredAbsoluteErrorWeightFinaliser, JuMPWeightFinaliser
public JuMPWeightFinaliserFormulation, set_clustering_weight_finaliser_alg!,
       WeightFinaliser, opt_weight_bounds
