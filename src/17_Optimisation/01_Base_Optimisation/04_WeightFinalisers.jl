"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the deviations that a JuMP weight finaliser can minimise.

A subtype is the field `alg` of a [`JuMPWeightFinaliser`](@ref), and the field `wf` of a [`DiscreteAllocation`](@ref). It selects the deviation from the target weights that the programme minimises. [`ErrorWeightFinaliser`](@ref) is the library's subtype.

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

  - [`ErrorWeightFinaliser`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`DiscreteAllocation`](@ref)
"""
abstract type JuMPWeightFinaliserFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the forms of the deviation of a vector from its target weights.

A subtype is the field `dev` of an [`ErrorWeightFinaliser`](@ref). It states the residual whose norm the programme minimises.

# Interfaces

To add a deviation, subtype `WeightDeviation` and implement the two methods below.

## `weight_deviation`

  - `weight_deviation(dev::MyDeviation, x, w0::VecNum, c::Number) -> AbstractVector`: Returns the residual of `x` against the target `c * w0`.

### Arguments

  - `dev`: The concrete subtype instance.
  - `x`: The model's vector that tracks the target. It is the weights in [`JuMPWeightFinaliser`](@ref), and the money in each position in [`DiscreteAllocation`](@ref).
  - `w0`: The target weights. The method must not change them.
  - `c`: The scale that turns `w0` into the units of `x`. It is `1` in [`JuMPWeightFinaliser`](@ref), and the cash of the side in [`DiscreteAllocation`](@ref).

### Returns

  - The residual, a vector of the length of `x`.

## `deviation_error_term`

  - `deviation_error_term(dev::MyDeviation, u, c::Number)`: Returns the error term of the objective of [`DiscreteAllocation`](@ref), in money.

### Arguments

  - `dev`: The concrete subtype instance.
  - `u`: The model's bound on the norm of the residual.
  - `c`: The cash of the side.

### Returns

  - The term that the objective adds to the cash left over.

# Related

  - [`RelativeWeightDeviation`](@ref)
  - [`AbsoluteWeightDeviation`](@ref)
  - [`ErrorWeightFinaliser`](@ref)
"""
abstract type WeightDeviation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects the deviation relative to the target weights, ``\\boldsymbol{x} \\oslash (c\\, \\tilde{\\boldsymbol{w}}_{0}) - \\boldsymbol{1}``.

A relative deviation weighs a change to a small weight more than the same change to a large weight. The deviation is defined only where every entry of ``\\boldsymbol{w}_{0}`` is not zero, so [`weight_deviation`](@ref) replaces each zero entry of a copy of the target with `eps`. The deviation has no unit, so [`DiscreteAllocation`](@ref) prices it in money as ``c\\, u``.

Where:

  - ``\\boldsymbol{x}``: The model's vector that tracks the target.
  - ``c``: The scale of the target.
  - ``\\tilde{\\boldsymbol{w}}_{0}``: The target weights with each zero entry replaced by `eps`.
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

# Examples

```jldoctest
julia> RelativeWeightDeviation()
RelativeWeightDeviation()
```

# Related

  - [`WeightDeviation`](@ref)
  - [`AbsoluteWeightDeviation`](@ref)
  - [`ErrorWeightFinaliser`](@ref)
"""
struct RelativeWeightDeviation <: WeightDeviation end
"""
$(DocStringExtensions.TYPEDEF)

Selects the deviation from the target weights, ``\\boldsymbol{x} - c\\, \\boldsymbol{w}_{0}``.

The deviation has the units of ``\\boldsymbol{x}``. So [`DiscreteAllocation`](@ref) uses its norm ``u`` as the money term of the objective.

Where:

  - ``\\boldsymbol{x}``: The model's vector that tracks the target.
  - ``c``: The scale of the target.
  - $(math_dict[:w_0_finaliser])

# Examples

```jldoctest
julia> AbsoluteWeightDeviation()
AbsoluteWeightDeviation()
```

# Related

  - [`WeightDeviation`](@ref)
  - [`RelativeWeightDeviation`](@ref)
  - [`ErrorWeightFinaliser`](@ref)
"""
struct AbsoluteWeightDeviation <: WeightDeviation end
"""
    weight_deviation(dev::RelativeWeightDeviation, x, w0::VecNum, c::Number)
    weight_deviation(dev::AbsoluteWeightDeviation, x, w0::VecNum, c::Number)

Returns the residual of `x` against the target `c * w0` that `dev` selects.

The relative method replaces each zero entry of a copy of `w0` with `eps(eltype(w0))` before it multiplies by `c`. It does not change `w0`.

# Arguments

  - `dev`: The deviation.
  - `x`: The model's vector that tracks the target.
  - `w0`: The target weights.
  - `c`: The scale that turns `w0` into the units of `x`.

# Returns

  - [`RelativeWeightDeviation`](@ref): ``\\boldsymbol{x} \\oslash (c\\, \\tilde{\\boldsymbol{w}}_{0}) - \\boldsymbol{1}``.
  - [`AbsoluteWeightDeviation`](@ref): ``\\boldsymbol{x} - c\\, \\boldsymbol{w}_{0}``.

# Related

  - [`WeightDeviation`](@ref)
  - [`deviation_error_term`](@ref)
  - [`ErrorWeightFinaliser`](@ref)
"""
function weight_deviation(::RelativeWeightDeviation, x, w0::VecNum, c::Number)
    mask = iszero.(w0)
    if any(mask)
        w0 = copy(w0)
        w0[mask] .= eps(eltype(w0))
    end
    wc = w0 * c
    return x ⊘ wc .- one(eltype(wc))
end
function weight_deviation(::AbsoluteWeightDeviation, x, w0::VecNum, c::Number)
    return x - w0 * c
end
"""
    deviation_error_term(dev::RelativeWeightDeviation, u, c::Number)
    deviation_error_term(dev::AbsoluteWeightDeviation, u, ::Number)

Returns the error term of the objective of [`DiscreteAllocation`](@ref), in money.

The cash left over is money. An absolute deviation is money too, so the term is ``u``. A relative deviation has no unit, so the term is ``c\\, u``, and a relative deviation of one over the whole side is worth the cash of the side.

# Arguments

  - `dev`: The deviation.
  - `u`: The model's bound on the norm of the residual.
  - `c`: The cash of the side.

# Returns

  - The error term of the objective.

# Related

  - [`WeightDeviation`](@ref)
  - [`weight_deviation`](@ref)
  - [`set_discrete_error!`](@ref)
"""
function deviation_error_term(::RelativeWeightDeviation, u, c::Number)
    return c * u
end
function deviation_error_term(::AbsoluteWeightDeviation, u, ::Number)
    return u
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the norm of a weight deviation as the objective of a JuMP weight finaliser.

The field `alg` is the norm, and the field `dev` states whether the deviation is relative to the target weights or absolute.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} &\\quad \\left\\lVert \\boldsymbol{d}(\\boldsymbol{w},\\, \\boldsymbol{w}_{0}) \\right\\rVert\\,, \\\\
\\textrm{s.t.} &\\quad \\boldsymbol{1}^\\intercal \\boldsymbol{w} = \\boldsymbol{1}^\\intercal \\boldsymbol{w}_{0}\\,, \\\\
&\\quad \\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - $(math_dict[:lb_ub_finaliser])
  - ``\\boldsymbol{d}``: The deviation of `dev`, ``\\boldsymbol{w} \\oslash \\boldsymbol{w}_{0} - \\boldsymbol{1}`` for a [`RelativeWeightDeviation`](@ref), and ``\\boldsymbol{w} - \\boldsymbol{w}_{0}`` for an [`AbsoluteWeightDeviation`](@ref).
  - ``\\lVert \\cdot \\rVert``: The norm of `alg`.
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

The programme bounds the norm itself. It does not divide the norm by [`norm_factor`](@ref), so the `ddof` of `alg` has no effect. A [`SquaredL2Norm`](@ref) and an [`L2Norm`](@ref) give the same programme, because the square increases on non-negative values and so does not move the minimiser.

Each pair of a norm and a deviation has its own minimiser:

  - The L1 norm of the relative deviation weighs a change to a small weight more than the same change to a large weight. So the finaliser moves mass to the largest free weights first.
  - The L2 norm of the relative deviation has the minimiser ``w_{i} = w_{0,i} + c\\, w_{0,i}^{2}`` where the bounds bind none of the free weights, with one ``c`` that restores the budget.
  - The L1 norm of the absolute deviation costs the same for every transfer of mass between two weights, so its minimiser is often not unique. The value of the objective is twice the mass that the finaliser moves.
  - The L2 norm of the absolute deviation has the Euclidean projection of ``\\boldsymbol{w}_{0}`` onto the feasible set as its minimiser, which [`EuclideanWeightFinaliser`](@ref) finds with no solver. Where the bounds bind none of the free weights, the minimiser is ``w_{i} = w_{0,i} + c``, with one ``c`` that restores the budget.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ErrorWeightFinaliser(;
        alg::NormError = L1Norm(),
        dev::WeightDeviation = RelativeWeightDeviation()
    ) -> ErrorWeightFinaliser

Keywords correspond to the struct's fields.

## Validation

  - If `alg` is an [`LpNorm`](@ref): `alg.p > 1`.

# Examples

```jldoctest
julia> ErrorWeightFinaliser()
ErrorWeightFinaliser
  alg ┼ L1Norm
      │   ddof ┴ Int64: 0
  dev ┴ RelativeWeightDeviation()
```

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`WeightDeviation`](@ref)
  - [`NormError`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`DiscreteAllocation`](@ref)
"""
@concrete struct ErrorWeightFinaliser <: JuMPWeightFinaliserFormulation
    """
    $(field_dict[:wfnorm])
    """
    alg
    """
    $(field_dict[:wfdev])
    """
    dev
    function ErrorWeightFinaliser(alg::NormError, dev::WeightDeviation)
        if isa(alg, LpNorm)
            @argcheck(alg.p > 1,
                      DomainError(alg.p,
                                  "`LpNorm.p` is $(alg.p), and the model states the `p`-norm of the deviation with a power cone of exponent `1 / p`, so `1 < p` must hold. State a value greater than `1`."))
        end
        return new{typeof(alg), typeof(dev)}(alg, dev)
    end
end
function ErrorWeightFinaliser(; alg::NormError = L1Norm(),
                              dev::WeightDeviation = RelativeWeightDeviation())::ErrorWeightFinaliser
    return ErrorWeightFinaliser(alg, dev)
end
"""
    set_norm_error_epigraph!(model::JuMP.Model, t, r::AbstractVector, alg::NormError)

Adds the rows that bound the norm of `r` by `t`, ``t \\geq \\lVert \\boldsymbol{r} \\rVert``, and returns them.

Every row is scaled by the constraint scale ``s_c`` of `model`. The norm gives the order, and [`norm_epigraph!`](@ref) writes the rows.

# JuMP formulation

## Variables

  - `z`: One auxiliary variable for each entry of `r`, created with no key, only for an [`LpNorm`](@ref) of an order other than `2` and `Inf`.

## Constraints

The rows that `alg` selects:

  - [`L1Norm`](@ref): ``(s_c t,\\, s_c \\boldsymbol{r}) \\in \\mathcal{K}_{1}``.
  - [`L2Norm`](@ref) and [`SquaredL2Norm`](@ref): ``(s_c t,\\, s_c \\boldsymbol{r}) \\in \\mathcal{K}_{2}``.
  - [`LInfNorm`](@ref): ``(s_c t,\\, s_c \\boldsymbol{r}) \\in \\mathcal{K}_{\\infty}``.
  - [`LpNorm`](@ref): ``(s_c z_{i},\\, s_c t,\\, s_c r_{i}) \\in \\mathcal{P}_{1/p}`` for each ``i``, and ``s_c (\\boldsymbol{1}^\\intercal \\boldsymbol{z} - t) = 0``. At ``p = 2`` and ``p = \\infty`` it is the row of [`L2Norm`](@ref) or of [`LInfNorm`](@ref).

Where:

  - ``t``: The bound on the norm.
  - ``\\boldsymbol{r}``: The residual.
  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])
  - ``\\mathcal{P}_{a}``: The power cone ``\\{(x, y, z) : x^{a} y^{1 - a} \\geq \\lvert z \\rvert,\\, x, y \\geq 0\\}``.

# Arguments

  - `model`: The JuMP model. It must already hold the constraint scale `sc`.
  - `t`: The bound on the norm.
  - `r`: The residual.
  - `alg`: The norm.

# Returns

  - `(cone, csum, z)`: The cone row or rows, the sum row and the auxiliary variables. `csum` and `z` are `nothing` off the power cone route.

# Related

  - [`ErrorWeightFinaliser`](@ref)
  - [`NormError`](@ref)
  - [`norm_epigraph!`](@ref)
"""
function set_norm_error_epigraph!(model::JuMP.Model, t, r::AbstractVector, ::L1Norm)
    return norm_epigraph!(model, t, r, 1, get_constraint_scale(model))
end
function set_norm_error_epigraph!(model::JuMP.Model, t, r::AbstractVector,
                                  ::Union{L2Norm, SquaredL2Norm})
    return norm_epigraph!(model, t, r, 2, get_constraint_scale(model))
end
function set_norm_error_epigraph!(model::JuMP.Model, t, r::AbstractVector, ::LInfNorm)
    return norm_epigraph!(model, t, r, Inf, get_constraint_scale(model))
end
function set_norm_error_epigraph!(model::JuMP.Model, t, r::AbstractVector, alg::LpNorm)
    return norm_epigraph!(model, t, r, alg.p, get_constraint_scale(model))
end
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

  - `w::VecNum`: The repaired weight vector. `finalise_weight_bounds` then pairs it with a return code.

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
        alg::JuMPWeightFinaliserFormulation = ErrorWeightFinaliser()
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
  alg ┼ ErrorWeightFinaliser
      │   alg ┼ L1Norm
      │       │   ddof ┴ Int64: 0
      │   dev ┴ RelativeWeightDeviation()
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
                             alg::JuMPWeightFinaliserFormulation = ErrorWeightFinaliser())::JuMPWeightFinaliser
    return JuMPWeightFinaliser(slv, sc, so, alg)
end
"""
    set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                         alg::ErrorWeightFinaliser,
                                         wi::VecNum)

Adds the deviation objective of `alg` to the weight finalisation model.

[`opt_weight_bounds`](@ref) adds the weights `w`, the budget row and the bound rows before it calls this function. A [`RelativeWeightDeviation`](@ref) divides by `wi`, so [`weight_deviation`](@ref) replaces each zero entry of a copy of `wi` with `eps(eltype(wi))`. It does not change `wi`.

# JuMP formulation

## Variables

  - `w`: The weights, read from the model.
  - `t`: The bound on the deviation, created.

## Constraints

  - The rows of [`set_norm_error_epigraph!`](@ref) for the norm `alg.alg`, which bound the norm of the residual ``\\boldsymbol{d}`` by ``t``. They are unnamed.

## Objective

  - `Min`: ``s_o t``. The objective pulls ``t`` down, so ``t`` equals the norm at the optimum.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_0_finaliser])
  - ``\\boldsymbol{d}``: The residual of [`weight_deviation`](@ref) for `alg.dev`, ``\\boldsymbol{w} \\oslash \\tilde{\\boldsymbol{w}}_{0} - \\boldsymbol{1}`` for a [`RelativeWeightDeviation`](@ref), and ``\\boldsymbol{w} - \\boldsymbol{w}_{0}`` for an [`AbsoluteWeightDeviation`](@ref).
  - ``\\tilde{\\boldsymbol{w}}_{0}``: ``\\boldsymbol{w}_{0}`` with each zero entry replaced by `eps(eltype(wi))`, so that the division is defined.
  - ``t``: Bound on the norm of the deviation.
  - $(math_dict[:so_scale])
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

# Arguments

  - `model`: The JuMP model. It must already hold `w` and the two scales `sc` and `so`.
  - `alg`: The deviation and its norm.
  - `wi`: The weights that the optimisation produced.

# Returns

  - `nothing`.

# Related

  - [`JuMPWeightFinaliserFormulation`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
  - [`opt_weight_bounds`](@ref)
"""
function set_clustering_weight_finaliser_alg!(model::JuMP.Model, alg::ErrorWeightFinaliser,
                                              wi::VecNum)
    w = get_w(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t)
    set_norm_error_epigraph!(model, t, weight_deviation(alg.dev, w, wi, one(eltype(wi))),
                             alg.alg)
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
        @warn("The weight finalisation model of $(wf.alg) could not be solved, so the weights are finalised with IterativeWeightFinaliser() instead.",
              maxlog = log_limit(:warn), _id = @log_id(typeof(wf.alg)))
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

export IterativeWeightFinaliser, RelativeWeightDeviation, AbsoluteWeightDeviation,
       ErrorWeightFinaliser, JuMPWeightFinaliser
public JuMPWeightFinaliserFormulation, set_clustering_weight_finaliser_alg!,
       WeightFinaliser, opt_weight_bounds, WeightDeviation, weight_deviation,
       deviation_error_term
