"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the deviations that a JuMP weight finaliser can minimise.

A subtype is the field `alg` of a [`JuMPWeightFinaliser`](@ref), and the field `wf` of a [`DiscreteAllocation`](@ref). It selects the deviation from the target weights that the programme minimises. [`ErrorWeightFinaliser`](@ref) is the library's subtype.

# Interfaces

To add a deviation, subtype `JuMPWeightFinaliserFormulation` and implement the method below.

## `set_clustering_weight_finaliser_alg!`

  - `set_clustering_weight_finaliser_alg!(model::JuMP.Model, alg::MyFormulation, wi::VecNum) -> Nothing`: Adds the deviation objective to a model that already holds the weights `w` and every row of the programme. It gives the deviation to `set_divergence_objective!`, which adds the penalties of the block and sets the objective.

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

A subtype runs after the optimisation. It keeps the sum of the weights. Except for a [`JuMPWeightFinaliser`](@ref) whose block holds a slot, it changes weights that already lie in the bounds not at all.

# Interfaces

To add a method, subtype `WeightFinaliser` and implement the method below.

## `opt_weight_bounds`

  - `opt_weight_bounds(wf::MyFinaliser, wb::WeightBounds, w::VecNum, pr, rd, sets, fees, strict::Bool) -> VecNum`: Moves `w` into the bounds `wb`, and keeps the sum of `w`. A method that reads no context takes `ctx...` in place of the arguments after `w`.

### Arguments

  - `wf`: The concrete subtype instance.
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `w`: The weights the optimisation produced.
  - `pr`, `rd`, `sets`, `fees`, `strict`: The context of the head on the assets of `w`, which `finalise_weight_bounds` gives: the prior result or `nothing`, the returns data, the sets or `nothing`, the fees or `nothing`, and the flag for an unknown name.

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
 2. Read the budget `s1` as `sum(w)`, and keep the input as `w0`. Read an absent bound as `typemin` or `typemax` of the element type of `w`, which must be infinite, through [`absent_weight_bound`](@ref).
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
    opt_weight_bounds(wf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum, ctx...) -> VecNum
    opt_weight_bounds(wf::EuclideanWeightFinaliser, wb::WeightBounds, w::VecNum, ctx...) -> VecNum
    opt_weight_bounds(wf::EntropicWeightFinaliser, wb::WeightBounds, w::VecNum, ctx...) -> VecNum

Moves a weight vector into the bounds `wb`, and keeps the sum of the vector.

Every one of these methods returns weights that break no bound unchanged. The bounds do not change. Each method runs the procedure of its finaliser type:

  - [`IterativeWeightFinaliser`](@ref): the clip and redistribution that its docstring states.
  - [`EuclideanWeightFinaliser`](@ref): the Euclidean projection, [`euclidean_weight_projection`](@ref).
  - [`EntropicWeightFinaliser`](@ref): the entropic projection, [`entropic_weight_projection`](@ref), or the Euclidean projection where the entropic one is not defined.

# Arguments

  - `wf`: The weight finaliser.
  - `wb`: The weight bounds. Either bound can be `nothing`.
  - `w`: The weights that the optimisation produced.
  - `ctx...`: The context of the head, which these methods do not read. [`finalise_weight_bounds`](@ref) states its arguments.

# Returns

  - `w::VecNum`: The weights in the bounds, or the input when it breaks no bound.

# Related

  - [`WeightBounds`](@ref)
  - [`JuMPWeightFinaliser`](@ref): its method solves a programme, and it reads the context.
  - [`IterativeWeightFinaliser`](@ref)
  - [`EuclideanWeightFinaliser`](@ref)
  - [`EntropicWeightFinaliser`](@ref)
  - [`finalise_weight_bounds`](@ref)
"""
function opt_weight_bounds end
"""
    finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum, pr = nothing,
                           rd = nothing, sets = nothing, fees = nothing,
                           strict::Bool = false)

Moves the weights into their bounds, and gives the return code that states whether the move succeeded.

A failure lets the fallback chain of [`optimise`](@ref) run. A set of bounds that cannot hold the budget, `sum(lb) > sum(w)` or `sum(ub) < sum(w)`, always fails.

The arguments after `w` are the context of the head. A [`JuMPWeightFinaliser`](@ref) reads them to resolve its [`ConstraintBlock`](@ref), and the other finalisers of the library read none of them. A head gives every one of them on the assets of `w`.

# Algorithm

 1. Read the budget `s` as `sum(w)`.
 2. Move `w` into the bounds with [`opt_weight_bounds`](@ref).
 3. Give the return code of [`weight_bounds_retcode`](@ref), together with the moved weights.

# Arguments

  - `wf::WeightFinaliser`: The weight finaliser.
  - `wb::WeightBounds`: The weight bounds.
  - `w::VecNum`: The weights that the optimisation produced.
  - $(arg_dict[:pr]) It is `nothing` for a head that fits no prior.
  - $(arg_dict[:rd])
  - $(arg_dict[:sets]) It is `nothing` when the head holds none.
  - $(arg_dict[:fees_opt])
  - $(arg_dict[:strict])

# Returns

  - `(retcode, w)`: The return code and the moved weights.

# Related

  - [`WeightFinaliser`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_retcode`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum,
                                pr = nothing, rd = nothing, sets = nothing, fees = nothing,
                                strict::Bool = false)
    s = sum(w)
    w = opt_weight_bounds(wf, wb, w, pr, rd, sets, fees, strict)
    return weight_bounds_retcode(wf, wb, w, s), w
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gives the return code of finalised weights, from a test of the bounds and the budget.

# Arguments

  - `wf`: The weight finaliser.
  - `wb`: The weight bounds.
  - `w`: The finalised weights.
  - `s`: The budget of the weights before the finaliser moved them.

# Returns

  - `retcode::OptimisationReturnCode`: An [`OptimisationSuccess`](@ref) when the weights meet the test of [`weights_meet_bounds`](@ref): they are finite, they lie in the bounds, and they sum to `s`, each to a tolerance. Otherwise an [`OptimisationFailure`](@ref) that names `wf` and `wb`.

# Related

  - [`finalise_weight_bounds`](@ref)
  - [`weights_meet_bounds`](@ref)
"""
function weight_bounds_retcode(wf::WeightFinaliser, wb::WeightBounds, w::VecNum, s::Number)
    return if weights_meet_bounds(wb, w, s)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "The weights do not meet the bounds.\n$wf\n$wb.")
    end
end
"""
    finaliser_readers(wf::WeightFinaliser) -> Tuple
    finaliser_readers(wf::JuMPWeightFinaliser) -> Tuple

Gives the distance readers of a weight finaliser, for the Investable Mask of its head.

A head composes its mask with the assets that its readers can read, see [`distance_reader_entry_mask`](@ref). The readers of a [`JuMPWeightFinaliser`](@ref) are the slots `ple` and `cte` of its block. Every other finaliser has none.

# Returns

  - `readers::Tuple`: The readers of `wf`.

# Related

  - [`distance_reader_entry_mask`](@ref)
  - [`fit_finaliser_readers`](@ref)
"""
function finaliser_readers(::WeightFinaliser)::Tuple
    return ()
end
"""
    investable_finaliser(wf::WeightFinaliser, sets, strict::Bool, imsk, X::MatNum) -> WeightFinaliser
    investable_finaliser(wf::JuMPWeightFinaliser, sets, strict::Bool, imsk, X::MatNum) -> JuMPWeightFinaliser

Places the weight finaliser of a head on the assets of the Investable Mask, as [`investable_fees_view`](@ref) places its fees.

A [`JuMPWeightFinaliser`](@ref) holds a [`ConstraintBlock`](@ref) with no `sets` of its own. So the head resolves the names of the block over its own `sets`, on the universe of the caller, before [`investable_reduction`](@ref) narrows them. The values then follow a view at the mask. This is stage 1 of the block, [`resolve_constraint_block_names`](@ref), in the order of a JuMP head. The method for any other finaliser returns it unchanged.

# Arguments

  - `wf`: The weight finaliser of the head, on the universe of the caller.
  - $(arg_dict[:sets]) They are the sets of the head before the reduction, or `nothing`.
  - $(arg_dict[:strict])
  - `imsk`: The Investable Mask of the head, or `nothing` when every asset is investable.
  - `X`: The returns matrix of the full universe. It gives the element type of the resolved values, and the tracking errors of the block view it.

# Returns

  - `wf::WeightFinaliser`: The finaliser on the assets of the mask.

# Related

  - [`resolve_constraint_block_names`](@ref)
  - [`investable_fees_view`](@ref)
  - [`finalise_weight_bounds`](@ref)
"""
function investable_finaliser(wf::WeightFinaliser, ::Any, ::Bool, ::Any, ::MatNum)
    return wf
end
"""
    fit_finaliser_readers(wf::WeightFinaliser, pr, rd, strict::Bool) -> WeightFinaliser
    fit_finaliser_readers(wf::JuMPWeightFinaliser, pr::AbstractPriorResult, rd::ReturnsResult, strict::Bool) -> JuMPWeightFinaliser

Fits the distance readers of a weight finaliser on the reduced prior and rows of its head.

A distance reader can refuse an asset that the prior keeps, see [`DistanceReaderRefusalError`](@ref). A head calls this function through [`distance_reader_call`](@ref) before its own solve, so a refusal costs no solve, and the head fits again without the asset. The method for any other finaliser returns it unchanged.

# Arguments

  - `wf`: The weight finaliser, on the assets of the mask.
  - $(arg_dict[:pr])
  - $(arg_dict[:rd])
  - $(arg_dict[:strict])

# Returns

  - `wf::WeightFinaliser`: The finaliser, with the results of its readers in the slots `cte` and `ple` of its block.

# Related

  - [`distance_reader_call`](@ref)
  - [`finaliser_readers`](@ref)
  - [`distance_reader_refusal_mask`](@ref)
"""
function fit_finaliser_readers(wf::WeightFinaliser, ::Any, ::Any, ::Bool)
    return wf
end
"""
    assert_prior_free_finaliser(wf, head::Symbol) -> Nothing
    assert_prior_free_finaliser(wf::JuMPWeightFinaliser, head::Symbol) -> Nothing

Refuses a weight finaliser whose block reads a prior, in the constructor of a head that fits none.

[`EqualWeighted`](@ref), [`RandomWeighted`](@ref) and [`BestConstantRebalancedPortfolio`](@ref) fit no prior, so a slot that reads one has no value there ([`block_reads_prior`](@ref)). The method for any other finaliser, and for a [`TimeDependent`](@ref) schedule, checks nothing. The constructor of the head runs again with each entry of a schedule.

# Arguments

  - `wf`: The weight finaliser of the head.
  - `head`: The name of the head, for the message.

# Validation

  - No slot of the block of `wf` reads a prior. An [`IsNothingError`](@ref) is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`block_reads_prior`](@ref)
  - [`JuMPWeightFinaliser`](@ref)
"""
function assert_prior_free_finaliser(::Any, ::Symbol)::Nothing
    return nothing
end

export IterativeWeightFinaliser, RelativeWeightDeviation, AbsoluteWeightDeviation,
       ErrorWeightFinaliser
public JuMPWeightFinaliserFormulation, WeightFinaliser, opt_weight_bounds, WeightDeviation,
       weight_deviation, deviation_error_term
