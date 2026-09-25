"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the tail view constraints of an entropy pooling problem.

A tail view constraint is the parsed, resolved form of a conditional or entropic value-at-risk view. It carries the loss series, the level, the operator and the target, in the shape the formulation that produced it needs. Unlike the linear views, which reduce to rows of a matrix that multiplies the posterior probabilities, a tail view constraint needs auxiliary variables, so it is handed to the optimiser as a struct and built into the model there.

# Related

  - [`entropy_pooling`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
abstract type AbstractEntropyPoolingTailView <: AbstractResult end
"""
    const VecEPTV = AbstractVector{<:AbstractEntropyPoolingTailView}

Alias for an abstract vector of [`AbstractEntropyPoolingTailView`](@ref) elements.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
"""
const VecEPTV = AbstractVector{<:AbstractEntropyPoolingTailView}
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                    opt::AbstractEntropyPoolingOptimiser)

Solve an entropy pooling problem that carries tail view constraints.

`entropy_pooling` extends the three-argument form with the conditional and entropic value-at-risk views of [EPTail](@cite). A tail view needs auxiliary variables, so it is built into the model by [`add_ep_tail_view!`](@ref) rather than reduced to rows of `epc`.

These methods carry the body of the [`JuMPEntropyPooling`](@ref) route. The [`OptimEntropyPooling`](@ref) method solves the dual, which has no room for an auxiliary variable, so it accepts an empty `tvs` alone and forwards to the three-argument form.

# Mathematical definition

The primal of the entropy pooling problem minimises the Kullback-Leibler divergence of the posterior probabilities from the prior ones, over the probabilities that meet every row:

```math
\\begin{align}
\\underset{\\boldsymbol{p}}{\\min} \\;\\; &\\sum_{t=1}^{T} p_{t} \\ln\\!\\left(\\dfrac{p_{t}}{q_{t}}\\right)\\\\
\\textrm{s.t.} \\;\\; &\\sum_{t=1}^{T} p_{t} = 1\\\\
&\\boldsymbol{p} \\geq \\boldsymbol{0}\\\\
&\\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{p} = \\boldsymbol{B}_{\\mathrm{eq}}\\\\
&\\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{p} \\leq \\boldsymbol{B}_{\\mathrm{ineq}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:eq])
  - $(math_dict[:ineq])
  - $(math_dict[:T])
  - ``q_{t}``, ``p_{t}``: Prior and posterior probability of observation ``t``.

# Algorithm

 1. Raise when `opt` is an [`OptimEntropyPooling`](@ref) and `tvs` names a tail view, and forward to the three-argument form otherwise.
 2. Return `w` when `epc` holds no row and `tvs` names no tail view. An empty view set states nothing, so the posterior is the prior, and it is answered exactly rather than solved for.
 3. Solve once with [`ep_jump_entropy_pooling`](@ref), giving `w1`.
 4. Read `iters`, the largest number of re-solves a carrier of `tvs` asks for, with [`ep_refine_iters`](@ref). It is zero where no carrier is sequential, and the steps below then do not run.
 5. Re-read every carrier at `w1` with [`ep_refine_tail_view`](@ref), which returns the carrier and whether its surrogate row is tight there. Stop where every carrier is tight.
 6. Solve again with the re-read carriers, giving a new `w1`, and return to step 5. Take at most `iters` re-solves, so the last posterior can hold a slack row when the sequence has not settled.
 7. Check every carrier at `w1` with [`ep_check_tail_window`](@ref), which warns where the window of an integer carrier can bind.
 8. Return `w1` as `StatsBase.pweights`.

The re-solves are the sequential convex formulations' half of the work. Each re-read row is tight at the posterior it was read at and still holds there, so that posterior stays feasible and the divergence of the next one is at most its own.

A stop on tight rows means that the sequence reached a fixed point, not that the posterior is the one of least divergence. The feasible set of a sequential view is not convex, so the fixed point can be a local minimiser. More re-solves or a smaller `tol` do not move it. On a sample of sixty observations, a view that bounds a CVaR at 0.7 times its prior value stopped with tight rows at a divergence 7.5% above the least one. No flag of the result marks this case.

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.

  - `tvs`: Tail view constraints.

  - `opt`: Entropy pooling optimiser.

      + `::JuMPEntropyPooling`: Builds every tail view into the model.
      + `::OptimEntropyPooling`: Solves the dual, which has no room for an auxiliary variable, so it accepts an empty `tvs` alone.

# Validation

  - `isa(opt, OptimEntropyPooling)` requires `isempty(tvs)`.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Related

  - [`ep_jump_entropy_pooling`](@ref)
  - [`ep_refine_tail_view`](@ref)
  - [`ep_refine_iters`](@ref)
  - [`ep_check_tail_window`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                         opt::OptimEntropyPooling)
    @argcheck(isempty(tvs),
              ArgumentError("$(join(unique(nameof.(typeof.(tvs))), ", ", " and ")) can only be expressed with auxiliary variables, which the dual formulation `OptimEntropyPooling` solves has no room for. Use `JuMPEntropyPooling` in `opt`."))
    return entropy_pooling(w, epc, opt)
end
function entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                         opt::JuMPEntropyPooling)
    #! See the twin note in the `OptimEntropyPooling` methods: an empty view set states
    #! nothing, so the posterior is the prior. Issue #852.
    if isempty(epc) && isempty(tvs)
        return isa(w, StatsBase.ProbabilityWeights) ? w : StatsBase.pweights(w)
    end
    w1 = ep_jump_entropy_pooling(w, epc, tvs, opt)
    iters = ep_refine_iters(tvs)
    tvs = AbstractEntropyPoolingTailView[tv for tv in tvs]
    for _ in 1:iters
        tight = true
        for (k, tv) in pairs(tvs)
            tvs[k], ok = ep_refine_tail_view(tv, w1)
            tight &= ok
        end
        if tight
            break
        end
        w1 = ep_jump_entropy_pooling(w, epc, tvs, opt)
    end
    foreach(tv -> ep_check_tail_window(tv, w1), tvs)
    return w1
end
"""
    ep_refine_iters(tv::AbstractEntropyPoolingTailView)
    ep_refine_iters(tvs::VecEPTV)

Read the number of re-solves a tail view carrier asks [`entropy_pooling`](@ref) for.

A carrier whose rows are fixed at construction asks for none. A sequential carrier asks for the `iters` its formulation holds, and a vector of carriers asks for the largest number among its entries, so one loop serves every sequential view of the model.

# Arguments

  - `tv`: Tail view constraint.
  - `tvs`: Tail view constraints.

# Returns

  - `iters::Integer`: Number of re-solves, zero for a carrier with fixed rows.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_refine_tail_view`](@ref)
  - [`AbstractEntropyPoolingTailView`](@ref)
"""
function ep_refine_iters(::AbstractEntropyPoolingTailView)
    return 0
end
function ep_refine_iters(tvs::VecEPTV)
    return maximum(ep_refine_iters, tvs; init = 0)
end
"""
    ep_refine_tail_view(tv::AbstractEntropyPoolingTailView, w::VecNum)

Re-read a tail view carrier at a posterior, and say whether its rows were already tight there.

A carrier whose rows are fixed at construction is returned unchanged, and is always tight. A sequential carrier re-reads the multipliers of its primal side at `w`, which is what tightens its surrogate row between two solves of [`entropy_pooling`](@ref); its method lives beside it in `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_a.jl`.

# Arguments

  - `tv`: Tail view constraint.
  - `w`: Posterior probabilities of the last solve.

# Returns

  - `tv::AbstractEntropyPoolingTailView`: The carrier to solve with next.
  - `tight::Bool`: Whether the rows `tv` held before the call were tight at `w`, within the tolerance the carrier holds. A tight row marks a fixed point of the sequence, which can be a local minimiser of the divergence and not the global one.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_refine_iters`](@ref)
  - [`AbstractEntropyPoolingTailView`](@ref)
"""
function ep_refine_tail_view(tv::AbstractEntropyPoolingTailView, ::VecNum)
    return tv, true
end
"""
    ep_check_tail_window(tv::AbstractEntropyPoolingTailView, w::VecNum)

Warn where the window of a tail view carrier can restrict the posterior.

Only [`IntegerConditionalValueatRiskViewConstraint`](@ref) reads a window: its model admits the posteriors that put at least `alpha` of their mass on the `sbar` largest losses of each asset. Where that restriction binds, the window holds exactly `alpha`, and the view can have a posterior of smaller divergence outside it, so the method warns. Every other carrier reads the whole sample and does nothing. The integer method lives in `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_b.jl`.

# Arguments

  - `tv`: Tail view constraint.
  - `w`: Posterior probabilities of the last solve.

# Returns

  - `nothing`.

# Related

  - [`entropy_pooling`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`AbstractEntropyPoolingTailView`](@ref)
"""
function ep_check_tail_window(::AbstractEntropyPoolingTailView, ::VecNum)
    return nothing
end
"""
    ep_jump_entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                            opt::JuMPEntropyPooling)

Build and solve the entropy pooling model of one set of tail view carriers, once.

`ep_jump_entropy_pooling` is the body of the [`JuMPEntropyPooling`](@ref) route of [`entropy_pooling`](@ref), which calls it once per set of carriers and re-calls it after re-reading the sequential ones. The two methods differ only in how they represent the divergence, and agree on every view.

# Algorithm

 1. Read `sc1`, `sc2`, `so` and `slv` off `opt`, and open an empty `JuMP.Model`.
 2. Create the posterior variable `x` and the divergence variable `t`. Under [`ExpEntropyPooling`](@ref) `x` also carries a non-negativity bound, which the cone of step 4 implies in either branch.
 3. Add the row that pins the sum of `x` to one.
 4. Add the relative entropy cone that bounds `t` below by the divergence. Under [`ExpEntropyPooling`](@ref) the cone reads the prior `w` as its reference, and under [`LogEntropyPooling`](@ref) it reads a unit reference instead.
 5. Register `obj_expr` as `so * t`.
 6. Add every view with [`ep_jump_views!`](@ref), which mutates `obj_expr` when a fixed equality is relaxed.
 7. Set the objective to `obj_expr` under [`ExpEntropyPooling`](@ref), and to `obj_expr` less `so` times the inner product of `x` with the prior log-probabilities under [`LogEntropyPooling`](@ref).
 8. Solve with `slv`, and raise when no solver configuration succeeds.
 9. Clamp the value of `x` at zero, because the solver meets the non-negativity of step 4 to its feasibility tolerance only, and a mixed-integer solve can return an entry slightly below zero. Divide by its sum, and return it as `StatsBase.pweights`.

# JuMP formulation

## Variables

  - `t`: ``\\eta``, the epigraph variable of the divergence, created here.
  - `x`: ``\\boldsymbol{p}``, the posterior probabilities, created here. Under [`ExpEntropyPooling`](@ref) it carries the bound ``\\boldsymbol{p} \\geq \\boldsymbol{0}``.

## Expressions

  - `obj_expr`: ``s_{o} \\eta``. [`ep_jump_views!`](@ref) adds ``s_{o} s_{c2} \\eta_{c}`` to it when `epc` holds a `:feq` block.

## Constraints

The two rows below are registered under no name.

  - ``s_{c1} \\left(\\sum_{t=1}^{T} p_{t} - 1\\right) = 0``
  - ``\\left(s_{c1} \\eta,\\; s_{c1} \\boldsymbol{q},\\; s_{c1} \\boldsymbol{p}\\right) \\in \\mathcal{K}_{\\mathrm{relent}}``, which states ``\\eta \\geq \\sum_{t=1}^{T} p_{t} \\ln(p_{t} / q_{t})``. Under [`LogEntropyPooling`](@ref) the reference is ``s_{c1} \\boldsymbol{1}`` in place of ``s_{c1} \\boldsymbol{q}``, so the row states ``\\eta \\geq \\sum_{t=1}^{T} p_{t} \\ln(p_{t})``.

[`ep_jump_views!`](@ref) registers every further row of the model, and its own `# JuMP formulation` names them.

## Objective

  - `Min` of `obj_expr` under [`ExpEntropyPooling`](@ref).
  - `Min` of `obj_expr` less ``s_{o} \\boldsymbol{p}^{\\intercal} \\ln(\\boldsymbol{q})`` under [`LogEntropyPooling`](@ref). The subtraction turns the cone's ``\\sum_{t} p_{t} \\ln(p_{t})`` into the divergence, so both branches minimise the same quantity.

## Relaxation

$(val_dict[:relax])

  - The bound is on `obj_expr`, and it lies **above** the divergence of the exactly constrained problem. [`ep_jump_views!`](@ref) relaxes the `:feq` rows with a penalised slack, and the penalty enters `obj_expr`.
  - The bound is tight when that slack is zero, which holds when the `:feq` rows and the views can be met together, and ``s_{c2}`` is large enough to pull the slack down.

Where:

  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:T])
  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_sc2])
  - $(math_dict[:ep_so])
  - ``\\eta``: Epigraph variable that bounds the divergence from above. Its model key is `t`.
  - ``\\eta_{c}``: Norm of the fixed equality slack, which [`ep_jump_views!`](@ref) creates under the model key `tc`.
  - ``\\mathcal{K}_{\\mathrm{relent}}``: Relative entropy cone, `JuMP.MOI.RelativeEntropyCone(2T + 1)`.

# Arguments

  - `w`: Prior weights (length = number of observations).
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints.
  - `opt`: Entropy pooling optimiser.

# Validation

  - One configuration of `opt.slv` solves the model. Otherwise an `ErrorException` is raised.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_jump_views!`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function ep_jump_entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                                 opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                         <:ExpEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    T = length(w)
    model = JuMP.Model()
    JuMP.@variables(model, begin
                        t
                        x[1:T] >= 0
                    end)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(x) - one(eltype(w))) == 0
                          [sc1 * t; sc1 * w; sc1 * x] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
    JuMP.@expression(model, obj_expr, so * t)
    ep_jump_views!(model, x, obj_expr, epc, tvs, sc1, sc2, so)
    JuMP.@objective(model, Min, obj_expr)
    @argcheck(optimise_JuMP_model!(model, slv).success,
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    # The solver meets `x >= 0` to its feasibility tolerance, so an entry can come back a
    # hair below zero, and every weighted estimator of the refit refuses it. Issue #1260.
    return StatsBase.pweights(LinearAlgebra.normalize(max.(JuMP.value.(x), 0), 1))
end
function ep_jump_entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                                 opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                         <:LogEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    model = JuMP.Model()
    T = length(w)
    log_p = log.(w)
    JuMP.@variables(model, begin
                        x[1:T]
                        t
                    end)
    JuMP.@expression(model, obj_expr, so * t)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(x) - one(eltype(w))) == 0
                          [sc1 * t; fill(sc1, T); sc1 * x] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
    ep_jump_views!(model, x, obj_expr, epc, tvs, sc1, sc2, so)
    JuMP.@objective(model, Min, obj_expr - so * LinearAlgebra.dot(x, log_p))
    @argcheck(optimise_JuMP_model!(model, slv).success,
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    # The solver meets `x >= 0` to its feasibility tolerance, so an entry can come back a
    # hair below zero, and every weighted estimator of the refit refuses it. Issue #1260.
    return StatsBase.pweights(LinearAlgebra.normalize(max.(JuMP.value.(x), 0), 1))
end
"""
    ep_jump_views!(model::JuMP.Model, x, obj_expr, epc::AbstractDict, tvs::VecEPTV,
                   sc1::Number, sc2::Number, so::Number)

Add every view constraint of an entropy pooling problem to a JuMP model.

`ep_jump_views!` is the shared body of the two [`JuMPEntropyPooling`](@ref) formulations: they differ only in how they represent the divergence, and agree on every view. It adds the linear rows of `epc`, relaxes the fixed equalities with a norm one cone bounded slack, and hands each tail view to [`add_ep_tail_view!`](@ref).

# Algorithm

 1. Add the row `ceq` when `epc` holds an `:eq` block.
 2. Add the row `cineq` when `epc` holds an `:ineq` block.
 3. Add the row `ccvareq` when `epc` holds a `:cvar_eq` block.
 4. When `epc` holds a `:feq` block, create the slack `c` and its norm `tc`, add the row `cfeq` that lets the block miss by `c`, bound the norm of `c` by `tc`, and add the penalty `so * sc2 * tc` to `obj_expr`.
 5. Hand each tail view of `tvs` to [`add_ep_tail_view!`](@ref) in turn.

# JuMP formulation

## Variables

  - `x`: ``\\boldsymbol{p}``, the posterior probabilities, read from the model.
  - `tc`: ``\\eta_{c}``, the norm of the fixed equality slack, created when `epc` holds a `:feq` block.
  - `c`: ``\\boldsymbol{c}``, the fixed equality slack, one entry per row of the `:feq` block, created with `tc`.

## Constraints

  - `ceq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{eq}}\\right) = 0``
  - `cineq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{ineq}}\\right) \\leq 0``
  - `ccvareq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{cvar}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{cvar}}\\right) = 0``
  - `cfeq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{feq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{feq}} - \\boldsymbol{c}\\right) = 0``
  - Registered under no name, alongside `cfeq`: ``\\left(s_{c1} \\eta_{c},\\; s_{c1} \\boldsymbol{c}\\right) \\in \\mathcal{K}_{1}``, which states ``\\eta_{c} \\geq \\lVert \\boldsymbol{c} \\rVert_{1}``.

Each of the four named rows is registered only when `epc` holds the block it carries. [`add_ep_tail_view!`](@ref) registers every row a tail view needs, in `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_a.jl` and `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_b.jl`.

## Relaxation

$(val_dict[:relax])

  - The bound is on the objective, and it lies **above** the objective of the problem whose `:feq` rows are enforced exactly. The rows `cfeq` let the `:feq` block miss its right-hand side by ``\\boldsymbol{c}``, and the penalty ``s_{o} s_{c2} \\eta_{c}`` is what holds ``\\boldsymbol{c}`` near zero.
  - The bound is tight when ``\\boldsymbol{c}`` is zero, which holds when the `:feq` rows and the views can be met together, and ``s_{c2}`` is large enough to pull the slack down. A `:feq` row states a moment the views must leave where the prior put it, so the two can genuinely disagree, and then the row yields.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:eq])
  - $(math_dict[:ineq])
  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_sc2])
  - $(math_dict[:ep_so])
  - ``\\mathrm{cvar}``, ``\\mathrm{feq}``: Subscripts for the `:cvar_eq` and `:feq` blocks of `epc`.
  - ``\\boldsymbol{c}``: Slack of the fixed equality rows, one entry per row of the `:feq` block.
  - ``\\eta_{c}``: Bound on the ``L_{1}`` norm of ``\\boldsymbol{c}``.
  - ``\\mathcal{K}_{1}``: Norm one cone, `JuMP.MOI.NormOneCone(N + 1)`.

# Arguments

  - `model`: Entropy pooling JuMP model.
  - `x`: Vector of posterior probability variables.
  - `obj_expr`: Objective expression, mutated when a fixed equality is relaxed.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints.
  - `sc1`: Constraint scaling factor.
  - `sc2`: Fixed equality slack penalty.
  - `so`: Objective scaling factor.

# Returns

  - `nothing`: The function mutates `model` and `obj_expr` in-place.

# Related

  - [`entropy_pooling`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
function ep_jump_views!(model::JuMP.Model, x, obj_expr, epc::AbstractDict, tvs::VecEPTV,
                        sc1::Number, sc2::Number, so::Number)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        JuMP.@constraint(model, ceq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        JuMP.@constraint(model, cineq, sc1 * (A * x ⊖ B) <= 0)
    end
    if haskey(epc, :cvar_eq)
        A, B = epc[:cvar_eq]
        JuMP.@constraint(model, ccvareq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :feq)
        A, B = epc[:feq]
        N = length(B)
        JuMP.@variables(model, begin
                            tc
                            c[1:N]
                        end)
        JuMP.@constraints(model, begin
                              cfeq, sc1 * (A * x ⊖ B ⊖ c) == 0
                              [sc1 * tc; sc1 * c] in JuMP.MOI.NormOneCone(N + 1)
                          end)
        JuMP.add_to_expression!(obj_expr, so * sc2 * tc)
    end
    for tv in tvs
        add_ep_tail_view!(model, x, tv, sc1)
    end
    return nothing
end
