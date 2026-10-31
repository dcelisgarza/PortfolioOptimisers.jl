"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the tail view constraints of an entropy pooling problem.

A tail view constraint is the parsed and resolved form of a conditional, entropic or relativistic value-at-risk view. It holds the loss series, the level, the operator and the target, in the shape that its formulation needs. A linear view becomes rows of a matrix that multiplies the posterior probabilities. A tail view constraint needs auxiliary variables, so the library gives it to the optimiser as a struct, and the optimiser builds it into the model.

# Related

  - [`entropy_pooling`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Sections 2.2 and 2.3.
  - $(ref_dict[:EPRLVaR]) Section 3.2.
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

Solve an entropy pooling problem that has tail view constraints.

This form of `entropy_pooling` adds two kinds of views to the three-argument form. They are the conditional and entropic value-at-risk views of [EPTail](@cite), and the relativistic value-at-risk views of [EPRLVaR](@cite). A tail view needs auxiliary variables, so [`add_ep_tail_view!`](@ref) builds it into the model, and it does not become rows of `epc`.

These methods hold the body of the [`JuMPEntropyPooling`](@ref) optimiser. The [`OptimEntropyPooling`](@ref) method solves the dual problem, which cannot hold an auxiliary variable. So it accepts only an empty `tvs`, and calls the three-argument form.

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

 1. If `opt` is an [`OptimEntropyPooling`](@ref), raise when `tvs` names a tail view, and call the three-argument form otherwise.
 2. Return `w` when `epc` holds no row and `tvs` names no tail view. An empty view set states nothing, so the posterior is the prior, and the method returns it exactly with no solve.
 3. Solve once with [`ep_jump_entropy_pooling`](@ref). The result is `w1`.
 4. Read `iters`, the largest number of re-solves that a tail view constraint of `tvs` asks for, with [`ep_refine_iters`](@ref). It is zero when no tail view constraint is sequential, and the next two steps then do not run.
 5. Read every tail view constraint again at `w1` with [`ep_refine_tail_view`](@ref). It returns the tail view constraint, and whether its surrogate row is tight at `w1`. Stop when every tail view constraint is tight.
 6. Solve again with the new tail view constraints. The result is a new `w1`. Go back to step 5. Do at most `iters` re-solves, so the last posterior can have a slack row when the sequence did not reach a fixed point.
 7. Check every tail view constraint at `w1` with [`ep_check_tail_window`](@ref), which warns when the window of an integer tail view constraint can bind.
 8. Return `w1`, which [`ep_jump_entropy_pooling`](@ref) returns in the kind of `w`.

[EPTail](@citet) and [EPRLVaR](@citet) state no re-solve. The loop of steps 4 to 6 is the library's own.

The sequential convex formulations need the re-solves. Each new row is tight at the posterior at which the method reads it, and holds there. So that posterior stays feasible, and the divergence of the next posterior is at most its divergence.

A stop on tight rows means that the sequence reached a fixed point, not that the posterior is the one of least divergence. The feasible set of a sequential view is not convex, so the fixed point can be a local minimiser. More re-solves or a smaller `tol` do not move it. On a sample of sixty observations, a view that bounds a CVaR at 0.7 times its prior value stopped with tight rows at a divergence 7.5% above the least one. No flag of the result marks this case.

# Arguments

  - `w`: Prior probabilities, one for each observation.

  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.

  - `tvs`: Tail view constraints.

  - `opt`: Entropy pooling optimiser.

      + `::JuMPEntropyPooling`: Build every tail view into the model.
      + `::OptimEntropyPooling`: Solve the dual problem, which cannot hold an auxiliary variable, so accept only an empty `tvs`.

# Validation

  - `isa(opt, OptimEntropyPooling)` requires `isempty(tvs)`.

# Returns

  - `pw::ScenarioProbabilities`: Posterior probabilities that meet the constraints, in the kind of `w`.

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

  - $(ref_dict[:meucci2008]) Section 4, Equations 25 to 27.
  - $(ref_dict[:EPTail]) Section 2.1, Equation 1, and Sections 2.2 and 2.3.
  - $(ref_dict[:EPRLVaR]) Section 3.2.
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
        return scenario_probabilities(w)
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

Read the number of re-solves that a tail view constraint asks [`entropy_pooling`](@ref) for.

A tail view constraint whose rows are fixed at construction asks for none. A sequential tail view constraint asks for the `iters` of its formulation. A vector of tail view constraints asks for the largest number of its entries, so one loop does the re-solves of every sequential view of the model.

# Arguments

  - `tv`: Tail view constraint.
  - `tvs`: Tail view constraints.

# Returns

  - `iters::Integer`: Number of re-solves, zero for a tail view constraint with fixed rows.

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

Read a tail view constraint again at a posterior, and return whether its rows were already tight there.

The method returns a tail view constraint whose rows are fixed at construction with no change, and that constraint is always tight. A sequential tail view constraint reads the multipliers of its primal side again at `w`. This makes its surrogate row tighter between two solves of [`entropy_pooling`](@ref). Its method is in `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_a.jl`.

# Arguments

  - `tv`: Tail view constraint.
  - `w`: Posterior probabilities of the last solve.

# Returns

  - `tv::AbstractEntropyPoolingTailView`: The tail view constraint to solve with next.
  - `tight::Bool`: Whether the rows that `tv` held before the call were tight at `w`, within the tolerance of the tail view constraint. A tight row marks a fixed point of the sequence, which can be a local minimiser of the divergence and not the global one.

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

Warn when the window of a tail view constraint can restrict the posterior.

Only [`IntegerConditionalValueatRiskViewConstraint`](@ref) reads a window. Its model admits only the posteriors that put at least `alpha` of their mass on the `sbar` largest losses of each asset. When that restriction binds, the window holds exactly `alpha`, and a posterior of smaller divergence outside the window can meet the view. The method then warns. Every other tail view constraint reads the full sample, and for it the method does nothing. The integer method is in `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_b.jl`.

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
    ep_diagnose_tail_view!(model::JuMP.Model, p, tv::AbstractEntropyPoolingTailView)

Add a tail view constraint to the feasibility programme of [`ep_diagnose_views`](@ref) when the constraint states its view exactly and convexly.

The programme must refuse only a view set that no posterior meets. So it holds a tail view constraint only when the constraint is exact. [`LinearConditionalValueatRiskViewConstraint`](@ref), [`ConicEntropicValueatRiskViewConstraint`](@ref) and [`ConicRelativisticValueatRiskViewConstraint`](@ref) are exact and convex, and their method adds them with [`add_ep_tail_view!`](@ref). That method is in `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_b.jl`.

Every other tail view constraint is a restriction of its view: the surrogate row of a sequential formulation, the rows of an upper bound under [`GridTailView`](@ref), and the window of the integer formulation. A restriction can be infeasible where its view is not, so this method adds nothing for it.

# Arguments

  - `model`: The JuMP model of the feasibility programme.
  - `p`: The posterior probability variables of the model.
  - `tv`: Tail view constraint.

# Returns

  - `nothing`: The function mutates `model` in place, or does nothing.

# Related

  - [`ep_diagnose_views`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`AbstractEntropyPoolingTailView`](@ref)
"""
function ep_diagnose_tail_view!(::JuMP.Model, p, ::AbstractEntropyPoolingTailView)
    return nothing
end
"""
    ep_diagnose_views(slv::Option{<:Slv_VecSlv}, w::VecNum, epc::AbstractDict,
                      tvs::VecEPTV, pe::AbstractPriorEstimator, fields::Tuple)

Refuse a set of entropy pooling views that no reweighting of the scenarios meets together.

[`ep_assert_attainable_rows`](@ref) checks each row of a view alone. A set of rows can pass that check row by row and still have no common posterior. Then the dual of [`OptimEntropyPooling`](@ref) runs to an unbounded multiplier, and the fit fails in a consumer of the weights, or returns a degenerate posterior. [`JuMPEntropyPooling`](@ref) reports an infeasible model, but its error names no view.

This function decides joint feasibility exactly with a linear programme, before the solve of each stage. It costs one solve of the programme for each stage, so it runs only when the `diagnose` field of the estimator holds solvers. With `nothing`, the method does nothing.

# Mathematical definition

The programme finds the largest margin ``\\tau`` by which a posterior stays above a fraction of the prior on its support ``\\mathcal{S}``:

```math
\\begin{align}
\\underset{\\boldsymbol{p},\\, \\tau}{\\max} &\\; \\tau \\\\
\\mathrm{s.t.} &\\; \\sum_{t=1}^{T} p_{t} = 1\\,, \\quad \\boldsymbol{p} \\geq 0\\,, \\\\
&\\; p_{t} \\geq \\tau q_{t}\\,, \\quad \\forall\\, t \\in \\mathcal{S}\\,, \\qquad p_{t} = 0\\,, \\quad \\forall\\, t \\notin \\mathcal{S}\\,, \\\\
&\\; \\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{p} = \\boldsymbol{B}_{\\mathrm{eq}}\\,, \\quad \\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{p} \\leq \\boldsymbol{B}_{\\mathrm{ineq}}\\,.
\\end{align}
```

The prior is a feasible point when every view is slack at it, and then ``\\tau = 1``. The views exclude each other when the programme is infeasible. When ``\\tau \\leq 0``, a posterior meets the views only with zero probability on a scenario of ``\\mathcal{S}``. The posterior of entropy pooling is positive wherever the prior is positive, so its dual then has no finite solution. This is the joint form of the edge that [`ep_assert_attainable_rows`](@ref) refuses for one row. [meucci2008](@citet) states the dual. The feasibility programme is the library's own.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:T])
  - $(math_dict[:eq])
  - $(math_dict[:ineq])
  - ``\\tau``: Margin of the posterior above the prior, as a fraction of the prior.
  - ``\\mathcal{S}``: Support of the prior, the scenarios with ``q_{t} > 0``.

# JuMP formulation

## Variables

  - `t`: ``\\tau``, the margin.
  - `p`: ``\\boldsymbol{p}``, the posterior probabilities, with a lower bound of zero. The variables off the support are fixed at zero.

## Constraints

  - `csum`: ``\\sum_{t=1}^{T} p_{t} - 1 = 0``.
  - `cmargin`: ``\\tau \\boldsymbol{q} - \\boldsymbol{p} \\leq 0``. Off the support, ``q_{t} = 0``, and the row is the bound ``p_{t} \\geq 0``.
  - `ceq`: ``\\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{eq}} = 0``, when `epc` holds an `:eq` block.
  - `cineq`: ``\\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{ineq}} \\leq 0``, when `epc` holds an `:ineq` block.
  - The rows of each exact tail view constraint, which [`ep_diagnose_tail_view!`](@ref) adds.

## Objective

  - Maximise `t`.

The programme leaves out three kinds of row. A `:feq` row fixes a moment of an earlier stage, and the solve relaxes it with a penalised slack, so it cannot make the solve infeasible. A `:cvar_eq` row is a trial of the root search of [`ep_cvar_views_solve!`](@ref), not a view. A tail view constraint that is a restriction of its view can be infeasible where the view is not, so [`ep_diagnose_tail_view!`](@ref) adds only the exact ones.

# Algorithm

 1. Return `nothing` when `slv` is `nothing`, or when `epc` holds no `:eq` and no `:ineq` block and `tvs` is empty.
 2. Build the programme of the `JuMP formulation` section, and solve it with [`optimise_JuMP_model!`](@ref).
 3. When no solver solves it and one solver reports it infeasible, raise a `DomainError`.
 4. When no solver solves it for another reason, raise an `ErrorException`.
 5. When the margin is not positive, raise a `DomainError`.

A simplex solver returns a vertex of the programme, so at the edge its margin is zero, and the check refuses the edge exactly. An interior-point solver meets the edge only to its tolerance. Its margin there can be a little above zero, and then the check passes, and the solve meets the edge to the same tolerance.

# Arguments

  - `slv`: Solvers of the programme, or `nothing` to skip the check. The estimator gives its `diagnose` field.
  - `w`: Prior probabilities of the solve that follows, one for each observation.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `tvs`: Tail view constraints of the solve that follows.
  - `pe`: Entropy pooling prior estimator, read only for the names of its view fields in an error message.
  - `fields`: Names of the view fields whose rows `epc` and `tvs` hold. The error message names those of them that are not `nothing`.

# Validation

  - The programme must be feasible, else a `DomainError` names the view fields.
  - A solver must solve the programme, else an `ErrorException` names the view fields.
  - The margin must be positive, else a `DomainError` names the view fields and the margin.

# Returns

  - `nothing`.

# Related

  - [`ep_assert_attainable_rows`](@ref)
  - [`ep_diagnose_tail_view!`](@ref)
  - [`entropy_pooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:meucci2008]) Appendix A.3, Equations 90 to 93.
"""
function ep_diagnose_views(::Nothing, ::VecNum, ::AbstractDict, ::VecEPTV,
                           ::AbstractPriorEstimator, ::Tuple)
    return nothing
end
function ep_diagnose_views(slv::Slv_VecSlv, w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                           pe::AbstractPriorEstimator, fields::Tuple)
    if !haskey(epc, :eq) && !haskey(epc, :ineq) && isempty(tvs)
        return nothing
    end
    T = length(w)
    # `collect` reads the values of a weights vector, which StatsBase cannot index when they
    # are a range.
    q = collect(w)
    Z = findall(iszero, q)
    model = JuMP.Model()
    JuMP.@variables(model, begin
                        t
                        p[1:T] >= 0
                    end)
    JuMP.fix.(p[Z], zero(eltype(w)); force = true)
    JuMP.@constraint(model, csum, sum(p) - one(eltype(w)) == 0)
    JuMP.@constraint(model, cmargin, t * q - p <= 0)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        JuMP.@constraint(model, ceq, A * p ⊖ B == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        JuMP.@constraint(model, cineq, A * p ⊖ B <= 0)
    end
    for tv in tvs
        ep_diagnose_tail_view!(model, p, tv)
    end
    JuMP.@objective(model, Max, t)
    res = optimise_JuMP_model!(model, slv)
    # The names are read only for a message, so a fit that passes builds no string.
    names = () -> join(("`$(f)`" for f in fields if !isnothing(getfield(pe, f))), ", ",
                       " and ")
    infeasible = (JuMP.INFEASIBLE, JuMP.LOCALLY_INFEASIBLE, JuMP.INFEASIBLE_OR_UNBOUNDED)
    @argcheck(res.success ||
              !any(tr -> get(tr, :termination_status, nothing) in infeasible,
                   values(res.trials)),
              DomainError(names(),
                          "No reweighting of the scenarios at a positive prior probability meets the views of $(names()) together. Each view row can be attainable alone while the rows exclude each other, so entropy pooling has no posterior. Relax or remove a view that conflicts with the others."))
    @argcheck(res.success,
              ErrorException("The solvers of `diagnose` did not solve the feasibility programme of the views of $(names()), and no solver reported it infeasible. Use a solver that takes the cones of the exact tail views, or other solver settings."))
    tau = JuMP.value(t)
    @argcheck(tau > zero(tau),
              DomainError(tau,
                          "The views of $(names()) hold together only when the posterior puts zero probability on a scenario of positive prior probability: the largest margin of the posterior above the prior is $(tau). The posterior of entropy pooling is positive wherever the prior is positive, so the solve has no finite dual solution. Move a target away from the edge of the views, or use a prior whose scenarios spread wider."))
    return nothing
end
"""
    ep_jump_entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
                            opt::JuMPEntropyPooling)

Build the entropy pooling model of one set of tail view constraints, and solve it once.

`ep_jump_entropy_pooling` is the body of the [`JuMPEntropyPooling`](@ref) method of [`entropy_pooling`](@ref). That method calls it once for each set of tail view constraints, and calls it again after it reads the sequential constraints again. The two methods of this function differ only in the form of the divergence, and write every view in the same way.

# Algorithm

 1. Read `sc1`, `sc2`, `so` and `slv` from `opt`, and make an empty `JuMP.Model`.
 2. Create the posterior variable `x` and the divergence variable `t`. Under [`ExpEntropyPooling`](@ref), `x` also has a non-negativity bound. The cone of step 4 implies this bound in both branches.
 3. Add the row that holds the sum of `x` at one.
 4. Add the relative entropy cone that bounds `t` below by the divergence. Under [`ExpEntropyPooling`](@ref) the cone reads the prior `w` as its reference, and under [`LogEntropyPooling`](@ref) it reads a unit reference instead.
 5. Register `obj_expr` as `so * t`.
 6. Add every view with [`ep_jump_views!`](@ref), which mutates `obj_expr` when it relaxes a fixed equality row.
 7. Set the objective to `obj_expr` under [`ExpEntropyPooling`](@ref). Under [`LogEntropyPooling`](@ref), set it to `obj_expr` less `so` times the inner product of `x` with the logarithms of the prior probabilities.
 8. Solve with `slv`, and raise when no solver configuration succeeds.
 9. Clamp the value of `x` at zero. The solver meets the non-negativity of step 4 only to its feasibility tolerance, and a mixed-integer solve can return an entry a small amount below zero. Divide the value by its sum, and return it through [`scenario_probabilities`](@ref), in the kind of `w`.

# JuMP formulation

## Variables

  - `t`: ``\\eta``, the epigraph variable of the divergence, created here.
  - `x`: ``\\boldsymbol{p}``, the posterior probabilities, created here. Under [`ExpEntropyPooling`](@ref), it has the bound ``\\boldsymbol{p} \\geq \\boldsymbol{0}``.

## Expressions

  - `obj_expr`: ``s_{o} \\eta``. [`ep_jump_views!`](@ref) adds ``s_{o} s_{c2} \\eta_{c}`` to it when `epc` holds a `:feq` block.

## Constraints

The method registers the two rows below with no name.

  - ``s_{c1} \\left(\\sum_{t=1}^{T} p_{t} - 1\\right) = 0``
  - ``\\left(s_{c1} \\eta,\\; s_{c1} \\boldsymbol{q},\\; s_{c1} \\boldsymbol{p}\\right) \\in \\mathcal{K}_{\\mathrm{relent}}``, which states ``\\eta \\geq \\sum_{t=1}^{T} p_{t} \\ln(p_{t} / q_{t})``. Under [`LogEntropyPooling`](@ref) the reference is ``s_{c1} \\boldsymbol{1}`` in place of ``s_{c1} \\boldsymbol{q}``, so the row states ``\\eta \\geq \\sum_{t=1}^{T} p_{t} \\ln(p_{t})``.

[`ep_jump_views!`](@ref) registers every other row of the model, and its own `# JuMP formulation` names them.

## Objective

  - `Min` of `obj_expr` under [`ExpEntropyPooling`](@ref).
  - `Min` of `obj_expr` less ``s_{o} \\boldsymbol{p}^{\\intercal} \\ln(\\boldsymbol{q})`` under [`LogEntropyPooling`](@ref). The subtraction changes the ``\\sum_{t} p_{t} \\ln(p_{t})`` of the cone into the divergence, so both branches minimise the same value.

## Relaxation

$(val_dict[:relax])

  - The bound is on `obj_expr`, and it lies above the divergence of the exactly constrained problem. [`ep_jump_views!`](@ref) relaxes the `:feq` rows with a penalised slack, and the penalty is a term of `obj_expr`.
  - The bound is tight when that slack is zero. This holds when the posterior can meet the `:feq` rows and the views together, and ``s_{c2}`` is large enough to make the slack zero.

Where:

  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:T])
  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_sc2])
  - $(math_dict[:ep_so])
  - ``\\eta``: Epigraph variable that bounds the divergence from above. Its model key is `t`.
  - ``\\eta_{c}``: Norm of the slack of the fixed equality rows, which [`ep_jump_views!`](@ref) creates under the model key `tc`.
  - ``\\mathcal{K}_{\\mathrm{relent}}``: Relative entropy cone, `JuMP.MOI.RelativeEntropyCone(2T + 1)`.

# Arguments

  - `w`: Prior probabilities, one for each observation.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `tvs`: Tail view constraints.
  - `opt`: Entropy pooling optimiser.

# Validation

  - At least one configuration of `opt.slv` must solve the model. Otherwise the method raises an `ErrorException`.

# Returns

  - `pw::ScenarioProbabilities`: Posterior probabilities that meet the constraints, in the kind of `w`.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_jump_views!`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008]) Section 4, Equations 26 and 27.
  - $(ref_dict[:EPTail]) Section 2.1, Equation 1.
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
    return scenario_probabilities(w, LinearAlgebra.normalize(max.(JuMP.value.(x), 0), 1))
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
    return scenario_probabilities(w, LinearAlgebra.normalize(max.(JuMP.value.(x), 0), 1))
end
"""
    ep_jump_views!(model::JuMP.Model, x, obj_expr, epc::AbstractDict, tvs::VecEPTV,
                   sc1::Number, sc2::Number, so::Number)

Add every view constraint of an entropy pooling problem to a JuMP model.

`ep_jump_views!` is the body that the two [`JuMPEntropyPooling`](@ref) formulations share. They differ only in the form of the divergence, and write every view in the same way. It adds the linear rows of `epc`, and relaxes the fixed equality rows with a slack whose norm a norm one cone bounds. It gives each tail view to [`add_ep_tail_view!`](@ref).

# Algorithm

 1. Add the row `ceq` when `epc` holds an `:eq` block.
 2. Add the row `cineq` when `epc` holds an `:ineq` block.
 3. Add the row `ccvareq` when `epc` holds a `:cvar_eq` block.
 4. When `epc` holds a `:feq` block, create the slack `c` and its norm `tc`. Add the row `cfeq`, which lets the block miss its target by `c`, and bound the norm of `c` by `tc`. Add the penalty `so * sc2 * tc` to `obj_expr`.
 5. Give each tail view of `tvs` to [`add_ep_tail_view!`](@ref) in turn.

# JuMP formulation

## Variables

  - `x`: ``\\boldsymbol{p}``, the posterior probabilities, read from the model.
  - `tc`: ``\\eta_{c}``, the norm of the slack of the fixed equality rows, which the function creates when `epc` holds a `:feq` block.
  - `c`: ``\\boldsymbol{c}``, the slack of the fixed equality rows, with one entry for each row of the `:feq` block. The function creates it with `tc`.

## Constraints

  - `ceq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{eq}}\\right) = 0``
  - `cineq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{ineq}}\\right) \\leq 0``
  - `ccvareq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{cvar}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{cvar}}\\right) = 0``
  - `cfeq`: ``s_{c1} \\left(\\mathbf{A}_{\\mathrm{feq}} \\boldsymbol{p} - \\boldsymbol{B}_{\\mathrm{feq}} - \\boldsymbol{c}\\right) = 0``
  - Registered with no name, together with `cfeq`: ``\\left(s_{c1} \\eta_{c},\\; s_{c1} \\boldsymbol{c}\\right) \\in \\mathcal{K}_{1}``, which states ``\\eta_{c} \\geq \\lVert \\boldsymbol{c} \\rVert_{1}``.

The function registers each of the four named rows only when `epc` holds the block of that row. [`add_ep_tail_view!`](@ref) registers every row that a tail view needs. Its methods are in `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_a.jl` and `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_b.jl`.

## Relaxation

$(val_dict[:relax])

  - The bound is on the objective, and it lies above the objective of the problem that enforces its `:feq` rows exactly. The rows `cfeq` let the `:feq` block miss its right-hand side by ``\\boldsymbol{c}``, and the penalty ``s_{o} s_{c2} \\eta_{c}`` holds ``\\boldsymbol{c}`` near zero.
  - The bound is tight when ``\\boldsymbol{c}`` is zero. This holds when the posterior can meet the `:feq` rows and the views together, and ``s_{c2}`` is large enough to make the slack zero. A `:feq` row holds a moment at its prior value, and a view can ask to move that moment. Then the row misses its target.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:eq])
  - $(math_dict[:ineq])
  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_sc2])
  - $(math_dict[:ep_so])
  - ``\\mathrm{cvar}``, ``\\mathrm{feq}``: Subscripts of the `:cvar_eq` and `:feq` blocks of `epc`.
  - ``\\boldsymbol{c}``: Slack of the fixed equality rows, with one entry for each row of the `:feq` block.
  - ``\\eta_{c}``: Bound on the ``L_{1}`` norm of ``\\boldsymbol{c}``.
  - ``\\mathcal{K}_{1}``: Norm one cone, `JuMP.MOI.NormOneCone(N + 1)`.

# Arguments

  - `model`: Entropy pooling JuMP model.
  - `x`: Vector of posterior probability variables.
  - `obj_expr`: Objective expression, which the function mutates when it relaxes a fixed equality row.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `tvs`: Tail view constraints.
  - `sc1`: Constraint scaling factor.
  - `sc2`: Penalty weight of the slack of the fixed equality rows.
  - `so`: Objective scaling factor.

# Returns

  - `nothing`: The function mutates `model` and `obj_expr` in place.

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
