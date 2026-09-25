"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the optimisers that solve an entropy pooling problem.

A subtype names the numerical route to the posterior probabilities: which package drives the solve, and whether it solves the dual or the primal.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
abstract type AbstractEntropyPoolingOptimiser <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that decide how the views of an entropy pooling problem reach the optimiser.

A subtype states whether every view is enforced in one optimisation or in stages, from the lower moments to the higher ones, and which probabilities each stage starts from.

# Related

  - [`H0_EntropyPooling`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`StagedEP`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:meucci2008])
  - $(ref_dict[:vorobets2021])
"""
abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Enforces every view in a single entropy pooling optimisation.

This is the original formulation. It solves once, so it is the cheapest of the three algorithms, and it pins nothing: a higher moment view is free to move a lower moment of the same asset.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Enforces the views in stages, and starts every stage from the prior probabilities.

Each stage carries the constraints of the stages before it, so the posterior is the projection of the *prior* onto the whole view set. This is the first of the two sequential heuristics.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`StagedEP`](@ref)

# References

  - $(ref_dict[:vorobets2021])
"""
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Enforces the views in stages, and starts every stage from the previous stage's probabilities.

Each stage carries the constraints of the stages before it, so the view set is the same as [`H1_EntropyPooling`](@ref)'s. What differs is the reference distribution: each stage projects the stage before it rather than the prior. This is the second of the two sequential heuristics.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`StagedEP`](@ref)

# References

  - $(ref_dict[:vorobets2021])
"""
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
    const StagedEP = Union{<:H1_EntropyPooling, <:H2_EntropyPooling}

Alias for a union of staged entropy pooling algorithm types.

# Related

  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
"""
const StagedEP = Union{<:H1_EntropyPooling, <:H2_EntropyPooling}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that decide how an entropy pooling optimiser evaluates its objective.

Every subtype minimises the same Kullback-Leibler divergence of the posterior probabilities from the prior ones. They differ only in the arithmetic that evaluates it, so they answer the same problem with the same posterior.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
abstract type AbstractEntropyPoolingOptAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Evaluates the entropy pooling objective in log space.

It carries the logarithms of the probabilities throughout, so it never exponentiates an intermediate quantity. Prefer it when a prior probability is small enough that the exponential form loses precision. It minimises the same Kullback-Leibler divergence as [`ExpEntropyPooling`](@ref) and reaches the same posterior.

# Related

  - [`AbstractEntropyPoolingOptAlgorithm`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
struct LogEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Evaluates the entropy pooling objective through the exponential of the dual variables.

It recovers each posterior probability from the prior one and the dual variables directly, without carrying logarithms. It minimises the same Kullback-Leibler divergence as [`LogEntropyPooling`](@ref) and reaches the same posterior.

# Related

  - [`AbstractEntropyPoolingOptAlgorithm`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
struct ExpEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Root-finds the value at risk level that meets a single conditional value-at-risk view.

The recursive route of [`MeucciEntropyPoolingPrior`](@ref) writes no CVaR constraint. It hunts the level instead: for each candidate it rebuilds the positive-part rows, re-solves the whole entropy pooling problem, and reads the tail mass off the posterior. This type holds the arguments of the [`Roots.jl`](https://github.com/JuliaMath/Roots.jl) call that drives the hunt. Where a group carries more than one view, [`OptimEntropyPooling`](@ref) takes over through the `dm_opt` field.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalValueatRiskEntropyPooling(;
        args::Tuple = (Roots.Brent(),),
        kwargs::NamedTuple = (;)
    ) -> ConditionalValueatRiskEntropyPooling

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> ConditionalValueatRiskEntropyPooling()
ConditionalValueatRiskEntropyPooling
    args ┼ Tuple{Roots.Brent}: (Roots.Brent(),)
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`Roots.jl`](https://github.com/JuliaMath/Roots.jl)

# References

  - $(ref_dict[:meucciardiakeel2011])
"""
@concrete struct ConditionalValueatRiskEntropyPooling <: AbstractEntropyPoolingOptimiser
    """
    $(field_dict[:optargs])
    """
    args
    """
    $(field_dict[:optkwargs])
    """
    kwargs
    function ConditionalValueatRiskEntropyPooling(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function ConditionalValueatRiskEntropyPooling(; args::Tuple = (Roots.Brent(),),
                                              kwargs::NamedTuple = (;))::ConditionalValueatRiskEntropyPooling
    return ConditionalValueatRiskEntropyPooling(args, kwargs)
end
"""
$(DocStringExtensions.TYPEDEF)

Solves the dual of the entropy pooling problem with Optim.jl.

The dual has one variable per constraint rather than one per observation, and it is box constrained, so it is the cheaper route wherever the views reduce to rows of the constraint set. It has no room for an auxiliary variable, so it cannot express a tail view: use [`JuMPEntropyPooling`](@ref) there. It drives [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) and takes either optimisation algorithm.

This route does not raise on an infeasible view set. [`entropy_pooling`](@ref) states the mechanism and the three signs that name that answer.

!!! tip "The default stopping rule leaves the dual short of its optimum"

    An empty `args` runs `Optim` under its own default `Options`, which stops on the step in `x`, on the change in the objective, or on the gradient, whichever comes first. The first two arrive before the dual reaches stationarity, so the posterior meets its views closely rather than exactly. Over the twelve single-view CVaR cases of `test/test_12a_entropy_pooling.jl` the default holds a posterior conditional value at risk to about `1e-7` of its target, and an `Optim.Options` with `g_abstol = 1e-12` and `outer_iterations = 50` holds it to about `1e-11`. The default is the right trade for a view read once, and the tighter rule costs about 9% more time over that file.

    Two readers magnify the shortfall. A staged chain refits the wrapped estimator between stages, so stage two projects a reference that already carries the error of stage one. The recursive CVaR route of [`MeucciEntropyPoolingPrior`](@ref) re-solves the whole problem at each candidate value at risk, so a solve that stops early moves the root the search returns. That route reports success on the outer variable alone, so the shortfall it carries is not bounded by the `1e-7` above: the same case has been measured at `3.1e-4` after other solves ran in the same process, and its own warning states that mode. Where the answer has to be repeatable, pass a tighter `Optim.Options` in `args`:

    ```julia
    using PortfolioOptimisers: Optim

    OptimEntropyPooling(;
                        args = (Optim.Fminbox(; mu0 = 1e-5),
                                Optim.Options(; x_abstol = 1e-12, f_reltol = 1e-14,
                                              g_abstol = 1e-12, outer_x_abstol = 1e-12,
                                              iterations = 10_000, outer_iterations = 50)))
    ```

    `Optim` reaches this package as an internal binding, so the block names it through `PortfolioOptimisers`. A non-empty `args` replaces the `Optim.Fminbox(; mu0 = 1e-5)` that [`entropy_pooling`](@ref) supplies, so carry it yourself.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimEntropyPooling(;
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        sc1::Number = 1,
        sc2::Number = 1e3,
        alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling(),
        err::Option{<:NormError} = nothing
    ) -> OptimEntropyPooling

Keywords correspond to the struct's fields.

## Validation

  - `sc1 >= 0`.
  - `sc2 >= 0`.

# Examples

```jldoctest
julia> OptimEntropyPooling()
OptimEntropyPooling
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
     sc1 ┼ Int64: 1
     sc2 ┼ Float64: 1000.0
     alg ┼ ExpEntropyPooling()
     err ┴ nothing
```

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)
  - [`NormError`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
@concrete struct OptimEntropyPooling <: AbstractEntropyPoolingOptimiser
    """
    $(field_dict[:optargs])
    """
    args
    """
    $(field_dict[:optkwargs])
    """
    kwargs
    """
    $(field_dict[:sc1]) It multiplies the dual objective and its gradient alike, so it leaves the minimiser unchanged and tightens the residual `Optim` stops at.
    """
    sc1
    """
    `sc2`: Half-width of the box the dual variables of the fixed equality rows are held in. It is the dual of a penalty of weight `sc2` on the norm of the slack the primal carries, so a larger value holds those rows tighter.
    """
    sc2
    """
    $(field_dict[:epoptalg])
    """
    alg
    """
    $(field_dict[:err]) Only used when there are multiple cvar views. If `nothing`, the L2 norm is used.
    """
    err
    function OptimEntropyPooling(args::Tuple, kwargs::NamedTuple, sc1::Number, sc2::Number,
                                 alg::AbstractEntropyPoolingOptAlgorithm,
                                 err::Option{<:NormError})
        @argcheck(sc1 >= zero(sc1), DomainError(sc1, "sc1 must be >= 0"))
        @argcheck(sc2 >= zero(sc2), DomainError(sc2, "sc2 must be >= 0"))
        return new{typeof(args), typeof(kwargs), typeof(sc1), typeof(sc2), typeof(alg),
                   typeof(err)}(args, kwargs, sc1, sc2, alg, err)
    end
end
function OptimEntropyPooling(; args::Tuple = (), kwargs::NamedTuple = (;), sc1::Number = 1,
                             sc2::Number = 1e3,
                             alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling(),
                             err::Option{<:NormError} = nothing)::OptimEntropyPooling
    return OptimEntropyPooling(args, kwargs, sc1, sc2, alg, err)
end
"""
$(DocStringExtensions.TYPEDEF)

Solves the primal of the entropy pooling problem with JuMP.jl.

The primal carries one variable per observation and writes the divergence as a relative entropy cone. It is the only route that expresses a tail view, because a tail view needs auxiliary variables that the dual has no room for. It drives [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) and takes either optimisation algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPEntropyPooling(;
        slv::Slv_VecSlv,
        sc1::Number = 1,
        sc2::Number = 1e5,
        so::Number = 1,
        alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling()
    ) -> JuMPEntropyPooling

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a vector, `!isempty(slv)`.
  - `sc1 >= 0`
  - `sc2 >= 0`
  - `so >= 0`

# Examples

```jldoctest
julia> JuMPEntropyPooling(; slv = Solver(; name = :fake_solver, solver = :MySolver))
JuMPEntropyPooling
  slv ┼ Solver
      │          name ┼ Symbol: :fake_solver
      │        solver ┼ Symbol: :MySolver
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
  sc1 ┼ Int64: 1
  sc2 ┼ Float64: 100000.0
   so ┼ Int64: 1
  alg ┴ ExpEntropyPooling()
```

# Related

  - [`AbstractEntropyPoolingOptimiser`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`ConditionalValueatRiskEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl)

# References

  - $(ref_dict[:meucci2008])
"""
@concrete struct JuMPEntropyPooling <: AbstractEntropyPoolingOptimiser
    """
    $(field_dict[:slv])
    """
    slv
    """
    `sc1`: Scale of every row the model registers. It multiplies both sides of a row, so a positive value leaves the feasible set unchanged.
    """
    sc1
    """
    $(field_dict[:sc2]) It weights the norm of the fixed equality slack in the objective, so a larger value holds those rows tighter.
    """
    sc2
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:epoptalg])
    """
    alg
    function JuMPEntropyPooling(slv::Slv_VecSlv, sc1::Number, sc2::Number, so::Number,
                                alg::AbstractEntropyPoolingOptAlgorithm)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc1 >= zero(sc1), DomainError(sc1, "sc1 must be >= 0"))
        @argcheck(sc2 >= zero(sc2), DomainError(sc2, "sc2 must be >= 0"))
        @argcheck(so >= zero(so), DomainError(so, "so must be >= 0"))
        return new{typeof(slv), typeof(sc1), typeof(sc2), typeof(so), typeof(alg)}(slv, sc1,
                                                                                   sc2, so,
                                                                                   alg)
    end
end
function JuMPEntropyPooling(; slv::Slv_VecSlv, sc1::Number = 1, sc2::Number = 1e5,
                            so::Number = 1,
                            alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())::JuMPEntropyPooling
    return JuMPEntropyPooling(slv, sc1, sc2, so, alg)
end
"""
    const NonCVaREP = Union{<:OptimEntropyPooling, <:JuMPEntropyPooling}

Alias for a union of non-CVaR entropy pooling algorithm types.

# Related

  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
const NonCVaREP = Union{<:OptimEntropyPooling, <:JuMPEntropyPooling}

export LogEntropyPooling, ExpEntropyPooling, H0_EntropyPooling, H1_EntropyPooling,
       H2_EntropyPooling, JuMPEntropyPooling, OptimEntropyPooling,
       ConditionalValueatRiskEntropyPooling
