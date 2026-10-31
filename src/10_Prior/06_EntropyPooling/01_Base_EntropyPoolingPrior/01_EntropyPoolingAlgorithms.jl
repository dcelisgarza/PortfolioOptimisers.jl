"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the optimisers that solve an entropy pooling problem.

A subtype selects the package that finds the posterior probabilities, and whether that package solves the dual problem or the primal problem.

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

Abstract supertype for the algorithms that give the views of an entropy pooling problem to the optimiser.

A subtype states whether the optimiser enforces every view in one optimisation or in stages, from the lower moments to the higher ones. It also states the probabilities from which each stage starts.

# Related

  - [`H0_EntropyPooling`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`StagedEP`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:meucci2008]) Section 4, Equations 26 and 27.
  - $(ref_dict[:vorobets2021]) Section 3, Algorithms 1 and 2.
"""
abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Enforces every view in a single entropy pooling optimisation.

This is the original formulation. It solves one problem, so it is the cheapest of the three algorithms. It fixes no moment, so a view on a higher moment can move a lower moment of the same asset. A variance view reads the prior mean, as [meucci2008](@cite) does. The library gives the name `H0`, and [vorobets2021](@citet) defines only H1 and H2.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008]) Section 4, Equations 26 and 27.
  - $(ref_dict[:vorobets2021]) Section 3.
"""
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Enforces the views in stages, and starts every stage from the prior probabilities.

Each stage keeps the constraints of the stages before it, so the posterior is the projection of the prior onto the set that all the views define. This is the first of the two sequential heuristics. [vorobets2021](@citet) takes one stage for each class of parameter: means, variances, skewness and kurtosis, correlations, and then the rest. The library takes three stages: the mean and value at risk views, the variance and covariance views, and then the correlation, skewness and kurtosis views.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`StagedEP`](@ref)

# References

  - $(ref_dict[:vorobets2021]) Section 3, Algorithm 1.
"""
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Enforces the views in stages, and starts every stage from the previous stage's probabilities.

Each stage keeps the constraints of the stages before it, so the set of views is the same as the set of [`H1_EntropyPooling`](@ref). The difference is the reference distribution. Each stage projects the probabilities of the stage before it, not the prior probabilities. This is the second of the two sequential heuristics. The library puts the views in the same stages as [`H1_EntropyPooling`](@ref).

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`StagedEP`](@ref)

# References

  - $(ref_dict[:vorobets2021]) Section 3, Algorithm 2.
"""
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
"""
    const StagedEP = Union{H1_EntropyPooling, H2_EntropyPooling}

Alias for the union of the two entropy pooling algorithms that enforce the views in stages.

# Related

  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
"""
const StagedEP = Union{H1_EntropyPooling, H2_EntropyPooling}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that set how an entropy pooling optimiser evaluates its objective.

Every subtype minimises the same Kullback-Leibler divergence of the posterior probabilities from the prior probabilities. The subtypes differ only in the arithmetic of the evaluation, so they return the same posterior for the same problem.

# Related

  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008]) Section 4, Equations 26 and 27.
"""
abstract type AbstractEntropyPoolingOptAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Evaluates the entropy pooling objective in log space.

With [`OptimEntropyPooling`](@ref), it adds the logarithm of each prior probability to the exponent, and then takes one exponential of the sum. It does not multiply a prior probability by a separate exponential, which can overflow when the exponent is large and the prior probability is small. With [`JuMPEntropyPooling`](@ref), the relative entropy cone reads a unit reference, and the objective subtracts the inner product of the posterior probabilities with the logarithms of the prior probabilities. Use it when a prior probability is so small that the exponential form can overflow or lose precision. It minimises the same Kullback-Leibler divergence as [`ExpEntropyPooling`](@ref) and returns the same posterior.

# Related

  - [`AbstractEntropyPoolingOptAlgorithm`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008]) Appendix A.3, Equations 90 and 93.
"""
struct LogEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Evaluates the entropy pooling objective through the exponential of the dual variables.

With [`OptimEntropyPooling`](@ref), it calculates each posterior probability as the product of the prior probability and the exponential of the dual term. With [`JuMPEntropyPooling`](@ref), the relative entropy cone reads the prior probabilities as its reference. It minimises the same Kullback-Leibler divergence as [`LogEntropyPooling`](@ref) and returns the same posterior.

# Related

  - [`AbstractEntropyPoolingOptAlgorithm`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008]) Appendix A.3, Equations 90 and 93.
"""
struct ExpEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Finds with a root finder the value at risk level that meets one conditional value-at-risk view.

With this optimiser, [`MeucciEntropyPoolingPrior`](@ref) writes no CVaR constraint. It searches for the level instead. For each candidate level, it makes the positive-part rows again, solves the full entropy pooling problem again, and reads the tail mass of the posterior. This type holds the arguments of the [`Roots.jl`](https://juliamath.github.io/Roots.jl/stable/) call that does the search. A group of more than one CVaR view uses the [`OptimEntropyPooling`](@ref) of the `dm_opt` field instead.

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
  - [`Roots.jl`](https://juliamath.github.io/Roots.jl/stable/)

# References

  - $(ref_dict[:meucciardiakeel2011]) Pages 39–49. $(ref_dict[:no_original_source])
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

The dual problem has one variable for each constraint, not one for each observation, and it has box constraints. So it costs less than the primal problem when every view is a row of the constraint set. The dual problem cannot hold an auxiliary variable, so it cannot express a tail view. Use [`JuMPEntropyPooling`](@ref) for a tail view.

This optimiser uses [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/) and accepts both optimisation algorithms. [meucci2008](@citet) maximises the dual function. The library minimises the negative of the dual function, and adds a box on the dual variables of the fixed equality rows, which [meucci2008](@citet) does not have.

This optimiser raises no error for an infeasible view set. [`entropy_pooling`](@ref) states the cause, and the three signs that show an infeasible view set.

!!! tip "The default stopping rule stops before the optimum of the dual problem"

    With an empty `args`, `Optim` uses its default `Options`. These stop the solve at the first of three tests: the step in `x`, the change in the objective, or the gradient. The first two tests stop the solve before the dual problem reaches stationarity, so the posterior meets its views approximately, not exactly. In the twelve CVaR cases with one view of `test/test_12a_entropy_pooling.jl`, the default puts the posterior conditional value at risk within about `1e-7` of its target. An `Optim.Options` with `g_abstol = 1e-12` and `outer_iterations = 50` puts it within about `1e-11`, and increases the time of that file by about 9%. The default is enough for a view that a caller reads once.

    Two callers make the error larger. A staged algorithm fits the wrapped estimator again between stages, so stage two projects a reference that already has the error of stage one. The recursive CVaR search of [`MeucciEntropyPoolingPrior`](@ref) solves the full problem again at each candidate value at risk. So a solve that stops early moves the root that the search returns.

    The recursive CVaR search reports success on the outer variable only, so the `1e-7` above is not a bound on its error. One measurement of the same case gave `3.1e-4` after other solves ran in the same process, and the warning of the search states this case. If the result must be repeatable, pass a tighter `Optim.Options` in `args`:

    ```julia
    using PortfolioOptimisers: Optim

    OptimEntropyPooling(;
                        args = (Optim.Fminbox(; mu0 = 1e-5),
                                Optim.Options(; x_abstol = 1e-12, f_reltol = 1e-14,
                                              g_abstol = 1e-12, outer_x_abstol = 1e-12,
                                              iterations = 10_000, outer_iterations = 50)))
    ```

    This package does not export `Optim`, so the block gets it from `PortfolioOptimisers`. A non-empty `args` replaces the `Optim.Fminbox(; mu0 = 1e-5)` that [`entropy_pooling`](@ref) gives, so put an `Optim.Fminbox` in `args` too.

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
  - [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/)
  - [`NormError`](@ref)

# References

  - $(ref_dict[:meucci2008]) Appendix A.3, Equations 91 to 93.
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
    $(field_dict[:sc1]) It multiplies the dual objective and its gradient, so it does not change the minimiser. A larger value makes the residual smaller at the point where `Optim` stops.
    """
    sc1
    """
    `sc2`: Half-width of the box that holds the dual variables of the fixed equality rows. The box is the dual of a penalty of weight `sc2` on the norm of the slack of the primal problem, so a larger value makes that slack smaller.
    """
    sc2
    """
    $(field_dict[:epoptalg])
    """
    alg
    """
    $(field_dict[:err]) The recursive CVaR search of [`MeucciEntropyPoolingPrior`](@ref) reads it only for a group of more than one CVaR view. If `nothing`, the search uses the L2 norm.
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

The primal problem has one variable for each observation, and it writes the divergence as a relative entropy cone. It is the only optimiser that can express a tail view, because a tail view needs auxiliary variables that the dual problem cannot hold. This optimiser uses [`JuMP.jl`](https://jump.dev/JuMP.jl/stable/) and accepts both optimisation algorithms. [meucci2008](@citet) states this primal problem and solves its dual. The library solves the primal problem with a conic solver, and adds the slack that relaxes the fixed equality rows.

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
  - [`JuMP.jl`](https://jump.dev/JuMP.jl/stable/)

# References

  - $(ref_dict[:meucci2008]) Section 4, Equations 26 and 27.
"""
@concrete struct JuMPEntropyPooling <: AbstractEntropyPoolingOptimiser
    """
    $(field_dict[:slv])
    """
    slv
    """
    `sc1`: Scale of every row that the model registers. It multiplies both sides of a row, so a positive value does not change the feasible set.
    """
    sc1
    """
    $(field_dict[:sc2]) A larger value makes that slack smaller.
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
    const NonCVaREP = Union{OptimEntropyPooling, JuMPEntropyPooling}

Alias for the union of the two entropy pooling optimisers that solve the problem with no root finder.

# Related

  - [`OptimEntropyPooling`](@ref)
  - [`JuMPEntropyPooling`](@ref)
"""
const NonCVaREP = Union{OptimEntropyPooling, JuMPEntropyPooling}

export LogEntropyPooling, ExpEntropyPooling, H0_EntropyPooling, H1_EntropyPooling,
       H2_EntropyPooling, JuMPEntropyPooling, OptimEntropyPooling,
       ConditionalValueatRiskEntropyPooling
