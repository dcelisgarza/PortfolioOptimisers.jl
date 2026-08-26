"""
$(DocStringExtensions.TYPEDEF)

Carries a parsed correlation or covariance view together with the asset pairs it names.

It extends [`ParsingResult`](@ref) with an `ij` field, which holds one index pair per term of the view, so a downstream routine can place the view in the covariance matrix without parsing the equation again.

# Fields

$(DocStringExtensions.FIELDS)

# Details

  - Produced by correlation view parsing routines, typically when the constraint involves asset pairs (e.g., `"(A, B) == 0.5"`).
  - The `ij` field enables downstream routines to map parsed correlation views to the appropriate entries in the correlation matrix.
  - A view over a pair of groups spans one asset pair per element of its `ij` entry, and emits one constraint row per pair. Its `rhs` is therefore a vector of the same length, one right-hand side per row. A view over a single asset pair keeps a scalar `rhs`.
  - Used internally for entropy pooling, Black-Litterman, and other advanced portfolio models that support correlation views.

# Related

  - [`AbstractParsingResult`](@ref)
  - [`ParsingResult`](@ref)
  - [`replace_prior_views`](@ref)
"""
@concrete struct RhoParsingResult <: AbstractParsingResult
    """
    $(field_dict[:vars])
    """
    vars
    """
    $(field_dict[:coef_c])
    """
    coef
    """
    $(field_dict[:op])
    """
    op
    """
    $(field_dict[:rhs_rho])
    """
    rhs
    """
    $(field_dict[:eqn])
    """
    eqn
    """
    $(field_dict[:ij])
    """
    ij
    function RhoParsingResult(vars::VecStr, coef::VecNum, op::AbstractString,
                              rhs::Union{<:Number, <:VecNum}, eqn::AbstractString,
                              ij::AbstractVector{<:Union{<:Tuple{<:Integer, <:Integer},
                                                         <:Tuple{<:VecInt, <:VecInt}}})
        @argcheck(length(vars) == length(coef), DimensionMismatch)
        if isa(rhs, AbstractVector)
            @argcheck(!isempty(ij) &&
                      all(x -> isa(x[1], AbstractVector) && length(x[1]) == length(rhs),
                          ij),
                      DimensionMismatch("A vector `rhs` carries one value per spanned asset pair, so every entry of `ij` must be a group pair of the same length. Got\nlength(rhs) => $(length(rhs))\nij => $(ij)"))
        end
        return new{typeof(vars), typeof(coef), typeof(op), typeof(rhs), typeof(eqn),
                   typeof(ij)}(vars, coef, op, rhs, eqn, ij)
    end
end
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
    $(field_dict[:sc1])
    """
    sc1
    """
    $(field_dict[:sc2])
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
    $(field_dict[:sc1])
    """
    sc1
    """
    $(field_dict[:sc2])
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
"""
    add_ep_constraint!(epc::AbstractDict, lhs::MatNum, rhs::VecNum, key::Symbol)

Add an entropy pooling view constraint to the constraint dictionary.

`add_ep_constraint!` normalises and adds a constraint to the entropy pooling constraint dictionary `epc`. If a constraint with the same key already exists, it concatenates the new constraint to the existing one. This function is used internally to build the set of linear constraints for entropy pooling optimisation.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `lhs`: Left-hand side constraint matrix.
  - `rhs`: Right-hand side constraint vector.
  - `key`: Constraint type key (`:eq`, `:ineq`, `:feq`, `:cvar_eq`).

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Related

  - [`entropy_pooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function add_ep_constraint!(epc::AbstractDict, lhs::MatNum, rhs::VecNum, key::Symbol)
    sc = LinearAlgebra.norm(lhs)
    lhs /= sc
    rhs /= sc
    epc[key] = if !haskey(epc, key)
        (lhs, rhs)
    else
        (vcat(epc[key][1], lhs), append!(epc[key][2], rhs))
    end
    return nothing
end
"""
    replace_prior_views(res::ParsingResult, pr::AbstractPriorResult, sets::UniverseSets,
                        key::Symbol, alpha::Option{<:Number} = nothing, params...;
                        strict::Bool = false)

Replace prior references in view parsing results with their corresponding prior values.

`replace_prior_views` scans a parsed view constraint [`ParsingResult`](@ref) for references to prior values (e.g., `prior(A)`), and replaces them with the actual prior value from the provided prior result object. This ensures that prior-based terms in view constraints are treated as constants and not as variables in the optimisation. If an asset referenced in a prior is not found in the asset set, a warning is issued (or an error if `strict=true`). If all variables in the view are prior references, an error is thrown.

# Arguments

  - `res`: Parsed view constraint containing variables and coefficients.

  - `pr`: Prior result object containing prior values.

  - `sets`: Asset set mapping asset names to indices.

  - `key`: Moment type key (`:mu`, `:var`, `:cvar`, etc.).

  - `alpha`: Optional confidence level for VaR/CVaR views.

  - `params...`: Further parameters of the statistic, forwarded to [`get_pr_value`](@ref). A relativistic value-at-risk view passes its deformation parameter here.

  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `res::ParsingResult`: Updated parsing result with prior references replaced by their values.

# Details

  - Prior references are matched using the pattern `prior(<asset>)`.
  - The right-hand side of the constraint is adjusted by subtracting the prior value times its coefficient.
  - Variables corresponding to prior references are removed from the constraint.
  - Throws an error if no non-prior variables remain.

# Related

  - [`ParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`UniverseSets`](@ref)
  - [`prior`](@ref)
"""
function replace_prior_views(res::ParsingResult, pr::AbstractPriorResult,
                             sets::UniverseSets, key::Symbol,
                             alpha::Option{<:Number} = nothing, params...;
                             strict::Bool = false)
    prior_pattern = r"prior\(([^()]*)\)"
    nx = sets.dict[sets.xkey]
    variables, coeffs = res.vars, res.coef
    idx_rm = Vector{Int}(undef, 0)
    rhs::typeof(res.rhs) = res.rhs
    non_prior = false
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            non_prior = true
            continue
        end
        j = findfirst(x -> x == m.captures[1], nx)
        if isnothing(j)
            msg = unknown_variable_msg(m.captures[1], nx, sets.xkey)
            strict_diagnostic(msg, strict)
            push!(idx_rm, i)
            continue
        end
        rhs -= get_pr_value(pr, j, Val(key), alpha, params...) * c
        push!(idx_rm, i)
    end
    if isempty(idx_rm)
        return res
    end
    @argcheck(non_prior,
              ArgumentError("Priors in views are replaced by their prior value, thus they are essentially part of the constant of the view, so you need a non-prior view to serve as the variable."))
    idx = setdiff(1:length(variables), idx_rm)
    variables_new = variables[idx]
    coeffs_new = coeffs[idx]
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "))
    return ParsingResult(variables_new, coeffs_new, res.op, rhs, "$(eqn) $(res.op) $(rhs)")
end
"""
    replace_prior_views(res::VecPR, args...; kwargs...)

Broadcast prior reference replacement across multiple view constraints.

`replace_prior_views` applies [`replace_prior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values.

# Arguments

  - `res:`: Vector of parsed view constraints.
  - `args...`: Additional positional arguments forwarded to [`replace_prior_views`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`replace_prior_views`](@ref).

# Returns

  - `res::Vector{<:ParsingResult}`: Vector of updated parsing results with prior references replaced by their values.

# Related

  - [`ParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`UniverseSets`](@ref)
"""
function replace_prior_views(res::VecPR, args...; kwargs...)
    return replace_prior_views.(res, args...; kwargs...)
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:mu}, args...)

Extract the mean (expected return) for asset `i` from a prior result.

`get_pr_value` returns the mean value for the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:mu}`: Dispatch tag for mean extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `mu::Number`: Mean (expected return) for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:mu}, args...)
    return pr.mu[i]
end
"""
    ep_mu_views!(mu_views::Nothing, args...; kwargs...)

Do nothing when no mean view constraints are specified.

`ep_mu_views!` is an internal API compatibility method that does nothing when mean view constraints (`mu_views`) are not provided (`mu_views = nothing`). This allows higher-level entropy pooling routines to uniformly call `ep_mu_views!` without special-casing the absence of mean views.

# Arguments

  - `mu_views::Nothing`: Indicates that no mean view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`ep_mu_views!`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_mu_views!(mu_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_mu_views!(mu_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)

Parse and add mean (expected return) view constraints to the entropy pooling constraint dictionary.

`ep_mu_views!` parses mean view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method is used internally by entropy pooling routines to enforce mean views in the optimisation.

# Arguments

  - `mu_views`: Mean view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Supports both equality and fixed equality constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_mu_views!(mu_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    X = pr.X
    mu_views = parse_equation(mu_views.val; datatype = eltype(X))
    mu_views = replace_group_by_assets(mu_views, sets, false, true, false)
    mu_views = replace_prior_views(mu_views, pr, sets, :mu; strict = strict)
    lcs = get_linear_constraints(mu_views, sets; datatype = eltype(X), strict = strict)
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        add_ep_constraint!(epc, getproperty(lcs, p).A * transpose(X), getproperty(lcs, p).B,
                           p)
    end
    return nothing
end
"""
    fix_mu!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
            pr::AbstractPriorResult)

Add constraints to fix the mean of specified assets in entropy pooling.

`fix_mu!` identifies assets in `to_fix` that are not yet fixed (i.e., not present in `fixed`), and adds constraints to the entropy pooling constraint dictionary `epc` to fix their mean to the prior value. This ensures that higher moment views (e.g., variance, skewness, kurtosis, correlation) do not inadvertently alter the mean of these assets. The function updates `fixed` in-place to reflect the newly fixed assets.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `fixed`: Boolean vector indicating which assets have their mean fixed.
  - `to_fix`: Boolean vector indicating which assets should have their mean fixed.
  - `pr`: Prior result containing asset return information.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in-place.

# Details

  - Adds a fixed equality constraint (`:feq`) for each asset in `to_fix` that is not yet fixed.
  - Uses the prior mean values from `pr.mu` for the constraint right-hand side.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function fix_mu!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
                 pr::AbstractPriorResult)
    fix = to_fix .& .!fixed
    if any(fix)
        add_ep_constraint!(epc, transpose(view(pr.X, :, fix)), pr.mu[fix], :feq)
        fixed .= fixed .| fix
    end
    return nothing
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Number)

Extract the Value-at-Risk (VaR) for asset `i` from a prior result.

`get_pr_value` computes the VaR at confidence level `alpha` for the asset indexed by `i` from the prior result object `pr`. This method uses the asset return samples in `pr` and applies the VaR calculation, typically using the empirical quantile.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:var}`: Dispatch tag for VaR extraction.
  - `alpha`: Confidence level (e.g., `0.05` for 5% VaR).

# Returns

  - `var::Number`: Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Number)
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ValueatRisk(; alpha = alpha)(view(pr.X, :, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that carry a group of entropy pooling views together with the settings those views are read under.

A significance level is a property of a view, not of the estimator that holds it: the value at risk at 1% and at 10% are different statistics of the same series. An estimator of this family pairs a group of view equations with the settings they are read under, so one entropy pooling estimator can hold views stated at several levels.

# Related

  - [`ValueatRiskView`](@ref)
  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
abstract type AbstractEntropyPoolingViewEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

A group of value-at-risk views, with the significance level they are read under.

Unlike a conditional or entropic value at risk view, a value at risk view is linear in the posterior probabilities: it reduces to rows of the constraint set through [`add_ep_constraint!`](@ref), so it needs no auxiliary variable, admits no choice of formulation, and reaches [`OptimEntropyPooling`](@ref) as readily as [`JuMPEntropyPooling`](@ref). That is why this estimator carries a level and nothing else.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05
    ) -> ValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.

# Details

  - A `prior(...)` reference inside `views` is replaced by the prior VaR at this view's `alpha`, so a view stated against the prior moves with the level.
  - Accepts `==` and `>=` alone, one asset per view, with a unit coefficient and a non-negative target.

# Examples

```jldoctest
julia> ValueatRiskView(; alpha = 0.01, views = LinearConstraintEstimator(; val = \"A >= 0.05\"))
ValueatRiskView
  views ┼ LinearConstraintEstimator
        │   val ┼ String: "A >= 0.05"
        │   key ┴ nothing
  alpha ┴ Float64: 0.01
```

# Related

  - [`AbstractEntropyPoolingViewEstimator`](@ref)
  - [`ep_var_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
@concrete struct ValueatRiskView <: AbstractEntropyPoolingViewEstimator
    """
    $(field_dict[:ep_vv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    function ValueatRiskView(views::LinearConstraintEstimator, alpha::Number)
        assert_unit_interval(alpha, :alpha)
        return new{typeof(views), typeof(alpha)}(views, alpha)
    end
end
function ValueatRiskView(; views::LinearConstraintEstimator,
                         alpha::Number = 0.05)::ValueatRiskView
    return ValueatRiskView(views, alpha)
end
"""
    const VV_VecVV = Union{<:ValueatRiskView, <:AbstractVector{<:ValueatRiskView}}

Alias for the shapes a `var_views` field accepts: one [`ValueatRiskView`](@ref), or a vector of them read under their own significance levels.

# Related

  - [`ValueatRiskView`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
const VV_VecVV = Union{<:ValueatRiskView, <:AbstractVector{<:ValueatRiskView}}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations that express a tail view inside an entropy pooling problem.

A tail view constrains a quantile-based risk measure of the posterior distribution. Unlike a mean, variance or correlation view, it is not a linear function of the posterior probabilities, so each measure admits more than one way of writing it as a solvable program. The concrete subtypes name those ways.

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
abstract type AbstractEntropyPoolingViewFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations of a conditional value-at-risk view.

# Related

  - [`AbstractEntropyPoolingViewFormulation`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
abstract type AbstractConditionalValueatRiskViewFormulation <:
              AbstractEntropyPoolingViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations of an entropic value-at-risk view.

# Related

  - [`AbstractEntropyPoolingViewFormulation`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
abstract type AbstractEntropicValueatRiskViewFormulation <:
              AbstractEntropyPoolingViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the formulations of a relativistic value-at-risk view.

# Related

  - [`AbstractEntropyPoolingViewFormulation`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
abstract type AbstractRelativisticValueatRiskViewFormulation <:
              AbstractEntropyPoolingViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Linear formulation of a conditional value-at-risk view [EPTail](@cite).

`LinearConditionalValueatRiskView` writes the view through the dual representation of CVaR. It adds ``T`` continuous variables and no integer variable, so it is the cheapest of the two CVaR formulations, and it is exact.

# Mathematical definition

Let ``\\boldsymbol{x}`` be the loss series of the asset the view names, ``\\boldsymbol{w}`` the posterior probabilities, ``\\alpha`` the significance level and ``\\bar{c}`` the target. The view ``\\mathrm{CVaR}_{\\alpha}(X) \\geq \\bar{c}`` is written as:

```math
\\begin{align}
&\\nu_{j} \\geq 0\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\nu_{j} \\leq \\dfrac{w_{j}}{\\alpha}\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{c}\\,.
\\end{align}
```

Where ``\\boldsymbol{\\nu}`` is the vector of weights that attains the CVaR. The constraint set is feasible if and only if ``\\mathrm{CVaR}_{\\alpha}(X) \\geq \\bar{c}``, so a lower-bound view is exact.

# Scope

  - Operators: `>=` and `==`.
  - One asset per view, with a positive coefficient.
  - An equality view needs a target greater than or equal to the prior CVaR of the asset. Below the prior CVaR the constraint is slack at the prior, so the entropy minimiser leaves the prior untouched and the view is not met. Use [`IntegerConditionalValueatRiskView`](@ref) there.

# Examples

```jldoctest
julia> LinearConditionalValueatRiskView()
LinearConditionalValueatRiskView()
```

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
struct LinearConditionalValueatRiskView <: AbstractConditionalValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Integer formulation of a conditional value-at-risk view [EPTail](@cite).

`IntegerConditionalValueatRiskView` writes the view through the ordered weights representation of CVaR, selecting the tail of the posterior with a monotone binary vector. It expresses every comparison operator and any linear combination of per-asset CVaRs, at the cost of `sbar` binary variables per asset named by the view. It needs a solver that handles mixed-integer exponential cone programs.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

Let ``x_{[1]} \\leq x_{[2]} \\leq \\ldots \\leq x_{[\\bar{s}]}`` be the ``\\bar{s}`` largest losses of the asset sorted in ascending order, so the largest loss is last, ``w_{[j]}`` the posterior probability of the observation in position ``j``, and ``\\alpha`` the significance level:

```math
\\begin{align}
&y_{j} \\leq y_{j+1}\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}-1\\\\
&q_{j} \\leq y_{j}\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}\\\\
&q_{j} \\leq w_{[j]}\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}\\\\
&q_{j} \\geq w_{[j]} - (1 - y_{j})\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}\\\\
&q_{j} \\geq 0\\,, &\\forall\\, j = 1,\\ldots,\\bar{s}\\\\
&\\alpha = \\sum_{j=1}^{\\bar{s}} q_{j}\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{\\bar{s}}\\\\
&\\mathrm{CVaR}_{\\alpha}(X) = \\dfrac{1}{\\alpha} \\sum_{j=1}^{\\bar{s}} q_{j} x_{[j]}\\,.
\\end{align}
```

The auxiliary vector ``\\boldsymbol{q}`` carries ``q_{j} = w_{[j]} y_{j}``, and ``\\boldsymbol{y}`` marks the observations that enter the tail. The monotonicity constraint makes the marked set a suffix of the ascending order, which is what makes the expression the CVaR rather than the mean of an arbitrary subset of probability ``\\alpha``.

# Details

  - `sbar` trades exactness for solve time. `sbar = T` is always exact. A smaller `sbar` is exact whenever the posterior puts at least ``\\alpha`` of its mass on the `sbar` largest losses, and infeasible otherwise.
  - If `nothing`, `sbar` is `max(2 * s, ceil(Int, 2 * alpha * T))` capped at `T`, where `s` is the number of positions, counted from the largest loss, at which the prior probabilities first reach `alpha`. This follows the rule of thumb of [EPTail](@cite): a view above the prior CVaR needs about `s` positions, a view below it needs more.
  - Raise `sbar` when the solve reports infeasibility.

# Constructors

    IntegerConditionalValueatRiskView(;
        sbar::Option{<:Number} = nothing
    ) -> IntegerConditionalValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - If `sbar` is an `Integer`, `sbar >= 1`.
  - If `sbar` is not an `Integer`, `0 < sbar < 1`. Use an `Integer` to name the whole sample.

# Examples

```jldoctest
julia> IntegerConditionalValueatRiskView()
IntegerConditionalValueatRiskView
  sbar ┴ nothing
```

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct IntegerConditionalValueatRiskView <:
                 AbstractConditionalValueatRiskViewFormulation
    """
    $(field_dict[:sbar])
    """
    sbar
    function IntegerConditionalValueatRiskView(sbar::Option{<:Number})
        if isa(sbar, Integer)
            @argcheck(sbar >= one(sbar), DomainError(sbar, "sbar must be >= 1"))
        elseif !isnothing(sbar)
            assert_unit_interval(sbar, :sbar)
        end
        return new{typeof(sbar)}(sbar)
    end
end
function IntegerConditionalValueatRiskView(;
                                           sbar::Option{<:Number} = nothing)::IntegerConditionalValueatRiskView
    return IntegerConditionalValueatRiskView(sbar)
end
"""
$(DocStringExtensions.TYPEDEF)

Exponential cone formulation of an entropic value-at-risk view [EPTail](@cite).

`ConicEntropicValueatRiskView` writes the view through the dual representation of EVaR. It adds ``T`` continuous variables and one relative entropy cone, and it is exact.

# Mathematical definition

Let ``\\boldsymbol{x}`` be the loss series of the asset the view names, ``\\boldsymbol{w}`` the posterior probabilities, ``\\alpha`` the significance level and ``\\bar{e}`` the target. The view ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}`` is written as:

```math
\\begin{align}
&0 \\leq \\nu_{j} \\leq 1\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} \\ln\\left(\\dfrac{\\nu_{j}}{w_{j}}\\right) \\leq \\ln\\left(\\dfrac{1}{\\alpha}\\right)\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{e}\\,.
\\end{align}
```

Where ``\\boldsymbol{\\nu}`` is the vector of weights that attains the EVaR. The relative entropy budget is the dual description of EVaR, so the constraint set is feasible if and only if ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}``.

# Scope

  - Operators: `>=` and `==`.
  - One asset per view, with a positive coefficient.
  - An equality view needs a target greater than or equal to the prior EVaR of the asset. Use [`GridEntropicValueatRiskView`](@ref) below it.

# Examples

```jldoctest
julia> ConicEntropicValueatRiskView()
ConicEntropicValueatRiskView()
```

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
struct ConicEntropicValueatRiskView <: AbstractEntropicValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Grid formulation of an entropic value-at-risk view [EPTail](@cite).

`GridEntropicValueatRiskView` writes the view on a grid of values of the EVaR dual variable, built around the value that attains the prior EVaR of the asset. A lower-bound view is a set of linear constraints and needs no integer variable. An upper-bound or equality view selects one grid point with a binary vector and a big-``M`` relaxation, and needs a solver that handles mixed-integer exponential cone programs.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The sample EVaR is the value of a scalar minimisation:

```math
\\mathrm{EVaR}_{\\alpha}(X) = \\min_{z > 0} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
```

So ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}`` holds exactly when the objective is at or above ``\\bar{e}`` at *every* ``z``, and ``\\mathrm{EVaR}_{\\alpha}(X) \\leq \\bar{e}`` holds when it is at or below ``\\bar{e}`` at *some* ``z``. On a grid ``\\bar{z}_{1},\\ldots,\\bar{z}_{K}`` that gives, for a lower-bound view:

```math
\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/\\bar{z}_{k})}{\\exp(\\bar{e}/\\bar{z}_{k})} \\geq \\alpha\\,, \\quad \\forall\\, k = 1,\\ldots,K
```

and for an upper-bound view, with ``\\boldsymbol{y}`` a binary selector and ``M`` a big constant:

```math
\\begin{align}
&\\boldsymbol{1}^{\\intercal} \\boldsymbol{y} = 1\\\\
&\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/\\bar{z}_{k})}{\\exp(\\bar{e}/\\bar{z}_{k})} \\leq \\alpha + M(1 - y_{k})\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{K}\\,.
\\end{align}
```

An equality view carries both blocks.

# Details

  - The grid is `K` equidistant points spanning `zstar * (1 - pct)` to `zstar * (1 + pct)`, where `zstar` attains the prior EVaR of the asset. `K` is odd so `zstar` sits in the middle.
  - The answer is approximate in both directions. A lower-bound view holds at the grid points and may fall short between them; an upper-bound view holds at one grid point and may be conservative. Widen `pct` or raise `K` when the posterior value misses the target, and prefer [`ConicEntropicValueatRiskView`](@ref) whenever the view admits it.
  - Rows are scaled by their largest coefficient before they reach the model, so the default `M` is far above the largest attainable violation.

# Constructors

    GridEntropicValueatRiskView(;
        pct::Number = 0.5,
        K::Integer = 11,
        M::Number = 10
    ) -> GridEntropicValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < pct < 1`.
  - `K >= 1` and `isodd(K)`.
  - `M > 0`.

# Examples

```jldoctest
julia> GridEntropicValueatRiskView()
GridEntropicValueatRiskView
  pct ┼ Float64: 0.5
    K ┼ Int64: 11
    M ┴ Int64: 10
```

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct GridEntropicValueatRiskView <: AbstractEntropicValueatRiskViewFormulation
    """
    $(field_dict[:zpct])
    """
    pct
    """
    $(field_dict[:zK])
    """
    K
    """
    $(field_dict[:bigM])
    """
    M
    function GridEntropicValueatRiskView(pct::Number, K::Integer, M::Number)
        assert_unit_interval(pct, :pct)
        @argcheck(K >= one(K) && isodd(K), DomainError(K, "K must be odd and >= 1"))
        @argcheck(M > zero(M), DomainError(M, "M must be > 0"))
        return new{typeof(pct), typeof(K), typeof(M)}(pct, K, M)
    end
end
function GridEntropicValueatRiskView(; pct::Number = 0.5, K::Integer = 11,
                                     M::Number = 10)::GridEntropicValueatRiskView
    return GridEntropicValueatRiskView(pct, K, M)
end
"""
$(DocStringExtensions.TYPEDEF)

Power cone formulation of a relativistic value-at-risk view [EPRLVaR](@cite).

`ConicRelativisticValueatRiskView` writes the view through the dual representation of RLVaR. It adds ``3T`` continuous variables and ``2T`` power cones, and it is exact.

# Mathematical definition

Let ``\\boldsymbol{x}`` be the loss series of the asset the view names, ``\\boldsymbol{w}`` the posterior probabilities, ``\\alpha`` the significance level, ``\\kappa`` the deformation parameter and ``\\bar{\\vartheta}`` the target. Write ``\\ln_{\\kappa}`` for the Kaniadakis logarithm, [`kappa_log`](@ref). The view ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) \\geq \\bar{\\vartheta}`` is written as:

```math
\\begin{align}
&0 \\leq \\nu_{j} \\leq 1\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\dfrac{\\tau_{j} - \\varsigma_{j}}{2\\kappa} \\leq \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\\\
&\\left(\\tau_{j},\\, T w_{j},\\, \\nu_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\left(\\dfrac{1}{1+\\kappa}\\right)\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\left(\\nu_{j},\\, T w_{j},\\, \\varsigma_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{\\vartheta}\\,.
\\end{align}
```

Where ``\\boldsymbol{\\nu}`` is the vector of weights that attains the RLVaR, ``\\boldsymbol{\\tau}`` and ``\\boldsymbol{\\varsigma}`` carry its Kaniadakis entropy budget, and ``\\mathcal{K}_{\\mathrm{pow}}(p) = \\{(a,b,c) : a^{p} b^{1-p} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}`` is the power cone. The budget is the dual description of RLVaR, so the constraint set is feasible if and only if ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) \\geq \\bar{\\vartheta}``.

# Scope

  - Operators: `>=` and `==`.
  - One asset per view, with a positive coefficient.
  - An equality view needs a target greater than or equal to the prior RLVaR of the asset. Use [`GridRelativisticValueatRiskView`](@ref) below it.
  - The solver must handle the power cone alongside the exponential cone the entropy pooling objective needs.
  - The programme is a demanding solve. A long sample, a small `alpha`, a small `kappa` or several of these views in one model can make a conic solver stop short of a solution. Give `opt` a vector of solver configurations, shorten the sample, or state the view under [`GridRelativisticValueatRiskView`](@ref), whose rows are linear in the posterior probabilities.

# Examples

```jldoctest
julia> ConicRelativisticValueatRiskView()
ConicRelativisticValueatRiskView()
```

# Related

  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`kappa_log`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
struct ConicRelativisticValueatRiskView <: AbstractRelativisticValueatRiskViewFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Grid formulation of a relativistic value-at-risk view.

`GridRelativisticValueatRiskView` writes the view on a grid of points of the primal programme of RLVaR, centred on the point a posterior that meets the view attains. A lower-bound view is a set of linear constraints and needs no integer variable. An upper-bound or equality view selects one grid point with a binary vector and a big-``M`` relaxation, and needs a solver that handles mixed-integer exponential cone programs.

As `kappa` approaches one the RLVaR approaches the largest loss, and [`ep_rlvar_tail`](@ref) overflows at the dual variable that attains it. The points it overflows at are dropped, and a grid that keeps none of them raises. The centre of the grid is found by an iteration that reads the same tail function, so it too stops converging there and the grid falls back to the prior's dual variable, which lands short of the target. Prefer a smaller `kappa`, or [`ConicRelativisticValueatRiskView`](@ref) where the operator admits it.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The sample RLVaR is the value of a two-variable minimisation, in which the pair of power cones of each observation is already minimised out:

```math
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) = \\min_{t,\\, z > 0} \\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,,
```

Where ``\\varphi_{\\kappa}(u, z)`` is the smallest ``\\psi + \\theta`` the two power cones of one observation allow, and has the closed form:

```math
\\begin{align}
\\varphi_{\\kappa}(u, z) &= \\dfrac{\\kappa}{1+\\kappa} \\left(\\dfrac{2\\kappa}{(1+\\kappa) z}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma - u}{2}\\right)^{\\frac{1+\\kappa}{\\kappa}} + \\kappa (1-\\kappa)^{\\frac{1-\\kappa}{\\kappa}} \\left(\\dfrac{z}{2\\kappa}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma + u}{2}\\right)^{-\\frac{1-\\kappa}{\\kappa}}\\,,\\\\
\\sigma &= \\sqrt{u^{2} + \\dfrac{(1 - \\kappa^{2}) z^{2}}{\\kappa^{2}}}\\,.
\\end{align}
```

The objective is linear in ``\\boldsymbol{w}`` once ``t`` and ``z`` are fixed, which is what makes a grid point a row. On a grid ``(\\bar{t}_{1}, \\bar{z}_{1}),\\ldots,(\\bar{t}_{K}, \\bar{z}_{K})`` that gives, for a lower-bound view:

```math
T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) \\geq \\bar{\\vartheta} - \\bar{t}_{k} - \\bar{z}_{k} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\,, \\quad \\forall\\, k = 1,\\ldots,K
```

and for an upper-bound view, with ``\\boldsymbol{y}`` a binary selector and ``M`` a big constant:

```math
\\begin{align}
&\\boldsymbol{1}^{\\intercal} \\boldsymbol{y} = 1\\\\
&T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) \\leq \\bar{\\vartheta} - \\bar{t}_{k} - \\bar{z}_{k} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + M(1 - y_{k})\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{K}\\,.
\\end{align}
```

An equality view carries both blocks. Every grid point is a feasible point of the primal programme, so the upper-bound block is never violated: it can only be tighter than the view asks. The lower-bound block holds at the grid points and may fall short between them, so prefer [`ConicRelativisticValueatRiskView`](@ref) whenever the view admits it. Rows are scaled by their largest coefficient before they reach the model, so the coefficients sit in `(0, 1]` and the posterior sums to one. The left-hand side is therefore bounded by one whatever the data, and the default `M` clears that bound by an order of magnitude.

The grid spans `zc * (1 - pct)` to `zc * (1 + pct)`, and `K` is odd so `zc` sits in the middle. A view that carries an upper-bound half takes `zc` from [`ep_rlvar_anchor`](@ref), which solves for the pair a posterior meeting the view attains, and each `t_k` is the shift that minimises the objective at `z_k` under that posterior. The view then lands on its target, and `pct` and `K` cover only the movement the other views of the model cause.

A lower-bound view, and a view whose anchor does not converge, takes `zc` from the prior's dual variable. RLVaR and the shift that attains it are both translation-equivariant, so a posterior that moves the RLVaR to the target behaves, to first order, like translating every loss by the same amount. Each `t_k` is then the shift that minimises the objective at `z_k` under the prior probabilities, less the distance from the prior RLVaR to the target.

# Constructors

    GridRelativisticValueatRiskView(;
        pct::Number = 0.5,
        K::Integer = 11,
        M::Number = 10
    ) -> GridRelativisticValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < pct < 1`.
  - `K >= 1` and `isodd(K)`.
  - `M > 0`.

# Examples

```jldoctest
julia> GridRelativisticValueatRiskView()
GridRelativisticValueatRiskView
  pct ┼ Float64: 0.5
    K ┼ Int64: 11
    M ┴ Int64: 10
```

# Related

  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`ep_rlvar_anchor`](@ref)
  - [`ep_rlvar_grid`](@ref)
  - [`ep_rlvar_tail`](@ref)
  - [`kappa_log`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
@concrete struct GridRelativisticValueatRiskView <:
                 AbstractRelativisticValueatRiskViewFormulation
    """
    $(field_dict[:rlvar_zpct])
    """
    pct
    """
    $(field_dict[:rlvar_zK])
    """
    K
    """
    $(field_dict[:rlvar_bigM])
    """
    M
    function GridRelativisticValueatRiskView(pct::Number, K::Integer, M::Number)
        assert_unit_interval(pct, :pct)
        @argcheck(K >= one(K) && isodd(K), DomainError(K, "K must be odd and >= 1"))
        @argcheck(M > zero(M), DomainError(M, "M must be > 0"))
        return new{typeof(pct), typeof(K), typeof(M)}(pct, K, M)
    end
end
function GridRelativisticValueatRiskView(; pct::Number = 0.5, K::Integer = 11,
                                         M::Number = 10)::GridRelativisticValueatRiskView
    return GridRelativisticValueatRiskView(pct, K, M)
end
"""
    const CVaRVF_VecCVaRVF = Union{<:AbstractConditionalValueatRiskViewFormulation,
                                   <:AbstractVector{<:AbstractConditionalValueatRiskViewFormulation}}

Alias for a union of a single conditional value-at-risk view formulation or a vector of them.

# Related

  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
"""
const CVaRVF_VecCVaRVF = Union{<:AbstractConditionalValueatRiskViewFormulation,
                               <:AbstractVector{<:AbstractConditionalValueatRiskViewFormulation}}
"""
    const EVaRVF_VecEVaRVF = Union{<:AbstractEntropicValueatRiskViewFormulation,
                                   <:AbstractVector{<:AbstractEntropicValueatRiskViewFormulation}}

Alias for a union of a single entropic value-at-risk view formulation or a vector of them.

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
"""
const EVaRVF_VecEVaRVF = Union{<:AbstractEntropicValueatRiskViewFormulation,
                               <:AbstractVector{<:AbstractEntropicValueatRiskViewFormulation}}
"""
    const RLVaRVF_VecRLVaRVF = Union{<:AbstractRelativisticValueatRiskViewFormulation,
                                     <:AbstractVector{<:AbstractRelativisticValueatRiskViewFormulation}}

Alias for a union of a single relativistic value-at-risk view formulation or a vector of them.

# Related

  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
"""
const RLVaRVF_VecRLVaRVF = Union{<:AbstractRelativisticValueatRiskViewFormulation,
                                 <:AbstractVector{<:AbstractRelativisticValueatRiskViewFormulation}}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the estimators that carry a group of tail views together with the settings those views are read under.

A significance level is a property of a view, not of the estimator that holds it: the conditional value at risk at 1% and at 10% are different statistics of the same series. An estimator of this family pairs a group of view equations with the level and the formulation they take, so one [`EntropyPoolingPrior`](@ref) can hold views stated at several levels.

# Related

  - [`AbstractEntropyPoolingViewEstimator`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
abstract type AbstractEntropyPoolingTailViewEstimator <: AbstractEntropyPoolingViewEstimator end
"""
$(DocStringExtensions.TYPEDEF)

A group of conditional value-at-risk views, with the significance level and formulation they are read under.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionalValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        alg::Option{<:CVaRVF_VecCVaRVF} = nothing
    ) -> ConditionalValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `alg` is a vector, `!isempty(alg)`.

# Details

  - `alg` left `nothing` lets each view in the group take the cheapest formulation that expresses it exactly.
  - A `prior(...)` reference inside `views` is replaced by the prior CVaR at this view's `alpha`, so a view stated against the prior moves with the level.

# Examples

```jldoctest
julia> ConditionalValueatRiskView(; alpha = 0.01,
                                  views = LinearConstraintEstimator(; val = \"A >= 0.07\"))
ConditionalValueatRiskView
  views ┼ LinearConstraintEstimator
        │   val ┼ String: "A >= 0.07"
        │   key ┴ nothing
  alpha ┼ Float64: 0.01
    alg ┴ nothing
```

# Related

  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`AbstractConditionalValueatRiskViewFormulation`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct ConditionalValueatRiskView <: AbstractEntropyPoolingTailViewEstimator
    """
    $(field_dict[:ep_tv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    """
    $(field_dict[:ep_tv_alg])
    """
    alg
    function ConditionalValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                        alg::Option{<:CVaRVF_VecCVaRVF})
        assert_unit_interval(alpha, :alpha)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        return new{typeof(views), typeof(alpha), typeof(alg)}(views, alpha, alg)
    end
end
function ConditionalValueatRiskView(; views::LinearConstraintEstimator,
                                    alpha::Number = 0.05,
                                    alg::Option{<:CVaRVF_VecCVaRVF} = nothing)::ConditionalValueatRiskView
    return ConditionalValueatRiskView(views, alpha, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

A group of entropic value-at-risk views, with the significance level and formulation they are read under.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropicValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        alg::Option{<:EVaRVF_VecEVaRVF} = nothing
    ) -> EntropicValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `alg` is a vector, `!isempty(alg)`.

# Details

  - `alg` left `nothing` lets each view in the group take the cheapest formulation that expresses it exactly.
  - `alg` is where the grid of dual variables and the big-M constant live: a [`GridEntropicValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`, so views at different significance levels can take different grids.
  - A `prior(...)` reference inside `views` is replaced by the prior EVaR at this view's `alpha`.

# Examples

```jldoctest
julia> EntropicValueatRiskView(; alpha = 0.01,
                               views = LinearConstraintEstimator(; val = \"A <= 0.09\"),
                               alg = GridEntropicValueatRiskView(; pct = 0.8, K = 21))
EntropicValueatRiskView
  views ┼ LinearConstraintEstimator
        │   val ┼ String: "A <= 0.09"
        │   key ┴ nothing
  alpha ┼ Float64: 0.01
    alg ┼ GridEntropicValueatRiskView
        │   pct ┼ Float64: 0.8
        │     K ┼ Int64: 21
        │     M ┴ Int64: 10
```

# Related

  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct EntropicValueatRiskView <: AbstractEntropyPoolingTailViewEstimator
    """
    $(field_dict[:ep_tv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    """
    $(field_dict[:ep_tv_alg])
    """
    alg
    function EntropicValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                     alg::Option{<:EVaRVF_VecEVaRVF})
        assert_unit_interval(alpha, :alpha)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        return new{typeof(views), typeof(alpha), typeof(alg)}(views, alpha, alg)
    end
end
function EntropicValueatRiskView(; views::LinearConstraintEstimator, alpha::Number = 0.05,
                                 alg::Option{<:EVaRVF_VecEVaRVF} = nothing)::EntropicValueatRiskView
    return EntropicValueatRiskView(views, alpha, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

A group of relativistic value-at-risk views, with the significance level, the deformation parameter and the formulation they are read under.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        alg::Option{<:RLVaRVF_VecRLVaRVF} = nothing
    ) -> RelativisticValueatRiskView

Keywords correspond to the struct's fields. `alg` left `nothing` lets each view in the group take the cheapest formulation that expresses it exactly, and is where the grid of primal points and the big-M constant live: a [`GridRelativisticValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`. A `prior(...)` reference inside `views` is replaced by the prior RLVaR at this group's `alpha` and `kappa`, so a view stated against the prior moves with both.

## Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.
  - If `alg` is a vector, `!isempty(alg)`.

# Examples

```jldoctest
julia> RelativisticValueatRiskView(; alpha = 0.01, kappa = 0.5,
                                   views = LinearConstraintEstimator(; val = \"A >= 0.09\"))
RelativisticValueatRiskView
  views ┼ LinearConstraintEstimator
        │   val ┼ String: "A >= 0.09"
        │   key ┴ nothing
  alpha ┼ Float64: 0.01
  kappa ┼ Float64: 0.5
    alg ┴ nothing
```

# Related

  - [`AbstractEntropyPoolingTailViewEstimator`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`AbstractRelativisticValueatRiskViewFormulation`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
@concrete struct RelativisticValueatRiskView <: AbstractEntropyPoolingTailViewEstimator
    """
    $(field_dict[:ep_tv_views])
    """
    views
    """
    $(field_dict[:ep_tv_alpha])
    """
    alpha
    """
    $(field_dict[:ep_tv_kappa])
    """
    kappa
    """
    $(field_dict[:ep_tv_alg])
    """
    alg
    function RelativisticValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                         kappa::Number, alg::Option{<:RLVaRVF_VecRLVaRVF})
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(kappa, :kappa)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        return new{typeof(views), typeof(alpha), typeof(kappa), typeof(alg)}(views, alpha,
                                                                             kappa, alg)
    end
end
function RelativisticValueatRiskView(; views::LinearConstraintEstimator,
                                     alpha::Number = 0.05, kappa::Number = 0.3,
                                     alg::Option{<:RLVaRVF_VecRLVaRVF} = nothing)::RelativisticValueatRiskView
    return RelativisticValueatRiskView(views, alpha, kappa, alg)
end
"""
    const CVV_VecCVV = Union{<:ConditionalValueatRiskView,
                             <:AbstractVector{<:ConditionalValueatRiskView}}

Alias for the shapes a `cvar_views` field accepts: one [`ConditionalValueatRiskView`](@ref), or a vector of them read under their own significance levels and formulations.

# Related

  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const CVV_VecCVV = Union{<:ConditionalValueatRiskView,
                         <:AbstractVector{<:ConditionalValueatRiskView}}
"""
    const EVV_VecEVV = Union{<:EntropicValueatRiskView,
                             <:AbstractVector{<:EntropicValueatRiskView}}

Alias for the shapes an `evar_views` field accepts: one [`EntropicValueatRiskView`](@ref), or a vector of them read under their own significance levels and formulations.

# Related

  - [`EntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const EVV_VecEVV = Union{<:EntropicValueatRiskView,
                         <:AbstractVector{<:EntropicValueatRiskView}}
"""
    const RVV_VecRVV = Union{<:RelativisticValueatRiskView,
                             <:AbstractVector{<:RelativisticValueatRiskView}}

Alias for the shapes an `rlvar_views` field accepts: one [`RelativisticValueatRiskView`](@ref), or a vector of them read under their own significance levels, deformation parameters and formulations.

# Related

  - [`RelativisticValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
const RVV_VecRVV = Union{<:RelativisticValueatRiskView,
                         <:AbstractVector{<:RelativisticValueatRiskView}}
"""
    ep_var_views!(var_views::Nothing, args...; kwargs...)

Do nothing when no value at risk (VaR) view constraints are specified.

`ep_var_views!` is an internal API compatibility method that does nothing when value at risk (VaR) view constraints (`var_views`) are not provided (`var_views = nothing`). This allows higher-level entropy pooling routines to uniformly call `ep_var_views!` without special-casing the absence of value at risk (VaR) views.

# Arguments

  - `var_views::Nothing`: Indicates that no value at risk (VaR) view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`ep_var_views!`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_var_views!(var_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::UniverseSets, alpha::Number; strict::Bool = false)

Parse and add value at risk (VaR) view constraints to the entropy pooling constraint dictionary.

`ep_var_views!` parses VaR view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. A VaR view is linear in the posterior probabilities: it constrains the probability mass at or below the target loss to be the significance level, so it needs no auxiliary variable. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method validates that only single-asset, non-negative, and unit-coefficient views are allowed, and throws informative errors for invalid or extreme views.

# Arguments

  - `var_views`: VaR view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `alpha`: Confidence level for VaR.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Validates that only equality and inequality constraints with unit coefficients are present.
  - Throws errors for negative or multi-asset views, or if the view is more extreme than the worst realisation.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_var_views!(var_views::ValueatRiskView, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    return ep_var_views!(var_views.views, epc, pr, sets, var_views.alpha; strict = strict)
end
"""
    ep_var_views!(var_views::AbstractVector{<:ValueatRiskView}, args...; kwargs...)

Add each group of value at risk views under its own significance level.

Every [`ValueatRiskView`](@ref) in the vector is added in turn, so the groups accumulate into the same constraint set and one entropy pooling solve answers all of them.

# Arguments

  - `var_views`: Groups of VaR views.
  - `args...`: Additional positional arguments forwarded to [`ep_var_views!`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`ep_var_views!`](@ref).

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Related

  - [`ValueatRiskView`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_var_views!(var_views::AbstractVector{<:ValueatRiskView}, args...; kwargs...)
    for var_view in var_views
        ep_var_views!(var_view, args...; kwargs...)
    end
    return nothing
end
function ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::UniverseSets, alpha::Number;
                       strict::Bool = false)
    X = pr.X
    var_views = parse_equation(var_views.val; ops1 = ("==", ">="),
                               ops2 = (:call, :(==), :(>=)), datatype = eltype(X))
    var_views = replace_group_by_assets(var_views, sets, false, true, false)
    var_views = replace_prior_views(var_views, pr, sets, :var, alpha; strict = strict)
    lcs = get_linear_constraints(var_views, sets; datatype = eltype(X), strict = strict)
    @argcheck(!(!isnothing(lcs.ineq) && !any(x -> (iszero(x) || isone(x)), lcs.A_ineq) ||
                !isnothing(lcs.eq) && !any(x -> (iszero(x) || isone(x)), lcs.A_eq)),
              ArgumentError("var_view only supports coefficients of 1.\n$var_views"))
    @argcheck(!(!isnothing(lcs.ineq) &&
                any(x -> x != 1, count(!iszero, lcs.A_ineq; dims = 2)) ||
                !isnothing(lcs.eq) && any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2))),
              ArgumentError("Cannot mix multiple assets in a single var_view.\n$var_views"))
    @argcheck(!(!isnothing(lcs.eq) && any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq) ||
                !isnothing(lcs.ineq) &&
                any(x -> x < zero(eltype(x)), lcs.A_ineq .* lcs.B_ineq)),
              DomainError("var_views cannot be negative.\n$var_views"))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        B = getproperty(lcs, p).B
        for i in eachindex(B)
            j = .!iszero.(A[i, :])
            #! Figure out a way to include pr.w, probably see how it's implemented in ValueatRisk.
            idx = findall(x -> x <= -abs(B[i]), view(X, :, j))
            @argcheck(!isempty(idx),
                      DomainError("View $(i) = $(var_views[i].eqn) is too extreme, the maximum viable for asset $(findfirst(x -> x == true, j)) is $(-minimum(X[:,j])). Please lower it or use a different prior with fatter tails."))
            sign = ifelse(p == :eq || B[i] >= zero(eltype(B)), one(eltype(B)),
                          -one(eltype(B)))
            Ai = zeros(eltype(X), 1, size(X, 1))
            Ai[1, idx] .= sign
            add_ep_constraint!(epc, Ai, [sign * alpha], p)
        end
    end
    return nothing
end
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, opt::OptimEntropyPooling)

Solve the dual of the entropy pooling problem using Optim.jl.

`entropy_pooling` computes posterior probabilities by minimising the Kullback-Leibler divergence of the posterior weights from the prior ones, subject to moment and view constraints. The optimisation is performed using [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), supporting box constraints and slack variables for relaxed equality constraints. This method is used internally by [`MeucciEntropyPoolingPrior`](@ref) and [`EntropyPoolingPrior`](@ref) when the optimiser is an [`OptimEntropyPooling`](@ref).

# Mathematical definition

The dual of the entropy pooling problem is solved for Lagrange multipliers ``\\boldsymbol{x}``. The dual objective is:

```math
\\begin{align}
\\underset{\\boldsymbol{x}}{\\min} &\\; \\boldsymbol{x}^\\intercal \\boldsymbol{b} + \\sum_{t=1}^{T} q_t \\exp\\!\\left(-\\boldsymbol{x}^\\intercal \\mathbf{A}_{\\cdot t} - 1\\right)\\,.
\\end{align}
```

The optimal posterior weights recover as:

```math
\\begin{align}
p_t^* &= q_t \\exp\\!\\left(-\\boldsymbol{x}^{*\\intercal} \\mathbf{A}_{\\cdot t} - 1\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: Lagrange multipliers (dual variables).
  - ``\\boldsymbol{b}``: Right-hand side constraint vector.
  - ``\\mathbf{A}_{\\cdot t}``: ``t``-th column of the constraint matrix ``\\mathbf{A}``.
  - ``q_t``: Prior weight for scenario ``t``.
  - ``p_t^*``: Optimal posterior weight for scenario ``t``.
  - $(math_dict[:T])

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.

  - `opt`: Optim.jl-based entropy pooling optimiser.

      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Evaluate the objective through the exponential of the dual variables.
      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Evaluate the objective in log space.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Details

  - Constructs the constraint matrix and bounds from `epc`.
  - Relaxes fixed equality constraints via slack variables to make the problem more tractable.
  - The two optimisation algorithms minimise the same objective and reach the same posterior. They differ only in the arithmetic that evaluates it.
  - Throws an error if optimisation fails.

# Related

  - [`OptimEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:ExpEntropyPooling})
    T = length(w)
    factor = inv(sqrt(T))
    A = fill(factor, 1, T)
    B = [factor]
    wb = [typemin(eltype(w)) typemax(eltype(w))]
    for (key, val) in epc
        A = vcat(A, val[1])
        B = vcat(B, val[2])
        s = length(val[2])
        wb = if key == :eq || key == :cvar_eq
            vcat(wb, [fill(typemin(eltype(w)), s) fill(typemax(eltype(w)), s)])
        elseif key == :ineq || key == :cvar_ineq
            vcat(wb, [zeros(eltype(w), s) fill(typemax(eltype(w)), s)])
        elseif key == :feq
            vcat(wb, [fill(-opt.sc2, s) fill(opt.sc2, s)])
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
    end
    x0 = fill(factor, size(A, 1))
    G = similar(x0)
    last_x = similar(x0)
    grad = similar(G)
    y = similar(w)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            y .= w .* exp.(-transpose(A) * x .- one(eltype(w)))
            grad .= B - A * y
        end
    end
    function f(x)
        common_op(x)
        return opt.sc1 * (sum(y) + LinearAlgebra.dot(x, B))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    #! Start: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @static if v"2.0.1" <= pkgversion(Optim) < v"2.3.0"
        args = ifelse(isempty(opt.args), (Optim.Fminbox(; mu0 = 1e-5),), opt.args)
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, args...;
                                opt.kwargs...)
    else
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                                opt.kwargs...)
    end
    #! End: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @argcheck(Optim.converged(result),
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    x = Optim.minimizer(result)
    return StatsBase.pweights(w .* exp.(-transpose(A) * x .- one(eltype(w))))
end
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:LogEntropyPooling})
    T = length(w)
    factor = inv(sqrt(T))
    A = fill(factor, 1, T)
    B = [factor]
    wb = [typemin(eltype(w)) typemax(eltype(w))]
    for (key, val) in epc
        A = vcat(A, val[1])
        B = vcat(B, val[2])
        s = length(val[2])
        wb = if key == :eq || key == :cvar_eq
            vcat(wb, [fill(typemin(eltype(w)), s) fill(typemax(eltype(w)), s)])
        elseif key == :ineq
            vcat(wb, [zeros(eltype(w), s) fill(typemax(eltype(w)), s)])
        elseif key == :feq
            vcat(wb, [fill(-opt.sc2, s) fill(opt.sc2, s)])
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
    end
    log_p = log.(w)
    x0 = fill(factor, size(A, 1))
    G = similar(x0)
    last_x = similar(x0)
    grad = similar(G)
    log_x = similar(log_p)
    y = similar(log_p)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            log_x .= log_p - (one(eltype(log_p)) .+ transpose(A) * x)
            y .= exp.(log_x)
            grad .= B - A * y
        end
    end
    function f(x)
        common_op(x)
        return opt.sc1 * (LinearAlgebra.dot(x, grad) - LinearAlgebra.dot(y, log_x - log_p))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    #! Start: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @static if v"2.0.1" <= pkgversion(Optim) < v"2.3.0"
        args = ifelse(isempty(opt.args), (Optim.Fminbox(; mu0 = 1e-5),), opt.args)
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, args...;
                                opt.kwargs...)
    else
        result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                                opt.kwargs...)
    end
    #! End: Optim.jl's Fminbox() initial_mu! with default mu0 is broken. Use this until it's fixed.
    @argcheck(Optim.converged(result),
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    x = Optim.minimizer(result)
    return StatsBase.pweights(exp.(log_p - (one(eltype(log_p)) .+ transpose(A) * x)))
end
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, opt::JuMPEntropyPooling)

Solve the primal of the entropy pooling problem using JuMP.jl.

`entropy_pooling` computes posterior probabilities by minimising the Kullback-Leibler divergence of the posterior weights from the prior ones, subject to moment and view constraints. The optimisation is performed using [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl), supporting relative entropy cones and slack variables for relaxed equality constraints. This method is used internally by [`MeucciEntropyPoolingPrior`](@ref) and [`EntropyPoolingPrior`](@ref) when the optimiser is a [`JuMPEntropyPooling`](@ref).

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.

  - `opt`: JuMP.jl-based entropy pooling optimiser.

      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Write the divergence against the prior probabilities directly.
      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Write the divergence against a unit reference and subtract the prior log-probabilities in the objective.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Details

  - Forwards to the four-argument method with no tail view, which carries the body.
  - Constructs the JuMP model with the chosen divergence representation and the constraints from `epc`.
  - Relaxes fixed equality constraints by adding norm one cone bounded slack variables to make the problem more tractable.
  - Throws an error if optimisation fails.

# Related

  - [`JuMPEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`OptimEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008])
"""
function entropy_pooling(w::VecNum, epc::AbstractDict, opt::JuMPEntropyPooling)
    return entropy_pooling(w, epc, AbstractEntropyPoolingTailView[], opt)
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:cvar}, alpha::Number)

Compute the Conditional Value-at-Risk (CVaR) for asset `i` from a prior result.

`get_pr_value` extracts the CVaR at confidence level `alpha` for the asset indexed by `i` from the prior result object `pr`. This method assumes the prior result contains the necessary asset return information (mean, covariance, or samples) to compute CVaR, typically under a normality assumption.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:cvar}`: Dispatch tag for CVaR computation.
  - `alpha`: Confidence level.

# Returns

  - `cvar::Number`: Conditional Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:cvar}, alpha::Number)
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ConditionalValueatRisk(; alpha = alpha)(view(pr.X, :, i))
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:sigma}, args...)

Extract the variance for asset `i` from a prior result.

`get_pr_value` returns the variance (diagonal element of the covariance matrix) for the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:sigma}`: Dispatch tag for variance extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `sigma::Number`: Variance for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:sigma}, args...)
    return LinearAlgebra.diag(pr.sigma)[i]
end
"""
    ep_sigma_views!(sigma_views::LinearConstraintEstimator, epc::AbstractDict,
                    pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)

Parse and add variance (sigma) view constraints to the entropy pooling constraint dictionary.

`ep_sigma_views!` parses variance view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding quadratic constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean to be fixed to the prior value, ensuring that variance views do not inadvertently alter the mean.

# Arguments

  - `sigma_views`: Variance view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior values.
  - Converts parsed views to quadratic constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean fixed due to variance constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_sigma_views!(sigma_views::LinearConstraintEstimator, epc::AbstractDict,
                         pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    X = pr.X
    sigma_views = parse_equation(sigma_views.val; datatype = eltype(X))
    sigma_views = replace_group_by_assets(sigma_views, sets, false, true, false)
    sigma_views = replace_prior_views(sigma_views, pr, sets, :sigma; strict = strict)
    lcs = get_linear_constraints(sigma_views, sets; datatype = eltype(X), strict = strict)
    tmp = transpose((X .- transpose(pr.mu)) .^ 2)
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
"""
    fix_sigma!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
               pr::AbstractPriorResult)

Add constraints to fix the variance of specified assets in entropy pooling.

`fix_sigma!` identifies assets in `to_fix` that are not yet fixed (i.e., not present in `fixed`), and adds constraints to the entropy pooling constraint dictionary `epc` to fix their variance to the prior value. This ensures that higher moment views (e.g., skewness, kurtosis, correlation) do not inadvertently alter the variance of these assets. The function updates `fixed` in-place to reflect the newly fixed assets.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `fixed`: Boolean vector indicating which assets have their variance fixed.
  - `to_fix`: Boolean vector indicating which assets should have their variance fixed.
  - `pr`: Prior result containing asset return information.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in-place.

# Details

  - Adds a fixed equality constraint (`:feq`) for each asset in `to_fix` that is not yet fixed.
  - Uses the prior variance values from `LinearAlgebra.diag(pr.sigma)` for the constraint right-hand side.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function fix_sigma!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
                    pr::AbstractPriorResult)
    sigma = LinearAlgebra.diag(pr.sigma)
    fix = to_fix .& .!fixed
    if any(fix)
        add_ep_constraint!(epc, transpose(view(pr.X, :, fix) .- transpose(pr.mu[fix])) .^ 2,
                           sigma[fix], :feq)
        fixed .= fixed .| fix
    end
    return nothing
end
"""
    replace_coprior_views(res::ParsingResult, pr::AbstractPriorResult, sets::UniverseSets, key::Symbol;
                          strict::Bool = false)

Replace correlation prior references in view parsing results with their corresponding prior values.

`replace_coprior_views` scans a parsed correlation view constraint (`ParsingResult`) for references to prior values (e.g., `prior(A, B)`), and replaces them with the actual prior correlation value from the provided prior result object. This ensures that prior-based terms in correlation view constraints are treated as constants and not as variables in the optimisation. If an asset referenced in a prior is not found in the asset set, a warning is issued (or an error if `strict=true`). If all variables in the view are prior references, an error is thrown.

# Arguments

  - `res`: Parsed correlation view constraint containing variables and coefficients.
  - `pr`: Prior result object containing prior correlation values.
  - `sets`: Asset set mapping asset names to indices.
  - `key`: Symbol representing whether it's a correlation or covariance view.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `res::RhoParsingResult`: Updated parsing result with prior references replaced by their values and correlation indices.

# Details

  - Prior references are matched using the pattern `prior(<asset1>, <asset2>)`.
  - The right-hand side of the constraint is adjusted by subtracting the prior value times its coefficient.
  - A `prior(gA, gB)` reference over a pair of groups resolves to one value per spanned asset pair, so the right-hand side becomes a vector of that length. The view emits one constraint row per pair, and each row takes its own entry.
  - Variables corresponding to prior references are removed from the constraint.
  - Throws an error if no non-prior variables remain.
  - Returns a `RhoParsingResult` containing the updated variables, coefficients, operator, right-hand side, equation string, and correlation indices.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`UniverseSets`](@ref)
  - [`prior`](@ref)
"""
function replace_coprior_views(res::ParsingResult, pr::AbstractPriorResult,
                               sets::UniverseSets, key::Symbol; strict::Bool = false)
    prior_pattern = r"prior\(([^()]*)\)"
    prior_corr_pattern = r"prior\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    nx = sets.dict[sets.xkey]
    variables, coeffs = res.vars, res.coef
    jk_idx = Vector{Union{Tuple{Int, Int}, Tuple{Vector{Int}, Vector{Int}}}}(undef, 0)
    idx_rm = Vector{Int}(undef, 0)
    rhs = res.rhs
    non_prior = false
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            non_prior = true
            n = match(corr_pattern, v)
            @argcheck(!isnothing(n),
                      ArgumentError("Correlation view $(v) must be of the form `(a, b)`."))
            asset1 = n.captures[1]
            asset2 = n.captures[2]
            if startswith(asset1, "[") && endswith(asset1, "]")
                asset1 = split(n.captures[1][2:(end - 1)], ", ")
                asset2 = split(n.captures[2][2:(end - 1)], ", ")
                j = [findfirst(x -> x == a1, nx) for a1 in asset1]
                k = [findfirst(x -> x == a2, nx) for a2 in asset2]
            else
                j = findfirst(x -> x == asset1, nx)
                k = findfirst(x -> x == asset2, nx)
                if isnothing(j)
                    msg = unknown_variable_msg(asset1, nx, sets.xkey)
                    strict_diagnostic(msg, strict)
                end
                if isnothing(k)
                    msg = unknown_variable_msg(asset2, nx, sets.xkey)
                    strict_diagnostic(msg, strict)
                end
                if isnothing(j) || isnothing(k)
                    push!(idx_rm, i)
                    continue
                end
            end
            push!(jk_idx, (j, k))
            continue
        end
        n = match(prior_corr_pattern, v)
        @argcheck(!isnothing(n),
                  ArgumentError("Correlation prior view $(v) must be of the form `prior(a, b)`."))
        asset1 = n.captures[1]
        asset2 = n.captures[2]
        if startswith(asset1, "[") && endswith(asset1, "]")
            asset1 = split(n.captures[1][2:(end - 1)], ", ")
            asset2 = split(n.captures[2][2:(end - 1)], ", ")
            j = [findfirst(x -> x == a1, nx) for a1 in asset1]
            k = [findfirst(x -> x == a2, nx) for a2 in asset2]
        else
            j = findfirst(x -> x == asset1, nx)
            k = findfirst(x -> x == asset2, nx)
            if isnothing(j)
                msg = unknown_variable_msg(asset1, nx, sets.xkey)
                strict_diagnostic(msg, strict)
            end
            if isnothing(k)
                msg = unknown_variable_msg(asset2, nx, sets.xkey)
                strict_diagnostic(msg, strict)
            end
            if isnothing(j) || isnothing(k)
                push!(idx_rm, i)
                continue
            end
        end
        # A group pair view emits one row per spanned pair, so `get_pr_value` answers with one
        # value per pair. Broadcast, so a scalar `rhs` widens to that vector.
        rhs = rhs ⊖ get_pr_value(pr, j, k, Val(key)) ⊙ c
        push!(idx_rm, i)
    end
    if isempty(idx_rm)
        return RhoParsingResult(res.vars, res.coef, res.op, res.rhs, res.eqn, jk_idx)
    end
    @argcheck(non_prior,
              ArgumentError("Priors in views are replaced by their prior value, thus they are essentially part of the constant of the view, so you need a non-prior view to serve as the variable."))
    idx = setdiff(1:length(variables), idx_rm)
    variables_new = variables[idx]
    coeffs_new = coeffs[idx]
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "))
    return RhoParsingResult(variables_new, coeffs_new, res.op, rhs,
                            "$(eqn) $(res.op) $(rhs)", jk_idx)
end
"""
    replace_coprior_views(res::VecPR, args...; kwargs...)

Broadcast prior reference replacement across multiple view constraints.

`replace_coprior_views` applies [`replace_coprior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values.

# Arguments

  - `res:`: Vector of parsed view constraints.
  - `args...`: Additional positional arguments forwarded to [`replace_coprior_views`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`replace_coprior_views`](@ref).

# Returns

  - `res::Vector{<:ParsingResult}`: Vector of updated parsing results with prior references replaced by their values.

# Related

  - [`ParsingResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`UniverseSets`](@ref)
"""
function replace_coprior_views(res::VecPR, args...; kwargs...)
    return replace_coprior_views.(res, args...; kwargs...)
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, ::Val{:rho}, args...)
    get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, ::Val{:cov}, args...)

Extract the prior correlation or covariance between assets `i` and `j` from a prior result.

`get_pr_value` returns the entry of the prior correlation or covariance matrix that the dispatch tag names. These methods are used internally to replace `prior(a, b)` references in correlation and covariance view constraints.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the first asset.
  - `j`: Index of the second asset.
  - `::Val{:rho}`: Dispatch tag for correlation extraction.
  - `::Val{:cov}`: Dispatch tag for covariance extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `val::Number`: Correlation coefficient or covariance between assets `i` and `j`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, ::Val{:rho}, args...)
    return StatsBase.cov2cor(pr.sigma)[i, j]
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, ::Val{:cov}, args...)
    return pr.sigma[i, j]
end
"""
    get_pr_value(pr::AbstractPriorResult, i::VecInt, j::VecInt, ::Val{:rho}, args...)
    get_pr_value(pr::AbstractPriorResult, i::VecInt, j::VecInt, ::Val{:cov}, args...)

Extract the prior correlations or covariances of the asset pairs that two groups span.

`get_pr_value` returns one value per spanned pair, in the order of `zip(i, j)`. A view over a pair of groups emits one constraint row per spanned pair, so a `prior(gA, gB)` reference inside such a view must give the row that pair's own prior value. These methods are used internally to replace `prior(gA, gB)` references in correlation and covariance view constraints.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Vector of indices for the first asset group.
  - `j`: Vector of indices for the second asset group.
  - `::Val{:rho}`: Dispatch tag for correlation extraction.
  - `::Val{:cov}`: Dispatch tag for covariance extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `val::Vector{<:Number}`: Correlation or covariance of each spanned pair, one entry per element of `zip(i, j)`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
  - [`RhoParsingResult`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::VecInt, j::VecInt, ::Val{:rho}, args...)
    rho = StatsBase.cov2cor(pr.sigma)
    return [rho[a, b] for (a, b) in zip(i, j)]
end
function get_pr_value(pr::AbstractPriorResult, i::VecInt, j::VecInt, ::Val{:cov}, args...)
    return [pr.sigma[a, b] for (a, b) in zip(i, j)]
end
"""
    ep_cov_views!(cov_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)

Parse and add covariance view constraints to the entropy pooling constraint dictionary.

`ep_cov_views!` parses covariance view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that covariance views do not inadvertently alter lower moments.

# Arguments

  - `cov_views`: Covariance view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior covariance values.
  - A view over a pair of groups emits one constraint row per spanned asset pair, and each row takes that pair's own right-hand side.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to covariance constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_coprior_views`](@ref)
  - [`get_pr_value`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`comparison_sign_ineq_flag`](@ref)
"""
function ep_cov_views!(cov_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    X = pr.X
    cov_views = parse_equation(cov_views.val; datatype = eltype(X))
    cov_views = replace_group_by_assets(cov_views, sets, false, true, true)
    cov_views = replace_coprior_views(cov_views, pr, sets, :cov; strict = strict)
    to_fix = falses(size(X, 2))
    for cov_view in cov_views
        @argcheck(length(cov_view.vars) == 1,
                  "Cannot mix multiple covariance pairs in a single view `$(cov_view.eqn)`.")
        d, flag = comparison_sign_ineq_flag(cov_view.op)
        i, j = cov_view.ij[1]
        Ai = d * cov_view.coef[1] * view(X, :, i) .* view(X, :, j)
        Bi = d * (cov_view.coef[1] * (pr.mu[i] ⊙ pr.mu[j]) ⊕ cov_view.rhs)
        if !isa(i, AbstractVector)
            Bi = [Bi]
        end
        add_ep_constraint!(epc, transpose(Ai), Bi, ifelse(flag, :ineq, :eq))
        to_fix[union(i, j)] .= true
    end
    return to_fix
end
"""
    ep_rho_views!(rho_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)

Parse and add correlation view constraints to the entropy pooling constraint dictionary.

`ep_rho_views!` parses correlation view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that correlation views do not inadvertently alter lower moments.

# Arguments

  - `rho_views`: Correlation view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior correlation values.
  - A view over a pair of groups emits one constraint row per spanned asset pair, and each row takes that pair's own right-hand side.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to correlation constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_coprior_views`](@ref)
  - [`get_pr_value`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`comparison_sign_ineq_flag`](@ref)
"""
function ep_rho_views!(rho_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    X = pr.X
    rho_views = parse_equation(rho_views.val; datatype = eltype(X))
    rho_views = replace_group_by_assets(rho_views, sets, false, true, true)
    rho_views = replace_coprior_views(rho_views, pr, sets, :rho; strict = strict)
    to_fix = falses(size(X, 2))
    sigma = LinearAlgebra.diag(pr.sigma)
    for rho_view in rho_views
        @argcheck(length(rho_view.vars) == 1,
                  "Cannot mix multiple correlation pairs in a single view `$(rho_view.eqn)`.")
        @argcheck(all(x -> -one(eltype(X)) <= x <= one(eltype(X)), rho_view.rhs),
                  "Correlation prior rho_view `$(rho_view.eqn)` must be in [-1, 1].")
        d, flag = comparison_sign_ineq_flag(rho_view.op)
        i, j = rho_view.ij[1]
        sigma_ij = if !isa(i, AbstractVector)
            sqrt(sigma[i] * sigma[j])
        else
            sqrt.(sigma[i] .* sigma[j])
        end
        Ai = d * rho_view.coef[1] * view(X, :, i) .* view(X, :, j)
        Bi = d * (rho_view.coef[1] * (pr.mu[i] ⊙ pr.mu[j]) ⊕ rho_view.rhs ⊙ sigma_ij)
        if !isa(i, AbstractVector)
            Bi = [Bi]
        end
        add_ep_constraint!(epc, transpose(Ai), Bi, ifelse(flag, :ineq, :eq))
        to_fix[union(i, j)] .= true
    end
    return to_fix
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:skew}, args...)

Extract the skewness for asset `i` from a prior result.

`get_pr_value` returns the skewness of the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for higher moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:skew}`: Dispatch tag for skewness extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `skew::Number`: Skewness for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:skew}, args...)
    #! Think about how to include pr.w
    return Skewness()(view(pr.X, :, i))
end
"""
    ep_sk_views!(skew_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)

Parse and add skewness view constraints to the entropy pooling constraint dictionary.

`ep_sk_views!` parses skewness view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that skewness views do not inadvertently alter lower moments.

# Arguments

  - `skew_views`: Skewness view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior skewness values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to skewness constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_sk_views!(skew_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    X = pr.X
    skew_views = parse_equation(skew_views.val; datatype = eltype(X))
    skew_views = replace_group_by_assets(skew_views, sets, false, true, false)
    skew_views = replace_prior_views(skew_views, pr, sets, :skew; strict = strict)
    lcs = get_linear_constraints(skew_views, sets; datatype = eltype(X), strict = strict)
    sigma = LinearAlgebra.diag(pr.sigma)
    tmp = transpose((X .^ 3 .- transpose(pr.mu) .^ 3 .- 3 * transpose(pr.mu .* sigma)) ./
                    transpose(sigma .* sqrt.(sigma)))
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:kurtosis}, args...)

Extract the kurtosis for asset `i` from a prior result.

`get_pr_value` returns the kurtosis of the asset indexed by `i` from the prior result object `pr`. This method is used internally to replace prior references in view constraints and for higher moment extraction in entropy pooling and other prior-based routines.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:kurtosis}`: Dispatch tag for kurtosis extraction.
  - `args...`: Additional arguments (ignored).

# Returns

  - `kurtosis::Number`: Kurtosis for asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:kurtosis}, args...)
    #! Think about how to include pr.w
    return HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))(view(pr.X,
                                                                                             :,
                                                                                             i))
end
"""
    ep_kt_views!(kurtosis_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)

Parse and add kurtosis view constraints to the entropy pooling constraint dictionary.

`ep_kt_views!` parses kurtosis view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. This method returns a boolean vector indicating which assets require their mean and variance to be fixed to the prior value, ensuring that kurtosis views do not inadvertently alter lower moments.

# Arguments

  - `kurtosis_views`: Kurtosis view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Details

  - Parses view equations and replaces groupings by assets.
  - Replaces prior references in views with their actual prior kurtosis values.
  - Converts parsed views to linear constraints and adds them to `epc`.
  - Returns a boolean vector for assets that need their mean and variance fixed due to kurtosis constraints.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_kt_views!(kurtosis_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    X = pr.X
    kurtosis_views = parse_equation(kurtosis_views.val; datatype = eltype(X))
    kurtosis_views = replace_group_by_assets(kurtosis_views, sets, false, true, false)
    kurtosis_views = replace_prior_views(kurtosis_views, pr, sets, :kurtosis;
                                         strict = strict)
    lcs = get_linear_constraints(kurtosis_views, sets; datatype = eltype(X),
                                 strict = strict)
    X_sq = X .^ 2
    mu_sq = pr.mu .^ 2
    tmp = transpose((X_sq .* X_sq .- 4 * transpose(pr.mu) .* X_sq .* X .+
                     6 * transpose(mu_sq) .* X_sq .- 3 * transpose(mu_sq .* mu_sq)) ./
                    transpose(LinearAlgebra.diag(pr.sigma)) .^ 2)
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
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
    return StatsBase.pweights(JuMP.value.(x))
end
function entropy_pooling(w::VecNum, epc::AbstractDict, tvs::VecEPTV,
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
    return StatsBase.pweights(JuMP.value.(x))
end
"""
    ep_jump_views!(model::JuMP.Model, x, obj_expr, epc::AbstractDict, tvs::VecEPTV,
                   sc1::Number, sc2::Number, so::Number)

Add every view constraint of an entropy pooling problem to a JuMP model.

`ep_jump_views!` is the shared body of the two [`JuMPEntropyPooling`](@ref) formulations: they differ only in how they represent the divergence, and agree on every view. It adds the linear rows of `epc`, relaxes the fixed equalities with a norm one cone bounded slack, and hands each tail view to [`add_ep_tail_view!`](@ref).

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

export RhoParsingResult, LogEntropyPooling, ExpEntropyPooling, H0_EntropyPooling,
       H1_EntropyPooling, H2_EntropyPooling, JuMPEntropyPooling, OptimEntropyPooling,
       ConditionalValueatRiskEntropyPooling, ValueatRiskView, ConditionalValueatRiskView,
       EntropicValueatRiskView, LinearConditionalValueatRiskView,
       IntegerConditionalValueatRiskView, ConicEntropicValueatRiskView,
       GridEntropicValueatRiskView, RelativisticValueatRiskView,
       ConicRelativisticValueatRiskView, GridRelativisticValueatRiskView
