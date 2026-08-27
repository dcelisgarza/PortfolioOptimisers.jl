"""
$(DocStringExtensions.TYPEDEF)

Carries a parsed correlation or covariance view together with the asset pairs it names.

It extends [`ParsingResult`](@ref) with an `ij` field, which holds one index pair per term of the view, so a downstream routine can place the view in the covariance matrix without parsing the equation again. [`replace_coprior_views`](@ref) produces it from a view of the form `"(A, B) == 0.5"`, and the entropy pooling and Black-Litterman routines that read pair views consume it.

A view over a pair of groups spans one asset pair per element of its `ij` entry, and emits one constraint row per pair. Its `rhs` is therefore a vector of the same length, one right-hand side per row. A view over a single asset pair keeps a scalar `rhs`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RhoParsingResult(
        vars::VecStr,
        coef::VecNum,
        op::AbstractString,
        rhs::Union{<:Number, <:VecNum},
        eqn::AbstractString,
        ij::AbstractVector{<:Union{<:Tuple{<:Integer, <:Integer},
                                   <:Tuple{<:VecInt, <:VecInt}}}
    ) -> RhoParsingResult

Positional arguments correspond to the struct's fields. There is no keyword constructor, because [`replace_coprior_views`](@ref) is the producer of this type.

## Validation

  - `length(vars) == length(coef)`.
  - If `rhs` is a vector, `!isempty(ij)` and every entry of `ij` is a group pair whose first half holds `length(rhs)` indices.

# Examples

```jldoctest
julia> PortfolioOptimisers.RhoParsingResult([\"(A, B)\"], [1.0], \"==\", 0.5, \"1.0*(A, B) == 0.5\",
                                            [(1, 2)])
RhoParsingResult
  vars ┼ Vector{String}: [\"(A, B)\"]
  coef ┼ Vector{Float64}: [1.0]
    op ┼ String: \"==\"
   rhs ┼ Float64: 0.5
   eqn ┼ String: \"1.0*(A, B) == 0.5\"
    ij ┴ Vector{Tuple{Int64, Int64}}: [(1, 2)]
```

# Related

  - [`AbstractParsingResult`](@ref)
  - [`ParsingResult`](@ref)
  - [`replace_coprior_views`](@ref): the producer of this type.
  - [`replace_prior_views`](@ref)
  - [`ep_cov_views!`](@ref): reads `ij` to place a covariance view.
  - [`ep_rho_views!`](@ref): reads `ij` to place a correlation view.
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

This route does not raise on an infeasible view set. [`entropy_pooling`](@ref) states the mechanism and the three signs that name that answer.

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
"""
    add_ep_constraint!(epc::AbstractDict, lhs::MatNum, rhs::VecNum, key::Symbol)

Add an entropy pooling view constraint to the constraint dictionary.

`add_ep_constraint!` normalises and adds a constraint to the entropy pooling constraint dictionary `epc`. If a constraint with the same key already exists, it concatenates the new constraint to the existing one. This function is used internally to build the set of linear constraints for entropy pooling optimisation.

Every view that is linear in the posterior probabilities reaches `epc` as a block ``(\\mathbf{A},\\, \\boldsymbol{B})`` of the system ``\\mathbf{A} \\boldsymbol{p} = \\boldsymbol{B}`` or ``\\mathbf{A} \\boldsymbol{p} \\leq \\boldsymbol{B}``. The key names the sense of the block, and the optimiser reads the block back by that key.

# Mathematical definition

The block is divided by the Frobenius norm of its left-hand side, which leaves the row it states unchanged:

```math
\\begin{align}
\\tilde{\\mathbf{A}} &= \\dfrac{\\mathbf{A}}{\\lVert \\mathbf{A} \\rVert_{F}}\\,, \\\\
\\tilde{\\boldsymbol{B}} &= \\dfrac{\\boldsymbol{B}}{\\lVert \\mathbf{A} \\rVert_{F}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:ep_post_probs])
  - ``\\lVert \\mathbf{A} \\rVert_{F}``: Frobenius norm of the left-hand side block.

# Algorithm

 1. Read the Frobenius norm of `lhs` into `sc`.
 2. Divide `lhs` and `rhs` by `sc`, giving the normalised block.
 3. Store the pair under `key` when `epc` holds no such key.
 4. Otherwise stack the normalised `lhs` under the block already held, and append `rhs` to the block's right-hand side.

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

`replace_prior_views` scans a parsed view constraint [`ParsingResult`](@ref) for references to prior values (e.g., `prior(A)`), and replaces them with the actual prior value from the provided prior result object. This ensures that prior-based terms in view constraints are treated as constants and not as variables in the optimisation.

# Mathematical definition

A parsed view is the row ``\\sum_{k} c_{k} v_{k} \\lessgtr b``, and a term whose variable is `prior(a)` carries the constant ``\\pi_{a}`` rather than an unknown. Moving every such term to the right-hand side gives an equivalent row over the remaining terms:

```math
\\sum_{k \\notin \\mathcal{P}} c_{k} v_{k} \\lessgtr b - \\sum_{k \\in \\mathcal{P}} c_{k} \\pi_{a_{k}}\\,.
```

Where:

  - ``c_{k}``, ``v_{k}``: Coefficient and variable of the ``k``-th term of the view.
  - ``b``: Right-hand side of the view.
  - ``\\mathcal{P}``: Terms whose variable is a `prior(...)` reference.
  - ``\\pi_{a}``: Prior value of the statistic `key` for asset ``a``, read by [`get_pr_value`](@ref) at the level `alpha` and the further parameters `params...`.

# Algorithm

 1. Match the pattern `prior(<asset>)` against the variable of each term in turn.
 2. When a term does not match, record that the view keeps a variable of its own, and take the next term.
 3. Find the named asset in the universe. When it is absent, report it through `strict_diagnostic`, record the term for removal, and take the next term.
 4. Subtract [`get_pr_value`](@ref) times the term's coefficient from `rhs`, and record the term for removal.
 5. Return `res` unchanged when step 3 and step 4 recorded no term.
 6. Drop the recorded terms from `vars` and `coef`, rebuild the equation string, and return a [`ParsingResult`](@ref) that carries the adjusted `rhs`.

# Arguments

  - `res`: Parsed view constraint containing variables and coefficients.

  - `pr`: Prior result object containing prior values.

  - `sets`: Asset set mapping asset names to indices.

  - `key`: Moment type key (`:mu`, `:var`, `:cvar`, etc.).

  - `alpha`: Optional confidence level for VaR/CVaR views.

  - `params...`: Further parameters of the statistic, forwarded to [`get_pr_value`](@ref). A relativistic value-at-risk view passes its deformation parameter here.

  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Validation

  - An asset a `prior(...)` reference names that the universe does not hold raises an `ArgumentError` when `strict` is `true`, and warns otherwise. The term is dropped either way.
  - At least one term of the view must keep a variable of its own. A view whose every term is a `prior(...)` reference is a statement about constants alone, and raises an `ArgumentError`.

# Returns

  - `res::ParsingResult`: Updated parsing result with prior references replaced by their values.

# Related

  - [`ParsingResult`](@ref)
  - [`get_pr_value`](@ref): reads the prior value each reference is replaced by.
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

`replace_prior_views` applies [`replace_prior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values. [`parse_equation`](@ref) answers a group of view equations with a vector of results, so this is the shape every caller in this file meets.

# Algorithm

 1. Broadcast the single-view method over `res`, forwarding `args...` and `kwargs...` to each call.
 2. Return the vector of the results, one per element of `res`, in the order of `res`.

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

Read the prior **mean** of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:mu)` names, the `i`-th entry of `pr.mu`. It is used internally by [`replace_prior_views`](@ref) and by the `ep_*_views!` verbs.

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

Do nothing when a problem states no **mean** view.

`ep_mu_views!` is the verb that turns a group of mean views into rows of the entropy pooling constraint dictionary. This method is the absent-view branch: it registers no row, so a higher-level routine can call the verb without special-casing `mu_views = nothing`.

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

Add the **mean** views of a group to the entropy pooling constraint dictionary.

`ep_mu_views!` parses mean view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. The statistic is the mean of the posterior distribution, which is linear in the posterior probabilities, so the view needs no auxiliary variable and no moment is fixed on its account.

# Mathematical definition

The posterior mean of an asset is the probability weighted average of its returns, and a view states a linear combination of such means:

```math
\\begin{align}
\\mathrm{E}_{\\boldsymbol{p}}[x_{i}] &= \\sum_{t=1}^{T} p_{t} x_{t,\\,i}\\,, \\\\
\\sum_{i=1}^{N} a_{k,\\,i} \\mathrm{E}_{\\boldsymbol{p}}[x_{i}] &\\lessgtr B_{k}\\,, \\quad \\forall\\, k = 1,\\ldots,K\\,.
\\end{align}
```

The left-hand side is linear in ``\\boldsymbol{p}``, so the ``K`` views are the block ``\\left(\\mathbf{A} \\mathbf{X}^{\\intercal},\\, \\boldsymbol{B}\\right)``.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:A])
  - $(math_dict[:B])
  - ``a_{k,\\,i}``: Coefficient asset ``i`` takes in view ``k``, the ``(k, i)`` entry of ``\\mathbf{A}``.
  - ``K``: Number of views the group states.
  - ``\\mathbf{X}``: ``T \\times N`` returns matrix of the prior.

# Algorithm

 1. Parse the view equations of `mu_views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior mean, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`.
 5. For each block present, add `A * transpose(X)` against `B` under that key with [`add_ep_constraint!`](@ref).

# Arguments

  - `mu_views`: Mean view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior mean a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref): holds a mean at the prior when a higher moment view would move it.
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

Hold the **mean** of the named assets at the prior value.

`fix_mu!` identifies assets in `to_fix` that are not yet fixed (i.e., not present in `fixed`), and adds constraints to the entropy pooling constraint dictionary `epc` to fix their mean to the prior value. This ensures that higher moment views (e.g., variance, skewness, kurtosis, correlation) do not inadvertently alter the mean of these assets. The function updates `fixed` in-place to reflect the newly fixed assets.

The rows go in under the `:feq` key, which the optimiser relaxes with a penalised slack rather than enforcing exactly. A fixing row is a wish, not a view: it competes with the views that were asked for, and it yields where the two cannot both hold.

# Mathematical definition

The posterior mean of every named asset is held at the prior mean:

```math
\\sum_{t=1}^{T} p_{t} x_{t,\\,i} = \\mu_{i}\\,, \\quad \\forall\\, i \\in \\mathcal{F}\\,.
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:T])
  - ``\\mathcal{F}``: Assets named by `to_fix` that `fixed` does not already hold.

# Algorithm

 1. Read the assets that `to_fix` names and `fixed` does not already hold into `fix`.
 2. Return when `fix` names no asset.
 3. Add one `:feq` row per named asset, `transpose(view(pr.X, :, fix))` against `pr.mu[fix]`, with [`add_ep_constraint!`](@ref).
 4. Mark the named assets in `fixed`, so a later call adds no second row for them.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `fixed`: Boolean vector indicating which assets have their mean fixed.
  - `to_fix`: Boolean vector indicating which assets should have their mean fixed.
  - `pr`: Prior result containing asset return information.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in-place.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`ep_mu_views!`](@ref)
  - [`fix_sigma!`](@ref): the same rule, one moment higher.
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

Read the prior **value at risk** of asset `i` at the level `alpha`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:var)` names, by applying [`ValueatRisk`](@ref) to the `i`-th column of `pr.X`. That is the empirical ``\\alpha``-quantile of the loss series, and it ignores `pr.w`: the counterpart in [`ep_var_views!`](@ref) writes an unweighted tail mass, so a weighted quantile here would state a different row.

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

A group of **value at risk** views, with the significance level they are read under.

Unlike a conditional or entropic value at risk view, a value at risk view is linear in the posterior probabilities: it reduces to rows of the constraint set through [`add_ep_constraint!`](@ref), so it needs no auxiliary variable, admits no choice of formulation, and reaches [`OptimEntropyPooling`](@ref) as readily as [`JuMPEntropyPooling`](@ref). That is why this estimator carries a level and nothing else.

The views this estimator holds accept `==` and `>=` alone, one asset per view, with a unit coefficient and a non-negative target. A `prior(...)` reference inside `views` is replaced by the prior value at risk at this group's `alpha`, so a view stated against the prior moves with the level.

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

It accepts the operators `>=` and `==`, one asset per view, with a positive coefficient. An equality view needs a target greater than or equal to the prior CVaR of the asset. Below the prior CVaR the constraint is slack at the prior, so the entropy minimiser leaves the prior untouched and the view is not met. Use [`IntegerConditionalValueatRiskView`](@ref) there.

# Mathematical definition

The view ``\\mathrm{CVaR}_{\\alpha}(X) \\geq \\bar{c}`` on the asset the view names is written as:

```math
\\begin{align}
&\\nu_{j} \\geq 0\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\nu_{j} \\leq \\dfrac{w_{j}}{\\alpha}\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cvar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:cvar_target])
  - $(math_dict[:ep_tail_nu])

The box and the simplex describe every reweighting of the sample that no observation gives more than ``1/\\alpha`` times its posterior probability, and the largest loss such a reweighting attains is the CVaR. So the constraint set is feasible if and only if ``\\mathrm{CVaR}_{\\alpha}(X) \\geq \\bar{c}``, and a lower-bound view is exact.

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

The conditional value at risk is the mean of the ``\\alpha`` heaviest tail mass of the posterior, and this formulation states it over the ``\\bar{s}`` largest losses alone:

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

Where:

  - $(math_dict[:cvar_stat])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - ``x_{[1]} \\leq x_{[2]} \\leq \\ldots \\leq x_{[\\bar{s}]}``: The ``\\bar{s}`` largest losses of the asset, sorted ascending, so the largest loss is last.
  - ``w_{[j]}``: Posterior probability of the observation in position ``j``.
  - ``\\bar{s}``: Number of largest losses the formulation reads.
  - ``\\boldsymbol{y}``: ``\\bar{s} \\times 1`` binary vector that marks the observations entering the tail.
  - ``\\boldsymbol{q}``: ``\\bar{s} \\times 1`` auxiliary vector that carries the product ``q_{j} = w_{[j]} y_{j}``.

The monotonicity constraint makes the marked set a suffix of the ascending order, which is what makes the expression the CVaR rather than the mean of an arbitrary subset of probability ``\\alpha``.

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

It accepts the operators `>=` and `==`, one asset per view, with a positive coefficient. An equality view needs a target greater than or equal to the prior EVaR of the asset. Use [`GridEntropicValueatRiskView`](@ref) below it.

# Mathematical definition

The view ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}`` on the asset the view names is written as:

```math
\\begin{align}
&0 \\leq \\nu_{j} \\leq 1\\,, &\\forall\\, j = 1,\\ldots,T\\\\
&\\sum_{j=1}^{T} \\nu_{j} \\ln\\left(\\dfrac{\\nu_{j}}{w_{j}}\\right) \\leq \\ln\\left(\\dfrac{1}{\\alpha}\\right)\\\\
&\\sum_{j=1}^{T} \\nu_{j} = 1\\\\
&\\sum_{j=1}^{T} \\nu_{j} x_{j} \\geq \\bar{e}\\,.
\\end{align}
```

Where:

  - $(math_dict[:evar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:evar_target])
  - $(math_dict[:ep_tail_nu])

The relative entropy budget is the dual description of EVaR, so the constraint set is feasible if and only if ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}``.

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

Rows reach the model divided by their largest coefficient, so the coefficients sit in `(0, 1]`, the posterior sums to one, and the left-hand side is bounded by one whatever the data. The default `M` clears that bound by an order of magnitude.

The answer is approximate in both directions. A lower-bound view holds at the grid points and may fall short between them, and an upper-bound view holds at one grid point and may be conservative. Widen `pct` or raise `K` when the posterior value misses the target, and prefer [`ConicEntropicValueatRiskView`](@ref) whenever the view admits it.

It accepts `==`, `>=` and `<=`, one asset per view. The view is normalised so its coefficient is one, which flips the operator when the coefficient is negative, so this formulation restricts neither the operator nor the sign.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The sample EVaR is the value of a scalar minimisation:

```math
\\begin{align}
\\mathrm{EVaR}_{\\alpha}(X) &= \\underset{z > 0}{\\min} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
\\end{align}
```

So ``\\mathrm{EVaR}_{\\alpha}(X) \\geq \\bar{e}`` holds exactly when the objective is at or above ``\\bar{e}`` at *every* ``z``, and ``\\mathrm{EVaR}_{\\alpha}(X) \\leq \\bar{e}`` holds when it is at or below ``\\bar{e}`` at *some* ``z``. The objective is linear in ``\\boldsymbol{w}`` once ``z`` is fixed, which is what makes a grid point a row. On a grid ``\\bar{z}_{1},\\ldots,\\bar{z}_{K}`` that gives, for a lower-bound view:

```math
\\begin{align}
&\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/\\bar{z}_{k})}{\\exp(\\bar{e}/\\bar{z}_{k})} \\geq \\alpha\\,, &\\forall\\, k = 1,\\ldots,K
\\end{align}
```

and for an upper-bound view, with ``\\boldsymbol{y}`` a binary selector and ``M`` a big constant:

```math
\\begin{align}
&\\boldsymbol{1}^{\\intercal} \\boldsymbol{y} = 1\\\\
&\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/\\bar{z}_{k})}{\\exp(\\bar{e}/\\bar{z}_{k})} \\leq \\alpha + M(1 - y_{k})\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{K}\\,.
\\end{align}
```

Where:

  - $(math_dict[:evar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:evar_target])
  - ``z > 0``: Dual variable of the entropic value at risk.
  - ``\\bar{z}_{k}``: Dual variable of the ``k``-th grid point.
  - ``K``: Number of grid points.
  - ``\\boldsymbol{y}``: ``K \\times 1`` binary selector, one entry per grid point.
  - ``M``: Big-M constant.

An equality view carries both blocks.

# Constructors

    GridEntropicValueatRiskView(;
        pct::Number = 0.5,
        K::Integer = 11,
        M::Number = 10,
        iters::Integer = 50,
        tol::Number = 1e-10,
        tilt_iters::Integer = 200
    ) -> GridEntropicValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < pct < 1`.
  - `K >= 1` and `isodd(K)`.
  - `M > 0`.
  - `iters >= 1`.
  - `tol >= 0`.
  - `tilt_iters >= 1`.

# Examples

```jldoctest
julia> GridEntropicValueatRiskView()
GridEntropicValueatRiskView
         pct ┼ Float64: 0.5
           K ┼ Int64: 11
           M ┼ Int64: 10
       iters ┼ Int64: 50
         tol ┼ Float64: 1.0e-10
  tilt_iters ┴ Int64: 200
```

# Related

  - [`AbstractEntropicValueatRiskViewFormulation`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`ep_evar_anchor`](@ref)
  - [`ep_evar_grid`](@ref)

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
    """
    $(field_dict[:ep_grid_iters])
    """
    iters
    """
    $(field_dict[:ep_grid_tol])
    """
    tol
    """
    $(field_dict[:ep_grid_tilt_iters])
    """
    tilt_iters
    function GridEntropicValueatRiskView(pct::Number, K::Integer, M::Number, iters::Integer,
                                         tol::Number, tilt_iters::Integer)
        assert_unit_interval(pct, :pct)
        @argcheck(K >= one(K) && isodd(K), DomainError(K, "K must be odd and >= 1"))
        @argcheck(M > zero(M), DomainError(M, "M must be > 0"))
        @argcheck(iters >= one(iters), DomainError(iters, "iters must be >= 1"))
        @argcheck(tol >= zero(tol), DomainError(tol, "tol must be >= 0"))
        @argcheck(tilt_iters >= one(tilt_iters),
                  DomainError(tilt_iters, "tilt_iters must be >= 1"))
        return new{typeof(pct), typeof(K), typeof(M), typeof(iters), typeof(tol),
                   typeof(tilt_iters)}(pct, K, M, iters, tol, tilt_iters)
    end
end
function GridEntropicValueatRiskView(; pct::Number = 0.5, K::Integer = 11, M::Number = 10,
                                     iters::Integer = 50, tol::Number = 1e-10,
                                     tilt_iters::Integer = 200)::GridEntropicValueatRiskView
    return GridEntropicValueatRiskView(pct, K, M, iters, tol, tilt_iters)
end
"""
$(DocStringExtensions.TYPEDEF)

Power cone formulation of a relativistic value-at-risk view [EPRLVaR](@cite).

`ConicRelativisticValueatRiskView` writes the view through the dual representation of RLVaR. It adds ``3T`` continuous variables and ``2T`` power cones, and it is exact.

It accepts the operators `>=` and `==`, one asset per view, with a positive coefficient. An equality view needs a target greater than or equal to the prior RLVaR of the asset. Use [`GridRelativisticValueatRiskView`](@ref) below it. The solver must handle the power cone alongside the exponential cone the entropy pooling objective needs.

The programme is a demanding solve. A long sample, a small `alpha`, a small `kappa` or several of these views in one model can make a conic solver stop short of a solution. Give `opt` a vector of solver configurations, shorten the sample, or state the view under [`GridRelativisticValueatRiskView`](@ref), whose rows are linear in the posterior probabilities.

# Mathematical definition

The view ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) \\geq \\bar{\\vartheta}`` on the asset the view names is written as:

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

Where:

  - $(math_dict[:rlvar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_target])
  - ``\\boldsymbol{\\nu}``: ``T \\times 1`` vector of weights that attains the RLVaR.
  - ``\\boldsymbol{\\tau}``, ``\\boldsymbol{\\varsigma}``: ``T \\times 1`` vectors that carry the Kaniadakis entropy budget of ``\\boldsymbol{\\nu}``.
  - ``\\mathcal{K}_{\\mathrm{pow}}(p) = \\{(a,b,c) : a^{p} b^{1-p} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}``: Power cone.

The budget is the dual description of RLVaR, so the constraint set is feasible if and only if ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) \\geq \\bar{\\vartheta}``.

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

Rows reach the model divided by their largest coefficient, so the coefficients sit in `(0, 1]`, the posterior sums to one, and the left-hand side is bounded by one whatever the data. The default `M` clears that bound by an order of magnitude.

It accepts `==`, `>=` and `<=`, one asset per view. The view is normalised so its coefficient is one, which flips the operator when the coefficient is negative, so this formulation restricts neither the operator nor the sign.

As `kappa` approaches one the RLVaR approaches the largest loss, and [`ep_rlvar_tail`](@ref) overflows at the dual variable that attains it. The points it overflows at are dropped, and a grid that keeps none of them raises. The centre of the grid is found by an iteration that reads the same tail function, so it too stops converging there and the grid falls back to the prior's dual variable, which lands short of the target. Prefer a smaller `kappa`, or [`ConicRelativisticValueatRiskView`](@ref) where the operator admits it.

# Fields

$(DocStringExtensions.FIELDS)

# Mathematical definition

The sample RLVaR is the value of a two-variable minimisation, in which the pair of power cones of each observation is already minimised out:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) &= \\underset{t,\\, z > 0}{\\min} \\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,,
\\end{align}
```

where ``\\varphi_{\\kappa}(u, z)`` is the smallest ``\\psi + \\theta`` the two power cones of one observation allow, and has the closed form:

```math
\\begin{align}
\\varphi_{\\kappa}(u, z) &= \\dfrac{\\kappa}{1+\\kappa} \\left(\\dfrac{2\\kappa}{(1+\\kappa) z}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma - u}{2}\\right)^{\\frac{1+\\kappa}{\\kappa}} + \\kappa (1-\\kappa)^{\\frac{1-\\kappa}{\\kappa}} \\left(\\dfrac{z}{2\\kappa}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma + u}{2}\\right)^{-\\frac{1-\\kappa}{\\kappa}}\\,,\\\\
\\sigma &= \\sqrt{u^{2} + \\dfrac{(1 - \\kappa^{2}) z^{2}}{\\kappa^{2}}}\\,.
\\end{align}
```

The objective is linear in ``\\boldsymbol{w}`` once ``t`` and ``z`` are fixed, which is what makes a grid point a row. On a grid ``(\\bar{t}_{1}, \\bar{z}_{1}),\\ldots,(\\bar{t}_{K}, \\bar{z}_{K})`` that gives, for a lower-bound view:

```math
\\begin{align}
&T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) \\geq \\bar{\\vartheta} - \\bar{t}_{k} - \\bar{z}_{k} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\,, &\\forall\\, k = 1,\\ldots,K
\\end{align}
```

and for an upper-bound view, with ``\\boldsymbol{y}`` a binary selector and ``M`` a big constant:

```math
\\begin{align}
&\\boldsymbol{1}^{\\intercal} \\boldsymbol{y} = 1\\\\
&T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(\\bar{t}_{k} - x_{j},\\, \\bar{z}_{k}) \\leq \\bar{\\vartheta} - \\bar{t}_{k} - \\bar{z}_{k} \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + M(1 - y_{k})\\,, &\\forall\\, k = 1,\\ldots,K\\\\
&\\boldsymbol{y} \\in \\{0,1\\}^{K}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_stat])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_target])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:rlvar_u])
  - $(math_dict[:rlvar_sigma])
  - $(math_dict[:rlvar_phi])
  - ``\\psi``, ``\\theta``: The two tail variables of one observation, whose smallest sum is ``\\varphi_{\\kappa}``.
  - ``\\bar{t}_{k}``, ``\\bar{z}_{k}``: Shift and dual variable of the ``k``-th grid point.
  - ``K``: Number of grid points.
  - ``\\boldsymbol{y}``: ``K \\times 1`` binary selector, one entry per grid point.
  - ``M``: Big-M constant.

An equality view carries both blocks. Every grid point is a feasible point of the primal programme, so the upper-bound block is never violated: it can only be tighter than the view asks. The lower-bound block holds at the grid points and may fall short between them, so prefer [`ConicRelativisticValueatRiskView`](@ref) whenever the view admits it.

# Constructors

    GridRelativisticValueatRiskView(;
        pct::Number = 0.5,
        K::Integer = 11,
        M::Number = 10,
        iters::Integer = 50,
        tol::Number = 1e-10,
        tilt_iters::Integer = 200
    ) -> GridRelativisticValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < pct < 1`.
  - `K >= 1` and `isodd(K)`.
  - `M > 0`.
  - `iters >= 1`.
  - `tol >= 0`.
  - `tilt_iters >= 1`.

# Examples

```jldoctest
julia> GridRelativisticValueatRiskView()
GridRelativisticValueatRiskView
         pct ┼ Float64: 0.5
           K ┼ Int64: 11
           M ┼ Int64: 10
       iters ┼ Int64: 50
         tol ┼ Float64: 1.0e-10
  tilt_iters ┴ Int64: 200
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
    """
    $(field_dict[:ep_grid_iters])
    """
    iters
    """
    $(field_dict[:ep_grid_tol])
    """
    tol
    """
    $(field_dict[:ep_grid_tilt_iters])
    """
    tilt_iters
    function GridRelativisticValueatRiskView(pct::Number, K::Integer, M::Number,
                                             iters::Integer, tol::Number,
                                             tilt_iters::Integer)
        assert_unit_interval(pct, :pct)
        @argcheck(K >= one(K) && isodd(K), DomainError(K, "K must be odd and >= 1"))
        @argcheck(M > zero(M), DomainError(M, "M must be > 0"))
        @argcheck(iters >= one(iters), DomainError(iters, "iters must be >= 1"))
        @argcheck(tol >= zero(tol), DomainError(tol, "tol must be >= 0"))
        @argcheck(tilt_iters >= one(tilt_iters),
                  DomainError(tilt_iters, "tilt_iters must be >= 1"))
        return new{typeof(pct), typeof(K), typeof(M), typeof(iters), typeof(tol),
                   typeof(tilt_iters)}(pct, K, M, iters, tol, tilt_iters)
    end
end
function GridRelativisticValueatRiskView(; pct::Number = 0.5, K::Integer = 11,
                                         M::Number = 10, iters::Integer = 50,
                                         tol::Number = 1e-10,
                                         tilt_iters::Integer = 200)::GridRelativisticValueatRiskView
    return GridRelativisticValueatRiskView(pct, K, M, iters, tol, tilt_iters)
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

Spans of the two searches that read a relativistic value at risk.

[`ep_rlvar_shift`](@ref) minimises over the shift of the primal programme, and [`ep_rlvar`](@ref) minimises over the logarithm of the dual variable. Neither bracket is a proof: each is a margin wide enough for the data this library was measured on. Widen one where the minimiser lands on an end of it. `Optim` reports an end as converged, so read the minimiser rather than trust the flag.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskViewBracket(;
        tspan::Number = 2,
        zlo::Number = -20,
        zhi::Number = 10
    ) -> RelativisticValueatRiskViewBracket

Keywords correspond to the struct's fields.

## Validation

  - `tspan > 0`.
  - `zlo < zhi`.

# Examples

```jldoctest
julia> RelativisticValueatRiskViewBracket()
RelativisticValueatRiskViewBracket
  tspan ┼ Int64: 2
    zlo ┼ Int64: -20
    zhi ┴ Int64: 10
```

# Related

  - [`RelativisticValueatRiskView`](@ref)
  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_shift`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
@concrete struct RelativisticValueatRiskViewBracket <: AbstractAlgorithm
    """
    $(field_dict[:ep_bracket_rlvar_tspan])
    """
    tspan
    """
    $(field_dict[:ep_bracket_rlvar_zlo])
    """
    zlo
    """
    $(field_dict[:ep_bracket_rlvar_zhi])
    """
    zhi
    function RelativisticValueatRiskViewBracket(tspan::Number, zlo::Number, zhi::Number)
        @argcheck(tspan > zero(tspan), DomainError(tspan, "tspan must be > 0"))
        @argcheck(zlo < zhi, DomainError((zlo, zhi), "zlo must be < zhi"))
        return new{typeof(tspan), typeof(zlo), typeof(zhi)}(tspan, zlo, zhi)
    end
end
function RelativisticValueatRiskViewBracket(; tspan::Number = 2, zlo::Number = -20,
                                            zhi::Number = 10)::RelativisticValueatRiskViewBracket
    return RelativisticValueatRiskViewBracket(tspan, zlo, zhi)
end
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

A group of **conditional value at risk** views, with the significance level and formulation they are read under.

A `prior(...)` reference inside `views` is replaced by the prior conditional value at risk at this group's `alpha`, so a view stated against the prior moves with the level.

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

A group of **entropic value at risk** views, with the significance level and formulation they are read under.

A `prior(...)` reference inside `views` is replaced by the prior entropic value at risk at this group's `alpha`, so a view stated against the prior moves with the level.

`alg` is where the grid of dual variables and the big-M constant live: a [`GridEntropicValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`, so views at different significance levels can take different grids.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropicValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        alg::Option{<:EVaRVF_VecEVaRVF} = nothing,
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        zlo::Option{<:Number} = nothing
    ) -> EntropicValueatRiskView

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - If `alg` is a vector, `!isempty(alg)`.
  - If `zlo` is a number, `0 < zlo < 1`.

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
         │          pct ┼ Float64: 0.8
         │            K ┼ Int64: 21
         │            M ┼ Int64: 10
         │        iters ┼ Int64: 50
         │          tol ┼ Float64: 1.0e-10
         │   tilt_iters ┴ Int64: 200
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
     zlo ┴ nothing
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
    """
    $(field_dict[:optargs]) It reaches every [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) call these views make: the sample EVaR of [`ep_evar`](@ref), and the centre of the grid of [`ep_evar_anchor`](@ref). Each is a bracketed scalar minimisation, so left empty it takes `Optim.Brent()`.
    """
    args
    """
    $(field_dict[:optkwargs]) They reach the same searches `args` does.
    """
    kwargs
    """
    $(field_dict[:ep_tv_evar_zlo])
    """
    zlo
    function EntropicValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                     alg::Option{<:EVaRVF_VecEVaRVF}, args::Tuple,
                                     kwargs::NamedTuple, zlo::Option{<:Number})
        assert_unit_interval(alpha, :alpha)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        if !isnothing(zlo)
            @argcheck(zero(zlo) < zlo < one(zlo), DomainError(zlo, "zlo must be in (0, 1)"))
        end
        return new{typeof(views), typeof(alpha), typeof(alg), typeof(args), typeof(kwargs),
                   typeof(zlo)}(views, alpha, alg, args, kwargs, zlo)
    end
end
function EntropicValueatRiskView(; views::LinearConstraintEstimator, alpha::Number = 0.05,
                                 alg::Option{<:EVaRVF_VecEVaRVF} = nothing,
                                 args::Tuple = (), kwargs::NamedTuple = (;),
                                 zlo::Option{<:Number} = nothing)::EntropicValueatRiskView
    return EntropicValueatRiskView(views, alpha, alg, args, kwargs, zlo)
end
"""
$(DocStringExtensions.TYPEDEF)

A group of **relativistic value at risk** views, with the significance level, the deformation parameter and the formulation they are read under.

A `prior(...)` reference inside `views` is replaced by the prior relativistic value at risk at this group's `alpha` and `kappa`, so a view stated against the prior moves with both.

`alg` is where the grid of primal points and the big-M constant live: a [`GridRelativisticValueatRiskView`](@ref) in this field gives these views their own `pct`, `K` and `M`, so views at different significance levels can take different grids.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativisticValueatRiskView(;
        views::LinearConstraintEstimator,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        alg::Option{<:RLVaRVF_VecRLVaRVF} = nothing,
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing
    ) -> RelativisticValueatRiskView

Keywords correspond to the struct's fields.

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
      alg ┼ nothing
     args ┼ Tuple{}: ()
   kwargs ┼ @NamedTuple{}: NamedTuple()
  bracket ┴ nothing
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
    """
    $(field_dict[:optargs]) It reaches every [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) call these views make: the two searches of [`ep_rlvar`](@ref), the shift of [`ep_rlvar_shift`](@ref), and the centre of the grid of [`ep_rlvar_anchor`](@ref). Each is a bracketed scalar minimisation, so left empty it takes `Optim.Brent()`.
    """
    args
    """
    $(field_dict[:optkwargs]) They reach the same searches `args` does.
    """
    kwargs
    """
    $(field_dict[:ep_tv_bracket])
    """
    bracket
    function RelativisticValueatRiskView(views::LinearConstraintEstimator, alpha::Number,
                                         kappa::Number, alg::Option{<:RLVaRVF_VecRLVaRVF},
                                         args::Tuple, kwargs::NamedTuple,
                                         bracket::Option{<:RelativisticValueatRiskViewBracket})
        assert_unit_interval(alpha, :alpha)
        assert_unit_interval(kappa, :kappa)
        if isa(alg, AbstractVector)
            @argcheck(!isempty(alg), IsEmptyError("alg cannot be empty"))
        end
        return new{typeof(views), typeof(alpha), typeof(kappa), typeof(alg), typeof(args),
                   typeof(kwargs), typeof(bracket)}(views, alpha, kappa, alg, args, kwargs,
                                                    bracket)
    end
end
function RelativisticValueatRiskView(; views::LinearConstraintEstimator,
                                     alpha::Number = 0.05, kappa::Number = 0.3,
                                     alg::Option{<:RLVaRVF_VecRLVaRVF} = nothing,
                                     args::Tuple = (), kwargs::NamedTuple = (;),
                                     bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)::RelativisticValueatRiskView
    return RelativisticValueatRiskView(views, alpha, kappa, alg, args, kwargs, bracket)
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

Do nothing when a problem states no **value at risk** view.

`ep_var_views!` is the verb that turns a group of value at risk views into rows of the entropy pooling constraint dictionary. This method is the absent-view branch: it registers no row, so a higher-level routine can call the verb without special-casing `var_views = nothing`.

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
    ep_var_views!(var_views::ValueatRiskView, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::UniverseSets, alpha::Number;
                  strict::Bool = false)

Add the **value at risk** views of a group to the entropy pooling constraint dictionary.

The first method unpacks a [`ValueatRiskView`](@ref) into its equations and its significance level, and hands both to the second. The second carries the body: it parses the view equations, replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The statistic is the value at risk of the posterior distribution, not the variance, which the `sigma_views` family holds.

A value at risk view is linear in the posterior probabilities: it constrains the probability mass at or beyond the target loss, so it needs no auxiliary variable and no moment is fixed on its account.

# Mathematical definition

The value at risk at level ``\\alpha`` is the smallest loss the posterior leaves at most ``\\alpha`` of its mass beyond, so a view on it is a statement about the tail mass of the sample:

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(x_{i}) \\geq \\bar{v} \\quad &\\Longleftrightarrow \\quad \\sum_{t \\in \\mathcal{T}_{i}(\\bar{v})} p_{t} \\geq \\alpha\\,, \\\\
\\mathrm{VaR}_{\\alpha}(x_{i}) = \\bar{v} \\quad &\\Longleftrightarrow \\quad \\sum_{t \\in \\mathcal{T}_{i}(\\bar{v})} p_{t} = \\alpha\\,, \\\\
\\mathcal{T}_{i}(\\bar{v}) &= \\left\\{ t : x_{t,\\,i} \\leq -\\lvert \\bar{v} \\rvert \\right\\}\\,.
\\end{align}
```

Both are linear in ``\\boldsymbol{p}``, which is why this view reaches [`OptimEntropyPooling`](@ref) as readily as [`JuMPEntropyPooling`](@ref).

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:alpha_rm])
  - ``\\bar{v}``: Target value at risk of the view.
  - ``\\mathcal{T}_{i}(\\bar{v})``: Observations of asset ``i`` whose loss reaches the target.

# Algorithm

 1. Parse the view equations of `var_views.val`, accepting `==` and `>=` alone.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior value at risk at `alpha`, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`.
 5. Check the three preconditions of the section below.
 6. For each block present, and each row `i` of it, read the asset the row names into `j`, and the observations of `view(X, :, j)` at or below `-abs(B[i])` into `idx`.
 7. Raise when `idx` names no observation.
 8. Read the sense the row takes into `sign`: it is one for an equality row and for a non-negative right-hand side, and minus one otherwise.
 9. Build the row `Ai` that carries `sign` at `idx` and zero elsewhere, and add it against `sign * alpha` with [`add_ep_constraint!`](@ref).

# Arguments

  - `var_views`: VaR view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `alpha`: Confidence level for VaR.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Validation

  - Every coefficient has magnitude zero or one. Any other coefficient raises an `ArgumentError`. The check reads the magnitude because the parser normalises a `>=` row to `<=` by negation, which carries a coefficient of one into the block as minus one.
  - Every view names one asset. A view over more than one asset raises an `ArgumentError`.
  - Every target is non-negative. A negative target raises a `DomainError`.
  - The sample must hold at least one observation whose loss reaches the target. A view more extreme than the worst realisation raises a `DomainError` naming the largest target the asset admits.

# Returns

  - `nothing`: The function mutates `epc` in-place.

# Related

  - [`ValueatRiskView`](@ref)
  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior value at risk a `prior(...)` reference resolves to.
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_var_views!(var_views::ValueatRiskView, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)
    return ep_var_views!(var_views.views, epc, pr, sets, var_views.alpha; strict = strict)
end
"""
    ep_var_views!(var_views::AbstractVector{<:ValueatRiskView}, args...; kwargs...)

Add each group of **value at risk** views under its own significance level.

Every [`ValueatRiskView`](@ref) in the vector is added in turn, so the groups accumulate into the same constraint set and one entropy pooling solve answers all of them.

# Algorithm

 1. Add each [`ValueatRiskView`](@ref) of `var_views` in turn, forwarding `args...` and `kwargs...` to each call.
 2. Return `nothing`. Each call has already written its rows into `epc`.

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
    #! `all`, not `any`: a row of a universe of more than one asset always carries a zero,
    #! so `any` held for every view and the guard never fired. The body then read the
    #! target off `B` while it discarded the coefficient, which doubles the threshold a
    #! `2*AAPL` view asks for. The magnitude, not the value: the parser normalises a `>=`
    #! row to `<=` by negation, so a coefficient of one reaches `A_ineq` as `-1`.
    unit_coef = x -> (iszero(x) || isone(abs(x)))
    @argcheck(!(!isnothing(lcs.ineq) && !all(unit_coef, lcs.A_ineq) ||
                !isnothing(lcs.eq) && !all(unit_coef, lcs.A_eq)),
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
            #! An `:ineq` row reaches here normalised to `A * p <= B`, and the parser accepts
            #! `>=` alone, so `B[i]` is the negated target and never positive. A zero target
            #! must therefore take the same sign as a positive one; `>=` here read it as a
            #! `<=` view and flipped the row.
            sign = ifelse(p == :eq || B[i] > zero(eltype(B)), one(eltype(B)),
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

`entropy_pooling` computes posterior probabilities by minimising the Kullback-Leibler divergence of the posterior weights from the prior ones, subject to moment and view constraints. The optimisation is performed using [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). This method is used internally by [`MeucciEntropyPoolingPrior`](@ref) and [`EntropyPoolingPrior`](@ref) when the optimiser is an [`OptimEntropyPooling`](@ref).

The two optimisation algorithms minimise the same objective and reach the same posterior. They differ only in the arithmetic that evaluates it.

The dual carries one variable per row rather than one per observation, and it has no room for an auxiliary variable, so it expresses no tail view. It also has no slack variable: the fixed equality rows of the `:feq` key are relaxed by holding their dual variables in the box ``[-s_{c2},\\, s_{c2}]``, which is the dual of a penalty of weight ``s_{c2}`` on the norm of the slack the primal would carry.

!!! warning

    An infeasible view set answers without a raise. The dual of such a set is unbounded below, so the minimiser runs away rather than settling. The iterate stops moving once the exponential underflows, `Optim` reports `x_converged` or `f_converged`, and `Optim.converged` accepts it. The posterior it returns is degenerate: the probability collapses onto the observation with the largest coefficient, and the view the caller wrote is missed by any margin. Read the answer rather than the flag. The effective number of scenarios falls to a handful out of ``T`` and one weight sits near one, the Kullback-Leibler divergence is large, and the posterior statistic the view named is far from its target. Views that pull one asset in two directions at once are the common way to reach it: a variance view that shrinks an asset, written beside a conditional value at risk view that fattens the same asset's tail, asks for a thin body and a fat tail at once. The same pair on two different assets is feasible and solves normally, so it is the direction and not the pairing. The gradient of this dual is ``\\boldsymbol{B} - \\mathbf{A} \\boldsymbol{y}``, the primal residual of the view set, so `Optim.g_converged` and `Optim.g_residual` do separate the two outcomes. Neither is read: `Optim.g_converged` also refuses a solve that is correct and merely loose, so acting on it needs a tolerance on the residual, and that tolerance is a policy this library does not set.

# Mathematical definition

The primal minimises the Kullback-Leibler divergence of the posterior from the prior, over the probabilities that meet every row. Its dual carries one Lagrange multiplier per row, and is unconstrained apart from the box the sense of each row imposes:

```math
\\begin{align}
\\underset{\\boldsymbol{x}}{\\min} &\\; \\boldsymbol{x}^\\intercal \\boldsymbol{B} + \\sum_{t=1}^{T} q_{t} \\exp\\!\\left(-\\boldsymbol{x}^\\intercal \\mathbf{A}_{\\cdot t} - 1\\right)\\,.
\\end{align}
```

The optimal posterior probabilities recover from the minimiser as:

```math
\\begin{align}
p_{t}^{*} &= q_{t} \\exp\\!\\left(-\\boldsymbol{x}^{*\\intercal} \\mathbf{A}_{\\cdot t} - 1\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:ep_prior_probs])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:T])
  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_sc2])
  - ``\\boldsymbol{x}``: Lagrange multipliers of the rows, the variable of the dual.
  - ``\\mathbf{A}_{\\cdot t}``: ``t``-th column of ``\\mathbf{A}``, the coefficient every row gives observation ``t``.
  - ``q_{t}``, ``p_{t}^{*}``: Prior and optimal posterior probability of observation ``t``.

# Algorithm

 1. Open `A` and `B` with the row that pins the posterior to sum to one, both sides divided by ``\\sqrt{T}``.
 2. Stack the block of every key of `epc` onto `A` and `B`, and set the box `wb` of that block's dual variables from the key: free for `:eq` and `:cvar_eq`, non-negative for `:ineq`, and ``[-s_{c2},\\, s_{c2}]`` for `:feq`. Raise on any other key. A `:feq` block is left out when ``s_{c2}`` is zero, because that box pins its dual variables to zero and the fixed rows then carry no weight.
 3. Start every dual variable at ``1/\\sqrt{T}``, clamped into its own box. A `:feq` box is ``[-s_{c2},\\, s_{c2}]``, so an `s_{c2}` below ``1/\\sqrt{T}`` would otherwise place the start outside it.
 4. Minimise the dual objective over that box with `Optim.optimize`, through the branch `alg` selects. Both the objective and its gradient are multiplied by ``s_{c1}``.
 5. Raise when `Optim.converged` reports that the solve failed.
 6. Recover the posterior probabilities from the minimiser, and return them as `StatsBase.pweights`.

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.

  - `opt`: Optim.jl-based entropy pooling optimiser.

      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Evaluate the objective through the exponential of the dual variables.
      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Evaluate the objective in log space.

# Validation

  - Every key of `epc` is one of `:eq`, `:ineq`, `:cvar_eq` and `:feq`. Any other key raises a `KeyError`.
  - The solve must converge. A solve that `Optim.converged` reports as failed raises an `ErrorException`.
  - An infeasible view set is **not** caught. `Optim.converged` is true on `x_converged` or `f_converged` alone, and the dual of an infeasible set stops on one of those. The summary paragraph states the shape of that answer and how to recognise it.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

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
        s = length(val[2])
        #! A `:feq` dual variable is boxed by `sc2`, so `sc2 == 0` pins it to zero and the
        #! fix carries no weight. That is what a primal penalty of weight `sc2` means, and
        #! it is what the JuMP route gives. `Optim` cannot start inside a box of zero
        #! width, so the block is left out rather than boxed to a point.
        if key == :feq && iszero(opt.sc2)
            continue
        end
        A = vcat(A, val[1])
        B = vcat(B, val[2])
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
    #! `wb` bounds a `:feq` dual variable by `sc2`, and the constructor admits any
    #! `sc2 >= 0`. An unclamped start of `factor` therefore sits outside the box whenever
    #! `sc2 < 1/sqrt(T)`, and `Optim` raises an opaque `ArgumentError` instead of solving.
    #! The box is halved before the clamp because `Optim` also refuses a start that sits
    #! *on* a boundary. Halving leaves the infinite bounds and the `:ineq` zero untouched.
    x0 = clamp.(fill(factor, size(A, 1)), view(wb, :, 1) / 2, view(wb, :, 2) / 2)
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
        G .= opt.sc1 .* grad
        return G
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
    #! An infeasible view set is not caught here. `Optim.converged` is true on
    #! `x_converged` or `f_converged` alone, and the dual of an infeasible set stops on one
    #! of those: the minimiser runs away, the recovered probabilities collapse onto the
    #! observation with the largest coefficient, and the solve is reported as converged
    #! while the views are missed. The gradient of this dual is `B - A * y`, the primal
    #! residual of the view set, so `Optim.g_converged` does separate the two. It is not
    #! read here: it also refuses a solve that is correct and merely loose, so it would
    #! need a tolerance on the residual, and that tolerance is a policy this library does
    #! not set. The docstring states the failure and how to recognise it. See issue #572.
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
        s = length(val[2])
        #! A `:feq` dual variable is boxed by `sc2`, so `sc2 == 0` pins it to zero and the
        #! fix carries no weight. That is what a primal penalty of weight `sc2` means, and
        #! it is what the JuMP route gives. `Optim` cannot start inside a box of zero
        #! width, so the block is left out rather than boxed to a point.
        if key == :feq && iszero(opt.sc2)
            continue
        end
        A = vcat(A, val[1])
        B = vcat(B, val[2])
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
    #! `wb` bounds a `:feq` dual variable by `sc2`, and the constructor admits any
    #! `sc2 >= 0`. An unclamped start of `factor` therefore sits outside the box whenever
    #! `sc2 < 1/sqrt(T)`, and `Optim` raises an opaque `ArgumentError` instead of solving.
    #! The box is halved before the clamp because `Optim` also refuses a start that sits
    #! *on* a boundary. Halving leaves the infinite bounds and the `:ineq` zero untouched.
    x0 = clamp.(fill(factor, size(A, 1)), view(wb, :, 1) / 2, view(wb, :, 2) / 2)
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
        G .= opt.sc1 .* grad
        return G
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
    #! An infeasible view set is not caught here. `Optim.converged` is true on
    #! `x_converged` or `f_converged` alone, and the dual of an infeasible set stops on one
    #! of those: the minimiser runs away, the recovered probabilities collapse onto the
    #! observation with the largest coefficient, and the solve is reported as converged
    #! while the views are missed. The gradient of this dual is `B - A * y`, the primal
    #! residual of the view set, so `Optim.g_converged` does separate the two. It is not
    #! read here: it also refuses a solve that is correct and merely loose, so it would
    #! need a tolerance on the residual, and that tolerance is a policy this library does
    #! not set. The docstring states the failure and how to recognise it. See issue #572.
    @argcheck(Optim.converged(result),
              ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    x = Optim.minimizer(result)
    return StatsBase.pweights(exp.(log_p - (one(eltype(log_p)) .+ transpose(A) * x)))
end
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, opt::JuMPEntropyPooling)

Solve the primal of the entropy pooling problem using JuMP.jl.

`entropy_pooling` computes posterior probabilities by minimising the Kullback-Leibler divergence of the posterior weights from the prior ones, subject to moment and view constraints. The optimisation is performed using [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl). This method is used internally by [`MeucciEntropyPoolingPrior`](@ref) and [`EntropyPoolingPrior`](@ref) when the optimiser is a [`JuMPEntropyPooling`](@ref).

This method registers no model entry of its own. It is the three-argument shape of a problem that states no tail view, and the four-argument method carries the model, its `# JuMP formulation` and its `# Algorithm`.

# Algorithm

 1. Call the four-argument method with an empty `AbstractEntropyPoolingTailView` vector, and return what it answers.

# Arguments

  - `w`: Prior weights (length = number of observations).

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.

  - `opt`: JuMP.jl-based entropy pooling optimiser.

      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Write the divergence against the prior probabilities directly.
      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Write the divergence against a unit reference and subtract the prior log-probabilities in the objective.

# Returns

  - `pw::StatsBase.ProbabilityWeights`: Posterior probability weights satisfying the constraints.

# Related

  - [`JuMPEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ep_jump_views!`](@ref): registers the rows of `epc`, and the slack that relaxes the fixed equalities.
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

Read the prior **conditional value at risk** of asset `i` at the level `alpha`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:cvar)` names, by applying [`ConditionalValueatRisk`](@ref) to the `i`-th column of `pr.X`. That is the sample conditional value at risk of the loss series, and it rests on no distributional assumption. It ignores `pr.w`, on the reasoning [`get_pr_value`](@ref) gives for the value at risk.

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

Read the prior **variance** of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:sigma)` names, the `i`-th diagonal entry of `pr.sigma`. The tag is `:sigma` and the statistic is the variance, not the standard deviation and not the value at risk, which the tag `:var` names.

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

Add the **variance** views of a group to the entropy pooling constraint dictionary.

`ep_sigma_views!` parses variance view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. The statistic is the variance of the posterior distribution, not the value at risk, which the `var_views` family holds.

The variance is quadratic in the returns and linear in the posterior probabilities only once the mean is a constant. This method therefore returns the assets whose mean [`fix_mu!`](@ref) must hold at the prior, so a variance view does not move the mean it is measured about.

# Mathematical definition

The row states the posterior second central moment about the **prior** mean, which is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{Var}_{\\boldsymbol{p}}[x_{i}] &= \\sum_{t=1}^{T} p_{t} \\left(x_{t,\\,i} - \\mu_{i}\\right)^{2}\\,, \\\\
\\sum_{i=1}^{N} a_{k,\\,i} \\mathrm{Var}_{\\boldsymbol{p}}[x_{i}] &\\lessgtr B_{k}\\,, \\quad \\forall\\, k = 1,\\ldots,K\\,.
\\end{align}
```

The identity holds only while the posterior mean of asset ``i`` equals ``\\mu_{i}``, which is why the assets this method names are handed to [`fix_mu!`](@ref).

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:B])
  - ``a_{k,\\,i}``: Coefficient asset ``i`` takes in view ``k``.
  - ``K``: Number of views the group states.

# Algorithm

 1. Parse the view equations of `sigma_views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior variance, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`.
 5. Build `tmp`, the squared deviations of every observation from the prior mean, transposed so a row of `lcs` multiplies it from the left.
 6. For each block present, add `A * tmp` against `B` under that key with [`add_ep_constraint!`](@ref), and mark in `to_fix` every asset the block names.

# Arguments

  - `sigma_views`: Variance view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean to be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior variance a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref): consumes the `to_fix` this method returns.
  - [`fix_sigma!`](@ref)
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

Hold the **variance** of the named assets at the prior value.

`fix_sigma!` identifies assets in `to_fix` that are not yet fixed (i.e., not present in `fixed`), and adds constraints to the entropy pooling constraint dictionary `epc` to fix their variance to the prior value. This ensures that higher moment views (e.g., skewness, kurtosis, correlation) do not inadvertently alter the variance of these assets. The function updates `fixed` in-place to reflect the newly fixed assets.

The rows go in under the `:feq` key, which the optimiser relaxes with a penalised slack rather than enforcing exactly. A fixing row is a wish, not a view: it competes with the views that were asked for, and it yields where the two cannot both hold.

# Mathematical definition

The posterior second central moment of every named asset, taken about the prior mean, is held at the prior variance:

```math
\\sum_{t=1}^{T} p_{t} \\left(x_{t,\\,i} - \\mu_{i}\\right)^{2} = \\sigma_{i}^{2}\\,, \\quad \\forall\\, i \\in \\mathcal{F}\\,.
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:T])
  - ``\\mathcal{F}``: Assets named by `to_fix` that `fixed` does not already hold.

# Algorithm

 1. Read the prior variances, the diagonal of `pr.sigma`, into `sigma`.
 2. Read the assets that `to_fix` names and `fixed` does not already hold into `fix`.
 3. Return when `fix` names no asset.
 4. Add one `:feq` row per named asset, the squared deviations of that asset from its prior mean against `sigma[fix]`, with [`add_ep_constraint!`](@ref).
 5. Mark the named assets in `fixed`, so a later call adds no second row for them.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `fixed`: Boolean vector indicating which assets have their variance fixed.
  - `to_fix`: Boolean vector indicating which assets should have their variance fixed.
  - `pr`: Prior result containing asset return information.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in-place.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`ep_sigma_views!`](@ref)
  - [`fix_mu!`](@ref): the same rule, one moment lower.
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

`replace_coprior_views` scans a parsed correlation view constraint (`ParsingResult`) for references to prior values (e.g., `prior(A, B)`), and replaces them with the actual prior correlation value from the provided prior result object. This ensures that prior-based terms in correlation view constraints are treated as constants and not as variables in the optimisation.

It is the pair counterpart of [`replace_prior_views`](@ref), and it answers a [`RhoParsingResult`](@ref) rather than a [`ParsingResult`](@ref): a pair view carries the index pair of every term, which the verb that places the view in the covariance matrix reads back.

# Mathematical definition

A parsed pair view is the row ``\\sum_{k} c_{k} v_{k} \\lessgtr b``, whose variables name asset pairs. A term whose variable is `prior(a, b)` carries a constant, so moving every such term to the right-hand side gives an equivalent row:

```math
\\sum_{k \\notin \\mathcal{P}} c_{k} v_{k} \\lessgtr b - \\sum_{k \\in \\mathcal{P}} c_{k} \\pi_{a_{k},\\, b_{k}}\\,.
```

A `prior(gA, gB)` reference over a pair of groups carries one constant per spanned asset pair, so the subtraction broadcasts and ``b`` widens to a vector of that length, one right-hand side per row the view emits.

Where:

  - ``c_{k}``, ``v_{k}``: Coefficient and variable of the ``k``-th term of the view.
  - ``b``: Right-hand side of the view.
  - ``\\mathcal{P}``: Terms whose variable is a `prior(...)` reference.
  - ``\\pi_{a,\\,b}``: Prior value of the statistic `key` for the asset pair ``(a, b)``, read by [`get_pr_value`](@ref).

# Algorithm

 1. Match the pattern `prior(<asset1>, <asset2>)` against the variable of each term in turn.
 2. When a term does not match, read its own pair `(a, b)` instead. Raise when the term is not of the form `(a, b)`.
 3. Find both names in the universe, reading a bracketed name as a group and every other as one asset. When either is absent, report it through `strict_diagnostic`, record the term for removal, and take the next term.
 4. Record the index pair in `jk_idx`, and take the next term.
 5. For a term that does match, find both names the same way, subtract [`get_pr_value`](@ref) times the term's coefficient from `rhs` with the broadcasting operators, and record the term for removal.
 6. Return a [`RhoParsingResult`](@ref) over the untouched terms when step 3 and step 5 recorded no term.
 7. Drop the recorded terms from `vars` and `coef`, rebuild the equation string, and return a [`RhoParsingResult`](@ref) that carries the adjusted `rhs` and `jk_idx`.

# Arguments

  - `res`: Parsed correlation view constraint containing variables and coefficients.
  - `pr`: Prior result object containing prior correlation values.
  - `sets`: Asset set mapping asset names to indices.
  - `key`: Symbol representing whether it's a correlation or covariance view.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Validation

  - Every term that is not a `prior(...)` reference must be of the form `(a, b)`. Any other form raises an `ArgumentError`.
  - Every `prior(...)` reference must be of the form `prior(a, b)`. Any other form raises an `ArgumentError`.
  - An asset a reference names that the universe does not hold raises an `ArgumentError` when `strict` is `true`, and warns otherwise. The term is dropped either way.
  - At least one term of the view must keep a variable of its own. A view whose every term is a `prior(...)` reference raises an `ArgumentError`.

# Returns

  - `res::RhoParsingResult`: Updated parsing result with prior references replaced by their values and correlation indices.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`replace_prior_views`](@ref): the single-asset counterpart.
  - [`get_pr_value`](@ref): reads the prior value each reference is replaced by.
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

`replace_coprior_views` applies [`replace_coprior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values. [`parse_equation`](@ref) answers a group of view equations with a vector of results, so this is the shape every caller in this file meets.

# Algorithm

 1. Broadcast the single-view method over `res`, forwarding `args...` and `kwargs...` to each call.
 2. Return the vector of the results, one [`RhoParsingResult`](@ref) per element of `res`, in the order of `res`.

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

Read the prior **correlation** or **covariance** of the asset pair `(i, j)`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. `Val(:cov)` names the entry `pr.sigma[i, j]`, and `Val(:rho)` names the same entry of `StatsBase.cov2cor(pr.sigma)`. These methods are used internally by [`replace_coprior_views`](@ref) to resolve a `prior(a, b)` reference.

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

Read the prior **correlations** or **covariances** of the asset pairs that two groups span.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. These methods read the same statistics their scalar siblings do, once per spanned pair, in the order of `zip(i, j)`. A view over a pair of groups emits one constraint row per spanned pair, so a `prior(gA, gB)` reference inside such a view must give each row that pair's own prior value.

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

Add the **covariance** views of a group to the entropy pooling constraint dictionary.

`ep_cov_views!` parses covariance view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. The statistic is the covariance of the posterior distribution, not the correlation, which the `rho_views` family holds.

The covariance is a product of returns and is linear in the posterior probabilities only once both means are constants. This method therefore returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior, so a covariance view does not move the lower moments it is measured about.

# Mathematical definition

The row states the posterior cross moment about the **prior** means, which is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{Cov}_{\\boldsymbol{p}}[x_{i}, x_{j}] &= \\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} - \\mu_{i} \\mu_{j}\\,, \\\\
c\\, \\mathrm{Cov}_{\\boldsymbol{p}}[x_{i}, x_{j}] &\\lessgtr b\\,,
\\end{align}
```

which the body writes with the constants gathered on the right:

```math
\\begin{align}
d\\, c \\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} &\\lessgtr d \\left(c\\, \\mu_{i} \\mu_{j} + b\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:T])
  - ``c``: Coefficient the view gives the pair.
  - ``b``: Target of the view, one value per spanned asset pair.
  - ``d``: Sign that [`comparison_sign_ineq_flag`](@ref) reads from the operator, so every inequality row reaches `epc` in the sense the `:ineq` key states.

The identity holds only while the posterior means of assets ``i`` and ``j`` equal ``\\mu_{i}`` and ``\\mu_{j}``, which is why the assets this method names are handed to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `cov_views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans, keeping a pair view a pair view.
 3. Replace every `prior(a, b)` reference by the prior covariance, through [`replace_coprior_views`](@ref). Each view is now a [`RhoParsingResult`](@ref) carrying its index pairs.
 4. For each view in turn, drop it when step 3 left it with no pair, and raise unless it names exactly one.
 5. Read the sign `d` and the inequality flag from the operator with [`comparison_sign_ineq_flag`](@ref).
 6. Read the index pair `(i, j)`, and build `Ai`, the product of the two return columns scaled by `d` and the view's coefficient.
 7. Build `Bi`, the target moved by the product of the prior means, scaled the same way. A single asset pair gives a scalar, which is wrapped into a one-element vector; a group pair gives one entry per spanned pair.
 8. Add the row against `Bi` under `:ineq` or `:eq` with [`add_ep_constraint!`](@ref), and mark both assets of the pair in `to_fix`.

# Arguments

  - `cov_views`: Covariance view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Validation

  - Every view names exactly one asset pair. A view that mixes pairs raises an `ArgumentError`. A view left with no pair, because every pair it named holds an asset the universe does not, is dropped with a report under `strict = false`; `strict = true` has already raised by then.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_coprior_views`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`get_pr_value`](@ref): reads the prior covariance a `prior(a, b)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): consume the `to_fix` this method returns.
  - [`ep_rho_views!`](@ref): the same rule on the correlation.
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
        #! `replace_coprior_views` drops a pair naming an asset the universe does not hold,
        #! which leaves the view with no pair at all. Under `strict = false` that must drop
        #! the row, as `get_linear_constraints` does for the linear families. Without this
        #! the guard below raised, called a view of no pairs one of several, and named an
        #! equation with no variable in it.
        if isempty(cov_view.vars)
            strict_diagnostic(empty_row_msg(cov_view.eqn, sets.dict[sets.xkey], sets.xkey;
                                            noun = "view"), strict)
            continue
        end
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

Add the **correlation** views of a group to the entropy pooling constraint dictionary.

`ep_rho_views!` parses correlation view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. The statistic is the correlation of the posterior distribution, not the covariance, which the `cov_views` family holds.

The correlation is a covariance divided by two standard deviations, and it is linear in the posterior probabilities only once both means and both variances are constants. This method therefore returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior, so a correlation view does not move the lower moments it is measured about.

# Mathematical definition

The row states the posterior cross moment about the **prior** means, with the target multiplied by the **prior** standard deviations, which is linear in the posterior probabilities:

```math
\\begin{align}
\\rho_{\\boldsymbol{p}}[x_{i}, x_{j}] &= \\dfrac{\\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} - \\mu_{i} \\mu_{j}}{\\sigma_{i} \\sigma_{j}}\\,, \\\\
c\\, \\rho_{\\boldsymbol{p}}[x_{i}, x_{j}] &\\lessgtr b\\,,
\\end{align}
```

which the body writes with the constants gathered on the right:

```math
\\begin{align}
d\\, c \\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} &\\lessgtr d \\left(c\\, \\mu_{i} \\mu_{j} + b\\, \\sigma_{i} \\sigma_{j}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:T])
  - ``\\sigma_{i}``: Prior standard deviation of asset ``i``, the positive root of ``\\sigma_{i}^{2}``.
  - ``c``: Coefficient the view gives the pair.
  - ``b``: Target of the view, one value per spanned asset pair, in ``[-1, 1]``.
  - ``d``: Sign that [`comparison_sign_ineq_flag`](@ref) reads from the operator, so every inequality row reaches `epc` in the sense the `:ineq` key states.

The identity holds only while the posterior means and variances of assets ``i`` and ``j`` equal the prior ones, which is why the assets this method names are handed to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `rho_views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans, keeping a pair view a pair view.
 3. Replace every `prior(a, b)` reference by the prior correlation, through [`replace_coprior_views`](@ref). Each view is now a [`RhoParsingResult`](@ref) carrying its index pairs.
 4. Read the prior variances, the diagonal of `pr.sigma`, into `sigma`.
 5. For each view in turn, drop it when step 3 left it with no pair, raise unless it names exactly one, and raise unless every target lies in ``[-1, 1]``.
 6. Read the sign `d` and the inequality flag from the operator with [`comparison_sign_ineq_flag`](@ref).
 7. Read the index pair `(i, j)`, and build `sigma_ij`, the root of the product of the two prior variances.
 8. Build `Ai`, the product of the two return columns scaled by `d` and the view's coefficient.
 9. Build `Bi`, the target multiplied by `sigma_ij` and moved by the product of the prior means, scaled the same way. A single asset pair gives a scalar, which is wrapped into a one-element vector; a group pair gives one entry per spanned pair.
10. Add the row against `Bi` under `:ineq` or `:eq` with [`add_ep_constraint!`](@ref), and mark both assets of the pair in `to_fix`.

# Arguments

  - `rho_views`: Correlation view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Validation

  - Every view names exactly one asset pair. A view that mixes pairs raises an `ArgumentError`. A view left with no pair, because every pair it named holds an asset the universe does not, is dropped with a report under `strict = false`; `strict = true` has already raised by then.
  - Every target lies in ``[-1, 1]``. A target outside that range raises an `ArgumentError`.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_coprior_views`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`get_pr_value`](@ref): reads the prior correlation a `prior(a, b)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): consume the `to_fix` this method returns.
  - [`ep_cov_views!`](@ref): the same rule on the covariance.
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
        #! See the twin note in `ep_cov_views!`: a pair naming an unknown asset leaves the
        #! view with no pair, and under `strict = false` that drops the row.
        if isempty(rho_view.vars)
            strict_diagnostic(empty_row_msg(rho_view.eqn, sets.dict[sets.xkey], sets.xkey;
                                            noun = "view"), strict)
            continue
        end
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

Read the prior **skewness** of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:skew)` names, by applying [`Skewness`](@ref) to the `i`-th column of `pr.X`. That is the standardised third central moment of the sample, and it ignores `pr.w`.

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

Add the **skewness** views of a group to the entropy pooling constraint dictionary.

`ep_sk_views!` parses skewness view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. The statistic is the standardised third central moment of the posterior distribution.

The skewness is a third central moment divided by a cube of the standard deviation, and it is linear in the posterior probabilities only once the mean and the variance are constants. This method therefore returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior, so a skewness view does not move the lower moments it is measured about.

# Mathematical definition

The third central moment expands into raw moments, and the two lowest of them are the **prior** constants, so what remains is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{3}\\right] &= \\sum_{t=1}^{T} p_{t} x_{t,\\,i}^{3} - 3 \\mu_{i} \\sigma_{i}^{2} - \\mu_{i}^{3}\\,, \\\\
\\mathrm{Skew}_{\\boldsymbol{p}}[x_{i}] &= \\dfrac{\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{3}\\right]}{\\left(\\sigma_{i}^{2}\\right)^{3/2}}
 = \\sum_{t=1}^{T} p_{t} \\dfrac{x_{t,\\,i}^{3} - \\mu_{i}^{3} - 3 \\mu_{i} \\sigma_{i}^{2}}{\\left(\\sigma_{i}^{2}\\right)^{3/2}}\\,, \\\\
\\sum_{i=1}^{N} a_{k,\\,i} \\mathrm{Skew}_{\\boldsymbol{p}}[x_{i}] &\\lessgtr B_{k}\\,, \\quad \\forall\\, k = 1,\\ldots,K\\,.
\\end{align}
```

The second line uses ``\\sum_{t} p_{t} = 1`` to carry the two constants inside the sum, which is the form the body builds.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:B])
  - ``a_{k,\\,i}``: Coefficient asset ``i`` takes in view ``k``.
  - ``K``: Number of views the group states.

The identity holds only while the posterior mean and variance of asset ``i`` equal ``\\mu_{i}`` and ``\\sigma_{i}^{2}``, which is why the assets this method names are handed to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `skew_views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior skewness, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`.
 5. Read the prior variances, the diagonal of `pr.sigma`, into `sigma`.
 6. Build `tmp`, the standardised third moment contribution of every observation, transposed so a row of `lcs` multiplies it from the left.
 7. For each block present, add `A * tmp` against `B` under that key with [`add_ep_constraint!`](@ref), and mark in `to_fix` every asset the block names.

# Arguments

  - `skew_views`: Skewness view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior skewness a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): consume the `to_fix` this method returns.
  - [`ep_kt_views!`](@ref): the same rule, one moment higher.
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

Read the prior **kurtosis** of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:kurtosis)` names, by applying [`HighOrderMoment`](@ref) with a [`StandardisedHighOrderMoment`](@ref) of [`FourthMoment`](@ref) to the `i`-th column of `pr.X`. That is the standardised fourth central moment of the sample, and it ignores `pr.w`.

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

Add the **kurtosis** views of a group to the entropy pooling constraint dictionary.

`ep_kt_views!` parses kurtosis view equations from a [`LinearConstraintEstimator`](@ref), replaces any prior references with their actual values, and constructs the corresponding linear constraints for entropy pooling. The constraints are then added to the entropy pooling constraint dictionary `epc`. The statistic is the standardised fourth central moment of the posterior distribution.

The kurtosis is a fourth central moment divided by a square of the variance, and it is linear in the posterior probabilities only once the mean and the variance are constants. This method therefore returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior, so a kurtosis view does not move the lower moments it is measured about.

# Mathematical definition

The fourth central moment expands into raw moments, and every constant of the expansion is a **prior** one, so what remains is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{4}\\right] &= \\sum_{t=1}^{T} p_{t} \\left(x_{t,\\,i}^{4} - 4 \\mu_{i} x_{t,\\,i}^{3} + 6 \\mu_{i}^{2} x_{t,\\,i}^{2} - 3 \\mu_{i}^{4}\\right)\\,, \\\\
\\mathrm{Kurt}_{\\boldsymbol{p}}[x_{i}] &= \\dfrac{\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{4}\\right]}{\\left(\\sigma_{i}^{2}\\right)^{2}}\\,, \\\\
\\sum_{i=1}^{N} a_{k,\\,i} \\mathrm{Kurt}_{\\boldsymbol{p}}[x_{i}] &\\lessgtr B_{k}\\,, \\quad \\forall\\, k = 1,\\ldots,K\\,.
\\end{align}
```

The first line uses ``\\sum_{t} p_{t} = 1`` to carry the constant ``3 \\mu_{i}^{4}`` inside the sum, which is the form the body builds.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:B])
  - ``a_{k,\\,i}``: Coefficient asset ``i`` takes in view ``k``.
  - ``K``: Number of views the group states.

The identity holds only while the posterior mean and variance of asset ``i`` equal ``\\mu_{i}`` and ``\\sigma_{i}^{2}``, which is why the assets this method names are handed to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `kurtosis_views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior kurtosis, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`.
 5. Build `X_sq` and `mu_sq`, the squares of the returns and of the prior means.
 6. Build `tmp`, the standardised fourth moment contribution of every observation, transposed so a row of `lcs` multiplies it from the left.
 7. For each block present, add `A * tmp` against `B` under that key with [`add_ep_constraint!`](@ref), and mark in `to_fix` every asset the block names.

# Arguments

  - `kurtosis_views`: Kurtosis view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `to_fix::BitVector`: Boolean vector indicating which assets require their mean and variance to be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior kurtosis a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): consume the `to_fix` this method returns.
  - [`ep_sk_views!`](@ref): the same rule, one moment lower.
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
 2. Read `sc1`, `sc2`, `so` and `slv` off `opt`, and open an empty `JuMP.Model`.
 3. Create the posterior variable `x` and the divergence variable `t`. Under [`ExpEntropyPooling`](@ref) `x` also carries a non-negativity bound, which the cone of step 5 implies in either branch.
 4. Add the row that pins the sum of `x` to one.
 5. Add the relative entropy cone that bounds `t` below by the divergence. Under [`ExpEntropyPooling`](@ref) the cone reads the prior `w` as its reference, and under [`LogEntropyPooling`](@ref) it reads a unit reference instead.
 6. Register `obj_expr` as `so * t`.
 7. Add every view with [`ep_jump_views!`](@ref), which mutates `obj_expr` when a fixed equality is relaxed.
 8. Set the objective to `obj_expr` under [`ExpEntropyPooling`](@ref), and to `obj_expr` less `so` times the inner product of `x` with the prior log-probabilities under [`LogEntropyPooling`](@ref).
 9. Solve with `slv`, and raise when no solver configuration succeeds.
10. Return the value of `x` as `StatsBase.pweights`.

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

Each of the four named rows is registered only when `epc` holds the block it carries. [`add_ep_tail_view!`](@ref) registers every row a tail view needs, in `src/13_Prior/12_EntropyPoolingPrior.jl`.

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

export RhoParsingResult, LogEntropyPooling, ExpEntropyPooling, H0_EntropyPooling,
       H1_EntropyPooling, H2_EntropyPooling, JuMPEntropyPooling, OptimEntropyPooling,
       ConditionalValueatRiskEntropyPooling, ValueatRiskView, ConditionalValueatRiskView,
       EntropicValueatRiskView, LinearConditionalValueatRiskView,
       IntegerConditionalValueatRiskView, ConicEntropicValueatRiskView,
       GridEntropicValueatRiskView, RelativisticValueatRiskView,
       ConicRelativisticValueatRiskView, GridRelativisticValueatRiskView,
       RelativisticValueatRiskViewBracket
