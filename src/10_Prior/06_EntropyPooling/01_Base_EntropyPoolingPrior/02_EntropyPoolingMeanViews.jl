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
    announce_ep_departures(ni::VecStr, ledger::VecStr, viewless::Bool) -> Nothing

Report, once per entropy pooling fit, who left the investable universe and what their leaving cost the view set.

This is the family's call of [`announce_non_investable`](@ref), written once so the four [`ep_prior`](@ref) methods each spend one line on it and none of them can word it differently. It names the process an entropy pooling fit, because the message is otherwise the optimisation door's and would tell a standalone `prior(pe, X)` call that it is inside an optimisation it is not.

It is said **at the end of the fit**, not at the reduction. A staged algorithm interleaves its view builders with its solves, so the ledger is only complete when the last stage has stated its views; reporting earlier would report a third of the truth and reporting per stage would be three messages for one departure.

`viewless` raises the message to a warning: a departure that took the **last** surviving view leaves the fit with nothing to condition on, so the posterior is the prior probabilities and the answer is not the one the caller asked for. Every other drop trims the view set and is `@info`.

# Arguments

  - `ni`: The names the Investable Mask left out, from [`investable_views`](@ref).
  - `ledger`: What the departures cost, as the view builders recorded it.
  - `viewless`: Whether the fit ended up with no view at all *because* of the departures.

# Returns

  - `nothing`.

# Related

  - [`announce_non_investable`](@ref)
  - [`investable_views`](@ref)
  - [`record_non_investable_drop!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function announce_ep_departures(ni::VecStr, ledger::VecStr, viewless::Bool)::Nothing
    consequence = if viewless
        "Every view stated named one of them, so the fit proceeds with no view at all and its posterior is its prior probabilities."
    else
        "A view row naming one of them is dropped whole, and a group sheds them before its coefficient is spread."
    end
    return announce_non_investable(ni, ledger, "entropy pooling fit", consequence;
                                   warn = viewless)
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

  - `params...`: Further parameters of the statistic, forwarded to [`get_pr_value`](@ref). A tail risk view passes the observation weights the reference is read under here, and a relativistic value-at-risk view its deformation parameter.

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
    other = counterpart_axis_names(sets, sets.xkey)
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
            # A `prior(...)` reference to an asset on the Non-Investable Axis is a name the
            # caller wrote correctly and the data moved. Its row goes whole and in silence,
            # as a written-out name's does — so the reference is rewritten to the bare name
            # and `get_linear_constraints` drops the row by the counterpart rule, which is
            # also where it is recorded, once. See ADR 0125.
            if m.captures[1] ∈ other
                vars = copy(variables)
                vars[i] = m.captures[1]
                return ParsingResult(vars, copy(coeffs), res.op, res.rhs, res.eqn)
            end
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

Replace the prior references of every view constraint of a group.

`replace_prior_views` applies [`replace_prior_views`](@ref) to each element of a vector of parsed view constraints, replacing prior references with their corresponding prior values. [`parse_equation`](@ref) answers a group of view equations with a vector of results, so this is the shape every caller in this file meets.

The loop is a comprehension rather than a broadcast. Every parameter after `res` is one value for the whole group, and a broadcast reads a `Tuple` or a vector of observation weights as a container to walk beside `res` instead. That raises a `DimensionMismatch` whenever a group holds more than one equation, which is the shape this method exists for.

# Algorithm

 1. Call the single-view method once per element of `res`, forwarding `args...` and `kwargs...` to each call.
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
    return [replace_prior_views(resi, args...; kwargs...) for resi in res]
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
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false` every row of the group can drop, and `lcs` is then `nothing`: the group states no view, and the call returns without adding a row.
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
                      pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false,
                      ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    mu_views = parse_equation(mu_views.val; datatype = eltype(X))
    mu_views = replace_group_by_assets(mu_views, sets, false, true, false; ledger = ledger)
    mu_views = replace_prior_views(mu_views, pr, sets, :mu; strict = strict)
    lcs = get_linear_constraints(mu_views, sets; datatype = eltype(X), strict = strict,
                                 ledger = ledger)
    #! Under `strict = false` a view that names no asset is warned about and dropped, and
    #! a group whose every row drops parses to `nothing`. The warning is the whole
    #! diagnosis, so the family states no view and the fit proceeds without one. Reading a
    #! block off the `nothing` raised a `FieldError` naming an internal field one call after
    #! that warning. See issue #852.
    if isnothing(lcs)
        return nothing
    end
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
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Number,
                 w::Option{<:ObsWeights} = nothing)

Read the prior **value at risk** of asset `i` at the level `alpha`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:var)` names, by applying [`ValueatRisk`](@ref) to the `i`-th column of `pr.X`. That is the ``\\alpha``-quantile of the loss series under `w`, the observation weights the initial prior result was read at. A caller who states a non-uniform `w` reads the reference off the distribution that caller's own prior carries.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:var}`: Dispatch tag for VaR extraction.
  - `alpha`: Confidence level (e.g., `0.05` for 5% VaR).
  - $(arg_dict[:oow])

# Returns

  - `var::Number`: Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Number,
                      w::Option{<:ObsWeights} = nothing)
    return ValueatRisk(; alpha = alpha, w = w)(view(pr.X, :, i))
end
