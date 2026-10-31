"""
    add_ep_constraint!(epc::AbstractDict, lhs::MatNum, rhs::VecNum, key::Symbol)

Add an entropy pooling view constraint to the constraint dictionary.

`add_ep_constraint!` normalises a constraint block and adds it to the entropy pooling constraint dictionary `epc`. If `epc` already holds a block under the same key, the function appends the new block to that block.

Every view that is linear in the posterior probabilities goes into `epc` as a block ``(\\mathbf{A},\\, \\boldsymbol{B})`` of the system ``\\mathbf{A} \\boldsymbol{p} = \\boldsymbol{B}`` or ``\\mathbf{A} \\boldsymbol{p} \\leq \\boldsymbol{B}``. The key names the sense of the block, and the optimiser reads the block by that key.

# Mathematical definition

The function divides the block by the Frobenius norm of its left-hand side. This does not change the rows that the block states:

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
 2. Divide `lhs` and `rhs` by `sc`. The result is the normalised block.
 3. Store the pair under `key` when `epc` holds no such key.
 4. Otherwise, stack the normalised `lhs` under the block that `epc` holds, and append `rhs` to the right-hand side of that block.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `lhs`: Left-hand side constraint matrix.
  - `rhs`: Right-hand side constraint vector.
  - `key`: Key of the sense of the block, one of `:eq`, `:ineq`, `:feq` and `:cvar_eq`.

# Returns

  - `nothing`: The function mutates `epc` in place.

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
    ep_assert_attainable_rows(lhs::MatNum, rhs::VecNum, key::Symbol,
                              w::Option{<:VecNum}, label, name::AbstractString)

Refuse a view row that no reweighting of the scenarios can meet.

A row of a block in the entropy pooling constraint dictionary states ``\\sum_{t} p_{t} y_{t} \\lessgtr b``, where ``y`` is the row of `lhs`. The posterior probabilities sum to one, so the left-hand side is an average of ``y``, and it is between the smallest and the largest entry of ``y``. The posterior of entropy pooling puts no mass where the prior puts none, so only the entries at a positive prior probability count. For each row, the band is exact. A target on the edge of the band needs all the mass on the scenarios at that edge. The dual of the problem then has no finite solution, so the method refuses the edge too, as [`ep_assert_reachable_view`](@ref) does for a tail view. A row that is constant on the support has no edge: every posterior meets it when the prior does, and none meets it otherwise.

Without this check, the dual solve of such a row runs to an unbounded multiplier, and the posterior weights overflow. The first consumer of the weights then fails with an error that names neither the view nor the bound.

# Algorithm

 1. For each row `k` of `lhs`, read `lo` and `hi` with `extrema` over a view of the row. When `w` is not `nothing`, read only the entries at `w > 0`. Neither step copies the row or makes a mask.
 2. When `lo == hi`, the row is constant on the support. When `key` is `:eq`, raise an error unless `rhs[k] == lo`. Otherwise, raise an error unless `lo <= rhs[k]`.
 3. When `lo < hi` and `key` is `:eq`, raise an error unless `lo < rhs[k] < hi`. Otherwise, the row is `<=`, and raise an error unless `lo < rhs[k]`.

# Arguments

  - `lhs`: Left-hand side block, one row for each view and one column for each scenario.
  - `rhs`: Right-hand side vector, one target for each row of `lhs`.
  - `key`: Sense of the block, `:eq` or `:ineq`. An `:ineq` row is `<=`.
  - `w`: Prior probabilities of the scenarios, or `nothing` to count every scenario. A caller resolves a [`DynamicAbstractWeights`](@ref) with [`get_observation_weights`](@ref) first.
  - `label`: Callable that takes the index `k` of a row and returns its view, for the error message. The method calls it only for a row that it refuses.
  - `name`: The name of the view family, for the error message.

# Validation

  - If `key` is `:eq`, `lo < rhs[k] < hi` for every row `k`. Else `lo < rhs[k]`. A row that is constant on the support, `lo == hi`, needs `rhs[k] == lo` under `:eq` and `lo <= rhs[k]` under `:ineq`. A row that breaks this raises a `DomainError` that names the view and the band.

# Returns

  - `nothing`.

# Related

  - [`ep_assert_reachable_view`](@ref)
  - [`add_ep_constraint!`](@ref)
  - [`ep_mu_views!`](@ref)
"""
function ep_assert_attainable_rows(lhs::MatNum, rhs::VecNum, key::Symbol,
                                   w::Option{<:VecNum}, label, name::AbstractString)
    for k in eachindex(rhs)
        row = view(lhs, k, :)
        # Only the support of `w` bounds the band: a posterior puts no mass where the prior
        # puts none (#1260).
        lo, hi = if isnothing(w)
            extrema(row)
        else
            extrema(y for (y, wt) in zip(row, w) if wt > zero(wt))
        end
        b = rhs[k]
        # A row that is constant on the support holds at every posterior when the prior
        # meets it, so its edge is no edge.
        flat = b == lo == hi
        @argcheck((lo < b || flat) && (key != :eq || b < hi || flat),
                  DomainError(b,
                              "The $(name) view `$(label(k))` cannot be met: its row states a weighted average of the scenarios $(ifelse(key == :eq, "==", "<=")) $(b), and every reweighting of the scenarios at a positive prior probability keeps that average $(ifelse(lo == hi, "at $(lo)", "inside the open interval ($(lo), $(hi))")). Move the target into that band, or use a prior whose scenarios spread wider."))
    end
    return nothing
end
"""
    ep_view_row_label(A::MatNum, B::VecNum, k::Integer, key::Symbol, sets::UniverseSets)

Write the view that one row of a linear constraint block states.

A block of [`get_linear_constraints`](@ref) keeps no equation for its rows. This method writes row `k` again from its coefficients, the names of the universe and its target, for the error message of [`ep_assert_attainable_rows`](@ref). An `:ineq` row is `<=`, because the parser negates a `>=` row.

# Arguments

  - `A`: Coefficient matrix of the block, one row for each view and one column for each asset.
  - `B`: Target of each row.
  - `k`: Index of the row.
  - `key`: Sense of the block, `:eq` or `:ineq`.
  - `sets`: Universe sets that name the columns of `A`.

# Returns

  - `label::String`: The view of row `k`, such as `"1.0*A1 == 0.003"`.

# Related

  - [`ep_assert_attainable_rows`](@ref)
"""
function ep_view_row_label(A::MatNum, B::VecNum, k::Integer, key::Symbol,
                           sets::UniverseSets)
    nx = sets.dict[sets.xkey]
    terms = join(("$(A[k, i])*$(nx[i])" for i in axes(A, 2) if !iszero(A[k, i])), " + ")
    return "$(terms) $(ifelse(key == :eq, "==", "<=")) $(B[k])"
end
"""
    announce_ep_departures(ni::VecStr, ledger::VecStr, viewless::Bool) -> Nothing

Report, once for each entropy pooling fit, the assets that left the investable universe and the views that their departure removed.

It is the call of [`announce_non_investable`](@ref) for the entropy pooling family. The four [`ep_prior`](@ref) methods call it, so all four use the same words. It names the process an entropy pooling fit. The default message names an optimisation, which is wrong for a call of `prior(pe, X)` outside an optimisation.

The fit calls it at its end, not at the reduction. A staged algorithm runs its view builders between its solves, so the Departure Ledger is complete only after the last stage states its views. An earlier report misses the departures of the later stages, and a report for each stage gives three messages for one departure.

`viewless` makes the message a warning. If the departures removed the last view, the fit has no view, and the posterior is the prior probabilities. That is not the result that the caller asked for. Every other departure removes only a part of the view set, and its message is an `@info`.

# Arguments

  - `ni`: The names that the Investable Mask left out, from [`investable_views`](@ref).
  - `ledger`: The Departure Ledger, which the view builders filled.
  - `viewless`: Whether the fit has no view left because of the departures.

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

Replace the prior references of a parsed view with their prior values.

`replace_prior_views` finds each reference of the form `prior(A)` in a parsed view constraint, a [`ParsingResult`](@ref). It replaces the reference with the prior value that `pr` gives. The optimisation then reads such a term as a constant, not as a variable.

# Mathematical definition

A parsed view is the row ``\\sum_{k} c_{k} v_{k} \\lessgtr b``. A term whose variable is `prior(a)` holds the constant ``\\pi_{a}``, not an unknown. The function moves each such term to the right-hand side, which gives an equivalent row over the other terms:

```math
\\sum_{k \\notin \\mathcal{P}} c_{k} v_{k} \\lessgtr b - \\sum_{k \\in \\mathcal{P}} c_{k} \\pi_{a_{k}}\\,.
```

Where:

  - ``c_{k}``, ``v_{k}``: Coefficient and variable of the ``k``-th term of the view.
  - ``b``: Right-hand side of the view.
  - ``\\mathcal{P}``: Terms whose variable is a `prior(...)` reference.
  - ``\\pi_{a}``: Prior value of the statistic `key` for asset ``a``, which [`get_pr_value`](@ref) reads at the level `alpha` and with the parameters `params...`.

# Algorithm

 1. Match the pattern `prior(<asset>)` against the variable of each term in turn.
 2. When a term does not match, record that the view keeps a variable of its own, and take the next term.
 3. Find the named asset in the universe. If the asset is on the Non-Investable Axis, write the bare name in place of the reference and return. [`get_linear_constraints`](@ref) then drops the row.
 4. If the universe does not hold the asset, report it through `strict_diagnostic`, record the term for removal, and take the next term.
 5. Subtract [`get_pr_value`](@ref) times the coefficient of the term from `rhs`, and record the term for removal.
 6. Return `res` unchanged when step 4 and step 5 recorded no term.
 7. Drop the recorded terms from `vars` and `coef`, make the equation string again, and return a [`ParsingResult`](@ref) that holds the adjusted `rhs`.

# Arguments

  - `res`: Parsed view constraint, with its variables and coefficients.

  - `pr`: Prior result that gives the prior values.

  - `sets`: Universe sets that map the asset names to indices.

  - `key`: Tag of the statistic, such as `:mu`, `:var` or `:cvar`.

  - `alpha`: Significance level of a tail view, or `nothing`.

  - `params...`: More parameters of the statistic, which the method passes to [`get_pr_value`](@ref). A tail view passes here the observation weights at which the method reads the reference. A relativistic value-at-risk view also passes its deformation parameter.

  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Validation

  - A `prior(...)` reference to an asset that the universe does not hold raises an `ArgumentError` when `strict` is `true`, and warns otherwise. The method drops the term in both cases.
  - At least one term of the view must keep a variable of its own. A view in which every term is a `prior(...)` reference states only constants, and raises an `ArgumentError`.

# Returns

  - `res::ParsingResult`: The parsing result, with each prior reference replaced by its value.

# Related

  - [`ParsingResult`](@ref)
  - [`get_pr_value`](@ref): reads the prior value that replaces each reference.
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
            strict_diagnostic(msg, strict; id = @log_id(m.captures[1], sets.xkey))
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

`replace_prior_views` applies [`replace_prior_views`](@ref) to each parsed view constraint of a vector. [`parse_equation`](@ref) returns a vector of results for a group of view equations, so every view verb gets this shape.

The loop is a comprehension, not a broadcast. Each parameter after `res` is one value for the full group. A broadcast reads a `Tuple` or a vector of observation weights as a container to iterate together with `res`. That raises a `DimensionMismatch` when a group holds more than one equation, which is the case that this method is for.

# Algorithm

 1. Call the method for one view once for each element of `res`, with `args...` and `kwargs...`.
 2. Return the vector of the results, one for each element of `res`, in the order of `res`.

# Arguments

  - `res`: Vector of parsed view constraints.
  - `args...`: The positional arguments that the method passes to [`replace_prior_views`](@ref).
  - `kwargs...`: The keyword arguments that the method passes to [`replace_prior_views`](@ref).

# Returns

  - `res::Vector{<:ParsingResult}`: One parsing result for each element of `res`, with each prior reference replaced by its value.

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

Read the prior mean of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic that the tag `Val(:mu)` names, the `i`-th entry of `pr.mu`. [`replace_prior_views`](@ref) calls it.

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Index of the asset.
  - `::Val{:mu}`: Dispatch tag of the mean.
  - `args...`: More arguments, which the method does not read.

# Returns

  - `mu::Number`: Prior mean of asset `i`.

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

Do nothing when a problem states no mean view.

`ep_mu_views!` is the verb that turns a group of mean views into rows of the entropy pooling constraint dictionary. This method is the branch for an absent view. It registers no row, so a caller can call the verb with `mu_views = nothing` and needs no special case.

# Arguments

  - `mu_views::Nothing`: Shows that the problem states no mean view.
  - `args...`: More positional arguments, which the method does not read.
  - `kwargs...`: More keyword arguments, which the method does not read.

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
                 pr::AbstractPriorResult, sets::UniverseSets,
                 w::Option{<:ObsWeights} = nothing; strict::Bool = false,
                 ledger::Option{<:AbstractVector} = nothing)

Add the mean views of a group to the entropy pooling constraint dictionary.

`ep_mu_views!` parses the mean view equations of a [`LinearConstraintEstimator`](@ref), replaces each prior reference with its value, and makes the linear constraints of the views. It adds the constraints to the entropy pooling constraint dictionary `epc`. The statistic is the mean of the posterior distribution. The mean is linear in the posterior probabilities, so the view needs no auxiliary variable and fixes no moment.

# Mathematical definition

The posterior mean of an asset is the average of its returns, weighted by the posterior probabilities. A view states a linear combination of such means:

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
  - ``a_{k,\\,i}``: Coefficient that asset ``i`` takes in view ``k``, the ``(k, i)`` entry of ``\\mathbf{A}``.
  - ``K``: Number of views that the group states.
  - ``\\mathbf{X}``: ``T \\times N`` matrix of the returns of the prior.

# Algorithm

 1. Parse the view equations of `mu_views.val` into one [`ParsingResult`](@ref) for each view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior mean, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false`, every row of the group can drop, and `lcs` is then `nothing`. The group then states no view, and the call returns with no row added.
 5. For each block that `lcs` holds, make `A * transpose(X)`. Refuse a row that no reweighting of the scenarios on the support of `w` meets, with [`ep_assert_attainable_rows`](@ref).
 6. Add `A * transpose(X)` against `B` under its key with [`add_ep_constraint!`](@ref).

# Arguments

  - `mu_views`: Mean view constraints.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `pr`: Prior result that holds the asset returns.
  - `sets`: Universe sets that map the asset names to indices.
  - $(arg_dict[:ep_view_support_w])
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.
  - $(arg_dict[:ledger]) [`replace_group_by_assets`](@ref) and [`get_linear_constraints`](@ref) fill it.

# Validation

  - $(val_dict[:ep_attainable_row])

# Returns

  - `nothing`: The function mutates `epc` in place.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior mean that a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref): holds a mean at the prior value when a view on a higher moment can move it.
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_mu_views!(mu_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::UniverseSets,
                      w::Option{<:ObsWeights} = nothing; strict::Bool = false,
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
    ws = get_observation_weights(w, X)
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        (; A, B) = getproperty(lcs, p)
        # Row `k` of `A * transpose(X)` is the scenario series of the left-hand side of
        # view `k`, so its band is exact for a view of one asset and of many (#1920).
        lhs = A * transpose(X)
        ep_assert_attainable_rows(lhs, B, p, ws, k -> ep_view_row_label(A, B, k, p, sets),
                                  "mean")
        add_ep_constraint!(epc, lhs, B, p)
    end
    return nothing
end
"""
    fix_mu!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
            pr::AbstractPriorResult)

Hold the mean of the named assets at the prior value.

`fix_mu!` finds the assets that `to_fix` names and `fixed` does not. It adds rows to the entropy pooling constraint dictionary `epc` that hold the mean of these assets at the prior value. With these rows, a view on a higher moment, such as the variance or the skewness, does not move the mean of these assets. The function marks the new assets in `fixed`.

The rows go under the `:feq` key. The optimiser relaxes the rows of this key with a penalised slack, and does not enforce them exactly. A fixing row is not a view. If a fixing row and a view cannot both hold, the optimiser lets the fixing row miss its target.

# Mathematical definition

The rows hold the posterior mean of each named asset at the prior mean:

```math
\\sum_{t=1}^{T} p_{t} x_{t,\\,i} = \\mu_{i}\\,, \\quad \\forall\\, i \\in \\mathcal{F}\\,.
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:T])
  - ``\\mathcal{F}``: Assets that `to_fix` names and `fixed` does not already hold.

# Algorithm

 1. Read the assets that `to_fix` names and `fixed` does not already hold into `fix`.
 2. Return when `fix` names no asset.
 3. Add one `:feq` row for each named asset, `transpose(view(pr.X, :, fix))` against `pr.mu[fix]`, with [`add_ep_constraint!`](@ref).
 4. Mark the named assets in `fixed`, so a later call adds no second row for them.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `fixed`: Boolean vector that marks the assets whose mean is already fixed.
  - `to_fix`: Boolean vector that marks the assets whose mean must be fixed.
  - `pr`: Prior result that holds the asset returns.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in place.

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

Read the prior value at risk of asset `i` at the level `alpha`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic that the tag `Val(:var)` names. It applies [`ValueatRisk`](@ref) to the `i`-th column of `pr.X`. That is the ``\\alpha``-quantile of the loss series under `w`, the observation weights of the first prior result. With a non-uniform `w`, the method reads the reference from the distribution that the prior of the caller has.

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Index of the asset.
  - `::Val{:var}`: Dispatch tag of the value at risk.
  - `alpha`: Significance level, for example `0.05` for the value at risk at 5%.
  - $(arg_dict[:oow])

# Returns

  - `var::Number`: Prior value at risk of asset `i` at the level `alpha`.

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
