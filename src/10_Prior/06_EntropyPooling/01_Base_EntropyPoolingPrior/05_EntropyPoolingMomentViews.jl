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
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:cvar}, alpha::Number,
                 w::Option{<:ObsWeights} = nothing)

Read the prior **conditional value at risk** of asset `i` at the level `alpha`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic the tag `Val(:cvar)` names, by applying [`ConditionalValueatRisk`](@ref) to the `i`-th column of `pr.X`. That is the conditional value at risk of the loss series under `w`, the observation weights the initial prior result was read at, and it rests on no distributional assumption. It reads `w` on the reasoning [`get_pr_value`](@ref) gives for the value at risk.

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:cvar}`: Dispatch tag for CVaR computation.
  - `alpha`: Confidence level.
  - $(arg_dict[:oow])

# Returns

  - `cvar::Number`: Conditional Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:cvar}, alpha::Number,
                      w::Option{<:ObsWeights} = nothing)
    return ConditionalValueatRisk(; alpha = alpha, w = w)(view(pr.X, :, i))
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
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false` every row of the group can drop, and `lcs` is then `nothing`: the group states no view, and the call returns a `to_fix` that names no asset.
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
                         pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false,
                         ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    sigma_views = parse_equation(sigma_views.val; datatype = eltype(X))
    sigma_views = replace_group_by_assets(sigma_views, sets, false, true, false;
                                          ledger = ledger)
    sigma_views = replace_prior_views(sigma_views, pr, sets, :sigma; strict = strict)
    lcs = get_linear_constraints(sigma_views, sets; datatype = eltype(X), strict = strict,
                                 ledger = ledger)
    #! Under `strict = false` a view that names no asset is warned about and dropped, and
    #! a group whose every row drops parses to `nothing`. The warning is the whole
    #! diagnosis, so the family states no view and the fit proceeds without one. Reading a
    #! block off the `nothing` raised a `FieldError` naming an internal field one call after
    #! that warning. See issue #852.
    if isnothing(lcs)
        return falses(size(X, 2))
    end
    tmp = transpose((X .- transpose(pr.mu)) .^ 2)
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(!iszero, A; dims = 1); dims = 1)
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
                               sets::UniverseSets, key::Symbol; strict::Bool = false,
                               ledger::Option{<:AbstractVector} = nothing)
    prior_pattern = r"prior\(([^()]*)\)"
    prior_corr_pattern = r"prior\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    nx = sets.dict[sets.xkey]
    other = counterpart_axis_names(sets, sets.xkey)
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
                asset1 = split(@view(n.captures[1][2:(end - 1)]), ", ")
                asset2 = split(@view(n.captures[2][2:(end - 1)]), ", ")
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
            asset1 = split(@view(n.captures[1][2:(end - 1)]), ", ")
            asset2 = split(@view(n.captures[2][2:(end - 1)]), ", ")
            # A pair group sheds its departed pairs before it is written out, so a departed
            # name reaching here means the group lost *every* pair and `replace_group_by_assets`
            # kept one to say so. The row goes whole, in silence. See ADR 0125.
            if any(∈(other), asset1) || any(∈(other), asset2)
                record_non_investable_drop!(ledger, "the row `$(res.eqn)`")
                return RhoParsingResult(empty(variables), empty(coeffs), res.op, res.rhs,
                                        res.eqn, empty(jk_idx))
            end
            j = [findfirst(x -> x == a1, nx) for a1 in asset1]
            k = [findfirst(x -> x == a2, nx) for a2 in asset2]
        else
            j = findfirst(x -> x == asset1, nx)
            k = findfirst(x -> x == asset2, nx)
            # A pair naming an asset on the Non-Investable Axis is a correlation the data no
            # longer holds. The row goes whole and in silence, and the emptied variable list
            # is how it says so to the builder one call later. See ADR 0125.
            if asset1 ∈ other || asset2 ∈ other
                record_non_investable_drop!(ledger, "the row `$(res.eqn)`")
                return RhoParsingResult(empty(variables), empty(coeffs), res.op, res.rhs,
                                        res.eqn, empty(jk_idx))
            end
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
                       pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false,
                       ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    cov_views = parse_equation(cov_views.val; datatype = eltype(X))
    cov_views = replace_group_by_assets(cov_views, sets, false, true, true; ledger = ledger)
    cov_views = replace_coprior_views(cov_views, pr, sets, :cov; strict = strict,
                                      ledger = ledger)
    to_fix = falses(size(X, 2))
    for cov_view in cov_views
        #! `replace_coprior_views` drops a pair naming an asset the universe does not hold,
        #! which leaves the view with no pair at all. Under `strict = false` that must drop
        #! the row, as `get_linear_constraints` does for the linear families. Without this
        #! the guard below raised, called a view of no pairs one of several, and named an
        #! equation with no variable in it. The drop is silent: an unknown name was already
        #! named by `replace_coprior_views`, and a departed one is silent by ADR 0125 and
        #! has already told the door's ledger.
        if isempty(cov_view.vars)
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
                       pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false,
                       ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    rho_views = parse_equation(rho_views.val; datatype = eltype(X))
    rho_views = replace_group_by_assets(rho_views, sets, false, true, true; ledger = ledger)
    rho_views = replace_coprior_views(rho_views, pr, sets, :rho; strict = strict,
                                      ledger = ledger)
    to_fix = falses(size(X, 2))
    sigma = LinearAlgebra.diag(pr.sigma)
    for rho_view in rho_views
        #! See the twin note in `ep_cov_views!`: a pair naming an unknown asset leaves the
        #! view with no pair, and that drops the row, in silence.
        if isempty(rho_view.vars)
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
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false` every row of the group can drop, and `lcs` is then `nothing`: the group states no view, and the call returns a `to_fix` that names no asset.
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
                      pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false,
                      ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    skew_views = parse_equation(skew_views.val; datatype = eltype(X))
    skew_views = replace_group_by_assets(skew_views, sets, false, true, false;
                                         ledger = ledger)
    skew_views = replace_prior_views(skew_views, pr, sets, :skew; strict = strict)
    lcs = get_linear_constraints(skew_views, sets; datatype = eltype(X), strict = strict,
                                 ledger = ledger)
    #! Under `strict = false` a view that names no asset is warned about and dropped, and
    #! a group whose every row drops parses to `nothing`. The warning is the whole
    #! diagnosis, so the family states no view and the fit proceeds without one. Reading a
    #! block off the `nothing` raised a `FieldError` naming an internal field one call after
    #! that warning. See issue #852.
    if isnothing(lcs)
        return falses(size(X, 2))
    end
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
        to_fix .= to_fix .| dropdims(any(!iszero, A; dims = 1); dims = 1)
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
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false` every row of the group can drop, and `lcs` is then `nothing`: the group states no view, and the call returns a `to_fix` that names no asset.
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
                      pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false,
                      ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    kurtosis_views = parse_equation(kurtosis_views.val; datatype = eltype(X))
    kurtosis_views = replace_group_by_assets(kurtosis_views, sets, false, true, false;
                                             ledger = ledger)
    kurtosis_views = replace_prior_views(kurtosis_views, pr, sets, :kurtosis;
                                         strict = strict)
    lcs = get_linear_constraints(kurtosis_views, sets; datatype = eltype(X),
                                 strict = strict, ledger = ledger)
    #! Under `strict = false` a view that names no asset is warned about and dropped, and
    #! a group whose every row drops parses to `nothing`. The warning is the whole
    #! diagnosis, so the family states no view and the fit proceeds without one. Reading a
    #! block off the `nothing` raised a `FieldError` naming an internal field one call after
    #! that warning. See issue #852.
    if isnothing(lcs)
        return falses(size(X, 2))
    end
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
        to_fix .= to_fix .| dropdims(any(!iszero, A; dims = 1); dims = 1)
    end
    return to_fix
end

export RhoParsingResult
