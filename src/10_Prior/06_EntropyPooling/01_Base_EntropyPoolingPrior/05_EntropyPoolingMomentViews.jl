"""
$(DocStringExtensions.TYPEDEF)

Holds a parsed correlation or covariance view and the asset pairs that it names.

It adds an `ij` field to [`ParsingResult`](@ref). The field holds one index pair for each term of the view, so a later routine can place the view in the covariance matrix and does not parse the equation again. [`replace_coprior_views`](@ref) makes it from a view of the form `"(A, B) == 0.5"`. [`ep_cov_views!`](@ref) and [`ep_rho_views!`](@ref) read it.

A view over a pair of groups spans one asset pair for each element of its `ij` entry, and makes one constraint row for each pair. Its `rhs` is therefore a vector of the same length, with one right-hand side for each row. A view over one asset pair keeps a scalar `rhs`.

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

Positional arguments correspond to the struct's fields. The type has no keyword constructor, because [`replace_coprior_views`](@ref) makes it.

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
  - [`replace_coprior_views`](@ref): makes this type.
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

Read the prior conditional value at risk of asset `i` at the level `alpha`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic that the tag `Val(:cvar)` names. It applies [`ConditionalValueatRisk`](@ref) to the `i`-th column of `pr.X`. That is the conditional value at risk of the loss series under `w`, the observation weights of the first prior result. It needs no assumption about the distribution. It reads `w` for the same reason as the value at risk method of [`get_pr_value`](@ref).

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Index of the asset.
  - `::Val{:cvar}`: Dispatch tag of the conditional value at risk.
  - `alpha`: Significance level.
  - $(arg_dict[:oow])

# Returns

  - `cvar::Number`: Prior conditional value at risk of asset `i` at the level `alpha`.

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

Read the prior variance of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic that the tag `Val(:sigma)` names, the `i`-th diagonal entry of `pr.sigma`. The tag is `:sigma` and the statistic is the variance, not the standard deviation and not the value at risk, which the tag `:var` names.

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Index of the asset.
  - `::Val{:sigma}`: Dispatch tag of the variance.
  - `args...`: More arguments, which the method does not read.

# Returns

  - `sigma::Number`: Prior variance of asset `i`.

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
    scenario_residual_variances(pr::AbstractPriorResult)
    scenario_residual_variances(rr, sigma::VecNum)
    scenario_residual_variances(rr::Regression, sigma::VecNum)

Return the residual variance of each asset that `pr.sigma` has and the scenarios `pr.X` do not have.

A time-series factor prior makes its scenarios from the regression, `F * transpose(M) .+ transpose(b)`, and adds the residual block only to the covariance. [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) do this under `rsd = true`. The variance of asset ``i`` in `pr.sigma` is then the variance of its scenarios plus ``\\delta_{i}``. An entropy pooling view row reads the scenarios. So a row that states a moment of `pr.sigma` subtracts ``\\delta_{i}`` from its target, and a row that reads the variance of the scenarios reads ``\\sigma_{i}^{2} - \\delta_{i}``. [`scenario_factory`](@ref) keeps the residual block of this fit in the posterior, so the posterior covariance meets the target of the view.

A prior with no regression adds no block to the scenarios. A cross-sectional factor prior also adds none, because its scenarios hold the idiosyncratic returns. For these priors, the result is zero.

# Mathematical definition

```math
\\sigma_{i}^{2} = \\mathrm{Var}[x_{i}] + \\delta_{i}\\,,
```

Where:

  - ``\\sigma_{i}^{2}``: The variance of asset ``i`` in `pr.sigma`.
  - ``\\mathrm{Var}[x_{i}]``: The variance of the scenarios of asset ``i``.
  - ``\\delta_{i}``: The residual variance of asset ``i``, which the function returns.

# Algorithm

The method that Julia selects is the algorithm.

 1. A prior result: apply this function to `pr.rr` and the diagonal of `pr.sigma`.
 2. A [`Regression`](@ref) whose `esigma` is set: return its [`idiosyncratic_variances`](@ref).
 3. Any other value: return zero for each asset.

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `rr`: The loadings result of `pr`, or `nothing`.
  - `sigma`: The variances of `pr.sigma`, which set the type and the size of a zero result.

# Returns

  - `d::VecNum`: The residual variance of each asset.

# Related

  - [`ep_sigma_views!`](@ref)
  - [`fix_sigma!`](@ref)
  - [`scenario_factory`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
function scenario_residual_variances(pr::AbstractPriorResult)
    return scenario_residual_variances(pr.rr, LinearAlgebra.diag(pr.sigma))
end
function scenario_residual_variances(::Any, sigma::VecNum)
    return zero(sigma)
end
function scenario_residual_variances(rr::Regression, sigma::VecNum)
    return isnothing(rr.esigma) ? zero(sigma) : idiosyncratic_variances(rr)
end
"""
    ep_sigma_views!(sigma_views::LinearConstraintEstimator, epc::AbstractDict,
                    pr::AbstractPriorResult, sets::UniverseSets; strict::Bool = false)

Add the variance views of a group to the entropy pooling constraint dictionary.

`ep_sigma_views!` parses the variance view equations of a [`LinearConstraintEstimator`](@ref), replaces each prior reference with its value, and makes the constraints of the views. It adds the constraints to the entropy pooling constraint dictionary `epc`. The statistic is the variance of the posterior distribution, not the value at risk, which the `var_views` family holds.

The variance is quadratic in the returns and linear in the posterior probabilities only when the mean is a constant. So this method returns the assets whose mean [`fix_mu!`](@ref) must hold at the prior. A variance view then does not move the mean about which the view measures the variance.

# Mathematical definition

The row states the posterior second central moment about the prior mean, plus the residual variance that the scenarios do not have. This sum is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{Var}_{\\boldsymbol{p}}[x_{i}] &= \\sum_{t=1}^{T} p_{t} \\left(x_{t,\\,i} - \\mu_{i}\\right)^{2} + \\delta_{i}\\,, \\\\
\\sum_{i=1}^{N} a_{k,\\,i} \\mathrm{Var}_{\\boldsymbol{p}}[x_{i}] &\\lessgtr B_{k}\\,, \\quad \\forall\\, k = 1,\\ldots,K\\,.
\\end{align}
```

The identity holds only while the posterior mean of asset ``i`` equals ``\\mu_{i}``. So this method gives the assets that it names to [`fix_mu!`](@ref). The constant ``\\delta_{i}`` moves to the right-hand side, so the row compares the scenarios against ``B_{k} - \\sum_{i} a_{k,\\,i} \\delta_{i}``.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:B])
  - ``a_{k,\\,i}``: Coefficient that asset ``i`` takes in view ``k``.
  - ``K``: Number of views that the group states.
  - $(math_dict[:ep_delta_i])

# Algorithm

 1. Parse the view equations of `sigma_views.val` into one [`ParsingResult`](@ref) for each view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior variance, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false`, every row of the group can drop, and `lcs` is then `nothing`. The group then states no view, and the call returns a `to_fix` that names no asset.
 5. Build `tmp`, the squared deviations of every observation from the prior mean, transposed so that a row of `lcs` multiplies it from the left. Read the residual variances `d` with [`scenario_residual_variances`](@ref).
 6. For each block that `lcs` holds, add `A * tmp` against `B - A * d` under its key with [`add_ep_constraint!`](@ref), and mark in `to_fix` every asset that the block names.

# Arguments

  - `sigma_views`: Variance view constraints.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `pr`: Prior result that holds the asset returns.
  - `sets`: Universe sets that map the asset names to indices.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Returns

  - `to_fix::BitVector`: Boolean vector that marks the assets whose mean must be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior variance that a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref): reads the `to_fix` that this method returns.
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
    # The view states the variance that `pr.sigma` reports, and the rows hold no residual
    # block, so the block comes off the target (#1649).
    d = scenario_residual_variances(pr)
    to_fix = falses(size(X, 2))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B - A * d, p)
        to_fix .= to_fix .| dropdims(any(!iszero, A; dims = 1); dims = 1)
    end
    return to_fix
end
"""
    fix_sigma!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
               pr::AbstractPriorResult)

Hold the variance of the named assets at the prior value.

`fix_sigma!` finds the assets that `to_fix` names and `fixed` does not. It adds rows to the entropy pooling constraint dictionary `epc` that hold the variance of these assets at the prior value. With these rows, a view on a higher moment, such as the skewness or the correlation, does not move the variance of these assets. The function marks the new assets in `fixed`.

The rows go under the `:feq` key. The optimiser relaxes the rows of this key with a penalised slack, and does not enforce them exactly. A fixing row is not a view. If a fixing row and a view cannot both hold, the optimiser lets the fixing row miss its target.

The left side is the uncorrected moment of the scenarios, and the right side reads the diagonal of `pr.sigma`. The two agree only when the covariance estimator of the prior is uncorrected. Under a corrected estimator, the row holds the variance at the corrected value, which is larger by the correction factor.

# Mathematical definition

The rows hold the posterior second central moment of each named asset, about the prior mean, at the prior variance of its scenarios:

```math
\\sum_{t=1}^{T} p_{t} \\left(x_{t,\\,i} - \\mu_{i}\\right)^{2} = \\sigma_{i}^{2} - \\delta_{i}\\,, \\quad \\forall\\, i \\in \\mathcal{F}\\,.
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:ep_delta_i])
  - $(math_dict[:T])
  - ``\\mathcal{F}``: Assets that `to_fix` names and `fixed` does not already hold.

# Algorithm

 1. Read the prior variances of the scenarios into `sigma`. They are the diagonal of `pr.sigma`, less [`scenario_residual_variances`](@ref).
 2. Read the assets that `to_fix` names and `fixed` does not already hold into `fix`.
 3. Return when `fix` names no asset.
 4. Add one `:feq` row for each named asset, the squared deviations of that asset from its prior mean against `sigma[fix]`, with [`add_ep_constraint!`](@ref).
 5. Mark the named assets in `fixed`, so a later call adds no second row for them.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `fixed`: Boolean vector that marks the assets whose variance is already fixed.
  - `to_fix`: Boolean vector that marks the assets whose variance must be fixed.
  - `pr`: Prior result that holds the asset returns.

# Returns

  - `nothing`: The function mutates `epc` and `fixed` in place.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`ep_sigma_views!`](@ref)
  - [`fix_mu!`](@ref): the same rule, one moment lower.
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function fix_sigma!(epc::AbstractDict, fixed::AbstractVector, to_fix::BitVector,
                    pr::AbstractPriorResult)
    sigma = LinearAlgebra.diag(pr.sigma) - scenario_residual_variances(pr)
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

Replace the prior references of a parsed correlation or covariance view with their prior values.

`replace_coprior_views` finds each reference of the form `prior(A, B)` in a parsed correlation or covariance view constraint, a [`ParsingResult`](@ref). It replaces the reference with the prior value that `pr` gives. The optimisation then reads such a term as a constant, not as a variable.

It is the counterpart of [`replace_prior_views`](@ref) for asset pairs, and it returns a [`RhoParsingResult`](@ref), not a [`ParsingResult`](@ref). A pair view holds the index pair of every term, and the verb that places the view in the covariance matrix reads it.

# Mathematical definition

A parsed pair view is the row ``\\sum_{k} c_{k} v_{k} \\lessgtr b``, whose variables name asset pairs. A term whose variable is `prior(a, b)` holds a constant. The function moves each such term to the right-hand side, which gives an equivalent row:

```math
\\sum_{k \\notin \\mathcal{P}} c_{k} v_{k} \\lessgtr b - \\sum_{k \\in \\mathcal{P}} c_{k} \\pi_{a_{k},\\, b_{k}}\\,.
```

A `prior(gA, gB)` reference over a pair of groups holds one constant for each spanned asset pair. So the subtraction broadcasts, and ``b`` becomes a vector of that length, with one right-hand side for each row of the view.

Where:

  - ``c_{k}``, ``v_{k}``: Coefficient and variable of the ``k``-th term of the view.
  - ``b``: Right-hand side of the view.
  - ``\\mathcal{P}``: Terms whose variable is a `prior(...)` reference.
  - ``\\pi_{a,\\,b}``: Prior value of the statistic `key` for the asset pair ``(a, b)``, which [`get_pr_value`](@ref) reads.

# Algorithm

 1. Match the pattern `prior(<asset1>, <asset2>)` against the variable of each term in turn.
 2. When a term does not match, read its own pair `(a, b)` instead. Raise when the term is not of the form `(a, b)`.
 3. Find both names in the universe. A name in brackets is a group, and every other name is one asset. When the universe does not hold one of the names, report it through `strict_diagnostic`, record the term for removal, and take the next term.
 4. Record the index pair in `jk_idx`, and take the next term.
 5. For a term that does match, find both names in the same way. If a name is on the Non-Investable Axis, record the drop in the Departure Ledger, and return a [`RhoParsingResult`](@ref) with no term.
 6. Subtract [`get_pr_value`](@ref) times the coefficient of the term from `rhs` with the broadcasting operators, and record the term for removal.
 7. Return a [`RhoParsingResult`](@ref) over the unchanged terms when steps 3, 5 and 6 recorded no term.
 8. Drop the recorded terms from `vars` and `coef`, make the equation string again, and return a [`RhoParsingResult`](@ref) that holds the adjusted `rhs` and `jk_idx`.

# Arguments

  - `res`: Parsed correlation or covariance view constraint, with its variables and coefficients.
  - `pr`: Prior result that gives the prior values.
  - `sets`: Universe sets that map the asset names to indices.
  - `key`: Tag of the statistic, `:rho` for a correlation view or `:cov` for a covariance view.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Validation

  - Every term that is not a `prior(...)` reference must be of the form `(a, b)`. Any other form raises an `ArgumentError`.
  - Every `prior(...)` reference must be of the form `prior(a, b)`. Any other form raises an `ArgumentError`.
  - A reference to an asset that the universe does not hold raises an `ArgumentError` when `strict` is `true`, and warns otherwise. The method drops the term in both cases.
  - At least one term of the view must keep a variable of its own. A view in which every term is a `prior(...)` reference raises an `ArgumentError`.

# Returns

  - `res::RhoParsingResult`: The parsing result, with each prior reference replaced by its value, and with the index pairs of its terms.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`replace_prior_views`](@ref): the counterpart for single assets.
  - [`get_pr_value`](@ref): reads the prior value that replaces each reference.
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
                    strict_diagnostic(msg, strict; id = @log_id(asset1, sets.xkey))
                end
                if isnothing(k)
                    msg = unknown_variable_msg(asset2, nx, sets.xkey)
                    strict_diagnostic(msg, strict; id = @log_id(asset2, sets.xkey))
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
                strict_diagnostic(msg, strict; id = @log_id(asset1, sets.xkey))
            end
            if isnothing(k)
                msg = unknown_variable_msg(asset2, nx, sets.xkey)
                strict_diagnostic(msg, strict; id = @log_id(asset2, sets.xkey))
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

Replace the prior references of every pair view constraint of a group.

`replace_coprior_views` applies [`replace_coprior_views`](@ref) to each parsed view constraint of a vector. [`parse_equation`](@ref) returns a vector of results for a group of view equations, so every pair view verb gets this shape.

# Algorithm

 1. Broadcast the method for one view over `res`, with `args...` and `kwargs...`.
 2. Return the vector of the results, one [`RhoParsingResult`](@ref) for each element of `res`, in the order of `res`.

# Arguments

  - `res`: Vector of parsed view constraints.
  - `args...`: The positional arguments that the method passes to [`replace_coprior_views`](@ref).
  - `kwargs...`: The keyword arguments that the method passes to [`replace_coprior_views`](@ref).

# Returns

  - `res::Vector{<:RhoParsingResult}`: One parsing result for each element of `res`, with each prior reference replaced by its value.

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

Read the prior correlation or covariance of the asset pair `(i, j)`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. `Val(:cov)` names the entry `pr.sigma[i, j]`, and `Val(:rho)` names the same entry of `StatsBase.cov2cor(pr.sigma)`. [`replace_coprior_views`](@ref) calls them to resolve a `prior(a, b)` reference.

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Index of the first asset.
  - `j`: Index of the second asset.
  - `::Val{:rho}`: Dispatch tag of the correlation.
  - `::Val{:cov}`: Dispatch tag of the covariance.
  - `args...`: More arguments, which the method does not read.

# Returns

  - `val::Number`: Prior correlation or prior covariance of the assets `i` and `j`.

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

Read the prior correlations or covariances of the asset pairs that two groups span.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. These methods read the same statistics as the methods for one pair, once for each spanned pair, in the order of `zip(i, j)`. A view over a pair of groups makes one constraint row for each spanned pair. So a `prior(gA, gB)` reference inside such a view must give each row the prior value of its own pair.

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Vector of the indices of the first asset group.
  - `j`: Vector of the indices of the second asset group.
  - `::Val{:rho}`: Dispatch tag of the correlation.
  - `::Val{:cov}`: Dispatch tag of the covariance.
  - `args...`: More arguments, which the method does not read.

# Returns

  - `val::Vector{<:Number}`: Prior correlation or prior covariance of each spanned pair, with one entry for each element of `zip(i, j)`.

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

Add the covariance views of a group to the entropy pooling constraint dictionary.

`ep_cov_views!` parses the covariance view equations of a [`LinearConstraintEstimator`](@ref), replaces each prior reference with its value, and makes the linear constraints of the views. It adds the constraints to the entropy pooling constraint dictionary `epc`. The statistic is the covariance of the posterior distribution, not the correlation, which the `rho_views` family holds.

The covariance is a product of returns and is linear in the posterior probabilities only when both means are constants. So this method returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior. A covariance view then keeps the mean and the variance that the prior states. The variance is the one that the covariance estimator of the prior gives, as [`fix_sigma!`](@ref) states.

# Mathematical definition

The row states the posterior cross moment about the prior means, plus the residual block that the scenarios do not have. This sum is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{Cov}_{\\boldsymbol{p}}[x_{i}, x_{j}] &= \\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} - \\mu_{i} \\mu_{j} + \\mathbb{1}_{i = j}\\, \\delta_{i}\\,, \\\\
c\\, \\mathrm{Cov}_{\\boldsymbol{p}}[x_{i}, x_{j}] &\\lessgtr b\\,,
\\end{align}
```

which the method writes with the constants on the right:

```math
\\begin{align}
d\\, c \\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} &\\lessgtr d \\left(c \\left(\\mu_{i} \\mu_{j} - \\mathbb{1}_{i = j}\\, \\delta_{i}\\right) + b\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_delta_i])
  - ``\\mathbb{1}_{i = j}``: One when the pair names one asset twice, and zero otherwise. The residual block is diagonal.
  - $(math_dict[:T])
  - ``c``: Coefficient that the view gives the pair.
  - ``b``: Target of the view, with one value for each spanned asset pair.
  - ``d``: Sign that [`comparison_sign_ineq_flag`](@ref) reads from the operator, so every inequality row goes into `epc` in the sense of the `:ineq` key.

The identity holds only while the posterior means of assets ``i`` and ``j`` equal ``\\mu_{i}`` and ``\\mu_{j}``. So this method gives the assets that it names to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `cov_views.val` into one [`ParsingResult`](@ref) for each view.
 2. Replace every group name by the assets that it spans. A pair view stays a pair view.
 3. Replace every `prior(a, b)` reference by the prior covariance, through [`replace_coprior_views`](@ref). Each view is now a [`RhoParsingResult`](@ref) that holds its index pairs.
 4. For each view in turn, drop it when step 3 left it with no pair, and raise unless it names exactly one.
 5. Read the sign `d` and the inequality flag from the operator with [`comparison_sign_ineq_flag`](@ref).
 6. Read the index pair `(i, j)`, and build `Ai`, the product of the two return columns scaled by `d` and the coefficient of the view.
 7. Build `Bi`, the target moved by the product of the prior means less the entry of the residual block `E` at the pair, scaled the same way. `E` is the diagonal matrix of [`scenario_residual_variances`](@ref). A single asset pair gives a scalar, which the method puts in a vector of one element. A group pair gives one entry for each spanned pair.
 8. Add the row against `Bi` under `:ineq` or `:eq` with [`add_ep_constraint!`](@ref), and mark both assets of the pair in `to_fix`.

# Arguments

  - `cov_views`: Covariance view constraints.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `pr`: Prior result that holds the asset returns.
  - `sets`: Universe sets that map the asset names to indices.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Validation

  - Every view names exactly one asset pair. A view that mixes pairs raises an `ArgumentError`. If every pair of a view holds an asset that the universe does not hold, the view has no pair left. Under `strict = false`, [`replace_coprior_views`](@ref) already reported the unknown names, and the method drops such a view with no second report. Under `strict = true`, that function already raised an error.

# Returns

  - `to_fix::BitVector`: Boolean vector that marks the assets whose mean and variance must be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_coprior_views`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`get_pr_value`](@ref): reads the prior covariance that a `prior(a, b)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): read the `to_fix` that this method returns.
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
    # The residual block is diagonal, so it reaches a pair of one asset alone (#1649).
    E = LinearAlgebra.Diagonal(scenario_residual_variances(pr))
    for cov_view in cov_views
        #! `replace_coprior_views` drops a pair naming an asset the universe does not hold,
        #! which leaves the view with no pair at all. Under `strict = false` that must drop
        #! the row, as `get_linear_constraints` does for the linear families. Without this
        #! the guard below raised, called a view of no pairs one of several, and named an
        #! equation with no variable in it. The drop is silent: an unknown name was already
        #! named by `replace_coprior_views`, and a departed one is silent by ADR 0125 and
        #! is already recorded in the Departure Ledger.
        if isempty(cov_view.vars)
            continue
        end
        @argcheck(length(cov_view.vars) == 1,
                  "Cannot mix multiple covariance pairs in a single view `$(cov_view.eqn)`.")
        d, flag = comparison_sign_ineq_flag(cov_view.op)
        i, j = cov_view.ij[1]
        Ai = d * cov_view.coef[1] * view(X, :, i) .* view(X, :, j)
        Bi = d * (cov_view.coef[1] * (pr.mu[i] ⊙ pr.mu[j] ⊖ getindex.(Ref(E), i, j)) ⊕
                  cov_view.rhs)
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

Add the correlation views of a group to the entropy pooling constraint dictionary.

`ep_rho_views!` parses the correlation view equations of a [`LinearConstraintEstimator`](@ref), replaces each prior reference with its value, and makes the linear constraints of the views. It adds the constraints to the entropy pooling constraint dictionary `epc`. The statistic is the correlation of the posterior distribution, not the covariance, which the `cov_views` family holds.

The correlation is a covariance divided by two standard deviations, and it is linear in the posterior probabilities only when both means and both variances are constants. So this method returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior. A correlation view then keeps the mean and the variance that the prior states. The variance is the one that the covariance estimator of the prior gives, as [`fix_sigma!`](@ref) states.

# Mathematical definition

The row states the posterior cross moment about the prior means, with the target multiplied by the prior standard deviations. This row is linear in the posterior probabilities:

```math
\\begin{align}
\\rho_{\\boldsymbol{p}}[x_{i}, x_{j}] &= \\dfrac{\\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} - \\mu_{i} \\mu_{j} + \\mathbb{1}_{i = j}\\, \\delta_{i}}{\\sigma_{i} \\sigma_{j}}\\,, \\\\
c\\, \\rho_{\\boldsymbol{p}}[x_{i}, x_{j}] &\\lessgtr b\\,,
\\end{align}
```

which the method writes with the constants on the right:

```math
\\begin{align}
d\\, c \\sum_{t=1}^{T} p_{t} x_{t,\\,i} x_{t,\\,j} &\\lessgtr d \\left(c \\left(\\mu_{i} \\mu_{j} - \\mathbb{1}_{i = j}\\, \\delta_{i}\\right) + b\\, \\sigma_{i} \\sigma_{j}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:ep_delta_i])
  - ``\\mathbb{1}_{i = j}``: One when the pair names one asset twice, and zero otherwise. The residual block is diagonal.
  - $(math_dict[:T])
  - ``\\sigma_{i}``: Prior standard deviation of asset ``i``, the positive root of ``\\sigma_{i}^{2}``.
  - ``c``: Coefficient that the view gives the pair.
  - ``b``: Target of the view, with one value for each spanned asset pair, in ``[-1, 1]``.
  - ``d``: Sign that [`comparison_sign_ineq_flag`](@ref) reads from the operator, so every inequality row goes into `epc` in the sense of the `:ineq` key.

The identity holds only while the posterior means and variances of assets ``i`` and ``j`` equal the prior ones. So this method gives the assets that it names to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `rho_views.val` into one [`ParsingResult`](@ref) for each view.
 2. Replace every group name by the assets that it spans. A pair view stays a pair view.
 3. Replace every `prior(a, b)` reference by the prior correlation, through [`replace_coprior_views`](@ref). Each view is now a [`RhoParsingResult`](@ref) that holds its index pairs.
 4. Read the prior variances, the diagonal of `pr.sigma`, into `sigma`.
 5. For each view in turn, drop it when step 3 left it with no pair, raise unless it names exactly one, and raise unless every target lies in ``[-1, 1]``.
 6. Read the sign `d` and the inequality flag from the operator with [`comparison_sign_ineq_flag`](@ref).
 7. Read the index pair `(i, j)`, and build `sigma_ij`, the root of the product of the two prior variances.
 8. Build `Ai`, the product of the two return columns scaled by `d` and the coefficient of the view.
 9. Build `Bi`, the target multiplied by `sigma_ij` and moved by the product of the prior means less the entry of the residual block `E` at the pair, scaled the same way. `E` is the diagonal matrix of [`scenario_residual_variances`](@ref). A single asset pair gives a scalar, which the method puts in a vector of one element. A group pair gives one entry for each spanned pair.
10. Add the row against `Bi` under `:ineq` or `:eq` with [`add_ep_constraint!`](@ref), and mark both assets of the pair in `to_fix`.

# Arguments

  - `rho_views`: Correlation view constraints.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `pr`: Prior result that holds the asset returns.
  - `sets`: Universe sets that map the asset names to indices.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Validation

  - Every view names exactly one asset pair. A view that mixes pairs raises an `ArgumentError`. If every pair of a view holds an asset that the universe does not hold, the view has no pair left. Under `strict = false`, [`replace_coprior_views`](@ref) already reported the unknown names, and the method drops such a view with no second report. Under `strict = true`, that function already raised an error.
  - Every target lies in ``[-1, 1]``. A target outside that range raises an `ArgumentError`.

# Returns

  - `to_fix::BitVector`: Boolean vector that marks the assets whose mean and variance must be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_coprior_views`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`get_pr_value`](@ref): reads the prior correlation that a `prior(a, b)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): read the `to_fix` that this method returns.
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
    # See the note at the same step in `ep_cov_views!` (#1649).
    E = LinearAlgebra.Diagonal(scenario_residual_variances(pr))
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
        Bi = d * (rho_view.coef[1] * (pr.mu[i] ⊙ pr.mu[j] ⊖ getindex.(Ref(E), i, j)) ⊕
                  rho_view.rhs ⊙ sigma_ij)
        if !isa(i, AbstractVector)
            Bi = [Bi]
        end
        add_ep_constraint!(epc, transpose(Ai), Bi, ifelse(flag, :ineq, :eq))
        to_fix[union(i, j)] .= true
    end
    return to_fix
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:skew}, ::Nothing = nothing,
                 w::Option{<:ObsWeights} = nothing)

Read the prior skewness of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic that the tag `Val(:skew)` names. It applies [`Skewness`](@ref) to the `i`-th column of `pr.X`. That is the standardised third central moment of the sample under `w`, the weights of `pr`. [`Skewness`](@ref) also gives the weights to its mean and to its variance estimator.

A refit prior does not keep its weights in `pr.w`, so the caller gives them, as for the value at risk method of [`get_pr_value`](@ref). Under a staged algorithm, they are the weights of the prior of the stage.

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Index of the asset.
  - `::Val{:skew}`: Dispatch tag of the skewness.
  - `::Nothing`: The level slot of [`replace_prior_views`](@ref), which a skewness does not read.
  - $(arg_dict[:oow])

# Returns

  - `skew::Number`: Prior skewness of asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
  - [`ep_sk_views!`](@ref): gives `w` to this method.
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:skew},
                      ::Nothing = nothing, w::Option{<:ObsWeights} = nothing)
    return Skewness(; w = w)(view(pr.X, :, i))
end
"""
    ep_sk_views!(skew_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::UniverseSets,
                 w::Option{<:ObsWeights} = nothing; strict::Bool = false)

Add the skewness views of a group to the entropy pooling constraint dictionary.

`ep_sk_views!` parses the skewness view equations of a [`LinearConstraintEstimator`](@ref), replaces each prior reference with its value, and makes the linear constraints of the views. It adds the constraints to the entropy pooling constraint dictionary `epc`. The statistic is the standardised third central moment of the posterior distribution.

The skewness is a third central moment divided by a cube of the standard deviation, and it is linear in the posterior probabilities only when the mean and the variance are constants. So this method returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior. A skewness view then keeps the mean and the variance that the prior states. The variance is the one that the covariance estimator of the prior gives, as [`fix_sigma!`](@ref) states.

# Mathematical definition

The third central moment expands into raw moments, and the two lowest raw moments are prior constants. So the rest is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{3}\\right] &= \\sum_{t=1}^{T} p_{t} x_{t,\\,i}^{3} - 3 \\mu_{i} \\sigma_{i}^{2} - \\mu_{i}^{3}\\,, \\\\
\\mathrm{Skew}_{\\boldsymbol{p}}[x_{i}] &= \\dfrac{\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{3}\\right]}{\\left(\\sigma_{i}^{2}\\right)^{3/2}}
 = \\sum_{t=1}^{T} p_{t} \\dfrac{x_{t,\\,i}^{3} - \\mu_{i}^{3} - 3 \\mu_{i} \\sigma_{i}^{2}}{\\left(\\sigma_{i}^{2}\\right)^{3/2}}\\,, \\\\
\\sum_{i=1}^{N} a_{k,\\,i} \\mathrm{Skew}_{\\boldsymbol{p}}[x_{i}] &\\lessgtr B_{k}\\,, \\quad \\forall\\, k = 1,\\ldots,K\\,.
\\end{align}
```

The second line uses ``\\sum_{t} p_{t} = 1`` to put the two constants inside the sum, which is the form that the method builds.

The expansion needs the variance of the scenarios that the sum reads, so ``\\sigma_{i}^{2}`` here is ``\\sigma_{i}^{2} - \\delta_{i}``, the variance that [`fix_sigma!`](@ref) holds. The view states the skewness of the scenarios. A `prior(...)` reference reads the sample skewness under `w` with [`Skewness`](@ref), which standardises with its own variance estimator.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:ep_delta_i])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:B])
  - ``a_{k,\\,i}``: Coefficient that asset ``i`` takes in view ``k``.
  - ``K``: Number of views that the group states.

The identity holds only while the posterior mean and variance of asset ``i`` equal ``\\mu_{i}`` and ``\\sigma_{i}^{2}``. So this method gives the assets that it names to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `skew_views.val` into one [`ParsingResult`](@ref) for each view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior skewness under `w`, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false`, every row of the group can drop, and `lcs` is then `nothing`. The group then states no view, and the call returns a `to_fix` that names no asset.
 5. Read the prior variances of the scenarios into `sigma`. They are the diagonal of `pr.sigma`, less [`scenario_residual_variances`](@ref).
 6. Build `tmp`, the contribution of every observation to the standardised third moment, transposed so that a row of `lcs` multiplies it from the left.
 7. For each block that `lcs` holds, add `A * tmp` against `B` under its key with [`add_ep_constraint!`](@ref), and mark in `to_fix` every asset that the block names.

# Arguments

  - `skew_views`: Skewness view constraints.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `pr`: Prior result that holds the asset returns.
  - `sets`: Universe sets that map the asset names to indices.
  - `w`: The weights of `pr`, or `nothing` for uniform weights. A `prior(...)` reference reads the skewness under them.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Returns

  - `to_fix::BitVector`: Boolean vector that marks the assets whose mean and variance must be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior skewness that a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): read the `to_fix` that this method returns.
  - [`ep_kt_views!`](@ref): the same rule, one moment higher.
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_sk_views!(skew_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::UniverseSets,
                      w::Option{<:ObsWeights} = nothing; strict::Bool = false,
                      ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    skew_views = parse_equation(skew_views.val; datatype = eltype(X))
    skew_views = replace_group_by_assets(skew_views, sets, false, true, false;
                                         ledger = ledger)
    skew_views = replace_prior_views(skew_views, pr, sets, :skew, nothing, w;
                                     strict = strict)
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
    sigma = LinearAlgebra.diag(pr.sigma) - scenario_residual_variances(pr)
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
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:kurtosis},
                 ::Nothing = nothing, w::Option{<:ObsWeights} = nothing)

Read the prior kurtosis of asset `i`.

`get_pr_value` is the dispatch table that resolves a `prior(...)` reference inside a view. This method reads the statistic that the tag `Val(:kurtosis)` names. It applies [`HighOrderMoment`](@ref) with a [`StandardisedHighOrderMoment`](@ref) of [`FourthMoment`](@ref) to the `i`-th column of `pr.X`. That is the standardised fourth central moment of the sample under `w`. It reads `w` for the same reason as the skewness method of [`get_pr_value`](@ref).

# Arguments

  - `pr`: Prior result that holds the asset returns.
  - `i`: Index of the asset.
  - `::Val{:kurtosis}`: Dispatch tag of the kurtosis.
  - `::Nothing`: The level slot of [`replace_prior_views`](@ref), which a kurtosis does not read.
  - $(arg_dict[:oow])

# Returns

  - `kurtosis::Number`: Prior kurtosis of asset `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`get_pr_value`](@ref)
  - [`ep_kt_views!`](@ref): gives `w` to this method.
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:kurtosis},
                      ::Nothing = nothing, w::Option{<:ObsWeights} = nothing)
    return HighOrderMoment(; w = w,
                           alg = StandardisedHighOrderMoment(; alg = FourthMoment()))(view(pr.X,
                                                                                           :,
                                                                                           i))
end
"""
    ep_kt_views!(kurtosis_views::LinearConstraintEstimator, epc::AbstractDict,
                 pr::AbstractPriorResult, sets::UniverseSets,
                 w::Option{<:ObsWeights} = nothing; strict::Bool = false)

Add the kurtosis views of a group to the entropy pooling constraint dictionary.

`ep_kt_views!` parses the kurtosis view equations of a [`LinearConstraintEstimator`](@ref), replaces each prior reference with its value, and makes the linear constraints of the views. It adds the constraints to the entropy pooling constraint dictionary `epc`. The statistic is the standardised fourth central moment of the posterior distribution.

The kurtosis is a fourth central moment divided by a square of the variance, and it is linear in the posterior probabilities only when the mean and the variance are constants. So this method returns the assets whose mean and variance [`fix_mu!`](@ref) and [`fix_sigma!`](@ref) must hold at the prior. A kurtosis view then keeps the mean and the variance that the prior states. The variance is the one that the covariance estimator of the prior gives, as [`fix_sigma!`](@ref) states.

# Mathematical definition

The fourth central moment expands into raw moments, and every constant of the expansion is a prior constant. So the rest is linear in the posterior probabilities:

```math
\\begin{align}
\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{4}\\right] &= \\sum_{t=1}^{T} p_{t} \\left(x_{t,\\,i}^{4} - 4 \\mu_{i} x_{t,\\,i}^{3} + 6 \\mu_{i}^{2} x_{t,\\,i}^{2} - 3 \\mu_{i}^{4}\\right)\\,, \\\\
\\mathrm{Kurt}_{\\boldsymbol{p}}[x_{i}] &= \\dfrac{\\mathrm{E}_{\\boldsymbol{p}}\\!\\left[(x_{i} - \\mu_{i})^{4}\\right]}{\\left(\\sigma_{i}^{2}\\right)^{2}}\\,, \\\\
\\sum_{i=1}^{N} a_{k,\\,i} \\mathrm{Kurt}_{\\boldsymbol{p}}[x_{i}] &\\lessgtr B_{k}\\,, \\quad \\forall\\, k = 1,\\ldots,K\\,.
\\end{align}
```

The first line uses ``\\sum_{t} p_{t} = 1`` to put the constant ``3 \\mu_{i}^{4}`` inside the sum, which is the form that the method builds.

The standardisation reads the variance of the scenarios, so ``\\sigma_{i}^{2}`` here is ``\\sigma_{i}^{2} - \\delta_{i}``, the variance that [`fix_sigma!`](@ref) holds. The view states the kurtosis of the scenarios. A `prior(...)` reference reads the sample kurtosis under `w` with [`HighOrderMoment`](@ref), which standardises with its own variance estimator.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:ep_mu_prior_i])
  - $(math_dict[:ep_sigma2_prior_i])
  - $(math_dict[:ep_delta_i])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:B])
  - ``a_{k,\\,i}``: Coefficient that asset ``i`` takes in view ``k``.
  - ``K``: Number of views that the group states.

The identity holds only while the posterior mean and variance of asset ``i`` equal ``\\mu_{i}`` and ``\\sigma_{i}^{2}``. So this method gives the assets that it names to [`fix_mu!`](@ref) and [`fix_sigma!`](@ref).

# Algorithm

 1. Parse the view equations of `kurtosis_views.val` into one [`ParsingResult`](@ref) for each view.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior kurtosis under `w`, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false`, every row of the group can drop, and `lcs` is then `nothing`. The group then states no view, and the call returns a `to_fix` that names no asset.
 5. Build `X_sq` and `mu_sq`, the squares of the returns and of the prior means.
 6. Build `tmp`, the contribution of every observation to the fourth moment, standardised by the prior variance of the scenarios. That variance is the diagonal of `pr.sigma` less [`scenario_residual_variances`](@ref). Transpose `tmp` so that a row of `lcs` multiplies it from the left.
 7. For each block that `lcs` holds, add `A * tmp` against `B` under its key with [`add_ep_constraint!`](@ref), and mark in `to_fix` every asset that the block names.

# Arguments

  - `kurtosis_views`: Kurtosis view constraints.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `pr`: Prior result that holds the asset returns.
  - `sets`: Universe sets that map the asset names to indices.
  - `w`: The weights of `pr`, or `nothing` for uniform weights. A `prior(...)` reference reads the kurtosis under them.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Returns

  - `to_fix::BitVector`: Boolean vector that marks the assets whose mean and variance must be fixed.

# Related

  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior kurtosis that a `prior(...)` reference resolves to.
  - [`fix_mu!`](@ref), [`fix_sigma!`](@ref): read the `to_fix` that this method returns.
  - [`ep_sk_views!`](@ref): the same rule, one moment lower.
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_kt_views!(kurtosis_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::UniverseSets,
                      w::Option{<:ObsWeights} = nothing; strict::Bool = false,
                      ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    kurtosis_views = parse_equation(kurtosis_views.val; datatype = eltype(X))
    kurtosis_views = replace_group_by_assets(kurtosis_views, sets, false, true, false;
                                             ledger = ledger)
    kurtosis_views = replace_prior_views(kurtosis_views, pr, sets, :kurtosis, nothing, w;
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
                    transpose(LinearAlgebra.diag(pr.sigma) -
                              scenario_residual_variances(pr)) .^ 2)
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
