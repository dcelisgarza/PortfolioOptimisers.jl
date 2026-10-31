# The keyword constructors of the sequential tail view constraints, which `ep_refine_tail_view`
# calls through `rebuild_estimator`. They are here and not beside their types, because the file
# of the types is at its limit of code lines.
function SequentialConditionalValueatRiskViewConstraint(; xd, cd, xp, cp, c, b, alpha, rhs,
                                                        iters, tol)
    return SequentialConditionalValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, rhs,
                                                          iters, tol)
end
function SequentialEntropicValueatRiskViewConstraint(; xd, cd, xp, cp, c, b, alpha, rhs,
                                                     iters, tol, args, kwargs, zlo_frac)
    return SequentialEntropicValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, rhs,
                                                       iters, tol, args, kwargs, zlo_frac)
end
function SequentialRelativisticValueatRiskViewConstraint(; xd, cd, xp, cp, c, b, alpha,
                                                         kappa, rhs, iters, tol, args,
                                                         kwargs, bracket)
    return SequentialRelativisticValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha,
                                                           kappa, rhs, iters, tol, args,
                                                           kwargs, bracket)
end
"""
    ep_check_tail_window(tv::IntegerConditionalValueatRiskViewConstraint, w::VecNum)

Warn when the window of an integer conditional value-at-risk view can restrict the posterior.

The model of [`IntegerConditionalValueatRiskView`](@ref) accepts only the posteriors that put at least `alpha` of their mass on the `sbar` largest losses of each asset. When this restriction is active, the window holds exactly `alpha`, and the function warns. A window that holds more is not active at the posterior, so a small change of the posterior does not reach the restriction.

# Algorithm

 1. For each asset of the view, skip a window that holds every observation, because it restricts nothing.
 2. Sum `w` over the window. Warn when the sum minus `alpha` is at most `alpha` times the cube root of the machine epsilon of `w`. This margin is far above the feasibility tolerance of a solver and far below the mass of one observation.

# Arguments

  - `tv`: The integer conditional value-at-risk view constraint.
  - `w`: The posterior probabilities of the last solve.

# Returns

  - `nothing`.

# Related

  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskViewConstraint`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_check_tail_window(tv::IntegerConditionalValueatRiskViewConstraint, w::VecNum)
    (; ord, alpha) = tv
    for o in ord
        # The window binds when it holds no more than `alpha`, since the model asks it to hold
        # at least that. A window of the whole sample restricts nothing.
        if length(o) < length(w) && sum(view(w, o)) - alpha <= alpha * cbrt(eps(eltype(w)))
            @warn("The integer CVaR view reads the $(length(o)) largest losses, and the posterior puts only the tail mass $(alpha) on them, so the window binds and a posterior of smaller divergence can lie outside it. Raise `sbar` in `IntegerConditionalValueatRiskView`.",
                  maxlog = log_limit(:warn), _id = @log_id())
        end
    end
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw, tv::GridEntropicValueatRiskViewConstraint,
                           sc1::Number)
    (; x, z, alpha, rhs, M) = tv
    K = length(z)
    y = JuMP.@variable(model, [1:K], binary = true)
    JuMP.@constraint(model, sc1 * (sum(y) - one(alpha)) == 0)
    for (k, zk) in pairs(z)
        c, isc = ep_evar_grid_row(x, rhs, zk)
        # The bound `alpha * isc` falls below a solver's feasibility tolerance at a small `z`,
        # and a row read at that scale is met by any posterior. Divided by its bound, every
        # row reads against one. `c` peaks at one, so `M * (ib - 1)` releases the row.
        # Issue #1264.
        ib = inv(alpha * isc)
        JuMP.@constraint(model,
                         sc1 * (ib * LinearAlgebra.dot(c, pw) - one(alpha) -
                                M * (ib - one(alpha)) * (one(alpha) - y[k])) <= 0)
    end
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::GridRelativisticValueatRiskViewConstraint, sc1::Number)
    (; x, t, z, alpha, kappa, rhs, M) = tv
    K = length(z)
    y = JuMP.@variable(model, [1:K], binary = true)
    JuMP.@constraint(model, sc1 * (sum(y) - one(alpha)) == 0)
    for k in 1:K
        c, b = ep_rlvar_grid_row(x, rhs, t[k], z[k], alpha, kappa)
        # As for the entropic value at risk: divided by its bound, every row reads against one.
        ib = inv(b)
        JuMP.@constraint(model,
                         sc1 * (ib * LinearAlgebra.dot(c, pw) - one(b) -
                                M * (ib - one(b)) * (one(b) - y[k])) <= 0)
    end
    return nothing
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:evar}, alpha::Number,
                 w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                 kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)

Compute the entropic value at risk (EVaR) of asset `i` from a prior result.

`get_pr_value` computes the EVaR of asset `i` at the significance level `alpha`. It minimises the scalar objective of the sample EVaR of `pr` with [`ep_evar`](@ref). The observations take the weights `w`, which are the prior probabilities from which the entropy pooling starts. So a `prior(...)` reference reads the same statistic as the prior value of the view. A `w` of `nothing` gives uniform weights.

# Arguments

  - `pr`: The prior result. The function reads only its returns matrix, under the weights `w`.
  - `i`: The index of the asset.
  - `::Val{:evar}`: The dispatch tag of the EVaR.
  - `alpha`: The significance level, for example `0.05` for the EVaR at 5 %.
  - $(arg_dict[:oow])
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: The lower end of the bracket of the dual variable, as a fraction of the upper end, which goes to [`ep_evar`](@ref).

# Returns

  - `evar::Number`: The entropic value at risk of asset `i` at the level `alpha`.

# Related

  - [`ep_evar`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:evar}, alpha::Number,
                      w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                      kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)
    T = size(pr.X, 1)
    iT = inv(T)
    w = isnothing(w) ? range(iT, iT; length = T) : w
    return ep_evar(-view(pr.X, :, i), w, alpha; args = args, kwargs = kwargs,
                   zlo_frac = zlo_frac).evar
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:rlvar}, alpha::Number,
                 kappa::Number, w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                 kwargs::NamedTuple = (;),
                 bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Compute the relativistic value at risk (RLVaR) of asset `i` from a prior result.

`get_pr_value` computes the RLVaR of asset `i` at the significance level `alpha` and the deformation parameter `kappa`. It minimises the primal objective of the sample RLVaR of `pr` with [`ep_rlvar`](@ref). The observations take the weights `w`, which are the prior probabilities from which the entropy pooling starts, for the reason that the method of the entropic value at risk gives.

# Arguments

  - `pr`: The prior result. The function reads only its returns matrix, under the weights `w`.
  - `i`: The index of the asset.
  - `::Val{:rlvar}`: The dispatch tag of the RLVaR.
  - `alpha`: The significance level, for example `0.05` for the RLVaR at 5 %.
  - `kappa`: The deformation parameter, in `(0, 1)`.
  - $(arg_dict[:oow])
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: The spans of the searches, which go to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Returns

  - `rlvar::Number`: The relativistic value at risk of asset `i` at the level `alpha` and the deformation `kappa`.

# Related

  - [`ep_rlvar`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:rlvar}, alpha::Number,
                      kappa::Number, w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                      kwargs::NamedTuple = (;),
                      bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    T = size(pr.X, 1)
    iT = inv(T)
    w = isnothing(w) ? range(iT, iT; length = T) : w
    return ep_rlvar(-view(pr.X, :, i), w, alpha, kappa; args = args, kwargs = kwargs,
                    bracket = bracket).rlvar
end
"""
    ep_view_terms(res::ParsingResult, sets::UniverseSets, X::MatNum; strict::Bool = false,
                  ledger::Option{<:AbstractVector} = nothing)

Resolve one parsed tail view into the assets that it names, their coefficients, its operator and its target.

`ep_view_terms` gives a [`ParsingResult`](@ref) to [`get_linear_constraints`](@ref). That function resolves the variable names against the universe, and reports the names that it cannot place. It also negates a `>=` equation, so `ep_view_terms` undoes the negation to keep the operator. A linear view does not need the operator, because a row of `A x <= b` holds it. A tail view needs it, because each operator selects a different formulation.

# Algorithm

 1. Resolve the view against the universe with [`get_linear_constraints`](@ref). The result is `lc`. Return `nothing` when it places no name of the view.
 2. Read the sign `sgn` and the inequality flag of the operator of the view with [`comparison_sign_ineq_flag`](@ref). From them, select the operator `op` of the view and the block `blk` of `lc` that holds the row.
 3. Multiply the row `A` and the target `rhs` by `sgn`. This undoes the negation that [`get_linear_constraints`](@ref) applies to a `>=` equation.
 4. Return the indices of the non-zero entries of `A`, the coefficients at those indices, `op` and `rhs`.

# Arguments

  - `res`: The parsed view constraint.
  - `sets`: The universe sets that map the asset names to indices.
  - `X`: The asset returns matrix. The function reads only its element type.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.
  - $(arg_dict[:ledger]) [`get_linear_constraints`](@ref) fills it.

# Returns

  - `nothing`: When the universe holds no name of the view.
  - `terms::@NamedTuple{idx::VecInt, coef::VecNum, op::Symbol, rhs::Number}`: The assets that the view names, their coefficients, the operator of the view (`:eq`, `:geq` or `:leq`) and its target.

# Related

  - [`ep_tail_views!`](@ref)
  - [`get_linear_constraints`](@ref)
  - [`comparison_sign_ineq_flag`](@ref)
"""
function ep_view_terms(res::ParsingResult, sets::UniverseSets, X::MatNum;
                       strict::Bool = false, ledger::Option{<:AbstractVector} = nothing)
    lc = get_linear_constraints([res], sets; datatype = eltype(X), strict = strict,
                                ledger = ledger)
    if isnothing(lc)
        return nothing
    end
    sgn, flag = comparison_sign_ineq_flag(res.op)
    op, blk = if !flag
        :eq, lc.eq
    elseif sgn == -1
        :geq, lc.ineq
    else
        :leq, lc.ineq
    end
    # `get_linear_constraints` negates a `>=` equation to file it as a `<=` row, scaling by
    # the same sign, so scaling by it again undoes the flip.
    d = eltype(X)(sgn)
    A = vec(blk.A) * d
    rhs = blk.B[1] * d
    idx = findall(!iszero, A)
    return (; idx = idx, coef = A[idx], op = op, rhs = rhs)
end
"""
    ep_normalise_view_term(coef::Number, op::Symbol, rhs::Number)

Divide a tail view of one asset by its coefficient.

The function returns the operator and the target of the same view with a coefficient of one. When the coefficient is negative, it reverses the operator.

# Algorithm

 1. Divide the target by the coefficient. The result is `rhs`.
 2. When the coefficient is negative, exchange `:geq` and `:leq`. The result is `op`. A division by a negative number keeps an equality, so `:eq` does not change.

# Arguments

  - `coef`: The coefficient that the view gives the risk measure of the asset.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.

# Returns

  - `op::Symbol`: The operator of the normalised view.
  - `rhs::Number`: The target of the normalised view.

# Related

  - [`ep_view_terms`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_normalise_view_term(coef::Number, op::Symbol, rhs::Number)
    rhs /= coef
    if coef < zero(coef)
        op = if op == :geq
            :leq
        elseif op == :leq
            :geq
        else
            op
        end
    end
    return op, rhs
end
"""
    ep_view_formulations(alg, N::Integer, key::Symbol)

Expand the formulation setting of a tail view group to one formulation for each view.

One formulation applies to every view. A vector gives one formulation for each view. `nothing` lets each view select its formulation.

# Arguments

  - `alg`: The formulation setting.
  - `N`: The number of views.
  - `key`: The name of the field, for the error message.

# Validation

  - If `alg` is a vector, `length(alg) == N`.

# Returns

  - `algs::AbstractVector`: One entry for each view.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_view_formulations(alg::Option{<:AbstractEntropyPoolingViewFormulation},
                              N::Integer, ::Symbol)
    return fill(alg, N)
end
function ep_view_formulations(alg::AbstractVector, N::Integer, key::Symbol)
    @argcheck(length(alg) == N,
              DimensionMismatch("length($key) ($(length(alg))) must match the number of views ($N)"))
    return alg
end
"""
    ep_sbar(sbar, T::Integer, alpha::Number, w::VecNum, ord::VecInt)

Find the number of largest losses that the integer conditional value-at-risk formulation reads.

# Algorithm

 1. An `Integer` `sbar` is a count. Return it, at most `T`.
 2. Any other number is a fraction of `T`. Return `ceil(Int, sbar * T)`, at least one and at most `T`.
 3. `nothing` uses the rule of thumb of [EPTail](@cite). Go through the losses from the largest down, and add their prior probabilities. Stop at the position `s` at which the sum first reaches `alpha`.
 4. Return two times `s`, at least `ceil(Int, 2 * alpha * T)`, at least one and at most `T`.

[EPTail](@citet) uses a count of at most `s` for a view at or above the prior CVaR. For a view below the prior CVaR, it uses a count above `s` and far below `T`. Its examples use `int(alpha * T)` for the first case and `int(2 * alpha * T)` for the second case. The library uses the larger count for every view, and the factor of two is its own.

# Arguments

  - `sbar`: The setting that [`IntegerConditionalValueatRiskView`](@ref) holds. An `Integer` is a count, a fraction in `(0, 1)` is a fraction of `T`, and `nothing` uses the rule of thumb of [EPTail](@cite).
  - `T`: The number of observations.
  - `alpha`: The significance level of the view.
  - `w`: The prior probability weights.
  - `ord`: The indices of the losses in ascending order, so the largest loss is last.

# Returns

  - `sbar::Int`: The number of largest losses, in `1:T`.

# Related

  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equation 4, and Section 4.
"""
function ep_sbar(sbar::Nothing, T::Integer, alpha::Number, w::VecNum, ord::VecInt)
    cw = zero(eltype(w))
    s = T
    for (j, o) in enumerate(Iterators.reverse(ord))
        cw += w[o]
        if cw >= alpha
            s = j
            break
        end
    end
    return min(T, max(2 * s, ceil(Int, 2 * alpha * T), 1))
end
function ep_sbar(sbar::Integer, T::Integer, args...)
    return min(T, sbar)
end
function ep_sbar(sbar::Number, T::Integer, args...)
    return min(T, max(1, ceil(Int, sbar * T)))
end
"""
    ep_assert_reachable_view(op::Symbol, rhs::Number, x::AbstractVector{<:VecNum},
                             coef::VecNum, w::VecNum, eqn::AbstractString,
                             name::AbstractString)

Refuse a tail view that no reweighting of the sample can reach.

A tail risk measure of a reweighted sample is between the smallest and the largest loss of the sample at a positive prior probability. So a coefficient-weighted sum of measures is between the sums of those bounds, and no change of the probabilities meets a view outside that band. For one asset the band is exact. For several assets it is an outer bound, because a reweighting that puts every asset at its worst loss together does not always exist. The posterior of entropy pooling puts no mass where the prior puts none, so an observation of zero prior probability does not widen the band.

# Algorithm

 1. Calculate `hi` over the observations with `w > 0`. It is the sum over the assets of the coefficient times the largest loss when the coefficient is positive, and times the smallest loss when the coefficient is negative. Calculate `lo` with the opposite choice.
 2. When `op` asks the statistic to be at least `rhs`, raise an error unless `rhs` is below `hi`.
 3. When `op` asks the statistic to be at most `rhs`, raise an error unless `rhs` is above `lo`.

# Arguments

  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `x`: The loss series of each asset that the view names.
  - `coef`: The coefficient that the view gives the risk measure of each asset.
  - `w`: The prior probability weights.
  - `eqn`: The equation of the view, for the error message.
  - `name`: The name of the view family, for the error message.

# Validation

  - If `op` is `:geq` or `:eq`, `rhs < hi`.
  - If `op` is `:leq` or `:eq`, `rhs > lo`.

# Returns

  - `nothing`.

# Related

  - [`ep_normalise_tail_view`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_assert_reachable_view(op::Symbol, rhs::Number, x::AbstractVector{<:VecNum},
                                  coef::VecNum, w::VecNum, eqn::AbstractString,
                                  name::AbstractString)
    # Only the support of `w` bounds the band: a posterior puts no mass where the prior puts
    # none. Issue #1260.
    sup = w .> zero(eltype(w))
    hi = sum(ci * ifelse(ci > zero(ci), maximum(view(xi, sup)), minimum(view(xi, sup)))
             for (xi, ci) in zip(x, coef))
    lo = sum(ci * ifelse(ci > zero(ci), minimum(view(xi, sup)), maximum(view(xi, sup)))
             for (xi, ci) in zip(x, coef))
    if op == :geq || op == :eq
        @argcheck(rhs < hi,
                  DomainError(rhs,
                              "View `$(eqn)` is too extreme: the largest $(name) any reweighting of this sample reaches is the worst realisation of its left hand side, $(hi). Lower the view, raise alpha, or use a prior with fatter tails."))
    end
    if op == :leq || op == :eq
        @argcheck(rhs > lo,
                  DomainError(rhs,
                              "View `$(eqn)` is too extreme: the smallest $(name) any reweighting of this sample reaches is the best realisation of its left hand side, $(lo). Raise the view, lower alpha, or use a prior with a thinner tail."))
    end
    return nothing
end
"""
    ep_cvar_formulation(alg::Option{<:CVaRVF}, mixed::Bool, op::Symbol, rhs::Number,
                        pv::Number)

Select the formulation of one conditional value-at-risk view.

The function returns a stated formulation with no change. For `nothing`, it selects [`LinearConditionalValueatRiskView`](@ref) when that formulation writes the view exactly, which it does for every view whose upper level set is convex. For the other views it selects [`IntegerConditionalValueatRiskView`](@ref). These are a view whose coefficients have both signs, an upper bound, and an equality below the prior value of the left hand side.

The table gives the branch for each input:

| `alg`     | `mixed` | `op`   | `rhs` against `pv` | Branch                                      |
|:--------- |:------- |:------ |:------------------ |:------------------------------------------- |
| stated    | any     | any    | any                | `alg`, unchanged                            |
| `nothing` | `false` | `:geq` | any                | [`LinearConditionalValueatRiskView`](@ref)  |
| `nothing` | `false` | `:eq`  | `rhs >= pv`        | [`LinearConditionalValueatRiskView`](@ref)  |
| `nothing` | `false` | `:eq`  | `rhs < pv`         | [`IntegerConditionalValueatRiskView`](@ref) |
| `nothing` | `false` | `:leq` | any                | [`IntegerConditionalValueatRiskView`](@ref) |
| `nothing` | `true`  | any    | any                | [`IntegerConditionalValueatRiskView`](@ref) |

[`SequentialTailView`](@ref) is never the default. It writes every view that the integer formulation writes, with no integer variable. But its posterior is a local minimiser of the divergence, and `nothing` selects an exact formulation.

# Arguments

  - `alg`: The stated formulation, or `nothing`.
  - `mixed`: Whether the coefficients of the view have both signs.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `pv`: The prior value of the left hand side of the view.

# Returns

  - `alg::CVaRVF`: The formulation to use.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`SequentialTailView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_cvar_formulation(alg::CVaRVF, args...)
    return alg
end
function ep_cvar_formulation(::Nothing, mixed::Bool, op::Symbol, rhs::Number, pv::Number)
    return if !mixed && (op == :geq || op == :eq && rhs >= pv)
        LinearConditionalValueatRiskView()
    else
        IntegerConditionalValueatRiskView()
    end
end
"""
    ep_evar_formulation(alg::Option{<:EVaRVF}, mixed::Bool, single::Bool, op::Symbol,
                        rhs::Number, pv::Number)

Select the formulation of one entropic value-at-risk view.

The function returns a stated formulation with no change. For `nothing`, it selects [`ConicEntropicValueatRiskView`](@ref) when that formulation writes the view exactly, which it does for every view whose upper level set is convex. An upper bound and an equality below the prior value use [`GridTailView`](@ref) when the view names one asset. The grid holds one asset only, so the same views over several assets use [`SequentialTailView`](@ref), the one formulation that can write them. A view whose coefficients have both signs also uses [`SequentialTailView`](@ref).

The table gives the branch for each input:

| `alg`     | `mixed` | `single` | `op`   | `rhs` against `pv` | Branch                                 |
|:--------- |:------- |:-------- |:------ |:------------------ |:-------------------------------------- |
| stated    | any     | any      | any    | any                | `alg`, unchanged                       |
| `nothing` | `false` | any      | `:geq` | any                | [`ConicEntropicValueatRiskView`](@ref) |
| `nothing` | `false` | any      | `:eq`  | `rhs >= pv`        | [`ConicEntropicValueatRiskView`](@ref) |
| `nothing` | `false` | `true`   | `:eq`  | `rhs < pv`         | [`GridTailView`](@ref)                 |
| `nothing` | `false` | `true`   | `:leq` | any                | [`GridTailView`](@ref)                 |
| `nothing` | `false` | `false`  | `:eq`  | `rhs < pv`         | [`SequentialTailView`](@ref)           |
| `nothing` | `false` | `false`  | `:leq` | any                | [`SequentialTailView`](@ref)           |
| `nothing` | `true`  | `false`  | any    | any                | [`SequentialTailView`](@ref)           |

# Arguments

  - `alg`: The stated formulation, or `nothing`.
  - `mixed`: Whether the coefficients of the view have both signs.
  - `single`: Whether the view names one asset.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `pv`: The prior value of the left hand side of the view.

# Returns

  - `alg::EVaRVF`: The formulation to use.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridTailView`](@ref)
  - [`SequentialTailView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_evar_formulation(alg::EVaRVF, args...)
    return alg
end
function ep_evar_formulation(::Nothing, mixed::Bool, single::Bool, op::Symbol, rhs::Number,
                             pv::Number)
    return if mixed
        SequentialTailView()
    elseif op == :geq || op == :eq && rhs >= pv
        ConicEntropicValueatRiskView()
    elseif single
        GridTailView()
    else
        # The grid is one asset's. Issue #1287.
        SequentialTailView()
    end
end
"""
    ep_rlvar_formulation(alg::Option{<:RLVaRVF}, mixed::Bool, single::Bool, op::Symbol,
                         rhs::Number, pv::Number)

Select the formulation of one relativistic value-at-risk view.

The function returns a stated formulation with no change. For `nothing`, it selects [`ConicRelativisticValueatRiskView`](@ref) when that formulation writes the view exactly, which it does for every view whose upper level set is convex. An upper bound and an equality below the prior value use [`GridTailView`](@ref) when the view names one asset. The grid holds one asset only, so the same views over several assets use [`SequentialTailView`](@ref), the one formulation that can write them. A view whose coefficients have both signs also uses [`SequentialTailView`](@ref).

The table gives the branch for each input:

| `alg`     | `mixed` | `single` | `op`   | `rhs` against `pv` | Branch                                     |
|:--------- |:------- |:-------- |:------ |:------------------ |:------------------------------------------ |
| stated    | any     | any      | any    | any                | `alg`, unchanged                           |
| `nothing` | `false` | any      | `:geq` | any                | [`ConicRelativisticValueatRiskView`](@ref) |
| `nothing` | `false` | any      | `:eq`  | `rhs >= pv`        | [`ConicRelativisticValueatRiskView`](@ref) |
| `nothing` | `false` | `true`   | `:eq`  | `rhs < pv`         | [`GridTailView`](@ref)                     |
| `nothing` | `false` | `true`   | `:leq` | any                | [`GridTailView`](@ref)                     |
| `nothing` | `false` | `false`  | `:eq`  | `rhs < pv`         | [`SequentialTailView`](@ref)               |
| `nothing` | `false` | `false`  | `:leq` | any                | [`SequentialTailView`](@ref)               |
| `nothing` | `true`  | `false`  | any    | any                | [`SequentialTailView`](@ref)               |

# Arguments

  - `alg`: The stated formulation, or `nothing`.
  - `mixed`: Whether the coefficients of the view have both signs.
  - `single`: Whether the view names one asset.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `pv`: The prior value of the left hand side of the view.

# Returns

  - `alg::RLVaRVF`: The formulation to use.

# Related

  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridTailView`](@ref)
  - [`SequentialTailView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_rlvar_formulation(alg::RLVaRVF, args...)
    return alg
end
function ep_rlvar_formulation(::Nothing, mixed::Bool, single::Bool, op::Symbol, rhs::Number,
                              pv::Number)
    return if mixed
        SequentialTailView()
    elseif op == :geq || op == :eq && rhs >= pv
        ConicRelativisticValueatRiskView()
    elseif single
        GridTailView()
    else
        # The grid is one asset's. Issue #1287.
        SequentialTailView()
    end
end
"""
    ep_sequential_sides(x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol, rhs::Number,
                        pv::Number)

Orient a tail view as a lower bound, and split its assets into the dual side and the primal side of a sequential formulation.

# Algorithm

 1. Orient the view. For an upper bound, and for an equality whose target is at or below the prior value of the left hand side, multiply both sides by minus one, so that the view reads `>=`. Keep a lower bound, and an equality whose target is above the prior value. So the function writes an equality as the bound that the prior does not meet, and the minimiser of the entropy makes that bound tight.
 2. Put every asset with a positive oriented coefficient on the dual side. Its measure is concave in the probabilities and takes its exact dual block. Put every asset with a negative oriented coefficient on the primal side. Its measure takes a linear upper bound.

# Arguments

  - `x`: The loss series of each asset that the view names.
  - `coef`: The coefficient that the view gives the risk measure of each asset.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `pv`: The prior value of the left hand side of the view.

# Returns

  - `xd::AbstractVector{<:VecNum}`: The loss series of the assets on the dual side.
  - `cd::VecNum`: Their oriented coefficients, which are positive.
  - `xp::AbstractVector{<:VecNum}`: The loss series of the assets on the primal side.
  - `cp::VecNum`: Their oriented coefficients, which are negative.
  - `rhs::Number`: The target of the oriented view.

# Related

  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`ep_add_cvar_view!`](@ref)
  - [`ep_add_evar_view!`](@ref)
  - [`ep_add_rlvar_view!`](@ref)
"""
function ep_sequential_sides(x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol,
                             rhs::Number, pv::Number)
    sgn = ifelse(op == :leq || op == :eq && rhs <= pv, -one(rhs), one(rhs))
    coef = coef .* sgn
    z = zero(eltype(coef))
    d = findall(>(z), coef)
    p = findall(<(z), coef)
    return x[d], coef[d], x[p], coef[p], rhs * sgn
end
"""
    ep_add_cvar_view!(tvs::AbstractVector, alg::CVaRVF,
                      x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol, rhs::Number,
                      alpha::Number, w::VecNum, pv::Number, eqn::AbstractString)

Lower one conditional value-at-risk view into the tail view constraint that its formulation needs.

# Algorithm

 1. [`LinearConditionalValueatRiskView`](@ref) checks the three conditions of the `Validation` section. Then it appends one [`LinearConditionalValueatRiskViewConstraint`](@ref) that holds `x`, `coef`, `alpha` and `rhs`.
 2. [`IntegerConditionalValueatRiskView`](@ref) sorts the loss series of each asset that the view names. The result is the ascending order `o`. It finds the length `sb` of the tail window of that asset with [`ep_sbar`](@ref).
 3. It keeps the last `sb` positions of `o` as `ord[k]`, and the losses at those positions as `xw[k]`.
 4. It appends one [`IntegerConditionalValueatRiskViewConstraint`](@ref) that holds those windows, `coef`, `alpha`, `op` and `rhs`.
 5. [`SequentialTailView`](@ref) orients the view and splits its assets with [`ep_sequential_sides`](@ref). It builds a [`SequentialConditionalValueatRiskViewConstraint`](@ref) with an empty surrogate row. It reads the first row at the prior `w` with [`ep_sequential_start`](@ref), and appends the constraint.

# Arguments

  - `tvs`: The vector of tail view constraints, to which the function appends.
  - `alg`: The formulation of the view.
  - `x`: The loss series of each asset that the view names.
  - `coef`: The coefficient that the view gives the CVaR of each asset.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `alpha`: The significance level of the view.
  - `w`: The prior probability weights.
  - `pv`: The prior value of the left hand side of the view.
  - `eqn`: The equation of the view, for the error messages.

# Validation

  - [`LinearConditionalValueatRiskView`](@ref) needs coefficients of one sign, an operator other than `<=`, and, for an equality, a target at or above the prior value of the left hand side.

# Returns

  - `nothing`: The function mutates `tvs` in-place.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`SequentialTailView`](@ref)
  - [`ep_sequential_sides`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_add_cvar_view!(tvs::AbstractVector, ::LinearConditionalValueatRiskView,
                           x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol,
                           rhs::Number, alpha::Number, w::VecNum, pv::Number,
                           eqn::AbstractString)
    @argcheck(all(>(zero(eltype(coef))), coef),
              ArgumentError("View `$(eqn)` carries coefficients of both signs. `LinearConditionalValueatRiskView` writes a positive combination of CVaRs, whose upper level set is convex; a relative view needs `IntegerConditionalValueatRiskView` or `SequentialTailView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `LinearConditionalValueatRiskView` bounds the CVaR from below only; use `IntegerConditionalValueatRiskView` or `SequentialTailView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior CVaR $(pv). `LinearConditionalValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `IntegerConditionalValueatRiskView` or `SequentialTailView`."))
    push!(tvs, LinearConditionalValueatRiskViewConstraint(x, coef, alpha, rhs))
    return nothing
end
function ep_add_cvar_view!(tvs::AbstractVector, alg::IntegerConditionalValueatRiskView,
                           x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol,
                           rhs::Number, alpha::Number, w::VecNum, pv::Number,
                           eqn::AbstractString)
    N = length(x)
    ord = Vector{Vector{Int}}(undef, N)
    xw = Vector{Vector{eltype(eltype(x))}}(undef, N)
    for (k, xj) in pairs(x)
        T = length(xj)
        o = sortperm(xj)
        sb = ep_sbar(alg.sbar, T, alpha, w, o)
        ord[k] = o[(T - sb + 1):T]
        xw[k] = xj[ord[k]]
    end
    push!(tvs, IntegerConditionalValueatRiskViewConstraint(ord, xw, coef, alpha, op, rhs))
    return nothing
end
function ep_add_cvar_view!(tvs::AbstractVector, alg::SequentialTailView,
                           x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol,
                           rhs::Number, alpha::Number, w::VecNum, pv::Number,
                           eqn::AbstractString)
    xd, cd, xp, cp, rhs = ep_sequential_sides(x, coef, op, rhs, pv)
    tv = SequentialConditionalValueatRiskViewConstraint(xd, cd, xp, cp,
                                                        zeros(typeof(rhs), length(w)),
                                                        zero(rhs), alpha, rhs, alg.iters,
                                                        alg.tol)
    push!(tvs, ep_sequential_start(tv, w))
    return nothing
end
"""
    ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                      alg::EVaRVF,
                      x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number, op::Symbol,
                      rhs::Number, w::VecNum, zstar::VecNum, pv::Number, eqn::AbstractString;
                      args::Tuple = (), kwargs::NamedTuple = (;),
                      zlo_frac::Option{<:Number} = nothing)

Lower one entropic value-at-risk view into the constraints that its formulation needs.

[`ConicEntropicValueatRiskView`](@ref) and [`SequentialTailView`](@ref) each make one tail view constraint. [`GridTailView`](@ref) makes linear rows on the posterior probabilities for the lower-bound half of the view, and a tail view constraint for the upper-bound half. So an equality view gets both.

# Algorithm

 1. [`ConicEntropicValueatRiskView`](@ref) checks the three conditions of the `Validation` section. Then it appends one [`ConicEntropicValueatRiskViewConstraint`](@ref) that holds `x`, `coef`, `alpha` and `rhs`.
 2. [`GridTailView`](@ref) checks that the view names one asset. It normalises `w` to sum to one, which gives `wi`.
 3. It builds the grid `z` of dual variables with [`ep_evar_grid`](@ref).
 4. It keeps the points whose row is finite, as `keep`, and raises an error if `keep` is empty. For the upper-bound half, it keeps only the points whose bound is positive, and raises an error if no point has a positive bound.
 5. For the lower-bound half of the view, it builds the row of each kept point with [`ep_evar_grid_row`](@ref). It multiplies the row by minus one, so that the row has the `<=` sense of the key `:ineq`, and adds the row to `epc` under that key with [`add_ep_constraint!`](@ref).
 6. For the upper-bound half of the view, it appends one [`GridEntropicValueatRiskViewConstraint`](@ref) that holds `x`, the kept grid, `alpha`, `rhs` and the big-M multiplier `M`.
 7. [`SequentialTailView`](@ref) orients the view and splits its assets with [`ep_sequential_sides`](@ref). It builds a [`SequentialEntropicValueatRiskViewConstraint`](@ref) with an empty surrogate row. It reads the first row at the prior `w` with [`ep_sequential_start`](@ref), and appends the constraint.

# Arguments

  - `epc`: The dictionary of entropy pooling constraints, from keys to `(lhs, rhs)` pairs.
  - `tvs`: The vector of tail view constraints, to which the function appends.
  - `alg`: The formulation of the view. A grid formulation also holds the number of steps and the tolerance of the anchor. A sequential formulation also holds the number of re-solves and their tolerance.
  - `x`: The loss series of each asset that the view names.
  - `coef`: The coefficient that the view gives the EVaR of each asset.
  - `alpha`: The significance level of the view.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `w`: The prior probability weights. The search for the centre of the grid starts from them. A sequential view reads its first surrogate row at them.
  - `zstar`: The dual variable that attains the prior EVaR of each asset.
  - `pv`: The prior value of the left hand side of the view.
  - `eqn`: The equation of the view, for the error messages.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: The lower end of the bracket of the dual variable, as a fraction of the upper end, which goes to [`ep_evar`](@ref).

# Validation

  - [`ConicEntropicValueatRiskView`](@ref) needs coefficients of one sign, an operator other than `<=`, and, for an equality, a target at or above the prior value of the left hand side.
  - [`GridTailView`](@ref) needs one asset. It also needs at least one grid point whose row is finite and, if the view has an upper-bound half, whose bound is positive. [`ep_evar_grid_row`](@ref) overflows at a dual variable near zero, and its bound underflows to zero there. Part of the grid is near zero when `pct` goes near one. All of the grid is near zero when `alpha * T` is less than one, because the minimiser of [`ep_evar`](@ref) is then at the end of its bracket. The function drops these points, and raises an error if it keeps no point.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridTailView`](@ref)
  - [`SequentialTailView`](@ref)
  - [`ep_evar_grid`](@ref)
  - [`ep_evar_grid_row`](@ref)
  - [`ep_sequential_sides`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                           ::ConicEntropicValueatRiskView, x::AbstractVector{<:VecNum},
                           coef::VecNum, alpha::Number, op::Symbol, rhs::Number, w::VecNum,
                           zstar::VecNum, pv::Number, eqn::AbstractString; args::Tuple = (),
                           kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)
    @argcheck(all(>(zero(eltype(coef))), coef),
              ArgumentError("View `$(eqn)` carries coefficients of both signs. `ConicEntropicValueatRiskView` writes a positive combination of EVaRs, whose upper level set is convex; a relative view needs `SequentialTailView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `ConicEntropicValueatRiskView` bounds the EVaR from below only; use `GridTailView` for a single asset, or `SequentialTailView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior EVaR $(pv). `ConicEntropicValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `GridTailView` for a single asset, or `SequentialTailView`."))
    push!(tvs, ConicEntropicValueatRiskViewConstraint(x, coef, alpha, rhs))
    return nothing
end
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector, alg::GridTailView,
                           x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number,
                           op::Symbol, rhs::Number, w::VecNum, zstar::VecNum, pv::Number,
                           eqn::AbstractString; args::Tuple = (), kwargs::NamedTuple = (;),
                           zlo_frac::Option{<:Number} = nothing)
    @argcheck(isone(length(x)),
              ArgumentError("View `$(eqn)` names $(length(x)) assets. `GridTailView` writes the EVaR of a single asset; use `ConicEntropicValueatRiskView` for a lower bound on a positive combination, and `SequentialTailView` for an upper bound or a relative view."))
    (; pct, K, M, iters, tol, tilt_iters) = alg
    x = x[1]
    wi = w ./ sum(w)
    z = ep_evar_grid(x, wi, alpha, op, rhs, zstar[1], pct, K; iters = iters, tol = tol,
                     tilt_iters = tilt_iters, args = args, kwargs = kwargs,
                     zlo_frac = zlo_frac)
    function row(zk)
        c, isc = ep_evar_grid_row(x, rhs, zk)
        return c, alpha * isc
    end
    # `exp((x - rhs) / z)` overflows at a dual variable near zero, and the bound of its row
    # underflows to zero there. The grid sits there when `pct` approaches one, and wholly
    # there when `alpha * T` falls below one, because the minimiser is then at the end of its
    # bracket.
    z = ep_add_grid_tail_view!(epc, z, op, row,
                               () -> "View `$(eqn)` builds no usable grid point. The row of every dual variable the grid spans overflows, or its bound underflows to zero, which happens when `alpha` ($(alpha)) leaves fewer than one observation in the tail, and when `pct` ($(pct)) approaches one. Raise `alpha`, or narrow `pct`.")
    if op == :leq || op == :eq
        push!(tvs, GridEntropicValueatRiskViewConstraint(x, z, alpha, rhs, M))
    end
    return nothing
end
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector, alg::SequentialTailView,
                           x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number,
                           op::Symbol, rhs::Number, w::VecNum, zstar::VecNum, pv::Number,
                           eqn::AbstractString; args::Tuple = (), kwargs::NamedTuple = (;),
                           zlo_frac::Option{<:Number} = nothing)
    xd, cd, xp, cp, rhs = ep_sequential_sides(x, coef, op, rhs, pv)
    tv = SequentialEntropicValueatRiskViewConstraint(xd, cd, xp, cp,
                                                     zeros(typeof(rhs), length(w)),
                                                     zero(rhs), alpha, rhs, alg.iters,
                                                     alg.tol, args, kwargs, zlo_frac)
    push!(tvs, ep_sequential_start(tv, w))
    return nothing
end
"""
    ep_add_grid_tail_view!(epc::AbstractDict, grid::AbstractVector, op::Symbol, row, msg)

Keep the usable points of a tail view grid, and add the rows of its lower-bound half.

The entropic and the relativistic methods of [`GridTailView`](@ref) share `ep_add_grid_tail_view!`. Both build a grid of points, drop the points whose row is not usable, and add one linear row for each kept point. They differ in what a point is and in how they build its row. Each gives its row builder to this function as `row`.

The function drops a point whose row is not finite, because a coefficient that is not finite goes to the solver as `NaN * x[j]`. The message `msg` of the caller names the setting that put the whole grid at such points.

The upper-bound half also drops a point whose bound is at or below zero. Its coefficients are positive and the posterior sums to one, so no posterior meets its row. The lower-bound half keeps such a point. Every posterior meets its row, so the row changes no posterior.

The rows of the lower-bound half keep the norm scale of [`add_ep_constraint!`](@ref). A row with a small bound is at the small end of the grid. A posterior that meets a lower bound moves mass toward the largest loss, where every row has its largest coefficient. So at such a posterior, a row with a small bound has a slack of many orders of magnitude. A division by the bound puts coefficients of the order of its reciprocal into the dual that [`OptimEntropyPooling`](@ref) solves.

# Algorithm

 1. Build the row of every point with `row`. Keep the points whose coefficients and right-hand side are all finite.
 2. Raise an error with `msg` when the function keeps no point.
 3. When the view has a lower-bound half, multiply the row of each kept point by minus one, so that the row has the `<=` sense of the key `:ineq`. Add the row to `epc` under that key with [`add_ep_constraint!`](@ref).
 4. When the view has an upper-bound half, keep only the points whose right-hand side is positive. Raise an error with `msg` when no point remains.
 5. Return the kept points. The caller puts them into the tail view constraint of the upper-bound half.

# Arguments

  - `epc`: The dictionary of entropy pooling constraints, from keys to `(lhs, rhs)` pairs.
  - `grid`: The points of the grid.
  - `op`: The comparison operator of the view.
  - `row`: A function that takes one point and returns the coefficients and the right-hand side of its row.
  - `msg`: A function of no arguments that returns the message of the error. The function calls it only when the grid keeps no point.

# Validation

  - At least one point of the grid has a finite row. If the view has an upper-bound half, at least one such row also has a positive bound. A grid that keeps no point raises an `ArgumentError` with the message `msg()`.

# Returns

  - `grid::AbstractVector`: The points of the grid whose row is finite and, if the view has an upper-bound half, whose bound is positive.

# Related

  - [`ep_add_evar_view!`](@ref)
  - [`ep_add_rlvar_view!`](@ref)
  - [`add_ep_constraint!`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_add_grid_tail_view!(epc::AbstractDict, grid::AbstractVector, op::Symbol, row,
                                msg)
    rows = map(row, grid)
    keep = findall(r -> all(isfinite, r[1]) && isfinite(r[2]), rows)
    @argcheck(!isempty(keep), ArgumentError(msg()))
    if op == :geq || op == :eq
        foreach(r -> add_ep_constraint!(epc, reshape(-r[1], 1, :), [-r[2]], :ineq),
                view(rows, keep))
    end
    # An upper-bound row whose bound is at or below zero holds at no posterior. Issue #1264.
    if op != :geq
        keep = filter(k -> rows[k][2] > zero(rows[k][2]), keep)
        @argcheck(!isempty(keep), ArgumentError(msg()))
    end
    return grid[keep]
end
"""
    ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector,
                       alg::RLVaRVF,
                       x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number,
                       kappa::Number, op::Symbol, rhs::Number, w::VecNum, zstar::VecNum,
                       pv::Number, eqn::AbstractString; args::Tuple = (),
                       kwargs::NamedTuple = (;),
                       bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Lower one relativistic value-at-risk view into the constraints that its formulation needs.

[`ConicRelativisticValueatRiskView`](@ref) and [`SequentialTailView`](@ref) each make one tail view constraint. [`GridTailView`](@ref) makes linear rows on the posterior probabilities for the lower-bound half of the view, and a tail view constraint for the upper-bound half. So an equality view gets both.

# Algorithm

 1. [`ConicRelativisticValueatRiskView`](@ref) checks the three conditions of the `Validation` section. Then it appends one [`ConicRelativisticValueatRiskViewConstraint`](@ref) that holds `x`, `coef`, `alpha`, `kappa` and `rhs`.
 2. [`GridTailView`](@ref) checks that the view names one asset. It normalises `w` to sum to one, which gives `wi`.
 3. It builds the grid `t`, `z` of primal points with [`ep_rlvar_grid`](@ref).
 4. It keeps the points whose row is finite, as `keep`, and raises an error if `keep` is empty. For the upper-bound half, it keeps only the points whose bound is positive, and raises an error if no point has a positive bound.
 5. For the lower-bound half of the view, it builds the row of each kept point with [`ep_rlvar_grid_row`](@ref). It multiplies the row by minus one, so that the row has the `<=` sense of the key `:ineq`, and adds the row to `epc` under that key with [`add_ep_constraint!`](@ref).
 6. For the upper-bound half of the view, it appends one [`GridRelativisticValueatRiskViewConstraint`](@ref) that holds `x`, the kept grid, `alpha`, `kappa`, `rhs` and the big-M multiplier `M`.
 7. [`SequentialTailView`](@ref) orients the view and splits its assets with [`ep_sequential_sides`](@ref). It builds a [`SequentialRelativisticValueatRiskViewConstraint`](@ref) with an empty surrogate row. It reads the first row at the prior `w` with [`ep_sequential_start`](@ref), and appends the constraint.

# Arguments

  - `epc`: The dictionary of entropy pooling constraints, from keys to `(lhs, rhs)` pairs.
  - `tvs`: The vector of tail view constraints, to which the function appends.
  - `alg`: The formulation of the view. A grid formulation also holds the number of steps and the tolerance of the anchor. A sequential formulation also holds the number of re-solves and their tolerance.
  - `x`: The loss series of each asset that the view names.
  - `coef`: The coefficient that the view gives the RLVaR of each asset.
  - `alpha`: The significance level of the view.
  - `kappa`: The deformation parameter of the view.
  - `op`: The comparison operator of the view.
  - `rhs`: The target value of the view.
  - `w`: The prior probability weights. The search for the centre of the grid starts from them. If that search does not converge, they set the shift of each grid point. A sequential view reads its first surrogate row at them.
  - `zstar`: The dual variable that attains the prior RLVaR of each asset.
  - `pv`: The prior value of the left hand side of the view. With `rhs`, it sets the translation of a grid that has its centre at the prior.
  - `eqn`: The equation of the view, for the error messages.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: The spans of the searches, which go to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Validation

  - [`ConicRelativisticValueatRiskView`](@ref) needs coefficients of one sign, an operator other than `<=`, and, for an equality, a target at or above the prior value of the left hand side.
  - [`GridTailView`](@ref) needs one asset. It also needs at least one grid point whose row is finite and, if the view has an upper-bound half, whose bound is positive. [`ep_rlvar_tail`](@ref) overflows at a dual variable near zero, and the grid is near zero when `kappa` goes near one. The bound `rhs - t - z * ln_kappa(1 / (alpha * T))` of a point is at or below zero when the target is below the value that the point can reach. The function drops these points, and raises an error if it keeps no point.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridTailView`](@ref)
  - [`SequentialTailView`](@ref)
  - [`ep_rlvar_grid`](@ref)
  - [`ep_rlvar_grid_row`](@ref)
  - [`ep_rlvar_tail`](@ref)
  - [`ep_sequential_sides`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector,
                            ::ConicRelativisticValueatRiskView, x::AbstractVector{<:VecNum},
                            coef::VecNum, alpha::Number, kappa::Number, op::Symbol,
                            rhs::Number, w::VecNum, zstar::VecNum, pv::Number,
                            eqn::AbstractString; args::Tuple = (), kwargs::NamedTuple = (;),
                            bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    @argcheck(all(>(zero(eltype(coef))), coef),
              ArgumentError("View `$(eqn)` carries coefficients of both signs. `ConicRelativisticValueatRiskView` writes a positive combination of RLVaRs, whose upper level set is convex; a relative view needs `SequentialTailView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `ConicRelativisticValueatRiskView` bounds the RLVaR from below only; use `GridTailView` for a single asset, or `SequentialTailView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior RLVaR $(pv). `ConicRelativisticValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `GridTailView` for a single asset, or `SequentialTailView`."))
    push!(tvs, ConicRelativisticValueatRiskViewConstraint(x, coef, alpha, kappa, rhs))
    return nothing
end
function ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector, alg::GridTailView,
                            x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number,
                            kappa::Number, op::Symbol, rhs::Number, w::VecNum,
                            zstar::VecNum, pv::Number, eqn::AbstractString;
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    @argcheck(isone(length(x)),
              ArgumentError("View `$(eqn)` names $(length(x)) assets. `GridTailView` writes the RLVaR of a single asset; use `ConicRelativisticValueatRiskView` for a lower bound on a positive combination, and `SequentialTailView` for an upper bound or a relative view."))
    (; pct, K, M, iters, tol, tilt_iters) = alg
    x = x[1]
    wi = w ./ sum(w)
    t, z = ep_rlvar_grid(x, wi, alpha, kappa, op, rhs, zstar[1], pv, pct, K; iters = iters,
                         tol = tol, tilt_iters = tilt_iters, args = args, kwargs = kwargs,
                         bracket = bracket)
    function row(g)
        return ep_rlvar_grid_row(x, rhs, g[1], g[2], alpha, kappa)
    end
    # `ep_rlvar_tail` overflows at a dual variable near zero, which is where the grid sits
    # when `kappa` approaches one. An upper-bound point whose bound is at or below zero is
    # dropped too.
    grid = ep_add_grid_tail_view!(epc, collect(zip(t, z)), op, row,
                                  () -> "View `$(eqn)` builds no usable grid point at `kappa = $(kappa)`. The tail function overflows at every dual variable the grid spans, or the bound of every row is at or below zero because the target lies below what each point can reach. State the view at a smaller `kappa`, or at a larger target.")
    if op == :leq || op == :eq
        push!(tvs,
              GridRelativisticValueatRiskViewConstraint(x, first.(grid), last.(grid), alpha,
                                                        kappa, rhs, M))
    end
    return nothing
end
function ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector, alg::SequentialTailView,
                            x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number,
                            kappa::Number, op::Symbol, rhs::Number, w::VecNum,
                            zstar::VecNum, pv::Number, eqn::AbstractString;
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    xd, cd, xp, cp, rhs = ep_sequential_sides(x, coef, op, rhs, pv)
    tv = SequentialRelativisticValueatRiskViewConstraint(xd, cd, xp, cp,
                                                         zeros(typeof(rhs), length(w)),
                                                         zero(rhs), alpha, kappa, rhs,
                                                         alg.iters, alg.tol, args, kwargs,
                                                         bracket)
    push!(tvs, ep_sequential_start(tv, w))
    return nothing
end
"""
    ep_tail_view_prior_args(tail_views::ConditionalValueatRiskView, w::VecNum)
    ep_tail_view_prior_args(tail_views::EntropicValueatRiskView, w::VecNum)
    ep_tail_view_prior_args(tail_views::RelativisticValueatRiskView, w::VecNum)

Return the trailing arguments under which a `prior(...)` reference of this tail view group resolves.

`ep_tail_view_prior_args` is one of the two functions that hold the differences between the measures for [`ep_tail_views!`](@ref). Its tuple names the statistic that [`get_pr_value`](@ref) reads, and holds the level, the prior probabilities and the settings of the search for that statistic. [`replace_prior_views`](@ref) passes the tuple on with no change. So a new measure needs a new method here, and no copy of the lowering.

# Arguments

  - `tail_views`: The tail view group whose settings the tuple holds.
  - `w`: The prior probability weights.

# Returns

  - `args::Tuple`: The trailing arguments of [`replace_prior_views`](@ref), which start with the tag of the statistic.

# Related

  - [`ep_tail_views!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_tail_view_prior_args(tail_views::ConditionalValueatRiskView, w::VecNum)
    return (:cvar, tail_views.alpha, StatsBase.pweights(w))
end
function ep_tail_view_prior_args(tail_views::EntropicValueatRiskView, w::VecNum)
    return (:evar, tail_views.alpha, StatsBase.pweights(w), tail_views.args,
            tail_views.kwargs, tail_views.zlo_frac)
end
function ep_tail_view_prior_args(tail_views::RelativisticValueatRiskView, w::VecNum)
    return (:rlvar, tail_views.alpha, tail_views.kappa, StatsBase.pweights(w),
            tail_views.args, tail_views.kwargs, tail_views.bracket)
end
"""
    ep_normalise_tail_view(terms::NamedTuple, X::MatNum, w::VecNum, eqn::AbstractString,
                           name::AbstractString)

Normalise the coefficients of a tail view, read the loss series of its assets, and tell whether its coefficients have both signs.

Every tail view goes through these steps. The function divides a view of one asset by its coefficient, so the target is a value of the measure. It multiplies a view of several assets whose coefficients have one sign by that sign. Then every coefficient is positive, and the view is a positive combination of measures, whose upper level set is convex. A view whose coefficients have both signs is a relative view, and the function keeps it as stated.

# Algorithm

 1. Read `mixed`, which is `true` if the coefficients have both signs.
 2. When the view names one asset, divide the view by the coefficient of that asset with [`ep_normalise_view_term`](@ref), which reverses the operator when the coefficient is negative. Set the coefficient to one.
 3. When the view names several assets and `mixed` is `false`, multiply both sides by the sign of the first coefficient with the same function. Then every coefficient is positive, and the operator reverses when the sign is negative.
 4. Read the loss series `x` of each asset as the negated column of its returns.
 5. Refuse a target that no reweighting of the sample on the support of `w` reaches, with [`ep_assert_reachable_view`](@ref).

# Arguments

  - `terms`: The resolved terms of the view, as [`ep_view_terms`](@ref) returns them.
  - `X`: The matrix of asset returns.
  - `w`: The prior probability weights.
  - `eqn`: The equation of the view, for the error messages.
  - `name`: The name of the risk measure, for the error messages.

# Returns

  - `x::AbstractVector{<:VecNum}`: The loss series of each asset that the view names.
  - `coef::VecNum`: The normalised coefficient of each asset.
  - `op::Symbol`: The operator of the normalised view.
  - `rhs::Number`: The target of the normalised view.
  - `mixed::Bool`: `true` if the coefficients have both signs.

# Related

  - [`ep_view_terms`](@ref)
  - [`ep_normalise_view_term`](@ref)
  - [`ep_assert_reachable_view`](@ref)
  - [`ep_add_tail_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_normalise_tail_view(terms::NamedTuple, X::MatNum, w::VecNum,
                                eqn::AbstractString, name::AbstractString)
    (; idx, coef, op, rhs) = terms
    z = zero(eltype(coef))
    mixed = any(<(z), coef) && any(>(z), coef)
    if isone(length(idx))
        op, rhs = ep_normalise_view_term(coef[1], op, rhs)
        coef = [one(eltype(coef))]
    elseif !mixed
        sgn = sign(coef[1])
        op, rhs = ep_normalise_view_term(sgn, op, rhs)
        coef = coef .* sgn
    end
    x = [-X[:, j] for j in idx]
    ep_assert_reachable_view(op, rhs, x, coef, w, eqn, name)
    return x, coef, op, rhs, mixed
end
"""
    ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                      tail_views::ConditionalValueatRiskView, alg, X::MatNum,
                      terms::NamedTuple, eqn::AbstractString, w::VecNum)
    ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                      tail_views::EntropicValueatRiskView, alg, X::MatNum,
                      terms::NamedTuple, eqn::AbstractString, w::VecNum)
    ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                      tail_views::RelativisticValueatRiskView, alg, X::MatNum,
                      terms::NamedTuple, eqn::AbstractString, w::VecNum)

Lower one resolved tail view into the constraints that its measure and its formulation need.

`ep_add_tail_view!` is the second of the two functions that hold the differences between the measures for [`ep_tail_views!`](@ref). [`ep_tail_views!`](@ref) parses the group, expands its asset groups, and resolves its prior references and its terms. This function does every step after that point, and these are the steps that the measures do not share. A new measure needs a new method here and in [`ep_tail_view_prior_args`](@ref).

Every measure accepts a view over several assets. Each method normalises the view with [`ep_normalise_tail_view`](@ref), which also tells whether the coefficients have both signs. It reads `pv`, the prior value of the left hand side, as the coefficient-weighted sum of the measures of the assets under `w`. Then it selects the formulation and lowers the view:

 1. A [`ConditionalValueatRiskView`](@ref) reads the prior CVaR of each asset with [`ConditionalValueatRisk`](@ref). It selects the formulation with [`ep_cvar_formulation`](@ref), and appends the constraint with [`ep_add_cvar_view!`](@ref).
 2. An [`EntropicValueatRiskView`](@ref) reads the prior EVaR of each asset, and the dual variable that attains it, with [`ep_evar`](@ref). It selects the formulation with [`ep_evar_formulation`](@ref), and appends the constraints with [`ep_add_evar_view!`](@ref).
 3. A [`RelativisticValueatRiskView`](@ref) reads the prior RLVaR of each asset, and the primal pair that attains it, with [`ep_rlvar`](@ref). It selects the formulation with [`ep_rlvar_formulation`](@ref), and appends the constraints with [`ep_add_rlvar_view!`](@ref).

# Arguments

  - `epc`: The dictionary of entropy pooling constraints, from keys to `(lhs, rhs)` pairs.
  - `tvs`: The vector of tail view constraints, to which the function appends.
  - `tail_views`: The tail view group of the view. It holds the level, the formulation setting and the settings of the searches.
  - `alg`: The formulation of this view, as [`ep_view_formulations`](@ref) expands it.
  - `X`: The matrix of asset returns.
  - `terms`: The resolved terms of the view, as [`ep_view_terms`](@ref) returns them.
  - `eqn`: The equation of the view, for the error messages.
  - `w`: The prior probability weights.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ep_tail_views!`](@ref)
  - [`ep_tail_view_prior_args`](@ref)
  - [`ep_normalise_tail_view`](@ref)
  - [`ep_add_cvar_view!`](@ref)
  - [`ep_add_evar_view!`](@ref)
  - [`ep_add_rlvar_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equations 2 to 6, and Section 2.3, Equations 7 to 11.
  - $(ref_dict[:EPRLVaR]) Section 2, Equations 1 to 5, and Section 3.2, Equation 7.
"""
function ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                           tail_views::ConditionalValueatRiskView, alg, X::MatNum,
                           terms::NamedTuple, eqn::AbstractString, w::VecNum)
    alpha = tail_views.alpha
    x, coef, op, rhs, mixed = ep_normalise_tail_view(terms, X, w, eqn, "CVaR")
    rm = ConditionalValueatRisk(; alpha = alpha, w = StatsBase.pweights(w))
    pv = sum(ci * rm(-xi) for (xi, ci) in zip(x, coef))
    alg = ep_cvar_formulation(alg, mixed, op, rhs, pv)
    ep_add_cvar_view!(tvs, alg, x, coef, op, rhs, alpha, w, pv, eqn)
    return nothing
end
function ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                           tail_views::EntropicValueatRiskView, alg, X::MatNum,
                           terms::NamedTuple, eqn::AbstractString, w::VecNum)
    (; alpha, args, kwargs, zlo_frac) = tail_views
    x, coef, op, rhs, mixed = ep_normalise_tail_view(terms, X, w, eqn, "EVaR")
    pv = zero(rhs)
    zstar = Vector{typeof(rhs)}(undef, length(x))
    for (k, (xi, ci)) in enumerate(zip(x, coef))
        e = ep_evar(xi, w, alpha; args = args, kwargs = kwargs, zlo_frac = zlo_frac)
        pv += ci * e.evar
        zstar[k] = e.z
    end
    alg = ep_evar_formulation(alg, mixed, isone(length(x)), op, rhs, pv)
    ep_add_evar_view!(epc, tvs, alg, x, coef, alpha, op, rhs, w, zstar, pv, eqn;
                      args = args, kwargs = kwargs, zlo_frac = zlo_frac)
    return nothing
end
function ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                           tail_views::RelativisticValueatRiskView, alg, X::MatNum,
                           terms::NamedTuple, eqn::AbstractString, w::VecNum)
    (; alpha, kappa, args, kwargs, bracket) = tail_views
    x, coef, op, rhs, mixed = ep_normalise_tail_view(terms, X, w, eqn, "RLVaR")
    pv = zero(rhs)
    zstar = Vector{typeof(rhs)}(undef, length(x))
    for (k, (xi, ci)) in enumerate(zip(x, coef))
        r = ep_rlvar(xi, w, alpha, kappa; args = args, kwargs = kwargs, bracket = bracket)
        pv += ci * r.rlvar
        zstar[k] = r.z
    end
    alg = ep_rlvar_formulation(alg, mixed, isone(length(x)), op, rhs, pv)
    ep_add_rlvar_view!(epc, tvs, alg, x, coef, alpha, kappa, op, rhs, w, zstar, pv, eqn;
                       args = args, kwargs = kwargs, bracket = bracket)
    return nothing
end
"""
    ep_tail_views!(tail_views::Nothing, args...; kwargs...)

Do nothing when a problem states no tail view.

`ep_tail_views!` is the function that lowers a group of tail views into entropy pooling constraints. This method is the branch for an absent group. It adds no constraint, so a caller can call the function with `tail_views = nothing` and needs no special case.

# Arguments

  - `tail_views::Nothing`: Shows that the problem states no tail view.
  - `args...`: More positional arguments, which the method does not read.
  - `kwargs...`: More keyword arguments, which the method does not read.

# Returns

  - `nothing`.

# Related

  - [`ep_tail_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_tail_views!(tail_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_tail_views!(tail_views::AbstractVector{<:AbstractEntropyPoolingTailViewEstimator},
                   args...; kwargs...)

Lower each group of tail views under its own settings.

The function lowers each [`AbstractEntropyPoolingTailViewEstimator`](@ref) of the vector in turn. All the groups add their constraints to the same constraint set, so one entropy pooling solve meets all of them.

# Algorithm

 1. Lower each group of `tail_views` in turn, and pass `args...` and `kwargs...` to each call.
 2. Return `nothing`. Each call already wrote its constraints into `epc` and `tvs`.

# Arguments

  - `tail_views`: The vector of tail view groups.
  - `args...`: More positional arguments, which go to [`ep_tail_views!`](@ref).
  - `kwargs...`: More keyword arguments, which go to [`ep_tail_views!`](@ref).

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ep_tail_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_tail_views!(tail_views::AbstractVector{<:AbstractEntropyPoolingTailViewEstimator},
                        args...; kwargs...)
    for tail_view in tail_views
        ep_tail_views!(tail_view, args...; kwargs...)
    end
    return nothing
end
"""
    ep_tail_views!(tail_views::AbstractEntropyPoolingTailViewEstimator, epc::AbstractDict,
                   tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                   w::VecNum; strict::Bool = false,
                   ledger::Option{<:AbstractVector} = nothing)

Parse a group of tail views and lower them into entropy pooling constraints.

`ep_tail_views!` lowers every tail view family. It parses the view equations of a [`LinearConstraintEstimator`](@ref), and replaces the prior references with their values. It resolves the asset names against the universe, selects a formulation for each view, and appends the constraints that the formulation needs. This function solves nothing, as against the recursive algorithm of [`MeucciEntropyPoolingPrior`](@ref). The views become part of the one entropy pooling problem that [`entropy_pooling`](@ref) solves.

The function accepts `==`, `>=` and `<=`. A group name expands to its members, and each member takes the coefficient of the group. So a view on a group constrains the sum of the risk measures of the members, not their average. A view whose coefficients have one sign is a positive combination of measures, and its lower-bound form is convex. A view whose coefficients have both signs is a relative view, and its lower-bound form is not convex in general.

These two functions hold every difference between the conditional, the entropic and the relativistic value at risk. So a fourth measure needs two methods, and no fourth copy of this function:

  - [`ep_tail_view_prior_args`](@ref) names the statistic to which a `prior(...)` reference resolves, and the settings of its search.
  - [`ep_add_tail_view!`](@ref) checks the shape of the view, reads its prior value, selects its formulation, and appends its constraints.

# Algorithm

 1. Parse the view equations of `tail_views.views.val`. The result is one [`ParsingResult`](@ref) for each view.
 2. Replace every group name with the assets of the group. The result is one term for each member.
 3. Replace every `prior(...)` reference with the prior value of the measure, through [`replace_prior_views`](@ref), under the tuple that [`ep_tail_view_prior_args`](@ref) gives.
 4. Expand the formulation setting to one formulation for each view with [`ep_view_formulations`](@ref). The result is `algs`.
 5. For each view in turn, resolve its terms with [`ep_view_terms`](@ref). Drop the view when the universe holds no name of the view.
 6. Lower the view with [`ep_add_tail_view!`](@ref).

# Arguments

  - `tail_views`: The tail view group.
  - `epc`: The dictionary of entropy pooling constraints, from keys to `(lhs, rhs)` pairs.
  - `tvs`: The vector of tail view constraints, to which the function appends.
  - `pr`: The prior result. The function reads its returns matrix.
  - `sets`: The universe sets that map the asset names to indices.
  - `w`: The prior probability weights.
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.
  - $(arg_dict[:ledger]) The group expansion and [`ep_view_terms`](@ref) fill it.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ep_tail_view_prior_args`](@ref)
  - [`ep_add_tail_view!`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`RelativisticValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail]) Section 2.2, Equations 2 to 6, and Section 2.3, Equations 7 to 11.
  - $(ref_dict[:EPRLVaR]) Section 2, Equations 1 to 5, and Section 3.2, Equation 7.
"""
function ep_tail_views!(tail_views::AbstractEntropyPoolingTailViewEstimator,
                        epc::AbstractDict, tvs::AbstractVector, pr::AbstractPriorResult,
                        sets::UniverseSets, w::VecNum; strict::Bool = false,
                        ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    views = parse_equation(tail_views.views.val; ops1 = ("==", ">=", "<="),
                           ops2 = (:call, :(==), :(>=), :(<=)), datatype = eltype(X))
    views = replace_group_by_assets(views, sets, false, true, false; ledger = ledger)
    views = replace_prior_views(views, pr, sets, ep_tail_view_prior_args(tail_views, w)...;
                                strict = strict)
    if !isa(views, AbstractVector)
        views = [views]
    end
    algs = ep_view_formulations(tail_views.alg, length(views), :alg)
    for (res, algi) in zip(views, algs)
        terms = ep_view_terms(res, sets, X; strict = strict, ledger = ledger)
        if isnothing(terms)
            continue
        end
        ep_add_tail_view!(epc, tvs, tail_views, algi, X, terms, res.eqn, w)
    end
    return nothing
end

"""
    ep_diagnose_tail_view!(model::JuMP.Model, p,
                           tv::Union{LinearConditionalValueatRiskViewConstraint,
                                     ConicEntropicValueatRiskViewConstraint,
                                     ConicRelativisticValueatRiskViewConstraint})

Add an exact tail view constraint to the feasibility programme of [`ep_diagnose_views`](@ref).

The three dual tail view constraints state a lower bound on a positive combination of concave measures. The bound is a convex set of the posterior probabilities, and the dual blocks of [`ep_tail_dual_block!`](@ref) state it exactly. So the method adds the rows of [`add_ep_tail_view!`](@ref), with the scale one. The solvers of the programme must then take the cones of those blocks: the exponential cone for an entropic view, and the power cone for a relativistic view.

# Arguments

  - `model`: The JuMP model of the feasibility programme.
  - `p`: The posterior probability variables of the model.
  - `tv`: Dual tail view constraint.

# Returns

  - `nothing`: The function mutates `model` in place.

# Related

  - [`ep_diagnose_views`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`LinearConditionalValueatRiskViewConstraint`](@ref)
  - [`ConicEntropicValueatRiskViewConstraint`](@ref)
  - [`ConicRelativisticValueatRiskViewConstraint`](@ref)
"""
function ep_diagnose_tail_view!(model::JuMP.Model, p,
                                tv::Union{LinearConditionalValueatRiskViewConstraint,
                                          ConicEntropicValueatRiskViewConstraint,
                                          ConicRelativisticValueatRiskViewConstraint})
    return add_ep_tail_view!(model, p, tv, one(eltype(tv.coef)))
end

public ep_add_cvar_view!, ep_add_evar_view!, ep_add_rlvar_view!
