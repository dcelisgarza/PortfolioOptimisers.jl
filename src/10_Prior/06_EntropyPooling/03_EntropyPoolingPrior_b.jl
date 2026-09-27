"""
    ep_check_tail_window(tv::IntegerConditionalValueatRiskViewConstraint, w::VecNum)

Warn where the window of an integer conditional value-at-risk view can restrict the posterior.

The model of [`IntegerConditionalValueatRiskView`](@ref) admits the posteriors that put at least `alpha` of their mass on the `sbar` largest losses of each asset. Where that restriction binds, the window holds exactly `alpha`, and the method warns. A window that holds more is not active at the posterior, so a small change of the posterior does not meet it.

# Algorithm

 1. For each asset of the view, skip a window that holds every observation, since it restricts nothing.
 2. Sum `w` over the window, and warn where the sum exceeds `alpha` by no more than `alpha` times the cube root of the machine epsilon of `w`. That margin is far above the feasibility tolerance of a solver and far below the mass of one observation.

# Arguments

  - `tv`: Integer conditional value-at-risk view constraint.
  - `w`: Posterior probabilities of the last solve.

# Returns

  - `nothing`.

# Related

  - [`ep_check_tail_window`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_check_tail_window(tv::IntegerConditionalValueatRiskViewConstraint, w::VecNum)
    (; ord, alpha) = tv
    for o in ord
        # The window binds when it holds no more than `alpha`, since the model asks it to hold
        # at least that. A window of the whole sample restricts nothing.
        if length(o) < length(w) && sum(view(w, o)) - alpha <= alpha * cbrt(eps(eltype(w)))
            @warn("The integer CVaR view reads the $(length(o)) largest losses, and the posterior puts only the tail mass $(alpha) on them, so the window binds and a posterior of smaller divergence can lie outside it. Raise `sbar` in `IntegerConditionalValueatRiskView`.")
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

Extract the Entropic Value-at-Risk (EVaR) for asset `i` from a prior result.

`get_pr_value` computes the EVaR at confidence level `alpha` for the asset indexed by `i` from the prior result object `pr`, by minimising the scalar objective of the sample EVaR formula with [`ep_evar`](@ref). The observations carry `w`, the weights the initial prior result was read at. A `w` of `nothing` leaves them uniform.

# Arguments

  - `pr`: Prior result containing asset return information. Only its returns matrix is read, under the weights `w` names.
  - `i`: Index of the asset.
  - `::Val{:evar}`: Dispatch tag for EVaR extraction.
  - `alpha`: Confidence level (e.g. `0.05` for 5% EVaR).
  - $(arg_dict[:oow])
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: Lower end of the bracket of the dual variable, as a fraction of the upper end, forwarded to [`ep_evar`](@ref).

# Returns

  - `evar::Number`: Entropic Value-at-Risk for asset `i` at level `alpha`.

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

Extract the Relativistic Value-at-Risk (RLVaR) for asset `i` from a prior result.

`get_pr_value` computes the RLVaR at confidence level `alpha` and deformation parameter `kappa` for the asset indexed by `i` from the prior result object `pr`, by minimising the primal objective of the sample RLVaR with [`ep_rlvar`](@ref). The observations carry `w`, the weights the initial prior result was read at, on the reasoning the entropic value at risk method above gives.

# Arguments

  - `pr`: Prior result containing asset return information. Only its returns matrix is read, under the weights `w` names.
  - `i`: Index of the asset.
  - `::Val{:rlvar}`: Dispatch tag for RLVaR extraction.
  - `alpha`: Confidence level (e.g. `0.05` for 5% RLVaR).
  - `kappa`: Deformation parameter, in `(0, 1)`.
  - $(arg_dict[:oow])
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: Spans of the searches, forwarded to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Returns

  - `rlvar::Number`: Relativistic Value-at-Risk for asset `i` at level `alpha` and deformation `kappa`.

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
    ep_view_terms(res::ParsingResult, sets::UniverseSets, X::MatNum; strict::Bool = false)

Resolve one parsed tail view into the assets it names, their coefficients, its operator and its target.

`ep_view_terms` routes a [`ParsingResult`](@ref) through [`get_linear_constraints`](@ref), which resolves the variable names against the universe and reports the ones it cannot place, then undoes the sign flip that entry point applies to a `>=` equation so the operator survives. The linear view machinery never needs the operator back, because a row of `A x <= b` carries it; a tail view does, because each operator picks a different formulation.

# Algorithm

 1. Resolve the view against the universe with [`get_linear_constraints`](@ref), giving `lc`. Return `nothing` where it places no name of the view.
 2. Read the sign `sgn` and the inequality flag of the view's operator with [`comparison_sign_ineq_flag`](@ref), and pick from them the operator `op` the view carries and the block `blk` it landed in.
 3. Scale the row `A` and the target `rhs` by `sgn`, which undoes the flip [`get_linear_constraints`](@ref) applies to a `>=` equation.
 4. Return the indices of the non-zero entries of `A`, the coefficients at those indices, `op` and `rhs`.

# Arguments

  - `res`: Parsed view constraint.
  - `sets`: Asset set mapping asset names to indices.
  - `X`: Asset returns matrix, read for its element type.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: If no name in the view could be placed in the universe.
  - `terms::@NamedTuple{idx::VecInt, coef::VecNum, op::Symbol, rhs::Number}`: The assets the view names, their coefficients, its operator (`:eq`, `:geq` or `:leq`) and its target.

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

Divide a single-asset tail view by its coefficient.

Returns the operator and target of the equivalent view whose coefficient is one, flipping the operator when the coefficient is negative.

# Algorithm

 1. Divide the target by the coefficient, giving `rhs`.
 2. Where the coefficient is negative, exchange `:geq` and `:leq`, giving `op`. An equality is unchanged, because dividing both sides by a negative number preserves it.

# Arguments

  - `coef`: Coefficient the view gives the asset's risk measure.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.

# Returns

  - `op::Symbol`: Operator of the normalised view.
  - `rhs::Number`: Target of the normalised view.

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

Spread the tail view formulation setting of an entropy pooling prior over its views.

A single formulation applies to every view, a vector supplies one per view, and `nothing` leaves the choice to the view.

# Arguments

  - `alg`: Formulation setting.
  - `N`: Number of views.
  - `key`: Field name, used in the error message.

# Validation

  - If `alg` is a vector, `length(alg) == N`.

# Returns

  - `algs::AbstractVector`: One entry per view.

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

Resolve the number of largest losses the integer conditional value-at-risk formulation considers.

# Algorithm

 1. An `Integer` `sbar` is a count. Return it, capped at `T`.
 2. Any other number is a fraction of `T`. Return `ceil(Int, sbar * T)`, never below one and capped at `T`.
 3. `nothing` takes the rule of thumb of [EPTail](@cite). Walk the losses from the largest down, accumulating the prior probabilities, and stop at the position `s` at which they first reach `alpha`.
 4. Return twice `s`, never below `ceil(Int, 2 * alpha * T)`, never below one and capped at `T`. A view above the prior CVaR moves mass into the tail and needs about the position `s`; a view below it moves mass out and needs more.

# Arguments

  - `sbar`: Setting held by [`IntegerConditionalValueatRiskView`](@ref). An `Integer` is a count, a fraction in `(0, 1)` is a fraction of `T`, and `nothing` applies the rule of thumb of [EPTail](@cite).
  - `T`: Number of observations.
  - `alpha`: Significance level of the view.
  - `w`: Prior probability weights.
  - `ord`: Indices of the losses in ascending order, so the largest loss is last.

# Returns

  - `sbar::Int`: Number of largest losses, in `1:T`.

# Related

  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)

# References

  - $(ref_dict[:EPTail])
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

Reject a tail view no reweighting of the sample can reach.

A tail risk measure of a reweighted sample lies between the smallest and the largest loss the sample holds where the prior probability is positive, so a coefficient-weighted sum of measures lies between the sums of those bounds, and a view outside that band is infeasible however the probabilities move. The band is exact for one asset, and an outer bound for several: a reweighting that puts every asset at its worst loss at once need not exist. The posterior of entropy pooling puts no mass where the prior puts none, so an observation of zero prior probability widens nothing.

# Algorithm

 1. Read `hi`, the sum over the assets of the coefficient times the largest loss where the coefficient is positive, and times the smallest loss where it is negative, over the observations with `w > 0`. Read `lo` the other way round.
 2. Where `op` asks the statistic to reach or exceed `rhs`, raise unless `rhs` sits below `hi`.
 3. Where `op` asks the statistic to reach or fall below `rhs`, raise unless `rhs` sits above `lo`.

# Arguments

  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `x`: Per asset the view names, its loss series.
  - `coef`: Per asset, the coefficient the view gives its risk measure.
  - `w`: Prior probability weights.
  - `eqn`: Equation of the view, used in the error message.
  - `name`: Name of the view family, used in the error message.

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
    ep_cvar_formulation(alg::Option{<:AbstractConditionalValueatRiskViewFormulation},
                        mixed::Bool, op::Symbol, rhs::Number, pv::Number)

Pick the formulation of one conditional value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`LinearConditionalValueatRiskView`](@ref) wherever it expresses the view exactly, which is every view whose lower level set is convex, and [`IntegerConditionalValueatRiskView`](@ref) otherwise: a view whose coefficients carry both signs, an upper bound, and an equality below the prior value of the left hand side.

The branch each input takes:

| `alg`     | `mixed` | `op`   | `rhs` against `pv` | Branch                                      |
|:--------- |:------- |:------ |:------------------ |:------------------------------------------- |
| stated    | any     | any    | any                | `alg`, unchanged                            |
| `nothing` | `false` | `:geq` | any                | [`LinearConditionalValueatRiskView`](@ref)  |
| `nothing` | `false` | `:eq`  | `rhs >= pv`        | [`LinearConditionalValueatRiskView`](@ref)  |
| `nothing` | `false` | `:eq`  | `rhs < pv`         | [`IntegerConditionalValueatRiskView`](@ref) |
| `nothing` | `false` | `:leq` | any                | [`IntegerConditionalValueatRiskView`](@ref) |
| `nothing` | `true`  | any    | any                | [`IntegerConditionalValueatRiskView`](@ref) |

[`SequentialConditionalValueatRiskView`](@ref) is never the default. It writes every view the integer formulation does with no integer variable, but its posterior is a local minimiser of the divergence, and `nothing` stands for the exact formulation.

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `mixed`: Whether the coefficients of the view carry both signs.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior value of the view's left hand side.

# Returns

  - `alg::AbstractConditionalValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_cvar_formulation(alg::AbstractConditionalValueatRiskViewFormulation, args...)
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
    ep_evar_formulation(alg::Option{<:AbstractEntropicValueatRiskViewFormulation}, mixed::Bool,
                        single::Bool, op::Symbol, rhs::Number, pv::Number)

Pick the formulation of one entropic value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`ConicEntropicValueatRiskView`](@ref) wherever it expresses the view exactly, which is every view whose lower level set is convex. An upper bound and an equality below the prior value take [`GridEntropicValueatRiskView`](@ref) when the view names one asset. The grid is one asset's, so the same views over several assets take [`SequentialEntropicValueatRiskView`](@ref), the one formulation that expresses them. So does a view whose coefficients carry both signs.

The branch each input takes:

| `alg`     | `mixed` | `single` | `op`   | `rhs` against `pv` | Branch                                      |
|:--------- |:------- |:-------- |:------ |:------------------ |:------------------------------------------- |
| stated    | any     | any      | any    | any                | `alg`, unchanged                            |
| `nothing` | `false` | any      | `:geq` | any                | [`ConicEntropicValueatRiskView`](@ref)      |
| `nothing` | `false` | any      | `:eq`  | `rhs >= pv`        | [`ConicEntropicValueatRiskView`](@ref)      |
| `nothing` | `false` | `true`   | `:eq`  | `rhs < pv`         | [`GridEntropicValueatRiskView`](@ref)       |
| `nothing` | `false` | `true`   | `:leq` | any                | [`GridEntropicValueatRiskView`](@ref)       |
| `nothing` | `false` | `false`  | `:eq`  | `rhs < pv`         | [`SequentialEntropicValueatRiskView`](@ref) |
| `nothing` | `false` | `false`  | `:leq` | any                | [`SequentialEntropicValueatRiskView`](@ref) |
| `nothing` | `true`  | `false`  | any    | any                | [`SequentialEntropicValueatRiskView`](@ref) |

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `mixed`: Whether the coefficients of the view carry both signs.
  - `single`: Whether the view names one asset.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior value of the view's left hand side.

# Returns

  - `alg::AbstractEntropicValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_evar_formulation(alg::AbstractEntropicValueatRiskViewFormulation, args...)
    return alg
end
function ep_evar_formulation(::Nothing, mixed::Bool, single::Bool, op::Symbol, rhs::Number,
                             pv::Number)
    return if mixed
        SequentialEntropicValueatRiskView()
    elseif op == :geq || op == :eq && rhs >= pv
        ConicEntropicValueatRiskView()
    elseif single
        GridEntropicValueatRiskView()
    else
        # The grid is one asset's. Issue #1287.
        SequentialEntropicValueatRiskView()
    end
end
"""
    ep_rlvar_formulation(alg::Option{<:AbstractRelativisticValueatRiskViewFormulation},
                         mixed::Bool, single::Bool, op::Symbol, rhs::Number, pv::Number)

Pick the formulation of one relativistic value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`ConicRelativisticValueatRiskView`](@ref) wherever it expresses the view exactly, which is every view whose lower level set is convex. An upper bound and an equality below the prior value take [`GridRelativisticValueatRiskView`](@ref) when the view names one asset. The grid is one asset's, so the same views over several assets take [`SequentialRelativisticValueatRiskView`](@ref), the one formulation that expresses them. So does a view whose coefficients carry both signs.

The branch each input takes:

| `alg`     | `mixed` | `single` | `op`   | `rhs` against `pv` | Branch                                          |
|:--------- |:------- |:-------- |:------ |:------------------ |:----------------------------------------------- |
| stated    | any     | any      | any    | any                | `alg`, unchanged                                |
| `nothing` | `false` | any      | `:geq` | any                | [`ConicRelativisticValueatRiskView`](@ref)      |
| `nothing` | `false` | any      | `:eq`  | `rhs >= pv`        | [`ConicRelativisticValueatRiskView`](@ref)      |
| `nothing` | `false` | `true`   | `:eq`  | `rhs < pv`         | [`GridRelativisticValueatRiskView`](@ref)       |
| `nothing` | `false` | `true`   | `:leq` | any                | [`GridRelativisticValueatRiskView`](@ref)       |
| `nothing` | `false` | `false`  | `:eq`  | `rhs < pv`         | [`SequentialRelativisticValueatRiskView`](@ref) |
| `nothing` | `false` | `false`  | `:leq` | any                | [`SequentialRelativisticValueatRiskView`](@ref) |
| `nothing` | `true`  | `false`  | any    | any                | [`SequentialRelativisticValueatRiskView`](@ref) |

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `mixed`: Whether the coefficients of the view carry both signs.
  - `single`: Whether the view names one asset.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior value of the view's left hand side.

# Returns

  - `alg::AbstractRelativisticValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_rlvar_formulation(alg::AbstractRelativisticValueatRiskViewFormulation, args...)
    return alg
end
function ep_rlvar_formulation(::Nothing, mixed::Bool, single::Bool, op::Symbol, rhs::Number,
                              pv::Number)
    return if mixed
        SequentialRelativisticValueatRiskView()
    elseif op == :geq || op == :eq && rhs >= pv
        ConicRelativisticValueatRiskView()
    elseif single
        GridRelativisticValueatRiskView()
    else
        # The grid is one asset's. Issue #1287.
        SequentialRelativisticValueatRiskView()
    end
end
"""
    ep_sequential_sides(x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol, rhs::Number,
                        pv::Number)

Orient a tail view as a lower bound, and split its assets into the dual side and the primal side of a sequential formulation.

# Algorithm

 1. Orient the view. An upper bound, and an equality whose target sits at or below the prior value of the left hand side, are negated on both sides, so the view reads `>=`. A lower bound, and an equality the prior sits below, are kept. An equality is therefore written as the bound the prior violates, which the entropy minimiser makes tight.
 2. Put every asset whose oriented coefficient is positive on the dual side, whose measure is concave in the probabilities and takes its exact dual block. Put every other asset on the primal side, whose measure takes a linear upper bound.

# Arguments

  - `x`: Per asset the view names, its loss series.
  - `coef`: Per asset, the coefficient the view gives its risk measure.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior value of the view's left hand side.

# Returns

  - `xd::AbstractVector{<:VecNum}`: Loss series of the assets on the dual side.
  - `cd::VecNum`: Their oriented coefficients, positive.
  - `xp::AbstractVector{<:VecNum}`: Loss series of the assets on the primal side.
  - `cp::VecNum`: Their oriented coefficients, negative.
  - `rhs::Number`: Target of the oriented view.

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
    ep_add_cvar_view!(tvs::AbstractVector, alg::AbstractConditionalValueatRiskViewFormulation,
                      x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol, rhs::Number,
                      alpha::Number, w::VecNum, pv::Number, eqn::AbstractString)

Lower one conditional value-at-risk view into the tail view constraint its formulation needs.

# Algorithm

 1. [`LinearConditionalValueatRiskView`](@ref) checks the three preconditions below, then appends one [`LinearConditionalValueatRiskViewConstraint`](@ref) carrying `x`, `coef`, `alpha` and `rhs`.
 2. [`IntegerConditionalValueatRiskView`](@ref) sorts the loss series of each asset the view names, giving the ascending order `o`, and resolves the length `sb` of that asset's tail window with [`ep_sbar`](@ref).
 3. It keeps the last `sb` positions of `o` as `ord[k]`, and the losses at those positions as `xw[k]`.
 4. It appends one [`IntegerConditionalValueatRiskViewConstraint`](@ref) carrying those windows, `coef`, `alpha`, `op` and `rhs`.
 5. [`SequentialConditionalValueatRiskView`](@ref) orients the view and splits its assets with [`ep_sequential_sides`](@ref), builds a [`SequentialConditionalValueatRiskViewConstraint`](@ref) with an empty surrogate row, reads its first row from the prior `w` with [`ep_sequential_start`](@ref), and appends it.

# Arguments

  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view.
  - `x`: Per asset the view names, its loss series.
  - `coef`: Per asset, the coefficient the view gives its CVaR.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `alpha`: Significance level of the view.
  - `w`: Prior probability weights.
  - `pv`: Prior value of the view's left hand side.
  - `eqn`: Equation of the view, used in the error messages.

# Validation

  - [`LinearConditionalValueatRiskView`](@ref) needs coefficients of one sign, an operator other than `<=`, and, for an equality, a target at or above the prior value of the left hand side.

# Returns

  - `nothing`: The function mutates `tvs` in-place.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`ep_sequential_sides`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_add_cvar_view!(tvs::AbstractVector, ::LinearConditionalValueatRiskView,
                           x::AbstractVector{<:VecNum}, coef::VecNum, op::Symbol,
                           rhs::Number, alpha::Number, w::VecNum, pv::Number,
                           eqn::AbstractString)
    @argcheck(all(>(zero(eltype(coef))), coef),
              ArgumentError("View `$(eqn)` carries coefficients of both signs. `LinearConditionalValueatRiskView` writes a positive combination of CVaRs, whose lower level set is convex; a relative view needs `IntegerConditionalValueatRiskView` or `SequentialConditionalValueatRiskView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `LinearConditionalValueatRiskView` bounds the CVaR from below only; use `IntegerConditionalValueatRiskView` or `SequentialConditionalValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior CVaR $(pv). `LinearConditionalValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `IntegerConditionalValueatRiskView` or `SequentialConditionalValueatRiskView`."))
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
function ep_add_cvar_view!(tvs::AbstractVector, alg::SequentialConditionalValueatRiskView,
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
                      alg::AbstractEntropicValueatRiskViewFormulation,
                      x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number, op::Symbol,
                      rhs::Number, w::VecNum, zstar::VecNum, pv::Number, eqn::AbstractString;
                      args::Tuple = (), kwargs::NamedTuple = (;),
                      zlo_frac::Option{<:Number} = nothing)

Lower one entropic value-at-risk view into the constraints its formulation needs.

[`ConicEntropicValueatRiskView`](@ref) and [`SequentialEntropicValueatRiskView`](@ref) produce one tail view constraint each. [`GridEntropicValueatRiskView`](@ref) produces linear rows on the posterior probabilities for the lower-bound half of the view, and a tail view constraint for the upper-bound half, so an equality view produces both.

# Algorithm

 1. [`ConicEntropicValueatRiskView`](@ref) checks the three preconditions below, then appends one [`ConicEntropicValueatRiskViewConstraint`](@ref) carrying `x`, `coef`, `alpha` and `rhs`.
 2. [`GridEntropicValueatRiskView`](@ref) checks that the view names one asset, and normalises `w` to sum to one, giving `wi`.
 3. It builds the grid `z` of dual variables with [`ep_evar_grid`](@ref).
 4. It keeps the points whose row is finite, giving `keep`, and raises where `keep` is empty. For the upper-bound half it keeps only the points whose bound is positive, and raises where none is.
 5. For the lower-bound half of the view, it builds the row of each kept point with [`ep_evar_grid_row`](@ref), and adds it to `epc` under `:ineq` with [`add_ep_constraint!`](@ref), negated so the row reads as the `<=` sense that key states.
 6. For the upper-bound half of the view, it appends one [`GridEntropicValueatRiskViewConstraint`](@ref) carrying `x`, the kept grid, `alpha`, `rhs` and the big-M multiplier `M`.
 7. [`SequentialEntropicValueatRiskView`](@ref) orients the view and splits its assets with [`ep_sequential_sides`](@ref), builds a [`SequentialEntropicValueatRiskViewConstraint`](@ref) with an empty surrogate row, reads its first row from the prior `w` with [`ep_sequential_start`](@ref), and appends it.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view. A grid formulation is also where the number of steps and the tolerance of the anchor live, and a sequential one where the number of re-solves and their tolerance live.
  - `x`: Per asset the view names, its loss series.
  - `coef`: Per asset, the coefficient the view gives its EVaR.
  - `alpha`: Significance level of the view.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `w`: Prior probability weights. They start the search for the dual variable the grid is centred on, and they are where the first surrogate row of a sequential view is read.
  - `zstar`: Per asset, the dual variable that attains its prior EVaR.
  - `pv`: Prior value of the view's left hand side.
  - `eqn`: Equation of the view, used in the error messages.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: Lower end of the bracket of the dual variable, as a fraction of the upper end, forwarded to [`ep_evar`](@ref).

# Validation

  - [`ConicEntropicValueatRiskView`](@ref) needs coefficients of one sign, an operator other than `<=`, and, for an equality, a target at or above the prior value of the left hand side.
  - [`GridEntropicValueatRiskView`](@ref) needs one asset, and at least one grid point whose row is finite, and whose bound is positive where the view carries an upper-bound half. [`ep_evar_grid_row`](@ref) overflows at a dual variable near zero, and its bound underflows to zero there. The grid sits there when `pct` approaches one, and wholly there when `alpha * T` falls below one, because [`ep_evar`](@ref)'s minimiser is then at the end of its bracket. Those points are dropped, and a grid that keeps none of them raises.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)
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
              ArgumentError("View `$(eqn)` carries coefficients of both signs. `ConicEntropicValueatRiskView` writes a positive combination of EVaRs, whose lower level set is convex; a relative view needs `SequentialEntropicValueatRiskView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `ConicEntropicValueatRiskView` bounds the EVaR from below only; use `GridEntropicValueatRiskView` for a single asset, or `SequentialEntropicValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior EVaR $(pv). `ConicEntropicValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `GridEntropicValueatRiskView` for a single asset, or `SequentialEntropicValueatRiskView`."))
    push!(tvs, ConicEntropicValueatRiskViewConstraint(x, coef, alpha, rhs))
    return nothing
end
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                           alg::GridEntropicValueatRiskView, x::AbstractVector{<:VecNum},
                           coef::VecNum, alpha::Number, op::Symbol, rhs::Number, w::VecNum,
                           zstar::VecNum, pv::Number, eqn::AbstractString; args::Tuple = (),
                           kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)
    @argcheck(isone(length(x)),
              ArgumentError("View `$(eqn)` names $(length(x)) assets. `GridEntropicValueatRiskView` writes the EVaR of a single asset; use `ConicEntropicValueatRiskView` for a lower bound on a positive combination, and `SequentialEntropicValueatRiskView` for an upper bound or a relative view."))
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
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                           alg::SequentialEntropicValueatRiskView,
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

`ep_add_grid_tail_view!` is the scaffold shared by [`GridEntropicValueatRiskView`](@ref) and [`GridRelativisticValueatRiskView`](@ref). Both build a grid of points, drop the points whose row is not usable, and add one linear row per kept point. They differ in what a point is and in how its row is built, and both reach the scaffold through `row`.

A point whose row is not finite is not a grid point, because a non-finite coefficient reaches the solver as `NaN * x[j]`. The caller's `msg` names the setting that put the whole grid there.

The upper-bound half also drops a point whose bound is at or below zero. Its coefficients are positive and the posterior sums to one, so its row holds at no posterior. The lower-bound half keeps such a point: its row holds at every posterior, so it costs nothing and changes no answer.

The rows of the lower-bound half keep the norm scale of [`add_ep_constraint!`](@ref). A row whose bound is small sits at the small end of the grid, and there the row is slack by many orders of magnitude at a posterior that meets a lower bound, because that posterior moves mass toward the largest loss, where every row takes its largest coefficient. Division by the bound would put coefficients of the order of the reciprocal of that bound into the dual that [`OptimEntropyPooling`](@ref) solves.

# Algorithm

 1. Build the row of every point with `row`, and keep the points whose coefficients and whose right-hand side are all finite.
 2. Raise with `msg` where no point is kept.
 3. Where the view carries a lower-bound half, add the row of each kept point to `epc` under `:ineq` with [`add_ep_constraint!`](@ref), negated so the row reads as the `<=` sense that key states.
 4. Where the view carries an upper-bound half, keep only the points whose right-hand side is positive, and raise with `msg` where none is kept.
 5. Return the kept points, which the caller carries into the tail view constraint of the upper-bound half.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `grid`: Points of the grid.
  - `op`: Comparison operator of the view.
  - `row`: Function taking one point to the coefficients and the right-hand side of its row.
  - `msg`: Function of no arguments giving the message of the error raised where no point is kept. It is called only where the grid keeps no point.

# Validation

  - At least one point of the grid has a finite row, and, where the view carries an upper-bound half, at least one such row has a positive bound. A grid that keeps no point raises an `ArgumentError` carrying `msg()`.

# Returns

  - `grid::AbstractVector`: Points of the grid whose row is finite and, where the view carries an upper-bound half, whose bound is positive.

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
                       alg::AbstractRelativisticValueatRiskViewFormulation,
                       x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number,
                       kappa::Number, op::Symbol, rhs::Number, w::VecNum, zstar::VecNum,
                       pv::Number, eqn::AbstractString; args::Tuple = (),
                       kwargs::NamedTuple = (;),
                       bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Lower one relativistic value-at-risk view into the constraints its formulation needs.

[`ConicRelativisticValueatRiskView`](@ref) and [`SequentialRelativisticValueatRiskView`](@ref) produce one tail view constraint each. [`GridRelativisticValueatRiskView`](@ref) produces linear rows on the posterior probabilities for the lower-bound half of the view, and a tail view constraint for the upper-bound half, so an equality view produces both.

# Algorithm

 1. [`ConicRelativisticValueatRiskView`](@ref) checks the three preconditions below, then appends one [`ConicRelativisticValueatRiskViewConstraint`](@ref) carrying `x`, `coef`, `alpha`, `kappa` and `rhs`.
 2. [`GridRelativisticValueatRiskView`](@ref) checks that the view names one asset, and normalises `w` to sum to one, giving `wi`.
 3. It builds the grid `t`, `z` of primal points with [`ep_rlvar_grid`](@ref).
 4. It keeps the points whose row is finite, giving `keep`, and raises where `keep` is empty. For the upper-bound half it keeps only the points whose bound is positive, and raises where none is.
 5. For the lower-bound half of the view, it builds the row of each kept point with [`ep_rlvar_grid_row`](@ref), and adds it to `epc` under `:ineq` with [`add_ep_constraint!`](@ref), negated so the row reads as the `<=` sense that key states.
 6. For the upper-bound half of the view, it appends one [`GridRelativisticValueatRiskViewConstraint`](@ref) carrying `x`, the kept grid, `alpha`, `kappa`, `rhs` and the big-M multiplier `M`.
 7. [`SequentialRelativisticValueatRiskView`](@ref) orients the view and splits its assets with [`ep_sequential_sides`](@ref), builds a [`SequentialRelativisticValueatRiskViewConstraint`](@ref) with an empty surrogate row, reads its first row from the prior `w` with [`ep_sequential_start`](@ref), and appends it.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view. A grid formulation is also where the number of steps and the tolerance of the anchor live, and a sequential one where the number of re-solves and their tolerance live.
  - `x`: Per asset the view names, its loss series.
  - `coef`: Per asset, the coefficient the view gives its RLVaR.
  - `alpha`: Significance level of the view.
  - `kappa`: Deformation parameter of the view.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `w`: Prior probability weights. They start the search for the point the grid is centred on, and they pin the shift of each grid point where that search does not converge. They are also where the first surrogate row of a sequential view is read.
  - `zstar`: Per asset, the dual variable that attains its prior RLVaR.
  - `pv`: Prior value of the view's left hand side. With `rhs` it fixes the translation a grid centred on the prior carries.
  - `eqn`: Equation of the view, used in the error messages.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: Spans of the searches, forwarded to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Validation

  - [`ConicRelativisticValueatRiskView`](@ref) needs coefficients of one sign, an operator other than `<=`, and, for an equality, a target at or above the prior value of the left hand side.
  - [`GridRelativisticValueatRiskView`](@ref) needs one asset, and at least one grid point whose row is finite, and whose bound is positive where the view carries an upper-bound half. [`ep_rlvar_tail`](@ref) overflows at a dual variable near zero, which is where the grid sits when `kappa` approaches one. The bound `rhs - t - z * ln_kappa(1 / (alpha * T))` of a point is at or below zero where the target lies below what the point can reach. Those points are dropped, and a grid that keeps none of them raises.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)
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
              ArgumentError("View `$(eqn)` carries coefficients of both signs. `ConicRelativisticValueatRiskView` writes a positive combination of RLVaRs, whose lower level set is convex; a relative view needs `SequentialRelativisticValueatRiskView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `ConicRelativisticValueatRiskView` bounds the RLVaR from below only; use `GridRelativisticValueatRiskView` for a single asset, or `SequentialRelativisticValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior RLVaR $(pv). `ConicRelativisticValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `GridRelativisticValueatRiskView` for a single asset, or `SequentialRelativisticValueatRiskView`."))
    push!(tvs, ConicRelativisticValueatRiskViewConstraint(x, coef, alpha, kappa, rhs))
    return nothing
end
function ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector,
                            alg::GridRelativisticValueatRiskView,
                            x::AbstractVector{<:VecNum}, coef::VecNum, alpha::Number,
                            kappa::Number, op::Symbol, rhs::Number, w::VecNum,
                            zstar::VecNum, pv::Number, eqn::AbstractString;
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    @argcheck(isone(length(x)),
              ArgumentError("View `$(eqn)` names $(length(x)) assets. `GridRelativisticValueatRiskView` writes the RLVaR of a single asset; use `ConicRelativisticValueatRiskView` for a lower bound on a positive combination, and `SequentialRelativisticValueatRiskView` for an upper bound or a relative view."))
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
function ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector,
                            alg::SequentialRelativisticValueatRiskView,
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

Give the trailing arguments a `prior(...)` reference of this tail view resolves under.

`ep_tail_view_prior_args` is one of the two kernels [`ep_tail_views!`](@ref) takes a measure from. It names the statistic [`get_pr_value`](@ref) reads, and it carries the level, the prior probabilities, and whatever settings the search for that statistic needs. [`replace_prior_views`](@ref) forwards the tuple unchanged, so a measure is added by adding a method here rather than by copying the lowering.

# Arguments

  - `tail_views`: Tail view group whose settings the tuple carries.
  - `w`: Prior probability weights.

# Returns

  - `args::Tuple`: Trailing arguments of [`replace_prior_views`](@ref), starting with the tag of the statistic.

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

Normalise the coefficients of a tail view, read the loss series it is stated on, and say whether its coefficients carry both signs.

The steps below are shared by every tail view. A view of one asset is divided by its coefficient, so its target is the measure itself. A view of several assets whose coefficients share one sign is multiplied by that sign, so every coefficient is positive and the view is a positive combination of measures, whose lower level set is convex. A view whose coefficients carry both signs is a relative view, and is left as stated.

# Algorithm

 1. Read `mixed`, whether the coefficients carry both signs.
 2. Where the view names one asset, divide it by that asset's coefficient with [`ep_normalise_view_term`](@ref), which flips the operator where the coefficient is negative, and set the coefficient to one.
 3. Where it names several assets and `mixed` is false, multiply both sides by the sign of the first coefficient through the same function, so every coefficient is positive and the operator flips where the sign is negative.
 4. Read the loss series of each asset, `x`, as its negated returns column.
 5. Reject a target no reweighting of the sample under the support of `w` reaches with [`ep_assert_reachable_view`](@ref).

# Arguments

  - `terms`: Resolved terms of the view, as [`ep_view_terms`](@ref) returns them.
  - `X`: Matrix of asset returns.
  - `w`: Prior probability weights.
  - `eqn`: Equation of the view, used in the error messages.
  - `name`: Name of the risk measure, used in the error messages.

# Returns

  - `x::AbstractVector{<:VecNum}`: Per asset the view names, its loss series.
  - `coef::VecNum`: Per asset, its normalised coefficient.
  - `op::Symbol`: Operator of the normalised view.
  - `rhs::Number`: Target of the normalised view.
  - `mixed::Bool`: Whether the coefficients carry both signs.

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

Lower one resolved tail view into the constraints its measure and its formulation need.

`ep_add_tail_view!` is the second of the two kernels [`ep_tail_views!`](@ref) takes a measure from. [`ep_tail_views!`](@ref) parses the group, expands its groups, and resolves its prior references and its terms; this verb carries everything past that point, which is everything the measures do not share. A measure is added by adding a method here and to [`ep_tail_view_prior_args`](@ref).

Every measure admits a view over several assets. Normalise it with [`ep_normalise_tail_view`](@ref), which also says whether the coefficients carry both signs, and read `pv`, the prior value of the left hand side, as the coefficient-weighted sum of the per-asset measures under `w`. Then pick the formulation and lower the view:

 1. A [`ConditionalValueatRiskView`](@ref) reads each asset's prior CVaR through [`ConditionalValueatRisk`](@ref), picks the formulation with [`ep_cvar_formulation`](@ref), and appends with [`ep_add_cvar_view!`](@ref).
 2. An [`EntropicValueatRiskView`](@ref) reads each asset's prior EVaR and the dual variable that attains it with [`ep_evar`](@ref), picks the formulation with [`ep_evar_formulation`](@ref), and appends with [`ep_add_evar_view!`](@ref).
 3. A [`RelativisticValueatRiskView`](@ref) reads each asset's prior RLVaR and the primal pair that attains it with [`ep_rlvar`](@ref), picks the formulation with [`ep_rlvar_formulation`](@ref), and appends with [`ep_add_rlvar_view!`](@ref).

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `tail_views`: Tail view group the view belongs to. It carries the level, the formulation setting, and the settings of the searches.
  - `alg`: Formulation of this view, as [`ep_view_formulations`](@ref) spread it.
  - `X`: Matrix of asset returns.
  - `terms`: Resolved terms of the view, as [`ep_view_terms`](@ref) returns them.
  - `eqn`: Equation of the view, used in the error messages.
  - `w`: Prior probability weights.

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

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
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

No-op pass-through for tail view constraints when none are specified.

# Arguments

  - `tail_views::Nothing`: Indicates that no tail view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

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

Every [`AbstractEntropyPoolingTailViewEstimator`](@ref) in the vector is lowered in turn, so the groups accumulate into the same constraint set and one entropy pooling solve answers all of them.

# Algorithm

 1. Lower each group of `tail_views` in turn, forwarding `args...` and `kwargs...` to each call.
 2. Return `nothing`. Each call has already written its constraints into `epc` and `tvs`.

# Arguments

  - `tail_views`: Groups of tail views.
  - `args...`: Additional positional arguments forwarded to [`ep_tail_views!`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`ep_tail_views!`](@ref).

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
                   w::VecNum; strict::Bool = false)

Parse a group of tail views and lower them into entropy pooling constraints.

`ep_tail_views!` is the one lowering of the tail view family. It parses the view equations of a [`LinearConstraintEstimator`](@ref), replaces prior references with their values, resolves the asset names against the universe, picks a formulation for each view, and appends the constraints that formulation needs. Unlike the recursive algorithm of [`MeucciEntropyPoolingPrior`](@ref), nothing is solved here: the views become part of the one entropy pooling problem [`entropy_pooling`](@ref) solves.

It accepts `==`, `>=` and `<=`. A group name expands to its members, each carrying the coefficient the group carried, so a view on a group constrains the *sum* of the members' risk measures and not their average. A view whose coefficients share one sign is a positive combination of measures, and its lower-bound form is convex. A view whose coefficients carry both signs is a relative view, and is not.

The two kernels below carry everything that differs between the conditional, the entropic and the relativistic measure, so a fourth measure supplies two methods rather than a fourth copy of this verb:

  - [`ep_tail_view_prior_args`](@ref) names the statistic a `prior(...)` reference resolves to, and the settings its search takes.
  - [`ep_add_tail_view!`](@ref) checks the shape of the view, reads its prior value, picks its formulation, and appends its constraints.

# Algorithm

 1. Parse the view equations of `tail_views.views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans, giving one term per member.
 3. Replace every `prior(...)` reference by the prior value of the measure, through [`replace_prior_views`](@ref) under the tuple [`ep_tail_view_prior_args`](@ref) gives.
 4. Spread the formulation setting over the views with [`ep_view_formulations`](@ref), giving `algs`.
 5. For each view in turn, resolve its terms with [`ep_view_terms`](@ref), and drop the view where no name of it is placed in the universe.
 6. Lower the view with [`ep_add_tail_view!`](@ref).

# Arguments

  - `tail_views`: Tail view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `w`: Prior probability weights.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

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

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
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
