"""
    ep_evar(x::VecNum, w::VecNum, alpha::Number)

Compute the sample entropic value-at-risk of a loss series and the dual variable that attains it.

`ep_evar` minimises the scalar convex objective of the sample EVaR formula with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl)'s Brent method. It is used by the entropy pooling view machinery, which needs both the value (to compare a view against its prior) and the minimiser (to centre the grid of [`GridEntropicValueatRiskView`](@ref)).

# Mathematical definition

```math
\\mathrm{EVaR}_{\\alpha}(X) = \\min_{z > 0} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
```

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Observation probabilities. Normalised to sum to one.
  - `alpha`: Significance level.

# Returns

  - `res::@NamedTuple{evar::Number, z::Number}`: The value and the dual variable that attains it.

# Details

  - The objective is evaluated through `LogExpFunctions.logsumexp`, so a small `z` does not overflow.
  - The search is bracketed by `(maximum(x) - dot(w, x)) / log(inv(alpha))`, above which the objective already exceeds `maximum(x)`, which bounds the EVaR from above.

# Related

  - [`GridEntropicValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropicValueatRisk`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function ep_evar(x::VecNum, w::VecNum, alpha::Number)
    lw = log.(w)
    lw .-= LogExpFunctions.logsumexp(lw)
    ila = -log(alpha)
    f = function (z)
        return z * (LogExpFunctions.logsumexp(lw .+ x ./ z) + ila)
    end
    hi = (maximum(x) - LinearAlgebra.dot(exp.(lw), x)) / ila
    hi = ifelse(hi > zero(hi), hi, eps(float(one(hi))))
    lo = hi * sqrt(eps(float(one(hi))))
    res = Optim.optimize(f, lo, hi, Optim.Brent())
    return (; evar = Optim.minimum(res), z = Optim.minimizer(res))
end
"""
    ep_evar_grid_row(x::VecNum, ebar::Number, z::Number)

Build one scaled row of the grid formulation of an entropic value-at-risk view.

`ep_evar_grid_row` returns the coefficients of `exp((x - ebar) / z)` divided by their largest entry, together with the reciprocal of that entry, which the right hand side must be multiplied by. Scaling the row keeps the coefficients in `(0, 1]` however small `z` is, which is what lets the big-M constant of [`GridEntropicValueatRiskView`](@ref) be a plain number rather than a function of the data.

# Arguments

  - `x`: Loss series (`-returns`).
  - `ebar`: Target entropic value-at-risk.
  - `z`: Grid point of the entropic value-at-risk dual variable.

# Returns

  - `c::VecNum`: Scaled coefficients.
  - `isc::Number`: Scaling factor to apply to the right hand side.

# Related

  - [`GridEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskViewConstraint`](@ref)
  - [`ep_evar_views!`](@ref)
"""
function ep_evar_grid_row(x::VecNum, ebar::Number, z::Number)
    c = exp.((x .- ebar) ./ z)
    sc = maximum(c)
    return c ./ sc, inv(sc)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series, the significance level and the target of a linear conditional value-at-risk view.

The view parser produces one of these per view that takes the linear formulation. [`add_ep_tail_view!`](@ref) then writes the dual representation of CVaR into the model from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearConditionalValueatRiskViewConstraint(x, alpha, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct LinearConditionalValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the ordered tail window of every asset an integer conditional value-at-risk view names.

Each entry of `ord`, `x` pairs one asset named by the view with its coefficient in `coef`, so an absolute view carries one entry and a relative view carries two. The window is sorted ascending, so the largest loss is last and the tail the binary vector marks is a suffix of it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IntegerConditionalValueatRiskViewConstraint(ord, x, coef, alpha, op, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct IntegerConditionalValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_ord])
    """
    ord
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_op])
    """
    op
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series, the significance level and the target of a conic entropic value-at-risk view.

The view parser produces one of these per view that takes the conic formulation. [`add_ep_tail_view!`](@ref) then writes the relative entropy cone that is the dual representation of EVaR from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConicEntropicValueatRiskViewConstraint(x, alpha, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct ConicEntropicValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the grid of dual variables that an upper-bound or equality entropic value-at-risk view selects one point of.

A lower-bound grid view is a set of rows on the posterior probabilities alone, so it goes into the constraint dictionary and never reaches this carrier. An equality view emits both: the rows go into the dictionary and the selector block comes here.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GridEntropicValueatRiskViewConstraint(x, z, alpha, rhs, M)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct GridEntropicValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_zgrid])
    """
    z
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:bigM])
    """
    M
end
"""
    add_ep_tail_view!(model::JuMP.Model, pw, tv::AbstractEntropyPoolingTailView,
                      sc1::Number)

Add the variables and constraints of one tail view to an entropy pooling JuMP model.

`add_ep_tail_view!` is the one seam through which a conditional or entropic value-at-risk view reaches the model. Each formulation has its own method, dispatched on the constraint carrier the view parser produced.

# Arguments

  - `model`: Entropy pooling JuMP model.
  - `pw`: Vector of posterior probability variables.
  - `tv`: Tail view constraint.
  - `sc1`: Constraint scaling factor.

# Returns

  - `nothing`: The function mutates `model` in-place.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`entropy_pooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::LinearConditionalValueatRiskViewConstraint, sc1::Number)
    (; x, alpha, rhs) = tv
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0)
    JuMP.@constraints(model, begin
                          [j = 1:T], sc1 * (nu[j] - pw[j] / alpha) <= 0
                          sc1 * (sum(nu) - one(alpha)) == 0
                          sc1 * (rhs - LinearAlgebra.dot(nu, x)) <= 0
                      end)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::ConicEntropicValueatRiskViewConstraint, sc1::Number)
    (; x, alpha, rhs) = tv
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0, upper_bound = 1)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(nu) - one(alpha)) == 0
                          sc1 * (rhs - LinearAlgebra.dot(nu, x)) <= 0
                          [sc1 * log(inv(alpha)); sc1 * pw; sc1 * nu] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::IntegerConditionalValueatRiskViewConstraint, sc1::Number)
    (; ord, x, coef, alpha, op, rhs) = tv
    expr = JuMP.AffExpr()
    for (ordi, xi, ci) in zip(ord, x, coef)
        sb = length(ordi)
        y = JuMP.@variable(model, [1:sb], binary = true)
        q = JuMP.@variable(model, [1:sb], lower_bound = 0)
        JuMP.@constraints(model,
                          begin
                              [j = 1:sb], sc1 * (q[j] - y[j]) <= 0
                              [j = 1:sb], sc1 * (q[j] - pw[ordi[j]]) <= 0
                              [j = 1:sb],
                              sc1 * (pw[ordi[j]] - (one(alpha) - y[j]) - q[j]) <= 0
                              [j = 1:(sb - 1)], sc1 * (y[j] - y[j + 1]) <= 0
                              sc1 * (sum(q) - alpha) == 0
                          end)
        JuMP.add_to_expression!(expr, ci / alpha, LinearAlgebra.dot(q, xi))
    end
    if op == :eq
        JuMP.@constraint(model, sc1 * (expr - rhs) == 0)
    elseif op == :geq
        JuMP.@constraint(model, sc1 * (rhs - expr) <= 0)
    else
        JuMP.@constraint(model, sc1 * (expr - rhs) <= 0)
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
        JuMP.@constraint(model,
                         sc1 *
                         (LinearAlgebra.dot(c, pw) - alpha * isc - M * (one(alpha) - y[k])) <=
                         0)
    end
    return nothing
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:evar}, alpha::Number)

Extract the Entropic Value-at-Risk (EVaR) for asset `i` from a prior result.

`get_pr_value` computes the EVaR at confidence level `alpha` for the asset indexed by `i` from the prior result object `pr`, by minimising the scalar objective of the sample EVaR formula with [`ep_evar`](@ref).

# Arguments

  - `pr`: Prior result containing asset return information.
  - `i`: Index of the asset.
  - `::Val{:evar}`: Dispatch tag for EVaR extraction.
  - `alpha`: Confidence level (e.g. `0.05` for 5% EVaR).

# Returns

  - `evar::Number`: Entropic Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`ep_evar`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:evar}, alpha::Number)
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    T = size(pr.X, 1)
    iT = inv(T)
    return ep_evar(-view(pr.X, :, i), range(iT, iT; length = T), alpha).evar
end
"""
    ep_view_terms(res::ParsingResult, sets::UniverseSets, X::MatNum; strict::Bool = false)

Resolve one parsed tail view into the assets it names, their coefficients, its operator and its target.

`ep_view_terms` routes a [`ParsingResult`](@ref) through [`get_linear_constraints`](@ref), which resolves the variable names against the universe and reports the ones it cannot place, then undoes the sign flip that entry point applies to a `>=` equation so the operator survives. The linear view machinery never needs the operator back, because a row of `A x <= b` carries it; a tail view does, because each operator picks a different formulation.

# Arguments

  - `res`: Parsed view constraint.
  - `sets`: Asset set mapping asset names to indices.
  - `X`: Asset returns matrix, read for its element type.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: If no name in the view could be placed in the universe.
  - `terms::@NamedTuple{idx::VecInt, coef::VecNum, op::Symbol, rhs::Number}`: The assets the view names, their coefficients, its operator (`:eq`, `:geq` or `:leq`) and its target.

# Related

  - [`ep_cvar_views!`](@ref)
  - [`ep_evar_views!`](@ref)
  - [`get_linear_constraints`](@ref)
  - [`comparison_sign_ineq_flag`](@ref)
"""
function ep_view_terms(res::ParsingResult, sets::UniverseSets, X::MatNum;
                       strict::Bool = false)
    lc = get_linear_constraints([res], sets; datatype = eltype(X), strict = strict)
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

# Arguments

  - `coef`: Coefficient the view gives the asset's risk measure.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.

# Returns

  - `op::Symbol`: Operator of the normalised view.
  - `rhs::Number`: Target of the normalised view.

# Related

  - [`ep_view_terms`](@ref)
  - [`ep_cvar_views!`](@ref)
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
  - [`ep_cvar_views!`](@ref)
  - [`ep_evar_views!`](@ref)
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

# Arguments

  - `sbar`: Setting held by [`IntegerConditionalValueatRiskView`](@ref). An `Integer` is a count, a fraction in `(0, 1)` is a fraction of `T`, and `nothing` applies the rule of thumb of [EPTail](@cite).
  - `T`: Number of observations.
  - `alpha`: Significance level of the view.
  - `w`: Prior probability weights.
  - `ord`: Indices of the losses in ascending order, so the largest loss is last.

# Returns

  - `sbar::Int`: Number of largest losses, in `1:T`.

# Details

  - The rule of thumb takes twice the position at which the prior probabilities first reach `alpha`, and never less than `ceil(2 * alpha * T)`. A view above the prior CVaR moves mass into the tail and needs about that position; a view below it moves mass out and needs more.

# Related

  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_cvar_views!`](@ref)

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
    ep_assert_reachable_view(op::Symbol, rhs::Number, x::VecNum, eqn::AbstractString,
                             name::AbstractString)

Reject a tail view no reweighting of the sample can reach.

A tail risk measure of a reweighted sample lies between the smallest and the largest loss the sample holds, so a view outside that band is infeasible however the probabilities move.

# Arguments

  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `x`: Loss series of the asset the view names.
  - `eqn`: Equation of the view, used in the error message.
  - `name`: Name of the view family, used in the error message.

# Validation

  - If `op` is `:geq` or `:eq`, `rhs < maximum(x)`.
  - If `op` is `:leq` or `:eq`, `rhs > minimum(x)`.

# Returns

  - `nothing`.

# Related

  - [`ep_cvar_views!`](@ref)
  - [`ep_evar_views!`](@ref)
"""
function ep_assert_reachable_view(op::Symbol, rhs::Number, x::VecNum, eqn::AbstractString,
                                  name::AbstractString)
    if op == :geq || op == :eq
        @argcheck(rhs < maximum(x),
                  DomainError(rhs,
                              "View `$(eqn)` is too extreme: the largest $(name) any reweighting of this sample reaches is its worst realisation, $(maximum(x)). Lower the view, raise alpha, or use a prior with fatter tails."))
    end
    if op == :leq || op == :eq
        @argcheck(rhs > minimum(x),
                  DomainError(rhs,
                              "View `$(eqn)` is too extreme: the smallest $(name) any reweighting of this sample reaches is its best realisation, $(minimum(x)). Raise the view, lower alpha, or use a prior with a thinner tail."))
    end
    return nothing
end
"""
    ep_cvar_formulation(alg::Option{<:AbstractConditionalValueatRiskViewFormulation}, single::Bool,
                        op::Symbol, rhs::Number, pv::Number)

Pick the formulation of one conditional value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`LinearConditionalValueatRiskView`](@ref) wherever it expresses the view exactly, and [`IntegerConditionalValueatRiskView`](@ref) otherwise, which is every view the linear formulation cannot express: a relative view, an upper bound, and an equality below the prior CVaR.

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `single`: Whether the view names one asset.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior value of the view's left hand side.

# Returns

  - `alg::AbstractConditionalValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_cvar_views!`](@ref)
"""
function ep_cvar_formulation(alg::AbstractConditionalValueatRiskViewFormulation, args...)
    return alg
end
function ep_cvar_formulation(::Nothing, single::Bool, op::Symbol, rhs::Number, pv::Number)
    return if single && (op == :geq || op == :eq && rhs >= pv)
        LinearConditionalValueatRiskView()
    else
        IntegerConditionalValueatRiskView()
    end
end
"""
    ep_evar_formulation(alg::Option{<:AbstractEntropicValueatRiskViewFormulation}, op::Symbol,
                        rhs::Number, pv::Number)

Pick the formulation of one entropic value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`ConicEntropicValueatRiskView`](@ref) wherever it expresses the view exactly, and [`GridEntropicValueatRiskView`](@ref) otherwise, which is an upper bound and an equality below the prior EVaR.

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior EVaR of the asset the view names.

# Returns

  - `alg::AbstractEntropicValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`ep_evar_views!`](@ref)
"""
function ep_evar_formulation(alg::AbstractEntropicValueatRiskViewFormulation, args...)
    return alg
end
function ep_evar_formulation(::Nothing, op::Symbol, rhs::Number, pv::Number)
    return if op == :geq || op == :eq && rhs >= pv
        ConicEntropicValueatRiskView()
    else
        GridEntropicValueatRiskView()
    end
end
"""
    ep_add_cvar_view!(tvs::AbstractVector, alg::AbstractConditionalValueatRiskViewFormulation, X::MatNum,
                      idx::VecInt, coef::VecNum, op::Symbol, rhs::Number, alpha::Number,
                      w::VecNum, pv::Number, eqn::AbstractString)

Lower one conditional value-at-risk view into the tail view constraint its formulation needs.

# Arguments

  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view.
  - `X`: Asset returns matrix.
  - `idx`: Indices of the assets the view names.
  - `coef`: Coefficient the view gives each asset's CVaR.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `alpha`: Significance level of the view.
  - `w`: Prior probability weights.
  - `pv`: Prior value of the view's left hand side.
  - `eqn`: Equation of the view, used in the error messages.

# Validation

  - [`LinearConditionalValueatRiskView`](@ref) needs one asset, an operator other than `<=`, and, for an equality, a target at or above the prior CVaR.

# Returns

  - `nothing`: The function mutates `tvs` in-place.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_cvar_views!`](@ref)
"""
function ep_add_cvar_view!(tvs::AbstractVector, ::LinearConditionalValueatRiskView,
                           X::MatNum, idx::VecInt, coef::VecNum, op::Symbol, rhs::Number,
                           alpha::Number, w::VecNum, pv::Number, eqn::AbstractString)
    @argcheck(isone(length(idx)),
              ArgumentError("View `$(eqn)` names $(length(idx)) assets. `LinearConditionalValueatRiskView` writes the CVaR of a single asset; a relative view needs `IntegerConditionalValueatRiskView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `LinearConditionalValueatRiskView` bounds the CVaR from below only; use `IntegerConditionalValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior CVaR $(pv). `LinearConditionalValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `IntegerConditionalValueatRiskView`."))
    push!(tvs, LinearConditionalValueatRiskViewConstraint(-X[:, idx[1]], alpha, rhs))
    return nothing
end
function ep_add_cvar_view!(tvs::AbstractVector, alg::IntegerConditionalValueatRiskView,
                           X::MatNum, idx::VecInt, coef::VecNum, op::Symbol, rhs::Number,
                           alpha::Number, w::VecNum, pv::Number, eqn::AbstractString)
    T = size(X, 1)
    N = length(idx)
    ord = Vector{Vector{Int}}(undef, N)
    x = Vector{Vector{eltype(X)}}(undef, N)
    for (k, j) in pairs(idx)
        xj = -X[:, j]
        o = sortperm(xj)
        sb = ep_sbar(alg.sbar, T, alpha, w, o)
        ord[k] = o[(T - sb + 1):T]
        x[k] = xj[ord[k]]
    end
    push!(tvs, IntegerConditionalValueatRiskViewConstraint(ord, x, coef, alpha, op, rhs))
    return nothing
end
"""
    ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                      alg::AbstractEntropicValueatRiskViewFormulation, x::VecNum, alpha::Number,
                      op::Symbol, rhs::Number, zstar::Number, pv::Number,
                      eqn::AbstractString)

Lower one entropic value-at-risk view into the constraints its formulation needs.

[`ConicEntropicValueatRiskView`](@ref) produces one tail view constraint. [`GridEntropicValueatRiskView`](@ref) produces linear rows on the posterior probabilities for the lower-bound half of the view, and a tail view constraint for the upper-bound half, so an equality view produces both.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view.
  - `x`: Loss series of the asset the view names.
  - `alpha`: Significance level of the view.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `zstar`: Dual variable that attains the prior EVaR of the asset.
  - `pv`: Prior EVaR of the asset.
  - `eqn`: Equation of the view, used in the error messages.

# Validation

  - [`ConicEntropicValueatRiskView`](@ref) needs an operator other than `<=`, and, for an equality, a target at or above the prior EVaR.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`ep_evar_views!`](@ref)
"""
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                           ::ConicEntropicValueatRiskView, x::VecNum, alpha::Number,
                           op::Symbol, rhs::Number, zstar::Number, pv::Number,
                           eqn::AbstractString)
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `ConicEntropicValueatRiskView` bounds the EVaR from below only; use `GridEntropicValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior EVaR $(pv). `ConicEntropicValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `GridEntropicValueatRiskView`."))
    push!(tvs, ConicEntropicValueatRiskViewConstraint(x, alpha, rhs))
    return nothing
end
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                           alg::GridEntropicValueatRiskView, x::VecNum, alpha::Number,
                           op::Symbol, rhs::Number, zstar::Number, pv::Number,
                           eqn::AbstractString)
    (; pct, K, M) = alg
    z = collect(range(zstar * (one(pct) - pct), zstar * (one(pct) + pct); length = K))
    if op == :geq || op == :eq
        for zk in z
            c, isc = ep_evar_grid_row(x, rhs, zk)
            add_ep_constraint!(epc, reshape(-c, 1, :), [-alpha * isc], :ineq)
        end
    end
    if op == :leq || op == :eq
        push!(tvs, GridEntropicValueatRiskViewConstraint(x, z, alpha, rhs, M))
    end
    return nothing
end
"""
    ep_cvar_views!(cvar_views::Nothing, args...; kwargs...)

No-op pass-through for conditional value at risk (CVaR) view constraints when none are specified.

# Arguments

  - `cvar_views::Nothing`: Indicates that no CVaR view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`ep_cvar_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_cvar_views!(cvar_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_cvar_views!(cvar_views::LinearConstraintEstimator, epc::AbstractDict,
                   tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                   alpha::Number, w::VecNum, alg; strict::Bool = false)

Parse conditional value-at-risk views and lower them into entropy pooling constraints.

`ep_cvar_views!` parses CVaR view equations from a [`LinearConstraintEstimator`](@ref), replaces prior references with their values, resolves the asset names against the universe, picks a formulation for each view, and appends the constraints that formulation needs. Unlike the recursive algorithm of [`MeucciEntropyPoolingPrior`](@ref), nothing is solved here: the views become part of the one entropy pooling problem [`entropy_pooling`](@ref) solves.

# Arguments

  - `cvar_views`: CVaR view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `alpha`: Confidence level for CVaR.
  - `w`: Prior probability weights.
  - `alg`: Formulation setting, spread over the views by [`ep_view_formulations`](@ref).
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Details

  - Accepts `==`, `>=` and `<=`.
  - A group name expands to its members, each carrying the coefficient the group carried, so a view on a group constrains the *sum* of the members' CVaRs, not their average. A group of more than one member is therefore a relative view.
  - A view naming one asset is normalised so its coefficient is one, which flips the operator when the coefficient is negative.
  - A view naming several assets is a relative view, and only [`IntegerConditionalValueatRiskView`](@ref) expresses it.

# Related

  - [`ep_cvar_formulation`](@ref)
  - [`ep_add_cvar_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_cvar_views!(cvar_views::ConditionalValueatRiskView, epc::AbstractDict,
                        tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                        w::VecNum; strict::Bool = false)
    return ep_cvar_views!(cvar_views.views, epc, tvs, pr, sets, cvar_views.alpha, w,
                          cvar_views.alg; strict = strict)
end
"""
    ep_cvar_views!(cvar_views::AbstractVector{<:ConditionalValueatRiskView}, args...; kwargs...)

Lower each group of conditional value-at-risk views under its own settings.

Every [`ConditionalValueatRiskView`](@ref) in the vector is lowered in turn, so the groups accumulate into the same constraint set and one entropy pooling solve answers all of them.

# Arguments

  - `cvar_views`: Groups of CVaR views.
  - `args...`: Additional positional arguments forwarded to [`ep_cvar_views!`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`ep_cvar_views!`](@ref).

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_cvar_views!(cvar_views::AbstractVector{<:ConditionalValueatRiskView}, args...;
                        kwargs...)
    for cvar_view in cvar_views
        ep_cvar_views!(cvar_view, args...; kwargs...)
    end
    return nothing
end
function ep_cvar_views!(cvar_views::LinearConstraintEstimator, epc::AbstractDict,
                        tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                        alpha::Number, w::VecNum, alg; strict::Bool = false)
    X = pr.X
    views = parse_equation(cvar_views.val; ops1 = ("==", ">=", "<="),
                           ops2 = (:call, :(==), :(>=), :(<=)), datatype = eltype(X))
    views = replace_group_by_assets(views, sets, false, true, false)
    views = replace_prior_views(views, pr, sets, :cvar, alpha; strict = strict)
    if !isa(views, AbstractVector)
        views = [views]
    end
    algs = ep_view_formulations(alg, length(views), :alg)
    rm = ConditionalValueatRisk(; alpha = alpha, w = StatsBase.pweights(w))
    for (res, algi) in zip(views, algs)
        terms = ep_view_terms(res, sets, X; strict = strict)
        if isnothing(terms)
            continue
        end
        (; idx, coef, op, rhs) = terms
        single = isone(length(idx))
        if single
            op, rhs = ep_normalise_view_term(coef[1], op, rhs)
            coef = [one(eltype(coef))]
            ep_assert_reachable_view(op, rhs, -X[:, idx[1]], res.eqn, "CVaR")
        end
        pv = sum(ci * rm(view(X, :, j)) for (j, ci) in zip(idx, coef))
        algi = ep_cvar_formulation(algi, single, op, rhs, pv)
        ep_add_cvar_view!(tvs, algi, X, idx, coef, op, rhs, alpha, w, pv, res.eqn)
    end
    return nothing
end
"""
    ep_evar_views!(evar_views::Nothing, args...; kwargs...)

No-op pass-through for entropic value at risk (EVaR) view constraints when none are specified.

# Arguments

  - `evar_views::Nothing`: Indicates that no EVaR view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`ep_evar_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_evar_views!(evar_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_evar_views!(evar_views::LinearConstraintEstimator, epc::AbstractDict,
                   tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                   alpha::Number, w::VecNum, alg; strict::Bool = false)

Parse entropic value-at-risk views and lower them into entropy pooling constraints.

`ep_evar_views!` parses EVaR view equations from a [`LinearConstraintEstimator`](@ref), replaces prior references with their values, resolves the asset names against the universe, picks a formulation for each view, and appends the constraints that formulation needs.

# Arguments

  - `evar_views`: EVaR view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `alpha`: Confidence level for EVaR.
  - `w`: Prior probability weights.
  - `alg`: Formulation setting, spread over the views by [`ep_view_formulations`](@ref).
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Details

  - Accepts `==`, `>=` and `<=`.
  - Each view names one asset. [EPTail](@cite) gives no formulation for a relative EVaR view.
  - A group name expands to its members, each carrying the coefficient the group carried, so only a group of one member names one asset.
  - The view is normalised so its coefficient is one, which flips the operator when the coefficient is negative.

# Related

  - [`ep_evar_formulation`](@ref)
  - [`ep_add_evar_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_evar_views!(evar_views::EntropicValueatRiskView, epc::AbstractDict,
                        tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                        w::VecNum; strict::Bool = false)
    return ep_evar_views!(evar_views.views, epc, tvs, pr, sets, evar_views.alpha, w,
                          evar_views.alg; strict = strict)
end
"""
    ep_evar_views!(evar_views::AbstractVector{<:EntropicValueatRiskView}, args...; kwargs...)

Lower each group of entropic value-at-risk views under its own settings.

Every [`EntropicValueatRiskView`](@ref) in the vector is lowered in turn, so the groups accumulate into the same constraint set and one entropy pooling solve answers all of them.

# Arguments

  - `evar_views`: Groups of EVaR views.
  - `args...`: Additional positional arguments forwarded to [`ep_evar_views!`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`ep_evar_views!`](@ref).

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`EntropicValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_evar_views!(evar_views::AbstractVector{<:EntropicValueatRiskView}, args...;
                        kwargs...)
    for evar_view in evar_views
        ep_evar_views!(evar_view, args...; kwargs...)
    end
    return nothing
end
function ep_evar_views!(evar_views::LinearConstraintEstimator, epc::AbstractDict,
                        tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                        alpha::Number, w::VecNum, alg; strict::Bool = false)
    X = pr.X
    views = parse_equation(evar_views.val; ops1 = ("==", ">=", "<="),
                           ops2 = (:call, :(==), :(>=), :(<=)), datatype = eltype(X))
    views = replace_group_by_assets(views, sets, false, true, false)
    views = replace_prior_views(views, pr, sets, :evar, alpha; strict = strict)
    if !isa(views, AbstractVector)
        views = [views]
    end
    algs = ep_view_formulations(alg, length(views), :alg)
    for (res, algi) in zip(views, algs)
        terms = ep_view_terms(res, sets, X; strict = strict)
        if isnothing(terms)
            continue
        end
        (; idx, coef, op, rhs) = terms
        @argcheck(isone(length(idx)),
                  ArgumentError("View `$(res.eqn)` names $(length(idx)) assets. An EVaR view names one asset: there is no formulation for a relative EVaR view."))
        op, rhs = ep_normalise_view_term(coef[1], op, rhs)
        x = -X[:, idx[1]]
        ep_assert_reachable_view(op, rhs, x, res.eqn, "EVaR")
        evr = ep_evar(x, w, alpha)
        pv, zstar = evr.evar, evr.z
        algi = ep_evar_formulation(algi, op, rhs, pv)
        ep_add_evar_view!(epc, tvs, algi, x, alpha, op, rhs, zstar, pv, res.eqn)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Reweights the observations of a prior so that its moments and its tails meet a set of views.

`EntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports views on the mean, the variance, the covariance, the correlation, the skewness and the kurtosis, views on the value at risk, and the conditional and entropic value at risk views of [EPTail](@cite).

The tail views are the difference with [`MeucciEntropyPoolingPrior`](@ref). There, a CVaR view is a target the recursive algorithm of Meucci et al. hunts by re-solving the whole entropy pooling problem for each candidate value at risk level, which supports equalities alone. Here each tail view is written as constraints of the single entropy pooling problem, so one solve answers every view, and the operators `==`, `>=` and `<=` are all available, along with relative CVaR views and views on the entropic value at risk.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropyPoolingPrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        mu_views::Option{<:LinearConstraintEstimator} = nothing,
        var_views::Option{<:VV_VecVV} = nothing,
        cvar_views::Option{<:CVV_VecCVV} = nothing,
        evar_views::Option{<:EVV_VecEVV} = nothing,
        sigma_views::Option{<:LinearConstraintEstimator} = nothing,
        sk_views::Option{<:LinearConstraintEstimator} = nothing,
        kt_views::Option{<:LinearConstraintEstimator} = nothing,
        cov_views::Option{<:LinearConstraintEstimator} = nothing,
        rho_views::Option{<:LinearConstraintEstimator} = nothing,
        sets::Option{<:UniverseSets} = nothing,
        opt::NonCVaREP = OptimEntropyPooling(),
        w::Option{<:StatsBase.ProbabilityWeights} = nothing,
        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling()
    ) -> EntropyPoolingPrior

Keywords correspond to the struct's fields.

## Validation

  - If any view constraint is not `nothing`, `sets` must not be `nothing`.
  - If `cvar_views` is not `nothing`, `opt` must be a [`JuMPEntropyPooling`](@ref).
  - If a view field is a vector, it must not be empty.
  - If `w` is not `nothing`, it must be non-empty and match the number of observations.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `pe`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Details

  - If `w` is not `nothing`, it is normalised to sum to 1; otherwise, uniform weights are used when `prior` is called.

# View comparison operators

The comparison operators accepted in each view's constraint strings depend on the moment being constrained. An unsupported operator raises a `ParseError` listing the operators allowed for that view.

  - `mu_views`, `sigma_views`, `sk_views`, `kt_views`, `cov_views`, `rho_views` accept `==`, `>=` and `<=`.
  - `var_views` (Value at Risk) accepts only `==` and `>=`.
  - `cvar_views` and `evar_views` accept `==`, `>=` and `<=`.

# Tail views

A tail view needs auxiliary variables, so it is expressed in the JuMP model rather than reduced to rows that multiply the posterior probabilities. Two consequences follow.

  - `opt` must be a [`JuMPEntropyPooling`](@ref) whenever `cvar_views` is set, and whenever an `evar_views` entry is anything other than a lower bound under [`GridEntropicValueatRiskView`](@ref), which is the one tail formulation that is linear in the posterior probabilities alone.
  - A view that needs a binary variable — every [`IntegerConditionalValueatRiskView`](@ref), and an upper bound or equality under [`GridEntropicValueatRiskView`](@ref) — needs a solver that handles mixed-integer exponential cone programs.

The `alg` field of a view group picks the formulation. A single formulation applies to every view in that group, a vector supplies one per view, and `nothing` lets each view take the cheapest formulation that expresses it exactly: [`LinearConditionalValueatRiskView`](@ref) and [`ConicEntropicValueatRiskView`](@ref) where they apply, [`IntegerConditionalValueatRiskView`](@ref) and [`GridEntropicValueatRiskView`](@ref) otherwise.

# Tail views at several significance levels

A significance level is part of the statistic, not a detail of the solve: the conditional value at risk at 1% and at 10% are different numbers on the same series. So the level lives on the view rather than on the estimator. `var_views`, `cvar_views` and `evar_views` each take one [`ValueatRiskView`](@ref), [`ConditionalValueatRiskView`](@ref) or [`EntropicValueatRiskView`](@ref), or a vector of them, and each group carries the `alpha` its equations are read under. A `prior(...)` reference inside a group is replaced by the prior value at *that* group's level.

A tail view group also carries `alg`, the formulation. For [`EntropicValueatRiskView`](@ref) that is where the grid of dual variables and the big-M constant live, so a [`GridEntropicValueatRiskView`](@ref) there gives one group its own `pct`, `K` and `M`. [`ValueatRiskView`](@ref) has no such field: a value at risk view is linear in the posterior probabilities, so there is no formulation to choose.

# Examples

```jldoctest
julia> EntropyPoolingPrior(;
                           sets = UniverseSets(; xkey = \"nx\",
                                               dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"])),
                           mu_views = LinearConstraintEstimator(;
                                                                val = [\"A == 0.03\",
                                                                       \"B + C == 0.04\"]))
EntropyPoolingPrior
           pe ┼ EmpiricalPrior
              │        ce ┼ PortfolioOptimisersCovariance
              │           │   ce ┼ Covariance
              │           │      │    me ┼ SimpleExpectedReturns
              │           │      │       │   w ┴ nothing
              │           │      │    ce ┼ GeneralCovariance
              │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
              │           │      │       │    w ┴ nothing
              │           │      │   alg ┼ FullMoment()
              │           │      │     w ┴ nothing
              │           │   mp ┼ MatrixProcessing
              │           │      │     pdm ┼ Posdef
              │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
              │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
              │           │      │      dn ┼ nothing
              │           │      │      dt ┼ nothing
              │           │      │     alg ┼ nothing
              │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
              │        me ┼ SimpleExpectedReturns
              │           │   w ┴ nothing
              │   horizon ┴ nothing
     mu_views ┼ LinearConstraintEstimator
              │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
              │   key ┴ nothing
    var_views ┼ nothing
   cvar_views ┼ nothing
   evar_views ┼ nothing
  sigma_views ┼ nothing
     sk_views ┼ nothing
     kt_views ┼ nothing
    cov_views ┼ nothing
    rho_views ┼ nothing
         sets ┼ UniverseSets
              │    xkey ┼ String: "nx"
              │   uxkey ┼ String: "ux"
              │    fkey ┼ String: "nf"
              │   ufkey ┼ String: "uf"
              │    zkey ┼ String: "nz"
              │    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
          opt ┼ OptimEntropyPooling
              │     args ┼ Tuple{}: ()
              │   kwargs ┼ @NamedTuple{}: NamedTuple()
              │      sc1 ┼ Int64: 1
              │      sc2 ┼ Float64: 1000.0
              │      alg ┼ ExpEntropyPooling()
              │      err ┴ nothing
            w ┼ nothing
          alg ┴ H1_EntropyPooling()
```

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)

# References

  - $(ref_dict[:meucci2008])
  - $(ref_dict[:vorobets2021])
  - $(ref_dict[:EPTail])
"""
@propagatable @concrete struct EntropyPoolingPrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:mu_views])
    """
    mu_views
    """
    $(field_dict[:var_views])
    """
    var_views
    """
    $(field_dict[:cvar_views])
    """
    cvar_views
    """
    $(field_dict[:evar_views])
    """
    evar_views
    """
    $(field_dict[:sigma_views])
    """
    sigma_views
    """
    $(field_dict[:sk_views])
    """
    sk_views
    """
    $(field_dict[:kt_views])
    """
    kt_views
    """
    $(field_dict[:cov_views])
    """
    cov_views
    """
    $(field_dict[:rho_views])
    """
    rho_views
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:opt_ep])
    """
    opt
    """
    $(field_dict[:ep_w])
    """
    @wprop w
    """
    $(field_dict[:epalg])
    """
    alg
    function EntropyPoolingPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mu_views::Option{<:LinearConstraintEstimator},
                                 var_views::Option{<:VV_VecVV},
                                 cvar_views::Option{<:CVV_VecCVV},
                                 evar_views::Option{<:EVV_VecEVV},
                                 sigma_views::Option{<:LinearConstraintEstimator},
                                 sk_views::Option{<:LinearConstraintEstimator},
                                 kt_views::Option{<:LinearConstraintEstimator},
                                 cov_views::Option{<:LinearConstraintEstimator},
                                 rho_views::Option{<:LinearConstraintEstimator},
                                 sets::Option{<:UniverseSets}, opt::NonCVaREP,
                                 w::Option{<:StatsBase.ProbabilityWeights},
                                 alg::AbstractEntropyPoolingAlgorithm)
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
            if ismutable(w.values)
                LinearAlgebra.normalize!(w, 1)
            else
                w = StatsBase.pweights(LinearAlgebra.normalize(w, 1))
            end
        end
        if !isnothing(mu_views) ||
           !isnothing(var_views) ||
           !isnothing(cvar_views) ||
           !isnothing(evar_views) ||
           !isnothing(sigma_views) ||
           !isnothing(sk_views) ||
           !isnothing(kt_views) ||
           !isnothing(cov_views) ||
           !isnothing(rho_views)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if !isnothing(cvar_views)
            @argcheck(isa(opt, JuMPEntropyPooling),
                      ArgumentError("A CVaR view needs auxiliary variables, which the dual formulation `OptimEntropyPooling` solves has no room for. Use `JuMPEntropyPooling` in `opt`."))
        end
        if isa(var_views, AbstractVector)
            @argcheck(!isempty(var_views), IsEmptyError("var_views cannot be empty"))
        end
        if isa(cvar_views, AbstractVector)
            @argcheck(!isempty(cvar_views), IsEmptyError("cvar_views cannot be empty"))
        end
        if isa(evar_views, AbstractVector)
            @argcheck(!isempty(evar_views), IsEmptyError("evar_views cannot be empty"))
        end
        return new{typeof(pe), typeof(mu_views), typeof(var_views), typeof(cvar_views),
                   typeof(evar_views), typeof(sigma_views), typeof(sk_views),
                   typeof(kt_views), typeof(cov_views), typeof(rho_views), typeof(sets),
                   typeof(opt), typeof(w), typeof(alg)}(pe, mu_views, var_views, cvar_views,
                                                        evar_views, sigma_views, sk_views,
                                                        kt_views, cov_views, rho_views,
                                                        sets, opt, w, alg)
    end
end
function EntropyPoolingPrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             mu_views::Option{<:LinearConstraintEstimator} = nothing,
                             var_views::Option{<:VV_VecVV} = nothing,
                             cvar_views::Option{<:CVV_VecCVV} = nothing,
                             evar_views::Option{<:EVV_VecEVV} = nothing,
                             sigma_views::Option{<:LinearConstraintEstimator} = nothing,
                             sk_views::Option{<:LinearConstraintEstimator} = nothing,
                             kt_views::Option{<:LinearConstraintEstimator} = nothing,
                             cov_views::Option{<:LinearConstraintEstimator} = nothing,
                             rho_views::Option{<:LinearConstraintEstimator} = nothing,
                             sets::Option{<:UniverseSets} = nothing,
                             opt::NonCVaREP = OptimEntropyPooling(),
                             w::Option{<:StatsBase.ProbabilityWeights} = nothing,
                             alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())::EntropyPoolingPrior
    return EntropyPoolingPrior(pe, mu_views, var_views, cvar_views, evar_views, sigma_views,
                               sk_views, kt_views, cov_views, rho_views, sets, opt, w, alg)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties EntropyPoolingPrior begin
    forward(pe, me, ce)
end
"""
    const VecEP = AbstractVector{<:Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}}

Alias for an abstract vector of entropy pooling prior estimators of either family.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
const VecEP = AbstractVector{<:Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}}
"""
    prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute the entropy pooling prior of asset returns with tail views.

`prior` orients the data and forwards the estimator's algorithm as a value to [`ep_prior`](@ref), which enforces the views in stages or in one optimisation (ADR 0064).

# Arguments

  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix.
  - `F`: Optional factor returns matrix.
  - `dims`: Dimension along which the observations lie.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`ep_prior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    return ep_prior(pe.alg, pe, X, F; strict = strict, kwargs...)
end
"""
    ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
             strict::Bool = false, kwargs...)

Compute entropy pooling prior moments with tail views, enforcing the views in stages.

`ep_prior` accumulates the views of each stage into one constraint set and solves once per stage, so a stage's views hold alongside every view of the stages before it. The mean of an asset a later stage constrains is pinned to the value the earlier stage produced, so a higher moment view does not silently move it.

# Arguments

  - `alg`: Staged entropy pooling algorithm.

      + `::H1_EntropyPooling`: Each stage re-solves from the original prior weights.
      + `::H2_EntropyPooling`: Each stage re-solves from the previous stage's weights.

  - `pe`: Entropy pooling prior estimator.

  - `X`: Asset returns matrix, already oriented.

  - `F`: Optional factor returns matrix, already oriented.

  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Details

The stages are:

 1. Mean, value at risk, conditional value at risk and entropic value at risk views.
 2. Variance and covariance views, with the mean of every asset they name pinned.
 3. Skewness, kurtosis and correlation views, with the mean and variance of every asset they name pinned.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
                  strict::Bool = false, kwargs...)
    T, N = size(X)
    w1 = w0 = if isnothing(pe.w)
        iT = inv(T)
        StatsBase.pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T,
                  DimensionMismatch("length(pe.w) ($(length(pe.w))) must match T ($T)"))
        pe.w
    end
    fixed = falses(N, 2)
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    tvs = Vector{AbstractEntropyPoolingTailView}(undef, 0)
    # mu, VaR, CVaR and EVaR
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets; strict = strict)
    ep_cvar_views!(pe.cvar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    ep_evar_views!(pe.evar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    if !isempty(epc) || !isempty(tvs)
        w1 = entropy_pooling(w0, epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    if !isnothing(pe.sigma_views) || !isnothing(pe.cov_views)
        # sigma
        if !isnothing(pe.sigma_views)
            to_fix = ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        end
        # cov
        if !isnothing(pe.cov_views)
            to_fix = ep_cov_views!(pe.cov_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        end
        w1 = entropy_pooling(ifelse(isa(alg, H1_EntropyPooling), w0, w1), epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            to_fix = ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            to_fix = ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # rho
        if !isnothing(pe.rho_views)
            to_fix = ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        w1 = entropy_pooling(ifelse(isa(alg, H1_EntropyPooling), w0, w1), epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same seam in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr, Z) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, Z = Z)
end
"""
    ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
             F::Option{<:MatNum}; strict::Bool = false, kwargs...)

Compute entropy pooling prior moments with tail views, enforcing every view in one optimisation.

`ep_prior` builds every view constraint against the same prior and solves once. It is faster than the staged algorithms and pins nothing, so a higher moment view is free to move a lower moment.

# Arguments

  - `alg`: Single-shot entropy pooling algorithm.
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix, already oriented.
  - `F`: Optional factor returns matrix, already oriented.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
                  F::Option{<:MatNum}; strict::Bool = false, kwargs...)
    T = size(X, 1)
    w0 = if isnothing(pe.w)
        iT = inv(T)
        StatsBase.pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T,
                  DimensionMismatch("length(pe.w) ($(length(pe.w))) must match T ($T)"))
        pe.w
    end
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    tvs = Vector{AbstractEntropyPoolingTailView}(undef, 0)
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    # mu, VaR, CVaR and EVaR
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets; strict = strict)
    ep_cvar_views!(pe.cvar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    ep_evar_views!(pe.evar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    # sigma
    if !isnothing(pe.sigma_views)
        ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
    end
    # cov
    if !isnothing(pe.cov_views)
        ep_cov_views!(pe.cov_views, epc, pr, pe.sets; strict = strict)
    end
    # skew
    if !isnothing(pe.sk_views)
        ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
    end
    # kurtosis
    if !isnothing(pe.kt_views)
        ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
    end
    # rho
    if !isnothing(pe.rho_views)
        ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
    end
    w1 = entropy_pooling(w0, epc, tvs, pe.opt)
    pe = factory(pe, w1)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same seam in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr, Z) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, Z = Z)
end
function factor_residual_config(pe::EntropyPoolingPrior)
    return factor_residual_config(pe.pe)
end

export EntropyPoolingPrior
