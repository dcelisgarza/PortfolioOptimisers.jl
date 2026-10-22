"""
    ep_var_views!(var_views::Nothing, args...; kwargs...)

Do nothing when a problem states no value at risk view.

`ep_var_views!` is the verb that turns a group of value at risk views into rows of the entropy pooling constraint dictionary. This method is the branch for an absent view. It registers no row, so a caller can call the verb with `var_views = nothing` and needs no special case.

# Arguments

  - `var_views::Nothing`: Shows that the problem states no value at risk view.
  - `args...`: More positional arguments, which the method does not read.
  - `kwargs...`: More keyword arguments, which the method does not read.

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
                  pr::AbstractPriorResult, sets::UniverseSets,
                  w::Option{<:ObsWeights} = nothing; strict::Bool = false)
    ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::UniverseSets, alpha::Number,
                  w::Option{<:ObsWeights} = nothing; strict::Bool = false)

Add the value at risk views of a group to the entropy pooling constraint dictionary.

The first method gives the equations and the significance level of a [`ValueatRiskView`](@ref) to the second method. The second method parses the view equations, replaces each prior reference with its value, and makes the linear constraints of the views. The statistic is the value at risk of the posterior distribution, not the variance, which the `sigma_views` family holds.

A value at risk view is linear in the posterior probabilities, because it constrains the probability mass at or beyond the target loss. So it needs no auxiliary variable and fixes no moment.

# Mathematical definition

The value at risk at the level ``\\alpha`` is the smallest loss beyond which the posterior has at most ``\\alpha`` of its mass. So a view on it is a statement about the tail mass of the sample:

```math
\\begin{align}
\\mathrm{VaR}_{\\alpha}(x_{i}) \\geq \\bar{v} \\quad &\\Longleftrightarrow \\quad \\sum_{t \\in \\mathcal{T}_{i}(\\bar{v})} p_{t} \\geq \\alpha\\,, \\\\
\\mathrm{VaR}_{\\alpha}(x_{i}) = \\bar{v} \\quad &\\Longleftrightarrow \\quad \\sum_{t \\in \\mathcal{T}_{i}(\\bar{v})} p_{t} = \\alpha\\,, \\\\
\\mathcal{T}_{i}(\\bar{v}) &= \\left\\{ t : x_{t,\\,i} \\leq -\\lvert \\bar{v} \\rvert \\right\\}\\,.
\\end{align}
```

Both are linear in ``\\boldsymbol{p}``, so both [`OptimEntropyPooling`](@ref) and [`JuMPEntropyPooling`](@ref) accept this view.

Where:

  - $(math_dict[:ep_post_probs])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:alpha_rm])
  - ``\\bar{v}``: Target value at risk of the view.
  - ``\\mathcal{T}_{i}(\\bar{v})``: Observations of asset ``i`` whose loss reaches the target.

!!! note "The view is met to one observation"

    The constraint fixes the posterior mass of ``\\mathcal{T}_{i}(\\bar{v})``, and the posterior value at risk is a sample order statistic. [`ValueatRisk`](@ref) reads the first observation whose cumulative weight is larger than ``\\alpha``. A mass of exactly ``\\alpha`` is not larger, so [`ValueatRisk`](@ref) passes every observation of ``\\mathcal{T}_{i}(\\bar{v})`` and returns the largest loss below ``\\bar{v}``. A mass above ``\\alpha`` by more than the slack of [`ValueatRisk`](@ref) gives the smallest loss in ``\\mathcal{T}_{i}(\\bar{v})``, which is at or above ``\\bar{v}``. An entropy pooling solve meets its constraints to its own tolerance, about `1e-8`, so the sign of its residual selects one of the two results.

    No tolerance on the solve removes this, because the posterior value at risk is a step function of the mass. The resolution of the view is the resolution of the sample, which is one observation. Compare a posterior value at risk with the two observations on each side of the target. Read the tail mass when an exact statement is necessary.

# Algorithm

 1. Parse the view equations of `var_views.val`. The parser accepts only `==` and `>=`.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior value at risk at `alpha` under `w`, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false`, every row of the group can drop, and `lcs` is then `nothing`. The group then states no view, and the call returns with no row added.
 5. Check the three preconditions of the section below.
 6. For each block that `lcs` holds, and each row `i` of it, read the asset that the row names into `j`. Read the observations of `view(X, :, j)` at or below `-abs(B[i])` into `idx`.
 7. Raise when `idx` names no observation.
 8. Read the sense of the row into `sign`. It is one for an equality row and for a positive right-hand side, and minus one otherwise. So an `:ineq` row with a target of zero takes minus one, as a row with a positive target does.
 9. Build the row `Ai` that has `sign` at `idx` and zero at the other observations, and add it against `sign * alpha` with [`add_ep_constraint!`](@ref).

# Arguments

  - `var_views`: Value at risk view constraints.
  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.
  - `pr`: Prior result that holds the asset returns.
  - `sets`: Universe sets that map the asset names to indices.
  - `alpha`: Significance level of the value at risk.
  - $(arg_dict[:oow])
  - `strict`: If `true`, raise an error for an asset that the universe does not hold. If `false`, warn.

# Validation

  - Every coefficient has magnitude zero or one. Any other coefficient raises an `ArgumentError`. The check reads the magnitude because the parser normalises a `>=` row to `<=` by negation, so a coefficient of one goes into the block as minus one.
  - Every view names one asset. A view over more than one asset raises an `ArgumentError`.
  - Every target is non-negative. A negative target raises a `DomainError`.
  - The sample must hold at least one observation whose loss reaches the target. A view more extreme than the worst realisation raises a `DomainError` that names the largest target that the asset admits.

# Returns

  - `nothing`: The function mutates `epc` in place.

# Related

  - [`ValueatRiskView`](@ref)
  - [`add_ep_constraint!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref): reads the prior value at risk that a `prior(...)` reference resolves to.
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
function ep_var_views!(var_views::ValueatRiskView, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::UniverseSets,
                       w::Option{<:ObsWeights} = nothing; kwargs...)
    return ep_var_views!(var_views.views, epc, pr, sets, var_views.alpha, w; kwargs...)
end
"""
    ep_var_views!(var_views::AbstractVector{<:ValueatRiskView}, args...; kwargs...)

Add each group of value at risk views under its own significance level.

The method adds each [`ValueatRiskView`](@ref) of the vector in turn. So all the groups go into the same constraint set, and one entropy pooling solve meets all of them.

# Algorithm

 1. Add each [`ValueatRiskView`](@ref) of `var_views` in turn, with `args...` and `kwargs...`.
 2. Return `nothing`. Each call already wrote its rows into `epc`.

# Arguments

  - `var_views`: Groups of value at risk views.
  - `args...`: The positional arguments that the method passes to [`ep_var_views!`](@ref).
  - `kwargs...`: The keyword arguments that the method passes to [`ep_var_views!`](@ref).

# Returns

  - `nothing`: The function mutates `epc` in place.

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
                       pr::AbstractPriorResult, sets::UniverseSets, alpha::Number,
                       w::Option{<:ObsWeights} = nothing; strict::Bool = false,
                       ledger::Option{<:AbstractVector} = nothing)
    X = pr.X
    var_views = parse_equation(var_views.val; ops1 = ("==", ">="),
                               ops2 = (:call, :(==), :(>=)), datatype = eltype(X))
    var_views = replace_group_by_assets(var_views, sets, false, true, false;
                                        ledger = ledger)
    var_views = replace_prior_views(var_views, pr, sets, :var, alpha, w; strict = strict)
    lcs = get_linear_constraints(var_views, sets; datatype = eltype(X), strict = strict,
                                 ledger = ledger)
    #! Under `strict = false` a view that names no asset is warned about and dropped, and
    #! a group whose every row drops parses to `nothing`. The warning is the whole
    #! diagnosis, so the family states no view and the fit proceeds without one. Reading a
    #! block off the `nothing` raised a `FieldError` naming an internal field one call after
    #! that warning. See issue #852.
    if isnothing(lcs)
        return nothing
    end
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
              DomainError(var_views,
                          "A `var_view` states a loss magnitude, so its target is non-negative, and one of these is negative:\n$var_views"))
    Ai = zeros(eltype(X), 1, size(X, 1))
    for p in (:ineq, :eq)
        if isnothing(getproperty(lcs, p))
            continue
        end
        (; A, B) = getproperty(lcs, p)
        for i in eachindex(B)
            j = .!iszero.(view(A, i, :))
            idx = findall(x -> x <= -abs(B[i]), view(X, :, j))
            @argcheck(!isempty(idx),
                      DomainError(abs(B[i]),
                                  "View $(i) = $(var_views[i].eqn) is too extreme, the maximum viable for asset $(findfirst(x -> x == true, j)) is $(-minimum(X[:,j])). Please lower it or use a different prior with fatter tails."))
            #! An `:ineq` row reaches here normalised to `A * p <= B`, and the parser accepts
            #! `>=` alone, so `B[i]` is the negated target and never positive. A zero target
            #! must therefore take the same sign as a positive one; `>=` here read it as a
            #! `<=` view and flipped the row.
            sign = ifelse(p == :eq || B[i] > zero(eltype(B)), one(eltype(B)),
                          -one(eltype(B)))
            fill!(Ai, zero(eltype(Ai)))
            Ai[1, idx] .= sign
            add_ep_constraint!(epc, Ai, [sign * alpha], p)
        end
    end
    return nothing
end
"""
    ep_prior_probabilities(w::Nothing, pr::AbstractPriorResult, Ti::Int)
    ep_prior_probabilities(w::ScenarioProbabilities, T::Integer, Ti::Integer)

Return the prior probabilities from which an entropy pooling fit starts.

A prior that reweights observations works on the observation axis of the result of its nested prior, not on the axis of its input data. A nested prior can drop rows. A [`CrossSectionalFactorPrior`](@ref) drops the warm-up observations of its Descriptors and the observations of its exposure lag, so its scenarios are the window of the fit. So the method reads the prior probabilities on the rows of the nested result, and [`ep_wrapped_prior`](@ref) finds the count `T` of those rows.

The method reads three sources in order. The `pe.w` of the caller comes first, because no fit can state that weighting. The `w` of the nested result comes next, because a nested pooling prior already changed the weights of its scenarios, and the uniform weights are then not the prior. The uniform weights over the rows of `pr.X` are the last. The method for the `pe.w` of the caller runs before the refit with those weights, so it reads the count `T` and not a result. When that refit returns its own `pr.w`, the nested prior solved its views from `pe.w`, and [`ep_wrapped_prior`](@ref) reads its posterior `pr.w` instead.

# Algorithm

The method that Julia selects is the algorithm.

 1. A `nothing` `w`: when `pr.w` is not `nothing`, return it through [`scenario_probabilities`](@ref). That function keeps the kind of a [`ScenarioProbabilities`](@ref), and makes any other vector `StatsBase.AnalyticWeights`. Otherwise, return the uniform `1/T` over the `T = size(pr.X, 1)` rows of `pr.X` as `StatsBase.AnalyticWeights`. [`ScenarioProbabilities`](@ref) states their bias correction.
 2. A `w` of the caller: check its length against `T`, and return it.

# Arguments

  - `w`: The prior probabilities of the caller, from the `w` field of the pooling estimator, or `nothing`.
  - `pr`: Prior result of the nested estimator.
  - `T`: Number of observations of the result of the nested prior.
  - `Ti`: Number of observations of the input data of the pooling estimator. Only the error message reads it.

# Validation

  - `length(w) == T`. A length that does not match raises a `DimensionMismatch` that names both counts and the number of observations of the input data.

# Returns

  - `w0::ScenarioProbabilities`: Prior probabilities, on the rows of the result of the nested prior.

# Related

  - [`ep_wrapped_prior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`entropy_pooling`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function ep_prior_probabilities(::Nothing, pr::AbstractPriorResult, ::Int)
    return if isnothing(pr.w)
        T = size(pr.X, 1)
        iT = inv(T)
        StatsBase.aweights(range(iT, iT; length = T))
    else
        scenario_probabilities(pr.w)
    end
end
function ep_prior_probabilities(w::ScenarioProbabilities, T::Integer, Ti::Integer)
    @argcheck(length(w) == T,
              DimensionMismatch("length(pe.w) ($(length(w))) must match the $T observations the nested prior answered. The estimator was handed $Ti. A prior that reweights observations states its prior probabilities on the scenarios its nested prior produced, so a nested prior that drops rows moves this axis."))
    return w
end
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, opt::OptimEntropyPooling)

Solve the dual of the entropy pooling problem with Optim.jl.

`entropy_pooling` finds the posterior probabilities that minimise the Kullback-Leibler divergence from the prior probabilities, subject to the moment and view constraints. It uses [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). [`MeucciEntropyPoolingPrior`](@ref) and [`EntropyPoolingPrior`](@ref) call this method when the optimiser is an [`OptimEntropyPooling`](@ref).

The two optimisation algorithms minimise the same objective and return the same posterior. They differ only in the arithmetic of the evaluation.

The dual problem has one variable for each row, not one for each observation. It cannot hold an auxiliary variable, so it expresses no tail view. It also has no slack variable. The method relaxes the fixed equality rows of the `:feq` key with the box ``[-s_{c2},\\, s_{c2}]`` on their dual variables. This box is the dual of a penalty of weight ``s_{c2}`` on the norm of the slack of the primal problem.

!!! warning

    An infeasible view set raises no error. The dual of such a set is unbounded below, so the minimiser does not converge to a point. The iterate stops when the exponential underflows, `Optim` reports `x_converged` or `f_converged`, and `Optim.converged` accepts it. The posterior that the method returns is degenerate. Almost all of the probability goes to the observation with the largest coefficient, and the posterior can miss the view of the caller by an arbitrary amount.

    Read the result, not the flag. Three signs show an infeasible view set. The effective number of scenarios falls to a few out of ``T``, and one weight is near one. The Kullback-Leibler divergence is large. The posterior statistic that the view names is far from its target.

    The usual cause is two views that move one asset in opposite directions. For example, a variance view that makes the distribution of an asset narrower, and a conditional value at risk view that makes the tail of the same asset heavier, ask for a thin body and a heavy tail together. The same two views on two different assets are feasible and solve normally. So the cause is the opposite directions, not the pair of views.

    The gradient of this dual is ``\\boldsymbol{B} - \\mathbf{A} \\boldsymbol{y}``, the primal residual of the view set, so `Optim.g_converged` and `Optim.g_residual` can tell the two outcomes apart. The method reads neither. `Optim.g_converged` also refuses a solve that is correct but has a loose tolerance. So a check on it needs a tolerance on the residual, and this library sets no such tolerance.

# Mathematical definition

The primal problem minimises the Kullback-Leibler divergence of the posterior from the prior, over the probabilities that meet every row. Its dual has one Lagrange multiplier for each row, and has no constraint other than the box that the sense of each row sets:

```math
\\begin{align}
\\underset{\\boldsymbol{x}}{\\min} &\\; \\boldsymbol{x}^\\intercal \\boldsymbol{B} + \\sum_{t=1}^{T} q_{t} \\exp\\!\\left(-\\boldsymbol{x}^\\intercal \\mathbf{A}_{\\cdot t} - 1\\right)\\,.
\\end{align}
```

The objective is the negative of the dual function of Equation 91 of [meucci2008](@cite), so the library minimises where [meucci2008](@citet) maximises. The method gets the optimal posterior probabilities from the minimiser as:

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
  - ``\\mathbf{A}_{\\cdot t}``: ``t``-th column of ``\\mathbf{A}``, the coefficient that each row gives observation ``t``.
  - ``q_{t}``, ``p_{t}^{*}``: Prior and optimal posterior probability of observation ``t``.

# Algorithm

 1. Return `w` when `epc` holds no row. An empty view set states nothing, so the posterior is the prior, and the method returns it exactly with no solve.
 2. Start `A` and `B` with the row that holds the sum of the posterior at one, with both sides divided by ``\\sqrt{T}``.
 3. Stack the block of every key of `epc` onto `A` and `B`, in the sorted order of the keys. Set the box `wb` of the dual variables of that block from the key. The box is free for `:eq` and `:cvar_eq`, non-negative for `:ineq`, and ``[-s_{c2},\\, s_{c2}]`` for `:feq`. Raise on any other key. Leave out a `:feq` block when ``s_{c2}`` is zero, because that box holds its dual variables at zero and the fixed rows then have no weight.
 4. Start every dual variable at ``1/\\sqrt{T}``, clamped into its own box. A `:feq` box is ``[-s_{c2},\\, s_{c2}]``, so with no clamp, an ``s_{c2}`` below ``1/\\sqrt{T}`` puts the start outside the box.
 5. Minimise the dual objective over that box with `Optim.optimize`, through the branch that `alg` selects. The method multiplies both the objective and its gradient by ``s_{c1}``.
 6. Raise when `Optim.converged` reports that the solve failed.
 7. Get the posterior probabilities from the minimiser, and return them through [`scenario_probabilities`](@ref), in the kind of `w`.

# Arguments

  - `w`: Prior probabilities, one for each observation.

  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.

  - `opt`: Entropy pooling optimiser that uses Optim.jl.

      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Evaluate the objective through the exponential of the dual variables.
      + `::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Evaluate the objective in log space.

# Validation

  - Every key of `epc` is one of `:eq`, `:ineq`, `:cvar_eq` and `:feq`. Any other key raises a `KeyError`.
  - The solve must converge. A solve that `Optim.converged` reports as failed raises an `ErrorException`.
  - The method does not catch an infeasible view set. `Optim.converged` is true on `x_converged` or `f_converged` only, and the dual of an infeasible set stops on one of those. The warning above states the shape of that result and the signs that show it.

# Returns

  - `pw::ScenarioProbabilities`: Posterior probabilities that meet the constraints, in the kind of `w`.

# Related

  - [`OptimEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`JuMPEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008]) Appendix A.3, Equations 90 to 93.
"""
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:ExpEntropyPooling})
    #! An empty view set states nothing, so the posterior is the prior. Every row of every
    #! family can drop under `strict = false`, and the solve is then skipped rather than run
    #! over the normalisation row alone: that answers `w` to solver tolerance where this
    #! answers it exactly, and `kld` is zero rather than a rounding of it. See issue #852.
    if isempty(epc)
        return scenario_probabilities(w)
    end
    T = length(w)
    factor = inv(sqrt(T))
    blocks = Any[(fill(factor, 1, T), [factor])]
    wbs = AbstractMatrix[[typemin(eltype(w)) typemax(eltype(w))]]
    #! The keys of `epc` are `Symbol`s, which hash by their address, so the order of the
    #! `Dict` moves between two processes. The blocks stack in the sorted order of their keys,
    #! so one view set gives one problem in every process.
    for key in sort!(collect(keys(epc)))
        val = epc[key]
        s = length(val[2])
        #! A `:feq` dual variable is boxed by `sc2`, so `sc2 == 0` pins it to zero and the
        #! fix carries no weight. That is what a primal penalty of weight `sc2` means, and
        #! it is what the JuMP route gives. `Optim` cannot start inside a box of zero
        #! width, so the block is left out rather than boxed to a point.
        if key == :feq && iszero(opt.sc2)
            continue
        end
        push!(blocks, val)
        lo, hi = if key == :eq || key == :cvar_eq
            typemin(eltype(w)), typemax(eltype(w))
        elseif key == :ineq
            zero(eltype(w)), typemax(eltype(w))
        elseif key == :feq
            -opt.sc2, opt.sc2
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
        push!(wbs, repeat([lo hi], s))
    end
    A, B, wb = reduce(vcat, first.(blocks)), reduce(vcat, last.(blocks)), reduce(vcat, wbs)
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
    @static if v"2.0.1" <= pkgversion(Optim) < v"2.4.0"
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
    return scenario_probabilities(w, w .* exp.(-transpose(A) * x .- one(eltype(w))))
end
function entropy_pooling(w::VecNum, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:LogEntropyPooling})
    #! An empty view set states nothing, so the posterior is the prior. Every row of every
    #! family can drop under `strict = false`, and the solve is then skipped rather than run
    #! over the normalisation row alone: that answers `w` to solver tolerance where this
    #! answers it exactly, and `kld` is zero rather than a rounding of it. See issue #852.
    if isempty(epc)
        return scenario_probabilities(w)
    end
    T = length(w)
    factor = inv(sqrt(T))
    blocks = Any[(fill(factor, 1, T), [factor])]
    wbs = AbstractMatrix[[typemin(eltype(w)) typemax(eltype(w))]]
    #! The keys of `epc` are `Symbol`s, which hash by their address, so the order of the
    #! `Dict` moves between two processes. The blocks stack in the sorted order of their keys,
    #! so one view set gives one problem in every process.
    for key in sort!(collect(keys(epc)))
        val = epc[key]
        s = length(val[2])
        #! A `:feq` dual variable is boxed by `sc2`, so `sc2 == 0` pins it to zero and the
        #! fix carries no weight. That is what a primal penalty of weight `sc2` means, and
        #! it is what the JuMP route gives. `Optim` cannot start inside a box of zero
        #! width, so the block is left out rather than boxed to a point.
        if key == :feq && iszero(opt.sc2)
            continue
        end
        push!(blocks, val)
        lo, hi = if key == :eq || key == :cvar_eq
            typemin(eltype(w)), typemax(eltype(w))
        elseif key == :ineq
            zero(eltype(w)), typemax(eltype(w))
        elseif key == :feq
            -opt.sc2, opt.sc2
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
        push!(wbs, repeat([lo hi], s))
    end
    A, B, wb = reduce(vcat, first.(blocks)), reduce(vcat, last.(blocks)), reduce(vcat, wbs)
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
            log_x .= log_p .- (one(eltype(log_p)) .+ transpose(A) * x)
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
    @static if v"2.0.1" <= pkgversion(Optim) < v"2.4.0"
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
    return scenario_probabilities(w,
                                  exp.(log_p .- (one(eltype(log_p)) .+ transpose(A) * x)))
end
"""
    entropy_pooling(w::VecNum, epc::AbstractDict, opt::JuMPEntropyPooling)

Solve the primal of the entropy pooling problem with JuMP.jl.

`entropy_pooling` finds the posterior probabilities that minimise the Kullback-Leibler divergence from the prior probabilities, subject to the moment and view constraints. It uses [`JuMP.jl`](https://jump.dev/JuMP.jl/stable/). [`MeucciEntropyPoolingPrior`](@ref) and [`EntropyPoolingPrior`](@ref) call this method when the optimiser is a [`JuMPEntropyPooling`](@ref).

This method registers no model entry of its own. It is the three-argument form for a problem that states no tail view. The four-argument method builds the model, and its docstring holds the `# JuMP formulation` and the `# Algorithm`.

# Algorithm

 1. Call the four-argument method with an empty `AbstractEntropyPoolingTailView` vector, and return its result.

# Arguments

  - `w`: Prior probabilities, one for each observation.

  - `epc`: Dictionary of entropy pooling constraints, which maps each key to an `(lhs, rhs)` pair.

  - `opt`: Entropy pooling optimiser that uses JuMP.jl.

      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:ExpEntropyPooling}`: Write the divergence with the prior probabilities as the reference.
      + `::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any, <:LogEntropyPooling}`: Write the divergence with a unit reference, and subtract the logarithms of the prior probabilities in the objective.

# Returns

  - `pw::ScenarioProbabilities`: Posterior probabilities that meet the constraints, in the kind of `w`.

# Related

  - [`JuMPEntropyPooling`](@ref)
  - [`ExpEntropyPooling`](@ref)
  - [`LogEntropyPooling`](@ref)
  - [`ep_jump_views!`](@ref): registers the rows of `epc`, and the slack that relaxes the fixed equality rows.
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`OptimEntropyPooling`](@ref)

# References

  - $(ref_dict[:meucci2008]) Section 4, Equations 26 and 27.
"""
function entropy_pooling(w::VecNum, epc::AbstractDict, opt::JuMPEntropyPooling)
    return entropy_pooling(w, epc, AbstractEntropyPoolingTailView[], opt)
end
