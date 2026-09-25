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
                  pr::AbstractPriorResult, sets::UniverseSets,
                  w::Option{<:ObsWeights} = nothing; strict::Bool = false)
    ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                  pr::AbstractPriorResult, sets::UniverseSets, alpha::Number,
                  w::Option{<:ObsWeights} = nothing; strict::Bool = false)

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

!!! note "The view is met to one observation"

    The constraint fixes the posterior mass of ``\\mathcal{T}_{i}(\\bar{v})``, and the posterior value at risk of that mass is a sample order statistic: [`ValueatRisk`](@ref) reads the first observation whose cumulative weight reaches ``\\alpha``. An entropy pooling solve meets its constraints to its own tolerance, and a mass short of ``\\alpha`` by as little as `1e-8` reads one observation further down the tail. The posterior value at risk then sits under ``\\bar{v}``, by the gap between two neighbouring losses. No tolerance on the solve removes this, because the reading is a step function of the mass. The view is met to the resolution the sample has, which is one observation. Read a posterior value at risk against the two observations that bracket the target, and read the tail mass where an exact statement is needed.

# Algorithm

 1. Parse the view equations of `var_views.val`, accepting `==` and `>=` alone.
 2. Replace every group name by the assets it spans.
 3. Replace every `prior(...)` reference by the prior value at risk at `alpha`, read under `w`, through [`replace_prior_views`](@ref).
 4. Turn the parsed views into the linear constraint blocks `lcs`, one for `:ineq` and one for `:eq`. Under `strict = false` every row of the group can drop, and `lcs` is then `nothing`: the group states no view, and the call returns without adding a row.
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
  - $(arg_dict[:oow])
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
                       pr::AbstractPriorResult, sets::UniverseSets,
                       w::Option{<:ObsWeights} = nothing; kwargs...)
    return ep_var_views!(var_views.views, epc, pr, sets, var_views.alpha, w; kwargs...)
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
    ep_prior_probabilities(w::Option{<:StatsBase.ProbabilityWeights},
                           pr::AbstractPriorResult, Ti::Int)

Return the prior probabilities an entropy pooling fit starts from.

A prior that reweights observations works on the observation axis its nested prior **answered**, not on the axis it was handed. A nested prior may drop rows: a [`CrossSectionalFactorPrior`](@ref) drops the observations its Descriptors warm up over and the observations its exposure lag consumes, so its scenarios are the window the fit is defined on. So `pr` is fitted first, and the prior probabilities are read on its rows.

The three sources are read in order. A caller's `pe.w` wins, because it is the one tilt no fit can state. The nested result's own `w` comes next, because a nested pooling prior already tilted the scenarios it answered, and uniform is then not the prior. Uniform over the rows of `pr.X` is the last.

# Algorithm

 1. Take `T` as `size(pr.X, 1)`, the observations the nested prior answered.
 2. When `w` is not `nothing`, check its length against `T` and return it.
 3. When `w` is `nothing` and `pr.w` is not, return `pr.w` as `StatsBase.pweights`.
 4. Otherwise return the uniform `1/T` as `StatsBase.pweights`.

# Arguments

  - `w`: A caller's prior probabilities, the `w` field of the pooling estimator, or `nothing`.
  - `pr`: Prior result of the nested estimator, fitted before this call.
  - `Ti`: Observations the pooling estimator was handed. It is read only by the refusal message.

# Validation

  - `length(w) == size(pr.X, 1)`. A length that does not match raises a `DimensionMismatch` naming both counts and the count the estimator was handed.

# Returns

  - `w0::StatsBase.ProbabilityWeights`: Prior probabilities, on the rows of `pr.X`.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`entropy_pooling`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function ep_prior_probabilities(::Nothing, pr::AbstractPriorResult, ::Int)
    return if isnothing(pr.w)
        T = size(pr.X, 1)
        iT = inv(T)
        StatsBase.pweights(range(iT, iT; length = T))
    else
        StatsBase.pweights(pr.w)
    end
end
function ep_prior_probabilities(w::StatsBase.ProbabilityWeights, pr::AbstractPriorResult,
                                Ti::Int)
    T = size(pr.X, 1)
    @argcheck(length(w) == T,
              DimensionMismatch("length(pe.w) ($(length(w))) must match the $T observations the nested prior answered. The estimator was handed $Ti. A prior that reweights observations states its prior probabilities on the scenarios its nested prior produced, so a nested prior that drops rows moves this axis."))
    return w
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

 1. Return `w` when `epc` holds no row. An empty view set states nothing, so the posterior is the prior, and it is answered exactly rather than solved for.
 2. Open `A` and `B` with the row that pins the posterior to sum to one, both sides divided by ``\\sqrt{T}``.
 3. Stack the block of every key of `epc` onto `A` and `B`, and set the box `wb` of that block's dual variables from the key: free for `:eq` and `:cvar_eq`, non-negative for `:ineq`, and ``[-s_{c2},\\, s_{c2}]`` for `:feq`. Raise on any other key. A `:feq` block is left out when ``s_{c2}`` is zero, because that box pins its dual variables to zero and the fixed rows then carry no weight.
 4. Start every dual variable at ``1/\\sqrt{T}``, clamped into its own box. A `:feq` box is ``[-s_{c2},\\, s_{c2}]``, so an `s_{c2}` below ``1/\\sqrt{T}`` would otherwise place the start outside it.
 5. Minimise the dual objective over that box with `Optim.optimize`, through the branch `alg` selects. Both the objective and its gradient are multiplied by ``s_{c1}``.
 6. Raise when `Optim.converged` reports that the solve failed.
 7. Recover the posterior probabilities from the minimiser, and return them as `StatsBase.pweights`.

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
    #! An empty view set states nothing, so the posterior is the prior. Every row of every
    #! family can drop under `strict = false`, and the solve is then skipped rather than run
    #! over the normalisation row alone: that answers `w` to solver tolerance where this
    #! answers it exactly, and `kld` is zero rather than a rounding of it. See issue #852.
    if isempty(epc)
        return isa(w, StatsBase.ProbabilityWeights) ? w : StatsBase.pweights(w)
    end
    T = length(w)
    factor = inv(sqrt(T))
    blocks = Any[(fill(factor, 1, T), [factor])]
    wbs = AbstractMatrix[[typemin(eltype(w)) typemax(eltype(w))]]
    for (key, val) in epc
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
    #! An empty view set states nothing, so the posterior is the prior. Every row of every
    #! family can drop under `strict = false`, and the solve is then skipped rather than run
    #! over the normalisation row alone: that answers `w` to solver tolerance where this
    #! answers it exactly, and `kld` is zero rather than a rounding of it. See issue #852.
    if isempty(epc)
        return isa(w, StatsBase.ProbabilityWeights) ? w : StatsBase.pweights(w)
    end
    T = length(w)
    factor = inv(sqrt(T))
    blocks = Any[(fill(factor, 1, T), [factor])]
    wbs = AbstractMatrix[[typemin(eltype(w)) typemax(eltype(w))]]
    for (key, val) in epc
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
    return StatsBase.pweights(exp.(log_p .- (one(eltype(log_p)) .+ transpose(A) * x)))
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
