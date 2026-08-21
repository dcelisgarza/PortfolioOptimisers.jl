"""
$(DocStringExtensions.TYPEDEF)

Result type for [`DiscreteAllocation`](@ref).

`shares`, `cost` and `w` are signed: a short position carries a negative share count, a negative cost and a negative weight. `retcode` is a failure when either sub-problem failed; `s_retcode` and `l_retcode` carry the short-side and long-side return codes on their own, and `s_model` and `l_model` carry the two JuMP models when `save` is `true`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DiscreteAllocationResult(;
        retcode::OptimisationReturnCode,
        s_retcode::Option{<:OptimisationReturnCode},
        l_retcode::Option{<:OptimisationReturnCode},
        shares::VecNum,
        cost::VecNum,
        w::VecNum,
        cash::Number,
        s_model::Option{<:JuMP.Model},
        l_model::Option{<:JuMP.Model},
        fb::Option{<:OptE_Opt}
    ) -> DiscreteAllocationResult

Keywords correspond to the struct's fields.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
  - [`GreedyAllocationResult`](@ref)

# References

  - $(ref_dict[:martin2021])
"""
@concrete struct DiscreteAllocationResult <: FiniteAllocationOptimisationResult
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:s_retcode])
    """
    s_retcode
    """
    $(field_dict[:l_retcode])
    """
    l_retcode
    """
    $(field_dict[:shares])
    """
    shares
    """
    $(field_dict[:cost_alloc])
    """
    cost
    """
    Realised portfolio weights.
    """
    w
    """
    $(field_dict[:cash_alloc])
    """
    cash
    """
    $(field_dict[:s_model])
    """
    s_model
    """
    $(field_dict[:l_model])
    """
    l_model
    """
    $(field_dict[:fb])
    """
    fb
    function DiscreteAllocationResult(retcode::OptimisationReturnCode,
                                      s_retcode::Option{<:OptimisationReturnCode},
                                      l_retcode::Option{<:OptimisationReturnCode},
                                      shares::VecNum, cost::VecNum, w::VecNum, cash::Number,
                                      s_model::Option{<:JuMP.Model},
                                      l_model::Option{<:JuMP.Model}, fb::Option{<:OptE_Opt})
        return new{typeof(retcode), typeof(s_retcode), typeof(l_retcode), typeof(shares),
                   typeof(cost), typeof(w), typeof(cash), typeof(s_model), typeof(l_model),
                   typeof(fb)}(retcode, s_retcode, l_retcode, shares, cost, w, cash,
                               s_model, l_model, fb)
    end
end
function DiscreteAllocationResult(; retcode::OptimisationReturnCode,
                                  s_retcode::Option{<:OptimisationReturnCode},
                                  l_retcode::Option{<:OptimisationReturnCode},
                                  shares::VecNum, cost::VecNum, w::VecNum, cash::Number,
                                  s_model::Option{<:JuMP.Model},
                                  l_model::Option{<:JuMP.Model},
                                  fb::Option{<:OptE_Opt})::DiscreteAllocationResult
    return DiscreteAllocationResult(retcode, s_retcode, l_retcode, shares, cost, w, cash,
                                    s_model, l_model, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Discrete Allocation portfolio optimiser.

`DiscreteAllocation` allocates a portfolio by solving a Mixed-Integer Programming (MIP) problem to find the optimal number of shares for each asset, minimising the deviation between the target continuous weights and the realised discrete allocation.

The long and the short side of a portfolio are allocated as two separate MIP sub-problems, each with its own share of the cash and its own budget. Each sub-problem holds a non-negative share vector, and the short side is negated when the two are recombined. `s_retcode` and `l_retcode` of the result carry the two return codes.

# Mathematical definition

One sub-problem, under the default [`AbsoluteErrorWeightFinaliser`](@ref):

```math
\\begin{align}
\\underset{\\boldsymbol{x} \\in \\mathbb{Z}_{\\geq 0}^N}{\\min} \\quad & u + r\\,, \\\\
\\text{s.t.} \\quad & u \\geq \\lVert \\boldsymbol{w} C - \\boldsymbol{x} \\odot \\boldsymbol{p} \\rVert_1\\,, \\\\
& r = C - \\boldsymbol{x}^\\intercal \\boldsymbol{p} \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: Integer share vector.
  - ``u``: Tracking error auxiliary variable.
  - ``r``: Residual cash.
  - ``\\boldsymbol{w}``: Target weight vector of this sub-problem.
  - ``C``: Cash allocated to this sub-problem.
  - ``\\boldsymbol{p}``: Asset price vector.
  - ``\\odot``: Element-wise (Hadamard) product.
  - ``N``: Number of assets in this sub-problem.

`wf` selects the deviation that ``u`` bounds. The objective, the integrality and the cash constraint do not change with it.

| `wf`                                          | Constraint on ``u``                                                                                                 |
|:--------------------------------------------- |:------------------------------------------------------------------------------------------------------------------- |
| [`AbsoluteErrorWeightFinaliser`](@ref)        | ``u \\geq \\lVert \\boldsymbol{w} C - \\boldsymbol{x} \\odot \\boldsymbol{p} \\rVert_1``                            |
| [`SquaredAbsoluteErrorWeightFinaliser`](@ref) | ``u \\geq \\lVert \\boldsymbol{w} C - \\boldsymbol{x} \\odot \\boldsymbol{p} \\rVert_2``                            |
| [`RelativeErrorWeightFinaliser`](@ref)        | ``u \\geq \\lVert \\boldsymbol{x} C \\oslash (\\boldsymbol{w} \\odot \\boldsymbol{p}) - \\boldsymbol{1} \\rVert_1`` |
| [`SquaredRelativeErrorWeightFinaliser`](@ref) | ``u \\geq \\lVert \\boldsymbol{x} C \\oslash (\\boldsymbol{w} \\odot \\boldsymbol{p}) - \\boldsymbol{1} \\rVert_2`` |

Where ``\\oslash`` is element-wise division, and ``\\boldsymbol{1}`` is the vector of ones.

!!! note

    The two `Squared` formulations bound the ``\\ell_2`` **norm** itself, not its square: they build a `JuMP.SecondOrderCone` over ``[u;\\, \\cdot]``. The square is monotonic on a non-negative norm, so the minimiser is the one a squared objective would give, but the objective *value* is the norm. The two relative formulations replace a zero target weight with `eps` so that the division is defined.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DiscreteAllocation(;
        slv::Slv_VecSlv,
        sc::Number = 1,
        so::Number = 1,
        wf::JuMPWeightFinaliserFormulation = AbsoluteErrorWeightFinaliser(),
        fb::Option{<:FOptE_FOpt} = GreedyAllocation()
    ) -> DiscreteAllocation

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a vector: `!isempty(slv)`.
  - `sc > 0`, `so > 0`.

# Examples

```jldoctest
julia> DiscreteAllocation(; slv = Solver(; solver = nothing))
DiscreteAllocation
  slv ┼ Solver
      │          name ┼ String: ""
      │        solver ┼ nothing
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
   sc ┼ Int64: 1
   so ┼ Int64: 1
   wf ┼ AbsoluteErrorWeightFinaliser()
   fb ┼ GreedyAllocation
      │     unit ┼ Int64: 1
      │     args ┼ Tuple{}: ()
      │   kwargs ┼ @NamedTuple{}: NamedTuple()
      │       fb ┴ nothing
```

# Related

  - [`optimise`](@ref)
  - [`DiscreteAllocationResult`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`set_discrete_error!`](@ref)

# References

  - $(ref_dict[:martin2021])
"""
@concrete struct DiscreteAllocation <: FiniteAllocationOptimisationEstimator
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:fb])
    """
    fb
    function DiscreteAllocation(slv::Slv_VecSlv, sc::Number, so::Number,
                                wf::JuMPWeightFinaliserFormulation,
                                fb::Option{<:FOptE_FOpt})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc > zero(sc), DomainError(sc, "sc must be > 0"))
        @argcheck(so > zero(so), DomainError(so, "so must be > 0"))
        return new{typeof(slv), typeof(sc), typeof(so), typeof(wf), typeof(fb)}(slv, sc, so,
                                                                                wf, fb)
    end
end
function DiscreteAllocation(; slv::Slv_VecSlv, sc::Number = 1, so::Number = 1,
                            wf::JuMPWeightFinaliserFormulation = AbsoluteErrorWeightFinaliser(),
                            fb::Option{<:FOptE_FOpt} = GreedyAllocation())::DiscreteAllocation
    return DiscreteAllocation(slv, sc, so, wf, fb)
end
"""
    set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                        wf::JuMPWeightFinaliserFormulation) -> Nothing

Bound the model's auxiliary variable `u` by the allocation error that `wf` selects.

Adds the one constraint that separates the four formulations of [`DiscreteAllocation`](@ref). The model already holds the share vector `x` and the auxiliary variable `u`; this method adds the cone that ties them together. The objective and the cash constraint are set by [`finite_sub_allocation`](@ref) and do not depend on `wf`.

# Arguments

  - `model::JuMP.Model`: Model holding `x`, `u` and the constraint scale.
  - `w::VecNum`: Target weights of this sub-problem.
  - `p::VecNum`: Asset prices, in the same order as `w`.
  - `cash::Number`: Cash allocated to this sub-problem.
  - `wf::JuMPWeightFinaliserFormulation`: Selects the error. See the table in [`DiscreteAllocation`](@ref).

# Returns

  - `nothing`.

# Details

  - The absolute formulations bound the error `w * cash - x .* p`; the relative ones bound `(x * cash) ⊘ (w .* p) .- 1`.
  - The unsquared formulations use a `JuMP.MOI.NormOneCone`; the squared ones use a `JuMP.SecondOrderCone`, which bounds the ``\\ell_2`` norm itself rather than its square.
  - The relative formulations replace a zero target weight with `eps(eltype(w))` on a **copy** of `w`, so the caller's vector is untouched and the division is defined.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`JuMPWeightFinaliserFormulation`](@ref)
"""
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::RelativeErrorWeightFinaliser)
    mask = iszero.(w)
    if any(mask)
        w = copy(w)
        w[mask] .= eps(eltype(w))
    end
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, crel_err,
                     [sc * u
                      sc *
                      ((x * cash) ⊘ (w .* p) .- one(promote_type(eltype(w), eltype(p))))] in
                     JuMP.MOI.NormOneCone(length(x) + 1))
    return nothing
end
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::SquaredRelativeErrorWeightFinaliser)
    mask = iszero.(w)
    if any(mask)
        w = copy(w)
        w[mask] .= eps(eltype(w))
    end
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, csqrel_err,
                     [sc * u;
                      sc *
                      ((x * cash) ⊘ (w .* p) .- one(promote_type(eltype(w), eltype(p))))] in
                     JuMP.SecondOrderCone())
    return nothing
end
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::AbsoluteErrorWeightFinaliser)
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, cabs_err,
                     [sc * u; sc * (w * cash - x .* p)] in
                     JuMP.MOI.NormOneCone(length(x) + 1))
    return nothing
end
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::SquaredAbsoluteErrorWeightFinaliser)
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, csqabs_err,
                     [sc * u;
                      sc * (w * cash - x .* p)] in JuMP.SecondOrderCone())
    return nothing
end
"""
    finite_sub_allocation(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                          da::DiscreteAllocation, str_names::Bool = false)

Build and solve the discrete allocation MIP for one side, long or short, of the portfolio.

Implements the sub-problem of [`DiscreteAllocation`](@ref). An empty `w` returns three empty vectors, the untouched `cash`, and `nothing` for both the return code and the model.

# Arguments

  - `w::VecNum`: Target weights of this side, non-negative.
  - `p::VecNum`: Asset prices of this side, in the same order as `w`.
  - `cash::Number`: Cash allocated to this side.
  - `bgt::Number`: Budget of this side, used to rescale the realised weights.
  - `da::DiscreteAllocation`: Allocator carrying the solvers, the scales and the formulation `wf`.
  - `str_names::Bool = false`: Whether to give the JuMP variables string names.

# Returns

  - `shares::VecNum`: Share count per asset, rounded to `Int`.
  - `cost::VecNum`: `shares .* p`.
  - `aw::VecNum`: Realised weights, rescaled to sum to `bgt`. All zero when nothing was bought.
  - `acash::Number`: Residual cash `r` of the solved model.
  - `res::OptimisationReturnCode`: An [`OptimisationSuccess`](@ref) or an [`OptimisationFailure`](@ref) carrying the solver trials.
  - `model::JuMP.Model`: The solved model.

# Details

  - The share vector is declared integer and non-negative, so a short side must be passed with its weights already negated.
  - [`set_discrete_error!`](@ref) adds the one constraint that `da.wf` selects. Everything else in the model is common to the four formulations.
  - `shares` is read back with `round(Int, ...)`, because a MIP solver returns an integer only to within its own tolerance.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`set_discrete_error!`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`adjust_long_cash`](@ref)
"""
function finite_sub_allocation(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                               da::DiscreteAllocation, str_names::Bool = false)
    if isempty(w)
        return Vector{eltype(w)}(undef, 0), Vector{eltype(w)}(undef, 0),
               Vector{eltype(w)}(undef, 0), cash, nothing, nothing
    end
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    JuMP.@expression(model, sc, da.sc)
    JuMP.@expression(model, so, da.so)
    N = length(w)
    # Integer allocation
    # x := number of shares
    # u := bounding variable
    JuMP.@variables(model, begin
                        x[1:N] >= 0, Int
                        u
                    end)
    # r := remaining money
    # eta := ideal_investment - discrete_investment
    JuMP.@expression(model, r, cash - LinearAlgebra.dot(x, p))
    JuMP.@constraint(model, cr, sc * r >= 0)
    set_discrete_error!(model, w, p, cash, da.wf)
    JuMP.@objective(model, Min, so * (u + r))
    res = optimise_JuMP_model!(model, da.slv)
    res = if res.success
        OptimisationSuccess(; res = res.trials)
    else
        OptimisationFailure(; res = res.trials)
    end
    shares = round.(Int, JuMP.value.(x))
    cost = shares .* p
    aw = if any(!iszero, cost)
        cost / sum(cost) * bgt
    else
        range(zero(eltype(w)), zero(eltype(w)); length = N)
    end
    acash = JuMP.value(r)
    return shares, cost, aw, acash, res, model
end
function _optimise(da::DiscreteAllocation, fai::FiniteAllocationInput;
                   str_names::Bool = false, save::Bool = true, kwargs...)
    w, p, cash, T, fees = fai.w, fai.prices, fai.cash, fai.horizon, fai.fees
    cash, bgt, lbgt, sbgt, lidx, sidx, lcash, scash = setup_alloc_optim(w, p, cash, T, fees)
    sshares, scost, sw, scash, sretcode, smodel = finite_sub_allocation(-view(w, sidx),
                                                                        view(p, sidx),
                                                                        scash, sbgt, da,
                                                                        str_names)
    lcash = adjust_long_cash(bgt, lcash, scash)
    lshares, lcost, lw, lcash, lretcode, lmodel = finite_sub_allocation(view(w, lidx),
                                                                        view(p, lidx),
                                                                        lcash, lbgt, da,
                                                                        str_names)

    res = Matrix{eltype(w)}(undef, length(w), 3)
    res[lidx, 1] = lshares
    res[sidx, 1] = -sshares
    res[lidx, 2] = lcost
    res[sidx, 2] = -scost
    res[lidx, 3] = lw
    res[sidx, 3] = -sw
    retcode = if isa(sretcode, OptimisationFailure) || isa(lretcode, OptimisationFailure)
        if isa(sretcode, OptimisationFailure)
            @warn("Failed to solve sub optimisation problem. Check `s_retcode.res` for details.")
        end
        if isa(lretcode, OptimisationFailure)
            @warn("Failed to solve sub optimisation problem. Check `l_retcode.res` for details.")
        end
        OptimisationFailure()
    else
        OptimisationSuccess()
    end
    return DiscreteAllocationResult(; retcode = retcode, s_retcode = sretcode,
                                    l_retcode = lretcode, shares = view(res, :, 1),
                                    cost = view(res, :, 2), w = view(res, :, 3),
                                    cash = lcash, s_model = ifelse(save, smodel, nothing),
                                    l_model = ifelse(save, lmodel, nothing), fb = nothing)
end
"""
    optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, <:Any, Nothing},
             fai::FiniteAllocationInput; str_names::Bool = false,
             save::Bool = true, kwargs...) -> DiscreteAllocationResult

Run the Discrete Allocation portfolio optimisation.

# Arguments

  - `da`: The discrete allocation optimiser to use.
  - `fai`: The [`FiniteAllocationInput`](@ref) carrying the target weights, prices, cash budget, and optional horizon and fees.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Returns

  - `res::DiscreteAllocationResult`: The realised allocation. `retcode` is an [`OptimisationFailure`](@ref) when either sub-problem failed, and each failure raises a warning naming the side.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`DiscreteAllocationResult`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
function optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, <:Any, Nothing},
                  fai::FiniteAllocationInput; str_names::Bool = false, save::Bool = true,
                  kwargs...)
    return _optimise(da, fai; str_names = str_names, save = save, kwargs...)
end

export DiscreteAllocationResult, DiscreteAllocation
