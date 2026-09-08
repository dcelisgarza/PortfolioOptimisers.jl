"""
$(DocStringExtensions.TYPEDEF)

Result type for [`DiscreteAllocation`](@ref).

`shares`, `cost` and `w` are signed: a short position carries a negative share count, a negative cost and a negative weight. `fees` is the charge the two sub-problems paid over the whole horizon, and it is never signed. `retcode` is a failure when either sub-problem failed; `s_retcode` and `l_retcode` carry the short-side and long-side return codes on their own, and `s_model` and `l_model` carry the two JuMP models when `save` is `true`.

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
        fees::Number,
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
    $(field_dict[:fees_alloc])
    """
    fees
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
                                      fees::Number, s_model::Option{<:JuMP.Model},
                                      l_model::Option{<:JuMP.Model}, fb::Option{<:OptE_Opt})
        return new{typeof(retcode), typeof(s_retcode), typeof(l_retcode), typeof(shares),
                   typeof(cost), typeof(w), typeof(cash), typeof(fees), typeof(s_model),
                   typeof(l_model), typeof(fb)}(retcode, s_retcode, l_retcode, shares, cost,
                                                w, cash, fees, s_model, l_model, fb)
    end
end
function DiscreteAllocationResult(; retcode::OptimisationReturnCode,
                                  s_retcode::Option{<:OptimisationReturnCode},
                                  l_retcode::Option{<:OptimisationReturnCode},
                                  shares::VecNum, cost::VecNum, w::VecNum, cash::Number,
                                  fees::Number, s_model::Option{<:JuMP.Model},
                                  l_model::Option{<:JuMP.Model},
                                  fb::Option{<:OptE_Opt})::DiscreteAllocationResult
    return DiscreteAllocationResult(retcode, s_retcode, l_retcode, shares, cost, w, cash,
                                    fees, s_model, l_model, fb)
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
& r = C - \\boldsymbol{x}^\\intercal \\boldsymbol{p}\\,, \\\\
& r - F(\\boldsymbol{x}) \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: Integer share vector.
  - ``u``: Tracking error auxiliary variable.
  - ``r``: Residual cash.
  - ``F(\\boldsymbol{x})``: Fee of this sub-problem, of [`set_allocation_fees!`](@ref). It is zero when the input states no fee.
  - ``\\boldsymbol{w}``: Target weight vector of this sub-problem.
  - ``C``: Cash allocated to this sub-problem.
  - ``\\boldsymbol{p}``: Asset price vector.
  - ``\\odot``: Element-wise (Hadamard) product.
  - ``N``: Number of assets in this sub-problem.

The fee enters the budget and never the objective. The objective reads *track well, and leave no capital idle*, so a fee added to ``r`` and minimised would reward the model for paying **more** fees: a larger fee shrinks the leftover. The budget instead states that the fee must be affordable. The objective still pushes ``\\boldsymbol{x}^\\intercal \\boldsymbol{p}`` up, so every unit of fee competes with a unit of position, and the model drops a position whose fixed fee buys too little tracking.

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
    set_allocation_fees!(model::JuMP.Model, p::VecNum, cash::Number, sf::Option{<:NamedTuple})

Write one side's fee in the allocation model's own variables, and return it.

A fee is a cost of the portfolio the allocator actually buys. The model holds the share vector `x` and the prices `p`, so `x .* p` is the money in each position exactly. Every term is written against that money, and no weight and no price appears on its own. This is the rule ADR 0123 states.

`sf` is one side's charge, of [`allocation_side_fees`](@ref). A `nothing` `sf` writes nothing and returns a zero expression, so the caller needs no branch.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{m} &= \\boldsymbol{x} \\odot \\boldsymbol{p}\\,, \\\\
t_{i} &\\geq \\lvert m_{i} - m_{0,i} \\rvert\\,, \\\\
b_{i} &\\leq x_{i} \\leq \\left\\lfloor C / p_{i} \\right\\rfloor b_{i}\\,, \\\\
F(\\boldsymbol{x}) &= T \\left( \\boldsymbol{f}_{\\text{p}}^\\intercal \\boldsymbol{m} + \\boldsymbol{f}_{\\text{Tn}}^\\intercal \\boldsymbol{t} \\right) + \\boldsymbol{f}_{\\text{f}}^\\intercal \\boldsymbol{b} + F_{\\text{lq}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{m}``: Money in each position.
  - ``\\boldsymbol{m}_{0}``: Money in each position before the trade, `sf.prev_money`.
  - ``\\boldsymbol{t}``: Epigraph of the money traded.
  - ``\\boldsymbol{b}``: Binary saying whether the position is held at all.
  - ``C``: Cash allocated to this sub-problem.
  - ``T``: Horizon, in periods.
  - ``\\boldsymbol{f}_{\\text{p}},\\, \\boldsymbol{f}_{\\text{Tn}},\\, \\boldsymbol{f}_{\\text{f}}``: Proportional, turnover and fixed rates of this side.
  - ``F_{\\text{lq}}``: Forced exit of [`allocation_liquidation_fee`](@ref), `sf.liq`. It is a constant, because the assets it sells are not among the share counts this model solves for.

The rates `l`, `s` and `tn` charge on each of the `T` periods, and the fixed amounts `fl` and `fs` charge one time for the whole horizon. That is the rule [`calc_total_fees`](@ref) states, written in the model's own variables.

A binary is emitted only when the side states a fixed fee, so a problem that states none keeps the variable count it had. `x` is integer and non-negative, so `b <= x` and `x <= ub * b` make `b` the indicator of `x > 0` exactly.

# Arguments

  - $(arg_dict[:model])
  - `p::VecNum`: Asset prices of this side.
  - `cash::Number`: Cash allocated to this side. It bounds the shares a binary can switch on.
  - `sf::Option{<:NamedTuple}`: This side's charge, or `nothing`.

# Returns

  - `fee`: The fee expression. It is registered as `model[:fee]`.

# Related

  - [`allocation_liquidation_fee`](@ref)
  - [`allocation_side_fees`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`Fees`](@ref)
"""
function set_allocation_fees!(model::JuMP.Model, ::VecNum, ::Number, ::Nothing)
    return model[:fee] = zero(JuMP.AffExpr)
end
function set_allocation_fees!(model::JuMP.Model, p::VecNum, cash::Number, sf::NamedTuple)
    x = model[:x]
    sc = get_constraint_scale(model)
    N = length(p)
    T = sf.T
    # The forced exit of `allocation_liquidation_fee` is a constant: the assets it sells are
    # not among the share counts this model solves for, so it needs no variable. It still
    # enters the budget, which is the whole point of charging it here.
    fee = zero(JuMP.AffExpr) + sf.liq
    # The money in each position, exactly. No weight and no price appears on its own.
    JuMP.@expression(model, money, x .* p)
    prop = sf.prop
    if !isnothing(prop)
        # Per period: the proportional rate of this side.
        JuMP.@expression(model, fee_prop, dot_scalar(prop, money))
        JuMP.add_to_expression!(fee, T, fee_prop)
    end
    tn_val = sf.tn_val
    if !isnothing(tn_val)
        # Per period: the turnover. An absolute value needs an epigraph.
        prev_money = sf.prev_money
        JuMP.@variable(model, t_ftn[1:N] >= 0)
        JuMP.@constraints(model, begin
                              cftn_ub, sc * (money .- prev_money .- t_ftn) .<= 0
                              cftn_lb, sc * (prev_money .- money .- t_ftn) .<= 0
                          end)
        JuMP.@expression(model, fee_tn, dot_scalar(tn_val, t_ftn))
        JuMP.add_to_expression!(fee, T, fee_tn)
    end
    fixed = sf.fixed
    if !isnothing(fixed)
        # One time: a fixed fee is charged per position held, so it needs a bit saying
        # whether the position is there at all. `ub` is the most shares this cash can buy.
        ub = floor.(cash ./ p)
        JuMP.@variable(model, b[1:N], Bin)
        JuMP.@constraints(model, begin
                              cb_ub, sc * (x .- ub .* b) .<= 0
                              cb_lb, sc * (b .- x) .<= 0
                          end)
        JuMP.@expression(model, fee_fixed, dot_scalar(fixed, b))
        JuMP.add_to_expression!(fee, fee_fixed)
    end
    return model[:fee] = fee
end
"""
    finite_sub_allocation(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                          sf::Option{<:NamedTuple}, da::DiscreteAllocation,
                          str_names::Bool = false)

Build and solve the discrete allocation MIP for one side, long or short, of the portfolio.

Implements the sub-problem of [`DiscreteAllocation`](@ref). An empty `w` returns three empty vectors, the untouched `cash`, a zero fee, and `nothing` for both the return code and the model.

# Arguments

  - `w::VecNum`: Target weights of this side, non-negative.
  - `p::VecNum`: Asset prices of this side, in the same order as `w`.
  - `cash::Number`: Cash allocated to this side.
  - `bgt::Number`: Budget of this side, used to rescale the realised weights.
  - `sf::Option{<:NamedTuple}`: This side's charge, of [`allocation_side_fees`](@ref), or `nothing`.
  - `da::DiscreteAllocation`: Allocator carrying the solvers, the scales and the formulation `wf`.
  - `str_names::Bool = false`: Whether to give the JuMP variables string names.

# Returns

  - `shares::VecNum`: Share count per asset, rounded to `Int`.
  - `cost::VecNum`: `shares .* p`.
  - `aw::VecNum`: Realised weights, rescaled to sum to `bgt`. All zero when nothing was bought.
  - `acash::Number`: Cash left over, `cash` less the cost of the shares and the fee.
  - `fee::Number`: The fee this side paid over the whole horizon.
  - `res::OptimisationReturnCode`: An [`OptimisationSuccess`](@ref) or an [`OptimisationFailure`](@ref) carrying the solver trials.
  - `model::JuMP.Model`: The solved model.

# Details

  - The share vector is declared integer and non-negative, so a short side must be passed with its weights already negated.
  - [`set_discrete_error!`](@ref) adds the one constraint that `da.wf` selects. Everything else in the model is common to the four formulations.
  - [`set_allocation_fees!`](@ref) adds the fee. It enters the budget constraint and never the objective.
  - The fee this verb **reports** is [`allocation_fee`](@ref) of the realised shares, not the value of the model's own expression. The turnover term of that expression is an epigraph, and only the budget pushes it down, so a solver leaves it slack whenever the budget does not bind. A slack epigraph overstates the fee, so the allocation the model bought is affordable under the exact charge.
  - `shares` is read back with `round(Int, ...)`, because a MIP solver returns an integer only to within its own tolerance. A fee larger than the cash makes the budget infeasible, and an infeasible model holds no finite value, so the book is then read as empty and `res` carries the failure.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`set_discrete_error!`](@ref)
  - [`set_allocation_fees!`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`adjust_long_cash`](@ref)
"""
function finite_sub_allocation(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                               sf::Option{<:NamedTuple}, da::DiscreteAllocation,
                               str_names::Bool = false)
    if isempty(w)
        # An empty side buys nothing, but it can still owe the forced exit: a delisted asset
        # carries a zero target weight, so it lands on the long side even when that side
        # holds nothing else. `allocation_fee` of the empty book is that charge exactly.
        fee = allocation_fee(sf, p, w)
        return Vector{eltype(w)}(undef, 0), Vector{eltype(w)}(undef, 0),
               Vector{eltype(w)}(undef, 0), cash - fee, fee, nothing, nothing
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
    fee = set_allocation_fees!(model, p, cash, sf)
    JuMP.@constraint(model, cr, sc * (r - fee) >= 0)
    set_discrete_error!(model, w, p, cash, da.wf)
    JuMP.@objective(model, Min, so * (u + r))
    res = optimise_JuMP_model!(model, da.slv)
    res = if res.success
        OptimisationSuccess(; res = res.trials)
    else
        OptimisationFailure(; res = res.trials)
    end
    xv = JuMP.value.(x)
    # A fee larger than the cash makes the budget constraint infeasible, and an infeasible
    # model holds no solution, so its values are not finite. `res` carries the failure, and
    # the book is read as empty rather than raising on the conversion to `Int`.
    shares = all(isfinite, xv) ? round.(Int, xv) : zeros(Int, N)
    cost = shares .* p
    aw = if any(!iszero, cost)
        cost / sum(cost) * bgt
    else
        range(zero(eltype(w)), zero(eltype(w)); length = N)
    end
    # The reported fee is priced on the realised integer shares, not read off the model.
    # `t_ftn` is an epigraph, and only the budget constraint pushes it down, so the solver
    # leaves it slack whenever the budget does not bind. A slack epigraph only overstates
    # the fee, so the allocation it bought stays affordable.
    afee = allocation_fee(sf, p, shares)
    acash = cash - sum(cost) - afee
    return shares, cost, aw, acash, afee, res, model
end
function _optimise(da::DiscreteAllocation, fai::FiniteAllocationInput;
                   str_names::Bool = false, save::Bool = true, kwargs...)
    w, p, cash, pcash, T, fees = fai.w, fai.prices, fai.cash, fai.prev_cash, fai.horizon,
                                 fai.fees
    bgt, lbgt, sbgt, lidx, sidx, lcash, scash = setup_alloc_optim(w, cash)
    lsf, ssf = allocation_side_fees(fees, fai.imsk, T, pcash, lidx, sidx)
    sshares, scost, sw, scash, sfee, sretcode, smodel = finite_sub_allocation(-view(w,
                                                                                    sidx),
                                                                              view(p, sidx),
                                                                              scash, sbgt,
                                                                              ssf, da,
                                                                              str_names)
    lcash = adjust_long_cash(bgt, lcash, scash)
    lshares, lcost, lw, lcash, lfee, lretcode, lmodel = finite_sub_allocation(view(w, lidx),
                                                                              view(p, lidx),
                                                                              lcash, lbgt,
                                                                              lsf, da,
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
                                    cash = lcash, fees = lfee + sfee,
                                    s_model = ifelse(save, smodel, nothing),
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
