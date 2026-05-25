"""
$(DocStringExtensions.TYPEDEF)

Result type for Discrete Allocation portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`DiscreteAllocation`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct DiscreteAllocationResult <: FiniteAllocationOptimisationResult
    "$(field_dict[:oe])"
    oe
    "$(field_dict[:retcode])"
    retcode
    "$(field_dict[:s_retcode])"
    s_retcode
    "$(field_dict[:l_retcode])"
    l_retcode
    "$(field_dict[:shares])"
    shares
    "$(field_dict[:cost_alloc])"
    cost
    "Realised portfolio weights."
    w
    "$(field_dict[:cash_alloc])"
    cash
    "$(field_dict[:s_model])"
    s_model
    "$(field_dict[:l_model])"
    l_model
    "$(field_dict[:fb])"
    fb
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a [`DiscreteAllocationResult`](@ref) with an updated fallback optimiser `fb`.
"""
function factory(res::DiscreteAllocationResult, fb::Option{<:FOptE_FOpt})
    return DiscreteAllocationResult(res.oe, res.retcode, res.s_retcode, res.l_retcode,
                                    res.shares, res.cost, res.w, res.cash, res.s_model,
                                    res.l_model, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Discrete Allocation portfolio optimiser.

`DiscreteAllocation` allocates a portfolio by solving a Mixed-Integer Programming (MIP) problem to find the optimal number of shares for each asset, minimising the deviation between the target continuous weights and the realised discrete allocation.

# Mathematical definition

Default (absolute error) MIP formulation:

```math
\\begin{align}
\\underset{\\boldsymbol{x} \\in \\mathbb{Z}_{\\geq 0}^N}{\\min} \\quad & u + r\\,, \\\\
\\text{s.t.} \\quad & u \\geq \\|\\boldsymbol{w} C - \\boldsymbol{x} \\odot \\boldsymbol{p}\\|_1\\,, \\\\
& r = C - \\boldsymbol{x}^\\intercal \\boldsymbol{p} \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: Integer share vector.
  - ``u``: L1 tracking error auxiliary variable.
  - ``r``: Residual cash.
  - ``\\boldsymbol{w}``: Target weight vector.
  - ``C``: Available cash.
  - ``\\boldsymbol{p}``: Asset price vector.
  - ``\\odot``: Element-wise (Hadamard) product.
  - ``N``: Number of assets.

The weight finaliser `wf` controls which norm/error is used.

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
julia> DiscreteAllocation(; slv = Solver())
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

  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`DiscreteAllocationResult`](@ref)
"""
@concrete struct DiscreteAllocation <: FiniteAllocationOptimisationEstimator
    "$(field_dict[:slv])"
    slv
    "$(field_dict[:sc])"
    sc
    "$(field_dict[:so])"
    so
    "$(field_dict[:wf])"
    wf
    "$(field_dict[:fb])"
    fb
    function DiscreteAllocation(slv::Slv_VecSlv, sc::Number, so::Number,
                                wf::JuMPWeightFinaliserFormulation,
                                fb::Option{<:FOptE_FOpt})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(sc > zero(sc))
        @argcheck(so > zero(so))
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
    set_discrete_error!(model, w, p, cash, ...)

Add discrete allocation error constraints to the JuMP model.

Sets up the tracking error objective between target weights and the discrete allocation, subject to available cash and price constraints.

# Arguments

  - `model`: JuMP model.
  - `w`: Target portfolio weights.
  - `p`: Asset prices.
  - `cash`: Cash budget.
  - Additional parameters.

# Returns

  - `nothing`.

# Related

  - [`finite_sub_allocation`](@ref)
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
    sc = model[:sc]
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
    sc = model[:sc]
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
    sc = model[:sc]
    JuMP.@constraint(model, cabs_err,
                     [sc * u; sc * (w * cash - x .* p)] in
                     JuMP.MOI.NormOneCone(length(x) + 1))
    return nothing
end
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::SquaredAbsoluteErrorWeightFinaliser)
    x = model[:x]
    u = model[:u]
    sc = model[:sc]
    JuMP.@constraint(model, csqabs_err,
                     [sc * u;
                      sc * (w * cash - x .* p)] in JuMP.SecondOrderCone())
    return nothing
end
"""
    finite_sub_allocation(w, p, cash, bgt, ...)

Compute the finite (integer) allocation for one side (long or short) of the portfolio.

Solves a discrete allocation sub-problem using the given weights, prices, and cash budget.

# Arguments

  - `w`: Target portfolio weights.
  - `p`: Asset prices.
  - `cash`: Cash available for this side.
  - `bgt`: Budget target.
  - Additional parameters for solver configuration.

# Returns

  - Allocation vector (number of units per asset) or similar.

# Related

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
function _optimise(da::DiscreteAllocation, w::VecNum, p::VecNum, cash::Number = 1e6,
                   T::Option{<:Number} = nothing, fees::Option{<:Fees} = nothing;
                   str_names::Bool = false, save::Bool = true, kwargs...)
    @argcheck(!isempty(w))
    @argcheck(!isempty(p))
    @argcheck(length(w) == length(p))
    @argcheck(cash > zero(cash))
    if !isnothing(fees)
        @argcheck(!isnothing(T))
    end
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
        OptimisationFailure(nothing)
    else
        OptimisationSuccess(nothing)
    end
    return DiscreteAllocationResult(typeof(da), retcode, sretcode, lretcode,
                                    view(res, :, 1), view(res, :, 2), view(res, :, 3),
                                    lcash, ifelse(save, smodel, nothing),
                                    ifelse(save, lmodel, nothing), nothing)
end
"""
    optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, Nothing}, w::VecNum,
             p::VecNum, cash::Number = 1e6, T::Option{<:Number} = nothing,
             fees::Option{<:Fees} = nothing; str_names::Bool = false,
             save::Bool = true, kwargs...) -> DiscreteAllocationResult

Run the Discrete Allocation portfolio optimisation.

# Arguments

  - `da`: The discrete allocation optimiser to use.
  - $(arg_dict[:pw])
  - `p`: The prices of the assets in the same order as `w`.
  - `cash`: The initial cash balance.
  - `T`: The time horizon for the optimisation. Used to adjust the initial cash balance according to the fees charged on the portfolio for the time horizon.
  - `fees`: The fees to apply to the portfolio.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`DiscreteAllocationResult`](@ref)
"""
function optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, Nothing}, w::VecNum,
                  p::VecNum, cash::Number = 1e6, T::Option{<:Number} = nothing,
                  fees::Option{<:Fees} = nothing; str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(da, w, p, cash, T, fees; str_names = str_names, save = save, kwargs...)
end

export DiscreteAllocationResult, DiscreteAllocation
