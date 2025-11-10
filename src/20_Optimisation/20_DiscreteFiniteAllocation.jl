struct DiscreteAllocationOptimisation{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11} <:
       OptimisationResult
    oe::T1
    shares::T2
    cost::T3
    w::T4
    retcode::T5
    s_retcode::T6
    l_retcode::T7
    s_model::T8
    l_model::T9
    cash::T10
    fb::T11
end
function opt_attempt_factory(res::DiscreteAllocationOptimisation, fb)
    return DiscreteAllocationOptimisation(res.oe, res.shares, res.cost, res.w, res.retcode,
                                          res.s_retcode, res.l_retcode, res.s_model,
                                          res.l_model, res.cash, fb)
end
struct DiscreteAllocation{T1, T2, T3, T4} <: FiniteAllocationOptimisationEstimator
    slv::T1
    sc::T2
    so::T3
    fb::T4
    function DiscreteAllocation(slv::Slv_VecSlv, sc::Number, so::Number,
                                fb::Option{<:FiniteAllocationOptimisationEstimator})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(sc > zero(sc))
        @argcheck(so > zero(so))
        return new{typeof(slv), typeof(sc), typeof(so), typeof(fb)}(slv, sc, so, fb)
    end
end
function DiscreteAllocation(; slv::Slv_VecSlv, sc::Number = 1, so::Number = 1,
                            fb::Option{<:FiniteAllocationOptimisationEstimator} = GreedyAllocation())
    return DiscreteAllocation(slv, sc, so, fb)
end
function finite_sub_allocation(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                               da::DiscreteAllocation, str_names::Bool = false)
    if isempty(w)
        return Vector{eltype(w)}(undef, 0), Vector{eltype(w)}(undef, 0),
               Vector{eltype(w)}(undef, 0), cash, nothing, nothing
    end
    sc = da.sc
    so = da.so
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    N = length(w)
    # Integer allocation
    # x := number of shares
    # u := bounding variable
    @variables(model, begin
                   x[1:N] >= 0, Int
                   u
               end)
    # r := remaining money
    # eta := ideal_investment - discrete_investment
    @expressions(model, begin
                     r, cash - dot(x, p)
                     eta, w * cash - x .* p
                 end)
    @constraints(model, begin
                     sc * r >= 0
                     [sc * u; sc * eta] in MOI.NormOneCone(N + 1)
                 end)
    @objective(model, Min, so * (u + r))
    res = optimise_JuMP_model!(model, da.slv)
    res = if res.success
        OptimisationSuccess(; res = res.trials)
    else
        OptimisationFailure(; res = res.trials)
    end
    shares = round.(Int, value.(x))
    cost = shares .* p
    aw = if any(!iszero, cost)
        cost / sum(cost) * bgt
    else
        range(zero(eltype(w)), zero(eltype(w)); length = N)
    end
    acash = value(r)
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
    return DiscreteAllocationOptimisation(typeof(da), view(res, :, 1), view(res, :, 2),
                                          view(res, :, 3), retcode, sretcode, lretcode,
                                          ifelse(save, smodel, nothing),
                                          ifelse(save, lmodel, nothing), lcash, nothing)
end
function optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, Nothing}, w::VecNum,
                  p::VecNum, cash::Number = 1e6, T::Option{<:Number} = nothing,
                  fees::Option{<:Fees} = nothing; str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(da, w, p, cash, T, fees; str_names = str_names, save = save, kwargs...)
end

export DiscreteAllocationOptimisation, DiscreteAllocation
