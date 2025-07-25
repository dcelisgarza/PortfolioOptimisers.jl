struct DiscreteAllocationResult{T1, T2 <: AbstractVector, T3 <: AbstractVector,
                                T4 <: AbstractVector, T5 <: OptimisationReturnCode,
                                T6 <: Union{Nothing, <:OptimisationReturnCode},
                                T7 <: Union{Nothing, <:OptimisationReturnCode},
                                T8 <: Union{Nothing, <:JuMP.Model},
                                T9 <: Union{Nothing, <:JuMP.Model}, T10 <: Real} <:
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
end
struct DiscreteAllocation{T1 <: Union{<:Solver, <:AbstractVector{<:Solver}}, T2 <: Real,
                          T3 <: Real} <: BaseFiniteAllocationOptimisationEstimator
    slv::T1
    sc::T2
    so::T3
end
function DiscreteAllocation(; slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                            sc::Real = 1, so::Real = 1)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(sc > zero(sc))
    @smart_assert(so > zero(so))
    return DiscreteAllocation{typeof(slv), typeof(sc), typeof(so)}(slv, sc, so)
end
function discrete_sub_allocation!(w::AbstractVector, p::AbstractVector, cash::Real,
                                  bgt::Real, da::DiscreteAllocation,
                                  str_names::Bool = false)
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
    return if res.success
        shares = round.(Int, value.(x))
        cost = shares .* p
        aw = if any(!iszero, cost)
            cost / sum(cost) * bgt
        else
            range(; start = 0, stop = 0, length = N)
        end
        acash = value(r)
        shares, cost, aw, acash, OptimisationSuccess(; res = res.trials), model
    else
        @warn("Discrete allocation failed. Reverting to greedy allocation.")
        greedy_sub_allocation!(w, p, cash, bgt, GreedyAllocation()),
        OptimisationFailure(; res = res.trials), nothing
    end
end
function optimise!(da::DiscreteAllocation, w::AbstractVector, p::AbstractVector,
                   cash::Real = 1e6, T::Union{Nothing, <:Real} = nothing,
                   fees::Union{Nothing, <:Fees} = nothing; str_names::Bool = false,
                   save::Bool = true, kwargs...)
    @smart_assert(!isempty(w) && !isempty(p) && length(w) == length(p))
    @smart_assert(cash > zero(cash))
    if !isnothing(fees)
        @smart_assert(!isnothing(T))
    end
    cash, bgt, lbgt, sbgt, lidx, sidx, lcash, scash = setup_alloc_optim(w, p, cash, T, fees)
    sshares, scost, sw, scash, sretcode, smodel = discrete_sub_allocation!(-view(w, sidx),
                                                                           view(p, sidx),
                                                                           scash, sbgt, da,
                                                                           str_names)
    lcash = adjust_long_cash(bgt, lcash, scash)
    lshares, lcost, lw, lcash, lretcode, lmodel = discrete_sub_allocation!(view(w, lidx),
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
    retcode = if isnothing(sretcode) && isa(lretcode, OptimisationSuccess) ||
                 isnothing(lretcode) && isa(sretcode, OptimisationSuccess) ||
                 isa(sretcode, OptimisationSuccess) && isa(lretcode, OptimisationSuccess)
        OptimisationSuccess(nothing)
    else
        OptimisationFailure(nothing)
    end
    return DiscreteAllocationResult(typeof(da), view(res, :, 1), view(res, :, 2),
                                    view(res, :, 3), retcode, sretcode, lretcode,
                                    ifelse(save, smodel, nothing),
                                    ifelse(save, lmodel, nothing), lcash)
end

export DiscreteAllocationResult, DiscreteAllocation
