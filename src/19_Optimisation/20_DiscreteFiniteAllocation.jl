struct DiscreteAllocationOptimisation{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
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
struct DiscreteAllocation{T1, T2, T3, T4} <: BaseFiniteAllocationOptimisationEstimator
    slv::T1
    sc::T2
    so::T3
    fallback::T4
end
function DiscreteAllocation(; slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                            sc::Real = 1, so::Real = 1,
                            fallback::BaseFiniteAllocationOptimisationEstimator = GreedyAllocation())
    if isa(slv, AbstractVector)
        @argcheck(!isempty(slv))
    end
    @argcheck(sc > zero(sc))
    @argcheck(so > zero(so))
    return DiscreteAllocation(slv, sc, so, fallback)
end
function finite_sub_allocation(w::AbstractVector, p::AbstractVector, cash::Real, bgt::Real,
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
        res = finite_sub_allocation!(w, p, cash, bgt, da.fallback, str_names)
        if length(res) == 4
            res = (res..., OptimisationFailure(; res = res.trials), nothing)
        end
        res
    end
end
function optimise!(da::DiscreteAllocation, w::AbstractVector, p::AbstractVector,
                   cash::Real = 1e6, T::Union{Nothing, <:Real} = nothing,
                   fees::Union{Nothing, <:Fees} = nothing; str_names::Bool = false,
                   save::Bool = true, kwargs...)
    @argcheck(!isempty(w) && !isempty(p) && length(w) == length(p))
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
        OptimisationFailure(nothing)
    else
        OptimisationSuccess(nothing)
    end
    return DiscreteAllocationOptimisation(typeof(da), view(res, :, 1), view(res, :, 2),
                                          view(res, :, 3), retcode, sretcode, lretcode,
                                          ifelse(save, smodel, nothing),
                                          ifelse(save, lmodel, nothing), lcash)
end

export DiscreteAllocationOptimisation, DiscreteAllocation
