abstract type AbstractEntropyPoolingEstimator <: AbstractEstimator end
abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct OptimEntropyPoolingEstimator{T1 <: Tuple, T2 <: NamedTuple, T3 <: Real} <:
       AbstractEntropyPoolingEstimator
    args::T1
    kwargs::T2
    scale::T3
end
function OptimEntropyPoolingEstimator(; args::Tuple = (), kwargs::NamedTuple = (;),
                                      scale::Real = 1)
    @smart_assert(scale >= zero(scale))
    return OptimEntropyPoolingEstimator{typeof(args), typeof(kwargs), typeof(scale)}(args,
                                                                                     kwargs,
                                                                                     scale)
end
struct JuMPEntropyPoolingEstimator{T1 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                                   T2 <: Real, T3 <: Real} <:
       AbstractEntropyPoolingEstimator
    slv::T1
    sc::T2
    so::T3
end
function JuMPEntropyPoolingEstimator(; slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                                     sc::Real = 1, so::Real = 1,)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(sc >= zero(sc))
    @smart_assert(so >= zero(so))
    return JuMPEntropyPoolingEstimator{typeof(slv), typeof(sc), typeof(so)}(slv, sc, so)
end
function entropy_pooling(w::AbstractVector, ::Nothing, ::Any)
    return w
end
function entropy_pooling(w::AbstractVector, epcs::LinearConstraintResult,
                         optim::OptimEntropyPoolingEstimator)
    T = length(w)
    factor = inv(sqrt(T))
    A_sum = fill(factor, 1, T)
    B_sum = [factor]
    (; A_eq, B_eq, A_ineq, B_ineq) = epcs
    lhs, rhs, lb, ub = if !isnothing(A_eq) && !isnothing(B_eq)
        A_eq = vcat(A_sum, A_eq)
        B_eq = vcat(B_sum, B_eq)
        A_eq, B_eq, fill(typemin(eltype(B_eq)), length(B_eq)),
        fill(typemax(eltype(B_eq)), length(B_eq))
    else
        A_sum, B_sum, [typemin(eltype(w))], [typemax(eltype(w))]
    end
    if !isnothing(A_ineq) && !isnothing(B_ineq)
        lhs = vcat(lhs, A_ineq)
        rhs = vcat(rhs, B_ineq)
        lb = vcat(lb, fill(zero(eltype(B_ineq)), length(B_ineq)))
        ub = vcat(ub, fill(typemax(eltype(B_ineq)), length(B_ineq)))
    end
    log_p = log.(w)
    x0 = fill(inv(T), size(lhs, 1))
    G = similar(x0)
    last_x = similar(x0)
    grad = similar(G)
    log_x = similar(log_p)
    y = similar(log_p)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            log_x .= log_p - (one(eltype(log_p)) .+ transpose(lhs) * x)
            y .= exp.(log_x)
            grad .= rhs - lhs * y
        end
    end
    function f(x)
        common_op(x)
        return optim.scale * (dot(x, grad) - dot(y, log_x - log_p))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return optim.scale * G
    end
    result = Optim.optimize(f, g!, lb, ub, x0, optim.args...; optim.kwargs...)
    x = if Optim.converged(result)
        # Compute posterior probabilities
        Optim.minimizer(result)
    else
        throw(ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    end
    return pweights(exp.(log_p - (one(eltype(log_p)) .+ transpose(lhs) * x)))
end
function entropy_pooling(w::AbstractVector, epcs::LinearConstraintResult,
                         optim::JuMPEntropyPoolingEstimator)
    model = Model()
    S = length(w)
    log_p = log.(w)
    # Decision variables (posterior probabilities)
    @variables(model, begin
                   q[1:S]
                   t
               end)
    (; A_eq, B_eq, A_ineq, B_ineq) = epcs
    (; sc, so, slv) = optim
    # Equality constraints from A_ineq and B_ineq if provided
    if !isnothing(A_eq) && !isnothing(B_eq)
        @constraint(model, ceq, sc * (A_eq * q - B_eq) == 0)
    end
    # Inequality constraints from A_ineq and B_ineq if provided
    if !isnothing(A_ineq) && !isnothing(B_ineq)
        @constraint(model, cineq, sc * (A_ineq * q - B_ineq) <= 0)
    end
    # Equality constraints from A_eq and B_eq and probabilities equal to 1
    @constraints(model,
                 begin
                     sc * (sum(q) - one(eltype(w))) == 0
                     [sc * t; fill(sc, S); sc * q] ∈ MOI.RelativeEntropyCone(2 * S + 1)
                 end)
    @objective(model, Min, so * (t - dot(q, log_p)))
    # Solve the optimization problem
    return if optimise_JuMP_model!(model, slv).success
        pweights(value.(q))
    else
        pweights(fill(NaN, length(q)))
    end
end
function entropy_pooling(w::AbstractVector, epcs::Union{Nothing, <:LinearConstraintResult},
                         optim::Union{<:OptimEntropyPoolingEstimator,
                                      <:JuMPEntropyPoolingEstimator}, ::Nothing, ::Nothing,
                         ::Any, ::Any)
    return entropy_pooling(w, epcs, optim)
end
function entropy_pooling(w::AbstractVector, epcs::Union{Nothing, <:LinearConstraintResult},
                         optim::Union{<:OptimEntropyPoolingEstimator,
                                      <:JuMPEntropyPoolingEstimator},
                         d_epcs::Union{Nothing, <:LinearConstraintResult},
                         d_views::Union{<:DiscontinuousEntropyPoolingViewEstimator,
                                        <:AbstractVector{<:DiscontinuousEntropyPoolingViewEstimator}},
                         d_opt1::Union{Nothing, <:OptimEntropyPoolingEstimator},
                         d_opt2::Union{Nothing, <:OptimEntropyPoolingEstimator})
    N = length(d_views)
    VN = Val(N)
    alpha, d_opt = if N == 1
        d_views.A.alpha, !isnothing(d_opt1) ? d_opt1 : OptimEntropyPoolingEstimator()
    else
        getproperty.(getproperty.(d_views, Ref(:A)), Ref(:alpha)),
        !isnothing(d_opt2) ? d_opt2 : OptimEntropyPoolingEstimator()
    end
    A = d_epcs.A_eq
    B = d_epcs.B_eq
    scale = d_opt.scale
    N = length(B)
    ialpha = inv(alpha)
    function g(x)
        @smart_assert(all(zero(eltype(x)) .< x .<= B))
        pos_part = max.(-A .- Ref(x), eltype(x))
        A_eq = vcat(epcs.A_eq, pos_part * ialpha)
        B_eq = vcat(epcs.B_eq, B .- Ref(x))
        epcs2 = LinearConstraintResult(; ineq = epcs.ineq,
                                       eq = PartialLinearConstraintResult(; A = A_eq,
                                                                          B = B_eq))
        return pos_part, entropy_pooling(w, epcs2, optim)
    end
    function h(::Val{1}, _A, _B, _w, _pos_part)
        return scale * (sum(_w[.!iszero.(_pos_part)]) - alpha)
    end
    function h(::Any, _A, _B, _w, _pos_part)
        return scale * norm(cvar(transpose(_A), alpha, _w) - _B) / sqrt(N)
    end
    function f(x)
        pos_part, w = g(x)
        return h(VN, A, B, w, pos_part)
    end
    result = if N == 1
        Optim.optimize(f, zero(eltype(B)), B[1], d_opt.args...; d_opt.kwargs...)
    else
        Optim.optimize(f, zeros(N), B, 0.5 * B, d_opt.args...; d_opt.kwargs...)
    end
    x = if Optim.converged(result)
        # Compute posterior probabilities
        Optim.minimizer(result)
    else
        throw(ErrorException("CVaR entropy pooling optimisation failed. Relax the view, incrase alpha, use different solver parameters, or use a different prior."))
    end
    return g(x)
end
function effective_number_scenarios(x::AbstractVector, y::AbstractVector)
    return exp(-kldivergence(x, y))
end
struct EntropyPoolingPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_1o2_1o2,
                                    T2 <: Union{<:ContinuousEntropyPoolingViewEstimator,
                                                <:AbstractVector{<:ContinuousEntropyPoolingViewEstimator}},
                                    T3 <: Union{Nothing,
                                                <:DiscontinuousEntropyPoolingViewEstimator,
                                                <:AbstractVector{<:DiscontinuousEntropyPoolingViewEstimator}},
                                    T4 <: DataFrame, T5 <: AbstractEntropyPoolingEstimator,
                                    T6 <: Union{Nothing, <:OptimEntropyPoolingEstimator},
                                    T7 <: Union{Nothing, <:OptimEntropyPoolingEstimator},
                                    T8 <: Union{Nothing, <:AbstractVector},
                                    T9 <: AbstractEntropyPoolingAlgorithm} <:
       AbstractLowOrderPriorEstimator_1o2_1o2
    pe::T1
    views::T2
    d_views::T3
    sets::T4
    opt::T5
    d_opt1::T6
    d_opt2::T7
    w::T8
    alg::T9
end
function EntropyPoolingPriorEstimator(;
                                      pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                                      views::Union{<:ContinuousEntropyPoolingViewEstimator,
                                                   <:AbstractVector{<:ContinuousEntropyPoolingViewEstimator}},
                                      d_views::Union{Nothing,
                                                     <:DiscontinuousEntropyPoolingViewEstimator,
                                                     <:AbstractVector{<:DiscontinuousEntropyPoolingViewEstimator}} = nothing,
                                      sets::DataFrame = DataFrame(),
                                      opt::AbstractEntropyPoolingEstimator = OptimEntropyPoolingEstimator(),
                                      d_opt1::Union{Nothing,
                                                    <:OptimEntropyPoolingEstimator} = nothing,
                                      d_opt2::Union{Nothing,
                                                    <:OptimEntropyPoolingEstimator} = nothing,
                                      w::Union{Nothing, <:AbstractWeights} = nothing,
                                      alg::AbstractEntropyPoolingAlgorithm = H0_EntropyPooling())
    if isa(views, AbstractVector)
        @smart_assert(!isempty(views))
    end
    if !isnothing(d_views)
        if isa(d_views, AbstractVector)
            @smart_assert(!isempty(d_views))
        end
    else
        @smart_assert(isnothing(d_opt1) && isnothing(d_opt2))
    end
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return EntropyPoolingPriorEstimator{typeof(pe), typeof(views), typeof(d_views),
                                        typeof(sets), typeof(opt), typeof(d_opt1),
                                        typeof(d_opt2), typeof(w), typeof(alg)}(pe, views,
                                                                                d_views,
                                                                                sets, opt,
                                                                                d_opt1,
                                                                                d_opt2, w,
                                                                                alg)
end
function Base.getproperty(obj::EntropyPoolingPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function factory(pe::EntropyPoolingPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return EntropyPoolingPriorEstimator(; pe = factory(pe.pe, w), views = pe.views,
                                        sets = pe.sets, alg = pe.alg, opt = pe.opt,
                                        w = nothing_scalar_array_factory(pe.w, w))
end
function prior(pe::EntropyPoolingPriorEstimator{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any, <:H0_EntropyPooling},
               X::AbstractMatrix, F::Union{Nothing, <:AbstractMatrix} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    T, N = size(X)
    w = if isnothing(pe.w)
        pweights(range(; start = inv(T), stop = inv(T), length = T))
    else
        pweights(pe.w)
    end
    @smart_assert(length(w) == T)
    @smart_assert(nrow(pe.sets) == N)
    pe = factory(pe, w)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    d_V = entropy_pooling_views(pr, pe.d_views, pe.sets; strict = strict)
    views = entropy_pooling_views(pr, pe.views, pe.sets; strict = strict)
    w = entropy_pooling(w, views, pe.opt, d_V, pe.d_views, pe.d_opt1, pe.d_opt2)
    pe = factory(pe, w)
    (; X, mu, sigma, chol, loadings, f_mu, f_sigma) = prior(pe.pe, X, F; strict = strict,
                                                            kwargs...)
    return LowOrderPriorResult(; X = X, mu = mu, sigma = sigma, chol = chol, w = w,
                               loadings = loadings, f_mu = f_mu, f_sigma = f_sigma,
                               f_w = !isnothing(loadings) ? w : nothing)
end
function _get_epw(::H1_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return w0
end
function _get_epw(::H2_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return wi
end
function prior(pe::EntropyPoolingPriorEstimator{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any,
                                                <:Union{<:H1_EntropyPooling,
                                                        <:H2_EntropyPooling}},
               X::AbstractMatrix, F::Union{Nothing, <:AbstractMatrix} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    T, N = size(X)
    wi = w0 = if isnothing(pe.w)
        pweights(range(; start = inv(T), stop = inv(T), length = T))
    else
        pweights(pe.w)
    end
    @smart_assert(length(w0) == T)
    @smart_assert(nrow(pe.sets) == N)
    views = sort(pe.views)
    vls = get_view_level.(views)
    uvls = sort!(unique(vls))
    idx = falses(length(views))
    excluded = Int[]
    included = Int[]
    cache = Set{Int}()
    # Compute the prior observations.
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    d_V = entropy_pooling_views(pr, pe.d_views, pe.sets; strict = strict)
    for uvl ∈ uvls
        # Freeze updated parameters, ie parameters which were free in the previous iteration plus the ones which were adjusted via views. If there were no views the previous iteration, the free parameters become the new updated parameters and therefore frozen. In the first iteration, there is nothing to freeze.
        constant_entropy_pooling_constraint!(pr, cache, [excluded;
                                                         included], views, pe.sets;
                                             strict = strict, w = wi)

        V_idx = vls .== uvl
        idx = idx .|| V_idx
        # Excluded contains all free parameters for this iteration. Meaning they are unrestricted and will be secondarily affected by the views.
        excluded = findall(V_idx)[to_be_frozen.(views[V_idx])]
        # Inlcuded contains the parameters to be updated, as well as the frozen parameters this iteration.
        included = setdiff(findall(idx), excluded)
        if isempty(included)
            continue
        end
        # Views of the parameters to be updated.
        v = views[included]
        # Equality and inequality constraints for the views.
        V_i = entropy_pooling_views(pr, v, pe.sets; w = w0, strict = strict)
        # Compute the posterior observations.
        wi = entropy_pooling(_get_epw(pe.alg, w0, wi), V_i, pe.opt, d_V, pe.d_views,
                             pe.d_opt1, pe.d_opt2)
        pe = factory(pe, wi)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    (; X, mu, sigma, chol, loadings, f_mu, f_sigma) = pr
    return LowOrderPriorResult(; X = X, mu = mu, sigma = sigma, chol = chol, w = wi,
                               loadings = loadings, f_mu = f_mu, f_sigma = f_sigma,
                               f_w = wi)
end

export H0_EntropyPooling, H1_EntropyPooling, H2_EntropyPooling,
       OptimEntropyPoolingEstimator, JuMPEntropyPoolingEstimator, entropy_pooling,
       EntropyPoolingPriorEstimator, effective_number_scenarios
