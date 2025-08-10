function entropy_pooling(w::AbstractVector, epcs::LinearConstraint,
                         optim::OptimEntropyPooling)
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
function entropy_pooling(w::AbstractVector, epcs::LinearConstraint,
                         optim::JuMPEntropyPooling)
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
                     [sc * t; fill(sc, S); sc * q] in MOI.RelativeEntropyCone(2 * S + 1)
                 end)
    @objective(model, Min, so * (t - dot(q, log_p)))
    # Solve the optimization problem
    return if optimise_JuMP_model!(model, slv).success
        pweights(value.(q))
    else
        pweights(fill(NaN, length(q)))
    end
end
function entropy_pooling(w::AbstractVector, epcs::Union{Nothing, <:LinearConstraint},
                         optim::Union{<:OptimEntropyPooling, <:JuMPEntropyPooling},
                         ::Nothing, ::Nothing, ::Any, ::Any)
    return entropy_pooling(w, epcs, optim)
end
function entropy_pooling(w::AbstractVector, epcs::Union{Nothing, <:LinearConstraint},
                         optim::Union{<:OptimEntropyPooling, <:JuMPEntropyPooling},
                         d_epcs::Union{Nothing, <:LinearConstraint},
                         d_views::Union{<:DiscontinuousEntropyPoolingViewEstimator,
                                        <:AbstractVector{<:DiscontinuousEntropyPoolingViewEstimator}},
                         ds_opt::Union{Nothing, <:OptimEntropyPooling},
                         dm_opt::Union{Nothing, <:OptimEntropyPooling})
    N = length(d_views)
    VN = Val(N)
    alpha, d_opt = if N == 1
        d_views.A.alpha, !isnothing(ds_opt) ? ds_opt : OptimEntropyPooling()
    else
        getproperty.(getproperty.(d_views, :A), :alpha),
        !isnothing(dm_opt) ? dm_opt : OptimEntropyPooling()
    end
    A = d_epcs.A_eq
    B = d_epcs.B_eq
    scale = d_opt.scale
    N = length(B)
    ialpha = inv(alpha)
    function g(x)
        @argcheck(all(zero(eltype(x)) .< x .<= B))
        pos_part = max.(-A .- x, zero(eltype(x)))
        A_eq = vcat(epcs.A_eq, pos_part * ialpha)
        B_eq = vcat(epcs.B_eq, B .- x)
        epcs2 = LinearConstraint(; ineq = epcs.ineq,
                                 eq = PartialLinearConstraint(; A = A_eq, B = B_eq))
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
struct EntropyPoolingPriorEstimator{T1, T2, T3, T4, T5, T6, T7, T8, T9} <:
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
                                      pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPrior(),
                                      views::Union{<:ContinuousEntropyPoolingViewEstimator,
                                                   <:AbstractVector{<:ContinuousEntropyPoolingViewEstimator}},
                                      d_views::Union{Nothing,
                                                     <:DiscontinuousEntropyPoolingViewEstimator,
                                                     <:AbstractVector{<:DiscontinuousEntropyPoolingViewEstimator}} = nothing,
                                      sets::Union{<:AssetSets,
                                                  #! Start: to delete
                                                  <:DataFrame
                                                  #! End: to delete
                                                  } = DataFrame(),
                                      opt::AbstractEntropyPoolingOptimiser = OptimEntropyPooling(),
                                      d_opt1::Union{Nothing, <:OptimEntropyPooling} = nothing,
                                      d_opt2::Union{Nothing, <:OptimEntropyPooling} = nothing,
                                      w::Union{Nothing, <:AbstractWeights} = nothing,
                                      alg::AbstractEntropyPoolingAlgorithm = H0_EntropyPooling())
    if isa(views, AbstractVector)
        @argcheck(!isempty(views))
    end
    if !isnothing(d_views)
        if isa(d_views, AbstractVector)
            @argcheck(!isempty(d_views))
        end
    else
        @argcheck(isnothing(d_opt1) && isnothing(d_opt2))
    end
    if isa(w, AbstractWeights)
        @argcheck(!isempty(w))
    end
    return EntropyPoolingPriorEstimator(pe, views, d_views, sets, opt, d_opt1, d_opt2, w,
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
    @argcheck(dims in (1, 2))
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
    @argcheck(length(w) == T)
    @argcheck(nrow(pe.sets) == N)
    pe = factory(pe, w)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    d_V = entropy_pooling_views(pr, pe.d_views, pe.sets; strict = strict)
    views = entropy_pooling_views(pr, pe.views, pe.sets; strict = strict)
    w = entropy_pooling(w, views, pe.opt, d_V, pe.d_views, pe.d_opt1, pe.d_opt2)
    pe = factory(pe, w)
    (; X, mu, sigma, chol, loadings, f_mu, f_sigma) = prior(pe.pe, X, F; strict = strict,
                                                            kwargs...)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w,
                         loadings = loadings, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = !isnothing(loadings) ? w : nothing)
end
function prior(pe::EntropyPoolingPriorEstimator{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any,
                                                <:Union{<:H1_EntropyPooling,
                                                        <:H2_EntropyPooling}},
               X::AbstractMatrix, F::Union{Nothing, <:AbstractMatrix} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
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
    @argcheck(length(w0) == T)
    @argcheck(nrow(pe.sets) == N)
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
    for uvl in uvls
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
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = wi,
                         loadings = loadings, f_mu = f_mu, f_sigma = f_sigma, f_w = wi)
end

export H0_EntropyPooling, H1_EntropyPooling, H2_EntropyPooling, OptimEntropyPooling,
       JuMPEntropyPooling, entropy_pooling, EntropyPoolingPriorEstimator,
       effective_number_scenarios
