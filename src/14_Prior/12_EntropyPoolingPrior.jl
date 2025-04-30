abstract type AbstractEntropyPoolingEstimator <: AbstractEstimator end
abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct OptimEntropyPoolingEstimator{T1 <: Real, T2 <: Tuple, T3 <: NamedTuple} <:
       AbstractEntropyPoolingEstimator
    scale::T1
    args::T2
    kwargs::T3
end
function OptimEntropyPoolingEstimator(; scale::Real = 1, args::Tuple = (),
                                      kwargs::NamedTuple = (;))
    @smart_assert(scale >= zero(scale))
    return OptimEntropyPoolingEstimator{typeof(scale), typeof(args), typeof(kwargs)}(scale,
                                                                                     args,
                                                                                     kwargs)
end
struct JuMPEntropyPoolingEstimator{T1 <: Real, T2 <: Real,
                                   T3 <: Union{<:Solver, <:AbstractVector{<:Solver}}} <:
       AbstractEntropyPoolingEstimator
    sc::T1
    so::T2
    slv::T3
end
function JuMPEntropyPoolingEstimator(; slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                                     sc::Real = 1, so::Real = 1)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(sc >= zero(sc))
    @smart_assert(so >= zero(so))
    return JuMPEntropyPoolingEstimator{typeof(sc), typeof(so), typeof(slv)}(sc, so, slv)
end
function entropy_pooling(w::AbstractVector, ::Nothing, ::Any)
    return w
end
function entropy_pooling(w::AbstractVector, epcs::LinearConstraintResult,
                         optim::OptimEntropyPoolingEstimator)
    (; A_eq, B_eq, A_ineq, B_ineq) = epcs
    lhs, rhs, lb, ub = if !isnothing(A_eq) && !isnothing(B_eq)
        A_eq = vcat(ones(eltype(A_eq), 1, size(A_eq, 2)), A_eq)
        B_eq = vcat(eltype(B_eq)[1], B_eq)
        A_eq, B_eq, fill(typemin(eltype(B_eq)), length(B_eq)),
        fill(typemax(eltype(B_eq)), length(B_eq))
    else
        ones(eltype(w), 1, length(w)), eltype(w)[1], [typemin(eltype(w))],
        [typemax(eltype(w))]
    end
    lhs_ineq, rhs_ineq, lb_ineq, ub_ineq = if !isnothing(A_ineq) && !isnothing(B_ineq)
        A_ineq, B_ineq, fill(zero(eltype(B_ineq)), length(B_ineq)),
        fill(typemax(eltype(B_ineq)), length(B_ineq))
    else
        nothing, nothing, nothing, nothing
    end
    if !isnothing(lhs_ineq)
        lhs = vcat(lhs, lhs_ineq)
        rhs = vcat(rhs, rhs_ineq)
        lb = vcat(lb, lb_ineq)
        ub = vcat(ub, ub_ineq)
    end
    log_p = log.(w)
    x0 = fill(inv(length(w)), size(lhs, 1))
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
    # Compute posterior probabilities
    x = Optim.minimizer(result)
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
function relative_entropy(x::AbstractVector, y::AbstractVector)
    @smart_assert(all(x .>= 0))
    @smart_assert(all(y .>= 0))
    @smart_assert(length(x) == length(y))
    return dot(x, log.(x) - log.(y))
end
function effective_number_scenarios(x::AbstractVector, y::AbstractVector)
    return exp(-relative_entropy(x, y))
end
struct EntropyPoolingPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_1o2_1o2,
                                    T2 <: Union{<:EntropyPoolingViewEstimator,
                                                <:AbstractVector{<:EntropyPoolingViewEstimator}},
                                    T3 <: DataFrame, T4 <: AbstractEntropyPoolingAlgorithm,
                                    T5 <: AbstractEntropyPoolingEstimator,
                                    T6 <: Union{Nothing, <:AbstractVector}} <:
       AbstractLowOrderPriorEstimator_1o2_1o2
    pe::T1
    views::T2
    sets::T3
    alg::T4
    opt::T5
    w::T6
end
function EntropyPoolingPriorEstimator(;
                                      pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                                      views::Union{<:EntropyPoolingViewEstimator,
                                                   <:AbstractVector{<:EntropyPoolingViewEstimator}},
                                      sets::DataFrame = DataFrame(),
                                      alg::AbstractEntropyPoolingAlgorithm = H0_EntropyPooling(),
                                      opt::AbstractEntropyPoolingEstimator = OptimEntropyPoolingEstimator(),
                                      w::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(views, AbstractVector)
        @smart_assert(!isempty(views))
    end
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return EntropyPoolingPriorEstimator{typeof(pe), typeof(views), typeof(sets),
                                        typeof(alg), typeof(opt), typeof(w)}(pe, views,
                                                                             sets, alg, opt,
                                                                             w)
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
struct EntropyPoolingPriorResult{T1 <: AbstractPriorResult, T2 <: AbstractWeights} <:
       AbstractEntropyPoolingPriorResult
    pr::T1
    w::T2
end
function EntropyPoolingPriorResult(; pr::AbstractPriorResult, w::AbstractWeights)
    @smart_assert(!isempty(w))
    @smart_assert(size(pr.X, 1) == length(w))
    return EntropyPoolingPriorResult{typeof(pr), typeof(w)}(pr, w)
end
function factory(pe::EntropyPoolingPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return EntropyPoolingPriorEstimator(; pe = factory(pe.pe, w), views = pe.views,
                                        sets = pe.sets, alg = pe.alg, opt = pe.opt,
                                        w = isnothing(w) ? pe.w : w)
end
function prior_view(pr::EntropyPoolingPriorResult, i::AbstractVector)
    return EntropyPoolingPriorResult(; pr = prior_view(pr.pr, i), w = pr.w)
end
function Base.getproperty(obj::EntropyPoolingPriorResult, sym::Symbol)
    return if sym == :X
        obj.pr.X
    elseif sym == :mu
        obj.pr.mu
    elseif sym == :sigma
        obj.pr.sigma
    elseif sym == :f_mu
        obj.pr.fpr.mu
    elseif sym == :f_sigma
        obj.pr.fpr.sigma
    elseif sym == :loadings
        obj.pr.fpr.loadings
    elseif sym == :chol
        obj.pr.chol
    else
        getfield(obj, sym)
    end
end
function prior(pe::EntropyPoolingPriorEstimator{<:Any, <:Any, <:Any, <:H0_EntropyPooling,
                                                <:Any, <:Any}, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
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
    views = entropy_pooling_views(pr, pe.views, pe.sets; strict = strict)
    w = entropy_pooling(w, views, pe.opt)
    pe = factory(pe, w)
    return EntropyPoolingPriorResult(; pr = prior(pe.pe, X, F; strict = strict, kwargs...),
                                     w = w)
end
function _get_epw(::H1_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return w0
end
function _get_epw(::H2_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return wi
end
function prior(pe::EntropyPoolingPriorEstimator{<:Any, <:Any, <:Any,
                                                <:Union{<:H1_EntropyPooling,
                                                        <:H2_EntropyPooling}, <:Any, <:Any},
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
        wi = entropy_pooling(_get_epw(pe.alg, w0, wi), V_i, pe.opt)
        pe = factory(pe, wi)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    return EntropyPoolingPriorResult(; pr = pr, w = wi)
end

export H0_EntropyPooling, H1_EntropyPooling, H2_EntropyPooling,
       OptimEntropyPoolingEstimator, JuMPEntropyPoolingEstimator, entropy_pooling,
       EntropyPoolingPriorEstimator, EntropyPoolingPriorResult, relative_entropy,
       effective_number_scenarios
