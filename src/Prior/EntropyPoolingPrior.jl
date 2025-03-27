abstract type EntropyPoolingOptimisation end
abstract type EntropyPoolingAlgorithm end
struct H0_EntropyPooling <: EntropyPoolingAlgorithm end
struct H1_EntropyPooling <: EntropyPoolingAlgorithm end
struct H2_EntropyPooling <: EntropyPoolingAlgorithm end
struct OptimEntropyPooling{T1 <: Real, T2 <: Tuple, T3 <: NamedTuple} <:
       EntropyPoolingOptimisation
    scale::T1
    args::T2
    kwargs::T3
end
function OptimEntropyPooling(; scale::Real = 1, args::Tuple = (), kwargs::NamedTuple = (;))
    @smart_assert(scale >= zero(scale))
    return OptimEntropyPooling{typeof(scale), typeof(args), typeof(kwargs)}(scale, args,
                                                                            kwargs)
end
struct JuMPEntropyPooling{T1 <: Real, T2 <: Real,
                          T3 <: Union{<:Solver, <:AbstractVector{<:Solver}}} <:
       EntropyPoolingOptimisation
    constr_scale::T1
    obj_scale::T2
    solvers::T3
end
function JuMPEntropyPooling(; solvers::Union{<:Solver, <:AbstractVector{<:Solver}},
                            constr_scale::Real = 1, obj_scale::Real = 1)
    @smart_assert(constr_scale >= zero(constr_scale))
    @smart_assert(obj_scale >= zero(obj_scale))
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    return JuMPEntropyPooling{typeof(constr_scale), typeof(obj_scale), typeof(solvers)}(constr_scale,
                                                                                        obj_scale,
                                                                                        solvers)
end
struct EntropyPoolingPriorEstimator{T1 <: AbstractPriorEstimatorMap_1o2_1o2,
                                    T2 <: Union{<:EntropyPoolingView,
                                                <:AbstractVector{<:EntropyPoolingView}},
                                    T3 <: DataFrame, T4 <: EntropyPoolingAlgorithm,
                                    T5 <: EntropyPoolingOptimisation,
                                    T6 <: Union{Nothing, <:AbstractVector}} <:
       AbstractPriorEstimator_1o2_1o2
    pe::T1
    views::T2
    sets::T3
    alg::T4
    opt::T5
    w::T6
end
function EntropyPoolingPriorEstimator(;
                                      pe::AbstractPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                                      views::Union{<:EntropyPoolingView,
                                                   <:AbstractVector{<:EntropyPoolingView}},
                                      sets::DataFrame = DataFrame(),
                                      alg::EntropyPoolingAlgorithm = H0_EntropyPooling(),
                                      opt::EntropyPoolingOptimisation = OptimEntropyPooling(),
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
struct EntropyPoolingModel{T1 <: AbstractPriorModel,
                           T2 <: Union{<:LinearConstraintModel,
                                       <:AbstractVector{<:LinearConstraintModel}},
                           T3 <: AbstractWeights} <: AbstractEntropyPoolingPriorModel
    pm::T1
    views::T2
    w::T3
end
function EntropyPoolingModel(; pm::AbstractPriorModel,
                             views::Union{<:LinearConstraintModel,
                                          <:AbstractVector{<:LinearConstraintModel}},
                             w::AbstractWeights)
    @smart_assert(!isempty(w))
    @smart_assert(size(pm.X, 1) == length(w))
    return EntropyPoolingModel{typeof(pm), typeof(views), typeof(w)}(pm, views, w)
end
function moment_factory_w(pe::EntropyPoolingPriorEstimator,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return EntropyPoolingPriorEstimator(; pe = moment_factory_w(pe.pe, w), views = pe.views,
                                        sets = pe.sets, alg = pe.alg, opt = pe.opt, w = w)
end
function Base.getproperty(obj::EntropyPoolingModel, sym::Symbol)
    return if sym == :X
        obj.pm.X
    elseif sym == :mu
        obj.pm.mu
    elseif sym == :sigma
        obj.pm.sigma
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
    pe = moment_factory_w(pe, w)
    pm = prior(pe.pe, X, F; strict = strict, kwargs...)
    views = entropy_pooling_views(pm, pe.views, pe.sets; strict = strict)
    w = entropy_pooling(w, views, pe.opt)
    pe = moment_factory_w(pe, w)
    return EntropyPoolingModel(; pm = prior(pe.pe, X, F; strict = strict, kwargs...),
                               views = views, w = w)
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
    V_i = nothing
    # Compute the prior observations.
    pe = moment_factory_w(pe, w0)
    pm = prior(pe.pe, X, F; strict = strict, kwargs...)
    for uvl ∈ uvls
        # Freeze updated parameters, ie parameters which were free in the previous iteration plus the ones which were adjusted via views. If there were no views the previous iteration, the free parameters become the new updated parameters and therefore frozen. In the first iteration, there is nothing to freeze.
        constant_entropy_pooling_constraint!(pm, cache, [excluded;
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
        V_i = entropy_pooling_views(pm, v, pe.sets; w = w0, strict = strict)
        # Compute the posterior observations.
        wi = entropy_pooling(_get_epw(pe.alg, w0, wi), V_i, pe.opt)
        pe = moment_factory_w(pe, wi)
        pm = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    return EntropyPoolingModel(; pm = pm, views = V_i, w = wi)
end
function entropy_pooling(w::AbstractVector, epcs::LinearConstraintModel,
                         optim::OptimEntropyPooling)
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
            log_x .= log_p .- one(eltype(log_p)) .- lhs' * x
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
    return pweights(exp.(log_p .- one(eltype(log_p)) .- lhs' * x))
end
function entropy_pooling(w::AbstractVector, epcs::LinearConstraintModel,
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
    (; constr_scale, obj_scale, solvers) = optim
    # Equality constraints from A_ineq and B_ineq if provided
    if !isnothing(A_eq) && !isnothing(B_eq)
        @constraint(model, constr_eq, constr_scale * A_eq * q == constr_scale * B_eq)
    end
    # Inequality constraints from A_ineq and B_ineq if provided
    if !isnothing(A_ineq) && !isnothing(B_ineq)
        @constraint(model, constr_ineq, constr_scale * A_ineq * q <= constr_scale * B_ineq)
    end
    # Equality constraints from A_eq and B_eq and probabilities equal to 1
    @constraints(model,
                 begin
                     sum(q) == 1
                     constr_scale * [t; ones(S); q] in MOI.RelativeEntropyCone(2 * S + 1)
                 end)
    @objective(model, Min, obj_scale * (t - dot(q, log_p)))
    # Solve the optimization problem
    success, solvers_tried = optimise_JuMP_model(model, solvers)
    return if success
        pweights(value.(q))
    else
        @warn("model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
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

export H0_EntropyPooling, H1_EntropyPooling, H2_EntropyPooling, OptimEntropyPooling,
       JuMPEntropyPooling, entropy_pooling, EntropyPoolingPriorEstimator, relative_entropy,
       effective_number_scenarios
