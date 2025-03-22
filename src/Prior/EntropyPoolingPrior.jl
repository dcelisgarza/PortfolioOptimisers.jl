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
                                                   <:AbstractVector{<:EntropyPoolingView}} = EntropyPoolingView(),
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
function create_constant_entropy_pooling_constraint(pm::AbstractPriorModel,
                                                    epvs::Union{<:EntropyPoolingView,
                                                                <:AbstractVector{<:EntropyPoolingView}},
                                                    sets::DataFrame; strict::Bool = false,
                                                    w::AbstractWeights = pweights(range(;
                                                                                        start = 1,
                                                                                        stop = 1,
                                                                                        length = size(pm.X,
                                                                                                      1))))
    res = EntropyPoolingView[]
    for (i, epv) ∈ enumerate(epvs)
        push!(res,
              EntropyPoolingView(;
                                 B = ConstantEntropyPoolingConstraint(;
                                                                      coef = freeze_B_view(pm,
                                                                                           epv.B,
                                                                                           sets,
                                                                                           strict;
                                                                                           w = w,
                                                                                           counter = (i,
                                                                                                      epv.B.coef))),
                                 A = freeze_A_view(epv.A), comp = epv.comp))
    end
    return res
end
function freeze_A_view(A::C0_LinearEntropyPoolingConstraint)
    return C0_LinearEntropyPoolingConstraint(; group = A.group, name = A.name,
                                             coef = sign(A.coef) * 1)
end
function freeze_A_view(A::C1_LinearEntropyPoolingConstraint)
    return C1_LinearEntropyPoolingConstraint(; group = A.group, name = A.name,
                                             coef = sign(A.coef) * 1)
end
function freeze_A_view(A::C2_LinearEntropyPoolingConstraint)
    return C2_LinearEntropyPoolingConstraint(; group = A.group, name = A.name,
                                             coef = sign(A.coef) * 1, kind = A.kind)
end
function freeze_A_view(A::C4_LinearEntropyPoolingConstraint)
    return C4_LinearEntropyPoolingConstraint(; group1 = A.group1, group2 = A.group2,
                                             name1 = A.name1, name2 = A.name2,
                                             coef = sign(A.coef) * 1)
end
function _freeze_view(epc::C0_LinearEntropyPoolingConstraint, pm::AbstractPriorModel,
                      idx::AbstractVector; kwargs...)
    mu = view(pm.mu, idx)
    return sign(epc.coef) * sum(mu)
end
function _freeze_view(epc::C1_LinearEntropyPoolingConstraint, pm::AbstractPriorModel,
                      idx::AbstractVector; kwargs...)
    sigma = pm.sigma
    dsigma = view(diag(sigma), idx)
    if !isone(epc.exponent)
        if isapprox(epc.exponent, 1 / 2)
            dsigma = sqrt.(dsigma)
        elseif isapprox(epc.exponent, 1 / 3)
            dsigma = cbrt.(dsigma)
        else
            dsigma = dsigma .^ epc.exponent
        end
    end
    return sign(epc.coef) * sum(dsigma)
end
function _freeze_view(epc::C2_LinearEntropyPoolingConstraint{<:Any, <:Any, <:Any,
                                                             <:SkewnessEntropyPoolingView},
                      pm::AbstractPriorModel, idx::AbstractVector;
                      w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                          length = size(pm.X, 1))),
                      kwargs...)
    X = view(pm.X, :, idx)
    return sign(epc.coef) * sum([skewness(X[:, i], w)] for i ∈ axes(X, 2))
end
function _freeze_view(epc::C2_LinearEntropyPoolingConstraint{<:Any, <:Any, <:Any,
                                                             <:KurtosisEntropyPoolingView},
                      pm::AbstractPriorModel, idx::AbstractVector;
                      w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                          length = size(pm.X, 1))),
                      kwargs...)
    X = view(pm.X, :, idx)
    return sign(epc.coef) * sum([kurtosis(X[:, i], w) + 3] for i ∈ axes(X, 2))
end
function _freeze_view(epc::C4_LinearEntropyPoolingConstraint, pm::AbstractPriorModel,
                      idx1::AbstractVector, idx2::AbstractVector; kwargs...)
    sigma = pm.sigma
    dsigma = diag(sigma)
    dsigma1 = sqrt.(view(dsigma, idx1))
    dsigma2 = sqrt.(view(dsigma, idx2))
    return sign(epc.coef) * sum(dsigma1 .* dsigma2)
end
function freeze_B_view(::AbstractPriorModel, epvb::ConstantEntropyPoolingConstraint,
                       ::DataFrame, ::Bool, args...; kwargs...)
    return epvb.coef
end
function freeze_B_view(pm::AbstractPriorModel,
                       epva::Union{<:C0_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                       <:Real},
                                   <:C1_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                       <:Real},
                                   <:C2_LinearEntropyPoolingConstraint{<:Any, <:Any,
                                                                       <:Real}},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group, name) = epva
    B = if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        _freeze_view(epva, pm, idx; w = w)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epva)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(epva)")
        zero(eltype(pm.X))
    end
    return B
end
function freeze_B_view(pm::AbstractPriorModel,
                       epva::Union{<:C0_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector},
                                   <:C1_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector},
                                   <:C2_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector}},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group, name) ∈ zip(epva.group, epva.name)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(B, _freeze_view(epva, pm, idx; w = w))
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(epva)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(epva).")
        end
    end
    return !isempty(B) ? sum(B) : zero(eltype(pm.X))
end
function freeze_B_view(pm::AbstractPriorModel,
                       epva::C4_LinearEntropyPoolingConstraint{<:Any, <:Any, <:Any, <:Any,
                                                               <:Real}, sets::DataFrame,
                       strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    (; group1, group2, name1, name2) = epva
    B = if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
        idx1 = sets[!, group1] .== name1
        idx2 = sets[!, group2] .== name2
        _freeze_view(epva, pm, idx1, idx2; w = w)
    elseif strict
        throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva)."))
    else
        @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva).")
    end
    return B
end
function freeze_B_view(pm::AbstractPriorModel,
                       epva::C4_LinearEntropyPoolingConstraint{<:AbstractVector,
                                                               <:AbstractVector,
                                                               <:AbstractVector,
                                                               <:AbstractVector,
                                                               <:AbstractVector},
                       sets::DataFrame, strict::Bool = false;
                       w::AbstractWeights = pweights(range(; start = 1, stop = 1,
                                                           length = size(pm.X, 1))),
                       kwargs...)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    B = Vector{eltype(pm.X)}(undef, 0)
    for (group1, group2, name1, name2) ∈
        zip(epva.group1, epva.group2, epva.name1, epva.name2, epva.coef)
        if !(isnothing(group1) ||
             string(group1) ∉ group_names ||
             isnothing(group2) ||
             string(group2) ∉ group_names)
            idx1 = sets[!, group1] .== name1
            idx2 = sets[!, group2] .== name2
            append!(B, _freeze_view(epva, pm, idx1, idx2; w = w))
        elseif strict
            throw(ArgumentError("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva)."))
        else
            @warn("$(string(group1)) or $(string(group1)) are not in $(group_names).\n$(epva).")
        end
    end
    return !isempty(B) ? sum(B) : zero(eltype(pm.X))
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
    pe = moment_factory_w(pe, w0)
    pm = prior(pe.pe, X, F; strict = strict, kwargs...)
    V_i = nothing
    for (i, uvl) ∈ enumerate(uvls)
        V_idx = vls .== uvl
        idx = idx .|| V_idx
        excluded = findall(V_idx)[to_be_frozen.(views[V_idx])]
        included = setdiff(findall(idx), excluded)
        if isempty(included)
            continue
        end
        v = views[included]
        V_i = entropy_pooling_views(pm, v, pe.sets; w = w0, strict = strict)
        wi = entropy_pooling(_get_epw(pe.alg, w0, wi), V_i, pe.opt)
        pe = moment_factory_w(pe, wi)
        pm = prior(pe.pe, X, F; strict = strict, kwargs...)
        views[[excluded; included]] = create_constant_entropy_pooling_constraint(pm,
                                                                                 views[[excluded;
                                                                                        included]],
                                                                                 pe.sets;
                                                                                 strict = strict,
                                                                                 w = w0)
        if i == 4
            return EntropyPoolingModel(; pm = pm, views = V_i, w = wi)
        end
        # println(included)
        # println(excluded)
    end
    return EntropyPoolingModel(; pm = pm, views = V_i, w = wi)
end
function entropy_pooling(w::AbstractVector, epcs::LinearConstraintModel,
                         optim::OptimEntropyPooling)
    (; A_eq, B_eq, A_ineq, B_ineq) = epcs
    lhs, rhs, lb, ub = if !isnothing(A_eq) && !isnothing(B_eq)
        A_eq = vcat(ones(eltype(A_eq), 1, size(A_eq, 2)), A_eq)
        B_eq = vcat(eltype(B_eq)[1], B_eq)
        A_eq, B_eq, fill(-Inf, length(B_eq)), fill(Inf, length(B_eq))
    else
        ones(eltype(w), 1, length(w)), eltype(w)[1], [-Inf], [Inf]
    end
    lhs_ineq, rhs_ineq, lb_ineq, ub_ineq = if !isnothing(A_ineq) && !isnothing(B_ineq)
        A_ineq, B_ineq, fill(0, length(B_ineq)), fill(Inf, length(B_ineq))
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

export H0_EntropyPooling, H1_EntropyPooling, H2_EntropyPooling, OptimEntropyPooling,
       JuMPEntropyPooling, entropy_pooling, EntropyPoolingPriorEstimator
