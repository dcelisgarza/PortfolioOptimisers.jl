abstract type AbstractEntropyPoolingEstimator <: AbstractEstimator end
abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
function _get_epw(::H1_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return w0
end
function _get_epw(::H2_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return wi
end
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
                                     sc::Real = 1, so::Real = 1)
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
# function entropy_pooling(w::AbstractVector, epcs::LinearConstraintResult,
#                          optim::OptimEntropyPoolingEstimator)
#     T = length(w)
#     factor = inv(sqrt(T))
#     A_sum = fill(factor, 1, T)
#     B_sum = [factor]
#     (; A_eq, B_eq, A_ineq, B_ineq) = epcs
#     lhs, rhs, lb, ub = if !isnothing(A_eq) && !isnothing(B_eq)
#         A_eq = vcat(A_sum, A_eq)
#         B_eq = vcat(B_sum, B_eq)
#         A_eq, B_eq, fill(typemin(eltype(B_eq)), length(B_eq)),
#         fill(typemax(eltype(B_eq)), length(B_eq))
#     else
#         A_sum, B_sum, [typemin(eltype(w))], [typemax(eltype(w))]
#     end
#     if !isnothing(A_ineq) && !isnothing(B_ineq)
#         lhs = vcat(lhs, A_ineq)
#         rhs = vcat(rhs, B_ineq)
#         lb = vcat(lb, fill(zero(eltype(B_ineq)), length(B_ineq)))
#         ub = vcat(ub, fill(typemax(eltype(B_ineq)), length(B_ineq)))
#     end
#     log_p = log.(w)
#     x0 = fill(inv(T), size(lhs, 1))
#     G = similar(x0)
#     last_x = similar(x0)
#     grad = similar(G)
#     log_x = similar(log_p)
#     y = similar(log_p)
#     function common_op(x)
#         if x != last_x
#             copy!(last_x, x)
#             log_x .= log_p - (one(eltype(log_p)) .+ transpose(lhs) * x)
#             y .= exp.(log_x)
#             grad .= rhs - lhs * y
#         end
#     end
#     function f(x)
#         common_op(x)
#         return optim.scale * (dot(x, grad) - dot(y, log_x - log_p))
#     end
#     function g!(G, x)
#         common_op(x)
#         G .= grad
#         return optim.scale * G
#     end
#     result = Optim.optimize(f, g!, lb, ub, x0, optim.args...; optim.kwargs...)
#     x = if Optim.converged(result)
#         # Compute posterior probabilities
#         Optim.minimizer(result)
#     else
#         throw(ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
#     end
#     return pweights(exp.(log_p - (one(eltype(log_p)) .+ transpose(lhs) * x)))
# end
# function entropy_pooling(w::AbstractVector, epcs::LinearConstraintResult,
#                          optim::JuMPEntropyPoolingEstimator)
#     model = Model()
#     S = length(w)
#     log_p = log.(w)
#     # Decision variables (posterior probabilities)
#     @variables(model, begin
#                    q[1:S]
#                    t
#                end)
#     (; A_eq, B_eq, A_ineq, B_ineq) = epcs
#     (; sc, so, slv) = optim
#     # Equality constraints from A_ineq and B_ineq if provided
#     if !isnothing(A_eq) && !isnothing(B_eq)
#         @constraint(model, ceq, sc * (A_eq * q - B_eq) == 0)
#     end
#     # Inequality constraints from A_ineq and B_ineq if provided
#     if !isnothing(A_ineq) && !isnothing(B_ineq)
#         @constraint(model, cineq, sc * (A_ineq * q - B_ineq) <= 0)
#     end
#     # Equality constraints from A_eq and B_eq and probabilities equal to 1
#     @constraints(model,
#                  begin
#                      sc * (sum(q) - one(eltype(w))) == 0
#                      [sc * t; fill(sc, S); sc * q] in MOI.RelativeEntropyCone(2 * S + 1)
#                  end)
#     @objective(model, Min, so * (t - dot(q, log_p)))
#     # Solve the optimization problem
#     return if optimise_JuMP_model!(model, slv).success
#         pweights(value.(q))
#     else
#         pweights(fill(NaN, length(q)))
#     end
# end
# function entropy_pooling(w::AbstractVector, epcs::Union{Nothing, <:LinearConstraintResult},
#                          optim::Union{<:OptimEntropyPoolingEstimator,
#                                       <:JuMPEntropyPoolingEstimator}, ::Nothing, ::Nothing,
#                          ::Any, ::Any)
#     return entropy_pooling(w, epcs, optim)
# end
# function entropy_pooling(w::AbstractVector, epcs::Union{Nothing, <:LinearConstraintResult},
#                          optim::Union{<:OptimEntropyPoolingEstimator,
#                                       <:JuMPEntropyPoolingEstimator},
#                          d_epcs::Union{Nothing, <:LinearConstraintResult},
#                          d_views::Union{<:DiscontinuousEntropyPoolingViewEstimator,
#                                         <:AbstractVector{<:DiscontinuousEntropyPoolingViewEstimator}},
#                          ds_opt::Union{Nothing, <:OptimEntropyPoolingEstimator},
#                          dm_opt::Union{Nothing, <:OptimEntropyPoolingEstimator})
#     N = length(d_views)
#     VN = Val(N)
#     alpha, d_opt = if N == 1
#         d_views.A.alpha, !isnothing(ds_opt) ? ds_opt : OptimEntropyPoolingEstimator()
#     else
#         getproperty.(getproperty.(d_views, :A), :alpha),
#         !isnothing(dm_opt) ? dm_opt : OptimEntropyPoolingEstimator()
#     end
#     A = d_epcs.A_eq
#     B = d_epcs.B_eq
#     scale = d_opt.scale
#     N = length(B)
#     ialpha = inv(alpha)
#     function g(x)
#         @smart_assert(all(zero(eltype(x)) .< x .<= B))
#         pos_part = max.(-A .- x, zero(eltype(x)))
#         A_eq = vcat(epcs.A_eq, pos_part * ialpha)
#         B_eq = vcat(epcs.B_eq, B .- x)
#         epcs2 = LinearConstraintResult(; ineq = epcs.ineq,
#                                        eq = PartialLinearConstraintResult(; A = A_eq,
#                                                                           B = B_eq))
#         return pos_part, entropy_pooling(w, epcs2, optim)
#     end
#     function h(::Val{1}, _A, _B, _w, _pos_part)
#         return scale * (sum(_w[.!iszero.(_pos_part)]) - alpha)
#     end
#     function h(::Any, _A, _B, _w, _pos_part)
#         return scale * norm(cvar(transpose(_A), alpha, _w) - _B) / sqrt(N)
#     end
#     function f(x)
#         pos_part, w = g(x)
#         return h(VN, A, B, w, pos_part)
#     end
#     result = if N == 1
#         Optim.optimize(f, zero(eltype(B)), B[1], d_opt.args...; d_opt.kwargs...)
#     else
#         Optim.optimize(f, zeros(N), B, 0.5 * B, d_opt.args...; d_opt.kwargs...)
#     end
#     x = if Optim.converged(result)
#         # Compute posterior probabilities
#         Optim.minimizer(result)
#     else
#         throw(ErrorException("CVaR entropy pooling optimisation failed. Relax the view, incrase alpha, use different solver parameters, or use a different prior."))
#     end
#     return g(x)
# end
function effective_number_scenarios(x::AbstractVector, y::AbstractVector)
    return exp(-kldivergence(x, y))
end
struct EPPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_1o2_1o2,
                        T2 <: Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                        T3 <: Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                        T4 <: Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                        T5 <: Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                        T6 <: Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                        T7 <: Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                        T8 <: Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                        T9 <: Real, T10 <: Real, T11 <: Union{Nothing, <:AssetSets},
                        T12 <: Union{Nothing, <:OptimEntropyPoolingEstimator},
                        T13 <: Union{Nothing, <:OptimEntropyPoolingEstimator},
                        T14 <: AbstractEntropyPoolingEstimator,
                        T15 <: Union{Nothing, <:AbstractVector},
                        T16 <: AbstractEntropyPoolingAlgorithm} <:
       AbstractLowOrderPriorEstimator_1o2_1o2
    pe::T1
    mu_views::T2
    var_views::T3
    cvar_views::T4
    sigma_views::T5
    sk_views::T6
    kt_views::T7
    rho_views::T8
    var_alpha::T9
    cvar_alpha::T10
    sets::T11
    ds_opt::T12
    dm_opt::T13
    opt::T14
    w::T15
    alg::T16
end
function EPPriorEstimator(;
                          pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                          mu_views::Union{Nothing, <:AbstractString, Expr,
                                          <:AbstractVector{<:AbstractString},
                                          <:AbstractVector{Expr},
                                          <:AbstractVector{<:Union{<:AbstractString, Expr}}} = nothing,
                          var_views::Union{Nothing, <:AbstractString, Expr,
                                           <:AbstractVector{<:AbstractString},
                                           <:AbstractVector{Expr},
                                           <:AbstractVector{<:Union{<:AbstractString, Expr}}} = nothing,
                          cvar_views::Union{Nothing, <:AbstractString, Expr,
                                            <:AbstractVector{<:AbstractString},
                                            <:AbstractVector{Expr},
                                            <:AbstractVector{<:Union{<:AbstractString,
                                                                     Expr}}} = nothing,
                          sigma_views::Union{Nothing, <:AbstractString, Expr,
                                             <:AbstractVector{<:AbstractString},
                                             <:AbstractVector{Expr},
                                             <:AbstractVector{<:Union{<:AbstractString,
                                                                      Expr}}} = nothing,
                          sk_views::Union{Nothing, <:AbstractString, Expr,
                                          <:AbstractVector{<:AbstractString},
                                          <:AbstractVector{Expr},
                                          <:AbstractVector{<:Union{<:AbstractString, Expr}}} = nothing,
                          kt_views::Union{Nothing, <:AbstractString, Expr,
                                          <:AbstractVector{<:AbstractString},
                                          <:AbstractVector{Expr},
                                          <:AbstractVector{<:Union{<:AbstractString, Expr}}} = nothing,
                          rho_views::Union{Nothing, <:AbstractString, Expr,
                                           <:AbstractVector{<:AbstractString},
                                           <:AbstractVector{Expr},
                                           <:AbstractVector{<:Union{<:AbstractString, Expr}}} = nothing,
                          var_alpha::Real = 0.05, cvar_alpha::Real = 0.05,
                          sets::Union{Nothing, <:AssetSets} = nothing,
                          ds_opt::Union{Nothing, <:OptimEntropyPoolingEstimator} = nothing,
                          dm_opt::Union{Nothing, <:OptimEntropyPoolingEstimator} = nothing,
                          opt::AbstractEntropyPoolingEstimator = OptimEntropyPoolingEstimator(),
                          w::Union{Nothing, AbstractVector} = nothing,
                          alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())
    if isa(mu_views, AbstractVector)
        @smart_assert(!isempty(mu_views))
    end
    if isa(var_views, AbstractVector)
        @smart_assert(!isempty(var_views))
    end
    if isa(cvar_views, AbstractVector)
        @smart_assert(!isempty(cvar_views))
    end
    if isa(sigma_views, AbstractVector)
        @smart_assert(!isempty(sigma_views))
    end
    if isa(sk_views, AbstractVector)
        @smart_assert(!isempty(sk_views))
    end
    if isa(kt_views, AbstractVector)
        @smart_assert(!isempty(kt_views))
    end
    if isa(rho_views, AbstractVector)
        @smart_assert(!isempty(rho_views))
    end
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    if !isnothing(mu_views) ||
       !isnothing(var_views) ||
       !isnothing(cvar_views) ||
       !isnothing(sigma_views) ||
       !isnothing(sk_views) ||
       !isnothing(kt_views) ||
       !isnothing(rho_views)
        @smart_assert(!isnothing(sets))
    end
    @smart_assert(zero(var_alpha) < var_alpha < one(var_alpha))
    @smart_assert(zero(cvar_alpha) < cvar_alpha < one(cvar_alpha))

    return EPPriorEstimator{typeof(pe), typeof(mu_views), typeof(var_views),
                            typeof(cvar_views), typeof(sigma_views), typeof(sk_views),
                            typeof(kt_views), typeof(rho_views), typeof(var_alpha),
                            typeof(cvar_alpha), typeof(sets), typeof(ds_opt),
                            typeof(dm_opt), typeof(opt), typeof(w), typeof(alg)}(pe,
                                                                                 mu_views,
                                                                                 var_views,
                                                                                 cvar_views,
                                                                                 sigma_views,
                                                                                 sk_views,
                                                                                 kt_views,
                                                                                 rho_views,
                                                                                 var_alpha,
                                                                                 cvar_alpha,
                                                                                 sets,
                                                                                 ds_opt,
                                                                                 dm_opt,
                                                                                 opt, w,
                                                                                 alg)
end
function Base.getproperty(obj::EPPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function factory(pe::EPPriorEstimator, w::Union{Nothing, <:AbstractWeights} = nothing)
    return EPPriorEstimator(; pe = factory(pe.pe, w), mu_views = pe.mu_views,
                            var_views = pe.var_views, cvar_views = pe.cvar_views,
                            sigma_views = pe.sigma_views, sk_views = pe.sk_views,
                            kt_views = pe.kt_views, rho_views = pe.rho_views,
                            var_alpha = pe.var_alpha, cvar_alpha = pe.cvar_alpha,
                            sets = pe.sets, ds_opt = pe.ds_opt, dm_opt = pe.dm_opt,
                            opt = pe.opt, w = nothing_scalar_array_factory(pe.w, w),
                            alg = pe.alg)
end
function get_pr_value end
function replace_prior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets,
                             key::Symbol, alpha::Union{Nothing, <:Real} = nothing;
                             strict::Bool = false)
    prior_pattern = r"prior\(([^()]*)\)"
    nx = sets.dict[sets.key]
    variables, coeffs = res.vars, res.coef
    idx_rm = Vector{Int}(undef, 0)
    rhs::typeof(res.rhs) = res.rhs
    non_prior = false
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            non_prior = true
            continue
        end
        j = findfirst(x -> x == m.captures[1], nx)
        if isnothing(j)
            msg = "Asset $(m.captures[1]) not found in $nx."
            strict ? throw(ArgumentError(msg)) : @warn(msg)
            push!(idx_rm, i)
            continue
        end
        rhs -= get_pr_value(pr, j, Val(key), alpha) * c
        push!(idx_rm, i)
    end
    if isempty(idx_rm)
        return res
    end
    if !non_prior
        throw(ArgumentError("Priors in views are replaced by their prior value, thus they are essentially part of the constant of the view, so you need a non-prior view to serve as the variable.\n$(res)"))
    end
    idx = setdiff(1:length(variables), idx_rm)
    variables_new = variables[idx]
    coeffs_new = coeffs[idx]
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "))
    return ParsingResult(variables_new, coeffs_new, res.op, rhs, "$(eqn) $(res.op) $(rhs)")
end
function add_ep_constraint!(epc::AbstractDict, lhs::AbstractMatrix, rhs::AbstractVector,
                            key::Symbol)
    scale = norm(lhs)
    lhs /= scale
    rhs /= scale
    epc[key] = if !haskey(epc, key)
        (lhs, rhs)
    else
        (vcat(epc[key][1], lhs), append!(epc[key][2], rhs))
    end
    return nothing
end
function replace_prior_views(res::AbstractVector{<:ParsingResult}, pr::AbstractPriorResult,
                             sets::AssetSets, key::Symbol,
                             alpha::Union{Nothing, <:Real} = nothing; strict::Bool = false)
    return replace_prior_views.(res, pr, sets, key, alpha; strict = strict)
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:mu}, args...)
    return pr.mu[i]
end
function ep_mu_views!(mu_views::Nothing, args...; kwargs...)
    return nothing
end
function ep_mu_views!(mu_views::Union{<:AbstractString, Expr,
                                      <:AbstractVector{<:AbstractString},
                                      <:AbstractVector{Expr},
                                      <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                      epc::AbstractDict, pr::AbstractPriorResult, sets::AssetSets;
                      strict::Bool = false)
    mu_views = parse_equation(mu_views)
    mu_views = replace_group_by_assets(mu_views, sets, false, true, false)
    mu_views = replace_prior_views(mu_views, pr, sets, :mu; strict = strict)
    lcs = get_linear_constraints(mu_views, sets; datatype = eltype(pr.X), strict = strict)
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        add_ep_constraint!(epc, getproperty(lcs, p).A * transpose(pr.X),
                           getproperty(lcs, p).B, p)
    end
    return nothing
end
function fix_mu!(epc::AbstractDict, fixed::AbstractVector, to_fix::AbstractVector,
                 pr::AbstractPriorResult)
    fix = to_fix .& .!fixed
    if any(fix)
        add_ep_constraint!(epc, transpose(pr.X[:, fix]), pr.mu[fix], :feq)
        fixed .= fixed .| fix
    end
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Real)
    return -partialsort(pr.X[:, i], ceil(Int, alpha * size(pr.X, 1)))
end
function ep_var_views!(var_views::Nothing, args...; kwargs...)
    return nothing
end
function ep_var_views!(var_views::Union{<:AbstractString, Expr,
                                        <:AbstractVector{<:AbstractString},
                                        <:AbstractVector{Expr},
                                        <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                       epc::AbstractDict, pr::AbstractPriorResult, sets::AssetSets,
                       alpha::Real; strict::Bool = false)
    var_views = parse_equation(var_views)
    var_views = replace_group_by_assets(var_views, sets, false, true, false)
    var_views = replace_prior_views(var_views, pr, sets, :var, alpha; strict = strict)
    lcs = get_linear_constraints(var_views, sets; datatype = eltype(pr.X), strict = strict)
    if !isnothing(lcs.ineq) && any(x -> x != 1, count(!iszero, lcs.A_ineq; dims = 2)) ||
       !isnothing(lcs.eq) && any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2))
        throw(ArgumentError("Cannot mix multiple assets in a single `var_view`."))
    end
    if !isnothing(lcs.eq) && any(x -> x < zero(eltype(x)), lcs.B_eq) ||
       !isnothing(lcs.ineq) && any(x -> x < zero(eltype(x)), lcs.A_ineq .* lcs.B_ineq)
        throw(ArgumentError("`var_view` cannot be negative."))
    end
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        B = getproperty(lcs, p).B
        for i in eachindex(B)
            j = .!iszero.(A[i, :])
            idx = findall(x -> x <= -abs(B[i]), pr.X[:, j])
            if isempty(idx)
                throw(ArgumentError("View $(B[i]) is too extreme, the maximum viable for asset $(findfirst(x->x==true,j)) is $(minimum(pr.X[:,j])). Please lower it or use a different prior with fatter tails."))
            end
            sign = p == :eq || B[i] >= zero(eltype(B)) ? one(eltype(B)) : -one(eltype(B))
            Ai = zeros(eltype(pr.X), 1, size(pr.X, 1))
            Ai[1, idx] .= sign
            add_ep_constraint!(epc, Ai, [sign * alpha], p)
        end
    end
    return nothing
end
function prior(pe::EPPriorEstimator{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                    <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                    <:H1_EntropyPooling}, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @smart_assert(dims in (1, 2))
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
        @smart_assert(length(pe.w) == T)
        pweights(pe.w)
    end
    pe = factory(pe, w)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    fixed = falses(N, 4)
    epc = Dict{Symbol, Tuple{<:AbstractMatrix, <:AbstractVector}}()
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    # d_V = entropy_pooling_views(pr, pe.d_views, pe.sets; strict = strict)
    # views = entropy_pooling_views(pr, pe.views, pe.sets; strict = strict)
    # w = entropy_pooling(w, views, pe.opt, d_V, pe.d_views, pe.d_opt1, pe.d_opt2)
    pe = factory(pe, w)
    (; X, mu, sigma, chol, loadings, f_mu, f_sigma) = prior(pe.pe, X, F; strict = strict,
                                                            kwargs...)
    return LowOrderPriorResult(; X = X, mu = mu, sigma = sigma, chol = chol, w = w,
                               loadings = loadings, f_mu = f_mu, f_sigma = f_sigma,
                               f_w = !isnothing(loadings) ? w : nothing)
end
