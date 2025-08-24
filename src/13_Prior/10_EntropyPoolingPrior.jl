abstract type AbstractEntropyPoolingOptimiser <: AbstractEstimator end
abstract type AbstractEntropyPoolingAlgorithm <: AbstractAlgorithm end
struct H0_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H1_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
struct H2_EntropyPooling <: AbstractEntropyPoolingAlgorithm end
abstract type AbstractEntropyPoolingOptAlgorithm <: AbstractAlgorithm end
struct LogEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
struct ExpEntropyPooling <: AbstractEntropyPoolingOptAlgorithm end
function get_epw(::H1_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return w0
end
function get_epw(::H2_EntropyPooling, w0::AbstractWeights, wi::AbstractWeights)
    return wi
end
struct CVaREntropyPooling{T1, T2} <: AbstractEntropyPoolingOptimiser
    args::T1
    kwargs::T2
end
function CVaREntropyPooling(; args::Tuple = (Roots.Brent(),), kwargs::NamedTuple = (;))
    return CVaREntropyPooling(args, kwargs)
end
struct OptimEntropyPooling{T1, T2, T3, T4, T5} <: AbstractEntropyPoolingOptimiser
    args::T1
    kwargs::T2
    sc1::T3
    sc2::T4
    alg::T5
end
function OptimEntropyPooling(; args::Tuple = (), kwargs::NamedTuple = (;), sc1::Real = 1,
                             sc2::Real = 1e3,
                             alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())
    @argcheck(sc1 >= zero(sc1))
    return OptimEntropyPooling(args, kwargs, sc1, sc2, alg)
end
struct JuMPEntropyPooling{T1, T2, T3, T4, T5} <: AbstractEntropyPoolingOptimiser
    slv::T1
    sc1::T2
    sc2::T3
    so::T4
    alg::T5
end
function JuMPEntropyPooling(; slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                            sc1::Real = 1, sc2::Real = 1e5, so::Real = 1,
                            alg::AbstractEntropyPoolingOptAlgorithm = ExpEntropyPooling())
    if isa(slv, AbstractVector)
        @argcheck(!isempty(slv))
    end
    @argcheck(sc1 >= zero(sc1))
    @argcheck(sc2 >= zero(sc2))
    @argcheck(so >= zero(so))
    return JuMPEntropyPooling(slv, sc1, sc2, so, alg)
end
struct EntropyPoolingPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
                           T16} <: AbstractLowOrderPriorEstimator_1o2_1o2
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
function EntropyPoolingPrior(;
                             pe::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPrior(),
                             mu_views::Union{Nothing, <:LinearConstraintEstimator} = nothing,
                             var_views::Union{Nothing, <:LinearConstraintEstimator} = nothing,
                             cvar_views::Union{Nothing, <:LinearConstraintEstimator} = nothing,
                             sigma_views::Union{Nothing, <:LinearConstraintEstimator} = nothing,
                             sk_views::Union{Nothing, <:LinearConstraintEstimator} = nothing,
                             kt_views::Union{Nothing, <:LinearConstraintEstimator} = nothing,
                             rho_views::Union{Nothing, <:LinearConstraintEstimator} = nothing,
                             var_alpha::Real = 0.05, cvar_alpha::Real = 0.05,
                             sets::Union{Nothing, <:AssetSets} = nothing,
                             ds_opt::Union{Nothing, <:CVaREntropyPooling} = nothing,
                             dm_opt::Union{Nothing, <:OptimEntropyPooling} = nothing,
                             opt::Union{<:OptimEntropyPooling, <:JuMPEntropyPooling} = OptimEntropyPooling(),
                             w::Union{Nothing, AbstractVector} = nothing,
                             alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())
    if isa(w, AbstractWeights)
        @argcheck(!isempty(w))
    end
    if !isnothing(mu_views) ||
       !isnothing(var_views) ||
       !isnothing(cvar_views) ||
       !isnothing(sigma_views) ||
       !isnothing(sk_views) ||
       !isnothing(kt_views) ||
       !isnothing(rho_views)
        @argcheck(!isnothing(sets))
    end
    @argcheck(zero(var_alpha) < var_alpha < one(var_alpha))
    @argcheck(zero(cvar_alpha) < cvar_alpha < one(cvar_alpha))
    return EntropyPoolingPrior(pe, mu_views, var_views, cvar_views, sigma_views, sk_views,
                               kt_views, rho_views, var_alpha, cvar_alpha, sets, ds_opt,
                               dm_opt, opt, w, alg)
end
function Base.getproperty(obj::EntropyPoolingPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function factory(pe::EntropyPoolingPrior, w::Union{Nothing, <:AbstractWeights} = nothing)
    return EntropyPoolingPrior(; pe = factory(pe.pe, w), mu_views = pe.mu_views,
                               var_views = pe.var_views, cvar_views = pe.cvar_views,
                               sigma_views = pe.sigma_views, sk_views = pe.sk_views,
                               kt_views = pe.kt_views, rho_views = pe.rho_views,
                               var_alpha = pe.var_alpha, cvar_alpha = pe.cvar_alpha,
                               sets = pe.sets, ds_opt = pe.ds_opt, dm_opt = pe.dm_opt,
                               opt = pe.opt, w = nothing_scalar_array_factory(pe.w, w),
                               alg = pe.alg)
end
function get_pr_value end
function add_ep_constraint!(epc::AbstractDict, lhs::AbstractMatrix, rhs::AbstractVector,
                            key::Symbol)
    sc = norm(lhs)
    lhs /= sc
    rhs /= sc
    epc[key] = if !haskey(epc, key)
        (lhs, rhs)
    else
        (vcat(epc[key][1], lhs), append!(epc[key][2], rhs))
    end
    return nothing
end
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
function replace_prior_views(res::AbstractVector{<:ParsingResult}, args...; kwargs...)
    return replace_prior_views.(res, args...; kwargs...)
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:mu}, args...)
    return pr.mu[i]
end
function ep_mu_views!(mu_views::Nothing, args...; kwargs...)
    return nothing
end
function ep_mu_views!(mu_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    mu_views = parse_equation(mu_views.val; datatype = eltype(pr.X))
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
    return nothing
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:var}, alpha::Real)
    #! Don't use a view, use a copy, value at risk uses partialsort!
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ValueatRisk(; alpha = alpha)(pr.X[:, i])
end
function ep_var_views!(var_views::Nothing, args...; kwargs...)
    return nothing
end
function ep_var_views!(var_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::AssetSets, alpha::Real;
                       strict::Bool = false)
    var_views = parse_equation(var_views.val; ops1 = ("==", ">="),
                               ops2 = (:call, :(==), :(>=)), datatype = eltype(pr.X))
    var_views = replace_group_by_assets(var_views, sets, false, true, false)
    var_views = replace_prior_views(var_views, pr, sets, :var, alpha; strict = strict)
    lcs = get_linear_constraints(var_views, sets; datatype = eltype(pr.X), strict = strict)
    if !isnothing(lcs.ineq) && !any(x -> (iszero(x) || isone(x)), lcs.A_ineq) ||
       !isnothing(lcs.eq) && !any(x -> (iszero(x) || isone(x)), lcs.A_eq)
        throw(ArgumentError("`var_view` only supports coefficients of 1.\n$var_views"))
    end
    if !isnothing(lcs.ineq) && any(x -> x != 1, count(!iszero, lcs.A_ineq; dims = 2)) ||
       !isnothing(lcs.eq) && any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2))
        throw(ArgumentError("Cannot mix multiple assets in a single `var_view`.\n$var_views"))
    end
    if !isnothing(lcs.eq) && any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq) ||
       !isnothing(lcs.ineq) && any(x -> x < zero(eltype(x)), lcs.A_ineq .* lcs.B_ineq)
        throw(ArgumentError("`var_view` cannot be negative.\n$var_views"))
    end
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        B = getproperty(lcs, p).B
        for i in eachindex(B)
            j = .!iszero.(A[i, :])
            #! Figure out a way to include pr.w, probably see how it's implemented in ValueatRisk.
            idx = findall(x -> x <= -abs(B[i]), view(pr.X, :, j))
            if isempty(idx)
                throw(ArgumentError("View `$(var_views[i].eqn)` is too extreme, the maximum viable for asset $(findfirst(x -> x == true, j)) is $(-minimum(pr.X[:,j])). Please lower it or use a different prior with fatter tails."))
            end
            sign = ifelse(p == :eq || B[i] >= zero(eltype(B)), one(eltype(B)),
                          -one(eltype(B)))
            Ai = zeros(eltype(pr.X), 1, size(pr.X, 1))
            Ai[1, idx] .= sign
            add_ep_constraint!(epc, Ai, [sign * alpha], p)
        end
    end
    return nothing
end
function entropy_pooling(w::AbstractVector, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:ExpEntropyPooling})
    T = length(w)
    factor = inv(sqrt(T))
    A = fill(factor, 1, T)
    B = [factor]
    wb = [typemin(eltype(w)) typemax(eltype(w))]
    for (key, val) in epc
        A = vcat(A, val[1])
        B = vcat(B, val[2])
        s = length(val[2])
        wb = if key == :eq || key == :cvar_eq
            vcat(wb, [fill(typemin(eltype(w)), s) fill(typemax(eltype(w)), s)])
        elseif key == :ineq || key == :cvar_ineq
            vcat(wb, [zeros(eltype(w), s) fill(typemax(eltype(w)), s)])
        elseif key == :feq
            vcat(wb, [fill(-opt.sc2, s) fill(opt.sc2, s)])
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
    end
    x0 = fill(factor, size(A, 1))
    G = similar(x0)
    last_x = similar(x0)
    grad = similar(G)
    y = similar(w)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            y .= w .* exp.(-transpose(A) * x .- one(eltype(w)))
            grad .= B - A * y
        end
    end
    function f(x)
        common_op(x)
        return opt.sc1 * sum(y) + dot(x, B)
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                            opt.kwargs...)
    x = if Optim.converged(result)
        # Compute posterior probabilities
        Optim.minimizer(result)
    else
        throw(ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    end
    return pweights(w .* exp.(-transpose(A) * x .- one(eltype(w))))
end
function entropy_pooling(w::AbstractVector, epc::AbstractDict,
                         opt::OptimEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                  <:LogEntropyPooling})
    T = length(w)
    factor = inv(sqrt(T))
    A = fill(factor, 1, T)
    B = [factor]
    wb = [typemin(eltype(w)) typemax(eltype(w))]
    for (key, val) in epc
        A = vcat(A, val[1])
        B = vcat(B, val[2])
        s = length(val[2])
        wb = if key == :eq || key == :cvar_eq
            vcat(wb, [fill(typemin(eltype(w)), s) fill(typemax(eltype(w)), s)])
        elseif key == :ineq
            vcat(wb, [zeros(eltype(w), s) fill(typemax(eltype(w)), s)])
        elseif key == :feq
            vcat(wb, [fill(-opt.sc2, s) fill(opt.sc2, s)])
        else
            throw(KeyError("Unknown key $(key) in epc."))
        end
    end
    log_p = log.(w)
    x0 = fill(factor, size(A, 1))
    G = similar(x0)
    last_x = similar(x0)
    grad = similar(G)
    log_x = similar(log_p)
    y = similar(log_p)
    function common_op(x)
        if x != last_x
            copy!(last_x, x)
            log_x .= log_p - (one(eltype(log_p)) .+ transpose(A) * x)
            y .= exp.(log_x)
            grad .= B - A * y
        end
    end
    function f(x)
        common_op(x)
        return opt.sc1 * (dot(x, grad) - dot(y, log_x - log_p))
    end
    function g!(G, x)
        common_op(x)
        G .= grad
        return opt.sc1 * G
    end
    result = Optim.optimize(f, g!, view(wb, :, 1), view(wb, :, 2), x0, opt.args...;
                            opt.kwargs...)
    x = if Optim.converged(result)
        # Compute posterior probabilities
        Optim.minimizer(result)
    else
        throw(ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    end
    return pweights(exp.(log_p - (one(eltype(log_p)) .+ transpose(A) * x)))
end
function entropy_pooling(w::AbstractVector, epc::AbstractDict,
                         opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                 <:ExpEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    T = length(w)
    model = Model()
    @variables(model, begin
                   t
                   x[1:T] >= 0
               end)
    @constraints(model,
                 begin
                     sc1 * (sum(x) - one(eltype(w))) == 0
                     [sc1 * t; sc1 * w; sc1 * x] in MOI.RelativeEntropyCone(2 * T + 1)
                 end)
    @expression(model, obj_expr, so * t)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        @constraint(model, ceq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        @constraint(model, cineq, sc1 * (A * x ⊖ B) <= 0)
    end
    if haskey(epc, :cvar_eq)
        A, B = epc[:cvar_eq]
        @constraint(model, ccvareq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :feq)
        A, B = epc[:feq]
        N = length(B)
        @variables(model, begin
                       tc
                       c[1:N]
                   end)
        @constraints(model, begin
                         cfeq, sc1 * (A * x ⊖ B ⊖ c) == 0
                         [sc1 * tc; sc1 * c] in MOI.NormOneCone(N + 1)
                     end)
        add_to_expression!(obj_expr, so * sc2 * tc)
    end
    @objective(model, Min, t)
    return if optimise_JuMP_model!(model, slv).success
        pweights(value.(x))
    else
        throw(ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    end
end
function entropy_pooling(w::AbstractVector, epc::AbstractDict,
                         opt::JuMPEntropyPooling{<:Any, <:Any, <:Any, <:Any,
                                                 <:LogEntropyPooling})
    (; sc1, sc2, so, slv) = opt
    model = Model()
    T = length(w)
    log_p = log.(w)
    # Decision variables (posterior probabilities)
    @variables(model, begin
                   x[1:T]
                   t
               end)
    # Equality constraints from A_eq and B_eq and probabilities equal to 1
    @constraints(model,
                 begin
                     sc1 * (sum(x) - one(eltype(w))) == 0
                     [sc1 * t; fill(sc1, T); sc1 * x] in MOI.RelativeEntropyCone(2 * T + 1)
                 end)
    if haskey(epc, :eq)
        A, B = epc[:eq]
        @constraint(model, ceq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :ineq)
        A, B = epc[:ineq]
        @constraint(model, cineq, sc1 * (A * x ⊖ B) <= 0)
    end
    if haskey(epc, :cvar_eq)
        A, B = epc[:cvar_eq]
        @constraint(model, ccvareq, sc1 * (A * x ⊖ B) == 0)
    end
    if haskey(epc, :feq)
        A, B = epc[:feq]
        N = length(B)
        @variables(model, begin
                       tc
                       c[1:N]
                   end)
        @constraints(model, begin
                         cfeq, sc1 * (A * x ⊖ B ⊖ c) == 0
                         [sc1 * tc; sc1 * c] in MOI.NormOneCone(N + 1)
                     end)
        add_to_expression!(obj_expr, so * sc2 * tc)
    end
    @objective(model, Min, so * (t - dot(x, log_p)))
    # Solve the optimization problem
    return if optimise_JuMP_model!(model, slv).success
        pweights(value.(x))
    else
        throw(ErrorException("Entropy pooling optimisation failed. Relax the views, use different solver parameters, or use a different prior."))
    end
end
function ep_cvar_views_solve!(cvar_views::Nothing, epc::AbstractDict, ::Any, ::Any, ::Real,
                              w::AbstractWeights, opt::AbstractEntropyPoolingOptimiser,
                              ::Any, ::Any; kwargs...)
    return entropy_pooling(w, epc, opt)
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:cvar}, alpha::Real)
    #! Don't use a view, use a copy, value at risk uses partialsort!
    #! Including pr.w needs the counterpart in ep_var_views! to be implemented.
    return ConditionalValueatRisk(; alpha = alpha)(pr.X[:, i])
end
function ep_cvar_views_solve!(cvar_views::LinearConstraintEstimator, epc::AbstractDict,
                              pr::AbstractPriorResult, sets::AssetSets, alpha::Real,
                              w::AbstractWeights, opt::AbstractEntropyPoolingOptimiser,
                              ds_opt::Union{Nothing, <:CVaREntropyPooling},
                              dm_opt::Union{Nothing, <:OptimEntropyPooling};
                              strict::Bool = false)
    cvar_views = parse_equation(cvar_views.val; ops1 = ("==",), ops2 = (:call, :(==)),
                                datatype = eltype(pr.X))
    cvar_views = replace_group_by_assets(cvar_views, sets, false, true, false)
    cvar_views = replace_prior_views(cvar_views, pr, sets, :cvar, alpha; strict = strict)
    lcs = get_linear_constraints(cvar_views, sets; datatype = eltype(pr.X), strict = strict)
    @argcheck(isnothing(lcs.ineq), "`cvar_view` can only have equality constraints.")
    if any(x -> x != 1, count(!iszero, lcs.A_eq; dims = 2))
        throw(ArgumentError("Cannot mix multiple assets in a single `cvar_view`."))
    end
    if any(x -> x < zero(eltype(x)), lcs.A_eq .* lcs.B_eq)
        throw(ArgumentError("`cvar_view` cannot be negative."))
    end
    idx = dropdims(.!iszero.(sum(lcs.A_eq; dims = 1)); dims = 1)
    idx2 = .!iszero.(lcs.A_eq)
    B = lcs.B_eq ./ view(lcs.A_eq, idx2)
    X = view(pr.X, :, idx)
    min_X = dropdims(-minimum(X; dims = 1); dims = 1)
    invalid = B .>= min_X
    if any(invalid)
        if !isa(cvar_views, AbstractVector)
            cvar_views = [cvar_views]
        end
        msg = "The following views are too extreme, the maximum viable view for a given asset is its worst realisation:"
        arr = [(v.eqn, m) for (v, m) in zip(cvar_views[invalid], min_X[invalid])]
        for (v, m) in arr
            msg *= "\n$v\t(> $m)."
        end
        msg *= "\nPlease lower the views or use a different prior with fatter tails."
        throw(ArgumentError(msg))
    end
    N = length(B)
    d_opt = if N == 1
        ifelse(!isnothing(ds_opt), ds_opt, CVaREntropyPooling())
    else
        ifelse(!isnothing(dm_opt), dm_opt,
               OptimEntropyPooling(;
                                   args = (Optim.Fminbox(),
                                           Optim.Options(; outer_x_abstol = 1e-4,
                                                         x_abstol = 1e-4))))
    end
    function func(etas)
        delete!(epc, :cvar_eq)
        @argcheck(all(zero(eltype(etas)) .<= etas .<= B))
        pos_part = max.(-X .- transpose(etas), zero(eltype(X)))
        add_ep_constraint!(epc, transpose(pos_part / alpha), B .- etas, :cvar_eq)
        wi = entropy_pooling(w, epc, opt)
        err = if N == 1
            sum(wi[.!iszero.(pos_part)]) - alpha
        else
            norm([ConditionalValueatRisk(; alpha = alpha, w = wi)(X[:, i]) - B[i]
                  for i in 1:N]) / sqrt(N)
        end
        return wi, err
    end
    return if N == 1
        try
            res = find_zero(x -> func(x)[2], (0, B[1]), d_opt.args...; d_opt.kwargs...)
            func([res])[1]
        catch e
            throw(ErrorException("CVaR entropy pooling optimisation failed. Relax the view, incrase alpha, use different solver parameters, use VaR views instead, or use a different prior.\n$(e)"))
        end
    else
        res = Optim.optimize(x -> func(x)[2], zeros(N), B, 0.5 * B, d_opt.args...;
                             d_opt.kwargs...)
        if Optim.converged(res)
            func(Optim.minimizer(res))[1]
        else
            throw(ErrorException("CVaR entropy pooling optimisation failed. Relax the view, incrase alpha, use different solver parameters, use VaR views instead, reduce the number of CVaR views, or use a different prior."))
        end
    end
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:sigma}, args...)
    return diag(pr.sigma)[i]
end
function ep_sigma_views!(sigma_views::LinearConstraintEstimator, epc::AbstractDict,
                         pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    sigma_views = parse_equation(sigma_views.val; datatype = eltype(pr.X))
    sigma_views = replace_group_by_assets(sigma_views, sets, false, true, false)
    sigma_views = replace_prior_views(sigma_views, pr, sets, :sigma; strict = strict)
    lcs = get_linear_constraints(sigma_views, sets; datatype = eltype(pr.X),
                                 strict = strict)
    tmp = transpose((pr.X .- transpose(pr.mu)) .^ 2)
    to_fix = falses(size(pr.X, 2))
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
function fix_sigma!(epc::AbstractDict, fixed::AbstractVector, to_fix::AbstractVector,
                    pr::AbstractPriorResult)
    sigma = diag(pr.sigma)
    fix = to_fix .& .!fixed
    if any(fix)
        add_ep_constraint!(epc, transpose(pr.X[:, fix] .- transpose(pr.mu[fix])) .^ 2,
                           sigma[fix], :feq)
        fixed .= fixed .| fix
    end
    return nothing
end
function replace_prior_views(res::ParsingResult, pr::AbstractPriorResult, sets::AssetSets;
                             strict::Bool = false)
    prior_pattern = r"prior\(([^()]*)\)"
    prior_corr_pattern = r"prior\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*,\s*([A-Za-z0-9_]+|\[[A-Za-z0-9_,\s]*\])\s*\)"
    nx = sets.dict[sets.key]
    variables, coeffs = res.vars, res.coef
    jk_idx = Vector{Union{Tuple{Int, Int}, Tuple{Vector{Int}, Vector{Int}}}}(undef, 0)
    idx_rm = Vector{Int}(undef, 0)
    rhs::typeof(res.rhs) = res.rhs
    non_prior = false
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            non_prior = true
            n = match(corr_pattern, v)
            if isnothing(n)
                throw(ArgumentError("Correlation prior view $(v) must be of the form `(a, b)`."))
            else
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                if startswith(asset1, "[") && endswith(asset1, "]")
                    asset1 = split(n.captures[1][2:(end - 1)], ", ")
                    asset2 = split(n.captures[2][2:(end - 1)], ", ")
                    j = [findfirst(x -> x == a1, nx) for a1 in asset1]
                    k = [findfirst(x -> x == a2, nx) for a2 in asset2]
                else
                    j = findfirst(x -> x == asset1, nx)
                    k = findfirst(x -> x == asset2, nx)
                    if isnothing(j)
                        msg = "Asset $(asset1) not found in $nx."
                        strict ? throw(ArgumentError(msg)) : @warn(msg)
                    end
                    if isnothing(k)
                        msg = "Asset $(asset2) not found in $nx."
                        strict ? throw(ArgumentError(msg)) : @warn(msg)
                    end
                    if isnothing(j) || isnothing(k)
                        push!(idx_rm, i)
                        continue
                    end
                end
                push!(jk_idx, (j, k))
            end
            continue
        end
        n = match(prior_corr_pattern, v)
        if isnothing(n)
            throw(ArgumentError("Correlation prior view $(v) must be of the form `prior(a, b)`."))
        else
            asset1 = n.captures[1]
            asset2 = n.captures[2]
            if startswith(asset1, "[") && endswith(asset1, "]")
                asset1 = split(n.captures[1][2:(end - 1)], ", ")
                asset2 = split(n.captures[2][2:(end - 1)], ", ")
                j = [findfirst(x -> x == a1, nx) for a1 in asset1]
                k = [findfirst(x -> x == a2, nx) for a2 in asset2]
            else
                j = findfirst(x -> x == asset1, nx)
                k = findfirst(x -> x == asset2, nx)
                if isnothing(j)
                    msg = "Asset $(asset1) not found in $nx."
                    strict ? throw(ArgumentError(msg)) : @warn(msg)
                end
                if isnothing(k)
                    msg = "Asset $(asset2) not found in $nx."
                    strict ? throw(ArgumentError(msg)) : @warn(msg)
                end
                if isnothing(j) || isnothing(k)
                    push!(idx_rm, i)
                    continue
                end
            end
            rhs -= get_pr_value(pr, j, k) * c
            push!(idx_rm, i)
        end
    end
    if isempty(idx_rm)
        return RhoParsingResult(res.vars, res.coef, res.op, res.rhs, res.eqn, jk_idx)
    end
    if !non_prior
        throw(ArgumentError("Priors in views are replaced by their prior value, thus they are essentially part of the constant of the view, so you need a non-prior view to serve as the variable.\n$(res)"))
    end
    idx = setdiff(1:length(variables), idx_rm)
    variables_new = variables[idx]
    coeffs_new = coeffs[idx]
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "))
    return RhoParsingResult(variables_new, coeffs_new, res.op, rhs,
                            "$(eqn) $(res.op) $(rhs)", jk_idx)
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, j::Integer, args...)
    return cov2cor(pr.sigma)[i, j]
end
function get_pr_value(pr::AbstractPriorResult, i::AbstractVector{<:Integer},
                      j::AbstractVector{<:Integer}, args...)
    return norm(cov2cor(pr.sigma)[i, j]) / length(i)
end
function ep_rho_views!(rho_views::LinearConstraintEstimator, epc::AbstractDict,
                       pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    rho_views = parse_equation(rho_views.val; datatype = eltype(pr.X))
    rho_views = replace_group_by_assets(rho_views, sets, false, true, true)
    rho_views = replace_prior_views(rho_views, pr, sets; strict = strict)
    to_fix = falses(size(pr.X, 2))
    sigma = diag(pr.sigma)
    for rho_view in rho_views
        @argcheck(length(rho_view.vars) == 1,
                  "Cannot mix multiple correlation pairs in a single view `$(rho_view.eqn)`.")
        @argcheck(-one(eltype(pr.X)) <= rho_view.rhs <= one(eltype(pr.X)),
                  "Correlation prior rho_view `$(rho_view.eqn)` must be in [-1, 1].")
        d = ifelse(rho_view.op == ">=", -1, 1)
        i, j = rho_view.ij[1]
        sigma_ij = if !isa(i, AbstractVector)
            sqrt(sigma[i] * sigma[j])
        else
            norm(sigma[i] .* sigma[j])
        end
        Ai = d * rho_view.coef[1] * view(pr.X, :, i) .* view(pr.X, :, j)
        Bi = d * pr.mu[i] ⊙ pr.mu[j] ⊕ rho_view.rhs ⊙ sigma_ij
        if !isa(i, AbstractVector)
            Bi = [Bi]
        end
        add_ep_constraint!(epc, transpose(Ai), Bi, ifelse(rho_view.op == "==", :eq, :ineq))
        to_fix[union(i, j)] .= true
    end
    return to_fix
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:skew}, args...)
    #! Think about how to include pr.w
    return Skewness()([1], reshape(pr.X[:, i], :, 1))
end
function ep_sk_views!(skew_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    skew_views = parse_equation(skew_views.val; datatype = eltype(pr.X))
    skew_views = replace_group_by_assets(skew_views, sets, false, true, false)
    skew_views = replace_prior_views(skew_views, pr, sets, :skew; strict = strict)
    lcs = get_linear_constraints(skew_views, sets; datatype = eltype(pr.X), strict = strict)
    sigma = diag(pr.sigma)
    tmp = transpose((pr.X .^ 3 .- transpose(pr.mu) .^ 3 .- 3 * transpose(pr.mu .* sigma)) ./
                    transpose(sigma .* sqrt.(sigma)))
    to_fix = falses(size(pr.X, 2))
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:kurtosis}, args...)
    #! Think about how to include pr.w
    return HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()))([1],
                                                                                      reshape(pr.X[:,
                                                                                                   i],
                                                                                              :,
                                                                                              1))
end
function ep_kt_views!(kurtosis_views::LinearConstraintEstimator, epc::AbstractDict,
                      pr::AbstractPriorResult, sets::AssetSets; strict::Bool = false)
    kurtosis_views = parse_equation(kurtosis_views.val; datatype = eltype(pr.X))
    kurtosis_views = replace_group_by_assets(kurtosis_views, sets, false, true, false)
    kurtosis_views = replace_prior_views(kurtosis_views, pr, sets, :kurtosis;
                                         strict = strict)
    lcs = get_linear_constraints(kurtosis_views, sets; datatype = eltype(pr.X),
                                 strict = strict)
    X_sq = pr.X .^ 2
    mu_sq = pr.mu .^ 2
    tmp = transpose((X_sq .* X_sq .- 4 * transpose(pr.mu) .* X_sq .* pr.X .+
                     6 * transpose(mu_sq) .* X_sq .- 3 * transpose(mu_sq .* mu_sq)) ./
                    transpose(diag(pr.sigma)) .^ 2)
    to_fix = falses(size(pr.X, 2))
    for p in propertynames(lcs)
        if isnothing(getproperty(lcs, p))
            continue
        end
        A = getproperty(lcs, p).A
        add_ep_constraint!(epc, A * tmp, getproperty(lcs, p).B, p)
        to_fix .= to_fix .| dropdims(any(.!iszero.(A); dims = 1); dims = 1)
    end
    return to_fix
end
function prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any,
                                       <:Union{<:H1_EntropyPooling, <:H2_EntropyPooling}},
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
    w1 = w0 = if isnothing(pe.w)
        pweights(range(; start = inv(T), stop = inv(T), length = T))
    else
        @argcheck(length(pe.w) == T)
        pweights(pe.w)
    end
    fixed = falses(N, 2)
    epc = Dict{Symbol, Tuple{<:AbstractMatrix, <:AbstractVector}}()

    # mu and VaR
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets, pe.var_alpha; strict = strict)
    if !isnothing(pe.mu_views) || !isnothing(pe.var_views) || !isnothing(pe.cvar_views)
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha, w0,
                                  pe.opt, pe.ds_opt, pe.dm_opt; strict = strict)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    # sigma
    if !isnothing(pe.sigma_views)
        to_fix = ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
        fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha,
                                  get_epw(pe.alg, w0, w1), pe.opt, pe.ds_opt, pe.dm_opt;
                                  strict = strict)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            to_fix = ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            to_fix = ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # rho
        if !isnothing(pe.rho_views)
            to_fix = ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha,
                                  get_epw(pe.alg, w0, w1), pe.opt, pe.ds_opt, pe.dm_opt;
                                  strict = strict)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    (; X, mu, sigma, chol, rr, f_mu, f_sigma) = pr
    ens = exp(entropy(w1))
    kld = kldivergence(w1, w0)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w1, ens = ens,
                         kld = kld, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = !isnothing(rr) ? w1 : nothing)
end
function prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:H0_EntropyPooling}, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    T = size(X, 1)
    w0 = if isnothing(pe.w)
        pweights(range(; start = inv(T), stop = inv(T), length = T))
    else
        @argcheck(length(pe.w) == T)
        pweights(pe.w)
    end
    epc = Dict{Symbol, Tuple{<:AbstractMatrix, <:AbstractVector}}()
    # mu and VaR
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets, pe.var_alpha; strict = strict)
    # sigma
    if !isnothing(pe.sigma_views)
        ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
        end
        # rho
        if !isnothing(pe.rho_views)
            ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
        end
    end
    w1 = ep_cvar_views_solve!(pe.cvar_views, epc, pr, pe.sets, pe.cvar_alpha, w0, pe.opt,
                              pe.ds_opt, pe.dm_opt; strict = strict)
    pe = factory(pe, w1)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    (; X, mu, sigma, chol, rr, f_mu, f_sigma) = pr
    ens = exp(entropy(w1))
    kld = kldivergence(w1, w0)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w1, ens = ens,
                         kld = kld, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = !isnothing(rr) ? w1 : nothing)
end

export LogEntropyPooling, ExpEntropyPooling, EntropyPoolingPrior, H0_EntropyPooling,
       H1_EntropyPooling, H2_EntropyPooling, JuMPEntropyPooling, OptimEntropyPooling
