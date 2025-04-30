abstract type AbstractOrderedWeightsArrayEstimator <: AbstractEstimator end
abstract type AbstractOrderedWeightsArrayAlgorithm <: AbstractAlgorithm end
struct MaximumEntropy <: AbstractOrderedWeightsArrayAlgorithm end
struct MinimumSquareDistance <: AbstractOrderedWeightsArrayAlgorithm end
struct MinimumSumSquares <: AbstractOrderedWeightsArrayAlgorithm end
struct NormalisedConstantRelativeRiskAversion{T1 <: Real} <:
       AbstractOrderedWeightsArrayEstimator
    g::T1
end
struct OWAJuMPEstimator{T1 <: AbstractOrderedWeightsArrayAlgorithm, T2 <: Real, T3 <: Real,
                        T4 <: Real, T5 <: Union{<:Solver, <:AbstractVector{<:Solver}}} <:
       AbstractOrderedWeightsArrayEstimator
    alg::T1
    max_phi::T2
    sc::T3
    so::T4
    slv::T5
end
function OWAJuMPEstimator(; alg::AbstractOrderedWeightsArrayAlgorithm = MaximumEntropy(),
                          max_phi::Real = 0.5, sc::Real = 1.0, so::Real = 1.0,
                          slv::Union{<:Solver, <:AbstractVector{<:Solver}} = Solver())
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(max_phi) < max_phi < one(max_phi))
    @smart_assert(sc > zero(sc))
    @smart_assert(so > zero(so))
    return OWAJuMPEstimator{typeof(alg), typeof(max_phi), typeof(sc), typeof(so),
                            typeof(slv)}(alg, max_phi, sc, so, slv)
end
function NormalisedConstantRelativeRiskAversion(; g::Real = 0.5)
    @smart_assert(zero(g) < g < one(g))
    return NormalisedConstantRelativeRiskAversion{typeof(g)}(g)
end
function ncrra_weights(weights::AbstractMatrix{<:Real}, g::Real)
    N = size(weights, 2)
    phis = Vector{eltype(weights)}(undef, N)
    e = 1
    for i ∈ eachindex(phis)
        e *= g + i - 1
        phis[i] = e / factorial(i + 1)
    end
    phis ./= sum(phis)
    a = weights * phis
    w = similar(a)
    w[1] = a[1]
    for i ∈ 2:length(a)
        w[i] = maximum(a[1:i])
    end
    return w
end
function owa_l_moment_crm(method::NormalisedConstantRelativeRiskAversion,
                          weights::AbstractMatrix{<:Real})
    return ncrra_weights(weights, method.g)
end
function owa_model_setup(method::OWAJuMPEstimator, weights::AbstractMatrix{<:Real})
    T, N = size(weights)
    model = JuMP.Model()
    max_phi = method.max_phi
    sc = method.sc
    @variables(model, begin
                   theta[1:T]
                   phi[1:N]
               end)
    @constraints(model, begin
                     sc * phi >= 0
                     sc * (phi .- max_phi) <= 0
                     sc * (sum(phi) - 1) == 0
                     sc * (theta - weights * phi) == 0
                     sc * (phi[2:end] - phi[1:(end - 1)]) <= 0
                     sc * (theta[2:end] - theta[1:(end - 1)]) >= 0
                 end)
    return model
end
function owa_model_solve(model::JuMP.Model, method::OWAJuMPEstimator,
                         weights::AbstractMatrix)
    slv = method.slv
    success = optimise_JuMP_model!(model, slv).success
    return if success
        phi = model[:phi]
        phis = value.(phi)
        phis ./= sum(phis)
        w = weights * phis
    else
        @warn("Type: $method\nReverting to ncrra_weights.")
        w = ncrra_weights(weights, 0.5)
    end
end
function owa_l_moment_crm(method::OWAJuMPEstimator{<:MaximumEntropy, <:Any, <:Any, <:Any,
                                                   <:Any}, weights::AbstractMatrix{<:Real})
    T = size(weights, 1)
    sc = method.sc
    so = method.so
    ovec = range(; start = sc, stop = sc, length = T)
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variables(model, begin
                   t
                   x[1:T]
               end)
    @constraints(model, begin
                     sc * (sum(x) - 1) == 0
                     [sc * t; ovec; sc * x] ∈ MOI.RelativeEntropyCone(2 * T + 1)
                     [i = 1:T], [sc * x[i]; sc * theta[i]] ∈ MOI.NormOneCone(2)
                 end)
    @objective(model, Max, -so * t)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMPEstimator{<:MinimumSquareDistance, <:Any, <:Any,
                                                   <:Any, <:Any},
                          weights::AbstractMatrix{<:Real})
    sc = method.sc
    so = method.so
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model,
                [sc * t; sc * (theta[2:end] - theta[1:(end - 1)])] ∈ SecondOrderCone())
    @objective(model, Min, so * t)
    return owa_model_solve(model, method, weights)
end
function owa_l_moment_crm(method::OWAJuMPEstimator{<:MinimumSumSquares, <:Any, <:Any, <:Any,
                                                   <:Any}, weights::AbstractMatrix{<:Real})
    sc = method.sc
    so = method.so
    model = owa_model_setup(method, weights)
    theta = model[:theta]
    @variable(model, t)
    @constraint(model, [sc * t; sc * theta] ∈ SecondOrderCone())
    @objective(model, Min, so * t)
    return owa_model_solve(model, method, weights)
end
function owa_gmd(T::Integer)
    # w = Vector{typeof(inv(T))}(undef, T)
    # for i ∈ eachindex(w)
    #     w[i] = 2 * i - 1 - T
    # end
    # w = 2 / (T * (T - 1)) * w
    return (4 * range(1; stop = T) .- 2 * (T + 1)) / (T * (T - 1))
end
function owa_cvar(T::Integer, alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    k = floor(Int, T * alpha)
    w = zeros(typeof(alpha), T)
    w[1:k] .= -1 / (T * alpha)
    w[k + 1] = -1 - sum(w[1:k])
    return w
end
function owa_wcvar(T::Integer, alphas::AbstractVector{<:Real},
                   weights::AbstractVector{<:Real})
    w = zeros(promote_type(eltype(alphas), eltype(weights)), T)
    for (i, j) ∈ zip(alphas, weights)
        w .+= owa_cvar(T, i) * j
    end
    return w
end
function owa_tg(T::Integer; alpha_i::Real = 1e-4, alpha::Real = 0.05, a_sim::Integer = 100)
    @smart_assert(zero(alpha) < alpha_i < alpha < one(alpha))
    @smart_assert(a_sim > zero(a_sim))
    alphas = range(; start = alpha_i, stop = alpha, length = a_sim)
    n = length(alphas)
    w = Vector{typeof(alpha)}(undef, n)
    w[1] = alphas[2] * alphas[1] / alphas[n]^2
    for i ∈ 2:(n - 1)
        w[i] = (alphas[i + 1] - alphas[i - 1]) * alphas[i] / alphas[n]^2
    end
    w[n] = (alphas[n] - alphas[n - 1]) / alphas[n]
    return owa_wcvar(T, alphas, w)
end
function owa_wr(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    return w
end
function owa_rg(T::Integer)
    w = zeros(typeof(inv(T)), T)
    w[1] = -1
    w[T] = 1
    return w
end
function owa_cvarrg(T::Integer; alpha::Real = 0.05, beta::Real = alpha)
    return owa_cvar(T, alpha) - reverse(owa_cvar(T, beta))
end
function owa_wcvarrg(T::Integer, alphas::AbstractVector{<:Real},
                     weights_a::AbstractVector{<:Real},
                     betas::AbstractVector{<:Real} = alphas,
                     weights_b::AbstractVector{<:Real} = weights_a)
    w = owa_wcvar(T, alphas, weights_a) - reverse(owa_wcvar(T, betas, weights_b))
    return w
end
function owa_tgrg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05,
                  a_sim::Integer = 100, beta_i::Real = alpha_i, beta::Real = alpha,
                  b_sim::Integer = a_sim)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) -
        reverse(owa_tg(T; alpha_i = beta_i, alpha = beta, a_sim = b_sim))

    return w
end
function owa_l_moment(T::Integer, k::Integer = 2)
    T, k = promote(T, k)
    w = Vector{typeof(inv(T * k))}(undef, T)
    for i ∈ eachindex(w)
        a = zero(k)
        for j ∈ 0:(k - 1)
            a += (-1)^j *
                 binomial(k - 1, j) *
                 binomial(i - 1, k - 1 - j) *
                 binomial(T - i, j)
        end
        a *= 1 / (k * binomial(T, k))
        w[i] = a
    end
    return w
end
function owa_l_moment_crm(T::Integer; k::Integer = 2,
                          method::AbstractOrderedWeightsArrayEstimator = NormalisedConstantRelativeRiskAversion())
    @smart_assert(k >= 2)
    rg = 2:k
    weights = Matrix{typeof(inv(T * k))}(undef, T, length(rg))
    for i ∈ rg
        wi = (-1)^i * owa_l_moment(T, i)
        weights[:, i - 1] .= wi
    end
    return owa_l_moment_crm(method, weights)
end

export MaximumEntropy, MinimumSquareDistance, MinimumSumSquares,
       NormalisedConstantRelativeRiskAversion, OWAJuMPEstimator, owa_gmd, owa_cvar,
       owa_wcvar, owa_tg, owa_wr, owa_rg, owa_cvarrg, owa_wcvarrg, owa_tgrg, owa_l_moment,
       owa_l_moment_crm
