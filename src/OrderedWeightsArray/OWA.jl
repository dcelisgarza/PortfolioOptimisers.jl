function owa_gmd(T::Integer)
    # w = Vector{typeof(inv(T))}(undef, T)
    # for i ∈ eachindex(w)
    #     w[i] = 2 * i - 1 - T
    # end
    # w = 2 / (T * (T - 1)) * w
    return collect((4 * range(1; stop = T) .- 2 * (T + 1)) / (T * (T - 1)))
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
    return owa_cvar(T, alpha) .- reverse(owa_cvar(T, beta))
end
function owa_wcvarrg(T::Integer, alphas::AbstractVector{<:Real},
                     weights_a::AbstractVector{<:Real},
                     betas::AbstractVector{<:Real} = alphas,
                     weights_b::AbstractVector{<:Real} = weights_a)
    w = owa_wcvar(T, alphas, weights_a) .- reverse(owa_wcvar(T, betas, weights_b))
    return w
end
function owa_tgrg(T::Integer; alpha_i::Real = 0.0001, alpha::Real = 0.05,
                  a_sim::Integer = 100, beta_i::Real = alpha_i, beta::Real = alpha,
                  b_sim::Integer = a_sim)
    w = owa_tg(T; alpha_i = alpha_i, alpha = alpha, a_sim = a_sim) .-
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
                          method::AbstractOrderedWeightsArray = OWA_NCRRA())
    @smart_assert(k >= 2)
    rg = 2:k
    weights = Matrix{typeof(inv(T * k))}(undef, T, length(rg))
    for i ∈ rg
        wi = (-1)^i * owa_l_moment(T, i)
        weights[:, i - 1] .= wi
    end
    return owa_l_moment_crm(method, weights)
end

export owa_gmd, owa_cvar, owa_wcvar, owa_tg, owa_wr, owa_rg, owa_cvarrg, owa_wcvarrg,
       owa_tgrg, owa_l_moment, owa_l_moment_crm
