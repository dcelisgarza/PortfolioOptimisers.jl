struct DistanceCovariance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple,
                          T4 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractCovarianceEstimator
    dist::T1
    args::T2
    kwargs::T3
    w::T4
end
function DistanceCovariance(; dist::Distances.Metric = Distances.Euclidean(),
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            w::Union{Nothing, <:AbstractWeights} = nothing)
    return DistanceCovariance{typeof(dist), typeof(args), typeof(kwargs), typeof(w)}(dist,
                                                                                     args,
                                                                                     kwargs,
                                                                                     w)
end
function factory(ce::DistanceCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return DistanceCovariance(; dist = ce.dist, args = ce.args, kwargs = ce.kwargs,
                              w = isnothing(w) ? ce.w : w)
end
function cor_distance(ce::DistanceCovariance, v1::AbstractVector, v2::AbstractVector)
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)
    N2 = N^2
    a, b = if isnothing(ce.w)
        Distances.pairwise(ce.dist, v1, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2, ce.args...; ce.kwargs...)
    else
        Distances.pairwise(ce.dist, v1 ⊙ ce.w, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2 ⊙ ce.w, ce.args...; ce.kwargs...)
    end
    mu_a1, mu_b1 = mean(a; dims = 1), mean(b; dims = 1)
    mu_a2, mu_b2 = mean(a; dims = 2), mean(b; dims = 2)
    mu_a3, mu_b3 = mean(a), mean(b)
    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3
    dcov2_xx = sum(A ⊙ A) / N2
    dcov2_xy = sum(A ⊙ B) / N2
    dcov2_yy = sum(B ⊙ B) / N2
    return sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))
end
function cor_distance(ce::DistanceCovariance, X::AbstractMatrix)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    for j ∈ axes(X, 2)
        xj = view(X, :, j)
        for i ∈ 1:j
            rho[j, i] = rho[i, j] = cor_distance(ce, view(X, :, i), xj)
        end
    end
    return rho
end
function StatsBase.cor(ce::DistanceCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cor_distance(ce, X)
end
function cov_distance(ce::DistanceCovariance, v1::AbstractVector, v2::AbstractVector)
    N = length(v1)
    @smart_assert(N == length(v2) && N > 1)
    N2 = N^2
    a, b = if isnothing(ce.w)
        Distances.pairwise(ce.dist, v1, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2, ce.args...; ce.kwargs...)
    else
        Distances.pairwise(ce.dist, v1 ⊙ ce.w, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2 ⊙ ce.w, ce.args...; ce.kwargs...)
    end
    mu_a1, mu_b1 = mean(a; dims = 1), mean(b; dims = 1)
    mu_a2, mu_b2 = mean(a; dims = 2), mean(b; dims = 2)
    mu_a3, mu_b3 = mean(a), mean(b)
    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3
    dcov2_xy = sum(A ⊙ B) / N2
    return sqrt(dcov2_xy)
end
function cov_distance(ce::DistanceCovariance, X::AbstractMatrix)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    for j ∈ axes(X, 2)
        xj = view(X, :, j)
        for i ∈ 1:j
            rho[j, i] = rho[i, j] = cov_distance(ce, view(X, :, i), xj)
        end
    end
    return rho
end
function StatsBase.cov(ce::DistanceCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cov_distance(ce, X)
end

export DistanceCovariance
