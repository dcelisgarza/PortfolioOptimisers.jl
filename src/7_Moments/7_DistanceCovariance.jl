struct DistanceCovariance{T1 <: Distances.Metric, T2 <: Tuple, T3 <: NamedTuple,
                          T4 <: Union{Nothing, <:AbstractWeights},
                          T5 <: FLoops.Transducers.Executor} <: AbstractCovarianceEstimator
    dist::T1
    args::T2
    kwargs::T3
    w::T4
    threads::T5
end
function DistanceCovariance(; dist::Distances.Metric = Distances.Euclidean(),
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            threads::FLoops.Transducers.Executor = ThreadedEx())
    return DistanceCovariance{typeof(dist), typeof(args), typeof(kwargs), typeof(w),
                              typeof(threads)}(dist, args, kwargs, w, threads)
end
function factory(ce::DistanceCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return DistanceCovariance(; dist = ce.dist, args = ce.args, kwargs = ce.kwargs,
                              w = isnothing(w) ? ce.w : w, threads = ce.threads)
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
    dcov2_xx = dot(A, A) / N2
    dcov2_xy = dot(A, B) / N2
    dcov2_yy = dot(B, B) / N2
    return sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))
end
function cor_distance(ce::DistanceCovariance, X::AbstractMatrix)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    @floop ce.threads for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            rho[j, i] = rho[i, j] = cor_distance(ce, view(X, :, i), xj)
        end
    end
    return rho
end
function Statistics.cor(ce::DistanceCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
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
    dcov2_xy = dot(A, B) / N2
    return sqrt(dcov2_xy)
end
function cov_distance(ce::DistanceCovariance, X::AbstractMatrix)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    @floop ce.threads for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            rho[j, i] = rho[i, j] = cov_distance(ce, view(X, :, i), xj)
        end
    end
    return rho
end
function Statistics.cov(ce::DistanceCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cov_distance(ce, X)
end

export DistanceCovariance
