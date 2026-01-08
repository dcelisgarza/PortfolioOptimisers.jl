function Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
    val = LinearAlgebra.diag(Statistics.cov(ce, X; dims = dims, kwargs...))
    return isone(dims) ? reshape(val, 1, length(val)) : reshape(val, length(val), 1)
end
function Statistics.std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
    val = sqrt.(LinearAlgebra.diag(Statistics.cov(ce, X; dims = dims, kwargs...)))
    return isone(dims) ? reshape(val, 1, length(val)) : reshape(val, length(val), 1)
end
