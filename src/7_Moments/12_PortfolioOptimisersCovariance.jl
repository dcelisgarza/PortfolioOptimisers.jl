struct PortfolioOptimisersCovariance{T1 <: AbstractCovarianceEstimator,
                                     T2 <: AbstractMatrixProcessingEstimator} <:
       AbstractCovarianceEstimator
    ce::T1
    mp::T2
end
function PortfolioOptimisersCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                                       mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())
    return PortfolioOptimisersCovariance{typeof(ce), typeof(mp)}(ce, mp)
end
#=
function Base.show(io::IO, ce::PortfolioOptimisersCovariance)
    println(io, "PortfolioOptimisersCovariance")
    for field in fieldnames(typeof(ce))
        val = getfield(ce, field)
        print(io, "  ", string(field), " ")
        ioalg = IOBuffer()
        show(ioalg, val)
        algstr = String(take!(ioalg))
        alglines = split(algstr, '\n')
        println(io, "| ", alglines[1])
        for l in alglines[2:end]
            println(io, "     | ", l)
        end
    end
end
=#
function factory(ce::PortfolioOptimisersCovariance,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return PortfolioOptimisersCovariance(; ce = factory(ce.ce, w), mp = ce.mp)
end
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                        kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                        kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end

export PortfolioOptimisersCovariance
