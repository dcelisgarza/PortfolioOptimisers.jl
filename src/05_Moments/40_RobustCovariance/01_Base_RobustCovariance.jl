"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the covariance estimators that resist outliers in the returns.

A bad tick, a split that the data source does not record, or one extreme day moves the sample covariance much. A robust estimator gives each observation a weight that falls with its distance from the centre of the data, so a few bad rows cannot control the estimate. [`MCDCovariance`](@ref) and [`TylerCovariance`](@ref) are its subtypes.

# Interfaces

A subtype implements one method:

  - `StatsAPI.fit(ce, X::MatNum; dims::Int = 1, mean = nothing, kwargs...) -> RobustCovarianceResult`: Fits the robust location and scatter, and returns them with the distance, the weight and the outlier flag of each observation in a [`RobustCovarianceResult`](@ref).

The family then gives `Statistics.cov(ce, X; dims, mean)`, which returns the `sigma` of the fit, and `Statistics.cor(ce, X; dims, mean)`, which rescales it to a correlation matrix. The fallbacks of `var` and `std` read the diagonal of `cov`. So every consumer of a covariance estimator, such as a prior, a distance or an uncertainty set, reads the robust covariance with no change. [`RobustExpectedReturns`](@ref) reads the `mu` of the fit.

# Related

  - [`MCDCovariance`](@ref)
  - [`TylerCovariance`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`RobustExpectedReturns`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractRobustCovarianceEstimator <: AbstractCovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Holds the fit of a robust covariance estimator: the location, the scatter, and the squared distance, the weight and the outlier flag of each observation.

`StatsAPI.fit` of an [`AbstractRobustCovarianceEstimator`](@ref) returns it. The weight of an observation is the weight that it has in the estimate of `sigma`. The flag marks an observation whose squared distance is above the quantile `q` of the ``\\chi^2_p`` distribution, where `q` is the field of the estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RobustCovarianceResult(;
        mu::VecNum,
        sigma::MatNum,
        d2::VecNum,
        w::VecNum,
        outlier::AbstractVector{Bool}
    ) -> RobustCovarianceResult

Keywords correspond to the struct's fields.

## Validation

  - `sigma` is square, and its size is the length of `mu`.
  - `d2`, `w` and `outlier` have one length, the number of observations.

# Examples

```jldoctest
julia> r = RobustCovarianceResult(; mu = [0.0, 0.0], sigma = [1.0 0.0; 0.0 1.0], d2 = [0.5, 9.0],
                                  w = [1.0, 0.0], outlier = BitVector([0, 1]));

julia> r.outlier
2-element BitVector:
 0
 1
```

# Related

  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`MCDCovariance`](@ref)
  - [`TylerCovariance`](@ref)
"""
@concrete struct RobustCovarianceResult <: AbstractResult
    """
    $(field_dict[:rob_mu])
    """
    mu
    """
    $(field_dict[:rob_sigma])
    """
    sigma
    """
    $(field_dict[:rob_d2])
    """
    d2
    """
    $(field_dict[:rob_w])
    """
    w
    """
    $(field_dict[:rob_outlier])
    """
    outlier
    function RobustCovarianceResult(mu::VecNum, sigma::MatNum, d2::VecNum, w::VecNum,
                                    outlier::AbstractVector{Bool})
        @argcheck(size(sigma) == (length(mu), length(mu)),
                  DimensionMismatch("sigma must be square, with the size of mu. Got size(sigma) => $(size(sigma)) and length(mu) => $(length(mu))"))
        @argcheck(length(d2) == length(w) == length(outlier),
                  DimensionMismatch("d2, w and outlier must have one length. Got $(length(d2)), $(length(w)) and $(length(outlier))"))
        return new{typeof(mu), typeof(sigma), typeof(d2), typeof(w), typeof(outlier)}(mu,
                                                                                      sigma,
                                                                                      d2, w,
                                                                                      outlier)
    end
end
function RobustCovarianceResult(; mu::VecNum, sigma::MatNum, d2::VecNum, w::VecNum,
                                outlier::AbstractVector{Bool})::RobustCovarianceResult
    return RobustCovarianceResult(mu, sigma, d2, w, outlier)
end
"""
    Statistics.cov(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1,
                   mean = nothing, kwargs...) -> Matrix

Returns the robust covariance matrix, the `sigma` of `StatsAPI.fit(ce, X; dims, mean, kwargs...)`.

# Arguments

  - `ce`: The robust covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) A given centre fixes the location of the fit.
  - `kwargs...`: More keyword arguments for the fit.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`cor(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return StatsAPI.fit(ce, X; dims = dims, mean = mean, kwargs...).sigma
end
"""
    Statistics.cor(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1,
                   mean = nothing, kwargs...) -> Matrix

Returns the robust correlation matrix, the robust covariance matrix rescaled with [`cov2cor_nan!`](@ref).

# Arguments

  - `ce`: The robust covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) A given centre fixes the location of the fit.
  - `kwargs...`: More keyword arguments for the fit.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`cov(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::AbstractRobustCovarianceEstimator, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return cov2cor_nan!(Statistics.cov(ce, X; dims = dims, mean = mean, kwargs...))
end
"""
    robust_centre(me::Option{<:AbstractExpectedReturnsEstimator}, X::MatNum; mean = nothing,
                  kwargs...) -> Option{<:VecNum}

Returns the fixed centre of a robust fit, or `nothing` for a joint location.

The `mean` keyword wins over `me`, as for [`Covariance`](@ref). With neither, the fit estimates its own location.

# Arguments

  - `me`: The expected returns estimator of the robust estimator, or `nothing`.
  - `X`: The data matrix, `observations × assets`.
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for the expected returns estimator.

# Returns

  - `mu::Option{<:VecNum}`: The fixed centre, or `nothing`.

# Related

  - [`MCDCovariance`](@ref)
  - [`TylerCovariance`](@ref)
"""
function robust_centre(me::Option{<:AbstractExpectedReturnsEstimator}, X::MatNum;
                       mean = nothing, kwargs...)
    return if !isnothing(mean)
        vec(mean)
    elseif isnothing(me)
        nothing
    else
        vec(Statistics.mean(me, X; dims = 1, kwargs...))
    end
end
"""
    assert_robust_rows(n::Integer, p::Integer, what::AbstractString)

Refuses a sample with too few observations for a robust scatter.

A robust scatter of `p` assets needs more than `p` observations in the set that defines it. Otherwise the matrix is singular. [`MCDCovariance`](@ref) passes the size `h` of its subset, and [`TylerCovariance`](@ref) passes the number of observations.

# Arguments

  - `n`: The number of observations that define the scatter.
  - `p`: The number of assets.
  - `what`: The name of `n` in the message.

# Validation

  - `n > p`. Otherwise, the function raises a `DomainError` that names `n`, `p` and the limit.

# Returns

  - `nothing`.

# Related

  - [`MCDCovariance`](@ref)
  - [`TylerCovariance`](@ref)
"""
function assert_robust_rows(n::Integer, p::Integer, what::AbstractString)
    @argcheck(n > p,
              DomainError((n, p),
                          "a robust scatter of p assets needs $(what) to be larger than p, got $(what) = $n and p = $p. Use a longer window or fewer assets"))
    return nothing
end
"""
    robust_result(mu::VecNum, sigma::MatNum, d2::VecNum, w::VecNum, q::Number) -> RobustCovarianceResult

Builds the [`RobustCovarianceResult`](@ref) of a robust fit.

The function flags each observation whose squared distance is above the quantile `q` of the ``\\chi^2_p`` distribution.

# Arguments

  - `mu`: The robust location.
  - `sigma`: The robust scatter.
  - `d2`: The squared distance of each observation, from [`squared_distances`](@ref).
  - `w`: The weight of each observation in the estimate of `sigma`.
  - `q`: The probability of the cut-off of the outlier flag.

# Returns

  - `res::RobustCovarianceResult`: The fit.

# Related

  - [`RobustCovarianceResult`](@ref)
  - [`squared_distances`](@ref)
"""
function robust_result(mu::VecNum, sigma::MatNum, d2::VecNum, w::VecNum, q::Number)
    cut = Distributions.quantile(Distributions.Chisq(length(mu)), q)
    return RobustCovarianceResult(mu, sigma, d2, w, d2 .> cut)
end

export RobustCovarianceResult
public AbstractRobustCovarianceEstimator
