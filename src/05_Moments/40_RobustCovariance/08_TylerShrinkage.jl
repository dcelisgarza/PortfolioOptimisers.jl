"""
    observation_weight_vector(X::MatNum, w::Option{<:StatsBase.AbstractWeights}) -> Vector

Returns the weight of each observation of `X` as a vector, a vector of ones when `w` is `nothing`.

# Arguments

  - `X`: The data, `observations × assets`.
  - `w`: The observation weights, or `nothing`.

# Returns

  - `wt::Vector`: The weight of each observation.

# Related

  - [`tyler_fixed_point`](@ref)
"""
function observation_weight_vector(X::MatNum, ::Nothing)
    return fill(one(eltype(X)), size(X, 1))
end
function observation_weight_vector(::MatNum, w::StatsBase.AbstractWeights)
    return map(i -> w[i], 1:length(w))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the form and the intensity of the shrinkage of one Tyler shape.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShapeShrinkage(
        alg::AbstractTylerShrinkageAlgorithm,
        rho::Number
    ) -> ShapeShrinkage

# Related

  - [`tyler_fixed_point`](@ref)
  - [`shrink_shape!`](@ref)
"""
@concrete struct ShapeShrinkage
    """
    $(field_dict[:tyler_shr_alg])
    """
    alg
    """
    $(field_dict[:rob_rho_fit])
    """
    rho
end
"""
    shrink_shape!(alg::TraceNormalisedShrinkage, V::MatNum, rho::Number) -> MatNum
    shrink_shape!(alg::PenalisedShrinkage, V::MatNum, rho::Number) -> MatNum

Blends one step `V` of the shape equation with the identity matrix in place.

[`TraceNormalisedShrinkage`](@ref) then scales the blend to the trace `p`, and [`PenalisedShrinkage`](@ref) does not.

# Arguments

  - `alg`: The form of the shrinkage.
  - `V`: The step of the shape equation, from [`tyler_shape_step`](@ref).
  - `rho`: The shrinkage intensity.

# Returns

  - `V::MatNum`: The input matrix.

# Related

  - [`tyler_fixed_point`](@ref)
  - [`AbstractTylerShrinkageAlgorithm`](@ref)
"""
function shrink_shape!(::PenalisedShrinkage, V::MatNum, rho::Number)
    V .*= one(rho) - rho
    view(V, LinearAlgebra.diagind(V)) .+= rho
    return V
end
function shrink_shape!(::TraceNormalisedShrinkage, V::MatNum, rho::Number)
    return trace_normalise!(shrink_shape!(PenalisedShrinkage(), V, rho))
end
"""
    tyler_shape(reg::Option{<:ShapeShrinkage}, W::MatNum,
                w::Option{<:StatsBase.AbstractWeights}) -> Matrix

Returns the Tyler shape of the centred data `W`, with the centre fixed at zero.

The rules of the intensity read it.

# Arguments

  - `reg`: The shrinkage, or `nothing`.
  - `W`: The centred data.
  - `w`: The observation weights, or `nothing`.

# Returns

  - `V::Matrix`: The shape.

# Related

  - [`tyler_fixed_point`](@ref)
  - [`tyler_intensity`](@ref)
"""
function tyler_shape(reg::Option{<:ShapeShrinkage}, W::MatNum,
                     w::Option{<:StatsBase.AbstractWeights})
    return last(tyler_fixed_point(W, w, zeros(eltype(W), size(W, 2)), reg))
end
"""
    tyler_intensity_bound(W::MatNum, w::Option{<:StatsBase.AbstractWeights}) -> Number

Returns the intensity above which the shape of [`PenalisedShrinkage`](@ref) exists.

In general position a subspace of dimension ``r`` holds at most ``r`` observations. Theorem 3 of [ollila2014](@cite) then needs ``(1 - \\rho) p`` times the largest share of the weight of one observation below one. With equal weights the bound is ``1 - n/p``. A negative bound means that the shape exists with no shrinkage. With `n == p` the bound is zero, and the shape does not exist with no shrinkage.

# Arguments

  - `W`: The centred data.
  - `w`: The observation weights, or `nothing`.

# Returns

  - `lo::Number`: The bound, below one.

# Related

  - [`PenalisedShrinkage`](@ref)
  - [`assert_tyler_intensity`](@ref)

# References

  - $(ref_dict[:ollila2014]) Section V, Theorem 3.
"""
function tyler_intensity_bound(W::MatNum, w::Option{<:StatsBase.AbstractWeights})
    wt = observation_weight_vector(W, w)
    return one(eltype(W)) - sum(wt) / (size(W, 2) * maximum(wt))
end
"""
    assert_tyler_intensity(alg::TraceNormalisedShrinkage, rho::Number, lo::Number)
    assert_tyler_intensity(alg::PenalisedShrinkage, rho::Number, lo::Number)

Refuses a shrinkage intensity at which the shape does not exist.

# Arguments

  - `alg`: The form of the shrinkage.
  - `rho`: The intensity.
  - `lo`: The bound of [`tyler_intensity_bound`](@ref).

# Validation

  - [`TraceNormalisedShrinkage`](@ref): `rho > 0`, or `lo` is negative.
  - [`PenalisedShrinkage`](@ref): `rho > lo`.

Otherwise, the function raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`tyler_intensity_bound`](@ref)
"""
function assert_tyler_intensity(::TraceNormalisedShrinkage, rho::Number, lo::Number)
    @argcheck(rho > zero(rho) || lo < zero(lo),
              DomainError(rho,
                          "the Tyler shape does not exist with no shrinkage on this sample. Use an intensity above zero, a longer window or fewer assets"))
    return nothing
end
function assert_tyler_intensity(::PenalisedShrinkage, rho::Number, lo::Number)
    @argcheck(rho > lo,
              DomainError(rho,
                          "`PenalisedShrinkage` needs an intensity above $lo on this sample, got $rho. Use `TraceNormalisedShrinkage`, or a larger intensity"))
    return nothing
end
"""
    smallest_intensity(reaches, lo::Number) -> Number

Returns the smallest intensity in `(max(lo, 0), 1]` at which `reaches` is `true`, up to the width ``2^{-10}`` of the last interval.

The function bisects the interval ten times. `reaches(1)` must be `true`, and the function does not call it there. The returned intensity is an end of an interval at which `reaches` is `true`.

# Arguments

  - `reaches`: A function of the intensity that returns a `Bool`.
  - `lo`: The lower end of the interval. A negative value stands for zero.

# Returns

  - `rho::Number`: The intensity.

# Related

  - [`tyler_intensity`](@ref)
"""
function smallest_intensity(reaches, lo::Number)
    lo = max(lo, zero(lo))
    hi = one(lo)
    for _ in 1:10
        mid = (lo + hi) / 2
        if reaches(mid)
            hi = mid
        else
            lo = mid
        end
    end
    return hi
end
"""
    penalised_intensity(weight::Number, W::MatNum, w::Option{<:StatsBase.AbstractWeights}) -> Number

Returns the intensity of [`PenalisedShrinkage`](@ref) at which the target has a given weight in the shape of trace `p`.

The shape has no fixed trace, so the weight of the target in it is the intensity over the mean of its diagonal.

# Arguments

  - `weight`: The weight of the target.
  - `W`: The centred data.
  - `w`: The observation weights, or `nothing`.

# Returns

  - `rho::Number`: The intensity, one when `weight` is one or more.

# Related

  - [`AsymptoticIntensity`](@ref)
  - [`SphericityIntensity`](@ref)

# References

  - $(ref_dict[:couillet2014]) Proposition 2.
"""
function penalised_intensity(weight::Number, W::MatNum,
                             w::Option{<:StatsBase.AbstractWeights})
    p = size(W, 2)
    if weight >= one(weight)
        return one(weight)
    end
    function reaches(rho)
        V = tyler_shape(ShapeShrinkage(PenalisedShrinkage(), rho), W, w)
        return rho * p >= weight * LinearAlgebra.tr(V)
    end
    return smallest_intensity(reaches, tyler_intensity_bound(W, w))
end
"""
    tyler_intensity(rho::Number, alg::AbstractTylerShrinkageAlgorithm, W::MatNum,
                    w::Option{<:StatsBase.AbstractWeights})
    tyler_intensity(rule::ConditionNumberIntensity, alg::AbstractTylerShrinkageAlgorithm,
                    W::MatNum, w::Option{<:StatsBase.AbstractWeights})
    tyler_intensity(rule::SphericityIntensity, alg::AbstractTylerShrinkageAlgorithm,
                    W::MatNum, w::Option{<:StatsBase.AbstractWeights})
    tyler_intensity(rule::OracleIntensity, alg::AbstractTylerShrinkageAlgorithm, W::MatNum,
                    w::Option{<:StatsBase.AbstractWeights})
    tyler_intensity(rule::AsymptoticIntensity, alg::AbstractTylerShrinkageAlgorithm,
                    W::MatNum, w::Option{<:StatsBase.AbstractWeights})

Returns the shrinkage intensity of a [`TylerShrinkage`](@ref).

A number is the intensity. Each rule states its formula under `# Mathematical definition`. [`SphericityIntensity`](@ref) and [`AsymptoticIntensity`](@ref) give the weight of the target in the shape of trace `p`. For [`PenalisedShrinkage`](@ref), [`penalised_intensity`](@ref) turns that weight into the intensity.

One asset has one shape, which every intensity gives. The formulas of [`SphericityIntensity`](@ref) and [`OracleIntensity`](@ref) are then zero over zero, and the two rules return one.

# Arguments

  - `rule`: The rule, or the intensity.
  - `alg`: The form of the shrinkage.
  - `W`: The centred data, in the coordinates that turn the target into the identity matrix.
  - `w`: The observation weights, or `nothing`.

# Returns

  - `rho::Number`: The intensity.

# Related

  - [`AbstractRobustShrinkageIntensity`](@ref)
  - [`mcd_intensity`](@ref)
  - [`TylerShrinkage`](@ref)
"""
function tyler_intensity(rho::Number, ::AbstractTylerShrinkageAlgorithm, ::MatNum,
                         ::Option{<:StatsBase.AbstractWeights})
    return rho
end
function tyler_intensity(rule::ConditionNumberIntensity,
                         alg::AbstractTylerShrinkageAlgorithm, W::MatNum,
                         w::Option{<:StatsBase.AbstractWeights})
    lo = tyler_intensity_bound(W, w)
    function reaches(rho)
        V = tyler_shape(ShapeShrinkage(alg, rho), W, w)
        vals = LinearAlgebra.eigvals!(LinearAlgebra.Symmetric(V))
        return last(vals) <= rule.max_cond * first(vals)
    end
    z = zero(lo)
    return lo < z && reaches(z) ? z : smallest_intensity(reaches, lo)
end
function tyler_intensity(::SphericityIntensity, ::TraceNormalisedShrinkage, W::MatNum,
                         w::Option{<:StatsBase.AbstractWeights})
    n, p = observation_weight_count(w, size(W, 1)), size(W, 2)
    if isone(p)
        return one(eltype(W))
    end
    g = sphericity(spatial_sign_scatter(W, observation_weight_vector(W, w)), n)
    b = n * (g - 1)
    return clamp(1 - b / (b * (1 - inv(n)) + (2 * g + p) * (p - 1) / (p + 2)), zero(g),
                 one(g))
end
function tyler_intensity(rule::SphericityIntensity, ::PenalisedShrinkage, W::MatNum,
                         w::Option{<:StatsBase.AbstractWeights})
    weight = tyler_intensity(rule, TraceNormalisedShrinkage(), W, w)
    return penalised_intensity(weight, W, w)
end
function tyler_intensity(::OracleIntensity, ::TraceNormalisedShrinkage, W::MatNum,
                         w::Option{<:StatsBase.AbstractWeights})
    n, p = observation_weight_count(w, size(W, 1)), size(W, 2)
    if isone(p)
        return one(eltype(W))
    end
    t = sum(abs2, spatial_sign_scatter(W, observation_weight_vector(W, w)))
    rho = (p^2 + (1 - 2 / p) * t) / (p^2 - n * p - 2 * n + (n + 1 + 2 * (n - 1) / p) * t)
    return clamp(rho, zero(rho), one(rho))
end
function tyler_intensity(::OracleIntensity, ::PenalisedShrinkage, W::MatNum,
                         w::Option{<:StatsBase.AbstractWeights})
    n, p = observation_weight_count(w, size(W, 1)), size(W, 2)
    if isone(p)
        return one(eltype(W))
    end
    lo = tyler_intensity_bound(W, w)
    V = if lo < zero(lo)
        tyler_shape(nothing, W, w)
    else
        tyler_shape(ShapeShrinkage(PenalisedShrinkage(), (1 + lo) / 2), W, w)
    end
    iv = inv.(LinearAlgebra.eigvals!(LinearAlgebra.Symmetric(V)))
    # The eigenvalues of the inverse of the pilot, scaled to the trace `p`.
    s = p / sum(iv)
    a = p - 2 + p * sum(inv, iv) / s
    return a / (a + n * (p + 2) * (s^2 * sum(abs2, iv) / p - 1))
end
"""
    asymptotic_weight(W::MatNum, w::Option{<:StatsBase.AbstractWeights}) -> Number

Returns the weight of the target that [`AsymptoticIntensity`](@ref) gives to a Tyler shape.

# Arguments

  - `W`: The centred data, in the coordinates that turn the target into the identity matrix.
  - `w`: The observation weights, or `nothing`.

# Returns

  - `weight::Number`: The weight, positive. It can be above one.

# Related

  - [`spatial_sign_scatter`](@ref)
  - [`tyler_intensity`](@ref)

# References

  - $(ref_dict[:couillet2014]) Proposition 2.
"""
function asymptotic_weight(W::MatNum, w::Option{<:StatsBase.AbstractWeights})
    p = size(W, 2)
    R = spatial_sign_scatter(W, observation_weight_vector(W, w))
    return p / observation_weight_count(w, size(W, 1)) / (sum(abs2, R) / p - 1)
end
function tyler_intensity(::AsymptoticIntensity, ::PenalisedShrinkage, W::MatNum,
                         w::Option{<:StatsBase.AbstractWeights})
    return penalised_intensity(asymptotic_weight(W, w), W, w)
end
function tyler_intensity(::AsymptoticIntensity, alg::TraceNormalisedShrinkage, W::MatNum,
                         w::Option{<:StatsBase.AbstractWeights})
    r = asymptotic_weight(W, w)
    if r >= one(r)
        return one(r)
    end
    wt = observation_weight_vector(W, w)
    nrm = vec(sum(abs2, W; dims = 2))
    pos = away_from_centre(nrm)
    u = ifelse.(pos, wt ./ nrm, zero(eltype(nrm))) ./ sum(wt[pos])
    z = zeros(eltype(W), size(W, 2))
    function reaches(rho)
        V = tyler_shape(ShapeShrinkage(alg, rho), W, w)
        a = LinearAlgebra.dot(u, squared_distances(W, z, V))
        return rho * a >= r * (1 - rho + rho * a)
    end
    return smallest_intensity(reaches, zero(r))
end
"""
    tyler_fit(shr::Nothing, ve::Option{<:StatsBase.CovarianceEstimator}, X::MatNum,
              w::Option{<:StatsBase.AbstractWeights}, centre::Option{<:VecNum}; kwargs...)
    tyler_fit(shr::TylerShrinkage, ve::Option{<:StatsBase.CovarianceEstimator}, X::MatNum,
              w::Option{<:StatsBase.AbstractWeights}, centre::Option{<:VecNum}; kwargs...)

Returns the location, the covariance and the shrinkage intensity of a [`TylerCovariance`](@ref).

# Algorithm

The `nothing` method refuses a sample with too few observations with [`assert_robust_rows`](@ref), solves the shape and the location with [`tyler_fixed_point`](@ref), and scales the shape with [`tyler_scale`](@ref). It returns `nothing` in place of the intensity.

The [`TylerShrinkage`](@ref) method does these steps:

 1. Refuse fewer than two observations of positive weight. Take the pilot centre: the coordinatewise weighted median, or `centre`. Take the robust scale of each centred column with [`quantile_scale`](@ref), and its square as the robust variance. An asset with no robust scale has the variance zero.
 2. Build the pilot covariance from these variances and from the correlations of the [`spatial_sign_scatter`](@ref) of the scaled data. Build the target with [`robust_target`](@ref).
 3. Multiply the centred data by the inverse of the symmetric root of the target, so that the target is the identity matrix.
 4. Find the intensity with [`tyler_intensity`](@ref), and refuse it with [`assert_tyler_intensity`](@ref) when the shape does not exist.
 5. Solve the shape and the location in these coordinates with [`tyler_fixed_point`](@ref), and map both back.
 6. Scale the shape with [`tyler_scale`](@ref) and the robust variances.

# Arguments

  - `shr`: The shrinkage, or `nothing`.
  - `ve`: The variance estimator of the scale, or `nothing`.
  - `X`: The data matrix, `observations × assets`.
  - `w`: The observation weights, or `nothing`.
  - `centre`: The fixed centre, or `nothing`.
  - `kwargs...`: More keyword arguments for `ve`.

# Returns

  - `(mu, sigma, rho)`: The location, the covariance, and the intensity.

# Related

  - [`TylerCovariance`](@ref)
  - [`TylerShrinkage`](@ref)
"""
function tyler_fit(::Nothing, ve::Option{<:StatsBase.CovarianceEstimator}, X::MatNum,
                   w::Option{<:StatsBase.AbstractWeights}, centre::Option{<:VecNum};
                   kwargs...)
    assert_robust_rows(count(!iszero, observation_weight_vector(X, w)), size(X, 2),
                       "the number n of observations of positive weight")
    mu, V = tyler_fixed_point(X, w, centre, nothing)
    return mu, tyler_scale(ve, X, mu, V, w, nothing; kwargs...), nothing
end
function tyler_fit(shr::TylerShrinkage, ve::Option{<:StatsBase.CovarianceEstimator},
                   X::MatNum, w::Option{<:StatsBase.AbstractWeights},
                   centre::Option{<:VecNum}; kwargs...)
    wt = observation_weight_vector(X, w)
    @argcheck(count(!iszero, wt) > 1,
              DomainError(size(X),
                          "a shrunk Tyler shape needs two observations of positive weight or more"))
    m = if isnothing(centre)
        [observation_median(view(X, :, j), w) for j in axes(X, 2)]
    else
        centre
    end
    D = X .- transpose(m)
    s = [quantile_scale(view(D, :, j), w) for j in axes(D, 2)]
    v = abs2.(s)
    R = spatial_sign_scatter(D ./ transpose(ifelse.(iszero.(s), one(eltype(s)), s)), wt)
    root = symmetric_root(robust_target(shr.tgt,
                                        StatsBase.cor2cov(StatsBase.cov2cor(R), s)))
    W = D / root
    rho = tyler_intensity(shr.intensity, shr.alg, W, w)
    assert_tyler_intensity(shr.alg, rho, tyler_intensity_bound(W, w))
    z = isnothing(centre) ? nothing : zeros(eltype(W), size(W, 2))
    mw, V = tyler_fixed_point(W, w, z, ShapeShrinkage(shr.alg, rho))
    mu = isnothing(centre) ? m + root * mw : centre
    # A product of one matrix with itself is exactly symmetric.
    A = root * Matrix(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(V)).L)
    return mu, tyler_scale(ve, X, mu, A * transpose(A), w, v; kwargs...), rho
end

public tyler_intensity, shrink_shape!
