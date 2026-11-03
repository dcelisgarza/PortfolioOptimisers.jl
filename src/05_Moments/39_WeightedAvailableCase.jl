"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the weighted available-case mean of each asset, over the valid observations of that asset.

An available-case fit under observation weights reads each asset over its own valid observations. Each observation that the fit drops takes its weight with it, through [`observed_sample`](@ref). The kept weights keep their type and are not scaled, and a [`DynamicAbstractWeights`](@ref) resolves on the observations of the asset. An asset with no observation of positive weight has no mean, so its entry is `NaN`, as [`has_positive_weight`](@ref) states.

# Arguments

  - `Xo`: The oriented block, `observations × assets`.
  - `msk`: The valid entries of the block, after the fold-time rule of the policy, from [`coverage_valid_block`](@ref).
  - $(arg_dict[:oow])

# Returns

  - `mu::VecNum`: The weighted mean of each asset over its valid observations, or `NaN`.

# Related

  - [`observed_sample`](@ref)
  - [`has_positive_weight`](@ref)
  - [`coverage_valid_block`](@ref)
"""
function weighted_coverage_centres(Xo::MatNum, msk::AbstractMatrix{Bool}, w::ObsWeights)
    Tf = float_if_integer(eltype(Xo))
    return map(axes(Xo, 2)) do j
        x, wj = observed_sample(view(Xo, :, j), w, view(msk, :, j))
        return has_positive_weight(wj, length(x)) ? Tf(Statistics.mean(x, wj)) : Tf(NaN)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weighted method of the available-case arm of [`coverage_mean`](@ref). An estimator with observation weights and a [`CoveragePolicy`](@ref) calls it.

A weighted mean does not fold, because the weights of a sample describe the whole sample. So this arm makes one pass over the block, in place of the fold of the unweighted arm. It reads the same valid entries and the same admission as that arm.

# Algorithm

 1. Read the valid entries of the block and the bookkeeping of each asset with [`coverage_valid_block`](@ref).
 2. Take the weighted mean of each asset over its valid observations with [`weighted_coverage_centres`](@ref).
 3. Admit the assets with [`coverage_admission`](@ref), against the observation count of each asset.
 4. Put the means in a frame with [`coverage_frame`](@ref), and orient the result as the `dims` of the caller sets.

# Related

  - [`coverage_mean`](@ref)
  - [`weighted_coverage_centres`](@ref)
  - [`observed_sample`](@ref)
"""
function coverage_mean(me::SimpleExpectedReturns{<:ObsWeights}, cvg::CoveragePolicy,
                       X::MatNum; dims::Int = 1,
                       active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    Xo, msk, _, active, stale = coverage_valid_block(X, active_mask, cvg.alg; dims = dims)
    nu = vec(count(msk; dims = 1))
    cmsk = coverage_admission(cvg, CoverageCounts(nu, nothing, active, stale), size(Xo, 1))
    mu = coverage_frame(weighted_coverage_centres(Xo, msk, me.w), nu, cmsk)
    return isone(dims) ? permutedims(mu) : reshape(mu, :, 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the weighted variance of one asset over its observed sample, or `NaN` when the sample gives it no value.

The function reads the observations that `obs` marks, with their weights, through [`observed_sample`](@ref). `StatsBase` then reads the bias correction of the type of the weights on that sample. A sample with no observation of positive weight is `NaN`, and so is a sample whose divisor is not positive, through [`weighted_moment_guard`](@ref).

# Arguments

  - `x`: The returns of the asset.
  - $(arg_dict[:oow])
  - `obs`: `true` at each observation that the fit reads.
  - `corrected`: The bias correction of the estimator.

# Returns

  - `sigma2::Number`: The weighted variance, or `NaN`.

# Related

  - [`observed_sample`](@ref)
  - [`weighted_moment_guard`](@ref)
  - [`has_positive_weight`](@ref)
"""
function weighted_observed_variance(x::VecNum, w::ObsWeights, obs::AbstractVector{Bool},
                                    corrected::Bool)
    xo, wo = observed_sample(x, w, obs)
    Tf = float_if_integer(eltype(x))
    return if has_positive_weight(wo, length(xo))
        Tf(weighted_moment_guard(Statistics.var(xo, wo; corrected = corrected), wo,
                                 corrected))
    else
        Tf(NaN)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weighted method of the available-case arm of [`coverage_variance`](@ref). A [`SimpleVariance`](@ref) with observation weights and a [`CoveragePolicy`](@ref) calls it.

Each asset is centred on its weighted mean over its own valid observations, as `StatsBase` centres a weighted variance. The arm refuses a `mean` from the caller, as the unweighted arm does.

# Algorithm

 1. Refuse a `mean` from the caller.
 2. Read the valid entries of the block and the bookkeeping of each asset with [`coverage_valid_block`](@ref).
 3. Take the weighted variance of each asset over its valid observations with [`weighted_observed_variance`](@ref).
 4. Admit the assets with [`coverage_admission`](@ref), and put the variances in a frame with [`coverage_frame`](@ref).
 5. Change the variance to the result that `f` asks for with [`coverage_moment_map`](@ref). Orient the result as the `dims` of the caller sets.

# Related

  - [`coverage_variance`](@ref)
  - [`weighted_observed_variance`](@ref)
  - [`observed_sample`](@ref)
"""
function coverage_variance(f::F, ve::SimpleVariance{<:Any, <:ObsWeights},
                           cvg::CoveragePolicy, ::AbstractExpectedReturnsEstimator,
                           X::MatNum; dims::Int = 1, mean = nothing,
                           active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           kwargs...) where {F}
    assert_dims(dims)
    @argcheck(isnothing(mean),
              ArgumentError("an available-case variance centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    Xo, msk, _, active, stale = coverage_valid_block(X, active_mask, cvg.alg; dims = dims)
    nu = vec(count(msk; dims = 1))
    cmsk = coverage_admission(cvg, CoverageCounts(nu, nothing, active, stale), size(Xo, 1))
    v = [weighted_observed_variance(view(Xo, :, j), ve.w, view(msk, :, j), ve.corrected)
         for j in axes(Xo, 2)]
    val = coverage_moment_map(f, coverage_frame(v, nu, cmsk))
    return isone(dims) ? permutedims(val) : reshape(val, :, 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weighted method of the vector arm of [`coverage_variance`](@ref). A [`SimpleVariance`](@ref) with observation weights and a [`CoveragePolicy`](@ref) calls it on the series of one asset.

The fit reads the finite entries of the series with their weights, through [`weighted_observed_variance`](@ref). The method compares the coverage floor with the fraction of finite entries, and it returns `NaN` when the policy does not admit the asset. It refuses a `mean` from the caller, as the matrix method does.

# Related

  - [`coverage_variance`](@ref)
  - [`weighted_observed_variance`](@ref)
  - [`admits`](@ref)
"""
function coverage_variance(f::F, ve::SimpleVariance{<:Any, <:ObsWeights},
                           cvg::CoveragePolicy, X::VecNum; mean = nothing) where {F}
    @argcheck(isnothing(mean),
              ArgumentError("an available-case variance centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    obs = isfinite.(X)
    v = if admits(cvg.alg, count(obs) / max(length(X), 1), true, 0, cvg.min_coverage)
        weighted_observed_variance(X, ve.w, obs, ve.corrected)
    else
        float_if_integer(eltype(X))(NaN)
    end
    return coverage_moment_map(f, v)
end
"""
    weighted_correlation_estimator(ce::StatsBase.CovarianceEstimator,
                                   w::StatsBase.AbstractWeights)
        -> StatsBase.CovarianceEstimator
    weighted_correlation_estimator(ce::StatsBase.SimpleCovariance,
                                   w::StatsBase.AbstractWeights) -> StatsBase.SimpleCovariance

Returns the estimator that a weighted correlation of [`GeneralCovariance`](@ref) calls.

A correlation does not read the divisor of the weights, because the divisor cancels between the covariance and the two standard deviations. But a divisor that is not positive gives a variance at or below zero, and the square root of the diagonal then has no value, or throws. Frequency weights that sum to one or less give such a divisor under the correction. So the `StatsBase.SimpleCovariance` method returns an uncorrected estimator when the corrected divisor is not positive. The divisor of an uncorrected fit is the sum of the weights, which is positive. A positive divisor keeps `ce`, so a valid correlation does not change. A sample with fewer than two observations of positive weight then has no correlation off the diagonal, as [`weighted_comoment_guard`](@ref) states.

The fallback returns `ce` without change, because the library does not know the divisor of any other estimator.

# Arguments

  - $(arg_dict[:ce])
  - `w`: The resolved observation weights of the fit.

# Returns

  - `ce`: The estimator of the correlation.

# Related

  - [`weighted_comoment_guard`](@ref)
  - [`weighted_moment_guard`](@ref)
  - [`cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function weighted_correlation_estimator(ce::StatsBase.CovarianceEstimator,
                                        ::StatsBase.AbstractWeights)
    return ce
end
function weighted_correlation_estimator(ce::StatsBase.SimpleCovariance,
                                        w::StatsBase.AbstractWeights)
    return if inv(StatsBase.varcorrection(w, ce.corrected)) > zero(eltype(w))
        ce
    else
        StatsBase.SimpleCovariance(; corrected = false)
    end
end
"""
    coverage_inner_corrected(ce::GeneralCovariance) -> Bool
    coverage_inner_corrected(ce::StatsBase.CovarianceEstimator) -> Bool

Returns the bias correction of the inner estimator of a weighted available-case [`Covariance`](@ref).

The weights of the `Covariance` replace the weights of a [`GeneralCovariance`](@ref), as [`covariance_inner_estimator`](@ref) states, so the function passes through a `GeneralCovariance` without reading its weights. Then [`partial_fit_corrected`](@ref) reads the flag of the `StatsBase.SimpleCovariance`, and it refuses every other estimator.

# Arguments

  - $(arg_dict[:ce])

# Returns

  - `corrected::Bool`: The bias correction of the innermost `StatsBase.SimpleCovariance`.

# Related

  - [`partial_fit_corrected`](@ref)
  - [`assert_coverage_fittable`](@ref)
"""
function coverage_inner_corrected(ce::GeneralCovariance)
    return partial_fit_corrected(ce.ce)
end
function coverage_inner_corrected(ce::StatsBase.CovarianceEstimator)
    return partial_fit_corrected(ce)
end
"""
    coverage_fittable_estimator(w::Nothing, ce::StatsBase.CovarianceEstimator) -> Bool
    coverage_fittable_estimator(w::Nothing, ce::GeneralCovariance) -> Bool
    coverage_fittable_estimator(w::ObsWeights, ce::StatsBase.CovarianceEstimator) -> Bool

Refuses the inner estimator of a [`Covariance`](@ref) that its available-case arm cannot fit, and returns its bias correction.

The type of the weights of the `Covariance` selects the arm. With no weights the arm folds, through [`partial_fit_corrected`](@ref). The available-case arm reads the weights of the `Covariance` alone, so an inner [`GeneralCovariance`](@ref) with weights of its own is refused, and the message tells the caller to set the weights on the `Covariance`. With weights the arm makes one pass, and [`coverage_inner_corrected`](@ref) passes through the weights of the inner estimator, which the weights of the `Covariance` replace.

# Arguments

  - `w`: The observation weights of the `Covariance`, or `nothing`.
  - $(arg_dict[:ce])

# Returns

  - `corrected::Bool`: The bias correction of the innermost `StatsBase.SimpleCovariance`.

# Related

  - [`assert_coverage_fittable`](@ref)
  - [`partial_fit_corrected`](@ref)
  - [`coverage_inner_corrected`](@ref)
"""
function coverage_fittable_estimator(::Nothing, ce::StatsBase.CovarianceEstimator)
    return partial_fit_corrected(ce)
end
function coverage_fittable_estimator(::Nothing, ce::GeneralCovariance)
    @argcheck(isnothing(ce.w),
              ArgumentError("a `Covariance` with a `CoveragePolicy` reads the observation weights of its own `w`, and not the weights of its inner `GeneralCovariance`. Set the weights on the `Covariance`, as `Covariance(; w = w, cvg = cvg)`, or clear `cvg`."))
    return partial_fit_corrected(ce.ce)
end
function coverage_fittable_estimator(::ObsWeights, ce::StatsBase.CovarianceEstimator)
    return coverage_inner_corrected(ce)
end
"""
    weighted_pair_sample(alg::FullMoment, Xo::MatNum, msk::AbstractMatrix{Bool},
                         w::ObsWeights) -> Tuple
    weighted_pair_sample(alg::SemiMoment, Xo::MatNum, msk::AbstractMatrix{Bool},
                         w::ObsWeights) -> Tuple

Returns the sample that each pair of a weighted available-case covariance reads, and the centre of the pair.

The two moment algorithms take the centre at a different level, as the unweighted arms do. [`FullMoment`](@ref) centres each pair on the weighted mean of the observations that the pair shares, so the pair reads the block itself and `StatsBase` centres it. [`SemiMoment`](@ref) clips each asset about its own weighted available-case mean, with [`weighted_coverage_centres`](@ref), and it takes the co-moment about zero.

# Arguments

  - `alg`: The moment algorithm of the estimator.
  - `Xo`: The oriented block, `observations × assets`.
  - `msk`: The valid entries of the block.
  - $(arg_dict[:oow])

# Returns

  - `(D, mean)::Tuple`: The matrix that each pair reads, and the `mean` keyword of the pair fit.

# Related

  - [`weighted_pair_covariance`](@ref)
  - [`coverage_comoment_deviations`](@ref)
"""
function weighted_pair_sample(::FullMoment, Xo::MatNum, ::AbstractMatrix{Bool},
                              ::ObsWeights)
    return Xo, nothing
end
function weighted_pair_sample(alg::SemiMoment, Xo::MatNum, msk::AbstractMatrix{Bool},
                              w::ObsWeights)
    D = coverage_comoment_deviations(alg, Xo, weighted_coverage_centres(Xo, msk, w))
    return D, zeros(eltype(D), 1, 2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the weighted covariance of the pair of assets `i` and `j` over the observations that both share, or `NaN` when the sample gives it no value.

The pair reads the rows at which both assets are valid, with their weights, through [`observed_sample`](@ref). A [`DynamicAbstractWeights`](@ref) resolves on those rows of `Xo`. The covariance goes through [`robust_cov`](@ref) with a `StatsBase.SimpleCovariance`, the call of the weighted path of [`GeneralCovariance`](@ref), and [`weighted_comoment_guard`](@ref) writes `NaN` when the divisor of the weights is not positive. A pair with no observation of positive weight is `NaN`.

# Arguments

  - `D`: The matrix that the pair reads, from [`weighted_pair_sample`](@ref).
  - `Xo`: The oriented block, on which a dynamic weight resolves.
  - `msk`: The valid entries of the block.
  - $(arg_dict[:oow])
  - `i`, `j`: The two assets.
  - `corrected`: The bias correction of the covariance.
  - `mean`: The centre of the pair, or `nothing` for its weighted mean.

# Returns

  - `sigma_ij::Number`: The weighted covariance of the pair, or `NaN`.

# Related

  - [`weighted_pair_sample`](@ref)
  - [`observed_sample`](@ref)
  - [`weighted_comoment_guard`](@ref)
"""
function weighted_pair_covariance(D::MatNum, Xo::MatNum, msk::AbstractMatrix{Bool},
                                  w::ObsWeights, i::Integer, j::Integer, corrected::Bool,
                                  mean)
    obs = view(msk, :, i) .& view(msk, :, j)
    _, wo = observed_sample(view(Xo, :, [i, j]), w, obs)
    Tf = float_if_integer(eltype(D))
    if !has_positive_weight(wo, count(obs))
        return Tf(NaN)
    end
    sc = StatsBase.SimpleCovariance(; corrected = corrected)
    sigma = robust_cov(sc, D[obs, [i, j]], wo; mean = mean)
    return Tf(weighted_comoment_guard(Statistics.cov, sc, sigma, wo)[1, 2])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the weighted available-case covariance of a [`Covariance`](@ref) with observation weights and a [`CoveragePolicy`](@ref). The two weighted methods of [`coverage_covariance`](@ref) call it.

Each pair reads the observations that its two assets share, with their weights, through [`weighted_pair_covariance`](@ref). The arm reads the valid entries, the admission and the peel of the unweighted arms, so the two arms read one universe. The counts of the admission and the peel are counts of observations, and the weights do not change them.

# Algorithm

 1. Refuse a `mean` from the caller. Read the bias correction of the inner estimator with [`coverage_inner_corrected`](@ref).
 2. Read the valid entries of the block and the bookkeeping of each asset with [`coverage_valid_block`](@ref).
 3. Take the sample and the centre of each pair with [`weighted_pair_sample`](@ref).
 4. Fill each entry of the upper triangle with [`weighted_pair_covariance`](@ref), and mirror it.
 5. Admit the assets with [`coverage_admission`](@ref), and peel the assets of the undetermined pairs with [`coverage_peel`](@ref).
 6. Put the matrix in a frame with [`coverage_frame`](@ref). Rescale it to a unit diagonal when the verb is `Statistics.cor`, with [`coverage_correlation`](@ref).

# Arguments

  - `f`: `Statistics.cov` or `Statistics.cor`.
  - `ce`: The covariance estimator.
  - `cvg`: The policy of the estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: Must be `nothing`.
  - `active_mask`: The active mask of the Asset Panel, or `nothing`.
  - `strict`: Whether a peel refuses rather than warns.

# Returns

  - `val::MatNum`: The covariance or the correlation matrix.

# Related

  - [`coverage_covariance`](@ref)
  - [`weighted_pair_covariance`](@ref)
  - [`coverage_peel`](@ref)
"""
function weighted_coverage_covariance(f::F, ce::Covariance, cvg::CoveragePolicy, X::MatNum;
                                      dims::Int = 1, mean = nothing,
                                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                                      strict::Bool = false) where {F}
    assert_dims(dims)
    @argcheck(isnothing(mean),
              ArgumentError("an available-case covariance centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    corrected = coverage_inner_corrected(ce.ce)
    Xo, msk, _, active, stale = coverage_valid_block(X, active_mask, cvg.alg; dims = dims)
    D, m = weighted_pair_sample(ce.alg, Xo, msk, ce.w)
    N = size(Xo, 2)
    sigma = Matrix{float_if_integer(eltype(D))}(undef, N, N)
    nu = Matrix{Int}(undef, N, N)
    for j in 1:N, i in 1:j
        sigma[i, j] = sigma[j, i] = weighted_pair_covariance(D, Xo, msk, ce.w, i, j,
                                                             corrected, m)
        nu[i, j] = nu[j, i] = count(t -> msk[t, i] & msk[t, j], axes(msk, 1))
    end
    cmsk = coverage_peel(cvg,
                         coverage_admission(cvg, CoverageCounts(nu, nothing, active, stale),
                                            size(Xo, 1)), nu, corrected, strict)
    return coverage_correlation(f, coverage_frame(sigma, nu, cmsk))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weighted [`FullMoment`](@ref) and [`SemiMoment`](@ref) methods of the available-case arm of [`coverage_covariance`](@ref). Both call [`weighted_coverage_covariance`](@ref).

# Related

  - [`coverage_covariance`](@ref)
  - [`weighted_coverage_covariance`](@ref)
"""
function coverage_covariance(f::F, ce::Covariance{<:Any, <:Any, <:FullMoment, <:ObsWeights},
                             cvg::CoveragePolicy, X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             strict::Bool = false, kwargs...) where {F}
    return weighted_coverage_covariance(f, ce, cvg, X; dims = dims, mean = mean,
                                        active_mask = active_mask, strict = strict)
end
function coverage_covariance(f::F, ce::Covariance{<:Any, <:Any, <:SemiMoment, <:ObsWeights},
                             cvg::CoveragePolicy, X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             strict::Bool = false, kwargs...) where {F}
    return weighted_coverage_covariance(f, ce, cvg, X; dims = dims, mean = mean,
                                        active_mask = active_mask, strict = strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the weighted co-moment of one cell of a higher-order available-case fit, or `NaN` when the cell has no observation of positive weight.

The cell reads the observations at which every asset of `idx` is valid, with their weights, through [`observed_sample`](@ref). A [`DynamicAbstractWeights`](@ref) resolves on those rows of `Xo`. The co-moment is the weighted mean of the product of the deviations of the assets of the cell. A higher-order co-moment takes no bias correction, so the divisor is the sum of the kept weights, as the count of observations is the divisor of the unweighted arm.

# Arguments

  - `Y`: The deviations of each asset from its weighted available-case centre, `observations × assets`.
  - `Xo`: The oriented block, on which a dynamic weight resolves.
  - `msk`: The valid entries of the block.
  - $(arg_dict[:oow])
  - `idx`: The assets of the cell, as a vector or a tuple of indices.

# Returns

  - `m::Number`: The weighted co-moment of the cell, or `NaN`.

# Related

  - [`weighted_comoment`](@ref)
  - [`observed_sample`](@ref)
  - [`has_positive_weight`](@ref)
"""
function weighted_comoment_cell(Y::MatNum, Xo::MatNum, msk::AbstractMatrix{Bool},
                                w::ObsWeights,
                                idx::Union{AbstractVector{<:Integer},
                                           Tuple{Vararg{Integer}}})
    cols = collect(idx)
    obs = vec(all(view(msk, :, cols); dims = 2))
    _, wo = observed_sample(view(Xo, :, cols), w, obs)
    if !has_positive_weight(wo, count(obs))
        return convert(eltype(Y), NaN)
    end
    return convert(eltype(Y), Statistics.mean(vec(prod(Y[obs, cols]; dims = 2)), wo))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the weighted available-case co-moment of order `K`, as an array of `K` axes of one entry for each asset.

The co-moment is symmetric in its `K` indices, so the function computes each cell once, with [`weighted_comoment_cell`](@ref). It walks the cells in column-major order. The permutation of a cell whose indices descend has the smallest linear index of all the permutations of that cell, so the walk reaches it first. The function computes the value there, and every other permutation copies it. The dense layouts of the library are reshapes of this array: the coskewness is `N × N²` and the cokurtosis is `N² × N²`.

A cell reads the rows of its own assets, and a dynamic weight resolves on each cell, so the function cannot take the products of the expansions that the unweighted arm takes.

# Arguments

  - `Y`: The deviations of each asset from its weighted available-case centre.
  - `Xo`: The oriented block, on which a dynamic weight resolves.
  - `msk`: The valid entries of the block.
  - $(arg_dict[:oow])
  - `::Val{K}`: The order of the co-moment.

# Returns

  - `A::Array`: The co-moment, with `K` axes of length `N`.

# Related

  - [`weighted_comoment_cell`](@ref)
  - [`coverage_coskewness`](@ref)
  - [`coverage_cokurtosis`](@ref)
"""
function weighted_comoment(Y::MatNum, Xo::MatNum, msk::AbstractMatrix{Bool}, w::ObsWeights,
                           ::Val{K}) where {K}
    N = size(Y, 2)
    A = Array{eltype(Y)}(undef, ntuple(_ -> N, Val(K)))
    for c in CartesianIndices(A)
        t = Tuple(c)
        s = sort(t; rev = true)
        A[c] = if t == s
            weighted_comoment_cell(Y, Xo, msk, w, t)
        else
            A[CartesianIndex(s)]
        end
    end
    return A
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the deviations, the valid entries and the admitted assets that a weighted higher-order available-case fit needs.

It is the weighted form of [`coverage_comoment_block`](@ref). Each asset is centred on its weighted mean over its own valid observations, with [`weighted_coverage_centres`](@ref), and the moment algorithm then clips the deviations, with [`coverage_comoment_deviations`](@ref). The admission reads the observation count of each asset, as in the unweighted arm.

# Arguments

  - `alg`: The moment algorithm of the estimator.
  - `cvg`: The policy of the estimator.
  - $(arg_dict[:X])
  - `active_mask`: The active mask of the Asset Panel, or `nothing`.
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])

# Returns

  - `(Xo, Y, msk, cmsk)::Tuple`: The oriented block, the deviations, the valid entries, and the admitted assets or `nothing`.

# Related

  - [`coverage_comoment_block`](@ref)
  - [`weighted_comoment`](@ref)
"""
function weighted_comoment_block(alg::AbstractMomentAlgorithm, cvg::CoveragePolicy,
                                 X::MatNum, active_mask::Option{<:AbstractMatrix{<:Bool}},
                                 w::ObsWeights, dims::Int)
    Xo, msk, _, active, stale = coverage_valid_block(X, active_mask, cvg.alg; dims = dims)
    Y = coverage_comoment_deviations(alg, Xo, weighted_coverage_centres(Xo, msk, w))
    counts = CoverageCounts(vec(count(msk; dims = 1)), nothing, active, stale)
    return Xo, Y, msk, coverage_admission(cvg, counts, size(Xo, 1))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weighted method of the available-case arm of [`coverage_coskewness`](@ref). A [`Coskewness`](@ref) with observation weights and a [`CoveragePolicy`](@ref) calls it.

# Algorithm

 1. Refuse a `mean` from the caller.
 2. Read the deviations, the valid entries and the admitted assets with [`weighted_comoment_block`](@ref).
 3. Build the third co-moment with [`weighted_comoment`](@ref), and reshape it to `N × N²`.
 4. Frame the refused assets with [`coverage_refuse_comoment!`](@ref), and take the negative spectral coskewness with [`negative_spectral_coskewness`](@ref).

# Related

  - [`coverage_coskewness`](@ref)
  - [`weighted_comoment`](@ref)
"""
function coverage_coskewness(ske::Coskewness{<:Any, <:Any, <:Any, <:ObsWeights},
                             cvg::CoveragePolicy, X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    assert_dims(dims)
    @argcheck(isnothing(mean),
              ArgumentError("an available-case coskewness centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    Xo, Y, msk, cmsk = weighted_comoment_block(ske.alg, cvg, X, active_mask, ske.w, dims)
    N = size(Xo, 2)
    cskew = reshape(weighted_comoment(Y, Xo, msk, ske.w, Val(3)), N, N^2)
    coverage_refuse_comoment!(cskew, cmsk, Val(:sk))
    return cskew, negative_spectral_coskewness(cskew, Xo, ske.mp)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weighted method of the available-case arm of [`coverage_cokurtosis`](@ref). A [`Cokurtosis`](@ref) with observation weights and a [`CoveragePolicy`](@ref) calls it.

# Algorithm

 1. Refuse a `mean` from the caller.
 2. Read the deviations, the valid entries and the admitted assets with [`weighted_comoment_block`](@ref).
 3. Build the fourth co-moment with [`weighted_comoment`](@ref), and reshape it to `N² × N²`.
 4. Frame the refused assets with [`coverage_refuse_comoment!`](@ref), and repair the block with [`matrix_processing_block!`](@ref).

# Related

  - [`coverage_cokurtosis`](@ref)
  - [`weighted_comoment`](@ref)
"""
function coverage_cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:Any, <:ObsWeights},
                             cvg::CoveragePolicy, X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...)
    assert_dims(dims)
    @argcheck(isnothing(mean),
              ArgumentError("an available-case cokurtosis centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    Xo, Y, msk, cmsk = weighted_comoment_block(kte.alg, cvg, X, active_mask, kte.w, dims)
    N = size(Xo, 2)
    ckurt = reshape(weighted_comoment(Y, Xo, msk, kte.w, Val(4)), N^2, N^2)
    coverage_refuse_comoment!(ckurt, cmsk, Val(:kt))
    matrix_processing_block!(kte.mp, ckurt, Xo)
    return ckurt
end
