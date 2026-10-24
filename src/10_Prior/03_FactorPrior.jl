"""
$(DocStringExtensions.TYPEDEF)

Factor-based prior estimator for asset returns.

`FactorPrior` is a low order prior estimator that computes the mean and covariance of asset returns from a factor model. `pe` fits the factor moments on the factor returns, and `re` fits the loadings of each asset on the factors. The estimator projects the factor moments through the loadings, and processes the covariance with `mp`. When `rsd` is `true`, it adds to the covariance the residual variances that `ve` measures.

The covariance is the plug-in estimator of the factor model that [fan2008](@citet) state. It is the loadings times the factor covariance times the transposed loadings, plus the diagonal matrix of the residual variances. The library differs from the source in three ways:

  - The mean adds the regression intercepts. The model of the source has no intercepts.
  - The default `re` is stepwise, so an asset can drop a factor. The source fits every factor by least squares.
  - `ve` sizes the residual variances. The default [`SimpleVariance`](@ref) divides by `T - 1`, and the source divides by `T`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorPrior(;
        pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        rsd::Bool = true,
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> FactorPrior

Keywords correspond to the struct's fields.

## Validation

  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pe`, `re` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).
  - When `rsd` is `true`, `mp` states the estimator that repairs the covariance, see [`assert_matrix_processing_posdef`](@ref).

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pe`: Recursively updated through [`factory`](@ref).
  - `re`: Recursively updated through [`factory`](@ref).
  - `ve`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `re`: Recursively viewed through [`port_opt_view`](@ref).
  - `ve`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## The incremental fit

This prior has no exact incremental recursion, so the online step refits from a sample buffer. [`Online`](@ref) seeds `cache`, and [`partial_fit!`](@ref) appends each observation to it. The one-argument [`prior`](@ref) runs the batch verb of this estimator over the rows in the buffer. The answer equals a batch fit over those rows, and a `max_history` on the wrapper cuts the whole fit.

[`obs_weights_view`](@ref) drops `cache`, because a partial-fit state has no slice on the observation axis. [`show_fields`](@ref) does not render it, because a running buffer is not part of the configuration.

## Composition: what this estimator forwards

This estimator lifts a factor-axis prior onto the asset axis. It rebuilds `X` as `F * transpose(M) .+ transpose(b)`, so it builds its prior result itself and does not forward one along its own axis. The forwarding rule still governs each field. This estimator does not change the factor distribution. [`FactorBlackLittermanPrior`](@ref) is this estimator with views on the factor block before the lift.

  - The factor block `fpr` is the result of the wrapped factor prior, forwarded whole. It needs no reconstruction, because the asset moments are its projection and not an update of it.
  - `mu` and `sigma` are that block projected through the loadings, so `mu == rr.M * fpr.mu + rr.b` holds for every investable asset. When `rsd` is `true`, `sigma` also holds the residual variances.
  - `chol` is not forwarded. The lift rebuilds it on the asset axis from `M * matrix_square_root(mtx_sqrt, fpr.sigma)`, and adds the residual block when `rsd` is `true`. It factorises `sigma` before matrix processing, see [`factor_lift`](@ref). An `mp` that denoises or detones moves `sigma` and leaves `chol` as it was.
  - `w` is the observation weights of the factor prior. This estimator wraps only a factor prior, and `posterior_X` has the rows of `F`, so these are the only weights, and they are on the correct axis. The `ens`, `kld` and `ow` of the factor prior come with them.

# Examples

```jldoctest
julia> FactorPrior()
FactorPrior
        pe ┼ EmpiricalPrior
           │           ce ┼ PortfolioOptimisersCovariance
           │              │   ce ┼ Covariance
           │              │      │    me ┼ SimpleExpectedReturns
           │              │      │       │   w ┴ nothing
           │              │      │    ce ┼ GeneralCovariance
           │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
           │              │      │       │    w ┴ nothing
           │              │      │   alg ┼ FullMoment()
           │              │      │     w ┴ nothing
           │              │   mp ┼ MatrixProcessing
           │              │      │     pdm ┼ Posdef
           │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
           │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │              │      │      dn ┼ nothing
           │              │      │      dt ┼ nothing
           │              │      │     alg ┼ nothing
           │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
           │           me ┼ SimpleExpectedReturns
           │              │   w ┴ nothing
           │      horizon ┼ nothing
           │   fill_limit ┼ nothing
           │        x_src ┴ nothing
        mp ┼ MatrixProcessing
           │     pdm ┼ Posdef
           │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
           │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │      dn ┼ nothing
           │      dt ┼ nothing
           │     alg ┼ nothing
           │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        re ┼ StepwiseRegression
           │       crit ┼ PValue
           │            │   t ┴ Float64: 0.05
           │        alg ┼ ForwardSelection()
           │        tgt ┼ LinearModel
           │            │   kwargs ┴ @NamedTuple{}: NamedTuple()
           │     choice ┼ BatchChoice()
           │   included ┼ nothing
           │      x_src ┴ nothing
        ve ┼ SimpleVariance
           │          me ┼ SimpleExpectedReturns
           │             │   w ┴ nothing
           │           w ┼ nothing
           │   corrected ┴ Bool: true
       rsd ┼ Bool: true
  mtx_sqrt ┼ EigenFallbackSquareRoot()
     x_src ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`SimpleVariance`](@ref)
  - [`prior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:fan2008]) Section 1.2, Equations 1.3 and 1.4. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct FactorPrior <: AbstractLowOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    @fprop pe
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:re])
    """
    @fprop @vprop re
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:rsd])
    """
    rsd
    """
    $(field_dict[:mtx_sqrt])
    """
    mtx_sqrt
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function FactorPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                         mp::AbstractMatrixProcessingEstimator,
                         re::AbstractTimeSeriesRegressionEstimator,
                         ve::AbstractVarianceEstimator, rsd::Bool,
                         mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm},
                         cache::Option{<:AbstractPartialFitState},
                         x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :FactorPrior)
        assert_inner_reader(pe, :pe, :FactorPrior)
        assert_inner_reader(re, :re, :FactorPrior)
        if rsd
            assert_matrix_processing_posdef(mp)
        end
        return new{typeof(pe), typeof(mp), typeof(re), typeof(ve), typeof(rsd),
                   typeof(mtx_sqrt), typeof(cache), typeof(x_src)}(pe, mp, re, ve, rsd,
                                                                   mtx_sqrt, cache, x_src)
    end
end
function FactorPrior(; pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                     mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                     re::AbstractTimeSeriesRegressionEstimator = StepwiseRegression(),
                     ve::AbstractVarianceEstimator = SimpleVariance(), rsd::Bool = true,
                     mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
                     cache::Option{<:AbstractPartialFitState} = nothing,
                     x_src::Option{<:AbstractReturnsSource} = nothing)::FactorPrior
    return FactorPrior(pe, mp, re, ve, rsd, mtx_sqrt, cache, x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`FactorPrior`](@ref) except `cache`.

The state in `cache` belongs to an incremental fit and is not part of the configuration, and it would print under the estimator at every site that renders one. Set `set_show_nothing_fields!(:FactorPrior, true)` to render it.

# Arguments

  - `::FactorPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pe, :mp, :re, :ve, :rsd, :mtx_sqrt, :x_src)`.

# Related

  - [`FactorPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::FactorPrior) = (:pe, :mp, :re, :ve, :rsd, :mtx_sqrt, :x_src)
# Expose `:me` and `:ce` from the embedded asset prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties FactorPrior begin
    forward(pe, me, ce)
end
"""
    factor_reconstruction(re::AbstractTimeSeriesRegressionEstimator, X::MatNum,
                          F::MatNum) -> Tuple{AbstractLoadingsRegressionResult, MatNum}

Fit the loadings and rebuild the asset returns from the factor returns.

This is the first half of the factor lift, and the only half that every factor-axis estimator uses. It fits `re` on `(X, F)`, and returns the regression result and the posterior returns matrix `F * transpose(M) .+ transpose(b)`. [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) each write this matrix into `LowOrderPrior.X`.

[`factor_lift`](@ref) is the second half, which projects the factor moments through the loadings. The two halves are separate functions because [`FactorBlackLittermanPrior`](@ref) needs the reconstruction before it has the moments to project. Its views change the factor distribution, so its factor moments exist only after the Black-Litterman update.

# Algorithm

 1. Fit `re` on `(X, F)` with [`regression`](@ref) to get `rr`. `rr.M` holds the ``N \\times K`` loadings, and `rr.b` holds the ``N \\times 1`` intercepts.
 2. Rebuild the asset returns as `F * transpose(rr.M) .+ transpose(rr.b)` to get `posterior_X`. It has the rows of `F` and the columns of `X`.

# Arguments

  - $(arg_dict[:re])
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Returns

  - `rr::AbstractLoadingsRegressionResult`: Regression result carrying the loadings `M` and intercepts `b`.
  - `posterior_X::MatNum`: Reconstructed asset returns, `observations × assets`.

# Related

  - [`factor_lift`](@ref)
  - [`regression`](@ref)
  - [`FactorPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function factor_reconstruction(re::AbstractTimeSeriesRegressionEstimator, X::MatNum,
                               F::MatNum)
    rr = regression(re, X, F)
    return rr, F * transpose(rr.M) .+ transpose(rr.b)
end
"""
    coverage_regression(re::AbstractTimeSeriesRegressionEstimator, cmsk::Nothing)
    coverage_regression(re::AbstractTimeSeriesRegressionEstimator, cmsk::BitVector)

Takes the view of a regression estimator at the assets that a factor prior fits.

[`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) fit the regression on the columns of the returns that a mask keeps. A regression can hold one value per asset, such as the factor set of each asset in `included` of a [`StepwiseRegression`](@ref). That value must describe the same columns as the returns that the fit reads, so the prior takes the view of the regression with the same mask.

# Algorithm

The method that Julia selects is the algorithm.

 1. If `cmsk` is `nothing`, every asset is in the fit, so return `re`.
 2. If `cmsk` is a mask, return [`port_opt_view`](@ref) of `re` at `findall(cmsk)`.

# Arguments

  - $(arg_dict[:re])
  - `cmsk`: The mask of the assets in the fit, or `nothing`.

# Returns

  - `re::AbstractTimeSeriesRegressionEstimator`: The regression on the assets in the fit.

# Related

  - [`factor_reconstruction`](@ref)
  - [`coverage_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function coverage_regression(re::AbstractTimeSeriesRegressionEstimator, ::Nothing)
    return re
end
function coverage_regression(re::AbstractTimeSeriesRegressionEstimator, cmsk::BitVector)
    return port_opt_view(re, findall(cmsk))
end
"""
    factor_lift(mp::AbstractMatrixProcessingEstimator, ve::AbstractVarianceEstimator,
                rsd::Bool, rr::AbstractLoadingsRegressionResult, f_mu::VecNum, f_sigma::MatNum,
                X::MatNum, posterior_X::MatNum;
                mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
                kwargs...) -> NamedTuple

Project factor moments onto the asset axis through the regression loadings.

This is the second half of the factor lift, and [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) both call it. It maps `f_mu` and `f_sigma` through the loadings and processes the covariance. When `rsd` is `true`, it adds the diagonal residual block to the covariance and to the square-root factor.

The two callers differ only in the factor moments that they pass. [`FactorPrior`](@ref) passes the moments of the wrapped prior, and [`FactorBlackLittermanPrior`](@ref) passes the Black-Litterman posterior moments.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{f}} + \\boldsymbol{\\alpha}\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` asset expected returns vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` asset covariance matrix.
  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix, `rr.M`.
  - ``\\boldsymbol{\\alpha}``: ``N \\times 1`` vector of regression intercepts, `rr.b`.
  - ``\\hat{\\boldsymbol{f}}``: ``K \\times 1`` factor expected returns vector, `f_mu`.
  - ``\\mathbf{\\Sigma}_f``: ``K \\times K`` factor covariance matrix, `f_sigma`.
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal matrix of residual variances, present only when `rsd` is `true`.

The returned `chol` is the transpose of ``[\\mathbf{B} \\mathbf{L}_f \\quad \\mathbf{\\Sigma}_\\varepsilon^{1/2}]``, where ``\\mathbf{L}_f`` is the square root of ``\\mathbf{\\Sigma}_f`` that [`matrix_square_root`](@ref) takes under `mtx_sqrt`. By default it is the lower Cholesky factor of a positive definite matrix, and the eigen square root of a singular one. `chol` is ``(K + N) \\times N`` when `rsd` is `true`, and ``K \\times N`` when `rsd` is `false`. With `rsd = false`, neither `chol` nor ``\\hat{\\mathbf{\\Sigma}}`` holds the residual block.

The identity ``\\mathtt{chol}^\\intercal \\mathtt{chol} = \\hat{\\mathbf{\\Sigma}}`` holds before matrix processing only. The lift builds `chol` from the `f_sigma` that the caller passed, so step 4 of the algorithm changes `sigma` and does not change `chol`. The default [`MatrixProcessing`](@ref) leaves the projected covariance as it is, because its `pdm` changes nothing on a positive semi-definite matrix. Under that `mp` the identity holds on the returned pair. An `mp` that denoises or detones moves `sigma` and leaves `chol` as it was, so the identity then holds against the unprocessed covariance only. A consumer that needs a factor of the returned `sigma` must factorise it again.

# Algorithm

 1. Read the loadings `M` and the intercepts `b` off `rr`.
 2. Project the factor mean through the loadings to get `posterior_mu`.
 3. Project the factor covariance through the loadings to get `posterior_sigma`, the systematic block.
 4. Process `posterior_sigma` in place with [`matrix_processing!`](@ref), under `mp` and `posterior_X`.
 5. Take the square root of `f_sigma` under `mtx_sqrt` with [`matrix_square_root`](@ref), and multiply it by the loadings to get `posterior_csigma`. This reads the `f_sigma` that the caller passed, which step 4 does not change.
 6. When `rsd` is `true`, take the reconstruction error `err = X - posterior_X`. Measure `esigma`, the column variances of `err`, with `ve`. Read the counts of those variances with [`variance_count`](@ref), and combine them with the `edof` of `rr` through [`residual_variance_counts`](@ref) to get `edof` and `ediv`. Let `err_sigma` be the diagonal matrix of the variances. When `rsd` is `false`, `esigma` and `ediv` are `nothing`, and `edof` is the `edof` of `rr`.
 7. Under `rsd`, add `err_sigma` to `posterior_sigma`, and re-condition the sum with [`posdef!`](@ref) under the estimator that [`matrix_processing_posdef`](@ref) gives for `mp`. This is the only explicit [`posdef!`](@ref) call of the function. For a [`MatrixProcessing`](@ref) that estimator is `mp.pdm`, and it also acts on `posterior_sigma` inside step 4 when `:pdm` is a member of `mp.order`.
 8. Under `rsd`, append the columns of `sqrt.(err_sigma)` to `posterior_csigma`, so the factor also holds the block that step 7 added to the covariance.
 9. Reshape `posterior_csigma` to `length(posterior_mu)` columns and transpose it into `chol`. Return `mu`, `sigma`, `chol`, `esigma`, `edof` and `ediv`.

# Arguments

  - $(arg_dict[:mp])
  - $(arg_dict[:ve])
  - $(arg_dict[:rsd])
  - `rr`: Regression result carrying the loadings `M` and intercepts `b`.
  - `f_mu`: Factor expected returns, `factors × 1`.
  - `f_sigma`: Factor covariance matrix, `factors × factors`.
  - $(arg_dict[:X])
  - `posterior_X`: Reconstructed asset returns from [`factor_reconstruction`](@ref).
  - $(arg_dict[:mtx_sqrt])
  - `kwargs...`: Additional keyword arguments passed to matrix processing.

# Returns

  - `(; mu, sigma, chol, esigma, edof, ediv)::NamedTuple`: Asset expected returns, asset covariance, the square-root factor whose trailing block is the residual standard deviations when `rsd` is `true`, the residual variances, and their degrees of freedom and divisors. `esigma` and `ediv` are `nothing` when `rsd` is `false`, because the lift added no residual block. A caller writes the last three onto the loadings result that it returns. A consumer that needs the idiosyncratic variances, or their sampling law, then reads them off the block, and does not compute them again from the reconstruction error.

# Related

  - [`factor_reconstruction`](@ref)
  - [`factor_residual_config`](@ref)
  - [`Regression`](@ref)
  - [`FactorPrior`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function factor_lift(mp::AbstractMatrixProcessingEstimator, ve::AbstractVarianceEstimator,
                     rsd::Bool, rr::AbstractLoadingsRegressionResult, f_mu::VecNum,
                     f_sigma::MatNum, X::MatNum, posterior_X::MatNum;
                     mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot(),
                     kwargs...)
    (; b, M) = rr
    posterior_mu = M * f_mu + b
    posterior_sigma = M * f_sigma * transpose(M)
    matrix_processing!(mp, posterior_sigma, posterior_X; kwargs...)
    posterior_csigma = M * matrix_square_root(mtx_sqrt, f_sigma)
    esigma = nothing
    edof = rr.edof
    ediv = nothing
    if rsd
        err = X - posterior_X
        esigma = vec(Statistics.var(ve, err; dims = 1))
        (; edof, ediv) = residual_variance_counts(variance_count(ve, err), rr.edof,
                                                  size(err, 1))
        err_sigma = LinearAlgebra.diagm(esigma)
        posterior_sigma .+= err_sigma
        posdef!(matrix_processing_posdef(mp), posterior_sigma)
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return (; mu = posterior_mu, sigma = posterior_sigma,
            chol = transpose(reshape(posterior_csigma, length(posterior_mu), :)),
            esigma = esigma, edof = edof, ediv = ediv)
end
"""
    residual_variance_counts(cnt::Nothing, edof, T::Integer)
    residual_variance_counts(cnt::NamedTuple, edof::Nothing, T::Integer)
    residual_variance_counts(cnt::NamedTuple, edof::VecNum, T::Integer)

Degrees of freedom and divisor of each residual variance that a lift measures.

The sampling law of a residual variance needs one fact from the regression and one from the variance estimator. The regression records in `edof` the count of equally weighted observations that remain after the parameters it spent on each asset. [`variance_count`](@ref) states in `cnt` the effective count of the observations that the variance estimator reads, and its divisor. This function combines the two. Each case is a method, so no caller writes an `isnothing` test.

# Mathematical definition

```math
\\begin{align}
\\nu_{i} &= n_{i} - \\left(T - \\nu_{i}^{\\mathrm{fit}}\\right)\\,.
\\end{align}
```

Where:

  - ``\\nu_{i}``: Degrees of freedom of the residual variance of asset ``i``.
  - ``n_{i}``: Effective count of the observations that the variance estimator reads, `cnt.n`.
  - $(math_dict[:T])
  - ``\\nu_{i}^{\\mathrm{fit}}``: Degrees of freedom that the regression left, `edof`. ``T - \\nu_{i}^{\\mathrm{fit}}`` is the number of parameters it spent on the asset.

With equal weights ``n_{i} = T``, and the result is the `edof` of the regression. Under unequal weights the function subtracts the parameters that the regression spent from the effective count. This approximates the law of a weighted sum of squared residuals.

# Arguments

  - `cnt`: The count and divisor of the variance estimator, or `nothing` when it states none.
  - `edof`: The `edof` of the loadings block, or `nothing` when the regression recorded none.
  - `T`: Number of observations of the residuals.

# Returns

  - `(; edof, ediv)::NamedTuple`: The degrees of freedom and the divisor of each variance. Both are `nothing` when the variance estimator states no count, because neither fact is then known. `edof` alone is `nothing` when the regression recorded no count.

# Related

  - [`factor_lift`](@ref)
  - [`variance_count`](@ref)
  - [`Regression`](@ref)
"""
function residual_variance_counts(::Nothing, ::Any, ::Integer)
    return (; edof = nothing, ediv = nothing)
end
function residual_variance_counts(cnt::NamedTuple, ::Nothing, ::Integer)
    return (; edof = nothing, ediv = cnt.m)
end
function residual_variance_counts(cnt::NamedTuple, edof::VecNum, T::Integer)
    return (; edof = cnt.n .- (T .- edof), ediv = cnt.m)
end
"""
    factor_residual_config(pe::AbstractPriorEstimator) -> Option{<:NamedTuple}

Declare how a prior estimator adds a residual block to the covariance it lifts.

[`HighOrderFactorPriorEstimator`](@ref) subtracts the residual block to recover the systematic covariance, because its residual cokurtosis correction is defined on that covariance. It cannot read `ve` and the repair of `mp` off the fields of the estimator that it wraps. Its `pe` field has the bound [`AbstractLowOrderPriorEstimator_F_AF`](@ref), and in that bound only [`FactorPrior`](@ref) and [`FactorBlackLittermanPrior`](@ref) carry both fields. The other members wrap a prior, pool priors, or build the asset moments with no residual block, as [`AugmentedBlackLittermanPrior`](@ref) does. A field access on one of them raises a `FieldError`.

The declaration states whether the estimator added a block, and how to re-condition the covariance after a consumer removes the block. The block itself is on the loadings result, at `rr.esigma`, and the fit measured it under its own weights. A prior that embeds this estimator can give the fit weights other than those of `ve`, for example the `w` of a caller on an entropy pooling prior, or the pooled weights of an opinion pool. So a consumer reads the block off `rr`, and does not measure it again under `ve`.

Each estimator declares its answer in a method beside its own definition:

  - The two estimators that add a residual block report it.
  - A wrapper forwards the answer of the estimator that it wraps. A pooling estimator is a wrapper here, and it forwards the one estimator that its moments come from.
  - An estimator that adds no residual block returns `nothing` from an explicit method.
  - [`CrossSectionalFactorPrior`](@ref) throws an `ArgumentError`. Its block is the idiosyncratic covariance that it measured, which a variance under `ve` does not rebuild.

An answer of `nothing` and an answer whose `rsd` is `false` both mean that the estimator added no residual block, so the consumer leaves the covariance as it is.

There is no default. A fallback that returns `nothing` cannot tell an estimator that adds no residual block from an estimator whose author forgot the method. In the second case the consumer does not subtract a residual block that the covariance holds. So the method for an undeclared type throws an `ArgumentError` that names the type. [`range_tails`](@ref) uses the same rule for a per-type declaration whose absence is a defect.

# Arguments

  - `pe`: Prior estimator.

# Validation

  - Throws an `ArgumentError` when the type of `pe` declares no method.

# Returns

  - `nothing`: The estimator adds no residual block.
  - `(; ve, pdm, rsd)::NamedTuple`: The variance estimator that sizes the residual block, the positive definite matrix estimator that re-conditions the covariance after a consumer removes the block, and whether the estimator adds the block. A factor prior gives `pdm` from [`matrix_processing_posdef`](@ref) when `rsd` is `true`, and `nothing` when `rsd` is `false`, because no consumer repairs a covariance that has no residual block.

# Related

  - [`factor_lift`](@ref)
  - [`assert_factor_residual_config`](@ref)
  - [`range_tails`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
  - [`FactorPrior`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
"""
function factor_residual_config(pe::AbstractPriorEstimator)
    return throw(ArgumentError("`factor_residual_config` is not defined for `$(nameof(typeof(pe)))`. Every concrete `AbstractPriorEstimator` must declare its residual block beside its own definition by adding a method returning `(; ve, pdm, rsd)`, forwarding the estimator it wraps, or returning `nothing` when it adds no residual block."))
end
function factor_residual_config(pe::FactorPrior)
    return (; ve = pe.ve, pdm = pe.rsd ? matrix_processing_posdef(pe.mp) : nothing,
            rsd = pe.rsd)
end
"""
    assert_factor_residual_config(pe::AbstractPriorEstimator, cfg) -> Nothing

Check the shape of a [`factor_residual_config`](@ref) answer before a consumer reads it.

A consumer reads `ve`, `pdm` and `rsd` off the answer by property access, which checks no shape. A declaration that returns a wrong value would raise a `FieldError` deep inside the correction, far from the declaration that caused it. This function checks the two shapes that the contract permits, and its error names the estimator that answered.

# Arguments

  - `pe`: Prior estimator that answered.
  - `cfg`: The answer of `factor_residual_config(pe)`.

# Validation

  - `cfg` is `nothing`, or a `NamedTuple` carrying `ve`, `pdm` and `rsd`.

# Related

  - [`factor_residual_config`](@ref)
  - [`HighOrderFactorPriorEstimator`](@ref)
"""
function assert_factor_residual_config(pe::AbstractPriorEstimator, cfg)::Nothing
    @argcheck(isnothing(cfg) ||
              (isa(cfg, NamedTuple) && all(k -> haskey(cfg, k), (:ve, :pdm, :rsd))),
              ArgumentError("`factor_residual_config(::$(nameof(typeof(pe))))` must return `nothing` or a `NamedTuple` carrying `ve`, `pdm` and `rsd`. Got\ncfg => $(cfg)::$(typeof(cfg))."))
    return nothing
end
"""
    prior(pe::FactorPrior, X::MatNum, F::MatNum, pnl::Option{<:AssetPanel} = nothing;
          dims::Int = 1, strict::Bool = false, ts::Option{<:VecDate} = nothing,
          active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, cache_reg::Bool = false,
          kwargs...)

Compute the prior moments of asset returns from a factor model.

`pe.pe` fits the factor moments on `F`, and `pe.re` fits the loadings of the assets on the factors. [`factor_lift`](@ref) projects the factor moments onto the assets through the loadings. When `pe.rsd` is `true`, the lift adds the residual variances to the covariance. The method returns a [`LowOrderPrior`](@ref).

# Mathematical definition

The factor model maps the factor moments to the assets through the loadings matrix ``\\mathbf{B}`` and the intercepts ``\\boldsymbol{\\alpha}``:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{f}} + \\boldsymbol{\\alpha}\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix, `rr.M`.
  - ``\\hat{\\boldsymbol{f}}``: ``K \\times 1`` vector of factor expected returns.
  - ``\\boldsymbol{\\alpha}``: ``N \\times 1`` vector of regression intercepts, `rr.b`.
  - ``\\mathbf{\\Sigma}_f``: ``K \\times K`` factor covariance matrix.
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal matrix of residual variances, present only when `rsd` is `true`.

`pe.pe` fits ``\\hat{\\boldsymbol{f}}`` and ``\\mathbf{\\Sigma}_f`` on `F`, and `pe.re` fits the loadings on `(X, F)`. [`factor_lift`](@ref) computes the two equations.

# Algorithm

 1. Orient `X` and `F` with [`dims_oriented`](@ref), to `observations × assets` and `observations × factors`.
 2. Fit the wrapped prior `pe.pe` on `F` to get `f_prior`, the factor-axis prior result. The method passes `strict` to it, because `pe.pe` can be a [`BlackLittermanPrior`](@ref) or an [`EntropyPoolingPrior`](@ref), which resolve view names against a universe. The method also passes `cache_reg`, so that such a prior stores the cache that a refit under new scenario weights reads from `fpr`.
 3. Reduce `X` to its Coverage Universe with [`coverage_reduction`](@ref). When the caller gives `active_mask`, the reduction reads it in place of the active mask of `pnl`. Take the view of `pe.re` at the Coverage Universe with [`coverage_regression`](@ref). Fit the loadings and rebuild the asset returns with [`factor_reconstruction`](@ref) to get `rr` and `posterior_X`.
 4. Project `f_prior.mu` and `f_prior.sigma` through `rr` with [`factor_lift`](@ref) to get `mu`, `sigma`, `chol`, `esigma`, `edof` and `ediv`.
 5. Write `esigma`, `edof` and `ediv` onto `rr` with [`set_idiosyncratic_covariance`](@ref). Under `pe.rsd = true` the fields hold the residual variances that the lift measured and their counts. Under `pe.rsd = false`, `esigma` and `ediv` hold `nothing`, because the lift added no residual block.
 6. Let `r` be the rows that `f_prior.X` carries, which are the last `size(f_prior.X, 1)` observations. A Scenario Cap of `pe.pe` keeps only the last rows of `F`. The asset side keeps the same rows, so the two blocks describe the same observations. With no cap, `r` is every observation. Steps 2 to 4 fit the moments over every observation, so the cut changes no moment.
 7. Write `r` and `ts[r]` onto `rr` as its row key with [`set_row_key`](@ref), or `nothing` for `ts` when `ts` is `nothing`.
 8. Assemble a [`LowOrderPrior`](@ref) over the rows `r` of `posterior_X`, with the rows `r` of the oriented `X` under `o_X`, both cut by [`scenario_window`](@ref). It holds the three lifted moments, the `w`, `ens`, `kld` and `ow` of the factor prior, the regression result under `rr`, and `f_prior` itself under `fpr`. It holds no Feature Matrix `Z`, because a Feature Matrix comes from the Asset Panel on the returns data, or from a producer on the distance that reads the loadings off this result.

# Arguments

  - `pe`: Factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:pnl_prior]) The nested prior fits the factors, and no panel describes the factor axis, so the method does not pass the panel on.
  - $(arg_dict[:dims])
  - $(arg_dict[:strict])
  - $(arg_dict[:ts]) It has one entry for each observation of `X`, and the loadings result records the entries of the rows the prior result carries, or `nothing`. The method that takes a [`ReturnsResult`](@ref) passes `rd.ts`.
  - $(arg_dict[:cvg_active_mask])
  - $(arg_dict[:cache_reg])
  - `kwargs...`: Additional keyword arguments passed to matrix processing and estimators.

# Validation

  - `dims in (1, 2)`.
  - If `ts` is not `nothing`, `length(ts)` equals the number of observations of `X`. Otherwise the method raises a `DimensionMismatch`.

# Returns

  - `pr::LowOrderPrior`: The prior result. It holds the posterior asset returns, the mean vector, the covariance matrix, the square-root factor `chol`, the regression result and the factor prior result.

# Related

  - [`FactorPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`factor_reconstruction`](@ref)
  - [`factor_lift`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::FactorPrior, X::MatNum, F::MatNum, pnl::Option{<:AssetPanel} = nothing;
               dims::Int = 1, strict::Bool = false, ts::Option{<:VecDate} = nothing,
               active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
               cache_reg::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    T = size(X, 1)
    @argcheck(isnothing(ts) || length(ts) == T,
              DimensionMismatch("ts ($(length(ts))) must have one entry for each of the $T observations of X"))
    # The regression is a per-asset fit, so it takes the Coverage Universe before it runs
    # rather than a frame after it: a stepwise search over a gapped column has no answer, and
    # `chol` is the factorisation of the block, not of the frame. Every block the result
    # carries is expanded at the end, the regression result included, because a Prior Result
    # has no mask field and every block it carries lives on the full asset universe.
    cmsk, Xc = coverage_reduction(X, pnl; dims = 1,
                                  active_mask = dims_oriented(dims, active_mask))
    # `strict` reaches the wrapped prior: `pe.pe` admits `BlackLittermanPrior` and
    # `EntropyPoolingPrior`, both of which resolve view names against a universe and honour it.
    # The factor prior stores its cache for a refit under new scenario weights, which reads
    # it from `fpr` (#1751).
    f_prior = prior(pe.pe, F; strict = strict, cache_reg = cache_reg)
    rr, posterior_X = factor_reconstruction(coverage_regression(pe.re, cmsk), Xc, F)
    (; mu, sigma, chol, esigma, edof, ediv) = factor_lift(pe.mp, pe.ve, pe.rsd, rr,
                                                          f_prior.mu, f_prior.sigma, Xc,
                                                          posterior_X;
                                                          mtx_sqrt = pe.mtx_sqrt, kwargs...)
    # The lift already measured the residual variances, so the block carries them instead of
    # making every consumer recompute them from the reconstruction error. Under `rsd = false`
    # the lift added no residual block and `esigma` is `nothing`, which is what the field then
    # holds. The counts of the variances travel beside them, for a consumer that prices their
    # sampling error.
    rr = set_idiosyncratic_covariance(rr, esigma, edof, ediv)
    # A Scenario Cap of the factor prior keeps the last rows of `F` alone, so the asset side
    # keeps the same rows: the two blocks describe the same observations, and the row key
    # names them. The moments above were fitted over every observation.
    n = size(f_prior.X, 1)
    r = (T - n + 1):T
    rr = set_row_key(rr, r, nothing_scalar_array_getindex(ts, r))
    # No panel travels on a prior result at all: a Feature Matrix is derived from the Asset
    # Panel on the returns data, or built by a producer on the distance that reads the
    # loadings back off this result.
    #
    # The factor block *is* the prior that was fit on the factors: it needs no reconstruction,
    # because nothing here modifies the factor distribution — the asset moments are its
    # projection through `rr`.
    #
    # The asset-side `w` is the factor prior's: this estimator wraps only a factor prior, and
    # `posterior_X = F*M' + b'` has exactly `F`'s rows, so it is the only weighting in
    # existence and it is over the right observation axis. Its `ens`/`kld`/`ow` travel with it
    # — a weighting with no provenance cannot be interrogated (ADR 0046), and `ens` is what
    # sizes every uncertainty set built on this result.
    return LowOrderPrior(; X = expand_columns(scenario_window(n, posterior_X), cmsk),
                         o_X = scenario_window(n, X), mu = expand_vector(mu, cmsk),
                         sigma = expand_moment(sigma, cmsk),
                         chol = expand_columns(chol, cmsk), w = f_prior.w,
                         ens = f_prior.ens, kld = f_prior.kld, ow = f_prior.ow,
                         rr = expand_regression(rr, cmsk), fpr = f_prior)
end
"""
    prior(pe::FactorPrior, rd::ReturnsResult; strict::Bool = false, dims::Int = 1,
          cache_reg::Bool = false) -> LowOrderPrior

Fit a Factor Prior from a [`ReturnsResult`](@ref).

This method passes the fields of `rd` to the returns-matrix method, which runs the fit. It replaces [`prior(pe::AbstractPriorEstimator, rd::ReturnsResult)`](@ref) for this estimator, so that the timestamps `rd.ts` reach the fit. The loadings result then records the timestamp of each observation that the prior result carries.

# Algorithm

 1. Check that `rd` carries asset returns and factor returns.
 2. Drop the observations of a Shared Series gap with [`shared_series_view`](@ref): a gap of the factor returns, and a gap of a vector benchmark under `pe.x_src = ExcessSource()`. The factor prior, the regression and the reconstruction then read the same observations.
 3. Read the asset returns with [`benchmark_relative_returns`](@ref) under `pe.x_src`, giving `X`.
 4. Call the returns-matrix method with `X`, `rd.F` and `rd.pnl`, forwarding `rd.iv`, `rd.ivpa`, `rd.ne`, `rd.E`, `rd.ts`, `strict` and `cache_reg` as keyword arguments, and `dims = 1`, because a `ReturnsResult` holds its observations along the rows.
 5. Record the observations that step 2 dropped with [`shared_series_record`](@ref), and `pe.x_src` with [`stamp_returns_source`](@ref).

# Arguments

  - `pe`: Factor prior estimator.
  - $(arg_dict[:rd]) It must carry asset returns in `rd.X` and factor returns in `rd.F`.
  - $(arg_dict[:strict_prior_rd])
  - $(arg_dict[:dims_rd])
  - $(arg_dict[:cache_reg])

# Validation

  - `rd.X` is not `nothing`. Otherwise the method raises an [`IsNothingError`](@ref).
  - The rules of [`benchmark_relative_returns`](@ref).
  - `rd.F` is not `nothing`, through [`assert_factor_returns`](@ref).
  - The rules of the returns-matrix method.

# Returns

  - `pr::LowOrderPrior`: The prior the returns-matrix method fitted.

# Related

  - [`FactorPrior`](@ref)
  - [`prior`](@ref)
  - [`ReturnsResult`](@ref)
  - [`Regression`](@ref)
"""
function prior(pe::FactorPrior, rd::ReturnsResult; strict::Bool = false, dims::Int = 1,
               cache_reg::Bool = false)
    @argcheck(!isnothing(rd.X), IsNothingError)
    assert_factor_returns(pe, rd.F)
    T = size(rd.X, 1)
    rows = shared_series_rows(rd, pe.x_src, true)
    rd = shared_series_view(rd, pe.x_src, true)
    pr = prior(pe, benchmark_relative_returns(rd, pe.x_src), rd.F, rd.pnl; iv = rd.iv,
               ivpa = rd.ivpa, ne = rd.ne, E = rd.E, ts = rd.ts, strict = strict, dims = 1,
               cache_reg = cache_reg)
    return stamp_returns_source(shared_series_record(pr, rows, T), pe)
end

export FactorPrior
