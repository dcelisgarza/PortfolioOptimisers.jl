"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all implied volatility algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing implied volatility estimation algorithms should be subtypes of `ImpliedVolatilityAlgorithm`.

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatilityPremium`](@ref)
  - [`ImpliedVolatility`](@ref)
"""
abstract type ImpliedVolatilityAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implied volatility algorithm that predicts realised volatility via regression on implied volatility.

`ImpliedVolatilityRegression` fits a regression model relating implied and realised volatility over rolling windows, then uses the fitted model to predict the next period's realised volatility from the most recent implied volatility observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ImpliedVolatilityRegression(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        ws::Number = 20,
        re::AbstractRegressionTarget = LinearModel()
    ) -> ImpliedVolatilityRegression

Keywords correspond to the struct's fields.

## Validation

  - `ws > 2`.

# Examples

```jldoctest
julia> ImpliedVolatilityRegression()
ImpliedVolatilityRegression
  ve ┼ SimpleVariance
     │          me ┼ SimpleExpectedReturns
     │             │   w ┴ nothing
     │           w ┼ nothing
     │   corrected ┴ Bool: true
  ws ┼ Int64: 20
  re ┼ LinearModel
     │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`ImpliedVolatilityAlgorithm`](@ref)
  - [`ImpliedVolatilityPremium`](@ref)
  - [`ImpliedVolatility`](@ref)
"""
@concrete struct ImpliedVolatilityRegression <: ImpliedVolatilityAlgorithm
    """
    $(field_dict[:ve])
    """
    ve
    """
    Window size for computing rolling realised volatility.
    """
    ws
    # crit
    """
    $(field_dict[:retgt])
    """
    re
    function ImpliedVolatilityRegression(ve::AbstractVarianceEstimator, ws::Number,
                                         re::AbstractRegressionTarget)
        @argcheck(2 < ws, DomainError)
        return new{typeof(ve), typeof(ws), typeof(re)}(ve, ws, re)
    end
end
function ImpliedVolatilityRegression(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                     ws::Number = 20,
                                     #  crit::AbstractStepwiseRegressionCriterion = RSquared(),
                                     re::AbstractRegressionTarget = LinearModel())::ImpliedVolatilityRegression
    return ImpliedVolatilityRegression(ve, ws, re)
end
"""
$(DocStringExtensions.TYPEDEF)

Implied volatility algorithm that scales implied volatility by a user-supplied premium factor.

The premium factor can be a scalar or a vector (one per asset) and is used to convert raw implied volatilities into predicted realised volatilities.

# Related

  - [`ImpliedVolatilityAlgorithm`](@ref)
  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatility`](@ref)
"""
struct ImpliedVolatilityPremium <: ImpliedVolatilityAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Covariance estimator based on implied volatility scaling.

`ImpliedVolatility` computes a covariance matrix by combining a base correlation estimator with predicted realised volatilities derived from implied volatility data. It supports two algorithms: [`ImpliedVolatilityRegression`](@ref), which fits a regression model to predict realised volatility from implied volatility, and [`ImpliedVolatilityPremium`](@ref), which scales implied volatility by a user-supplied factor.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ImpliedVolatility(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        alg::ImpliedVolatilityAlgorithm = ImpliedVolatilityRegression(),
        af::Number = 252
    ) -> ImpliedVolatility

Keywords correspond to the struct's fields.

## Validation

  - `af > 0`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@prop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`ImpliedVolatilityAlgorithm`](@ref)
  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatilityPremium`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct ImpliedVolatility <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    @prop ce
    """
    $(field_dict[:mp])
    """
    mp
    """
    Implied volatility algorithm for predicting realised volatility.
    """
    alg
    """
    Annualisation factor for converting annualised implied volatility to the data frequency.
    """
    af
    function ImpliedVolatility(ce::StatsBase.CovarianceEstimator,
                               mp::AbstractMatrixProcessingEstimator,
                               alg::ImpliedVolatilityAlgorithm, af::Number)
        @argcheck(zero(af) < af, DomainError)
        return new{typeof(ce), typeof(mp), typeof(alg), typeof(af)}(ce, mp, alg, af)
    end
end
#= Old factory function:
function factory(ce::ImpliedVolatility, w::ObsWeights)::ImpliedVolatility
    return ImpliedVolatility(; ce = factory(ce.ce, w), mp = ce.mp)
end
=#
function ImpliedVolatility(; ce::StatsBase.CovarianceEstimator = Covariance(),
                           mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                           alg::ImpliedVolatilityAlgorithm = ImpliedVolatilityRegression(),
                           af::Number = 252)::ImpliedVolatility
    return ImpliedVolatility(ce, mp, alg, af)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the covariance estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:ce])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:cev])

# Related

  - [`ImpliedVolatility`](@ref)
"""
function moment_view(ce::ImpliedVolatility, i)::ImpliedVolatility
    return ImpliedVolatility(; ce = moment_view(ce.ce, i), mp = ce.mp)
end
"""
    realised_vol(ce::AbstractVarianceEstimator, X::MatNum, ws::Integer,
                 chunk::Option{<:Integer} = nothing, T::Option{<:Integer} = nothing,
                 N::Option{<:Integer} = nothing)

Compute realised volatility over non-overlapping rolling windows.

This function reshapes the last `chunk * ws` rows of `X` into blocks of size `ws` and computes the standard deviation within each block using the estimator `ce`. The result is a matrix of size `(chunk, N)` representing rolling realised volatilities.

# Arguments

  - `ce`: Variance estimator used to compute standard deviations within each window.
  - `X`: Data matrix of asset returns (observations × assets).
  - `ws`: Window size (number of observations per block).
  - `chunk`: Number of windows (computed as `div(T, ws)` if not provided).
  - `T`: Total number of observations (inferred from `X` if not provided).
  - `N`: Number of assets (inferred from `X` if not provided).

# Returns

  - `rv::Matrix{<:Number}`: Rolling realised volatility matrix (chunks × assets).

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`implied_vol`](@ref)
"""
function realised_vol(ce::AbstractVarianceEstimator, X::MatNum, ws::Integer,
                      chunk::Option{<:Integer} = nothing, T::Option{<:Integer} = nothing,
                      N::Option{<:Integer} = nothing)
    if isnothing(chunk) || isnothing(T) || isnothing(N)
        T, N = size(X)
        chunk = div(T, ws)
    end
    return dropdims(Statistics.std(ce,
                                   reshape(view(X, (1 + T - chunk * ws):T, :), ws, chunk,
                                           N); dims = 1); dims = 1)
end
"""
    implied_vol(X::MatNum, ws::Integer, chunk::Option{<:Integer} = nothing,
                T::Option{<:Integer} = nothing, N::Option{<:Integer} = nothing)

Extract non-overlapping implied volatility observations from `X` at the end of each rolling window.

This function selects rows of `X` at positions corresponding to the end of each rolling window of size `ws`, starting from row `T - (chunk - 1) * ws` and sampling every `ws` rows. It returns a view of shape `(chunk, N)` containing end-of-window implied volatility values.

# Arguments

  - `X`: Implied volatility matrix (observations × assets).
  - `ws`: Window size (number of observations per block).
  - `chunk`: Number of windows (computed as `div(T, ws)` if not provided).
  - `T`: Total number of observations (inferred from `X` if not provided).
  - `N`: Number of assets (inferred from `X` if not provided).

# Returns

  - `iv::SubArray`: End-of-window implied volatility matrix (chunks × assets).

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`realised_vol`](@ref)
"""
function implied_vol(X::MatNum, ws::Integer, chunk::Option{<:Integer} = nothing,
                     T::Option{<:Integer} = nothing, N::Option{<:Integer} = nothing)
    if isnothing(chunk) || isnothing(T) || isnothing(N)
        T, N = size(X)
        chunk = div(T, ws)
    end
    return view(X, (T - (chunk - 1) * ws):ws:T, :)
end
"""
    predict_realised_vols(alg::ImpliedVolatilityPremium, iv::MatNum, ::Any, ivpa::Nothing)

Error method: [`ImpliedVolatilityPremium`](@ref) requires an implied volatility premium adjustment factor.

Throws an `ArgumentError` because `ImpliedVolatilityPremium` requires `ivpa` to be a `<:Number` or `<:VecNum`.

# Arguments

  - `alg`: Implied volatility premium algorithm.
  - `iv`: Implied volatility matrix (unused).
  - `ivpa::Nothing`: Implied volatility premium adjustment (must not be `nothing`).

# Related

  - [`ImpliedVolatilityPremium`](@ref)
  - [`predict_realised_vols`](@ref)
"""
function predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any, ivpa::Nothing)
    throw(ArgumentError("ImpliedVolatilityPremium requires `ivpa` to be a `<:Number` or `<:VecNum`"))
end
"""
    predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any,
                          ivpa::Num_VecNum)

Predict realised volatilities by scaling the latest implied volatility row by the premium adjustment factor.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_i &= \\frac{\\mathrm{iv}_{T,i}}{\\mathrm{ivpa}_i}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_i``: Predicted realised volatility for asset ``i``.
  - ``\\mathrm{iv}_{T,i}``: Latest implied volatility for asset ``i`` (last row).
  - ``\\mathrm{ivpa}_i``: Implied volatility premium adjustment factor for asset ``i``.

# Arguments

  - `::ImpliedVolatilityPremium`: Implied volatility premium algorithm.
  - `iv`: Implied volatility matrix (chunks × assets); the last row is used.
  - `ivpa`: Implied volatility premium adjustment factor (scalar or vector).

# Returns

  - `rv::AbstractArray`: Predicted realised volatilities (last row of `iv` divided by `ivpa`).

# Related

  - [`ImpliedVolatilityPremium`](@ref)
  - [`ImpliedVolatility`](@ref)
"""
function predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any,
                               ivpa::Num_VecNum)
    return view(iv, size(iv, 1), :) ⊘ ivpa
end
"""
    predict_realised_vols(alg::ImpliedVolatilityRegression, iv::MatNum, X::MatNum, ::Any)

Predict realised volatilities using a regression model fitted on implied and realised volatility.

For each asset, this function fits a regression model relating lagged implied volatility and lagged realised volatility (computed from rolling windows of `X`) to the next-period realised volatility. The fitted model is then used to predict the next-period realised volatility from the most recent data.

# Mathematical definition

For asset ``i``, fit the log-linear model over windows ``t = 1, \\ldots, T-1``:

```math
\\begin{align}
\\ln \\hat{\\sigma}^{\\mathrm{rv}}_{t+1,i} &= \\beta_0 + \\beta_1 \\ln \\sigma^{\\mathrm{iv}}_{t,i} + \\beta_2 \\ln \\hat{\\sigma}^{\\mathrm{rv}}_{t,i} + \\varepsilon_t\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}^{\\mathrm{rv}}_{t+1,i}``: Predicted realised volatility for asset ``i`` at time ``t+1``.
  - ``\\sigma^{\\mathrm{iv}}_{t,i}``: Implied volatility for asset ``i`` at time ``t``.
  - ``\\beta_0, \\beta_1, \\beta_2``: Regression coefficients.
  - ``\\varepsilon_t``: Regression residual.

Then predict:

```math
\\begin{align}
\\hat{\\sigma}^{\\mathrm{rv}}_{T+1,i} &= \\exp\\!\\left(\\hat{\\beta}_0 + \\hat{\\beta}_1 \\ln \\sigma^{\\mathrm{iv}}_{T,i} + \\hat{\\beta}_2 \\ln \\hat{\\sigma}^{\\mathrm{rv}}_{T,i}\\right)\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}^{\\mathrm{rv}}_{T+1,i}``: Predicted next-period realised volatility for asset ``i``.
  - ``\\hat{\\beta}_0, \\hat{\\beta}_1, \\hat{\\beta}_2``: Fitted regression coefficients.

# Arguments

  - `alg`: Implied volatility regression algorithm specifying the window size and regression target.
  - `iv`: Implied volatility matrix (observations × assets).
  - `X`: Asset returns matrix (observations × assets) used to compute realised volatility.
  - `::Any`: Ignored (placeholder for `ivpa`).

# Returns

  - `rv_p::Vector{<:Number}`: Predicted next-period realised volatilities (one per asset).

# Validation

  - `chunk > 2` (i.e., there must be more than 2 windows of data to fit the regression).

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`realised_vol`](@ref)
  - [`implied_vol`](@ref)
"""
function predict_realised_vols(alg::ImpliedVolatilityRegression, iv::MatNum, X::MatNum,
                               ::Any)
    T, N = size(X)
    chunk = div(T, alg.ws)
    @argcheck(2 < chunk, DomainError)
    rv = realised_vol(alg.ve, X, alg.ws, chunk, T, N)
    iv = implied_vol(iv, alg.ws, chunk, T, N)
    @argcheck(size(rv) == size(iv), DimensionMismatch)
    T2 = size(iv, 1)
    rv = log.(rv)
    iv = log.(iv)
    # criterion_func = regression_criterion_func(alg.crit)
    ovec = range(one(promote_type(eltype(rv), eltype(iv))),
                 one(promote_type(eltype(rv), eltype(iv))); length = T2 - 1)
    # reg = Matrix{promote_type(eltype(rv), eltype(iv))}(undef, N, 3)
    # r2s = Vector{promote_type(eltype(rv), eltype(iv))}(undef, N)
    rv_p = Vector{promote_type(eltype(rv), eltype(iv))}(undef, N)
    # fr = []
    for i in 1:N
        X = [view(iv, :, i) view(rv, :, i)]
        X_t = [ovec view(X, 1:(T2 - 1), :)]
        X_p = [one(eltype(X)) transpose(view(X, T2, :))]
        y_t = view(rv, 2:T2, i)
        fri = StatsAPI.fit(alg.re, X_t, y_t)
        # params = StatsAPI.coef(fri)
        # reg[i, 1] = params[1]
        # reg[i, 2:3] .= params[2:end]
        # r2s[i] = criterion_func(fri)
        rv_pi = StatsAPI.predict(fri, X_p)[1]
        rv_p[i] = exp(rv_pi)
        # push!(fr, fri)
    end
    #, Regression(; b = view(reg, :, 1), M = view(reg, :, 2:3)), r2s, fr
    return rv_p
end
"""
    Statistics.cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                   iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)

Compute the covariance matrix using implied volatility scaling.

This method computes the correlation matrix of `X` using the base estimator in `ce`, then predicts realised volatilities from `iv` using the implied volatility algorithm in `ce.alg`. The predicted realised volatilities are used to convert the correlation matrix to a covariance matrix, which is then post-processed by the matrix processing estimator `ce.mp`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\mathrm{diag}(\\hat{\\boldsymbol{\\sigma}}) \\hat{\\boldsymbol{\\rho}} \\,\\mathrm{diag}(\\hat{\\boldsymbol{\\sigma}})\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{\\Sigma}}``: Implied volatility-scaled covariance matrix.
  - ``\\hat{\\boldsymbol{\\rho}} = \\operatorname{cor}(\\mathbf{X})``: Correlation matrix from asset returns.
  - ``\\hat{\\boldsymbol{\\sigma}}``: Predicted realised volatilities from ``\\mathbf{iv} / \\sqrt{\\mathrm{af}}``.

# Arguments

  - `ce`: Implied volatility covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean (passed to the base estimator).
  - `iv`: Implied volatility matrix (observations × assets).
  - `ivpa`: Optional implied volatility premium adjustment factor (required for [`ImpliedVolatilityPremium`](@ref)).
  - `kwargs...`: Additional keyword arguments passed to the base estimator.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref)
"""
function Statistics.cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        iv = transpose(iv)
    end
    sigma = Statistics.cor(ce.ce, X; dims = 1, mean = mean, iv = iv, kwargs...)
    iv = iv / sqrt(ce.af)
    iv = predict_realised_vols(ce.alg, X, iv, ivpa)
    StatsBase.cov2cor!(sigma, iv)
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end
"""
    Statistics.cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                   iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)

Compute the correlation matrix using implied volatility scaling.

This method computes the correlation matrix of `X` using the base estimator in `ce`, then predicts realised volatilities from `iv` using the implied volatility algorithm in `ce.alg`. The predicted realised volatilities are used to convert the correlation matrix to a covariance and back to a correlation matrix, which is then post-processed by the matrix processing estimator `ce.mp`.

# Arguments

  - `ce`: Implied volatility covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean (passed to the base estimator).
  - `iv`: Implied volatility matrix (observations × assets).
  - `ivpa`: Optional implied volatility premium adjustment factor (required for [`ImpliedVolatilityPremium`](@ref)).
  - `kwargs...`: Additional keyword arguments passed to the base estimator.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref)
"""
function Statistics.cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        iv = transpose(iv)
    end
    rho = Statistics.cor(ce.ce, X; dims = 1, mean = mean, iv = iv, kwargs...)
    iv = iv / sqrt(ce.af)
    iv = predict_realised_vols(ce.alg, X, iv, ivpa)
    StatsBase.cor2cov!(rho, iv)
    StatsBase.cov2cor!(rho)
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end
export ImpliedVolatility, ImpliedVolatilityPremium, ImpliedVolatilityRegression
