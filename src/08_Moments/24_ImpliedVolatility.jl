"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all implied volatility algorithms.

All concrete and/or abstract types implementing implied volatility estimation algorithms should be subtypes of `ImpliedVolatilityAlgorithm`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `ImpliedVolatilityAlgorithm` and implement the following method:

## Required method name

  - `predict_realised_vols(alg::ImpliedVolatilityAlgorithm, iv::MatNum, X::MatNum, ivpa::Any)`: Predict the realised volatility of the period that follows the sample, one value per asset.

### Arguments

The implied volatilities are the **second** positional argument and the returns the **third**. The two are matrices of the same size, so a call that swaps them is well typed and silently wrong.

  - `alg`: The concrete subtype instance.
  - `iv`: Implied volatility matrix `observations × assets`, already divided by ``\\sqrt{\\mathrm{af}}`` by the caller.
  - `X`: Asset returns matrix `observations × assets`.
  - `ivpa`: Implied volatility premium adjustment factor. It is `nothing` when the caller supplies none, so an algorithm that needs one raises on that method.

### Returns

  - `rv_p::VecNum`: Predicted realised volatility, one entry per asset, in the units of `X`.

### Examples

```jldoctest
julia> struct MyImpliedVolatilityAlgorithm <: PortfolioOptimisers.ImpliedVolatilityAlgorithm end

julia> function PortfolioOptimisers.predict_realised_vols(::MyImpliedVolatilityAlgorithm,
                                                          iv::PortfolioOptimisers.MatNum, ::Any,
                                                          ::Any)
           return vec(iv[end, :])
       end

julia> cov(ImpliedVolatility(; alg = MyImpliedVolatilityAlgorithm(), af = 1),
           [0.1 0.2; 0.3 0.1; 0.2 0.4]; iv = [0.5 0.6; 0.4 0.7; 0.3 0.8])
2×2 Matrix{Float64}:
  0.09       -0.0785584
 -0.0785584   0.64
```

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatilityPremium`](@ref)
  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref)

# References

  - $(ref_dict[:andersen2006])
"""
abstract type ImpliedVolatilityAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implied volatility algorithm that predicts realised volatility via regression on implied volatility.

`ImpliedVolatilityRegression` fits a regression model relating implied and realised volatility over rolling windows, then uses the fitted model to predict the next period's realised volatility from the most recent implied volatility observation. The model, the steps that fit it and the number of windows it needs are stated by [`predict_realised_vols`](@ref), which is the method this tag selects.

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ve`: Recursively updated via [`factory`](@ref).

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
  - [`predict_realised_vols`](@ref): the model and the steps of the branch this tag selects.
  - [`realised_vol`](@ref)
  - [`implied_vol`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:christensenprabhala1998])
  - $(ref_dict[:christensenhansen2002])
  - $(ref_dict[:andersen2006])
"""
@propagatable @concrete struct ImpliedVolatilityRegression <: ImpliedVolatilityAlgorithm
    """
    $(field_dict[:ve])
    """
    @fprop ve
    """
    Window size for computing rolling realised volatility. It also sets the number of windows, `div(size(X, 1), ws)`, and the regression needs more than two of them.
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
                                     #  crit::Union{Symbol, MinMaxValStepwiseRegressionCriterion,
                                     #              AbstractStepwiseRegressionCriterion} = :r2,
                                     re::AbstractRegressionTarget = LinearModel())::ImpliedVolatilityRegression
    return ImpliedVolatilityRegression(ve, ws, re)
end
"""
$(DocStringExtensions.TYPEDEF)

Implied volatility algorithm that divides the latest implied volatility by a volatility risk premium adjustment.

The adjustment factor is not a field of this type. The caller passes it as the `ivpa` keyword of the `cov` and `cor` methods of [`ImpliedVolatility`](@ref), as a scalar or as one value per asset. The factor is mandatory: `ivpa = nothing` raises an `ArgumentError`. Every entry of it must be finite and strictly positive, and one that is not raises a `DomainError`, because a non-positive factor makes a negative volatility whose sign `StatsBase.cor2cov!` then hides. The closed form of the branch, and the rules it enforces, are stated by [`predict_realised_vols`](@ref), which is the method this tag selects.

# Constructors

    ImpliedVolatilityPremium() -> ImpliedVolatilityPremium

# Examples

```jldoctest
julia> ImpliedVolatilityPremium()
ImpliedVolatilityPremium()
```

# Related

  - [`ImpliedVolatilityAlgorithm`](@ref)
  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref): the closed form of the branch this tag selects.

# References

  - $(ref_dict[:egbersswinkels2015])
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

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `alg`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ImpliedVolatility()
ImpliedVolatility
   ce ┼ Covariance
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    ce ┼ GeneralCovariance
      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │    w ┴ nothing
      │   alg ┼ FullMoment()
      │     w ┴ nothing
   mp ┼ MatrixProcessing
      │     pdm ┼ Posdef
      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      dn ┼ nothing
      │      dt ┼ nothing
      │     alg ┼ nothing
      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
  alg ┼ ImpliedVolatilityRegression
      │   ve ┼ SimpleVariance
      │      │          me ┼ SimpleExpectedReturns
      │      │             │   w ┴ nothing
      │      │           w ┼ nothing
      │      │   corrected ┴ Bool: true
      │   ws ┼ Int64: 20
      │   re ┼ LinearModel
      │      │   kwargs ┴ @NamedTuple{}: NamedTuple()
   af ┴ Int64: 252
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`ImpliedVolatilityAlgorithm`](@ref)
  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatilityPremium`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:andersen2006])
"""
@propagatable @concrete struct ImpliedVolatility <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:mp])
    """
    mp
    """
    Implied volatility algorithm for predicting realised volatility.
    """
    @fprop alg
    """
    Annualisation factor for converting annualised implied volatility to the data frequency. The `cov` and `cor` methods divide the implied volatilities by `sqrt(af)` before the algorithm reads them.
    """
    af
    function ImpliedVolatility(ce::StatsBase.CovarianceEstimator,
                               mp::AbstractMatrixProcessingEstimator,
                               alg::ImpliedVolatilityAlgorithm, af::Number)
        @argcheck(zero(af) < af, DomainError)
        return new{typeof(ce), typeof(mp), typeof(alg), typeof(af)}(ce, mp, alg, af)
    end
end
function ImpliedVolatility(; ce::StatsBase.CovarianceEstimator = Covariance(),
                           mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                           alg::ImpliedVolatilityAlgorithm = ImpliedVolatilityRegression(),
                           af::Number = 252)::ImpliedVolatility
    return ImpliedVolatility(ce, mp, alg, af)
end
"""
    realised_vol(ce::AbstractVarianceEstimator, X::MatNum, ws::Integer,
                 chunk::Option{<:Integer} = nothing, T::Option{<:Integer} = nothing,
                 N::Option{<:Integer} = nothing)

Compute realised volatility over non-overlapping rolling windows.

This function splits the last `chunk * ws` rows of `X` into `chunk` non-overlapping blocks of `ws` rows each, and computes the standard deviation of every asset within each block using the estimator `ce`. The result is a matrix of size `(chunk, N)` representing rolling realised volatilities.

Any estimator declared with [`@propagatable`](@ref) gets its [`obs_weights_view`](@ref) method generated from the [`@wprop`](@ref) tag its weights field already carries, so both shipped variance estimators answer with nothing written by hand. An estimator that holds weights outside that tag and defines no method of its own meets a block of `ws` rows with a vector of `T` weights, and the call raises.

# Mathematical definition

Write ``C`` for `chunk` and ``w_s`` for `ws`. The blocks are counted back from the last observation, so the leading

```math
\\begin{align}
o &= T - C w_s
\\end{align}
```

rows lie in no block. Block ``c`` holds the rows

```math
\\begin{align}
\\mathcal{R}_c &= \\{o + (c - 1) w_s + 1,\\, \\ldots,\\, o + c w_s\\}\\,, \\qquad c = 1, \\ldots, C\\,,
\\end{align}
```

and the realised volatility of asset ``i`` over that block is

```math
\\begin{align}
\\mathrm{rv}_{c,\\,i} &= \\operatorname{std}\\left(\\{x_{t,\\,i} : t \\in \\mathcal{R}_c\\}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:T])
  - $(math_dict[:x_ti_ret])
  - ``o``: Number of leading rows that no block covers.
  - ``\\mathcal{R}_c``: Rows of block ``c``.
  - ``\\mathrm{rv}_{c,\\,i}``: Realised volatility of asset ``i`` over block ``c``.

The last row of block ``c`` is ``o + c w_s``, which is the row [`implied_vol`](@ref) samples for the same window. A window count that does not divide ``T`` therefore drops the oldest rows and never the newest.

# Algorithm

 1. When `chunk`, `T` or `N` is `nothing`, read `T` and `N` from `size(X)` and set `chunk` to `div(T, ws)`.
 2. Compute `offset`, the number of leading rows of `X` that no block covers.
 3. For each block `c` in `1:chunk`, form `rows`, the `ws` row indices of that block.
 4. Slice `ce` to `rows` with [`obs_weights_view`](@ref), giving the estimator that measures this block. It indexes every weights field on its own, so an estimator with a weighted mean and an unweighted dispersion keeps that shape, and an unweighted `ce` is returned unchanged.
 5. Call `Statistics.std` on that estimator and on the block's rows of `X` along `dims = 1`, giving one row of standard deviations. The estimator is called once per block, so it only ever reads a matrix.
 6. Stack the `chunk` rows with `vcat`, giving `rv`.

# Arguments

  - `ce`: Variance estimator used to compute standard deviations within each window. Its observation weights describe the whole sample, and step 4 slices them to each block.
  - `X`: Data matrix of asset returns (observations × assets).
  - `ws`: Window size (number of observations per block).
  - `chunk`: Number of windows (computed as `div(T, ws)` if not provided).
  - `T`: Total number of observations (inferred from `X` if not provided).
  - `N`: Number of assets (inferred from `X` if not provided).

# Returns

  - `rv::Matrix{<:Number}`: Rolling realised volatility matrix (chunks × assets).

# Examples

```jldoctest
julia> PortfolioOptimisers.realised_vol(SimpleVariance(),
                                        [0.1 0.2; 0.3 0.1; 0.2 0.4; 0.1 0.1; 0.4 0.2;
                                         0.2 0.3], 2)
3×2 Matrix{Float64}:
 0.141421   0.0707107
 0.0707107  0.212132
 0.141421   0.0707107
```

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`implied_vol`](@ref)
  - [`obs_weights_view`](@ref)
  - [`predict_realised_vols`](@ref)
"""
function realised_vol(ce::AbstractVarianceEstimator, X::MatNum, ws::Integer,
                      chunk::Option{<:Integer} = nothing, T::Option{<:Integer} = nothing,
                      N::Option{<:Integer} = nothing)
    if isnothing(chunk) || isnothing(T) || isnothing(N)
        T, N = size(X)
        chunk = div(T, ws)
    end
    offset = T - chunk * ws
    return mapreduce(vcat, 1:chunk) do c
        rows = (offset + (c - 1) * ws + 1):(offset + c * ws)
        return Statistics.std(obs_weights_view(ce, rows), view(X, rows, :); dims = 1)
    end
end
"""
    implied_vol(X::MatNum, ws::Integer, chunk::Option{<:Integer} = nothing,
                T::Option{<:Integer} = nothing, N::Option{<:Integer} = nothing)

Extract non-overlapping implied volatility observations from `X` at the end of each rolling window.

This function selects the rows of `X` that close each rolling window of size `ws`, and returns them as a view of shape `(chunk, N)`.

# Mathematical definition

Write ``C`` for `chunk` and ``w_s`` for `ws`. The sampled rows are

```math
\\begin{align}
\\mathcal{S} &= \\{T - (C - 1) w_s,\\, T - (C - 2) w_s,\\, \\ldots,\\, T\\}\\,,
\\end{align}
```

whose ``c``-th entry is

```math
\\begin{align}
\\mathcal{S}_c &= o + c w_s\\,, \\qquad o = T - C w_s\\,, \\qquad c = 1, \\ldots, C\\,.
\\end{align}
```

Where:

  - $(math_dict[:T])
  - ``\\mathcal{S}``: Sampled rows, one per window.
  - ``o``: Number of leading rows that no window covers.

``\\mathcal{S}_c`` is the last row of block ``c`` of [`realised_vol`](@ref), so the two functions read the same windows. The last sampled row is ``T`` whatever ``w_s`` is, so a window count that does not divide ``T`` drops the oldest rows and never the newest.

# Algorithm

 1. When `chunk`, `T` or `N` is `nothing`, read `T` and `N` from `size(X)` and set `chunk` to `div(T, ws)`.
 2. Return the view of `X` over the rows `(T - (chunk - 1) * ws):ws:T`. The result is a view, so it shares its memory with `X`.

# Arguments

  - `X`: Implied volatility matrix (observations × assets).
  - `ws`: Window size (number of observations per block).
  - `chunk`: Number of windows (computed as `div(T, ws)` if not provided).
  - `T`: Total number of observations (inferred from `X` if not provided).
  - `N`: Number of assets (inferred from `X` if not provided).

# Returns

  - `iv::SubArray`: End-of-window implied volatility matrix (chunks × assets).

# Examples

```jldoctest
julia> PortfolioOptimisers.implied_vol([0.1 0.2; 0.3 0.1; 0.2 0.4; 0.1 0.1; 0.4 0.2;
                                        0.2 0.3], 2)
3×2 view(::Matrix{Float64}, 2:2:6, :) with eltype Float64:
 0.3  0.1
 0.1  0.1
 0.2  0.3
```

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`realised_vol`](@ref)
  - [`predict_realised_vols`](@ref)
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
    predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any, ivpa::Nothing)

Error method: [`ImpliedVolatilityPremium`](@ref) requires an implied volatility premium adjustment factor.

The adjustment factor is not a field of [`ImpliedVolatilityPremium`](@ref), so a caller that selects that algorithm and passes no `ivpa` reaches this method.

# Arguments

The implied volatilities are the **second** positional argument and the returns the **third**.

  - `::ImpliedVolatilityPremium`: Implied volatility premium algorithm.
  - `iv`: Implied volatility matrix (unused).
  - `::Any`: Asset returns matrix (unused).
  - `ivpa::Nothing`: Implied volatility premium adjustment (must not be `nothing`).

# Validation

  - `ivpa` is not `nothing`. This method is the failing branch, and it raises an `ArgumentError`.

# Related

  - [`ImpliedVolatilityPremium`](@ref)
  - [`ImpliedVolatility`](@ref)
"""
function predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any, ivpa::Nothing)
    return throw(ArgumentError("ImpliedVolatilityPremium requires `ivpa` to be a `<:Number` or `<:VecNum`"))
end
"""
    predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any,
                          ivpa::Num_VecNum)

Predict realised volatilities by scaling the latest implied volatility row by the premium adjustment factor.

The row read is the last row of `iv` itself, not the last row of a window, so this method needs no window size and no returns.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}^{\\mathrm{rv}}_i &= \\frac{\\sigma^{\\mathrm{iv}}_{T,\\,i}}{\\mathrm{ivpa}_i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma_rv_hat_i])
  - ``\\sigma^{\\mathrm{iv}}_{T,\\,i}``: Implied volatility of asset ``i`` at the last observation.
  - ``\\mathrm{ivpa}_i``: Implied volatility premium adjustment factor for asset ``i``. A scalar applies to every asset.
  - $(math_dict[:T])

# Arguments

The implied volatilities are the **second** positional argument and the returns the **third**.

  - `::ImpliedVolatilityPremium`: Implied volatility premium algorithm.
  - `iv`: Implied volatility matrix (observations × assets); the last row is used.
  - `::Any`: Asset returns matrix (unused).
  - `ivpa`: Implied volatility premium adjustment factor (scalar or vector).

# Validation

  - Every entry of `ivpa` is finite and strictly positive. A non-positive factor turns a volatility negative, and `StatsBase.cor2cov!` hides the sign: it squares the factor on the diagonal, so a negative scalar returns the matrix its absolute value returns, and a negative entry of a vector flips the sign of every covariance of that asset alone. Both answers stay positive definite, so [`matrix_processing!`](@ref) finds nothing to repair and no later step sees the defect.
  - A vector `ivpa` carries one entry per asset. A wrong length raises a `DimensionMismatch` from the broadcast.

# Returns

  - `rv::AbstractArray`: Predicted realised volatilities (last row of `iv` divided by `ivpa`).

# Examples

```jldoctest
julia> PortfolioOptimisers.predict_realised_vols(ImpliedVolatilityPremium(),
                                                 [0.1 0.2; 0.3 0.1; 0.2 0.4; 0.1 0.1;
                                                  0.4 0.2; 0.2 0.3], nothing, 1.25)
2-element Vector{Float64}:
 0.16
 0.24
```

# Related

  - [`ImpliedVolatilityPremium`](@ref)
  - [`ImpliedVolatility`](@ref)

# References

  - $(ref_dict[:egbersswinkels2015])
"""
function predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any,
                               ivpa::Num_VecNum)
    @argcheck(all(x -> isfinite(x) && x > zero(x), ivpa), DomainError)
    return view(iv, size(iv, 1), :) ⊘ ivpa
end
"""
    predict_realised_vols(alg::ImpliedVolatilityRegression, iv::MatNum, X::MatNum, ::Any)

Predict realised volatilities using a regression model fitted on implied and realised volatility.

For each asset, this function fits a regression model relating the implied volatility and the realised volatility of one window to the realised volatility of the next window, then predicts from the last window. The windows are the blocks of [`realised_vol`](@ref) and the rows of [`implied_vol`](@ref), so both series are read over the same rows of the sample.

# Mathematical definition

Write ``C`` for the number of windows, ``\\mathrm{div}(T, w_s)``. For asset ``i``, fit the log-linear model over the windows ``c = 1, \\ldots, C-1``:

```math
\\begin{align}
\\ln \\sigma^{\\mathrm{rv}}_{c+1,\\,i} &= \\beta_0 + \\beta_1 \\ln \\sigma^{\\mathrm{iv}}_{c,\\,i} + \\beta_2 \\ln \\sigma^{\\mathrm{rv}}_{c,\\,i} + \\varepsilon_c\\,.
\\end{align}
```

Then predict from the last window:

```math
\\begin{align}
\\hat{\\sigma}^{\\mathrm{rv}}_i &= \\exp\\!\\left(\\hat{\\beta}_0 + \\hat{\\beta}_1 \\ln \\sigma^{\\mathrm{iv}}_{C,\\,i} + \\hat{\\beta}_2 \\ln \\sigma^{\\mathrm{rv}}_{C,\\,i}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma_rv_hat_i])
  - ``\\sigma^{\\mathrm{rv}}_{c,\\,i}``: Realised volatility of asset ``i`` over window ``c``.
  - ``\\sigma^{\\mathrm{iv}}_{c,\\,i}``: Implied volatility of asset ``i`` at the last row of window ``c``.
  - ``\\beta_0, \\beta_1, \\beta_2``: Regression coefficients.
  - ``\\varepsilon_c``: Regression residual.
  - $(math_dict[:T])

The fit takes ``C - 1`` rows, so ``C`` must exceed two for the model to have more rows than coefficients.

# Algorithm

 1. Read `T` and `N` from `size(X)`, and set `chunk` to `div(T, alg.ws)`.
 2. Check that `chunk` exceeds two.
 3. Call [`realised_vol`](@ref) with `alg.ve`, giving `rv`, the realised volatility of every window.
 4. Call [`implied_vol`](@ref) on `iv`, giving `iv`, the implied volatility at the last row of every window. The window count comes from `X`, so `iv` is read over the rows of the returns sample.
 5. Check that `rv` and `iv` have the same size.
 6. Replace `rv` and `iv` by their natural logarithms.
 7. Build `ovec`, the intercept column of ones, of length `T2 - 1`.
 8. For each asset `i`, build the design matrix `X_t` from `ovec` and the first `T2 - 1` rows of `iv` and `rv`, the response `y_t` from rows `2:T2` of `rv`, and the prediction row `X_p` from row `T2`.
 9. Fit `alg.re` on `X_t` and `y_t`, giving `fri`, then predict from `X_p` and exponentiate, giving `rv_p[i]`.
10. Return `rv_p`.

# Arguments

The implied volatilities are the **second** positional argument and the returns the **third**. Both are matrices of the same size, so a call that swaps them is well typed and silently wrong.

  - `alg`: Implied volatility regression algorithm specifying the variance estimator, the window size and the regression target.
  - `iv`: Implied volatility matrix (observations × assets).
  - `X`: Asset returns matrix (observations × assets) used to compute realised volatility. It also fixes the window count, so `iv` must have as many rows as `X`.
  - `::Any`: Ignored (placeholder for `ivpa`).

# Validation

  - `chunk > 2` (i.e., there must be more than 2 windows of data to fit the regression).
  - `size(rv) == size(iv)`, one realised volatility per implied volatility.

# Returns

  - `rv_p::Vector{<:Number}`: Predicted next-period realised volatilities (one per asset).

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatility`](@ref)
  - [`realised_vol`](@ref)
  - [`implied_vol`](@ref)

# References

  - $(ref_dict[:christensenprabhala1998])
  - $(ref_dict[:christensenhansen2002])
  - $(ref_dict[:andersen2006])
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
\\hat{\\mathbf{\\Sigma}} &= \\mathrm{diag}(\\hat{\\boldsymbol{\\sigma}}^{\\mathrm{rv}}) \\hat{\\boldsymbol{\\rho}} \\,\\mathrm{diag}(\\hat{\\boldsymbol{\\sigma}}^{\\mathrm{rv}})\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``\\hat{\\boldsymbol{\\rho}} = \\operatorname{cor}(\\mathbf{X})``: Correlation matrix from asset returns, computed by `ce.ce`.
  - ``\\hat{\\boldsymbol{\\sigma}}^{\\mathrm{rv}}``: Predicted realised volatilities, from ``\\mathbf{iv} / \\sqrt{\\mathrm{af}}``.

The diagonal of ``\\hat{\\mathbf{\\Sigma}}`` is therefore the square of the predicted realised volatility of each asset, and never a unit.

# Algorithm

 1. Orient `X` and `iv` to `observations × assets` with [`dims_oriented`](@ref), which validates `dims` and transposes both when `dims` is `2`.
 2. Check that `X` and `iv` have the same size, so row `t` of `iv` is the implied volatility of observation `t` of `X`.
 3. Call `Statistics.cor(ce.ce, X; dims = 1, mean = mean, iv = iv, kwargs...)`, giving `sigma`, the base correlation matrix. The oriented `iv` is forwarded so that a base estimator that reads its own implied volatility series, such as a nested [`ImpliedVolatility`](@ref), receives it. Every other shipped estimator absorbs it into its own `kwargs...` and ignores it.
 4. Divide `iv` by `sqrt(ce.af)`, converting the annualised implied volatility to the frequency of `X`.
 5. Call [`predict_realised_vols`](@ref) with `ce.alg`, giving `iv`, one predicted realised volatility per asset. The implied volatilities are the second argument and the returns the third.
 6. Scale `sigma` in place with `StatsBase.cor2cov!`, which applies the closed form above.
 7. Post-process `sigma` in place with [`matrix_processing!`](@ref) and `ce.mp`.

# Arguments

  - `ce`: Implied volatility covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean (passed to the base estimator).
  - `iv`: Implied volatility matrix, annualised, oriented as `X` and of the same size.
  - `ivpa`: Optional implied volatility premium adjustment factor (required for [`ImpliedVolatilityPremium`](@ref)).
  - `kwargs...`: Additional keyword arguments passed to the base estimator.

# Validation

  - `dims in (1, 2)`, by [`dims_oriented`](@ref).
  - `size(X) == size(iv)`, one implied volatility per return.
  - Whatever `ce.alg` refuses, by [`predict_realised_vols`](@ref). [`ImpliedVolatilityPremium`](@ref) needs an `ivpa` that is not `nothing`, and whose every entry is finite and strictly positive.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref)
  - [`cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing, iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
    X, iv = dims_oriented(dims, X, iv)
    @argcheck(size(X) == size(iv), DimensionMismatch)
    sigma = Statistics.cor(ce.ce, X; dims = 1, mean = mean, iv = iv, kwargs...)
    iv = iv / sqrt(ce.af)
    iv = predict_realised_vols(ce.alg, iv, X, ivpa)
    StatsBase.cor2cov!(sigma, iv)
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end
"""
    Statistics.cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                   iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)

Compute the correlation matrix using implied volatility scaling.

This method computes the correlation matrix of `X` using the base estimator in `ce`, normalises it, then post-processes it with the matrix processing estimator `ce.mp`.

A correlation is scale free, so the predicted realised volatilities cannot move the answer, and the returned matrix is the base correlation of `X`. The volatility model runs even so, because `cor` must refuse every configuration `cov` refuses: a `ce.alg` that cannot answer the call raises here as it does in `cov`. Its prediction is discarded, and never multiplied into `rho` and divided back out again.

That round trip was the identity in exact arithmetic alone. In floating point the round-off of one multiplication and one division moved an off-diagonal entry, and a predicted volatility of zero made the second call divide zero by zero. One asset whose last implied volatility was zero therefore turned a whole row and column of the correlation into `NaN`, and `matrix_processing!` raised on a matrix that carried no defect of its own.

# Algorithm

 1. Orient `X` and `iv` to `observations × assets` with [`dims_oriented`](@ref), which validates `dims` and transposes both when `dims` is `2`.
 2. Check that `X` and `iv` have the same size, so row `t` of `iv` is the implied volatility of observation `t` of `X`.
 3. Call `Statistics.cor(ce.ce, X; dims = 1, mean = mean, iv = iv, kwargs...)`, giving `rho`, the base correlation matrix. The oriented `iv` is forwarded so that a base estimator that reads its own implied volatility series, such as a nested [`ImpliedVolatility`](@ref), receives it.
 4. Call [`predict_realised_vols`](@ref) with `ce.alg` and `iv / sqrt(ce.af)`, and discard the result. The call runs for its raises alone, and `iv` is divided by `sqrt(ce.af)` for it exactly as it is in `cov`.
 5. Normalise `rho` in place with `StatsBase.cov2cor!`, which divides the entry in row `i` and column `j` by the square roots of the diagonal entries `i` and `j`. The call also mirrors the lower triangle into the upper one, clamps every off-diagonal entry into `[-1, 1]`, and sets the diagonal to exactly one. The exact diagonal is what step 6 needs: [`matrix_processing!`](@ref) reads the value of the diagonal to decide whether it holds a correlation matrix or a covariance matrix.
 6. Post-process `rho` in place with [`matrix_processing!`](@ref) and `ce.mp`.

# Arguments

  - `ce`: Implied volatility covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean (passed to the base estimator).
  - `iv`: Implied volatility matrix, annualised, oriented as `X` and of the same size.
  - `ivpa`: Optional implied volatility premium adjustment factor (required for [`ImpliedVolatilityPremium`](@ref)).
  - `kwargs...`: Additional keyword arguments passed to the base estimator.

# Validation

  - `dims in (1, 2)`, by [`dims_oriented`](@ref).
  - `size(X) == size(iv)`, one implied volatility per return.
  - Whatever `ce.alg` refuses, by [`predict_realised_vols`](@ref). [`ImpliedVolatilityPremium`](@ref) needs an `ivpa` that is not `nothing`, and whose every entry is finite and strictly positive.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref)
  - [`cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing, iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
    X, iv = dims_oriented(dims, X, iv)
    @argcheck(size(X) == size(iv), DimensionMismatch)
    rho = Statistics.cor(ce.ce, X; dims = 1, mean = mean, iv = iv, kwargs...)
    # The prediction is discarded. It runs so that `cor` refuses what `cov` refuses.
    predict_realised_vols(ce.alg, iv / sqrt(ce.af), X, ivpa)
    StatsBase.cov2cor!(rho)
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end
export ImpliedVolatility, ImpliedVolatilityPremium, ImpliedVolatilityRegression
