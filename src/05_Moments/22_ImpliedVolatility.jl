"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all implied volatility algorithms.

A subtype predicts the realised volatility of each asset for the period that follows the sample.

# Interfaces

To add an implied volatility algorithm, subtype `ImpliedVolatilityAlgorithm` and add this method:

## Required method name

  - `predict_realised_vols(alg::ImpliedVolatilityAlgorithm, iv::MatNum, X::MatNum, ivpa::Any)`: Returns the predicted realised volatility of each asset for the period that follows the sample.

### Arguments

The implied volatilities are the second positional argument, and the returns are the third. The two matrices have the same size, so a call that swaps them raises no error and returns a wrong answer.

  - `alg`: The instance of the concrete subtype.
  - `iv`: Implied volatility matrix `observations × assets`. The caller divides it by ``\\sqrt{\\mathrm{ppy}}`` before the call.
  - `X`: Asset returns matrix `observations × assets`.
  - `ivpa`: Implied volatility premium adjustment factor. It is `nothing` when the caller gives no factor, so an algorithm that needs a factor must raise an error on `nothing`.

### Returns

  - `rv_p::VecNum`: The predicted realised volatility of each asset, in the units of `X`.

### Examples

The example below defines a minimal algorithm that returns the last row of `iv`.

```jldoctest
julia> struct MyImpliedVolatilityAlgorithm <: PortfolioOptimisers.ImpliedVolatilityAlgorithm end

julia> function PortfolioOptimisers.predict_realised_vols(::MyImpliedVolatilityAlgorithm,
                                                          iv::PortfolioOptimisers.MatNum, ::Any,
                                                          ::Any)
           return vec(iv[end, :])
       end

julia> cov(ImpliedVolatility(; alg = MyImpliedVolatilityAlgorithm(), ppy = 1),
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
"""
abstract type ImpliedVolatilityAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Predicts realised volatility by a log-linear regression on implied volatility and on the realised volatility of the window before.

`ImpliedVolatilityRegression` splits the sample into windows of `ws` rows that do not overlap. For each asset, it regresses the realised volatility of a window on the implied volatility at the end of the window before, and on the realised volatility of the window before. It then predicts the realised volatility of the next window from the last window. The regression is Equation 5 of [christensenprabhala1998](@cite), and both series enter it as natural logarithms. This type selects a method of [`predict_realised_vols`](@ref), and the docstring of that method states the model, the steps and the number of windows that the fit needs.

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

  - `ws > 1`. A window of one row has no dispersion. Its corrected standard deviation is `NaN`, and its uncorrected one is zero, which has no logarithm.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated through [`factory`](@ref).

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

  - $(ref_dict[:christensenprabhala1998]) Section 3.1, Equation 5.
"""
@propagatable @concrete struct ImpliedVolatilityRegression <: ImpliedVolatilityAlgorithm
    """
    $(field_dict[:ve])
    """
    @fprop ve
    """
    Number of rows in each window. The number of windows is `div(size(X, 1), ws)`, and the regression needs more than four windows.
    """
    ws
    # crit
    """
    $(field_dict[:retgt])
    """
    re
    function ImpliedVolatilityRegression(ve::AbstractVarianceEstimator, ws::Number,
                                         re::AbstractRegressionTarget)
        @argcheck(1 < ws,
                  DomainError(ws,
                              "`ImpliedVolatilityRegression.ws` is $ws, and a window of one row has no standard deviation whose logarithm the regression can take. State an integer greater than `1`."))
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

Divides the last implied volatility of each asset by a volatility risk premium adjustment factor.

The method follows [egbersswinkels2015](@cite). The library has no copy of that work, so no check compares the division with it.

The adjustment factor is not a field of this type. The caller gives it as the `ivpa` keyword of the `cov` and `cor` methods of [`ImpliedVolatility`](@ref), as a scalar or as one value per asset. The factor is mandatory, and `ivpa = nothing` raises an `ArgumentError`. Each entry must be finite and strictly positive, or the method raises a `DomainError`. A factor that is not positive makes a negative volatility, and `StatsBase.cor2cov!` hides its sign. This type selects a method of [`predict_realised_vols`](@ref), and the docstring of that method states the closed form and the checks.

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

  - $(ref_dict[:egbersswinkels2015]) Pages 14 to 26. $(ref_dict[:no_original_source])
"""
struct ImpliedVolatilityPremium <: ImpliedVolatilityAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Covariance estimator that scales a correlation matrix by the realised volatilities that it predicts from implied volatilities.

`ImpliedVolatility` estimates the correlation matrix of the returns with a base estimator. It predicts the realised volatility of each asset from the implied volatilities, and scales the correlation matrix by those volatilities. Two algorithms make the prediction. [`ImpliedVolatilityRegression`](@ref) fits a regression of realised volatility on implied volatility. [`ImpliedVolatilityPremium`](@ref) divides the implied volatility by a factor that the caller gives.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ImpliedVolatility(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        alg::ImpliedVolatilityAlgorithm = ImpliedVolatilityRegression(),
        ppy::Number,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> ImpliedVolatility

Keywords correspond to the struct's fields. `ppy` has no default. The implied volatility is annual, and `ppy` converts it to the period of the returns. No single value is correct for all data frequencies, so the caller must give it, for example `ppy = 252` for daily returns.

## Validation

  - `ppy > 0`.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `alg`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ce`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ImpliedVolatility(; ppy = 252)
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
  ppy ┴ Int64: 252
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`ImpliedVolatilityAlgorithm`](@ref)
  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatilityPremium`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
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
    Implied volatility algorithm that predicts the realised volatility of each asset.
    """
    @fprop alg
    """
    Periods per year of the returns. It converts the annual implied volatility to the frequency of the returns. The `cov` and `cor` methods divide the implied volatilities by `sqrt(ppy)` before the algorithm reads them.
    """
    ppy
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function ImpliedVolatility(ce::StatsBase.CovarianceEstimator,
                               mp::AbstractMatrixProcessingEstimator,
                               alg::ImpliedVolatilityAlgorithm, ppy::Number,
                               cache::Option{<:AbstractPartialFitState})
        assert_gt0(ppy, :ppy)
        return new{typeof(ce), typeof(mp), typeof(alg), typeof(ppy), typeof(cache)}(ce, mp,
                                                                                    alg,
                                                                                    ppy,
                                                                                    cache)
    end
end
function ImpliedVolatility(; ce::StatsBase.CovarianceEstimator = Covariance(),
                           mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                           alg::ImpliedVolatilityAlgorithm = ImpliedVolatilityRegression(),
                           ppy::Number,
                           cache::Option{<:AbstractPartialFitState} = nothing)::ImpliedVolatility
    return ImpliedVolatility(ce, mp, alg, ppy, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`ImpliedVolatility`](@ref) except `cache`.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:ImpliedVolatility, true)` to render it.

# Arguments

  - `::ImpliedVolatility`: The implied volatility covariance estimator. The method reads only its type.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ce, :mp, :alg, :ppy)`.

# Related

  - [`ImpliedVolatility`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::ImpliedVolatility) = (:ce, :mp, :alg, :ppy)
"""
    realised_vol(ce::AbstractVarianceEstimator, X::MatNum, ws::Integer,
                 chunk::Option{<:Integer} = nothing, T::Option{<:Integer} = nothing,
                 N::Option{<:Integer} = nothing)

Returns the realised volatility of each asset over consecutive blocks of `ws` rows.

The function splits the last `chunk * ws` rows of `X` into `chunk` blocks of `ws` rows that do not overlap. The estimator `ce` gives the standard deviation of each asset in each block. The result is a `chunk × N` matrix.

[`@propagatable`](@ref) makes an [`obs_weights_view`](@ref) method for each estimator that it declares, from the [`@wprop`](@ref) tag of the weights field. So both variance estimators of the library slice their weights with no method written by hand. An estimator can keep its weights in a field with no `@wprop` tag, and have no method of its own. Such an estimator receives a block of `ws` rows and a vector of `T` weights, and the call raises an error.

# Mathematical definition

Write ``C`` for `chunk` and ``w_s`` for `ws`. The function counts the blocks back from the last observation, so the first

```math
\\begin{align}
o &= T - C w_s
\\end{align}
```

rows are in no block. Block ``c`` holds the rows

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
  - ``o``: Number of rows at the start of the sample that no block covers.
  - ``\\mathcal{R}_c``: Rows of block ``c``.
  - ``\\mathrm{rv}_{c,\\,i}``: Realised volatility of asset ``i`` over block ``c``.

The last row of block ``c`` is ``o + c w_s``. [`implied_vol`](@ref) samples the same row for the same window. When ``w_s`` does not divide ``T``, the function drops the oldest rows and keeps the newest.

# Algorithm

 1. When `chunk`, `T` or `N` is `nothing`, read `T` and `N` from `size(X)` and set `chunk` to `div(T, ws)`.
 2. Set `offset` to `T - chunk * ws`, the number of rows at the start of `X` that no block covers.
 3. For each block `c` in `1:chunk`, set `rows` to the `ws` row indices of that block.
 4. Slice `ce` to `rows` with [`obs_weights_view`](@ref). The result is the estimator for this block. [`obs_weights_view`](@ref) indexes each weights field on its own. So an estimator with a weighted mean and an unweighted dispersion keeps that shape, and an unweighted `ce` stays unchanged.
 5. Call `Statistics.std` with that estimator on the rows of the block of `X`, with `dims = 1`. The result is one row of standard deviations. The function calls the estimator once for each block, so the estimator reads only a matrix.
 6. Stack the `chunk` rows with `vcat` to make `rv`.

# Arguments

  - `ce`: Variance estimator that gives the standard deviations in each block. Its observation weights describe the whole sample, and step 4 slices them to each block.
  - `X`: Asset returns matrix `observations × assets`.
  - `ws`: Window size, the number of rows in each block.
  - `chunk`: Number of blocks, or `nothing`. Step 1 states the rule when one of `chunk`, `T` and `N` is `nothing`.
  - `T`: Number of observations, or `nothing`.
  - `N`: Number of assets, or `nothing`.

# Returns

  - `rv::Matrix{<:Number}`: Realised volatility matrix `chunk × assets`.

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

Returns the row of `X` at the end of each window of `ws` rows.

The function selects the last row of each window, and returns a `chunk × N` view of `X`.

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
  - ``o``: Number of rows at the start of the sample that no window covers.

``\\mathcal{S}_c`` is the last row of block ``c`` of [`realised_vol`](@ref), so the two functions read the same windows. The last sampled row is ``T`` for each ``w_s``. So when ``w_s`` does not divide ``T``, the function drops the oldest rows and keeps the newest.

# Algorithm

 1. When `chunk`, `T` or `N` is `nothing`, read `T` and `N` from `size(X)` and set `chunk` to `div(T, ws)`.
 2. Return the view of `X` over the rows `(T - (chunk - 1) * ws):ws:T`. The view shares its memory with `X`.

# Arguments

  - `X`: Implied volatility matrix `observations × assets`.
  - `ws`: Window size, the number of rows in each window.
  - `chunk`: Number of windows, or `nothing`. Step 1 states the rule when one of `chunk`, `T` and `N` is `nothing`.
  - `T`: Number of observations, or `nothing`.
  - `N`: Number of assets, or `nothing`.

# Returns

  - `iv::SubArray`: The implied volatility at the end of each window, `chunk × assets`.

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

Raises an `ArgumentError`, because the premium algorithm needs an implied volatility premium adjustment factor.

The adjustment factor is not a field of [`ImpliedVolatilityPremium`](@ref). A caller that selects that algorithm and gives no `ivpa` reaches this method.

# Arguments

The implied volatilities are the second positional argument, and the returns are the third.

  - `::ImpliedVolatilityPremium`: Implied volatility premium algorithm.
  - `iv`: Implied volatility matrix. The method does not read it.
  - `::Any`: Asset returns matrix. The method does not read it.
  - `ivpa::Nothing`: `nothing`, because the caller gave no implied volatility premium adjustment factor.

# Validation

  - `ivpa` is not `nothing`. This method is the branch that fails, and it raises an `ArgumentError`.

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

Predicts the realised volatility of each asset as its last implied volatility divided by the premium adjustment factor.

The method reads the last row of `iv` itself, not the last row of a window. So it needs no window size and no returns. The division follows [egbersswinkels2015](@cite). The library has no copy of that work, so no check compares the formula with it.

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

The implied volatilities are the second positional argument, and the returns are the third.

  - `::ImpliedVolatilityPremium`: Implied volatility premium algorithm.
  - `iv`: Implied volatility matrix `observations × assets`. The method reads its last row.
  - `::Any`: Asset returns matrix. The method does not read it.
  - `ivpa`: Implied volatility premium adjustment factor, a scalar or a vector.

# Validation

  - Each entry of `ivpa` is finite and strictly positive. A factor that is not positive makes a negative volatility, and `StatsBase.cor2cov!` hides its sign. That function squares the factor on the diagonal. So a negative scalar gives the same matrix as its absolute value, and a negative entry of a vector flips the sign of each covariance of that asset only. Both results stay positive definite, so [`matrix_processing!`](@ref) finds no defect, and no later step sees it.
  - A vector `ivpa` has one entry for each asset. A vector of a different length raises a `DimensionMismatch` from the broadcast.

# Returns

  - `rv::AbstractArray`: The predicted realised volatilities, which are the last row of `iv` divided by `ivpa`.

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

  - $(ref_dict[:egbersswinkels2015]) Pages 14 to 26. $(ref_dict[:no_original_source])
"""
function predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any,
                               ivpa::Num_VecNum)
    @argcheck(all(x -> isfinite(x) && x > zero(x), ivpa),
              DomainError(ivpa,
                          "every entry of `ivpa` must be finite and strictly positive. A non-positive factor turns the predicted realised volatility negative, and `StatsBase.cor2cov!` hides the sign, so no later step sees the defect"))
    return view(iv, size(iv, 1), :) ⊘ ivpa
end
"""
    predict_realised_vols(alg::ImpliedVolatilityRegression, iv::MatNum, X::MatNum, ::Any)

Predicts the realised volatility of each asset from a regression on its implied and realised volatilities.

For each asset, the function regresses the realised volatility of a window on the implied volatility and the realised volatility of the window before. It then predicts the next window from the last window. The windows are the blocks of [`realised_vol`](@ref) and the rows of [`implied_vol`](@ref), so the two series read the same rows of the sample.

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

The fit has ``C - 1`` rows and three coefficients, and the method needs ``C > 4``. At ``C = 3`` the fit has fewer rows than coefficients, so the coefficients have no unique value. At ``C = 4`` the fit passes through each row and has no residual, so the data cannot estimate ``\\varepsilon_c``. The fit has more rows than coefficients only when ``C > 4``. The oracle accepts ``C = 3``. The library refuses ``C = 3`` and ``C = 4``.

This is Equation 5 of [christensenprabhala1998](@cite). It regresses the log realised volatility of period ``t`` on the log implied volatility at the start of ``t`` and on the log realised volatility of period ``t - 1``. Section 2.2 of that work defines both as natural logarithms. It defines the realised volatility as the sample standard deviation of the returns of the period. Section 3.3 of that work forecasts one period ahead from the fitted equation, and it states no back-transformation. The exponential of the fitted logarithm is the choice of this library.

# Algorithm

 1. Read `T` and `N` from `size(X)`, and set `chunk` to `div(T, alg.ws)`.
 2. Check that `chunk` exceeds four.
 3. Call [`realised_vol`](@ref) with `alg.ve`. The result is `rv`, the realised volatility of each window.
 4. Call [`implied_vol`](@ref) on `iv` with the same `chunk`, `T` and `N`. The result replaces `iv`, and holds the implied volatility at the last row of each window. The window count comes from `X`, so the function reads `iv` over the rows of the returns sample.
 5. Check that `rv` and `iv` have the same size, and that each of their entries is positive. Set `T2` to their number of rows, which is `chunk`.
 6. Replace `rv` and `iv` by their natural logarithms.
 7. Make `ovec`, the intercept column of ones, of length `T2 - 1`.
 8. For each asset `i`, make the design matrix `X_t` from `ovec` and the first `T2 - 1` rows of `iv` and `rv`. Make the response `y_t` from the rows `2:T2` of `rv`, and the prediction row `X_p` from the row `T2`.
 9. Fit `alg.re` on `X_t` and `y_t`. Predict from `X_p` with the fit, and set `rv_p[i]` to the exponential of the prediction.
10. Return `rv_p`.

# Arguments

The implied volatilities are the second positional argument, and the returns are the third. The two matrices have the same size, so a call that swaps them raises no error and returns a wrong answer.

  - `alg`: Implied volatility regression algorithm. It gives the variance estimator, the window size and the regression target.
  - `iv`: Implied volatility matrix `observations × assets`.
  - `X`: Asset returns matrix `observations × assets`, from which the function computes the realised volatility. It also sets the number of windows, so `iv` must have as many rows as `X`.
  - `::Any`: The place of `ivpa`. The method does not read it.

# Validation

  - `chunk > 4`. The fit has `chunk - 1` rows for three coefficients, so it has a residual only when `chunk > 4`.
  - `size(rv) == size(iv)`, one realised volatility for each implied volatility.
  - Each implied volatility on the last row of a window is positive. Otherwise the function throws a `DomainError`, because the fit takes its logarithm. An entry on a row that ends no window is not read.
  - Each window has a positive realised volatility. Otherwise the function throws a `DomainError`. Constant returns over the `ws` rows of a window give a realised volatility of zero.

# Returns

  - `rv_p::Vector{<:Number}`: The predicted realised volatility of each asset for the next window.

# Related

  - [`ImpliedVolatilityRegression`](@ref)
  - [`ImpliedVolatility`](@ref)
  - [`realised_vol`](@ref)
  - [`implied_vol`](@ref)

# References

  - $(ref_dict[:christensenprabhala1998]) Section 2.2, Equation 2, and Section 3.1, Equation 5.
"""
function predict_realised_vols(alg::ImpliedVolatilityRegression, iv::MatNum, X::MatNum,
                               ::Any)
    T, N = size(X)
    chunk = div(T, alg.ws)
    @argcheck(4 < chunk,
              DomainError(chunk,
                          "`X` holds $T observations, and `ImpliedVolatilityRegression.ws` is $(alg.ws), which gives $chunk windows. The fit has one row fewer than the windows and three coefficients, so it needs more than four windows to have a residual. State a shorter window or supply more observations."))
    rv = realised_vol(alg.ve, X, alg.ws, chunk, T, N)
    iv = implied_vol(iv, alg.ws, chunk, T, N)
    @argcheck(size(rv) == size(iv), DimensionMismatch)
    # The fit takes the logarithm of both series, so a volatility of zero has no value in it.
    k = findfirst(x -> !(x > zero(x)), iv)
    @argcheck(isnothing(k),
              DomainError(iv[k],
                          "`iv` holds $(iv[k]) for asset $(k[2]) on the last row of window $(k[1]) of $chunk. `ImpliedVolatilityRegression` takes the logarithm of the implied volatility on the last row of each window, so that value must be positive. State a positive implied volatility on that row, or use `ImpliedVolatilityPremium`."))
    k = findfirst(x -> !(x > zero(x)), rv)
    @argcheck(isnothing(k),
              DomainError(rv[k],
                          "Window $(k[1]) of $chunk has no dispersion for asset $(k[2]): its realised volatility is $(rv[k]). `ImpliedVolatilityRegression` takes the logarithm of the realised volatility of each window, so that value must be positive. State another `ImpliedVolatilityRegression.ws`, or remove the asset."))
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
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a fit of an [`ImpliedVolatility`](@ref) estimator that receives no implied volatilities.

The estimator scales a correlation by the realised volatilities that it predicts from `iv`, so it cannot compute a result without them. A [`prior`](@ref) over a [`ReturnsResult`](@ref) gives `rd.iv` to each estimator that it holds. `rd.iv` is `nothing` when the returns have no implied volatilities. So the `iv` keyword of each `cov` and `cor` method accepts `nothing`. This function then refuses `nothing` with an error that names the input and the remedy.

# Arguments

  - `iv`: The implied volatilities, `observations × assets`, or `nothing`.

# Validation

  - `iv` is not `nothing`. Otherwise the function throws an `IsNothingError`.

# Returns

  - `nothing`.

# Related

  - [`ImpliedVolatility`](@ref)
  - [`ReturnsResult`](@ref)
"""
function assert_implied_volatility(iv::Option{<:MatNum})
    @argcheck(!isnothing(iv),
              IsNothingError("`ImpliedVolatility` predicts the volatility of each asset from the implied volatilities `iv`, `observations × assets`, and none were given. Pass `iv` to `cov` or `cor`, or fit the prior on a `ReturnsResult` that carries `iv`."))
    return nothing
end
"""
    Statistics.cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                   iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)

Returns the covariance matrix of `X`, scaled by the realised volatilities that `ce.alg` predicts from `iv`.

This method computes the correlation matrix of `X` with the base estimator `ce.ce`. It predicts the realised volatilities from `iv` with the algorithm `ce.alg`, and converts the correlation matrix to a covariance matrix with those volatilities. The matrix processing estimator `ce.mp` then processes the result.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\mathrm{diag}(\\hat{\\boldsymbol{\\sigma}}^{\\mathrm{rv}}) \\hat{\\boldsymbol{\\rho}} \\,\\mathrm{diag}(\\hat{\\boldsymbol{\\sigma}}^{\\mathrm{rv}})\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``\\hat{\\boldsymbol{\\rho}} = \\operatorname{cor}(\\mathbf{X})``: Correlation matrix of the asset returns, from `ce.ce`.
  - ``\\hat{\\boldsymbol{\\sigma}}^{\\mathrm{rv}}``: Predicted realised volatilities, from ``\\mathbf{iv} / \\sqrt{\\mathrm{ppy}}``. ``\\mathrm{ppy}`` is `ce.ppy`, the number of periods of `X` in one year.

So each diagonal entry of ``\\hat{\\mathbf{\\Sigma}}`` is the square of the predicted realised volatility of that asset, not one.

# Algorithm

 1. Refuse a call with no `iv` through [`assert_implied_volatility`](@ref). Orient `X` and `iv` to `observations × assets` with [`dims_oriented`](@ref). It checks `dims`, and it transposes both matrices when `dims` is `2`.
 2. Check that `X` and `iv` have the same size, so that row `t` of `iv` is the implied volatility of observation `t` of `X`. Refuse a gap in `X` with [`assert_finite_sample`](@ref). Refuse a gap or a negative entry in `iv` with [`assert_nonempty_nonneg_finite_val`](@ref).
 3. Call `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = 1, mean = mean, iv = iv, kwargs...)`. The result is `sigma`, the base correlation matrix. The call forwards the oriented `iv`, so a base estimator that reads its own implied volatility series receives it. A nested [`ImpliedVolatility`](@ref) is such an estimator. Each other estimator of the library takes `iv` into its own `kwargs...` and does not read it.
 4. Divide `iv` by `sqrt(ce.ppy)`. This converts the annual implied volatility to the frequency of `X`.
 5. Call [`predict_realised_vols`](@ref) with `ce.alg`. The result replaces `iv`, and holds one predicted realised volatility for each asset. The implied volatilities are the second argument, and the returns are the third.
 6. Scale `sigma` in place with `StatsBase.cor2cov!`, which applies the closed form above.
 7. Process `sigma` in place with [`matrix_processing!`](@ref) and `ce.mp`.

# Arguments

  - `ce`: Implied volatility covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: Optional precomputed mean. The method gives it to the base estimator.
  - `iv`: Annual implied volatility matrix, with the orientation and the size of `X`.
  - `ivpa`: Optional implied volatility premium adjustment factor. [`ImpliedVolatilityPremium`](@ref) needs it.
  - `kwargs...`: More keyword arguments for the base estimator and for [`matrix_processing!`](@ref).

# Validation

  - `iv` is not `nothing`. Otherwise [`assert_implied_volatility`](@ref) throws an `IsNothingError`.
  - `dims in (1, 2)`. [`dims_oriented`](@ref) checks it.
  - `size(X) == size(iv)`, one implied volatility for each return.
  - Each entry of `X` is finite. Otherwise the method throws an `IsNonFiniteError`. The panel method removes an asset with a gap from the Coverage Universe, and this check keeps the plain call in agreement with it. Without the check, a plain call reads past the gap and gives a value for an asset that the panel method does not price.
  - Each entry of `iv` is finite and not negative. Otherwise the method throws a `DomainError`. The method refuses a gap in `iv` for the same reason as a gap in `X`. [`ImpliedVolatilityPremium`](@ref) turns a negative entry into a negative volatility, and `StatsBase.cor2cov!` hides its sign. [`ImpliedVolatilityRegression`](@ref) takes the logarithm of the entry on the last row of each window, and refuses a zero there, through [`predict_realised_vols`](@ref).
  - The checks of `ce.alg`, through [`predict_realised_vols`](@ref). [`ImpliedVolatilityPremium`](@ref) needs an `ivpa` that is not `nothing`, with each entry finite and strictly positive.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref)
  - [`cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::Option{<:MatNum} = nothing,
                        ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
    assert_implied_volatility(iv)
    X, iv = dims_oriented(dims, X, iv)
    @argcheck(size(X) == size(iv), DimensionMismatch)
    assert_finite_sample(X)
    assert_nonempty_nonneg_finite_val(iv, :iv)
    sigma = Statistics.cor(library_covariance_estimator(ce.ce), X; dims = 1, mean = mean,
                           iv = iv, kwargs...)
    iv = iv / sqrt(ce.ppy)
    iv = predict_realised_vols(ce.alg, iv, X, ivpa)
    StatsBase.cor2cov!(sigma, iv)
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end
"""
    Statistics.cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                   iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)

Returns the base correlation matrix of `X`, after the same checks as the covariance method.

This method computes the correlation matrix of `X` with the base estimator `ce.ce`. It normalises the matrix, and processes it with the matrix processing estimator `ce.mp`.

A correlation has no scale, so the predicted realised volatilities cannot change it. The result is the base correlation of `X`. The volatility model runs in `cor` too, because `cor` must refuse each configuration that `cov` refuses. So a `ce.alg` that cannot answer the call raises an error here, as it does in `cov`. The method discards the prediction. It does not multiply `rho` by the prediction and then divide the prediction out again.

That round trip is the identity only in exact arithmetic. In floating point, the round-off of one multiplication and one division moves an off-diagonal entry. A predicted volatility of zero also makes the division compute zero divided by zero. So one asset with a last implied volatility of zero turns a full row and column of the correlation into `NaN`. Then `matrix_processing!` raises an error on a matrix that has no defect of its own.

# Algorithm

 1. Refuse a call with no `iv` through [`assert_implied_volatility`](@ref). Orient `X` and `iv` to `observations × assets` with [`dims_oriented`](@ref). It checks `dims`, and it transposes both matrices when `dims` is `2`.
 2. Check that `X` and `iv` have the same size, so that row `t` of `iv` is the implied volatility of observation `t` of `X`. Refuse a gap in `X` with [`assert_finite_sample`](@ref). Refuse a gap or a negative entry in `iv` with [`assert_nonempty_nonneg_finite_val`](@ref).
 3. Call `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = 1, mean = mean, iv = iv, kwargs...)`. The result is `rho`, the base correlation matrix. The call forwards the oriented `iv`, so a base estimator that reads its own implied volatility series receives it. A nested [`ImpliedVolatility`](@ref) is such an estimator.
 4. Call [`predict_realised_vols`](@ref) with `ce.alg` and `iv / sqrt(ce.ppy)`, and discard the result. The call runs only for its errors. It divides `iv` by `sqrt(ce.ppy)`, as `cov` does.
 5. Normalise `rho` in place with `StatsBase.cov2cor!`. It divides the entry in row `i` and column `j` by the square roots of the diagonal entries `i` and `j`. It also copies the lower triangle into the upper triangle, clamps each off-diagonal entry into `[-1, 1]`, and sets each diagonal entry to one. Step 6 needs that diagonal of ones, because [`matrix_processing!`](@ref) reads the diagonal to find whether the matrix is a correlation matrix or a covariance matrix.
 6. Process `rho` in place with [`matrix_processing!`](@ref) and `ce.mp`.

# Arguments

  - `ce`: Implied volatility covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: Optional precomputed mean. The method gives it to the base estimator.
  - `iv`: Annual implied volatility matrix, with the orientation and the size of `X`.
  - `ivpa`: Optional implied volatility premium adjustment factor. [`ImpliedVolatilityPremium`](@ref) needs it.
  - `kwargs...`: More keyword arguments for the base estimator and for [`matrix_processing!`](@ref).

# Validation

  - `iv` is not `nothing`. Otherwise [`assert_implied_volatility`](@ref) throws an `IsNothingError`.
  - `dims in (1, 2)`. [`dims_oriented`](@ref) checks it.
  - `size(X) == size(iv)`, one implied volatility for each return.
  - Each entry of `X` is finite. Otherwise the method throws an `IsNonFiniteError`. The panel method removes an asset with a gap from the Coverage Universe, and this check keeps the plain call in agreement with it. Without the check, a plain call reads past the gap and gives a value for an asset that the panel method does not price.
  - Each entry of `iv` is finite and not negative. Otherwise the method throws a `DomainError`. The method refuses a gap in `iv` for the same reason as a gap in `X`. [`ImpliedVolatilityPremium`](@ref) turns a negative entry into a negative volatility, and `StatsBase.cor2cov!` hides its sign. [`ImpliedVolatilityRegression`](@ref) takes the logarithm of the entry on the last row of each window, and refuses a zero there, through [`predict_realised_vols`](@ref).
  - The checks of `ce.alg`, through [`predict_realised_vols`](@ref). [`ImpliedVolatilityPremium`](@ref) needs an `ivpa` that is not `nothing`, with each entry finite and strictly positive.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`ImpliedVolatility`](@ref)
  - [`predict_realised_vols`](@ref)
  - [`cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::Option{<:MatNum} = nothing,
                        ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
    assert_implied_volatility(iv)
    X, iv = dims_oriented(dims, X, iv)
    @argcheck(size(X) == size(iv), DimensionMismatch)
    assert_finite_sample(X)
    assert_nonempty_nonneg_finite_val(iv, :iv)
    rho = Statistics.cor(library_covariance_estimator(ce.ce), X; dims = 1, mean = mean,
                         iv = iv, kwargs...)
    # The prediction is discarded. It runs so that `cor` refuses what `cov` refuses.
    predict_realised_vols(ce.alg, iv / sqrt(ce.ppy), X, ivpa)
    StatsBase.cov2cor!(rho)
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end
"""
    coverage_reduced_ivpa(ivpa::Option{<:Number}, cmsk) -> Option{<:Number}
    coverage_reduced_ivpa(ivpa::VecNum, cmsk::BitVector) -> VecNum

Reduces a per-asset implied volatility premium to the Coverage Universe.

A scalar premium applies to each asset, so the function returns it unchanged. A vector premium has one entry for each asset, so the function slices it as the caller slices `X`.

# Arguments

  - `ivpa`: The implied volatility premium adjustment factor, as a scalar, a vector, or `nothing`.
  - `cmsk`: The Coverage Universe as a mask of the assets, or `nothing`.

# Returns

  - `ivpa`: The premium on the Coverage Universe.

# Related

  - [`ImpliedVolatility`](@ref)
  - [`coverage_reduction`](@ref)
"""
function coverage_reduced_ivpa(ivpa, ::Nothing)
    return ivpa
end
function coverage_reduced_ivpa(ivpa::Option{<:Number}, ::BitVector)
    return ivpa
end
function coverage_reduced_ivpa(ivpa::VecNum, cmsk::BitVector)
    return ivpa[cmsk]
end
"""
    reads_implied_volatility(est::Union{<:AbstractEstimator, <:AbstractCovarianceEstimator}) -> Bool
    reads_implied_volatility(est::ImpliedVolatility) -> Bool
    reads_implied_volatility(v::AbstractVector) -> Bool
    reads_implied_volatility(p::Pair) -> Bool
    reads_implied_volatility(x) -> Bool

Returns `true` when an estimator tree holds an [`ImpliedVolatility`](@ref), which reads the implied volatility surface `iv`.

The reduce-and-expand root of a moment verb, [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref), reads the result. A gap in the surface removes an asset from the Coverage Universe only for a fit that reads the surface. A wrapper that holds an `ImpliedVolatility` and has no panel method of its own reaches the root. So the root must know that the tree reads the surface.

# Algorithm

Each step is one method, and Julia selects the method from the type of the argument.

 1. An `ImpliedVolatility` returns `true`.
 2. Each other estimator returns `true` when one of its fields returns `true`. A covariance estimator subtypes `StatsBase.CovarianceEstimator` and not `AbstractEstimator`, so the method takes both supertypes.
 3. A vector of estimators returns `true` when one of its entries returns `true`. A vector of numbers returns `false`. A `Pair` returns the result for its value.
 4. Each other value returns `false`.

# Arguments

  - `est`: An estimator, or one field of an estimator.

# Returns

  - `flag::Bool`: `true` when the tree reads the implied volatility surface.

# Related

  - [`ImpliedVolatility`](@ref)
  - [`coverage_keyword_reduction`](@ref)
"""
function reads_implied_volatility(est::Union{<:AbstractEstimator,
                                             <:AbstractCovarianceEstimator})
    return any(f -> reads_implied_volatility(getfield(est, f)), fieldnames(typeof(est)))
end
function reads_implied_volatility(::ImpliedVolatility)
    return true
end
function reads_implied_volatility(v::AbstractVector)
    return eltype(v) <: Number ? false : any(reads_implied_volatility, v)
end
function reads_implied_volatility(p::Pair)
    return reads_implied_volatility(last(p))
end
function reads_implied_volatility(::Any)
    return false
end
"""
    Statistics.cov(ce::ImpliedVolatility, X::MatNum, pnl::Option{<:AssetPanel};
                   dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing,
                   ivpa::Option{<:Num_VecNum} = nothing,
                   active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                   kwargs...) -> MatNum
    Statistics.cor(ce::ImpliedVolatility, X::MatNum, pnl::Option{<:AssetPanel};
                   dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing,
                   ivpa::Option{<:Num_VecNum} = nothing,
                   active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                   kwargs...) -> MatNum

Fits an implied volatility estimate on the Coverage Universe, and expands it to the full asset universe.

[`ImpliedVolatility`](@ref) has its own method of the reduce-and-expand root of its verb. The reason is that it reads two per-asset inputs that the root does not know. One is the implied volatility surface `iv`, `observations × assets`. The other is the premium `ivpa`, which has one number for each asset when it is a vector. Both take the slice that `X` takes, so the three inputs describe the same universe.

The surface can also make that universe smaller. An implied volatility holds `NaN` where it has no value, as a return does. So the Coverage Universe of this fit is the one that [`coverage_mask`](@ref) finds from `X` and `iv`. The fit excludes an asset whose returns are complete but whose implied volatilities are not. [`expand_moment`](@ref) then writes `NaN` into the row and the column of that asset.

# Algorithm

 1. Refuse a call with no `iv` through [`assert_implied_volatility`](@ref). Check `dims`, and orient `X` and `iv` to `observations × assets`.
 2. Check that `X` and `iv` have the same size.
 3. Find the Coverage Universe of `X` and `iv` with [`coverage_mask`](@ref). When the caller gives `active_mask`, [`coverage_mask_source`](@ref) puts it in the place of the active mask of `pnl`.
 4. Slice `X`, `iv` and `ivpa` to the Coverage Universe.
 5. Call the two-argument method on the slices.
 6. Expand the matrix with [`expand_moment`](@ref).

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `iv`: Implied volatility surface `observations × assets`.
  - `ivpa`: Implied volatility premium adjustment factor, or `nothing`.
  - $(arg_dict[:cvg_active_mask])
  - `kwargs...`: More keyword arguments for the two-argument method.

# Validation

  - `iv` is not `nothing`. Otherwise [`assert_implied_volatility`](@ref) throws an `IsNothingError`.
  - $(val_dict[:dims])
  - `size(X) == size(iv)`.
  - The Coverage Universe holds at least one asset.

# Returns

  - `sigma::MatNum`: The covariance matrix or the correlation matrix on the full asset universe.

# Related

  - [`ImpliedVolatility`](@ref)
  - [`coverage_mask`](@ref)
  - [`expand_moment`](@ref)
"""
function Statistics.cov(ce::ImpliedVolatility, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing,
                        ivpa::Option{<:Num_VecNum} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_implied_volatility(iv)
    src = coverage_mask_source(pnl, active_mask, dims)
    X, iv = dims_oriented(dims, X, iv)
    @argcheck(size(X) == size(iv), DimensionMismatch)
    cmsk = coverage_mask(X, iv, src; dims = 1)
    Xc, ivc = coverage_reduced_pair(X, iv, cmsk)
    sigma = Statistics.cov(ce, Xc; dims = 1, mean = mean, iv = ivc,
                           ivpa = coverage_reduced_ivpa(ivpa, cmsk), kwargs...)
    return expand_moment(sigma, cmsk)
end
function Statistics.cor(ce::ImpliedVolatility, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing,
                        ivpa::Option{<:Num_VecNum} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_implied_volatility(iv)
    src = coverage_mask_source(pnl, active_mask, dims)
    X, iv = dims_oriented(dims, X, iv)
    @argcheck(size(X) == size(iv), DimensionMismatch)
    cmsk = coverage_mask(X, iv, src; dims = 1)
    Xc, ivc = coverage_reduced_pair(X, iv, cmsk)
    rho = Statistics.cor(ce, Xc; dims = 1, mean = mean, iv = ivc,
                         ivpa = coverage_reduced_ivpa(ivpa, cmsk), kwargs...)
    return expand_moment(rho, cmsk)
end

export ImpliedVolatility, ImpliedVolatilityPremium, ImpliedVolatilityRegression
public ImpliedVolatilityAlgorithm, predict_realised_vols
