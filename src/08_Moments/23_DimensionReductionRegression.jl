"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all dimension reduction regression algorithm targets.

All concrete and/or abstract types implementing dimension reduction algorithms for regression (such as PCA or PPCA) should be subtypes of `DimensionReductionTarget`.

These types are used to specify the dimension reduction method when constructing a [`DimensionReductionRegression`](@ref) estimator. A target must answer `StatsAPI.fit(tgt, X)` with a model that `StatsAPI.predict` and `MultivariateStats.projection` both accept.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`prep_dim_red_reg`](@ref)
"""
abstract type DimensionReductionTarget <: AbstractRegressionAlgorithm end
"""
    factory(drtgt::DimensionReductionTarget, args...; kwargs...) -> DimensionReductionTarget

No-op factory for [`DimensionReductionTarget`](@ref) subtypes. Returns the target unchanged.

Dimension reduction targets (such as [`PCA`](@ref) and [`PPCA`](@ref)) do not depend on observation weights, so this method returns `drtgt` unchanged. This allows generic code to call `factory` on dimension reduction targets without special-casing. The weights reach the reduction through [`DimensionReductionRegression`](@ref)'s `ve` instead, which standardises the factors before the target ever sees them.

# Arguments

  - `drtgt`: Dimension reduction target.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `drtgt`: The input dimension reduction target, unchanged.

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`factory`](@ref)
"""
function factory(drtgt::DimensionReductionTarget, args...; kwargs...)
    return drtgt
end
"""
$(DocStringExtensions.TYPEDEF)

Replaces the factors with the principal components of their standardised covariance.

The `kwargs` field is forwarded to `MultivariateStats.fit(MultivariateStats.PCA, X; kwargs...)`, and it is the only place the retained width is set: `pratio` caps the share of variance the retained components must explain and `maxoutdim` caps their number. The default `kwargs = (;)` takes that library's own defaults, which on a factor matrix of full rank retain every component and reduce nothing: on five factors of full rank `PCA()` retained five components, while `PCA(; kwargs = (; pratio = 0.8))` retained four and `PCA(; kwargs = (; maxoutdim = 2))` retained two.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PCA(;
        kwargs::NamedTuple = (;)
    ) -> PCA

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PCA()
PCA
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PPCA`](@ref)

# References

  - $(ref_dict[:pearson1901])
  - $(ref_dict[:hotelling1933])
"""
@concrete struct PCA <: DimensionReductionTarget
    """
    Keyword arguments passed to `fit(MultivariateStats.PCA, X; kwargs...)`
    """
    kwargs
    function PCA(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function PCA(; kwargs::NamedTuple = (;))::PCA
    return PCA(kwargs)
end
"""
    StatsAPI.fit(drtgt::PCA, X::MatNum)

Fit a Principal Component Analysis (PCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PCA as a dimension reduction technique for regression-based moment estimation.

# Algorithm

 1. Read `drtgt.kwargs`, which carries the retained width through `pratio` and `maxoutdim`.
 2. Call [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit) on `MultivariateStats.PCA` with `X` and those keyword arguments, giving the fitted model.

# Arguments

  - `drtgt`: A [`PCA`](@ref) dimension reduction target, specifying keyword arguments for PCA.
  - `X`: Data matrix `factors × observations`, standardised by the caller.

# Returns

  - `model::PCA`: A fitted PCA model object from `MultivariateStats.jl`.

# Related

  - [`PCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
"""
function StatsAPI.fit(drtgt::PCA, X::MatNum)
    return StatsAPI.fit(MultivariateStats.PCA, X; drtgt.kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Replaces the factors with the latent components of a Gaussian latent-variable model.

The model is the maximum-likelihood factor analyser with an isotropic noise variance; its latent directions span the same subspace as the principal components of [`PCA`](@ref), and they coincide with them in the zero-noise limit. The `kwargs` field is forwarded to `MultivariateStats.fit(MultivariateStats.PPCA, X; kwargs...)`. Its default width is one fewer than [`PCA`](@ref)'s, because that library caps a latent-variable model at one less than the number of input dimensions: on five factors of full rank `PCA()` retained five components and `PPCA()` retained four. `maxoutdim` lowers that width and **must not raise it to the factor count**: at the full width the third-party fit succeeds and `MultivariateStats.projection` then raises an `ArgumentError` out of its singular value decomposition, so the failure would surface inside [`prep_dim_red_reg`](@ref) rather than at construction. [`StatsAPI.fit(::PPCA, ::MatNum)`](@ref) checks the cap before it calls that library, and raises a `DomainError` naming `maxoutdim` instead. The constructor cannot hold the check, because it never sees the factor matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PPCA(;
        kwargs::NamedTuple = (;)
    ) -> PPCA

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PPCA()
PPCA
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)

# References

  - $(ref_dict[:tipping1999])
"""
@concrete struct PPCA <: DimensionReductionTarget
    """
    Keyword arguments passed to `fit(MultivariateStats.PPCA, X; kwargs...)`
    """
    kwargs
    function PPCA(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function PPCA(; kwargs::NamedTuple = (;))::PPCA
    return PPCA(kwargs)
end
"""
    StatsAPI.fit(drtgt::PPCA, X::MatNum)

Fit a Probabilistic Principal Component Analysis (PPCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PPCA as a dimension reduction technique for regression-based moment estimation.

# Algorithm

 1. Read `drtgt.kwargs`, which carries the retained width through `maxoutdim`.
 2. If `maxoutdim` is present, check it against the number of factors, `size(X, 1)`.
 3. Call [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit) on `MultivariateStats.PPCA` with `X` and those keyword arguments, giving the fitted model.

# Arguments

  - `drtgt`: A [`PPCA`](@ref) dimension reduction target, specifying keyword arguments for PPCA.
  - `X`: Data matrix `factors × observations`, standardised by the caller.

# Validation

  - If `drtgt.kwargs` carries a `maxoutdim` entry, `0 < drtgt.kwargs.maxoutdim < size(X, 1)` must hold. `MultivariateStats` caps a probabilistic PCA at one latent dimension fewer than the number of factors, and its own fit accepts the full width and returns a model whose weights are `NaN`. Without this check the failure reaches the caller as an `ArgumentError` from LAPACK, raised by `MultivariateStats.projection` inside [`prep_dim_red_reg`](@ref), which names neither the cause nor the keyword.

# Returns

  - `model::PPCA`: A fitted PPCA model object from `MultivariateStats.jl`.

# Related

  - [`PPCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
"""
function StatsAPI.fit(drtgt::PPCA, X::MatNum)
    if haskey(drtgt.kwargs, :maxoutdim)
        maxoutdim = drtgt.kwargs.maxoutdim
        @argcheck(zero(maxoutdim) < maxoutdim < size(X, 1),
                  DomainError(maxoutdim,
                              "MultivariateStats caps a probabilistic PCA at one latent dimension fewer than the number of factors, so 0 < kwargs.maxoutdim < size(X, 1) must hold. Got\nkwargs.maxoutdim => $maxoutdim\nsize(X, 1) => $(size(X, 1))"))
    end
    return StatsAPI.fit(MultivariateStats.PPCA, X; drtgt.kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates a loadings matrix by regressing each asset on the leading components of the factors.

`drtgt` reduces the standardised factors to a smaller orthogonal basis, `retgt` fits each asset in that basis, and the coefficients are then mapped back to the original factors. `ve` supplies the mean and the standard deviation that mapping divides by; the expected returns estimator it reads is `ve.me`, and a `nothing` there falls back to `SimpleExpectedReturns()`. Unlike [`StepwiseRegression`](@ref), every asset keeps every factor. **The standardisation and the recovery read the same statistics**: [`prep_dim_red_reg`](@ref) computes them from `ve`, and `_regression` recovers the coefficients with the pair it returned, so a weighted `ve` — the one [`factory`](@ref) builds from the incoming observation weights — is honoured end to end, as Equations 4.13, 4.15 and 4.20 of $(ref_dict[:cajas2025]) require.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionReductionRegression(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        drtgt::DimensionReductionTarget = PCA(),
        retgt::AbstractRegressionTarget = LinearModel()
    ) -> DimensionReductionRegression

Keywords correspond to the struct's fields.

## Validation

  - If `retgt.kwargs` carries a `weights` entry, it must be an `ObsWeights` and, when it is a vector, `!isempty(retgt.kwargs.weights)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ve`: Recursively updated via [`factory`](@ref).
  - `drtgt`: Recursively updated via [`factory`](@ref).
  - `retgt`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> DimensionReductionRegression()
DimensionReductionRegression
     ve ┼ SimpleVariance
        │          me ┼ SimpleExpectedReturns
        │             │   w ┴ nothing
        │           w ┼ nothing
        │   corrected ┴ Bool: true
  drtgt ┼ PCA
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
  retgt ┼ LinearModel
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`Regression`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.12-4.20.
  - $(ref_dict[:fekedulegn2002])
"""
@propagatable @concrete struct DimensionReductionRegression <:
                               AbstractTimeSeriesRegressionEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:drtgt])
    """
    @fprop drtgt
    """
    $(field_dict[:dretgt])
    """
    @fprop retgt
    function DimensionReductionRegression(ve::AbstractVarianceEstimator,
                                          drtgt::DimensionReductionTarget,
                                          retgt::AbstractRegressionTarget)
        if haskey(retgt.kwargs, :weights)
            @argcheck(isa(retgt.kwargs.weights, ObsWeights),
                      ArgumentError("retgt.kwargs.weights must be a vector of observation weights, one element per observation, of type ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}. Got\nretgt.kwargs.weights => $(typeof(retgt.kwargs.weights))"))
            if isa(retgt.kwargs.weights, AbstractVector)
                @argcheck(!isempty(retgt.kwargs.weights), IsEmptyError)
            end
        end
        return new{typeof(ve), typeof(drtgt), typeof(retgt)}(ve, drtgt, retgt)
    end
end
function DimensionReductionRegression(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                      drtgt::DimensionReductionTarget = PCA(),
                                      retgt::AbstractRegressionTarget = LinearModel())::DimensionReductionRegression
    return DimensionReductionRegression(ve, drtgt, retgt)
end
"""
    prep_dim_red_reg(re::DimensionReductionRegression, X::MatNum)

Standardises the factors, fits the dimension reduction model, and projects the factors into the reduced basis.

It returns the two statistics that did the standardisation along with the projection, because the caller must undo that same scale. Equations 4.13, 4.15 and 4.20 of $(ref_dict[:cajas2025]) hold only when the two are the same statistic.

# Algorithm

 1. Read the expected returns estimator from `re.ve.me`, giving `me`. Fall back to `SimpleExpectedReturns()` when it is `nothing`.
 2. Take the standard deviation of each column of `X` under `re.ve`, giving `sigma`, and raise every entry to at least `eps(eltype(sigma))`, so a constant factor cannot divide by zero.
 3. Take the mean of each column of `X` under `me`, giving `mu`.
 4. Centre `X` with [`demean_returns`](@ref) at `mu`, divide each column by its entry of `sigma`, and transpose, giving `X_std`.
 5. Fit `re.drtgt` to `X_std`, giving `model`.
 6. Project `X_std` through `model` and transpose, giving `Xp`, the factors in the reduced basis.
 7. Read the projection matrix of `model`, giving `Vp`.
 8. Prepend a column of ones to `Xp`, giving `x1`.

# Arguments

  - `re`: Dimension reduction regression estimator. Its `ve` supplies the standard deviation, and its `ve.me` the mean. A `nothing` in `ve.me` falls back to `SimpleExpectedReturns()`.
  - `X`: Factor matrix `observations × factors`, to be reduced.

# Returns

  - `x1::MatNum`: Projected factor matrix `observations × components`, with an intercept column prepended.
  - `Vp::MatNum`: Projection matrix `factors × components`, from the fitted dimension reduction model.
  - `mu::VecNum`: Factor means used to centre `X`.
  - `sigma::VecNum`: Factor standard deviations used to scale `X`.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`demean_returns`](@ref)
  - [`_regression(::DimensionReductionRegression, ::VecNum, ::VecNum, ::VecNum, ::MatNum, ::MatNum)`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.13, 4.16-4.17.
  - $(ref_dict[:fekedulegn2002])
"""
function prep_dim_red_reg(re::DimensionReductionRegression, X::MatNum)
    N = size(X, 1)
    me = ifelse(isnothing(re.ve.me), SimpleExpectedReturns(), re.ve.me)
    sigma = vec(Statistics.std(re.ve, X; dims = 1))
    sigma .= max.(sigma, eps(eltype(sigma)))
    mu = Statistics.mean(me, X; dims = 1)
    X_std = permutedims(demean_returns(X, me; dims = 1, mean = mu) ./ transpose(sigma))
    model = StatsAPI.fit(re.drtgt, X_std)
    Xp = transpose(StatsAPI.predict(model, X_std))
    Vp = MultivariateStats.projection(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp, vec(mu), sigma
end
"""
    _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum,
               sigma::VecNum, x1::MatNum, Vp::MatNum)

Fits one asset in the reduced basis and maps its coefficients back to the original factors.

The reduced-space intercept is discarded and rebuilt from the response mean, so a fit and its recovery agree only while `mu` is the mean under the weights that fit used. Matched, the two paths predict the same values to `4.4e-16` on a 200×5 sample, weighted and unweighted alike; standardise with an unweighted mean and fit with weights, and the same sample parts by `2.1e-3`.

# Mathematical definition

```math
\\begin{align}
\\hat{y} &= \\hat{\\beta}_{0,\\mathrm{pc}} + \\mathbf{X}_1 \\hat{\\boldsymbol{\\beta}}_{\\mathrm{pc}}\\,, \\\\
\\hat{\\boldsymbol{\\beta}} &= \\mathbf{V}_p \\hat{\\boldsymbol{\\beta}}_{\\mathrm{pc}} \\oslash \\boldsymbol{\\sigma}\\,, \\\\
\\hat{\\beta}_0 &= \\bar{y} - \\hat{\\boldsymbol{\\beta}}^{\\intercal} \\boldsymbol{\\mu}\\,.
\\end{align}
```

Where:

  - ``\\hat{y}``: Fitted response.
  - ``\\hat{\\beta}_{0,\\mathrm{pc}}``: Intercept of the fit in the reduced space, which this method discards.
  - ``\\hat{\\boldsymbol{\\beta}}_{\\mathrm{pc}}``: Regression coefficients in the reduced (PC) space.
  - ``\\hat{\\boldsymbol{\\beta}}``: Regression coefficients in the original factor space.
  - ``\\hat{\\beta}_0``: Intercept adjusted to the original space.
  - ``\\mathbf{X}_1``: Projected factor matrix in the reduced space, with its leading column of ones.
  - ``\\mathbf{V}_p``: PCA/PPCA projection matrix.
  - ``\\boldsymbol{\\sigma}``: Factor standard deviations.
  - ``\\boldsymbol{\\mu}``: Factor means.
  - ``\\bar{y}``: Mean of the response.
  - $(math_dict[:oslash])

# Algorithm

 1. Take the mean of `y`, giving `mean_y`. Weight it by `re.retgt.kwargs.weights` when that entry is present.
 2. Fit `re.retgt` to `x1` and `y`, and drop the leading coefficient, giving `beta_pc`.
 3. Map `beta_pc` through `Vp` and divide by `sigma`, giving `beta`, the coefficients in the original factor space.
 4. Subtract the `mu`-weighted sum of `beta` from `mean_y`, giving `beta0`.
 5. Prepend `beta0` to `beta`.

# Arguments

  - `re`: Dimension reduction regression.
  - `y`: Response vector `observations × 1`.
  - `mu`: Mean vector of the original factors. It must be the mean that standardised them, which is why [`prep_dim_red_reg`](@ref) returns it.
  - `sigma`: Standard deviation vector of the original factors. It must be the scale that standardised them, for the same reason.
  - `x1`: Projected factor matrix with intercept column, from [`prep_dim_red_reg`](@ref).
  - `Vp`: Projection matrix from the fitted dimension reduction model.

# Returns

  - `beta::VecNum`: Regression coefficients in the original factor space, with the intercept as the first element.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.18-4.20.
  - $(ref_dict[:fekedulegn2002])
"""
function _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum, sigma::VecNum,
                     x1::MatNum, Vp::MatNum)
    mean_y = if !haskey(re.retgt.kwargs, :weights)
        Statistics.mean(y)
    else
        Statistics.mean(y, re.retgt.kwargs.weights)
    end
    fit_result = StatsAPI.fit(re.retgt, x1, y)
    beta_pc = StatsAPI.coef(fit_result)[2:end]
    beta = Vp * beta_pc ./ sigma
    beta0 = mean_y - LinearAlgebra.dot(beta, mu)
    pushfirst!(beta, beta0)
    return beta
end
"""
    regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)

Reduces the factors once and regresses every asset on the same reduced basis.

The reduction is fitted on `F` alone, so it does not depend on the assets and one asset's response cannot move another's loadings.

# Algorithm

 1. Allocate `rr`, a dense `assets × (factors + 1)` buffer of zeros.
 2. Reduce `F` with [`prep_dim_red_reg`](@ref), giving `f1`, `Vp`, `mu` and `sigma`.
 3. For each asset `i`, fit that column of `X` in the reduced basis and map the coefficients back, and write the result into row `i` of `rr`.
 4. Take the first column of `rr` as `b` and its remaining columns as `M`.
 5. Undo the rescaling and the projection of `M` in turn, giving `L`, the coefficients in the reduced basis.
 6. Build a [`Regression`](@ref) from `b`, `M` and `L`.

# Arguments

  - `re`: Dimension reduction regression estimator that supplies the variance estimator, the dimension reduction target and the regression target.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Returns

  - `reg::Regression`: Regression result carrying:

      + `b`: Intercept of each asset, a view of the first column of `rr`.
      + `M`: Coefficient of each asset and factor in the original factor space, a view of the remaining columns of `rr`. Every asset keeps every factor, so `M` carries no structural zero.
      + `L`: Coefficient of each asset and retained component, ``(\\mathbf{M} \\odot \\boldsymbol{\\sigma}^{\\intercal}) \\mathbf{V}_p^{+\\intercal}``. It reproduces the reduced-space coefficients the fits of step 3 produced, checked at `2.2e-16` against them on a 200×5 sample, and `size(L, 2)` is the number of retained components, which is the width risk is decomposed in.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
  - [`Regression`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.12-4.22.
  - $(ref_dict[:fekedulegn2002])
"""
function regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
    cols = size(F, 2) + 1
    rows = size(X, 2)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    f1, Vp, mu, sigma = prep_dim_red_reg(re, F)
    for i in axes(rr, 1)
        rr[i, :] = _regression(re, view(X, :, i), mu, sigma, f1, Vp)
    end
    b = view(rr, :, 1)
    M = view(rr, :, 2:cols)
    L = transpose(LinearAlgebra.pinv(Vp) * transpose(M .* transpose(sigma)))
    return Regression(; b = b, M = M, L = L)
end

export PCA, PPCA, DimensionReductionRegression
