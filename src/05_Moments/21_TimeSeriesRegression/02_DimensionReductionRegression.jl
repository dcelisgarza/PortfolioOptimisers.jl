"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all dimension reduction regression algorithm targets.

All concrete and/or abstract types implementing dimension reduction algorithms for regression (such as PCA or PPCA) should be subtypes of `DimensionReductionTarget`.

These types are used to specify the dimension reduction method when constructing a [`DimensionReductionRegression`](@ref) estimator. A target must answer `StatsAPI.fit(tgt, X)` with a model that `StatsAPI.predict` and [`dimension_reduction_map`](@ref) both accept. The generic method of [`dimension_reduction_map`](@ref) reads `MultivariateStats.projection`, which is correct for a model whose `predict` applies the transpose of that projection.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`prep_dim_red_reg`](@ref)
  - [`dimension_reduction_map`](@ref)
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

The model is the maximum-likelihood factor analyser with an isotropic noise variance; its latent directions span the same subspace as the principal components of [`PCA`](@ref), and they coincide with them in the zero-noise limit. The `kwargs` field is forwarded to `MultivariateStats.fit(MultivariateStats.PPCA, X; kwargs...)`. Its default width is one fewer than [`PCA`](@ref)'s, because that library caps a latent-variable model at one less than the number of input dimensions: on five factors of full rank `PCA()` retained five components and `PPCA()` retained four. `maxoutdim` lowers that width and **must not raise it to the factor count**: at the full width the third-party fit succeeds and `MultivariateStats.projection` then raises an `ArgumentError` out of its singular value decomposition, so the failure would surface inside [`prep_dim_red_reg`](@ref) rather than at construction. The latent components are the posterior means that `StatsAPI.predict` gives, so the regression maps its coefficients back through the matrix that [`dimension_reduction_map`](@ref) reads off that prediction, not through `MultivariateStats.projection`. [`StatsAPI.fit(::PPCA, ::MatNum)`](@ref) checks the cap before it calls that library, and raises a `DomainError` naming `maxoutdim` instead. The constructor cannot hold the check, because it never sees the factor matrix.

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

  - If `drtgt.kwargs` carries a `maxoutdim` entry, `0 < drtgt.kwargs.maxoutdim < size(X, 1)` must hold. `MultivariateStats` caps a probabilistic PCA at one latent dimension fewer than the number of factors, and its own fit accepts the full width and returns a model whose weights are `NaN`. Without this check the weights of the model are `NaN`, and the failure reaches the caller inside [`prep_dim_red_reg`](@ref), with a message that names neither the cause nor the keyword.

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

The reduction reads every observation, so a fit over more observations can reduce to other components. `proj` fixes the components as linear combinations of the original factors, and the fit then makes no reduction. `choice` states whether the online step of a prior that fits this regression writes the components of the first fit into `proj`. A batch fit gives the same answer under both rules.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionReductionRegression(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        drtgt::DimensionReductionTarget = PCA(),
        retgt::AbstractRegressionTarget = LinearModel(),
        choice::AbstractChoiceRule = BatchChoice(),
        proj::Option{<:MatNum} = nothing
    ) -> DimensionReductionRegression

Keywords correspond to the struct's fields.

## Validation

  - `retgt` states its observation weights through [`regression_target_weights`](@ref), which returns `retgt.kwargs.weights` for a [`LinearModel`](@ref) or a [`GeneralisedLinearModel`](@ref). A caller's own target without that method is refused with an `ArgumentError`.
  - If `regression_target_weights(retgt)` is not `nothing`, it must be an `ObsWeights` and, when it is a vector, not empty.
  - If `proj` is not `nothing`, `!isempty(proj)`, and every entry of `proj` is finite, as [`assert_nonempty_finite_val`](@ref) checks.

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
  choice ┼ BatchChoice()
    proj ┴ nothing
```

# Related

  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`regression_target_weights`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`Regression`](@ref)
  - [`PinnedChoice`](@ref)
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
    """
    Choice Rule of the components. Under [`BatchChoice`](@ref), the default, each fit reduces the factors again over every observation. Under [`PinnedChoice`](@ref), the online step of a prior that fits this regression writes the components of the first fit into `proj` of the estimator that it returns. The two agree in a batch fit.
    """
    choice
    """
    Components as linear combinations of the original factors, `factors × components`, or `nothing`. Column `k` holds the weight of each factor in component `k`, after the mean of the factors is subtracted. A matrix makes the fit regress each asset on these components and make no reduction. `nothing` makes the fit reduce the factors with `drtgt`.
    """
    proj
    function DimensionReductionRegression(ve::AbstractVarianceEstimator,
                                          drtgt::DimensionReductionTarget,
                                          retgt::AbstractRegressionTarget,
                                          choice::AbstractChoiceRule,
                                          proj::Option{<:MatNum})
        w = regression_target_weights(retgt)
        if !isnothing(w)
            @argcheck(isa(w, ObsWeights),
                      ArgumentError("The weights of retgt, which regression_target_weights(retgt) returns and which a LinearModel or a GeneralisedLinearModel keeps in retgt.kwargs.weights, must be a vector of observation weights, one element per observation, of type ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}. Got\nregression_target_weights(retgt) => $(typeof(w))"))
            if isa(w, AbstractVector)
                @argcheck(!isempty(w), IsEmptyError)
            end
        end
        assert_nonempty_finite_val(proj, :proj)
        return new{typeof(ve), typeof(drtgt), typeof(retgt), typeof(choice), typeof(proj)}(ve,
                                                                                           drtgt,
                                                                                           retgt,
                                                                                           choice,
                                                                                           proj)
    end
end
function DimensionReductionRegression(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                      drtgt::DimensionReductionTarget = PCA(),
                                      retgt::AbstractRegressionTarget = LinearModel(),
                                      choice::AbstractChoiceRule = BatchChoice(),
                                      proj::Option{<:MatNum} = nothing)::DimensionReductionRegression
    return DimensionReductionRegression(ve, drtgt, retgt, choice, proj)
end
"""
    prep_dim_red_reg(re::DimensionReductionRegression{<:Any, <:Any, <:Any, <:Any, Nothing},
                     X::MatNum)
    prep_dim_red_reg(re::DimensionReductionRegression{<:Any, <:Any, <:Any, <:Any, <:MatNum},
                     X::MatNum)

Standardises the factors, fits the dimension reduction model, and projects the factors into the reduced basis.

It returns the two statistics that did the standardisation along with the projection, because the caller must undo that same scale. Equations 4.13, 4.15 and 4.20 of $(ref_dict[:cajas2025]) hold only when the two are the same statistic.

When `re.proj` holds the components, the method makes no reduction. It centres the factors at their mean and multiplies them by `re.proj`. It returns `re.proj` as the projection and a scale of ones, so the caller recovers the coefficients with the same formula. A constant shift of the components changes only the intercept of the fit in the reduced basis, which `_regression` discards, so the components need no centre of their own.

# Algorithm

The method that Julia selects is the algorithm.

 1. Read the expected returns estimator from `re.ve.me`, giving `me`. Fall back to `SimpleExpectedReturns()` when it is `nothing`.
 2. Take the mean of each column of `X` under `me`, giving `mu`.
 3. `re.proj` is `nothing`:
     1. Take the standard deviation of each column of `X` under `re.ve`, giving `sigma`, and raise every entry to at least `eps(eltype(sigma))`, so a constant factor cannot divide by zero.
     2. Centre `X` with [`demean_returns`](@ref) at `mu`, divide each column by its entry of `sigma`, and transpose, giving `X_std`.
     3. Fit `re.drtgt` to `X_std`, giving `model`.
     4. Project `X_std` through `model` and transpose, giving `Xp`, the factors in the reduced basis.
     5. Read the matrix of the projection of `model` with [`dimension_reduction_map`](@ref), giving `Vp`.
 4. `re.proj` is a matrix:
     1. Centre `X` with [`demean_returns`](@ref) at `mu`, and multiply it by `re.proj`, giving `Xp`.
     2. Take `Vp = re.proj`, and a vector of ones as `sigma`.
 5. Prepend a column of ones to `Xp`, giving `x1`.

# Arguments

  - `re`: Dimension reduction regression estimator. Its `ve` supplies the standard deviation, and its `ve.me` the mean. A `nothing` in `ve.me` falls back to `SimpleExpectedReturns()`.
  - `X`: Factor matrix `observations × factors`, to be reduced.

# Validation

  - If `re.proj` is a matrix, `size(re.proj, 1) == size(X, 2)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `x1::MatNum`: Projected factor matrix `observations × components`, with an intercept column prepended.
  - `Vp::MatNum`: Projection matrix `factors × components`. Its transpose maps one standardised observation to its components.
  - `mu::VecNum`: Factor means used to centre `X`.
  - `sigma::VecNum`: Factor standard deviations used to scale `X`, or ones when `re.proj` holds the components.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`demean_returns`](@ref)
  - [`dimension_reduction_map`](@ref)
  - [`_regression(::DimensionReductionRegression, ::VecNum, ::VecNum, ::VecNum, ::MatNum, ::MatNum)`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.13, 4.16-4.17.
  - $(ref_dict[:fekedulegn2002])
"""
function prep_dim_red_reg(re::DimensionReductionRegression{<:Any, <:Any, <:Any, <:Any,
                                                           Nothing}, X::MatNum)
    N = size(X, 1)
    me = ifelse(isnothing(re.ve.me), SimpleExpectedReturns(), re.ve.me)
    sigma = vec(Statistics.std(re.ve, X; dims = 1))
    sigma .= max.(sigma, eps(eltype(sigma)))
    mu = Statistics.mean(me, X; dims = 1)
    X_std = permutedims(demean_returns(X, me; dims = 1, mean = mu) ./ transpose(sigma))
    model = StatsAPI.fit(re.drtgt, X_std)
    Xp = transpose(StatsAPI.predict(model, X_std))
    Vp = dimension_reduction_map(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp, vec(mu), sigma
end
function prep_dim_red_reg(re::DimensionReductionRegression{<:Any, <:Any, <:Any, <:Any,
                                                           <:MatNum}, X::MatNum)
    @argcheck(size(re.proj, 1) == size(X, 2),
              DimensionMismatch("re.proj holds one row per factor. Got\nsize(re.proj, 1) => $(size(re.proj, 1))\nsize(X, 2) => $(size(X, 2))"))
    me = ifelse(isnothing(re.ve.me), SimpleExpectedReturns(), re.ve.me)
    mu = Statistics.mean(me, X; dims = 1)
    Xp = demean_returns(X, me; dims = 1, mean = mu) * re.proj
    x1 = [ones(eltype(X), size(X, 1)) Xp]
    return x1, re.proj, vec(mu), fill(one(eltype(re.proj)), size(re.proj, 1))
end
"""
    dimension_reduction_map(model)
    dimension_reduction_map(model::MultivariateStats.PPCA)

Returns the matrix whose transpose maps one centred, standardised observation of the factors to its components, as `StatsAPI.predict` of `model` maps it.

[`prep_dim_red_reg`](@ref) regresses each asset on the components that `StatsAPI.predict` gives, and `_regression` maps the coefficients back to the factors through this matrix. The recovery is exact only when this matrix is the one that the prediction applies.

# Mathematical definition

The prediction of a probabilistic PCA is the posterior mean of its latent components:

```math
\\begin{align}
\\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] &= \\mathbf{C}^{-1} \\mathbf{W}^{\\intercal} (\\mathbf{x} - \\boldsymbol{\\mu}_{\\mathrm{pp}})\\,, \\\\
\\mathbf{C} &= \\mathbf{W}^{\\intercal} \\mathbf{W} + \\sigma^2 \\mathbf{I}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{z}``: Latent components of one observation.
  - ``\\mathbf{x}``: One standardised observation of the factors.
  - ``\\mathbf{W}``: Weights of the model, `factors × components`.
  - ``\\sigma^2``: Noise variance of the model.
  - ``\\boldsymbol{\\mu}_{\\mathrm{pp}}``: Mean of the model.
  - ``\\mathbf{C}``: The matrix that `Statistics.cov` of the model gives.
  - $(math_dict[:I_identity])

So the matrix is ``\\mathbf{W} \\mathbf{C}^{-1}``, which differs from `MultivariateStats.projection`, the left singular vectors of ``\\mathbf{W}``.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any model: return `MultivariateStats.projection(model)`. For a PCA this is the matrix that the prediction applies.
 2. A probabilistic PCA: return ``\\mathbf{W} \\mathbf{C}^{-1}``, computed as `model.W / Statistics.cov(model)`.

# Arguments

  - `model`: Fitted dimension reduction model, from `StatsAPI.fit` of a [`DimensionReductionTarget`](@ref).

# Returns

  - `Vp::MatNum`: Matrix `factors × components`.

# Related

  - [`prep_dim_red_reg`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)

# References

  - $(ref_dict[:tipping1999])
"""
function dimension_reduction_map(model)
    return MultivariateStats.projection(model)
end
function dimension_reduction_map(model::MultivariateStats.PPCA)
    return model.W / Statistics.cov(model)
end
"""
    _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum,
               sigma::VecNum, x1::MatNum, Vp::MatNum)

Fits one asset in the reduced basis and maps its coefficients back to the original factors.

The reduced-space intercept is discarded and rebuilt from the response mean, so a fit and its recovery agree only while `mu` is the mean under the weights that fit used. Matched, the two paths predict the same values, weighted and unweighted alike; standardise with an unweighted mean and fit with weights, and they part.

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

 1. Read the weights of `re.retgt` with [`regression_target_weights`](@ref), giving `w`. Take the mean of `y`, weighted by `w` when it is not `nothing`, giving `mean_y`.
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
  - [`regression_target_weights`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.18-4.20.
  - $(ref_dict[:fekedulegn2002])
"""
function _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum, sigma::VecNum,
                     x1::MatNum, Vp::MatNum)
    w = regression_target_weights(re.retgt)
    mean_y = isnothing(w) ? Statistics.mean(y) : Statistics.mean(y, w)
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
 6. Take `edof`, the degrees of freedom of the residuals, as the number of rows of `f1` less its number of columns, for every asset.
 7. Build a [`Regression`](@ref) from `b`, `M`, `L` and `edof`.

# Arguments

  - `re`: Dimension reduction regression estimator that supplies the variance estimator, the dimension reduction target and the regression target.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Returns

  - `reg::Regression`: Regression result carrying:

      + `b`: Intercept of each asset, a view of the first column of `rr`.
      + `M`: Coefficient of each asset and factor in the original factor space, a view of the remaining columns of `rr`. Every asset keeps every factor, so `M` carries no structural zero.
      + `L`: Coefficient of each asset and retained component, ``(\\mathbf{M} \\odot \\boldsymbol{\\sigma}^{\\intercal}) \\mathbf{V}_p^{+\\intercal}``. It reproduces the reduced-space coefficients the fits of step 3 produced, checked at `2.2e-16` against them on a 200×5 sample, and `size(L, 2)` is the number of retained components, which is the width risk is decomposed in.
      + `edof`: Degrees of freedom of the residuals of each asset, the number of observations less the intercept and the retained components. Every asset spends the same count.

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
    edof = fill(size(f1, 1) - size(f1, 2), rows)
    return Regression(; b = b, M = M, L = L, edof = edof)
end
"""
    pin_regression_choice(re::DimensionReductionRegression{<:Any, <:Any, <:Any,
                                                           <:PinnedChoice, Nothing},
                          X::MatNum, F::MatNum)

Writes the components of the first fit into `proj` of a dimension reduction regression under a [`PinnedChoice`](@ref).

The online step of a prior calls it after the fold, over the rows of the buffer of the prior. The reduction reads `F` alone, so the components that it writes are the components that the fit over the same rows uses. The method writes them in the units of the original factors: the matrix of the projection divided, row by row, by the scale that standardised each factor. A later fit centres the factors at their mean over its own rows and multiplies them by `proj`. So the components keep the weights of the first fit, and the intercept of each asset is still its mean less the part that the factors explain. A regression whose `proj` is a matrix, and a regression under [`BatchChoice`](@ref), get the generic method, which returns `re` unchanged.

# Algorithm

 1. Return `re` when `F` holds fewer than two rows. The regression refuses one row, so no fit is made yet.
 2. Reduce `F` with [`prep_dim_red_reg`](@ref), giving `Vp` and `sigma`.
 3. Return `re` rebuilt with `proj = Vp ./ sigma`.

# Arguments

  - `re`: Dimension reduction regression under a [`PinnedChoice`](@ref), whose `proj` is `nothing`.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Validation

  - The rules of [`prep_dim_red_reg`](@ref).

# Returns

  - `re::DimensionReductionRegression`: The regression with the components of the first fit in `proj`.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PinnedChoice`](@ref)
  - [`pin_prior_choice`](@ref)
"""
function pin_regression_choice(re::DimensionReductionRegression{<:Any, <:Any, <:Any,
                                                                <:PinnedChoice, Nothing},
                               ::MatNum, F::MatNum)
    if size(F, 1) < 2
        return re
    end
    _, Vp, _, sigma = prep_dim_red_reg(re, F)
    return rebuild_estimator(re, (; proj = Vp ./ sigma))
end

export PCA, PPCA, DimensionReductionRegression
