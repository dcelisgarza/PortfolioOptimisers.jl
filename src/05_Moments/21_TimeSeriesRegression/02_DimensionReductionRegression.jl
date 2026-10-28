"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all dimension reduction target types.

A subtype sets the method that reduces the factors of a [`DimensionReductionRegression`](@ref), for example [`PCA`](@ref) or [`PPCA`](@ref). A target must answer `StatsAPI.fit(tgt, X)` with a model that `StatsAPI.predict` and [`dimension_reduction_map`](@ref) both accept. The generic method of [`dimension_reduction_map`](@ref) reads `MultivariateStats.projection`. That matrix is correct for a model whose `predict` applies the transpose of that projection.

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

Returns a [`DimensionReductionTarget`](@ref) unchanged.

A dimension reduction target, for example [`PCA`](@ref) or [`PPCA`](@ref), does not read observation weights. So this method returns `drtgt` unchanged, and generic code can call `factory` on a target with no special case. The weights reach the reduction through the `ve` field of [`DimensionReductionRegression`](@ref), which standardises the factors before the target sees them.

# Arguments

  - `drtgt`: Dimension reduction target.
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

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

The principal components are those of [pearson1901](@cite) and [hotelling1933](@cite). No copy of either work was found, so the components of the standardised factors follow Equation 34 of [fekedulegn2002](@cite).

The fit passes the `kwargs` field to `MultivariateStats.fit(MultivariateStats.PCA, X; kwargs...)`, and only this field sets the retained width. `pratio` sets the share of the variance that the retained components must explain, and `maxoutdim` caps their number. The default `kwargs = (;)` takes the defaults of that library. On a factor matrix of full rank, these defaults retain every component and reduce nothing. On a sample of five factors of full rank, `PCA()` retained five components, `PCA(; kwargs = (; pratio = 0.8))` retained four, and `PCA(; kwargs = (; maxoutdim = 2))` retained two.

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
  - $(ref_dict[:fekedulegn2002]) Equation 34. $(ref_dict[:no_original_source])
"""
@concrete struct PCA <: DimensionReductionTarget
    """
    Keyword arguments that the fit passes to `fit(MultivariateStats.PCA, X; kwargs...)`.
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

Fits a principal component analysis model of `MultivariateStats` to the data matrix `X`, with the keyword arguments of `drtgt`.

# Algorithm

 1. Read `drtgt.kwargs`, which sets the retained width through `pratio` and `maxoutdim`.
 2. Call [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit) on `MultivariateStats.PCA` with `X` and those keyword arguments, and return the fitted model.

# Arguments

  - `drtgt`: [`PCA`](@ref) target that holds the keyword arguments of the fit.
  - `X`: Data matrix `factors × observations`, which the caller standardises.

# Returns

  - `model::MultivariateStats.PCA`: The fitted model.

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

The model is the maximum-likelihood factor analyser with an isotropic noise variance. Its latent directions span the same subspace as the principal components of [`PCA`](@ref), and they are equal to them in the limit of zero noise. The fit is the maximum-likelihood estimate of Section 3.2 of [tipping1999](@cite). The weights are those of Equation 7 with the rotation set to the identity. The noise variance is that of Equation 8, the mean of the discarded eigenvalues. `MultivariateStats` takes the eigenvalues from the sample covariance with divisor ``N - 1``, where Equation 5 divides by ``N``. The subspace is the same, and the eigenvalues, the weights and the noise variance carry that divisor.

The fit passes the `kwargs` field to `MultivariateStats.fit(MultivariateStats.PPCA, X; kwargs...)`. The default width is one fewer than the width of [`PCA`](@ref), because that library caps a latent-variable model at one less than the number of input dimensions. On five factors of full rank, `PCA()` retained five components and `PPCA()` retained four. `maxoutdim` can lower that width, and it must stay below the factor count.

[`StatsAPI.fit(::PPCA, ::MatNum)`](@ref) checks that bound before it calls `MultivariateStats`, and it raises a `DomainError` that names `maxoutdim`. Without the check, the fit of that library accepts the full width and returns weights of `NaN`. `StatsAPI.predict` in [`prep_dim_red_reg`](@ref) then raises an `ArgumentError` that names neither the cause nor the keyword. The constructor cannot make the check, because it does not see the factor matrix.

The latent components are the posterior means that `StatsAPI.predict` gives. So the regression maps its coefficients back through the matrix that [`dimension_reduction_map`](@ref) reads off that prediction, not through `MultivariateStats.projection`.

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

  - $(ref_dict[:tipping1999]) Section 3.2, Equations 7-8.
"""
@concrete struct PPCA <: DimensionReductionTarget
    """
    Keyword arguments that the fit passes to `fit(MultivariateStats.PPCA, X; kwargs...)`.
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

Fits a probabilistic principal component analysis model of `MultivariateStats` to the data matrix `X`, with the keyword arguments of `drtgt`.

# Algorithm

 1. Read `drtgt.kwargs`, which sets the retained width through `maxoutdim`.
 2. If `maxoutdim` is present, check it against the number of factors, `size(X, 1)`.
 3. Call [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit) on `MultivariateStats.PPCA` with `X` and those keyword arguments, and return the fitted model.

# Arguments

  - `drtgt`: [`PPCA`](@ref) target that holds the keyword arguments of the fit.
  - `X`: Data matrix `factors × observations`, which the caller standardises.

# Validation

  - If `drtgt.kwargs` holds a `maxoutdim` entry, `0 < drtgt.kwargs.maxoutdim < size(X, 1)` must hold. `MultivariateStats` caps a probabilistic PCA at one latent dimension fewer than the number of factors. Its own fit accepts the full width and returns a model whose weights are `NaN`. Without this check, the failure reaches the caller later, inside [`prep_dim_red_reg`](@ref), with a message that names neither the cause nor the keyword.

# Returns

  - `model::MultivariateStats.PPCA`: The fitted model.

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

`drtgt` maps the standardised factors to a basis of orthogonal components, and `retgt` fits each asset in that basis. The basis is smaller than the factor set only when `drtgt` retains fewer components than factors. The method then maps the coefficients back to the original factors. `ve` gives the mean, and the standard deviation by which that map divides. The mean comes from the expected returns estimator `ve.me`, or from `SimpleExpectedReturns()` when `ve.me` is `nothing`. Every asset keeps every factor, which is not true of [`StepwiseRegression`](@ref).

The standardisation and the recovery read the same statistics. [`prep_dim_red_reg`](@ref) computes them from `ve`, and `_regression` recovers the coefficients with the pair that it returns. So a weighted `ve`, which [`factory`](@ref) builds from the observation weights, gives its weights to both steps. Equations 8, 11, 12, 35 and 36 of [fekedulegn2002](@cite) need this. That work fits the regression on the principal components of the standardised factors, with the eigenvectors of their correlation matrix. A probabilistic target puts the matrix of Equation 9 of [tipping1999](@cite) in place of the eigenvectors, and neither source states this composition.

A penalised `retgt`, such as [`ElasticNet`](@ref), keeps the fit and the recovery in agreement. The estimator names the column of ones with [`InterceptColumn`](@ref), so the target does not penalise the intercept. The intercept is then the weighted mean of the residual of the slopes. The components have a weighted mean of zero, so the intercept is the weighted mean of the response, which the recovery rebuilds. The condition is the same as for an unpenalised target: `ve` and `retgt` carry the same weights, as [`factory`](@ref) gives them. The penalty shrinks the coefficients of the components, which are in the units of the standardised factors. The degrees of freedom of the residuals count one for each column of the reduced design, and a penalised fit has fewer.

The reduction reads every observation, so a fit over more observations can reduce to other components. `proj` fixes the components as linear combinations of the original factors, and the fit then makes no reduction. `choice` states whether the online step of a prior that fits this regression writes the components of the first fit into `proj`. A batch fit gives the same answer under both rules.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionReductionRegression(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        drtgt::DimensionReductionTarget = PCA(),
        retgt::AbstractRegressionTarget = LinearModel(),
        choice::AbstractChoiceRule = BatchChoice(),
        proj::Option{<:MatNum} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> DimensionReductionRegression

Keywords correspond to the struct's fields.

## Validation

  - `retgt` states its observation weights through [`regression_target_weights`](@ref), which returns `retgt.kwargs.weights` for a [`LinearModel`](@ref) or a [`GeneralisedLinearModel`](@ref). The constructor raises an `ArgumentError` for a target of the caller that has no such method.
  - If `regression_target_weights(retgt)` is not `nothing`, it must be an `ObsWeights` and, when it is a vector, not empty.
  - If `proj` is not `nothing`, `!isempty(proj)`, and every entry of `proj` is finite, as [`assert_nonempty_finite_val`](@ref) checks.
  - `x_src in ALL_RETURNS_SOURCES`, as [`assert_returns_source`](@ref) checks. A regression reader admits every source.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated through [`factory`](@ref).
  - `drtgt`: Recursively updated through [`factory`](@ref).
  - `retgt`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed through [`port_opt_view`](@ref).

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
    proj ┼ nothing
   x_src ┴ nothing
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
  - [`AbstractReturnsSource`](@ref)
  - [`resolve_factor_regression`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:fekedulegn2002]) Pages 4-7 and 21-22, Equations 8, 11-12, 20-23 and 34-36.
  - $(ref_dict[:tipping1999]) Section 3.4, Equation 9.
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
    Choice Rule of the components. Under [`BatchChoice`](@ref), the default, each fit reduces the factors again over every observation. Under [`PinnedChoice`](@ref), the online step of a prior that fits this regression writes the components of the first fit into `proj` of the estimator that it returns. The two agree in a batch fit. An entropy pooling prior refits the prior under the posterior weights. It holds the regression of the fit that its views read, so its refits agree too.
    """
    choice
    """
    Components as linear combinations of the original factors, `factors × components`, or `nothing`. Column `k` holds the weight of each factor in component `k`, after the fit subtracts the mean of the factors. A matrix makes the fit regress each asset on these components and make no reduction. `nothing` makes the fit reduce the factors with `drtgt`.
    """
    proj
    """
    $(field_dict[:x_src_re])
    """
    x_src
    function DimensionReductionRegression(ve::AbstractVarianceEstimator,
                                          drtgt::DimensionReductionTarget,
                                          retgt::AbstractRegressionTarget,
                                          choice::AbstractChoiceRule,
                                          proj::Option{<:MatNum},
                                          x_src::Option{<:AbstractReturnsSource})
        w = regression_target_weights(retgt)
        if !isnothing(w)
            @argcheck(isa(w, ObsWeights),
                      ArgumentError("The weights of retgt, which regression_target_weights(retgt) returns and which a LinearModel or a GeneralisedLinearModel keeps in retgt.kwargs.weights, must be a vector of observation weights, one element per observation, of type ObsWeights = Union{DynamicAbstractWeights, StatsBase.AbstractWeights}. Got\nregression_target_weights(retgt) => $(typeof(w))"))
            if isa(w, AbstractVector)
                @argcheck(!isempty(w), IsEmptyError)
            end
        end
        assert_nonempty_finite_val(proj, :proj)
        return new{typeof(ve), typeof(drtgt), typeof(retgt), typeof(choice), typeof(proj),
                   typeof(x_src)}(ve, drtgt, retgt, choice, proj, x_src)
    end
end
function DimensionReductionRegression(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                      drtgt::DimensionReductionTarget = PCA(),
                                      retgt::AbstractRegressionTarget = LinearModel(),
                                      choice::AbstractChoiceRule = BatchChoice(),
                                      proj::Option{<:MatNum} = nothing,
                                      x_src::Option{<:AbstractReturnsSource} = nothing)::DimensionReductionRegression
    assert_returns_source(x_src, ALL_RETURNS_SOURCES, :DimensionReductionRegression)
    return DimensionReductionRegression(ve, drtgt, retgt, choice, proj, x_src)
end
"""
    prep_dim_red_reg(re::DimensionReductionRegression{<:Any, <:Any, <:Any, <:Any, Nothing},
                     X::MatNum)
    prep_dim_red_reg(re::DimensionReductionRegression{<:Any, <:Any, <:Any, <:Any, <:MatNum},
                     X::MatNum)

Standardises the factors, fits the dimension reduction model, and projects the factors into the reduced basis.

It returns the two statistics of the standardisation with the projection, because the caller must undo that same scale. Equations 11 and 12 of [fekedulegn2002](@cite) hold only when the standardisation and the recovery use the same statistics.

When `re.proj` holds the components, the method makes no reduction. It centres the factors at their mean and multiplies them by `re.proj`. It returns `re.proj` as the projection and a scale of ones, so the caller recovers the coefficients with the same formula. A constant shift of the components changes only the intercept of the fit in the reduced basis, and `_regression` discards that intercept. So the components need no centre of their own.

# Algorithm

The method that Julia selects is the algorithm.

 1. Read the expected returns estimator from `re.ve.me`, and call it `me`. Use `SimpleExpectedReturns()` when it is `nothing`.
 2. Take the mean of each column of `X` under `me`, and call it `mu`.
 3. `re.proj` is `nothing`:
     1. Take the standard deviation of each column of `X` under `re.ve`, and call it `sigma`. Raise every entry to at least `eps(eltype(sigma))`, so that a constant factor does not cause a division by zero.
     2. Centre `X` with [`demean_returns`](@ref) at `mu`, divide each column by its entry of `sigma`, and transpose. Call the result `X_std`.
     3. Fit `re.drtgt` to `X_std`, and call the fit `model`.
     4. Project `X_std` through `model` and transpose. Call the result `Xp`, the factors in the reduced basis.
     5. Read the matrix of the projection of `model` with [`dimension_reduction_map`](@ref), and call it `Vp`.
 4. `re.proj` is a matrix:
     1. Centre `X` with [`demean_returns`](@ref) at `mu`, and multiply it by `re.proj`. Call the result `Xp`.
     2. Take `Vp = re.proj`, and a vector of ones as `sigma`.
 5. Prepend a column of ones to `Xp`, and call the result `x1`.

# Arguments

  - `re`: Dimension reduction regression estimator. Its `ve` gives the standard deviation, and its `ve.me` gives the mean. When `ve.me` is `nothing`, the method uses `SimpleExpectedReturns()`.
  - `X`: Factor matrix `observations × factors`, which the method reduces.

# Validation

  - If `re.proj` is a matrix, `size(re.proj, 1) == size(X, 2)`. The method raises a `DimensionMismatch` otherwise.

# Returns

  - `x1::MatNum`: Projected factor matrix `observations × components`, with an intercept column prepended.
  - `Vp::MatNum`: Projection matrix `factors × components`. Its transpose maps one standardised observation to its components.
  - `mu::VecNum`: Factor means that centre `X`.
  - `sigma::VecNum`: Factor standard deviations that scale `X`, or ones when `re.proj` holds the components.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`demean_returns`](@ref)
  - [`dimension_reduction_map`](@ref)
  - [`_regression(::DimensionReductionRegression, ::VecNum, ::VecNum, ::VecNum, ::MatNum, ::MatNum)`](@ref)

# References

  - $(ref_dict[:fekedulegn2002]) Pages 4-5 and 21, Equations 8, 13-17 and 34.
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

The prediction of a probabilistic PCA is the posterior mean of its latent components, Equation 9 of [tipping1999](@cite):

```math
\\begin{align}
\\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] &= \\mathbf{M}^{-1} \\mathbf{W}^{\\intercal} (\\mathbf{x} - \\boldsymbol{\\mu}_{\\mathrm{pp}})\\,, \\\\
\\mathbf{M} &= \\mathbf{W}^{\\intercal} \\mathbf{W} + \\sigma^2 \\mathbf{I}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{z}``: Latent components of one observation.
  - ``\\mathbf{x}``: One standardised observation of the factors.
  - ``\\mathbf{W}``: Weights of the model, `factors × components`.
  - ``\\sigma^2``: Noise variance of the model.
  - ``\\boldsymbol{\\mu}_{\\mathrm{pp}}``: Mean of the model.
  - ``\\mathbf{M}``: The ``q \\times q`` matrix that `Statistics.cov` of the model gives, with ``q`` the number of components.
  - $(math_dict[:I_identity])

So the matrix is ``\\mathbf{W} \\mathbf{M}^{-1}``, which differs from `MultivariateStats.projection`, the left singular vectors of ``\\mathbf{W}``.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any model: return `MultivariateStats.projection(model)`. For a PCA this is the matrix that the prediction applies.
 2. A probabilistic PCA: return ``\\mathbf{W} \\mathbf{M}^{-1}``, computed as `model.W / Statistics.cov(model)`.

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

  - $(ref_dict[:tipping1999]) Section 3.4, Equation 9.
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

The method discards the intercept of the reduced space and builds a new one from the mean of the response. So a fit and its recovery agree only while `mu` is the mean under the weights that the fit used. When the weights match, the two paths predict the same values, with or without weights. When the standardisation uses an unweighted mean and the fit uses weights, the predictions of the two paths are different.

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
  - ``\\hat{\\boldsymbol{\\beta}}_{\\mathrm{pc}}``: Regression coefficients in the reduced space.
  - ``\\hat{\\boldsymbol{\\beta}}``: Regression coefficients in the original factor space.
  - ``\\hat{\\beta}_0``: Intercept in the original factor space.
  - ``\\mathbf{X}_1``: Projected factor matrix in the reduced space, with its leading column of ones.
  - ``\\mathbf{V}_p``: Projection matrix of the dimension reduction model.
  - ``\\boldsymbol{\\sigma}``: Factor standard deviations.
  - ``\\boldsymbol{\\mu}``: Factor means.
  - ``\\bar{y}``: Mean of the response.
  - $(math_dict[:oslash])

# Algorithm

 1. Read the weights of `re.retgt` with [`regression_target_weights`](@ref), and call them `w`. Take the mean of `y`, weighted by `w` when it is not `nothing`, and call it `mean_y`.
 2. Fit `re.retgt` to `x1` and `y`, with the leading column of ones named to it by `factory(re.retgt, InterceptColumn(1))`, and drop the leading coefficient. Call the rest `beta_pc`.
 3. Map `beta_pc` through `Vp` and divide by `sigma`. Call the result `beta`, the coefficients in the original factor space.
 4. Subtract the `mu`-weighted sum of `beta` from `mean_y`, and call the result `beta0`.
 5. Prepend `beta0` to `beta`.

# Arguments

  - `re`: Dimension reduction regression.
  - `y`: Response vector `observations × 1`.
  - `mu`: Mean vector of the original factors. It must be the mean that standardised them. That is why [`prep_dim_red_reg`](@ref) returns it.
  - `sigma`: Standard deviation vector of the original factors. It must be the scale that standardised them, for the same reason.
  - `x1`: Projected factor matrix with an intercept column, from [`prep_dim_red_reg`](@ref).
  - `Vp`: Projection matrix of the fitted dimension reduction model.

# Returns

  - `beta::VecNum`: Regression coefficients in the original factor space, with the intercept as the first element.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
  - [`regression_target_weights`](@ref)

# References

  - $(ref_dict[:fekedulegn2002]) Pages 7 and 21-22, Equations 20-23 and 34-36.
"""
function _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum, sigma::VecNum,
                     x1::MatNum, Vp::MatNum)
    w = regression_target_weights(re.retgt)
    mean_y = isnothing(w) ? Statistics.mean(y) : Statistics.mean(y, w)
    fit_result = StatsAPI.fit(factory(re.retgt, InterceptColumn(1)), x1, y)
    beta_pc = StatsAPI.coef(fit_result)[2:end]
    beta = Vp * beta_pc ./ sigma
    beta0 = mean_y - LinearAlgebra.dot(beta, mu)
    pushfirst!(beta, beta0)
    return beta
end
"""
    regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)

Reduces the factors once and regresses every asset on the same reduced basis.

The method fits the reduction on `F` alone. So the reduction does not depend on the assets, and the response of one asset cannot change the loadings of another.

# Algorithm

 1. Allocate `rr`, a dense `assets × (factors + 1)` buffer of zeros.
 2. Reduce `F` with [`prep_dim_red_reg`](@ref), and call the results `f1`, `Vp`, `mu` and `sigma`.
 3. For each asset `i`, fit that column of `X` in the reduced basis, map the coefficients back, and write the result into row `i` of `rr`.
 4. Take the first column of `rr` as `b` and its remaining columns as `M`.
 5. Undo the rescaling and the projection of `M` in turn. Call the result `L`, the coefficients in the reduced basis.
 6. Negate each column of `L` whose sum is negative. The sign of a component is arbitrary, and a build flag such as `--check-bounds=yes` can flip it. After this step, the equal-weight portfolio has a non-negative exposure to each component. `M` does not change, because a component and its coefficient change sign together.
 7. Set `edof`, the degrees of freedom of the residuals, to the number of rows of `f1` less its number of columns, for every asset.
 8. Build a [`Regression`](@ref) from `b`, `M`, `L` and `edof`.

# Arguments

  - `re`: Dimension reduction regression estimator that gives the variance estimator, the dimension reduction target and the regression target.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Validation

  - The rules of [`assert_finite_factor_rows`](@ref) for `F`.

# Returns

  - `reg::Regression`: Regression result that holds:

      + `b`: Intercept of each asset, a view of the first column of `rr`.
      + `M`: Coefficient of each asset and factor in the original factor space, a view of the remaining columns of `rr`. Every asset keeps every factor, so `M` has no structural zero.
      + `L`: Coefficient of each asset and retained component, ``(\\mathbf{M} \\odot \\boldsymbol{\\sigma}^{\\intercal}) \\mathbf{V}_p^{+\\intercal}``. It reproduces the coefficients of the fits of step 3 in the reduced space, up to the sign of each component that step 6 sets. On a 200×5 sample the two agreed to `2.2e-16`. `size(L, 2)` is the number of retained components, which is the width of a risk decomposition.
      + `edof`: Degrees of freedom of the residuals of each asset, the number of observations less the intercept and the retained components. The count is the same for every asset.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
  - [`Regression`](@ref)

# References

  - $(ref_dict[:fekedulegn2002]) Pages 4-7 and 21-22, Equations 8, 11-12, 20-23 and 34-36.
"""
function regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
    assert_finite_factor_rows(F)
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
    # The sign of a component is arbitrary, and the last bits of the fit decide it (#1756).
    L = L .* transpose(map(c -> ifelse(sum(c) < zero(eltype(L)), -1, 1), eachcol(L)))
    edof = fill(size(f1, 1) - size(f1, 2), rows)
    return Regression(; b = b, M = M, L = L, edof = edof)
end
"""
    pin_regression_choice(re::DimensionReductionRegression{<:Any, <:Any, <:Any,
                                                           <:PinnedChoice, Nothing},
                          X::MatNum, F::MatNum, w::Option{<:ObsWeights} = nothing)

Writes the components of the first fit into `proj` of a dimension reduction regression under a [`PinnedChoice`](@ref).

The online step of a prior calls it after the fold, over the rows of the buffer of the prior. The reduction reads `F` alone, so the components that it writes are the components that the fit over the same rows uses. A prior that refits the regression under observation weights, as [`OpinionPoolingPrior`](@ref) refits `pe2`, gives those weights as `w`, and the reduction reads them as that fit does.

The method writes the components in the units of the original factors. That is the matrix of the projection, with each row divided by the scale that standardised its factor. A later fit centres the factors at their mean over its own rows and multiplies them by `proj`. So the components keep the weights of the first fit, and the intercept of each asset is still its mean less the part that the factors explain. A regression whose `proj` is a matrix, and a regression under [`BatchChoice`](@ref), get the generic method, which returns `re` unchanged.

# Algorithm

 1. Return `re` when `F` holds fewer than two rows. The regression refuses one row, so no fit happens yet.
 2. Reduce `F` with [`prep_dim_red_reg`](@ref), and keep `Vp` and `sigma`. When `w` is not `nothing`, the reduction is the reduction of `factory(re, w)`.
 3. Return `re` rebuilt with `proj = Vp ./ sigma`.

# Arguments

  - `re`: Dimension reduction regression under a [`PinnedChoice`](@ref), whose `proj` is `nothing`.
  - $(arg_dict[:X])
  - $(arg_dict[:F])
  - $(arg_dict[:w_pin])

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
                               ::MatNum, F::MatNum, w::Option{<:ObsWeights} = nothing)
    if size(F, 1) < 2
        return re
    end
    _, Vp, _, sigma = prep_dim_red_reg(isnothing(w) ? re : factory(re, w), F)
    return rebuild_estimator(re, (; proj = Vp ./ sigma))
end
# A method of `pending_choice`, which `src/10_Prior/12_PriorPartialFit.jl` documents. It lives
# beside the pin it answers for.
function pending_choice(::DimensionReductionRegression{<:Any, <:Any, <:Any, <:PinnedChoice,
                                                       Nothing})
    return true
end
"""
    returns_source(re::Union{StepwiseRegression, DimensionReductionRegression})

Returns the source that a regression reader states, which is its `x_src` field.

A prior that holds a regression reads the returns data and gives the regression a matrix. So the prior refuses a regression whose source is not `nothing`. [`assert_inner_reader`](@ref) makes that check, and it reads the source with this method.

# Arguments

  - `re`: Regression reader of the returns.

# Returns

  - `x_src::Option{<:AbstractReturnsSource}`: The source of the regression.

# Related

  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`assert_inner_reader`](@ref)
"""
function returns_source(re::Union{StepwiseRegression, DimensionReductionRegression})
    return re.x_src
end
"""
    reader_frames(re::Union{StepwiseRegression, DimensionReductionRegression},
                  path::AbstractString, refit::Bool)

Returns the frames that a regression reader adds to the group of the optimiser that holds it.

A regression reader at `x_src = nothing` takes the loadings of the Prior Result when it holds them, and then it reads the frame of the prior. It fits on the raw returns `rd.X` only when the Prior Result holds none. So it counts as a raw reader only when `refit` is `true`. [`refits_regression`](@ref) finds `refit` from the prior estimator of the optimiser. A regression reader is outermost when it is in a constraint space or in a factor risk optimiser. A regression inside a prior keeps `x_src = nothing`, and the prior counts for it.

# Arguments

  - `re`: Regression reader of the returns.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of `re`, from [`source_frames`](@ref) with `raw = refit`.

# Related

  - [`reader_frames`](@ref)
  - [`source_frames`](@ref)
  - [`refits_regression`](@ref)
  - [`resolve_factor_regression`](@ref)
"""
function reader_frames(re::Union{StepwiseRegression, DimensionReductionRegression},
                       path::AbstractString, refit::Bool)
    return source_frames(re.x_src, path, refit)
end

export PCA, PPCA, DimensionReductionRegression
