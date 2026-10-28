"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that set the regularisation strength of an [`ElasticNet`](@ref).

The strength scales the penalty against the weighted squared residuals. [`MiddleOfPath`](@ref) reads it off the data, as [lai2018ktpt](@citet) do. [`FixedStrength`](@ref) takes it from the caller.

# Interfaces

To add a strength rule, subtype `AbstractPenaltyStrength`, keep its parameters in the fields of the struct, and add the method below.

## `penalty_strength`

  - `penalty_strength(s::MyPenaltyStrength, g::VecNum, theta::Real) -> Real`: Return the regularisation strength ``\\gamma`` of the fit.

[`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref) calls it once, before the coordinate sweeps. When every entry of `g` is zero, the fit returns its start point at every strength, and it does not read the strength. It records the strength in the fit.

### Arguments

  - `s`: The new strength rule.
  - `g`: The product of each penalised column with the residual of the start point, ``\\mathbf{P}_k^\\intercal \\mathbf{V} \\boldsymbol{r}_0`` for ``k \\in \\mathcal{K}``. It is empty when the design has no penalised column.
  - `theta`: The share of the ``L_1`` penalty, the `theta` field of the target.

### Returns

  - `gamma::Real`: The regularisation strength, non-negative.

# Related

  - [`ElasticNet`](@ref)
  - [`MiddleOfPath`](@ref)
  - [`FixedStrength`](@ref)
  - [`penalty_strength`](@ref)
"""
abstract type AbstractPenaltyStrength <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Sets the regularisation strength of an [`ElasticNet`](@ref) at the geometric middle of its regularisation path.

[lai2018ktpt](@citet) take "the middle one in the regularization path" and state no grid. Section 2.5 of [friedman2010](@cite) runs the path over 100 strengths on a log scale, from the smallest strength at which every penalised coefficient is zero down to `ratio` times that strength. Such a grid has no single middle point. The rule takes the geometric middle of the path, which lies between the 50th and the 51st point of that grid at the default `ratio = 1e-3`.

# Mathematical definition

```math
\\begin{align}
\\gamma_{\\max} &= \\max_{k \\in \\mathcal{K}} \\frac{\\lvert \\mathbf{P}_k^\\intercal \\mathbf{V} \\boldsymbol{r}_0 \\rvert}{\\vartheta}\\,,\\\\
\\gamma &= \\gamma_{\\max} \\sqrt{\\rho}\\,.
\\end{align}
```

Where:

  - ``\\gamma_{\\max}``: Smallest strength at which every penalised coefficient is zero.
  - $(math_dict[:K_enet])
  - $(math_dict[:P_enet])
  - $(math_dict[:V_enet])
  - ``\\boldsymbol{r}_0``: Residual of the start point, the response less the fit of the intercept column alone, or the response when the design has no intercept column.
  - $(math_dict[:theta_enet])
  - $(math_dict[:gamma_enet])
  - ``\\rho``: Floor of the path as a fraction of ``\\gamma_{\\max}``, the `ratio` field.

Every penalised coefficient is zero at and above ``\\gamma_{\\max}``. The path runs from ``\\gamma_{\\max}`` down to ``\\rho\\, \\gamma_{\\max}`` on a log scale, so its geometric middle is ``\\gamma_{\\max} \\sqrt{\\rho}``. A ridge alone, ``\\vartheta = 0``, has no ``\\gamma_{\\max}``, so the constructor of [`ElasticNet`](@ref) refuses `theta = 0` with this rule.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MiddleOfPath(; ratio::Real = 1e-3) -> MiddleOfPath

Keywords correspond to the struct's fields.

## Validation

  - `0 < ratio < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> MiddleOfPath()
MiddleOfPath
  ratio ┴ Float64: 0.001
```

# Related

  - [`AbstractPenaltyStrength`](@ref)
  - [`FixedStrength`](@ref)
  - [`ElasticNet`](@ref)
  - [`penalty_strength`](@ref)

# References

  - $(ref_dict[:friedman2010]) Section 2.5.
  - $(ref_dict[:lai2018ktpt]) Section 3.1.2.
"""
@concrete struct MiddleOfPath <: AbstractPenaltyStrength
    """
    The floor of the path as a fraction of the strength above which every penalised coefficient is zero.
    """
    ratio
    function MiddleOfPath(ratio::Real)
        @argcheck(zero(ratio) < ratio < one(ratio),
                  DomainError(ratio, "ratio must be in (0, 1)"))
        return new{typeof(ratio)}(ratio)
    end
end
function MiddleOfPath(; ratio::Real = 1e-3)::MiddleOfPath
    return MiddleOfPath(ratio)
end
"""
$(DocStringExtensions.TYPEDEF)

Sets the regularisation strength of an [`ElasticNet`](@ref) to a value that the caller states.

`gamma` is the ``\\gamma`` of the objective of [`ElasticNet`](@ref). It weighs the penalty against the sum of the weighted squared residuals, and it does not divide that sum by the number of observations. So the same `gamma` penalises a fit over more observations less, in relation to its residuals. [friedman2010](@citet) divide the objective by twice the sum of the weights, so their strength ``\\lambda`` of Equation (1) is ``\\gamma / T_{e}``, with ``T_{e}`` the sum of the scaled weights, which is the number of observations of an unweighted fit. With a variance estimator in the `ve` field of the target, `gamma` acts on the standardised columns. `theta = 0` with this rule is a ridge regression.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FixedStrength(; gamma::Real) -> FixedStrength

Keywords correspond to the struct's fields.

## Validation

  - `gamma > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> FixedStrength(; gamma = 0.5)
FixedStrength
  gamma ┴ Float64: 0.5
```

# Related

  - [`AbstractPenaltyStrength`](@ref)
  - [`MiddleOfPath`](@ref)
  - [`ElasticNet`](@ref)
  - [`penalty_strength`](@ref)

# References

  - $(ref_dict[:friedman2010]) Equation (1).
"""
@concrete struct FixedStrength <: AbstractPenaltyStrength
    """
    The regularisation strength ``\\gamma`` of the objective.
    """
    gamma
    function FixedStrength(gamma::Real)
        @argcheck(gamma > zero(gamma), DomainError(gamma, "gamma must be positive"))
        return new{typeof(gamma)}(gamma)
    end
end
function FixedStrength(; gamma::Real)::FixedStrength
    return FixedStrength(gamma)
end
"""
    penalty_strength(s::MiddleOfPath, g::AbstractVector, theta::Real) -> Real
    penalty_strength(s::FixedStrength, g::AbstractVector, theta::Real) -> Real

Return the regularisation strength of an [`ElasticNet`](@ref) under its strength rule.

[`MiddleOfPath`](@ref) states the middle of the path, and [`FixedStrength`](@ref) returns its own `gamma`. The `# Interfaces` section of [`AbstractPenaltyStrength`](@ref) states the method that a rule of the caller adds.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`MiddleOfPath`](@ref): divide the largest absolute entry of `g` by `theta`, which gives `lmax`, and return `lmax * sqrt(s.ratio)`. An empty `g` gives zero.
 2. [`FixedStrength`](@ref): return `s.gamma`, and read neither `g` nor `theta`.

# Arguments

  - `s`: The strength rule.
  - `g`: The product of each penalised column with the residual of the start point.
  - `theta`: The share of the ``L_1`` penalty.

# Returns

  - `gamma::Real`: The regularisation strength.

# Related

  - [`AbstractPenaltyStrength`](@ref)
  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)
"""
function penalty_strength(s::MiddleOfPath, g::AbstractVector, theta::Real)
    lmax = maximum(abs, g; init = zero(eltype(g))) / theta
    return lmax * sqrt(s.ratio)
end
function penalty_strength(s::FixedStrength, ::AbstractVector, ::Real)
    return s.gamma
end
"""
$(DocStringExtensions.TYPEDEF)

Fits each response by the elastic net, a least-squares fit with a penalty on the ``L_1`` norm and on the squared ``L_2`` norm of the coefficients.

The objective is Equation (15) of [lai2018ktpt](@cite), the elastic net of [zouhastie2005](@citet), under observation weights. It is Equation (3) of [zouhastie2005](@cite) with ``\\lambda_1 = 2 \\vartheta \\gamma`` and ``\\lambda_2 = (1 - \\vartheta) \\gamma``, so ``\\vartheta`` is the mixing weight of Equation (2) of [friedman2010](@cite), not the ratio ``\\lambda_2 / (\\lambda_1 + \\lambda_2)`` of [zouhastie2005](@cite). Equation (10) of [friedman2010](@cite) is the coordinate update. `strength` sets ``\\gamma``: [`MiddleOfPath`](@ref), the default, reads it off the data as [lai2018ktpt](@citet) do, and [`FixedStrength`](@ref) takes it from the caller. `theta = 0.99` is the value of [lai2018ktpt](@cite). It is almost the lasso, and it keeps the objective strictly convex.

The target adds no intercept and does not standardise the columns by default, as in the model of [lai2018ktpt](@cite). The glmnet package does both by default. An estimator that adds a column of ones names it with [`InterceptColumn`](@ref) through [`factory`](@ref), and the fit then leaves that coefficient out of the penalty. A variance estimator in `ve` divides each penalised column by its standard deviation before the fit, and the fit scales the coefficients back. A column whose standard deviation is zero has no scale, so it keeps its units. `ve` reads the observation weights of the target, so the scale and the fit weigh the rows alike.

The observation weights enter as the frequency weights of [`glm_weights`](@ref), the weights that [`LinearModel`](@ref) passes to `GLM`. Weights that are not `StatsBase.FrequencyWeights` then sum to their Kish effective sample size. Uniform weights give the unweighted fit at the same strength. The scale of the weights moves the fit under [`FixedStrength`](@ref) alone, because [`MiddleOfPath`](@ref) scales with them.

When the columns are almost collinear, for example the levels of consecutive periods in [`KernelTrendPattern`](@ref), the coordinate sweeps converge slowly. Each sweep therefore ends with an exact solve on the sign pattern that it leaves, see [`elastic_net_polish`](@ref). The sweeps must still find the sign pattern of the optimum. On the 4 × 5 window of the test fixture of [`KernelTrendPattern`](@ref) they find it after between 2000 and 5000 sweeps, and a cap of 1000 stops more than 0.5 from the optimum in one coefficient. When the sweeps stop at `tol` or at `iters` before that, the result is the last sweep, which is not the optimum.

The penalty depends on the units of each column, so the fit is not invariant to the basis of the design, and [`is_basis_invariant`](@ref) returns `false`. [`StepwiseRegression`](@ref) refuses the target, because its criteria count one degree of freedom for each coefficient.

# Mathematical definition

```math
\\begin{align}
\\mathbf{P} &= \\mathbf{X} \\mathbf{S}^{-1}\\,,\\\\
\\hat{\\boldsymbol{z}} &= \\underset{\\boldsymbol{z}}{\\arg\\min} \\left(\\boldsymbol{y} - \\mathbf{P} \\boldsymbol{z}\\right)^\\intercal \\mathbf{V} \\left(\\boldsymbol{y} - \\mathbf{P} \\boldsymbol{z}\\right) + \\gamma \\sum_{k \\in \\mathcal{K}} \\left( 2 \\vartheta \\lvert z_k \\rvert + (1 - \\vartheta) z_k^2 \\right)\\,,\\\\
\\hat{\\boldsymbol{b}} &= \\mathbf{S}^{-1} \\hat{\\boldsymbol{z}}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{X}``: Design matrix of the caller, one row per observation and one column per regressor.
  - ``\\mathbf{S}``: Diagonal matrix of the standard deviations of the columns of ``\\mathbf{X}`` under `ve`, with one in place of the intercept column and of a column of zero standard deviation. It is the identity when `ve` is `nothing`.
  - $(math_dict[:P_enet])
  - $(math_dict[:y_enet])
  - $(math_dict[:z_enet])
  - $(math_dict[:V_enet]) Its diagonal holds the frequency weights of [`glm_weights`](@ref).
  - $(math_dict[:K_enet])
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - ``\\hat{\\boldsymbol{b}}``: Coefficients of the fit on the columns of ``\\mathbf{X}``, which `StatsAPI.coef` returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ElasticNet(;
        theta::Real = 0.99,
        strength::AbstractPenaltyStrength = MiddleOfPath(),
        ve::Option{<:AbstractVarianceEstimator} = nothing,
        iters::Integer = 10_000,
        tol::Real = 1e-10,
        w::Option{<:ObsWeights} = nothing,
        ic::Option{<:Integer} = nothing
    ) -> ElasticNet

Keywords correspond to the struct's fields. The constructor passes `w` to `ve` through [`factory`](@ref).

## Validation

  - `0 <= theta <= 1`. A `DomainError` is thrown otherwise.
  - `theta > 0` when `strength` is a [`MiddleOfPath`](@ref). A `DomainError` is thrown otherwise, because a ridge alone has no strength above which every coefficient is zero.
  - `iters >= 1`. A `DomainError` is thrown otherwise.
  - `tol > 0`. A `DomainError` is thrown otherwise.
  - If `w` is a vector, `!isempty(w)`. An `IsEmptyError` is thrown otherwise.
  - If `ic` is not `nothing`, `ic >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ElasticNet()
ElasticNet
     theta ┼ Float64: 0.99
  strength ┼ MiddleOfPath
           │   ratio ┴ Float64: 0.001
        ve ┼ nothing
     iters ┼ Int64: 10000
       tol ┼ Float64: 1.0e-10
         w ┼ nothing
        ic ┴ nothing

julia> X = [1.0 0.5; 1.0 -0.2; 1.0 0.9; 1.0 0.1];

julia> y = [1.2, 0.4, 1.9, 0.6];

julia> tgt = factory(ElasticNet(; strength = FixedStrength(; gamma = 1e6)), InterceptColumn(1));

julia> m = PortfolioOptimisers.StatsAPI.fit(tgt, X, y);

julia> PortfolioOptimisers.StatsAPI.coef(m) ≈ [sum(y) / 4, 0.0]
true
```

# Related

  - [`AbstractPenalisedRegressionTarget`](@ref)
  - [`AbstractPenaltyStrength`](@ref)
  - [`MiddleOfPath`](@ref)
  - [`FixedStrength`](@ref)
  - [`InterceptColumn`](@ref)
  - [`LinearModel`](@ref)
  - [`KernelTrendPattern`](@ref)
  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)

# References

  - $(ref_dict[:friedman2010]) Section 2.5, Equations (1), (2) and (10).
  - $(ref_dict[:zouhastie2005]) Equation (3).
  - $(ref_dict[:lai2018ktpt]) Equation (15).
"""
@concrete struct ElasticNet <: AbstractPenalisedRegressionTarget
    """
    The share of the ``L_1`` penalty, `1` for the lasso and `0` for the ridge.
    """
    theta
    """
    The rule that sets the regularisation strength.
    """
    strength
    """
    $(field_dict[:ve]) It standardises each penalised column, or `nothing` to fit the columns as they are.
    """
    ve
    """
    Maximum number of coordinate-descent sweeps. Each sweep ends with the exact solve on its sign pattern.
    """
    iters
    """
    Tolerance on the largest change of a coefficient over one sweep. The sweeps stop below it when no exact solve has given the optimum.
    """
    tol
    """
    $(field_dict[:ow])
    """
    w
    """
    The index of the column of ones that the estimator names with [`InterceptColumn`](@ref), or `nothing`. The fit does not penalise or standardise that column.
    """
    ic
    function ElasticNet(theta::Real, strength::AbstractPenaltyStrength,
                        ve::Option{<:AbstractVarianceEstimator}, iters::Integer, tol::Real,
                        w::Option{<:ObsWeights}, ic::Option{<:Integer})
        @argcheck(zero(theta) <= theta <= one(theta),
                  DomainError(theta, "theta must be in [0, 1]"))
        @argcheck(!(iszero(theta) && isa(strength, MiddleOfPath)),
                  DomainError(theta,
                              "theta = 0 is a ridge alone, which has no strength above which every coefficient is zero, so MiddleOfPath has no path to read. State theta > 0, or a FixedStrength"))
        @argcheck(iters >= 1, DomainError(iters, "iters must be at least 1"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be positive"))
        if isa(w, AbstractVector)
            @argcheck(!isempty(w), IsEmptyError)
        end
        if !isnothing(ic)
            @argcheck(ic >= 1, DomainError(ic, "ic must be at least 1"))
        end
        return new{typeof(theta), typeof(strength), typeof(ve), typeof(iters), typeof(tol),
                   typeof(w), typeof(ic)}(theta, strength, ve, iters, tol, w, ic)
    end
end
function ElasticNet(; theta::Real = 0.99,
                    strength::AbstractPenaltyStrength = MiddleOfPath(),
                    ve::Option{<:AbstractVarianceEstimator} = nothing,
                    iters::Integer = 10_000, tol::Real = 1e-10,
                    w::Option{<:ObsWeights} = nothing,
                    ic::Option{<:Integer} = nothing)::ElasticNet
    return ElasticNet(theta, strength, factory(ve, w), iters, tol, w, ic)
end
"""
    factory(tgt::ElasticNet, w::ObsWeights) -> ElasticNet

Return a new [`ElasticNet`](@ref) whose fit weighs the observations by `w`.

# Algorithm

 1. Build a new [`ElasticNet`](@ref) with `w` in place of `tgt.w` and every other field unchanged. Its keyword constructor passes `w` to `tgt.ve` through [`factory`](@ref), so the standardisation reads the same weights.

# Arguments

  - `tgt`: The regression target.
  - $(arg_dict[:ow])

# Returns

  - `tgt::ElasticNet`: A new regression target that holds the weights.

# Related

  - [`ElasticNet`](@ref)
  - [`factory`](@ref)
  - [`regression_target_weights(::ElasticNet)`](@ref)
"""
function factory(tgt::ElasticNet, w::ObsWeights)::ElasticNet
    return ElasticNet(; theta = tgt.theta, strength = tgt.strength, ve = tgt.ve,
                      iters = tgt.iters, tol = tgt.tol, w = w, ic = tgt.ic)
end
"""
    factory(tgt::ElasticNet, ic::InterceptColumn) -> ElasticNet

Return a new [`ElasticNet`](@ref) whose fit leaves the column of ones that an estimator names out of the penalty.

# Algorithm

 1. Build a new [`ElasticNet`](@ref) with `ic.k` in place of `tgt.ic` and every other field unchanged.

# Arguments

  - `tgt`: The regression target.
  - `ic`: The intercept column.

# Returns

  - `tgt::ElasticNet`: A new regression target that does not penalise or standardise column `ic.k`.

# Related

  - [`ElasticNet`](@ref)
  - [`InterceptColumn`](@ref)
  - [`factory(::AbstractRegressionTarget, ::InterceptColumn)`](@ref)
"""
function factory(tgt::ElasticNet, ic::InterceptColumn)::ElasticNet
    return ElasticNet(; theta = tgt.theta, strength = tgt.strength, ve = tgt.ve,
                      iters = tgt.iters, tol = tgt.tol, w = tgt.w, ic = ic.k)
end
"""
    regression_target_weights(tgt::ElasticNet)

Return the observation weights that an [`ElasticNet`](@ref) carries into its fit, or `nothing`.

The method returns `tgt.w` as it is, so it does not resolve a [`DynamicAbstractWeights`](@ref) against a design.

# Arguments

  - `tgt`: The regression target.

# Returns

  - `w::Option{ObsWeights}`: `tgt.w`.

# Related

  - [`ElasticNet`](@ref)
  - [`regression_target_weights`](@ref)
  - [`factory(::ElasticNet, ::ObsWeights)`](@ref)
"""
function regression_target_weights(tgt::ElasticNet)
    return tgt.w
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the coefficients of a fitted [`ElasticNet`](@ref) and the strength at which the fit ran.

[`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref) returns it. `StatsAPI.coef` returns `b`, and `StatsAPI.predict` multiplies a design by `b`.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ElasticNet`](@ref)
  - [`StatsAPI.coef(::ElasticNetFit)`](@ref)
  - [`StatsAPI.predict(::ElasticNetFit, ::MatNum)`](@ref)
"""
@concrete struct ElasticNetFit <: AbstractResult
    """
    The coefficients on the columns of the design of the caller, one per column.
    """
    b
    """
    The regularisation strength ``\\gamma`` that the strength rule gave, on the standardised columns when the target has a `ve`.
    """
    gamma
end
"""
    StatsAPI.coef(m::ElasticNetFit) -> VecNum

Return the coefficients of a fitted [`ElasticNet`](@ref), one per column of the design.

# Arguments

  - `m`: The fitted model.

# Returns

  - `b::VecNum`: `m.b`.

# Related

  - [`ElasticNetFit`](@ref)
  - [`StatsAPI.predict(::ElasticNetFit, ::MatNum)`](@ref)
"""
function StatsAPI.coef(m::ElasticNetFit)
    return m.b
end
"""
    StatsAPI.predict(m::ElasticNetFit, X::MatNum) -> VecNum

Return the fitted response of a fitted [`ElasticNet`](@ref) on a design.

# Arguments

  - `m`: The fitted model.
  - `X`: The design matrix, `observations × regressors`, with the columns of the design of the fit.

# Returns

  - `yhat::VecNum`: `X * m.b`, one entry per row of `X`.

# Related

  - [`ElasticNetFit`](@ref)
  - [`StatsAPI.coef(::ElasticNetFit)`](@ref)
"""
function StatsAPI.predict(m::ElasticNetFit, X::MatNum)
    return X * m.b
end
"""
    elastic_net_design(tgt::ElasticNet, X::MatNum) -> Tuple

Return the standardised design of an [`ElasticNet`](@ref), the design with each row weighted, and the scale of the columns.

# Algorithm

 1. When `tgt.ve` is `nothing`, set the scale `s` to `nothing`. Otherwise take the standard deviation of each column of `X` under `tgt.ve`, and put one in place of a zero and in place of the intercept column `tgt.ic`. That is `s`.
 2. Divide each column of `X` by its entry of `s`, which gives `P`. When `s` is `nothing`, `P` is `X`.
 3. Resolve `tgt.w` against `X` with [`get_observation_weights`](@ref). When it is `nothing`, `Q` is `P`. Otherwise scale it with [`glm_weights`](@ref), and multiply each row of `P` by its weight, which gives `Q`.

# Arguments

  - `tgt`: The regression target.
  - `X`: The design matrix `observations × regressors`.

# Returns

  - `(P, Q, s)::Tuple`: The standardised design ``\\mathbf{P}``, the weighted design ``\\mathbf{V} \\mathbf{P}``, and the scale of the columns or `nothing`.

# Related

  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)
  - [`glm_weights`](@ref)
"""
function elastic_net_design(tgt::ElasticNet, X::MatNum)
    s = if isnothing(tgt.ve)
        nothing
    else
        sd = map(v -> iszero(v) ? one(v) : v, vec(Statistics.std(tgt.ve, X; dims = 1)))
        if !isnothing(tgt.ic)
            sd[tgt.ic] = one(eltype(sd))
        end
        sd
    end
    P = isnothing(s) ? X : X ./ transpose(s)
    w = get_observation_weights(tgt.w, X)
    Q = isnothing(w) ? P : glm_weights(w) .* P
    return P, Q, s
end
"""
    StatsAPI.fit(tgt::ElasticNet, X::MatNum, y::VecNum) -> ElasticNetFit

Fit an [`ElasticNet`](@ref) to the design `X` and the response `y` by cyclic coordinate descent.

[`ElasticNet`](@ref) states the objective. The fit solves it at one strength, which the strength rule gives. A design with few columns does not need the rest of the path as warm starts.

# Algorithm

 1. Build the standardised design `P`, the weighted design `Q` and the scale `s` with [`elastic_net_design`](@ref).
 2. Mark the penalised columns in `pen`: every column but `tgt.ic`. Compute `nrm2`, the weighted squared norm of each column.
 3. Start from the coefficients `z` at zero and the residual `r` equal to `y`. With an intercept column, fit that column alone first, and take its residual as `r`.
 4. Compute `g`, the product of each penalised column of `Q` with `r`, and the strength `lam` with [`penalty_strength`](@ref).
 5. When every entry of `g` is zero, the start point is the optimum. Otherwise compute the threshold `a` and the ridge term `ridge` of each column, both zero at the intercept column, and run [`elastic_net_sweeps`](@ref) from the start point.
 6. Divide the coefficients by `s`, and return them with `lam` in an [`ElasticNetFit`](@ref).

# Arguments

  - `tgt`: The regression target.
  - `X`: The design matrix `observations × regressors`.
  - `y`: The response vector.

# Returns

  - `m::ElasticNetFit`: The fitted model.

# Related

  - [`ElasticNet`](@ref)
  - [`ElasticNetFit`](@ref)
  - [`elastic_net_design`](@ref)
  - [`elastic_net_sweeps`](@ref)
"""
function StatsAPI.fit(tgt::ElasticNet, X::MatNum, y::VecNum)
    P, Q, s = elastic_net_design(tgt, X)
    n = size(P, 2)
    pen = trues(n)
    if !isnothing(tgt.ic)
        pen[tgt.ic] = false
    end
    nrm2 = vec(sum(P .* Q; dims = 1))
    z = zeros(float_if_integer(promote_type(eltype(Q), eltype(y))), n)
    r = copyto!(similar(y, eltype(z)), y)
    if !isnothing(tgt.ic)
        z[tgt.ic] = LinearAlgebra.dot(view(Q, :, tgt.ic), y) / nrm2[tgt.ic]
        r .-= z[tgt.ic] .* view(P, :, tgt.ic)
    end
    g = [LinearAlgebra.dot(view(Q, :, k), r) for k in findall(pen)]
    lam = penalty_strength(tgt.strength, g, tgt.theta)
    if !all(iszero, g)
        a = (lam * tgt.theta) .* pen
        ridge = (lam * (one(tgt.theta) - tgt.theta)) .* pen
        z = elastic_net_sweeps(tgt, P, Q, y, z, r, nrm2, a, ridge, pen)
    end
    return ElasticNetFit(isnothing(s) ? z : z ./ s, lam)
end
"""
    elastic_net_sweeps(tgt::ElasticNet, P::MatNum, Q::MatNum, y::VecNum, z::VecNum,
                       r::VecNum, nrm2::VecNum, a::VecNum, ridge::VecNum,
                       pen::AbstractVector{Bool}) -> VecNum

Return the coefficients of an [`ElasticNet`](@ref) by cyclic coordinate descent from `z`, with an exact solve at the end of each sweep.

# Mathematical definition

The minimiser of the objective of [`ElasticNet`](@ref) over one coefficient, with the others fixed, is

```math
\\begin{align}
z_k &= \\frac{S\\left(\\mathbf{P}_k^\\intercal \\mathbf{V} \\boldsymbol{r}_k,\\ a_k\\right)}{\\mathbf{P}_k^\\intercal \\mathbf{V} \\mathbf{P}_k + d_k}\\,,\\quad S(v, a) = \\operatorname{sign}(v) \\max(\\lvert v \\rvert - a, 0)\\,,\\\\
a_k &= \\gamma \\vartheta\\, \\mathbb{1}[k \\in \\mathcal{K}]\\,,\\quad d_k = \\gamma (1 - \\vartheta)\\, \\mathbb{1}[k \\in \\mathcal{K}]\\,.
\\end{align}
```

Where:

  - $(math_dict[:z_enet])
  - $(math_dict[:P_enet])
  - $(math_dict[:V_enet])
  - ``\\boldsymbol{r}_k``: Residual of the response ``\\boldsymbol{y}`` without the term of column ``k``.
  - ``S``: Soft threshold.
  - ``a_k``, ``d_k``: Threshold and ridge term of column ``k``, both zero at the intercept column.
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - $(math_dict[:K_enet])

At the intercept column the update is the least-squares coefficient of that column on the residual.

# Algorithm

 1. Take `den`, the denominator of each coordinate update.
 2. Sweep the columns in order. For each column `k`, compute `rho`, the product of the column of `Q` with the residual without its own term, and apply the coordinate update, which gives `znew`. Update `r`, and record `delta`, the largest change of a coefficient in the sweep.
 3. Try the exact solve of [`elastic_net_polish`](@ref) on the sign pattern of `z`. When it gives the optimum, return it.
 4. Repeat steps 2 and 3 until `delta` is below `tgt.tol` or `tgt.iters` sweeps have run, and then return `z`.

# Arguments

  - `tgt`: The regression target, which gives `iters` and `tol`.
  - `P`: The standardised design, `observations × columns`.
  - `Q`: The weighted design, ``\\mathbf{V} \\mathbf{P}``.
  - `y`: The response.
  - `z`: The coefficients of the start point. The function changes them.
  - `r`: The residual of the start point. The function changes it.
  - `nrm2`: The weighted squared norm of each column.
  - `a`: The threshold of each column.
  - `ridge`: The ridge term of each column.
  - `pen`: Whether each column is penalised.

# Returns

  - `z::VecNum`: The coefficients on the columns of `P`.

# Related

  - [`StatsAPI.fit(::ElasticNet, ::MatNum, ::VecNum)`](@ref)
  - [`elastic_net_polish`](@ref)
"""
function elastic_net_sweeps(tgt::ElasticNet, P::MatNum, Q::MatNum, y::VecNum, z::VecNum,
                            r::VecNum, nrm2::VecNum, a::VecNum, ridge::VecNum,
                            pen::AbstractVector{Bool})
    den = nrm2 .+ ridge
    for _ in 1:(tgt.iters)
        delta = zero(eltype(z))
        for k in eachindex(z)
            rho = LinearAlgebra.dot(view(Q, :, k), r) + nrm2[k] * z[k]
            znew = sign(rho) * max(abs(rho) - a[k], zero(a[k])) / den[k]
            if znew != z[k]
                r .-= (znew - z[k]) .* view(P, :, k)
                delta = max(delta, abs(znew - z[k]))
                z[k] = znew
            end
        end
        zp = elastic_net_polish(P, Q, y, z, a, ridge, pen)
        if !isnothing(zp)
            return zp
        end
        delta < tgt.tol && break
    end
    return z
end
"""
    elastic_net_polish(P::MatNum, Q::MatNum, y::VecNum, z::VecNum, a::VecNum,
                       ridge::VecNum, pen::AbstractVector{Bool}) -> Option{VecNum}

Returns the exact elastic-net optimum on the sign pattern of `z` when that pattern is the pattern of the optimum, and `nothing` otherwise.

A point that meets the optimality conditions below minimises the objective of [`ElasticNet`](@ref), because the objective is convex. For `theta < 1` the objective is strictly convex in the penalised coefficients, and the point is its only minimiser. At `theta = 1` the objective is the lasso, and it can have more than one minimiser. Then the equations of the active columns are singular when the pattern holds more columns than there are observations, and the function returns `nothing` for such a pattern. The intercept column is always active, and its sign is free.

# Mathematical definition

With ``\\mathcal{A}`` the active columns, the optimum on the sign pattern ``\\boldsymbol{s}`` solves

```math
\\begin{align}
\\left(\\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\mathbf{P}_{\\mathcal{A}} + \\gamma (1 - \\vartheta) \\mathbf{D}_{\\mathcal{A}}\\right) \\boldsymbol{z}_{\\mathcal{A}} &= \\mathbf{P}_{\\mathcal{A}}^\\intercal \\mathbf{V} \\boldsymbol{y} - \\gamma \\vartheta\\, \\boldsymbol{s}\\,,
\\end{align}
```

and it is the optimum of the objective when

```math
\\begin{align}
\\operatorname{sign}(z_k) &= s_k\\,,\\quad k \\in \\mathcal{A} \\cap \\mathcal{K}\\,,\\\\
\\lvert \\mathbf{P}_k^\\intercal \\mathbf{V} (\\boldsymbol{y} - \\mathbf{P}_{\\mathcal{A}} \\boldsymbol{z}_{\\mathcal{A}}) \\rvert &\\leq \\gamma \\vartheta\\,,\\quad k \\notin \\mathcal{A}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{A}``: Active set, the columns whose coefficient in ``\\boldsymbol{z}`` is nonzero, and the intercept column.
  - ``\\mathbf{P}_{\\mathcal{A}}``, ``\\boldsymbol{z}_{\\mathcal{A}}``: Columns of the regressors and coefficients of the active set.
  - ``\\mathbf{D}_{\\mathcal{A}}``: Diagonal matrix with one for each penalised column of the active set and zero for the intercept column.
  - ``\\boldsymbol{s}``: Signs of the coefficients of the active set in ``\\boldsymbol{z}``, zero for the intercept column.
  - $(math_dict[:z_enet])
  - $(math_dict[:P_enet])
  - $(math_dict[:V_enet])
  - $(math_dict[:y_enet])
  - $(math_dict[:gamma_enet])
  - $(math_dict[:theta_enet])
  - $(math_dict[:K_enet])

# Algorithm

 1. Find `active`, the nonzero coefficients of `z` and the column that `pen` does not penalise, and their indices `A`. When `A` is empty, return `nothing`.
 2. Take `s`, the signs of the coefficients of `A`, zero at the intercept column. Factorise the matrix of the equations of the active columns, which gives `F`. When the matrix is singular, return `nothing`.
 3. Solve the equations with `F`, which gives `zA`. When a penalised coefficient of `zA` does not keep its sign in `s`, return `nothing`.
 4. Write `zA` into `zp`, zero elsewhere, and compute `corr`, the product of each column of `Q` with the residual.
 5. When an inactive column has `abs(corr[k]) > a[k]`, return `nothing`. Otherwise return `zp`.

# Arguments

  - `P`: The standardised design, `observations × columns`.
  - `Q`: The weighted design, ``\\mathbf{V} \\mathbf{P}``.
  - `y`: The response.
  - `z`: The coefficients whose sign pattern the function tries.
  - `a`: The threshold of each column, ``\\gamma \\vartheta`` or zero at the intercept column.
  - `ridge`: The ridge term of each column, ``\\gamma (1 - \\vartheta)`` or zero at the intercept column.
  - `pen`: Whether each column is penalised.

# Returns

  - `z'::Option{VecNum}`: The optimum, or `nothing`.

# Related

  - [`elastic_net_sweeps`](@ref)
  - [`ElasticNet`](@ref)
"""
function elastic_net_polish(P::MatNum, Q::MatNum, y::VecNum, z::VecNum, a::VecNum,
                            ridge::VecNum, pen::AbstractVector{Bool})
    active = .!iszero.(z) .| .!pen
    A = findall(active)
    if isempty(A)
        return nothing
    end
    PA = view(P, :, A)
    QA = view(Q, :, A)
    s = sign.(view(z, A)) .* view(pen, A)
    # At `theta = 1` the matrix is singular when the pattern holds more columns than
    # observations, and the pattern is then refused rather than solved.
    F = LinearAlgebra.lu(QA' * PA + LinearAlgebra.Diagonal(view(ridge, A)); check = false)
    if !LinearAlgebra.issuccess(F)
        return nothing
    end
    zA = F \ (QA' * y .- view(a, A) .* s)
    if any(j -> pen[A[j]] && zA[j] * s[j] <= zero(zA[j]), eachindex(zA))
        return nothing
    end
    zp = zero(z)
    zp[A] .= zA
    corr = Q' * (y .- P * zp)
    for k in eachindex(zp)
        if !active[k] && abs(corr[k]) > a[k]
            return nothing
        end
    end
    return zp
end
export ElasticNet, MiddleOfPath, FixedStrength
public AbstractPenaltyStrength, penalty_strength
