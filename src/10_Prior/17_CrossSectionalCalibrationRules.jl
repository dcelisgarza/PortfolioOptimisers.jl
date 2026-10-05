"""
$(DocStringExtensions.TYPEDEF)

Chooses the spanned coefficients whose distance from the factor mean measures the error of a Return Forecast.

[`PrecisionBlend`](@ref) holds a member of this family in its field `err`. The blend divides the sampling error of the factor mean by the sum of that error and the error of the spanned part `g` of the forecast. It measures the second error by the distance of `g` from the factor mean, and this family chooses the rows of `g` that it reads: the latest row alone, or the rows of the Return Forecast history.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractForecastErrorAlgorithm` and implement the following method:

## `spanned_forecast_sample`

  - `spanned_forecast_sample(err::MyForecastError, cs::NamedTuple) -> AbstractMatrix`: Returns the spanned coefficients that the blend reads, one column per row of the forecast and one row per estimated factor.

### Arguments

  - `err`: The member of the family.
  - $(arg_dict[:cal_ctx_cs])

### Returns

  - `G::AbstractMatrix`: The spanned coefficients, `estimated factors × rows`. Each column is finite.

A member whose method reads `cs.hist` also adds a method of [`reads_forecast_history`](@ref) that answers `true`, so the prior makes the history.

# Related

  - [`PrecisionBlend`](@ref)
  - [`CurrentForecastError`](@ref)
  - [`ForecastHistoryError`](@ref)
  - [`spanned_forecast_sample`](@ref)
"""
abstract type AbstractForecastErrorAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the error of the spanned part of a Return Forecast from its latest row alone.

[`PrecisionBlend`](@ref) with this member reads the spanned coefficients `g` that the prior blends, and nothing more, so it needs no Return Forecast history. With no Return Forecast Estimator, `g` is zero.

# Examples

```jldoctest
julia> PortfolioOptimisers.reads_forecast_history(CurrentForecastError())
false
```

# Related

  - [`AbstractForecastErrorAlgorithm`](@ref)
  - [`ForecastHistoryError`](@ref)
  - [`PrecisionBlend`](@ref)
"""
struct CurrentForecastError <: AbstractForecastErrorAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the error of the spanned part of a Return Forecast from the rows of its history.

[`PrecisionBlend`](@ref) with this member splits each row of the Return Forecast history against the Factor Exposures of that row, and averages the distance of each spanned part from the factor mean. The history ends at the latest observation, and the member reads the rows before it. A forecast that has no row before the latest gives the latest row alone, which is the reading of [`CurrentForecastError`](@ref).

The member reads the history, so the prior makes it, and a prior whose `rfe` has no history refuses the member at construction.

# Examples

```jldoctest
julia> PortfolioOptimisers.reads_forecast_history(ForecastHistoryError())
true
```

# Related

  - [`AbstractForecastErrorAlgorithm`](@ref)
  - [`CurrentForecastError`](@ref)
  - [`PrecisionBlend`](@ref)
  - [`forecast_history`](@ref)
"""
struct ForecastHistoryError <: AbstractForecastErrorAlgorithm end
function reads_forecast_history(::ForecastHistoryError)
    return true
end
"""
    spanned_forecast_sample(err::CurrentForecastError, cs::NamedTuple) -> MatNum
    spanned_forecast_sample(err::ForecastHistoryError, cs::NamedTuple) -> MatNum

Return the spanned coefficients of a Return Forecast that [`PrecisionBlend`](@ref) reads, one column per row.

# Algorithm

 1. [`CurrentForecastError`](@ref): take `cs.g` as the one column, or a zero column when the prior states no Return Forecast Estimator.
 2. [`ForecastHistoryError`](@ref): split the Return Forecast history with [`cross_sectional_split_history`](@ref), and keep the columns before the latest row whose coefficients are all finite. When no column is left, take the column of [`CurrentForecastError`](@ref).

# Arguments

  - `err`: The member of [`AbstractForecastErrorAlgorithm`](@ref).
  - $(arg_dict[:cal_ctx_cs])

# Validation

  - The rules of [`cross_sectional_split_history`](@ref), for [`ForecastHistoryError`](@ref).

# Returns

  - `G::MatNum`: The spanned coefficients, `estimated factors × rows`.

# Related

  - [`AbstractForecastErrorAlgorithm`](@ref)
  - [`PrecisionBlend`](@ref)
  - [`cross_sectional_split_history`](@ref)
"""
function spanned_forecast_sample(::CurrentForecastError, cs::NamedTuple)
    csr = cs.csfm.csr
    g = cs.g
    G = isnothing(g) ? zeros(eltype(csr.f), size(csr.f, 2)) : g
    return reshape(G, :, 1)
end
function spanned_forecast_sample(::ForecastHistoryError, cs::NamedTuple)
    G = cross_sectional_split_history(cs).g
    cols = [t for t in 1:(size(G, 2) - 1) if all(isfinite, view(G, :, t))]
    if isempty(cols)
        return spanned_forecast_sample(CurrentForecastError(), cs)
    end
    return G[:, cols]
end
"""
    cross_sectional_reduced_history(fcb::Nothing, Ms::Arr3Num) -> Arr3Num
    cross_sectional_reduced_history(fcb::FactorFamilyBasis, Ms::Arr3Num) -> Arr3Num

Return the exposure history of the block of a [`CrossSectionalFactorPrior`](@ref) on the reduced factor axis.

The block carries the raw exposure history `Ms` and the Factor Family Basis `fcb` of the same rows. The latest loadings `L` are the reduced exposures of the last row, so a split of a past row reads the reduced exposures of that row, and a split of the last row gives the coefficients that the prior blends. A prior that constrains no Factor Family carries no basis, and its raw history is the reduced one.

# Arguments

  - `fcb`: The Factor Family Basis of the block, or `nothing`.
  - `Ms`: The exposure history of the block on the raw axis, `observations × assets × factors`.

# Validation

  - The rules of [`reduce_exposures`](@ref), when `fcb` is a [`FactorFamilyBasis`](@ref).

# Returns

  - `Ms::Arr3Num`: The exposure history on the reduced axis, `observations × assets × reduced factors`.

# Related

  - [`cross_sectional_split_history`](@ref)
  - [`reduce_exposures`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cross_sectional_reduced_history(::Nothing, Ms::Arr3Num)
    return Ms
end
function cross_sectional_reduced_history(fcb::FactorFamilyBasis, Ms::Arr3Num)
    return reduce_exposures(fcb, Ms)
end
"""
    cross_sectional_split_history(cs::NamedTuple) -> NamedTuple

Split each row of the Return Forecast history of a [`CrossSectionalFactorPrior`](@ref) against the Factor Exposures of that row.

The prior splits the latest forecast against the latest exposures, under the regression weights of the latest fit. This function makes the same split at each row of the history, with the exposures and the weights of that row. So the last row gives the `g` and the `ap` that the prior holds, and each earlier row gives the parts that the forecast stated at that row.

# Algorithm

 1. Read the reduced exposure history with [`cross_sectional_reduced_history`](@ref), and keep the columns of the estimated factors, the columns of `cs.csfm.csr.f`.
 2. Fill `g`, `estimated factors × rows`, and `ap`, `rows × assets`, with `NaN`.
 3. At each row of `cs.hist` that holds a finite forecast, split the row with [`cross_sectional_alpha_split`](@ref) under `cs.cre`, against the exposures of that row and the regression weights `cs.csfm.rw` of that row, and write the two parts.

# Arguments

  - $(arg_dict[:cal_ctx_cs])

# Validation

  - `cs.hist` is given. Raises an [`IsNothingError`](@ref).
  - `cs.hist`, the exposure history and the weight history of the block have one row per observation of the block. Raises a `DimensionMismatch`.
  - The rules of [`cross_sectional_alpha_split`](@ref).

# Returns

  - `g::MatNum`: The spanned coefficients of each row, `estimated factors × rows`. A row with no finite forecast carries `NaN`.
  - `ap::MatNum`: The orthogonal part of each row, `rows × assets`. A row with no finite forecast carries `NaN`.

# Related

  - [`cross_sectional_alpha_split`](@ref)
  - [`cross_sectional_reduced_history`](@ref)
  - [`spanned_forecast_sample`](@ref)
  - [`orthogonal_forecast_pairs`](@ref)
  - [`forecast_history`](@ref)
"""
function cross_sectional_split_history(cs::NamedTuple)
    hist = cs.hist
    @argcheck(!isnothing(hist),
              IsNothingError("a rule that reads the Return Forecast history reads it from `cs.hist`, and `cs.hist` is nothing. The prior makes the history when a slot answers true to `reads_forecast_history`, so add a method of `reads_forecast_history` for the rule that answers true."))
    H::MatNum = hist
    csfm = cs.csfm
    Ke = size(csfm.csr.f, 2)
    Lh::Arr3Num = cross_sectional_reduced_history(csfm.fcb, csfm.Ms)
    rw::MatNum = csfm.rw
    Tb, N = size(H)
    @argcheck(size(Lh, 1) == Tb == size(rw, 1),
              DimensionMismatch("the Return Forecast history has $Tb rows, the exposure history $(size(Lh, 1)) and the regression weight history $(size(rw, 1)). The three must have one row per observation of the block."))
    Tf = promote_type(real(eltype(H)), real(eltype(Lh)), real(eltype(rw)))
    g = fill(Tf(NaN), Ke, Tb)
    ap = fill(Tf(NaN), Tb, N)
    for t in 1:Tb
        if !any(isfinite, view(H, t, :))
            continue
        end
        sp = cross_sectional_alpha_split(cs.cre, view(H, t, :), view(Lh, t, :, 1:Ke),
                                         view(rw, t, :))
        g[:, t] = sp.g
        ap[t, :] = sp.ap
    end
    return (; g = g, ap = ap)
end
"""
    cross_sectional_context(ctx) -> NamedTuple

Return the fit of a Cross-Sectional Factor Prior that a rule of its Calibration Slots reads.

A rule of [`AbstractSpannedShrinkageCalibrationAlgorithm`](@ref) or of [`AbstractOrthogonalForecastScaleCalibrationAlgorithm`](@ref) reads `ctx.cs`, which only the prior states. A caller who runs such a rule outside the prior builds that context, and this function refuses a context that has none.

# Arguments

  - `ctx`: The site's [`CalibrationContext`](@ref).

# Validation

  - `ctx.cs` is given. Raises an [`IsNothingError`](@ref).

# Returns

  - `cs::NamedTuple`: The fit of the prior, as [`CalibrationContext`](@ref) states it.

# Related

  - [`CalibrationContext`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_context(ctx)
    cs = ctx.cs
    @argcheck(!isnothing(cs),
              IsNothingError("a rule of the Spanned Shrinkage or of the Orthogonal Forecast Scale reads the fit of a Cross-Sectional Factor Prior from `ctx.cs`, and `ctx.cs` is nothing. Put the rule in the `lambda` or the `c` slot of a CrossSectionalFactorPrior, or build the context that the prior builds."))
    return cs::NamedTuple
end
"""
    spanned_shrinkage_moments(pr::AbstractPriorResult, cs::NamedTuple, G::AbstractMatrix)
        -> NamedTuple

Return the factor mean, the spanned coefficients and the factor covariance in the form that a Spanned Shrinkage rule reads.

The blend acts on the estimated factors alone, so the function keeps the leading entries of the factor moments, one per column of `cs.csfm.csr.f`. The factor covariance of those factors can be singular. A market factor beside a full one-hot industry block is the common case: the market exposure is the sum of the industry exposures, so the fitted factor returns, their mean and the spanned coefficients all lie in a subspace, and the covariance is zero across it. An Empty Factor carries a zero variance and no return, which is the same case in one direction. So the function reads the covariance through its pseudo-inverse over the eigenvalues it keeps, and `K` is the number of those eigenvalues, the dimension in which the factor mean has a sampling error.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{f} &= \\sum_{i} \\ell_{i} \\boldsymbol{u}_{i} \\boldsymbol{u}_{i}^{\\intercal}\\,, \\\\
\\mathbf{W} &= \\left[ \\boldsymbol{u}_{i} / \\sqrt{\\ell_{i}} \\right]_{\\ell_{i} > \\tau}\\,, \\\\
\\tau &= m \\, \\epsilon\\left(\\max_{i} \\ell_{i}\\right)\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{\\Sigma}}_{f}``: Factor covariance of the nested factor prior, over the estimated factors.
  - ``\\ell_{i}``, ``\\boldsymbol{u}_{i}``: Eigenvalue ``i`` of the factor covariance and its eigenvector.
  - ``\\mathbf{W}``: The kept eigenvectors, each divided by the square root of its eigenvalue, so ``\\mathbf{W} \\mathbf{W}^{\\intercal}`` is the pseudo-inverse of the factor covariance.
  - ``\\tau``: Threshold below which an eigenvalue is taken as zero.
  - ``m``: Number of estimated factors.
  - ``\\epsilon(x)``: Spacing of the floating-point numbers at ``x``.

# Arguments

  - `pr`: The factor moments of the nested factor prior, on the reduced factor axis.
  - $(arg_dict[:cal_ctx_cs])
  - `G`: The spanned coefficients, `estimated factors × rows`.

# Returns

  - `mu::VecNum`: The factor mean of the estimated factors.
  - `G::AbstractMatrix`: The spanned coefficients, unchanged.
  - `W::MatNum`: The whitening matrix, `estimated factors × K`.
  - `ev::VecNum`: The kept eigenvalues.
  - `K::Int`: Number of kept eigenvalues.

# Related

  - [`PrecisionBlend`](@ref)
  - [`SteinShrinkage`](@ref)
  - [`spanned_forecast_sample`](@ref)
"""
function spanned_shrinkage_moments(pr::AbstractPriorResult, cs::NamedTuple,
                                   G::AbstractMatrix)
    Ke = size(cs.csfm.csr.f, 2)
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(pr.sigma[1:Ke, 1:Ke]))
    lmax = maximum(E.values)
    keep = E.values .> Ke * eps(lmax)
    ev = E.values[keep]
    return (; mu = pr.mu[1:Ke], G = G, W = E.vectors[:, keep] ./ transpose(sqrt.(ev)),
            ev = ev, K = count(keep))
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the factor mean towards the spanned part of a Return Forecast by the precision of each.

The fitted factor mean ``\\hat{\\boldsymbol{\\mu}}_{f}`` and the spanned part ``\\boldsymbol{g}`` are two estimates of the expected factor returns. The rule gives the factor mean a weight equal to the error of ``\\boldsymbol{g}`` over the sum of the two errors, so the estimate with the smaller error gets the larger weight. The sampling error of the factor mean, in the metric of the pseudo-inverse of the factor covariance, is ``K / T_{e}``. The rule measures the error of ``\\boldsymbol{g}`` by its distance from the factor mean less that sampling error. That distance holds both errors, so the difference estimates the error of ``\\boldsymbol{g}`` alone, and the rule clips it at zero.

It is the default of the `lambda` slot of [`CrossSectionalFactorPrior`](@ref). A prior that states no Return Forecast Estimator has ``\\boldsymbol{g} = \\boldsymbol{0}``, and the rule then shrinks the factor mean towards zero, with a weight that falls as the mean becomes less distinct from zero. The rule takes no warm-up: on a short history the factor mean has its largest error, so a pull towards a weight of one is the wrong direction.

The form is that of the shrinkage of a sample mean towards a target in [jorion1986](@cite): the weight of the target is the sampling error of the mean over the squared distance of the mean from the target. The rule estimates that ratio from the fit and takes its positive part.

# Mathematical definition

```math
\\begin{align}
B &= \\frac{1}{n} \\sum_{j=1}^{n} \\left( \\boldsymbol{g}_{j} - \\hat{\\boldsymbol{\\mu}}_{f} \\right)^{\\intercal} \\hat{\\mathbf{\\Sigma}}_{f}^{+} \\left( \\boldsymbol{g}_{j} - \\hat{\\boldsymbol{\\mu}}_{f} \\right)\\,, \\\\
\\lambda &= \\frac{\\max\\left(B - K / T_{e},\\, 0\\right)}{K / T_{e} + \\max\\left(B - K / T_{e},\\, 0\\right)} = \\max\\left(0,\\, 1 - \\frac{K}{T_{e} B}\\right)\\,.
\\end{align}
```

Where:

  - ``B``: Mean squared distance of the spanned coefficients from the factor mean, in the metric of the pseudo-inverse of the factor covariance.
  - ``\\boldsymbol{g}_{j}``: Spanned coefficients of row ``j`` of the Return Forecast, which `err` chooses. Under [`CurrentForecastError`](@ref) there is one row, the latest.
  - ``n``: Number of rows that `err` chooses.
  - ``\\hat{\\boldsymbol{\\mu}}_{f}``: Factor mean of the nested factor prior, over the estimated factors.
  - ``\\hat{\\mathbf{\\Sigma}}_{f}^{+}``: Pseudo-inverse of the factor covariance of the nested factor prior, over the estimated factors, as [`spanned_shrinkage_moments`](@ref) states it.
  - ``\\lambda``: Spanned Shrinkage, the weight of the factor mean.
  - ``K``: Rank of the factor covariance over the estimated factors.
  - $(math_dict[:cal_T_e])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PrecisionBlend(;
        err::AbstractForecastErrorAlgorithm = CurrentForecastError()
    ) -> PrecisionBlend

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PrecisionBlend().err isa CurrentForecastError
true
```

# Related

  - [`AbstractSpannedShrinkageCalibrationAlgorithm`](@ref)
  - [`SteinShrinkage`](@ref): the shrinkage intensity of an expected-returns estimator, towards `g`.
  - [`AbstractForecastErrorAlgorithm`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_forecast_mu`](@ref)

# References

  - $(ref_dict[:jorion1986])
"""
@concrete struct PrecisionBlend <: AbstractSpannedShrinkageCalibrationAlgorithm
    """
    The rows of the spanned part of the Return Forecast whose distance from the factor mean measures the error of the forecast, a member of [`AbstractForecastErrorAlgorithm`](@ref). [`CurrentForecastError`](@ref) reads the latest row, and [`ForecastHistoryError`](@ref) reads the Return Forecast history.
    """
    err
    function PrecisionBlend(err::AbstractForecastErrorAlgorithm)
        return new{typeof(err)}(err)
    end
end
function PrecisionBlend(; err::AbstractForecastErrorAlgorithm = CurrentForecastError())
    return PrecisionBlend(err)
end
function reads_forecast_history(alg::PrecisionBlend)
    return reads_forecast_history(alg.err)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the Spanned Shrinkage of a Cross-Sectional Factor Prior by the precision blend of [`PrecisionBlend`](@ref).

# Algorithm

 1. Read the fit of the prior off `ctx` with [`cross_sectional_context`](@ref), and the spanned coefficients that `alg.err` chooses with [`spanned_forecast_sample`](@ref).
 2. Take the factor mean of the estimated factors and the pseudo-inverse and the rank `K` of their factor covariance with [`spanned_shrinkage_moments`](@ref).
 3. Average the squared distance of each column of the coefficients from the factor mean, in the metric of the pseudo-inverse of the factor covariance, giving `B`.
 4. Read the effective sample size with [`effective_sample_size`](@ref), and return `1 - K / (T_e B)` when `B > K / T_e`, and zero otherwise.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The rule does not read it.
  - `pr`: The factor moments of the nested factor prior, on the reduced factor axis.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The site's [`CalibrationContext`](@ref). The rule reads `ctx.cs`.

# Validation

  - The rules of [`cross_sectional_context`](@ref) and of [`spanned_forecast_sample`](@ref).

# Returns

  - `lambda::Number`: The Spanned Shrinkage, in `[0, 1]`.

# Related

  - [`PrecisionBlend`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::PrecisionBlend)(::Symbol, pr::AbstractPriorResult, w, ::Any, ctx)
    cs = cross_sectional_context(ctx)
    (; mu, G, W, K) = spanned_shrinkage_moments(pr, cs,
                                                spanned_forecast_sample(alg.err, cs))
    B = zero(promote_type(real(eltype(G)), real(eltype(W))))
    for j in axes(G, 2)
        B += sum(abs2, transpose(W) * (view(G, :, j) - mu))
    end
    B /= size(G, 2)
    m1 = K / effective_sample_size(pr, w)
    return B > m1 ? one(B) - m1 / B : zero(B)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the factor mean towards the spanned part of a Return Forecast by the intensity of a shrunk expected-returns estimator.

The rule computes the shrinkage intensity of the algorithm in `alg`, with the spanned coefficients ``\\boldsymbol{g}`` as the target, and returns the weight of the factor mean. [`ShrunkExpectedReturns`](@ref) shrinks a sample mean towards a target that its own field `tgt` computes. This rule reads the same intensity with ``\\boldsymbol{g}`` in place of that target, so it ignores the `tgt` of the algorithm. With no Return Forecast Estimator, ``\\boldsymbol{g}`` is zero.

Each intensity is read off the factor mean and the factor covariance of the nested factor prior, over the estimated factors, and off the effective sample size. The factor covariance enters through its pseudo-inverse and its rank, as [`spanned_shrinkage_moments`](@ref) states. [`BayesStein`](@ref) gives a weight in `[0, 1)` with no clip, as in [jorion1986](@cite). [`JamesStein`](@ref) can give an intensity outside `[0, 1]`, and the rule clips it, which is the positive-part form of [meucci2005](@cite). At two factors or fewer its intensity is not positive, so the rule keeps the factor mean. [`BodnarOkhrinParolya`](@ref) gives the weight of the sample mean and a separate scale of the target, as in [bodnar2019](@cite). The blend of the prior gives the target the weight that the sample mean leaves, so the rule returns the clipped weight of the sample mean and reads no scale of the target.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{d} &= \\hat{\\boldsymbol{\\mu}}_{f} - \\boldsymbol{g}\\,, \\\\
\\lambda_{\\mathrm{BS}} &= 1 - \\frac{K + 2}{K + 2 + T_{e} \\, \\boldsymbol{d}^{\\intercal} \\hat{\\mathbf{\\Sigma}}_{f}^{+} \\boldsymbol{d}}\\,, \\\\
\\lambda_{\\mathrm{JS}} &= 1 - \\min\\left(\\max\\left(\\frac{\\sum_{i} \\ell_{i} - 2 \\, \\ell_{\\max}}{T_{e} \\, \\boldsymbol{d}^{\\intercal} \\boldsymbol{d}},\\, 0\\right),\\, 1\\right)\\,, \\\\
\\lambda_{\\mathrm{BOP}} &= \\min\\left(\\max\\left(\\frac{\\left(u - K / (T_{e} - K)\\right) v - s^{2}}{u v - s^{2}},\\, 0\\right),\\, 1\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{d}``: Difference of the factor mean and the spanned coefficients.
  - ``\\hat{\\boldsymbol{\\mu}}_{f}``: Factor mean of the nested factor prior, over the estimated factors.
  - ``\\hat{\\mathbf{\\Sigma}}_{f}^{+}``: Pseudo-inverse of the factor covariance of the nested factor prior, over the estimated factors, as [`spanned_shrinkage_moments`](@ref) states it.
  - $(math_dict[:g_span])
  - ``\\lambda_{\\mathrm{BS}}``, ``\\lambda_{\\mathrm{JS}}``, ``\\lambda_{\\mathrm{BOP}}``: Spanned Shrinkage under [`BayesStein`](@ref), [`JamesStein`](@ref) and [`BodnarOkhrinParolya`](@ref).
  - ``\\ell_{i}``, ``\\ell_{\\max}``: Kept eigenvalues of the factor covariance over the estimated factors, and the largest of them.
  - ``u``, ``v``, ``s``: The quadratic forms ``\\hat{\\boldsymbol{\\mu}}_{f}^{\\intercal} \\hat{\\mathbf{\\Sigma}}_{f}^{+} \\hat{\\boldsymbol{\\mu}}_{f}``, ``\\boldsymbol{g}^{\\intercal} \\hat{\\mathbf{\\Sigma}}_{f}^{+} \\boldsymbol{g}`` and ``\\hat{\\boldsymbol{\\mu}}_{f}^{\\intercal} \\hat{\\mathbf{\\Sigma}}_{f}^{+} \\boldsymbol{g}``.
  - ``K``: Rank of the factor covariance over the estimated factors.
  - $(math_dict[:cal_T_e])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SteinShrinkage(;
        alg::AbstractShrunkExpectedReturnsAlgorithm = BayesStein()
    ) -> SteinShrinkage

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> SteinShrinkage().alg isa BayesStein
true
```

# Related

  - [`AbstractSpannedShrinkageCalibrationAlgorithm`](@ref)
  - [`PrecisionBlend`](@ref): the default rule of the slot.
  - [`ShrunkExpectedReturns`](@ref)
  - [`stein_spanned_shrinkage`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)

# References

  - $(ref_dict[:jorion1986])
  - $(ref_dict[:meucci2005])
  - $(ref_dict[:bodnar2019])
"""
@concrete struct SteinShrinkage <: AbstractSpannedShrinkageCalibrationAlgorithm
    """
    The algorithm whose shrinkage intensity the rule reads, a member of [`AbstractShrunkExpectedReturnsAlgorithm`](@ref). Its target `tgt` is not read: the target is the spanned part of the Return Forecast.
    """
    alg
    function SteinShrinkage(alg::AbstractShrunkExpectedReturnsAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function SteinShrinkage(; alg::AbstractShrunkExpectedReturnsAlgorithm = BayesStein())
    return SteinShrinkage(alg)
end
"""
    stein_spanned_shrinkage(alg::BayesStein, mu::VecNum, g::VecNum, W::MatNum, ev::VecNum,
                            T::Number)
    stein_spanned_shrinkage(alg::JamesStein, mu::VecNum, g::VecNum, W::MatNum, ev::VecNum,
                            T::Number)
    stein_spanned_shrinkage(alg::BodnarOkhrinParolya, mu::VecNum, g::VecNum, W::MatNum,
                            ev::VecNum, T::Number)

Return the weight of the factor mean that the shrinkage intensity of `alg` gives, with `g` as the target.

[`SteinShrinkage`](@ref) states the three formulas. This function takes one method per algorithm.

# Arguments

  - `alg`: The shrunk expected-returns algorithm.
  - `mu`: The factor mean of the estimated factors.
  - `g`: The spanned coefficients.
  - `W`: The whitening matrix of [`spanned_shrinkage_moments`](@ref), so `W W'` is the pseudo-inverse of the factor covariance and `K` is its number of columns.
  - `ev`: The kept eigenvalues of the factor covariance.
  - `T`: The effective sample size.

# Validation

  - [`JamesStein`](@ref): when `mu == g`, every weight gives the same blend, and the function returns one.
  - [`BodnarOkhrinParolya`](@ref): `T > K`, with `K` the rank of the factor covariance. Raises a `DomainError`.
  - [`BodnarOkhrinParolya`](@ref): `u v - s^2` is not zero, which fails when `g` is a multiple of `mu`, and so when `g` is zero. Raises a `DomainError`.

# Returns

  - `lambda::Number`: The Spanned Shrinkage, in `[0, 1]`.

# Related

  - [`SteinShrinkage`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
function stein_spanned_shrinkage(::BayesStein, mu::VecNum, g::VecNum, W::MatNum, ::VecNum,
                                 T::Number)
    K = size(W, 2)
    alpha = (K + 2) / ((K + 2) + T * sum(abs2, transpose(W) * (mu - g)))
    return one(alpha) - alpha
end
function stein_spanned_shrinkage(::JamesStein, mu::VecNum, g::VecNum, ::MatNum, ev::VecNum,
                                 T::Number)
    d = mu - g
    dd = LinearAlgebra.dot(d, d)
    if iszero(dd)
        return one(dd)
    end
    alpha = (sum(ev) - 2 * maximum(ev)) / (T * dd)
    return one(alpha) - clamp(alpha, zero(alpha), one(alpha))
end
function stein_spanned_shrinkage(::BodnarOkhrinParolya, mu::VecNum, g::VecNum, W::MatNum,
                                 ::VecNum, T::Number)
    K = size(W, 2)
    @argcheck(T > K,
              DomainError((T, K),
                          "the Bodnar-Okhrin-Parolya weight contains `K / (T - K)`, so it needs an effective sample size above the rank of the factor covariance. The term is undefined at T == K and negative below it, got T = $T, K = $K"))
    zm = transpose(W) * mu
    zg = transpose(W) * g
    u = LinearAlgebra.dot(zm, zm)
    v = LinearAlgebra.dot(zg, zg)
    sc = LinearAlgebra.dot(zm, zg)
    gap = u * v - sc^2
    @argcheck(!iszero(gap),
              DomainError(gap,
                          "the Bodnar-Okhrin-Parolya weight divides by the Cauchy-Schwarz gap `u * v - s^2`, which is exactly zero because the spanned part of the Return Forecast is a multiple of the factor mean. A prior with no Return Forecast Estimator has a spanned part of zero, which is such a multiple. State a Return Forecast Estimator, or take BayesStein or JamesStein in `alg`."))
    alpha = ((u - K / (T - K)) * v - sc^2) / gap
    return clamp(alpha, zero(alpha), one(alpha))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the Spanned Shrinkage of a Cross-Sectional Factor Prior by the shrinkage intensity of [`SteinShrinkage`](@ref).

# Algorithm

 1. Read the fit of the prior off `ctx` with [`cross_sectional_context`](@ref), and the latest spanned coefficients with [`spanned_forecast_sample`](@ref) under [`CurrentForecastError`](@ref).
 2. Take the factor mean of the estimated factors and the pseudo-inverse, the eigenvalues and the rank of their factor covariance with [`spanned_shrinkage_moments`](@ref).
 3. Read the effective sample size with [`effective_sample_size`](@ref), and return the weight of [`stein_spanned_shrinkage`](@ref).

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The rule does not read it.
  - `pr`: The factor moments of the nested factor prior, on the reduced factor axis.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The site's [`CalibrationContext`](@ref). The rule reads `ctx.cs`.

# Validation

  - The rules of [`cross_sectional_context`](@ref) and of [`stein_spanned_shrinkage`](@ref).

# Returns

  - `lambda::Number`: The Spanned Shrinkage, in `[0, 1]`.

# Related

  - [`SteinShrinkage`](@ref)
  - [`effective_sample_size`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::SteinShrinkage)(::Symbol, pr::AbstractPriorResult, w, ::Any, ctx)
    cs = cross_sectional_context(ctx)
    m = spanned_shrinkage_moments(pr, cs,
                                  spanned_forecast_sample(CurrentForecastError(), cs))
    return stein_spanned_shrinkage(alg.alg, m.mu, view(m.G, :, 1), m.W, m.ev,
                                   effective_sample_size(pr, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Chooses how the calibration slope of an orthogonal forecast reaches its target on a short history.

[`ForecastCalibrationSlope`](@ref) holds a member of this family in its field `wu`. The slope is noisy on a short history, so the member decides how much of it the rule takes, and how much of the number `target` that the rule falls back to. [`ThresholdWarmUp`](@ref) takes the target below a count of rows and the slope above it. [`PlugInWarmUp`](@ref) and [`PositivePartWarmUp`](@ref) shrink the slope towards the target by its standard error.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractForecastScaleWarmUpAlgorithm` and implement the following method:

## `forecast_scale_warm_up`

  - `forecast_scale_warm_up(wu::MyForecastScaleWarmUp, c::Number, se::Number, n::Integer, target::Number) -> Number`: Returns the Orthogonal Forecast Scale.

### Arguments

  - `wu`: The member of the family.
  - `c`: The calibration slope. It is `NaN` when no pair has a positive weighted square.
  - `se`: The standard error of the slope.
  - `n`: Number of rows of the history that give at least one pair.
  - `target`: The number that the rule falls back to.

### Returns

  - `c::Number`: The Orthogonal Forecast Scale, finite and `>= 0`.

# Related

  - [`ForecastCalibrationSlope`](@ref)
  - [`ThresholdWarmUp`](@ref)
  - [`PlugInWarmUp`](@ref)
  - [`PositivePartWarmUp`](@ref)
  - [`forecast_scale_warm_up`](@ref)
"""
abstract type AbstractForecastScaleWarmUpAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Takes the target below a count of history rows, and the calibration slope at or above it.

The slope is clipped at zero, and a slope that is not finite gives the target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThresholdWarmUp(;
        min_obs::Integer = 252
    ) -> ThresholdWarmUp

Keywords correspond to the struct's fields. The default is one year of daily rows.

## Validation

  - $(val_dict[:min_obs])

# Examples

```jldoctest
julia> ThresholdWarmUp().min_obs
252
```

# Related

  - [`AbstractForecastScaleWarmUpAlgorithm`](@ref)
  - [`PlugInWarmUp`](@ref)
  - [`PositivePartWarmUp`](@ref)
  - [`ForecastCalibrationSlope`](@ref)
"""
@concrete struct ThresholdWarmUp <: AbstractForecastScaleWarmUpAlgorithm
    """
    Number of history rows, each with at least one pair, below which the rule takes the target.
    """
    min_obs
    function ThresholdWarmUp(min_obs::Integer)
        @argcheck(min_obs > zero(min_obs), DomainError(min_obs, "min_obs must be > 0"))
        return new{typeof(min_obs)}(min_obs)
    end
end
function ThresholdWarmUp(; min_obs::Integer = 252)
    return ThresholdWarmUp(min_obs)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the calibration slope towards the target by the plug-in weight of its standard error.

The weight of the slope is the squared distance of the slope from the target over the sum of that distance and the squared standard error. So a slope far from the target, measured in standard errors, keeps most of its value.

# Mathematical definition

```math
\\begin{align}
\\delta &= \\max\\left(\\hat{c},\\, 0\\right) - c_{0}\\,, \\\\
c &= c_{0} + \\frac{\\delta^{2}}{\\delta^{2} + \\mathrm{se}^{2}} \\, \\delta\\,.
\\end{align}
```

Where:

  - ``\\hat{c}``: The calibration slope.
  - ``c_{0}``: The target of the rule.
  - ``\\mathrm{se}``: The standard error of the slope.
  - ``c``: The Orthogonal Forecast Scale.

# Examples

```jldoctest
julia> PortfolioOptimisers.forecast_scale_warm_up(PlugInWarmUp(), 2.0, 1.0, 10, 1)
1.5
```

# Related

  - [`AbstractForecastScaleWarmUpAlgorithm`](@ref)
  - [`PositivePartWarmUp`](@ref)
  - [`ThresholdWarmUp`](@ref)
  - [`ForecastCalibrationSlope`](@ref)
"""
struct PlugInWarmUp <: AbstractForecastScaleWarmUpAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the calibration slope towards the target by the positive-part weight of its standard error.

The weight of the slope is one less the squared standard error over the squared distance of the slope from the target, clipped at zero. So a slope within one standard error of the target gives the target.

# Mathematical definition

```math
\\begin{align}
\\delta &= \\max\\left(\\hat{c},\\, 0\\right) - c_{0}\\,, \\\\
c &= c_{0} + \\max\\left(0,\\, 1 - \\frac{\\mathrm{se}^{2}}{\\delta^{2}}\\right) \\delta\\,.
\\end{align}
```

Where:

  - ``\\hat{c}``: The calibration slope.
  - ``c_{0}``: The target of the rule.
  - ``\\mathrm{se}``: The standard error of the slope.
  - ``c``: The Orthogonal Forecast Scale.

# Examples

```jldoctest
julia> PortfolioOptimisers.forecast_scale_warm_up(PositivePartWarmUp(), 3.0, 1.0, 10, 1)
2.5
```

# Related

  - [`AbstractForecastScaleWarmUpAlgorithm`](@ref)
  - [`PlugInWarmUp`](@ref)
  - [`ThresholdWarmUp`](@ref)
  - [`ForecastCalibrationSlope`](@ref)
"""
struct PositivePartWarmUp <: AbstractForecastScaleWarmUpAlgorithm end
"""
    forecast_scale_warm_up(wu::ThresholdWarmUp, c::Number, se::Number, n::Integer,
                           target::Number) -> Number
    forecast_scale_warm_up(wu::PlugInWarmUp, c::Number, se::Number, n::Integer,
                           target::Number) -> Number
    forecast_scale_warm_up(wu::PositivePartWarmUp, c::Number, se::Number, n::Integer,
                           target::Number) -> Number

Return the Orthogonal Forecast Scale that a warm-up form takes from a calibration slope and its target.

Each form clips the slope at zero first. A slope that is not finite gives the target in each form, and so does a standard error that is not finite in the two shrinkage forms. [`PlugInWarmUp`](@ref) and [`PositivePartWarmUp`](@ref) state their formulas.

# Arguments

  - `wu`: The member of [`AbstractForecastScaleWarmUpAlgorithm`](@ref).
  - `c`: The calibration slope.
  - `se`: The standard error of the slope.
  - `n`: Number of rows of the history that give at least one pair. Only [`ThresholdWarmUp`](@ref) reads it.
  - `target`: The number that the rule falls back to.

# Returns

  - `c::Number`: The Orthogonal Forecast Scale.

# Related

  - [`AbstractForecastScaleWarmUpAlgorithm`](@ref)
  - [`ForecastCalibrationSlope`](@ref)
"""
function forecast_scale_warm_up(wu::ThresholdWarmUp, c::Number, ::Number, n::Integer,
                                target::Number)
    if n < wu.min_obs || !isfinite(c)
        return target
    end
    return max(c, zero(c))
end
function forecast_scale_warm_up(::PlugInWarmUp, c::Number, se::Number, ::Integer,
                                target::Number)
    d = forecast_scale_distance(c, se, target)
    if isnothing(d)
        return target
    end
    return target + d^2 / (d^2 + se^2) * d
end
function forecast_scale_warm_up(::PositivePartWarmUp, c::Number, se::Number, ::Integer,
                                target::Number)
    d = forecast_scale_distance(c, se, target)
    if isnothing(d)
        return target
    end
    return target + max(zero(d), one(d) - se^2 / d^2) * d
end
"""
    forecast_scale_distance(c::Number, se::Number, target::Number) -> Option{<:Number}

Return the distance of the clipped calibration slope from the target, or `nothing` when a shrinkage form takes the target.

The two shrinkage forms of [`AbstractForecastScaleWarmUpAlgorithm`](@ref) divide by the squared distance, and they read the standard error. So a slope that is not finite, a standard error that is not finite, and a distance of zero each give `nothing`, and the form returns the target.

# Arguments

  - `c`: The calibration slope.
  - `se`: The standard error of the slope.
  - `target`: The number that the rule falls back to.

# Returns

  - `d::Option{<:Number}`: `max(c, 0) - target`, or `nothing`.

# Related

  - [`forecast_scale_warm_up`](@ref)
  - [`PlugInWarmUp`](@ref)
  - [`PositivePartWarmUp`](@ref)
"""
function forecast_scale_distance(c::Number, se::Number, target::Number)
    if !isfinite(c) || !isfinite(se)
        return nothing
    end
    d = max(c, zero(c)) - target
    return iszero(d) ? nothing : d
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Orthogonal Forecast Scale as the calibration slope of the orthogonal forecast on the forward idiosyncratic return.

The rule splits each row of the Return Forecast history against the Factor Exposures of that row, and pairs the orthogonal part of each asset with the idiosyncratic return of the asset at the next row. The slope of a weighted regression of the returns on the forecasts, through the origin, is the scale that maps the forecast onto the return it predicts. A scale of the orthogonal part with the least squared error is that slope, as the forecast refinement of [grinoldkahn1999](@cite) states. The rule clips the slope at zero and puts no upper bound on it, so a forecast whose magnitude is too small is scaled up.

The rule reads the history, so the prior makes it, and a prior whose `rfe` has no history refuses the rule at construction. The default `c` of [`CrossSectionalFactorPrior`](@ref) stays one, because a history of a fitted forecast is expensive, so a caller states this rule to take it.

# Mathematical definition

```math
\\begin{align}
\\hat{c} &= \\frac{\\sum_{k} q_{k} a_{k} b_{k}}{\\sum_{k} q_{k} a_{k}^{2}}\\,, \\\\
\\mathrm{se} &= \\frac{\\sqrt{\\sum_{k} \\left( q_{k} a_{k} \\left( b_{k} - \\hat{c} a_{k} \\right) \\right)^{2}}}{\\sum_{k} q_{k} a_{k}^{2}}\\,.
\\end{align}
```

Where:

  - ``\\hat{c}``: The calibration slope, before the warm-up form `wu` takes it.
  - ``\\mathrm{se}``: The heteroskedasticity-consistent standard error of the slope, the HC0 sandwich.
  - ``a_{k}``: Orthogonal part of the forecast of the ``k``-th pair, an asset at a row of the history before the latest.
  - ``b_{k}``: Idiosyncratic return of the same asset at the next row.
  - ``q_{k}``: Regression weight of the same asset at the row of the forecast, zero where that weight is not finite.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastCalibrationSlope(;
        wu::AbstractForecastScaleWarmUpAlgorithm = ThresholdWarmUp(),
        target::Number = 1
    ) -> ForecastCalibrationSlope

Keywords correspond to the struct's fields.

## Validation

  - `target` is finite and `>= 0`.

# Examples

```jldoctest
julia> ForecastCalibrationSlope().target
1
```

# Related

  - [`AbstractOrthogonalForecastScaleCalibrationAlgorithm`](@ref)
  - [`AbstractForecastScaleWarmUpAlgorithm`](@ref)
  - [`forecast_calibration_slope`](@ref)
  - [`orthogonal_forecast_pairs`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)

# References

  - $(ref_dict[:grinoldkahn1999])
"""
@concrete struct ForecastCalibrationSlope <:
                 AbstractOrthogonalForecastScaleCalibrationAlgorithm
    """
    The warm-up form, a member of [`AbstractForecastScaleWarmUpAlgorithm`](@ref). It decides how much of the slope the rule takes on a short history.
    """
    wu
    """
    The number that the rule falls back to under [`ThresholdWarmUp`](@ref), and that it shrinks the slope towards under the two standard-error forms. It is finite and `>= 0`.
    """
    target
    function ForecastCalibrationSlope(wu::AbstractForecastScaleWarmUpAlgorithm,
                                      target::Number)
        assert_orthogonal_forecast_scale(target)
        return new{typeof(wu), typeof(target)}(wu, target)
    end
end
function ForecastCalibrationSlope(;
                                  wu::AbstractForecastScaleWarmUpAlgorithm = ThresholdWarmUp(),
                                  target::Number = 1)
    return ForecastCalibrationSlope(wu, target)
end
function reads_forecast_history(::ForecastCalibrationSlope)
    return true
end
"""
    orthogonal_forecast_pairs(cs::NamedTuple) -> NamedTuple

Pool the pairs of the orthogonal part of each forecast row and the idiosyncratic return of the next row.

# Algorithm

 1. Split the Return Forecast history with [`cross_sectional_split_history`](@ref), giving the orthogonal part `ap` of each row.
 2. Take the idiosyncratic return of the next row as the target of each row, with [`forward_mean_returns`](@ref) at a horizon and a lag of one, over the residuals of `cs.csfm.csr`.
 3. Pool the pairs of the rows before the latest with [`forecast_calibration_pairs`](@ref), under the regression weights `cs.csfm.rw`.
 4. Count the rows that give at least one pair, giving `n`.

# Arguments

  - $(arg_dict[:cal_ctx_cs])

# Validation

  - The rules of [`cross_sectional_split_history`](@ref).

# Returns

  - `a::VecNum`: The orthogonal forecast of each pair.
  - `b::VecNum`: The idiosyncratic return of each pair, at the next row.
  - `q::VecNum`: The weight of each pair.
  - `n::Int`: Number of rows that give at least one pair.

# Related

  - [`ForecastCalibrationSlope`](@ref)
  - [`cross_sectional_split_history`](@ref)
  - [`forecast_calibration_pairs`](@ref)
"""
function orthogonal_forecast_pairs(cs::NamedTuple)
    ap = cross_sectional_split_history(cs).ap
    csfm = cs.csfm
    Y = forward_mean_returns(csfm.csr.eps, 1, 1)
    rw::MatNum = csfm.rw
    dates = 1:(size(ap, 1) - 1)
    a, b, q = forecast_calibration_pairs(ap, Y, rw, dates)
    n = count(t -> any(i -> isfinite(ap[t, i]) && isfinite(Y[t, i]), axes(ap, 2)), dates)
    return (; a = a, b = b, q = q, n = n)
end
"""
    forecast_calibration_slope_se(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                                  q::AbstractVector{<:Real}, c::Real) -> Real

Return the heteroskedasticity-consistent standard error of a calibration slope through the origin.

[`ForecastCalibrationSlope`](@ref) states the formula, the HC0 sandwich of the weighted regression of `b` on `a`.

# Arguments

  - `a`: The forecast of each pair.
  - `b`: The target of each pair.
  - `q`: The weight of each pair.
  - `c`: The slope, from [`forecast_calibration_slope`](@ref).

# Returns

  - `se::Real`: The standard error. It is `NaN` when no pair has a positive weighted square.

# Related

  - [`forecast_calibration_slope`](@ref)
  - [`ForecastCalibrationSlope`](@ref)
"""
function forecast_calibration_slope_se(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                                       q::AbstractVector{<:Real}, c::Real)
    Tf = promote_type(real(eltype(a)), real(eltype(b)), real(eltype(q)), typeof(c))
    meat = zero(Tf)
    den = zero(Tf)
    for k in eachindex(a)
        den += q[k] * a[k]^2
        meat += (q[k] * a[k] * (b[k] - c * a[k]))^2
    end
    return den > zero(Tf) ? sqrt(meat) / den : Tf(NaN)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the Orthogonal Forecast Scale of a Cross-Sectional Factor Prior by the calibration slope of [`ForecastCalibrationSlope`](@ref).

# Algorithm

 1. Read the fit of the prior off `ctx` with [`cross_sectional_context`](@ref).
 2. Pool the pairs with [`orthogonal_forecast_pairs`](@ref).
 3. Take the slope with [`forecast_calibration_slope`](@ref), and its standard error with [`forecast_calibration_slope_se`](@ref).
 4. Return the scale of [`forecast_scale_warm_up`](@ref) under `alg.wu` and `alg.target`.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The rule does not read it.
  - `pr`: The factor moments of the nested factor prior. The rule does not read them.
  - `w`: Effective observation weights. The rule does not read them.
  - `slv`: Effective solver. This rule needs none.
  - `ctx`: The site's [`CalibrationContext`](@ref). The rule reads `ctx.cs`.

# Validation

  - The rules of [`cross_sectional_context`](@ref) and of [`orthogonal_forecast_pairs`](@ref).

# Returns

  - `c::Number`: The Orthogonal Forecast Scale, finite and `>= 0`.

# Related

  - [`ForecastCalibrationSlope`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::ForecastCalibrationSlope)(::Symbol, ::AbstractPriorResult, ::Any, ::Any, ctx)
    (; a, b, q, n) = orthogonal_forecast_pairs(cross_sectional_context(ctx))
    c = forecast_calibration_slope(a, b, q)
    se = forecast_calibration_slope_se(a, b, q, c)
    return forecast_scale_warm_up(alg.wu, c, se, n, alg.target)
end

export PrecisionBlend, CurrentForecastError, ForecastHistoryError, SteinShrinkage,
       ForecastCalibrationSlope, ThresholdWarmUp, PlugInWarmUp, PositivePartWarmUp
public AbstractForecastErrorAlgorithm, AbstractForecastScaleWarmUpAlgorithm,
       spanned_forecast_sample, forecast_scale_warm_up
