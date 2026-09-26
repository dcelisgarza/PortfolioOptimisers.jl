"""
$(DocStringExtensions.TYPEDEF)

The shared recipe that turns Descriptors into cross-sectional scores.

Every fitted member of the Return Forecast family starts with the same steps. It computes each Descriptor, transforms it cross-sectionally and stacks the results. When the caller names Neutralisation targets, it also residualises every score against the named Factor Exposures and scores it once more. The recipe is one struct in a slot, so no member repeats its six fields. It has a verb of its own, [`descriptor_scores`](@ref), so a caller can read the scores without the fit of a forecast.

The Descriptors carry no names, because no code reads a name. The weights of a member are positional, and [`descriptor_scores`](@ref) stacks the scores in the order of the Descriptors.

# Mathematical definition

```math
\\begin{align}
\\tilde{d}_{tij} &= \\mathcal{Z}_{t}\\left(\\mathcal{O}_{t}\\left(\\boldsymbol{d}_{t \\cdot j}\\right)\\right)_{i}\\,, \\\\
\\omega_{tij} &= u_{ti} \\, \\mathbb{1}\\left[u_{ti} > 0,\\ \\tilde{d}_{tij} \\in \\mathbb{R},\\ \\boldsymbol{b}_{ti} \\in \\mathbb{R}^{\\lvert \\mathcal{K} \\rvert}\\right]\\,, \\\\
\\left(\\hat{a}_{tj}, \\hat{\\boldsymbol{\\gamma}}_{tj}\\right) &= \\underset{a,\\, \\boldsymbol{\\gamma}}{\\arg\\min} \\sum_{i = 1}^{N} \\omega_{tij} \\left(\\tilde{d}_{tij} - a - \\boldsymbol{b}_{ti}^{\\intercal} \\boldsymbol{\\gamma}\\right)^{2}\\,, \\\\
e_{tij} &= \\tilde{d}_{tij} - \\hat{a}_{tj} - \\boldsymbol{b}_{ti}^{\\intercal} \\hat{\\boldsymbol{\\gamma}}_{tj}\\,, \\\\
s_{tij} &= \\mathcal{Z}_{t}\\left(\\boldsymbol{e}_{t \\cdot j}\\right)_{i}\\,.
\\end{align}
```

Where:

  - ``d_{tij}``: Descriptor ``j`` of asset ``i`` at observation ``t``, and ``\\boldsymbol{d}_{t \\cdot j}`` its cross-section.
  - ``\\mathcal{O}_{t}``, ``\\mathcal{Z}_{t}``: The outlier transform and the scoring transform of one cross-section of observation ``t``. Each reads the weights ``u_{ti}`` and the group labels of `group` as its own docstring states. A slot that holds `nothing` is the identity.
  - $(math_dict[:u_ti_cs]) It is one where the estimation mask of the Asset Panel is `true`, and zero where it is `false`.
  - ``\\tilde{d}_{tij}``: Score of Descriptor ``j`` of asset ``i`` at observation ``t`` before the Neutralisation.
  - ``\\mathcal{K}``: The raw factors that the Neutralisation names. A name resolves to a factor before it resolves to a Factor Family.
  - $(math_dict[:B_tik_cs])
  - ``\\boldsymbol{b}_{ti}``: The exposures ``B_{tik}`` of asset ``i`` at observation ``t`` to the factors ``k \\in \\mathcal{K}``.
  - ``\\omega_{tij}``: Regression weight of asset ``i`` in the Neutralisation of Descriptor ``j`` at observation ``t``. It is zero where the weight, the score or an exposure is not finite.
  - ``\\hat{a}_{tj}``, ``\\hat{\\boldsymbol{\\gamma}}_{tj}``: Intercept and slopes of the weighted fit. The intercept is ``0`` when `cre.intercept` is `false`, and the solve algorithm of `cre` chooses among the minimisers of a rank deficient fit.
  - ``e_{tij}``: Residual of asset ``i``. It is defined for every asset, including an asset with ``\\omega_{tij} = 0``.
  - ``s_{tij}``: Score of Descriptor ``j`` of asset ``i`` at observation ``t``. Without a Neutralisation, ``s_{tij} = \\tilde{d}_{tij}``. With a Neutralisation, ``s_{tij}`` is `NaN` at every observation before the first observation of the factor-model block, because the block states no exposure there.
  - $(math_dict[:N])

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Constructors

    DescriptorScores(; descriptors::AbstractVector{<:AbstractDescriptorEstimator},
                     neutralise::Option{<:Union{<:AbstractString, <:VecStr}} = nothing,
                     cre::AbstractCrossSectionalRegressionEstimator = CrossSectionalLinearRegression(),
                     outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                     scoring::Option{<:AbstractCrossSectionalTransform} = CrossSectionalStandardiser(),
                     group::Option{<:AbstractString} = nothing)

# Related

  - [`descriptor_scores`](@ref)
  - [`AbstractDescriptorEstimator`](@ref)
  - [`AbstractCrossSectionalTransform`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
"""
@concrete struct DescriptorScores <: AbstractEstimator
    """
    Descriptor Estimators that give the scores, in the order that a member weights them.
    """
    descriptors
    """
    Names of the factors or the Factor Families that the recipe neutralises every score against, or `nothing` to neutralise none.
    """
    neutralise
    """
    Cross-Sectional Regression Estimator of the Neutralisation. Its `intercept` sets what the residual is orthogonal to. Under `false`, the default, the fit removes the component along the raw exposure. Under `true`, it removes the component along the cross-sectional deviation of the exposure from its mean, so the residual is also uncorrelated with the exposure.
    """
    cre
    """
    Cross-sectional transform that the recipe applies to each Descriptor before the scoring step, or `nothing` to skip the step.
    """
    outlier
    """
    Cross-sectional transform that the recipe applies to each Descriptor after the outlier step, and once more after the Neutralisation, or `nothing` to skip the step.
    """
    scoring
    """
    Name of the categorical Panel Field whose group labels the transforms read, or `nothing` to transform each observation as one cross-section.
    """
    group
    function DescriptorScores(descriptors::AbstractVector{<:AbstractDescriptorEstimator},
                              neutralise::Option{<:Union{<:AbstractString, <:VecStr}},
                              cre::AbstractCrossSectionalRegressionEstimator,
                              outlier::Option{<:AbstractCrossSectionalTransform},
                              scoring::Option{<:AbstractCrossSectionalTransform},
                              group::Option{<:AbstractString})
        @argcheck(!isempty(descriptors),
                  IsEmptyError("Descriptor Scores are built from Descriptors, so they need at least one"))
        if !isnothing(neutralise)
            assert_neutralisation_names(neutralise)
        end
        if !isnothing(group)
            assert_panel_terms(group, :group)
        end
        return new{typeof(descriptors), typeof(neutralise), typeof(cre), typeof(outlier),
                   typeof(scoring), typeof(group)}(descriptors, neutralise, cre, outlier,
                                                   scoring, group)
    end
end
function DescriptorScores(; descriptors::AbstractVector{<:AbstractDescriptorEstimator},
                          neutralise::Option{<:Union{<:AbstractString, <:VecStr}} = nothing,
                          cre::AbstractCrossSectionalRegressionEstimator = CrossSectionalLinearRegression(),
                          outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                          scoring::Option{<:AbstractCrossSectionalTransform} = CrossSectionalStandardiser(),
                          group::Option{<:AbstractString} = nothing)::DescriptorScores
    return DescriptorScores(descriptors, neutralise, cre, outlier, scoring, group)
end
"""
    assert_neutralisation_names(neutralise::AbstractString) -> nothing
    assert_neutralisation_names(neutralise::VecStr) -> nothing

Check that every Neutralisation name of a [`DescriptorScores`](@ref) names something.

[`descriptor_scores`](@ref) resolves a name against the factor axis of the block. This function refuses the two forms that never resolve, the empty list and the empty string.

# Arguments

  - `neutralise`: One name, or a list of names.

# Validation

  - `!isempty(neutralise)`. Raises an [`IsEmptyError`](@ref).
  - No entry of a list is the empty string. Raises an [`IsEmptyError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`DescriptorScores`](@ref)
  - [`descriptor_scores`](@ref)
  - [`neutralisation_indices`](@ref)
"""
function assert_neutralisation_names(neutralise::AbstractString)::Nothing
    @argcheck(!isempty(neutralise),
              IsEmptyError("a Neutralisation name names a factor or a Factor Family, so it cannot be the empty string"))
    return nothing
end
function assert_neutralisation_names(neutralise::VecStr)::Nothing
    @argcheck(!isempty(neutralise),
              IsEmptyError("neutralise names the factors the scores are neutralised against, so it cannot be empty. Use nothing to neutralise none"))
    for (k, nm) in enumerate(neutralise)
        @argcheck(!isempty(nm),
                  IsEmptyError("name $k of neutralise names a factor or a Factor Family, so it cannot be the empty string"))
    end
    return nothing
end
"""
    descriptor_scores_axis(csfm::CrossSectionalFactorModel) -> Tuple

Return the factor axis and the exposure history a Neutralisation resolves against.

The three fields are optional on the block. This function is the one place that states which of them a Neutralisation needs, and its refusal names the missing field.

# Arguments

  - `csfm`: The fitted factor-model block.

# Validation

  - `csfm.Ms`, `csfm.nf` and `csfm.fam` are all given. Raises an [`IsNothingError`](@ref).

# Returns

  - `(Ms, nf, fam)::Tuple`: The exposure history, the factor names and the family labels.

# Related

  - [`descriptor_scores`](@ref)
  - [`neutralise_scores!`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`neutralisation_targets`](@ref)
"""
function descriptor_scores_axis(csfm::CrossSectionalFactorModel)
    Ms = csfm.Ms
    nf = csfm.nf
    fam = csfm.fam
    @argcheck(!isnothing(Ms),
              IsNothingError("a Neutralisation regresses each score on the Factor Exposures of its observation, and the block carries no exposure history in Ms"))
    @argcheck(!isnothing(nf),
              IsNothingError("a Neutralisation names factors, and the block carries no factor names in nf"))
    @argcheck(!isnothing(fam),
              IsNothingError("a Neutralisation names factors or Factor Families, and the block carries no family labels in fam"))
    return Ms, nf, fam
end
"""
    neutralise_scores!(S::AbstractArray{<:Real, 3}, neutralise::Nothing,
                       cre::AbstractCrossSectionalRegressionEstimator,
                       csfm::CrossSectionalFactorModel, w::MatNum,
                       scoring::Option{<:AbstractCrossSectionalTransform},
                       groups::Option{<:AbstractMatrix{<:Integer}},
                       rows::AbstractUnitRange) -> nothing
    neutralise_scores!(S::AbstractArray{<:Real, 3},
                       neutralise::Union{<:AbstractString, <:VecStr},
                       cre::AbstractCrossSectionalRegressionEstimator,
                       csfm::CrossSectionalFactorModel, w::MatNum,
                       scoring::Option{<:AbstractCrossSectionalTransform},
                       groups::Option{<:AbstractMatrix{<:Integer}},
                       rows::AbstractUnitRange) -> nothing

Neutralise the Descriptor scores against the named Factor Exposures, in place.

# Algorithm

The method that Julia selects is the algorithm. The method for a recipe that names no target does nothing.

 1. Resolve the names to the raw factor indices `tidx`. A name resolves to a factor before it resolves to a Factor Family label. Take those columns of the exposure history as the design `X`.
 2. For each score in turn, build the regression weights `W` over the rows of the block. They are `w`, with a zero where the score or a design exposure of the asset is not finite.
 3. Regress the score across the assets on the design under those weights with `cre`, and take the residual `csr.eps`. Under `cre.intercept = false` the residual is orthogonal to the raw design. Under `true` it is also uncorrelated with the design, as [`CrossSectionalLinearRegression`](@ref) states.
 4. Score the residual once more under `w` and the group labels, so that every score leaves the step on one scale.
 5. Write `NaN` on the rows before the block, because the block states no exposure there.

# Arguments

  - `S`: The Descriptor scores, `observations × assets × descriptors`, on the observation axis of the carrier. The function changes it in place.
  - `neutralise`: The Neutralisation names, or `nothing`.
  - `cre`: Cross-Sectional Regression Estimator that fits the residualisation.
  - `csfm`: The fitted factor-model block.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `scoring`: The scoring transform, or `nothing`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.
  - `rows`: The rows of the carrier that the block covers.

# Validation

  - The rules of [`descriptor_scores_axis`](@ref) and of [`neutralisation_targets`](@ref).
  - When the block starts after the first row of the carrier, the element type of `S` holds `NaN`. An `Integer` or a `Rational` element type raises an `ArgumentError`.

# Returns

  - `nothing`. `S` carries the neutralised scores.

# Related

  - [`DescriptorScores`](@ref)
  - [`descriptor_scores`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`neutralisation_weights`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
"""
function neutralise_scores!(::AbstractArray{<:Real, 3}, ::Nothing,
                            ::AbstractCrossSectionalRegressionEstimator,
                            ::CrossSectionalFactorModel, ::MatNum,
                            ::Option{<:AbstractCrossSectionalTransform},
                            ::Option{<:AbstractMatrix{<:Integer}},
                            ::AbstractUnitRange)::Nothing
    return nothing
end
function neutralise_scores!(S::AbstractArray{<:Real, 3},
                            neutralise::Union{<:AbstractString, <:VecStr},
                            cre::AbstractCrossSectionalRegressionEstimator,
                            csfm::CrossSectionalFactorModel, w::MatNum,
                            scoring::Option{<:AbstractCrossSectionalTransform},
                            groups::Option{<:AbstractMatrix{<:Integer}},
                            rows::AbstractUnitRange)::Nothing
    Ms, nf, fam = descriptor_scores_axis(csfm)
    tidx = neutralisation_targets(neutralisation_names(neutralise), nf, fam)
    X = Ms[:, :, tidx]
    wb = return_forecast_cut(w, rows)
    gb = return_forecast_cut(groups, rows)
    Tf = eltype(S)
    @argcheck(first(rows) == 1 || !(Tf <: Union{Integer, Rational}),
              ArgumentError("a neutralised score is NaN on the $(first(rows) - 1) observations before the factor model block, and the element type $Tf of the scores cannot hold NaN. Convert the Panel Fields to a floating-point type, or hand the carrier the block was fitted on."))
    for k in axes(S, 3)
        y = S[rows, :, k]
        W = neutralisation_weights(y, X, wb)
        csr = cross_sectional_regression(cre, X, y, W)
        S[rows, :, k] = exposure_transform(scoring, csr.eps, wb, gb)
    end
    S[1:(first(rows) - 1), :, :] .= Tf(NaN)
    return nothing
end
"""
    descriptor_scores(ds::DescriptorScores, rd::ReturnsResult,
                      csfm::CrossSectionalFactorModel) -> NamedTuple

Compute the cross-sectional scores of the Descriptors of a [`DescriptorScores`](@ref).

The function computes the Descriptors over the whole carrier. A Descriptor with a warm-up therefore warms up on every observation of the panel, and not a second time inside the window of the block. The function also returns the rows of the block, and each member cuts the scores to those rows once.

# Algorithm

 1. Read the cross-sectional weights off the estimation mask of the Asset Panel, the group labels off the named categorical Panel Field, and the block's rows with [`return_forecast_rows`](@ref).
 2. Compute each Descriptor over the whole carrier, and apply the outlier slot and then the scoring slot to it.
 3. Stack the scores on a third axis of `S`, in the order of the Descriptors. The number type of `S` is the promotion of the number types of the scores and, when the recipe names Neutralisation targets, of the exposure history.
 4. When the recipe names Neutralisation targets, residualise every score of the block's rows against those Factor Exposures, score it once more, and write `NaN` on the rows before the block.

# Arguments

  - `ds`: The Descriptor Scores recipe.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block. Its histories give the rows of the block. The function reads its exposure history only when the recipe names Neutralisation targets.

# Validation

  - The rules of [`return_forecast_weights`](@ref), of [`return_forecast_rows`](@ref), of [`exposure_group_labels`](@ref) and of [`cross_sectional_transform`](@ref).
  - The rules of [`neutralise_scores!`](@ref) when the recipe names Neutralisation targets.

# Returns

  - `S::Array{<:Real, 3}`: The Descriptor scores, `observations × assets × descriptors`, on the carrier's observation axis.
  - `rows::AbstractUnitRange`: The rows of the carrier that the block covers.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"a\", vals = [1.0 2.0; 3.0 4.0]),
                          NumericPanelInput(; name = \"b\", vals = [5.0 6.0; 7.0 8.0])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> csfm = CrossSectionalFactorModel(; M = reshape([1.0, 1.0], 2, 1), b = [0.0, 0.0]);

julia> ds = DescriptorScores(;
                             descriptors = [Passthrough(; field = \"a\"),
                                            Passthrough(; field = \"b\")], outlier = nothing,
                             scoring = nothing);

julia> descriptor_scores(ds, rd, csfm).S
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 1.0  2.0
 3.0  4.0

[:, :, 2] =
 5.0  6.0
 7.0  8.0
```

# Related

  - [`DescriptorScores`](@ref)
  - [`descriptor`](@ref)
  - [`composite_score`](@ref)
  - [`neutralise_scores!`](@ref)
  - [`return_forecast_rows`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
"""
function descriptor_scores(ds::DescriptorScores, rd::ReturnsResult,
                           csfm::CrossSectionalFactorModel)
    w = return_forecast_weights(rd)
    groups = exposure_group_labels(rd, ds.group)
    rows = return_forecast_rows(rd, csfm)
    # `stack` promotes the number types of the scores. Under a Neutralisation the residual is
    # fitted on the exposure history, so `S` also takes the number type of that history.
    S = stack(composite_score(de, rd, ds.outlier, ds.scoring, w, groups)
              for de in ds.descriptors)
    if !isnothing(ds.neutralise)
        Tf = promote_type(eltype(S), eltype(first(descriptor_scores_axis(csfm))))
        S = convert(Array{Tf, 3}, S)
    end
    neutralise_scores!(S, ds.neutralise, ds.cre, csfm, w, ds.scoring, groups, rows)
    return (; S = S, rows = rows)
end

export DescriptorScores, descriptor_scores
