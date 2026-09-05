"""
$(DocStringExtensions.TYPEDEF)

The shared recipe that turns Descriptors into cross-sectional scores.

Every fitted member of the Return Forecast family starts from the same steps: compute each Descriptor, transform it cross-sectionally, stack the results, and, when the caller names Neutralisation targets, residualise every score against the named Factor Exposures and score it once more. The recipe is one struct in a slot rather than five fields repeated on each member, and it answers a verb of its own, [`descriptor_scores`](@ref), so a caller inspects the scores without fitting a forecast.

The Descriptors carry no names, because nothing reads them: the weights of a member are positional, and the scores are stacked in the order the Descriptors are written in.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`descriptor_scores`](@ref)
  - [`AbstractDescriptorEstimator`](@ref)
  - [`AbstractCrossSectionalTransform`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
@concrete struct DescriptorScores <: AbstractEstimator
    """
    Descriptor Estimators the scores are built from, in the order a member weights them in.
    """
    descriptors
    """
    Names of the factors or the Factor Families every score is neutralised against, or `nothing` to neutralise none.
    """
    neutralise
    """
    Cross-sectional transform applied to each Descriptor before it is scored, or `nothing` to skip the step.
    """
    outlier
    """
    Cross-sectional transform applied to each Descriptor after the outlier step, and once more after the Neutralisation, or `nothing` to skip the step.
    """
    scoring
    """
    Name of the categorical Panel Field the transforms are applied within, or `nothing` to transform each observation as one cross-section.
    """
    group
    function DescriptorScores(descriptors::AbstractVector{<:AbstractDescriptorEstimator},
                              neutralise::Option{<:Union{<:AbstractString, <:VecStr}},
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
        return new{typeof(descriptors), typeof(neutralise), typeof(outlier),
                   typeof(scoring), typeof(group)}(descriptors, neutralise, outlier,
                                                   scoring, group)
    end
end
function DescriptorScores(; descriptors::AbstractVector{<:AbstractDescriptorEstimator},
                          neutralise::Option{<:Union{<:AbstractString, <:VecStr}} = nothing,
                          outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                          scoring::Option{<:AbstractCrossSectionalTransform} = CrossSectionalStandardiser(),
                          group::Option{<:AbstractString} = nothing)::DescriptorScores
    return DescriptorScores(descriptors, neutralise, outlier, scoring, group)
end
"""
    assert_neutralisation_names(neutralise::AbstractString) -> nothing
    assert_neutralisation_names(neutralise::VecStr) -> nothing

Check that every Neutralisation name of a [`DescriptorScores`](@ref) names something.

A name is resolved against the factor axis of the block when the scores are computed, and this refuses the two forms that can never resolve: the empty list and the empty string.

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

The three reads are optional fields of the block, so this is the one place that states which of them a Neutralisation needs, and it names the missing one in the refusal.

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
                       csfm::CrossSectionalFactorModel, w::MatNum,
                       scoring::Option{<:AbstractCrossSectionalTransform},
                       groups::Option{<:AbstractMatrix{<:Integer}}) -> nothing
    neutralise_scores!(S::AbstractArray{<:Real, 3},
                       neutralise::Union{<:AbstractString, <:VecStr},
                       csfm::CrossSectionalFactorModel, w::MatNum,
                       scoring::Option{<:AbstractCrossSectionalTransform},
                       groups::Option{<:AbstractMatrix{<:Integer}}) -> nothing

Neutralise the Descriptor scores against the named Factor Exposures, in place.

# Algorithm

The method that Julia selects is the algorithm, and the recipe that names no target does nothing.

 1. Resolve the names to raw factor indices, a factor name beating a Factor Family label, and take those columns of the exposure history as the design.
 2. For each score in turn, build the regression weights: `w`, with a zero wherever the score or a design exposure of that asset is not finite.
 3. Regress the score across the assets on the design under those weights, with no intercept, and take the residual.
 4. Score the residual once more under `w`, so that every score leaves the step on one scale.

# Arguments

  - `S`: The Descriptor scores, `observations × assets × descriptors`. It is changed in place.
  - `neutralise`: The Neutralisation names, or `nothing`.
  - `csfm`: The fitted factor-model block.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `scoring`: The scoring transform, or `nothing`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.

# Validation

  - The rules of [`descriptor_scores_axis`](@ref) and of [`neutralisation_targets`](@ref).

# Returns

  - `nothing`. `S` carries the neutralised scores.

# Related

  - [`DescriptorScores`](@ref)
  - [`descriptor_scores`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`neutralisation_weights`](@ref)
"""
function neutralise_scores!(::AbstractArray{<:Real, 3}, ::Nothing,
                            ::CrossSectionalFactorModel, ::MatNum,
                            ::Option{<:AbstractCrossSectionalTransform},
                            ::Option{<:AbstractMatrix{<:Integer}})::Nothing
    return nothing
end
function neutralise_scores!(S::AbstractArray{<:Real, 3},
                            neutralise::Union{<:AbstractString, <:VecStr},
                            csfm::CrossSectionalFactorModel, w::MatNum,
                            scoring::Option{<:AbstractCrossSectionalTransform},
                            groups::Option{<:AbstractMatrix{<:Integer}})::Nothing
    Ms, nf, fam = descriptor_scores_axis(csfm)
    tidx = neutralisation_targets(neutralisation_names(neutralise), nf, fam)
    X = Ms[:, :, tidx]
    cre = CrossSectionalLinearRegression()
    for k in axes(S, 3)
        y = S[:, :, k]
        W = neutralisation_weights(y, X, w)
        csr = cross_sectional_regression(cre, X, y, W)
        S[:, :, k] = exposure_transform(scoring, csr.eps, w, groups)
    end
    return nothing
end
"""
    descriptor_scores(ds::DescriptorScores, rd::ReturnsResult,
                      csfm::CrossSectionalFactorModel) -> Array{<:Real, 3}

Compute the cross-sectional scores of the Descriptors of a [`DescriptorScores`](@ref).

# Algorithm

 1. Read the cross-sectional weights off the estimation mask of the Asset Panel, and the group labels off the named categorical Panel Field.
 2. Compute each Descriptor, and apply the outlier slot and then the scoring slot to it.
 3. Stack the scores on a third axis, in the order the Descriptors are written in.
 4. When the recipe names Neutralisation targets, residualise every score against those Factor Exposures and score it once more.

# Arguments

  - `ds`: The Descriptor Scores recipe.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block. It is read only when the recipe names Neutralisation targets.

# Validation

  - The rules of [`return_forecast_weights`](@ref), of [`exposure_group_labels`](@ref) and of [`cross_sectional_transform`](@ref).
  - The rules of [`neutralise_scores!`](@ref) when the recipe names Neutralisation targets.

# Returns

  - `S::Array{<:Real, 3}`: The Descriptor scores, `observations × assets × descriptors`.

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

julia> descriptor_scores(ds, rd, csfm)
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
  - [`FixedWeightedReturnForecast`](@ref)
"""
function descriptor_scores(ds::DescriptorScores, rd::ReturnsResult,
                           csfm::CrossSectionalFactorModel)::Array{<:Real, 3}
    w = return_forecast_weights(rd)
    groups = exposure_group_labels(rd, ds.group)
    des = ds.descriptors
    S1 = composite_score(des[1], rd, ds.outlier, ds.scoring, w, groups)
    Tf = float(eltype(S1))
    S = Array{Tf, 3}(undef, size(S1, 1), size(S1, 2), length(des))
    S[:, :, 1] = S1
    for k in 2:length(des)
        S[:, :, k] = composite_score(des[k], rd, ds.outlier, ds.scoring, w, groups)
    end
    neutralise_scores!(S, ds.neutralise, csfm, w, ds.scoring, groups)
    return S
end

export DescriptorScores, descriptor_scores
