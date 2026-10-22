"""
$(DocStringExtensions.TYPEDEF)

Measures codependence with mutual information, which captures a non-linear relationship a correlation misses.

The mutual information matrix is optionally normalised by the smaller of the two marginal entropies, then rescaled by the marginal standard deviations of `ve` to give a covariance matrix.

Mutual information is non-negative, so every entry of the matrix is non-negative and no pair is ever reported as opposed. A negative linear relationship reads as a strong one, not as a negative one. When `normalise` is `false` the correlation matrix is unbounded above and its diagonal carries the marginal entropy in nats rather than one, so it is a codependence matrix rather than a correlation matrix in the usual sense.

Section 12 of [shannon1948](@cite) defines the rate of transmission ``R = H(x) + H(y) - H(x,\\,y)``, which is the mutual information, with the entropies of Section 6. The estimate here replaces those entropies with histogram plug-in values. The division by the smaller marginal entropy is Equation 6.19 of [cajas2025](@cite). Section 3 of [meila2007](@cite) proves ``I \\leq \\min(H(C),\\, H(C'))`` for clusterings (Equation 18), which is the bound that makes the quotient lie in ``[0,\\, 1]``. No work that first states this normalisation was found.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\rho}}_{ij} &= \\mathrm{MI}(X_i,\\, X_j)\\,, \\\\
\\hat{\\mathbf{\\Sigma}}_{ij} &= \\begin{cases} \\hat{\\sigma}_i^2 & i = j \\\\ \\hat{\\boldsymbol{\\rho}}_{ij}\\,\\hat{\\sigma}_i\\,\\hat{\\sigma}_j & i \\neq j \\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\rho}}_{ij}``: Mutual information-based correlation between assets ``i`` and ``j``.
  - ``\\hat{\\mathbf{\\Sigma}}_{ij}``: Covariance between assets ``i`` and ``j``.
  - ``\\mathrm{MI}(X_i, X_j)``: Mutual information between assets ``i`` and ``j``, computed by [`mutual_info`](@ref). When `normalise` is `true` it is divided by ``\\min(H(X_i), H(X_j))``, which bounds it to ``[0,\\, 1]``.
  - ``\\hat{\\sigma}_i``: Marginal standard deviation of asset ``i`` from the variance estimator `ve`.

The diagonal of ``\\hat{\\mathbf{\\Sigma}}`` is the variance whatever ``\\hat{\\boldsymbol{\\rho}}`` carries there, so the two values of `normalise` give the same diagonal and differ only off it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MutualInfoCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        bins::Int_Bin = HacineGharbiRavier(),
        normalise::Bool = true,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> MutualInfoCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:bins])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> MutualInfoCovariance()
MutualInfoCovariance
         ve ┼ SimpleVariance
            │          me ┼ SimpleExpectedReturns
            │             │   w ┴ nothing
            │           w ┼ nothing
            │   corrected ┴ Bool: true
       bins ┼ HacineGharbiRavier()
  normalise ┴ Bool: true
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractBins`](@ref)
  - [`mutual_info`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:shannon1948]) Section 12.
  - $(ref_dict[:meila2007]) Section 3, Equation 18.
  - $(ref_dict[:cajas2025]) Section 6.1.6, Equation 6.19. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct MutualInfoCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:bins])
    """
    bins
    """
    $(field_dict[:normalise])
    """
    normalise
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function MutualInfoCovariance(ve::AbstractVarianceEstimator, bins::Int_Bin,
                                  normalise::Bool, cache::Option{<:AbstractPartialFitState})
        if isa(bins, Integer)
            @argcheck(zero(bins) < bins, DomainError(bins, "bins must be positive"))
            assert_resource_cap(bins, RESOURCE_LIMITS[].max_bins, :bins, :max_bins)
        end
        return new{typeof(ve), typeof(bins), typeof(normalise), typeof(cache)}(ve, bins,
                                                                               normalise,
                                                                               cache)
    end
end
function MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                              bins::Int_Bin = HacineGharbiRavier(), normalise::Bool = true,
                              cache::Option{<:AbstractPartialFitState} = nothing)::MutualInfoCovariance
    return MutualInfoCovariance(ve, bins, normalise, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`MutualInfoCovariance`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:MutualInfoCovariance, true)` to render it.

# Arguments

  - `::MutualInfoCovariance`: Mutual information covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ve, :bins, :normalise)`.

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::MutualInfoCovariance) = (:ve, :bins, :normalise)
"""
    Statistics.cor(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the mutual information (MI) correlation matrix using a [`MutualInfoCovariance`](@ref) estimator.

This method computes the pairwise mutual information correlation matrix for the input data matrix `X`, using the binning strategy and normalisation specified in `ce`. The MI correlation captures both linear and nonlinear dependencies between asset returns, making it robust to complex relationships that may not be detected by traditional correlation measures.

The result is bounded by ``[0, 1]`` with a unit diagonal only when `ce.normalise` is `true`. When it is `false` the entries are mutual information in nats and the diagonal is the marginal entropy, as [`mutual_info`](@ref) states.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref), which validates `dims` and transposes when `dims` is `2`.
 2. Return [`mutual_info`](@ref) of the oriented matrix, under the binning algorithm `ce.bins` and the flag `ce.normalise`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:rho])

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`Int_Bin`](@ref)
  - [`dims_oriented`](@ref)
  - [`cov(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    return mutual_info(X, ce.bins, ce.normalise)
end

export MutualInfoCovariance
