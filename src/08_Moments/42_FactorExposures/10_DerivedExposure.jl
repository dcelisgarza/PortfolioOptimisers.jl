"""
$(DocStringExtensions.TYPEDEF)

A Factor Exposure derived from the Factor Exposure of another factor.

The member applies a function to the exposure of the factor it names, then transforms the result cross-sectionally. A non-linear size factor is the standard case: the square of a size exposure is a factor of its own, and it is neutralised against the size factor it was derived from.

The member reads no Panel Field of its own, so it cannot compute its source. The caller that holds the factor list computes the factors in dependency order and passes the source exposure to the three-argument [`factor_exposure`](@ref) method. The two-argument method refuses, naming the source.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DerivedExposure(;
        source::AbstractString,
        f,
        outlier::Option{<:AbstractCrossSectionalTransform} = nothing,
        scoring::Option{<:AbstractCrossSectionalTransform} = CrossSectionalStandardiser(),
        group::Option{<:AbstractString} = nothing,
        bw::AbstractString = "benchmark_weights",
        family::AbstractString = "style"
    ) -> DerivedExposure

Keywords correspond to the struct's fields. The outlier slot defaults to `nothing` and not to a winsoriser, because the source exposure was already transformed when it was computed.

## Validation

  - `!isempty(source)`.
  - `!isempty(group)`, `!isempty(bw)` and `!isempty(family)`.

# Examples

```jldoctest
julia> DerivedExposure(; source = \"size\", f = abs2, scoring = nothing)
DerivedExposure
   source ┼ String: \"size\"
        f ┼ typeof(abs2): abs2
  outlier ┼ nothing
  scoring ┼ nothing
    group ┼ nothing
       bw ┼ String: \"benchmark_weights\"
   family ┴ String: \"style\"
```

# Related

  - [`AbstractExposureEstimator`](@ref)
  - [`factor_exposure`](@ref)
  - [`CompositeExposure`](@ref)
  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
"""
@concrete struct DerivedExposure <: AbstractExposureEstimator
    """
    Name of the factor whose Factor Exposure this one is derived from.
    """
    source
    """
    Function applied to the source Factor Exposure. It takes an `observations × assets` matrix and returns one of the same size.
    """
    f
    """
    Cross-sectional transform applied to the derived exposure, or `nothing` to skip the step.
    """
    outlier
    """
    Cross-sectional transform applied after the outlier step, or `nothing` to skip the step.
    """
    scoring
    """
    Name of the categorical Panel Field the transforms are applied within, or `nothing` to transform each observation as one cross-section.
    """
    group
    """
    Name of the numeric Panel Field holding the benchmark weights the transforms are weighted by.
    """
    bw
    """
    Label of the Factor Family the factor belongs to.
    """
    family
    function DerivedExposure(source::AbstractString, f,
                             outlier::Option{<:AbstractCrossSectionalTransform},
                             scoring::Option{<:AbstractCrossSectionalTransform},
                             group::Option{<:AbstractString}, bw::AbstractString,
                             family::AbstractString)
        assert_panel_terms(source, :source)
        if !isnothing(group)
            assert_panel_terms(group, :group)
        end
        assert_panel_terms(bw, :bw)
        assert_exposure_family(family)
        return new{typeof(source), typeof(f), typeof(outlier), typeof(scoring),
                   typeof(group), typeof(bw), typeof(family)}(source, f, outlier, scoring,
                                                              group, bw, family)
    end
end
function DerivedExposure(; source::AbstractString, f,
                         outlier::Option{<:AbstractCrossSectionalTransform} = nothing,
                         scoring::Option{<:AbstractCrossSectionalTransform} = CrossSectionalStandardiser(),
                         group::Option{<:AbstractString} = nothing,
                         bw::AbstractString = "benchmark_weights",
                         family::AbstractString = "style")::DerivedExposure
    return DerivedExposure(source, f, outlier, scoring, group, bw, family)
end
"""
    factor_exposure(xe::DerivedExposure, rd::ReturnsResult) -> Union{}
    factor_exposure(xe::DerivedExposure, rd::ReturnsResult, xs::MatNum) -> Matrix{<:Real}

Compute the Factor Exposure derived from the Factor Exposure of another factor.

# Algorithm

 1. Read the benchmark weights and the group labels off the carrier.
 2. Apply `f` to the source exposure, and check that it kept the shape.
 3. Apply the outlier slot and then the scoring slot.

# Arguments

  - `xe`: Derived Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `xs`: Factor Exposure of the source factor, `observations × assets`.

# Validation

  - `xs` is `observations × assets`. Raises a `DimensionMismatch`.
  - `f` returns a matrix of the size it was given. Raises a `DimensionMismatch`.
  - The two-argument method always raises an `ArgumentError`, because a derived exposure has no source to read.

# Returns

  - `L::Matrix{<:Real}`: The Factor Exposure, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"benchmark_weights\",
                                            vals = [1.0 1.0; 1.0 1.0])]; amsk = trues(2, 2),
                         emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> xe = DerivedExposure(; source = \"size\", f = x -> x .^ 2, scoring = nothing);

julia> factor_exposure(xe, rd, [1.0 2.0; 3.0 4.0])
2×2 Matrix{Float64}:
 1.0   4.0
 9.0  16.0
```

# Related

  - [`DerivedExposure`](@ref)
  - [`AbstractExposureEstimator`](@ref)
  - [`exposure_transform`](@ref)
  - [`CompositeExposure`](@ref)
"""
function factor_exposure(xe::DerivedExposure, ::ReturnsResult)
    return throw(ArgumentError("a derived Factor Exposure is computed from the Factor Exposure of the factor \"$(xe.source)\", which it cannot read from the carrier. The caller that holds the factor list computes the factors in dependency order, and passes the source exposure to the three-argument method factor_exposure(xe, rd, xs)"))
end
function factor_exposure(xe::DerivedExposure, rd::ReturnsResult, xs::MatNum)::Matrix{<:Real}
    w = exposure_benchmark_weights(rd, xe.bw)
    @argcheck(size(xs) == size(w),
              DimensionMismatch("the source Factor Exposure is observations × assets, like the Asset Panel it was computed on, got size(xs) = $(size(xs)) and $(size(w))"))
    groups = exposure_group_labels(rd, xe.group)
    D0 = xe.f(xs)
    @argcheck(size(D0) == size(xs),
              DimensionMismatch("f maps one Factor Exposure to another, so it must return a matrix of the size it was given, got $(size(D0)) from size(xs) = $(size(xs))"))
    Tf = float(promote_type(eltype(D0), eltype(xs)))
    D = exposure_transform(xe.outlier, convert(Matrix{Tf}, D0), w, groups)
    return exposure_transform(xe.scoring, D, w, groups)
end

export DerivedExposure
