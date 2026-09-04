"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Descriptor Estimator types.

A Descriptor Estimator produces one Descriptor: a per-asset value at every observation, computed from one or more Panel Fields of an Asset Panel. A point-in-time ratio of two fundamentals, the logarithm of a market capitalisation and the growth of a field over a lag are each one Descriptor. The estimator is configuration, so it names the Panel Fields it reads and holds no data.

All concrete types producing a Descriptor should be subtypes of `AbstractDescriptorEstimator`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractDescriptorEstimator` and implement the following methods:

## `descriptor`

  - [`descriptor(de::AbstractDescriptorEstimator, rd::ReturnsResult)`](@ref): Computes the Descriptor of a carrier.

### Arguments

  - `de`: The concrete subtype instance.
  - `rd`: The returns result that carries the Asset Panel.

### Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`, `NaN` wherever the active mask is `false`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`PanelFieldRatio`](@ref)
  - [`PanelFieldLog`](@ref)
  - [`Passthrough`](@ref)
  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`EWMean`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`DaysToCover`](@ref)
  - [`EWVolatility`](@ref)
  - [`RollingLogReturn`](@ref)
  - [`RollingMax`](@ref)
  - [`AssetPanel`](@ref)
"""
abstract type AbstractDescriptorEstimator <: AbstractEstimator end
"""
    descriptor(de::AbstractDescriptorEstimator, rd::ReturnsResult) -> Matrix{<:Real}

Compute the Descriptor of a carrier.

This is the verb every Descriptor Estimator answers. It reads the Panel Fields the estimator names from the carrier's feature matrix `rd.Z`, through the field index of `rd.pnl`, and never from the column names of `rd.nz`. Returns are not a Panel Field, so a member that reads them reads `rd.X`. Every member follows two conventions: the value at an observation uses information up to and including that observation, and every cell where the active mask of the Asset Panel is `false` is `NaN`.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`panel_field_values`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor end
"""
    panel_field_values(rd::ReturnsResult, name::AbstractString) -> Matrix{<:Real}
    panel_field_values(rd::ReturnsResult,
                       terms::AbstractVector{<:Pair{<:AbstractString, <:Real}}) -> Matrix{<:Real}

Read one numeric Panel Field, or a linear combination of numeric Panel Fields, out of a carrier.

This is the one route from a Panel Field's name to its values, and every Descriptor Estimator reads through it. A blank cell never reaches a carrier: [`asset_panel`](@ref) resolves each one to a fill value and records the resolution in the field's observed-mask column. The read undoes that resolution, so a cell the fill touched comes back as `NaN` and a Descriptor cannot mistake a fill value for data.

# Algorithm

 1. Look the Panel Field up by name through [`panel_field`](@ref), and copy its value column of `rd.Z` into a floating point matrix.
 2. When the Panel Field carries an observed-mask column, write `NaN` into every cell whose mask entry is zero.
 3. For a vector of `name => coefficient` pairs, read each named field the same way, and return the sum of the fields, each multiplied by its coefficient. A `NaN` in any term is a `NaN` in the sum.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `name`: The Panel Field's name.
  - `terms`: The Panel Fields to combine, each paired with its coefficient. `[\"a\" => 1, \"b\" => -1]` reads `a - b`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - `name` names a Panel Field, and its kind is [`NumericPanelField`](@ref). Raises a `KeyError` or an `ArgumentError`.
  - `!isempty(terms)`. Raises an [`IsEmptyError`](@ref).

# Returns

  - `V::Matrix{<:Real}`: The values, `observations × assets`, `NaN` where the Panel Field was not observed.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"mcap\", vals = [1.0 2.0; NaN 4.0],
                                            alg = ForwardPanelFill(; val = 0.0)),
                          NumericPanelInput(; name = \"debt\", vals = [0.5 1.0; 1.5 2.0])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), res...);

julia> PortfolioOptimisers.panel_field_values(rd, \"mcap\")
2×2 Matrix{Float64}:
   1.0  2.0
 NaN    4.0

julia> PortfolioOptimisers.panel_field_values(rd, [\"mcap\" => 1, \"debt\" => 1])
2×2 Matrix{Float64}:
   1.5  3.0
 NaN    6.0
```

# Related

  - [`descriptor`](@ref)
  - [`panel_field`](@ref)
  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`NumericPanelField`](@ref)
"""
function panel_field_values(rd::ReturnsResult, name::AbstractString)::Matrix{<:Real}
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Descriptor reads its Panel Fields through the field index of an Asset Panel, and rd.pnl is nothing. Build the carrier with the `pnl`, `nz` and `Z` that asset_panel returns."))
    f = panel_field(pnl, name)
    @argcheck(isa(f.kind, NumericPanelField),
              ArgumentError("a Descriptor reads one number per observation and asset, so the Panel Field \"$name\" must be a NumericPanelField, got a $(nameof(typeof(f.kind))), which occupies $(length(f.cols)) column(s) of the feature axis"))
    Z = rd.Z
    Tf = float(eltype(Z))
    V = Matrix{Tf}(view(Z, :, :, f.cols[1]))
    ocols = f.ocols
    if !isnothing(ocols)
        O = view(Z, :, :, ocols[1])
        for k in eachindex(V, O)
            if iszero(O[k])
                V[k] = Tf(NaN)
            end
        end
    end
    return V
end
function panel_field_values(rd::ReturnsResult,
                            terms::AbstractVector{<:Pair{<:AbstractString, <:Real}})::Matrix{<:Real}
    @argcheck(!isempty(terms),
              IsEmptyError("a Panel Field combination needs at least one `name => coefficient` term"))
    V = panel_field_values(rd, terms[1][1]) .* terms[1][2]
    for k in 2:length(terms)
        V .+= panel_field_values(rd, terms[k][1]) .* terms[k][2]
    end
    return V
end
"""
    descriptor_asset_panel(rd::ReturnsResult) -> AssetPanel

Read the Asset Panel a Descriptor needs out of a carrier.

[`panel_field_values`](@ref) reaches the panel through the name of a Panel Field, and every Descriptor that reads one meets its refusal. A Descriptor over the returns reads no Panel Field, so it takes this route to the same refusal and to the active mask [`descriptor_active_fill!`](@ref) needs.

# Arguments

  - $(arg_dict[:rd])

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel the carrier holds.

# Related

  - [`descriptor`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`panel_field_values`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_asset_panel(rd::ReturnsResult)::AssetPanel
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Descriptor is `NaN` wherever the active mask of an Asset Panel is `false`, and rd.pnl is nothing. Build the carrier with the `pnl`, `nz` and `Z` that asset_panel returns."))
    return pnl
end
"""
    assert_log_returns(X::AbstractMatrix{<:Real}) -> nothing

Check that every return that is not missing is greater than `-1`.

A Descriptor that compounds returns takes the logarithm of one plus each return. A return of `-1` is a total loss, and the logarithm is undefined below it, so the check refuses the whole matrix rather than write an infinity into one cell of the Descriptor. A missing return is a `NaN`, and it passes the check. Every Descriptor that reads `log1p(rd.X)` runs it, exponentially weighted and rolling alike.

# Arguments

  - `X`: The returns, `observations × assets`.

# Validation

  - Every entry of `X` that is not `NaN` is greater than `-1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`descriptor`](@ref)
  - [`EWMean`](@ref)
  - [`RollingLogReturn`](@ref)
"""
function assert_log_returns(X::AbstractMatrix{<:Real})::Nothing
    k = findfirst(x -> !isnan(x) && x <= -one(x), X)
    @argcheck(isnothing(k),
              DomainError(isnothing(k) ? NaN : X[k],
                          "a Descriptor over log returns takes the logarithm of one plus each return, so every return that is not missing must be greater than -1, and it is $(isnothing(k) ? NaN : X[k]) at observation $(isnothing(k) ? 0 : k[1]) for asset $(isnothing(k) ? 0 : k[2]). A return at or below -1 is a data error, so clean the input rather than pass it through."))
    return nothing
end
"""
    descriptor_active_fill!(D::AbstractMatrix{<:Real}, pnl::AssetPanel) -> nothing

Write `NaN` into every cell of a Descriptor where the active mask of the Asset Panel is `false`, in place.

Every Descriptor Estimator ends with this call, so the convention that an inactive cell is `NaN` is written once. An asset that is not listed at an observation has no Descriptor there, whatever its Panel Fields hold.

# Arguments

  - `D`: The Descriptor, `observations × assets`, changed in place.
  - `pnl`: The Asset Panel whose active mask is read.

# Validation

  - `size(D) == size(pnl.amsk)`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`. `D` carries the filled Descriptor.

# Examples

```jldoctest
julia> pnl = AssetPanel(; pf = [PanelField(; name = \"a\", kind = NumericPanelField(), cols = [1])],
                        amsk = [true false; true true], emsk = [true false; true true]);

julia> D = [1.0 2.0; 3.0 4.0];

julia> PortfolioOptimisers.descriptor_active_fill!(D, pnl)

julia> D
2×2 Matrix{Float64}:
 1.0  NaN
 3.0    4.0
```

# Related

  - [`descriptor`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_active_fill!(D::AbstractMatrix{<:Real}, pnl::AssetPanel)::Nothing
    amsk = pnl.amsk
    @argcheck(size(D) == size(amsk),
              DimensionMismatch("a Descriptor is observations × assets, so it must match the active mask of the Asset Panel, got size(D) = $(size(D)) and size(pnl.amsk) = $(size(amsk))"))
    Tf = eltype(D)
    for k in CartesianIndices(D)
        if !amsk[k]
            D[k] = Tf(NaN)
        end
    end
    return nothing
end
"""
    positive_divide(a::Real, b::Real) -> Real

Divide `a` by `b` where `b` is strictly positive, and return `NaN` otherwise.

A ratio Descriptor is undefined where its denominator is zero, and it is meaningless where a quantity that is positive by construction, a market capitalisation or a total of assets, is negative. Both cases answer `NaN` rather than a number or an error, so one bad cell costs one cell of the Descriptor and not the whole fit. A `NaN` denominator compares `false` against zero, so it also answers `NaN`.

# Arguments

  - `a`: The numerator.
  - `b`: The denominator.

# Returns

  - `q::Real`: `a / b` when `b > 0`, `NaN` otherwise, in the floating point type of the quotient.

# Examples

```jldoctest
julia> PortfolioOptimisers.positive_divide(1.0, 4.0)
0.25

julia> PortfolioOptimisers.positive_divide(1.0, 0.0)
NaN

julia> PortfolioOptimisers.positive_divide(1.0, -2.0)
NaN
```

# Related

  - [`descriptor`](@ref)
  - [`PanelFieldRatio`](@ref)
"""
function positive_divide(a::Real, b::Real)::Real
    q = a / b
    return b > zero(b) ? q : oftype(q, NaN)
end

export descriptor
