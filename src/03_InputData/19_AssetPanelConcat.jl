"""
    vcat(a::AssetPanel, bs::AssetPanel...) -> AssetPanel

Concatenates Asset Panels of one universe along the observation axis, `a` first.

Each part holds the same Panel Fields, and the function joins the values and the observed mask of each field, the active mask and the estimation mask. It refuses a part whose schema differs from the schema of `a`. A static panel has no observation axis and states the same values at each observation, so the concatenation of equal static panels is `a`.

The panel holds no asset names and no timestamps. The [`ReturnsResult`](@ref) or the [`PricesResult`](@ref) that holds the panel holds them, and each of the two refuses a repeated timestamp. To join two blocks of returns data, concatenate `X`, `ts` and each other series with `vcat`, concatenate the panels with this function, and make a new `ReturnsResult`.

The element types promote as `vcat` promotes them: a `Float32` part and a `Float64` part give a `Float64` field. A field that each part lifts from the same static array stays a lazy [`RepeatedLeading`](@ref) array, and masks that are true everywhere stay an [`AllTrueMask`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. Check each part `b` against `a` with [`assert_panel_concat`](@ref): the shape, the asset count, the Panel Field names in their order, and the schema of each field.
 2. `a` is static: check that the Feature Matrix of each part, which [`panel_feature_matrix`](@ref) derives, equals the one of `a`. Return `a`.
 3. `a` is time-varying: concatenate the parts of each Panel Field with [`panel_field_vcat`](@ref), and the parts of each mask with [`panel_mask_vcat`](@ref). Make the `AssetPanel` of the results.

# Arguments

  - `a`: The panel of the first observations.
  - `bs`: The panels of the later observations, in the order of their observations.

# Validation

  - Every part is static, or every part is time-varying. An `ArgumentError` is thrown otherwise.
  - Every part has the asset count of `a`. A `DimensionMismatch` is thrown otherwise.
  - Every part has the Panel Field names of `a`, in the same order. An `ArgumentError` is thrown otherwise.
  - Each Panel Field has the kind of its field in `a`, and a categorical field has the same levels. A tensor field has the same axis name, labels and groups. An `ArgumentError` is thrown otherwise.
  - A static part equals `a`. An `ArgumentError` is thrown otherwise.

# Returns

  - `pnl::AssetPanel`: The panel of the observations of `a`, followed by the observations of each part of `bs`.

# Examples

```jldoctest
julia> a = AssetPanel(; pf = [NumericPanelField(; name = \"mcap\", vals = [1.0 2.0; 3.0 4.0])],
                      amsk = trues(2, 2), emsk = trues(2, 2));

julia> b = AssetPanel(; pf = [NumericPanelField(; name = \"mcap\", vals = [5.0 6.0])],
                      amsk = trues(1, 2), emsk = [true false]);

julia> pnl = vcat(a, b);

julia> panel_field(pnl, \"mcap\").vals
3×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0
 5.0  6.0

julia> pnl.emsk
3×2 BitMatrix:
 1  1
 1  1
 1  0
```

# Related

  - [`AssetPanel`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`port_opt_view`](@ref): the inverse operation, which takes observations out of a panel.
  - [`vcat_observations`](@ref)
"""
function Base.vcat(a::AssetPanel, bs::AssetPanel...)
    for (k, b) in enumerate(bs)
        assert_panel_concat(a, b, k + 1)
    end
    parts = (a, bs...)
    #! A panel with no Panel Field keeps its own empty vector, as `port_opt_view` does, so
    #! the field type it was made with stays.
    pf = if isempty(a.pf)
        a.pf
    else
        [panel_field_vcat(f, [p.pf[i] for p in parts]) for (i, f) in pairs(a.pf)]
    end
    return AssetPanel(; pf = pf, amsk = panel_mask_vcat([p.amsk for p in parts]),
                      emsk = panel_mask_vcat([p.emsk for p in parts]))
end
function Base.vcat(a::AssetPanel{<:Any, Nothing, Nothing}, bs::AssetPanel...)
    for (k, b) in enumerate(bs)
        assert_panel_concat(a, b, k + 1)
        @argcheck(isequal(panel_feature_matrix(a), panel_feature_matrix(b)),
                  ArgumentError("a static Asset Panel has no observation axis, so it states the same values at each observation, and a concatenation of static panels needs equal parts; part $(k + 1) holds values that differ from the ones of the first part"))
    end
    return a
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a part of a concatenation of Asset Panels whose schema differs from the schema of the first part.

# Algorithm

 1. Check that `b` is static when `a` is static, and time-varying when `a` is time-varying.
 2. Check that the asset axis of `b` has the length of the asset axis of `a`.
 3. Check that the Panel Field names of `b` equal the names of `a`, in the same order.
 4. Check each pair of fields with [`assert_panel_field_concat`](@ref).

# Arguments

  - `a`: The first part.
  - `b`: A later part.
  - `k`: The position of `b` in the concatenation, which the messages quote.

# Validation

  - `panel_is_static(a) == panel_is_static(b)`. An `ArgumentError` is thrown otherwise.
  - The two asset counts are equal. A `DimensionMismatch` is thrown otherwise.
  - The two lists of Panel Field names are equal. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`Base.vcat(a::AssetPanel, bs::AssetPanel...)`](@ref)
  - [`assert_panel_field_concat`](@ref)
"""
function assert_panel_concat(a::AssetPanel, b::AssetPanel, k::Integer)::Nothing
    @argcheck(panel_is_static(a) == panel_is_static(b),
              ArgumentError("a concatenation of Asset Panels joins parts of one shape: the first part is $(panel_is_static(a) ? "static" : "time-varying") and part $k is $(panel_is_static(b) ? "static" : "time-varying"). Lift the static panel onto its observations before the concatenation."))
    na = panel_axes(a)[end]
    nb = panel_axes(b)[end]
    @argcheck(na == nb,
              DimensionMismatch("the parts of a concatenation of Asset Panels are one universe, so each has the asset count of the first part; got $nb assets in part $k against $na"))
    fa = [f.name for f in a.pf]
    fb = [f.name for f in b.pf]
    @argcheck(isequal(fa, fb),
              ArgumentError("the parts of a concatenation of Asset Panels hold the same Panel Fields in the same order; got $(pinned_repr(fb)) in part $k against $(pinned_repr(fa))"))
    for (f, g) in zip(a.pf, b.pf)
        assert_panel_field_concat(f, g, k)
    end
    return nothing
end
"""
    assert_panel_field_concat(f::AbstractPanelField, g::AbstractPanelField, k::Integer) -> nothing
    assert_panel_field_concat(f::CategoricalPanelField, g::CategoricalPanelField, k::Integer) -> nothing
    assert_panel_field_concat(f::TensorPanelField, g::TensorPanelField, k::Integer) -> nothing

Refuses a Panel Field of a later part of a concatenation of Asset Panels whose schema differs from its field in the first part.

# Algorithm

The method that Julia selects is the algorithm.

 1. Two fields of different kinds: throw. Two numeric fields, or two fields of one kind that the library does not define, have no more schema to check.
 2. Two categorical fields: check that the levels are equal. A code indexes the levels of its own part, so a code over other levels names another category.
 3. Two tensor fields: check that the axis names, the labels and the groups are equal.

# Arguments

  - `f`: The field of the first part.
  - `g`: The field of the same name in a later part.
  - `k`: The position of the later part, which the messages quote.

# Validation

  - The two fields are of one kind. An `ArgumentError` is thrown otherwise.
  - Two categorical fields have equal levels. An `ArgumentError` is thrown otherwise.
  - Two tensor fields have equal axis names, labels and groups. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`assert_panel_concat`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
"""
function assert_panel_field_concat(f::AbstractPanelField, g::AbstractPanelField,
                                   k::Integer)::Nothing
    @argcheck(nameof(typeof(f)) === nameof(typeof(g)),
              ArgumentError("the Panel Field \"$(f.name)\" is a `$(nameof(typeof(g)))` in part $k and a `$(nameof(typeof(f)))` in the first part, and a concatenation joins the values of one kind of field only"))
    return nothing
end
function assert_panel_field_concat(f::CategoricalPanelField, g::CategoricalPanelField,
                                   k::Integer)::Nothing
    @argcheck(isequal(f.levels, g.levels),
              ArgumentError("the categorical Panel Field \"$(f.name)\" has the levels $(pinned_repr(g.levels)) in part $k against $(pinned_repr(f.levels)) in the first part. A code indexes the levels of its own part, so the codes of the two parts name different categories."))
    return nothing
end
function assert_panel_field_concat(f::TensorPanelField, g::TensorPanelField,
                                   k::Integer)::Nothing
    @argcheck(isequal(f.axis, g.axis) &&
              isequal(f.labels, g.labels) &&
              isequal(f.groups, g.groups),
              ArgumentError("the tensor Panel Field \"$(f.name)\" has the axis $(pinned_repr(g.axis)), the labels $(pinned_repr(g.labels)) and the groups $(pinned_repr(g.groups)) in part $k, against $(pinned_repr(f.axis)), $(pinned_repr(f.labels)) and $(pinned_repr(f.groups)) in the first part. Each entry of the trailing axis is one column of the derived Feature Matrix, so each part must hold the same entries."))
    return nothing
end
"""
    panel_field_vcat(f::NumericPanelField, fs::AbstractVector) -> NumericPanelField
    panel_field_vcat(f::CategoricalPanelField, fs::AbstractVector) -> CategoricalPanelField
    panel_field_vcat(f::TensorPanelField, fs::AbstractVector) -> TensorPanelField

Concatenates the parts of one time-varying Panel Field along the observation axis.

[`assert_panel_field_concat`](@ref) checked the schema of each part before the call, so the result takes the name, the levels, the axis name, the labels and the groups of `f`.

# Algorithm

The method that Julia selects is the algorithm. Each kind concatenates the value arrays of its parts with [`panel_values_vcat`](@ref), and their observed masks with [`panel_omsk_vcat`](@ref).

# Arguments

  - `f`: The field of the first part.
  - `fs`: The field of each part, `f` first, in the order of the observations.

# Returns

  - A Panel Field of the kind of `f`, over the observations of every part.

# Related

  - [`Base.vcat(a::AssetPanel, bs::AssetPanel...)`](@ref)
  - [`panel_values_vcat`](@ref)
  - [`panel_omsk_vcat`](@ref)
"""
function panel_field_vcat(f::NumericPanelField, fs::AbstractVector)
    vals = [g.vals for g in fs]
    return NumericPanelField(; name = f.name, vals = panel_values_vcat(vals),
                             omsk = panel_omsk_vcat([g.omsk for g in fs], vals))
end
function panel_field_vcat(f::CategoricalPanelField, fs::AbstractVector)
    codes = [g.codes for g in fs]
    return CategoricalPanelField(; name = f.name, levels = f.levels,
                                 codes = panel_values_vcat(codes),
                                 omsk = panel_omsk_vcat([g.omsk for g in fs], codes))
end
function panel_field_vcat(f::TensorPanelField, fs::AbstractVector)
    vals = [g.vals for g in fs]
    return TensorPanelField(; name = f.name, axis = f.axis, labels = f.labels,
                            groups = f.groups, vals = panel_values_vcat(vals),
                            omsk = panel_omsk_vcat([g.omsk for g in fs], vals))
end
"""
    panel_values_vcat(vs::AbstractVector{<:RepeatedLeading}) -> AbstractArray
    panel_values_vcat(vs::AbstractVector) -> AbstractArray

Concatenates the value arrays of the parts of one Panel Field along the observation axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. Every part is a [`RepeatedLeading`](@ref) array, and every part lifts an array equal to the one of the first part: return one lazy lift of that array over the observations of every part. Otherwise concatenate the parts with `vcat`.
 2. Any other parts: concatenate them with `vcat`.

# Arguments

  - `vs`: The value array of each part, in the order of the observations.

# Returns

  - `v::AbstractArray`: The values over the observations of every part.

# Related

  - [`panel_field_vcat`](@ref)
  - [`RepeatedLeading`](@ref)
"""
function panel_values_vcat(vs::AbstractVector{<:RepeatedLeading})
    p = first(vs).parent
    return if all(v -> isequal(v.parent, p), vs)
        RepeatedLeading(p, sum(v -> v.n, vs))
    else
        reduce(vcat, vs)
    end
end
function panel_values_vcat(vs::AbstractVector)
    return reduce(vcat, vs)
end
"""
    panel_omsk_vcat(ms::AbstractVector{Nothing}, vals::AbstractVector) -> Nothing
    panel_omsk_vcat(ms::AbstractVector, vals::AbstractVector) -> AbstractArray{Bool}

Concatenates the observed masks of the parts of one Panel Field along the observation axis.

A part with no observed mask cannot blank, so each of its cells was observed.

# Algorithm

The method that Julia selects is the algorithm.

 1. No part has an observed mask: return `nothing`.
 2. Otherwise: write an array of `true` values of the size of its values for each part with no mask, and concatenate the masks with `vcat`.

# Arguments

  - `ms`: The observed mask of each part, or `nothing`, in the order of the observations.
  - `vals`: The value array of each part, which gives the size of a missing mask.

# Returns

  - `omsk::Option{<:AbstractArray{Bool}}`: The observed mask over the observations of every part, or `nothing`.

# Related

  - [`panel_field_vcat`](@ref)
  - [`Option`](@ref)
"""
function panel_omsk_vcat(::AbstractVector{Nothing}, ::AbstractVector)
    return nothing
end
function panel_omsk_vcat(ms::AbstractVector, vals::AbstractVector)
    return reduce(vcat, [isnothing(m) ? trues(size(v)) : m for (m, v) in zip(ms, vals)])
end
"""
    panel_mask_vcat(ms::AbstractVector{<:AllTrueMask}) -> AllTrueMask
    panel_mask_vcat(ms::AbstractVector) -> AbstractMatrix{Bool}

Concatenates one universe mask of the parts of a time-varying Asset Panel along the observation axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. Every part is an [`AllTrueMask`](@ref): return one `AllTrueMask` over the observations of every part, which stores no cell.
 2. Any other parts: concatenate them with `vcat`.

# Arguments

  - `ms`: The mask of each part, in the order of the observations.

# Returns

  - `msk::AbstractMatrix{Bool}`: The mask over the observations of every part.

# Related

  - [`Base.vcat(a::AssetPanel, bs::AssetPanel...)`](@ref)
  - [`AllTrueMask`](@ref)
"""
function panel_mask_vcat(ms::AbstractVector{<:AllTrueMask})
    return AllTrueMask(sum(m -> m.n, ms), first(ms).N)
end
function panel_mask_vcat(ms::AbstractVector)
    return reduce(vcat, ms)
end
