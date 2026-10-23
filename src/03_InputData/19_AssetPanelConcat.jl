"""
    vcat(a::AssetPanel, bs::AssetPanel...) -> AssetPanel

Concatenates Asset Panels of one universe along the observation axis, `a` first.

Each part holds the same Panel Fields. The function joins the values, the observed mask and the placeholder mask of each field, and it joins the active mask and the estimation mask. It refuses a part whose schema differs from the schema of `a`. A static panel has no observation axis and states the same values at each observation, so the concatenation of equal static panels is `a`.

The panel holds no asset names and no timestamps. The [`ReturnsResult`](@ref) or the [`PricesResult`](@ref) that holds the panel holds them, and each of the two refuses a repeated timestamp. To join two blocks of returns data, concatenate `X`, `ts` and each other series with `vcat`. Then concatenate the panels with this function, and make a new `ReturnsResult`.

The element types promote as `vcat` promotes them, so a `Float32` part and a `Float64` part give a `Float64` field. A field that each part lifts from an equal static array stays a lazy [`RepeatedLeading`](@ref) array, and masks that are true everywhere stay an [`AllTrueMask`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. Check each part `b` against `a` with [`assert_panel_concat`](@ref): the shape, the asset count, the Panel Field names in their order, and the schema of each field.
 2. `a` is static: check that each Panel Field of each part holds the data of its field in `a`, with [`panel_field_isequal`](@ref). A field with no observed mask equals a field whose observed mask is `true` everywhere. A field with no placeholder mask equals a field whose placeholder mask is `false` everywhere, as in step 3. Return `a`.
 3. `a` is time-varying: concatenate the parts of each Panel Field with [`panel_field_vcat`](@ref), and the parts of each mask with [`panel_mask_vcat`](@ref).
 4. Make the `AssetPanel` of the joined Panel Fields and the two joined masks.

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
  - [`panel_field_isequal`](@ref)
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
        @argcheck(all(i -> panel_field_isequal(a.pf[i], b.pf[i]), eachindex(a.pf)),
                  ArgumentError("a static Asset Panel has no observation axis, so it states the same values at each observation, and a concatenation of static panels needs equal parts; part $(k + 1) holds values, observed masks or placeholder masks that differ from the ones of the first part. A missing observed mask reads as true in each cell, and a missing placeholder mask as false"))
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
  - The rules of [`assert_panel_field_concat`](@ref) for each pair of fields.

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

 1. Any other two fields: check that the two fields are of one kind, by the name of their type. Two numeric fields, and two fields of a kind that the library does not define, have no more schema to check.
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

[`assert_panel_field_concat`](@ref) checks the schema of each part before the call, so the result takes the name, the levels, the axis name, the labels and the groups of `f`.

# Algorithm

Each method takes the same steps for its kind of field.

 1. Concatenate the value arrays of the parts with [`panel_values_vcat`](@ref). For a categorical field, the value arrays are the codes.
 2. Concatenate the observed masks of the parts with [`panel_omsk_vcat`](@ref), which reads the value array of a part for the size of a missing mask.
 3. Concatenate the placeholder masks of the parts with [`panel_pmsk_vcat`](@ref) in the same way.
 4. Make a Panel Field of the kind of `f` from the results, with the name of `f` and the rest of its schema.

# Arguments

  - `f`: The field of the first part.
  - `fs`: The field of each part, `f` first, in the order of the observations.

# Returns

  - A Panel Field of the kind of `f`, over the observations of every part.

# Related

  - [`Base.vcat(a::AssetPanel, bs::AssetPanel...)`](@ref)
  - [`panel_values_vcat`](@ref)
  - [`panel_omsk_vcat`](@ref)
  - [`panel_pmsk_vcat`](@ref)
"""
function panel_field_vcat(f::NumericPanelField, fs::AbstractVector)
    vals = [g.vals for g in fs]
    return NumericPanelField(; name = f.name, vals = panel_values_vcat(vals),
                             omsk = panel_omsk_vcat([g.omsk for g in fs], vals),
                             pmsk = panel_pmsk_vcat([g.pmsk for g in fs], vals))
end
function panel_field_vcat(f::CategoricalPanelField, fs::AbstractVector)
    codes = [g.codes for g in fs]
    return CategoricalPanelField(; name = f.name, levels = f.levels,
                                 codes = panel_values_vcat(codes),
                                 omsk = panel_omsk_vcat([g.omsk for g in fs], codes),
                                 pmsk = panel_pmsk_vcat([g.pmsk for g in fs], codes))
end
function panel_field_vcat(f::TensorPanelField, fs::AbstractVector)
    vals = [g.vals for g in fs]
    return TensorPanelField(; name = f.name, axis = f.axis, labels = f.labels,
                            groups = f.groups, vals = panel_values_vcat(vals),
                            omsk = panel_omsk_vcat([g.omsk for g in fs], vals),
                            pmsk = panel_pmsk_vcat([g.pmsk for g in fs], vals))
end
"""
    panel_values_vcat(vs::AbstractVector{<:RepeatedLeading}) -> AbstractArray
    panel_values_vcat(vs::AbstractVector) -> AbstractArray

Concatenates the value arrays of the parts of one Panel Field along the observation axis.

The element type of the result is the element type that `vcat` of the parts gives. A lift of `Float32` values and an equal lift of `Float64` values give a lazy lift of `Float64` values.

# Algorithm

The method that Julia selects is the algorithm.

 1. Every part is a [`RepeatedLeading`](@ref) array:
     1. Read the lifted array `p` of the first part. When a part lifts an array that `isequal` does not find equal to `p`, go to step 2.
     2. Find the element type `T` with `promote_type` over the element types of the parts.
     3. Return one lazy lift of `p`, converted to `T`, over the observations of every part.
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
        #! `isequal` holds across element types, as for `[1, 2]` and `[1.0, 2.0]`, so the lift
        #! takes the element type that `vcat` of the parts gives.
        T = mapreduce(eltype, promote_type, vs)
        RepeatedLeading(convert(AbstractArray{T}, p), sum(v -> v.n, vs))
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

A part with no observed mask cannot blank, so each of its cells is observed.

# Algorithm

The method that Julia selects is the algorithm.

 1. No part has an observed mask: return `nothing`.
 2. Any other parts: put an array of `true` values, of the size of the values of the part, in place of each missing mask.
 3. Concatenate the masks with `vcat`.

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
    panel_pmsk_vcat(ms::AbstractVector{Nothing}, vals::AbstractVector) -> Nothing
    panel_pmsk_vcat(ms::AbstractVector, vals::AbstractVector) -> AbstractArray{Bool}

Concatenates the placeholder masks of the parts of one Panel Field along the observation axis.

A part with no placeholder mask holds no placeholder, so each of its cells holds data.

# Algorithm

The method that Julia selects is the algorithm.

 1. No part has a placeholder mask: return `nothing`.
 2. Any other parts: put an array of `false` values, of the size of the values of the part, in place of each missing mask.
 3. Concatenate the masks with `vcat`.

# Arguments

  - `ms`: The placeholder mask of each part, or `nothing`, in the order of the observations.
  - `vals`: The value array of each part, which gives the size of a missing mask.

# Returns

  - `pmsk::Option{<:AbstractArray{Bool}}`: The placeholder mask over the observations of every part, or `nothing`.

# Related

  - [`panel_field_vcat`](@ref)
  - [`panel_omsk_vcat`](@ref)
  - [`Option`](@ref)
"""
function panel_pmsk_vcat(::AbstractVector{Nothing}, ::AbstractVector)
    return nothing
end
function panel_pmsk_vcat(ms::AbstractVector, vals::AbstractVector)
    return reduce(vcat, [isnothing(m) ? falses(size(v)) : m for (m, v) in zip(ms, vals)])
end
"""
    panel_field_isequal(f::AbstractPanelField, g::AbstractPanelField) -> Bool

Tests whether two Panel Fields of one schema hold the same data: the same stored cells, the same observed mask and the same placeholder mask.

The function reads a missing mask as the concatenation along the observation axis reads it. A field with no observed mask holds data in each cell, as a mask that is `true` everywhere states. A field with no placeholder mask holds no placeholder, as a mask that is `false` everywhere states. [`assert_panel_concat`](@ref) checks the schema before the call, which is the kind, the levels, the axis name, the labels and the groups.

# Algorithm

 1. Compare the stored arrays, which [`panel_field_array`](@ref) reads, with `isequal`.
 2. Compare the observed masks with [`panel_mask_isequal`](@ref), with `true` for a missing mask.
 3. Compare the placeholder masks with [`panel_mask_isequal`](@ref), with `false` for a missing mask.

# Arguments

  - `f`: The field of the first part.
  - `g`: The field of the same name in a later part.

# Returns

  - `flag::Bool`: `true` when the two fields hold the same data.

# Related

  - [`Base.vcat(a::AssetPanel, bs::AssetPanel...)`](@ref)
  - [`panel_omsk_vcat`](@ref)
  - [`panel_pmsk_vcat`](@ref)
"""
function panel_field_isequal(f::AbstractPanelField, g::AbstractPanelField)::Bool
    return isequal(panel_field_array(f), panel_field_array(g)) &&
           panel_mask_isequal(f.omsk, g.omsk, true) &&
           panel_mask_isequal(f.pmsk, g.pmsk, false)
end
"""
    panel_mask_isequal(m::Nothing, n::Nothing, fill::Bool) -> Bool
    panel_mask_isequal(m::AbstractArray{Bool}, n::Nothing, fill::Bool) -> Bool
    panel_mask_isequal(m::Nothing, n::AbstractArray{Bool}, fill::Bool) -> Bool
    panel_mask_isequal(m::AbstractArray{Bool}, n::AbstractArray{Bool}, fill::Bool) -> Bool

Tests whether two masks of one Panel Field are equal, with `fill` in each cell of a missing mask.

The caller compares the stored arrays of the two fields first. So a mask has the shape of the values, and a missing mask needs no array to compare.

# Algorithm

The method that Julia selects is the algorithm.

 1. Two missing masks: return `true`.
 2. One missing mask: return whether each cell of the other mask is `fill`.
 3. Two masks: compare them with `isequal`.

# Arguments

  - `m`: The mask of the first field, or `nothing`.
  - `n`: The mask of the second field, or `nothing`.
  - `fill`: The value of each cell of a missing mask.

# Returns

  - `flag::Bool`: `true` when the two masks are equal.

# Related

  - [`panel_field_isequal`](@ref)
  - [`Option`](@ref)
"""
function panel_mask_isequal(::Nothing, ::Nothing, ::Bool)::Bool
    return true
end
function panel_mask_isequal(m::AbstractArray{Bool}, ::Nothing, fill::Bool)::Bool
    return all(==(fill), m)
end
function panel_mask_isequal(::Nothing, n::AbstractArray{Bool}, fill::Bool)::Bool
    return all(==(fill), n)
end
function panel_mask_isequal(m::AbstractArray{Bool}, n::AbstractArray{Bool}, ::Bool)::Bool
    return isequal(m, n)
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
"""
    panel_fields_panel(pf::AbstractVector{<:AbstractPanelField}) -> AssetPanel

Makes the time-varying Asset Panel that carries the Panel Fields of a sample buffer, so that the checks and the `vcat` of an [`AssetPanel`](@ref) apply to them.

A [`SampleBufferState`](@ref) keeps its active mask in `A`, its estimation mask in `M`, and the Panel Fields alone in `P`. The masks of this panel are two [`AllTrueMask`](@ref) arrays, which store no cell. They are not the masks of the buffer, and no caller reads them. The constructor of the panel checks that the Panel Fields share one observation axis and one asset axis, and that their names are unique.

# Algorithm

 1. Refuse an empty vector.
 2. Refuse a vector whose first Panel Field is static. The constructor of the panel refuses a later static field, because its axes differ from the axes of the first field.
 3. Read the observation count `T` and the asset count `N` from the stored array of the first Panel Field.
 4. Make the `AssetPanel` of the Panel Fields and of two `AllTrueMask` arrays of size `(T, N)`.

# Arguments

  - `pf`: The Panel Fields.

# Validation

  - `pf` is not empty. An `IsEmptyError` is thrown otherwise.
  - Each Panel Field is time-varying. A `DimensionMismatch` is thrown otherwise.
  - The rules of the [`AssetPanel`](@ref) constructor.

# Returns

  - `pnl::AssetPanel`: The time-varying panel of the Panel Fields.

# Related

  - [`SampleBufferState`](@ref)
  - [`AllTrueMask`](@ref)
  - [`panel_fields_append`](@ref)
"""
function panel_fields_panel(pf::AbstractVector{<:AbstractPanelField})
    @argcheck(!isempty(pf),
              IsEmptyError("a sample buffer records the Panel Fields of a time-varying Asset Panel, and this vector holds none. Pass `nothing` for a fold with no Panel Field."))
    ax = panel_field_axes(first(pf))
    @argcheck(length(ax) == 2,
              DimensionMismatch("a sample buffer records Panel Fields row by row, so each one carries an observation axis, and \"$(first(pf).name)\" is static, of shape $ax. Lift it onto the observations with `panel_field_lift`, or leave it out of the fold."))
    #! The two axes are read as sizes, not as entries of `ax`: the type of `ax` does not
    #! state its length, and JET reports an index into a tuple that can be empty.
    A = panel_field_array(first(pf))
    T, N = size(A, 1), size(A, 2)
    return AssetPanel(; pf = pf, amsk = AllTrueMask(T, N), emsk = AllTrueMask(T, N))
end
"""
    assert_buffer_panel_shape(X::AbstractMatrix, n::Integer, P::Nothing) -> nothing
    assert_buffer_panel_shape(X::AbstractMatrix, n::Integer, P::AbstractVector{<:AbstractPanelField}) -> nothing

Refuses Panel Fields that do not cover the valid region of a sample buffer.

The keyword constructor of [`SampleBufferState`](@ref) calls it. `P` holds the valid region alone, so its observation axis has `n` rows, and its asset axis is the width of `X`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `P` is `nothing`: return.
 2. Make the panel of `P` with [`panel_fields_panel`](@ref), and refuse axes that are not `(n, size(X, 2))`.

# Arguments

  - `X`: The backing matrix of the buffer.
  - `n`: The number of observations in the valid region.
  - `P`: The Panel Fields of the buffer, or `nothing`.

# Validation

  - The rules of [`panel_fields_panel`](@ref).
  - The axes of `P` are `(n, size(X, 2))`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`panel_fields_panel`](@ref)
"""
function assert_buffer_panel_shape(::AbstractMatrix, ::Integer, ::Nothing)
    return nothing
end
function assert_buffer_panel_shape(X::AbstractMatrix, n::Integer,
                                   P::AbstractVector{<:AbstractPanelField})
    ax = panel_axes(panel_fields_panel(P))
    @argcheck(ax == (n, size(X, 2)),
              DimensionMismatch("the Panel Fields of a sample buffer cover its valid region, so they are $((n, size(X, 2))) observations × assets, got $ax."))
    return nothing
end
"""
    assert_buffer_panel_block(Xo::AbstractMatrix, Ao, Eo, Po::Nothing) -> nothing
    assert_buffer_panel_block(Xo::AbstractMatrix, Ao, Eo, Po::AbstractVector{<:AbstractPanelField}) -> nothing

Refuses Panel Fields that do not make a time-varying Asset Panel over a block of a sample buffer.

The method of [`partial_fit!`](@ref) for a block and a [`SampleBufferState`](@ref) calls it before it records a row. The Panel Fields of a block and the masks of the block are one time-varying [`AssetPanel`](@ref). So the Panel Fields need both masks, and they share the observations and the assets of the block.

# Algorithm

The method that Julia selects is the algorithm.

 1. `Po` is `nothing`: return.
 2. Make the panel of `Po` with [`panel_fields_panel`](@ref), and refuse axes that are not the size of `Xo`.
 3. Refuse a block that gives Panel Fields without both masks.

# Arguments

  - `Xo`: The oriented block.
  - `Ao`: The oriented active mask of the block, or `nothing`.
  - `Eo`: The oriented estimation mask of the block, or `nothing`.
  - `Po`: The Panel Fields of the block, or `nothing`.

# Validation

  - The rules of [`panel_fields_panel`](@ref).
  - The axes of `Po` are `size(Xo)`. A `DimensionMismatch` is thrown otherwise.
  - `Ao` and `Eo` are not `nothing` when `Po` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`panel_fields_panel`](@ref)
  - [`partial_fit!`](@ref)
"""
function assert_buffer_panel_block(::AbstractMatrix, ::Any, ::Any, ::Nothing)
    return nothing
end
function assert_buffer_panel_block(Xo::AbstractMatrix, Ao, Eo,
                                   Po::AbstractVector{<:AbstractPanelField})
    ax = panel_axes(panel_fields_panel(Po))
    @argcheck(ax == size(Xo),
              DimensionMismatch("the Panel Fields of a fold describe the observations and the assets of its block, so they are $(size(Xo)) observations × assets, got $ax."))
    @argcheck(!isnothing(Ao) && !isnothing(Eo),
              ArgumentError("the Panel Fields of a fold belong to a time-varying Asset Panel, which carries both masks, and this fold gives $(isnothing(Ao) ? "no active mask" : "no estimation mask"). Fold the Panel Fields with the `amsk` and the `emsk` of their panel."))
    return nothing
end
"""
    panel_fields_view(P::Nothing, i, j) -> nothing
    panel_fields_view(P::AbstractVector{<:AbstractPanelField}, i, j) -> Vector{AbstractPanelField}

Views each Panel Field of a sample buffer over the observations `i` and the assets `j`, or gives `nothing` for a buffer that records no Panel Field.

The method of [`partial_fit!`](@ref) for a block and a [`SampleBufferState`](@ref) takes the rows that a cap keeps with it. [`port_opt_view`](@ref) of the buffer takes the selected assets with it. The function passes no asset names to [`panel_field_view`](@ref), so a tensor Panel Field keeps its whole label axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. `P` is `nothing`: return `nothing`.
 2. Any other `P`: view each Panel Field over `i` and `j` with [`panel_field_view`](@ref), and collect the views in a `Vector{AbstractPanelField}`.

# Arguments

  - `P`: The Panel Fields, or `nothing`.
  - `i`: The observation index.
  - `j`: The asset index.

# Returns

  - `P′`: The viewed Panel Fields, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`panel_field_view`](@ref)
"""
function panel_fields_view(::Nothing, ::Any, ::Any)
    return nothing
end
function panel_fields_view(P::AbstractVector{<:AbstractPanelField}, i, j)
    return AbstractPanelField[panel_field_view(f, i, j, nothing) for f in P]
end
"""
    panel_fields_append(P::Nothing, Po::Nothing, n::Integer) -> nothing
    panel_fields_append(P::AbstractVector{<:AbstractPanelField}, Po::AbstractVector{<:AbstractPanelField}, n::Integer) -> AbstractVector

Joins the Panel Fields of a sample buffer to the Panel Fields of a block, and keeps the last `n` rows.

The method of [`partial_fit!`](@ref) for a block and a [`SampleBufferState`](@ref) calls it, and so does [`merge_states`](@ref) for two buffers. `n` is the number of observations of the buffer after the append, and the cap sets it before the call. The function keeps every row of `Po`, because a cap truncates a block before the append. `vcat` of two Asset Panels joins the rows, so it checks that the two lists hold the same Panel Fields.

# Algorithm

The method that Julia selects is the algorithm.

 1. `P` and `Po` are `nothing`: return `nothing`.
 2. Make the panel `b` of `Po` with [`panel_fields_panel`](@ref).
 3. Check that the panel of `P` and `b` can join, with [`assert_panel_concat`](@ref).
 4. Set `keep = n - tₒ`, where `tₒ` is the number of rows of `Po`. Return `Po` when `keep` is zero.
 5. When `keep` is less than the number of rows `tₚ` of `P`, view the last `keep` rows of `P` with [`panel_fields_view`](@ref). Otherwise keep all of `P`.
 6. Join the kept rows of `P` and `b` with `vcat`, and return the Panel Fields of the joined panel.

# Arguments

  - `P`: The Panel Fields of the buffer, or `nothing`.
  - `Po`: The Panel Fields of the block, or `nothing`.
  - `n`: The number of observations of the buffer after the append.

# Validation

  - The rules of [`panel_fields_panel`](@ref) and of [`assert_panel_concat`](@ref).

# Returns

  - `P′`: The Panel Fields of the last `n` observations, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`Base.vcat(a::AssetPanel, bs::AssetPanel...)`](@ref)
  - [`panel_fields_view`](@ref)
"""
function panel_fields_append(::Nothing, ::Nothing, ::Integer)
    return nothing
end
function panel_fields_append(P::AbstractVector{<:AbstractPanelField},
                             Po::AbstractVector{<:AbstractPanelField}, n::Integer)
    b = panel_fields_panel(Po)
    assert_panel_concat(panel_fields_panel(P), b, 2)
    tp = size(panel_field_array(first(P)), 1)
    keep = n - size(panel_field_array(first(Po)), 1)
    if iszero(keep)
        return Po
    end
    Pk = keep < tp ? panel_fields_view(P, (tp - keep + 1):tp, :) : P
    return vcat(panel_fields_panel(Pk), b).pf
end
"""
    sample_buffer_panel(state::SampleBufferState) -> Option{<:AssetPanel}

Rebuilds the time-varying Asset Panel of the observations that a sample buffer holds, or gives `nothing` when the buffer records no Panel Field.

A fit over the rows of a buffer reads it, for example the call with no data of a refit prior, and the Choice Rule of an online step in [`pin_buffer_choice`](@ref). [`buffer_asset_panel`](@ref) reads it for a buffer that records Panel Fields. A buffer that records Panel Fields records both masks too, so the panel has its Panel Fields and both of its masks. The masks are copies of the valid region, because a later fold writes into the backing masks. The Panel Fields are the Panel Fields of the buffer, because nothing writes into them.

# Algorithm

The method that Julia selects is the algorithm.

 1. `state.P` is `nothing`: return `nothing`.
 2. Any other `state.P`: copy the active mask `A` and the estimation mask `M` over the valid region into two `Matrix{Bool}`.
 3. Make the `AssetPanel` of `state.P` and the two masks.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `pnl::Option{<:AssetPanel}`: The panel of the rows of the buffer, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`returns_result`](@ref)
  - [`reads_panel_fields`](@ref)
"""
function sample_buffer_panel(state::SampleBufferState)
    return sample_buffer_panel(state.P, state)
end
function sample_buffer_panel(::Nothing, ::SampleBufferState)
    return nothing
end
function sample_buffer_panel(P::AbstractVector{<:AbstractPanelField},
                             state::SampleBufferState)
    rows = (state.off + 1):(state.off + state.n)
    #! A buffer that records `P` records both masks, which the fold checks. The assertions
    #! state it to the compiler, so the constructor meets two concrete masks.
    return AssetPanel(; pf = P,
                      amsk = Matrix{Bool}(view(state.A::AbstractMatrix{Bool}, rows, :)),
                      emsk = Matrix{Bool}(view(state.M::AbstractMatrix{Bool}, rows, :)))
end
"""
    sample_buffer_fit_kwargs(state::SampleBufferState, pnl::Nothing) -> NamedTuple
    sample_buffer_fit_kwargs(state::SampleBufferState, pnl::AssetPanel) -> NamedTuple

Gives the keywords that a fit over the rows of a sample buffer passes to the batch verb of a prior, beside the panel `pnl` of [`sample_buffer_panel`](@ref).

The call with no data of a refit prior, [`buffer_prior`](@ref), and the Choice Rule of an online step, [`pin_buffer_choice`](@ref), both read them, so the two fits read the same data. A buffer that records no Panel Field gives its masks as keywords, as a matrix call of the batch verb takes them. A buffer that records Panel Fields gives its masks inside `pnl`, so the keywords hold no mask. Both give the Exogenous Series and the implied volatilities that the buffer records.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: merge [`sample_buffer_kwargs`](@ref), [`exogenous_buffer_kwargs`](@ref) and [`implied_volatility_buffer_kwargs`](@ref) of `state`.
 2. `pnl` is an Asset Panel: merge [`exogenous_buffer_kwargs`](@ref) and [`implied_volatility_buffer_kwargs`](@ref) of `state`.

# Arguments

  - `state`: The buffer to read.
  - `pnl`: The panel of the rows of the buffer, which [`sample_buffer_panel`](@ref) gives, or `nothing`.

# Returns

  - `kwargs::NamedTuple`: The keywords of the fit.

# Related

  - [`sample_buffer_panel`](@ref)
  - [`buffer_prior`](@ref)
  - [`pin_buffer_choice`](@ref)
  - [`SampleBufferState`](@ref)
"""
function sample_buffer_fit_kwargs(state::SampleBufferState, ::Nothing)
    return (; sample_buffer_kwargs(state)..., exogenous_buffer_kwargs(state)...,
            implied_volatility_buffer_kwargs(state)...)
end
function sample_buffer_fit_kwargs(state::SampleBufferState, ::AssetPanel)
    return (; exogenous_buffer_kwargs(state)..., implied_volatility_buffer_kwargs(state)...)
end
"""
    buffer_asset_panel(state::SampleBufferState) -> Option{<:AssetPanel}

Rebuilds the Asset Panel of the observations that a sample buffer holds, from every part of a panel that the buffer records, or gives `nothing` when the buffer records no active mask.

A buffer that records Panel Fields gives the panel of [`sample_buffer_panel`](@ref). A buffer that records an active mask and no Panel Field gives a panel with no Panel Field. Its estimation mask is the estimation mask of the buffer. When the buffer records no estimation mask, it is a copy of the active mask.

The carry of [`EmpiricalPrior`](@ref) can record the active mask alone. Its call with no data refits a member on the rows of the buffer and this panel, as the batch fit refits the member on `rd.pnl`. [`returns_result`](@ref) rebuilds the panel of a `ReturnsResult` with it. The masks are copies of the valid region, because a later fold writes into the backing masks.

# Algorithm

 1. Return [`sample_buffer_panel`](@ref) of `state` when `state.P` is not `nothing`.
 2. Read the masks with [`sample_buffer_kwargs`](@ref), and return `nothing` when they hold no active mask.
 3. Copy the active mask, and the estimation mask or a second copy of the active mask, into two matrices.
 4. Make the `AssetPanel` of no Panel Field and the two masks.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `pnl::Option{<:AssetPanel}`: The panel of the rows of the buffer, or `nothing`.

# Related

  - [`sample_buffer_panel`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`SampleBufferState`](@ref)
  - [`returns_result`](@ref)
  - [`read_member`](@ref)
"""
function buffer_asset_panel(state::SampleBufferState)
    return buffer_asset_panel(state.P, state)
end
function buffer_asset_panel(P::AbstractVector{<:AbstractPanelField},
                            state::SampleBufferState)
    return sample_buffer_panel(P, state)
end
function buffer_asset_panel(::Nothing, state::SampleBufferState)
    msk = sample_buffer_kwargs(state)
    if !haskey(msk, :active_mask)
        return nothing
    end
    amsk = Matrix(msk.active_mask)
    emsk = haskey(msk, :estimation_mask) ? Matrix(msk.estimation_mask) : copy(amsk)
    return AssetPanel(; pf = AbstractPanelField[], amsk = amsk, emsk = emsk)
end
