"""
    panel_field_array(f::NumericPanelField) -> AbstractArray{<:Real}
    panel_field_array(f::CategoricalPanelField) -> AbstractArray{<:Integer}
    panel_field_array(f::TensorPanelField) -> AbstractArray{<:Real}

Return the array that holds the stored cells of one Panel Field.

A numeric and a tensor Panel Field store their values in `vals`, and a categorical Panel Field stores integer codes over its levels in `codes`. [`panel_field_values`](@ref) reads each kind through this function, so it applies one rule to all three.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `A::AbstractArray`: The stored array of the Panel Field, not a copy.

# Related

  - [`panel_field_values`](@ref)
  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
"""
function panel_field_array(f::NumericPanelField)
    return f.vals
end
function panel_field_array(f::CategoricalPanelField)
    return f.codes
end
function panel_field_array(f::TensorPanelField)
    return f.vals
end
"""
    panel_read_eltype(::Type{T}, v::Nothing) -> Type
    panel_read_eltype(::Type{T}, v::Integer) -> Type
    panel_read_eltype(::Type{T}, v::Real) -> Type

Return the element type that holds the stored type `T` and the policy value `v` of a read.

[`panel_field_values`](@ref) and [`feature_matrix`](@ref) derive the element type of their answer through this function. The type is derived from the two arguments, and never forced. A policy of `nothing` writes no cell, so it keeps `T`. An integer value keeps `T` too. The value converts to `T` only when it writes a cell, so `T` must hold it. A value of `-1` in an unsigned type raises an `InexactError` there. Any other value, such as `NaN` or `0.5`, needs a type that holds a fraction. It floats an integer `T` through [`float_if_integer`](@ref) and keeps every other `T`. A `Float32` field read with `NaN` stays `Float32`.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `T`: The element type of the stored array.
  - `v`: The policy value, or `nothing`.

# Returns

  - `Tr::Type`: The element type of the read.

# Related

  - [`panel_field_values`](@ref)
  - [`feature_matrix`](@ref)
  - [`float_if_integer`](@ref)
"""
function panel_read_eltype(::Type{T}, ::Union{Nothing, Integer}) where {T}
    return T
end
function panel_read_eltype(::Type{T}, ::Real) where {T}
    return float_if_integer(T)
end
"""
    panel_read_write!(V::AbstractArray, msk::Option{<:AbstractArray{Bool}}, v::Option{<:Real}) -> nothing

Write the policy value `v` into every cell of `V` where the mask `msk` is `false`, in place.

[`panel_field_values`](@ref) writes its two policies through this function, and [`feature_matrix`](@ref) writes its `placeholder`. `msk` covers the leading axes of `V`. A mask with the shape of `V` marks one entry for each cell. The observed mask of a Panel Field is one, and so is the data mask of one column of a Feature Matrix. The active mask of a time-varying panel is `observations × assets`, so on a tensor Panel Field one `false` entry covers every label of that observation and asset.

# Algorithm

 1. Return at once when `msk` or `v` is `nothing`. A mask of `nothing` marks no cell, and a policy of `nothing` keeps the stored value.
 2. Otherwise, for each `false` entry of `msk`, write `v` into the slab of `V` at that entry.

# Arguments

  - `V`: The values of the read. It is changed in place.
  - `msk`: The mask whose `false` entries mark the cells to write, or `nothing`.
  - `v`: The policy value, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`panel_field_values`](@ref)
  - [`feature_matrix`](@ref)
  - [`Option`](@ref)
"""
function panel_read_write!(::AbstractArray, ::Any, ::Nothing)::Nothing
    return nothing
end
function panel_read_write!(::AbstractArray, ::Nothing, ::Real)::Nothing
    return nothing
end
function panel_read_write!(V::AbstractArray, msk::AbstractArray{Bool}, v::Real)::Nothing
    tail = ntuple(Returns(Colon()), ndims(V) - ndims(msk))
    for k in CartesianIndices(msk)
        if !msk[k]
            view(V, Tuple(k)..., tail...) .= v
        end
    end
    return nothing
end
"""
    panel_field_values(pnl::AssetPanel, name::AbstractString;
                       inactive::Option{<:Real} = nothing,
                       unobserved::Option{<:Real} = NaN) -> AbstractArray
    panel_field_values(rd::ReturnsResult, name::AbstractString;
                       inactive::Option{<:Real} = nothing,
                       unobserved::Option{<:Real} = NaN) -> AbstractArray

Read the values of one Panel Field, with a policy for its inactive cells and a policy for its unobserved cells.

An [`AssetPanel`](@ref) keeps a finite value in every cell of a numeric Panel Field. The two universe masks say which cells are in the universe, and the observed mask of the field says which cells a fill policy wrote. A consumer reads the masks, so it never needs a marker in the values. A caller who wants the marker, for example to give the values to code that reads no mask, chooses it here at read time. Nothing is stored, and the panel does not change.

A policy is a value or `nothing`:

  - `inactive = NaN` writes `NaN` into every cell where the active mask is `false`. This is the usual convention that a cell outside the universe is missing.
  - `inactive = 0` writes a zero there.
  - `inactive = nothing` keeps the stored value.

`unobserved` takes the same forms for the cells whose observed mask is `false`, and its default `NaN` shows where the data held a blank. A cell that holds the placeholder of a blank that no fill reached is unobserved too. So `unobserved` writes it, and `unobserved = nothing` reads the stored placeholder back. A cell that is both inactive and unobserved takes the `inactive` policy when that policy is not `nothing`, because the asset is outside the universe whatever the data held. On a static panel no cell is inactive.

The same rule reads all three kinds of Panel Field. A [`TensorPanelField`](@ref) writes the policy into every label of an inactive observation and asset. A [`CategoricalPanelField`](@ref) reads its integer codes, so `inactive = CS_MISSING_GROUP` marks an inactive cell with the code of a missing group, and `inactive = NaN` gives floating-point codes with `NaN` there.

The element type of the answer comes from the stored type and the two policies, through [`panel_read_eltype`](@ref). An integer field read with a `NaN` policy is floated, and every other type is kept. An integer field read with `inactive = 0` and `unobserved = nothing` stays an integer field. The type depends on the policies, not on the cells that they write. So the default `unobserved = NaN` floats the codes of a categorical Panel Field that holds no blank.

# Mathematical definition

```math
\\begin{align}
V_{t,\\,i,\\,l} &= \\begin{cases}
\\alpha & a_{t,\\,i} = 0 \\\\
\\beta & m_{t,\\,i,\\,l} = 0 \\\\
x_{t,\\,i,\\,l} & \\text{otherwise}
\\end{cases}\\,, \\\\
& t = 1, \\ldots, T\\,, \\quad i = 1, \\ldots, N\\,, \\quad l = 1, \\ldots, L\\,.
\\end{align}
```

The first case that holds applies. With `inactive = nothing` the first case does not apply, and with `unobserved = nothing` the second case does not apply. A static panel has no observation index and no inactive cell, so ``t`` and the first case drop out. A numeric and a categorical Panel Field have no label axis, so ``l`` drops out. A Panel Field that cannot blank has ``m = 1`` in every cell.

Where:

  - ``V_{t,\\,i,\\,l}``: Entry of the answer at observation ``t``, asset ``i`` and label ``l``.
  - ``x_{t,\\,i,\\,l}``: Stored value of the Panel Field. On a categorical Panel Field it is the integer code of the level.
  - ``a_{t,\\,i}``: Active mask of the panel, ``1`` where asset ``i`` is in the universe at observation ``t``.
  - ``m_{t,\\,i,\\,l}``: Observed mask of the Panel Field, ``1`` where the raw input carried a value. It is ``0`` where a fill policy wrote the value, and where the cell holds a placeholder.
  - ``\\alpha``: The value of `inactive`.
  - ``\\beta``: The value of `unobserved`.
  - ``L``: Number of labels of a tensor Panel Field.
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

 1. Look the Panel Field up by name through [`panel_field`](@ref), and read its stored array through [`panel_field_array`](@ref).
 2. Copy the array into the element type that holds both policies.
 3. Write `unobserved` into every cell whose observed mask is `false`, through [`panel_read_write!`](@ref).
 4. Write `inactive` into every cell whose active mask is `false`. This step runs last, so it decides a cell that is both.
 5. For a [`ReturnsResult`](@ref), read the Asset Panel it holds in `rd.pnl` by steps 1 to 4.

# Arguments

  - `pnl`: The Asset Panel.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `name`: The Panel Field's name.
  - `inactive`: The value for a cell outside the universe, or `nothing` to keep the stored value.
  - `unobserved`: The value for a cell that a fill policy wrote or that holds a placeholder, or `nothing` to keep the stored value.

# Validation

  - The panel holds a Panel Field named `name`. Raises a `KeyError`.
  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - A policy value converts to the element type of the answer. An integer field read with `inactive = 0.5` is floated, and a `Rational` field read with `NaN` raises an `InexactError` at the first cell that the policy writes.
  - An integer policy value keeps an integer field when the other policy keeps it too, so the integer type must hold it. The default `unobserved = NaN` floats an integer field. A `UInt` field read with `inactive = -1` and `unobserved = nothing` raises an `InexactError` at the first cell that the policy writes. A float type converts every integer, and an integer past its range reads as an infinity. A `Float16` field read with `inactive = 100000` holds `Inf16` there.

# Returns

  - `V::AbstractArray`: A copy of the values, with the shape of the Panel Field: `assets` or `observations × assets`, with a trailing label axis on a tensor Panel Field.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"mcap\", vals = [1 2; 3 4; 5 6])];
                         amsk = [true false; true true; true true],
                         emsk = [true false; true true; true true]);

julia> panel_field_values(pnl, \"mcap\")
3×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0
 5.0  6.0

julia> panel_field_values(pnl, \"mcap\"; inactive = NaN)
3×2 Matrix{Float64}:
 1.0  NaN
 3.0    4.0
 5.0    6.0

julia> panel_field_values(pnl, \"mcap\"; inactive = 0, unobserved = nothing)
3×2 Matrix{Int64}:
 1  0
 3  4
 5  6
```

# Related

  - [`panel_field`](@ref)
  - [`AssetPanel`](@ref)
  - [`asset_panel`](@ref)
  - [`panel_dataframe`](@ref)
  - [`descriptor_field_values`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
  - [`Option`](@ref)
"""
function panel_field_values(pnl::AssetPanel, name::AbstractString;
                            inactive::Option{<:Real} = nothing,
                            unobserved::Option{<:Real} = NaN)
    f = panel_field(pnl, name)
    A = panel_field_array(f)
    T = eltype(A)
    V = Array{promote_type(panel_read_eltype(T, inactive),
                           panel_read_eltype(T, unobserved))}(A)
    panel_read_write!(V, f.omsk, unobserved)
    panel_read_write!(V, pnl.amsk, inactive)
    return V
end
function panel_field_values(rd::ReturnsResult, name::AbstractString;
                            inactive::Option{<:Real} = nothing,
                            unobserved::Option{<:Real} = NaN)
    @argcheck(!isnothing(rd.pnl),
              IsNothingError("a Panel Field is read off an Asset Panel, and rd.pnl is nothing. Build the ReturnsResult with the `pnl` that asset_panel returns."))
    return panel_field_values(rd.pnl, name; inactive = inactive, unobserved = unobserved)
end

export panel_field_values
