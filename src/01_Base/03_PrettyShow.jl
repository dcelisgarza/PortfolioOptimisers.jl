"""
    @define_pretty_show(T, flag::Bool = true)

Defines a `Base.show` method for `T` that prints the type name and one aligned line per field.

A field whose value also has a pretty `show` prints under its parent with an indent. A field of that kind that is too tall prints as `Name ⋯`. The budget that [`compact_show_budget`](@ref) reads sets the height at which a nested field collapses, and [`set_compact_show!`](@ref) sets the budget. The method prints the fields that [`pretty_show_fields`](@ref) resolves. The default of [`SHOW_NOTHING_FIELDS`](@ref) hides a field that holds `nothing`, and a type hides a field of its own choice through [`show_fields`](@ref).

# Algorithm

The macro emits two definitions. Steps 3 to 9 are the body of the `Base.show` method that it emits.

 1. When `flag` is `true`, define `has_pretty_show_method(::T)::Bool = true`. The method annotates its return type, as do the four methods that [`has_pretty_show_method`](@ref) declares by hand.

 2. Define `Base.show(io::IO, obj::T)`.

 3. Read `fields`, the field names that [`pretty_show_fields`](@ref) resolves for `obj`. When `fields` is empty, print `T()` and return. A type with no field reaches this step, and so does a type whose every field is hidden.

 4. When the `IO` context sets `:compact` or `:multiline`, print the type name alone and return.

 5. Print the wrapper name of the type, then compute `padding`, the length of the longest field name plus two.

 6. For each field in declaration order, read `val` with `getproperty`, so that a property a rule of [`@forward_properties`](@ref) swaps prints the swapped value.

 7. Choose the connector `sym1`. It is `┴` for the last printed line, and `┼` for each other line. The last printed line is the last field of `fields`, so a hidden field never gets the `┴`. A nested value whose own resolved field list is empty prints on one line, so it also gets `┴`.

 8. Print the field name, right-aligned to `padding`.

 9. Print the rest of the line from `val`, through the first branch that matches it.

      + `nothing` prints as `nothing`.
      + A value that has a pretty-show method renders into a buffer, and the lines of the buffer are `alglines`. When more than `compact_show_budget(io)` lines are not empty, print the wrapper name of the value and `⋯`. Otherwise print the first line after the connector, and indent the other lines under `│`.
      + A non-empty vector whose every element has a pretty-show method prints the summary from [`pretty_show_vector_summary`](@ref). The lines from [`pretty_show_vector_body`](@ref) follow, each with an indent under `│`.
      + A matrix prints its size and its type.
      + A vector of more than six entries, or a vector of arrays, prints its length and its type.
      + A `DataType` prints `DataType`, which is its type, and then the wrapper name of the value. A parametrised type thus shows the wrapper that it instantiates, and `Vector{Float64}` prints as `DataType: Array`.
      + Each other value prints its type and `repr(val)`.

# Arguments

  - `T`: The type for which to define the pretty-printing method.
  - `flag::Bool = true`: When `true`, the macro also defines `has_pretty_show_method(::T) = true`. A parent reads that method to find that `T` prints through this `show`. Pass `false` for a type that its parent must print with `repr`.

# Returns

  - Defines a `Base.show(io::IO, obj::T)` method for the given type.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`has_pretty_show_method`](@ref)
  - [`compact_show_budget`](@ref)
  - [`pretty_show_fields`](@ref)
  - [`show_fields`](@ref)
  - [`SHOW_NOTHING_FIELDS`](@ref)
  - [`pretty_show_vector_summary`](@ref)
  - [`pretty_show_vector_body`](@ref)
  - [`Base.show`](https://docs.julialang.org/en/v1/base/io/#Base.show)
"""
macro define_pretty_show(T, flag::Bool = true)
    esc(quote
            if $flag
                has_pretty_show_method(::$T)::Bool = true
            end
            function Base.show(io::IO, obj::$T)
                fields = pretty_show_fields(obj)
                tobj = typeof(obj)
                if isempty(fields)
                    return print(io, string(tobj, "()"), '\n')
                end
                if get(io, :compact, false) || get(io, :multiline, false)
                    return print(io, string(tobj), '\n')
                end
                name = Base.typename(tobj).wrapper
                print(io, name, '\n')
                padding = maximum(map(length, map(string, fields))) + 2
                ellines = String[]
                for (i, field) in enumerate(fields)
                    val = getproperty(obj, field)
                    flag = has_pretty_show_method(val)
                    sym1 = ifelse(i == length(fields) &&
                                      (!flag || (flag && isempty(pretty_show_fields(val)))),
                                  '┴', '┼')
                    print(io, lpad(string(field), padding), " ")
                    if isnothing(val)
                        print(io, "$(sym1) nothing", '\n')
                    elseif flag
                        ioalg = IOContext(IOBuffer(), :limit => get(io, :limit, false),
                                          :displaysize => displaysize(io))
                        pc = get(io, :po_compact, :__unset__)
                        if pc !== :__unset__
                            ioalg = IOContext(ioalg, :po_compact => pc)
                        end
                        show(ioalg, val)
                        algstr = String(take!(ioalg.io))
                        alglines = split(algstr, '\n')
                        budget = compact_show_budget(io)
                        if !isnothing(budget) &&
                           count(l -> !(isempty(l) || l == "\n"), alglines) > budget
                            conn = ifelse(i == length(fields), '┴', '┼')
                            print(io, "$(conn) ", Base.typename(typeof(val)).wrapper, " ⋯",
                                  '\n')
                        else
                            print(io, "$(sym1) ", alglines[1], '\n')
                            for l in alglines[2:end]
                                if isempty(l) || l == '\n'
                                    continue
                                end
                                sym2 = '│'
                                print(io, lpad("$sym2 ", padding + 3), l, '\n')
                            end
                        end
                    elseif isa(val, AbstractVector) &&
                           !isempty(val) &&
                           all(has_pretty_show_method, val)
                        print(io, "┼ ", pretty_show_vector_summary(val), '\n')
                        map!(pretty_show_vector_element, resize!(ellines, length(val)),
                             val)
                        for l in pretty_show_vector_body(io, ellines)
                            print(io, lpad("│ ", padding + 3), l, '\n')
                        end
                    elseif isa(val, AbstractMatrix)
                        print(io, "$(sym1) $(size(val,1))×$(size(val,2)) $(typeof(val))",
                              '\n')
                    elseif isa(val, AbstractVector) && length(val) > 6 ||
                           isa(val, AbstractVector{<:AbstractArray})
                        print(io, "$(sym1) $(length(val))-element $(typeof(val))", '\n')
                    elseif isa(val, DataType)
                        tval = typeof(val)
                        valstr = Base.typename(val).wrapper
                        print(io, "$(sym1) $(tval): ", valstr, '\n')
                    else
                        print(io, "$(sym1) $(typeof(val)): ", repr(val), '\n')
                    end
                end
                return nothing
            end
        end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the one-line summary of a vector field that [`@define_pretty_show`](@ref) prints.

The summary has the form `"N-element Vector{Name}"`. When all elements have the same wrapper-type name, `Name` is that name. Two elements that differ only in their type parameters have the same wrapper-type name. In other cases, `Name` is the wrapper name of the element type, or the element type itself when it is a `Union`.

# Algorithm

 1. Collect `names`, the wrapper-type name of every element.
 2. Read `et`, the element type of the vector.
 3. Choose `tname`. It is the common name when every entry of `names` is equal, the string of `et` when `et` is a `Union`, and the wrapper name of `et` in the other cases.
 4. Build the summary from the length of the vector and `tname`.

# Arguments

  - `val`: A non-empty vector whose elements all have a custom pretty-printing method.

# Returns

  - `summary::String`: The one-line `"N-element Vector{Name}"` summary.

# Related

  - [`@define_pretty_show`](@ref)
  - [`pretty_show_vector_element`](@ref)
  - [`pretty_show_vector_body`](@ref)
"""
function pretty_show_vector_summary(val::AbstractVector)
    names = [string(Base.typename(typeof(v)).wrapper) for v in val]
    et = eltype(val)
    tname = if allequal(names)
        first(names)
    elseif et isa Union
        string(et)
    else
        string(Base.typename(et).wrapper)
    end
    return "$(length(val))-element Vector{$(tname)}"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render one vector element as a collapsed line for [`@define_pretty_show`](@ref).

Each element of a listed vector prints as its wrapper-type name only. When the element is a struct with fields, a `" ⋯"` after the name marks the collapsed struct. A struct field over the budget collapses to `Name ⋯` in the same way. An element with no field prints its name alone.

# Algorithm

 1. Take `s`, the wrapper-type name of `v`.
 2. Return `s` unchanged when `v` has no field, and otherwise return `s` followed by `" ⋯"`.

# Arguments

  - `v`: The vector element to render.

# Returns

  - `line::String`: The collapsed one-line rendering of `v`.

# Related

  - [`@define_pretty_show`](@ref)
  - [`pretty_show_vector_summary`](@ref)
  - [`pretty_show_vector_body`](@ref)
"""
function pretty_show_vector_element(@nospecialize(v))
    s = string(Base.typename(typeof(v)).wrapper)
    return isempty(fieldnames(typeof(v))) ? s : s * " ⋯"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply the collapse budget to the element lines of a vector that [`@define_pretty_show`](@ref) prints.

The budget comes from [`compact_show_budget`](@ref). A vector thus obeys the same `:limit` key, global [`set_compact_show!`](@ref) setting and per-call `:po_compact` override as a struct field. The budget is `nothing` when the collapse is off, when the output is not limited, or when the override turns it off. The function then returns every line. When the list has more lines than the budget, the function keeps the first lines and the last lines, as `Base` does for a long array. One `"⋮"` line marks the lines that it removes.

# Algorithm

 1. Read `budget` from [`compact_show_budget`](@ref), and `n`, the number of lines.
 2. Return `lines` unchanged when `budget` is `nothing`, or when `n` does not exceed it.
 3. Split the budget into `nhead`, its half rounded up, and `ntail`, the rest.
 4. Return the first `nhead` lines, one `"⋮"` line, and the last `ntail` lines.

# Arguments

  - `io`: The output stream. [`compact_show_budget`](@ref) reads the budget from it.
  - `lines`: The display string of each element, from [`pretty_show_vector_element`](@ref).

# Returns

  - `body::Vector{String}`: The lines to print. When the function removes lines, a `"⋮"` line stands in their place.

# Related

  - [`@define_pretty_show`](@ref)
  - [`compact_show_budget`](@ref)
  - [`pretty_show_vector_element`](@ref)
"""
function pretty_show_vector_body(io::IO, lines::AbstractVector{<:AbstractString})
    budget = compact_show_budget(io)
    n = length(lines)
    if isnothing(budget) || n <= budget
        return lines
    end
    nhead = cld(budget, 2)
    ntail = budget - nhead
    return vcat(view(lines, 1:nhead), "⋮", view(lines, (n - ntail + 1):n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Name the fields of `obj` that [`@define_pretty_show`](@ref) can print, in declaration order.

The default method returns every declared field. A type that hides one of its fields overloads this method for itself and returns the other fields. The overload gets the instance, so it can hide a field always or only for some values. A partial-fit `cache` holds the state of an incremental fit and not configuration, so its type always hides it. The moment estimators hide an optional `cvg` only when it is `nothing`. The caller did not select that option, so it is not part of the configuration. A field added in this way changes no rendered docstring.

This method selects fields by type, and [`SHOW_NOTHING_FIELDS`](@ref) selects them by configuration. [`pretty_show_fields`](@ref) combines the two, and a per-name entry of the configuration overrides this method in both directions.

# Arguments

  - `obj`: The value to print.

# Returns

  - `fields`: The field names that the method can print, as a tuple or a vector of `Symbol`.

# Related

  - [`@define_pretty_show`](@ref)
  - [`pretty_show_fields`](@ref)
  - [`SHOW_NOTHING_FIELDS`](@ref)
"""
show_fields(@nospecialize(obj)) = fieldnames(typeof(obj))
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the fields that [`@define_pretty_show`](@ref) renders for `obj`.

Three sources decide, in a fixed order of precedence. A per-name entry of [`SHOW_NOTHING_FIELDS`](@ref) overrides the [`show_fields`](@ref) method of the type, and that method overrides the global default of [`SHOW_NOTHING_FIELDS`](@ref). The name is the name of the type without its module, so two types with one name in two modules share one entry.

# Algorithm

 1. Read `cfg`, the active value of [`SHOW_NOTHING_FIELDS`](@ref). Read `entry`, its per-name entry for the name of the type of `obj`, or `nothing` when the type has no entry.
 2. Choose `fields`. It is every declared field when `entry` is `true`, so that a per-name request to show all fields overrides a [`show_fields`](@ref) overload. In the other cases it is the result of [`show_fields`](@ref).
 3. Choose `show`. It is `entry` when the type has an entry, and the global default of `cfg` in the other case.
 4. Return `fields` when `show` is `true`. Otherwise return the members of `fields` whose value, read with `getproperty`, is not `nothing`.

# Arguments

  - `obj`: The value to print.

# Returns

  - `fields`: The field names to render, in declaration order. Every declared field, the result of [`show_fields`](@ref), or the members of that result whose value is not `nothing`.

# Related

  - [`@define_pretty_show`](@ref)
  - [`show_fields`](@ref)
  - [`SHOW_NOTHING_FIELDS`](@ref)
  - [`set_show_nothing_fields!`](@ref)
  - [`with_show_nothing_fields`](@ref)
"""
function pretty_show_fields(obj)
    cfg = SHOW_NOTHING_FIELDS[]
    entry = get(cfg.by_type, nameof(typeof(obj)), nothing)
    fields = entry === true ? fieldnames(typeof(obj)) : show_fields(obj)
    if something(entry, cfg.default)
        return fields
    end
    return [f for f in fields if !isnothing(getproperty(obj, f))]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the type of a value has a custom pretty-printing `show` method. The default method returns `false`.

A method that returns `true` for a type tells [`@define_pretty_show`](@ref) that the type has a custom pretty-printing method.

# Arguments

  - `::Any`: A value of any type.

# Returns

  - `flag::Bool`: `false` for the default method, which means that the type has no custom pretty-printing method.

# Related

  - [`@define_pretty_show`](@ref)
"""
has_pretty_show_method(::Any)::Bool = false
has_pretty_show_method(::JuMP.Model)::Bool = true
has_pretty_show_method(::Clustering.Hclust)::Bool = true
has_pretty_show_method(::Clustering.KmeansResult)::Bool = true
@define_pretty_show(Union{AbstractEstimator, AbstractAlgorithm, AbstractResult})
