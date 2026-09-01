"""
    @define_pretty_show(T, flag::Bool = true)

Defines a `Base.show` method for `T` that prints the type name and one aligned line per field.

A field that is itself pretty-printable is rendered under its parent and indented, and an oversized one is collapsed to `Name ⋯`. The height at which a nested field collapses is the budget that [`compact_show_budget`](@ref) reads; see [`set_compact_show!`](@ref).

# Algorithm

The macro emits two definitions. Steps 3 to 9 are the body of the `Base.show` method it emits.

 1. When `flag` is `true`, define `has_pretty_show_method(::T)::Bool = true`. The return type is annotated, as it is on the four methods that [`has_pretty_show_method`](@ref) declares by hand.

 2. Define `Base.show(io::IO, obj::T)`.

 3. Read `fields`, the field names of `obj`. When `fields` is empty, print `T()` and return.

 4. When the `IO` context sets `:compact` or `:multiline`, print the type name alone and return.

 5. Print the wrapper name of the type, then compute `padding`, the length of the longest field name plus two.

 6. For each field in declaration order, read `val` with `getproperty`, so that a property a rule of [`@forward_properties`](@ref) swaps prints the swapped value.

 7. Choose the connector `sym1`, giving `┴` for the last printed line and `┼` otherwise.

 8. Print the field name, right-aligned to `padding`.

 9. Print `val` through the first branch that matches it, giving the rest of the line:

      + `nothing` prints as `nothing`.
      + A value that has a pretty-show method is rendered into a buffer, giving `alglines`. When the number of non-empty lines exceeds `compact_show_budget(io)`, print the wrapper name of the value followed by `⋯`. Otherwise print the first line beside the connector, and indent the rest under `│`.
      + A non-empty vector whose every element has a pretty-show method prints the summary from [`pretty_show_vector_summary`](@ref), then the lines from [`pretty_show_vector_body`](@ref), each indented under `│`.
      + A matrix prints its size and its type.
      + A vector of more than six entries, or a vector of arrays, prints its length and its type.
      + A `DataType` prints `DataType`, which is its type, then the wrapper name of the value, so a parametrised type reports the wrapper it instantiates and `Vector{Float64}` prints as `DataType: Array`.
      + Any other value prints its type and `repr(val)`.

# Arguments

  - `T`: The type for which to define the pretty-printing method.
  - `flag::Bool = true`: When `true`, the macro also defines `has_pretty_show_method(::T) = true`, which is how a parent finds that `T` renders through this method. Pass `false` for a type whose parent must print it by `repr` instead.

# Returns

  - Defines a `Base.show(io::IO, obj::T)` method for the given type.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`has_pretty_show_method`](@ref)
  - [`compact_show_budget`](@ref)
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
                fields = fieldnames(typeof(obj))
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
                for (i, field) in enumerate(fields)
                    val = getproperty(obj, field)
                    flag = has_pretty_show_method(val)
                    sym1 = ifelse(i == length(fields) &&
                                      (!flag || (flag && isempty(fieldnames(typeof(val))))),
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
                        ellines = [pretty_show_vector_element(v) for v in val]
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

Build the single-line summary for a vector field rendered by [`@define_pretty_show`](@ref).

Returns a string of the form `"N-element Vector{Name}"`. A vector is treated as homogeneous when every element shares the same wrapper-type name (so elements that differ only in type parameters are still homogeneous): a homogeneous vector uses that common wrapper name, otherwise the wrapper of the element type, falling back to the raw `eltype` for `Union`s.

# Algorithm

 1. Collect `names`, the wrapper-type name of every element. Two elements that differ only in their type parameters share one name.
 2. Read `et`, the element type of the vector.
 3. Take `tname` from the branch that `names` selects: the common name when every entry of `names` is equal, the string of `et` when `et` is a `Union`, and the wrapper name of `et` otherwise.
 4. Build the summary from the length of the vector and `tname`.

# Arguments

  - `val`: Non-empty vector whose elements all have a custom pretty-printing method.

# Returns

  - `summary::String`: Single-line `"N-element Vector{Name}"` summary.

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
    else
        (et isa Union ? string(et) : string(Base.typename(et).wrapper))
    end
    return "$(length(val))-element Vector{$(tname)}"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render a single vector element as a collapsed line for [`@define_pretty_show`](@ref).

Every element of a listed vector is shown as just its wrapper-type name. When the element is a struct with fields, a trailing `" ⋯"` marks it as a collapsed struct (consistent with how an over-budget struct field collapses to `Name ⋯`); fieldless elements are left bare.

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

Apply the shared collapse budget to the per-element lines of a vector rendered by [`@define_pretty_show`](@ref).

The budget comes from [`compact_show_budget`](@ref), so vector truncation honours the same `:limit` gate, global [`set_compact_show!`](@ref) setting, and per-call `:po_compact` override as struct collapsing. When the budget is `nothing` (disabled, unlimited output, or override-off) every line is returned. Otherwise, when the listing exceeds the budget it is split head-and-tail, mirroring how `Base` truncates long arrays, with a single `"⋮"` line marking the elision.

# Algorithm

 1. Read `budget` from [`compact_show_budget`](@ref), and `n`, the number of lines.
 2. Return `lines` unchanged when `budget` is `nothing`, or when `n` does not exceed it.
 3. Split the budget into `nhead`, its half rounded up, and `ntail`, the rest.
 4. Return the first `nhead` lines, a single `"⋮"` line, and the last `ntail` lines.

# Arguments

  - `io`: Output stream; drives the budget via [`compact_show_budget`](@ref).
  - `lines`: Per-element display strings from [`pretty_show_vector_element`](@ref).

# Returns

  - `body::Vector{String}`: Lines to print, possibly truncated with a `"⋮"` separator.

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
    return vcat(lines[1:nhead], "⋮", lines[(n - ntail + 1):n])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Default method indicating whether a type has a custom pretty-printing `show` method.

Overloading this method to return `true` indicates that type already has a custom pretty-printing method.

# Arguments

  - `::Any`: Any type.

# Returns

  - `flag::Bool`: `false` by default, indicating no custom pretty-printing method.

# Related

  - [`@define_pretty_show`](@ref)
"""
has_pretty_show_method(::Any)::Bool = false
has_pretty_show_method(::JuMP.Model)::Bool = true
has_pretty_show_method(::Clustering.Hclust)::Bool = true
has_pretty_show_method(::Clustering.KmeansResult)::Bool = true
@define_pretty_show(Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult})
