# Writes a capability catalogue back out as the source of `capability_catalogue.jl`.
#
# A person edits the catalogue by hand, and neither the docs build nor a test runs this file. It
# is for an edit that a script makes more easily than a person over hundreds of entries, such as
# the removal of every label that repeats its docstring, or a batch of new estimators. Load the
# catalogue, change `CATALOGUE` as plain Julia data, and write it back:
#
#     include("docs/emit_capability_catalogue.jl")
#     entries = map_caps(f, CATALOGUE)    # `f` takes a `Cap`, returns a `Cap` or `nothing`
#     write_catalogue("docs/capability_catalogue.jl", entries)
#
# The layout of the output is not the layout of JuliaFormatter. Almost every line of the
# written file differs from the committed one. The commit hook formats the file again, and after
# that the diff of a scripted edit shows the edit alone.
#
# No test checks the round trip. Before you commit a scripted edit, include the written file in a
# fresh module, and compare its `CATALOGUE` node by node with the tree that you wrote.

include(joinpath(@__DIR__, "capability_catalogue.jl"))

const _INDENT = 5      # the indent of a child line, relative to its node

is_ident(s::AbstractString) = occursin(r"^[A-Za-z_][A-Za-z0-9_!]*$", s)

function jl_string(s::AbstractString)
    out = replace(s, '\\' => "\\\\", '"' => "\\\"", '$' => "\\\$")
    return string('"', replace(out, "\n" => "\\n"), '"')
end

jl_name(n::Symbol) = is_ident(string(n)) ? string(':', n) : jl_string(string(n))
jl_name(n::String) = is_ident(n) ? string(':', n) : jl_string(n)

function jl_cap(c::Cap)
    args = join(jl_name.(c.names), ", ")
    if !isnothing(c.label)
        args *= string(isempty(args) ? "" : "; ", "label = ", jl_string(c.label))
    end
    return string("Cap(", args, ")")
end

jl_head(h::Cap) = jl_cap(h)
jl_head(h::String) = jl_string(h)

"""
    emit(node, indent) -> String

The Julia literal of one node, indented to column `indent`.
"""
emit(n::Cap, indent::Int) = string(" "^indent, jl_cap(n))
emit(n::Prose, indent::Int) = string(" "^indent, "Prose(", jl_string(n.text), ")")
function emit(n::Note, indent::Int)
    if isempty(n.children)
        return string(" "^indent, "Note(", jl_string(n.text), ")")
    end
    return _with_children(string("Note(", jl_string(n.text)), n.children, indent)
end
function emit(n::Group, indent::Int)
    return _with_children(string("Group(", jl_head(n.head)), n.children, indent)
end
function emit(n::Section, indent::Int)
    return _with_children(string("Section(", jl_string(n.title)), n.children, indent)
end

function _with_children(head::String, children::Vector, indent::Int)
    pad = " "^indent
    if isempty(children)
        return string(pad, head, ", [])")
    end
    inner = indent + _INDENT
    body = join((emit(c, inner) for c in children), ",\n")
    return string(pad, head, ",\n", pad, "    [", lstrip(body), "])")
end

"""
    emit_catalogue(entries = CATALOGUE) -> String

The whole `const CATALOGUE = [...]` block as a string.
"""
function emit_catalogue(entries::Vector = CATALOGUE)
    body = join((emit(node, 4) for node in entries), ",\n")
    return string("const CATALOGUE = [\n", body, "]\n")
end

"""
    write_catalogue(path, entries = CATALOGUE)

Replace the `const CATALOGUE = [...]` block of `path` with `emit_catalogue(entries)`. Every
line above the block stays as it is.
"""
function write_catalogue(path::String, entries::Vector = CATALOGUE)
    src = read(path, String)
    marker = findfirst("const CATALOGUE = [", src)
    if isnothing(marker)
        error("emit_capability_catalogue: no `const CATALOGUE = [` block in $path.")
    end
    write(path, string(src[1:(first(marker) - 1)], emit_catalogue(entries)))
    return path
end

"""
    map_caps(f, entries)

Build the tree again with `f` applied to every `Cap`, and to the `Cap` at the head of each
`Group`. When `f` returns `nothing` for a `Cap` in a list, that `Cap` leaves the tree. When `f`
returns `nothing` for the head of a `Group`, the `Group` keeps its old head.
"""
map_caps(f, entries::Vector) = filter(!isnothing, map(n -> map_caps(f, n), entries))
map_caps(f, n::Cap) = f(n)
map_caps(f, n::Prose) = n
map_caps(f, n::Note) = Note(n.text, map_caps(f, n.children))
map_caps(f, n::Section) = Section(n.title, map_caps(f, n.children))
function map_caps(f, n::Group)
    head = n.head isa Cap ? f(n.head) : n.head
    return Group(isnothing(head) ? n.head : head, map_caps(f, n.children))
end
