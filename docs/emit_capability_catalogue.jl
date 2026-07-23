# Serialises a capability catalogue back to `capability_catalogue.jl` source.
#
# The catalogue is hand-edited, so most of the time nothing here runs. It exists
# for the edits that are better made mechanically than by hand across ~500
# entries: dropping every label that merely restates its docstring, folding in a
# batch of newly-added estimators, or re-sorting a section. Load the catalogue,
# transform `CATALOGUE` as plain Julia data, and write it back:
#
#     include("docs/emit_capability_catalogue.jl")
#     entries = strip_redundant_labels(CATALOGUE)          # or any transform
#     write_catalogue("docs/capability_catalogue.jl", entries)
#
# The output is formatted to match what a person would write, so a mechanical
# rewrite produces a reviewable diff rather than a wall of reformatting.
#
# Round-tripping is checked by `test/test_26_docs.jl`: emitting the catalogue
# and re-parsing it must yield the same tree, which is what makes it safe to use
# this as an editing tool rather than a one-off migration script.

include(joinpath(@__DIR__, "capability_catalogue.jl"))

const _INDENT = 5      # continuation indent inside a node's child vector

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

One node as a Julia literal, indented to sit at column `indent`.
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

The full `const CATALOGUE = [...]` block.
"""
function emit_catalogue(entries::Vector = CATALOGUE)
    body = join((emit(node, 4) for node in entries), ",\n")
    return string("const CATALOGUE = [\n", body, "]\n")
end

"""
    write_catalogue(path, entries = CATALOGUE)

Rewrite `path`, replacing its `const CATALOGUE = [...]` block and leaving the
node-type definitions and header commentary above it untouched.
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

Rebuild the tree with `f` applied to every `Cap`, including the `Cap`s that head
a `Group`. Returning `nothing` from `f` drops that capability.
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
