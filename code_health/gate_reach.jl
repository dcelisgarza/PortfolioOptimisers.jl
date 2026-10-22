#!/usr/bin/env julia
#=
Name the gates that the `.jl` change of a branch under src/ or ext/ can reach.

    julia --startup-file=no code_health/gate_reach.jl <base> <to>
    julia --startup-file=no code_health/gate_reach.jl <base> WORKTREE

It compares each changed file at `<base>` and at `<to>`, or at the working tree with its untracked
files, and prints one `<class> <path>` line per file, then `reach <class>` for the whole change.
The class of a change is the highest of its files:

  - `prose`: the parsed code is the same once docstrings are removed, and no line inside a
    `jldoctest` fence changed. JET and the doctests cannot give another result.
  - `block`: the same code, and a `jldoctest` line changed. The scoped doctest run of the changed
    files runs every changed block.
  - `literal`: the same code once every string literal is blank. JET cannot give another result,
    and a doctest can print the changed string, in any file.
  - `code`: any other change, a new file, a deleted file, or a file that does not parse.

A key or a string under `src/01_Base/01_DocstringDictionaries/` feeds only docstrings, so a change
to one there is `prose`. `code_health/gate_stamp.sh` reads the last line: its push guard demands `jet` for
`code` alone, and `doctest` for `literal` and `code`. Sessions ran an eight-minute doctest run and
JET on prose-only pushes, which were four in five of the ticket commits of October 2026. The script
loads no package, and it runs in about a second.
=#
module GateReach

const CLASSES = ("prose", "block", "literal", "code")
const DICTIONARIES = "src/01_Base/01_DocstringDictionaries/"

git(root, args...) = readchomp(Cmd(`git $args`; dir = root))

function blob(root::AbstractString, rev::AbstractString, path::AbstractString)
    if rev == "WORKTREE"
        full = joinpath(root, path)
        return isfile(full) ? read(full, String) : nothing
    end
    exists = success(pipeline(Cmd(`git cat-file -e $rev:$path`; dir = root);
                              stderr = devnull))
    return exists ? read(Cmd(`git show $rev:$path`; dir = root), String) : nothing
end

"""
    changed_files(root, base, to) -> Vector{String}

The `.jl` files under src/ and ext/ that differ between `base` and `to`. For `WORKTREE` this is the
working tree against `base`, plus the untracked files.
"""
function changed_files(root::AbstractString, base::AbstractString, to::AbstractString)
    names = if to == "WORKTREE"
        vcat(split(git(root, "diff", "--name-only", "--no-renames", base, "--", "src",
                       "ext")),
             split(git(root, "ls-files", "--others", "--exclude-standard", "src", "ext")))
    else
        split(git(root, "diff", "--name-only", "--no-renames", base, to, "--", "src",
                  "ext"))
    end
    return sort!(unique!(filter(p -> endswith(p, ".jl"), String.(names))))
end

is_doc(x) = x == Symbol("@doc") || (x isa GlobalRef && x.name == Symbol("@doc"))
is_text(x) = x isa String || (x isa Expr && x.head === :string)

"""
    code_tree(x; blank = false)

`x` without docstrings and line nodes. A bare string in the body of a `struct` is a field
docstring. With `blank`, every string literal reads as the empty string, and a `unique_key_dict!`
call keeps no key and no value.
"""
code_tree(x; blank::Bool = false) = (blank && x isa String) ? "" : x
function code_tree(x::Expr; blank::Bool = false)
    if x.head === :macrocall && is_doc(x.args[1]) && length(x.args) >= 4
        return code_tree(x.args[end]; blank)
    end
    if blank && x.head === :call && x.args[1] === :unique_key_dict! && length(x.args) >= 3
        return Expr(:call, x.args[1:3]...)
    end
    fields = x.head === :struct
    args = Any[]
    for a in x.args
        if a isa LineNumberNode
            continue
        end
        if fields && a isa Expr && a.head === :block
            a = Expr(:block, filter(b -> !is_text(b), a.args)...)
        end
        push!(args, code_tree(a; blank))
    end
    return Expr(x.head, args...)
end

function doctest_lines(text::AbstractString)
    out = SubString{String}[]
    inside = false
    for line in split(text, '\n')
        s = lstrip(line)
        if startswith(s, "```jldoctest")
            inside = true
        elseif inside && startswith(s, "```")
            push!(out, line)
            inside = false
        end
        inside && push!(out, line)
    end
    return out
end

"""
    file_class(path, old, new) -> String

The class of the change of one file from the text `old` to the text `new`.
"""
function file_class(path::AbstractString, old, new)
    if (old === nothing || new === nothing)
        return "code"
    end
    a, b = try
        Meta.parseall(old), Meta.parseall(new)
    catch
        return "code"
    end
    if code_tree(a) != code_tree(b)
        if !(code_tree(a; blank = true) == code_tree(b; blank = true))
            return "code"
        end
        startswith(path, DICTIONARIES) || return "literal"
    end
    return doctest_lines(old) == doctest_lines(new) ? "prose" : "block"
end

function main(args = ARGS; root::AbstractString = pwd(), io::IO = stdout)
    if !(length(args) == 2)
        error("usage: gate_reach.jl <base> <to | WORKTREE>")
    end
    base, to = args
    top = 1
    for path in changed_files(root, base, to)
        class = file_class(path, blob(root, base, path), blob(root, to, path))
        println(io, class, " ", path)
        top = max(top, findfirst(==(class), CLASSES))
    end
    println(io, "reach ", CLASSES[top])
    return CLASSES[top]
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    GateReach.main()
end
