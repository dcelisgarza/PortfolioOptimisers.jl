#!/usr/bin/env julia
#
# Fold the fragments of `docs/src/next_release/` into `docs/src/migration.md`, for a release.
#
#     julia docs/fold_next_release.jl <from> <to>      for example 0.34.5 0.35.0
#
# A change that a user must know about states it in a page of its own,
# `docs/src/next_release/<feature>.md`. Every such change once added its section at the end of
# `migration.md`, so two open pull requests always conflicted there, and each text named a version
# that another pull request took first. A fragment names no version. The docs build shows the
# fragments under the migration guide, and the release pull request runs this script, which appends
# the section `From v<from> to v<to>` and deletes the fragments.
#
# A fragment is an ordinary page: a `@meta` block with its `Description` alone, one H1 with the
# label `(@id migration-next-<feature>)`, and the text. `fragment_defects` holds the form. The
# census `test/test_64_docs_page_metadata_census.jl` runs it on each fragment of the tree, and so
# does this script. The label stays the same in the migration guide, so a link to it holds.
#
# The counts of `code_health/prose_baseline.toml` move with the text. Run
# `julia --project=code_health code_health/prose.jl refresh` after this script.

const NEXT_RELEASE = joinpath(@__DIR__, "src", "next_release")
const MIGRATION = joinpath(@__DIR__, "src", "migration.md")

"""
    fragment_lines(text) -> Vector{Pair{Symbol, String}}

Each line of a fragment with its kind: `:meta` in a `@meta` block, `:code` in any other fenced
block, and `:text` outside. The two lines of the fence take the kind of their block.
"""
function fragment_lines(text::AbstractString)
    out = Pair{Symbol, String}[]
    fence = ""
    kind = :text
    for line in String.(split(text, '\n'))
        if isempty(fence)
            # The info string of a backtick fence holds no backtick. A line that does is inline code.
            m = match(r"^\s*(?:(`{3,})[^`]*|(~{3,}).*)$", line)
            if !isnothing(m)
                fence = String(something(m.captures...))
                kind = startswith(line, "```@meta") ? :meta : :code
            end
            push!(out, kind => line)
        else
            push!(out, kind => line)
            mark = strip(line)
            if startswith(mark, fence) && all(==(first(fence)), mark)
                fence = ""
                kind = :text
            end
        end
    end
    return out
end

"""
    fragment_defects(feature, text, migration) -> Vector{String}

What keeps the fragment `<feature>.md` from a fold into the guide `migration`, one line each.
"""
function fragment_defects(feature::AbstractString, text::AbstractString,
                          migration::AbstractString)
    out = String[]
    lines = fragment_lines(text)
    label = "migration-next-$feature"
    if !occursin(r"^[a-z0-9]+(-[a-z0-9]+)*$", feature)
        push!(out,
              "the name of the file is not lower-case words with a hyphen between them")
    end
    # The guide has no `CurrentModule`, so a link that needs one stops the docs build at the fold.
    if !all(occursin(r"^(```.*|Description = .*)$", l) for (k, l) in lines if k === :meta)
        push!(out, "the `@meta` block holds a line that is not its `Description`")
    end
    if fragment_lines(text * "\n")[end].first !== :text
        push!(out, "a fenced block has no end")
    end
    h1 = [l for (k, l) in lines if k === :text && startswith(l, "# ")]
    if length(h1) != 1 || !startswith(h1[1], "# [") || !endswith(h1[1], "](@id $label)")
        push!(out, "the page does not hold one H1, `# [Title](@id $label)`")
    end
    if any(k === :text && occursin(r"^#{5,} ", l) for (k, l) in lines)
        push!(out, "a heading is below level four, and the fold lowers it by two")
    end
    # The name of the file makes a label unique in the directory, and the guide holds the old ones.
    for m in eachmatch(r"\(@id ([^)]+)\)", text)
        id = m.captures[1]
        if id != label && !startswith(id, label * "-")
            push!(out, "the label `$id` does not start with `$label-`")
        elseif occursin("(@id $id)", migration)
            push!(out, "the migration guide holds the label `$id`")
        end
    end
    # Another pull request can take the version that a text names.
    for m in eachmatch(r"\bv\d+\.\d+(?:\.\d+)?", text)
        push!(out, "the text names the version `$(m.match)`")
    end
    return out
end

"""
    fold_fragment(text) -> String

The section of one fragment: the text with no `@meta` block, and with each heading two levels lower.
A `#` inside a fenced block is not a heading.
"""
function fold_fragment(text::AbstractString)
    out = [k === :text && occursin(r"^#+ ", l) ? "##" * l : l
           for (k, l) in fragment_lines(text) if k !== :meta]
    return strip(join(out, '\n'))
end

"""
    fold(migration, fragments, from, to) -> String

The migration guide with the section `From v<from> to v<to>` of the fragments, in their order. The
section comes after the last one of the guide, so `from` is not below the version it ends at.
"""
function fold(migration::AbstractString, fragments, from::AbstractString,
              to::AbstractString)
    last_to = maximum((VersionNumber(m.captures[1])
                       for m in eachmatch(r"^## From v\S+ to v(\S+)$"m, migration));
                      init = v"0")
    if !(last_to <= VersionNumber(from) < VersionNumber(to))
        error("a section from v$from to v$to does not follow v$last_to, where the guide ends.")
    end
    parts = [strip(migration, ['\n']); "## From v$from to v$to"; fold_fragment.(fragments)]
    return join(parts, "\n\n") * "\n"
end

function fragment_paths(dir = NEXT_RELEASE)
    return isdir(dir) ? sort(filter(endswith(".md"), readdir(dir; join = true))) : String[]
end

"""
    tree_defects(paths, migration) -> Vector{String}

Every defect of the fragments `paths`, each with the name of its file.
"""
function tree_defects(paths, migration::AbstractString)
    return ["$(basename(p)): $d" for p in paths
            for d in
                fragment_defects(first(splitext(basename(p))), read(p, String), migration)]
end

function main(args; migration_path = MIGRATION, dir = NEXT_RELEASE)
    if length(args) != 2 || !all(a -> occursin(r"^\d+\.\d+\.\d+$", a), args)
        error("usage: fold_next_release.jl <from> <to>, two versions such as 0.34.5 0.35.0")
    end
    paths = fragment_paths(dir)
    if isempty(paths)
        println("No fragment. `migration.md` does not change.")
        return nothing
    end
    migration = read(migration_path, String)
    defects = tree_defects(paths, migration)
    if !isempty(defects)
        error("no fold, because a fragment has a wrong form:\n  " * join(defects, "\n  "))
    end
    write(migration_path, fold(migration, [read(p, String) for p in paths], args...))
    foreach(rm, paths)
    println("Folded ", length(paths), " fragment(s). Run `julia --project=code_health ",
            "code_health/prose.jl refresh`.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
