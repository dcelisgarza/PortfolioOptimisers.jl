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
# A fragment is an ordinary page: a `@meta` block with its `Description`, one H1 with a label
# `(@id migration-next-<feature>)`, and the text. `test/test_64_docs_page_metadata_census.jl` reads
# it as a page of the site, and checks that the script folds it. The label stays the same in the
# migration guide, so a link to it holds.
#
# The counts of `code_health/prose_baseline.toml` move with the text. Run
# `julia --project=code_health code_health/prose.jl refresh` after this script.

const NEXT_RELEASE = joinpath(@__DIR__, "src", "next_release")
const MIGRATION = joinpath(@__DIR__, "src", "migration.md")

"""
    fold_fragment(text) -> String

The section of one fragment: the text with no `@meta` block, and with each heading two levels lower.
A `#` inside a fenced block is not a heading.
"""
function fold_fragment(text::AbstractString)
    out = String[]
    fence = ""
    meta = false
    for line in split(text, '\n')
        if meta
            meta = line != "```"
            continue
        end
        if isempty(fence)
            if startswith(line, "```@meta")
                meta = true
                continue
            end
            m = match(r"^\s*(```+|~~~+)", line)
            if m !== nothing
                fence = String(m.captures[1])
            elseif occursin(r"^#+ ", line)
                line = "##" * line
            end
        elseif strip(line) == fence
            fence = ""
        end
        push!(out, line)
    end
    return strip(join(out, '\n'))
end

"""
    fold(migration, fragments, from, to) -> String

The migration guide with the section `From v<from> to v<to>` of the fragments, in their order.
"""
function fold(migration::AbstractString, fragments, from::AbstractString,
              to::AbstractString)
    title = "## From v$from to v$to"
    if occursin(title * "\n", migration)
        error("`migration.md` has the section `$title`.")
    end
    parts = [strip(migration, ['\n']); title; fold_fragment.(fragments)]
    return join(parts, "\n\n") * "\n"
end

function fragment_paths(dir = NEXT_RELEASE)
    return isdir(dir) ? sort(filter(endswith(".md"), readdir(dir; join = true))) : String[]
end

function main(args)
    if length(args) != 2 || !all(a -> occursin(r"^\d+\.\d+\.\d+$", a), args)
        error("usage: fold_next_release.jl <from> <to>, two versions such as 0.34.5 0.35.0")
    end
    paths = fragment_paths()
    if isempty(paths)
        println("No fragment. `migration.md` does not change.")
        return nothing
    end
    text = fold(read(MIGRATION, String), [read(p, String) for p in paths], args...)
    write(MIGRATION, text)
    foreach(rm, paths)
    println("Folded ", length(paths), " fragment(s). Run `julia --project=code_health ",
            "code_health/prose.jl refresh`.")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
