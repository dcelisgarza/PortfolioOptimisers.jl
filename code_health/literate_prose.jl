#!/usr/bin/env julia
#
# The Literate-prose half of the documentation gates.
#
#     julia --project=code_health code_health/literate_prose.jl scan examples/00_Examples.jl
#     julia --project=code_health code_health/literate_prose.jl scan
#     julia --project=code_health code_health/literate_prose.jl refresh
#
# It reads the prose of every file under `examples/**/*.jl` and `user_guide/*.jl` and counts the
# rules of `.github/instructions/julia-prose.instructions.md` that a parser can read. The
# counts of one page are its row in `code_health/literate_prose_baseline.toml`. A count may fall
# and may not rise, a page with a count above zero needs a row, and a page whose every counted rule
# is at zero carries no row at all, so the baseline empties as the rewrites of map #1218 land.
#
# `test/test_72_literate_prose_census.jl` is the gate. It includes this file, so the suite and a
# rewrite session read the prose with one reader and cannot drift. `scan` is what a rewrite session
# runs on its own pages before and after the rewrite. `refresh` writes the whole baseline, which
# generates it once. A rewrite session does NOT refresh: fourteen rewrite tickets run in parallel,
# and two sessions that each write the whole file lose one of the two writes. It pastes the row the
# census prints for its own pages instead.
#
# This script reads TEXT and loads only `TOML`, as every census in this repository does. It parses
# no Julia and loads no package under measurement, so it costs well under a second over the corpus.

using TOML

const NAME = "literate_prose_baseline.toml"
const DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(DIR, ".."))

# --- what a row counts -----------------------------------------------------
#
# The order here is the order a row prints. Every name below binds: it may fall and may not rise.
# `words` is the one column that does not bind, and it prints last.

"""
The counted rules, in the order a row prints them. The key is the column name, and the value is the
rule of the `unslop` skill, or the section of the instruction file, that the column reads.
"""
const COUNTED = ("emdash" => "rule 13", "endash" => "rule 13", "curly" => "rule 19",
                 "notjust" => "rule 9", "aivocab" => "rule 7", "fancy_is" => "rule 8",
                 "filler" => "rule 23", "metaphor" => "rule 26", "plainword" => "rule 31",
                 "bold_label" => "rule 16", "title_case" => "rule 17", "emoji" => "rule 18",
                 "glossary" => "a concept in plain words",
                 "mechanism" => "a concept in plain words",
                 "verdict" => "a check is a number")

"""
The column that records the prose word count. It carries no limit: a page is long because it covers
more ground, and the paragraph rule is what holds its length down.
"""
const WORDS = "words"

const COLUMNS = (first.(COUNTED)..., WORDS)

# The five words of rules 8, 26 and 31 this library owns are absent from the patterns below, and
# `.github/instructions/julia-prose.instructions.md` § *The Gate* says why: "the expected
# returns vector", "a Pareto surface", "a feature matrix" and "a leveraged portfolio" write the
# concrete word the rule asks for, so a census over the lists as written could never reach zero.
const PATTERNS = Dict("emdash" => r"—", "endash" => r"–", "curly" => r"[“”‘’]",
                      "notjust" =>
                          r"\b(?:not|isn't|is not|aren't|are not|wasn't|weren't)\s+(?:just|only|merely)\b"i,
                      "aivocab" =>
                          r"\b(?:additionally|crucial(?:ly)?|delv(?:e|es|ed|ing)|enduring|enhanc(?:e|es|ed|ing|ement|ements)|foster(?:s|ed|ing)?|garner(?:s|ed|ing)?|interplay|intricate(?:ly)?|landscapes?|pivotal|showcas(?:e|es|ed|ing)|tapestry|testament|underscor(?:e|es|ed|ing)|vibrant)\b"i,
                      "fancy_is" => r"\b(?:serves? as|stands? as|boasts?)\b"i,
                      "filler" =>
                          r"\b(?:in order to|due to the fact that|it is important to note that)\b"i,
                      "metaphor" =>
                          r"\b(?:substrates?|wedges?|locus|vantage|nexus|bedrock|scaffolding|modality|paradigms?|gold-plating|evacuat(?:e|es|ed|ing)|endgame|north star|flywheel)\b"i,
                      "plainword" =>
                          r"\b(?:utilis(?:e|es|ed|ing)|utiliz(?:e|es|ed|ing)|facilitat(?:e|es|ed|ing)|numerous|in the event that)\b"i,
                      "bold_label" => r"\*\*[^*\n]{1,80}:\*\*|\*\*[^*\n]{1,80}\*\*\s*:",
                      "mechanism" =>
                          r"\bseams?\b|\bcarriers?\b|\bread-?outs?\b|\bhosts?\b|\brefused? by name\b|§"i,
                      # "the identity matrix" is the matrix, not the verdict, and a page that
                      # shrinks a covariance towards it says so in those words.
                      "verdict" =>
                          r"\bto the bit\b|\bby construction\b|\bhonest\w*\b|\b(?:the|each) identity\b(?! matrix)|\bagrees?\b"i)

# --- reading the prose of a page -------------------------------------------
#
# Literate renders two things as markdown: a `#= ... =#` block, and a line whose first two
# characters are `# `. A `##` comment at column zero renders as a `#` INSIDE a code cell, so it is
# code and not prose. A line that carries the `#src` trailer never reaches an output at all, and
# `test/test_71_process_citation_census.jl` skips it for the same reason.

"""
    markdown_lines(text) -> Vector{String}

The lines of `text` that Literate renders as markdown, in order, with the comment markers removed.
A `#src` line is dropped.
"""
function markdown_lines(text::AbstractString)
    acc = String[]
    inblock = false
    for ln in split(text, '\n')
        if endswith(rstrip(ln), "#src")
            continue
        end
        if inblock
            s = ln
            if occursin("=#", s)
                inblock = false
                s = s[1:(first(findfirst("=#", s)) - 1)]
            end
            push!(acc, String(s))
        elseif startswith(ln, "#=")
            inblock = true
            s = ln[3:end]
            if occursin("=#", s)
                inblock = false
                s = s[1:(first(findfirst("=#", s)) - 1)]
            end
            push!(acc, String(s))
        elseif startswith(ln, "# ")
            push!(acc, String(ln[3:end]))
        elseif ln == "#"
            push!(acc, "")
        end
    end
    return acc
end

"""
    unfenced(lines) -> Vector{String}

`lines` with every fenced block removed. The `Description` line of a ````` ```@meta ````` block survives as
its text alone: the docs build renders it into the page's metadata, so a reader reads it, and
`test/test_64_docs_page_metadata_census.jl` already holds it to a shape.
"""
function unfenced(lines)
    acc = String[]
    fence, meta = false, false
    for ln in lines
        s = lstrip(ln)
        if startswith(s, "```")
            if fence
                fence, meta = false, false
            else
                fence = true
                meta = startswith(s, "```@meta")
            end
            continue
        end
        if fence
            if meta
                m = match(r"^\s*Description\s*=\s*\"(.*)\"\s*$", ln)
                m === nothing || push!(acc, String(m.captures[1]))
            end
            continue
        end
        push!(acc, ln)
    end
    return acc
end

"""
    readable(line) -> String

`line` with everything the rule does not govern removed: an inline code span, an inline LaTeX
expression, the target of a markdown link, and the `!!!` marker of an admonition. The heading
markers stay, because rule 17 reads them.
"""
function readable(line::AbstractString)
    s = replace(line, r"``[^`]*``" => " ")
    s = replace(s, r"`[^`]*`" => " ")
    s = replace(s, r"\$[^\$\n]*\$" => " ")
    s = replace(s, r"\]\([^)\n]*\)" => "]")
    s = replace(s, r"^\s*!!!\s+\w+\s*" => "")
    return s
end

"""
    prose(path) -> Vector{String}

The prose of one Literate source: every line a reader reads, cleaned of what the rule does not
govern. This is the one reader. The census and a rewrite session's scan both call it.
"""
prose(path::AbstractString) = readable.(unfenced(markdown_lines(read(path, String))))

# --- the counts a line carries ---------------------------------------------

matches(re::Regex, s::AbstractString) = count(_ -> true, eachmatch(re, s))

"""
    word_count(line) -> Int

The words of one prose line. A token counts when it carries a letter or a digit, so a table rule, a
bullet marker and a heading's `#` marks count for nothing.
"""
function word_count(line::AbstractString)
    n = 0
    for tok in split(line)
        any(c -> isletter(c) || isdigit(c), tok) && (n += 1)
    end
    return n
end

"""
    title_case(line) -> Bool

Whether `line` is a heading in title case, which is rule 17. The reading is deliberately narrow: it
asks that every long word after the first start with a capital, and that there be two such words.
"The Black-Litterman prior" therefore passes, because `prior` is lower case, and "Hierarchical Risk
Parity" does not.
"""
function title_case(line::AbstractString)
    m = match(r"^\s*(#{1,6})\s+(.*\S)\s*$", line)
    if m === nothing
        return false
    end
    toks = [t for t in split(m.captures[2]) if all(c -> isletter(c) || c == '-', t)]
    if !(length(toks) >= 3)
        return false
    end
    long = [t for t in toks[2:end] if length(t) >= 4]
    if !(length(long) >= 2)
        return false
    end
    return all(t -> isuppercase(first(t)), long)
end

"""
    emoji_count(line) -> Int

The decorative emoji of one line, which is rule 18. A mathematical symbol such as `≈` sits outside
every range below, so it counts for nothing.
"""
function emoji_count(line::AbstractString)
    n = 0
    for c in line
        u = UInt32(c)
        if (0x1F300 <= u <= 0x1FAFF) ||
           (0x2600 <= u <= 0x27BF) ||
           (0x2B00 <= u <= 0x2BFF) ||
           u == 0xFE0F ||
           u == 0x2049 ||
           u == 0x203C
            n += 1
        end
    end
    return n
end

# --- the glossary of CONTEXT.md --------------------------------------------

"""
    glossary_terms(; root = REPO_ROOT) -> Vector{String}

The multi-word bold terms of `CONTEXT.md` in their capitalised form, longest first. The census
reads them off that file rather than holding a copy, so the list never goes stale. A term counts
only when every word of it starts with a capital: `Coverage Universe` is the glossary's name for
the concept, and `Black-Litterman family` is a proper noun with an ordinary word after it.
"""
function glossary_terms(; root::AbstractString = REPO_ROOT)
    path = joinpath(root, "CONTEXT.md")
    if !(isfile(path))
        return String[]
    end
    acc = String[]
    for m in eachmatch(r"\*\*([^*\n]+)\*\*", read(path, String))
        term = strip(m.captures[1])
        toks = split(term)
        if !(length(toks) >= 2)
            continue
        end
        if !(all(t -> isuppercase(first(t)) && all(c -> isletter(c) || c == '-', t), toks))
            continue
        end
        push!(acc, String(term))
    end
    return sort!(unique!(acc); by = length, rev = true)
end

"""
    glossary_pattern(terms) -> Regex

One alternation over `terms`, longest first, read at a word boundary and in the capitalised form.
"the coverage universe" in lower case, which the rule asks a page to write, matches nothing. The
pattern is built once per run: a `Regex` per term per line compiles three hundred patterns for
every line of the corpus, and that reading costs minutes instead of a second.
"""
function glossary_pattern(terms)
    if isempty(terms)
        return r"(?!)"
    end
    body = join((replace(t, "-" => "\\-") for t in terms), "|")
    return Regex("\\b(?:" * body * ")\\b")
end

# --- measuring one page and the corpus -------------------------------------

"""
    counts(path; glossary = glossary_pattern(glossary_terms())) -> Dict{String, Int}

Every column of one page's row. A column that is absent from a row is zero, so a row shrinks as a
rewrite lands.
"""
function counts(path::AbstractString; glossary = glossary_pattern(glossary_terms()))
    row = Dict{String, Int}(c => 0 for c in COLUMNS)
    for line in prose(path)
        for (name, re) in PATTERNS
            row[name] += matches(re, line)
        end
        row["title_case"] += title_case(line) ? 1 : 0
        row["emoji"] += emoji_count(line)
        row["glossary"] += matches(glossary, line)
        row[WORDS] += word_count(line)
    end
    return row
end

"""
    pages(; root = REPO_ROOT) -> Vector{String}

Every file the rule governs, as a path relative to `root`: `examples/**/*.jl` and
`user_guide/*.jl`. The glob is the `applyTo` line of
`.github/instructions/julia-prose.instructions.md`.
"""
function pages(; root::AbstractString = REPO_ROOT)
    acc = String[]
    examples = joinpath(root, "examples")
    if isdir(examples)
        for (dir, _, files) in walkdir(examples), f in files
            endswith(f, ".jl") && push!(acc, relpath(joinpath(dir, f), root))
        end
    end
    guide = joinpath(root, "user_guide")
    if isdir(guide)
        for f in readdir(guide)
            endswith(f, ".jl") && push!(acc, relpath(joinpath(guide, f), root))
        end
    end
    return sort!(acc)
end

"""
    measure(; root = REPO_ROOT) -> Dict{String, Dict{String, Int}}

One row per page, keyed by the path the baseline names.
"""
function measure(; root::AbstractString = REPO_ROOT)
    glossary = glossary_pattern(glossary_terms(; root = root))
    return Dict(p => counts(joinpath(root, p); glossary = glossary)
                for p in pages(; root = root))
end

binding(row) = Dict(k => row[k] for k in first.(COUNTED) if row[k] > 0)

"""
    row_text(path, row) -> String

The baseline line for one page, which is what the census prints for a page that fails. A count at
zero is absent from the line, and `words` always prints.
"""
function row_text(path::AbstractString, row)
    parts = ["$k = $(row[k])" for k in first.(COUNTED) if row[k] > 0]
    push!(parts, "$WORDS = $(row[WORDS])")
    return "\"$path\" = { " * join(parts, ", ") * " }"
end

"""
    rises(row, recorded) -> Vector{Pair{String, Pair{Int, Int}}}

Every column of one page that stands above its recorded ceiling, as `column => recorded => now`. An
absent column is a ceiling of zero.
"""
function rises(row, recorded)
    acc = Pair{String, Pair{Int, Int}}[]
    for k in first.(COUNTED)
        old = get(recorded, k, 0)
        row[k] > old && push!(acc, k => (old => row[k]))
    end
    return acc
end

"""
    read_baseline(; root = REPO_ROOT) -> Dict{String, Dict{String, Int}}

The `[file]` table of the baseline, or an empty table when the file is absent.
"""
function read_baseline(; root::AbstractString = REPO_ROOT)
    path = joinpath(root, "code_health", NAME)
    if !(isfile(path))
        return Dict{String, Dict{String, Int}}()
    end
    data = TOML.parsefile(path)
    files = get(data, "file", Dict{String, Any}())
    return Dict{String, Dict{String, Int}}(k => Dict{String, Int}(kk => Int(vv)
                                                                  for (kk, vv) in v)
                                           for (k, v) in files)
end

# --- the commands ----------------------------------------------------------

function git_short_commit(; root::AbstractString = REPO_ROOT)
    try
        cmd = pipeline(Cmd(`git rev-parse --short HEAD`; dir = root); stderr = devnull)
        return strip(read(cmd, String))
    catch
        return "unknown"
    end
end

"""
    baseline_text(m; root = REPO_ROOT) -> String

The whole baseline file, with one row per page whose counted rules are not all at zero.
"""
function baseline_text(m; root::AbstractString = REPO_ROOT)
    io = IOBuffer()
    println(io, "# Generated by code_health/literate_prose.jl. Do not edit by hand.")
    println(io, "# One row per Literate page that still carries a counted rule of")
    println(io, "# .github/instructions/julia-prose.instructions.md. A count may fall and")
    println(io, "# may not rise, and a page whose every count is zero carries no row.")
    println(io, "# An absent count is zero. `words` is context and carries no limit.")
    println(io, "# ADR 0171.")
    println(io)
    println(io, "[provenance]")
    println(io, "julia = \"", VERSION, "\"")
    println(io, "commit = \"", git_short_commit(; root = root), "\"")
    println(io, "glossary_terms = ", length(glossary_terms(; root = root)))
    println(io)
    println(io, "[file]")
    for p in sort!(collect(keys(m)))
        if isempty(binding(m[p]))
            continue
        end
        println(io, row_text(p, m[p]))
    end
    return String(take!(io))
end

"""
    scan(paths; root = REPO_ROOT)

Print the counts of each page in `paths`, and the row the baseline would carry for it. This is what
a rewrite session runs on its own pages, before and after the rewrite.
"""
function scan(paths; root::AbstractString = REPO_ROOT)
    glossary = glossary_pattern(glossary_terms(; root = root))
    for p in paths
        full = isabspath(p) ? p : joinpath(root, p)
        rel = relpath(full, root)
        row = counts(full; glossary = glossary)
        b = binding(row)
        if isempty(b)
            println(rel, ": every counted rule is at zero, over ", row[WORDS],
                    " prose words. It needs no row.")
        else
            println(rel, ": ", row[WORDS], " prose words, ",
                    join(["$k = $(b[k])" for k in sort!(collect(keys(b)))], ", "))
            println("  ", row_text(rel, row))
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    verb = isempty(ARGS) ? "scan" : ARGS[1]
    if verb == "refresh"
        write(joinpath(DIR, NAME), baseline_text(measure()))
        println("Refreshed code_health/", NAME, ".")
    elseif verb == "scan"
        scan(length(ARGS) > 1 ? ARGS[2:end] : pages())
    else
        println(stderr, "Usage: literate_prose.jl [scan [file...] | refresh]")
        exit(2)
    end
end
