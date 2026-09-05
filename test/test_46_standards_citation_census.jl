@testset "Standards citation census: a citation resolves, and no file states a count" begin
    using Test, TOML

    #=
    A standards file tells a contributor what to do. When it cites a name that no longer
    exists, or a path that was never there, it does not fail loudly: it sends the reader --
    a person or an agent -- to look for something that is not there. The reader then either
    invents a replacement or copies the dead name into new code. A wrong instruction is
    worse than a missing one.

    Nothing gated the standards files themselves before this census. `test_26_docs.jl`
    checks the citations inside `src/` and `ext/` docstrings. `markdownlint` checks the
    structure of a markdown file, not the truth of a name inside a code span. So a rename in
    `src/` left every mention of the old name in `.github/` standing, and no check noticed.

    Two crude greps over the standards files during #478 found four dead names in one pass,
    and writing this file found four more. All eight are fixed in the commit that adds it:

      - `AbstractDetoneAlgorithm` and `AbstractPosdefAlgorithm`, cited by
        `.github/prompts/add-algorithm.prompt.md` as abstract algorithm supertypes to pick
        from. Detone and posdef carry an ESTIMATOR family, not an algorithm family, so
        neither name has ever existed.
      - `AbstractLowOrderPriorResult`, cited by `.github/prompts/add-result.prompt.md`. The
        low-order/high-order split is on the ESTIMATOR side; both results share
        `AbstractPriorResult`.
      - `ClustersResult`, cited by `.github/instructions/julia-source-code.instructions.md`
        as an example result type. The type is called `Clusters`.
      - `moment_view` and `prior_view`, cited five times over three files as the slicing
        seam an estimator implements. The seam is `port_opt_view`, with `obs_weights_view`
        beside it, and neither cited name has ever existed.
      - `test-*.jl` and `test-<feature>.jl`, cited six times over five files as the name a
        new test file takes. Every test file in this repository is `test_*.jl`, with an
        underscore, and `test/runtests.jl:62` discovers on that prefix. A file named as the
        standard said would never run.

    The second check is a different failure. `STANDARDS.md` § *Precedence* carried a
    measurement: `dev` holds "25 ADRs that `main` has never seen", taken against `main` at a
    named commit. `main` reached that commit, the count fell to zero, and the paragraph went
    on stating it. A measurement in a standards file is a copy of the repository, and a copy
    drifts. This repository already says so twice -- state a rule without its history,
    and do not restate a count -- and neither statement was gated.

    ------------------------------------------------------- how the two checks are drawn

    CHECK 1 reads the SPANS. Everything inside single or double backticks, on a line that a
    fenced code block does not cover, is a citation.

    A FENCED BLOCK IS OUT OF SCOPE, and that was measured rather than assumed. Scanning
    every fenced block of every file in scope for a call head and a capitalised name gave 27
    candidates that the corpus does not hold, and all 27 are placeholders -- `MyType`,
    `ConcreteSubtype1`, `my_function` -- or ordinary English words inside a code comment:
    "Allows", "Bad", "Longer", "Preferred", "Specific". Gating a fenced block would cost an
    allow-list of that size, which would then grow with every new example. An allow-list
    that large is a place for a real defect to hide, so the trade is refused.

    The cost is real and is stated: the worked example in
    `julia-test-writing.instructions.md` called `moment_view` from inside a fenced block,
    and only a `grep` found it after the census was green. A dead name in an example is
    corrected by hand.

    A span is then classified before it is resolved, because a standards file puts many
    things in backticks that are not names:

      - A PATH citation has no whitespace, does not start with `/` -- that is a slash
        command -- and either ends in a known extension or ends in `/`. `1/N` and
        `Gerber0/1/2` fall out here, and so does a shell line, which carries a space.
      - A NAME citation is a single identifier. An initial capital gives a type or a
        constant; an initial lowercase gives a function, a field or a keyword.
      - Everything else is prose or a code fragment, and is not checked.

    Resolution is by TEXT, not by binding. The census reads the source of every tracked
    `.jl`, `.toml`, `.yml`, `.yaml` and `.json` file and asks whether the name occurs in it
    as a whole word. That loads no package and costs about a second, which is the same
    trade `test_41_constructor_docstring_drift.jl` and `test_45_sweep_census.jl` make.

    The cost of the trade is stated plainly: a name that survives only in a comment or in
    prose still resolves. The check catches a name that is GONE, which is the failure that
    happened five times, and it does not catch a name that is merely wrong about its role.

    CHECK 2 reads the same prose for a MEASUREMENT. The distinction it must draw is between
    a RULE CONSTANT and a MEASUREMENT, because both are numbers:

      - A RULE CONSTANT is part of the rule. Change it and the rule changes. "The margin is
        92." "Do not use a noun cluster of more than three nouns." Nothing in the repository
        can make it stale.
      - A MEASUREMENT counts something the repository holds. Change the repository and the
        number is wrong, with no edit to the file that states it. "25 ADRs." "233 of 240
        abstract types are unexported."

    The two are separated by the NOUN the number governs, not by the number. A measurement
    counts a repository ARTEFACT -- an ADR, a file, a type, an export, a test, a commit. A
    rule constant governs a unit of language or of layout -- a word, a column, a noun, a
    sentence. So the check flags a numeral followed within three words by an artefact noun,
    and it flags a bare commit SHA, which is the anchor a measurement is taken against.

    Three forms are stripped first, because each is an IDENTIFIER that merely contains
    digits, not a count: an issue reference, an ADR or a version number, and an ISO date. A
    numeral whose preceding word is itself an artefact noun is skipped for the same reason:
    "ADR 0037" names a decision, while "25 ADRs" counts them.
    =#

    root = normpath(joinpath(@__DIR__, ".."))

    # ------------------------------------------------------------------------ the scope

    #=
    The four fixed files, plus every markdown file in the two instruction directories, so
    that a new instructions file or a new prompt joins the census on the day it lands.
    `docs/src/contribute/` is the sixth tier of the Precedence list and is NOT in scope: it
    is a rendered documentation page, and `.github/workflows/LinkChecker.yml` already walks
    the links of the built documentation.
    =#
    standards_files = let fs = String["STANDARDS.md", "CONTEXT.md", "CLAUDE.md",
                                      ".github/copilot-instructions.md"]
        for d in (".github/instructions", ".github/prompts")
            for f in sort(readdir(joinpath(root, d)))
                endswith(f, ".md") && push!(fs, joinpath(d, f))
            end
        end
        fs
    end

    for f in standards_files
        @test isfile(joinpath(root, f))
    end

    # ------------------------------------------------------------------ the two corpora

    tracked = split(read(`git -C $root ls-files`, String), '\n'; keepempty = false)

    #=
    THIS FILE IS EXCLUDED FROM ITS OWN CORPUS. The comment above names every dead name the
    census was built to catch, and the comment is source text in a tracked `.jl` file. Left
    in, it would make all seven of them resolve, and the census would pass on the very
    defects that motivated it. A mutation test caught exactly that:
    `AbstractDetoneAlgorithm` was put back into `add-algorithm.prompt.md`, and the census
    stayed green.
    =#
    this_file = relpath(normpath(@__FILE__), root)
    @test isfile(joinpath(root, this_file))   # the exclusion must name a real file

    #=
    Every word of every tracked source and configuration file. `.md` is deliberately absent:
    a name that exists only in another markdown file has not been shown to exist in the
    library, and one standards file agreeing with another is exactly the drift this census
    is here to find.
    =#
    corpus = let w = Set{String}()
        for f in tracked
            f == this_file && continue
            any(endswith(f, e) for e in (".jl", ".toml", ".yml", ".yaml", ".json")) ||
                continue
            for m in eachmatch(r"[A-Za-z_][A-Za-z0-9_!]*", read(joinpath(root, f), String))
                push!(w, m.match)
            end
        end
        w
    end

    #=
    A bare `Foo.jl` span in prose names a PACKAGE, not a file in this repository. It
    resolves against the environments this repository declares.
    =#
    packages = let names = Set{String}()
        for pf in ("Project.toml", "test/Project.toml", "docs/Project.toml",
                   "code_health/Project.toml")
            isfile(joinpath(root, pf)) || continue
            proj = TOML.parsefile(joinpath(root, pf))
            haskey(proj, "name") && push!(names, proj["name"])
            for section in ("deps", "weakdeps", "extras")
                haskey(proj, section) && union!(names, keys(proj[section]))
            end
        end
        names
    end

    #=
    A `git` that answered nothing, or a corpus that read nothing, would make every check
    below vacuously green. `test_45_sweep_census.jl` guards its own measurement the same way.
    =#
    @test !isempty(tracked)
    @test !isempty(corpus)
    @test !isempty(packages)

    # ------------------------------------------------------------------ the allow-lists

    #=
    A deliberate placeholder. It stands for the name the reader is about to choose, so it
    must NOT resolve. Prose carries only a handful; the rest live inside fenced blocks and
    never reach the scan.
    =#
    placeholders = ["My[A-Z]\\w*", "TypeName", "RelatedType", "PublicType", "PrivateType",
                    "AbstractSupertype", "ConcreteSubtype\\d*", "SomeFeature", "SomeFolder",
                    "AnotherFeature", "my_\\w+", "myfile", "f", "g"]
    placeholder_re = Regex("^(" * join(placeholders, "|") * ")\$")
    is_placeholder(s) = occursin(placeholder_re, s)

    #=
    A placeholder inside a path stands in one SEGMENT of it, and it carries the extension of
    the file it stands for: `src/SomeFeature.jl`, `docs/src/api/myfile.md`. An angle bracket
    is the other placeholder form this repository writes, as in `test_<feature>.jl`.
    =#
    function has_placeholder(s)
        occursin(r"<[^>]*>", s) && return true
        return any(is_placeholder(replace(part, r"\.\w+$" => "")) for part in split(s, '/'))
    end

    #=
    A name that is real, but is not a name of THIS repository, so the corpus cannot hold it.
    Each entry says why it is here.
    =#
    external_names = Dict("grep_code" => "a kaimon MCP tool, cited by `CLAUDE.md`",
                          "search_code" => "a kaimon MCP tool, cited by `CLAUDE.md`",
                          "start_session" =>
                              "a kaimon MCP tool, cited by `CLAUDE.md` " *
                              "§ Running Julia",
                          "EnterWorktree" =>
                              "a Claude Code tool, cited by `CLAUDE.md` " *
                              "§ Parallel sessions",
                          "ExitWorktree" =>
                              "a Claude Code tool, cited by `CLAUDE.md` " *
                              "§ Parallel sessions",
                          "servedocs" =>
                              "LiveServer.jl, a declared dependency of " *
                              "`docs/Project.toml`",
                          "_A" =>
                              "a suffix quoted alone in `CONTEXT.md` § Prior; real " *
                              "only inside `AbstractLowOrderPriorEstimator_A`",
                          "_F" => "the same, for `AbstractLowOrderPriorEstimator_F`",
                          "_AF" => "the same, for `AbstractLowOrderPriorEstimator_AF`")

    #=
    A path that names an artefact CI builds and never commits, so no tracked file can
    satisfy it. `CLAUDE.md` § Editing cites this one to say: do not edit it, edit its source.
    =#
    generated_paths = Dict("examples/**/*.ipynb" =>
                               "rendered by CI from " *
                               "`examples/**/*.jl`; never committed")

    # ----------------------------------------------------------------- the span reader

    #=
    A fenced block is skipped whole. The opening fence sets the marker; a fence of the same
    character and at least the same length closes it. Four backticks are used in these files
    to wrap an example that itself contains a three-backtick block, so the length matters.
    =#
    function prose_lines(text)
        out = Tuple{Int, String}[]
        fence = ""
        for (i, ln) in enumerate(split(text, '\n'))
            m = match(r"^\s*(`{3,}|~{3,})", ln)
            if !isempty(fence)
                if m !== nothing && m[1][1] == fence[1] && length(m[1]) >= length(fence)
                    fence = ""
                end
            elseif m !== nothing
                fence = String(m[1])
            else
                push!(out, (i, String(ln)))
            end
        end
        return out
    end

    code_spans(ln) = [strip(String(m.match), '`')
                      for m in eachmatch(r"``[^`]+``|`[^`]+`", ln)]

    citations = Dict{String, Vector{String}}()
    prose = Dict{String, Vector{Tuple{Int, String}}}()
    for f in standards_files
        text = read(joinpath(root, f), String)
        lines = prose_lines(text)
        prose[f] = lines
        for (i, ln) in lines, s in code_spans(ln)
            push!(get!(citations, s, String[]), "$f:$i")
        end
    end

    @test !isempty(citations)

    # ----------------------------------------------------- check 1a: a path resolves

    path_exts = (".jl", ".md", ".toml", ".yml", ".yaml", ".json", ".typ", ".cff", ".bib",
                 ".ipynb", ".sh")

    function is_path_citation(s)
        occursin(r"\s", s) && return false
        startswith(s, '/') && return false
        occursin(r"^[A-Za-z0-9_.\-*?/\[\]<>]+$", s) || return false
        occursin(r"^\.[A-Za-z]+$", s) && return false
        endswith(s, "/") && return true
        return any(endswith(s, e) for e in path_exts)
    end

    #=
    A glob is expanded against the tracked file list rather than the filesystem, so an
    ignored scratch file cannot satisfy a citation. A glob with no `/` is a BASENAME glob
    and matches at any depth, which is how `CLAUDE.md` and the test-writing standard use
    `test_*.jl`.
    =#
    function glob_matches(pat)
        rx = replace(pat, "." => "\\.", "**/" => "\x01", "*" => "[^/]*", "?" => "[^/]")
        rx = replace(rx, "\x01" => "(?:.*/)?")
        anchored = occursin('/', pat) ? "^$rx\$" : "^(?:.*/)?$rx\$"
        r = Regex(anchored)
        return filter(f -> occursin(r, f), tracked)
    end

    function path_resolves(s)
        p = String(rstrip(s, '/'))
        occursin(r"[*?\[]", p) && return !isempty(glob_matches(p))
        ispath(joinpath(root, p)) && return true
        occursin('/', p) && return false
        endswith(p, ".jl") && chop(p; tail = 3) in packages && return true
        return any(basename(f) == p for f in tracked)
    end

    dead_paths = String[]
    for (s, sites) in citations
        is_path_citation(s) || continue
        has_placeholder(s) && continue
        haskey(generated_paths, s) && continue
        path_resolves(s) || push!(dead_paths, "$s  cited by  " * join(sites, ", "))
    end

    if !isempty(dead_paths)
        @info "Standards files cite paths that do not exist" join(sort(dead_paths), "\n")
    end
    @test isempty(dead_paths)

    # ----------------------------------------------------- check 1b: a name resolves

    is_name_citation(s) = occursin(r"^[A-Za-z_][A-Za-z0-9_!]*$", s) && !endswith(s, ".jl")

    #=
    A span is often a CALL rather than a bare name: `port_opt_view(estimator, i)`,
    `factory(estimator, w::ObsWeights)`. The head of that call is a citation exactly as a
    bare name is, and it is where two dead names hid -- `moment_view` and `prior_view`
    survived the first pass of this census because their spans carry a comma and a space,
    and a span with whitespace is not an identifier. Only the head is read. An argument is
    a type or a value the example chose, not a claim that the name exists.
    =#
    function call_head(s)
        m = match(r"^([A-Za-z_][A-Za-z0-9_!]*)\(", s)
        return m === nothing ? nothing : String(m[1])
    end

    named = Dict{String, Vector{String}}()
    for (s, sites) in citations
        name = is_name_citation(s) ? s : call_head(s)
        name === nothing && continue
        append!(get!(named, name, String[]), sites)
    end

    dead_names = String[]
    for (s, sites) in named
        is_placeholder(s) && continue
        haskey(external_names, s) && continue
        s in corpus || push!(dead_names, "$s  cited by  " * join(sort(sites), ", "))
    end

    if !isempty(dead_names)
        @info "Standards files cite names that exist nowhere" join(sort(dead_names), "\n")
    end
    @test isempty(dead_names)

    # ------------------------------------------------ check 2: no repository count

    artefact_nouns = ["ADRs?", "files?", "types?", "exports?", "docstrings?", "tests?",
                      "commits?", "functions?", "methods?", "names?", "rows?", "lines?",
                      "issues?", "entries", "entry", "subtypes?", "supertypes?", "macros?",
                      "modules?", "structs?", "fields?", "packages?", "dependency",
                      "dependencies", "workflows?", "gates?", "rules?", "estimators?",
                      "measures?", "citations?", "references?", "constructors?", "alias",
                      "aliases", "arguments?", "keywords?", "checks?"]
    artefact = Regex("^(" * join(artefact_nouns, "|") * ")\$", "i")

    #=
    A noun that NAMES the thing the number belongs to, rather than counting a population of
    it. "ADR 0037" and "issue #404" are identifiers.
    =#
    labelling_nouns = ["ADRs?", "version", "issue", "chapter", "section", "eq", "equation",
                       "item", "step", "number", "figure", "table", "line", "page", "test",
                       "listing", "rule"]
    labelling = Regex("^(" * join(labelling_nouns, "|") * ")\$", "i")

    function measurements(lines)
        found = String[]
        for (i, ln) in lines
            for s in code_spans(ln)
                if occursin(r"^[0-9a-f]{7,40}$", s) &&
                   occursin(r"[a-f]", s) &&
                   occursin(r"[0-9]", s)
                    push!(found, "line $i: the commit `$s` is an anchor for a measurement")
                end
            end
            #=
            An ordered-list marker and an ATX heading number are POSITIONS, not counts.
            "3. Add docstrings" and "## 5. Risk Measures" both put a numeral in front of an
            artefact noun, and neither states anything about the repository. They are
            removed before the line is read.
            =#
            bare = replace(ln, r"^\s*#{1,6}\s+\d+[.)]?\s" => " ")
            bare = replace(bare, r"^\s*\d+[.)]\s" => " ")
            bare = replace(bare, r"``[^`]+``|`[^`]+`" => " ")
            bare = replace(bare, r"\]\([^)]*\)" => "]")
            bare = replace(bare, r"#\d+" => " ")
            bare = replace(bare, r"\bv?\d+\.\d+(\.\d+)?\b" => " ")
            bare = replace(bare, r"\b\d{4}-\d{2}-\d{2}\b" => " ")
            words = [strip(w,
                           ['*', '_', '"', '(', ')', ',', '.', ';', ':', '!', '?', '“',
                            '”']) for w in split(bare)]
            filter!(!isempty, words)
            for (j, w) in enumerate(words)
                occursin(r"^\d+$", w) || continue
                j > 1 && occursin(labelling, words[j - 1]) && continue
                for k in (j + 1):min(j + 3, length(words))
                    if occursin(artefact, words[k])
                        push!(found, "line $i: \"$w … $(words[k])\" counts the repository")
                        break
                    end
                end
            end
        end
        return found
    end

    counts = String[]
    for f in standards_files
        for m in measurements(prose[f])
            push!(counts, "$f, $m")
        end
    end

    if !isempty(counts)
        @info "Standards files state a count of the repository" join(sort(counts), "\n")
    end
    @test isempty(counts)
end
