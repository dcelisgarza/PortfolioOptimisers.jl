@testset "Sweep census: every source file carries a row, and its unit count holds" begin
    using Test, TOML

    #=
    The map of maps, issue #404, sweeps every file under `src/` and `ext/` for three things
    at once: that its documentation states the mathematics, that its code agrees with that
    statement when checked with real numbers, and that its lines are covered or exempted
    with a reason. Thirteen child maps carry the work, one per subsystem.

    That map's standing rule is: ANY ADDITION JOINS ITS CHILD MAP, and a closed child map --
    and the umbrella -- is reopened. Nothing here can enforce that rule directly. A Julia
    test cannot read the state of a GitHub issue, and it cannot reopen one. So this census
    makes the omission impossible to MISS instead. It turns the build red on the day the
    addition lands, and the failure names the child map that owns the file.

    A row per file with no measurement would not have done it. The case the rule is really
    aimed at is a type or a function added to an EXISTING file, and such a file already has
    a row. So each row carries a NUMBER, and check 3 compares it.

    #351 is why the rule exists. It swept 719 type docstrings over 32 tickets and found more
    than fifteen code defects on the way, in a sweep whose destination was documentation
    alone. Not one pass came back clean. A file that joins the library after its child map
    closes gets none of that attention, and before this census nothing said so. The rule was
    given on 2026-08-21, while #404 was charted, and #407 built this file.

    ------------------------------------------------------------------- what a unit is

    A UNIT is a docstring that attaches to a binding: a type, a function, a method, a
    constant, a macro or a module. The count is taken from the file's SOURCE TEXT, by
    parsing with `Meta.parseall` and counting the `Core.@doc` macrocalls at any depth.

    Three properties earn it the row.

     1. It is one parse per file and it loads no package, so the whole census costs about a
        second. `test_41_constructor_docstring_drift.jl` reads `src/` the same way, and for
        the same reason.
     2. It cannot move under a reformat. JuliaFormatter rewrites whitespace and rewraps
        docstring prose. It never adds or removes a docstring.
     3. It rises on every addition the docstring standard demands a docstring for. A new
        type, a new function, and a second documented method of a function already in the
        file each add one. `test_26_docs.jl` holds `Base.undocumented_names` at zero for the
        public and the private names both, so an addition cannot dodge the count by carrying
        no docstring.

    A FIELD docstring is NOT a unit. Inside a struct body a docstring parses as a bare
    string literal rather than as a `Core.@doc` macrocall, so it never reaches this count.
    That is the same boundary #404 measured when it recorded 2764 docstrings against 720
    types, so the manifest's numbers and the map's are one measurement.

    Two blind spots follow, and both are accepted.

      - A file whose types are declared by a Declaration Macro measures ZERO, because the
        macro writes the docstring and the calling file's text holds none. The five
        `Windowed*` files under `src/08_Moments/` are the case, and `code_health`'s
        complexity baseline records the same five as structural zeros (ADR 0072). Adding a
        sixth windowed estimator to one of those files stays green. Adding a new FILE does
        not.
      - `ext/` carries no docstring at all today, so both of its rows start at zero. #404
        puts `ext/` in scope for exactly that reason, and `test_26_docs.jl` does not yet
        reach it. Issue #409 is the ticket that teaches it to.
    =#

    root = normpath(joinpath(@__DIR__, ".."))
    manifest_path = joinpath(root, "sweep", "manifest.toml")

    manifest = TOML.parsefile(manifest_path)
    rows = manifest["file"]
    map_names = manifest["map"]

    # ------------------------------------------------------------- the expected file set

    #=
    ADR 0074 rules the row set TOTAL: the gate derives the expected set itself and compares
    two SETS, rather than inventing a number for a file it has never measured. A deleted
    file and an added file then share one rule.

    `git ls-files` is used rather than `walkdir`, for the reason `code_health/CodeHealth.jl`
    gives: `walkdir` ignores `.gitignore` and picks up the untracked `NOTRACK_*` scratch
    files. Those three lines are repeated here rather than included, because that module
    lives in its own environment and pulls JET and CodeComplexity in with it.
    =#
    function tracked_sources(dir)
        out = read(Cmd(`git ls-files -z -- '*.jl'`; dir = dir), String)
        fs = filter!(!isempty, split(out, '\0'))
        filter!(f -> startswith(f, "src/") || startswith(f, "ext/"), fs)
        return sort!(String.(fs))
    end

    expected = tracked_sources(root)
    # A `git` that answers nothing would make every check below vacuously green.
    @test !isempty(expected)

    # ------------------------------------------------------------------- the measurement

    #=
    `CodeHealth.documented_units` holds a second copy of the next fifteen lines, because
    `code_health/sweep_check.jl` measures the same number before the commit and this test may not
    depend on `code_health/`. THIS FILE IS THE AUTHORITY for the definition. A change here changes
    that copy in the same edit.
    =#
    doc_macro = GlobalRef(Core, Symbol("@doc"))
    isdocstring(x) = Meta.isexpr(x, :macrocall) &&
                     !isempty(x.args) &&
                     (x.args[1] === doc_macro || x.args[1] === Symbol("@doc"))

    function count_units(path::AbstractString)
        n = 0
        function walk(node)
            node isa Expr || return nothing
            isdocstring(node) && (n += 1)
            foreach(walk, node.args)
            return nothing
        end
        walk(Meta.parseall(read(path, String)))
        return n
    end

    #=
    Print the row a human must paste back into the manifest. A swept row also carries the
    `algorithm` key that `test_26_docs.jl` ratchets, so the printer takes it: a line pasted
    without it would delete the ratchet's floor and the deletion would read as a correction.
    An unswept row has no such key, and a file that has no row at all is never swept.
    =#
    function row_line(f, m, u, s; algorithm = nothing)
        a = isnothing(algorithm) ? "" : string(", algorithm = ", algorithm)
        return string("\"", f, "\" = { map = ", m, ", units = ", u, a, ", swept = ", s,
                      " }")
    end

    #=
    The map a file belongs to is NOT derivable from its path, so the census prints the
    CANDIDATES and a person chooses by subject. Each of the nine subdirectories of `src/`
    and `ext/` maps to exactly one child map, and there the answer is printed outright. The
    top level of `src/` holds sixteen files across FIVE maps, and the numeric prefix does
    not rescue the lookup: the blocks are not contiguous. `10_` sits between map 2 and map
    8, and `25_` returns to map 1. #428 measured this.

    The candidates are the maps the file's own directory already uses, so nothing here
    repeats the cut. A brand-new directory has no sibling row, and then every map is a
    candidate.
    =#
    all_maps() = sort(parse.(Int, collect(keys(map_names))))
    function candidate_maps(f)
        d = dirname(f)
        ms = sort(unique(r["map"] for (g, r) in rows if dirname(g) == d))
        return isempty(ms) ? all_maps() : ms
    end

    # The printer below runs only when the census is already red, so these two hold it to
    # its contract on a green run. A file's own directory must offer that file's map, and
    # the top level of `src/` must offer more than one -- which is the whole reason a
    # candidate list is printed rather than an answer.
    @test all(f -> rows[f]["map"] in candidate_maps(f), collect(keys(rows)))
    @test length(candidate_maps("src/00_Nothing.jl")) > 1

    # ------------------------------------------------------- 1. a file that has no row

    missing_rows = sort(collect(setdiff(expected, keys(rows))))
    @test isempty(missing_rows)
    if !isempty(missing_rows)
        println("Files under `src/` or `ext/` with no row in `sweep/manifest.toml`. Join ",
                "each one to a child map of #404, reopen that map if it is closed, then ",
                "add its row:")
        for f in missing_rows
            cands = candidate_maps(f)
            units = count_units(joinpath(root, f))
            if length(cands) == 1
                m = only(cands)
                println("  ", row_line(f, m, units, false), "   [map ", m, ": ",
                        map_names[string(m)], "]")
            else
                println("  ", row_line(f, "?", units, false))
                println("    The directory offers no single map. Choose by subject:")
                for m in cands
                    println("      map ", m, ": ", map_names[string(m)])
                end
            end
        end
    end

    # ------------------------------------------------------- 2. a row that names no file

    dead_rows = sort(collect(setdiff(keys(rows), expected)))
    @test isempty(dead_rows)
    if !isempty(dead_rows)
        println("Rows in `sweep/manifest.toml` that name no tracked file. A deletion drops ",
                "the row. A rename moves it, and the file keeps its `swept` flag:")
        for f in dead_rows
            println("  ", f)
        end
    end

    # ------------------------------------------------------- 3. the unit count drifted

    drifted = Tuple{String, Int, Int}[]
    for f in expected
        haskey(rows, f) || continue
        measured = count_units(joinpath(root, f))
        measured == rows[f]["units"] || push!(drifted, (f, rows[f]["units"], measured))
    end

    @test isempty(drifted)
    if !isempty(drifted)
        println("Files whose documented-unit count no longer matches `sweep/manifest.toml`.",
                " Join the addition to the file's child map of #404, reopen that map if it ",
                "is closed, then record the new count:")
        for (f, was, now) in drifted
            row = rows[f]
            println("  ", f, "  ", was, " -> ", now, "   [map ", row["map"], ": ",
                    map_names[string(row["map"])], "]")
            println("    ",
                    row_line(f, row["map"], now, row["swept"];
                             algorithm = get(row, "algorithm", nothing)))
        end
    end

    # --------------------------------------------------- the cut is total and disjoint

    #=
    #404 cuts the files into thirteen child maps and asserts the cut total and disjoint. A
    TOML table cannot hold one key twice, so disjointness is free, and check 1 is totality.
    What is left is that every row names a real child map, and that no child map has lost
    its last file. A map with no file is a map that nobody will ever close.
    =#
    unknown_map = sort([f for (f, r) in rows if !haskey(map_names, string(r["map"]))])
    @test isempty(unknown_map)
    if !isempty(unknown_map)
        println("Rows naming a child map that `[map]` does not list:")
        for f in unknown_map
            println("  ", f, "  map = ", rows[f]["map"])
        end
    end

    @test issetequal(Set(string(r["map"]) for r in values(rows)), keys(map_names))

    # A `swept` that is not a Bool would read as true under a later gate.
    @test all(r -> r["swept"] isa Bool, values(rows))
    @test all(r -> r["units"] isa Int, values(rows))
    # `test_26_docs.jl` ratchets this key on a swept row, and demands it there. A `Float64`
    # or a string would compare against a measured `Int` in ways that no reader expects.
    @test all(r -> !haskey(r, "algorithm") || r["algorithm"] isa Int, values(rows))
end
