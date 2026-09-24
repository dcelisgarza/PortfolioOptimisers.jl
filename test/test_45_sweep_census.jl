#=
`code_health/CodeHealth.jl` is a module rather than a script, and it loads only `TOML`, which
`test/Project.toml` already carries. So this census reads the source text with the SAME parser
`code_health/sweep_check.jl` runs before the commit, and neither reading can drift from the other.
`test_49_coverage_attribution_census.jl` loads `code_health/coverage.jl` the same way, and for the
same reason.

The load sits OUTSIDE the `@testset` on purpose. `include` defines methods, and a method defined
inside one top-level statement is not visible to a call in that same statement. The module wrapper
keeps that module's own names out of the worker module.
=#
module SweepCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Sweep census: every source file carries a row, its unit count holds, and a swept file's unit names hold" begin
    using Test, TOML

    CH = SweepCensusHealth.CodeHealth

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
    `CodeHealth.documented_units` is that count and `CodeHealth.isdocstring` is that
    predicate. THIS FILE STATES THE DEFINITION; that module holds the one implementation of
    it, and every other census in the repository reads the source through the same two.

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
        `Windowed*` files under `src/05_Moments/` are the case, and `code_health`'s
        complexity baseline records the same five as structural zeros (ADR 0072). Adding a
        sixth windowed estimator to one of those files stays green. Adding a new FILE does
        not.
      - `ext/` carries no docstring at all today, so both of its rows start at zero. #404
        puts `ext/` in scope for exactly that reason, and `test_26_docs.jl` does not yet
        reach it. Issue #409 is the ticket that teaches it to.
    =#

    root = normpath(joinpath(@__DIR__, ".."))
    manifest_path = joinpath(root, "code_health", "sweep_manifest.toml")

    manifest = TOML.parsefile(manifest_path)
    rows = manifest["file"]
    map_names = manifest["map"]

    # ------------------------------------------------------------- the expected file set

    #=
    ADR 0074 rules the row set TOTAL: the gate derives the expected set itself and compares
    two SETS, rather than inventing a number for a file it has never measured. A deleted
    file and an added file then share one rule.

    `CodeHealth.source_files` answers it, and it reads `git ls-files` rather than `walkdir`:
    `walkdir` ignores `.gitignore` and picks up the untracked `NOTRACK_*` scratch files
    (issue #336).
    =#
    expected = CH.source_files()
    # A `git` that answers nothing would make every check below vacuously green.
    @test !isempty(expected)

    # ------------------------------------------------------------------- the measurement

    #=
    The measurement and the two printers below are `CodeHealth`'s, so the number this
    census reds the build on is the number `code_health/sweep_check.jl` prints before the
    commit.

      - `count_units` counts the documented units of one file.
      - `count_bindings` names them, sorted, one entry per unit. Check 4 states why.
      - `row_line` prints the manifest row a person pastes back. A swept row also carries
        the `algorithm` key that `test_26_docs.jl` ratchets and the `bindings` list check 4
        compares, so the printer takes both: a line pasted without either would delete a
        record and the deletion would read as a correction. An unswept row has neither, and
        a file that has no row at all is never swept.
      - `candidate_maps` lists the child maps a file's own directory already uses, because
        the map a file belongs to is NOT derivable from its path. A person chooses by
        subject. #428 measured why the numeric prefix does not rescue the lookup.
    =#
    count_units(path::AbstractString) = CH.documented_units(path)
    count_bindings(path::AbstractString) = CH.documented_bindings(path)
    function row_line(f, m, u, s; algorithm = nothing, bindings = nothing)
        return CH.row_line(f, m, u, s; algorithm, bindings)
    end
    candidate_maps(f) = CH.candidate_maps(rows, map_names, f)

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
        println("Files under `src/` or `ext/` with no row in `code_health/sweep_manifest.toml`. Join ",
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
        println("Rows in `code_health/sweep_manifest.toml` that name no tracked file. A deletion drops ",
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
        println("Files whose documented-unit count no longer matches `code_health/sweep_manifest.toml`.",
                " On an unswept row, join the addition to the file's child map of #404 and ",
                "reopen that map if it is closed. On a swept row, sweep the new units in this ",
                "commit (ADR 0148). Then record the new count:")
        for (f, was, now) in drifted
            row = rows[f]
            println("  ", f, "  ", was, " -> ", now, "   [map ", row["map"], ": ",
                    map_names[string(row["map"])], "]")
            println("    ",
                    row_line(f, row["map"], now, row["swept"];
                             algorithm = get(row, "algorithm", nothing),
                             bindings = if row["swept"]
                                 count_bindings(joinpath(root, f))
                             else
                                 nothing
                             end))
        end
    end

    # ------------------------------------------- 4. a swept file's unit set drifted

    #=
    Check 3 sees an addition and a deletion. It cannot see a REPLACEMENT: a unit deleted and a
    unit added in one change leave the count where it was. Issue #1065 is the case.
    `src/06_Distance/05_FeatureDistance.jl` was swept, and map #802 then rewrote its
    estimator and its entry points -- two units deleted, two added -- while the count held
    at 24 and the row kept `swept = true`. Nothing red. The swept flag then vouched for text
    the sweep had never read.

    So a swept row also records the NAMES: the binding each unit attaches to, one entry per
    unit, sorted. `CodeHealth.documented_bindings` is that list, and it walks the same
    macrocalls the count does, so its length is `units`. A documented method names its
    function, and a function with three documented methods is listed three times. The
    names are stable under a reformat for the same reason the count is.

    An unswept row carries no list. The file has not passed the sweep, so there is no swept
    text to record, and a list on such a row would be a stale claim the moment the file
    changed. The session that flips `swept` writes the list in the same edit, as it writes
    `algorithm`, and the printer below hands it the line.

    What the failure asks for is the same as check 3's on a swept row: record the new row,
    and sweep the rewritten units in the same commit. The row keeps its `swept` flag: the flag
    arms the swept standard in `test/test_26_docs.jl`, and a rewrite must still meet it. No
    sub-issue is owed, because it would record nothing that the commit does not (ADR 0148
    § Amendment, issue #1310). A commit that cannot meet the standard sets `swept = false`,
    and the change then joins the child map as an addition to an unswept file does. The blind spot that remains is a unit rewritten UNDER its own name -- a method whose
    signature changed but whose function did not -- and that is accepted: the names gate the
    set, and the sweep gates the text.
    =#
    swept_rows = sort([f for (f, r) in rows if r["swept"] === true])
    unswept_rows = sort([f for (f, r) in rows if r["swept"] !== true])

    # The key is demanded of a swept row and refused on an unswept one, so a row that flips
    # either way is edited in full.
    no_bindings = filter(f -> !haskey(rows[f], "bindings"), swept_rows)
    @test isempty(no_bindings)
    if !isempty(no_bindings)
        println("Rows marked `swept = true` in `code_health/sweep_manifest.toml` that carry no ",
                "`bindings` list. A swept row records the name of the binding each documented ",
                "unit attaches to. Paste the line:")
        for f in no_bindings
            row = rows[f]
            println("  ",
                    row_line(f, row["map"], row["units"], true;
                             algorithm = get(row, "algorithm", nothing),
                             bindings = count_bindings(joinpath(root, f))))
        end
    end

    stale_bindings = filter(f -> haskey(rows[f], "bindings"), unswept_rows)
    @test isempty(stale_bindings)
    if !isempty(stale_bindings)
        println("Rows marked `swept = false` in `code_health/sweep_manifest.toml` that carry a ",
                "`bindings` list. Only a swept row records one. Drop the key:")
        for f in stale_bindings
            row = rows[f]
            println("  ", row_line(f, row["map"], row["units"], false))
        end
    end

    @test all(f -> !haskey(rows[f], "bindings") || (rows[f]["bindings"] isa Vector &&
                                                    all(x -> x isa String, rows[f]["bindings"])),
              swept_rows)

    renamed = Tuple{String, Vector{String}, Vector{String}}[]
    for f in swept_rows
        haskey(rows[f], "bindings") || continue
        f in expected || continue
        recorded = sort(String.(rows[f]["bindings"]))
        measured = count_bindings(joinpath(root, f))
        measured == recorded || push!(renamed, (f, recorded, measured))
    end

    @test isempty(renamed)
    if !isempty(renamed)
        println("Swept files whose documented units are no longer the ones ",
                "`code_health/sweep_manifest.toml` records. The sweep passed a text these files ",
                "no longer hold. Sweep the rewritten units in this commit, or set `swept = ",
                "false` and join the change to the file's child map of #404 as an addition ",
                "(ADR 0148). Then record the new list:")
        for (f, recorded, measured) in renamed
            row = rows[f]
            # A multiset difference: a method docstring added to a function that already has
            # one is one more entry under the same name, so a `setdiff` would miss it.
            added = copy(measured)
            for n in recorded
                i = findfirst(==(n), added)
                i === nothing || deleteat!(added, i)
            end
            removed = copy(recorded)
            for n in measured
                i = findfirst(==(n), removed)
                i === nothing || deleteat!(removed, i)
            end
            println("  ", f, "   [map ", row["map"], ": ", map_names[string(row["map"])],
                    "]")
            isempty(added) || println("    added:   ", join(added, ", "))
            isempty(removed) || println("    removed: ", join(removed, ", "))
            println("    ",
                    row_line(f, row["map"], length(measured), true;
                             algorithm = get(row, "algorithm", nothing),
                             bindings = measured))
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
