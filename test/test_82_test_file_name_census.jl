@testset "Test file name census: a test file named in live text exists (#1763)" begin
    using Test

    #=
    A comment, a doc or a tool that names a test file sends the reader to that file. When a
    commit merges, renames or deletes test files, every name of the old file stays where it
    was, and nothing fails. The balance commit `2a6bfe5d59` merged most of the `test_NNx`
    files into their family file, and eleven live sites went on naming the merged files: the
    usage examples of `test/run_files.jl`, a Rationale of `code_health/rulings.toml`, two
    `#src` notes of an example, and comments in six test files. #1763 corrected them and
    added this census, so that the next merge cannot leave the same kind of name.

    WHAT IS A NAME. Two shapes are checked, and both start with `test_` and a number:

      - a full file name, `test_<number><anything>.jl`, must be a file under `test/`;
      - a prefix, `test_<number>` or `test_<number><letters>`, with no word character after
        it, must start the name of a file under `test/`. `test_06b` needs a file that starts
        with `test_06b_`, and a family such as `test_12` needs a file that starts with
        `test_12` and then a letter or `_`.

    A name with no number after `test_` is a placeholder (`test_X.jl`, `test_fixture.jl`) or
    a synthetic input of a census, and no test file has that shape, so it is not checked.

    WHAT IS LIVE TEXT. Every tracked text file of the repository, with two exclusions:

      - `docs/adr/`. A released ADR is amended, never rewritten, so an old name there is
        correct history. A draft ADR is excluded with it, which costs little: an ADR that
        names a test file names it as evidence, and the merge that made it stale is history
        too.
      - this file, which names the dead shapes it refuses in its own comments and tests.

    The generated pages are untracked, so their `.jl` sources are what this census reads.
    =#

    root = normpath(joinpath(@__DIR__, ".."))
    test_files = sort!(filter(f -> endswith(f, ".jl"), readdir(joinpath(root, "test"))))
    tracked = split(read(`git -C $root ls-files`, String), '\n'; keepempty = false)
    self = "test/" * basename(@__FILE__)
    exts = (".jl", ".md", ".toml", ".yml", ".yaml", ".sh", ".py", ".txt")
    corpus = sort!([String(p)
                    for p in tracked
                    if any(e -> endswith(p, e), exts) &&
                           !startswith(p, "docs/adr/") &&
                           p != self])

    full_name = r"\btest_\d[A-Za-z0-9_]*\.jl"
    prefix = r"\btest_(\d+[a-z]*)(?![A-Za-z0-9_]|\.jl)"

    function dead_names(line, files)
        dead = String[]
        for m in eachmatch(full_name, line)
            m.match in files || push!(dead, m.match)
        end
        for m in eachmatch(prefix, line)
            p = "test_" * m.captures[1]
            live = if isletter(last(p))
                any(f -> startswith(f, p * "_"), files)
            else
                any(f -> startswith(f, p) &&
                         length(f) > length(p) &&
                         (f[length(p) + 1] == '_' || isletter(f[length(p) + 1])), files)
            end
            live || push!(dead, p)
        end
        return dead
    end

    @testset "the matcher" begin
        files = ["test_04_owa.jl", "test_06b_asset_panel.jl", "test_12k_factor.jl",
                 "test_12zb_carry.jl", "test_17a_hrp.jl"]
        @test dead_names("see `test_04_owa.jl` and `test_06b`", files) == String[]
        @test dead_names("the `test_12` family, and test_17*", files) == String[]
        @test dead_names("`test_12zb` and `test_12zb_carry.jl`", files) == String[]
        @test dead_names("`test_X.jl`, `test_fixture.jl`, `mytest_31`", files) == String[]
        @test dead_names("`test_12k_parity.jl` drives it", files) == ["test_12k_parity.jl"]
        @test dead_names("`test_06g` owns it, and `test_31` covers it", files) ==
              ["test_06g", "test_31"]
        @test dead_names("which test_06i pins.", files) == ["test_06i"]
        # `test_1` is not a prefix of the family `test_12`.
        @test dead_names("`test_1`", files) == ["test_1"]
    end

    # One entry per file and dead name, so that an edit above a site does not move its key.
    # `grep -n` finds the lines.
    found = String[]
    for p in corpus
        path = joinpath(root, p)
        isfile(path) || continue
        for line in eachline(path), n in dead_names(line, test_files)
            push!(found, "$p $n")
        end
    end
    unique!(sort!(found))

    # A file that names a test file which does not exist, and is right to. Each entry needs a
    # reason.
    KEEP = String[
                  # A synthetic input of `mechanism_name`: the census asks whether a
                  # file name is a mechanism name, and the file can be absent.
                  "test/test_74_source_citation_census.jl test_28_seam_lock.jl"]

    # A dead name: correct it to the file that holds the test now. The first `@testset`
    # name of the old file finds that file. If the site is right to name a missing file,
    # add it to `KEEP` with its reason.
    @test setdiff(found, KEEP) == String[]

    # A stale entry names a site that is gone. Remove it.
    @test setdiff(KEEP, found) == String[]
end
