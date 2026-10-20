#!/usr/bin/env julia
#
# The two package-level censuses an addition under `src/` meets, run ahead of the test files that
# hold them.
#
#     julia -t 1 --project=test code_health/census_preview.jl
#
# `code_health/sweep_check.jl` reads source text and loads no package, so it cannot answer the two
# questions only the loaded module answers, and an agent met both only after a 215 s run of the
# targeted test files:
#
#   is every Choice in the Capability Catalogue?    `test/test_26_docs.jl`,
#                                                   "every choice is catalogued"
#   does every abstract type with a concrete        `test/test_61_interfaces_section_census.jl`
#   member carry an `# Interfaces` section?
#
# This script answers both through the code that owns each rule, never through a copy of it:
#
#   - the catalogue through `assert_complete` of `docs/generate_capability_catalogue.jl`, which the
#     docs build calls too. It reads the Choice Surface with `choice_surface_names`, the function
#     `test/test_26_docs.jl` reads it with. It reports a choice with no entry, which is the failure
#     an addition meets. The two other directions of that testset, a stale and a contradictory
#     `NOT_A_CHOICE` entry, follow a removal or a rename, and `test/test_26_docs.jl` alone holds them;
#   - the `# Interfaces` census by running `test/test_61_interfaces_section_census.jl` itself, with
#     `run_file` of `test/run_files.jl`, and without the `init_code` preamble of `test/runtests.jl`:
#     the file loads what it reads, and the preamble's data packages cost time and answer nothing.
#
# It loads the package, so it costs what loading the package costs, plus about 30 s for the
# catalogue's walk of `subtypes` and 3 s for the census. The two test files cost about 225 s on
# the same tree (2026-10-15).
#
# It writes nothing: no gate stamp, no baseline. The test files stay the Authority, so a green run
# here is an early answer and not a reason to skip them. `sweep_check.jl` prints this command when a
# branch adds a declaration that the two censuses read.

const T0 = time()

module Catalogue
include(joinpath(@__DIR__, "..", "docs", "generate_capability_catalogue.jl"))
end

include(joinpath(@__DIR__, "..", "test", "run_files.jl"))

"""
    catalogue_check(io) -> Bool

Whether every choice has an entry in the Capability Catalogue, through `assert_complete`. Its
refusal is printed as it stands, because it names each missing choice and the file to add it to.
"""
function catalogue_check(io::IO = stdout)
    try
        Catalogue.assert_complete()
    catch e
        if !(e isa ErrorException)
            rethrow()
        end
        println(io, "[fail] ", e.msg)
        println(io, "       Owner: `test/test_26_docs.jl`, \"every choice is catalogued\".")
        return false
    end
    println(io, "[ ok ] every choice is catalogued (`test/test_26_docs.jl`).")
    return true
end

"""
    interfaces_check(io) -> Bool

Whether `test/test_61_interfaces_section_census.jl` is green, run as the file stands. A failure is
printed where it happens by the test set of `test/run_files.jl`, and it carries the list the test
compares: an abstract type in it owes an `# Interfaces` section.
"""
function interfaces_check(io::IO = stdout)
    path = joinpath(RunFiles.TEST_DIR, "test_61_interfaces_section_census.jl")
    ts, seconds = RunFiles.run_file(path; code = :(using Test))
    ok = RunFiles.green(ts)
    println(io, ok ? "[ ok ] " : "[fail] ", RunFiles.result_line(ts, seconds))
    return ok
end

function main()
    loaded = time() - T0
    ok = catalogue_check()
    ok = interfaces_check() && ok
    println(ok ? "Green" : "Red", ". ", round(time() - T0; digits = 1), " s in all, ",
            round(loaded; digits = 1), " s of it loading the package.")
    return ok ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
