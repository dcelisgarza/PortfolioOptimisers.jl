using PortfolioOptimisers
using ParallelTestRunner

#=
Tests run in parallel via ParallelTestRunner.jl: every `test_*.jl` file is
auto-discovered and executed in its own isolated worker module. Don't add
`include`s here — create a new `test_*.jl` file instead.

`init_code` runs in each test file's sandbox module before the file, so shared
setup (the common `using` core and `find_tol`) lives here once rather than being
duplicated per file. Files still `using` their own rare/heavy deps (StatsPlots,
GraphRecipes, Pajarito, HiGHS, SCS, Clustering, AverageShiftedHistograms, FLoops).
See docs/adr/0001-parallelise-test-suite.md.
=#
const init_code = quote
    using Test, PortfolioOptimisers
    using Logging
    using CSV, TimeSeries, DataFrames, StableRNGs, StatsBase, LinearAlgebra

    # Silence all logging by default.
    global_logger(NullLogger())

    # Headless GR for plotting tests; respect an externally set value.
    ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

    # Suggests the tightest passing rtol/atol for a failing comparison. Debug aid.
    function find_tol(a1, a2; name1 = :lhs, name2 = :rhs)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
end

# Discover only `test_*.jl` as test files. Other `.jl` files (e.g. `test18_setup.jl`,
# shared fixtures `include`d by split files) are intentionally excluded from the run.
testsuite = find_tests(@__DIR__)
filter!(kv -> startswith(basename(first(kv)), "test_"), testsuite)

runtests(PortfolioOptimisers, ARGS; init_code, testsuite)
