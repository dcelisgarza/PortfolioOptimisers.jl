#=
`code_health/perf.jl` reads the source text for performance traps, and ADR 0175 states which shapes
it reads and why each one has a replacement with the same meaning. This file is the fixture adapter
of that reader: each rule is driven over a small file in a temporary tree, once on a shape it must
flag and once on a near shape it must leave alone. A near shape is where a lint goes wrong, so every
silent case below is one the rule's docstring names.

The gate itself runs in CI, as `jet.jl` and `complexity.jl` do. This file checks the counting, the
Dismissal, the ratchet and the entry rule for an added file.

The load sits OUTSIDE the `@testset`, for the reason `test_49_coverage_attribution_census.jl`
gives: `include` defines methods, and a method defined inside one top-level statement is not
visible to a call in that same statement. The module wrapper keeps the script's own names --
`measure`, `verify`, `render`, `publish` -- out of the worker module.
=#
module PerformanceTrap
include(joinpath(@__DIR__, "..", "code_health", "perf.jl"))
end

using Test, TOML

@testset "Performance trap census" begin
    P = PerformanceTrap

    #=
    The rules one source text sets off, as `(rule, code)` pairs in source order within each rule.
    =#
    function flagged(src::AbstractString)
        return mktempdir() do dir
            mkpath(joinpath(dir, "src"))
            write(joinpath(dir, "src", "a.jl"), src)
            return [(f.rule, f.code) for f in P.scan_file("src/a.jl"; root = dir)]
        end
    end
    rules(src) = first.(flagged(src))

    @testset "slice_copy" begin
        @test rules("f(X) = g(X[:, 1])") == ["slice_copy"]
        @test rules("f(x, n) = g(x[2:n])") == ["slice_copy"]
        # A Dismissal copies the code `scan` prints, so `end` prints as the word a reader wrote.
        @test flagged("f(x) = g(x[2:end])") == [("slice_copy", "x[2:end]")]
        @test rules("function f(X)\n    for j in axes(X, 2)\n        g(X[:, j])\n    end\nend") ==
              ["slice_copy"]
        # A vector index gives a view BLAS cannot stride, so the copy may be the faster read.
        @test isempty(rules("f(X, idx) = g(X[:, idx])"))
        # A `!` call mutates, and a view would carry the mutation back to the parent.
        @test isempty(rules("f(x) = sort!(x[1:3])"))
        # A constructor may keep the argument, and a view would alias it.
        @test isempty(rules("f(X) = Result(X[:, 1])"))
        @test isempty(rules("f(X) = @views g(X[:, 1])"))
        # `x[1:N] >= 0` in a JuMP macro declares a variable.
        @test isempty(rules("f(m, N) = JuMP.@variables(m, begin\n    x[1:N] >= 0\nend)"))
        # A tuple has no view.
        @test isempty(rules("f(X) = g(size(X)[1:2])"))
        # An error message runs once, on the failing call.
        @test isempty(rules("f(x) = throw(ArgumentError(string(x[1:3])))"))
    end

    @testset "reduce_temporary" begin
        @test rules("f(x) = sum(abs.(x))") == ["reduce_temporary"]
        @test rules("f(X) = mean(abs.(X); dims = 2)") == ["reduce_temporary"]
        @test rules("f(x) = any(x .< zero(eltype(x)))") == ["reduce_temporary"]
        @test rules("f(x, y) = sum(x .* y)") == ["reduce_temporary"]
        @test rules("f(xs) = sum([g(x) for x in xs])") == ["reduce_temporary"]
        # Two operands and `dims`: a generator takes no `dims`, so no replacement exists.
        @test isempty(rules("f(X, Y) = sum(X .* Y; dims = 1)"))
        @test isempty(rules("f(x) = sum(abs, x)"))
        @test isempty(rules("f(xs) = sum(g(x) for x in xs)"))
    end

    @testset "unfused_broadcast" begin
        @test rules("f(x, y, z) = x .* y + z") == ["unfused_broadcast"]
        @test rules("f(x) = -(log.(x))") == ["unfused_broadcast"]
        # The operand of the inner `-` may be a scalar, as `1 - a` is.
        @test isempty(rules("f(a, u) = (1 - a) .* u"))
        @test isempty(rules("f(x, y, z) = x .* y .+ z"))
        # `I` does not broadcast.
        @test isempty(rules("f(x) = Int.(x) - I"))
    end

    @testset "search_temporary" begin
        @test rules("f(m) = length(findall(m))") == ["search_temporary"]
        @test rules("f(x) = sort(x)[1]") == ["search_temporary"]
        @test rules("f(x, k) = sortperm(x; rev = true)[1:k]") == ["search_temporary"]
        @test rules("function f(n)\n    for i in collect(1:n)\n        g(i)\n    end\nend") ==
              ["search_temporary"]
        # `first(sort(x; by = g))` is not `minimum(x)`.
        @test isempty(rules("f(x) = first(sort(x; by = abs))"))
        @test isempty(rules("f(m) = count(m)"))
    end

    @testset "linalg_temporary" begin
        @test rules("f(A, b) = inv(A) * b") == ["linalg_temporary"]
        @test rules("f(v, A) = diagm(v) * A") == ["linalg_temporary"]
        @test rules("f(A, B) = tr(A * B)") == ["linalg_temporary"]
        @test rules("f(A, B) = diag(A * B)") == ["linalg_temporary"]
        @test rules("f(w, sigma) = w' * sigma * w") == ["linalg_temporary"]
        @test rules("f(A, B, x) = (A * B) * x") == ["linalg_temporary"]
        # `inv(alpha)` is a scalar.
        @test isempty(rules("f(alpha, b) = inv(alpha) * b"))
        # `d' * d * l` is a scalar times a vector, not a quadratic form.
        @test isempty(rules("f(d, l) = transpose(d) * d * l"))
        @test isempty(rules("f(A, B, x) = A * B * x"))
    end

    @testset "repeated_call" begin
        @test flagged("f(X) = g(cov(X)) + h(cov(X))") == [("repeated_call", "cov(X)")]
        @test rules("function f(x)\n    a = sum(x) / 2\n    return a + sum(x)\nend") ==
              ["repeated_call"]
        # The two arms of one ternary never both run.
        @test isempty(rules("f(X, a) = a ? g(cov(X)) : h(cov(X))"))
        # A mutated input may give a different answer the second time.
        @test isempty(rules("function f(x)\n    a = sum(x)\n    x .= 0\n    return a + sum(x)\nend"))
        # Two copies are often the point: each is mutated alone.
        @test isempty(rules("f(w) = State(copy(w), copy(w))"))
        # The message of an exception runs on the failing call alone.
        @test isempty(rules("f(D) = maximum(D) > 1 && throw(DomainError(maximum(D), \"\"))"))
    end

    @testset "loop_allocation" begin
        loop(body) = "function f(x, n)\n    for i in 1:n\n        $body\n    end\nend"
        @test rules(loop("g(zeros(n))")) == ["loop_allocation"]
        @test rules(loop("g([x[j] + i for j in 1:3])")) == ["loop_allocation"]
        @test rules("function f(x)\n    while true\n        g(copy(x))\n    end\nend") ==
              ["loop_allocation"]
        # A nested allocation is part of the outer one: one site, one Finding.
        @test flagged(loop("g(vcat(x, fill(i, n)))")) ==
              [("loop_allocation", "vcat(x, fill(i, n))")]
        # The loop keeps the array: it is the output, not a temporary.
        @test isempty(rules(loop("push!(out, zeros(n))")))
        @test isempty(rules(loop("out[i] = copy(x)")))
        # Outside a loop, and in a comprehension, which is the output of its own loop.
        @test isempty(rules("f(n) = g(zeros(n))"))
        @test isempty(rules("f(n) = [zeros(n) for i in 1:n]"))
        # The iterator of a `for` is read once. `search_temporary` reads this `collect`.
        @test !("loop_allocation" in
                rules("function f(x)\n    for v in collect(x)\n        g(v)\n    end\nend"))
        # A closure prints on one line, so its Dismissal is one TOML string.
        @test !any(c -> occursin('\n', c),
                   last.(flagged(loop("g([findfirst(y -> y == i, x) for k in 1:2])"))))
    end

    @testset "loop_allocation hints" begin
        hints(src) = mktempdir() do dir
            mkpath(joinpath(dir, "src"))
            write(joinpath(dir, "src", "a.jl"), src)
            return [f.hint for f in P.scan_file("src/a.jl"; root = dir)]
        end
        loop(body) = "function f(x, n)\n    for i in 1:n\n        $body\n    end\nend"
        @test only(hints(loop("g(zeros(n))"))) == P.LOOP_HINTS[:invariant]
        @test only(hints(loop("g(fill(i, n))"))) == P.LOOP_HINTS[:varying]
        @test only(hints(loop("x = vcat(x, i)"))) == P.LOOP_HINTS[:growth]
        # A growth chosen by an `if` is still a growth.
        @test only(hints(loop("x = if i > 1\n vcat(x, i)\n else\n x\n end"))) ==
              P.LOOP_HINTS[:growth]
    end

    #=
    The gate over a fixture tree: a Dismissal subtracts from the rule's count and not from `raw`, a
    rise over the recorded row fails `verify`, and an added file must enter at zero.
    =#
    mktempdir() do dir
        mkpath(joinpath(dir, "src"))
        write(joinpath(dir, "src", "a.jl"), "f(x) = sum(abs.(x))\ng(X) = h(X[:, 1])\n")
        files = ["src/a.jl"]
        m0 = P.measure(; root = dir, files, rulings = Dict{String, Any}())
        @test m0.counts["src/a.jl"]["reduce_temporary"] == 1
        @test m0.counts["src/a.jl"]["slice_copy"] == 1
        @test m0.counts["src/a.jl"]["raw"] == 2

        dismissal = Dict("file" => "src/a.jl", "rule" => "slice_copy", "definition" => "g",
                         "code" => "X[:, 1]", "rationale" => "fixture")
        m1 = P.measure(; root = dir, files,
                       rulings = Dict{String, Any}("perf_dismissal" => [dismissal]))
        @test m1.counts["src/a.jl"]["slice_copy"] == 0
        @test m1.counts["src/a.jl"]["raw"] == 2

        # A Dismissal without `code` covers every Finding of its rule in its definition, as a
        # complexity Exemption covers `(path, definition, metric)`.
        write(joinpath(dir, "src", "c.jl"),
              "function k(x, n)\n    for i in 1:n\n        g(zeros(n))\n        h(ones(n))\n    end\nend\n")
        whole = Dict("file" => "src/c.jl", "rule" => "loop_allocation", "definition" => "k",
                     "rationale" => "fixture")
        mc = P.measure(; root = dir, files = ["src/c.jl"],
                       rulings = Dict{String, Any}("perf_dismissal" => [whole]))
        @test mc.counts["src/c.jl"]["loop_allocation"] == 0
        @test mc.counts["src/c.jl"]["raw"] == 2
        rm(joinpath(dir, "src", "c.jl"))

        recorded = TOML.parse(P.render(m1, Dict{String, Any}(), false))
        failures, provenance_ok = P.verify(m1, recorded)
        @test provenance_ok
        @test isempty(failures)

        # Without the Dismissal the slice counts again, and the row recorded 0.
        failures, _ = P.verify(m0, recorded)
        @test any(f -> occursin("performance ratchet tripped on 1", f), failures)

        # An added file enters at zero reviewed Findings.
        write(joinpath(dir, "src", "b.jl"), "k(x) = maximum(abs.(x))\n")
        m2 = P.measure(; root = dir, files = ["src/a.jl", "src/b.jl"],
                       rulings = Dict{String, Any}("perf_dismissal" => [dismissal]))
        @test_throws P.CodeHealth.RefreshRefused P.render(m2, recorded, true)
    end
end
