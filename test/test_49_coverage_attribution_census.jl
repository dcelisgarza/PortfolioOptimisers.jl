#=
`code_health/coverage.jl` is a script rather than a package. The `coverage` job of
`ReusableTest.yml` runs it as
`julia --project=code_health code_health/coverage.jl check`, and before this file nothing
under `test/` loaded it at all. Its command block is guarded by
`abspath(PROGRAM_FILE) == @__FILE__`, so it loads cleanly as a library, and it needs only
`TOML`, which `test/Project.toml` already carries.

The load sits OUTSIDE the `@testset` on purpose. `include` defines methods, and a method
defined inside one top-level statement is not visible to a call in that same statement. The
module wrapper keeps the script's own names -- `measure`, `verify`, `render`, `publish` --
out of the worker module, where `using PortfolioOptimisers` has already bound a large
surface.
=#
module CoverageAttribution
include(joinpath(@__DIR__, "..", "code_health", "coverage.jl"))
end

@testset "Coverage attribution census: the name is the function, not the return type" begin
    using Test, TOML

    #=
    A Coverage Exemption is keyed `(path, definition)`, and ADR 0082 states the reason for
    that key: "a human writes and reads these rows and a human names the method". So the
    name `coverage.jl` attributes a miss line to is not a detail of the instrument. It is
    the key a contributor types into `code_health/rulings.toml`, and a wrong name is a row
    that stands for the wrong lines.

    Issue #521 found two ways a RETURN TYPE broke that key, and both are gated here.

      - `function f(x)::Nothing` was named `Nothing`. `defname`'s `::` branch took the
        right-hand side of the annotation. `06_AssetSetsMatrix.jl` alone held seven
        definitions named `Nothing`, and the key sums a name over the file, so a row
        written for one method silently stood for seven.
      - `has_pretty_show_method(::Any)::Bool = false` was named `<toplevel>`.
        `definition_name` read the short form as an assignment, because it accepted only a
        `:call` or a `:where` on the left of the `=`, and a return type wraps both in a
        `::`.

    The rule the three shapes obey has ONE implementation, `CodeHealth.definition_name`.
    `code_health/coverage.jl` reads it to write a Coverage Exemption key, and
    `test/test_26_docs.jl` reads it to key a docstring by the name it binds. A second copy
    stood in `test_26_docs.jl` until it was removed: it took the LEFT side of every `::`, so
    it named a functor method after its receiver variable and collapsed 11 documented
    functors of `src/14_UncertaintySets/06_CalibrationRules.jl` onto the key `alg`. This
    file is the differential oracle over the one that remains.

    The fix is `is_signature`, which separates a signature from an assignment. The
    separation is needed because a THIRD shape shares the `::` head and is named from the
    other side: the receiver slot of a functor method, `function (r::MaximumDrawdown)(x)`,
    is named `MaximumDrawdown`, which is what a reader calls that method. This library holds
    many of them, so a fix that simply took the left side everywhere would have broken every
    one.

    --------------------------------------------------------- how the checks are drawn

    CHECK 1 states the three shapes against a literal source string, so the expected name is
    written beside the code that produces it and no corpus is needed to read the test.

    CHECK 2 runs the same rule over the real corpus. It finds every return-annotated
    top-level definition under `src/` and `ext/` by parsing, works the name out of the
    signature itself, and asks `definition_name` for the same answer. It is the check that
    would have gone red before the fix.

    CHECK 3 closes the loop on the file the whole mechanism exists for. Every Coverage
    Exemption in `code_health/rulings.toml` must name a definition its file actually holds.
    `coverage.jl check` holds a row's count to equality and so implies the name, but it
    needs an `lcov.info` and therefore runs only in CI. This check needs neither, and it is
    what catches a row whose key went stale under a rename.

    Checks 2 and 3 both guard their own measurement. A parse that returned nothing, or a
    rulings file that lost its rows, would leave them vacuously green.
    =#

    CA = CoverageAttribution
    ROOT = normpath(joinpath(@__DIR__, ".."))

    # -------------------------------------------------------------- check 1: the shapes

    @testset "The three shapes a `::` node arrives in" begin
        src = """
              function long_form(x)::Nothing
                  return nothing
              end
              short_form(x)::Bool = true
              function parametric(x::T)::Vector{Int} where {T}
                  return Int[]
              end
              function (r::NamedReceiver)(x)
                  return x
              end
              function (::AnonymousReceiver)(x)
                  return x
              end
              plain(x) = x
              typed_global::Int = 1
              plain_global = 2
              """
        got = [CA.definition_name(a) for a in Meta.parseall(src).args if a isa Expr]
        @test got == ["long_form", "short_form", "parametric", "NamedReceiver",
                      "AnonymousReceiver", "plain", "", ""]
    end

    # --------------------------------------------- check 2: every definition in the corpus

    #=
    The name a return-annotated definition MUST take, or `""` when the expression is not
    one.

    `unwrap` drops the docstring wrapper, and the `:where` peel handles
    `function f(x)::T where {S}`, whose annotation sits under the `:where` rather than over
    it.
    =#
    function annotated_name(a)
        e = CA.unwrap(a)
        if !(e isa Expr) || !(e.head === :function || e.head === :(=))
            return ""
        end
        sig = e.args[1]
        while sig isa Expr && sig.head === :where
            sig = sig.args[1]
        end
        if !(sig isa Expr && sig.head === :(::) && CA.is_signature(sig.args[1]))
            return ""
        end
        return CA.defname(sig.args[1])
    end

    @testset "Every return-annotated definition under `src/` and `ext/`" begin
        files = filter(CA.CodeHealth.in_scope, CA.CodeHealth.tracked_jl_files())
        @test !isempty(files)

        annotated = 0
        wrong = String[]
        for f in files
            top = CA.CodeHealth.parse_file(f)
            top isa Expr || continue
            for a in top.args
                a isa Expr || continue
                want = annotated_name(a)
                isempty(want) && continue
                annotated += 1
                got = CA.definition_name(a)
                if got != want
                    push!(wrong,
                          string(f, ": expected ", want, ", got ",
                                 isempty(got) ? CA.TOPLEVEL : got))
                end
            end
        end

        # The bound guards against a vacuous pass, and is deliberately far below the count
        # the corpus holds. It is not a measurement of this library and must never be
        # tightened into one: a count restated in a test goes stale the next time a file is
        # edited.
        @test annotated >= 20
        @test isempty(wrong)
        isempty(wrong) || println(join(wrong, "\n"))
    end

    # -------------------------------------- check 3: every Coverage Exemption key resolves

    @testset "Every Coverage Exemption names a definition its file holds" begin
        rulings = TOML.parsefile(joinpath(ROOT, "code_health", "rulings.toml"))
        rows = get(rulings, "coverage_exemption", Dict{String, Any}[])
        @test !isempty(rows)

        unresolved = String[]
        for r in rows
            path, definition = r["path"], r["definition"]
            definition == CA.TOPLEVEL && continue
            held = Set(n for (n, _, _) in CA.definition_ranges(path))
            definition in held ||
                push!(unresolved, string(path, ": no definition named ", definition))
        end
        @test isempty(unresolved)
        isempty(unresolved) || println(join(unresolved, "\n"))
    end

    # ------------------------------------- check 4: there is only one exemption mechanism

    @testset "No source file carries a `COV_EXCL` marker" begin
        #=
        `CoverageTools` reads a `# COV_EXCL_START` / `# COV_EXCL_STOP` pair when the CI test
        job converts the `.cov` files to `lcov.info`, and it DROPS every line between the
        two from `lcov.info` altogether. Such a line is neither a hit nor a miss, so the
        gate of ADR 0082 never sees it, and a file reports `misses = 0` while a third of it
        is untested.

        That is a second exemption mechanism, and ADR 0082 admits one. A Coverage Exemption
        is a named row in `code_health/rulings.toml`, keyed `(path, definition)`, carrying
        an exact count and a Rationale, and held to equality in both directions. A
        `COV_EXCL` pair carries no count, no rationale and no key, and before this check
        nothing gated it.

        Issue #552 removed the two pairs the library held and covered the code they hid.
        This check is what stops a third from arriving in silence. The marker only works as
        a comment, so a plain text search over the tracked files is the whole test.
        =#
        files = filter(CA.CodeHealth.in_scope, CA.CodeHealth.tracked_jl_files())
        @test !isempty(files)

        marked = String[]
        for f in files
            for (n, line) in enumerate(eachline(joinpath(ROOT, f)))
                occursin("COV_EXCL", line) && push!(marked, string(f, ":", n, ": ", line))
            end
        end
        @test isempty(marked)
        isempty(marked) || println(join(marked, "\n"))
    end
end
