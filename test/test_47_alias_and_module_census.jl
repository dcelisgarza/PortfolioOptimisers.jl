#=
`code_health/CodeHealth.jl` is a module rather than a script, and it loads only `TOML`, which
`test/Project.toml` already carries. It holds the one parser every census in this repository reads
the source text with: `walk_ast`, `isdocstring` and `documented_units`.
`test_45_sweep_census.jl` states what those measure, and
`test_49_coverage_attribution_census.jl` loads `code_health/coverage.jl` the same way.

The load sits OUTSIDE the `@testset` on purpose. `include` defines methods, and a method defined
inside one top-level statement is not visible to a call in that same statement. The module wrapper
keeps that module's own names out of the worker module.
=#
module AliasCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Alias census: every alias of src/25_Aliases.jl keeps its claim" begin
    using Test

    #=
    `src/25_Aliases.jl` is 132 units and 43 executable lines, and it is the only file under
    `src/` that holds an ACRONYM alias or a FACTORY alias. ADR 0086 (issue #436) scopes both
    kinds to this file, and `test_26_docs.jl` gates the sections each kind may carry.

    Sections are not the claim. An alias makes ONE claim and it is exactly checkable, which
    is what this census checks. Issue #442 is the ticket, under child map 1 (#415) of the
    sweep map of maps (#404), and this file pays that map's condition 2 for the alias file:
    the code agrees with the documentation, checked by running it.

      - An ACRONYM alias claims an IDENTITY. `HRP === HierarchicalRiskParity` must hold, and
        the docstring must name the same binding the `const` points at. 111 of these, and
        one loop asserts both halves at once. 111 separate tests would state the same thing
        111 times and would go stale one alias at a time.
      - A FACTORY alias claims an EQUALITY. `FLM(; kwargs...)` must equal
        `LowOrderMoment(; kwargs..., alg = FirstLowerMoment())` for every keyword it
        forwards. 21 of these, and the equalities are written out below because the long form
        IS the claim: deriving it from the body would compare the code with itself.

    The docstring half of the acronym check reads the SOURCE TEXT, not the runtime docstring.
    A `const` binds its target at load time, so a docstring that names the wrong type still
    resolves; only the text can be wrong, and only the text is read here.

    `ZeroVarianceFilter` is the 21st factory and the only one whose body raises. Its two
    prose claims -- the exclusive bound, so `tol = 0` still drops an exactly-constant asset,
    and the `DomainError` on a negative tolerance -- are pinned by
    `test_35_asset_selection.jl`, which owns the selector. They are not repeated here.
    =#

    PO = PortfolioOptimisers
    ROOT = normpath(joinpath(@__DIR__, ".."))
    ALIAS_FILE = joinpath(ROOT, "src", "25_Aliases.jl")

    # The two kinds this file holds. Both numbers are gated: `test_45_sweep_census.jl` holds
    # their sum at the manifest row's 132 units, and a new alias must land in one of them.
    ACRONYM_TOTAL = 111
    FACTORY_TOTAL = 21

    # ------------------------------------------------------------------ the parse

    # The same instrument `test_26_docs.jl` and `test_45_sweep_census.jl` read the file with:
    # one parse, no package needed, and a reading that cannot move under a reformat. It is
    # `CodeHealth`'s, so all three read one predicate.
    CH = AliasCensusHealth.CodeHealth
    isdocstring = CH.isdocstring

    # A docstring that interpolates parses to an `Expr(:string, ...)`; the literal pieces
    # carry every heading and every `@ref`.
    function docstring_text(x)
        for a in x.args[2:end]
            a isa AbstractString && return String(a)
            Meta.isexpr(a, :string) &&
                return join(p isa AbstractString ? p : " " for p in a.args)
        end
        return ""
    end

    # Every name a body CONSTRUCTS, which is what a factory alias's summary sentence must
    # name. `parentmodule` is the filter: `DomainError` is called by `ZeroVarianceFilter` and
    # is Base's, `zero` is not a constructor, and `@argcheck` is a macrocall and not a call.
    function constructed_names(node, acc = Set{Symbol}())
        node isa Expr || return acc
        if Meta.isexpr(node, :call) && !isempty(node.args)
            f = node.args[1]
            if f isa Symbol && isdefined(PO, f) && parentmodule(getfield(PO, f)) === PO
                push!(acc, f)
            end
        end
        for a in node.args
            constructed_names(a, acc)
        end
        return acc
    end

    # Each alias of the file: its name, its kind, its docstring text, and -- for a factory --
    # the names its body constructs. The body of the definition is read, never its signature,
    # so a keyword's default value is not mistaken for part of the composition.
    acronyms = Tuple{Symbol, Symbol, String}[]
    factories = Tuple{Symbol, String, Set{Symbol}}[]
    let
        # Only the top level: ADR 0086 scopes both kinds of alias to this file's top level.
        for node in CH.parse_file(ALIAS_FILE).args
            (isdocstring(node) && length(node.args) >= 4) || continue
            text, d = docstring_text(node), node.args[4]
            if Meta.isexpr(d, :const) && Meta.isexpr(d.args[1], :(=))
                lhs, rhs = d.args[1].args[1], d.args[1].args[2]
                (lhs isa Symbol && rhs isa Symbol) || continue
                push!(acronyms, (lhs, rhs, text))
            elseif Meta.isexpr(d, :function)
                sig = d.args[1]
                while Meta.isexpr(sig, :where) || Meta.isexpr(sig, :(::))
                    sig = sig.args[1]
                end
                Meta.isexpr(sig, :call) || continue
                push!(factories, (sig.args[1], text, constructed_names(d.args[2])))
            end
        end
    end

    @testset "the file holds the two kinds, and nothing else" begin
        @test length(acronyms) == ACRONYM_TOTAL
        @test length(factories) == FACTORY_TOTAL

        # A third kind would be a dispatch alias, which ADR 0086 permits in any file. It
        # carries `# Related`, and this file's sweep recorded none, so one arriving here
        # changes what `test_26_docs.jl` demands of the file.
        parsed = Set(vcat(first.(acronyms), first.(factories)))
        declared = Set{Symbol}()
        for m in eachmatch(r"^(?:const\s+|function\s+)([A-Za-z_][A-Za-z_0-9]*)"m,
                           read(ALIAS_FILE, String))
            push!(declared, Symbol(m.captures[1]))
        end
        @test setdiff(declared, parsed) == Set{Symbol}()
    end

    # ------------------------------------------------ 111 acronyms: one identity each

    # Every way one acronym alias can break its claim. It is a function and not a loop body
    # so that the loop below can be shown to be able to FAIL: a census whose per-item check
    # is unreachable passes on an empty list and states nothing.
    function acronym_offences(name, target, text)
        out = String[]
        isdefined(PO, name) || return push!(out, "$name  is not defined")
        isdefined(PO, target) ||
            return push!(out, "$name  names `$target`, which is not defined")
        getfield(PO, name) === getfield(PO, target) || push!(out, "$name  !== $target")
        Base.isexported(PO, name) || push!(out, "$name  is not exported")

        #=
        ADR 0086: the header line is the alias name alone, and the summary is exactly one
        sentence naming the target.

        THE INDENTATION IS PART OF THE CLAIM, so the lines are read raw. Julia strips a
        `"""` block's common indentation, which leaves the header inside its four-space
        signature block and the summary at column 0. `GAO` carried its sentence indented to
        four spaces, so the signature block swallowed it and the docstring rendered with no
        prose at all. Stripping each line first reads that defect as correct.
        =#
        lines = [rstrip(l) for l in split(text, '\n') if !isempty(strip(l))]
        length(lines) >= 2 || return push!(out, "$name  has no summary line")
        lines[1] == string("    ", name) ||
            push!(out, "$name  header line reads `$(lines[1])`")
        wanted = string("Alias for [`", target, "`](@ref).")
        lines[2] == wanted || push!(out, "$name  summary reads `$(lines[2])`")
        length(lines) == 2 ||
            push!(out, "$name  carries $(length(lines) - 2) line(s) past its summary")
        return out
    end

    @testset "an acronym alias is its target, and its docstring names its target" begin
        # Each breach is proved to be reported before the census below is trusted.
        good = "    HRP\n\nAlias for [`HierarchicalRiskParity`](@ref).\n"
        @test isempty(acronym_offences(:HRP, :HierarchicalRiskParity, good))
        # the `const` points elsewhere
        @test !isempty(acronym_offences(:HRP, :ValueatRisk, good))
        # the summary names another type
        @test !isempty(acronym_offences(:HRP, :HierarchicalRiskParity,
                                        "    HRP\n\nAlias for [`RiskBudgeting`](@ref).\n"))
        # the `GAO` shape: the sentence is indented into the signature block
        @test !isempty(acronym_offences(:HRP, :HierarchicalRiskParity,
                                        "    HRP\n\n    Alias for [`HierarchicalRiskParity`](@ref).\n"))
        # a section, which no acronym alias carries
        @test !isempty(acronym_offences(:HRP, :HierarchicalRiskParity,
                                        string(good, "\n# Related\n\n  - x\n")))

        offenders = String[]
        for (name, target, text) in acronyms
            append!(offenders, acronym_offences(name, target, text))
        end
        if !isempty(offenders)
            @warn """$(length(offenders)) acronym alias claim(s) in `src/25_Aliases.jl` do not
                     hold. An acronym alias and its target are the SAME object, and its
                     docstring is one sentence naming that object (ADR 0086). Correct the
                     `const`, or correct the sentence:\n  $(join(offenders, "\n  "))"""
        end
        @test isempty(offenders)
    end

    # ------------------------------------ 21 factories: the summary names what it builds

    function factory_offences(name, text, built)
        out = String[]
        for b in sort(collect(built))
            b === name && continue
            occursin(string("[`", b, "`](@ref)"), text) ||
                push!(out, "$name  builds `$b` and does not @ref it")
        end
        Base.isexported(PO, name) || push!(out, "$name  is not exported")
        return out
    end

    @testset "a factory alias's summary names every type it composes" begin
        @test isempty(factory_offences(:FLM, "Alias for [`LowOrderMoment`](@ref).",
                                       Set([:LowOrderMoment])))
        @test !isempty(factory_offences(:FLM, "Alias for [`LowOrderMoment`](@ref).",
                                        Set([:LowOrderMoment, :FirstLowerMoment])))

        offenders = String[]
        for (name, text, built) in factories
            append!(offenders, factory_offences(name, text, built))
        end
        if !isempty(offenders)
            @warn """$(length(offenders)) factory alias summary sentence(s) in
                     `src/25_Aliases.jl` omit a type the factory composes. The composition IS
                     the claim, so the sentence `@ref`s every type it builds (ADR
                     0086):\n  $(join(offenders, "\n  "))"""
        end
        @test isempty(offenders)
    end

    # --------------------------------------------- 21 factories: one equality each

    #=
    Two estimators built by two routes are two objects, and the library defines no `==` for
    them, so `==` is `===` and every equality below would fail. The comparison is structural:
    same type, and every field equal by the same rule. An empty field set bottoms out on
    `==`, which is what compares a number, a `Symbol` and an array; `===` catches `nothing`
    and a function such as `owa_gmd` before either.
    =#
    function alias_equal(a, b)
        a === b && return true
        typeof(a) === typeof(b) || return false
        if a isa AbstractArray
            axes(a) == axes(b) || return false
            return all(alias_equal(x, y) for (x, y) in zip(a, b))
        end
        fns = fieldnames(typeof(a))
        isempty(fns) && return a == b
        return all(alias_equal(getfield(a, f), getfield(b, f)) for f in fns)
    end

    @testset "a factory alias equals the long form it stands for" begin
        # Every keyword is given a NON-DEFAULT value, so a forward that drops a keyword or
        # sends it to the wrong slot separates the two sides. A run with the defaults alone
        # would pass whatever the body forwarded.
        st = RiskMeasureSettings(; scale = 2.5, ub = 0.3, rke = false)
        ow = pweights([0.25, 0.25, 0.5])
        mu = [0.01, 0.02, 0.03]
        ve = SimpleVariance(; me = nothing, corrected = false)
        sm = SOCRiskExpr()
        ex = ExactOrderedWeightsArray()

        # Moment risk measures: `LowOrderMoment` and `HighOrderMoment` with an `alg`.
        @test alias_equal(FLM(; settings = st, w = ow, mu = mu),
                          LowOrderMoment(; settings = st, w = ow, mu = mu,
                                         alg = FirstLowerMoment()))
        @test alias_equal(MAD(; settings = st, w = ow, mu = mu),
                          LowOrderMoment(; settings = st, w = ow, mu = mu,
                                         alg = MeanAbsoluteDeviation()))
        @test alias_equal(SCM(; settings = st, w = ow, mu = mu, ve = ve, alg = sm),
                          LowOrderMoment(; settings = st, w = ow, mu = mu,
                                         alg = SecondMoment(; ve = ve, alg1 = FullMoment(),
                                                            alg2 = sm)))
        @test alias_equal(SLM(; settings = st, w = ow, mu = mu, ve = ve, alg = sm),
                          LowOrderMoment(; settings = st, w = ow, mu = mu,
                                         alg = SecondMoment(; ve = ve, alg1 = SemiMoment(),
                                                            alg2 = sm)))
        @test alias_equal(ECM(; settings = st, w = ow, mu = mu, p = 3, ddof = 1),
                          LowOrderMoment(; settings = st, w = ow, mu = mu,
                                         alg = EvenMoment(; p = 3, ddof = 1,
                                                          alg = FullMoment())))
        @test alias_equal(ELM(; settings = st, w = ow, mu = mu, p = 3, ddof = 1),
                          LowOrderMoment(; settings = st, w = ow, mu = mu,
                                         alg = EvenMoment(; p = 3, ddof = 1,
                                                          alg = SemiMoment())))
        @test alias_equal(TLM(; settings = st, w = ow, mu = mu),
                          HighOrderMoment(; settings = st, w = ow, mu = mu,
                                          alg = ThirdLowerMoment()))
        @test alias_equal(SSK(; settings = st, w = ow, mu = mu, ve = ve),
                          HighOrderMoment(; settings = st, w = ow, mu = mu,
                                          alg = StandardisedHighOrderMoment(; ve = ve,
                                                                            alg = ThirdLowerMoment())))
        @test alias_equal(FTCM(; settings = st, w = ow, mu = mu),
                          HighOrderMoment(; settings = st, w = ow, mu = mu,
                                          alg = FourthMoment(; alg = FullMoment())))
        @test alias_equal(FTLM(; settings = st, w = ow, mu = mu),
                          HighOrderMoment(; settings = st, w = ow, mu = mu,
                                          alg = FourthMoment(; alg = SemiMoment())))
        @test alias_equal(KT(; settings = st, w = ow, mu = mu, ve = ve),
                          HighOrderMoment(; settings = st, w = ow, mu = mu,
                                          alg = StandardisedHighOrderMoment(; ve = ve,
                                                                            alg = FourthMoment(;
                                                                                               alg = FullMoment()))))
        @test alias_equal(SKT(; settings = st, w = ow, mu = mu, ve = ve),
                          HighOrderMoment(; settings = st, w = ow, mu = mu,
                                          alg = StandardisedHighOrderMoment(; ve = ve,
                                                                            alg = FourthMoment(;
                                                                                               alg = SemiMoment()))))

        # The eight OWA aliases. Three take a weight FUNCTION and five a weight ESTIMATOR,
        # and both reach the same `w` slot of `OrderedWeightsArray`.
        @test alias_equal(OWA_GMD(; settings = st, alg = ex),
                          OrderedWeightsArray(; settings = st, w = PO.owa_gmd, alg = ex))
        @test alias_equal(OWA_WR(; settings = st, alg = ex),
                          OrderedWeightsArray(; settings = st, w = PO.owa_wr, alg = ex))
        @test alias_equal(OWA_RG(; settings = st, alg = ex),
                          OrderedWeightsArray(; settings = st, w = PO.owa_rg, alg = ex))
        @test alias_equal(OWA_CVaR(; settings = st, alpha = 0.1, alg = ex),
                          OrderedWeightsArray(; settings = st,
                                              w = OrderedWeightsArrayConditionalValueatRisk(;
                                                                                            alpha = 0.1),
                                              alg = ex))
        @test alias_equal(OWA_CVaR_RG(; settings = st, alpha = 0.1, beta = 0.2, alg = ex),
                          OrderedWeightsArray(; settings = st,
                                              w = OrderedWeightsArrayConditionalValueatRiskRange(;
                                                                                                 alpha = 0.1,
                                                                                                 beta = 0.2),
                                              alg = ex))
        @test alias_equal(OWA_TG(; settings = st, alpha_i = 1e-3, alpha = 0.1, a_sim = 50,
                                 alg = ex),
                          OrderedWeightsArray(; settings = st,
                                              w = OrderedWeightsArrayTailGini(;
                                                                              alpha_i = 1e-3,
                                                                              alpha = 0.1,
                                                                              a_sim = 50),
                                              alg = ex))
        @test alias_equal(OWA_TG_RG(; settings = st, alpha_i = 1e-3, alpha = 0.1,
                                    a_sim = 50, beta_i = 2e-3, beta = 0.2, b_sim = 40,
                                    alg = ex),
                          OrderedWeightsArray(; settings = st,
                                              w = OrderedWeightsArrayTailGiniRange(;
                                                                                   alpha_i = 1e-3,
                                                                                   alpha = 0.1,
                                                                                   a_sim = 50,
                                                                                   beta_i = 2e-3,
                                                                                   beta = 0.2,
                                                                                   b_sim = 40),
                                              alg = ex))
        @test alias_equal(OWA_LMoment(; settings = st,
                                      method = NormalisedConstantRelativeRiskAversion(;
                                                                                      g = 0.4),
                                      k = 4, alg = ex),
                          OrderedWeightsArray(; settings = st,
                                              w = LinearMoment(;
                                                               method = NormalisedConstantRelativeRiskAversion(;
                                                                                                               g = 0.4),
                                                               k = 4), alg = ex))

        # The 21st. Its `score` is `SCM()` and not `Variance`, which its docstring explains
        # and `test_35_asset_selection.jl` proves by raising on `Variance`.
        @test alias_equal(ZeroVarianceFilter(; tol = 1e-8),
                          ScoreSelector(; score = SCM(), rule = ThresholdRule(; lo = 1e-8)))

        #=
        The two RANGE keywords default to their lower twin, and a default that is written as
        the wrong twin's name reads correctly and forwards the wrong value. Passing one side
        alone is what separates them.
        =#
        @test alias_equal(OWA_CVaR_RG(; alpha = 0.1),
                          OWA_CVaR_RG(; alpha = 0.1, beta = 0.1))
        @test alias_equal(OWA_TG_RG(; alpha_i = 1e-3, alpha = 0.1, a_sim = 50),
                          OWA_TG_RG(; alpha_i = 1e-3, alpha = 0.1, a_sim = 50,
                                    beta_i = 1e-3, beta = 0.1, b_sim = 50))

        # `alias_equal` must be able to FAIL, or every assertion above is vacuous.
        @test !alias_equal(FLM(), MAD())
        @test !alias_equal(FLM(; settings = st), FLM())
        @test !alias_equal(ECM(; p = 3), ECM(; p = 2))
        @test !alias_equal(OWA_GMD(), OWA_WR())
    end
end

@testset "Module census: src/PortfolioOptimisers.jl reaches every source file" begin
    using Test

    CH = AliasCensusHealth.CodeHealth

    #=
    `src/PortfolioOptimisers.jl` is the module file, and it is the second file issue #442
    sweeps. It carries ONE documented unit, the module docstring, and ZERO executable lines,
    so it is coverage terminal by construction and its `# Algorithm` count is zero.

    Its claim is its include list. 193 `include` calls put every other file under `src/` into
    the module, in the order the declarations need, and NOTHING checked that list. A file
    added to `src/` and not included is dead: it carries a manifest row, so
    `test_45_sweep_census.jl` stays green; its types are never declared, so no other test can
    name them; and the omission surfaces only when a caller reaches for something that was
    never there. The library holds no other list of its own source files, so this check reads
    the directory.

    The list is read from the PARSE and not from a regex, so it cannot move under a reformat.
    =#

    ROOT = normpath(joinpath(@__DIR__, ".."))
    SRC = joinpath(ROOT, "src")
    MODULE_FILE = joinpath(SRC, "PortfolioOptimisers.jl")

    # Every path an `include` call names, in source order. The commented-out `walkdir` block
    # at the head of the file is inside a `#= =#`, so the parse never sees it.
    included = String[]
    CH.walk_ast(CH.parse_file(MODULE_FILE)) do x
        if Meta.isexpr(x, :call) &&
           length(x.args) == 2 &&
           x.args[1] === :include &&
           x.args[2] isa AbstractString
            push!(included, String(x.args[2]))
        end
        return nothing
    end

    # Every `.jl` file under `src/`, module file excluded, as a path relative to `src/`.
    on_disk = String[]
    for (root, _, files) in walkdir(SRC)
        for f in files
            endswith(f, ".jl") || continue
            p = relpath(joinpath(root, f), SRC)
            p == "PortfolioOptimisers.jl" && continue
            push!(on_disk, replace(p, '\\' => '/'))
        end
    end

    @testset "every source file is included, exactly once" begin
        missing_from_module = sort(setdiff(on_disk, included))
        if !isempty(missing_from_module)
            @warn """$(length(missing_from_module)) file(s) under `src/` are not `include`d by
                     `src/PortfolioOptimisers.jl`. Such a file is dead: its declarations never
                     run, and its `sweep/manifest.toml` row keeps
                     `test_45_sweep_census.jl` green. Add an `include` in declaration
                     order:\n  $(join(missing_from_module, "\n  "))"""
        end
        @test isempty(missing_from_module)

        # The other direction: an `include` that names no file raises at precompilation, so
        # this can only red on a source tree that does not load. It is here because the two
        # directions are one statement, and a rename that moves a file pays both at once.
        absent = sort(setdiff(included, on_disk))
        if !isempty(absent)
            @warn "`include` call(s) naming no file under `src/`: $(join(absent, ", "))"
        end
        @test isempty(absent)

        # A file included twice re-runs its declarations. The second run redefines every
        # method it holds, and the reader of the list sees no error.
        duplicated = sort([p for p in unique(included) if count(==(p), included) > 1])
        if !isempty(duplicated)
            @warn "File(s) `include`d more than once: $(join(duplicated, ", "))"
        end
        @test isempty(duplicated)
    end

    @testset "the module docstring is the README" begin
        # The docstring interpolates `DocStringExtensions.README`, so the file's own text
        # carries no prose and only the rendered string can say what a reader sees.
        rendered = string(Base.Docs.doc(Base.Docs.Binding(Main, :PortfolioOptimisers)))
        @test occursin("PortfolioOptimisers", rendered)
        @test length(rendered) > 70
    end
end
