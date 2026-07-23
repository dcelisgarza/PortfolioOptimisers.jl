@testset "Seam-lock: Model State is reached only through its interface" begin
    using Test

    # Model State (the `JuMP.Model` object dictionary shared by every constraint and risk
    # builder) is reached through the typed interface in `08_Base_JuMPOptimisation.jl`:
    # `state_key` / `state_set!` / `state_has` / `state_get` / `state_build!` /
    # `nested_prefix`, plus the named accessors built on them (`get_X`, `get_dd`, …).
    #
    # Two rules hold the seam shut (ADR 0037, amending ADR 0004 §2 and §6.5):
    #
    #   1. CONSTRUCTION. No file outside the interface may build a prefixed key by hand.
    #      This rule is *closed*: it names no keys, so an entry added in future is covered
    #      the day it is written. It replaces the old 40-symbol managed-key allowlist,
    #      which had to be kept in sync and admitted its own hole — a sloppy
    #      `Symbol(prefix, :newCatA)` looked like Category-B scratch and slipped through.
    #
    #   2. BARE ACCESS. Outside the interface, Model State is never touched by literal key
    #      at all — no `model[:key]`, no `haskey(model, :key)`. Deliberately-shared entries
    #      go through `shared_get`/`shared_has`/`shared_set!`, which validate the name
    #      against `SHARED_STATE` **at run time**. So this rule, too, names no keys: the
    #      classification lives with the code it describes, and reaching for a per-build
    #      entry without a prefix throws rather than merely failing CI.
    #
    #      The polarity is what matters. The old lint enumerated the *managed* keys, so
    #      forgetting to list a newly-managed key silently opened a hole. `SHARED_STATE`
    #      enumerates the deliberately-*shared* entries, so forgetting to classify a new key
    #      FAILS instead. That is what closes ADR 0004 §2's residual hole.
    #
    # A bare read of a prefix-managed entry is the regression class that broke
    # IndependentVariableTracking (the OWA / PowerNorm / Turnover / DR-CDaR gaps fixed in
    # Phase 3) and made `set_iplg_constraints!` throw `KeyError(:ib)` on a short +
    # threshold/fee/xbgt model (ADR 0033/0034).

    srcdir = normpath(joinpath(@__DIR__, "..", "src"))
    interface = "08_Base_JuMPOptimisation.jl"

    # Files that build a DIFFERENT `JuMP.Model` — not the portfolio model, so the Model
    # State vocabulary does not apply to them at all.
    other_models = ["19_RiskMeasures/10_OWARiskMeasures.jl",          # OWA weight fitting
                    "20_Optimisation/22_DiscreteFiniteAllocation.jl"]  # allocation MIP

    # Strip `#` line comments, `\"\"\"` docstrings and `#=` block comments so prose mentions
    # of a key (e.g. "stored as model[:W]") and dead legacy code do not trip the lock.
    # NOTE: the pre-ADR-0037 lock skipped only `#` and `\"\"\"`; `#=` blocks (such as the
    # legacy `port`-API code in 09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl)
    # were not excluded.
    function code_lines(path)
        out = Tuple{Int, String}[]
        indoc = false
        blockdepth = 0
        for (lineno, raw) in enumerate(eachline(path))
            stripped = lstrip(raw)
            opens = count("#=", raw)
            closes = count("=#", raw)
            wasblock = blockdepth > 0
            blockdepth = max(0, blockdepth + opens - closes)
            inprose = indoc ||
                      wasblock ||
                      blockdepth > 0 ||
                      startswith(stripped, "#") ||
                      startswith(stripped, "\"\"\"")
            if !inprose
                push!(out, (lineno, raw))
            end
            if isodd(count("\"\"\"", raw))
                indoc = !indoc
            end
        end
        return out
    end

    construction = Regex("Symbol\\(\\s*prefix\\b")
    bare = Regex("(?:model\\[|haskey\\(\\s*model\\s*,\\s*):([A-Za-z_][A-Za-z_0-9]*)")

    build_violations = String[]
    read_violations = String[]
    for (root, _, files) in walkdir(srcdir), file in files
        if !endswith(file, ".jl") || file == interface
            continue
        end
        path = joinpath(root, file)
        rel = replace(relpath(path, srcdir), '\\' => '/')
        if rel in other_models
            continue
        end
        for (lineno, raw) in code_lines(path)
            if occursin(construction, raw)
                push!(build_violations, string(rel, ":", lineno, "  ", strip(raw)))
            end
            for m in eachmatch(bare, raw)
                push!(read_violations,
                      string(rel, ":", lineno, "  [", m.captures[1], "]  ", strip(raw)))
            end
        end
    end

    # Self-checks: both matchers fire on the thing they are meant to catch, and stay quiet
    # on the sanctioned spellings.
    @test occursin(construction, "x = model[Symbol(prefix, :net_X)]")
    @test occursin(construction, "tp = Symbol(prefix, :tr_iv_, i, :_)")
    @test !occursin(construction, "x = state_get(model, prefix, :net_X)")
    @test !occursin(construction, "tp = nested_prefix(prefix, :tr_iv_, i)")
    let hit(s) = [Symbol(m.captures[1]) for m in eachmatch(bare, s)]
        @test hit("x = model[:net_X]") == [:net_X]
        @test hit("if haskey(model, :W)") == [:W]
        @test hit("model[:risk_vec] = v") == [:risk_vec]
        @test isempty(hit("x = state_get(model, prefix, :net_X)"))
        @test isempty(hit("x = shared_get(model, :risk_vec)"))
        @test isempty(hit("if shared_has(model, :fees)"))
    end

    # SHARED_STATE is the runtime half of rule 2; the lock is only as good as that set
    # actually being enforced, so check the guard rejects a per-build entry.
    @test_throws ArgumentError PortfolioOptimisers.assert_shared_state(:W)
    @test PortfolioOptimisers.assert_shared_state(:risk_vec) === nothing

    if !isempty(build_violations)
        println("Seam-lock (construction): build prefixed keys via the Model State ",
                "interface — state_get/state_set!/state_build!/nested_prefix:")
        foreach(v -> println("  ", v), build_violations)
    end
    if !isempty(read_violations)
        println("Seam-lock (bare access): these touch Model State by literal key. Use a ",
                "named accessor, a prefixed state_* call, or — if the entry really is ",
                "shared across nested builds — shared_get/shared_has/shared_set! after ",
                "adding it to SHARED_STATE with the reason:")
        foreach(v -> println("  ", v), read_violations)
    end
    @test isempty(build_violations)
    @test isempty(read_violations)
end
