@testset "Seam-lock: Model State is reached only through its interface" begin
    using Test

    # Model State (the `JuMP.Model` object dictionary shared by every constraint and risk
    # builder) is reached through the typed interface in `08_Base_JuMPOptimisation.jl`:
    # `state_key` / `state_set!` / `state_has` / `state_get` / `state_build!` /
    # `nested_prefix`, plus the named accessors built on them (`get_X`, `get_dd`, …).
    #
    # Three rules hold the seam shut (ADR 0037, amending ADR 0004 §2 and §6.5):
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
    #   3. INDEXED CONSTRUCTION. A Model State key is disambiguated on TWO axes — the
    #      per-build `prefix` and the per-measure index `i` — and rules 1 and 2 policed
    #      only the first. The index suffix carries the same do-not-collide invariant and
    #      was spelled by hand at ~200 sites (`model[Symbol(:cvar_risk_, i)]`), matching
    #      neither rule. Both axes now resolve in `state_key`, so no emitter composes a key
    #      inside a Model State access at all. Like the other two, this rule names no keys.
    #
    #      A composed key must not escape into a local either, so the rule polices the
    #      access site: `model[<anything but a plain identifier>]`. The identifiers that
    #      remain are exempt for a stated reason, and the exemptions name files and
    #      functions — never keys (see `registry_readers` and `mip_key` below).
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

    # No file is exempt from rule 3. The frontier sweep used to be: the registry stores
    # `(bound_var_key, bound_key)` pairs built by `set_risk_upper_bound!` and
    # `set_return_bounds!`, and the sweep loops in `11_MeanRisk.jl` and
    # `13_NearOptimalCentering.jl` created the parameter and constraint under exactly those
    # keys, so both files were exempted by name. The exemption was the tell that the sweep
    # had no interface (ADR 0062): `set_ret_frontier_parameters!`,
    # `set_risk_frontier_parameters!` and `set_frontier_point!` are now the sweep's own seam
    # and live in the interface, so the two optimisers reach for no key at all.
    registry_readers = String[]

    construction = Regex("Symbol\\(\\s*prefix\\b")
    bare = Regex("(?:model\\[|haskey\\(\\s*model\\s*,\\s*):([A-Za-z_][A-Za-z_0-9]*)")
    # Rule 3 polices the ACCESS, not the spelling of the key: outside the interface,
    # `model[…]` and `haskey(model, …)` are simply not reached for. That subsumes a key
    # composed inline AND one laundered through a local first, which a
    # construction-shaped rule would miss.
    #
    # `mip_key(sp, :ib)` is the MIP space's own single-home resolver (ADR 0034) — a third
    # namespace that already has one home, not a convention re-spelled per site — so it is
    # named here rather than policed. Naming it is a claim: if a second such resolver
    # appears, it fails here until someone decides whether it belongs in the interface.
    indexed = Regex("(?:model\\[|haskey\\(\\s*model\\s*,\\s*)\\s*(?!mip_key\\()")

    build_violations = String[]
    read_violations = String[]
    index_violations = String[]
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
            if !(file in registry_readers) && occursin(indexed, raw)
                push!(index_violations, string(rel, ":", lineno, "  ", strip(raw)))
            end
        end
    end

    # Self-checks: every matcher fires on the thing it is meant to catch, and stays quiet
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
    # Rule 3 catches the index suffix in every spelling it had — inline, laundered through
    # a local, and probed — and stays quiet on the interface calls that replaced them.
    @test occursin(indexed, "model[Symbol(:cvar_risk_, i)] = v")
    @test occursin(indexed, "key = Symbol(:cvar_risk_, i); model[key] = v")
    @test occursin(indexed, "if haskey(model, Symbol(:cvar_risk_, i))")
    @test occursin(indexed, "x = model[keys[1]]")
    @test !occursin(indexed, "state_set!(model, prefix, :cvar_risk_, i, v)")
    @test !occursin(indexed, "x = state_get(model, prefix, :cvar_risk_, i)")
    @test !occursin(indexed, "if state_has(model, prefix, :cvar_risk_, i)")
    @test !occursin(indexed, "k = state_key(prefix, :cvar_risk_, i)")
    @test !occursin(indexed, "model[mip_key(sp, :ib)] = v")

    # SHARED_STATE is the runtime half of rule 2; the lock is only as good as that set
    # actually being enforced, so check the guard rejects a per-build entry.
    @test_throws ArgumentError PortfolioOptimisers.assert_shared_state(:W)
    @test PortfolioOptimisers.assert_shared_state(:risk_vec) === nothing

    # The three rules police the SPELLING of an access and the NAME of a bare entry. None
    # of them can see that two well-spelled keys COMPOSE the same Symbol: `state_key`
    # concatenates on both axes, so a name that ends in a digit at a low index collides
    # with a shorter name at a higher index. `:te_dr_11` is how that defect presented (ADR
    # 0037 amendment §4), and a collision is a WRONG ANSWER — the second write replaced the
    # first, the build carried one entry where it needed two, and a constraint bound the
    # wrong variable. `state_set!` is the runtime rule that closes the class: registration
    # is fresh, so a collision throws where it used to overwrite.
    let m = PortfolioOptimisers.JuMP.Model(), p = Symbol("")
        @test PortfolioOptimisers.state_key(p, :te_dr_, 11) ==
              PortfolioOptimisers.state_key(p, :te_dr_1, 1)
        @test PortfolioOptimisers.state_set!(m, p, :te_dr_, 11, 1) == 1
        @test_throws ArgumentError PortfolioOptimisers.state_set!(m, p, :te_dr_1, 1, 2)
        @test m[:te_dr_11] == 1
        # An entry is refused under its own key too, so a second emitter cannot replace an
        # entry the first one owns.
        @test_throws ArgumentError PortfolioOptimisers.state_set!(m, p, :te_dr_, 11, 2)
        @test_throws ArgumentError PortfolioOptimisers.state_set!(m, p, :te_dr_11, 3)
        # A free key still registers, on either arity.
        @test PortfolioOptimisers.state_set!(m, p, :cvar_risk_, 1, 3) == 3
        @test PortfolioOptimisers.state_set!(m, p, :net_X, 4) == 4
        # Reuse is the other two verbs' job, and they stay idempotent.
        @test PortfolioOptimisers.state_build!(() -> 5, m, p, :te_dr_, 11) == 1
        @test PortfolioOptimisers.state_build!(() -> 5, m, p, :net_X) == 4
        @test PortfolioOptimisers.mark_state!(m, p, :variance_flag) === nothing
        @test PortfolioOptimisers.mark_state!(m, p, :variance_flag) === nothing
        @test PortfolioOptimisers.mark_state!(m, p, :variance_flag, 1) === nothing
        @test PortfolioOptimisers.mark_state!(m, p, :variance_flag, 1) === nothing
    end

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
    if !isempty(index_violations)
        println("Seam-lock (indexed construction): these reach Model State directly. A ",
                "per-measure entry goes through state_set!/state_get/state_has/",
                "state_build! with its bare name and its index — state_key resolves both ",
                "the prefix and the index, so neither is spelled at the call site:")
        foreach(v -> println("  ", v), index_violations)
    end
    @test isempty(index_violations)
    @test isempty(build_violations)
    @test isempty(read_violations)
end
