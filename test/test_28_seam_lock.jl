@testset "Seam-lock: prefix-managed keys never read bare" begin
    using Test

    # The Step 4 migration (ADR 0005) namespaces every per-build Category-A model-state
    # key under a `prefix`. Outside the typed state interface in
    # `08_Base_JuMPOptimisation.jl`, these keys MUST be reached via a prefixed accessor or
    # a computed `model[Symbol(prefix, name)]` — NEVER a literal `model[:key]`. A literal
    # read reintroduces the bare-read-under-tracking regression class that broke
    # IndependentVariableTracking (cf. the OWA / PowerNorm / Turnover / DR-CDaR gaps fixed
    # in Phase 3). Computed Category-B keys `model[Symbol(:variance_risk_, i)]` and the
    # bare-by-design prior caches (`:G`, `:GV`, `:Gkt`, `:vals_Akt`, `:vecs_Akt`, `:frc_*`)
    # are naturally exempt. Residual hole (accepted, ADR 0004 §2): a sloppy
    # `model[Symbol(prefix, :newCatA)]` looks like Cat-B and slips through.
    #
    # Keep this list in sync with the prefix-namespaced keys (ADR 0005 swap groups + the
    # shared infra keys `:w`/`:X`/`:net_X`/`:Xap1`/`:ddap1`/`:dd`).
    #
    # The MIP indicator-bundle keys (ADR 0033) are locked for a related reason: a feature that
    # consumes an indicator must take the typed bundle and read it through `held`/`lb_gate`/…,
    # never `model[:ib]`. Reading by key is exactly the regression that made `set_iplg_constraints!`
    # throw `KeyError(:ib)` on a short + threshold/fee/xbgt model (only the long-only builder
    # registers `:ib`; the long-short builder registers `:ilb`/`:isb`). These keys are also
    # `mip_key`-namespaced under a sub-group space, so a bare read grabs the wrong (or absent) bit.
    managed = [:w, :X, :net_X, :Xap1, :ddap1, :dd, :cdd_start, :cdd_geq_0, :cdd, :W, :M,
               :M_PSD, :L2W, :variance_flag, :rc_variance, :Au, :Al, :cbucs_variance, :E,
               :WpE, :ceucs_variance, :wr_risk, :cwr, :range_risk, :br_risk, :cbr,
               :mdd_risk, :cmdd_risk, :uci, :uci_risk, :cuci_soc, :owa, :owac,
               :bdvariance_risk, :Dt, :Dx, :W1_vr_sk_kt, :W2_vr_sk_kt, :W3_vr_sk_kt,
               :L2W1_vr_sk_kt, :M_vr_sk_kt, :M_vr_sk_kt_PSD,
               # MIP indicator-bundle keys (ADR 0033): read via the bundle, never by key.
               :ib, :ibf, :i_mip, :ilb, :isb, :il, :is, :ilf, :isf, :xbgt_ib]
    pat = Regex("model\\[:(" * join(managed, "|") * ")\\]")
    srcdir = normpath(joinpath(@__DIR__, "..", "src"))
    interface = "08_Base_JuMPOptimisation.jl"

    violations = String[]
    for (root, _, files) in walkdir(srcdir), file in files
        if !(endswith(file, ".jl") && file != interface)
            continue
        end
        path = joinpath(root, file)
        indoc = false
        for (lineno, raw) in enumerate(eachline(path))
            # Skip `"""` docstring blocks and `#` line comments so prose key mentions
            # (e.g. "stored as model[:W]") do not trip the lock.
            stripped = lstrip(raw)
            inprose = indoc || startswith(stripped, "#") || startswith(stripped, "\"\"\"")
            if !inprose && occursin(pat, raw)
                push!(violations,
                      string(relpath(path, srcdir), ":", lineno, "  ", strip(raw)))
            end
            if isodd(count("\"\"\"", raw))
                (indoc = !indoc)
            else
                false
            end
        end
    end

    # Self-check: the matcher actually fires on a bare prefix-managed read.
    @test occursin(pat, "x = model[:net_X]")
    @test !occursin(pat, "x = model[Symbol(prefix, :net_X)]")
    @test !occursin(pat, "G = model[:G]")  # bare-by-design cache, exempt

    if !isempty(violations)
        println("Seam-lock violations — read these via a prefixed accessor or ",
                "model[Symbol(prefix, ...)]:")
        foreach(v -> println("  ", v), violations)
    end
    @test isempty(violations)
end
