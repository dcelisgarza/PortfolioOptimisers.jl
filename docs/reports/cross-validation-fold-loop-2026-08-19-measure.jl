# Fold-loop comparison: performance, inferrability, type stability.
# Runs unchanged in v0.23.2 (A), 0.27.0 (B) and HEAD (C).
using PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Chairmarks, JET, Printf
BLAS.set_num_threads(1)
const PO = PortfolioOptimisers
const LBL = get(ENV, "PO_LBL", "?")

rng = StableRNG(987654321)
const T_, N_ = 1008, 20
X = randn(rng, T_, N_) ./ 100
nx = ["A$i" for i in 1:N_]
rd = ReturnsResult(; nx = nx, X = X)

ew = EqualWeighted()
iv = InverseVolatility()

cvk = KFold(; n = 10)
cvw = IndexWalkForward(127, 171)
cvc = CombinatorialCrossValidation(; n_folds = 6, n_test_folds = 2, purged_size = 5,
                                   embargo_size = 3)
cvm = MultipleRandomised(IndexWalkForward(127, 171); subset_size = 5, n_subsets = 3,
                         rng = StableRNG(666), seed = 69)

const CVS = ["KFold" => cvk, "WalkForward" => cvw, "Combinatorial" => cvc,
             "MultipleRandomised" => cvm]
const OPTS = ["EqualWeighted" => ew, "InverseVolatility" => iv]

println("### VERSION $LBL  (PortfolioOptimisers ", pkgversion(PO), ", julia ", VERSION, ")")

# ---------------------------------------------------------------- 1. correctness / warmup
println("\n## warmup")
for (on, o) in OPTS, (cn, c) in CVS
    try
        r = cross_val_predict(o, rd, c)
        @printf("  ok   %-18s %-18s -> %s\n", on, cn, nameof(typeof(r)))
    catch e
        @printf("  FAIL %-18s %-18s :: %s\n", on, cn, first(sprint(showerror, e), 160))
    end
end

# ---------------------------------------------------------------- 2. inferrability
println("\n## return_types(cross_val_predict, ...)")
for (on, o) in OPTS, (cn, c) in CVS
    rt = try
        Base.return_types(cross_val_predict, Tuple{typeof(o), typeof(rd), typeof(c)})
    catch e
        [Symbol("ERR")]
    end
    @printf("  %-18s %-18s n=%d  concrete=%-5s  %s\n", on, cn, length(rt),
            length(rt) == 1 && isconcretetype(rt[1]), rt)
end

# ---------------------------------------------------------------- 3. type stability of the loop
function ir_stats(f, tt)
    ci = try
        code_typed(f, tt; optimize = true)
    catch e
        return nothing
    end
    if isempty(ci)
        return nothing
    end
    (c, rt) = ci[1]
    s = sprint(show, c)
    boxes = count(_ -> true, eachmatch(r"Core\.Box", s))
    ssat = c.ssavaluetypes
    types = isa(ssat, Vector) ? ssat : Any[]
    nconc = count(t -> isa(t, Type) && isconcretetype(t), types)
    nany = count(t -> t === Any, types)
    return (; nstmt = length(c.code), nssa = length(types), nconc, nany, boxes, rt)
end

println("\n## code_typed of the fold-loop entry (fit_and_predict(opt, rd, cv))")
for (on, o) in OPTS, (cn, c) in CVS
    st = ir_stats(PO.fit_and_predict, Tuple{typeof(o), typeof(rd), typeof(c)})
    if isnothing(st)
        @printf("  %-18s %-18s  (no method / not inferrable)\n", on, cn)
        continue
    end
    @printf("  %-18s %-18s stmts=%4d ssa=%4d concrete=%4d Any=%3d Core.Box=%2d rt=%s\n", on,
            cn, st.nstmt, st.nssa, st.nconc, st.nany, st.boxes, st.rt)
end

# ---------------------------------------------------------------- 4. JET
println("\n## JET @report_opt (target_modules = (PortfolioOptimisers,))")
for (on, o) in OPTS, (cn, c) in CVS
    n = try
        rep = JET.report_opt(cross_val_predict, Tuple{typeof(o), typeof(rd), typeof(c)};
                             target_modules = (PO,))
        length(JET.get_reports(rep))
    catch e
        -1
    end
    @printf("  %-18s %-18s opt_reports=%5d\n", on, cn, n)
end
println("\n## JET @report_call (target_modules = (PortfolioOptimisers,))")
for (on, o) in OPTS, (cn, c) in CVS
    n = try
        rep = JET.report_call(cross_val_predict, Tuple{typeof(o), typeof(rd), typeof(c)};
                              target_modules = (PO,))
        length(JET.get_reports(rep))
    catch e
        -1
    end
    @printf("  %-18s %-18s call_reports=%5d\n", on, cn, n)
end

# ---------------------------------------------------------------- 5. Chairmarks
println("\n## Chairmarks @be cross_val_predict (single thread)")
for (on, o) in OPTS, (cn, c) in CVS
    try
        b = @be cross_val_predict($o, $rd, $c) seconds = 3
        m = minimum(b)
        @printf("  %-18s %-18s min=%10.4f ms  allocs=%9d  bytes=%12d  samples=%d\n", on, cn,
                m.time * 1e3, m.allocs, m.bytes, length(b.samples))
    catch e
        @printf("  %-18s %-18s BENCH FAIL :: %s\n", on, cn,
                first(sprint(showerror, e), 120))
    end
end
println("\n### DONE $LBL")
