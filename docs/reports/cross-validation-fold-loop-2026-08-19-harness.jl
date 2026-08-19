# Isolate the LOOP HARNESS from the per-fold work: three loop shapes, one process,
# one Julia, one package version (HEAD). Only the shape of the fold loop varies.
using PortfolioOptimisers, Chairmarks, Printf, LinearAlgebra, Test
BLAS.set_num_threads(1)
const PO = PortfolioOptimisers
const FLoops = PO.FLoops

const est = EqualWeighted()                       # is_time_dependent = false, needs_prev_w = false
const rd = ReturnsResult(; nx = ["A", "B"], X = randn(64, 2) ./ 100)
const n = 60
const tr = [collect(1:i) for i in 1:n]
const te = [collect((i + 1):(i + 2)) for i in 1:n]
const ex = FLoops.SequentialEx()

@inline function work(k)
    s = 0.0
    for i in 1:k
        s += sqrt(i)
    end
    return s
end

# ---- shape C: today's single seam ------------------------------------------------
function shapeC(k)
    PO.fold_loop(est, n, ex; rd = rd, train_idx = tr, test_idx = te, ElT = Float64
                 ) do i, esti, rdi, a, b
        return work(k)
    end
end

# ---- shape B: run_folds + a hand-written per-fold prologue at the call site -------
function shapeB(k)
    td_flag = PO.is_time_dependent(est)
    if td_flag
        PO.assert_time_dependent_fold_count(est, n)
    end
    return PO.run_folds(est, n, ex; ElT = Float64) do i, prev
        esti = est
        if td_flag
            ctx = PO.TimeDependentContext(; i = i, n = n, rd = rd, train_idx = tr,
                                          test_idx = te,
                                          w_prev = isnothing(prev) ? nothing : prev.res.w)
            esti = PO.update_time_dependent_estimator(esti, ctx)
        end
        if !isnothing(prev) && PO.needs_previous_weights(est)
            esti = PO.factory(esti, prev.res.w)
        end
        return work(k)
    end
end

# ---- shape A: the v0.23.2 explicit form ------------------------------------------
function shapeA(k)
    predictions = Vector{Float64}(undef, length(tr))
    prev_w_flag = PO.needs_previous_weights(est)
    time_dep_flag = PO.is_time_dependent(est)
    if prev_w_flag || time_dep_flag
        opt = est
        for (i, (train, test)) in enumerate(zip(tr, te))
            if i > 1
                if prev_w_flag
                    opt = PO.factory(opt, predictions[i - 1])
                end
            end
            predictions[i] = work(k)
        end
    else
        let est = est
            FLoops.@floop ex for (i, (train, test)) in enumerate(zip(tr, te))
                predictions[i] = work(k)
            end
        end
    end
    return predictions
end

println("### harness micro-benchmark (n = $n folds, SequentialEx, 1 thread)")
println("julia $(VERSION), PortfolioOptimisers $(pkgversion(PO))")

for (nm, f) in ("A explicit" => shapeA, "B run_folds" => shapeB, "C fold_loop" => shapeC)
    r = f(0)
    @printf("  %-12s returns %s len=%d\n", nm, nameof(typeof(r)), length(r))
end

println("\n## inferrability of each shape (per-fold work k = 0)")
for (nm, f) in ("A explicit" => shapeA, "B run_folds" => shapeB, "C fold_loop" => shapeC)
    rt = Base.return_types(f, Tuple{Int})
    inf = try
        @inferred f(0)
        true
    catch
        false
    end
    ct = code_typed(f, Tuple{Int}; optimize = true)[1]
    s = sprint(show, ct[1])
    @printf("  %-12s return_types=%-22s @inferred=%-5s Core.Box=%d stmts=%d\n", nm,
            string(rt), inf, count(_ -> true, eachmatch(r"Core\.Box", s)),
            length(ct[1].code))
end

println("\n## Chairmarks: total loop time for $n folds, by per-fold work size k")
@printf("  %-8s %-14s %-14s %-14s\n", "k", "A explicit", "B run_folds", "C fold_loop")
for k in (0, 10, 100, 1_000, 10_000, 100_000)
    res = map(("A" => shapeA, "B" => shapeB, "C" => shapeC)) do (nm, f)
        return minimum(@be f($k) seconds = 2)
    end
    @printf("  %-8d %-14s %-14s %-14s\n", k,
            @sprintf("%.3f us/%d a", res[1].time * 1e6, res[1].allocs),
            @sprintf("%.3f us/%d a", res[2].time * 1e6, res[2].allocs),
            @sprintf("%.3f us/%d a", res[3].time * 1e6, res[3].allocs))
end
println("\n### DONE micro")
