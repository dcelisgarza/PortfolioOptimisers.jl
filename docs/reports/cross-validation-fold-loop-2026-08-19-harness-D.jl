# Four loop shapes, one process, one package version (D = HEAD + the ElT fix + the Fold
# record). A, B and the OLD C are reimplemented locally so only the shape varies.
using PortfolioOptimisers, Chairmarks, Printf, LinearAlgebra, Test
BLAS.set_num_threads(1)
const PO = PortfolioOptimisers
const FLoops = PO.FLoops

const est = EqualWeighted()
const rd = ReturnsResult(; nx = ["A", "B"], X = randn(64, 2) ./ 100)
const n = 60
const tr = [collect(1:i) for i in 1:n]
const te = [collect((i + 1):(i + 2)) for i in 1:n]
const ex = FLoops.SequentialEx()

# --- local copies of the PRE-FIX helpers, so shapes A/B/C emulate the old code exactly ---
function parallel_folds_old(fit_fold, n::Integer, ex; ElT = PO.PredictionResult)
    predictions = Vector{ElT}(undef, n)
    FLoops.@floop ex for i in 1:n
        predictions[i] = fit_fold(i)
    end
    return predictions
end
function run_folds_old(fit_fold, opt, n::Integer, ex; ElT = PO.PredictionResult,
                       prev_w_flag::Bool = PO.needs_previous_weights(opt))
    if prev_w_flag
        predictions = Vector{ElT}(undef, n)
        for i in 1:n
            predictions[i] = fit_fold(i, i > 1 ? predictions[i - 1] : nothing)
        end
        return predictions
    end
    return parallel_folds_old(i -> fit_fold(i, nothing), n, ex; ElT = ElT)
end

@inline function work(k)
    s = 0.0
    for i in 1:k
        s += sqrt(i)
    end
    return s
end

# ---- shape A: the v0.23.2 explicit form ------------------------------------------
function shapeA(k)
    predictions = Vector{Float64}(undef, length(tr))
    prev_w_flag = PO.needs_previous_weights(est)
    time_dep_flag = PO.is_time_dependent(est)
    if prev_w_flag || time_dep_flag
        opt = est
        for (i, (train, test)) in enumerate(zip(tr, te))
            if i > 1 && prev_w_flag
                opt = PO.factory(opt, predictions[i - 1])
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

# ---- shape B: run_folds + a hand-written per-fold prologue at the call site -------
function shapeB(k)
    td_flag = PO.is_time_dependent(est)
    if td_flag
        PO.assert_time_dependent_fold_count(est, n)
    end
    return run_folds_old(est, n, ex; ElT = Float64) do i, prev
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

# ---- shape C: the OLD seam — ElT as a keyword, five positional callback params ----
function fold_loop_old(fit_fold, est, n::Integer, ex; rd, train_idx, test_idx,
                       path_id = nothing, ElT = PO.PredictionResult,
                       time_ordered::Bool = true, fold_view = nothing)
    td_flag = PO.is_time_dependent(est)
    if td_flag
        PO.assert_time_dependent_fold_count(est, n)
    end
    prev_w_flag = PO.needs_previous_weights(est)
    function fold(i, prev)
        w_prev = isnothing(prev) ? nothing : prev.res.w
        (esti, rdi) = isnothing(fold_view) ? (est, rd) : fold_view(i)
        if td_flag
            ctx = PO.TimeDependentContext(; i = i, n = n, rd = rdi, train_idx = train_idx,
                                          test_idx = test_idx, w_prev = w_prev,
                                          path_id = path_id)
            esti = PO.update_time_dependent_estimator(esti, ctx)
        end
        if !isnothing(w_prev) && prev_w_flag
            esti = PO.factory(esti, w_prev)
        end
        return fit_fold(i, esti, rdi, train_idx[i], test_idx[i])
    end
    return if time_ordered
        run_folds_old(fold, est, n, ex; ElT = ElT, prev_w_flag = prev_w_flag)
    else
        parallel_folds_old(i -> fold(i, nothing), n, ex; ElT = ElT)
    end
end
function shapeC(k)
    fold_loop_old(est, n, ex; rd = rd, train_idx = tr, test_idx = te, ElT = Float64
                  ) do i, esti, rdi, a, b
        return work(k)
    end
end

# ---- shape D: today's seam — positional ElT, one Fold record ----------------------
function shapeD(k)
    PO.fold_loop(est, n, ex, Float64; rd = rd, train_idx = tr, test_idx = te) do fold
        return work(k)
    end
end

println("### harness micro-benchmark (n = $n folds, SequentialEx, 1 thread)")
println("julia $(VERSION), PortfolioOptimisers $(pkgversion(PO))")

const SHAPES = ("A explicit" => shapeA, "B run_folds" => shapeB, "C old seam" => shapeC,
                "D new seam" => shapeD)

for (nm, f) in SHAPES
    r = f(0)
    @printf("  %-12s returns %s len=%d\n", nm, nameof(typeof(r)), length(r))
end

println("\n## inferrability of each shape (per-fold work k = 0)")
for (nm, f) in SHAPES
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
@printf("  %-8s %-16s %-16s %-16s %-16s\n", "k", "A explicit", "B run_folds", "C old seam",
        "D new seam")
for k in (0, 10, 100, 1_000, 10_000, 100_000)
    res = map(SHAPES) do (nm, f)
        return minimum(@be f($k) seconds = 2)
    end
    @printf("  %-8d %-16s %-16s %-16s %-16s\n", k,
            @sprintf("%.3f us/%d a", res[1].time * 1e6, res[1].allocs),
            @sprintf("%.3f us/%d a", res[2].time * 1e6, res[2].allocs),
            @sprintf("%.3f us/%d a", res[3].time * 1e6, res[3].allocs),
            @sprintf("%.3f us/%d a", res[4].time * 1e6, res[4].allocs))
end
println("\n### DONE micro D")
