using CodeComplexity, Printf

const T = "/home/danielcelisgarza/.claude/jobs/a3454584/tmp"
const VERSIONS = ["A" => "$T/A", "B" => "$T/B",
                  "C" => "/mnt/storage/dev/PortfolioOptimisers.jl/dev"]
const FILES = ["src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl",
               "src/20_Optimisation/02_CrossValidation/03_Combinatorial.jl",
               "src/20_Optimisation/02_CrossValidation/04_WalkForward.jl",
               "src/20_Optimisation/02_CrossValidation/05_MultipleRandomised.jl",
               "src/20_Optimisation/02_CrossValidation/06_Validation.jl",
               "src/23_Pipeline/04_PredictionCV.jl"]
const WANT = ["fit_and_predict", "cross_val_predict", "path_fit_and_predict", "fold_loop",
              "run_folds", "parallel_folds", "assert_unshuffled_folds",
              "assert_time_dependent_fold_count", "cv_sequential_info"]
const METRICS = (CyclomaticComplexity(), CognitiveComplexity())

hit(name) = any(w -> occursin(w, name), WANT)

for (lbl, root) in VERSIONS
    println("\n#### version $lbl  ($root)")
    for m in METRICS
        tot = 0
        n = 0
        rows = Tuple{Int, String, String, Int}[]
        for f in FILES
            p = joinpath(root, f)
            if !(isfile(p))
                continue
            end
            fm = measure_file(m, p)
            for fn in fm.functions
                if !(hit(fn.name))
                    continue
                end
                tot += fn.value
                n += 1
                push!(rows, (fn.value, fn.name, basename(f), fn.line))
            end
        end
        @printf("  %-11s defs=%3d  sum=%4d  max=%3d  mean=%5.2f\n", metric_label(m), n, tot,
                isempty(rows) ? 0 : maximum(first, rows), n == 0 ? 0.0 : tot/n)
        sort!(rows; by = x -> -x[1])
        for (v, nm, f, l) in first(rows, 6)
            @printf("       %3d  %-22s %-32s :%d\n", v, nm, f, l)
        end
    end
end
