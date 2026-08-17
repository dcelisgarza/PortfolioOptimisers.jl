####
# Ticket #336 — measure the CodeComplexity baseline over the whole tree.
# Runs in an isolated project so the shared REPL is untouched.
####
using CodeComplexity, JuliaSyntax, Statistics, Printf

const REPO = "/mnt/storage/dev/PortfolioOptimisers.jl/dev"
const OUT = "/home/danielcelisgarza/.claude/jobs/2da2f9e5/tmp"

const METRICS = (CyclomaticComplexity(), CognitiveComplexity(), ArgumentCountComplexity())
const MLABEL = Dict(CyclomaticComplexity() => "cyclomatic",
                    CognitiveComplexity() => "cognitive",
                    ArgumentCountComplexity() => "argcount")

# ---- scope -----------------------------------------------------------------

const ROOTS = ["src", "ext", "test", "docs", "examples", "user_guide"]

function tracked_jl()
    out = read(`git -C $REPO ls-files -z -- "*.jl"`, String)
    return sort(filter(!isempty, split(out, '\0')))
end

function walked_jl(root)
    paths = String[]
    d = joinpath(REPO, root)
    if !(isdir(d))
        return paths
    end
    for (r, _, files) in walkdir(d), f in files
        endswith(f, ".jl") && push!(paths, relpath(joinpath(r, f), REPO))
    end
    return sort(paths)
end

part_of(p) = begin
    i = findfirst(r -> startswith(p, r * "/"), ROOTS)
    i === nothing ? "OTHER" : ROOTS[i]
end

# ---- parseability ----------------------------------------------------------

function parse_status(path)
    code = read(joinpath(REPO, path), String)
    try
        JuliaSyntax.parseall(Expr, code; ignore_errors = false)
        return (:ok, "")
    catch e
        return (:parse_error, first(sprint(showerror, e), 300))
    end
end

# ---- measure ---------------------------------------------------------------

pct(v, q) = isempty(v) ? 0.0 : quantile(sort(v), q)

function main()
    files = tracked_jl()
    files = filter(p -> part_of(p) != "OTHER" || true, files)

    # 1. scope diff: what a blind walkdir sees that git does not.
    tracked = Set(files)
    walk_extra = Dict{String, Vector{String}}()
    for r in ROOTS
        extra = filter(p -> !(p in tracked), walked_jl(r))
        isempty(extra) || (walk_extra[r] = extra)
    end

    # 2. parseability of every tracked file.
    bad = Tuple{String, String}[]
    for p in files
        st, msg = parse_status(p)
        st == :ok || push!(bad, (p, msg))
    end

    # 3. the measurement itself.
    #    data[metric][file] = Vector{FunctionMeasure}
    data = Dict{String, Dict{String, Vector{Any}}}()
    times = Dict{String, Float64}()
    for m in METRICS
        lbl = MLABEL[m]
        d = Dict{String, Vector{Any}}()
        t = @elapsed for p in files
            fm = measure_file(m, joinpath(REPO, p))
            d[p] = fm.functions
        end
        data[lbl] = d
        times[lbl] = t
    end

    open(joinpath(OUT, "report_336.txt"), "w") do io
        println(io, "# Ticket #336 — CodeComplexity baseline")
        println(io, "repo    : ", REPO)
        println(io, "commit  : ", strip(read(`git -C $REPO rev-parse HEAD`, String)))
        println(io, "julia   : ", VERSION)
        println(io, "CodeComplexity: ", pkgversion(CodeComplexity))
        println(io, "JuliaSyntax   : ", pkgversion(JuliaSyntax))
        println(io)

        println(io, "## 1. Scope")
        println(io, "tracked .jl files in scope: ", length(files))
        for r in ROOTS
            n = count(p -> part_of(p) == r, files)
            @printf(io, "  %-12s %4d files\n", r, n)
        end
        other = filter(p -> part_of(p) == "OTHER", files)
        @printf(io, "  %-12s %4d files: %s\n", "OTHER", length(other), join(other, ", "))
        println(io)
        println(io, "### What a blind walkdir picks up that git does not")
        if isempty(walk_extra)
            println(io, "  (nothing)")
        else
            for (r, v) in sort(collect(walk_extra); by = first)
                @printf(io, "  %-12s %4d extra files\n", r, length(v))
                for p in first(v, 12)
                    println(io, "      ", p)
                end
                length(v) > 12 && println(io, "      … +", length(v) - 12, " more")
            end
        end
        println(io)

        println(io, "## 2. Parseability (parseall with ignore_errors=false)")
        if isempty(bad)
            println(io, "  every tracked .jl file parses cleanly.")
        else
            println(io, "  ", length(bad), " file(s) FAIL to parse:")
            for (p, msg) in bad
                println(io, "    ", p, " :: ", replace(msg, '\n' => " | "))
            end
        end
        println(io)

        println(io, "## 3. Wall time (single thread, warm package)")
        for m in METRICS
            @printf(io, "  %-12s %8.3f s\n", MLABEL[m], times[MLABEL[m]])
        end
        @printf(io, "  %-12s %8.3f s\n", "TOTAL", sum(values(times)))
        println(io)

        for m in METRICS
            lbl = MLABEL[m]
            d = data[lbl]
            println(io, "## 4.", lbl, " — distribution per part of the tree")
            @printf(io, "  %-12s %7s %7s %7s %7s %7s %7s %7s\n", "part", "files", "defs",
                    "median", "p90", "p99", "max", "sum")
            for r in vcat(ROOTS, ["OTHER"])
                ps = filter(p -> part_of(p) == r, files)
                if isempty(ps)
                    continue
                end
                vals = Int[]
                for p in ps, f in d[p]
                    push!(vals, f.value)
                end
                if isempty(vals)
                    @printf(io, "  %-12s %7d %7d %7s %7s %7s %7s %7s\n", r, length(ps), 0,
                            "-", "-", "-", "-", "-")
                    continue
                end
                @printf(io, "  %-12s %7d %7d %7.1f %7.1f %7.1f %7d %7d\n", r, length(ps),
                        length(vals), median(vals), pct(vals, 0.90), pct(vals, 0.99),
                        maximum(vals), sum(vals))
            end
            allvals = Int[]
            for p in files, f in d[p]
                push!(allvals, f.value)
            end
            @printf(io, "  %-12s %7d %7d %7.1f %7.1f %7.1f %7d %7d\n", "WHOLE TREE",
                    length(files), length(allvals), median(allvals), pct(allvals, 0.90),
                    pct(allvals, 0.99), maximum(allvals), sum(allvals))
            println(io)

            println(io, "  ### ten worst definitions (", lbl, ")")
            worst = Tuple{Int, String, String, Int}[]
            for p in files, f in d[p]
                push!(worst, (f.value, f.name, p, f.line))
            end
            sort!(worst; by = x -> -x[1])
            for (v, n, p, l) in first(worst, 10)
                @printf(io, "    %5d  %-34s %s:%d\n", v, first(n, 34), p, l)
            end
            println(io)

            println(io, "  ### per-file aggregate candidates (", lbl, ")")
            thr = CodeComplexity.default_max_value(m)
            sums = [(sum(f.value for f in d[p]; init = 0), p) for p in files]
            maxs = [(maximum((f.value for f in d[p]); init = 0), p) for p in files]
            cnts = [(count(f -> f.value > thr, d[p]), p) for p in files]
            for (nm, v) in
                (("SUM over defs", sums), ("MAX over defs", maxs), ("COUNT > $thr", cnts))
                nums = first.(v)
                nz = count(>(0), nums)
                @printf(io,
                        "    %-16s median=%6.1f p90=%7.1f max=%6d sum=%7d nonzero_files=%d\n",
                        nm, median(nums), pct(nums, 0.90), maximum(nums), sum(nums), nz)
            end
            println(io, "    ten worst files by SUM:")
            sort!(sums; by = x -> -x[1])
            for (v, p) in first(sums, 10)
                @printf(io, "      %6d  %s\n", v, p)
            end
            println(io, "    ten worst files by MAX:")
            sort!(maxs; by = x -> -x[1])
            for (v, p) in first(maxs, 10)
                @printf(io, "      %6d  %s\n", v, p)
            end
            println(io, "    ten worst files by COUNT > ", thr, ":")
            sort!(cnts; by = x -> -x[1])
            for (v, p) in first(cnts, 10)
                @printf(io, "      %6d  %s\n", v, p)
            end
            println(io)
        end

        # anonymous-function share: relevant because the baseline is per file
        println(io, "## 5. Share of definitions that are `<anonymous>`")
        for m in METRICS
            lbl = MLABEL[m]
            d = data[lbl]
            tot = 0
            anon = 0
            for p in files, f in d[p]
                tot += 1
                f.name == "<anonymous>" && (anon += 1)
            end
            @printf(io, "  %-12s %d / %d (%.1f%%)\n", lbl, anon, tot, 100 * anon / tot)
        end
        println(io)

        println(io, "## 6. Files with zero definitions found")
        for m in METRICS
            lbl = MLABEL[m]
            z = filter(p -> isempty(data[lbl][p]), files)
            @printf(io, "  %-12s %d files\n", lbl, length(z))
            if lbl == "cyclomatic"
                for p in first(z, 20)
                    println(io, "      ", p)
                end
                length(z) > 20 && println(io, "      … +", length(z) - 20, " more")
            end
        end
    end

    # machine-readable dump: metric, file, name, value, line
    open(joinpath(OUT, "measures_336.csv"), "w") do io
        println(io, "metric,file,name,value,line")
        for m in METRICS
            lbl = MLABEL[m]
            for p in files, f in data[lbl][p]
                nm = replace(f.name, ',' => ";", '"' => "'")
                println(io, lbl, ",", p, ",\"", nm, "\",", f.value, ",", f.line)
            end
        end
    end
    return nothing
end

main()
