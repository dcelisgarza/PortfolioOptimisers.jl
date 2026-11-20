#=
The rows of `test/known_red.toml`, and the filter that lets a known red leave CI green (#1781).

Two programs read the rows, and both include this file:

  - `test/run_files.jl` names a known red on the `RESULT` line of its file. Its verdict does not
    change: a known red is still red there.
  - `test/runtests.jl`, the suite that CI runs. ParallelTestRunner prints every failure and then
    throws `Test.FallbackTestSetException`, and it returns no results. So `runtests.jl` copies
    what the runner prints to `stdout` into a buffer with a [`Tee`](@ref), and [`verdict`](@ref)
    reads the failure blocks of that text. The run passes when every failure is covered by a row,
    and it is red when one failure is not. Before 2026-10-21, one row that was red on the CI host
    turned every run red, so a new red behind it was not visible.

A failure block, as `Test.print_test_errors` prints it, starts with two lines:

    Error in testset <the innermost test set>:
    Test Failed at <file>:<line>            or    Error During Test at <file>:<line>

The log names only the innermost test set, not the path that `run_files.jl` joins with " > ". A
row covers a failure on CI when its `file` is the base name of `<file>` and its `testset` is a
substring of the innermost name. That is a substring of the path too, so a failure that CI covers
is covered by `run_files.jl`. A row whose `testset` names an outer set covers less on CI, never
more. A block that does not have this form, such as `Interrupted` or the `none:1` of a crashed
worker, is covered by no row.

This file is not a test: `test/runtests.jl` discovers only `test_*.jl`.
=#
module KnownRedRows

using TOML

const KNOWN_RED = joinpath(@__DIR__, "known_red.toml")

"""
    KnownRed

One row of `test/known_red.toml`: a test file, or one of its test sets, that is red for a reason
an open issue records.
"""
struct KnownRed
    file::String
    testset::Union{Nothing, String}
    issue::Int
    reason::String
    host::Union{Nothing, String}
end

const KNOWN_RED_KEYS = ("file", "testset", "issue", "reason", "host")

"""
    known_red(path = KNOWN_RED) -> Vector{KnownRed}

The rows of `path`, each a `[[red]]` table. A row with a key the driver does not read, or without
`file`, `issue` or `reason`, is an error, so a typo does not silently disable a row.
"""
function known_red(path::AbstractString = KNOWN_RED)
    if !(isfile(path))
        return KnownRed[]
    end
    rows = get(TOML.parsefile(path), "red", Any[])
    return map(rows) do r
        extra = setdiff(keys(r), KNOWN_RED_KEYS)
        if !(isempty(extra))
            error("`$path`: a row has the unknown key(s) $(join(sort(collect(extra)), ", ")).")
        end
        for k in ("file", "issue", "reason")
            if !(haskey(r, k))
                error("`$path`: a row has no `$k`.")
            end
        end
        return KnownRed(r["file"], get(r, "testset", nothing), r["issue"], r["reason"],
                        get(r, "host", nothing))
    end
end

function covers(r::KnownRed, path::AbstractString)
    return r.testset === nothing || occursin(r.testset, path)
end

function row_name(r::KnownRed)
    return r.testset === nothing ? r.file : "$(r.file) \"$(r.testset)\""
end

"""
    FailureSite

One failure block of the log of the suite: the innermost test set, and the file and line of the
failure. `file` is `nothing` for a block whose second line names no file and line.
"""
struct FailureSite
    testset::String
    file::Union{Nothing, String}
    line::Int
end

const ANSI = r"\e\[[0-9;]*[A-Za-z]"
const BLOCK_HEAD = r"^Error in testset (.*):$"
const BLOCK_SITE = r"^(?:Test Failed|Error During Test) at (.+):(\d+)$"

"""
    failure_sites(text) -> Vector{FailureSite}

The failure blocks that `Test.print_test_errors` printed in `text`, after the last
`Test Summary:` line. The output of a test file comes before that line, so a test that prints a
failure of its own fixture adds no site.
"""
function failure_sites(text::AbstractString)
    lines = split(replace(text, ANSI => ""), '\n')
    summary = findlast(l -> startswith(l, "Test Summary:"), lines)
    sites = FailureSite[]
    if summary === nothing
        return sites
    end
    for i in (summary + 1):length(lines)
        m = match(BLOCK_HEAD, rstrip(lines[i]))
        if m === nothing
            continue
        end
        s = i < length(lines) ? match(BLOCK_SITE, rstrip(lines[i + 1])) : nothing
        if s === nothing
            push!(sites, FailureSite(m[1], nothing, 0))
        else
            push!(sites, FailureSite(m[1], s[1], parse(Int, s[2])))
        end
    end
    return sites
end

"""
    covering_row(site, rows) -> Union{KnownRed, Nothing}

The first row that covers `site`: its `file` is the base name of the site's file, and its
`testset`, when it has one, is a substring of the site's innermost test set.
"""
function covering_row(site::FailureSite, rows::Vector{KnownRed})
    if site.file === nothing
        return nothing
    end
    i = findfirst(r -> r.file == basename(site.file) && covers(r, site.testset), rows)
    return i === nothing ? nothing : rows[i]
end

"""
    verdict(text, rows) -> (lines, green)

One line per failure block of `text`, `KNOWN RED` when a row covers it and `RED` when no row does,
and whether the run passes. A run passes only when it has at least one block and every block is
covered: a red run whose blocks this parser cannot find stays red.
"""
function verdict(text::AbstractString, rows::Vector{KnownRed})
    sites = failure_sites(text)
    lines = String[]
    for s in sites
        where = s.file === nothing ? "(no file and line)" : "$(s.file):$(s.line)"
        r = covering_row(s, rows)
        if r === nothing
            push!(lines,
                  "RED $where in \"$(s.testset)\": no row of test/known_red.toml covers it")
        else
            host = r.host === nothing ? "" : " ($(r.host))"
            push!(lines,
                  "KNOWN RED $where in \"$(s.testset)\": #$(r.issue): $(r.reason)$host")
        end
    end
    green = !(isempty(sites)) && all(startswith("KNOWN RED"), lines)
    if isempty(sites)
        push!(lines,
              "RED the run failed, and its log holds no failure block that a row can cover")
    elseif green
        push!(lines,
              "KNOWN RED only: test/known_red.toml covers every failure, so the run passes")
    end
    return lines, green
end

"""
    Tee(out, log)

A stream that writes every byte to `out` and to the buffer `log`, so the log of CI still streams
while the text is kept for [`verdict`](@ref).
"""
struct Tee <: IO
    out::IO
    log::IOBuffer
    lock::ReentrantLock
end
Tee(out::IO, log::IOBuffer) = Tee(out, log, ReentrantLock())

function Base.write(t::Tee, b::UInt8)
    @lock t.lock begin
        write(t.log, b)
        return write(t.out, b)
    end
end
function Base.unsafe_write(t::Tee, p::Ptr{UInt8}, n::UInt)
    @lock t.lock begin
        unsafe_write(t.log, p, n)
        return unsafe_write(t.out, p, n)
    end
end
Base.flush(t::Tee) = flush(t.out)

end # module
