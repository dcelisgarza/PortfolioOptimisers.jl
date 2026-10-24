#=
A machine-wide slot for a Julia run that loads the package: a test file, a doctest run, JET.

Parallel sessions share one machine. Each run that loads the package holds one to three GB, and a
retro of 2026-10-21 found runs killed for low memory while the maintainer counted the busy sessions
by hand. A run that takes a slot waits until a slot is free and the machine has enough free memory,
so a background run queues instead of being killed.

    JuliaSlot.with_slot(f, "run_files test_04_owa.jl")

A slot is a file `slot-<i>.pid` under the slot directory, created with `O_EXCL`, that holds the PID
and the label of its holder. A slot whose PID is dead is free, so a run that crashed releases its
slot. The run prints one `SLOT` line when it must wait, with the holders, and one when it starts.

Environment:
  PO_JULIA_SLOTS        the number of slots. Default 2. 0 turns the lock off.
  PO_JULIA_MIN_FREE_GB  the free memory (MemAvailable) a run waits for. Default 5.
  PO_JULIA_SLOT_DIR     the slot directory. Default `<tempdir>/po-julia-slots`.
  PO_JULIA_SLOT_HELD    set by a holder for its child processes, which then take no slot.
  CI                    `true` on a CI runner, where one job runs alone and takes no slot.
=#
module JuliaSlot

using FileWatching.Pidfile: isvalidpid

slots() = something(tryparse(Int, get(ENV, "PO_JULIA_SLOTS", "2")), 2)
min_free_gb() = something(tryparse(Float64, get(ENV, "PO_JULIA_MIN_FREE_GB", "5")), 5.0)
slot_dir() = get(ENV, "PO_JULIA_SLOT_DIR", joinpath(tempdir(), "po-julia-slots"))

"""
    mem_available_gb() -> Float64

The `MemAvailable` line of `/proc/meminfo` in GB, or `Inf` where the file does not exist.
"""
function mem_available_gb()
    if !(isfile("/proc/meminfo"))
        return Inf
    end
    for line in eachline("/proc/meminfo")
        m = match(r"^MemAvailable:\s+(\d+) kB", line)
        m === nothing || return parse(Int, m[1]) / 1024^2
    end
    return Inf
end

"""
    holder(path) -> Union{Nothing, Tuple{Int, String}}

The PID and the label in the slot file `path`, or `nothing` for a file that is absent or unread.
"""
function holder(path::AbstractString)
    text = try
        read(path, String)
    catch
        return nothing
    end
    words = split(strip(text), ' '; limit = 2)
    pid = tryparse(Int, first(words))
    return pid === nothing ? nothing : (pid, length(words) == 2 ? String(words[2]) : "")
end

alive(pid::Integer) = isvalidpid(gethostname(), Cuint(pid))

"""
    try_take(path, label) -> Bool

Create the slot file `path` for this process. A file whose holder is dead is removed first.
"""
function try_take(path::AbstractString, label::AbstractString)
    h = holder(path)
    if h !== nothing && !(alive(first(h)))
        rm(path; force = true)
    end
    flags = Base.Filesystem.JL_O_CREAT | Base.Filesystem.JL_O_EXCL |
            Base.Filesystem.JL_O_WRONLY
    f = try
        Base.Filesystem.open(path, flags, 0o644)
    catch e
        if e isa Base.IOError
            return false
        end
        rethrow()
    end
    try
        write(f, "$(getpid()) $label\n")
    finally
        close(f)
    end
    return true
end

"""
    active() -> Bool

Whether this process takes slots: not on CI, not under a holder, and not with zero slots.
"""
function active()
    if get(ENV, "CI", "") == "true"
        return false
    end
    if haskey(ENV, "PO_JULIA_SLOT_HELD")
        return false
    end
    return slots() > 0
end

"""
    acquire(label; poll = 15, io = stderr) -> Union{Nothing, String}

Wait for a free slot and enough free memory, take the slot, and return its path. It returns
`nothing` when the lock is off.
"""
function acquire(label::AbstractString; poll::Real = 15, io::IO = stderr)
    if !(active())
        return nothing
    end
    dir = slot_dir()
    mkpath(dir)
    waited = false
    while true
        if mem_available_gb() >= min_free_gb()
            for i in 1:slots()
                path = joinpath(dir, "slot-$i.pid")
                if try_take(path, label)
                    if waited
                        println(io, "SLOT start: $label")
                    end
                    return path
                end
            end
        end
        if !(waited)
            busy = filter(!isnothing,
                          [holder(joinpath(dir, "slot-$i.pid")) for i in 1:slots()])
            who = isempty(busy) ? "none" : join(("$p ($l)" for (p, l) in busy), ", ")
            println(io, "SLOT wait: $label waits for one of $(slots()) slot(s) and ",
                    "$(min_free_gb()) GB free; free now ",
                    round(mem_available_gb(); digits = 1), " GB; holders: ", who)
            flush(io)
            waited = true
        end
        sleep(poll)
    end
end

function release(path::Union{Nothing, AbstractString})
    if path === nothing
        return nothing
    end
    h = holder(path)
    if h !== nothing && first(h) == getpid()
        rm(path; force = true)
    end
    return nothing
end

"""
    with_slot(f, label)

Run `f()` while this process holds a slot, and release the slot after it. A child process that
`f` starts inherits `PO_JULIA_SLOT_HELD` and takes no slot of its own.
"""
function with_slot(f, label::AbstractString)
    path = acquire(label)
    if path === nothing
        return f()
    end
    ENV["PO_JULIA_SLOT_HELD"] = path
    try
        return f()
    finally
        delete!(ENV, "PO_JULIA_SLOT_HELD")
        release(path)
    end
end

end # module
