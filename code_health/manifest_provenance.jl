#!/usr/bin/env julia
#=
Refuse a commit whose `code_health/Manifest.toml` names an analyser version that a baseline's
`[provenance]` block does not record.

    julia --startup-file=no code_health/manifest_provenance.jl

The gates refuse to compare a baseline measured by another version of their analyser, and say so on
CI. A commit that moved JET in the Manifest without the refresh turned the JET check red on three
heads of `dev` in a row (2026-10-20). This check runs at commit time, in a second, on the same
comparison: the Julia version and the versions of JET, CodeComplexity and JuliaSyntax. It loads no
package but TOML.

The route when it fails: keep the new Manifest, and in a commit that changes nothing under src/ or
ext/ run `julia -t 1 --project=code_health code_health/<gate>.jl refresh --accept-rise` for each
baseline it names.
=#
module ManifestProvenance

using TOML

const DIR = @__DIR__

# A provenance key, and the Manifest package whose version it records. `julia` reads the
# Manifest's `julia_version`.
const PACKAGES = ("jet" => "JET", "code_complexity" => "CodeComplexity",
                  "julia_syntax" => "JuliaSyntax")

"""
    manifest_versions(path) -> Dict{String, String}

The version of each provenance key in the Manifest `path`, keyed as a `[provenance]` block keys it.
"""
function manifest_versions(path::AbstractString)
    m = TOML.parsefile(path)
    out = Dict("julia" => string(get(m, "julia_version", "")))
    deps = get(m, "deps", Dict{String, Any}())
    for (key, name) in PACKAGES
        rows = get(deps, name, Any[])
        if !(isempty(rows))
            out[key] = string(get(first(rows), "version", ""))
        end
    end
    return out
end

"""
    mismatches(dir = DIR) -> Vector{String}

One line per baseline under `dir` whose `[provenance]` records a version that the Manifest does not
hold. A key the Manifest does not name is not compared.
"""
function mismatches(dir::AbstractString = DIR)
    now = manifest_versions(joinpath(dir, "Manifest.toml"))
    out = String[]
    for f in sort(readdir(dir))
        if !(endswith(f, "_baseline.toml") || f == "expansion_bound.toml")
            continue
        end
        prov = get(TOML.parsefile(joinpath(dir, f)), "provenance", Dict{String, Any}())
        for (key, recorded) in sort(collect(prov); by = first)
            if haskey(now, key) && string(recorded) != now[key]
                push!(out,
                      "code_health/$f: $key is \"$recorded\", the Manifest holds \"$(now[key])\"")
            end
        end
    end
    return out
end

function main()
    bad = mismatches()
    if isempty(bad)
        return 0
    end
    println("code_health/Manifest.toml moved under these baselines:")
    for line in bad
        println("  ", line)
    end
    println("Keep the new Manifest. In a commit that changes nothing under src/ or ext/, run ",
            "`julia -t 1 --project=code_health code_health/<gate>.jl refresh --accept-rise` for ",
            "each baseline named above, and commit the Manifest with the refreshed baselines.")
    return 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(ManifestProvenance.main())
end
