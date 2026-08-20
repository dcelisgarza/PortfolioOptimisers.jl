# =============================================================================
# Prototype 22 — Provenance, and compatibility checked before the solve.
#
# Purpose
#   Report 4 raises two ideas that look like housekeeping and are not.
#
#   **Provenance.** A portfolio should be able to describe how it was made.
#   Once the library supports synthetic data, model ensembles and uncertainty
#   propagation, a weight vector on its own stops being reproducible, because
#   the seed, the estimator chain and the data version all matter and none of
#   them is in the vector.
#
#   **Compatibility.** The library has 229 abstract types and a great many
#   valid combinations of them. It also has many invalid ones, and today an
#   invalid one is discovered when a `MethodError` surfaces from deep inside a
#   solve. A declared capability layer turns that into a sentence, before any
#   work is done.
#
#   The second idea is the one with teeth. A risk measure **requires** certain
#   moments; a prior **provides** certain moments. Both facts are already
#   implicit in the method tables. Making them explicit costs one method per
#   type and buys a checkable contract.
#
# Status
#   Standalone. Depends on `Dates` and `SHA` from the standard library.
#
# Notation used throughout this file
#   capability  A symbol naming a computed quantity: `:mu`, `:sigma`, `:sk`,
#               `:kt`, `:X`, `:F`.
#   provides    The set of capabilities a component produces.
#   requires    The set a component consumes.
#   step        One recorded stage of a construction.
#
# Sources
#   Sandve, G. K., Nekrutenko, A., Taylor, J. and Hovig, E. (2013). Ten simple
#     rules for reproducible computational research. PLoS Computational Biology
#     9(10), e1003285. Rule 6 is "always record seeds".
#   Peng, R. D. (2011). Reproducible research in computational science. Science
#     334(6060), 1226-1227.
#   Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2014).
#     Pseudo-mathematics and financial charlatanism: the effects of backtest
#     overfitting on out-of-sample performance. Notices of the American
#     Mathematical Society 61(5), 458-471. Why the number of configurations
#     tried is itself a quantity that must be recorded.
#   Wadler, P. (2015). Propositions as types. Communications of the ACM 58(12),
#     75-84. The framing that a declared capability is a proposition the
#     compiler could check.
# =============================================================================
module ProvenanceAndCapability

using Dates, SHA

export ProvenanceStep, ProvenanceRecord, record_step, fingerprint, data_fingerprint,
       describe, Capability, provides, requires, check_compatibility, CapabilityReport

# -----------------------------------------------------------------------------
# Provenance
# -----------------------------------------------------------------------------
"""
    ProvenanceStep

One recorded stage of a portfolio's construction.

# Fields

  - `stage::Symbol`: Which stage, for example `:preprocessing`, `:prior`,
    `:optimisation`.
  - `component::String`: The concrete type or function used.
  - `params::Dict{Symbol, Any}`: The settings that were in force.
  - `timestamp::DateTime`.

# Notes

  - **`params` must hold the resolved values, not the defaults.** A record that
    says `alpha = default` is not a record. The point of provenance is to
    reconstruct, and a default can change between versions.
"""
struct ProvenanceStep
    stage::Symbol
    component::String
    params::Dict{Symbol, Any}
    timestamp::DateTime
end
function ProvenanceStep(stage::Symbol, component::AbstractString;
                        params::Dict{Symbol, <:Any} = Dict{Symbol, Any}())
    return ProvenanceStep(stage, String(component), Dict{Symbol, Any}(params), now())
end

"""
    ProvenanceRecord

The complete construction history of one result.

# Fields

  - `steps::Vector{ProvenanceStep}`: In execution order.
  - `data_hash::String`: A fingerprint of the input data.
  - `seeds::Dict{Symbol, Int}`: Every random seed used.
  - `versions::Dict{String, String}`: Package versions.
  - `created::DateTime`.

# Notes

  - **The three fields that actually matter are `data_hash`, `seeds` and
    `versions`.** Everything else is documentation. Without those three, a
    record describes a computation that cannot be repeated.
"""
struct ProvenanceRecord
    steps::Vector{ProvenanceStep}
    data_hash::String
    seeds::Dict{Symbol, Int}
    versions::Dict{String, String}
    created::DateTime
end
function ProvenanceRecord(; data_hash::AbstractString = "",
                          seeds::Dict{Symbol, <:Integer} = Dict{Symbol, Int}(),
                          versions::Dict{<:AbstractString, <:AbstractString} = Dict{String,
                                                                                    String}())
    return ProvenanceRecord(ProvenanceStep[], String(data_hash), Dict{Symbol, Int}(seeds),
                            Dict{String, String}(versions), now())
end

"""
    record_step(r::ProvenanceRecord, stage::Symbol, component::AbstractString;
                params...) -> ProvenanceRecord

Return a new record with one more step appended.

# Notes

  - Returns a new record rather than mutating, so a record can be shared
    between branches of an experiment without one branch corrupting another.
    That matches the library's immutability rule.
"""
function record_step(r::ProvenanceRecord, stage::Symbol, component::AbstractString;
                     params...)
    step = ProvenanceStep(stage, component; params = Dict{Symbol, Any}(pairs(params)))
    return ProvenanceRecord(vcat(r.steps, step), r.data_hash, r.seeds, r.versions,
                            r.created)
end

"""
    data_fingerprint(X::AbstractMatrix; digits::Integer = 10) -> String

Return a stable hash of a data matrix.

# Arguments

  - `X`: The data.
  - `digits`: Number of significant digits retained before hashing.

# Returns

  - The first sixteen hexadecimal characters of a SHA-256 digest.

# Details

Values are rounded to `digits` significant figures before hashing, so that a
difference below the precision anyone cares about does not change the
fingerprint.

# Notes

  - **Rounding is a trade-off, not a nicety.** Without it the fingerprint
    changes when the same data is loaded through a different reader, because
    the last bit differs. With it, two genuinely different data sets that agree
    to ten digits collide. Ten digits is far beyond any real return series.
  - The shape is hashed as well as the values, so a transposed matrix does not
    collide with the original.
"""
function data_fingerprint(X::AbstractMatrix; digits::Integer = 10)
    ctx = SHA.SHA256_CTX()
    SHA.update!(ctx, Vector{UInt8}(string(size(X))))
    for v in X
        SHA.update!(ctx, Vector{UInt8}(string(round(float(v); sigdigits = digits))))
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end

"""
    fingerprint(r::ProvenanceRecord) -> String

Return a stable hash of the whole record: data, seeds, versions and every step
with its parameters.

# Notes

  - **Two runs that produce the same fingerprint must produce the same
    weights**, and two that differ anywhere must differ here. That is the
    contract, and it is what makes a fingerprint usable as an experiment
    identifier. The driver checks both directions.
  - The timestamp is deliberately **excluded**. A record is identified by what
    it did, not by when.
"""
function fingerprint(r::ProvenanceRecord)
    ctx = SHA.SHA256_CTX()
    SHA.update!(ctx, Vector{UInt8}(r.data_hash))
    for k in sort(collect(keys(r.seeds)))
        SHA.update!(ctx, Vector{UInt8}("$(k)=$(r.seeds[k]);"))
    end
    for k in sort(collect(keys(r.versions)))
        SHA.update!(ctx, Vector{UInt8}("$(k)=$(r.versions[k]);"))
    end
    for s in r.steps
        SHA.update!(ctx, Vector{UInt8}("$(s.stage)|$(s.component)|"))
        for k in sort(collect(keys(s.params)))
            SHA.update!(ctx, Vector{UInt8}("$(k)=$(s.params[k]);"))
        end
    end
    return bytes2hex(SHA.digest!(ctx))[1:16]
end

"""
    describe(r::ProvenanceRecord) -> String

Return a human-readable rendering of a record.
"""
function describe(r::ProvenanceRecord)
    io = IOBuffer()
    println(io, "Provenance ", fingerprint(r))
    println(io, "  data      ", r.data_hash)
    println(io, "  seeds     ",
            if isempty(r.seeds)
                "(none)"
            else
                join(["$k=$v" for (k, v) in sort(collect(r.seeds); by = first)], ", ")
            end)
    for (i, s) in enumerate(r.steps)
        ps = if isempty(s.params)
            ""
        else
            " (" *
            join(["$k=$v" for (k, v) in sort(collect(s.params); by = first)], ", ") *
            ")"
        end
        println(io, "  ", lpad(i, 2), ". ", rpad(String(s.stage), 16), s.component, ps)
    end
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# Capability checking
# -----------------------------------------------------------------------------
"""
    Capability

The set of named quantities a component produces or consumes.

The vocabulary matches the library's own: `:mu`, `:sigma`, `:sk` (coskewness),
`:kt` (cokurtosis), `:X` (a returns matrix), `:F` (factor returns), `:Z` (a
feature matrix).
"""
const Capability = Symbol

"""
    provides(component) -> Set{Capability}

Return the capabilities a component produces.

# Notes

  - **This must be one method per concrete type, declared by its author**, not
    a walk over the fields. The library's `CONTEXT.md` reaches the same
    conclusion for `deferred_slots`, and for the same reason: the presence of a
    field does not imply the semantics. A `SimpleVariance` inside a `Skewness`
    is a legitimate component, not a promise about what the outer type
    produces.
  - The fallback below returns an empty set, so an undeclared component is
    reported as providing nothing rather than silently passing a check.
"""
provides(::Any) = Set{Capability}()

"""
    requires(component) -> Set{Capability}

Return the capabilities a component consumes. Fallback is the empty set.
"""
requires(::Any) = Set{Capability}()

"""
    CapabilityReport

The result of a compatibility check.

# Fields

  - `ok::Bool`.
  - `available::Set{Capability}`: Everything produced upstream.
  - `missing_caps::Vector{Pair{String, Set{Capability}}}`: For each component
    that cannot run, the capabilities it needs and did not get.
  - `messages::Vector{String}`: One sentence per problem, ready to show.
"""
struct CapabilityReport
    ok::Bool
    available::Set{Capability}
    missing_caps::Vector{Pair{String, Set{Capability}}}
    messages::Vector{String}
end

"""
    check_compatibility(producers::AbstractVector, consumers::AbstractVector;
                        labels = nothing) -> CapabilityReport

Check that every consumer's requirements are met by the producers.

# Arguments

  - `producers`: Components whose [`provides`](@ref) sets are unioned. The
    prior, the moment estimators, the data carrier.
  - `consumers`: Components whose [`requires`](@ref) sets must be satisfied.
    The risk measures, the objective, the constraints.
  - `labels`: Optional names for the consumers, used in the messages.

# Returns

  - A [`CapabilityReport`](@ref).

# Notes

  - **The value is the message, not the boolean.** "`Skewness` needs `:sk`,
    which no configured prior provides; use `HighOrderPrior`" is a sentence a
    caller can act on. A `MethodError` from inside a JuMP builder is not.
  - The check is **necessary and not sufficient**. It catches a missing
    quantity. It cannot catch a quantity of the wrong shape, or one computed on
    a different universe. Those need the dimension assertions the library
    already has.
  - Ordering is ignored: every producer is assumed to run before every
    consumer. A pipeline where that is false needs the check run per stage,
    which the library's `Data Slot` invalidation rule already knows how to
    sequence.
"""
function check_compatibility(producers::AbstractVector, consumers::AbstractVector;
                             labels::Union{Nothing, AbstractVector} = nothing)
    available = Set{Capability}()
    for p in producers
        union!(available, provides(p))
    end
    missing_caps = Pair{String, Set{Capability}}[]
    messages = String[]
    for (i, c) in enumerate(consumers)
        need = requires(c)
        gap = setdiff(need, available)
        if isempty(gap)
            continue
        end
        name = isnothing(labels) ? string(typeof(c)) : String(labels[i])
        push!(missing_caps, name => gap)
        push!(messages,
              "$(name) requires $(join(sort(collect(gap)), ", ")), which nothing upstream provides. Available: $(isempty(available) ? "(nothing)" : join(sort(collect(available)), ", ")).")
    end
    return CapabilityReport(isempty(missing_caps), available, missing_caps, messages)
end

end # module ProvenanceAndCapability
