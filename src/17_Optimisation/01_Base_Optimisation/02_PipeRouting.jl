"""
    pipe_route(x, ::Val{target}, v)

Absorb a [Routing Target](@ref PIPELINE_ROUTING_TARGETS)'s value into an optimiser, returning the rebuilt optimiser.

This is the optimiser-owned half of the [`Pipeline`](@ref) seam: [`inject_context`](@ref) fans a [`PipelineContext`](@ref) slot out into routing targets and delivers each one here, knowing nothing about where it lands.

Targets are named after the field they land in — `:pe`, `:cle`, `:wb`, `:lcse`, `:ple` — because those names are this package's shared vocabulary (see `field_dict`) rather than any one optimiser's private layout. The default method therefore *is* the routing rule: a target lands in the like-named field of any optimiser that has one. Nothing is declared per type, so nothing can drift.

Two targets are exceptions, because they carry validation policy and name no plain field: `:mu_ucs` (requires an [`ArithmeticReturn`](@ref), lands in `ret.ucs`) and `:sigma_ucs` (lands in the [`UncertaintySetVariance`](@ref) measures of `r`, see [`@pipe_route_sigma_ucs`](@ref)).

Optimisers holding their configuration in a field rather than carrying the target fields themselves declare [`@pipe_delegates`](@ref).

The lookup is `hasfield` rather than `hasproperty`, because routing rebuilds the object through the field: a name reachable only as a forwarded property could be read but not set.

A target with no home falls through to [`unroutable_target`](@ref), which ignores the [optional](@ref PIPELINE_OPTIONAL_TARGETS) ones and throws for the rest.

Internal machinery — not part of the user-facing API.

# Arguments

  - `x`: The optimiser or optimiser configuration.
  - `::Val{target}`: One of [`PIPELINE_ROUTING_TARGETS`](@ref).
  - `v`: The computed result to absorb.

# Returns

  - `x′`: The rebuilt optimiser.

# Related

  - [`pipe_accepts`](@ref)
  - [`@pipe_delegates`](@ref)
  - [`inject_context`](@ref)
"""
function pipe_route(x, ::Val{target}, v) where {target}
    return if hasfield(typeof(x), target)
        Accessors.set(x, Accessors.PropertyLens{target}(), v)
    else
        unroutable_target(x, Val(target), v)
    end
end
"""
    pipe_accepts(x, ::Val{target}) -> Bool

Whether a [Routing Target](@ref PIPELINE_ROUTING_TARGETS) has a home in `x`.

For the field-named targets this is exactly `hasfield` — the same fact [`pipe_route`](@ref) dispatches on, so acceptance and routing cannot disagree. The `:mu_ucs`/`:sigma_ucs` exceptions and the [`@pipe_delegates`](@ref) forwarders override it.

Internal machinery — not part of the user-facing API.

# Related

  - [`pipe_route`](@ref)
  - [`unroutable_target`](@ref)
"""
function pipe_accepts(x, ::Val{target})::Bool where {target}
    return hasfield(typeof(x), target)
end
"""
    unroutable_target(x, ::Val{target}, v)

Handle a [Routing Target](@ref PIPELINE_ROUTING_TARGETS) that `x` has no home for.

Declared here so [`pipe_route`](@ref) can call it; defined beside [`PIPELINE_OPTIONAL_TARGETS`](@ref), which is where the ignore-versus-throw policy belongs.

Internal machinery — not part of the user-facing API.

# Related

  - [`pipe_route`](@ref)
  - [`PIPELINE_OPTIONAL_TARGETS`](@ref)
"""
function unroutable_target end
"""
    pipe_config_field(x) -> Union{Nothing, Symbol}

The field holding `x`'s optimiser configuration, or `nothing` if it has none.

Declared by [`@pipe_delegates`](@ref) rather than probed, so an estimator whose `opt` field holds an inner *estimator* — [`SubsetResampling`](@ref) — is never mistaken for one holding a [`JuMPOptimiser`](@ref) configuration.

Internal machinery — not part of the user-facing API.

# Related

  - [`@pipe_delegates`](@ref)
  - [`pipe_route`](@ref)
"""
pipe_config_field(::Any) = nothing
"""
    @pipe_delegates T field

Declare that optimiser type `T` forwards every [Routing Target](@ref PIPELINE_ROUTING_TARGETS) to the configuration held in `field`.

Emits [`pipe_config_field`](@ref) plus forwarding [`pipe_route`](@ref) and [`pipe_accepts`](@ref) methods. Targets the configuration has no home for reach its own [`unroutable_target`](@ref), so the resulting error names the configuration — matching the pre-inversion messages.

A type that absorbs a target *itself* rather than through its configuration declares that target on the concrete type (see [`@pipe_route_sigma_ucs`](@ref)), which out-specialises this forwarder.

# Examples

```julia
@pipe_delegates MeanRisk opt
```

# Related

  - [`pipe_route`](@ref)
  - [`pipe_config_field`](@ref)
"""
macro pipe_delegates(T, field)
    f = QuoteNode(isa(field, QuoteNode) ? field.value : field)
    #! The block is escaped whole: hygiene would otherwise rename the three generics being
    #! extended into gensyms, silently defining the methods on throwaway functions.
    return esc(quote
                   pipe_config_field(::$T) = $f
                   function pipe_route(x::$T, t::Val, v)
                       return Accessors.set(x, Accessors.PropertyLens{$f}(),
                                            pipe_route(getfield(x, $f), t, v))
                   end
                   function pipe_accepts(x::$T, t::Val)::Bool
                       return pipe_accepts(getfield(x, $f), t)
                   end
               end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Route a covariance uncertainty set into the [`UncertaintySetVariance`](@ref) risk measure(s) held in an optimiser's `r` field.

Each [`UncertaintySetVariance`](@ref) found — directly or inside a vector — has its `ucs` replaced with `sig`. An `r` field carrying no such measure is an error rather than a silent no-op: a computed uncertainty set that reaches no risk measure would be dropped.

Internal machinery — not part of the user-facing API.

# Arguments

  - `x`: The optimiser, which must carry an `r` field.
  - `sig`: The covariance uncertainty set result.

# Returns

  - `x′`: The rebuilt optimiser.

# Related

  - [`@pipe_route_sigma_ucs`](@ref)
  - [`pipe_route`](@ref)
"""
function route_sigma_ucs(x, sig::AbstractUncertaintySetResult)
    replace_usv(y) =
        if isa(y, UncertaintySetVariance)
            Accessors.set(y, Accessors.PropertyLens{:ucs}(), sig)
        else
            y
        end
    r = x.r
    newr = isa(r, AbstractVector) ? identity.([replace_usv(y) for y in r]) : replace_usv(r)
    found = if isa(newr, AbstractVector)
        any(y -> isa(y, UncertaintySetVariance), newr)
    else
        isa(newr, UncertaintySetVariance)
    end
    @argcheck(found,
              ArgumentError("cannot route a covariance uncertainty set into a $(Base.typename(typeof(x)).wrapper): no UncertaintySetVariance risk measure in its r field"))
    return Accessors.set(x, Accessors.PropertyLens{:r}(), newr)
end
"""
    @pipe_route_sigma_ucs T

Declare that optimiser type `T` absorbs the `:sigma_ucs` [Routing Target](@ref PIPELINE_ROUTING_TARGETS) into its own `r` field via [`route_sigma_ucs`](@ref).

Declared per concrete type rather than on a supertype: the covariance uncertainty set lands in the *estimator's* risk measures while every other target is forwarded to its configuration, so this method must out-specialise the [`@pipe_delegates`](@ref) forwarder on the same type. It is opt-in because carrying a configuration does not imply carrying risk measures — [`RelaxedRiskBudgeting`](@ref) has no `r` field.

# Related

  - [`route_sigma_ucs`](@ref)
  - [`@pipe_delegates`](@ref)
"""
macro pipe_route_sigma_ucs(T)
    #! Escaped whole, for the same reason as `@pipe_delegates`.
    return esc(quote
                   function pipe_route(x::$T, ::Val{:sigma_ucs}, v)
                       return route_sigma_ucs(x, v)
                   end
                   pipe_accepts(::$T, ::Val{:sigma_ucs})::Bool = true
               end)
end
"""
    @pipe_route_rkb T

Declare that optimiser type `T` absorbs the `:rkb` [Routing Target](@ref PIPELINE_ROUTING_TARGETS) into the `rkb` field of its risk-budgeting algorithm.

`:rkb` is the one target named after a field an optimiser does not carry directly. A risk budget belongs to the *algorithm* — `AssetRiskBudgeting` budgets assets, `FactorRiskBudgeting` budgets factors — so it lands one level down, at `rba.rkb`, and the derived `hasfield` rule cannot reach it.

Acceptance is not a constant: it asks the algorithm the optimiser is actually carrying. A [`TimeDependent`](@ref) schedule in `rba` has no `rkb` to write, so such an optimiser declines the target and a pipeline computing a budget for it is refused at construction rather than failing in the fold loop.

Declared per concrete type, for the same reason as [`@pipe_route_sigma_ucs`](@ref): it must out-specialise the [`@pipe_delegates`](@ref) forwarder on the same type.

# Related

  - [`@pipe_delegates`](@ref)
  - [`pipe_route`](@ref)
"""
macro pipe_route_rkb(T)
    #! Escaped whole, for the same reason as `@pipe_delegates`.
    return esc(quote
                   function pipe_route(x::$T, ::Val{:rkb}, v)
                       @argcheck(hasfield(typeof(x.rba), :rkb),
                                 ArgumentError("cannot route a risk budget into a $(Base.typename(typeof(x)).wrapper): its rba field holds a $(Base.typename(typeof(x.rba)).wrapper), which has no rkb field to receive it"))
                       rba = Accessors.set(x.rba, Accessors.PropertyLens{:rkb}(), v)
                       return Accessors.set(x, Accessors.PropertyLens{:rba}(), rba)
                   end
                   function pipe_accepts(x::$T, ::Val{:rkb})::Bool
                       return hasfield(typeof(x.rba), :rkb)
                   end
               end)
end
