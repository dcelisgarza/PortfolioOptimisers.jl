"""
    pipe_route(x, ::Val{target}, v)

Puts the value of a [Routing Target](@ref PIPELINE_ROUTING_TARGETS) into an optimiser, and returns the rebuilt optimiser.

This is the optimiser half of the routing of a [`Pipeline`](@ref) slot into an optimiser. [`inject_context`](@ref) divides a slot of the [`PipelineContext`](@ref) into Routing Targets, and gives each of them to this function. It does not know which field receives the value.

A target has the name of the field that receives it, for example `:pe`, `:cle`, `:wb`, `:lcse` or `:ple`. These names are the field names that the whole package uses, see `field_dict`, and not the private layout of one optimiser. So the fallback method is the routing rule: a target goes into the field of the same name in each optimiser that has one. No type declares its targets, so no declaration can become wrong.

Two targets have methods of their own, because they check their value and name no plain field. `:mu_ucs` needs an [`ArithmeticReturn`](@ref), and goes into `ret.ucs`. `:sigma_ucs` goes into the [`UncertaintySetVariance`](@ref) measures of `r`, see [`@pipe_route_sigma_ucs`](@ref).

An optimiser that holds its configuration in a field declares [`@pipe_delegates`](@ref). A target with no field in `x` goes to [`unroutable_target`](@ref), which ignores the [optional](@ref PIPELINE_OPTIONAL_TARGETS) targets and throws for the others.

# Algorithm

 1. When the type of `x` has a field named `target`, return `x` with `v` in that field.
 2. Else return `unroutable_target(x, Val(target), v)`.

The test is `hasfield` and not `hasproperty`, because the rebuild writes the field. A name that only property forwarding reaches can be read but not written.

# Arguments

  - `x`: The optimiser or the optimiser configuration.
  - `::Val{target}`: One of [`PIPELINE_ROUTING_TARGETS`](@ref).
  - `v`: The value to put into `x`.

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

Returns `true` if `x` has a field for a [Routing Target](@ref PIPELINE_ROUTING_TARGETS).

For a target that has the name of a field, the answer is `hasfield`, which is the test of [`pipe_route`](@ref). So `pipe_accepts` returns `true` exactly for the targets that `pipe_route` puts into a field. The targets `:mu_ucs`, `:sigma_ucs` and `:rkb`, and the optimisers that declare [`@pipe_delegates`](@ref), have methods of their own.

# Related

  - [`pipe_route`](@ref)
  - [`unroutable_target`](@ref)
"""
function pipe_accepts(x, ::Val{target})::Bool where {target}
    return hasfield(typeof(x), target)
end
"""
    unroutable_target(x, ::Val{target}, v)

Handles a [Routing Target](@ref PIPELINE_ROUTING_TARGETS) for which `x` has no field.

The function is declared here, so that [`pipe_route`](@ref) can call it. Its method is beside [`PIPELINE_OPTIONAL_TARGETS`](@ref), which states which targets it ignores and which it refuses.

# Related

  - [`pipe_route`](@ref)
  - [`PIPELINE_OPTIONAL_TARGETS`](@ref)
"""
function unroutable_target end
"""
    pipe_config_field(x) -> Union{Nothing, Symbol}

Returns the name of the field that holds the optimiser configuration of `x`, or `nothing` when `x` has none.

[`@pipe_delegates`](@ref) declares the field, and no code guesses it from the field types. So an estimator whose field `opt` holds an inner estimator, such as [`SubsetResampling`](@ref), is not taken for an estimator whose `opt` holds a [`JuMPOptimiser`](@ref).

# Related

  - [`@pipe_delegates`](@ref)
  - [`pipe_route`](@ref)
"""
pipe_config_field(::Any) = nothing
"""
    @pipe_delegates T field

Declares that the optimiser type `T` gives every [Routing Target](@ref PIPELINE_ROUTING_TARGETS) to the configuration in `field`.

The macro makes a method of [`pipe_config_field`](@ref), and methods of [`pipe_route`](@ref) and [`pipe_accepts`](@ref) that call the same function on the configuration. A target for which the configuration has no field goes to the [`unroutable_target`](@ref) of the configuration, so the error names the configuration.

A type that puts a target into its own field, and not into its configuration, declares that target on the concrete type, see [`@pipe_route_sigma_ucs`](@ref). That method is more specific than the methods of this macro.

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
                   pipe_config_field(::$T)::Symbol = $f
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

Puts a covariance uncertainty set into the [`UncertaintySetVariance`](@ref) risk measures in the field `r` of an optimiser.

A field `r` that holds no such measure is an error. Else the computed uncertainty set reaches no risk measure, and no message tells the caller.

# Algorithm

 1. Read the field `r` of `x`.
 2. Replace the `ucs` of each [`UncertaintySetVariance`](@ref) with `sig`. When `r` is a vector, do this for each element, and keep the other elements.
 3. Check that the new `r` holds at least one `UncertaintySetVariance`.
 4. Return `x` with the new `r`.

# Arguments

  - `x`: The optimiser. It must have a field `r`.
  - `sig`: The covariance uncertainty set.

# Validation

  - The field `r` holds an [`UncertaintySetVariance`](@ref), or a vector with one, else an `ArgumentError` is thrown.

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

Declares that the optimiser type `T` puts the [Routing Target](@ref PIPELINE_ROUTING_TARGETS) `:sigma_ucs` into its own field `r`, with [`route_sigma_ucs`](@ref).

The covariance uncertainty set goes into the risk measures of the estimator, and every other target goes to its configuration. So this method must be more specific than the method of [`@pipe_delegates`](@ref) on the same type, and each concrete type declares it. A type declares it only when it has risk measures, because a configuration does not bring them. [`RelaxedRiskBudgeting`](@ref) has no field `r`.

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

Declares that the optimiser type `T` puts the [Routing Target](@ref PIPELINE_ROUTING_TARGETS) `:rkb` into the field `rkb` of its risk budgeting algorithm.

`:rkb` is the one target that names a field one level down. A risk budget belongs to the algorithm, because `AssetRiskBudgeting` budgets assets and `FactorRiskBudgeting` budgets factors. So the budget goes into `rba.rkb`, which the rule of `hasfield` does not reach.

The answer of `pipe_accepts` reads the algorithm that the optimiser holds. A [`TimeDependent`](@ref) in `rba` has no field `rkb`, so such an optimiser refuses the target. Then the constructor of a pipeline that computes a budget for it refuses the pipeline, before the fold loop runs.

Each concrete type declares it, for the reason that [`@pipe_route_sigma_ucs`](@ref) states: it must be more specific than the method of [`@pipe_delegates`](@ref) on the same type.

# Validation

  - The field `rba` of the optimiser has a field `rkb`, else `pipe_route` throws an `ArgumentError` that names the type of `rba`.

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
