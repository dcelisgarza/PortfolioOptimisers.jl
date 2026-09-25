"""
    SHARED_STATE

The Model State entries deliberately shared **bare** across a nested risk build.

The complement of Per-Build Risk State: an entry belongs here iff it is *not* a function of
the weights being optimised and *not* a build-scoped presence flag, so the inner and outer
builds want the same object and prefixing it would break sharing rather than protect it.
[`shared_get`](@ref) and friends validate against this set, so the classification
is enforced at run time rather than only by the seam-lock test.

Each grouping records *why* those entries are shared. Adding a name here is a claim that a
nested build may safely see the enclosing build's copy — check that claim before adding.
"""
const SHARED_STATE = Set{Symbol}([# Pure functions of the prior `pr`: identical in the inner
                                  # and outer build (Cholesky/eigendecompositions, factor
                                  # risk contribution lift).
                                  :G, :GV, :Gkt, :vals_Akt, :vecs_Akt, :frc_W, :frc_M,
                                  :frc_M_PSD,
                                  # Model-wide singletons established once, before the risk
                                  # spine runs. `:T` is the observation count of the fit,
                                  # registered beside `:sc` and `:so` by the head itself, so
                                  # every builder may rely on it whatever the model carries.
                                  :sc, :so, :T, :k, :w, :ret, :risk, :fees, :unit_budget,
                                  # The fee spine, established once by the fee builder and
                                  # read by the return and the net-series builders. `:fees`
                                  # holds the per period terms and `:one_time_fees` the two
                                  # fixed ones, and `:fee_fa` names the clock the second
                                  # falls on. A nested build charges the same fee as its
                                  # parent, so all three are shared rather than prefixed.
                                  :one_time_fees, :fee_fa, :decomposition_contract,
                                  :mip_indicators, :ss,
                                  # Weight shaping: outer level only. A nested build shifts
                                  # the weights through its own prefixed `:w`; it does not
                                  # reshape the long/short parts. `:w_gross_ub` bounds the
                                  # gross exposure of the head's weights, for the big-M
                                  # constant of the quantile programmes.
                                  :lw, :sw, :wp, :wn, :w1, :w_obj, :wip, :w_gross_ub,
                                  # Returns and objective plumbing, outer level only. The
                                  # robust-cone scratch a return term raises (`bucs_w_i`,
                                  # `t_eucs_gw_i`) is index-suffixed and read by index, so
                                  # it is not on this list; `sr_risk` stays because the
                                  # ratio constraint is hoisted and registers exactly one.
                                  :ohf, :op, :cost_bgt_expr, :sr_risk,
                                  # Risk and return accumulation and frontier bookkeeping:
                                  # collected by the terminal scalarise seams (ADR 0024),
                                  # which run once at the outer level. A nested build
                                  # returns its expression to its caller instead of pushing
                                  # here.
                                  :risk_vec, :risk_frontier, :ret_vec, :ret_frontier,
                                  # Per-optimiser scratch on the outer model.
                                  :noc_rk, :noc_rt, :psi,
                                  # The one deliberate write-prefixed / read-bare entry
                                  # (ADR 0005). A variance that is a positive term of the
                                  # objective's risk marks the namespace that owns its
                                  # weights, `weights_prefix`: a build on shifted weights
                                  # marks its own prefix, so its presence does not leak
                                  # outward, and a build on the head's own weights marks
                                  # the bare entry. `risk_minimised` records that the
                                  # objective minimises the risk. The readers are the
                                  # phylogeny builders, which omit the `p·tr(W)` penalty
                                  # only when both are present.
                                  :variance_flag, :risk_minimised])
"""
    assert_shared_state(name::Symbol)

Assert `name` is a sanctioned bare Model State entry.

Guards the [`shared_get`](@ref) family so that reaching for a per-build entry without a
prefix fails loudly at the call site, rather than silently aliasing the enclosing build's
copy — the regression class that broke `IndependentVariableTracking`.
"""
function assert_shared_state(name::Symbol)
    @argcheck(name in SHARED_STATE,
              ArgumentError("model[:$name] is not a sanctioned bare Model State entry. If it is per-build risk state (a function of the weights, or a build-scoped presence flag) reach it with a prefix via state_get/state_build!; if it is genuinely shared across nested builds, add it to SHARED_STATE with the reason."))
    return nothing
end
"""
    shared_set!(model::JuMP.Model, name::Symbol, val)

Register `val` as the sanctioned bare Model State entry `name` and return it.

The unprefixed counterpart of [`state_set!`](@ref), for entries on [`SHARED_STATE`](@ref).

# Related

  - [`shared_get`](@ref)
"""
function shared_set!(model::JuMP.Model, name::Symbol, val)
    assert_shared_state(name)
    model[name] = val
    return val
end
"""
    shared_has(model::JuMP.Model, name::Symbol)

Return `true` if the sanctioned bare Model State entry `name` is registered.
"""
function shared_has(model::JuMP.Model, name::Symbol)
    assert_shared_state(name)
    return haskey(model, name)
end
"""
    shared_get(model::JuMP.Model, name::Symbol)

Return the sanctioned bare Model State entry `name`, asserting it has been registered.

The unprefixed counterpart of [`state_get`](@ref). Prefer a named accessor
([`get_w`](@ref), [`get_k`](@ref), [`get_ret`](@ref), …) where one exists.

# Related

  - [`shared_has`](@ref)
  - [`SHARED_STATE`](@ref)
"""
function shared_get(model::JuMP.Model, name::Symbol)
    assert_shared_state(name)
    @argcheck(haskey(model, name),
              ArgumentError("model[:$name] has not been registered; it is being read before the builder that produces it has run"))
    return model[name]
end
"""
    state_key(prefix::Symbol, name::Symbol)
    state_key(prefix::Symbol, name::Symbol, i)

Resolve the Model State key for entry `name` under `prefix`, optionally at measure index `i`.

Internal to the Model State interface: the single place the two namespacing conventions are
spelled. A Model State key is disambiguated on two axes, and both are resolved here:

  - `prefix` separates one *build* from another, so a nested risk build cannot collide with
    the build that encloses it.
  - `i` separates one *measure instance* from another inside a single build, so two
    `ConditionalValueatRisk` measures in the same vector get their own scratch entries.

Keeping both here is what lets the seam-lock test assert that no emitter builds a key by
hand — emitters reach Model State through [`state_get`](@ref), [`state_has`](@ref),
[`state_set!`](@ref) and [`state_build!`](@ref).

Neither axis carries a delimiter, so composition is **not injective**: `(:tr_dr_, 11)` and
`(:tr_dr_1, 1)` both give `:tr_dr_11`. The spelling is kept — a delimiter would move every
top-level key a caller reads — and the collision is caught where it does harm, by
[`assert_state_key_free`](@ref) at registration.

# Related

  - [`state_build!`](@ref)
  - [`nested_prefix`](@ref)
  - [`assert_state_key_free`](@ref)
"""
function state_key(prefix::Symbol, name::Symbol)
    return Symbol(prefix, name)
end
function state_key(prefix::Symbol, name::Symbol, i)
    return Symbol(prefix, name, i)
end
"""
    assert_state_key_free(model::JuMP.Model, key::Symbol)

Assert Model State key `key` is not registered yet, so a write cannot replace an entry.

Neither axis of [`state_key`](@ref) is separated by a delimiter, so key composition is
**not injective**: a name that ends in a digit and a low index compose the same `Symbol` as
a shorter name and a higher index — `state_key(p, :tr_dr_, 11) == state_key(p, :tr_dr_1, 1)`.
Without this guard the second write wins, the model carries one entry where the build
expected two, and a constraint binds the wrong variable. That is a wrong answer, not a
crash, so the registration verb fails closed instead.

A delimiter was rejected as the fix: it would move every top-level key spelling
(`state_key(Symbol(""), :ret_, 1)` is `:ret_1`, a key callers read), and it would still let
one emitter overwrite another's entry under a key both spell correctly. The guard closes
both. Re-registration under one key has no legitimate reading either: the build-once case
is [`state_build!`](@ref), which returns the existing entry untouched, and the flag case is
[`mark_state!`](@ref), which is idempotent.

# Returns

  - `nothing`.

# Throws

  - `ArgumentError` if `key` is already registered. The message names the key and the two
    verbs that do accept a repeat.

# Related

  - [`state_key`](@ref)
  - [`state_set!`](@ref)
  - [`state_build!`](@ref)
"""
function assert_state_key_free(model::JuMP.Model, key::Symbol)
    @argcheck(!haskey(model, key),
              ArgumentError("model[$key] is already registered, so this registration would replace it and lose the earlier entry. Model State keys compose by concatenation and are not injective — state_key(prefix, :tr_dr_, 11) and state_key(prefix, :tr_dr_1, 1) are both :tr_dr_11 — so either two entries composed the same key and one of them must be renamed, or the same entry is registered twice. Use state_build! to build an entry once and reuse it, or mark_state! for an idempotent presence flag."))
    return nothing
end
"""
    state_set!(model::JuMP.Model, prefix::Symbol, name::Symbol, val)
    state_set!(model::JuMP.Model, prefix::Symbol, name::Symbol, i, val)

Register `val` in the model under the prefixed Model State key and return it.

A nested risk build (e.g. risk tracking) passes a non-empty `prefix` so the shared
infrastructure entries it creates (`:X`, `:net_X`, `:W`, `:dd`, …) do not collide with the
outer model's; the default empty prefix reproduces the bare key.

The indexed method registers per-measure scratch (`:cvar_risk_`, `:z_cvar_`, …) at measure
index `i`, so two instances of the same measure in one build get their own entries. Both
disambiguators are resolved by [`state_key`](@ref).

Registration is *fresh*: the composed key must be free, because key composition is not
injective and a replaced entry is a wrong answer rather than an error
([`assert_state_key_free`](@ref)). Reuse is the other two verbs' job.

# Throws

  - `ArgumentError` if the composed key is already registered.

# Related

  - [`state_get`](@ref)
  - [`state_build!`](@ref)
  - [`assert_state_key_free`](@ref)
"""
function state_set!(model::JuMP.Model, prefix::Symbol, name::Symbol, val)
    key = state_key(prefix, name)
    assert_state_key_free(model, key)
    model[key] = val
    return val
end
function state_set!(model::JuMP.Model, prefix::Symbol, name::Symbol, i, val)
    key = state_key(prefix, name, i)
    assert_state_key_free(model, key)
    model[key] = val
    return val
end
"""
    state_has(model::JuMP.Model, prefix::Symbol, name::Symbol)
    state_has(model::JuMP.Model, prefix::Symbol, name::Symbol, i)

Return `true` if Model State entry `name` is registered under `prefix`, at index `i` if given.

# Related

  - [`state_get`](@ref)
"""
function state_has(model::JuMP.Model, prefix::Symbol, name::Symbol)
    return haskey(model, state_key(prefix, name))
end
function state_has(model::JuMP.Model, prefix::Symbol, name::Symbol, i)
    return haskey(model, state_key(prefix, name, i))
end
"""
    state_get(model::JuMP.Model, prefix::Symbol, name::Symbol)
    state_get(model::JuMP.Model, prefix::Symbol, name::Symbol, i)

Return Model State entry `name` under `prefix`, asserting it has been registered.

Prefer a named accessor ([`get_X`](@ref), [`get_net_X`](@ref), [`get_dd`](@ref), …) where
one exists: those name the builder that produces the entry, so an out-of-order read reports
which builder to call instead of a generic missing-entry error.

The indexed method reads per-measure scratch registered at measure index `i`.

# Related

  - [`state_has`](@ref)
  - [`state_build!`](@ref)
"""
function state_get(model::JuMP.Model, prefix::Symbol, name::Symbol)
    key = state_key(prefix, name)
    @argcheck(haskey(model, key),
              ArgumentError("model[$key] has not been registered; it is being read before the builder that produces it has run"))
    return model[key]
end
function state_get(model::JuMP.Model, prefix::Symbol, name::Symbol, i)
    key = state_key(prefix, name, i)
    @argcheck(haskey(model, key),
              ArgumentError("model[$key] has not been registered; it is being read before the builder that produces it has run"))
    return model[key]
end
"""
    state_build!(f, model::JuMP.Model, prefix::Symbol, name::Symbol)
    state_build!(f, model::JuMP.Model, prefix::Symbol, name::Symbol, i)

Return Model State entry `name` under `prefix`, building it with `f()` exactly once.

The memoise-on-prefixed-key idiom shared by every risk and constraint emitter: if the entry
is already registered — an earlier measure in the same build produced it, or an outer build
already did — it is returned untouched; otherwise `f()` runs and its value is registered
under the prefixed key. Companion entries created inside `f` register with
[`state_set!`](@ref).

Because the key is resolved here rather than at the call site, a Model State entry added in
future participates in the prefix discipline with no further work. That is what closes a
residual hole an earlier, more permissive design left open.

# Related

  - [`state_set!`](@ref)
  - [`state_get`](@ref)
"""
function state_build!(f, model::JuMP.Model, prefix::Symbol, name::Symbol)
    key = state_key(prefix, name)
    if haskey(model, key)
        return model[key]
    end
    val = f()
    model[key] = val
    return val
end
function state_build!(f, model::JuMP.Model, prefix::Symbol, name::Symbol, i)
    key = state_key(prefix, name, i)
    if haskey(model, key)
        return model[key]
    end
    val = f()
    model[key] = val
    return val
end
"""
    mark_state!(model::JuMP.Model, prefix::Symbol, name::Symbol)

Record that this build has `name` present, idempotently.

A build-scoped presence flag: `name` carries no value beyond its own existence, and readers
test it with [`state_has`](@ref) rather than reading it. Marking under `prefix` is what keeps
a nested build's flags out of the enclosing build — the second half of Per-Build Risk State,
the half that is not weight-dependent.

# Related

  - [`state_has`](@ref)
  - [`state_build!`](@ref)
"""
function mark_state!(model::JuMP.Model, prefix::Symbol, name::Symbol)
    state_build!(() -> true, model, prefix, name)
    return nothing
end
function mark_state!(model::JuMP.Model, prefix::Symbol, name::Symbol, i)
    state_build!(() -> true, model, prefix, name, i)
    return nothing
end
"""
    nested_prefix(prefix::Symbol, tag::Symbol)
    nested_prefix(prefix::Symbol, tag::Symbol, i)

Compose the Model State namespace a nested build threads down its own spine.

Distinct from a Model State *key*: this produces a `prefix`, not an entry name, so a nested
build's entries cannot alias the enclosing build's. `tag` names the nesting kind (`:tr_iv_`,
`:tr_dv_`, `:tr_ir_`, `:tr_dr_`, `:gain_`) and the optional `i` disambiguates the measure
index, which is what makes tracking-nested-in-tracking collision-free.

# Related

  - [`nested_index`](@ref)
  - [`state_build!`](@ref)
"""
function nested_prefix(prefix::Symbol, tag::Symbol)
    return Symbol(prefix, tag)
end
function nested_prefix(prefix::Symbol, tag::Symbol, i)
    return Symbol(prefix, tag, i, :_)
end
"""
    nested_index(tag::Symbol, i)

Compose the Model State measure index a sub-measure build threads down.

The twin of [`nested_prefix`](@ref) on the other disambiguating axis. A composite measure
that builds its parts *in the same build* — `GenericValueatRiskRange` over its `loss` and
`gain` sides — separates the parts by index rather than by namespace, because they share
the build's infrastructure entries and must not each rebuild them. `tag` names the part
(`:loss_`, `:gain_`), and the composition nests, so a range inside a range stays
collision-free.

Distinct from a Model State *key*: this produces an index, not an entry name.

# Related

  - [`nested_prefix`](@ref)
  - [`state_key`](@ref)
"""
function nested_index(tag::Symbol, i)
    return Symbol(tag, i)
end
