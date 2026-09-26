"""
    SHARED_STATE

The names of the Model State entries that a nested risk build shares with its enclosing build, under the bare key.

It is the complement of Per-Build Risk State. A name belongs here if and only if its entry is not a function of the weights under optimisation and is not a presence flag of one build. The inner and the outer build then want the same object, and a prefix would break the sharing that it exists to protect. [`shared_get`](@ref), [`shared_has`](@ref) and [`shared_set!`](@ref) check each name against this set, so the classification holds at run time, in addition to the seam-lock test.

The comments in the source group the names and give the reason for each group. A new name here states that a nested build can safely read the copy of the enclosing build. Check that statement before you add a name.

# Related

  - [`assert_shared_state`](@ref)
  - [`shared_get`](@ref)
  - [`state_key`](@ref)
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
                                  # holds the per period terms and `:one_time_fees` the
                                  # one-off terms, and `:fee_fa` names the clock the second
                                  # falls on. A nested build charges the same fee as its
                                  # parent, so all three are shared rather than prefixed.
                                  :one_time_fees, :fee_fa, :decomposition_contract,
                                  :mip_indicators, :ss,
                                  # Weight shaping: outer level only. A nested build shifts
                                  # the weights through its own prefixed `:w`; it does not
                                  # reshape the long/short parts. `:w_gross_ub` bounds the
                                  # gross exposure of the head's weights, for the big-M
                                  # constant of the quantile programmes.
                                  :lw, :sw, :wp, :wn, :w1, :w2, :w_obj, :wip, :w_gross_ub,
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

Check that `name` is on [`SHARED_STATE`](@ref), the list of the Model State entries that a nested build reads under the bare key.

[`shared_get`](@ref), [`shared_has`](@ref) and [`shared_set!`](@ref) call it first. A read of an entry of one build without its prefix then fails at the call, and does not read the copy of the enclosing build in silence. That silent read is the defect that once broke `IndependentVariableTracking`.

# Validation

  - `name in SHARED_STATE`. Otherwise an `ArgumentError` names the two verbs for Per-Build Risk State, [`state_get`](@ref) and [`state_build!`](@ref).

# Returns

  - `nothing`.
"""
function assert_shared_state(name::Symbol)
    @argcheck(name in SHARED_STATE,
              ArgumentError("model[:$name] is not a sanctioned bare Model State entry. If it is per-build risk state (a function of the weights, or a build-scoped presence flag) reach it with a prefix via state_get/state_build!; if it is genuinely shared across nested builds, add it to SHARED_STATE with the reason."))
    return nothing
end
"""
    shared_set!(model::JuMP.Model, name::Symbol, val)

Register `val` as the shared Model State entry `name`, under the bare key, and return `val`.

It is the verb without a prefix that matches [`state_set!`](@ref), for a name on [`SHARED_STATE`](@ref). Unlike [`state_set!`](@ref) it does not check that the key is free, so a second call replaces the entry.

# Validation

  - `name` is on [`SHARED_STATE`](@ref), checked by [`assert_shared_state`](@ref).

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

Return `true` when the shared Model State entry `name` is registered.

# Validation

  - `name` is on [`SHARED_STATE`](@ref), checked by [`assert_shared_state`](@ref).
"""
function shared_has(model::JuMP.Model, name::Symbol)
    assert_shared_state(name)
    return haskey(model, name)
end
"""
    shared_get(model::JuMP.Model, name::Symbol)

Return the shared Model State entry `name`.

It is the verb without a prefix that matches [`state_get`](@ref). Use a named accessor, such as [`get_w`](@ref), [`get_k`](@ref) or [`get_ret`](@ref), where one exists.

# Validation

  - `name` is on [`SHARED_STATE`](@ref), checked by [`assert_shared_state`](@ref).
  - The entry is registered. Otherwise an `ArgumentError` states that a reader ran before the builder of the entry.

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

Compose the Model State key of the entry `name` under `prefix`, and at the measure index `i` when it is given.

It is the one place that spells the two conventions of the namespace. A key separates entries on two axes:

  - `prefix` separates one build from another, so a nested risk build cannot collide with the build that encloses it.
  - `i` separates one measure from another inside one build, so two `ConditionalValueatRisk` measures in one vector get their own scratch entries.

The seam-lock test checks that no builder composes a key by hand. A builder reaches Model State through [`state_get`](@ref), [`state_has`](@ref), [`state_set!`](@ref) and [`state_build!`](@ref).

The key is the concatenation of the parts, with no delimiter, so two different sets of parts can give one key: `(:tr_dr_, 11)` and `(:tr_dr_1, 1)` both give `:tr_dr_11`. A delimiter would change every key that a caller reads at the top level, so the spelling stays, and [`assert_state_key_free`](@ref) refuses the collision when an entry is registered.

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

Check that the Model State key `key` is not registered, so that a registration cannot replace an entry.

[`state_key`](@ref) joins its parts with no delimiter, so a name that ends in a digit at a low index gives the same key as a shorter name at a higher index: `state_key(p, :tr_dr_, 11) == state_key(p, :tr_dr_1, 1)`. Without this check the second registration replaces the first. The model then holds one entry where the build expects two, and a constraint binds the wrong variable. That is a wrong answer and not a crash, so the registration raises.

A delimiter is not the fix. It changes every key that a caller reads at the top level: `state_key(Symbol(""), :ret_, 1)` is `:ret_1`, a key that callers read. It also still lets one builder replace the entry of another under a key that both spell correctly. The check refuses both cases. A second registration under one key has no valid use. To build an entry once, use [`state_build!`](@ref), which returns the existing entry, and to set a flag, use [`mark_state!`](@ref), which has no effect the second time.

# Validation

  - `!haskey(model, key)`. Otherwise an `ArgumentError` names the key and the two verbs that accept a repeat.

# Returns

  - `nothing`.

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

Register `val` under the Model State key of `name` and `prefix`, and return `val`.

A nested risk build, such as a risk tracking measure, passes a non-empty `prefix`, so the entries it makes (`:X`, `:net_X`, `:W`, `:dd` and others) do not collide with the entries of the outer model. The empty prefix gives the bare key.

The method with `i` registers the scratch of one measure, such as `:cvar_risk_` or `:z_cvar_`, at the measure index `i`, so two measures of one type in one build get their own entries. [`state_key`](@ref) composes both parts of the key.

# Algorithm

 1. Compose the key with [`state_key`](@ref).
 2. Check with [`assert_state_key_free`](@ref) that the key is free.
 3. Register `val` under the key.

# Validation

  - The composed key is free. Otherwise [`assert_state_key_free`](@ref) raises an `ArgumentError`. To reuse an entry, use [`state_build!`](@ref) or [`mark_state!`](@ref).

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

Return `true` when the Model State entry `name` is registered under `prefix`, and at the index `i` when it is given.

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

Return the Model State entry `name` under `prefix`, and at the index `i` when it is given.

Use a named accessor, such as [`get_w`](@ref) or [`get_net_X`](@ref), where one exists. Its error names the builder of the entry, where this error names only the key.

# Validation

  - The entry is registered. Otherwise an `ArgumentError` names the key and states that a reader ran before the builder of the entry.

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

Return the Model State entry `name` under `prefix`, and build it with `f()` the first time.

Every risk and constraint builder uses it to build an entry once. An entry that an earlier measure of the same build, or an outer build, already registered comes back unchanged. The builder `f` registers each companion entry that it makes with [`state_set!`](@ref).

This verb composes the key, not the caller, so a new Model State entry gets the prefix of its build with no more work.

# Algorithm

 1. Compose the key with [`state_key`](@ref).
 2. When the key is registered, return its entry.
 3. Otherwise call `f()`, giving `val`.
 4. Register `val` under the key and return it.

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
    mark_state!(model::JuMP.Model, prefix::Symbol, name::Symbol, i)

Record that this build has `name`, under `prefix` and at the index `i` when it is given.

The entry is a presence flag of one build. It holds `true` and nothing more, and a reader tests it with [`state_has`](@ref). A second call has no effect. The prefix keeps the flags of a nested build out of the enclosing build. This is the half of Per-Build Risk State that does not depend on the weights.

# Returns

  - `nothing`.

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

Compose the Model State prefix that a nested build passes down to its own builders.

It gives a prefix and not a key, so no entry of the nested build can take the key of an entry of the enclosing build. `tag` names the kind of nesting (`:tr_iv_`, `:tr_dv_`, `:tr_ir_`, `:tr_dr_`, `:gain_`). The method with `i` adds the measure index and a closing `_`, so `nested_prefix(:a_, :tr_dr_, 3)` is `:a_tr_dr_3_`. The index keeps a tracking measure inside a tracking measure free of collisions.

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

Compose the Model State measure index that the parts of a composite measure pass down.

It is the twin of [`nested_prefix`](@ref) on the other axis of the key. A composite measure that builds its parts in the same build, such as `GenericValueatRiskRange` over its `loss` and `gain` sides, separates the parts by index and not by prefix. The parts share the infrastructure entries of the build, and neither may build them a second time. `tag` names the part (`:loss_`, `:gain_`), and the composition nests: `nested_index(:gain_, nested_index(:loss_, 2))` is `:gain_loss_2`, so a range inside a range stays free of collisions.

The result is an index and not a key.

# Related

  - [`nested_prefix`](@ref)
  - [`state_key`](@ref)
"""
function nested_index(tag::Symbol, i)
    return Symbol(tag, i)
end
