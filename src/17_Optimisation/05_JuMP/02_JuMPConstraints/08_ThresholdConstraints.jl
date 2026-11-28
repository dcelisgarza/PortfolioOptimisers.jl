"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add minimum-holding threshold constraints to the JuMP optimisation model.

A threshold says an asset, *if held*, must carry at least `lt` (long) or `st` (short) of the
budget — never a sliver. The constraint therefore has to be gated by the held/sign indicator,
which is why it lives with the indicator bundle rather than with the plain weight bounds, and
why its shape depends on which bundle the builder produced.

# Mathematical definition

For [`HeldIndicators`](@ref) the bounds give the side of each entry, so the held gate
multiplies the threshold of that side directly and needs no big-M slack:

```math
\\begin{align}
w_i &\\geq g_i\\, \\ell_i\\,, \\quad i \\notin \\mathcal{S}\\,, & w_i &\\leq -g_i\\, s_i\\,, \\quad i \\in \\mathcal{S}\\,,
\\end{align}
```

where ``g_i`` is the held gate ([`held`](@ref)), which relaxes to a continuous variable when
the budget is free, and ``\\mathcal{S}`` holds the short entries of the bundle, its field
`sht`. A side with no entry adds no row.

For [`LongShortIndicators`](@ref) the long and short thresholds are separate, each switched off
by a big-M slack when the asset is on the other side:

```math
\\begin{align}
w_i &\\geq g^l_i\\, \\ell_i - M(1 - b^l_i)\\,, & w_i &\\leq -g^s_i\\, s_i + M(1 - b^s_i)\\,.
\\end{align}
```

Where:

  - ``g^l_i``, ``g^s_i``: Long and short gates ([`long_gate`](@ref), [`short_gate`](@ref)).
  - ``b^l_i``, ``b^s_i``: Long and short binaries ([`long_bin`](@ref), [`short_bin`](@ref)), which switch the constraint off — they must be the binaries, not the gates, so the switch is exactly 0 or 1 even when the gates relax.
  - ``\\ell_i``, ``s_i``: Long and short minimum-holding thresholds.
  - ``M``: Big-M constant.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraints act on.
  - `ind::AbstractMIPIndicators`: Indicator bundle the thresholds gate on.
  - `wx`: Weight expression in the space of `sp`.
  - `lt::Option{<:Threshold}`: Long-side minimum-holding threshold.
  - `lt_flag::Bool`: Whether to add the long threshold.
  - `st::Option{<:Threshold}`: Short-side minimum-holding threshold.
  - `st_flag::Bool`: Whether to add the short threshold.
  - `ss`: Big-M expression (long-short method only), as registered by [`set_mip_ss_expr!`](@ref). It defaults to the registered expression, `shared_get(model, :ss)`.

The long-short method called with the eight arguments of the held method serves a held call that reads a long-short bundle of the weights, which an earlier call declared ([`mip_indicators!`](@ref)). It adds the thresholds with the registered big-M expression `ss`.

# Returns

  - `nothing`.

# Related

  - [`mip_constraints`](@ref)
  - [`short_mip_threshold_constraints`](@ref)
  - [`AbstractMIPIndicators`](@ref)
  - [`Threshold`](@ref)
"""
function set_threshold_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                    ind::HeldIndicators, wx, lt::Option{<:Threshold},
                                    lt_flag::Bool, st::Option{<:Threshold}, st_flag::Bool)
    sc = get_constraint_scale(model)
    g = held(ind)
    il = findall(!, ind.sht)
    is = findall(ind.sht)
    if lt_flag && !isempty(il)
        lv = nothing_scalar_array_view(lt.val, il)
        model[mip_key(sp, :w_mip_lt)] = JuMP.@constraint(model,
                                                         sc *
                                                         (view(wx, il) - view(g, il) ⊙ lv) >=
                                                         0)
    end
    if st_flag && !isempty(is)
        sv = nothing_scalar_array_view(st.val, is)
        model[mip_key(sp, :w_mip_st)] = JuMP.@constraint(model,
                                                         sc *
                                                         (view(wx, is) + view(g, is) ⊙ sv) <=
                                                         0)
    end
    return nothing
end
function set_threshold_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                    ind::LongShortIndicators, wx, lt::Option{<:Threshold},
                                    lt_flag::Bool, st::Option{<:Threshold}, st_flag::Bool,
                                    ss = shared_get(model, :ss))
    sc = get_constraint_scale(model)
    if lt_flag
        model[mip_key(sp, :w_mip_lt)] = JuMP.@constraint(model,
                                                         sc *
                                                         (wx - long_gate(ind) ⊙ lt.val +
                                                          ss * (1 .- long_bin(ind))) >= 0)
    end
    if st_flag
        model[mip_key(sp, :w_mip_st)] = JuMP.@constraint(model,
                                                         sc *
                                                         (wx + short_gate(ind) ⊙ st.val -
                                                          ss * (1 .- short_bin(ind))) <= 0)
    end
    return nothing
end
