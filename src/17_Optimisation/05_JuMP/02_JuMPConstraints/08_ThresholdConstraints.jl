"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add minimum-holding threshold constraints to the JuMP optimisation model.

A threshold says an asset, *if held*, must carry at least `lt` (long) or `st` (short) of the
budget — never a sliver. The constraint therefore has to be gated by the held/sign indicator,
which is why it lives with the indicator bundle rather than with the plain weight bounds, and
why its shape depends on which bundle the builder produced.

# Mathematical definition

For [`HeldIndicators`](@ref) the held gate multiplies the threshold directly, so a single
constraint suffices and there is no short side:

```math
w_i \\geq g_i\\, \\ell_i\\,,
```

where ``g_i`` is the held gate ([`held`](@ref)), which relaxes to a continuous variable when
the budget is free.

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
  - `st::Option{<:Threshold}`: Short-side minimum-holding threshold (long-short method only).
  - `ss`: Big-M expression (long-short method only), as registered by [`set_mip_ss_expr!`](@ref).
  - `lt_flag::Bool`: Whether to add the long threshold.
  - `st_flag::Bool`: Whether to add the short threshold (long-short method only).

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
                                    lt_flag::Bool)
    if lt_flag
        sc = get_constraint_scale(model)
        model[mip_key(sp, :w_mip_lt)] = JuMP.@constraint(model,
                                                         sc * (wx - held(ind) ⊙ lt.val) >=
                                                         0)
    end
    return nothing
end
function set_threshold_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                    ind::LongShortIndicators, wx, lt::Option{<:Threshold},
                                    st::Option{<:Threshold}, ss, lt_flag::Bool,
                                    st_flag::Bool)
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
