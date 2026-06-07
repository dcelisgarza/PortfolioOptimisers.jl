"""
    set_l1_regularisation!(args...)

No-op fallback for L1 regularisation setup.

Called when no L1 regularisation is configured. Returns `nothing`.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_1 \\|\\boldsymbol{w}\\|_1\\,.
\\end{align}
```

Where:

  - ``\\lambda_1``: L1 regularisation coefficient.
  - $(math_dict[:w_port])

# Related

  - [`LpRegularisation`](@ref)
  - [`set_lp_regularisation!`](@ref)
"""
function set_l1_regularisation!(args...)
    return nothing
end
"""
    set_l2_regularisation!(args...)

No-op fallback for L2 regularisation setup.

Called when no L2 regularisation is configured. Returns `nothing`.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_2 \\|\\boldsymbol{w}\\|_2\\,.
\\end{align}
```

Where:

  - ``\\lambda_2``: L2 regularisation coefficient.
  - $(math_dict[:w_port])

# Related

  - [`LpRegularisation`](@ref)
  - [`set_lp_regularisation!`](@ref)
"""
function set_l2_regularisation!(args...)
    return nothing
end
"""
    set_lp_regularisation!(args...)

No-op fallback for Lp regularisation setup.

Called when no Lp regularisation is configured. Returns `nothing`.

# Related

  - [`LpRegularisation`](@ref)
"""
function set_lp_regularisation!(args...)
    return nothing
end
"""
    set_linf_regularisation!(args...)

No-op fallback for L∞ regularisation setup.

Called when no L∞ regularisation is configured. Returns `nothing`.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\lambda_\\infty \\|\\boldsymbol{w}\\|_\\infty\\,.
\\end{align}
```

Where:

  - ``\\lambda_\\infty``: L∞ regularisation coefficient.
  - $(math_dict[:w_port])

# Related

  - [`LpRegularisation`](@ref)
"""
function set_linf_regularisation!(args...)
    return nothing
end
function set_l1_regularisation!(model::JuMP.Model, l1_val::Number)
    w = model[:w]
    sc = model[:sc]
    JuMP.@variable(model, t_l1)
    JuMP.@constraint(model, cl1_noc,
                     [sc * t_l1; sc * w] in JuMP.MOI.NormOneCone(1 + length(w)))
    JuMP.@expression(model, l1, l1_val * t_l1)
    add_to_objective_penalty!(model, l1)
    return nothing
end
function set_l2_regularisation!(model::JuMP.Model, l2_val::Number)
    w = model[:w]
    sc = model[:sc]
    JuMP.@variable(model, t_l2)
    JuMP.@constraint(model, cl2_soc, [sc * t_l2; sc * w] in JuMP.SecondOrderCone())
    JuMP.@expression(model, l2, l2_val * t_l2)
    add_to_objective_penalty!(model, l2)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all portfolio weight regularisation estimators.

# Related

  - [`LpRegularisation`](@ref)
"""
abstract type AbstractRegularisationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Lp-norm regularisation term added to the optimisation objective.

# Mathematical definition

```math
\\begin{align}
\\text{penalty} &= \\mathrm{val} \\cdot \\left( \\sum_{i=1}^N |w_i|^p \\right)^{1/p}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{val}``: Regularisation coefficient.
  - ``w_i``: Portfolio weight for asset ``i``.
  - ``p > 1``: Norm order.
  - $(math_dict[:N])

Penalises concentrated portfolios by encouraging weight smoothness for ``p > 1``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LpRegularisation(; p::Number = 3, val::Number = 1e-3) -> LpRegularisation

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(p)`.
  - `p > 1`.
  - `val > 0` and finite.

# Related

  - [`AbstractRegularisationEstimator`](@ref)
"""
@concrete struct LpRegularisation <: AbstractRegularisationEstimator
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:val])
    """
    val
    function LpRegularisation(p::Number, val::Number)
        @argcheck(isfinite(p), IsNonFiniteError)
        @argcheck(p > one(p), DomainError)
        assert_nonempty_gt0_finite_val(val, :val)
        return new{typeof(p), typeof(val)}(p, val)
    end
end
function LpRegularisation(; p::Number = 3, val::Number = 1e-3)
    return LpRegularisation(p, val)
end
"""
    const VecLpReg = AbstractVector{<:LpRegularisation}

Alias for a vector of [`LpRegularisation`](@ref) objects.

Represents a collection of Lp-norm regularisation terms to be added to the optimisation objective.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
"""
const VecLpReg = AbstractVector{<:LpRegularisation}
"""
    const LpReg_VecLpReg = Union{<:LpRegularisation, <:VecLpReg}

Alias for a single or vector of Lp regularisation terms.

Matches either a single [`LpRegularisation`](@ref) or a vector of them ([`VecLpReg`](@ref)).

# Related

  - [`LpRegularisation`](@ref)
  - [`VecLpReg`](@ref)
  - [`set_lp_regularisation!`](@ref)
"""
const LpReg_VecLpReg = Union{<:LpRegularisation, <:VecLpReg}
function set_lp_regularisation!(model::JuMP.Model, lps::LpReg_VecLpReg)
    w = model[:w]
    sc = model[:sc]
    N = length(w)
    for (i, lp) in enumerate(lps)
        val = lp.val
        p_inv = inv(lp.p)
        t_lp, r_lp = model[Symbol(:t_lp_, i)], model[Symbol(:r_lp_, i)] = JuMP.@variables(model,
                                                                                          begin
                                                                                              ()
                                                                                              [1:N]
                                                                                          end)
        model[(Symbol(:clp_, i))], model[Symbol(:cslp_, i)] = JuMP.@constraints(model,
                                                                                begin
                                                                                    [i = 1:N],
                                                                                    [sc *
                                                                                     r_lp[i],
                                                                                     sc *
                                                                                     t_lp,
                                                                                     sc *
                                                                                     w[i]] in
                                                                                    JuMP.MOI.PowerCone(p_inv)
                                                                                    sc *
                                                                                    (sum(r_lp) -
                                                                                     t_lp) ==
                                                                                    0
                                                                                end)
        lp_expr = model[Symbol(:lp_, i)] = JuMP.@expression(model, val * t_lp)
        add_to_objective_penalty!(model, lp_expr)
    end
end
function set_linf_regularisation!(model::JuMP.Model, linf::Number)
    w = model[:w]
    sc = model[:sc]
    t_linf = model[:t_linf] = JuMP.@variable(model)
    model[:clinf_nic] = JuMP.@constraint(model,
                                         [sc * t_linf;
                                          sc * w] in
                                         JuMP.MOI.NormInfinityCone(1 + length(w)))
    linf = model[:linf] = JuMP.@expression(model, linf * t_linf)
    add_to_objective_penalty!(model, linf)
    return nothing
end

export LpRegularisation
