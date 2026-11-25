"""
$(DocStringExtensions.TYPEDEF)

Wrapper that applies the rows of a linear constraint to the gross, long or short positions, and not to the weights.

`lce` states the frame of the rows: assets, groups, the factors of a [`FactorSpace`](@ref), or the nodes of a [`DependencySpace`](@ref). `pos` states the reading of the weights. The two choices are independent, so `"tech <= 0.4"` caps the gross position of a group, and `"Protocol <= 0.3"` caps the gross dependency on a node. The wrapper changes no row. [`linear_constraints`](@ref) resolves `lce` as it does in the slot `lcse`, and gives a [`PositionConstraint`](@ref) for each [`LinearConstraint`](@ref).

A net cap does not limit a hedged position: two weights of +50 % and -50 % in one group have a net position of zero and a gross position of 100 %. The reading of the weights needs no long or short budget. It needs weight bounds, and the parts of the weights come from them, see [`PositionConstraint`](@ref).

The cap of the absolute value of the net exposure needs no reading. `abs(e) <= b` is the two net rows `e <= b` and `e >= -b`.

The slot `lcse` is the only slot that admits the wrapper. `gcarde`, `sgcarde`, `cte` and the field `rc` of a [`Variance`](@ref) hold rows over quantities that have no long and no short part, and their bounds refuse it.

!!! warning

    The parts of the weights are upper bounds: the model has `lw >= max(w, 0)` and `sw >= max(-w, 0)`. A cap with non-negative coefficients holds on the realised weights, because the row is feasible only when it holds on the smallest parts. A floor, an equality, or a row with a negative coefficient does not hold with certainty. The solver can enlarge the parts, and then the realised position can be below the floor. Three rows have a negative coefficient: a difference of two assets, a `>=` row, and a row over a [`FactorSpace`](@ref) that has a negative loading. State `xbgt = true` in the block for such a row. It pins the parts to the weights with one binary for each asset. The constructor accepts the row in both cases, as it accepts the floor of a `gbgt`. A [`NetPosition`](@ref) reads the weights, so this limit does not apply to it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PositionConstraintEstimator(;
        lce::Union{EcE_LcE_Lc, AbstractVector},
        pos::AbstractPositionReading = GrossPosition()
    ) -> PositionConstraintEstimator

Keywords correspond to the struct's fields. A vector `lce` can mix the shapes that the field takes, and [`narrow_mixed_vector`](@ref) narrows its element type.

## Validation

  - If `lce` is an `AbstractVector`, `!isempty(lce)`. Raises an [`IsEmptyError`](@ref).

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"tech\" => [\"A\", \"B\"]));

julia> pce = PositionConstraintEstimator(; lce = LinearConstraintEstimator(; val = \"tech <= 0.4\"))
PositionConstraintEstimator
  lce ┼ LinearConstraintEstimator
      │   val ┼ String: \"tech <= 0.4\"
      │   key ┴ nothing
  pos ┴ GrossPosition()

julia> linear_constraints(pce, sets).lc.ineq.A
1×3 transpose(::Matrix{Float64}) with eltype Float64:
 1.0  1.0  0.0
```

# Related

  - [`PositionConstraint`](@ref)
  - [`AbstractPositionReading`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`linear_constraints`](@ref)
  - [`ConstraintBlock`](@ref)
"""
@concrete struct PositionConstraintEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:pce_lce])
    """
    lce
    """
    $(field_dict[:pce_pos])
    """
    pos
    function PositionConstraintEstimator(lce::EcE_LcE_Lc_VecEcE_LcE_Lc,
                                         pos::AbstractPositionReading)::PositionConstraintEstimator
        if isa(lce, AbstractVector)
            @argcheck(!isempty(lce), IsEmptyError("lce cannot be empty"))
        end
        return new{typeof(lce), typeof(pos)}(lce, pos)
    end
end
function PositionConstraintEstimator(; lce::Union{EcE_LcE_Lc, AbstractVector},
                                     pos::AbstractPositionReading = GrossPosition())::PositionConstraintEstimator
    return PositionConstraintEstimator(narrow_mixed_vector(lce, EcE_LcE_Lc, :lce), pos)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds a linear constraint over the assets and the reading of the weights that its rows apply to.

[`linear_constraints`](@ref) makes it from a [`PositionConstraintEstimator`](@ref), and it enters the slot `lcse` as given, as a [`LinearConstraint`](@ref) does.

# Mathematical definition

```math
\\begin{align}
  \\mathbf{A}_\\text{ineq}\\, r(\\boldsymbol{w}) &\\leq \\boldsymbol{B}_\\text{ineq} \\\\
  \\mathbf{A}_\\text{eq}\\, r(\\boldsymbol{w}) &= \\boldsymbol{B}_\\text{eq}\\,.
\\end{align}
```

Where:

  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:ineq])
  - $(math_dict[:eq])
  - ``r``: The reading `pos`. It is ``\\lvert\\boldsymbol{w}\\rvert`` for a [`GrossPosition`](@ref), the positive part of ``\\boldsymbol{w}`` for a [`LongPosition`](@ref), the positive part of ``-\\boldsymbol{w}`` for a [`ShortPosition`](@ref), and ``\\boldsymbol{w}`` for a [`NetPosition`](@ref).
  - $(math_dict[:w_port])

# The parts of the weights

The model reads `pos` from the weights and their long and short parts, see [`position_reading`](@ref).

  - A block whose weight bounds admit a short position reads the parts that its budgets read, see [`weight_parts!`](@ref). [`PositionConstraintEstimator`](@ref) states the rows that hold on the parts and not on the weights.
  - A block whose weight bounds admit no short position adds no variable. The gross, long and net readings are the weights, so they give the rows of the plain constraint. The short reading is zero: a cap on it has no effect, and a floor above zero is infeasible.
  - A block with no weight bounds, or with no lower bound, admits a short position.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PositionConstraint(;
        lc::LinearConstraint,
        pos::AbstractPositionReading = GrossPosition()
    ) -> PositionConstraint

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> lc = LinearConstraint(; ineq = PartialLinearConstraint(; A = [1.0 1.0 0.0], B = [0.4]));

julia> PositionConstraint(; lc = lc, pos = ShortPosition())
PositionConstraint
   lc ┼ LinearConstraint
      │   ineq ┼ PartialLinearConstraint
      │        │   A ┼ 1×3 Matrix{Float64}
      │        │   B ┴ Vector{Float64}: [0.4]
      │     eq ┴ nothing
  pos ┴ ShortPosition()
```

# Related

  - [`PositionConstraintEstimator`](@ref)
  - [`LinearConstraint`](@ref)
  - [`AbstractPositionReading`](@ref)
  - [`set_linear_weight_constraints!`](@ref)
"""
@concrete struct PositionConstraint <: AbstractConstraintResult
    """
    $(field_dict[:pcr_lc])
    """
    lc
    """
    $(field_dict[:pce_pos])
    """
    pos
    function PositionConstraint(lc::LinearConstraint,
                                pos::AbstractPositionReading)::PositionConstraint
        return new{typeof(lc), typeof(pos)}(lc, pos)
    end
end
function PositionConstraint(; lc::LinearConstraint,
                            pos::AbstractPositionReading = GrossPosition())::PositionConstraint
    return PositionConstraint(lc, pos)
end
"""
    const PcE_Pc_EcE_LcE_Lc = Union{PositionConstraintEstimator, PositionConstraint,
                                    EcE_LcE_Lc}

One entry of the slot `lcse`: a [`PositionConstraintEstimator`](@ref), its Result [`PositionConstraint`](@ref), or a constraint that the estimator wraps.

# Related

  - [`PositionConstraintEstimator`](@ref)
  - [`PositionConstraint`](@ref)
  - [`EcE_LcE_Lc`](@ref)
"""
const PcE_Pc_EcE_LcE_Lc = Union{PositionConstraintEstimator, PositionConstraint, EcE_LcE_Lc}
"""
    const VecPcE_Pc_EcE_LcE_Lc = AbstractVector{<:PcE_Pc_EcE_LcE_Lc}

Alias for an abstract vector of [`PcE_Pc_EcE_LcE_Lc`](@ref) elements.

# Related

  - [`PcE_Pc_EcE_LcE_Lc`](@ref)
"""
const VecPcE_Pc_EcE_LcE_Lc = AbstractVector{<:PcE_Pc_EcE_LcE_Lc}
"""
    const PcE_Pc_EcE_LcE_Lc_VecPcE_Pc_EcE_LcE_Lc = Union{PcE_Pc_EcE_LcE_Lc,
                                                         VecPcE_Pc_EcE_LcE_Lc}

One [`PcE_Pc_EcE_LcE_Lc`](@ref), or a vector of them. It is the bound of the slot `lcse`, and no other slot takes it.

# Related

  - [`PcE_Pc_EcE_LcE_Lc`](@ref)
  - [`VecPcE_Pc_EcE_LcE_Lc`](@ref)
"""
const PcE_Pc_EcE_LcE_Lc_VecPcE_Pc_EcE_LcE_Lc = Union{PcE_Pc_EcE_LcE_Lc,
                                                     VecPcE_Pc_EcE_LcE_Lc}
"""
    const Pc_Lc = Union{PositionConstraint, LinearConstraint}

One resolved entry of the slot `lcse`: a [`LinearConstraint`](@ref) over the weights, or a [`PositionConstraint`](@ref) over a reading of them.

# Related

  - [`PositionConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`Pc_Lc_VecPc_Lc`](@ref)
"""
const Pc_Lc = Union{PositionConstraint, LinearConstraint}
"""
    const VecPc_Lc = AbstractVector{<:Pc_Lc}

Alias for an abstract vector of [`Pc_Lc`](@ref) elements.

# Related

  - [`Pc_Lc`](@ref)
  - [`Pc_Lc_VecPc_Lc`](@ref)
"""
const VecPc_Lc = AbstractVector{<:Pc_Lc}
"""
    const Pc_Lc_VecPc_Lc = Union{Pc_Lc, VecPc_Lc}

One [`Pc_Lc`](@ref), or a vector of them. It is the bound of the resolved slot `lcsr`.

# Related

  - [`Pc_Lc`](@ref)
  - [`VecPc_Lc`](@ref)
  - [`ResolvedConstraintBlock`](@ref)
"""
const Pc_Lc_VecPc_Lc = Union{Pc_Lc, VecPc_Lc}
"""
    position_constraints(lc::LinearConstraint, pos::AbstractPositionReading)
    position_constraints(lc::Nothing, pos::AbstractPositionReading)
    position_constraints(lcs::AbstractVector, pos::AbstractPositionReading)

Wrap each resolved linear constraint with a reading of the weights.

A dropped row is `nothing` and stays `nothing`. A vector keeps its shape.

# Arguments

  - `lc`, `lcs`: The result of [`linear_constraints`](@ref) on the wrapped constraint.
  - `pos`: The reading.

# Returns

  - `pc`: A [`PositionConstraint`](@ref), `nothing`, or a vector of either.

# Related

  - [`PositionConstraint`](@ref)
  - [`linear_constraints`](@ref)
"""
function position_constraints(lc::LinearConstraint,
                              pos::AbstractPositionReading)::PositionConstraint
    return PositionConstraint(; lc = lc, pos = pos)
end
function position_constraints(::Nothing, ::AbstractPositionReading)::Nothing
    return nothing
end
function position_constraints(lcs::AbstractVector, pos::AbstractPositionReading)
    return [position_constraints(lc, pos) for lc in lcs]
end
"""
    linear_constraints(pce::PositionConstraintEstimator, sets::Option{<:UniverseSets};
                       kwargs...)
    linear_constraints(pc::PositionConstraint, args...; kwargs...)

Resolve a position constraint over the assets.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`PositionConstraintEstimator`](@ref): resolve `pce.lce` against `sets`, and wrap each result with [`position_constraints`](@ref).
 2. A [`PositionConstraint`](@ref): return it unchanged.

# Arguments

  - `pce`, `pc`: The position constraint.
  - `sets`: The declared universe the names resolve against. It can be `nothing` when `pce.lce` holds no name.
  - `kwargs...`: Passed to the method for `pce.lce`.

# Returns

  - `pc`: A [`PositionConstraint`](@ref) over the assets, `nothing` when every row dropped, or a vector of either when the estimator wraps a vector.

# Related

  - [`PositionConstraintEstimator`](@ref)
  - [`PositionConstraint`](@ref)
  - [`position_constraints`](@ref)
"""
function linear_constraints(pce::PositionConstraintEstimator, sets::Option{<:UniverseSets};
                            kwargs...)
    return position_constraints(linear_constraints(pce.lce, sets; kwargs...), pce.pos)
end
function linear_constraints(pc::PositionConstraint, args...; kwargs...)::PositionConstraint
    return pc
end
"""
    linear_constraints(lcs::VecPcE_Pc_EcE_LcE_Lc, sets::Option{<:UniverseSets}; kwargs...)

Apply [`linear_constraints`](@ref) to each element of a vector that can mix position constraints, re-based constraints and constraints over the assets. The narrower `VecLcE` method still applies to a vector that holds only [`LinearConstraintEstimator`](@ref)s.

Each element resolves its own basis, so one vector can hold a space that reads the prior and a space that states or refits its own basis.

# Arguments

  - `lcs`: The vector of shapes.
  - `sets`: The declared universe the names resolve against. It can be `nothing` when no element holds a name.
  - `kwargs...`: Passed to the method for each element. The loadings `rr` and the returns `rd` reach the elements that read them.

# Returns

  - `lcs`: One result per entry of the input, in the order of the input. An entry is a [`LinearConstraint`](@ref), a [`PositionConstraint`](@ref), `nothing`, or a vector of them when the element wraps a vector.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`PositionConstraintEstimator`](@ref)
  - [`flat_linear_constraints`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::VecPcE_Pc_EcE_LcE_Lc, sets::Option{<:UniverseSets};
                            kwargs...)
    return [linear_constraints(lc, sets; kwargs...) for lc in lcs]
end
"""
    flat_linear_constraints(lcs::Option{<:Pc_Lc_VecPc_Lc})
    flat_linear_constraints(lcs::AbstractVector)

Make one vector of the resolved entries of the slot `lcse`.

An entry that wraps a vector resolves to a vector, and an entry whose every row dropped resolves to `nothing`. The slot `lcsr` holds one constraint, or one vector of constraints, so the function puts the entries of each inner vector in the outer vector, in order, and leaves out `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. One constraint, `nothing`, or a vector of constraints: return it.
 2. Any other vector: collect its constraints with [`collect_linear_constraints!`](@ref). Return `nothing` when there is none, and a vector of their types otherwise.

# Arguments

  - `lcs`: The result of [`linear_constraints`](@ref) on the slot.

# Returns

  - `lcsr::Option{<:Pc_Lc_VecPc_Lc}`: The value of the resolved slot.

# Related

  - [`collect_linear_constraints!`](@ref)
  - [`resolve_constraint_block`](@ref)
"""
function flat_linear_constraints(lcs::Option{<:Pc_Lc_VecPc_Lc})
    return lcs
end
function flat_linear_constraints(lcs::AbstractVector)
    out = collect_linear_constraints!(Pc_Lc[], lcs)
    return isempty(out) ? nothing : concrete_typed_array(out)
end
"""
    collect_linear_constraints!(out::Vector{Pc_Lc}, lc::Nothing)
    collect_linear_constraints!(out::Vector{Pc_Lc}, lc::Pc_Lc)
    collect_linear_constraints!(out::Vector{Pc_Lc}, lcs::AbstractVector)

Add each constraint of a resolved value to `out`, in order.

# Arguments

  - `out`: The vector of constraints.
  - `lc`, `lcs`: The resolved value: `nothing`, one constraint, or a vector of resolved values.

# Returns

  - `out::Vector{Pc_Lc}`: The vector.

# Related

  - [`flat_linear_constraints`](@ref)
"""
function collect_linear_constraints!(out::Vector{Pc_Lc}, ::Nothing)
    return out
end
function collect_linear_constraints!(out::Vector{Pc_Lc}, lc::Pc_Lc)
    return push!(out, lc)
end
function collect_linear_constraints!(out::Vector{Pc_Lc}, lcs::AbstractVector)
    foreach(lc -> collect_linear_constraints!(out, lc), lcs)
    return out
end
"""
    port_opt_view(pce::PositionConstraintEstimator, i, args...) -> PositionConstraintEstimator
    port_opt_view(pc::PositionConstraint, i, args...) -> PositionConstraint

Return a copy of a position constraint over the selected assets.

The method views the wrapped constraint and keeps the reading. A precomputed [`LinearConstraint`](@ref) holds no names, so its view is unchanged, see `port_opt_view(lc::LinearConstraint, i, args...)`.

# Arguments

  - `pce`, `pc`: The position constraint to view.
  - `i`: The asset index the view keeps.
  - `args...`: More positional arguments, which the method passes on to the view of the wrapped constraint.

# Returns

  - The position constraint over the viewed constraint.

# Related

  - [`port_opt_view`](@ref)
  - [`PositionConstraintEstimator`](@ref)
  - [`PositionConstraint`](@ref)
"""
function port_opt_view(pce::PositionConstraintEstimator, i,
                       args...)::PositionConstraintEstimator
    return PositionConstraintEstimator(; lce = port_opt_view(pce.lce, i, args...),
                                       pos = pce.pos)
end
function port_opt_view(pc::PositionConstraint, i, args...)::PositionConstraint
    return PositionConstraint(; lc = port_opt_view(pc.lc, i, args...), pos = pc.pos)
end
function assert_investable_constraint_width(pc::PositionConstraint, N::Integer,
                                            slot::AbstractString)::Nothing
    return assert_investable_constraint_width(pc.lc, N, slot)
end
"""
    reader_frames(pce::PositionConstraintEstimator, path::AbstractString, refit::Bool)

Return the frames that the wrapped constraint adds to the group of the optimiser that holds it, with the path of the field `lce`.

# Arguments

  - `pce`: The position constraint.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the readers in `pce.lce`.

# Related

  - [`reader_frames`](@ref)
  - [`PositionConstraintEstimator`](@ref)
"""
function reader_frames(pce::PositionConstraintEstimator, path::AbstractString, refit::Bool)
    return reader_frames(pce.lce, join_path(path, :lce), refit)
end

export PositionConstraintEstimator, PositionConstraint
