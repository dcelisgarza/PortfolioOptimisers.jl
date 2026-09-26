"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the cardinality row, which caps the number of positions that the space `sp` holds.

The row sums the held binary of the bundle `ind`, which [`held_bin`](@ref) returns, so it counts held positions and reads no sign. [`mip_key`](@ref) names the row for `sp`, so one method serves the asset space and every sub-group. The method for `card = nothing` adds nothing, so it never reads a held binary that the bundle does not carry.

# JuMP formulation

## Variables

  - The held binary of `ind`, read through [`held_bin`](@ref). The MIP builder that returned `ind` created it.

## Constraints

  - `card` in the asset space, and `scard_i_` in a sub-group space: ``s_c \\left(\\sum_{j=1}^{M} b_j - c\\right) \\leq 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:b_held])
  - ``c``: The largest number of held positions, `card`.
  - $(math_dict[:i_smtx])

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: The weight space that the row acts on.
  - `ind::AbstractMIPIndicators`: The indicator bundle that the MIP builder returned.
  - `card::Option{<:Integer}`: The largest number of held positions, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_gcard_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref): Calls it in the asset space.
  - [`set_scardmip_constraints!`](@ref): Calls it in a sub-group space.
  - [`held_bin`](@ref)
  - [`mip_key`](@ref)
"""
function set_card_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                               ind::AbstractMIPIndicators, card::Integer)
    sc = get_constraint_scale(model)
    ib = held_bin(ind)
    model[mip_key(sp, :card)] = JuMP.@constraint(model, sc * (sum(ib) - card) <= 0)
    return nothing
end
function set_card_constraints!(::JuMP.Model, ::AbstractMIPSpace, ::Any, ::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the group-cardinality rows, which bound the number of held positions in each group.

`gcard` states each group as a row of a linear constraint on the held binary of the bundle `ind`, which [`held_bin`](@ref) returns. An inequality row caps a count, and an equality row fixes it. [`mip_key`](@ref) names the rows for `sp`, so one method serves the asset space and every sub-group. The method for `gcard = nothing` adds nothing, so it never reads a held binary that the bundle does not carry.

# JuMP formulation

## Variables

  - The held binary of `ind`, read through [`held_bin`](@ref). The MIP builder that returned `ind` created it.

## Constraints

  - `gcard_ineq`, when `gcard.ineq` is not `nothing`: ``s_c \\left(\\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{b} - \\boldsymbol{B}_{\\mathrm{ineq}}\\right) \\leq 0``.
  - `gcard_eq`, when `gcard.eq` is not `nothing`: ``s_c \\left(\\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{b} - \\boldsymbol{B}_{\\mathrm{eq}}\\right) = 0``.

In a sub-group space, [`mip_key`](@ref) adds the prefix and the index of the space to each name. [`set_smip_constraints!`](@ref) states the names that each sub-group path writes.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:b_held])
  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:ineq])
  - $(math_dict[:eq])

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: The weight space that the rows act on.
  - `ind::AbstractMIPIndicators`: The indicator bundle that the MIP builder returned.
  - `gcard::Option{<:LinearConstraint}`: The group-cardinality constraint, with one column for each position of `sp`, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_card_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref): Calls it in the asset space.
  - [`set_sgcardmip_constraints!`](@ref): Calls it in a sub-group space.
  - [`held_bin`](@ref)
  - [`mip_key`](@ref)
"""
function set_gcard_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                ind::AbstractMIPIndicators, gcard::LinearConstraint)
    sc = get_constraint_scale(model)
    ib = held_bin(ind)
    if !isnothing(gcard.ineq)
        A = gcard.ineq.A
        B = gcard.ineq.B
        model[mip_key(sp, :gcard_ineq)] = JuMP.@constraint(model, sc * (A * ib ⊖ B) <= 0)
    end
    if !isnothing(gcard.eq)
        A = gcard.eq.A
        B = gcard.eq.B
        model[mip_key(sp, :gcard_eq)] = JuMP.@constraint(model, sc * (A * ib ⊖ B) == 0)
    end
    return nothing
end
function set_gcard_constraints!(::JuMP.Model, ::AbstractMIPSpace, ::Any, ::Nothing)
    return nothing
end
"""
    subgroup_entry(x, i)

Return the entry of a sub-group slot that belongs to the `i`-th selection matrix.

A vector slot holds one entry for each selection matrix, so the method for a vector returns `x[i]`. Any other value, a scalar or `nothing`, applies to every selection matrix, so the other method returns it unchanged.

# Related

  - [`set_all_smip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
"""
function subgroup_entry(x::AbstractVector, i::Integer)
    return x[i]
end
function subgroup_entry(x, ::Integer)
    return x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the sub-group cardinality and group-cardinality rows when both kinds of row read one selection matrix.

The rows act on the sub-group weights `smtx * w`, so a held position is a held sub-group. One indicator builder serves both kinds of row. The vector method runs the single-matrix method once for each matrix, with the position of the matrix as the index `i`.

# Algorithm

 1. Return `nothing` when `card`, `gcard`, `lt` and `st` are all `nothing`.
 2. Make `sp`, a [`SubsetMIPSpace`](@ref) over `smtx` with the prefix `:s` and the index `i`.
 3. Run [`run_mip_builder!`](@ref) on `sp`, which gives `sib`, the indicator bundle of the sub-group weights. The builder adds the rows of the long threshold `lt` and of the short threshold `st`.
 4. Add the cardinality row of `card` with [`set_card_constraints!`](@ref).
 5. Add the group-cardinality rows of `gcard` with [`set_gcard_constraints!`](@ref).

The vector method gives the `i`-th entry of each vector slot to the `i`-th matrix, through [`subgroup_entry`](@ref). A slot that is not a vector applies to every matrix.

# JuMP formulation

## Constraints

  - `scard_i_`: The row of [`set_card_constraints!`](@ref) on the held binary of `sib`, when `card` is not `nothing`.
  - `sgcard_ineq_i_` and `sgcard_eq_i_`: The rows of [`set_gcard_constraints!`](@ref) on the held binary of `sib`, when `gcard` is not `nothing`.

[`run_mip_builder!`](@ref) states the indicator and threshold rows that step 3 adds.

Where:

  - $(math_dict[:i_smtx])

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: The largest number of held sub-groups, one for each matrix in the vector method, or `nothing`.
  - `gcard`: The group-cardinality constraint on the held sub-groups, one for each matrix in the vector method, or `nothing`. It has one column for each row of its matrix.
  - `smtx`: The selection matrix, or a vector of them. Each row selects the assets of one sub-group.
  - `lt`: The long threshold of the sub-group weights, or a vector with one entry for each matrix.
  - `st`: The short threshold of the sub-group weights, or a vector with one entry for each matrix.
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: The index of the matrix, which names the rows. The single-matrix method only.

# Returns

  - `nothing`.

# Related

  - [`set_smip_constraints!`](@ref): Calls it when the two kinds of row share a matrix and a threshold.
  - [`set_scardmip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`run_mip_builder!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer},
                                   gcard::Option{<:LinearConstraint},
                                   smtx::Option{<:MatNum}, lt::Option{<:Threshold},
                                   st::Option{<:Threshold}, ss::Option{<:Number},
                                   i::Integer = 1)
    card_flag = !isnothing(card)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = run_mip_builder!(model, sp, wb, lt, st, ss, lt_flag, st_flag)
    set_card_constraints!(model, sp, sib, card)
    set_gcard_constraints!(model, sp, sib, gcard)
    return nothing
end
function set_all_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:VecInt}, gcard::Option{<:VecLc},
                                   smtx::VecMatNum, lt::Option{<:Bt_VecOptBt},
                                   st::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    for (i, s) in enumerate(smtx)
        set_all_smip_constraints!(model, wb, subgroup_entry(card, i),
                                  subgroup_entry(gcard, i), s, subgroup_entry(lt, i),
                                  subgroup_entry(st, i), ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the sub-group cardinality rows, which cap the number of held sub-groups of a selection matrix.

The row acts on the sub-group weights `smtx * w`, so a held position is a held sub-group. The vector method runs the single-matrix method once for each matrix, with the position of the matrix as the index `i`.

# Algorithm

 1. Return `nothing` when `card`, `lt` and `st` are all `nothing`.
 2. Make `sp`, a [`SubsetMIPSpace`](@ref) over `smtx` with the prefix `:s` and the index `i`.
 3. Run [`run_mip_builder!`](@ref) on `sp`, which gives `sib`, the indicator bundle of the sub-group weights. The builder adds the rows of the long threshold `lt` and of the short threshold `st`.
 4. Add the cardinality row of `card` with [`set_card_constraints!`](@ref).

The vector method gives the `i`-th entry of each vector slot to the `i`-th matrix, through [`subgroup_entry`](@ref). A slot that is not a vector applies to every matrix.

# JuMP formulation

## Constraints

  - `scard_i_`: The row of [`set_card_constraints!`](@ref) on the held binary of `sib`, when `card` is not `nothing`.

[`run_mip_builder!`](@ref) states the indicator and threshold rows that step 3 adds.

Where:

  - $(math_dict[:i_smtx])

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: The largest number of held sub-groups, one for each matrix in the vector method, or `nothing`.
  - `smtx`: The selection matrix, or a vector of them. Each row selects the assets of one sub-group.
  - `lt`: The long threshold of the sub-group weights, or a vector with one entry for each matrix.
  - `st`: The short threshold of the sub-group weights, or a vector with one entry for each matrix.
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: The index of the matrix, which names the rows. The single-matrix method only.

# Returns

  - `nothing`.

# Related

  - [`set_smip_constraints!`](@ref)
  - [`set_all_smip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`run_mip_builder!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:Integer}, smtx::Option{<:MatNum},
                                   lt::Option{<:Threshold}, st::Option{<:Threshold},
                                   ss::Option{<:Number}, i::Integer = 1)
    card_flag = !isnothing(card)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(card_flag || lt_flag || st_flag)
        return nothing
    end
    sp = SubsetMIPSpace(smtx, :s, i)
    sib = run_mip_builder!(model, sp, wb, lt, st, ss, lt_flag, st_flag)
    set_card_constraints!(model, sp, sib, card)
    return nothing
end
function set_scardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                   card::Option{<:VecInt}, smtx::VecMatNum,
                                   lt::Option{<:Bt_VecOptBt}, st::Option{<:Bt_VecOptBt},
                                   ss::Option{<:Number})
    for (i, s) in enumerate(smtx)
        set_scardmip_constraints!(model, wb, subgroup_entry(card, i), s,
                                  subgroup_entry(lt, i), subgroup_entry(st, i), ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the sub-grouped group-cardinality rows, which bound the number of held sub-groups in each group of sub-groups.

The rows act on the sub-group weights `sgmtx * w`, so a held position is a held sub-group. The vector method runs the single-matrix method once for each matrix, with the position of the matrix as the index `i`.

# Algorithm

 1. Return `nothing` when `gcard`, `lt` and `st` are all `nothing`.
 2. Make `sp`, a [`SubsetMIPSpace`](@ref) over `sgmtx` with the prefix `:sg` and the index `i`.
 3. Run [`run_mip_builder!`](@ref) on `sp`, which gives `sib`, the indicator bundle of the sub-group weights. The builder adds the rows of the long threshold `lt` and of the short threshold `st`.
 4. Add the group-cardinality rows of `gcard` with [`set_gcard_constraints!`](@ref).

The vector method gives the `i`-th entry of each vector slot to the `i`-th matrix, through [`subgroup_entry`](@ref). A slot that is not a vector applies to every matrix.

# JuMP formulation

## Constraints

  - `sggcard_ineq_i_` and `sggcard_eq_i_`: The rows of [`set_gcard_constraints!`](@ref) on the held binary of `sib`, when `gcard` is not `nothing`.

[`run_mip_builder!`](@ref) states the indicator and threshold rows that step 3 adds.

Where:

  - $(math_dict[:i_smtx])

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `gcard`: The group-cardinality constraint on the held sub-groups, one for each matrix in the vector method, or `nothing`. It has one column for each row of its matrix.
  - `sgmtx`: The selection matrix, or a vector of them. Each row selects the assets of one sub-group.
  - `lt`: The long threshold of the sub-group weights, or a vector with one entry for each matrix.
  - `st`: The short threshold of the sub-group weights, or a vector with one entry for each matrix.
  - $(arg_dict[:ss_arg])
  - `i::Integer = 1`: The index of the matrix, which names the rows. The single-matrix method only.

# Returns

  - `nothing`.

# Related

  - [`set_smip_constraints!`](@ref)
  - [`set_all_smip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`run_mip_builder!`](@ref)
  - [`WeightBounds`](@ref)
"""
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::Option{<:LinearConstraint},
                                    sgmtx::Option{<:MatNum}, lt::Option{<:Threshold},
                                    st::Option{<:Threshold}, ss::Option{<:Number},
                                    i::Integer = 1)
    gcard_flag = !isnothing(gcard)
    lt_flag = !isnothing(lt)
    st_flag = !isnothing(st)
    if !(gcard_flag || lt_flag || st_flag)
        return nothing
    end
    sp = SubsetMIPSpace(sgmtx, :sg, i)
    sib = run_mip_builder!(model, sp, wb, lt, st, ss, lt_flag, st_flag)
    set_gcard_constraints!(model, sp, sib, gcard)
    return nothing
end
function set_sgcardmip_constraints!(model::JuMP.Model, wb::WeightBounds,
                                    gcard::Option{<:VecLc}, sgmtx::VecMatNum,
                                    lt::Option{<:Bt_VecOptBt}, st::Option{<:Bt_VecOptBt},
                                    ss::Option{<:Number})
    for (i, s) in enumerate(sgmtx)
        set_sgcardmip_constraints!(model, wb, subgroup_entry(gcard, i), s,
                                   subgroup_entry(lt, i), subgroup_entry(st, i), ss, i)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add every sub-group MIP row, through one indicator builder when the two kinds of row share their matrix and their thresholds.

The shared path needs `smtx === sgmtx`, `lt === glt` and `st === gst`, because the one indicator bundle it builds carries one pair of thresholds. When any pair differs, each kind of row gets its own builder, so each reads its own thresholds. The test is identity and not equality, so two equal matrices that are distinct objects take the separate path, which builds two indicator bundles for the same sub-groups.

# Algorithm

 1. When `smtx === sgmtx`, `lt === glt` and `st === gst`, call [`set_all_smip_constraints!`](@ref) with `smtx`, `lt` and `st`.
 2. Otherwise, call [`set_scardmip_constraints!`](@ref) with `smtx`, `lt` and `st`, then call [`set_sgcardmip_constraints!`](@ref) with `sgmtx`, `glt` and `gst`.

# JuMP formulation

## Constraints

  - Step 1: `scard_i_` from `card`, and `sgcard_ineq_i_` and `sgcard_eq_i_` from `gcard`, which [`set_all_smip_constraints!`](@ref) adds.
  - Step 2: `scard_i_` from `card`, which [`set_scardmip_constraints!`](@ref) adds, and `sggcard_ineq_i_` and `sggcard_eq_i_` from `gcard`, which [`set_sgcardmip_constraints!`](@ref) adds.

So the name of a group-cardinality row tells which path added it.

Where:

  - $(math_dict[:i_smtx])

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `card`: The largest number of held sub-groups of `smtx`, one for each matrix when `smtx` is a vector, or `nothing`.
  - `gcard`: The group-cardinality constraint on the held sub-groups of `sgmtx`, one for each matrix when `sgmtx` is a vector, or `nothing`.
  - `smtx`: The selection matrix of the cardinality rows, or a vector of them.
  - `sgmtx`: The selection matrix of the group-cardinality rows, or a vector of them.
  - `lt`: The long threshold of the sub-group weights of `smtx`.
  - `st`: The short threshold of the sub-group weights of `smtx`.
  - `glt`: The long threshold of the sub-group weights of `sgmtx`.
  - `gst`: The short threshold of the sub-group weights of `sgmtx`.
  - $(arg_dict[:ss_arg])

# Returns

  - `nothing`.

# Related

  - [`set_all_smip_constraints!`](@ref)
  - [`set_scardmip_constraints!`](@ref)
  - [`set_sgcardmip_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref): Adds the rows of the asset space.
  - [`assert_subgroup_mip_fields`](@ref)
  - [`assert_subgrouped_mip_fields`](@ref)
"""
function set_smip_constraints!(model::JuMP.Model, wb::WeightBounds,
                               card::Option{<:Int_VecInt}, gcard::Option{<:Lc_VecLc},
                               smtx::Option{<:MatNum_VecMatNum},
                               sgmtx::Option{<:MatNum_VecMatNum}, lt::Option{<:Bt_VecOptBt},
                               st::Option{<:Bt_VecOptBt}, glt::Option{<:Bt_VecOptBt},
                               gst::Option{<:Bt_VecOptBt}, ss::Option{<:Number})
    if smtx === sgmtx && lt === glt && st === gst
        set_all_smip_constraints!(model, wb, card, gcard, smtx, lt, st, ss)
    else
        set_scardmip_constraints!(model, wb, card, smtx, lt, st, ss)
        set_sgcardmip_constraints!(model, wb, gcard, sgmtx, glt, gst, ss)
    end
    return nothing
end
"""
    assert_subgroup_mip_fields(scard, smtx, slt, sst)

Refuses a set of sub-group cardinality slots whose shapes do not agree.

The constructors of [`JuMPOptimiser`](@ref) and [`ProgrammeAllocationSet`](@ref) call it, because both carry the four slots. A `TimeDependent` in any slot is not yet a value, so the check skips all four, and the substitution of each fold checks them.

# Validation

  - When `scard` is an `Integer`, it is finite and `scard > 0`. `smtx` is a matrix or an [`AssetSetsMatrixEstimator`](@ref), and `slt` and `sst` are each a scalar or `nothing`.
  - When `scard` is a vector, it is not empty and each entry is finite and `> 0`. `smtx` is a vector of the same length, and a vector `slt` or `sst` is not empty and has that length too.
  - When `scard` is `nothing` and `slt` or `sst` is a scalar, `smtx` is a matrix or an estimator.
  - When `scard` is `nothing` and `slt` or `sst` is a vector, `smtx` is a vector that is not empty, and each vector threshold has its length.

A value check throws an `IsEmptyError` or a `DomainError`. A type check throws an `ArgumentError`, a length check a `DimensionMismatch`, and an empty threshold or `smtx` an `IsEmptyError`.

# Related

  - [`set_smip_constraints!`](@ref)
  - [`assert_subgrouped_mip_fields`](@ref)
"""
function assert_subgroup_mip_fields(scard, smtx, slt, sst)::Nothing
    if any(x -> isa(x, TimeDependent), (scard, smtx, slt, sst))
        return nothing
    end
    if isa(scard, Integer)
        assert_nonempty_gt0_finite_val(scard, :scard)
        @argcheck(isa(smtx, MatNum_ASetMatE),
                  ArgumentError("smtx must be a MatNum_ASetMatE when scard is an Integer, got $(typeof(smtx))"))
        @argcheck(isa(slt, Option{<:BtE_Bt}),
                  ArgumentError("slt must be a scalar BtE_Bt or nothing when scard is an Integer, got $(typeof(slt))"))
        @argcheck(isa(sst, Option{<:BtE_Bt}),
                  ArgumentError("sst must be a scalar BtE_Bt or nothing when scard is an Integer, got $(typeof(sst))"))
    elseif isa(scard, VecInt)
        assert_nonempty_gt0_finite_val(scard, :scard)
        @argcheck(isa(smtx, AbstractVector),
                  ArgumentError("smtx must be an AbstractVector when scard is a VecInt, got $(typeof(smtx))"))
        @argcheck(length(scard) == length(smtx),
                  DimensionMismatch("scard ($(length(scard))) must match smtx ($(length(smtx)))"))
        if isa(slt, AbstractVector)
            @argcheck(!isempty(slt), IsEmptyError("slt cannot be empty"))
            @argcheck(length(scard) == length(slt),
                      DimensionMismatch("scard ($(length(scard))) must match slt ($(length(slt)))"))
        end
        if isa(sst, AbstractVector)
            @argcheck(!isempty(sst), IsEmptyError("sst cannot be empty"))
            @argcheck(length(scard) == length(sst),
                      DimensionMismatch("scard ($(length(scard))) must match sst ($(length(sst)))"))
        end
    elseif isnothing(scard) && (isa(slt, BtE_Bt) || isa(sst, BtE_Bt))
        @argcheck(isa(smtx, MatNum_ASetMatE),
                  ArgumentError("smtx must be a MatNum_ASetMatE when slt or sst is a scalar BtE_Bt, got $(typeof(smtx))"))
    elseif isnothing(scard) && (isa(slt, AbstractVector) || isa(sst, AbstractVector))
        @argcheck(isa(smtx, AbstractVector),
                  ArgumentError("smtx must be an AbstractVector when slt or sst is a vector, got $(typeof(smtx))"))
        @argcheck(!isempty(smtx), IsEmptyError("smtx cannot be empty"))
        if isa(slt, AbstractVector)
            @argcheck(!isempty(slt), IsEmptyError("slt cannot be empty"))
            @argcheck(length(slt) == length(smtx),
                      DimensionMismatch("slt ($(length(slt))) must match smtx ($(length(smtx)))"))
        end
        if isa(sst, AbstractVector)
            @argcheck(!isempty(sst), IsEmptyError("sst cannot be empty"))
            @argcheck(length(sst) == length(smtx),
                      DimensionMismatch("sst ($(length(sst))) must match smtx ($(length(smtx)))"))
        end
    end
    return nothing
end
"""
    assert_subgrouped_constraint_width(sgc, smt)

Refuses a precomputed sub-grouped cardinality constraint whose width is not the number of sub-groups in its selection matrix.

[`set_gcard_constraints!`](@ref) multiplies the matrix `A` of each half of `sgc` by the held binary of the sub-groups, which has one entry for each row of `smt`. So every half needs one column for each row of `smt`, and it can have any number of rows. The other method accepts every other pair, such as an estimator, because the width of an estimator is not known until the optimiser resolves it.

# Validation

  - Each half of `sgc` that is not `nothing` has `size(A, 2) == size(smt, 1)`. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`assert_subgrouped_mip_fields`](@ref)
  - [`set_gcard_constraints!`](@ref)
"""
function assert_subgrouped_constraint_width(sgc::LinearConstraint, smt::MatNum)::Nothing
    G = size(smt, 1)
    for half in (sgc.ineq, sgc.eq)
        if isnothing(half)
            continue
        end
        @argcheck(size(half.A, 2) == G,
                  DimensionMismatch("the sub-grouped cardinality constraint is written over $(size(half.A, 2)) sub-groups, but its selection matrix has $G rows, one for each sub-group"))
    end
    return nothing
end
function assert_subgrouped_constraint_width(::Any, ::Any)::Nothing
    return nothing
end
"""
    assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)

Refuses a set of sub-grouped cardinality slots whose shapes do not agree.

The constructors of [`JuMPOptimiser`](@ref) and [`ProgrammeAllocationSet`](@ref) call it, because both carry the four slots. The builder pairs `sgcarde` with `sgmtx`, so a precomputed [`LinearConstraint`](@ref) needs one column for each row of `sgmtx`. A `TimeDependent` in any slot is not yet a value, so the check skips all four, and the substitution of each fold checks them.

# Validation

  - When `sgcarde` is a scalar, `sgmtx` is a matrix or an [`AssetSetsMatrixEstimator`](@ref), and `sglt` and `sgst` are each a scalar or `nothing`. A [`LinearConstraint`](@ref) with a matrix `sgmtx` passes [`assert_subgrouped_constraint_width`](@ref).
  - When `sgcarde` is a vector, it is not empty. `sgmtx` is a vector of the same length that is not empty, a vector `sglt` or `sgst` is not empty and has that length too, and each pair of a constraint and a matrix passes [`assert_subgrouped_constraint_width`](@ref).
  - When `sgcarde` is `nothing` and `sglt` or `sgst` is a scalar, `sgmtx` is a matrix or an estimator.
  - When `sgcarde` is `nothing` and `sglt` or `sgst` is a vector, `sgmtx` is a vector that is not empty, and each vector threshold has its length.

A type check throws an `ArgumentError`, a length or width check a `DimensionMismatch`, and an empty slot an `IsEmptyError`.

# Related

  - [`set_smip_constraints!`](@ref)
  - [`assert_subgroup_mip_fields`](@ref)
  - [`assert_subgrouped_constraint_width`](@ref)
"""
function assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)::Nothing
    if any(x -> isa(x, TimeDependent), (sgcarde, sgmtx, sglt, sgst))
        return nothing
    end
    if isa(sgcarde, LcE_Lc)
        @argcheck(isa(sgmtx, MatNum_ASetMatE),
                  ArgumentError("sgmtx must be a MatNum_ASetMatE when sgcarde is a scalar LcE_Lc, got $(typeof(sgmtx))"))
        @argcheck(isa(sglt, Option{<:BtE_Bt}),
                  ArgumentError("sglt must be a scalar BtE_Bt or nothing when sgcarde is a scalar LcE_Lc, got $(typeof(sglt))"))
        @argcheck(isa(sgst, Option{<:BtE_Bt}),
                  ArgumentError("sgst must be a scalar BtE_Bt or nothing when sgcarde is a scalar LcE_Lc, got $(typeof(sgst))"))
        assert_subgrouped_constraint_width(sgcarde, sgmtx)
    elseif isa(sgcarde, AbstractVector)
        @argcheck(!isempty(sgcarde), IsEmptyError("sgcarde cannot be empty"))
        @argcheck(isa(sgmtx, AbstractVector),
                  ArgumentError("sgmtx must be an AbstractVector when sgcarde is a vector, got $(typeof(sgmtx))"))
        @argcheck(!isempty(sgmtx), IsEmptyError("sgmtx cannot be empty"))
        @argcheck(length(sgcarde) == length(sgmtx),
                  DimensionMismatch("sgcarde ($(length(sgcarde))) must match sgmtx ($(length(sgmtx)))"))
        if isa(sglt, AbstractVector)
            @argcheck(!isempty(sglt), IsEmptyError("sglt cannot be empty"))
            @argcheck(length(sgcarde) == length(sglt),
                      DimensionMismatch("sgcarde ($(length(sgcarde))) must match sglt ($(length(sglt)))"))
        end
        if isa(sgst, AbstractVector)
            @argcheck(!isempty(sgst), IsEmptyError("sgst cannot be empty"))
            @argcheck(length(sgcarde) == length(sgst),
                      DimensionMismatch("sgcarde ($(length(sgcarde))) must match sgst ($(length(sgst)))"))
        end
        for (sgc, smt) in zip(sgcarde, sgmtx)
            assert_subgrouped_constraint_width(sgc, smt)
        end
    elseif isnothing(sgcarde) && (isa(sglt, BtE_Bt) || isa(sgst, BtE_Bt))
        @argcheck(isa(sgmtx, MatNum_ASetMatE),
                  ArgumentError("sgmtx must be a MatNum_ASetMatE when sglt or sgst is a scalar BtE_Bt, got $(typeof(sgmtx))"))
    elseif isnothing(sgcarde) && (isa(sglt, AbstractVector) || isa(sgst, AbstractVector))
        @argcheck(isa(sgmtx, AbstractVector),
                  ArgumentError("sgmtx must be an AbstractVector when sglt or sgst is a vector, got $(typeof(sgmtx))"))
        @argcheck(!isempty(sgmtx), IsEmptyError("sgmtx cannot be empty"))
        if isa(sglt, AbstractVector)
            @argcheck(!isempty(sglt), IsEmptyError("sglt cannot be empty"))
            @argcheck(length(sglt) == length(sgmtx),
                      DimensionMismatch("sglt ($(length(sglt))) must match sgmtx ($(length(sgmtx)))"))
        end
        if isa(sgst, AbstractVector)
            @argcheck(!isempty(sgst), IsEmptyError("sgst cannot be empty"))
            @argcheck(length(sgst) == length(sgmtx),
                      DimensionMismatch("sgst ($(length(sgst))) must match sgmtx ($(length(sgmtx)))"))
        end
    end
    return nothing
end
