struct BlackLittermanView{T1 <: LinearConstraintSide, T2 <: LinearConstraintSide}
    lhs::T1
    rhs::T2
end
function BlackLittermanView(; lhs::LinearConstraintSide = LinearConstraintSide(),
                            rhs::LinearConstraintSide = LinearConstraintSide())
    return BlackLittermanView{typeof(lhs), typeof(rhs)}(lhs, rhs)
end
function views_constraints(vcs::Union{<:BlackLittermanView,
                                      <:AbstractVector{<:BlackLittermanView}},
                           asset_sets::DataFrame, datatype::Type = Float64,
                           strict::Bool = false)
    N = nrow(asset_sets)

    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)

    for vc ∈ vcs
        lhs = vc.lhs
        rhs = vc.rhs

        lhs_A, lhs_B = get_asset_constraint_data(lhs, asset_sets, strict)
        rhs_A, rhs_B = get_asset_constraint_data(rhs, asset_sets, strict)

        lhs_flag = isempty(lhs_A) || all(iszero.(lhs_A))
        rhs_flag = isempty(rhs_A) || all(iszero.(rhs_A))

        if lhs_flag && rhs_flag
            continue
        end

        rlhs_A = if lhs_flag
            -rhs_A
        elseif rhs_flag
            lhs_A
        else
            sign = relative_factor_constraint_sign(vc.kind)
            sign * (lhs_A - rhs_A)
        end

        rlhs_B = (rhs_B - lhs_B)

        s = sum(rlhs_A)
        if !iszero(s)
            rlhs_A ./= abs(sum(rlhs_A))
        end

        append!(P, rlhs_A)
        append!(Q, rlhs_B)
    end

    if !isempty(P)
        P = transpose(reshape(P, N, :))
        P = convert.(typeof(promote(P...)[1]), P)
        Q = convert.(typeof(promote(Q...)[1]), Q)
    end

    return P, Q
end

export BlackLittermanView, views_constraints
