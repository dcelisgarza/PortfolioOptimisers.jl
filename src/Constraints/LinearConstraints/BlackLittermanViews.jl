function views_constraints(vcs::Union{<:LinearConstraintSide,
                                      <:AbstractVector{<:LinearConstraintSide}},
                           asset_sets::DataFrame, datatype::Type = Float64)
    N = nrow(asset_sets)

    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)

    for vc ∈ vcs
        vc_A, vc_B = try
            get_asset_constraint_data(vc, asset_sets; normalise = true, strict = true)
        catch err
            if isa(err, ArgumentError)
                continue
            else
                throw(err)
            end
        end

        if isempty(vc_A) || all(iszero.(vc_A))
            continue
        end

        vc_B *= -1
        # s = sum(rvc_A)
        # if !iszero(s)
        #     rvc_A ./= abs(sum(rvc_A))
        # end
        append!(P, vc_A)
        append!(Q, vc_B)
    end

    if !isempty(P)
        P = transpose(reshape(P, N, :))
        P = convert.(typeof(promote(P...)[1]), P)
        Q = convert.(typeof(promote(Q...)[1]), Q)
    end

    return P, Q
end

export views_constraints
