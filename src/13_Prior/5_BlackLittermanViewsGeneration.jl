struct BlackLittermanViews{T1, T2} <: AbstractResult
    P::T1
    Q::T2
    function BlackLittermanViews(P::AbstractMatrix, Q::AbstractVector)
        @argcheck(!isempty(P) && !isempty(Q))
        @argcheck(size(P, 1) == length(Q))
        return new{typeof(P), typeof(Q)}(P, Q)
    end
end
function BlackLittermanViews(; P::AbstractMatrix, Q::AbstractVector)
    return BlackLittermanViews(P, Q)
end
function black_litterman_views(::Nothing, args...; kwargs...)
    return nothing
end
function black_litterman_views(blves::LinearConstraintEstimator, args...; kwargs...)
    return blves
end
function get_black_litterman_views(lcs::Union{<:ParsingResult,
                                              <:AbstractVector{<:ParsingResult}},
                                   sets::AssetSets; datatype::DataType = Float64,
                                   strict::Bool = false)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs))
    end
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
    nx = sets.dict[sets.key]
    At = Vector{datatype}(undef, length(nx))
    for lc in lcs
        fill!(At, zero(eltype(At)))
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = "$(v) is not found in $(nx)."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
                continue
            end
            At += Ai * c
        end
        @argcheck(any(!iszero, At),
                  DomainError("At least one entry in At must be non-zero:\nany(!iszero, At) => $(any(!iszero, At))"))
        append!(P, At)
        append!(Q, lc.rhs)
    end
    return if !isempty(P)
        P = transpose(reshape(P, length(nx), :))
        BlackLittermanViews(; P = P, Q = Q)
    else
        nothing
    end
end
function black_litterman_views(eqn::Union{<:AbstractString, Expr,
                                          <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                               sets::AssetSets; datatype::DataType = Float64,
                               strict::Bool = false)
    lcs = parse_equation(eqn; ops1 = ("==",), ops2 = (:call, :(==)), datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, true)
    return get_black_litterman_views(lcs, sets; datatype = datatype, strict = strict)
end
function black_litterman_views(lcs::LinearConstraintEstimator, sets::AssetSets;
                               datatype::DataType = Float64, strict::Bool = false)
    return black_litterman_views(lcs.val, sets; datatype = datatype, strict = strict)
end
function black_litterman_views(views::BlackLittermanViews, args...; kwargs...)
    return views
end
function assert_bl_views_conf(::Nothing, args...)
    return nothing
end
function assert_bl_views_conf(views_conf::Real,
                              ::Union{<:AbstractString, Expr,
                                      <:AbstractVector{<:Union{<:AbstractString, Expr}}})
    @argcheck(zero(views_conf) < views_conf < one(views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::AbstractVector{<:Real},
                              val::Union{<:AbstractString, Expr,
                                         <:AbstractVector{<:Union{<:AbstractString, Expr}}})
    if isa(val, AbstractVector)
        @argcheck(length(val) == length(views_conf))
    end
    @argcheck(all(x -> zero(x) < x < one(x), views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::Union{<:Real, <:AbstractVector{<:Real}},
                              views::LinearConstraintEstimator)
    return assert_bl_views_conf(views_conf, views.val)
end
function assert_bl_views_conf(views_conf::Union{<:Real, <:AbstractVector{<:Real}},
                              views::BlackLittermanViews)
    return @argcheck(length(views_conf) == length(views.Q))
end

export black_litterman_views, BlackLittermanViews
