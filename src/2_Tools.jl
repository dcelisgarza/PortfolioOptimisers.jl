abstract type AbstractReturnsResult <: AbstractResult end
function issquare(A::AbstractMatrix)
    @smart_assert(size(A, 1) == size(A, 2))
end
function drop_correlated(X::AbstractMatrix; threshold::Real = 0.95, absolute::Bool = false)
    N = size(X, 2)
    rho = !absolute ? cor(X) : abs.(cor(X))
    mean_rho = mean(rho; dims = 1)
    tril_idx = findall(tril!(trues(size(rho)), -1))
    candidate_idx = findall(rho[tril_idx] .>= threshold)
    candidate_idx = candidate_idx[sortperm(rho[tril_idx][candidate_idx]; rev = true)]
    to_remove = sizehint!(Set{Int}(), div(length(candidate_idx), 2))
    for idx in candidate_idx
        i, j = tril_idx[idx][1], tril_idx[idx][2]
        if i ∉ to_remove && j ∉ to_remove
            if mean_rho[i] > mean_rho[j]
                push!(to_remove, i)
            else
                push!(to_remove, j)
            end
        end
    end
    return setdiff(1:N, to_remove)
end
function drop_incomplete(X::AbstractMatrix, any_missing::Bool = true)
    N = size(X, 2)
    return if any_missing
        to_remove = Vector{Int}(undef, 0)
        for i in axes(X, 2)
            if any(isnan, view(X, :, i)) || any(ismissing, view(X, :, i))
                push!(to_remove, i)
            end
        end
        setdiff(1:N, to_remove)
    else
        (1:N)[(.!isnan.(X[1, :]) .|| ismissing.(X[1, :])) .& (.!isnan.(X[end, :]) .|| ismissing.(X[end,
                                                                                                   :]))]
    end
end
function select_kextremes(X::AbstractMatrix) end
struct ReturnsResult{T1 <: Union{Nothing, <:AbstractVector},
                     T2 <: Union{Nothing, <:AbstractMatrix},
                     T3 <: Union{Nothing, <:AbstractVector},
                     T4 <: Union{Nothing, <:AbstractMatrix},
                     T5 <: Union{Nothing, <:AbstractVector},
                     T6 <: Union{Nothing, <:AbstractMatrix},
                     T7 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
    iv::T6
    ivpa::T7
end
function ReturnsResult(; nx::Union{Nothing, <:AbstractVector} = nothing,
                       X::Union{Nothing, <:AbstractMatrix} = nothing,
                       nf::Union{Nothing, <:AbstractVector} = nothing,
                       F::Union{Nothing, <:AbstractMatrix} = nothing,
                       ts::Union{Nothing, <:AbstractVector} = nothing,
                       iv::Union{Nothing, <:AbstractMatrix} = nothing,
                       ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
    nxs_flag = !isnothing(nx)
    X_flag = !isnothing(X)
    if nxs_flag || X_flag
        @smart_assert(nxs_flag && X_flag)
        @smart_assert(!isempty(nx) && !isempty(X))
        @smart_assert(length(nx) == size(X, 2))
    end
    nfs_flag = !isnothing(nf)
    F_flag = !isnothing(F)
    if (nfs_flag || F_flag) && nxs_flag
        @smart_assert(nfs_flag && F_flag)
        @smart_assert(!isempty(nf) && !isempty(F))
        @smart_assert(length(nf) == size(F, 2))
        @smart_assert(size(X, 1) == size(F, 1))
    end
    if !isnothing(ts)
        @smart_assert(!isempty(ts))
        @smart_assert(length(ts) == size(X, 1))
    end
    if !isnothing(iv)
        @smart_assert(!isempty(iv))
        @smart_assert(size(iv) == size(X))
        @smart_assert(all(x -> x > zero(eltype(iv)), iv))
        if !isnothing(ivpa)
            if isa(ivpa, Real)
                @smart_assert(isfinite(ivpa) && ivpa > zero(ivpa))
            elseif isa(ivpa, AbstractVector)
                @smart_assert(!isempty(ivpa))
                @smart_assert(all(isfinite, ivpa) && all(x -> x > zero(eltype(ivpa)), ivpa))
                @smart_assert(length(ivpa) == size(iv, 2))
            end
        end
    end
    return ReturnsResult{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts),
                         typeof(iv), typeof(ivpa)}(nx, X, nf, F, ts, iv, ivpa)
end
function returns_result_view(rd::ReturnsResult, i::AbstractVector)
    nx = nothing_scalar_array_view(rd.nx, i)
    X = isnothing(rd.X) ? nothing : view(rd.X, :, i)
    iv = isnothing(rd.iv) ? nothing : view(rd.iv, :, i)
    ivpa = nothing_scalar_array_view(rd.ivpa, i)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, ts = rd.ts, iv = iv,
                         ivpa = ivpa)
end
function prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                           iv::Union{Nothing, <:TimeArray} = nothing,
                           ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                           ret_method::Symbol = :simple, padding::Bool = false,
                           missing_col_percent::Real = 1.0,
                           missing_row_percent::Union{Nothing, <:Real} = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Union{Nothing, Function} = nothing,
                           join_method::Symbol = :outer,
                           impute_method::Union{Nothing, <:Impute.Imputor} = nothing)
    @smart_assert(zero(missing_col_percent) <
                  missing_col_percent <=
                  one(missing_col_percent))
    if !isnothing(missing_row_percent)
        @smart_assert(zero(missing_row_percent) <
                      missing_row_percent <=
                      one(missing_row_percent))
    end
    if !isempty(F)
        asset_names = string.(colnames(X))
        factor_names = string.(colnames(F))
        X = merge(X, F; method = join_method)
    else
        asset_names = string.(colnames(X))
        factor_names = String[]
    end
    if !isnothing(map_func)
        X = map(map_func, X)
    end
    if !isempty(collapse_args)
        X = collapse(X, collapse_args...)
    end
    X = DataFrame(X)
    f(x) =
        if isa(x, Number) && isnan(x)
            missing
        else
            x
        end
    DataFrames.transform!(X, 2:ncol(X) .=> ByRow((x) -> f(x)); renamecols = false)
    if !isnothing(impute_method)
        X = Impute.impute(X, impute_method)
    end
    missing_mtx = ismissing.(Matrix(X[!, 2:end]))
    missings_cols = vec(count(missing_mtx; dims = 2))
    keep_rows = missings_cols .<= (ncol(X) - 1) * missing_col_percent
    X = X[keep_rows, :]
    missings_rows = vec(count(missing_mtx; dims = 1))
    keep_cols = if !isnothing(missing_row_percent)
        missings_rows .<= nrow(X) * missing_row_percent
    else
        missings_rows .== StatsBase.mode(missings_rows)
    end
    X = X[!, [true; keep_cols]]
    select!(X, Not(names(X, Missing)))
    dropmissing!(X)
    X = DataFrame(percentchange(TimeArray(X; timestamp = :timestamp), ret_method;
                                padding = padding))
    col_names = names(X)
    nx = intersect(col_names, asset_names)
    nf = intersect(col_names, factor_names)
    oc = setdiff(col_names, union(nx, nf))
    ts = isempty(oc) ? nothing : vec(Matrix(X[!, oc]))
    if !isnothing(ts) && !isnothing(iv)
        @smart_assert(ts == timestamp(iv))
    end
    if isempty(nf)
        nf = nothing
        F = nothing
    else
        F = Matrix(X[!, nf])
    end
    if isempty(nx)
        nx = nothing
        X = nothing
    else
        X = Matrix(X[!, nx])
    end
    return ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F, iv = values(iv),
                         ivpa = ivpa)
end
function brinson_attribution(X::TimeArray, w::AbstractVector, wb::AbstractVector,
                             asset_classes::DataFrame, col, date0 = nothing,
                             date1 = nothing)
    #! Make this efficient with filter.
    # sort!(filter!(:timestamp => x -> date0 <= x <= date1, prices), :timestamp)
    idx1, idx2 = if !isnothing(date0) && !isnothing(date1)
        timestamps = timestamp(X)
        idx = DateTime(date0) .<= timestamps .<= DateTime(date1)
        findfirst(idx), findlast(idx)
    else
        1, length(X)
    end

    ret = vec(values(X[idx2]) ./ values(X[idx1]) .- 1)

    # ret_w = dot(ret, w)
    ret_b = dot(ret, wb)

    classes = asset_classes[!, col]
    unique_classes = unique(classes)

    df = DataFrame(;
                   index = ["Asset Allocation", "Security Selection", "Interaction",
                            "Total Excess Return"])

    for class_i in unique_classes
        sets_i = BitVector(undef, 0)
        for class_j in classes
            push!(sets_i, class_i == class_j)
        end

        w_i = dot(sets_i, w)
        wb_i = dot(sets_i, wb)

        ret_i = dot(ret .* sets_i, w) / w_i
        ret_b_i = dot(ret .* sets_i, wb) / wb_i

        w_diff_i = w_i - wb_i
        ret_diff_i = ret_i - ret_b_i

        AA_i = w_diff_i * (ret_b_i - ret_b)
        SS_i = wb_i * ret_diff_i
        I_i = w_diff_i * ret_diff_i
        TER_i = AA_i + SS_i + I_i

        df[!, class_i] = [AA_i, SS_i, I_i, TER_i]
    end

    df[!, "Total"] = sum(eachcol(df[!, 2:end]))

    return df
end
⊗(A::AbstractArray, B::AbstractArray) = reshape(kron(B, A), (length(A), length(B)))
outer_prod(A::AbstractArray, B::AbstractArray) = reshape(kron(B, A), (length(A), length(B)))
⊙(A::AbstractArray, B::AbstractArray) = A .* B
⊙(A::AbstractArray, B) = A * B
⊙(A, B::AbstractArray) = A * B
⊘(A::AbstractArray, B::AbstractArray) = A ./ B
⊘(A::AbstractArray, B) = A / B
⊘(A, B::AbstractArray) = A ./ B
⊕(A::AbstractArray, B::AbstractArray) = A .+ B
⊕(A::Real, B::AbstractArray) = A .+ B
⊕(A::AbstractArray, B::Real) = A .+ B
⊕(A, B) = A + B
⊖(A::AbstractArray, B::AbstractArray) = A - B
⊖(A::AbstractArray, B) = A .- B
⊖(A, B::AbstractArray) = A .- B
⊖(A, B) = A - B
function dot_scalar(a::Real, b::AbstractVector)
    return a * sum(b)
end
function dot_scalar(a::AbstractVector, b::Real)
    return sum(a) * b
end
function dot_scalar(a::AbstractVector, b::AbstractVector)
    return dot(a, b)
end
function nothing_scalar_array_view(::Nothing, ::Any)
    return nothing
end
function nothing_scalar_array_view(x::Real, ::Any)
    return x
end
function nothing_scalar_array_view(x::AbstractVector, i)
    return view(x, i)
end
function nothing_scalar_array_view(x::AbstractVector{<:AbstractVector}, i)
    return view.(x, Ref(i))
end
function nothing_scalar_array_view(x::AbstractArray, i)
    return view(x, i, i)
end
function nothing_scalar_array_view_odd_order(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_view_odd_order(x::AbstractArray, i, j)
    return view(x, i, j)
end
function nothing_scalar_array_getindex(x::Real, ::Any)
    return x
end
function nothing_scalar_array_getindex(::Nothing, ::Any)
    return nothing
end
function nothing_scalar_array_getindex(x::AbstractVector, i)
    return x[i]
end
function nothing_scalar_array_getindex(x::AbstractMatrix, i)
    return x[i, i]
end
function nothing_scalar_array_getindex(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_getindex(x::AbstractMatrix, i, j)
    return x[i, j]
end
function nothing_dataframe_view(::Nothing, ::Any)
    return nothing
end
function nothing_dataframe_view(x::AbstractDataFrame, i)
    return view(x, i, :)
end
function nothing_dataframe_getindex(::Nothing, i)
    return nothing
end
function nothing_dataframe_getindex(x::AbstractDataFrame, i)
    return x[i, :]
end
function fourth_moment_index_factory(N::Integer, i)
    idx = sizehint!(Int[], length(i)^2)
    for c in i
        append!(idx, (((c - 1) * N + 1):(c * N))[i])
    end
    return idx
end
function traverse_subtypes(types, ctarr = nothing)
    if isnothing(ctarr)
        ctarr = []
    end
    stypes = subtypes(types)
    for stype in stypes
        if !isstructtype(stype)
            traverse_subtypes(stype, ctarr)
        else
            push!(ctarr, stype)
        end
    end
    return ctarr
end
function concrete_typed_array(A::AbstractArray)
    return reshape(Union{typeof.(A)...}[A...], size(A))
end
function factory(::Nothing, args...; kwargs...)
    return nothing
end

export drop_correlated, drop_incomplete, ReturnsResult, prices_to_returns,
       concrete_typed_array
