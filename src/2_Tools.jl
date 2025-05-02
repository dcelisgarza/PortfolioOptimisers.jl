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
    to_remove = Set{Int}()
    sizehint!(to_remove, div(length(candidate_idx), 2))
    for idx ∈ candidate_idx
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
        for i ∈ axes(X, 2)
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
                     T5 <: Union{Nothing, <:AbstractVector}} <: AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
end
function ReturnsResult(; nx::Union{Nothing, <:AbstractVector} = nothing,
                       X::Union{Nothing, <:AbstractMatrix} = nothing,
                       nf::Union{Nothing, <:AbstractVector} = nothing,
                       F::Union{Nothing, <:AbstractMatrix} = nothing,
                       ts::Union{Nothing, <:AbstractVector} = nothing)
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
    return ReturnsResult{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts)}(nx, X,
                                                                                   nf, F,
                                                                                   ts)
end
function prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                           ret_method::Symbol = :simple, padding::Bool = false,
                           missing_col_percent::Real = 1.0, missing_row_percent::Real = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Union{Nothing, Function} = nothing,
                           join_method::Symbol = :outer,
                           impute_method::Union{Nothing, <:Impute.Imputor} = nothing)
    @smart_assert(zero(missing_col_percent) <
                  missing_col_percent <=
                  one(missing_col_percent))
    if !isinf(missing_row_percent)
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
    keep_cols = if !isinf(missing_row_percent)
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
    return ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F)
end
⊗(A::AbstractArray, B::AbstractArray) = reshape(kron(B, A), (length(A), length(B)))
outer_prod(A::AbstractArray, B::AbstractArray) = reshape(kron(B, A), (length(A), length(B)))
⊙(A::AbstractArray, B::AbstractArray) = A .* B
⊙(A::AbstractArray, B) = A * B
⊙(A, B::AbstractArray) = A * B
⊘(A::AbstractArray, B::AbstractArray) = A ./ B
⊘(A::AbstractArray, B) = A / B
⊘(A, B::AbstractArray) = A ./ B
⊖(A::AbstractArray, B::AbstractArray) = A - B
⊖(A::AbstractArray, B) = A .- B
⊖(A, B::AbstractArray) = A .- B
function nothing_scalar_array_view(::Nothing, ::Any)
    return nothing
end
function nothing_scalar_array_view(x::Real, ::Any)
    return x
end
function nothing_scalar_array_view(x::AbstractVector, i)
    return view(x, i)
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
function nothing_scalar_array_getindex(x::AbstractArray, i, j)
    return x[i, j]
end
function fourth_moment_index_factory(N::Integer, i)
    idx = Vector{Int}(undef, 0)
    sizehint!(idx, length(i)^2)
    for c ∈ i
        append!(idx, (((c - 1) * N + 1):(c * N))[i])
    end
    return idx
end

export drop_correlated, drop_incomplete, ReturnsResult, prices_to_returns
