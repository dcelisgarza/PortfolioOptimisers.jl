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

export drop_correlated, drop_incomplete
