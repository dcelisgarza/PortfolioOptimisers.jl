abstract type BaseAssetAllocationOptimisationEstimator <: BaseOptimisationEstimator end
abstract type AssetAllocationOptimisationAlgorithm <: OptimisationAlgorithm end
function setup_alloc_optim(w::AbstractVector, p::AbstractVector, cash::Real,
                           T::Union{Nothing, <:Real} = nothing,
                           fees::Union{Nothing, <:Fees} = nothing)
    if !isnothing(T) && !isnothing(fees)
        cash -= calc_fees(w, p, fees) * T
    end
    bgt = sum(w)
    lidx = w .>= zero(eltype(w))
    long = all(lidx)
    if long
        lbgt = bgt
        sbgt = zero(eltype(w))
        sidx = Vector{eltype(w)}(undef, 0)
        scash = zero(eltype(w))
    else
        sidx = .!lidx
        lbgt = sum(view(w, lidx))
        sbgt = -sum(view(w, sidx))
        scash = cash * sbgt
    end
    lcash = cash * lbgt
    return cash, bgt, lbgt, sbgt, lidx, sidx, lcash, scash
end
