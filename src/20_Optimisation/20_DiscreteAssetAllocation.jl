struct DiscreteAllocation{T1 <: Union{<:Solver, <:AbstractVector{<:Solver}}} <:
       BaseAssetAllocationOptimisationEstimator
    slv::T1
end
function DiscreteAllocation(; slv::Union{<:Solver, <:AbstractVector{<:Solver}})
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return DiscreteAllocation{typeof(slv)}(slv)
end
function optimise!(da::DiscreteAllocation, w::AbstractVector, p::AbstractVector,
                   cash::Real = 1e6, T::Union{Nothing, <:Real} = nothing,
                   fees::Union{Nothing, <:Fees} = nothing)
    @smart_assert(!isempty(w) && !isempty(p) && length(w) == length(p))
    @smart_assert(cash > zero(cash))
    if !isnothing(fees)
        @smart_assert(!isnothing(T))
    end
end

export DiscreteAllocation
