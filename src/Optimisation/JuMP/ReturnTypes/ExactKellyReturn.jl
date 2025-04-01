struct ExactKellyReturn{T1 <: Union{Nothing, <:Real}} <: PortfolioReturnType
    lb::T1
end
function ExactKellyReturn(; lb::Union{Nothing, <:Real} = nothing)
    return ExactKellyReturn{typeof(lb)}(lb)
end

export ExactKellyReturn
