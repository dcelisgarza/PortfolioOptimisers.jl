mutable struct ArithmeticReturn{T1 <: UncertaintySet} <: OptimisationReturnsType
    uc_set::T1
end
function ArithmeticReturn(; uc_set::UncertaintySet = NoUncertaintySet())
    return ArithmeticReturn{typeof(uc_set)}(uc_set)
end

export ArithmeticReturn
