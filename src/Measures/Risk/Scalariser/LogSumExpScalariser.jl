struct LogSumExpScalariser{T1 <: Real} <: Scalariser
    gamma::T1
end
function LogSumExpScalariser(; gamma::Real = 1.0)
    @smart_assert(gamma > zero(gamma))
    return LogSumExpScalariser{typeof(gamma)}(gamma)
end

export LogSumExpScalariser
