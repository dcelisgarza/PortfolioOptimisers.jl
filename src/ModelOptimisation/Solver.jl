struct Solver{T1 <: Union{Symbol, <:AbstractString}, T2, T3 <: NamedTuple,
              T4 <: Union{Nothing, <:AbstractDict}, T5 <: Bool}
    name::T1
    solver::T2
    check_sol::T3
    settings::T4
    add_bridges::T5
end
function Solver(; name::Union{Symbol, <:AbstractString} = "", solver::Any = nothing,
                check_sol::NamedTuple = (;),
                settings::Union{Nothing, <:AbstractDict} = nothing,
                add_bridges::Bool = true)
    return Solver{typeof(name), typeof(solver), typeof(check_sol), typeof(settings),
                  typeof(add_bridges)}(name, solver, check_sol, settings, add_bridges)
end
function Base.isequal(A::Solver, B::Solver)
    for property ∈ propertynames(A)
        prop_A = getproperty(A, property)
        prop_B = getproperty(B, property)
        if !isequal(prop_A, prop_B)
            return false
        end
    end
    return true
end

export Solver
