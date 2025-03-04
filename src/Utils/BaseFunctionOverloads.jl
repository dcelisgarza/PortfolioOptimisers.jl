for (op, name) ∈ zip((Solver, LinearConstraint, BlackLittermanView),
                     ("Solver", "LinearConstraint", "BlackLittermanView"))
    eval(quote
             Base.iterate(S::$op, state = 1) = state > 1 ? nothing : (S, state + 1)
             function Base.String(s::$op)
                 return $name
             end
             function Base.Symbol(::$op)
                 return Symbol($name)
             end
             function Base.length(::$op)
                 return 1
             end
             function Base.getindex(S::$op, ::Any)
                 return S
             end
             function Base.view(S::$op, ::Any)
                 return S
             end
         end)
end
