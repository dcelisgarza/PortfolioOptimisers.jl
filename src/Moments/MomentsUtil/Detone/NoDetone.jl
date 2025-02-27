struct NoDetone <: AbstractDetone end
function detone!(::NoDetone, args...)
    return nothing
end

export NoDetone
