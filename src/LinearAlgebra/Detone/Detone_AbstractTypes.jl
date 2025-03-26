abstract type AbstractDetone end
function detone! end
function detone!(::Nothing, args...)
    return nothing
end

export detone!
