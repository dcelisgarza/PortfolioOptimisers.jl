struct NoLoGo <: AbstractLoGo end
function logo!(::NoLoGo, args...)
    return nothing
end
