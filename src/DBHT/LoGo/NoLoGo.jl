struct NoLoGo <: AbstractLoGo end
function LoGo!(::NoLoGo, args...; kwargs...)
    return nothing
end

export NoLoGo
