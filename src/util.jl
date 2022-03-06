# Types to flag safe and unsafe methods
struct Safe end;   const SAFE = Safe()
struct Unsafe end; const UNSAFE = Unsafe()
# Methods marked `UNSAFE` may produce unpredictable behavior
const Safety = Union{Safe,Unsafe}

@inline inbounds_safety() = (@boundscheck return SAFE; UNSAFE)


const Maybe{T} = Union{T,Nothing}

# Apply f to arg, or return nothing if arg is nothing
@inline maybe(f, arg) = f(arg)
@inline maybe(_, ::Nothing) = nothing

# Use generalized version?
# @inline maybe(f, args...) = f(args...)
# @inline maybe(_, ::Vararg{Maybe}) = nothing

# Base.@propagate_inbounds map_propagate_inbounds(f::F, args::Tuple) where F
