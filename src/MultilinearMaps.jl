module MultilinearMaps

using Base: @propagate_inbounds
import Base
using Static
using StaticArrays

include("util.jl")
include("MultilinearMap.jl")
include("SlicedMultilinearMap.jl")
include("stdbasis.jl")
include("arraylike.jl")
include("iteration.jl")

end
