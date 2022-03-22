module MultilinearMaps


using Base: @propagate_inbounds, tail
import Base
using Static
import ArrayInterface as Arr
using StaticArrays: StaticArray, sacollect
import StaticArrays: StaticArrays as _SA, similar_type


include("util.jl")
include("MultilinearMap.jl")
include("PartialMap.jl")
include("stdbasis.jl")
include("arraylike.jl")
include("algebra.jl")
include("materialize.jl")


end # module
