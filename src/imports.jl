using Base: @propagate_inbounds, tail, promote_eltype
import Base
using Static
import ArrayInterface as Arr
using StaticArrays: StaticArray, sacollect
import StaticArrays: StaticArrays as _SA, similar_type
