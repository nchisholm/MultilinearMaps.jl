module MultilinearMaps

import Base
import ArrayInterface as Arr

using Base.Cartesian
using Base: @assume_effects, @propagate_inbounds, tail
using Static
import StaticArrays
using StaticArrays: StaticArray, sacollect
using ArrayInterfaceStaticArrays

export materialize!, materialize

# include("imports.jl")
include("util.jl")
include("FixedFunctions.jl")
include("LinearFunctionSpaces.jl")
include("StandardBasis.jl")
include("MultilinearMap.jl")
include("algebra.jl")
# include("PartialMap.jl")
# include("arraylike.jl")
# include("materialize.jl")
# include("exterior.jl")

materialize!(tgt::AbstractArray, f::MultilinearMap) =
    (samesize(tgt, f); _unsafe_materialize!(tgt, f))

# NOTE: need to force specialization on f
@generated function _unsafe_materialize!(tgt::AbstractArray{<:Any, N},
                                         f::MultilinearMap{<:Dims{N}}) where {N}
    quote
        @nloops $N i tgt begin
            # tgt[i,j,k,...] = f[i,j,k,...]
            @inbounds (@nref $N tgt i) = (@nref $N f i)
        end
        tgt
    end
end

function _unsafe_materialize_CI!(tgt::AbstractArray{<:Any, N},
                                 f::MultilinearMap{<:Dims{N}}) where {N}
    # Much nicer but slow...
    @inbounds for I ∈ CartesianIndices(tgt)
        tgt[I] = f[I]
    end
    return tgt
end

# materialize(::Type{Array}, f::MultilinearMap) = collect(f)
# materialize(::Type{Array{ElT}}, f::MultilinearMap) where ElT = collect(ElT, f)
# materialize(::Type{A}, f::MultilinearMap) where {N, A<:Array{<:Any,N}} = collect(f)

# StaticArrays support

StaticArrays.Size(f::MultilinearMap) =
    StaticArrays.Size(map(d -> d === nothing ? StaticArrays.Dynamic() : d, Arr.known_size(f)))

# XXX: slow if you try to specify a type
@inline function materialize(::Type{SA}, f::MultilinearMap{<:SDims}) where {SA<:StaticArray}
    sacollect(_staticarray_target_type(SA, typeof(f)), f)
end

@generated function _staticarray_target_type(::Type{SA}, ::Type{MM}) where {SA<:StaticArray, MM<:MultilinearMap}
    sz_src = Arr.known_size(MM)
    SA1 = typeintersect(SA, StaticArray{Tuple{sz_src...}, <:Any, arity(MM)})
    if SA1 === Union{}  # no match in size type parameter
        throw(DimensionMismatch("Dimensions of the target type \n\t$SA\n do not match the source type \n\t$MM"))
        # sz_tgt = Arr.known_size(SA)
        # throw(DimensionMismatch("Cannot store a MultilinearMap of size $sz_src in a container type of size $sz_tgt)"))
    end
    return SA1
end

# XXX _SA.Length is not in StaticArraysCore

end # module
