module MultilinearMaps

import Base
import ArrayInterface as Arr

using Base.Cartesian
using Base: @assume_effects, @propagate_inbounds
using Static

export materialize!

# include("imports.jl")
include("util.jl")
include("FixedFunctions.jl")  # TODO: move to independent package
include("StandardBasis.jl")
include("MultilinearMap.jl")
# include("PartialMap.jl")
# include("algebra.jl")
# include("arraylike.jl")
# include("materialize.jl")
# include("exterior.jl")

# NOTE: need to force specialization on f
@generated function materialize!(tgt::AbstractArray{<:Any, N},
                                 f::MultilinearMap{<:Dims{N}}) where {N}
    quote
        @nloops $N i tgt begin
            # tgt[i,j,k,...] = f[i,j,k,...]
            @inbounds (@nref $N tgt i) = (@nref $N f i)
        end
        tgt
    end
end

function materialize_CI!(tgt::AbstractArray{<:Any, N},
                         f::MultilinearMap{<:Dims{N}}) where {N}
    # Much nicer but slow...
    @inbounds for I ∈ CartesianIndices(tgt)
        tgt[I] = f[I]
    end
    return tgt
end


# function materialize!(tgt::AbstractArray{<:Any,N}, f::MultilinearMap{<:Dims{N}}) where N
#     materialize!(tgt, f.impl)
# end


end # module
