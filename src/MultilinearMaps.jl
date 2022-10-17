module MultilinearMaps

# include("imports.jl")
include("FixedFunctions.jl")  # TODO: move to independent package
include("StandardBasis.jl")
include("MultilinearMap.jl")
# include("PartialMap.jl")
# include("algebra.jl")
# include("arraylike.jl")
# include("materialize.jl")
# include("exterior.jl")

export materialize!

# NOTE: need to force specialization on f
function materialize!(f::F, tgt::AbstractArray) where F
    bases = map(StandardBasis, size(tgt))
    @inbounds for I ∈ CartesianIndices(tgt)
        es = ntuple(k -> bases[k][I[k]], Val(ndims(tgt)))  # basis vectors
        tgt[I] = f(es...)
    end
    return tgt
end

materialize!(f::MultilinearMap{N}, tgt::AbstractArray{<:Any,N}) where N =
    materialize!(f.impl, tgt)


end # module
