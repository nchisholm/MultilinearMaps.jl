using StaticArrays: SArray

export materialize!, materialize


"""
    materialize!(tgt::AbstractArray, f::MultilinearMap)

Fill `tgt` with the comopnents of `f`.
"""
materialize!(tgt::AbstractArray, f::MultilinearMap) =
    (samesize(tgt, f); _unsafe_materialize!(tgt, f))

# Sequantially fills `tgt` with the components of `f`.
@generated function _unsafe_materialize!(tgt::AbstractArray{<:Any, N},
                                         f::MultilinearMap{N}) where N
    quote
        @nloops $N i tgt begin
            # tgt[i,j,k,...] = f[i,j,k,...]
            @inbounds (@nref $N tgt i) = (@nref $N f i)
        end
        tgt
    end
end


# Support general `AbstractArray`s

"""
    materialize(T::Type{<:AbstractArray}, f::MultilinearMap)

Create an array of type `T` filled with the components of `f`.
"""
@inline materialize(AA::Type{<:AbstractArray}, f::MultilinearMap) =
    materialize!(empty_target(AA, f), f)

empty_target(AA::Type{<:AbstractArray{T}}, f::MultilinearMap) where T =
    similar(AA, Arr.axes(f))

empty_target(AA::Type{<:AbstractArray}, f::MultilinearMap) =
    similar(AA{eltype(f)}, Arr.axes(f))


# Support `StaticArray`s

@inline function materialize(SA::Type{<:StaticArray}, f::MultilinearMap)
    SA′ = target_type(SA, f)
    samesize(SA′, f)
    sacollect(SA′, f)
end

# target_type(SA::Type{<:StaticArray{SA_Sz, T}}, ::MultilinearMap) where {SA_Sz<:Tuple, T} =
#     similar_type(SA, T, _SA.Size(SA_Sz))

# @inline target_type(SA::Type{<:StaticArray{<:Tuple, T}}, f::MultilinearMap) where T =
#     similar_type(SA, T, _SA.Size(Base.size(f)))  # XXX slow

target_type(SA::Type{<:StaticArray{SA_Sz}}, f::MultilinearMap) where {SA_Sz<:Tuple} =
    @isdefined(SA_Sz) ? similar_type(SA, eltype(f), _SA.Size(SA_Sz)) :
        _SA.missing_size_error(SA)
        # similar_type(Union{SA,StaticArray}, eltype(f), _SA.Size(f))

target_type(SA::Type{<:StaticArray}, f::MultilinearMap) =
    similar_type(SA, eltype(f), _SA.Size(f))


# Allow materialization via constructors: `Array(f)`, `SArray(mf)`,
# `MArray(mf)`, etc.
@inline (AA::Type{<:AbstractArray})(f::MultilinearMap) = materialize(AA, f)
@inline (SA::Type{<:StaticArray})(f::MultilinearMap) = materialize(SA, f)


# Get a StaticArrays.Size trait for a MultilinearMap
@inline _SA.Size(f::MultilinearMap) = _SA.Size(Arr.known_size(f))
@inline _SA.Length(f::MultilinearMap) = _SA.Length(Arr.known_length(f))
