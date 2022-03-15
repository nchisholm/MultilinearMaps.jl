import ArrayInterface


Base.IndexStyle(::Type{<:MultilinearMap}) = IndexCartesian()
Base.IndexStyle(mf::MultilinearMap) = Base.IndexStyle(typeof(mf))

@inline Base.ndims(::Type{<:MultilinearMapN{N}}) where N = N
@inline Base.ndims(f::MultilinearMap) = Base.ndims(typeof(f))

@generated ArrayInterface.axes_types(::Type{<:MultilinearMap{Sz}}) where Sz =
    Tuple{map(D -> ArrayInterface.SOneTo{known(D)}, fieldtypes(Sz))...}

@generated ArrayInterface.axes(::T) where {T<:MultilinearMap} =
    map(T -> T(), tupletypes(ArrayInterface.axes_types(T)))

@generated Base.length(::Type{<:MultilinearMap{Sz}}) where {Sz} =
    mapreduce(known, *, tupletypes(Sz))
@inline Base.length(f::MultilinearMap) = Base.length(typeof(f))

@inline Base.axes(f::MultilinearMap) = ArrayInterface.axes(f)

@inline Base.size(f::MultilinearMap, args...) =
    map(Int, ArrayInterface.size(f, args...))

@inline Base.CartesianIndices(f::MultilinearMap) = CartesianIndices(axes(f))

# TODO: decide how to handle trailing singleton indices
@propagate_inbounds Base.getindex(f::MultilinearMapN{N}, I::Vararg{Int,N}) where N =
    _getindex(inbounds_safety(), f, I...)

@propagate_inbounds Base.getindex(f::MultilinearMapN{N}, I::CartesianIndex{N}) where N =
    _getindex(inbounds_safety(), f, I)

@inline _getindex(✓::Safety, f::MultilinearMapN{N}, I::Vararg{Int,N}) where N =
    f(map((L,i) -> StdUnitVector{known(L)}(✓,i), ArrayInterface.size(f), I)...)

@inline _getindex(✓::Safety, f::MultilinearMapN{N}, I::CartesianIndex{N}) where N =
    _getindex(✓, f, Tuple(I)...)


@inline Base.firstindex(f::MultilinearMap) = first(CartesianIndices(f))

@inline Base.lastindex(f::MultilinearMap) = last(CartesianIndices(f))

@inline Base.eltype(f::MultilinearMap) = eltype(first(f))


# XXX: hack.  For some reason plain-old collect(::MultilinearMap) doesn't work.
# It looks like it doesn't like OptionallyStaticUnitRange
Base.collect(f::MultilinearMap) = Base.collect(eltype(f), f)

# @inline Base.similar(f::MultilinearMap, ::Type{T}=eltype(f), s::Sz=size(f)) where T =
#     Array{T, ndims(f)}(undef, s)
#
# @inline Base.similar(f::MultilinearMap, s::Sz) =
#     Array{eltype(f), ndims(f)}(undef, s)


# `StaticArrays` triats

@inline StaticArrays.Size(::Type{MM}) where {MM<:MultilinearMap} =
    Size(ArrayInterface.known_size(MM))
@inline StaticArrays.Size(::MM) where {MM<:MultilinearMap} =
    StaticArrays.Size(MM)

@inline StaticArrays.Length(::Type{MM}) where {MM<:MultilinearMap} =
    Length(ArrayInterface.known_length(MM))
@inline StaticArrays.Length(::MM) where {MM<:MultilinearMap} =
    StaticArrays.Length(MM)
# @inline StaticArrays.Length(f::MultilinearForm) = StaticArrays.Length(typeof(f))

StaticArrays.similar_type(MM::Type{<:MultilinearMap}, ElType::Type, S::Size = Size(MM)) =
    similar_type(StaticArray, ElType, S)

StaticArrays.similar_type(f::MultilinearMap, ElType::Type = eltype(f), S::Size = Size(f)) =
    similar_type(StaticArray, ElType, S)

StaticArrays.similar_type(f::MultilinearMap, S::Size) =
    similar_type(StaticArray, eltype(f), S)

# XXX AHOY MATIES! We be type pirates XXX
# Also, this appears to have a small runtime cost, perhaps to compute `eltype(sized_gen)`
@inline function StaticArrays.sacollect(::Type{SA}, sized_gen) where {SA<:StaticArray}
    SA′ = similar_type(SA, eltype(sized_gen), Size(sized_gen))
    return StaticArrays.sacollect(SA′, sized_gen)
end

# Allow to collect MultilinearForms as `SArray(mf)`, `MArray(mf)`, etc.
@inline (::Type{SA})(f::MultilinearMap) where {SA<:StaticArray} =
    StaticArrays.sacollect(SA, f)

# NOTE it would be nice if ~sacollect~ had a generic method that could
# handle iterators that possessed a ~Size~ trait without having to specify the
# size in the type ~SA~.  We have hacked that together above, but maybe
# something like this should be considered for inclusion in ~StaticArrays~
# itself.
