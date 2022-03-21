# Act like an array

@inline Base.IteratorSize(::Type{<:MultilinearMap{<:Size{N}}}) where N =
    Base.HasShape{N}()

Base.IndexStyle(::Type{<:MultilinearMap}) = IndexCartesian()
Base.IndexStyle(mf::MultilinearMap) = Base.IndexStyle(typeof(mf))

@inline Base.eltype(::Type{<:MultilinearMap{<:Size,T}}) where T = T

# Needs to be defined if `MultilinearMap` is not an `AbstractArray` even though
# it `HasShape{N}`
@inline Base.ndims(::Type{<:MultilinearMap{<:Size{N}}}) where N = N
@inline Base.ndims(f::MultilinearMap) = Base.ndims(typeof(f))

# TODO: support dynamic dimensions?
@generated Arr.axes_types(::Type{<:MultilinearMap{Sz}}) where Sz =
    Tuple{map(D -> Arr.SOneTo{known(D)}, fieldtypes(Sz))...}

@generated Arr.axes(::T) where {N, T<:MultilinearMap{<:Size{N}}} =
    ntuple(i -> Arr.axes_types(T, i)(), Val(N))

@inline Base.length(f::MultilinearMap) = dynamic(Arr.static_length(f))

@inline Base.axes(f::MultilinearMap) = Arr.axes(f)

@inline Base.size(f::MultilinearMap, dim...) = dynamic(Arr.size(f, dim...))

@inline Base.CartesianIndices(f::MultilinearMap) = CartesianIndices(axes(f))

# TODO: decide how to handle trailing singleton indices
@propagate_inbounds Base.getindex(f::MultilinearMapN{N}, I::Vararg{Int,N}) where N =
    _getindex(inbounds_safety(), f, I...)

@propagate_inbounds Base.getindex(f::MultilinearMapN{N}, I::CartesianIndex{N}) where N =
    _getindex(inbounds_safety(), f, I)

@inline _getindex(✓::Safety, f::MultilinearMapN{N}, I::Vararg{Int,N}) where N =
    f(map((L,i) -> StdUnitVector{known(L)}(✓, i), Arr.size(f), I)...)

@inline _getindex(✓::Safety, f::MultilinearMapN{N}, I::CartesianIndex{N}) where N =
    _getindex(✓, f, Tuple(I)...)

@inline Base.firstindex(f::MultilinearMap) = first(CartesianIndices(f))

@inline Base.lastindex(f::MultilinearMap) = last(CartesianIndices(f))


# NOTE inlining is important to performance here
@inline function Base.iterate(f::MultilinearMap)
    # Piggy-back off of iterate(::CartesianIndices)
    (I, state) = iterate(CartesianIndices(f))
    return (_getindex(UNSAFE, f, I), state)
    #   -> (f[I], state)
    # Should be safe to elide the unit vector validity check
end

@inline function Base.iterate(f::MultilinearMap{K}, state) where {K}
    maybe(iterate(CartesianIndices(f), state)) do (I′, state′)
        (_getindex(UNSAFE, f, I′), state′)
    end
end


# For some reason collect does not work for things that HasShape{N}() but are
# not AbstractArrays.  Perhaps this is an omission in the iterator interface in
# Base.
Base.collect(f::MultilinearMap) = Base.collect(eltype(f), f)

# Q: should similar be implemented?  Probably not; MultilinarMaps are backed by
# a function rather than stored values, and hence inherently immutable.
#
# @inline Base.similar(f::MultilinearMap, ::Type{T}=eltype(f), s::Sz=size(f)) where T =
#     Array{T, ndims(f)}(undef, s)
#
# @inline Base.similar(f::MultilinearMap, s::Sz) =
#     Array{eltype(f), ndims(f)}(undef, s)


# `_SA` triats

@inline _SA.Size(::Type{MM}) where {MM<:MultilinearMap} =
    _SA.Size(Arr.known_size(MM))
@inline _SA.Size(::MM) where {MM<:MultilinearMap} =
    _SA.Size(MM)

@inline _SA.Length(::Type{MM}) where {MM<:MultilinearMap} =
    _SA.Length(Arr.known_length(MM))
@inline _SA.Length(::MM) where {MM<:MultilinearMap} =
    _SA.Length(MM)
# @inline _SA.Length(f::MultilinearForm) = _SA.Length(typeof(f))

_SA.similar_type(MM::Type{<:MultilinearMap},
                          ElType::Type,
                          S::_SA.Size = _SA.Size(MM)) =
    _SA.similar_type(StaticArray, ElType, S)

_SA.similar_type(f::MultilinearMap,
                          ElType::Type = eltype(f),
                          S::_SA.Size = _SA.Size(f)) =
    _SA.similar_type(StaticArray, ElType, S)

_SA.similar_type(f::MultilinearMap, S::_SA.Size) =
    _SA.similar_type(StaticArray, eltype(f), S)

# XXX AHOY MATIES! We be type pirates XXX
# Also, this appears to have a small runtime cost, perhaps to compute `eltype(sized_gen)`
@inline function _SA.sacollect(::Type{SA}, sized_gen) where {SA<:StaticArray}
    SA′ = similar_type(SA, eltype(sized_gen), _SA.Size(sized_gen))
    return sacollect(SA′, sized_gen)
end

# Allow to collect MultilinearForms as `SArray(mf)`, `MArray(mf)`, etc.
@inline (::Type{SA})(f::MultilinearMap) where {SA<:StaticArray} =
    sacollect(SA, f)

# NOTE it would be nice if ~sacollect~ had a generic method that could
# handle iterators that possessed a ~Size~ trait without having to specify the
# size in the type ~SA~.  We have hacked that together above, but maybe
# something like this should be considered for inclusion in ~_SA~
# itself.
