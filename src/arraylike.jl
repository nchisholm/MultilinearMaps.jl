# Act like an array

@inline Base.IteratorSize(::Type{<:MultilinearMap{N}}) where N =
    Base.HasShape{N}()

Base.IndexStyle(::Type{<:MultilinearMap}) = IndexCartesian()
Base.IndexStyle(mf::MultilinearMap) = Base.IndexStyle(typeof(mf))

@inline Base.eltype(::Type{<:MultilinearMap{N, <:Size{N}, T}}) where {N,T} = T

# Needs to be defined if `MultilinearMap` is not an `AbstractArray` even though
# it `HasShape{N}`
@inline Base.ndims(::Type{<:MultilinearMap{N}}) where N = N
@inline Base.ndims(f::MultilinearMap) = Base.ndims(typeof(f))

# TODO: support dynamic dimensions?
@generated Arr.axes_types(::Type{<:MultilinearMap{N,Sz}}) where {N, Sz<:Size{N}} =
    Tuple{map(D -> Arr.SOneTo{known(D)}, fieldtypes(Sz))...}

@generated Arr.axes(::T) where {N, T<:MultilinearMap{N}} =
    ntuple(i -> Arr.axes_types(T, i)(), Val(N))

@inline Base.length(f::MultilinearMap) = dynamic(Arr.static_length(f))

@inline Base.axes(f::MultilinearMap) = Arr.axes(f)

@inline Base.size(f::MultilinearMap, dim...) = dynamic(Arr.size(f, dim...))

@inline Base.CartesianIndices(f::MultilinearMap) = CartesianIndices(axes(f))

# TODO: decide how to handle trailing singleton indices
@propagate_inbounds Base.getindex(f::MultilinearMap{N}, I::Vararg{Int,N}) where N =
    _getindex(inbounds_safety(), f, I...)

@propagate_inbounds Base.getindex(f::MultilinearMap{N}, I::CartesianIndex{N}) where N =
    _getindex(inbounds_safety(), f, I)

@inline _getindex(✓::Safety, f::MultilinearMap{N}, I::Vararg{Int,N}) where N =
    f(map((L,i) -> StdUnitVector{known(L)}(✓, i), Arr.size(f), I)...)

@inline _getindex(✓::Safety, f::MultilinearMap{N}, I::CartesianIndex{N}) where N =
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

@inline function Base.iterate(f::MultilinearMap, state)
    maybe(iterate(CartesianIndices(f), state)) do (I′, state′)
        (_getindex(UNSAFE, f, I′), state′)
    end
end


# For some reason collect does not work for things that HasShape{N}() but are
# not AbstractArrays.  Perhaps this is an omission in the iterator interface in
# Base.
# Base.collect(f::MultilinearMap) = Base.collect(eltype(f), f)

# Q: should similar be implemented?  Probably not; MultilinarMaps are backed by
# a function rather than stored values, and hence inherently immutable.
#
# @inline Base.similar(f::MultilinearMap, ::Type{T}=eltype(f), s::Sz=size(f)) where T =
#     Array{T, ndims(f)}(undef, s)
#
# @inline Base.similar(f::MultilinearMap, s::Sz) =
#     Array{eltype(f), ndims(f)}(undef, s)
