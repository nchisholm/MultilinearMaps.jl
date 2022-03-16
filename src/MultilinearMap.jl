export MultilinearMap, MultilinearForm, AtomicMultilinearMap


abstract type MultilinearMap{Sz} end

@inline MultilinearMap{Sz}(f) where {Sz<:StaticSize} =
    AtomicMultilinearMap{Sz}(f)

# Accept MultilinearMap{(M,N)} -> MultlinearMap{Tuple{StaticInt(M), StaticInt(N)}}
@generated function MultilinearMap{Sz}(f) where {Sz}
    TupleDims = Tuple{map(i -> StaticInt{i}, Sz)...}
    :(MultilinearMap{$TupleDims}(f))
end

const MultilinearMapN{N} = MultilinearMap{<:NTuple{N,StaticInt}}
const MultilinearForm{N,D} = MultilinearMap{NTuple{N,StaticInt{D}}}
const ScalarMap = MultilinearMap{Tuple{}}

struct AtomicMultilinearMap{Sz<:StaticSize, F} <: MultilinearMap{Sz}
    impl::F
    AtomicMultilinearMap{Sz}(impl::F) where {Sz,F} = new{Sz,F}(impl)
end

@generated argtypes(::Type{<:MultilinearMap{Sz}}) where Sz =
    Tuple{map(D -> StaticVector{known(D)}, fieldtypes(Sz))...}

@inline argtypes(f::MultilinearMap) = argtypes(typeof(f))

@inline (f::ScalarMap)() = f.impl()
@inline (f::MultilinearMapN{N})(vs::Vararg{AbstractVector,N}) where N =
    f.impl(vs...)
