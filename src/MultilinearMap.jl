export MultilinearMap, MultilinearForm, AtomicMultilinearMap


# XXX remove and change to SizeS
const Size{N} = NTuple{N,StaticInt}


abstract type MultilinearMap{Sz} end

# const MMap = MultilinearMap

# TODO docstring
@inline MultilinearMap{Sz}(f) where {Sz<:Size} =
    AtomicMultilinearMap{Sz}(f)

# For brevity, accept MultilinearMap{(M,N)}(...) and convert
# -> MultlinearMap{Tuple{StaticInt(M), StaticInt(N)}}(...)
@generated function MultilinearMap{Sz}(f) where {Sz}
    TupleDims = Tuple{map(i -> StaticInt{i}, Sz)...}
    :(MultilinearMap{$TupleDims}(f))
end

const MultilinearMapN{N} = MultilinearMap{<:NTuple{N,StaticInt}}
const MultilinearForm{N,D} = MultilinearMap{NTuple{N,StaticInt{D}}}
# const SingletonMap = MultilinearMap{Tuple{}}  # produces a single value


# Default Evaluation

const VecOrColon = Union{AbstractVector, Colon}

@inline (f::MultilinearMap{<:Size{N}})(::Vararg{Colon, N}) where N = f
@inline (f::MultilinearMap{<:Size{N}})(args::Vararg{VecOrColon, N}) where N =
    PartialMap(f, args...)


# Wraps a function of vectors
struct AtomicMultilinearMap{Sz<:Size, F} <: MultilinearMap{Sz}
    impl::F
    AtomicMultilinearMap{Sz}(impl::F) where {Sz,F} = new{Sz,F}(impl)
end

@inline (f::AtomicMultilinearMap{<:Size{0}})() = f.impl()  # disambiguate
@inline (f::AtomicMultilinearMap{<:Size{N}})(vs::Vararg{AbstractVector,N}) where N =
    f.impl(vs...)

function Base.show(io::IO, f::AtomicMultilinearMap)
    fn_name = Symbol(f.impl)
    print(io, "MultilinearMap{", size(f), "}(", fn_name, ")")
end

function Base.show(io::IO, ::MIME"text/plain", f::AtomicMultilinearMap)
    print(io, "MultilinearMap{", join(size(f), " × "), "} → ", eltype(f))
end
