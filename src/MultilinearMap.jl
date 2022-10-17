import ArrayInterface as Arr

using .FixedFunctions: Fix, Slot

export MultilinearMap

const TupleN{T,N} = NTuple{N,T}
const VecOrColon = Union{AbstractVector, Colon}

# abstract type AbstractMultilinearMap{N} end

"""
Wraps a function or callable value `impl` that should have a method
`impl(::Vararg{AbstractVector,N})`, which is assumed to be multilinear in all of
its arguments, i.e., linear when all of the arguments but one argument is held
fixed.
"""
struct MultilinearMap{N,F}  # <: AbstractMultilinearMap{N}
    impl::F
    size::Dims{N}
    # NOTE: only allow fully static sizes for now
    function MultilinearMap(impl::F, dims::NTuple{N,Integer}) where {N, F}
        # M = slotcount(impl)
        # M == N || throw(DimensionMismatch("Number of dimensions $N does not match arity $M of function"))
        all(≥(0), dims) || throw(DomainError(dims, "Dimension lengths must be nonnegative integers"))
        new{N,F}(impl, dims)
    end
end
# TODO: maybe allow one or more of the size to be `nothing`, and allow the map
# to represent an array of arbitrary size the `nothing` dimensions

MultilinearMap(f::MultilinearMap, dims=size(f)) = MultilinearMap(f.impl, dims)
MultilinearMap(impl, dims::Integer...) = MultilinearMap(impl, dims)
# MultilinearMap(impl::Function, dims::TupleN{Integer}) =
#     MultilinearMap(Fix(impl, map(_ -> Slot(), dims)), dims)

Base.IteratorSize(::Type{MultilinearMap{N}}) where N = Base.HasShape{N}()
Base.IteratorEltype(::Type{MultilinearMap}) = Base.EltypeUnknown()
Base.IndexStyle(::Type{<:MultilinearMap}) = IndexCartesian()
Base.ndims(::Type{<:MultilinearMap{N}}) where N = N
Base.ndims(f::MultilinearMap) = Base.ndims(typeof(f))
Base.size(f::MultilinearMap) = f.size
Base.size(f::MultilinearMap, dim) = size(f)[dim]
Base.length(f::MultilinearMap) = prod(size(f))

_eltype(f::MultilinearMap{N}) where N =
    Base.promote_op(f.impl, ntuple(_ -> StdUnitVec, Val(N))...)

Base.@propagate_inbounds function Base.getindex(f::MultilinearMap, I::Vararg{Int,N}) where N
    es = ntuple(k -> StdUnitVec(size(f,k), I[k]), Val(N))  # basis vectors
    f(es...)
end

(f::MultilinearMap{N})(args::Vararg{AbstractVector,N}) where N = f.impl(args...)
(f::MultilinearMap{N})(::Vararg{Colon,N}) where N = f   # identity op

function (f::MultilinearMap{N})(args::Vararg{VecOrColon,N}) where N
    size1 = _appliedsize(f, args)
    args1 = _colons_to_slots(args)
    f1 = Fix(f.impl, args1)
    @assert 0 < length(size1) < N
    MultilinearMap(f1, size1)
end

@inline _appliedsize(f::MultilinearMap, args::Tuple) =
    _appliedsize(size(f), args)
@inline _appliedsize(::Tuple{}, ::Tuple{}) = ()
@inline _appliedsize((dim, sz...)::Dims{N}, (arg, args...)::NTuple{N,Any}) where N =
    arg isa Colon ? (dim, _appliedsize(sz, args)...) : _appliedsize(sz, args)
# Do we need to specialize on N or will that create unnecessary overhead?

@inline _colons_to_slots(::Tuple{}) = ()
@inline _colons_to_slots((arg, args...)::Tuple) =
    (arg isa Colon ? Slot() : arg, _colons_to_slots(args)...)
    #    ^ should we use `isa` here or rely on dispatch?
