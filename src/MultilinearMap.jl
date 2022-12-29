using .FixedFunctions: Fix, Slot
using .LinearFunctionSpaces


const VecOrColon = Union{AbstractVector, Colon}

abstract type AbstractMultilinearMap{Sz<:Dims} end

"""
Wraps a function or callable value `func` that should have a method
`func(::Vararg{AbstractVector,N})`, which is assumed to be multilinear in all of
its arguments, i.e., linear when all of the arguments but one is held fixed.
"""
struct MultilinearMap{Sz<:Dims,F}  <: AbstractMultilinearMap{Sz}
    func::F
    dims::Sz
    function MultilinearMap(func::F, dims::Sz) where {Sz<:Dims, F}
        # T = Base.promote_op(f, ntuple(_ -> StdUnitVec, Val(N))...)
        # @assert isconcretetype(T)
        any(<(0), dims) && throw(ArgumentError("invalid dimensions"))
        new{Sz,F}(func, dims)
    end
end

MultilinearMap(f::MultilinearMap, dims=Arr.size(f)) =
    MultilinearMap(f.func, dims)
MultilinearMap(func, dims::Integer...) = MultilinearMap(func, dims)

arity(::Type{<:MultilinearMap{<:Dims{N}}}) where N = N
arity(f::MultilinearMap) = arity(typeof(f))

Base.@assume_effects :foldable Arr.axes_types(::Type{<:MultilinearMap{Sz}}) where {Sz<:Dims} =
    Tuple{map(T -> Static.OptionallyStaticUnitRange{StaticInt{1}, T}, fieldtypes(Sz))...}
Arr.axes(f::MultilinearMap) = map(i -> static(1):i, f.dims)
Arr.size(f::MultilinearMap) = f.dims
Arr.length(f::MultilinearMap) = prod(Arr.size(f))

Base.IteratorSize(::Type{F}) where {F<:MultilinearMap} =
    Base.HasShape{arity(F)}()
Base.IteratorEltype(::Type{<:MultilinearMap}) = Base.EltypeUnknown()
Base.IndexStyle(::Type{<:MultilinearMap}) = IndexCartesian()

Base.ndims(::Type{F}) where {F<:MultilinearMap} = arity(F)
Base.ndims(f::MultilinearMap) = arity(f)
Base.size(f::MultilinearMap) = dynamic(Arr.size(f))
# Base.size(f::MultilinearMap, dim) = dynamic(Arr.size(f, dim))
Base.length(f::MultilinearMap) = dynamic(Arr.length(f))

# NOTE: why doesn't ndims return static value in ArrayInterface?
# NOTE: why not pull from Base.HasShape for default ndims?

# Useful for initializing storage arrays
_eltype(f::MultilinearMap) = Base.promote_op(f, ntuple(_ -> StdUnitVec, Val(arity(f)))...)

Base.@propagate_inbounds function Base.getindex(f::MultilinearMap{<:Dims{N}}, I::Vararg{Int,N}) where N
    # TODO: handle colons
    f(map(StdUnitVec, I, Arr.size(f))...)
end

Base.@propagate_inbounds Base.getindex(f::MultilinearMap{<:Dims{N}}, I::CartesianIndex{N}) where N =
    f[Tuple(I)...]

@inline (f::MultilinearMap{<:Dims{0}})() = f.func()  # avoid ambiguity
@inline (f::MultilinearMap{<:Dims{N}})(args::Vararg{AbstractVector,N}) where N =
    f.func(args...)
@inline (f::MultilinearMap{<:Dims{N}})(::Vararg{Colon,N}) where N = f   # identity op

function (f::MultilinearMap{<:Dims{N}})(args::Vararg{VecOrColon,N}) where N
    size1 = _appliedsize(f, args)
    args1 = _colons_to_slots(args)
    f1 = Fix(f.func, args1)
    @assert 0 < length(size1) < N
    MultilinearMap(f1, size1)
end

@noinline (f::MultilinearMap)(args::Vararg{VecOrColon}) =
    throw(ArgumentError("$(length(args)) arguments provided to a MultilinearMap of arity $(arity(f))"))

@inline _appliedsize(f::MultilinearMap, args::Tuple) =
    _appliedsize(Arr.size(f), args)
@inline _appliedsize(::Tuple{}, ::Tuple{}) = ()
@inline _appliedsize((dim, sz...)::Dims{N}, (arg, args...)::NTuple{N,VecOrColon}) where N =
    arg isa Colon ? (dim, _appliedsize(sz, args)...) : _appliedsize(sz, args)
# Do we need to specialize on N or will that create unnecessary overhead?

@inline _colons_to_slots(::Tuple{}) = ()
@inline _colons_to_slots((arg, args...)::TupleN{VecOrColon}) =
    (arg isa Colon ? Slot() : arg, _colons_to_slots(args)...)
    #    ^ should we use `isa` here or rely on dispatch?

Base.CartesianIndices(f::MultilinearMap) = CartesianIndices(axes(f))
Base.firstindex(f::MultilinearMap) = first(CartesianIndices(f))
Base.lastindex(f::MultilinearMap) = last(CartesianIndices(f))

# NOTE inlining is important to performance here
@inline function Base.iterate(f::MultilinearMap)
    # Piggy-back on iterate(::CartesianIndices)
    (I, state) = iterate(CartesianIndices(f))
    return (f[I], state)
end

@inline function Base.iterate(f::MultilinearMap, state)
    maybe(((I′, state′),) -> (f[I′], state′),
          iterate(CartesianIndices(f), state))
end

Base.show(io::IO, f::MultilinearMap) =
    print(io, "MultilinearMap(", f.func, ", ", f.dims, ")")

Base.show(io::IO, ::MIME"text/plain", f::MultilinearMap) =
    print(io, join(f.dims, "×"), " MultilinearMap")
