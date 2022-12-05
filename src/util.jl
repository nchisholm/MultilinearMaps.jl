const TupleN{T,N} = NTuple{N,T}

const Dim = Union{StaticInt, Int}
const SDim = StaticInt
const Dims = TupleN{Dim}
const SDims = TupleN{SDim}

const SOneTo{L} = Arr.OptionallyStaticUnitRange{StaticInt{1}, StaticInt{L}}
const Axes{N} = NTuple{N,AbstractUnitRange}


# Concatenate ("splat") n tuples into one big tuple
@inline tuplejoin(t::Tuple) = joinargs(t...)
@inline tuplejoin_deep(t::Tuple) = joinargs_deep(t...)

@inline joinargs() = ()
@inline joinargs(a, rest...) = (a, joinargs(rest...)...)
@inline joinargs(a::Tuple, rest...) = (a..., joinargs(rest...)...)

@inline joinargs_deep() = ()
@inline joinargs_deep(a, rest...) = (a, joinargs_deep(rest...)...)
@inline joinargs_deep(t::Tuple, rest...) =
    (joinargs_deep(t...)..., joinargs_deep(rest...)...)

# Return the types contained in a tuple (fieldtypes itself is not type stable)
@generated tupletypes(::Type{T}) where {T<:Tuple} = fieldtypes(T)
@generated tuplecount(::Type{T}) where {T<:Tuple} = fieldcount(T)
@inline tupletype(::Type{T}, i) where {T<:Tuple} = fieldtype(T, i)

# From a given tuple, return the same tuple but with elements of a given type
# `T` removed.
@inline _removetype(::Type, ::Tuple{}) = ()
@inline _removetype(::Type{T}, (_, rest...)::Tuple{T,Vararg}) where T =
    _removetype(T, rest)
@inline _removetype(T::Type, (x, rest...)::Tuple) = (x, _removetype(T, rest)...)

_mlkernel(u, v) = one(eltype(u)) * one(eltype(v)) + one(eltype(u)) * one(eltype(v))
# NOTE: there is a function with this name in Base we could replace this with.
# @inline promote_eltype(Ts...) = promote_type(map(eltype, Ts)...)


const Maybe{T} = Union{T,Nothing}

# Apply f to arg, or return nothing if arg is nothing
@inline maybe(f, arg) = f(arg)
@inline maybe(_, ::Nothing) = nothing

# generalized version?
# @inline maybe(f, args...) = f(args...)
# @inline maybe(_, ::Vararg{Maybe}) = nothing

# @inline apply(args::Tuple, f::F) where F = f(args...)
# @inline apply(args::Tuple) = f -> apply(args, f)


"""
    samesize(as...)

Return the size of `as` if they all compare equal (==).  Otherwise throw a
`DimensionMismatch`. (The dynamic/static status of each dimension is not
considered.)
"""
@inline function samesize(as...)
    sz = _size(first(as))
    _sizes_match(sz, Base.tail(as)...) || _throw_size_mismatch(as...)
    return sz
end

@inline _sizes_match(sz0::Dims, szs::Dims...) = all(==(sz0), szs)
@inline _sizes_match(sz::Dims, as...) = _sizes_match(sz, map(_size, as)...)
@inline function _sizes_match(a, as...)
    sz = _size(a)
    _sizes_match(sz, as...)
end

# Can get rid of first method if ArrayInterface works around problems finding
# known sizes of container types
# TODO: but maybe consider avoiding trying to size UnionAll types
@inline _size(T::Type) = _determinant_size(T)
@inline _size(a) = Arr.size(a)

@inline _determinant_size(A) = _determinant_size(Arr.known_size(A))
@inline _determinant_size(sz::TupleN{Int}) = static(sz)
@noinline _determinant_size(sz::TupleN{Union{Int,Nothing}}) = throw(error(
    "Indeterminant size of type with known size $sz."
))

@noinline function _throw_size_mismatch(as...)
    sizes = map(dynamic ∘ _size, as)
    throw(DimensionMismatch("Sizes $sizes of inputs do not match"))
end

"""
    samelength(as...)

Return the size of `as` if they all compare equal (==).  Otherwise throw a
`DimensionMismatch`. (The dynamic/static status of each dimension is not
considered.)
"""
function samelength(as...)
    l0 = Arr.length(first(as))
    _lengths_match(l0, tail(as)...) || _throw_length_mismatch(as...)
    return l0
end

@inline _lengths_match(l0::Dim, ls::Dim...) = all(==(l0), ls)
@inline _lengths_match(l0::Dim, as...) =
    _lengths_match(l0, map(Arr.length, as)...)
@inline _lengths_match(a, as...) = _lengths_match(Arr.length(a), as...)

@noinline function _throw_length_mismatch(as...)
    ls = map(dynamic ∘ Arr.length, as)
    throw(DimensionMismatch("lengths $ls of inputs do not match"))
end

"""
Fill an `N`-dimensional array `arr` using a function `f` of the `N` indices.
"""
@generated function fillfn!(arr::AbstractArray{<:Any,N}, f::F) where {N,F}
    quote
        @nloops $N i arr begin
            @inbounds (@nref $N arr i) = (@ncall $N f i)
        end
        arr
    end
end

# Similar to the function above, but instead iterates over `CartesianIndices`.
# Its code is idomatic, but for some reason has poorer performance
# versus the "explicit looping" of the generated function above.
#
# https://github.com/JuliaArrays/StaticArrays.jl/issues/1010
#
"""
Fill an `N`-dimensional array `arr` using a function `f` of the `N` indices.
"""
function fillfn_cartesianindices!(arr, f::F) where F  # force specialization
    for I ∈ CartesianIndices(arr)
        @inbounds arr[I] = f(I)
    end
    return arr
end


# Produce the operands of a lazy operator
# Generically assume instances of P have a field called operands
@inline operands(::Type, op)               = (op,)
@inline operands(::Type{P}) where P        = op -> operands(P, op)
@inline operands(op::P) where P            = operands(P, op)


# Convert digits to subscripts, superscripts
subscript(d::Integer) = Char(0x2080) + _valid_scriptdigit(d)
subscripts(i::Integer) = join(subscript(d) for d ∈ reverse!(digits(i)))

function _valid_scriptdigit(d::Integer)
    0 ≤ d ≤ 9 || throw(DomainError(d, "Must be a digit between 0 and 9"))
    return d
end
