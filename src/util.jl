using Base.Cartesian


const TupleN{T,N} = NTuple{N,T}

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


# NOTE: there is a function with this name in Base we could replace this with.
@inline promote_eltype(Ts...) = promote_type(map(eltype, Ts)...)


const Maybe{T} = Union{T,Nothing}

# Apply f to arg, or return nothing if arg is nothing
@inline maybe(f, arg) = f(arg)
@inline maybe(_, ::Nothing) = nothing

# generalized version?
# @inline maybe(f, args...) = f(args...)
# @inline maybe(_, ::Vararg{Maybe}) = nothing

@inline apply(args::Tuple, f::F) where F = f(args...)
@inline apply(args::Tuple) = f -> apply(args, f)


# Types to flag safe and unsafe methods
struct Safe end;   const SAFE = Safe()
struct Unsafe end; const UNSAFE = Unsafe()
# Methods marked `UNSAFE` may produce unpredictable behavior
const Safety = Union{Safe,Unsafe}

# Acts with @inbounds as a safety switch
@inline inbounds_safety() = (@boundscheck return SAFE; UNSAFE)


# Static or dynamic size
const Size = TupleN{Union{<:StaticInt,Int}}
# Competely static size
const SizeS = TupleN{StaticInt}
# Static size with all same dimensions
const CubeSize{N,D} = NTuple{N,StaticInt{D}}


"""
    samesize(as...)

Return the size of `as` if they all compare equal (==).  Otherwise throw a
`DimensionMismatch`. (The dynamic/static status of each dimension is not
considered.)
"""
@inline function samesize(as...)
    sz = _size(first(as))
    _sizes_match(sz, tail(as)...) || _throw_size_mismatch(as...)
    return sz
end

@inline _sizes_match(sz0::Size, szs::Size...) = all(==(sz0), szs)
@inline _sizes_match(sz::Size, as...) = _sizes_match(sz, map(_size, as)...)
@inline function _sizes_match(a, as...)
    sz = _size(a)
    _sizes_match(sz, as...)
end

# Can get rid of first method if ArrayInterface works around problems finding
# known sizes of container types
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
