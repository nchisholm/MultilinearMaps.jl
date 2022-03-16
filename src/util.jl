using Base.Cartesian


const TupleN{T,N} = NTuple{N,T}

# Concatenate ("splat") n tuples into one big tuple
@inline tuplecat() = ()
@inline tuplecat(t1::Tuple) = t1
@inline tuplecat(t0::Tuple, ts::Tuple...) = (t0..., tuplecat(ts)...)

# Return the types contained in a tuple
# fieldtypes itself is not type stable
@generated tupletypes(::Type{T}) where {T<:Tuple} = fieldtypes(T)
@generated tuplecount(::Type{T}) where {T<:Tuple} = fieldcount(T)
@inline tupletype(::Type{T}, i) where {T<:Tuple} = fieldtype(T, i)


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
