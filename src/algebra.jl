#
# Algebra
#
# Multilinear maps form a vector space.  That is, we can take linear
# combinations of multilinear maps and generally produce another multilinear
# map.  Here, we define the necessary operations `ScalarMultiple` and `Sum`.
#
# TODO: tensor products, contractions?
#


"""
    ScalarMultiple(f::MultilinearMap, s::Number)

Represents a scalar product on a `MultilinearMap`.
"""
struct ScalarMultiple{Sz<:Size, T, MM<:MultilinearMap{Sz}, S<:Number} <: MultilinearMap{Sz,T}
    parent::MM
    scalar::S
    ScalarMultiple(f::MultilinearMap{Sz,T}, s::S) where {Sz<:Size, T, S<:Number} =
        # TODO: promote type...
        new{Sz, promote_type(T, S), typeof(f), S}(f, s)
end

# Prevent nesting of `ScalarMultiple`s (associvity)
@inline ScalarMultiple(f::ScalarMultiple, a::Number) =
    ScalarMultiple(f.parent, f.scalar * a)

@inline operands(::Type{<:ScalarMultiple}, f::ScalarMultiple) = (f.parent, f.scalar)

# Evaluation
@inline (f::ScalarMultiple)(args::Vararg{VecOrColon}) = f.scalar * f.parent(args...)

# Construction by scalar multiplication
@inline Base.:*(f::MultilinearMap, a::Number) = ScalarMultiple(f, a)
@inline Base.:*(a::Number, f::MultilinearMap) = ScalarMultiple(f, a)

# Division
@inline Base.:/(f::MultilinearMap, a::Number) = ScalarMultiple(f, inv(a))
@inline Base.://(f::MultilinearMap, a::Number) = ScalarMultiple(f, one(a) // a)

Base.show(io::IO, f::ScalarMultiple) = print(io, f.scalar, " * ", f.parent)


"""
    Sum(fs::MultilinearMap...)

Represents a sum of `MultilinearMaps`, all of which must share a
common size.
"""
struct Sum{Sz<:Size, T, MMs<:TupleN{MultilinearMap{Sz}}} <: MultilinearMap{Sz,T}
    operands::MMs
    function Sum(fs::Vararg{MultilinearMap})
        sz = samesize(fs...)
        args1 = map(dim -> StdUnitVector{known(dim)}(1), sz)
        T = typeof(_eval_sum(fs, args1))                 # determine output type
        new{typeof(sz), T, typeof(fs)}(fs)
    end
end
@inline Sum(f::MultilinearMap, fsum::Sum) = Sum(f, fsum.operands...)
@inline Sum(fsum::Sum, f::MultilinearMap) = Sum(fsum.operands..., f)

@inline _eval_sum(operands::TupleN{MultilinearMap{<:Size}},
                  args::TupleN{VecOrColon}) =
    mapreduce(apply(args), +, operands)


# Evaluation
@inline (sumf::Sum)(args::Vararg{VecOrColon,N}) where N =
    mapreduce(apply(args), +, sumf.operands)
# XXX slow.  Try forcing specialization on argument length

@inline Base.:-(f::MultilinearMap) = -one(eltype(f)) * f
@inline Base.:+(f1::MultilinearMap, f2::MultilinearMap) = Sum(f1, f2)
@inline Base.:-(f1::MultilinearMap, f2::MultilinearMap) = f1 + -(f2)


function Base.show(io::IO, fsum::Sum)
    print(io, join(fsum.operands, " + "))
end


