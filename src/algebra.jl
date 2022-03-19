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
struct ScalarMultiple{Sz<:Size, MM<:MultilinearMap{Sz}, T<:Number} <: MultilinearMap{Sz}
    parent::MM
    scalar::T
    ScalarMultiple(f::MultilinearMap{Sz}, s::Number) where {Sz<:Size} =
        new{Sz,typeof(f),typeof(s)}(f, s)
end

# Prevent nesting of `ScalarMultiple`s (associvity)
@inline ScalarMultiple(f::ScalarMultiple, a::Number) =
    ScalarMultiple(f.parent, f.scalar * a)

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
struct Sum{Sz<:Size,MMs<:TupleN{MultilinearMap{Sz}}} <: MultilinearMap{Sz}
    operands::MMs
    Sum(fs::Vararg{MultilinearMap{Sz}}) where Sz = new{Sz, typeof(fs)}(fs)
end
@inline Sum(f::MultilinearMap, fsum::Sum) = Sum(f, fsum.operands...)
@inline Sum(fsum::Sum, f::MultilinearMap) = Sum(fsum.operands..., f)

# Evaluation
@inline (sumf::Sum)(args::Vararg{VecOrColon}) =
    mapreduce(apply(args), +, sumf.operands)

@inline Base.:-(f::MultilinearMap) = -one(eltype(f)) * f
@inline Base.:+(f1::MultilinearMap, f2::MultilinearMap) = Sum(f1, f2)
@inline Base.:-(f1::MultilinearMap, f2::MultilinearMap) = f1 + -(f2)


function Base.show(io::IO, fsum::Sum)
    print(io, join(fsum.operands, " + "))
end



