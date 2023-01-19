"""
    LinearFunctionSpaces

A module that allows creating functions via linear combinations of other
functions.  For example

    using LinearFunctionSpaces

    f = Primative(x -> x^2 + 1)
    g = Primative(x -> x^3 / 2 + 1)
    h = 2(f + g) / 3

    h(4.2) = 2(f(4.2) + g(4.2)) / 3
"""
module LinearFunctionSpaces

export Primative, Const


# Specialized Scalars
# -------------------

"""
    SpecialScalar

! Intended for internal use !

Scaling of vectors by `SpecialScalar`s represents some special scaling
operation.  Namely, scaling by `One` and `NegOne` are the identity operation and
additive inversion, respecitvely.  `Recip` is for (lazily) scaling by the
reciprocal of a number, and is mostly for the performance consideration that
dividing by a number `x` is less expensive than computing `1/x` and then
multiplying.
"""
abstract type SpecialScalar end
# TODO: There is probably no harm in implementing the full Number interface

const Scalar = Union{SpecialScalar, Number}

struct One    <: SpecialScalar end
struct NegOne <: SpecialScalar end

(::Type{T})(::One)    where {T<:Number} = one(T)
(::Type{T})(::NegOne) where {T<:Number} = -one(T)

"""Lazy multiplicative inverse."""
struct Recip{T<:Scalar} <: SpecialScalar
    divisor::T
end

Recip{One}(::One) = One()
Recip{NegOne}(::NegOne) = NegOne()
Recip{Recip{T}}(r::Recip) where {T<:Number} = r.divisor
Recip{T}(r::Recip) where {T<:Number} = Recip{T}(r.divisor)

Recip(x::T) where {T<:Number} = Recip{T}(x)

# like in base/number.jl
(::Type{T})(r::Recip) where {T<:Number} = convert(T, inv(r.divisor))
Base.convert(::Type{T}, s::SpecialScalar) where {T<:Number} = T(s)

@inline Base.:+(s::SpecialScalar) = s           # identity operation
@inline Base.:-(::NegOne) = One()               # scaling by -1
@inline Base.:-(::One)    = NegOne()
@inline Base.:-(r::Recip) = Recip(-r.divisor)
# NOTE: we do not bother defining binary addition/subtraction since
# `SpecialScalar`s exist solely for "specializing" scalar multiplication.

@inline Base.:*(s::SpecialScalar, x::SpecialScalar) = _scale(s, x)
@inline Base.:*(s::SpecialScalar, x) = _scale(s, x)
@inline Base.:*(x, s::SpecialScalar) = _scale(s, x)

# Scalar division, multiplicative inverse
@inline Base.inv(s::Recip) = s.divisor
@inline Base.inv(s::Union{One,NegOne}) = s
@inline Base.:/(s::SpecialScalar, x::SpecialScalar) = _scale(inv(s), x)
@inline Base.:/(s::SpecialScalar, x) = _scale(inv(s), x)
@inline Base.:/(x, s::SpecialScalar) = _scale(inv(s), x)
# TODO: implement left division(\).
# TODO: implement (//)

# Maybe not needed if SpecialScalar <: Number
Base.:(==)(s::SpecialScalar, x::T) where {T<:Number} = x == convert(T, s)
Base.:(==)(x::T, s::SpecialScalar) where {T<:Number} = x == convert(T, s)
Base.:(<)(s::SpecialScalar, x::T) where {T<:Number} = x < convert(T, s)
Base.:(<)(x::T, s::SpecialScalar) where {T<:Number} = x < convert(T, s)
Base.:(<)(s::Recip, r::Recip) = s.divisor >= r.divisor

Base.abs(r::Recip) = inv(abs(r.divisor))

Base.adjoint(r::Recip{<:Real}) = r
Base.adjoint(r::Recip{C}) where {C<:Complex} = (adjoint ∘ C ∘ inv)(r.divisor)
# TODO: probably better to implement the full number interface at this point

"""
    _scale(s::Scalar, x)

Eagerly scale the "vector" `x` by the scalar quantity `a`, specializing on the
`SpecialScalar`s.  `x` may also be scalar.
"""
@inline _scale(::One, x) = +x                   # see below about using `+` here
@inline _scale(::NegOne, x) = -x
@inline _scale(r1::Recip, r2::Recip) = Recip(r1.divisor * r2.divisor)
@inline _scale(r::Recip, x) = x / r.divisor
@inline _scale(r::Recip, s::SpecialScalar) = _scale(s, r)
@inline _scale(a::Number, x) = a*x
@noinline _scale(::Scalar, x) = @assert false "unhandled case"
# NOTE: We consider something to be scalable by One() if it has a unary `+`
# method defined.  For example, it would be strange for _scale(::One, ::Char) to
# work, but we benefit from a method _scale(::One, ::AbstractVector) returning a
# scaled vector.  In any case, the `SpecialScalar`s are for internal use only so
# hopefully this works well enough.


# Linear Function Spaces
# ----------------------
#
# Consider a set of functions (a function space) whose elements all take as
# inputs the same kinds of arguments (i.e. they have similar function
# signatures) and, for all possible inputs, output elements of a common vector
# space.  Then, linear combinations of these functions are also members of the
# same function space.  Hence, we call these function spaces "linear function
# spaces", which are just a kind of abstract vector space.
#
# The purpose of this module is simply to give a way to easily construct linear
# combinations of such functions, given a "basis set" defined by the user.

"""
    Element <: Function

Abstract type representing an element of a linear function space (vector space).
"""
abstract type Element <: Function end

"""
    Element(f #= callable =#)

Produce a generic element of a linear function space.
"""
Element(f) = PrimativeWrapper(f)
"Identity operation on `ex`."
Element(ex::Element) = ex

"""
    Primative <: Element

Represents a primative element of a linear function space.  Usually, this is a
function or callable object wrapped in an containter type that subtypes
`Primative`.
"""
abstract type Primative <: Element end

"""
    Primative(f)

Wraps a callable `f` using `PrimativeWrapper`, making it a generic, "primative"
element of a linear function space.
"""
Primative(ex::Primative) = ex
Primative(f #= callable =#) = PrimativeWrapper(f)
@noinline Primative(::Element) = throw(ArgumentError("Not a primative element"))

"""
    Const{T} <: Primative
    Const{T}(val)
    Const(val)

Represent a constant of value `val`.  When called as a function, satisfies
    Const(val)(args...) === val
"""
struct Const{T} <: Primative
    val::T
end

"Wrapper to indicate that the function is an element of a function space."
struct PrimativeWrapper{F} <: Primative
    func::F  # some callable object
end

PrimativeWrapper(f::Element) = f

"Addition of function space elements."
struct Add{S<:Element, T<:Element} <: Element
    x::S
    y::T
    Add(x::S, y::T) where {S<:Element, T<:Element} = new{S,T}(x,y)
end

"""
    Scale{S<:Scalar,T<:Element} <: Element

Represents a scaled element `T`.
"""
struct Scale{S<:Scalar, T<:Element} <: Element
    a::S  # coefficent
    x::T  # callable object
    Scale{S}(a, x::T) where {S<:Scalar, T<:Element} = new{S,T}(a,x)
end

Scale{One}(::One, x::Element) = x
Scale{S}(b, (;a,x)::Scale) where {S<:Scalar} = Scale{S}(b*a, x)
Scale(b::Scalar, (;a,x)::Scale) = Scale(b*a, x)
Scale(a::S, x::T) where {S<:Scalar, T<:Element} = Scale{S}(a,x)

const Neg{T} = Scale{NegOne,T} where {T<:Element}
const Sub{S,T} = Add{S,Neg{T}} where {S<:Element, T<:Element}

Base.:+(x::Element) = x
Base.:+(x::Element, y::Element) = Add(x, y)
Base.:-(x::Element) = Scale(-One(), x)
Base.:-(x::Element, y::Element) = Add(x, -y)
Base.:*(a, x::Element) = Scale(a, x)
Base.:*(x::Element, a) = Scale(a, x)
Base.:*(::Element, ::Element) =
    throw(ArgumentError("multiplication of two vector space elements is undefined."))
Base.:/(x::Element, a) = Scale(Recip(a), x)
# Base.:\(a, x::Element) = Scale(Recip(a), x)   # XXX: needed if we have `/`?
Base.://(x::Element, a::Int) = Scale(1//a, x)

# Evaluation of concrete `Element`s
@inline (c::Const)(args...) = c.val
@inline (pw::PrimativeWrapper)(args...) = pw.func(args...)
@inline ((;x, y)::Add)(args...) = x(args...) + y(args...)
@inline ((;a, x)::Scale)(args...) = a * x(args...)


# Pretty printing
# ---------------

parenthesized(x) = "(" * string(x) * ")"
parenthesized_iftype(T::Type, x) =
    (x isa T ? parenthesized : identity)(x)

Base.show(io::IO, c::Const) = print(io, Const, parenthesized(c.val))

Base.show(io::IO, wf::PrimativeWrapper) =
    print(io, Primative, parenthesized(wf.func))

Base.show(io::IO, (;x, y)::Sub) =
    print(io, x, " - ", y.x)
Base.show(io::IO, (;x, y)::Add) =
    print(io, parenthesized_iftype(Add, x), " + ", y)

Base.show(io::IO, (;a, x)::Scale) =
    print(io, parenthesized_iftype(Rational, a), "*",
          parenthesized_iftype(Add, x))
Base.show(io::IO, (;a, x)::Neg) =
    print(io, "-", parenthesized_iftype(Add, x))
Base.show(io::IO, (;a, x)::Scale{<:Recip}) =
    print(io, parenthesized_iftype(Add, x), "/", a.divisor)

# Override printing methods in Base for `Function`
Base.show(io::IO, ::MIME"text/plain", ex::Element) = show(io, ex)
Base.print(io::IO, ex::Element) = Base.@invoke Base.print(io::IO, ex::Any)

end # module
