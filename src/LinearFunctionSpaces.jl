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

const Scalar = Union{SpecialScalar, Number}

struct One    <: SpecialScalar end
struct NegOne <: SpecialScalar end

"""Lazy multiplicative inverse."""
struct Recip{T<:Scalar} <: SpecialScalar
    divisor::T
end

Recip{One}(::One) = One()
Recip{NegOne}(::NegOne) = NegOne()
Recip{Recip{T}}(r::Recip) where {T<:Number} = r.divisor
Recip{T}(r::Recip) where {T<:Number} = Recip{T}(r.divisor)

Recip(x::T) where {T<:Number} = Recip{T}(x)

# Handle conversion
(::Type{T})(::One)    where {T<:Number} = one(T)
(::Type{T})(::NegOne) where {T<:Number} = -one(T)
(::Type{T})(r::Recip) where {T<:Number} = convert(T, inv(r.divisor))

# like in base/number.jl
Base.convert(::Type{T}, s::SpecialScalar) where {T<:Number} = T(s)

"""
    _scale(s::Scalar, x)

Internal use. Eagerly scale `x` by `a`, accounting for the action of scaling
by the `SpecialScalar`s.
"""
@inline _scale(::One, x) = +x                   # see below about using `+` here
@inline _scale(::NegOne, x) = -x
@inline _scale(r::Recip, x) = x / r.divisor
@inline _scale(a::Scalar, x) = a*x
# NOTE: We consider something to be scalable by One() if it has a unary `+`
# method defined.  For example, it would be strange for _scale(::One, ::Char) to
# work, but we benefit from a method _scale(::One, ::AbstractVector) returning a
# scaled vector.  In any case, the `SpecialScalar`s are for internal use only so
# hopefully this works well enough.

@inline Base.:+(s::SpecialScalar) = s
@inline Base.:-(::NegOne) = One()
@inline Base.:-(::One) = NegOne()

@inline Base.inv(s::Recip) = s.divisor
@inline Base.inv(s::Union{One,NegOne}) = s

@inline Base.:*(s::SpecialScalar, x::SpecialScalar) = _scale(s, x)
@inline Base.:*(s::SpecialScalar, x) = _scale(s, x)
@inline Base.:*(x, s::SpecialScalar) = _scale(s, x)

@inline Base.:/(s::SpecialScalar, x::SpecialScalar) = _scale(inv(s), x)
@inline Base.:/(s::SpecialScalar, x) = _scale(inv(s), x)
@inline Base.:/(x, s::SpecialScalar) = _scale(inv(s), x)

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
    callable::F  # some callable object
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
Base.:/(x::Element, a) = Scale(Recip(a), x)
Base.:\(a, x::Element) = Scale(Recip(a), x)
Base.://(x::Element, a::Int) = Scale(1//a, x)

# Evaluation of concrete `Element`s
@inline (c::Const)(args...) = c.val
@inline (pw::PrimativeWrapper)(args...) = pw.callable(args...)
@inline ((;x, y)::Add)(args...) = x(args...) + y(args...)
@inline ((;a, x)::Scale)(args...) = a * x(args...)


# Pretty printing
# ---------------

parenthesized(x) = "(" * string(x) * ")"
parenthesized_iftype(T::Type, x) =
    (x isa T ? parenthesized : identity)(x)

Base.show(io::IO, c::Const) = print(io, Const, parenthesized(c.val))

Base.show(io::IO, wf::PrimativeWrapper) =
    print(io, Primative, parenthesized(wf.callable))

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

Base.show(io::IO, ::MIME"text/plain", ex::Element) = show(io, ex)

Base.print(io::IO, ex::Element) = Base.@invoke Base.print(io::IO, ex::Any)

end # module

# using .LinearFunctionSpaces
# const LFS = LinearFunctionSpaces

# f(x) = x + 1
# g(x) = x^2 - x
# h(x) = x^3/3 + x^2/2

# wf, wg, wh = map(LFS.Element, (f,g,h))
