module FixedFunctions

struct Slot end

@inline _evalfunction(f::Function, args::Tuple) = f(args...)

"""
    Fix(f::Function, args::NTuple{N,Any}) where N

A type representing a lazy application of an `N`-argument function `f`.
Arguments `args` that are `Slot()`s indicate the yet-to-be applied arguments
of the function.

Calling an instance of `Fix` applies its arguments to the `Slot()`s from
left-to-right.  If none of the arguments are `Slot()`s, the wrapped function is
evaluated.  Otherwise, a new `Fix` object with equal or lower arity is created.

`Fix` is similar to but generalizes `Base.Fix1` and `Base.Fix2`.
"""
struct Fix{F<:Function, T<:Tuple} <: Function
    func::F
    args::T
    Fix(f, args) = new{typeof(f), typeof(args)}(f, args)
end

Fix(f::Fix, args) = Fix(f.func, _reapplyargs(f.args, args))

(f::Fix)(args...) = apply(f.func, _reapplyargs(f.args, args)...)

@inline apply(f, args...) = _eval_or_Fix(args)(f, args)

# Based on whether Slot()'s are present in the arguments, decide whether to
# construct a Fix instance or to simply apply and evaluate the function
@inline _eval_or_Fix(::Tuple{}) = _evalfunction
@inline _eval_or_Fix(::Tuple{Slot,Vararg}) = Fix
@inline _eval_or_Fix((_, args...)::Tuple) = _eval_or_Fix(args)

($)(f, args) = apply(f, args...)

# Returns the set of arguments to be reapplied to a function that has been
# `Fix`ed.  The first argument is a tuple of arguments, presumably containing
# some `Slot()`s, and the second argument is a tuple of arguments to fill the
# `Slot()`s from left to right.  Slots must be filled, but may potentially be
# replaced with another `Slot()`.
#
@inline _reapplyargs(::Tuple{}, ::Tuple{}) = ()

@noinline _reapplyargs(args::Tuple, ::Tuple{}) =
    throw(ArgumentError("$(_slotcount(args)) too few arguments applied"))
# NOTE: this method would allow for slots to be left unfilled
# @inline _reapplyargs(args::Tuple, ::Tuple{}) = args
#
@noinline _reapplyargs(::Tuple{}, args::Tuple) =
    throw(ArgumentError("$(length(args)) too many arguments applied"))
# NOTE: this method would allow extra positional arguments
# @inline _reapplyargs(::Tuple{}, args::Tuple) = args
#
@inline _reapplyargs((_, fixedargs...)::Tuple{Slot, Vararg}, (newarg, args...)::Tuple) =
    (newarg, _reapplyargs(fixedargs, args)...)
@inline _reapplyargs((oldarg, fixedargs...)::Tuple, args::Tuple) =
    (oldarg, _reapplyargs(fixedargs, args)...)


@inline _slotcount(::Tuple{}, n=0) = n
@inline _slotcount((arg, args...)::Tuple, n=0) =
    _slotcount(args, arg isa Slot ? n+1 : n)

slotcount(f::Fix) = _slotcount(f.args)


Base.show(io::IO, f::Fix) = print(io, Fix, (f.func, f.args))

# Override printing methods in Base for `Function`
Base.show(io::IO, ::MIME"text/plain", ex::Fix) = show(io, ex)
Base.print(io::IO, ex::Fix) = Base.@invoke Base.print(io::IO, ex::Any)

# TODO Extended function composition
# ----------------------------------

# Since functions wrapped by `Fix` have a fixed arity, we can define generalized
# function composition notation.  Say the arity of f = is n and the arity of gᵢ
# for i = 1, ⋯, n is a₁, ⋯, aₙ
#
#   h = (f::Fix) ∘ (g₁::Fix, ⋯, gₙ::Fix) =
#       f(g₁(x₁,₁ ⋯, x₁,ₐ₁), ⋯, gₙ(xₙ,₁, ⋯, xₙ,ₐₙ))
#
# which gives a function `h` of arity a₁ + ⋯ + aₙ.
# To compose at some argument slots but not others, one could simply set any
# number of the gᵢ's essentially be `identity` functions, or perhaps an
# `Idnetity <: AbstractFix` singleton type with unit arity.

end # module
