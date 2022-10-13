module Special

using Base: tail
using ..MultilinearMaps: StdUnitVector, samelength
using LinearAlgebra
using Static
import ArrayInterface as Arr

export wedge

const TupleN{T,N} = NTuple{N,T}
const CanonicalInt = Union{StaticInt, Int}

diag(vs...) = mapreduce(*, +, vs...)

kronecker_δ(u, v) = diag(u, v)
kronecker_δ(u::AbstractVector, v::AbstractVector) = u ⋅ v


# Like `ntuple`, but uses `StaticInt`s
sntuple(f, n::StaticInt) = _sntuple(f, n, n)
_sntuple(_, ::StaticInt, ::StaticInt{0}) = ()
_sntuple(f, n::StaticInt, i::StaticInt) =
    (f(n - i + static(1)), _sntuple(f, n, i - static(1))...)

"""
    altsigns(init)

Returns an iterator that alternates `init` and `-init`.

    altsigns(T::Type)

Uses `oneunit(T)` as the initial values to `altsigns`.

```julia
julia> using Static

julia> NTuple{5,Any}(altsigns(StaticInt))
(static(1), static(-1), static(1), static(-1), static(1))
```
"""
altsigns(init) = Iterators.cycle((init, -init))
altsigns(T::Type) = altsigns(oneunit(T))

"""
    swap_1st(i::StaticInt, tup::Tuple)

Return the input tuple with the element at the index `i` swapped with the first
element.

```julia
julia> swap_1st(static(2), ('a', 'b', 'c'))
('b', 'a', 'c')
```
"""
function swap_1st(i::StaticInt, tup::Tuple)
    N = Arr.static_length(tup)
    1 ≤ i ≤ N || return tup
    (tup[i], sntuple(j -> tup[j], i - static(1))...,
             sntuple(j -> tup[j+i], N - i)...)
end

"""
    swapeach_1st(tup::Tuple)

Return a tuple of tuples where the ith tuple swaps the 1st and i-th elements
of the input tuple.

```julia
julia> swapeach_1st(('a', 'b', 'c'))
(('a', 'b', 'c'), ('b', 'a', 'c'), ('c', 'a', 'b'))
```
"""
swapeach_1st(tup::NTuple{N,Any}) where {N} =
    sntuple(i -> swap_1st(i, tup), static(N))

@inline _findfirst(_, ::Tuple{}, _) = nothing
@inline _findfirst(p, (x, xs...)::Tuple, i=1) =
    p(x) ? i : _findfirst(p, xs, i + 1)

@inline deleteat(::Tuple{}, i) = error("index out of bounds")
@inline deleteat((x, xs...)::Tuple, i=1) =
    i == 1 ? xs : (x, deleteat(xs, i-1)...)

@inline function wedge(vs...)
    # Length of each vector must match number vectors passed as arguments
    N = Arr.length(vs)
    L = samelength(vs...)
    N == L || _wedge_arg_err(N, L)
    return _wedge(ntuple(identity, N), vs...)
end

@noinline function _wedge_arg_err(N, L)
    msg = "Length of each input vector $L must match the number of " *
        "input vectors, but $N vectors were supplied."
    throw(ArgumentError(msg))
end

@inline _wedge(inds::Tuple, v) = #=@inbounds=# getindex(v, only(inds))

# TODO: this algorithm doesn't work too well if unit vectors aren't the first
# arguments
@inline function _wedge(inds::Tuple, e::StdUnitVector, v, vs...)
    k = _findfirst(==(e.dimension), inds)
    k === nothing && return false  # FIXME: "fast exit" causes type instability
    sgn = ifelse(isodd(k), +, -)
    # flipsign(_wedge(deleteat(inds, k), v, vs...), sgn)
    sgn(_wedge(deleteat(inds, k), v, vs...))
end

@inline function _wedge(inds::Tuple, v, vs...)
    L = length(inds)
    @assert L == length(vs) + 1
    sgns = NTuple{L,Int}(altsigns(Int))
    terms = map((sgn, (i, is′...)) -> flipsign(v[i], sgn) * _wedge(is′, vs...),
                sgns, swapeach_1st(inds))
    +(terms...)
end

end # module
