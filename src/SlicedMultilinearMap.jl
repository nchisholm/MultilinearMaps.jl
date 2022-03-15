# Represent a partially contracted form
struct SlicedMultilinearMap{Sz<:StaticSize, M<:AtomicMultilinearMap, TT<:Tuple} <: MultilinearMap{Sz}
    parent::M
    args::TT
    function SlicedMultilinearMap(parent::M, args::TT) where {M, TT}
        Sz′ = _contracted_size(M, TT)
        new{Sz′, M, TT}(parent, args)
    end
end

@generated function _contracted_size(::Type{MM}, ::Type{ArgsType}) where {MM<:MultilinearMap, ArgsType<:Tuple}
    Sz = ArrayInterface.known_size(MM)
    ArgTs = fieldtypes(ArgsType)
    @assert length(Sz) == length(ArgTs)
    Tuple{[StaticInt{dimlen} for (dimlen, ArgT) ∈ zip(Sz, ArgTs) if ArgT === Colon]...}
end

function _contracted_args(T::Type{<:SlicedMultilinearMap})
    # We need to intercalate the "concrete" parent arguments with the "free
    # arguments" of the contracted form.
    #
    # Here, we let "vs" be the free arguments and "us" be the parent arguments.
    parent_argTs = fieldtypes(fieldtype(T, :args))
    j = 0
    [parent_argTs[i] === Colon ? :(vs[$(j += 1)]) : :(cmf.args[$i])
     for i ∈ eachindex(parent_argTs)]
end

# Use @_inline_meta?
@generated function (cmf::SlicedMultilinearMap{<:StaticSize{N}})(vs::Vararg{AbstractVector, N}) where N
    :(cmf.parent.impl($(_contracted_args(cmf)...)))
end

# Call that produces a "contracted" form
@generated function (f::MultilinearMapN{N})(args::Vararg{Union{Colon,AbstractVector}, N}) where N
    # "Dispatch" is controled by where `:`s appear in `args`.
    N′ = count(arg -> arg === Colon, args)
    if N′ == N  # All arguments are (:), so this is an identity operation
        :(f)
    elseif N′ < N
        :(SlicedMultilinearMap(f, args))
    else # should never happen
        :(@assert false)
    end
end
