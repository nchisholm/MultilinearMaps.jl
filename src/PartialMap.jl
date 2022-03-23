# Represent a partially contracted form by wrapping another MultilinearMap of a
# "larger" size.
struct PartialMap{Sz<:Size, T,
                  MM<:MultilinearMap{<:Size, T},
                  TT<:Tuple} <: MultilinearMap{Sz,T}
    parent::MM
    args::TT
    function PartialMap(parent::MultilinearMap{Sz, T},
                        args::Vararg{Union{Colon,AbstractVector}}
                        ) where {Sz<:Size, T}
        MM = typeof(parent)
        AT = typeof(args)
        Sz′ = _contract_size(MM, AT)
        # FIXME: compute new output type
        new{Sz′, T #=XXX wrong=#, MM, AT}(parent, args)
    end
end

@generated function _contract_size(::Type{MM}, ::Type{ArgsType}) where {MM<:MultilinearMap, ArgsType<:Tuple}
    Sz = Arr.known_size(MM)
    ArgTs = fieldtypes(ArgsType)
    @assert length(Sz) == length(ArgTs)
    Tuple{[StaticInt{dimlen} for (dimlen, ArgT) ∈ zip(Sz, ArgTs) if ArgT === Colon]...}
end

# Use @_inline_meta?
@generated function (f::PartialMap{<:Size{N}})(vs::Vararg{AbstractVector, N}) where N
    # We need to intercalate the "concrete" parent arguments with the "free
    # arguments" of the contracted form.
    #
    # Here, we let "vs" be the free arguments and "us" be the parent arguments.
    parent_argTs = fieldtypes(fieldtype(f, :args))
    j = 0
    vs′ = [parent_argTs[i] === Colon ? :(vs[$(j += 1)]) : :(f.args[$i])
           for i ∈ eachindex(parent_argTs)]
    return :(f.parent.impl($(vs′...)))
end

function Base.show(io::IO, f::PartialMap)
    pp_args = "(" * join(map(arg -> arg isa Colon ? ":" : arg, f.args), ", ") * ")"
    print(io, f.parent, pp_args)
end
