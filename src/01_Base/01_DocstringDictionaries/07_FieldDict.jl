# `field_dict`: the prose of every `arg_dict` entry, without its "`name`: " prefix. It loads
# after the last `arg_dict` file, so it reads the whole table.
for (key, val) in arg_dict
    field_dict[key] = strip(@view(val[(findfirst(":", val)[1] + 1):end]))
end
