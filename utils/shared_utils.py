def untuple(x):
    return x[0] if isinstance(x, tuple) else x


def make_hook(states_dict, layer_name):
    def hook(module, input, output):
        if "attn_head" in layer_name:
            states_dict[layer_name] = input
        else:
            states_dict[layer_name] = output
    return hook


def remove_all_hooks(model):
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
