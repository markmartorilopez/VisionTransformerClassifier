import numpy as np
import timm
import torch
from model import VisionTransformer

# Helpers
def get_n_params(module):
    """ Count the number of learnable parameters """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    """ Compare if 2 tensor are equal """
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)


"""
Take timm package from Github Repo and load a pre-trained visual transformer model.
"""
model_name = "vit_base_patch16_224" # "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
print(type(model_official))

""" 
Hyperparameters corresponding to pretrained model
"""
custom_config = {
        "img_size": 384,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

# Instantiate the Vision Transformer model that we implemented. And evaluate it.
model_custom = VisionTransformer(**custom_config)
model_custom.eval()

"""
Iterate and compare official network params. and our custom network parameters.

For each parameter we check whether the num of elements is equal.

"""
for (n_o, p_o), (n_c, p_c) in zip(
        model_official.named_parameters(), model_custom.named_parameters()
):
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")

    p_c.data[:] = p_o.data # equal the custom param to the original param

    assert_tensors_equal(p_c.data, p_o.data)

inp = torch.rand(1, 3, 384, 384)
res_c = model_custom(inp)
res_o = model_official(inp)

# Asserts
assert get_n_params(model_custom) == get_n_params(model_official) # Num of trainable params is the same?
assert_tensors_equal(res_c, res_o)

# Save custom model
torch.save(model_custom, "model.pth") # save models with correct weights.
torch.save(model_official, "of_model.pth")