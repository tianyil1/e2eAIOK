import torch
import json
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM

layer_dict = {
    "transformer.h.0.ln_1.weight":"transformer.h.0.ln_1.weight",
	"transformer.h.0.ln_1.bias":"transformer.h.0.ln_1.bias",
	"transformer.h.0.attn.bias":"transformer.h.0.attn.bias",
	"transformer.h.0.attn.masked_bias":"transformer.h.0.attn.masked_bias",
	"transformer.h.0.attn.c_attn.weight":"transformer.h.0.attn.c_attn.weight",
	"transformer.h.0.attn.c_attn.bias":"transformer.h.0.attn.c_attn.bias",
	"transformer.h.0.attn.c_proj.weight":"transformer.h.0.attn.c_proj.weight",
	"transformer.h.0.attn.c_proj.bias":"transformer.h.0.attn.c_proj.bias",
	"transformer.h.0.ln_2.weight":"transformer.h.0.ln_2.weight",
	"transformer.h.0.ln_2.bias":"transformer.h.0.ln_2.bias",
	"transformer.h.0.mlp.c_fc.weight":"transformer.h.0.mlp.c_fc.weight",
	"transformer.h.0.mlp.c_fc.bias":"transformer.h.0.mlp.c_fc.bias",
	"transformer.h.0.mlp.c_proj.weight":"transformer.h.0.mlp.c_proj.weight",
	"transformer.h.0.mlp.c_proj.bias":"transformer.h.0.mlp.c_proj.bias",
	"transformer.h.1.ln_1.weight":"transformer.h.2.ln_1.weight",
	"transformer.h.1.ln_1.bias":"transformer.h.2.ln_1.bias",
	"transformer.h.1.attn.bias":"transformer.h.2.attn.bias",
	"transformer.h.1.attn.masked_bias":"transformer.h.2.attn.masked_bias",
	"transformer.h.1.attn.c_attn.weight":"transformer.h.2.attn.c_attn.weight",
	"transformer.h.1.attn.c_attn.bias":"transformer.h.2.attn.c_attn.bias",
	"transformer.h.1.attn.c_proj.weight":"transformer.h.2.attn.c_proj.weight",
	"transformer.h.1.attn.c_proj.bias":"transformer.h.2.attn.c_proj.bias",
	"transformer.h.1.ln_2.weight":"transformer.h.2.ln_2.weight",
	"transformer.h.1.ln_2.bias":"transformer.h.2.ln_2.bias",
	"transformer.h.1.mlp.c_fc.weight":"transformer.h.2.mlp.c_fc.weight",
	"transformer.h.1.mlp.c_fc.bias":"transformer.h.2.mlp.c_fc.bias",
	"transformer.h.1.mlp.c_proj.weight":"transformer.h.2.mlp.c_proj.weight",
	"transformer.h.1.mlp.c_proj.bias":"transformer.h.2.mlp.c_proj.bias",
	"transformer.h.2.ln_1.weight":"transformer.h.4.ln_1.weight",
	"transformer.h.2.ln_1.bias":"transformer.h.4.ln_1.bias",
	"transformer.h.2.attn.bias":"transformer.h.4.attn.bias",
	"transformer.h.2.attn.masked_bias":"transformer.h.4.attn.masked_bias",
	"transformer.h.2.attn.c_attn.weight":"transformer.h.4.attn.c_attn.weight",
	"transformer.h.2.attn.c_attn.bias":"transformer.h.4.attn.c_attn.bias",
	"transformer.h.2.attn.c_proj.weight":"transformer.h.4.attn.c_proj.weight",
	"transformer.h.2.attn.c_proj.bias":"transformer.h.4.attn.c_proj.bias",
	"transformer.h.2.ln_2.weight":"transformer.h.4.ln_2.weight",
	"transformer.h.2.ln_2.bias":"transformer.h.4.ln_2.bias",
	"transformer.h.2.mlp.c_fc.weight":"transformer.h.4.mlp.c_fc.weight",
	"transformer.h.2.mlp.c_fc.bias":"transformer.h.4.mlp.c_fc.bias",
	"transformer.h.2.mlp.c_proj.weight":"transformer.h.4.mlp.c_proj.weight",
	"transformer.h.2.mlp.c_proj.bias":"transformer.h.4.mlp.c_proj.bias",
	"transformer.h.3.ln_1.weight":"transformer.h.6.ln_1.weight",
	"transformer.h.3.ln_1.bias":"transformer.h.6.ln_1.bias",
	"transformer.h.3.attn.bias":"transformer.h.6.attn.bias",
	"transformer.h.3.attn.masked_bias":"transformer.h.6.attn.masked_bias",
	"transformer.h.3.attn.c_attn.weight":"transformer.h.6.attn.c_attn.weight",
	"transformer.h.3.attn.c_attn.bias":"transformer.h.6.attn.c_attn.bias",
	"transformer.h.3.attn.c_proj.weight":"transformer.h.6.attn.c_proj.weight",
	"transformer.h.3.attn.c_proj.bias":"transformer.h.6.attn.c_proj.bias",
	"transformer.h.3.ln_2.weight":"transformer.h.6.ln_2.weight",
	"transformer.h.3.ln_2.bias":"transformer.h.6.ln_2.bias",
	"transformer.h.3.mlp.c_fc.weight":"transformer.h.6.mlp.c_fc.weight",
	"transformer.h.3.mlp.c_fc.bias":"transformer.h.6.mlp.c_fc.bias",
	"transformer.h.3.mlp.c_proj.weight":"transformer.h.6.mlp.c_proj.weight",
	"transformer.h.3.mlp.c_proj.bias":"transformer.h.6.mlp.c_proj.bias",
	"transformer.h.4.ln_1.weight":"transformer.h.8.ln_1.weight",
	"transformer.h.4.ln_1.bias":"transformer.h.8.ln_1.bias",
	"transformer.h.4.attn.bias":"transformer.h.8.attn.bias",
	"transformer.h.4.attn.masked_bias":"transformer.h.8.attn.masked_bias",
	"transformer.h.4.attn.c_attn.weight":"transformer.h.8.attn.c_attn.weight",
	"transformer.h.4.attn.c_attn.bias":"transformer.h.8.attn.c_attn.bias",
	"transformer.h.4.attn.c_proj.weight":"transformer.h.8.attn.c_proj.weight",
	"transformer.h.4.attn.c_proj.bias":"transformer.h.8.attn.c_proj.bias",
	"transformer.h.4.ln_2.weight":"transformer.h.8.ln_2.weight",
	"transformer.h.4.ln_2.bias":"transformer.h.8.ln_2.bias",
	"transformer.h.4.mlp.c_fc.weight":"transformer.h.8.mlp.c_fc.weight",
	"transformer.h.4.mlp.c_fc.bias":"transformer.h.8.mlp.c_fc.bias",
	"transformer.h.4.mlp.c_proj.weight":"transformer.h.8.mlp.c_proj.weight",
	"transformer.h.4.mlp.c_proj.bias":"transformer.h.8.mlp.c_proj.bias",
	"transformer.h.5.ln_1.weight":"transformer.h.9.ln_1.weight",
	"transformer.h.5.ln_1.bias":"transformer.h.9.ln_1.bias",
	"transformer.h.5.attn.bias":"transformer.h.9.attn.bias",
	"transformer.h.5.attn.masked_bias":"transformer.h.9.attn.masked_bias",
	"transformer.h.5.attn.c_attn.weight":"transformer.h.9.attn.c_attn.weight",
	"transformer.h.5.attn.c_attn.bias":"transformer.h.9.attn.c_attn.bias",
	"transformer.h.5.attn.c_proj.weight":"transformer.h.9.attn.c_proj.weight",
	"transformer.h.5.attn.c_proj.bias":"transformer.h.9.attn.c_proj.bias",
	"transformer.h.5.ln_2.weight":"transformer.h.9.ln_2.weight",
	"transformer.h.5.ln_2.bias":"transformer.h.9.ln_2.bias",
	"transformer.h.5.mlp.c_fc.weight":"transformer.h.9.mlp.c_fc.weight",
	"transformer.h.5.mlp.c_fc.bias":"transformer.h.9.mlp.c_fc.bias",
	"transformer.h.5.mlp.c_proj.weight":"transformer.h.9.mlp.c_proj.weight",
	"transformer.h.5.mlp.c_proj.bias":"transformer.h.9.mlp.c_proj.bias",
	"transformer.h.6.ln_1.weight":"transformer.h.10.ln_1.weight",
	"transformer.h.6.ln_1.bias":"transformer.h.10.ln_1.bias",
	"transformer.h.6.attn.bias":"transformer.h.10.attn.bias",
	"transformer.h.6.attn.masked_bias":"transformer.h.10.attn.masked_bias",
	"transformer.h.6.attn.c_attn.weight":"transformer.h.10.attn.c_attn.weight",
	"transformer.h.6.attn.c_attn.bias":"transformer.h.10.attn.c_attn.bias",
	"transformer.h.6.attn.c_proj.weight":"transformer.h.10.attn.c_proj.weight",
	"transformer.h.6.attn.c_proj.bias":"transformer.h.10.attn.c_proj.bias",
	"transformer.h.6.ln_2.weight":"transformer.h.10.ln_2.weight",
	"transformer.h.6.ln_2.bias":"transformer.h.10.ln_2.bias",
	"transformer.h.6.mlp.c_fc.weight":"transformer.h.10.mlp.c_fc.weight",
	"transformer.h.6.mlp.c_fc.bias":"transformer.h.10.mlp.c_fc.bias",
	"transformer.h.6.mlp.c_proj.weight":"transformer.h.10.mlp.c_proj.weight",
	"transformer.h.6.mlp.c_proj.bias":"transformer.h.10.mlp.c_proj.bias",
	"transformer.h.7.ln_1.weight":"transformer.h.11.ln_1.weight",
	"transformer.h.7.ln_1.bias":"transformer.h.11.ln_1.bias",
	"transformer.h.7.attn.bias":"transformer.h.11.attn.bias",
	"transformer.h.7.attn.masked_bias":"transformer.h.11.attn.masked_bias",
	"transformer.h.7.attn.c_attn.weight":"transformer.h.11.attn.c_attn.weight",
	"transformer.h.7.attn.c_attn.bias":"transformer.h.11.attn.c_attn.bias",
	"transformer.h.7.attn.c_proj.weight":"transformer.h.11.attn.c_proj.weight",
	"transformer.h.7.attn.c_proj.bias":"transformer.h.11.attn.c_proj.bias",
	"transformer.h.7.ln_2.weight":"transformer.h.11.ln_2.weight",
	"transformer.h.7.ln_2.bias":"transformer.h.11.ln_2.bias",
	"transformer.h.7.mlp.c_fc.weight":"transformer.h.11.mlp.c_fc.weight",
	"transformer.h.7.mlp.c_fc.bias":"transformer.h.11.mlp.c_fc.bias",
	"transformer.h.7.mlp.c_proj.weight":"transformer.h.11.mlp.c_proj.weight",
	"transformer.h.7.mlp.c_proj.bias":"transformer.h.11.mlp.c_proj.bias"
}

class SuperHFModel(AutoModel):

    @classmethod
    def set_sample_config(cls, pretrained_model_name_or_path, **kwargs):
        is_pretrained = kwargs.pop("is_pretrained", True)
        # Create the candidate net with random initialization
        candidate_hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        candidate_hf = cls.from_config(candidate_hf_config)

        if is_pretrained:
            # Create the super net
            super_hf = cls.from_pretrained(pretrained_model_name_or_path)
            # Load pre-trained weight from super net
            candidate_hf_state_dict = candidate_hf.state_dict()
            super_hf_state_dict = super_hf.state_dict()

            new_candidate_hf_state_dict = {}
            for k in candidate_hf_state_dict:
                super_hf_state = super_hf_state_dict[k]
                candidate_hf_state = super_hf_state
                for dim, size in enumerate(candidate_hf_state_dict[k].size()):
                    candidate_hf_state = candidate_hf_state.index_select(dim, torch.tensor(range(size)))
                new_candidate_hf_state_dict[k] = candidate_hf_state
            candidate_hf.load_state_dict(new_candidate_hf_state_dict)

        return candidate_hf

    @classmethod
    def search_space_generation(cls, pretrained_model_name_or_path, **kwargs):
        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        search_space = {}
        search_space["num_hidden_layers"] = list(range(int(hf_config.num_hidden_layers/2), int(hf_config.num_hidden_layers), 1))
        search_space["num_attention_heads"] = list(range(int(hf_config.num_attention_heads/2), int(hf_config.num_attention_heads), 1))
        search_space["hidden_size"] = list(range(int(hf_config.hidden_size/2), int(hf_config.hidden_size), 16))
        for k in kwargs:
            if 'max' not in kwargs[k]:
                raise ValueError("Please specify the up bound of {} in search space".format(k))
            search_range = {"min": int(kwargs[k]['max']/2), "max": int(kwargs[k]['max']), "step": 1}
            if 'min' in kwargs[k]:
                search_range["min"] = int(kwargs[k]["min"])
            if 'step' in kwargs[k]:
                search_range["step"] = int(kwargs[k]["step"])
            search_space[k] = list(range(search_range["min"], search_range["max"], search_range["step"]))
        return search_space

class SuperHFModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def set_sample_config(cls, pretrained_model_name_or_path, **kwargs):
        is_pretrained = kwargs.pop("is_pretrained", True)
        # Create the candidate net with random initialization
        candidate_hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        candidate_hf = cls.from_config(candidate_hf_config)

        if is_pretrained:
            # Create the super net
            super_hf = cls.from_pretrained(pretrained_model_name_or_path)
            # Load pre-trained weight from super net
            candidate_hf_state_dict = candidate_hf.state_dict()
            super_hf_state_dict = super_hf.state_dict()

            new_candidate_hf_state_dict = {}
            for k in candidate_hf_state_dict:
                if k.startswith("transformer.h"):
                    k_super = layer_dict[k]
                else:
                    k_super = k
                print(k, k_super)
                super_hf_state = super_hf_state_dict[k_super]
                candidate_hf_state = super_hf_state
                for dim, size in enumerate(candidate_hf_state_dict[k].size()):
                    candidate_hf_state = candidate_hf_state.index_select(dim, torch.tensor(range(size)))
                new_candidate_hf_state_dict[k] = candidate_hf_state
            candidate_hf.load_state_dict(new_candidate_hf_state_dict)
        return candidate_hf