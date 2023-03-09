import torch
import json
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoModelForCausalLM,AutoConfig

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

class SuperHFTaskModel(nn.Module):
    def __init__(self, encoder_model, cfg):
        super(SuperHFTaskModel, self).__init__()
        self.cfg = cfg
        self.encoder_model = encoder_model
        if self.cfg["task_type"] == "classification":
            self.output = nn.Linear(self.cfg["hidden_size"], self.cfg["num_labels"])
        else:
            raise NotImplementedError("Task Type are not supported yet!")
        self.output.apply(self.init_output_weight)
    
    def init_output_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        input_keys = self.cfg["input_id"].strip().split()
        item = x.split(1, -1)
        inputs = dict()
        for id, input_key in enumerate(input_keys):
            inputs[input_key] = item[id].squeeze(-1)
        output = self.encoder_model(**inputs)
        last_hidden_state = output.last_hidden_state
        logits = self.output(last_hidden_state)
        return logits
    
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
                super_hf_state = super_hf_state_dict[k]
                candidate_hf_state = super_hf_state
                for dim, size in enumerate(candidate_hf_state_dict[k].size()):
                    candidate_hf_state = candidate_hf_state.index_select(dim, torch.tensor(range(size)))
                new_candidate_hf_state_dict[k] = candidate_hf_state
            candidate_hf.load_state_dict(new_candidate_hf_state_dict)

        return candidate_hf
