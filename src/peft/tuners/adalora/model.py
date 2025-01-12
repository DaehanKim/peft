# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional

import torch
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraConfig, LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
    get_auto_gptq_quant_linear,
    get_quantization_config,
)
from peft.utils.integrations import gather_params_ctx

from .gptq import SVDQuantLinear
from .layer import AdaLoraLayer, RankAllocator, SVDLinear


class AdaLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model. Paper:
    https://openreview.net/forum?id=lq62uWRJjiY

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The AdaLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM >>> from peft import LoraConfig, AdaLoraModel, AdaLoraConfig
        >>> config = AdaLoraConfig(
                peft_type="ADALORA", task_type="SEQ_2_SEQ_LM", init_r=12, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = AdaLoraModel(model, config, "default")

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLora model.
    """

    # Note: don't redefine prefix here, it should be inherited from LoraModel

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name
            self.rankallocator = RankAllocator(self.model, self.peft_config[adapter_name], self.trainable_adapter_name)

    def _check_new_adapter_config(self, config: LoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        traininable_mode_counter = 0
        for config_ in self.peft_config.values():
            if not config_.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one "
                "you want to train."
            )

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "r": lora_config.init_r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        if (kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]) and not is_bnb_available():
            raise ImportError(
                "To use AdaLora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

        quantization_config = get_quantization_config(self.model, method="gptq")
        if quantization_config is not None:
            kwargs["gptq_quantization_config"] = quantization_config

        # If it is not an AdaLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, AdaLoraLayer):
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                lora_config.init_r,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import SVDLinear8bitLt
        if is_bnb_4bit_available():
            from .bnb import SVDLinear4bit

        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            new_module = SVDLinear8bitLt(target, adapter_name, **kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = SVDLinear4bit(target, adapter_name, **fourbit_kwargs)
        elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
            new_module = SVDQuantLinear(target, adapter_name, **kwargs)
        else:
            if isinstance(target_base_layer, torch.nn.Linear):
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target_base_layer, Conv1D):
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = SVDLinear(target, adapter_name, **kwargs)

        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)

        if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, torch.Tensor):
            # Calculate the orthogonal regularization
            orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight

            if orth_reg_weight <= 0:
                raise ValueError("orth_reg_weight should be greater than 0. ")

            regu_loss = 0
            num_param = 0
            for n, p in self.model.named_parameters():
                if ("lora_A" in n or "lora_B" in n) and self.trainable_adapter_name in n:
                    if p.shape == torch.Size([0]):
                        with gather_params_ctx(p, fwd_module=self):
                            para_cov = p @ p.T if "lora_A" in n else p.T @ p
                    else:
                        para_cov = p @ p.T if "lora_A" in n else p.T @ p
                    I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))  # noqa: E741
                    I.requires_grad = False
                    num_param += 1
                    regu_loss += torch.norm(para_cov - I, p="fro")
            if num_param > 0:
                regu_loss = regu_loss / num_param
            else:
                regu_loss = 0
            outputs.loss += orth_reg_weight * regu_loss
        return outputs

    def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name):
        lora_config = self.peft_config[adapter_name]
        for name, rank_idx in rank_pattern.items():
            if isinstance(rank_idx, list):
                rank = sum(rank_idx)
            elif isinstance(rank_idx, torch.Tensor):
                rank_idx = rank_idx.view(-1)
                rank = rank_idx.sum().item()
            else:
                raise ValueError("Unexpected type of rank_idx")
            key = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            _, target, _ = _get_submodules(self.model, key)
            lora_E_weights = target.lora_E[adapter_name][rank_idx]
            lora_A_weights = target.lora_A[adapter_name][rank_idx]
            lora_B_weights = target.lora_B[adapter_name][:, rank_idx]
            ranknum = target.ranknum[adapter_name]
            target.update_layer(
                adapter_name,
                rank,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
            with torch.no_grad():
                if rank > 0:
                    target.lora_E[adapter_name].copy_(lora_E_weights)
                    target.lora_A[adapter_name].copy_(lora_A_weights)
                    target.lora_B[adapter_name].copy_(lora_B_weights)
                    # The scaling is exactly as the previous
                    target.ranknum[adapter_name].copy_(ranknum)

    def resize_state_dict_by_rank_pattern(self, rank_pattern, state_dict, adapter_name):
        for name, rank_idx in rank_pattern.items():
            rank = sum(rank_idx)
            prefix = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            for layer in ["lora_E", "lora_A", "lora_B"]:
                key = f"base_model.model.{prefix}.{layer}.{adapter_name}"
                if layer != "lora_B":
                    state_dict[key] = (
                        state_dict[key][rank_idx] if rank != state_dict[key].shape[0] else state_dict[key]
                    )
                else:
                    state_dict[key] = (
                        state_dict[key][:, rank_idx] if rank != state_dict[key].shape[1] else state_dict[key]
                    )
        return state_dict

    def update_and_allocate(self, global_step):
        """
        This method updates Adalora budget and mask.

        This should be called in every training step after `loss.backward()` and before `zero_grad()`.

        `tinit`, `tfinal` and `deltaT` are handled with in the method.

        Args:
            global_step (`int`): The current training step, it is used to calculate adalora budget.

        Example:

        ```python
        >>> loss = model(**input).loss
        >>> loss.backward()
        >>> optimizer.step()
        >>> model.base_model.update_and_allocate(i_step)
        >>> optimizer.zero_grad()
        ```
        """
        lora_config = self.peft_config[self.trainable_adapter_name]
        # Update the importance score and allocate the budget
        if global_step < lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step)
            if rank_pattern:
                lora_config.rank_pattern = rank_pattern
        # Finalize the budget allocation
        elif global_step == lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step, force_mask=True)
            # for some reason, this freezes the trainable parameters and nothing gets updates
            # self.resize_modules_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
            lora_config.rank_pattern = rank_pattern
            self.rankallocator.reset_ipt()
        # Currently using inefficient way to mask the unimportant weights using the rank pattern
        #  due to problem mentioned above
        elif global_step > lora_config.total_step - lora_config.tfinal:
            self.rankallocator.mask_using_rank_pattern(self.model, lora_config.rank_pattern)
        # Pass the function and do forward propagation
        else:
            return None

    def add_weighted_adapter(self, *args, **kwargs):
        """This method is not supported for AdaLoRA, use LoRA instead."""
        raise TypeError(f"{self.__class__.__name__} does not support add_weighted_adapter method.")

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
        eps: float = 1e-5,
    ) -> torch.nn.Module:
        """
        This method unloads the AdaLoRA adapter modules and optionally merges them into the base model weights.

        Args:
            merge (`bool`, defaults to `True`):
                If True, merges the adapter weights into base model weights.
                If False, it will only unload the adapters without merging.
            safe_merge (`bool`, defaults to `False`):
                If True, performs the merge operation with extra safety checks.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names to merge. If None, all active adapters will be merged.
            eps (`float`, defaults to 1e-5):
                Small constant for numerical stability when dividing by ranknum.

        Returns:
            model (`torch.nn.Module`):
                The resulting PyTorch model.
        """
        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge adalora layers when the model is loaded in 8-bit mode")

        if getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge adalora layers when the model is loaded in 4-bit mode")

        if adapter_names is not None:
            raise ValueError("AdaLoRA does not support merging specific adapters. Got adapter_names={adapter_names}")

        # Create a copy of the base model state dict to modify
        original_state_dict = self.model.state_dict()

        if merge:
            for name, module in self.model.named_modules():
                if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
                    # Extract base layer weight name
                    layer_name = name.replace(".lora_A", "")
                    layer_name = layer_name.replace("base_model.model.", "")
                    base_weight_name = f"{layer_name}.weight"

                    # Get SVD parameters
                    lora_A = module.lora_A["default"]  # [r x d_in]
                    lora_B = module.lora_B["default"]  # [d_out x r]
                    lora_E = module.lora_E["default"]  # [r x 1]

                    # Calculate active ranks
                    ranknum = (lora_E != 0).sum()
                    scaling = module.scaling["default"] if hasattr(module, "scaling") else 16

                    # Safety check if requested
                    if safe_merge and (
                        torch.isnan(lora_A).any() or torch.isnan(lora_B).any() or torch.isnan(lora_E).any()
                    ):
                        raise ValueError(f"NaN detected in adapter weights for layer {name}")

                    # Scale A with E: A' = AE
                    scaled_A = lora_A * lora_E  # [r x d_in]

                    # Compute update: ΔW = BA'
                    if ranknum > 0:
                        update = (lora_B @ scaled_A) * scaling / (ranknum + eps)
                    else:
                        update = torch.zeros_like(original_state_dict[base_weight_name])

                    # Update base weights
                    if base_weight_name in original_state_dict:
                        original_state_dict[base_weight_name] += update

        # Load the merged state dict back into a clean version of the model
        self.model.load_state_dict(original_state_dict)

        return self.model

    def merge_and_unload(
        self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None, eps: float = 1e-5
    ) -> torch.nn.Module:
        """
        Merge the active adapters into the base model and unload the adapters.

        Args:
            safe_merge (`bool`, defaults to `False`):
                If True, performs the merge operation with extra safety checks.
            adapter_names (`List[str]`, *optional*):
                List of adapter names to merge. If None, merges all active adapters.
            eps (`float`, defaults to 1e-5):
                Small constant for numerical stability when dividing by ranknum.

        Returns:
            `torch.nn.Module`: The merged model.
        """
        return self._unload_and_optionally_merge(safe_merge=safe_merge, adapter_names=adapter_names, eps=eps)

    def unload(self) -> torch.nn.Module:
        """
        Unload the adapters without merging them into the base model.

        Returns:
            `torch.nn.Module`: The unloaded model.
        """
        return self._unload_and_optionally_merge(merge=False)
