import typing
from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    FlexibleDataManager,
    FlexibleDataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils import profiler

from permuto.permuto_sdf import PermutoSDFModel, PermutoSDFModelConfig

def map_range_val( input_val, input_start, input_end,  output_start,  output_end):
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    input_clamped=max(input_start, min(input_end, input_val))
    # input_clamped=torch.clamp(input_val, input_start, input_end)
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)


@dataclass
class PermutoPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: PermutoPipeline)
    """target class to instantiate"""
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: PermutoSDFModelConfig = PermutoSDFModelConfig()
    """specifies the model config"""


class PermutoPipeline(VanillaPipeline):
    def __init__(
        self,
        config: PermutoPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(PermutoSDFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
            
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            # Report the camera optimization metrics
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        # 9500 10501
        global_weight_curvature=map_range_val(step, self.model.config.start_reduce_curv, self.model.config.finish_reduce_curv, 1.0, 0.000)
        if global_weight_curvature > 0.0:
            loss_dict["curv_loss"] = loss_dict["curv_loss"] * global_weight_curvature
        else:
            del loss_dict["curv_loss"]
        
        if step < self.model.config.start_reduce_curv:
            del loss_dict["loss_lipshitz"]
            
        # if (step == self.model.config.start_reduce_curv-1) or (step == self.model.config.start_reduce_curv) or (step == self.model.config.start_reduce_curv+1):
        #     print(step)
        #     print(global_weight_curvature)
        #     print(loss_dict)
            
        
        

        return model_outputs, loss_dict, metrics_dict