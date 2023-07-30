"""
Field for permuto model, rather then estimating density to generate a surface,
a signed distance function (SDF) for surface representation is used to help with extracting high fidelity surfaces
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import SDFField, SDFFieldConfig

import permutohedral_encoding as permuto_enc

@dataclass
class PermutoFieldConfig(SDFFieldConfig):
    """SDF Field Config"""

    _target: Type = field(default_factory=lambda: PermutoField)
    

class PermutoField(SDFField):
    """
    A field for Signed Distance Functions (SDF).

    Args:
        config: The configuration for the SDF field.
        aabb: An axis-aligned bounding box for the SDF field.
        num_images: The number of images for embedding appearance.
        use_average_appearance_embedding: Whether to use average appearance embedding. Defaults to False.
        spatial_distortion: The spatial distortion. Defaults to None.
    """
    
    config: PermutoFieldConfig
    
    def __init__(
        self,
        config: PermutoFieldConfig,
        aabb: Float[Tensor, "2 3"],
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor
        
        # create encoding
        pos_dim=3
        capacity=pow(2,18) #2pow18
        nr_levels=16 
        nr_feat_per_level=2 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
            #                                        3         2**18    24             2               np.geomspace(1.0, 0.0001, 24)
        self.encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1e-3)           
        print('permuto!!!')
        
        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=6, min_freq_exp=0.0, max_freq_exp=5.0, include_input=False
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # initialize geometric network
        self.initialize_geo_layers()

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = LearnedVariance(init_val=self.config.beta_init)

        # color network
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        # point, view_direction, normal, feature, embedding
        in_dim = (
            3
            + self.direction_encoding.get_out_dim()
            + 3
            + self.config.geo_feat_dim
            + self.embedding_appearance.get_out_dim()
        )
        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)
        
        self.lipshitz_bound_per_layer=torch.nn.ParameterList()

        for layer in range(0, self.num_layers_color - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "clin" + str(layer), lin)
            
            max_w = torch.max(torch.sum(torch.abs(lin.weight), dim=1))
            c = torch.nn.Parameter(  torch.ones((1))*max_w*2 ) 
            self.lipshitz_bound_per_layer.append(c)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0

        if self.use_grid_feature:
            assert self.spatial_distortion is not None, "spatial distortion must be provided when using grid feature"

    def lipshitz_bound_full(self):
        lipshitz_full=1
        for i in range(0, self.num_layers_color - 1):
            lipshitz_full=lipshitz_full*torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])

        return lipshitz_full
    
    def initialize_geo_layers(self) -> None:
        """
        Initialize layers for geometric network (sdf)
        """
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        # in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.output_dims()
        dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]
        self.num_layers = len(dims)
        self.skip_in = [4]

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if self.config.geometric_init:
                if layer == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif layer == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "glin" + str(layer), lin)
            
    def forward_geonetwork(self, inputs: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch geo_features+1"]:
        """forward the geonetwork"""
        if self.use_grid_feature:
            assert self.spatial_distortion is not None, "spatial distortion must be provided when using grid feature"
            positions = self.spatial_distortion(inputs)
            # map range [-2, 2] to [0, 1]
            positions = (positions + 2.0) / 4.0
            feature = self.encoding(positions)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.output_dims()))
            # feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

        pe = self.position_encoding(inputs)

        inputs = torch.cat((inputs, pe, feature), dim=-1)

        # Pass through layers
        outputs = inputs

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(layer))

            if layer in self.skip_in:
                outputs = torch.cat([outputs, inputs], 1) / np.sqrt(2)

            outputs = lin(outputs)

            if layer < self.num_layers - 2:
                outputs = self.softplus(outputs)
        return outputs   