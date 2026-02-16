from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures


class SwinUNETR(MonaiSwinUNETR, AbstractDynamicNetworkArchitectures):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 deep_supervision = False,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, in_channels=input_channels, out_channels=num_classes, **kwargs)

        self.key_to_encoder = "encoder1"
        self.key_to_stem = "swinViT"
        self.keys_to_in_proj = ("encoder1.layer.conv1",)

        if deep_supervision:
            raise NotImplementedError
        self.deep_supervision = deep_supervision
