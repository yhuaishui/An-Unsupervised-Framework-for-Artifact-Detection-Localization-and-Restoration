import os
import huggingface_hub

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes

class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name='hf_hub:prov-gigapath/prov-gigapath',
        backbone_pretrained=True,
        flow_steps=20,
        input_size=224,
        conv3x3_only=False,
        hidden_ratio=0.16,
        output_size=256,
    ):
        super(FastFlow, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.feature_extractor = timm.create_model(backbone_name, pretrained=backbone_pretrained)
        self.channels = [1536]
        self.scales = [16]
        
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(self.channels, self.scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        
    def train(self, mode=True):
        super(FastFlow, self).train(mode)
        self.feature_extractor.eval()
        return self
    
    def backbone_fertures(self, x):
        x = self.feature_extractor.forward_features(x)
        x = x[:, 1:, :]
        N, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(N, C, self.input_size // self.scales[0], self.input_size // self.scales[0])
        features = [x]
        return features

    def forward(self, x):
        features = self.backbone_fertures(x)

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        ret = {"loss": loss}

        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.output_size, self.output_size],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
                        
            ret["anomaly_map"] = anomaly_map
            ret["anomaly_score"] = anomaly_map.view(anomaly_map.size(0), -1).max(dim=1).values
        return ret

    
def build_model(config, backbone_pretrained):
    model = FastFlow(
        backbone_name=config["backbone_name"],
        backbone_pretrained=backbone_pretrained,
        flow_steps=config["flow_steps"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
        output_size=config["output_size"],
    )
    # print(
    #     "Model A.D. Param#: {}".format(
    #         sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     )
    # )
    print("Load Model FastFlow")
    return model
