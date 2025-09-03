import torch
import torch.nn as nn
from torchvision import models as tv
from collections import namedtuple
import lpips

class OfflineVGG16(torch.nn.Module):
    """VGG16 that loads weights from local file instead of downloading"""
    def __init__(self, requires_grad=False, weights_path="lpips_weights/vgg16_imagenet.pth"):
        super(OfflineVGG16, self).__init__()
        
        # Create VGG16 architecture without pretrained weights
        vgg_features = tv.vgg16(pretrained=False).features
        
        # Load the downloaded weights
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Create a temporary full model to load weights, then extract features
        temp_model = tv.vgg16(pretrained=False)
        temp_model.load_state_dict(state_dict)
        vgg_pretrained_features = temp_model.features
        
        # Build the sliced architecture (same as original)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class OfflineLPIPS(lpips.LPIPS):
    """LPIPS that loads all weights from local files"""
    def __init__(self, weights_dir="lpips_weights", **kwargs):
        # Override pretrained to False initially to prevent online downloads
        original_pretrained = kwargs.get('pretrained', True)
        kwargs['pretrained'] = False  # Prevent automatic download
        
        super().__init__(**kwargs)
        
        # Replace the network with our offline version
        if self.pnet_type in ['vgg', 'vgg16']:
            vgg_weights_path = f"{weights_dir}/vgg16_imagenet.pth"
            self.net = OfflineVGG16(
                requires_grad=self.pnet_tune,
                weights_path=vgg_weights_path
            )
        
        # Load LPIPS linear layer weights if they were originally pretrained
        if original_pretrained and self.lpips:
            lpips_weights_path = f"{weights_dir}/vgg_lpips.pth"
            print(f"Loading LPIPS weights from: {lpips_weights_path}")
            self.load_state_dict(torch.load(lpips_weights_path, map_location='cpu'), strict=False)


def create_offline_lpips(device, weights_dir="lpips_weights"):
    """Convenience function to create offline LPIPS model"""
    return OfflineLPIPS(net='vgg', weights_dir=weights_dir).to(device)
