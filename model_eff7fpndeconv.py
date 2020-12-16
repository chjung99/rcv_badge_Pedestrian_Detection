from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
import pdb
from EfficientDetPytorch.models.efficientnet import EfficientNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    
    def __init__(self, version=0):
        super(VGGBase, self).__init__()
        self.conv1=nn.Conv2d(56,512,kernel_size=3,padding=1,bias=True)
        self.conv1_bn= nn.BatchNorm2d(512, affine=True)
        
        
        
        self.conv2=nn.Conv2d(112,512,kernel_size=3,padding=1,bias=True)
        self.conv2_bn= nn.BatchNorm2d(512, affine=True)
        
        
        self.conv3=nn.Conv2d(160,512,kernel_size=3,padding=1,bias=True)
        self.conv3_bn= nn.BatchNorm2d(512, affine=True)
        
#         self.deconv6=nn.Conv2d(448,272,kernel_size=3,padding=1,bias=True)
#         self.deconv6_bn= nn.BatchNorm2d(272, affine=True)
        
#         self.deconv5=nn.Conv2d(272,160,kernel_size=3,padding=1,bias=True)
#         self.deconv5_bn= nn.BatchNorm2d(160, affine=True)
        
#         self.deconv4=nn.Conv2d(160,112,kernel_size=3,padding=1,bias=True)
#         self.deconv4_bn= nn.BatchNorm2d(112, affine=True)
        
#         self.deconv3=nn.Conv2d(112,56,kernel_size=3,padding=1,bias=True)
#         self.deconv3_bn= nn.BatchNorm2d(56, affine=True)
        
        
        self.deconv6=nn.ConvTranspose2d(448,272,kernel_size=2,stride=2)
        self.deconv6_bn= nn.BatchNorm2d(272, affine=True)
        self.deconv5=nn.ConvTranspose2d(272,160,kernel_size=2,stride=2)
        self.deconv5_bn= nn.BatchNorm2d(160, affine=True)
        self.deconv4=nn.ConvTranspose2d(160,112,kernel_size=2,stride=2)
        self.deconv4_bn= nn.BatchNorm2d(112, affine=True)
        self.deconv3=nn.ConvTranspose2d(112,56,kernel_size=2,stride=2)
        self.deconv3_bn= nn.BatchNorm2d(56, affine=True)
        
        self.efficientNet = EfficientNet.from_pretrained('efficientnet-b4')
        state_dict=self.efficientNet.state_dict()
        
        
        self.load_state_dict(state_dict,strict=False)
        
#     def _upsample_add(self, x, y):
#         '''Upsample and add two feature maps.
#         Args:
#           x: (Variable) top feature map to be upsampled.
#           y: (Variable) lateral feature map.
#         Returns:
#           (Variable) added feature map.
#         Note in PyTorch, when input size is odd, the upsampled feature map
#         with `F.upsample(..., scale_factor=2, mode='nearest')`
#         maybe not equal to the lateral feature map size.
#         e.g.
#         original input size: [N,_,15,15] ->
#         conv2d feature map size: [N,_,8,8] ->
#         upsampled feature map size: [N,_,16,16]
#         So we choose bilinear upsample which supports arbitrary output sizes.
#         '''
        
#         _,_,H,W = y.size()
#         nn.ConvTranspose2d
#         return F.upsample(x, size=(H,W), mode='bilinear') + y
    
    def forward(self, image):
        features = self.efficientNet(image)

        
        feat6=features[6]
        cfeat6=(self.deconv6_bn(self.deconv6(feat6)))
        
        feat5=cfeat6+features[5]
        cfeat5=(self.deconv5_bn(self.deconv5(feat5)))
        
        feat4=(cfeat5+features[4])
        cfeat4=(self.deconv4_bn(self.deconv4(feat4)))
        
        feat3=(cfeat4+features[3])
        cfeat3=(self.deconv3_bn(self.deconv3(feat3)))
        
        feat2=(cfeat3+features[2])
        
        
        feat2= F.relu(self.conv1_bn(self.conv1(feat2)))
        feat3= F.relu(self.conv2_bn(self.conv2(feat3)))
        feat4= F.relu(self.conv3_bn(self.conv3(feat4)))
        
        return feat2,feat3,feat4
    
    
class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

    
        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)  # stride = 1, by default
 
        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv9_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)

        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv10_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        # out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        # out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        # conv8_2_feats = out  # (N, 512, 10, 10)

        # out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        # out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        # conv9_2_feats = out  # (N, 256, 5, 5)

        # out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        # out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        # conv10_2_feats = out  # (N, 256, 3, 3)

        # out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        # conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out)) 

        conv8_feats = out  

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out)) 
        conv9_feats = out 

        out = F.relu(self.conv10_1(out)) 
        out = F.relu(self.conv10_2(out)) 
        conv10_feats = out       

        # Higher-level feature maps
        # return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
        return conv8_feats, conv9_feats, conv10_feats

class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        # n_boxes = {'conv4_3': 4,
        #            'conv7': 6,
        #            'conv8_2': 6,
        #            'conv9_2': 6,
        #            'conv10_2': 4,
        #            'conv11_2': 4}
        n_boxes = {'conv4_3': 6,
                    'conv6': 6,
                    'conv7': 6,
                    'conv8': 6,
                    'conv9': 6,
                    'conv10': 6,}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.


        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv6 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv8 = nn.Conv2d(512, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv9 = nn.Conv2d(512, n_boxes['conv8'] * 4, kernel_size=3, padding=1)
        self.loc_conv10 = nn.Conv2d(512, n_boxes['conv9'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv6 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8 = nn.Conv2d(512, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9 = nn.Conv2d(512, n_boxes['conv8'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10 = nn.Conv2d(512, n_boxes['conv9'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    # def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
    def forward(self, conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_1_feats: conv8_1 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_1_feats: conv9_1 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param conv10_1_feats: conv10_1 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_1_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """


        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv6 = self.loc_conv6(conv6_feats)  # (N, 24, 19, 19)
        l_conv6 = l_conv6.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv6 = l_conv6.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        # l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        # l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        # l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        # l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        # l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        # l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        # l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        # l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        # l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        # l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        # l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        # l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        l_conv8 = self.loc_conv8(conv8_feats)  # (N, 24, 10, 10)
        l_conv8 = l_conv8.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8 = l_conv8.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9 = self.loc_conv9(conv9_feats)  # (N, 24, 5, 5)
        l_conv9 = l_conv9.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9 = l_conv9.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10 = self.loc_conv10(conv10_feats)  # (N, 16, 3, 3)
        l_conv10 = l_conv10.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10 = l_conv10.view(batch_size, -1, 4)  # (N, 36, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map


        c_conv6 = self.cl_conv6(conv6_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv6 = c_conv6.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv6 = c_conv6.view(batch_size, -1, self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        # c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        # c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        # c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        # c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        # c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        # c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        # c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        # c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        # c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        # c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        # c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        # c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        c_conv8 = self.cl_conv8(conv8_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8 = c_conv8.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8 = c_conv8.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9 = self.cl_conv9(conv9_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9 = c_conv9.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9 = c_conv9.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10 = self.cl_conv10(conv10_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10 = c_conv10.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10 = c_conv10.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)       

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        # locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        # classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
        #                            dim=1)  # (N, 8732, n_classes)

        locs = torch.cat([l_conv4_3, l_conv6, l_conv7, l_conv8, l_conv9, l_conv10], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv6, c_conv7, c_conv8, c_conv9, c_conv10],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1,512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv6_feats , conv7_feats= self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        # conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
        #     self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)
        conv8_feats, conv9_feats, conv10_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        # locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
        #                                        conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats)  # (N, 8732, 4), (N, 8732, n_classes)
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """

        fmap_dims = {'conv4_3': [80,64],
                     'conv6': [40,32],
                     'conv7': [20,16],
                     'conv8': [10,8],
                     'conv9': [10,8],
                     'conv10': [10,8]}

        scale_ratios = {'conv4_3': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv6': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv7': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv8': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv9': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv10': [1., pow(2,1/3.), pow(2,2/3.)]}


        aspect_ratios = {'conv4_3': [1/2., 1/1.],
                         'conv6': [1/2., 1/1.],
                         'conv7': [1/2., 1/1.],
                         'conv8': [1/2., 1/1.],
                         'conv9': [1/2., 1/1.],
                         'conv10': [1/2., 1/1.]}


        anchor_areas = {'conv4_3': [40*40.],
                         'conv6': [80*80.],
                         'conv7': [160*160.],
                         'conv8': [200*200.],
                         'conv9': [280*280.],
                         'conv10': [360*360.]} 


        # fmaps = list(fmap_dims.keys())
        fmaps = ['conv4_3', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10']

        prior_boxes = []



        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap][1]):
                for j in range(fmap_dims[fmap][0]):
                    cx = (j + 0.5) / fmap_dims[fmap][0]
                    cy = (i + 0.5) / fmap_dims[fmap][1]
                    for s in anchor_areas[fmap]:
                        for ar in aspect_ratios[fmap]: 
                            h = sqrt(s/ar)                
                            w = ar * h
                            for sr in scale_ratios[fmap]: # scale
                                anchor_h = h*sr/512.
                                anchor_w = w*sr/640.
                                prior_boxes.append([cx, cy, anchor_w, anchor_h])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)


        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to

            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)) 

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=4, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, ignore_index=-1)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
    
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior
            
            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        # positive_priors = true_classes != 0  # (N, 8732)
        positive_priors = true_classes > 0  # (N, 8732)

        # LOCALIZATION LOSS
        # Localization loss is computed only over positive (non-background) priors
        if true_locs[positive_priors].shape[0] == 0:
            loc_loss = 0.
        else:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        
        # n_hard_negatives = torch.clamp(self.neg_pos_ratio * n_positives, min=32 )   # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  

        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / ( 1e-10 + n_positives.sum().float() )  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss , conf_loss , loc_loss, n_positives
