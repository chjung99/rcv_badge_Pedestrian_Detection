import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import pdb

# from torchcv.corr.modules.correlation import Correlation
# from torchcv.defconv.modules import ConvOffset2d, AdaptiveConvOffset2d


#from torchcv.layers.modules import Correlation, ConvOffset2d, AdaptiveConvOffset2d, AdaptiveConv2d
# from torchcv.cpp_extensions.deformable_conv.modules import DeformConv
from modules.deform_conv import DeformConv
from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation
    
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FUSION_TYPES = ('Mul_1x1', 'Mul_1x1_Single', 'Mul_HxW+1x1', 'Conv', 'Mul_HxW')


class FusionModule(nn.Module):
    def __init__(self, params, fusion_order='TM'):
        super(FusionModule, self).__init__()

        assert fusion_order in ['TM', 'MT']

        self.fusion_modules = nn.ModuleList()

        for p in params:
            if p['type'].lower() == 'adaptive':
                ## in_ch, dims, out_plane, kernel_size, fusion_type
                self.fusion_modules += [ AdaptiveFusion(p['in_ch'], p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type']) ]

            elif p['type'].lower() == 'static':
                ## dims, out_plane, kernel_size, fusion_type
                self.fusion_modules += [ StaticFusion(p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type']) ]

            elif p['type'].lower() == 'temporal':
                ## in_ch, out_ch, num_groups, kernel_size=3, n_iter=1
                k = p['kernel_size'] if 'kernel_size' in p else 3
                n = p['n_iter'] if 'n_iter' in p else 1
                self.fusion_modules += [ TemporalFusion(p['in_ch'], p['out_ch'], p['num_groups'], k, n) ]
            
            elif p['type'].lower() == 'sum':                                
                self.fusion_modules +=  [ SumFusion() ]

            elif p['type'].lower() == 'conv1x1':
                in_ch, out_ch = p['in_ch'], p['out_ch']
                self.fusion_modules += [ Conv1x1Fusion(in_ch, out_ch) ]

            elif p['type'].lower() == 'adf':
                i, o, t, s = p['in_ch'], p['out_ch'], p['t_window'], p['swap']
                k = p['kernel_size'] if 'kernel_size' in p else 3
                self.fusion_modules += [ AdaptiveDeformedFusion( i, o, t, s, k ) ]

            elif p['type'].lower() == 'stsn':
                i, o = p['in_ch'], p['out_ch']
                k = p['kernel_size'] if 'kernel_size' in p else 3
                self.fusion_modules += [ SpatioTemporalFusion( i, o, k ) ]

            elif p['type'].lower() == 'astsn':
                i, o, s, m = p['in_ch'], p['out_ch'], p['swap'], p['mode']
                k = p['kernel_size'] if 'kernel_size' in p else 3
                self.fusion_modules += [ AdaptiveSpatioTemporalFusion( i, o, s, m, k ) ]
                
            else:
                raise NotImplementedError
        
        self.fusion_order = fusion_order
        

    def forward(self, x0, y0, x, y):
        # x0, x: RGB / y0, y: T

        B0, T0, C0, H0, W0 = x0.shape
        B, T, C, H, W = x.shape

        # if self.fusion_order == 'TM':


        ### Eg. (temporal, temporal), sum
        # x0 = x0.view(B0*T0, C0, H0, W0)
        # y0 = y0.view(B0*T0, C0, H0, W0)

        x = x.view(B*T, C, H, W)
        y = y.view(B*T, C, H, W)

        h1 = self.fusion_modules[0](x0, y0, x)
        h2 = self.fusion_modules[1](x0, y0, y)

        h = self.fusion_modules[2](h1, h2)


        # h1 = self.fusion_modules[2](x0.mean(dim=1), y0.mean(dim=1), h1)
        # h2 = self.fusion_modules[3](x0.mean(dim=1), y0.mean(dim=1), h2)

        # h = self.fusion_modules[4](h1, h2)

        # else:
        #     ### Eg. (adaptive, adaptive, sum), temporal
        #     h1_tt = []

        #     # pdb.set_trace()

        #     for tt in range(T):
        #         x0_tt = x0[:,tt,:,:,:]
        #         y0_tt = y0[:,tt,:,:,:]

        #         x_tt = x[:,tt,:,:,:]
        #         y_tt = y[:,tt,:,:,:]

        #         # x0_tt = x0[tt::B]
        #         # y0_tt = y0[tt::B]
        #         # x_tt = x[tt::B]
        #         # y_tt = y[tt::B]

        #         h1x_tt  = x_tt * 0.5
        #         h1y_tt  = y_tt * 0.5
        #         # h1x_tt = self.fusion_modules[0](x0_tt, y0_tt, x_tt)
        #         # h1y_tt = self.fusion_modules[1](x0_tt, y0_tt, y_tt)

        #         h1_tt.append( self.fusion_modules[2]( h1x_tt, h1y_tt ) )

        #     h = self.fusion_modules[3]( None, None, torch.cat( h1_tt, 0 ) )

        return h
        # return self.module( x0, y0, x)


# class FusionModule(nn.Module):
#     def __init__(self, params, fusion_order='TM'):
#         super(FusionModule, self).__init__()

#         assert fusion_order in ['TM', 'MT']

#         self.fusion_modules = nn.ModuleList()
            
#         for p in params:
#             if p['type'].lower() == 'adaptive':
#                 ## in_ch, dims, out_plane, kernel_size, fusion_type
#                 self.fusion_modules += [ AdaptiveFusion(p['in_ch'], p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type']) ]

#             elif p['type'].lower() == 'static':
#                 ## dims, out_plane, kernel_size, fusion_type
#                 self.fusion_modules += [ StaticFusion(p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type']) ]

#             elif p['type'].lower() == 'temporal':
#                 ## in_ch, out_ch, num_groups, kernel_size=3, n_iter=1
#                 k = p['kernel_size'] if 'kernel_size' in p else 3
#                 n = p['n_iter'] if 'n_iter' in p else 1
#                 self.fusion_modules += [ TemporalFusion(p['in_ch'], p['out_ch'], p['num_groups'], k, n) ]
            
#             elif p['type'].lower() == 'sum':                                
#                 self.fusion_modules +=  [ SumFusion() ]

#             elif p['type'].lower() == 'conv1x1':
#                 in_ch, out_ch = p['in_ch'], p['out_ch']
#                 self.fusion_modules += [ Conv1x1Fusion(in_ch, out_ch) ]
#             else:
#                 raise NotImplementedError
        
#         self.fusion_order = fusion_order
        

#     def forward(self, x0, y0, x, y):
#         # x0, x: RGB / y0, y: T

#         B0, T0, C0, H0, W0 = x0.shape
#         B, T, C, H, W = x.shape

#         if self.fusion_order == 'TM':
#             ### Eg. (temporal, temporal), sum
#             # x0 = x0.view(B0*T0, C0, H0, W0)
#             # y0 = y0.view(B0*T0, C0, H0, W0)

#             x = x.view(B*T, C, H, W)
#             y = y.view(B*T, C, H, W)

#             h1 = self.fusion_modules[0](x0, y0, x)
#             h2 = self.fusion_modules[1](x0, y0, y)

#             # h = self.fusion_modules[2](h1, h2)


#             h1 = self.fusion_modules[2](x0.mean(dim=1), y0.mean(dim=1), h1)
#             h2 = self.fusion_modules[3](x0.mean(dim=1), y0.mean(dim=1), h2)

#             h = self.fusion_modules[4](h1, h2)

#         else:
#             ### Eg. (adaptive, adaptive, sum), temporal
#             h1_tt = []

#             # pdb.set_trace()

#             for tt in range(T):
#                 x0_tt = x0[:,tt,:,:,:]
#                 y0_tt = y0[:,tt,:,:,:]

#                 x_tt = x[:,tt,:,:,:]
#                 y_tt = y[:,tt,:,:,:]

#                 # x0_tt = x0[tt::B]
#                 # y0_tt = y0[tt::B]
#                 # x_tt = x[tt::B]
#                 # y_tt = y[tt::B]

#                 h1x_tt  = x_tt * 0.5
#                 h1y_tt  = y_tt * 0.5
#                 # h1x_tt = self.fusion_modules[0](x0_tt, y0_tt, x_tt)
#                 # h1y_tt = self.fusion_modules[1](x0_tt, y0_tt, y_tt)

#                 h1_tt.append( self.fusion_modules[2]( h1x_tt, h1y_tt ) )

#             h = self.fusion_modules[3]( None, None, torch.cat( h1_tt, 0 ) )

#         return h
#         # return self.module( x0, y0, x)


# class FusionModule(nn.Module):
#     def __init__(self, p):
#         super(FusionModule, self).__init__()

#         if p['type'].lower() == 'adaptive':
#             ## in_ch, dims, out_plane, kernel_size, fusion_type
#             self.module = AdaptiveFusion(p['in_ch'], p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type'])

#         elif p['type'].lower() == 'static':
#             ## dims, out_plane, kernel_size, fusion_type
#             self.module = StaticFusion(p['dims'], p['out_plane'], p['kernel_size'], p['fusion_type'])

#         elif p['type'].lower() == 'temporal':
#             ## in_ch, out_ch, num_groups, kernel_size=3, n_iter=1
#             k = p['kernel_size'] if 'kernel_size' in p else 3
#             n = p['n_iter'] if 'n_iter' in p else 1
#             self.module = TemporalFusion(p['in_ch'], p['out_ch'], p['num_groups'], k, n)
        
#         elif p['type'].lower() == 'sum':
#                 self.fusion_modules +=  [ SumFusion() ]

#         elif p['type'].lower() == 'adf':
#             i, o, t, s = p['in_ch'], p['out_ch'], p['t_window'], p['swap']
#             k = p['kernel_size'] if 'kernel_size' in p else 3
#             self.module = AdaptiveDeformedFusion( i, o, t, s, k )

#         else:
#             raise NotImplementedError

#     def forward(self, x0, y0, x):
#         return self.module( x0, y0, x)

############################################################################################################
############################################################################################################


class Conv1x1Fusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv1x1Fusion, self).__init__()
        self.layer = nn.Conv2d( in_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=False )

    def forward(self, x, y):
        return self.layer( torch.cat( [x, y], 1 ) )



class SumFusion(nn.Module):
    def __init__(self):
        super(SumFusion, self).__init__()

    def forward(self, x, y):
        return x + y


class PredWeights(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):

        super(PredWeights, self).__init__()
                        
        self.in_ch = in_ch
        self.out_ch = out_ch                
        self.kernel_size = kernel_size
        self.padding = math.floor(kernel_size/2)
    
        self.conv = nn.Sequential(
                nn.Conv2d( in_ch, out_ch, kernel_size=5, padding=2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                
                nn.Conv2d( out_ch, out_ch, kernel_size=5, padding=2, stride=2, bias=False),                
                nn.ReLU(inplace=True),
                
                nn.Conv2d( out_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            )
               
        self.wx = nn.Linear( out_ch, out_ch*out_ch*kernel_size*kernel_size )
        self.bx = nn.Linear( out_ch, out_ch )        

    def forward(self, feat):
                           
        B = feat.size(0)

        # feat = F.avg_pool2d( self.conv(feat), kernel_size=11, padding=(2,0) )
        feat = F.adaptive_avg_pool2d( self.conv(feat), (1,1) )
        feat = feat.view( B, -1 )            

        wx = self.wx( feat ).view( B, self.out_ch, self.out_ch, self.kernel_size, self.kernel_size)
        bx = self.bx( feat ).view( B, self.out_ch, 1, 1)
        
        return wx, bx


class PredOffsets(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):

        super(PredOffsets, self).__init__()
                        
        self.in_ch = in_ch
        self.out_ch = out_ch                
        self.kernel_size = kernel_size
        self.padding = math.floor(kernel_size/2)
    
        self.conv = nn.Sequential(
                nn.Conv2d( in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
                nn.ReLU(inplace=True),

                nn.Conv2d( in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=False),
                nn.ReLU(inplace=True),


                
                nn.Conv2d( out_ch, out_ch, kernel_size=5, padding=2, stride=2, bias=False),                
                nn.ReLU(inplace=True),
                
                nn.Conv2d( out_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            )
               
        self.wx = nn.Linear( out_ch, out_ch*out_ch*kernel_size*kernel_size )
        self.bx = nn.Linear( out_ch, out_ch )        

    def forward(self, feat):
                           
        B = feat.size(0)

        # feat = F.avg_pool2d( self.conv(feat), kernel_size=11, padding=(2,0) )
        feat = F.adaptive_avg_pool2d( self.conv(feat), (1,1) )
        feat = feat.view( B, -1 )            

        wx = self.wx( feat ).view( B, self.out_ch, self.out_ch, self.kernel_size, self.kernel_size)
        bx = self.bx( feat ).view( B, self.out_ch, 1, 1)
        
        return wx, bx


class AdaptiveDeformedFusion(nn.Module):

    max_disp = 7

    def __init__(self, in_ch, out_ch, t_window, swap, kernel_size=3):        
        super(AdaptiveDeformedFusion, self).__init__()

        assert t_window == 2, 'Only support 2 frames input case.'
        dims = 128

        self.in_ch = in_ch
        self.out_ch = out_ch                        
        self.kernel_size = kernel_size         
        self.swap = swap           
        self.padding = math.floor(kernel_size/2)


        # 1. correlation
        self.corr = Correlation(pad_size=self.max_disp, kernel_size=1, max_displacement=self.max_disp, stride1=1, stride2=2, corr_multiply=1)

        # 2. Predict offets        
        in_ch_offset = in_ch + (self.max_disp)**2
        out_ch_offset = 2 * kernel_size * kernel_size
        self.pred_offset = nn.ModuleList( [ 
            nn.Conv2d( in_ch_offset, out_ch_offset, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Conv2d( in_ch_offset, out_ch_offset, kernel_size=3, padding=1, stride=1, bias=False)
            ] )

        # 3. Predict weights
        in_ch_weights = 2*in_ch + (self.max_disp)**2
        out_ch_weights = dims
        self.pred_weight = nn.ModuleList( [ 
            PredWeights(in_ch_weights, out_ch_weights, kernel_size),
            PredWeights(in_ch_weights, out_ch_weights, kernel_size)
            ] )

        # 4. Adaptive deformed conv.
        self.encoding = nn.ModuleList([ 
            nn.Conv2d( 256, out_ch_weights, kernel_size=1 ),
            nn.Conv2d( 256, out_ch_weights, kernel_size=1 )
            ])

        self.adaptive_offset_conv = nn.ModuleList( [ 
            AdaptiveConvOffset2d( dims, dims, \
                kernel_size=kernel_size, stride=1, padding=self.padding, \
                num_deformable_groups = t_window-1),
            AdaptiveConvOffset2d( dims, dims, \
                kernel_size=kernel_size, stride=1, padding=self.padding, \
                num_deformable_groups = t_window-1)
            ] )        

        # self.bn_x = nn.BatchNorm2d(256, affine=False)
        # self.bn_y = nn.BatchNorm2d(256, affine=False)

        # 5. Final conv. for fusion
        self.fusion = nn.Conv2d( 2*dims, out_ch, kernel_size=3, padding=1, stride=1, bias=False)

    
    def reset_parameters(self):
        pass

    def forward(self, R0, T0, inputs):

        if self.swap:
            R0, T0 = T0, R0

        BT, C, H, W = inputs.shape
        ### R0: same modality with inputs
        _R_pre = F.adaptive_avg_pool2d( R0[:,0,:,:,:].detach(), (H, W) )
        _R_cur = F.adaptive_avg_pool2d( R0[:,1,:,:,:].detach(), (H, W) )
        
        ### T0: the other modality with inputs
        _T_cur = F.adaptive_avg_pool2d( T0[:,1,:,:,:].detach(), (H, W) )


        x = inputs[0::2]    # previous frame
        y = inputs[1::2]    # current frame

        # x_bn = self.bn_x(x.contiguous())
        # y_bn = self.bn_y(y.contiguous())

        # input_for_offset0 = torch.cat( [ self.corr( _R_pre, _T_cur ), x_bn ], 1 )
        # input_for_weight0 = torch.cat( [ input_for_offset0, y_bn ], 1 )

        input_for_offset0 = torch.cat( [ self.corr( _R_pre, _T_cur ), x ], 1 )
        input_for_weight0 = torch.cat( [ input_for_offset0, y ], 1 )

        offset0 = self.pred_offset[0]( input_for_offset0 )
        weight0, bias0 = self.pred_weight[0]( input_for_weight0 )

        # To apply batch-wise different weight,        
        # output0 = self.adaptive_offset_conv[0]( x, offset0, weight0 )

        output0 = []
        x_encoded = self.encoding[0](x)
        for ii in range(int(BT/2)):
            output0.append( self.adaptive_offset_conv[0]( x_encoded[ii:ii+1], offset0[ii:ii+1], weight0[ii] ) + bias0[ii:ii+1] )
        output0 = torch.cat( output0, 0 )


        # input_for_offset1 = torch.cat( [ self.corr( _R_cur, _T_cur ), y_bn ], 1 )
        # input_for_weight1 = torch.cat( [ input_for_offset1, x_bn ], 1 )
        input_for_offset1 = torch.cat( [ self.corr( _R_cur, _T_cur ), y ], 1 )
        input_for_weight1 = torch.cat( [ input_for_offset1, x ], 1 )

        offset1 = self.pred_offset[1]( input_for_offset1 )
        weight1, bias1 = self.pred_weight[1]( input_for_weight1 )
        # output1 = self.adaptive_offset_conv[1]( y, offset1, weight1 )

        y_encoded = self.encoding[1](y)
        output1 = []
        for ii in range(int(BT/2)):
            output1.append( self.adaptive_offset_conv[1]( y_encoded[ii:ii+1], offset1[ii:ii+1], weight1[ii] ) + bias1[ii:ii+1] )
        output1 = torch.cat( output1, 0 )

        fusion = self.fusion( torch.cat( [ output0, output1 ], 1 ) )

        return fusion


# self.offset = nn.Conv2d( in_ch + (self.max_disp)**2, num_groups*2 * kernel_size * kernel_size, 

class STSN_Offset(nn.Module):
    
    N = 4

    def __init__(self, in_ch, kernel_size):
        super(STSN_Offset, self).__init__()    

        self.offset = nn.ModuleList()
        self.deform = nn.ModuleList()

        for ii in range(self.N-1):
            self.offset += [ nn.Conv2d( in_ch, 2*kernel_size*kernel_size, kernel_size=3, padding=1, bias=False ) ]
            self.deform += [ ConvOffset2d( in_ch, in_ch, kernel_size=3, padding=1 ) ]
        
        self.offset += [ nn.Conv2d( in_ch, 2*kernel_size*kernel_size, kernel_size=3, padding=1, bias=False ) ]


    def forward(self, x, y):
        feat = torch.cat( [x, y], 1 )

        for ii in range(self.N-1):
            offset = self.offset[ii](feat)
            feat = self.deform[ii](feat, offset)
        offset = self.offset[-1](feat)

        return offset



class SpatioTemporalFusion(nn.Module):    

    # def __init__(self, in_ch, out_ch, num_groups, kernel_size=3, n_iter=1):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        
        super(SpatioTemporalFusion, self).__init__()
                
        self.in_ch = in_ch
        self.out_ch = out_ch                        
        self.kernel_size = kernel_size
        # self.n_iter = n_iter         
        # self.num_groups = num_groups

        self.padding = math.floor(kernel_size/2)
        
        ## 1. Predict offset (we have two frames for input)
        in_ch_offset = 2*in_ch # stack of x, y        
        self.pred_offset = STSN_Offset(in_ch_offset, kernel_size)

        ## 2. Predict deformed feature: g_{t,t-k}^(4)
        in_ch_deform = in_ch
        out_ch_deform = out_ch
        self.pred_deform = ConvOffset2d(in_ch_deform, out_ch_deform, \
            kernel_size=kernel_size, padding=self.padding)

        ## 3. 3-layer weight prediction network: S(x)
        in_ch_weight = out_ch_deform        
        self.S = nn.Sequential(
            nn.Conv2d( in_ch_weight, int(in_ch_weight/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d( int(in_ch_weight/2), int(in_ch_weight/4), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d( int(in_ch_weight/4), 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )


    def forward(self, d1, d2, inputs):

        x = inputs[0::2]    # previous frame
        y = inputs[1::2]    # current frame

        offset0 = self.pred_offset( x, y )  # o_{t,t-1}^(4)        
        deform0 = self.pred_deform( x, offset0 )
        weight0 = self.S( deform0 )

        offset1 = self.pred_offset( y, y )  # o_{t,t}^(4)
        deform1 = self.pred_deform( y, offset1 )
        weight1 = self.S( deform1 )

    
        ## Compute cosine similarity
        Wx = F.cosine_similarity( weight0, weight1 )
        Wy = F.cosine_similarity( weight1, weight1 )    # Should be equal to torch.ones_like(Wx)

        # Ensure sum of weights at any location p == 1
        weights = F.softmax( torch.stack( [Wx, Wy], 1 ), dim=1 )

        fusion = deform0 * weights[:,0:1] + deform1 * weights[:,1:2]

        return fusion


class ASTSN_Weight(nn.Module):
    
    N = 2
    MAX_DISPLACEMENT = 7

    def __init__(self, in_ch, out_ch, kernel_size):
        super(ASTSN_Weight, self).__init__()    

        self.in_ch = in_ch        
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = math.floor(kernel_size/2)

        self.corr = Correlation(pad_size=self.MAX_DISPLACEMENT, kernel_size=1, 
            max_displacement=self.MAX_DISPLACEMENT, stride1=1, stride2=2, corr_multiply=1)

        self.weight = nn.ModuleList()
        # self.deform = nn.ModuleList()

        in_ch_weight = 2*in_ch + (self.MAX_DISPLACEMENT)**2       # stack of x, y

        for ii in range(self.N-1):
            self.weight += [ nn.Sequential(
                nn.Conv2d( in_ch_weight, in_ch, kernel_size=5, padding=2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                
                nn.Conv2d( in_ch, in_ch, kernel_size=5, padding=2, stride=2, bias=False),                
                nn.ReLU(inplace=True),
                
                nn.Conv2d( in_ch, in_ch_weight, kernel_size=3, padding=1, stride=2, bias=False),
            ) ]
            # self.deform += [ nn.Conv2d( in_ch, in_ch, kernel_size=3, padding=1 ) ]
        self.deform = AdaptiveConv2d(padding=self.padding, stride=1, dilation=1)
        
        self.weight += [ nn.Sequential(
                nn.Conv2d( in_ch_weight, in_ch, kernel_size=5, padding=2, stride=2, bias=False),
                nn.ReLU(inplace=True),
                
                nn.Conv2d( in_ch, in_ch, kernel_size=5, padding=2, stride=2, bias=False),                
                nn.ReLU(inplace=True),
                
                nn.Conv2d( in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            ) ]

        self.wx_shape = [ in_ch_weight, in_ch_weight, kernel_size, kernel_size ]
        self.bx_shape = [ in_ch_weight ]

        
        # self.wx = nn.Linear( in_ch_weight, int(np.prod(self.wx_shape)) )
        # self.bx = nn.Linear( in_ch_weight, int(np.prod(self.bx_shape)) )
        groups = in_ch_weight
        self.wx = nn.Conv2d( in_ch_weight, int(np.prod(self.wx_shape)), groups=groups, kernel_size=1, padding=0, stride=1 )
        # self.bx = nn.Conv2d( in_ch_weight, int(np.prod(self.bx_shape)), kernel_size=1, padding=0, stride=1 )


        self.wx_shape_final = [ out_ch, out_ch, kernel_size, kernel_size ]
        self.bx_shape_final = [ out_ch ]            

        # self.wx_final = nn.Linear( in_ch_weight, int(np.prod(self.wx_shape_final)) )
        # self.bx_final = nn.Linear( in_ch_weight, int(np.prod(self.bx_shape_final)) )
        groups = out_ch
        self.wx_final = nn.Conv2d( out_ch, int(np.prod(self.wx_shape_final)), groups=groups, kernel_size=1, padding=0, stride=1 )
        # self.bx_final = nn.Conv2d( out_ch, int(np.prod(self.bx_shape_final)), kernel_size=1, padding=0, stride=1 )
       

        

    def forward(self, x0, y0, x, y):

        B, C, H, W = x.shape

        x0 = F.adaptive_avg_pool2d( x0, (H, W) )
        y0 = F.adaptive_avg_pool2d( y0, (H, W) )

        corr = self.corr( x0, y0 )        
        feat = torch.cat( [corr, x, y], 1 )

        for ii in range(self.N-1):

            # Predict weight/bias
            feat_weight = self.weight[ii](feat)

            # feat_weight = F.adaptive_avg_pool2d( feat_weight, (1,1) ).view(B, -1)
            feat_weight = F.adaptive_avg_pool2d( feat_weight, (1,1) )

            wx = self.wx(feat_weight).view( [B] + self.wx_shape )
            # bx = self.bx(feat_weight).view( [B] + self.bx_shape )

            # Apply predicted weight/bias to make deformed feat
            # feat = self.deform[ii](feat, wx, bx)
            # feat = self.deform(feat, wx, bx)
            feat = F.relu( self.deform(feat, wx) )
            
        feat_weight = self.weight[-1](feat)
        # feat_weight = F.adaptive_avg_pool2d( feat_weight, (1,1) ).view(B, -1)
        feat_weight = F.adaptive_avg_pool2d( feat_weight, (1,1) )
        
        wx = self.wx_final(feat_weight).view( [B] + self.wx_shape_final )
        # bx = self.bx_final(feat_weight).view( [B] + self.bx_shape_final )

        # return wx, bx
        return wx



class AdaptiveSpatioTemporalFusion(nn.Module):
    
    def __init__(self, in_ch, out_ch, swap, fusion_mode='sum', kernel_size=3):
        
        super(AdaptiveSpatioTemporalFusion, self).__init__()
                
        assert fusion_mode in ['sum', 'concat']

        self.in_ch = in_ch
        self.out_ch = out_ch                        
        self.kernel_size = kernel_size
        self.fusion_mode = fusion_mode
        self.swap = swap

        # self.n_iter = n_iter         
        # self.num_groups = num_groups

        self.padding = math.floor(kernel_size/2)
        
        ## 1. Predict offset (we have two frames for input)
        in_ch_offset = 2*in_ch # stack of x, y        
        self.pred_offset = STSN_Offset(in_ch_offset, kernel_size)

        ## 2. Predict weights (we have two frames for input)
        dim = 64
        self.encoding = nn.ModuleList([ 
            nn.Conv2d( 256, dim, kernel_size=1 ),
            nn.Conv2d( 256, dim, kernel_size=1 )
            ])

        in_ch_weight = dim # channel of x
        out_ch_weight = in_ch
        self.pred_weight = ASTSN_Weight(in_ch_weight, out_ch_weight, kernel_size)

        ## 3. Predict deformed feature: g_{t,t-k}^(4)
        in_ch_deform = in_ch
        out_ch_deform = out_ch
        self.pred_deform = AdaptiveConvOffset2d(in_ch_deform, out_ch_deform, \
            kernel_size=kernel_size, padding=self.padding)
        # self.pred_deform = ConvOffset2d(in_ch_deform, out_ch_deform, \
        #     kernel_size=kernel_size, padding=self.padding)

        ## 4. 3-layer weight prediction network: S(x)
        in_ch_weight = out_ch_deform        
        self.S = nn.Sequential(
            nn.Conv2d( in_ch_weight, int(in_ch_weight/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d( int(in_ch_weight/2), int(in_ch_weight/4), kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d( int(in_ch_weight/4), 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )



    def forward(self, R0, T0, inputs):

        if self.swap:
            R0, T0 = T0, R0

        BT, C, H, W = inputs.shape

        ### R0: same modality with inputs
        _R_pre = R0[:,0,:,:,:].detach()
        _R_cur = R0[:,1,:,:,:].detach()
        
        ### T0: the other modality with inputs
        _T_cur = T0[:,1,:,:,:].detach()


        x = inputs[0::2]    # previous frame
        y = inputs[1::2]    # current frame

        x_encoded = self.encoding[0](x)
        y_encoded = self.encoding[1](y)

        offset0 = self.pred_offset( x, y )  # o_{t,t-1}^(4)        
        weight0 = self.pred_weight( _R_pre, _T_cur, x_encoded, y_encoded )  # w_{t,t-1}^(4)        

        
        deform0 = []        
        for ii in range(int(BT/2)):
            deform0.append( self.pred_deform( x[ii:ii+1], offset0[ii:ii+1], weight0[ii] ) )
        deform0 = torch.cat( deform0, 0 )



        # deform0 = self.pred_deform( x, offset0, weight0 )
        weight0 = self.S( deform0 )

        offset1 = self.pred_offset( y, y )  # o_{t,t}^(4)
        weight1 = self.pred_weight( _R_cur, _T_cur, y_encoded, y_encoded )  # w_{t,t-1}^(4)        

        deform1 = []        
        for ii in range(int(BT/2)):
            deform1.append( self.pred_deform( y[ii:ii+1], offset1[ii:ii+1], weight1[ii] ) )
        deform1 = torch.cat( deform1, 0 )


        # deform1 = self.pred_deform( y, offset1, weight1 )
        weight1 = self.S( deform1 )

    
        ## Compute cosine similarity
        Wx = F.cosine_similarity( weight0, weight1 )
        Wy = F.cosine_similarity( weight1, weight1 )    # Should be equal to torch.ones_like(Wx)

        # Ensure sum of weights at any location p == 1
        weights = F.softmax( torch.stack( [Wx, Wy], 1 ), dim=1 )

        if self.fusion_mode == 'sum':
            fusion = deform0 * weights[:,0:1] + deform1 * weights[:,1:2]
        elif self.fusion_mode == 'concat':
            fusion = torch.cat( [deform0 * weights[:,0:1], deform1 * weights[:,1:2]], 1)

        return fusion

    def extra_repr(self):
        s = 'in_ch={in_ch}, out_ch={out_ch}, kernel_size={kernel_size}, fusion_mode={fusion_mode}'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class TemporalFusion(nn.Module):

    max_disp = 7

    def __init__(self, in_ch, out_ch, num_groups, kernel_size=3, n_iter=1):
        
        super(TemporalFusion, self).__init__()
                

        self.in_ch = in_ch
        self.out_ch = out_ch                        
        self.kernel_size = kernel_size
        self.n_iter = n_iter 
        
        self.num_groups = num_groups

        self.padding = math.floor(kernel_size/2)
        
        # 1. correlation
        self.corr = Correlation(pad_size=self.max_disp, kernel_size=1, max_displacement=self.max_disp, stride1=1, stride2=2, corr_multiply=1)

        # 2. deformable conv.
        # self.offset = nn.Conv2d( (num_groups-1)*(in_ch + (self.max_disp)**2), (num_groups-1) * 2 * kernel_size * kernel_size, 
        #     groups=num_groups-1, kernel_size=3, padding=1, stride=1, bias=False)
        self.offset = nn.Conv2d( in_ch + (self.max_disp)**2, num_groups*2 * kernel_size * kernel_size, 
            kernel_size=3, padding=1, stride=1, bias=False)
        self.defconv = ConvOffset2d( in_ch, in_ch, (kernel_size, kernel_size), stride=1, padding=self.padding, \
            num_deformable_groups = num_groups)

        # 3. (static)fusion
        self.fusion = nn.Conv2d( in_ch*num_groups, out_ch, kernel_size=3, padding=1, stride=1, bias=False)


    def forward(self, d1, d2, inputs):

        # only consider t_window == 2 case. So, there are only x (t-1) and y (t).
        # x = inputs[0::self.num_deformable_groups]        # 2 will be changed to handle t_window
        # y = inputs[1::self.num_deformable_groups]
        

        # (t-k, t-k+1, ..., t-1) vs. t        

        # y = inputs[0::self.num_deformable_groups]
               
        y = inputs[self.num_groups-1::self.num_groups]        
        _y = y.detach()

        deformed_feats = []
        for ii in range( self.num_groups-1 ):

            x = inputs[ii::self.num_groups]
            _x = x.detach()

            # # CascadeSampling2
            # off = 0            
            for _ in range(self.n_iter):                    
                corr = self.corr( _x.contiguous(), _y.contiguous() )
                ip_offset = torch.cat( [x, corr], 1 )
                ot_offset = self.offset( ip_offset )

                # # CascadeSampling2
                # off += ot_offset

                x_deforms = self.defconv( x, ot_offset )

                x = x_deforms
                _x = x_deforms.detach()

            # # CascadeSampling2
            # deformed_feats.append( self.offset_conv( inputs[ii::self.num_groups], off/self.n_iter ) )
            deformed_feats.append( x_deforms )

        xy_concat = torch.cat( deformed_feats + [y], 1 )
        fusion = self.fusion( xy_concat )


        # bDraw = False
        # # pdb.set_trace()

        # if ot_offset.max().item() > 1.:
        
        #     B, G, H, W = ot_offset[:,:18,:,:].shape
        #     xx, yy = np.meshgrid( np.arange(W), np.arange(H) )
        #     xx = xx.reshape(1, 1, H, W)
        #     yy = yy.reshape(1, 1, H, W)
        #     xy_grid = np.tile( np.concatenate( (yy, xx), axis=1), (1, 9, 1, 1) )

           
        #     offset = ot_offset[:,:18,:,:].clone().cpu().numpy()
        #     pos = xy_grid + offset
        #     pos = pos.reshape( 18, -1 )

        #     pos_x = pos[1::2, :].reshape(-1)
        #     pos_y = pos[0::2, :].reshape(-1)

        #     plt.plot( pos_x, pos_y, 'b+',markersize=2)
        #     plt.grid(True, 'both', 'both', color='k')
        #     plt.savefig('tmp.jpg')

        #     pdb.set_trace()

        return fusion


    def __str__(self):
        s = ('{name} (in={in_ch}, out={out_ch}, deformable groups={num_deformable_groups}, k={kernel_size}, p={padding}) x {n_iter}')
        return s.format(name=self.__class__.__name__, **self.__dict__)




class AdaptiveFusion(nn.Module):
    def __init__(self, in_ch, dims, out_plane, kernel_size, fusion_type, fusion_resolution=None):

        super(AdaptiveFusion, self).__init__()

        assert fusion_type in FUSION_TYPES, \
            'Unknown fusion types. Should be one of them. {:s}.'.format( '_'.join(*FUSION_TYPES) )

        self.fusion_resolution = fusion_resolution
        self.fusion_type = fusion_type
        self.dims = dims
        self.out_plane = out_plane
        self.in_ch = in_ch
        self.out_plane = out_plane
        self.kernel_size = kernel_size
        self.padding = math.floor(kernel_size/2)

        max_disp = 10

        self.corr = Correlation(pad_size=max_disp, kernel_size=1, max_displacement=max_disp, stride1=1, stride2=2, corr_multiply=1)

        self.conv = nn.Sequential(
                nn.Conv2d( in_ch*4 + (max_disp+1)*(max_disp+1), 2*dims, kernel_size=3, padding=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d( 2*dims, 2*dims, kernel_size=3, padding=1, stride=1, bias=False),                
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # nn.Conv2d( dims, dims, kernel_size=20, padding=(2,0), stride=1),                
                nn.Conv2d( 2*dims, dims, kernel_size=3, padding=1, stride=1, bias=False),
            )
        self.conv_post = nn.Conv2d( dims, dims, kernel_size=1, padding=0, stride=1)
        
        if self.fusion_type == 'Mul_1x1':
            self.wx = nn.Linear( dims, dims )
            self.bx = nn.Linear( dims, dims )
            self.wy = nn.Linear( dims, dims )
            self.by = nn.Linear( dims, dims )

        elif self.fusion_type == 'Mul_1x1_Single':
            self.wx = nn.Linear( dims, dims )
            self.bx = nn.Linear( dims, dims )            

        elif self.fusion_type == 'Mul_HxW+1x1':
            self.wx = nn.ModuleList( [
                nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] ),
                nn.Linear( dims, dims ),
                ])
            self.wy = nn.ModuleList( [
                nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] ),
                nn.Linear( dims, dims ),
                ])
            self.bx = nn.ModuleList( [
                nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] ),
                nn.Linear( dims, dims ),
                ])
            self.by = nn.ModuleList( [
                nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] ),
                nn.Linear( dims, dims ),
                ])            

        elif self.fusion_type == 'Mul_HxW':
            self.wx = nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] )
            self.wy = nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] )
            self.bx = nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] )
            self.by = nn.Linear( dims, fusion_resolution[0]*fusion_resolution[1] )

        elif self.fusion_type == 'Conv':

            self.wx = nn.Linear( dims, dims*out_plane*kernel_size*kernel_size )
            self.bx = nn.Linear( dims, out_plane )
            # self.wy = nn.Linear( dims, dims*dims )
            # self.by = nn.Linear( dims, dims )

        else:
            raise NotImplementedError


    def forward(self, x, y, hx):
        
        # raise NotImplementedError

        bs = x.size(0)

        # _x = F.avg_pool2d(x.detach(), kernel_size=8, stride=8)
        # _y = F.avg_pool2d(y.detach(), kernel_size=8, stride=8)        
        
        # _x = F.adaptive_avg_pool2d(x.detach(), (96, 120) )
        # _y = F.adaptive_avg_pool2d(y.detach(), (96, 120) )
        # _x = x.detach()
        # _y = y.detach()


        ### Due to memory issue
        _x = F.avg_pool2d(x.detach(), kernel_size=4, stride=4)
        _y = F.avg_pool2d(y.detach(), kernel_size=4, stride=4)

        _x2 = F.max_pool2d(x.detach(), kernel_size=4, stride=4)
        _y2 = F.max_pool2d(y.detach(), kernel_size=4, stride=4)

        c = self.corr( _x, _y )                
        
        # feat = torch.cat( [_x, _y, c], 1 )
        feat = torch.cat( [_x, _y, _x2, _y2, c], 1 )
        # feat = F.avg_pool2d( self.conv(feat), kernel_size=11, padding=(2,0) )
        feat = F.adaptive_avg_pool2d( self.conv(feat), (1, 1) )
        feat = self.conv_post( feat )        

        feat = feat.view( bs, -1 )
                        
        if self.fusion_type == 'Mul_1x1':
            wx = self.wx( feat ).view( bs, self.dims, 1, 1)
            bx = self.bx( feat ).view( bs, self.dims, 1, 1)
            wy = self.wy( feat ).view( bs, self.dims, 1, 1)
            by = self.by( feat ).view( bs, self.dims, 1, 1)

            return wx * hx + bx + wy * hy + by

        elif self.fusion_type == 'Mul_1x1_Single':
            wx = self.wx( feat ).view( bs, self.dims, 1, 1)
            bx = self.bx( feat ).view( bs, self.dims, 1, 1)
            
            gate = F.sigmoid(wx)
            return hx * gate + hy * (1-gate) + bx

        elif self.fusion_type == 'Mul_HxW+1x1':
            wx1 = self.wx[0]( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])
            bx1 = self.bx[0]( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])
            wx2 = self.wx[1]( feat ).view( bs, self.dims, 1, 1)
            bx2 = self.bx[1]( feat ).view( bs, self.dims, 1, 1)

            wy1 = self.wy[0]( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])
            by1 = self.by[0]( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])
            wy2 = self.wy[1]( feat ).view( bs, self.dims, 1, 1)
            by2 = self.by[1]( feat ).view( bs, self.dims, 1, 1)

            xx = wx2 * ( wx1 * hx + bx1 ) + bx2
            yy = wy2 * ( wy1 * hy + by1 ) + by2

            return xx + yy

        elif self.fusion_type == 'Mul_HxW':
            wx = self.wx( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])
            bx = self.bx( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])
            wy = self.wy( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])
            by = self.by( feat ).view( bs, 1, self.fusion_resolution[0], self.fusion_resolution[1])

            return wx * hx + bx + wy * hy + by
        
        elif self.fusion_type == 'Conv':
            xx = list()        
            # yy = list()        
    
            # wx = self.wx( feat ).view( bs, self.dims, self.dims, 1, 1)
            wx = self.wx( feat ).view( bs, self.dims, self.out_plane, 1, 1)
            bx = self.bx( feat ).view( bs, self.dims)
            # wy = self.wy( feat ).view( bs, self.dims, self.dims, 1, 1)
            # by = self.by( feat ).view( bs, self.dims)

            for ii in range(bs):
                xx.append( F.conv2d(hx[ii:ii+1,...], wx[ii,...], bias=bx[ii,...], padding=self.padding) )
                # yy.append( F.conv2d(hy[ii:ii+1,...], wy[ii,...], bias=by[ii,...], padding=0) )

            x = torch.cat( xx, 0 )
            # y = torch.cat( yy, 0 )

            # return x + y
            return x

    def __repr__(self):
        s = ('{name}(kernel_size={kernel_size}, padding={padding}, fusion_type={fusion_type}, resolution={fusion_resolution}, dimensions={dims})')
        return s.format(name=self.__class__.__name__, **self.__dict__)

class StaticFusion(nn.Module):
    def __init__(self, dims, out_plane, kernel_size, fusion_type, fusion_resolution=None):
        
        super(StaticFusion, self).__init__()
        
        assert fusion_type in FUSION_TYPES, \
            'Unknown fusion types. Should be one of them. {:s}.'.format( '_'.join(*FUSION_TYPES) )

        self.dims = dims
        self.fusion_resolution = fusion_resolution
        self.fusion_type = fusion_type
        self.out_plane = out_plane
        self.kernel_size = kernel_size
        self.padding = math.floor(kernel_size/2)

        if self.fusion_type == 'Mul_1x1':
            self.wx = nn.Parameter( torch.Tensor(dims, 1, 1) )
            self.wy = nn.Parameter( torch.Tensor(dims, 1, 1) )
            self.bx = nn.Parameter( torch.Tensor(dims, 1, 1) )
            self.by = nn.Parameter( torch.Tensor(dims, 1, 1) )

        if self.fusion_type == 'Mul_1x1_Single':
            self.wx = nn.Parameter( torch.Tensor(dims, 1, 1) )            
            self.bx = nn.Parameter( torch.Tensor(dims, 1, 1) )            

        elif self.fusion_type == 'Mul_HxW+1x1':
            self.wx1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.wx2 = nn.Parameter( torch.Tensor(dims, 1, 1) )

            self.wy1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.wy2 = nn.Parameter( torch.Tensor(dims, 1, 1) )

            self.bx1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.bx2 = nn.Parameter( torch.Tensor(dims, 1, 1) )
                
            self.by1 = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )
            self.by2 = nn.Parameter( torch.Tensor(dims, 1, 1) )

        elif self.fusion_type == 'Mul_HxW':
            self.wx = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )        
            self.wy = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )            

            self.bx = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )                            
            self.by = nn.Parameter( torch.Tensor(1, fusion_resolution[0], fusion_resolution[1]) )            
        
        elif self.fusion_type == 'Conv':
            self.wxbx = nn.Conv2d(dims, out_plane, kernel_size=kernel_size, padding=self.padding, stride=1)
            # self.wyby = nn.Conv2d(dims, dims, kernel_size=1, padding=0, stride=1)
        
        else:
            raise NotImplementedError



        self.reset_parameters()
                

    def reset_parameters(self):
        if self.fusion_type == 'Mul_1x1':
            init.normal_(self.wx.data, std=0.01)
            init.normal_(self.wy.data, std=0.01)
            self.bx.data.zero_()
            self.by.data.zero_()

        elif self.fusion_type == 'Mul_1x1_Single':
            init.normal_(self.wx.data, std=0.01)            
            self.bx.data.zero_()            

        elif self.fusion_type == 'Mul_HxW+1x1':
            init.normal_(self.wx1.data, std=0.01)
            init.normal_(self.wx2.data, std=0.01)
            init.normal_(self.wy1.data, std=0.01)
            init.normal_(self.wy2.data, std=0.01)
            self.bx1.data.zero_()
            self.bx2.data.zero_()
            self.by1.data.zero_()
            self.by2.data.zero_()

        elif self.fusion_type == 'Mul_HxW':
            init.normal_(self.wx.data, std=0.01)            
            init.normal_(self.wy.data, std=0.01)
            self.bx.data.zero_()            
            self.by.data.zero_()            

        elif self.fusion_type == 'Conv':
            init.normal_(self.wxbx.weight.data, std=0.01)
            # init.normal_(self.wyby.weight.data, std=0.01)
            self.wxbx.bias.data.zero_()
            # self.wyby.bias.data.zero_()
        else:
            raise NotImplementedError


    def forward(self, d1, d2, x):
        
        # BT, C, H, W = x.shape        
        # x = x.view(-1, C*T, H, W)

        
        # hy = hy.view(-1, C*T, H, W)
        

        if self.fusion_type == 'Mul_1x1':
            return x * self.wx + self.bx + y * self.wy + self.by

        elif self.fusion_type == 'Mul_1x1_Single':
            gate = F.sigmoid( self.wx )
            return x * gate  + y * (1 - gate) + self.bx

        elif self.fusion_type == 'Mul_HxW+1x1':
            x1 = x  * self.wx1 + self.bx1   # Spatial
            x2 = x1 * self.wx2 + self.bx2   # Channel        
            y1 = y  * self.wy1 + self.by1   # Spatial
            y2 = y1 * self.wy2 + self.by2   # Channel
            return x2 + y2

        elif self.fusion_type == 'Mul_HxW':
            x1 = x  * self.wx + self.bx   # Spatial            
            y1 = y  * self.wy + self.by   # Spatial            
            return x1 + y1

        elif self.fusion_type == 'Conv':
            # return 0.5*self.wxbx(x) + 0.5*self.wyby(y)
            return 0.5*self.wxbx(x)

        else:
            raise NotImplementedError


    def __str__(self):
        s = ('{name}(kernel_size={kernel_size}, padding={padding}, fusion_type={fusion_type}, resolution={fusion_resolution}, dimensions={dims})')
        return s.format(name=self.__class__.__name__, **self.__dict__)
