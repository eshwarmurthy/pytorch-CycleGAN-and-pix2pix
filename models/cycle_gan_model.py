import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torchvision import models
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ContentLoss(nn.Module):
    """Content Loss using VGG features"""
    
    def __init__(self, device='cpu'):
        super(ContentLoss, self).__init__()
        # Load pre-trained VGG19 and use features up to conv4_4 (before ReLU)
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential()
        
        # We'll use layers up to conv4_4 (layer 25 in VGG19 features)
        for i, layer in enumerate(vgg[:26]):  # Up to conv4_4
            self.feature_extractor.add_module(str(i), layer)
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.feature_extractor.to(device)
        self.criterion = nn.MSELoss()
        
    def forward(self, input_img, target_img):
        """Compute content loss between input and target images"""
        # Normalize images to [0,1] range if they're in [-1,1] range
        if input_img.min() < 0:
            input_img = (input_img + 1) / 2
        if target_img.min() < 0:
            target_img = (target_img + 1) / 2
            
        # Extract features
        input_features = self.feature_extractor(input_img)
        target_features = self.feature_extractor(target_img)
        
        # Compute MSE loss between features
        return self.criterion(input_features, target_features)


class HueLoss(nn.Module):
    """Hue Loss to preserve color hue during image translation"""
    
    def __init__(self, loss_type='l1'):
        super(HueLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        # Ensure input is in [0, 1] range
        if rgb.min() < 0:
            rgb = (rgb + 1) / 2
        
        rgb = torch.clamp(rgb, 0, 1)
        
        # RGB to HSV conversion
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        
        max_rgb, argmax_rgb = torch.max(rgb, dim=1, keepdim=True)
        min_rgb, _ = torch.min(rgb, dim=1, keepdim=True)
        
        delta = max_rgb - min_rgb
        
        # Saturation
        s = torch.where(max_rgb > 0, delta / max_rgb, torch.zeros_like(max_rgb))
        
        # Value
        v = max_rgb
        
        # Hue calculation
        h = torch.zeros_like(max_rgb)
        
        # Red is max
        idx_r = (argmax_rgb == 0) & (delta > 0)
        h[idx_r] = (60 * ((g[idx_r] - b[idx_r]) / delta[idx_r]) + 360) % 360
        
        # Green is max
        idx_g = (argmax_rgb == 1) & (delta > 0)
        h[idx_g] = (60 * ((b[idx_g] - r[idx_g]) / delta[idx_g]) + 120) % 360
        
        # Blue is max
        idx_b = (argmax_rgb == 2) & (delta > 0)
        h[idx_b] = (60 * ((r[idx_b] - g[idx_b]) / delta[idx_b]) + 240) % 360
        
        # Normalize hue to [0, 1]
        h = h / 360.0
        
        return h, s, v
    
    def circular_distance(self, h1, h2):
        """Compute circular distance between two hue values"""
        # Both h1 and h2 should be in [0, 1] range
        diff = torch.abs(h1 - h2)
        # Handle circular nature of hue
        circular_diff = torch.min(diff, 1.0 - diff)
        return circular_diff
    
    def forward(self, input_img, target_img, saturation_threshold=0.1):
        """
        Compute hue loss between input and target images
        
        Args:
            input_img: Generated image
            target_img: Target image  
            saturation_threshold: Minimum saturation to consider for hue loss
        """
        # Convert to HSV
        h_input, s_input, v_input = self.rgb_to_hsv(input_img)
        h_target, s_target, v_target = self.rgb_to_hsv(target_img)
        
        # Create mask for pixels with sufficient saturation
        # Only compute hue loss for colored pixels (not grayscale)
        sat_mask = ((s_input > saturation_threshold) | (s_target > saturation_threshold)).float()
        
        if sat_mask.sum() == 0:
            # If no colored pixels, return zero loss
            return torch.tensor(0.0, device=input_img.device, requires_grad=True)
        
        # Compute circular hue distance
        hue_diff = self.circular_distance(h_input, h_target)
        
        # Apply saturation mask and compute weighted loss
        masked_hue_diff = hue_diff * sat_mask
        
        # Compute mean loss only over valid pixels
        hue_loss = masked_hue_diff.sum() / (sat_mask.sum() + 1e-8)
        
        return hue_loss


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model with Content Loss and Hue Loss, 
    for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, lambda_identity, lambda_content, and lambda_hue for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Content loss: lambda_content * (||VGG(G_A(A)) - VGG(B)|| + ||VGG(G_B(B)) - VGG(A)||)
        Hue loss: lambda_hue * (||H(G_A(A)) - H(A)|| + ||H(G_B(B)) - H(B)||) where H extracts hue channel
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )
            parser.add_argument(
                "--lambda_content",
                type=float,
                default=0.5,
                help="weight for content loss using VGG features. This helps preserve semantic content during translation.",
            )
            parser.add_argument(
                "--lambda_hue",
                type=float,
                default=0.0,
                help="weight for hue preservation loss. This helps maintain color hue consistency during translation.",
            )
            parser.add_argument(
                "--hue_loss_type",
                type=str,
                default="l1",
                choices=["l1", "l2"],
                help="type of loss function for hue loss computation",
            )

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "content_A", "hue_A", "D_B", "G_B", "cycle_B", "idt_B", "content_B", "hue_B"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_B(A)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert opt.input_nc == opt.output_nc
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionContent = ContentLoss(self.device)  # define content loss using VGG features
            self.criterionHue = HueLoss(getattr(opt, 'hue_loss_type', 'l1'))  # define hue preservation loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_content = self.opt.lambda_content
        lambda_hue = getattr(self.opt, 'lambda_hue', 1.0)
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # Content loss - preserve semantic content using VGG features
        if lambda_content > 0:
            # Content loss for A->B translation: VGG features of fake_B should be similar to real_B
            self.loss_content_A = self.criterionContent(self.fake_B, self.real_B) * lambda_content
            # Content loss for B->A translation: VGG features of fake_A should be similar to real_A  
            self.loss_content_B = self.criterionContent(self.fake_A, self.real_A) * lambda_content
        else:
            self.loss_content_A = 0
            self.loss_content_B = 0
        
        # Hue loss - preserve color hue during translation
        if lambda_hue > 0:
            # Hue loss for A->B translation: preserve hue from real_A in fake_B
            self.loss_hue_A = self.criterionHue(self.fake_B, self.real_A) * lambda_hue
            # Hue loss for B->A translation: preserve hue from real_B in fake_A
            self.loss_hue_B = self.criterionHue(self.fake_A, self.real_B) * lambda_hue
        else:
            self.loss_hue_A = 0
            self.loss_hue_B = 0
        
        # combined loss and calculate gradients
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + 
                      self.loss_idt_A + self.loss_idt_B + self.loss_content_A + self.loss_content_B +
                      self.loss_hue_A + self.loss_hue_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights