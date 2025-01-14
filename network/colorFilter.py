import torch
import torch.nn as nn
import numpy as np

class colorFilter(nn.Module):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
    Ref: Hong, et al., "A study of digital camera colorimetric characterization
        based on polynomial modeling." Color Research & Application, 2001. """
    def __init__(self):
        super(colorFilter, self).__init__()
        
        # Define the linear layer with 10 input features (r, g, b, rg, rb, gb, r^2, g^2, b^2, rgb) and 3 output features (r, g, b)
        self.linear = nn.Linear(10, 3)
        
        # Initialize the weights
        nn.init.normal_(self.linear.weight, std=0.02)
        # self.linear.weight.data.fill_(1e-3)  # Set all initial weights to 1e-5
        self.linear.weight.data[0,0] = 1    # Set the weights for r, g, b to 1
        self.linear.weight.data[1,1] = 1
        self.linear.weight.data[2,2] = 1
        self.sigmoid = nn.Sigmoid()
        # Initialize the biases to zero
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        # Extract r, g, b channels
        r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        
        # Compute the polynomial terms
        rg = r * g
        rb = r * b
        gb = g * b
        r2 = r * r
        g2 = g * g
        b2 = b * b
        rgb = r * g * b
        
        # Stack the terms along the channel dimension
        poly_terms = torch.stack([r, g, b, rg, rb, gb, r2, g2, b2, rgb], dim=1) # B 10 H W
        # Reshape to (batch_size, H*W, 10) for the linear layer
        batch_size, _, H, W = x.size()
        poly_terms = poly_terms.permute(0,2,3,1)   # B H W 10
        poly_terms = poly_terms.view(-1,10)
        
        # Apply the linear transformation
        transformed = self.linear(poly_terms)   # B*H*W 3
        # transformed = self.sigmoid(transformed)
        # Reshape back to (batch_size, 3, H, W)
        transformed = transformed.view(batch_size, H, W, 3)
        transformed = transformed.permute(0,3,1,2)  # B 3 H W
        
        return transformed

class colorFilter(nn.Module):
    ''' Another version, color filter based on CNN '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ct_conv_1 = nn.Conv2d(3,32,(1,1))
        self.ct_conv_2 = nn.Conv2d(32,32,(1,1))
        self.ct_conv_3 = nn.Conv2d(32,3,(1,1))
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.ct_conv_1(x)
        out = self.relu(out)
        out = self.ct_conv_2(out)
        out = self.relu(out)
        out = self.ct_conv_3(out)
        return out

if __name__ == '__main__':
    # Create a dummy input tensor of shape (1, 3, H, W)
    dummy_input = torch.tensor([0.1,0.2,0.7]).reshape((1,3,1,1))
    # Example usage with the final corrected model
    model_final = colorFilter()
    output_final = model_final(dummy_input)
    print(output_final)  # Should be (1, 3, 64, 64)
