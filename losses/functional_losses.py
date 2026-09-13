import torch
import math
import torch.nn as nn

def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    """central_diff_3d computes derivatives 
    df(x,y,z)/dx and df(x,y,z)/dy for f(x,y,z) defined 
    on a regular 2d grid using finite-difference

    Parameters
    ----------
    x : torch.Tensor
        input function defined x[:,i,j,k] = f(x_i, y_j,z_k)
    h : float or list
        discretization size of grid for each dimension
    fix_x_bnd : bool, optional
        whether to fix dx on the x boundaries, by default False
    fix_y_bnd : bool, optional
        whether to fix dy on the y boundaries, by default False
    fix_z_bnd : bool, optional
        whether to fix dz on the z boundaries, by default False

    Returns
    -------
    dx, dy, dz
        tuple such that dx[:, i,j,k]= df(x_i,y_j,z_k)/dx
        and dy[:, i,j,k]= df(x_i,y_j,z_k)/dy
        and dz[:, i,j,k]= df(x_i,y_j,z_k)/dz
    """
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]
    
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]
        
    return dx, dy, dz


class H1Loss(object):
    def __init__(self, d=1, measure=1., reduction='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        
        allowed_reductions = ["sum", "mean", 'none']
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"H1_{self.d}DLoss"
     
    def compute_terms(self, x, y, quadrature):
        dict_x = {}
        dict_y = {}
        
        if self.d == 3:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_quadrature(self, x):
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature
    
    def reduce_all(self, x):
        if self.reduction == 'sum':
            x = torch.sum(x)
        elif self.reduction == 'none':
            return x
        else:
            x = torch.mean(x)
        
        return x
        
    def abs(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = (diff**0.5)/(ynorm**0.5)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, quadrature=None, **kwargs):
        return self.rel(y_pred, y, quadrature=quadrature)


class LpLoss(object):
    def __init__(self, d=1, p=2, measure=1., reduction='sum'):
        super().__init__()

        self.d = d
        self.p = p
        
        allowed_reductions = ["sum", "mean", 'none']
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_quadrature(self, x):
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

    def reduce_all(self, x):
        if self.reduction == 'sum':
            x = torch.sum(x)
        elif self.reduction == 'none':
            return x
        else:
            x = torch.mean(x)
        
        return x

    def abs(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        const = math.prod(quadrature)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)


class HdivLoss(object):
    def __init__(self, d=1, measure=1., reduction='sum', eps=1e-8, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        
        self.eps = eps
        
        allowed_reductions = ["sum", "mean", 'none']
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"Hdiv_{self.d}DLoss"
     
    def compute_terms(self, x, y, quadrature):
        dict_x = {}
        dict_y = {}
        
        if self.d == 3:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            div_x = torch.flatten(x_x + x_y + x_z, start_dim=-3)
            div_y = torch.flatten(y_x + y_y + y_z, start_dim=-3)
        else:
            raise NotImplementedError("HdivLoss is only implemented for 3D at the moment.")
        
        dict_x[1] = div_x
        dict_y[1] = div_y
        
        return dict_x, dict_y

    def uniform_quadrature(self, x):
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature
    
    def reduce_all(self, x):
        if self.reduction == 'sum':
            x = torch.sum(x)
        elif self.reduction == 'none':
            return x
        else:
            x = torch.mean(x)
        
        return x
        
    def abs(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2  #compute L2 norm of x-y

        diff += const*torch.norm(dict_x[1] - dict_y[1], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, quadrature=None):
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        diff += torch.norm(dict_x[1] - dict_y[1], p=2, dim=-1, keepdim=False) ** 2
        ynorm += torch.norm(dict_y[1], p=2, dim=-1, keepdim=False) ** 2

        diff = (diff**0.5)/(ynorm**0.5 + self.eps)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, quadrature=None, **kwargs):
        return self.rel(y_pred, y, quadrature=quadrature)    