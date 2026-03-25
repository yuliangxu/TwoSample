import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------
# 2. Define an MLP That Outputs w(x)
# ------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32,output_dim=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # ensures the output is > 0
        )
    
    def forward(self, x):
        return self.model(x)

# ------------------------------------------------------
# 3. Define the Empirical Loss Function
# ------------------------------------------------------
def Long_loss(model, x_p, x_q):
    """
    Approximates the loss:
       L(w) = - E_{x ~ p}[log(w(x))] + E_{x ~ q}[log(w(x))] - 1,
    using the empirical averages of samples from p and q.
    """
    # Compute w(x) for samples from p and q.
    w_p = model(x_p)  # shape: (N_p, 1)
    w_q = model(x_q)  # shape: (N_q, 1)
    
    # Compute the empirical averages:
    # For p(x): log(w(x))
    loss_p = torch.mean(torch.log(w_p))
    # For q(x): the square root, w(x)^{1/2}
    loss_q = torch.mean(torch.log(w_q))
    
    # Total loss is the sum of the two terms.
    loss = - loss_p + loss_q - 1
    return loss


def default_ratio_loss(model, x_p, x_q, xi=0.5):
    """
    Approximates the loss:
       L(w) = E_{x ~ p}[w(x)^{-1/2}] + E_{x ~ q}[w(x)^{1/2}],
    using the empirical averages of samples from p and q.
    """
    # Compute w(x) for samples from p and q.
    w_p = model(x_p)  # shape: (N_p, 1)
    w_q = model(x_q)  # shape: (N_q, 1)
    
    # Compute the empirical averages:
    # For p(x): the inverse square root, w(x)^{-1/2}
    loss_p = torch.mean(w_p.pow(-0.5))
    # For q(x): the square root, w(x)^{1/2}
    loss_q = torch.mean(w_q.pow(0.5))
    
    # Total loss is the sum of the two terms.
    loss = xi*loss_p + (1-xi)*loss_q
    return loss

def run_DRE(x_p,x_q,xi=0.5,num_epochs = 2000, loss_method='default',device = None, display_progress = False):

    input_dim = x_p.shape[1]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(x_p, np.ndarray):
        x_p = torch.from_numpy(x_p).to(dtype=torch.float32, device=device)
    if isinstance(x_q, np.ndarray):
        x_q = torch.from_numpy(x_q).to(dtype=torch.float32, device=device)
    
   

    model = MLP(input_dim=input_dim).to(device)
    # Move input tensors to the same device
    x_p = x_p.float().to(device)
    x_q = x_q.float().to(device)

    # ------------------------------------------------------
    # 4. Set Up the Optimizer and Training Loop
    # ------------------------------------------------------
    if loss_method == 'default':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler_DRE = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    elif loss_method == 'Long':
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    losses = []  # To store loss at each epoch

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        if loss_method == 'default':
            loss = default_ratio_loss(model, x_p, x_q, xi)
        elif loss_method == 'Long':
            loss = Long_loss(model, x_p, x_q)
        

        loss.backward()
        optimizer.step()
        scheduler_DRE.step(loss)
        
        losses.append(loss.item())

        show_iter = num_epochs // 10
        if display_progress:
            if epoch % show_iter == 0:
                print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")
        
        # stop if loss is na
        if torch.isnan(loss):
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")
            break
        

    return model,losses


# %% Modified GAN loss

def new_GAN_loss(DRE_model, x_p0 , x_pg):
    """
    Computes the loss:
       L(w) = -E_{x ~ p0}[ log( 1+w[x] ) ] + E_{x ~ pg}[ log( w[x]/{1+w[x]} ) ],
    """
    # Compute w(x) for the samples
    w_p0 = DRE_model(x_p0)  # for samples from p0
    w_pg = DRE_model(x_pg)  # for samples from p_q

    loss_p0 = torch.mean(torch.log(1 + w_p0))
    loss_pg = torch.mean(torch.log(w_pg / (1 + w_pg)))
    
    # Total loss is the sum of the two terms.
    loss = - loss_p0 + loss_pg
    return loss

def generator_loss(DRE_model, x_pg):
    """
    Computes the loss:
       L(w) = 0.5*E_{x ~ pg}[ log( w[x] ) ] + 0.5*E_{x ~ pg}[ (w[x]-1)^2 ],
    """
    w_pg = DRE_model(x_pg)  # for samples from p_q
    

    loss_pg = torch.mean(torch.log( w_pg))
    
    return loss_pg
# ------------------------------------------------------
# %% default loss with balance constraint
# ------------------------------------------------------
# 4. Define the Empirical Loss Function with balance constraint
# ------------------------------------------------------
def empirical_loss_balanced(model, x_p, x_q, scale, xi=0.5):
    """
    Computes the loss:
       L(w) = E_{x ~ p}[w(x)^{-1/2}] + E_{x ~ q}[w(x)^{1/2}],
    with w(x) = scale * model(x).
    """
    # Compute w(x) for the samples
    w_p = scale * model(x_p)  # for samples from p
    w_q = scale * model(x_q)  # for samples from q

    # Compute the two empirical expectations
    loss_p = torch.mean(w_p.pow(-0.5))  # E_p[w(x)^{-1/2}]
    loss_q = torch.mean(w_q.pow(0.5))     # E_q[w(x)^{1/2}]
    
    # Total loss is the sum of the two terms.
    loss = xi*loss_p + (1-xi)*loss_q
    return loss

def run_balDRE(x_p,x_q,xi=0.5,num_epochs = 2000):

    input_dim = x_p.shape[1]
    scale = torch.tensor(1.0, requires_grad=False)

    model = MLP(input_dim=input_dim)

    # ------------------------------------------------------
    # 4. Set Up the Optimizer and Training Loop
    # ------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []  # To store loss at each epoch

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = empirical_loss_balanced(model, x_p, x_q, scale, xi)

        loss.backward()
        optimizer.step()
        # -----------------------------------------------------
        with torch.no_grad():
            # Recompute w for the current model and scale
            w_p = scale * model(x_p)
            w_q = scale * model(x_q)
            Ep = torch.mean(w_p.pow(-0.5))
            Eq = torch.mean(w_q.pow(0.5))
            c_w = xi*Ep / (1-xi) /Eq
            # Update the scale parameter
            scale *= c_w
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")

    return model,losses


def ensure_tensor(x, dtype=torch.float32, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
    except ImportError:
        pass
    return np.asarray(x)

# CLR transform for compositional rows
def clr_transform(X):
    """
    X: (n × k) compositional array (each row sums to 1).
    Returns (n × k) array of CLR coordinates.
    """
    eps = 1e-16
    Xsafe = np.clip(X, eps, None)             # Avoid log(0)
    logX = np.log(Xsafe)
    gm   = np.exp(np.mean(logX, axis=1, keepdims=True))  # Geometric mean per row
    return logX - np.log(gm)                  # Subtract log(gm) from each entry
