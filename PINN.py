import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
from pyDOE import lhs
import utilities as ut

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# Set random seeds for reproducibility
torch.manual_seed(1618)
np.random.seed(1618)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device == 'cuda': 
    print(f"GPU: {torch.cuda.get_device_name()}")


class PINNSolver:
    def __init__(self, steps, learning_rate, layers, x_min, x_max, t_min, t_max,
                 total_points_x, total_points_t, num_boundary_points, num_collocation_points,
                 pde_equation, exact_solution):
        # Training parameters
        self.steps = steps
        self.learning_rate = learning_rate
        self.layers = layers
        
        # Domain boundaries
        self.x_min = x_min
        self.x_max = x_max
        self.t_min = t_min
        self.t_max = t_max
        
        # Grid resolution for testing
        self.total_points_x = total_points_x
        self.total_points_t = total_points_t
        
        # Training set sizes
        self.num_boundary_points = num_boundary_points
        self.num_collocation_points = num_collocation_points
        
        # PDE and exact solution functions
        self.pde_equation = pde_equation
        self.exact_solution = exact_solution
        
        # Initialize neural network
        self.model = self._build_network()
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=False)
        
        # Loss function
        self.loss_function = nn.MSELoss(reduction='mean')
        
        # Training data (will be populated by prepare_training_data)
        self.x_boundary_train = None
        self.y_boundary_train = None
        self.x_collocation_train = None
        self.residual_target = None
        
        # Testing data (will be populated by prepare_testing_data)
        self.x_test = None
        self.y_test = None
        self.x_grid = None
        self.t_grid = None
        
    def _build_network(self):
        return FullyConnectedNetwork(self.layers)
    
    def prepare_testing_data(self):
        # Create uniform grid (keep on CPU for storage)
        x_values = torch.linspace(self.x_min, self.x_max, self.total_points_x)
        t_values = torch.linspace(self.t_min, self.t_max, self.total_points_t)
        x_mesh, t_mesh = torch.meshgrid(x_values, t_values, indexing='ij')
        
        # Store grids for plotting (keep on CPU)
        self.x_grid = x_values
        self.t_grid = t_values
        
        # Compute exact solution on grid
        y_exact = self.exact_solution(x_mesh, t_mesh)
        
        # Flatten for testing and move to device
        self.x_test = torch.hstack((
            x_mesh.transpose(1, 0).flatten()[:, None],
            t_mesh.transpose(1, 0).flatten()[:, None]
        )).float().to(device)
        
        self.y_test = y_exact.transpose(1, 0).flatten()[:, None].float().to(device)
        
        return x_mesh, t_mesh, y_exact
    
    def prepare_training_data(self):
        # First, create testing grid to extract boundaries
        x_mesh, t_mesh, y_exact = self.prepare_testing_data()
        
        # --- Boundary conditions ---
        
        # Left boundary: x = x_min for all t
        left_boundary_x = torch.hstack((
            x_mesh[:, 0].reshape(-1, 1),
            t_mesh[:, 0].reshape(-1, 1)
        ))
        left_boundary_y = self.exact_solution(left_boundary_x[:, 0], left_boundary_x[:, 1])
        
        # Bottom boundary: t = t_min for all x (initial condition)
        bottom_boundary_x = torch.hstack((
            x_mesh[0, :].reshape(-1, 1),
            t_mesh[0, :].reshape(-1, 1)
        ))
        bottom_boundary_y = self.exact_solution(bottom_boundary_x[:, 0], bottom_boundary_x[:, 1])
        
        # Top boundary: t = t_max for all x
        top_boundary_x = torch.hstack((
            x_mesh[-1, :].reshape(-1, 1),
            t_mesh[-1, :].reshape(-1, 1)
        ))
        top_boundary_y = self.exact_solution(top_boundary_x[:, 0], top_boundary_x[:, 1])
        
        # Combine all boundary data
        all_boundary_x = torch.vstack([left_boundary_x, bottom_boundary_x, top_boundary_x])
        all_boundary_y = torch.vstack([
            left_boundary_y.reshape(-1, 1),
            bottom_boundary_y.reshape(-1, 1),
            top_boundary_y.reshape(-1, 1)
        ])
        
        # Randomly sample boundary points
        boundary_indices = np.random.choice(
            all_boundary_x.shape[0], 
            self.num_boundary_points, 
            replace=False
        )
        self.x_boundary_train = all_boundary_x[boundary_indices, :].float().to(device)
        self.y_boundary_train = all_boundary_y[boundary_indices, :].float().to(device)
        
        # --- Collocation points (for PDE residual) ---
        
        # Domain bounds (move to CPU for numpy operations)
        lower_bound = self.x_test[0].cpu()
        upper_bound = self.x_test[-1].cpu()
        
        # Latin Hypercube Sampling for better space coverage
        lhs_samples = lhs(2, self.num_collocation_points)  # Returns numpy array
        x_collocation = lower_bound.numpy() + (upper_bound.numpy() - lower_bound.numpy()) * lhs_samples
        x_collocation = torch.from_numpy(x_collocation).float()
        
        # Include boundary points in collocation set
        self.x_collocation_train = torch.vstack((x_collocation, self.x_boundary_train.cpu())).float().to(device)
        
        # Target residual is zero everywhere (PDE should be satisfied)
        self.residual_target = torch.zeros(self.x_collocation_train.shape[0], 1).float().to(device)
    
    def compute_boundary_loss(self, x_boundary, y_boundary):
        prediction = self.model(x_boundary)
        return self.loss_function(prediction, y_boundary)
    
    def compute_pde_loss(self, x_collocation):
            # Enable gradient computation for collocation points
            x_col = x_collocation.clone()
            x_col.requires_grad = True
            
            # Forward pass
            u_prediction = self.model(x_col)
            
            # Compute first derivatives: du/dx and du/dt
            # grad_outputs shape should match output shape: [N, 1]
            u_grad = autograd.grad(
                outputs=u_prediction,
                inputs=x_col,
                grad_outputs=torch.ones([x_col.shape[0], 1]).to(device),
                retain_graph=True,
                create_graph=True
            )[0]  # Shape: [N, 2] where columns are [du/dx, du/dt]
            
            # Compute second derivatives: d²u/dx², d²u/dt²
            # grad_outputs shape should match u_grad shape: [N, 2]
            u_grad_grad = autograd.grad(
                outputs=u_grad,
                inputs=x_col,
                grad_outputs=torch.ones(x_col.shape).to(device),
                create_graph=True
            )[0]  # Shape: [N, 2] where columns are [d²u/dx², d²u/dt²]
            
            # Extract derivatives: u_t, u_xx
            u_t = u_grad[:, [1]]        # ∂u/∂t (second column) - use [[1]] to keep 2D
            u_xx = u_grad_grad[:, [0]]  # ∂²u/∂x² (first column) - use [[0]] to keep 2D
            
            # Compute PDE residual using provided equation
            pde_residual = self.pde_equation(x_col, u_t, u_xx)
            
            return self.loss_function(pde_residual, self.residual_target)
        
    def compute_total_loss(self):
        loss_boundary = self.compute_boundary_loss(self.x_boundary_train, self.y_boundary_train)
        loss_pde = self.compute_pde_loss(self.x_collocation_train)
        return loss_boundary + loss_pde
    
    def train(self):
        print(f"\n{'-'*60}")
        print(f"Starting PINN Training")
        print(f"{'-'*60}")
        print(f"Total steps: {self.steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Network architecture: {self.layers}")
        print(f"Boundary points: {self.num_boundary_points}")
        print(f"Collocation points: {self.num_collocation_points}")
        print(f"{'-'*60}\n")
        
        start_time = time.time()
        
        for iteration in range(self.steps):
            # Compute loss
            loss = self.compute_total_loss()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print progress every 10% of total steps
            if iteration % (self.steps // 10) == 0:
                with torch.no_grad():
                    test_loss = self.compute_boundary_loss(self.x_test, self.y_test)
                
                elapsed_time = time.time() - start_time
                print(f"Iteration {iteration:5d}/{self.steps} | "
                      f"Training Loss: {loss.item():.6f} | "
                      f"Testing Loss: {test_loss.item():.6f} | "
                      f"Time: {elapsed_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\n{'-'*60}")
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"{'-'*60}\n")
    
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
    
    def visualize_results(self):
        # Get predictions on test grid
        u_prediction = self.predict(self.x_test)
        
        # Reshape predictions to match grid (move to CPU first)
        u_pred_grid = u_prediction.cpu().numpy().reshape(self.total_points_t, self.total_points_x).T
        u_exact_grid = self.y_test.cpu().numpy().reshape(self.total_points_t, self.total_points_x).T
        
        # Compute error
        error = np.abs(u_exact_grid - u_pred_grid)
        
        # Convert grids to numpy (keep on CPU, so direct conversion)
        x_np = self.x_grid.numpy()
        t_np = self.t_grid.numpy()
        
        # Create 3D plots
        fig = plt.figure(figsize=(18, 5))
        
        # Plot 1: Exact solution
        ax1 = fig.add_subplot(131, projection='3d')
        X_plot, T_plot = np.meshgrid(x_np, t_np)
        ax1.plot_surface(X_plot, T_plot, u_exact_grid.T, cmap='viridis', alpha=0.9)
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u(x,t)')
        ax1.set_title('Exact Solution')
        
        # Plot 2: PINN prediction
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X_plot, T_plot, u_pred_grid.T, cmap='viridis', alpha=0.9)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('u(x,t)')
        ax2.set_title('PINN Prediction')
        
        # Plot 3: Absolute error
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(X_plot, T_plot, error.T, cmap='hot', alpha=0.9)
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('|Error|')
        ax3.set_title(f'Absolute Error (Max: {error.max():.2e})')
        
        plt.tight_layout()
        plt.show()
        
        # Print error statistics
        print(f"\n{'-'*60}")
        print(f"Error Statistics")
        print(f"{'-'*60}")
        print(f"Max absolute error:  {error.max():.6e}")
        print(f"Mean absolute error: {error.mean():.6e}")
        print(f"L2 relative error:   {np.linalg.norm(error) / np.linalg.norm(u_exact_grid):.6e}")
        print(f"{'-'*60}\n")


class FullyConnectedNetwork(nn.Module):    
    def __init__(self, layers):
        super().__init__()
        
        self.activation = nn.Tanh()
        
        # Create linear layers
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) 
            for i in range(len(layers) - 1)
        ])
        
        # Xavier initialization
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
    
    def forward(self, x):
        # Convert to tensor if needed
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        
        # Pass through hidden layers with activation
        a = x.float()
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        
        # Output layer (no activation)
        a = self.linears[-1](a)
        
        return a

