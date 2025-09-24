"""
Basis function implementations for KAN layers
Supports: Chebyshev, B-spline, Fourier, and PQC circuit algorithms
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any


class BasisFunctionFactory:
    """Factory for creating different basis functions - unified interface to compute basis tensors"""
    
    @staticmethod
    def create_basis_function(basis_type: str, num_basis: int, **kwargs) -> nn.Module:
        """
        Create basis function instance based on type
        
        Args:
            basis_type: Type of basis function ('chebyshev', 'b_spline', 'fourier', 'pqc')
            num_basis: Number of basis functions
            **kwargs: Additional parameters for specific basis functions
            
        Returns:
            nn.Module: Basis function instance
        """
        if basis_type == "chebyshev":
            return ChebyshevBasis(num_basis, **kwargs)
        elif basis_type == "b_spline":
            return BSplineBasis(num_basis, **kwargs)
        elif basis_type == "fourier":
            return FourierBasis(num_basis, **kwargs)
        elif basis_type == "pqc":
            return PQCBasis(num_basis, **kwargs)
        elif basis_type == "pqc_gpu":
            return PQCGpuBasis(num_basis, **kwargs)
        else:
            raise ValueError(f"Unsupported basis function type: {basis_type}")


class ChebyshevBasis(nn.Module):
    """Chebyshev polynomial basis functions - maintains current behavior for backward compatibility"""
    
    def __init__(self, num_basis: int, spline_order: int = 3, **kwargs):
        super().__init__()
        self.num_basis = num_basis
        self.spline_order = spline_order
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Chebyshev basis functions using current implementation logic
        Returns shape: [batch, input_dim, num_basis]
        """
        batch_size, input_dim = x.shape
        
        # Normalize input to [-1, 1] range (Chebyshev polynomials standard domain)
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True) + 1e-8
        x_normalized = (x - x_mean) / x_std
        x_grid = torch.tanh(x_normalized)  # Non-linear mapping to [-1,1]
        
        # Generate basis functions for each input dimension
        basis_functions_list = []
        
        for dim_idx in range(input_dim):
            x_dim = x_grid[:, dim_idx:dim_idx+1]  # [batch_size, 1]
            dim_basis = []
            
            # T_0(x) = 1
            dim_basis.append(torch.ones_like(x_dim))
            
            if self.num_basis > 1:
                # T_1(x) = x
                dim_basis.append(x_dim)
            
            # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x) (Chebyshev recurrence)
            for n in range(2, self.num_basis):
                if len(dim_basis) >= 2:
                    t_next = 2 * x_dim * dim_basis[-1] - dim_basis[-2]
                    t_next = torch.clamp(t_next, -5.0, 5.0)  # Numerical stability
                    dim_basis.append(t_next)
                else:
                    # Safe fallback
                    dim_basis.append(torch.zeros_like(x_dim))
            
            # Ensure correct number of basis functions
            while len(dim_basis) < self.num_basis:
                dim_basis.append(torch.zeros_like(x_dim))
            
            # Truncate to correct number
            dim_basis = dim_basis[:self.num_basis]
            
            # Stack to [batch_size, num_basis]
            dim_basis_tensor = torch.cat(dim_basis, dim=1)
            basis_functions_list.append(dim_basis_tensor)
        
        # Stack to [batch_size, input_dim, num_basis]
        basis_tensor = torch.stack(basis_functions_list, dim=1)
        
        return basis_tensor


class BSplineBasis(nn.Module):
    """True B-spline basis functions implementation using de Boor's algorithm"""
    
    def __init__(self, num_basis: int, spline_order: int = 3, grid_size: int = 8, **kwargs):
        super().__init__()
        self.num_basis = num_basis
        self.spline_order = spline_order
        self.grid_size = grid_size
        
        # Generate knot vector for B-splines
        self.register_buffer('knot_vector', self._generate_knot_vector())
        
    def _generate_knot_vector(self) -> torch.Tensor:
        """Generate uniform knot vector for B-splines"""
        # Create uniform knots in [-1, 1] range
        knots = torch.linspace(-1.0, 1.0, self.grid_size + 2 * self.spline_order + 1)
        return knots
    
    def _bspline_basis_function(self, x: torch.Tensor, i: int, p: int) -> torch.Tensor:
        """Compute B-spline basis function using Cox-de Boor recursion"""
        knots = self.knot_vector
        
        if p == 0:
            # Base case: B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, 0 otherwise
            return ((x >= knots[i]) & (x < knots[i + 1])).float()
        else:
            # Recursive case: B_{i,p}(x) = (x - t_i)/(t_{i+p} - t_i) * B_{i,p-1}(x) + 
            #                              (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) * B_{i+1,p-1}(x)
            result = torch.zeros_like(x)
            
            # First term
            denom1 = knots[i + p] - knots[i]
            if denom1 > 1e-8:
                term1 = (x - knots[i]) / denom1 * self._bspline_basis_function(x, i, p - 1)
                result += term1
            
            # Second term
            denom2 = knots[i + p + 1] - knots[i + 1]
            if denom2 > 1e-8:
                term2 = (knots[i + p + 1] - x) / denom2 * self._bspline_basis_function(x, i + 1, p - 1)
                result += term2
            
            return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions
        Returns shape: [batch, input_dim, num_basis]
        """
        batch_size, input_dim = x.shape
        
        # Normalize input to [-1, 1] range
        x_normalized = torch.clamp(x, min=-2.0, max=2.0) / 2.0
        
        # Generate basis functions for each input dimension
        basis_functions_list = []
        
        for dim_idx in range(input_dim):
            x_dim = x_normalized[:, dim_idx:dim_idx+1]  # [batch_size, 1]
            dim_basis = []
            
            # Generate B-spline basis functions
            for i in range(self.num_basis):
                basis_func = self._bspline_basis_function(x_dim, i, self.spline_order)
                dim_basis.append(basis_func)
            
            # Stack to [batch_size, num_basis]
            dim_basis_tensor = torch.cat(dim_basis, dim=1)
            basis_functions_list.append(dim_basis_tensor)
        
        # Stack to [batch_size, input_dim, num_basis]
        basis_tensor = torch.stack(basis_functions_list, dim=1)
        
        return basis_tensor


class FourierBasis(nn.Module):
    """Fourier basis functions: [1, sin(kx), cos(kx)] for k=1..K"""
    
    def __init__(self, num_basis: int, frequencies: Optional[torch.Tensor] = None, **kwargs):
        super().__init__()
        self.num_basis = num_basis
        
        if frequencies is None:
            # Default frequencies: 1, 2, 3, ..., K
            self.frequencies = torch.arange(1, num_basis, dtype=torch.float32)
        else:
            self.frequencies = frequencies
            
        # Ensure we have the right number of basis functions
        if len(self.frequencies) != num_basis - 1:  # -1 because we include constant term
            self.frequencies = torch.arange(1, num_basis, dtype=torch.float32)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier basis functions
        Returns shape: [batch, input_dim, num_basis]
        """
        batch_size, input_dim = x.shape
        
        # Normalize input to [0, 2π] range for Fourier series
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 * math.pi
        
        # Generate basis functions for each input dimension
        basis_functions_list = []
        
        for dim_idx in range(input_dim):
            x_dim = x_normalized[:, dim_idx:dim_idx+1]  # [batch_size, 1]
            dim_basis = []
            
            # Constant term: 1
            dim_basis.append(torch.ones_like(x_dim))
            
            # Fourier terms: sin(kx), cos(kx)
            for k in self.frequencies:
                sin_term = torch.sin(k * x_dim)
                cos_term = torch.cos(k * x_dim)
                dim_basis.append(sin_term)
                dim_basis.append(cos_term)
            
            # Ensure we have exactly num_basis functions
            while len(dim_basis) < self.num_basis:
                dim_basis.append(torch.zeros_like(x_dim))
            
            # Truncate to correct number
            dim_basis = dim_basis[:self.num_basis]
            
            # Stack to [batch_size, num_basis]
            dim_basis_tensor = torch.cat(dim_basis, dim=1)
            basis_functions_list.append(dim_basis_tensor)
        
        # Stack to [batch_size, input_dim, num_basis]
        basis_tensor = torch.stack(basis_functions_list, dim=1)
        
        return basis_tensor


class PQCBasis(nn.Module):
    """Parameterized Quantum Circuit basis functions - CPU-computable approximation"""
    
    def __init__(self, num_basis: int, num_qubits: int = 4, **kwargs):
        super().__init__()
        self.num_basis = num_basis
        self.num_qubits = num_qubits
        
        # Initialize parameterized quantum circuit parameters
        self.theta = nn.Parameter(torch.randn(num_qubits * 2) * 0.1)
        self.phi = nn.Parameter(torch.randn(num_qubits * 2) * 0.1)
        
    def _quantum_circuit_simulation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate parameterized quantum circuit on CPU
        This is a simplified approximation of quantum circuit behavior
        """
        batch_size = x.shape[0]
        
        # Simulate quantum state preparation
        # For simplicity, we use trigonometric functions to approximate quantum behavior
        x_normalized = torch.tanh(x)  # Normalize to [-1, 1]
        
        # Simulate quantum gates with parameterized rotations
        quantum_features = []
        
        for i in range(self.num_qubits):
            # Rotation gates simulation
            rot_x = torch.cos(self.theta[i] * x_normalized + self.phi[i])
            rot_y = torch.sin(self.theta[i] * x_normalized + self.phi[i])
            quantum_features.append(rot_x)
            quantum_features.append(rot_y)
        
        # Combine quantum features
        quantum_state = torch.cat(quantum_features, dim=-1)
        
        # Simulate measurement (expectation values)
        measurement = torch.mean(quantum_state, dim=-1, keepdim=True)
        
        return measurement
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PQC basis functions
        Returns shape: [batch, input_dim, num_basis]
        """
        batch_size, input_dim = x.shape
        
        # Generate basis functions for each input dimension
        basis_functions_list = []
        
        for dim_idx in range(input_dim):
            x_dim = x[:, dim_idx:dim_idx+1]  # [batch_size, 1]
            
            # Simulate quantum circuit for this dimension
            quantum_output = self._quantum_circuit_simulation(x_dim)
            
            # Generate basis functions based on quantum output
            dim_basis = []
            
            # Constant term
            dim_basis.append(torch.ones_like(x_dim))
            
            # Quantum-inspired basis functions
            for i in range(self.num_basis - 1):
                # Use quantum output to generate non-linear basis functions
                basis_func = torch.sin(quantum_output * (i + 1) * math.pi)
                dim_basis.append(basis_func)
            
            # Stack to [batch_size, num_basis]
            dim_basis_tensor = torch.cat(dim_basis, dim=1)
            basis_functions_list.append(dim_basis_tensor)
        
        # Stack to [batch_size, input_dim, num_basis]
        basis_tensor = torch.stack(basis_functions_list, dim=1)
        
        return basis_tensor


class PQCGpuBasis(nn.Module):
    """Parameterized Quantum Circuit basis functions using PennyLane GPU"""
    
    def __init__(self, num_basis: int, num_qubits: int = 4, device: str = 'cuda', **kwargs):
        super().__init__()
        self.num_basis = num_basis
        self.num_qubits = num_qubits
        self.device = device
        
        # Check if PennyLane is available
        try:
            import pennylane as qml
            self.qml = qml
            self.pennylane_available = True
        except ImportError:
            print("⚠️ PennyLane not available, falling back to CPU simulation")
            self.pennylane_available = False
        
        # Initialize quantum circuit parameters
        self.theta = nn.Parameter(torch.randn(num_qubits * 3) * 0.1)  # RX, RY, RZ rotations
        self.phi = nn.Parameter(torch.randn(num_qubits * 2) * 0.1)    # Phase parameters
        
        # Create quantum device
        if self.pennylane_available:
            try:
                self.qdev = qml.device('default.qubit', wires=num_qubits, shots=1000)
                self._create_quantum_circuit()
            except Exception as e:
                print(f"⚠️ Failed to create quantum device: {e}")
                self.pennylane_available = False
    
    def _create_quantum_circuit(self):
        """Create the parameterized quantum circuit using PennyLane"""
        if not self.pennylane_available:
            return
            
        @self.qml.qnode(device=self.qdev, interface='torch')
        def quantum_circuit(x, theta, phi):
            """
            Parameterized quantum circuit for feature encoding
            Args:
                x: Input data (batch_size,)
                theta: Rotation parameters (num_qubits * 3,)
                phi: Phase parameters (num_qubits * 2,)
            """
            # Data encoding layer
            for i in range(self.num_qubits):
                if i < len(x):
                    self.qml.RY(x[i] * np.pi, wires=i)  # Encode input data
            
            # Parameterized layers
            for layer in range(2):  # 2 layers of parameterized gates
                # Rotation gates
                for i in range(self.num_qubits):
                    # 確保索引不超出範圍
                    idx_base = i * 3
                    if idx_base + 2 < len(theta):
                        self.qml.RX(theta[idx_base], wires=i)
                        self.qml.RY(theta[idx_base + 1], wires=i)
                        self.qml.RZ(theta[idx_base + 2], wires=i)
                
                # Entangling gates
                for i in range(self.num_qubits - 1):
                    self.qml.CNOT(wires=[i, i + 1])
            
            # Measurement
            return [self.qml.expval(self.qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        self.quantum_circuit = quantum_circuit
    
    def _quantum_circuit_simulation_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute quantum circuit on GPU using PennyLane
        """
        if not self.pennylane_available:
            # Fallback to CPU simulation
            return self._quantum_circuit_simulation_cpu(x)
        
        batch_size = x.shape[0]
        quantum_outputs = []
        
        try:
            for i in range(batch_size):
                # Prepare input for quantum circuit
                x_input = x[i].cpu().numpy()
                
                # Execute quantum circuit
                q_output = self.quantum_circuit(x_input, self.theta, self.phi)
                quantum_outputs.append(torch.tensor(q_output, device=x.device))
            
            # Stack results
            quantum_output = torch.stack(quantum_outputs, dim=0)
            return quantum_output  # [batch_size, num_qubits]
            
        except Exception as e:
            print(f"⚠️ Quantum circuit execution failed: {e}, falling back to CPU simulation")
            return self._quantum_circuit_simulation_cpu(x)
    
    def _quantum_circuit_simulation_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback CPU simulation when PennyLane is not available
        """
        batch_size = x.shape[0]
        
        # Simulate quantum state preparation
        x_normalized = torch.tanh(x)  # Normalize to [-1, 1]
        
        # Simulate quantum gates with parameterized rotations
        quantum_features = []
        
        for i in range(self.num_qubits):
            # Rotation gates simulation
            rot_x = torch.cos(self.theta[i * 3] * x_normalized + self.phi[i * 2])
            rot_y = torch.sin(self.theta[i * 3 + 1] * x_normalized + self.phi[i * 2 + 1])
            rot_z = torch.cos(self.theta[i * 3 + 2] * x_normalized)
            quantum_features.append(rot_x)
            quantum_features.append(rot_y)
            quantum_features.append(rot_z)
        
        # Combine quantum features
        quantum_state = torch.cat(quantum_features, dim=-1)
        
        # Simulate measurement (expectation values)
        measurement = torch.mean(quantum_state, dim=-1, keepdim=True)
        
        return measurement
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PQC basis functions using PennyLane GPU
        Returns shape: [batch, input_dim, num_basis]
        """
        batch_size, input_dim = x.shape
        
        # Generate basis functions for each input dimension
        basis_functions_list = []
        
        for dim_idx in range(input_dim):
            x_dim = x[:, dim_idx:dim_idx+1]  # [batch_size, 1]
            
            # Execute quantum circuit for this dimension
            quantum_output = self._quantum_circuit_simulation_gpu(x_dim)
            
            # Generate basis functions based on quantum output
            dim_basis = []
            
            # Constant term
            dim_basis.append(torch.ones_like(x_dim))
            
            # Quantum-inspired basis functions
            for i in range(self.num_basis - 1):
                # Use quantum output to generate non-linear basis functions
                if len(quantum_output.shape) == 2 and quantum_output.shape[1] > 0:
                    # Use first qubit measurement for basis generation
                    q_measurement = quantum_output[:, 0:1]  # [batch_size, 1]
                else:
                    q_measurement = torch.zeros_like(x_dim)
                
                # Generate basis functions with quantum enhancement
                basis_func = torch.sin(q_measurement * (i + 1) * math.pi)
                dim_basis.append(basis_func)
            
            # Stack to [batch_size, num_basis]
            dim_basis_tensor = torch.cat(dim_basis, dim=1)
            basis_functions_list.append(dim_basis_tensor)
        
        # Stack to [batch_size, input_dim, num_basis]
        basis_tensor = torch.stack(basis_functions_list, dim=1)
        
        return basis_tensor


# Convenience function for backward compatibility
def create_basis_function(basis_type: str, num_basis: int, **kwargs) -> nn.Module:
    """Convenience function to create basis function instances"""
    return BasisFunctionFactory.create_basis_function(basis_type, num_basis, **kwargs)
