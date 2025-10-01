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

        # Vectorized basis computation without in-place mutation of the output tensor
        # Build basis terms as independent tensors to keep autograd graph intact
        T0 = torch.ones_like(x_grid)
        basis_terms = [T0]

        if self.num_basis > 1:
            T1 = x_grid
            basis_terms.append(T1)

            for _ in range(2, self.num_basis):
                Tn = 2.0 * x_grid * basis_terms[-1] - basis_terms[-2]
                Tn = torch.clamp(Tn, -5.0, 5.0)
                basis_terms.append(Tn)

        # If fewer terms than required, pad with zeros
        while len(basis_terms) < self.num_basis:
            basis_terms.append(torch.zeros_like(x_grid))

        # Stack along the last dim to shape [batch, input_dim, num_basis]
        basis = torch.stack(basis_terms, dim=-1)
        return basis


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

        # Normalize input to [-1, 1] range (preserve original semantics)
        x_normalized = torch.clamp(x, min=-2.0, max=2.0) / 2.0  # [B, D]

        # Prepare knot-related tensors
        knots = self.knot_vector  # [M]
        M = knots.shape[0]
        p = self.spline_order

        # Base case B_{i,0}(x): indicator t_i <= x < t_{i+1}
        # Vectorize across batch and dims; i dimension is last
        x_exp = x_normalized.unsqueeze(-1)  # [B, D, 1]
        t_i = knots[:-1].view(1, 1, -1)     # [1,1,M-1]
        t_ip1 = knots[1:].view(1, 1, -1)    # [1,1,M-1]

        B_prev = ((x_exp >= t_i) & (x_exp < t_ip1)).to(x.dtype)  # [B, D, M-1]

        # Iteratively compute up to degree p using Cox–de Boor
        for deg in range(1, p + 1):
            # For degree 'deg', valid i range shrinks by 1 on the right each step
            # Denominators
            denom1 = (knots[deg:M-1] - knots[:M-1-deg]).view(1, 1, -1)  # [1,1,M-1-deg]
            denom2 = (knots[deg+1:M] - knots[1:M-deg]).view(1, 1, -1)   # [1,1,M-1-deg]

            # Align B_prev segments
            left = B_prev[..., :B_prev.shape[-1]-1]   # [B,D,M-2-(deg-1)+1] => [B,D,M-1-deg]
            right = B_prev[..., 1:]                   # [B,D,M-1-deg]

            # Compute coefficients a and b with safe division
            x_left = x_exp[..., :left.shape[-1]]      # [B,D,M-1-deg]
            x_right = x_exp[..., :right.shape[-1]]    # [B,D,M-1-deg]

            # a = (x - t_i) / (t_{i+deg} - t_i)
            num1 = x_left - knots[:M-1-deg].view(1, 1, -1)
            a = torch.where(denom1.abs() > 1e-8, num1 / denom1, torch.zeros_like(num1))

            # b = (t_{i+deg+1} - x) / (t_{i+deg+1} - t_{i+1})
            num2 = knots[deg+1:M].view(1, 1, -1) - x_right
            b = torch.where(denom2.abs() > 1e-8, num2 / denom2, torch.zeros_like(num2))

            B_curr = a * left + b * right            # [B,D,M-1-deg]
            B_prev = B_curr

        # After degree p, B_prev corresponds to B_{i,p} for i=0..M-1-p-1 (length L)
        L = B_prev.shape[-1]
        # Select first num_basis; pad zeros if needed
        if self.num_basis <= L:
            basis_tensor = B_prev[..., :self.num_basis]
        else:
            pad = torch.zeros(batch_size, input_dim, self.num_basis - L, dtype=B_prev.dtype, device=B_prev.device)
            basis_tensor = torch.cat([B_prev, pad], dim=-1)

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

        # Normalize input to [0, 2π] range for Fourier series (preserve original semantics)
        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 * math.pi  # [B, D]

        # Constant term
        ones = torch.ones(batch_size, input_dim, 1, dtype=x.dtype, device=x.device)

        if self.num_basis <= 1:
            return ones

        # Frequencies: shape [K] where K = num_basis - 1 (consistent with original)
        freqs = self.frequencies.to(x.device)
        if len(freqs) != self.num_basis - 1:
            freqs = torch.arange(1, self.num_basis, dtype=x.dtype, device=x.device)

        # Compute sin and cos for all dims and frequencies in one shot
        # xk: [B, D, K]
        xk = x_normalized.unsqueeze(-1) * freqs.view(1, 1, -1)
        sin_xk = torch.sin(xk)
        cos_xk = torch.cos(xk)

        # Interleave sin and cos per frequency: [sin1, cos1, sin2, cos2, ...]
        interleaved = torch.stack([sin_xk, cos_xk], dim=-1).reshape(batch_size, input_dim, -1)

        # Take first (num_basis - 1) terms to match original truncation behavior
        fourier_terms = interleaved[..., : self.num_basis - 1]

        # Concatenate constant term to form [B, D, num_basis]
        basis_tensor = torch.cat([ones, fourier_terms], dim=-1)
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
        # Vectorized CPU simulation for all qubits
        # Normalize input to [-1, 1]
        x_normalized = torch.tanh(x)  # [B, 1]

        # Prepare parameters
        theta = self.theta.to(x.device)
        phi = self.phi.to(x.device)

        # Expand to [B, 1, Q] then broadcast
        x_exp = x_normalized.unsqueeze(-1)  # [B, 1, 1]
        theta_exp = theta.view(1, 1, -1)
        phi_exp = phi.view(1, 1, -1)

        # Compute rotations in parallel: [B, 1, Q]
        rot_x = torch.cos(theta_exp * x_exp + phi_exp)
        rot_y = torch.sin(theta_exp * x_exp + phi_exp)

        # Concatenate along qubit-feature axis -> [B, 1, 2Q]
        quantum_state = torch.cat([rot_x, rot_y], dim=-1)

        # Measurement: mean over features axis -> [B, 1]
        measurement = torch.mean(quantum_state, dim=-1, keepdim=True)
        return measurement
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PQC basis functions
        Returns shape: [batch, input_dim, num_basis]
        """
        batch_size, input_dim = x.shape

        # Vectorize across all input dims: reshape to [B*D, 1]
        x_flat = x.reshape(-1, 1)
        quantum_output = self._quantum_circuit_simulation(x_flat)  # [B*D, 1]

        # Build basis terms vectorized
        ones = torch.ones(quantum_output.shape[0], 1, dtype=x.dtype, device=x.device)
        if self.num_basis <= 1:
            basis_flat = ones
        else:
            freqs = torch.arange(1, self.num_basis, dtype=x.dtype, device=x.device)
            angles = quantum_output @ (freqs.view(1, -1) * math.pi)  # [B*D, K]
            sin_terms = torch.sin(angles)  # [B*D, K]
            basis_flat = torch.cat([ones, sin_terms], dim=-1)  # [B*D, num_basis]

        # Reshape back to [B, D, num_basis]
        basis_tensor = basis_flat.reshape(batch_size, input_dim, self.num_basis)
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
        Execute quantum circuit on GPU using vectorized trig simulation
        """
        # Fallback to vectorized CPU sim if PennyLane not available
        if not self.pennylane_available:
            return self._quantum_circuit_simulation_cpu(x)

        # Use vectorized trig simulation on current device to avoid per-sample qnode
        x_normalized = torch.tanh(x)  # [B, 1]
        theta = self.theta.to(x.device)
        phi = self.phi.to(x.device)

        # Expand to [B, 1, Q]
        x_exp = x_normalized.unsqueeze(-1)
        theta_exp = theta.view(1, 1, -1)
        phi_exp = phi.view(1, 1, -1)

        rot_x = torch.cos(theta_exp * x_exp + phi_exp)
        rot_y = torch.sin(theta_exp * x_exp + phi_exp)
        rot_z = torch.cos((theta_exp * 0.5) * x_exp)  # simple extra feature

        quantum_state = torch.cat([rot_x, rot_y, rot_z], dim=-1)
        measurement = torch.mean(quantum_state, dim=-1, keepdim=True)
        return measurement
    
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
