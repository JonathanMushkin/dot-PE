"""
Device management for GPU acceleration using PyTorch.

This module provides device management utilities for running computations
on either CPU or GPU (CUDA), with automatic fallback to CPU when GPU is not available.
"""

import torch
import numpy as np
from typing import Union, Optional, Any


class DeviceManager:
    """
    Manages device selection and tensor operations for GPU acceleration.

    Provides automatic fallback to CPU when GPU is not available,
    and handles tensor conversions between numpy and torch.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize device manager.

        Parameters:
        -----------
        device : str
            Device to use. Options: 'auto', 'cpu', 'cuda'
            'auto' will select CUDA if available, otherwise CPU
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = "cuda"
            else:
                self.device = torch.device("cpu")
                self.device_name = "cpu"
        else:
            self.device = torch.device(device)
            self.device_name = device

        print(f"DeviceManager initialized with device: {self.device_name}")

    def to_tensor(
        self,
        input_data: Union[np.ndarray, torch.Tensor, Any],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Convert input_data to torch tensor on the managed device.

        Parameters:
        -----------
        input_data : Union[np.ndarray, torch.Tensor, Any]
            Data to convert to tensor
        dtype : Optional[torch.dtype]
            Desired data type. If None, preserves original type

        Returns:
        --------
        torch.Tensor
            Tensor on the managed device
        """
        if isinstance(input_data, torch.Tensor):
            if input_data.device != self.device:
                return input_data.to(self.device, dtype=dtype)
            elif dtype is not None and input_data.dtype != dtype:
                return input_data.to(dtype=dtype)
            return input_data
        elif isinstance(input_data, np.ndarray):
            if dtype is None:
                # Preserve numpy dtype when converting to torch
                if input_data.dtype == np.complex128:
                    torch_dtype = torch.complex128
                elif input_data.dtype == np.complex64:
                    torch_dtype = torch.complex64
                elif input_data.dtype == np.float64:
                    torch_dtype = torch.float64
                elif input_data.dtype == np.float32:
                    torch_dtype = torch.float32
                elif input_data.dtype == np.int64:
                    torch_dtype = torch.int64
                elif input_data.dtype == np.int32:
                    torch_dtype = torch.int32
                else:
                    torch_dtype = torch.float32  # default
            else:
                torch_dtype = dtype
            return torch.tensor(input_data, dtype=torch_dtype, device=self.device)
        elif isinstance(input_data, list) and len(input_data) > 0 and isinstance(input_data[0], np.ndarray):
            # Convert list of numpy arrays to single numpy array first to avoid warning
            input_array = np.array(input_data)
            return self.to_tensor(input_array, dtype=dtype)
        else:
            # Handle scalars and other types
            return torch.tensor(input_data, dtype=dtype, device=self.device)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.

        Parameters:
        -----------
        tensor : torch.Tensor
            Tensor to convert

        Returns:
        --------
        np.ndarray
            Numpy array
        """
        return tensor.cpu().numpy()

    def is_gpu_available(self) -> bool:
        """
        Check if GPU is available and being used.

        Returns:
        --------
        bool
            True if using GPU, False otherwise
        """
        return self.device_name == "cuda"

    def get_device_info(self) -> dict:
        """
        Get information about the current device.

        Returns:
        --------
        dict
            Device information
        """
        info = {
            "device_name": self.device_name,
            "device": str(self.device),
            "is_gpu": self.is_gpu_available(),
        }

        if self.device_name == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name()
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory

        return info


def create_device_manager(device: str = "auto") -> DeviceManager:
    """
    Factory function to create a device manager.

    Parameters:
    -----------
    device : str
        Device specification

    Returns:
    --------
    DeviceManager
        Configured device manager
    """
    return DeviceManager(device)


# Global device manager instance
_global_device_manager = None


def get_device_manager() -> DeviceManager:
    """
    Get the global device manager instance.

    Returns:
    --------
    DeviceManager
        Global device manager
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager("auto")
    return _global_device_manager


def set_device_manager(device_manager: DeviceManager) -> None:
    """
    Set the global device manager instance.

    Parameters:
    -----------
    device_manager : DeviceManager
        Device manager to set as global
    """
    global _global_device_manager
    _global_device_manager = device_manager


def initialize_device(device: str = "auto") -> DeviceManager:
    """
    Initialize the global device manager with the specified device.

    Parameters:
    -----------
    device : str
        Device to use. Options: "auto", "cpu", "cuda"

    Returns:
    --------
    DeviceManager
        The initialized device manager
    """
    device_manager = DeviceManager(device)
    set_device_manager(device_manager)
    return device_manager
