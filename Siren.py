import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset


def paper_init_(weight, is_first=False, omega=1):
    """Initialize the weights of the Linear Layer

    Args:
        weight (torch.Tensor ): 2d representation of pixel

        is_first (bool, optional): Used to check if its the
         first layer of the network. Defaults to False.

        omega (float, optional): Used to control scaling. Defaults to 1.
    """

    in_features = weight.shape[1]
    with torch.no_grad():
        if is_first:
            bound = 1 / in_features
        else:
            bound = np.sqrt(6 / in_features) / omega
        weight.uniform_(-bound, bound)


class SineLayer(nn.Module):
    """Linear layer foilowed by a sine Layer

    Args:
        in_features (int): input features
        out_features (int): output features

        bias (bool, optional): To choose if bias is included.
        Defaults to True.


        is_first bool, optional): Tells if is the first. Defaults to True.

        omega (float, optional): Hyperparameter. Defaults to 30.

        custom_init_func (None or callbacks, optional):
            If None we are gonna use paper_init_ otherwise
            we are gonna use custom function on the weight matrix
        . Defaults to None.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=True,
        omega=30,
        custom_init_func=None,
    ):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if custom_init_func is None:
            paper_init_(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_func(self.linear.weight)

    def forward(self, x):
        """This is to propagate the input through the network

        Args:
            x (torch.tensor): 2d matrix representation
            Tensor of shape (n_Samples,in_features)

        Returns:
            torch.Tensor: Tensor of shape(n_Samples,out_features)
        """
        return torch.sin(self.omega * self.linear(x))


class ImageSiren(nn.Module):
    def __init__(
        self,
        hidden_features,
        hidden_layer=1,
        first_omega=30,
        hidden_omega=30,
        custom_init_func=None,
    ):
        """Network composed of Sinelayers

        Args:
            hidden_layer (int): Number of hidden layers

            hidden_features (int): NUmber of hidden features

            first_omega (float): Hyperparameter

            hidden_omega (float): Hyperparameter

            custom_init_func (None or callbacks, optional):
                If None we are gonna use paper_init_ otherwise
                we are gonna use custom function on the weight matrix
            . Defaults to None."""

        super().__init__()
        in_features = 2
        out_features = 1

        net = []
        net.append(
            SineLayer(
                in_features=in_features,
                out_features=hidden_features,
                is_first=True,
                custom_init_func=custom_init_func,
                omega=first_omega,
            ))

        for _ in range(hidden_layer):
            net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    custom_init_func=custom_init_func,
                    omega=hidden_omega,
                ))
        final_layer = nn.Linear(hidden_features, out_features)

        if custom_init_func is None:
            paper_init_(final_layer.weight, is_first=False, omega=hidden_omega)
        else:
            custom_init_func(final_layer.weight)
        net.append(final_layer)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """This is the forward pass of the networkfor

        Args:
            x (torch.Tensor): tensor shape (n_samples,2)

        Returns:
            torch.Tensor: Tensor shape (n_smaples,1)
        """
        return self.net(x)


def generate_coordinates(n):
    """Generate regular grid of 2D coorindates

    Args:
        n (int): Number of points per dimension
    Returns:
        coord_abs: np.ndarray
        Array of row and column coordinates

    """
    """
    Generates a Rectangular Grid of n rows and n column (n x n)

    for example:
        if n = 5 then
        np.mesh grid will result in
            [
                array (
                        [0,0,0,0,0],
                        [1,1,1,1,1],
                        [2,2,2,2,2],
                        [3,3,3,3,3],
                        [4,4,4,4,4],  )

             , array  ( [0,1,2,3,4],
                        [0,1,2,3,4],
                        [0,1,2,3,4],
                        [0,1,2,3,4],
                        [0,1,2,3,4], )
                                         ]
    Then we flatten the rows and cols using the ravel func

    This gives us a 1D array

        Flattened Cols
        np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])

        length of Flattened Rows 25

        Flattened Cols
        np.array([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4])

        length of Flattened Cols 25

    np.stack will just take one element from both flattened rows and cols

    sequentially and generate an array of [x,y] coords

    """
    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")
    coord_abs = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return coord_abs


class PixelDataset(Dataset):
    """Dataste yielding coordinates,intenstiies and ( higher derivates )

    Args:
        img:np.ndarray

    Attributes:
        size: int
            height and width of the square image

        coords_abs: np.ndarray
            Array of shape(n ** 2,2) representating all coordinates of img

        grad: np.ndarray
            Array of shape(size,size,2) representing the approximate gradient
            int he two direction

        grad_norm: np.ndarray
            Array of shape (size,size) representing the approximate gradient
            norm of img

        laplace: np.ndarray
            Array of shape(size,size) representing the approximate
            laplace operator
    """
    def __init__(self, img):
        if not (img.ndim == 2 and img.shape[0] == img.shape[1]):
            raise ValueError("Only 2D square images are supported")
        self.img = img
        self.size = img.shape[0]
        self.coords_abs = generate_coordinates(self.size)
        self.grad = np.stack([sobel(img, axis=0), sobel(img, axis=1)], axis=-1)
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(img)

    def __len__(self):
        """Determines the nubmer of pixels"""
        return self.size**2

    def __getitem__(self, idx):
        """Get the relevant pixel data from the coordinates

        Args:
            idx (int): index of the pixel
        """
        coords_abs = self.coords_abs[idx]
        r, c = coords_abs
        coords = 2 * ((coords_abs / self.size) - 0.5)
        return {
            "coords": coords,
            "coords_abs": coords_abs,
            "intensity": self.img[r, c],
            "grad_norm": self.grad_norm[r, c],
            "grad": self.grad[r, c],
            "laplace": self.laplace[r, c],
        }


class ImplicitAudioWrapper(Dataset):
    def __init__(self, dataset):
        """This is the wrapper for Audio Dataset.

        Args:
            dataset (Dataset): The audio dataset for PreProcessing
        """

        self.dataset = dataset
        """
            Creating a grid from -100 to 100 with the size of the
            audio file the audio points
        """
        self.grid = np.linspace(start=-100, stop=100, num=dataset.file_length)

        self.grid = self.grid.astype(np.float32)

        self.grid = torch.Tensor(self.grid).view(-1, 1)

    def num_samples_grid(self):
        return self.grid.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data, rate = self.dataset[idx]
        scale = np.max(np.abs(data))
        data = data / scale
        data = torch.Tensor(data).view(-1, 1)
        return {
            "idx": idx,
            "coords": self.grid,
        }, {
            "func": data,
            "rate": rate,
            "scale": scale
        }


class AudioDataset(Dataset):
    """This is the dataset for Audio Processing

    Args:
        Dataset : Torch boilerplate to make a torch dataset

    Returns:
        np.ndarray: np array of shape
    """
    def __init__(self, path_to_audio):
        """This is the constructor for AudioDataset

        Args:
            path_to_audio (str): Path to audio file format wav
        Return:
            Type Dataset: torch.utils.data.Dataset
        """
        super().__init__()
        self.rate, self.data = wavfile.read(path_to_audio)
        if len(self.data.shape) > 1 and self.data.shape[1] == 2:
            self.data = np.mean(self.data, axis=1)

        self.data = self.data.astype(np.float32)
        self.file_length = len(self.data)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data, self.rate


class GradientUtils:
    @staticmethod
    def gradient(target, coords):
        """Compute the gradient wrt to the image

        Args:
            target (torch.Tensor): Tensor of shape (n_coords,?)
            representing the gradient wrt x and y

            coords (torch.Tensor): Tensor of shape (n_coords,2)
            representing the coordinates

        Returns:

        gradient : torch.tensor
            2D tensor of shape (n_coords,2) representing the gradient
        """
        """
            Point:
                create_graph is set to True so we can continously calculate
                higher order derivatives like 1st and 2nd order derivative
                , By default is set to False so we cant calculate the
                higher order derivatives it throws an error

        """

        return torch.autograd.grad(target,
                                   coords,
                                   grad_outputs=torch.ones_like(target),
                                   create_graph=True)[0]

    @staticmethod
    def divergence(grad, coords):
        """Compute divergence.
        Parameters
        ----------
        grad : torch.Tensor
            2D tensor of shape `(n_coords, 2)` representing the gradient wrt
            x and y.
        coords : torch.Tensor
            2D tensor of shape `(n_coords, 2)` representing the coordinates.
        Returns
        -------
        div : torch.Tensor
            2D tensor of shape `(n_coords, 1)` representing the divergence.
        Notes
        -----
        In a 2D case this will give us f_{xx} + f_{yy}.
        """
        div = 0.0

        for i in range(coords.shape[1]):
            div += torch.autograd.grad(
                grad[..., i],
                coords,
                torch.ones_like(grad[..., i]),
                create_graph=True,
            )[0][..., i:i + 1]
        return div

    @staticmethod
    def laplace(target, coords):
        """Compute laplace operator.
        Parameters
        ----------
        target : torch.Tensor
            2D tesnor of shape `(n_coords, 1)` representing the targets.
        coords : torch.Tensor
            2D tensor of shape `(n_coords, 2)` representing the coordinates.
        Returns
        -------
        torch.Tensor
            2D tensor of shape `(n_coords, 1)` representing the laplace.
        """
        grad = GradientUtils.gradient(target, coords)
        return GradientUtils.divergence(grad, coords)
