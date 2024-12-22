import torch
from torch import nn
import torch.nn.functional as F


class ShallowPrivateCollapsedDictNetSlow(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, num_subjects=9):
        """
        A neural network model that creates a separate spatio-temporal convolutional layer 
        for each subject, then applies a common set of operations (activation, normalization, pooling, dropout, and 
        a fully connected layer) to produce outputs. This model expects that the input tensor 
        includes subject IDs encoded at the final time point of one channel.

        Parameters
        ----------
        n_chans : int
            Number of input channels (e.g., EEG channels).
        n_outputs : int
            Number of output classes or regression targets.
        n_times : int, optional
            Number of time samples per input. Default is 1001.
        dropout : float, optional
            Dropout probability. Default is 0.5.
        num_kernels : int, optional
            Number of output kernels (filters) for convolution layers. Default is 40.
        kernel_size : int, optional
            Size (length) of the temporal convolution kernel. Default is 25.
        pool_size : int, optional
            Size of the temporal pooling window. Default is 100.
        num_subjects : int, optional
            Number of different subjects (used to create a separate dictionary or layer per subject). Default is 9.
        """
        super(ShallowPrivateCollapsedDictNetSlow, self).__init__()
        
        self.num_subjects = num_subjects
        
        # Create a dictionary of subject-specific convolution layers.
        # Each subject_#: Convolves from n_chans -> num_kernels with a (1 x kernel_size) filter.
        # The input shape after unsqueezing will be [batch_size, n_chans, 1, n_times], so the kernel 
        # operates along the time dimension.
        self.spatio_temporal_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Conv2d(n_chans, num_kernels, (1, kernel_size))
            for i in range(num_subjects)
        })   

        # Average Pooling over time to reduce feature dimension.
        # pool_size is applied over the time dimension (the second spatial dimension = width).
        self.pool = nn.AvgPool2d((1, pool_size))
        
        # Batch normalization to normalize across the batch dimension for stable training.
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        
        # Dropout for regularization to reduce overfitting.
        self.dropout = nn.Dropout(dropout)
        
        # LazyLinear will infer input features automatically at runtime.
        # This fully connected layer maps from the flattened features to n_outputs.
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, n_chans, n_times].
            The last time point in the first channel contains the subject ID encoded as subject_id * 1,000,000.

        Returns
        -------
        torch.Tensor
            Output logits or predictions of shape [batch_size, n_outputs].
        """
           
        # Extract the subject IDs from the last time point of channel 0.
        # subject_ids will be something like subject_id * 1,000,000, so dividing by 1,000,000 recovers the subject_id.
        subject_ids = x[:, 0, -1] / 1000000
        
        # Remove the last time point (which contains the subject ID data) from the input
        # since it's not an actual data point for prediction.
        x = x[:, :, :-1]
        
        # Ensure that all samples in the batch belong to the same subject.
        unique_subject_ids = torch.unique(subject_ids)
        if unique_subject_ids.size(0) != 1:
            # If there's more than one subject ID, something is wrong since this model 
            # expects a batch from a single subject.
            print("Error: More than one subject ID detected in the batch")
            return None
        
        # Get the single subject ID for the batch.
        subject_id = subject_ids[0].long().item()
        
        # Add a dimension at index 2 to match the shape required by the subject-specific layer:
        # New shape is [batch_size, n_chans, 1, n_times]
        x = torch.unsqueeze(x, dim=2)
        
        # Pass through the subject-specific convolution layer chosen by the subject_id.
        x = self.spatio_temporal_layers[f'subject_{subject_id}'](x)
        
        # Apply ELU activation to introduce non-linearity.
        x = F.elu(x)
        
        # Apply batch_norm normalization.
        x = self.batch_norm(x)
        
        # Apply average pooling over the time dimension to reduce temporal resolution.
        x = self.pool(x)
        
        # Flatten the feature maps from [batch, num_kernels, 1, reduced_time] to [batch, num_kernels * reduced_time].
        x = x.view(x.size(0), -1)
        
        # Apply dropout for regularization.
        x = self.dropout(x)
        
        # Pass through the final fully connected layer to get the output logits.
        x = self.fc(x)
        
        return x
    

class ShallowPrivateSpatialDictNetSlow(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, num_subjects=9):
        """
        Similar to the previous model, but the approach differs slightly:
        First applies a common temporal convolution, then uses a dictionary of subject-specific 
        spatial filters before the rest of the pipeline (activation, batch norm, pooling, etc.).
        
        Parameters
        ----------
        n_chans : int
            Number of input channels.
        n_outputs : int
            Number of output classes or targets.
        n_times : int
            Number of time samples.
        dropout : float
            Dropout probability.
        num_kernels : int
            Number of output kernels (filters) for convolution layers.
        kernel_size : int
            Temporal kernel size for convolution.
        pool_size : int
            Temporal pooling size.
        num_subjects : int
            Number of subjects for subject-specific layers.
        """
        super(ShallowPrivateSpatialDictNetSlow, self).__init__()
        self.num_subjects = num_subjects
        
        # Temporal convolution first: from 1 "input channel dimension" to num_kernels over the time dimension.
        # Input will be reshaped to [batch, 1, n_chans, n_times].
        self.temporal = nn.Conv2d(1, num_kernels, (1, kernel_size))
        
        # Dictionary of subject-specific spatial layers:
        # Each subject layer: from num_kernels -> num_kernels using a spatial kernel that covers all n_chans,
        # but not extending over time (kernel size: (n_chans, 1)).
        self.spatial_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Conv2d(num_kernels, num_kernels, (n_chans, 1))
            for i in range(num_subjects)
        })        
        
        # Average pooling over the time dimension.
        self.pool = nn.AvgPool2d((1, pool_size))
        
        # Batch normalization to normalize across the batch dimension for stable training.
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        
        # Dropout for regularization.
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer that will adapt to the flattened input size at runtime.
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        """
        Forward pass of the ShallowPrivateSpatialDictNetSlow.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch_size, n_chans, n_times], with the last time point holding subject IDs.

        Returns
        -------
        torch.Tensor
            Output predictions of shape [batch_size, n_outputs].
        """
        
        # Extract subject ID from the last time point of the first channel.
        subject_ids = x[:, 0, -1] / 1000000
        # Remove the last time point.
        x = x[:, :, :-1]
        
        # Check that the batch has only one subject.
        unique_subject_ids = torch.unique(subject_ids)
        if unique_subject_ids.size(0) != 1:
            print("Error: More than one subject ID detected in the batch")
            return None
        
        # Reshape input to [batch_size, 1, n_chans, n_times] for the temporal convolution.
        x = torch.unsqueeze(x, dim=1)
        
        # Apply the temporal convolution (common to all subjects).
        x = self.temporal(x)
        
        # Determine the subject ID and use the corresponding spatial layer.
        subject_id = subject_ids[0].long().item()
        x = self.spatial_layers[f'subject_{subject_id}'](x)

        # Apply ELU activation function.
        x = F.elu(x)
        
        # Normalize across the batch dimension.
        x = self.batch_norm(x)
        
        # Pool over the time dimension to reduce dimensionality.
        x = self.pool(x)
        
        # Flatten the features.
        x = x.view(x.size(0), -1)
        
        # Apply dropout.
        x = self.dropout(x)
        
        # Fully connected layer to produce final outputs.
        x = self.fc(x)
        
        return x


class ShallowPrivateTemporalDictNetSlow(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, kernel_size=25, pool_size=100, num_subjects=9):
        """
        This model applies subject-specific temporal filters first, then a common spatial filter, 
        followed by batch normalization, pooling, dropout, and a fully connected layer.
        
        Parameters
        ----------
        n_chans : int
            Number of input channels.
        n_outputs : int
            Number of output classes or regression targets.
        n_times : int
            Number of time samples.
        dropout : float
            Dropout probability.
        num_kernels : int
            Number of output kernels (filters) for convolution layers.
        kernel_size : int
            Size of the temporal convolution kernel.
        pool_size : int
            Temporal pooling size.
        num_subjects : int
            Number of subjects for subject-specific layers.
        """
        super(ShallowPrivateTemporalDictNetSlow, self).__init__()
        self.num_subjects = num_subjects
        
        # Create a dictionary of subject-specific temporal layers.
        # Each layer: [batch, 1, n_chans, n_times] -> [batch, num_kernels, n_chans, reduced_time]
        self.temporal_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Conv2d(1, num_kernels, (1, kernel_size))
            for i in range(num_subjects)
        })
        
        # A common spatial convolution that filters across all channels: (n_chans, 1) kernel.
        # After temporal convolution, the shape is [batch, num_kernels, n_chans, something].
        self.spatial = nn.Conv2d(num_kernels, num_kernels, (n_chans, 1))
        
        # Average pooling over the time dimension.
        self.pool = nn.AvgPool2d((1, pool_size))
        
        # Batch normalization across the batch dimension.
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        
        # Dropout for regularization.
        self.dropout = nn.Dropout(dropout)
        
        # LazyLinear for final output layer.
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, n_chans, n_times], with last time point holding subject IDs.

        Returns
        -------
        torch.Tensor
            Predictions of shape [batch_size, n_outputs].
        """
        
        # Extract the subject ID from the last time point.
        subject_ids = x[:, 0, -1] / 1000000
        # Remove the last time point.
        x = x[:, :, :-1]
        
        # Determine the subject ID and ensure batch uniformity.
        subject_id = subject_ids[0].long().item()
        unique_subject_ids = torch.unique(subject_ids)
        if unique_subject_ids.size(0) != 1:
            print("Error: More than one subject ID detected in the batch")
            return None
        
        # Add a dimension for the "input channel" to match Conv2D input: [batch, 1, n_chans, n_times].
        x = torch.unsqueeze(x, dim=1)
        
        # Apply the subject-specific temporal convolution.
        x = self.temporal_layers[f'subject_{subject_id}'](x)
        
        # Apply the common spatial convolution across channels.
        x = self.spatial(x)
        
        # ELU activation for non-linearity.
        x = F.elu(x)
        
        x = self.batch_norm(x)
        
        x = self.pool(x)
        
        # Flatten the features for the fully connected layer.
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        
        # Final fully connected layer to produce outputs.
        x = self.fc(x)
        
        return x
    
    
class SubjectDicionaryFCNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                kernel_size=25, pool_size=100, num_subjects=9):
        """
        A model that uses a common spatio-temporal convolution, pooling, batch normalization,
        and dropout, but then routes the flattened features to a subject-specific fully 
        connected layer. This allows having distinct output transformations per subject.
        
        Parameters
        ----------
        n_chans : int
            Number of channels in the input.
        n_outputs : int
            Number of outputs (classes or targets).
        n_times : int
            Number of time points.
        dropout : float
            Dropout probability.
        num_kernels : int
            Number of output kernels (filters) from the convolution layer.
        kernel_size : int
            Temporal kernel size for the convolution.
        pool_size : int
            Temporal pooling window size.
        num_subjects : int
            Number of subjects, each having a separate fully connected layer.
        """
        super(SubjectDicionaryFCNet, self).__init__()
        self.num_subjects = num_subjects
        
        # Spatio-temporal convolution: 
        # [batch, n_chans, n_times] -> [batch, num_kernels, 1, reduced_times]
        # after we reshape the input to add the 'channel' dimension for the convolution over time.
        self.spatio_temporal = nn.Conv2d(
            n_chans, num_kernels, (1, kernel_size))
        
        # Average pool over time to reduce dimension.
        self.pool = nn.AvgPool2d((1, pool_size))
        
        # Batch normalization.
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        
        # Dropout for regularization.
        self.dropout = nn.Dropout(dropout)
        
        # Subject-specific fully connected layers:
        # Each subject gets a separate linear layer. We must compute the input size for the FC layer:
        # After conv and pool, time dimension is reduced. Specifically:
        # (n_times - kernel_size + 1) // pool_size gives the reduced time dimension.
        fc_input_dim = num_kernels * ((n_times - kernel_size + 1) // pool_size)
        
        self.fc_layers = nn.ModuleDict({
            f'subject_{i+1}': nn.Linear(fc_input_dim, n_outputs)
            for i in range(num_subjects)           
        })

    def forward(self, x):
        """
        Forward pass of the SubjectDicionaryFCNet.

        Parameters
        ----------
        x : torch.Tensor
            Input [batch, n_chans, n_times], last time point holds subject IDs.

        Returns
        -------
        torch.Tensor
            Output predictions [batch, n_outputs].
        """
        
        # Extract subject IDs from the last time point of the first channel.
        subject_ids = x[:, 0, -1] / 1000000
        
        # Remove the last time point (not used for actual data).
        x = x[:, :, :-1]

        # Check for uniform subject ID in the batch.
        unique_subject_ids = torch.unique(subject_ids)
        subject_id = subject_ids[0].long().item()
        
        if unique_subject_ids.size(0) != 1:
            print("Error: More than one subject ID detected in the batch")
            return None
        
        # Reshape input to [batch, n_chans, 1, n_times] for the convolution.
        x = torch.unsqueeze(x, dim=2) 
        
        # Apply spatio-temporal convolution.
        x = self.spatio_temporal(x)
        
        x = F.elu(x)
        
        x = self.batch_norm(x)
        
        x = self.pool(x)
        
        # Flatten to a 2D tensor [batch, features].
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        
        # Use the subject-specific fully connected layer corresponding to the subject ID.
        x = self.fc_layers[f'subject_{subject_id}'](x)

        return x
