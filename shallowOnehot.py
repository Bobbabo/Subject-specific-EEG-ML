import torch
from torch import nn
import torch.nn.functional as F


class SubjectOneHotConvNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                 kernel_size=25, pool_size=100, num_subjects=9):

        super(SubjectOneHotConvNet, self).__init__()
        self.num_subjects = num_subjects
        self.num_kernels = num_kernels

        self.spatio_temporal = nn.Conv2d(n_chans, num_kernels * num_subjects, (1, kernel_size))

        self.pool = nn.AvgPool2d((1, pool_size))

        self.batch_norm = nn.BatchNorm2d(num_kernels)

        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.LazyLinear(n_outputs)

    def forward(self, x):

        # Extract subject IDs and convert to zero-based
        subject_ids = (x[:, 0, -1] // 1000000).long()-1
        x = x[:, :, :-1]

        # One-hot encode subject IDs: [batch, num_subjects]
        subject_one_hot = F.one_hot(subject_ids, num_classes=self.num_subjects).float()
        subject_one_hot = subject_one_hot.view(x.size(0), self.num_subjects, 1, 1, 1)

        # Reshape input for convolution
        x = x.unsqueeze(2)  # [batch, n_chans, 1, n_times]
        x = self.spatio_temporal(x)  # [batch, num_kernels*num_subjects, 1, new_time_len]

        # Reshape to separate the subject dimension
        batch_size, total_kernels, _, time_len = x.size()
        x = x.view(batch_size, self.num_subjects, self.num_kernels, 1, time_len)

        # Select subject-specific kernels
        x = x * subject_one_hot
        x = x.sum(dim=1)  # [batch, num_kernels, 1, time_len]

        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SubjectOneHotNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                 kernel_size=25, pool_size=100, num_subjects=9):
  
        super(SubjectOneHotNet, self).__init__()
        self.n_outputs = n_outputs
        self.num_subjects = num_subjects

        # Convolution layer (common to all subjects)
        self.spatio_temporal = nn.Conv2d(n_chans, num_kernels, (1, kernel_size))

        # Pooling, normalization, dropout
        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)

        # Compute the length after conv and pooling
        conv_output_length = (n_times - kernel_size + 1) // pool_size
        fc_input_features = num_kernels * conv_output_length

        # Shared FC layer that outputs n_outputs for every subject: total n_outputs * num_subjects
        self.fc_shared = nn.Linear(fc_input_features, n_outputs * num_subjects)

    def forward(self, x):
  
  
        # Extract subject IDs
        subject_ids = (x[:, 0, -1] // 1000000).long()-1
        x = x[:, :, :-1]

        # One-hot encode subject IDs
        subject_one_hot = F.one_hot(subject_ids, num_classes=self.num_subjects).float()

        # Convolution step
        x = x.unsqueeze(2)   # [batch, n_chans, 1, n_times]
        x = self.spatio_temporal(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # Shared FC: outputs [batch, n_outputs * num_subjects]
        fc_output = self.fc_shared(x)

        # Reshape to [batch, num_subjects, n_outputs]
        fc_output = fc_output.view(-1, self.num_subjects, self.n_outputs)

        # Select the correct subject's outputs via one-hot encoding
        # out = (subject_one_hot * fc_output) summed over subjects
        # Using einsum: 'bi,bio->bo' for clarity
        out = torch.einsum('bi,bio->bo', subject_one_hot, fc_output)

        return out


class SubjectAdvIndexFCNet(nn.Module):
    def __init__(self, n_chans, n_outputs, n_times=1001, dropout=0.5, num_kernels=40, 
                 kernel_size=25, pool_size=100, num_subjects=9):

        super(SubjectAdvIndexFCNet, self).__init__()
        self.n_outputs = n_outputs
        self.num_subjects = num_subjects

        self.spatio_temporal = nn.Conv2d(n_chans, num_kernels, (1, kernel_size))

        self.pool = nn.AvgPool2d((1, pool_size))
        self.batch_norm = nn.BatchNorm2d(num_kernels)
        self.dropout = nn.Dropout(dropout)

        # Compute output size for FC
        conv_output_length = (n_times - kernel_size + 1) // pool_size
        fc_input_features = num_kernels * conv_output_length
        fc_output_features = n_outputs * num_subjects

        self.fc_shared = nn.Linear(fc_input_features, fc_output_features)

    def forward(self, x):

        subject_ids = (x[:, 0, -1] // 1000000).long()-1
        x = x[:, :, :-1]

        # Convolution and pooling
        x = x.unsqueeze(2)
        x = self.spatio_temporal(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # Shared FC
        fc_output = self.fc_shared(x)  # [batch, n_outputs * num_subjects]
        fc_output = fc_output.view(-1, self.num_subjects, self.n_outputs)

        # Use advanced indexing to select the correct subject's output
        batch_indices = torch.arange(fc_output.size(0), device=x.device)
        out = fc_output[batch_indices, subject_ids, :]  # [batch, n_outputs]

        return out
