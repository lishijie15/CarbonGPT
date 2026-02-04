import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.fft
from transformers.configuration_utils import PretrainedConfig
from carbongpt.model.st_layers.ModernTCN import extendmodel
from sklearn.preprocessing import StandardScaler

def gaussian_weights(length, center, sigma):
    """Generate Gaussian weights for a given length centered at 'center' with standard deviation 'sigma'."""
    x = torch.arange(length, dtype=torch.float32)
    center = center.item()
    weights = torch.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    weights /= weights.sum()  # Normalize
    return weights

def block_index_to_coordinates(row_index):
    if not 0 <= row_index < 1177:
        raise ValueError("The index of a single block must be between 0 and 1177.")
    block_size = 1

    top_left_y1 = row_index * block_size
    top_left_y2 = row_index * block_size

    return (top_left_y1, top_left_y2)

def mixup_region(graph_array, x, y, block_size):

    h, w, _ = graph_array.shape
    x_start = max(x - block_size, 0)
    x_end = min(x + 2*block_size, h)
    y_start = max(y - block_size, 0)
    y_end = min(y + 2*block_size, w)

    if y_end > graph_array.shape[1] - block_size:
        block_size = 1

    region = graph_array[:, y:y+block_size, :]
    surrounding_regions = []

    for j in range(int(y_start / block_size), int(y_end / block_size)):
        surrounding_region = graph_array[:, j*block_size:(j+1)*block_size, :]
        surrounding_regions.append(surrounding_region)

    mixed_region = 0
    for j in range(len(surrounding_regions)):
        mixed_region += surrounding_regions[j] / len(surrounding_regions)

    new_graph_array = graph_array.clone()
    new_graph_array[:, y:y + block_size, :] = mixed_region.to(torch.uint8)
    return new_graph_array

def mixup_region_ST(graph_array, x, y, block_size, sigma):
    h, w, _ = graph_array.shape
    x_start = 0
    x_end = x
    y_start = max(y - block_size, 0)
    y_end = min(y + 2*block_size, w)
    if y_end > graph_array.shape[1] - block_size:
        block_size = 1

    time_region = graph_array[:, y, :]
    target_region = time_region[x_start:x_end, :]

    weights = gaussian_weights(x_end, x - 1, sigma).to("cuda")

    mixed_value = torch.sum(weights.view(-1, 1) * target_region, dim=0)
    time_graph_array = graph_array.clone()
    time_graph_array[x, y, :] = mixed_value

    surrounding_regions = []

    for j in range(int(y_start / block_size), int(y_end / block_size)):

        surrounding_region = time_graph_array[:, j*block_size:(j+1)*block_size, :]
        surrounding_regions.append(surrounding_region)

    mixed_region = 0
    for j in range(len(surrounding_regions)):

        mixed_region += surrounding_regions[j] / len(surrounding_regions)

    new_graph_array = time_graph_array.clone()
    new_graph_array[:, y:y + block_size, :] = mixed_region.to(torch.uint8)

    return new_graph_array

class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class Spatial_Attention(nn.Module):
    def __init__(self, out_features):
        super(Spatial_Attention, self).__init__()
        self.out_features = out_features

        self.xff = nn.Linear(self.out_features, 3 * self.out_features)

        self.ff = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.GELU(),
            nn.Linear(self.out_features, self.out_features),
        )

        self.ln = nn.LayerNorm(self.out_features)

    def forward(self, input, adapt_G):
        x_ = self.xff(input)

        x_ = torch.stack(torch.split(x_, self.out_features, -1), 0)
        query = x_[0]
        key = x_[1]
        value = x_[2]

        e = torch.matmul(query, key.transpose(-1, -2)) / (self.out_features ** 0.5)

        e = torch.matmul(e, adapt_G)
        attn = torch.softmax(e, -1)

        self.attention_maps = attn.detach()
        processed_images = []
        for batch_attention_maps, graph_array in zip(self.attention_maps, value):
            graph_array_front = graph_array[:, :, :78]
            graph_array_back = graph_array[:, :, 78:]
            min_number = 2
            column_sums = torch.sum(batch_attention_maps, dim=1)
            _, min_weight_rows = torch.topk(column_sums, min_number, dim=1, largest=False)

            for column_index_all, i in zip(min_weight_rows, range(len(min_weight_rows)//2)):
                for column_index in column_index_all:
                    x, y = i, column_index
                    x = torch.tensor(x)
                    if x == 0:
                        graph_array = mixup_region(graph_array_front, x, y, 2)
                    else:
                        graph_array = mixup_region_ST(graph_array_front, x, y, 6, sigma=10.0)  

            processed_image = torch.cat((graph_array, graph_array_back), dim=2)

            processed_images.append(processed_image)

        new_value = torch.stack(processed_images, dim=0)

        value_i = torch.matmul(attn, new_value)

        value = self.ff(value_i) + input
        return self.ln(value)

class Spatial_Attention2(nn.Module):
    def __init__(self, out_features):
        super(Spatial_Attention2, self).__init__()
        self.out_features = out_features

        self.xff = nn.Linear(self.out_features, 3 * self.out_features)

        self.ln = nn.LayerNorm(self.out_features)

    def forward(self, input, adapt_G):
        x_ = self.xff(input)
        x_ = torch.stack(torch.split(x_, self.out_features, -1), 0)
        query = x_[0]
        key = x_[1]
        value = x_[2]

        e = torch.matmul(query, key.transpose(-1, -2)) / (self.out_features ** 0.5)

        e = torch.matmul(e, adapt_G)
        attn = torch.softmax(e, -1)
        new_value = torch.matmul(attn, value)
        return self.ln(new_value)

class ST_Enc(nn.Module):
    def __init__(self, args):
        super(ST_Enc, self).__init__()
        self.config = PretrainedConfig()
        self.num_nodes = args.num_nodes
        self.feature_dim = args.input_dim

        self.input_window = args.input_window
        self.output_window = args.output_window
        self.output_dim = args.output_dim

        self.dropout = 0.
        self.dilation_exponential = 1

        self.conv_channels = args.conv_channels
        self.residual_channels = args.residual_channels
        self.skip_channels = args.skip_channels
        self.end_channels = args.end_channels

        self.layers = 3
        self.propalpha = 0.05

        self.plus_window = self.input_window + self.output_window
        self.plus_proj = nn.Linear(self.input_window, self.plus_window)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.d_model = args.d_model
        self.extend = extendmodel(args)
        self.mem_num = args.mem_num
        self.mem_dim = args.mem_dim
        self.memory = self.construct_memory()
        self.sattn = Spatial_Attention(self.residual_channels)

        self.sattn2 = Spatial_Attention2(self.residual_channels)

        self.proj_test = nn.Linear(110, self.output_dim)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                       / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
                                / (self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers+1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
                                    / (self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(DilatedInception(self.residual_channels,
                                                          self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(self.residual_channels,
                                                        self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels, kernel_size=(1, 1)))
                if self.plus_window > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.plus_window-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.residual_channels,
                                    out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window, kernel_size=(1, 1), bias=True)
        if self.plus_window > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.plus_window), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels,
                                   kernel_size=(1, self.plus_window-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
                                   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels,
                                   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.d_model, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)
        value = torch.matmul(att_score, self.memory['Memory'])
        _, ind = torch.topk(att_score, k=2, dim=-1)
        return value

    def create_scalers(self, dim):
        scalers = []
        for _ in range(dim):
            scaler = StandardScaler()
            scalers.append(scaler)
        return scalers

    def forward(self, source):
        batch_size, time_steps, spatial_dim, feature_dim = source.shape
        scalers = self.create_scalers(self.feature_dim)

        flattened_feature_0 = source[:, :, :, 0:1].reshape(-1, 1).cpu().numpy()
        flattened_feature_1 = source[:, :, :, 1:2].reshape(-1, 1).cpu().numpy()
        flattened_feature_2 = source[:, :, :, 2:3].reshape(-1, 1).cpu().numpy()

        scalers[0].fit(flattened_feature_0)
        scalers[1].fit(flattened_feature_1)
        scalers[2].fit(flattened_feature_2)

        data = torch.empty_like(source)

        data[:, :, :, 0:1] = torch.tensor(scalers[0].transform(flattened_feature_0).reshape(batch_size, time_steps, spatial_dim, 1), device=source.device)
        data[:, :, :, 1:2] = torch.tensor(scalers[1].transform(flattened_feature_1).reshape(batch_size, time_steps, spatial_dim, 1), device=source.device)
        data[:, :, :, 2:3] = torch.tensor(scalers[2].transform(flattened_feature_2).reshape(batch_size, time_steps, spatial_dim, 1), device=source.device)

        inputs = data

        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['We2'].T)
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['We1'].T)
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)  # E,ET
        inputs = inputs.transpose(1, 3)
        assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'
        inputs = self.plus_proj(inputs)

        if self.plus_window < self.receptive_field:
            inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))

        x = self.extend(inputs)

        x = self.sattn(x.transpose(1, 3), g1).transpose(1, 3)

        skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filters = self.filter_convs[i](x)
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filters * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x_emb = x.clone()
        h_att = self.query_memory(x.permute(0, 3, 2, 1))
        x_ = torch.cat([x, h_att.permute(0, 3, 2, 1)], dim=1)

        he = self.extend(x_.permute(0, 3, 2, 1))
        h_de = self.sattn2(he.permute(0, 3, 2, 1), g1)
        h_de = h_de.permute(0, 3, 2, 1)
        for i in range(self.layers):
            residual = h_de
            filters = self.filter_convs[i](h_de)
            filters = torch.tanh(filters)
            gate = self.gate_convs[i](h_de)
            gate = torch.sigmoid(gate)
            h_de = filters * gate
            h_de = F.dropout(h_de, self.dropout, training=self.training)

            h_de = self.residual_convs[i](h_de)
            h_de = h_de + residual[:, :, :, -h_de.size(3):]

        output = self.proj_test(h_de)
        x = F.relu(self.end_conv_1(output))

        x = self.end_conv_2(x)

        return x, x_emb

