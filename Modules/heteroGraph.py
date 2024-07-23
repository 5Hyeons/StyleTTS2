import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear

class HGT(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super(HGT, self).__init__()
        self.gru = nn.GRU(768, hidden_channels, batch_first=True)
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            if(node_type == "text"):
                # text
                self.lin_dict[node_type] = Linear(hidden_channels, hidden_channels)
            else:
                # acoustic/prosody
                self.lin_dict[node_type] = Linear(128, hidden_channels)


        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, out_text_len):
        for node_type, x in x_dict.items():
            if node_type == 'text':
                x, _ = self.gru(x)
                # 아래 둘 중 뭐 쓸지 고민중,,
                x = x[:, -1, :]
                # x = x.mean(1)
            x = self.lin_dict[node_type](x).relu()
            x_dict[node_type] = x

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        out_text = x_dict['text']
        out_text = self.lin(out_text)

        return out_text


