from torch import nn

class MLP(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features, 
        out_features, 
        act_layer=nn.GELU, 
        norm_layer=None, 
        bias=True, 
        kernel_size=5,
        stride=1,
        dropout=0.1
    ):
        super(MLP, self).__init__()

        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1)//2, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.fc1(x.transpose(1,2))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x * x_mask) * x_mask
        x = self.drop(x) 
        return x.transpose(1,2)