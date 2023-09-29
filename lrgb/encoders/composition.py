import torch


class Concat2NodeEncoder(torch.nn.Module):
    """Encoder that concatenates two node encoders.
    """

    def __init__(self, enc1_cls, enc2_cls, in_dim, emb_dim, enc2_dim_pe):
        super().__init__()
        # PE dims can only be gathered once the cfg is loaded.
        self.encoder1 = enc1_cls(in_dim=in_dim, emb_dim=emb_dim - enc2_dim_pe)
        self.encoder2 = enc2_cls(in_dim=in_dim, emb_dim=emb_dim, expand_x=False)

    def forward(self, x, pestat):
        x = self.encoder1(x, pestat)
        x = self.encoder2(x, pestat)
        return x
