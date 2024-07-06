import lightning as L
import torch
from torch import nn


PENULTIMATE_LAYER_NAME = "model.fc[0]"
LAST_LAYER_NAME = "model.fc[1]"
# PENULTIMATE_LAYER_NAME = "fc[0]"
# LAST_LAYER_NAME = "fc[1]"


# Based on https://github.com/omihub777/ViT-CIFAR/tree/f5c8f122b4a825bf284bc9b471ec895cc9f847ae # noqa: W505
class TransformerEncoder(nn.Module):
    def __init__(
        self, feats: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0
    ):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = (
            self.q(x)
            .view(b, n, self.head, self.feats // self.head)
            .transpose(1, 2)
        )
        k = (
            self.k(x)
            .view(b, n, self.head, self.feats // self.head)
            .transpose(1, 2)
        )
        v = (
            self.v(x)
            .view(b, n, self.head, self.feats // self.head)
            .transpose(1, 2)
        )

        score = torch.nn.functional.softmax(
            torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, dim=-1
        )  # (b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)  # (b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o


class ViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 32,
        patch: int = 8,
        dropout: float = 0.0,
        num_layers: int = 7,
        hidden: int = 384,
        mlp_hidden: int = 384 * 4,
        head: int = 8,
        is_cls_token: bool = True,
    ):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch  # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3  # 48 # patch vec length
        num_tokens = (
            (self.patch**2) + 1 if self.is_cls_token else (self.patch**2)
        )

        self.emb = nn.Linear(f, hidden)  # (b, n, f)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        )
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        enc_list = [
            TransformerEncoder(
                hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head
            )
            for _ in range(num_layers)
        ]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),  # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token and self.cls_token is not None:
            out = torch.cat(
                [self.cls_token.repeat(out.size(0), 1, 1), out], dim=1
            )
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class ViTWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.is_vit = True

    def forward(self, img):
        return self.model(img)

    def configure_optimizers(self):
        INIT_LR = 1e-3
        N_WARMUP_STEPS = 5

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=INIT_LR,
            betas=(0.9, 0.999),
            weight_decay=5e-5,
        )
        initial_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=INIT_LR,
            total_iters=N_WARMUP_STEPS,  # warm
        )
        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,
            eta_min=1e-5,
        )
        # self.base_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizer,
        #     gamma=0.8,
        #     last_epoch=200,
        # )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[initial_scheduler, base_scheduler],
            milestones=[N_WARMUP_STEPS],
        )
        return [optimizer], [scheduler]


def DefaultViT(num_classes: int) -> torch.nn.Module:
    vit = ViT(num_classes=num_classes)
    # return vit
    return ViTWrapper(vit)

    # Parameters based on https://github.com/omihub777/ViT-CIFAR/tree/f5c8f122b4a825bf284bc9b471ec895cc9f847ae # noqa: W505
    # vit = vit_pytorch.SimpleViT(
    #     image_size=32,
    #     patch_size=8,
    #     num_classes=num_classes,
    #     heads=12,
    #     depth=7,
    #     dim=384,
    #     mlp_dim=384,
    # )
