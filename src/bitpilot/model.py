import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

# 工作流程图
# https://mermaid.live/edit#pako:eNp9kstq20AUhl9lOCsZFKPRzbIWhZCQkkUhNG0XqYoZorHsYF0YS8Gt8aIkcZNCqCGkUAJtFiXpphdoF83NeZlO5LxFR1IUGwrVas7oO9_8I50-rIcuBRs8RqIWerLoBEg8y0GUxI1ln3j0uQP8aMy33iEJ97Daw2bFgRdobu4BWo2pL153aadZ7Yo1kianw8npHv-8nY6GGVbYMi5vWKIks3r3znTvjI9fC3NdTS9-TTtmU6zSoBsy0fLn8vhm-xP_8AVtIql-x-feZ40VFm6UUTYbkaiQJAh-fsK_n91ejabqgp3G-Vf__1DlHXLDvOtKUnr0m789nnzc56PdSmWGKtQlOCsRZb798PHTMrXHEvH9vv3g10O-f8h3tqYHC6rIu1DCzXUk8eHP9Ov7yfiAvzkvWJDFb2y7YMcsoTL4lPkkK6GfeRyIW9SnDthi6dImSTqxA04wEG0RCdbC0C87WZh4LbCbpNMVVRK5JKaLbSJmxL_fZTRwKVsIkyAGG2Mzl4Ddhx7Yak2vYkO3NAWbmq7iuibDS7B1q6oZmq5Z2DJquqka5kCGV_m5StUy6zo2aqai6IpqaaoM1G3HIXtUzGc-poO_LoP25Q

class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            # nn.Conv2d(输入通道数， 输出通道数， 卷积核大小， 步长， 偏置)
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8； 12/2=6, 16/2=8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)

        self.gru = nn.GRUCell(192, 192) # 输入特征192维，隐状态记忆192维
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass

    def forward(self, x: torch.Tensor, v, hx=None):
        # 1. 处理图像
        # x 是图像数据 (Batch, 1, Height, Width)
        img_feat = self.stem(x)
        
        # 2. 融合特征
        # v 是传感器数据。self.v_proj(v) 把它变成 192 维。
        # img_feat 也是 192 维。
        # 它们直接相加，然后过一个激活函数。
        x = self.act(img_feat + self.v_proj(v))
        
        # 3. 更新记忆
        # 把融合后的特征 x 和 上一步的记忆 hx 扔进 GRU
        # 得到新的记忆 hx
        hx = self.gru(x, hx)
        
        # 4. 计算动作
        # 新的记忆经过激活函数，再经过全连接层，得到最终动作 act
        act = self.fc(self.act(hx))
        
        return act, None, hx


if __name__ == '__main__':
    Model()
