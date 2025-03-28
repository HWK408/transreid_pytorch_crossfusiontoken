import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """ 优化的LoRA线性层实现 """

    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 冻结原始参数
        for param in linear_layer.parameters():
            param.requires_grad = False

        # 获取特征维度
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # 初始化LoRA参数
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # 使用更好的初始化策略
        nn.init.normal_(self.lora_A, std=1 / self.rank)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)
        self.active_adapter = True

    def forward(self, x):
        # 基础层输出
        base_output = self.linear(x)

        if not self.active_adapter:
            return base_output

        # LoRA路径
        lora_input = self.dropout(x)
        # 更高效的矩阵乘法顺序
        lora_output = (lora_input @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_output + lora_output

    def enable_adapter(self):
        self.active_adapter = True

    def disable_adapter(self):
        self.active_adapter = False


import torch
import torch.nn as nn
from datetime import datetime


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.1):
    """为TransReID模型的所有transformer blocks应用LoRA

    Args:
        model: TransReID模型
        rank (int): LoRA的秩
        alpha (int): LoRA缩放因子
        dropout (float): Dropout概率

    Returns:
        model: 应用了LoRA的模型
    """
    print(f"Starting LoRA application process at {datetime.utcnow().strftime('%Y-%m-%D %H:%M:%S')}")
    print(f"User: HWK408")
    print(f"Configuration: rank={rank}, alpha={alpha}, dropout={dropout}")

    try:
        if not hasattr(model, 'base') or not hasattr(model.base, 'blocks'):
            raise AttributeError("Model structure does not match expected TransReID architecture")

        blocks = model.base.blocks
        num_blocks = len(blocks)

        if num_blocks != 12:
            print(f"Warning: Expected 12 blocks, found {num_blocks} blocks")

        total_lora_layers = 0

        # 遍历所有blocks
        for i, block in enumerate(blocks):
            print(f"\nProcessing Block {i + 1}/{num_blocks}")

            # 1. 处理Attention层
            if hasattr(block, 'attn'):
                # QKV投影
                if hasattr(block.attn, 'qkv'):
                    original_qkv = block.attn.qkv
                    block.attn.qkv = LoRALinear(
                        original_qkv,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    total_lora_layers += 1
                    print(f"✓ Block {i + 1}: Applied LoRA to QKV projection (768 → 2304)")

                # 输出投影
                if hasattr(block.attn, 'proj'):
                    original_proj = block.attn.proj
                    block.attn.proj = LoRALinear(
                        original_proj,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    total_lora_layers += 1
                    print(f"✓ Block {i + 1}: Applied LoRA to attention output (768 → 768)")

            # 2. 处理MLP层
            if hasattr(block, 'mlp'):
                # FC1
                if hasattr(block.mlp, 'fc1'):
                    original_fc1 = block.mlp.fc1
                    block.mlp.fc1 = LoRALinear(
                        original_fc1,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    total_lora_layers += 1
                    print(f"✓ Block {i + 1}: Applied LoRA to MLP FC1 (768 → 3072)")

                # FC2
                if hasattr(block.mlp, 'fc2'):
                    original_fc2 = block.mlp.fc2
                    block.mlp.fc2 = LoRALinear(
                        original_fc2,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    total_lora_layers += 1
                    print(f"✓ Block {i + 1}: Applied LoRA to MLP FC2 (3072 → 768)")

            print(f"✓ Completed Block {i + 1}/{num_blocks}")

        print(f"\nLoRA application complete!")
        print(f"Total LoRA layers added: {total_lora_layers}")
        print(f"Memory impact estimation: {total_lora_layers * rank * (768 + 768) * 4 / (1024 * 1024):.2f} MB")

        return model

    except Exception as e:
        print(f"Error occurred while applying LoRA: {str(e)}")
        print("Model structure:")
        print(model)
        raise e


# 辅助函数用于启用/禁用LoRA
def enable_lora(model):
    """启用所有LoRA层"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enable_adapter()
    print("All LoRA layers enabled")


def disable_lora(model):
    """禁用所有LoRA层"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.disable_adapter()
    print("All LoRA layers disabled")

