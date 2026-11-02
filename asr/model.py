"""
语音识别模型: 5层双向LSTM + 线性投影 + CTC损失
Speech Recognition Model: 5-layer BiLSTM + Linear Projection + CTC Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSpeech2(nn.Module):
    """5层双向LSTM语音识别模型，使用CTC损失函数训练"""

    def __init__(
        self,
        input_dim: int = 80,         # 输入特征维度 (如80维梅尔滤波器组)
        hidden_dim: int = 512,       # LSTM隐藏层维度
        num_layers: int = 5,         # LSTM层数
        num_classes: int = 5000,     # 输出类别数 (字符集大小+1, 含CTC blank)
        dropout: float = 0.3,        # Dropout比率
        bidirectional: bool = True,  # 是否双向LSTM
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # 输入投影: 将特征映射到LSTM输入维度
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 5层双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 输出投影层: LSTM输出 -> 字符概率分布
        self.output_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        """
        前向传播

        Args:
            x: 输入特征 [batch, time, input_dim]
            x_lengths: 每个样本实际长度 [batch]

        Returns:
            log_probs: log softmax概率 [batch, time, num_classes]
            output_lengths: 输出序列长度 [batch]
        """
        batch_size, max_time, _ = x.shape

        # 输入投影
        x = self.input_fc(x)  # [B, T, hidden_dim]

        # 打包变长序列以提高效率
        x_lengths = x_lengths.to("cpu")
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True, enforce_sorted=False
        )

        # 5层双向LSTM
        packed_output, _ = self.lstm(packed_input)

        # 解包
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # [B, T, hidden_dim * num_directions]

        # 输出投影
        logits = self.output_fc(lstm_out)  # [B, T, num_classes]

        # Log softmax (CTC损失需要)
        log_probs = F.log_softmax(logits, dim=-1)

        # 计算输出长度 (LSTM不改变时间维度)
        output_lengths = x_lengths.to(logits.device)

        return log_probs, output_lengths

    def get_intermediate(self, x: torch.Tensor, x_lengths: torch.Tensor):
        """
        获取中间层表示(用于分析)
        返回每层LSTM的输出
        """
        x = self.input_fc(x)

        x_lengths = x_lengths.to("cpu")
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True, enforce_sorted=False
        )

        # 手动逐层获取LSTM输出
        layer_outputs = []
        current = packed_input
        for i in range(self.num_layers):
            lstm_layer = nn.LSTM(
                input_size=self.hidden_dim if i > 0 else self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=0,
            ).to(x.device)
            # 使用原有权重
            for name, param in lstm_layer.named_parameters():
                weight_name = f'lstm.weight_ih_l{i}'
                reverse_name = f'lstm.weight_ih_l{i}_reverse'
                if name == 'weight_ih_l0':
                    param.data.copy_(getattr(self.lstm, weight_name).data)
                elif name == 'weight_hh_l0':
                    param.data.copy_(getattr(self.lstm, f'lstm.weight_hh_l{i}').data)
                # 简化处理: 仅做前向传播记录

            packed_out, _ = lstm_layer(current)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            layer_outputs.append(out)
            current = packed_out

        return layer_outputs


class DeepSpeech2Small(nn.Module):
    """轻量版: 5层单向LSTM，适合快速实验"""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_classes: int = 5000,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        batch_size, max_time, _ = x.shape

        x = self.input_fc(x)

        x_lengths = x_lengths.to("cpu")
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        logits = self.output_fc(lstm_out)
        log_probs = F.log_softmax(logits, dim=-1)
        output_lengths = x_lengths.to(logits.device)

        return log_probs, output_lengths
