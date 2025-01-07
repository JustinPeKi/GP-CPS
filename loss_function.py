import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn_semi(out1, out2):
    mask = torch.where((out2 > 0.5), torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    mask.requires_grad = False
    # print(torch.min(out2))
    loss = loss_fn_dice(out1, mask)
    return loss


def loss_fn_latent(latent1, latent2):
    loss = F.mse_loss(latent1, latent2)
    return loss


def loss_fn_dice(out, label):
    label = label.float()
    smooth = 1e-5
    intersect = torch.sum(out * label)
    y_sum = torch.sum(label * label)
    z_sum = torch.sum(out * out)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def loss_fn_bce(out, label):
    # 确保标签是长整型，因为PyTorch的cross_entropy函数需要
    batch_size, channels, H, W = out.shape

    out = out.view(batch_size, channels, -1)  # 调整维度，-1会自动计算高度*宽度的值

    label = label.float()
    label = label.view(batch_size, channels, -1)
    # 计算交叉熵损失
    bce = nn.BCELoss()
    loss = bce(out, label)
    return loss


def mysoftmax(image):
    batch_size, channels, H, W = image.shape
    # 对每个像素使用softmax
    # 注意：softmax默认是在最后一个维度上应用，这里是W
    # 但是我们需要在每个像素上应用，所以我们需要先调整维度
    # 将每个像素的通道值展平，然后应用softmax，最后再恢复原形状
    image = image.view(batch_size, channels, -1)  # 调整维度，-1会自动计算高度*宽度的值
    image = F.softmax(image, dim=-1)  # 在通道维度上应用softmax
    image = image.view(batch_size, channels, H, W)  # 恢复到原来的形状
    return image


def acc_fn(out, label):
    out = torch.where((out > 0.5), torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    acc = 1 - loss_fn_dice(out, label)
    return acc


def loss_fn_lip():
    pass
