import torch
from tqdm import tqdm
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse

def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    correct = 0
    total = 0
    start = time.perf_counter()
    pbar = tqdm(total=len(data_loader), desc="正在评估")
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            _, inputs, targets1, targets2, targets3 = data
            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)
            outputs = model(inputs)
            loss = F.nll_loss(outputs[0], targets1.squeeze(1))
            losses.append(loss.item())

            # 计算准确度
            preds = torch.argmax(outputs[0], dim=1, keepdim=True)  # 调整预测值形状
            correct_batch = (preds == targets1).sum().item()  # 计算正确预测的像素点数量
            total_batch = targets1.numel()  # 计算总的像素点数量
            correct += correct_batch
            total += total_batch
            pbar.update(1)
    pbar.close()  # 关闭评估进度条
    writer.add_scalar("Dev_Loss", np.mean(losses), epoch)
    accuracy = correct / total  # 计算准确度
    return np.mean(losses), time.perf_counter() - start, accuracy

def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _,_ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy()
            output_final = np.argmax(output_mask, axis=1).astype(float)
            output_final = torch.from_numpy(output_final).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train",val_batch_size)
                save_image(output_final, "Prediction_train",val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break

def create_train_arg_parser():
    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to img tif files")
    parser.add_argument("--val_path", type=str, help="path to img tif files")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_contour",
        help="select distance transform type - dist_mask,dist_contour,dist_contour_tif",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=4, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", type=str, help="Model save path.")
    # parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training.')

    return parser

def create_validation_arg_parser():
    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: bsinet",
    )
    parser.add_argument("--test_path", type=str, help="path to img tif files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")

    return parser
