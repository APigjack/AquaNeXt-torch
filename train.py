import os
import glob
import logging
import random
import torch
from dataset import DatasetImageMaskContourDist
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import create_train_arg_parser, evaluate
from losses import LossAquaNeXt
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from AquaNeXt import AquaNeXt


USE_AMP = True  # 是否使用混合精度训练
USE_EMA = True  # 是否使用EMA


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 检查张量中的NaN或Inf值
def check_for_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"NaN value found in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf value found in {name}")


# 定义损失函数
def define_loss(loss_type, weights=[1, 1, 1]):
    if loss_type == "AquaNeXt_SD":
        criterion = LossAquaNeXt(weights)
    if loss_type == "AquaNeXt":
        criterion = LossAquaNeXt(weights)

    return criterion


# 构建模型
def build_model(model_type):
    if model_type == "AquaNeXt_SD":
        model = AquaNeXt(num_classes=2, use_ms_cam=False)
    return model


# 训练模型
def train_model(model, inputs, targets, criterion, optimizer, scaler=None):
    optimizer.zero_grad()
    if USE_AMP and scaler is not None:
        with autocast():
            outputs = model(inputs)
            for output, name in zip(outputs, ["output1", "output2", "output3"]):
                check_for_nan_inf(output, name)
            loss = criterion(*outputs, *targets)
            check_for_nan_inf(loss, "loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(inputs)
        for output, name in zip(outputs, ["output1", "output2", "output3"]):
            check_for_nan_inf(output, name)
        loss = criterion(*outputs, *targets)
        check_for_nan_inf(loss, "loss")
        loss.backward()
        optimizer.step()
    return loss


# 初始化数据加载器
def initialize_data_loaders(train_path, train_file, val_file, distance_type, batch_size, val_batch_size):
    trainLoader = DataLoader(
        DatasetImageMaskContourDist(train_path, train_file, distance_type),
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(train_path, val_file, distance_type),
        batch_size=val_batch_size, drop_last=True, shuffle=True,
        num_workers=6,  # 增加数据加载线程数
        pin_memory=True  # 确保数据加载到固定内存中
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(train_path, val_file, distance_type),
        batch_size=val_batch_size, drop_last=True,
        num_workers=6,  # 增加数据加载线程数
        pin_memory=True  # 确保数据加载到固定内存中

    )
    return trainLoader, devLoader, displayLoader


if __name__ == "__main__":
    args = create_train_arg_parser().parse_args()

    print("混合精度训练 (AMP): {}".format("开启" if USE_AMP else "关闭"))
    print("指数移动平均 (EMA): {}".format("开启" if USE_EMA else "关闭"))

    args.distance_type = 'dist_contour_tif'
    args.train_path = r'G:\qj\trans_gd\image'
    args.model_type = 'AquaNeXt_SD'
    args.save_path = r'G:\qj\trans_gd\modelsave'
    args.use_pretrained = True
    args.pretrained_model_path= r'G:\qj\paper\AquaNeXt_SD_gd\modelsave\AquaNeXt_NM_qd\best_val_model_epoch_115.pt'

    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)

    # 配置日志
    log_filename = os.path.join(args.save_path, f"{args.object_type}_training.log")
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    logging.info("Training started")

    # 加载训练文件
    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"))
    random.shuffle(train_file_names)

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 打印训练和验证样本数
    print(f"训练样本数: {len(train_file)}")
    print(f"验证样本数: {len(val_file)}")

    # 选择设备
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    print("使用设备: {}".format(device))

    # 构建模型
    model = build_model(args.model_type).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    print('预训练权重：', args.use_pretrained)

    model = model.to(device)

    # 初始化 EMA
    ema = EMA(model, decay=0.9999) if USE_EMA else None
    if ema:
        ema.register()

    epoch_start = "0"

    # 加载预训练模型
    if args.use_pretrained:
        try:
            print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
            epoch_start = os.path.basename(args.pretrained_model_path).split("_")[-1].split(".")[0]  # 提取数字部分
            print("Start from epoch:", epoch_start)
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    scaler = GradScaler() if USE_AMP else None

    trainLoader, devLoader, displayLoader = initialize_data_loaders(
        args.train_path, train_file, val_file, args.distance_type, args.batch_size, args.val_batch_size
    )

    # 打印 DataLoader 的长度
    print(f"训练 DataLoader 批次数: {len(trainLoader)}")
    print(f"验证 DataLoader 批次数: {len(devLoader)}")

    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999))
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = define_loss(args.model_type)

    # 在主函数中初始化最低损失记录变量
    min_train_loss = float('inf')
    min_val_loss = float('inf')
    best_train_model_state = None
    best_val_model_state = None

    # 开始训练
    for epoch in range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs):
        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        print(f"Epoch {epoch}/{args.num_epochs}:")
        with tqdm(total=len(trainLoader), desc="正在训练") as pbar:
            for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(trainLoader):
                model.train()

                inputs = inputs.to(device)
                targets1 = targets1.to(device)
                targets2 = targets2.to(device)
                targets3 = targets3.to(device)

                targets = [targets1, targets2, targets3]

                # 检查输入数据
                check_for_nan_inf(inputs, "inputs")
                check_for_nan_inf(targets1, "targets1")
                check_for_nan_inf(targets2, "targets2")
                check_for_nan_inf(targets3, "targets3")

                loss = train_model(model, inputs, targets, criterion, optimizer, scaler=scaler)

                # 更新 EMA
                if ema:
                    ema.update()

                writer.add_scalar("loss", loss.item(), global_step)

                running_loss += loss.item() * inputs.size(0)

                pbar.update(1)

        scheduler_cosine.step()

        epoch_loss = running_loss / len(train_file_names)

        # 评估模型
        if epoch % 1 == 0:
            if ema:
                ema.apply_shadow()  # 使用 EMA 参数进行验证
            dev_loss, dev_time, dev_accuracy = evaluate(device, epoch, model, devLoader, writer)
            if ema:
                ema.restore()  # 恢复原始模型参数
            writer.add_scalar("loss_valid", dev_loss, global_step)
            writer.add_scalar("accuracy_valid", dev_accuracy, global_step)
            print(f"训练损失: {epoch_loss} 验证损失: {dev_loss} 验证准确度: {dev_accuracy * 100:.2f}%")

            logging.info(
                f"epoch: {epoch} train_loss: {epoch_loss} valid_loss: {dev_loss} valid_accuracy: {dev_accuracy * 100:.2f}%")
        else:
            print(f"训练损失: {epoch_loss}")

            logging.info(f"epoch: {epoch} train_loss: {epoch_loss}")

        # 检查并记录当前周期的最低训练损失和模型状态
        if epoch_loss < min_train_loss:
            min_train_loss = epoch_loss
            best_train_model_state = model.state_dict()

        # 检查并记录当前周期的最低验证损失和模型状态
        if dev_loss < min_val_loss:
            min_val_loss = dev_loss
            best_val_model_state = model.state_dict()

        # 每5个epoch保存损失最低的训练和验证模型
        try:
            if (epoch - int(epoch_start)) % 5 == 0:
                if best_train_model_state is not None:
                    save_path = os.path.join(args.save_path, f"best_train_model_epoch_{epoch}.pt")
                    torch.save(best_train_model_state, save_path)
                    print(f"Saved best train model from epochs {epoch - 4} to {epoch} with train loss {min_train_loss}")
                    min_train_loss = float('inf')  # 重置最低训练损失
                    best_train_model_state = None  # 重置最佳训练模型状态

                if best_val_model_state is not None:
                    save_path = os.path.join(args.save_path, f"best_val_model_epoch_{epoch}.pt")
                    torch.save(best_val_model_state, save_path)
                    print(f"Saved best val model from epochs {epoch - 4} to {epoch} with val loss {min_val_loss}")
                    min_val_loss = float('inf')  # 重置最低验证损失
                    best_val_model_state = None  # 重置最佳验证模型状态
        except Exception as e:
            print(f"Error saving model: {e}")

    if USE_AMP:
        torch.cuda.empty_cache()
