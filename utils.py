import os

import torch
import torchvision
from scipy import ndimage
import numpy as np
from dataset import Fluo_N2DH_SIM_PLUS
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=0,
    pin_memory=True
):
    train_ds = Fluo_N2DH_SIM_PLUS(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Fluo_N2DH_SIM_PLUS(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    print("=> checking accuracy")
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y = y.to(device)
            y = Fluo_N2DH_SIM_PLUS.split_mask(y).to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()  # bitwise check
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


# def save_predictions_as_imgs(
#     loader, model, folder=r"C:\BGU\seg_unet\Fluo-N2DH-SIM+_training-datasets\saved", device="cuda"
# ):
#     print("=> saving images")
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         # os.makedirs("C:\\BGU\\u-net_seg\\Fluo-N2DH-SIM+_training-datasets\\saved\\", exist_ok=True)
#         # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
#         # torchvision.utils.save_image(y, f"{folder}/{idx}.png")
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
#
#     model.train()
def apply_color_map(input_tensor):
    color_map = torch.tensor([
        [0, 0, 0],  # Label 0 (black)
        [0, 255, 0],  # Label 1 (green)
        [255, 255, 255]  # Label 2 (white)
    ], dtype=torch.uint8)
    color_map = color_map.to(input_tensor.device)

    color_map_tensor = color_map[input_tensor]

    color_map_tensor = color_map_tensor.permute(0, 3, 1, 2)
    return color_map_tensor


def save_predictions_as_imgs(loader, model, folder=r"C:\BGU\seg_unet\Fluo-N2DH-SIM+_training-datasets\saved",
                             device="cuda"):
    print("=> saving images")
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds_softmax = F.softmax(preds, dim=1)  # Apply softmax along the class dimension
        print(f"preds:\n\{preds}\n\n")
        # Convert softmax probabilities to class predictions (integer values)
        _, predicted_classes = torch.max(preds_softmax, dim=1)  # Get the index of the maximum probability

        break

        colored_preds = apply_color_map(predicted_classes).type(torch.uint8)

        colored_gt = apply_color_map(y.long()).type(torch.uint8)

        for i in range(colored_preds.shape[0]):  # Loop through the batch
            # Permute and move to CPU
            pred_img = colored_preds[i].permute(1, 2, 0).cpu().numpy()
            gt_img = colored_gt[i].permute(1, 2, 0).cpu().numpy()

            pred_img_pil = Image.fromarray(pred_img)
            gt_img_pil = Image.fromarray(gt_img)

            pred_img_pil.save(f"{folder}/pred_{idx}_{i}.png")
            gt_img_pil.save(f"{folder}/gt_{idx}_{i}.png")

    model.train()

def save_test_predictions_as_imgs(
    loader, model, folder=r"C:\BGU\seg_unet\Fluo-N2DH-SIM+_training-datasets\saved", device="cuda", type="train"
):
    model.eval()
    if type == "train":

        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")
            torchvision.utils.save_image(Fluo_N2DH_SIM_PLUS.split_mask(y), f"{folder}/{idx}.png")

    else:
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        for idx, x in enumerate(loader):
            x = x.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            # x = torch.where(x > 0, torch.tensor(1.0), torch.tensor(0.0))
            # torchvision.utils.save_image(x, f"{folder}/x_{idx}.png")
            # num_correct += (preds == x).sum()
            # num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * x).sum()) / ((preds + x).sum() + 1e-8)

        # print(
        #     f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}%"
        # )
        # print(f"Dice score: {dice_score / len(loader)}")
    model.train()


def seg_measure(channel_axis, three_d=False, foreground_class_index=2):
    if not three_d:
        strel = np.zeros([3, 3])
        strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    else:
        strel = np.zeros([3, 3, 3, 3, 3])
        strel[1][1] = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    def connected_components(input_np):
        labeled = np.zeros_like(input_np, dtype=np.uint16)
        max_num = 0
        for d1, images in enumerate(input_np):
            for d2, image in enumerate(images):
                labeled_image, max_num_temp = ndimage.label(image, structure=strel)
                labeled[d1, d2] = labeled_image
                max_num = np.maximum(max_num, max_num_temp)

        return labeled, np.array(max_num).astype(np.float32)

    def seg_numpy(gt, seg):
        gt_labeled, _ = connected_components(gt)
        seg_labeled, _ = connected_components(seg)
        all_iou = []
        for gt1, seg1 in zip(gt_labeled, seg_labeled):
            for gt, seg in zip(gt1, seg1):
                for this_label in np.unique(gt):
                    if this_label == 0:
                        continue
                    all_iou.append(0.)
                    bw = gt == this_label
                    l_area = np.sum(bw).astype(np.float32)
                    overlaping_inds = seg[bw]
                    for s in np.unique(overlaping_inds):
                        if s == 0:
                            continue
                        intersection = np.sum(overlaping_inds == s).astype(np.float32)
                        overlap = intersection / l_area
                        if overlap > 0.5:
                            s_area = np.sum(seg == s).astype(np.float32)
                            iou = intersection / (l_area + s_area - intersection)
                            all_iou[-1] = iou
        if not len(all_iou):
            return np.nan
        return np.mean(all_iou)


    def calc_seg(gt_sequence, output_sequence):
        gt_sequence = torch.squeeze(gt_sequence, dim=channel_axis)
        gt_valid = gt_sequence > -1
        gt_sequence = gt_sequence.float() * gt_valid.float()
        gt_fg = (gt_sequence == foreground_class_index).float()
        output_classes = torch.argmax(output_sequence, dim=channel_axis)
        output_foreground = (output_classes == foreground_class_index).float()

        gt_fg_np = gt_fg.cpu().numpy() if gt_fg.is_cuda else gt_fg.numpy()
        output_foreground_np = output_foreground.cpu().numpy() if output_foreground.is_cuda else output_foreground.numpy()

        seg_measure_value = seg_numpy(gt_fg_np, output_foreground_np)

        return seg_measure_value

    return calc_seg



def seg_measure_unit_test():
    three_d = False
    channel_axis = 4 if not three_d else 5
    calc_seg_meas = seg_measure(channel_axis=channel_axis, three_d=three_d,
                                foreground_class_index=1)
    h = w = d = 30
    batch_size = 3
    unroll_len = 2
    if three_d:
        gt_sequence = np.zeros((batch_size, unroll_len, d, h, w, 1)).astype(np.float32)
        output_sequence = np.zeros((batch_size, unroll_len, d, h, w, 3)).astype(np.float32)
        output_sequence[:, :, :, :, 0] = 0.25
    else:
        gt_sequence = np.zeros((batch_size, unroll_len, h, w, 1)).astype(np.float32)
        output_sequence = np.zeros((batch_size, unroll_len, h, w, 3)).astype(np.float32)
        output_sequence[:, :, :, :, 0] = 0.25
    objects = [(12, 20, 0, 5), (0, 9, 0, 5), (12, 20, 9, 20), (0, 9, 9, 20)]
    i = 0
    for b in range(batch_size):
        for u in range(unroll_len):
            for obj_id, (xs, xe, ys, ye) in enumerate(objects):
                gt_sequence[b, u, ys + i:ye + i, xs + i:xe + i] = obj_id + 1
                output_sequence[b, u, max(ys + i + 2, 0):max(ye + i, 0), max(xs + i, 0):max(xe + i, 0), 1] = 0.5

            # i += 1
    print(calc_seg_meas(torch.from_numpy(gt_sequence), torch.from_numpy(output_sequence)))

if __name__ == '__main__':
    seg_measure_unit_test()