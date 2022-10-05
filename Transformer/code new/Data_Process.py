from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


class MyData(Dataset):
    def __init__(self, MS4, MS4_up, Pan_down_up, Pan, Label, xy, cut_size):
        # self.train_data1 = MS4
        # self.train_data2 = F.interpolate(MS4_up, scale_factor=2, mode='linear')
        # self.train_data3 = F.interpolate(F.interpolate(Pan_down_up, scale_factor=0.5, mode='linear'), scale_factor=2,
        #                                  mode='linear')
        # self.train_data4 = Pan

        self.train_data1 = MS4
        self.train_data2 = MS4_up
        self.train_data3 = Pan_down_up
        self.train_data4 = Pan
        self.train_labels = Label

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_ms_size = cut_size * 4

    def __getitem__(self, index):
        # 同样用index裁剪I与HS
        x_ms, y_ms = self.gt_xy[index]
        x_up_ms = int(4 * x_ms)
        y_up_ms = int(4 * y_ms)
        x_up_down_pan = int(4 * x_ms)
        y_up_down_pan = int(4 * y_ms)
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)

        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size, y_ms:y_ms + self.cut_ms_size]
        image_up_ms = self.train_data2[:, x_up_ms:x_up_ms + self.cut_pan_ms_size, y_up_ms:y_up_ms + self.cut_pan_ms_size]
        image_up_down_pan = self.train_data3[:, x_up_down_pan:x_up_down_pan + self.cut_pan_ms_size, y_up_down_pan:y_up_down_pan + self.cut_pan_ms_size]
        image_pan = self.train_data4[:, x_pan:x_pan + self.cut_pan_ms_size, y_pan:y_pan + self.cut_pan_ms_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_up_ms, image_up_down_pan, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)
