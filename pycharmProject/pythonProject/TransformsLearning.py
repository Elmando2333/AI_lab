from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
# Transform就像工具箱
# 比如转换类型，重新定义大小
# 图片-》工具-》结果
# tensor

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
print(img)


writer = SummaryWriter("logs")
# 创建一个工具
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image("Tensor_img",tensor_img)

writer.close()