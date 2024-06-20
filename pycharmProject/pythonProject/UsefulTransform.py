from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



writer = SummaryWriter("logs")
img=Image.open("data/train/ants_image/0013035.jpg")
print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#hahaha
# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.6,0.3,0.5],[0.3,0.2,0.1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize1",img_norm,3)

# rersize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize)
print(img_resize)

#Compose
trans_resize_2 = transforms.Resize(512)
trans_compose= transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)
writer.close()