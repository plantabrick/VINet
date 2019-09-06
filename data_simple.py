#coding=utf-8
import torch
import torchvision
import torchvision.transforms as transforms

# torchvision输出的是PILImage，值的范围是[0, 1].
# 我们将其转化为tensor数据，并归一化为[-1, 1]。
transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print len(trainset)
print len(trainloader)



#下面是代码只是为了给小伙伴们显示一个图片例子，让大家有个直觉感受。
# functions to show an image
import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()




# show some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))