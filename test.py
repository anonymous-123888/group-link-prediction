import torch
print("Pytorch version：")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is :")
print(torch.backends.cudnn.version())


a = torch.randn(4, 4)
print(a)
b = torch.argsort(a, dim=1, descending=True)
print(b)

c = torch.argsort(b, dim=1, descending=False)
print(c)