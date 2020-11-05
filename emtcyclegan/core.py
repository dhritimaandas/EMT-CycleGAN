if torch.cuda.is_available():
    cuda = torch.device('cuda')
    Tensor = torch.cuda.FloatTensor
else:
    cuda = torch.device('cpu')
    Tensor = torch.FloatTensor



