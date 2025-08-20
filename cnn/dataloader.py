import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


#Downloading the datasets
train_set = torchvision.datasets.FashionMNIST('./data', train = True, transform = transform, download = True)
validation_set = torchvision.datasets.FashionMNIST('./data', train = False, transform = transform, download = True)

#Preparing the dataloader
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 4, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = 4, shuffle = False)

#class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

print('Training set has {} instances'.format(len(train_set)))
print('Validation set has {} instances'.format(len(validation_set)))