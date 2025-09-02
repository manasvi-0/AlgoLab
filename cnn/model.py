import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import AlexNet_Weights, VGG16_Weights
import matplotlib.pyplot as plt

model_name = 'vgg_16'

def alexnetVisualise(input_image):
    #Using pre-trained model architectures rather than training 
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1)
    model.eval()


    # filename = "/home/gayathri/Downloads/dog.jpeg"
    # input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # for i, module in enumerate(model.modules()):
    #     print(f"Module {i}: {module}")

    #register activations using a hook function
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    model.features[0].register_forward_hook(get_activation('conv1'))
    model.features[3].register_forward_hook(get_activation('conv2'))
    model.features[6].register_forward_hook(get_activation('conv3'))
    model.features[8].register_forward_hook(get_activation('conv4'))
    model.features[10].register_forward_hook(get_activation('conv5'))

    #run the model
    output = model(input_batch)

    output_images = []

    for layer_name, activation in activations.items():
        print(f'Layer: {layer_name}, Shape: {activation.shape}')
        output_images.append(activation[0, 0, : , :].cpu().numpy())
        # plt.imshow(activation[0, 0, : , :].cpu().numpy(), cmap='viridis')
        # plt.title(layer_name)
        # plt.show()
    
    return output_images

def resnetVisualise(input_image):
    model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    model.eval()

    # filename = "/home/gayathri/Downloads/dog.jpeg"
    # input_image = Image.open(filename)

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for i, module in enumerate(model.modules()):
    #     print(f"Module {i}: {module}")

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4[2].conv3.register_forward_hook(get_activation('layer4'))

    output = model(input_batch)

    output_images = []

    for layer_name, activation in activations.items():
        print(f'Layer: {layer_name}, Shape: {activation.shape}')
        output_images.append(activation[0, 0].cpu().numpy())
        # plt.imshow(activation[0, 0].cpu().numpy(), cmap='gray')
        # plt.title(layer_name)
        # plt.show()

    return output_images

def vgg16Visualise(input_image):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights = VGG16_Weights.IMAGENET1K_V1)
    model.eval()

    # filename = "/home/gayathri/Downloads/dog.jpeg"
    # input_image = Image.open(filename)

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # for i, module in enumerate(model.modules()):
    #     print(f"Module {i}: {module}")

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    model.features[0].register_forward_hook(get_activation('conv1'))
    model.features[5].register_forward_hook(get_activation('conv2'))
    model.features[10].register_forward_hook(get_activation('conv3'))
    model.features[17].register_forward_hook(get_activation('conv4'))
    model.features[28].register_forward_hook(get_activation('conv5'))

    output = model(input_batch)

    output_images = []

    for layer_name, activation in activations.items():
        print(f'Layer: {layer_name}, Shape: {activation.shape}')
        output_images.append(activation[0, 0].cpu().numpy())
        # plt.imshow(activation[0, 0].cpu().numpy(), cmap='gray')
        # plt.title(layer_name)
        # plt.show()

    return output_images
