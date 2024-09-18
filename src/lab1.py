#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2023
#

import argparse
import random

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from model import autoencoderMLP4Layer


def main():
    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')

    args = argParser.parse_args()

    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z

    device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28  # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    while idx >= 0:
        idx = int(input("Enter index > "))
        if 0 <= idx <= train_set.data.size()[0]:
            print('label = ', train_set.targets[idx].item())
            img = train_set.data[idx]
            print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.type(torch.float32)
            print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
            img = (img - torch.min(img)) / torch.max(img)
            print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))

            # plt.imshow(img, cmap='gray')
            # plt.show()

            img = img.to(device=device)
            # print('break 7: ', torch.max(img), torch.min(img), torch.mean(img))
            print('break 8 : ', img.shape, img.dtype)
            img = img.view(1, img.shape[0] * img.shape[1]).type(torch.FloatTensor)
            print('break 9 : ', img.shape, img.dtype)
            with torch.no_grad():
                output = model(img)

            output = output.view(28, 28).type(torch.FloatTensor)
            print('break 10 : ', output.shape, output.dtype)
            print('break 11: ', torch.max(output), torch.min(output), torch.mean(output))

            img = img.view(28, 28).type(torch.FloatTensor)

            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 2, 2)
            plt.imshow(output, cmap='gray')
            plt.show()

            # Generate a noisy image and adding random noise to the image
            noisyImage = img + torch.rand([28, 28])

            # Ensure noisyImage has correct shape for the model
            noisyImage_flat = noisyImage.view(1, 28 * 28).type(torch.FloatTensor)

            # Pass noisyImage through model to get the denoised image
            denoisedImage_flat = model(noisyImage_flat)

            # Reshape the denoised image back to 28x28 for display
            denoisedImage = denoisedImage_flat.view(28, 28)

            # Second window displaying img, noisyImage, and denoisedImage
            f2 = plt.figure(figsize=(10, 5))  # Adjust size to accommodate three images

            # First image: Original image
            f2.add_subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')

            # Second image: Noisy image
            f2.add_subplot(1, 3, 2)
            plt.imshow(noisyImage, cmap='gray')
            plt.title('Noisy')

            # Third image: Denoised image
            f2.add_subplot(1, 3, 3)
            plt.imshow(denoisedImage.detach().numpy(), cmap='gray')
            plt.title('Denoised')

            plt.show()

            img = img.view(1, img.shape[0] * img.shape[1]).type(torch.FloatTensor)
            print('break 9 : ', img.shape, img.dtype)
            with torch.no_grad():
                output = model.encode(img)

            #Beggining of linear interpolation section

            rand1 = random.randint(a=1, b=train_set.data.size()[0])
            rand2 = random.randint(a=1, b=train_set.data.size()[0])


            # Flatten the awdimages (since the network expects 28*28 dimensional input)
            img1_flat = train_set.data[rand1].view(1,28* 28).type(torch.FloatTensor)/255.0
            img2_flat = train_set.data[rand2].view(1,28* 28).type(torch.FloatTensor)/255.0

            # Encode both images to get their bottleneck representations
            bottleneck1 = model.encode(img1_flat)
            bottleneck2 = model.encode(img2_flat)

            print("Break 12: bottleneck1 size:",bottleneck1.shape, "  bottleneck2 size: ", bottleneck2.shape)

            # Number of interpolation steps
            n_steps = 8
            interpolations = []

            # Linear interpolation between bottleneck representations
            for i in range(1, n_steps):
                alpha = i / n_steps
                interpolatedImage = bottleneck1 * (1 - alpha) + bottleneck2 * alpha
                interpolations.append(interpolatedImage)

            # Add start and end points to the interpolations
            interpolations.insert(0, bottleneck1)
            interpolations.append(bottleneck2)

            # Decode each interpolated bottleneck tensor to get the reconstructed images
            with torch.no_grad():
                decoded_images = [model.decode(interp).view(28, 28) for interp in interpolations]

            # Plot the full set of interpolated images
            plt.figure(figsize=(15, 5))
            for i, decoded_image in enumerate(decoded_images):
                plt.subplot(1, n_steps + 2, i + 1)  # n_steps + 2 to account for both ends
                plt.imshow(decoded_image.detach().numpy(), cmap='gray')
                plt.axis('off')

            plt.show()


###################################################################

if __name__ == '__main__':
    main()
