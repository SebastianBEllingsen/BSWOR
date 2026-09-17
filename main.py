import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


image = cv2.imread("images.png")
print(torch.is_tensor(image))
#convert to tensor
transform = transforms.Compose([transforms.ToTensor()])

tensor = transform(image)


#scrap this shit for now, do an object detection tutorial
# https://medium.com/@ml_dl_explained/understanding-2d-convolutions-in-pytorch-b35841149f5f
edge_kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)
conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=2, padding=1, dilation=2)
conv_layer.weight = torch.nn.Parameter(edge_kernel)

#image = image.unsqueeze(0)  # Add batch dimension
output = conv_layer(image)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image.squeeze(), cmap='gray')
ax[0].set_title("Original Image")
ax[1].imshow(output.detach().squeeze(), cmap='gray')
ax[1].set_title("Edge Detection Output")
plt.show()

cap = cv2.VideoCapture("test.mp4")
cap2 = cv2.VideoCapture(1)


while(True):
    #reads the feed
    ret, frame = cap.read()
    frame

    #print (frame)
    
    #do edits like this
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #then pass in like this
    #cv2.imshow('frame',gray)
    

    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
