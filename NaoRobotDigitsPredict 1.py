import time
from PIL import Image
from naoqi import ALProxy
import numpy as np
import digits_predict
video_service = ALProxy("ALVideoDevice", "172.18.16.54", 9559)
tts= ALProxy('ALTextToSpeech', '172.18.16.54',9559)
tts.say('Hello, I am nao, and I could help with recognizing the fashion items. Let\'s do that')
resolution = 2    # VGA
colorSpace = 11

videoClient = video_service.subscribeCamera("Test",0, resolution, colorSpace, 5)
tts.say('are you prepare to show me something, just put it in front of my camera')
tts.say('are you ready? three two one')
t0 = time.time()
# Get a camera image.
# image[6] contains the image data passed as an array of ASCII chars.
naoImage = video_service.getImageRemote(videoClient)
t1 = time.time()
tts.say('Kaca')
tts.say('ok I got it')
print "acquisition delay ", t1 - t0

video_service.unsubscribe(videoClient)


# Now we work with the image returned and save it as a PNG  using ImageDraw
# package.

# Get the image size and pixel array.
imageWidth = naoImage[0]
imageHeight = naoImage[1]
array = naoImage[6]
image_string = str(bytearray(array))

# Create a PIL Image from our pixel array.
im =  Image.frombytes("RGB", (imageWidth,imageHeight), image_string)

# Save the image.
im.save("camImage.png", "PNG")

image = Image.open('camImage.png')

image = image.resize((32, 32))

image_array = np.array(image)
image_array = image_array.astype('float32') / 255.0

input_data = image_array.reshape((1, 32, 32, 3))
import matplotlib.pyplot as plt

plt.imshow(image_array)
plt.show()

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

res = digits_predict.prediction(input_data)
tts.say('ok I think it is '+cifar10_classes[res[0]])
print(cifar10_classes[res[0]])

