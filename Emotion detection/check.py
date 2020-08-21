import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch.hub
import os
import model
from PIL import Image
from torchvision import transforms
from torchsummary import summary
from visualize.grad_cam import BackPropagation, GradCAM,GuidedBackPropagation

faceCascade = cv2.CascadeClassifier('./visualize/haarcascade_frontalface_default.xml')
shape = (48,48)
classes = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprised',
    'Neutral'
]

def preprocess(image_path):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path)
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        print('no face found')
        face = cv2.resize(image, shape)
    else:
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, shape)

    img = Image.fromarray(face).convert('L')
    inputs = transform_test(img)
    return inputs, face


def get_gradient_image(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    return np.uint8(gradient)


def get_gradcam_image(gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    return np.uint8(gcam)


def guided_backprop(images, model_name):

    for i, image in enumerate(images):
        target, raw_image = preprocess(image['path'])
        image['image'] = target
        image['raw_image'] = raw_image

    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join('../trained', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    summary(net, (1, shape[0], shape[1]))

    result_images = []
    for index, image in enumerate(images):
        img = torch.stack([image['image']])
        bp = BackPropagation(model=net)
        probs, ids = bp.forward(img)
        gcam = GradCAM(model=net)
        _ = gcam.forward(img)

        gbp = GuidedBackPropagation(model=net)
        _ = gbp.forward(img)

        # Guided Backpropagation
        actual_emotion = ids[:,0]
        gbp.backward(ids=actual_emotion.reshape(1,1))
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=actual_emotion.reshape(1,1))
        regions = gcam.generate(target_layer='last_conv')

        # Get Images
        label_image = np.zeros((shape[0],65, 3), np.uint8)
        cv2.putText(label_image, classes[actual_emotion.data], (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        prob_image = np.zeros((shape[0],60,3), np.uint8)
        cv2.putText(prob_image, '%.1f%%' % (probs.data[:,0] * 100), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        guided_bpg_image = get_gradient_image(gradients[0])
        guided_bpg_image = cv2.merge((guided_bpg_image, guided_bpg_image, guided_bpg_image))

        grad_cam_image = get_gradcam_image(gcam=regions[0, 0],raw_image=image['raw_image'])

        guided_gradcam_image = get_gradient_image(torch.mul(regions, gradients)[0])
        guided_gradcam_image = cv2.merge((guided_gradcam_image, guided_gradcam_image, guided_gradcam_image))

        img = cv2.hconcat([image['raw_image'],label_image,prob_image,guided_bpg_image,grad_cam_image,guided_gradcam_image])
        result_images.append(img)
        print(image['path'],classes[actual_emotion.data], probs.data[:,0] * 100)

    cv2.imwrite('../test/guided_gradcam.jpg',cv2.resize(cv2.vconcat(result_images), None, fx=2,fy=2))


def main():
    guided_backprop(
        images=[
            {'path': '../test/angry.jpg'},
            {'path': '../test/happy.jpg'},
            {'path': '../test/sad.jpg'},
            {'path': '../test/surprised.jpg'},
        ],
        model_name='private_model_233_66.t7'
    )


if __name__ == "__main__":
    main()
