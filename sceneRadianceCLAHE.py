import cv2
def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(2, 2))
    # clipLimit用于限制对比度的放大
    # tileGridSize是进行像素均衡化的网格大小
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance