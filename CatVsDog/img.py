import cv2
def img_convert(imagePath,lable):
    for i in range(12500):
        imageFile = imagePath + lable + "." + str(i) + ".jpg"
        reimage = "D:\BaiduNetdiskDownload\kaggle\\train2\\" +lable+ "." + str(i) + ".jpg"
        reimg = cv2.imread(imageFile, 0)

        # 把图片的大小统一更改为280；
        resized = cv2.resize(reimg, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(reimage,resized)

img_convert("D:\BaiduNetdiskDownload\CatVSdogtrain\\train\\", "cat")
img_convert("D:\BaiduNetdiskDownload\CatVSdogtrain\\train\\", "dog")