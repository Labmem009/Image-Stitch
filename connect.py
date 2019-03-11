import numpy as np
import imutils
import cv2


class Stitcher:
    def __init__(self):
        # opencv版本
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # 将需要拼接的图像成为图像组
        (imageB, imageA) = images
        #调用自定义函数找出特征向量以及关键点
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 在两图间匹配
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)

        # 无匹配则直接返回
        if M is None:
            return None
        #将M矩阵划分为匹配点，单应性变换矩阵以及匹配点状态
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 是否显示
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # 返回结果与匹配点图示
            return (result, vis)

        # 返回结果
        return result

    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 版本区别
        if self.isv3:
            # 使用SIFT_create()方法实例化DOG空间关键点
            descriptor = cv2.xfeatures2d.SIFT_create()
            #将关键点与特征向量分离
            (kps, features) = descriptor.detectAndCompute(image, None)

        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # 将关键点转化为Numpy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回关键点与特征向量的元组
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # 使用BruteForceMatcher对象对两幅图片的特征向量进行匹配
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # 在原始匹配点间循环
        for m in rawMatches:
            # 确保彼此之间的距离在比率内
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # 单应性变换需要至少四对匹配
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算出匹配点之间的单应性变换以及状态
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)
            return (matches, H, status)

        # 不足四对无返回
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 对连接图初始化
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 在匹配点循环
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                # 将两图间匹配点连线
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化连接图像
        return vis

import imutils
import cv2
import matplotlib.pyplot as plt

imageF = cv2.imread('C://Users//Administrator//Desktop//p1.png')
imageE = cv2.imread('C://Users//Administrator//Desktop//p2.png')
imageD = cv2.imread('C://Users//Administrator//Desktop//p3.png')
imageC = cv2.imread('C://Users//Administrator//Desktop//p4.png')
imageA = cv2.imread('C://Users//Administrator//Desktop//p5.png')
imageB = cv2.imread('C://Users//Administrator//Desktop//p6.png')
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
imageC = imutils.resize(imageC, width=400)
imageD = imutils.resize(imageD, width=400)
imageE = imutils.resize(imageE, width=400)
imageF = imutils.resize(imageF, width=400)

# 从右至左依次拼接图像
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
cv2.imshow("Keypoint Matches56", vis)
cv2.imshow("Result56", result)
#cv2.imwrite('C://Users//Administrator//Desktop//l56.png',vis)
(result, vis) = stitcher.stitch([imageC, result], showMatches=True)
cv2.imshow("Keypoint Matches456", vis)
cv2.imshow("Result456", result)
#cv2.imwrite('C://Users//Administrator//Desktop//l456.png',vis)
(result, vis) = stitcher.stitch([imageD, result], showMatches=True)
cv2.imshow("Keypoint Matches3456", vis)
cv2.imshow("Result3456", result)
#cv2.imwrite('C://Users//Administrator//Desktop//l3456.png',vis)
(result, vis) = stitcher.stitch([imageE, result], showMatches=True)
cv2.imshow("Keypoint Matches23456", vis)
cv2.imshow("Result23456", result)
#cv2.imwrite('C://Users//Administrator//Desktop//l23456.png',vis)
(result, vis) = stitcher.stitch([imageF, result], showMatches=True)
cv2.imshow("Keypoint Matches all", vis)
cv2.imshow("Result all", result)
#cv2.imwrite('C://Users//Administrator//Desktop//lall.png',vis)
cv2.waitKey(0)
'''
# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)

plt.subplot(211)
(r, g, b) = cv2.split(vis)
vis = cv2.merge([b, g, r])
plt.imshow(vis)  # added20161111

plt.subplot(212)
(r, g, b) = cv2.split(result)
result = cv2.merge([b, g, r])
plt.imshow(result)  # added20161111
plt.show()  # added20161111
result = cv2.merge([r, g, b])
cv2.imwrite('C://Users//Administrator//Desktop//p56.png',result)
cv2.waitKey(0)
'''