import numpy as np
import cv2
from skimage.measure import compare_ssim
# import time

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.125
default_scale = 35
default_score_threshold = 0.8
img2_name = "12.jpg"


def get_settings_from_ini(filename:str)->dict:
    """
    gets sample, scale and threshold from settings file
    :param filename: name of settings file
    :return: dict with settings
    """
    settings = dict()
    try:
        f = open(filename)
        for line in f:
            settings[line.split(":")[0].strip()] = line.split(":")[1].strip()
        if "Sample" not in settings.keys():
            print("No sample file in settings")
            return {}
        if "Scale" not in settings.keys():
            print("Scale setting  not found, default used (%i pxs)" % default_scale)
            settings["Scale"] = default_scale
        if "Threshold" not in settings.keys():
            print("Threshold setting not found, default used (%i)" % default_score_threshold)
            settings["Threshold"] = default_score_threshold
        return settings
    except FileNotFoundError:
        return settings


def alignImages(im1, im2, gray=False):
    """
    aligns im1  to im2 using key features
    :param im1: image to align
    :param im2: sample image
    :param gray: are images already gray or not
    :return: aligned image and homography
    """
    try:
        if not gray:
            # Convert images to grayscale
            im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        else:
            im1Gray = im1
            im2Gray = im2

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        if gray:
            height, width = im2.shape
        else:
            height, width, channel = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))

        return im1Reg, h
    except Exception:
        e = sys.exc_info()[1]
        print(e.args[0])
        return None, None

def main():

    # get settings
    settings = get_settings_from_ini("settings.ini")
    if not settings:
        print("Can not continue without settings")
        return

    # get sample image
    img1 = cv2.imread(settings['Sample'])
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # edgeA = cv2.Canny(grayA, 100, 400)
    blurA = cv2.GaussianBlur(grayA, (5, 5), 0)
    cv2.imshow("Sample image", img1)
    scale = int(settings['Scale'])
    score_threshold = float(settings['Threshold'])
    # open camera
    #cap = cv2.VideoCapture(0)
    #if not cap.isOpened():
    #cap.open(0)


    while(True):
        # Capture frame-by-frame
        key = cv2.waitKey(0)
        if key == 32:

            #ret, img2 = cap.read()
            img2 = cv2.imread(img2_name)
            cv2.imshow("Camera image", img2)
            grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            blurB = cv2.GaussianBlur(grayB, (5, 5), 0)
            #edgeB = cv2.Canny(grayB, 100, 400)
            blurB, h = alignImages(blurB, blurA, True)
            img2, h = alignImages(img2, img1)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(blurA)
            #edgeclA = cv2.Canny(cl1, 100, 400)
            cl2 = clahe.apply(blurB)
            #edgeclB = cv2.Canny(cl2, 100, 400)

            m, n = blurA.shape
            m = m // scale
            n = n // scale
            for i in range(m):
                for j in range(n):
                    partA = cl1[i*scale:i*scale+scale-1, j*scale:j*scale+scale-1]
                    partB = cl2[i*scale:i*scale+scale-1, j*scale:j*scale+scale-1]

                    (score, diff) = compare_ssim(partA, partB, full=True, multichannel=True)
                    if score < score_threshold:
                        cv2.rectangle(img1, (j*scale, i*scale), (j*scale+scale-1, i*scale+scale-1), (255, 0, 0), 2)
                        cv2.rectangle(img2, (j * scale, i * scale), (j * scale + scale-1, i * scale + scale-1), (255, 0, 0), 2)

                #try:
                #    for k in range(33):
                #        for l in range(33):
                #            if diff[k, l] < 0.02:
                #                img2[i * scale + k, j * scale + l] = [255, 0, 0]
                #except:
                #    pass


            cv2.imshow("Compare result", np.hstack([img1, img2]))
        else:
            if key == 27:
               return

if __name__ ==  '__main__':
    main()