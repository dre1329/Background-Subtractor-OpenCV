import cv2
import numpy as np


ref_img = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=1000)

kp_ref, des_ref = orb.detectAndCompute(ref_img, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    fgmask = subtractor.apply(gray)


    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  
            x, y, w, h = cv2.boundingRect(cnt)
            roi = gray[y:y+h, x:x+w]

          
            kp_roi, des_roi = orb.detectAndCompute(roi, None)
            if des_roi is None:
                continue

            matches = bf.match(des_ref, des_roi)
            matches = sorted(matches, key=lambda x: x.distance)

        
            if len(matches) > 15:  
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, "Movement Detected", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    cv2.imshow("Live Feed", frame)
    cv2.imshow("Foreground Mask", fgmask)

    if cv2.waitKey(30) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
