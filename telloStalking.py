from djitellopy import Tello
import cv2
import numpy as np

fly = 0 # 0 - stream only, 1 - fly

# connect and set initial velocities with speed
tello = Tello()
tello.connect()
tello.for_back_velocity = 0
tello.left_right_velocity = 0
tello.up_down_velocity = 0
tello.yaw_velocity = 0
tello.speed = 0

#initial imagge parameters
width = 640  
height = 480  
outVis = 100 #zone range where object is out of center


#turn on the stream
tello.streamon()


halfH = int(height/2)
halfW = int(width/2)


global imgContour
global direction #future decision where to move

#skip for activation function (as while loop updates the state)
def empty(a):
    pass

#initialize Trackbars for HSV and Area/Thresholds with default for color: BRIGHT ORANGE
cv2.namedWindow("HSV parameters")
cv2.resizeWindow("HSV parameters",640,240)
cv2.createTrackbar("Hue Min","HSV parameters",110,179,empty)
cv2.createTrackbar("Hue Max","HSV parameters",141, 179,empty)
cv2.createTrackbar("Saturation Min","HSV parameters",167,255,empty)
cv2.createTrackbar("Saturation Max","HSV parameters",255,255,empty)
cv2.createTrackbar("Value Min","HSV parameters",48,255,empty)
cv2.createTrackbar("Value Max","HSV parameters",231,255,empty)
cv2.namedWindow("Countour Parameters")
cv2.resizeWindow("Countour Parameters",640,240)
cv2.createTrackbar("T1","Countour Parameters",64,255,empty)
cv2.createTrackbar("T2","Countour Parameters",19,255,empty)
cv2.createTrackbar("Area","Countour Parameters",3009,30000,empty)


def getContours(img,imgContour):
    global direction
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        #if abs(area - areaMin) <= 2000:
        if area>=areaMin:
            cv2.drawContours(imgContour, contour, -1, (255, 0, 255), 7) #7 - width of the count
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cx = int(x + (w / 2))  # x coordinate of center
            cy = int(y + (h / 2))  # y coordinate of center

            if (cx < halfW - outVis):
                cv2.putText(imgContour, "LEFT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                cv2.rectangle(imgContour, (0,halfH-outVis), (halfW-outVis,halfH+outVis), (255, 0, 0), cv2.FILLED) 
                direction = 1
            elif (cx > halfW + outVis):
                cv2.putText(imgContour, "RIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                cv2.rectangle(imgContour, (halfW+outVis,halfH-outVis), (width,halfH+outVis), (255, 0, 0), cv2.FILLED)
                direction = 2
            elif (cy < halfH - outVis):
                cv2.putText(imgContour, "UP", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                cv2.rectangle(imgContour, (halfW-outVis,0), (halfW+outVis,halfH-outVis), (255, 0, 0), cv2.FILLED)
                direction = 3
            elif (cy > halfH + outVis):
                cv2.putText(imgContour, "DOWN", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                cv2.rectangle(imgContour, (halfW-outVis,halfH+outVis), (halfW+outVis,height), (255, 0, 0), cv2.FILLED)
                direction = 4
            else: direction=0

            cv2.line(imgContour, (halfW,halfH), (cx,cy),(0, 0, 255), 4)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv2.putText(imgContour, "#points: " + str(len(approx)), (x + w + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 2)
            cv2.putText(imgContour, "area: " + str(int(area)), (x + w + 15, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            direction = 0
        #future work for forward/back moving: 
        """elif area < areaMin and areaMin - area < 4000:
            direction = 5
            cv2.putText(imgContour, "FORWARD", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0), 4)
        elif area>areaMin:
            direction = 6
            cv2.putText(imgContour, "BACK", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0), 4)"""
        

while True:

    #read the current frame
    frame_read = tello.get_frame_read()
    img = cv2.resize(frame_read.frame, (width, height))
    imgContour = img.copy() #used in future for contour detection (in BGR)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR (default tello) to HSV

    hue_min = cv2.getTrackbarPos("Hue Min","HSV parameters")
    hue_max = cv2.getTrackbarPos("Hue Max", "HSV parameters")
    sat_min = cv2.getTrackbarPos("Saturation Min", "HSV parameters")
    sat_max = cv2.getTrackbarPos("Saturation Max", "HSV parameters")
    value_min = cv2.getTrackbarPos("Value Min", "HSV parameters")
    value_max = cv2.getTrackbarPos("Value Max", "HSV parameters")


    lowerBound = np.array([hue_min,sat_min,value_min])
    upperBound = np.array([hue_max,sat_max,value_max])

    #mask creation
    mask = cv2.inRange(imgHsv,lowerBound,upperBound)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    imgBlur = cv2.GaussianBlur(result, (7, 7), 1) #blur the masked image
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    t1 = cv2.getTrackbarPos("T1", "Countour Parameters")
    t2 = cv2.getTrackbarPos("T2", "Countour Parameters")

    #canny for contour detection
    imgCanny = cv2.Canny(imgGray, t1, t2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDil, imgContour)

    #transfer images to RGB and create a grid above the contour image
    imgLeft = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    imgCont = cv2.cvtColor(imgContour, cv2.COLOR_BGR2RGB) 
    cv2.line(imgCont, (halfW - outVis,0),(halfW - outVis, height),(255,255,0),3)
    cv2.line(imgCont,(halfW + outVis,0),(halfW + outVis, height),(255,255,0),3)
    cv2.line(imgCont, (0,halfH - outVis), (width ,halfH - outVis), (255, 255, 0), 3)
    cv2.line(imgCont, (0, halfH + outVis), (width, halfH + outVis), (255, 255, 0), 3)
    cv2.circle(imgCont, (halfW, halfH), 5, (0,0,255) ,5)
    
    #start the fly
    if fly == 1:
       tello.takeoff()
       fly = 0

    #perform movements according to decision made
    if dir == 1:
       tello.yaw_velocity = -50
    elif dir == 2:
       tello.yaw_velocity = 50
    elif dir == 3:
       tello.up_down_velocity= 50
    elif dir == 4:
       tello.up_down_velocity= -50   
    else:
       tello.left_right_velocity = 0 
       tello.for_back_velocity = 0
       tello.up_down_velocity = 0 
       tello.yaw_velocity = 0
    #future work for forward/back moving: 
    """elif dir == 5:
        tello.for_back_velocity = 20
    elif dir == 6:
        tello.for_back_velocity = -20;""" 
    
    if tello.send_rc_control:
       tello.send_rc_control(tello.left_right_velocity, tello.for_back_velocity, tello.up_down_velocity, tello.yaw_velocity)
    #print(dir)

    #show the images
    import cv2


    
    scale = 0.6 #here you can configure images for your screen size
    imgLeft_resized = cv2.resize(imgLeft, (0, 0), None, scale, scale)
    result_resized = cv2.resize(result, (0, 0), None, scale, scale)
    imgDil_resized = cv2.resize(imgDil, (0, 0), None, scale, scale)
    imgCont_resized = cv2.resize(imgCont, (0, 0), None, scale, scale)
    imgDil_resized = cv2.cvtColor(imgDil_resized, cv2.COLOR_GRAY2BGR)

    #stack images horizontally and then vertically
    top_row = np.hstack((imgLeft_resized, result_resized))
    bottom_row = np.hstack((imgDil_resized, imgCont_resized))
    stacked_image = np.vstack((top_row, bottom_row))

    #display the final image
    cv2.imshow('final', stacked_image)

    #landing: use 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break

cv2.destroyAllWindows()
