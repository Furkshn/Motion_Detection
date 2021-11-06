import cv2



def motion_detect():

    st_back = None
    motion_list = [None,None]

    video = cv2.VideoCapture(0)

    while True:

        check, frame = video.read()
        motion = 0

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray,(21,21),0)

        if st_back is None:
            st_back = gray
            continue

        difference_frame = cv2.absdiff(st_back,gray)

        thresh_frame = cv2.threshold(difference_frame,30,255, cv2.THRESH_BINARY)[1]

        thresh_frame = cv2.dilate(thresh_frame,None,iterations=2)
        cnts,_ = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


        for contour in cnts:

            if cv2.contourArea(contour) < 10000:
                continue

            motion = 1

            (x,y,w,h) = cv2.boundingRect(contour)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        motion_list.append(motion)
        motion_list = motion_list[-2:]


        cv2.imshow("Gray Frame",gray)
        cv2.imshow("Difference Frame",difference_frame)
        cv2.imshow("Threshold Frame",thresh_frame)
        cv2.imshow("Color Frame",frame)

        key = cv2.waitKey(1)

        if key == ord("q"):

            break

    video.release()
    cv2.destroyAllWindows()


motion_detect()