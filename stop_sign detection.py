import cv2

cap = cv2.VideoCapture(0)


class stopsigndetection():
    stop_cascade = cv2.CascadeClassifier("stop_sign.xml")
    def handle(self):

        global sensor_data
        stream_bytes = b' '
        stop_flag = False
        stop_sign_active = True

        try:
            # stream video frames one by one
            while True:

                    _,image=cap.read()
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


                    # lower half of the image
                    height, width = gray.shape
                    roi = gray[int(height / 2):height, :]

                    # object detection
                    v_param = self.detect(self.stop_cascade, gray, image)

                    # distance measurement
                    if v_param == 1:
                        print('stop')
                    else:
                        print('no stop')


                    cv2.imshow('image', image)
                    k = cv2.waitKey(5) & 0xFF
                    if k == 27:
                        break

                    # cv2.imshow('mlp_image', roi)
        finally:
            cv2.destroyAllWindows()
            cap.release()

    def detect(self, cascade_classifier, gray_image, image):

        # y camera coordinate of the target point 'P'
        v = 0

        # minimum value to proceed traffic light state validation
        threshold = 150

        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos + 5, y_pos + 5), (x_pos + width - 5, y_pos + height - 5), (255, 255, 255), 2)
            # print(x_pos+5, y_pos+5, x_pos+width-5, y_pos+height-5, width, height)

            # stop sign
            if width / height == 1:
                cv2.putText(image, 'STOP', (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                v=1

        return v


ob = stopsigndetection()
ob.handle()



# release the captured frame

