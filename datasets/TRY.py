@app.route('/recog')
def recog():
    import RPi.GPIO as GPIO
    import time
    import sys
    import I2C_LCD_driver
    import board
    import adafruit_mlx90614
    GPIO.setmode(GPIO.BCM)
    i2c = board.I2C()
    mylcd = I2C_LCD_driver.lcd()
    mlx = adafruit_mlx90614.MLX90614(i2c)
    mylcd.lcd_display_string('mlx', 2, 1)
    mylcd = I2C_LCD_driver.lcd()

    TRIG = 17
    ECHO = 27
    i = 0

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    GPIO.output(TRIG, False)
    print("Calibrating.....")
    time.sleep(1)

    print("Place the object......")

    try:
        while True:
            GPIO.output(TRIG, True)
            time.sleep(0.000001)
            GPIO.output(TRIG, False)

            while GPIO.input(ECHO) == 0:
                pulse_start = time.time()

            while GPIO.input(ECHO) == 1:
                pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start

            distance = pulse_duration * 17150

            distance = round(distance + 0.15, 2)
            print("distance:", distance, "cm")
            if distance < 6:

                # mylcd.lcd_display_string('AmbTemp:{0:0.1f} C '.format(mlx.ambient_temperature),1,2)
                mylcd.lcd_display_string('ObjTemp:{0:0.1f} C'.format(mlx.object_temperature), 1, 2)
                i = 1
                time.sleep(2)
                mylcd.lcd_clear()
                size = 4
                haar_file = 'haarcascade_frontalface_default.xml'
                datasets = 'datasets'

                # Part 1: Create fisherRecognizer
                print('Recognizing Face Please Be in sufficient Lights...')

                # Create a list of images and a list of corresponding names
                (images, lables, names, id) = ([], [], {}, 0)
                for (subdirs, dirs, files) in os.walk(datasets):
                    for subdir in dirs:
                        names[id] = subdir
                        subjectpath = os.path.join(datasets, subdir)
                        for filename in os.listdir(subjectpath):
                            path = subjectpath + '/' + filename
                            lable = id
                            images.append(cv2.imread(path, 0))
                            lables.append(int(lable))
                        id += 1
                (width, height) = (130, 100)

                # Create a Numpy array from the two lists above
                (images, lables) = [numpy.array(lis) for lis in [images, lables]]

                # OpenCV trains a model from the images
                # NOTE FOR OpenCV2: remove '.face'
                # model = cv2.LBPHFaceRecognizer_create() #cv2.face.LBPHFaceRecognizer_create() cv2.face.createLBPHFaceRecognizer()
                model = cv2.face_LBPHFaceRecognizer.create()
                # model=cv2.face.createLBPHFaceRecognizer()
                model.train(images, lables)

                # Part 2: Use fisherRecognizer on camera stream
                face_cascade = cv2.CascadeClassifier(haar_file)
                webcam = cv2.VideoCapture(0)
                while True:
                    (_, im) = webcam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        face = gray[y:y + h, x:x + w]
                        face_resize = cv2.resize(face, (width, height))
                        # Try to recognize the face
                        prediction = model.predict(face_resize)
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

                        if prediction[1] < 300:
                            cv2.putText(im, '% s - %.0f' %
                                        (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                            print(names[prediction[0]])

                            name = names[prediction[0]]
                            x1 = datetime.datetime.now()
                            x = x1.strftime("%x")
                            t = x1.strftime("%X")
                            temp=mlx.object_temperature
                            conn = sqlite3.connect('candidates.db')
                            cur = conn.execute("SELECT * FROM Sheet1 WHERE FULLNAME=? and DATE=?", (name, x))
                            row = cur.fetchone()
                            if row == None:
                                conn.execute("INSERT INTO Sheet1(FULLNAME,STATUS,DATE,TIME,TEMP)VALUES(?,?,?,?,?)",
                                             (name, "Present", x, t,temp))
                                conn.commit()
                                return render_template("userpage.html", msg="Attendance Recorded")
                            else:
                                return render_template("userpage.html", msg="Limit Exceeded")


                        else:
                            x = datetime.datetime.now()
                            x = x.strftime("%x")
                            t = x.strftime("%X")
                            cv2.imwrite("datasets/visitors/", str(t) + '.jpg')
                            return render_template("userpage.html", msg="Not Recognised")
            else:
                mylcd.lcd_display_string('PLS WEAR MASK', 1, 2)
                mylcd.lcd_display_string('PLACE YOUR HAND', 2, 1)
                i = 1
                time.sleep(2)
                mylcd.lcd_clear()







    except KeyboardInterrupt:
        GPIO.cleanup()


