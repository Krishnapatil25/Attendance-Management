from flask import *
import datetime
import time
import sqlite3
import cv2,os
import numpy as np
import speech_recognition as sr
from pygame import mixer
import requests
from flask_session import sessions
import numpy



note = ""
cap = cv2.VideoCapture(0)
app = Flask(__name__)
app.secret_key="super secret key"


@app.route('/')
def login():
    return render_template("start.html")
@app.route('/init')
def init():
    return render_template("init.html")

@app.route('/admin')
def admin():
    return render_template("ad.html")
@app.route('/user')
def user():
    return render_template("userlogin.html")

@app.route('/register')
def register():
    return render_template("registration.html")
@app.route('/signup')
def signup():
    return render_template("mssg.html")
@app.route('/capture',methods=["POST"])
def capture():
    if request.method == "POST":
        name = request.form["fname"]
        mail = request.form['mail']
        pwd = request.form['pwd']
        conn = sqlite3.connect('candidates.db')
        conn.execute("INSERT INTO product(FULLNAME,USID,PASSWORD) VALUES(?,?,?)", (name, mail, pwd))
        conn.commit()
        conn.close()
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'
        sub_data = name
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
        (width, height) = (130, 100)
        face_cascade = cv2.CascadeClassifier(haar_file)
        webcam = cv2.VideoCapture(0)
        count = 1
        while count < 30:
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (path, count), face_resize)
            count += 1

            #cv2.imshow('OpenCV', im)
            #key = cv2.waitKey(10)
            #if key == 27:
                #break


        return render_template("registration.html",msg="Registered Successfully!",img1="static/symbol.png")
@app.route('/recognize')

def recognise():
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

            if prediction[1] < 500:
                cv2.putText(im, '% s - %.0f' %
                            (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                print(names[prediction[0]])
                stu_usn = names[prediction[0]]
                print(stu_usn)
                conn = sqlite3.connect('viva.db')
                cur = conn.execute("SELECT * FROM student WHERE usn=?", (stu_usn,))
                row = cur.fetchone()
                if row[9] == 'accept':

                    cur = conn.execute("SELECT * FROM student WHERE usn=?", (stu_usn,))
                    rows = cur.fetchall()
                    for row in rows:
                        usn = row[2]
                    return render_template("Reg.html", rows=rows, usn=stu_usn)
            else:
                msg = "Invalid User"
                cv2.putText(im, 'not recognized',
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                return render_template("login1.html")

        cv2.imshow('OpenCV', im)

        key = cv2.waitKey(10)
        if key == 27:
            break
        return render_template('login.html')
@app.route('/valid',methods=['POST'])
def admin_login():
    if request.method == 'POST':
        user = request.form['username']
        psw = request.form['psw']
        if (user == 'admin') & (psw == 'admin'):
            x = datetime.datetime.now()
            x= x.strftime("%x")
            conn = sqlite3.connect('candidates.db')
            cur = conn.execute("SELECT * FROM product")
            r = cur.fetchall()
            conn = sqlite3.connect('candidates.db')
            cur = conn.execute('SELECT * FROM Sheet1 WHERE STATUS=? and DATE=?', ('Present',x))
            r1 = cur.fetchall()
            a = len(r)-len(r1)
            print(a,len(r1))

            return render_template('adminpage.html',a=a,p=len(r1))
        else:
            return render_template('ad1.html')

@app.route('/list',methods=['POST'])
def list():
    if request.method=='POST':
        fname = request.form['fname']
        mail = request.form['mail']
        pwd = request.form['pwd']


        conn = sqlite3.connect('candidates.db')
        conn.execute("INSERT INTO product(FULLNAME,USID,PASSWORD) VALUES(?,?,?)",(fname,mail,pwd))
        conn.commit()
        conn.close()
        return render_template('registration.html',msg="added successfully")
@app.route('/view')
def view():
    conn=sqlite3.connect('candidates.db')
    cur=conn.execute('SELECT * FROM product')
    rows=cur.fetchall()
    return render_template('VIEW.html',rows=rows)
@app.route('/view4')
def view4():
    conn=sqlite3.connect('candidates.db')
    cur=conn.execute('SELECT * FROM Sheet1')
    rows=cur.fetchall()
    return render_template('view4.html',rows=rows)

@app.route('/delete/<a>')
def delete(a):
    conn=sqlite3.connect('candidates.db')
    conn.execute('DELETE FROM product WHERE id=?',(a,))
    conn.commit()
    return redirect(url_for('view'))
@app.route('/userlogin',methods=['POST'])
def student_login():
    if request.method=='POST':
        mail = request.form['mail']
        pwd = request.form['pwd']
        conn = sqlite3.connect("candidates.db")
        cur = conn.execute("SELECT * FROM product WHERE USID=? AND PASSWORD=?",(mail,pwd))
        row = cur.fetchone()
        if row != None:
            cur = conn.execute("SELECT * FROM Sheet1 WHERE FULLNAME=?",(row[1],))

            r1 = cur.fetchall()
            t = r1[-1]
            return render_template("userpage.html",rows=row,mail=mail,r1=r1,t=t)
        else:
            msg = "invalid ID or Password"
            return render_template('userlogin.html',msg=msg)
@app.route('/calendar')
def calendar():
    return render_template("ca.html")
@app.route('/about')
def about():
    return render_template("about.html")
@app.route('/recog')
def recog():
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
    #model = cv2.LBPHFaceRecognizer_create() #cv2.face.LBPHFaceRecognizer_create() cv2.face.createLBPHFaceRecognizer()
    model = cv2.face_LBPHFaceRecognizer.create()
    #model=cv2.face.createLBPHFaceRecognizer()
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
                conn = sqlite3.connect('candidates.db')
                cur = conn.execute("SELECT * FROM Sheet1 WHERE FULLNAME=? and DATE=?",(name,x))
                row = cur.fetchone()
                if row==None:
                    conn.execute("INSERT INTO Sheet1(FULLNAME,STATUS,DATE,TIME)VALUES(?,?,?,?)",(name,"Present",x,t))
                    conn.commit()
                    return render_template("userpage.html",msg="Attendance Recorded")
                else:
                    return render_template("userpage.html", msg="Limit Exceeded")


            else:
                x = datetime.datetime.now()
                x = x.strftime("%x")
                t = x.strftime("%X")
                cv2.imwrite("datasets/visitors/",str(t)+'.jpg')
                return render_template("userpage.html",msg="Not Recognised")


@app.route('/edit/<a>')
def edit(a):
    conn = sqlite3.connect('candidates.db')
    cur = conn.execute('SELECT * FROM product WHERE id=?',(a,))
    row=cur.fetchone()
    return render_template('edit.html',row=row)
@app.route('/update/<a>',methods=['POST'])
def update(a):
    if request.method=='POST':
        fname = request.form['fname']
        mail = request.form['mail']
        pwd = request.form['pwd']
        conn=sqlite3.connect('candidates.db')
        cur=conn.execute('UPDATE product SET FULLNAME=?,USID=?,PASSWORD=? WHERE id=?',(fname,mail,pwd,a))
        conn.commit()
        return redirect(url_for('view'))


@app.route('/temp')
def index():
    conn = sqlite3.connect("profiles.db")
    cur = conn.execute('SELECT *  FROM Sheet2 WHERE id=1 ')
    row = cur.fetchone()
    return render_template("VIEW3.html", row=row)

@app.route('/add/<t>/<h>')
def add(t, h):
    conn = sqlite3.connect('profiles.db')
    conn.execute('UPDATE Sheet2 SET temp=?,hum=? WHERE id=?', (t, h, '1'))
    conn.commit()
    return ("updated successfully!")

@app.route('/start')
def start():
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
                conn = sqlite3.connect('candidates.db')
                cur = conn.execute("SELECT * FROM Sheet1 WHERE FULLNAME=? and DATE=?", (name, x))
                row = cur.fetchone()
                if row == None:
                    conn.execute("INSERT INTO Sheet1(FULLNAME,STATUS,DATE,TIME)VALUES(?,?,?,?)",
                                 (name, "Present", x, t))
                    conn.commit()
                    return render_template("start.html", msg="Attendance Recorded")
                else:
                    return render_template("start.html", msg1="Limit Exceeded")


           # else:
               # x = datetime.datetime.now()
                #x = x.strftime("%x")
                #t = x.strftime("%X")
                #cv2.imwrite("datasets/visitors/", str(t) + '.jpg')
                #return render_template("userpage.html", msg="Not Recognised")

@app.route('/xl')
def xl():
    from xlsxwriter.workbook import Workbook
    filename1 = datetime.datetime.now().strftime("%Y%m%d")
    workbook = Workbook(filename1+'.xlsx')

    worksheet = workbook.add_worksheet()

    conn = sqlite3.connect('candidates.db')
    c = conn.cursor()
    c.execute("select * from Sheet1")
    mysel = c.execute("select * from Sheet1")
    for i, row in enumerate(mysel):
        for j, value in enumerate(row):
            worksheet.write(i, j, value)
    workbook.close()
    return render_template('mssg.html')

    '''
@app.route('/login1')
def login1():
    return render_template('login1.html')
@app.route('/admin')
def admin():
    return render_template('admin.html')
@app.route('/registration_staff')
def register_staff():
    return render_template("staff_register.html")
@app.route('/registration_student')
def register_student():
    return render_template("registration_student.html")
@app.route('/admin_dashboard')
def admin_dashboard():
    conn = sqlite3.connect("viva.db")![](datasets/Kanti/24.png)
    cur = conn.execute("SELECT COUNT(id) FROM staff")
    staff = cur.fetchone()
    cur = conn.execute("SELECT COUNT(id) FROM student")
    student = cur.fetchone()
    cur = conn.execute("SELECT * FROM status")
    num = cur.fetchone()
    return render_template("dashboard.html",staff=staff[0],student=student[0],num=num[0])
@app.route('/admin_login',methods=['POST'])
def admin_login():
    if request.method == 'POST':
        user = request.form['user']
        psw = request.form['psw']
        if (user == 'admin') & (psw == 'admin'):
            return render_template('admin.html')
            session['loggedin'] = True
            conn = sqlite3.connect("viva.db")
            cur = conn.execute("SELECT COUNT(id) FROM staff")
            staff = cur.fetchone()
            cur = conn.execute("SELECT COUNT(id) FROM student")
            student = cur.fetchone()
            cur = conn.execute("SELECT * FROM status")
            num = cur.fetchone()
            print(staff,student,num)

            return render_template("dashboard.html",staff=staff[0],student=student[0],num=num[0])
        else:
            return render_template('admin.html')
@app.route('/logout')
def logout():
    session.pop('loggedin',None)
    return render_template('login.html')
@app.route('/capture',methods=['POST'])
def capture():
    if request.method == 'POST':
        name = request.form['name']
        name = name.strip()
        usn = request.form['usn']
        usn = usn.strip()
        branch = request.form['branch']
        branch = branch.strip()
        clg = request.form['clg']
        clg = clg.strip()
        stu_adr = request.form['stu_adr']
        stu_adr = stu_adr.strip()
        mail = request.form['mail']
        mail = mail.strip()
        psw = request.form['psw']
        psw = psw.strip()

        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'
        sub_data = usn
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
        (width, height) = (130, 100)
        face_cascade = cv2.CascadeClassifier(haar_file)
        webcam = cv2.VideoCapture(0)
        count = 1
        while count < 30:
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (path, count), face_resize)
            count += 1

            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
            if key == 27:
                break
        img = 'static/checked.png'
        return render_template("registration_student.html",img=img,name=name,usn=usn,branch=branch,clg=clg,stu_adr=stu_adr,mail=mail,psw=psw)

@app.route('/record',methods=['POST'])
def record():
    if request.method == 'POST':
        name = request.form['name']
        name = name.strip()
        usn = request.form['usn']
        usn = usn.strip()
        branch = request.form['branch']
        branch = branch.strip()
        clg = request.form['clg']
        clg = clg.strip()
        stu_adr = request.form['stu_adr']
        stu_adr = stu_adr.strip()
        mail = request.form['mail']
        mail = mail.strip()
        psw = request.form['psw']
        psw = psw.strip()
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source2:

                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                with open("E://pythonProject/viva/audio/"+name+'.wav','wb') as f:
                    f.write(audio2.get_wav_data())
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                note = MyText
                MyText = "Voice id is " + MyText
                print("Did you say " + MyText)
            conn = sqlite3.connect("viva.db")
            conn.execute("INSERT INTO student(name,usn,branch,clg,stu_adr,voice) VALUES(?,?,?,?,?,?)",(name,usn,branch,clg,stu_adr,note))
            conn.commit()
            conn.close()
            return render_template("registration_student.html", MyText=MyText, name=name, usn=usn, branch=branch, clg=clg,
                                   stu_adr=stu_adr, mail=mail, psw=psw)


        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("try again")
        return render_template("registration_student.html", MyText='Try Again', name=name, usn=usn, branch=branch, clg=clg,
                               stu_adr=stu_adr, mail=mail, psw=psw)
@app.route('/signup_student',methods=['POST'])
def signup_student():
    if request.method == 'POST':
        name = request.form['name']
        name = name.strip()
        usn = request.form['usn']
        usn = usn.strip()
        branch = request.form['branch']
        branch = branch.strip()
        clg = request.form['clg']
        clg = clg.strip()
        stu_adr = request.form['stu_adr']
        stu_adr = stu_adr.strip()
        mail = request.form['mail']
        mail = mail.strip()
        psw = request.form['psw']
        psw = psw.strip()
        print(note)
        conn = sqlite3.connect("viva.db")
        conn.execute("UPDATE student SET mail=?,psw=? WHERE usn=?",(mail,psw,usn))
        conn.commit()
        conn.close()
        msg = "Registred Successfully"
        return render_template("registration_student.html",msg=msg)
@app.route('/recognise')
def recognise():
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
    #model = cv2.LBPHFaceRecognizer_create() #cv2.face.LBPHFaceRecognizer_create() cv2.face.createLBPHFaceRecognizer()
    model = cv2.face_LBPHFaceRecognizer.create()
    #model=cv2.face.createLBPHFaceRecognizer()
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

            if prediction[1] < 500:
                cv2.putText(im, '% s - %.0f' %
                            (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                print(names[prediction[0]])
                stu_usn = names[prediction[0]]
                conn = sqlite3.connect('viva.db')
                cur = conn.execute("SELECT * FROM student WHERE usn=?", (stu_usn,))
                row = cur.fetchone()
                if row[9] =='accept':

                    cur = conn.execute("SELECT * FROM student WHERE usn=?",(stu_usn,))
                    rows = cur.fetchall()
                    for row in rows:
                        usn=row[2]
                    return render_template("student_page.html",rows=rows,usn=stu_usn)
            else:
                msg = "Invalid User"
                cv2.putText(im, 'not recognized',
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                return render_template("login1.html")

        cv2.imshow('OpenCV', im)

        key = cv2.waitKey(10)
        if key == 27:
            break
        return render_template('login1.html')
@app.route('/student_page/<usn>')
def student_page(usn):
    usn=usn
    conn = sqlite3.connect('viva.db')
    cur = conn.execute("SELECT * FROM student WHERE usn=?", (usn,))
    rows = cur.fetchall()
    return render_template("student_page.html",rows=rows,usn=usn)
@app.route('/voice_id')
def voice_id():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source2:

            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio2 = r.listen(source2)
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print("Did you say " + MyText)
            conn = sqlite3.connect("viva.db")
            cur = conn.execute("SELECT * FROM student WHERE voice=?",(MyText,))
            rows = cur.fetchall()
            usn=''
            for row in rows:
                usn=row[2]
            return render_template("student_page.html",rows=rows,usn=usn)


    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("try again")
        return render_template("login1.html",rows=rows,usn=usn)
@app.route('/student_login',methods=['POST'])
def student_login():
    if request.method=='POST':
        user = request.form['user']
        psw = request.form['psw']
        conn = sqlite3.connect("viva.db")
        cur = conn.execute("SELECT * FROM student WHERE mail=? AND psw=? AND en=?",(user,psw,'accept'))
        account = cur.fetchone()
        if account:
            session['loggedin'] = True
            cur = conn.execute("SELECT * FROM student WHERE mail=?",(user,))
            rows= cur.fetchall()
            for row in rows:
                usn=row[2]
            print(usn)
            return render_template("student_page.html",rows=rows,usn=usn)
        else:
            msg = "invalid ID or Password"
            return render_template('login1.html',msg=msg)

@app.route('/staff_capture',methods=['POST'])
def staff_capture():
    if request.method == 'POST':
        name = request.form['name']
        name1 = name.strip()
        sub = request.form['sub']
        sub = sub.strip()
        branch = request.form['branch']
        branch = branch.strip()
        mail = request.form['mail']
        mail= mail.strip()
        psw = request.form['psw']
        psw = psw.strip()
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'
        sub_data = name
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
        (width, height) = (130, 100)
        face_cascade = cv2.CascadeClassifier(haar_file)
        webcam = cv2.VideoCapture(0)
        count = 1
        while count < 30:
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (path, count), face_resize)
            count += 1

            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
            if key == 27:
                break
        img = 'static/checked.png'

    return render_template("staff_register.html",sub=sub, img=img, name=name1, branch=branch, mail=mail, psw=psw)
@app.route('/staff_register',methods=['POST'])
def staff_register():
    if request.method == 'POST':
        name = request.form['name']
        name1 = name.strip()
        sub = request.form['sub']
        sub = sub.strip()
        branch = request.form['branch']
        branch = branch.strip()
        mail = request.form['mail']
        mail= mail.strip()
        psw = request.form['psw']
        psw = psw.strip()
        conn = sqlite3.connect("viva.db")
        conn.execute("INSERT INTO staff(name,subject,branch,mail,psw) VALUES(?,?,?,?,?)",(name1,sub,branch,mail,psw))
        conn.commit()
        conn.close()
        msg= "Registered Successfully"
        return render_template("staff_register.html",msg=msg)
@app.route('/staff_recognise')
def staff_recognise():
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
    model = cv2.face.LBPHFaceRecognizer_create()

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

            if prediction[1] < 100:
                cv2.putText(im, '% s - %.0f' %
                            (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                print(names[prediction[0]])
                staff_name = names[prediction[0]]

                conn = sqlite3.connect('viva.db')
                cur = conn.execute("SELECT * FROM staff WHERE name=?",(staff_name,))
                rows = cur.fetchall()
                for row in rows:
                    sub=row[2]
                if rows == []:
                    return render_template("login.html")

                return render_template("staff_page.html",rows=rows,sub=sub)
            else:
                print("not")
                cv2.putText(im, 'not recognized',
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                return render_template("login.html")

        cv2.imshow('OpenCV', im)

        key = cv2.waitKey(10)
        if key == 27:
            break
@app.route('/staff_login',methods=['POST'])
def staff_login():
    if request.method == 'POST':
        user = request.form['user']
        print(user)
        psw = request.form['psw']
        conn = sqlite3.connect("viva.db")
        cur = conn.execute("SELECT * FROM staff WHERE mail=? AND psw=? AND en=?", (user, psw,'accept'))
        account = cur.fetchone()
        if account:
            session['loggedin'] = True
            cur = conn.execute("SELECT * FROM staff WHERE mail=?", (user,))
            rows = cur.fetchall()
            for row in rows:
                sub= row[2]
            return render_template("staff_page.html", rows=rows,sub=sub)
        else:
            msg= "Invalid ID or Password"
            return render_template('login.html', msg=msg)
@app.route('/staff_exam/<sub>',methods=['POST'])
def staf_exam(sub):

    if request.method=='POST':
        sub = sub
        conn = sqlite3.connect("viva.db")
        cur = conn.execute("SELECT * FROM exams WHERE sub=?",(sub,))
        rows = cur.fetchall()
        print(rows)
        msg = 'Exam is scheduled'
        return render_template("schedule_exam.html",rows=rows,sub=sub)
@app.route('/staff_exam/home/<sub>')
def staff_page(sub):
    sub=sub
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM staff WHERE subject=?",(sub,))
    rows = cur.fetchall()
    print(rows)
    return render_template("staff_page.html",rows=rows,sub=sub)
@app.route('/admin_schedule_exam')
def admin_schedule():
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM exams")
    rows= cur.fetchall()
    cur = conn.execute("SELECT * FROM staff")
    st = cur.fetchall()
    return render_template("admin_schedule.html",rows=rows,st=st)
@app.route('/schedule_exam',methods=['POST'])
def schedule_exam():
    if request.method == 'POST':
        sub = request.form['sub']
        usn = request.form['usn']
        staff = request.form['staff']
        date1 = request.form['date1']
        time1 = request.form['time1']
        duration = request.form['duration']
        conn = sqlite3.connect("viva.db")
        conn.execute("INSERT INTO exams(sub,usn,date1,time1,duration,staff) VALUES(?,?,?,?,?,?)",(sub,usn,date1,time1,duration,staff))
        conn.commit()
        cur=conn.execute("SELECT * FROM exams")
        rows = cur.fetchall()


        msg = "Exam is Scheduled at Below time "
        return render_template("admin_schedule.html",msg=msg,rows=rows)
@app.route('/del_exam/<id>',methods=['POST'])
def del_exam(id):
    id=id
    if request.method=='POST':
        conn=sqlite3.connect("viva.db")
        conn.execute("DELETE FROM exams WHERE id=?",(id,))
        conn.commit()
        cur = conn.execute("SELECT * FROM exams")
        rows= cur.fetchall()
        msg = "Exam is Scheduled at Below time "
        return render_template("admin_schedule.html",msg=msg,rows=rows)
@app.route('/students_list')
def student_list():
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM student")
    rows = cur.fetchall()
    return render_template("student_list.html",rows=rows)
def up_staf(a):
    if request.method=='POST':
        usn=str(a)
        st=request.form['en']
        conn= sqlite3.connect("viva.db")
        conn.execute("update student SET en=? WHERE usn=?",(st,usn))
        conn.commit()
        cur=conn.execute("SELECT * FROM student")
        rows = cur.fetchall()
        return render_template("student_list.html",rows=rows)
@app.route('/up_staf/<a>', methods=['POST'])
def up_stf(a):
    if request.method=='POST':
        usn=str(a)
        st=request.form['en']
        conn= sqlite3.connect("viva.db")
        conn.execute("update staff SET en=? WHERE mail=?",(st,usn))
        conn.commit()
        cur=conn.execute("SELECT * FROM staff")
        rows = cur.fetchall()
        return render_template("staff_list.html",rows=rows)
@app.route('/up_stu/<a>', methods=['POST'])
def up_stu(a):
    if request.method=='POST':
        usn=str(a)
        st=request.form['en']
        conn= sqlite3.connect("viva.db")
        conn.execute("update student SET en=? WHERE usn=?",(st,usn))
        conn.commit()
        cur=conn.execute("SELECT * FROM student")
        rows = cur.fetchall()
        return render_template("student_list.html",rows=rows)
@app.route('/staff_list')
def staff_list():
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM staff")
    rows = cur.fetchall()
    return render_template("staff_list.html",rows=rows)
@app.route('/student_exam/<usn>')
def student_exam(usn):
    a= str.strip(usn)
    print(a)
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM exams WHERE usn=?",(a,))
    rows = cur.fetchall()
    return render_template("student_exam.html",rows=rows,usn=usn)

@app.route('/start_exam/<usn>/<int:id>')
def start_exam(usn,id):
    a=id
    usn = str.strip(usn)
    print(a,usn)
    tm1 = datetime.now()
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM exams WHERE id=?",(a,))
    row = cur.fetchone()
    tm2 = row[3]+" "+row[4]
    sub = row[1]
    dt = datetime.fromisoformat(tm2)
    cur = conn.execute("SELECT * FROM exams WHERE id=?", (a,))
    rows = cur.fetchall()
    if tm1 >= dt:
        return render_template("exam.html",rows=rows,usn=usn,sub=sub)
    else:
        msg = "Exam Yet To Start"
    return render_template("student_exam.html",msg=msg,rows=rows,usn=usn)
@app.route('/staff_exam/<sub>')
def stof_exam(sub):
    a=sub
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM exams WHERE sub=?",(a,))
    rows=cur.fetchall()
    return render_template("schedule_exam.html",rows=rows,sub=sub)
@app.route('/start_staff_exam/<sub>',methods=['POST'])
@app.route('/staff_exam/start_staff_exam/<sub>',methods=['POST'])
def staff_exam(sub):
    a = sub
    print(a)
    if request.method =='POST':

        usn = request.form['usn']
        tm1 = datetime.now()
        conn = sqlite3.connect("viva.db")
        cur = conn.execute("SELECT * FROM exams WHERE sub=?", (a,))
        row = cur.fetchone()
        tm2 = row[3] + " " + row[4]
        dt = datetime.fromisoformat(tm2)
        cur = conn.execute("SELECT * FROM exams WHERE id=?", (a,))
        rows = cur.fetchall()
        if tm1 >= dt:
            return render_template("staff_exam.html", rows=rows,sub=sub,usn=usn)
        else:
            msg = "Exam Yet To Start"
        return render_template("schedule_exam.html", msg=msg, rows=rows,sub=sub)
@app.route('/voice_to_staff/<usn>/<sub>')
def voice_to_staff(usn,sub):
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    cv2.imwrite('static/student.png', frame)
    cv2.destroyAllWindows()
    a = str.strip(usn)
    sub =str.strip(sub)
    r = sr.Recognizer()
    path = "E://pythonProject/viva/audio/answers/"
    print(path)
    name = 1
    try:
        with sr.Microphone() as source2:

            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio2 = r.listen(source2)
            with open(path+"1"+'.wav','wb') as f:
                f.write(audio2.get_wav_data())
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            note = MyText
            MyText = "Voice answer is " + MyText
            print("Did you say " + MyText)
            msg = 'answered!'
            conn= sqlite3.connect("viva.db")
            conn.execute("INSERT INTO iot_a(usn,sub,answer) VALUES(?,?,?)",(a,sub,note))
            conn.commit()
            return render_template("exam.html",msg=msg,usn=a,sub=sub)
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("try again")
        msg = 'try again!'
        return render_template("exam.html",msg=msg,usn=a,sub=sub)
@app.route('/stu_refresh/<usn>/<sub>')
def stu_refresh(usn,sub):

    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM iot_q ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    mixer.init()
    mixer.music.load("audio/questions/1.wav")
    mixer.music.play()
    return render_template("exam.html",row=row[3],usn=usn,sub=sub)
@app.route('/staff_refresh/<sub>/<usn>')
def staff_refresh(sub,usn):
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM iot_a ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    mixer.init()
    mixer.music.load("audio/answers/1.wav")
    mixer.music.play()
    img='/static/student.png'
    return render_template("staff_exam.html",row=row[3],sub=sub,usn=usn,img=img)
@app.route('/voice_to_student/<sub>/<usn>')
def voice_to_student(sub,usn):
    sub=sub
    usn=usn
    r = sr.Recognizer()
    name = 1
    try:
        with sr.Microphone() as source2:

            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio2 = r.listen(source2)
            with open("E://pythonProject/viva/audio/questions/"+str(name)+'.wav','wb') as f:
                f.write(audio2.get_wav_data())
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            note = MyText
            MyText = "Voice answer is " + MyText
            print("Did you say " + MyText)
            msg = 'Successfull!'
            conn= sqlite3.connect("viva.db")
            conn.execute("INSERT INTO iot_q(usn,sub,question) VALUES(?,?,?)",(usn,sub,note))
            conn.commit()
            return render_template("staff_exam.html",msg=msg,usn=usn,sub=sub)
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("try again")
        msg = 'try again!'
        return render_template("staff_exam.html",msg=msg)
@app.route('/end_exam/<sub>/<usn>')
def end_exam(sub,usn):
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM status")
    row = cur.fetchone()
    num = row[0] + 1
    conn.execute("UPDATE status SET num=?",(num,))
    conn.commit()

    return render_template("staff_page.html")
@app.route('/que_ans/<sub>')
def que_ans(sub):
    return render_template('que_ans.html',sub=sub)
@app.route('/staff_que/<sub>',methods=['POST'])
def staff_que(sub):
    a=sub
    if request.method=='POST':
        usn=request.form['usn']
        conn = sqlite3.connect("viva.db")
        cur = conn.execute("SELECT * FROM iot_q WHERE usn=? AND sub=?",(usn,a))
        rows1 = cur.fetchall()

        cur=conn.execute("SELECT * FROM iot_a WHERE usn=? AND sub=?",(usn,a))
        rows2 = cur.fetchall()
        cur=conn.execute("SELECT * FROM remarks WHERE usn=? AND sub=?",(usn,a))
        rows3 = cur.fetchall()

        return render_template("que_ans.html",sub=sub,rows1=rows1,rows2=rows2,usn=usn,rows3=rows3)

@app.route('/marks_staff/<sub>')
def marks_staff(sub):
    sub=sub
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM student")
    rows=cur.fetchall()
    return render_template("staff_marks.html",sub=sub,rows=rows)

@app.route('/upload_marks/<sub>',methods=['POST'])
def upload_marks(sub):
    sub=sub
    if request.method=='POST':
        id = request.form['id']
        usn = request.form['usn']
        marks = request.form['marks']
        a = int(id)

        print(a,marks,usn)
        i=0
        conn = sqlite3.connect("viva.db")
        conn=sqlite3.connect("viva.db")
        conn.execute("UPDATE marks SET marks=? WHERE id=?",(marks,id))
        conn.commit()
        i = i+1
        cur = conn.execute("SELECT * FROM marks")
        rows1 =cur.fetchall()
        return render_template("staff_marks.html",rows1=rows1,usn=usn,sub=sub)

@app.route('/student_marks/<usn>')
def student_marks(usn):
    usn = usn
    conn = sqlite3.connect("viva.db")
    cur = conn.execute("SELECT * FROM marks WHERE usn=?",(usn,))
    rows = cur.fetchall()
    return render_template("student_marks.html",rows=rows,usn=usn)

@app.route('/remark/<sub>/<usn>',methods=['POST'])
def remark(sub,usn):
    if request.method=='POST':
        rmk=request.form['remark']
        conn = sqlite3.connect('viva.db')
        conn.execute("INSERT INTO remarks(usn,sub,remarks) VALUES(?,?,?)",(usn,sub,rmk))
        conn.commit()
        conn.close()
        return render_template("staff_exam.html",us=usn,sub=sub)
'''
if __name__ == '__main__':
    app.run()


