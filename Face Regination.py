import cv2

#Yüz algoritmasını import etme.
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#0. indexdeki kamerayı yakalar.
camera=cv2.VideoCapture(0)

#Kameradan sürekli almak için verileri, programı sonsuz döngüye soruyoruz.
while True:

    #Cameradan sürekli olarak okuduğumuz verileri değişkenin içine atıp bunu dönüştürüyoruz
    ret,frame=camera.read()

    #Kameradan sürekli olarak okuduğumuz verileri griye dönüştürüp değişkene atıyoruz.
    graytone=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #gritone değişkenini 1.3 oranunda ölçeklendiriyoruz ve sonuncu parametrede videoda yüzün var olup olmadığını kaç kere teyid etsin onu belirliyoruz.
    faces=face_cascade.detectMultiScale(graytone,1.2,4)
    #Burada tespit edilen yüzün başlangıç noktası olarak bize x, y, yükseklik (height) ve genişlik (width) kordinatlarını döndürüyor.
    for(x,y,w,h) in faces:

        #Recangle metodu dikdörtgen çizmeye yarar. ilk parametre neye işlem yapılacak, 2. parametre başlangıç noktası, 3. parametre bitiş noktası, 4. parametre dikdörtgen rengi, 
        # 5. parametre ise dikdörtgenin kalınlığı
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)

    #Frame değişkenini bir pencerede göstermesini söylüyoruz. 1. Parametre pencere başlığı
    cv2.imshow("Face Recognition", frame)

    #'q' tuşuna bastığımızda while döngüsünden çıkmasını söylüyoruz.
    if cv2.waitKey(25) & 0xFF ==ord("q"):
        break

#Kamerayı kapatıyoruz.
camera.release()

#Tüm pencereleri kapatıyoruz.
cv2.destroyAllWindows()