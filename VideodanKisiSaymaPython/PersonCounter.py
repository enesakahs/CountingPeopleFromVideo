#Bu iki kütüphanenin kurulu olması lazım. Eğer sizde kurulu değilse alttaki iki hücreyi yorumdan çıkarıp çalıştırıp kurabilirsiniz.
#pip install tf-centernet
#pip install filterpy==1.4.5

import centernet
# burada sort.py dosyasından alıyor bu modülü o yüzden bu dosyayı indirmeyi unutmayın
from Sort import Sort
import numpy as np
import cv2
# Burada kullmacağımız object detection modelini tanımlıyoruz. Ağırlığı yüklerken None diyoruz.
# Bu sayede orijinal ağırlıklar kaynağından indirilecek.
obj = centernet.ObjectDetection(num_classes=80)
obj.load_weights(weights_path=None)
# Burada kullanacağımız takip algoritmasını tanımlıyoruz.
sort = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
font = cv2.FONT_HERSHEY_COMPLEX
# burada videonun adını tanımlıyoruz.
kamera= cv2.VideoCapture("video.mp4")
# Burada alanı belirliyoruz. Ben burada bunu dikdörtgen şeklinde ayarladım.
# Siz isterseniz paralel kenar veya altıgen gibi şekiller olabilir.
# burada şeklin köşe koordinatlarını giriyoruz.
# Burada önemli nokta köşeler sırayla gitmeli.
# Yani her bir köşe bir sonraki köşenin komşusu olmalı. Karşılıklı köşeler yan yana olmamalı.
# Burada köşeler sırayla sol üst köşe, sağ üst köşe, sağ alt köşe ve sol alt köşe şeklinde.
region=np.array([(270,180),(500,180),(500,320),(270,320)])
region = region.reshape((-1,1,2))
# Eğer bir kişi o bölgeye girmişse o kişinin id'sini bu kümenin içine ekliyoruz.
# küme kullanma sebebimiz, kümede aynı elemandan sadece bir tane olabilir.
toplam_id=set()
while True:


    ret, frame = kamera.read()
    # burada görünütünün boyutunu küçültüyoruz. Bu opsiyonel
    frame=cv2.resize(frame,None,fx=0.4,fy=0.4) 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # RGB görüntüyü modele verdik.
    boxes, classes, scores = obj.predict(img)
    detections_in_frame = len(boxes)
    # insanların etiketlerdeki indeskleri 0 olduğu için sadece nesne 0 mı yani insan  mı diye kontrol ediyoruz.
    if detections_in_frame:
        idxs = np.where(classes == 0)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]
        classes = classes[idxs]
    else:
        boxes = np.empty((0, 5))

    dets = np.hstack((boxes, scores[:,np.newaxis]))
    # Tespit sonuçlarını düzenleyip takip sistemine veriyoruz.
    res = sort.update(dets)
    # Bu nesneleri içine alan dikdörtgenin koordinatları
    boxes_track = res[:,:-1]
    # Bu her bir insana ait farklı id'lerin yer aldığı bir liste
    boces_ids = res[:,-1].astype(int)
    
    # bölgenin şeklini çizidiriyoruz.
    cv2.polylines(frame,[region],True,(0,0,255),4)
    # Burada tüm nesnelere ait bilgiler 3 farklı listede.
    # Her bir nesneye ait bilgileri almak için for döngüsü kullanıyoruz.
    for score,(xmin,ymin,xmax,ymax),ids in zip(scores, boxes_track, boces_ids):
        # Eğer nesnenin tespit oranı 0.2 den yüksekse takip işlemi uygulayacağız
        if score < 0.2:
            continue
        # burada koordinatlar integer tipine çeviriyoruz.
        ymin,xmin,ymax,xmax=int(ymin),int(xmin),int(ymax),int(xmax)
        # insanın orta noktasının koordinatlarını buluyoruz.
        cx=int((xmin+xmax)/2)
        cy=int((ymin+ymax)/2)
        # burada insanın orta noktası belirledğimiz bölgenin içinde mi diye kontrol ediyoruz.
        
        inside_region=cv2.pointPolygonTest(region,(cx,cy),False)
        # eğer bir insan buraya girmişse inside_region 0'dan büyük olacaktır.
        # böylece koşul doğru olur ve koşullu ifadenin içindeki işlem yapılır.
        if inside_region>0:
            # Bölgeye giren kişinin id'sini bu kümenin ekliyoruz
            toplam_id.add(ids)
        # kümenin uzunluğu ne kadar ise toplam o kadar kişi geçmiştir.
        toplam='Toplam:'+str(len(toplam_id))
        # burada toplam geçen kişi sayısı ekranın sol üst köşesinde olacak
        # görünüm güzel olsun diye beyaza boyadım o alanı
        frame[0:60,0:270]=(255,255,255)
        
        cv2.putText(frame,toplam,(0, 40), font, 1.5, (128,0,128), 2,)
        # Burada kişilerin orta noktasını mavi bir daire ile gösteriyoruz.
        cv2.circle(frame,(cx,cy),5,(255,0,0),-1)
        #frame = cv2.rectangle(frame,(xmin, ymax),(xmax, ymin),(0,255,0),2)      
        #cv2.putText(frame,str(ids),(xmin, ymin+30), font, 1, (255,0,0), 2,)

    cv2.imshow("kamera",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
kamera.release()
cv2.destroyAllWindows()