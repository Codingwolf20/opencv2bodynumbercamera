import cv2

# take camera/ for camera videocapture o  /for file video address video
body_classifire=cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('hi.mp4')

# for filter body human

while cap.isOpened():
    ret,frame =cap.read()
    

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    bodies =body_classifire.detectMultiScale(gray,1.1,3)

    for(x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        human_number=len(bodies)
        cv2.putText(frame,f'human:{human_number}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2) 
    cv2.imshow("body",frame)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        break


cap.release()
cv2.destroyAllWindows()    

