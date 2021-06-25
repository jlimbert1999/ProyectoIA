from django.http import HttpResponse
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import keyboard
import numpy
import face_recognition
import h5py
import os
from django.shortcuts import render
import codecs



def ejemplo(request):
    return render(request, "Formularios/ejemplo.html")

def buscar(request):
    mensaje="nombre: %r" %request.GET["pido"]
    return HttpResponse(mensaje)


#===================================================
def saludo(request):
    return HttpResponse("primera vista")

def saludo1(request):
    return HttpResponse("hola trocha")


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def getSoloFrame(self):
        return self.frame
        

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ubicacion = face_recognition.face_locations(image)
        descriptor = face_recognition.face_encodings(image, ubicacion)

        for (y1, x1, y2, x2), descriptor in zip(ubicacion, descriptor):

            cv2.rectangle(image, (x2, y1), (x1, y2), (0, 0, 255), 2)
            cv2.rectangle(image, (x2, y2 - 35), (x1, y2), (0, 0, 255), cv2.FILLED)
        
        frame_fli=cv2.flip(image,1)
        _, jpeg = cv2.imencode('.jpg', frame_fli)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
            


def gen(camera):
    while True:
        frame = camera.get_frame()
        #caputar img
        # if keyboard.is_pressed('a'):
        #     img_name = "ProjectIA/img/foto1.jpg"
        #     cv2.imwrite(img_name,camera.getSoloFrame())
        #     print("Guardado!")
        #======================
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def livefe(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass

directorio = 'ProjectIA/img/'

def generar(request, nombre):
    nombre=request.GET["nombre"]
    prueba = h5py.File("rostros_id.h5", 'r')
    X = prueba["X"][:]#valores
    Y = prueba["y"][:]#identificador
    #cargar imagen a face_recognition
    foto = face_recognition.load_image_file("ProjectIA/img/foto1.jpg")
    descriptor_foto = face_recognition.face_encodings(foto)[0]
    A = descriptor_foto.reshape(-1, 1)
    res = numpy.concatenate([X, A], axis=1)
    identificaciones=Y.astype('U13')
    identificaciones=numpy.append(identificaciones, nombre)
    print(identificaciones)
    prueba.close()
    archivo=h5py.File("data.h5", "w")
    archivo.create_dataset("X", data=res)
    lista2h5=[n.encode("ascii", "ignore") for n in identificaciones]
    archivo.create_dataset("y", data=lista2h5)
    archivo.close()
    os.remove('ProjectIA/img/foto1.jpg')

    # try:
    #    

    # except ValueError:
    #     print("error al crear", ValueError)

    return HttpResponse("listoooo")





def reconocer(request):
    archivo = h5py.File("rostros_id.h5", "r")
    captura_video = cv2.VideoCapture(0)
    rosstros_imagenes = []
    rosstros_imagenes.append(archivo["X"][:][:, 0])
    rosstros_imagenes.append(archivo["X"][:][:, 1])

    ids = []
    ids.append(archivo["y"][:][0])
    ids.append(archivo["y"][:][1])

    while True:
        ret, frame = captura_video.read()
        rgb_frame = frame[:, :, ::-1]
        ubicacion = face_recognition.face_locations(rgb_frame)
        descriptor = face_recognition.face_encodings(rgb_frame, ubicacion)

        for (y1, x1, y2, x2), descriptor in zip(ubicacion, descriptor):
            matches = face_recognition.compare_faces(rosstros_imagenes, descriptor)
            nombre = "Desconocido"
            face_distancia = face_recognition.face_distance(rosstros_imagenes, descriptor)
            mejor_distancia = numpy.argmin(face_distancia)
            if matches[mejor_distancia]:
                nombre = ids[mejor_distancia]

            cv2.rectangle(frame, (x2, y1), (x1, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x2, y2 - 35), (x1, y2), (0, 0, 255), cv2.FILLED)

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.rectangle(frame, (x1, y1 - 35), (x2, y2), (0, 0, 255), cv2.INTER_AREA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(nombre), (x2 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("Reconocimiento", frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
        # cv2.waitKey(0)
    captura_video.release()
    cv2.destroydWindows()
