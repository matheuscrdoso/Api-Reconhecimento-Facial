import face_recognition as fr

def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)
    if(len(rostos) > 0):
        return True, rostos

    return False, []

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    matheus1 = reconhece_face("./img/matheus1.jpg")
    if(matheus1[0]):
        rostos_conhecidos.append(matheus1[1][0])
        nomes_dos_rostos.append("Matheus Cardoso")

    #pedro = reconhece_face("./img/gab1.jpg")
    #if(pedro[0]):
     #   rostos_conhecidos.append(pedro[1][0])
      #  nomes_dos_rostos.append("Pedro Lucas")

    return rostos_conhecidos, nomes_dos_rostos