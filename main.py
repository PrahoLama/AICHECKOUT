import cv2
import numpy as np

# Incarcare YOLO
model_cfg = "yolo/yolov3.cfg"  # Fisierul de configurare al modelului YOLO
model_weights = "yolo/yolov3.weights"  # Fisierul cu greutatile antrenate pentru YOLO
coco_names = "yolo/coco.names"  # Fisierul care contine denumirile claselor (obiectelor)

# Incarca numele claselor
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Defineste produsele de detectat si preturile lor
target_products = {
    "banana": 0.50,  # Pretul unei banane
    "orange": 0.70,  # Pretul unei portocale
    "apple": 0.60,  # Pretul unui mar
    "biscuits": 1.20,  # Pretul unui pachet de biscuiti
    "cellphone": 300,  # Pretul unui telefon
    "suc": 10  # Pretul unei sticle
}

# Configureaza reteaua neuronala
net = cv2.dnn.readNet(model_weights, model_cfg)

# Optimizeaza utilizarea hardware-ului
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Obtinere numele straturilor retelei
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Pornire capturare video
cap = cv2.VideoCapture(0)

# Dictionar pentru a stoca numarul de produse detectate
product_counts = {product: 0 for product in target_products.keys()}

# Variabila pentru a verifica daca un produs a fost deja detectat
product_detected = False

while True:
    # Citeste un cadru din fluxul video
    _, frame = cap.read()
    height, width, channels = frame.shape  # Obtine dimensiunile imaginii

    # Pregateste imaginea pentru model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Variabile pentru a stoca datele detectiilor
    class_ids = []  # ID-urile claselor detectate
    confidences = []  # Nivelurile de incredere ale detectiilor
    boxes = []  # Coordonatele casetelor de delimitare (bounding boxes)

    # Proceseaza fiecare iesire
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Scorurile pentru fiecare clasa
            class_id = np.argmax(scores)  # Gaseste ID-ul clasei cu cel mai mare scor
            confidence = scores[class_id]  # Nivelul de incredere al detectiei
            if confidence > 0.5:  # Filtreaza detectiile cu o incredere mai mica de 50%
                # Calculeaza coordonatele casetei de delimitare
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Converteste coordonatele in format (x, y, latime, inaltime)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Adauga informatiile detectiei
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplica Non-Max Suppression pentru a elimina suprapunerile
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Verifica daca un produs a fost detectat
    if not product_detected:
        for i in indices.flatten():
            label = str(classes[class_ids[i]])  # Obtine eticheta clasei detectate
            if label in target_products:  # Verifica daca produsul este unul dintre cele dorite
                product_counts[label] += 1  # Incrementeaza contorul produsului
                product_detected = True  # Previne detectarea repetata pana la resetare

    # Creeaza un meniu de self-checkout pe feed-ul camerei
    overlay = frame.copy()  # Creeaza o copie a cadrului curent
    cv2.rectangle(overlay, (0, 0), (300, frame.shape[0]), (0, 0, 0), -1)  # Adauga un fundal negru pe partea stanga

    # Afiseaza numarul de produse si pretul total
    y_offset = 50
    cv2.putText(overlay, "Meniu Self-Checkout", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Filtreaza si afiseaza doar produsele detectate
    total_price = 0
    for product, count in product_counts.items():
        if count > 0:  # Afiseaza doar produsele detectate
            price = count * target_products[product]
            total_price += price
            text = f"{product.capitalize()}: {count} buc"
            cv2.putText(overlay, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30

    # Afiseaza pretul total
    total_price_text = f"Total: RON{total_price:.2f}"
    cv2.putText(overlay, total_price_text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Adauga un buton "Pay" pentru simularea platii
    button_x, button_y, button_width, button_height = 10, y_offset + 60, 200, 40
    cv2.rectangle(overlay, (button_x, button_y), (button_x + button_width, button_y + button_height), (0, 255, 0), -1)
    cv2.putText(overlay, "Pay", (button_x + 65, button_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Imbina overlay-ul cu cadrul camerei
    alpha = 0.7  # Factorul de transparenta
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Afiseaza feed-ul camerei cu overlay-ul aplicat
    cv2.imshow("Sistem Self-Checkout", frame)

    # Comenzi pentru utilizator: 'r' pentru resetare, 'q' pentru iesire
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):  # Reseteaza detectiile
        product_detected = False
        product_counts = {product: 0 for product in target_products.keys()}  # Optional: reseteaza contoarele
    elif key == ord("q"):  # Iesire
        break

# Elibereaza resursele
cap.release()
cv2.destroyAllWindows()
