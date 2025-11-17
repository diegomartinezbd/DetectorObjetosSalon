// =============================
// Cargar modelo TFLite
// =============================
let model;
const TFLITE_PATH = "best.tflite";

// Clases
const labels = ["CPU","Mesa","Mouse","Pantalla","Silla","Teclado"];

// Cargar el modelo
async function loadModel() {
    model = await tflite.loadTFLiteModel(TFLITE_PATH);
    console.log("Modelo TFLite cargado.");
}

loadModel();

// =============================
// Manejar carga de imagen
// =============================
document.getElementById("fileInput").addEventListener("change", function (event) {

    const file = event.target.files[0];
    if (!file) return;

    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);

    img.onload = () => runInference(img);
});

// =============================
// Inferencia
// =============================
async function runInference(imgElement) {

    if (!model) {
        alert("El modelo aún está cargando, intenta de nuevo.");
        return;
    }

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    // Ajustar canvas al tamaño de la imagen
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;

    // Dibujar imagen base
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);

    // Convertir imagen a tensor
    const input = tf.browser.fromPixels(imgElement)
        .resizeBilinear([640, 640])
        .div(255.0)
        .expandDims(0);

    // Ejecutar inferencia
    const output = model.predict(input);

    // YOLO exporta: [boxes, scores, classes, count]
    const [boxes, scores, classes, count] = output;

    const nDet = count.dataSync()[0];
    let detectionsText = "<b>Objetos detectados:</b><br>";

    for (let i = 0; i < nDet; i++) {

        const score = scores.dataSync()[i];
        if (score < 0.4) continue;

        const cls = classes.dataSync()[i];
        const label = labels[cls];

        const [ymin, xmin, ymax, xmax] = boxes.dataSync().slice(i * 4, i * 4 + 4);

        // Convertir a coordenadas reales
        const x = xmin * imgElement.width;
        const y = ymin * imgElement.height;
        const w = (xmax - xmin) * imgElement.width;
        const h = (ymax - ymin) * imgElement.height;

        // Dibujar caja
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillText(`${label} (${score.toFixed(2)})`, x, y - 5);

        detectionsText += `• ${label} (${score.toFixed(2)})<br>`;
    }

    document.getElementById("detections").innerHTML = detectionsText;
}


