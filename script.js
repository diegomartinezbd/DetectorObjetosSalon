// =============================
// Cargar modelo TFLite
// =============================
let model;
const TFLITE_PATH = "modelo.tflite";

// Clases del modelo
const labels = ["CPU","Mesa","Mouse","Pantalla","Silla","Teclado"];

// Cargar el modelo
async function loadModel() {
    try {
        model = await tflite.loadTFLiteModel(TFLITE_PATH);
        console.log("Modelo TFLite cargado.");
    } catch (err) {
        console.error("Error cargando el modelo:", err);
    }
}

loadModel();

// =============================
// Manejar carga de imagen
// =============================
document.getElementById("fileInput").addEventListener("change", (event) => {

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

    // Ajustar canvas al tamaño original
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;

    // Dibujar imagen original
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);

    // Preprocesar imagen → tensor
    const input = tf.browser.fromPixels(imgElement)
        .resizeBilinear([640, 640])
        .div(255)
        .expandDims(0);

    // Ejecutar inferencia
    const output = model.predict(input);

    // YOLO exporta: [boxes, scores, classes, count]
    const [boxes, scores, classes, count] = output;

    const nDet = count.dataSync()[0];
    let detectionsText = "<b>Objetos detectados:</b><br>";

    const boxData = boxes.dataSync();
    const scoreData = scores.dataSync();
    const classData = classes.dataSync();

    for (let i = 0; i < nDet; i++) {

        const score = scoreData[i];
        if (score < 0.4) continue;

        const cls = classData[i];
        const label = labels[cls];

        const ymin = boxData[i * 4];
        const xmin = boxData[i * 4 + 1];
        const ymax = boxData[i * 4 + 2];
        const xmax = boxData[i * 4 + 3];

        // Coordenadas reales en la imagen original
        const x = xmin * imgElement.width;
        const y = ymin * imgElement.height;
        const w = (xmax - xmin) * imgElement.width;
        const h = (ymax - ymin) * imgElement.height;

        // Dibujar bounding box
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Dibujar etiqueta
        ctx.fillStyle = "red";
        ctx.font = "16px Arial";
        ctx.fillText(`${label} (${score.toFixed(2)})`, x, y - 5);

        detectionsText += `• ${label} (${score.toFixed(2)})<br>`;
    }

    document.getElementById("detections").innerHTML = detectionsText;
}
