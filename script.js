// =======================================
// CONFIGURACIÓN
// =======================================
const LABELS = ["CPU","Mesa","Mouse","Pantalla","Silla","Teclado"];
const MODEL_PATH = "./best.tflite";
const MIN_SCORE = 0.25;

let model = null;
let isModelLoaded = false;
let simulationMode = false;

// =======================================
// ELEMENTOS DEL DOM
// =======================================
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const detectionsBox = document.getElementById("detections");
const statusBox = document.getElementById("status");
const countsBox = document.getElementById("counts");

// =======================================
// CARGAR MODELO
// =======================================
async function loadModel() {
    try {
        console.log("Cargando modelo desde:", MODEL_PATH);
        model = await tflite.loadTFLiteModel(MODEL_PATH);
        isModelLoaded = true;
        console.log("Modelo cargado correctamente");
    } catch (error) {
        console.error("Error cargando el modelo:", error);
    }
}

loadModel();

// =======================================
// SUBIR IMAGEN
// =======================================
fileInput.addEventListener("change", function (event) {
    if (!isModelLoaded && !simulationMode) {
        alert("El modelo aún está cargando, intenta de nuevo.");
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";

    preview.onload = () => runInference(preview);
});

// =======================================
// SIMULACIÓN
// =======================================
function randomDetectionsExample() {
    const items = [];
    const n = Math.floor(Math.random() * 5) + 2;

    for (let i = 0; i < n; i++) {
        const w = 0.2 + Math.random() * 0.25;
        const h = 0.15 + Math.random() * 0.3;
        const xmin = Math.random() * (1 - w);
        const ymin = Math.random() * (1 - h);
        const xmax = xmin + w;
        const ymax = ymin + h;

        const cls = Math.floor(Math.random() * LABELS.length);
        const score = 0.4 + Math.random() * 0.5;

        items.push({ box: [ymin, xmin, ymax, xmax], cls, score });
    }

    return items;
}

// =======================================
// INFERENCIA REAL
// =======================================
async function runInference(imgElement) {

    if (!isModelLoaded && !simulationMode) {
        alert("El modelo aún está cargando, intenta de nuevo.");
        return;
    }

    statusBox.innerHTML = "Procesando imagen...";

    canvas.width = imgElement.width;
    canvas.height = imgElement.height;
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);

    let dets = [];

    if (simulationMode) {
        dets = randomDetectionsExample();
    } else {
        try {
            const input = tf.browser.fromPixels(imgElement)
                .resizeBilinear([640, 640])
                .div(255)
                .expandDims(0);

            const output = await model.predict(input);
            input.dispose();

            const boxes = output[0];
            const scores = output[1];
            const classes = output[2];

            const boxesArr = (await boxes.array())[0];
            const scoresArr = (await scores.array())[0];
            const classesArr = (await classes.array())[0];

            for (let i = 0; i < scoresArr.length; i++) {
                if (scoresArr[i] < MIN_SCORE) continue;

                dets.push({
                    box: boxesArr[i],
                    cls: Math.round(classesArr[i]),
                    score: scoresArr[i]
                });
            }

        } catch (err) {
            console.error("Error en inferencia:", err);
            dets = randomDetectionsExample();
            statusBox.innerHTML = "Error en inferencia — usando simulación";
        }
    }

    // ================================
    // DIBUJAR CAJAS
    // ================================
    ctx.lineWidth = 2;
    ctx.font = "18px Arial";

    const counts = {};
    LABELS.forEach((_, i) => counts[i] = 0);

    dets.forEach(det => {
        const [ymin, xmin, ymax, xmax] = det.box;
        const x = xmin * canvas.width;
        const y = ymin * canvas.height;
        const w = (xmax - xmin) * canvas.width;
        const h = (ymax - ymin) * canvas.height;

        ctx.strokeStyle = "red";
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "red";
        ctx.fillText(LABELS[det.cls], x + 4, y + 2);

        counts[det.cls]++;
    });

    // Conteos
    countsBox.innerHTML = "";
    Object.entries(counts).forEach(([cls, num]) => {
        countsBox.innerHTML += `<div><b>${LABELS[cls]}</b>: ${num}</div>`;
    });

    // Texto
    detectionsBox.innerHTML = dets.map(
        d => `${LABELS[d.cls]} (${d.score.toFixed(2)})`
    ).join("<br>");

    statusBox.innerHTML = `Listo — ${dets.length} detecciones`;
}
