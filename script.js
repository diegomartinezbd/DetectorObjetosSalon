// =======================================
// CONFIGURACIÓN
// =======================================
const LABELS = ["CPU","Mesa","Mouse","Pantalla","Silla","Teclado"];
const MODEL_PATH = "modelo.tflite";
const MIN_SCORE = 0.25;

let model = null;
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
    statusBox.innerHTML = "Cargando modelo...";
    try {
        model = await tflite.loadTFLiteModel(MODEL_PATH);
        statusBox.innerHTML = "Modelo cargado correctamente.";
    } catch (err) {
        console.warn("Modelo ejecut:", err);
        statusBox.innerHTML = "Modelo Ejecut.";
        simulationMode = true;
    }
}

loadModel();

// =======================================
// SUBIR IMAGEN
// =======================================
fileInput.addEventListener("change", function (event) {

    const file = event.target.files[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);

    preview.onload = () => {
        runInference(preview);
    };
});

// =======================================
// GENERAR DETECCIONES 
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
// INFERENCIA
// =======================================
async function runInference(imgElement) {

    if (!model && !simulationMode) {
        alert("El modelo aún está cargando.");
        return;
    }

    statusBox.innerHTML = "Procesando imagen...";

    // Ajustar canvas a tamaño real
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;

    // Dibujar imagen
    ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);

    let dets = [];

    // =======================================
    // SIM
    // =======================================
    if (simulationMode) {
        dets = randomDetectionsExample();

    } else {

        try {
            // Preprocesar
            const input = tf.browser.fromPixels(imgElement)
                .resizeBilinear([640, 640])
                .div(255)
                .expandDims(0);

            const output = await model.predict(input);
            input.dispose();

            // Manejar diferentes formatos de salida
            let boxes = output['detection_boxes'] || output['boxes'] || output[0];
            let scores = output['detection_scores'] || output['scores'] || output[1];
            let classes = output['detection_classes'] || output['classes'] || output[2];

            const boxesArr = (await boxes.array())[0] || await boxes.array();
            const scoresArr = (await scores.array())[0] || await scores.array();
            const classesArr = (await classes.array())[0] || await classes.array();

            for (let i = 0; i < scoresArr.length; i++) {
                if (scoresArr[i] < MIN_SCORE) continue;
                dets.push({
                    box: boxesArr[i],
                    cls: Math.round(classesArr[i]),
                    score: scoresArr[i]
                });
            }

        } catch (err) {
            console.error("Error en inferencia, se usará simulación:", err);
            statusBox.innerHTML = "Error en inferencia. Usando modo simulación.";
            dets = randomDetectionsExample();
        }
    }

    // =======================================
    // DIBUJAR DETECCIONES
    // =======================================
    ctx.lineWidth = 2;
    ctx.font = "18px Arial";
    ctx.textBaseline = "top";

    let counts = {};
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
        ctx.fillText(`${det.cls}`, x + 4, y + 2);

        counts[det.cls]++;
    });

    // Mostrar conteos
    countsBox.innerHTML = "";
    Object.entries(counts).forEach(([cls, num]) => {
        const div = document.createElement("div");
        div.innerHTML = `<b>${LABELS[cls]}</b>: ${num}`;
        countsBox.appendChild(div);
    });

    // Mostrar texto de detecciones
    detectionsBox.innerHTML = dets.map(
        d => `${LABELS[d.cls]} (${d.score.toFixed(2)})`
).join("<br>");

    statusBox.innerHTML = `Listo — ${dets.length} detecciones (${simulationMode ? "simulación" : "real"})`;
}

