let model;
let labels = ["cat", "dog"];
let modelPronto = false;
let video = document.getElementById('webcam');

async function carregarModelo() {
  try {
    model = await tf.loadLayersModel('model/model.json');
    modelPronto = true;
    console.log("Modelo carregado com sucesso!");
    document.getElementById("resultado").innerHTML = "Modelo pronto! Envie uma imagem.";
  } catch (erro) {
    console.error("Erro ao carregar modelo:", erro);
    document.getElementById("resultado").innerHTML = "Erro ao carregar o modelo. Você está usando Live Server?";
  }
}

document.getElementById("upload").addEventListener("change", async (event) => {
  if (!modelPronto) return alert("Modelo ainda está carregando...");

  const file = event.target.files[0];
  if (!file) return;

  const img = document.getElementById("imagem");
  img.src = URL.createObjectURL(file);
  img.style.display = "block";
  video.style.display = "none";

  img.onload = async () => {
    fazerPredicao(img);
  };
});

async function ligarWebcam() {
  if (!modelPronto) return alert("Modelo ainda está carregando...");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = "block";
    document.getElementById("imagem").style.display = "none";
    document.getElementById("btnCapturar").disabled = false;
  } catch (err) {
    alert("Erro ao acessar a webcam: " + err);
  }
}

async function capturarEClassificar() {
  const canvas = document.getElementById('canvasWebcam');
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, 224, 224);
  fazerPredicao(canvas);
}

async function fazerPredicao(elementoImagem) {
  let tensor = tf.browser.fromPixels(elementoImagem)
    .resizeNearestNeighbor([224, 224])
    .toFloat()
    .div(255)
    .expandDims();

  let predicoes = await model.predict(tensor).data();
  tf.dispose(tensor);

  let resultadoHTML = "<h3>Resultado:</h3>";
  predicoes.forEach((p, i) => {
    resultadoHTML += `<p><strong>${labels[i]}</strong>: ${(p * 100).toFixed(2)}%</p>`;
  });

  document.getElementById("resultado").innerHTML = resultadoHTML;
}

async function avaliarModelo() {
  const filesCat = document.getElementById("testImagesCat").files;
  const filesDog = document.getElementById("testImagesDog").files;

  let testSet = [];
  for (let f of filesCat) testSet.push({ file: f, real: "cat" });
  for (let f of filesDog) testSet.push({ file: f, real: "dog" });

  if (testSet.length === 0) {
    alert("Selecione imagens de teste em pelo menos uma das categorias!");
    return;
  }

  document.getElementById("metricas").innerText = "Avaliando " + testSet.length + " imagens. Aguarde...";

  let TP = 0, TN = 0, FP = 0, FN = 0;

  for (let item of testSet) {
    const img = new Image();
    img.src = URL.createObjectURL(item.file);

    await new Promise(resolve => {
      img.onload = async () => {
        let tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .div(255)
          .expandDims();

        let pred = await model.predict(tensor).data();
        tf.dispose(tensor);

        let predIndex = pred.indexOf(Math.max(...pred));
        let predicted = labels[predIndex];
        
        let real = item.real;

        if (real === "cat" && predicted === "cat") TP++;
        if (real === "cat" && predicted === "dog") FN++;
        if (real === "dog" && predicted === "cat") FP++;
        if (real === "dog" && predicted === "dog") TN++;

        resolve();
      };
    });
  }

  let total = TP + TN + FP + FN;

  let acuracia = total === 0 ? 0 : (TP + TN) / total;
  let precisao = (TP + FP) === 0 ? 0 : TP / (TP + FP);
  let recall = (TP + FN) === 0 ? 0 : TP / (TP + FN);
  let f1 = (precisao + recall) === 0 ? 0 : 2 * (precisao * recall) / (precisao + recall);

  document.getElementById("metricas").innerText = `
--- RESULTADOS DA AVALIAÇÃO ---
Total de imagens processadas: ${total}

Métricas:
Acurácia: ${(acuracia * 100).toFixed(2)}%
Precisão: ${(precisao * 100).toFixed(2)}%
Recall: ${(recall * 100).toFixed(2)}%
F1-Score: ${(f1 * 100).toFixed(2)}%

Matriz de Confusão (Cat = Positivo):
Verdadeiros Positivos (TP - cat→cat): ${TP}
Falsos Negativos (FN - cat→dog): ${FN}
Falsos Positivos (FP - dog→cat): ${FP}
Verdadeiros Negativos (TN - dog→dog): ${TN}
  `;
}

carregarModelo();