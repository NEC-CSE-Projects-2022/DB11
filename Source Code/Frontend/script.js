document.getElementById("uploadBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("imageUpload");
    const previewDiv = document.getElementById("preview");
    const resultDiv = document.getElementById("result");

    if (!fileInput.files.length) {
        alert("Please upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    // Show image preview
    const reader = new FileReader();
    reader.onload = e => {
        previewDiv.innerHTML = `<img src="${e.target.result}" alt="Preview Image"/>`;
    };
    reader.readAsDataURL(fileInput.files[0]);

    resultDiv.innerHTML = "Analyzing image... ⏳";

    try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<span style='color: red;'>${data.error}</span>`;
        } else {
            const color = data.prediction === "Forged Image" ? "#ff4d4d" : "#00ff7f";
            resultDiv.innerHTML = `
                <p style="color:${color};">
                    <b>${data.prediction}</b><br>
                    Confidence: ${(data.confidence * 100).toFixed(2)}%
                </p>`;
        }
    } catch (err) {
        console.error(err);
        resultDiv.innerHTML = "<span style='color: red;'>Error processing image</span>";
    }
});