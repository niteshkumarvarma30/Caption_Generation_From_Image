document.getElementById('generateBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const captionText = document.getElementById('captionText');
    const heatmapImg = document.getElementById('heatmapImg');

    if (fileInput.files.length === 0) {
        alert("Please upload an image first!");
        return;
    }

    // Hide results, show loading
    resultsDiv.classList.add('hidden');
    loadingDiv.classList.remove('hidden');

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            // Update UI with AI Data
            captionText.innerText = `"${data.caption}"`;
            heatmapImg.src = "data:image/png;base64," + data.heatmap_image;
            
            // Hide loading, show results
            loadingDiv.classList.add('hidden');
            resultsDiv.classList.remove('hidden');
        } else {
            alert("Error: " + data.error);
            loadingDiv.classList.add('hidden');
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong with the server.");
        loadingDiv.classList.add('hidden');
    }
});