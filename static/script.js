  
  // to validate phone no
  const phoneInput = document.getElementById('phone');

  // Pre-fill with '+91'
  phoneInput.value = '+91';

  phoneInput.addEventListener('input', (event) => {
    // Ensure the input starts with '+91'
    if (!phoneInput.value.startsWith('+91')) {
      phoneInput.value = '+91';
    }

    // Limit the length to '+91' followed by 10 digits (total 13 characters)
    if (phoneInput.value.length > 13) {
      phoneInput.value = phoneInput.value.slice(0, 13);
    }
  });

  

  function showPrompt(event) {
    event.preventDefault(); // Prevents the form from submitting normally
    alert("Thank you for submitting your bin code. We will soon show you the disaster status.");
  }

  async function getPrediction() {
    const pincode = document.getElementById('pincode').value;
    if (!pincode || pincode.length !== 6) {
        alert('Please enter a valid 6-digit pincode');
        return;
    }

    try {
        const response = await fetch('/predict_by_pincode', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pincode: parseInt(pincode) })
        });
        const data = await response.json();
        displayResult(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = '<div class="error">Error fetching prediction. Please try again.</div>';
    }
}

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';

    if (data.error) {
        resultDiv.innerHTML = `<div class="error">${data.error}</div>`;
        return;
    }

    let html = `
        <h2>Location: ${data.location}</h2>
        <p>Coordinates: Latitude ${data.coordinates.latitude}, Longitude ${data.coordinates.longitude}</p>
    `;

    if (data.message) {
        html += `<p>${data.message}</p>`;
    } else if (data.predictions && data.predictions.length > 0) {
        html += '<h3>Disaster Risk Predictions:</h3>';
        data.predictions.forEach(pred => {
            html += `
                <div class="prediction">
                    <p>Location (approx): (${pred.location.latitude}, ${pred.location.longitude})</p>
                    <p>Flood Risk: ${pred.flood_risk}</p>
                    <p>Cyclone Risk: ${pred.cyclone_risk}</p>
                    <p>Earthquake Risk: ${pred.earthquake_risk}</p>
                    <p>Combined Risk (Model): ${pred.combined_risk}</p>
                </div>
            `;
        });
    }

    resultDiv.innerHTML = html;
}