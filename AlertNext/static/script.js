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

  

  // function showPrompt(event) {
  //   event.preventDefault(); // Prevents the form from submitting normally
  //   alert("Thank you for submitting your bin code. We will soon show you the disaster status.");
  // }
  async function getPrediction() {
    const pincode = document.getElementById('prediction-pincode').value;
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
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get prediction');
        }
        displayResult(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `
            <div class="error">
                Error: ${error.message || 'Failed to fetch prediction. Please try again.'}
            </div>
        `;
    }
}

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '';

    if (data.error) {
        resultDiv.innerHTML = `<div class="error">${data.error}</div>`;
        return;
    }

    const location = data.location;
    const predictions = data.predictions;

    let html = `
        <div class="result-container">
            <h2>📍 Location Details</h2>
            <p>Pincode: ${location.pincode}</p>
            <p>Coordinates: (${location.latitude.toFixed(6)}, ${location.longitude.toFixed(6)})</p>
            
            <h2>🌊 Flood Risk</h2>
            <p>Rainfall: ${predictions.flood.rainfall.toFixed(2)} mm</p>
            <p>Water Level: ${predictions.flood.water_level.toFixed(2)} m</p>
            <p>Humidity: ${predictions.flood.humidity.toFixed(2)}%</p>
            <p>Nearest Data Point: ${predictions.flood.distance.toFixed(2)} km away</p>
            
            <h2>🌍 Earthquake Risk</h2>
            <p>Magnitude: ${predictions.earthquake.magnitude.toFixed(2)}</p>
            <p>Nearest Data Point: ${predictions.earthquake.distance.toFixed(2)} km away</p>
            
            <h2>🌪 Cyclone Risk</h2>
            <p>Wind Speed: ${predictions.cyclone.wind_speed.toFixed(2)} km/h</p>
            <p>Pressure: ${predictions.cyclone.pressure.toFixed(2)} mb</p>
            <p>Nearest Data Point: ${predictions.cyclone.distance.toFixed(2)} km away</p>
        </div>
    `;

    resultDiv.innerHTML = html;
}

////////////////////////////////////////////////////////////////////////////////

// Map variables
let map;
let markers = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Event listener for the search button
    document.getElementById('search-btn').addEventListener('click', searchNGOs);
    
    // Event listener for Enter key on PIN code input
    document.getElementById('ngo_pincode').addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
            searchNGOs();
        }
    });
    
    // Initialize the map (hidden initially)
    initializeMap();
});

// Initialize Leaflet map
function initializeMap() {
    // Create a map centered on India
    map = L.map('map').setView([20.5937, 78.9629], 5);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Force map to refresh when container becomes visible
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.target.classList.contains('d-none') === false) {
                map.invalidateSize();
            }
        });
    });
    
    observer.observe(document.getElementById('results-container'), {
        attributes: true,
        attributeFilter: ['class']
    });
}

// Search for NGOs by PIN code
async function searchNGOs() {
    const pincodeInput = document.getElementById('ngo_pincode');
    const pincode = pincodeInput.value.trim();
    const errorMessage = document.getElementById('error-message');
    
    // Reset error message
    errorMessage.style.display = 'none';
    errorMessage.textContent = '';
    
    // Validate PIN code (6 digits)
    if (!pincode.match(/^\d{6}$/)) {
        errorMessage.textContent = 'Please enter a valid 6-digit PIN code';
        errorMessage.style.display = 'block';
        return;
    }
    
    try {
        // Show loading state
        document.getElementById('search-btn').innerHTML = '<span class="spinner-border" role="status" aria-hidden="true"></span> Searching...';
        document.getElementById('search-btn').disabled = true;
        
        // Fetch NGOs from the API
        const response = await fetch('/api/ngos/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ pincode: pincode }),
        });
        
        const data = await response.json();
        
        // Reset button state
        document.getElementById('search-btn').innerHTML = 'NGO Nearby';
        document.getElementById('search-btn').disabled = false;
        
        // Show results container
        document.getElementById('results-container').classList.remove('d-none');
        
        if (!response.ok) {
            // Display error message
            document.getElementById('ngo-list').innerHTML = '';
            document.getElementById('no-results').classList.remove('d-none');
            clearMarkers();
            return;
        }
        
        if (data.ngos && data.ngos.length > 0) {
            // Display NGOs on the map and in the list
            displayNGOs(data.ngos);
            document.getElementById('no-results').classList.add('d-none');
        } else {
            // No NGOs found
            document.getElementById('ngo-list').innerHTML = '';
            document.getElementById('no-results').classList.remove('d-none');
            clearMarkers();
        }
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('search-btn').innerHTML = 'NGO Nearby';
        document.getElementById('search-btn').disabled = false;
        errorMessage.textContent = 'An error occurred. Please try again.';
        errorMessage.style.display = 'block';
    }
}

// Display NGOs on the map and in the list
function displayNGOs(ngos) {
    // Clear existing markers and list
    clearMarkers();
    const ngoListContainer = document.getElementById('ngo-list');
    ngoListContainer.innerHTML = '';
    
    // Create bounds object to fit all markers
    const bounds = L.latLngBounds();
    
    // Add each NGO to the map and list
    ngos.forEach((ngo, index) => {
        // Add marker to the map
        const marker = L.marker(ngo.coordinates).addTo(map);
        marker.bindPopup(`<strong>${ngo.name}</strong><br>${ngo.address}`);
        markers.push(marker);
        
        // Extend map bounds to include this marker
        bounds.extend(ngo.coordinates);
        
        // Create NGO card in the list
        const ngoCard = document.createElement('div');
        ngoCard.className = 'ngo-card';
        ngoCard.innerHTML = `
            <h5>${ngo.name}</h5>
            <p><strong>Description:</strong> ${ngo.description}</p>
            <p><strong>Address:</strong> ${ngo.address}</p>
            <p><strong>Contact:</strong> ${ngo.contact}</p>
            <button class="btn btn-sm btn-outline-secondary view-on-map" data-index="${index}">View on Map</button>
        `;
        ngoListContainer.appendChild(ngoCard);
        
        // Add event listener to the "View on Map" button
        ngoCard.querySelector('.view-on-map').addEventListener('click', function() {
            const idx = this.getAttribute('data-index');
            focusOnNGO(idx);
        });
    });
    
    // Fit the map to show all markers
    if (markers.length > 0) {
        map.fitBounds(bounds, { padding: [50, 50] });
    }
    
    // Force map to refresh
    map.invalidateSize();
}

// Focus on a specific NGO on the map
function focusOnNGO(index) {
    const marker = markers[index];
    map.setView(marker.getLatLng(), 14);
    marker.openPopup();
}

// Clear all markers from the map
function clearMarkers() {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
}
