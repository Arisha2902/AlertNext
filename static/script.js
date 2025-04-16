  
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