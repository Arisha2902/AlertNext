  
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

  new bootstrap.Collapse(document.getElementById('navbarTogglerDemo02')).toggle();

  document.querySelector('.navbar-toggler').addEventListener('click', function () {
    document.querySelector('#navbarTogglerDemo02').classList.toggle('show');
  });


