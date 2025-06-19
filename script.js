// Función para mostrar una sección
function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.style.display = 'none';
    });
    document.getElementById(sectionId).style.display = 'block';

    // Actualizar enlace activo
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector(`a[href="#${sectionId}"]`).classList.add('active');

    // Verificar autenticación para investigación
    if (sectionId === 'investigacion') {
        checkAuthForResearch();
    }
}

// Verificar estado de autenticación
function checkAuth() {
    const email = localStorage.getItem('email');
    const role = localStorage.getItem('role');
    console.log('CheckAuth:', { email, role }); // Depuración
    if (email && role) {
        document.getElementById('auth-nav').style.display = 'none';
        document.getElementById('logout-nav').style.display = 'block';
        return { email, role };
    } else {
        document.getElementById('auth-nav').style.display = 'block';
        document.getElementById('logout-nav').style.display = 'none';
        return null;
    }
}

// Verificar autenticación y mostrar contenido según rol
function checkAuthForResearch() {
    const auth = checkAuth();
    if (auth) {
        console.log('Autenticación para investigación:', auth); // Depuración
        document.getElementById('auth-required').style.display = 'none';
        document.getElementById('investigacion-authenticated').style.display = 'block';
        document.getElementById('username-display').textContent = auth.email.split('@')[0];

        // Reiniciar visibilidad
        document.getElementById('admin-panel').style.display = 'none';
        document.getElementById('cubesat-director').style.display = 'none';
        document.getElementById('cubesat-miembro').style.display = 'none';
        document.getElementById('cubesat-publico').style.display = 'none';

        // Mostrar contenido según rol
        if (auth.role === 'Director') {
            console.log('Mostrando contenido para Director'); // Depuración
            document.getElementById('admin-panel').style.display = 'block';
            document.getElementById('cubesat-director').style.display = 'block';
        } else if (auth.role === 'Miembro') {
            console.log('Mostrando contenido para Miembro'); // Depuración
            document.getElementById('cubesat-miembro').style.display = 'block';
        } else if (auth.role === 'Público General') {
            console.log('Mostrando contenido para Público General'); // Depuración
            document.getElementById('cubesat-publico').style.display = 'block';
        } else {
            console.error('Rol no reconocido:', auth.role); // Depuración
        }
    } else {
        document.getElementById('auth-required').style.display = 'block';
        document.getElementById('investigacion-authenticated').style.display = 'none';
    }
}

// Cargar usuarios desde users.json o localStorage
async function loadUsers() {
    let users = [];
    try {
        const response = await fetch('users.json');
        if (response.ok) {
            const data = await response.json();
            users = data.users;
        }
    } catch (error) {
        console.error('Error al cargar users.json:', error);
        users = [
            {
                email: 'director@laccei.org',
                name: 'Director LACCEI',
                password: 'LACCEI2025',
                role: 'Director'
            }
        ];
    }
    // Cargar usuarios adicionales desde localStorage
    const storedUsers = localStorage.getItem('users');
    if (storedUsers) {
        users = users.concat(JSON.parse(storedUsers));
    }
    console.log('Usuarios cargados:', users); // Depuración
    return { users };
}

// Guardar usuarios en localStorage
function saveUsers(users) {
    try {
        localStorage.setItem('users', JSON.stringify(users.users.filter(u => u.email !== 'director@laccei.org')));
    } catch (error) {
        console.error('Error al guardar usuarios:', error);
    }
}

// Validar correo electrónico
function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// Manejar login
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('login-email').value.trim().toLowerCase();
    const password = document.getElementById('login-password').value;
    const errorDiv = document.getElementById('login-error');

    if (!isValidEmail(email)) {
        errorDiv.textContent = 'Por favor, ingresa un correo electrónico válido.';
        errorDiv.style.display = 'block';
        return;
    }

    const data = await loadUsers();
    const user = data.users.find(u => u.email === email && u.password === password);

    if (user) {
        console.log('Usuario encontrado:', user); // Depuración
        localStorage.setItem('email', email);
        localStorage.setItem('role', user.role);
        errorDiv.style.display = 'none';
        bootstrap.Modal.getInstance(document.getElementById('authModal')).hide();
        checkAuth();
        showSection('inicio');
    } else {
        errorDiv.textContent = 'Correo o contraseña incorrectos.';
        errorDiv.style.display = 'block';
    }
});

// Manejar registro
document.getElementById('register-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = document.getElementById('register-email').value.trim().toLowerCase();
    const name = document.getElementById('register-name').value.trim();
    const password = document.getElementById('register-password').value;
    const role = document.getElementById('register-role').value;
    const errorDiv = document.getElementById('register-error');
    const successDiv = document.getElementById('register-success');

    if (!isValidEmail(email)) {
        errorDiv.textContent = 'Por favor, ingresa un correo electrónico válido.';
        errorDiv.style.display = 'block';
        successDiv.style.display = 'none';
        return;
    }

    if (name.length < 2) {
        errorDiv.textContent = 'El nombre debe tener al menos 2 caracteres.';
        errorDiv.style.display = 'block';
        successDiv.style.display = 'none';
        return;
    }

    if (password.length < 6) {
        errorDiv.textContent = 'La contraseña debe tener al menos 6 caracteres.';
        errorDiv.style.display = 'block';
        successDiv.style.display = 'none';
        return;
    }

    if (role === 'Director') {
        errorDiv.textContent = 'No se permite registrar usuarios como Director.';
        errorDiv.style.display = 'block';
        successDiv.style.display = 'none';
        return;
    }

    const data = await loadUsers();
    if (data.users.find(u => u.email === email)) {
        errorDiv.textContent = 'El correo ya está registrado.';
        errorDiv.style.display = 'block';
        successDiv.style.display = 'none';
        return;
    }

    data.users.push({ email, name, password, role });
    saveUsers(data);
    errorDiv.style.display = 'none';
    successDiv.textContent = 'Registro exitoso. Ahora puedes iniciar sesión.';
    successDiv.style.display = 'block';
    document.getElementById('register-form').reset();
});

// Manejar logout
function logout() {
    localStorage.removeItem('email');
    localStorage.removeItem('role');
    checkAuth();
    showSection('inicio');
}

// Inicializar página
document.addEventListener('DOMContentLoaded', () => {
    showSection('inicio');
    checkAuth();
});