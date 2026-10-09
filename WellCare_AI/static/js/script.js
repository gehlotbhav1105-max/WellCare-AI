/**
 * WellCare AI - Frontend Interactive Scripts
 * ==========================================
 * Handles UI interactions, live input synchronization, client-side validation,
 * BMI calculators, and interactive alerts.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Mobile Navbar Toggle
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('nav-open');
        });
    }

    // 2. BMI Live Calculator (on assessment page)
    const heightInput = document.getElementById('Height');
    const weightInput = document.getElementById('Weight');
    if (heightInput && weightInput) {
        const updateBMI = () => {
            const h = parseFloat(heightInput.value);
            const w = parseFloat(weightInput.value);
            if (h > 50 && w > 20) {
                const bmi = (w / ((h / 100) ** 2)).toFixed(1);
                let cat = 'Normal';
                if (bmi < 18.5) cat = 'Underweight';
                else if (bmi >= 30.0) cat = 'Obese';
                else if (bmi >= 25.0) cat = 'Overweight';
                
                // Show hint if hint element exists
                let hint = document.getElementById('bmiLiveHint');
                if (!hint) {
                    hint = document.createElement('div');
                    hint.id = 'bmiLiveHint';
                    hint.className = 'form-hint text-teal mt-1 font-weight-bold';
                    weightInput.parentElement.appendChild(hint);
                }
                hint.innerHTML = `<i class="fa-solid fa-calculator"></i> Estimated BMI: <strong>${bmi}</strong> (${cat})`;
            }
        };
        heightInput.addEventListener('input', updateBMI);
        weightInput.addEventListener('input', updateBMI);
        updateBMI();
    }

    // 3. Form Submission Progress Feedback
    const assessmentForm = document.getElementById('assessmentForm');
    const submitBtn = document.getElementById('submitBtn');
    if (assessmentForm && submitBtn) {
        assessmentForm.addEventListener('submit', () => {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Assessment & Running AI Inference...';
        });
    }
});
