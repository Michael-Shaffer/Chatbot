document.addEventListener('DOMContentLoaded', () => {
    const sky = document.querySelector('.sky');
    const starContainers = document.querySelectorAll('.star-container');
    const proximityThreshold = 75;

    // --- CRITIQUE ADDRESSED: Adds a parallax effect to make the background immersive ---
    document.addEventListener('mousemove', (e) => {
        // Calculate mouse position from center (-0.5 to 0.5)
        const mouseX = e.clientX / window.innerWidth - 0.5;
        const mouseY = e.clientY / window.innerHeight - 0.5;
        
        // Move the sky opposite to the mouse direction. The multiplier determines the depth.
        const parallaxFactor = 30;
        const skyX = -mouseX * parallaxFactor;
        const skyY = -mouseY * parallaxFactor;
        
        sky.style.transform = `translate(${skyX}px, ${skyY}px)`;
    });


    // --- Proximity detection for star labels (same logic as before) ---
    document.addEventListener('mousemove', (e) => {
        const mouseX = e.clientX;
        const mouseY = e.clientY;

        starContainers.forEach(container => {
            const label = container.querySelector('.star-label');
            if (!label) return;

            const rect = container.getBoundingClientRect();
            const starX = rect.left + (rect.width / 2);
            const starY = rect.top + (rect.height / 2);

            const distance = Math.sqrt(Math.pow(mouseX - starX, 2) + Math.pow(mouseY - starY, 2));

            if (distance < proximityThreshold) {
                label.style.opacity = 1;
            } else {
                label.style.opacity = 0;
            }
        });
    });
});
