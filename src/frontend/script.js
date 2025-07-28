document.addEventListener('DOMContentLoaded', () => {
    const sky = document.querySelector('.sky');
    const starContainers = document.querySelectorAll('.star-container');
    const proximityThreshold = 75;

    // --- OPTIMIZATION 1: Store star positions so we don't recalculate them constantly ---
    let starPositions = [];
    function cacheStarPositions() {
        starPositions = []; // Clear cache before recalculating
        starContainers.forEach(container => {
            const rect = container.getBoundingClientRect();
            starPositions.push({
                label: container.querySelector('.star-label'),
                x: rect.left + (rect.width / 2),
                y: rect.top + (rect.height / 2),
            });
        });
    }
    // Initial calculation on page load
    cacheStarPositions();
    // Recalculate if the window is resized
    window.addEventListener('resize', cacheStarPositions);


    // --- OPTIMIZATION 2: "Throttle" the mousemove event to limit how often it runs ---
    let canRun = true;
    document.addEventListener('mousemove', (e) => {
        if (!canRun) return; // Exit if we're in the cooldown period
        canRun = false;
        setTimeout(() => {
            canRun = true;
        }, 100); // Only run this code every 100ms

        // --- OPTIMIZATION 3: Combined the two old mousemove functions into one ---
        // Parallax effect
        const mouseX = e.clientX / window.innerWidth - 0.5;
        const mouseY = e.clientY / window.innerHeight - 0.5;
        const parallaxFactor = 30;
        const skyX = -mouseX * parallaxFactor;
        const skyY = -mouseY * parallaxFactor;
        sky.style.transform = `translate(${skyX}px, ${skyY}px)`;

        // Proximity detection (now using the cached positions)
        starPositions.forEach(star => {
            if (!star.label) return;
            const distance = Math.sqrt(Math.pow(e.clientX - star.x, 2) + Math.pow(e.clientY - star.y, 2));
            if (distance < proximityThreshold) {
                star.label.style.opacity = 1;
            } else {
                star.label.style.opacity = 0;
            }
        });
    });
});
