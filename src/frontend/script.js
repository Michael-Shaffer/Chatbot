document.addEventListener('DOMContentLoaded', () => {
    const starContainers = document.querySelectorAll('.star-container');
    const proximityThreshold = 75;

    // Store star positions so we don't recalculate them constantly
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


    // This event listener now ONLY handles showing star labels
    document.addEventListener('mousemove', (e) => {
        // Proximity detection for star labels
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
